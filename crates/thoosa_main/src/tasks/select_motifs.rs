use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use async_trait::async_trait;
use futures_util::{StreamExt, TryStreamExt};
use log::{debug, error, info, trace};
use parking_lot::RwLock;
use tch::Kind::Float;
use tch::Tensor;
use caching_runner::common::cachable::{Cached, DevicesWrap};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage};
use caching_runner::tmp_cache::TmpCached;
use thoosa_common::applicables::ThoDevice;
use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig, SelectMotifsRunConfig};
use thoosa_common::runs::Labeled;
use thoosa_common::utils::pad_to;
use thoosa_common::ws::Workspace;
use thoosa_motifs::find::{FixedFindMotifsConfig, SelectMotifsSavableResult};
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{DataId, SimpleId};
use thoosa_store::Store;
use torch_runner::data::ThoTensor2;
use crate::cli::Commands::SelectMotifs;
use crate::tasks::cross_batch_clc::{unfold_and_cat, CrossInput, UnfoldParams};
use crate::tasks::cross_dims::{ChannelsData2, ChannelsMotifSlices, MotifSlices};
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::motif_loader::{MotifLoader, MotifMeta};
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::RawTensorLoader;

#[derive(Debug)]
pub struct SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + 'static
{

    store: Arc<RwLock<Box<dyn Store>>>,
    ws:Arc<RwLock<Box<dyn Workspace>>>,
    loader:Arc<L>,
    motif_loader:Arc<ML>,
    pd:PhantomData<C>,
    run_configs:Vec<SelectMotifsRunConfig>,
    find_motifs_run_ids:Vec<RunId>,
}

impl <L, ML, C> SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + 'static
{

    pub fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        loader:Arc<L>,
        motif_loader:Arc<ML>,
        run_configs:Vec<SelectMotifsRunConfig>,
        find_motifs_run_ids:Vec<RunId>,
    ) -> Self {
        Self {
            store: Arc::clone(store),
            ws: Arc::clone(ws),
            loader,
            motif_loader,
            pd: Default::default(),
            run_configs,
            find_motifs_run_ids
        }
    }

}


impl <L,ML, C> Stage  for  SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + 'static

{
    type Ctx = C;
    type In<'a> = ResourcesConfigInput<Labeled>;
    type Out = SelectMotifsSavableResult;
}

#[async_trait]
impl <L, ML, C> AsyncStage  for  SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + 'static
{
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
        let (device_idx, cache_area) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area())
        };

        let labels_resources = input.with_data_ref(0, None, |input|{
            Ok(input
                .resources_config()
                .to_resources(&self.store)?
                .into_iter()
                .map(|r| Arc::new(r))
                .collect::<Vec<_>>())
        }).await?;
        let max_batch = 1000;//TODO

        debug!("SelectMotifsStage::run: resources={labels_resources:?}");



        let tmp_cache:TmpCached<DataWithExtra<ThoTensor2<ChannelsMotifSlices, SimpleId>, (Vec<Vec<DataId>>, FindMotifsRunConfig)>> = TmpCached::new("motif slices", cache_area).await;
        let num_channels = self.ws.read().channels();

        for find_motifs_run_id in &self.find_motifs_run_ids {

            let find_motifs_run_config = FindMotifsRunConfig::from_properties_map__(
                *find_motifs_run_id,
                &self.store.read().list_run_config_values(*find_motifs_run_id)?,
            )?;

            for select_motifs_run_config in self.run_configs.iter() {
                let motif_ids = self.store.read().list_motifs(*find_motifs_run_id)?;
                let motifs_len = motif_ids.len();
                let done_cnt = Arc::new(AtomicU32::new(0));

                let loaded_motifs = futures::stream::iter(motif_ids)
                    .map( |motif_id| {
                        let done_cnt = Arc::clone(&done_cnt);
                        let motif_loader_impl = Arc::clone(&self.motif_loader);
                        let ctx = Arc::clone(&ctx);
                        async move {
                            let motif = motif_loader_impl.run_cached(Arc::new(DevicesWrap::wrap_cpu((motif_id as u32).into())), ctx).await?;
                            debug!("SelectMotifsStage::run: find_motifs_run_id={find_motifs_run_id} run_config.id={:?} motif={motif:?}", select_motifs_run_config.id);

                            Ok::<_, ThoosaError>(motif)
                        }
                    })
                    .buffer_unordered(10)
                    .try_collect::<Vec<_>>()
                    .await?;

                debug!("SelectMotifsStage::run: loaded_motifs={loaded_motifs:?}");

                DevicesWrap::with_data_array(&loaded_motifs, device_idx, None, |data|{
                    let channels_motifs = data.into_iter().fold(

                        (0..num_channels).map(|_|vec![]).collect::<Vec<_>>(),//vec for each channel

                        |mut acc, item| {

                            assert_eq!(
                                find_motifs_run_config.window, item.extra().window,
                                "window is not same: find_motifs_run_config.window={} item.extra().window={}item.extra().window, motif_id={:?}", find_motifs_run_config.window, item.extra().window, item.data().id()
                            );

                            acc[item.extra().channel as usize].push(item.data());
                        acc
                    });

                    let max_motif_slices = channels_motifs
                        .iter()
                        .map(|v|
                            v.iter()
                            .map(|t|t.size()[0] as usize)//num motif slices known for each motif
                            .sum::<usize>())
                        .max().unwrap_or(0);

                    // let empty_slice = Tensor::zeros(&[find_motifs_run_config.window as i64], (Float, ThoDevice::from(device_idx).into()));

                    // let dummy_slice
                    let mut channels_motif_ids = vec![Vec::<DataId>::new(); num_channels as usize];
                    let channels_motifs_tensors_vec = (0..num_channels).map(|c|{

                        let channel_motifs = &channels_motifs.get(c as usize).expect(&format!("not enough channels: len={} c={}", channels_motifs.len(), c));

                        debug!("SelectMotifsStage::run: channel {c} max_motif_slices={max_motif_slices} channel_motifs={channel_motifs:?}");

                        let channel_motifs_tensors = channel_motifs.iter().map(|motif|  {
                            debug!("SelectMotifsStage::run: motif={motif:?} id={}", motif.id());
                            for _ in 0..motif.size()[0] {
                                channels_motif_ids[c as usize].push(motif.id().clone());
                            }

                            motif.t_ref().copy()
                        }).collect::<Vec<_>>();

                        debug!("SelectMotifsStage::run: c={c}, channel_motifs_tensors={:?}", channel_motifs_tensors);

                        if channel_motifs_tensors.is_empty() {
                            Tensor::zeros(&[max_motif_slices as i64, find_motifs_run_config.window  as i64], (Float, ThoDevice::from(device_idx).into()))

                        } else {
                            let channel_t = Tensor::cat(&channel_motifs_tensors, 0);

                            if channel_t.size()[0] < max_motif_slices as i64 {
                                debug!("SelectMotifsStage::run: c={c}, channel_t.size={:?} padding to {:?}", channel_t.size(), [max_motif_slices as i64, find_motifs_run_config.window  as i64]);
                                pad_to(&channel_t, &[max_motif_slices as i64, find_motifs_run_config.window  as i64]).expect("pad to failed")

                            } else {
                                debug!("SelectMotifsStage::run: c={c}, channel_t.size={:?}", channel_t.size());

                                channel_t
                            }
                        }

                    }).collect::<Vec<_>>();

                    debug!("SelectMotifsStage::run: channels_motifs_tensors_vec={channels_motifs_tensors_vec:?}");

                    let channels_motifs_t = Tensor::stack(&channels_motifs_tensors_vec, 0);

                    debug!("SelectMotifsStage::run: num_channels={num_channels}, channels_motif_ids={channels_motif_ids:?} channels_motifs_t.size={:?}", channels_motifs_t.size());
                    trace!("SelectMotifsStage::run: channels_motifs_t=\n{channels_motifs_t}");

                    tmp_cache.insert(
                        DataWithExtra::new(
                            ThoTensor2::new(
                                SimpleId(format!("{}_{}", find_motifs_run_config.id.expect("no id in find_motifs_run_config"), select_motifs_run_config.id.expect("no id in select_motifs_run_config")).leak()),
                                channels_motifs_t
                            ),
                            (channels_motif_ids, find_motifs_run_config.clone())
                        )//TODO leak
                    );

                    Ok(())
                }).await?;
            }
        }


        let motifs = tmp_cache.clone_to_arcs_vec();



        for label_resources in labels_resources {
            let resources = &label_resources.1;

            let mut t_0_batch: Vec<_> = vec![];
            let mut t_0_batch_resources: Vec<_> = vec![];
            let mut t_0_batch_len = 0;

            let mut skip_0_i = 0;

            for (i_0, resource_0) in resources.iter().enumerate() {
                info!("on i_0={i_0} resource_0={resource_0:?}");

                let t_0 = Arc::clone(&self.loader)
                    .run_cached(Arc::new(DevicesWrap::wrap_cpu(resource_0.clone())), Arc::clone(&ctx))
                    .await?;

                t_0_batch_len += t_0.size().dim_1();
                t_0_batch.push(t_0);
                t_0_batch_resources.push(resource_0);

                if t_0_batch_len < max_batch && i_0 < resources.len() - 1 {
                    continue;
                }

                let r = futures::stream::iter(motifs.clone())
                    .map(|value| {
                        let t_0_batch = t_0_batch.clone();//cones Arcs
                        let value = Arc::clone(&value);

                        debug!("SelectMotifsStage::run: t_0_batch={t_0_batch:?} value={value:?}");

                        async move {

                        let _ = DevicesWrap::with_two_arrays(t_0_batch.as_slice(), &[value], device_idx, None, |batch, motifs|{
                            debug!("SelectMotifsStage::run: batch={batch:?} motifs={motifs:?}");

                            let (batch_0_t, batch_0_sizes) =  unfold_and_cat(&batch, &UnfoldParams::new(1, motifs[0].extra().1.window as i64, motifs[0].extra().1.step as i64))?;


                            Ok(())
                        }).await?;

                        Ok::<(), ThoosaError>(())
                    }})
                    .buffer_unordered(10)
                    .try_collect::<Vec<_>>()
                    .await?;

            }
        }



        Ok(Arc::new(DevicesWrap::wrap_cpu(SelectMotifsSavableResult::new(0, self.run_configs[0].clone(), vec![]))))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use log::debug;
    use log::LevelFilter::Trace;
    use parking_lot::RwLock;
    use tch::kind::FLOAT_CPU;
    use tch::Tensor;
    use caching_runner::common::cachable::DevicesWrap;
    use caching_runner::common::{mock_run_ctx, MockRunCtx};
    use caching_runner::common::runnable::AsyncStage;
    use caching_runner::global_ctx;
    use caching_runner::global_ctx::{GlobalCtx};
    use thoosa_common::find_motifs_run_config::{ByLabelVarianceConfig, FindMotifsRunConfig, MotifSelector, SelectMotifsRunConfig};
    use thoosa_common::runs::Labeled;
    use thoosa_common::utils::reset_print_options;
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::{gen_motif, MotifKind, TensorIdSliceIdx, ThoosaError};
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::traits::{DataId, SimpleId};
    use thoosa_store::MockStoreBuilder;
    use torch_runner::data::ThoTensor2;
    use crate::tasks::cross_dims::ChannelsData2;
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::motif_loader::{mock_motif_loader_2, MotifMeta};
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::select_motifs::SelectMotifsStage;
    use crate::tasks::tensor_loader::{mock_tensor_loader};

    #[ctor::ctor]
    fn test_init() {
        thoosa_shared::test_utils::init_test_logging();
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_two_labels() -> Result<(), ThoosaError> {
        log::set_max_level(Trace);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 3, 100, 1000)));
        let run_id = 1;

        let resource_a: Resource = (1, 1, "resource_a", 1000).into();
        let resource_b: Resource = (2, 1, "resource_b", 1000).into();
        let resource_c: Resource = (3, 1, "resource_b", 1000).into();

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 1;

        let motif_0_in_channel_0 = 1u32;
        let motif_1_in_channel_0 = 2u32;
        let motif_0_in_channel_1 = 3u32;

        let resources_group = "group";

        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
                .with_list_run_config_values(
                    run_id,
                    Ok(FindMotifsRunConfig::new(Some(run_id), 100, 100, None, None).to_properties_iter(None).collect()),
                )
                .with_list_motifs([
                    (find_motifs_run_id as RunId, Ok(vec![
                        motif_0_in_channel_0 as usize,
                        motif_1_in_channel_0 as usize,
                        motif_0_in_channel_1 as usize
                    ]))
                ])
                .with_list_group_labeled_resources_sets([
                    (resources_group, Ok(vec![
                        ("Lable_0".to_string(), "resource_set_for_label_0".to_string()),
                        ("Lable_1".to_string(), "resource_set_for_label_1".to_string())
                    ]))
                ])
                .with_list_resource_set_resources_by_descr([
                    ("resource_set_for_label_0", Ok(vec![resource_a.clone(), resource_b.clone()])),
                    ("resource_set_for_label_1", Ok(vec![resource_c.clone()]))
                ])

                .build()));


        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached([
                    (resource_a.clone(), Tensor::rand(&[3, 10000], FLOAT_CPU)),
                    (resource_b.clone(), Tensor::rand(&[3, 10000], FLOAT_CPU)),
                    (resource_c.clone(), Tensor::rand(&[3, 10000], FLOAT_CPU)),
                ])
                .finish(),
        );

        let motif_loader = Arc::new(mock_motif_loader_2::build().with_run_cached([
            (motif_0_in_channel_0, (MotifMeta { channel: 0, run_id: find_motifs_run_id, window: 100 }, motifs_tensor(100, 3, MotifKind::Pulse))),
            (motif_1_in_channel_0, (MotifMeta { channel: 0, run_id: find_motifs_run_id, window: 100 }, motifs_tensor(100, 2, MotifKind::Triangle))),
            (motif_0_in_channel_1, (MotifMeta { channel: 1, run_id: find_motifs_run_id, window: 100 }, motifs_tensor(100, 4, MotifKind::Sin))),
        ]).finish());


        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new_(Some(select_motifs_run_id), MotifSelector::ByLabelVariance(ByLabelVarianceConfig { _dummy: 0 }))],
            vec![find_motifs_run_id],
        ));


        let result = select_motifs_stage.run(
            Arc::new(DevicesWrap::wrap_cpu(ResourcesConfigInput::new("test".into(), Labeled::new(resources_group.into())))), Arc::new(RwLock::new(mock_run_ctx::default())),
        ).await?;

        Ok(())
    }

    ///found motif's slices of same kind
    //TODO make slices a bit different
    fn motifs_tensor(motif_len:usize, num_slices: usize, motif_kind: MotifKind) -> Tensor {
        let t = Tensor::stack(
            (0..num_slices).map(|_| Tensor::from_slice(gen_motif(motif_len, &motif_kind).as_slice())).collect::<Vec<_>>().as_slice(), 0
        );

        debug!("motifs_tensor: t.size={:?}", t.size());

        t
        // Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
    }
}
