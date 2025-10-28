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
use tokio::io::AsyncWriteExt;
use caching_runner::common::cachable::{with_two_arrays_ext, with_two_vars_ext, Cached, DevicesWrap};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage};
use caching_runner::tmp_cache::TmpCached;
use thoosa_common::applicables::ThoDevice;
use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig, SelectMotifsRunConfig};
use thoosa_common::runs::Labeled;
use thoosa_common::utils::pad_to;
use thoosa_common::ws::Workspace;
use thoosa_motifs::distance::MaskMaker::{Le, LocalMin};
use thoosa_motifs::find::{FixedFindMotifsConfig, SelectMotifsSavableResult};
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Adjustable, DataId, SimpleId, WithId};
use thoosa_shims::stream::{StreamGuard, ThoCudaStream};
use thoosa_store::Store;
use thoosa_with_data::with_data;
use torch_runner::data::ThoTensor2;
use crate::cli::Commands::SelectMotifs;
use crate::tasks::average_euclidean_distance::{AverageDistanceValue, AverageEuclideanDistanceStage};
use crate::tasks::cross_batch_clc::{unfold_and_cat, CrossInput, UnfoldParams};
use crate::tasks::cross_dims::{ChannelsData2, ChannelsMotifsDists, ChannelsMotifsSlices, MotifSlices};
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::load_motif_and_clc_dist::LoadMotifAndClcDistanceStage;
use crate::tasks::motif_loader::{MotifLoader, MotifMeta};
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::slices_iter::SlicesIter;
use crate::tasks::tensor_loader::RawTensorLoader;

#[derive(Debug)]
pub struct SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{

    store: Arc<RwLock<Box<dyn Store>>>,
    ws:Arc<RwLock<Box<dyn Workspace>>>,
    loader:Arc<L>,
    motif_loader:Arc<ML>,
    pd:PhantomData<C>,
    run_configs:Vec<SelectMotifsRunConfig>,
    find_motifs_run_ids:Vec<RunId>,
    load_and_clc_avg_dist:Arc<LoadMotifAndClcDistanceStage<ML, AverageEuclideanDistanceStage<C>, C>>
}

impl <L, ML, C> SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{

    pub fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        loader:Arc<L>,
        motif_loader:Arc<ML>,
        run_configs:Vec<SelectMotifsRunConfig>,
        find_motifs_run_ids:Vec<RunId>,
    ) -> Self {
        let load_and_clc_avg_dist = Arc::new(LoadMotifAndClcDistanceStage::new(
            Arc::clone(&motif_loader), // as Arc<dyn MotifLoaderStage<Output = <AverageEuclideanDistanceStage as Stage>::In<'static>>>,
            Arc::new(AverageEuclideanDistanceStage::<C>::new()),
        ));

        Self {
            store: Arc::clone(store),
            ws: Arc::clone(ws),
            loader,
            motif_loader,
            pd: Default::default(),
            run_configs,
            find_motifs_run_ids,
            load_and_clc_avg_dist
        }
    }

}


impl <L,ML, C> Stage  for  SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static

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
    C: RunCtx + Adjustable + 'static
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



        let tmp_moitfs_cache:TmpCached<DataWithExtra<ThoTensor2<ChannelsMotifsSlices, SimpleId>, (Vec<Vec<DataId>>, FindMotifsRunConfig)>> = TmpCached::new("motif slices", cache_area).await;
        let tmp_motifs_avg_dists_cache:TmpCached<ThoTensor2<ChannelsMotifsDists, SimpleId>> = TmpCached::new("motif dists", cache_area).await;

        let num_channels = self.ws.read().channels();
        let mut motif_avg_dists = Arc::new(parking_lot::RwLock::new(HashMap::new()));

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
                        let motif_loader_clone = Arc::clone(&self.motif_loader);
                        let load_and_clc_avg_dist_clone = Arc::clone(&self.load_and_clc_avg_dist);
                        let ctx = Arc::clone(&ctx);
                        let ctx_for_clc = Arc::clone(&ctx);
                        let motif_avg_dists = Arc::clone(&motif_avg_dists);

                        async move {
                            let motif_id_wrap = Arc::new(DevicesWrap::wrap_cpu((motif_id as u32).into()));
                            let mut motif = motif_loader_clone.run_cached(Arc::clone(&motif_id_wrap), ctx).await?;
                            debug!("SelectMotifsStage::run: find_motifs_run_id={find_motifs_run_id} run_config.id={:?} motif={motif:?}", select_motifs_run_config.id);


                            let avg_motif_dist = load_and_clc_avg_dist_clone
                                .run_cached(Arc::clone(&motif_id_wrap), ctx_for_clc)
                                .await?
                                .with_data_ref(0, None, |dist_val: &AverageDistanceValue| Ok(dist_val.value()))
                                .await?;

                            motif_avg_dists.write().insert(DataId(motif_id as u32), avg_motif_dist);


                            Ok::<_, ThoosaError>(motif)
                        }
                    })
                    .buffer_unordered(10)
                    .try_collect::<Vec<_>>()
                    .await?;

                debug!("SelectMotifsStage::run: loaded_motifs={loaded_motifs:?}");

                //building tensor with all motifs' slices (ChannelsMotifSlices) found by current find_motifs_run_id, storing them in tmp_cache. Channels where number of motifs is less than max slices are padded with zeros.
                // channels x motif_0_slice_0...motif_0_slice_last + motif_1_slice_0...motif_1_slice_last + ...
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
                    let mut channels_motifs_tensors_vec = vec![];
                    let mut channels_motif_avg_dists_tensors_vec = vec![];
                    for c in 0..num_channels {

                        let channel_motifs = &channels_motifs.get(c as usize).expect(&format!("not enough channels: len={} c={}", channels_motifs.len(), c));

                        debug!("SelectMotifsStage::run: channel {c} max_motif_slices={max_motif_slices} channel_motifs={channel_motifs:?}");

                        let channel_motifs_tensors = channel_motifs.iter().map(|motif|  {
                            debug!("SelectMotifsStage::run: motif={motif:?} id={}", motif.id());
                            for _ in 0..motif.size()[0] {
                                channels_motif_ids[c as usize].push(motif.id().clone());
                            }

                            motif.t_ref().copy()
                        }).collect::<Vec<_>>();

                        //building dists tensors for each channel to compare matches with
                        let channel_avg_dists_tensors = channel_motifs.iter().map(|motif|  {
                            let read_guard = motif_avg_dists.read();
                            let avg_dist = read_guard.get(&motif.id()).unwrap();

                            debug!("SelectMotifsStage::run: motif.id={} dist={avg_dist}", motif.id());

                            Tensor::from_slice(vec![*avg_dist; motif.size()[0] as usize].as_slice())
                        }).collect::<Vec<_>>();

                        debug!("SelectMotifsStage::run: c={c}, channel_motifs_tensors={:?}", channel_motifs_tensors);

                        let (motifs_t, moitf_avg_dists_t) = if channel_motifs_tensors.is_empty() {
                            (
                                Tensor::zeros(&[max_motif_slices as i64, find_motifs_run_config.window  as i64], (Float, ThoDevice::from(device_idx).into())),
                                Tensor::zeros(&[max_motif_slices as i64], (Float, ThoDevice::from(device_idx).into()))
                            )

                        } else {
                            let channel_t = Tensor::cat(&channel_motifs_tensors, 0);
                            let motif_avg_dists_t = Tensor::cat(&channel_avg_dists_tensors, 0);

                            if channel_t.size()[0] < max_motif_slices as i64 {
                                debug!("SelectMotifsStage::run: c={c}, channel_t.size={:?} padding to {:?}", channel_t.size(), [max_motif_slices as i64, find_motifs_run_config.window  as i64]);
                                (
                                    pad_to(&channel_t, &[max_motif_slices as i64, find_motifs_run_config.window  as i64]).expect("pad to failed"),
                                    pad_to(&motif_avg_dists_t, &[max_motif_slices as i64]).expect("pad to failed")
                                )

                            } else {
                                debug!("SelectMotifsStage::run: c={c}, channel_t.size={:?}", channel_t.size());

                                (channel_t, motif_avg_dists_t)
                            }
                        };

                        channels_motif_avg_dists_tensors_vec.push(moitf_avg_dists_t);
                        channels_motifs_tensors_vec.push(motifs_t);

                    };

                    debug!("SelectMotifsStage::run: channels_dists_tensors_vec={channels_motif_avg_dists_tensors_vec:?}");
                    debug!("SelectMotifsStage::run: channels_motifs_tensors_vec={channels_motifs_tensors_vec:?}");

                    let channels_motifs_t = Tensor::stack(&channels_motifs_tensors_vec, 0);
                    let channels_motif_avg_dists_t = Tensor::stack(&channels_motif_avg_dists_tensors_vec, 0);

                    debug!("SelectMotifsStage::run: num_channels={num_channels}, channels_motif_ids={channels_motif_ids:?} channels_motifs_t.size={:?}", channels_motifs_t.size());
                    trace!("SelectMotifsStage::run: channels_dists_t=\n{channels_motif_avg_dists_t}");
                    trace!("SelectMotifsStage::run: channels_motifs_t=\n{channels_motifs_t}");

                    let find_and_select_motifs_id = SimpleId(format!("{}_{}", find_motifs_run_config.id.expect("no id in find_motifs_run_config"), select_motifs_run_config.id.expect("no id in select_motifs_run_config")).leak());
                    tmp_moitfs_cache.insert(
                        DataWithExtra::new(
                            ThoTensor2::new(
                                find_and_select_motifs_id.clone(),
                                channels_motifs_t
                            ),
                            (channels_motif_ids, find_motifs_run_config.clone())
                        )//TODO leak
                    );

                    tmp_motifs_avg_dists_cache.insert(
                        ThoTensor2::new(
                            find_and_select_motifs_id,
                            channels_motif_avg_dists_t
                        )
                    );

                    Ok(())
                }).await?;
            }
        }


        let find_motifs_runs_motifs = tmp_moitfs_cache.clone_to_arcs_vec();


        debug!("SelectMotifsStage::run: motif_dists={motif_avg_dists:?}");

        for label_resources in labels_resources {
            let resources = &label_resources.1;
            //t_0_batch - moitfs
            let mut t_1_batch: Vec<_> = vec![];
            let mut t_1_batch_resources: Vec<_> = vec![];
            let mut t_1_batch_len = 0;

            let mut skip_0_i = 0;

            for (i_0, resource_0) in resources.iter().enumerate() {
                info!("on i_0={i_0} resource_0={resource_0:?}");

                let t_0 = Arc::clone(&self.loader)
                    .run_cached(Arc::new(DevicesWrap::wrap_cpu(resource_0.clone())), Arc::clone(&ctx))
                    .await?;

                t_1_batch_len += t_0.size().dim_1();
                t_1_batch.push(t_0);
                t_1_batch_resources.push(resource_0);

                if t_1_batch_len < max_batch && i_0 < resources.len() - 1 {
                    continue;
                }

                let r = futures::stream::iter(find_motifs_runs_motifs.clone())
                    .map(|motifs_arc_wrap| {
                        let t_1_batch = t_1_batch.clone(); //cones Arcs
                        // let value_vec = vec![Arc::clone(&value)];

                        let motifs_avg_dists_arc_wrap = tmp_motifs_avg_dists_cache.get(motifs_arc_wrap.id()).expect(&format!("no dists for {}", motifs_arc_wrap.id()));


                        debug!("SelectMotifsStage::run: t_0_batch={t_1_batch:?} \nmotifs_arc_wrap={motifs_arc_wrap:?} \ndists_arc_wrap={motifs_avg_dists_arc_wrap:?}");
                        let t_1_batch_resources = t_1_batch_resources.clone();
                        async move {

                            with_data!(1, [t_1_batch], motifs_arc_wrap, motifs_avg_dists_arc_wrap;
                                |
                                    batch_v:&[&ThoTensor2<ChannelsData2, SimpleId>],
                                    motifs_v:&DataWithExtra<ThoTensor2<ChannelsMotifsSlices, SimpleId>, (Vec<Vec<DataId>>, FindMotifsRunConfig)>,
                                    motifs_avg_dists:&ThoTensor2<ChannelsMotifsDists, SimpleId>
                                |{

                                let unfold_params = UnfoldParams::new(1, motifs_v.extra().1.window as i64, motifs_v.extra().1.step as i64);
                                let (batch_1_t, batch_1_sizes) = unfold_and_cat(&batch_v, &unfold_params)?;
                                assert_eq!(batch_1_sizes.len(), batch_v.len());

                                let resource_ids_sizes_0: Vec<_> = t_1_batch_resources
                                    .into_iter()
                                    .zip(batch_1_sizes.into_iter())
                                    .map(|v| (v.0.id(), v.1 as usize))
                                    .collect();

                                let resources_iter = SlicesIter::new(unfold_params.step() as usize, resource_ids_sizes_0);

                                debug!("SelectMotifsStage::run: batch_0_t.size()={:?} motifs_v.data().t_ref().size={:?} motifs_avg_dists.t_ref().size()={:?}", batch_1_t.size(), motifs_v.data().t_ref().size(), motifs_avg_dists.t_ref().size());

                                let motifs_stream = ThoCudaStream::new(0); //TODO
                                let guard = StreamGuard::set(&motifs_stream); //TODO Device

                                let mut mask_makers = vec![
                                    LocalMin { dim: 1 },
                                    Le {
                                       // we'll have mask/dists from clc_dist_d2_and_mask in shape[channels x num_motifs x num_blocks_in_batch] so motif's average dists which are  [channels x num_motifs] should be expanded to match calculated dists
                                        v:motifs_avg_dists.t_ref().unsqueeze(-1).expand([-1, -1, batch_1_t.data().size()[1]], false)
                                    }
                                ];

                                let motifs_stream = ThoCudaStream::new(0); //TODO
                                let guard = StreamGuard::set(&motifs_stream); //TODO Device


                                let (mask, dists) = thoosa_motifs::distance::clc_dist_d2_and_mask(
                                    motifs_v.data().t_ref().shallow_clone(),
                                    batch_1_t,
                                    false, //if true dists are returned
                                    &mut mask_makers,
                                )?;
                                drop(guard);

                                debug!("SelectMotifsStage::run: mask={mask:?} dists={dists:?}");
                                trace!("SelectMotifsStage::run: mask=\n{mask}");

                                for (channel, motif_ids) in motifs_v.extra().0.iter().enumerate() {
                                    for motif_id in motif_ids {
                                        let resources_iter= resources_iter.clone();

                                        for slice_1 in resources_iter {

                                        }
                                    }
                                }



                                Ok::<(), ThoosaError>(())
                            })?;

                            Ok::<(), ThoosaError>(())
                        }
                    })
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
    use thoosa_motifs::generators::non_overlapping::{generate_series_with_motif, Interval};
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

        let resource_a: Resource = Resource::new(1, 1, "resource_a", 900);
        let resource_b: Resource = Resource::new(2, 1, "resource_b", 800);
        let resource_c: Resource = Resource::new(3, 1, "resource_b", 600);

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

        let (resource_a_t, resoruce_a_intervals) = resource_tensor(resource_a.length() as usize, 100, 3, &[MotifKind::Pulse, MotifKind::Triangle, MotifKind::Sin] );
        let (resource_b_t, resoruce_b_intervals) = resource_tensor(resource_b.length() as usize, 100, 3, &[MotifKind::Pulse, MotifKind::Triangle, MotifKind::Sin] );
        let (resource_c_t, resoruce_c_intervals) = resource_tensor(resource_c.length() as usize, 100, 2, &[MotifKind::Pulse, MotifKind::Triangle, MotifKind::Sin] );


        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached([
                    (resource_a.clone(), resource_a_t),
                    (resource_b.clone(), resource_b_t),
                    (resource_c.clone(), resource_c_t),
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
    pub fn motifs_tensor(motif_len:usize, num_slices: usize, motif_kind: MotifKind) -> Tensor {
        let t = Tensor::stack(
            (0..num_slices).map(|_| Tensor::from_slice(gen_motif(motif_len, &motif_kind).as_slice())).collect::<Vec<_>>().as_slice(), 0
        );

        debug!("motifs_tensor: t.size={:?}", t.size());

        t
    }

    pub fn resource_tensor(resource_len:usize, motif_len:usize, num_motifs:usize, channels_motif_kinds:&[MotifKind]) -> (Tensor, Vec<Vec<Interval>>) {

        let (channels_signals, intervals) = channels_motif_kinds.into_iter().map(|kind| {
            generate_series_with_motif(resource_len, motif_len, num_motifs, 300, 0.4, (0.8, 1.8), None, kind)
        }).fold((vec![], vec![]), |mut a, v|{
            a.0.push(v.0);
            a.1.push(v.1);
            a
        });

        let t = Tensor::stack(channels_signals.into_iter().map(|data|Tensor::from_slice(data.as_slice())).collect::<Vec<Tensor>>().as_slice(), 0);

        debug!("resource_tensor: t.size={:?} intervals={intervals:?}", t.size());

        (t, intervals)
    }
}