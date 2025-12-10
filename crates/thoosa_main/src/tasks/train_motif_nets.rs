use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use async_trait::async_trait;
use futures_util::{StreamExt, TryStreamExt};
use log::{debug, error, info, trace};
use parking_lot::RwLock;
use plotters::prelude::BLACK;
use tch::Device::{Cpu, Cuda};
use tch::kind::FLOAT_CUDA;
use tch::nn::VarStore;
use tch::{IndexOp, Tensor};
use caching_runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage, SyncStage};
use caching_runner::global_ctx::GlobalCtx;
use thoosa_common::applicables::ThoDevice;
use thoosa_common::batches::AsyncDataBatch;
use thoosa_common::dims::{Batch, Dims, MotifProbability, Slice, ThoBatch};
use thoosa_common::runs::Empty;
use thoosa_common::tho_tensor::{Labels, ThoTensor};
use thoosa_common::train_nets::{opt_train_2, run_train_motif_net, run_train_motif_net_2, MotifNetParams, OptConfig, TensorsDataBatch, TrainMotifsNetsRunConfig, TrainMotifsNetsRunConfigFields, TrainMotifsNetsResult, TrainMotifsNetResult};
use thoosa_common::utils::{mix_x_binary_with_labels, plot_series_with_lanes, vec_f32_from, DataBatch, DummyListener};
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_run_config_args::OptRunConfigSource;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::resource::Resource;
use thoosa_shared::run_configs::{OptRunConfigs, RunId};
use thoosa_shared::{CopyToDevice, ThoosaError};
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, DataId, SimpleId, WithId};
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_store::Store;
use thoosa_with_data::with_data;
use crate::tasks::cross_dims::{ChannelsData2, MotifSlices};
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta, RandomSlicesLoader, SlicesLoaderInput};
use crate::tasks::motifs_batch_builder::{make_batch_with, MixupJitterCfg};
use crate::tasks::run_ctx_2::MotifsCtx;
use crate::tasks::tensor_loader::RawTensorLoader;

#[derive(Debug)]
pub struct TrainMotifNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    run_configs: Arc<OptRunConfigs<
        Vec<NamedVal<TrainMotifsNetsRunConfigFields>>,
        Vec<NamedRange<TrainMotifsNetsRunConfigFields>>,
        Vec<Hyperparameter<TrainMotifsNetsRunConfigFields>>,
        TrainMotifsNetsRunConfigFields,
    >>,
    motif_loader:Arc<MotifLoader<L,C>>,
    random_slices_loader:Arc<RandomSlicesLoader<L,C>>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<RwLock<Box<dyn Store>>>,
    pd:PhantomData<C>

}


impl <L, C> OutId for TrainMotifNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        0 //TODO format!("TrainMotifNet_{}", input_id).into()
    }
}


#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct TrainMotifNetInput{
    id:SimpleId,
    select_motifs_run_id: Vec<RunId>,
    opt_config:Arc<OptConfig>
}

impl Input for TrainMotifNetInput {}
impl TrainMotifNetInput {
    pub fn new(select_motifs_run_id: Vec<RunId>, opt_config:Arc<OptConfig>) -> Self {
        Self { id: SimpleId("TrainMotifNetInput"), select_motifs_run_id, opt_config }
    }
}

impl WithId for TrainMotifNetInput {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}
impl <L, C> Stage for TrainMotifNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    type Ctx = C;
    type In<'a> = TrainMotifNetInput;//MotifID
    type Out = TrainMotifsNetsResult;
}

impl <L, C> TrainMotifNets<L, C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
{
    pub fn new(
        loader:Arc<L>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        run_configs:  OptRunConfigs<
            Vec<NamedVal<TrainMotifsNetsRunConfigFields>>,
            Vec<NamedRange<TrainMotifsNetsRunConfigFields>>,
            Vec<Hyperparameter<TrainMotifsNetsRunConfigFields>>,
            TrainMotifsNetsRunConfigFields,
        >    ) -> Self {
        let motif_loader = Arc::new(MotifLoader::new(
            Arc::clone(&store),
            Arc::clone(&loader)
        ));

        let random_slices_loader = Arc::new(RandomSlicesLoader::new(
            Arc::clone(&store),
            Arc::clone(&loader)
        ));

        Self {
            motif_loader,
            random_slices_loader,
            run_configs: Arc::new(run_configs),
            ws: Arc::clone(&ws),
            store: Arc::clone(&store),
            pd: PhantomData
        }
    }

}
// #[async_trait]
impl <L, C> AsyncStage for TrainMotifNets<L, C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
{
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
        let (device_idx, cache_area, ctx_name) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area(), guard.name())
        };

        let (select_motifs_run_ids, opt_config) = input.with_data_ref(
            0,
            Arc::clone(&ctx), None, |input| {
                Ok((input.select_motifs_run_id.clone(), input.opt_config.clone()))
            },
        ).await?;

        let channels = self.ws.read().channels();

        let out_key = {
            let out_id = self.out_id(input.id());
            Arc::new(out_id) as Arc<dyn CachableIdTr>
        };


        let results = futures::stream::iter(select_motifs_run_ids.into_iter().flat_map(|select_motifs_run_id|{
            let selected_motifs = self.store.read().list_selected_motifs(select_motifs_run_id, channels).unwrap();

            debug!("TrainMotifNets::run: select_motifs_run_id={select_motifs_run_id} selected_motifs={selected_motifs:?}");

            selected_motifs.into_iter()
                .flat_map(move |channels_motifs|
                    channels_motifs.into_iter().map(move |v| (v, select_motifs_run_id)))
        }))
        .map(|(motif_id_rate, select_motifs_run_id)|{
            let motif_loader = Arc::clone(&self.motif_loader);
            let random_slices_loader = Arc::clone(&self.random_slices_loader);

            let ctx = Arc::clone(&ctx);
            let opt_config = Arc::clone(&opt_config);
            let train_motifs_nodes_run_configs = Arc::clone(&self.run_configs);
            let store = Arc::clone(&self.store);
            let out_key = Arc::clone(&out_key);

            tokio::spawn(async move {
                let motif = motif_loader.run(Arc::new(DevicesWrap::wrap_cpu(MotifLoaderInput::all_slices(motif_id_rate.0.into()))), Arc::clone(&ctx)).await?;
                let random_slices = random_slices_loader.run(Arc::new(DevicesWrap::wrap_cpu(SlicesLoaderInput::new(motif_id_rate.0.into(), motif.size().dim_0() as u32))), Arc::clone(&ctx)).await?;


                // let batch_ref = &batch;
                //TODO async OOM
                opt_train_2(
                    opt_config,
                    async |vs: &VarStore, run_config: TrainMotifsNetsRunConfig, _listener| {
                        if ctx.read().acquired_permits() == 0 { //TODO move to with_data!
                            let result = GlobalCtx::global()
                                .acquire_many(device_idx, 1, out_key.clone())
                                .await?;
                            ctx.write().add_permit(result); //permits can be dropped only from OOM handler or when context is dropped
                        }

                        let batch = MotifsBatch::new(Arc::clone(&motif), Arc::clone(&random_slices), run_config.common_training_args.batch_size as i64, device_idx, cache_area).await?;

                        run_train_motif_net_2(batch.motif_meta.window, &run_config, &ThoDevice::from(device_idx), &batch, vs, &mut DummyListener).await
                    },
                    train_motifs_nodes_run_configs,
                    ThoDevice::from(device_idx).into(),
                    "motifs",
                ).await.map(|(var_store, motif_node_params, train_motifs_nets_run_config, min_loss)| {
                    info!("TrainMotifNet::run: motif_node_params={motif_node_params:?} train_motifs_nets_run_config={train_motifs_nets_run_config:?} min_loss={min_loss}");
                    store.read().get_motif_meta(motif_id_rate.0).and_then(|(channel, find_motifs_run_id)|{
                        Ok(TrainMotifsNetResult::new(
                            min_loss,
                            channel as u32,
                            motif_id_rate.0,
                            var_store,
                            select_motifs_run_id,
                            motif_node_params,
                            train_motifs_nets_run_config,
                        ))
                    })
                })
            })

        })
        .buffer_unordered(10)//TODO config
        .filter_map(|join_res| async move {
            match join_res {
                Ok(opt_i) => opt_i.ok().and_then(|v|v.ok()),
                Err(e) => {
                    error!("train motif failed: {e}");
                    None
                }
            }
        })
        .collect::<Vec<_>>()
        .await;


        Ok(Arc::new(DevicesWrap::wrap_cpu(TrainMotifsNetsResult::new(0, results))))
    }

}

#[derive(Debug)]
struct DataWithLabels {
    slices:ThoBatch<Slice>,
    labels: ThoTensor<Labels, SimpleId>
}

impl DataWithLabels {
    pub fn new(slices: ThoBatch<Slice>, labels: ThoTensor<Labels, SimpleId>) -> Self {
        Self { slices, labels }
    }
}

#[async_trait]
impl CopyToDevice for DataWithLabels {

    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoosaError> {
        Ok(DataWithLabels {
            slices: self.slices.copy_to_device(dst_device).await?,
            labels: self.labels.copy_to_device(dst_device).await?,
        })
    }
}

impl WithId for DataWithLabels {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl Data for DataWithLabels {
    type Metrics = <Batch<Slice> as Dims>::Metrics;

    fn size(&self) -> Self::Metrics {
        self.slices.size().try_into().unwrap()//TODO unwrap
    }

    fn device(&self) -> usize {
        self.slices.device()
    }
}

#[derive(Debug)]
pub struct MotifsBatch {
    batch_size: u32,

    train_motifs:Arc<DevicesWrap<ThoTensor<MotifSlices, SimpleId>>>,
    valid_motifs:Arc<DevicesWrap<ThoTensor<MotifSlices, SimpleId>>>,
    
    random_slices:Arc<DevicesWrap<ThoTensor<MotifSlices, DataId>>>,
    motif_meta:MotifMeta,
    device_idx:usize
}

impl MotifsBatch {

    pub async fn new(motif: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>>>, random_slices:Arc<DevicesWrap<ThoTensor<MotifSlices, DataId>>>, batch_size:i64, device_idx:usize, cache_area:u64) -> Result<Self, ThoosaError> {
        debug!("TrainMotifNet::run: motif={motif:?}");


        with_data!(device_idx, motif; |motif:&DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>|{
            let motif_meta = motif.extra().clone();

            let num_motif_slices = motif.data().t_ref().size()[0];
            assert!(num_motif_slices > 1);
            let num_validation_slices = (num_motif_slices as f32 * 0.2) as i64;
            let num_validation_slices = if num_validation_slices <= 0 {1} else {num_validation_slices};
            let split = num_motif_slices - num_validation_slices;

            debug!("MotifsBatch::new: motif={:?} num_motif_slices={num_motif_slices} num_validation_slices={num_validation_slices} split={split}", motif.extra());

            let motifs_t = motif.data().t_ref();
            let train_motifs = motifs_t.i((0..split, .. ));
            let valid_motifs = motifs_t.i((split.., .. ));

            Ok::<_, ThoosaError>(Self {

                train_motifs:  Arc::new(DevicesWrap::wrap(train_motifs.into(), cache_area)),
                valid_motifs: Arc::new(DevicesWrap::wrap(valid_motifs.into(), cache_area)),
                random_slices,
                motif_meta,
                batch_size: batch_size as u32,
                device_idx
            })
        })
    }

    async fn make_batch(&self, motifs: Arc<DevicesWrap<ThoTensor<MotifSlices, SimpleId>>>) -> Result<(ThoBatch<Slice>, Tensor), ThoosaError> {

        static COUNTER:AtomicU32 = AtomicU32::new(0);

        let random_slices = Arc::clone(&self.random_slices);

        let num_slices = (self.batch_size / 2) as i64;

        with_data!(self.device_idx, motifs, random_slices; |motifs:&ThoTensor<MotifSlices, SimpleId>, random_slices:&ThoTensor<MotifSlices, DataId>|{

            let motifs = make_batch_with(motifs.t_ref(), num_slices, MixupJitterCfg { do_mixup: true, do_jitter: true, alpha: 0.4, sigma: 0.0 })?.set_requires_grad(true);//TODO config
            let random = make_batch_with(random_slices.t_ref(), num_slices, MixupJitterCfg { do_mixup: true, do_jitter: true, alpha: 0.4, sigma: 0.0 })?.set_requires_grad(true);

           /*
           let test_dir = Path::new("/tmp/motifs");//.join(cdt_for_dir());
            plot_series_with_lanes(
                (0..random.size()[0]).map(|i| {
                    (format!("i={i}"), BLACK, vec_f32_from(&random.i((i, ..))).into_iter().enumerate().map(|(i, v)| (i as f32, v)).collect::<Vec<(f32, f32)>>())
                }).collect::<Vec<_>>(),
                format!("sig").as_str(),
                (2000, 1000),
                &test_dir.join(&format!("motif__{}.png", COUNTER.fetch_add(1, Relaxed))),
                &[(0, 1000)]
            ).unwrap();
            */

            let (x,y) = mix_x_binary_with_labels(&motifs, &random);

            debug!("MotifsBatch::train_batch: motifs={:?} random={:?}", motifs.device(), random.device());
            debug!("MotifsBatch::train_batch: x={:?} y={:?}", x.device(), y.device());

            Ok::<_, ThoosaError>((x.into(), y))
        })
    }
}

#[async_trait]
impl AsyncDataBatch<Slice, MotifProbability> for MotifsBatch {
    async fn train_batch(&self) -> Result<(ThoBatch<Slice>, Tensor), ThoosaError> {
        debug!("MotifsBatch::train_batch: self.device_idx={}", self.device_idx);

        let motifs = Arc::clone(&self.train_motifs);

        self.make_batch(motifs).await
    }

    async fn valid_batch(&self) -> Result<(ThoBatch<Slice>, Tensor), ThoosaError> {
        debug!("MotifsBatch::valid_batch: self.device_idx={}", self.device_idx);

        let motifs = Arc::clone(&self.valid_motifs);

        self.make_batch(motifs).await
    }
}


// impl SyncStage

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Arc;
    use log::debug;
    use log::LevelFilter::{Debug, Trace};
    use parking_lot::RwLock;
    use tch::Device;
    use tokio::fs;
    use caching_runner::common::cachable::DevicesWrap;
    use caching_runner::common::mock_run_ctx;
    use caching_runner::common::runnable::AsyncStage;
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_common::dims::{Batch, MotifProbability, Slice};
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::train_nets::{CommonTrainingRunConfigFields, MotifNetParams, OptConfig, ThoModuleWithStore, TrainMotifsNetsResult, TrainMotifsNetsRunConfigFields};
    use thoosa_common::utils::{reset_print_options, test_net};
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::{MotifKind, TensorIdSliceIdx, ThoosaError};
    use thoosa_shared::hyperparameter::Hyperparameter;
    use thoosa_shared::val::{NamedRange, NamedVal, Val};
    use thoosa_store::MockStoreBuilder;
    use thoosa_with_data::with_data;
    use crate::tasks::select_motifs::tests::resource_tensor;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use crate::tasks::train_motif_nets::{TrainMotifNets, TrainMotifNetInput};

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_train_motif_nets() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp/test_train_motif_nets");//.join(cdt_for_dir());
        fs::create_dir_all(test_dir).await?;

        log::set_max_level(Debug);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 10000)));
        let find_motifs_run_id:RunId = 1;
        let select_motifs_run_id:RunId = 2;
        let motif_width = 100;

        let resources = (0..10).map(|i|Resource::new(i, 1, format!("resource_{i}"), 900)).collect::<Vec<_>>();

        let (resources_map, motif_intervals) = resources.iter().try_fold((HashMap::new(), vec![]), |mut a, r| {

            let (t, intervals) = resource_tensor(r.length() as usize, &[
                vec![(3, motif_width, MotifKind::Pulse)],
                vec![(1, motif_width, MotifKind::Sin)],
                vec![]
            ], 100)?;

            a.0.insert(r.clone(), t);

            //0 channel - the only motif
            a.1.extend(intervals[0][0].iter().map(|interval|TensorIdSliceIdx(r.id(), interval.start as u32)));

            Ok::<_, ThoosaError>(a)
        })?;


       let resources_id_map = resources.iter().fold(HashMap::new(), |mut a, v|{
           a.insert(v.id().clone(), v.clone());
           a
       });


        // let motif_slices = [resource_a_intervals_a, resource_a_intervals_b].into_iter().enumerate().flat_map(|(intervals|)

        let motif_id = 10u32;
        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
                .with_list_run_config_values(
                    find_motifs_run_id,
                    Ok(FindMotifsRunConfig::new(Some(find_motifs_run_id), 50, 100, None, None).to_properties_iter(None).collect()),
                )
                .with_list_motif_slices(motif_id, None, None, None, Ok((0i64, find_motifs_run_id, /*vec![
                    TensorIdSliceIdx(1, 100),
                    TensorIdSliceIdx(1, 200),
                    TensorIdSliceIdx(2, 100)
                ]*/motif_intervals,
                                                                  resources_id_map.clone()

                )))
                .with_list_random_slices(motif_id, motif_width as u32, None, Ok((0i64, find_motifs_run_id, vec![
                    TensorIdSliceIdx(1, 100),
                    TensorIdSliceIdx(1, 200),
                    TensorIdSliceIdx(2, 100)
                ], resources_id_map)))

                .with_list_selected_motifs([
                    (select_motifs_run_id, Ok(vec![vec![(motif_id, 1.0f64)]]))
                ])
                .with_get_motif_meta( [(motif_id, Ok((0usize, find_motifs_run_id)))])

                .build()));

        let run_configs =         thoosa_shared::run_configs::OptRunConfigs::new(
            thoosa_shared::run_configs::GridRunConfigs::new(
                thoosa_shared::run_configs::ConstConfigs::new(vec![
                    NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Cycles), Val::Int(20)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::BatchSize), Val::Int(11)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), Val::Real(0.001)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::OptMaxEvals), Val::Int(10)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::OptNInit), Val::Int(10)),
                ]),
                vec![
                    // NamedRange::new(TrainMotifsNetsRunConfigFields::NumLayers, Val::Int(0)..Val::Int(10), (Val::Int(10), 0))
                ],
            ),
            vec![
                Hyperparameter::UniformInt{name: TrainMotifsNetsRunConfigFields::NumLayers, lower: 3, upper: 4}
            ],
        );

        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached(resources_map)
                .finish(),
        );


        let train_motif_net = Arc::new(TrainMotifNets::new(
            loader,
            &ws,
            &store,
            run_configs
        ));

        let input = Arc::new(DevicesWrap::wrap_cpu(TrainMotifNetInput::new(vec![select_motifs_run_id], Arc::new(OptConfig::new(100, 3)))));

        let r = train_motif_net.run(input, Arc::new(RwLock::new(mock_run_ctx::default(1)))).await?;

        with_data!(0, r; |r:&TrainMotifsNetsResult|{

            debug!("test_num_layers_opt: result={r:?}");

            assert!(r.train_motif_nets_results[0].loss < 0.2);

            Ok::<_, ThoosaError>(())
        })?;

        Ok(())
    }

}
