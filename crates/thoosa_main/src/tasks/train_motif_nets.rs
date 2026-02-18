#![allow(unexpected_cfgs)]

use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use async_trait::async_trait;
use futures_util::{FutureExt, StreamExt};
use futures_util::stream::BoxStream;
use log::{debug, error, info, trace, warn};
use parking_lot::{Mutex, RwLock};
use plotters::prelude::BLACK;
use plotters::style::RED;
use tch::Device::{Cpu, Cuda};
use tch::kind::FLOAT_CUDA;
use tch::nn::{AdamW, OptimizerConfig, VarStore};
use tch::{IndexOp, Reduction, Tensor};
use tch::Kind::Float;
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_shared::runner::cache_areas::CacheArea;
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_shared::runner::global_ctx::GlobalCtx;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::batches::{AsyncDataBatch, BatchLengths};
use thoosa_common::dims::{Batch, Dims, MotifProbability, Slice, ThoBatch};
use thoosa_common::runs::Empty;
use thoosa_common::tho_tensor::{Labels, ThoTensor};
use thoosa_common::train_nets::{opt_train, MotifNetParams, MotifNetTypeArgs, OptConfig, TensorsDataBatch, TrainMotifNetResult, TrainMotifsNetsResult, TrainMotifsNetsRunConfig, TrainMotifsNetsRunConfigFields};
use thoosa_common::utils::{mix_x_binary_with_labels, plot_series_with_lanes, plot_t, to_vec_f32, DataBatch, DummyListener};
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_run_config_args::OptRunConfigSource;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::resource::Resource;
use thoosa_shared::run_configs::{OptRunConfigs, RunId};
use thoosa_shared::{collect_futures_with_progress_i, mono_key, CopyToDevice, StreamBuffering, ThoError};
use thoosa_shared::traits::{Adjustable, Data, DataId, WithId};
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_store::Store;
use thoosa_with_data::with_data;
use thoosa_shared::comp_id::SynKey;
use crate::tasks::cross_dims::{ChannelsData2, MotifSlices};
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta, RandomSlicesLoader, SlicesLoaderInput};
use crate::tasks::motifs_batch_builder::{make_batch_with, MixupJitterCfg};
use crate::tasks::tensor_loader::RawTensorLoader;
use crate::tasks::train_net::{TrainNet, TrainNetInput};

#[derive(Debug)]
pub struct TrainMotifNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
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
    store: Arc<Box<dyn Store>>,
    pd:PhantomData<C>

}


impl <L, C> OutId for TrainMotifNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        0 //TODO format!("TrainMotifNet_{}", input_id).into()
    }
}


#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct TrainMotifNetInput{
    id: SynKey,
    select_motifs_run_id: Vec<RunId>,
    opt_config:Arc<OptConfig>
}

impl Input for TrainMotifNetInput {}
impl TrainMotifNetInput {
    pub fn new(select_motifs_run_id: Vec<RunId>, opt_config:Arc<OptConfig>) -> Self {
        Self { id: SynKey::new([mono_key!("TrainMotifNetInput", "stage")]), select_motifs_run_id, opt_config }
    }
}

impl WithId for TrainMotifNetInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}
impl <L, C> Stage for TrainMotifNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    type Ctx = C;
    type In<'a> = TrainMotifNetInput;//MotifID
    type Out = TrainMotifsNetsResult;
}

impl <L, C> TrainMotifNets<L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
{
    pub fn new(
        loader:Arc<L>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        run_configs:  OptRunConfigs<
            Vec<NamedVal<TrainMotifsNetsRunConfigFields>>,
            Vec<NamedRange<TrainMotifsNetsRunConfigFields>>,
            Vec<Hyperparameter<TrainMotifsNetsRunConfigFields>>,
            TrainMotifsNetsRunConfigFields,
        >    ) -> Self {
        let motif_loader = Arc::new(MotifLoader::new(
            Arc::clone(&store),
            Arc::clone(&loader),
            ws
        ));

        let random_slices_loader = Arc::new(RandomSlicesLoader::new(
            Arc::clone(&store),
            Arc::clone(&loader),
            ws
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

impl <L, C> AsyncStage for TrainMotifNets<L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainMotifNets::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {

        let (device_idx, cache_area) = (ctx.device_idx(), ctx.cache_area());

        let (select_motifs_run_ids, opt_config) = with_data!(ctx.branch(Some("train_motif_nets_input"), false, false), input; |_ctx, input: &TrainMotifNetInput| {
            Ok::<(Vec<RunId>, Arc<OptConfig>), ThoError>((input.select_motifs_run_id.clone(), input.opt_config.clone()))
        })?;

        let channels = thoosa_shared::with_perf_log(
            || self.ws.try_read_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
            "TrainMotifNets/run/read_ws",
        )
        .channels();


        //gpt Expand selection run ids into individual motif ids for training.
        let select_motifs_run_ids_clone = select_motifs_run_ids.clone();
        let motifs_rates_selected_motifs_run_ids = select_motifs_run_ids_clone.iter().flat_map(|select_motifs_run_id| {
            let selected_motifs = self.store.list_selected_motifs(*select_motifs_run_id, channels).expect("list_selected_motifs failed");

            debug!("TrainMotifNets::run: select_motifs_run_id={select_motifs_run_id} selected_motifs={selected_motifs:?}");

            selected_motifs.into_iter()
                .flat_map(move |channels_motifs|
                    channels_motifs.into_iter().map(move |motif_id_rate| (motif_id_rate, select_motifs_run_id)))
        }).collect::<Vec<_>>();
        let motifs_rates_selected_motifs_run_ids_len = motifs_rates_selected_motifs_run_ids.len();
        let futures = motifs_rates_selected_motifs_run_ids.into_iter()
        .map(|(motif_id_rate, select_motifs_run_id)|{
            let motif_loader = Arc::clone(&self.motif_loader);
            let random_slices_loader = Arc::clone(&self.random_slices_loader);

            let mut ctx = ctx.branch(Some(&format!("train_motif_nets_task_{}", motif_id_rate.0)), true, false);
            let opt_config = Arc::clone(&opt_config);
            let train_motifs_nodes_run_configs = Arc::clone(&self.run_configs);
            let store = Arc::clone(&self.store);
            async move {

                //gpt TODO!! MotifLoaderInput::all_slices ignores select_motifs_run_id; training may mix slices from other selection runs.
                let motif: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>>> = motif_loader.run(
                    Arc::new(DevicesWrap::wrap(MotifLoaderInput::all_slices(motif_id_rate.0.into()), CacheArea::default())),
                    ctx.branch(Some("train_motif_nets_motif_loader"), false, false),
                ).await?;

                let random_slices = random_slices_loader
                    .run(
                        Arc::new(DevicesWrap::wrap(SlicesLoaderInput::new(
                            motif_id_rate.0.into(),
                            motif.size().dim_0() as u32,
                        ), CacheArea::default())),
                        ctx.branch(Some("train_motif_nets_random_slices"), false, false),
                    )
                    .await?;


                let motif_width = with_data!(ctx.branch(Some("train_motif_nets_motif_width"), false, false), Arc::clone(&motif); |_ctx, motif:&DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>|{
                Ok::<_, ThoError>(motif.extra().window)
            })?;

                let random_slices_for_batch = Arc::clone(&random_slices);

                let train_net = Arc::new(TrainNet::new(
                    move |ctx:C, rc:&TrainMotifsNetsRunConfig| {
                        let motif = Arc::clone(&motif);
                        let random_slices = Arc::clone(&random_slices_for_batch);
                        async move {
                            MotifsBatch::new(motif, Arc::clone(&random_slices), rc.common_training_args.batch_size as i64, device_idx, cache_area, ctx).await
                        }.boxed()
                    },
                    move |rc:&TrainMotifsNetsRunConfig, vs: &VarStore| {
                        let params = match &rc.net {
                            MotifNetTypeArgs::ConvNet(net) => MotifNetParams::new_conv(motif_width, net.num_layers, net.step, net.channels, net.kernel),
                            MotifNetTypeArgs::LinNet(net) => MotifNetParams::new_lin(motif_width, net.num_layers, net.step),
                            MotifNetTypeArgs::CosineNet(net) => MotifNetParams::new_cosine(motif_width, net.step, net.channels, net.kernel),
                        };
                        Ok((params.motif_net()?.build_tch_module(&vs.root())?, params))
                    },
                    opt_config.clone(), //TODO clone
                    train_motifs_nodes_run_configs.clone(), //TODO clone
                    |pred_y: &Tensor, target: &Tensor| {

                        // let loss = pred_y.binary_cross_entropy::<Tensor>(&target.totype(Float), None, Reduction::Mean).set_requires_grad(true);
                        let loss = pred_y.f_binary_cross_entropy_with_logits::<Tensor>(&target.totype(Float), None, None, Reduction::Mean)?.f_set_requires_grad(true)?;
                        debug!("TrainMotifNets::run, loss_fn: pred_y ={pred_y} target={target} loss={loss}");

                        Ok(loss)
                    },
                    move |rc:&TrainMotifsNetsRunConfig, vs: &VarStore|{
                        AdamW::default().amsgrad(true).build(&vs, rc.common_training_args.lr).map_err(|e|e.into())
                    },
                    |t:ThoBatch<Slice>| {
                        t
                    },
                    |rc:&TrainMotifsNetsRunConfig|{
                        rc.common_training_args.cycles
                    },
                    "motifs".to_string()
                ));

                let result = Arc::into_inner(train_net.run(
                    Arc::new(DevicesWrap::wrap(TrainNetInput::new(SynKey::new([mono_key!("", "empty")])), CacheArea::default())),
                    ctx.branch(Some("train_motif_net"), false, false)
                ).await?).unwrap().take_data(0).await?;

                debug!("TrainMotifNets::run, done: motif_id_rate={motif_id_rate:?} min_loss={}", result.min_loss);

                store.get_motif_meta(motif_id_rate.0)
                .map(|(channel, find_motifs_run_id)| {
                    TrainMotifNetResult::new(
                        result.min_loss,
                        channel as u32,
                        motif_id_rate.0,
                        result.var_store,
                        *select_motifs_run_id,
                        result.node_params,
                        result.run_config,
                    )
                })
            }

        });

        debug!("TrainMotifNets::run: futures.len={}", futures.len());


        let results = collect_futures_with_progress_i(
            futures,
            motifs_rates_selected_motifs_run_ids_len,
            crate::buffer_limits::TRAIN_MOTIF_NETS_RUN_BUFFER_UNORDERED_LIMIT,
            StreamBuffering::Unordered,
            "train motif",
        ).await;
        //gpt TODO!! collect_futures_with_progress drops errors; results.len() may be smaller than the number of selected motifs.


        Ok(Arc::new(DevicesWrap::wrap(TrainMotifsNetsResult::new(select_motifs_run_ids, results), CacheArea::default())))
    }

}

#[derive(Debug)]
struct DataWithLabels {
    slices:ThoBatch<Slice>,
    labels: ThoTensor<Labels, SynKey>
}

impl DataWithLabels {
    pub fn new(slices: ThoBatch<Slice>, labels: ThoTensor<Labels, SynKey>) -> Self {
        Self { slices, labels }
    }
}

#[async_trait]
impl CopyToDevice for DataWithLabels {

    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
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
    type M = CpuOrGpu<Self>;

    fn size(&self) -> Self::Metrics {
        self.slices.size().try_into().unwrap()//TODO unwrap
    }

    fn device(&self) -> usize {
        self.slices.device_idx()
    }
}

#[derive(Debug)]
pub struct MotifsBatch<C> {
    motif_id: DataId,
    batch_size: u32,

    train_motifs:Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>,
    valid_motifs:Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>,
    
    random_slices:Arc<DevicesWrap<ThoTensor<MotifSlices, DataId>>>,
    motif_meta:MotifMeta,
    device_idx:usize,
    ctx: Mutex<C>,
}

impl<C> MotifsBatch<C>
where
    C: RunCtx,
{

    pub async fn new(motif: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>>>, random_slices:Arc<DevicesWrap<ThoTensor<MotifSlices, DataId>>>, batch_size:i64, device_idx:usize, cache_area:u64, mut ctx: C) -> Result<Self, ThoError> {
        debug!("TrainMotifNet::run: motif={motif:?}");


        let ctx_for_split = ctx.branch(Some("motifs_batch_split"), false, false);
        with_data!(ctx_for_split, motif; |_ctx, motif:&DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>|{
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
            let train_motifs = ThoTensor::new(SynKey::new([mono_key!("train_motifs", "split")]), train_motifs)?;
            let valid_motifs = ThoTensor::new(SynKey::new([mono_key!("valid_motifs", "split")]), valid_motifs)?;

            Ok::<_, ThoError>(Self {

                motif_id: motif.id().clone(),
                train_motifs:  Arc::new(DevicesWrap::wrap(train_motifs, cache_area.into())),
                valid_motifs: Arc::new(DevicesWrap::wrap(valid_motifs, cache_area.into())),
                random_slices,
                motif_meta,
                batch_size: batch_size as u32,
                device_idx,
                ctx: Mutex::new(ctx)
            })
        })
    }

    async fn make_batch(&self, motifs: Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>, motif_id:&DataId) -> Result<(ThoBatch<Slice>, Tensor), ThoError> {

        let random_slices = Arc::clone(&self.random_slices);

        let num_slices = (self.batch_size / 2) as i64;

        with_data!({
            let mut guard = thoosa_shared::with_perf_log(
                || self.ctx.try_lock_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
                "TrainMotifNets/make_batch/lock_ctx",
            );
            guard.branch(Some("motifs_batch_make"), false, false)
        }, motifs, random_slices; |_ctx, motifs:&ThoTensor<MotifSlices, SynKey>, random_slices:&ThoTensor<MotifSlices, DataId>|{

            let motifs = make_batch_with(motifs.t_ref(), num_slices, MixupJitterCfg { do_mixup: true, do_jitter: true, alpha: 0.4, sigma: 0.0 })?;//TODO config
            let random = make_batch_with(random_slices.t_ref(), num_slices, MixupJitterCfg { do_mixup: true, do_jitter: true, alpha: 0.4, sigma: 0.0 })?;

            #[allow(unexpected_cfgs)]
            #[cfg(tho_debug_plots)]
            {
                static COUNTER:AtomicU32 = AtomicU32::new(0);

                plot_t(&motifs.as_ref(), &format!("batch_motif_{motif_id}__{}", COUNTER.fetch_add(1, Relaxed)), false)?;
                plot_t(&motifs.as_ref(), &format!("batch_random_motif_in_batch_{motif_id}__{}", COUNTER.fetch_add(1, Relaxed)), false)?;
            }


            let (x, _lengths, y) = mix_x_binary_with_labels(&motifs, None, &random, None)?;

            debug!("MotifsBatch::train_batch: motifs={:?} random={:?}", motifs.device(), random.device());
            debug!("MotifsBatch::train_batch: x={:?} y={:?}", x.device(), y.device());

            Ok::<_, ThoError>((x.into(), y))
        })
    }
}

#[async_trait]
impl<C> AsyncDataBatch<Slice, MotifProbability> for MotifsBatch<C>
where
    C: RunCtx,
{
    async fn train_batch_stream(&self) -> BoxStream<'_, Result<(ThoBatch<Slice>, Option<BatchLengths>, ThoBatch<MotifProbability>), ThoError>> {
        futures_util::stream::once( self.make_batch(Arc::clone(&self.train_motifs), &self.motif_id))//TODO this corresponds to number of resources in label training , replace with configurable val?
        .map(|fut_res|{
            fut_res.map(|v|{
                (v.0, None,  v.1.into())
            })
        })
        .boxed()
    }

    async fn valid_batch_stream(&self) -> BoxStream<'_, Result<(ThoBatch<Slice>, Option<BatchLengths>, ThoBatch<MotifProbability>), ThoError>> {
        futures_util::stream::once( self.make_batch(Arc::clone(&self.valid_motifs), &self.motif_id))
            .map(|fut_res|{
                fut_res.map(|v|{
                    (v.0, None, v.1.into())
                })
            })
            .boxed()
    }
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Arc;
    use log::debug;
    use log::LevelFilter::{Debug, Trace};
    use parking_lot::RwLock;
    use tch::{Device, IndexOp};
    use tch::Device::Cuda;
    use tch::nn::VarStore;
    use tokio::fs;
    use thoosa_shared::runner::common::cachable::DevicesWrap;
    use thoosa_shared::runner::cache_areas::CacheArea;
    use thoosa_shared::runner::common::mock_run_ctx;
    use thoosa_shared::runner::common::{PermitsCtx, RunCtx};
    use thoosa_shared::runner::common::runnable::AsyncStage;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_common::dims::{Batch, MotifProbability, Slice};
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::tch_test_utils;
    use thoosa_common::train_nets::{CommonTrainingRunConfigFields, MotifConvNetArgsFields, MotifCosineNetArgsFields, MotifNetParams, MotifNetTypeArgsFields, OptConfig, ThoModuleWithStore, TrainMotifsNetsResult, TrainMotifsNetsRunConfigFields};
    use thoosa_common::utils::{plot_t, reset_print_options, test_net};
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::{OptRunConfigs, RunId};
    use thoosa_shared::{TensorIdSliceIdx, ThoError};
    use thoosa_shared::traits::Named;
    use thoosa_shared::hyperparameter::Hyperparameter;
    use thoosa_shared::val::{NamedRange, NamedVal, Val};
    use thoosa_store::MockStoreBuilder;
    use thoosa_with_data::with_data;
    use thoosa_common::tch_test_utils::resource_tensor;
    use thoosa_shared::test_utils::MotifKind;
    use crate::tasks::cross_dims::MotifSlices;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use crate::tasks::train_motif_nets::{TrainMotifNetInput, TrainMotifNets};

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_train_motif_nets() -> Result<(), ThoError> {
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
            ], 100, 10.0)?;

            a.0.insert(r.clone(), t);

            //0 channel - the only motif
            a.1.extend(intervals[0][0].iter().map(|interval|TensorIdSliceIdx(r.id(), interval.start as u32)));

            Ok::<_, ThoError>(a)
        })?;


       let resources_id_map = resources.iter().fold(HashMap::new(), |mut a, v|{
           a.insert(v.id().clone(), v.clone());
           a
       });

        let motif_id = 10u32;
        let mut store = Arc::new(
            MockStoreBuilder::new()
                .with_list_run_config_values(
                    find_motifs_run_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_id), 50, 100, None, None).to_properties_iter(None).collect()),
                )
                .with_list_motif_slices(motif_id, None, None, None, Ok((0i64, find_motifs_run_id, motif_intervals, resources_id_map.clone()

                )))
                .with_list_random_motif_slices(motif_id, motif_width as u32, None, Ok((0i64, find_motifs_run_id, (0..10).flat_map(|i|vec![
                    TensorIdSliceIdx(i, 100),
                    TensorIdSliceIdx(i, 300),
                    TensorIdSliceIdx(i, 500)
                ]).collect(), resources_id_map)))

                .with_list_selected_motifs([
                    (select_motifs_run_id, Ok(vec![vec![(motif_id, 1.0f64)]]))
                ])
                .with_get_motif_meta( [(motif_id, Ok((0usize, find_motifs_run_id)))])

                .build());

        // let run_configs = cos_net_run_configs();
        let run_configs = cosine_net_run_configs();

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

        let input = Arc::new(DevicesWrap::wrap(TrainMotifNetInput::new(vec![select_motifs_run_id], Arc::new(OptConfig::new(100, 10))), CacheArea::default()));

        let mut run_ctx = mock_run_ctx::default(1);
        let run_ctx_branch = run_ctx.branch(Some("train_motif_nets_run"), false, false);
        let r:Arc<DevicesWrap<TrainMotifsNetsResult>> = train_motif_net.run(input, run_ctx_branch).await?;

        fn run_test_net(params:&MotifNetParams, vs:&VarStore)->Result<(), ThoError>{
            let net = params.motif_net()?.build_tch_module::<MotifSlices, MotifProbability>(&vs.root())?;

            let (t, intervals) = tch_test_utils::resource_tensor(1000usize, &[
                vec![(2, 100, MotifKind::Pulse)],
            ], 100, 10.0)?;

            debug!("intervals={intervals:?} t.size={:?}", t.size());
            plot_t(&t, "channel_0", true)?;

            let output = net.tho_forward(&t.to_device(Cuda(0)).i((0, ..)).unfold(0, 100, 1).into(), None, false)?;

            debug!("output.t=\n{}", output.t.view((1,-1)));

            plot_t(&output.t.view((1,-1)), "applied 0", true)?;

            Ok(())
        }

        with_data!(run_ctx.branch(Some("train_motif_nets_assert"), false, false), r; |_ctx, r:&TrainMotifsNetsResult|{

            debug!("test_num_layers_opt: result={r:?}");

            assert!(r.train_motif_nets_results[0].loss < 0.69);

            run_test_net(&r.train_motif_nets_results[0].params, &r.train_motif_nets_results[0].vs)?;

            Ok::<_, ThoError>(())
        })?;

        Ok(())
    }

    fn cos_net_run_configs() -> OptRunConfigs<Vec<NamedVal<TrainMotifsNetsRunConfigFields>>, Vec<NamedRange<TrainMotifsNetsRunConfigFields>>, Vec<Hyperparameter<TrainMotifsNetsRunConfigFields>>, TrainMotifsNetsRunConfigFields> {
        thoosa_shared::run_configs::OptRunConfigs::new(
            thoosa_shared::run_configs::GridRunConfigs::new(
                thoosa_shared::run_configs::ConstConfigs::new(vec![
                    NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Cycles), Val::Int(300)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::BatchSize), Val::Int(10)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Patience), Val::Int(10)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::MinDelta), Val::Real(1e-4)),
                    // NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), Val::Real(0.0001)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::Net(MotifNetTypeArgsFields::ConvNet(MotifConvNetArgsFields::Step)), Val::Int(1)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::Net(MotifNetTypeArgsFields::ConvNet(MotifConvNetArgsFields::Kernel)), Val::Int(100)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::Net(MotifNetTypeArgsFields::ConvNet(MotifConvNetArgsFields::NumLayers)), Val::Int(8)),
                ]),
                vec![
                    // NamedRange::new(TrainMotifsNetsRunConfigFields::NumLayers, Val::Int(0)..Val::Int(10), (Val::Int(10), 0))
                ],
            ),
            vec![
                // Hyperparameter::UniformInt{name: TrainMotifsNetsRunConfigFields::NumLayers, lower: 6, upper: 8},
                Hyperparameter::UniformFloat { name: TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), lower: 0.00001, upper: 0.0001 },
                Hyperparameter::UniformInt { name: TrainMotifsNetsRunConfigFields::Net(MotifNetTypeArgsFields::ConvNet(MotifConvNetArgsFields::Channels)), lower: 8, upper: 32 }
            ],
        )
    }

    fn cosine_net_run_configs() -> OptRunConfigs<Vec<NamedVal<TrainMotifsNetsRunConfigFields>>, Vec<NamedRange<TrainMotifsNetsRunConfigFields>>, Vec<Hyperparameter<TrainMotifsNetsRunConfigFields>>, TrainMotifsNetsRunConfigFields> {
        thoosa_shared::run_configs::OptRunConfigs::new(
            thoosa_shared::run_configs::GridRunConfigs::new(
                thoosa_shared::run_configs::ConstConfigs::new(vec![
                    NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Cycles), Val::Int(300)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::BatchSize), Val::Int(10)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Patience), Val::Int(10)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::MinDelta), Val::Real(1e-4)),
                    // NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), Val::Real(0.0001)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::Net(MotifNetTypeArgsFields::CosineNet(MotifCosineNetArgsFields::Step)), Val::Int(1)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::Net(MotifNetTypeArgsFields::CosineNet(MotifCosineNetArgsFields::Kernel)), Val::Int(100)),
                ]),
                vec![
                    // NamedRange::new(TrainMotifsNetsRunConfigFields::NumLayers, Val::Int(0)..Val::Int(10), (Val::Int(10), 0))
                ],
            ),
            vec![
                // Hyperparameter::UniformInt{name: TrainMotifsNetsRunConfigFields::NumLayers, lower: 6, upper: 8},
                Hyperparameter::UniformFloat { name: TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), lower: 0.00001, upper: 0.0001 },
                Hyperparameter::UniformInt { name: TrainMotifsNetsRunConfigFields::Net(MotifNetTypeArgsFields::CosineNet(MotifCosineNetArgsFields::Channels)), lower: 2, upper: 32 }
            ],
        )
    }

}
