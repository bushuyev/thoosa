#![allow(unexpected_cfgs)]

use async_trait::async_trait;
use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use futures_util::{FutureExt, StreamExt};
use log::{debug, error, info, trace, warn};
use std::collections::{HashMap, HashSet};
use std::future::ready;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use parking_lot::{Mutex, RwLock};
use plotters::prelude::BLACK;
use plotters::style::RED;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tch::Device::{Cpu, Cuda};
use tch::Kind::Float;
use tch::kind::FLOAT_CUDA;
use tch::nn::{AdamW, Optimizer, OptimizerConfig, VarStore};
use tch::{IndexOp, Reduction, Tensor};
use thoosa_common::ipc::codec::{IpcCodec, JsonIpcCodec};
use thoosa_common::ipc::cuda_wokrer::{AllTensors, InitFn, WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use thoosa_common::ipc::json_ser::SyncStageIpcWrapper;
use thoosa_common::ipc::proxy::{DynSyncStageEnum, DynSyncStageWithStateEnum, ProxyStage};
use thoosa_common::ipc::wraps::{WorkerWrapper, WorkerWrapperWithState};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap, SelectRef, ThoDataRef};
use thoosa_common::runner::oom_handler::{NoContexts, NoInputAdjuster, NoRunCtxAdjuster, NoTaskAdjuster, OOMHandler};
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_shared::traits::MaxParallelCtx;

use mockall_double::double;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;
use crate::stages::cross_loader::ResourcesConfigInput;
use crate::stages::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta, RandomSlicesLoader, SYNC_LOAD_MOTIF_RESOURCES_ENDPOINT, SYNC_LOAD_RANDOM_SLICES_ENDPOINT, SYNC_MOTIF_LOADER_ENDPOINT, SlicesLoaderInput, sync_load_motif_resources_endpoint_init, sync_load_random_slices_endpoint_init, sync_motif_loader_endpoint_init};
use crate::stages::motifs_batch_builder::{MixupJitterCfg, make_batch_with};
use crate::stages::stage_ids::TRAIN_MOTIF_NETS;
use crate::stages::stft::{Stft, SyncStft, SyncStftSelector};
use crate::stages::tensor_loader::RawTensorLoader;
use crate::stages::train_label_nets::{SYNC_LABEL_BATCH_ENDPOINT, sync_label_batch_endpoint_init};
use crate::stages::train_net::{TrainNet, TrainNetInput, TrainStepFactory};
use crate::stages::train_step::{TrainStep, TrainStepXFn};
use thoosa_common::batches::{AsyncDataBatch, BatchLengths};
use thoosa_common::data_with_extra::DataWithExtra;
use thoosa_common::debug_plot;
use thoosa_common::dims::ChannelsData;
use thoosa_common::dims::{Batch, MotifProbability, MotifSlices, Slice, ThoBatch};
use thoosa_common::ipc::worker_msg::{WorkerInput, WorkerOutput};
use thoosa_common::my_tch::MotifInitMode;
use thoosa_common::parzen::optimizer::CompletedTrial;
use thoosa_common::runs::{Empty, SavableResult};
use thoosa_common::stages_seq::ABStage;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::tho_varstore::{ModConfig, ThoTrainModule};
use thoosa_common::train_nets::{MotifNetParams, MotifNetTypeArgs, MotifNetTypeArgsFields, MotifNetTypeParams, MotifsBy, OptConfig, StftRunConfig, TrainMotifNetResult, TrainMotifNetRunConfig, TrainMotifNetRunConfigFields, lin_motif_node};
use thoosa_common::utils::{DummyListener, mix_x_binary_with_labels, plot_series_with_lanes, plot_spectrogram, to_vec_f32};
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_run_config_args::OptRunConfigSource;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::resource::Resource;
use thoosa_shared::run_configs::{OptRunConfigs, RunConfig, RunConfigId};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, AsyncStage, Data, DataId, DynSyncStageBox, OutId, RunCtx, Stage, SyncStage, WithId};
use thoosa_shared::val::{NamedRange, NamedVal, Val};
use thoosa_shared::{CopyToDevice, FuturesOrdering, ThoError, collect_futures_with_progress, mono_key, perf_log, sync_read_lock, val};
use thoosa_store::Store;

pub(crate) const TRAIN_MOTIF_STEP_ENDPOINT: &str = "TrainMotifStep";

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct TrainMotifStepIpcInit {
    params: MotifNetParams,
    motif: ThoTensor<MotifSlices, SynKey>,
    run_config: TrainMotifNetRunConfig,
    stft_config: Option<StftRunConfig>,
    tho_device_idx: usize,
    vs_path: PathBuf,
}

fn train_motif_x_f(batch: ThoBatch<Slice>) -> ThoBatch<Slice> {
    batch
}

fn train_motif_loss(pred_y: &Tensor, target: &Tensor) -> Result<Tensor, ThoError> {
    pred_y.f_binary_cross_entropy_with_logits::<Tensor>(&target.f_totype(Float)?, None, None, Reduction::Mean).map_err(ThoError::from)
}

type SyncTrainMotifStep<C> = TrainStep<TrainStepXFn<Slice>, Slice, MotifProbability, C>;

fn train_motif_step_params(run_config: &TrainMotifNetRunConfig, motif_width: u32) -> Result<(MotifNetParams, Option<StftRunConfig>), ThoError> {
    match &run_config.net {
        MotifNetTypeArgs::ConvNet(net) => Ok((MotifNetParams::new_conv(motif_width, net.conv_layers, net.head_layers, net.step, net.channels, net.kernel), None)),
        MotifNetTypeArgs::LinNet(net) => Ok((MotifNetParams::new_lin(motif_width, net.num_layers, net.step), None)),
        MotifNetTypeArgs::CosineNet(net) => Ok((MotifNetParams::new_cosine(motif_width, net.a, net.b, net.init_mode.clone()), None)),
        MotifNetTypeArgs::Cosine2DNet(args) => {
            args.stft_config.validate()?;
            let n_fft = args.stft_config.n_fft;
            let hop_length = args.stft_config.hop_length.unwrap_or(n_fft / 4);
            let motif_width_i64 = i64::from(motif_width);
            if hop_length <= 0 || motif_width_i64 < n_fft {
                return Err(ThoError::new(format!("TrainMotifStepFactory invalid STFT dimensions: motif_width={motif_width} n_fft={n_fft} hop_length={hop_length}")));
            }
            let kw = u32::try_from((motif_width_i64 - n_fft) / hop_length + 1).map_err(|error| ThoError::new(format!("TrainMotifStepFactory invalid STFT time dimension: {error}")))?;
            let kh_i64 = if args.stft_config.onesided { n_fft / 2 + 1 } else { n_fft };
            let kh = u32::try_from(kh_i64).map_err(|error| ThoError::new(format!("TrainMotifStepFactory invalid STFT frequency dimension: {error}")))?;
            Ok((MotifNetParams::new_cosine_2d(kw, kh, args.a, args.b, args.init_mode.clone(), motif_width), Some(args.stft_config.clone())))
        }
    }
}

impl WorkerStageInput for TrainMotifStepIpcInit {
}

#[typetag::serde(name = "TrainMotifStepIpcInit")]
impl WorkerInput for TrainMotifStepIpcInit {}

#[typetag::serde(name = "TrainMotifStepInput")]
impl WorkerInput for crate::stages::train_step::TrainStepInput<Slice, MotifProbability> {}

fn train_motif_step_endpoint_init<'a>(init: Box<dyn WorkerInput>, ws: Arc<RwLock<Box<dyn Workspace>>>, all_cuda_tensors: &'a AllTensors, tho_device:&'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let init = *(init as Box<dyn std::any::Any>).downcast::<TrainMotifStepIpcInit>().map_err(|_| ThoError::new("TrainMotifStep endpoint received invalid initializer"))?;
        let motif = ThoTensor::ref_to_tensor(all_cuda_tensors, init.motif)?;
        let cosine_motif = match &init.params.net_type {
            MotifNetTypeParams::CosineNetParams(_) => Some(Arc::new(copy_motif(&motif)?)),
            MotifNetTypeParams::CosineNet2DParams(_) => {
                let stft_config = init.stft_config.clone().ok_or_else(|| ThoError::new("TrainMotifStepIpcInit missing STFT config for Cosine2DNet"))?;
                let stft = SyncStft::<RunCtxImpl, MotifSlices, MotifSlices>::new(SynKey::new([mono_key!("TrainMotifStepIpc", "stft")]), stft_config)?;
                let motif_specs = stft.build_spectrogram(motif.t_ref())?;
                let mut copied = motif_specs.f_zeros_like()?;
                copied.f_copy_(&motif_specs)?;
                Some(Arc::new(ThoTensor::new(motif.id().clone(), copied)?))
            }
            _ => None,
        };
        let params_for_module = init.params.clone();
        let cosine_for_module = cosine_motif.clone();
        let learning_rate = init.run_config.common_training_args.lr;
        let x_f: Arc<TrainStepXFn<Slice>> = Arc::new(Box::new(train_motif_x_f));
        let (train_step, _): (SyncTrainMotifStep<WokerRunCtx>, ()) = TrainStep::new_local(
            init.tho_device_idx.into(),
            move |_run_config, device| async move {
                let config = ModConfig::new(
                    AdamW::default().amsgrad(true).into(),
                    move |var_store| params_for_module.motif_net(cosine_for_module.clone())?.build_tch_module::<Batch<Slice>, Batch<MotifProbability>>(&var_store.root()),
                    learning_rate,
                );
                Ok((ThoTrainModule::new_on_device(config, device)?, ()))
            },
            x_f,
            train_motif_loss,
            init.run_config,
            init.vs_path,
        ).await?;
        // Shared stateful dispatch leaves trainable motif-net state in the worker.
        Ok(Box::new(WorkerWrapperWithState::<JsonIpcCodec, _, _, _>::new(Arc::new(train_step), PhantomData)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}

#[derive(Debug)]
struct TrainMotifStepFactory<C> {
    motif: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>,
    motif_width: u32,
    pd: PhantomData<C>,
}

impl<C> TrainMotifStepFactory<C> {
    fn new(motif: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>, motif_width: u32) -> Self {
        Self { motif, motif_width, pd: PhantomData }
    }
}

#[async_trait]
impl<C> TrainStepFactory<Slice, MotifProbability, C, MotifNetParams, TrainMotifNetRunConfig> for TrainMotifStepFactory<C>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    async fn new(&self, run_config: TrainMotifNetRunConfig, mut ctx: C, vs_path: PathBuf) -> Result<(DynSyncStageWithStateEnum<SyncTrainMotifStep<C>>, MotifNetParams), ThoError> {
        let (node_params, stft_config) = train_motif_step_params(&run_config, self.motif_width)?;
        if GlobalCtx::global().is_ipc() {
            let motif = track_await(self.motif.borrow_to(ctx.tho_device_idx(), None::<SyncStftSelector>)).await?;
            let motif = motif.as_ref().data();
            let motif = ThoTensor::new_ref(
                motif.id().clone(),
                motif.ref_key().ok_or_else(|| ThoError::new(format!("TrainMotifStepFactory expected motif worker reference, got {motif:?}")))?,
                motif.size(),
                motif.kind(),
                motif.selector(),
                motif.tho_device().tho_device_idx(),
            );
            static TRAIN_MOTIF_STEP_ID: AtomicU32 = AtomicU32::new(0);
            let stage_id = TRAIN_MOTIF_STEP_ID.fetch_add(1, Ordering::AcqRel);
            let init = TrainMotifStepIpcInit { params: node_params.clone(), motif, run_config, stft_config, tho_device_idx: ctx.tho_device_idx(), vs_path };
            let input_cloner = Arc::new(|input: &crate::stages::train_step::TrainStepInput<Slice, MotifProbability>| -> Result<Box<dyn WorkerInput>, ThoError> { Ok(Box::new(input.clone_refs_for_worker()?)) });
            Ok((DynSyncStageWithStateEnum::Proxy { stage: Arc::new(ProxyStage::proxy_with_input_cloner(TRAIN_MOTIF_STEP_ENDPOINT, stage_id, ctx.tho_device_idx(), Box::new(init), input_cloner).await?), init: () }, node_params))
        } else {
            let cosine_motif = match &run_config.net {
                MotifNetTypeArgs::CosineNet(_) => {
                    let motif_ref = track_await(self.motif.borrow_to(ctx.tho_device_idx(), None::<SyncStftSelector>)).await?;
                    Some(Arc::new(copy_motif(motif_ref.as_ref().data())?))
                }
                MotifNetTypeArgs::Cosine2DNet(args) => {
                    let stft = Arc::new(RandomSlicesStft::new_for_dims(args.stft_config.clone())?);
                    let motif_input = {
                        let motif_ref = track_await(self.motif.borrow_to(ctx.tho_device_idx(), None::<SyncStftSelector>)).await?;
                        copy_motif(motif_ref.as_ref().data())
                    }?;
                    let motif_specs = track_await(run_stft_as_motif_slices(&stft, motif_input, ctx.branch(false), "train_motif_nets_motif_stft")).await?;
                    let motif_id = motif_specs.id().clone();
                    let cosine_motif = motif_specs.as_ref();
                    let mut copied = cosine_motif.f_zeros_like()?;
                    copied.f_copy_(cosine_motif)?;
                    Some(Arc::new(ThoTensor::new(motif_id, copied)?))
                }
                _ => None,
            };
            let params_for_module = node_params.clone();
            let cosine_for_module = cosine_motif.clone();

            let x_f: Arc<TrainStepXFn<Slice>> = Arc::new(Box::new(train_motif_x_f));
            let (train_step, _) = TrainStep::new_local(
                ctx.tho_device_idx().into(),
                move |run_config, device| async move {

                    let config = ModConfig::new(
                        run_config.common_training_args.tho_optimizer(),
                        move |var_store| params_for_module.motif_net(cosine_for_module.clone())?.build_tch_module::<Batch<Slice>, Batch<MotifProbability>>(&var_store.root()),
                        run_config.common_training_args.lr,
                    );
                    Ok((ThoTrainModule::new_on_device(config, device)?, ()))
                },
                x_f,
                train_motif_loss,
                run_config,
                vs_path,
            ).await?;
            Ok((DynSyncStageWithStateEnum::Local(Arc::new(train_step)), node_params))
        }
    }
}

pub fn train_motif_step_endpoint_inits() -> HashMap<String, InitFn> {
    HashMap::from([
        (
            TRAIN_MOTIF_STEP_ENDPOINT.to_string(),
            Box::new(train_motif_step_endpoint_init) as InitFn,
        ),
        (
            SYNC_MOTIF_BASTCH_ENDPOINT.to_string(),
            Box::new(sync_motif_batch_endpoint_init) as InitFn,
        ),
        (
            SPLIT_MOTIF_INTO_TRAIN_VALID_ENDPOINT.to_string(),
            Box::new(split_motif_into_train_valid_endpoint_init) as InitFn,
        ),
        (
            SYNC_LOAD_MOTIF_RESOURCES_ENDPOINT.to_string(),
            Box::new(sync_load_motif_resources_endpoint_init) as InitFn,
        ),
        (
            SYNC_LOAD_RANDOM_SLICES_ENDPOINT.to_string(),
            Box::new(sync_load_random_slices_endpoint_init) as InitFn,
        ),
        (
            SYNC_MOTIF_LOADER_ENDPOINT.to_string(),
            Box::new(sync_motif_loader_endpoint_init) as InitFn,
        ),
        (
            SYNC_LABEL_BATCH_ENDPOINT.to_string(),
            Box::new(sync_label_batch_endpoint_init) as InitFn,
        ),
    ])
}

type RandomSlicesStft<C> = Stft<C, MotifSlices, MotifSlices>;
type RandomSlicesLoaderWithStft<L, C> = ABStage<RandomSlicesLoader<L, C>, RandomSlicesStft<C>, C>;

#[derive(Debug)]
enum RandomSlicesStage<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    Raw(Arc<RandomSlicesLoader<L, C>>),
    WithStft(Arc<RandomSlicesLoaderWithStft<L, C>>),
}

impl<L, C> Stage for RandomSlicesStage<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = Arc<SlicesLoaderInput>;
    type Out = Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>;
}

impl<L, C> OutId for RandomSlicesStage<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        match self {
            RandomSlicesStage::Raw(stage) => stage.out_id(input_id),
            RandomSlicesStage::WithStft(stage) => stage.out_id(input_id),
        }
    }
}

impl<L, C> AsyncStage for RandomSlicesStage<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    async fn run(self: Arc<Self>, input: Self::In<'static>, ctx: Self::Ctx) -> Result<Self::Out, ThoError> {
        match &*self {
            RandomSlicesStage::Raw(stage) => track_await(Arc::clone(stage).run_cached(input, ctx)).await,
            RandomSlicesStage::WithStft(stage) => track_await(Arc::clone(stage).run_cached(input, ctx)).await,
        }
    }
}

#[derive(Debug, Default)]
struct CurrentRandomSlices {
    slices: Option<Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>>,
    uses: u32,
}

#[derive(Debug)]
pub struct TrainMotifNets<L, C>
where
    L: for <'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MaxParallelCtx + Adjustable + 'static
{
    id: SynKey,
    run_configs: Arc<OptRunConfigs<
        Vec<NamedVal<TrainMotifNetRunConfigFields>>,
        Vec<NamedRange<TrainMotifNetRunConfigFields>>,
        Vec<Hyperparameter<TrainMotifNetRunConfigFields>>,
        TrainMotifNetRunConfigFields,
    >>,
    motif_loader:Arc<MotifLoader<L,C>>,
    random_slices_loader:Arc<RandomSlicesLoader<L,C>>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<Box<dyn Store>>,
    epoch_size: u32,
    descr: Option<String>,
    pd:PhantomData<C>,

}


#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct TrainMotifNetInput{
    motifs_by: MotifsBy,
    opt_config:Arc<OptConfig>
}

impl Input for TrainMotifNetInput {}
impl TrainMotifNetInput {
    pub fn new(motifs_by: MotifsBy, opt_config:Arc<OptConfig>) -> Self {
        Self { motifs_by, opt_config }
    }
}

fn copy_motif(input: &ThoTensor<MotifSlices, SynKey>) -> Result<ThoTensor<MotifSlices, SynKey>, ThoError> {
    let mut copied = input.t_ref().f_zeros_like()?;
    copied.f_copy_(input.t_ref())?;
    ThoTensor::new(input.id().clone(), copied)
}

async fn run_stft_as_motif_slices<C>(
    stft: &Arc<RandomSlicesStft<C>>,
    input: ThoTensor<MotifSlices, SynKey>,
    mut ctx: C,
    _branch_name: &'static str,
) -> Result<ThoTensor<MotifSlices, SynKey>, ThoError>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    let device_idx = ctx.tho_device_idx();
    let out_id = input.id().clone();
    let specs = track_await(Arc::clone(&stft)
        .run(
            Arc::new(DevicesWrap::wrap(input)),
            ctx.branch(false),
        )).await?;
    let specs = Arc::into_inner(specs)
        .ok_or_else(|| ThoError::new("run_stft_as_motif_slices: stft output should not be shared"))?
        .take_data(device_idx)?;
    ThoTensor::new(out_id, specs.t())
}

impl <L, C> Stage for TrainMotifNets<L, C>
where
    L: for <'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MaxParallelCtx + Adjustable + 'static
{
    type Ctx = C;
    type In<'a> = TrainMotifNetInput;//MotifID
    type Out = Arc<TrainMotifNetsResult>;
}

impl <L, C> TrainMotifNets<L, C>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    L: for <'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx=C> + Cached + OutId + 'static,
{
    pub async fn new(
        loader:Arc<L>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        run_configs:  OptRunConfigs<
            Vec<NamedVal<TrainMotifNetRunConfigFields>>,
            Vec<NamedRange<TrainMotifNetRunConfigFields>>,
            Vec<Hyperparameter<TrainMotifNetRunConfigFields>>,
            TrainMotifNetRunConfigFields,
        >,
        epoch_size: u32,
        descr: Option<String>
    ) -> Result<Self, ThoError> {
        let motif_loader = MotifLoader::new(
            Arc::clone(&store),
            Arc::clone(&loader),
            ws
        ).await?;

        let random_slices_loader = Arc::new(RandomSlicesLoader::new(
            Arc::clone(&store),
            Arc::clone(&loader),
            ws
        ));

        Ok(Self {
            id: mono_key!(TRAIN_MOTIF_NETS, "TrainMotifNets").into(),
            motif_loader,
            random_slices_loader,
            run_configs: Arc::new(run_configs),
            ws: Arc::clone(&ws),
            store: Arc::clone(&store),
            epoch_size,
            descr,
            pd: PhantomData
        })
    }
}


async fn completed_train_motif_trials(store: &Arc<Box<dyn Store>>, select_motifs_run_config_ids: &[RunConfigId]) -> Result<Vec<CompletedTrial>, ThoError> {
    let mut train_parent_ids = Vec::new();
    let mut seen_parent_candidates = HashSet::new();
    for select_motifs_run_config_id in select_motifs_run_config_ids {
        for output_run_config_id in track_await(store.list_output_run_config_ids(*select_motifs_run_config_id)).await? {
            if seen_parent_candidates.insert(output_run_config_id) {
                train_parent_ids.push(output_run_config_id);
            }
        }
    }

    if train_parent_ids.is_empty() {
        return Ok(vec![]);
    }

    let train_parent_ids = track_await(store
        .list_run_configs_by_ids(&train_parent_ids)).await?
        .into_iter()
        .filter(|entry| entry.run_config_type == TrainMotifNetRunConfig::type_name() && entry.descr == "TrainMotifsNetsResult")
        .map(|entry| entry.id)
        .collect::<Vec<_>>();

    let mut child_ids = Vec::new();
    let mut seen_child_candidates = HashSet::new();
    for train_parent_id in train_parent_ids {
        for output_run_config_id in track_await(store.list_output_run_config_ids(train_parent_id)).await? {
            if seen_child_candidates.insert(output_run_config_id) {
                child_ids.push(output_run_config_id);
            }
        }
    }

    if child_ids.is_empty() {
        return Ok(vec![]);
    }

    let child_entries = track_await(store
        .list_run_configs_by_ids(&child_ids)).await?
        .into_iter()
        .filter(|entry| entry.run_config_type == TrainMotifNetRunConfig::type_name() && entry.descr == "TrainMotifNetResult")
        .collect::<Vec<_>>();

    let mut trials = Vec::new();
    for child_entry in child_entries {
        let values = track_await(store.list_run_config_values(child_entry.id)).await?;
        let Some(loss) = values.get("loss").and_then(val::val_to_f32) else {
            continue;
        };
        if loss.is_finite() {
            trials.push(CompletedTrial::new(loss, values));
        }
    }
    Ok(trials)
}

impl <L, C> AsyncStage for TrainMotifNets<L, C>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    L: for <'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx=C> + Cached + OutId + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainMotifNets::run"))]
    async fn run(self: Arc<Self>, input: TrainMotifNetInput, mut ctx: Self::Ctx) -> Result<Self::Out, ThoError> {

        let device_idx = ctx.tho_device_idx();

        let channels = perf_log!(sync_read_lock!(self.ws)).channels();

        debug!("TrainMotifNets::run: input={input:?}");

        let (select_motifs_run_config_ids, motifs_rates_selected_motifs_run_config_ids) = match input.motifs_by {

            MotifsBy::SelectMotifsRunIds { select_motifs_run_config_ids } => {

                let mut motifs_rates_selected_motifs_run_config_ids = Vec::new();
                for select_motifs_run_config_id in &select_motifs_run_config_ids {
                    let selected_motifs = track_await(self.store.list_selected_motifs(*select_motifs_run_config_id, channels)).await?;
                    debug!("TrainMotifNets::run: select_motifs_run_config_id={select_motifs_run_config_id} selected_motifs={selected_motifs:?}");
                    motifs_rates_selected_motifs_run_config_ids.extend(
                        selected_motifs
                            .into_iter()
                            .flat_map(|channels_motifs| channels_motifs.into_iter().map(move |motif_id_rate| (motif_id_rate, *select_motifs_run_config_id))),
                    );
                }
                (select_motifs_run_config_ids, motifs_rates_selected_motifs_run_config_ids)
            }
            MotifsBy::MotifId { motif_id, select_motifs_run_config_ids } => {

                let (channel, find_motifs_run_config_id) = track_await(self.store.get_motif_meta(motif_id)).await?;
                let select_motifs_run_config_ids = match select_motifs_run_config_ids {
                    Some(select_motifs_run_config_ids) => select_motifs_run_config_ids,
                    None => track_await(self.store.list_output_run_config_ids(find_motifs_run_config_id)).await?,
                };

                let mut motifs_rates_selected_motifs_run_config_ids = Vec::new();
                for select_motifs_run_config_id in &select_motifs_run_config_ids {
                    let selected_motifs = track_await(self.store.list_selected_motifs(*select_motifs_run_config_id, channels)).await?;
                    let motif_is_selected = selected_motifs
                        .get(channel)
                        .map(|channel_motifs| channel_motifs.iter().any(|(selected_motif_id, _)| *selected_motif_id == motif_id))
                        .unwrap_or(false);
                    if motif_is_selected {
                        motifs_rates_selected_motifs_run_config_ids.push(((motif_id, 1.0f64), *select_motifs_run_config_id));
                    }
                }
                (select_motifs_run_config_ids, motifs_rates_selected_motifs_run_config_ids)
            }
        };
        let motifs_rates_selected_motifs_run_config_ids_len = motifs_rates_selected_motifs_run_config_ids.len();
        debug!("TrainMotifNets::run: motifs_rates_selected_motifs_run_config_ids_len={motifs_rates_selected_motifs_run_config_ids_len} motifs_rates_selected_motifs_run_config_ids={motifs_rates_selected_motifs_run_config_ids:?}");

        let reusable_completed_trials = track_await(completed_train_motif_trials(&self.store, &select_motifs_run_config_ids)).await?;
        debug!("TrainMotifNets::run: reusable_completed_trials.len={}", reusable_completed_trials.len());

        let futures = motifs_rates_selected_motifs_run_config_ids.into_iter()
        .map(|(motif_id_rate, select_motifs_run_config_id)|{
            let mut ctx = ctx.branch(false);
            let mut ctx_run = ctx.branch(false);

            let motif_loader = Arc::clone(&self.motif_loader);
            let random_slices_loader = Arc::clone(&self.random_slices_loader);
            let opt_config = Arc::clone(&input.opt_config);
            let train_motifs_nodes_run_configs = Arc::clone(&self.run_configs);
            let store = Arc::clone(&self.store);
            let epoch_size = self.epoch_size;
            let reusable_completed_trials = reusable_completed_trials.clone();
            let descr = self.descr.clone();

            async move {

                let motif: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>> = track_await(motif_loader.run_cached(
                    Arc::new(MotifLoaderInput::new((motif_id_rate.0, "motif_id").into(), Some(vec![select_motifs_run_config_id]))),
                    ctx.branch(false),
                )).await?;

                let motif_for_net = Arc::clone(&motif);

                let motif_meta = track_await(motif.extra()).await?;
                let motif_width = motif_meta.window;

                let random_slices_loader_for_batch = Arc::clone(&random_slices_loader);

                debug!("TrainMotifNets::run: motif={motif:?}");

                let mut ctx_net = ctx.branch(false);
                let motif_batch = Arc::clone(&motif);


                let  ctx_net_0 = ctx_net.branch(false);
                let  ctx_net_1 = ctx_net.branch(false);


                let train_net = Arc::new(TrainNet::new(
                    move |mut ctx:C, rc:&TrainMotifNetRunConfig| {
                        match &rc.net {
                            MotifNetTypeArgs::Cosine2DNet(args) => {
                                debug!("TrainMotifNets::run: batch, rc={rc:?} motif={motif_batch:?}");

                                let motif = Arc::clone(&motif_batch);
                                let raw_random_slices_loader = Arc::clone(&random_slices_loader_for_batch);
                                let stft = Arc::new(RandomSlicesStft::new_for_dims(args.stft_config.clone()).unwrap());//TODO unwrap
                                let random_slices_loader: Arc<RandomSlicesStage<L, C>> = Arc::new(RandomSlicesStage::WithStft(Arc::new(ABStage::new(
                                    SynKey::new([mono_key!("RandomSlicesLoaderWithStft", "stage")]),
                                    raw_random_slices_loader,
                                    Arc::clone(&stft),
                                ))));
                                let motif_meta = motif_meta.clone();
                                async move {
                                    let stft_input = {
                                        let motif_ref = motif.borrow_to(ctx.tho_device_idx(), None::<SyncStftSelector>).await?;
                                        let motif = motif_ref.as_ref();
                                        copy_motif(motif.data())
                                    }?;

                                    let motif_specs = track_await(run_stft_as_motif_slices(
                                        &stft,
                                        stft_input,
                                        ctx.branch(false),
                                        "train_motif_nets_motif_stft",
                                    )).await?;

                                    let motif_specs: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>> = Arc::new(DevicesWrap::wrap(DataWithExtra::new(motif_specs, motif_meta)));

                                    debug!("TrainMotifNets::run: batch, rc.id={:?} motif_specs={motif_specs:?}", rc.id);

                                    track_await(MotifsBatch::new(motif_specs, Arc::clone(&random_slices_loader), rc.common_training_args.batch_size as i64, device_idx, ctx, epoch_size)).await
                                }.boxed()
                            }
                            MotifNetTypeArgs::CosineNet(args) => {
                                debug!("TrainMotifNets::run: batch, rc={rc:?}");

                                let motif = Arc::clone(&motif_batch);
                                let random_slices_loader: Arc<RandomSlicesStage<L, C>> = Arc::new(RandomSlicesStage::Raw(Arc::clone(&random_slices_loader_for_batch)));
                                async move {
                                    track_await(MotifsBatch::new(motif, Arc::clone(&random_slices_loader), rc.common_training_args.batch_size as i64, device_idx, ctx, epoch_size)).await
                                }.boxed()
                            }
                            _ => {
                                debug!("TrainMotifNets::run: batch, rc={rc:?}");

                                let motif = Arc::clone(&motif_batch);
                                let random_slices_loader: Arc<RandomSlicesStage<L, C>> = Arc::new(RandomSlicesStage::Raw(Arc::clone(&random_slices_loader_for_batch)));
                                async move {
                                    track_await(MotifsBatch::new(motif, Arc::clone(&random_slices_loader), rc.common_training_args.batch_size as i64, device_idx, ctx, epoch_size)).await
                                }.boxed()
                            }
                        }

                    },
                    Arc::new(TrainMotifStepFactory::<C>::new(motif_for_net, motif_width)),
                    opt_config.clone(), //TODO clone
                    train_motifs_nodes_run_configs.clone(), //TODO clone
                    |rc:&TrainMotifNetRunConfig|{
                        rc.common_training_args.epochs
                    },
                    1,
                    format!("motif_{}", motif_id_rate.0)
                )
                    .with_reusable_completed_trials(reusable_completed_trials));

                debug!("TrainMotifNets::run: train_net={train_net:?}");

                let train_net_output = track_await(train_net.run(
                    Arc::new(TrainNetInput::new()),

                    ctx_run
                )).await?;

                info!("TrainMotifNets::run, done: motif_id_rate={motif_id_rate:?} min_loss={}", train_net_output.min_loss);

                let (channel, _find_motifs_run_config_id) = track_await(store.get_motif_meta(motif_id_rate.0)).await?;
                Ok(TrainMotifNetResult::new(
                    train_net_output.min_loss,
                    channel as u32,
                    motif_id_rate.0,
                    train_net_output.var_store_path,
                    select_motifs_run_config_id,
                    train_net_output.node_params,
                    train_net_output.run_config,
                    descr
                ))
            }

        });

        debug!("TrainMotifNets::run: futures.len={}", futures.len());


        let results = track_await(collect_futures_with_progress(
            futures,
            motifs_rates_selected_motifs_run_config_ids_len,
            FuturesOrdering::Unordered,
            "train motif",
        )).await;

        debug!("TrainMotifNets::run: results={results:?}");

        Ok(Arc::new(TrainMotifNetsResult::new(select_motifs_run_config_ids, results)))
    }

}

pub(crate) const SYNC_MOTIF_BASTCH_ENDPOINT: &str = "SyncMotifBatch";

#[derive(Debug)]
pub struct SyncMotifBatchInput<'a> {
    id: SynKey,
    motifs: ThoDataRef<'a, ThoTensor<MotifSlices, SynKey>>,
    random_slices: ThoDataRef<'a, ThoTensor<MotifSlices, SynKey>>,
    num_slices: i64,
    descr: String,
}

impl Input for SyncMotifBatchInput<'_> {}

impl WithId for SyncMotifBatchInput<'_> {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug)]
struct SyncMotifBatchInputWrapper {
    id: SynKey,
    motifs: Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>,
    random_slices: Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>,
    num_slices: i64,
    descr: String,
}

impl SyncMotifBatchInputWrapper {
    fn new(id: SynKey, motifs: Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>, random_slices: Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>, num_slices: i64, descr: String) -> Self {
        Self { id, motifs, random_slices, num_slices, descr }
    }
}

impl thoosa_common::runner::common::cachable::BorrowRef for SyncMotifBatchInputWrapper {
    type Out<'a> = SyncMotifBatchInput<'a> where Self: 'a;

    fn borrow_to<'a>(&'a self, to_device_idx: usize) -> impl Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        async move {
            let motifs = track_await(thoosa_common::runner::common::cachable::BorrowRef::borrow_to(&self.motifs, to_device_idx)).await?;
            let random_slices = track_await(thoosa_common::runner::common::cachable::BorrowRef::borrow_to(&self.random_slices, to_device_idx)).await?;
            Ok(SyncMotifBatchInput { id: self.id.clone(), motifs, random_slices, num_slices: self.num_slices, descr: self.descr.clone() })
        }
    }

}

#[derive(Debug, Serialize, Deserialize)]
pub struct SyncMotifBatchOutput {
    id: SynKey,
    x: ThoTensor<Batch<Slice>, SynKey>,
    y: ThoTensor<Batch<MotifProbability>, SynKey>,
}

impl SyncMotifBatchOutput {
    fn into_parts(self) -> (ThoBatch<Slice>, ThoBatch<MotifProbability>) {
        (self.x.with_id(thoosa_shared::simple_id!("sync_motif_bastch_x")), self.y.with_id(thoosa_shared::simple_id!("sync_motif_bastch_y")))
    }
}

impl WithId for SyncMotifBatchOutput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for SyncMotifBatchOutput {}

impl WorkerStageOutput for SyncMotifBatchOutput {
    fn encode_from_worker(self, all_cuda_tensors: &AllTensors, tho_device: &ThoDevice) -> Result<Self, ThoError> {
        let Self { id, x, y } = self;
        // Convert tensor fields first so the wrapper can encode this output exactly once.
        let x = x.encode_from_worker(all_cuda_tensors, tho_device)?;
        let y = y.encode_from_worker(all_cuda_tensors, tho_device)?;
        Ok(Self { id, x, y })
    }
}

#[derive(Debug, Serialize)]
struct SyncMotifBatchIpcInputRef<'a> {
    id: &'a SynKey,
    motifs: &'a ThoTensor<MotifSlices, SynKey>,
    random_slices: &'a ThoTensor<MotifSlices, SynKey>,
    num_slices: i64,
    descr: &'a str,
}

impl<'a> SyncMotifBatchIpcInputRef<'a> {
    fn new(input: &'a SyncMotifBatchInput<'_>) -> Self {
        Self {
            id: &input.id,
            motifs: input.motifs.as_ref(),
            random_slices: input.random_slices.as_ref(),
            num_slices: input.num_slices,
            descr: input.descr.as_str(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct SyncMotifBatchIpcInput {
    id: SynKey,
    motifs: ThoTensor<MotifSlices, SynKey>,
    random_slices: ThoTensor<MotifSlices, SynKey>,
    num_slices: i64,
    descr: String,
}

impl Serialize for SyncMotifBatchInput<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SyncMotifBatchIpcInputRef::new(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SyncMotifBatchInput<'static> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let input = SyncMotifBatchIpcInput::deserialize(deserializer)?;
        Ok(Self { id: input.id, motifs: ThoDataRef::Owned(input.motifs), random_slices: ThoDataRef::Owned(input.random_slices), num_slices: input.num_slices, descr: input.descr })
    }
}

#[typetag::serde(name = "SyncMotifBatchInput")]
impl WorkerInput for SyncMotifBatchInput<'static> {}

impl WorkerStageInput for SyncMotifBatchInput<'_> {
    fn clone_refs(&self) -> Result<Box<dyn WorkerInput>, ThoError> {
        // Own cloned worker references so no borrowed batch lifetime crosses IPC.
        Ok(Box::new(SyncMotifBatchInput {
            id: self.id.clone(),
            motifs: ThoDataRef::Owned(self.motifs.as_ref().clone_ref()?),
            random_slices: ThoDataRef::Owned(self.random_slices.as_ref().clone_ref()?),
            num_slices: self.num_slices,
            descr: self.descr.clone(),
        }))
    }

    fn ref_to_tensor(&self, all_cuda_tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {
        Ok(Self {
            id: self.id.clone(),
            motifs: ThoDataRef::Owned(WorkerStageInput::ref_to_tensor(self.motifs.as_ref(), all_cuda_tensors, remove)?),
            random_slices: ThoDataRef::Owned(WorkerStageInput::ref_to_tensor(self.random_slices.as_ref(), all_cuda_tensors, remove)?),
            num_slices: self.num_slices,
            descr: self.descr.clone(),
        })
    }

}

#[derive(Debug)]
pub struct SyncMotifBatch<C>
where
    C:RunCtx {
    pb:PhantomData<C>
}

impl<C> SyncMotifBatch<C>
where
    C:RunCtx
{
    async fn new(tho_device_idx: usize) -> Result<DynSyncStageEnum<SyncMotifBatch<C>>, ThoError> {
        if GlobalCtx::global().is_ipc() {
            static STAGE_ID: AtomicU32 = AtomicU32::new(0);
            let stage_id = STAGE_ID.fetch_add(1, Ordering::AcqRel);
            // Install one typed-wrapper endpoint per proxy so worker tensor ownership stays isolated.
            Ok(DynSyncStageEnum::Proxy { stage: Arc::new(ProxyStage::proxy(SYNC_MOTIF_BASTCH_ENDPOINT, stage_id, tho_device_idx, Box::new(())).await?), init: () })
        } else {
            Ok(DynSyncStageEnum::Local(Arc::new(Self::new_local())))
        }
    }

    fn new_local() -> SyncMotifBatch<C> {
        SyncMotifBatch {
            pb: Default::default(),
        }
    }
}

impl <C> Adjustable for SyncMotifBatch<C>
where
    C:RunCtx
{}

impl <C> OutId for SyncMotifBatch<C>
where
    C:RunCtx
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

impl <C> Stage for SyncMotifBatch<C>
where
    C:RunCtx
{
    type Ctx = C;
    type In<'a> = SyncMotifBatchInput<'a>;
    type Out = SyncMotifBatchOutput;
}

impl <C> SyncStage for SyncMotifBatch<C>
where
    C:RunCtx
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SyncMotifBatch::do_run"))]
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        debug!("SyncMotifBatch::do_run: descr={} num_slices={}", input.descr, input.num_slices);

        let motifs = make_batch_with(input.motifs.t_ref(), input.num_slices, None)?;
        let random = make_batch_with(input.random_slices.t_ref(), input.num_slices, None)?;

        #[cfg(tho_debug_plots)]
        plot_spectrogram(&motifs.i((0, .., ..)), format!("/tmp/plots/spectrogram_motif_{}.png", input.id), 1200, 600)?;

        #[cfg(tho_debug_plots)]
        plot_spectrogram(&random.i((0, .., ..)), format!("/tmp/plots/spectrogram_random_{}.png", input.id), 1200, 600)?;

        #[cfg(tho_debug_plots)]
        {
            let motif_plot_label = format!("motifs_batch_make_batch_{}_{}", input.id, input.descr);
            let random_plot_label = format!("motifs_batch_make_batch_random_{}_{}", input.id, input.descr);
            debug_plot!(&motifs.as_ref(), motif_plot_label, plot_repeats=11)?;
            debug_plot!(&random.as_ref(), random_plot_label, plot_repeats=11)?;
        }

        let (x, _lengths, y) = mix_x_binary_with_labels(&motifs, None, &random, None)?;

        debug!("SyncMotifBatch::do_run: motifs={motifs:?} random={random:?}");
        debug!("SyncMotifBatch::do_run: x={x:?} y={y:?}");

        Ok(SyncMotifBatchOutput {
            id: input.id.clone(),
            x: ThoTensor::new(input.id.clone(), x)?,
            y: ThoTensor::new(input.id.clone(), y)?,
        })
    }
}

#[typetag::serde]
impl WorkerOutput for SyncMotifBatchOutput{}

pub(crate) fn sync_motif_batch_endpoint_init<'a>(init: Box<dyn WorkerInput>, ws: Arc<RwLock<Box<dyn Workspace>>>, all_cuda_tensors: &'a AllTensors, tho_device:&'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        // Stateless motif batching uses the same codec and dispatch path as every shared proxy.
        Ok(Box::new(WorkerWrapper::<JsonIpcCodec, _, _, _>::new(Arc::new(SyncMotifBatch::<WokerRunCtx>::new_local()), PhantomData)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}

pub(crate) const SPLIT_MOTIF_INTO_TRAIN_VALID_ENDPOINT: &str = "SplitMotifIntoTrainValid";

#[derive(Debug)]
enum SplitMotifTensorRef<'a> {
    WithExtra(ThoDataRef<'a, DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>),
    Plain(ThoDataRef<'a, ThoTensor<MotifSlices, SynKey>>),
}

impl SplitMotifTensorRef<'_> {
    fn motif(&self) -> &ThoTensor<MotifSlices, SynKey> {
        match self {
            Self::WithExtra(motif) => motif.as_ref().data(),
            Self::Plain(motif) => motif.as_ref(),
        }
    }
}

#[derive(Debug)]
pub struct SplitMotifIntoTrainValidInput<'a> {
    id: SynKey,
    motif: SplitMotifTensorRef<'a>,
}

impl Input for SplitMotifIntoTrainValidInput<'_> {}

impl WithId for SplitMotifIntoTrainValidInput<'_> {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug)]
struct SplitMotifIntoTrainValidInputWrapper {
    id: SynKey,
    motif: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>,
}

impl SplitMotifIntoTrainValidInputWrapper {
    fn new(motif: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>) -> Self {
        Self { id: motif.id().clone(), motif }
    }
}

impl thoosa_common::runner::common::cachable::BorrowRef for SplitMotifIntoTrainValidInputWrapper {
    type Out<'a> = SplitMotifIntoTrainValidInput<'a> where Self: 'a;

    fn borrow_to<'a>(&'a self, to_device_idx: usize) -> impl Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        async move {
            let motif = track_await(thoosa_common::runner::common::cachable::BorrowRef::borrow_to(&self.motif, to_device_idx)).await?;
            Ok(SplitMotifIntoTrainValidInput { id: self.id.clone(), motif: SplitMotifTensorRef::WithExtra(motif) })
        }
    }

}

#[derive(Debug, Serialize, Deserialize)]
pub struct SplitMotifIntoTrainValidOutput {
    id: SynKey,
    train_motifs: ThoTensor<MotifSlices, SynKey>,
    valid_motifs: ThoTensor<MotifSlices, SynKey>,
    num_motif_slices: u32,
}

impl WithId for SplitMotifIntoTrainValidOutput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for SplitMotifIntoTrainValidOutput {}

impl WorkerStageOutput for SplitMotifIntoTrainValidOutput {
    fn encode_from_worker(self, all_cuda_tensors: &AllTensors, tho_device: &ThoDevice) -> Result<Self, ThoError> {
        let Self { id, train_motifs, valid_motifs, num_motif_slices } = self;
        // Convert tensor fields first so the wrapper can encode this output exactly once.
        let train_motifs = train_motifs.encode_from_worker(all_cuda_tensors, tho_device)?;
        let valid_motifs = valid_motifs.encode_from_worker(all_cuda_tensors, tho_device)?;
        Ok(Self { id, train_motifs, valid_motifs, num_motif_slices })
    }
}

#[derive(Debug, Serialize)]
struct SplitMotifIntoTrainValidIpcInputRef<'a> {
    id: &'a SynKey,
    motif: &'a ThoTensor<MotifSlices, SynKey>,
}

impl<'a> SplitMotifIntoTrainValidIpcInputRef<'a> {
    fn new(input: &'a SplitMotifIntoTrainValidInput<'_>) -> Self {
        Self { id: &input.id, motif: input.motif.motif() }
    }
}

#[derive(Debug, Deserialize)]
struct SplitMotifIntoTrainValidIpcInput {
    id: SynKey,
    motif: ThoTensor<MotifSlices, SynKey>,
}

impl Serialize for SplitMotifIntoTrainValidInput<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SplitMotifIntoTrainValidIpcInputRef::new(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SplitMotifIntoTrainValidInput<'static> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let input = SplitMotifIntoTrainValidIpcInput::deserialize(deserializer)?;
        Ok(Self { id: input.id, motif: SplitMotifTensorRef::Plain(ThoDataRef::Owned(input.motif)) })
    }
}

#[typetag::serde(name = "SplitMotifIntoTrainValidInput")]
impl WorkerInput for SplitMotifIntoTrainValidInput<'static> {}

impl WorkerStageInput for SplitMotifIntoTrainValidInput<'_> {
    fn clone_refs(&self) -> Result<Box<dyn WorkerInput>, ThoError> {
        // Flatten the metadata wrapper because this stage consumes only the motif tensor.
        Ok(Box::new(SplitMotifIntoTrainValidInput { id: self.id.clone(), motif: SplitMotifTensorRef::Plain(ThoDataRef::Owned(self.motif.motif().clone_ref()?)) }))
    }

    fn ref_to_tensor(&self, all_cuda_tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {
        Ok(Self { id: self.id.clone(), motif: SplitMotifTensorRef::Plain(ThoDataRef::Owned(WorkerStageInput::ref_to_tensor(self.motif.motif(), all_cuda_tensors, remove)?)) })
    }
}

#[derive(Debug)]
pub struct SplitMotifIntoTrainValid<C>
where
    C: RunCtx  + 'static
{
    pd:PhantomData<C>
}

impl <C> SplitMotifIntoTrainValid<C>
where
    C: RunCtx  + 'static
{
    async fn new(tho_device_idx: usize) -> Result<DynSyncStageEnum<SplitMotifIntoTrainValid<C>>, ThoError> {
        if GlobalCtx::global().is_ipc() {
            static STAGE_ID: AtomicU32 = AtomicU32::new(0);
            let stage_id = STAGE_ID.fetch_add(1, Ordering::AcqRel);
            // The typed proxy carries throttling separately from the borrowed tensor payload.
            Ok(DynSyncStageEnum::Proxy { stage: Arc::new(ProxyStage::proxy(SPLIT_MOTIF_INTO_TRAIN_VALID_ENDPOINT, stage_id, tho_device_idx, Box::new(())).await?), init: () })
        } else {
            Ok(DynSyncStageEnum::Local(Arc::new(SplitMotifIntoTrainValid::new_local())))
        }
    }

    pub fn new_local()->Self {
        SplitMotifIntoTrainValid{
            pd: Default::default(),
        }
    }
}

pub(crate) fn split_motif_into_train_valid_endpoint_init<'a>(init: Box<dyn WorkerInput>, ws: Arc<RwLock<Box<dyn Workspace>>>, all_cuda_tensors: &'a AllTensors, tho_device:&'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        // Decode borrowed motif input into worker-owned storage before running the shared stage.
        Ok(Box::new(WorkerWrapper::<JsonIpcCodec, _, _, _>::new(Arc::new(SplitMotifIntoTrainValid::<WokerRunCtx>::new_local()), PhantomData)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}


impl <C> Adjustable for SplitMotifIntoTrainValid<C>
where
    C: RunCtx  + 'static
{}

impl <C>OutId for SplitMotifIntoTrainValid<C>
where
    C: RunCtx  + 'static
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

impl <C> Stage for SplitMotifIntoTrainValid<C>
where
    C: RunCtx  + 'static
{
    type Ctx = C;
    type In<'a> = SplitMotifIntoTrainValidInput<'a>;
    type Out = SplitMotifIntoTrainValidOutput;
}

impl <C> SyncStage for SplitMotifIntoTrainValid<C>
where
    C: RunCtx  + 'static
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SplitMotifIntoTrainValid::do_run"))]
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let motif = input.motif.motif();
        let motif_size = motif.size();
        let num_motif_slices = *motif_size.first().ok_or_else(|| ThoError::new("SplitMotifIntoTrainValid: motif tensor has no dimensions"))?;
        if num_motif_slices <= 1 {
            return Err(ThoError::new(format!("SplitMotifIntoTrainValid: expected more than one motif slice, got {num_motif_slices}")));
        }
        if !matches!(motif_size.len(), 2 | 3) {
            return Err(ThoError::new(format!("SplitMotifIntoTrainValid: unsupported motif tensor rank={} size={motif_size:?}", motif_size.len())));
        }
        let num_validation_slices = ((num_motif_slices as f32 * 0.2) as i64).max(1);
        let split = num_motif_slices - num_validation_slices;
        debug!("SplitMotifIntoTrainValid::do_run: id={:?} num_motif_slices={num_motif_slices} num_validation_slices={num_validation_slices} split={split}", input.id);

        let motifs_t = motif.t_ref();
        debug_plot!(&motifs_t, "motifs_t{}", input.id)?;
        let train_motifs = motifs_t.f_narrow(0, 0, split)?;
        let valid_motifs = motifs_t.f_narrow(0, split, num_validation_slices)?;

        Ok(SplitMotifIntoTrainValidOutput {
            id: input.id.clone(),
            train_motifs: ThoTensor::new(SynKey::new([mono_key!("train_motifs", "split")]), train_motifs)?,
            valid_motifs: ThoTensor::new(SynKey::new([mono_key!("valid_motifs", "split")]), valid_motifs)?,
            num_motif_slices: u32::try_from(num_motif_slices).map_err(|error| ThoError::new(format!("SplitMotifIntoTrainValid: invalid motif slice count: {error}")))?,
        })
    }
}

#[typetag::serde]
impl WorkerOutput for SplitMotifIntoTrainValidOutput{}


#[derive(Debug)]
pub struct MotifsBatch<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    motif_id: SynKey,
    batch_size: u32,
    epoch_size: u32,
    train_motifs: Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>,
    valid_motifs: Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>,
    random_slices_loader: Arc<RandomSlicesStage<L, C>>,
    random_slices: Mutex<CurrentRandomSlices>,
    random_slices_limit: u32,
    motif_meta: MotifMeta,
    sync_motif_batch: Arc<DynSyncStageEnum<SyncMotifBatch<C>>>,
}

impl<L, C> MotifsBatch<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + MaxParallelCtx + Adjustable,
{
    pub async fn new(
        motif: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>,
        random_slices_loader: Arc<RandomSlicesStage<L, C>>,
        batch_size: i64,
        _device_idx: usize,
        mut ctx: C,
        epoch_size: u32,
    ) -> Result<Self, ThoError> {
        debug!("MotifsBatch::new: motif={motif:?}");
        let sync_motif_batch = Arc::new(SyncMotifBatch::new(ctx.tho_device_idx()).await?);
        let motif_meta = track_await(motif.extra()).await?;
        let split_motif = Arc::new(SplitMotifIntoTrainValid::<C>::new(ctx.tho_device_idx()).await?);
        let split_input = Arc::new(SplitMotifIntoTrainValidInputWrapper::new(motif));
        let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoTaskAdjuster, NoRunCtxAdjuster, ()> = OOMHandler::new(vec![], None);
        let split_output = track_await(oom_handler.run(&split_motif, split_input, ctx.branch(false), NoContexts)).await?;

        Ok(Self {
            motif_id: split_output.id,
            train_motifs: Arc::new(DevicesWrap::wrap(split_output.train_motifs)),
            valid_motifs: Arc::new(DevicesWrap::wrap(split_output.valid_motifs)),
            random_slices_loader,
            random_slices: Mutex::new(CurrentRandomSlices::default()),
            random_slices_limit: split_output.num_motif_slices,
            motif_meta,
            batch_size: batch_size as u32,
            epoch_size,
            sync_motif_batch,
        })
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="MotifsBatch::make_batch"))]
    async fn make_batch(&self, motifs: Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>, motif_id: &SynKey, descr: &str, mut ctx: C) -> Result<(ThoBatch<Slice>, ThoBatch<MotifProbability>), ThoError> {
        let random_slices = track_await(self.current_random_slices(ctx.branch(false))).await?;
        let num_slices = (self.batch_size / 2) as i64;

        debug!("MotifsBatch::make_batch: descr={descr} random_slices={random_slices:?} num_slices={num_slices}");
        let input = Arc::new(SyncMotifBatchInputWrapper::new(motif_id.clone(), motifs, random_slices, num_slices, descr.to_string()));
        let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoTaskAdjuster, NoRunCtxAdjuster, ()> = OOMHandler::new(vec![], None);
        let output = track_await(oom_handler.run(&self.sync_motif_batch, input, ctx.branch(false), NoContexts)).await?;
        Ok(output.into_parts())
    }

    async fn current_random_slices(&self, mut ctx: C) -> Result<Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>, ThoError> {
        let epoch_size = self.epoch_size.max(1);
        {
            let mut current_random_slices = self.random_slices.lock();
            if let Some(random_slices) = current_random_slices.slices.clone() {
                if current_random_slices.uses < epoch_size {
                    current_random_slices.uses += 1;
                    return Ok(random_slices);
                }
                current_random_slices.slices = None;
                current_random_slices.uses = 0;
            }
        }


        let random_slices: Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>> = track_await(Arc::clone(&self.random_slices_loader)
            .run(
                Arc::new(SlicesLoaderInput::new(self.motif_id.vals[0].to_data_id(), self.random_slices_limit)),
                ctx.branch(false),
            )).await?;

        let mut current_random_slices = self.random_slices.lock();
        current_random_slices.slices = Some(Arc::clone(&random_slices));
        current_random_slices.uses = 1;
        Ok(random_slices)
    }
}

impl<L, C> AsyncDataBatch<Slice, MotifProbability, C> for MotifsBatch<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + MaxParallelCtx + Adjustable,
{
    fn train_batch_stream<'a>(&'a self, mut ctx: C) -> BoxStream<'a, Result<(ThoBatch<Slice>, Option<BatchLengths>, ThoBatch<MotifProbability>), ThoError>> {
        debug!("MotifsBatch::train_batch_stream: 0. self.epoch_size={}", self.epoch_size);


        let train_motifs = Arc::clone(&self.train_motifs);
        let motif_id = self.motif_id.clone();
        let epoch_size = self.epoch_size as usize;

        futures_util::stream::iter(0..epoch_size)
            .then(move |epoch| {
                let train_motifs = Arc::clone(&train_motifs);
                let motif_id = motif_id.clone();
                let ctx = ctx.branch(false);
                async move {

                    track_await(self.make_batch(
                        train_motifs,
                        &motif_id,
                        "train",
                        ctx,
                    )).await
                }
            })
            .map(|fut_res| fut_res.map(|v| (v.0, None, v.1)))
            .boxed()
    }

    fn valid_batch_stream<'a>(&'a self, mut ctx: C) -> BoxStream<'a, Result<(ThoBatch<Slice>, Option<BatchLengths>, ThoBatch<MotifProbability>), ThoError>> {
        debug!("MotifsBatch::valid_batch_stream: 0. self.epoch_size={}", self.epoch_size);


        let valid_motifs = Arc::clone(&self.valid_motifs);
        let motif_id = self.motif_id.clone();
        let epoch_size = self.epoch_size as usize;

        futures_util::stream::iter(0..epoch_size)
            .then(move |_| {
                let valid_motifs = Arc::clone(&valid_motifs);
                let motif_id = motif_id.clone();
                let ctx = ctx.branch(false);
                async move {

                    track_await(self.make_batch(
                        valid_motifs,
                        &motif_id,
                        "valid",
                        ctx,
                    )).await
                }
            })
            .map(|fut_res| fut_res.map(|v| (v.0, None, v.1)))
            .boxed()
    }

    fn num_batches(&self) -> usize {
        self.epoch_size as usize
    }
}


#[cfg(test)]
mod tests {
    use crate::stages::find_motifs_run_config::FindMotifsRunConfig;
    use crate::stages::motif_loader::{MotifMeta, SYNC_LOAD_MOTIF_RESOURCES_ENDPOINT, SYNC_MOTIF_LOADER_ENDPOINT, sync_load_motif_resources_endpoint_init, sync_motif_loader_endpoint_init};
    use crate::stages::tensor_loader::mock_tensor_loader;
    use crate::stages::train_motif_nets::{SPLIT_MOTIF_INTO_TRAIN_VALID_ENDPOINT, SYNC_MOTIF_BASTCH_ENDPOINT, TRAIN_MOTIF_STEP_ENDPOINT, TrainMotifNetInput, TrainMotifNets, TrainMotifNetsResult, split_motif_into_train_valid_endpoint_init, sync_motif_batch_endpoint_init, train_motif_step_endpoint_init, train_motif_step_endpoint_inits};
    use crate::stages::train_step::{TrainStep, TrainStepInput, TrainStepXFn};
    use async_trait::async_trait;
    use futures_util::FutureExt;
    use futures_util::future::BoxFuture;
    use log::LevelFilter::{Debug, Trace};
    use log::debug;
    use parking_lot::{Mutex, RwLock};
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;
    use std::marker::PhantomData;
    use std::path::Path;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tch::Device::Cuda;
    use tch::Kind::Float;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::nn::{AdamW, OptimizerConfig, VarStore};
    use tch::{Device, IndexOp, Kind, Reduction, Tensor};
    use thoosa_common::batches::BatchLengths;
    use thoosa_common::builder::{Node, SeqBuilder};
    use thoosa_common::data_with_extra::DataWithExtra;
    use thoosa_common::debug_plot;
    use thoosa_common::dims::MotifSlices;
    use thoosa_common::dims::{Batch, Dims, LabelProbability, LabelsProbabilities, MotifProbability, Slice, ThoBatch};
    use thoosa_common::ipc::codec::{IpcCodec, JsonIpcCodec};
    use thoosa_common::ipc::cuda_wokrer::{AllTensors, WorkerStageInput};
    use thoosa_common::ipc::cuda_wokrer::{CudaWorker, InitFn, WokerRunCtx};
    use thoosa_common::ipc::json_ser::SyncStageIpcWrapper;
    use thoosa_common::ipc::proxy::{DynSyncStageWithStateEnum, ProxyStage};
    use thoosa_common::ipc::tests::run_ipc_test;
    use thoosa_common::ipc::worker_msg::{WorkerInput, WorkerOutput};
    use thoosa_common::ipc::wraps::WorkerWrapperWithState;
    use thoosa_common::my_tch::MotifInitModeFields;
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::runner::common::mock_run_ctx;
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runner::purge::FuturesRunnerMapPurger;
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use thoosa_common::tch_test_utils;
    use thoosa_common::tch_test_utils::resource_tensor;
    use thoosa_common::tho_device::ThoDevice;
    use thoosa_common::tho_tensor::{ThoTensor, ThoTensorInner};
    use thoosa_common::tho_varstore::{ModConfig, ThoTrainModule};
    use thoosa_common::train_nets::{CommonTrainingRunConfig, CommonTrainingRunConfigFields, FinalNetParams, MotifConvNetArgsFields, MotifCosineNetArgsFields, MotifLinNetArgs, MotifNetParams, MotifNetTypeArgs, MotifNetTypeArgsFields, MotifsBy, OptConfig, ThoModuleWithStore, TrainMotifNetRunConfig, TrainMotifNetRunConfigFields, final_net};
    use thoosa_common::utils::{reset_print_options, test_net};
    use thoosa_common::ws::{mock_ws_simp, Workspace};
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::hyperparameter::Hyperparameter;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::{OptRunConfigs, RunConfigId};
    use thoosa_shared::test_utils::MotifKind;
    use thoosa_shared::traits::{AsyncStage, DynSyncStage, DynSyncStageBox, MonoMetrics, Named, RunCtx, SimpleId, SyncStage};
    use thoosa_shared::traits::{DynSyncStageOutId, MaxParallelCtx};
    use thoosa_shared::val::{NamedRange, NamedVal, Val};
    use thoosa_shared::{TensorIdSliceIdx, ThoError, cdt_for_dir, mono_key, simple_id};
    use thoosa_store::{MockStoreBuilder, RunConfigRow};
    use tokio::fs;

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_train_motif_nets() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
        run_train_motif_nets().await
    }

    async fn run_train_motif_nets() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp/test_train_motif_nets").join(cdt_for_dir());

        debug!("test_train_motif_nets: test_dir={test_dir:?}");

        fs::create_dir_all(test_dir).await?;


        log::set_max_level(Debug);
        reset_print_options();

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 10000)));
        let find_motifs_run_config_id: RunConfigId = 1;
        let select_motifs_run_config_id: RunConfigId = 2;
        let motif_width = 100;

        let resources = (0..10).map(|i|Resource::new(i, 1, format!("resource_{i}"), 900)).collect::<Vec<_>>();

        let (resources_map, motif_intervals) = resources.iter().try_fold((HashMap::new(), vec![]), |mut a, r| {

            let (t, intervals) = resource_tensor(
                r.length() as usize,
                 &[
                    vec![(3, motif_width, MotifKind::Pulse)],
                    vec![(1, motif_width, MotifKind::Sin)],
                    vec![]
                ],
                 100, 10.0, true, true
            )?;

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
                .with_list_run_config_values([(
                    find_motifs_run_config_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_config_id), 50, 100, None, None).to_properties_iter(None).collect()),
                )])
                .with_run_config_descr([(
                    find_motifs_run_config_id,
                    Ok(RunConfigRow {
                        id: find_motifs_run_config_id,
                        r#type: "FindMotifsRunConfig".to_string(),
                        descr: "test find".to_string(),
                        run_on: "2026-01-01 00:00:00".to_string(),
                    }),
                )])
                .with_list_motif_slices(motif_id, Some(vec![select_motifs_run_config_id]), None, None, Ok((0i64, find_motifs_run_config_id, motif_intervals, resources_id_map.clone()

                )))
                .with_list_random_slices_out_of_motif(motif_id, motif_width as u32, None, Ok((0i64, find_motifs_run_config_id, (0..10).flat_map(|i|vec![
                    TensorIdSliceIdx(i, 100),
                    TensorIdSliceIdx(i, 300),
                    TensorIdSliceIdx(i, 500)
                ]).collect(), resources_id_map)))

                .with_list_selected_motifs([
                    (select_motifs_run_config_id, Ok(vec![vec![(motif_id, 1.0f64)]]))
                ])
                .with_list_output_run_config_ids([(select_motifs_run_config_id, Ok(vec![]))])
                .with_get_motif_meta( [(motif_id, Ok((0usize, find_motifs_run_config_id)))])

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
            run_configs,
            10,
            None
        ).await?);

        let input = TrainMotifNetInput::new(MotifsBy::SelectMotifsRunIds { select_motifs_run_config_ids: vec![select_motifs_run_config_id] }, Arc::new(OptConfig::new(3, 2)));

        let mut run_ctx = mock_run_ctx::default(1);
        let run_ctx_branch = run_ctx.branch(false);
        let r = train_motif_net.run(input, run_ctx_branch).await?;

        debug!("test_train_motif_nets: r={r:?}");

        fn run_test_net(params: &MotifNetParams, vs_path: &thoosa_shared::tmp_file::TmpFile) -> Result<(), ThoError> {
            // Training now returns the worker-owned checkpoint path, so rebuild
            // the module registry before loading its saved weights for this test.
            let mut vs = VarStore::new(Cuda(0));
            let net = params.motif_net(None)?.build_tch_module::<MotifSlices, MotifProbability>(&vs.root())?;
            vs.load(vs_path.path())?;

            let (t, intervals) = resource_tensor(
                1000usize,
                &[
                    vec![(2, 100, MotifKind::Pulse)],
                ],
                100,
                10.0,
                true,
                true,
            )?;

            debug!("intervals={intervals:?} t.size={:?}", t.size());
            debug_plot!(&t, "channel_0")?;

            let input = t.to_device(vs.device()).i((0, ..)).unfold(0, 100, 1).into();
            debug!("test_train_motif_nets: input={input:?}");

            let output = net.tho_forward(&input, None, false)?;

            debug!("test_train_motif_nets: output.t=\n{}", output.t_ref().view((1,-1)));

            debug_plot!(&output.t_ref().view((1,-1)), "applied 0")?;

            Ok(())
        }

        debug!("test_num_layers_opt: result={r:?}");
        assert!(r.train_motif_nets_results[0].loss < 0.69);
        run_test_net(&r.train_motif_nets_results[0].params, &r.train_motif_nets_results[0].vs)?;

        Ok(())
    }

    // #[test]
    fn test_train_motif_nets_ipc() -> Result<(), ThoError> {
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
                let cuda_actor = GlobalCtx::global().cuda_actor(1);
                let result = run_train_motif_nets().await?;

                Ok(())
            },
            async |sock_path| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    train_motif_step_endpoint_inits(),
                    ThoDevice::from(Cuda(0)),
                ).await
            },
        )
    }

    #[test]
    fn test_train_step_ipc() -> Result<(), ThoError> {
        const TRAIN_TEST_STEP_ENDPOINT: &str = "TrainTestStep";

        #[derive(Debug, Serialize, Deserialize)]
        struct TrainTestStepInit {
            lr: f64,
            vs_path: PathBuf,
        }

        impl WorkerStageInput for TrainTestStepInit {
        }

        #[typetag::serde(name = "TrainTestStepInit")]
        impl WorkerInput for TrainTestStepInit {}

        #[derive(Debug, Clone)]
        struct InDims;
        impl Dims for InDims { type Metrics = MonoMetrics<0, 1>; }

        #[derive(Debug, Clone)]
        struct OutDims;
        impl Dims for OutDims { type Metrics = MonoMetrics<0, 1>; }

        #[typetag::serde(name = "TrainTestStepInput")]
        impl WorkerInput for TrainStepInput<InDims, OutDims> {}

        #[derive(Clone, Debug)]
        struct TestNetParams {
            in_features: u32,
            layers: usize,
        }

        fn test_train_net(params: &TestNetParams) -> Result<Node, ThoError> {
            let mut builder = SeqBuilder::new(params.in_features, None)?;
            let mut net = builder
                .closure(Arc::new(Box::new(|t: &Tensor, _lengths: Option<&BatchLengths>| t.f_alias().map_err(ThoError::from))), None)?
                .linear_s(2 * params.in_features, "final_net_first_layer".to_string())?
                .func(|t| Tensor::f_relu(t).map_err(ThoError::from))?;
            for i in 0..params.layers {
                net = net.linear_s(params.in_features, format!("final_net_{i}th_layer"))?.func(|t| Tensor::f_relu(t).map_err(ThoError::from))?;
            }
            net.linear_s(1, "final_net_last_layer")?.end_seq()
        }

        fn train_test_step_endpoint_init<'a>(init: Box<dyn WorkerInput>, ws: Arc<RwLock<Box<dyn Workspace>>>, all_cuda_tensors: &'a AllTensors, tho_device:&'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
            async move {
                let init = *(init as Box<dyn std::any::Any>).downcast::<TrainTestStepInit>().map_err(|_| ThoError::new("TrainTestStep endpoint received invalid initializer"))?;
                let params = TestNetParams { in_features: 100, layers: 2 };
                let params_for_module = params.clone();
                let x_f: Arc<TrainStepXFn<InDims>> = Arc::new(Box::new(|batch| batch));
                let (stage, _): (TrainStep<TrainStepXFn<InDims>, InDims, OutDims, WokerRunCtx>, ()) = TrainStep::new_local(
                    tho_device.clone(),
                    move |(), device| async move {
                        let config = ModConfig::new(
                            AdamW::default().into(),
                            move |var_store| test_train_net(&params_for_module)?.build_tch_module::<Batch<InDims>, Batch<OutDims>>(&var_store.root()),
                            init.lr,
                        );
                        Ok((ThoTrainModule::new_on_device(config, device)?, ()))
                    },
                    x_f,
                    |predicted, expected| predicted.f_binary_cross_entropy_with_logits::<Tensor>(&expected.f_totype(Float)?, None, None, Reduction::Mean).map_err(ThoError::from),
                    (),
                    init.vs_path,
                ).await?;
                Ok(Box::new(WorkerWrapperWithState::<JsonIpcCodec, _, _, _>::new(Arc::new(stage), PhantomData)) as Box<dyn SyncStageIpcWrapper>)
            }.boxed()
        }

        // Exercise the same shared stateful wrapper used by production train steps.
        run_ipc_test(
            module_path!(),
            async {
                let stage_id = 0;
                let init = TrainTestStepInit { lr: 0.001, vs_path: PathBuf::from("/tmp/thoosa_train_step_ipc.safetensors") };
                let input_cloner = Arc::new(|input: &TrainStepInput<InDims, OutDims>| -> Result<Box<dyn WorkerInput>, ThoError> { Ok(Box::new(input.clone_refs_for_worker()?)) });
                let train_step: DynSyncStageWithStateEnum<TrainStep<TrainStepXFn<InDims>, InDims, OutDims, RunCtxImpl>> = DynSyncStageWithStateEnum::Proxy {
                    stage: Arc::new(ProxyStage::proxy_with_input_cloner(TRAIN_TEST_STEP_ENDPOINT, stage_id, 1, Box::new(init), input_cloner).await?),
                    init: (),
                };
                let cuda_actor = GlobalCtx::global().cuda_actor(1);
                let input = TrainStepInput::<InDims, OutDims>::new(
                    cuda_actor.copy(ThoTensor::new((), Tensor::rand([3, 100], FLOAT_CPU))?, 1.into()).await?.to_types(simple_id!("2"))?,
                    None,
                    cuda_actor.copy(ThoTensor::new((), Tensor::rand([3, 1], FLOAT_CPU))?, 1.into()).await?.to_types(simple_id!("3"))?,
                    false,
                    false,
                );
                let output = Arc::new(train_step).do_run(&input, &RunCtxImpl::new(1, "train_step_ipc"))?;
                assert!(output.loss.is_finite());
                Ok(())
            },
            async |sock_path| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    HashMap::from([(TRAIN_TEST_STEP_ENDPOINT.to_string(), Box::new(train_test_step_endpoint_init) as InitFn)]),
                    ThoDevice::from(Cuda(0)),
                ).await
            },
        )
    }


    fn cos_net_run_configs() -> OptRunConfigs<Vec<NamedVal<TrainMotifNetRunConfigFields>>, Vec<NamedRange<TrainMotifNetRunConfigFields>>, Vec<Hyperparameter<TrainMotifNetRunConfigFields>>, TrainMotifNetRunConfigFields> {
        thoosa_shared::run_configs::OptRunConfigs::new(
            thoosa_shared::run_configs::GridRunConfigs::new(
                thoosa_shared::run_configs::ConstConfigs::new(vec![
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Epochs), Val::Int(300)),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::BatchSize), Val::Int(10)),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Patience), Val::Int(10)),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::MinDelta), Val::Real(1e-4)),
                    // NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), Val::Real(0.0001)),
                    NamedVal::new(TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::ConvNet(MotifConvNetArgsFields::Step)), Val::Int(1)),
                    NamedVal::new(TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::ConvNet(MotifConvNetArgsFields::Kernel)), Val::Int(100)),
                    NamedVal::new(TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::ConvNet(MotifConvNetArgsFields::HeadLayers)), Val::Int(8)),
                ]),
                vec![
                    // NamedRange::new(TrainMotifNetRunConfigFields::NumLayers, Val::Int(0)..Val::Int(10), (Val::Int(10), 0))
                ],
            ),
            vec![
                // Hyperparameter::UniformInt{name: TrainMotifNetRunConfigFields::NumLayers, lower: 6, upper: 8},
                Hyperparameter::UniformFloat { name: TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), lower: 0.00001, upper: 0.0001 },
                Hyperparameter::UniformInt { name: TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::ConvNet(MotifConvNetArgsFields::Channels)), lower: 8, upper: 32 }
            ],
        )
    }

    fn cosine_net_run_configs() -> OptRunConfigs<Vec<NamedVal<TrainMotifNetRunConfigFields>>, Vec<NamedRange<TrainMotifNetRunConfigFields>>, Vec<Hyperparameter<TrainMotifNetRunConfigFields>>, TrainMotifNetRunConfigFields> {
        thoosa_shared::run_configs::OptRunConfigs::new(
            thoosa_shared::run_configs::GridRunConfigs::new(
                thoosa_shared::run_configs::ConstConfigs::new(vec![
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Epochs), Val::Int(300)),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::BatchSize), Val::Int(10)),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Patience), Val::Int(10)),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::MinDelta), Val::Real(1e-4)),
                    // NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), Val::Real(0.0001)),
                    // NamedVal::new(TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::CosineNet(MotifCosineNetArgsFields::A)), Val::Real(5.0)),
                    NamedVal::new(TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::CosineNet(MotifCosineNetArgsFields::B)), Val::Real(0.0)),
                    NamedVal::new(TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::CosineNet(MotifCosineNetArgsFields::InitMode(MotifInitModeFields::Mean__Selected))), Val::UInt(0)),
                ]),
                vec![
                    // NamedRange::new(TrainMotifNetRunConfigFields::NumLayers, Val::Int(0)..Val::Int(10), (Val::Int(10), 0))
                ],
            ),
            vec![
                // Hyperparameter::UniformInt{name: TrainMotifNetRunConfigFields::NumLayers, lower: 6, upper: 8},
                Hyperparameter::UniformFloat { name: TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), lower: 0.00001, upper: 0.0001 },
                Hyperparameter::UniformFloat { name: TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::CosineNet(MotifCosineNetArgsFields::A)), lower: 1.0, upper: 10.0 }
            ],
        )
    }

}

#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct TrainMotifNetsResult {
    select_motifs_run_config_ids: Vec<RunConfigId>,
    pub train_motif_nets_results:Vec<TrainMotifNetResult>
}

impl TrainMotifNetsResult {
    pub fn new(select_motifs_run_config_ids: Vec<RunConfigId>, train_motif_nets_results: Vec<TrainMotifNetResult>) -> Self {
        Self {  select_motifs_run_config_ids, train_motif_nets_results }
    }
}

#[async_trait]
impl SavableResult for TrainMotifNetsResult {
    type R = Vec<u32>;

    async fn save(&mut self, store: &Arc<Box<dyn Store>>, ws: &Arc<RwLock<Box<dyn Workspace>>>, p: Self::P<'_>) -> Result<Vec<u32>, ThoError> {

        let mut result = Vec::with_capacity(self.train_motif_nets_results.len());
        for r in self.train_motif_nets_results.iter_mut() {
            result.push(
            track_await(r.save(store, ws, self.select_motifs_run_config_ids.as_slice())).await?
            );
        }

        Ok(result)
    }
}


#[async_trait]
impl Output for TrainMotifNetsResult {
}
