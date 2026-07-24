#![allow(unexpected_cfgs)]

use std::collections::{HashMap, HashSet};
use std::future::ready;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;
use async_trait::async_trait;
use futures_util::{FutureExt, StreamExt};
use futures_util::stream::BoxStream;
use log::{debug, error, info, trace, warn};

use parking_lot::{Mutex, RwLock};
use plotters::prelude::BLACK;
use plotters::style::RED;
use tch::Device::{Cpu, Cuda};
use tch::kind::FLOAT_CUDA;
use tch::nn::{AdamW, Optimizer, OptimizerConfig, VarStore};
use tch::{IndexOp, Reduction, Tensor};
use tch::Kind::Float;
use serde::{Deserialize, Serialize};
use thoosa_common::ipc::codec::{IpcCodec, JsonIpcCodec};
use thoosa_common::ipc::cuda_wokrer::WorkerStageOutput;
use thoosa_common::ipc::json_ser::{SyncStageIpcResponse, SyncStageIpcStreamResponse, SyncStageIpcWrapper};
use thoosa_common::runner::common::cachable::{SelectRef, Cached, DevicesWrap, ThoDataRef};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::common::{MaxParallelCtx, OutId, RunCtx};
use thoosa_common::runner::common::runnable::{AsyncStage, DynSyncStageBox, Stage, SyncStage};
use thoosa_common::runner::oom_handler::{NoContexts, NoInputAdjuster, NoRunCtxAdjuster, NoTaskAdjuster, OOMHandler};
use thoosa_common::runner::run_ctx_2::RunCtxImpl;

use mockall_double::double;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::debug_plot;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::batches::{AsyncDataBatch, BatchLengths};
use thoosa_common::dims::{Batch, MotifProbability, MotifSlices, Slice, ThoBatch};
use thoosa_common::my_tch::MotifInitMode;
use thoosa_common::parzen::optimizer::CompletedTrial;
use thoosa_common::runs::Empty;
use thoosa_common::tho_tensor::{ThoTensor};
use thoosa_common::train_nets::{lin_motif_node, opt_train, MotifNetParams, MotifNetTypeArgs, MotifNetTypeArgsFields, MotifNetTypeParams, MotifsBy, OptConfig, StftRunConfig, TrainMotifNetResult, TrainMotifsNetsResult, TrainMotifNetRunConfig, TrainMotifNetRunConfigFields};
use thoosa_common::utils::{mix_x_binary_with_labels, plot_spectrogram, plot_series_with_lanes, to_vec_f32, DummyListener};
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_run_config_args::OptRunConfigSource;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::resource::Resource;
use thoosa_shared::run_configs::{OptRunConfigs, RunConfig, RunConfigId};
use thoosa_shared::{collect_futures_with_progress, mono_key, perf_log, sync_read_lock, CopyToDevice, FuturesOrdering, ThoError};
use thoosa_shared::traits::{Adjustable, Data, DataId, WithId};
use thoosa_shared::val::{NamedRange, NamedVal, Val};
use thoosa_store::Store;
use thoosa_shared::comp_id::SynKey;
use thoosa_common::dims::ChannelsData;
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::motif_loader::{sync_load_motif_resources_endpoint_init, sync_load_random_slices_endpoint_init, sync_motif_loader_endpoint_init, MotifLoader, MotifLoaderInput, MotifMeta, RandomSlicesLoader, SlicesLoaderInput, SYNC_LOAD_MOTIF_RESOURCES_ENDPOINT, SYNC_LOAD_RANDOM_SLICES_ENDPOINT, SYNC_MOTIF_LOADER_ENDPOINT};
use crate::tasks::train_label_nets::{sync_label_batch_endpoint_init, SYNC_LABEL_BATCH_ENDPOINT};
use crate::tasks::motifs_batch_builder::{make_batch_with, MixupJitterCfg};
use crate::tasks::stage_ids::TRAIN_MOTIF_NETS;
use thoosa_common::stages_seq::ABStage;
use crate::tasks::stft::{Stft, SyncStft, SyncStftSelector};
use crate::tasks::tensor_loader::RawTensorLoader;
use crate::tasks::train_net::{TrainNet, TrainNetInput, TrainStep, TrainStepInput, TrainStepOutput, TrainStepProxyFactory, TrainStepProxyParts, TrainStepWeightsSync, TrainStepXFn};
use thoosa_shared::tracked_await::track_await;
use thoosa_proxy_derive::Proxy;

pub(crate) const TRAIN_MOTIF_STEP_ENDPOINT: &str = "TrainMotifStep";

#[derive(Debug, Serialize, Deserialize)]
struct TrainStepTensorRef {//TODO replace with ThoTensor
    ref_key: u64,
    size: Vec<i64>,
    kind: String,
    tho_device_idx: usize,
    remove_after_run: bool,
}

impl TrainStepTensorRef {
    fn from_ref<D, Id>(tensor: &ThoTensor<D, Id>) -> Result<Self, ThoError>
    where
        D: thoosa_common::dims::Dims,
        Id: thoosa_shared::traits::Named + std::fmt::Debug + Clone,
    {
        Ok(Self {
            ref_key: tensor.ref_key().ok_or_else(|| ThoError::new(format!("TrainMotifStep expected worker tensor ref, got {tensor:?}")))?,
            size: tensor.size(),
            kind: thoosa_common::ipc::json_ser::tho_tensor::kind_to_str(&tensor.kind()).map_err(ThoError::new)?.to_string(),
            tho_device_idx: tensor.tho_device().tho_device_idx(),
            remove_after_run: false,
        })
    }

    async fn from_tensor<D>(tensor: &ThoBatch<D>, tho_device_idx: usize) -> Result<Self, ThoError>
    where
        D: thoosa_common::dims::Dims,
    {
        if tensor.ref_key().is_some() {
            return Self::from_ref(tensor);
        }

        let cpu_tensor = tensor.t_ref().f_to_device(Cpu)?;
        let worker_ref = track_await(thoosa_common::runner::global_ctx::GlobalCtx::global()
            .cuda_actor(tho_device_idx)
            .copy(cpu_tensor.into(), ThoDevice::from(tho_device_idx))).await?;
        let mut tensor_ref = Self::from_ref(&worker_ref)?;
        tensor_ref.remove_after_run = true;
        Ok(tensor_ref)
    }

    fn into_tensor<D>(self) -> Result<ThoBatch<D>, ThoError>
    where
        D: thoosa_common::dims::Dims,
    {
        let kind = thoosa_common::ipc::json_ser::tho_tensor::str_to_kind(&self.kind).ok_or_else(|| ThoError::new(format!("TrainMotifStep received unsupported tensor kind {:?}", self.kind)))?;
        Ok(ThoTensor::new_ref(thoosa_shared::simple_id!("train_motif_step_ipc"), self.ref_key, self.size, kind, None, self.tho_device_idx))
    }
}

#[derive(Debug, Serialize, Deserialize)]
enum TrainMotifStepIpcRequest {
    Run {
        x: TrainStepTensorRef,
        lengths: Option<Vec<i64>>,
        expected_y: TrainStepTensorRef,
        train: bool,
        max_other_parallel: u8,
    },
    ExportWeights,
}

impl TrainMotifStepIpcRequest {
    async fn run(input: &TrainStepInput<Slice, MotifProbability>, ctx: &RunCtxImpl) -> Result<Self, ThoError> {
        Ok(Self::Run {
            x: track_await(TrainStepTensorRef::from_tensor(input.x_ref(), ctx.tho_device_idx())).await?,
            lengths: input.lengths_ref().map(|lengths| lengths.as_slice().to_vec()),
            expected_y: track_await(TrainStepTensorRef::from_tensor(input.expected_y_ref(), ctx.tho_device_idx())).await?,
            train: input.train(),
            max_other_parallel: ctx.max_other_parallel(),
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct TrainMotifStepIpcInit {
    params: MotifNetParams,
    learning_rate: f64,
    motif: ThoTensor<MotifSlices, SynKey>,
    run_config:TrainMotifNetRunConfig,
    stft_config: Option<StftRunConfig>,
}

fn train_motif_x_f(batch: ThoBatch<Slice>) -> ThoBatch<Slice> {
    batch
}

fn train_motif_loss(pred_y: &Tensor, target: &Tensor) -> Result<Tensor, ThoError> {
    pred_y.f_binary_cross_entropy_with_logits::<Tensor>(&target.f_totype(Float)?, None, None, Reduction::Mean).map_err(ThoError::from)
}

type SyncTrainMotifStep = TrainStep<TrainStepXFn<Slice>, Slice, MotifProbability, RunCtxImpl>;

const OPTMIZER_FN: fn(&TrainMotifNetRunConfig, &VarStore) -> Result<Optimizer, ThoError> = move |rc:&TrainMotifNetRunConfig, vs: &VarStore|{
    AdamW::default().amsgrad(true).build(&vs, rc.common_training_args.lr).map_err(|e|e.into())
};

fn new_train_motif_step_for_ipc(init: &TrainMotifStepIpcInit, all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>) -> Result<SyncTrainMotifStep, ThoError> {
    let motif_ref = ThoTensor::new_ref(//TODO ref clone
        init.motif.id().clone(),
        init.motif.ref_key().ok_or_else(|| ThoError::new(format!("TrainMotifStepIpcInit expected motif worker reference, got {:?}", init.motif)))?,
        init.motif.size(),
        init.motif.kind(),
        init.motif.selector(),
        init.motif.tho_device().tho_device_idx(),
    );
    let motif = ThoTensor::ref_to_tensor(all_cuda_tensors, motif_ref)?;
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
    let params = init.params.clone();
    let learning_rate = init.learning_rate;
    let x_f: Arc<TrainStepXFn<Slice>> = Arc::new(Box::new(train_motif_x_f));
    let (train_step, ()) = futures::executor::block_on(TrainStep::new_local(
        Cuda(0).into(),
        move |var_store| async move {//TODO use const like OPTMIZER_FN to make sure it's same as in local
            let net = params.motif_net(cosine_motif)?.build_tch_module::<thoosa_common::dims::Batch<Slice>, thoosa_common::dims::Batch<MotifProbability>>(&var_store.lock().root())?;
            Ok((net, ()))
        },
        OPTMIZER_FN,
        x_f,//TODO use const like OPTMIZER_FN to make sure it's same as in local
        train_motif_loss,
        &init.run_config
    ))?;
    Ok(train_step)
}

#[derive(Debug)]
struct TrainMotifStepIpcInitStage {
    init: TrainMotifStepIpcInit,
    stage: Mutex<Option<Arc<SyncTrainMotifStep>>>,
}

impl TrainMotifStepIpcInitStage {
    fn new(init: TrainMotifStepIpcInit) -> Self {
        Self { init, stage: Mutex::new(None) }
    }

    fn stage(&self, all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>) -> Result<Arc<SyncTrainMotifStep>, ThoError> {
        let mut stage = self.stage.lock();
        if stage.is_none() {
            stage.replace(Arc::new(new_train_motif_step_for_ipc(&self.init, all_cuda_tensors)?));
        }
        Ok(Arc::clone(stage.as_ref().ok_or_else(|| ThoError::new("TrainMotifStepIpcInitStage failed to initialize stage"))?))
    }
}

#[derive(Debug)]
struct TrainMotifStepIpcWrapper {
    stage: Arc<TrainMotifStepIpcInitStage>,
}

impl SyncStageIpcWrapper for TrainMotifStepIpcWrapper {
    fn do_run(&self, input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, _tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError> {
        match JsonIpcCodec::decode::<TrainMotifStepIpcRequest>(input)? {
            TrainMotifStepIpcRequest::Run { x, lengths, expected_y, train, max_other_parallel } => {
                let stage = self.stage.stage(all_cuda_tensors)?;
                let remove_x = x.remove_after_run.then_some(x.ref_key);
                let remove_expected_y = expected_y.remove_after_run.then_some(expected_y.ref_key);

                let x = ThoTensor::ref_to_tensor(all_cuda_tensors, x.into_tensor::<Slice>()?)?;
                let expected_y = ThoTensor::ref_to_tensor(all_cuda_tensors, expected_y.into_tensor::<MotifProbability>()?)?;

                let input = TrainStepInput::new(x, lengths.map(BatchLengths::new), expected_y, train);

                let mut ctx = RunCtxImpl::new(1, TRAIN_MOTIF_STEP_ENDPOINT);
                ctx.set_max_other_parallel(max_other_parallel);
                let result = stage.do_run(&input, &ctx);
                drop(input);
                if remove_x.is_some() || remove_expected_y.is_some() {
                    let mut tensors = all_cuda_tensors.lock();
                    if let Some(key) = remove_x {
                        tensors.remove(&key);
                    }
                    if let Some(key) = remove_expected_y {
                        tensors.remove(&key);
                    }
                }
                JsonIpcCodec::encode(&result?)
            }
            TrainMotifStepIpcRequest::ExportWeights => {
                Err(ThoError::new("TrainMotifStep ExportWeights requires binary endpoint invocation"))
            }
        }
    }

    fn do_run_response(&self, input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<SyncStageIpcResponse, ThoError> {
        if matches!(JsonIpcCodec::decode::<TrainMotifStepIpcRequest>(input)?, TrainMotifStepIpcRequest::ExportWeights) {
            let stage = self.stage.stage(all_cuda_tensors)?;
            Ok(SyncStageIpcResponse::Stream(SyncStageIpcStreamResponse::new(move |output| stage.save_var_store_to_stream(output))))
        } else {
            self.do_run(input, all_cuda_tensors, tho_device).map(SyncStageIpcResponse::Encoded)
        }
    }
}

#[derive(Debug, Proxy)]
#[proxy(
    common = "thoosa_common",
    name = "ProxyTrainMotifStep",
    endpoint = "TRAIN_MOTIF_STEP_ENDPOINT",
    new_args = "init: TrainMotifStepIpcInit, tho_device_idx: usize",
    init = "init",
    fields = "tho_device_idx: usize = tho_device_idx",
    out_id = "()",
    run_input = "thoosa_shared::tracked_await::track_await(TrainMotifStepIpcRequest::run(input, ctx)).await?",
    endpoint_init = "train_motif_step_endpoint_init",
    init_type = "TrainMotifStepIpcInit",
    init_stage = "TrainMotifStepIpcInitStage::new(init)",
    wrapper = "TrainMotifStepIpcWrapper"
)]
struct TrainMotifStepProxyTarget {
    base: TrainStep<TrainStepXFn<Slice>, Slice, MotifProbability, RunCtxImpl>,
}

impl ProxyTrainMotifStep {
    async fn export_weights(&self) -> Result<std::fs::File, ThoError> {
        thoosa_common::ipc::codec::invoke_ipc_endpoint_to_file::<JsonIpcCodec, _>(
            thoosa_common::runner::global_ctx::GlobalCtx::global().cuda_actor(self.tho_device_idx).as_ref(),
            TRAIN_MOTIF_STEP_ENDPOINT,
            &TrainMotifStepIpcRequest::ExportWeights,
        ).await
    }
}

//TODO Proxies are made for concreate Ctx, Local stages take Ctx as type param
#[derive(Debug)]
struct TrainMotifStepProxyCtx<C>
where
    C: RunCtx + 'static,
{
    proxy: Arc<ProxyTrainMotifStep>,
    pd: PhantomData<C>,
}

impl<C> Stage for TrainMotifStepProxyCtx<C>
where
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = TrainStepInput<Slice, MotifProbability>;
    type Out = TrainStepOutput;
}

impl<C> SyncStage for TrainMotifStepProxyCtx<C>
where
    C: RunCtx + 'static,
{
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let mut proxy_ctx = RunCtxImpl::new(ctx.tho_device_idx(), TRAIN_MOTIF_STEP_ENDPOINT);
        proxy_ctx.set_max_other_parallel(ctx.max_other_parallel());
        Arc::clone(&self.proxy).do_run(input, &proxy_ctx)
    }
}

impl<C> OutId for TrainMotifStepProxyCtx<C>
where
    C: RunCtx + 'static,
{
    fn out_id<'a>(&self, _input_id: &SynKey) -> () {
        ()
    }
}

impl<C> Adjustable for TrainMotifStepProxyCtx<C> where C: RunCtx + 'static {}

#[derive(Debug)]
struct TrainMotifStepWeightsSync {
    proxy: Arc<ProxyTrainMotifStep>,
}

#[async_trait]
impl TrainStepWeightsSync for TrainMotifStepWeightsSync {
    async fn into_var_store(&self) -> Result<VarStore, ThoError> {
        let weights = track_await(self.proxy.export_weights()).await?;
        let named_tensors = Tensor::load_multi_from_stream_with_device(weights, Cpu)?;
        let var_store = VarStore::new(Cpu);
        var_store.variables_.lock().map_err(|_| ThoError::new("TrainMotifStepWeightsSync VarStore lock poisoned"))?.named_variables.extend(named_tensors);
        Ok(var_store)
    }
}

#[derive(Debug)]
struct TrainMotifStepProxyFactory<C> {
    motif: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>,
    motif_width: u32,
    pd: PhantomData<C>,
}

impl<C> TrainMotifStepProxyFactory<C> {
    fn new(motif: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>, motif_width: u32) -> Self {
        Self { motif, motif_width, pd: PhantomData }
    }
}

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
                return Err(ThoError::new(format!("TrainMotifStepProxyFactory invalid STFT dimensions: motif_width={motif_width} n_fft={n_fft} hop_length={hop_length}")));
            }
            let kw = u32::try_from((motif_width_i64 - n_fft) / hop_length + 1).map_err(|error| ThoError::new(format!("TrainMotifStepProxyFactory invalid STFT time dimension: {error}")))?;
            let kh_i64 = if args.stft_config.onesided { n_fft / 2 + 1 } else { n_fft };
            let kh = u32::try_from(kh_i64).map_err(|error| ThoError::new(format!("TrainMotifStepProxyFactory invalid STFT frequency dimension: {error}")))?;
            Ok((MotifNetParams::new_cosine_2d(kw, kh, args.a, args.b, args.init_mode.clone(), motif_width), Some(args.stft_config.clone())))
        }
    }
}

#[async_trait]
impl<C> TrainStepProxyFactory<Slice, MotifProbability, C, MotifNetParams, TrainMotifNetRunConfig> for TrainMotifStepProxyFactory<C>
where
    C: RunCtx + 'static,
{
    async fn new_proxy(&self, run_config: &TrainMotifNetRunConfig, tho_device_idx: usize) -> Result<TrainStepProxyParts<Slice, MotifProbability, C, MotifNetParams>, ThoError> {
        let (node_params, stft_config) = train_motif_step_params(run_config, self.motif_width)?;
        let motif = track_await(self.motif.borrow_to(tho_device_idx, None::<SyncStftSelector>)).await?;
        let motif = motif.as_ref().data();
        let motif = ThoTensor::new_ref(//TODO ref clone
            motif.id().clone(),
            motif.ref_key().ok_or_else(|| ThoError::new(format!("TrainMotifStepProxyFactory expected motif worker reference, got {motif:?}")))?,
            motif.size(),
            motif.kind(),
            motif.selector(),
            motif.tho_device().tho_device_idx(),
        );
        let init = TrainMotifStepIpcInit { params: node_params.clone(), learning_rate: run_config.common_training_args.lr, motif, run_config: run_config.clone(), stft_config };
        let proxy = Arc::new(ProxyTrainMotifStep::new(init, tho_device_idx).await?);
        let stage = Arc::new(DynSyncStageBox::new(Arc::new(TrainMotifStepProxyCtx::<C> { proxy: Arc::clone(&proxy), pd: PhantomData })));
        Ok(TrainStepProxyParts::new(stage, Arc::new(TrainMotifStepWeightsSync { proxy }), node_params))
    }
}

pub fn train_motif_step_endpoint_inits() -> HashMap<String, Box<dyn Fn(Vec<u8>) -> Result<Box<dyn SyncStageIpcWrapper>, ThoError>>> {
    HashMap::from([
        (
            TRAIN_MOTIF_STEP_ENDPOINT.to_string(),
            Box::new(train_motif_step_endpoint_init) as Box<dyn Fn(Vec<u8>) -> Result<Box<dyn SyncStageIpcWrapper>, ThoError>>,
        ),
        (
            SYNC_MOTIF_BASTCH_ENDPOINT.to_string(),
            Box::new(sync_motif_bastch_endpoint_init) as Box<dyn Fn(Vec<u8>) -> Result<Box<dyn SyncStageIpcWrapper>, ThoError>>,
        ),
        (
            SPLIT_MOTIF_INTO_TRAIN_VALID_ENDPOINT.to_string(),
            Box::new(split_motif_into_train_valid_endpoint_init) as Box<dyn Fn(Vec<u8>) -> Result<Box<dyn SyncStageIpcWrapper>, ThoError>>,
        ),
        (
            SYNC_LOAD_MOTIF_RESOURCES_ENDPOINT.to_string(),
            Box::new(sync_load_motif_resources_endpoint_init) as Box<dyn Fn(Vec<u8>) -> Result<Box<dyn SyncStageIpcWrapper>, ThoError>>,
        ),
        (
            SYNC_LOAD_RANDOM_SLICES_ENDPOINT.to_string(),
            Box::new(sync_load_random_slices_endpoint_init) as Box<dyn Fn(Vec<u8>) -> Result<Box<dyn SyncStageIpcWrapper>, ThoError>>,
        ),
        (
            SYNC_MOTIF_LOADER_ENDPOINT.to_string(),
            Box::new(sync_motif_loader_endpoint_init) as Box<dyn Fn(Vec<u8>) -> Result<Box<dyn SyncStageIpcWrapper>, ThoError>>,
        ),
        (
            SYNC_LABEL_BATCH_ENDPOINT.to_string(),
            Box::new(sync_label_batch_endpoint_init) as Box<dyn Fn(Vec<u8>) -> Result<Box<dyn SyncStageIpcWrapper>, ThoError>>,
        ),
    ])
}

type RandomSlicesStft<C> = Stft<C, MotifSlices, MotifSlices>;
type RandomSlicesLoaderWithStft<L, C> = ABStage<RandomSlicesLoader<L, C>, RandomSlicesStft<C>, C>;

#[derive(Debug)]
enum RandomSlicesStage<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    Raw(Arc<RandomSlicesLoader<L, C>>),
    WithStft(Arc<RandomSlicesLoaderWithStft<L, C>>),
}

impl<L, C> Stage for RandomSlicesStage<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = Arc<SlicesLoaderInput>;
    type Out = Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>;
}

impl<L, C> OutId for RandomSlicesStage<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
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
    C: RunCtx + Adjustable + 'static,
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
    C: RunCtx + Adjustable + 'static
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


impl <L, C> OutId for TrainMotifNets<L, C>
where
    L: for <'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{

    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        ()
    }
}


#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct TrainMotifNetInput{
    id: SynKey,
    motifs_by: MotifsBy,
    opt_config:Arc<OptConfig>
}

impl Input for TrainMotifNetInput {}
impl TrainMotifNetInput {
    pub fn new(motifs_by: MotifsBy, opt_config:Arc<OptConfig>) -> Self {
        Self { id: SynKey::new([mono_key!("TrainMotifNetInput", "stage")]), motifs_by, opt_config }
    }
}

impl WithId for TrainMotifNetInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
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
    C: RunCtx + Adjustable + 'static,
{
    let device_idx = ctx.tho_device_idx();
    let out_id = input.id().clone();
    let specs = track_await(Arc::clone(&stft)
        .run(
            Arc::new(DevicesWrap::wrap(input)),
            ctx.branch(false),
        )).await?;
    let specs = track_await(Arc::into_inner(specs)
        .ok_or_else(|| ThoError::new("run_stft_as_motif_slices: stft output should not be shared"))?
        .take_data(device_idx)).await?;
    ThoTensor::new(out_id, specs.t())
}

impl <L, C> Stage for TrainMotifNets<L, C>
where
    L: for <'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    type Ctx = C;
    type In<'a> = TrainMotifNetInput;//MotifID
    type Out = Arc<TrainMotifsNetsResult>;
}

impl <L, C> TrainMotifNets<L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for <'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx=C> + Cached + OutId + 'static,
{
    pub fn new(
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
        let Some(loss) = values.get("loss").and_then(val_to_f32) else {
            continue;
        };
        if loss.is_finite() {
            trials.push(CompletedTrial::new(loss, values));
        }
    }
    Ok(trials)
}

fn val_to_f32(val: &Val) -> Option<f32> {
    match val {
        Val::Real(v) => Some(*v as f32),
        Val::Int(v) => Some(*v as f32),
        Val::UInt(v) => Some(*v as f32),
        Val::Str(v) => v.parse().ok(),
        Val::Bool(_) => None,
    }
}

impl <L, C> AsyncStage for TrainMotifNets<L, C>
where
    C: RunCtx + Adjustable + 'static,
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
                    move |mut ctx:C, rc:&TrainMotifNetRunConfig, vs: Arc<Mutex<VarStore>>| {
                        // let mut ctx = ctx_net.branch(Some(&format!("train_motif_nets_task_{}_3", motif_id_rate.0)), false);
                        let motif = Arc::clone(&motif);


                        async move {
                            let motif = Arc::clone(&motif);
                            // let mut ctx = ctx_net.branch(Some(&format!("train_motif_nets_task_{}_3", motif_id_rate.0)), false);

                            match &rc.net {
                                MotifNetTypeArgs::ConvNet(net) => {
                                    let params = MotifNetParams::new_conv(motif_width, net.conv_layers, net.head_layers, net.step, net.channels, net.kernel);

                                    Ok((params.motif_net(None)?.build_tch_module(&vs.lock().root())?, params))
                                },
                                MotifNetTypeArgs::LinNet(net) => {
                                    let params = MotifNetParams::new_lin(motif_width, net.num_layers, net.step);
                                    Ok((params.motif_net(None)?.build_tch_module(&vs.lock().root())?, params))
                                },
                                MotifNetTypeArgs::CosineNet(net) => {
                                    let params = MotifNetParams::new_cosine(motif_width, net.a, net.b, net.init_mode.clone());


                                    let cosine_motif = Some({
                                        let motif_ref = motif.borrow_to(ctx.tho_device_idx(), None::<SyncStftSelector>).await?;
                                        let motif = motif_ref.as_ref();

                                        let src = motif.data().as_ref();
                                        let mut dst = src.f_zeros_like()?;
                                        dst.f_copy_(&src)?;
                                        Ok::<_, ThoError>(Arc::new(ThoTensor::<MotifSlices, SynKey>::new(motif.id().clone(), dst)?))
                                    }?);


                                    Ok((params.motif_net(cosine_motif)?.build_tch_module(&vs.lock().root())?, params))
                                },
                                MotifNetTypeArgs::Cosine2DNet(args) => {
                                    let stft = Arc::new(RandomSlicesStft::new_for_dims(args.stft_config.clone())?);
                                    debug!("TrainMotifNets::run: motif_id={} motif_width={motif_width}", motif_id_rate.0);


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
                                    let size = motif_specs.size();
                                    let params = MotifNetParams::new_cosine_2d(size[2] as u32, size[1] as u32, args.a, args.b, args.init_mode.clone(), motif_width);

                                    let cosine_motif = motif_specs.as_ref();
                                    let mut dst = cosine_motif.as_ref().as_ref().f_zeros_like()?;
                                    dst.f_copy_(&cosine_motif)?;//kernel will be changed during training
                                    debug!("TrainMotifNets::run: dst={dst:?}");

                                    Ok((params.motif_net(Some(Arc::new(ThoTensor::new(motif.id().clone(), dst)?)))?.build_tch_module(&vs.lock().root())?, params))
                                }
                            }
                        }.boxed()
                    },
                    opt_config.clone(), //TODO clone
                    train_motifs_nodes_run_configs.clone(), //TODO clone
                    |pred_y: &Tensor, target: &Tensor| {

                        // let loss = pred_y.binary_cross_entropy::<Tensor>(&target.totype(Float), None, Reduction::Mean).set_requires_grad(true);
                        let loss = pred_y.f_binary_cross_entropy_with_logits::<Tensor>(&target.f_totype(Float)?, None, None, Reduction::Mean)?;
                        debug!("TrainMotifNets::run, loss_fn: pred_y ={pred_y} target={target} loss={loss}");

                        Ok(loss)
                    },
                    OPTMIZER_FN,
                    |t:ThoBatch<Slice>| {
                        t
                    },
                    |rc:&TrainMotifNetRunConfig|{
                        rc.common_training_args.epochs
                    },
                    1,
                    format!("motif_{}", motif_id_rate.0)
                )
                    .with_reusable_completed_trials(reusable_completed_trials)
                    .with_train_step_proxy_factory(Arc::new(TrainMotifStepProxyFactory::<C>::new(motif_for_net, motif_width))));

                debug!("TrainMotifNets::run: train_net={train_net:?}");

                let train_net_output = track_await(Arc::into_inner(track_await(train_net.run(
                    Arc::new(TrainNetInput::new(SynKey::new([mono_key!("", "empty")]))),

                    ctx_run
                )).await?).unwrap().take_data(0)).await?;

                debug!("TrainMotifNets::run, done: motif_id_rate={motif_id_rate:?} min_loss={}", train_net_output.min_loss);

                let (channel, _find_motifs_run_config_id) = track_await(store.get_motif_meta(motif_id_rate.0)).await?;
                Ok(TrainMotifNetResult::new(
                    train_net_output.min_loss,
                    channel as u32,
                    motif_id_rate.0,
                    train_net_output.var_store,
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

        Ok(Arc::new(TrainMotifsNetsResult::new(select_motifs_run_config_ids, results)))
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

impl WithId for SyncMotifBatchInputWrapper {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
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
    fn encode_from_worker<C: IpcCodec>(self, all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError> {
        let Self { id, x, y } = self;
        let x = C::decode::<ThoTensor<Batch<Slice>, SynKey>>(&<ThoTensor<Batch<Slice>, SynKey> as WorkerStageOutput>::encode_from_worker::<C>(x, all_cuda_tensors, tho_device)?)?;
        let y = C::decode::<ThoTensor<Batch<MotifProbability>, SynKey>>(&<ThoTensor<Batch<MotifProbability>, SynKey> as WorkerStageOutput>::encode_from_worker::<C>(y, all_cuda_tensors, tho_device)?)?;
        C::encode(&Self { id, x, y })
    }
}

#[derive(Debug, Serialize)]
struct SyncMotifBatchIpcInputRef<'a> {
    id: &'a SynKey,
    motifs: &'a ThoTensor<MotifSlices, SynKey>,
    random_slices: &'a ThoTensor<MotifSlices, SynKey>,
    num_slices: i64,
    descr: &'a str,
    max_other_parallel: u8,
}

impl<'a> SyncMotifBatchIpcInputRef<'a> {
    fn new(input: &'a SyncMotifBatchInput<'_>, ctx: &RunCtxImpl) -> Self {
        Self {
            id: &input.id,
            motifs: input.motifs.as_ref(),
            random_slices: input.random_slices.as_ref(),
            num_slices: input.num_slices,
            descr: input.descr.as_str(),
            max_other_parallel: ctx.max_other_parallel(),
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
    max_other_parallel: u8,
}

#[derive(Debug)]
struct SyncMotifBatchIpcWrapper {
    stage: Arc<SyncMotifBatch>,
}

impl SyncStageIpcWrapper for SyncMotifBatchIpcWrapper {
    fn do_run(&self, input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError> {
        let input = JsonIpcCodec::decode::<SyncMotifBatchIpcInput>(input)?;
        let motifs = ThoTensor::ref_to_tensor(all_cuda_tensors, input.motifs)?;
        let random_slices = ThoTensor::ref_to_tensor(all_cuda_tensors, input.random_slices)?;
        let stage_input = SyncMotifBatchInput {
            id: input.id,
            motifs: ThoDataRef::Borrowed(&motifs),
            random_slices: ThoDataRef::Borrowed(&random_slices),
            num_slices: input.num_slices,
            descr: input.descr,
        };
        let mut ctx = RunCtxImpl::new(tho_device.tho_device_idx(), SYNC_MOTIF_BASTCH_ENDPOINT);
        ctx.set_max_other_parallel(input.max_other_parallel);
        let output = Arc::clone(&self.stage).do_run(&stage_input, &ctx)?;
        output.encode_from_worker::<JsonIpcCodec>(all_cuda_tensors, tho_device)
    }
}

#[derive(Debug, Proxy)]
#[proxy(
    common = "thoosa_common",
    name = "ProxySyncMotifBatch",
    endpoint = "SYNC_MOTIF_BASTCH_ENDPOINT",
    new_args = "tho_device_idx: usize",
    out_id = "input_id.clone()",
    run_input = "SyncMotifBatchIpcInputRef::new(input, ctx)",
    endpoint_init = "sync_motif_bastch_endpoint_init",
    wrapper = "SyncMotifBatchIpcWrapper"
)]
pub struct SyncMotifBatch;

impl SyncMotifBatch {
    fn new() -> Self {
        Self
    }
}

impl Adjustable for SyncMotifBatch {}

impl OutId for SyncMotifBatch {
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

impl Stage for SyncMotifBatch {
    type Ctx = RunCtxImpl;
    type In<'a> = SyncMotifBatchInput<'a>;
    type Out = SyncMotifBatchOutput;
}

impl SyncStage for SyncMotifBatch {
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SyncMotifBatch::do_run"))]
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        debug!("SyncMotifBatch::do_run: descr={} num_slices={}", input.descr, input.num_slices);

        let motifs = make_batch_with(input.motifs.t_ref(), input.num_slices, None)?;
        let random = make_batch_with(input.random_slices.t_ref(), input.num_slices, None)?;

        #[cfg(tho_debug_plots)]
        plot_spectrogram(&motifs.i((0, .., ..)), format!("/tmp/plots/spectrogram_motif_{}.png", input.id), 1200, 600)?;

        #[cfg(tho_debug_plots)]
        plot_spectrogram(&random.i((0, .., ..)), format!("/tmp/plots/spectrogram_random_{}.png", input.id), 1200, 600)?;

        let motif_plot_label = format!("motifs_batch_make_batch_{}_{}", input.id, input.descr);
        let random_plot_label = format!("motifs_batch_make_batch_random_{}_{}", input.id, input.descr);
        debug_plot!(&motifs.as_ref(), motif_plot_label, plot_repeats=11)?;
        debug_plot!(&random.as_ref(), random_plot_label, plot_repeats=11)?;

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

#[derive(Debug)]
enum SyncMotifBatchStage {
    Local(Arc<SyncMotifBatch>),
    Proxy(Arc<ProxySyncMotifBatch>),
}

#[derive(Debug)]
struct SyncMotifBatchDynStage<C> {
    stage: SyncMotifBatchStage,
    pd: PhantomData<C>,
}

impl<C> SyncMotifBatchDynStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    async fn new(tho_device_idx: usize) -> Result<Self, ThoError> {
        let stage = if GlobalCtx::global().is_ipc() {
            SyncMotifBatchStage::Proxy(Arc::new(ProxySyncMotifBatch::new(tho_device_idx).await?))
        } else {
            SyncMotifBatchStage::Local(Arc::new(SyncMotifBatch::new()))
        };
        Ok(Self { stage, pd: PhantomData })
    }
}

impl<C> Adjustable for SyncMotifBatchDynStage<C> where C: RunCtx + Adjustable + 'static {}

impl<C> OutId for SyncMotifBatchDynStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

impl<C> Stage for SyncMotifBatchDynStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = SyncMotifBatchInput<'a>;
    type Out = SyncMotifBatchOutput;
}

impl<C> SyncStage for SyncMotifBatchDynStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let mut stage_ctx = RunCtxImpl::new(ctx.tho_device_idx(), SYNC_MOTIF_BASTCH_ENDPOINT);
        stage_ctx.set_max_other_parallel(ctx.max_other_parallel());
        match &self.stage {
            SyncMotifBatchStage::Local(stage) => Arc::clone(stage).do_run(input, &stage_ctx),
            SyncMotifBatchStage::Proxy(stage) => Arc::clone(stage).do_run(input, &stage_ctx),
        }
    }
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

impl WithId for SplitMotifIntoTrainValidInputWrapper {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
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
    fn encode_from_worker<C: IpcCodec>(self, all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError> {
        let Self { id, train_motifs, valid_motifs, num_motif_slices } = self;
        let train_motifs = C::decode::<ThoTensor<MotifSlices, SynKey>>(&<ThoTensor<MotifSlices, SynKey> as WorkerStageOutput>::encode_from_worker::<C>(train_motifs, all_cuda_tensors, tho_device)?)?;
        let valid_motifs = C::decode::<ThoTensor<MotifSlices, SynKey>>(&<ThoTensor<MotifSlices, SynKey> as WorkerStageOutput>::encode_from_worker::<C>(valid_motifs, all_cuda_tensors, tho_device)?)?;
        C::encode(&Self { id, train_motifs, valid_motifs, num_motif_slices })
    }
}

#[derive(Debug, Serialize)]
struct SplitMotifIntoTrainValidIpcInputRef<'a> {
    id: &'a SynKey,
    motif: &'a ThoTensor<MotifSlices, SynKey>,
    max_other_parallel: u8,
}

impl<'a> SplitMotifIntoTrainValidIpcInputRef<'a> {
    fn new(input: &'a SplitMotifIntoTrainValidInput<'_>, ctx: &RunCtxImpl) -> Self {
        Self { id: &input.id, motif: input.motif.motif(), max_other_parallel: ctx.max_other_parallel() }
    }
}

#[derive(Debug, Deserialize)]
struct SplitMotifIntoTrainValidIpcInput {
    id: SynKey,
    motif: ThoTensor<MotifSlices, SynKey>,
    max_other_parallel: u8,
}

#[derive(Debug)]
struct SplitMotifIntoTrainValidIpcWrapper {
    stage: Arc<SplitMotifIntoTrainValid>,
}

impl SyncStageIpcWrapper for SplitMotifIntoTrainValidIpcWrapper {
    fn do_run(&self, input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError> {
        let input = JsonIpcCodec::decode::<SplitMotifIntoTrainValidIpcInput>(input)?;
        let motif = ThoTensor::ref_to_tensor(all_cuda_tensors, input.motif)?;
        let stage_input = SplitMotifIntoTrainValidInput { id: input.id, motif: SplitMotifTensorRef::Plain(ThoDataRef::Borrowed(&motif)) };
        let mut ctx = RunCtxImpl::new(tho_device.tho_device_idx(), SPLIT_MOTIF_INTO_TRAIN_VALID_ENDPOINT);
        ctx.set_max_other_parallel(input.max_other_parallel);
        let output = Arc::clone(&self.stage).do_run(&stage_input, &ctx)?;
        output.encode_from_worker::<JsonIpcCodec>(all_cuda_tensors, tho_device)
    }
}

#[derive(Debug, Proxy)]
#[proxy(
    common = "thoosa_common",
    name = "ProxySplitMotifIntoTrainValid",
    endpoint = "SPLIT_MOTIF_INTO_TRAIN_VALID_ENDPOINT",
    new_args = "tho_device_idx: usize",
    out_id = "input_id.clone()",
    run_input = "SplitMotifIntoTrainValidIpcInputRef::new(input, ctx)",
    endpoint_init = "split_motif_into_train_valid_endpoint_init",
    wrapper = "SplitMotifIntoTrainValidIpcWrapper"
)]
pub struct SplitMotifIntoTrainValid;

impl SplitMotifIntoTrainValid {
    fn new() -> Self {
        Self
    }
}

impl Adjustable for SplitMotifIntoTrainValid {}

impl OutId for SplitMotifIntoTrainValid {
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

impl Stage for SplitMotifIntoTrainValid {
    type Ctx = RunCtxImpl;
    type In<'a> = SplitMotifIntoTrainValidInput<'a>;
    type Out = SplitMotifIntoTrainValidOutput;
}

impl SyncStage for SplitMotifIntoTrainValid {
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

#[derive(Debug)]
enum SplitMotifIntoTrainValidStage {
    Local(Arc<SplitMotifIntoTrainValid>),
    Proxy(Arc<ProxySplitMotifIntoTrainValid>),
}

#[derive(Debug)]
struct SplitMotifIntoTrainValidDynStage<C> {
    stage: SplitMotifIntoTrainValidStage,
    pd: PhantomData<C>,
}

impl<C> SplitMotifIntoTrainValidDynStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    async fn new(tho_device_idx: usize) -> Result<Self, ThoError> {
        let stage = if GlobalCtx::global().is_ipc() {
            SplitMotifIntoTrainValidStage::Proxy(Arc::new(ProxySplitMotifIntoTrainValid::new(tho_device_idx).await?))
        } else {
            SplitMotifIntoTrainValidStage::Local(Arc::new(SplitMotifIntoTrainValid::new()))
        };
        Ok(Self { stage, pd: PhantomData })
    }
}

impl<C> Adjustable for SplitMotifIntoTrainValidDynStage<C> where C: RunCtx + Adjustable + 'static {}

impl<C> OutId for SplitMotifIntoTrainValidDynStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

impl<C> Stage for SplitMotifIntoTrainValidDynStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = SplitMotifIntoTrainValidInput<'a>;
    type Out = SplitMotifIntoTrainValidOutput;
}

impl<C> SyncStage for SplitMotifIntoTrainValidDynStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let mut stage_ctx = RunCtxImpl::new(ctx.tho_device_idx(), SPLIT_MOTIF_INTO_TRAIN_VALID_ENDPOINT);
        stage_ctx.set_max_other_parallel(ctx.max_other_parallel());
        match &self.stage {
            SplitMotifIntoTrainValidStage::Local(stage) => Arc::clone(stage).do_run(input, &stage_ctx),
            SplitMotifIntoTrainValidStage::Proxy(stage) => Arc::clone(stage).do_run(input, &stage_ctx),
        }
    }
}


#[derive(Debug)]
pub struct MotifsBatch<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
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
    sync_motif_batch: Arc<SyncMotifBatchDynStage<C>>,
}

impl<L, C> MotifsBatch<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable,
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
        let sync_motif_batch = Arc::new(SyncMotifBatchDynStage::new(ctx.tho_device_idx()).await?);
        let motif_meta = track_await(motif.extra()).await?;
        let split_motif = Arc::new(SplitMotifIntoTrainValidDynStage::<C>::new(ctx.tho_device_idx()).await?);
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

        // TODO(perf): asynchronously prefetch the next random-slice epoch before this cache misses so RandomSlicesLoader latency is hidden behind TrainStep work.
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
    C: RunCtx + Adjustable,
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
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Arc;
    use async_trait::async_trait;
    use log::debug;
    use log::LevelFilter::{Debug, Trace};
    use parking_lot::{Mutex, RwLock};
    use serde::{Deserialize, Serialize};
    use tch::{Device, IndexOp, Kind, Reduction, Tensor};
    use tch::Device::Cuda;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::Kind::Float;
    use tch::nn::{AdamW, OptimizerConfig, VarStore};
    use tokio::fs;
    use thoosa_common::batches::BatchLengths;
    use thoosa_common::builder::{Node, SeqBuilder};
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::common::mock_run_ctx;
    use thoosa_common::runner::common::{MaxParallelCtx, RunCtx};
    use thoosa_common::runner::common::runnable::{AsyncStage, DynSyncStage, DynSyncStageBox, SyncStage};
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::dims::{Batch, Dims, LabelProbability, LabelsProbabilities, MotifProbability, Slice, ThoBatch};
    use crate::tasks::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::tch_test_utils;
    use thoosa_common::train_nets::{final_net, CommonTrainingRunConfig, CommonTrainingRunConfigFields, MotifConvNetArgsFields, MotifCosineNetArgsFields, MotifLinNetArgs, MotifNetParams, MotifNetTypeArgs, MotifNetTypeArgsFields, MotifsBy, OptConfig, ThoModuleWithStore, TrainMotifNetRunConfig, TrainMotifNetRunConfigFields, TrainMotifsNetsResult, FinalNetParams};
    use thoosa_common::debug_plot;
    use thoosa_common::utils::{reset_print_options, test_net};
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::{OptRunConfigs, RunConfigId};
    use thoosa_shared::{cdt_for_dir, mono_key, simple_id, TensorIdSliceIdx, ThoError};
    use thoosa_shared::traits::{MonoMetrics, Named, SimpleId};
    use thoosa_shared::hyperparameter::Hyperparameter;
    use thoosa_shared::val::{NamedRange, NamedVal, Val};
    use thoosa_store::{MockStoreBuilder, RunConfigRow};
    use thoosa_common::tch_test_utils::resource_tensor;
    use thoosa_shared::test_utils::MotifKind;
    use thoosa_common::dims::MotifSlices;
    use thoosa_common::ipc::codec::{IpcCodec, JsonIpcCodec};
    use thoosa_common::ipc::cuda_wokrer::CudaWorker;
    use thoosa_common::ipc::json_ser::SyncStageIpcWrapper;
    use thoosa_common::ipc::tests::run_ipc_test;
    use thoosa_common::my_tch::MotifInitModeFields;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runner::purge::FuturesRunnerMapPurger;
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use thoosa_common::tho_device::ThoDevice;
    use thoosa_common::tho_tensor::{ThoTensor, ThoTensorInner};
    use thoosa_proxy_derive::Proxy;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use crate::tasks::motif_loader::{sync_load_motif_resources_endpoint_init, sync_motif_loader_endpoint_init, MotifMeta, SYNC_LOAD_MOTIF_RESOURCES_ENDPOINT, SYNC_MOTIF_LOADER_ENDPOINT};
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use crate::tasks::train_motif_nets::{split_motif_into_train_valid_endpoint_init, sync_motif_bastch_endpoint_init, train_motif_step_endpoint_init, train_motif_step_endpoint_inits, TrainMotifNetInput, TrainMotifNets, TrainMotifStepProxyFactory, SPLIT_MOTIF_INTO_TRAIN_VALID_ENDPOINT, SYNC_MOTIF_BASTCH_ENDPOINT, TRAIN_MOTIF_STEP_ENDPOINT, TrainMotifStepIpcWrapper, TrainMotifStepIpcInitStage, TrainMotifStepIpcRequest, TrainMotifStepWeightsSync};
    use crate::tasks::train_net::{TrainStep, TrainStepDynSyncStage, TrainStepInput, TrainStepOutput, TrainStepProxyFactory, TrainStepWeightsSync, TrainStepXFn};

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_train_motif_nets() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
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
        )?);

        let input = TrainMotifNetInput::new(MotifsBy::SelectMotifsRunIds { select_motifs_run_config_ids: vec![select_motifs_run_config_id] }, Arc::new(OptConfig::new(3, 2)));

        let mut run_ctx = mock_run_ctx::default(1);
        let run_ctx_branch = run_ctx.branch(false);
        let r = train_motif_net.run(input, run_ctx_branch).await?;

        debug!("test_train_motif_nets: r={r:?}");

        fn run_test_net(params: &MotifNetParams, vs: &VarStore) -> Result<(), ThoError> {
            let net = params.motif_net(None)?.build_tch_module::<MotifSlices, MotifProbability>(&vs.root())?;

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

    #[test]
    fn test_train_motif_nets_ipc() -> Result<(), ThoError> {
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(async {
            let module = module_path!().split_once("::").map(|(_, module)| module).unwrap_or("");
            let test_name = format!("{module}::test_train_motif_nets_ipc");
            GlobalCtx::init_devices_ipc(2, vec![test_name, "--exact".to_string(), "--nocapture".to_string()]).await?;
            let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, true);
            let cuda_actor = GlobalCtx::global().cuda_actor(1);
            let result = run_train_motif_nets().await?;
            _ = cuda_actor.shutdown().await;

            Ok(())

        }, async |sock_path| { CudaWorker::run_worker(sock_path, train_motif_step_endpoint_inits(), ThoDevice::from(Cuda(0))).await })
    }

    #[test]
    fn test_train_step_ipc() -> Result<(), ThoError> {
        pub(crate) const TRAIN_TEST_STEP_ENDPOINT: &str = "TrainTestStep";

        #[derive(Debug, Serialize, Deserialize)]
        struct TrainTestStepInit {

        }
        #[derive(Debug)]
        struct TrainTestStepIpcWrapper {
            stage: Arc<TrainTestStepProxyTarget>,
        }
        impl SyncStageIpcWrapper for TrainTestStepIpcWrapper {
            fn do_run(&self, input: &[u8], all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, _tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError> {
                debug!("TrainTestStepIpcWrapper::do_run:input={:?}", String::from_utf8_lossy(input));
                let input_dec = JsonIpcCodec::decode::<TrainStepInput<InDims, OutDims>>(input)?;

                let x = ThoTensor::ref_to_tensor(all_cuda_tensors, input_dec.x)?;
                let expected_y = ThoTensor::ref_to_tensor(all_cuda_tensors, input_dec.expected_y)?;

                let input = TrainStepInput::new(x, input_dec.lengths, expected_y, true);

                Ok(JsonIpcCodec::encode(&Arc::clone(&self.stage.base).do_run(&input, &RunCtxImpl::new(1, TRAIN_MOTIF_STEP_ENDPOINT))?)?)
            }
        }

        #[derive(Debug, Serialize, Deserialize)]
        struct  TrainTestStepIpcRequest {

        }

        impl TrainTestStepIpcRequest {
            fn new ()->TrainTestStepIpcRequest {
                TrainTestStepIpcRequest{

                }
             }
        }

        #[derive(Debug, Serialize, Deserialize, Clone)]
        pub struct InDims {}
        impl Dims for InDims{ type Metrics = MonoMetrics<0, 1>; }

        #[derive(Debug, Serialize, Deserialize, Clone)]
        pub struct OutDims {}
        impl Dims for OutDims{ type Metrics = MonoMetrics<0, 1>; }

        #[derive(Debug)]
        struct TestNetParams {
            in_features:u32,
            layers:usize
        }

        #[derive(Debug)]
        struct TestRunConfig {
            lr:f64
        }

        #[derive(Debug, Proxy)]
        #[proxy(
            common = "thoosa_common",
            name = "ProxyTrainTestStep",
            endpoint = "TRAIN_TEST_STEP_ENDPOINT",
            new_args = "init: TrainTestStepInit, tho_device_idx: usize",
            init = "init",
            out_id = "()",
            endpoint_init = "train_test_step_endpoint_init",
            wrapper = "TrainTestStepIpcWrapper"

        )]
        struct TrainTestStepProxyTarget {
            base: Arc<TrainStep<Box<dyn Fn(ThoBatch<InDims>) -> ThoBatch<InDims> + Send + Sync + 'static>, InDims, OutDims, RunCtxImpl>>,
        }

        impl TrainTestStepProxyTarget{
            pub fn new()->Self {
                let run_config = TestRunConfig{ lr: 0.001 };

                let (train_step, params) : (TrainStep<Box<dyn Fn<_, Output = _> + std::marker::Send + Sync>, _, _, _>, TestNetParams)=  futures::executor::block_on(TrainStep::new_local(
                    Cuda(0).into(),
                    |vs|  {
                        async move {
                            let params = TestNetParams { in_features: 100, layers: 2 };
                            let net = test_net(&params)?.build_tch_module::<Batch<InDims>, Batch<OutDims>>(&vs.lock().root())?;

                            Ok((net, params))
                        }
                    },
                    |rc, vs|  {
                        AdamW::default().build(&vs, rc.lr).map_err(|e|e.into())
                    },
                    Arc::new(Box::new(|t|{
                        t
                    }) as Box<dyn Fn(ThoBatch<InDims>) -> ThoBatch<InDims> + Send + Sync + 'static>),
                    |pred_y: &Tensor, target: &Tensor|{
                        Ok(pred_y.f_binary_cross_entropy_with_logits::<Tensor>(&target.f_totype(Float)?, None, None, Reduction::Mean)?)
                    },
                    &run_config,
                )).unwrap();

                Self {
                    base: Arc::new(train_step),
                }
            }
        }

        pub fn test_net(params: &TestNetParams) -> Result<Node, ThoError> {
            debug!("final_node: params={:?}", params);

            let mut builder = SeqBuilder::new(params.in_features, None)?;
            let mut net = builder
                .closure(
                    Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                        debug!("final_net: input={t:?}");
                        Ok(t.shallow_clone())
                    })),
                    None,
                )?
                .linear_s(
                    2 * params.in_features,
                    "final_net_first_layer".to_string(),
                )?
                .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?;

            for i in 0..params.layers {
                net = net
                    .linear_s(
                        params.in_features,
                        format!("final_net_{i}th_layer"),
                    )?
                    .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?;
            }

            net
                // .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
                .linear_s(
                    1,
                    "final_net_last_layer",
                )?
                .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?
                .end_seq()
        }


        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(async {
            let module = module_path!().split_once("::").map(|(_, module)| module).unwrap_or("");
            let test_name = format!("{module}::test_train_step_ipc");
            GlobalCtx::init_devices_ipc(2, vec![test_name, "--exact".to_string(), "--nocapture".to_string()]).await?;
            let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, true);
            let cuda_actor = GlobalCtx::global().cuda_actor(1);

            let ipc_ref = cuda_actor.copy(ThoTensor::new((), tch::Tensor::rand(&[100], FLOAT_CPU))?, 1.into()).await?;
            let ThoTensorInner::Ref{ key, size, kind, selector, tho_device_idx } = ipc_ref.inner  else {
                panic!("should be ref")
            };

            let proxy = Arc::new(ProxyTrainTestStep::new(
                TrainTestStepInit{},
                1
            ).await?);

            #[derive(Debug)]
            struct TestWeightSync{}
            #[async_trait]
            impl TrainStepWeightsSync for TestWeightSync {
                async fn into_var_store(&self) -> Result<VarStore, ThoError> {
                    Ok(VarStore::new(Cuda(0)))
                }
            }
            let (train_step, params):
                (TrainStepDynSyncStage<TrainStepXFn<InDims>, InDims, OutDims, RunCtxImpl>, TestNetParams)
                = (
                TrainStepDynSyncStage::Proxy {
                    stage: Arc::new(DynSyncStageBox::new(proxy as Arc<dyn DynSyncStage<TrainStepInput<InDims, OutDims>, TrainStepOutput,RunCtxImpl>>)),
                    weights_sync: Arc::new(TestWeightSync{}) as Arc<dyn TrainStepWeightsSync>
                },
                TestNetParams { in_features: 100, layers: 2 }
            );

            let train_step = Arc::new(train_step);
            GlobalCtx::global().register_cache(("train_step", Arc::clone(&train_step) as Arc<dyn FuturesRunnerMapPurger>));


            let input = TrainStepInput::<InDims, OutDims>::new(
                to_tho_tensor(cuda_actor.copy(ThoTensor::new((), tch::Tensor::rand(&[3, 100], FLOAT_CPU))?, 1.into()).await?.inner, "2".into()),
                None,
                to_tho_tensor(cuda_actor.copy(ThoTensor::new((), tch::Tensor::rand(&[3, 1], FLOAT_CPU))?, 1.into()).await?.inner, "3".into()),
                false
            );
            let ctx = RunCtxImpl::new(1, "x");

            let r = Arc::clone(&train_step).do_run(&input, &ctx)?;

            GlobalCtx::global().dump_all_to_cpu(&HashMap::new(), 1, "test").await?;


            let r = train_step.do_run(&input, &ctx)?;

            debug!("test_train_step_ipc: r={r:?}");


            Ok(())
        }, async |sock_path| {
            CudaWorker::run_worker(
                sock_path,
                HashMap::from([
                    (
                        TRAIN_TEST_STEP_ENDPOINT.to_string(),
                        Box::new(train_test_step_endpoint_init) as Box<dyn Fn(Vec<u8>) -> Result<Box<dyn SyncStageIpcWrapper>, ThoError>>,
                    ),

                ])
                ,
                ThoDevice::from(Cuda(0)),
            ).await
        })


    }

    fn to_tho_tensor<D>(inner: ThoTensorInner, id: String) -> ThoTensor<D, SimpleId>
    where
        D:Dims
    {
        let ThoTensorInner::Ref{ key, size, kind, selector, tho_device_idx } = inner  else {
            panic!("should be ref")
        };
        ThoTensor::new_ref(simple_id!(id), key, size.clone(), kind, None, tho_device_idx)
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
