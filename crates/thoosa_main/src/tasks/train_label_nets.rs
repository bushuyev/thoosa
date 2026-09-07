use async_trait::async_trait;
use cached_derive::Cached;
use futures::FutureExt;
use futures::Stream;
use futures_util::future::{BoxFuture, try_join_all};
use futures_util::stream::BoxStream;
use futures_util::{StreamExt, TryStreamExt};
use log::{debug, error, info, trace, warn};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::future::ready;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use tch::Device::{Cpu, Cuda};
use tch::nn::{Optimizer, VarStore};
use tch::{Reduction, Tensor};
use thoosa_common::data_with_extra::DataWithExtra;
use thoosa_common::dims::ChannelsData;
use thoosa_common::ipc::codec::{IpcCodec, JsonIpcCodec};
use thoosa_common::ipc::cuda_wokrer::{AllTensors, InitFn, WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use thoosa_common::ipc::json_ser::SyncStageIpcWrapper;
use thoosa_common::ipc::wraps::{WorkerWrapper, WorkerWrapperWithState};
use thoosa_common::ipc::proxy::{DynSyncStageEnum, DynSyncStageWithStateEnum, ProxyStage};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::common::cachable::{BorrowRef, Cached, DevicesWrap, SelectRef, ThoDataRef};
use thoosa_common::runner::oom_handler::{run_ctx_adjuster, Adjusters, NoContexts, NoInputAdjuster, NoRunCtxAdjuster, NoTaskAdjuster, OOMHandler, WithRunCtxContext};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::resource::{Resource, ResourcesConfig, build_resource_maps};
use thoosa_shared::traits::MaxParallelCtx;
use thoosa_shared::traits::{Adjustable, AsyncStage, Data, DynSyncStageBox, OutId, RunCtx, Stage, SyncStage, WithId};

use mockall_double::double;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;
use crate::tasks::apply_cosine2d_motif_nets::ApplyCosine2DMotifNets;
use crate::tasks::apply_motif_nets::ApplyMotifNets;
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::stage_ids::RAW_TENSOR_LOADER;
use crate::tasks::stft::SyncStftSelector;
use crate::tasks::train_net::{TrainNet, TrainNetInput, TrainNetOutput, TrainStepFactory};
use crate::tasks::train_step::{TrainStep, TrainStepXFn};
use rand::rng;
use thoosa_common::batches::{AsyncDataBatch, BatchLengths, prefetch_next_batch_stream, preload_one};
use thoosa_common::cuda_utils::cuda_mem;
use thoosa_common::dims::{Batch, Dims, LabelProbability, MotifProbability, ThoBatch};
use thoosa_common::ipc::worker_msg::{WorkerInput, WorkerOutput};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjuster, MAX_PARALLEL, MaxParallelStep, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelAdjustContextImpl};
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_common::runs::{Labeled, NetResources, SavableResult};
use thoosa_common::stages_seq::ABStage;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_varstore::{ModConfig, ThoTrainModule};
use thoosa_common::train_nets::{CommonTrainingRunConfig, ConvNetArgs, ConvNetParams, LabelNetParams, LabelNetTypeArgs, LabelNetTypeParams, LinLenNetParams, LinNetArgs, OptConfig, SelfAttentionNetArgs, SelfAttentionNetParams, TrainLabelNetRunConfig, TrainLabelNetRunConfigFields, TrainLabelNetRunConfigResult, TrainLabelNetRunConfigResultParams, conv_label_node, lin_label_node, opt_train, self_attention_label_node};
use thoosa_common::utils::{ListenerResponse, TrainListener, mix_x_binary_with_labels, pad_stack_with_mask};
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::run_configs::RunConfig;
use thoosa_shared::run_configs::{OptRunConfigs, RunConfigId};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_shared::{CopyToDevice, FuturesOrdering, ThoError, collect_futures_with_progress, mono_key, shuffle_and_chunk_with_rng};
use thoosa_store::Store;
use tokio::{join, try_join};
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};

pub const SYNC_LABEL_BATCH_ENDPOINT: &str = "SyncLabelBatch";
pub(crate) const TRAIN_LABEL_STEP_ENDPOINT: &str = "TrainLabelStep";

type AppliedMotifOutput = DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>;

#[derive(Debug)]
pub struct SyncLabelBatchInput<'a> {
    id: SynKey,
    x_1: Vec<ThoDataRef<'a, AppliedMotifOutput>>,
    x_0: Vec<ThoDataRef<'a, AppliedMotifOutput>>,
    num_nets: i64,
    label: String,
}

impl Data for SyncLabelBatchInput<'_> {
    type Metrics = i64;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        (self.x_1.len() + self.x_0.len()) as i64
    }

    fn device(&self) -> usize {
        self.x_1.iter().chain(self.x_0.iter()).next().map(|value| value.as_ref().device()).unwrap_or(0)
    }

    fn mem_size(&self) -> u64 {
        self.x_1.iter().chain(self.x_0.iter()).map(|value| value.as_ref().mem_size()).sum()
    }
}

impl WithId for SyncLabelBatchInput<'_> {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Input for SyncLabelBatchInput<'_> {}

#[derive(Debug)]
struct SyncLabelBatchInputWrapper {
    id: SynKey,
    x_1: Vec<Arc<DevicesWrap<AppliedMotifOutput>>>,
    x_0: Vec<Arc<DevicesWrap<AppliedMotifOutput>>>,
    num_nets: i64,
    label: String,
}

impl SyncLabelBatchInputWrapper {
    fn new(x_1: Vec<Arc<DevicesWrap<AppliedMotifOutput>>>, x_0: Vec<Arc<DevicesWrap<AppliedMotifOutput>>>, num_nets: i64, label: String) -> Self {
        Self { id: SynKey::new([mono_key!(label.clone(), "SyncLabelBatch")]), x_1, x_0, num_nets, label }
    }
}

impl BorrowRef for SyncLabelBatchInputWrapper {
    type Out<'a> = SyncLabelBatchInput<'a> where Self: 'a;

    fn borrow_to<'a>(&'a self, to_device_idx: usize) -> impl Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        async move {
            let x_1 = try_join_all(self.x_1.iter().map(|value| BorrowRef::borrow_to(value, to_device_idx))).await?;
            let x_0 = try_join_all(self.x_0.iter().map(|value| BorrowRef::borrow_to(value, to_device_idx))).await?;
            Ok(SyncLabelBatchInput { id: self.id.clone(), x_1, x_0, num_nets: self.num_nets, label: self.label.clone() })
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SyncLabelBatchOutput {
    id: SynKey,
    x: ThoTensor<Batch<MotifProbability>, SynKey>,
    lengths: Vec<i64>,
    y: ThoTensor<Batch<LabelProbability>, SynKey>,
}

impl SyncLabelBatchOutput {
    fn into_parts(self) -> (ThoBatch<MotifProbability>, Option<BatchLengths>, ThoBatch<LabelProbability>) {
        (
            self.x.with_id(thoosa_shared::simple_id!("sync_label_batch_x")),
            Some(BatchLengths::new(self.lengths)),
            self.y.with_id(thoosa_shared::simple_id!("sync_label_batch_y")),
        )
    }
}

impl WithId for SyncLabelBatchOutput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for SyncLabelBatchOutput {}

impl WorkerStageOutput for SyncLabelBatchOutput {
    fn encode_from_worker(self, all_cuda_tensors: &AllTensors, tho_device: &ThoDevice) -> Result<Self, ThoError> {
        let Self { id, x, lengths, y } = self;
        // Convert tensor fields first so the wrapper can encode this output exactly once.
        let x = x.encode_from_worker(all_cuda_tensors, tho_device)?;
        let y = y.encode_from_worker(all_cuda_tensors, tho_device)?;
        Ok(Self { id, x, lengths, y })
    }
}

#[derive(Debug, Serialize)]
struct SyncLabelBatchIpcInputRef<'a> {
    id: &'a SynKey,
    x_1: Vec<&'a ThoTensor<MotifProbability, SynKey>>,
    x_1_lengths: Vec<&'a Vec<i64>>,
    x_0: Vec<&'a ThoTensor<MotifProbability, SynKey>>,
    x_0_lengths: Vec<&'a Vec<i64>>,
    num_nets: i64,
    label: &'a str,
}

impl<'a> SyncLabelBatchIpcInputRef<'a> {
    fn new(input: &'a SyncLabelBatchInput<'_>) -> Self {
        Self {
            id: &input.id,
            x_1: input.x_1.iter().map(|value| value.as_ref().data()).collect(),
            x_1_lengths: input.x_1.iter().map(|value| value.as_ref().extra()).collect(),
            x_0: input.x_0.iter().map(|value| value.as_ref().data()).collect(),
            x_0_lengths: input.x_0.iter().map(|value| value.as_ref().extra()).collect(),
            num_nets: input.num_nets,
            label: input.label.as_str(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct SyncLabelBatchIpcInput {
    id: SynKey,
    x_1: Vec<ThoTensor<MotifProbability, SynKey>>,
    x_1_lengths: Vec<Vec<i64>>,
    x_0: Vec<ThoTensor<MotifProbability, SynKey>>,
    x_0_lengths: Vec<Vec<i64>>,
    num_nets: i64,
    label: String,
}

impl Serialize for SyncLabelBatchInput<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        SyncLabelBatchIpcInputRef::new(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SyncLabelBatchInput<'static> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let input = SyncLabelBatchIpcInput::deserialize(deserializer)?;
        if input.x_1.len() != input.x_1_lengths.len() || input.x_0.len() != input.x_0_lengths.len() {
            return Err(serde::de::Error::custom("SyncLabelBatch tensor and length metadata counts differ"));
        }
        let x_1 = input.x_1.into_iter().zip(input.x_1_lengths).map(|(tensor, lengths)| ThoDataRef::Owned(DataWithExtra::new(tensor, lengths))).collect();
        let x_0 = input.x_0.into_iter().zip(input.x_0_lengths).map(|(tensor, lengths)| ThoDataRef::Owned(DataWithExtra::new(tensor, lengths))).collect();
        Ok(Self { id: input.id, x_1, x_0, num_nets: input.num_nets, label: input.label })
    }
}

#[typetag::serde(name = "SyncLabelBatchInput")]
impl WorkerInput for SyncLabelBatchInput<'static> {}

impl WorkerStageInput for SyncLabelBatchInput<'_> {
    fn clone_refs(&self) -> Result<Box<dyn WorkerInput>, ThoError> {
        let clone_value = |value: &AppliedMotifOutput| -> Result<AppliedMotifOutput, ThoError> { Ok(DataWithExtra::new(value.data().clone_ref()?, value.extra().clone())) };
        // Convert every borrowed cached value into an owned worker-reference payload.
        let x_1 = self.x_1.iter().map(|value| Ok(ThoDataRef::Owned(clone_value(value.as_ref())?))).collect::<Result<Vec<_>, ThoError>>()?;
        let x_0 = self.x_0.iter().map(|value| Ok(ThoDataRef::Owned(clone_value(value.as_ref())?))).collect::<Result<Vec<_>, ThoError>>()?;
        Ok(Box::new(SyncLabelBatchInput { id: self.id.clone(), x_1, x_0, num_nets: self.num_nets, label: self.label.clone() }))
    }

    fn ref_to_tensor(&self, all_cuda_tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {
        let materialize = |value: &AppliedMotifOutput| -> Result<AppliedMotifOutput, ThoError> { Ok(DataWithExtra::new(WorkerStageInput::ref_to_tensor(value.data(), all_cuda_tensors, remove)?, value.extra().clone())) };
        let x_1 = self.x_1.iter().map(|value| Ok(ThoDataRef::Owned(materialize(value.as_ref())?))).collect::<Result<Vec<_>, ThoError>>()?;
        let x_0 = self.x_0.iter().map(|value| Ok(ThoDataRef::Owned(materialize(value.as_ref())?))).collect::<Result<Vec<_>, ThoError>>()?;
        Ok(Self { id: self.id.clone(), x_1, x_0, num_nets: self.num_nets, label: self.label.clone() })
    }

}

#[derive(Debug)]
pub struct SyncLabelBatch<C>
where
    C: RunCtx + 'static
{
    pd:PhantomData<C>
}

impl <C> SyncLabelBatch<C>
where
    C: RunCtx + 'static
{
    pub async fn new(device: ThoDevice) -> Result<DynSyncStageEnum<SyncLabelBatch<C>>, ThoError> {
        if GlobalCtx::global().is_ipc() {
            let stage_id = 0;
            Ok(DynSyncStageEnum::Proxy {
                stage: Arc::new(ProxyStage::proxy(SYNC_LABEL_BATCH_ENDPOINT, stage_id, device.tho_device_idx(), Box::new(())).await?),
                init: (),
            })
        } else {
            Ok(DynSyncStageEnum::Local(Arc::new(Self::new_local().await?)))
        }
    }

    pub async fn new_local() -> Result<Self, ThoError>{
        Ok(Self {
            pd: PhantomData,
        })

    }
}

impl <C> Adjustable for SyncLabelBatch<C>
where
    C: RunCtx + 'static
{}

impl <C> OutId for SyncLabelBatch<C>
where
    C: RunCtx + 'static
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

impl <C> Stage for SyncLabelBatch<C>
where
    C: RunCtx + 'static
{
    type Ctx = C;
    type In<'a> = SyncLabelBatchInput<'a>;
    type Out = SyncLabelBatchOutput;
}

impl <C> SyncStage for SyncLabelBatch<C>
where
    C: RunCtx + 'static
{
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        debug!("SyncLabelBatch::do_run: input={input:?}");
        let x_1 = input.x_1.iter().map(ThoDataRef::as_ref).collect::<Vec<_>>();
        let x_0 = input.x_0.iter().map(ThoDataRef::as_ref).collect::<Vec<_>>();

        if x_1.is_empty() || x_0.is_empty() {
            return Err(ThoError::new(format!("SyncLabelBatch: empty positive or negative resources for {}", input.label)));
        }

        let max_input = x_1.iter()
            .chain(x_0.iter())
            .map(|value| value.data().size().get(1).copied().ok_or_else(|| ThoError::new(format!("SyncLabelBatch: invalid motif output shape for {}", input.label))))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .ok_or_else(|| ThoError::new(format!("batch resources are empty for {}", input.label)))?;

        let x_1_refs = x_1.iter().map(|value| value.data()).collect::<Vec<_>>();
        let x_0_refs = x_0.iter().map(|value| value.data()).collect::<Vec<_>>();

        let (x_1_batch, lengths_1) = pad_stack_with_mask(x_1_refs.as_slice(), input.num_nets, max_input)?;
        let (x_0_batch, lengths_0) = pad_stack_with_mask(x_0_refs.as_slice(), input.num_nets, max_input)?;

        let (x, lengths, y) = mix_x_binary_with_labels(&x_1_batch, Some(&BatchLengths::new(lengths_1)), &x_0_batch, Some(&BatchLengths::new(lengths_0)))?;
        let lengths = lengths.ok_or_else(|| ThoError::new("SyncLabelBatch: mixed batch did not return lengths"))?;

        debug!("SyncLabelBatch::do_run: lengths={lengths:?} x={x:?} y={y:?}");

        Ok(SyncLabelBatchOutput {
            id: input.id.clone(),
            x: ThoTensor::new(SynKey::new([mono_key!("x", "SyncLabelBatch")]), x)?,
            lengths: lengths.as_slice().to_vec(),
            y: ThoTensor::new(SynKey::new([mono_key!("y", "SyncLabelBatch")]), y)?,
        })
    }
}

#[typetag::serde]
impl WorkerOutput for SyncLabelBatchOutput{}


pub fn sync_label_batch_endpoint_init<'a>(init: Box<dyn WorkerInput>, ws: Arc<RwLock<Box<dyn Workspace>>>, all_cuda_tensors: &'a AllTensors, tho_device:&'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        // WorkerStageInput reconstructs owned tensor-plus-length values for the borrowed stage API.
        Ok(Box::new(WorkerWrapper::<JsonIpcCodec, _, _, _>::new(Arc::new(SyncLabelBatch::<WokerRunCtx>::new_local().await?), PhantomData)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}

#[derive(Debug)]
pub struct TrainLabelNets<L, LM, C>
where
    L: for <'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx=C> + Cached + OutId + 'static,
    LM: for <'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>>>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    train_motif_nets_run_config_ids: Vec<RunConfigId>,
    motif_nets_ids: Vec<RunConfigId>,
    run_configs: Arc<OptRunConfigs<
        Vec<NamedVal<TrainLabelNetRunConfigFields>>,
        Vec<NamedRange<TrainLabelNetRunConfigFields>>,
        Vec<Hyperparameter<TrainLabelNetRunConfigFields>>,
        TrainLabelNetRunConfigFields,
    >>,
    pd: PhantomData<L>,
    applied_motif_loader: Arc<LM>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    num_nets:u32,
    store: Arc<Box<dyn Store>>,
    opt_config:Arc<OptConfig>,
    labels: Option<Vec<String>>,
    sync_label_batch:Arc<DynSyncStageEnum<SyncLabelBatch<C>>>

}

pub trait AppliedMotifNetsStage<C>: for<'i> AsyncStage<In<'i> = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Out = Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>>>, Ctx = C> + Cached + OutId + 'static
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    fn num_nets(&self) -> usize;
    fn motif_nets_ids(&self) -> Vec<u32>;
}

impl<C> AppliedMotifNetsStage<C> for ApplyMotifNets<C>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    fn num_nets(&self) -> usize {
        self.num_nets()
    }

    fn motif_nets_ids(&self) -> Vec<u32> {
        self.motif_nets_ids()
    }
}

impl<C> AppliedMotifNetsStage<C> for ApplyCosine2DMotifNets<C>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    fn num_nets(&self) -> usize {
        self.num_nets()
    }

    fn motif_nets_ids(&self) -> Vec<u32> {
        self.motif_nets_ids()
    }
}

pub async fn new_train_label_nets<L, AMN, C>(
    train_motif_net_run_config_ids: Vec<RunConfigId>,
    loader: Arc<L>,
    apply_motif_nets: Arc<AMN>,
    run_configs: OptRunConfigs<Vec<NamedVal<TrainLabelNetRunConfigFields>>, Vec<NamedRange<TrainLabelNetRunConfigFields>>, Vec<Hyperparameter<TrainLabelNetRunConfigFields>>, TrainLabelNetRunConfigFields>,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    opt_config:Arc<OptConfig>,
    labels: Option<Vec<String>>,
    ) -> Result<TrainLabelNets<L, ABStage<L, AMN, C>, C>, ThoError>
where
    L: for <'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx=C> + Cached + OutId + 'static,
    AMN: AppliedMotifNetsStage<C>,
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    let id = mono_key!(RAW_TENSOR_LOADER, "stage") + &SynKey::new(train_motif_net_run_config_ids.iter().map(|run_config_id| mono_key!(*run_config_id, format!("train_motif_nets_run_config_ids_{run_config_id}"))).collect::<Vec<_>>());

    let num_nets = apply_motif_nets.num_nets() as u32;
    if num_nets == 0 {
        return Err(ThoError::new("No motifs nets to train on"));
    }
    let motif_nets_ids = apply_motif_nets.motif_nets_ids();


    let applied_motif_loader = Arc::new(ABStage::new(
        //train_motif_nets_run_config_ids are only for logging. can't get RawTensorLoader id, but it's a singleton and ABStage isn't used - A instances are cached under A::out_id key.
        id,
        loader,
        apply_motif_nets
    ));

    track_await(TrainLabelNets::new(
        train_motif_net_run_config_ids,
        motif_nets_ids,
        applied_motif_loader,
        num_nets,
        run_configs,
        ws,
        store,
        opt_config,
        labels,
    )).await
}

impl <L, LM, C> TrainLabelNets<L, LM, C>
where
    L: for <'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx=C> + Cached + OutId + 'static,
    LM: for <'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>>>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{

    pub async fn new(
        train_motif_nets_run_config_ids: Vec<RunConfigId>,
        motif_nets_ids: Vec<u32>,
        applied_motif_loader: Arc<LM>,
        num_nets:u32,
        run_configs: OptRunConfigs<Vec<NamedVal<TrainLabelNetRunConfigFields>>, Vec<NamedRange<TrainLabelNetRunConfigFields>>, Vec<Hyperparameter<TrainLabelNetRunConfigFields>>, TrainLabelNetRunConfigFields>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        opt_config:Arc<OptConfig>,
        labels: Option<Vec<String>>,
    ) -> Result<Self, ThoError>
    {
        let labels = labels
            .map(|labels| {
                labels
                    .into_iter()
                    .map(|label| label.trim().to_ascii_lowercase())
                    .filter(|label| !label.is_empty())
                    .collect::<Vec<_>>()
            })
            .filter(|labels| !labels.is_empty());

        Ok(Self {
            train_motif_nets_run_config_ids,
            motif_nets_ids,
            run_configs: Arc::new(run_configs),
            pd: PhantomData,
            applied_motif_loader,
            ws:Arc::clone(&ws),
            num_nets,
            store: Arc::clone(store),
            opt_config,
            labels,

            sync_label_batch:  Arc::new(SyncLabelBatch::<C>::new(1.into()).await?),
        })
    }

}

#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct TrainLabelNetsOutput {
    train_motif_nets_run_config_ids: Vec<RunConfigId>,
    motif_nets_ids:Vec<u32>,
    results: Vec<TrainLabelNetRunConfigResult>
}

impl Output for TrainLabelNetsOutput {

}


#[async_trait]
impl SavableResult for TrainLabelNetsOutput {
    type R = Vec<u32>;

    async fn save(&mut self, store: &Arc<Box<dyn Store>>, ws: &Arc<RwLock<Box<dyn Workspace>>>, p: Self::P<'_>) -> Result<Vec<u32>, ThoError> {

        let mut run_config_ids = Vec::with_capacity(self.results.len());

        for result in &mut self.results {

            run_config_ids.push(
                track_await(result.save(store, ws, TrainLabelNetRunConfigResultParams {
                    train_motif_nets_run_config_ids:&self.train_motif_nets_run_config_ids,
                    motif_nets_ids:&self.motif_nets_ids
                })).await?
            );
        }

        Ok(run_config_ids)
    }
}

impl TrainLabelNetsOutput {
    pub fn new(train_motif_nets_run_config_ids: Vec<RunConfigId>, motif_nets_ids:Vec<u32>, results: Vec<TrainLabelNetRunConfigResult>) -> Self {
        Self {
            train_motif_nets_run_config_ids,
            motif_nets_ids,
            results
        }
    }
}

impl <L, LM, C> Stage for TrainLabelNets<L, LM, C>
where
    L: for <'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx=C> + Cached + OutId + 'static,
    LM: for <'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>>>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    type Ctx = C;
    type In<'a> = Arc<ResourcesConfigInput<NetResources<Labeled>>>;
    type Out = Arc<TrainLabelNetsOutput>;
}

impl<L, LM, C> AsyncStage for TrainLabelNets<L, LM, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    LM: for <'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>>>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MaxParallelCtx + Adjustable + 'static
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainLabelNets::run"))]
    fn run(self: Arc<Self>, input: Self::In<'static>, mut ctx: Self::Ctx) -> impl Future<Output=Result<Self::Out, ThoError>> + Send {
        let train_label_nets_run_configs = Arc::clone(&self.run_configs);


        debug!("TrainLabelNets::run: num_nets={}", self.num_nets);

        async move {
            //Split resources into train/valid sets per label for binary label-vs-rest training.
            let resources_config = input.resources_config().clone();
            let (train_labels_resource_keys, valid_label_resource_keys) = track_await(resources_config.to_resources(&self.store)).await?;

            let train_labels_resource_keys =  train_labels_resource_keys.into_iter().filter(|(label, resources)|{
                let is_empty = resources.is_empty();
                if is_empty {
                    warn!("no resources for {label} in train resources");
                }
                !is_empty
            }).collect::<Vec<_>>();

            if train_labels_resource_keys.is_empty() {
                warn!("No resources in resources_config={resources_config:?}");
                return Err(ThoError::new(format!("No resources in resources_config={resources_config:?}")));
            }

            //TODO util validate_resources function
            let valid_label_resource_keys =  valid_label_resource_keys.into_iter().filter(|(label, resources)|{
                let is_empty = resources.is_empty();
                if is_empty {
                    warn!("no resources for {label} in valid resources");
                }
                !is_empty
            }).collect::<Vec<_>>();

            if valid_label_resource_keys.is_empty() {
                warn!("No resources in resources_config={resources_config:?}");
                return Err(ThoError::new(format!("No resources in resources_config={resources_config:?}")));
            }


            let (labels_train_and_valid_sets, labels_others_train_valid_resources) = build_resource_maps(train_labels_resource_keys, valid_label_resource_keys);
            debug!("TrainLabelNets::run: labels_others_train_valid_resources={labels_others_train_valid_resources:?}");
            debug!("TrainLabelNets::run: labels_train_and_valid_sets={labels_train_and_valid_sets:?}");

            let labels_train_and_valid_sets_len = self.labels.as_ref().map(|filtered_labels|filtered_labels.len()).unwrap_or(labels_train_and_valid_sets.len());
            let labels_filter = self.labels.clone();
            let futures = labels_train_and_valid_sets.into_iter()
            .filter(|(label, _)|{
                labels_filter
                    .as_ref()
                    .map(|labels| labels.iter().any(|filter_label| filter_label.eq_ignore_ascii_case(label)))
                    .unwrap_or(true)
            })
            .map(|label_tain_and_valid_sets|{
                let (label, resources_1) = label_tain_and_valid_sets;
                let mut ctx_for_task = ctx.branch(true);

                let opt_config = Arc::clone(&self.opt_config);
                let train_label_nets_run_configs = Arc::clone(&train_label_nets_run_configs);
                let applied_motif_loader = Arc::clone(&self.applied_motif_loader);

                debug!("TrainLabelNets::run: label={label} train_motif_nets_run_config_ids={:?} train_label_nets_run_configs={train_label_nets_run_configs:?}", self.train_motif_nets_run_config_ids);
                let motif_nets_ids = self.motif_nets_ids.clone();

                let labels_others_train_valid_resources = Arc::clone(&labels_others_train_valid_resources);
                let num_nets = self.num_nets;
                let sync_label_batch = Arc::clone(&self.sync_label_batch);
                async move {
                    debug!("TrainLabelNets::run: label={label} resources={resources_1:?}");

                    let resources_0 = labels_others_train_valid_resources.get(&label).expect(&format!("no resources not belonging to {label}"));
                    info!("TrainLabelNets::run: label={label} resources_x={resources_0:?}");

                    let resources_1_train = Arc::new(resources_1.0.clone());
                    let resources_1_valid = Arc::new(resources_1.1.clone());
                    let resources_0_train = Arc::new(resources_0.0.clone());
                    let resources_0_valid = Arc::new(resources_0.1.clone());
                    let label = label.clone();
                    let label_for_batch = label.clone();


                    debug!("TrainLabelNets::run: label={label} resources_1_train={resources_1_train:?}");
                    debug!("TrainLabelNets::run: label={label} resources_1_valid={resources_1_valid:?}");
                    debug!("TrainLabelNets::run: label={label} resources_0_train={resources_0_train:?}");
                    debug!("TrainLabelNets::run: label={label} resources_0_valid={resources_0_valid:?}");

                    let train_net = Arc::new(TrainNet::new(
                        move |_ctx_for_batch:C,rc:&TrainLabelNetRunConfig| {

                            let applied_motif_loader = Arc::clone(&applied_motif_loader);
                            let label = label_for_batch.clone();
                            let sync_label_batch = Arc::clone(&sync_label_batch);
                            ready(Ok(ResourcesBatch::new(
                                   Arc::clone(&applied_motif_loader),
                                   rc.common_training_args.batch_size,
                                   Arc::clone(&resources_1_train),
                                   Arc::clone(&resources_1_valid),
                                   Arc::clone(&resources_0_train),
                                   Arc::clone(&resources_0_valid),
                                   Arc::new(move |x_1_wrapped: Vec<Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>>>>, x_0_wrapped: Vec<Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>>>>, mut ctx: C| {
                                       let label = label.clone();
                                       let sync_label_batch = Arc::clone(&sync_label_batch);
                                       async move {
                                           let input = Arc::new(SyncLabelBatchInputWrapper::new(x_1_wrapped, x_0_wrapped, num_nets as i64, label));
                                           let required_ids: HashMap<&'static str, &std::collections::HashSet<&dyn thoosa_shared::traits::CachableIdTr>> = HashMap::new();//TODO fill with x_1_wrapped x_0_wrapped ids
                                           let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoTaskAdjuster, _, ()> = OOMHandler::new(vec![
                                               Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::FAST))),
                                               Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("clc_stats"))),
                                           ], None);
                                           let output = track_await(oom_handler.run(
                                               &sync_label_batch,
                                               input,
                                               ctx.branch(false),
                                               WithRunCtxContext::new(
                                                   MaxParallelAndDumpToCpuAdjustableContex::new(
                                                       MaxParallelAdjustContextImpl::new(),
                                                       DumpToCpuContextImpl::new(&required_ids),
                                                   ),
                                               ),
                                           )).await?;
                                           Ok(output.into_parts())
                                       }.boxed()
                                   })
                               ))).boxed()


                       },
                        Arc::new(TrainLabelStepFactory::new(num_nets)),
                        opt_config,
                        train_label_nets_run_configs,
                        |rc:&TrainLabelNetRunConfig|{
                            rc.common_training_args.epochs
                        },
                        1,//TODO params
                        format!("label_{label}")
                    ));

                    let result = track_await(train_net.run(
                        Arc::new(TrainNetInput::new()),
                        ctx_for_task
                    )).await?;


                    info!("TrainLabelNets::run, done: label={label} min_loss={:?}", result.min_loss);

                    Ok::<_, ThoError>(
                        TrainLabelNetRunConfigResult::new(
                            label,
                            result.min_loss,
                            result.node_params,
                            result.run_config,
                            result.var_store_path,
                        )
                    )

                }

            });

            let results = track_await(collect_futures_with_progress(
                futures,
                labels_train_and_valid_sets_len,
                FuturesOrdering::Unordered,
                "train labels",
            )).await;

            Ok(Arc::new(TrainLabelNetsOutput::new(self.train_motif_nets_run_config_ids.clone(), self.motif_nets_ids.clone(), results)))
        }
    }
}

#[allow(type_alias_bounds)]
type BuildBatchFn<LD: Data + CopyToDevice + WithId + Send + Sync + 'static, X:Dims+ 'static, Y:Dims+ 'static, C> = Arc<dyn Fn(
    Vec<Arc<DevicesWrap<LD>>>,
    Vec<Arc<DevicesWrap<LD>>>, C

) -> BoxFuture<'static, Result<(ThoBatch<X>, Option<BatchLengths>, ThoBatch<Y>), ThoError>> + Send + Sync>;

// #[derive(Debug)]
struct ResourcesBatch<L, C, LD, X, Y>
where
    L: for<'i> Cached<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<LD>>, Ctx=C> + AsyncStage + OutId + 'static,
    C:RunCtx+'static,
    LD: Data + CopyToDevice + WithId + Send + Sync + 'static,
    X:Dims+ 'static,
    Y:Dims+ 'static,
{

    half_batch_size: u32,
    resources_1_train:Arc<Vec<Resource>>,
    resources_1_valid:Arc<Vec<Resource>>,
    resources_0_train:Arc<Vec<Resource>>,
    resources_0_valid:Arc<Vec<Resource>>,
    loader: Arc<L>,
    build_batch_fn: BuildBatchFn<LD, X, Y, C>,
}

impl <L, C, LD, X, Y> Debug for  ResourcesBatch<L, C, LD, X, Y>
where
    L: for<'i> Cached<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<LD>>, Ctx=C> + AsyncStage + OutId + 'static,
    C:RunCtx+'static,
    LD: Data + CopyToDevice + WithId + Send + Sync + 'static,
    X:Dims+ 'static,
    Y:Dims+ 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourcesBatch").finish()//TODO
    }
}

impl <L, C, LD, X, Y> ResourcesBatch<L, C, LD, X, Y>
where
    L: for<'i> Cached<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<LD>>, Ctx=C> + AsyncStage + OutId + 'static,
    C:RunCtx+'static,
    LD: Data + CopyToDevice + WithId + Send + Sync + 'static,
    X:Dims+ 'static,
    Y:Dims+ 'static,
{
    pub fn new(
        loader: Arc<L>,
        batch_size: u32,
        resources_1_train:Arc<Vec<Resource>>,
        resources_1_valid:Arc<Vec<Resource>>,
        resources_0_train:Arc<Vec<Resource>>,
        resources_0_valid:Arc<Vec<Resource>>,
        build_batch_fn:BuildBatchFn<LD, X, Y, C>,
    ) -> Self {
        debug!("ResourcesBatch::new: batch_size={batch_size:?}");
        debug!("ResourcesBatch::new: resources_1_train={resources_1_train:?}");
        debug!("ResourcesBatch::new: resources_1_valid={resources_1_valid:?}");
        debug!("ResourcesBatch::new: resources_0_train={resources_0_train:?}");
        debug!("ResourcesBatch::new: resources_0_valid={resources_0_valid:?}");

        assert_eq!(resources_1_train.len(), resources_0_train.len());//train
        assert_eq!(resources_1_valid.len(), resources_0_valid.len());//valid

        //TODO replacement?
        let half_batch_size = (batch_size / 2).max(1);

        assert!(half_batch_size <= resources_1_train.len() as u32);
        assert!(half_batch_size <= resources_1_valid.len() as u32);
        assert!(half_batch_size <= resources_0_train.len() as u32);
        assert!(half_batch_size <= resources_0_valid.len() as u32);

        Self {
            resources_1_train,
            resources_1_valid,
            resources_0_train,
            resources_0_valid,
            half_batch_size,
            loader,
            build_batch_fn,
        }
    }
}

impl <L,C, LD, X, Y> AsyncDataBatch<X, Y, C> for ResourcesBatch<L, C, LD, X, Y>
where
    L: for<'i> Cached<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<LD>>, Ctx=C> + AsyncStage + OutId + 'static,
    C:RunCtx+'static,
    LD: Data + CopyToDevice + WithId + Send + Sync + 'static,
    X:Dims+ 'static,
    Y:Dims+ 'static,
{
    fn train_batch_stream<'a>(&'a self, ctx: C) -> BoxStream<'a, Result<(ThoBatch<X>, Option<BatchLengths>, ThoBatch<Y>), ThoError>> {
        let batches = shuffle_and_chunk_with_rng(self.resources_1_train.len(), self.half_batch_size, &mut rng());

        debug!("ResourcesBatch::train_batch_stream: batches={batches:?}");
        assert!(!batches.is_empty(), "train_batch_stream: batch is empty, self.resources_1_train={:?} self.resources_0_train={:?}",self.resources_1_train,  self.resources_0_train);

        Self::build_batch_stream(
            Arc::clone(&self.build_batch_fn),
            batches,
            Arc::clone(&self.loader),
            ctx,
            Arc::clone(&self.resources_1_train),
            Arc::clone(&self.resources_0_train),
            "train"
        )
    }

    fn valid_batch_stream<'a>(&'a self, ctx: C) -> BoxStream<'a, Result<(ThoBatch<X>, Option<BatchLengths>, ThoBatch<Y>), ThoError>> {
        let batches = shuffle_and_chunk_with_rng(self.resources_1_valid.len(), self.half_batch_size, &mut rng());

        debug!("ResourcesBatch::valid_batch_stream: batches={batches:?}");
        assert!(!batches.is_empty(), "valid_batch_stream: batch is empty, self.resources_1_train={:?} self.resources_0_train={:?}",self.resources_1_train,  self.resources_0_train);

        Self::build_batch_stream(
            Arc::clone(&self.build_batch_fn),
            batches,
            Arc::clone(&self.loader),
            ctx,
            Arc::clone(&self.resources_1_valid),
            Arc::clone(&self.resources_0_valid),
            "valid"
        )
    }

    fn num_batches(&self) -> usize {
        self.resources_1_train.len() / self.half_batch_size as usize
    }
}

impl<L, C, LD, X, Y> ResourcesBatch<L, C, LD, X, Y>
where
    C: 'static + RunCtx,
    L: 'static + AsyncStage + OutId + for<'i> Cached<In<'i>=Arc<Resource>, Out=Arc<DevicesWrap<LD>>, Ctx=C>+ 'static,
    LD: Data + CopyToDevice + WithId + Send + Sync + 'static,
    X:Dims+ 'static,
    Y:Dims+ 'static,
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="ResourcesBatch::build_batch_stream"))]
    fn build_batch_stream(build_batch_fn: BuildBatchFn<LD, X, Y, C>, resource_indexes_for_batches: Vec<Box<[usize]>>, loader: Arc<L>, mut ctx: C, x_1: Arc<Vec<Resource>>, x_0: Arc<Vec<Resource>>, descr: &'static str) -> BoxStream<'static, Result<(ThoBatch<X>, Option<BatchLengths>, ThoBatch<Y>), ThoError>> {
        debug!("ResourcesBatch::build_batch_stream: resource_indexes_for_batches={resource_indexes_for_batches:?}");

        // TODO(perf): each produced batch waits for collect_futures_with_progress to load/apply both sides; add a bounded prefetch queue so ApplyMotifNets overlaps TrainNet forward/backward.
        let futures = futures::stream::iter(resource_indexes_for_batches.into_iter().enumerate()).then(move |(i, batch_keys)|   {
            let build_batch_fn = Arc::clone(&build_batch_fn);
            let loader = Arc::clone(&loader);
            let x_1 = Arc::clone(&x_1);
            let x_0 = Arc::clone(&x_0);
            let mut ctx_for_build_fn = ctx.branch(false);

            async move {
                debug!("ResourcesBatch::build_batch_stream: descr={descr} batch_i={i} batch_keys={batch_keys:?}");

                let batch_len = batch_keys.len();
                let futures = batch_keys.into_iter().map(|index| {
                    debug!("ResourcesBatch::build_batch_stream: descr={descr} batch_i={i} in future 0");

                    let loader = Arc::clone(&loader);
                    let x_1 = Arc::clone(&x_1);
                    let x_0 = Arc::clone(&x_0);
                    let ctx_x1 = ctx_for_build_fn.branch(false);
                    let ctx_x0 = ctx_for_build_fn.branch(false);

                    async move {
                        debug!("ResourcesBatch::build_batch_stream: descr={descr} batch_i={i} in future 1");

                        try_join!(
                            Arc::clone(&loader).run_cached(Arc::new(x_1[index].clone()), ctx_x1),
                            Arc::clone(&loader).run_cached(Arc::new(x_0[index].clone()), ctx_x0)
                        )
                    }
                });

                let futures = futures.map(|future| async move { Ok::<_, ThoError>(track_await(future).await) });

                debug!("ResourcesBatch::build_batch_stream: descr={descr} futures.len={}", futures.len());

                let (x_1_wrapped, x_0_wrapped) = track_await(collect_futures_with_progress(
                    futures,
                    batch_len,
                    FuturesOrdering::Ordered,
                    "ResourcesBatch::build_batch_stream",
                )).await
                    .into_iter()
                    .collect::<Result<Vec<_>, ThoError>>()?
                    .into_iter()
                    .fold((vec![], vec![]), |mut a, v| {
                        a.0.push(v.0);
                        a.1.push(v.1);
                        a
                    });
                //
                debug!("ResourcesBatch::build_batch_stream: descr={descr} x_1_wrapped={x_1_wrapped:?} x_0_wrapped={x_0_wrapped:?}");

                let r: Result<(ThoBatch<X>, Option<BatchLengths>, ThoBatch<Y>), ThoError> = track_await(build_batch_fn(x_1_wrapped, x_0_wrapped, ctx_for_build_fn)).await;

                debug!("ResourcesBatch::build_batch_stream: descr={descr} batch_i={i} r={r:?}");

                r
            }
        });

        // preload_one(futures)
        futures.boxed()
    }

}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct TrainLabelStepIpcInit {
    params: LabelNetParams,
    training_config: CommonTrainingRunConfig,
    tho_device_idx: usize,
    vs_path: PathBuf,
}

type SyncTrainLabelStep<C> = TrainStep<TrainStepXFn<MotifProbability>, MotifProbability, LabelProbability, C>;

fn train_label_step_params(run_config: &TrainLabelNetRunConfig, num_nets: u32) -> LabelNetParams {
    match &run_config.net {
        LabelNetTypeArgs::ConvNet(ConvNetArgs { num_conv_layers, hidden_conv_channels, num_head_layers, num_head_features, kernel, joint }) => {
            LabelNetParams::new(num_nets, LabelNetTypeParams::ConvNetParams(ConvNetParams::new(*num_conv_layers, *hidden_conv_channels, *kernel, *num_head_layers, *num_head_features, joint.into())))
        }
        LabelNetTypeArgs::LinNet(LinNetArgs { num_layers, features }) => {
            LabelNetParams::new(num_nets, LabelNetTypeParams::LinNetParams(LinLenNetParams::new(*features, *num_layers)))
        }
        LabelNetTypeArgs::SelfAttentionNet(SelfAttentionNetArgs { head_size, n_heads, ff_mult, dropout }) => {
            LabelNetParams::new(num_nets, LabelNetTypeParams::SelfAttentionNetParams(SelfAttentionNetParams::new(*head_size, *n_heads, *ff_mult, *dropout)))
        }
    }
}

fn train_label_loss(pred_y: &Tensor, target: &Tensor) -> Result<Tensor, ThoError> {
    pred_y.f_binary_cross_entropy_with_logits::<Tensor>(target, None, None, Reduction::Mean).map_err(ThoError::from)
}

impl WorkerStageInput for TrainLabelStepIpcInit {
}

#[typetag::serde(name = "TrainLabelStepIpcInit")]
impl WorkerInput for TrainLabelStepIpcInit {}

#[typetag::serde(name = "TrainLabelStepInput")]
impl WorkerInput for crate::tasks::train_step::TrainStepInput<MotifProbability, LabelProbability> {}

fn train_label_step_endpoint_init<'a>(init: Box<dyn WorkerInput>, ws: Arc<RwLock<Box<dyn Workspace>>>, all_cuda_tensors: &'a AllTensors, tho_device:&'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let init = *(init as Box<dyn std::any::Any>).downcast::<TrainLabelStepIpcInit>().map_err(|_| ThoError::new("TrainLabelStep endpoint received invalid initializer"))?;
        let params = init.params.clone();
        let params_for_module = params.clone();
        let training_config = init.training_config.clone();
        let (train_step, _): (SyncTrainLabelStep<WokerRunCtx>, ()) = TrainStep::new_local(
            init.tho_device_idx.into(),
            move |_run_config, device| async move {
                let config = ModConfig::new(
                    training_config.tho_optimizer(),
                    move |var_store| params_for_module.label_net()?.build_tch_module::<Batch<MotifProbability>, Batch<LabelProbability>>(&var_store.root()),
                    training_config.lr,
                );
                Ok((ThoTrainModule::new_on_device(config, device)?, ()))
            },
            Arc::new(Box::new(|batch| batch) as TrainStepXFn<MotifProbability>),
            train_label_loss,
            init.training_config,
            init.vs_path,
        ).await?;
        // Shared stateful dispatch keeps the module, optimizer, and VarStore in the worker.
        Ok(Box::new(WorkerWrapperWithState::<JsonIpcCodec, _, _, _>::new(Arc::new(train_step), PhantomData)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}

#[derive(Debug)]
struct TrainLabelStepFactory {
    num_nets: u32,
}

impl TrainLabelStepFactory {
    fn new(num_nets: u32) -> Self {
        Self { num_nets }
    }
}

#[async_trait]
impl<C> TrainStepFactory<MotifProbability, LabelProbability, C, LabelNetParams, TrainLabelNetRunConfig> for TrainLabelStepFactory
where
    C: RunCtx + 'static,
{
    async fn new(&self, run_config: TrainLabelNetRunConfig, ctx: C, vs_path: PathBuf) -> Result<(DynSyncStageWithStateEnum<SyncTrainLabelStep<C>>, LabelNetParams), ThoError> {
        let node_params = train_label_step_params(&run_config, self.num_nets);
        if GlobalCtx::global().is_ipc() {
            static TRAIN_LABEL_STEP_ID: AtomicU32 = AtomicU32::new(0);
            let stage_id = TRAIN_LABEL_STEP_ID.fetch_add(1, Ordering::AcqRel);
            let init = TrainLabelStepIpcInit { params: node_params.clone(), training_config: run_config.common_training_args.clone(), tho_device_idx: ctx.tho_device_idx(), vs_path };
            let input_cloner = Arc::new(|input: &crate::tasks::train_step::TrainStepInput<MotifProbability, LabelProbability>| -> Result<Box<dyn WorkerInput>, ThoError> { Ok(Box::new(input.clone_refs_for_worker()?)) });
            Ok((DynSyncStageWithStateEnum::Proxy { stage: Arc::new(ProxyStage::proxy_with_input_cloner(TRAIN_LABEL_STEP_ENDPOINT, stage_id, ctx.tho_device_idx(), Box::new(init), input_cloner).await?), init: () }, node_params))
        } else {
            let params_for_module = node_params.clone();
            let training_config = run_config.common_training_args.clone();
            let (train_step, _) = TrainStep::new_local(
                ctx.tho_device_idx().into(),
                move |_run_config, device| async move {
                    let config = ModConfig::new(
                        training_config.tho_optimizer(),
                        move |var_store| params_for_module.label_net()?.build_tch_module::<Batch<MotifProbability>, Batch<LabelProbability>>(&var_store.root()),
                        training_config.lr,
                    );
                    Ok((ThoTrainModule::new_on_device(config, device)?, ()))
                },
                Arc::new(Box::new(|batch| batch) as TrainStepXFn<MotifProbability>),
                train_label_loss,
                run_config,
                vs_path,
            ).await?;
            Ok((DynSyncStageWithStateEnum::Local(Arc::new(train_step)), node_params))
        }
    }
}

pub fn train_label_step_endpoint_inits() -> HashMap<String, InitFn> {
    HashMap::from([(
        TRAIN_LABEL_STEP_ENDPOINT.to_string(),
        Box::new(train_label_step_endpoint_init) as InitFn,
    )])
}

#[cfg(test)]
mod tests {
    use crate::tasks::apply_motif_nets::{ApplyMotifNets, motif_test_net_};
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::stft::SyncStftSelector;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use crate::tasks::train_label_nets::{ResourcesBatch, TrainLabelNets, TrainLabelNetsOutput, new_train_label_nets};
    use futures::FutureExt;
    use futures_util::StreamExt;
    use futures_util::future::try_join_all;
    use log::LevelFilter::{Debug, Info, Trace};
    use log::{Level, LevelFilter, debug, info, log_enabled, set_max_level};
    use mockall::predicate::always;
    use parking_lot::RwLock;
    use rand::{Rng, RngExt, rng};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Duration;
    use tch::Device::Cpu;
    use tch::Tensor;
    use tch::kind::FLOAT_CPU;
    use thoosa_common::batches::AsyncDataBatch;
    use thoosa_common::data_with_extra::DataWithExtra;
    use thoosa_common::dims::ChannelsData;
    use thoosa_common::dims::{Batch, LabelProbability, MotifProbability};
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::common::cachable::{DevicesWrap, SelectRef};
    use thoosa_common::runner::common::{MockRunCtx, mock_run_ctx};
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runs::{Labeled, NetResources};
    use thoosa_common::tch_test_utils::motifs_tensor;
    use thoosa_common::tch_test_utils::resource_tensor;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{CommonTrainingRunConfigFields, LabelNetTypeArgsFields, LinNetArgsFields, OptConfig, TrainLabelNetRunConfigFields};
    use thoosa_common::utils;
    use thoosa_common::utils::{mix_x_binary_with_labels, pad_stack_with_mask, reset_print_options};
    use thoosa_common::ws::{WsMockBuilder, mock_ws_simp};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::hyperparameter::Hyperparameter;
    use thoosa_shared::mono_key;
    use thoosa_shared::perf_logging::{PERF, format_perf_map};
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::RunConfigId;
    use thoosa_shared::test_utils::MotifKind;
    use thoosa_shared::traits::AsyncStage;
    use thoosa_shared::traits::MaxParallelCtx;
    use thoosa_shared::traits::{Named, RunCtx};
    use thoosa_shared::val::{NamedVal, Val};
    use thoosa_shared::{TensorIdSliceIdx, ThoError};
    use thoosa_store::{MockStoreBuilder, TrainedNet};
    use tokio::fs;
    use tokio::io::Join;
    use tokio::task::JoinSet;

    #[ctor::ctor]
    pub fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn heavy_test_train_label_net_3x3() -> Result<(), ThoError>
    {

        let mut rand = rng();


        thoosa_shared::test_utils::maybe_init_perf_tracing_stdout("info");
        let test_dir = Path::new("/tmp/test_train_motif_nets");//.join(cdt_for_dir());
        fs::create_dir_all(test_dir).await?;

        log::set_max_level(Debug);
        reset_print_options();
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let input_n_0 = 1000;
        let find_motifs_run_config_id: RunConfigId = 1;
        let select_motifs_run_config_id: RunConfigId = 2;
        let motif_width = 100;

        let motif_0_0_id = 0; //channel 0 motif 0
        let motif_0_1_id = 1; //channel 0 motif 1
        let motif_1_0_id = 0; //channel 1 motif 0

        let pulse_motif = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_0_0", "motif_id")]), motifs_tensor(100, 3, MotifKind::Pulse, 0.3, 0.4))?);
        let triangle_motif = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_0_1", "motif_id")]), motifs_tensor(100, 3, MotifKind::Triangle, 0.3, 0.4))?);
        let sin_motif = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_1_0", "motif_id")]), motifs_tensor(100, 3, MotifKind::Sin, 0.3, 0.4))?);
        let train_group = "train_group";
        let valid_group = "valid_group";

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 3, 100, input_n_0 as u32)
                //TODO use //use with_load_motif_net
                .with_load_motif_nets(vec![
                    vec![
                        //channel 0
                        motif_test_net_(Cpu, Arc::clone(&pulse_motif), 10, 100),
                        motif_test_net_(Cpu, Arc::clone(&triangle_motif), 10, 100),
                        motif_test_net_(Cpu, Arc::clone(&sin_motif), 10, 100),
                    ],
                    vec![
                        //channel 1
                        motif_test_net_(Cpu, Arc::clone(&sin_motif), 10, 100),
                    ],
                    vec![],
                ])
                .build(),
        ));

        let resources_a = (0..10).map(|i|Resource::new(i, 1, format!("resource_a_{i}"), rand.random_range(700..1000))).collect::<Vec<_>>();
        let resources_b = (10..20).map(|i|Resource::new(i, 1, format!("resource_b_{i}"), rand.random_range(700..1000))).collect::<Vec<_>>();
        let resources_c = (20..30).map(|i|Resource::new(i, 1, format!("resource_c_{i}"), rand.random_range(700..1000))).collect::<Vec<_>>();
        let resources_x = (30..40).map(|i|Resource::new(i, 1, format!("resource_x_{i}"), rand.random_range(700..1000))).collect::<Vec<_>>();

        let (mut resources_map, _motif_intervals) = resources_a.iter().zip(resources_b.iter().zip(resources_c.iter().zip(resources_x.iter()))).try_fold((HashMap::new(), vec![]), |mut a, (r_a, (r_b, (r_c, r_x)))| {

            let (t, intervals) = resource_tensor(r_a.length() as usize, &[
                            vec![(3, motif_width, MotifKind::Pulse)],
                            vec![(1, motif_width, MotifKind::Sin)],
                            vec![]
                        ], 100, 10.0,
                                                 true,
                                                 true)?;

            a.0.insert(r_a.clone(), t);

            a.1.extend(intervals[0][0].iter().map(|interval|TensorIdSliceIdx(r_a.id(), interval.start as u32)));


            let (t, intervals) = resource_tensor(r_b.length() as usize, &[
                            vec![(3, motif_width, MotifKind::Triangle)],
                            vec![(1, motif_width, MotifKind::Sin)],
                            vec![]
                        ], 100, 10.0,
                                                 true,
                                                 true)?;

            a.0.insert(r_b.clone(), t);

            a.1.extend(intervals[0][0].iter().map(|interval|TensorIdSliceIdx(r_b.id(), interval.start as u32)));


            let (t, intervals) = resource_tensor(r_c.length() as usize, &[
                            vec![(3, motif_width, MotifKind::Sin)],
                            vec![(1, motif_width, MotifKind::Sin)],
                            vec![]
                        ], 100, 10.0,
                                                 true,
                                                 true)?;

            a.0.insert(r_b.clone(), t);

            a.1.extend(intervals[0][0].iter().map(|interval|TensorIdSliceIdx(r_c.id(), interval.start as u32)));


            let (t, intervals) = resource_tensor(r_x.length() as usize, &[
                            vec![],
                            vec![],
                            vec![]
                        ], 100, 10.0,
                                                 true,
                                                 true)?;

            a.0.insert(r_c.clone(), t);


            Ok::<_, ThoError>(a)
        })?;


        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached(resources_map)
                .finish(),
        );

        let mut store = Arc::new(
            MockStoreBuilder::new()
            .with_list_selected_motifs([
                (select_motifs_run_config_id, Ok(
                    vec![
                        vec![(motif_0_0_id, 1.0f64), (motif_0_1_id, 1.0f64)], //channel 0
                        vec![(motif_1_0_id, 1.0f64)], //channel 1
                        vec![]//channel 2
                    ]
                ))
            ])
            .with_list_group_labeled_resources_sets([
                (train_group, Ok(vec![
                    ("Label_A".to_string(), "train_resource_set_for_label_a".to_string()),
                    ("Label_B".to_string(), "train_resource_set_for_label_b".to_string()),
                    ("Label_C".to_string(), "train_resource_set_for_label_c".to_string())
                ])),
                (valid_group, Ok(vec![
                    ("Label_A".to_string(), "valid_resource_set_for_label_a".to_string()),
                    ("Label_B".to_string(), "valid_resource_set_for_label_b".to_string()),
                    ("Label_C".to_string(), "valid_resource_set_for_label_c".to_string())
                ]))

            ])
            .with_list_resource_set_resources_by_descr([
                ("train_resource_set_for_label_a", Ok(resources_a.clone())),
                ("train_resource_set_for_label_b", Ok(resources_b.clone())),
                ("train_resource_set_for_label_c", Ok(resources_c.clone())),

                //same resources, should be ok for test
                ("valid_resource_set_for_label_a", Ok(resources_a.clone())),
                ("valid_resource_set_for_label_b", Ok(resources_b.clone())),
                ("valid_resource_set_for_label_c", Ok(resources_c.clone())),

            ])
            .with_list_resource_random_not_labeled([
                (train_group, Ok(resources_x.clone())),
                (valid_group, Ok(resources_x.clone())),
            ])
            .with_list_trained_nets([
                (vec![select_motifs_run_config_id], Some(0.3), Some(3), Ok(vec![
                    TrainedNet { id: 0, run_config_id: select_motifs_run_config_id, net_path: PathBuf::new(), loss: 0.1f32 },
                    TrainedNet { id: 0, run_config_id: select_motifs_run_config_id, net_path: PathBuf::new(), loss: 0.2f32 },
                ])),
            ])
            .build());


        let run_configs = thoosa_shared::run_configs::OptRunConfigs::new(
            thoosa_shared::run_configs::GridRunConfigs::new(
                thoosa_shared::run_configs::ConstConfigs::new(vec![
                    NamedVal::new(TrainLabelNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Epochs), Val::Int(100)),
                    NamedVal::new(TrainLabelNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::BatchSize), Val::Int(8)),
                    // NamedVal::new(TrainLabelNetRunConfigFields::Net(LabelNetTypeArgsFields::FixedNumSlices(FixedNumSlicesNetArgsFields::NumLayers)), Val::Int(3)),
                    NamedVal::new(TrainLabelNetRunConfigFields::Net(LabelNetTypeArgsFields::LinNet(LinNetArgsFields::Features)), Val::Int(1000)),//TODO used as num features in any_len_label_node
                ]),
                vec![
                    // NamedRange::new(TrainLabelNetRunConfigFields::NumLayers, Val::Int(0)..Val::Int(10), (Val::Int(10), 0))
                ],
            ),
            vec![
                Hyperparameter::UniformFloat{name: TrainLabelNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), lower: 0.0001, upper: 0.001},
                Hyperparameter::UniformInt{name: TrainLabelNetRunConfigFields::Net(LabelNetTypeArgsFields::LinNet(LinNetArgsFields::NumLayers)), lower: 1, upper: 10}
            ],
        );

        // let ctx = MockRunCtx::purge_gpu_caches_context();
        // ctx.expect().with(always()).times(1..).returning( |_| Box::pin(async { Ok(()) }));

        let apply_motif_nets = Arc::new(ApplyMotifNets::new(
            &ws,
            &store,
            vec![select_motifs_run_config_id],
            mock_run_ctx::default(1),
            Some(0.3),
            Some(3),
            10,
            None,
            false,
        ).await?);

        let train_label_nets = Arc::new(new_train_label_nets(
            vec![select_motifs_run_config_id],
            loader,
            apply_motif_nets,
            run_configs,
            &ws,
            &store,
            Arc::new(OptConfig::new(30, 10)),
            None,
        ).await?);


        let input = Arc::new(ResourcesConfigInput::new(NetResources::new(Labeled::new(train_group.to_string()), Labeled::new("valid_group".to_string()))));


        let mut run_ctx = mock_run_ctx::default(1);

        let out = train_label_nets.run(input, mock_run_ctx::default(1)).await?;

        debug!("test_train_label_net_3x3: PERF={PERF:?}");
        debug!("test_train_label_net_3x3: PERF=\n{}", format_perf_map(&PERF));


        if log_enabled!(Level::Debug) {
            for r in &out.results {
                debug!("label={} loss={}", r.label, r.loss);
            }
        }

        debug!("test_train_label_net_3x3: 0. out={out:?}");

        assert!(out.results[0].loss < 0.5);
        assert!(out.results[1].loss < 0.5);
        assert!(out.results[2].loss < 0.5);


        debug!("test_train_label_net_3x3: out={out:?}");

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_resources_batch() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let resources_1_train = Arc::new(vec![
            Resource::new(0, 1, "aaa_1_tr", 10),
            Resource::new(1, 1, "bbb_1_tr", 10),
            Resource::new(2, 1, "ccc_1_tr", 10)
        ]);
        let resources_1_valid = Arc::new(vec![
            Resource::new(3, 1, "aaa_1_vl", 10),
            Resource::new(4, 1, "bbb_1_vl", 10),
            Resource::new(5, 1, "ccc_1_vl", 10),
        ]);


        let resources_0_train = Arc::new(vec![
            Resource::new(6, 1, "aaa_0_tr", 10),
            Resource::new(7, 1, "bbb_0_tr", 10),
            Resource::new(8, 1, "ccc_0_tr", 10)
        ]);
        let resources_0_valid = Arc::new(vec![
            Resource::new(9,  1, "aaa_0_vl", 10),
            Resource::new(10, 1, "bbb_0_vl", 10),
            Resource::new(11, 1, "ccc_0_vl", 10),
        ]);

        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached(HashMap::<Resource, Tensor>::from_iter(
                    resources_1_train.iter().chain(resources_1_valid.iter()).chain(resources_0_train.iter()).chain(resources_0_valid.iter()).map(|r|{
                        if r.file_stem().contains("_1_") {
                             (r.clone(), Tensor::ones(&[1, 10], FLOAT_CPU))
                        } else {
                             (r.clone(), Tensor::zeros(&[1, 10], FLOAT_CPU))
                        }

                    })
                ))
                .finish(),
        );


        let resource_batch: ResourcesBatch<_, _, ThoTensor<ChannelsData, SynKey>, ChannelsData, LabelProbability> = ResourcesBatch::new(
            loader,
            6,
            resources_1_train,
            resources_1_valid,
            resources_0_train,
            resources_0_valid,
            Arc::new(|x_1_wrapped:Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>, x_0_wrapped:Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>, mut ctx: MockRunCtx|{
                async move {
                    let x_1 = try_join_all(x_1_wrapped.iter().map(|x| x.borrow_to(ctx.tho_device_idx(), None::<SyncStftSelector>))).await?;
                    let x_0 = try_join_all(x_0_wrapped.iter().map(|x| x.borrow_to(ctx.tho_device_idx(), None::<SyncStftSelector>))).await?;
                    {
                        debug!("ResourcesBatch::train_batch_iter: x_1={x_1:?} x_0={x_0:?}");
                        let x_1 = Tensor::stack(&x_1.into_iter().map(|v|v.t_ref().copy()).collect::<Vec<_>>(), 0);
                        let x_0 = Tensor::stack(&x_0.into_iter().map(|v|v.t_ref().copy()).collect::<Vec<_>>(), 0);

                        let (x, _lengths, y) = mix_x_binary_with_labels(&x_1, None, &x_0, None)?;

                         Ok((x.into(), None, y.into()))
                    }
                }.boxed()
            }),
        );

        let mut batch = resource_batch.train_batch_stream(mock_run_ctx::default(0));
        let (_batch_x, _lengths,  _batch_y) = batch.next().await.expect("no batch").expect("batch failed");
        assert!(batch.next().await.is_none()); //should be only one

        let mut batch = resource_batch.valid_batch_stream(mock_run_ctx::default(0));
        let (_batch_x, _lengths, _batch_y) = batch.next().await.expect("no batch").expect("batch failed");
        assert!(batch.next().await.is_none()); //should be only one



        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_1_resource_batch() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let resources_1_train = Arc::new(vec![
            Resource::new(0, 1, "aaa_1_tr", 10),
        ]);
        let resources_1_valid = Arc::new(vec![
            Resource::new(3, 1, "aaa_1_vl", 10),
        ]);

        let resources_0_train = Arc::new(vec![
            Resource::new(7, 1, "bbb_0_tr", 10),
        ]);
        let resources_0_valid = Arc::new(vec![
            Resource::new(10, 1, "bbb_0_vl", 10),
        ]);

        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached(HashMap::<Resource, Tensor>::from_iter(
                    resources_1_train.iter().chain(resources_1_valid.iter()).chain(resources_0_train.iter()).chain(resources_0_valid.iter()).map(|r|{
                        if r.file_stem().contains("_1_") {
                            (r.clone(), Tensor::ones(&[1, 10], FLOAT_CPU))
                        } else {
                            (r.clone(), Tensor::zeros(&[1, 10], FLOAT_CPU))
                        }

                    })
                ))
                .finish(),
        );


        let resource_batch: ResourcesBatch<_, _, ThoTensor<ChannelsData, SynKey>, ChannelsData, LabelProbability> = ResourcesBatch::new(
            loader,
            2,
            resources_1_train,
            resources_1_valid,
            resources_0_train,
            resources_0_valid,
            Arc::new(|x_1_wrapped:Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>, x_0_wrapped:Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>, mut ctx: MockRunCtx|{
                async move {
                    let x_1 = try_join_all(x_1_wrapped.iter().map(|x| x.borrow_to(ctx.tho_device_idx(), None::<SyncStftSelector>))).await?;
                    let x_0 = try_join_all(x_0_wrapped.iter().map(|x| x.borrow_to(ctx.tho_device_idx(), None::<SyncStftSelector>))).await?;
                    {
                        debug!("ResourcesBatch::train_batch_iter: x_1={x_1:?} x_0={x_0:?}");
                        let x_1 = Tensor::stack(&x_1.into_iter().map(|v|v.t_ref().copy()).collect::<Vec<_>>(), 0);
                        let x_0 = Tensor::stack(&x_0.into_iter().map(|v|v.t_ref().copy()).collect::<Vec<_>>(), 0);

                        let (x, _lengths, y) = mix_x_binary_with_labels(&x_1, None, &x_0, None)?;

                         Ok((x.into(), None, y.into()))
                    }
                }.boxed()
            }),
        );

        let mut batch = resource_batch.train_batch_stream(mock_run_ctx::default(0));
        let (_batch_x, _lengths,  _batch_y) = batch.next().await.expect("no batch").expect("batch failed");
        assert!(batch.next().await.is_none()); //should be only one

        let mut batch = resource_batch.valid_batch_stream(mock_run_ctx::default(0));
        let (_batch_x, _lengths, _batch_y) = batch.next().await.expect("no batch").expect("batch failed");
        assert!(batch.next().await.is_none()); //should be only one



        Ok(())
    }


}
