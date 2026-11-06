use crate::motifs_store::MotifsStore;
use async_trait::async_trait;
use futures::FutureExt;
use futures_util::future::{BoxFuture, try_join_all};
use futures_util::stream::BoxStream;
use futures_util::StreamExt;
use log::{debug, info, warn};
use parking_lot::RwLock;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::future::ready;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use tch::{Reduction, Tensor};
use thoosa_common::data_with_extra::DataWithExtra;
use thoosa_common::dims::ChannelsData;
use thoosa_common::ipc::cuda_wokrer::{AllTensors, InitFn, WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use thoosa_common::ipc::serialization::json::JsonIpcCodec;
use thoosa_common::tch_error::tch_result;

type MotifsInitFn = InitFn<dyn crate::motifs_ws::MotifsWorkspace>;
use thoosa_common::ipc::proxy::{DynSyncStageEnum, DynSyncStageWithStateEnum, ProxyStage};
use thoosa_common::ipc::wraps::SyncStageIpcWrapper;
use thoosa_common::ipc::wraps::{WorkerWrapper, WorkerWrapperWithState};
use thoosa_common::runner::common::cachable::{BorrowRef, Cached, DevicesWrap, ThoDataRef};
use thoosa_common::runner::oom_handler::{Adjusters, NoInputAdjuster, NoStageAdjuster, OOMHandler, WithRunCtxContext, run_ctx_adjuster};
use thoosa_common::runner::sync_stage::SyncStage;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::resource::{Resource, ResourcesConfig, build_resource_maps};
use thoosa_shared::traits::MaxParallelCtx;
use thoosa_shared::traits::{Adjustable, AsyncStage, Data, OutId, RunCtx, Stage, WithId};

use mockall_double::double;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;
use crate::dims::{LabelProbability, MotifProbability};
use crate::stages::apply_motif_nets::ApplyMotifNets;
use crate::stages::cross_loader::ResourcesConfigInput;
use crate::stages::stage_ids::RAW_TENSOR_LOADER;
use crate::train_nets::{LabelNetParams, LabelNetTypeArgs, LabelNetTypeArgsBuilder, LabelNetTypeArgsFields, LabelNetTypeArgsRangeOrValue, LabelNetTypeParams, LinNetArgs, TrainLabelNetRunConfigResultParams};
use clap::Args;
use rand::rng;
use thoosa_common::dims::{Batch, Dims, ThoBatch};
use thoosa_common::instance_cache::InstanceCache;
use thoosa_common::ipc::worker_msg::WorkerInput;
use thoosa_common::nets::batches::{AsyncDataBatch, BatchLengths};
use thoosa_common::nets::common_nets::{CommonTrainingRunConfig, CommonTrainingRunConfigBuilder, CommonTrainingRunConfigFields, CommonTrainingRunConfigRangeOrValue, ConvNetArgs, ConvNetParams, LinLenNetParams, OptConfig, SelfAttentionNetArgs, SelfAttentionNetParams, WithCommonTrainingRunConfig};
use thoosa_common::nets::tho_varstore::{ModConfig, ThoTrainModule};
use thoosa_common::resources::{Labeled, NetResources, SavableResult};
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep};
use thoosa_common::stages::stages_seq::PairStage;
use thoosa_common::stages::train_net::{TrainNet, TrainNetInput, TrainStepFactory};
use thoosa_common::stages::train_step::{TrainStep, TrainStepInput, TrainStepXFn};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::utils::{mix_x_binary_with_labels, pad_stack_with_mask};
use thoosa_cpu_data_derive::CpuData;
use thoosa_properties_map::ToHashmap;
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::ipc_types::WorkerOutput;
use thoosa_shared::run_configs::RunConfig;
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::tmp_file::TmpFile;
use thoosa_shared::tracked_await::t;
use thoosa_shared::{CopyToDevice, FuturesOrdering, ThoError, collect_futures_with_progress, mono_key, panic_to_err, shuffle_and_chunk_with_rng};
use tokio::try_join;

pub const SYNC_LABEL_BATCH_ENDPOINT: &str = "SyncLabelBatch";
pub const TRAIN_LABEL_STEP_ENDPOINT: &str = "TrainLabelStep";

pub type AppliedMotifOutput = DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>;

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


    fn size(&self) -> Self::Metrics {
        (self.x_1.len() + self.x_0.len()) as i64
    }

    fn tho_device_idx(&self) -> usize {
        self.x_1.iter().chain(self.x_0.iter()).next().map(|value| value.as_ref().tho_device_idx()).unwrap_or(0)
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
    fn protect(&self, device: usize) -> thoosa_common::runner::residency::ResidencyLease { thoosa_common::runner::residency::ResidencyLease::combine(self.x_1.iter().chain(&self.x_0).map(|value| value.protect(device))) }

    type Out<'a> = SyncLabelBatchInput<'a> where Self: 'a;

    fn borrow_ref<'a>(&'a self, to_device_idx: usize) -> impl Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        async move {
            let x_1 = try_join_all(self.x_1.iter().map(|value| BorrowRef::borrow_ref(value, to_device_idx))).await?;
            let x_0 = try_join_all(self.x_0.iter().map(|value| BorrowRef::borrow_ref(value, to_device_idx))).await?;
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
            self.x.with_id(thoosa_shared::debug_id!("sync_label_batch_x")),
            Some(BatchLengths::new(self.lengths)),
            self.y.with_id(thoosa_shared::debug_id!("sync_label_batch_y")),
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

static SYNC_LABEL_BATCH_INSTANCES: InstanceCache = InstanceCache::new();

impl <C> SyncLabelBatch<C>
where
    C: RunCtx + 'static
{
    pub async fn new(tho_device_idx:usize) -> Result<Arc<DynSyncStageEnum<SyncLabelBatch<C>>>, ThoError> {
        SYNC_LABEL_BATCH_INSTANCES.get_or_try_insert_recreatable_with::<C, _, _, _, _>(tho_device_idx, GlobalCtx::global().cuda_activity_gate(tho_device_idx), GlobalCtx::global().is_ipc(), 0, move || async move {
            if GlobalCtx::global().is_ipc() {

                Ok(Arc::new(DynSyncStageEnum::Proxy {
                    stage: Arc::new(ProxyStage::proxy_recreatable(SYNC_LABEL_BATCH_ENDPOINT, Some(0.to_string()), tho_device_idx, Box::new(())).await?),
                    init: (),
                }))
            } else {
                Ok(Arc::new(DynSyncStageEnum::Local(Arc::new(Self::new_local().await?))))
            }
        }).await


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
            // An absent class is a data incompatibility, distinct from failure to load its resources.
            return Err(ThoError::Data { kind: thoosa_shared::DataErrorKind::EmptyTrainingClass, msg: format!("SyncLabelBatch: empty positive or negative resources for {}", input.label) });
        }

        let max_input = x_1.iter()
            .chain(x_0.iter())
            .map(|value| value.data().size().get(1).copied().ok_or_else(|| ThoError::new(format!("SyncLabelBatch: invalid motif output shape for {}", input.label))))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .ok_or_else(|| ThoError::new(format!("batch resources are empty for {}", input.label)))?;

        // Borrow one handle at a time: collecting guards could lock shared inputs in opposite orders.
        // These aliases stay in this synchronous call and are copied into the padded batches.
        let x_1_tensors = x_1.iter().map(|value| tch_result(value.data().local()?.f_alias())).collect::<Result<Vec<_>, ThoError>>()?;
        let x_0_tensors = x_0.iter().map(|value| tch_result(value.data().local()?.f_alias())).collect::<Result<Vec<_>, ThoError>>()?;
        let x_1_refs = x_1_tensors.iter().collect::<Vec<_>>();
        let x_0_refs = x_0_tensors.iter().collect::<Vec<_>>();

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


pub fn sync_label_batch_endpoint_init<'a>(_stage_id:Option<String>, _init: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, _all_cuda_tensors: &'a AllTensors, _tho_device:&'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
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
    run_configs: Arc<thoosa_shared::run_configs::TrainingRunConfigs<TrainLabelNetRunConfigFields>>,
    pd: PhantomData<L>,
    applied_motif_loader: Arc<LM>,
    #[expect(dead_code, reason = "Retains the shared owner for the lifetime of this stage or input.")]
    ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
    num_nets:u32,
    store: Arc<dyn MotifsStore>,
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


pub async fn new_train_label_nets<L, AMN, C>(
    train_motif_net_run_config_ids: Vec<RunConfigId>,
    loader: Arc<L>,
    apply_motif_nets: Arc<AMN>,
    run_configs: thoosa_shared::run_configs::TrainingRunConfigs<TrainLabelNetRunConfigFields>,
    ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
    store: &Arc<dyn MotifsStore>,
    opt_config:Arc<OptConfig>,
    labels: Option<Vec<String>>,
    ctx: C,
) -> Result<TrainLabelNets<L, PairStage<L, AMN, C>, C>, ThoError>
where
    L: for <'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx=C> + Cached + OutId + 'static,
    AMN: AppliedMotifNetsStage<C>,
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    let _id = mono_key!(RAW_TENSOR_LOADER, "stage") + &SynKey::new(train_motif_net_run_config_ids.iter().map(|run_config_id| mono_key!(*run_config_id, format!("train_motif_nets_run_config_ids_{run_config_id}"))).collect::<Vec<_>>());

    let num_nets = apply_motif_nets.num_nets() as u32;
    if num_nets == 0 {
        return Err(ThoError::new("No motifs nets to train on"));
    }
    let motif_nets_ids = apply_motif_nets.motif_nets_ids();


    let applied_motif_loader = Arc::new(PairStage::new(
        loader,
        apply_motif_nets
    ));

    t!(TrainLabelNets::new(
        train_motif_net_run_config_ids,
        motif_nets_ids,
        applied_motif_loader,
        num_nets,
        run_configs,
        ws,
        store,
        opt_config,
        labels,
        ctx
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
        run_configs: thoosa_shared::run_configs::TrainingRunConfigs<TrainLabelNetRunConfigFields>,
        ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
        store: &Arc<dyn MotifsStore>,
        opt_config:Arc<OptConfig>,
        labels: Option<Vec<String>>,
        ctx:C
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

            sync_label_batch:  SyncLabelBatch::<C>::new(ctx.tho_device_idx()).await?,
        })
    }

}

#[derive(Debug, CpuData)]
#[cpu_data(shallow)]
pub struct TrainLabelNetsOutput {
    train_motif_nets_run_config_ids: Vec<RunConfigId>,
    motif_nets_ids:Vec<u32>,
    results: Vec<TrainLabelNetRunConfigResult>
}

impl Output for TrainLabelNetsOutput {

}


#[async_trait]
impl SavableResult for TrainLabelNetsOutput {
    type Store = dyn MotifsStore;
    type Workspace = dyn crate::motifs_ws::MotifsWorkspace;
    type R = Vec<u32>;

    async fn save(&mut self, store: &Arc<dyn MotifsStore>, ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, _p: Self::P<'_>) -> anyhow::Result<Vec<u32>> {

        let mut run_config_ids = Vec::with_capacity(self.results.len());

        for result in &mut self.results {

            run_config_ids.push(
                t!(result.save(store, ws, TrainLabelNetRunConfigResultParams {
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
            let (train_labels_resource_keys, valid_label_resource_keys) = t!(resources_config.to_resources(&self.store)).await?;

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
                let ctx_for_task = ctx.branch(true);

                let opt_config = Arc::clone(&self.opt_config);
                let train_label_nets_run_configs = Arc::clone(&train_label_nets_run_configs);
                let applied_motif_loader = Arc::clone(&self.applied_motif_loader);

                debug!("TrainLabelNets::run: label={label} train_motif_nets_run_config_ids={:?} train_label_nets_run_configs={train_label_nets_run_configs:?}", self.train_motif_nets_run_config_ids);
                let _motif_nets_ids = self.motif_nets_ids.clone();

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
                                           //TODO




                                           let input = Arc::new(SyncLabelBatchInputWrapper::new(x_1_wrapped, x_0_wrapped, num_nets as i64, label));
                                           let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoStageAdjuster, _, ()> = OOMHandler::new(vec![
                                               Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(*MaxParallelStep::FAST))),
                                               Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("sync_label_batch"))),
                                           ], None);
                                           let output = t!(oom_handler.run(
                                               &sync_label_batch,
                                               input,
                                               ctx.branch(false),
                                               WithRunCtxContext::new(
                                                   MaxParallelAndDumpToCpuAdjustableContex::new(
                                                       MaxParallelAdjustContextImpl::new(),
                                                       DumpToCpuContextImpl::new(),
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

                    let result = t!(train_net.run(
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

            let results = t!(collect_futures_with_progress(
                futures,
                labels_train_and_valid_sets_len,
                FuturesOrdering::Unordered,
                thoosa_shared::CollectionErrorPolicy::SkipAll,
                "train labels",
            )).await?;

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

                let futures = futures.map(|future| async move { Ok::<_, ThoError>(t!(future).await) });

                debug!("ResourcesBatch::build_batch_stream: descr={descr} futures.len={}", futures.len());

                let (x_1_wrapped, x_0_wrapped) = t!(collect_futures_with_progress(
                    futures,
                    batch_len,
                    FuturesOrdering::Ordered,
                    thoosa_shared::CollectionErrorPolicy::Propagate,
                    "ResourcesBatch::build_batch_stream",
                )).await?
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

                let r: Result<(ThoBatch<X>, Option<BatchLengths>, ThoBatch<Y>), ThoError> = t!(build_batch_fn(x_1_wrapped, x_0_wrapped, ctx_for_build_fn)).await;

                debug!("ResourcesBatch::build_batch_stream: descr={descr} batch_i={i} r={r:?}");

                r
            }
        });

        // TrainNet polls one batch ahead and owns its temporary cache and drain.
        futures.boxed()
    }

}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainLabelStepIpcInit {
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

fn train_label_loss(pred_y: &Tensor, target: &Tensor) -> anyhow::Result<Tensor> {
    Ok(pred_y.f_binary_cross_entropy_with_logits::<Tensor>(target, None, None, Reduction::Mean)?)
}

impl WorkerStageInput for TrainLabelStepIpcInit {
}

#[typetag::serde(name = "TrainLabelStepIpcInit")]
impl WorkerInput for TrainLabelStepIpcInit {}


#[derive(Debug, Serialize, Deserialize)]
#[serde(transparent)]
struct TrainLabelStepInput(TrainStepInput<MotifProbability, LabelProbability>);

#[typetag::serde(name = "TrainLabelStepInput")]
impl WorkerInput for TrainLabelStepInput {}

impl std::borrow::Borrow<TrainStepInput<MotifProbability, LabelProbability>> for TrainLabelStepInput {
    fn borrow(&self) -> &TrainStepInput<MotifProbability, LabelProbability> {
        &self.0
    }
}


fn train_label_step_endpoint_init<'a>(_stage_id:Option<String>, init: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, _all_cuda_tensors: &'a AllTensors, _tho_device:&'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let Ok(init) = (init as Box<dyn std::any::Any>).downcast::<TrainLabelStepIpcInit>() else {
            return Err(ThoError::new("TrainLabelStep endpoint received invalid initializer"));
        };
        let init = *init;
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
        Ok(Box::new(WorkerWrapperWithState::<JsonIpcCodec, _, _, _>::new_with_checkpoint_input::<TrainLabelStepInput>(Arc::new(train_step), PhantomData)) as Box<dyn SyncStageIpcWrapper>)
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
    async fn new_step(&self, run_config: TrainLabelNetRunConfig, ctx: C, vs_path: PathBuf) -> Result<(DynSyncStageWithStateEnum<SyncTrainLabelStep<C>>, LabelNetParams), ThoError> {
        let node_params = train_label_step_params(&run_config, self.num_nets);
        if GlobalCtx::global().is_ipc() {


            let init = TrainLabelStepIpcInit { params: node_params.clone(), training_config: run_config.common_training_args.clone(), tho_device_idx: ctx.tho_device_idx(), vs_path };
            let input_cloner = Arc::new(|input: &TrainStepInput<MotifProbability, LabelProbability>| -> Result<Box<dyn WorkerInput>, ThoError> { Ok(Box::new(TrainLabelStepInput(input.clone_refs_for_worker()?))) });
            Ok((DynSyncStageWithStateEnum::Proxy {
                stage: Arc::new(ProxyStage::proxy_with_input_cloner(TRAIN_LABEL_STEP_ENDPOINT, Some(SyncTrainLabelStep::<C>::next_id().to_string()), ctx.tho_device_idx(), Box::new(init), input_cloner).await?),
                init: (),
            }, node_params))
        } else {
            let params_for_module = node_params.clone();

            let (train_step, _) = TrainStep::new_local(
                ctx.tho_device_idx().into(),
                move |run_config, device| async move {

                    let config = ModConfig::new(
                        run_config.common_training_args.tho_optimizer(),
                        move |var_store| params_for_module.label_net()?.build_tch_module::<Batch<MotifProbability>, Batch<LabelProbability>>(&var_store.root()),
                        run_config.common_training_args.lr,
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

pub fn train_label_step_endpoint_inits() -> HashMap<String, MotifsInitFn> {
    HashMap::from([(
        TRAIN_LABEL_STEP_ENDPOINT.to_string(),
        Box::new(train_label_step_endpoint_init) as MotifsInitFn,
    )])
}

#[cfg(test)]
mod tests {
    use crate::dims::LabelProbability;
    use crate::motifs_store::{MockMotifsStoreBuilder, MotifsStore, create_motif_tables};
    use crate::stages::apply_motif_nets::{ApplyMotifNets, motif_test_net_};
    use crate::stages::cross_loader::ResourcesConfigInput;
    use crate::stages::stft::SyncStftSelector;
    use crate::stages::tensor_loader::mock_tensor_loader;
    use crate::stages::train_label_nets::{ResourcesBatch, TrainLabelNetRunConfig, TrainLabelNetRunConfigFields, TrainLabelNetRunConfigResult, new_train_label_nets};
    use crate::tch_test_utils::motifs_tensor;
    use crate::tch_test_utils::resource_tensor;
    use crate::train_nets::{LabelNetParams, LabelNetTypeArgs, LabelNetTypeArgsFields, LabelNetTypeParams, LinNetArgs, LinNetArgsFields, TrainLabelNetRunConfigResultParams, TrainMotifNetRunConfig};
    use futures::FutureExt;
    use futures_util::StreamExt;
    use futures_util::future::try_join_all;
    use log::LevelFilter::Debug;
    use log::{Level, debug, log_enabled};
    use parking_lot::RwLock;
    use rand::{RngExt, rng};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use tch::Device::Cpu;
    use tch::Tensor;
    use tch::kind::FLOAT_CPU;
    use tch::nn::VarStore;
    use thoosa_common::dims::ChannelsData;
    use thoosa_common::nets::batches::AsyncDataBatch;
    use thoosa_common::nets::common_nets::{CommonTrainingRunConfig, CommonTrainingRunConfigFields, LinLenNetParams, OptConfig};
    use thoosa_common::resources::{Labeled, NetResources, SavableResult};
    use thoosa_common::runner::common::cachable::{DevicesWrap, SelectRef};
    use thoosa_common::runner::common::{MockRunCtx, mock_run_ctx};
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::tch_error::tch_result;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::utils::{mix_x_binary_with_labels, reset_print_options};
    use thoosa_common::ws::WsConfigs;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::hyperparameter::Hyperparameter;
    use thoosa_shared::perf_logging::{PERF, format_perf_map};
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::{RunConfig, RunConfigId};
    use thoosa_shared::test_utils::TestSignalKind;
    use thoosa_shared::traits::AsyncStage;
    use thoosa_shared::traits::RunCtx;
    use thoosa_shared::val::{NamedVal, Val};
    use thoosa_shared::{TensorIdSliceIdx, ThoError};
    use thoosa_shared::{cdt_for_dir, mono_key};
    use thoosa_store::{SqlStore, TrainedNet, TrainedNetsBy};
    use tokio::fs;

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
        let _find_motifs_run_config_id: RunConfigId = 1;
        let select_motifs_run_config_id: RunConfigId = 2;
        let motif_width = 100;

        let motif_0_0_id = 0; //channel 0 motif 0
        let motif_0_1_id = 1; //channel 0 motif 1
        let motif_1_0_id = 0; //channel 1 motif 0

        let pulse_motif = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_0_0", "motif_id")]), motifs_tensor(100, 3, TestSignalKind::Pulse, 0.3, 0.4))?);
        let triangle_motif = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_0_1", "motif_id")]), motifs_tensor(100, 3, TestSignalKind::Triangle, 0.3, 0.4))?);
        let sin_motif = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_1_0", "motif_id")]), motifs_tensor(100, 3, TestSignalKind::Sin, 0.3, 0.4))?);
        let train_group = "train_group";
        let valid_group = "valid_group";

        #[expect(deprecated, reason = "Preserve this regression's legacy channel-indexed mock behavior.")]
        let ws = Arc::new(RwLock::new(
            crate::motifs_ws::MotifsWsMockBuilder::new(None, 3, 100, input_n_0 as u32)
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

        let (resources_map, _motif_intervals) = resources_a.iter().zip(resources_b.iter().zip(resources_c.iter().zip(resources_x.iter()))).try_fold((HashMap::new(), vec![]), |mut a, (r_a, (r_b, (r_c, r_x)))| {

            let (t, intervals) = resource_tensor(r_a.length() as usize, &[
                            vec![(3, motif_width, TestSignalKind::Pulse)],
                            vec![(1, motif_width, TestSignalKind::Sin)],
                            vec![]
                        ], 100, 10.0,
                                                 true,
                                                 true)?;

            a.0.insert(r_a.clone(), t);

            a.1.extend(intervals[0][0].iter().map(|interval|TensorIdSliceIdx(r_a.id(), interval.start as u32)));


            let (t, intervals) = resource_tensor(r_b.length() as usize, &[
                            vec![(3, motif_width, TestSignalKind::Triangle)],
                            vec![(1, motif_width, TestSignalKind::Sin)],
                            vec![]
                        ], 100, 10.0,
                                                 true,
                                                 true)?;

            a.0.insert(r_b.clone(), t);

            a.1.extend(intervals[0][0].iter().map(|interval|TensorIdSliceIdx(r_b.id(), interval.start as u32)));


            let (t, intervals) = resource_tensor(r_c.length() as usize, &[
                            vec![(3, motif_width, TestSignalKind::Sin)],
                            vec![(1, motif_width, TestSignalKind::Sin)],
                            vec![]
                        ], 100, 10.0,
                                                 true,
                                                 true)?;

            a.0.insert(r_b.clone(), t);

            a.1.extend(intervals[0][0].iter().map(|interval|TensorIdSliceIdx(r_c.id(), interval.start as u32)));


            let (t, _intervals) = resource_tensor(r_x.length() as usize, &[
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

        let store = MockMotifsStoreBuilder::new()
            .with_list_selected_motifs([
                (vec![select_motifs_run_config_id], Ok(
                    HashMap::from([(
                        select_motifs_run_config_id,
                        vec![
                            vec![(motif_0_0_id, 1.0f64), (motif_0_1_id, 1.0f64)], //channel 0
                            vec![(motif_1_0_id, 1.0f64)], //channel 1
                            vec![] //channel 2
                        ]
                    )])
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
            .build();


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
            mock_run_ctx::default(1),
        ).await?);


        let input = Arc::new(ResourcesConfigInput::new(NetResources::new(Labeled::new(train_group.to_string()), Labeled::new("valid_group".to_string()))));


        let _run_ctx = mock_run_ctx::default(1);

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
            Arc::new(|x_1_wrapped:Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>, x_0_wrapped:Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>, ctx: MockRunCtx|{
                async move {
                    let x_1 = try_join_all(x_1_wrapped.iter().map(|x| x.select_ref(ctx.tho_device_idx(), None::<SyncStftSelector>))).await?;
                    let x_0 = try_join_all(x_0_wrapped.iter().map(|x| x.select_ref(ctx.tho_device_idx(), None::<SyncStftSelector>))).await?;
                    {
                        debug!("ResourcesBatch::train_batch_iter: x_1={x_1:?} x_0={x_0:?}");
                        let x_1 = tch_result(Tensor::f_stack(&x_1.into_iter().map(|v| tch_result(v.local()?.f_detach_copy())).collect::<Result<Vec<_>, ThoError>>()?, 0))?;
                        let x_0 = tch_result(Tensor::f_stack(&x_0.into_iter().map(|v| tch_result(v.local()?.f_detach_copy())).collect::<Result<Vec<_>, ThoError>>()?, 0))?;

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
            Arc::new(|x_1_wrapped:Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>, x_0_wrapped:Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>, ctx: MockRunCtx|{
                async move {
                    let x_1 = try_join_all(x_1_wrapped.iter().map(|x| x.select_ref(ctx.tho_device_idx(), None::<SyncStftSelector>))).await?;
                    let x_0 = try_join_all(x_0_wrapped.iter().map(|x| x.select_ref(ctx.tho_device_idx(), None::<SyncStftSelector>))).await?;
                    {
                        debug!("ResourcesBatch::train_batch_iter: x_1={x_1:?} x_0={x_0:?}");
                        let x_1 = tch_result(Tensor::f_stack(&x_1.into_iter().map(|v| tch_result(v.local()?.f_detach_copy())).collect::<Result<Vec<_>, ThoError>>()?, 0))?;
                        let x_0 = tch_result(Tensor::f_stack(&x_0.into_iter().map(|v| tch_result(v.local()?.f_detach_copy())).collect::<Result<Vec<_>, ThoError>>()?, 0))?;

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


    #[tokio::test]
    async fn test_save_train_label_net_result_persists_trained_net_and_loss() -> Result<(), ThoError> {
        let store: Arc<dyn MotifsStore> = Arc::new(SqlStore::in_mem_(vec![
            thoosa_store::create_tables,
            create_motif_tables
        ]).await?);
        store.bind_run(store.save_run("test".into()).await?)?;

        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>> = Arc::new(RwLock::new(crate::motifs_ws::new_motifs_workspace(test_dir.clone(), WsConfigs::new(2, 1000, 32000))?));
        let train_motif_nets_run_config_id = store.save_run_config(None, TrainMotifNetRunConfig::type_name(), "source", HashMap::new()).await?;
        // Trained results now own a worker-written checkpoint rather than an in-memory VarStore.
        let checkpoint_path = test_dir.join("train_label_checkpoint.safetensors");
        tch_result(VarStore::new(Cpu).save(&checkpoint_path))?;

        let mut result = TrainLabelNetRunConfigResult::new(
            // vec![source_run_config_id],
            "alpha".to_string(),
            0.125,
            LabelNetParams::new(4, LabelNetTypeParams::LinNetParams(LinLenNetParams::new(16, 2))),
            TrainLabelNetRunConfig::new(None, CommonTrainingRunConfig::default(), LabelNetTypeArgs::LinNet(LinNetArgs::new(2, 16)), 1),
            thoosa_shared::tmp_file::TmpFile::new(checkpoint_path),
        );

        let train_motif_nets_run_config_id_vec = vec![train_motif_nets_run_config_id];
        let motif_nets_ids= vec![];
        let train_label_nets_run_config_id = SavableResult::save(&mut result, &store, &ws, TrainLabelNetRunConfigResultParams{
            train_motif_nets_run_config_ids: &train_motif_nets_run_config_id_vec,
            motif_nets_ids: &motif_nets_ids,
        }).await?;
        let _run_config_values = store.list_run_config_values(train_label_nets_run_config_id).await?;
        let trained_nets = store.list_trained_nets(TrainedNetsBy::RunConfigIds(vec![train_label_nets_run_config_id]), None, None).await?;

        assert_eq!(vec![train_label_nets_run_config_id], store.list_output_run_config_ids(train_motif_nets_run_config_id).await?);
        assert_eq!(vec![TrainedNet { id: 1, run_config_id: train_label_nets_run_config_id, net_path: PathBuf::from(format!("alpha/{train_label_nets_run_config_id}")), loss: 0.125 }], trained_nets);

        Ok(())
    }

}

#[derive(Debug)]
pub struct TrainLabelNetRunConfigResult {
    #[expect(dead_code, reason = "Retains stored metadata and its existing construction and diagnostic representation.")]
    train_label_nets_run_config_id: Option<RunConfigId>,
    pub label: String,
    pub loss: f32,
    label_node_params: LabelNetParams,
    train_label_nets_run_config: TrainLabelNetRunConfig,
    var_store_path: TmpFile,
    motif_nets_ids:Vec<u32>
}

impl TrainLabelNetRunConfigResult {
    pub fn new(
        label: String,
        loss: f32,
        label_node_params: LabelNetParams,
        train_label_nets_run_config: TrainLabelNetRunConfig,
        var_store_path: TmpFile,
    ) -> Self {
        Self {
            train_label_nets_run_config_id: None,
            label,
            loss,
            label_node_params,
            train_label_nets_run_config,
            var_store_path,
            motif_nets_ids: vec![],
        }
    }
}

#[async_trait]
impl SavableResult for TrainLabelNetRunConfigResult {
    type Store = dyn MotifsStore;
    type Workspace = dyn crate::motifs_ws::MotifsWorkspace;
    type P<'a> = TrainLabelNetRunConfigResultParams<'a>;

    type R = u32;

    async fn save(
        &mut self,
        store: &Arc<dyn MotifsStore>,
        ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
        p: Self::P<'_>
    ) -> anyhow::Result<u32> {

        let run_config_id = t!(store.save_run_config(
            Some(p.train_motif_nets_run_config_ids), //it's used as input for TrainLabelNetsOutput
            &TrainLabelNetRunConfig::type_name(),
            "Label",
            self.train_label_nets_run_config.to_properties_iter(None).collect(),
        )).await?;

        let path = panic_to_err!(ws.read()
            .save_label_net(
                run_config_id,
                &self.label,
                (&self.var_store_path).into(),
                &self.label_node_params,
            ))?;

        debug!("TrainLabelNetRunConfigResult::save: self.motif_nets_ids={:?}", self.motif_nets_ids);

        t!(store.save_trained_net(run_config_id, path, self.loss, Some(p.motif_nets_ids.clone()))).await?;

        Ok(run_config_id)
    }
}

#[derive(Args, Debug, Clone)]
pub struct TrainLabelNetsArgs {
    #[arg(long, value_delimiter = ',', value_parser = thoosa_common::utils::parse_run_config_ids)]
    pub train_motif_nets_run_config_ids: Vec<RunConfigId>,

    #[arg(long)]
    pub train_resource_group: String,

    #[arg(long)]
    valid_resource_group: String,

    #[command(flatten)]
    pub opt_config: OptConfig,

    #[arg(long)]
    pub run_config_cnt: String, //TrainLabelNetRunConfig

    #[arg(long)]
    pub max_motif_net_loss: Option<f32>,

    #[arg(long)]
    pub selected_motif_nets_limit: Option<usize>,

    #[arg(long)]
    pub motif_slices_step: Option<u32>,

    #[arg(long)]
    pub labels: Option<String>,
}

impl TrainLabelNetsArgs {
    pub fn resource_config(&self) -> NetResources<Labeled> {
        NetResources::new(
            Labeled::new(self.train_resource_group.clone()),
            Labeled::new(self.valid_resource_group.clone()),
        )
    }

    fn parse_labels_csv(raw: &str) -> Vec<String> {
        raw.split(',').map(|label| label.trim().to_string()).filter(|label| !label.is_empty()).collect()
    }

    pub fn labels_filter(&self) -> Option<Vec<String>> {
        self.labels.as_deref().map(Self::parse_labels_csv).filter(|labels| !labels.is_empty())
    }
}

#[allow(non_snake_case, unused_assignments, unused_variables)]
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
#[run_config_opt]
pub struct TrainLabelNetRunConfig {
    #[in_map]
    #[config_scalar]
    pub id: Option<RunConfigId>,

    #[in_map]
    #[run_config]
    pub common_training_args: CommonTrainingRunConfig,

    #[in_map]
    #[run_config]
    pub net: LabelNetTypeArgs,

    #[in_map]
    #[run_config]
    pub validation_step: u32,
}

impl WithCommonTrainingRunConfig for TrainLabelNetRunConfig {
    fn config(&self) -> &CommonTrainingRunConfig {
        &self.common_training_args
    }
}

impl AsRef<CommonTrainingRunConfig> for TrainLabelNetRunConfig {
    fn as_ref(&self) -> &CommonTrainingRunConfig {
        &self.common_training_args
    }
}

impl TrainLabelNetRunConfig {
    pub fn new(
        id: Option<RunConfigId>,
        common_training_args: CommonTrainingRunConfig,
        net: LabelNetTypeArgs,
        validation_step: u32,
    ) -> Self {
        Self {
            id,
            common_training_args,
            net,
            validation_step,
        }
    }
}
