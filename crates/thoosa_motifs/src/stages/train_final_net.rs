use crate::dims::{LabelProbability, LabelsProbabilities};
use crate::motifs_store::MotifsStore;
use crate::stages::apply_label_nets::ApplyLabelNets;
use crate::stages::cross_loader::ResourcesConfigInput;
use async_trait::async_trait;
use futures::FutureExt;
use futures_util::StreamExt;
use futures_util::future::BoxFuture;
use futures_util::future::try_join_all;
use futures_util::stream::BoxStream;
use log::Level::{Debug as DebugLevel, Trace};
use log::{debug, info, log_enabled, trace};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::future::ready;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use thoosa_common::dims::ChannelsData;
use thoosa_common::runner::global_ctx::CacheKey;
use thoosa_common::stages::stages_seq::PairStage;
use thoosa_common::stages::train_net::{TrainNet, TrainNetInput, TrainStepFactory};
use thoosa_common::tch_error::tch_result;

use tch::nn::AdamW;
use tch::{Kind, Tensor};
use thoosa_common::dims::{Batch, ThoBatch};
use thoosa_common::instance_cache::InstanceCache;
use thoosa_common::ipc::cuda_wokrer::{AllTensors, InitFn, WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use thoosa_common::ipc::serialization::json::JsonIpcCodec;
use thoosa_common::nets::batches::{AsyncDataBatch, BatchLengths};
use thoosa_common::stages::train_step::{TrainStep, TrainStepInput, TrainStepXFn};

type MotifsInitFn = InitFn<dyn crate::motifs_ws::MotifsWorkspace>;
use crate::stages::apply_motif_nets::ApplyMotifNets;
use crate::train_nets::final_net;
use crate::train_nets::{FinalNetParams, TrainFinalNetRunConfig, TrainFinalNetRunConfigFields};
use thoosa_common::ipc::proxy::{DynSyncStageEnum, DynSyncStageWithStateEnum, ProxyStage};
use thoosa_common::ipc::worker_msg::WorkerInput;
use thoosa_common::ipc::wraps::SyncStageIpcWrapper;
use thoosa_common::ipc::wraps::{WorkerWrapper, WorkerWrapperWithState};
use thoosa_common::nets::common_nets::{CommonTrainingRunConfig, OptConfig};
use thoosa_common::nets::tho_varstore::{ModConfig, ThoTrainModule};
use thoosa_common::resources::{Logits, NetResources, SavableResult};
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep};
use thoosa_common::runner::common::cachable::{BorrowRef, Cached, DevicesWrap, ThoDataRef};
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::oom_handler::{Adjusters, NoInputAdjuster, NoStageAdjuster, OOMHandler, WithRunCtxContext, run_ctx_adjuster};
use thoosa_common::runner::sync_stage::SyncStage;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_cpu_data_derive::CpuData;
use thoosa_properties_map::ToHashmap;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::ipc_types::WorkerOutput;
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_shared::run_configs::{RunConfig, RunConfigId};
use thoosa_shared::tmp_file::TmpFile;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::MaxParallelCtx;
use thoosa_shared::traits::{Adjustable, AsyncStage, Data, OutId, RunCtx, Stage, WithId};
use thoosa_shared::{FuturesOrdering, ThoError, collect_futures_with_progress, mono_key, shuffle_and_chunk_with_rng};

pub const TRAIN_FINAL_STEP_ENDPOINT: &str = "TrainFinalStep";

#[derive(Debug)]
pub struct TrainFinalNet<L, LL, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    LL: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<LabelProbability, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    train_label_nets_run_config_ids: Vec<RunConfigId>,
    run_configs: Arc<thoosa_shared::run_configs::TrainingRunConfigs<TrainFinalNetRunConfigFields>>,
    pd: PhantomData<L>,
    label_loader: Arc<LL>,
    num_label_nets: u32,
    // label_net_run_config_ids: Vec<RunConfigId>,
    #[expect(dead_code, reason = "Retains the shared owner for the lifetime of this stage or input.")]
    ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
    store: Arc<dyn MotifsStore>,
    opt_config: Arc<OptConfig>,
}

pub async fn new_train_final_net<L, C, R>(
    train_label_nets_run_config_ids: Vec<RunConfigId>,
    loader: Arc<L>,
    run_configs: thoosa_shared::run_configs::TrainingRunConfigs<TrainFinalNetRunConfigFields>,
    ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
    store: &Arc<dyn MotifsStore>,
    stage_ctx: R,
    opt_config: Arc<OptConfig>,
    slices_step: u32,
) -> Result<TrainFinalNet<L, PairStage<L, ApplyLabelNets<C, ApplyMotifNets<C>>, C>, C>, ThoError>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    R: RunCtx + MaxParallelCtx + Adjustable + 'static,

{
    // Final training consumes the complete ordered label-feature schema; limiting
    // motif models belongs to label training rather than this constructor.
    //Build a final-net application stage that wraps the raw loader with motif+label inference.
    let apply_label_nets = Arc::new(track_await!(ApplyLabelNets::new_with_apply_motif_nets(ws, store, train_label_nets_run_config_ids.clone(), stage_ctx, slices_step)).await?);

    let label_loader = Arc::new(PairStage::new(
        Arc::clone(&loader),
        Arc::clone(&apply_label_nets),
    ));

    track_await!(TrainFinalNet::new(train_label_nets_run_config_ids, label_loader, apply_label_nets.num_nets() as u32, run_configs, ws, store, opt_config)).await
}

impl<L, LL, C> TrainFinalNet<L, LL, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    LL: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<LabelProbability, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    pub async fn new(
        train_label_nets_run_config_ids: Vec<RunConfigId>,
        label_loader: Arc<LL>,
        num_label_nets: u32,
        run_configs: thoosa_shared::run_configs::TrainingRunConfigs<TrainFinalNetRunConfigFields>,
        ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
        store: &Arc<dyn MotifsStore>,
        opt_config: Arc<OptConfig>,
    ) -> Result<Self, ThoError> {
        debug!("TrainFinalNet::new: train_label_nets_run_config_ids={train_label_nets_run_config_ids:?}");

        Ok(Self {
            train_label_nets_run_config_ids,
            run_configs: Arc::new(run_configs),
            pd: PhantomData,
            label_loader,
            num_label_nets,
            ws: Arc::clone(ws),
            store: Arc::clone(store),
            opt_config,
        })
    }
}

#[derive(Debug, CpuData)]
#[cpu_data(shallow)]
pub struct TrainFinalNetOutput {
    result: TrainFinalNetsRunConfigResult,
}

impl TrainFinalNetOutput {
    pub fn new(result: TrainFinalNetsRunConfigResult) -> Self {
        Self { result }
    }
}

impl Output for TrainFinalNetOutput {}


#[async_trait]
impl SavableResult for TrainFinalNetOutput {
    type Store = dyn MotifsStore;
    type Workspace = dyn crate::motifs_ws::MotifsWorkspace;

    type R = Vec<u32>;

    async fn save(&mut self, store: &Arc<dyn MotifsStore>, ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, _p: Self::P<'_>) -> anyhow::Result<Vec<u32>> {
        // Report success only after persistence completes, using the actual saved run ID.
        Ok(vec![track_await!(self.result.save(store, ws, ())).await?])
    }
}

impl<L, LL, C> Stage for TrainFinalNet<L, LL, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    LL: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<LabelProbability, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = Arc<ResourcesConfigInput<NetResources<Logits>>>;
    type Out = Arc<TrainFinalNetOutput>;
}

impl<L, LL, C> AsyncStage for TrainFinalNet<L, LL, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    LL: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<LabelProbability, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "TrainFinalNet::run"))]
    fn run(self: Arc<Self>, input: Self::In<'static>, mut ctx: Self::Ctx) -> impl Future<Output = Result<Self::Out, ThoError>> + Send {

        let train_final_run_configs = Arc::clone(&self.run_configs);
        let label_loader = Arc::clone(&self.label_loader);
        let train_label_nets_run_config_ids = self.train_label_nets_run_config_ids.clone();
        let num_label_nets = self.num_label_nets;
        let opt_config = Arc::clone(&self.opt_config);
        let store = Arc::clone(&self.store);

        async move {
            //Load train/valid logits resources and validate label vector sizes.
            let resources_config = input.resources_config().clone();
            let (train_resources, valid_resources): (Vec<(Resource, Vec<f32>)>, Vec<(Resource, Vec<f32>)>) =
                track_await!(resources_config.to_resources(&store)).await?;

            let labels_len = train_resources.first().ok_or_else(|| ThoError::new("TrainFinalNet::run: no train resources"))?.1.len();
            if labels_len == 0 {
                return Err(ThoError::new("TrainFinalNet::run: empty label vector"));
            }
            for (_res, labels) in train_resources.iter().chain(valid_resources.iter()) {
                if labels.len() != labels_len {
                    return Err(ThoError::new("TrainFinalNet::run: inconsistent label vector sizes"));
                }
            }

            let train_resources = Arc::new(train_resources);
            let valid_resources = Arc::new(valid_resources);

            let train_net = Arc::new(TrainNet::new(
                move |_ctx: C, rc: &TrainFinalNetRunConfig| {
                    let label_loader = Arc::clone(&label_loader);
                    let train_resources = Arc::clone(&train_resources);
                    let valid_resources = Arc::clone(&valid_resources);

                    ready(LogitsResourcesBatch::new(label_loader, rc.common_training_args.batch_size, train_resources, valid_resources, labels_len)).boxed()
                },
                Arc::new(TrainFinalStepFactory::new(num_label_nets, labels_len as u32, GlobalCtx::global().execution(ctx.tho_device_idx()))),
                opt_config,
                train_final_run_configs,
                |rc: &TrainFinalNetRunConfig| rc.common_training_args.epochs,
                1,
                "final".to_string(),
            ));

            let result = track_await!(train_net.run(
                Arc::new(TrainNetInput::new()),
                ctx.branch(false))).await?;

            let final_result = TrainFinalNetsRunConfigResult::new(Some(&train_label_nets_run_config_ids), result.min_loss, result.var_store_path, result.node_params, result.run_config);
            info!("TrainFinalNet::run, done: min_loss={:?}", final_result.loss);

            Ok(Arc::new(TrainFinalNetOutput::new(final_result)))
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct SyncLogitsBatchInput<'a> {
    id: SynKey,
    outputs: Vec<ThoDataRef<'a, ThoTensor<LabelProbability, SynKey>>>,
    labels: Vec<Vec<f32>>,
    labels_len: usize,
}

impl WorkerStageInput for SyncLogitsBatchInput<'_> {
    fn clone_refs(&self) -> Result<Box<dyn WorkerInput>, ThoError> {
        // Own cloned worker references so the borrowed batch cannot escape into typetag.
        let outputs = self.outputs.iter().map(|output| Ok(ThoDataRef::Owned(output.as_ref().clone_ref()?))).collect::<Result<Vec<_>, ThoError>>()?;
        Ok(Box::new(SyncLogitsBatchInput { id: self.id.clone(), outputs, labels: self.labels.clone(), labels_len: self.labels_len }))
    }

    fn ref_to_tensor(&self, all_cuda_tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {
        let outputs = self.outputs.iter().map(|output| Ok(ThoDataRef::Owned(WorkerStageInput::ref_to_tensor(output.as_ref(), all_cuda_tensors, remove)?))).collect::<Result<Vec<_>, ThoError>>()?;
        Ok(Self { id: self.id.clone(), outputs, labels: self.labels.clone(), labels_len: self.labels_len })
    }
}

#[typetag::serde(name = "SyncLogitsBatchInput")]
impl WorkerInput for SyncLogitsBatchInput<'static> {}

impl Data for SyncLogitsBatchInput<'_> {
    type Metrics = i64;


    fn size(&self) -> Self::Metrics {
        self.outputs.len() as i64
    }

    fn tho_device_idx(&self) -> usize {
        self.outputs.first().map(|output| output.as_ref().tho_device_idx()).unwrap_or(0)
    }

    fn mem_size(&self) -> u64 {
        self.outputs.iter().map(|output| output.as_ref().mem_size()).sum()
    }
}

impl WithId for SyncLogitsBatchInput<'_> {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Input for SyncLogitsBatchInput<'_> {}

#[derive(Debug)]
struct SyncLogitsBatchInputWrapper {
    id: SynKey,
    outputs: Vec<Arc<DevicesWrap<ThoTensor<LabelProbability, SynKey>>>>,
    labels: Vec<Vec<f32>>,
    labels_len: usize,
}

impl SyncLogitsBatchInputWrapper {
    fn new(outputs: Vec<Arc<DevicesWrap<ThoTensor<LabelProbability, SynKey>>>>, labels: Vec<Vec<f32>>, labels_len: usize) -> Self {
        Self { id: SynKey::new([mono_key!("logits_batch", "SyncLogitsBatch")]), outputs, labels, labels_len }
    }
}

impl BorrowRef for SyncLogitsBatchInputWrapper {
    type Out<'a> = SyncLogitsBatchInput<'a> where Self: 'a;

    fn borrow_ref<'a>(&'a self, to_device_idx: usize) -> impl std::future::Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        async move {

            let outputs = try_join_all(self.outputs.iter().map(|output| BorrowRef::borrow_ref(output, to_device_idx))).await?;
            Ok(SyncLogitsBatchInput { id: self.id.clone(), outputs, labels: self.labels.clone(), labels_len: self.labels_len })
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct SyncLogitsBatchOutput {
    id: SynKey,
    x: ThoBatch<LabelProbability>,
    y: ThoBatch<LabelsProbabilities>,
}

impl SyncLogitsBatchOutput {
    fn into_parts(self) -> (ThoBatch<LabelProbability>, Option<BatchLengths>, ThoBatch<LabelsProbabilities>) {
        (self.x, None, self.y)
    }
}

impl WorkerStageOutput for SyncLogitsBatchOutput {
    fn encode_from_worker(self, all_cuda_tensors: &AllTensors, tho_device: &ThoDevice) -> Result<Self, ThoError> {
        Ok(SyncLogitsBatchOutput{
            id: self.id,
            x: self.x.encode_from_worker(all_cuda_tensors, tho_device)?,
            y: self.y.encode_from_worker(all_cuda_tensors, tho_device)?,
        })
    }
}

impl WithId for SyncLogitsBatchOutput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for SyncLogitsBatchOutput {}


pub const SYNC_LOGITS_BATCH_ENDPOINT: &str = "SYNC_LOGITS_BATCH_ENDPOINT";
static SYNC_LOGITS_BATCH_INSTANCES: InstanceCache = InstanceCache::new();

#[derive(Debug)]
struct SyncLogitsBatch<C>
where
    C: RunCtx  + 'static
{
    pd: PhantomData<C>,
}

impl<C> SyncLogitsBatch<C>
where
    C: RunCtx  + 'static
{
    async fn new(device: ThoDevice) -> Result<Arc<DynSyncStageEnum<SyncLogitsBatch<C>>>, ThoError> {
        SYNC_LOGITS_BATCH_INSTANCES.get_or_try_insert_recreatable_with::<C, _, _, _, _>(device.tho_device_idx(), GlobalCtx::global().cuda_activity_gate(device.tho_device_idx()), GlobalCtx::global().is_ipc(), 0, move || async move {
                if GlobalCtx::global().is_ipc() {
                Ok(Arc::new(DynSyncStageEnum::Proxy {
                    stage: Arc::new(ProxyStage::proxy_recreatable(SYNC_LOGITS_BATCH_ENDPOINT, Some(0.to_string()), device.tho_device_idx(), Box::new(())).await?),
                    init: (),
                }))
            } else {
                Ok(
                    Arc::new(DynSyncStageEnum::Local(Arc::new(SyncLogitsBatch::new_local().await?)))
                )
            }
        }).await
    }

    async fn new_local()-> Result<Self, ThoError>{
        Ok(Self {
            pd: PhantomData
        })
    }

}

#[typetag::serde]
impl WorkerOutput for SyncLogitsBatchOutput{}

pub fn sync_logits_batch_init<'a>(_stage_id:Option<String>, _init: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, _all_cuda_tensors: &'a AllTensors, _tho_device:&'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let stage:SyncLogitsBatch<WokerRunCtx> = SyncLogitsBatch::new_local().await?;

        Ok(Box::new(WorkerWrapper::new(Arc::new(stage), PhantomData::<JsonIpcCodec>)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}

impl<C> Adjustable for SyncLogitsBatch<C> where C: RunCtx + 'static {}

impl<C> OutId for SyncLogitsBatch<C>
where
    C: RunCtx + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

impl<C> Stage for SyncLogitsBatch<C>
where
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = SyncLogitsBatchInput<'a>;
    type Out = SyncLogitsBatchOutput;
}

impl<C> SyncStage for SyncLogitsBatch<C>
where
    C: RunCtx + 'static,
{
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        debug!("SyncLogitsBatch::do_run: input={input:?}");
        if input.outputs.is_empty() {
            return Err(ThoError::new("SyncLogitsBatch: empty outputs"));
        }
        let x_values = input.outputs.iter()
            .map(|output| tch_result(output.as_ref().local()?.f_flatten(0, 1)))
            .collect::<Result<Vec<_>, ThoError>>()?;

        let x = tch_result(tch_result(Tensor::f_stack(x_values.as_slice(), 0))?.f_squeeze_dim(-1))?;

        debug!("SyncLogitsBatch::do_run: x={x:?}");

        let device = input.outputs[0].as_ref().local()?.device();

        let y_values = input.labels.iter().map(|labels| {
            if labels.len() != input.labels_len {
                return Err(ThoError::new("SyncLogitsBatch: bad label size"));
            }
            Ok(tch_result(tch_result(Tensor::f_from_slice(labels.as_slice()))?.f_to_device(device))?)
        }).collect::<Result<Vec<_>, ThoError>>()?;

        let y = tch_result(Tensor::f_stack(y_values.as_slice(), 0))?;

        trace!("SyncLogitsBatch::do_run: x=\n{x} \ny=\n{y}");

        Ok(SyncLogitsBatchOutput {
            id: input.id.clone(),
            x: ThoTensor::new(thoosa_shared::debug_id!("sync_logits_batch_x"), x)?,
            y: ThoTensor::new(thoosa_shared::debug_id!("sync_logits_batch_y"), y)?,
        })
    }
}

struct LogitsResourcesBatch<L, C>
where
    L: for<'i> Cached<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<LabelProbability, SynKey>>>, Ctx = C> + AsyncStage + OutId + 'static,
    C: RunCtx + 'static,
{
    batch_size: u32,
    train_resources: Arc<Vec<(Resource, Vec<f32>)>>,
    valid_resources: Arc<Vec<(Resource, Vec<f32>)>>,
    loader: Arc<L>,
    labels_len: usize,
}

impl<L, C> Debug for LogitsResourcesBatch<L, C>
where
    L: for<'i> Cached<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<LabelProbability, SynKey>>>, Ctx = C> + AsyncStage + OutId + 'static,
    C: RunCtx + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogitsResourcesBatch").finish()
    }
}

impl<L, C> LogitsResourcesBatch<L, C>
where
    L: for<'i> Cached<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<LabelProbability, SynKey>>>, Ctx = C> + AsyncStage + OutId + 'static,
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    pub fn new(loader: Arc<L>, batch_size: u32, train_resources: Arc<Vec<(Resource, Vec<f32>)>>, valid_resources: Arc<Vec<(Resource, Vec<f32>)>>, labels_len: usize) -> Result<Self, ThoError> {
        if train_resources.is_empty() || valid_resources.is_empty() {
            return Err(ThoError::new("LogitsResourcesBatch::new: empty resources"));
        }
        if batch_size == 0 {
            return Err(ThoError::new("LogitsResourcesBatch::new: batch size is 0"));
        }
        if batch_size as usize > train_resources.len() || batch_size as usize > valid_resources.len() {
            return Err(ThoError::new("LogitsResourcesBatch::new: batch size exceeds resources"));
        }
        Ok(Self {
            batch_size,
            train_resources,
            valid_resources,
            loader,
            labels_len,
        })
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="LogitsResourcesBatch::build_batch_stream"))]
    fn build_batch_stream(loader: Arc<L>, resources: Arc<Vec<(Resource, Vec<f32>)>>, batch_size: u32, labels_len: usize, mut ctx: C) -> BoxStream<'static, Result<(ThoBatch<LabelProbability>, Option<BatchLengths>, ThoBatch<LabelsProbabilities>), ThoError>> {
        let batches = shuffle_and_chunk_with_rng(resources.len(), batch_size, &mut rand::rng());
        let futures = futures::stream::iter(batches.into_iter()).then(move |batch_keys| {
            let loader = Arc::clone(&loader);
            let resources = Arc::clone(&resources);
            let mut ctx_for_batch = ctx.branch(false);

            async move {
                let batch_len = batch_keys.len();
                let futures = batch_keys.into_iter().map(|index| {
                    let loader = Arc::clone(&loader);
                    let (resource, labels) = resources[index].clone();
                    let ctx_for_loader = ctx_for_batch.branch(false);

                    async move {
                        debug!("LogitsResourcesBatch::build_batch_stream: resource={resource:?} labels={labels:?}");
                        let tho_device_idx = ctx_for_loader.tho_device_idx();
                        let label_probability: Arc<DevicesWrap<ThoTensor<LabelProbability, SynKey>>> = track_await!(Arc::clone(&loader).run_cached(Arc::new(resource), ctx_for_loader)).await?;

                        if log_enabled!(DebugLevel) {

                            let val = BorrowRef::borrow_ref(&label_probability, tho_device_idx).await?;
                            debug!("LogitsResourcesBatch::build_batch_stream: out={val:?} \nlabels={labels:?}");
                            if log_enabled!(Trace) && !GlobalCtx::global().is_ipc() {
                                trace!("LogitsResourcesBatch::build_batch_stream: out=\n{} \nlabels={labels:?}", val.as_ref());
                            }

                        }

                        Ok::<_, ThoError>((label_probability, labels))
                    }
                });

                let (label_probabilities, labels) = track_await!(collect_futures_with_progress(
                    futures,
                    batch_len,
                    FuturesOrdering::Ordered,
                    thoosa_shared::CollectionErrorPolicy::Propagate,
                    "LogitsResourcesBatch::build_batch_stream",
                )).await?
                .into_iter()
                .collect::<Vec<_>>()
                .into_iter()
                .fold((vec![], vec![]), |mut a, v| {
                    a.0.push(v.0);
                    a.1.push(v.1);
                    a
                });

                debug!("LogitsResourcesBatch::build_batch_stream: label_probabilities={label_probabilities:?} labels={labels:?}");
                let hash_set = label_probabilities.iter().map(|wrap| wrap.id().clone()).collect::<std::collections::HashSet<_>>();
                let dyn_refs_set = hash_set.iter().map(|id|id as &dyn thoosa_shared::traits::CachableIdTr).collect::<std::collections::HashSet<&dyn thoosa_shared::traits::CachableIdTr>>();
                let required_ids: HashMap<CacheKey, &std::collections::HashSet<&dyn thoosa_shared::traits::CachableIdTr>> = HashMap::from([
                    (loader.output_cache_key(), &dyn_refs_set)
                ]);
                let stage = SyncLogitsBatch::<C>::new(ctx_for_batch.tho_device_idx().into()).await?;
                let input = Arc::new(SyncLogitsBatchInputWrapper::new(label_probabilities, labels, labels_len));
                let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoStageAdjuster, _, ()> = OOMHandler::new(vec![
                    Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(*MaxParallelStep::FAST))),
                    Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("sync_label_batch"))),
                ], None);
                let output = track_await!(oom_handler.run(
                    &stage,
                    input,
                    ctx_for_batch.branch(false),
                    WithRunCtxContext::new(
                       MaxParallelAndDumpToCpuAdjustableContex::new(
                           MaxParallelAdjustContextImpl::new(),
                           DumpToCpuContextImpl::new(&required_ids),
                       ),
               ))).await?;

                debug!("LogitsResourcesBatch::build_batch_stream: SyncLogitsBatch::outputs={output:?}");


                Ok::<_, ThoError>(output.into_parts())
            }
        });

        // TrainNet polls one batch ahead and owns its temporary cache and drain.
        futures.boxed()
    }
}

impl<L, C> AsyncDataBatch<LabelProbability, LabelsProbabilities, C> for LogitsResourcesBatch<L, C>
where
    L: for<'i> Cached<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<LabelProbability, SynKey>>>, Ctx = C> + AsyncStage + OutId + 'static,
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    fn train_batch_stream<'a>(&'a self, ctx: C) -> BoxStream<'a, Result<(ThoBatch<LabelProbability>, Option<BatchLengths>, ThoBatch<LabelsProbabilities>), ThoError>> {
        Self::build_batch_stream(Arc::clone(&self.loader), Arc::clone(&self.train_resources), self.batch_size, self.labels_len, ctx)
    }

    fn valid_batch_stream<'a>(&'a self, ctx: C) -> BoxStream<'a, Result<(ThoBatch<LabelProbability>, Option<BatchLengths>, ThoBatch<LabelsProbabilities>), ThoError>> {
        Self::build_batch_stream(Arc::clone(&self.loader), Arc::clone(&self.valid_resources), self.batch_size, self.labels_len, ctx)
    }
}



#[derive(Debug, Serialize, Deserialize)]
pub struct TrainFinalStepIpcInit {
    params: FinalNetParams,
    training_config: CommonTrainingRunConfig,
    tho_device_idx: usize,
    vs_path: PathBuf,
}

type SyncTrainFinalStep<C> = TrainStep<TrainStepXFn<LabelProbability>, LabelProbability, LabelsProbabilities, C>;


/// Cross-entropy for soft/probabilistic targets.
///
/// For each sample batch and class c:
///   p[b,c] = softmax(logits[b,*])[c]
///
/// Loss:
///   L = -(1 / B) * sum_b sum_c y[b,c] * log(p[b,c])
///
/// `log_softmax` computes log(p) directly and more numerically stably than
/// `softmax(...).log()`.
///
/// Shapes:
///   logits:    [B, C]  raw model logits
///   ys:        [B, C]  target class probabilities, sum_c ys[b,c] = 1
///   log_probs: [B, C]
///
/// `sum_dim_intlist(-1)` sums the per-class cross-entropy for each sample,
/// producing [B], and `mean()` averages it over the batch.
fn train_final_loss(pred_y: &Tensor, target: &Tensor) -> anyhow::Result<Tensor> {
    debug!("train_final_loss: pred_y={pred_y:?} target={target:?}");
    trace!("train_final_loss: pred_y=\n{pred_y} target=\n{target}");

    let log_probs = pred_y.f_log_softmax(-1, Kind::Float)?;

    Ok((target * log_probs)
        .f_sum_dim_intlist(&[-1i64][..], false, Kind::Float)?
        .f_mean(Kind::Float)?
        .f_multiply_scalar(-1.)?)
}

impl WorkerStageInput for TrainFinalStepIpcInit {
}

#[typetag::serde(name = "TrainFinalStepIpcInit")]
impl WorkerInput for TrainFinalStepIpcInit {}


#[derive(Debug, Serialize, Deserialize)]
#[serde(transparent)]
struct TrainFinalStepInput(TrainStepInput<LabelProbability, LabelsProbabilities>);

#[typetag::serde(name = "TrainFinalStepInput")]
impl WorkerInput for TrainFinalStepInput {}

impl std::borrow::Borrow<TrainStepInput<LabelProbability, LabelsProbabilities>> for TrainFinalStepInput {
    fn borrow(&self) -> &TrainStepInput<LabelProbability, LabelsProbabilities> {
        &self.0
    }
}

fn train_final_step_endpoint_init<'a>(_stage_id:Option<String>, init: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, _all_cuda_tensors: &'a AllTensors, _tho_device:&'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let Ok(init) = (init as Box<dyn std::any::Any>).downcast::<TrainFinalStepIpcInit>() else {
            return Err(ThoError::new("TrainFinalStep endpoint received invalid initializer"));
        };
        let init = *init;
        let params = init.params.clone();
        let params_for_module = params.clone();
        let training_config = init.training_config.clone();
        let (train_step, _): (SyncTrainFinalStep<WokerRunCtx>, ()) = TrainStep::new_local(
            init.tho_device_idx.into(),
            move |_run_config, device| async move {
                let config = ModConfig::new(
                    AdamW::default().amsgrad(true).into(),
                    move |var_store| final_net(&params_for_module)?.build_tch_module::<Batch<LabelProbability>, Batch<LabelsProbabilities>>(&var_store.root()),
                    training_config.lr,
                );
                Ok((ThoTrainModule::new_on_device(config, device)?, ()))
            },
            Arc::new(Box::new(|batch| batch) as TrainStepXFn<LabelProbability>),
            train_final_loss,
            init.training_config,
            init.vs_path,
        ).await?;
        // Shared stateful dispatch keeps final-net weights inside the CUDA worker.
        Ok(Box::new(WorkerWrapperWithState::<JsonIpcCodec, _, _, _>::new_with_checkpoint_input::<TrainFinalStepInput>(Arc::new(train_step), PhantomData)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}

#[derive(Debug)]
struct TrainFinalStepFactory {
    execution: thoosa_common::runner::global_ctx::DeviceExecutionCtx,
    num_label_nets: u32,
    labels_len: u32,
}

impl TrainFinalStepFactory {
    fn new(num_label_nets: u32, labels_len: u32, execution: thoosa_common::runner::global_ctx::DeviceExecutionCtx) -> Self {
        Self { num_label_nets, labels_len, execution }
    }
}

#[async_trait]
impl<C> TrainStepFactory<LabelProbability, LabelsProbabilities, C, FinalNetParams, TrainFinalNetRunConfig> for TrainFinalStepFactory
where
    C: RunCtx + 'static,
{
    async fn new_step(&self, run_config: TrainFinalNetRunConfig, ctx: C, vs_path: PathBuf) -> Result<(DynSyncStageWithStateEnum<SyncTrainFinalStep<C>>, FinalNetParams), ThoError> {
        let node_params = FinalNetParams::new(self.num_label_nets, run_config.net.mid_features, self.labels_len, run_config.net.num_layers);
        let device = ctx.tho_device_idx();
        let execution = &self.execution;
        let init = TrainFinalStepIpcInit { params: node_params.clone(), training_config: run_config.common_training_args.clone(), tho_device_idx: device, vs_path: vs_path.clone() };
        let params_for_module = node_params.clone();
        let input_cloner = Arc::new(|input: &thoosa_common::stages::train_step::TrainStepInput<LabelProbability, LabelsProbabilities>| -> Result<Box<dyn WorkerInput>, ThoError> { Ok(Box::new(TrainFinalStepInput(input.clone_refs_for_worker()?))) });
        let stage = execution.install_stateful(
            TRAIN_FINAL_STEP_ENDPOINT, thoosa_common::ipc::endpoints::StageRecovery::OwnerManaged,
            || async move {
                let (train_step, _) = TrainStep::new_local(
                    device.into(),
                    move |run_config, device| async move {
                        let config = ModConfig::new(run_config.common_training_args.tho_optimizer(),
                            move |var_store| final_net(&params_for_module)?.build_tch_module::<Batch<LabelProbability>, Batch<LabelsProbabilities>>(&var_store.root()),
                            run_config.common_training_args.lr);
                        Ok((ThoTrainModule::new_on_device(config, device)?, ()))
                    },
                    Arc::new(Box::new(|batch| batch) as TrainStepXFn<LabelProbability>), train_final_loss, run_config, vs_path,
                ).await?;
                Ok(Arc::new(train_step))
            },
            || async move { Ok((Box::new(init) as Box<dyn WorkerInput>, ())) }, Some(input_cloner),
        ).await?;
        Ok((stage, node_params))
    }
}

pub fn train_final_step_endpoint_inits() -> HashMap<String, MotifsInitFn> {
    HashMap::from([(
        TRAIN_FINAL_STEP_ENDPOINT.to_string(),
        Box::new(train_final_step_endpoint_init) as MotifsInitFn
    )])
}

#[cfg(test)]
mod tests {
    use crate::dims::{ChannelsMotifsData, LabelProbability};
    use crate::motifs_store::{MockMotifsStoreBuilder, MotifsStore, create_motif_tables};
    use crate::stages::apply_motif_nets::motif_sum_net_;
    use crate::stages::cross_loader::ResourcesConfigInput;
    use crate::stages::find_motifs_run_config::FindMotifsRunConfig;
    use crate::stages::tensor_loader::mock_tensor_loader;
    use crate::stages::train_final_net::{TrainFinalNetOutput, TrainFinalNetsRunConfigResult, new_train_final_net};
    use crate::stages::train_label_nets::TrainLabelNetRunConfig;
    use crate::train_nets::{FinalNetArgs, FinalNetArgsFields, FinalNetParams, TrainFinalNetRunConfig, TrainFinalNetRunConfigFields};
    use crate::train_nets::{LabelNetParams, LabelNetTypeParams};
    use log::{debug, LevelFilter};
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use tch::Device::Cpu;
    use tch::nn::{self, VarStore};
    use tch::{Device, Kind, Tensor};
    use thoosa_common::dims::Batch;
    use thoosa_common::nets::batches::BatchLengths;
    use thoosa_common::nets::common_nets::{CommonTrainingRunConfig, CommonTrainingRunConfigFields, LinLenNetParams, OptConfig, ThoModuleWithStore};
    use thoosa_common::nets::nodes::{ThoClosureModule, ThoModule};
    use thoosa_common::resources::{Logits, NetResources, SavableResult};
    use thoosa_common::runner::common::mock_run_ctx;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::tch_error::tch_result;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::utils::reset_print_options;
    use thoosa_common::ws::WsConfigs;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::hyperparameter::Hyperparameter;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::RunConfigId;
    use thoosa_shared::run_configs::{OptRunConfigs, RunConfig};
    use thoosa_shared::traits::AsyncStage;
    use thoosa_shared::traits::RunCtx;
    use thoosa_shared::val::NamedVal;
    use thoosa_shared::val::Val;
    use thoosa_shared::{ThoError, cdt_for_dir, mono_key};
    use thoosa_store::{RunConfigRow, SqlStore, TrainedNet, TrainedNetsBy};

    fn const_label_net(motif_nets: u32, slices_len: u32, k: f64, device: Device) -> (Box<dyn Fn() -> LabelNetParams + Sync + Send + 'static>, Box<dyn Fn() -> ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>> + Sync + Send + 'static>) {
        (
            Box::new(move || LabelNetParams::new(motif_nets, LabelNetTypeParams::LinNetParams(LinLenNetParams::new(slices_len, 1)))),
            Box::new(move || {
                let k = k;
                let builder = Arc::new(move |_path: &nn::Path| {
                    let module = ThoClosureModule::new(Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                        let mut sum = t.f_sum_dim_intlist(&[1, 2][..], false, None)?;
                        // let mut sum = sum.f_unsqueeze(1)?;
                        let out = sum.f_mul_scalar_(k)?;

                        debug!("const_label_net: t=\n{t} out={out:?}");
                        Ok(out)
                    })));
                    Ok(Box::new(module) as Box<dyn ThoModule<ChannelsMotifsData, Batch<LabelProbability>>>)
                });
                let vs = VarStore::new(device);
                let module = builder(&vs.root()).expect("const_label_net builder");
                ThoModuleWithStore::with_factory(SynKey::new([mono_key!(format!("const_label_net_{k}"), "const_label_net")]), module, vs, Some(builder))
            }),
        )
    }

    //all nets sum up their inputs and final net classifies odd/even
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_train_final_net_smoke() -> Result<(), ThoError> {
        log::set_max_level(LevelFilter::Trace);
        reset_print_options();
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let train_motif_nets_run_config_id: RunConfigId = 2;
        let train_label_nets_run_config_id: RunConfigId = 3;
        let motif_width = 4;
        let step = 1;
        let device = Device::Cpu;
        let train_group = "train_group";
        let valid_group = "valid_group";
        let motif_t_0 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_0", "motif_id")]), tch_result(Tensor::f_ones(&[1, motif_width as i64], (Kind::Float, device)))?)?);
        let motif_t_1 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_1", "motif_id")]), tch_result(Tensor::f_ones(&[1, motif_width as i64], (Kind::Float, device)))?)?);

        let (motif_0_params, motif_0_net) = motif_sum_net_(device, Arc::clone(&motif_t_0), step, motif_width);
        let (motif_1_params, motif_1_net) = motif_sum_net_(device, Arc::clone(&motif_t_1), step, motif_width);
        let motif_net_0_path = PathBuf::from("0/0_2");
        let motif_net_0_id = 100;
        let motif_net_1_path = PathBuf::from("0/1_2");
        let motif_net_1_id = 101;

        let label_a_run_config_id = 111;
        let label_b_run_config_id = 222;

        let ws = Arc::new(RwLock::new(
            crate::motifs_ws::MotifsWsMockBuilder::new(None, 1, 100, 1000)
                .with_try_load_motif_net(vec![
                    (motif_net_0_path.clone(), motif_0_params, motif_0_net),
                    (motif_net_1_path.clone(), motif_1_params, motif_1_net),
                ])
                .with_load_label_nets(HashMap::from([
                    ("aaa".to_string(), vec![(label_a_run_config_id, const_label_net(2, 1, 0.1, device).0, const_label_net(2, 1, 0.1, device).1)]),
                    ("bbb".to_string(), vec![(label_b_run_config_id, const_label_net(2, 1, 0.2, device).0, const_label_net(2, 1, 0.2, device).1)]),
                ]))
                // Final-net training reloads each selected label checkpoint after listing the available nets.
                .with_load_label_net(vec![
                    ("aaa".into(), label_a_run_config_id, const_label_net(2, 1, 0.1, device).0, const_label_net(2, 1, 0.1, device).1),
                    ("bbb".into(), label_b_run_config_id, const_label_net(2, 1, 0.2, device).0, const_label_net(2, 1, 0.2, device).1),
                ])
                .build(),
        ));

        let resources_train = (0..10).map(|i| Resource::new(i, 1, format!("resource_train_{i}"), 32)).collect::<Vec<_>>();
        let resources_valid = (10..14).map(|i| Resource::new(i, 1, format!("resource_valid_{i}"), 32)).collect::<Vec<_>>();

        let mut resources_map = HashMap::new();
        for (i, r) in resources_train.iter().chain(resources_valid.iter()).enumerate() {
            if i % 2 == 0 {
                resources_map.insert(r.clone(), tch_result(Tensor::f_zeros(&[1, 32], (Kind::Float, device)))?);
            } else {
                resources_map.insert(r.clone(), tch_result(Tensor::f_ones(&[1, 32], (Kind::Float, device)))?);
            }

        }

        let loader = Arc::new(mock_tensor_loader::build().with_out_id().with_run_cached(resources_map).finish());

        let train_logits = resources_train.iter().enumerate().map(|(i, r)| (r.clone(), if i % 2 == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] })).collect::<Vec<_>>();
        let valid_logits = resources_valid.iter().enumerate().map(|(i, r)| (r.clone(), if i % 2 == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] })).collect::<Vec<_>>();

        let store =
            MockMotifsStoreBuilder::new()
                .with_get_input_run_config_ids([(train_label_nets_run_config_id, Ok(vec![train_motif_nets_run_config_id]))])
                .with_list_trained_nets([
                    (
                        vec![train_motif_nets_run_config_id],
                        Some(0.3),
                        Some(2),
                        Ok(vec![
                            TrainedNet { id: motif_net_0_id, run_config_id: train_motif_nets_run_config_id, net_path: motif_net_0_path.clone(), loss: 0.1f32 },
                            TrainedNet { id: motif_net_1_id, run_config_id: train_motif_nets_run_config_id, net_path: motif_net_1_path.clone(), loss: 0.2f32 },
                        ]),
                    ),
                    (
                        vec![train_motif_nets_run_config_id],
                        None,
                        None,
                        Ok(vec![
                            TrainedNet { id: motif_net_0_id, run_config_id: train_motif_nets_run_config_id, net_path: motif_net_0_path, loss: 0.1f32 },
                            TrainedNet { id: motif_net_1_id, run_config_id: train_motif_nets_run_config_id, net_path: motif_net_1_path, loss: 0.2f32 },
                        ]),
                    ),
                    (
                        vec![label_a_run_config_id],
                        None,
                        None,
                        Ok(vec![
                            TrainedNet { id: label_a_run_config_id*10/*net id != run id*/, run_config_id: label_a_run_config_id, net_path: "aaa".into(), loss: 0.1f32 },
                        ]),
                    ),
                    (
                        vec![label_b_run_config_id],
                        None,
                        None,
                        Ok(vec![
                            TrainedNet { id: label_b_run_config_id*10, run_config_id: label_b_run_config_id, net_path: "bbb".into(), loss: 0.2f32 },
                        ]),
                    )
                ])
                .with_get_motif_meta([
                    (0, Ok((0usize, 0))),
                    (1, Ok((1usize, 0))),
                ])
                .with_run_config_descr([(
                    0,
                    Ok(RunConfigRow {
                        id: 0,
                        r#type: "FindMotifsRunConfig".to_string(),
                        descr: "test find motifs".to_string(),
                        run_on: "2026-01-01 00:00:00".to_string(),
                    }),
                )])
                .with_list_run_config_values([(
                    0,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(0), 10, 100, None, None).to_properties_iter(None).collect()),
                )])
                .with_list_output_run_config_ids([(train_label_nets_run_config_id, Ok(vec![100, 200]))])
                .with_list_resource_random_in_group([(train_group, Ok(train_logits)), (valid_group, Ok(valid_logits))])
                .with_list_input_nets([
                    (label_a_run_config_id*10, Ok(vec![train_motif_nets_run_config_id])),
                    (label_b_run_config_id*10, Ok(vec![train_motif_nets_run_config_id]))
                ])
                .build();

        let run_configs = OptRunConfigs::new(
            thoosa_shared::run_configs::GridRunConfigs::new(
                thoosa_shared::run_configs::ConstConfigs::new(vec![
                    NamedVal::new(TrainFinalNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Epochs), Val::Int(10)),
                    NamedVal::new(TrainFinalNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::BatchSize), Val::Int(4)),
                    NamedVal::new(TrainFinalNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Patience), Val::Int(2)),
                    NamedVal::new(TrainFinalNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::MinDelta), Val::Real(0.02)),
                    NamedVal::new(TrainFinalNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), Val::Real(0.001)),
                    NamedVal::new(TrainFinalNetRunConfigFields::Net(FinalNetArgsFields::NumLayers), Val::Int(10)),
                    NamedVal::new(TrainFinalNetRunConfigFields::Net(FinalNetArgsFields::MidFeatures), Val::Int(20)),
                ]),
                vec![],
            ),
            vec![Hyperparameter::UniformFloat {
                name: TrainFinalNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr),
                lower: 0.0001,
                upper: 0.001,
            }],
        );

        let mut run_ctx = mock_run_ctx::default(1);
        let train_final_net = Arc::new(new_train_final_net(vec![label_a_run_config_id, label_b_run_config_id], loader, run_configs, &ws, &store, run_ctx.branch(false), Arc::new(OptConfig::new(10, 3)), step).await?);

        let input = Arc::new(ResourcesConfigInput::new(NetResources::new(Logits::new(train_group.to_string()), Logits::new(valid_group.to_string()))));

        let r = train_final_net.run(input, run_ctx.branch(false)).await?;
        debug!("test_train_final_net_smoke: result={r:?}");
        assert!(r.result.loss < 0.2);

        Ok(())
    }

    #[tokio::test]
    async fn test_save_train_final_net_result_persists_loss_in_run_config_values() -> Result<(), ThoError> {
        let store: Arc<dyn MotifsStore> = Arc::new(SqlStore::in_mem_(vec![
            thoosa_store::create_tables,
            create_motif_tables
        ]).await?);
        store.bind_run(store.save_run("test".into()).await?)?;
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>> = Arc::new(RwLock::new(crate::motifs_ws::new_motifs_workspace(test_dir.clone(), WsConfigs::new(2, 1000, 32000))?));
        let label_run_config_id_0 = store.save_run_config(None, TrainLabelNetRunConfig::type_name(), "label_0", HashMap::new()).await?;
        let label_run_config_id_1 = store.save_run_config(None, TrainLabelNetRunConfig::type_name(), "label_1", HashMap::new()).await?;
        // Exercise the same checkpoint handoff used by local and IPC training.
        let checkpoint_path = test_dir.join("train_final_checkpoint.safetensors");
        tch_result(VarStore::new(Cpu).save(&checkpoint_path))?;
        let result = TrainFinalNetsRunConfigResult::new(
            Some(&vec![label_run_config_id_0, label_run_config_id_1]),
            0.125,
            thoosa_shared::tmp_file::TmpFile::new(checkpoint_path),
            FinalNetParams::new(2, 100, 2, 3),
            TrainFinalNetRunConfig::new(None, CommonTrainingRunConfig::default(), FinalNetArgs::new(3, 100)),
        );

        let mut output = TrainFinalNetOutput::new(result);
        let saved_ids = SavableResult::save(&mut output, &store, &ws, ()).await?;
        assert_eq!(saved_ids, vec![3]);
        let run_config_id = saved_ids[0];
        let _run_config_values = store.list_run_config_values(run_config_id).await?;
        let trained_nets = store.list_trained_nets(TrainedNetsBy::RunConfigIds(vec![run_config_id]), None, None).await?;

        assert_eq!(vec![TrainedNet { id: 1, run_config_id, net_path: PathBuf::from(format!("3")), loss: 0.125 }], trained_nets);
        let (_, params_path) = ws.read().final_model(run_config_id)?;
        assert_eq!(FinalNetParams::from_file(&params_path)?.label_feature_schema()?, &[label_run_config_id_0, label_run_config_id_1]);

        // Losing the checkpoint must propagate through the public output wrapper as an I/O error.
        output.result.var_store_path = thoosa_shared::tmp_file::TmpFile::new(test_dir.join("missing.safetensors"));
        let error = SavableResult::save(&mut output, &store, &ws, ()).await.unwrap_err();
        assert!(error.chain().any(|cause| cause.downcast_ref::<std::io::Error>().is_some()), "{error:#}");
        Ok(())
    }

}

#[derive(Debug)]
pub struct TrainFinalNetsRunConfigResult {
    train_label_nets_run_config_ids: Vec<RunConfigId>,
    pub loss: f32,
    var_store_path: TmpFile,
    final_node_params: FinalNetParams,
    train_final_net_run_config: TrainFinalNetRunConfig,
}

impl TrainFinalNetsRunConfigResult {
    pub fn new(
        train_label_nets_run_config_ids: Option<&Vec<RunConfigId>>,
        loss: f32,
        var_store_path: TmpFile,
        final_node_params: FinalNetParams,
        train_final_nodes_values: TrainFinalNetRunConfig,
    ) -> Self {
        Self {
            train_label_nets_run_config_ids: train_label_nets_run_config_ids
                .expect("train_label_nodes_run_config_ids")
                .clone(),
            loss,
            var_store_path,
            final_node_params,
            train_final_net_run_config: train_final_nodes_values,
        }
    }
}

#[async_trait]
impl SavableResult for TrainFinalNetsRunConfigResult {
    type Store = dyn MotifsStore;
    type Workspace = dyn crate::motifs_ws::MotifsWorkspace;
    type R = u32;

    async fn save(
        &mut self,
        store: &Arc<dyn MotifsStore>,
        ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
        _p: Self::P<'_>
    ) -> anyhow::Result<u32> {
        // The constructor uses this exact order for feature positions; persist it with the weights.
        self.final_node_params.set_label_feature_schema(self.train_label_nets_run_config_ids.clone())?;

        let run_config_id = track_await!(store.save_run_config(
            Some(&self.train_label_nets_run_config_ids),
            &TrainFinalNetRunConfig::type_name(),
            "Final",
            self.train_final_net_run_config.to_properties_iter(None).collect(),
        )).await?;

        let path = ws.read()
            .save_final_net(run_config_id, (&self.var_store_path).into(), &self.final_node_params)?;

        track_await!(store.save_trained_net(run_config_id, path, self.loss, None)).await?;

        Ok(run_config_id)
    }
}
