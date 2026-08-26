use crate::tasks::apply_label_nets::{ApplyLabelNetInit, ApplyLabelNets, MotifNetsOutput, APPLY_LABEL_NET_ENDPOINT};
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::train_net::{TrainNet, TrainNetInput, TrainStepFactory};
use async_trait::async_trait;
use futures::FutureExt;
use futures_util::StreamExt;
use futures_util::future::try_join_all;
use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use log::{debug, info, trace};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::future::ready;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::atomic::Ordering::Relaxed;
use thoosa_common::dims::{ChannelsData, MotifProbability};
use thoosa_common::stages_seq::ABStage;

use crate::ApplyMotifNets;
use crate::tasks::stft::SyncStftSelector;
use crate::tasks::train_step::{TrainStep, TrainStepXFn};
use tch::Device::{Cpu, Cuda};
use tch::nn::{AdamW, Optimizer, OptimizerConfig, VarStore};
use tch::{Reduction, Tensor};
use thoosa_common::batches::{AsyncDataBatch, BatchLengths, prefetch_next_batch_stream};
use thoosa_common::dims::{Batch, LabelProbability, LabelsProbabilities, ThoBatch};
use thoosa_common::ipc::codec::{IpcCodec, JsonIpcCodec};
use thoosa_common::ipc::cuda_wokrer::{InitFn, WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use thoosa_common::ipc::json_ser::SyncStageIpcWrapper;
use thoosa_common::ipc::worker_msg::{WorkerInput, WorkerOutput};
use thoosa_common::ipc::wraps::{WorkerWrapper, WorkerWrapperWithState};
use thoosa_common::ipc::proxy::{DynSyncStageEnum, DynSyncStageWithStateEnum, ProxyStage};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_shared::traits::MaxParallelCtx;
use thoosa_common::runner::common::cachable::{BorrowRef, Cached, DevicesWrap, SelectRef, ThoDataRef};
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::oom_handler::{NoContexts, NoInputAdjuster, NoRunCtxAdjuster, NoTaskAdjuster, OOMHandler};
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_common::runs::{Logits, NetResources, SavableResult};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::tho_varstore::{ModConfig, ThoTrainModule};
use thoosa_common::train_nets::final_net;
use thoosa_common::train_nets::{CommonTrainingRunConfig, FinalNetParams, OptConfig, TrainFinalNetRunConfig, TrainFinalNetRunConfigFields, TrainFinalNetsRunConfigResult};
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_shared::run_configs::{OptRunConfigs, RunConfigId};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, AsyncStage, Data, DynSyncStageBox, OutId, RunCtx, Stage, SyncStage, WithId};
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_shared::{FuturesOrdering, ThoError, collect_futures_with_progress, mono_key, shuffle_and_chunk_with_rng};
use thoosa_store::Store;

pub(crate) const TRAIN_FINAL_STEP_ENDPOINT: &str = "TrainFinalStep";

#[derive(Debug)]
pub struct TrainFinalNet<L, LL, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    LL: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<LabelProbability, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    train_label_nets_run_config_ids: Vec<RunConfigId>,
    run_configs: Arc<OptRunConfigs<Vec<NamedVal<TrainFinalNetRunConfigFields>>, Vec<NamedRange<TrainFinalNetRunConfigFields>>, Vec<Hyperparameter<TrainFinalNetRunConfigFields>>, TrainFinalNetRunConfigFields>>,
    pd: PhantomData<L>,
    label_loader: Arc<LL>,
    num_label_nets: u32,
    // label_net_run_config_ids: Vec<RunConfigId>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<Box<dyn Store>>,
    opt_config: Arc<OptConfig>,
}

pub async fn new_train_final_net<L, C, R>(
    train_label_nets_run_config_ids: Vec<RunConfigId>,
    loader: Arc<L>,
    run_configs: OptRunConfigs<Vec<NamedVal<TrainFinalNetRunConfigFields>>, Vec<NamedRange<TrainFinalNetRunConfigFields>>, Vec<Hyperparameter<TrainFinalNetRunConfigFields>>, TrainFinalNetRunConfigFields>,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    stage_ctx: R,
    opt_config: Arc<OptConfig>,
    slices_step: u32,
    selected_motif_nets_limit: Option<usize>,
) -> Result<TrainFinalNet<L, ABStage<L, ApplyLabelNets<C, ApplyMotifNets<C>>, C>, C>, ThoError>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    R: RunCtx + MaxParallelCtx + Adjustable + 'static,

{
    //gpt Build a label-net application stage that wraps the raw loader with motif+label inference.
    let apply_label_nets = Arc::new(track_await(ApplyLabelNets::new_with_apply_motif_nets(ws, store, train_label_nets_run_config_ids.clone(), stage_ctx, slices_step, 1)).await?);//TODO config batch_in_chunks

    let label_loader = Arc::new(ABStage::new(
        SynKey::new([mono_key!("Loader_with_label_nets", "Loader_with_label_nets"), mono_key!(format!("{train_label_nets_run_config_ids:?}"), "train_label_nets_run_config_ids")]),
        Arc::clone(&loader),
        Arc::clone(&apply_label_nets),
    ));

    track_await(TrainFinalNet::new(train_label_nets_run_config_ids, label_loader, apply_label_nets.num_nets() as u32, run_configs, ws, store, opt_config)).await
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
        run_configs: OptRunConfigs<Vec<NamedVal<TrainFinalNetRunConfigFields>>, Vec<NamedRange<TrainFinalNetRunConfigFields>>, Vec<Hyperparameter<TrainFinalNetRunConfigFields>>, TrainFinalNetRunConfigFields>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        opt_config: Arc<OptConfig>,
    ) -> Result<Self, ThoError> {
        debug!("TrainFinalNet::new: train_label_nets_run_config_ids={train_label_nets_run_config_ids:?}");
        //gpt Collect all label-net output run ids produced by the input train_label_nets runs.
        // let label_net_run_config_ids = {
        //     let store_guard = store;
        //     let mut run_config_ids = HashSet::new();
        //     for run_config_id in &train_label_nets_run_config_ids {
        //         run_config_ids.extend(track_await(store_guard.list_output_run_config_ids(*run_config_id)).await?);
        //     }
        //     let mut run_config_ids = run_config_ids.into_iter().collect::<Vec<_>>();
        //     run_config_ids.sort();
        //     run_config_ids
        // };
        // if label_net_run_config_ids.is_empty() {
        //     return Err(ThoError::new("TrainFinalNet::new: no label nets found"));
        // }

        Ok(Self {
            train_label_nets_run_config_ids,
            run_configs: Arc::new(run_configs),
            pd: PhantomData,
            label_loader,
            num_label_nets,
            // label_net_run_config_ids,
            ws: Arc::clone(ws),
            store: Arc::clone(store),
            opt_config,
        })
    }
}

#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
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

    type R = Vec<u32>;

    async fn save(&mut self, store: &Arc<Box<dyn Store>>, ws: &Arc<RwLock<Box<dyn Workspace>>>, p: Self::P<'_>) -> Result<Vec<u32>, ThoError> {
        let _ = track_await(self.result.save(store, ws, ())).await;

        Ok(vec![0])//TODO
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
            //gpt Load train/valid logits resources and validate label vector sizes.
            let resources_config = input.resources_config().clone();
            let (train_resources, valid_resources): (Vec<(Resource, Vec<f32>)>, Vec<(Resource, Vec<f32>)>) =
                track_await(resources_config.to_resources(&store)).await?;

            let labels_len = train_resources.first().ok_or_else(|| ThoError::new("TrainFinalNet::run: no train resources"))?.1.len();
            if labels_len == 0 {
                return Err(ThoError::new("TrainFinalNet::run: empty label vector"));
            }
            for (_res, labels) in train_resources.iter().chain(valid_resources.iter()) {
                if labels.len() != labels_len {
                    return Err(ThoError::new("TrainFinalNet::run: inconsistent label vector sizes"));
                }
            }
            let labels_len_u32 = u32::try_from(labels_len).map_err(|error| ThoError::new(format!("TrainFinalNet::run: label vector is too large: {error}")))?;

            let train_resources = Arc::new(train_resources);
            let valid_resources = Arc::new(valid_resources);

            let train_net = Arc::new(TrainNet::new(
                move |_ctx: C, rc: &TrainFinalNetRunConfig| {
                    let label_loader = Arc::clone(&label_loader);
                    let train_resources = Arc::clone(&train_resources);
                    let valid_resources = Arc::clone(&valid_resources);

                    ready(LogitsResourcesBatch::new(label_loader, rc.common_training_args.batch_size, train_resources, valid_resources, labels_len)).boxed()
                },
                Arc::new(TrainFinalStepFactory::new(num_label_nets, labels_len_u32)),
                opt_config,
                train_final_run_configs,
                |rc: &TrainFinalNetRunConfig| rc.common_training_args.epochs,
                1,
                "final".to_string(),
            ));

            let result = track_await(Arc::into_inner(track_await(train_net.run(
                Arc::new(TrainNetInput::new(SynKey::new([mono_key!("", "empty")]))),
                ctx.branch(false))).await?
            ).unwrap()
                .take_data(0)).await?;

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

    fn ref_to_tensor(&self, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, remove: bool) -> Result<Self, ThoError> {
        let outputs = self.outputs.iter().map(|output| Ok(ThoDataRef::Owned(WorkerStageInput::ref_to_tensor(output.as_ref(), all_cuda_tensors, remove)?))).collect::<Result<Vec<_>, ThoError>>()?;
        Ok(Self { id: self.id.clone(), outputs, labels: self.labels.clone(), labels_len: self.labels_len })
    }

    fn decode_for_worker<C: IpcCodec>(input: &[u8], all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError> {
        let input = C::decode::<Self>(input)?;

        Ok(
            SyncLogitsBatchInput{
                id: input.id,
                outputs: input.outputs.into_iter().map(|input_ref|{
                    let ThoDataRef::Owned(input_ref) = input_ref else {
                        panic!("unexpected ref: input_ref={:?}", input_ref);
                    };

                    Ok(ThoDataRef::Owned(ThoTensor::ref_to_tensor(all_cuda_tensors, input_ref)?))
                }).collect::<Result<Vec<_>, ThoError>>()?,
                labels: input.labels,
                labels_len: input.labels_len,
            }
        )
    }
}

#[typetag::serde(name = "SyncLogitsBatchInput")]
impl WorkerInput for SyncLogitsBatchInput<'static> {}

impl Data for SyncLogitsBatchInput<'_> {
    type Metrics = i64;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        self.outputs.len() as i64
    }

    fn device(&self) -> usize {
        self.outputs.first().map(|output| output.as_ref().device()).unwrap_or(0)
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

    fn borrow_to<'a>(&'a self, to_device_idx: usize) -> impl std::future::Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        async move {

            let outputs = try_join_all(self.outputs.iter().map(|output| BorrowRef::borrow_to(output, to_device_idx))).await?;
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
    fn encode_from_worker(self, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Self, ThoError> {
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
    async fn new(device: ThoDevice) -> Result<DynSyncStageEnum<SyncLogitsBatch<C>>, ThoError> {
        if GlobalCtx::global().is_ipc() {
            static STAGE_ID:AtomicU32 = AtomicU32::new(0);//TODO
            let stage_id = STAGE_ID.fetch_add(1, Relaxed);
            Ok(DynSyncStageEnum::Proxy {
                stage: Arc::new(ProxyStage::proxy(SYNC_LOGITS_BATCH_ENDPOINT, stage_id, device.tho_device_idx(), Box::new(())).await?),
                init: (),
            })
        } else {
            Ok(
                DynSyncStageEnum::Local(Arc::new(SyncLogitsBatch::new_local().await?))
            )
        }
    }

    async fn new_local()-> Result<Self, ThoError>{
        Ok(Self {
            pd: PhantomData
        })
    }

}

#[typetag::serde]
impl WorkerOutput for SyncLogitsBatchOutput{}

pub fn sync_logits_batch_init(_init_msg: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn Workspace>>>, _all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> BoxFuture<Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
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
        if input.outputs.is_empty() {
            return Err(ThoError::new("SyncLogitsBatch: empty outputs"));
        }
        let x_values = input.outputs.iter().map(|output| output.as_ref().t_ref().f_flatten(0, 1)).collect::<Result<Vec<_>, _>>()?;
        let x = Tensor::f_stack(x_values.as_slice(), 0)?.f_squeeze()?;
        let device = input.outputs[0].as_ref().t_ref().device();
        let y_values = input.labels.iter().map(|labels| {
            if labels.len() != input.labels_len {
                return Err(ThoError::new("SyncLogitsBatch: bad label size"));
            }
            Ok(Tensor::f_from_slice(labels.as_slice())?.f_to_device(device)?)
        }).collect::<Result<Vec<_>, ThoError>>()?;
        let y = Tensor::f_stack(y_values.as_slice(), 0)?;
        Ok(SyncLogitsBatchOutput {
            id: input.id.clone(),
            x: ThoTensor::new(thoosa_shared::simple_id!("sync_logits_batch_x"), x)?,
            y: ThoTensor::new(thoosa_shared::simple_id!("sync_logits_batch_y"), y)?,
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
    C: RunCtx + MaxParallelCtx + 'static,
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

    fn build_batch_stream(loader: Arc<L>, resources: Arc<Vec<(Resource, Vec<f32>)>>, batch_size: u32, labels_len: usize, mut ctx: C) -> BoxStream<'static, Result<(ThoBatch<LabelProbability>, Option<BatchLengths>, ThoBatch<LabelsProbabilities>), ThoError>> {
        let batches = shuffle_and_chunk_with_rng(resources.len(), batch_size, &mut rand::rng());
        let futures = batches.into_iter().map(move |batch_keys| {
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

                        let out = track_await(Arc::clone(&loader).run_cached(Arc::new(resource), ctx_for_loader)).await?;

                        debug!("LogitsResourcesBatch::build_batch_stream: out={out:?}");

                        Ok::<_, ThoError>((out, labels))
                    }
                });

                let (outputs, labels) = track_await(collect_futures_with_progress(
                    futures,
                    batch_len,
                    FuturesOrdering::Ordered,
                    "LogitsResourcesBatch::build_batch_stream",
                )).await
                .into_iter()
                .collect::<Vec<_>>()
                .into_iter()
                .fold((vec![], vec![]), |mut a, v| {
                    a.0.push(v.0);
                    a.1.push(v.1);
                    a
                });

                let stage = Arc::new(SyncLogitsBatch::<C>::new(1.into()).await?);
                let input = Arc::new(SyncLogitsBatchInputWrapper::new(outputs, labels, labels_len));
                let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoTaskAdjuster, NoRunCtxAdjuster, ()> = OOMHandler::new(vec![], None);
                let output = track_await(oom_handler.run(&stage, input, ctx_for_batch.branch(false), NoContexts)).await?;
                Ok::<_, ThoError>(output.into_parts())
            }
        });

        prefetch_next_batch_stream(futures)
    }
}

impl<L, C> AsyncDataBatch<LabelProbability, LabelsProbabilities, C> for LogitsResourcesBatch<L, C>
where
    L: for<'i> Cached<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<LabelProbability, SynKey>>>, Ctx = C> + AsyncStage + OutId + 'static,
    C: RunCtx + MaxParallelCtx + 'static,
{
    fn train_batch_stream<'a>(&'a self, ctx: C) -> BoxStream<'a, Result<(ThoBatch<LabelProbability>, Option<BatchLengths>, ThoBatch<LabelsProbabilities>), ThoError>> {
        Self::build_batch_stream(Arc::clone(&self.loader), Arc::clone(&self.train_resources), self.batch_size, self.labels_len, ctx)
    }

    fn valid_batch_stream<'a>(&'a self, ctx: C) -> BoxStream<'a, Result<(ThoBatch<LabelProbability>, Option<BatchLengths>, ThoBatch<LabelsProbabilities>), ThoError>> {
        Self::build_batch_stream(Arc::clone(&self.loader), Arc::clone(&self.valid_resources), self.batch_size, self.labels_len, ctx)
    }
}



#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct TrainFinalStepIpcInit {
    params: FinalNetParams,
    training_config: CommonTrainingRunConfig,
    tho_device_idx: usize,
    vs_path: PathBuf,
}

type SyncTrainFinalStep<C> = TrainStep<TrainStepXFn<LabelProbability>, LabelProbability, LabelsProbabilities, C>;

fn train_final_loss(pred_y: &Tensor, target: &Tensor) -> Result<Tensor, ThoError> {
    pred_y.f_binary_cross_entropy_with_logits::<Tensor>(target, None, None, Reduction::Mean).map_err(ThoError::from)
}

impl WorkerStageInput for TrainFinalStepIpcInit {
    fn decode_for_worker<C: IpcCodec>(input: &[u8], _all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError> {
        C::decode(input)
    }
}

#[typetag::serde(name = "TrainFinalStepIpcInit")]
impl WorkerInput for TrainFinalStepIpcInit {}

#[typetag::serde(name = "TrainFinalStepInput")]
impl WorkerInput for crate::tasks::train_step::TrainStepInput<LabelProbability, LabelsProbabilities> {}

fn train_final_step_endpoint_init(init_msg: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn Workspace>>>, _all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>) -> BoxFuture<'_, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let init = *(init_msg as Box<dyn std::any::Any>).downcast::<TrainFinalStepIpcInit>().map_err(|_| ThoError::new("TrainFinalStep endpoint received invalid initializer"))?;
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
        Ok(Box::new(WorkerWrapperWithState::<JsonIpcCodec, _, _, _>::new(Arc::new(train_step), PhantomData)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}

#[derive(Debug)]
struct TrainFinalStepFactory {
    num_label_nets: u32,
    labels_len: u32,
}

impl TrainFinalStepFactory {
    fn new(num_label_nets: u32, labels_len: u32) -> Self {
        Self { num_label_nets, labels_len }
    }
}

#[async_trait]
impl<C> TrainStepFactory<LabelProbability, LabelsProbabilities, C, FinalNetParams, TrainFinalNetRunConfig> for TrainFinalStepFactory
where
    C: RunCtx + 'static,
{
    async fn new(&self, run_config: TrainFinalNetRunConfig, ctx: C, vs_path: PathBuf) -> Result<(DynSyncStageWithStateEnum<SyncTrainFinalStep<C>>, FinalNetParams), ThoError> {
        let node_params = FinalNetParams::new(self.num_label_nets, self.labels_len, run_config.net.num_layers);
        if GlobalCtx::global().is_ipc() {
            static TRAIN_FINAL_STEP_ID: AtomicU32 = AtomicU32::new(0);
            let stage_id = TRAIN_FINAL_STEP_ID.fetch_add(1, Ordering::AcqRel);
            let init = TrainFinalStepIpcInit { params: node_params.clone(), training_config: run_config.common_training_args.clone(), tho_device_idx: ctx.tho_device_idx(), vs_path };
            let input_cloner = Arc::new(|input: &crate::tasks::train_step::TrainStepInput<LabelProbability, LabelsProbabilities>| -> Result<Box<dyn WorkerInput>, ThoError> { Ok(Box::new(input.clone_refs_for_worker()?)) });
            Ok((DynSyncStageWithStateEnum::Proxy { stage: Arc::new(ProxyStage::proxy_with_input_cloner(TRAIN_FINAL_STEP_ENDPOINT, stage_id, ctx.tho_device_idx(), Box::new(init), input_cloner).await?), init: () }, node_params))
        } else {
            let params_for_module = node_params.clone();
            let training_config = run_config.common_training_args.clone();
            let (train_step, _) = TrainStep::new_local(
                ctx.tho_device_idx().into(),
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
                run_config,
                vs_path,
            ).await?;
            Ok((DynSyncStageWithStateEnum::Local(Arc::new(train_step)), node_params))
        }
    }
}

pub fn train_final_step_endpoint_inits() -> HashMap<String, InitFn> {
    HashMap::from([(
        TRAIN_FINAL_STEP_ENDPOINT.to_string(),
        Box::new(train_final_step_endpoint_init) as InitFn
    )])
}

#[cfg(test)]
mod tests {
    use crate::tasks::apply_motif_nets::motif_test_net_;
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::find_motifs_run_config::FindMotifsRunConfig;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use crate::tasks::train_final_net::{TrainFinalNetOutput, new_train_final_net};
    use log::LevelFilter::Debug;
    use log::debug;
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tch::nn::{self, VarStore};
    use tch::{Device, Kind, Tensor};
    use thoosa_common::batches::BatchLengths;
    use thoosa_common::dims::{Batch, ChannelsMotifsData, LabelProbability, Slice};
    use thoosa_common::my_tch::{MyClosureModule, ThoModule};
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::runner::common::mock_run_ctx;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runner::global_ctx::{GlobalCtx, mock_global_ctx};
    use thoosa_common::runs::{Logits, NetResources};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{CommonTrainingRunConfigFields, FinalNetArgsFields, OptConfig, TrainFinalNetRunConfigFields};
    use thoosa_common::train_nets::{ConvNetParams, LabelNetParams, LabelNetTypeParams, LinLenNetParams, ThoModuleWithStore};
    use thoosa_common::utils::reset_print_options;
    use thoosa_common::ws::WsMockBuilder;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::hyperparameter::Hyperparameter;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::OptRunConfigs;
    use thoosa_shared::run_configs::RunConfigId;
    use thoosa_shared::traits::AsyncStage;
    use thoosa_shared::traits::RunCtx;
    use thoosa_shared::val::NamedVal;
    use thoosa_shared::val::Val;
    use thoosa_shared::{ThoError, mono_key};
    use thoosa_store::{MockStoreBuilder, RunConfigRow, TrainedNet};

    fn const_label_net(motif_nets: u32, slices_len: u32, k: f64, device: Device) -> (Box<dyn Fn() -> LabelNetParams + Sync + Send + 'static>, Box<dyn Fn() -> ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>> + Sync + Send + 'static>) {
        (
            Box::new(move || LabelNetParams::new(motif_nets, LabelNetTypeParams::LinNetParams(LinLenNetParams::new(slices_len, 1)))),
            Box::new(move || {
                let k = k;
                let builder = Arc::new(move |_path: &nn::Path| {
                    let module = MyClosureModule::new(Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                        let mut sum = t.f_sum_dim_intlist(&[1, 2][..], false, None)?;
                        // let mut sum = sum.f_unsqueeze(1)?;
                        let out = sum.f_mul_scalar_(k)?;

                        debug!("const_label_net: t={t:?} out={out:?}");
                        Ok(out)
                    })));
                    Ok(Box::new(module) as Box<dyn ThoModule<ChannelsMotifsData, Batch<LabelProbability>>>)
                });
                let mut vs = VarStore::new(device);
                let module = builder(&vs.root()).expect("const_label_net builder");
                ThoModuleWithStore::with_factory(SynKey::new([mono_key!(format!("const_label_net_{k}"), "const_label_net")]), module, vs, Some(builder))
            }),
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_train_final_net_smoke() -> Result<(), ThoError> {
        reset_print_options();
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let train_motif_nets_run_config_id: RunConfigId = 2;
        let train_label_nets_run_config_id: RunConfigId = 3;
        let motif_width = 4;
        let step = 1;
        let device = Device::Cpu;
        let train_group = "train_group";
        let valid_group = "valid_group";
        let motif_t_0 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_0", "motif_id")]), Tensor::f_ones(&[1, motif_width as i64], (Kind::Float, device))?)?);
        let motif_t_1 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_1", "motif_id")]), Tensor::f_ones(&[1, motif_width as i64], (Kind::Float, device))?)?);

        let (motif_0_params, motif_0_net) = motif_test_net_(device, Arc::clone(&motif_t_0), step, motif_width);
        let (motif_1_params, motif_1_net) = motif_test_net_(device, Arc::clone(&motif_t_1), step, motif_width);
        let net_id_0 = PathBuf::from("0/0_2");
        let net_id_1 = PathBuf::from("0/1_2");

        let label_a_run_config_id = 111;
        let label_b_run_config_id = 222;

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 1, 100, 1000)
                .with_load_motif_net(vec![
                    (net_id_0.clone(), motif_0_params, motif_0_net),
                    (net_id_1.clone(), motif_1_params, motif_1_net),
                ])
                .with_load_label_nets(HashMap::from([
                    ("A".to_string(), vec![(label_a_run_config_id, const_label_net(2, 1, 0.1, device).0, const_label_net(2, 1, 0.1, device).1)]),
                    ("B".to_string(), vec![(label_b_run_config_id, const_label_net(2, 1, 0.2, device).0, const_label_net(2, 1, 0.2, device).1)]),
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
                resources_map.insert(r.clone(), Tensor::f_zeros(&[1, 32], (Kind::Float, device))?);
            } else {
                resources_map.insert(r.clone(), Tensor::f_ones(&[1, 32], (Kind::Float, device))?);
            }

        }

        let loader = Arc::new(mock_tensor_loader::build().with_out_id().with_run_cached(resources_map).finish());

        let train_logits = resources_train.iter().enumerate().map(|(i, r)| (r.clone(), if i % 2 == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] })).collect::<Vec<_>>();
        let valid_logits = resources_valid.iter().enumerate().map(|(i, r)| (r.clone(), if i % 2 == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] })).collect::<Vec<_>>();

        let store = Arc::new(
            MockStoreBuilder::new()
                .with_get_input_run_config_ids([(train_label_nets_run_config_id, Ok(vec![train_motif_nets_run_config_id]))])
                .with_list_trained_nets([
                    (
                        vec![train_motif_nets_run_config_id],
                        Some(0.3),
                        Some(2),
                        Ok(vec![
                            TrainedNet { id: 0, run_config_id: train_motif_nets_run_config_id, net_path: net_id_0.clone(), loss: 0.1f32 },
                            TrainedNet { id: 0, run_config_id: train_motif_nets_run_config_id, net_path: net_id_1.clone(), loss: 0.2f32 },
                        ]),
                    ),
                    (
                        vec![train_motif_nets_run_config_id],
                        None,
                        None,
                        Ok(vec![
                            TrainedNet { id: 0, run_config_id: train_motif_nets_run_config_id, net_path: net_id_0, loss: 0.1f32 },
                            TrainedNet { id: 0, run_config_id: train_motif_nets_run_config_id, net_path: net_id_1, loss: 0.2f32 },
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
                .build(),
        );

        let run_configs = OptRunConfigs::new(
            thoosa_shared::run_configs::GridRunConfigs::new(
                thoosa_shared::run_configs::ConstConfigs::new(vec![
                    NamedVal::new(TrainFinalNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Epochs), Val::Int(2)),
                    NamedVal::new(TrainFinalNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::BatchSize), Val::Int(2)),
                    NamedVal::new(TrainFinalNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Patience), Val::Int(2)),
                    NamedVal::new(TrainFinalNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::MinDelta), Val::Real(0.02)),
                    NamedVal::new(TrainFinalNetRunConfigFields::Net(FinalNetArgsFields::NumLayers), Val::Int(1)),
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
        let train_final_net = Arc::new(new_train_final_net(vec![train_label_nets_run_config_id], loader, run_configs, &ws, &store, run_ctx.branch(false), Arc::new(OptConfig::new(10, 3)), step, Some(2)).await?);

        let input = Arc::new(ResourcesConfigInput::new(NetResources::new(Logits::new(train_group.to_string()), Logits::new(valid_group.to_string()))));

        let r = train_final_net.run(input, run_ctx.branch(false)).await?;
        debug!("test_train_final_net_smoke: result={r:?}");
        assert!(r.result.loss < 1.0);//TODO < 0.2

        Ok(())
    }
}
