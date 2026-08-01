use crate::tasks::apply_label_nets::{ApplyLabelNets, MotifNetsOutput};
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::train_net::{TrainNet, TrainNetInput};
use async_trait::async_trait;
use futures::FutureExt;
use futures_util::StreamExt;
use futures_util::future::try_join_all;
use futures_util::stream::BoxStream;
use log::{debug, info, trace};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::future::ready;
use std::marker::PhantomData;
use std::sync::Arc;
use thoosa_common::dims::{ChannelsData, MotifProbability};
use thoosa_common::stages_seq::ABStage;

use crate::ApplyMotifNets;
use crate::tasks::stft::SyncStftSelector;
use crate::tasks::train_step::{TrainStep, TrainStepInput, TrainStepIpcRequest, TrainStepOutput, TrainStepProxyFactory, TrainStepProxyParts, TrainStepWeightsSync, TrainStepXFn};
use tch::Device::{Cpu, Cuda};
use tch::nn::{AdamW, Optimizer, OptimizerConfig, VarStore};
use tch::{Reduction, Tensor};
use thoosa_common::batches::{AsyncDataBatch, BatchLengths, prefetch_next_batch_stream};
use thoosa_common::dims::{Batch, LabelProbability, LabelsProbabilities, ThoBatch};
use thoosa_common::ipc::codec::{IpcCodec, JsonIpcCodec};
use thoosa_common::ipc::cuda_wokrer::InitFn;
use thoosa_common::ipc::json_ser::{SyncStageIpcResponse, SyncStageIpcStreamResponse, SyncStageIpcWrapper};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::common::cachable::{BorrowRef, Cached, DevicesWrap, SelectRef, ThoDataRef};
use thoosa_common::runner::common::runnable::{AsyncStage, DynSyncStageBox, Stage, SyncStage};
use thoosa_common::runner::common::{MaxParallelCtx, OutId, RunCtx};
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::oom_handler::{NoContexts, NoInputAdjuster, NoRunCtxAdjuster, NoTaskAdjuster, OOMHandler};
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_common::runs::{Logits, NetResources, SavableResult};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::final_net;
use thoosa_common::train_nets::{CommonTrainingRunConfig, FinalNetParams, OptConfig, TrainFinalNetRunConfig, TrainFinalNetRunConfigFields, TrainFinalNetsRunConfigResult};
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_proxy_derive::Proxy;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_shared::run_configs::{OptRunConfigs, RunConfigId};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, Data, WithId};
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
    C: RunCtx + Adjustable + 'static,
    R: RunCtx + Adjustable + 'static,

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
    id: SynKey,
    result: TrainFinalNetsRunConfigResult,
}

impl TrainFinalNetOutput {
    pub fn new(id: SynKey, result: TrainFinalNetsRunConfigResult) -> Self {
        Self { id, result }
    }
}

impl WithId for TrainFinalNetOutput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
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
    C: RunCtx + Adjustable + 'static,
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
                move |_:C, rc: &TrainFinalNetRunConfig, vs: Arc<Mutex<VarStore>>| {
                    let net_params = FinalNetParams::new(num_label_nets, labels_len_u32, rc.net.num_layers);
                    ready(Ok((final_net(&net_params).unwrap().build_tch_module::<Batch<LabelProbability>, Batch<LabelsProbabilities>>(&vs.lock().root()).unwrap(), net_params))).boxed()
                },
                opt_config,
                train_final_run_configs,
                |pred: &Tensor, target: &Tensor| {
                    trace!("TrainFinalNet::run, loss_fn: pred.size={:?} target.size={:?}", pred.size(), target.size());
                    Ok(pred.binary_cross_entropy_with_logits::<Tensor>(target, None, None, Reduction::Mean).set_requires_grad(true))
                },
                move |rc: &TrainFinalNetRunConfig, vs: &VarStore| AdamW::default().amsgrad(true).build(&vs, rc.common_training_args.lr).map_err(|e| e.into()),
                |t: ThoBatch<LabelProbability>| {
                    debug!("TrainFinalNet::run, x_f: t=\n{t}");

                    t
                },
                |rc: &TrainFinalNetRunConfig| rc.common_training_args.epochs,
                1,
                "final".to_string(),
            )
                // In IPC mode the final classifier's trainable state must remain in the CUDA worker.
                .with_train_step_proxy_factory(Arc::new(TrainFinalNetProxyFactory::new(num_label_nets, labels_len_u32))));

            let result = track_await(Arc::into_inner(track_await(train_net.run(
                Arc::new(TrainNetInput::new(SynKey::new([mono_key!("", "empty")]))),
                ctx.branch(false))).await?
            ).unwrap()
                .take_data(0)).await?;

            let final_result = TrainFinalNetsRunConfigResult::new(Some(&train_label_nets_run_config_ids), result.min_loss, result.var_store, result.node_params, result.run_config);
            info!("TrainFinalNet::run, done: min_loss={:?}", final_result.loss);

            Ok(Arc::new(TrainFinalNetOutput::new(SynKey::new([mono_key!("", "empty")]), final_result)))
        }
    }
}

#[derive(Debug)]
struct SyncLogitsBatchInput<'a> {
    id: SynKey,
    outputs: Vec<ThoDataRef<'a, ThoTensor<LabelProbability, SynKey>>>,
    labels: Vec<Vec<f32>>,
    labels_len: usize,
}

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

impl WithId for SyncLogitsBatchInputWrapper {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
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

#[derive(Debug)]
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

impl WithId for SyncLogitsBatchOutput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for SyncLogitsBatchOutput {}

#[derive(Debug)]
struct SyncLogitsBatch<C> {
    pd: PhantomData<C>,
}

impl<C> SyncLogitsBatch<C> {
    fn new() -> Self {
        Self { pd: PhantomData }
    }
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
    C: RunCtx + 'static,
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

                let stage = Arc::new(SyncLogitsBatch::<C>::new());
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
    C: RunCtx + 'static,
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
}

type SyncTrainFinalStep = TrainStep<TrainStepXFn<LabelProbability>, LabelProbability, LabelsProbabilities, RunCtxImpl>;

fn train_final_loss(pred_y: &Tensor, target: &Tensor) -> Result<Tensor, ThoError> {
    pred_y.f_binary_cross_entropy_with_logits::<Tensor>(target, None, None, Reduction::Mean).map_err(ThoError::from)
}

const TRAIN_FINAL_OPTIMIZER_FN: fn(&CommonTrainingRunConfig, &VarStore) -> Result<Optimizer, ThoError> = |training_config, var_store| {
    AdamW::default().amsgrad(true).build(var_store, training_config.lr).map_err(ThoError::from)
};

fn new_train_final_step_for_ipc(init: &TrainFinalStepIpcInit) -> Result<SyncTrainFinalStep, ThoError> {
    let params = init.params.clone();
    let x_f: Arc<TrainStepXFn<LabelProbability>> = Arc::new(Box::new(|batch| batch));
    let (train_step, ()) = futures::executor::block_on(TrainStep::new_local(
        Cuda(0).into(),
        move |var_store| async move {
            let net = final_net(&params)?.build_tch_module::<Batch<LabelProbability>, Batch<LabelsProbabilities>>(&var_store.lock().root())?;
            Ok((net, ()))
        },
        TRAIN_FINAL_OPTIMIZER_FN,
        x_f,
        train_final_loss,
        &init.training_config,
    ))?;
    Ok(train_step)
}

// The worker initializes the final classifier only on first use, after its CUDA context is active.
#[derive(Debug)]
struct TrainFinalStepIpcInitStage {
    init: TrainFinalStepIpcInit,
    stage: Mutex<Option<Arc<SyncTrainFinalStep>>>,
}

impl TrainFinalStepIpcInitStage {
    fn new(init: TrainFinalStepIpcInit) -> Self {
        Self { init, stage: Mutex::new(None) }
    }

    fn stage(&self) -> Result<Arc<SyncTrainFinalStep>, ThoError> {
        let mut stage = self.stage.lock();
        if stage.is_none() {
            stage.replace(Arc::new(new_train_final_step_for_ipc(&self.init)?));
        }
        Ok(Arc::clone(stage.as_ref().ok_or_else(|| ThoError::new("TrainFinalStepIpcInitStage failed to initialize stage"))?))
    }
}

#[derive(Debug)]
struct TrainFinalStepIpcWrapper {
    stage: Arc<TrainFinalStepIpcInitStage>,
}

impl SyncStageIpcWrapper for TrainFinalStepIpcWrapper {
    fn do_run(&self, input: &[u8], all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError> {
        match JsonIpcCodec::decode::<TrainStepIpcRequest>(input)? {
            TrainStepIpcRequest::Run { x, lengths, expected_y, train, max_other_parallel } => {
                let stage = self.stage.stage()?;
                let remove_x = x.remove_after_run.then_some(x.ref_key);
                let remove_expected_y = expected_y.remove_after_run.then_some(expected_y.ref_key);
                let x = ThoTensor::ref_to_tensor(all_cuda_tensors, x.into_tensor::<LabelProbability>()?)?;
                let expected_y = ThoTensor::ref_to_tensor(all_cuda_tensors, expected_y.into_tensor::<LabelsProbabilities>()?)?;
                let input = TrainStepInput::new(x, lengths.map(BatchLengths::new), expected_y, train);
                let mut ctx = RunCtxImpl::new(tho_device.tho_device_idx(), TRAIN_FINAL_STEP_ENDPOINT);
                ctx.set_max_other_parallel(max_other_parallel);
                let result = stage.do_run(&input, &ctx);
                drop(input);
                // Request-owned copies can be released once the synchronous forward/backward call returns.
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
            TrainStepIpcRequest::ExportWeights => Err(ThoError::new("TrainFinalStep ExportWeights requires binary endpoint invocation")),
        }
    }

    fn do_run_response(&self, input: &[u8], all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<SyncStageIpcResponse, ThoError> {
        // if matches!(JsonIpcCodec::decode::<TrainStepIpcRequest>(input)?, TrainStepIpcRequest::ExportWeights) {
        //     let stage = self.stage.stage()?;
        //     Ok(SyncStageIpcResponse::Stream(SyncStageIpcStreamResponse::new(move |output| stage.save_var_store_to_stream(output))))
        // } else {
            self.do_run(input, all_cuda_tensors, tho_device).map(SyncStageIpcResponse::Encoded)
        // }
    }
}

#[derive(Debug, Proxy)]
#[proxy(
    common = "thoosa_common",
    name = "ProxyTrainFinalStep",
    endpoint = "TRAIN_FINAL_STEP_ENDPOINT",
    new_args = "init: TrainFinalStepIpcInit, tho_device_idx: usize",
    init = "init",
    fields = "tho_device_idx: usize = tho_device_idx",
    out_id = "()",
    run_input = "thoosa_shared::tracked_await::track_await(TrainStepIpcRequest::run(input, ctx)).await?",
    endpoint_init = "train_final_step_endpoint_init",
    init_type = "TrainFinalStepIpcInit",
    init_stage = "TrainFinalStepIpcInitStage::new(init)",
    wrapper = "TrainFinalStepIpcWrapper"
)]
struct TrainFinalStepProxyTarget {
    base: TrainStep<TrainStepXFn<LabelProbability>, LabelProbability, LabelsProbabilities, RunCtxImpl>,
}

impl ProxyTrainFinalStep {
    async fn export_weights(&self) -> Result<std::fs::File, ThoError> {
        thoosa_common::ipc::codec::invoke_ipc_endpoint_to_file::<JsonIpcCodec, _>(
            GlobalCtx::global().cuda_actor(self.tho_device_idx).as_ref(),
            TRAIN_FINAL_STEP_ENDPOINT,
            &TrainStepIpcRequest::ExportWeights,
        ).await
    }
}

// This adapter keeps TrainNet generic over its original context while forwarding through RunCtxImpl.
#[derive(Debug)]
struct TrainFinalStepProxyCtx<C>
where
    C: RunCtx + 'static,
{
    proxy: Arc<ProxyTrainFinalStep>,
    pd: PhantomData<C>,
}

impl<C> Stage for TrainFinalStepProxyCtx<C>
where
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = TrainStepInput<LabelProbability, LabelsProbabilities>;
    type Out = TrainStepOutput;
}

impl<C> SyncStage for TrainFinalStepProxyCtx<C>
where
    C: RunCtx + 'static,
{
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let mut proxy_ctx = RunCtxImpl::new(ctx.tho_device_idx(), TRAIN_FINAL_STEP_ENDPOINT);
        proxy_ctx.set_max_other_parallel(ctx.max_other_parallel());
        Arc::clone(&self.proxy).do_run(input, &proxy_ctx)
    }
}

impl<C> OutId for TrainFinalStepProxyCtx<C>
where
    C: RunCtx + 'static,
{
    fn out_id<'a>(&self, _input_id: &SynKey) -> () {
        ()
    }
}

impl<C> Adjustable for TrainFinalStepProxyCtx<C> where C: RunCtx + 'static {}

#[derive(Debug)]
struct TrainFinalStepWeightsSync {
    proxy: Arc<ProxyTrainFinalStep>,
}

#[async_trait]
impl TrainStepWeightsSync for TrainFinalStepWeightsSync {
    async fn into_var_store(&self) -> Result<VarStore, ThoError> {
        let weights = track_await(self.proxy.export_weights()).await?;
        let named_tensors = Tensor::load_multi_from_stream_with_device(weights, Cpu)?;
        let var_store = VarStore::new(Cpu);
        var_store.variables_.lock().map_err(|_| ThoError::new("TrainFinalStepWeightsSync VarStore lock poisoned"))?.named_variables.extend(named_tensors);
        Ok(var_store)
    }
}

#[derive(Debug)]
struct TrainFinalNetProxyFactory {
    num_label_nets: u32,
    labels_len: u32,
}

impl TrainFinalNetProxyFactory {
    fn new(num_label_nets: u32, labels_len: u32) -> Self {
        Self { num_label_nets, labels_len }
    }
}

#[async_trait]
impl<C> TrainStepProxyFactory<LabelProbability, LabelsProbabilities, C, FinalNetParams, TrainFinalNetRunConfig> for TrainFinalNetProxyFactory
where
    C: RunCtx + 'static,
{
    async fn new_proxy(&self, run_config: &TrainFinalNetRunConfig, tho_device_idx: usize) -> Result<TrainStepProxyParts<LabelProbability, LabelsProbabilities, C, FinalNetParams>, ThoError> {
        let node_params = FinalNetParams::new(self.num_label_nets, self.labels_len, run_config.net.num_layers);
        let init = TrainFinalStepIpcInit { params: node_params.clone(), training_config: run_config.common_training_args.clone() };
        let proxy = Arc::new(ProxyTrainFinalStep::new(init, tho_device_idx).await?);
        let stage = Arc::new(DynSyncStageBox::new(Arc::new(TrainFinalStepProxyCtx::<C> { proxy: Arc::clone(&proxy), pd: PhantomData })));
        Ok(TrainStepProxyParts::new(stage, Arc::new(TrainFinalStepWeightsSync { proxy }), node_params))
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
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::common::{RunCtx, mock_run_ctx};
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
