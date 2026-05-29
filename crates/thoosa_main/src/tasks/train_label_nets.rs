use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::future::ready;
use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use futures::Stream;
use futures_util::StreamExt;
use futures_util::future::BoxFuture;
use futures::FutureExt;
use futures_util::stream::BoxStream;
use log::{debug, error, info, trace, warn};
use parking_lot::{Mutex, RwLock};
use polars_arrow::io::iterator::StreamingIterator;
use tch::nn::{AdamW, OptimizerConfig, VarStore};
use tch::{Device, Reduction, Tensor};
use cached_derive::Cached;
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap, DevicesWrapCopyMode};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::tho_tensor::{ThoTensor};
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::{build_resource_maps, Resource, ResourcesConfig};
use thoosa_shared::traits::{Adjustable, Data, WithId};
use thoosa_common::dims::ChannelsData;
use thoosa_common::runner::common::{OutId, RunCtx};

use mockall_double::double;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_shared::run_configs::RunConfig;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::batches::{prefetch_next_batch_stream, preload_one, AsyncDataBatch, BatchLengths};
use thoosa_common::cuda_utils::cuda_mem;
use thoosa_common::dims::{Batch, Dims, LabelProbability, MotifProbability, ThoBatch};
use thoosa_common::runs::{Labeled, NetResources, SavableResult};
use thoosa_common::train_nets::{conv_label_node, lin_label_node, opt_train, self_attention_label_node, CommonTrainingRunConfig, ConvNetArgs, ConvNetParams, LabelNetParams, LabelNetTypeArgs, LabelNetTypeParams, LinLenNetParams, LinNetArgs, OptConfig, SelfAttentionNetArgs, SelfAttentionNetParams, TrainLabelNetRunConfig, TrainLabelNetRunConfigFields, TrainLabelNetRunConfigResult, TrainLabelNetRunConfigResultParams};
use thoosa_common::utils::{mix_x_binary_with_labels, pad_stack_with_mask, ListenerResponse, TrainListener};
use thoosa_common::ws::Workspace;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::{collect_futures_with_progress, mono_key, shuffle_and_chunk_with_rng, CopyToDevice, StreamBuffering, ThoError};
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::run_configs::{OptRunConfigs, RunConfigId};
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_store::Store;
use thoosa_with_data::with_data;
use crate::tasks::cross_loader::ResourcesConfigInput;
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use rand::rng;
use tokio::{join, try_join};
use thoosa_common::runner::adjusters::max_parallel::MAX_PARALLEL;
use thoosa_dummy_data_derive::DummyData;
use crate::tasks::apply_motif_nets::ApplyMotifNets;
use crate::tasks::stages_seq::ABStage;
use crate::tasks::train_net::{TrainNet, TrainNetInput, TrainNetOutput};
use thoosa_shared::comp_id::SynKey;
use crate::tasks::apply_motif_nets_merged::ApplyMotifNetsMerged;
use crate::tasks::stage_ids::RAW_TENSOR_LOADER;

#[derive(Debug)]
pub struct TrainLabelNets<L, LM, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C> + Cached + OutId + 'static,
    LM: for <'i> AsyncStage<In<'i>=Resource, Out=DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>, Ctx=C> + Cached + OutId + 'static,
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

}

pub async fn new_train_label_nets<L, C, R>(
    train_motif_net_run_config_ids: Vec<RunConfigId>,
    loader: Arc<L>,
    run_configs: OptRunConfigs<Vec<NamedVal<TrainLabelNetRunConfigFields>>, Vec<NamedRange<TrainLabelNetRunConfigFields>>, Vec<Hyperparameter<TrainLabelNetRunConfigFields>>, TrainLabelNetRunConfigFields>,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    stage_ctx:R,
    opt_config:Arc<OptConfig>,
    labels: Option<Vec<String>>,
    max_loss: Option<f32>,
    selected_motif_nets_limit: Option<usize>,
    slices_step: u32
) -> Result<TrainLabelNets<L, ABStage<L, ApplyMotifNetsMerged<C>, C>, C>, ThoError>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
    R:RunCtx + Adjustable + 'static
{
    let id = mono_key!(RAW_TENSOR_LOADER, "stage") + &SynKey::new(train_motif_net_run_config_ids.iter().map(|run_config_id| mono_key!(*run_config_id, format!("train_motif_nets_run_config_ids_{run_config_id}"))).collect::<Vec<_>>());

    //gpt Build a motif-net application stage to turn raw resources into motif probabilities.
    // let apply_motif_nets = Arc::new(ApplyMotifNets::new(
    let apply_motif_nets = Arc::new(ApplyMotifNetsMerged::new(
        ws,
        store,
        train_motif_net_run_config_ids.clone(),
        stage_ctx,
        max_loss, //max_loss and selected_motif_nets_limit are used to select motif_nets_ids which will be saved as label net input
        selected_motif_nets_limit,
        slices_step,
        // #[cfg(test)]
        None,
        false
    ).await?);

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

    TrainLabelNets::new(
        train_motif_net_run_config_ids,
        motif_nets_ids,
        applied_motif_loader,
        num_nets,
        run_configs,
        ws,
        store,
        opt_config,
        labels,
    ).await
}

impl <L, LM, C> TrainLabelNets<L, LM, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C> + Cached + OutId + 'static,
    LM: for <'i> AsyncStage<In<'i>=Resource, Out=DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>, Ctx=C> + Cached + OutId + 'static,
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

        })
    }

}
//TODO remove
#[derive(Debug)]
pub struct TrainLabelNetsInput {
    id: SynKey
}

impl TrainLabelNetsInput {
    pub fn new(id: SynKey) -> Self {
        Self { id }
    }
}

#[async_trait]
impl CopyToDevice for TrainLabelNetsInput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> where Self:Sized {
        todo!()
    }
}

impl Data for TrainLabelNetsInput {
    type Metrics = ();
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        ()
    }

    fn device(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        0
    }
}

impl Input for TrainLabelNetsInput {

}

impl WithId for TrainLabelNetsInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}


#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct TrainLabelNetsOutput {
    id: RunConfigId,
    train_motif_nets_run_config_ids: Vec<RunConfigId>,
    motif_nets_ids:Vec<u32>,
    results: Vec<TrainLabelNetRunConfigResult>
}
impl WithId for TrainLabelNetsOutput {
    type ID = RunConfigId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl WithId for Box<TrainLabelNetsOutput> {
    type ID = RunConfigId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
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
                result.save(store, ws, TrainLabelNetRunConfigResultParams {
                    train_motif_nets_run_config_ids:&self.train_motif_nets_run_config_ids,
                    motif_nets_ids:&self.motif_nets_ids
                }).await?
            );
        }

        Ok(run_config_ids)
    }
}

impl TrainLabelNetsOutput {
    pub fn new(train_motif_nets_run_config_ids: Vec<RunConfigId>, motif_nets_ids:Vec<u32>, results: Vec<TrainLabelNetRunConfigResult>) -> Self {
        Self {
            id: 0,
            train_motif_nets_run_config_ids,
            motif_nets_ids,
            results
        }
    }
}

impl <L, LM, C> Stage for TrainLabelNets<L, LM, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C> + Cached + OutId + 'static,
    LM: for <'i> AsyncStage<In<'i>=Resource, Out=DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    type Ctx = C;
    type In<'a> = ResourcesConfigInput<NetResources<Labeled>>;
    type Out = TrainLabelNetsOutput;
}

impl<L, LM, C> AsyncStage for TrainLabelNets<L, LM, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData, SynKey>, Ctx = C> + Cached + OutId + 'static,
    LM: for <'i> AsyncStage<In<'i>=Resource, Out=DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainLabelNets::run"))]
    fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> impl Future<Output=Result<Arc<DevicesWrap<Self::Out>>, ThoError>> + Send {
        let train_label_nets_run_configs = Arc::clone(&self.run_configs);


        debug!("TrainLabelNets::run: num_nets={}", self.num_nets);

        async move {
            //Split resources into train/valid sets per label for binary label-vs-rest training.
            let resources_config = with_data!(ctx.branch(false), input; |_ctx, input:&ResourcesConfigInput<NetResources<Labeled>>|{
                Ok::<_, ThoError>(input.resources_config().clone())
            })?;
            let (train_labels_resource_keys, valid_label_resource_keys) = resources_config.to_resources(&self.store).await?;

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
            let futures = labels_train_and_valid_sets.iter()
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
                async move {
                    debug!("TrainLabelNets::run: label={label} resources={resources_1:?}");

                    let resources_0 = labels_others_train_valid_resources.get(label).expect(&format!("no resources not belonging to {label}"));
                    info!("TrainLabelNets::run: label={label} resources_x={resources_0:?}");

                    let resources_1_train = Arc::new(resources_1.0.clone());
                    let resources_1_valid = Arc::new(resources_1.1.clone());
                    let resources_0_train = Arc::new(resources_0.0.clone());
                    let resources_0_valid = Arc::new(resources_0.1.clone());
                    let label = label.clone();
                    let label_for_batch = label.clone();


                    let train_net = Arc::new(TrainNet::new(
                        move |_ctx_for_batch:C,rc:&TrainLabelNetRunConfig| {

                            let applied_motif_loader = Arc::clone(&applied_motif_loader);
                            let label = label_for_batch.clone();
                            ready(Ok(ResourcesBatch::new(
                                   Arc::clone(&applied_motif_loader),
                                   rc.common_training_args.batch_size,
                                   Arc::clone(&resources_1_train),
                                   Arc::clone(&resources_1_valid),
                                   Arc::clone(&resources_0_train),
                                   Arc::clone(&resources_0_valid),
                                   Arc::new(move |x_1_wrapped: Vec<Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>>>>, x_0_wrapped: Vec<Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>>>>, mut ctx: C| {
                                       let label = label.clone();
                                       async move {
                                           let (x, lengths, y) = with_data!(ctx.branch(false),[x_1_wrapped], [x_0_wrapped];|_ctx, x_1:&[&DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>], x_0:&[&DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>]|{

                                               if x_1.is_empty() || x_1.is_empty(){
                                                 error!("TrainLabelNets/ResourcesBatch::train_batch_iter: label={label} x_1={x_1:?} x_0={x_0:?}");
                                               } else {
                                                 debug!("TrainLabelNets/ResourcesBatch::train_batch_iter: label={label} x_1={x_1:?} x_0={x_0:?}");
                                               }

                                               let max_input = x_1.iter()
                                                   .chain(x_0.iter())
                                                   .map(|x|x.data().t_ref().size()[1])
                                                   .max().ok_or_else(||ThoError::new(format!("batch resources are empty for {label}")))?;

                                               debug!("TrainLabelNets/ResourcesBatch::train_batch_iter: label={label} max_input={max_input}");

                                               let x_1_refs = x_1.iter().map(|x| x.data()).collect::<Vec<_>>();
                                               let x_0_refs = x_0.iter().map(|x| x.data()).collect::<Vec<_>>();

                                               // TODO(perf): pad_stack reallocates/copies variable-length motif outputs for every label batch; group by output length or reuse padded buffers across batches.
                                               let (x_1_batch, lengths_1) = pad_stack_with_mask(x_1_refs.as_slice(), num_nets as i64, max_input)?;
                                               let (x_0_batch, lengths_0) = pad_stack_with_mask(x_0_refs.as_slice(), num_nets as i64, max_input)?;
                                               let lengths_1 = BatchLengths::new(lengths_1);
                                               let lengths_0 = BatchLengths::new(lengths_0);

                                               debug!("TrainLabelNets/ResourcesBatch::train_batch_iter: x_1.size={:?}, x_0.size={:?}, lengths_1={:?}, lengths_0={:?}", x_1_batch.size(), x_0_batch.size(), lengths_1, lengths_0);

                                               mix_x_binary_with_labels(&x_1_batch, Some(&lengths_1), &x_0_batch, Some(&lengths_0))
                                           })?;

                                           debug!("TrainLabelNets/ResourcesBatch::train_batch_iter: x.size={:?} y.size={:?} lengths={lengths:?}", x.size(), y.size());


                                           Ok((x.into(), lengths, y.into()))
                                       }.boxed()
                                   })
                               ))).boxed()


                       },
                        move |_:C, rc:&TrainLabelNetRunConfig, vs: Arc<Mutex<VarStore>>| {
                            async move {
                                match rc.net {
                                    LabelNetTypeArgs::ConvNet(ConvNetArgs { num_conv_layers, hidden_conv_channels, num_head_layers, num_head_features, kernel, ref joint }) => {
                                        //TODO impl From<(num_nets, LabelNetTypeArgs)> for LabelNetParams
                                        let net_params = LabelNetParams::new(num_nets, LabelNetTypeParams::ConvNetParams(ConvNetParams::new(num_conv_layers, hidden_conv_channels, kernel, num_head_layers, num_head_features, joint.into())));

                                        Ok((conv_label_node(&net_params)?.build_tch_module::<Batch<MotifProbability>, Batch<LabelProbability>>(&vs.lock().root())?, net_params))
                                    }
                                    LabelNetTypeArgs::LinNet(LinNetArgs { num_layers, features }) => {
                                        let net_params = LabelNetParams::new(num_nets, LabelNetTypeParams::LinNetParams(LinLenNetParams::new(features, num_layers)));

                                        Ok((lin_label_node(&net_params)?.build_tch_module::<Batch<MotifProbability>, Batch<LabelProbability>>(&vs.lock().root())?, net_params))
                                    }
                                    LabelNetTypeArgs::SelfAttentionNet(SelfAttentionNetArgs { head_size, n_heads, ff_mult, dropout }) => {
                                        let net_params = LabelNetParams::new(num_nets, LabelNetTypeParams::SelfAttentionNetParams(SelfAttentionNetParams::new(head_size, n_heads, ff_mult, dropout)));
                                        Ok((self_attention_label_node(&net_params)?.build_tch_module::<Batch<MotifProbability>, Batch<LabelProbability>>(&vs.lock().root())?, net_params))
                                    }
                                }
                            }.boxed()

                        },
                        opt_config,
                        train_label_nets_run_configs,
                        |pred: &Tensor, target: &Tensor| {
                            debug!("TrainLabelNets::run, loss_fn: pred.size={:?}", pred.size());
                            let pred_y = pred;//.squeeze();
                            debug!("TrainLabelNets::run, loss_fn: pred_y={:?} target={:?} kind={:?}",pred_y.size(),target.size(),target.kind());
                            trace!("TrainLabelNets::run, loss_fn: pred_y=\n{pred_y} target=\n{target} kind={:?}",target.kind());
                            // Ok(pred_y.binary_cross_entropy::<Tensor>(&target, None, Reduction::Mean).set_requires_grad(true)) - with sigmoid in any_len_label_node
                            Ok(pred_y.binary_cross_entropy_with_logits::<Tensor>(&target, None, None, Reduction::Mean).set_requires_grad(true))
                        },
                        move |rc:&TrainLabelNetRunConfig, vs: &VarStore|{
                            rc.common_training_args.build_optimizer(vs)
                        },
                        |t:ThoBatch<MotifProbability>| {
                            debug!("TrainLabelNets::run, x_f: t.size={:?} cuda_mem={:?}",t.size(),cuda_mem());
                            t
                        },
                        |rc:&TrainLabelNetRunConfig|{
                            rc.common_training_args.epochs
                        },
                        1,//TODO params
                        format!("label_{label}")
                    ));

                    let result = train_net.run(
                        Arc::new(DevicesWrap::wrap(TrainNetInput::new(SynKey::new([mono_key!("", "empty")])))),
                        ctx_for_task
                    ).await?;


                    let result = Arc::into_inner(result).expect("can't get result from train_net").take_data(0).await?;
                    debug!("TrainLabelNets::run, done: label={label} min_loss={:?}", result.min_loss);

                    Ok::<_, ThoError>(
                        TrainLabelNetRunConfigResult::new(
                            label,
                            result.min_loss,
                            result.node_params,
                            result.run_config,
                            result.var_store,
                        )
                    )

                }

            });

            let results = collect_futures_with_progress(
                futures,
                labels_train_and_valid_sets_len,
                crate::buffer_limits::TRAIN_LABEL_NETS_RUN_BUFFER_UNORDERED_LIMIT,
                StreamBuffering::Unordered,
                "train labels",
            ).await;

            Ok(Arc::new(DevicesWrap::wrap(TrainLabelNetsOutput::new(self.train_motif_nets_run_config_ids.clone(), self.motif_nets_ids.clone(), results))))
        }
    }
}

#[allow(type_alias_bounds)]
type BuildBatchFn<LD:Data + WithId + Send + Sync + 'static, X:Dims+ 'static, Y:Dims+ 'static, C> = Arc<dyn Fn(
    Vec<Arc<DevicesWrap<LD>>>,
    Vec<Arc<DevicesWrap<LD>>>, C

) -> BoxFuture<'static, Result<(ThoBatch<X>, Option<BatchLengths>, ThoBatch<Y>), ThoError>> + Send + Sync>;

// #[derive(Debug)]
struct ResourcesBatch<L, C, LD, X, Y>
where
    L: for<'i> Cached<In<'i> = Resource, Out = LD, Ctx=C> + OutId + 'static,
    C:RunCtx+'static,
    LD:Data + WithId + Send + Sync + 'static,
    <LD as Data>::M: DevicesWrapCopyMode<LD>,
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
    L: for<'i> Cached<In<'i> = Resource, Out = LD, Ctx=C> + OutId + 'static,
    C:RunCtx+'static,
    LD:Data + WithId + Send + Sync + 'static,
    <LD as Data>::M: DevicesWrapCopyMode<LD>,
    X:Dims+ 'static,
    Y:Dims+ 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourcesBatch").finish()//TODO
    }
}

impl <L, C, LD, X, Y> ResourcesBatch<L, C, LD, X, Y>
where
    L: for<'i> Cached<In<'i> = Resource, Out = LD, Ctx=C> + OutId + 'static,
    C:RunCtx+'static,
    LD:Data + WithId + Send + Sync + 'static,
    <LD as Data>::M: DevicesWrapCopyMode<LD>,
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
    L: for<'i> Cached<In<'i> = Resource, Out = LD, Ctx=C> + OutId + 'static,
    C:RunCtx+'static,
    LD:Data + WithId + Send + Sync + 'static,
    <LD as Data>::M: DevicesWrapCopyMode<LD>,
    X:Dims+ 'static,
    Y:Dims+ 'static,
{
    fn train_batch_stream<'a>(&'a self, ctx: C) -> BoxStream<'a, Result<(ThoBatch<X>, Option<BatchLengths>, ThoBatch<Y>), ThoError>> {
        let batches = shuffle_and_chunk_with_rng(self.resources_1_train.len(), self.half_batch_size, &mut rng());

        debug!("ResourcesBatch::train_batch_iter: batches={batches:?}");
        assert!(!batches.is_empty(), "train_batch_stream: batch is empty, self.resources_1_train={:?} self.resources_0_train={:?}",self.resources_1_train,  self.resources_0_train);

        Self::build_batch_stream(
            Arc::clone(&self.build_batch_fn),
            batches,
            Arc::clone(&self.loader),
            ctx,
            Arc::clone(&self.resources_1_train),
            Arc::clone(&self.resources_0_train)
        )
    }

    fn valid_batch_stream<'a>(&'a self, ctx: C) -> BoxStream<'a, Result<(ThoBatch<X>, Option<BatchLengths>, ThoBatch<Y>), ThoError>> {
        let batches = shuffle_and_chunk_with_rng(self.resources_1_valid.len(), self.half_batch_size, &mut rng());

        debug!("ResourcesBatch::train_batch_iter: batches={batches:?}");
        assert!(!batches.is_empty(), "valid_batch_stream: batch is empty, self.resources_1_train={:?} self.resources_0_train={:?}",self.resources_1_train,  self.resources_0_train);

        Self::build_batch_stream(
            Arc::clone(&self.build_batch_fn),
            batches,
            Arc::clone(&self.loader),
            ctx,
            Arc::clone(&self.resources_1_valid),
            Arc::clone(&self.resources_0_valid)
        )
    }

    fn num_batches(&self) -> usize {
        self.resources_1_train.len() / self.half_batch_size as usize
    }
}

impl<L, C, LD, X, Y> ResourcesBatch<L, C, LD, X, Y>
where
    C: 'static + RunCtx,
    L: 'static + OutId + for<'i> Cached<In<'i>=Resource, Out=LD, Ctx=C>+ 'static,
    LD:Data + WithId + Send + Sync + 'static,
    <LD as Data>::M: DevicesWrapCopyMode<LD>,
    X:Dims+ 'static,
    Y:Dims+ 'static,
{
    fn build_batch_stream(build_batch_fn: BuildBatchFn<LD, X, Y, C>, resource_indexes_for_batches: Vec<Box<[usize]>>, loader: Arc<L>, mut ctx: C, x_1: Arc<Vec<Resource>>, x_0: Arc<Vec<Resource>>) -> BoxStream<'static, Result<(ThoBatch<X>, Option<BatchLengths>, ThoBatch<Y>), ThoError>> {
        debug!("ResourcesBatch::build_batch_stream: resource_indexes_for_batches={resource_indexes_for_batches:?}");

        // TODO(perf): each produced batch waits for collect_futures_with_progress to load/apply both sides; add a bounded prefetch queue so ApplyMotifNets overlaps TrainNet forward/backward.
        let futures = futures::stream::iter(resource_indexes_for_batches.into_iter().enumerate()).then(move |(i, batch_keys)|   {
            let build_batch_fn = Arc::clone(&build_batch_fn);
            let loader = Arc::clone(&loader);
            let x_1 = Arc::clone(&x_1);
            let x_0 = Arc::clone(&x_0);
            let mut ctx_for_build_fn = ctx.branch(false);

            async move {
                debug!("ResourcesBatch::build_batch_stream: batch_i={i} batch_keys={batch_keys:?}");

                let batch_len = batch_keys.len();
                let futures = batch_keys.into_iter().map(|index| {
                    let loader = Arc::clone(&loader);
                    let x_1 = Arc::clone(&x_1);
                    let x_0 = Arc::clone(&x_0);
                    let ctx_x1 = ctx_for_build_fn.branch(false);
                    let ctx_x0 = ctx_for_build_fn.branch(false);

                    async move {
                        try_join!(
                            Arc::clone(&loader).run_cached(Arc::new(DevicesWrap::wrap(x_1[index].clone())), ctx_x1),
                            Arc::clone(&loader).run_cached(Arc::new(DevicesWrap::wrap(x_0[index].clone())), ctx_x0)
                        )
                    }
                });

                let (x_1_wrapped, x_0_wrapped) = collect_futures_with_progress(
                    futures,
                    batch_len,
                    crate::buffer_limits::TRAIN_LABEL_NETS_RESOURCES_BATCH_LOAD_BUFFERED_LIMIT,
                    StreamBuffering::Ordered,
                    "ResourcesBatch::build_batch_stream",
                )
                    .await
                    .into_iter()
                    .collect::<Vec<_>>()
                    .into_iter()
                    .fold((vec![], vec![]), |mut a, v| {
                        a.0.push(v.0);
                        a.1.push(v.1);
                        a
                    });

                let r = build_batch_fn(x_1_wrapped, x_0_wrapped, ctx_for_build_fn).await;

                debug!("ResourcesBatch::build_batch_stream: batch_i={i} r={r:?}");

                r
            }
        });

        // preload_one(futures)
        futures.boxed()
    }

}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use tokio::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Duration;
    use futures_util::StreamExt;
    use futures::FutureExt;
    use log::{debug, info, log_enabled, set_max_level, Level, LevelFilter};
    use log::LevelFilter::{Debug, Info, Trace};
    use mockall::predicate::always;
    use parking_lot::RwLock;
    use rand::{rng, Rng, RngExt};
    use tch::Device::Cpu;
    use tch::kind::FLOAT_CPU;
    use tch::Tensor;
    use tokio::io::Join;
    use tokio::task::JoinSet;
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::common::{mock_run_ctx, MockRunCtx, RunCtx};
    use thoosa_common::runner::common::MaxParallelCtx;
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::batches::AsyncDataBatch;
    use thoosa_common::dims::{Batch, LabelProbability, MotifProbability};
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runs::{Labeled, NetResources};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{CommonTrainingRunConfigFields, LabelNetTypeArgsFields, LinNetArgsFields, OptConfig, TrainLabelNetRunConfigFields};
    use thoosa_common::utils::{mix_x_binary_with_labels, pad_stack_with_mask, reset_print_options};
    use thoosa_common::ws::{mock_ws_simp, WsMockBuilder};
    use thoosa_common::tch_test_utils::resource_tensor;
    use thoosa_common::utils;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::RunConfigId;
    use thoosa_shared::{TensorIdSliceIdx, ThoError};
    use thoosa_shared::hyperparameter::Hyperparameter;
    use thoosa_shared::perf_logging::{format_perf_map, PERF};
    use thoosa_shared::traits::Named;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_shared::test_utils::MotifKind;
    use thoosa_shared::val::{NamedVal, Val};
    use thoosa_store::{MockStoreBuilder, TrainedNet};
    use thoosa_with_data::with_data;
    use crate::tasks::apply_motif_nets::motif_test_net_;
    use thoosa_common::dims::ChannelsData;
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use thoosa_common::tch_test_utils::motifs_tensor;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use crate::tasks::train_label_nets::{new_train_label_nets, ResourcesBatch, TrainLabelNets, TrainLabelNetsInput, TrainLabelNetsOutput};

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
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

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

        let train_label_nets = Arc::new(new_train_label_nets(
            vec![select_motifs_run_config_id],
            loader,
            run_configs,
            &ws,
            &store,
            mock_run_ctx::default(1),
            Arc::new(OptConfig::new(30, 10)),
            None,
            Some(0.3),
            Some(3),
            10
        ).await?);


        let input = Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(NetResources::new(Labeled::new(train_group.to_string()), Labeled::new("valid_group".to_string())))));


        let mut run_ctx = mock_run_ctx::default(1);

        let out = train_label_nets.run(input, mock_run_ctx::default(1)).await?;

        debug!("test_train_label_net_3x3: PERF={PERF:?}");
        debug!("test_train_label_net_3x3: PERF=\n{}", format_perf_map(&PERF));


        let ctx_for_data = run_ctx.branch(false);
        with_data!(ctx_for_data, out; |_ctx, out:&TrainLabelNetsOutput|{
            if log_enabled!(Level::Debug) {
                for r in &out.results {
                    debug!("label={} loss={}", r.label, r.loss);
                }
            }

            debug!("test_train_label_net_3x3: 0. out={out:?}");

            assert!(out.results[0].loss < 0.5);
            assert!(out.results[1].loss < 0.5);
            assert!(out.results[2].loss < 0.5);
            Ok::<_, ThoError>(())
        })?;


        debug!("test_train_label_net_3x3: out={out:?}");

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_resources_batch() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

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
                    let ctx_for_data = ctx.branch(false);
                    with_data!(ctx_for_data, [x_1_wrapped], [x_0_wrapped]; |_ctx, x_1:&[&ThoTensor<ChannelsData, SynKey>], x_0:&[&ThoTensor<ChannelsData, SynKey>]|{
                        debug!("ResourcesBatch::train_batch_iter: x_1={x_1:?} x_0={x_0:?}");
                        let x_1 = Tensor::stack(&x_1.into_iter().map(|v|v.t_ref().copy()).collect::<Vec<_>>(), 0);
                        let x_0 = Tensor::stack(&x_0.into_iter().map(|v|v.t_ref().copy()).collect::<Vec<_>>(), 0);

                        let (x, _lengths, y) = mix_x_binary_with_labels(&x_1, None, &x_0, None)?;

                         Ok((x.into(), None, y.into()))
                    })
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
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

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
                    let ctx_for_data = ctx.branch(false);
                    with_data!(ctx_for_data, [x_1_wrapped], [x_0_wrapped]; |_ctx, x_1:&[&ThoTensor<ChannelsData, SynKey>], x_0:&[&ThoTensor<ChannelsData, SynKey>]|{
                        debug!("ResourcesBatch::train_batch_iter: x_1={x_1:?} x_0={x_0:?}");
                        let x_1 = Tensor::stack(&x_1.into_iter().map(|v|v.t_ref().copy()).collect::<Vec<_>>(), 0);
                        let x_0 = Tensor::stack(&x_0.into_iter().map(|v|v.t_ref().copy()).collect::<Vec<_>>(), 0);

                        let (x, _lengths, y) = mix_x_binary_with_labels(&x_1, None, &x_0, None)?;

                         Ok((x.into(), None, y.into()))
                    })
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
