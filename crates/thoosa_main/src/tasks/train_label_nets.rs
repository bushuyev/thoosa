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
use log::{debug, info, trace};
use parking_lot::{Mutex, RwLock};
use polars_arrow::io::iterator::StreamingIterator;
use tch::nn::{AdamW, OptimizerConfig, VarStore};
use tch::{Device, Reduction, Tensor};
use cached_derive::Cached;
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_shared::runner::cache_areas::CacheArea;
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::tho_tensor::{Labels, ThoTensor};
use thoosa_shared::resource::{build_resource_maps, Resource, ResourcesConfig};
use thoosa_shared::traits::{Adjustable, Data, WithId};
use crate::tasks::cross_dims::ChannelsData2;
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::runner::global_ctx::GlobalCtx;
use thoosa_shared::run_configs::RunConfig;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::batches::{AsyncDataBatch, BatchLengths};
use thoosa_common::cuda_utils::cuda_mem;
use thoosa_common::dims::{Batch, Dims, LabelProbability, MotifProbability, ThoBatch};
use thoosa_common::runs::{Labeled, NetResources, SavableResult};
use thoosa_common::train_nets::{conv_label_node, lin_label_node, opt_train, self_attention_label_node, CommonTrainingRunConfig, ConvNetArgs, ConvNetParams, LabelNetParams, LabelNetTypeArgs, LabelNetTypeParams, LinLenNetParams, LinNetArgs, OptConfig, SelfAttentionNetArgs, SelfAttentionNetParams, TrainLabelNetsRunConfig, TrainLabelNetsRunConfigFields, TrainLabelNetsRunConfigResult};
use thoosa_common::utils::{mix_x_binary_with_labels, pad_stack_with_mask, ListenerResponse, TrainListener};
use thoosa_common::ws::Workspace;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::{collect_futures_with_progress, mono_key, shuffle_and_chunk_with_rng, CopyToDevice, StreamBuffering, ThoError};
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::run_configs::{OptRunConfigs, RunId};
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_store::Store;
use thoosa_with_data::with_data;
use crate::tasks::cross_loader::ResourcesConfigInput;
use thoosa_shared::runner::run_ctx_2::RunCtxWithPermits;
use rand::rng;
use tch::kind::FLOAT_CPU;
use tokio::{join, try_join};
use thoosa_shared::runner::permits::{RunnerPerm, MAX_PERMITS};
use thoosa_dummy_data_derive::DummyData;
use crate::tasks::apply_motif_nets::ApplyMotifNets;
use crate::tasks::stages_seq::ABStage;
use crate::tasks::train_net::{TrainNet, TrainNetInput, TrainNetOutput};
use thoosa_shared::comp_id::SynKey;

#[derive(Debug)]
pub struct TrainLabelNets<L, LM, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    LM: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<MotifProbability, SynKey>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    train_motif_nets_run_ids: Vec<RunId>,
    run_configs: Arc<OptRunConfigs<
        Vec<NamedVal<TrainLabelNetsRunConfigFields>>,
        Vec<NamedRange<TrainLabelNetsRunConfigFields>>,
        Vec<Hyperparameter<TrainLabelNetsRunConfigFields>>,
        TrainLabelNetsRunConfigFields,
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
    train_motif_nets_run_ids: Vec<RunId>,
    loader: Arc<L>,
    run_configs: OptRunConfigs<Vec<NamedVal<TrainLabelNetsRunConfigFields>>, Vec<NamedRange<TrainLabelNetsRunConfigFields>>, Vec<Hyperparameter<TrainLabelNetsRunConfigFields>>, TrainLabelNetsRunConfigFields>,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    stage_ctx:R,
    opt_config:Arc<OptConfig>,
    labels: Option<Vec<String>>,
    max_loss: Option<f32>,
    slices_step: u32
) -> Result<TrainLabelNets<L, ABStage<L, ApplyMotifNets<C>, C>, C>, ThoError>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
    R:RunCtx + Adjustable + 'static
{
    //gpt Build a motif-net application stage to turn raw resources into motif probabilities.
    let apply_motif_nets = Arc::new(ApplyMotifNets::new(
        ws,
        store,
        train_motif_nets_run_ids.clone(),
        stage_ctx,
        max_loss,
        slices_step,
        #[cfg(test)]
        None
    ).await?);

    let num_nets = apply_motif_nets.num_nets() as u32;
    if num_nets == 0 {
        return Err(ThoError::new("No motifs nets to train on"));
    }

    let applied_motif_loader = Arc::new(ABStage::new(
        SynKey::new([
            mono_key!("Loader_with_motifs", "stage"),
            mono_key!(format!("{train_motif_nets_run_ids:?}"), "train_motif_nets_run_ids"),
        ]),
        loader,
        apply_motif_nets
    ));

    TrainLabelNets::new(
        train_motif_nets_run_ids,
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
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    LM: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<MotifProbability, SynKey>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{

    pub async fn new(
        train_motif_nets_run_ids: Vec<RunId>,
        applied_motif_loader: Arc<LM>,
        num_nets:u32,
        run_configs: OptRunConfigs<Vec<NamedVal<TrainLabelNetsRunConfigFields>>, Vec<NamedRange<TrainLabelNetsRunConfigFields>>, Vec<Hyperparameter<TrainLabelNetsRunConfigFields>>, TrainLabelNetsRunConfigFields>,
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
            run_configs: Arc::new(run_configs),
            pd: PhantomData,
            applied_motif_loader,
            ws:Arc::clone(&ws),
            num_nets,
            store: Arc::clone(store),
            opt_config,
            train_motif_nets_run_ids,
            labels,

        })
    }

}

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
    id: RunId,
    train_motif_nets_run_ids: Vec<RunId>,
    results: Vec<TrainLabelNetsRunConfigResult>
}
impl WithId for TrainLabelNetsOutput {
    type ID = RunId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl WithId for Box<TrainLabelNetsOutput> {
    type ID = RunId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}
impl Output for TrainLabelNetsOutput {

}

#[async_trait]
impl SavableResult for TrainLabelNetsOutput {
    fn value(&self) -> f32 {
        0.0
    }

    async fn save(&mut self, store: &Arc<Box<dyn Store>>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<u32, ThoError> {
        let run_id = store
            .save_run_config(
                Some(&self.train_motif_nets_run_ids),
                TrainLabelNetsRunConfig::type_name(),
                "TrainLabelNetsResult",
                "todo",
                HashMap::new(),
            )
            .await?;

        self.id = run_id;

        for result in &mut self.results {
            result.set_train_label_nets_run_id(run_id);
            result.save(store, ws).await?;
        }

        Ok(run_id)
    }
}

impl TrainLabelNetsOutput {
    pub fn new(train_motif_nets_run_ids: Vec<RunId>, results: Vec<TrainLabelNetsRunConfigResult>) -> Self {
        Self { id: 0, train_motif_nets_run_ids, results }
    }
}

impl <L, LM, C> Stage for TrainLabelNets<L, LM, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    LM: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<MotifProbability, SynKey>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    type Ctx = C;
    type In<'a> = ResourcesConfigInput<NetResources<Labeled>>;
    type Out = TrainLabelNetsOutput;
}

impl<L, LM, C> AsyncStage for TrainLabelNets<L, LM, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    LM: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<MotifProbability, SynKey>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainLabelNets::run"))]
    fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> impl Future<Output=Result<Arc<DevicesWrap<Self::Out>>, ThoError>> + Send {
        let cache_area = ctx.cache_area();

        let train_label_nets_run_configs = Arc::clone(&self.run_configs);
        let cache_area = cache_area.clone();
        let channels = thoosa_shared::with_perf_log(
            || self.ws.try_read_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout").channels() as i64,
            "TrainLabelNets::run/ws_read_channels",
        );

        debug!("TrainLabelNets::run: channels={channels} num_nets={}", self.num_nets);

        async move {
            //gpt Split resources into train/valid sets per label for binary label-vs-rest training.
            let resources_config = with_data!(ctx.branch(Some("train_label_nets_resources"), false, false), input; |_ctx, input:&ResourcesConfigInput<NetResources<Labeled>>|{
                Ok::<_, ThoError>(input.resources_config().clone())
            })?;
            let (train_labels_resource_keys, valid_label_resource_keys) = resources_config.to_resources(&self.store).await?;

            let (labels_train_and_valid_sets, labels_others_train_valid_resources) = build_resource_maps(train_labels_resource_keys, valid_label_resource_keys);
            debug!("TrainLabelNets::run: labels_others_train_valid_resources={labels_others_train_valid_resources:?}");
            debug!("TrainLabelNets::run: labels_train_and_valid_sets={labels_train_and_valid_sets:?}");

            let labels_train_and_valid_sets_len = labels_train_and_valid_sets.len();
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
                let mut ctx_for_task = ctx.branch(Some(&format!("Train_{label}")), true, false);//into is needed because mock needs SynKey or &str

                let opt_config = Arc::clone(&self.opt_config);
                let train_label_nets_run_configs = Arc::clone(&train_label_nets_run_configs);
                let applied_motif_loader = Arc::clone(&self.applied_motif_loader);
                let train_motif_nets_run_ids = self.train_motif_nets_run_ids.clone();

                debug!("TrainLabelNets::run: label={label} train_motif_nets_run_ids={train_motif_nets_run_ids:?} train_label_nets_run_configs={train_label_nets_run_configs:?}");


                let labels_others_train_valid_resources = Arc::clone(&labels_others_train_valid_resources);
                let num_nets = self.num_nets;
                async move {
                    debug!("TrainLabelNets::run: label={label} resources={resources_1:?}");

                    let resources_0 = labels_others_train_valid_resources.get(label).expect(&format!("no resources not belonging to {label}"));
                    debug!("TrainLabelNets::run: label={label} resources_x={resources_0:?}");

                    let resources_1_train = Arc::new(resources_1.0.clone());
                    let resources_1_valid = Arc::new(resources_1.1.clone());
                    let resources_0_train = Arc::new(resources_0.0.clone());
                    let resources_0_valid = Arc::new(resources_0.1.clone());
                    let label = label.clone();


                    let train_net = Arc::new(TrainNet::new(
                        move |ctx_for_batch:C,rc:&TrainLabelNetsRunConfig| {

                            let applied_motif_loader = Arc::clone(&applied_motif_loader);

                            ready(Ok(ResourcesBatch::new(
                                   Arc::clone(&applied_motif_loader),
                                   rc.common_training_args.batch_size,
                                   Arc::clone(&resources_1_train),
                                   Arc::clone(&resources_1_valid),
                                   Arc::clone(&resources_0_train),
                                   Arc::clone(&resources_0_valid),
                                   Box::new(move |x_1_wrapped: Vec<Arc<DevicesWrap<ThoTensor<MotifProbability, SynKey>>>>, x_0_wrapped: Vec<Arc<DevicesWrap<ThoTensor<MotifProbability, SynKey>>>>, mut ctx: C| {
                                       async move {
                                           let (x, lengths, y) = with_data!(ctx.branch(Some("train_label_nets_batch_data"), false, false), [x_1_wrapped], [x_0_wrapped]; |_ctx, x_1:&[&ThoTensor<MotifProbability, SynKey>], x_0:&[&ThoTensor<MotifProbability, SynKey>]|{
                                               debug!("TrainLabelNets/ResourcesBatch::train_batch_iter: x_1={x_1:?} x_0={x_0:?}");

                                               let max_input = x_1.iter().chain(x_0.iter()).map(|x|x.t_ref().size()[1]).max().expect("no max_input");

                                               debug!("TrainLabelNets/ResourcesBatch::train_batch_iter: max_input={max_input}");

                                               let (x_1, lengths_1) = pad_stack_with_mask(x_1, num_nets as i64, max_input)?;
                                               let (x_0, lengths_0) = pad_stack_with_mask(x_0, num_nets as i64, max_input)?;

                                               debug!("TrainLabelNets/ResourcesBatch::train_batch_iter: x_1.size={:?}, x_0.size={:?}, lengths_1={:?}, lengths_0={:?}", x_1.size(), x_0.size(), lengths_1, lengths_0);

                                               mix_x_binary_with_labels(&x_1, Some(lengths_1.as_slice()), &x_0, Some(lengths_0.as_slice()))
                                           })?;

                                           debug!("TrainLabelNets/ResourcesBatch::train_batch_iter: x.size={:?} y.size={:?} lengths={lengths:?}", x.size(), y.size());


                                           Ok((x.into(), lengths, y.into()))
                                       }.boxed()
                                   }),
                                ctx_for_batch
                               ))).boxed()


                       },
                        move |rc:&TrainLabelNetsRunConfig, vs: &VarStore| {
                            match rc.net {
                                LabelNetTypeArgs::ConvNet(ConvNetArgs {
                                     num_conv_layers, hidden_conv_channels, num_head_layers, num_head_features, kernel, ref joint

                                }) => {
                                    //TODO impl From<(num_nets, LabelNetTypeArgs)> for LabelNetParams
                                    let net_params = LabelNetParams::new(num_nets, LabelNetTypeParams::ConvNetParams(ConvNetParams::new(num_conv_layers, hidden_conv_channels, kernel, num_head_layers, num_head_features, joint.into())));

                                    Ok((conv_label_node(&net_params)?.build_tch_module::<Batch<MotifProbability>, Batch<LabelProbability>>(&vs.root())?, net_params))
                                }
                                LabelNetTypeArgs::LinNet(LinNetArgs {
                                     num_layers, slices_len
                                }) => {
                                    let net_params = LabelNetParams::new(num_nets, LabelNetTypeParams::LinNetParams(LinLenNetParams::new(slices_len, num_layers)));

                                    Ok((lin_label_node(&net_params)?.build_tch_module::<Batch<MotifProbability>, Batch<LabelProbability>>(&vs.root())?, net_params))
                                }
                                LabelNetTypeArgs::SelfAttentionNet(SelfAttentionNetArgs { d_model_heads, n_heads, ff_mult, dropout }) => {
                                    let net_params = LabelNetParams::new(num_nets, LabelNetTypeParams::SelfAttentionNetParams(SelfAttentionNetParams::new(d_model_heads, n_heads, ff_mult, dropout)));
                                    Ok((self_attention_label_node(&net_params)?.build_tch_module::<Batch<MotifProbability>, Batch<LabelProbability>>(&vs.root())?, net_params))
                                }
                            }

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
                        move |rc:&TrainLabelNetsRunConfig, vs: &VarStore|{
                            let o = AdamW::default().amsgrad(true).build(&vs, rc.common_training_args.lr).map_err(|e|e.into());
                            debug!("TrainLabelNets::run, optimizer: run optimizer: o={o:?}");
                            o
                        },
                        |t:ThoBatch<MotifProbability>| {
                            debug!("TrainLabelNets::run, x_f: t.size={:?} cuda_mem={:?}",t.size(),cuda_mem());
                            t
                        },
                        |rc:&TrainLabelNetsRunConfig|{
                            rc.common_training_args.cycles
                        },
                        1,
                        format!("label_{label}")
                    ));

                    let result = train_net.run(
                        Arc::new(DevicesWrap::wrap(TrainNetInput::new(SynKey::new([mono_key!("", "empty")])), cache_area.into())),
                        ctx_for_task
                    ).await?;


                    let result = Arc::into_inner(result).expect("can't get result from train_net").take_data(0).await?;
                    debug!("TrainLabelNets::run, done: label={label} min_loss={:?}", result.min_loss);

                    Ok::<_, ThoError>(
                        TrainLabelNetsRunConfigResult::new(
                            train_motif_nets_run_ids, //TODO
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

            Ok(Arc::new(DevicesWrap::wrap(TrainLabelNetsOutput::new(self.train_motif_nets_run_ids.clone(), results), cache_area.into())))
        }
    }
}

#[allow(type_alias_bounds)]
type BuildBatchFn<LD:Dims+ 'static, X:Dims+ 'static, Y:Dims+ 'static, C> = Box<dyn Fn(

    Vec<Arc<DevicesWrap<ThoTensor<LD, SynKey>>>>,
    Vec<Arc<DevicesWrap<ThoTensor<LD, SynKey>>>>, C

) -> BoxFuture<'static, Result<(ThoBatch<X>, Option<BatchLengths>, ThoBatch<Y>), ThoError>> + Send + Sync>;

// #[derive(Debug)]
struct ResourcesBatch<L, C, LD, X, Y>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<LD, SynKey>, Ctx=C> + OutId + 'static,
    C:RunCtx+'static,
    LD:Dims+ 'static,
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
    ctx: parking_lot::Mutex<C>
}

impl <L, C, LD, X, Y> Debug for  ResourcesBatch<L, C, LD, X, Y>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<LD, SynKey>, Ctx=C> + OutId + 'static,
    C:RunCtx+'static,
    LD:Dims+ 'static,
    X:Dims+ 'static,
    Y:Dims+ 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourcesBatch").finish()//TODO
    }
}

impl <L, C, LD, X, Y> ResourcesBatch<L, C, LD, X, Y>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<LD, SynKey>, Ctx=C> + OutId + 'static,
    C:RunCtx+'static,
    LD:Dims+ 'static,
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
        ctx: C
    ) -> Self {
        debug!("ResourcesBatch::new: resources_1_train={resources_1_train:?}");
        debug!("ResourcesBatch::new: resources_1_valid={resources_1_valid:?}");
        debug!("ResourcesBatch::new: resources_0_train={resources_0_train:?}");
        debug!("ResourcesBatch::new: resources_0_valid={resources_0_valid:?}");

        assert_eq!(resources_1_train.len(), resources_0_train.len());//train
        assert_eq!(resources_1_valid.len(), resources_0_valid.len());//valid

        //TODO replacement?
        let half_batch_size = batch_size / 2;
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
            ctx: parking_lot::Mutex::new(ctx)
        }
    }
}

#[async_trait]
impl <L,C, LD, X, Y> AsyncDataBatch<X, Y> for ResourcesBatch<L, C, LD, X, Y>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<LD, SynKey>, Ctx=C> + OutId + 'static,
    C:RunCtx+'static,
    LD:Dims+ 'static,
    X:Dims+ 'static,
    Y:Dims+ 'static,
{
    async fn train_batch_stream(&self) -> BoxStream<'_, Result<(ThoBatch<X>, Option<BatchLengths>, ThoBatch<Y>), ThoError>> {
        //TODO if self.batch_size > train_len?
        let batches = shuffle_and_chunk_with_rng(self.resources_1_train.len(), self.half_batch_size, &mut rng());

        debug!("ResourcesBatch::train_batch_iter: batches={batches:?}");


        Self::build_batch_stream(
            &self.build_batch_fn,
            batches,
            Arc::clone(&self.loader),
            &self.ctx,
            Arc::clone(&self.resources_1_train),
            Arc::clone(&self.resources_0_train)
        )
    }

    async fn valid_batch_stream(&self) -> BoxStream<'_, Result<(ThoBatch<X>, Option<BatchLengths>, ThoBatch<Y>), ThoError>> {
        let batches = shuffle_and_chunk_with_rng(self.resources_1_valid.len(), self.half_batch_size, &mut rng());

        debug!("ResourcesBatch::train_batch_iter: batches={batches:?}");


        Self::build_batch_stream(
            &self.build_batch_fn,
            batches,
            Arc::clone(&self.loader),
            &self.ctx,
            Arc::clone(&self.resources_1_valid),
            Arc::clone(&self.resources_0_valid)
        )
    }
}

impl<L, C, LD, X, Y> ResourcesBatch<L, C, LD, X, Y>
where
    C: 'static + RunCtx,
    L: 'static + OutId + for<'i> Cached<In<'i>=Resource, Out=ThoTensor<LD, SynKey>, Ctx=C>+ 'static,
    LD:Dims+ 'static,
    X:Dims+ 'static,
    Y:Dims+ 'static,
{
    fn build_batch_stream<'a>(build_batch_fn:&'a BuildBatchFn<LD, X, Y, C>, resource_indexes_for_batches: Vec<Box<[usize]>>, loader: Arc<L>, ctx: &'a parking_lot::Mutex<C>, x_1: Arc<Vec<Resource>>, x_0: Arc<Vec<Resource>>) -> BoxStream<'a, Result<(ThoBatch<X>, Option<BatchLengths>, ThoBatch<Y>), ThoError>> {

        futures_util::stream::iter(resource_indexes_for_batches)
            .map(move |batch_keys| {
                let loader = Arc::clone(&loader);
                let x_1 = Arc::clone(&x_1);
                let x_0 = Arc::clone(&x_0);
                let mut ctx_for_build_fn = {
                    let mut guard = thoosa_shared::with_perf_log(
                        || ctx.try_lock_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
                        "ResourcesBatch::build_batch_stream/ctx_lock",
                    );
                    guard.branch(Some("for_build_fn"), false, false)
                };
                async move {
                    debug!("ResourcesBatch::build_batch_stream: ctx={ctx:?}");
                    let batch_len = batch_keys.len();

                    // load samples for this batch concurrently
                    let futures = batch_keys
                        .into_iter()
                        .map(|index| {
                            let loader = Arc::clone(&loader);
                            let x_1 = Arc::clone(&x_1);
                            let x_0 = Arc::clone(&x_0);

                            debug!("ResourcesBatch::build_batch_stream: before move");

                            let ( ctx_x1, ctx_x0) = {

                                (ctx_for_build_fn.branch(Some("x_1_batch_loader"), false, false), ctx_for_build_fn.branch(Some("x_0_batch_loader"), false, false))
                            };
                            async move {
                                (async {
                                    try_join!(
                                        Arc::clone(&loader).run_cached(Arc::new(DevicesWrap::wrap(x_1[index].clone(), CacheArea::default())), ctx_x1),
                                        Arc::clone(&loader).run_cached(Arc::new(DevicesWrap::wrap(x_0[index].clone(), CacheArea::default())), ctx_x0)
                                    )
                                })
                                .await
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
                        a.0.push(v.0); //1
                        a.1.push(v.1); //0
                        a
                    });

                    build_batch_fn(x_1_wrapped, x_0_wrapped, ctx_for_build_fn).await
                }
            })
            .buffer_unordered(crate::buffer_limits::TRAIN_LABEL_NETS_RESOURCES_BATCH_STREAM_BUFFER_UNORDERED_LIMIT)//TODO config
            .boxed()
    }

}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use tokio::fs;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Duration;
    use futures_util::StreamExt;
    use futures::FutureExt;
    use log::{debug, info, set_max_level, LevelFilter};
    use log::LevelFilter::{Debug, Info, Trace};
    use mockall::predicate::always;
    use parking_lot::RwLock;
    use rand::{rng, Rng, RngExt};
    use tch::Device::Cpu;
    use tch::kind::FLOAT_CPU;
    use tch::Tensor;
    use tokio::io::Join;
    use tokio::task::JoinSet;
    use thoosa_shared::runner::common::cachable::DevicesWrap;
    use thoosa_shared::runner::cache_areas::CacheArea;
    use thoosa_shared::runner::common::{mock_run_ctx, MockRunCtx, RunCtx};
    use thoosa_shared::runner::common::PermitsCtx;
    use thoosa_shared::runner::common::runnable::AsyncStage;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_common::batches::AsyncDataBatch;
    use thoosa_common::dims::{Batch, LabelProbability};
    use thoosa_common::runs::{Labeled, NetResources};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{CommonTrainingRunConfigFields, LabelNetTypeArgsFields, LinNetArgsFields, OptConfig, TrainLabelNetsRunConfigFields};
    use thoosa_common::utils::{mix_x_binary_with_labels, reset_print_options};
    use thoosa_common::ws::{mock_ws_simp, WsMockBuilder};
    use thoosa_common::tch_test_utils::resource_tensor;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::{TensorIdSliceIdx, ThoError};
    use thoosa_shared::hyperparameter::Hyperparameter;
    use thoosa_shared::perf_logging::{format_perf_map, PERF};
    use thoosa_shared::traits::Named;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_shared::test_utils::MotifKind;
    use thoosa_shared::val::{NamedVal, Val};
    use thoosa_store::MockStoreBuilder;
    use thoosa_with_data::with_data;
    use crate::tasks::apply_motif_nets::motif_test_net_;
    use crate::tasks::cross_dims::ChannelsData2;
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::select_motifs::tests::motifs_tensor;
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

        let input_n_0 = 1000;
        let find_motifs_run_id:RunId = 1;
        let select_motifs_run_id:RunId = 2;
        let motif_width = 100;

        let motif_0_0_id = 0; //channel 0 motif 0
        let motif_0_1_id = 1; //channel 0 motif 1
        let motif_1_0_id = 0; //channel 1 motif 0

        let pulse_motif = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_0_0", "motif_id")]), motifs_tensor(100, 3, MotifKind::Pulse))?);
        let triangle_motif = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_0_1", "motif_id")]), motifs_tensor(100, 3, MotifKind::Triangle))?);
        let sin_motif = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_1_0", "motif_id")]), motifs_tensor(100, 3, MotifKind::Sin))?);
        let train_group = "train_group";
        let valid_group = "valid_group";

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 3, 100, input_n_0 as u32)
                .with_load_motif_nets(vec![
                    vec![
                        //channel 0
                        motif_test_net_(Cpu, Arc::clone(&pulse_motif), 10),
                        motif_test_net_(Cpu, Arc::clone(&triangle_motif), 10),
                        motif_test_net_(Cpu, Arc::clone(&sin_motif), 10),
                    ],
                    vec![
                        //channel 1
                        motif_test_net_(Cpu, Arc::clone(&sin_motif), 10),
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
            ], 100, 10.0)?;

            a.0.insert(r_a.clone(), t);

            a.1.extend(intervals[0][0].iter().map(|interval|TensorIdSliceIdx(r_a.id(), interval.start as u32)));


            let (t, intervals) = resource_tensor(r_b.length() as usize, &[
                vec![(3, motif_width, MotifKind::Triangle)],
                vec![(1, motif_width, MotifKind::Sin)],
                vec![]
            ], 100, 10.0)?;

            a.0.insert(r_b.clone(), t);

            a.1.extend(intervals[0][0].iter().map(|interval|TensorIdSliceIdx(r_b.id(), interval.start as u32)));


            let (t, intervals) = resource_tensor(r_c.length() as usize, &[
                vec![(3, motif_width, MotifKind::Sin)],
                vec![(1, motif_width, MotifKind::Sin)],
                vec![]
            ], 100, 10.0)?;

            a.0.insert(r_b.clone(), t);

            a.1.extend(intervals[0][0].iter().map(|interval|TensorIdSliceIdx(r_c.id(), interval.start as u32)));


            let (t, intervals) = resource_tensor(r_x.length() as usize, &[
                vec![],
                vec![],
                vec![]
            ], 100, 10.0)?;

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
                (select_motifs_run_id, Ok(
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
            .build());


        let run_configs = thoosa_shared::run_configs::OptRunConfigs::new(
            thoosa_shared::run_configs::GridRunConfigs::new(
                thoosa_shared::run_configs::ConstConfigs::new(vec![
                    NamedVal::new(TrainLabelNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Cycles), Val::Int(100)),
                    NamedVal::new(TrainLabelNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::BatchSize), Val::Int(8)),
                    // NamedVal::new(TrainLabelNetsRunConfigFields::Net(LabelNetTypeArgsFields::FixedNumSlices(FixedNumSlicesNetArgsFields::NumLayers)), Val::Int(3)),
                    NamedVal::new(TrainLabelNetsRunConfigFields::Net(LabelNetTypeArgsFields::LinNet(LinNetArgsFields::SlicesLen)), Val::Int(1000)),//TODO used as num features in any_len_label_node
                ]),
                vec![
                    // NamedRange::new(TrainLabelNetsRunConfigFields::NumLayers, Val::Int(0)..Val::Int(10), (Val::Int(10), 0))
                ],
            ),
            vec![
                Hyperparameter::UniformFloat{name: TrainLabelNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), lower: 0.0001, upper: 0.001},
                Hyperparameter::UniformInt{name: TrainLabelNetsRunConfigFields::Net(LabelNetTypeArgsFields::LinNet(LinNetArgsFields::NumLayers)), lower: 1, upper: 10}
            ],
        );

        // let ctx = MockRunCtx::purge_gpu_caches_context();
        // ctx.expect().with(always()).times(1..).returning( |_| Box::pin(async { Ok(()) }));

        let train_label_nets = Arc::new(new_train_label_nets(
            vec![select_motifs_run_id],
            loader,
            run_configs,
            &ws,
            &store,
            mock_run_ctx::default(1),
            Arc::new(OptConfig::new(30, 10)),
            None,
            Some(0.3),
            10
        ).await?);


        let input = Arc::new(DevicesWrap::wrap(
            ResourcesConfigInput::new(NetResources::new(Labeled::new(train_group.to_string()), Labeled::new("valid_group".to_string()))),
            CacheArea::default()
        ));

        // let ctx = MockRunCtx::purge_gpu_caches_context();
        // ctx.expect().with(always()).times(0..).returning( |_| Box::pin(async { Ok(()) }));

        let mut run_ctx = mock_run_ctx::default(1);

        let out = train_label_nets.run(input, mock_run_ctx::default(1)).await?;

        info!("test_train_label_net_3x3: PERF={PERF:?}");
        info!("test_train_label_net_3x3: PERF=\n{}", format_perf_map(&PERF));


        let ctx_for_data = run_ctx.branch(Some("train_label_nets_assert"), false, false);
        with_data!(ctx_for_data, out; |_ctx, out:&TrainLabelNetsOutput|{
             for r in &out.results {
                info!("label={} loss={}", r.label, r.loss);
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


        let resource_batch: ResourcesBatch<_, _, ChannelsData2, ChannelsData2, LabelProbability> = ResourcesBatch::new(
            loader,
            6,
            resources_1_train,
            resources_1_valid,
            resources_0_train,
            resources_0_valid,
            Box::new(|x_1_wrapped:Vec<Arc<DevicesWrap<ThoTensor<ChannelsData2, SynKey>>>>, x_0_wrapped:Vec<Arc<DevicesWrap<ThoTensor<ChannelsData2, SynKey>>>>, mut ctx: MockRunCtx|{
                async move {
                    let ctx_for_data = ctx.branch(Some("train_label_nets_batch_data"), false, false);
                    with_data!(ctx_for_data, [x_1_wrapped], [x_0_wrapped]; |_ctx, x_1:&[&ThoTensor<ChannelsData2, SynKey>], x_0:&[&ThoTensor<ChannelsData2, SynKey>]|{
                        debug!("ResourcesBatch::train_batch_iter: x_1={x_1:?} x_0={x_0:?}");
                        let x_1 = Tensor::stack(&x_1.into_iter().map(|v|v.t_ref().copy()).collect::<Vec<_>>(), 0);
                        let x_0 = Tensor::stack(&x_0.into_iter().map(|v|v.t_ref().copy()).collect::<Vec<_>>(), 0);

                        let (x, _lengths, y) = mix_x_binary_with_labels(&x_1, None, &x_0, None)?;

                         Ok((x.into(), None, y.into()))
                    })
                }.boxed()
            }),
            mock_run_ctx::default(0)
        );

        let mut batch = resource_batch.train_batch_stream().await;
        let (_batch_x, _lengths,  _batch_y) = batch.next().await.expect("no batch").expect("batch failed");
        assert!(batch.next().await.is_none()); //should be only one

        let mut batch = resource_batch.valid_batch_stream().await;
        let (_batch_x, _lengths, _batch_y) = batch.next().await.expect("no batch").expect("batch failed");
        assert!(batch.next().await.is_none()); //should be only one



        Ok(())
    }

}
