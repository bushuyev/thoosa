use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use futures::Stream;
use futures_util::{StreamExt, TryStreamExt};
use futures_util::future::BoxFuture;
use futures::FutureExt;
use futures_util::stream::BoxStream;
use log::{debug, trace};
use parking_lot::RwLock;
use polars_arrow::io::iterator::StreamingIterator;
use tch::nn::{AdamW, OptimizerConfig, VarStore};
use tch::{Device, Reduction, Tensor};
use cached_derive::Cached;
use caching_runner::common::cachable::{Cached, DevicesWrap};
use caching_runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::tho_tensor::{Labels, ThoTensor};
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, SimpleId, WithId};
use crate::tasks::cross_dims::ChannelsData2;
use caching_runner::common::{OutId, RunCtx};
use caching_runner::global_ctx::GlobalCtx;
use thoosa_common::applicables::ThoDevice;
use thoosa_common::batches::{AsyncDataBatch, ThoBatchIter};
use thoosa_common::cuda_utils::cuda_mem;
use thoosa_common::dims::{Batch, Dims, LabelProbability, MotifProbability, ThoBatch};
use thoosa_common::runs::{Labeled, NetResources, SavableResult};
use thoosa_common::train_nets::{any_len_label_node, flat_label_node, opt_train_2, FixedNumSlicesNetArgs, FixedNumSlicesNetParams, LabelNetParams, LabelNetTypeArgs, LabelNetTypeParams, MaxPaddingNetArgs, OptConfig, TrainLabelNetsRunConfig, TrainLabelNetsRunConfigFields, TrainLabelNetsRunConfigResult};
use thoosa_common::utils::{mix_x_binary_with_labels, pad_to_2, train_2, train_3};
use thoosa_common::ws::Workspace;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::{shuffle_and_chunk_with_rng, CopyToDevice, ThoosaError};
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::run_configs::{OptRunConfigs, RunId};
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_store::Store;
use thoosa_with_data::with_data;
use crate::tasks::apply_motif_nets::ApplyMotifNets;
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::run_ctx_2::MotifsCtx;
use rand::{rng, seq::SliceRandom};
use tch::kind::FLOAT_CPU;
use tokio::{join, try_join};
use thoosa_dummy_data_derive::DummyData;
use crate::tasks::stages_seq::ABStage;

#[derive(Debug)]
pub struct TrainLabelNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    select_motifs_run_ids: Vec<RunId>,
    run_configs: Arc<OptRunConfigs<
        Vec<NamedVal<TrainLabelNetsRunConfigFields>>,
        Vec<NamedRange<TrainLabelNetsRunConfigFields>>,
        Vec<Hyperparameter<TrainLabelNetsRunConfigFields>>,
        TrainLabelNetsRunConfigFields,
    >>,
    loader: Arc<L>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    apply_motif_nets:Arc<ApplyMotifNets<C>>,
    store: Arc<RwLock<Box<dyn Store>>>,
    opt_config:Arc<OptConfig>,
}

impl <L, C> TrainLabelNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    pub async fn new<R>(
        select_motifs_run_ids: Vec<RunId>,
        loader: Arc<L>,
        run_configs: OptRunConfigs<Vec<NamedVal<TrainLabelNetsRunConfigFields>>, Vec<NamedRange<TrainLabelNetsRunConfigFields>>, Vec<Hyperparameter<TrainLabelNetsRunConfigFields>>, TrainLabelNetsRunConfigFields>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        stage_ctx:Arc<parking_lot::RwLock<R>>,
        opt_config:Arc<OptConfig>
    ) -> Result<Self, ThoosaError>
where
    R:RunCtx + MotifsCtx + Adjustable + 'static
    {

        Ok(Self {
            run_configs: Arc::new(run_configs),
            loader,
            ws:Arc::clone(&ws),
            apply_motif_nets: Arc::new(ApplyMotifNets::new("".into(), ws, store, select_motifs_run_ids.clone(), stage_ctx).await?),
            store: Arc::clone(store),
            opt_config,
            select_motifs_run_ids
        })
    }

}

#[derive(Debug)]
pub struct TrainLabelNetsInput {
    id:SimpleId
}

impl TrainLabelNetsInput {
    pub fn new(id: SimpleId) -> Self {
        Self { id }
    }
}

#[async_trait]
impl CopyToDevice for TrainLabelNetsInput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoosaError> where Self:Sized {
        todo!()
    }
}

impl Data for TrainLabelNetsInput {
    type Metrics = ();

    fn size(&self) -> Self::Metrics {
        ()
    }

    fn device(&self) -> usize {
        0
    }
}


impl Input for TrainLabelNetsInput {

}

impl WithId for TrainLabelNetsInput {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}


#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct TrainLabelNetsOutput {
    id:SimpleId,
    results: Vec<TrainLabelNetsRunConfigResult>
}
impl WithId for TrainLabelNetsOutput {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}
impl Output for TrainLabelNetsOutput {

}

impl SavableResult for TrainLabelNetsOutput {
    fn value(&self) -> f32 {
        todo!()
    }

    fn save(&mut self, store: &Arc<RwLock<Box<dyn Store>>>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<u32, ThoosaError> {
       for result in &mut self.results {

           result.save(store, ws)?;
       }

        Ok(0)
    }
}

impl TrainLabelNetsOutput {
    pub fn new(id: SimpleId, results: Vec<TrainLabelNetsRunConfigResult>) -> Self {
        Self { id, results }
    }
}

impl <L, C> Stage for TrainLabelNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    type Ctx = C;
    type In<'a> = ResourcesConfigInput<NetResources<Labeled>>;
    type Out = TrainLabelNetsOutput;
}

impl<L, C> AsyncStage for TrainLabelNets<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> impl Future<Output=Result<Arc<DevicesWrap<Self::Out>>, ThoosaError>> + Send {
        let (device_idx, cache_area, ctx_name) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area(), guard.name())
        };

        let train_motifs_nodes_run_configs = Arc::clone(&self.run_configs);
        let ctx = Arc::clone(&ctx);
        let store = Arc::clone(&self.store);
        let cache_area = cache_area.clone();
        let channels = self.ws.read().channels() as i64;
        let max_input = 10_000; //self.ws.read().max_input() as i64;
        let slices_len = (max_input as u32 - self.apply_motif_nets.motif_width)/self.apply_motif_nets.slice_step + 1;
        let num_nets = self.apply_motif_nets.num_nets() as u32;

        debug!("TrainLabelNets::run: channels={channels} max_input={max_input} self.apply_motif_nets.slice_step={} slices_len={slices_len}", self.apply_motif_nets.slice_step);

        async move {
            let (train_labels_resource_keys, valid_label_resource_keys) = with_data!(0, input; |input:&ResourcesConfigInput<NetResources<Labeled>>|{
                Ok::<_, ThoosaError>(input.resources_config().to_resources(&self.store)?)
            })?;

            let (labels_tain_and_valid_sets, labels_others_train_valid_resources) = Self::build_resource_maps(train_labels_resource_keys, valid_label_resource_keys);
            debug!("TrainLabelNets::run: labels_others_train_valid_resources={labels_others_train_valid_resources:?}");

            let out_key = {
                let out_id = SimpleId("todo");  //self.out_id(input.id());
                Arc::new(out_id) as Arc<dyn CachableIdTr>
            };

            let results = futures::stream::iter(labels_tain_and_valid_sets.clone())
            .map(|label_tain_and_valid_sets|{


                let opt_config = Arc::clone(&self.opt_config);
                let ctx = Arc::clone(&ctx);
                let out_key = Arc::clone(&out_key);
                let train_motifs_nodes_run_configs = Arc::clone(&train_motifs_nodes_run_configs);
                let loader = Arc::clone(&self.loader);
                let apply_motif_nets = Arc::clone(&self.apply_motif_nets);
                let select_motifs_run_ids = self.select_motifs_run_ids.clone();
                let labels_others_train_valid_resources = Arc::clone(&labels_others_train_valid_resources);

                tokio::spawn(async move {
                    let (label, resources_1) = label_tain_and_valid_sets;
                    debug!("TrainLabelNets::run: label={label} resources={resources_1:?}");

                    let resources_0 = labels_others_train_valid_resources.get(&label).expect(&format!("no resources not belonging to {label}"));
                    debug!("TrainLabelNets::run: label={label} resources_x={resources_0:?}");

                    let resources_1_train = Arc::new(resources_1.0);
                    let resources_1_valid = Arc::new(resources_1.1);
                    let resources_0_train = Arc::new(resources_0.0.clone());
                    let resources_0_valid = Arc::new(resources_0.1.clone());

                    //TODO async OOM
                    opt_train_2(
                        opt_config,

                    async move  |vs: &VarStore, run_config: TrainLabelNetsRunConfig, _listener| {
                            if ctx.read().acquired_permits() == 0 { //TODO move to with_data!
                                let result = GlobalCtx::global()
                                    .acquire_many(device_idx, 1, out_key.clone())
                                    .await?;
                                ctx.write().add_permit(result); //permits can be dropped only from OOM handler or when context is dropped
                            }
                            let apply_motif_nets = Arc::clone(&apply_motif_nets);

                            let resources_batch = ResourcesBatch::new(
                                Arc::clone(&loader),
                                run_config.common_training_args.batch_size,
                                Arc::clone(&resources_1_train),
                                Arc::clone(&resources_1_valid),
                                Arc::clone(&resources_0_train),
                                Arc::clone(&resources_0_valid),

                                Box::new(move |x_1_wrapped:Vec<Arc<DevicesWrap<ThoTensor<ChannelsData2, SimpleId>>>>,  x_0_wrapped:Vec<Arc<DevicesWrap<ThoTensor<ChannelsData2, SimpleId>>>>, ctx: Arc<RwLock<C>>|{
                                    let apply_motif_nets = Arc::clone(&apply_motif_nets);

                                    async move {
                                        let (x,y) = with_data!(ctx.read().device_idx(), [x_1_wrapped], [x_0_wrapped]; |x_1:&[&ThoTensor<ChannelsData2, SimpleId>], x_0:&[&ThoTensor<ChannelsData2, SimpleId>]|{
                                            debug!("ResourcesBatch::train_batch_iter: x_1={x_1:?} x_0={x_0:?}");

                                            let x_1 = Tensor::stack(&x_1.into_iter().map(|v|{
                                                Ok(pad_to_2(v.as_ref(), &[channels, max_input], true)?.copy())//TODO
                                            }).collect::<Result<Vec<_>, ThoosaError>>()?, 0);

                                            let x_0 = Tensor::stack(&x_0.into_iter().map(|v|{
                                                Ok(pad_to_2(v.as_ref(), &[channels, max_input], true)?.copy())
                                            }).collect::<Result<Vec<_>, ThoosaError>>()?, 0);

                                            Ok::<_, ThoosaError>(mix_x_binary_with_labels(&x_1, &x_0))
                                        })?;

                                        let x_batch:Arc<DevicesWrap<ThoTensor<Batch<ChannelsData2>, SimpleId>>> = Arc::new(DevicesWrap::wrap(ThoTensor::new("".into(), x.shallow_clone()), cache_area));
                                        let motif_probs = Arc::clone(&apply_motif_nets).run(
                                            x_batch,
                                            ctx
                                        ).await?;

                                        //apply_motif_nets is not caching so Arc can't have other refs
                                        let motifs_probs_batch = Arc::into_inner(motif_probs).expect("can't take arc").take_data(0).await?;
                                        debug!("ResourcesBatch::train_batch_iter: motifs_probs_batch={motifs_probs_batch:?} ");


                                        Ok((
                                            ThoTensor::<Batch<MotifProbability>, SimpleId>::new("".into(), Tensor::stack(motifs_probs_batch.matches.into_iter().map(|tt|tt.t).collect::<Vec<_>>().as_slice(), 1)),//motifs_probs_batch.matches contains [batch x probs] for each motif, stack(.., 1) - dim 1 makes [batch x motifs x dists]
                                            ThoTensor::<Batch<LabelProbability>, SimpleId>::new("".into(), y),
                                        ))
                                    }.boxed()
                                }),
                                Arc::clone(&ctx)
                            );

                            //TODO build proper LabelNetParams
                            let num_layers = match run_config.net {
                                LabelNetTypeArgs::MaxPadding(MaxPaddingNetArgs{num_layers}) => {
                                    num_layers
                                }
                                LabelNetTypeArgs::FixedNumSlices(FixedNumSlicesNetArgs{num_layers, ..}) => {
                                    num_layers
                                }
                            };

                            let params = LabelNetParams::new(channels as u32, num_nets, num_layers, slices_len, LabelNetTypeParams::FixedNumSlices(FixedNumSlicesNetParams::new()));
                            let loss = train_label_net_2(&run_config, &resources_batch, &ThoDevice::from(device_idx), vs, params.clone()).await?;

                            Ok((loss, params))
                        },
                        train_motifs_nodes_run_configs,
                        ThoDevice::from(device_idx).into(),
                        "labels",
                    ).await.map(|(var_store, label_node_params, train_motifs_nets_run_config, min_loss)| {

                        debug!("TrainLabelNet::run: label_node_params={label_node_params:?} train_motifs_nets_run_config={train_motifs_nets_run_config:?} min_loss={min_loss}");
                        TrainLabelNetsRunConfigResult::new(
                            select_motifs_run_ids,//TODO
                            label,
                            min_loss,
                            label_node_params,
                            train_motifs_nets_run_config,
                            var_store,

                        )
                    })
                })

            })
                .buffer_unordered(10)//TODO config
                .try_collect::<Vec<_>>().await?
                .into_iter()
                .collect::<Result<Vec<_>, ThoosaError>>()?;

            Ok(Arc::new(DevicesWrap::wrap_cpu(TrainLabelNetsOutput::new("".into(), results))))
        }
    }
}

/// builds map containing train and validation resources belonging to the label and another map with same number of resources not belonging to the label.
// TODO move to a util, can be used for any binary classification
impl<L, C> TrainLabelNets<L, C>
where
    C: 'static + Adjustable + MotifsCtx + RunCtx,
    L: 'static + Cached + OutId + for<'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C>,
{
    fn build_resource_maps(train_labels_resource_keys:  Vec<(String, Vec<Resource>)>, valid_label_resource_keys: Vec<(String, Vec<Resource>)>) -> (HashMap<String, (Vec<Resource>, Vec<Resource>)>, Arc<HashMap<String, (Vec<Resource>, Vec<Resource>)>>) {
        debug!("TrainLabelNets::run: train_labels_resource_keys={train_labels_resource_keys:?}");
        debug!("TrainLabelNets::run: valid_label_resource_keys={train_labels_resource_keys:?}");

        assert_eq!(train_labels_resource_keys.len(), valid_label_resource_keys.len());

        //should not be needed
        // train_labels_resource_keys.sort_by(|a, b|a.0.cmp(&b.0));
        // valid_label_resource_keys.sort_by(|a, b|a.0.cmp(&b.0));

        let mut labels_others_train_valid_resources: HashMap<String, (Vec<&Resource>, Vec<&Resource>)> = train_labels_resource_keys.iter().fold(HashMap::new(), |mut a, v| {
            assert!(a.insert(v.0.clone(), (vec![], vec![])).is_none());
            a
        });

        let mut num_resources = None::<(usize, usize)>;
        for (label, (train_resources, valid_resource)) in &labels_others_train_valid_resources {
            if let Some(num_resources) = num_resources.as_ref() {
                assert_eq!(num_resources.0, train_resources.len(), "uneven number of train resources in {labels_others_train_valid_resources:?}");
                assert_eq!(num_resources.1, valid_resource.len(), "uneven number of valid resources in {labels_others_train_valid_resources:?}");
            } else {
                num_resources = Some((train_resources.len(), valid_resource.len()));
            }
        }

        let labels_tain_and_valid_sets = train_labels_resource_keys.into_iter().zip(valid_label_resource_keys.into_iter()).fold(HashMap::new(), |mut a, (train, valid)| {
            assert_eq!(train.0, valid.0); //train and valid labels should come in same order from db

            assert!(a.insert(train.0, (train.1, valid.1)).is_none());
            a
        });


        for (other_label, train_valid_resources) in labels_others_train_valid_resources.iter_mut() {
            for (label, resources) in &labels_tain_and_valid_sets {
                if label.ne(other_label) {
                    train_valid_resources.0.extend(resources.0.iter());
                    train_valid_resources.1.extend(resources.1.iter());
                }
            }
        }

        //collecting resources not belonging to the label, shuffling and taking as many as in the label.
        let labels_others_train_valid_resources = Arc::new(labels_others_train_valid_resources.into_iter().map(|(label, resources)| {
            let mut train_resources = resources.0;
            train_resources.shuffle(&mut rng());
            train_resources.truncate(labels_tain_and_valid_sets.get(&label).expect(&format!("no {label} for train resources")).0.len());


            let mut valid_resources = resources.1;
            valid_resources.shuffle(&mut rng());
            valid_resources.truncate(labels_tain_and_valid_sets.get(&label).expect(&format!("no {label} for valid resources")).1.len());

            (
                label,
                (
                    train_resources.into_iter().map(|r| r.clone()).collect::<Vec<_>>(),
                    valid_resources.into_iter().map(|r| r.clone()).collect::<Vec<_>>()
                )
            )
        }).collect::<HashMap<String, (Vec<Resource>, Vec<Resource>)>>());
        (labels_tain_and_valid_sets, labels_others_train_valid_resources)
    }
}

type BuildBatchFn<
    LD:Dims+ 'static,
    X:Dims+ 'static,
    Y:Dims+ 'static,
    C
> = Box<dyn Fn(Vec<Arc<DevicesWrap<ThoTensor<LD, SimpleId>>>>,  Vec<Arc<DevicesWrap<ThoTensor<LD, SimpleId>>>>, Arc<RwLock<C>>) -> BoxFuture<'static, Result<(ThoTensor<Batch<X>, SimpleId>, ThoTensor<Batch<Y>, SimpleId>), ThoosaError>> + Send + Sync>;

// #[derive(Debug)]
struct ResourcesBatch<L, C, LD, X, Y>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<LD, SimpleId>, Ctx=C> + OutId + 'static,
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
    ctx: Arc<RwLock<C>>
}

impl <L, C, LD, X, Y> Debug for  ResourcesBatch<L, C, LD, X, Y>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<LD, SimpleId>, Ctx=C> + OutId + 'static,
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
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<LD, SimpleId>, Ctx=C> + OutId + 'static,
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
        ctx: Arc<RwLock<C>>
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
            ctx
        }
    }
}

#[async_trait]
impl <L,C, LD, X, Y> AsyncDataBatch<X, Y> for ResourcesBatch<L, C, LD, X, Y>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<LD, SimpleId>, Ctx=C> + OutId + 'static,
    C:RunCtx+'static,
    LD:Dims+ 'static,
    X:Dims+ 'static,
    Y:Dims+ 'static,
{
    async fn train_batch(&self) -> Result<(ThoBatch<X>, Tensor), ThoosaError> {
        todo!()
    }

    async fn valid_batch(&self) -> Result<(ThoBatch<X>, Tensor), ThoosaError> {
        todo!()
    }

    async fn train_batch_stream(&self) -> BoxStream<'_, Result<(ThoBatch<X>, ThoBatch<Y>), ThoosaError>> {
        //TODO if self.batch_size > train_len?
        let mut batches = shuffle_and_chunk_with_rng(self.resources_1_train.len(), self.half_batch_size, &mut rng());

        debug!("ResourcesBatch::train_batch_iter: batches={batches:?}");


        Self::build_batch_stream(
            &self.build_batch_fn,
            batches,
            Arc::clone(&self.loader),
            Arc::clone(&self.ctx),
            Arc::clone(&self.resources_1_train),
            Arc::clone(&self.resources_0_train)
        )
    }

    async fn valid_batch_stream(&self) -> BoxStream<'_, Result<(ThoBatch<X>, ThoBatch<Y>), ThoosaError>> {
        let mut batches = shuffle_and_chunk_with_rng(self.resources_1_valid.len(), self.half_batch_size, &mut rng());

        debug!("ResourcesBatch::train_batch_iter: batches={batches:?}");


        Self::build_batch_stream(
            &self.build_batch_fn,
            batches,
            Arc::clone(&self.loader),
            Arc::clone(&self.ctx),
            Arc::clone(&self.resources_1_valid),
            Arc::clone(&self.resources_0_valid)
        )
    }
}

impl<L, C, LD, X, Y> ResourcesBatch<L, C, LD, X, Y>
where
    C: 'static + RunCtx,
    L: 'static + OutId + for<'i> Cached<In<'i>=Resource, Out=ThoTensor<LD, SimpleId>, Ctx=C>+ 'static,
    LD:Dims+ 'static,
    X:Dims+ 'static,
    Y:Dims+ 'static,
{
    fn build_batch_stream<'a>(build_batch_fn:&'a BuildBatchFn<LD, X, Y, C>, mut batches: Vec<Box<[usize]>>, loader: Arc<L>, ctx: Arc<RwLock<C>>, x_1: Arc<Vec<Resource>>, x_0: Arc<Vec<Resource>>) -> BoxStream<'a, Result<(ThoBatch<X>, ThoBatch<Y>), ThoosaError>> {
        futures_util::stream::iter(batches)
            .map(move |batch_keys| {
                let loader = Arc::clone(&loader);
                let ctx = Arc::clone(&ctx);
                let x_1 = Arc::clone(&x_1);
                let x_0 = Arc::clone(&x_0);

                async move {
                    // load samples for this batch concurrently
                    let (x_1_wrapped, x_0_wrapped) = futures_util::stream::iter(batch_keys)
                        .map(|index| {
                            let loader = Arc::clone(&loader);
                            let x_1 = Arc::clone(&x_1);
                            let x_0 = Arc::clone(&x_0);
                            let ctx = Arc::clone(&ctx);

                            async move {
                                try_join!(
                                    Arc::clone(&loader).run_cached(Arc::new(DevicesWrap::wrap_cpu(x_1[index].clone())), Arc::clone(&ctx)),
                                    Arc::clone(&loader).run_cached(Arc::new(DevicesWrap::wrap_cpu(x_0[index].clone())), Arc::clone(&ctx))
                                )
                            }
                        })
                        .buffered(10)//TODO config
                        .try_collect::<Vec<_>>()
                        .await?
                        .into_iter()
                        .fold((vec![], vec![]), |mut a, v| {
                            a.0.push(v.0); //1
                            a.1.push(v.1); //0
                            a
                        });

                    build_batch_fn(x_1_wrapped, x_0_wrapped, ctx).await
                }
            })
            .buffered(10)
            .boxed()
    }

}

pub async fn train_label_net_2(
    run_config: &TrainLabelNetsRunConfig,
    batch: &dyn AsyncDataBatch<MotifProbability, LabelProbability>,
    device: &Device,
    vs: &VarStore, net_params: LabelNetParams,
) -> Result<f32, ThoosaError> {
    debug!("train_label_net:  ");

    let device = device.clone();

    let optimizer = AdamW::default().amsgrad(true);

    //TODO
    // if let Some(adam_w) = run_config.common_training_args.adam_w {
    //     debug!("train_label_nodes: setting adam_w={adam_w:?}");
    //     optimizer = optimizer.beta1(adam_w.beta1).beta2(adam_w.beta2);
    // }

    debug!("train_label_net: cuda_mem={:?}", cuda_mem());

    let loss = train_3(
        &device,
        batch,
        any_len_label_node(&net_params).build_tch_module::<Batch<MotifProbability>, Batch<LabelProbability>>(&vs.root())?,
        |pred: &Tensor, target: &Tensor| {
            debug!("train_label_net: pred.size={:?}", pred.size());
            let pred_y = pred;//.squeeze();
            debug!("train_label_net: pred_y={:?} target={:?} kind={:?}",pred_y.size(),target.size(),target.kind());
            trace!("train_label_net: pred_y=\n{pred_y} target=\n{target} kind={:?}",target.kind());
            pred_y.binary_cross_entropy::<Tensor>(&target, None, Reduction::Mean)
        },
        run_config.common_training_args.cycles as usize,
        optimizer.build(&vs, run_config.common_training_args.lr)?,
        |t| {
            debug!("train_label_net: t.size={:?} cuda_mem={:?}",t.size(),cuda_mem());
            t
        },
        None,
    )
        .await?;

    Ok(loss)
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use tokio::fs;
    use std::path::Path;
    use std::sync::Arc;
    use futures_util::StreamExt;
    use futures::FutureExt;
    use log::debug;
    use log::LevelFilter::{Debug, Trace};
    use parking_lot::RwLock;
    use tch::Device::Cpu;
    use tch::kind::FLOAT_CPU;
    use tch::Tensor;
    use caching_runner::common::cachable::DevicesWrap;
    use caching_runner::common::{mock_run_ctx, MockRunCtx, RunCtx};
    use caching_runner::common::runnable::AsyncStage;
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_common::batches::AsyncDataBatch;
    use thoosa_common::dims::{Batch, LabelProbability};
    use thoosa_common::runs::{Labeled, NetResources};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{CommonTrainingRunConfigFields, FixedNumSlicesNetArgsFields, LabelNetTypeArgsFields, OptConfig, TrainLabelNetsRunConfigFields};
    use thoosa_common::utils::{mix_x_binary_with_labels, reset_print_options};
    use thoosa_common::ws::{mock_ws_simp, WsMockBuilder};
    use thoosa_motifs::generators::noise::resource_tensor;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::{MotifKind, TensorIdSliceIdx, ThoosaError};
    use thoosa_shared::hyperparameter::Hyperparameter;
    use thoosa_shared::traits::SimpleId;
    use thoosa_shared::val::{NamedVal, Val};
    use thoosa_store::MockStoreBuilder;
    use thoosa_with_data::with_data;
    use crate::tasks::apply_motif_nets::tests::motif_test_net_;
    use crate::tasks::cross_dims::ChannelsData2;
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::select_motifs::tests::motifs_tensor;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use crate::tasks::train_label_nets::{ResourcesBatch, TrainLabelNets, TrainLabelNetsInput, TrainLabelNetsOutput};

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_train_label_net_3x3() -> Result<(), ThoosaError> {
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

        let pulse_motif = Arc::new(ThoTensor::new("motif_0_0".into(), motifs_tensor(100, 3, MotifKind::Pulse)));
        let triangle_motif = Arc::new(ThoTensor::new("motif_0_1".into(), motifs_tensor(100, 3, MotifKind::Triangle)));
        let sin_motif = Arc::new(ThoTensor::new("motif_1_0".into(), motifs_tensor(100, 3, MotifKind::Sin)));
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

        let resources_a = (0..10).map(|i|Resource::new(i, 1, format!("resource_a_{i}"), 700)).collect::<Vec<_>>();
        let resources_b = (10..20).map(|i|Resource::new(i, 1, format!("resource_b_{i}"), 800)).collect::<Vec<_>>();
        let resources_c = (20..30).map(|i|Resource::new(i, 1, format!("resource_c_{i}"), 900)).collect::<Vec<_>>();
        let resources_x = (30..40).map(|i|Resource::new(i, 1, format!("resource_x_{i}"), 1000)).collect::<Vec<_>>();

        let (mut resources_map, motif_intervals) = resources_a.iter().zip(resources_b.iter().zip(resources_c.iter().zip(resources_x.iter()))).try_fold((HashMap::new(), vec![]), |mut a, (r_a, (r_b, (r_c, r_x)))| {

            let (t, intervals) = resource_tensor(r_a.length() as usize, &[
                vec![(3, motif_width, MotifKind::Pulse)],
                vec![(1, motif_width, MotifKind::Sin)],
                vec![]
            ], 100)?;

            a.0.insert(r_a.clone(), t);

            a.1.extend(intervals[0][0].iter().map(|interval|TensorIdSliceIdx(r_a.id(), interval.start as u32)));


            let (t, intervals) = resource_tensor(r_b.length() as usize, &[
                vec![(3, motif_width, MotifKind::Triangle)],
                vec![(1, motif_width, MotifKind::Sin)],
                vec![]
            ], 100)?;

            a.0.insert(r_b.clone(), t);

            a.1.extend(intervals[0][0].iter().map(|interval|TensorIdSliceIdx(r_b.id(), interval.start as u32)));


            let (t, intervals) = resource_tensor(r_c.length() as usize, &[
                vec![(3, motif_width, MotifKind::Sin)],
                vec![(1, motif_width, MotifKind::Sin)],
                vec![]
            ], 100)?;

            a.0.insert(r_b.clone(), t);

            a.1.extend(intervals[0][0].iter().map(|interval|TensorIdSliceIdx(r_c.id(), interval.start as u32)));


            let (t, intervals) = resource_tensor(r_x.length() as usize, &[
                vec![],
                vec![],
                vec![]
            ], 100)?;

            a.0.insert(r_c.clone(), t);


            Ok::<_, ThoosaError>(a)
        })?;


        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached(resources_map)
                .finish(),
        );

        let mut store = Arc::new(RwLock::new(
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
            .build()));


        let run_configs = thoosa_shared::run_configs::OptRunConfigs::new(
            thoosa_shared::run_configs::GridRunConfigs::new(
                thoosa_shared::run_configs::ConstConfigs::new(vec![
                    NamedVal::new(TrainLabelNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Cycles), Val::Int(100)),
                    NamedVal::new(TrainLabelNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::BatchSize), Val::Int(8)),
                    NamedVal::new(TrainLabelNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::OptMaxEvals), Val::Int(10)),
                    NamedVal::new(TrainLabelNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::OptNInit), Val::Int(10)),
                    NamedVal::new(TrainLabelNetsRunConfigFields::Net(LabelNetTypeArgsFields::FixedNumSlices(FixedNumSlicesNetArgsFields::NumLayers)), Val::Int(3)),
                    NamedVal::new(TrainLabelNetsRunConfigFields::Net(LabelNetTypeArgsFields::FixedNumSlices(FixedNumSlicesNetArgsFields::SlicesLen)), Val::Int(10)),
                ]),
                vec![
                    // NamedRange::new(TrainLabelNetsRunConfigFields::NumLayers, Val::Int(0)..Val::Int(10), (Val::Int(10), 0))
                ],
            ),
            vec![
                Hyperparameter::UniformFloat{name: TrainLabelNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), lower: 0.0001, upper: 0.001}
            ],
        );

        let train_label_nets = Arc::new(TrainLabelNets::new(
            vec![select_motifs_run_id],
            loader,
            run_configs,
            &ws,
            &store,
            Arc::new(RwLock::new(mock_run_ctx::default(1))),
            Arc::new(OptConfig::new(10, 3))
        ).await?);


        let input = Arc::new(DevicesWrap::wrap(
            ResourcesConfigInput::new("test".into(), NetResources::new(Labeled::new(train_group.to_string()), Labeled::new("valid_group".to_string()))),
            0
        ));

        let out = train_label_nets.run(input, Arc::new(RwLock::new(mock_run_ctx::default(1)))).await?;

        with_data!(0, out; |out:&TrainLabelNetsOutput|{

            debug!("test_train_label_net_3x3: 0. out={out:?}");

            assert!(out.results[0].loss < 0.5);
            assert!(out.results[1].loss < 0.5);
            assert!(out.results[2].loss < 0.5);
            Ok::<_, ThoosaError>(())
        })?;

        debug!("test_train_label_net_3x3: out={out:?}");

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_resources_batch() -> Result<(), ThoosaError> {
        GlobalCtx::init_devices(2);

        let resources_1_train = Arc::new(vec![
            Resource::new(0, 1, "aaa_1_tr", 10),
            Resource::new(1, 1, "bbb_1_tr", 10),
            Resource::new(2, 1, "ccc_1_tr", 10)
        ]);
        let resources_1_valid = Arc::new(vec![
            Resource::new(3, 1, "aaa_1_vl", 10),
            Resource::new(4, 1, "bbb_1_vl", 10),
            Resource::new(4, 1, "ccc_1_vl", 10),
        ]);


        let resources_0_train = Arc::new(vec![
            Resource::new(5, 1, "aaa_0_tr", 10),
            Resource::new(6, 1, "bbb_0_tr", 10),
            Resource::new(7, 1, "ccc_0_tr", 10)
        ]);
        let resources_0_valid = Arc::new(vec![
            Resource::new(8, 1, "aaa_0_vl", 10),
            Resource::new(9, 1, "bbb_0_vl", 10),
            Resource::new(9, 1, "ccc_0_vl", 10),
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


        let resource_batch = ResourcesBatch::new(
            loader,
            6,
            resources_1_train,
            resources_1_valid,
            resources_0_train,
            resources_0_valid,
            Box::new(|x_1_wrapped:Vec<Arc<DevicesWrap<ThoTensor<ChannelsData2, SimpleId>>>>,  x_0_wrapped:Vec<Arc<DevicesWrap<ThoTensor<ChannelsData2, SimpleId>>>>, ctx: Arc<RwLock<MockRunCtx>>|{
                async move {
                    with_data!(ctx.read().device_idx(), [x_1_wrapped], [x_0_wrapped]; |x_1:&[&ThoTensor<ChannelsData2, SimpleId>], x_0:&[&ThoTensor<ChannelsData2, SimpleId>]|{
                        debug!("ResourcesBatch::train_batch_iter: x_1={x_1:?} x_0={x_0:?}");
                        let x_1 = Tensor::stack(&x_1.into_iter().map(|v|v.as_ref().copy()).collect::<Vec<_>>(), 0);
                        let x_0 = Tensor::stack(&x_0.into_iter().map(|v|v.as_ref().copy()).collect::<Vec<_>>(), 0);

                        let (x,y) = mix_x_binary_with_labels(&x_1, &x_0);

                         Ok((
                            ThoTensor::<Batch<ChannelsData2>, SimpleId>::new("".into(), x),
                            ThoTensor::<Batch<LabelProbability>, SimpleId>::new("".into(), y),
                        ))
                    })
                }.boxed()
            }),
            Arc::new(RwLock::new(mock_run_ctx::default(1)))
        );

        let mut batch = resource_batch.train_batch_stream().await;
        let (batch_x, batch_y) = batch.next().await.expect("no batch").expect("batch failed");
        assert!(batch.next().await.is_none()); //should be only one

        let mut batch = resource_batch.valid_batch_stream().await;
        let (batch_x, batch_y) = batch.next().await.expect("no batch").expect("batch failed");
        assert!(batch.next().await.is_none()); //should be only one



        Ok(())
    }
}