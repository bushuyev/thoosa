use crate::tasks::apply_label_nets::ApplyLabelNets;
use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::stages_seq::ABStage;
use crate::tasks::train_net::{TrainNet, TrainNetInput};
use async_trait::async_trait;
use futures::FutureExt;
use futures_util::stream::BoxStream;
use futures_util::{StreamExt, TryStreamExt};
use log::{debug, trace};
use parking_lot::{Mutex, RwLock};
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::future::ready;
use std::marker::PhantomData;
use std::sync::Arc;
use tch::nn::{AdamW, OptimizerConfig, VarStore};
use tch::{Reduction, Tensor};
use thoosa_common::batches::{AsyncDataBatch, BatchLengths};
use thoosa_common::dims::{Batch, LabelProbability, LabelsProbabilities, ThoBatch};
use thoosa_common::runs::{Logits, NetResources, SavableResult};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::final_net;
use thoosa_common::train_nets::{FinalNetParams, OptConfig, TrainFinalNetRunConfig, TrainFinalNetRunConfigFields, TrainFinalNetsRunConfigResult};
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_shared::run_configs::{OptRunConfigs, RunId};
use thoosa_shared::runner::cache_areas::CacheArea;
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage};
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::traits::{Adjustable, Data, WithId};
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_shared::{mono_key, shuffle_and_chunk_with_rng, ThoError};
use thoosa_store::Store;
use thoosa_with_data::with_data;

#[derive(Debug)]
pub struct TrainFinalNet<L, LL, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    LL: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<LabelProbability, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    train_label_nets_run_ids: Vec<RunId>,
    run_configs: Arc<OptRunConfigs<Vec<NamedVal<TrainFinalNetRunConfigFields>>, Vec<NamedRange<TrainFinalNetRunConfigFields>>, Vec<Hyperparameter<TrainFinalNetRunConfigFields>>, TrainFinalNetRunConfigFields>>,
    pd: PhantomData<L>,
    label_loader: Arc<LL>,
    num_label_nets: u32,
    label_net_run_ids: Vec<RunId>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<RwLock<Box<dyn Store>>>,
    opt_config: Arc<OptConfig>,
}

pub async fn new_train_final_net_with_ml<L, C, R>(
    train_label_nets_run_ids: Vec<RunId>,
    loader: Arc<L>,
    run_configs: OptRunConfigs<Vec<NamedVal<TrainFinalNetRunConfigFields>>, Vec<NamedRange<TrainFinalNetRunConfigFields>>, Vec<Hyperparameter<TrainFinalNetRunConfigFields>>, TrainFinalNetRunConfigFields>,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    stage_ctx: R,
    opt_config: Arc<OptConfig>,
) -> Result<TrainFinalNet<L, ABStage<L, ApplyLabelNets<C>, C>, C>, ThoError>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
    R: RunCtx + Adjustable + 'static,
{
    let apply_label_nets = Arc::new(ApplyLabelNets::new(ws, store, train_label_nets_run_ids.clone(), stage_ctx).await?);

    let label_loader = Arc::new(ABStage::new(
        SynKey::new([mono_key!("Loader_with_label_nets", "Loader_with_label_nets"), mono_key!(format!("{train_label_nets_run_ids:?}"), "train_label_nets_run_ids")]),
        Arc::clone(&loader),
        Arc::clone(&apply_label_nets),
    ));

    TrainFinalNet::new(train_label_nets_run_ids, label_loader, apply_label_nets.num_nets() as u32, run_configs, ws, store, opt_config).await
}

impl<L, LL, C> TrainFinalNet<L, LL, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    LL: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<LabelProbability, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    pub async fn new(
        train_label_nets_run_ids: Vec<RunId>,
        label_loader: Arc<LL>,
        num_label_nets: u32,
        run_configs: OptRunConfigs<Vec<NamedVal<TrainFinalNetRunConfigFields>>, Vec<NamedRange<TrainFinalNetRunConfigFields>>, Vec<Hyperparameter<TrainFinalNetRunConfigFields>>, TrainFinalNetRunConfigFields>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        opt_config: Arc<OptConfig>,
    ) -> Result<Self, ThoError> {
        let label_net_run_ids = {
            let mut store_guard = store.write();
            let mut run_ids = HashSet::new();
            for run_id in &train_label_nets_run_ids {
                run_ids.extend(store_guard.list_output_run_ids(*run_id)?);
            }
            let mut run_ids = run_ids.into_iter().collect::<Vec<_>>();
            run_ids.sort();
            run_ids
        };
        if label_net_run_ids.is_empty() {
            return Err(ThoError::new("TrainFinalNet::new: no label nets found"));
        }

        Ok(Self {
            train_label_nets_run_ids,
            run_configs: Arc::new(run_configs),
            pd: PhantomData,
            label_loader,
            num_label_nets,
            label_net_run_ids,
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

impl SavableResult for TrainFinalNetOutput {
    fn value(&self) -> f32 {
        self.result.value()
    }

    fn save(&mut self, store: &Arc<RwLock<Box<dyn Store>>>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<u32, ThoError> {
        self.result.save(store, ws)
    }
}

impl<L, LL, C> Stage for TrainFinalNet<L, LL, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    LL: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<LabelProbability, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = ResourcesConfigInput<NetResources<Logits>>;
    type Out = TrainFinalNetOutput;
}

impl<L, LL, C> AsyncStage for TrainFinalNet<L, LL, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    LL: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<LabelProbability, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "TrainFinalNet::run"))]
    fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> impl Future<Output = Result<Arc<DevicesWrap<Self::Out>>, ThoError>> + Send {
        let cache_area = ctx.cache_area();

        let train_final_run_configs = Arc::clone(&self.run_configs);
        let label_loader = Arc::clone(&self.label_loader);
        let label_net_run_ids = self.label_net_run_ids.clone();
        let num_label_nets = self.num_label_nets;
        let opt_config = Arc::clone(&self.opt_config);
        let store = Arc::clone(&self.store);

        async move {
            let (train_resources, valid_resources): (Vec<(Resource, Vec<f32>)>, Vec<(Resource, Vec<f32>)>) = with_data!(ctx.branch(Some("train_final_resources"), false, false), input; |_ctx, input:&ResourcesConfigInput<NetResources<Logits>>|{
                Ok::<_, ThoError>(input.resources_config().to_resources(&store)?)
            })?;

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
                move |ctx: C, rc: &TrainFinalNetRunConfig| {
                    let label_loader = Arc::clone(&label_loader);
                    let train_resources = Arc::clone(&train_resources);
                    let valid_resources = Arc::clone(&valid_resources);

                    ready(LogitsResourcesBatch::new(label_loader, rc.common_training_args.batch_size, train_resources, valid_resources, labels_len, ctx)).boxed()
                },
                move |rc: &TrainFinalNetRunConfig, vs: &VarStore| {
                    let net_params = FinalNetParams::new(num_label_nets, labels_len as u32, rc.net.num_layers);
                    Ok((final_net(&net_params)?.build_tch_module::<Batch<LabelProbability>, Batch<LabelsProbabilities>>(&vs.root())?, net_params))
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
                |rc: &TrainFinalNetRunConfig| rc.common_training_args.cycles,
                "final".to_string(),
            ));

            let result = Arc::into_inner(train_net.run(Arc::new(DevicesWrap::wrap(TrainNetInput::new(SynKey::new([mono_key!("", "empty")])), cache_area.into())), ctx.branch(Some("train_final_net"), false, false)).await?)
                .unwrap()
                .take_data(0)
                .await?;

            let final_result = TrainFinalNetsRunConfigResult::new(Some(&label_net_run_ids), result.min_loss, result.var_store, result.node_params, result.run_config);
            debug!("TrainFinalNet::run: final_result={final_result:?}");

            Ok(Arc::new(DevicesWrap::wrap(TrainFinalNetOutput::new(SynKey::new([mono_key!("", "empty")]), final_result), cache_area.into())))
        }
    }
}

struct LogitsResourcesBatch<L, C>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<LabelProbability, SynKey>, Ctx = C> + OutId + 'static,
    C: RunCtx + 'static,
{
    batch_size: u32,
    train_resources: Arc<Vec<(Resource, Vec<f32>)>>,
    valid_resources: Arc<Vec<(Resource, Vec<f32>)>>,
    loader: Arc<L>,
    labels_len: usize,
    ctx: Mutex<C>,
}

impl<L, C> Debug for LogitsResourcesBatch<L, C>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<LabelProbability, SynKey>, Ctx = C> + OutId + 'static,
    C: RunCtx + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogitsResourcesBatch").finish()
    }
}

impl<L, C> LogitsResourcesBatch<L, C>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<LabelProbability, SynKey>, Ctx = C> + OutId + 'static,
    C: RunCtx + 'static,
{
    pub fn new(loader: Arc<L>, batch_size: u32, train_resources: Arc<Vec<(Resource, Vec<f32>)>>, valid_resources: Arc<Vec<(Resource, Vec<f32>)>>, labels_len: usize, ctx: C) -> Result<Self, ThoError> {
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
            ctx: Mutex::new(ctx),
        })
    }

    fn build_batch_stream<'a>(loader: Arc<L>, ctx: &'a Mutex<C>, resources: Arc<Vec<(Resource, Vec<f32>)>>, batch_size: u32, labels_len: usize) -> BoxStream<'a, Result<(ThoBatch<LabelProbability>, Option<BatchLengths>, ThoBatch<LabelsProbabilities>), ThoError>> {
        let batches = shuffle_and_chunk_with_rng(resources.len(), batch_size, &mut rand::rng());

        futures_util::stream::iter(batches)
            .map(move |batch_keys| {
                let loader = Arc::clone(&loader);
                let resources = Arc::clone(&resources);
                async move {
                    let (outputs, labels) = futures_util::stream::iter(batch_keys)
                        .map(|index| {
                            let loader = Arc::clone(&loader);
                            let (resource, labels) = resources[index].clone();
                            let ctx_for_loader = {
                                let mut guard = ctx.lock();
                                guard.branch(Some("final_net_batch_loader"), false, false)
                            };

                            async move {
                                let out = Arc::clone(&loader).run_cached(Arc::new(DevicesWrap::wrap(resource, CacheArea::default())), ctx_for_loader).await?;
                                Ok::<_, ThoError>((out, labels))
                            }
                        })
                        .buffered(crate::buffer_limits::LOGITS_RESOURCES_BATCH_LOAD_BUFFERED_LIMIT)
                        .try_collect::<Vec<_>>()
                        .await?
                        .into_iter()
                        .fold((vec![], vec![]), |mut a, v| {
                            a.0.push(v.0);
                            a.1.push(v.1);
                            a
                        });

                    let (device_idx, ctx_for_build) = {
                        let mut guard = ctx.lock();
                        (guard.device_idx(), guard.branch(Some("final_net_batch_build"), false, false))
                    };

                    with_data!(ctx_for_build, [outputs]; |_ctx, xs:&[&ThoTensor<LabelProbability, SynKey>]|{
                        let x = Tensor::f_stack(
                            xs.iter()
                                .map(|x| x.t_ref().f_flatten(0, 1))
                                .collect::<Result<Vec<_>, _>>()?
                                .as_slice(),
                            0,
                        )?;
                        let device = ThoDevice::from(device_idx).into();
                        let y = Tensor::f_stack(
                            labels.iter()
                                .map(|labels| {
                                    if labels.len() != labels_len {
                                        return Err(ThoError::new("LogitsResourcesBatch::build: bad label size"));
                                    }
                                    let t = Tensor::f_from_slice(labels.as_slice())?;
                                    Ok(t.f_to_device(device)?)
                                })
                                .collect::<Result<Vec<_>, _>>()?
                                .as_slice(),
                            0,
                        )?;

                        debug!("LogitsResourcesBatch::build_batch_stream: x=\n{x} y=\n{y}");

                        Ok::<_, ThoError>((x.into(), None, y.into()))
                    })
                }
            })
            .buffer_unordered(crate::buffer_limits::LOGITS_RESOURCES_BATCH_STREAM_BUFFER_UNORDERED_LIMIT)
            .boxed()
    }
}

#[async_trait]
impl<L, C> AsyncDataBatch<LabelProbability, LabelsProbabilities> for LogitsResourcesBatch<L, C>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<LabelProbability, SynKey>, Ctx = C> + OutId + 'static,
    C: RunCtx + 'static,
{
    async fn train_batch_stream(&self) -> BoxStream<'_, Result<(ThoBatch<LabelProbability>, Option<BatchLengths>, ThoBatch<LabelsProbabilities>), ThoError>> {
        Self::build_batch_stream(Arc::clone(&self.loader), &self.ctx, Arc::clone(&self.train_resources), self.batch_size, self.labels_len)
    }

    async fn valid_batch_stream(&self) -> BoxStream<'_, Result<(ThoBatch<LabelProbability>, Option<BatchLengths>, ThoBatch<LabelsProbabilities>), ThoError>> {
        Self::build_batch_stream(Arc::clone(&self.loader), &self.ctx, Arc::clone(&self.valid_resources), self.batch_size, self.labels_len)
    }
}

#[cfg(test)]
mod tests {
    use crate::tasks::apply_motif_nets::motif_test_net_;
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use crate::tasks::train_final_net::{new_train_final_net_with_ml, TrainFinalNetOutput};
    use log::debug;
    use log::LevelFilter::Debug;
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tch::nn::{self, VarStore};
    use tch::{Device, Kind, Tensor};
    use thoosa_common::dims::{Batch, LabelProbability, MotifsNetsSelectionDims, Slice};
    use thoosa_common::my_tch::{MyClosureModule, ThoModule};
    use thoosa_common::runs::{Logits, NetResources};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{LinLenNetParams, ConvNetParams, LabelNetParams, LabelNetTypeParams, ThoModuleWithStore};
    use thoosa_common::train_nets::{CommonTrainingRunConfigFields, FinalNetArgsFields, OptConfig, TrainFinalNetRunConfigFields};
    use thoosa_common::utils::reset_print_options;
    use thoosa_common::ws::WsMockBuilder;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::hyperparameter::Hyperparameter;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::OptRunConfigs;
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::runner::cache_areas::CacheArea;
    use thoosa_shared::runner::common::cachable::DevicesWrap;
    use thoosa_shared::runner::common::runnable::AsyncStage;
    use thoosa_shared::runner::common::{mock_run_ctx, RunCtx};
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_shared::val::NamedVal;
    use thoosa_shared::val::Val;
    use thoosa_shared::{mono_key, ThoError};
    use thoosa_store::MockStoreBuilder;
    use thoosa_with_data::with_data;

    fn const_label_net(motif_nets: u32, slices_len: u32, k: f64, device: Device) -> (Box<dyn Fn() -> LabelNetParams + Sync + Send + 'static>, Box<dyn Fn() -> ThoModuleWithStore<MotifsNetsSelectionDims, Batch<LabelProbability>> + Sync + Send + 'static>) {
        (
            Box::new(move || LabelNetParams::new(motif_nets, LabelNetTypeParams::LinNetParams(LinLenNetParams::new(slices_len, 1)))),
            Box::new(move || {
                let k = k;
                let builder = Arc::new(move |_path: &nn::Path| {
                    let module = MyClosureModule::new(Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&[i64]>| {
                        let sum = t.f_sum_dim_intlist(&[1, 2][..], false, None)?;
                        let mut sum = sum.f_unsqueeze(1)?;
                        let out = sum.f_mul_scalar_(k)?;
                        Ok(out)
                    })));
                    Ok(Box::new(module) as Box<dyn ThoModule<MotifsNetsSelectionDims, Batch<LabelProbability>>>)
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

        let train_motif_nets_run_id: RunId = 2;
        let train_label_nets_run_id: RunId = 3;
        let motif_width = 4;
        let step = 1;
        let device = Device::Cpu;
        let train_group = "train_group";
        let valid_group = "valid_group";
        let motif_t_0 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_0", "motif_id")]), Tensor::f_ones(&[1, motif_width as i64], (Kind::Float, device))?)?);
        let motif_t_1 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_1", "motif_id")]), Tensor::f_ones(&[1, motif_width as i64], (Kind::Float, device))?)?);

        let (motif_0_params, motif_0_net) = motif_test_net_(device, Arc::clone(&motif_t_0), step);
        let (motif_1_params, motif_1_net) = motif_test_net_(device, Arc::clone(&motif_t_1), step);
        let net_id_0 = PathBuf::from("0/0_2");
        let net_id_1 = PathBuf::from("0/1_2");

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 1, 100, 1000)
                .with_load_motif_net(vec![
                    (net_id_0.clone(), motif_0_params, motif_0_net),
                    (net_id_1.clone(), motif_1_params, motif_1_net),
                ])
                .with_label_nets(HashMap::from([
                    ("A".to_string(), vec![(100, const_label_net(2, 1, 0.1, device).0, const_label_net(2, 1, 0.1, device).1)]),
                    ("B".to_string(), vec![(200, const_label_net(2, 1, 0.2, device).0, const_label_net(2, 1, 0.2, device).1)]),
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

        let store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
                .with_get_input_run_ids([(train_label_nets_run_id, Ok(vec![train_motif_nets_run_id]))])
                .with_list_selected_nets([
                    (
                        train_motif_nets_run_id,
                        Some(0.3),
                        Ok(vec![
                            (net_id_0, 0.1f32),
                            (net_id_1, 0.2f32),
                        ]),
                    ),
                ])
                .with_list_output_run_ids([(train_label_nets_run_id, Ok(vec![100, 200]))])
                .with_list_resource_random_in_group([(train_group, Ok(train_logits)), (valid_group, Ok(valid_logits))])
                .build(),
        ));

        let run_configs = OptRunConfigs::new(
            thoosa_shared::run_configs::GridRunConfigs::new(
                thoosa_shared::run_configs::ConstConfigs::new(vec![
                    NamedVal::new(TrainFinalNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Cycles), Val::Int(2)),
                    NamedVal::new(TrainFinalNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::BatchSize), Val::Int(2)),
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
        let train_final_net = Arc::new(new_train_final_net_with_ml(vec![train_label_nets_run_id], loader, run_configs, &ws, &store, run_ctx.branch(Some("new_train_final_net_with_ml"), false, false), Arc::new(OptConfig::new(10, 3))).await?);

        let input = Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(NetResources::new(Logits::new(train_group.to_string()), Logits::new(valid_group.to_string()))), CacheArea::default()));

        let r = train_final_net.run(input, run_ctx.branch(Some("train_final_net"), false, false)).await?;
        with_data!(run_ctx.branch(Some("train_final_nets_run"), false, false), r; |_ctx, r:&TrainFinalNetOutput|{

            debug!("test_train_final_net_smoke: result={r:?}");

            assert!(r.result.loss < 0.2);

            Ok::<_, ThoError>(())
        })?;

        Ok(())
    }
}
