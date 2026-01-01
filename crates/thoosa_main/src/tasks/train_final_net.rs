use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::future::ready;
use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use futures::FutureExt;
use futures_util::{StreamExt, TryStreamExt};
use futures_util::stream::BoxStream;
use log::{debug, trace};
use parking_lot::RwLock;
use tch::nn::{AdamW, OptimizerConfig, VarStore};
use tch::{Reduction, Tensor};
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage};
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_common::batches::AsyncDataBatch;
use thoosa_common::dims::{Batch, LabelProbability, LabelsProbabilities, ThoBatch};
use thoosa_common::runs::{Logits, NetResources, SavableResult};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::{FinalNetParams, OptConfig, TrainFinalNetRunConfig, TrainFinalNetRunConfigFields, TrainFinalNetsRunConfigResult};
use thoosa_common::train_nets::final_net;
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_shared::run_configs::{OptRunConfigs, RunId};
use thoosa_shared::traits::{Adjustable, Data, WithId};
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_shared::{mono_key, shuffle_and_chunk_with_rng, ThoosaError};
use thoosa_shared::comp_id::SynKey;
use thoosa_store::Store;
use thoosa_with_data::with_data;
use crate::tasks::apply_label_nets::ApplyLabelNets;
use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::run_ctx_2::MotifsCtx;
use crate::tasks::stages_seq::ABStage;
use crate::tasks::train_net::{TrainNet, TrainNetInput};

#[derive(Debug)]
pub struct TrainFinalNet<L, LL, C>
where
    L: for<'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    LL: for<'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<LabelProbability, SynKey>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static,
{
    select_motifs_run_ids: Vec<RunId>,
    run_configs: Arc<OptRunConfigs<
        Vec<NamedVal<TrainFinalNetRunConfigFields>>,
        Vec<NamedRange<TrainFinalNetRunConfigFields>>,
        Vec<Hyperparameter<TrainFinalNetRunConfigFields>>,
        TrainFinalNetRunConfigFields,
    >>,
    pd: PhantomData<L>,
    label_loader: Arc<LL>,
    num_label_nets: u32,
    label_net_run_ids: Vec<RunId>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<RwLock<Box<dyn Store>>>,
    opt_config: Arc<OptConfig>,
}

pub async fn new_train_final_net_with_ml<L, C, R>(
    select_motifs_run_ids: Vec<RunId>,
    loader: Arc<L>,
    run_configs: OptRunConfigs<Vec<NamedVal<TrainFinalNetRunConfigFields>>, Vec<NamedRange<TrainFinalNetRunConfigFields>>, Vec<Hyperparameter<TrainFinalNetRunConfigFields>>, TrainFinalNetRunConfigFields>,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    stage_ctx: Arc<RwLock<R>>,
    opt_config: Arc<OptConfig>,
) -> Result<TrainFinalNet<L, ABStage<L, ApplyLabelNets<C>, C>, C>, ThoosaError>
where
    L: for<'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    R: RunCtx + MotifsCtx + Adjustable + 'static,
{
    let apply_label_nets = Arc::new(ApplyLabelNets::new(ws, store, select_motifs_run_ids.clone(), stage_ctx).await?);

    let label_loader = Arc::new(ABStage::new(
        SynKey::new([
            mono_key!("Loader_with_label_nets", "stage"),
            mono_key!(format!("{select_motifs_run_ids:?}"), "select_motifs_run_ids"),
        ]),
        Arc::clone(&loader),
        Arc::clone(&apply_label_nets),
    ));

    TrainFinalNet::new(
        select_motifs_run_ids,
        label_loader,
        apply_label_nets.num_nets() as u32,
        run_configs,
        ws,
        store,
        opt_config,
    )
    .await
}

impl<L, LL, C> TrainFinalNet<L, LL, C>
where
    L: for<'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    LL: for<'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<LabelProbability, SynKey>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static,
{
    pub async fn new(
        select_motifs_run_ids: Vec<RunId>,
        label_loader: Arc<LL>,
        num_label_nets: u32,
        run_configs: OptRunConfigs<Vec<NamedVal<TrainFinalNetRunConfigFields>>, Vec<NamedRange<TrainFinalNetRunConfigFields>>, Vec<Hyperparameter<TrainFinalNetRunConfigFields>>, TrainFinalNetRunConfigFields>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        opt_config: Arc<OptConfig>,
    ) -> Result<Self, ThoosaError> {
        let label_net_run_ids = {
            let mut store_guard = store.write();
            let mut run_ids = HashSet::new();
            for run_id in &select_motifs_run_ids {
                run_ids.extend(store_guard.list_output_run_ids(*run_id)?);
            }
            let mut run_ids = run_ids.into_iter().collect::<Vec<_>>();
            run_ids.sort();
            run_ids
        };
        if label_net_run_ids.is_empty() {
            return Err(ThoosaError::new("TrainFinalNet::new: no label nets found"));
        }

        Ok(Self {
            select_motifs_run_ids,
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

    fn save(
        &mut self,
        store: &Arc<RwLock<Box<dyn Store>>>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
    ) -> Result<u32, ThoosaError> {
        self.result.save(store, ws)
    }
}

impl<L, LL, C> Stage for TrainFinalNet<L, LL, C>
where
    L: for<'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    LL: for<'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<LabelProbability, SynKey>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = ResourcesConfigInput<NetResources<Logits>>;
    type Out = TrainFinalNetOutput;
}

impl<L, LL, C> AsyncStage for TrainFinalNet<L, LL, C>
where
    L: for<'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    LL: for<'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<LabelProbability, SynKey>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainFinalNet::run"))]
    fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        ctx: Arc<RwLock<Self::Ctx>>,
    ) -> impl Future<Output=Result<Arc<DevicesWrap<Self::Out>>, ThoosaError>> + Send {
        let (device_idx, cache_area) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area())
        };

        let train_final_run_configs = Arc::clone(&self.run_configs);
        let label_loader = Arc::clone(&self.label_loader);
        let label_net_run_ids = self.label_net_run_ids.clone();
        let num_label_nets = self.num_label_nets;
        let opt_config = Arc::clone(&self.opt_config);
        let ctx_for_batches = Arc::clone(&ctx);
        let store = Arc::clone(&self.store);
        let cache_area = cache_area.clone();

        async move {
            let (train_resources, valid_resources): (Vec<(Resource, Vec<f32>)>, Vec<(Resource, Vec<f32>)>) = with_data!(ctx, 0, input; |input:&ResourcesConfigInput<NetResources<Logits>>|{
                Ok::<_, ThoosaError>(input.resources_config().to_resources(&store)?)
            })?;

            let labels_len = train_resources
                .first()
                .ok_or_else(|| ThoosaError::new("TrainFinalNet::run: no train resources"))?
                .1
                .len();
            if labels_len == 0 {
                return Err(ThoosaError::new("TrainFinalNet::run: empty label vector"));
            }
            for (_res, labels) in train_resources.iter().chain(valid_resources.iter()) {
                if labels.len() != labels_len {
                    return Err(ThoosaError::new("TrainFinalNet::run: inconsistent label vector sizes"));
                }
            }

            let train_resources = Arc::new(train_resources);
            let valid_resources = Arc::new(valid_resources);

            let train_net = Arc::new(TrainNet::new(
                move |rc:&TrainFinalNetRunConfig| {
                    let label_loader = Arc::clone(&label_loader);
                    let train_resources = Arc::clone(&train_resources);
                    let valid_resources = Arc::clone(&valid_resources);
                    let ctx = Arc::clone(&ctx_for_batches);
                    ready(LogitsResourcesBatch::new(
                        label_loader,
                        rc.common_training_args.batch_size,
                        train_resources,
                        valid_resources,
                        labels_len,
                        ctx,
                    )).boxed()
                },
                move |rc:&TrainFinalNetRunConfig, vs: &VarStore| {
                    let net_params = FinalNetParams::new(num_label_nets, labels_len as u32, rc.net.num_layers);
                    Ok((final_net(&net_params).build_tch_module::<Batch<LabelProbability>, Batch<LabelsProbabilities>>(&vs.root())?, net_params))
                },
                opt_config,
                train_final_run_configs,
                |pred: &Tensor, target: &Tensor| {
                    trace!("train_final_net: pred.size={:?} target.size={:?}", pred.size(), target.size());
                    Ok(pred.binary_cross_entropy_with_logits::<Tensor>(target, None, None, Reduction::Mean).set_requires_grad(true))
                },
                move |rc:&TrainFinalNetRunConfig, vs: &VarStore|{
                    AdamW::default().amsgrad(true).build(&vs, rc.common_training_args.lr).map_err(|e|e.into())
                },
                |t:ThoBatch<LabelProbability>| t,
                |rc:&TrainFinalNetRunConfig| rc.common_training_args.cycles,
                "final".to_string()
            ));

            let result = Arc::into_inner(train_net.run(Arc::new(DevicesWrap::wrap(TrainNetInput::new(SynKey::new([mono_key!("", "empty")])) , cache_area)), Arc::clone(&ctx)).await?).unwrap().take_data(0).await?;

            let final_result = TrainFinalNetsRunConfigResult::new(
                Some(&label_net_run_ids),
                result.min_loss,
                result.var_store,
                result.node_params,
                result.run_config,
            );

            Ok(Arc::new(DevicesWrap::wrap(
                TrainFinalNetOutput::new(SynKey::new([mono_key!("", "empty")]), final_result),
                cache_area,
            )))
        }
    }
}

struct LogitsResourcesBatch<L, C>
where
    L: for<'i> Cached<In<'i>=Resource, Out=ThoTensor<LabelProbability, SynKey>, Ctx=C> + OutId + 'static,
    C: RunCtx + 'static,
{
    batch_size: u32,
    train_resources: Arc<Vec<(Resource, Vec<f32>)>>,
    valid_resources: Arc<Vec<(Resource, Vec<f32>)>>,
    loader: Arc<L>,
    labels_len: usize,
    ctx: Arc<RwLock<C>>,
}

impl<L, C> Debug for LogitsResourcesBatch<L, C>
where
    L: for<'i> Cached<In<'i>=Resource, Out=ThoTensor<LabelProbability, SynKey>, Ctx=C> + OutId + 'static,
    C: RunCtx + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogitsResourcesBatch").finish()
    }
}

impl<L, C> LogitsResourcesBatch<L, C>
where
    L: for<'i> Cached<In<'i>=Resource, Out=ThoTensor<LabelProbability, SynKey>, Ctx=C> + OutId + 'static,
    C: RunCtx + 'static,
{
    pub fn new(
        loader: Arc<L>,
        batch_size: u32,
        train_resources: Arc<Vec<(Resource, Vec<f32>)>>,
        valid_resources: Arc<Vec<(Resource, Vec<f32>)>>,
        labels_len: usize,
        ctx: Arc<RwLock<C>>,
    ) -> Result<Self, ThoosaError> {
        if train_resources.is_empty() || valid_resources.is_empty() {
            return Err(ThoosaError::new("LogitsResourcesBatch::new: empty resources"));
        }
        if batch_size == 0 {
            return Err(ThoosaError::new("LogitsResourcesBatch::new: batch size is 0"));
        }
        if batch_size as usize > train_resources.len() || batch_size as usize > valid_resources.len() {
            return Err(ThoosaError::new("LogitsResourcesBatch::new: batch size exceeds resources"));
        }
        Ok(Self {
            batch_size,
            train_resources,
            valid_resources,
            loader,
            labels_len,
            ctx,
        })
    }

    fn build_batch_stream<'a>(
        loader: Arc<L>,
        ctx: Arc<RwLock<C>>,
        resources: Arc<Vec<(Resource, Vec<f32>)>>,
        batch_size: u32,
        labels_len: usize,
    ) -> BoxStream<'a, Result<(ThoBatch<LabelProbability>, ThoBatch<LabelsProbabilities>), ThoosaError>> {
        let batches = shuffle_and_chunk_with_rng(resources.len(), batch_size, &mut rand::rng());

        futures_util::stream::iter(batches)
            .map(move |batch_keys| {
                let loader = Arc::clone(&loader);
                let ctx = Arc::clone(&ctx);
                let resources = Arc::clone(&resources);
                async move {
                    let (outputs, labels) = futures_util::stream::iter(batch_keys)
                        .map(|index| {
                            let loader = Arc::clone(&loader);
                            let ctx = Arc::clone(&ctx);
                            let (resource, labels) = resources[index].clone();
                            let ctx_for_loader = ctx.write().branch("final_net_batch_loader");
                            async move {
                                let out = Arc::clone(&loader)
                                    .run_cached(Arc::new(DevicesWrap::wrap(resource, 0)), ctx_for_loader)
                                    .await?;
                                Ok::<_, ThoosaError>((out, labels))
                            }
                        })
                        .buffered(10)
                        .try_collect::<Vec<_>>()
                        .await?
                        .into_iter()
                        .fold((vec![], vec![]), |mut a, v| {
                            a.0.push(v.0);
                            a.1.push(v.1);
                            a
                        });

                    let device_idx = ctx.read().device_idx();
                    let ctx_for_build = ctx.write().branch("final_net_batch_build");
                    with_data!(Arc::clone(&ctx_for_build), device_idx, [outputs]; |xs:&[&ThoTensor<LabelProbability, SynKey>]|{
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
                                        return Err(ThoosaError::new("LogitsResourcesBatch::build: bad label size"));
                                    }
                                    let t = Tensor::f_from_slice(labels.as_slice())?;
                                    Ok(t.f_to_device(device)?)
                                })
                                .collect::<Result<Vec<_>, _>>()?
                                .as_slice(),
                            0,
                        )?;
                        Ok::<_, ThoosaError>((x.into(), y.into()))
                    })
                }
            })
            .buffer_unordered(10)
            .boxed()
    }
}

#[async_trait]
impl<L, C> AsyncDataBatch<LabelProbability, LabelsProbabilities> for LogitsResourcesBatch<L, C>
where
    L: for<'i> Cached<In<'i>=Resource, Out=ThoTensor<LabelProbability, SynKey>, Ctx=C> + OutId + 'static,
    C: RunCtx + 'static,
{
    async fn train_batch_stream(&self) -> BoxStream<'_, Result<(ThoBatch<LabelProbability>, ThoBatch<LabelsProbabilities>), ThoosaError>> {
        Self::build_batch_stream(
            Arc::clone(&self.loader),
            Arc::clone(&self.ctx),
            Arc::clone(&self.train_resources),
            self.batch_size,
            self.labels_len,
        )
    }

    async fn valid_batch_stream(&self) -> BoxStream<'_, Result<(ThoBatch<LabelProbability>, ThoBatch<LabelsProbabilities>), ThoosaError>> {
        Self::build_batch_stream(
            Arc::clone(&self.loader),
            Arc::clone(&self.ctx),
            Arc::clone(&self.valid_resources),
            self.batch_size,
            self.labels_len,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use parking_lot::RwLock;
    use tch::{Device, Kind, Tensor};
    use tch::nn::{self, VarStore};
    use thoosa_common::dims::{Batch, LabelProbability, MotifsNetsSelectionDims, Slice};
    use thoosa_common::my_tch::{MyClosureModule, ThoModule};
    use thoosa_common::runs::{Logits, NetResources};
    use thoosa_common::train_nets::{FixedNumSlicesNetParams, LabelNetParams, LabelNetTypeParams, ThoModuleWithStore};
    use thoosa_common::train_nets::{CommonTrainingRunConfigFields, FinalNetArgsFields, OptConfig, TrainFinalNetRunConfigFields};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::utils::reset_print_options;
    use thoosa_common::ws::WsMockBuilder;
    use thoosa_shared::hyperparameter::Hyperparameter;
    use thoosa_shared::runner::common::cachable::DevicesWrap;
    use thoosa_shared::runner::common::mock_run_ctx;
    use thoosa_shared::runner::common::runnable::AsyncStage;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::run_configs::OptRunConfigs;
    use thoosa_shared::val::Val;
    use thoosa_shared::val::NamedVal;
    use thoosa_shared::{mono_key, ThoosaError};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_store::MockStoreBuilder;
    use crate::tasks::apply_motif_nets::motif_test_net_;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use crate::tasks::train_final_net::new_train_final_net_with_ml;
    use crate::tasks::cross_loader::ResourcesConfigInput;

    fn const_label_net(
        motif_nets: u32,
        slices_len: u32,
        k: f64,
        device: Device,
    ) -> (
        Box<dyn Fn() -> LabelNetParams + Sync + Send + 'static>,
        Box<dyn Fn() -> ThoModuleWithStore<MotifsNetsSelectionDims, Batch<LabelProbability>> + Sync + Send + 'static>,
    ) {
        (
            Box::new(move || {
                LabelNetParams::new(
                    1,
                    motif_nets,
                    1,
                    slices_len,
                    LabelNetTypeParams::FixedNumSlices(FixedNumSlicesNetParams::new()),
                )
            }),
            Box::new(move || {
                let k = k;
                let builder = Arc::new(move |_path: &nn::Path| {
                    let module = MyClosureModule::new(Arc::new(Box::new(move |t: &Tensor| {
                        let sum = t.f_sum_dim_intlist(&[1, 2][..], false, None)?;
                        let mut sum = sum.f_unsqueeze(1)?;
                        let out = sum.f_mul_scalar_(k)?;
                        Ok(out)
                    })));
                    Ok(Box::new(module) as Box<dyn ThoModule<MotifsNetsSelectionDims, Batch<LabelProbability>>>)
                });
                let mut vs = VarStore::new(device);
                let module = builder(&vs.root()).expect("const_label_net builder");
                ThoModuleWithStore::with_factory(
                    SynKey::new([mono_key!(format!("const_label_net_{k}"), "const_label_net")]),
                    module,
                    vs,
                    Some(builder),
                )
            }),
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_train_final_net_smoke() -> Result<(), ThoosaError> {
        reset_print_options();
        GlobalCtx::init_devices(2);

        let select_motifs_run_id: RunId = 2;
        let motif_width = 4;
        let step = 1;
        let device = Device::Cpu;
        let train_group = "train_group";
        let valid_group = "valid_group";
        let motif_t = Arc::new(ThoTensor::new(
            SynKey::new([mono_key!("motif_0", "motif_id")]),
            Tensor::f_zeros(&[1, motif_width as i64], (Kind::Float, device))?,
        )?);

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 1, 100, 1000)
                .with_load_motif_nets(vec![vec![
                    motif_test_net_(device, Arc::clone(&motif_t), step),
                    motif_test_net_(device, Arc::clone(&motif_t), step),
                ]])
                .with_label_nets(HashMap::from([
                    ("A".to_string(), vec![(100, const_label_net(2, 1, 0.1, device).0, const_label_net(2, 1, 0.1, device).1)]),
                    ("B".to_string(), vec![(200, const_label_net(2, 1, 0.2, device).0, const_label_net(2, 1, 0.2, device).1)]),
                ]))
                .build(),
        ));

        let resources_train = (0..4)
            .map(|i| Resource::new(i, 1, format!("resource_train_{i}"), 32))
            .collect::<Vec<_>>();
        let resources_valid = (10..14)
            .map(|i| Resource::new(i, 1, format!("resource_valid_{i}"), 32))
            .collect::<Vec<_>>();

        let mut resources_map = HashMap::new();
        for r in resources_train.iter().chain(resources_valid.iter()) {
            resources_map.insert(
                r.clone(),
                Tensor::f_zeros(&[1, 32], (Kind::Float, device))?,
            );
        }

        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached(resources_map)
                .finish(),
        );

        let train_logits = resources_train
            .iter()
            .enumerate()
            .map(|(i, r)| (r.clone(), if i % 2 == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] }))
            .collect::<Vec<_>>();
        let valid_logits = resources_valid
            .iter()
            .enumerate()
            .map(|(i, r)| (r.clone(), if i % 2 == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] }))
            .collect::<Vec<_>>();

        let store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
                .with_list_selected_motifs([
                    (select_motifs_run_id, Ok(vec![vec![(0, 1.0f64), (1, 1.0f64)]])),
                ])
                .with_list_output_run_ids([
                    (select_motifs_run_id, Ok(vec![100, 200])),
                ])
                .with_list_resource_random_in_group([
                    (train_group, Ok(train_logits)),
                    (valid_group, Ok(valid_logits)),
                ])
                .build(),
        ));

        let run_configs = OptRunConfigs::new(
            thoosa_shared::run_configs::GridRunConfigs::new(
                thoosa_shared::run_configs::ConstConfigs::new(vec![
                    NamedVal::new(TrainFinalNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Cycles), Val::Int(1)),
                    NamedVal::new(TrainFinalNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::BatchSize), Val::Int(2)),
                    NamedVal::new(TrainFinalNetRunConfigFields::Net(FinalNetArgsFields::NumLayers), Val::Int(1)),
                ]),
                vec![],
            ),
            vec![
                Hyperparameter::UniformFloat {
                    name: TrainFinalNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr),
                    lower: 0.0001,
                    upper: 0.001,
                },
            ],
        );

        let train_final_net = Arc::new(new_train_final_net_with_ml(
            vec![select_motifs_run_id],
            loader,
            run_configs,
            &ws,
            &store,
            Arc::new(RwLock::new(mock_run_ctx::default(0))),
            Arc::new(OptConfig::new(1, 1)),
        ).await?);

        let input = Arc::new(DevicesWrap::wrap(
            ResourcesConfigInput::new(NetResources::new(Logits::new(train_group.to_string()), Logits::new(valid_group.to_string()))),
            0,
        ));

        train_final_net.run(input, Arc::new(RwLock::new(mock_run_ctx::default(0)))).await?;

        Ok(())
    }
}
