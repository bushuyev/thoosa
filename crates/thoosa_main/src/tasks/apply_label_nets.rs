use std::collections::HashSet;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;
use cached_derive::Cached;
use thoosa_named_derive::Named;
use log::{debug, trace};
use parking_lot::RwLock;
use tch::{IndexOp, Tensor};
use tch::Kind::Float;
use thoosa_common::batches::BatchLengths;
use thoosa_common::dims::{Batch, LabelProbability, MotifProbability, MotifsNetsSelectionDims};
use thoosa_common::my_tch::ThoModule;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::{LabelNetParams, ThoModuleWithStore};
use thoosa_common::ws::Workspace;
use thoosa_shared::comp_id::SynKey;
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::runner::common::{DropIf, OutId, RunCtx};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::{collect_futures_with_progress, mono_key, StreamBuffering, ThoError};
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::traits::{Adjustable, WithId};
use thoosa_store::Store;
use thoosa_with_data::with_data;

use crate::tasks::apply_motif_nets::ApplyMotifNets;
use crate::tasks::cross_dims::ChannelsData2;

#[derive(Debug)]
struct ApplyNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
    apply_motif_nets: Arc<ApplyMotifNets<C>>,
    apply_label_nets: Arc<ApplyLabelNet<C>>
}

/// Applies label nets and returns per-model probabilities. Output is a flat list of label-model
/// scores (Model1ForLabelA, Model2ForLabelA, ...), not a per-label aggregate.
/// This can be used both for applying models for various labels or many different models for same label (for final model or inference)
#[derive(Debug, Cached, Named)]
pub struct ApplyLabelNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    apply_nets: Vec<ApplyNets<C>>,
    pd: PhantomData<C>,
}

impl<C> ApplyLabelNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
    pub async fn new<R>(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        train_label_nets_run_ids: Vec<RunId>,
        mut stage_ctx: R,
        slices_step: u32,
    ) -> Result<Self, ThoError>
    where
        R: RunCtx + Adjustable + 'static,
    {
        let tho_device_idx = stage_ctx.device_idx();

        let label_nets = thoosa_shared::with_perf_log(
            || ws.try_read_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
            "ApplyLabelNets/new/read_ws",
        ).load_label_nets(Some(train_label_nets_run_ids.as_slice()), &ThoDevice::from(tho_device_idx).into())?;

        let mut apply_nets = vec![];

        for (label, nets) in label_nets {


            for (train_label_net_run_id, label_net_params, net) in nets {
                let label_nets = store.list_trained_nets(vec![train_label_net_run_id], None, None).await?;

                assert_eq!(label_nets.len(), 1); //for now there is always single net for run_id
                let (label_net_id, label_net_path, label_net_loss) = &label_nets[0];

                let motif_net_ids = store.list_input_nets(*label_net_id).await?;
                if motif_net_ids.is_empty() {
                    continue//TODO, change to assert, just for tests
                }

                let motif_nets = store.list_trained_nets(motif_net_ids, None, None).await?;

                apply_nets.push(ApplyNets {
                    apply_motif_nets: Arc::new(ApplyMotifNets::from_trained_nets(
                        ws,
                        store,
                        stage_ctx.branch("from_trained_nets".into(), false),
                        slices_step,
                        None,
                        motif_nets,
                    )
                        .await?),
                    apply_label_nets: Arc::new(ApplyLabelNet::new(label.clone(), train_label_net_run_id, net)),
                });
            }
        }


        Ok(Self {
            id: mono_key!(0, "todo").into(),
            apply_nets,
            pd: PhantomData,
        })
    }

    pub fn num_nets(&self) -> usize {
        self.apply_nets.len()
    }
}

impl<C> Stage for ApplyLabelNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = ThoTensor<ChannelsData2, SynKey>;
    type Out = ThoTensor<LabelProbability, SynKey>;
}

impl<C> AsyncStage for ApplyLabelNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="ApplyLabelNets::run"))]
    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        mut ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        debug!("ApplyLabelNets::run: input={input:?}");
        let input_id = input.id().clone();

        #[cfg(debug_assertions)]
        let input_debug = input.clone();

        debug!("ApplyLabelNets::run: label_nets={:?}", self.apply_nets);

        let futures = self.apply_nets
            .iter()
            .map(|apply_net| {
                let input = Arc::clone(&input);
                let mut ctx = ctx.branch(Some("apply_label_nets_motif_probs_0"), false);
                async move {

                    //gpt First apply motif nets to the raw input to produce motif probabilities used by label nets.
                    let motif_probs = Arc::clone(&apply_net.apply_motif_nets)
                        .run_cached(input, ctx.branch(Some("apply_label_nets_motif_probs"), false))
                        .await?;

                    let motif_probs = Arc::clone(&motif_probs);
                    let apply_label_net_ctx = ctx.branch(Some(&format!("apply_label_net_ctx_{}_{}", apply_net.apply_label_nets.label, apply_net.apply_label_nets.train_label_net_run_id)), false);


                    //gpt Each label net runs independently; ordering is preserved by buffer mode below.
                    let result = Arc::clone(&apply_net.apply_label_nets).run(motif_probs, apply_label_net_ctx).await;
                    Ok::<_, ThoError>(result)
                }
            });

        let label_outputs = collect_futures_with_progress(
            futures,
            self.apply_nets.len(),
            crate::buffer_limits::APPLY_LABEL_NETS_RUN_BUFFERED_LIMIT,
            StreamBuffering::Ordered,
            "ApplyLabelNets::run",
        )
        .await
        .into_iter()
        .collect::<Result<Vec<_>, ThoError>>()?;

        debug!("ApplyLabelNets::run: label_outputs={label_outputs:?}");
        //TODO consuming SyncStage for gpu
        let out_id = self.out_id(&input_id);
        if label_outputs.is_empty() {
            Ok(Arc::new(DevicesWrap::wrap(ThoTensor::new(out_id, Tensor::empty(0, (Float, ThoDevice::from(ctx.device_idx()).into())) )?)))
        } else {
            let tensor = with_data!(ctx, [label_outputs]; |_ctx, outs:&[&ThoTensor<LabelProbability, SynKey>]|{
                let stacked = Tensor::f_stack(
                    outs.iter()
                        .map(|o| o.t_ref().shallow_clone())
                        .collect::<Vec<_>>()
                        .as_slice(),
                    0,
                )?;
                ThoTensor::new(out_id, stacked)
            })?;

            #[cfg(debug_assertions)]
            debug!("ApplyLabelNets::run: done, input={input_debug:?}");

            Ok(Arc::new(DevicesWrap::wrap(tensor)))
        }
    }
}

impl<C> OutId for ApplyLabelNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

#[derive(Debug)]
struct ApplyLabelNet<C>
where
    C: RunCtx + Adjustable,
{
    id: SynKey,
    label: String,
    train_label_net_run_id: RunId,
    net_wrapper: Arc<DevicesWrap<ThoModuleWithStore<MotifsNetsSelectionDims, Batch<LabelProbability>>>>,
    pd: PhantomData<C>,
}

impl<C> ApplyLabelNet<C>
where
    C: RunCtx + Adjustable,
{
    fn new(
        label: String,
        train_label_net_run_id: RunId,
        net: ThoModuleWithStore<MotifsNetsSelectionDims, Batch<LabelProbability>>,
    ) -> Self {
        Self {
            id: mono_key!(label.clone(), "ApplyLabelNet") + net.id(),
            label,
            train_label_net_run_id,
            net_wrapper: Arc::new(DevicesWrap::wrap(net)),
            pd: PhantomData,
        }
    }
}

impl<C> Adjustable for ApplyLabelNet<C>
where
    C: RunCtx + Adjustable,
{
}

impl<C> Stage for ApplyLabelNet<C>
where
    C: RunCtx + Adjustable,
{
    type Ctx = C;
    type In<'a> = DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>;
    type Out = ThoTensor<LabelProbability, SynKey>;
}

impl<C> AsyncStage for ApplyLabelNet<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        mut ctx: Self::Ctx,
    ) -> impl Future<Output=Result<Arc<DevicesWrap<Self::Out>>, ThoError>> + Send {
        debug!("ApplyLabelNet::run: input={input:?}");

        let net_wrapper = Arc::clone(&self.net_wrapper);



        async move {
            let out_id = self.out_id(input.id());

            debug!("ApplyLabelNet::run: in async out_id={out_id:?}");

            with_data!(ctx, net_wrapper, input; |mut ctx:Self::Ctx, net:&ThoModuleWithStore<MotifsNetsSelectionDims, Batch<LabelProbability>>, input:&DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>|{

                let batch_input = input.data().t_ref().f_unsqueeze(0)?;//TODO
                let batch_input: ThoTensor<MotifsNetsSelectionDims, thoosa_shared::traits::SimpleId> = batch_input.into();
                let mut lengths = BatchLengths::new(vec![input.data().t_ref().size()[1]]);

                debug!("ApplyLabelNet::run: batch_input={batch_input:?}");
                trace!("ApplyLabelNet::run: batch_input=\n{batch_input}");

                let out = net.tho_forward(&batch_input, Some(&mut lengths), false);
                debug!("ApplyLabelNet::run: out={out:?}");

                let out = out?;
                trace!("ApplyLabelNet::run: out.t_ref=\n{}", out.t_ref());

                let out = out.t_ref().f_i(0)?;
                let out = ThoTensor::new(out_id, out)?;

                debug!("ApplyLabelNet::run: done, input={input:?}");

                Ok(Arc::new(DevicesWrap::wrap(out)))
            })
        }
    }
}

impl<C> OutId for ApplyLabelNet<C>
where
    C: RunCtx + Adjustable,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;
    use log::LevelFilter::Warn;
    use log::{debug, set_max_level, trace};
    use parking_lot::RwLock;
    use tch::{Device, Kind, Tensor};
    use tch::Kind::Float;
    use tch::nn::{self, VarStore};
    use thoosa_properties_map::ToHashmap;
    use thoosa_common::batches::BatchLengths;
    use thoosa_common::dims::{Batch, LabelProbability, MotifProbability, MotifsNetsSelectionDims, Slice};
    use crate::tasks::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::my_tch::{MyClosureModule, ThoModule};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{LinLenNetParams, LabelNetParams, LabelNetTypeParams, MotifNetParams, ThoModuleWithStore};
    use thoosa_common::utils::{assert_close, to_vec_f32};
    use thoosa_common::ws::WsMockBuilder;
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::runner::common::{mock_run_ctx, RunCtx};
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::global_ctx::{mock_global_ctx, GlobalCtx};
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_shared::{mono_key, ThoError};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::run_configs::RunId;
    use thoosa_store::{MockStoreBuilder, RunConfigRow};
    use thoosa_with_data::with_data;
    use crate::tasks::apply_label_nets::ApplyLabelNets;
    use crate::tasks::cross_dims::ChannelsData2;

    fn const_motif_net(
        motif_width: u32,
        step: u32,
        k: f64,
        device: Device,
    ) -> (
        Box<dyn Fn() -> MotifNetParams + Sync + Send + 'static>,
        Box<dyn Fn() -> ThoModuleWithStore<Batch<Slice>, MotifProbability> + Sync + Send + 'static>,
    ) {
        (
            Box::new(move || MotifNetParams::new(motif_width, 1, 1, step, 32, 10)),
            Box::new(move || {
                let k = k;
                let builder = Arc::new(move |_path: &nn::Path| {
                    let module = MyClosureModule::new(Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                        let batch = t.size()[0];
                        let out = Tensor::f_ones(&[batch, 1], (Kind::Float, t.device()))?
                            .f_mul_scalar_(k)?;

                        trace!("const_motif_net: k={k} \nout={out}");
                        Ok(out)
                    })));
                    Ok(Box::new(module) as Box<dyn ThoModule<Batch<Slice>, MotifProbability>>)
                });
                let mut vs = VarStore::new(device);
                let module = builder(&vs.root()).expect("const_motif_net builder");
                ThoModuleWithStore::with_factory(
                    SynKey::new([mono_key!(format!("const_motif_net_{k}"), "const_motif_net")]),
                    module,
                    vs,
                    Some(builder),
                )
            }),
        )
    }

    fn sum_label_net(
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
                    motif_nets,
                    LabelNetTypeParams::LinNetParams(LinLenNetParams::new(slices_len, 1)),
                )
            }),
            Box::new(move || {
                let k = k;
                let builder = Arc::new(move |_path: &nn::Path| {
                    let module = MyClosureModule::new(Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                        let out = sum_label_t(k, t)?;
                        Ok(out)
                    })));
                    Ok(Box::new(module) as Box<dyn ThoModule<MotifsNetsSelectionDims, Batch<LabelProbability>>>)
                });
                let mut vs = VarStore::new(device);
                let module = builder(&vs.root()).expect("sum_label_net builder");
                ThoModuleWithStore::with_factory(
                    SynKey::new([mono_key!(format!("sum_label_net_{k}"), "sum_label_net")]),
                    module,
                    vs,
                    Some(builder),
                )
            }),
        )
    }

    fn sum_label_t(k: f64, t: &Tensor) -> Result<Tensor, ThoError> {
        let sum = t.f_sum_dim_intlist(&[1, 2][..], false, None)?;
        let mut sum = sum.f_unsqueeze(1)?;
        let out = sum.f_mul_scalar_(k)?;
        Ok(out)
    }

    fn weighted_label_net(
        motif_nets: u32,
        slices_len: u32,
        weights: Vec<f64>,
        device: Device,
    ) -> (
        Box<dyn Fn() -> LabelNetParams + Sync + Send + 'static>,
        Box<dyn Fn() -> ThoModuleWithStore<MotifsNetsSelectionDims, Batch<LabelProbability>> + Sync + Send + 'static>,
    ) {
        (
            Box::new(move || {
                LabelNetParams::new(
                    motif_nets,
                    LabelNetTypeParams::LinNetParams(LinLenNetParams::new(slices_len, 1)),
                )
            }),
            Box::new(move || {
                let weights = weights.clone();
                let key_weights = weights.clone();
                let builder = Arc::new(move |_path: &nn::Path| {
                    let weights = weights.clone();
                    let module = MyClosureModule::new(Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                        weighted_label_t(&weights, t)
                    })));
                    Ok(Box::new(module) as Box<dyn ThoModule<MotifsNetsSelectionDims, Batch<LabelProbability>>>)
                });
                let mut vs = VarStore::new(device);
                let module = builder(&vs.root()).expect("weighted_label_net builder");
                ThoModuleWithStore::with_factory(
                    SynKey::new([mono_key!(format!("weighted_label_net_{key_weights:?}"), "weighted_label_net")]),
                    module,
                    vs,
                    Some(builder),
                )
            }),
        )
    }

    fn weighted_label_t(weights: &[f64], t: &Tensor) -> Result<Tensor, ThoError> {
        let weights = weights.iter().map(|v| *v as f32).collect::<Vec<_>>();
        let weights = Tensor::from_slice(&weights)
            .f_to_device(t.device())?
            .f_totype(t.kind())?
            .f_view([1, weights.len() as i64, 1])?;
        let weighted = t.f_mul(&weights)?;
        let sum = weighted.f_sum_dim_intlist(&[1, 2][..], false, None)?;
        Ok(sum.f_unsqueeze(1)?)
    }

    #[derive(Clone, Debug)]
    struct TestMotifNetSpec {
        train_run_id: RunId,
        trained_net_id: u32,
        motif_id: u32,
        channel: u32,
        value: f64,
        loss: f32,
        path: PathBuf,
    }

    impl TestMotifNetSpec {
        fn new(train_run_id: RunId, trained_net_id: u32, motif_id: u32, channel: u32, value: f64, loss: f32) -> Self {
            Self {
                train_run_id,
                trained_net_id,
                motif_id,
                channel,
                value,
                loss,
                path: PathBuf::from(format!("{channel}/{motif_id}_{train_run_id}")),
            }
        }
    }

    #[derive(Clone, Debug)]
    struct TestLabelNetSpec {
        label: String,
        train_run_id: RunId,
        trained_net_id: u32,
        motif_run_ids: Vec<RunId>,
        weights: Vec<f64>,
        loss: f32,
        path: PathBuf,
    }

    impl TestLabelNetSpec {
        fn new(label: &str, train_run_id: RunId, trained_net_id: u32, motif_run_ids: Vec<RunId>, weights: Vec<f64>, loss: f32) -> Self {
            Self {
                label: label.to_string(),
                train_run_id,
                trained_net_id,
                motif_run_ids,
                weights,
                loss,
                path: PathBuf::from(format!("{}/{train_run_id}", label.to_ascii_lowercase())),
            }
        }
    }

    fn motif_nets_for_run_ids<'a>(motifs: &'a [TestMotifNetSpec], run_ids: &[RunId]) -> Vec<&'a TestMotifNetSpec> {
        let mut selected = motifs
            .iter()
            .filter(|motif| run_ids.contains(&motif.train_run_id))
            .collect::<Vec<_>>();
        selected.sort_by(|a, b| a.loss.partial_cmp(&b.loss).expect("test motif losses must be comparable").then(a.trained_net_id.cmp(&b.trained_net_id)));
        selected
    }

    fn slices_len(motif_width: u32, step: u32) -> u32 {
        ((motif_width * 3 - motif_width) / step) + 1
    }

    fn sigmoid(v: f64) -> f64 {
        1.0 / (1.0 + (-v).exp())
    }

    fn expected_label_value(label: &TestLabelNetSpec, motifs: &[TestMotifNetSpec], motif_width: u32, step: u32) -> f32 {
        let motif_nets = motif_nets_for_run_ids(motifs, &label.motif_run_ids);
        assert_eq!(label.weights.len(), motif_nets.len(), "weights must match selected motif nets for {label:?}");
        let per_slice = motif_nets
            .iter()
            .zip(label.weights.iter())
            .map(|(motif, weight)| sigmoid(motif.value) * weight)
            .sum::<f64>();
        (slices_len(motif_width, step) as f64 * per_slice) as f32
    }

    fn assert_values_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_close(*actual as f64, *expected as f64, 0.0001);
        }
    }

    fn sorted_values(mut values: Vec<f32>) -> Vec<f32> {
        values.sort_by(|a, b| a.partial_cmp(b).expect("test values must be comparable"));
        values
    }

    async fn run_apply_label_case(
        labels: &[TestLabelNetSpec],
        motifs: &[TestMotifNetSpec],
        motif_width: u32,
        step: u32,
        channels: u32,
    ) -> Result<(Vec<i64>, Vec<f32>), ThoError> {
        set_max_level(Warn);
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);
        let find_motifs_run_id = 30;
        let device = Device::Cpu;
        let slices_len = slices_len(motif_width, step);

        let motif_ws_entries = motifs
            .iter()
            .map(|motif| {
                let (params, net) = const_motif_net(motif_width, step, motif.value, device);
                (motif.path.clone(), params, net)
            })
            .collect::<Vec<_>>();

        let mut label_ws_entries: HashMap<String, Vec<_>> = HashMap::new();
        for label in labels {
            let motif_nets = motif_nets_for_run_ids(motifs, &label.motif_run_ids);
            let (params, net) = weighted_label_net(motif_nets.len() as u32, slices_len, label.weights.clone(), device);
            label_ws_entries.entry(label.label.clone()).or_default().push((label.train_run_id, params, net));
        }

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, channels, 100, motif_width)
                .with_load_motif_net(motif_ws_entries)
                .with_label_nets(label_ws_entries)
                .build(),
        ));

        let mut list_trained_nets = labels
            .iter()
            .map(|label| {
                (
                    vec![label.train_run_id],
                    None,
                    None,
                    Ok(vec![(label.trained_net_id, label.path.clone(), label.loss)]),
                )
            })
            .collect::<Vec<_>>();
        let mut seen_motif_run_sets = Vec::<Vec<RunId>>::new();
        for label in labels {
            if seen_motif_run_sets.iter().any(|run_ids| *run_ids == label.motif_run_ids) {
                continue;
            }
            let trained_nets = motif_nets_for_run_ids(motifs, &label.motif_run_ids)
                .into_iter()
                .map(|motif| (motif.trained_net_id, motif.path.clone(), motif.loss))
                .collect::<Vec<_>>();
            list_trained_nets.push((label.motif_run_ids.clone(), None, None, Ok(trained_nets)));
            seen_motif_run_sets.push(label.motif_run_ids.clone());
        }

        let store = Arc::new(
            MockStoreBuilder::new()
                .with_run_config_descr([(
                    find_motifs_run_id,
                    Ok(RunConfigRow {
                        id: find_motifs_run_id,
                        r#type: "FindMotifsRunConfig".to_string(),
                        descr: "test find motifs".to_string(),
                        run_on: "2026-01-01 00:00:00".to_string(),
                    }),
                )])
                .with_get_motif_meta(
                    motifs
                        .iter()
                        .map(|motif| (motif.motif_id, Ok((motif.channel as usize, find_motifs_run_id))))
                        .collect::<HashMap<_, _>>(),
                )
                .with_list_run_config_values(
                    find_motifs_run_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_id), step, 100, None, None).to_properties_iter(None).collect()),
                )
                .with_list_trained_nets(list_trained_nets)
                .with_list_input_nets(
                    labels
                        .iter()
                        .map(|label| (label.trained_net_id, Ok(label.motif_run_ids.clone())))
                        .collect::<HashMap<_, _>>(),
                )
                .build(),
        );

        let apply_label_nets = Arc::new(ApplyLabelNets::new(
            &ws,
            &store,
            labels.iter().map(|label| label.train_run_id).collect(),
            mock_run_ctx::default(0),
            step,
        ).await?);
        assert_eq!(labels.len(), apply_label_nets.num_nets());

        let input = ThoTensor::<ChannelsData2, SynKey>::new(
            SynKey::new([mono_key!("input", "input")]),
            Tensor::f_zeros(&[channels as i64, (motif_width as i64) * 3], (Kind::Float, device))?,
        )?;

        let mut run_ctx = mock_run_ctx::default(0);
        let out = apply_label_nets.run(Arc::new(DevicesWrap::wrap(input)), run_ctx.branch(Some("apply_label_nets_run".into()), false)).await?;

        with_data!(run_ctx.branch(Some("apply_label_nets_assert".into()), false), out; |_ctx, out:&ThoTensor<LabelProbability, SynKey>|{
            debug!("t=\n{}", out.t_ref());
            Ok::<_, ThoError>((out.t_ref().size(), to_vec_f32(&out.t_ref().totype(Float))))
        })
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_apply_label_nets_orders_and_values() -> Result<(), ThoError> {
        let motif_width = 4;
        let step = 1;
        let motifs = vec![
            TestMotifNetSpec::new(10, 1000, 0, 0, 1.0, 0.1),
            TestMotifNetSpec::new(10, 1001, 1, 0, 2.0, 0.2),
        ];
        let labels = vec![
            TestLabelNetSpec::new("A", 100, 2000, vec![10], vec![0.1, 0.1], 0.1),
            TestLabelNetSpec::new("A", 101, 2001, vec![10], vec![0.2, 0.2], 0.1),
        ];

        let (size, values) = run_apply_label_case(&labels, &motifs, motif_width, step, 3).await?;
        let expected = labels
            .iter()
            .map(|label| expected_label_value(label, &motifs, motif_width, step))
            .collect::<Vec<_>>();

        assert_eq!(vec![2, 1], size);
        assert_values_close(&values, &expected);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_apply_label_nets_handles_three_motif_nets_for_one_label() -> Result<(), ThoError> {
        let motif_width = 4;
        let step = 1;
        let motifs = vec![
            TestMotifNetSpec::new(10, 1000, 0, 0, 1.0, 0.1),
            TestMotifNetSpec::new(20, 1001, 1, 1, 2.0, 0.2),
            TestMotifNetSpec::new(30, 1002, 2, 2, 3.0, 0.3),
        ];
        let labels = vec![
            TestLabelNetSpec::new("A", 100, 2000, vec![10, 20, 30], vec![1.0, 10.0, 100.0], 0.1),
        ];

        let (size, values) = run_apply_label_case(&labels, &motifs, motif_width, step, 3).await?;
        let expected = vec![expected_label_value(&labels[0], &motifs, motif_width, step)];

        assert_eq!(vec![1, 1], size);
        assert_values_close(&values, &expected);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_apply_label_nets_handles_more_labels_and_label_nets() -> Result<(), ThoError> {
        let motif_width = 4;
        let step = 1;
        let motifs = vec![
            TestMotifNetSpec::new(10, 1000, 0, 0, 1.0, 0.1),
            TestMotifNetSpec::new(20, 1001, 1, 1, 2.0, 0.2),
            TestMotifNetSpec::new(30, 1002, 2, 2, 3.0, 0.3),
        ];
        let labels = vec![
            TestLabelNetSpec::new("A", 100, 2000, vec![10, 20, 30], vec![1.0, 10.0, 100.0], 0.1),
            TestLabelNetSpec::new("A", 101, 2001, vec![20, 30], vec![2.0, 20.0], 0.2),
            TestLabelNetSpec::new("B", 200, 3000, vec![10], vec![3.0], 0.3),
            TestLabelNetSpec::new("C", 300, 4000, vec![30], vec![4.0], 0.4),
        ];

        let (size, values) = run_apply_label_case(&labels, &motifs, motif_width, step, 3).await?;
        let expected = labels
            .iter()
            .map(|label| expected_label_value(label, &motifs, motif_width, step))
            .collect::<Vec<_>>();

        assert_eq!(vec![4, 1], size);
        assert_values_close(&sorted_values(values), &sorted_values(expected));

        Ok(())
    }
}
