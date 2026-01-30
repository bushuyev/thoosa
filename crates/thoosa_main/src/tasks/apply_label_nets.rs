use std::collections::HashSet;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;
use cached_derive::Cached;
use thoosa_named_derive::Named;
use futures_util::{StreamExt, TryStreamExt};
use log::debug;
use parking_lot::RwLock;
use tch::{IndexOp, Tensor};

use thoosa_common::dims::{Batch, LabelProbability, MotifProbability, MotifsNetsSelectionDims};
use thoosa_common::my_tch::ThoModule;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::ThoModuleWithStore;
use thoosa_common::ws::Workspace;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage};
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::{mono_key, ThoError};
use thoosa_shared::traits::{Adjustable, WithId};
use thoosa_store::Store;
use thoosa_with_data::with_data;

use crate::tasks::apply_motif_nets::ApplyMotifNets;
use crate::tasks::cross_dims::ChannelsData2;

/// Applies label nets and returns per-model probabilities. Output is a flat list of label-model
/// scores (Model1ForLabelA, Model2ForLabelA, ...), not a per-label aggregate.
#[derive(Debug, Cached, Named)]
pub struct ApplyLabelNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    apply_motif_nets: Arc<ApplyMotifNets<C>>,
    apply_label_nets: Vec<Arc<ApplyLabelNet<C>>>,
    pd: PhantomData<C>,
}

impl<C> ApplyLabelNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
    pub async fn new<R>(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        select_motifs_run_ids: Vec<RunId>,
        stage_ctx: R,
    ) -> Result<Self, ThoError>
    where
        R: RunCtx + Adjustable + 'static,
    {
        let (device_idx, cache_area) = {
            (stage_ctx.device_idx(), stage_ctx.cache_area())
        };

        let apply_motif_nets = Arc::new({
            #[cfg(test)]
            {
                ApplyMotifNets::new(
                    ws,
                    store,
                    select_motifs_run_ids.clone(),
                    stage_ctx,
                    None,
                )
                .await?
            }
            #[cfg(not(test))]
            {
                ApplyMotifNets::new(
                    ws,
                    store,
                    select_motifs_run_ids.clone(),
                    stage_ctx,
                )
                .await?
            }
        });

        let label_net_run_ids = {
            let mut store_guard = store.write();
            let mut run_ids = HashSet::new();
            for run_id in &select_motifs_run_ids {
                run_ids.extend(store_guard.list_output_run_ids(*run_id)?);
            }
            run_ids
        };

        let mut label_net_run_ids = label_net_run_ids.into_iter().collect::<Vec<_>>();
        label_net_run_ids.sort();
        if label_net_run_ids.is_empty() {
            return Err(ThoError::new("ApplyLabelNets::new: no label nets found"));
        }

        let label_nets = ws.read().load_label_nets(
            Some(label_net_run_ids.as_slice()),
            &ThoDevice::from(device_idx).into(),
        )?;

        let mut label_nets = label_nets
            .into_iter()
            .flat_map(|(label, nets)| {
                nets.into_iter()
                    .map(move |(run_id, params, net)| (label.clone(), run_id, params, net))
            })
            .collect::<Vec<_>>();

        label_nets.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        if label_nets.is_empty() {
            return Err(ThoError::new("ApplyLabelNets::new: no label nets loaded"));
        }

        let apply_label_nets = label_nets
            .into_iter()
            .map(|(label, run_id, _params, net)| {
                Arc::new(ApplyLabelNet::new(label, run_id, net, cache_area))
            })
            .collect::<Vec<_>>();

        let id = SynKey::new(
            apply_label_nets
                .iter()
                .map(|net| {
                    mono_key!(
                        net.run_id,
                        format!("label_net_{}_{}", net.label, net.run_id)
                    )
                })
                .collect::<Vec<_>>(),
        );

        debug!("ApplyLabelNets::new: label_nets.len={} run_ids={:?}",apply_label_nets.len(),label_net_run_ids);

        Ok(Self {
            id,
            apply_motif_nets,
            apply_label_nets,
            pd: PhantomData,
        })
    }

    pub fn num_nets(&self) -> usize {
        self.apply_label_nets.len()
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
        let cache_area = ctx.cache_area();
        let input_id = input.id().clone();

        let motif_probs = Arc::clone(&self.apply_motif_nets)
            .run_cached(input, ctx.branch(Some("apply_label_nets_motif_probs"), false, false))
            .await?;

        let label_nets = self.apply_label_nets.clone();
        let label_outputs = futures::stream::iter(label_nets.into_iter())
            .map(|apply_label_net| {
                let motif_probs = Arc::clone(&motif_probs);
                let apply_label_net_ctx = ctx.branch(Some(&format!("apply_label_net_ctx_{}_{}", apply_label_net.label, apply_label_net.run_id)), false, false);

                async move {
                    apply_label_net.run(motif_probs, apply_label_net_ctx).await
                }
            })
            .buffered(crate::buffer_limits::APPLY_LABEL_NETS_RUN_BUFFERED_LIMIT)
            .try_collect::<Vec<_>>()
            .await?;

        let out_id = self.out_id(&input_id);
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

        Ok(Arc::new(DevicesWrap::wrap(tensor, cache_area.into())))
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
    run_id: RunId,
    net_wrapper: Arc<DevicesWrap<ThoModuleWithStore<MotifsNetsSelectionDims, Batch<LabelProbability>>>>,
    pd: PhantomData<C>,
}

impl<C> ApplyLabelNet<C>
where
    C: RunCtx + Adjustable,
{
    fn new(
        label: String,
        run_id: RunId,
        net: ThoModuleWithStore<MotifsNetsSelectionDims, Batch<LabelProbability>>,
        cache_area: u64,
    ) -> Self {
        Self {
            id: mono_key!(label.clone(), "ApplyLabelNet") + net.id(),
            label,
            run_id,
            net_wrapper: Arc::new(DevicesWrap::wrap(net, cache_area.into())),
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
    type In<'a> = ThoTensor<MotifProbability, SynKey>;
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
        let net_wrapper = Arc::clone(&self.net_wrapper);
        let out_id = self.out_id(input.id());

        async move {

            let cache_area = ctx.cache_area();

            with_data!(ctx, net_wrapper, input; |_ctx, net:&ThoModuleWithStore<MotifsNetsSelectionDims, Batch<LabelProbability>>, input:&ThoTensor<MotifProbability, SynKey>|{
                let batch_input = input.t_ref().f_unsqueeze(0)?;
                let batch_input: ThoTensor<MotifsNetsSelectionDims, thoosa_shared::traits::SimpleId> = batch_input.into();
                let out = net.tho_forward(&batch_input, None/*TODO mask*/, false)?;
                let out = out.t_ref().f_i(0)?;
                let out = ThoTensor::new(out_id, out)?;

                Ok(Arc::new(DevicesWrap::wrap(out, cache_area.into())))
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
    use std::sync::Arc;
    use parking_lot::RwLock;
    use tch::{Device, Kind, Tensor};
    use tch::nn::{self, VarStore};
    use thoosa_common::dims::{Batch, LabelProbability, MotifProbability, MotifsNetsSelectionDims, Slice};
    use thoosa_common::my_tch::{MyClosureModule, ThoModule};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets;
    use thoosa_common::train_nets::{LinLenNetParams, ConvNetParams, LabelNetParams, LabelNetTypeParams, MotifNetParams, ThoModuleWithStore};
    use thoosa_common::utils::{assert_close, vec_f32_from};
    use thoosa_common::ws::WsMockBuilder;
    use thoosa_shared::runner::common::cachable::DevicesWrap;
    use thoosa_shared::runner::cache_areas::CacheArea;
    use thoosa_shared::runner::common::{mock_run_ctx, RunCtx};
    use thoosa_shared::runner::common::runnable::AsyncStage;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_shared::{mono_key, ThoError};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_store::MockStoreBuilder;
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
            Box::new(move || MotifNetParams::new(motif_width, 1, step)),
            Box::new(move || {
                let k = k;
                let builder = Arc::new(move |_path: &nn::Path| {
                    let module = MyClosureModule::new(Arc::new(Box::new(move |t: &Tensor, _mask:Option<&Tensor>| {
                        let batch = t.size()[0];
                        let out = Tensor::f_ones(&[batch, 1], (Kind::Float, t.device()))?
                            .f_mul_scalar_(k)?;
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
                    let module = MyClosureModule::new(Arc::new(Box::new(move |t: &Tensor, _mask:Option<&Tensor>| {
                        let sum = t.f_sum_dim_intlist(&[1, 2][..], false, None)?;
                        let mut sum = sum.f_unsqueeze(1)?;
                        let out = sum.f_mul_scalar_(k)?;
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_apply_label_nets_orders_and_values() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);

        let select_motifs_run_id = 10;
        let motif_width = 4;
        let step = 1;
        let device = Device::Cpu;
        let (label_a_0_params, label_a_0_net) = sum_label_net(2, 1, 0.1, device);
        let (label_a_1_params, label_a_1_net) = sum_label_net(2, 1, 0.2, device);
        let (label_b_0_params, label_b_0_net) = sum_label_net(2, 1, 0.5, device);

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 1, 100, motif_width)
                .with_load_motif_nets(vec![vec![
                    const_motif_net(motif_width, step, 1.0, device),
                    const_motif_net(motif_width, step, 2.0, device),
                ]])
                .with_label_nets(HashMap::from([
                    (
                        "A".to_string(),
                        vec![
                            (100, label_a_0_params, label_a_0_net),
                            (101, label_a_1_params, label_a_1_net),
                        ],
                    ),
                    (
                        "B".to_string(),
                        vec![
                            (200, label_b_0_params, label_b_0_net),
                        ],
                    ),
                ]))
                .build(),
        ));

        let store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
                .with_list_selected_motifs([
                    (
                        select_motifs_run_id,
                        Ok(vec![vec![(0, 1.0f64), (1, 1.0f64)]]),
                    ),
                ])
                .with_list_output_run_ids([
                    (select_motifs_run_id, Ok(vec![100, 101, 200])),
                ])
                .build(),
        ));

        let apply_label_nets = Arc::new(ApplyLabelNets::new(
            &ws,
            &store,
            vec![select_motifs_run_id],
            mock_run_ctx::default(0),
        ).await?);

        let input = ThoTensor::<ChannelsData2, SynKey>::new(
            SynKey::new([mono_key!("input", "input")]),
            Tensor::f_zeros(&[1, motif_width as i64], (Kind::Float, device))?,
        )?;

        let mut run_ctx = mock_run_ctx::default(0);
        let out = apply_label_nets.run(Arc::new(DevicesWrap::wrap(input, CacheArea::default())), run_ctx.branch(Some("apply_label_nets_run".into()), false, false)).await?;

        with_data!(run_ctx.branch(Some("apply_label_nets_assert".into()), false, false), out; |_ctx, out:&ThoTensor<LabelProbability, SynKey>|{
            assert_eq!(out.t_ref().size(), vec![3, 1]);
            let vals = vec_f32_from(out.t_ref());
            assert_close(vals[0], 0.3, 1e-4);
            assert_close(vals[1], 0.6, 1e-4);
            assert_close(vals[2], 1.5, 1e-4);
            Ok::<_, ThoError>(())
        })?;

        Ok(())
    }
}
