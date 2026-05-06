use std::collections::HashSet;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;
use cached_derive::Cached;
use thoosa_named_derive::Named;
use log::debug;
use parking_lot::RwLock;
use tch::{IndexOp, Tensor};

use thoosa_common::batches::BatchLengths;
use thoosa_common::dims::{Batch, LabelProbability, MotifProbability, MotifsNetsSelectionDims};
use thoosa_common::my_tch::ThoModule;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::ThoModuleWithStore;
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
    // apply_motif_nets: Arc<ApplyMotifNets<C>>,
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
        stage_ctx: R,
        slices_step: u32,
        selected_motif_nets_limit: Option<usize>,
    ) -> Result<Self, ThoError>
    where
        R: RunCtx + Adjustable + 'static,
    {
        let mut apply_nets:Vec<ApplyNets<C>> = Vec::with_capacity(train_label_nets_run_ids.len());
        for train_label_nets_run_id in &train_label_nets_run_ids {
            let label_nets = store.list_trained_nets(*train_label_nets_run_id, None, None).await?;
            assert_eq!(label_nets.len(), 1);//for now there is always single net for run_id
            let (label_net_id, label_net_path, label_net_loss) = &label_nets[0];

            // store.get_input_run_ids(*train_label_nets_run_id, None).await?
            // let motif_nets = store.list_input_run_ids(&[label_net_id]).await?;

            // let apply_motif_nets = Arc::new({
            //     ApplyMotifNets::from_trained_nets(
            //         ws,
            //         store,
            //         train_motif_nets_run_ids.clone(),
            //         stage_ctx,
            //         slices_step,
            //         None,
            //         vec![]
            //     )
            //     .await?
            // });
        }
        // let tho_device_idx = stage_ctx.device_idx();
        //
        // //gpt Resolve motif-net inputs by looking up train_motif_nets_run_ids used to produce the label nets.
        // let train_motif_nets_run_ids = {
        //     let store_guard = store;
        //     let mut run_ids = HashSet::new();
        //     for run_id in &train_label_nets_run_ids {
        //         run_ids.extend(store_guard.get_input_run_ids(*run_id, None).await?);
        //     }
        //     let mut run_ids = run_ids.into_iter().collect::<Vec<_>>();
        //     run_ids.sort();
        //     run_ids
        // };
        // if train_motif_nets_run_ids.is_empty() {
        //     return Err(ThoError::new("ApplyLabelNets::new: no train motif nets found"));
        // }
        //
        // let apply_motif_nets = Arc::new({
        //     ApplyMotifNets::from_trained_nets(
        //         ws,
        //         store,
        //         train_motif_nets_run_ids.clone(),
        //         stage_ctx,
        //         slices_step,
        //         None,
        //         vec![]
        //     )
        //     .await?
        // });
        //
        // let label_net_run_ids = {
        //     let mut run_ids = HashSet::new();
        //     for run_id in &train_label_nets_run_ids {
        //         run_ids.extend(store.list_output_run_ids(*run_id).await?);
        //     }
        //     run_ids
        // };
        //
        // debug!("ApplyLabelNets::new: train_label_nets_run_ids={train_label_nets_run_ids:?} label_net_run_ids={label_net_run_ids:?}");
        //
        // let mut label_net_run_ids = label_net_run_ids.into_iter().collect::<Vec<_>>();
        // label_net_run_ids.sort();
        // if label_net_run_ids.is_empty() {
        //     return Err(ThoError::new("ApplyLabelNets::new: no label nets found"));
        // }
        //
        // let label_nets = thoosa_shared::with_perf_log(
        //     || ws.try_read_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
        //     "ApplyLabelNets/new/read_ws",
        // )
        // .load_label_nets(Some(label_net_run_ids.as_slice()), &ThoDevice::from(tho_device_idx).into())?;
        //
        // let mut label_nets = label_nets
        //     .into_iter()
        //     .flat_map(|(label, nets)| {
        //         nets.into_iter()
        //             .map(move |(run_id, params, net)| (label.clone(), run_id, params, net))
        //     })
        //     .collect::<Vec<_>>();
        //
        // label_nets.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        //
        // if label_nets.is_empty() {
        //     return Err(ThoError::new("ApplyLabelNets::new: no label nets loaded"));
        // }
        //
        // let apply_label_nets = label_nets
        //     .into_iter()
        //     .map(|(label, run_id, _params, net)| {
        //         Arc::new(ApplyLabelNet::new(label, run_id, net))
        //     })
        //     .collect::<Vec<_>>();
        //
        // let id = SynKey::new(
        //     apply_label_nets
        //         .iter()
        //         .map(|net| {
        //             mono_key!(
        //                 net.run_id,
        //                 format!("label_net_{}_{}", net.label, net.run_id)
        //             )
        //         })
        //         .collect::<Vec<_>>(),
        // );
        //
        // debug!("ApplyLabelNets::new: label_nets.len={} run_ids={:?}",apply_label_nets.len(),label_net_run_ids);

        Ok(Self {
            id: mono_key!(0, "todo").into(),
            apply_nets: vec![],
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
                let apply_label_net_ctx = ctx.branch(Some(&format!("apply_label_net_ctx_{}_{}", apply_net.apply_label_nets.label, apply_net.apply_label_nets.run_id)), false);


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

        #[cfg(debug_assertions)]
        debug!("ApplyLabelNets::run: done, input={input_debug:?}");

        Ok(Arc::new(DevicesWrap::wrap(tensor)))
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
    ) -> Self {
        Self {
            id: mono_key!(label.clone(), "ApplyLabelNet") + net.id(),
            label,
            run_id,
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

                let batch_input = input.data().t_ref().f_unsqueeze(0)?;
                let batch_input: ThoTensor<MotifsNetsSelectionDims, thoosa_shared::traits::SimpleId> = batch_input.into();
                let mut lengths = BatchLengths::new(vec![input.data().t_ref().size()[1]]);
                debug!("ApplyLabelNet::run: batch_input=\n{batch_input}");
                let out = net.tho_forward(&batch_input, Some(&mut lengths), false);
                debug!("ApplyLabelNet::run: out={out:?}");
                let out = out?;
                debug!("ApplyLabelNet::run: out.t_ref=\n{}", out.t_ref());

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
    use std::sync::Arc;
    use log::LevelFilter::Trace;
    use log::{set_max_level, trace};
    use parking_lot::RwLock;
    use tch::{Device, Kind, Tensor};
    use tch::nn::{self, VarStore};
    use thoosa_properties_map::ToHashmap;
    use thoosa_common::batches::BatchLengths;
    use thoosa_common::dims::{Batch, LabelProbability, MotifProbability, MotifsNetsSelectionDims, Slice};
    use crate::tasks::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::my_tch::{MyClosureModule, ThoModule};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{LinLenNetParams, ConvNetParams, LabelNetParams, LabelNetTypeParams, MotifNetParams, ThoModuleWithStore};
    use thoosa_common::utils::{assert_close, to_vec_f32};
    use thoosa_common::ws::WsMockBuilder;
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::common::{mock_run_ctx, RunCtx};
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::global_ctx::{mock_global_ctx, GlobalCtx};
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_shared::{mono_key, ThoError};
    use thoosa_shared::comp_id::SynKey;
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
        // set_max_level(Trace);
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);
        let batch = 3;
        let train_motif_nets_run_id = 10;
        let train_label_nets_run_id = 20;
        let find_motifs_run_id = 30;
        let motif_width = 4;
        let step = 1;
        let device = Device::Cpu;
        let (label_a_0_params, label_a_0_net) = sum_label_net(2, 1, 0.1, device);
        let (label_a_1_params, label_a_1_net) = sum_label_net(2, 1, 0.2, device);
        let (label_b_0_params, label_b_0_net) = sum_label_net(2, 1, 0.5, device);

        let (motif_0_params, motif_0_net) = const_motif_net(motif_width, step, 1.0, device);
        let (motif_1_params, motif_1_net) = const_motif_net(motif_width, step, 2.0, device);
        let net_id_0 = std::path::PathBuf::from("0/0_10");
        let net_id_1 = std::path::PathBuf::from("0/1_10");

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 1, 100, motif_width)
                .with_load_motif_net(vec![
                    (net_id_0.clone(), motif_0_params, motif_0_net),
                    (net_id_1.clone(), motif_1_params, motif_1_net),
                ])
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

        let store = Arc::new(
            MockStoreBuilder::new()
                .with_get_input_run_ids([
                    (train_label_nets_run_id, Ok(vec![train_motif_nets_run_id])),
                ])
                .with_run_config_descr([(
                    find_motifs_run_id,
                    Ok(RunConfigRow {
                        id: find_motifs_run_id,
                        r#type: "FindMotifsRunConfig".to_string(),
                        descr: "test find motifs".to_string(),
                        run_on: "2026-01-01 00:00:00".to_string(),
                    }),
                )])
                .with_get_motif_meta([
                    (0, Ok((0usize, find_motifs_run_id))),
                    (1, Ok((0usize, find_motifs_run_id))),
                ])
                .with_list_run_config_values(
                    find_motifs_run_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_id), step, 100, None, None).to_properties_iter(None).collect()),
                )
                .with_list_trained_nets([
                    (
                        train_motif_nets_run_id,
                        Some(0.3),
                        Some(2),
                        Ok(vec![
                            (0, net_id_0, 0.1f32),
                            (0, net_id_1, 0.2f32),
                        ]),
                    ),
                ])
                .with_list_output_run_ids([
                    (train_label_nets_run_id, Ok(vec![100, 101, 200])),
                ])
                .build(),
        );

        let apply_label_nets = Arc::new(ApplyLabelNets::new(
            &ws,
            &store,
            vec![train_label_nets_run_id],
            mock_run_ctx::default(0),
            step,
            Some(2),
        ).await?);

        let input = ThoTensor::<ChannelsData2, SynKey>::new(
            SynKey::new([mono_key!("input", "input")]),
            Tensor::f_zeros(&[batch, motif_width as i64], (Kind::Float, device))?,
        )?;

        let mut run_ctx = mock_run_ctx::default(0);
        let out = apply_label_nets.run(Arc::new(DevicesWrap::wrap(input)), run_ctx.branch(Some("apply_label_nets_run".into()), false)).await?;

        with_data!(run_ctx.branch(Some("apply_label_nets_assert".into()), false), out; |_ctx, out:&ThoTensor<LabelProbability, SynKey>|{
            assert_eq!(out.t_ref().size(), vec![3, 1]);
            let vals = to_vec_f32(out.t_ref());
            let motif_sum = (1.0f32 / (1.0f32 + (-1.0f32).exp())) + (1.0f32 / (1.0f32 + (-2.0f32).exp()));
            assert_close(vals[0], motif_sum * 0.1, 1e-4);
            assert_close(vals[1], motif_sum * 0.2, 1e-4);
            assert_close(vals[2], motif_sum * 0.5, 1e-4);
            Ok::<_, ThoError>(())
        })?;

        Ok(())
    }
}
