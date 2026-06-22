use std::sync::Arc;

use log::debug;
use serde::{Deserialize, Serialize};
use tch::{nn, nn::OptimizerConfig, Kind, Tensor};
use thoosa_named_derive::Named;
use thoosa_shared::traits::{Adjustable, Data, WithId};
use thoosa_shared::{CopyToDevice, ThoError};

use crate::ipc::cuda_process::encode_endpoint_payload;
use crate::ipc::json_ser::{self, encode_json_sync_stage_run_input, SyncStageJsonWrapper};
use crate as thoosa_common;
use crate::runner::common::cachable::{DevicesWrap, WithDataRef};
use crate::runner::common::runnable::{AsyncStage, DynSyncStageBox, Stage, SyncStage};
use crate::runner::common::{MaxParallelCtx, OutId, RunCtx};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::runner::test::FailUntil;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;

use super::common::{ABData, ClsOut, ModelTestId};

pub(crate) const SYNC_CLASSIFIER_ENDPOINT: &str = "ModelTestClassifier";

type ClassifierDynSyncStage = DynSyncStageBox<
    ThoTensor<ABData, ModelTestId>,
    ThoTensor<ClsOut, ModelTestId>,
    RunCtxImpl,
>;

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SyncClassifierStageInit {
    id: ModelTestId,
    in_dim: i64,
    epochs: i64,
    lr: f64,
    fail_until: Vec<FailUntil>,
}

impl SyncClassifierStageInit {
    fn new(id: ModelTestId, in_dim: i64, epochs: i64, lr: f64, fail_until: Vec<FailUntil>) -> Self {
        Self {
            id,
            in_dim,
            epochs,
            lr,
            fail_until,
        }
    }
}

pub(crate) fn sync_classifier_endpoint_init(init_msg: String) -> Result<Box<dyn SyncStageJsonWrapper>, ThoError> {
    let init = serde_json::from_str::<SyncClassifierStageInit>(&init_msg)?;
    Ok(json_ser::make_json_sync_stage_wrapper::<SyncClassifierStage, ThoTensor<ABData, ModelTestId>, ThoTensor<ClsOut, ModelTestId>>(
        SYNC_CLASSIFIER_ENDPOINT,
        SyncClassifierStage::init(init),
    ))
}

#[derive(Debug, cached_derive::Cached, Named)]
#[cache_name = "ClsOutId"]
pub(crate) struct Classifier {
    pub(crate) id: ModelTestId,
    sync_classifier: Arc<ClassifierDynSyncStage>,
}

impl Classifier {
    pub(crate) async fn new(
        id: ModelTestId,
        in_dim: i64,
        epochs: i64,
        lr: f64,
        fail_until: Vec<FailUntil>,
        device: ThoDevice,
    ) -> Result<Self, ThoError>
    {
        if GlobalCtx::global().is_ipc() {
            let proxy = ProxySyncClassifierStage::new(id.clone(), in_dim, epochs, lr, fail_until, device).await?;
            Ok(Self {
                id,
                sync_classifier: Arc::new(ClassifierDynSyncStage::new(Arc::new(proxy))),
            })
        } else {
            let local = SyncClassifierStage::new(id.clone(), in_dim, epochs, lr, fail_until);
            Ok(Self {
                id,
                sync_classifier: Arc::new(ClassifierDynSyncStage::new(Arc::new(local))),
            })
        }
    }
}

impl Adjustable for Classifier {}

impl OutId for Classifier {
    fn out_id(
        &self,
        input_id: &<Self::In<'static> as WithId>::ID,
    ) -> <Self::Out as WithId>::ID {
        self.sync_classifier.out_id(input_id)
    }
}

impl Stage for Classifier {
    type In<'a> = ThoTensor<ABData, ModelTestId>;
    type Out = ThoTensor<ClsOut, ModelTestId>;
    type Ctx = RunCtxImpl;
}

impl SyncStage for Classifier {
    fn do_run<'a>(
        self: Arc<Self>,
        input: &Self::In<'a>,
        ctx: &Self::Ctx,
    ) -> Result<Self::Out, ThoError>
    where
        Self: Sized,
    {
        if GlobalCtx::global().is_ipc() && input.ref_key().is_none() {
            if input.device() == ctx.tho_device_idx() {
                return Err(ThoError::new(format!("Classifier IPC sync path expected a worker tensor ref, got local tensor already on device {}: {input:?}", ctx.tho_device_idx())));
            }

            let input = tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(input.copy_to_device(ctx.tho_device_idx())))?;
            let output = Arc::clone(&self.sync_classifier).do_run(&input, ctx)?;
            if output.ref_key().is_some() {
                return tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(output.copy_to_device(0)));
            }
            return Ok(output);
        }

        let output = Arc::clone(&self.sync_classifier).do_run(input, ctx)?;
        if GlobalCtx::global().is_ipc() && output.ref_key().is_some() {
            return tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(output.copy_to_device(0)));
        }

        Ok(output)
    }
}

impl AsyncStage for Classifier {
    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let device_idx = ctx.tho_device_idx();
        let stage = Arc::clone(&self.sync_classifier);
        let output = input
            .with_data_ref(device_idx, ctx, None, move |tensor: &Self::In<'_>, run_ctx| {
                Arc::clone(&stage).do_run(tensor, run_ctx)
            })
            .await?;

        Ok(Arc::new(DevicesWrap::wrap(output)))
    }
}

#[derive(Debug)]
pub(crate) struct SyncClassifierStage {
    id: ModelTestId,
    in_dim: i64,
    epochs: i64,
    lr: f64,
    fail_until: Vec<FailUntil>,
}

impl SyncClassifierStage {
    fn new(id: ModelTestId, in_dim: i64, epochs: i64, lr: f64, fail_until: Vec<FailUntil>) -> Self {
        Self {
            id,
            in_dim,
            epochs,
            lr,
            fail_until,
        }
    }

    fn init(init: SyncClassifierStageInit) -> Self {
        SyncClassifierStage::new(init.id, init.in_dim, init.epochs, init.lr, init.fail_until)
    }
}

impl Adjustable for SyncClassifierStage {}

impl OutId for SyncClassifierStage {
    fn out_id(
        &self,
        input_id: &<Self::In<'static> as WithId>::ID,
    ) -> <Self::Out as WithId>::ID {
        ModelTestId(format!("cls_{}_inp_{}", self.id.0, input_id.0).leak())
    }
}

impl Stage for SyncClassifierStage {
    type In<'a> = ThoTensor<ABData, ModelTestId>;
    type Out = ThoTensor<ClsOut, ModelTestId>;
    type Ctx = RunCtxImpl;
}

impl SyncStage for SyncClassifierStage {
    fn do_run<'a>(
        self: Arc<Self>,
        input: &Self::In<'a>,
        ctx: &Self::Ctx,
    ) -> Result<Self::Out, ThoError>
    where
        Self: Sized,
    {
        let feat = self.in_dim;
        let fails = &self.fail_until;
        let out_id = self.out_id(input.id());
        Ok(ThoTensor::new(out_id, input.run_inner(|t: &Tensor| {
            debug!("Classifier::do_run: t.size()={:?}", t.size());
            for fail_until in fails {
                match fail_until {
                    FailUntil::MemLimit(m) => {
                        if t.size()[0] > *m as i64 {
                            debug!("Classifier::do_run: OOM MemLimit");
                            return Err(ThoError::oom());
                        }
                    }
                    FailUntil::NaxParallel(max_other_parallel_threshold) => {
                        let max_other_parallel = ctx.max_other_parallel();
                        debug!("Classifier::do_run: ctx={ctx:?} max_other_parallel={max_other_parallel} max_other_parallel_threshold={max_other_parallel_threshold}");
                        if max_other_parallel > *max_other_parallel_threshold {
                            return Err(ThoError::oom());
                        }
                    }
                    FailUntil::DumpedToCpu => {
                        let is_all_on_cpu = GlobalCtx::global().is_all_on_cpu(ctx.tho_device_idx());
                        if !is_all_on_cpu {
                            debug!("Classifier::do_run: OOM DumpedToCpu");
                            return Err(ThoError::oom());
                        }
                    }
                    FailUntil::Never => {}
                }
            }

            let xs = t.narrow(1, 0, feat);
            let ys = t.narrow(1, feat, 1);
            let vs = nn::VarStore::new(t.device());
            let root = &vs.root();
            let linear = nn::linear(root, feat, 1, Default::default());
            let mut opt = nn::Sgd::default().build(&vs, self.lr)?;

            for _epoch in 0..self.epochs {
                let logits = xs.apply(&linear);
                let loss = logits.binary_cross_entropy_with_logits::<&Tensor>(&ys, None, None, tch::Reduction::Mean);
                opt.backward_step(&loss);
            }

            let preds = xs.apply(&linear).sigmoid().ge(0.5).to_kind(Kind::Float);
            let correct = preds.eq_tensor(&ys).to_kind(Kind::Float).mean(Kind::Float);
            debug!("Classifier::do_run: correct=\n{correct}");
            Ok(correct.view([1, 1]))
        })?)?)
    }
}

#[derive(Debug)]
pub(crate) struct ProxySyncClassifierStage {
    id: ModelTestId,
}

impl ProxySyncClassifierStage {
    async fn new(id: ModelTestId, in_dim: i64, epochs: i64, lr: f64, fail_until: Vec<FailUntil>, device: ThoDevice) -> Result<Self, ThoError> {
        GlobalCtx::global()
            .cuda_actor(device)
            .install_endpoint(encode_endpoint_payload(
                SYNC_CLASSIFIER_ENDPOINT,
                &serde_json::to_string(&SyncClassifierStageInit::new(id.clone(), in_dim, epochs, lr, fail_until))?,
            ))
            .await?;

        Ok(Self { id })
    }
}

impl Adjustable for ProxySyncClassifierStage {}

impl OutId for ProxySyncClassifierStage {
    fn out_id(
        &self,
        input_id: &<Self::In<'static> as WithId>::ID,
    ) -> <Self::Out as WithId>::ID {
        ModelTestId(format!("cls_{}_inp_{}", self.id.0, input_id.0).leak())
    }
}

impl Stage for ProxySyncClassifierStage {
    type In<'a> = ThoTensor<ABData, ModelTestId>;
    type Out = ThoTensor<ClsOut, ModelTestId>;
    type Ctx = RunCtxImpl;
}

impl SyncStage for ProxySyncClassifierStage {
    fn do_run<'a>(
        self: Arc<Self>,
        input: &Self::In<'a>,
        ctx: &Self::Ctx,
    ) -> Result<Self::Out, ThoError>
    where
        Self: Sized,
    {
        if input.ref_key().is_none() {
            return Err(ThoError::new(format!("ProxySyncClassifierStage expected worker tensor ref input, got {input:?}")));
        }

        tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(async move {
            let response = GlobalCtx::global()
                .cuda_actor(ctx.tho_device_idx())
                .invoke_endpoint(encode_endpoint_payload(
                    SYNC_CLASSIFIER_ENDPOINT,
                    &encode_json_sync_stage_run_input(serde_json::to_string(input)?, ctx)?,
                ))
                .await?;

            serde_json::from_str::<ThoTensor<ClsOut, ModelTestId>>(&response).map_err(ThoError::from)
        }))
    }
}
