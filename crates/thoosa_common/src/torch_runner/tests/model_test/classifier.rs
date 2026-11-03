use thoosa_common::runner::execution::{StageExecution, ExecutionContext};
use std::future::Future;
use std::borrow::Borrow;
use thoosa_common::tch_error::tch_result;
use futures_util::future::BoxFuture;
use log::debug;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tch::{Kind, Tensor, nn, nn::OptimizerConfig};
use thoosa_named_derive::Named;
use thoosa_shared::traits::{Adjustable, OutId, RunCtx, Stage, WithId};
use thoosa_common::runner::sync_stage::{SyncStage};
use thoosa_shared::{CopyToDevice, ThoError};

use super::common::{ABData, ClsOut, ModelTestId};
use crate as thoosa_common;
use crate::ipc::cuda_wokrer::AllTensors;
use crate::ipc::proxy::{DynSyncStageEnum, ProxyStage};
use crate::ipc::serialization::json::JsonIpcCodec;
use crate::ipc::worker_msg::{WorkerInput};
use crate::ipc::wraps::SyncStageIpcWrapper;
use crate::ipc::wraps::WorkerWrapper;
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::run_ctx::RunCtxImpl;
use crate::runner::test::FailUntil;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use crate::ws::Workspace;
use thoosa_shared::traits::MaxParallelCtx;
use thoosa_shared::ipc_types::WorkerOutput;

pub(crate) const SYNC_CLASSIFIER_ENDPOINT: &str = "ModelTestClassifier";

type ClassifierDynSyncStage = DynSyncStageEnum<SyncClassifierStage>;

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SyncClassifierStageInit {
    id: ModelTestId,
    in_dim: i64,
    epochs: i64,
    lr: f64,
    fail_until: Vec<FailUntil>,
}

// Register the classifier configuration for postcard's typed install message.
#[typetag::serde]
impl WorkerInput for SyncClassifierStageInit {}

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

#[derive(Debug, Named)]
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
            let init = SyncClassifierStageInit::new(id.clone(), in_dim, epochs, lr, fail_until);
            Ok(Self {
                id,
                // ProxyStage2 installs the typed initializer and keeps the existing endpoint stage id.
                sync_classifier: Arc::new(DynSyncStageEnum::Proxy {
                    stage: Arc::new(ProxyStage::proxy_with_input_cloner(
                        SYNC_CLASSIFIER_ENDPOINT,
                        Some(0.to_string()),
                        device.tho_device_idx(),
                        Box::new(init),
                        Arc::new(|input: &ThoTensor<ABData, ModelTestId>| Ok(Box::new(input.clone_ref()?) as Box<dyn WorkerInput>)),
                    ).await?),
                    init: (),
                }),
            })
        } else {
            let local = SyncClassifierStage::new(id.clone(), in_dim, epochs, lr, fail_until);
            Ok(Self {
                id,
                sync_classifier: Arc::new(DynSyncStageEnum::Local(Arc::new(local))),
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
        ModelTestId(format!("cls_{}_inp_{}", self.id.0, input_id.0).leak())
    }
}

impl Stage for Classifier {
    type In<'a> = ThoTensor<ABData, ModelTestId>;
    type Out = ThoTensor<ClsOut, ModelTestId>;
    type Ctx = RunCtxImpl;
}







#[derive(Debug)]
pub struct SyncClassifierStage {
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
        let t = input.as_ref();

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
        let mut opt = tch_result(nn::Sgd::default().build(&vs, self.lr))?;

        for _epoch in 0..self.epochs {
            let logits = xs.apply(&linear);
            let loss = logits.binary_cross_entropy_with_logits::<&Tensor>(&ys, None, None, tch::Reduction::Mean);
            opt.backward_step(&loss);
        }

        let preds = xs.apply(&linear).sigmoid().ge(0.5).to_kind(Kind::Float);
        let correct = preds.eq_tensor(&ys).to_kind(Kind::Float).mean(Kind::Float);
        debug!("Classifier::do_run: correct=\n{correct}");
        ThoTensor::new(out_id, correct.view([1, 1]).into())
    }
}

#[typetag::serde(name = "ThoTensor_ClsOut")]
impl WorkerOutput for ThoTensor<ClsOut, ModelTestId> {}

// The typed classifier proxy sends this concrete tensor reference through WorkerInput.
#[typetag::serde(name = "ThoTensor_ABData_ModelTestId_Input")]
impl WorkerInput for ThoTensor<ABData, ModelTestId> {}

pub(crate) fn sync_classifier_endpoint_init<'a>(_stage_id:Option<String>, init: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn Workspace>>>, _all_cuda_tensors: &'a AllTensors, _tho_device:&'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    use futures_util::FutureExt;
    let init: Box<dyn std::any::Any> = init;
    let init = init.downcast::<SyncClassifierStageInit>().expect("SyncClassifierStage endpoint received an unexpected initializer type");
    async move {
        // The worker reconstructs the stateless classifier configuration once per endpoint.
        Ok(Box::new(WorkerWrapper::<JsonIpcCodec, _, _, _>::new(Arc::new(SyncClassifierStage::init(*init)), std::marker::PhantomData)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}

impl StageExecution for Classifier {
    fn execute<'a, 'i: 'a, I, IFut>(self: Arc<Self>, input: IFut, execution: &'a ExecutionContext<'_>, ctx: &'a Self::Ctx) -> futures_util::future::BoxFuture<'a, Result<Self::Out, ThoError>>
    where Self: 'a, Self::In<'i>: Sync, Self::Out: Send, I: Borrow<Self::In<'i>> + Send + 'a, IFut: Future<Output = Result<I, ThoError>> + Send + 'a {
        Box::pin(async move {
            let output = if matches!(execution, ExecutionContext::Remote) {
                // This test's batch builder can lend CPU tensors. Preserve its
                // upload boundary while dispatching through the async contract.
                let borrowed = input.await?;
                let input = borrowed.borrow();
                let prepared;
                let input = if input.ref_key().is_none() {
                    if input.tho_device_idx() == ctx.tho_device_idx() {
                        return Err(ThoError::new("classifier IPC input must be a worker tensor reference"));
                    }
                    prepared = input.copy_to_device(ctx.tho_device_idx()).await?;
                    &prepared
                } else { input };
                Arc::clone(&self.sync_classifier).execute(async { Ok(input) }, execution, ctx).await?
            } else {
                Arc::clone(&self.sync_classifier).execute(input, execution, ctx).await?
            };
            if output.ref_key().is_some() { output.copy_to_device(0).await } else { Ok(output) }
        })
    }
}
