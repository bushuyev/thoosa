use futures_util::future::BoxFuture;
use log::debug;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tch::{Kind, Tensor, nn, nn::OptimizerConfig};
use thoosa_named_derive::Named;
use thoosa_shared::traits::{Adjustable, AsyncStage, Data, DynSyncStageBox, OutId, RunCtx, Stage, SyncStage, WithId};
use thoosa_shared::{CopyToDevice, ThoError};

use super::common::{ABData, ClsOut, ModelTestId};
use crate as thoosa_common;
use crate::ipc::cuda_wokrer::AllTensors;
use crate::ipc::proxy::{DynSyncStageEnum, ProxyStage};
use crate::ipc::serialization::json::JsonIpcCodec;
use crate::ipc::worker_msg::{WorkerInput, WorkerOutput};
use crate::ipc::wraps::SyncStageIpcWrapper;
use crate::ipc::wraps::WorkerWrapper;
use crate::runner::common::cachable::DevicesWrap;
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::run_ctx::RunCtxImpl;
use crate::runner::test::FailUntil;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use crate::ws::Workspace;
use thoosa_shared::traits::MaxParallelCtx;

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
            if input.tho_device_idx() == ctx.tho_device_idx() {
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
                    let is_all_on_cpu = GlobalCtx::global().is_all_on_cpu(ctx.tho_device_idx(), false);
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
        ThoTensor::new(out_id, correct.view([1, 1]).into())
    }
}

#[typetag::serde(name = "ThoTensor_ClsOut")]
impl WorkerOutput for ThoTensor<ClsOut, ModelTestId> {}

// The typed classifier proxy sends this concrete tensor reference through WorkerInput.
#[typetag::serde(name = "ThoTensor_ABData_ModelTestId_Input")]
impl WorkerInput for ThoTensor<ABData, ModelTestId> {}

pub(crate) fn sync_classifier_endpoint_init<'a>(init: Box<dyn WorkerInput>, ws: Arc<RwLock<Box<dyn Workspace>>>, all_cuda_tensors: &'a AllTensors, tho_device:&'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    use futures_util::FutureExt;
    let init: Box<dyn std::any::Any> = init;
    let init = init.downcast::<SyncClassifierStageInit>().expect("SyncClassifierStage endpoint received an unexpected initializer type");
    async move {
        // The worker reconstructs the stateless classifier configuration once per endpoint.
        Ok(Box::new(WorkerWrapper::<JsonIpcCodec, _, _, _>::new(Arc::new(SyncClassifierStage::init(*init)), std::marker::PhantomData)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}
