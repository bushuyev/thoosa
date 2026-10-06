use super::*;
use crate::dims::AnyDims;
use crate::ipc::cuda_process::CudaIPCActorHandle;
use crate::ipc::cuda_wokrer::{CudaWorker, InitFn, WokerRunCtx};
use crate::ipc::proxy::{DynSyncStageWithStateEnum, ProxyStage};
use crate::ipc::serialization::json::JsonIpcCodec;
use crate::ipc::worker_msg::OpResult;
use crate::ipc::wraps::{SyncStageIpcWrapper, WorkerWrapperWithState};
use crate::nets::builder::SeqBuilder;
use crate::nets::tho_varstore::{ModConfig, ThoOptimizers};
use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
use crate::runner::run_ctx::{RunCtxImpl};
use crate::ws::{Workspace, mock_ws_simp};
use parking_lot::RwLock;
use std::collections::HashMap;
use tch::{Device, Kind, Reduction, nn};
use thoosa_shared::debug_id;

const ENDPOINT: &str = "TrainStepStateTest";
type TestStep<C> = TrainStep<TrainStepXFn<AnyDims>, AnyDims, AnyDims, C>;

#[derive(Debug, Serialize, Deserialize)]
struct Init { optimizer: u8 }
impl WorkerStageInput for Init {}
#[typetag::serde(name = "TrainStepStateTestInit")]
impl WorkerInput for Init {}

#[derive(Debug, Serialize, Deserialize)]
struct WireInput(TrainStepInput<AnyDims, AnyDims>);
#[typetag::serde(name = "TrainStepStateTestInput")]
impl WorkerInput for WireInput {}
impl std::borrow::Borrow<TrainStepInput<AnyDims, AnyDims>> for WireInput {
    fn borrow(&self) -> &TrainStepInput<AnyDims, AnyDims> { &self.0 }
}

async fn local_step<C: RunCtx>(optimizer: u8, device: ThoDevice) -> Result<TestStep<C>, ThoError> {
    let config: ThoOptimizers = match optimizer {
        0 => nn::Adam::default().amsgrad(true).into(),
        1 => nn::AdamW::default().amsgrad(true).into(),
        _ => nn::Sgd { momentum: 0.9, dampening: 0., wd: 0.01, nesterov: true }.into(),
    };
    let (step, ()) = TrainStep::new_local(device, move |(), device| async move {
        let config = ModConfig::new(config, |vs| {
            let initialize = vs.variables().is_empty();
            let mut builder = SeqBuilder::new(3, None)?;
            let module = builder.linear_s(1, "output")?.end_seq()?.build_tch_module(&vs.root())?;
            if initialize {
                tch::no_grad(|| -> anyhow::Result<()> {
                    for mut tensor in vs.variables().into_values() { let _ = tensor.f_fill_(0.25)?; }
                    Ok(())
                })?;
            }
            Ok(module)
        }, 0.01);
        let mut module = ThoTrainModule::new_on_device(config, device)?;
        // Restoration must preserve live group options, not just the initial config.
        module.set_lr(0.017);
        Ok((module, ()))
    }, Arc::new(Box::new(|batch| batch) as TrainStepXFn<AnyDims>), |predicted, expected| Ok(predicted.f_mse_loss(expected, Reduction::Mean)?), (), PathBuf::from("/tmp/unused_train_state_checkpoint")).await?;
    Ok(step)
}

fn endpoint_init<'a>(_id: Option<String>, init: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn Workspace>>>, _tensors: &'a AllTensors, device: &'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let init: Box<dyn std::any::Any> = init;
        let Ok(init) = init.downcast::<Init>() else { return Err(ThoError::new("invalid training state test initializer")); };
        let step = local_step::<WokerRunCtx>(init.optimizer, *device).await?;
        Ok(Box::new(WorkerWrapperWithState::<JsonIpcCodec, _, _, _>::new_with_input::<WireInput>(Arc::new(step), PhantomData)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}

async fn step(optimizer: u8, id: String) -> Result<Arc<DynSyncStageWithStateEnum<TestStep<RunCtxImpl>>>, ThoError> {
    Ok(Arc::new(if GlobalCtx::global().is_ipc() {
        DynSyncStageWithStateEnum::Proxy {
            stage: Arc::new(ProxyStage::proxy_with_input_cloner(ENDPOINT, Some(id), 1, Box::new(Init { optimizer }), Arc::new(|input: &TrainStepInput<AnyDims, AnyDims>| Ok(Box::new(WireInput(input.clone_refs_for_worker()?))))).await?),
            init: (), state: Default::default(),
        }
    } else {
        DynSyncStageWithStateEnum::Local(Arc::new(local_step(optimizer, 1.into()).await?))
    }))
}

fn input(train: bool) -> Result<TrainStepInput<AnyDims, AnyDims>, ThoError> {
    Ok(TrainStepInput::new(
        ThoTensor::new(debug_id!("train state x"), Tensor::f_from_slice(&[1f32, 2., 3., 2., -1., 1.])?.f_reshape([2, 3])?)?,
        None,
        ThoTensor::new(debug_id!("train state y"), Tensor::f_zeros([2, 1], (Kind::Float, Device::Cpu))?)?,
        train, false,
    ))
}

async fn round_trips() -> Result<(), ThoError> {
    let global = GlobalCtx::global();
    let actor = global.cuda_actor(1);
    let ctx = RunCtxImpl::new(1, "train state round trip");
    for optimizer in 0..3 {
        let id = format!("restored-{optimizer}");
        let restored = step(optimizer, id.clone()).await?;
        let reference = step(optimizer, format!("reference-{optimizer}")).await?;
        // Compare with uninterrupted training, including a purge before any
        // optimizer state exists and repeated purges after momentum is populated.
        for cycle in 0..5 {
            let train = cycle > 0;
            let expected = actor.run_stage(reference.clone(), input(train)?, 1, &ctx).await?.loss;
            let actual = actor.run_stage(restored.clone(), input(train)?, 1, &ctx).await?.loss;
            assert!((actual - expected).abs() < 1e-6, "optimizer={optimizer}, cycle={cycle}: {actual} != {expected}");
            restored.purge_gpu(None, 2, "wrong device").await?;
            assert!(restored.has_gpu_data(1).await);
            restored.purge_gpu(None, 1, "training round trip").await?;
            assert!(!restored.has_gpu_data(1).await);
            restored.purge_gpu(None, 1, "repeated purge").await?;
            if global.is_ipc() {
                let ipc: &dyn std::any::Any = actor.as_ref();
                let ipc = ipc.downcast_ref::<CudaIPCActorHandle>().unwrap();
                let raw_input = input(false)?.copy_to_device(1).await?;
                let result = ipc.invoke_stage(ipc.worker_id()?, ENDPOINT.into(), Some(id.clone()), Box::new(WireInput(raw_input)), false).await?;
                let OpResult::Failed(error) = result.status else { panic!("worker kept training state after purge") };
                assert!(ThoError::from(error).to_string().contains("offloaded"));
            }
        }
        // Check the final update too: training returns the loss before its update.
        let expected = actor.run_stage(reference, input(false)?, 1, &ctx).await?.loss;
        let actual = actor.run_stage(restored, input(false)?, 1, &ctx).await?.loss;
        assert!((actual - expected).abs() < 1e-6, "optimizer={optimizer}: final loss {actual} != {expected}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_train_state_local() -> Result<(), ThoError> {
    let global = GlobalCtx::init_devices(2)?;
    let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
    let result = tokio::time::timeout(std::time::Duration::from_secs(90), round_trips()).await?;
    global.shutdown().await;
    result
}

#[test]
fn test_train_state_ipc() -> Result<(), ThoError> {
    crate::ipc::tests::run_ipc_test(module_path!(), async {
        tokio::time::timeout(std::time::Duration::from_secs(90), round_trips()).await?
    }, async |sock| {
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
        CudaWorker::run_worker(sock, &ws, HashMap::from([(ENDPOINT.into(), Box::new(endpoint_init) as InitFn)]), Device::Cuda(0).into()).await
    })
}
