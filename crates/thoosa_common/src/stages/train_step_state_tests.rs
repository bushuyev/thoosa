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
struct Init { optimizer: u8, checkpoint: PathBuf }
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

async fn local_step<C: RunCtx>(optimizer: u8, device: ThoDevice, checkpoint: PathBuf) -> Result<TestStep<C>, ThoError> {
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
                // A registered non-trainable buffer exercises the same rollback path as BatchNorm statistics.
                let _ = vs.root().add("test_buffer", Tensor::f_zeros([1], (Kind::Float, vs.device()))?, false);
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
    }, Arc::new(Box::new(|batch| batch) as TrainStepXFn<AnyDims>), |predicted, expected| Ok(predicted.f_mse_loss(expected, Reduction::Mean)?), (), checkpoint).await?;
    Ok(step)
}

fn endpoint_init<'a>(_id: Option<String>, init: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn Workspace>>>, _tensors: &'a AllTensors, device: &'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let init: Box<dyn std::any::Any> = init;
        let Ok(init) = init.downcast::<Init>() else { return Err(ThoError::new("invalid training state test initializer")); };
        let step = local_step::<WokerRunCtx>(init.optimizer, *device, init.checkpoint).await?;
        Ok(Box::new(WorkerWrapperWithState::<JsonIpcCodec, _, _, _>::new_with_input::<WireInput>(Arc::new(step), PhantomData)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}

async fn step(optimizer: u8, id: String, checkpoint: PathBuf) -> Result<Arc<DynSyncStageWithStateEnum<TestStep<RunCtxImpl>>>, ThoError> {
    Ok(Arc::new(if GlobalCtx::global().is_ipc() {
        DynSyncStageWithStateEnum::Proxy {
            stage: Arc::new(ProxyStage::proxy_with_input_cloner(ENDPOINT, Some(id), 1, Box::new(Init { optimizer, checkpoint }), Arc::new(|input: &TrainStepInput<AnyDims, AnyDims>| Ok(Box::new(WireInput(input.clone_refs_for_worker()?))))).await?),
            init: (), state: Default::default(),
        }
    } else {
        DynSyncStageWithStateEnum::Local(Arc::new(local_step(optimizer, 1.into(), checkpoint).await?))
    }))
}

fn input(train: bool) -> Result<TrainStepInput<AnyDims, AnyDims>, ThoError> {
    Ok(TrainStepInput::new(
        ThoTensor::new(debug_id!("train state x"), Tensor::f_from_slice(&[1f32, 2., 3., 2., -1., 1.])?.f_reshape([2, 3])?)?,
        None,
        ThoTensor::new(debug_id!("train state y"), Tensor::f_zeros([2, 1], (Kind::Float, Device::Cpu))?)?,
        train,
    ))
}

async fn round_trips() -> Result<(), ThoError> {
    let global = GlobalCtx::global();
    let actor = global.cuda_actor(1);
    let ctx = RunCtxImpl::new(1, "train state round trip");
    for optimizer in 0..3 {
        let id = format!("restored-{optimizer}");
        let restored = step(optimizer, id.clone(), PathBuf::from(format!("/tmp/unused-restored-{optimizer}"))).await?;
        let reference = step(optimizer, format!("reference-{optimizer}"), PathBuf::from(format!("/tmp/unused-reference-{optimizer}"))).await?;
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
    // Use the real selector/actor/worker path: one selected training invocation must
    // still consume the whole logical batch, including its uneven final microbatch.
    for optimizer in 0..3 {
        let checkpoint = thoosa_shared::tmp_file::TmpFile::new(format!("/tmp/train-transaction-{}-{optimizer}.safetensors", std::process::id()));
        let split = step(optimizer, format!("split-{optimizer}"), checkpoint.path().to_owned()).await?;
        let full = step(optimizer, format!("full-{optimizer}"), PathBuf::from("/tmp/unused-full")).await?;
        for _ in 0..3 {
            let expected = actor.run_stage(full.clone(), uneven_input(true)?, 1, &ctx).await?.loss;
            let wrapper = TrainStepInputWrapper::new(uneven_input(true)?);
            let actual = actor.run_stage_with_select(split.clone(), wrapper, Some(TrainStepInputSelector::from(0..2)), 1, &ctx).await?.loss;
            assert!((actual - expected).abs() < 1e-5, "split loss mismatch: {actual} vs {expected}");
        }
        let expected = actor.run_stage(full, uneven_input(false)?, 1, &ctx).await?.loss;
        let actual = actor.run_stage(split.clone(), uneven_input(false)?, 1, &ctx).await?.loss;
        assert!((actual - expected).abs() < 1e-5, "split updates differ: {actual} vs {expected}");
        // A final validation has no later training call to trigger checkpointing.
        split.checkpoint(&ctx).await?;
        let saved_loss = checkpoint_loss(checkpoint.path())?;
        assert!((saved_loss - actual).abs() < 1e-5, "checkpoint does not match final validation: {saved_loss} vs {actual}");
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

fn uneven_input(train: bool) -> Result<TrainStepInput<AnyDims, AnyDims>, ThoError> {
    Ok(TrainStepInput::new(
        ThoTensor::new(debug_id!("transaction x"), Tensor::f_from_slice(&[1f32, 2., 3., 2., -1., 1., 0.5, 0.25, -0.5])?.f_reshape([3, 3])?)?,
        None,
        ThoTensor::new(debug_id!("transaction y"), Tensor::f_from_slice(&[0f32, 0., 1.])?.f_reshape([3, 1])?)?,
        train,
    ))
}

fn checkpoint_loss(path: &std::path::Path) -> Result<f32, ThoError> {
    let variables = Tensor::read_safetensors(path)?;
    let weight = variables.iter().find(|(name, _)| name.ends_with("weight")).ok_or_else(|| ThoError::new("missing checkpoint weight"))?;
    let bias = variables.iter().find(|(name, _)| name.ends_with("bias")).ok_or_else(|| ThoError::new("missing checkpoint bias"))?;
    let input = uneven_input(false)?;
    let predicted = input.x_ref()?.as_ref().f_linear(&weight.1, Some(&bias.1))?;
    Ok(predicted.f_mse_loss(&input.expected_y_ref()?.as_ref(), Reduction::Mean)?.f_double_value(&[])? as f32)
}

#[tokio::test]
async fn test_train_transaction_cpu() -> Result<(), ThoError> {
    use std::sync::atomic::{AtomicBool, Ordering};
    static FAIL_LAST_CHUNK: AtomicBool = AtomicBool::new(false);
    fn loss(predicted: &Tensor, expected: &Tensor) -> anyhow::Result<Tensor> {
        if expected.f_double_value(&[0, 0])? == 1.0 && FAIL_LAST_CHUNK.swap(false, Ordering::SeqCst) { return Err(ThoError::oom().into()); }
        Ok(predicted.f_mse_loss(expected, Reduction::Mean)?)
    }
    let ctx = RunCtxImpl::new(0, "training transaction CPU");
    for optimizer in 0..3 {
        let checkpoint = thoosa_shared::tmp_file::TmpFile::new(format!("/tmp/train-cpu-{}-{optimizer}.safetensors", std::process::id()));
        let mut split = local_step::<RunCtxImpl>(optimizer, 0.into(), checkpoint.path().to_owned()).await?;
        split.loss_fn = loss;
        let split = Arc::new(split);
        let full = Arc::new(local_step::<RunCtxImpl>(optimizer, 0.into(), PathBuf::from("/tmp/unused-cpu-full")).await?);
        for cycle in 0..3 {
            let input = uneven_input(true)?;
            let wrapper = TrainStepInputWrapper::new(uneven_input(true)?);
            let selected = wrapper.select_ref(0, Some((0..2).into())).await?;
            let mut state = split.train_module.lock().await;
            FAIL_LAST_CHUNK.store(true, Ordering::SeqCst);
            let error = split.clone().do_run_with_state(selected.as_ref(), &mut state, &ctx).unwrap_err();
            assert!(error.is_retryable_oom());
            let actual = split.clone().do_run_with_state(selected.as_ref(), &mut state, &ctx)?.loss;
            let mut full_state = full.train_module.lock().await;
            let expected = full.clone().do_run_with_state(&input, &mut full_state, &ctx)?.loss;
            // Replaying a committed request (for example after lost acknowledgement)
            // returns its previous result without a second optimizer update.
            let replay = split.clone().do_run_with_state(selected.as_ref(), &mut state, &ctx)?.loss;
            assert_eq!(replay, actual);
            assert!((actual - expected).abs() < 1e-5, "optimizer={optimizer}, cycle={cycle}: loss differs after retry");
            let valid = uneven_input(false)?;
            let actual = split.clone().do_run_with_state(&valid, &mut state, &ctx)?.loss;
            let expected = full.clone().do_run_with_state(&valid, &mut full_state, &ctx)?.loss;
            assert!((actual - expected).abs() < 1e-5, "optimizer={optimizer}, cycle={cycle}: update differs after retry");
            // Inject failure after mutating optimizer state and registered variables,
            // proving rollback covers a partially applied update, not just gradients.
            let failed: Result<(), ThoError> = state.training_transaction(|module| {
                let x = input.x()?;
                let predicted = module.module().tho_forward(&x, None, true)?;
                predicted.as_ref().f_mse_loss(&input.expected_y_ref()?.as_ref(), Reduction::Mean)?.f_backward()?;
                module.opt().step();
                let _ = module.state_tensor("var/test_buffer")?.f_fill_(42.0)?;
                Err(ThoError::oom())
            });
            assert!(failed.unwrap_err().is_retryable_oom());
            assert_eq!(state.state_tensor("var/test_buffer")?.f_double_value(&[0])?, 0.25, "registered buffer was not rolled back");
            let rolled_back = split.clone().do_run_with_state(&valid, &mut state, &ctx)?.loss;
            assert!((rolled_back - expected).abs() < 1e-5, "partial optimizer update was not rolled back");
        }
        let mut state = split.train_module.lock().await;
        let valid = split.clone().do_run_with_state(&uneven_input(false)?, &mut state, &ctx)?.loss;
        split.clone().do_run_with_state(&TrainStepInput::checkpoint()?, &mut state, &ctx)?;
        assert!((checkpoint_loss(checkpoint.path())? - valid).abs() < 1e-5, "final checkpoint differs from validation");
    }
    // No tensors are needed by a checkpoint, including on its serialized IPC path.
    let checkpoint = TrainStepInput::<AnyDims, AnyDims>::checkpoint()?;
    let wire = serde_json::to_vec(&checkpoint.clone_refs_for_worker()?)?;
    let restored: TrainStepInput<AnyDims, AnyDims> = serde_json::from_slice(&wire)?;
    assert_eq!(restored.command, TrainStepCommand::Checkpoint);
    assert_eq!(restored.mem_size(), 0);
    Ok(())
}
