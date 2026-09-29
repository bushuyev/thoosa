//! Exercise the real actor/worker protocol. CUDA poisoning is confined to a
//! worker or an explicitly spawned child, including under plain `cargo test`.
use std::any::Any;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use futures_util::future::{BoxFuture, FutureExt};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tch::{Device, Tensor};
use thoosa_shared::{CudaAction, ThoError};
use thoosa_shims::context::ThoCudaContext;
use super::cuda_process::CudaIPCActorHandle;
use super::cuda_wokrer::{AllTensors, CudaWorker, InitFn, WokerRunCtx};
use super::worker_msg::{DebugOutput, OpResult, WorkerInput, WorkerOutput};
use super::wraps::SyncStageIpcWrapper;
use crate::runner::global_ctx::GlobalCtx;
use crate::tho_device::ThoDevice;
use crate::ws::{Workspace, mock_ws_simp};

const ENDPOINT: &str = "recovery-lifecycle";

#[derive(Debug, Clone, Serialize, Deserialize)]
enum RecoveryAction {
    Probe,
    Fatal,
    Poison,
    Hold { started: String },
    DeferredFatal { started: String, release: String },
}

// Internally tagged WorkerInput values must have a struct payload for Postcard.
// Keep the enum inside that payload, just as production endpoint inputs do.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RecoveryInput {
    action: RecoveryAction,
}

#[typetag::serde]
impl WorkerInput for RecoveryInput {}

fn input(action: RecoveryAction) -> Box<dyn WorkerInput> {
    Box::new(RecoveryInput { action })
}

fn fatal() -> ThoError {
    ThoError::Cuda { action: CudaAction::NeedsRestart, msg: "injected fatal lifecycle error".into() }
}

// A real out-of-range CUDA gather raises a device assertion. Retain allocations
// until process exit so their destructors do not obscure the original failure.
fn enqueue_cuda_poison() -> Result<(), ThoError> {
    let input = Tensor::f_ones([1], (tch::Kind::Float, Device::Cuda(0)))?;
    let index = Tensor::f_from_slice(&[2i64])?.f_to_device(Device::Cuda(0))?;
    let result = input.f_index_select(0, &index);
    std::mem::forget(input);
    std::mem::forget(index);
    std::mem::forget(result?);
    Ok(())
}

fn poison_and_synchronize() -> Result<(), ThoError> {
    let context = ThoCudaContext::new(0)?;
    enqueue_cuda_poison()?;
    context.synchronize()
}

mockall::mock! {
    #[derive(Debug)]
    RecoveryEndpoint {}
    impl SyncStageIpcWrapper for RecoveryEndpoint {
        fn do_run(&self, input: &Box<dyn WorkerInput>, tensors: &AllTensors, device: &ThoDevice, ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError>;
        fn consume(&self, input: Box<dyn WorkerInput>, tensors: &AllTensors, device: &ThoDevice, ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError>;
    }
}

struct MockRecoveryEndpointBuilder;

impl MockRecoveryEndpointBuilder {
    fn build() -> Box<dyn SyncStageIpcWrapper> {
        let mut endpoint = MockRecoveryEndpoint::new();
        endpoint.expect_do_run().returning(|input, _, _, _| endpoint_result(input.as_ref()));
        endpoint.expect_consume().returning(|input, _, _, _| endpoint_result(input.as_ref()));
        Box::new(endpoint)
    }
}

fn endpoint_result(input: &dyn WorkerInput) -> Result<Box<dyn WorkerOutput>, ThoError> {
    match &(input as &dyn Any).downcast_ref::<RecoveryInput>().unwrap().action {
        RecoveryAction::Fatal => return Err(fatal()),
        RecoveryAction::Poison => enqueue_cuda_poison()?,
        RecoveryAction::Probe => {},
        _ => panic!("initializer-only input"),
    }
    Ok(Box::new(DebugOutput::new(&vec![std::process::id().to_string()])))
}

async fn wait_for_file(path: &str) {
    while !Path::new(path).exists() { tokio::time::sleep(Duration::from_millis(5)).await; }
}

fn initialize<'a>(input: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn Workspace>>>, _tensors: &'a AllTensors, _device: &'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let input: Box<dyn Any> = input;
        match input.downcast::<RecoveryInput>().unwrap().action {
            RecoveryAction::Fatal => return Err(fatal()),
            RecoveryAction::Poison => {
                // Fault injection only: suppress this initializer's CUDA error
                // so the next direct copy, rather than install, must report it.
                let _ = enqueue_cuda_poison();
            }
            RecoveryAction::Hold { started } => {
                std::fs::write(started, "accepted")?;
                return std::future::pending().await;
            }
            RecoveryAction::DeferredFatal { started, release } => {
                std::fs::write(started, "accepted")?;
                wait_for_file(&release).await;
                return Err(fatal());
            }
            RecoveryAction::Probe => {},
        }
        Ok(MockRecoveryEndpointBuilder::build())
    }.boxed()
}

async fn install_and_probe(actor: &CudaIPCActorHandle) -> Result<u32, ThoError> {
    let worker = actor.worker_id()?;
    let installed = actor.install_stage(worker, ENDPOINT.into(), Some(1.to_string()), input(RecoveryAction::Probe)).await?;
    assert!(matches!(installed.status, OpResult::Ok(_)));
    let response = actor.invoke_stage(worker, ENDPOINT.into(), Some(1.to_string()), input(RecoveryAction::Probe), false).await?;
    let OpResult::Ok(Some(output)) = response.status else { panic!("probe failed") };
    let output: Box<dyn Any> = output;
    Ok(output.downcast::<DebugOutput>().unwrap().debug_data_snap()[0].parse()?)
}

#[test]
fn test_fatal_actor_paths_replace_worker_ipc() -> Result<(), ThoError> {
    super::tests::run_ipc_test(module_path!(), async {
        tokio::time::timeout(Duration::from_secs(120), async {
            let global = GlobalCtx::global();
            let actor: &dyn Any = global.cuda_actor(1).as_ref();
            let actor = actor.downcast_ref::<CudaIPCActorHandle>().unwrap();
            let gate = global.cuda_activity_gate(1).unwrap();
            let mut pid = install_and_probe(actor).await?;
            for mode in ["install", "invoke", "consume", "cancelled", "copy", "poison"] {
                let worker = actor.worker_id()?;
                let dir = tempfile::tempdir()?;
                let started = dir.path().join("started").to_str().unwrap().to_owned();
                let release = dir.path().join("release").to_str().unwrap().to_owned();
                // A request accepted by the old worker must finish with an
                // error on retirement even if that worker never replies to it.
                let pending = tokio::spawn({
                    let actor = actor.clone();
                    let started = started.clone();
                    async move { actor.install_stage(worker, ENDPOINT.into(), Some(2.to_string()), input(RecoveryAction::Hold { started })).await }
                });
                wait_for_file(&started).await;
                if mode == "cancelled" {
                    let started = dir.path().join("fatal-started").to_str().unwrap().to_owned();
                    let caller = tokio::spawn({
                        let actor = actor.clone();
                        let started = started.clone();
                        let release = release.clone();
                        async move { actor.install_stage(worker, ENDPOINT.into(), Some(3.to_string()), input(RecoveryAction::DeferredFatal { started, release })).await }
                    });
                    wait_for_file(&started).await;
                    caller.abort();
                    assert!(caller.await.unwrap_err().is_cancelled());
                    std::fs::write(&release, "fail now")?;
                    // No live caller is available to report this failure.
                    gate.wait_for_generation_end(worker.generation).await;
                } else if mode == "copy" {
                    use crate::copy_ctx::CudaActorHandle;
                    let response = actor.install_stage(worker, ENDPOINT.into(), Some(3.to_string()), input(RecoveryAction::Poison)).await?;
                    assert!(matches!(response.status, OpResult::Ok(_)));
                    let error = actor.copy(Tensor::f_from_slice(&[1f32])?.into(), 1.into()).await.unwrap_err();
                    assert!(matches!(error.what(), ThoError::ContextReset), "fatal direct copy did not report recovery: {error:#}");
                } else {
                    let response = if mode == "install" {
                        actor.install_stage(worker, ENDPOINT.into(), Some(3.to_string()), input(RecoveryAction::Fatal)).await
                    } else {
                        actor.invoke_stage(worker, ENDPOINT.into(), Some(1.to_string()), input(if mode == "poison" { RecoveryAction::Poison } else { RecoveryAction::Fatal }), mode == "consume").await
                    };
                    let error: ThoError = match response {
                        Ok(response) => match response.status { OpResult::Failed(error) => error.into(), _ => panic!("fatal request succeeded") },
                        Err(error) => error.into(),
                    };
                    assert!(matches!(error.what(), ThoError::ContextReset), "fatal actor path {mode}: {error:#}");
                }
                assert!(pending.await.unwrap().is_err(), "retirement must resolve outstanding requests");
                gate.wait_if_paused().await?;
                assert_eq!(gate.device_generation().0, worker.generation.0 + 1, "one replacement per failed generation");
                assert!(!Path::new(&format!("/proc/{pid}")).exists(), "replacement cannot precede old worker reaping");
                let next_pid = install_and_probe(actor).await?;
                assert_ne!(pid, next_pid);
                pid = next_pid;
                assert!(GlobalCtx::process_failure().is_none(), "worker failure cannot terminate the parent process");
                assert_eq!(global.get_runner(0).submit(0, async { Ok(42) }, None).await?, 42);
            }
            Ok::<_, ThoError>(())
        }).await?
    }, async |sock| {
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
        CudaWorker::run_worker(sock, &ws, HashMap::from([(ENDPOINT.into(), Box::new(initialize) as InitFn)]), Device::Cuda(0).into()).await
    })
}

#[test]
fn test_failed_replacement_readiness_is_terminal_only_for_its_device_ipc() -> Result<(), ThoError> {
    super::tests::run_ipc_test(module_path!(), async {
        tokio::time::timeout(Duration::from_secs(50), async {
            let global = GlobalCtx::global();
            let actor: &dyn Any = global.cuda_actor(1).as_ref();
            let actor = actor.downcast_ref::<CudaIPCActorHandle>().unwrap();
            let pid = install_and_probe(actor).await?;
            let worker = actor.worker_id()?;
            let gate = global.cuda_activity_gate(1).unwrap();
            let _ = actor.invoke_stage(worker, ENDPOINT.into(), Some(1.to_string()), input(RecoveryAction::Fatal), false).await;
            assert_eq!(global.get_runner(0).submit(0, async { Ok(42) }, None).await?, 42);
            let failure = gate.wait_if_paused().await.unwrap_err();
            assert!(failure.is_cuda_recovery_failed());
            assert_eq!(gate.device_generation(), worker.generation, "unready replacement must never publish its generation");
            assert!(!Path::new(&format!("/proc/{pid}")).exists());
            assert!(actor.worker_id().unwrap_err().is_cuda_recovery_failed());
            assert!(GlobalCtx::process_failure().is_none());
            assert_eq!(global.get_runner(0).submit(0, async { Ok(43) }, None).await?, 43);
            Ok::<_, ThoError>(())
        }).await?
    }, async |sock| {
        let worker: super::worker_msg::CudaWorkerId = serde_json::from_str(&std::env::var(super::cuda_process::WORKER_ID_ENV)?)?;
        if worker.generation.0 > 0 { anyhow::bail!("injected failure before replacement readiness"); }
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
        CudaWorker::run_worker(sock, &ws, HashMap::from([(ENDPOINT.into(), Box::new(initialize) as InitFn)]), Device::Cuda(0).into()).await
    })
}

#[test]
fn test_fatal_local_cuda_error_requests_process_shutdown_in_child() -> Result<(), ThoError> {
    const CHILD: &str = "THOOSA_LOCAL_RECOVERY_TEST_CHILD";
    if let Ok(mode) = std::env::var(CHILD) {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build()?;
        let result = rt.block_on(async {
            let global = GlobalCtx::init_devices(2)?;
            let error = match mode.as_str() {
                "execution" => global.get_runner(1).submit(0, async { tokio::task::spawn_blocking(poison_and_synchronize).await? }, None).await.unwrap_err(),
                "consume" => {
                    use thoosa_shared::traits::ConsumeResult;
                    let result = global.get_runner(1).submit_consume(0, async {
                        let error = tokio::task::spawn_blocking(poison_and_synchronize).await?.unwrap_err();
                        Ok(ConsumeResult::<_, ()>::Failed { input: vec![7], error })
                    }, None).await?;
                    let ConsumeResult::Failed { input, error } = result else { panic!("poisoned consume succeeded") };
                    assert_eq!(input, vec![7]);
                    error
                }
                "copy" => {
                    let source = Tensor::f_from_slice(&[1f32])?;
                    assert!(tokio::task::spawn_blocking(poison_and_synchronize).await?.unwrap_err().needs_cuda_ctx_reset());
                    global.cuda_actor(1).copy(source.into(), 1.into()).await.unwrap_err()
                }
                _ => panic!("unknown local failure path"),
            };
            assert!(error.is_cuda_recovery_failed(), "fatal local execution must be terminal: {error:#}");
            let failure = tokio::time::timeout(Duration::from_secs(2), GlobalCtx::wait_for_process_failure()).await?;
            let ThoError::CudaRecoveryFailed(failure) = failure.what() else { panic!("lost terminal failure") };
            assert!(failure.original.needs_cuda_ctx_reset());
            assert!(GlobalCtx::is_shutting_down());
            assert_eq!(global.cuda_activity_gate(1).unwrap().device_generation().0, 0);
            assert!(global.get_runner(1).submit(0, async { panic!("local device reopened"); #[allow(unreachable_code)] Ok(()) }, None).await.unwrap_err().is_cuda_recovery_failed());
            global.shutdown().await;
            Ok::<_, ThoError>(())
        });
        rt.shutdown_background();
        result?;
        // Avoid returning to a shared libtest process with a poisoned context.
        std::process::exit(0);
    }
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    rt.block_on(async {
        let name = concat!(module_path!(), "::test_fatal_local_cuda_error_requests_process_shutdown_in_child").split_once("::").unwrap().1;
        for mode in ["execution", "consume", "copy"] {
            let mut child = tokio::process::Command::new(std::env::current_exe()?)
                .args([name, "--exact", "--nocapture"]).env(CHILD, mode).kill_on_drop(true).spawn()?;
            let status = match tokio::time::timeout(Duration::from_secs(45), child.wait()).await {
                Ok(status) => status?,
                Err(error) => { child.kill().await?; return Err(ThoError::from(error)); }
            };
            assert!(status.success(), "isolated local fatal path {mode} failed: {status}");
        }
        Ok(())
    })
}
