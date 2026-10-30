//! Exercise the real actor/worker protocol. CUDA poisoning is confined to a
//! worker or an explicitly spawned child, including under plain `cargo test`.
use thoosa_common::tch_error::tch_result;
use std::any::Any;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use futures_util::future::{BoxFuture, FutureExt};
use log::debug;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tch::{Device, Tensor};
use thoosa_shared::{CudaAction, ThoError};
use thoosa_shims::context::ThoCudaContext;
use super::cuda_process::CudaIPCActorHandle;
use super::cuda_wokrer::{AllTensors, CudaWorker, InitFn, WokerRunCtx};
use super::worker_msg::{CudaWorkerMsgOut, DebugOutput, OpResult, WorkerInput};
use super::wraps::SyncStageIpcWrapper;
use crate::runner::global_ctx::GlobalCtx;
use crate::tho_device::ThoDevice;
use crate::ws::{Workspace, mock_ws_simp};
use thoosa_shared::ipc_types::WorkerOutput;
const ENDPOINT: &str = "recovery-lifecycle";

#[derive(Debug, Clone, Serialize, Deserialize)]
enum RecoveryAction {
    Probe,
    Fatal,
    StageFailure,
    Poison,
    Hold { started: String },
    DeferredFatal { started: String, release: String },
    DeferredSuccess { started: String, release: String },
    OomOnce { marker: String },
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
    // Use the constructor so injected CUDA errors capture their backtrace.
    thoosa_shared::tho_error::cuda_error(CudaAction::NeedsRestart, "injected fatal lifecycle error".into())
}

// A real out-of-range CUDA gather raises a device assertion. Retain allocations
// until process exit so their destructors do not obscure the original failure.
fn enqueue_cuda_poison() -> Result<(), ThoError> {
    let input = tch_result(Tensor::f_ones([1], (tch::Kind::Float, Device::Cuda(0))))?;
    let index = tch_result(tch_result(Tensor::f_from_slice(&[2i64]))?.f_to_device(Device::Cuda(0)))?;
    let result = input.f_index_select(0, &index);
    std::mem::forget(input);
    std::mem::forget(index);
    std::mem::forget(tch_result(result)?);
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
        RecoveryAction::StageFailure => return Err(ThoError::new("device-side assert triggered")),
        RecoveryAction::Poison => enqueue_cuda_poison()?,
        RecoveryAction::Probe => {},
        _ => panic!("initializer-only input"),
    }
    Ok(Box::new(DebugOutput::new(&vec![std::process::id().to_string()])))
}

async fn wait_for_file(path: &str) {
    while !Path::new(path).exists() { tokio::time::sleep(Duration::from_millis(5)).await; }
}

fn initialize<'a>(_tage_id:Option<String>, input: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn Workspace>>>, _tensors: &'a AllTensors, _device: &'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let input: Box<dyn Any> = input;
        match input.downcast::<RecoveryInput>().unwrap().action {
            RecoveryAction::Fatal => return Err(fatal()),
            RecoveryAction::StageFailure => return Err(ThoError::new("device-side assert triggered")),
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
            RecoveryAction::DeferredSuccess { started, release } => {
                // Cross-process barriers make cancellation happen after the
                // worker accepted installation but before it acknowledged it.
                std::fs::write(started, "accepted")?;
                wait_for_file(&release).await;
            }
            RecoveryAction::OomOnce { marker } => {
                if !Path::new(&marker).exists() {
                    std::fs::write(marker, "retry can initialize")?;
                    return Err(thoosa_shared::tho_error::cuda_error(CudaAction::HandleOOM, "injected initialization OOM".into()));
                }
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

async fn submit_worker_request(request: impl std::future::Future<Output = anyhow::Result<CudaWorkerMsgOut>> + Send + 'static) -> Result<(), ThoError> {
    // Stage errors must cross the same runner boundary as production requests;
    // invoking the transport alone does not exercise stage recovery reporting.
    GlobalCtx::global().get_runner(1).submit(0, async move {
        match request.await?.status {
            OpResult::Failed(error) => Err(error.into()),
            OpResult::Ok(_) => Ok(()),
        }
    }, None).await
}

#[test]
fn test_endpoint_install_cancellation_and_drop_during_reset_ipc() -> Result<(), ThoError> {
    super::tests::run_ipc_test(module_path!(), async {
        tokio::time::timeout(Duration::from_secs(60), async {
            let global = GlobalCtx::global();
            let actor: &dyn Any = global.cuda_actor(1).as_ref();
            let actor = actor.downcast_ref::<CudaIPCActorHandle>().unwrap();
            let gate = global.cuda_activity_gate(1).unwrap();
            let dir = tempfile::tempdir()?;
            let started = dir.path().join("started").to_str().unwrap().to_owned();
            let release = dir.path().join("release").to_str().unwrap().to_owned();
            let worker = actor.worker_id()?;
            let owner = actor.register_stage(worker, ENDPOINT.into(), Some("cancelled".into()), input(RecoveryAction::DeferredSuccess { started: started.clone(), release: release.clone() })).await?;
            let caller = tokio::spawn({
                let actor = actor.clone();
                let owner = owner.clone();
                async move { actor.ensure_stage(owner, worker).await }
            });
            wait_for_file(&started).await;
            caller.abort();
            assert!(caller.await.unwrap_err().is_cancelled());
            drop(owner);
            assert!(actor.register_stage(worker, ENDPOINT.into(), Some("cancelled".into()), input(RecoveryAction::Probe)).await.is_err(), "accepted installation must retain ownership until acknowledgement");
            std::fs::write(&release, "finish")?;
            gate.wait_until_idle().await;
            // Reuse the same key while its old deletion may still be pending.
            let owner = actor.register_stage(worker, ENDPOINT.into(), Some("cancelled".into()), input(RecoveryAction::Probe)).await?;
            actor.ensure_stage(owner.clone(), worker).await?;
            let response = actor.invoke_stage(worker, ENDPOINT.into(), Some("cancelled".into()), input(RecoveryAction::Probe), false).await?;
            assert!(matches!(response.status, OpResult::Ok(_)));
            let marker = dir.path().join("oom").to_str().unwrap().to_owned();
            std::fs::write(&marker, "initial generation succeeds")?;
            let retryable = actor.register_stage(worker, ENDPOINT.into(), Some("oom".into()), input(RecoveryAction::OomOnce { marker: marker.clone() })).await?;
            actor.ensure_stage(retryable.clone(), worker).await?;
            std::fs::remove_file(marker)?;
            global.get_runner(1).request_cuda_reset(ThoError::ContextReset)?;
            assert!(gate.is_paused());
            drop(owner);
            gate.wait_if_paused().await?;
            let worker = actor.worker_id()?;
            // Failed lazy reconstruction must remain retryable in this same
            // generation, without publishing an installed endpoint or restarting.
            assert!(actor.ensure_stage(retryable.clone(), worker).await.unwrap_err().is_oom());
            assert_ne!(retryable.installed_worker(), Some(worker));
            actor.ensure_stage(retryable.clone(), worker).await?;
            assert_eq!(retryable.installed_worker(), Some(worker));
            let response = actor.invoke_stage(worker, ENDPOINT.into(), Some("cancelled".into()), input(RecoveryAction::Probe), false).await?;
            let OpResult::Failed(error) = response.status else { panic!("released endpoint was replayed") };
            assert!(ThoError::from(error).is_missing_endpoint());
            let owner = actor.register_stage(worker, ENDPOINT.into(), Some("cancelled".into()), input(RecoveryAction::Probe)).await?;
            actor.ensure_stage(owner, worker).await?;
            Ok(())
        }).await?
    }, async |sock| {
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));
        CudaWorker::run_worker(sock, &ws, HashMap::from([(ENDPOINT.into(), Box::new(initialize) as InitFn)]), thoosa_common::tho_device::ThoDevice::from(1usize)).await
    })
}

#[test]
fn test_stage_failure_is_reported_by_runner_ipc() -> Result<(), ThoError> {
    super::tests::run_ipc_test(module_path!(), async {
        tokio::time::timeout(Duration::from_secs(60), async {
            let global = GlobalCtx::global();
            let actor: &dyn Any = global.cuda_actor(1).as_ref();
            let actor = actor.downcast_ref::<CudaIPCActorHandle>().unwrap();
            let gate = global.cuda_activity_gate(1).unwrap();
            let mut pid = install_and_probe(actor).await?;
            for mode in ["install", "invoke", "consume", "cancelled"] {
                let worker = actor.worker_id()?;
                let (observed_tx, observed_rx) = tokio::sync::oneshot::channel();
                let (release_tx, release_rx) = tokio::sync::oneshot::channel();
                let caller = tokio::spawn({
                    let actor = actor.clone();
                    async move {
                        global.get_runner(1).submit(0, async move {
                            let response = if mode == "install" {
                                actor.install_stage(worker, ENDPOINT.into(), Some(2.to_string()), input(RecoveryAction::StageFailure)).await?
                            } else {
                                actor.invoke_stage(worker, ENDPOINT.into(), Some(1.to_string()), input(RecoveryAction::StageFailure), mode == "consume").await?
                            };
                            let OpResult::Failed(error) = response.status else { panic!("fatal stage succeeded") };
                            // Hold the stage result before the runner observes it:
                            // the transport must preserve it without starting recovery.
                            observed_tx.send(error.clone()).unwrap();
                            release_rx.await.unwrap();
                            Err::<(), ThoError>(error.into())
                        }, None).await
                    }
                });
                let error = ThoError::from(observed_rx.await.unwrap());
                debug!("test_stage_failure_is_reported_by_runner_ipc: error={error:?}");
                assert!(error.to_string().contains("device-side assert triggered"), "stage reply was rewritten");
                assert!(error.needs_cuda_restart());
                assert!(gate.check_admission().is_ok(), "actor reported a stage failure before the runner");
                if mode == "cancelled" {
                    caller.abort();
                    assert!(caller.await.unwrap_err().is_cancelled());
                    // The admitted runner job still owns error reporting after
                    // its submitting task disappears.
                    release_tx.send(()).unwrap();
                    gate.wait_for_generation_end(worker.generation).await;
                } else {
                    release_tx.send(()).unwrap();
                    assert!(matches!(caller.await?.unwrap_err().what(), ThoError::ContextReset));
                }
                gate.wait_if_paused().await?;
                assert_eq!(gate.device_generation().0, worker.generation.0 + 1);
                assert!(!Path::new(&format!("/proc/{pid}")).exists());
                pid = install_and_probe(actor).await?;
            }
            Ok::<_, ThoError>(())
        }).await?
    }, async |sock| {
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));
        CudaWorker::run_worker(sock, &ws, HashMap::from([(ENDPOINT.into(), Box::new(initialize) as InitFn)]), thoosa_common::tho_device::ThoDevice::from(1usize)).await
    })
}

#[test]
fn test_training_trial_survives_fatal_worker_restart_ipc() -> Result<(), ThoError> {
    super::tests::run_ipc_test(module_path!(), async {
        tokio::time::timeout(Duration::from_secs(60), async {
            let global = GlobalCtx::global();
            let actor: &dyn Any = global.cuda_actor(1).as_ref();
            let actor = actor.downcast_ref::<CudaIPCActorHandle>().unwrap();
            let gate = global.cuda_activity_gate(1).unwrap();
            let pid = install_and_probe(actor).await?;
            let generation = gate.device_generation();
            let mut attempts = 0;
            // Exercise the same trial boundary as TrainNet, with a real fatal
            // worker reply. The failed attempt must unwind before waiting.
            let next_pid = crate::stages::train_net::retry_training_after_cuda_reset(Some(Arc::clone(&gate)), || {
                attempts += 1;
                let first = attempts == 1;
                async move {
                    if first {
                        let actor = actor.clone();
                        submit_worker_request(async move { actor.invoke_stage(actor.worker_id()?, ENDPOINT.into(), Some(1.to_string()), input(RecoveryAction::Fatal), false).await }).await?;
                        panic!("fatal request succeeded");
                    }
                    install_and_probe(actor).await
                }
            }).await?;
            assert_eq!(attempts, 2);
            assert_eq!(gate.device_generation().0, generation.0 + 1);
            assert_ne!(pid, next_pid);
            assert!(!Path::new(&format!("/proc/{pid}")).exists());
            Ok::<_, ThoError>(())
        }).await?
    }, async |sock| {
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));
        CudaWorker::run_worker(sock, &ws, HashMap::from([(ENDPOINT.into(), Box::new(initialize) as InitFn)]), thoosa_common::tho_device::ThoDevice::from(1usize)).await
    })
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
                        async move { submit_worker_request(async move { actor.install_stage(worker, ENDPOINT.into(), Some(3.to_string()), input(RecoveryAction::DeferredFatal { started, release })).await }).await }
                    });
                    wait_for_file(&started).await;
                    caller.abort();
                    assert!(caller.await.unwrap_err().is_cancelled());
                    std::fs::write(&release, "fail now")?;
                    // The admitted runner job must report despite caller cancellation.
                    gate.wait_for_generation_end(worker.generation).await;
                } else if mode == "copy" {
                    use crate::copy_ctx::CudaActorHandle;
                    let response = actor.install_stage(worker, ENDPOINT.into(), Some(3.to_string()), input(RecoveryAction::Poison)).await?;
                    assert!(matches!(response.status, OpResult::Ok(_)));
                    let error = actor.copy(tch_result(Tensor::f_from_slice(&[1f32]))?.into(), 1.into()).await.unwrap_err();
                    assert!(matches!(error.what(), ThoError::ContextReset), "fatal direct copy did not report recovery: {error:#}");
                } else {
                    let actor = actor.clone();
                    let error = submit_worker_request(async move {
                        if mode == "install" {
                            actor.install_stage(worker, ENDPOINT.into(), Some(3.to_string()), input(RecoveryAction::Fatal)).await
                        } else {
                            actor.invoke_stage(worker, ENDPOINT.into(), Some(1.to_string()), input(if mode == "poison" { RecoveryAction::Poison } else { RecoveryAction::Fatal }), mode == "consume").await
                        }
                    }).await.unwrap_err();
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
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));
        CudaWorker::run_worker(sock, &ws, HashMap::from([(ENDPOINT.into(), Box::new(initialize) as InitFn)]), thoosa_common::tho_device::ThoDevice::from(1usize)).await
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
            let failed_actor = actor.clone();
            let _ = submit_worker_request(async move { failed_actor.invoke_stage(worker, ENDPOINT.into(), Some(1.to_string()), input(RecoveryAction::Fatal), false).await }).await;
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
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));
        CudaWorker::run_worker(sock, &ws, HashMap::from([(ENDPOINT.into(), Box::new(initialize) as InitFn)]), thoosa_common::tho_device::ThoDevice::from(1usize)).await
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
                    use thoosa_common::runner::sync_stage::ConsumeResult;
                    let result = global.get_runner(1).submit_consume(0, async {
                        let error = tokio::task::spawn_blocking(poison_and_synchronize).await?.unwrap_err();
                        Ok(ConsumeResult::<_, ()>::Failed { input: vec![7], error })
                    }, None).await?;
                    let ConsumeResult::Failed { input, error } = result else { panic!("poisoned consume succeeded") };
                    assert_eq!(input, vec![7]);
                    error
                }
                "copy" => {
                    let source = tch_result(Tensor::f_from_slice(&[1f32]))?;
                    assert!(tokio::task::spawn_blocking(poison_and_synchronize).await?.unwrap_err().needs_cuda_restart());
                    global.cuda_actor(1).copy(source.into(), 1.into()).await.unwrap_err()
                }
                _ => panic!("unknown local failure path"),
            };
            assert!(error.is_cuda_recovery_failed(), "fatal local execution must be terminal: {error:#}");
            let failure = tokio::time::timeout(Duration::from_secs(2), GlobalCtx::wait_for_process_failure()).await?;
            let ThoError::CudaRecoveryFailed(failure) = failure.what() else { panic!("lost terminal failure") };
            assert!(failure.original.needs_cuda_restart());
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
