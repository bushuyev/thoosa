use thoosa_common::tch_error::tch_result;
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
    let restore_failures = checkpoint.with_extension("restore-oom");
    let config: ThoOptimizers = match optimizer {
        0 => nn::Adam::default().amsgrad(true).into(),
        1 => nn::AdamW::default().amsgrad(true).into(),
        _ => nn::Sgd { momentum: 0.9, dampening: 0., wd: 0.01, nesterov: true }.into(),
    };
    let (step, ()) = TrainStep::new_local(device, move |(), device| async move {
        let config = ModConfig::new(config, move |vs| {
            let initialize = vs.variables().is_empty();
            // A file lets the same injected model-construction failure run in
            // the real worker process, after tensor import has completed.
            if !initialize && restore_failures.exists() {
                let remaining: usize = std::fs::read_to_string(&restore_failures)?.parse()?;
                if remaining > 0 {
                    std::fs::write(&restore_failures, (remaining - 1).to_string())?;
                    return Err(ThoError::oom().into());
                }
            }
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
    }, Arc::new(Box::new(|batch| batch) as TrainStepXFn<AnyDims>), |predicted, expected| {
        if expected.f_double_value(&[0, 0])? == -999.0 { return Err(ThoError::oom().into()); }
        Ok(predicted.f_mse_loss(expected, Reduction::Mean)?)
    }, (), checkpoint).await?;
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
            init: (),
        }
    } else {
        DynSyncStageWithStateEnum::Local(Arc::new(local_step(optimizer, 1.into(), checkpoint).await?))
    }))
}

fn input(train: bool) -> Result<TrainStepInput<AnyDims, AnyDims>, ThoError> {
    Ok(TrainStepInput::new(
        ThoTensor::new(debug_id!("train state x"), tch_result(tch_result(Tensor::f_from_slice(&[1f32, 2., 3., 2., -1., 1.]))?.f_reshape([2, 3]))?)?,
        None,
        ThoTensor::new(debug_id!("train state y"), tch_result(Tensor::f_zeros([2, 1], (Kind::Float, Device::Cpu)))?)?,
        train,
    ))
}

fn batch_parts(mut input: TrainStepInput<AnyDims, AnyDims>) -> Result<(ThoBatch<AnyDims>, Option<BatchLengths>, ThoBatch<AnyDims>), ThoError> {
    Ok((input.x.take().ok_or_else(|| ThoError::new("missing batch input"))?, input.lengths.take(), input.expected_y.take().ok_or_else(|| ThoError::new("missing batch target"))?))
}

async fn prepared_batch_round_trips() -> Result<(), ThoError> {
    let mut ctx = RunCtxImpl::new(1, "prepared batches");
    let pipeline = step(0, "pipeline".into(), PathBuf::from("/tmp/unused-pipeline")).await?;
    let reference = step(0, "pipeline-reference".into(), PathBuf::from("/tmp/unused-pipeline-reference")).await?;

    // A queued entry owns its registration even after the producer disappears.
    // Exercise the real registry and GPU/worker-reference offload, not a mock copy.
    let cache = Arc::new(TmpCached::new("queued_batch_offload").await);
    let queued = input(true)?.copy_to_device(1).await?;
    let id = queued.id().clone();
    let entry = cache.insert_guarded(queued);
    let wrapped = cache.get(&id).ok_or_else(|| ThoError::new("queued batch missing"))?;
    let cache_alive = Arc::downgrade(&cache);
    drop(cache);
    GlobalCtx::global().dump_all_to_cpu(&HashMap::new(), 1, "queued batch test").await?;
    assert!(wrapped.is_all_on_cpu().await, "queued batch must be visible to OOM offload");
    drop(wrapped);
    let actual = pipeline.run_once(entry, ctx.branch(false), 0).await?.loss;
    let expected = GlobalCtx::global().cuda_actor(1).run_stage(reference.clone(), input(true)?, 1, &ctx).await?.loss;
    assert!((actual - expected).abs() < 1e-6, "offloaded batch did not reload correctly");
    assert!(cache_alive.upgrade().is_none(), "consumed batch retained its temporary cache");

    for prefetch in [true, false] {
        for train in [true, false] {
            let source = futures_util::stream::iter(0..4).then(move |_| async move {
                // Label/final preparation also returns device-resident batches.
                batch_parts(input(train)?.copy_to_device(1).await?)
            }).boxed();
            let mut actual = vec![];
            pipeline.run_batches(source, ctx.branch(false), train, 32, prefetch, |loss| actual.push(loss)).await?;
            assert_eq!(actual.len(), 4);
            for actual in actual {
                let expected = GlobalCtx::global().cuda_actor(1).run_stage(reference.clone(), input(train)?, 1, &ctx).await?.loss;
                assert!((actual - expected).abs() < 1e-6, "prefetch={prefetch}, train={train}: batch order or update changed");
            }
        }
    }

    // A failed step must finish its one in-flight preparation before rollback,
    // and must never poll a third batch or train the speculative second batch.
    let saved = pipeline.save_epoch_checkpoint(1).await?;
    let drained = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut failed = input(true)?;
    failed.expected_y = Some(ThoTensor::new(debug_id!("prefetch OOM"), tch_result(Tensor::f_full([2, 1], -999.0, (Kind::Float, Device::Cpu)))?)?);
    let source = futures_util::stream::iter([batch_parts(failed)])
        .chain(futures_util::stream::once({
            let drained = drained.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                drained.store(true, std::sync::atomic::Ordering::SeqCst);
                batch_parts(input(true)?)
            }
        }))
        .chain(futures_util::stream::once(async { panic!("preparation continued after a failed epoch") }))
        .boxed();
    let error = pipeline.run_batches(source, ctx.branch(false), true, 32, true, |_| panic!("failed or speculative batch was committed")).await.unwrap_err();
    assert!(error.is_oom());
    assert!(drained.load(std::sync::atomic::Ordering::SeqCst), "rollback can race an undrained producer");
    pipeline.schedule_epoch_restore(saved, 1).await?;
    let actual = GlobalCtx::global().cuda_actor(1).run_stage(pipeline.clone(), input(false)?, 1, &ctx).await?.loss;
    let expected = GlobalCtx::global().cuda_actor(1).run_stage(reference, input(false)?, 1, &ctx).await?.loss;
    assert!((actual - expected).abs() < 1e-6, "failed epoch changed committed state");

    // Cancellation must drop the pending producer instead of leaving a spawned
    // task holding its captures (and potentially cached resources) indefinitely.
    let alive = Arc::new(());
    let producer = Arc::downgrade(&alive);
    let preparing = Arc::new(tokio::sync::Notify::new());
    let source = futures_util::stream::iter([batch_parts(input(true)?)])
        .chain(futures_util::stream::once({
            let preparing = preparing.clone();
            async move {
                let _alive = alive;
                preparing.notify_one();
                std::future::pending().await
            }
        })).boxed();
    let mut run = Box::pin(pipeline.run_batches(source, ctx.branch(false), true, 32, true, |_| {}));
    tokio::select! {
        _ = preparing.notified() => {},
        result = &mut run => panic!("pipeline finished before cancellation: {result:?}"),
    }
    assert!(producer.upgrade().is_some());
    drop(run);
    assert!(producer.upgrade().is_none(), "cancelled pipeline retained its producer");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_prepared_batches_local() -> Result<(), ThoError> {
    let global = GlobalCtx::init_devices(2)?;
    let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
    let result = tokio::time::timeout(std::time::Duration::from_secs(30), prepared_batch_round_trips()).await?;
    global.shutdown().await;
    result
}

#[test]
fn test_prepared_batches_ipc() -> Result<(), ThoError> {
    crate::ipc::tests::run_ipc_test(module_path!(), async {
        tokio::time::timeout(std::time::Duration::from_secs(30), prepared_batch_round_trips()).await?
    }, async |sock| {
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));
        CudaWorker::run_worker(sock, &ws, HashMap::from([(ENDPOINT.into(), Box::new(endpoint_init) as InitFn)]), thoosa_common::tho_device::ThoDevice::from(1usize)).await
    })
}

#[test]
fn test_prepared_batches_retry_after_worker_reset_ipc() -> Result<(), ThoError> {
    crate::ipc::tests::run_ipc_test(module_path!(), async {
        tokio::time::timeout(std::time::Duration::from_secs(60), async {
            let gate = GlobalCtx::global().cuda_activity_gate(1).unwrap();
            let generation = gate.device_generation();
            let preparing = Arc::new(tokio::sync::Notify::new());
            let attempts = std::sync::atomic::AtomicUsize::new(0);
            let run = crate::stages::train_net::retry_training_after_cuda_reset(Some(Arc::clone(&gate)), || {
                let attempt = attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let preparing = Arc::clone(&preparing);
                async move {
                    let pipeline = step(0, format!("reset-pipeline-{attempt}"), PathBuf::from("/tmp/unused-reset-pipeline")).await?;
                    let source = futures_util::stream::iter(0..3).then(move |index| {
                        let preparing = Arc::clone(&preparing);
                        async move {
                            // Suspend the producer while run_batches owns its
                            // first device-resident entry and an in-flight step.
                            // Reset must discard both, then rebuild the pipeline.
                            if attempt == 0 && index == 1 {
                                preparing.notify_one();
                                std::future::pending::<()>().await;
                            }
                            batch_parts(input(true)?.copy_to_device(1).await?)
                        }
                    }).boxed();
                    let mut losses = vec![];
                    pipeline.run_batches(source, RunCtxImpl::new(1, "reset batches"), true, 32, true, |loss| losses.push(loss)).await?;
                    Ok(losses)
                }
            });
            let reset = async {
                preparing.notified().await;
                GlobalCtx::global().cuda_actor(1).restart().await
            };
            let (losses, reset) = tokio::join!(run, reset);
            reset?;
            let losses = losses?;
            assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 2);
            assert_eq!(gate.device_generation().0, generation.0 + 1);
            let reference = step(0, "reset-reference".into(), PathBuf::from("/tmp/unused-reset-reference")).await?;
            assert_eq!(losses.len(), 3);
            for actual in losses {
                let expected = GlobalCtx::global().cuda_actor(1).run_stage(Arc::clone(&reference), input(true)?, 1, &RunCtxImpl::new(1, "reset reference")).await?.loss;
                assert!((actual - expected).abs() < 1e-6, "replacement trial changed batch order or optimizer updates");
            }
            Ok(())
        }).await?
    }, async |sock| {
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));
        CudaWorker::run_worker(sock, &ws, HashMap::from([(ENDPOINT.into(), Box::new(endpoint_init) as InitFn)]), thoosa_common::tho_device::ThoDevice::from(1usize)).await
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_prepared_batches_overlap_and_bound() -> Result<(), ThoError> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let global = GlobalCtx::init_devices(2)?;
    let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
    let started = Arc::new(tokio::sync::Notify::new());
    let prepared = Arc::new(tokio::sync::Notify::new());
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let release_rx = std::sync::Mutex::new(release_rx);
    let mut local = local_step::<RunCtxImpl>(0, 1.into(), PathBuf::from("/tmp/unused-overlap")).await?;
    let started_for_step = started.clone();
    let calls = AtomicUsize::new(0);
    // Hold the first real synchronous forward pass until preparation progresses.
    local.x_f = Arc::new(Box::new(move |batch| {
        if calls.fetch_add(1, Ordering::SeqCst) == 0 {
            started_for_step.notify_one();
            release_rx.lock().unwrap().recv_timeout(std::time::Duration::from_secs(10)).expect("next batch did not overlap training");
        }
        batch
    }));
    let pipeline = Arc::new(DynSyncStageWithStateEnum::Local(Arc::new(local)));
    let completed = Arc::new(AtomicUsize::new(0));
    let loaded = Arc::new(AtomicUsize::new(0));
    let source = futures_util::stream::iter(0..4).then({
        let completed = completed.clone();
        let loaded = loaded.clone();
        let prepared = prepared.clone();
        let started = started.clone();
        move |index| {
            let completed = completed.clone();
            let loaded = loaded.clone();
            let prepared = prepared.clone();
            let started = started.clone();
            async move {
                assert!(index <= completed.load(Ordering::SeqCst) + 1, "prepared more than one batch ahead");
                // Require preparation to run during the blocked synchronous
                // invocation, not merely before the runner starts that invocation.
                if index == 1 { started.notified().await; }
                let batch = batch_parts(input(true)?)?;
                loaded.fetch_add(1, Ordering::SeqCst);
                if index == 1 { prepared.notify_one(); }
                Ok(batch)
            }
        }
    }).boxed();
    let observe = async {
        prepared.notified().await;
        assert_eq!(completed.load(Ordering::SeqCst), 0);
        assert_eq!(loaded.load(Ordering::SeqCst), 2);
        release_tx.send(()).expect("training receiver disappeared");
    };
    let (result, ()) = tokio::time::timeout(std::time::Duration::from_secs(15), async {
        tokio::join!(pipeline.run_batches(source, RunCtxImpl::new(1, "overlap"), true, 32, true, |_| { completed.fetch_add(1, Ordering::SeqCst); }), observe)
    }).await?;
    result?;
    assert_eq!(completed.load(Ordering::SeqCst), 4);
    global.shutdown().await;
    Ok(())
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
            // Save before/after lazy optimizer initialization, then discard both
            // successful updates and a failed invocation from the next epoch.
            let was_loaded = restored.has_gpu_data(1).await;
            let saved = restored.save_epoch_checkpoint(1).await?;
            assert_eq!(restored.has_gpu_data(1).await, was_loaded, "saving must neither purge nor reload training state");
            for _ in 0..2 { actor.run_stage(restored.clone(), input(true)?, 1, &ctx).await?; }
            // Capture multiple later epochs while retaining the recovery point.
            // Staging-buffer reuse must not overwrite saved parameters/moments.
            let newer = restored.save_epoch_checkpoint(1).await?;
            let newest = restored.save_epoch_checkpoint(1).await?;
            assert_eq!(newer.tensors.len(), newest.tensors.len());
            for (key, tensor) in &newer.tensors { assert!(tch_result(tensor.as_ref().f_equal(newest.tensors[key].as_ref().as_ref()))?); }
            let mut failed = input(true)?;
            failed.expected_y = Some(ThoTensor::new(debug_id!("failed epoch target"), tch_result(Tensor::f_full([2, 1], -999.0, (Kind::Float, Device::Cpu)))?)?);
            let cache = Arc::new(TmpCached::new("failed_epoch_batch").await);
            let error = restored.run_once(cache.insert_guarded(failed), RunCtxImpl::new(1, "epoch failure"), 0).await.unwrap_err();
            assert!(error.is_oom());
            restored.schedule_epoch_restore(Arc::clone(&saved), 1).await?;
            assert!(!restored.has_gpu_data(1).await, "scheduling rollback must not allocate GPU state");
            let pending = restored.save_epoch_checkpoint(1).await?;
            assert_eq!(serde_json::to_value(pending.info()?.scalars)?, serde_json::to_value(saved.info()?.scalars)?, "deferred saves must preserve optimizer metadata");
            assert!(!restored.has_gpu_data(1).await, "saving deferred state must not upload it");
            if let DynSyncStageWithStateEnum::Local(stage) = restored.as_ref() {
                // Model a copy failure after the first restored tensor. Purging
                // must release partial imports without requiring a live optimizer.
                let (key, tensor) = saved.tensors.iter().next().unwrap();
                let source = tch_result(tensor.as_ref().f_alias())?;
                let partial = copy_state_tensor(source, 1).await?;
                stage.train_module.lock().await.module.set_state_tensor(key, partial, 1)?;
                assert!(restored.has_gpu_data(1).await);
            }
            // The checkpoint survives a restore and unrelated CPU purge.
            restored.purge_gpu(None, 1, "checkpoint retained through purge").await?;
            assert!(!restored.has_gpu_data(1).await);
            restored.schedule_epoch_restore(saved, 1).await?;
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
        split.checkpoint(RunCtxImpl::new(1, "checkpoint"), 32).await?;
        let saved_loss = checkpoint_loss(checkpoint.path())?;
        assert!((saved_loss - actual).abs() < 1e-5, "checkpoint does not match final validation: {saved_loss} vs {actual}");
    }
    Ok(())
}

async fn restore_oom_round_trip() -> Result<(), ThoError> {
    let global = GlobalCtx::global();
    let checkpoint = thoosa_shared::tmp_file::TmpFile::new(format!("/tmp/train-restore-oom-{}.safetensors", std::process::id()));
    let failure_file = thoosa_shared::tmp_file::TmpFile::new(checkpoint.path().with_extension("restore-oom"));
    let restored = step(0, "restore-oom".into(), checkpoint.path().to_owned()).await?;
    let reference = step(0, "restore-oom-reference".into(), PathBuf::from("/tmp/unused-restore-oom-reference")).await?;
    let registered = global.register_cache((restored.stage_id(), restored.clone() as Arc<dyn CachePurger>));
    let reference_registered = global.register_cache((reference.stage_id(), reference.clone() as Arc<dyn CachePurger>));
    let mut ctx = RunCtxImpl::new(1, "restore OOM");
    let cache = Arc::new(TmpCached::new("restore OOM input").await);
    restored.run_once(cache.insert_guarded(input(true)?), ctx.branch(false), 0).await?;
    reference.run_once(cache.insert_guarded(input(true)?), ctx.branch(false), 0).await?;
    let saved = restored.save_epoch_checkpoint(1).await?;
    restored.schedule_epoch_restore(Arc::clone(&saved), 1).await?;
    // The initial and max-parallel attempts fail, forcing DumpToCpu. If the
    // epoch's exclusive cap were reset to 32, this would succeed without dumping.
    std::fs::write(failure_file.path(), "2")?;
    let actual = restored.run_once(cache.insert_guarded(input(true)?), ctx.branch(false), 0).await?.loss;
    assert_eq!(std::fs::read_to_string(failure_file.path())?, "0");
    assert!(!reference.has_gpu_data(1).await, "reconstruction OOM did not reach DumpToCpu");
    let expected = reference.run_once(cache.insert_guarded(input(true)?), ctx.branch(false), 0).await?.loss;
    assert!((actual - expected).abs() < 1e-6, "restoration retries changed the training update");
    let actual = restored.run_once(cache.insert_guarded(input(false)?), ctx.branch(false), 0).await?.loss;
    let expected = reference.run_once(cache.insert_guarded(input(false)?), ctx.branch(false), 0).await?.loss;
    assert!((actual - expected).abs() < 1e-6, "restoration retries applied an update twice");
    // The checkpoint command itself can be the first invocation after rollback.
    restored.schedule_epoch_restore(saved, 1).await?;
    std::fs::write(failure_file.path(), "2")?;
    restored.checkpoint(ctx, 0).await?;
    assert!(checkpoint.path().exists());
    assert_eq!(std::fs::read_to_string(failure_file.path())?, "0");
    global.remove_cache(registered.key());
    global.remove_cache(reference_registered.key());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_train_restore_oom_local() -> Result<(), ThoError> {
    let global = GlobalCtx::init_devices(2)?;
    let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
    let result = tokio::time::timeout(std::time::Duration::from_secs(90), restore_oom_round_trip()).await?;
    global.shutdown().await;
    result
}

#[test]
fn test_train_restore_oom_ipc() -> Result<(), ThoError> {
    crate::ipc::tests::run_ipc_test(module_path!(), async {
        tokio::time::timeout(std::time::Duration::from_secs(90), restore_oom_round_trip()).await?
    }, async |sock| {
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));
        CudaWorker::run_worker(sock, &ws, HashMap::from([(ENDPOINT.into(), Box::new(endpoint_init) as InitFn)]), 1usize.into()).await
    })
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
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));
        CudaWorker::run_worker(sock, &ws, HashMap::from([(ENDPOINT.into(), Box::new(endpoint_init) as InitFn)]), thoosa_common::tho_device::ThoDevice::from(1usize)).await
    })
}

fn uneven_input(train: bool) -> Result<TrainStepInput<AnyDims, AnyDims>, ThoError> {
    Ok(TrainStepInput::new(
        ThoTensor::new(debug_id!("transaction x"), tch_result(tch_result(Tensor::f_from_slice(&[1f32, 2., 3., 2., -1., 1., 0.5, 0.25, -0.5]))?.f_reshape([3, 3]))?)?,
        None,
        ThoTensor::new(debug_id!("transaction y"), tch_result(tch_result(Tensor::f_from_slice(&[0f32, 0., 1.]))?.f_reshape([3, 1]))?)?,
        train,
    ))
}

fn checkpoint_loss(path: &std::path::Path) -> Result<f32, ThoError> {
    let variables = tch_result(Tensor::read_safetensors(path))?;
    let weight = variables.iter().find(|(name, _)| name.ends_with("weight")).ok_or_else(|| ThoError::new("missing checkpoint weight"))?;
    let bias = variables.iter().find(|(name, _)| name.ends_with("bias")).ok_or_else(|| ThoError::new("missing checkpoint bias"))?;
    let input = uneven_input(false)?;
    let predicted = tch_result(input.x_ref()?.as_ref().f_linear(&weight.1, Some(&bias.1)))?;
    Ok(tch_result(tch_result(predicted.f_mse_loss(&input.expected_y_ref()?.as_ref(), Reduction::Mean))?.f_double_value(&[]))? as f32)
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
            let mut state = MutexGuard::map(split.train_module.lock().await, |state| &mut state.module);
            let saved_info = state.state_info()?;
            let mut saved_tensors = Vec::new();
            for tensor in &saved_info.tensors {
                let key = tensor.tensor_key.to_tensor_key();
                let tensor = state.state_tensor(&key)?;
                saved_tensors.push((key, tch_result(tensor.f_to_device_(Device::Cpu, tensor.kind(), false, true))?));
            }
            FAIL_LAST_CHUNK.store(true, Ordering::SeqCst);
            let error = split.clone().do_run_with_state(selected.as_ref(), &mut state, &ctx).unwrap_err();
            assert!(error.is_oom());
            assert!(!error.is_retryable_oom());
            state.reset_state(saved_info)?;
            for (key, tensor) in saved_tensors { state.set_state_tensor(&key, tensor, 0)?; }
            state.prepare_state(0)?;
            let actual = split.clone().do_run_with_state(selected.as_ref(), &mut state, &ctx)?.loss;
            let mut full_state = MutexGuard::map(full.train_module.lock().await, |state| &mut state.module);
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

        }
        let mut state = MutexGuard::map(split.train_module.lock().await, |state| &mut state.module);
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
