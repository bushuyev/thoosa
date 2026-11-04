use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use thoosa_common::tch_error::tch_result;
use super::*;
use crate::motifs_ws::{MotifsWorkspace, MotifsWsMockBuilder};
use std::time::Duration;
use thoosa_common::ipc::cuda_wokrer::{CudaWorker, InitFn};
use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
use thoosa_common::runner::run_ctx::RunCtxImpl;

async fn request(resource: u32, models: &[u32], retry: bool) -> Result<StackAppliedLabelNetsInputWrapper, ThoError> {
    let cache = TmpCached::new("label stack request").await;
    let order = models.iter().enumerate().map(|(index, model)| (SynKey::new([mono_key!(*model, "model"), mono_key!(resource, "resource")]), index)).collect::<HashMap<_, _>>();
    // Temporary-cache iteration order is arbitrary; the request must determine the output order.
    for (id, index) in &order {
        cache.insert(ThoTensor::new(id.clone(), tch_result(tch_result(Tensor::f_from_slice(&[resource as f32 * 10. + models[*index] as f32]))?.f_view([1, 1]))?)?);
    }
    let input = StackAppliedLabelNetsInputWrapper::new(mono_key!(resource, "aggregate").into(), order, cache);
    if !retry { return Ok(input); }
    let cache_key = input.tmp_cache.cache_key().clone();
    let input_ids = input.tmp_cache.ids().into_iter().collect::<HashSet<_>>();
    // Exercise the same ownership handoff used after a failed consume, without allocating until real OOM.
    let device = if GlobalCtx::global().is_ipc() { 1 } else { 0 };
    let failed: ConsumeResult<StackAppliedLabelNetsInput, ()> = input.consume(device, RunCtxImpl::new(device, "retry handoff"), |input, _| ConsumeResult::Failed { input, error: ThoError::oom() }).await;
    let ConsumeResult::Failed { input, .. } = failed else { panic!("injected consume failure must retain its input") };
    // Only worker tensor references are serializable. Exercise their wire
    // round trip in IPC mode and the owned tensor handoff in local mode.
    let input = if device == 1 { JsonIpcCodec::decode::<StackAppliedLabelNetsInput>(&JsonIpcCodec::encode(&input)?)? } else { input };
    let recovered = StackAppliedLabelNetsInputWrapper::burp(input).await;
    assert_eq!(recovered.tmp_cache.cache_key(), &cache_key);
    assert_eq!(recovered.tmp_cache.ids().into_iter().collect::<HashSet<_>>(), input_ids);
    Ok(recovered)
}

async fn run_request(stage: Arc<DynSyncStageEnum<StackAppliedLabelNets<RunCtxImpl>>>, resource: u32, models: &[u32], retry: bool) -> Result<(), ThoError> {
    let input = request(resource, models, retry).await?;
    let expected_id = input.id.clone();
    let handler = OOMHandler::<NoInputAdjuster, NoStageAdjuster, NoRunCtxAdjuster, ()>::new(vec![], None);
    let output = handler.consume(&stage, input, RunCtxImpl::new(1, "reusable label stack"), NoContexts).await?.into_inner();
    assert_eq!(output.id(), &expected_id);
    let output = output.copy_to_device(0).await?;
    let actual = tch_result(Vec::<f32>::try_from(&tch_result(output.local()?.f_view([-1]))?))?;
    let expected = models.iter().map(|model| resource as f32 * 10. + *model as f32).collect::<Vec<_>>();
    assert_eq!(actual, expected, "resource-specific membership and order must survive IPC and retries");
    Ok(())
}

async fn reuse_stack() -> Result<(), ThoError> {
    let (first, second) = tokio::try_join!(StackAppliedLabelNets::<RunCtxImpl>::new(1.into()), StackAppliedLabelNets::<RunCtxImpl>::new(1.into()))?;
    assert!(Arc::ptr_eq(&first, &second), "concurrent construction must share one installed stage");
    tokio::try_join!(
        run_request(first.clone(), 11, &[9, 1, 5], false),
        run_request(second, 12, &[5, 9], true),
    )?;
    // Dropping an invocation must not uninstall the endpoint needed by later resources.
    let later = StackAppliedLabelNets::<RunCtxImpl>::new(1.into()).await?;
    assert!(Arc::ptr_eq(&first, &later));
    run_request(later.clone(), 13, &[1, 5, 9], false).await?;
    run_request(later, 14, &[], false).await
}

fn counted_init<'a>(stage_id: Option<String>, init: Box<dyn WorkerInput>, ws: Arc<RwLock<Box<dyn MotifsWorkspace>>>, tensors: &'a AllTensors, device: &'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        // Count actual worker-side installs so an accidentally uncached constructor fails the IPC test.
        static INSTALLS: AtomicU32 = AtomicU32::new(0);
        if INSTALLS.fetch_add(1, Relaxed) != 0 { return Err(ThoError::new("label stack endpoint installed more than once")); }
        stack_applied_label_nets_init(stage_id, init, ws, tensors, device).await
    }.boxed()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_label_stack_reuses_stage_local() -> Result<(), ThoError> {
    let global = GlobalCtx::init_devices(2)?;
    let _mock = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
    let result = tokio::time::timeout(Duration::from_secs(60), reuse_stack()).await?;
    global.shutdown().await;
    result
}

#[test]
fn test_label_stack_reuses_stage_ipc() -> Result<(), ThoError> {
    thoosa_common::ipc::tests::run_ipc_test(module_path!(), async {
        tokio::time::timeout(Duration::from_secs(60), reuse_stack()).await?
    }, async |sock| {
        let ws = Arc::new(RwLock::new(MotifsWsMockBuilder::new(None, 1, 100, 16).build()));
        CudaWorker::run_worker(sock, &ws, HashMap::from([
            (STACK_APPLIED_LABEL_NETS_ENDPOINT.into(), Box::new(counted_init) as InitFn<dyn MotifsWorkspace>),
        ]), thoosa_common::tho_device::ThoDevice::from(1usize)).await
    })
}
