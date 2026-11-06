use thoosa_common::tch_error::tch_result;
use super::*;
use crate::motifs_store::MockMotifsStoreBuilder;
use crate::motifs_ws::{MotifsWorkspace, MotifsWsMockBuilder};
use std::sync::atomic::AtomicUsize;
use std::time::Duration;
use thoosa_common::ipc::cuda_wokrer::{CudaWorker, InitFn};
use thoosa_common::nets::nodes::ThoClosureModule;
use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::stages::freq_filter::{SYNC_FREQ_FILTER_ENDPOINT, SyncFreqFilter, SyncFreqFilterIpcInit};
use thoosa_common::transforms::filters::AverageFilterParams;
use thoosa_properties_map::ToHashmap;
use thoosa_store::RunConfigRow;

fn workspace(reject_duplicate_forward: bool) -> Arc<RwLock<Box<dyn MotifsWorkspace>>> {
    let entries = [(1u32, 4, 10.), (2u32, 6, 20.)].into_iter().map(|(motif, width, bias)| {
        let params: Box<dyn Fn() -> MotifNetParams + Send + Sync> = Box::new(move || MotifNetParams::new_lin(width, 1, 1));
        let module: Box<dyn Fn() -> ThoModuleWithStore<Batch<Slice>, MotifProbability> + Send + Sync> = Box::new(move || {
            // The same counter survives device moves; a duplicate inference fails in the worker too.
            let calls = Arc::new(AtomicUsize::new(0));
            let factory = Arc::new(move |_path: &nn::Path| {
                let calls = Arc::clone(&calls);
                let module = ThoClosureModule::new(Arc::new(Box::new(move |input: &Tensor, _: Option<&BatchLengths>| {
                    let count = calls.fetch_add(1, Relaxed);
                    if reject_duplicate_forward && count != 0 { return Err(anyhow::anyhow!("duplicate motif forward")); }
                    Ok(input.f_mean_dim(1, true, Float)?.f_add_scalar(bias)?)
                })));
                Ok(Box::new(module) as Box<dyn ThoModule<Batch<Slice>, MotifProbability>>)
            });
            let vs = VarStore::new(Cpu);
            let module = factory(&vs.root()).expect("test module construction");
            ThoModuleWithStore::with_factory(mono_key!(motif, "fixture"), module, vs, Some(factory))
        });
        (PathBuf::from(format!("0/{motif}_100")), params, module)
    }).collect();
    Arc::new(RwLock::new(MotifsWsMockBuilder::new(None, 1, 100, 96).with_load_motif_net(entries).build()))
}

fn store() -> Arc<dyn MotifsStore> {
    MockMotifsStoreBuilder::new()
        .with_get_motif_meta([(1, Ok((0, 101))), (2, Ok((0, 102)))])
        .with_run_config_descr([101, 102].map(|id| (id, Ok(RunConfigRow { id, r#type: "FindMotifsRunConfig".into(), descr: "cache regression".into(), run_on: "2026-10-15".into() }))))
        .with_list_run_config_values([101, 102].map(|id| {
            let filters = (id == 102).then(|| vec![InputFilter::AverageFilter(AverageFilterParams { window: 5, step: 2 })]);
            (id, Ok(FindMotifsRunConfig::new_with_running_mean(Some(id), 1, 4, filters, None).to_properties_iter(None).collect()))
        }))
        .build()
}

fn trained_nets() -> Vec<TrainedNet> {
    // Model ids deliberately reverse motif order, and the two models use different filters and widths.
    vec![
        TrainedNet { id: 10, run_config_id: 100, net_path: "0/2_100".into(), loss: 0.1 },
        TrainedNet { id: 90, run_config_id: 100, net_path: "0/1_100".into(), loss: 0.2 },
    ]
}

fn input(name: &str) -> Result<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, ThoError> {
    Ok(Arc::new(DevicesWrap::wrap(ThoTensor::new(mono_key!(name.to_owned(), "input").into(), tch_result(tch_result(Tensor::f_arange(96, (Float, Cpu)))?.f_view([1, 96]))?)?)))
}

async fn cached_sets() -> Result<(), ThoError> {
    let ws = workspace(true);
    let store = store();
    let mut ctx = RunCtxImpl::new(1, "motif cache regression");
    let all = Arc::new(ApplyMotifNets::<RunCtxImpl>::from_trained_nets(&ws, &store, ctx.branch(true), 2, None, trained_nets(), false).await?);
    let subset = Arc::new(ApplyMotifNets::<RunCtxImpl>::from_trained_nets(&ws, &store, ctx.branch(true), 2, None, vec![trained_nets().remove(1)], false).await?);
    let input = input("shared motif input")?;
    // This is the path used by label models: aggregate run is uncached, individual inference is cached.
    let (first, second) = tokio::try_join!(all.clone().run(input.clone(), ctx.branch(true)), subset.clone().run(input.clone(), ctx.branch(true)))?;
    assert_eq!(first.id(), &all.out_id(input.id()));
    assert_eq!(second.id(), &subset.out_id(input.id()));
    assert_ne!(first.id(), second.id());
    let raw_stage = &all.apply_motif_nets[1].2;
    let cached = raw_stage.clone().run_cached(input.clone(), ctx.branch(true)).await?;
    let other = subset.apply_motif_nets[0].2.clone().run_cached(input.clone(), ctx.branch(true)).await?;
    assert!(Arc::ptr_eq(&cached, &other), "different motif sets must share the per-model cache entry");
    assert_eq!(cached.id(), &raw_stage.out_id(input.id()));
    let filter = &all.apply_motif_nets[0].1.as_ref().expect("filtered motif").1;
    let filtered = filter.clone().run_cached(input.clone(), ctx.branch(true)).await?;
    let filtered_probs = all.apply_motif_nets[0].2.clone().run_cached(filtered.clone(), ctx.branch(true)).await?;
    assert_eq!(filtered_probs.id(), &all.apply_motif_nets[0].2.out_id(filtered.id()));
    let lengths = vec![47, (filtered.size().dim::<1>() - 6) / 2 + 1];
    assert_eq!(first.extra().await?, lengths);
    {
        let stack = first.borrow_ref(0).await?;
        let raw = cached.borrow_ref(0).await?;
        let filtered = filtered.borrow_ref(0).await?;
        let expected_raw = tch_result(tch_result(tch_result(tch_result(Tensor::f_arange(96, (Float, Cpu)))?.f_unfold(0, 4, 2))?.f_mean_dim(1, false, Float))?.f_add_scalar(10.))?;
        let expected_filtered = tch_result(tch_result(tch_result(tch_result(filtered.as_ref().local()?.f_get(0))?.f_unfold(0, 6, 2))?.f_mean_dim(1, false, Float))?.f_add_scalar(20.))?;
        assert!(tch_result(raw.as_ref().local()?.f_allclose(&expected_raw, 1e-6, 1e-6, false))?);
        assert!(tch_result(tch_result(stack.as_ref().data().local()?.f_get(0))?.f_allclose(&expected_raw, 1e-6, 1e-6, false))?, "row order must follow motif ids, not trained-net ids");
        assert!(tch_result(tch_result(stack.as_ref().data().local()?.f_get(1))?.f_allclose(&stretch(&expected_filtered, 47, 0)?, 1e-6, 1e-6, false))?, "the second motif must use its filtered input");
    }
    // Stacking and GPU offload must leave cached probabilities usable in both backends.
    GlobalCtx::global().dump_all_to_cpu(1, "cached motif regression").await?;
    let again = all.clone().run(input.clone(), ctx.branch(true)).await?;
    let after = raw_stage.clone().run_cached(input.clone(), ctx.branch(true)).await?;
    assert!(Arc::ptr_eq(&cached, &after));
    assert_eq!(again.extra().await?, lengths);

    // The output cache and sync instance cache must both distinguish inference settings.
    let path = PathBuf::from("0/1_100");
    let different_step = ApplyMotifNet::<RunCtxImpl>::new(ApplyMotifNetInit::new(90, 3, path.clone(), 0, 1, false, None), ctx.branch(true), &ws).await?;
    let quantized = ApplyMotifNet::<RunCtxImpl>::new(ApplyMotifNetInit::new(90, 2, path, 0, 1, true, None), ctx.branch(true), &ws).await?;
    assert!(!Arc::ptr_eq(&raw_stage.sync_stage, &different_step.sync_stage));
    assert!(!Arc::ptr_eq(&raw_stage.sync_stage, &quantized.sync_stage));
    let stepped = different_step.clone().run_cached(input.clone(), ctx.branch(true)).await?;
    let quantized_output = quantized.clone().run_cached(input.clone(), ctx.branch(true)).await?;
    assert_eq!(stepped.borrow_ref(0).await?.as_ref().size()[0], 31);
    assert_ne!(cached.id(), stepped.id());
    assert_ne!(cached.id(), quantized_output.id());
    assert_eq!(stepped.id(), &different_step.out_id(input.id()));
    assert_eq!(quantized_output.id(), &quantized.out_id(input.id()));
    Ok(())
}

async fn filtered_oom() -> Result<(), ThoError> {
    let ws = workspace(false);
    let store = store();
    let mut ctx = RunCtxImpl::new(1, "filtered motif OOM regression");
    let normal = Arc::new(ApplyMotifNets::<RunCtxImpl>::from_trained_nets(&ws, &store, ctx.branch(true), 3, None, trained_nets(), false).await?);
    let input = input("filtered OOM input")?;
    let expected = normal.clone().run(input.clone(), ctx.branch(true)).await?;
    // A separate trained-model identity prevents either cache from masking the forced split path.
    let nets = trained_nets().into_iter().map(|mut net| { net.id += 1000; net }).collect();
    let split = Arc::new(ApplyMotifNets::<RunCtxImpl>::from_trained_nets(&ws, &store, ctx.branch(true), 3, Some(24), nets, false).await?);
    let actual = split.clone().run(input.clone(), ctx.branch(true)).await?;
    assert_eq!(actual.id(), &split.out_id(input.id()));
    assert_eq!(actual.extra().await?, expected.extra().await?);
    {
        let expected = expected.borrow_ref(0).await?;
        let actual = actual.borrow_ref(0).await?;
        assert!(tch_result(actual.as_ref().data().local()?.f_allclose(&*expected.as_ref().data().local()?, 1e-6, 1e-6, false))?, "filtered and unfiltered split outputs must match unsplit inference");
    }
    for (_, filter, stage) in &split.apply_motif_nets {
        let filtered = if let Some((_, filter)) = filter { filter.clone().run_cached(input.clone(), ctx.branch(true)).await? } else { input.clone() };
        let output = stage.clone().run_cached(filtered.clone(), ctx.branch(true)).await?;
        assert_eq!(output.id(), &stage.out_id(filtered.id()), "OOM assembly must use the cache key as its output identity");
    }
    Ok(())
}

fn filter_init<'a>(_id: Option<String>, init: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn MotifsWorkspace>>>, _tensors: &'a AllTensors, device: &'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let init: Box<dyn std::any::Any> = init;
        let mut init = *init.downcast::<SyncFreqFilterIpcInit>().map_err(|_| ThoError::new("invalid filter fixture initializer"))?;
        init.tho_device_idx = device.tho_device_idx();
        Ok(Box::new(WorkerWrapperWithState::new(Arc::new(SyncFreqFilter::<WokerRunCtx>::new_local(init)?), PhantomData::<JsonIpcCodec>)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}

async fn worker(sock: std::ffi::OsString, reject_duplicate_forward: bool) -> anyhow::Result<()> {
    CudaWorker::run_worker(sock, &workspace(reject_duplicate_forward), HashMap::from([
        (APPLY_MOTIF_NET_ENDPOINT.into(), Box::new(apply_motif_net_endpoint_init) as InitFn<dyn MotifsWorkspace>),
        (STACK_APPLIED_MOTIF_NETS_ENDPOINT.into(), Box::new(stack_applied_motif_nets_init) as InitFn<dyn MotifsWorkspace>),
        (SYNC_FREQ_FILTER_ENDPOINT.into(), Box::new(filter_init) as InitFn<dyn MotifsWorkspace>),
    ]), thoosa_common::tho_device::ThoDevice::from(1usize)).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_cached_motif_sets_local() -> Result<(), ThoError> {
    let global = GlobalCtx::init_devices(2)?;
    let _mock = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
    let result = tokio::time::timeout(Duration::from_secs(90), cached_sets()).await?;
    global.shutdown().await;
    result
}

#[test]
fn test_cached_motif_sets_ipc() -> Result<(), ThoError> {
    thoosa_common::ipc::tests::run_ipc_test(module_path!(), async { tokio::time::timeout(Duration::from_secs(90), cached_sets()).await? }, async |sock| worker(sock, true).await)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_cached_motif_filtered_oom_local() -> Result<(), ThoError> {
    let global = GlobalCtx::init_devices(2)?;
    let _mock = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
    let result = tokio::time::timeout(Duration::from_secs(90), filtered_oom()).await?;
    global.shutdown().await;
    result
}

#[test]
fn test_cached_motif_filtered_oom_ipc() -> Result<(), ThoError> {
    thoosa_common::ipc::tests::run_ipc_test(module_path!(), async { tokio::time::timeout(Duration::from_secs(90), filtered_oom()).await? }, async |sock| worker(sock, false).await)
}
