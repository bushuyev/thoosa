use thoosa_common::tch_error::tch_result;
use super::*;
use crate::motifs_ws::MotifsWsMockBuilder;
use crate::stages::apply_motif_nets::{ApplyMotifNetInit, ApplyMotifNetInput, ApplyMotifNetInputWrapper, apply_motif_net_endpoint_init};
use crate::train_nets::{LabelNetParams, LabelNetTypeParams, MotifNetParams};
use thoosa_common::ipc::cuda_wokrer::{CudaWorker, InitFn};
use thoosa_common::ipc::worker_msg::OpResult;
use thoosa_common::nets::common_nets::LinLenNetParams;
use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
use thoosa_common::runner::run_ctx::RunCtxImpl;
use std::time::Duration;
use tch::{Device, nn::VarStore};

fn motif_params() -> MotifNetParams { MotifNetParams::new_lin(4, 1, 1) }
fn label_params() -> LabelNetParams { LabelNetParams::new(2, LabelNetTypeParams::LinNetParams(LinLenNetParams::new(8, 1))) }

fn motif_module() -> anyhow::Result<ThoModuleWithStore<Batch<Slice>, MotifProbability>> {
    let factory = Arc::new(|path: &tch::nn::Path| motif_params().motif_net(None)?.build_tch_module::<Batch<Slice>, MotifProbability>(path));
    let mut vs = VarStore::new(Device::Cuda(0));
    let module = factory(&vs.root())?;
    vs.freeze();
    Ok(ThoModuleWithStore::with_factory(mono_key!("motif state", "test"), module, vs, Some(factory)))
}

fn label_module(fail_restore: bool) -> anyhow::Result<ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>>> {
    let attempts = std::sync::atomic::AtomicUsize::new(0);
    let factory = Arc::new(move |path: &tch::nn::Path| {
        let attempt = attempts.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Fail after all uploads reached the worker. Both a direct retry and
        // an OOM purge must retain the parent's original CPU backup.
        if fail_restore && matches!(attempt, 1 | 2) { return Err(ThoError::oom().into()); }
        label_params().label_net()?.build_tch_module::<ChannelsMotifsData, Batch<LabelProbability>>(path)
    });
    let mut vs = VarStore::new(Device::Cuda(0));
    let module = factory(&vs.root())?;
    vs.freeze();
    Ok(ThoModuleWithStore::with_factory(mono_key!("label state", "test"), module, vs, Some(factory)))
}

// Validate protocol guards on the actual stage before running the local cycle.
// Restoring in reverse order also ensures names, not upload order, bind weights.
async fn restore_guards<S: StageWithState + thoosa_common::ipc::state_transfer::StateTransfer + CachePurger>(stage: &S) -> Result<(), ThoError> {
    let info = stage.get_state_info(1).await?;
    assert!(info.tensors.len() > 1);
    let mut tensors = Vec::new();
    for info in info.tensors {
        let key = info.tensor_key.to_tensor_key();
        tensors.push((key.clone(), tch_result(stage.get_state_tensor(1, key).await?.f_to_device(Device::Cpu))?));
    }
    stage.drop_gpu(1, "test worker drop").await?;
    stage.drop_gpu(1, "repeat worker drop").await?;
    assert!(!stage.has_gpu_data(1).await);
    assert!(stage.pre_run(1).await.is_err());
    assert!(stage.set_state_tensor(1, "unknown".into(), tch_result(Tensor::f_zeros([1], (tch::Kind::Float, Device::Cuda(0))))?).await.is_err());
    let (key, tensor) = &tensors[0];
    assert!(stage.set_state_tensor(1, key.clone(), tch_result(tensor.f_alias())?).await.is_err());
    assert!(stage.set_state_tensor(1, key.clone(), tch_result(tch_result(tensor.f_to_device(Device::Cuda(0)))?.f_to_kind(tch::Kind::Double))?).await.is_err());
    assert!(stage.set_state_tensor(1, key.clone(), tch_result(Tensor::f_zeros([999], (tensor.kind(), Device::Cuda(0))))?).await.is_err());
    let count = tensors.len();
    for (index, (key, tensor)) in tensors.into_iter().rev().enumerate() {
        stage.set_state_tensor(1, key.clone(), tch_result(tensor.f_to_device(Device::Cuda(0)))?).await?;
        assert!(stage.set_state_tensor(1, key, tch_result(tensor.f_to_device(Device::Cuda(0)))?).await.is_err());
        if index + 1 < count { assert!(stage.pre_run(1).await.is_err()); }
    }
    drop(stage.pre_run(1).await?);
    Ok(())
}

async fn round_trips() -> Result<(), ThoError> {
    let global = GlobalCtx::global();
    let ctx = RunCtxImpl::new(1, "inference state round trips");
    let motif_init = ApplyMotifNetInit::new(8101, 1, PathBuf::from("motif"), 0, 19, false, None);
    let label_init = ApplyLabelNetInit::new(8102, "label".into(), 23, PathBuf::from("label"), false);
    let motif: Arc<DynSyncStageWithStateEnum<SyncApplyMotifNet<RunCtxImpl>, MotifNetMeta>> = Arc::new(if global.is_ipc() {
        DynSyncStageWithStateEnum::Proxy {
            stage: Arc::new(ProxyStage::proxy(APPLY_MOTIF_NET_ENDPOINT, Some("8101".into()), 1, Box::new(motif_init)).await?),
            init: MotifNetMeta::new(19, 0, motif_params()),
        }
    } else {
        let stage = Arc::new(SyncApplyMotifNet::new_local(motif_init.inference_id(), 1, motif_params(), motif_module()?, 0, 19, None).await?);
        restore_guards(stage.as_ref()).await?;
        DynSyncStageWithStateEnum::Local(stage)
    });
    let label: Arc<DynSyncStageWithStateEnum<ApplyLabelNet<RunCtxImpl>, ApplyLabelNetInit>> = Arc::new(if global.is_ipc() {
        DynSyncStageWithStateEnum::Proxy {
            stage: Arc::new(ProxyStage::proxy(APPLY_LABEL_NET_ENDPOINT, Some("8102".into()), 1, Box::new(label_init.clone())).await?),
            init: label_init,
        }
    } else {
        let stage = Arc::new(ApplyLabelNet::new_local(mono_key!(0u32, "test").into(), "label".into(), 23, label_module(false)?).await?);
        restore_guards(stage.as_ref()).await?;
        DynSyncStageWithStateEnum::Local(stage)
    });
    let motif_cache = global.register_cache(("inference_state_motif".into(), motif.clone() as Arc<dyn CachePurger>));
    let label_cache = global.register_cache(("inference_state_label".into(), label.clone() as Arc<dyn CachePurger>));
    let mut expected_motif = None;
    let mut expected_label = None;
    for cycle in 0..3 {
        let motif_input = Arc::new(DevicesWrap::wrap(ThoTensor::new(mono_key!("motif input", "test").into(), tch_result(tch_result(tch_result(Tensor::f_arange(16, (tch::Kind::Float, Device::Cpu)))?.f_reshape([1, 16]))?.f_div_scalar(16.))?)?));
        let motif_output = GlobalCtx::global().execution(1).run_stage(motif.clone(), ApplyMotifNetInputWrapper::new(motif_input), 1, &ctx).await?.copy_to_device(0).await?;
        let label_input = Arc::new(DevicesWrap::wrap(DataWithExtra::new(ThoTensor::new(mono_key!("label input", "test").into(), tch_result(tch_result(tch_result(Tensor::f_arange(16, (tch::Kind::Float, Device::Cpu)))?.f_reshape([2, 8]))?.f_div_scalar(16.))?)?, vec![8, 6])));
        if global.is_ipc() && cycle == 1 {
            for _ in 0..2 {
                let result = GlobalCtx::global().execution(1).run_stage(label.clone(), ApplyLabelNetInputWrapper::new(label_input.clone()), 1, &ctx).await;
                assert!(result.unwrap_err().is_oom(), "injected reconstruction OOM must remain typed");
                assert!(label.has_gpu_data(1).await, "partial restore must remain purgeable");
            }
            label.purge_gpu(None, 1, "discard partial restore after OOM").await?;
            assert!(!label.has_gpu_data(1).await);
        }
        let label_output = GlobalCtx::global().execution(1).run_stage(label.clone(), ApplyLabelNetInputWrapper::new(label_input), 1, &ctx).await?.copy_to_device(0).await?;
        if let (Some(expected_motif), Some(expected_label)) = (&expected_motif, &expected_label) {
            assert!(tch_result(motif_output.t_ref().f_allclose(expected_motif, 1e-6, 1e-6, false))?, "motif output changed after restore {cycle}");
            assert!(tch_result(label_output.t_ref().f_allclose(expected_label, 1e-6, 1e-6, false))?, "label output changed after restore {cycle}");
        } else {
            expected_motif = Some(motif_output.t());
            expected_label = Some(label_output.t());
        }
        motif.purge_gpu(None, 2, "wrong device").await?;
        label.purge_gpu(None, 2, "wrong device").await?;
        assert!(motif.has_gpu_data(1).await && label.has_gpu_data(1).await);
        global.dump_all_to_cpu(&HashMap::new(), 1, "inference state").await?;
        assert!(!motif.has_gpu_data(1).await && !label.has_gpu_data(1).await);
        global.dump_all_to_cpu(&HashMap::new(), 1, "repeated inference purge").await?;
        motif.purge_gpu(None, 1, "repeat").await?;
        label.purge_gpu(None, 1, "repeat").await?;
        if global.is_ipc() {
            let ipc = GlobalCtx::global().execution(1).ipc_actor()?;
            // Direct worker calls skip the proxy restore; neither model may run
            // after its GPU state was dropped.
            let input = ThoTensor::<ChannelsData, SynKey>::new(mono_key!("raw", "test").into(), tch_result(Tensor::f_ones([1, 16], (tch::Kind::Float, Device::Cpu)))?)?.copy_to_device(1).await?;
            let wire = ApplyMotifNetInput::new(&input).clone_refs()?;
            let result = ipc.invoke_stage(ipc.worker_id()?, APPLY_MOTIF_NET_ENDPOINT.into(), Some("8101".into()), wire, false).await?;
            let OpResult::Failed(error) = result.status else { panic!("worker retained motif model") };
            assert!(ThoError::from(error).to_string().contains("offloaded"));
            let input = DataWithExtra::new(ThoTensor::<MotifProbability, SynKey>::new(mono_key!("raw label", "test").into(), tch_result(Tensor::f_ones([2, 8], (tch::Kind::Float, Device::Cpu)))?)?.copy_to_device(1).await?, vec![8, 6]);
            let wire = ApplyLabelNetInput { id: input.id().clone(), input: ThoDataRef::Borrowed(&input) }.clone_refs()?;
            let result = ipc.invoke_stage(ipc.worker_id()?, APPLY_LABEL_NET_ENDPOINT.into(), Some("8102".into()), wire, false).await?;
            let OpResult::Failed(error) = result.status else { panic!("worker retained label model") };
            assert!(ThoError::from(error).to_string().contains("offloaded"));
        }
    }
    global.remove_cache(motif_cache.key());
    global.remove_cache(label_cache.key());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_apply_nets_state_local() -> Result<(), ThoError> {
    let global = GlobalCtx::init_devices(2)?;
    let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
    let result = tokio::time::timeout(Duration::from_secs(90), round_trips()).await?;
    global.shutdown().await;
    result
}

#[test]
fn test_apply_nets_state_ipc() -> Result<(), ThoError> {
    thoosa_common::ipc::tests::run_ipc_test(module_path!(), async {
        tokio::time::timeout(Duration::from_secs(90), round_trips()).await?
    }, async |sock| {
        let ws = Arc::new(RwLock::new(MotifsWsMockBuilder::new(None, 1, 100, 16)
            .with_load_motif_net(vec![(PathBuf::from("motif"), Box::new(motif_params), Box::new(|| motif_module().expect("motif fixture")))])
            .with_load_label_net(vec![(PathBuf::from("label"), 23, Box::new(label_params), Box::new(|| label_module(true).expect("label fixture")))])
            .build()));
        CudaWorker::run_worker(sock, &ws, HashMap::from([
            (APPLY_MOTIF_NET_ENDPOINT.into(), Box::new(apply_motif_net_endpoint_init) as InitFn<dyn crate::motifs_ws::MotifsWorkspace>),
            (APPLY_LABEL_NET_ENDPOINT.into(), Box::new(apply_label_net_endpoint_init) as InitFn<dyn crate::motifs_ws::MotifsWorkspace>),
        ]), thoosa_common::tho_device::ThoDevice::from(1usize)).await
    })
}
