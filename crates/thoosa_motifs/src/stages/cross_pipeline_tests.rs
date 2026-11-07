use super::*;
use crate::motifs_store::MockMotifsStoreBuilder;
use crate::stages::cross_batch_clc::{CrossUnfoldClc, UnfoldParams};
use crate::stages::find_motifs::{FindMotifsCrossUnfoldClcInit, FindMotifsMasksCtxInit};
use crate::stages::tensor_loader::{RawTensorLoader, mock_tensor_loader};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tch::{Device, Kind, Tensor};
use thoosa_common::ipc::{cuda_worker::CudaWorker, tests::run_ipc_test};
use thoosa_common::resources::Simple;
use thoosa_common::runner::common::{MockRunCtx, mock_run_ctx};
use thoosa_common::runner::global_ctx::{GlobalCtx, mock_global_ctx::MockGlobalCtxBuilder};
use thoosa_common::runner::oom_handler::{NoContexts, NoRunCtxAdjuster};
use thoosa_common::runner::run_ctx::RunCtxImpl;
use thoosa_common::tch_error::tch_result;
use thoosa_common::tho_device::ThoDevice;
use tokio::sync::Notify;

type NoRetry = OOMHandler<'static, 'static, NoInputAdjuster, NoStageAdjuster, NoRunCtxAdjuster, ()>;

#[derive(Debug, Clone, CpuData)]
#[cpu_data(shallow)]
struct Block(usize);
impl Input for Block {}
impl Output for Block {}
impl CrossClcResultMerge for Block {
    fn merge_batch_1(self, _other: Self) -> Result<Self, ThoError> {
        Ok(self)
    }
}

async fn check_pipeline(fail_consumer: bool) -> Result<(), ThoError> {
    GlobalCtx::init_devices(2)?;
    let _global = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
    let resources: Vec<Arc<Resource>> = (1..=3).map(|i| Arc::new((i, 1, format!("pipeline_{i}").as_str(), 2).into())).collect();
    let mut fixtures = Vec::new();
    for resource in &resources {
        fixtures.push((resource.as_ref().clone(), tch_result(Tensor::f_zeros([1, 2], (Kind::Float, Device::Cpu)))?));
    }
    let loader = Arc::new(mock_tensor_loader::build().with_run_cached(fixtures).finish());
    let started = Arc::new(Notify::new());
    let calculated_next = Arc::new(Notify::new());
    let completed = Arc::new(AtomicBool::new(false));
    let calculated = Arc::new(AtomicUsize::new(0));
    let consumed = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let (release, wait) = std::sync::mpsc::channel();
    let wait = parking_lot::Mutex::new(wait);
    let receiver = Arc::new(
        mock_cross_loader_receiver::build::<CrossUnfoldClcOutput<Block>, MockRunCtx>()
            .with_receive_fn({
                let started = started.clone();
                let completed = completed.clone();
                let consumed = consumed.clone();
                move |input, _| {
                    let iter_0 = input.iter_0();
                    let iter_1 = input.iter_1();
                    let id = input.id().clone();
                    let block = input.clc_result();
                    let index = block.0;
                    consumed.lock().push(index);
                    if index == 0 {
                        started.notify_one();
                        wait.lock().recv_timeout(Duration::from_secs(10)).expect("next comparison must overlap this consumer");
                        completed.store(true, Ordering::SeqCst);
                        if fail_consumer {
                            return ConsumeResult::Failed {
                                input: CrossUnfoldClcOutput::new(block, iter_0, iter_1, id),
                                error: ThoError::new("consumer failed"),
                            };
                        }
                    }
                    ConsumeResult::Ok(())
                }
            })
            .finish(),
    );
    let calculation = Arc::new(
        mock_cross_loader_clc::build::<Block, MockRunCtx>()
            .with_run_batch_fn({
                let completed = completed.clone();
                let calculated = calculated.clone();
                let calculated_next = calculated_next.clone();
                move |_, _| {
                    let index = calculated.fetch_add(1, Ordering::SeqCst);
                    if index == 1 {
                        calculated_next.notify_one();
                    }
                    if index == 2 {
                        assert!(completed.load(Ordering::SeqCst), "only one block may run ahead of consumption");
                    }
                    Ok(CrossUnfoldClcOutput::new(Block(index), SlicesIter::new(1, vec![]), SlicesIter::new(1, vec![]), "pipeline".into()))
                }
            })
            .finish(),
    );
    let store = MockMotifsStoreBuilder::new().build();
    let cross_loader = CrossLoader::<Simple, _, _, _, _, _>::new(
        1,
        loader.clone(),
        &store,
        calculation,
        {
            let receiver = receiver.clone();
            move |_| receiver.clone()
        },
        2,
        |block| async { Ok(block) }.boxed(),
        |_| async { Ok(()) }.boxed(),
        CrossUnfoldClcOutputMergeMode::GpuMerge,
        0,
    );
    let outer = loader.run_cached(resources[0].clone(), mock_run_ctx::default(0)).await?;
    // The first consumer cannot finish until the next comparison has run. This
    // detects serial execution without timing assertions, and failure must drain.
    let (result, ()) = tokio::join!(cross_loader.run_inner_batches(&resources, vec![outer], vec![resources[0].clone()], 0, 0, 2, receiver, mock_run_ctx::default(0)), async {
        tokio::time::timeout(Duration::from_secs(10), async {
            started.notified().await;
            calculated_next.notified().await;
        })
        .await
        .expect("pipeline made no progress");
        assert!(!completed.load(Ordering::SeqCst));
        release.send(()).expect("consumer still running");
    });
    assert!(completed.load(Ordering::SeqCst));
    if fail_consumer {
        assert!(result.unwrap_err().to_string().contains("consumer failed"));
        assert_eq!(calculated.load(Ordering::SeqCst), 2);
        assert_eq!(*consumed.lock(), vec![0]);
    } else {
        result?;
        assert_eq!(*consumed.lock(), vec![0, 1, 2]);
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_discovery_pipeline_overlap_and_order() -> Result<(), ThoError> {
    check_pipeline(false).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_discovery_pipeline_drains_failed_consumer() -> Result<(), ThoError> {
    check_pipeline(true).await
}

async fn check_prepared_outer() -> Result<(), ThoError> {
    let mut ctx = RunCtxImpl::new(1, "prepared_outer");
    let params = UnfoldParams::new(1, 3, 2);
    let preparation = PrepareCrossBatch::new(params, 1).await?;
    let calculation = CrossUnfoldClc::new_dyn_for_find_motifs(FindMotifsCrossUnfoldClcInit::new(crate::distance::SimilarityMetric::default(), true, 1, FindMotifsMasksCtxInit::ExactS { euc_dist: 100.0, channels: 1 }, params), 1).await?;
    let loader = Arc::new(RawTensorLoader::<RunCtxImpl>::new(
        |resource, _| {
            // A short filtered resource must keep its zero-sized boundary between
            // valid resources; caching must not shift the resulting slice IDs.
            let len = if resource.id() == 2 { 1 } else { 7 };
            let values: Vec<f32> = (0..len).map(|i| (i * i + resource.id()) as f32).collect();
            ThoTensor::new(format!("prepared_resource_{}", resource.id()).into(), tch_result(tch_result(Tensor::f_from_slice(&values))?.f_view([1, len as i64]))?)
        },
        ThoDevice::from(0usize),
    ));
    let resources: Vec<Arc<Resource>> = (1..=5).map(|i| Arc::new((i, 1, format!("prepared_{i}").as_str(), 7).into())).collect();
    let mut tensors = Vec::new();
    for resource in &resources {
        tensors.push(loader.clone().run_cached(resource.clone(), ctx.branch(false)).await?);
    }
    let input = Arc::new(CrossInputSelectorWrap::new("outer".into(), tensors[..3].to_vec(), vec![], resources[..3].to_vec(), vec![], CacheArea::default()));
    let cache = Arc::new(TmpCached::new("test_prepared_outer").await);
    let prepared = Arc::new(cache.insert_guarded(NoRetry::new(vec![], None).run(&preparation, input, ctx.branch(false), NoContexts).await?));
    drop(cache); // The entry itself must keep offloading registration alive.
    assert_eq!(BorrowRef::borrow_ref(&prepared, 1).await?.as_ref().sizes, vec![3, 0, 3]);
    let mut match_count = 0;
    for inner in 3..5 {
        // Force the actual cache-purge path before the second use, then reload
        // the same prepared windows through either local or IPC tensor copying.
        if inner == 4 {
            GlobalCtx::global().dump_all_to_cpu(1, "prepared outer regression").await?;
            assert_eq!(BorrowRef::borrow_ref(&prepared, 0).await?.as_ref().tho_device_idx(), 0);
        }
        let make_input = || CrossInputSelectorWrap::new("cross".into(), tensors[..3].to_vec(), vec![tensors[inner].clone()], resources[..3].to_vec(), vec![resources[inner].clone()], CacheArea::default());
        let raw = NoRetry::new(vec![], None).run(&calculation, Arc::new(make_input()), ctx.branch(false), NoContexts).await?;
        let expected_slices: Vec<_> = raw.iter_0().collect();
        let expected = raw.clc_result().to_cpu().await?.into_indices_iter()?.collect::<Vec<_>>();
        let mut input = make_input();
        input.prepared_outer = Some(prepared.clone());
        let reused = NoRetry::new(vec![], None).run(&calculation, Arc::new(input), ctx.branch(false), NoContexts).await?;
        assert_eq!(reused.iter_0().collect::<Vec<_>>(), expected_slices);
        assert_eq!(
            expected_slices,
            vec![
                thoosa_shared::TensorIdSliceIdx(1, 0),
                thoosa_shared::TensorIdSliceIdx(1, 2),
                thoosa_shared::TensorIdSliceIdx(1, 4),
                thoosa_shared::TensorIdSliceIdx(3, 0),
                thoosa_shared::TensorIdSliceIdx(3, 2),
                thoosa_shared::TensorIdSliceIdx(3, 4)
            ]
        );
        let actual = reused.clc_result().to_cpu().await?.into_indices_iter()?.collect::<Vec<_>>();
        match_count += actual.len();
        assert_eq!(actual, expected);
    }
    assert!(match_count > 0, "comparison must exercise matches and distances");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_prepared_discovery_outer_local() -> Result<(), ThoError> {
    GlobalCtx::init_devices(2)?;
    let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
    check_prepared_outer().await
}

#[test]
fn test_prepared_discovery_outer_ipc() -> Result<(), ThoError> {
    run_ipc_test(module_path!(), check_prepared_outer(), async |socket| {
        let ws = Arc::new(parking_lot::RwLock::new(crate::motifs_ws::mock_motifs_ws_simp(None, 1, 100, 1000)));
        CudaWorker::run_worker(socket, &ws, crate::cli::cuda_worker_endpoint_inits(), ThoDevice::from(1usize)).await
    })
}
