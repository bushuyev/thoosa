use std::collections::HashMap;
use std::sync::Arc;

use tch::Kind;
use thoosa_shared::ThoError;

use crate::ipc::json_ser::SyncStageJsonWrapper;
use crate::runner::common::cachable::{Cached, WithDataRef};
use crate::runner::common::runnable::AsyncStage;
use crate::runner::common::RunCtx;
use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::runner::test::FailUntil;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;

pub(crate) mod batch_builder;
pub(crate) mod classifier;
pub(crate) mod common;
pub(crate) mod join_and_classify;
pub(crate) mod loader;


use classifier::{sync_classifier_endpoint_init, Classifier, SYNC_CLASSIFIER_ENDPOINT};
use common::{ClsOut, FileListInput, ModelTestId};
use join_and_classify::JoinAndClassifyTask;
use loader::ABLoader;

#[ctor::ctor]
fn init() {
    thoosa_shared::test_utils::init_test_logging();
}

fn init_devices_once() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let _ = RealGlobalCtx::init_devices(2);
    });
}

pub(crate) fn sync_model_test_endpoint_inits() -> HashMap<String, Box<dyn Fn(String) -> Result<Box<dyn SyncStageJsonWrapper>, ThoError>>> {
    HashMap::from([(
        SYNC_CLASSIFIER_ENDPOINT.to_string(),
        Box::new(sync_classifier_endpoint_init) as Box<dyn Fn(String) -> Result<Box<dyn SyncStageJsonWrapper>, ThoError>>,
    )])
}

pub(crate) async fn run_model_test_pipeline(
    ctx_device_idx: usize,
    check_device_idx: usize,
    files: Vec<(&'static str, f64)>,
    input_id: ModelTestId,
    classifier_id: ModelTestId,
    batch_size: usize,
    epochs: i64,
    fail_until: Vec<FailUntil>,
    min_accuracy: f64,
) -> Result<(), ThoError> {
    let mut ctx = RunCtxImpl::new(ctx_device_idx, "test");
    let loader = Arc::new(ABLoader::new(64, 4, Kind::Float));
    let clf = Arc::new(Classifier::new(
        classifier_id,
        4,
        epochs,
        0.1,
        fail_until,
        ThoDevice::from(ctx_device_idx),
    ).await?);
    let join_and_cls = Arc::new(JoinAndClassifyTask::new(
        Arc::clone(&loader),
        Arc::clone(&clf),
        batch_size,
    ));
    let input = FileListInput::new(files, input_id);

    let acc = Arc::clone(&join_and_cls)
        .run_cached(input, ctx.branch(false))
        .await?;

    acc.with_data_ref(check_device_idx,
                      ctx.branch(false), None, |acc_t: &ThoTensor<ClsOut, ModelTestId>, _ctx| {
              let v = ClsOut::accuracy(acc_t);
              assert!(v >= min_accuracy, "training accuracy too low: {v}");
              Ok(())
          })
    .await?;

    Ok(())
}

#[tokio::test]
async fn test_ab_loader_and_classifier_cpu() -> Result<(), ThoError> {
    unsafe {
        let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
    }
    init_devices_once();
    let _global_mock_ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);

    run_model_test_pipeline(
        1,
        0,
        vec![("file_0", 1.0), ("file_1", 0.0)],
        ModelTestId("files_0_1"),
        ModelTestId("lin_cls"),
        2,
        200,
        vec![FailUntil::Never],
        0.8,
    ).await
}

#[tokio::test]
async fn test_ab_loader_and_classifier_cpu_many_files() -> Result<(), ThoError> {
    unsafe {
        let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
    }
    init_devices_once();
    let _global_mock_ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);

    run_model_test_pipeline(
        0,
        1,
        vec![
            ("file_0", 1.0),
            ("file_1", 0.0),
            ("file_2", 1.0),
            ("file_3", 0.0),
        ],
        ModelTestId("files_0_1_2_3"),
        ModelTestId("lin_cls_many"),
        4,
        100,
        vec![FailUntil::Never],
        0.8,
    ).await
}

#[tokio::test]
async fn test_classifier_oom_max_parallel_escalates() -> Result<(), ThoError> {
    unsafe {
        let _lib = libloading::Library::new("libtorch_cpu.so").expect("no libtorch_cpu.so");
        let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
    }
    init_devices_once();
    let _global_ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);

    let files: Vec<(&'static str, f64)> = (0..16)
        .map(|i| {
            (
                Box::leak(format!("file_{}", i).into_boxed_str()) as &'static str,
                if i % 2 == 0 { 1.0 } else { 0.0 },
            )
        })
        .collect();

    run_model_test_pipeline(
        1,
        1,
        files,
        ModelTestId("files_oom_permits"),
        ModelTestId("lin_cls_oom"),
        16,
        50,
        vec![FailUntil::NaxParallel(crate::runner::adjusters::max_parallel::MAX_PARALLEL), FailUntil::MemLimit(128)],
        0.5,
    ).await
}
