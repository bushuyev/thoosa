use thoosa_common::tch_error::tch_result;
use super::*;
use crate::ipc::worker_msg::WorkerError;
use std::cell::Cell;

fn module(optimizer: u8) -> Result<ThoTrainModule<AnyDims, AnyDims>, ThoError> {
    let optimizer: ThoOptimizers = match optimizer {
        0 => nn::Adam::default().into(),
        1 => nn::AdamW::default().into(),
        _ => nn::Sgd { momentum: 0.9, ..Default::default() }.into(),
    };
    Ok(ThoTrainModule::new_on_device(ModConfig::new(optimizer, |vs| {
        let initialize = vs.variables().is_empty();
        let mut builder = SeqBuilder::new(1, None)?;
        let module = builder.linear_s(1, "output")?.end_seq()?.build_tch_module(&vs.root())?;
        // Frozen parameters still belong to the optimizer registration and must
        // not become per-batch rollback copies merely because requires_grad is false.
        if initialize {
            let frozen = vs.root().f_var("frozen", &[1024], nn::Init::Const(0.5))?;
            let _ = frozen.f_set_requires_grad(false)?;
            let _ = vs.root().add("running_stat", Tensor::f_from_slice(&[0.25f32])?, false);
            tch::no_grad(|| -> Result<(), ThoError> {
                for var in &vs.variables_.lock().or_else(|_| Err(ThoError::new("test VarStore lock poisoned")))?.trainable_variables {
                    let _ = tch_result(tch_result(var.tensor.f_alias())?.f_fill_(0.5))?;
                }
                Ok(())
            })?;
        }
        Ok(module)
    }, 0.01), thoosa_common::tho_device::ThoDevice::from(0usize))?)
}

fn variable(module: &ThoTrainModule<AnyDims, AnyDims>, name: &str) -> Result<Tensor, ThoError> {
    let OptSnapshot::Opt { vs, .. } = &module.opt_state else { return Err(ThoError::new("test module not loaded")); };
    let variables = vs.variables();
    let tensor = variables.get(name).ok_or_else(|| ThoError::NotFound(Some(name.into())))?;
    Ok(tch_result(tensor.f_alias())?)
}

fn accumulate(module: &mut ThoTrainModule<AnyDims, AnyDims>) -> Result<(), ThoError> {
    let input = ThoTensor::new(thoosa_shared::debug_id!("training phases"), tch_result(Tensor::f_ones([2, 1], (Kind::Float, Cpu)))?)?;
    let output = module.module().tho_forward(&input, None, true)?;
    tch_result(tch_result(tch_result(output.as_ref().f_square())?.f_mean(Kind::Float))?.f_backward())?;
    let _ = tch_result(variable(module, "running_stat")?.f_fill_(42.0))?;
    Ok(())
}

fn assert_terminal_over_ipc(error: ThoError) -> Result<(), ThoError> {
    assert!(!error.is_retryable_oom(), "invalidated training must not be retried locally");
    let caused_by_oom = error.is_oom();
    if caused_by_oom { assert!(format!("{error:#}").contains("out of memory"), "terminal failures must retain their OOM diagnostics"); }
    let wire = serde_json::to_vec(&WorkerError::from(error))?;
    let decoded: WorkerError = serde_json::from_slice(&wire)?;
    let decoded = ThoError::from(decoded);
    assert!(!decoded.is_retryable_oom(), "invalidated training must not be retried over IPC");
    assert_eq!(decoded.is_oom(), caused_by_oom, "IPC must retain the OOM cause separately from retry policy");
    Ok(())
}

// Exercise the serialized metadata used by IPC, with independent CPU allocations.
fn checkpoint(module: &mut ThoTrainModule<AnyDims, AnyDims>) -> Result<(Vec<u8>, Vec<(String, Tensor)>), ThoError> {
    let info = module.state_info()?;
    let mut tensors = Vec::new();
    for tensor in &info.tensors {
        let key = tensor.tensor_key.to_tensor_key();
        let tensor = module.state_tensor(&key)?;
        tensors.push((key, tch_result(tensor.f_to_device_(Cpu, tensor.kind(), false, true))?));
    }
    Ok((serde_json::to_vec(&info)?, tensors))
}

fn restore(module: &mut ThoTrainModule<AnyDims, AnyDims>, checkpoint: &(Vec<u8>, Vec<(String, Tensor)>)) -> Result<(), ThoError> {
    module.reset_state(serde_json::from_slice(&checkpoint.0)?)?;
    for (key, tensor) in &checkpoint.1 { module.set_state_tensor(key, tch_result(tensor.f_to_device_(Cpu, tensor.kind(), false, true))?, 0)?; }
    module.prepare_state(0)?;
    module.ensure_training_usable()
}

#[test]
fn test_import_defers_reconstruction_and_preserves_state_after_oom() -> Result<(), ThoError> {
    let mut module = module(0)?;
    module.training_transaction(accumulate)?;
    let saved = checkpoint(&mut module)?;
    let builds = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let build_count = builds.clone();
    let build = module.config.optimizer_builder;
    module.config.optimizer_builder = Box::new(move |vs| {
        if build_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 { return Err(ThoError::oom().into()); }
        build(vs)
    });
    module.reset_state(serde_json::from_slice(&saved.0)?)?;
    let (key, tensor) = &saved.1[0];
    module.set_state_tensor(key, tch_result(tensor.f_to_device_(Cpu, tensor.kind(), false, true))?, 0)?;
    assert!(module.prepare_state(0).is_err(), "partial imports must not construct a module");
    for (key, tensor) in &saved.1[1..] { module.set_state_tensor(key, tch_result(tensor.f_to_device_(Cpu, tensor.kind(), false, true))?, 0)?; }
    assert_eq!(builds.load(std::sync::atomic::Ordering::SeqCst), 0, "the last tensor import must not reconstruct the optimizer");
    assert!(module.prepare_state(0).unwrap_err().is_retryable_oom());
    assert!(module.ensure_training_usable().is_err());
    // DumpToCpu needs to export complete imports after a reconstruction failure.
    let exported = checkpoint(&mut module)?;
    assert_eq!(exported.0, saved.0);
    for ((key, actual), (expected_key, expected)) in exported.1.iter().zip(&saved.1) {
        assert_eq!(key, expected_key);
        assert!(tch_result(actual.f_equal(expected))?, "reconstruction failure changed {key}");
    }
    module.prepare_state(0)?;
    module.ensure_training_usable()?;
    assert_eq!(builds.load(std::sync::atomic::Ordering::SeqCst), 2);
    module.training_transaction(accumulate)?;
    Ok(())
}

#[test]
fn test_epoch_checkpoint_restores_complete_optimizer_and_buffers() -> Result<(), ThoError> {
    for optimizer in 0..3 {
        for initialized in [false, true] {
            let mut restored = module(optimizer)?;
            let mut reference = module(optimizer)?;
            if initialized {
                restored.training_transaction(accumulate)?;
                reference.training_transaction(accumulate)?;
            }
            let saved = checkpoint(&mut restored)?;
            for fail_commit in [false, true] {
                let error = restored.training_transaction_with(|module| {
                    accumulate(module)?;
                    if fail_commit { Ok(()) } else { Err(ThoError::oom()) }
                }, |module| {
                    module.opt().step();
                    Err(ThoError::oom())
                }, || Ok(())).unwrap_err();
                assert!(error.is_oom());
                assert_terminal_over_ipc(error)?;
                assert!(restored.ensure_training_usable().is_err());
                restore(&mut restored, &saved)?;
                let reference_state = checkpoint(&mut reference)?;
                let actual = checkpoint(&mut restored)?;
                assert_eq!(actual.0, reference_state.0, "optimizer metadata must return to the saved epoch, including absent state");
                for ((actual_key, actual), (expected_key, expected)) in actual.1.iter().zip(&reference_state.1) {
                    assert_eq!(actual_key, expected_key);
                    assert!(tch_result(actual.f_allclose(expected, 1e-7, 1e-7, false))?, "checkpoint differs for {actual_key}");
                }
            }
            // The next update matches uninterrupted training, including momentum.
            restored.training_transaction(accumulate)?;
            reference.training_transaction(accumulate)?;
            assert_eq!(tch_result(variable(&restored, "seq_0.output.weight")?.f_double_value(&[0, 0]))?, tch_result(variable(&reference, "seq_0.output.weight")?.f_double_value(&[0, 0]))?);
            restore(&mut restored, &saved)?;
            assert_eq!(checkpoint(&mut restored)?.0, saved.0, "restoring must not consume the checkpoint");
        }
    }
    Ok(())
}

#[test]
fn test_training_accumulation_completion_oom_invalidates_epoch() -> Result<(), ThoError> {
    let mut module = module(0)?;
    let completions = Cell::new(0);
    let error = module.training_transaction_with(accumulate, |_| panic!("commit must not run"), || {
        completions.set(completions.get() + 1);
        Err(ThoError::oom())
    }).unwrap_err();
    assert!(error.is_oom());
    assert!(!error.is_retryable_oom());
    assert_eq!(completions.get(), 1, "failed epochs do not attempt buffer rollback");
    assert!(module.ensure_training_usable().is_err());
    Ok(())
}

#[test]
fn test_training_commit_and_completion_failures_are_terminal() -> Result<(), ThoError> {
    for optimizer in 0..3 {
        for fail_completion in [false, true] {
            let mut module = module(optimizer)?;
            let before = tch_result(variable(&module, "seq_0.output.weight")?.f_double_value(&[0, 0]))?;
            let completions = Cell::new(0);
            let error = module.training_transaction_with(accumulate, |module| {
                // Simulate an optimizer that modified some state before reporting OOM.
                module.opt().step();
                if fail_completion { Ok(()) } else { Err(ThoError::oom()) }
            }, || {
                completions.set(completions.get() + 1);
                if fail_completion && completions.get() == 2 { Err(ThoError::oom()) } else { Ok(()) }
            }).unwrap_err();
            assert_eq!(completions.get(), 2, "a failed commit must still synchronize");
            assert!(error.is_oom(), "commit failure must retain its typed OOM cause");
            assert_terminal_over_ipc(error)?;
            assert_ne!(tch_result(variable(&module, "seq_0.output.weight")?.f_double_value(&[0, 0]))?, before, "partial optimizer updates must not be presented as rolled back");
            assert!(module.ensure_training_usable().is_err());
            let retried: Result<(), ThoError> = module.training_transaction(|_| panic!("invalidated session must reject subsequent batches"));
            assert_terminal_over_ipc(retried.unwrap_err())?;
        }
    }
    Ok(())
}

#[test]
fn test_training_fatal_commit_error_preserves_cuda_recovery() -> Result<(), ThoError> {
    let mut module = module(0)?;
    let error = module.training_transaction_with(accumulate, |_| Err(ThoError::from_cuda_error("optimizer update", 700)), || Ok(())).unwrap_err();
    assert!(error.needs_cuda_recovery());
    let wire: WorkerError = error.into();
    assert!(ThoError::from(wire).needs_cuda_recovery());
    assert!(module.ensure_training_usable().is_err());
    Ok(())
}

#[test]
fn test_training_optimizer_panic_preserves_terminal_oom() -> Result<(), ThoError> {
    let mut module = module(0)?;
    // tch::Optimizer::step exposes C++ failures through unwrap/panic rather than Result.
    let error = module.training_transaction_with(accumulate, |_| panic!("CUDA out of memory during optimizer update"), || Ok(())).unwrap_err();
    assert!(error.is_oom());
    assert_terminal_over_ipc(error)?;
    assert!(module.ensure_training_usable().is_err());
    Ok(())
}
