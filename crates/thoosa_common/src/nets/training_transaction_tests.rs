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
        let mut builder = SeqBuilder::new(1, None)?;
        let module = builder.linear_s(1, "output")?.end_seq()?.build_tch_module(&vs.root())?;
        // Frozen parameters still belong to the optimizer registration and must
        // not become per-batch rollback copies merely because requires_grad is false.
        let frozen = vs.root().f_var("frozen", &[1024], nn::Init::Const(0.5))?;
        let _ = frozen.f_set_requires_grad(false)?;
        let _ = vs.root().add("running_stat", Tensor::f_from_slice(&[0.25f32])?, false);
        tch::no_grad(|| -> Result<(), ThoError> {
            for var in &vs.variables_.lock().or_else(|_| Err(ThoError::new("test VarStore lock poisoned")))?.trainable_variables {
                let _ = var.tensor.f_alias()?.f_fill_(0.5)?;
            }
            Ok(())
        })?;
        Ok(module)
    }, 0.01), Cpu.into())?)
}

fn variable(module: &ThoTrainModule<AnyDims, AnyDims>, name: &str) -> Result<Tensor, ThoError> {
    let OptSnapshot::Opt { vs, .. } = &module.opt_state else { return Err(ThoError::new("test module not loaded")); };
    let variables = vs.variables();
    let tensor = variables.get(name).ok_or_else(|| ThoError::NotFound(Some(name.into())))?;
    Ok(tensor.f_alias()?)
}

fn accumulate(module: &mut ThoTrainModule<AnyDims, AnyDims>) -> Result<(), ThoError> {
    let input = ThoTensor::new(thoosa_shared::debug_id!("training phases"), Tensor::f_ones([2, 1], (Kind::Float, Cpu))?)?;
    let output = module.module().tho_forward(&input, None, true)?;
    output.as_ref().f_square()?.f_mean(Kind::Float)?.f_backward()?;
    let _ = variable(module, "running_stat")?.f_fill_(42.0)?;
    Ok(())
}

fn assert_terminal_over_ipc(error: ThoError) -> Result<(), ThoError> {
    assert!(!error.is_retryable_oom(), "invalidated training must not be retried locally");
    let wire = serde_json::to_vec(&WorkerError::from(error))?;
    let decoded: WorkerError = serde_json::from_slice(&wire)?;
    assert!(!ThoError::from(decoded).is_retryable_oom(), "invalidated training must not be retried over IPC");
    Ok(())
}

#[test]
fn test_training_snapshot_contains_only_buffers() -> Result<(), ThoError> {
    let module = module(0)?;
    let buffers = module.capture_training_buffers()?;
    assert_eq!(buffers.len(), 1, "parameters, including frozen ones, must not be captured");
    assert!(buffers[0].0.is_set_to(&variable(&module, "running_stat")?));
    assert!(!buffers[0].0.is_set_to(&buffers[0].1), "rollback requires an independent buffer copy");
    let _ = variable(&module, "running_stat")?.f_fill_(42.0)?;
    assert_eq!(buffers[0].1.f_double_value(&[0])?, 0.25);
    Ok(())
}

#[test]
fn test_training_accumulation_completion_oom_is_retryable() -> Result<(), ThoError> {
    for optimizer in 0..3 {
        let mut module = module(optimizer)?;
        let before = variable(&module, "seq_0.output.weight")?.f_double_value(&[0, 0])?;
        let completions = Cell::new(0);
        let error = module.training_transaction_with(accumulate, |_| panic!("commit must not run after failed accumulation completion"), || {
            completions.set(completions.get() + 1);
            if completions.get() == 1 { Err(ThoError::oom()) } else { Ok(()) }
        }).unwrap_err();
        assert!(error.is_retryable_oom());
        assert_eq!(completions.get(), 2, "restore must also complete before retry");
        assert_eq!(variable(&module, "running_stat")?.f_double_value(&[0])?, 0.25);
        assert_eq!(variable(&module, "seq_0.output.weight")?.f_double_value(&[0, 0])?, before);
        module.ensure_training_usable()?;
        module.training_transaction(accumulate)?;
        assert_ne!(variable(&module, "seq_0.output.weight")?.f_double_value(&[0, 0])?, before);
    }
    Ok(())
}

#[test]
fn test_training_commit_and_completion_failures_are_terminal() -> Result<(), ThoError> {
    for optimizer in 0..3 {
        for fail_completion in [false, true] {
            let mut module = module(optimizer)?;
            let before = variable(&module, "seq_0.output.weight")?.f_double_value(&[0, 0])?;
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
            assert_terminal_over_ipc(error)?;
            assert_ne!(variable(&module, "seq_0.output.weight")?.f_double_value(&[0, 0])?, before, "partial optimizer updates must not be presented as rolled back");
            assert!(module.ensure_training_usable().is_err());
            let retried: Result<(), ThoError> = module.training_transaction(|_| panic!("invalidated session must reject subsequent batches"));
            assert_terminal_over_ipc(retried.unwrap_err())?;
        }
    }
    Ok(())
}

#[test]
fn test_training_cleanup_failure_invalidates_session() -> Result<(), ThoError> {
    let mut module = module(0)?;
    let completions = Cell::new(0);
    let error = module.training_transaction_with(|module| {
        accumulate(module)?;
        Err::<(), _>(ThoError::oom())
    }, |_| panic!("failed accumulation must not commit"), || {
        completions.set(completions.get() + 1);
        if completions.get() == 2 { Err(ThoError::oom()) } else { Ok(()) }
    }).unwrap_err();
    assert_terminal_over_ipc(error)?;
    assert!(module.ensure_training_usable().is_err());
    Ok(())
}

#[test]
fn test_training_fatal_commit_error_preserves_cuda_recovery() -> Result<(), ThoError> {
    let mut module = module(0)?;
    let error = module.training_transaction_with(accumulate, |_| Err(ThoError::from_cuda_error("optimizer update", 700)), || Ok(())).unwrap_err();
    assert!(error.needs_cuda_ctx_reset());
    let wire: WorkerError = error.into();
    assert!(ThoError::from(wire).needs_cuda_ctx_reset());
    assert!(module.ensure_training_usable().is_err());
    Ok(())
}
