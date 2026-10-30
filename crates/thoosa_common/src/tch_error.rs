//! Adapt LibTorch failures at the tensor boundary, preserving their original
//! sources and the shared error's CUDA recovery semantics.
use thoosa_shared::{cuda_cause_from_msg, tho_error, ThoError};

#[ctor::ctor]
fn register_libtorch_error_adapter() {
    // Anyhow-returning tensor code also needs classification when its error later
    // reaches a shared runner boundary, including after application context is added.
    thoosa_shared::register_error_adapter(classify_libtorch_error);
}

fn classify_libtorch_error(error: anyhow::Error) -> anyhow::Error {
    // LibTorch exposes no CUDA status in TchError. Inspect only its raw Torch
    // payload and retain the source for diagnostics and downcasting.
    if let Some(tch::TchError::Torch(msg)) = error.downcast_ref::<tch::TchError>() {
        if msg.to_ascii_lowercase().contains("cuda") {
            let ThoError::Cuda { action, .. } = cuda_cause_from_msg(msg.clone()) else { unreachable!() };
            return error.context(tho_error::cuda_error(action, "LibTorch operation failed".into()));
        }
    }
    error
}

/// Explicit conversion replaces the From<TchError> implementation, which cannot
/// live here because both the source and destination error types are external.
pub fn tch_error(error: tch::TchError) -> ThoError {
    classify_libtorch_error(anyhow::Error::new(error)).into()
}

/// Keep fallible tensor operations concise while converting at the owning crate.
pub fn tch_result<T>(result: Result<T, tch::TchError>) -> Result<T, ThoError> {
    match result {
        Ok(value) => Ok(value),
        Err(error) => Err(tch_error(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tho_error_libtorch_adapter_preserves_source_and_recovery() {
        for (message, oom, reset) in [
            ("CUDA out of memory. Tried to allocate 128 MiB", true, false),
            ("CUDA error: an illegal memory access was encountered", false, true),
        ] {
            let source = anyhow::Error::new(tch::TchError::Torch(message.into())).context("forward model");
            let error = ThoError::from(source);
            assert_eq!((error.is_oom(), error.needs_cuda_restart()), (oom, reset));
            let ThoError::Anyhow(source) = &error else { panic!("expected original Torch error") };
            assert!(source.chain().any(|cause| cause.downcast_ref::<tch::TchError>().is_some()));
            assert!(format!("{error:#}").contains(message));
        }

        let error = tch_error(tch::TchError::Shape("CUDA out of memory".into()));
        assert!(error.is_oom(), "CUDA out of memory even if it's in TchError::Shape ");
    }

    #[test]
    fn test_tch_result_preserves_source_and_recovery_after_cloning() {
        for (message, oom, reset) in [
            ("CUDA out of memory", true, false),
            ("CUDA error: an illegal memory access was encountered", false, true),
        ] {
            let error = tch_result::<()>(Err(tch::TchError::Torch(message.into()))).unwrap_err();
            let error = error.clone().context("run stage");
            assert_eq!((error.is_retryable_oom(), error.needs_cuda_restart()), (oom, reset));
            let ThoError::Anyhow(source) = &error else { panic!("expected original Torch error") };
            assert!(source.chain().any(|cause| cause.downcast_ref::<tch::TchError>().is_some()));
            assert!(format!("{error:#}").contains(message));
        }
    }

    #[test]
    fn test_libtorch_adapter_preserves_existing_typed_recovery() {
        // A terminal marker must take precedence over its original retryable Torch error.
        let terminal = ThoError::TrainingSessionFailed(std::sync::Arc::new(thoosa_shared::TrainingSessionFailure { original: ThoError::oom(), recovery: None }));
        let source = anyhow::Error::new(tch::TchError::Torch("CUDA out of memory".into())).context(terminal);
        let error = ThoError::from(source).context("training failed");
        assert!(matches!(error.what(), ThoError::TrainingSessionFailed(_)));
        assert!(error.is_oom());
        assert!(!error.is_retryable_oom());
    }

    #[test]
    fn test_panic_adapter_classifies_libtorch_and_preserves_typed_errors() {
        let error = thoosa_shared::panic_to_err!(Err::<(), _>(tch::TchError::Torch("CUDA out of memory".into()))).unwrap_err();
        assert!(matches!(error.what(), ThoError::Cuda { action: thoosa_shared::CudaAction::HandleOOM, .. }));
        let error = thoosa_shared::panic_to_err!(Err::<(), _>(ThoError::ContextReset)).unwrap_err();
        assert!(matches!(error.what(), ThoError::ContextReset));
    }

}
