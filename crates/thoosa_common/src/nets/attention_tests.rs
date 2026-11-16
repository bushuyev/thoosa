use super::{scaled_attention, valid_time_mask};
use tch::{Device, Kind, Tensor};

fn attention_inputs(device: Device, kind: Kind) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
    // Use an odd head width and the projection/transposition layout from the
    // model to exercise CUDA alignment and the original (unpadded) scale.
    let base = Tensor::f_arange(2 * 7 * 3 * 19, (Kind::Float, device))?.f_view([2, 7, 3, 19])?.f_transpose(1, 2)?.f_mul_scalar(0.013)?;
    Ok((base.f_sin()?.f_to_kind(kind)?.f_set_requires_grad(true)?, base.f_cos()?.f_to_kind(kind)?.f_set_requires_grad(true)?, base.f_mul_scalar(0.7)?.f_sin()?.f_to_kind(kind)?.f_set_requires_grad(true)?))
}

fn compare_dense_attention(device: Device, kind: Kind, masked: bool) -> anyhow::Result<()> {
    let (q, k, v) = attention_inputs(device, kind)?;
    let mask = if masked { Some(valid_time_mask(&[7, 3], 2, 7, device)?) } else { None };
    let actual = scaled_attention(q.f_alias()?, k.f_alias()?, v.f_alias()?, mask.as_ref().map(Tensor::f_alias).transpose()?, 0.0, true)?;

    // Independent reference for the old attention calculation, including its
    // finite padding sentinel. Every row here has at least one valid key.
    let mut scores = q.f_matmul(&k.f_transpose(-2, -1)?)?.f_div_scalar(19_f64.sqrt())?;
    if let Some(mask) = &mask {
        scores = scores.f_masked_fill(&mask.f_logical_not()?.f_unsqueeze(1)?.f_unsqueeze(2)?, -1e4)?;
    }
    let expected = scores.f_softmax(-1, kind)?.f_matmul(&v)?;
    let tolerance = if kind == Kind::Float { 2e-4 } else { 2e-2 };
    assert_eq!(actual.size(), vec![2, 3, 7, 19]);
    assert!(actual.f_allclose(&expected, tolerance, tolerance, false)?, "SDPA output differs from dense attention: {device:?} {kind:?} masked={masked}");

    let actual_grads = Tensor::f_run_backward(&[actual.f_square()?.f_mean(Kind::Float)?], &[&q, &k, &v], false, false)?;
    let expected_grads = Tensor::f_run_backward(&[expected.f_square()?.f_mean(Kind::Float)?], &[&q, &k, &v], false, false)?;
    let grad_tolerance = if kind == Kind::Float { 2e-5 } else { 2e-3 };
    for (actual, expected) in actual_grads.iter().zip(&expected_grads) {
        assert!(actual.f_allclose(expected, tolerance, grad_tolerance, false)?, "SDPA gradient differs from dense attention: {device:?} {kind:?} masked={masked}");
    }
    Ok(())
}

fn check_attention_dropout(device: Device, kind: Kind, masked: bool) -> anyhow::Result<()> {
    let (q, k, v) = attention_inputs(device, kind)?;
    let apply = |dropout, train| {
        let mask = if masked { Some(valid_time_mask(&[7, 3], 2, 7, device)?) } else { None };
        scaled_attention(q.f_alias()?, k.f_alias()?, v.f_alias()?, mask, dropout, train)
    };
    let evaluation = apply(0.4, false)?;
    assert!(evaluation.f_allclose(&apply(0.0, false)?, 0.0, 0.0, false)?, "evaluation must disable SDPA dropout");
    let training = apply(0.4, true)?;
    assert!(training.f_isfinite()?.f_all()?.f_int64_value(&[])? != 0, "training dropout must remain finite");
    assert!(!training.f_allclose(&evaluation, 1e-5, 1e-5, false)?, "training must apply attention dropout");
    for grad in Tensor::f_run_backward(&[training.f_square()?.f_mean(Kind::Float)?], &[&q, &k, &v], false, false)? {
        assert!(grad.f_isfinite()?.f_all()?.f_int64_value(&[])? != 0, "dropout backward must remain finite");
    }
    Ok(())
}

#[test]
fn test_sdpa_matches_dense_outputs_and_gradients() -> anyhow::Result<()> {
    for masked in [false, true] {
        compare_dense_attention(Device::Cpu, Kind::Float, masked)?;
        check_attention_dropout(Device::Cpu, Kind::Float, masked)?;
    }
    Ok(())
}

#[test]
fn test_sdpa_padding_masks() -> anyhow::Result<()> {
    let (q, k, v) = attention_inputs(Device::Cpu, Kind::Float)?;
    let full_mask = valid_time_mask(&[7, 7], 2, 7, Device::Cpu)?;
    let masked = scaled_attention(q.f_alias()?, k.f_alias()?, v.f_alias()?, Some(full_mask), 0.0, false)?;
    let unmasked = scaled_attention(q.f_alias()?, k.f_alias()?, v.f_alias()?, None, 0.0, false)?;
    assert!(masked.f_allclose(&unmasked, 2e-4, 2e-4, false)?, "omitting an all-valid mask must preserve attention");

    // A fully padded resource must not inject NaNs into the batch.
    let empty_mask = valid_time_mask(&[0, 3], 2, 7, Device::Cpu)?;
    let output = scaled_attention(q, k, v, Some(empty_mask), 0.0, false)?;
    assert!(output.f_isfinite()?.f_all()?.f_int64_value(&[])? != 0);
    assert_eq!(output.f_select(0, 0)?.f_abs()?.f_max()?.f_double_value(&[])?, 0.0);
    Ok(())
}

#[test]
#[ignore = "requires an idle CUDA device; reports SDPA backend choices at INFO"]
fn test_sdpa_cuda_masks_dropout_and_gradients() -> anyhow::Result<()> {
    // Explicit opt-in: run after the active workload finishes. Production uses
    // this same kernel path locally and inside the IPC worker.
    thoosa_shared::test_utils::init_test_logging();
    anyhow::ensure!(tch::Cuda::is_available(), "CUDA is required for this SDPA check");
    for kind in [Kind::Float, Kind::Half] {
        for masked in [false, true] {
            compare_dense_attention(Device::Cuda(0), kind, masked)?;
            check_attention_dropout(Device::Cuda(0), kind, masked)?;
        }
    }
    Ok(())
}
