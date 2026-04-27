#![allow(non_snake_case, unused_variables, unused_assignments, non_camel_case_types)]

use clap::{Args, Subcommand};
use crate::utils::from;
use serde::Deserialize;
use tch::kind::FLOAT_CPU;
use tch::{Device, IndexOp, Kind, Tensor};
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::math::sg_smoothing_coeffs;
use thoosa_shared::ThoError;

pub trait InputFilterLengthAdjuster {
    fn adjust_length(&self) -> Box<dyn Fn(u32) -> u32 + Send + Sync>;
}

#[derive(Args, ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)] //PartialEq - for tests
pub struct BandPassFilterParams {
    #[in_map]
    #[run_config]
    pub low_hz: f64,

    #[in_map]
    #[run_config]
    pub band_hz: f64,

    #[in_map]
    #[run_config]
    pub kernel_size: u32,

    #[in_map]
    #[run_config]
    pub stride: u32,

    #[in_map]
    #[run_config]
    #[command(subcommand)]
    pub padding: Padding,
}

impl InputFilterLengthAdjuster for BandPassFilterParams {
    fn adjust_length(&self) -> Box<dyn Fn(u32) -> u32 + Send + Sync> {
        let kernel_size = self.kernel_size;
        let stride = self.stride;
        let padding = self.padding;
        Box::new(move |l: u32| {
            assert!(kernel_size > 0, "kernel_size({kernel_size}) must be > 0");
            assert!(stride > 0, "stride({stride}) must be > 0");
            assert!(l > 0, "l({l}) must be > 0");
            let pad = kernel_size / 2;
            if let Padding::Reflect(_) = padding {
                assert!(pad < l, "reflect padding requires pad({pad}) < l({l})");
            }
            let padded = l as u64 + 2 * pad as u64;
            assert!(padded >= kernel_size as u64, "invalid conv output length: l={l} kernel_size={kernel_size} stride={stride} pad={pad}");
            let out = (padded - kernel_size as u64) / stride as u64 + 1;
            assert!(out > 0 && out <= u32::MAX as u64, "invalid conv output length: l={l} kernel_size={kernel_size} stride={stride} pad={pad} out={out}");
            out as u32
        })
    }
}

#[derive(Args, ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)] //PartialEq - for tests
pub struct NotchBandFilterParams {
    #[in_map]
    #[run_config]
    pub low_hz: f64,

    #[in_map]
    #[run_config]
    pub high_hz: f64,
}

impl InputFilterLengthAdjuster for NotchBandFilterParams {
    fn adjust_length(&self) -> Box<dyn Fn(u32) -> u32 + Send + Sync> {
        Box::new(|l: u32| l)
    }
}

#[derive(Args, ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)]
pub struct SavGolFilterParams {
    #[in_map]
    #[run_config]
    pub window_len: u32,

    #[in_map]
    #[run_config]
    pub poly_order: i64,

    #[in_map]
    #[run_config]
    pub stride: u32,

    #[in_map]
    #[run_config]
    #[command(subcommand)]
    pub padding: Padding,
}

impl InputFilterLengthAdjuster for SavGolFilterParams {
    fn adjust_length(&self) -> Box<dyn Fn(u32) -> u32 + Send + Sync> {
        let window_len = self.window_len;
        let stride = self.stride;
        let padding = self.padding;
        Box::new(move |l: u32| {
            assert!(window_len > 0 && window_len <= u32::MAX , "window_len({window_len}) must be in 1..={}", u32::MAX);
            let kernel_size = window_len;
            assert!(stride > 0, "stride({stride}) must be > 0");
            assert!(l > 0, "l({l}) must be > 0");
            let pad = kernel_size / 2;
            if let Padding::Reflect(_) = padding {
                assert!(pad < l, "reflect padding requires pad({pad}) < l({l})");
            }
            let padded = l as u64 + 2 * pad as u64;
            assert!(padded >= kernel_size as u64, "invalid conv output length: l={l} window_len={window_len} stride={stride} pad={pad}");
            let out = (padded - kernel_size as u64) / stride as u64 + 1;
            assert!(out > 0 && out <= u32::MAX as u64, "invalid conv output length: l={l} window_len={window_len} stride={stride} pad={pad} out={out}");
            out as u32
        })
    }
}

#[derive(Args, ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)]
pub struct MedianFilterParams {
    #[in_map]
    #[run_config]
    pub window: u32,
}

impl InputFilterLengthAdjuster for MedianFilterParams {
    fn adjust_length(&self) -> Box<dyn Fn(u32) -> u32 + Send + Sync> {
        let window = self.window;

        Box::new(move |l: u32| l.saturating_sub(window))
    }
}

#[derive(Args, ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)]
pub struct AverageFilterParams {
    #[in_map]
    #[run_config]
    pub window: u32,

    #[in_map]
    #[run_config]
    pub step: u32,
}

impl InputFilterLengthAdjuster for AverageFilterParams {
    fn adjust_length(&self) -> Box<dyn Fn(u32) -> u32 + Send + Sync> {
        let window = self.window;
        let step = self.step;

        Box::new(move |l: u32| l.saturating_sub(window) / step)
    }
}


/// Create a sinc-based band-pass filter
pub fn band_pass_filter(
    low_hz: f64,
    high_hz: f64,
    kernel_size: i64,
    sample_rate: f64,
    eps: f64,
) -> Result<Tensor, ThoError> {
    let n = Tensor::f_arange(kernel_size, (Kind::Float, Device::Cpu))?.f_sub_scalar((kernel_size - 1) as f64 / 2.0)?;

    let low = std::f64::consts::TAU * low_hz / sample_rate;
    let high = std::f64::consts::TAU * high_hz / sample_rate;

    let sinc = |x: &Tensor| -> Result<Tensor, ThoError> {
        let x_eps = x.f_add_scalar(eps)?;
        Ok(x.f_sin()?.f_div(&x_eps)?)
    };

    let high_sinc = sinc(&n.f_mul_scalar(high)?)?;
    let low_sinc = sinc(&n.f_mul_scalar(low)?)?;
    let low_pass = high_sinc.f_sub(&low_sinc)?;

    let window = hann_window(kernel_size)?;

    let band_pass = low_pass.f_mul(&window)?;

    let sum = band_pass.f_sum(Kind::Float)?;
    Ok(band_pass.f_div(&sum)?)
}

pub fn hann_window(length: i64) -> Result<Tensor, ThoError> {
    if length <= 1 {
        return Err(ThoError::new(format!("hann_window length must be > 1, got {length}")));
    }

    // w[n] = 0.5 * (1 - cos(2πn / (N-1)))
    let n = Tensor::f_arange(length, (Kind::Float, Device::Cpu))?;
    let factor = 2.0 * std::f64::consts::PI / (length as f64 - 1.0);
    let phase = n.f_mul_scalar(factor)?;
    let cos = phase.f_cos()?;
    Ok(cos.f_neg()?.f_add_scalar(1.0)?.f_mul_scalar(0.5)?)
}

pub fn clc_frequencies(input_signal_0: &Tensor, n: i64, rate: f64, take: usize) -> Result<Vec<(f64, f64)>, ThoError> {
    let input_signal_rfft = input_signal_0.f_fft_rfft(Some(n), 0, "forward")?;

    let absolutes = input_signal_rfft.f_abs()?;
    let freq = Tensor::f_fft_rfftfreq(n, 1. / rate, FLOAT_CPU)?;

    let ranked = absolutes.f_argsort(0, true)?;

    let mut i: Box<dyn Iterator<Item = f32>> = Box::new(from::<Vec<f32>>(&ranked).into_iter());
    if take > 0 {
        i = Box::new(i.take(take));
    }

    Ok(i.map(|i| {
        (
            freq.double_value(&[i as i64]),
            absolutes.double_value(&[i as i64]),
        )
    })
    .collect())
}

pub fn notch_band_rfft(x: &Tensor, fs: f64, fr0: f64, fr1: f64) -> Result<Tensor, ThoError> {
    // Sanity
    if x.dim() != 2 {
        return Err(ThoError::new(format!("expected [C, T], got shape {:?}", x.size())));
    }
    let t = x.size()[1];
    if t <= 1 {
        return Err(ThoError::new(format!("sequence length must be > 1, got {t}")));
    }
    if fs <= 0.0 {
        return Err(ThoError::new(format!("fs must be positive, got {fs}")));
    }

    // Map Hz -> rfft bin indices.
    // rfft output length along time dim is n_fft//2 + 1
    let n = t;
    let nhalf = n / 2;
    let (mut lo, mut hi) = (fr0.min(fr1), fr0.max(fr1));
    // clamp into [0, Nyquist]
    lo = lo.max(0.0);
    hi = hi.min(fs * 0.5);
    if hi <= lo {
        // nothing to do
        return Ok(x.shallow_clone());
    }
    let k0 = ((lo * n as f64 / fs).floor() as i64).clamp(0, nhalf);
    let k1 = ((hi * n as f64 / fs).floor() as i64).clamp(k0, nhalf);

    // FFT along dim=1 (time axis), one-sided complex spectrum
    // (Uses PyTorch’s torch.fft.rfft/irfft through tch’s generated ops.)
    let spec = x.f_fft_rfft(Some(n), 1, "forward")?;

    // Zero the bins in-place for all channels: spec[:, k0..=k1] = 0
    let spec_z = spec.shallow_clone();
    let mut notch_slice = spec_z.f_i((.., k0..(k1 + 1)))?;
    let _ = notch_slice.f_fill_(0)?;

    // Inverse FFT back to time domain, preserve original length n along dim=1
    let y = spec_z.f_fft_irfft(Some(n), 1, "forward")?;
    Ok(y)
}


#[derive(Subcommand, ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Padding {
    /// Pad with zeros (uses conv1d's built-in padding).
    Zero(ZeroPadding),
    /// Mirror the signal at the boundary, **excluding** the edge value (PyTorch "reflect").
    Reflect(ReflectPadding),
    /// Repeat the edge value (PyTorch "replicate").
    Replicate(ReplicatePadding),
}

impl Padding {
    pub fn zero() -> Self {
        Self::Zero(ZeroPadding {})
    }

    pub fn reflect() -> Self {
        Self::Reflect(ReflectPadding {})
    }

    pub fn replicate() -> Self {
        Self::Replicate(ReplicatePadding {})
    }
}

#[derive(Args, ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct ZeroPadding {}

#[derive(Args, ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct ReflectPadding {}

#[derive(Args, ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct ReplicatePadding {}


/// Pad a [N, C, T] tensor along the last dimension by `pad` on both sides.
fn pad_1d(x: &Tensor, pad: i64, mode: Padding) -> Result<Tensor, ThoError> {
    if pad == 0 {
        return Ok(x.shallow_clone());
    }

    assert_eq!(x.dim(), 3, "pad_1d expects [N,C,T], got dim={} size={:?}", x.dim(), x.size());

    let sz = x.size();
    let t = sz[2];

    match mode {
        Padding::Zero(_) => {
            // handled elsewhere (conv padding), but allow here if needed:
            let zeros_l = Tensor::zeros(&[sz[0], sz[1], pad], (x.kind(), x.device()));
            let zeros_r = Tensor::zeros(&[sz[0], sz[1], pad], (x.kind(), x.device()));
            Ok(Tensor::cat(&[&zeros_l, x, &zeros_r], 2))
        }
        Padding::Replicate(_) => {
            let left = x.narrow(2, 0, 1).repeat(&[1, 1, pad]);
            let right = x.narrow(2, t - 1, 1).repeat(&[1, 1, pad]);
            Ok(Tensor::cat(&[&left, x, &right], 2))
        }
        Padding::Reflect(_) => {
            // PyTorch reflect requires pad <= T-1
            if pad >= t {
                return Err(ThoError::new(format!(
                    "reflect padding requires pad < T, got pad={} T={}",
                    pad, t
                )));
            }
            // left: x[..., 1..=pad] reversed
            let left = x.narrow(2, 1, pad).flip(&[2]);
            // right: x[..., T-pad-1 .. T-1) reversed (exclude last element)
            let right = x.narrow(2, t - pad - 1, pad).flip(&[2]);
            Ok(Tensor::cat(&[&left, x, &right], 2))
        }
    }
}

/// Apply Savitzky–Golay smoothing to x shaped [C, T].
/// Keeps length the same. Padding choice: Zero / Reflect / Replicate.
pub fn savgol_smooth_1d_channels(
    x: &Tensor,
    window_len: i64,
    poly_order: i64,
    padding: Padding,
) -> Result<Tensor, ThoError> {

    assert_eq!(x.dim(), 2,  "expected x to have shape [C, T], got dim={} size={:?}", x.dim(), x.size());

    let sz = x.size();
    let c = sz[0];
    let t = sz[1];

    if window_len < 3 || window_len % 2 == 0 {
        return Err(ThoError::new("window_len must be odd and >= 3"));
    }
    if poly_order < 0 || poly_order >= window_len {
        return Err(ThoError::new("poly_order must be >= 0 and < window_len"));
    }
    if t < window_len {
        return Err(ThoError::new(format!("time length T={} must be >= window_len={}", t, window_len)));
    }

    let pad = window_len / 2;

    // Preserve Float/Double if possible; otherwise use Float.
    let kind = match x.kind() {
        Kind::Float | Kind::Double => x.kind(),
        _ => Kind::Float,
    };

    let coeffs = sg_smoothing_coeffs(window_len as usize, poly_order as usize)?;

    // Weight for grouped conv1d: [C, 1, W] (same kernel per channel)
    let w = Tensor::from_slice(&coeffs)
        .to_kind(kind)
        .to_device(x.device())
        .view([1, 1, window_len])
        .repeat(&[c, 1, 1]);

    // Input: [N=1, C, T]
    let x3 = x.to_kind(kind).unsqueeze(0);

    let y3 = match padding {
        Padding::Zero(_) => {
            // conv1d built-in zero padding
            x3.conv1d(&w, None::<Tensor>, 1, pad, 1, c)
        }
        Padding::Reflect(_) | Padding::Replicate(_) => {
            // pre-pad, then conv without padding
            let xpad = pad_1d(&x3, pad, padding)?;
            xpad.conv1d(&w, None::<Tensor>, 1, 0, 1, c)
        }
    };

    Ok(y3.squeeze_dim(0))
}

#[derive(Args, ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)]
pub struct AbsFilterParams {

}

impl InputFilterLengthAdjuster for AbsFilterParams {
    fn adjust_length(&self) -> Box<dyn Fn(u32) -> u32 + Send + Sync> {
        Box::new(move |l: u32| l)
    }
}


#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use crate::transforms::filters::{band_pass_filter, clc_frequencies, hann_window};
    use crate::transforms::utils::sin_t;
    use crate::utils::{from, plot_series, to_vec_f32};
    use log::debug;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::f64;
    use tch::kind::FLOAT_CPU;
    use tch::nn::*;
    use tch::{Device, IndexOp, Kind, Tensor};
    use thoosa_shared::ThoError;
    use thoosa_shared::test_utils::are_close;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_band_pass_filter() -> Result<(), ThoError> {
        let samples = 16000;
        let sample_rate = 16000.0;
        let kernel_size = 101;
        let low_hz = 5.0;
        let high_hz = 15.0;

        let t = Tensor::arange(samples, (Kind::Float, Device::Cpu)) / sample_rate;
        let input_signal_0 = sin_t(&t, 10.);

        let input_signal_1 = sin_t(&t, 300.);

        let input_signal = input_signal_0 + input_signal_1;
        let input_freqs = clc_frequencies(&input_signal, samples, sample_rate, 2)?;
        let input = input_signal.view((1, 1, -1)).expand(&[1, 2, -1], true); // batch x channel for conv1d

        let kernel = band_pass_filter(low_hz, high_hz, kernel_size, sample_rate, 1e-6)?;

        // debug!("kernel.size={:?}", kernel.size());
        // let kernel = kernel.unsqueeze(0).unsqueeze(0).expand(&[1, 2, -1], true); // [out_channels, in_channels/groups, kernel_size]
        let kernel = kernel.repeat([2, 1, 1]); // [out_channels, in_channels/groups, kernel_size]

        //apply filter
        let output = input.conv1d(&kernel, None::<Tensor>, 1, 0, 1, 2);

        let output_freqs = clc_frequencies(&output.i((0, 0, ..)), samples, sample_rate, 5)?;


        assert_eq!([(300.0, 0.5), (10.0, 0.5)], input_freqs.as_slice());
        assert!(are_close(
            &[0.5, 0.2648],
            &[output_freqs[0].1, output_freqs[1].1],
            1e-2
        ));


        /*
            plot_series(
                vec![
                    (format!("input"), plotters::prelude::GREEN,   vec_f32_from(&t).into_iter().zip(from::<Vec<f32>>(&input.i((0, 0, ..)))).collect::<Vec<(f32, f32)>>()),//sig_0_v match
                    (format!("output"), plotters::prelude::RED,     vec_f32_from(&t).into_iter().zip(from::<Vec<f32>>(&output.i((0, 0, ..)))).collect::<Vec<(f32, f32)>>()),//no matches
                ], "band pass", (2000, 1000), &format!("/tmp/plots/filter_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str()
            )?;

           plot_series(
               vec![
                   (format!("0ch"), plotters::prelude::GREEN,   vec_f32_from(&t).into_iter().zip(from::<Vec<f32>>(&input_signal_0_fft)).collect::<Vec<(f32, f32)>>()),//sig_0_v match
                   (format!("1ch"), plotters::prelude::RED,     vec_f32_from(&t).into_iter().zip(from::<Vec<f32>>(&input_signal_1_fft)).collect::<Vec<(f32, f32)>>()),//no matches
               ], "distance", (2000, 1000), &format!("/tmp/plots/filter_fft_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str()
           )?;
       */

        Ok(())
    }

    #[test]
    fn test_band_pass_filter_mult_channels() -> Result<(), ThoError> {
        let samples = 16000;
        let sample_rate = 16000.0;
        let kernel_size = 101;
        let low_hz = 5.0;
        let high_hz = 15.0;

        let t = Tensor::arange(samples, (Kind::Float, Device::Cpu)) / sample_rate;
        let input_signal_0 = sin_t(&t, 10.);

        let input_signal_1 = sin_t(&t, 300.);

        let input_signal = input_signal_0 + input_signal_1;
        let input_freqs = clc_frequencies(&input_signal, samples, sample_rate, 2)?;
        let input = input_signal.view((1, 1, -1)); // batch x channel for conv1d
        let input = input.repeat([10, 3, 1]); // [out_channels, in_channels/groups, kernel_size]

        let kernel = band_pass_filter(low_hz, high_hz, kernel_size, sample_rate, 1e-6)?;

        let kernel = kernel.view([1, 1, -1]).expand([3, 1, -1], true);

        //apply filter
        let output = input.conv1d(&kernel, None::<Tensor>, 1, 0, 1, 3);

        let output_freqs = clc_frequencies(&output.i((0, 0, ..)), samples, sample_rate, 5)?;


        assert_eq!([(300.0, 0.5), (10.0, 0.5)], input_freqs.as_slice());
        //output_freqs -[(10.0, 0.4967866837978363), (300.0, 0.26481497287750244)
        assert!(are_close(
            &[0.5, 0.2648],
            &[output_freqs[0].1, output_freqs[1].1],
            1e-2
        ));

        Ok(())
    }

}
