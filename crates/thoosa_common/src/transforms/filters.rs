use crate::utils::from;
use log::debug;
use serde::Deserialize;
use tch::kind::FLOAT_CPU;
use tch::{Device, Kind, Tensor};
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)] //PartialEq - for tests
pub struct BandPassFilterParams {
    #[in_map]
    #[run_config]
    pub low_hz: f64,

    #[in_map]
    #[run_config]
    pub band_hz: f64,

    #[in_map]
    #[run_config]
    pub kernel_size: i64,

    #[in_map]
    #[run_config]
    pub stride: i64,

    #[in_map]
    #[run_config]
    pub padding: i64,
}

/// Create a sinc-based band-pass filter
pub fn band_pass_filter(
    low_hz: f64,
    high_hz: f64,
    kernel_size: i64,
    sample_rate: f64,
    eps: f64,
) -> Tensor {
    let n =
        Tensor::arange(kernel_size, (Kind::Float, Device::Cpu)) - (kernel_size - 1) as f64 / 2.0;

    let low = std::f64::consts::TAU * low_hz / sample_rate;
    let high = std::f64::consts::TAU * high_hz / sample_rate;

    let sinc = |x: &Tensor| {
        let x_eps = x + eps;
        x.sin() / x_eps
    };

    let low_pass = sinc(&(n.copy() * high)) - sinc(&(n.copy() * low));

    let window = hann_window(kernel_size);

    let band_pass: Tensor = low_pass * window;

    let sum = band_pass.sum(Kind::Float);
    band_pass / sum
}

pub fn hann_window(length: i64) -> Tensor {
    // w[n] = 0.5 * (1 - cos(2πn / (N-1)))
    let n = Tensor::arange(length, (Kind::Float, Device::Cpu));
    let factor = 2.0 * std::f64::consts::PI / (length as f64 - 1.0);
    0.5 * (1.0 - (n * factor).cos())
}

pub fn clc_frequencies(input_signal_0: &Tensor, n: i64, rate: f64, take: usize) -> Vec<(f64, f64)> {
    let input_signal_rfft = input_signal_0.fft_rfft(n, 0, "forward");

    let absolutes = input_signal_rfft.abs();
    let freq = Tensor::fft_rfftfreq(n, 1. / rate, FLOAT_CPU);

    let ranked = absolutes.argsort(0, true);

    let mut i: Box<dyn Iterator<Item = f32>> = Box::new(from::<Vec<f32>>(&ranked).into_iter());
    if take > 0 {
        i = Box::new(i.take(take));
    }

    i.map(|i| {
        (
            freq.double_value(&[i as i64]),
            absolutes.double_value(&[i as i64]),
        )
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use crate::transforms::filters::{band_pass_filter, clc_frequencies};
    use crate::transforms::utils::sin_t;
    use crate::utils::{from, plot_series, vec_f32_from};
    use log::debug;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::f64;
    use tch::kind::FLOAT_CPU;
    use tch::nn::*;
    use tch::{Device, IndexOp, Kind, Tensor};
    use thoosa_shared::ThoosaError;
    use thoosa_shared::test_utils::are_close;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_band_pass_filter() -> Result<(), ThoosaError> {
        let samples = 16000;
        let sample_rate = 16000.0;
        let kernel_size = 101;
        let low_hz = 5.0;
        let high_hz = 15.0;

        let t = Tensor::arange(samples, (Kind::Float, Device::Cpu)) / sample_rate;
        let input_signal_0 = sin_t(&t, 10.);

        let input_signal_1 = sin_t(&t, 300.);

        let input_signal = input_signal_0 + input_signal_1;
        let input_freqs = clc_frequencies(&input_signal, samples, sample_rate, 2);
        let input = input_signal.view((1, 1, -1)); // batch x channel for conv1d

        let kernel = band_pass_filter(low_hz, high_hz, kernel_size, sample_rate, 1e-6);
        let kernel = kernel.unsqueeze(0).unsqueeze(0); // [out_channels, in_channels/groups, kernel_size]

        //apply filter
        let output = input.conv1d(&kernel, None::<Tensor>, 1, 0, 1, 1);
        // debug!("t.size={:?}, input.size={:?}, output.size={:?}", t.size(), input.size(), output.size());

        let output_freqs = clc_frequencies(&output.i((0, 0, ..)), samples, sample_rate, 5);

        debug!("input_freqs={input_freqs:?}");
        debug!("output_freqs={output_freqs:?}");

        assert_eq!([(300.0, 0.5), (10.0, 0.5)], input_freqs.as_slice());
        //output_freqs -[(10.0, 0.4967866837978363), (300.0, 0.26481497287750244)
        assert!(are_close(
            &[0.5, 0.2648],
            &[output_freqs[0].1, output_freqs[1].1],
            1e-2
        ));

        /*

        plot_series(
            vec![
                (format!("0ch"), plotters::prelude::GREEN,   vec_f32_from(&t).into_iter().zip(from::<Vec<f32>>(&input.i((0, 0, ..)))).collect::<Vec<(f32, f32)>>()),//sig_0_v match
                (format!("1ch"), plotters::prelude::RED,     vec_f32_from(&t).into_iter().zip(from::<Vec<f32>>(&output.i((0, 0, ..)))).collect::<Vec<(f32, f32)>>()),//no matches
            ], "distance", (2000, 1000), &format!("/tmp/plots/filter_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str()
        )?;

        plot_series(
            vec![
                (format!("0ch"), plotters::prelude::GREEN,   vec_f32_from(&t).into_iter().zip(from::<Vec<f32>>(&input_signal_0_fft)).collect::<Vec<(f32, f32)>>()),//sig_0_v match
                (format!("1ch"), plotters::prelude::RED,     vec_f32_from(&t).into_iter().zip(from::<Vec<f32>>(&input_signal_1_fft)).collect::<Vec<(f32, f32)>>()),//no matches
            ], "distance", (2000, 1000), &format!("/tmp/plots/filter_fft_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str()
        )?;*/

        Ok(())
    }

    #[test]
    fn test_band_pass_filter_mult_channels() -> Result<(), ThoosaError> {
        let samples = 16000;
        let sample_rate = 16000.0;
        let kernel_size = 101;
        let low_hz = 5.0;
        let high_hz = 15.0;

        let t = Tensor::arange(samples, (Kind::Float, Device::Cpu)) / sample_rate;
        let input_signal_0 = sin_t(&t, 10.);

        let input_signal_1 = sin_t(&t, 300.);

        let input_signal = input_signal_0 + input_signal_1;
        let input_freqs = clc_frequencies(&input_signal, samples, sample_rate, 2);
        let input = input_signal.view((1, 1, -1)); // batch x channel for conv1d
        debug!("0. input={input:?}");
        let input = input.repeat([10, 3, 1]); // [out_channels, in_channels/groups, kernel_size]
        debug!("1. input={input:?}");

        let kernel = band_pass_filter(low_hz, high_hz, kernel_size, sample_rate, 1e-6);
        debug!("1. kernel={kernel:?}");
        // let kernel = kernel.repeat([3,1,1]);
        let kernel = kernel.view([1, 1, -1]).expand([3, 1, -1], true);
        debug!("1. kernel={kernel:?}");

        //apply filter
        let output = input.conv1d(&kernel, None::<Tensor>, 1, 0, 1, 3);

        let output_freqs = clc_frequencies(&output.i((0, 0, ..)), samples, sample_rate, 5);

        debug!("input_freqs={input_freqs:?}");
        debug!("output_freqs={output_freqs:?}");

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
