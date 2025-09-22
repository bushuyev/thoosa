use tch::{Device, Kind, Tensor};

fn generate_morlet_wavelet(center_freq: f64, sample_rate: f64, n_cycles: f64) -> (Tensor, Tensor) {
    let sigma = n_cycles / (2.0 * std::f64::consts::PI * center_freq);
    let half_support = (3.0 * sigma * sample_rate).round() as i64;
    let t = Tensor::arange(2 * half_support + 1, (Kind::Float, Device::Cpu)) - half_support;
    let t = t / sample_rate;

    let real = (-t.pow_tensor_scalar(2.0) / (2.0 * sigma.powi(2))).exp()
        * (2.0 * std::f64::consts::PI * center_freq * &t).cos();
    let imag = (-t.pow_tensor_scalar(2.0) / (2.0 * sigma.powi(2))).exp()
        * (2.0 * std::f64::consts::PI * center_freq * &t).sin();

    (real, imag)
}

#[cfg(test)]
mod tests {
    use crate::transforms::filters::clc_frequencies;
    use crate::transforms::utils::{complex_conv, generate_signal, sin_t};
    use crate::transforms::wavelets::generate_morlet_wavelet;
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
    fn test_morlet_wavelet() -> Result<(), ThoosaError> {
        let sample_rate = 1000.0; // Hz
        let duration_secs = 2.0;
        let samples = (duration_secs * sample_rate) as i64;

        let target_freq = 10.0; // Hz
        let n_cycles = 6.0; // Adjust trade-off time/frequency resolution

        let (wavelet_real, wavelet_imag) =
            generate_morlet_wavelet(target_freq, sample_rate, n_cycles);
        let t = Tensor::arange(
            (sample_rate * duration_secs) as i64,
            (Kind::Float, Device::Cpu),
        ) / sample_rate;

        let input_signal_0 = sin_t(&t, target_freq);
        let input_signal_1 = sin_t(&t, 3. * target_freq);
        // generate_signal(&t, 1.0, target_freq).view((1, 1, -1));
        let input = (input_signal_0 + input_signal_1);
        let input_freqs = clc_frequencies(&input, samples, sample_rate, 2);
        let input = input.view((1, 1, -1));

        let envelope = complex_conv(&input, &wavelet_real, &wavelet_imag);

        let output_freqs = clc_frequencies(&envelope, samples, sample_rate, 5);

        debug!("input_freqs={input_freqs:?}");
        debug!("output_freqs={output_freqs:?}");

        plot_series(
            vec![
                // (format!("0ch"), plotters::prelude::GREEN,   vec_f32_from(&t).into_iter().zip(from::<Vec<f32>>(&signal.i((0, 0, ..)))).collect::<Vec<(f32, f32)>>()),//sig_0_v match
                (
                    format!("1ch"),
                    plotters::prelude::RED,
                    vec_f32_from(&t)
                        .into_iter()
                        .zip(from::<Vec<f32>>(&envelope))
                        .collect::<Vec<(f32, f32)>>(),
                ), //no matches
            ],
            "wavelet",
            (2000, 1000),
            &format!(
                "/tmp/plots/wavelet_{:?}.png",
                chrono::Utc::now().to_rfc3339()
            )
            .as_str(),
        )?;

        //TODO

        Ok(())
    }
}
