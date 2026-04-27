use std::f32::consts::TAU;

/// Generate signal containing the specified frequencies.
///
/// `n`           - number of samples
/// `sample_rate` - samples per second, e.g. 44_100
/// `freqs`       - frequencies in Hz
pub fn compose_freqs(
    n: usize,
    sample_rate: usize,
    freqs: Vec<f32>,
) -> Vec<f32> {
    let sample_rate = sample_rate as f32;

    let mut signal = vec![0.0_f32; n];

    for &freq in &freqs {
        for i in 0..n {
            let t = i as f32 / sample_rate;
            signal[i] += (TAU * freq * t).sin();
        }
    }

    // Optional normalization, so amplitude does not grow with number of freqs.
    if !freqs.is_empty() {
        let scale = 1.0 / freqs.len() as f32;
        for x in &mut signal {
            *x *= scale;
        }
    }

    signal
}

#[cfg(test)]
mod tests {
    use rustfft::FftPlanner;
    use rustfft::num_complex::Complex;
    use crate::audio_utils::compose_freqs;

    #[test]
    fn test_compose_freqs_has_expected_fft_peaks() {
        let n = 1024;
        let sample_rate = 1024;

        let expected_freqs = vec![50.0, 120.0, 300.0];

        let signal = compose_freqs(
            n,
            sample_rate,
            expected_freqs.clone(),
        );

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);

        let mut buffer: Vec<Complex<f32>> = signal
            .into_iter()
            .map(|x| Complex { re: x, im: 0.0 })
            .collect();

        fft.process(&mut buffer);

        // For real input, positive frequencies are in bins 0..=n/2.
        let magnitudes: Vec<f32> = buffer[..=n / 2]
            .iter()
            .map(|c| c.norm())
            .collect();

        let mut peak_bins: Vec<usize> = magnitudes
            .iter()
            .enumerate()
            // Ignore DC bin.
            .filter(|(bin, _)| *bin != 0)
            .filter(|&(_, &mag)| mag > 100.0)
            .map(|(bin, _)| bin)
            .collect();

        peak_bins.sort();

        let expected_bins: Vec<usize> = expected_freqs
            .iter()
            .map(|&f| f as usize)
            .collect();

        assert_eq!(peak_bins, expected_bins);
    }
}