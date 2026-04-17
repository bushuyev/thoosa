use anyhow::{anyhow, bail, Context, Result};
use tch::{Device, Kind, Tensor};

#[derive(Debug)]
pub struct AudioFeatures {
    /// Raw STFT output from tch.
    /// In this binding it is expected to be real-valued with trailing size 2:
    /// either [B, F, N, 2] or [B, N, F, 2].
    pub stft_raw: Tensor,

    /// Canonicalized magnitude spectrogram [B, F, N].
    pub stft_magnitude: Tensor,

    /// log(stft_magnitude + eps), shape [B, F, N].
    pub log_magnitude: Tensor,

    /// Mel spectrogram [B, M, N].
    pub mel_spectrogram: Tensor,

    /// log(mel_spectrogram + eps), shape [B, M, N].
    pub log_mel_spectrogram: Tensor,
}

fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

fn make_mel_filterbank(
    sample_rate: i64,
    n_freq: i64,
    n_mels: i64,
    f_min: f64,
    f_max: f64,
    device: Device,
) -> Result<Tensor> {
    if n_freq <= 0 {
        bail!("n_freq must be > 0, got {n_freq}");
    }
    if n_mels <= 0 {
        bail!("n_mels must be > 0, got {n_mels}");
    }
    if !(0.0..=sample_rate as f64 / 2.0).contains(&f_min) {
        bail!(
            "f_min must be in [0, sr/2], got f_min={} sr={}",
            f_min,
            sample_rate
        );
    }
    if !(f_min..=sample_rate as f64 / 2.0).contains(&f_max) {
        bail!(
            "f_max must be in [f_min, sr/2], got f_min={} f_max={} sr={}",
            f_min,
            f_max,
            sample_rate
        );
    }

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    let mut mel_points = Vec::with_capacity((n_mels + 2) as usize);
    for i in 0..(n_mels + 2) {
        let t = i as f64 / (n_mels + 1) as f64;
        mel_points.push(mel_min + t * (mel_max - mel_min));
    }

    let hz_points: Vec<f64> = mel_points.into_iter().map(mel_to_hz).collect();

    // We assume n_freq bins span [0, sr/2].
    // For onesided STFT this is exact.
    // If the original STFT was two-sided, we will already have narrowed it to the positive half.
    let denom = (n_freq - 1).max(1) as f64;
    let fft_bin_hz: Vec<f64> = (0..n_freq)
        .map(|k| k as f64 * (sample_rate as f64 / 2.0) / denom)
        .collect();

    let mut fb = vec![0f32; (n_mels * n_freq) as usize];

    for m in 0..n_mels {
        let left = hz_points[m as usize];
        let center = hz_points[(m + 1) as usize];
        let right = hz_points[(m + 2) as usize];

        for k in 0..n_freq {
            let f = fft_bin_hz[k as usize];
            let val = if f < left || f > right {
                0.0
            } else if f <= center {
                let denom = (center - left).max(1e-12);
                ((f - left) / denom).max(0.0)
            } else {
                let denom = (right - center).max(1e-12);
                ((right - f) / denom).max(0.0)
            };

            fb[(m * n_freq + k) as usize] = val as f32;
        }
    }

    Ok(Tensor::f_from_slice(&fb)?
        .f_reshape([n_mels, n_freq])?
        .f_to_device(device)?)
}

/// Convert raw STFT output into canonical magnitude shape [B, F, N].
///
/// Handles:
/// - [B, F, N, 2]
/// - [B, N, F, 2]
///
/// Also handles two-sided spectra:
/// - if F == n_fft, keeps only bins [0 .. n_fft/2] inclusive, so final F becomes n_fft/2 + 1.
fn canonicalize_stft_magnitude(stft_raw: &Tensor, n_fft: i64) -> Result<Tensor> {
    let sz = stft_raw.size();
    if sz.len() != 4 {
        bail!("Expected raw STFT rank 4, got shape {:?}", sz);
    }
    if sz[3] != 2 {
        bail!(
            "Expected raw STFT trailing complex-pair dimension of size 2, got shape {:?}",
            sz
        );
    }

    let d1 = sz[1];
    let d2 = sz[2];
    let onesided = n_fft / 2 + 1;

    // Compute magnitude from real/imag.
    let real = stft_raw.f_select(-1, 0)?;
    let imag = stft_raw.f_select(-1, 1)?;
    let mag = real
        .f_mul(&real)?
        .f_add(&imag.f_mul(&imag)?)?
        .f_sqrt()?;

    // Decide whether mag is [B, F, N] already or [B, N, F].
    let mag_bfn = if d1 == n_fft || d1 == onesided {
        mag
    } else if d2 == n_fft || d2 == onesided {
        mag.f_transpose(1, 2)?
    } else {
        bail!(
            "Could not identify frequency axis in STFT shape {:?}; expected one axis to be n_fft={} or n_fft/2+1={}",
            sz,
            n_fft,
            onesided
        );
    };

    let sz2 = mag_bfn.size();
    if sz2.len() != 3 {
        bail!("Expected canonical magnitude rank 3, got {:?}", sz2);
    }

    let f = sz2[1];

    // If full two-sided spectrum is present, keep only nonnegative frequencies.
    if f == n_fft {
        Ok(mag_bfn.f_narrow(1, 0, onesided)?)
    } else if f == onesided {
        Ok(mag_bfn)
    } else {
        bail!(
            "Unexpected frequency dimension after canonicalization: got F={}, expected {} or {}",
            f,
            n_fft,
            onesided
        );
    }
}

pub fn audio_features(
    waveform: &Tensor, // [T] or [B, T]
    sample_rate: i64,
    n_fft: i64,
    win_length: i64,
    hop_length: i64,
    n_mels: i64,
    f_min: f64,
    f_max: Option<f64>,
    eps: f64,
) -> Result<AudioFeatures> {
    if sample_rate <= 0 {
        bail!("sample_rate must be > 0, got {sample_rate}");
    }
    if n_fft <= 0 || win_length <= 0 || hop_length <= 0 {
        bail!(
            "n_fft, win_length, hop_length must be > 0, got n_fft={} win_length={} hop_length={}",
            n_fft,
            win_length,
            hop_length
        );
    }
    if win_length > n_fft {
        bail!("win_length must be <= n_fft, got win_length={} n_fft={}", win_length, n_fft);
    }
    if eps <= 0.0 {
        bail!("eps must be > 0, got {eps}");
    }

    let device = waveform.device();
    let f_max = f_max.unwrap_or(sample_rate as f64 / 2.0);

    let waveform = match waveform.dim() {
        1 => waveform.f_unsqueeze(0)?,
        2 => waveform.shallow_clone(),
        d => bail!("Expected waveform [T] or [B, T], got dim={d} shape={:?}", waveform.size()),
    };

    let window = Tensor::f_hann_window(win_length, (Kind::Float, device))?;

    // Important:
    // tch/libtorch bindings vary. In your environment this returns a real tensor
    // with trailing dimension 2, not a native complex tensor.
    //
    // We do not rely on the exact boolean semantics here; instead we canonicalize
    // the result afterward and, if necessary, drop the negative-frequency half.
    let stft_raw = waveform.f_stft(
        n_fft,
        Some(hop_length),
        Some(win_length),
        Some(&window),
        true,
        false,
        false,
        false,
    )?;

    let stft_magnitude =
        canonicalize_stft_magnitude(&stft_raw, n_fft).context("failed to canonicalize STFT output")?;

    let log_magnitude = stft_magnitude.f_add_scalar(eps)?.f_log()?;
    let power_spectrogram = stft_magnitude.f_mul(&stft_magnitude)?;

    let power_sz = power_spectrogram.size();
    if power_sz.len() != 3 {
        bail!("Expected power spectrogram [B, F, N], got {:?}", power_sz);
    }

    let actual_n_freq = power_sz[1];

    let mel_fb = make_mel_filterbank(
        sample_rate,
        actual_n_freq,
        n_mels,
        f_min,
        f_max,
        device,
    )?; // [M, F]

    let mel_fb_sz = mel_fb.size();
    if mel_fb_sz != vec![n_mels, actual_n_freq] {
        bail!(
            "Unexpected mel filterbank shape {:?}, expected [{}, {}]",
            mel_fb_sz,
            n_mels,
            actual_n_freq
        );
    }

    // [B, F, N] -> [B, N, F]
    // [M, F] -> [F, M]
    // result [B, N, M] -> [B, M, N]
    let mel_spectrogram = power_spectrogram
        .f_transpose(1, 2)?                // [B, N, F]
        .f_matmul(&mel_fb.f_transpose(0, 1)?)? // [B, N, M]
        .f_transpose(1, 2)?;               // [B, M, N]

    let log_mel_spectrogram = mel_spectrogram.f_add_scalar(eps)?.f_log()?;

    Ok(AudioFeatures {
        stft_raw,
        stft_magnitude,
        log_magnitude,
        mel_spectrogram,
        log_mel_spectrogram,
    })
}

#[test]
fn test_audio_features() -> Result<()> {
    let device = Device::Cpu;
    let sample_rate = 32_000;
    let seconds = 5;
    let t = sample_rate * seconds;

    let waveform = Tensor::f_randn([t], (Kind::Float, device))?;

    let feats = audio_features(
        &waveform,
        sample_rate,
        1024,  // n_fft
        1024,  // win_length
        320,   // hop_length
        128,   // n_mels
        0.0,   // f_min
        None,  // f_max
        1e-8,  // eps
    )?;

    println!("stft_raw             {:?}", feats.stft_raw.size());
    println!("stft_magnitude       {:?}", feats.stft_magnitude.size());
    println!("log_magnitude        {:?}", feats.log_magnitude.size());
    println!("mel_spectrogram      {:?}", feats.mel_spectrogram.size());
    println!("log_mel_spectrogram  {:?}", feats.log_mel_spectrogram.size());

    let stft_mag_sz = feats.stft_magnitude.size();
    let mel_sz = feats.mel_spectrogram.size();
    let log_mel_sz = feats.log_mel_spectrogram.size();

    assert_eq!(stft_mag_sz.len(), 3);
    assert_eq!(stft_mag_sz[0], 1);
    assert_eq!(stft_mag_sz[1], 1024 / 2 + 1);

    assert_eq!(mel_sz.len(), 3);
    assert_eq!(mel_sz[0], 1);
    assert_eq!(mel_sz[1], 128);

    assert_eq!(log_mel_sz, mel_sz);

    Ok(())
}