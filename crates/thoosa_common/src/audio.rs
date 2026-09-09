use std::cmp::Ordering;
use log::debug;
use tch::{Device, Kind, TchError, Tensor};

use thoosa_shared::{panic_to_err, ThoError};

pub const LOG_EPS:f64 = 1e-6;


pub fn build_mel_filterbank(
    sample_rate: f64,
    n_fft: i64,
    n_mels: i64,
    f_min: f64,
    f_max: f64,
    device: Device,
) -> Result<Tensor, ThoError> {
    let freq_bins = n_fft / 2 + 1;
    let f_max = if f_max > 0.0 {
        f_max
    } else {
        sample_rate / 2.0
    };

    if f_min < 0.0 || f_max <= f_min || f_max > sample_rate / 2.0 {
        return Err(ThoError::new(format!(
            "invalid mel frequency range: f_min={f_min}, f_max={f_max}, sample_rate={sample_rate}"
        )));
    }

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    let mut mel_points = Vec::with_capacity((n_mels + 2) as usize);
    for i in 0..(n_mels + 2) {
        let t = i as f64 / (n_mels + 1) as f64;
        let mel = mel_min + t * (mel_max - mel_min);
        mel_points.push(mel_to_hz(mel));
    }

    let mut bins = Vec::with_capacity((n_mels + 2) as usize);
    for hz in mel_points {
        let bin = ((n_fft as f64 + 1.0) * hz / sample_rate).floor() as i64;
        bins.push(bin.clamp(0, freq_bins - 1));
    }

    let mut fb = vec![0.0f32; (n_mels * freq_bins) as usize];

    for m in 0..n_mels {
        let left = bins[m as usize];
        let center = bins[(m + 1) as usize];
        let right = bins[(m + 2) as usize];

        if center <= left || right <= center {
            continue;
        }

        for k in left..center {
            let value = (k - left) as f32 / (center - left) as f32;
            fb[(m * freq_bins + k) as usize] = value;
        }

        for k in center..right {
            let value = (right - k) as f32 / (right - center) as f32;
            fb[(m * freq_bins + k) as usize] = value;
        }
    }

    Ok(Tensor::from_slice(&fb)
        .f_view([n_mels, freq_bins])?
        .to_device(device))
}

fn hz_to_mel(hz: f64) -> f64 {
    // HTK-style mel scale.
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}
//---buckets
/// Assign log-mel-like patches to buckets, keeping channels separate.
///
/// Input shape:
///
///     [channels, patches, n_mels, patch_frames]
///
/// Output shape:
///
///     [channels, patches]
///
/// Each value is in:
///
///     0 .. n_buckets - 1
///
/// Important:
///
/// Bucket ids are local to each channel. So:
///
///     output[0, p] = 5
///     output[1, q] = 5
///
/// do not necessarily mean the same bucket semantically. They are both
/// bucket 5 inside their own channel namespace.
pub fn assign_lsh_buckets_per_channel(
    patches: &Tensor,
    n_buckets: i64,
) -> Result<Vec<Vec<usize>>, ThoError> {
    if n_buckets <= 0 {
        return Err(ThoError::new(format!(
            "n_buckets must be positive, got {n_buckets}"
        )));
    }

    let size = patches.size();
    if size.len() != 4 {
        return Err(ThoError::new(format!(
            "expected [channels, patches, n_mels, patch_frames], got {size:?}"
        )));
    }

    let channels = size[0];
    let n_patches = size[1];

    assert!(channels > 0, "unexpected channels={channels}");
    assert!(n_patches > 0, "unexpected n_patches={n_patches}");


    let n_bits = hash_bits_for_bucket_count(n_buckets);

    assign_lsh_buckets_per_channel_with_bits(patches, n_buckets, n_bits)
}

/// Same as `assign_lsh_buckets_per_channel`, but lets you control the hash size.
///
/// Reasonable values: 12..32.
pub fn assign_lsh_buckets_per_channel_with_bits(
    patches: &Tensor,
    n_buckets: i64,
    n_bits: i64,
) -> Result<Vec<Vec<usize>>, ThoError> {
    debug!("assign_lsh_buckets_per_channel_with_bits: patches={patches:?} n_buckets={n_buckets} n_bits={n_bits}");
    if n_buckets <= 0 {
        return Err(ThoError::new(format!(
            "n_buckets must be positive, got {n_buckets}"
        )));
    }

    if n_bits <= 0 || n_bits > 63 {
        return Err(ThoError::new(format!(
            "n_bits must be in 1..=63, got {n_bits}"
        )));
    }

    let size = patches.size();
    if size.len() != 4 {
        return Err(ThoError::new(format!(
            "expected [channels, patches, n_mels, patch_frames], got {size:?}"
        )));
    }

    let channels = size[0];
    let n_patches = size[1];
    let n_mels = size[2];
    let patch_frames = size[3];

    assert!(channels > 0, "unexpected channels={channels}");
    assert!(n_patches > 0, "unexpected n_patches={n_patches}");

    let device = patches.device();
    let input_dim = n_mels * patch_frames;

    // ---------------------------------------------------------------------
    // Flatten each patch independently:
    //
    // [C, P, M, T] -> [C, P, M * T]
    // ---------------------------------------------------------------------
    debug!("patches.size={:?} channels={channels}, n_patches={n_patches}, input_dim={input_dim}", patches.size());
    let x = patches
        .f_to_kind(Kind::Float)?
        .f_reshape([channels, n_patches, input_dim])?;

    // ---------------------------------------------------------------------
    // Per-patch normalization.
    //
    // Normalize over [M * T], independently for every [channel, patch].
    // This keeps the hash mostly shape/spectrum-based rather than pure
    // loudness-based.
    // ---------------------------------------------------------------------
    let mean = x.f_mean_dim(&[2i64][..], true, Kind::Float)?;
    let centered = x.f_sub(&mean)?;

    let var = panic_to_err!(centered
        .f_pow_tensor_scalar(2.0)?
        .f_mean_dim(&[2i64][..], true, Kind::Float))?;

    let std = var.f_add_scalar(1e-6)?.f_sqrt()?;
    let x_norm = centered.f_div(&std)?;

    // ---------------------------------------------------------------------
    // Deterministic projection:
    //
    // [input_dim, n_bits]
    //
    // Shared projection across channels. Channels remain separate because
    // we do not flatten C into P for the output.
    // ---------------------------------------------------------------------
    let proj = deterministic_projection(input_dim, n_bits, device)?;

    // ---------------------------------------------------------------------
    // Project:
    //
    // [C, P, D] @ [D, n_bits] -> [C, P, n_bits]
    // ---------------------------------------------------------------------
    let scores = x_norm.f_matmul(&proj)?;

    let bits = scores
        .f_gt(0.0)?
        .f_to_kind(Kind::Int64)?
        .to_device(Device::Cpu)
        .f_contiguous()?;

    debug!("assign_lsh_buckets_per_channel_with_bits: bits={bits:?}");
    let bits_vec = Vec::<i64>::try_from(&bits.f_reshape([-1])?.f_contiguous()?)?;
    debug!("assign_lsh_buckets_per_channel_with_bits: bits_vec={bits_vec:?}");

    let mut channels_bucket_ids = vec![Vec::with_capacity(n_patches as usize); channels as usize];

    for c in 0..channels {
        for p in 0..n_patches {
            let row_start = ((c * n_patches + p) * n_bits) as usize;
            let row_end = row_start + n_bits as usize;
            let row = &bits_vec[row_start..row_end];

            // Include the channel id in the hash salt so that channels get
            // independent bucket partitioning even if their patch values are
            // identical.
            let hash = hash_bit_row_with_channel_salt(row, c as u64);
            let bucket = (hash % n_buckets as u64) as usize;

            channels_bucket_ids[c as usize].push(bucket);
        }
    }
    debug!("assign_lsh_buckets_per_channel_with_bits: bucket_ids={channels_bucket_ids:?}, channels={channels}, n_patches={n_patches}");


    Ok(channels_bucket_ids)
}

fn hash_bits_for_bucket_count(n_buckets: i64) -> i64 {
    let mut bits = 0i64;
    let mut x = 1i64;

    while x < n_buckets {
        x <<= 1;
        bits += 1;
    }

    // Extra bits reduce collisions and modulo bias.
    (bits + 4).clamp(8, 32)
}

fn deterministic_projection(
    input_dim: i64,
    n_bits: i64,
    device: Device,
) -> Result<Tensor, TchError> {
    let i = Tensor::arange(input_dim, (Kind::Float, device))
        .f_view([input_dim, 1])?;

    let j = Tensor::arange(n_bits, (Kind::Float, device))
        .f_view([1, n_bits])?;

    // Cheap deterministic pseudo-random-ish values in [-1, 1].
    //
    // Shape:
    //
    //     [input_dim, n_bits]
    let proj = i
        .f_mul_scalar(12.9898)?
        .f_add(&j.f_mul_scalar(78.233)?)?
        .f_sin()?
        .f_mul_scalar(43_758.5453)?
        .f_sin()?;

    Ok(proj)
}

fn hash_bit_row_with_channel_salt(bits: &[i64], channel: u64) -> u64 {
    // FNV-like mixing with a per-channel salt.
    let mut h = 0xcbf29ce484222325u64;

    h ^= channel.wrapping_mul(0x9e3779b97f4a7c15);
    h = h.wrapping_mul(0x100000001b3);

    for (i, &bit) in bits.iter().enumerate() {
        if bit != 0 {
            let v = (i as u64).wrapping_mul(0x9e3779b97f4a7c15);
            h ^= v;
            h = h.wrapping_mul(0x100000001b3);
        } else {
            h ^= 0x517cc1b727220a95u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }

    h
}


#[derive(Debug, Clone)]
pub struct FreqPeak {
    pub hz: f32,
    pub magnitude: f32,
    pub bin: i64,
}

/// `audio`: Tensor of shape [channels, audio_data]
/// `sample_rate`: e.g. 48_000.0
/// `n_peaks`: number of strongest frequencies per channel
pub fn strongest_freqs_per_channel(
    audio: &Tensor,
    sample_rate: f32,
    n_peaks: i64,
) -> Result<Vec<Vec<FreqPeak>>, TchError> {
    let size = audio.size();
    assert_eq!(size.len(), 2, "expected [channels, audio_data]");

    let channels = size[0];
    let audio_len = size[1];

    assert!(audio_len > 1);
    assert!(n_peaks > 0);

    // Ensure floating point input.
    let audio = audio.to_kind(Kind::Float);

    // FFT over the last dimension: [channels, audio_data] -> [channels, freq_bins]
    //
    // rfft output is complex.
    // Number of bins is audio_len / 2 + 1.
    let spectrum = audio.f_fft_rfft(None, -1, "backward")?;

    // Complex magnitude: [channels, freq_bins]
    let magnitude = spectrum.f_abs()?;

    let bins = magnitude.size()[1];

    // Ignore DC bin 0, otherwise "frequency 0" can dominate.
    let magnitude_no_dc = magnitude.f_narrow(1, 1, bins - 1)?;

    let k = n_peaks.min(bins - 1);

    // topk along frequency-bin dimension.
    //
    // values:  [channels, k]
    // indices: [channels, k], but relative to `magnitude_no_dc`,
    // so real FFT bin = index + 1.
    let (values, indices) = magnitude_no_dc.f_topk(k, 1, true, true)?;

    let values = values.to_device(Device::Cpu);
    let indices = indices.to_device(Device::Cpu);

    let values_flat: Vec<f32> = Vec::<f32>::try_from(values.f_reshape([-1])?)?;
    let indices_flat: Vec<i64> = Vec::<i64>::try_from(indices.f_reshape([-1])?)?;

    let mut result = Vec::with_capacity(channels as usize);

    for ch in 0..channels {
        let mut ch_peaks = Vec::with_capacity(k as usize);

        for j in 0..k {
            let flat_idx = (ch * k + j) as usize;

            let fft_bin = indices_flat[flat_idx] + 1;
            let magnitude = values_flat[flat_idx];

            let hz = fft_bin as f32 * sample_rate / audio_len as f32;

            ch_peaks.push(FreqPeak {
                hz,
                magnitude,
                bin: fft_bin,
            });
        }

        result.push(ch_peaks);
    }

    Ok(result)
}

pub fn strongest_freqs_per_channel_windowed(
    audio: &Tensor,
    sample_rate: f32,
    n_peaks: i64,
) -> Result<Vec<Vec<FreqPeak>>, TchError> {
    let size = audio.size();
    assert_eq!(size.len(), 2);

    let audio_len = size[1];

    let audio = audio.to_kind(Kind::Float);

    // Remove per-channel DC offset.
    let audio = &audio - audio.f_mean_dim([1].as_slice(), true, Kind::Float)?;

    // Hann window: [audio_data]
    let window = Tensor::hann_window(audio_len, (Kind::Float, audio.device()));

    // Broadcast window over channels: [channels, audio_data]
    let audio = audio * window;

    strongest_freqs_per_channel(&audio, sample_rate, n_peaks)
}

fn parabolic_interpolated_bin(m: &[f32], bin: usize) -> f32 {
    if bin == 0 || bin + 1 >= m.len() {
        return bin as f32;
    }

    let ym1 = m[bin - 1];
    let y0 = m[bin];
    let yp1 = m[bin + 1];

    let denom = ym1 - 2.0 * y0 + yp1;

    if denom.abs() < 1e-12 {
        return bin as f32;
    }

    // Offset relative to integer bin.
    // For a peak halfway between two bins, this becomes approximately +/-0.5.
    let delta = 0.5 * (ym1 - yp1) / denom;

    bin as f32 + delta.clamp(-0.5, 0.5)
}

pub fn strongest_freqs_per_channel_windowed_nms(
    audio: &Tensor,
    sample_rate: f32,
    n_peaks: usize,

    // Peaks closer than this are treated as the same peak.
    //
    // For your example, something like 20.0 Hz is reasonable.
    min_separation_hz: f32,

    // Stop after peaks weaker than this fraction of the strongest peak.
    //
    // Use 0.0 to disable.
    // Use e.g. 0.05 to ignore small leakage/sidelobes.
    min_relative_magnitude: f32,
) -> Result<Vec<Vec<FreqPeak>>, TchError> {
    let size = audio.size();
    assert_eq!(size.len(), 2, "expected [channels, audio_data]");

    let audio_len = size[1];
    assert!(audio_len > 1);
    assert!(n_peaks > 0);

    let audio = audio.to_kind(Kind::Float);

    // Remove per-channel DC offset.
    let audio = &audio - audio.f_mean_dim([1i64].as_slice(), true, Kind::Float)?;

    // Hann window reduces spectral leakage, but also spreads one true frequency
    // over a few neighboring bins. That is why NMS is useful.
    let window = Tensor::hann_window(audio_len, (Kind::Float, audio.device()));
    let audio = audio.f_mul(&window)?;

    strongest_freqs_per_channel_nms(
        &audio,
        sample_rate,
        n_peaks,
        min_separation_hz,
        min_relative_magnitude,
    )
}

pub fn strongest_freqs_per_channel_nms(
    audio: &Tensor,
    sample_rate: f32,
    n_peaks: usize,
    min_separation_hz: f32,
    min_relative_magnitude: f32,
) -> Result<Vec<Vec<FreqPeak>>, TchError> {
    let size = audio.size();
    assert_eq!(size.len(), 2, "expected [channels, audio_data]");

    let channels = size[0] as usize;
    let audio_len = size[1] as usize;

    let bin_hz = sample_rate / audio_len as f32;

    let suppression_radius_bins = ((min_separation_hz / bin_hz).ceil() as usize).max(1);

    let spectrum = audio
        .to_kind(Kind::Float)
        .f_fft_rfft(None, -1, "backward")?;

    let magnitude = spectrum
        .f_abs()?
        .to_device(Device::Cpu)
        .to_kind(Kind::Float);

    let bins = magnitude.size()[1] as usize;

    let flat: Vec<f32> = Vec::<f32>::try_from(magnitude.f_reshape([-1])?)?;

    let mut result = Vec::with_capacity(channels);

    for ch in 0..channels {
        let offset = ch * bins;
        let m = &flat[offset..offset + bins];

        // Ignore DC bin 0.
        let strongest = m
            .iter()
            .skip(1)
            .copied()
            .filter(|x| x.is_finite())
            .fold(0.0_f32, f32::max);

        let min_abs_magnitude = strongest * min_relative_magnitude;

        let mut candidates: Vec<usize> = (1..bins).collect();

        candidates.sort_unstable_by(|&a, &b| {
            m[b].partial_cmp(&m[a]).unwrap_or(Ordering::Equal)
        });

        let mut chosen_bins: Vec<usize> = Vec::with_capacity(n_peaks);

        'candidate_loop: for bin in candidates {
            let mag = m[bin];

            if !mag.is_finite() {
                continue;
            }

            if mag < min_abs_magnitude {
                break;
            }

            // Keep only local maxima.
            //
            // This removes ordinary shoulders of a peak.
            let is_local_max = if bin + 1 < bins {
                m[bin] >= m[bin - 1] && m[bin] >= m[bin + 1]
            } else {
                m[bin] >= m[bin - 1]
            };

            if !is_local_max {
                continue;
            }

            // Non-maximum suppression:
            // if this bin is close to an already selected stronger bin,
            // treat it as the same frequency peak.
            for &selected in &chosen_bins {
                if bin.abs_diff(selected) <= suppression_radius_bins {
                    continue 'candidate_loop;
                }
            }

            chosen_bins.push(bin);

            if chosen_bins.len() >= n_peaks {
                break;
            }
        }

        let mut peaks = Vec::with_capacity(chosen_bins.len());

        for bin in chosen_bins {
            let interpolated_bin = parabolic_interpolated_bin(m, bin);
            let hz = interpolated_bin * bin_hz;

            peaks.push(FreqPeak {
                hz,
                magnitude: m[bin],
                bin: bin as i64,
            });
        }

        peaks.sort_by(|a, b| {
            b.magnitude
                .partial_cmp(&a.magnitude)
                .unwrap_or(Ordering::Equal)
        });

        result.push(peaks);
    }

    Ok(result)
}


#[derive(Debug, Clone, Copy)]
pub enum BandAggregation {
    Sum,
    Mean,
}

/// Converts `[channels, audio_data]` into `[channels, freq_bands]`.
///
/// Band ranges are:
///
/// ```text
/// 0..step_hz
/// step_hz..2*step_hz
/// 2*step_hz..3*step_hz
/// ...
/// up to sample_rate / 2
/// ```
///
/// Example:
///
/// ```text
/// sample_rate = 32_000
/// step_hz = 1_000
/// output bands = 16
/// band 0  = 0..1000 Hz
/// band 1  = 1000..2000 Hz
/// ...
/// band 15 = 15000..16000 Hz
/// ```
pub fn fft_band_amplitudes(
    audio: &Tensor,
    sample_rate: f64,
    step_hz: f64,
    aggregation: BandAggregation,
    apply_hann_window: bool,
    remove_dc_offset: bool,
) -> Result<Tensor, TchError> {
    let size = audio.size();
    assert_eq!(size.len(), 2, "expected [channels, audio_data]");

    let channels = size[0];
    let audio_len = size[1];

    assert!(audio_len > 1);
    assert!(sample_rate > 0.0);
    assert!(step_hz > 0.0);

    let device = audio.device();

    let mut audio = audio.to_kind(Kind::Float);

    if remove_dc_offset {
        audio = &audio - audio.f_mean_dim([1i64].as_slice(), true, Kind::Float)?;
    }

    if apply_hann_window {
        let window = Tensor::hann_window(audio_len, (Kind::Float, device));
        audio = audio * window;
    }

    // [channels, audio_data] -> [channels, rfft_bins]
    let spectrum = audio.f_fft_rfft(None, -1, "backward")?;

    // [channels, rfft_bins]
    let magnitude = spectrum.f_abs()?;

    // Optional normalization.
    //
    // This makes magnitudes less dependent on audio_len.
    // For exact amplitude reconstruction, DC/Nyquist need special handling,
    // but for band aggregation this is usually good enough.
    let magnitude = magnitude * (2.0 / audio_len as f64);

    let rfft_bins = magnitude.size()[1];

    let nyquist_hz = sample_rate / 2.0;
    let band_count = (nyquist_hz / step_hz).ceil() as i64;

    // Frequency of every rfft bin:
    //
    // bin_hz = bin_index * sample_rate / audio_len
    //
    // Shape: [rfft_bins]
    let bin_freqs = Tensor::arange(rfft_bins, (Kind::Float, device))
        * (sample_rate / audio_len as f64);

    // Map frequency -> band index.
    //
    // Example:
    // 0 Hz      -> band 0
    // 999 Hz    -> band 0
    // 1000 Hz   -> band 1
    // 15000 Hz  -> band 15, if step_hz = 1000
    //
    // Shape: [rfft_bins]
    let band_idx_1d = (bin_freqs / step_hz)
        .to_kind(Kind::Int64)
        .f_clamp_max(band_count - 1)?;

    // Expand to [channels, rfft_bins], because scatter_add needs same shape as src.
    let band_idx_2d = band_idx_1d
        .f_unsqueeze(0)?
        .f_expand([channels, rfft_bins], true)?;

    // Aggregate magnitudes into fixed frequency bands.
    //
    // [channels, band_count]
    let zeros = Tensor::zeros([channels, band_count], (Kind::Float, device));

    let summed = zeros.f_scatter_add(1, &band_idx_2d, &magnitude)?;

    match aggregation {
        BandAggregation::Sum => Ok(summed),

        BandAggregation::Mean => {
            // Count how many FFT bins landed in each frequency band.
            //
            // Shape: [band_count]
            let counts = Tensor::zeros([band_count], (Kind::Float, device))
                .f_scatter_add(
                    0,
                    &band_idx_1d,
                    &Tensor::ones([rfft_bins], (Kind::Float, device)),
                )?
                .f_clamp_min(1.0)?;

            Ok(summed / counts.f_unsqueeze(0)?)
        }
    }
}


/// Input:
///   audio: [channels, slices, audio_data]
///
/// Output:
///   [channels, slices, bands]
///
/// This is equivalent to applying the old `[channels, audio_data] -> [channels, bands]`
/// function independently to every slice.
pub fn fft_band_amplitudes_sliced(
    audio: &Tensor,
    sample_rate: f64,
    step_hz: f64,
    aggregation: BandAggregation,
    apply_hann_window: bool,
    remove_dc_offset: bool,
) -> Result<Tensor, ThoError> {
    let size = audio.size();
    assert_eq!(size.len(), 3, "expected [channels, slices, audio_data]");

    let channels = size[0];
    let slices = size[1];
    let audio_len = size[2];

    assert!(channels > 0);
    assert!(slices > 0);
    assert!(audio_len > 1);
    assert!(sample_rate > 0.0);
    assert!(step_hz > 0.0);

    let device = audio.device();

    let mut audio = audio.f_to_kind(Kind::Float)?;

    // Remove per-[channel, slice] DC offset.
    //
    // [channels, slices, audio_data]
    // mean over last dim -> [channels, slices, 1]
    if remove_dc_offset {
        let mean = audio.f_mean_dim([2i64].as_slice(), true, Kind::Float)?;
        audio = audio.f_sub(&mean)?;
    }

    // Apply Hann window over the last dimension.
    //
    // window: [audio_data]
    // view:   [1, 1, audio_data]
    // broadcast over [channels, slices, audio_data]
    if apply_hann_window {
        let window = Tensor::f_hann_window(audio_len, (Kind::Float, device))?;
        let window = window.f_view([1, 1, audio_len])?;
        audio = audio.f_mul(&window)?;
    }

    // FFT over last dimension:
    //
    // [channels, slices, audio_data]
    //   -> [channels, slices, rfft_bins]
    let spectrum = audio.f_fft_rfft(None, -1, "backward")?;

    // Complex magnitude:
    //
    // [channels, slices, rfft_bins]
    let magnitude = spectrum.f_abs()?;

    // Optional approximate amplitude normalization.
    //
    // For exact amplitude reconstruction, DC/Nyquist need special handling,
    // but for band features this is usually fine.
    let magnitude = magnitude.f_mul_scalar(2.0 / audio_len as f64)?;

    let rfft_bins = magnitude.size()[2];

    let nyquist_hz = sample_rate / 2.0;
    let band_count = (nyquist_hz / step_hz).ceil() as i64;

    assert!(band_count > 0);

    // Frequency of every rfft bin:
    //
    // bin_freq = bin_index * sample_rate / audio_len
    //
    // Shape: [rfft_bins]
    let bin_freqs = Tensor::f_arange(rfft_bins, (Kind::Float, device))?
        .f_mul_scalar(sample_rate / audio_len as f64)?;

    // Map each FFT bin to a fixed frequency band.
    //
    // Shape: [rfft_bins]
    let band_idx_1d = bin_freqs
        .f_div_scalar(step_hz)?
        .f_to_kind(Kind::Int64)?
        .f_clamp_max(band_count - 1)?; //Drop for Scalar->tch::error::TchError::unwrap

    // Expand band indices to match magnitude shape.
    //
    // [rfft_bins]
    //   -> [1, 1, rfft_bins]
    //   -> [channels, slices, rfft_bins]
    let band_idx_3d = band_idx_1d
        .f_view([1, 1, rfft_bins])?
        .f_expand([channels, slices, rfft_bins], true)?;

    // Aggregate into bands along dim=2.
    //
    // src:   [channels, slices, rfft_bins]
    // index: [channels, slices, rfft_bins]
    // out:   [channels, slices, band_count]
    let zeros = Tensor::f_zeros([channels, slices, band_count], (Kind::Float, device))?;

    let summed = zeros.f_scatter_add(2, &band_idx_3d, &magnitude)?;

    match aggregation {
        BandAggregation::Sum => Ok(summed),

        BandAggregation::Mean => {
            // Count how many FFT bins fall into each band.
            //
            // counts: [band_count]
            let ones = Tensor::f_ones([rfft_bins], (Kind::Float, device))?;

            let counts = Tensor::f_zeros([band_count], (Kind::Float, device))?
                .f_scatter_add(0, &band_idx_1d, &ones)?
                .f_clamp_min(1.0)?;

            // [band_count] -> [1, 1, band_count]
            let counts = counts.f_view([1, 1, band_count])?;

            Ok(summed.f_div(&counts)?)
        }
    }
}

/// Create random LSH hyperplanes.
///
/// `data_dim` = size of the last dimension of your input tensor.
/// `num_bits` = number of LSH bits.
/// Number of buckets will be `2^num_bits`.
///
/// planes shape: [data_dim, num_bits]
pub fn make_lsh_planes(
    data_dim: i64,
    num_bits: i64,
    kind: Kind,
    device: Device,
) -> Result<Tensor, ThoError> {
    if data_dim <= 0 {
        return Err(ThoError::new("data_dim must be > 0"));
    }
    if num_bits <= 0 || num_bits > 30 {
        return Err(ThoError::new("num_bits must be in 1..=30 to avoid huge bucket tensors"));
    }

    Ok(Tensor::f_randn(&[data_dim, num_bits], (kind, device))?)
}


fn ceil_log2_i64(x: i64) -> Result<i64, TchError> {
    if x <= 1 {
        return Err(TchError::Kind(
            "number of buckets must be > 1".into(),
        ));
    }

    let mut bits = 0_i64;
    let mut value = 1_i64;

    while value < x {
        bits += 1;

        if bits > 30 {
            return Err(TchError::Kind(
                "too many buckets requested; refusing num_bits > 30".into(),
            ));
        }

        value <<= 1;
    }

    Ok(bits)
}

#[derive(Debug)]
pub struct LshPlanes {
    pub planes: Tensor,
    pub num_bits: i64,
    pub raw_buckets: i64,
    pub wanted_buckets: i64,
}

/// Create random cosine-LSH hyperplanes from desired bucket count.
///
/// If `wanted_buckets` is not a power of two, `raw_buckets` will be rounded up
/// to the next power of two.
///
/// Example:
/// wanted_buckets = 1000
/// num_bits = 10
/// raw_buckets = 1024
pub fn make_lsh_planes_for_buckets(
    data_dim: i64,
    wanted_buckets: i64,
    kind: Kind,
    device: Device,
) -> Result<LshPlanes, TchError> {
    if data_dim <= 0 {
        return Err(TchError::Kind("data_dim must be > 0".into()));
    }

    if wanted_buckets <= 1 {
        return Err(TchError::Kind(
            "wanted_buckets must be > 1".into(),
        ));
    }

    let num_bits = ceil_log2_i64(wanted_buckets)?;
    let raw_buckets = 1_i64 << num_bits;

    let planes = Tensor::f_randn(&[data_dim, num_bits], (kind, device))?;

    Ok(LshPlanes {
        planes,
        num_bits,
        raw_buckets,
        wanted_buckets,
    })
}

/// Convert input `[M, N, data]` into `[M, N, buckets]` using LSH.
///
/// Each `[data]` vector is assigned to exactly one bucket.
/// Output is one-hot.
///
/// `x` shape:      [M, N, data]
/// `planes` shape: [data, num_bits]
/// output shape:   [M, N, 2^num_bits]
pub fn lsh_bucket_one_hot_3d(
    x: &Tensor,
    planes: &Tensor,
    normalize: bool,
) -> Result<Tensor, ThoError> {
    let x_size = x.size();
    if x_size.len() != 3 {
        return Err(ThoError::new(format!(
            "expected x shape [M, N, data], got {:?}",
            x_size
        )));
    }

    let p_size = planes.size();
    if p_size.len() != 2 {
        return Err(ThoError::new(format!(
            "expected planes shape [data, num_bits], got {:?}",
            p_size
        )));
    }

    let m = x_size[0];
    let n = x_size[1];
    let data_dim = x_size[2];

    let planes_data_dim = p_size[0];
    let num_bits = p_size[1];

    if data_dim != planes_data_dim {
        return Err(ThoError::new(format!("x data dim {} does not match planes data dim {}", data_dim, planes_data_dim)));
    }

    if num_bits <= 0 || num_bits > 30 {
        return Err(ThoError::new("num_bits must be in 1..=30 to avoid huge bucket tensors"));
    }

    let num_buckets = 1_i64 << num_bits;

    // [M, N, data] -> [M * N, data]
    let x_flat = x.f_view([m * n, data_dim])?;

    // Optional cosine normalization.
    let x_flat = if normalize {
        let eps = 1e-12;

        // norm = sqrt(sum(x^2, dim=-1, keepdim=true))
        let norm = x_flat
            .f_pow_tensor_scalar(2.0)?
            .f_sum_dim_intlist(&[-1i64][..], true, x.kind())?
            .f_clamp_min(eps)?
            .f_sqrt()?;

        x_flat.f_div(&norm)?
    } else {
        x_flat
    };

    // [M * N, data] @ [data, num_bits] -> [M * N, num_bits]
    let projections = x_flat.f_matmul(planes)?;

    // Sign bits: [M * N, num_bits], values 0/1.
    let bits = projections
        .f_ge(0.0)?
        .f_to_kind(Kind::Int64)?;

    // Convert bit-vector to integer bucket id.
    //
    // Example for num_bits = 4:
    // bits [b0, b1, b2, b3]
    // id = b0*1 + b1*2 + b2*4 + b3*8
    let weights_vec: Vec<i64> = (0..num_bits).map(|i| 1_i64 << i).collect();

    let weights = Tensor::f_from_slice(&weights_vec)?
        .f_to_device(x.device())?
        .f_view([1, num_bits])?;

    // [M * N]
    let bucket_ids = bits
        .f_mul(&weights)?
        .f_sum_dim_intlist(&[-1i64][..], false, Kind::Int64)?;

    // [M * N, buckets]
    let one_hot = bucket_ids
        .f_one_hot(num_buckets)?
        .f_to_kind(x.kind())?;

    // [M, N, buckets]
    Ok(one_hot.f_view([m, n, num_buckets])?)
}


pub fn lsh_bucket_one_hot_3d_exact_buckets(
    x: &Tensor,
    lsh: &LshPlanes,
    normalize: bool,
) -> Result<Tensor, TchError> {
    let x_size = x.size();
    if x_size.len() != 3 {
        return Err(TchError::Kind(format!(
            "expected x shape [M, N, data], got {:?}",
            x_size
        )));
    }

    let p_size = lsh.planes.size();
    if p_size.len() != 2 {
        return Err(TchError::Kind(format!(
            "expected planes shape [data, num_bits], got {:?}",
            p_size
        )));
    }

    let m = x_size[0];
    let n = x_size[1];
    let data_dim = x_size[2];

    let planes_data_dim = p_size[0];
    let num_bits = p_size[1];

    if data_dim != planes_data_dim {
        return Err(TchError::Kind(format!(
            "x data dim {} does not match planes data dim {}",
            data_dim, planes_data_dim
        )));
    }

    if num_bits != lsh.num_bits {
        return Err(TchError::Kind(format!(
            "planes num_bits {} does not match lsh.num_bits {}",
            num_bits, lsh.num_bits
        )));
    }

    let x_flat = x.f_view([m * n, data_dim])?;

    let x_flat = if normalize {
        let eps = 1e-12;

        let norm = x_flat
            .f_pow_tensor_scalar(2.0)?
            .f_sum_dim_intlist(&[-1i64][..], true, x.kind())?
            .f_clamp_min(eps)?
            .f_sqrt()?;

        x_flat.f_div(&norm)?
    } else {
        x_flat
    };

    // [M * N, data] @ [data, num_bits] -> [M * N, num_bits]
    let projections = x_flat.f_matmul(&lsh.planes)?;

    // [M * N, num_bits], values 0/1
    let bits = projections
        .f_ge(0.0)?
        .f_to_kind(Kind::Int64)?;

    let weights_vec: Vec<i64> = (0..num_bits).map(|i| 1_i64 << i).collect();

    let weights = Tensor::f_from_slice(&weights_vec)?
        .f_to_device(x.device())?
        .f_view([1, num_bits])?;

    // Raw bucket ids in range [0, raw_buckets).
    let raw_bucket_ids = bits
        .f_mul(&weights)?
        .f_sum_dim_intlist(&[-1i64][..], false, Kind::Int64)?;

    // Compress raw 2^bits buckets into exactly wanted_buckets.
    //
    // If wanted_buckets == raw_buckets, this changes nothing.
    let bucket_ids = raw_bucket_ids.f_fmod(lsh.wanted_buckets)?;

    // [M * N, wanted_buckets]
    let one_hot = bucket_ids
        .f_one_hot(lsh.wanted_buckets)?
        .f_to_kind(x.kind())?;

    // [M, N, wanted_buckets]
    one_hot.f_view([m, n, lsh.wanted_buckets])
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use log::debug;
    use tch::kind::FLOAT_CPU;
    use tch::{Device, IndexOp, Kind, Tensor};
    use thoosa_shared::audio_utils::compose_freqs;
    use thoosa_shared::test_utils::{gen_motif, MotifKind};
    use thoosa_shared::{sin_v_f, ThoError};
    use crate::audio::{assign_lsh_buckets_per_channel, fft_band_amplitudes, fft_band_amplitudes_sliced, lsh_bucket_one_hot_3d, lsh_bucket_one_hot_3d_exact_buckets, make_lsh_planes, make_lsh_planes_for_buckets, strongest_freqs_per_channel_windowed, strongest_freqs_per_channel_windowed_nms, BandAggregation};
    use crate::debug_plot;
    use crate::utils::{debug_plot_impl, debug_plot_impl_wh, plot_spectrogram, reset_print_options};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_audio() -> Result<(), ThoError>{
        reset_print_options();
        let sample_rate = 32000.;
        // plot_t(&resource_t, &format!("resource_{i}"))?;
        // debug_plot_impl(&Tensor::rand([1, 1000], FLOAT_CPU), "test", 0)?;
        // let tensor = Tensor::load("/home/bu/tmp/ws_clef/tensors/1_hyamac1_XC497685.ogg.pt")?;

        let tensor = Tensor::cat(&[
            Tensor::from_slice(compose_freqs(4000, sample_rate as usize, vec![2500.0]).as_slice()), //300.0,
            Tensor::from_slice(compose_freqs(4000, sample_rate as usize, vec![5500.0]).as_slice()), //600.0,
            Tensor::from_slice(compose_freqs(4000, sample_rate as usize, vec![10500.0]).as_slice()) //1000.0,
        ], 0).view((1, -1));


        let channels_slices_freqamp = fft_band_amplitudes_sliced(
            &tensor.unfold(
                1,
                3000, //should be taken from RunConfig
                3000
            ), // [channels, audio_data]
            32_000.0,             // sample rate - from workspace
            1_000.0,              // frequency band width - from RunConfig
            BandAggregation::Sum, // or Mean - from RunConfig
            false,    // Hann window
            true,      // remove DC offset - from RunConfig
        )?;

        debug!("bands size = {:?} channels_slices_freqamp=\n{channels_slices_freqamp}", channels_slices_freqamp.size());
        let num_bits = 4; //should be taken from RunConfig
        let planes = make_lsh_planes_for_buckets(channels_slices_freqamp.size()[2], 10, Kind::Float, Device::Cpu)?;

        let buckets = lsh_bucket_one_hot_3d_exact_buckets(&channels_slices_freqamp, &planes, true)?;

        debug!("buckets=\n{buckets}");
        assert_eq!(channels_slices_freqamp.size(), vec![1, 4, 16]);
        assert_eq!(buckets.size(), vec![1, 4, 10]);
        let bucket_counts = buckets.sum_dim_intlist(&[2i64][..], false, Kind::Float);
        assert_eq!(bucket_counts, Tensor::ones([1, 4], FLOAT_CPU));

        Ok(())

    }

}
