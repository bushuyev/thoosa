use log::debug;
use serde::Deserialize;
use tch::{Device, Kind, TchError, Tensor};
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::{panic_to_err, ThoError};

pub const LOG_EPS:f64 = 1e-6;

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
pub struct LogMelPatchConfig {

    /// FFT/window size in samples, e.g. 1024.
    #[run_config]
    #[in_map]
    pub time_size: i64,

    /// STFT hop in samples, e.g. 320 for 20 ms at 16 kHz.
    #[run_config]
    #[in_map]
    pub time_step: i64,

    /// Number of mel bands, e.g. 64 or 128.
    #[run_config]
    #[in_map]
    pub n_mels: i64,

    /// Lowest mel frequency.
    #[run_config]
    #[in_map]
    pub f_min: f64,

    /// Highest mel frequency. If <= 0.0, sample_rate / 2.0 is used.
    #[run_config]
    #[in_map]
    pub f_max: f64,

    /// Number of STFT frames per output patch.
    #[run_config]
    #[in_map]
    pub mel_patch_size: i64,

    /// Step between patches in STFT frames.
    #[run_config]
    #[in_map]
    pub mel_patch_step: i64,


}

impl Default for LogMelPatchConfig {
    fn default() -> Self {
        Self {
            time_size: 1024,
            time_step: 320,
            n_mels: 128,
            f_min: 20.0,
            f_max: 16_000.0,
            mel_patch_size: 64,
            mel_patch_step: 32,
        }
    }
}

/// Convert audio `[channels, samples]` into log-mel patches:
///
/// Output shape:
///
/// `[channels, patches, n_mels, patch_frames]`
pub fn audio_to_log_mel_patches(
    audio: &Tensor,
    cfg: &LogMelPatchConfig,
    sample_rate: f64,
) -> Result<Tensor, ThoError> {
    debug!("audio_to_log_mel_patches: audio={audio:?} cfg={cfg:?}");

    let size = audio.size();

    if size.len() != 2 {
        return Err(ThoError::new(format!("expected audio shape [channels, samples], got {size:?}")));
    }

    let channels = size[0];
    let samples = size[1];

    if samples < cfg.time_size {
        return Err(ThoError::new(format!("audio too short: samples={samples}, n_fft={}", cfg.time_size)));
    }

    if cfg.time_size <= 0 || cfg.time_step <= 0 || cfg.n_mels <= 0 {
        return Err(ThoError::new(format!("invalid config: n_fft={}, hop_length={}, n_mels={}", cfg.time_size, cfg.time_step, cfg.n_mels)));
    }

    if cfg.mel_patch_size <= 0 || cfg.mel_patch_step <= 0 {
        return Err(ThoError::new(format!("invalid patch config: patch_frames={}, patch_step={}", cfg.mel_patch_size, cfg.mel_patch_step)));
    }

    let device = audio.device();

    // Work in float.
    let audio = audio.f_to_kind(Kind::Float)?;

    // ---------------------------------------------------------------------
    // 1. Frame audio with unfold.
    //
    // audio:  [C, samples]
    // frames: [C, frames, n_fft]
    // ---------------------------------------------------------------------
    let frames = audio.f_unfold(1, cfg.time_size, cfg.time_step)?;
    debug!("audio_to_log_mel_patches: frames={frames:?}");
    // ---------------------------------------------------------------------
    // 2. Apply Hann window.
    //
    // window: [n_fft]
    // frames: [C, frames, n_fft]
    // ---------------------------------------------------------------------
    let hann = Tensor::f_hann_window(cfg.time_size, (Kind::Float, device))?;
    debug!("audio_to_log_mel_patches: hann={hann:?}");

    let frames = frames.f_mul(&hann.f_view([1, 1, cfg.time_size])?)?;
    debug!("audio_to_log_mel_patches: frames x hann={frames:?}");

    // ---------------------------------------------------------------------
    // 3. RFFT over the last dimension.
    //
    // spectrum: [C, frames, freq_bins] complex
    // power:    [C, frames, freq_bins] real
    // ---------------------------------------------------------------------
    let spectrum = frames.f_fft_rfft(None, -1, "backward")?;
    debug!("audio_to_log_mel_patches: spectrum={spectrum:?}");

    let power = spectrum.f_abs()?.f_pow_tensor_scalar(2.0)?;

    debug!("audio_to_log_mel_patches: power={power:?}");

    let freq_bins = cfg.time_size / 2 + 1;

    // ---------------------------------------------------------------------
    // 4. Mel projection.
    //
    // power:        [C, frames, F]
    // mel_fb:       [M, F]
    // power_flat:   [C * frames, F]
    // mel_flat:     [C * frames, M]
    // mel:          [C, M, frames]
    // ---------------------------------------------------------------------
    //called once per RunConfig
    let mel_fb = build_mel_filterbank(
        sample_rate,
        cfg.time_size,
        cfg.n_mels,
        cfg.f_min,
        cfg.f_max,
        device,
    )?;

    debug!("audio_to_log_mel_patches: mel_fb=\n{mel_fb}");

    let n_frames = power.size()[1];

    let power_flat = power.f_view([channels * n_frames, freq_bins])?;

    debug!("audio_to_log_mel_patches: power_flat={power_flat:?}");

    let mel_flat = power_flat.f_matmul(&mel_fb.f_transpose(0, 1)?)?;

    debug!("audio_to_log_mel_patches: mel_flat={mel_flat:?}");

    let mel = mel_flat
        .f_view([channels, n_frames, cfg.n_mels])?
        .f_transpose(1, 2)?; // [C, M, frames]

    debug!("audio_to_log_mel_patches: mel={mel:?}");
    // ---------------------------------------------------------------------
    // 5. Log compression.
    //
    // log_mel: [C, M, frames]
    // ---------------------------------------------------------------------
    let log_mel = mel.f_clamp_min(LOG_EPS)?.f_log()?;

    // ---------------------------------------------------------------------
    // 6. Slice time axis into patches.
    //
    // log_mel: [C, M, frames]
    // patches: [C, M, patches, patch_frames]
    // output:  [C, patches, M, patch_frames]
    // ---------------------------------------------------------------------
    if log_mel.size()[2] < cfg.mel_patch_size {
        return Err(ThoError::new(format!("not enough STFT frames for one patch: frames={}, patch_frames={}", log_mel.size()[2], cfg.mel_patch_size)));
    }

    let patches = log_mel.f_unfold(2, cfg.mel_patch_size, cfg.mel_patch_step)?;
    let patches = patches.f_permute(&[0, 2, 1, 3])?;

    Ok(patches)
}

fn build_mel_filterbank(
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

/// For patches shaped [channels, patches, n_mels, patch_frames],
/// returns [channels x patches, (sample_start, sample_end)].
pub fn patch_sample_ranges_flat(
    patches: &Tensor,
    cfg: &LogMelPatchConfig,
) -> Result<Vec<Vec<(usize, usize)>>, TchError> {
    let size = patches.size();

    if size.len() != 4 {
        return Err(TchError::Shape(format!(
            "expected [channels, patches, n_mels, patch_frames], got {size:?}"
        )));
    }

    let channels = size[0];
    let n_patches = size[1];

    // let mut values = Vec::<i64>::with_capacity((channels * n_patches * 4) as usize);
    let mut values = vec![vec![]; channels as usize];

    for c in 0..channels {
        for p in 0..n_patches {
            let first_frame = p * cfg.mel_patch_step;
            let last_frame = first_frame + cfg.mel_patch_size - 1;

            let sample_start = first_frame * cfg.time_step;
            let sample_end = last_frame * cfg.time_step + cfg.time_size;


            values[c as usize].push((sample_start as usize, sample_end as usize))
        }
    }

    Ok(values)
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use cudarc::driver::HostSlice;
    use log::debug;
    use tch::kind::FLOAT_CPU;
    use tch::{Device, IndexOp, Kind, Tensor};
    use thoosa_shared::audio_utils::compose_freqs;
    use thoosa_shared::test_utils::{gen_motif, MotifKind};
    use thoosa_shared::ThoError;
    use crate::audio::{assign_lsh_buckets_per_channel, audio_to_log_mel_patches, patch_sample_ranges_flat, LogMelPatchConfig};
    use crate::debug_plot;
    use crate::utils::{debug_plot_impl, debug_plot_impl_wh};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_audio_to_log_mel_patches() -> Result<(), ThoError>{
        let sample_rate = 32000.;
        // plot_t(&resource_t, &format!("resource_{i}"))?;
        // debug_plot_impl(&Tensor::rand([1, 1000], FLOAT_CPU), "test", 0)?;
        // let tensor = Tensor::load("/home/bu/tmp/ws_clef/tensors/1_hyamac1_XC497685.ogg.pt")?;
        // let tensor = Tensor::from_slice(gen_motif((sample_rate * 3.) as usize, &MotifKind::HarmonicStack).as_slice()).view((1, -1));
        let n = (sample_rate * 1.) as usize;
        let tensor = Tensor::cat(&[
            Tensor::from_slice(compose_freqs(1234, sample_rate as usize, vec![300.0,  3000.0]).as_slice()),
            Tensor::from_slice(compose_freqs(5678, sample_rate as usize, vec![600.0,  6000.0]).as_slice()),
            Tensor::from_slice(compose_freqs(9876, sample_rate as usize, vec![1200.0, 12000.0]).as_slice())
        ], 0).view((1, -1));
        let freqs = tensor.f_fft_rfft(Some(32000), -1, "backward")?;
        debug!("tensor={tensor:?} freqs=\n{freqs:?}");
        debug_plot_impl_wh(&(freqs.abs() *  (2.0 / n as f64)), "freqs", 0, (2000, 1000))?; //((freqs.size()[1] as u32).min(40_000), 1000))?;
        debug_plot_impl_wh(&tensor, "test", 0, (2000, 1000))?;

        return Ok(());
        let cfg = LogMelPatchConfig {
            time_size: 32000,
            time_step: 800,
            n_mels: 4,
            mel_patch_size: 64,
            mel_patch_step: 32,
            ..Default::default()
        };

        // Example: mono audio, 10 seconds at 32 kHz.


        let patches = audio_to_log_mel_patches(&tensor, &cfg, sample_rate)?;


        debug!("patches shape = {:?}", patches.size());
        let patches_indexes = patch_sample_ranges_flat(&patches, &cfg)?;
        debug!("patches indexes len={} patches_indexes{patches_indexes:?}", patches_indexes[0].len());


        let bucket_ids = assign_lsh_buckets_per_channel(&patches, 256)?;

        debug!("bucket_ids.len={} bucket_ids={bucket_ids:?}", bucket_ids[0].len());

        for channel in 0..bucket_ids.len() {
            let mut motifs = HashMap::<usize, Vec<(usize, usize)>>::new();

            for (patch, bucket) in patches_indexes[channel].iter().zip(bucket_ids[channel].iter()) {
                let bucket_slices = motifs.entry(*bucket).or_insert(vec![]);
                bucket_slices.push(*patch);
            }

            debug!("motifs.len={}", motifs.len());

            for (id, motif_slices) in motifs {
                debug!("id={id} motif_slices={motif_slices:?}");
                let num_slices = motif_slices.len();
                let motif_t = Tensor::stack( motif_slices.into_iter().map(|(from, to)|tensor.i((channel as i64, from as i64..to as i64))).collect::<Vec<_>>().as_slice(), 0);
                debug!("motif_t={motif_t:?}");
                // debug_plot_impl_wh(&motif_t, &format!("bucket_{id}, {} slices", num_slices), 0, ((tensor.size()[1] as u32).min(40_000), 2000))?;
            }
        }

        Ok(())
    }
}