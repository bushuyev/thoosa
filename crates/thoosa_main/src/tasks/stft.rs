use cached_derive::Cached;
use log::debug;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use tch::Tensor;
use thoosa_common::dims::{Batch, MotifSlices, Spectorgram};
use thoosa_common::runner::common::cachable::{DevicesWrap, WithDataRef};
use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::runner::common::{OutId, RunCtx};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::traits::{Data, WithId};
use thoosa_shared::{mono_key, ThoError};

use crate::tasks::stage_ids::STFT;

#[derive(Clone, PartialEq, Debug)]
pub struct StftRunConfig {
    pub n_fft: i64,
    pub hop_length: Option<i64>,
    pub win_length: Option<i64>,
    pub normalized: bool,
    pub onesided: bool,
    pub align_to_window: bool,
}

impl StftRunConfig {
    pub fn new(n_fft: i64, hop_length: Option<i64>, win_length: Option<i64>, normalized: bool, onesided: bool, align_to_window: bool) -> Self {
        Self { n_fft, hop_length, win_length, normalized, onesided, align_to_window }
    }

    fn validate(&self) -> Result<(), ThoError> {
        if self.n_fft <= 0 {
            return Err(ThoError::new(format!("StftRunConfig::validate: n_fft must be positive, got {}", self.n_fft)));
        }
        if matches!(self.hop_length, Some(v) if v <= 0) {
            return Err(ThoError::new(format!("StftRunConfig::validate: hop_length must be positive, got {:?}", self.hop_length)));
        }
        if matches!(self.win_length, Some(v) if v <= 0) {
            return Err(ThoError::new(format!("StftRunConfig::validate: win_length must be positive, got {:?}", self.win_length)));
        }
        Ok(())
    }
}

impl Default for StftRunConfig {
    fn default() -> Self {
        Self { n_fft: 16000, hop_length: Some(4000), win_length: Some(16000), normalized: false, onesided: true, align_to_window: true }
    }
}

#[derive(Debug, cached_derive::Cached, Named)]
#[cache_name = "Stft"]
pub struct Stft<C>
where
    C: RunCtx + 'static,
{
    id: SynKey,
    run_config: StftRunConfig,
    pd: PhantomData<C>,
}

impl<C> Stft<C>
where
    C: RunCtx + 'static,
{
    pub fn new(run_config: StftRunConfig) -> Result<Self, ThoError> {
        run_config.validate()?;
        Ok(Self { id: stft_stage_id(&run_config), run_config, pd: PhantomData })
    }

    pub fn run_config(&self) -> &StftRunConfig {
        &self.run_config
    }
}

impl<C> WithId for Stft<C>
where
    C: RunCtx + 'static,
{
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<C> OutId for Stft<C>
where
    C: RunCtx + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<C> Stage for Stft<C>
where
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = ThoTensor<MotifSlices, SynKey>;
    type Out = ThoTensor<Spectorgram, SynKey>;
}

impl<C> AsyncStage for Stft<C>
where
    C: RunCtx + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "Stft::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let device_idx = ctx.device_idx();

        let out_id = self.out_id(input.id());
        let output = input.with_data_ref(device_idx, ctx, Some("stft"), |input, _ctx| {
            let spectrogram = build_spectrogram(input.t_ref(), &self.run_config)?;
            debug!("Stft::run: input_id={:?} input_size={:?} output_size={:?} run_config={:?}", input.id(), input.size(), spectrogram.size(), self.run_config);
            ThoTensor::new(out_id, spectrogram)
        }).await?;
        Ok(Arc::new(DevicesWrap::wrap(output)))
    }
}

pub fn build_spectrogram(input: &Tensor, run_config: &StftRunConfig) -> Result<Tensor, ThoError> {
    run_config.validate()?;
    let input_size = input.size();
    if input_size.len() != 1 && input_size.len() != 2 {
        return Err(ThoError::new(format!("build_spectrogram: expected [window] or [slices, window], got {:?}", input_size)));
    }

    let spec = input.f_stft(run_config.n_fft, run_config.hop_length, run_config.win_length, None::<Tensor>, run_config.normalized, run_config.onesided, true, run_config.align_to_window)?;
    let spec_abs = spec.f_abs()?;
    let spec_size = spec_abs.size();

    match spec_size.as_slice() {
        [_freq_bins, _frames] => Ok(spec_abs.f_transpose(0, 1)?.f_contiguous()?),
        [slices, freq_bins, frames] => Ok(spec_abs.f_transpose(1, 2)?.f_contiguous()?.f_view([slices * frames, *freq_bins])?),
        _ => Err(ThoError::new(format!("build_spectrogram: unexpected stft output size {:?} for input size {:?}", spec_size, input_size))),
    }
}

fn stft_stage_id(run_config: &StftRunConfig) -> SynKey {
    SynKey::new([
        mono_key!(STFT, "Stft"),
        mono_key!(run_config.n_fft.to_string(), "n_fft"),
        mono_key!(format!("{:?}", run_config.hop_length), "hop_length"),
        mono_key!(format!("{:?}", run_config.win_length), "win_length"),
        mono_key!(run_config.normalized, "normalized"),
        mono_key!(run_config.onesided, "onesided"),
        mono_key!(run_config.align_to_window, "align_to_window"),
    ])
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};
    use thoosa_shared::ThoError;

    use crate::tasks::stft::{build_spectrogram, StftRunConfig};

    #[test]
    fn test_build_spectrogram_flattens_batched_motif_slices() -> Result<(), ThoError> {
        let run_config = StftRunConfig::new(4, Some(2), Some(4), false, true, true);
        let input = Tensor::from_slice(&[0.0f32, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]).f_view([2, 8])?;

        let output = build_spectrogram(&input, &run_config)?;

        assert_eq!(output.size(), vec![6, 3]);
        assert_eq!(output.kind(), Kind::Float);
        assert_eq!(output.device(), Device::Cpu);
        Ok(())
    }
}
