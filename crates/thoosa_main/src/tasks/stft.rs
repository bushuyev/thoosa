use cached_derive::Cached;
use log::debug;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use tch::Tensor;
use thoosa_common::dims::{Batch, MotifSlices, Spectorgram};
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep, MAX_PARALLEL};
use thoosa_common::runner::common::cachable::DevicesWrap;
use thoosa_common::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_common::runner::common::{OutId, RunCtx};
use thoosa_common::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::traits::{Adjustable, Data, Select, WithId};
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
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    run_config: StftRunConfig,
    sync_stft: Arc<SyncStft<C>>,
    pd: PhantomData<C>,
}

impl<C> Stft<C>
where
    C: RunCtx + Adjustable + 'static,
{
    pub fn new(run_config: StftRunConfig) -> Result<Self, ThoError> {
        run_config.validate()?;
        let id = stft_stage_id(&run_config);
        Ok(Self { id: id.clone(), sync_stft: Arc::new(SyncStft::new(id, run_config.clone())), run_config, pd: PhantomData })
    }

    pub fn run_config(&self) -> &StftRunConfig {
        &self.run_config
    }
}

impl<C> WithId for Stft<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<C> OutId for Stft<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<C> Stage for Stft<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = ThoTensor<Batch<MotifSlices>, SynKey>;
    type Out = ThoTensor<Spectorgram, SynKey>;
}

impl<C> AsyncStage for Stft<C>
where
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "Stft::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let required_ids: std::collections::HashMap<&'static str, &std::collections::HashSet<&dyn thoosa_shared::traits::CachableIdTr>> = std::collections::HashMap::new();
        let oom_handler: OOMHandler<'_, '_, _, _, _, _, SyncStftSelector> = OOMHandler::new(
            vec![
                Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::FIFTY))),
                Adjusters::RunContext(Box::new(DumpToCpu::new("sync_stft"))),
            ],
            None,
        );
        let output = oom_handler
            .run(
                &self.sync_stft,
                input,
                ctx.branch(Some("sync_stft"), false),
                Contexts::new(
                    NoopInputContext(None),
                    NoopRunnableContext {},
                    MaxParallelAndDumpToCpuAdjustableContex::new(
                        MaxParallelAdjustContextImpl::new(),
                        DumpToCpuContextImpl::new(&required_ids),
                    ),
                ),
            )
            .await?;
        Ok(Arc::new(DevicesWrap::wrap(output)))
    }
}

#[derive(Clone, Debug, Default)]
struct SyncStftSelector;

impl Adjustable for SyncStftSelector {}

impl Select<SyncStftSelector> for ThoTensor<Batch<MotifSlices>, SynKey> {
    type Output = ThoTensor<Batch<MotifSlices>, SynKey>;

    fn subview(&self, _selector: SyncStftSelector) -> Result<Self::Output, ThoError> {
        Err(ThoError::new("SyncStftSelector::subview: input selection is not supported"))
    }
}

#[derive(Debug, Named)]
struct SyncStft<C>
where
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    run_config: StftRunConfig,
    pd: PhantomData<C>,
}

impl<C> SyncStft<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn new(id: SynKey, run_config: StftRunConfig) -> Self {
        Self { id, run_config, pd: PhantomData }
    }

    fn build_spectrogram(&self, input: &Tensor) -> Result<Tensor, ThoError> {
        self.run_config.validate()?;
        let input_size = input.size();
        if input_size.len() != 2 && input_size.len() != 3 {
            return Err(ThoError::new(format!("SyncStft::build_spectrogram: expected [batch, window] or [batch, slices, window], got {:?}", input_size)));
        }

        let motifs = input.unbind(0);
        let mut specs = Vec::with_capacity(motifs.len());
        for (i, motif) in motifs.iter().enumerate() {
            let spec = match motif.size().len() {
                1 => motif
                    .f_unsqueeze(0)?
                    .f_stft(
                        self.run_config.n_fft,
                        self.run_config.hop_length,
                        self.run_config.win_length,
                        None::<Tensor>,
                        self.run_config.normalized,
                        self.run_config.onesided,
                        true,
                        self.run_config.align_to_window,
                    )?
                    .f_squeeze_dim(0)?
                    .f_abs()?,
                2 => motif
                    .f_stft(
                        self.run_config.n_fft,
                        self.run_config.hop_length,
                        self.run_config.win_length,
                        None::<Tensor>,
                        self.run_config.normalized,
                        self.run_config.onesided,
                        true,
                        self.run_config.align_to_window,
                    )?
                    .f_abs()?,
                _ => return Err(ThoError::new(format!("SyncStft::build_spectrogram: unexpected motif size {:?}", motif.size()))),
            };
            debug!("SyncStft::build_spectrogram: motif_idx={i} motif_size={:?} spec_size={:?}", motif.size(), spec.size());
            specs.push(spec);
        }

        Ok(Tensor::f_stack(&specs, 0)?)
    }
}

impl<C> Adjustable for SyncStft<C> where C: RunCtx + Adjustable + 'static {}

impl<C> OutId for SyncStft<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<C> Stage for SyncStft<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = ThoTensor<Batch<MotifSlices>, SynKey>;
    type Out = ThoTensor<Spectorgram, SynKey>;
}

impl<C> SyncStage for SyncStft<C>
where
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "SyncStft::do_run"))]
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let spectrogram = self.build_spectrogram(input.t_ref())?;
        debug!("SyncStft::do_run: input_id={:?} input_size={:?} output_size={:?} run_config={:?}", input.id(), input.size(), spectrogram.size(), self.run_config);
        ThoTensor::new(self.out_id(input.id()), spectrogram)
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
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use thoosa_shared::ThoError;

    use crate::tasks::stft::{stft_stage_id, StftRunConfig, SyncStft};

    #[test]
    fn test_sync_stft_build_spectrogram_stacks_batched_motifs() -> Result<(), ThoError> {
        let run_config = StftRunConfig::new(4, Some(2), Some(4), false, true, true);
        let input = Tensor::from_slice(&[0.0f32, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]).f_view([2, 8])?;
        let stage = SyncStft::<RunCtxImpl>::new(stft_stage_id(&run_config), run_config);

        let output = stage.build_spectrogram(&input)?;

        assert_eq!(output.size(), vec![2, 3, 3]);
        assert_eq!(output.kind(), Kind::Float);
        assert_eq!(output.device(), Device::Cpu);
        Ok(())
    }
}
