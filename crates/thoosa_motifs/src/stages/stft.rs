use thoosa_common::tch_error::tch_result;
use crate::dims::{MotifSlices, Spectorgram};
use log::debug;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use tch::Kind::Float;
use tch::Tensor;
use thoosa_common::dims::{Batch, Dims, LocalSelect};
use thoosa_common::nets::common_nets::StftRunConfig;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep};
use thoosa_common::runner::common::cachable::DevicesWrap;
use thoosa_common::runner::oom_handler::{Adjusters, NoInputAdjuster, NoStageAdjuster, OOMHandler, WithRunCtxContext, run_ctx_adjuster};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_named_derive::Named;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::traits::{Adjustable, AsyncStage, MaxParallelCtx, OutId, RunCtx, Stage, ThoSelector, WithId};
use thoosa_common::runner::sync_stage::{SyncStage};
use thoosa_shared::{ThoError, mono_key};

use crate::stages::stage_ids::STFT;
use thoosa_shared::tracked_await::track_await;

#[derive(Debug, cached_derive::Cached, Named)]
#[cache_name = "Stft"]
pub struct Stft<C, InD = Batch<MotifSlices>, OutD = Spectorgram>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    InD: Dims + 'static,
    OutD: Dims + 'static,
{
    id: SynKey,
    run_config: StftRunConfig,
    sync_stft: Arc<SyncStft<C, InD, OutD>>,
    pd: PhantomData<(C, InD, OutD)>,
}

impl<C> Stft<C>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    pub fn new(run_config: StftRunConfig) -> Result<Self, ThoError> {
        Self::new_for_dims(run_config)
    }
}

impl<C, InD, OutD> Stft<C, InD, OutD>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    InD: Dims + 'static,
    OutD: Dims + 'static,
{
    pub fn new_for_dims(run_config: StftRunConfig) -> Result<Self, ThoError> {
        run_config.validate()?;
        let id = stft_stage_id(&run_config);
        Ok(Self { id: id.clone(), sync_stft: Arc::new(SyncStft::new(id, run_config.clone())?), run_config, pd: PhantomData })
    }

    pub fn run_config(&self) -> &StftRunConfig {
        &self.run_config
    }
}

impl<C, InD, OutD> OutId for Stft<C, InD, OutD>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    InD: Dims + 'static,
    OutD: Dims + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<C, InD, OutD> Stage for Stft<C, InD, OutD>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    InD: Dims + 'static,
    OutD: Dims + 'static,
{
    type Ctx = C;
    type In<'a> = Arc<DevicesWrap<ThoTensor<InD, SynKey>>>;
    type Out = Arc<DevicesWrap<ThoTensor<OutD, SynKey>>>;
}

impl<C, InD, OutD> AsyncStage for Stft<C, InD, OutD>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    InD: Dims + 'static,
    OutD: Dims + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "Stft::run"))]
    async fn run(self: Arc<Self>, input: Self::In<'static>, mut ctx: Self::Ctx) -> Result<Self::Out, ThoError> {
        // The input can come from different loaders; follow its registered owners.

        let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoStageAdjuster, _, SyncStftSelector> = OOMHandler::new(
            vec![
                Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(*MaxParallelStep::FAST))),
                Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("sync_stft"))),
            ],
            None,
        );
        let output = track_await!(oom_handler
            .run(
                &self.sync_stft,
                Arc::clone(&input),
                ctx.branch(false),
                WithRunCtxContext::new(
                    MaxParallelAndDumpToCpuAdjustableContex::new(
                        MaxParallelAdjustContextImpl::new(),
                        DumpToCpuContextImpl::new(),
                    ),
                ),
            )).await?;
        Ok(Arc::new(DevicesWrap::wrap(output)))
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SyncStftSelector;

#[typetag::serde]
impl ThoSelector for SyncStftSelector {
    fn clone_box(&self) -> Box<dyn ThoSelector> {
        Box::new(self.clone())
    }

    fn tensor_ranges(&self) -> Result<Vec<std::ops::Range<i64>>, ThoError> {
        Ok(vec![])
    }
}

impl Adjustable for SyncStftSelector {}

impl<D> LocalSelect<SynKey, SyncStftSelector> for ThoTensor<D, SynKey>
where
    D: Dims,
{
    type Output = ThoTensor<D, SynKey>;

    fn local_subview(id:&SynKey, t:&Tensor, _selector: SyncStftSelector) -> Result<Self::Output, ThoError> {
        // Err(ThoError::new("SyncStftSelector::subview: input selection is not supported"))
        ThoTensor::new(id.clone(), t.copy())//TODO
    }
}

#[derive(Debug, Named)]
pub struct SyncStft<C, InD = Batch<MotifSlices>, OutD = Spectorgram>
where
    C: RunCtx + Adjustable + 'static,
    InD: Dims + 'static,
    OutD: Dims + 'static,
{
    id: SynKey,
    run_config: StftRunConfig,
    pd: PhantomData<(C, InD, OutD)>,
}

impl<C, InD, OutD> SyncStft<C, InD, OutD>
where
    C: RunCtx + Adjustable + 'static,
    InD: Dims + 'static,
    OutD: Dims + 'static,
{
    pub fn new(id: SynKey, run_config: StftRunConfig) -> Result<Self, ThoError> {
        run_config.validate()?;
        Ok(Self { id, run_config, pd: PhantomData })
    }

    pub fn build_spectrogram(&self, input: &Tensor) -> Result<Tensor, ThoError> {

        let input_size = input.size();
        debug!("SyncStft::build_spectrogram: input_size={input_size:?} run_config={:?}", self.run_config);

        Ok(
            tch_result(tch_result(tch_result(input.f_totype(Float))?
            .f_stft(
                self.run_config.n_fft,
                self.run_config.hop_length,
                self.run_config.win_length,
                None::<Tensor>,
                self.run_config.normalized,
                self.run_config.onesided,
                true,
                self.run_config.align_to_window,
            ))?
            .f_abs())?
        )
    }
}

impl<C, InD, OutD> Adjustable for SyncStft<C, InD, OutD>
where
    C: RunCtx + Adjustable + 'static,
    InD: Dims + 'static,
    OutD: Dims + 'static,
{}

impl<C, InD, OutD> OutId for SyncStft<C, InD, OutD>
where
    C: RunCtx + Adjustable + 'static,
    InD: Dims + 'static,
    OutD: Dims + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<C, InD, OutD> Stage for SyncStft<C, InD, OutD>
where
    C: RunCtx + Adjustable + 'static,
    InD: Dims + 'static,
    OutD: Dims + 'static,
{
    type Ctx = C;
    type In<'a> = ThoTensor<InD, SynKey>;
    type Out = ThoTensor<OutD, SynKey>;
}

impl<C, InD, OutD> SyncStage for SyncStft<C, InD, OutD>
where
    C: RunCtx + Adjustable + 'static,
    InD: Dims + 'static,
    OutD: Dims + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "SyncStft::do_run"))]
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let spectrogram = self.build_spectrogram(&*input.local()?)?;
        debug!("SyncStft::do_run: input_id={:?} input_size={:?} output_size={:?} run_config={:?}", input.id(), input.size(), spectrogram.size(), self.run_config);
        ThoTensor::new(self.out_id(input.id()), spectrogram)
    }
}

fn stft_stage_id(run_config: &StftRunConfig) -> SynKey {
    SynKey::new([
        mono_key!(STFT, "Stft"),
        mono_key!(run_config.n_fft as usize, "n_fft"),
        mono_key!(run_config.hop_length.unwrap_or(0) as usize, "hop_length"),
        mono_key!(run_config.win_length.unwrap_or(0) as usize, "win_length"),
        mono_key!(run_config.normalized, "normalized"),
        mono_key!(run_config.onesided, "onesided"),
        mono_key!(run_config.align_to_window, "align_to_window"),
    ])
}

#[cfg(test)]
mod tests {
    use thoosa_common::tch_error::tch_result;
    use tch::{Device, Kind, Tensor};
    use thoosa_common::nets::common_nets::StftRunConfig;
    use thoosa_common::runner::run_ctx::RunCtxImpl;
    use thoosa_shared::ThoError;

    use crate::stages::stft::{SyncStft, stft_stage_id};

    #[test]
    fn test_sync_stft_build_spectrogram_stacks_batched_motifs() -> Result<(), ThoError> {
        let run_config = StftRunConfig::new(4, Some(2), Some(4), false, true, true);
        let input = tch_result(Tensor::from_slice(&[0.0f32, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]).f_view([2, 8]))?;
        let stage = SyncStft::<RunCtxImpl>::new(stft_stage_id(&run_config), run_config)?;

        let output = stage.build_spectrogram(&input)?;

        assert_eq!(output.size(), vec![2, 3, 3]);
        assert_eq!(output.kind(), Kind::Float);
        assert_eq!(output.device(), Device::Cpu);
        Ok(())
    }

    // #[test]
    // fn test_sync_stft_build_spectrogram_stacks_batched_motifs_1() -> Result<(), ThoError> {
    //     let run_config = StftRunConfig::new(16000, Some(4000), Some(4000), true, true, true);
    //     let input = Tensor::rand(&[15999], FLOAT_CUDA);
    //     let stage = SyncStft::<RunCtxImpl>::new(stft_stage_id(&run_config), run_config)?;
    //
    //     let output = stage.build_spectrogram(&input)?;
    //
    //     debug!("output={output:?}");
    //
    //     // assert_eq!(output.size(), vec![2, 3, 3]);
    //     // assert_eq!(output.kind(), Kind::Float);
    //     // assert_eq!(output.device(), Device::Cpu);
    //     Ok(())
    // }

}
