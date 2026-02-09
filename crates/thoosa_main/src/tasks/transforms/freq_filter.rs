use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use log::debug;
use parking_lot::RwLock;
use tch::{IndexOp, TchError, Tensor};
use tch::Device::Cuda;
use cached_derive::Cached;
use thoosa_named_derive::Named;
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use thoosa_shared::runner::common::OutId;
use thoosa_shared::runner::common::RunCtx;
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage};
use thoosa_shared::runner::global_ctx::GlobalCtx;
use thoosa_common::dims::{AnyMetrics, Dims};
use thoosa_common::find_motifs_run_config::InputFilter;
use thoosa_common::transforms::filters::{band_pass_filter, notch_band_rfft};
use thoosa_shared::{mono_key, ThoError};
use thoosa_shared::traits::{CachableIdTr, WithId};
use thoosa_shared::comp_id::SynKey;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::ws::Workspace;
use crate::tasks::cross_dims::ChannelsData2;
use thoosa_shared::runner::run_ctx_2::RunCtxWithPermits;

#[derive(Debug, Clone)]
struct BandPassDims;

impl Dims for BandPassDims {
    type Metrics = AnyMetrics;//TODO
}

#[derive(Debug, Cached, Named)]
pub struct FreqFilter<C>
where
    C: RunCtx + 'static,
{
    // filters:Vec<> TODO
    band_pass_tensor: Option<ThoTensor<BandPassDims, SynKey>>,
    _ctx: PhantomData<C>,
    num_channels:i64,
    sample_rate:f64
}

impl<C> FreqFilter<C>
where
    C: RunCtx + 'static,
{
    pub fn new(filter_config: &InputFilter, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<Self, ThoError> {
        let (num_channels, sample_rate)  = {
            let guard = ws.read();
            (guard.channels() as i64, guard.sample_rate() as f64)
        };

        let band_pass_tensor = match filter_config {
            InputFilter::Conv1(_) => {
                None//TODO
            }
            InputFilter::BandPassFilter(band_pass_config) => {
                let high_hz = band_pass_config.low_hz + band_pass_config.band_hz;

                let kernel = band_pass_filter(
                    band_pass_config.low_hz,
                    high_hz,
                    band_pass_config.kernel_size,
                    sample_rate, //TODO take from ws
                    1e-6
                ).repeat([num_channels, 1, 1]).to_device(Cuda(0));

                Some(ThoTensor::new(
                    SynKey::new([
                        mono_key!(format!("{}", band_pass_config.low_hz), "low_hz"),
                        mono_key!(format!("{}", high_hz), "high_hz"),
                    ]),
                    kernel
                )?)

            }
        };
        Ok(Self {
            band_pass_tensor,
            _ctx: PhantomData,
            num_channels,
            sample_rate
        })
    }
}

impl<C> OutId for FreqFilter<C>
where
    C: RunCtx + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        let stage_key = SynKey::new([mono_key!("filtered_inp", "stage")]);
        &stage_key + input_id
    }
}

impl<C> Stage for FreqFilter<C>
where
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = ThoTensor<ChannelsData2, SynKey>;
    type Out = ThoTensor<ChannelsData2, SynKey>;
}

// #[async_trait]
impl<C> AsyncStage for FreqFilter<C>
where
    C: RunCtx + 'static,
{

    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        mut ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {

        let device_idx = ctx.device_idx();
        let cache_area = ctx.cache_area();
        let out_key = {
            let out_id = self.out_id(input.id());
            Arc::new(out_id) as Arc<dyn CachableIdTr>
        };

        let num_devices = GlobalCtx::global().device_count();
        let mut ctx_for_data = ctx.branch(Some("freq_filter_data"), false, false);

        let filtered = input
            .with_data_ref(
                device_idx,
               &mut ctx_for_data, None, |input_tensor: &ThoTensor<ChannelsData2, SynKey>, _ctx| {

               let noise_filtered_t = notch_band_rfft(input_tensor.t_ref(), self.sample_rate, 58., 62.)?;//TODO params - noise from electricity 60Hz

               if let Some(band_filter) = self.band_pass_tensor.as_ref() {

                   let t =  band_filter.run_inner(|band_filter_inner| {
                       let input = noise_filtered_t.unsqueeze(0);


                       let output = input.conv1d(band_filter_inner, None::<Tensor>, 1, 0, 1, self.num_channels).view((self.num_channels, -1));//TODO channels from WS

                       debug!("FreqFilter::run: band_filter.size={:?} input.size={:?} output.size={:?}", band_filter_inner.size(), input.size(), output.size());
                       Ok(output)
                   })?;

                   Ok(ThoTensor::new(self.out_id(input_tensor.id()), t)?)

               } else {
                   Ok(ThoTensor::new(self.out_id(input_tensor.id()), noise_filtered_t)?)
               }
           })
            .await?;

        Ok(Arc::new(DevicesWrap::wrap(filtered, cache_area.into())))
    }
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use parking_lot::RwLock;
    use tch::kind::FLOAT_CPU;
    use tch::Tensor;
    use thoosa_shared::runner::common::cachable::DevicesWrap;
    use thoosa_shared::runner::cache_areas::CacheArea;
    use thoosa_shared::runner::common::runnable::AsyncStage;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_common::find_motifs_run_config::InputFilter;
    use thoosa_common::transforms::filters::BandPassFilterParams;
    use thoosa_shared::ThoError;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_shared::runner::run_ctx_2::RunCtxWithPermits;
    use crate::tasks::transforms::freq_filter::FreqFilter;
    use thoosa_shared::runner::common::RunCtx;

    #[tokio::test]
    async fn test_freq_filter() ->Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let filter_config = InputFilter::BandPassFilter(BandPassFilterParams{
            low_hz: 0.0,
            band_hz: 100.0,
            kernel_size: 100,
            stride: 1,
            padding: 0,
        });
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 20, 100, 10000)));

        let bf = FreqFilter::new(&filter_config, &ws)?;

        let mut ctx = RunCtxWithPermits::new(1, "test");

        let r = Arc::new(bf).run(
            Arc::new(DevicesWrap::wrap(ThoTensor::new(SynKey::new([mono_key!("test", "test")]), Tensor::rand(&[20, 100], FLOAT_CPU))?, CacheArea::default())),
            ctx.branch(Some("freq_filter_test"), false, false)
        ).await?;

        println!("r={r:?}");
        //TODO asserts

        Ok(())
    }

}
