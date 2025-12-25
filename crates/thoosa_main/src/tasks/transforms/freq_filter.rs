use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use log::debug;
use parking_lot::RwLock;
use tch::{IndexOp, TchError, Tensor};
use tch::Device::Cuda;
use cached_derive::Cached;
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use thoosa_shared::runner::common::OutId;
use thoosa_shared::runner::common::RunCtx;
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage};
use thoosa_shared::runner::global_ctx::GlobalCtx;
use thoosa_common::dims::{AnyMetrics, Dims};
use thoosa_common::find_motifs_run_config::InputFilter;
use thoosa_common::transforms::filters::{band_pass_filter, notch_band_rfft};
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{CachableIdTr, SimpleId, WithId};
use thoosa_common::tho_tensor::ThoTensor;
use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::run_ctx_2::RunCtx2;

#[derive(Debug, Clone)]
struct BandPassDims;

impl Dims for BandPassDims {
    type Metrics = AnyMetrics;//TODO
}

#[derive(Debug, Cached)]
pub struct FreqFilter<C>
where
    C: RunCtx + 'static,
{
    // filters:Vec<> TODO
    band_pass_tensor: Option<ThoTensor<BandPassDims, SimpleId>>,
    _ctx: PhantomData<C>,
}

impl<C> FreqFilter<C>
where
    C: RunCtx + 'static,
{
    pub fn new(filter_config: &InputFilter) -> Result<Self, ThoosaError> {
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
                    200., //TODO take from ws
                    1e-6
                ).repeat([20, 1, 1]).to_device(Cuda(0));

                Some(ThoTensor::new(
                    format!("{}-{}", band_pass_config.low_hz, high_hz).into(),
                    kernel
                )?)

            }
        };
        Ok(Self {
            band_pass_tensor,
            _ctx: PhantomData,
        })
    }
}

impl<C> OutId for FreqFilter<C>
where
    C: RunCtx + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        SimpleId::from(format!("filtered_{}_inp", input_id))
    }
}

impl<C> Stage for FreqFilter<C>
where
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = ThoTensor<ChannelsData2, SimpleId>;
    type Out = ThoTensor<ChannelsData2, SimpleId>;
}

// #[async_trait]
impl<C> AsyncStage for FreqFilter<C>
where
    C: RunCtx + 'static,
{

    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {

        let (device_idx, cache_area) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area())
        };
        let out_key = {
            let out_id = self.out_id(input.id());
            Arc::new(out_id) as Arc<dyn CachableIdTr>
        };

        // if ctx.read().acquired_permits() == 0 {
        //     let result = GlobalCtx::global()
        //         .acquire_many(device_idx, 1, out_key.clone())
        //         .await?;
        //     ctx.write().add_permit(result);
        // }


        let num_devices = GlobalCtx::global().device_count();

        let filtered = input
            .with_data_ref(
                device_idx,
               Arc::clone(&ctx), None, |tensor: &ThoTensor<ChannelsData2, SimpleId>| {

               let t = notch_band_rfft(tensor.t_ref(), 200., 58., 62.)?;//TODO params

               if let Some(band_filter) = self.band_pass_tensor.as_ref() {

                   let t =  band_filter.run_inner(|band_filter_inner| {
                       let input = t.unsqueeze(0);


                       let output = input.conv1d(band_filter_inner, None::<Tensor>, 1, 0, 1, 20).view((20, -1));//TODO channels from WS

                       debug!("FreqFilter::run: band_filter.size={:?} input.size={:?} output.size={:?}", band_filter_inner.size(), input.size(), output.size());
                       Ok(output)
                   })?;

                   Ok(ThoTensor::new(self.out_id(tensor.id()), t)?)

               } else {
                   Ok(ThoTensor::new(self.out_id(tensor.id()), t)?)
               }
           })
            .await?;

        Ok(Arc::new(DevicesWrap::wrap(filtered, cache_area)))
    }
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use parking_lot::RwLock;
    use tch::kind::FLOAT_CPU;
    use tch::Tensor;
    use thoosa_shared::runner::common::cachable::DevicesWrap;
    use thoosa_shared::runner::common::runnable::AsyncStage;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_common::find_motifs_run_config::InputFilter;
    use thoosa_common::transforms::filters::BandPassFilterParams;
    use thoosa_shared::ThoosaError;
    use thoosa_common::tho_tensor::ThoTensor;
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::transforms::freq_filter::FreqFilter;

    #[tokio::test]
    async fn test_freq_fultder() ->Result<(),ThoosaError> {
        GlobalCtx::init_devices(2);
        let filter_config = InputFilter::BandPassFilter(BandPassFilterParams{
            low_hz: 0.0,
            band_hz: 100.0,
            kernel_size: 100,
            stride: 1,
            padding: 0,
        });

        let bf = FreqFilter::new(&filter_config)?;

        let ctx = Arc::new(RwLock::new(RunCtx2::new(1, 1, "test")));

        let r = Arc::new(bf).run(
            Arc::new(DevicesWrap::wrap(ThoTensor::new("test".into(), Tensor::rand(&[20, 100], FLOAT_CPU))?, 0)),
            ctx.clone()
        ).await?;

        println!("r={r:?}");


        Ok(())
    }

}
