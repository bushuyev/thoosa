use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::RwLock;
use tch::{IndexOp, TchError, Tensor};
use cached_derive::Cached;
use caching_runner::common::cachable::{Cached, DataCacheWrapper};
use caching_runner::common::OutId;
use caching_runner::common::RunCtx;
use caching_runner::common::runnable::{AsyncStage, Stage};
use caching_runner::global_ctx::GlobalCtx;
use thoosa_common::transforms::filters::notch_band_rfft;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{SimpleId, WithId};
use torch_runner::data::ThoTensor2;
use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::run_ctx_2::RunCtx2;

#[derive(Debug, Cached)]
pub struct FreqFilter {
}

impl FreqFilter {
    pub fn new() -> Self {
        Self {}
    }
}

impl  OutId for FreqFilter {
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        SimpleId::from(format!("filtered_{}_inp", input_id))
    }
}

impl Stage for FreqFilter {
    type Ctx = RunCtx2;
    type In<'a> = ThoTensor2<ChannelsData2, SimpleId>;
    type Out = ThoTensor2<ChannelsData2, SimpleId>;
}

#[async_trait]
impl AsyncStage for FreqFilter {
    async fn run(
        self: Arc<Self>,
        input: Arc<DataCacheWrapper<Self::In<'static>>>,
        ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Arc<DataCacheWrapper<Self::Out>>, ThoosaError> {
        let (device_idx, cache_area) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area())
        };
        let num_devices = GlobalCtx::global().device_count();

        let filtered = input
            .with_data_ref(device_idx, None, |tensor| {

                notch_band_rfft(tensor.t_ref(), 200., 58., 62.)
                    .map(|t|ThoTensor2::new(self.out_id(tensor.id()), t))
                    .map_err(|e|e.into())

            })
            .await?;

        Ok(Arc::new(DataCacheWrapper::new(
            filtered,
            device_idx,
            num_devices,
            cache_area,
        )))
    }
}


#[cfg(test)]
mod tests {
    use crate::tasks::transforms::freq_filter::FreqFilter;

    #[test]
    fn test_0(){

        let bf = FreqFilter::new();

    }

}
