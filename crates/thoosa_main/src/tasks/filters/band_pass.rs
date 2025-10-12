use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::RwLock;
use cached_derive::Cached;
use caching_runner::common::cachable::{Cached, DataCacheWrapper};
use caching_runner::common::OutId;
use caching_runner::common::runnable::Stage;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::resource::Resource;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{SimpleId, WithId};
use torch_runner::data::ThoTensor2;
use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::motif_loader::MotifLoader;
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::TensorLoader;

#[derive(Debug, Cached)]
struct BandPassFilter {
}

impl BandPassFilter {
    pub fn new() -> Self {
        Self {}
    }
}

impl  OutId for BandPassFilter {
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        SimpleId::from(format!("filtered_{}_inp", input_id))
    }
}

#[async_trait]
impl Stage for BandPassFilter {
    type Ctx = RunCtx2;
    type In<'a> = ThoTensor2<ChannelsData2, SimpleId>;
    type Out = ThoTensor2<ChannelsData2, SimpleId>;

    async fn run(self: Arc<Self>, _input: &'static Self::In<'static>, _ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    where
            for<'a> Self::In<'a>: Send + Sync
    {
        _input.run_inner(|t|{
            Ok(ThoTensor2::new(self.out_id(_input.id()), t.shallow_clone()))
        })

    }
}

#[cfg(test)]
mod tests {
    use crate::tasks::filters::band_pass::BandPassFilter;

    #[test]
    fn test_0(){

        let bf = BandPassFilter::new();

    }

}