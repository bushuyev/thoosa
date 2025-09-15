use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::RwLock;
use caching_runner::common::{CopyToDevice, Data, OutId};
use caching_runner::common::runnable::Stage;
use thoosa_common::dims::{ChannelsData, Dims, ThoTensor, TCD};
use thoosa_shared::applicables_runner::{Input, Output, Runner, ToOrdinal};
use thoosa_shared::resource::Resource;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Named, SimpleId, WithId};
use crate::tasks::run_ctx_2::RunCtx2;

#[derive(Debug)]
pub struct RawChannels {
    t:ThoTensor<ChannelsData, SimpleId>
}

impl WithId for RawChannels {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        todo!()
    }
}

impl Input for RawChannels {

}

impl Output for RawChannels {}

#[derive(Debug, cached_derive::Cached)]
#[cache_name = "LoadedTensorId"]
pub struct TensorLoader{
}

impl  Stage for TensorLoader {
    type In = Resource;
    type Out = RawChannels;
    type Ctx = RunCtx2;
}

impl Data for RawChannels {
    type Metrics = ();
    type Sub = RawChannels;
    type SubSelect = ();

    fn size(&self) -> Self::Metrics {
        todo!()
    }

    fn device(&self) -> usize {
        todo!()
    }
}

impl OutId for TensorLoader {
    fn out_id(&self, input_id: &<Self::In as WithId>::ID) -> <Self::Out as WithId>::ID {
        todo!()
    }
}
#[async_trait]
impl CopyToDevice for RawChannels {
    async fn copy_to_device(&self, dst_device: usize) ->Self {
        todo!()
    }
}


impl  TensorLoader {
    pub fn new() -> Self {
        Self {}
    }
}
#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use caching_runner::common::cachable::Cached;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::ThoosaError;
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::TensorLoader;

    #[tokio::test]
    async fn test_load() -> Result<(), ThoosaError>{
        let loader = Arc::new(TensorLoader::new());

        let ctx = Arc::new(parking_lot::RwLock::new(RunCtx2::new(0, 0, "aa")));
        
        loader.run_cached(
            Arc::new(Resource::new(0, 0, "aaa".to_string(), 20)), ctx
        ).await.map_err(|e|ThoosaError::new("zzz"))?;



        Ok(())
    }

}