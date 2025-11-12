use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::RwLock;
use cached_derive::Cached;
use caching_runner::common::cachable::DevicesWrap;
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::dims::Dims;
use thoosa_common::train_nets::ThoModuleWithStore;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::run_configs::RunId;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Adjustable, SimpleId, WithId};
use crate::tasks::run_ctx_2::MotifsCtx;




#[derive(Debug, Clone, DummyData)]
pub struct LoadMotifNetInput {
    id:SimpleId,
    motif_id:u32,
    train_motif_nets_run_config:RunId
}

impl LoadMotifNetInput {
    pub fn new(motif_id: u32, train_motif_nets_run_config: RunId) -> Self {
        Self {
            id: format!("{}_{}", motif_id, train_motif_nets_run_config).into(),
            motif_id,
            train_motif_nets_run_config,
        }
    }
}

impl Input for LoadMotifNetInput {

}

impl WithId for  LoadMotifNetInput{
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}



#[derive(Debug, Cached)]
pub struct LoadMotifNet<DI, DO, C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DI: Dims + Send + 'static,
    DO: Dims + Send + 'static
{

    pd:PhantomData<(DI, DO, C)>
}

impl <DI, DO, C> LoadMotifNet<DI, DO, C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DI: Dims + Send + 'static,
    DO: Dims + Send + 'static
{
    pub fn new() -> Self {
        Self { pd:PhantomData }
    }
}

impl <DI, DO, C> Stage for LoadMotifNet<DI, DO, C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DI: Dims + Send + 'static,
    DO: Dims + Send + 'static
{
    type Ctx = C;
    type In<'a> = LoadMotifNetInput;
    type Out = ThoModuleWithStore<DI, DO>;
}

#[async_trait]
impl <DI, DO, C> AsyncStage for LoadMotifNet<DI, DO, C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DI: Dims + Send + 'static,
    DO: Dims + Send + 'static
{
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
        todo!()
    }
}

impl <DI, DO, C> OutId for LoadMotifNet<DI, DO, C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DI: Dims + Send + 'static,
    DO: Dims + Send + 'static
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        ()//TODO format!("Loaded_{}", input_id).into()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use parking_lot::RwLock;
    use caching_runner::common::cachable::{Cached, DevicesWrap};
    use caching_runner::common::{mock_run_ctx, MockRunCtx};
    use thoosa_common::dims::{AnyMetrics, Dims};
    use thoosa_shared::ThoosaError;
    use crate::tasks::load_motif_net::{LoadMotifNet, LoadMotifNetInput};

    #[derive(Debug, Clone)]
    struct InDims {}
    impl Dims for InDims {
        type Metrics = AnyMetrics;
    }
    unsafe impl Send for InDims {}

    #[derive(Debug, Clone)]
    struct OutDims {}
    impl Dims for OutDims {
        type Metrics = AnyMetrics;
    }
    unsafe impl Send for OutDims {}

    #[tokio::test]
    async fn test_0() -> Result<(), ThoosaError>{

        let load_motif_net = Arc::new(LoadMotifNet::<InDims, OutDims, MockRunCtx>::new());

        load_motif_net.run_cached(Arc::new(DevicesWrap::wrap_cpu(LoadMotifNetInput::new(1, 2))), Arc::new(RwLock::new(mock_run_ctx::default(1)))).await?;

        Ok(())
    }

}