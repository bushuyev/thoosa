use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::RwLock;
use cached_derive::Cached;
use caching_runner::common::cachable::DevicesWrap;
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::applicables::ThoDevice;
use thoosa_common::dims::{Batch, Dims, MotifProbability, Slice};
use thoosa_common::train_nets::{MotifNetParams, ThoModuleWithStore};
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::run_configs::RunId;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Adjustable, SimpleId, WithId};
use crate::tasks::run_ctx_2::MotifsCtx;




#[derive(Debug, Clone, DummyData)]
pub struct LoadMotifNetInput {
    id:SimpleId,
    motif_id:u32,
    channel:u32,
    // train_motif_nets_run_id:RunId
}


impl LoadMotifNetInput {
    pub fn new(motif_id: u32, channel:u32/*train_motif_nets_run_id: RunId*/) -> Self {
        Self {
            id: format!("{}_{}", motif_id, channel).into(),
            motif_id,
            channel,
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
pub struct LoadMotifNet<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static,
{

    ws: Arc<RwLock<Box<dyn Workspace>>>,
    pd:PhantomData<C>
}

impl <C> LoadMotifNet<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static,
{
    pub fn new(ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Self {
        Self {
            ws: Arc::clone(ws),
            pd: PhantomData,
        }
    }
}

impl <C> Stage for LoadMotifNet<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = LoadMotifNetInput;
    type Out = DataWithExtra<ThoModuleWithStore<Slice, MotifProbability>, MotifNetParams>;
}

#[async_trait]
impl <C> AsyncStage for LoadMotifNet<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static,
{
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
        let (device_idx, cache_area) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area())
        };

        let input = input.clone_from_cpu().await?;
        let (params, module) = self.ws.read().load_motif_nets(input.channel, input.motif_id, &ThoDevice::from(device_idx).into())?.pop().ok_or(ThoosaError::new(format!("no nets for input={input:?}")))?;
        Ok(Arc::new(DevicesWrap::wrap(DataWithExtra::new(module, params), cache_area)))
    }
}

impl <C> OutId for LoadMotifNet<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        format!("Loaded_{}", input_id).into()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use parking_lot::RwLock;
    use tch::Device::Cuda;
    use caching_runner::common::cachable::{Cached, DevicesWrap};
    use caching_runner::common::{mock_run_ctx, MockRunCtx};
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_common::dims::{AnyMetrics, Dims};
    use thoosa_common::utils;
    use thoosa_common::ws::{mock_ws_simp, WsMockBuilder};
    use thoosa_shared::ThoosaError;
    use crate::tasks::load_motif_net::{LoadMotifNet, LoadMotifNetInput};



    #[tokio::test]
    async fn test_0() -> Result<(), ThoosaError>{

        GlobalCtx::init_devices(2);

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 5, 100, 1000u32)
                .with_load_motif_nets(vec![vec![
                    //channel 0
                    utils::motif_test_net(100, 1., Cuda(0)),
                ]])
                .build(),
        ));


        let load_motif_net = Arc::new(LoadMotifNet::<MockRunCtx>::new(&ws));

        load_motif_net.run_cached(Arc::new(DevicesWrap::wrap_cpu(LoadMotifNetInput::new(0, 0))), Arc::new(RwLock::new(mock_run_ctx::default(1)))).await?;

        Ok(())
    }

}