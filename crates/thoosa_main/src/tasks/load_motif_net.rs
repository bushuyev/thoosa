use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use log::debug;
use parking_lot::RwLock;
use cached_derive::Cached;
use thoosa_shared::runner::common::cachable::DevicesWrap;
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::dims::{Batch, Dims, MotifProbability, Slice};
use thoosa_common::train_nets::{MotifNetParams, ThoModuleWithStore};
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::run_configs::RunId;
use thoosa_shared::{mono_key, simple_id, ThoError};
use thoosa_shared::traits::{Adjustable, WithId};
use thoosa_shared::comp_id::SynKey;
use crate::tasks::run_ctx_2::MotifsCtx;




#[derive(Debug, Clone, DummyData)]
pub struct LoadMotifNetInput {
    id: SynKey,
    motif_id:u32,
    channel:u32,
    // train_motif_nets_run_id:RunId
}


impl LoadMotifNetInput {
    pub fn new(motif_id: u32, channel:u32/*train_motif_nets_run_id: RunId*/) -> Self {
        Self {
            id: SynKey::new([
                mono_key!(motif_id, format!("motif_id={motif_id}")),
                mono_key!(channel, format!("channel={channel}")),
            ]),
            motif_id,
            channel,
        }
    }
}

impl Input for LoadMotifNetInput {

}

impl WithId for  LoadMotifNetInput{
    type ID = SynKey;

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

#[derive(Debug, Clone)]
pub struct MotifNetMeta {
    motif_id:u32,
    channel:u32,
    pub motif_net_params:MotifNetParams
}

impl MotifNetMeta {
    pub fn new(motif_id:u32, channel: u32, motif_net_params: MotifNetParams) -> Self {
        Self { motif_id, channel, motif_net_params }
    }

    pub fn motif_id(&self) -> u32 {
        self.motif_id
    }

    pub fn channel(&self) -> u32 {
        self.channel
    }

    pub fn motif_net_params(&self) -> &MotifNetParams {
        &self.motif_net_params
    }
}

impl <C> Stage for LoadMotifNet<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = LoadMotifNetInput;
    type Out = DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>;
}

// #[async_trait]
impl <C> AsyncStage for LoadMotifNet<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static,
{
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let (device_idx, cache_area) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area())
        };

        let input = input.load_from_any_device_(0).await?;//TODO clone_from_cpu
        let mut motif_nets = self.ws.read().load_motif_nets(input.channel, input.motif_id, &ThoDevice::from(device_idx).into())?;

        debug!("LoadMotifNet::run: motif_nets={motif_nets:?}");

        let (params, module) = motif_nets
            .pop().ok_or_else(|| ThoError::new(format!("no nets for input={input:?}")))?;//TODO select best net or select with run config

        Ok(Arc::new(DevicesWrap::wrap(DataWithExtra::new(module, MotifNetMeta::new(input.motif_id, input.channel, params)), cache_area)))
    }
}

impl <C> OutId for LoadMotifNet<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        mono_key!("MotifNet", "MotifNet") + input_id//TODO should be same as in Ws::load_motif_nets

    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use parking_lot::RwLock;
    use tch::Device::Cuda;
    use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap};
    use thoosa_shared::runner::common::{mock_run_ctx, MockRunCtx};
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_common::utils;
    use thoosa_common::ws::WsMockBuilder;
    use thoosa_shared::ThoError;
    use crate::tasks::load_motif_net::{LoadMotifNet, LoadMotifNetInput};



    #[tokio::test]
    async fn test_0() -> Result<(), ThoError>{

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

        load_motif_net.run_cached(Arc::new(DevicesWrap::wrap(LoadMotifNetInput::new(0, 0), 0)), Arc::new(RwLock::new(mock_run_ctx::default(1)))).await?;

        Ok(())
    }

}
