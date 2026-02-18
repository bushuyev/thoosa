use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use async_trait::async_trait;
use log::debug;
use parking_lot::RwLock;
use cached_derive::Cached;
use thoosa_named_derive::Named;
use thoosa_shared::runner::common::cachable::DevicesWrap;
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::dims::{Batch, Dims, MotifProbability, Slice};
use thoosa_common::train_nets::{MotifNetParams, ThoModuleWithStore};
use thoosa_common::ws::{FsWorkspace, Workspace};
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::run_configs::RunId;
use thoosa_shared::{mono_key, simple_id, ThoError};
use thoosa_shared::traits::{Adjustable, WithId};
use thoosa_shared::comp_id::SynKey;
#[derive(Debug, Clone, DummyData)]
pub struct LoadMotifNetInput {
    id: SynKey,
    net_id: PathBuf,
    // train_motif_nets_run_id:RunId
}


impl LoadMotifNetInput {
    pub fn new(net_id: PathBuf/*train_motif_nets_run_id: RunId*/) -> Self {
        let net_id_str = net_id.to_string_lossy().to_string();
        Self {
            id: SynKey::new([
                mono_key!(net_id_str.clone(), format!("net_id={net_id_str}")),
            ]),
            net_id,
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


#[derive(Debug, Cached, Named)]
pub struct LoadMotifNet<C>
where
    C: RunCtx + Adjustable + 'static,
{

    ws: Arc<RwLock<Box<dyn Workspace>>>,
    pd:PhantomData<C>
}

impl <C> LoadMotifNet<C>
where
    C: RunCtx + Adjustable + 'static,
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
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = LoadMotifNetInput;
    type Out = DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>;
}

// #[async_trait]
impl <C> AsyncStage for LoadMotifNet<C>
where
    C: RunCtx + Adjustable + 'static,
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="LoadMotifNet::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let (device_idx, cache_area) = (ctx.device_idx(), ctx.cache_area());

        let input = input.load_from_any_device_(0).await?;//TODO clone_from_cpu
        let (params, module) = self
            .ws
            .try_read_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout")
            .load_motif_net(&input.net_id, &ThoDevice::from(device_idx).into())?;

        let (channel, motif_id) = FsWorkspace::parse_motif_path(&input.net_id)?;

        Ok(Arc::new(DevicesWrap::wrap(DataWithExtra::new(module, MotifNetMeta::new(motif_id, channel, params)), cache_area.into())))
    }
}

impl <C> OutId for LoadMotifNet<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        mono_key!("MotifNet", "MotifNet") + input_id//TODO should be same as in Ws::load_motif_nets

    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::path::PathBuf;
    use parking_lot::RwLock;
    use tch::Device::Cuda;
    use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap};
    use thoosa_shared::runner::cache_areas::CacheArea;
    use thoosa_shared::runner::common::{mock_run_ctx, MockRunCtx};
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_common::utils;
    use thoosa_common::ws::WsMockBuilder;
    use thoosa_shared::ThoError;
    use crate::tasks::load_motif_net::{LoadMotifNet, LoadMotifNetInput};



    #[tokio::test]
    async fn test_0() -> Result<(), ThoError>{

        GlobalCtx::init_devices(2);

        let net_id = PathBuf::from("0/0_0");
        let (params_fn, module_fn) = utils::motif_test_net(100, 1., Cuda(0));
        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 5, 100, 1000u32)
                .with_load_motif_net(vec![(net_id.clone(), params_fn, module_fn)])
                .build(),
        ));


        let load_motif_net = Arc::new(LoadMotifNet::<MockRunCtx>::new(&ws));

        load_motif_net.run_cached(Arc::new(DevicesWrap::wrap(LoadMotifNetInput::new(net_id), CacheArea::default())), mock_run_ctx::default(1)).await?;

        Ok(())
    }

}
