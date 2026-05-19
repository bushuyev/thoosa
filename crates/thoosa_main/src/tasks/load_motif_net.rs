use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use async_trait::async_trait;
use log::debug;
use parking_lot::RwLock;
use cached_derive::Cached;
use thoosa_named_derive::Named;
use thoosa_common::runner::common::cachable::DevicesWrap;
use thoosa_common::runner::common::{OutId, RunCtx};
use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::dims::{Batch, Dims, MotifProbability, Slice};
use thoosa_common::train_nets::{MotifNetParams, ThoModuleWithStore};
use thoosa_common::ws::{FsWorkspace, Workspace};
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::{mono_key, simple_id, ThoError};
use thoosa_shared::traits::{Adjustable, WithId};
use thoosa_shared::comp_id::SynKey;
#[derive(Debug, Clone, DummyData)]
pub struct LoadMotifNetInput {
    id: SynKey,
    net_id: PathBuf,
    // train_motif_nets_run_config_id:RunId
}


impl LoadMotifNetInput {
    pub fn new(net_id: PathBuf/*train_motif_nets_run_config_id: RunId*/) -> Self {
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
        let device_idx = ctx.device_idx();

        let input = input.load_from_any_device_(0).await?;//TODO clone_from_cpu
        let ws_guard = thoosa_shared::perf_log!(thoosa_shared::sync_read_lock!(self.ws));
        // TODO(perf): resolve the path under the workspace lock, then load the model outside it; also consider caching loaded net params by net_id/precision.
        let (params, module) = ws_guard.load_motif_net(&input.net_id, &ThoDevice::from(device_idx).into())?;

        let (channel, motif_id) = FsWorkspace::parse_motif_path(&input.net_id)?;

        Ok(Arc::new(DevicesWrap::wrap(DataWithExtra::new(module, MotifNetMeta::new(motif_id, channel, params)))))
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
