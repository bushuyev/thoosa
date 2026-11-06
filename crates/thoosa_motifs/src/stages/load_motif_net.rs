use crate::dims::MotifProbability;
use crate::train_nets::MotifNetParams;
use cached_derive::Cached;
use parking_lot::RwLock;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use thoosa_common::data_with_extra::DataWithExtra;
use thoosa_common::dims::{Batch, Slice};
use thoosa_common::nets::common_nets::ThoModuleWithStore;
use thoosa_common::runner::common::cachable::DevicesWrap;
use thoosa_common::tho_device::ThoDevice;
use thoosa_cpu_data_derive::CpuData;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::traits::{Adjustable, AsyncStage, OutId, RunCtx, Stage, WithId};
use thoosa_shared::{ThoError, mono_key, perf_log, sync_read_lock};
#[derive(Debug, Clone, CpuData)]
#[cpu_data(shallow)]
pub struct LoadMotifNetInput {
    id: SynKey,
    net_id: PathBuf,
    quantize: bool,
}

impl LoadMotifNetInput {
    pub fn new(net_id: PathBuf, quantize: bool) -> Self {
        Self {
            id: SynKey::new([mono_key!(net_id.to_string_lossy().to_string(), format!("net_id={net_id:?}")), mono_key!(quantize, format!("quantize={quantize:?}"))]),
            net_id,
            quantize,
        }
    }
}

impl Input for LoadMotifNetInput {}

impl WithId for LoadMotifNetInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug, Cached, Named)]
pub struct LoadMotifNet<C>
//use only from ApplyCosine2D, to be removed - load is done in ApplyMotifNet
where
    C: RunCtx + Adjustable + 'static,
{
    ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
    #[expect(dead_code, reason = "Retains stored metadata and its existing construction and diagnostic representation.")]
    quantize: bool,
    pd: PhantomData<C>,
}

impl<C> LoadMotifNet<C>
where
    C: RunCtx + Adjustable + 'static,
{
    pub fn new(ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, quantize: bool) -> Self {
        Self { ws: Arc::clone(ws), quantize, pd: PhantomData }
    }
}

#[derive(Debug, Clone)]
pub struct MotifNetMeta {
    motif_id: u32,
    channel: u32,
    pub motif_net_params: MotifNetParams,
}

impl MotifNetMeta {
    pub fn new(motif_id: u32, channel: u32, motif_net_params: MotifNetParams) -> Self {
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

impl<C> Stage for LoadMotifNet<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = LoadMotifNetInput;
    type Out = Arc<DevicesWrap<DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>>>;
}

impl<C> AsyncStage for LoadMotifNet<C>
where
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "LoadMotifNet::run"))]
    async fn run(self: Arc<Self>, input: LoadMotifNetInput, ctx: Self::Ctx) -> Result<Self::Out, ThoError> {
        let (params, module) = perf_log!(sync_read_lock!(self.ws)).load_motif_net(&input.net_id, &ThoDevice::from(ctx.tho_device_idx()).into(), input.quantize, true)?;

        let (channel, motif_id) = crate::motifs_ws::parse_motif_path(&input.net_id)?;

        Ok(Arc::new(DevicesWrap::wrap(DataWithExtra::new(module.expect("no module"), MotifNetMeta::new(motif_id, channel, params)))))
    }
}

impl<C> OutId for LoadMotifNet<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        mono_key!("MotifNet", "MotifNet") + input_id //TODO should be same as in Ws::load_motif_nets
    }
}
