use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use cached_derive::Cached;
use parking_lot::RwLock;
use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::resource::{Resource, ResourceId};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage};
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::traits::{Adjustable, Data, DataId, MonoMetrics, WithId};
use thoosa_shared::{mono_key, CopyToDevice, TensorIdSliceIdx, ThoError};
use thoosa_store::{Store, block_on_store_future};
use thoosa_with_data::with_data;

use crate::tasks::cross_dims::{ChannelsData2, MotifSlices};
use crate::tasks::motif_loader::load_motif_slices;
use crate::tasks::random_slices_stats::{random_non_overlapping_slices, RANDOM_STATS_GROUP};

#[derive(Debug, Clone, DummyData)]
pub struct RandomMotifInput {
    id: SynKey,
    max_slices: u32,
    motif_width: u32,
}

impl RandomMotifInput {
    pub fn new(max_slices: u32, motif_width: u32) -> Self {
        Self {
            id: SynKey::new([
                mono_key!(max_slices, format!("max_slices={max_slices}")),
                mono_key!(motif_width, format!("motif_width={motif_width}")),
            ]),
            max_slices,
            motif_width,
        }
    }

    pub fn max_slices(&self) -> u32 {
        self.max_slices
    }

    pub fn motif_width(&self) -> u32 {
        self.motif_width
    }
}

impl Input for RandomMotifInput {}

impl WithId for RandomMotifInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug)]
pub struct RandomMotifOutput {
    id: SynKey,
    motifs: Vec<ThoTensor<MotifSlices, DataId>>,
}

impl RandomMotifOutput {
    pub fn new(id: SynKey, motifs: Vec<ThoTensor<MotifSlices, DataId>>) -> Self {
        Self { id, motifs }
    }

    pub fn motifs(&self) -> &Vec<ThoTensor<MotifSlices, DataId>> {
        &self.motifs
    }
}

impl Data for RandomMotifOutput {
    type Metrics = MonoMetrics<0, 1>;
    type M = CpuOrGpu<Self>;

    fn size(&self) -> Self::Metrics {
        self.motifs.len().into()
    }

    fn device(&self) -> usize {
        self.motifs.first().map_or(0, |motif| motif.device())
    }
}

impl WithId for RandomMotifOutput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for RandomMotifOutput {}

#[async_trait]
impl CopyToDevice for RandomMotifOutput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        let mut motifs = Vec::with_capacity(self.motifs.len());
        for motif in &self.motifs {
            motifs.push(motif.copy_to_device(dst_device).await?);
        }
        Ok(Self { id: self.id.clone(), motifs })
    }
}

#[derive(Debug, Cached, Named)]
pub struct RandomMotif<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    store: Arc<Box<dyn Store>>,
    loader: Arc<L>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    find_motifs_run_id: RunId,
    pd: PhantomData<C>,
}

impl<L, C> RandomMotif<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    pub fn new(store: Arc<Box<dyn Store>>, loader: Arc<L>, ws: Arc<RwLock<Box<dyn Workspace>>>, find_motifs_run_id: RunId) -> Self {
        Self {
            id: SynKey::new([
                mono_key!("RandomMotif", "stage"),
                mono_key!(find_motifs_run_id, format!("find_motifs_run_id={find_motifs_run_id}")),
            ]),
            store,
            loader,
            ws,
            find_motifs_run_id,
            pd: PhantomData,
        }
    }
}

impl<L, C> OutId for RandomMotif<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<L, C> Stage for RandomMotif<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type In<'a> = RandomMotifInput;
    type Out = RandomMotifOutput;
    type Ctx = C;
}

impl<L, C> AsyncStage for RandomMotif<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="RandomMotif::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<RandomMotifInput>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError>
    where
        for<'a> Self::In<'a>: Send + Sync,
    {
        let cache_area = ctx.cache_area();
        let (max_slices, motif_width) = with_data!(
            ctx.branch(Some("random_motif_input"), false, false),
            input;
            |_ctx, input: &RandomMotifInput| {
                Ok::<_, ThoError>((input.max_slices(), input.motif_width()))
            }
        )?;

        if max_slices < 2 {
            return Err(ThoError::new(format!("RandomMotif: max_slices must be >= 2, got {max_slices}")));
        }
        if motif_width == 0 {
            return Err(ThoError::new("RandomMotif: motif_width must be > 0"));
        }

        let find_motifs_run_config = FindMotifsRunConfig::from_properties_map_with_id(
            self.find_motifs_run_id,
            &block_on_store_future(self.store.list_run_config_values(self.find_motifs_run_id))?,
        )?;
        let input_len_adjuster = find_motifs_run_config.input_len_adjuster();

        let resources = block_on_store_future(self.store.list_resources_not_in_group(RANDOM_STATS_GROUP, max_slices * 100/*TODO this is to compensate for filter, add proper coefficient */))?;
        let (slices, resources_map) = random_non_overlapping_slices(&resources, max_slices, motif_width, input_len_adjuster)?;

        let channels = thoosa_shared::with_perf_log(
            || self.ws.try_read_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
            "RandomMotif/run/read_ws",
        )
        .channels();
        assert_ne!(channels, 0, "no channels");

        let mut motifs = Vec::with_capacity(channels as usize);
        for channel in 0..channels {
            let (_window, motif) = load_motif_slices(
                DataId(0),
                self.find_motifs_run_id,
                channel as i64,
                resources_map.clone(),
                slices.clone(),
                Arc::clone(&self.store),
                Arc::clone(&self.loader),
                ctx.branch(Some(&format!("random_motif_channel_{channel}")), false, false),
                &self.ws,
                "random_stats",
            )
            .await?;

            let slice_count = motif.t_ref().size()[0];
            if slice_count < 2 {
                return Err(ThoError::new(format!("RandomMotif: not enough slices for channel {channel}, got {slice_count}")));
            }

            motifs.push(motif);
        }

        Ok(Arc::new(DevicesWrap::wrap(RandomMotifOutput::new(self.out_id(input.id()), motifs), cache_area.into())))
    }
}
