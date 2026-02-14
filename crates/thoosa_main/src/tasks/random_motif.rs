use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use cached_derive::Cached;
use parking_lot::RwLock;
use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
use thoosa_dummy_data_derive::DummyData;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::resource::{Resource, ResourceId};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage};
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::traits::{Adjustable, Data, MonoMetrics, WithId};
use thoosa_shared::{mono_key, CopyToDevice, TensorIdSliceIdx, ThoError};
use thoosa_store::Store;
use thoosa_with_data::with_data;

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

#[derive(Debug, Clone)]
pub struct RandomMotifOutput {
    id: SynKey,
    slices: Vec<TensorIdSliceIdx>,
    resources_map: HashMap<ResourceId, Resource>,
}

impl RandomMotifOutput {
    pub fn new(id: SynKey, slices: Vec<TensorIdSliceIdx>, resources_map: HashMap<ResourceId, Resource>) -> Self {
        Self {
            id,
            slices,
            resources_map,
        }
    }

    pub fn slices(&self) -> &Vec<TensorIdSliceIdx> {
        &self.slices
    }

    pub fn resources_map(&self) -> &HashMap<ResourceId, Resource> {
        &self.resources_map
    }
}

impl Data for RandomMotifOutput {
    type Metrics = MonoMetrics<0, 1>;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        self.slices.len().into()
    }

    fn device(&self) -> usize {
        0
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
    async fn copy_to_device(&self, _dst_device: usize) -> Result<Self, ThoError> {
        Ok(self.clone())
    }
}

#[derive(Debug, Cached, Named)]
pub struct RandomMotif<C>
where
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    store: Arc<RwLock<Box<dyn Store>>>,
    find_motifs_run_id: RunId,
    pd: PhantomData<C>,
}

impl<C> RandomMotif<C>
where
    C: RunCtx + Adjustable + 'static,
{
    pub fn new(store: Arc<RwLock<Box<dyn Store>>>, find_motifs_run_id: RunId) -> Self {
        Self {
            id: SynKey::new([
                mono_key!("RandomMotif", "stage"),
                mono_key!(find_motifs_run_id, format!("find_motifs_run_id={find_motifs_run_id}")),
            ]),
            store,
            find_motifs_run_id,
            pd: PhantomData,
        }
    }
}

impl<C> OutId for RandomMotif<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<C> Stage for RandomMotif<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type In<'a> = RandomMotifInput;
    type Out = RandomMotifOutput;
    type Ctx = C;
}

impl<C> AsyncStage for RandomMotif<C>
where
    C: RunCtx + Adjustable + 'static,
{
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
            &self.store.read().list_run_config_values(self.find_motifs_run_id)?,
        )?;
        let input_len_adjuster = find_motifs_run_config.input_len_adjuster();

        let resources = self.store.read().list_resources_not_in_group(RANDOM_STATS_GROUP, max_slices)?;
        let (slices, resources_map) = random_non_overlapping_slices(&resources, max_slices, motif_width, input_len_adjuster)?;

        Ok(Arc::new(DevicesWrap::wrap(RandomMotifOutput::new(self.out_id(input.id()), slices, resources_map), cache_area.into())))
    }
}
