use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use cached_derive::Cached;
use log::{debug, warn};
use parking_lot::RwLock;
use rand::seq::{IndexedRandom, SliceRandom};
use rand::Rng;
use tch::Tensor;
use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::{Resource, ResourceId};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::runner::adjusters::splits::{SplitsAdjuster, SplitsAdjusterContextImpl};
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage};
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::runner::oom_handler::{Adjusters, Contexts, NoopRunCtxContext, NoopRunnableContext, OOMHandler};
use thoosa_shared::traits::{Adjustable, DataId, MonoMetrics, WithId};
use thoosa_shared::{mono_key, TensorIdSliceIdx, ThoError};
use thoosa_store::Store;
use thoosa_with_data::with_data;

use crate::tasks::cross_dims::{ChannelsData2, ChannelsMotifsDists};
use crate::tasks::load_motif_and_clc_dist::{DistOutputBuilder, MotifSlicesSelector};
use crate::tasks::motif_loader::{load_motif_slices, MotifMeta};
use crate::tasks::slices_stats::{Aggr, Metrics, SlicesStats, SlicesStatsInput};

pub const RANDOM_STATS_GROUP: &str = "random_avg_dist"; //dummy group to load random resources, TODO - add proper method to SqlStore
pub const RANDOM_AVG_DIST_GROUP: &str = RANDOM_STATS_GROUP;

#[derive(Debug, Clone, DummyData)]
pub struct RandomStatsInput {
    id: SynKey,
    max_slices: u32,
    find_motifs_run_id: RunId,
    metrics: Metrics,
    aggr: Aggr,
}

impl RandomStatsInput {
    pub fn new(max_slices: u32, find_motifs_run_id: RunId, metrics: Metrics, aggr: Aggr) -> Self {
        Self {
            //TODO
            id: SynKey::new([
                mono_key!("RandomStatsInput", "stage"),
                mono_key!(find_motifs_run_id, format!("find_motifs_run_id={find_motifs_run_id}")),
                mono_key!(format!("{metrics:?}"), format!("metrics={metrics:?}")),
                mono_key!(format!("{aggr:?}"), format!("aggr={aggr:?}")),
            ]),
            max_slices,
            find_motifs_run_id,
            metrics,
            aggr,
        }
    }
}

impl Input for RandomStatsInput {}

impl WithId for RandomStatsInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug, Named)]
pub struct RandomStats<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    store: Arc<RwLock<Box<dyn Store>>>,
    loader: Arc<L>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    clc_stats: Arc<SlicesStats<C>>,
    pd: PhantomData<C>,
}

impl<L, C> RandomStats<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    pub fn new(store: Arc<RwLock<Box<dyn Store>>>, loader: Arc<L>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Self {
        Self {
            store,
            loader,
            ws: Arc::clone(ws),
            clc_stats: Arc::new(SlicesStats::<C>::new()),
            pd: PhantomData,
        }
    }
}

impl<L, C> Stage for RandomStats<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type In<'a> = RandomStatsInput;
    type Out = ThoTensor<ChannelsMotifsDists, SynKey>;
    type Ctx = C;
}

// #[async_trait]
impl<L, C> AsyncStage for RandomStats<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="RandomStats::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<RandomStatsInput>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError>
    where
        for<'a> Self::In<'a>: Send + Sync,
    {
        let device_idx = ctx.device_idx();
        let cache_area = ctx.cache_area();
        let input_id = input.id().clone();

        let (max_slices, find_motifs_run_id, metrics, aggr) = with_data!(
            ctx.branch(Some("random_stats_input"), false, false),
            input;
            |_ctx, input: &RandomStatsInput| {
                Ok::<(u32, RunId, Metrics, Aggr), ThoError>((input.max_slices, input.find_motifs_run_id, input.metrics, input.aggr))
            }
        )?;

        if max_slices < 2 {
            return Err(ThoError::new(format!("RandomStats: max_slices must be >= 2, got {max_slices}")));
        }

        let find_motifs_run_config = FindMotifsRunConfig::from_properties_map_with_id(find_motifs_run_id, &self.store.read().list_run_config_values(find_motifs_run_id)?)?;
        let motif_width = find_motifs_run_config.window;
        let input_len_adjuster = find_motifs_run_config.input_len_adjuster();
        assert_ne!(motif_width, 0, "RandomStats: motif_width must be > 0");

        let resources = self.store.read().list_resources_not_in_group(RANDOM_STATS_GROUP, max_slices)?;
        let (slices, resources_map) = random_non_overlapping_slices(&resources, max_slices, motif_width, input_len_adjuster)?;

        debug!("RandomStats::run: slices.len={} resources_map.len={}", slices.len(), resources_map.len());
        debug!("RandomStats::run: resources={resources:?}");

        let channels = self.ws.read().channels();
        assert_ne!(channels, 0, "no channels");

        let mut stats_values = Vec::with_capacity(channels as usize);
        for channel in 0..channels {
            let (window, motif) = load_motif_slices(
                DataId(channel),
                find_motifs_run_id,
                channel as i64,
                resources_map.clone(),
                slices.clone(),
                Arc::clone(&self.store),
                Arc::clone(&self.loader),
                ctx.branch(Some(&format!("random_stats_channel_{channel}")), false, false),
                &self.ws,
                "random_stats",
            )
            .await?;

            let slice_count = motif.t_ref().size()[0];
            if slice_count < 2 {
                return Err(ThoError::new(format!("RandomStats: not enough slices for channel {channel}, got {slice_count}")));
            }

            let stats_input = SlicesStatsInput::new(
                metrics,
                aggr,
                DataWithExtra::new(motif, MotifMeta { channel: channel as i64, find_motifs_run_id, window }),
            );
            let stats_input = Arc::new(DevicesWrap::wrap(stats_input, cache_area.into()));
            let stats_input_size = stats_input.size().clone();
            let stats_out_id = self.clc_stats.out_id(stats_input.id());
            let stats = OOMHandler::new(vec![Adjusters::Input(Box::new(SplitsAdjuster::<MotifSlicesSelector, MonoMetrics<0, 2>>::new([2, -1].into())))], None)
                .run(
                    &self.clc_stats,
                    Arc::clone(&stats_input),
                    ctx.branch(Some(&format!("random_stats_channel_{channel}")), false, false).branch(Some("random_stats_dist"), false, false),
                    Contexts::new(
                        SplitsAdjusterContextImpl::new(stats_input_size, DistOutputBuilder::new(stats_out_id, aggr)),
                        NoopRunnableContext {},
                        NoopRunCtxContext {},
                    ),
                )
                .await?;
            stats_values.push(stats.value());
        }

        let stats_t = Tensor::from_slice(stats_values.as_slice()).view([channels as i64, 1]).to_device(ThoDevice::from(device_idx).into());

        Ok(Arc::new(DevicesWrap::wrap(ThoTensor::new(&input_id + &SynKey::new(["random_stats".into()]), stats_t)?, cache_area.into())))
    }
}

fn random_non_overlapping_slices(
    resources: &[Resource],
    max_slices: u32,
    motif_width: u32,
    input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync>,
) -> Result<(Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>), ThoError> {
    if resources.is_empty() {
        return Err(ThoError::new("RandomStats: no resources available for random slices"));
    }

    let mut candidates = resources
        .iter()
        .filter_map(|resource| {
            let adjusted_length = input_len_adjuster(resource.length());
            (adjusted_length >= motif_width).then_some((resource.clone(), adjusted_length))
        })
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        return Err(ThoError::new("RandomStats: no resources long enough for motif width"));
    }

    let mut rng = rand::rng();
    candidates.shuffle(&mut rng);

    let mut slices = Vec::new();
    let mut resources_map = HashMap::new();
    let mut used_positions: HashMap<ResourceId, Vec<u32>> = HashMap::new();

    let max_attempts = (max_slices as usize).saturating_mul(50).saturating_add(100);
    let mut attempts = 0usize;

    while slices.len() < max_slices as usize && attempts < max_attempts {
        attempts += 1;

        let (resource, adjusted_length) = match candidates.choose(&mut rng) {
            Some(candidate) => candidate,
            None => break,
        };

        let max_start = match adjusted_length.checked_sub(motif_width) {
            Some(max_start) => max_start,
            None => continue,
        };

        let candidate = rng.random_range(0..=max_start);
        let positions = used_positions.entry(resource.id()).or_default();

        if positions.iter().any(|&pos| slices_overlap(pos, candidate, motif_width)) {
            continue;
        }

        positions.push(candidate);
        resources_map.entry(resource.id()).or_insert_with(|| resource.clone());
        slices.push(TensorIdSliceIdx(resource.id(), candidate));
    }

    if slices.len() < 2 {
        return Err(ThoError::new(format!("RandomStats: expected at least 2 slices, got {}", slices.len())));
    }

    if slices.len() < max_slices as usize {
        warn!("RandomStats: requested max_slices={max_slices}, generated {}", slices.len());
    }

    Ok((slices, resources_map))
}

fn slices_overlap(a: u32, b: u32, width: u32) -> bool {
    let a_end = a.saturating_add(width);
    let b_end = b.saturating_add(width);
    a < b_end && b < a_end
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use tch::kind::FLOAT_CPU;
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::runner::cache_areas::CacheArea;
    use thoosa_shared::runner::common::cachable::WithDataRef;
    use thoosa_shared::runner::common::mock_run_ctx;
    use thoosa_shared::runner::common::runnable::AsyncStage;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_store::MockStoreBuilder;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_random_stats_basic() -> Result<(), ThoError> {
        GlobalCtx::init_devices(1);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 3, 100, 1000)));
        let resources = vec![
            Resource::new(1, 1, "resource_a", 10),
            Resource::new(2, 1, "resource_b", 10),
            Resource::new(3, 1, "resource_c", 10),
            Resource::new(4, 1, "resource_d", 10),
        ];

        let run_id = 42;
        let motif_width = 6;
        let store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
                .with_list_resources_not_in_group([(RANDOM_STATS_GROUP, Ok(resources.clone()))])
                .with_list_run_config_values(
                    run_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(run_id), 1, motif_width, None, None)
                        .to_properties_iter(None)
                        .collect()),
                )
                .build(),
        ));

        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached([
                    (resources[0].clone(), Tensor::full(30, 0., FLOAT_CPU).view((3, -1))),
                    (resources[1].clone(), Tensor::full(30, 1., FLOAT_CPU).view((3, -1))),
                    (resources[2].clone(), Tensor::full(30, 2., FLOAT_CPU).view((3, -1))),
                    (resources[3].clone(), Tensor::full(30, 3., FLOAT_CPU).view((3, -1))),
                ])
                .finish(),
        );

        let stage = Arc::new(RandomStats::new(Arc::clone(&store), loader, &ws));
        let ctx = Arc::new(RwLock::new(mock_run_ctx::default(0)));
        let input = Arc::new(DevicesWrap::wrap(RandomStatsInput::new(4, run_id, Metrics::EucDist, Aggr::Mean(1.0)), CacheArea::default()));

        let output = stage.run(input, ctx.write().branch(Some("random_stats"), false, false)).await?;

        let mut ctx_for_data = ctx.write().branch(Some("random_stats_data"), false, false);
        let size = output
            .with_data_ref(0, &mut ctx_for_data, None, |t: &ThoTensor<ChannelsMotifsDists, SynKey>, _ctx| {
                debug!("t=\n{}", t.as_ref());
                assert!(Tensor::from_slice(&[4.0825f32, 4.0825, 4.0825]).view((3, 1)).allclose(t.as_ref(), 0.02, 0.02, false));

                Ok::<_, ThoError>(t.t_ref().size())
            })
            .await?;

        assert_eq!(size, vec![3, 1]);

        Ok(())
    }

    #[test]
    fn test_random_non_overlapping_slices_uses_adjusted_length() -> Result<(), ThoError> {
        let resources = vec![
            Resource::new(1, 1, "resource_a", 11),
            Resource::new(2, 1, "resource_b", 11),
        ];
        let motif_width = 10;
        let input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync> = Arc::new(|len| len - 1);

        let (slices, resources_map) = random_non_overlapping_slices(&resources, 2, motif_width, input_len_adjuster)?;
        assert_eq!(slices.len(), 2);
        assert_eq!(resources_map.len(), 2);
        assert!(slices.iter().all(|slice| slice.1 == 0));

        Ok(())
    }

    #[test]
    fn test_random_non_overlapping_slices_filters_resources_by_adjusted_length() {
        let resources = vec![Resource::new(1, 1, "resource_a", 10)];
        let motif_width = 10;
        let input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync> = Arc::new(|len| len - 1);

        let err = random_non_overlapping_slices(&resources, 2, motif_width, input_len_adjuster).expect_err("expected adjusted-length filtering error");
        assert!(format!("{err:?}").contains("no resources long enough"));
    }
}
