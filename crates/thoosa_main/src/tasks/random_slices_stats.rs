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
use thoosa_shared::traits::{Adjustable, MonoMetrics, WithId};
use thoosa_shared::{mono_key, TensorIdSliceIdx, ThoError};
use thoosa_store::Store;
use thoosa_with_data::with_data;

use crate::tasks::cross_dims::{ChannelsData2, ChannelsMotifsDists};
use crate::tasks::load_motif_and_clc_dist::{DistOutputBuilder, MotifSlicesSelector};
use crate::tasks::motif_loader::MotifMeta;
use crate::tasks::random_motif::{RandomMotif, RandomMotifInput, RandomMotifOutput};
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
    random_motifs: RwLock<HashMap<RunId, Arc<RandomMotif<L, C>>>>,
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
            random_motifs: RwLock::new(HashMap::new()),
            pd: PhantomData,
        }
    }

    fn random_motif_stage(&self, find_motifs_run_id: RunId) -> Arc<RandomMotif<L, C>> {
        if let Some(stage) = self.random_motifs.read().get(&find_motifs_run_id) {
            return Arc::clone(stage);
        }

        let mut random_motifs = self.random_motifs.write();
        Arc::clone(random_motifs.entry(find_motifs_run_id).or_insert_with(|| {
            Arc::new(RandomMotif::new(
                Arc::clone(&self.store),
                Arc::clone(&self.loader),
                Arc::clone(&self.ws),
                find_motifs_run_id,
            ))
        }))
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
        assert_ne!(motif_width, 0, "RandomStats: motif_width must be > 0");

        let random_motif = self
            .random_motif_stage(find_motifs_run_id)
            .run_cached(
                Arc::new(DevicesWrap::wrap(RandomMotifInput::new(max_slices, motif_width), cache_area.into())),
                ctx.branch(Some("random_stats_random_motif"), false, false),
            )
            .await?;
        let motifs = with_data!(
            ctx.branch(Some("random_stats_random_motif_with_data"), false, false),
            random_motif;
            |_ctx, random_motif: &RandomMotifOutput| {
                random_motif
                    .motifs()
                    .iter()
                    .map(|motif| ThoTensor::new(motif.id().clone(), motif.t_ref().shallow_clone()))
                    .collect::<Result<Vec<_>, _>>()
            }
        )?;

        debug!("RandomStats::run: motifs.len={}", motifs.len());

        let channels = motifs.len();
        assert_ne!(channels, 0, "no channels");

        let mut stats_values = Vec::with_capacity(channels);
        for (channel, motif) in motifs.into_iter().enumerate() {
            let slice_count = motif.t_ref().size()[0];
            if slice_count < 2 {
                return Err(ThoError::new(format!("RandomStats: not enough slices for channel {channel}, got {slice_count}")));
            }

            let stats_input = SlicesStatsInput::new(
                metrics,
                aggr,
                DataWithExtra::new(motif, MotifMeta { channel: channel as i64, find_motifs_run_id, window: motif_width }),
            );
            let stats_input = Arc::new(DevicesWrap::wrap(stats_input, cache_area.into()));
            let stats_input_size = stats_input.size().clone();
            let stats_out_id = self.clc_stats.out_id(stats_input.id());
            let stats = OOMHandler::new(vec![Adjusters::Input(Box::new(SplitsAdjuster::<MotifSlicesSelector, MonoMetrics<0, 2>>::new([2, -1].into())))], None)
                .run(
                    &self.clc_stats,
                    Arc::clone(&stats_input),
                    ctx.branch(Some(&format!("random_stats_channel_{channel}")), false, false),
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
#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="random_non_overlapping_slices"))]
pub(crate) fn random_non_overlapping_slices(
    resources: &[Resource],
    number_of_slices: u32,
    slice_width: u32,
    input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync>,
) -> Result<(Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>), ThoError> {


    assert_ne!(number_of_slices, 0, "number_of_slices is 0");
    assert_ne!(slice_width, 0, "slice_width is 0");

    #[derive(Clone)]
    struct ResState {
        res: Resource,
        starts: Vec<u32>, // available slice starts (aligned, non-overlapping by construction)
        picked: usize,
    }

    let mut rng = rand::thread_rng();

    let mut states: Vec<ResState> = resources
        .iter()
        .filter_map(|r| {
            let resource_len = input_len_adjuster(r.length());

            if resource_len < slice_width {
                return None;
            }
            let bins = resource_len / slice_width; // number of non-overlapping aligned slices
            if bins == 0 {
                return None;
            }
            let mut starts: Vec<u32> = (0..bins).map(|i| i * slice_width).collect();
            starts.shuffle(&mut rng);
            Some(ResState {
                res: r.clone(),
                starts,
                picked: 0,
            })
        })
        .collect();

    if states.is_empty() {
        return Err(ThoError::new(format!("random_non_overlapping_slices: no suitable resource for {number_of_slices} of {slice_width} width")));
    }

    let available_total: usize = states.iter().map(|s| s.starts.len()).sum();
    if available_total < number_of_slices as usize {
        return Err(ThoError::new(format!("random_non_overlapping_slices: not enough resources ({available_total}) available for {number_of_slices} random slices")));
    }

    let mut chosen: Vec<TensorIdSliceIdx> = Vec::with_capacity(number_of_slices as usize);
    let mut used_resources: HashMap<ResourceId, Resource> = HashMap::new();

    while chosen.len() < number_of_slices as usize {
        // Find the minimum picked-count among resources that still have available slices.
        let mut min_picked: Option<usize> = None;
        for s in &states {
            if s.starts.is_empty() {
                continue;
            }
            min_picked = Some(match min_picked {
                None => s.picked,
                Some(m) => m.min(s.picked),
            });
        }


        let min_p = min_picked.expect("no slices");

        // Candidates = those with min_picked and remaining starts.
        let mut candidates: Vec<usize> = Vec::new();
        for (i, s) in states.iter().enumerate() {
            if !s.starts.is_empty() && s.picked == min_p {
                candidates.push(i);
            }
        }

        let pick_i = *candidates.choose(&mut rng).expect("cant choose slices");

        let s = &mut states[pick_i];
        let start = s.starts.pop().expect("no more slices");
        s.picked += 1;

        chosen.push(TensorIdSliceIdx(s.res.id(), start));
        used_resources.entry(s.res.id()).or_insert(s.res.clone());
    }

    Ok((chosen, used_resources))
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
            Resource::new(1, 1, "resource_a", 10),
            Resource::new(2, 1, "resource_b", 10),
        ];
        let motif_width = 10;
        let input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync> = Arc::new(|len| len);

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
        
        assert!(format!("{err:?}").contains("no suitable resource for 2 of 10 width"));
    }
}
