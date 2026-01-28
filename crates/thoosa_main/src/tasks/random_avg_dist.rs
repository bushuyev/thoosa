use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use log::{debug, warn};
use parking_lot::RwLock;
use rand::seq::{IndexedRandom, SliceRandom};
use rand::Rng;
use tch::Kind::Float;
use tch::Tensor;
use cached_derive::Cached;
use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::utils::z_norm_dims;
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_motifs::distance::average_euclidean_distance;
use thoosa_motifs::pearson_corr::pearson_corr_distance_matrix_0_2_single;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::resource::{Resource, ResourceId};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage};
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::{mono_key, TensorIdSliceIdx, ThoError};
use thoosa_shared::traits::WithId;
use thoosa_store::Store;
use thoosa_with_data::with_data;

use crate::tasks::cross_dims::{ChannelsData2, ChannelsMotifsDists};
use crate::tasks::motif_loader::load_motif_slices;

pub const RANDOM_AVG_DIST_GROUP: &str = "random_avg_dist";//dummy group to load random resources, TODO - add proper method to SqlStore

#[derive(Debug, Clone, DummyData)]
pub struct RandomAvgDistInput {
    id: SynKey,
    max_slices: u32,
    find_motifs_run_id: RunId,
}

impl RandomAvgDistInput {
    pub fn new(max_slices: u32, find_motifs_run_id: RunId) -> Self {
        Self {
            id: SynKey::new([
                mono_key!("RandomAvgDistInput", "stage"),
                mono_key!(max_slices, format!("max_slices={max_slices}")),
                mono_key!(find_motifs_run_id, format!("find_motifs_run_id={find_motifs_run_id}")),
            ]),
            max_slices,
            find_motifs_run_id,
        }
    }
}

impl Input for RandomAvgDistInput {}

impl WithId for RandomAvgDistInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug, Named, Cached)]
pub struct RandomAvgDist<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{
    store: Arc<RwLock<Box<dyn Store>>>,
    loader: Arc<L>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    pd: PhantomData<C>,
}

impl<L, C> RandomAvgDist<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{
    pub fn new(store: Arc<RwLock<Box<dyn Store>>>, loader: Arc<L>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Self {
        Self {
            store,
            loader,
            ws: Arc::clone(ws),
            pd: PhantomData,
        }
    }
}

impl<L, C> Stage for RandomAvgDist<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{
    type In<'a> = RandomAvgDistInput;
    type Out = ThoTensor<ChannelsMotifsDists, SynKey>;
    type Ctx = C;
}

// #[async_trait]
impl<L, C> AsyncStage for RandomAvgDist<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="RandomAvgDist::run"))]
    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<RandomAvgDistInput>>,
        mut ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError>
    where
        for<'a> Self::In<'a>: Send + Sync,
    {
        let device_idx = ctx.device_idx();
        let cache_area = ctx.cache_area();
        let input_id = input.id().clone();

        let (max_slices, find_motifs_run_id) = with_data!(
            ctx.branch(Some("random_avg_dist_input"), false, false),
            input;
            |_ctx, input: &RandomAvgDistInput| {
                Ok::<(u32, RunId), ThoError>((input.max_slices, input.find_motifs_run_id))
            }
        )?;

        if max_slices < 2 {
            return Err(ThoError::new(format!("RandomAvgDist: max_slices must be >= 2, got {max_slices}")));
        }

        let find_motifs_run_config = FindMotifsRunConfig::from_properties_map_with_id(
            find_motifs_run_id,
            &self.store.read().list_run_config_values(find_motifs_run_id)?,
        )?;
        let motif_width = find_motifs_run_config.window;
        if motif_width == 0 {
            return Err(ThoError::new("RandomAvgDist: motif_width must be > 0"));
        }

        let resources = self.store.read().list_resources_not_in_group(RANDOM_AVG_DIST_GROUP, max_slices)?;
        let (slices, resources_map) = random_non_overlapping_slices(&resources, max_slices, motif_width)?;

        debug!("RandomAvgDist::run: slices.len={} resources_map.len={}", slices.len(), resources_map.len());

        let channels = self.ws.read().channels();
        if channels == 0 {
            return Err(ThoError::new("RandomAvgDist: workspace reports 0 channels"));
        }

        let mut avg_dists = Vec::with_capacity(channels as usize);
        for channel in 0..channels {
            let mut channel_ctx = ctx.branch(Some(&format!("random_avg_dist_channel_{channel}")), false, false);
            let (_window, motif) = load_motif_slices(
                0u32.into(),
                Arc::clone(&self.store),
                Arc::clone(&self.loader),
                None,
                &mut channel_ctx,
                find_motifs_run_id,
                slices.clone(),
                resources_map.clone(),
                channel as i64,
                device_idx,
                cache_area,
                &self.ws,
                "random_avg_dist",
            ).await?;

            let slice_count = motif.t_ref().size()[0];
            if slice_count < 2 {
                return Err(ThoError::new(format!("RandomAvgDist: not enough slices for channel {channel}, got {slice_count}")));
            }

            //TODO options for z_norm_dims and pearson_corr
            // let dist = average_euclidean_distance(&z_norm_dims(motif.t_ref(), &[1], 1e-8)?)?;
            // let dist = pearson_corr_distance_matrix_0_2_single(motif.t_ref(), 1e-8)?.mean_dim(&[0,1][..], false, Float).double_value(&[]) as f32;
            let dist = average_euclidean_distance(motif.t_ref())?;
            avg_dists.push(dist);
        }

        let out_id = &input_id + &SynKey::new([mono_key!("random_avg_dist", "stage")]);
        let avg_t = Tensor::from_slice(avg_dists.as_slice())
            .view([channels as i64, 1])
            .to_device(ThoDevice::from(device_idx).into());

        let out = ThoTensor::new(out_id, avg_t)?;
        Ok(Arc::new(DevicesWrap::wrap(out, cache_area.into())))
    }
}

fn random_non_overlapping_slices(
    resources: &[Resource],
    max_slices: u32,
    motif_width: u32,
) -> Result<(Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>), ThoError> {
    if resources.is_empty() {
        return Err(ThoError::new("RandomAvgDist: no resources available for random slices"));
    }

    let mut candidates = resources
        .iter()
        .filter(|r| r.length() >= motif_width)
        .cloned()
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        return Err(ThoError::new("RandomAvgDist: no resources long enough for motif width"));
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

        let resource = match candidates.choose(&mut rng) {
            Some(resource) => resource,
            None => break,
        };

        let max_start = match resource.length().checked_sub(motif_width) {
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
        return Err(ThoError::new(format!(
            "RandomAvgDist: expected at least 2 slices, got {}",
            slices.len()
        )));
    }

    if slices.len() < max_slices as usize {
        warn!("RandomAvgDist: requested max_slices={max_slices}, generated {}", slices.len());
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
    use parking_lot::RwLock;
    use std::sync::Arc;
    use thoosa_common::utils::lin_t;
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_shared::runner::cache_areas::CacheArea;
    use thoosa_shared::runner::common::cachable::WithDataRef;
    use thoosa_shared::runner::common::mock_run_ctx;
    use thoosa_shared::runner::common::runnable::AsyncStage;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_store::MockStoreBuilder;
    use thoosa_properties_map::ToHashmap;
    use crate::tasks::tensor_loader::mock_tensor_loader;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_random_avg_dist_basic() -> Result<(), ThoError> {
        GlobalCtx::init_devices(1);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
        let resources = vec![
            Resource::new(1, 1, "resource_a", 20),
            Resource::new(2, 1, "resource_b", 20),
        ];

        let run_id = 42;
        let motif_width = 4;
        let store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
                .with_list_resources_not_in_group([(RANDOM_AVG_DIST_GROUP, Ok(resources.clone()))])
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
                    (resources[0].clone(), lin_t(40).view((2, -1))),
                    (resources[1].clone(), lin_t(40).view((2, -1))),
                ])
                .finish(),
        );

        let stage = Arc::new(RandomAvgDist::new(Arc::clone(&store), loader, &ws));
        let ctx = Arc::new(RwLock::new(mock_run_ctx::default(0)));
        let input = Arc::new(DevicesWrap::wrap(RandomAvgDistInput::new(4, run_id), CacheArea::default()));

        let output = stage
            .run(
                input,
                ctx.write().branch(Some("random_avg_dist"), false, false),
            )
            .await?;

        let mut ctx_for_data = ctx.write().branch(Some("random_avg_dist_data"), false, false);
        let size = output
            .with_data_ref(0, &mut ctx_for_data, None, |t: &ThoTensor<ChannelsMotifsDists, SynKey>, _ctx| {
                Ok::<_, ThoError>(t.t_ref().size())
            })
            .await?;

        assert_eq!(size, vec![2, 1]);

        Ok(())
    }
}
