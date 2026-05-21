use crate::buffer_limits::SELECT_MOTIFS_STAGE_RUN_CONFIGS;
use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::find_motifs::config_to_stage;
use crate::tasks::find_motifs_run_config::{FindMotifsRunConfig, SelectMotifsRunConfig, SelectMotifsRunResult, SelectMotifsSavableResult};
use crate::tasks::motif_loader::{MotifLoaderInput, MotifMeta};
use crate::tasks::select_motifs::{SelectMotifsInput, SelectMotifsOutputMode};
use log::{debug, info, warn};
use parking_lot::RwLock;
use serde::Deserialize;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use tch::{Kind, TchError, Tensor};
use thoosa_common::dims::MotifSlices;
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::runner::common::{OutId, RunCtx};
use thoosa_common::runs::Simple;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::utils::load_run_config;
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_shared::run_configs::{RunConfig, RunConfigId};
use thoosa_shared::traits::{Adjustable, Data, WithId};
use thoosa_shared::{collect_futures_with_progress, mono_key, StreamBuffering, ThoError};
use thoosa_store::Store;
use crate::tasks::stage_ids::SELECT_BUCKETS;

const SELECT_BUCKETS_MOTIFS_BUFFER_LIMIT: usize = 16;

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, DummyData)]
#[run_config_grid]
pub struct SelectBucketsRunConfig {
    #[in_map]
    pub id: Option<RunConfigId>,

    #[in_map]
    #[run_config]
    pub energy_k: f64,
}

impl SelectBucketsRunConfig {
    pub fn new(id: Option<RunConfigId>, energy_k: f64) -> Self {
        Self { id, energy_k }
    }

    fn to_select_motifs_run_config(&self) -> SelectMotifsRunConfig {
        SelectMotifsRunConfig::new_with_fields(self.id, None, vec![])
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ReferenceStats {
    mean: f64,
    std_population: f64,
    std_sample: f64,
    mean_energy: f64,
    rms: f64,
    count: f64,
}

#[derive(Debug, Default)]
pub struct RunningStats {
    n: f64,
    mean: f64,
    m2: f64,
}

impl RunningStats {
    pub fn update(&mut self, t: &Tensor) -> Result<(), TchError> {
        let n_b = t.numel() as f64;
        if n_b == 0.0 {
            return Ok(());
        }

        let td = t.f_to_kind(Kind::Double)?;
        let mean_b = td.f_mean(Kind::Double)?.double_value(&[]);
        let centered = td.f_sub_scalar(mean_b)?;
        let m2_b = centered.f_mul(&centered)?.f_sum(Kind::Double)?.double_value(&[]);

        if self.n == 0.0 {
            self.n = n_b;
            self.mean = mean_b;
            self.m2 = m2_b;
            return Ok(());
        }

        let delta = mean_b - self.mean;
        let n_new = self.n + n_b;

        self.mean += delta * n_b / n_new;
        self.m2 += m2_b + delta * delta * self.n * n_b / n_new;
        self.n = n_new;

        Ok(())
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn std_population(&self) -> f64 {
        if self.n == 0.0 {
            f64::NAN
        } else {
            (self.m2 / self.n).sqrt()
        }
    }

    pub fn std_sample(&self) -> f64 {
        if self.n <= 1.0 {
            f64::NAN
        } else {
            (self.m2 / (self.n - 1.0)).sqrt()
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct RunningEnergy {
    n: f64,
    sum_sq: f64,
}

impl RunningEnergy {
    pub fn update(&mut self, t: &Tensor) -> Result<(), TchError> {
        let n = t.numel() as f64;
        if n == 0.0 {
            return Ok(());
        }

        let td = t.f_to_kind(Kind::Double)?;
        let energy = td.f_mul(&td)?.f_sum(Kind::Double)?.double_value(&[]);

        self.n += n;
        self.sum_sq += energy;

        Ok(())
    }

    pub fn total_energy(&self) -> f64 {
        self.sum_sq
    }

    pub fn mean_energy(&self) -> f64 {
        if self.n == 0.0 {
            f64::NAN
        } else {
            self.sum_sq / self.n
        }
    }

    pub fn rms(&self) -> f64 {
        self.mean_energy().sqrt()
    }

    pub fn count(&self) -> f64 {
        self.n
    }
}

#[derive(Debug, Clone)]
struct BucketEvaluation {
    motif_id: u32,
    energy: f64,
    score: f64,
    selected: bool,
}

pub struct SelectBucketsStage<L, ML, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    store: Arc<Box<dyn Store>>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    loader: Arc<L>,
    motif_loader: Arc<ML>,
    run_configs: Vec<SelectBucketsRunConfig>,
    find_motifs_run_config_ids: Vec<RunConfigId>,
    find_motif_run_config_ids_motif_ids: HashMap<RunConfigId, Vec<u32>>,
    pd: PhantomData<C>,
}

impl<L, ML, C> Debug for SelectBucketsStage<L, ML, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectBucketsStage")
            .field("id", &self.id)
            .field("run_configs", &self.run_configs)
            .field("find_motifs_run_config_ids", &self.find_motifs_run_config_ids)
            .finish()
    }
}

impl<L, ML, C> WithId for SelectBucketsStage<L, ML, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<L, ML, C> SelectBucketsStage<L, ML, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    pub async fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        loader: Arc<L>,
        motif_loader: Arc<ML>,
        run_configs: Vec<SelectBucketsRunConfig>,
        find_motifs_run_config_ids: Vec<RunConfigId>,
    ) -> Result<Self, ThoError> {
        let mut find_motif_run_config_ids_motif_ids = HashMap::new();
        for find_motifs_run_config_id in &find_motifs_run_config_ids {
            find_motif_run_config_ids_motif_ids.insert(*find_motifs_run_config_id, store.list_motifs(*find_motifs_run_config_id).await?);
        }

        Ok(Self {
            id: SynKey::new([mono_key!(SELECT_BUCKETS, "SelectBucketsStage")]),
            store: Arc::clone(store),
            ws: Arc::clone(ws),
            loader,
            motif_loader,
            run_configs,
            find_motifs_run_config_ids,
            find_motif_run_config_ids_motif_ids,
            pd: PhantomData,
        })
    }

    async fn load_resources_for_selection(&self, input: &Arc<DevicesWrap<SelectMotifsInput>>, mut ctx: C) -> Result<(Arc<Vec<Resource>>, SelectMotifsOutputMode), ThoError> {
        let (resources_config, output_mode) = input
            .with_data_ref(0, ctx.branch(Some("select_buckets_input"), false), None, |input: &SelectMotifsInput, _ctx| {
                Ok::<_, ThoError>((input.resources_input().resources_config().clone(), input.output_mode()))
            })
            .await?;
        Ok((Arc::new(resources_config.to_resources(&self.store).await?.into_iter().collect::<Vec<_>>()), output_mode))
    }

    async fn reference_stats_for_find_run(&self, resources: &[Resource], find_motifs_run_config: &FindMotifsRunConfig, mut ctx: C) -> Result<ReferenceStats, ThoError> {
        let configured_loader = config_to_stage(&self.loader, find_motifs_run_config, &self.ws)?;
        let mut stats = RunningStats::default();
        let mut energy = RunningEnergy::default();

        for resource in resources {
            // TODO(perf): load reference tensors with bounded concurrency and merge per-worker RunningStats/RunningEnergy; the current sequential pass repeats for every select config.
            let t = Arc::clone(&configured_loader)
                .run_cached(
                    Arc::new(DevicesWrap::wrap(resource.clone())),
                    ctx.branch(Some("select_buckets_reference_resource_loader"), false),
                )
                .await?;
            t.with_data_ref(0, ctx.branch(Some("select_buckets_reference_stats"), false), None, |t: &ThoTensor<ChannelsData2, SynKey>, _ctx| {
                stats.update(t.t_ref())?;
                energy.update(t.t_ref())?;
                Ok::<_, ThoError>(())
            })
            .await?;
        }

        Ok(ReferenceStats {
            mean: stats.mean(),
            std_population: stats.std_population(),
            std_sample: stats.std_sample(),
            mean_energy: energy.mean_energy(),
            rms: energy.rms(),
            count: energy.count(),
        })
    }

    async fn evaluate_motif(&self, motif_id: u32, select_config: &SelectBucketsRunConfig, reference: ReferenceStats, mut ctx: C) -> Result<BucketEvaluation, ThoError> {
        // TODO(perf): persist motif/bucket energy as metadata when motifs are created so selection configs do not reload every motif tensor just to recompute energy.
        let motif = Arc::clone(&self.motif_loader)
            .run_cached(
                Arc::new(DevicesWrap::wrap(MotifLoaderInput::find_motif_slices_only((motif_id, "motif_id").into()))),
                ctx.branch(Some("select_buckets_motif_loader"), false),
            )
            .await?;

        let energy = motif
            .with_data_ref(0, ctx.branch(Some("select_buckets_motif_energy"), false), None, |motif: &DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, _ctx| {
                let mut energy = RunningEnergy::default();
                energy.update(motif.data().t_ref())?;
                Ok::<_, ThoError>(energy.mean_energy())
            })
            .await?;

        let score = energy / reference.mean_energy;
        Ok(BucketEvaluation {
            motif_id,
            energy,
            score,
            selected: energy > reference.mean_energy * select_config.energy_k,
        })
    }

    async fn select_for_run_config(
        &self,
        select_config: &SelectBucketsRunConfig,
        resources: Arc<Vec<Resource>>,
        output_mode: SelectMotifsOutputMode,
        mut ctx: C,
    ) -> Result<SelectMotifsRunResult, ThoError> {
        let mut selected = Vec::new();
        let mut metrics = HashMap::new();

        for find_motifs_run_config_id in &self.find_motifs_run_config_ids {
            let find_motifs_run_config: FindMotifsRunConfig = load_run_config(&self.store, *find_motifs_run_config_id).await?;
            // TODO(perf): cache ReferenceStats by (find_motifs_run_config_id, resources set); energy_k changes do not require recomputing the reference distribution.
            let reference = self
                .reference_stats_for_find_run(resources.as_ref(), &find_motifs_run_config, ctx.branch(Some("select_buckets_reference_stats_for_find_run"), false))
                .await?;
            debug!("SelectBucketsStage::select_for_run_config: find_motifs_run_config_id={find_motifs_run_config_id} reference={reference:?} select_config={select_config:?}");

            let motif_ids = self
                .find_motif_run_config_ids_motif_ids
                .get(find_motifs_run_config_id)
                .cloned()
                .unwrap_or_default();
            let motif_ids_len = motif_ids.len();
            let futures = motif_ids.into_iter().map(|motif_id| {
                let self_for_fut = self;
                let select_config = select_config.clone();
                let mut ctx = ctx.branch(Some("select_buckets_evaluate_motif"), false);
                async move {
                    self_for_fut
                        .evaluate_motif(motif_id, &select_config, reference, ctx.branch(Some("select_buckets_evaluate_motif_inner"), false))
                        .await
                }
            });

            let evaluations = collect_futures_with_progress(
                futures,
                motif_ids_len,
                SELECT_BUCKETS_MOTIFS_BUFFER_LIMIT,
                StreamBuffering::Unordered,
                "SelectBucketsStage::select_for_run_config",
            )
            .await;
            if evaluations.len() != motif_ids_len {
                return Err(ThoError::new(format!("SelectBucketsStage::select_for_run_config: evaluated {} of {motif_ids_len} motifs", evaluations.len())));
            }

            let mut bad_motifs = vec![];
            for evaluation in evaluations {
                debug!("SelectBucketsStage::select_for_run_config: motif_id={} energy={} score={} selected={}", evaluation.motif_id, evaluation.energy, evaluation.score, evaluation.selected);
                if evaluation.selected {
                    selected.push((evaluation.motif_id, evaluation.score));
                } else {
                    bad_motifs.push((evaluation.motif_id, evaluation.score));
                }
            }
            info!("SelectBucketsStage::select_for_run_config: bad_motifs={bad_motifs:?}");
            info!("SelectBucketsStage::select_for_run_config: selected={selected:?}");

            if output_mode == SelectMotifsOutputMode::WithConstraintMetrics {
                metrics.insert(format!("find_motifs_run_{find_motifs_run_config_id}_ref_mean"), reference.mean as f32);
                metrics.insert(format!("find_motifs_run_{find_motifs_run_config_id}_ref_std_population"), reference.std_population as f32);
                metrics.insert(format!("find_motifs_run_{find_motifs_run_config_id}_ref_std_sample"), reference.std_sample as f32);
                metrics.insert(format!("find_motifs_run_{find_motifs_run_config_id}_ref_energy"), reference.mean_energy as f32);
                metrics.insert(format!("find_motifs_run_{find_motifs_run_config_id}_ref_rms"), reference.rms as f32);
                metrics.insert(format!("find_motifs_run_{find_motifs_run_config_id}_ref_count"), reference.count as f32);
            }
        }

        Ok(SelectMotifsRunResult::new(
            select_config.to_select_motifs_run_config(),
            selected,
            (output_mode == SelectMotifsOutputMode::WithConstraintMetrics).then_some(metrics),
        ))
    }
}

impl<L, ML, C> Stage for SelectBucketsStage<L, ML, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = SelectMotifsInput;
    type Out = SelectMotifsSavableResult;
}

impl<L, ML, C> AsyncStage for SelectBucketsStage<L, ML, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "SelectBucketsStage::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        debug!("SelectBucketsStage::run: input={input:?}");

        let (resources, output_mode) = self.load_resources_for_selection(&input, ctx.branch(Some("select_buckets_load_resources_for_selection"), false)).await?;
        if resources.is_empty() {
            warn!("SelectBucketsStage::run: no resources");
            return Err(ThoError::new(format!("no resources found in {input:?}")));
        }

        let run_configs_len = self.run_configs.len();
        let futures = self.run_configs.iter().map(|select_config| {
            let self_for_fut = Arc::clone(&self);
            let resources = Arc::clone(&resources);
            let mut ctx = ctx.branch(Some("select_buckets_run_config"), false);
            async move {
                self_for_fut
                    .select_for_run_config(select_config, resources, output_mode, ctx.branch(Some("select_buckets_select_for_run_config"), false))
                    .await
            }
        });

        let run_results = collect_futures_with_progress(
            futures,
            run_configs_len,
            SELECT_MOTIFS_STAGE_RUN_CONFIGS,
            StreamBuffering::Unordered,
            "SelectBucketsStage::run",
        )
        .await;
        if run_results.len() != run_configs_len {
            return Err(ThoError::new(format!("SelectBucketsStage::run: completed {} of {run_configs_len} run configs", run_results.len())));
        }

        info!("SelectBucketsStage::run: selected motifs for {} run configs", run_results.len());
        Ok(Arc::new(DevicesWrap::wrap(SelectMotifsSavableResult::new(run_results))))
    }
}

#[cfg(test)]
mod tests {
    use super::{SelectBucketsStage, SelectBucketsRunConfig};
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::find_motifs_run_config::FindMotifsRunConfig;
    use crate::tasks::motif_loader::{mock_motif_loader, MotifLoaderInput, MotifMeta};
    use crate::tasks::select_motifs::SelectMotifsInput;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tch::{Kind, Tensor};
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::runner::common::mock_run_ctx;
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::runs::Simple;
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::{RunConfig, RunConfigId};
    use thoosa_shared::ResourceType::NotInGroup;
    use thoosa_shared::{mono_key, ThoError};
    use thoosa_store::{MockStoreBuilder, RunConfigRow};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_select_buckets_stage_selects_motifs_over_reference_energy() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));
        let resource_a = Resource::new(1, 1, "resource_a", 4);
        let resource_b = Resource::new(2, 1, "resource_b", 4);
        let resources_group = "group";
        let find_motifs_run_config_id = 11;
        let select_buckets_run_config_id = 22;
        let low_motif_id = 101;
        let high_motif_id = 102;

        let store = Arc::new(
            MockStoreBuilder::new()
                .with_run_config_descr([find_motifs_run_config_descr(find_motifs_run_config_id)])
                .with_list_run_config_values(
                    find_motifs_run_config_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_config_id), 1, 4, None, None).to_properties_iter(None).collect()),
                )
                .with_list_motifs([(find_motifs_run_config_id, Ok(vec![low_motif_id, high_motif_id]))])
                .with_list_resources_not_in_group([(resources_group, Ok(vec![resource_a.clone(), resource_b.clone()]))])
                .build(),
        );

        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached([
                    (resource_a.clone(), Tensor::f_ones([1, 4], (Kind::Float, tch::Device::Cpu))?),
                    (resource_b.clone(), Tensor::f_full([1, 4], 2.0, (Kind::Float, tch::Device::Cpu))?),
                ])
                .finish(),
        );

        let motif_loader = Arc::new(
            mock_motif_loader::build()
                .with_run_cached([
                    (
                        MotifLoaderInput::find_motif_slices_only((low_motif_id, "motif_id").into()),
                        (MotifMeta { channel: 0, find_motifs_run_config_id, window: 4 }, Tensor::f_full([1, 4], 2.0, (Kind::Float, tch::Device::Cpu))?),
                    ),
                    (
                        MotifLoaderInput::find_motif_slices_only((high_motif_id, "motif_id").into()),
                        (MotifMeta { channel: 0, find_motifs_run_config_id, window: 4 }, Tensor::f_full([1, 4], 3.0, (Kind::Float, tch::Device::Cpu))?),
                    ),
                ])
                .finish(),
        );

        let stage = Arc::new(
            SelectBucketsStage::new(
                &ws,
                &store,
                loader,
                motif_loader,
                vec![SelectBucketsRunConfig::new(Some(select_buckets_run_config_id), 2.0)],
                vec![find_motifs_run_config_id],
            )
            .await?,
        );

        let input = Arc::new(DevicesWrap::wrap(SelectMotifsInput::standalone(ResourcesConfigInput::new(Simple::new(NotInGroup(resources_group.into()))))));
        let result = stage.run(input, mock_run_ctx::default(0)).await?;
        let result = Arc::into_inner(result).expect("stage output should not be shared").take_data(0).await?;

        assert_eq!(result.selected_motif_ids_scores(), &[(high_motif_id, 3.6)]);

        Ok(())
    }

    fn find_motifs_run_config_descr(run_config_id: RunConfigId) -> (RunConfigId, Result<RunConfigRow, ThoError>) {
        (
            run_config_id,
            Ok(RunConfigRow {
                id: run_config_id,
                r#type: FindMotifsRunConfig::type_name().to_string(),
                descr: "test find motifs".to_string(),
                run_on: "2026-01-01 00:00:00".to_string(),
            }),
        )
    }
}
