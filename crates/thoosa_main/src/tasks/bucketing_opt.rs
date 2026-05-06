use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use log::{debug, info, warn};
use parking_lot::RwLock;
use crate::tasks::find_motifs_run_config::{AccumulateStats, Aggr, FindMotifsRunConfig, MaxAggr, SlicesFilter};
use thoosa_common::parzen::optimizer::{AsyncTPEOptimizer, ConfigurationSpace};
use thoosa_common::runner::common::cachable::DevicesWrap;
use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::runner::common::RunCtx;
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_common::runs::{SavableResult, Simple};
use thoosa_common::train_nets::OptConfig;
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_motifs::distance::{EucDistConfig, SimilarityMetric};
use thoosa_properties_map::ToHashmap;
use thoosa_run_config_args::OptRunConfigSource;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::run_configs::{ConstConfigs, GridRunConfigs, OptRunConfigs, RunConfig, RunId};
use thoosa_shared::traits::{Adjustable, Data, WithId};
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_shared::{collect_futures_with_progress, panic_to_err_async, StreamBuffering, ThoError, LOCK_TIMEOUT};
use thoosa_store::Store;
use thoosa_with_data::with_data;

use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::bucketing::{BucketRunConfig, BucketRunConfigFields, BucketRunConfigRangeOrValue, BucketingStage, MelPatchOutput, MelPatchResourcesInput};
use crate::tasks::tensor_loader::RawTensorLoader;
use thoosa_shared::comp_id::SynKey;

const DEFAULT_BUCKET_OPT_WINDOW: u32 = 1024;
const DEFAULT_BUCKET_OPT_STEP: u32 = 320;

impl OptRunConfigSource for BucketRunConfigRangeOrValue<BucketRunConfigFields> {
    type P = BucketRunConfigFields;

    fn opt_run_configs(
        &self,
    ) -> Result<OptRunConfigs<Vec<NamedVal<Self::P>>, Vec<NamedRange<Self::P>>, Vec<Hyperparameter<Self::P>>, Self::P>, ThoError> {
        let mut const_params = Vec::new();
        let mut grid_params = Vec::new();
        let mut opt_params = Vec::new();

        self.fill_run_configs(|v| v, &mut const_params, &mut grid_params, &mut opt_params);
        let has_grid_opt_params = grid_params.iter().any(|grid_param| grid_param.grid_value_sets().into_iter().any(|value_set| !value_set.opt_params().is_empty()));
        let has_pending_opt_params = grid_params.iter().any(|grid_param| grid_param.grid_value_sets().into_iter().any(|value_set| !value_set.pending_fields().is_empty()));
        if opt_params.is_empty() && !has_grid_opt_params && !has_pending_opt_params {
            return Err(ThoError::new("no opt params in run config"));
        }

        Ok(OptRunConfigs::new(GridRunConfigs::new(ConstConfigs::new(const_params), grid_params), opt_params))
    }
}

#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct MelPatchOptResult {
    best_configs: Vec<BucketRunConfig>,
}

impl MelPatchOptResult {
    pub fn new(best_configs: Vec<BucketRunConfig>) -> Self {
        Self { best_configs }
    }

    pub fn best_configs(&self) -> &[BucketRunConfig] {
        &self.best_configs
    }
}

impl Output for MelPatchOptResult {}

impl WithId for MelPatchOptResult {
    type ID = RunId;

    fn id(&self) -> &Self::ID {
        &0
    }
}

#[async_trait]
impl SavableResult for MelPatchOptResult {
    type R = Vec<u32>;

    async fn save(
        &mut self,
        store: &Arc<Box<dyn Store>>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        p: Self::P<'_>
    ) -> Result<Vec<u32>, ThoError> {
        let best_run_ids = self.best_configs.iter().map(|v| v.id.expect("no mel_patch_run_config id")).collect::<Vec<_>>();
        let input_run_ids = if best_run_ids.is_empty() { None } else { Some(best_run_ids.as_slice()) };

        store
            .save_run_config(
                input_run_ids,
                "MelPatchOptRunConfig",
                "OptMelPatchRunConfig",
                HashMap::from([]),
            )
            .await.map(|v|vec![v])
    }
}

#[derive(Debug)]
struct MelPatchOptEvalMetrics {
    motif_count: usize,
}

impl MelPatchOptEvalMetrics {
    fn failed() -> Self {
        Self { motif_count: 0 }
    }
}

#[derive(Debug)]
pub struct BucketingOptStage {
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<Box<dyn Store>>,
    loader: Arc<RawTensorLoader<RunCtxImpl>>,
    run_configs: Arc<
        OptRunConfigs<
            Vec<NamedVal<BucketRunConfigFields>>,
            Vec<NamedRange<BucketRunConfigFields>>,
            Vec<Hyperparameter<BucketRunConfigFields>>,
            BucketRunConfigFields,
        >,
    >,
    opt_config: Arc<OptConfig>,
}

impl BucketingOptStage {
    pub fn new(
        loader: Arc<RawTensorLoader<RunCtxImpl>>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        run_configs: OptRunConfigs<
            Vec<NamedVal<BucketRunConfigFields>>,
            Vec<NamedRange<BucketRunConfigFields>>,
            Vec<Hyperparameter<BucketRunConfigFields>>,
            BucketRunConfigFields,
        >,
        opt_config: Arc<OptConfig>,
    ) -> Self {
        Self {
            ws: Arc::clone(ws),
            store: Arc::clone(store),
            loader,
            run_configs: Arc::new(run_configs),
            opt_config,
        }
    }

    async fn save_run_config_with_input<R>(
        store: &Arc<Box<dyn Store>>,
        input_run_ids: Option<&[RunId]>,
        run_config: &mut R,
        descr: &str,
    ) -> Result<RunId, ThoError>
    where
        R: RunConfig + ToHashmap,
    {
        let run_id = store
            .save_run_config(input_run_ids, R::type_name(), descr, run_config.to_properties_iter(None).collect())
            .await?;
        run_config.set_id(run_id);
        Ok(run_id)
    }

    fn find_motifs_run_config_for_mel_patch(_run_config: &BucketRunConfig) -> FindMotifsRunConfig {
        FindMotifsRunConfig::new(
            None,
            DEFAULT_BUCKET_OPT_STEP,
            DEFAULT_BUCKET_OPT_WINDOW,
            None,
            SimilarityMetric::EucDist(EucDistConfig { z_normalize: true }),
            SlicesFilter::Accumulate(AccumulateStats { aggr: Aggr::Max(MaxAggr { k: 0.0 }) }),
            None,
        )
    }

    async fn run_mel_patch(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        loader: Arc<RawTensorLoader<RunCtxImpl>>,
        run_config: &mut BucketRunConfig,
        resources_config: &Simple,
        mut ctx: RunCtxImpl,
    ) -> Result<MelPatchOptEvalMetrics, ThoError> {
        let mut find_motifs_run_config = Self::find_motifs_run_config_for_mel_patch(run_config);
        let find_run_id = Self::save_run_config_with_input(
            store,
            None,
            &mut find_motifs_run_config,
            "MelPatchOpt/find_motifs_compat",
        )
        .await?;

        let run_id = Self::save_run_config_with_input(
            store,
            Some(&[find_run_id]),
            run_config,
            "MelPatchOpt/mel_patch",
        )
        .await?;

        let mel_patch_stage = Arc::new(BucketingStage::<Simple, RawTensorLoader<RunCtxImpl>, ChannelsData2, SynKey, RunCtxImpl>::new(
            ws,
            store,
            loader,
            run_config.clone(),
            find_motifs_run_config.window,
            find_motifs_run_config.step,
        )?);

        let output = mel_patch_stage
            .run(
                Arc::new(DevicesWrap::wrap(MelPatchResourcesInput::new(resources_config.clone()).check(store).await?)),
                ctx.branch(Some("mel_patch_opt_run_mel_patch"), false),
            )
            .await?;

        let motif_count = with_data!(ctx.branch(Some("mel_patch_opt_output"), false), output; |_ctx, output: &MelPatchOutput| {
            Ok::<_, ThoError>(
                output.motif_ids.1.len()
            )
        })?;

        Ok(MelPatchOptEvalMetrics { motif_count })
    }

    async fn eval_opt_config(
        const_vals: &[NamedVal<BucketRunConfigFields>],
        grid_vals: &[NamedVal<BucketRunConfigFields>],
        opt_vals: &[NamedVal<BucketRunConfigFields>],
        min_loss_ref: &Arc<RwLock<(f32, Option<BucketRunConfig>)>>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        loader: &Arc<RawTensorLoader<RunCtxImpl>>,
        resources_config: &Simple,
        grid_idx: usize,
        ctx: &mut RunCtxImpl,
    ) -> (MelPatchOptEvalMetrics, f32) {
        let mut run_config: BucketRunConfig = [const_vals, grid_vals, opt_vals].concat().into_iter().into();

        let ws = Arc::clone(ws);
        let store = Arc::clone(store);
        let loader = Arc::clone(loader);
        let resources_config = resources_config.clone();

        let eval_metrics = panic_to_err_async!({
            Self::run_mel_patch(
                &ws,
                &store,
                loader,
                &mut run_config,
                &resources_config,
                ctx.branch(Some(format!("{grid_idx}_mel_patch_opt_eval").as_str()), false),
            )
        })
        .unwrap_or_else(|e| {
            warn!("BucketingOptStage::run: failed run_config={run_config:?}: {e:?}");
            MelPatchOptEvalMetrics::failed()
        });

        let loss = if eval_metrics.motif_count == 0 { f32::MAX } else { 1.0 / eval_metrics.motif_count as f32 };
        debug!("BucketingOptStage::run: run_config={run_config:?} motif_count={} loss={loss}", eval_metrics.motif_count);

        if loss < thoosa_shared::with_perf_log(|| min_loss_ref.try_read_for(LOCK_TIMEOUT).expect("lock timeout"), "BucketingOptStage/run/read_min_loss").0 {
            *thoosa_shared::with_perf_log(|| min_loss_ref.try_write_for(LOCK_TIMEOUT).expect("lock timeout"), "BucketingOptStage/run/write_min_loss") = (loss, Some(run_config));
        }

        (eval_metrics, loss)
    }
}

impl Stage for BucketingOptStage {
    type Ctx = RunCtxImpl;
    type In<'a> = MelPatchResourcesInput<Simple>;
    type Out = MelPatchOptResult;
}

impl AsyncStage for BucketingOptStage {
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "BucketingOptStage::run"))]
    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        mut ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let resources_config = with_data!(ctx.branch(Some("mel_patch_opt_resources"), false), input; |_ctx, input: &MelPatchResourcesInput<Simple>| {
            Ok::<_, ThoError>(input.resources_config().clone())
        })?;

        let const_vals = self.run_configs.grid_run_config_source().const_params().clone();
        let grid_runs = self.run_configs.grid_to_vals_and_opt_params();
        let opt_config = self.opt_config.as_ref().clone();
        let opt_max_evals = opt_config.opt_max_evals;
        let mut mel_patch_opt_ctx = ctx.branch(Some("mel_patch_opt"), true);
        let grid_runs_len = grid_runs.len();
        info!("BucketingOptStage::run: grid_runs_len={grid_runs_len:?} opt_max_evals={opt_max_evals}");

        let futures = grid_runs.into_iter().enumerate().map(|(grid_idx, (grid_vals, opt_params))| {
            let min_loss = Arc::new(RwLock::new((f32::MAX, None::<BucketRunConfig>)));
            let min_loss_ref = Arc::clone(&min_loss);

            let const_vals = const_vals.clone();
            let ws = Arc::clone(&self.ws);
            let store = Arc::clone(&self.store);
            let loader = Arc::clone(&self.loader);
            let resources_config = resources_config.clone();
            let mut mel_patch_opt_ctx = mel_patch_opt_ctx.branch(Some(format!("grid_idx_{grid_idx}").as_str()), false);
            let opt_config = opt_config.clone();

            async move {
                let mut opt = AsyncTPEOptimizer::new(
                    async |opt_vals| {
                        Self::eval_opt_config(
                            const_vals.as_slice(),
                            grid_vals.as_slice(),
                            opt_vals,
                            &min_loss_ref,
                            &ws,
                            &store,
                            &loader,
                            &resources_config,
                            grid_idx,
                            &mut mel_patch_opt_ctx,
                        )
                        .await
                        .1
                    },
                    ConfigurationSpace::new(opt_params),
                    opt_config,
                    &format!("mel_patch_opt_{grid_idx}"),
                );

                opt.optimize().await;

                let guard = thoosa_shared::with_perf_log(|| min_loss.try_read_for(LOCK_TIMEOUT).expect("lock timeout"), "BucketingOptStage/run/read_min_loss");
                let (best_loss, best_config) = (guard.0, guard.1.as_ref().cloned());

                if let Some(best_config) = best_config {
                    info!("BucketingOptStage::run: best_loss={best_loss:?} best_config={best_config:?}");
                    Ok::<_, ThoError>((grid_idx, Some(best_config)))
                } else {
                    warn!("BucketingOptStage::run: no best config for grid_vals={grid_vals:?} best_loss={best_loss:?}");
                    Ok::<_, ThoError>((grid_idx, None))
                }
            }
        });

        let indexed_best_configs = collect_futures_with_progress(
            futures,
            grid_runs_len,
            crate::buffer_limits::MEL_PATCH_OPT_STAGE_RUN_BUFFER_UNORDERED_LIMIT,
            StreamBuffering::Unordered,
            "BucketingOptStage::run",
        )
        .await;

        let best_configs = indexed_best_configs.into_iter().filter_map(|(_, config)| config).collect::<Vec<_>>();

        Ok(Arc::new(DevicesWrap::wrap(MelPatchOptResult::new(best_configs))))
    }
}
