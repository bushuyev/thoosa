#![allow(unused_assignments)]

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use log::{debug, info, warn};
use parking_lot::RwLock;
use tch::Device::Cpu;
use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig, FindMotifsRunConfigBuilder, FindMotifsRunConfigFields, FindMotifsRunConfigRangeOrValue, SelectMotifsRunConfig, SelectMotifsRunConfigBuilder, SelectMotifsRunConfigFields, SelectMotifsRunConfigRangeOrValue};
use thoosa_common::parzen::optimizer::{AsyncTPEOptimizer, ConfigurationSpace};
use thoosa_common::runs::{SavableResult, Simple};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::train_nets::OptConfig;
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_motifs::find::{FixedFindMotifsConfig, SelectMotifsSavableResult};
use thoosa_motifs::fragments::{DataBy, PlotDataArgs};
use thoosa_properties_map::ToHashmap;
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::runner::cache_areas::CacheArea;
use thoosa_shared::runner::common::cachable::DevicesWrap;
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage};
use thoosa_shared::runner::common::RunCtx;
use thoosa_shared::runner::run_ctx_2::RunCtxWithPermits;
use thoosa_shared::run_configs::{OptRunConfigs, RunConfig, RunId};
use thoosa_shared::traits::{Adjustable, WithId};
use thoosa_shared::val::{NamedRange, NamedVal, Val};
use thoosa_shared::{collect_futures_with_progress, mono_key, panic_to_err_async, parse_resources, ResourceType, StreamBuffering, ThoError};
use thoosa_store::Store;
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::find_motifs::FindMotifsStage;
use crate::tasks::motif_loader::MotifLoader;
use crate::tasks::select_motifs::SelectMotifsStage;
use crate::tasks::tensor_loader::RawTensorLoader;
use thoosa_shared::comp_id::SynKey;
use crate::cli::run_plot;

#[allow(non_snake_case, unused_assignments, unused_variables)]
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug)]
#[run_config_opt]
pub struct FindAndSelectMotifsRunConfig {
    #[in_map]
    pub id: Option<RunId>,

    #[in_map]
    #[run_config]
    pub find_motifs_run_config: FindMotifsRunConfig,

    #[in_map]
    #[run_config]
    pub select_motifs_run_config: SelectMotifsRunConfig,
}

#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct FindAndSelectMotifsInput {
    id: SynKey,
    pub find_resources: String,
    pub select_resources: String,
}

impl FindAndSelectMotifsInput {
    pub fn new(find_resources: String, select_resources: String) -> Self {
        Self {
            id: SynKey::new([mono_key!("FindAndSelectMotifsInput", "stage")]),
            find_resources,
            select_resources,
        }
    }
}

impl Input for FindAndSelectMotifsInput {}

impl WithId for FindAndSelectMotifsInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct FindAndSelectMotifsResult {
    best_configs: Vec<FindAndSelectMotifsRunConfig>,
}

impl FindAndSelectMotifsResult {
    pub fn new(best_configs: Vec<FindAndSelectMotifsRunConfig>) -> Self {
        Self { best_configs }
    }
}

impl Output for FindAndSelectMotifsResult {}

impl WithId for FindAndSelectMotifsResult {
    type ID = RunId;

    fn id(&self) -> &Self::ID {
        &0
    }
}

#[async_trait]
impl SavableResult for FindAndSelectMotifsResult {
    fn value(&self) -> f32 {
        0.0
    }

    async fn save(
        &mut self,
        store: &Arc<Box<dyn Store>>,
        _ws: &Arc<RwLock<Box<dyn Workspace>>>,
    ) -> Result<u32, ThoError> {

        // let best_configs_ids = self.best_configs.iter().map(|config|{
        //     store.try_write_for(thoosa_shared::TIMEOUT).expect("lock timeout").save_run_config(
        //         Some(self.best_configs.iter().map(|v|v.id.unwrap()).collect::<Vec<_>>().as_ref()),
        //         &FindAndSelectMotifsRunConfig::type_name(),
        //         "FindAndSelectMotifs",
        //         "FindAndSelectMotifs",
        //         HashMap::from([]),
        //     )
        // }).collect::<Result<Vec<_>, ThoError>>();

        //TODO it's not run_id, it's only to track best  find & select ids
        let best_run_ids = self
            .best_configs
            .iter()
            .map(|v| v.select_motifs_run_config.id.expect("no select_motifs_run_config"))
            .collect::<Vec<_>>();
        let id = store.save_run_config(
            Some(&best_run_ids),
            FindAndSelectMotifsRunConfig::type_name(),
            "OptSelectMotifsRunConfig",
            "best_run_ids",
            HashMap::from([]),
        ).await?;
        Ok(id)
    }
}


///for each grid values combo search for optimal opt values (RangeOrValue::Range without step)
#[derive(Debug)]
pub struct FindAndSelectMotifsStage {
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<Box<dyn Store>>,
    loader: Arc<RawTensorLoader<RunCtxWithPermits>>,
    run_configs: Arc<
        OptRunConfigs<
            Vec<NamedVal<FindAndSelectMotifsRunConfigFields>>,
            Vec<NamedRange<FindAndSelectMotifsRunConfigFields>>,
            Vec<Hyperparameter<FindAndSelectMotifsRunConfigFields>>,
            FindAndSelectMotifsRunConfigFields,
        >,
    >,
    fixed_find_motifs_config: FixedFindMotifsConfig,
    opt_config: Arc<OptConfig>,
    max_chunk_size: i64,
}

impl FindAndSelectMotifsStage {
    pub fn new(
        loader: Arc<RawTensorLoader<RunCtxWithPermits>>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        run_configs: OptRunConfigs<
            Vec<NamedVal<FindAndSelectMotifsRunConfigFields>>,
            Vec<NamedRange<FindAndSelectMotifsRunConfigFields>>,
            Vec<Hyperparameter<FindAndSelectMotifsRunConfigFields>>,
            FindAndSelectMotifsRunConfigFields,
        >,
        fixed_find_motifs_config: FixedFindMotifsConfig,
        opt_config: Arc<OptConfig>,
        max_chunk_size: i64,
    ) -> Self {
        Self {
            ws: Arc::clone(ws),
            store: Arc::clone(store),
            loader,
            run_configs: Arc::new(run_configs),
            fixed_find_motifs_config,
            opt_config,
            max_chunk_size,
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
            .save_run_config(input_run_ids, R::type_name(), descr, "todo", run_config.to_properties_iter(None).collect())
            .await?;
        run_config.set_id(run_id);
        Ok(run_id)
    }

    async fn run_find_and_select<C:RunCtx + Adjustable>(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        loader: Arc<RawTensorLoader<C>>,
        run_config: &mut FindAndSelectMotifsRunConfig,
        find_resources: &ResourceType,
        select_resources: &ResourceType,
        fixed_find_motifs_config: FixedFindMotifsConfig,
        max_chunk_size: i64,
        device_idx: usize,
        mut ctx: C
    ) -> Result<f64, ThoError> {
        let find_input = Arc::new(DevicesWrap::wrap(
            ResourcesConfigInput::new(Simple::new(find_resources.clone())).check(store).await?,
            CacheArea::default(),
        ));

        let select_input = Arc::new(DevicesWrap::wrap(
            ResourcesConfigInput::new(Simple::new(select_resources.clone())).check(store).await?,
            CacheArea::default(),
        ));

        let find_run_id = Self::save_run_config_with_input(
            store,
            None,
            &mut run_config.find_motifs_run_config,
            "FindAndSelectMotifs/find",
        ).await?;
        #[allow(unused_variables)]
        let select_run_id = Self::save_run_config_with_input(
            store,
            Some(&[find_run_id]),
            &mut run_config.select_motifs_run_config,
            "FindAndSelectMotifs/select",
        ).await?;

        let find_stage: Arc<FindMotifsStage<Simple, RawTensorLoader<C>, C>> = Arc::new(
            FindMotifsStage::new(
                ws,
                store,
                Arc::clone(&loader),
                vec![run_config.find_motifs_run_config.clone()],
                fixed_find_motifs_config,
                ThoDevice::from(device_idx),
                ctx.branch(Some("find_and_select_init_find"), true, false),
            ).await?,
        );
        find_stage
            .run(find_input, ctx.branch(Some("find_and_select_run_find"), true, false))
            .await?;

        #[allow(unexpected_cfgs)]
        #[cfg(tho_debug_plots)]
        run_plot(&ws, &store, &loader, &PlotDataArgs::new(DataBy::RunId{ find_motifs_run_id: run_config.find_motifs_run_config.id.unwrap(), select_motifs_run_ids: None }), &Cpu, ctx.branch(Some("plot"), false, false)).await?;

        let select_stage = Arc::new(
            SelectMotifsStage::<
                RawTensorLoader<C>,
                MotifLoader<RawTensorLoader<C>, C>,
                C,
            >::new(
                ws,
                store,
                Arc::clone(&loader),
                Arc::new(MotifLoader::new(Arc::clone(store), Arc::clone(&loader), ws)),
                vec![run_config.select_motifs_run_config.clone()],
                vec![find_run_id],
                max_chunk_size,
                #[cfg(test)]
                None,
            )
            .await?,
        );

        let select_stage_output:Arc<DevicesWrap<SelectMotifsSavableResult>> = select_stage
            .run(select_input, ctx.branch(Some("select_stage_output"), true, false))
            .await?;

        let mut select_savable:SelectMotifsSavableResult = Arc::into_inner(select_stage_output)
            .expect("FindAndSelectMotifsStage::run_find_and_select: select output still shared")
            .take_data(device_idx)
            .await?;
        let score_sum = select_savable.score_sum();
        select_savable.save(store, ws).await?;


        #[allow(unexpected_cfgs)]
        #[cfg(tho_debug_plots)]
        run_plot(&ws, &store, &loader, &PlotDataArgs::new(DataBy::SelectedMotifs { select_motifs_run_ids: vec![select_run_id] }), &Cpu, ctx.branch(Some("plot"), false, false)).await?;

        Ok(score_sum)
    }
}

impl Stage for FindAndSelectMotifsStage {
    type Ctx = RunCtxWithPermits;
    type In<'a> = FindAndSelectMotifsInput;
    type Out = FindAndSelectMotifsResult;
}

impl AsyncStage for FindAndSelectMotifsStage {
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="FindAndSelectMotifsStage::run"))]
    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        mut ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let input = Arc::into_inner(input)
            .expect("FindAndSelectMotifsStage::run: input still shared")
            .take_data(ctx.device_idx())
            .await?;

        let find_resources = parse_resources(&input.find_resources)
            .map_err(|e| ThoError::from_boxed_error("FindAndSelectMotifsStage::run parse find_resources", e))?;
        let select_resources = parse_resources(&input.select_resources)
            .map_err(|e| ThoError::from_boxed_error("FindAndSelectMotifsStage::run parse select_resources", e))?;

        let const_vals = self.run_configs.grid_run_config_source().const_params().clone();
        let grid_vals_sets = self.run_configs.grid_to_vals();
        let opt_params = self.run_configs.opt_params().clone();
        let opt_max_evals = self.opt_config.opt_max_evals;
        let opt_num_init = self.opt_config.opt_num_init;
        let mut find_and_select_run_find_ctx = ctx.branch("find_and_select_run_find".into(), true, false);
        let device_idx = ctx.device_idx();
        let grid_vals_sets_len = grid_vals_sets.len();
        info!("FindAndSelectMotifsStage::run: grid_vals_sets_len={grid_vals_sets_len:?} opt_max_evals={opt_max_evals}");

        let futures = grid_vals_sets
            .into_iter()
            .enumerate()
            .map(|(grid_idx, grid_vals)| {
                let min_loss = Arc::new(RwLock::new((f32::MAX, None::<FindAndSelectMotifsRunConfig>)));
                let min_loss_ref = Arc::clone(&min_loss);

                let const_vals = const_vals.clone();
                let opt_params = opt_params.clone();
                let ws = Arc::clone(&self.ws);
                let store = Arc::clone(&self.store);
                let loader = Arc::clone(&self.loader);
                let find_resources = find_resources.clone();
                let select_resources = select_resources.clone();

                let fixed_find_motifs_config = self.fixed_find_motifs_config.clone();
                let max_chunk_size = self.max_chunk_size;
                let mut find_and_select_run_find_ctx = find_and_select_run_find_ctx.branch(Some(format!("grid_idx_{grid_idx}").as_str()), false, false);
                async move {

                    let mut opt = AsyncTPEOptimizer::new(
                        async  |opt_vals| {
                            let mut run_config: FindAndSelectMotifsRunConfig = [const_vals.as_slice(), grid_vals.as_slice(), opt_vals].concat().into_iter().into();

                            let ws = Arc::clone(&ws);
                            let store = Arc::clone(&store);
                            let loader = Arc::clone(&loader);
                            let find_resources = find_resources.clone();
                            let select_resources = select_resources.clone();
                            let fixed_find_motifs_config = fixed_find_motifs_config.clone();
                            let min_loss_ref = Arc::clone(&min_loss_ref);

                            let selected_score_sum = panic_to_err_async!({
                                Self::run_find_and_select(
                                    &ws,
                                    &store,
                                    Arc::clone(&loader),
                                    &mut run_config,
                                    &find_resources,
                                    &select_resources,
                                    fixed_find_motifs_config,
                                    max_chunk_size,
                                    device_idx,
                                    find_and_select_run_find_ctx.branch(Some(format!("{grid_idx}_in_opt").as_str()), false, false)
                                ).await
                            })
                            .unwrap_or_else(|e| {
                                warn!("FindAndSelectMotifsStage::run: failed run_config={run_config:?}: {e:?}");
                                0f64
                            });

                            let loss = if selected_score_sum == 0.0 { f32::MAX } else { 1.0 / selected_score_sum as f32 };
                            debug!("FindAndSelectMotifsStage::run: run_config={run_config:?} selected_count={selected_score_sum} loss={loss}");

                            if loss < thoosa_shared::with_perf_log(|| min_loss_ref.try_read_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"), "FindAndSelectMotifs/run/read_min_loss").0 {
                                *thoosa_shared::with_perf_log(|| min_loss_ref.try_write_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"), "FindAndSelectMotifs/run/write_min_loss") = (loss, Some(run_config.clone()));
                            }

                            loss
                        },
                        ConfigurationSpace::new(opt_params),
                        opt_max_evals,
                        opt_num_init,
                    );

                    opt.optimize().await;

                    let guard = thoosa_shared::with_perf_log(|| min_loss.try_read_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
                        "FindAndSelectMotifs/run/read_min_loss",
                    );
                    let (best_loss, best_config) = (guard.0, guard.1.as_ref().map(|v|v.clone()));
                    if let Some(best_config) = best_config {
                        info!("FindAndSelectMotifsStage::run: best_loss={best_loss:?} best_config={best_config:?}");
                        Ok::<_, ThoError>((grid_idx, Some(best_config)))
                    } else {
                        warn!("FindAndSelectMotifsStage::run: no best config for grid_vals={grid_vals:?} best_loss={best_loss:?}");
                        Ok::<_, ThoError>((grid_idx, None))
                    }
                }
            });
        let mut indexed_best_configs = collect_futures_with_progress(
            futures,
            grid_vals_sets_len,
            crate::buffer_limits::FIND_AND_SELECT_MOTIFS_RUN_BUFFER_ORDERED_LIMIT,
            StreamBuffering::Unordered,
            "FindAndSelectMotifsStage::run",
        )
        .await;

        // indexed_best_configs.sort_by_key(|(grid_idx, _)| *grid_idx);

        let best_configs = indexed_best_configs.into_iter().filter_map(|(_, config)| config).collect::<Vec<_>>();



        Ok(Arc::new(DevicesWrap::wrap(
            FindAndSelectMotifsResult::new(best_configs),
            CacheArea::default(),
        )))
    }
}
