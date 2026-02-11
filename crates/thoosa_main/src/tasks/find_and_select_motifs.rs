use std::collections::HashMap;
use std::sync::Arc;
use log::{debug, warn};
use parking_lot::RwLock;
use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig, FindMotifsRunConfigBuilder, FindMotifsRunConfigFields, FindMotifsRunConfigRangeOrValue, SelectMotifsRunConfig, SelectMotifsRunConfigBuilder, SelectMotifsRunConfigFields, SelectMotifsRunConfigRangeOrValue};
use thoosa_common::parzen::optimizer::{AsyncTPEOptimizer, ConfigurationSpace};
use thoosa_common::runs::{SavableResult, Simple};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::train_nets::OptConfig;
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_motifs::find::FixedFindMotifsConfig;
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
use thoosa_shared::traits::WithId;
use thoosa_shared::val::{NamedRange, NamedVal, Val};
use thoosa_shared::{mono_key, panic_to_err_async, parse_resources, ResourceType, ThoError};
use thoosa_store::Store;
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::find_motifs::FindMotifsStage;
use crate::tasks::motif_loader::MotifLoader;
use crate::tasks::select_motifs::SelectMotifsStage;
use crate::tasks::tensor_loader::RawTensorLoader;
use thoosa_shared::comp_id::SynKey;

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
    id: RunId,
    best_loss: f32,
}

impl FindAndSelectMotifsResult {
    pub fn new(best_loss: f32) -> Self {
        Self { id: 0, best_loss }
    }
}

impl Output for FindAndSelectMotifsResult {}

impl WithId for FindAndSelectMotifsResult {
    type ID = RunId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl SavableResult for FindAndSelectMotifsResult {
    fn value(&self) -> f32 {
        self.best_loss
    }

    fn save(
        &mut self,
        store: &Arc<RwLock<Box<dyn Store>>>,
        _ws: &Arc<RwLock<Box<dyn Workspace>>>,
    ) -> Result<u32, ThoError> {
        self.id = store.write().save_run_config(
            None,
            &FindAndSelectMotifsRunConfig::type_name(),
            "FindAndSelectMotifsResult",
            "best_loss",
            HashMap::from([("best_loss".to_string(), Val::Real(self.best_loss as f64))]),
        )?;
        Ok(self.id)
    }
}

#[derive(Debug)]
pub struct FindAndSelectMotifsStage {
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<RwLock<Box<dyn Store>>>,
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
        store: &Arc<RwLock<Box<dyn Store>>>,
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

    fn save_run_config_with_input<R>(
        store: &Arc<RwLock<Box<dyn Store>>>,
        input_run_ids: Option<&[RunId]>,
        run_config: &mut R,
        descr: &str,
    ) -> Result<RunId, ThoError>
    where
        R: RunConfig + ToHashmap,
    {
        let run_id = store.write().save_run_config(
            input_run_ids,
            &R::type_name(),
            descr,
            "todo",
            run_config.to_properties_iter(None).collect(),
        )?;
        run_config.set_id(run_id);
        Ok(run_id)
    }

    async fn run_find_and_select(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        loader: Arc<RawTensorLoader<RunCtxWithPermits>>,
        mut run_config: FindAndSelectMotifsRunConfig,
        find_resources: &ResourceType,
        select_resources: &ResourceType,
        fixed_find_motifs_config: FixedFindMotifsConfig,
        max_chunk_size: i64,
        device_idx: usize,
    ) -> Result<usize, ThoError> {
        let find_input = Arc::new(DevicesWrap::wrap(
            ResourcesConfigInput::new(Simple::new(find_resources.clone())).check(store)?,
            CacheArea::default(),
        ));

        let select_input = Arc::new(DevicesWrap::wrap(
            ResourcesConfigInput::new(Simple::new(select_resources.clone())).check(store)?,
            CacheArea::default(),
        ));

        let find_run_id = Self::save_run_config_with_input(
            store,
            None,
            &mut run_config.find_motifs_run_config,
            "FindAndSelectMotifs/find",
        )?;
        let select_run_id = Self::save_run_config_with_input(
            store,
            Some(&[find_run_id]),
            &mut run_config.select_motifs_run_config,
            "FindAndSelectMotifs/select",
        )?;

        let mut init_find_ctx = RunCtxWithPermits::new(device_idx, "find_and_select_init");
        let find_stage: Arc<FindMotifsStage<Simple, RawTensorLoader<RunCtxWithPermits>, RunCtxWithPermits>> = Arc::new(
            FindMotifsStage::new(
                ws,
                store,
                Arc::clone(&loader),
                vec![run_config.find_motifs_run_config.clone()],
                fixed_find_motifs_config,
                ThoDevice::from(device_idx),
                init_find_ctx.branch(Some("find_and_select_init_find"), true, false),
            ).await?,
        );
        find_stage
            .run(find_input, RunCtxWithPermits::new(device_idx, "find_and_select_run_find"))
            .await?;

        let select_stage = Arc::new(
            SelectMotifsStage::<
                RawTensorLoader<RunCtxWithPermits>,
                MotifLoader<RawTensorLoader<RunCtxWithPermits>, RunCtxWithPermits>,
                RunCtxWithPermits,
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
            ),
        );

        let select_stage_output = select_stage
            .run(select_input, RunCtxWithPermits::new(device_idx, "find_and_select_run_select"))
            .await?;

        let mut select_savable = Arc::into_inner(select_stage_output)
            .expect("FindAndSelectMotifsStage::run_find_and_select: select output still shared")
            .take_data(device_idx)
            .await?;
        select_savable.save(store, ws)?;

        let selected_count = store
            .read()
            .list_selected_motifs(select_run_id, ws.read().channels())?
            .into_iter()
            .map(|channel_selected_motifs| channel_selected_motifs.len())
            .sum();

        Ok(selected_count)
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
        ctx: Self::Ctx,
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
        let min_loss = Arc::new(RwLock::new(f32::MAX));
        let device_idx = ctx.device_idx();

        for grid_vals in grid_vals_sets {
            let const_vals = const_vals.clone();
            let grid_vals = grid_vals.clone();
            let ws = Arc::clone(&self.ws);
            let store = Arc::clone(&self.store);
            let loader = Arc::clone(&self.loader);
            let find_resources = find_resources.clone();
            let select_resources = select_resources.clone();
            let min_loss_ref = Arc::clone(&min_loss);
            let fixed_find_motifs_config = self.fixed_find_motifs_config.clone();
            let max_chunk_size = self.max_chunk_size;

            let mut opt = AsyncTPEOptimizer::new(
                async |opt_vals| {
                    let run_config: FindAndSelectMotifsRunConfig = [const_vals.as_slice(), grid_vals.as_slice(), opt_vals].concat().into_iter().into();
                    let run_config_dbg = run_config.clone();
                    let ws = Arc::clone(&ws);
                    let store = Arc::clone(&store);
                    let loader = Arc::clone(&loader);
                    let find_resources = find_resources.clone();
                    let select_resources = select_resources.clone();
                    let fixed_find_motifs_config = fixed_find_motifs_config.clone();
                    let min_loss_ref = Arc::clone(&min_loss_ref);

                    let selected_count = panic_to_err_async!({
                        Self::run_find_and_select(
                            &ws,
                            &store,
                            Arc::clone(&loader),
                            run_config,
                            &find_resources,
                            &select_resources,
                            fixed_find_motifs_config,
                            max_chunk_size,
                            device_idx,
                        ).await
                    })
                    .unwrap_or_else(|e| {
                        warn!("FindAndSelectMotifsStage::run: failed run_config={run_config_dbg:?}: {e:?}");
                        0usize
                    });

                    let loss = if selected_count == 0 { f32::MAX } else { 1.0 / selected_count as f32 };

                    if loss < *min_loss_ref.read() {
                        *min_loss_ref.write() = loss;
                    }

                    debug!("FindAndSelectMotifsStage::run: run_config={run_config_dbg:?} selected_count={selected_count} loss={loss}");
                    loss
                },
                ConfigurationSpace::new(self.run_configs.opt_params().clone()),
                self.opt_config.opt_max_evals,
                self.opt_config.opt_num_init,
            );

            opt.optimize().await;
        }

        let best_loss = *min_loss.read();
        debug!("FindAndSelectMotifsStage::run: best_loss={best_loss}");

        Ok(Arc::new(DevicesWrap::wrap(
            FindAndSelectMotifsResult::new(best_loss),
            CacheArea::default(),
        )))
    }
}
