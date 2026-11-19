//! Configuration expansion and stage execution shared by commands and projects.
use crate::motifs_store::MotifsStore;
use crate::stages::find_and_select_motifs::FindAndSelectMotifsRunConfigFields;
use crate::stages::find_motifs_run_config::FindMotifsRunConfigFields;
use log::{debug, info};
use parking_lot::RwLock;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;
use thoosa_common::resources::SavableResult;
use thoosa_run_config_args::{GridRunConfigSource, OptRunConfigSource};
use thoosa_shared::ThoError;
use thoosa_shared::hyperparameter::FieldName;
use thoosa_shared::run_configs::{RunConfig, RunConfigId};
use thoosa_shared::tracked_await::t;
use thoosa_shared::traits::AsyncStage;
use thoosa_shared::traits::Stage;
use thoosa_shared::val::NamedVal;
use thoosa_shared::{RangeOrValue, panic_to_err_async};

fn resolve_median_window<P: FieldName + PartialEq>(named_vals: &[NamedVal<P>], window_path: &P) -> RangeOrValue<u32> {
    match named_vals.iter().find(|field| field.path() == window_path) {
        None => RangeOrValue::Value(100),
        Some(value) => {
            let window = value.val().to_i64(0) as u32;
            info!("Window: window={window}");
            // RangeOrValue::Value(window)
            RangeOrValue::Range {
                range: Range { start: window / 2, end: window * 2 },
                step: None,
            }
        }
    }
}

fn resolve_median_window_grid<P: FieldName + PartialEq>(named_vals: &[NamedVal<P>], window_path: &P) -> RangeOrValue<u32> {
    match named_vals.iter().find(|field| field.path() == window_path) {
        None => RangeOrValue::Value(100),
        Some(value) => {
            let window = value.val().to_i64(0) as u32;
            info!("Window: window={window}");
            // RangeOrValue::Value(window)
            RangeOrValue::Range {
                range: Range { start: window / 2, end: window * 2 },
                step: Some((window / 4, 0)),
            }
        }
    }
}

pub fn median_window_from_find_motifs(named_vals: &[NamedVal<FindMotifsRunConfigFields>]) -> RangeOrValue<u32> {
    resolve_median_window(named_vals, &FindMotifsRunConfigFields::Window)
}

pub fn median_window_from_find_motifs_grid(named_vals: &[NamedVal<FindMotifsRunConfigFields>]) -> RangeOrValue<u32> {
    resolve_median_window_grid(named_vals, &FindMotifsRunConfigFields::Window)
}

pub fn median_window_from_find_and_select(named_vals: &[NamedVal<FindAndSelectMotifsRunConfigFields>]) -> RangeOrValue<u32> {
    resolve_median_window(named_vals, &FindAndSelectMotifsRunConfigFields::FindMotifsRunConfig(FindMotifsRunConfigFields::Window))
}

fn dyn_step<P: FieldName + PartialEq>(named_vals: &[NamedVal<P>], window_path: &P) -> RangeOrValue<u32> {
    match named_vals.iter().find(|field| field.path() == window_path) {
        None => RangeOrValue::Value(10),
        Some(value) => {
            let window = value.val().to_i64(0) as u32;
            // info!("Window: window={window}");

            if window <= 10 {
                RangeOrValue::Value(5)
            // } else if window <= 500 {
            //     RangeOrValue::Value(10)
            } else {
                RangeOrValue::Value(10)
            }
        }
    }
}

pub fn dyn_step_find_and_select(named_vals: &[NamedVal<FindAndSelectMotifsRunConfigFields>]) -> RangeOrValue<u32> {
    dyn_step(named_vals, &FindAndSelectMotifsRunConfigFields::FindMotifsRunConfig(FindMotifsRunConfigFields::Window))
}

async fn save_run_configs<R: RunConfig + thoosa_properties_map::ToHashmap>(input_run_config_ids: Option<&[RunConfigId]>, store: &Arc<dyn MotifsStore>, run_configs: &mut Vec<R>, descr: &str) -> anyhow::Result<Vec<RunConfigId>> {
    let mut run_config_ids = Vec::with_capacity(run_configs.len());
    for run_config in run_configs.iter_mut() {
        let run_config_id = t!(store.save_run_config(input_run_config_ids, &<R>::type_name(), descr, run_config.to_properties_iter(None).collect(),)).await?;
        run_config.set_id(run_config_id);
        run_config_ids.push(run_config_id);
    }
    Ok(run_config_ids)
}
pub async fn run_grid<RCRV, F, RC, ST, CtxF>(input_run_config_ids: Option<Vec<RunConfigId>>, rcrv: RCRV, input: ST::In<'static>, ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, store: &Arc<dyn MotifsStore>, f: F, ctx_f: CtxF, descr: &str) -> anyhow::Result<Vec<RunConfigId>>
where
    F: AsyncFnOnce(Vec<RC>) -> anyhow::Result<Arc<ST>>,
    ST: Stage + AsyncStage + Send + Sync, // + From<(Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, Arc<dyn MotifsStore>, Arc<RwLock<LoadRunner<Tensor>>>, Vec<RC>, FC)>,
    RC: RunConfig + TryFrom<std::vec::IntoIter<NamedVal<<RCRV as GridRunConfigSource>::P>>, Error = anyhow::Error> + thoosa_properties_map::ToHashmap,
    RCRV: GridRunConfigSource + Debug,
    for<'c> <ST as Stage>::In<'c>: Send + Sync,
    for<'a> <ST as Stage>::Out: Send + Sync + SavableResult<Store = dyn MotifsStore, Workspace = dyn crate::motifs_ws::MotifsWorkspace, P<'a>: Default>,
    CtxF: FnOnce() -> <ST as Stage>::Ctx,
{
    debug!("run_all_grid_step: rcrv={rcrv:?}");

    let grid_run_configs = rcrv.grid_run_configs()?;
    debug!("run_grid: grid_run_configs={grid_run_configs:?}");

    let mut run_configs: Vec<RC> = grid_run_configs.to_run_configs()?;

    debug!("run_grid: run_configs={run_configs:?}");

    let run_config_ids = t!(save_run_configs(input_run_config_ids.as_ref().map(|v| v.as_slice()), &store, &mut run_configs, descr,)).await?;

    let stage = t!(f(run_configs)).await?;

    info!("run_grid: stage={stage:?}");

    // let mut savable = track_await(CachedStageOutput::take_data(track_await(stage.run(input, ctx_f())).await?, 0)).await?;
    let savable_r = t!(stage.run(input, ctx_f())).await;

    info!("run_grid: savable_result={savable_r:?}");

    let mut savable = savable_r?;

    info!("run_grid: savable={savable:?}");

    t!(savable.save(store, ws, <<ST as Stage>::Out as SavableResult>::P::default())).await?;

    Ok(run_config_ids)
}

// The stage factory determines its configuration type; no separate RC parameter is needed.
pub async fn run_opt<RCRV, F, ST, CtxF>(rcrv: RCRV, input: ST::In<'static>, ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, store: &Arc<dyn MotifsStore>, init_stage: F, ctx_f: CtxF) -> anyhow::Result<Vec<RunConfigId>>
where
    F: AsyncFnOnce(thoosa_shared::run_configs::TrainingRunConfigs<<RCRV as OptRunConfigSource>::P>) -> anyhow::Result<Arc<ST>>,
    ST: Stage + AsyncStage + Send + Sync, // + thoosa_common::runner::common::runnable::WrappedRunInputStage,
    RCRV: OptRunConfigSource,
    for<'c> <ST as Stage>::In<'c>: Send + Sync,
    for<'a> <ST as Stage>::Out: Send + Sync + SavableResult<Store = dyn MotifsStore, Workspace = dyn crate::motifs_ws::MotifsWorkspace, P<'a>: Default, R = Vec<RunConfigId>>,
    CtxF: FnOnce() -> <ST as Stage>::Ctx,
{
    let opt_run_configs = rcrv.opt_run_configs()?;
    debug!("run_all_grid_step: run_opt={opt_run_configs:?}");

    let stage = t!(init_stage(opt_run_configs)).await?;

    let mut savable = t!(stage.run(input, ctx_f())).await?;

    let run_config_id = panic_to_err_async!(savable.save(store, ws, <<ST as Stage>::Out as SavableResult>::P::default()))?;

    Ok(run_config_id)
}
