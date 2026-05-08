#![recursion_limit = "1024"] //for deep async future type - to avoid "error: queries overflow the depth limit"
#![allow(deprecated)]
#![allow(unused_imports, unused_variables, unused_mut, noop_method_call)]

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use crate::data::{ogg_to_tensor, resource_to_tensor};
use crate::labels::LABELS;
use log::{debug, error, info, warn};
use std::time::Instant;
use parking_lot::RwLock;
use tch::IndexOp;
use thoosa_common::dims::LabelProbability;
use thoosa_common::runner::common::{mock_run_ctx, RunCtx};
use thoosa_common::runner::common::cachable::DevicesWrap;
use thoosa_common::runner::common::runnable::AsyncStage;
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::InferArgs;
use thoosa_common::utils::kick_register_cuda_hooks;
use thoosa_common::ws::Workspace;
use thoosa_main::cli::{run_cli, RunAllArgs};
use thoosa_main::{ChannelsData2, RawTensorLoader};
use thoosa_main::tasks::apply_label_nets::ApplyLabelNets;
use thoosa_shared::{mono_key, ThoError};
use thoosa_shared::comp_id::SynKey;
use thoosa_shims::context::ThoCudaContext;
use thoosa_store::rusqlite::Transaction;
use thoosa_store::{LimitBy, SqlStore, Store};
use thoosa_with_data::with_data;
use crate::run_all::{run_all, run_all_2, run_all_grid_config_0};

mod data;
mod labels;
mod run_all;

fn main() -> Result<(), ThoError>{
    debug!("bird_clef main");
    kick_register_cuda_hooks();

    let t1 = Instant::now();

    let cpu_num = std::thread::available_parallelism().map(|nz| nz.get()).unwrap_or(2);

    // Navigation note: each Tokio worker pins CUDA device 0 before any stage runs.
    // CUDA init failures from BirdCLEF usually happen here or later in GlobalCtx::init_devices inside run_cli.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .on_thread_start(|| {
            //runs cuCtxSetCurrent
            if let Err(err) = ThoCudaContext::new(0) {
                panic!("Failed to create context: {err:?}"); //TODO device
            }
        })
        .worker_threads(cpu_num/2)
        .max_blocking_threads(cpu_num/2)
        .enable_all()
        .build()
        .map_err(|err| {
            error!("main: {err:?}");
            ThoError::from_error("tokio runtime build failed", err)
        })?;

    let run_result = rt.block_on(async move {
        //gpt Navigation note: run_cli is the shared thoosa entry point. The arguments below are the
        //gpt BirdCLEF adapters: label import, OGG->tensor loading, dataset-set seeding, run-all orchestration, and optional inference.
        run_cli(
            |source_path| data::labeled_resourses(source_path),
            resource_to_tensor,
            |t: &Transaction|{
                SqlStore::save_all_sets(&t)?;

                //gpt This import-resources hook seeds named resource sets. Later CLI args like --find-resources=H1000000 resolve through these generated sets.
                //resources used in previous N-set not included, sorted by size
                SqlStore::save_leaky_data_sets("A", LimitBy::Count(1), None, 0.6, 0.2, 1.0, &t)?;
                SqlStore::save_leaky_data_sets("A", LimitBy::Count(3), None, 0.6, 0.2, 1.0, &t)?;
                SqlStore::save_leaky_data_sets("A", LimitBy::Count(6), None, 0.6, 0.2, 1.0, &t)?;

                SqlStore::save_leaky_data_sets("B", LimitBy::Count(3), Some(32000), 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("B", LimitBy::Count(6), Some(32000), 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("B", LimitBy::Count(10), Some(32000), 0.6, 0.2, 0.9, &t)?;

                SqlStore::save_leaky_data_sets("C", LimitBy::Count(10), None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("C", LimitBy::Count(20), None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("C", LimitBy::Count(30), None, 0.6, 0.2, 0.9, &t)?;

                SqlStore::save_leaky_data_sets("D", LimitBy::Count(100), None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("E", LimitBy::Count(200), None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("F", LimitBy::Count(300), None, 0.6, 0.2, 0.9, &t)?;

                SqlStore::save_leaky_data_sets("G", LimitBy::Length(500_000), None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("H", LimitBy::Length(1_000_000), None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("I", LimitBy::Length(3_000_000), None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("J", LimitBy::Length(10_000_000), None, 0.6, 0.2, 0.9, &t)?;


                //shuffles randomly
                SqlStore::save_leaky_data_sets("X", LimitBy::None, None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("Y", LimitBy::None, None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("Z", LimitBy::None, None, 0.6, 0.2, 0.9, &t)?;

                Ok(())
            },
            async |
                ws: &Arc<RwLock<Box<dyn Workspace>>>,
                store: &Arc<Box<dyn Store>>,
                loader: &Arc<RawTensorLoader<RunCtxImpl>>,
                root_ctx: RunCtxImpl,
                run_all_args: &RunAllArgs
            |{
                debug!("bird_clef run all");
                let all_configs = run_all_grid_config_0(run_all_args);

                //gpt run-all starts here after run_cli has opened the workspace/store and built the RawTensorLoader over saved *.pt tensors.
                run_all(
                    ws,
                    store,
                    loader,
                    root_ctx,
                    run_all_args,
                    all_configs
                ).await?;

                Ok(())
            },
            &LABELS,
            1,
            u32::MAX,
            32000,
            Some(async |params:&InferArgs, ws:&Arc<RwLock<Box<dyn Workspace>>>, store:&Arc<Box<dyn Store>>,|{
                debug!("bird_clef: infer");
                infer_labels(params, ws, store).await

            }),
        ).await
    });


    match run_result {
        Ok(_) => {
            info!("All done in {} mls", t1.elapsed().as_millis());
        }

        Err(e) => {
            error!("failed with {:?} in {} mls", e, t1.elapsed().as_millis());
        }
    }
    std::process::exit(0x0000); //for jupyter
}


async fn infer_labels(params:&InferArgs, ws:&Arc<RwLock<Box<dyn Workspace>>>, store:&Arc<Box<dyn Store>>) -> Result<(), ThoError>{
    GlobalCtx::init_devices(1);
    let mut ctx = RunCtxImpl::new(0, "root");

    // let mut labels_apply_label_nets = HashMap::new();
    let label_models = ws.read().label_models()?;

    let mut label_results = HashMap::new();
    for entry in fs::read_dir(&params.input_dir)?.take(600) {
        let entry = entry?;
        let path = entry.path();


        let is_ogg = path.extension().map(|e| e.eq("ogg")).unwrap_or(false);
        debug!("path: {path:?} is_file={} is_ogg={}", path.is_file(), is_ogg);

        if path.is_file() && is_ogg {
            let ogg_t = ogg_to_tensor(&path)?.unfold(1, 160000, 160000).view((12, 160000));
            debug!("path: {path:?} _ogg_t.size={:?}", ogg_t.size());

            // for slice in 0..ogg_t.size()[1] {
            //     let slice_t = ogg_t.i((..1, slice, ..));
            //     debug!("path: {path:?} slice={slice} e={slice_t:?}");


                let ogg_tt = Arc::new(DevicesWrap::wrap(ThoTensor::<ChannelsData2, SynKey>::new(mono_key!(0, path.to_string_lossy().to_string()).into(), ogg_t)?));


                for label in &LABELS {
                    let label_model = label_models.get(*label);
                    let run_ids = store.list_label_nets_run_ids(label).await?;
                    debug!("label={label:?} label_model={label_model:?} run_ids={run_ids:?}");

                    let apply_label_nets = Arc::new(ApplyLabelNets::<RunCtxImpl>::new(
                        ws,
                        store,
                        run_ids,
                        ctx.branch(Some(&format!("ctx for load {label}")), false),
                        100,
                        None,
                    ).await?);

                    debug!("label={label:?} path={path:?} apply_label_nets loaded");

                    // let input = ThoTensor::new(label.into(), _ogg_t)?;
                    let results = apply_label_nets.run(Arc::clone(&ogg_tt), ctx.branch(Some(&format!("ctx for {label}")), false)).await?;
                    with_data!(ctx.branch(Some(&format!("ctx for {label}")), false), results; |ctx, results:&ThoTensor<LabelProbability, SynKey>| {
                        label_results.insert(label.to_string(), results.as_ref().shallow_clone());
                        Ok(())
                    })?;

                    debug!("label={label:?} path={path:?} inferred");
                    // assert!(labels_apply_label_nets.insert(label.to_string(), apply_label_nets).is_none(), "got {label} twice");
                    info!("path={path:?} {label:?} done");
                }
                info!("--------------------------path={path:?} done");

            // }

        } else {
            warn!("path {path:?} is not an ogg");
        }
    }

    info!("label_results={}", label_results.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use parking_lot::RwLock;
    use thoosa_main::tasks::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::train_nets::InferArgs;
    use thoosa_common::ws::{FsWorkspace, WsMockBuilder};
    use thoosa_shared::ThoError;
    use thoosa_store::{MockStoreBuilder, SqlStore};
    use crate::infer_labels;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_infer_labels() -> Result<(), ThoError> {

        //TODO
        // let ws = Arc::new(RwLock::new(
        //     WsMockBuilder::new(None, 1, 100, u32::MAX)
        //
        //     .build(),
        // ));
        //
        // let store = Arc::new(
        //     MockStoreBuilder::new()
        //
        //     .build(),
        // );

        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            "/home/bu/tmp/ws_clef",
            1,
            0,//not used
            32000,
        )?));
        let store = Arc::new(SqlStore::from_path(ws.read().store_path()).await?);

        let params = InferArgs::new(0, "test_data/train_soundscapes".to_string(), "".to_string(), 0);

        infer_labels(&params, &ws, &store).await?;

        Ok(())
    }
}
