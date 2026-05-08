#![recursion_limit = "1024"] //for deep async future type - to avoid "error: queries overflow the depth limit"
#![allow(deprecated)]
#![allow(unused_imports, unused_variables, unused_mut, noop_method_call)]

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use crate::data::{ogg_to_tensor, resource_to_tensor};
use crate::labels::LABELS;
use log::{debug, error, info, warn};
use std::time::Instant;
use futures::{StreamExt, TryStreamExt};
use parking_lot::RwLock;
use tch::IndexOp;
use tokio::task::JoinHandle;
use thoosa_common::dims::LabelProbability;
use thoosa_common::runner::common::{mock_run_ctx, RunCtx};
use thoosa_common::runner::common::cachable::DevicesWrap;
use thoosa_common::runner::common::runnable::AsyncStage;
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::InferArgs;
use thoosa_common::utils::{kick_register_cuda_hooks, to_vec_f32};
use thoosa_common::ws::Workspace;
use thoosa_main::cli::{run_cli, RunAllArgs};
use thoosa_main::{ChannelsData2, RawTensorLoader};
use thoosa_main::tasks::apply_label_nets::ApplyLabelNets;
use thoosa_shared::{collect_futures_with_progress, mono_key, StreamBuffering, ThoError};
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


    let mut oggs = Vec::with_capacity(600);
    for (entry_i, entry) in fs::read_dir(&params.input_dir)?.take(600).enumerate() {
        let entry = entry?;
        let path = entry.path();
        let is_ogg = path.extension().map(|e| e.eq("ogg")).unwrap_or(false);
        debug!("path: {path:?} is_file={} is_ogg={}", path.is_file(), is_ogg);

        if path.is_file() && is_ogg {
            let ogg_t = ogg_to_tensor(&path)?; //.unfold(1, 160000, 160000); //.view((12, 160000));

            debug!("path: {path:?} loaded={ogg_t:?} requires_grad={}", ogg_t.requires_grad());

            oggs.push((path, ogg_t));
        } else {
            warn!("path {path:?} is not an ogg");
        }
    }

    debug!("loaded {} ", oggs.len());
    let label_models = Arc::new(ws.read().label_models()?);

    let ctx_0 = ctx.branch("".into(), false);
    let mut next: Option<JoinHandle<Result<Arc<ApplyLabelNets<RunCtxImpl>>, ThoError>>> = Some(tokio::spawn(load_apply_label_nets_model(Arc::clone(ws), Arc::clone(store), ctx_0, Arc::clone(&label_models), &LABELS[0])));

    // let mut labels_apply_label_nets = HashMap::new();


    let mut label_results = Arc::new(RwLock::new(HashMap::new()));
    for (label_i, label) in LABELS.iter().enumerate().skip(1)/*.take(20) */{

        let apply_label_nets = next
            .take()
            .expect("prefetch task missing")
            .await??;

            // Start loading the next item immediately.
            // This runs while `stuff` is being used below.
            if label_i + 1 < 600 {
                let ctx_0 = ctx.branch("".into(), false);
                next = Some(tokio::spawn(load_apply_label_nets_model(Arc::clone(ws), Arc::clone(store), ctx_0, Arc::clone(&label_models), &label)));
            }


        let futures = futures::stream::iter(oggs.iter().enumerate().map(|(entry_i, (path, ogg_t)) | {
            let label_results = Arc::clone(&label_results);
            let apply_label_nets = Arc::clone(&apply_label_nets);
            let mut ctx = ctx.branch(Some("async"), false);
            async move {

                debug!("label={label:?} path={path:?} apply_label_nets loaded");

                let ogg_tt = Arc::new(DevicesWrap::wrap(ThoTensor::<ChannelsData2, SynKey>::new(mono_key!(entry_i, path.to_string_lossy().to_string()).into(), ogg_t.shallow_clone())?));//


                // let input = ThoTensor::new(label.into(), _ogg_t)?;
                let results = Arc::clone(&apply_label_nets).run(Arc::clone(&ogg_tt), ctx.branch(Some(&format!("ctx for {label}")), false)).await?;
                with_data!(ctx.branch(Some(&format!("ctx for {label}")), false), results; |ctx, results:&ThoTensor<LabelProbability, SynKey>| {

                    let probs = to_vec_f32(results.as_ref());
                    info!("{label_i}/{entry_i} label={label:?} path={path:?} probs={probs:?}");// slice={slice}
                    // label_results.write().enry(label.to_string(), results.as_ref().shallow_clone());

                    label_results.write()
                        .entry(label.to_string())
                        .or_insert(HashMap::new())
                        .insert(path.to_string_lossy().to_string(), probs);
                        // .or_insert(vec![])
                        // .push(results.as_ref().f_double_value(&[]).unwrap_or(0.0));

                    Ok(())
                })?;

                debug!("label={label:?} path={path:?} inferred");
                // assert!(labels_apply_label_nets.insert(label.to_string(), apply_label_nets).is_none(), "got {label} twice");

                debug!("path={path:?} done");


                Ok::<_, ThoError>(())
            }
        })) .buffer_unordered(3) // at most 3 spawned tasks are awaited/in-flight
            .map(|join_result| {
                // join_result: Result<anyhow::Result<String>, JoinError>
                match join_result {
                    Ok(inner_result) => Ok(inner_result),
                    Err(join_err) => Err(ThoError::from(join_err)),
                }
            })
            .try_collect::<Vec<_>>()
            .await?;

        info!("label {label:?} ({label_i}) done");
    }



    info!("label_results={}", label_results.read().len());
    Ok(())
}

async fn load_apply_label_nets_model(ws: Arc<RwLock<Box<dyn Workspace>>>, store: Arc<Box<dyn Store>>, ctx:  RunCtxImpl, label_models: Arc<HashMap<String, Vec<(PathBuf, PathBuf)>>>, label: &str) -> Result<Arc<ApplyLabelNets<RunCtxImpl>>, ThoError> {
    let label_model = label_models.get(&label.to_string());
    let run_ids = store.list_label_nets_run_ids(label).await?;
    debug!("label={label:?} label_model={label_model:?} run_ids={run_ids:?}");

    let apply_label_nets = Arc::new(ApplyLabelNets::<RunCtxImpl>::new(
        &ws,
        &store,
        run_ids,
        ctx,
        1000,
    ).await?);
    Ok(apply_label_nets)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use log::info;
    use parking_lot::RwLock;
    use thoosa_main::tasks::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::train_nets::InferArgs;
    use thoosa_common::ws::{FsWorkspace, WsMockBuilder};
    use thoosa_shared::perf_logging::{format_perf_map, PERF};
    use thoosa_shared::ThoError;
    use thoosa_store::{MockStoreBuilder, SqlStore};
    use crate::infer_labels;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
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

        // let params = InferArgs::new(0, "test_data/train_soundscapes".to_string(), "".to_string(), 0);
        let params = InferArgs::new(0, "/home/bu/dvl/data/kaggle/birdclef-2026/train_soundscapes".to_string(), "".to_string(), 0);

        infer_labels(&params, &ws, &store).await?;


        info!("\n{}", format_perf_map(&PERF));

        Ok(())
    }
}
