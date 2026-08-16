#![recursion_limit = "256"]
#![allow(deprecated)]
#![allow(unused_imports, unused_variables, unused_mut, noop_method_call)]

mod data;
mod run_all;

use crate::run_all::{run_all, run_all_config_0};
use clap::Parser;
use log::{debug, error, info};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tch::Device;
use tch::Device::{Cpu, Cuda};
use thoosa_common::ipc::cuda_process::IPC_SOCK_ENV;
use thoosa_common::ipc::cuda_wokrer::{CudaWorker, InitFn, WorkerCli};
use thoosa_common::ipc::json_ser::SyncStageIpcWrapper;
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::train_nets::InferArgs;
use thoosa_common::utils::kick_register_cuda_hooks;
use thoosa_common::ws::{Workspace, WsConfigs};
use thoosa_main::RawTensorLoader;
use thoosa_main::cli::{Cli, RunAllArgs, run_cli};
use thoosa_main::tasks::apply_motif_nets::{apply_motif_net_endpoint_init, ApplyMotifNet, APPLY_MOTIF_NET_ENDPOINT, STACK_APPLIED_MOTIF_NETS_ENDPOINT, stack_applied_motif_nets};
use thoosa_main::tasks::cross_batch_clc::{CROSS_UNFOLD_CLC_ENDPOINT, sync_cross_unfold_clc_endpoint_init};
use thoosa_main::tasks::motif_stats::{MOTIF_STATS_CONSUME_ENDPOINT, MOTIF_STATS_SYNC_ENDPOINT, motif_stats_consume_stage_endpoint_init, motif_stats_sync_stage_endpoint_init};
use thoosa_main::tasks::tensor_loader::{SYNC_RAW_TENSOR_LOADER_ENDPOINT, sync_raw_tensor_loader_endpoint_init};
use thoosa_main::tasks::train_final_net::train_final_step_endpoint_inits;
use thoosa_main::tasks::train_label_nets::{train_label_step_endpoint_inits, SYNC_LABEL_BATCH_ENDPOINT, sync_label_batch_endpoint_init};
use thoosa_main::tasks::train_motif_nets::train_motif_step_endpoint_inits;
use thoosa_main::tasks::transforms::freq_filter::{SYNC_FREQ_FILTER_ENDPOINT, sync_freq_filter_endpoint_init};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::{ThoError, cdt_for_dir};
use thoosa_shims::context::ThoCudaContext;
use thoosa_store::rusqlite::Transaction;
use thoosa_store::{SqlStore, Store};

fn cuda_worker_endpoint_inits() -> HashMap<String, InitFn> {
    let mut endpoints = train_motif_step_endpoint_inits();
    // Label and final training now execute through the same worker-owned proxy lifecycle as motif training.
    endpoints.extend(train_label_step_endpoint_inits());
    endpoints.extend(train_final_step_endpoint_inits());
    endpoints.extend([
        (
            CROSS_UNFOLD_CLC_ENDPOINT.to_string(),
            Box::new(sync_cross_unfold_clc_endpoint_init) as InitFn,
        ),
        (
            MOTIF_STATS_SYNC_ENDPOINT.to_string(),
            Box::new(motif_stats_sync_stage_endpoint_init) as InitFn,
        ),
        (
            MOTIF_STATS_CONSUME_ENDPOINT.to_string(),
            Box::new(motif_stats_consume_stage_endpoint_init) as InitFn,
        ),
        (
            SYNC_FREQ_FILTER_ENDPOINT.to_string(),
            Box::new(sync_freq_filter_endpoint_init) as InitFn,
        ),
        (
            SYNC_RAW_TENSOR_LOADER_ENDPOINT.to_string(),
            Box::new(sync_raw_tensor_loader_endpoint_init) as InitFn,
        ),
        (
            APPLY_MOTIF_NET_ENDPOINT.to_string(),
            Box::new(apply_motif_net_endpoint_init) as InitFn,
        ),
        (
            STACK_APPLIED_MOTIF_NETS_ENDPOINT.to_string(),
            Box::new(stack_applied_motif_nets) as InitFn,
        ),
        (
            SYNC_LABEL_BATCH_ENDPOINT.to_string(),
            Box::new(sync_label_batch_endpoint_init) as InitFn,
        ),
    ]);
    endpoints
}

fn main() -> Result<(), ThoError> {
    let cdt = cdt_for_dir();
    let app_configs = WsConfigs::new(20, 684400, 200);

    CudaWorker::run_if_sock(&cdt, &app_configs, cuda_worker_endpoint_inits())?;

    println!("CudaWorker::run_if_sock: running main");

    // kick_register_cuda_hooks();

    let t1 = Instant::now();

    /*
    sqlite> select count(*), lb.label from (select count(*), label_id from labeled_resources group by resource_id having sum(p)=max(p)) as cnt left join labels lb on lb.id=cnt.label_id group by label_id;
    2227|seizure
    1181|lpd
    806|gpdh
    384|lrda
    1154|grda
    4208|other
    */
    let cpu_num = std::thread::available_parallelism().map(|nz| nz.get()).unwrap_or(2);

    //Cli::parse();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .on_thread_start(|| {
            //TODO enable if not in ipc mode
            //runs cuCtxSetCurrent
            // if let Err(err) = ThoCudaContext::new(0) {
            //     panic!("Failed to create context: {err:?}"); //TODO device
            // }
        })
        .worker_threads(cpu_num)
        .max_blocking_threads(cpu_num)
        .enable_all()
        .build()
        .map_err(|err| {
            error!("main: {err:?}");
            ThoError::from_error("tokio runtime build failed", err)
        })?;


    let run_result = rt.block_on(async move {
        let cdt = cdt_for_dir();

        track_await(run_cli(
            |source_path| data::labeled_resourses(source_path, "train.csv"),
            |source_path, key| {
                thoosa_main::data::parquet_to_tensor_by_path(
                    source_path
                        .as_ref()
                        .join("train_eegs")
                        .join(format!("{}.parquet", key)),
                    Cpu,
                )
            },
            |t: &Transaction|{

                for pure_sets_len in [(3, true), (10, true), (50, true), (100, true), (150, true), (10000, false)] {
                    SqlStore::save_pure_sets(pure_sets_len.0, pure_sets_len.1, &t)?;
                }


                SqlStore::save_groups_by_label_cnt(100, &t)?;

                Ok(())
            },
            async |
                ws: &Arc<RwLock<Box<dyn Workspace>>>,
                store: &Arc<Box<dyn Store>>,
                loader: &Arc<RawTensorLoader<RunCtxImpl>>,
                root_ctx: RunCtxImpl,
                run_all_args: &RunAllArgs,
            |{

                let all_configs = run_all_config_0(run_all_args);

                track_await(run_all(ws, store, loader, root_ctx, run_all_args, all_configs)).await
            },
            &["seizure", "lpd", "gpd", "lrda", "grda", "other"],
            app_configs,
            Some(async |_params:&InferArgs, _ws:&Arc<RwLock<Box<dyn Workspace>>>, _store:&Arc<Box<dyn Store>>,|{
                debug!("bird_clef: infer");
                todo!()
            }),
            cdt
        )).await
    });

    match run_result {
        Ok(_) => {
            info!("All done in {} mls", t1.elapsed().as_millis());
            rt.shutdown_background();
            Ok(())
        }
        Err(e) => {
            error!("failed with {:?} in {} mls", e, t1.elapsed().as_millis());
            Err(e)
        }
    }
    
}
