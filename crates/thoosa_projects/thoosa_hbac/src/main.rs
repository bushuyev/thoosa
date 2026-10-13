#![recursion_limit = "256"]
#![allow(deprecated)]
#![allow(unused_imports, unused_variables, unused_mut, noop_method_call)]

mod data;
mod run_all;

use anyhow::Context as _;
use crate::run_all::{run_all, run_all_config_0, run_all_config_1, RunAllSettings};
use async_trait::async_trait;
use clap::{Parser, Subcommand};
use log::{debug, error, info};
use parking_lot::RwLock;
use polars::polars_utils::itertools::Itertools;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tch::Device;
use tch::Device::{Cpu, Cuda};
use tch::nn::Path;
use thoosa_common::args::{CommonArgs, StageCommand};
use thoosa_common::commands::common_cli::run_common_cli;
use thoosa_common::ipc::cuda_process::IPC_SOCK_ENV;
use thoosa_common::ipc::cuda_wokrer::{CudaWorker, InitFn, WorkerCli};
use thoosa_common::ipc::wraps::SyncStageIpcWrapper;
use thoosa_common::nets::common_nets::OptConfig;
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::run_ctx::RunCtxImpl;
use thoosa_common::stages::freq_filter::{SYNC_FREQ_FILTER_ENDPOINT, SyncFreqFilter, SyncFreqFilterIpcInit};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::utils::kick_register_cuda_hooks;
use thoosa_common::ws::{FsWorkspace, Workspace, WsConfigs};
use thoosa_motifs::cli::{CommonCommands, ImportResourcesArgs, MotifCommands, RunAllArgs, cuda_worker_endpoint_inits, import_resources, init_loader, run_grid};
use thoosa_motifs::find::FindMotifsArgs;
use thoosa_motifs::motifs_store::MotifsStore;
use thoosa_motifs::stages::apply_label_nets::{APPLY_LABEL_NET_ENDPOINT, STACK_APPLIED_LABEL_NETS_ENDPOINT, apply_label_net_endpoint_init, stack_applied_label_nets_init};
use thoosa_motifs::stages::apply_motif_nets::{APPLY_MOTIF_NET_ENDPOINT, ApplyMotifNet, STACK_APPLIED_MOTIF_NETS_ENDPOINT, apply_motif_net_endpoint_init, stack_applied_motif_nets_init};
use thoosa_motifs::stages::cross_batch_clc::{CROSS_UNFOLD_CLC_ENDPOINT, sync_cross_unfold_clc_endpoint_init};
use thoosa_motifs::stages::cross_loader::ResourcesConfigInput;
use thoosa_motifs::stages::find_motifs_run_config::FindMotifsRunConfigRangeOrValue;
use thoosa_motifs::stages::motif_stats::{MOTIF_STATS_CONSUME_ENDPOINT, MOTIF_STATS_SYNC_ENDPOINT, motif_stats_consume_stage_endpoint_init, motif_stats_sync_stage_endpoint_init};
use thoosa_motifs::stages::train_final_net::{SYNC_LOGITS_BATCH_ENDPOINT, sync_logits_batch_init, train_final_step_endpoint_inits};
use thoosa_motifs::stages::train_label_nets::{SYNC_LABEL_BATCH_ENDPOINT, sync_label_batch_endpoint_init, train_label_step_endpoint_inits};
use thoosa_motifs::stages::train_motif_nets::train_motif_step_endpoint_inits;
use thoosa_motifs::train_nets::InferArgs;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::RunCtx;
use thoosa_shared::{ThoError, cdt_for_dir};
use thoosa_shims::context::ThoCudaContext;
use thoosa_store::rusqlite::Transaction;
use thoosa_store::{SqlStore, Store};


#[derive(clap::Subcommand, Clone, Debug)]
pub enum HbacAppCommands {
    #[command(flatten)]
    CommonCommands(CommonCommands),

    #[command(flatten)]
    Motifs(MotifCommands),

    #[command(flatten)]
    Hbac(HbacCommands),
}

#[async_trait]
impl StageCommand for HbacAppCommands {
    type Store = Arc<dyn MotifsStore>;
    type Workspace = dyn thoosa_motifs::motifs_ws::MotifsWorkspace;
    type Ctx = RunCtxImpl;

    async fn run(self, service:&Arc<dyn MotifsStore>, ws: &Arc<RwLock<Box<dyn thoosa_motifs::motifs_ws::MotifsWorkspace>>>, ctx:RunCtxImpl) -> anyhow::Result<()> {
        match self {
            HbacAppCommands::CommonCommands(c) => {
                c.run(&(Arc::clone(&service) as Arc<dyn Store>), ws, ctx).await
            }
            HbacAppCommands::Motifs(c) => {
                c.run(&(Arc::clone(&service) as Arc<dyn MotifsStore>), ws, ctx).await
            }
            HbacAppCommands::Hbac(c) => {
                c.run(&(Arc::clone(&service) as Arc<dyn MotifsStore>), ws, ctx).await
            }
        }
    }
}


#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct HbacArgs {

    #[command(flatten)]
    pub common_args: CommonArgs<HbacAppCommands>,
}

#[derive(Debug, Subcommand, Clone)]
pub enum HbacCommands {
    ImportResources(ImportResourcesArgs),
    RunAll(RunAllArgs)
}

const LABELS: [&str; 6] = ["seizure", "lpd", "gpd", "lrda", "grda", "other"];

#[async_trait]
impl StageCommand for HbacCommands {
    type Store = Arc<dyn MotifsStore>;
    type Workspace = dyn thoosa_motifs::motifs_ws::MotifsWorkspace;
    type Ctx = RunCtxImpl;

    async fn run(self, store: &Self::Store, ws: &Arc<RwLock<Box<dyn thoosa_motifs::motifs_ws::MotifsWorkspace>>>, ctx:RunCtxImpl) -> anyhow::Result<()> {
        match self {
            HbacCommands::ImportResources(import_resources_args) => {
                debug!("ImportResources2: import_resources_params={:?}",import_resources_args);
                let freq_filter = if import_resources_args.freq_filters.is_empty() {
                    None
                } else {
                    Some(Arc::new(SyncFreqFilter::<RunCtxImpl>::new_local(SyncFreqFilterIpcInit::new(
                        import_resources_args.freq_filters.clone(),
                        ws.read().channels() as i64,
                        ws.read().sample_rate() as f64,
                        0,
                        ctx.tho_device_idx(),
                    ))?))
                };
                track_await(import_resources(
                    &ws,
                    store,
                    &LABELS,
                    || data::labeled_resourses(&import_resources_args.source_path, "train.csv"),

                    | key| {
                        thoosa_motifs::data::parquet_to_tensor_by_path(
                            PathBuf::from(&import_resources_args.source_path)
                                .join("train_eegs")
                                .join(format!("{}.parquet", key)),
                            Cpu,
                        )
                    },
                    freq_filter,
                    true,
                    |t: &Transaction|{

                        for pure_sets_len in [(3, true), (10, true), (50, true), (100, true), (150, true), (10000, false)] {
                            SqlStore::save_pure_sets(pure_sets_len.0, pure_sets_len.1, &t)?;
                        }

                        SqlStore::save_groups_by_label_cnt(100, &t)?;

                        Ok(())
                    }
                )).await?;

                Ok(())
            }
            HbacCommands::RunAll(run_all_args) => {
                let loader = init_loader(ws);

                // let all_configs = run_all_config_0(&run_all_args);


                let all_configs = run_all_config_1(&run_all_args, &RunAllSettings {
                    resources_per_label: 0,
                    pure_sets_len: [(0, true), (0, true)],
                    train_resource_group: "".to_string(),
                    valid_resource_group: "".to_string(),

                    motifs_epochs : 30,
                    labels_epochs : 150,
                    final_epochs : 150,
                    train_motifs_opt_config : Arc::new(OptConfig::new(30, 10)),
                    train_labels_opt_config : Arc::new(OptConfig::new(100, 10)),
                    train_final_opt_config : Arc::new(OptConfig::new(100, 10)),
                    max_batch_size : 20,
                });

                track_await(run_all(
                    &ws,
                    &store,
                    &loader,
                    ctx,
                    &run_all_args,
                    all_configs
                )).await?;

                Ok(())
            }
        }
    }
}


fn main() -> anyhow::Result<()> {
    let cdt = cdt_for_dir();
    let app_configs = WsConfigs::new(20, 684400, 200);

    let args = HbacArgs::parse();

    CudaWorker::run_if_sock_with_workspace(&cdt, &app_configs, cuda_worker_endpoint_inits(), thoosa_motifs::motifs_ws::new_motifs_workspace)?;

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
        .context("tokio runtime build failed")?;


    let run_result = rt.block_on(async move {

        let ws = Arc::new(RwLock::new(thoosa_motifs::motifs_ws::new_motifs_workspace(
            args.common_args.ws_path.clone(),
            app_configs.clone(),
        )?));

        let store = Arc::new(SqlStore::from_path_with_tables(
            ws.read().store_path(),
            vec![
                thoosa_store::create_tables,
                thoosa_motifs::motifs_store::create_motif_tables
            ]
        ).await?);

        run_common_cli(
            args.common_args.clone(),
            Arc::clone(&store) as Arc<dyn MotifsStore>,
            Arc::clone(&ws),
            |source_path, key| {
                thoosa_motifs::data::parquet_to_tensor_by_path(
                    source_path
                        .as_ref()
                        .join("train_eegs")
                        .join(format!("{}.parquet", key)),
                    Cpu,
                )
            },
            cdt.clone()
        ).await?;

        Ok(())
    });

    match run_result {
        Ok(_) => {
            info!("All done in {} mls", t1.elapsed().as_millis());
            rt.shutdown_background();
            Ok(())
        }
        Err(e) => {
            error!("failed with {:?} in {} mls", e, t1.elapsed().as_millis());
            // A poisoned local CUDA call may never return from spawn_blocking.
            // Runtime Drop must not prevent the failed application from exiting.
            if GlobalCtx::process_failure().is_some() { rt.shutdown_background(); }
            Err(e)
        }
    }
    
}

#[cfg(test)]
mod tests {
    use crate::{HbacAppCommands, HbacArgs, HbacCommands};
    use clap::Parser;
    use thoosa_common::transforms::filters::{AverageFilterParams, InputFilter, MedianFilterParams};
    use thoosa_motifs::fragments::Resolution;

    #[test]
    fn test_import_resources_freq_filters_parse() {
        let cli = HbacArgs::parse_from([
            "thoosa_main",
            "import-resources",
            "--source-path",
            "/tmp/input",
            "--freq-filters",
            "{MedianFilter:{window:5}}",
            "--freq-filters",
            "{AverageFilter:{window:7,step:1}}",
        ]);

        match cli.common_args.command {
            Some(HbacAppCommands::Hbac(HbacCommands::ImportResources(args))) => {
                assert_eq!(
                    args.freq_filters,
                    vec![
                        InputFilter::MedianFilter(MedianFilterParams { window: 5 }),
                        InputFilter::AverageFilter(AverageFilterParams { window: 7, step: 1 }),
                    ],
                );
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn test_run_all_resolution_flag_parses_high() {
        let cli = HbacArgs::parse_from([
            "thoosa_main",
            "run-all",
            "--find-resources", "group:find",
            "--select-resources", "group:select",
            "--resolution", "high",
        ]);

        match cli.common_args.command {
            Some(HbacAppCommands::Hbac(HbacCommands::RunAll(args))) => assert_eq!(args.resolution, Resolution::High),
            other => panic!("unexpected command: {other:?}"),
        }
    }

}
