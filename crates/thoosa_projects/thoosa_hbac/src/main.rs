#![recursion_limit = "256"]

mod data;
mod run_all;

use std::sync::Arc;
use log::{error, info};
use std::time::Instant;
use parking_lot::RwLock;
use tch::Device::Cpu;
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_common::utils::kick_register_cuda_hooks;
use thoosa_common::ws::Workspace;
use thoosa_main::cli::{run_cli, RunAllArgs};
use thoosa_main::RawTensorLoader;
use thoosa_shims::context::ThoCudaContext;
use thoosa_shared::ThoError;
use thoosa_store::rusqlite::Transaction;
use thoosa_store::{SqlStore, Store};
use crate::run_all::{run_all, run_all_config_0};

fn main() -> Result<(), ThoError> {
    kick_register_cuda_hooks();

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
            //runs cuCtxSetCurrent
            if let Err(err) = ThoCudaContext::new(0) {
                panic!("Failed to create context: {err:?}"); //TODO device
            }
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
        run_cli(
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

                run_all(ws, store, loader, root_ctx, run_all_args, all_configs).await
            },
            &["seizure", "lpd", "gpd", "lrda", "grda", "other"],
            20,
            684400,
            200,
            None,
        )
        .await
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
