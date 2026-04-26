#![recursion_limit = "1024"] //for deep async future type - to avoid "error: queries overflow the depth limit"

use std::sync::Arc;
use crate::data::resource_to_tensor;
use crate::labels::LABELS;
use log::{debug, error, info};
use std::time::Instant;
use parking_lot::RwLock;
use tch::Device::Cpu;
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_common::utils::kick_register_cuda_hooks;
use thoosa_common::ws::Workspace;
use thoosa_main::cli::{run_cli, RunAllArgs};
use thoosa_main::RawTensorLoader;
use thoosa_shared::{collect_futures_with_progress, StreamBuffering, ThoError};
use thoosa_shims::context::ThoCudaContext;
use thoosa_store::rusqlite::Transaction;
use thoosa_store::{SqlStore, Store};
use crate::run_all::{run_all, run_all_config_0};

mod data;
mod labels;
mod run_all;

fn main() -> Result<(), ThoError>{
    debug!("bird_clef main");
    kick_register_cuda_hooks();

    let t1 = Instant::now();

    let cpu_num = std::thread::available_parallelism().map(|nz| nz.get()).unwrap_or(2);

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
        run_cli(
            |source_path| data::labeled_resourses(source_path),
            resource_to_tensor,
            |t: &Transaction|{
                SqlStore::save_all_sets(&t)?;

               //resources used in previous N-set not included, sorted by size
                SqlStore::save_leaky_data_sets("A", Some(1), 0.6, 0.2, 1.0, &t)?;
                SqlStore::save_leaky_data_sets("A", Some(3), 0.6, 0.2, 1.0, &t)?;
                SqlStore::save_leaky_data_sets("A", Some(6), 0.6, 0.2, 1.0, &t)?;

                SqlStore::save_leaky_data_sets("B", Some(3), 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("B", Some(6), 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("B", Some(10), 0.6, 0.2, 0.9, &t)?;

                SqlStore::save_leaky_data_sets("C", Some(10), 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("C", Some(20), 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("C", Some(30), 0.6, 0.2, 0.9, &t)?;

                ///shuffles randomly
                SqlStore::save_leaky_data_sets("X", None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("Y", None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("Z", None, 0.6, 0.2, 0.9, &t)?;

                Ok(())
            },
            async |
                ws: &Arc<RwLock<Box<dyn Workspace>>>,
                store: &Arc<Box<dyn Store>>,
                loader: &Arc<RawTensorLoader<RunCtxImpl>>,
                mut root_ctx: RunCtxImpl,
                run_all_args: &RunAllArgs
            |{
                debug!("bird_clef run all");
                let all_configs = run_all_config_0(run_all_args);

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

    Ok(())
}
