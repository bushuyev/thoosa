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
use thoosa_store::Store;
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
            |source_path| data::labeled_resourses(source_path),
            resource_to_tensor,
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
        )
            .await
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

