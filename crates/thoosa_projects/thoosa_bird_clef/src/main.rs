use crate::data::resource_to_tensor;
use crate::labels::LABELS;
use log::{error, info};
use std::time::Instant;
use tch::Device::Cpu;
use thoosa_common::utils::kick_register_cuda_hooks;
use thoosa_main::cli::run_cli;
use thoosa_shared::ThoError;
use thoosa_shims::context::ThoCudaContext;

mod data;
mod labels;

#[tokio::main(flavor = "multi_thread", worker_threads = 32)]
async fn main() -> Result<(), ThoError>{
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
            |source_path| data::labeled_resourses(source_path.as_ref().join("train.csv")),
            resource_to_tensor,
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
