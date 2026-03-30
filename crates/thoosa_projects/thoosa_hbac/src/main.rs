mod data;

use log::{error, info};
use std::time::Instant;
use tch::Device::Cpu;
use thoosa_common::utils::kick_register_cuda_hooks;
use thoosa_main::cli::run_cli;
use thoosa_shims::context::ThoCudaContext;
use thoosa_shared::ThoError;


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
        .worker_threads(cpu_num) //std::thread::available_parallelism().expect("available_parallelism failed").into()) // async workers
        .max_blocking_threads(cpu_num)        // more threads for spawn_blocking jobs
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
            &["seizure", "lpd", "gpd", "lrda", "grda", "other"],
            20,
            684400,
            200,
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
