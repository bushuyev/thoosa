mod data;

use log::{error, info};
use std::time::Instant;
use tch::Device::Cpu;
use thoosa_common::utils::kick_register_cuda_hooks;
use thoosa_main::cli::run_cli;
use thoosa_shims::context::ThoCudaContext;


fn main() {
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
            ThoCudaContext::new(0).expect("Failed to create context"); //TODO device
        })
        .worker_threads(cpu_num/2) //std::thread::available_parallelism().expect("available_parallelism failed").into()) // async workers
        .max_blocking_threads(cpu_num/2)        // more threads for spawn_blocking jobs
        .enable_all()
        .build()
        .unwrap();


    rt.block_on(async move {
        match run_cli(
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
        {
            Ok(_) => {
                info!("All done in {} mls", t1.elapsed().as_millis());
            }

            Err(e) => {
                error!("failed with {:?} in {} mls", e, t1.elapsed().as_millis());
            }
        }
    });


    // std::process::exit(0x0000); //for jupyter
}