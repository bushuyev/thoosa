mod data;

use std::time::Instant;
use log::{debug, error, info};
use tch::Device::Cpu;
use thoosa_main::cli::run_cli;

fn main()  {
    let t1 = Instant::now();

    unsafe {
        //TODO to kick REGISTER_CUDA_HOOKS(CUDAHooks); in
        // ATen   cuda
        //     │ │ │   detail
        //     │ │ │ │ │ 󰙲 CUDAHooks.cpp
        let lib = libloading::Library::new("libtorch_cuda.so").expect("pytorch-dst/lib/libtorch_cuda.so");
        // let func: libloading::Symbol<unsafe extern fn() -> u32> = lib.get(b"my_func")?;
        // Ok(func())
        println!("lib={lib:?}"); //logging is not set up yet
    };

    /*
   sqlite> select count(*), lb.label from (select count(*), label_id from labeled_resources group by resource_id having sum(p)=max(p)) as cnt left join labels lb on lb.id=cnt.label_id group by label_id;
   2227|seizure
   1181|lpd
   806|gpd
   384|lrda
   1154|grda
   4208|other
   */
    match run_cli(
        |source_path|data::labeled_resourses(source_path.as_ref().join("train.csv")),
        |source_path, key| thoosa_main::data::parquet_to_tensor_by_path(source_path.as_ref().join("train_eegs").join(format!("{}.parquet", key)), Cpu),
        vec![
            "seizure",
            "lpd",
            "gpd",
            "lrda",
            "grda",
            "other"
        ],
        20,
        684400
    ) {

        Ok(_) => {
            info!("All done in {} mls", t1.elapsed().as_millis());
        }

        Err(e) => {
            error!("failed with {:?} in {} mls", e, t1.elapsed().as_millis());
        }
    }
    std::process::exit(0x0000);//for jupyter
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_0(){
        println!("Ok");
    }
}