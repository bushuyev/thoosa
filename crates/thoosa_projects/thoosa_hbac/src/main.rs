mod data;

use std::time::Instant;
use log::{debug, error, info};
use thoosa_main::cli::run_cli;
use thoosa_main::data::parquet_to_tensor_by_path;

fn main()  {
    let t1 = Instant::now();

    unsafe {
        //TODO to kick REGISTER_CUDA_HOOKS(CUDAHooks); in
        // ATen   cuda
        //     │ │ │   detail
        //     │ │ │ │ │ 󰙲 CUDAHooks.cpp
        // let lib = libloading::Library::new("libtorch_cuda.so").expect("pytorch-dst/lib/libtorch_cuda.so");
        // let func: libloading::Symbol<unsafe extern fn() -> u32> = lib.get(b"my_func")?;
        // Ok(func())
        // println!("lib={lib:?}"); //logging is not set up yet
    };

    match run_cli(
        |base_path|data::labeled_resourses(base_path.as_ref().join("train.csv")),
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