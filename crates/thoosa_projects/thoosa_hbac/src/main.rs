mod data;

use std::time::Instant;
use log::{error, info};
use thoosa_main::*;
use thoosa_main::cli::run_cli;
use thoosa_main::data::parquet_to_tensor_by_path;

fn main()  {
    let t1 = Instant::now();

    match run_cli(
        |base_path|data::labeled_resourses(base_path.as_ref().join("train.csv")),
        |path, device| parquet_to_tensor_by_path(path, device)
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

    }
}