use std::time::Instant;
use log::{error, info};
use tch::Device::Cpu;
use tch::Tensor;
use thoosa_common::utils::kick_register_cuda_hooks;
use thoosa_main::cli::run_cli;
use crate::data::resource_to_tensor;
use crate::labels::LABELS;

mod data;
mod labels;

fn main()  {
    kick_register_cuda_hooks();

    let t1 = Instant::now();

    match run_cli(
        |source_path|data::labeled_resourses(source_path.as_ref().join("train.csv")),
        resource_to_tensor,
        &LABELS,
        1,
        usize::MAX,
        32000
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
