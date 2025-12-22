mod data;

use log::{error, info};
use std::time::Instant;
use tch::Device::Cpu;
use thoosa_common::utils::kick_register_cuda_hooks;
use thoosa_main::cli::run_cli;
use thoosa_shared::MyError;
use thoosa_shims::context::ThoCudaContext;

// #[tokio::main(flavor = "multi_thread", worker_threads = 32)]
 fn main() {
    kick_register_cuda_hooks();
    //
    let t1 = Instant::now();
    //
    // /*
    // sqlite> select count(*), lb.label from (select count(*), label_id from labeled_resources group by resource_id having sum(p)=max(p)) as cnt left join labels lb on lb.id=cnt.label_id group by label_id;
    // 2227|seizure
    // 1181|lpd
    // 806|gpdh
    // 384|lrda
    // 1154|grda
    // 4208|other
    // */
    let rt = tokio::runtime::Builder::new_multi_thread()
        .on_thread_start(|| {
            //runs cuCtxSetCurrent
            ThoCudaContext::new(0).expect("Failed to create context");//TODO device
        })
        .worker_threads(4) //std::thread::available_parallelism().expect("available_parallelism failed").into()) // async workers
        .max_blocking_threads(32)        // more threads for spawn_blocking jobs
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

#[cfg(test)]
mod tests {
    #![allow(unused_imports, dead_code, unused_variables)]
    use cudarc::driver::sys::CUdeviceptr_v2;
    use log::{debug, info};
    use parking_lot::Mutex;
    use std::any::Any;
    use std::ffi::c_void;
    use std::sync::RwLock;
    use std::sync::atomic::AtomicU64;
    use std::time::Instant;

    use tch::Device::{Cpu, Cuda};
    use tch::Kind::Float;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::{IndexOp, Kind, Tensor};
    use thoosa_motifs::chunks::ChunkOfSlicedTensors;
    use thoosa_motifs::distance::{clc_dist_d2_and_mask, dist_d2};
    use thoosa_shared::ThoosaError;

    use thoosa_common::loader::SyncWrapper;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    // #[test]
    fn test_0() {
        println!("Ok");
    }

    #[test]
    fn test_to_cpu() {
        unsafe {
            let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");

            // let n = 2_147_483_647-1;
            let n = 4_000_000_000;
            let t = Tensor::rand([n], FLOAT_CUDA).view((1, -1));
            let inst_0 = Instant::now();

            for i in 0..10 {
                let indices = Tensor::randint(n, &[1000], (Kind::Int64, t.device()));
                let samples = t.index_select(1, &indices);
                // println!("samples.size={:?}", samples.size());

                let q = samples.quantile_scalar(0.1, 1, true, "higher");
                // println!("q.size={:?}", q.size());
                // println!("q={q}");

                let le = t.le_tensor(&q);
                // println!("le.size={:?}", le.size());
                // let masked = t.masked_select(&le);

                // let masked_cpu = masked.to_device(Cpu);
                let le_cpu = le.to_device(Cpu); //1488 mls
                // let buf = copy_cudarc(&masked) as *mut f32;
                // let le_buf = copy_cudarc(&le) as *mut f32; //8128 mls
                // free_cudarc(le_buf);
                // free_cudarc(buf);

                // let t_cpu =t.to_device(Cpu);
                // let buf = copy_cudarc(&t) as *mut f32;
                // free_cudarc(buf);
            }

            //
            println!(
                "t={:?} inst={}",
                t.size(),
                inst_0.elapsed().as_millis() / 10
            );
        }
    }
}
