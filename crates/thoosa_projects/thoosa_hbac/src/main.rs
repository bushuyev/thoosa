mod data;

use std::time::Instant;
use log::{debug, error, info};
use tch::Device::Cpu;
use tch::kind::FLOAT_CUDA;
use tch::Tensor;
use thoosa_main::cli::run_cli;

fn main()  {
    let t1 = Instant::now();

    unsafe {
        //TODO to kick REGISTER_CUDA_HOOKS(CUDAHooks); in
        // ATen   cuda
        //     │ │ │   detail
        //     │ │ │ │ │ 󰙲 CUDAHooks.cpp
        let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        // let func: libloading::Symbol<unsafe extern fn() -> u32> = lib.get(b"my_func")?;
        // Ok(func())
        println!("lib={lib:?}"); //logging is not set up yet
        let t = Tensor::rand(1_000_000, FLOAT_CUDA);
        let buf = Vec::<f32>::try_from(&t).unwrap();
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

// #[cfg(test)]
mod tests {
    use std::any::Any;
    use std::ffi::c_void;
    use std::sync::atomic::AtomicU64;
    use std::sync::RwLock;
    use std::time::Instant;
    use cudarc::driver::sys::CUdeviceptr_v2;
    use log::{debug, info};
    use parking_lot::Mutex;
    use polars::export::rayon::iter::IntoParallelIterator;
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::Float;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::{IndexOp, Kind, Tensor};
    use thoosa_common::utils::{copy_cudarc, free_cudarc, from};
    use thoosa_motifs::distance::{dist_d2, clc_dist_d2_and_mask};
    use thoosa_motifs::chunks::ChunkOfSlicedTensors;
    use thoosa_shared::ThoosaError;
    use polars::export::rayon::iter::ParallelIterator;
    use thoosa_common::loader::SyncWrapper;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::init_test_logging();
    }

    #[test]
    fn test_0(){
        println!("Ok");
    }

    #[test]
    fn test_to_cpu() {
        unsafe {
            let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");

            // let n = 2_147_483_647-1;
            let n = 4_000_000_000;
            let t = Tensor::rand([n], FLOAT_CUDA).view((1,-1));
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
            println!("t={:?} inst={}", t.size(), inst_0.elapsed().as_millis()/10);
        }
    }


    // #[test]
    fn test_data(){
        unsafe {

            let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");


            // let t = Tensor::from_slice(&[0.7, 0.2, 0.3]).totype(Float).to_device(Cuda(0));
            //6_809_600, 7_022_400, 6_968_900
            let t = Tensor::rand(10_000_000, FLOAT_CUDA);

            println!("t.size()={:?} device={:?}", t.size(), t.device());
            let mut cuda_t = 0;
            let  mut tch_t = 0;
            for c in 0..1000 {
                let i_0 = Instant::now();
                for i in 0..3 {
                    let buf = copy_cudarc(&t) as *mut f32;
                    cudarc::driver::sys::lib().cuMemFreeHost(buf as *mut c_void);
                }
                cuda_t += i_0.elapsed().as_millis();


                let i_0 = Instant::now();
                for i in 0..3 {
                    let buf = Vec::<f32>::try_from(&t).unwrap();
                }
                tch_t += i_0.elapsed().as_millis();
            }

            println!("cuda_t: {cuda_t} tch_t={tch_t}");

            // println!("buf={:?}", buf);
            // println!("x={:?}", *(buf));

            // let data = t.data_ptr();
            // println!("data={:?}", data);
            // println!("x={:?}", *(data as *mut f32));
            // println!("x={:?}", *((data.add(4)) as *mut f32));
        }

    }

}