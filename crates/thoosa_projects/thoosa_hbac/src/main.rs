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

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::time::Instant;
    use cudarc::driver::sys::CUdeviceptr_v2;
    use log::{debug, info};
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::Float;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::{Kind, Tensor};
    use thoosa_common::utils::copy_cudarc;
    use thoosa_motifs::distance::{dist_d2, dist_d2_0};
    use thoosa_motifs::find::ChunkOfSlicedTensors;
    use thoosa_shared::ThoosaError;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::init_test_logging();
    }

    #[test]
    fn test_0(){
        println!("Ok");
    }

    #[test]
    fn test_max_dist() {
        unsafe {
            let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");



            let tensors = 100;
            let step = 100;
            let window = 1000;
            let channels = 20;
            let input_n = 10_000;
            let chunk_size = 3;
            let dist_2_0 = true;

            info!("tensors={tensors} step={step} window={window} channels={channels} input_n={input_n} chunk_size={chunk_size}");

            let tensors = (0..tensors).map(|_| {
                Tensor::rand([channels, input_n], FLOAT_CPU).into()
            }).collect::<Vec<_>>();

            let chunks = tensors.chunks(chunk_size);

            let t_0 = Instant::now();
            let mut i = 0;
            for chunk_0 in chunks.clone() {

                let t_0_sliced = ChunkOfSlicedTensors::slice_tensors_chunk(step, window, chunk_0, vec!["x"], Cpu).t.transpose(0, 1).to_device(Cuda(0));

                for chunk_1 in chunks.clone() {
                    let t_1_sliced = ChunkOfSlicedTensors::slice_tensors_chunk(step, window, chunk_1, vec!["x"], Cpu).t.transpose(0, 1).to_device(Cuda(0));

                    if dist_2_0 {
                        match dist_d2_0(&t_0_sliced, &t_1_sliced) {
                            Ok(dists) => {
                                debug!("dists size={:?} device={:?}", dists.size(), dists.device());

                                let buf = copy_cudarc(&dists) as *mut f32;
                                cudarc::driver::sys::lib().cuMemFreeHost(buf as *mut c_void);
                            }
                            Err(e) => {
                                debug!("e={e:?}");
                            }


                        }

                    } else {
                        for c in 0..channels {
                            let tensor_0 = chunk_0.i((.., c as i64, ..)).to_device_(Cuda(0), Kind::Float, true, true).squeeze();
                            let tensor_1 = chunk_1.i((.., c as i64, ..)).to_device_(Cuda(0), Kind::Float, true, true).squeeze();

                            let euc_dist_t = dist_d2(
                                &tensor_0,
                                &tensor_1
                            ).expect(format!("dist_d2 on channel={}", c, ).as_str());

                            let buf = copy_cudarc(&euc_dist_t) as *mut f32;
                            cudarc::driver::sys::lib().cuMemFreeHost(buf as *mut c_void);
                        }
                    }
                }
            }

            info!("done in {}", t_0.elapsed().as_millis());

        }
    }

    #[test]
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