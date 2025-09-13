use tch::{Device, Kind, Tensor};
use std::sync::Arc;
use tokio_utils::Pool;
use cudarc::driver::{sys, CudaStream, PinnedHostSlice};
use tch::Device::Cpu;
use std::{iter, usize};
use tokio::task::JoinSet;
use std::ops::Add;
// test-only imports moved into test modules
use caching_runner::common::MyError;

//free_mem, total_mem
pub fn cuda_mem() -> (usize, usize) {
    use cudarc::driver::CudaContext;
    use cust::sys::cuMemGetInfo_v2;

    /*// 1. Initialise / retain the primary context on GPU 0.
    //    The returned Arc is kept alive for the rest of this scope.
    let _ctx = CudaContext::new(0).expect("failed to create CUDA context");

    // 2. Query memory usage via the safe runtime helper.
    result::get_mem_info().expect("cudaMemGetInfo failed")*/
    let _ctx = CudaContext::new(0).unwrap();
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        let _res = cuMemGetInfo_v2(&mut free, &mut total);
        // assert_eq!(res, sys::CUresult::CUDA_SUCCESS);
    }
    (free, total)
}

pub fn tensor_mem_size(init_type: Kind, size: &[i64]) -> usize {
    size.iter().fold(1, |a, v| a * (*v as usize)) * init_type.elt_size_in_bytes()
}



#[allow(dead_code)]
pub fn speed_gib_per_s(bytes: u64, time_ms: u64) -> f64 {
    if time_ms == 0 {
        return f64::INFINITY;
    }
    // (bytes × 8 bits/byte) ÷ seconds  →  bits/s, then ÷ 1e9 → Gb/s
    (bytes as f64 * 8.0 * 1_000.0) / (time_ms as f64 * 1_000_000_000.0)
}

#[allow(dead_code)]
pub fn speed_gb_per_s(bytes: u64, millis: u64) -> f64 {
    // Avoid division by zero
    if millis == 0 {
        return f64::INFINITY;
    }

    let gigabytes = bytes as f64 / 1_000_000_000.0; // bytes → GB
    let seconds = millis as f64 / 1_000.0; // ms    → s

    gigabytes / seconds
}

#[allow(dead_code)]
pub async fn copy_d2h<R:AsRef<Tensor>>(
    init_type: Kind,
    src_t: &R,
    buf_size: usize,
    bufs_pool: &Arc<Pool<(PinnedHostSlice<u8>, Arc<CudaStream>)>>,
) -> Tensor {
    let src_ptr: usize = src_t.as_ref().data_ptr() as usize;
    let src_size = src_t.as_ref().size();

    copy_d2h_from_ptr(init_type, buf_size, &bufs_pool, src_ptr, &src_size).await
}

pub async fn copy_d2h_from_ptr_result(
    init_type: Kind,
    buf_size_bytes: usize,
    bufs_pool: &Arc<Pool<(PinnedHostSlice<u8>, Arc<CudaStream>)>>,
    src_ptr: usize,
    src_size: &[i64],
) -> Result<Tensor, MyError> {
    let tensor_mem_size = tensor_mem_size(init_type, &src_size);

    // println!(
    //     "copy_d2h: src_t.size={:?} tensor_mem_size={tensor_mem_size} buf_size={buf_size_bytes}",
    //     src_size
    // );

    let chunks = tensor_mem_size / buf_size_bytes;

    let last = tensor_mem_size % buf_size_bytes;

    // println!("chunks={chunks} last={last}");
    let dst_t = Tensor::empty(src_size, (init_type, Cpu));
    unsafe {
        let dst_ptr: usize = dst_t.data_ptr() as usize;

        let iter = (0..=chunks)
            .into_iter()
            .zip(iter::repeat(buf_size_bytes).take(chunks));

        let iter = if last == 0 {
            Box::new(iter) as Box<dyn Iterator<Item=(usize, usize)>>
        } else {
            Box::new(iter.chain(iter::once((chunks, last))))
                as Box<dyn Iterator<Item=(usize, usize)>>
        };

        iter.map(|(i, c_buf_size)| {
            let off = i * buf_size_bytes;

            let src_off_ptr = src_ptr.add(off as usize) as sys::CUdeviceptr;
            let dst_off_ptr: usize = dst_ptr.add(off as usize);

            let pool = Arc::clone(&bufs_pool);
            async move {
                // println!("starting {i}");
                let buf_stream = pool.acquire().await;
                let buf_ptr = buf_stream.0.as_ptr().unwrap() as usize;

                let stream = buf_stream.1.clone();
                // from cuda to pinned buffer
                let r = cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                    buf_ptr as *mut _,
                    src_off_ptr,
                    c_buf_size,
                    stream.cu_stream(), // as CUstream
                );
                if r != sys::CUresult::CUDA_SUCCESS {
                    return Err(MyError::SomethingElse(format!(
                        "cuMemcpyDtoHAsync_v2 failed at chunk {i} with {r:?}"
                    )));
                }

                buf_stream.1.synchronize().unwrap();


                // from pinned buffer into the destination tensor
                // task::spawn_blocking(move || {
                    std::ptr::copy_nonoverlapping(
                        buf_ptr as *mut _,
                        dst_off_ptr as *mut u8,
                        c_buf_size,
                    );
                // })
                //     .await
                //     .unwrap();

                drop(buf_stream);
                Ok(())
            }
        })
            .collect::<JoinSet<Result<(), MyError>>>()
            .join_all()
            .await
            .into_iter()
            .collect::<Result<(), MyError>>()?;
    }
    Ok(dst_t)
}

pub async fn copy_d2h_from_ptr(
    init_type: Kind,
    buf_size_bytes: usize,
    bufs_pool: &Arc<Pool<(PinnedHostSlice<u8>, Arc<CudaStream>)>>,
    src_ptr: usize,
    src_size: &[i64],
) -> Tensor {
    match copy_d2h_from_ptr_result(init_type, buf_size_bytes, bufs_pool, src_ptr, src_size).await {
        Ok(t) => t,
        Err(e) => panic!("copy_d2h_from_ptr: {e:?}"),
    }
}

#[allow(dead_code)]
pub async fn copy_h2d<R:AsRef<Tensor>>(
    init_type: Kind,
    src_t: &R,
    buf_size: usize,
    bufs_pool: &Arc<Pool<(PinnedHostSlice<u8>, Arc<CudaStream>)>>,
    dst_device_idx: usize,
) -> Tensor {
    assert!(dst_device_idx > 0);
    let src_ptr = src_t.as_ref().data_ptr() as usize; // CPU mem
    let src_size = src_t.as_ref().size();


    copy_h2d_from_ptr(init_type, buf_size, bufs_pool, dst_device_idx, src_ptr, &src_size).await

}

pub async fn copy_h2d_from_ptr(init_type: Kind, buf_size: usize, bufs_pool: &Arc<Pool<(PinnedHostSlice<u8>, Arc<CudaStream>)>>, dst_device_idx: usize, src_ptr: usize, size: &[i64]) -> Tensor {
    let tensor_mem_size = tensor_mem_size(init_type, &size);
    // println!(
    //     "copy_h2d: src_t.size={:?} tensor_mem_size={tensor_mem_size} buf_size={buf_size}",
    //     size
    // );

    // chunk layout
    let chunks = tensor_mem_size / buf_size;
    let last = tensor_mem_size % buf_size;
    // println!("chunks={chunks} last={last}");

    // destination tensor lives on GPU 0
    let dst_t = Tensor::empty(size, (init_type, Device::Cuda(dst_device_idx - 1)));

    unsafe {
        // base pointers

        let dst_ptr = dst_t.data_ptr() as usize; // GPU mem

        // CUdriver needs CUdeviceptr for the destination side
        let dst_base: sys::CUdeviceptr = dst_ptr as sys::CUdeviceptr;

        // iterator over all full-sized + optional tail chunk
        let iter = (0..=chunks).zip(iter::repeat(buf_size).take(chunks));
        let iter: Box<dyn Iterator<Item=(usize, usize)>> = if last == 0 {
            Box::new(iter)
        } else {
            Box::new(iter.chain(iter::once((chunks, last))))
        };

        // launch one asynchronous task per chunk
        iter.map(|(i, c_buf_size)| {
            let off = i * buf_size;

            // raw pointers for this chunk
            let src_off_ptr = src_ptr.add(off as usize) as usize;
            let dst_off_ptr = dst_base + off as u64; // CUdeviceptr is a u64

            // clone the pool inside the async block
            let pool = Arc::clone(bufs_pool);

            async move {
                // grab (pinned slice, stream) from the pool
                let buf_stream = pool.acquire().await;
                let buf_ptr = buf_stream.0.as_ptr().unwrap() as usize;
                let stream = buf_stream.1.clone();

                //----------------------------------------------------------
                // 1. copy   CPU-tensor  →  pinned host buffer   (memcpy)
                //----------------------------------------------------------
                // task::spawn_blocking(move || {
                    std::ptr::copy_nonoverlapping(
                        src_off_ptr as *const u8,
                        buf_ptr as *mut u8,
                        c_buf_size,
                    );
                // })
                //     .await
                //     .unwrap();

                //----------------------------------------------------------
                // 2. copy   pinned host buffer  →  GPU tensor   (CUDA)
                //----------------------------------------------------------
                // task::spawn_blocking(move || {
                    let _  = sys::cuMemcpyHtoDAsync_v2(
                        dst_off_ptr,
                        buf_ptr as *const _,
                        c_buf_size,
                        stream.cu_stream(),
                        );
                    // })
                    // .await
                    // .unwrap();

                // wait for GPU work to finish before recycling buffer
                buf_stream.1.synchronize().unwrap();
                drop(buf_stream); // return buffer/stream to pool
            }
        })
            .collect::<JoinSet<()>>()
            .join_all()
            .await;
    }

    dst_t
}


#[allow(dead_code)]
pub fn equal_bytes(a: &Tensor, b: &Tensor) -> bool {
    assert!(a.device() == Cpu && b.device() == Cpu);
    assert_eq!(a.kind(), b.kind());
    assert_eq!(a.size(), b.size());
    assert!(a.is_contiguous() && b.is_contiguous());

    // nbytes: prefer numel * element_size; tch exposes element_size() on Tensor in recent versions.
    let nbytes = a.numel() * a.kind().elt_size_in_bytes();

    let pa = a.data_ptr() as *const u8;
    let pb = b.data_ptr() as *const u8;

    unsafe {
        std::slice::from_raw_parts(pa, nbytes) == std::slice::from_raw_parts(pb, nbytes)
    }
}


#[cfg(test)]
mod tests {
    use std::cell::UnsafeCell;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering::Relaxed;
    use rand::random_range;
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::{Float, Int64};
    use tch::{Kind, Tensor};
    use tokio::task::JoinSet;
    use tokio::time::Instant;
    use tokio_utils::Pool;
    use crate::copy_ctx::CopyCtx;
    use crate::cuda_utils::{copy_d2h_from_ptr, copy_h2d_from_ptr, cuda_mem, equal_bytes, speed_gb_per_s};
    use crate::tests::lin_t;


    // #[tokio::test(flavor = "multi_thread", worker_threads = 64)]
    #[cfg_attr(miri, ignore)]
    #[allow(dead_code)]
    async  fn heavy_test_copy_cudarc() {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");


            let mut ctx : Option<UnsafeCell< crate::copy_ctx::CopyCtx >> = None;

            for mem_k in [0.1, 0.5, 1.0, 2.]{
                for buffs in [8, 16, 32, 64] {

                    if ctx.is_none()  {
                         ctx = Some(UnsafeCell::new(CopyCtx::make_instance(mem_k, buffs)));
                    } else {
                        ctx.as_mut().unwrap().get_mut().rebuild(mem_k, buffs)
                    }

                    run_test_copy_cudarc(&*ctx.as_ref().unwrap().get()).await;

                }
            }
        }
    }

    #[allow(dead_code)]
    async fn run_test_copy_cudarc(ctx: &'static CopyCtx){

        use cudarc::driver::CudaContext;
        let _cu_ctx = Arc::new(CudaContext::new(0).unwrap());

        let (free_mem, _total_mem) = cuda_mem();

        let init_type = Float;

        let total_size = (free_mem as f64 * 0.2) as u64 / init_type.elt_size_in_bytes() as u64; //123456; //25000; //123456; //
        let dim_1 = 4u64;
        let input_n = total_size / dim_1;


        let d2h_time = Arc::new(AtomicU64::new(0));
        let h2d_time = Arc::new(AtomicU64::new(0));

        let threads = 3;
        let runs = 10;

        (0..threads)
            .map(|thread_i| {
                let d2h_time = d2h_time.clone();
                let h2d_time = h2d_time.clone();


                run_copy_(
                    thread_i,
                    input_n as i64,
                    dim_1 as i64,
                    runs,
                    Float,
                    Arc::clone(&d2h_time),
                    Arc::clone(&h2d_time),
                    ctx,
                )

            })
            .collect::<JoinSet<()>>()
            .join_all()
            .await;

        let total_copied_mem_size_kb =
            input_n * dim_1 * init_type.elt_size_in_bytes() as u64 * runs as u64 * threads as u64;

        println!(
            "\nd2h speed: {:#2} Gb/s, h2d speed {:#2} Gb/s",
            speed_gb_per_s(total_copied_mem_size_kb, d2h_time.load(Relaxed)),
            speed_gb_per_s(total_copied_mem_size_kb, h2d_time.load(Relaxed))
        );
    }


    #[allow(dead_code)]
    async fn run_copy_(
        _thread_i: u8,
        input_n:i64, dim_1:i64,
        runs: u8,
        init_type: Kind,

        d2h_time: Arc<AtomicU64>,
        h2d_time: Arc<AtomicU64>,
        ctx: &CopyCtx,
    ) {
        let dim_0 = random_range(0.5 * input_n as f64..1.5 * input_n as f64) as i64;
        // println!("run_copy_: dim_0={dim_0} dim_1={dim_1}");
        let mut src_t = Tensor::rand([dim_0, dim_1], (Float, Cpu));

        let mut src_ptr = src_t.data_ptr() as usize;
        let mut src_size = src_t.size();
        for _t in 0..runs {
            // println!("run_copy_: thread_i={thread_i} t={t} src_t.size={:?}", src_t.size());
            let inst_0 = Instant::now();
            let tensor = ctx.copy_from_ptr(src_ptr, &src_size, 1, init_type).await;
            h2d_time.fetch_add(inst_0.elapsed().as_millis() as u64, Relaxed);

            (src_ptr, src_size) = (tensor.data_ptr() as usize, tensor.size());

            // println!("run_copy_: thread_i={thread_i} t={t} copy_d2h done in {}", inst_0.elapsed().as_millis());

            let inst_0 = Instant::now();
            let tensor = ctx.copy_from_ptr(src_ptr, &src_size, 0, init_type).await;
            d2h_time.fetch_add(inst_0.elapsed().as_millis() as u64, Relaxed);

            (src_ptr, src_size) = (tensor.data_ptr() as usize, tensor.size());


            assert!(equal_bytes(&tensor, &src_t));
            print!(".");
            src_t = tensor;

        }
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 64)]
    async fn test_small_one_copy_types(){
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        use cudarc::driver::CudaContext;
        let cu_ctx = Arc::new(CudaContext::new(0).unwrap());

        let (free_mem, _total_mem) = cuda_mem();

        let init_type = Int64;

        let elts_in_mem = free_mem / init_type.elt_size_in_bytes() as usize; //123456; //25000; //123456; //

        let buffs = 64;
        let elts_in_buff = elts_in_mem / buffs;

        let buf_size = elts_in_buff * init_type.elt_size_in_bytes() + 17; //64 * 1024 * 1024; // total_size/ buffs as i64;
        let bufs_pool = Arc::new(Pool::from_iter((0..buffs).map(|_i| {
            (
                unsafe { cu_ctx.alloc_pinned::<u8>(buf_size as usize).unwrap() },
                cu_ctx.new_stream().unwrap(),
            )
        })));

        for tensor_kind in [Kind::Uint8, Kind::Int16, Kind::Int, Kind::Int64,  Kind::Double] {

            let src_t = Tensor::rand(&[10, 1], (Float, Cuda(0))).multiply_scalar(10.).totype(tensor_kind);

            println!("src_t=\t{src_t}");

            let dst = copy_d2h_from_ptr(tensor_kind, buf_size, &bufs_pool, src_t.data_ptr() as usize, &src_t.size()).await;
            println!("dst=\t{dst}");

            let dst = copy_h2d_from_ptr(tensor_kind, buf_size, &bufs_pool, 1, dst.data_ptr() as usize, &dst.size()).await;

            assert!(src_t.allclose(&dst, 0.01, 0.01, false));
        }

    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 64)]
    async fn heavy_one_copy_types(){
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        use cudarc::driver::CudaContext;
        let cu_ctx = Arc::new(CudaContext::new(0).unwrap());

        let (free_mem, _total_mem) = cuda_mem();

        let init_type = Int64;

        let elts_in_mem = free_mem / init_type.elt_size_in_bytes() as usize; //123456; //25000; //123456; //
        let dim_1 = 4i64;
        let input_n = ((elts_in_mem as f64 * 0.1 ) as usize/ dim_1 as usize) as i64;

        let buffs = 64;
        let elts_in_buff = elts_in_mem / buffs;

        let buf_size = elts_in_buff * init_type.elt_size_in_bytes() + 17; //64 * 1024 * 1024; // total_size/ buffs as i64;
        let bufs_pool = Arc::new(Pool::from_iter((0..buffs).map(|_i| {
            (
                unsafe { cu_ctx.alloc_pinned::<u8>(buf_size as usize).unwrap() },
                cu_ctx.new_stream().unwrap(),
            )
        })));

        for tensor_kind in [Kind::Uint8, Kind::Int16, Kind::Int, Kind::Int64,  Kind::Double] {


            let src_t = lin_t(1., [input_n, dim_1], tensor_kind, Cuda(0));

            println!("src_t=\t{src_t}");

            let dst = copy_d2h_from_ptr(tensor_kind, buf_size, &bufs_pool, src_t.data_ptr() as usize, &src_t.size()).await;
            println!("dst=\t{dst}");

            let dst = copy_h2d_from_ptr(tensor_kind, buf_size, &bufs_pool, 1, dst.data_ptr() as usize, &dst.size()).await;

            assert!(src_t.allclose(&dst, 0.01, 0.01, false));
        }


    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_copy_d2h_from_ptr_reports_error() {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        use cudarc::driver::{CudaContext};

        // Setup a small pool
        let cu_ctx = Arc::new(CudaContext::new(0).unwrap());
        let buffs = 2usize;
        let buf_size = 1024usize;
        let bufs_pool = Arc::new(Pool::from_iter((0..buffs).map(|_| {
            (
                unsafe { cu_ctx.alloc_pinned::<u8>(buf_size).unwrap() },
                cu_ctx.new_stream().unwrap(),
            )
        })));

        // Use a CPU pointer as if it were a device pointer — should error
        let bogus_cpu_tensor = Tensor::zeros([4], (Kind::Int, Cpu));
        let bogus_device_ptr = bogus_cpu_tensor.data_ptr() as usize; // not a CUdeviceptr

        let res = super::copy_d2h_from_ptr_result(Kind::Int, buf_size, &bufs_pool, bogus_device_ptr, &bogus_cpu_tensor.size()).await;
        assert!(res.is_err(), "expected copy_d2h_from_ptr_result to error for invalid device ptr");
    }

}
