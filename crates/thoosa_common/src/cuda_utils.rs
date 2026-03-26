use cudarc::driver::sys;
use std::ffi::c_void;
use std::ops::Add;
use std::sync::Arc;
use cust::sys::CU_MEMHOSTALLOC_WRITECOMBINED;
use futures_util::{stream, StreamExt};
use log::{debug, error};
use tch::Device::Cpu;
use tch::{Device, Kind, Tensor};
use thoosa_shared::{Cause, ThoError};
use tokio_utils::Pool;
use thoosa_shims::context::ThoCudaContext;
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use thoosa_shared::os_thread_id;

pub struct PinnedHostBuffer {
    ptr: *mut u8,
    len: usize,
}


impl PinnedHostBuffer {
    pub fn new(len: usize) -> Result<Self, ThoError> {
        use cudarc::driver::sys::CUresult::CUDA_SUCCESS;

        let mut raw: *mut c_void = std::ptr::null_mut();
        let flags = CU_MEMHOSTALLOC_WRITECOMBINED;
        let res = unsafe {
            thoosa_shims::memcpy::tho_cu_mem_host_alloc(&mut raw, len, flags)
        };

        if res != CUDA_SUCCESS as i32 {
            return Err(Cause::SomethingElse(format!(
                "cuMemHostAlloc failed with {res:?} (len={len})"
            ))
            .into());
        }

        Ok(Self {
            ptr: raw as *mut u8,
            len,
        })
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

unsafe impl Send for PinnedHostBuffer {}
unsafe impl Sync for PinnedHostBuffer {}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        use cudarc::driver::sys::CUresult::CUDA_SUCCESS;

        if self.ptr.is_null() {
            return;
        }

        let res = unsafe {
            thoosa_shims::memcpy::tho_cu_mem_free_host(self.ptr as *mut c_void)
        };

        if res != CUDA_SUCCESS as i32 {
            log::error!("cuMemFreeHost failed with {res:?}");
        }
    }
}
// test-only imports moved into test modules

//free_mem, total_mem
pub fn cuda_mem() -> (usize, usize) {
    use cust::sys::cuMemGetInfo_v2;

    let ctx = ThoCudaContext::new(0).expect("failed to create CUDA context");
    ctx
        .bind_to_thread()
        .expect("failed to bind CUDA context while querying memory usage");

    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        let _res = cuMemGetInfo_v2(&mut free, &mut total);
        // assert_eq!(res, sys::CUresult::CUDA_SUCCESS);
    }
    (free, total)
}

///for cases when only size and type is known
pub fn tensor_mem_size(init_type: Kind, size: &[i64]) -> usize {//TODO rename to not confuse with one from thoosa_common::utils and move
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

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="cuda_utils::copy_d2h"))]
pub async fn copy_d2h<R: AsRef<Tensor>>(
    init_type: Kind,
    src_t: &R,
    buf_size: usize,
    bufs_pool: &Arc<Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>>,
) -> Result<Tensor, ThoError> {

    assert!(src_t.as_ref().device().ne(&Device::Cpu));


    let src_ptr: usize = src_t.as_ref().data_ptr() as usize;
    let src_size = src_t.as_ref().size();

    copy_d2h_from_ptr(init_type, buf_size, &bufs_pool, src_ptr, &src_size).await
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="cuda_utils::move_d2h"))]
pub async fn move_d2h(
    init_type: Kind,
    src_t: Tensor,
    buf_size: usize,
    bufs_pool: &Arc<Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>>,
) -> Result<Tensor, ThoError> {

    assert!(src_t.as_ref().device().ne(&Device::Cpu));


    let src_ptr: usize = src_t.data_ptr() as usize;
    let src_size = src_t.size();

    let res = copy_d2h_from_ptr(init_type, buf_size, &bufs_pool, src_ptr, &src_size).await;
    drop(src_t); // keep source alive until the copy finishes
    res
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="cuda_utils::copy_d2h_from_ptr"))]
pub async fn copy_d2h_from_ptr(
    init_type: Kind,
    buf_size_bytes: usize,
    bufs_pool: &Arc<Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>>,
    src_ptr: usize,
    src_size: &[i64],
) -> Result<Tensor, ThoError> {
    let tensor_mem_size = tensor_mem_size(init_type, &src_size);

    debug!("copy_d2h: src_t.size={:?} tensor_mem_size={tensor_mem_size} buf_size={buf_size_bytes}",src_size);

    let chunks = tensor_mem_size / buf_size_bytes;

    let last = tensor_mem_size % buf_size_bytes;

    let dst_t = Tensor::empty(src_size, (init_type, Cpu));
    unsafe {
        let dst_ptr: usize = dst_t.data_ptr() as usize;

        let chunk_ranges = (0..chunks)
            .map(|i| (i, buf_size_bytes))
            .chain((last != 0).then_some((chunks, last)))
            .collect::<Vec<_>>();

        let concurrency = 32; //chunk_ranges.len().max(1);

        stream::iter(chunk_ranges)
            .map(|(i, c_buf_size)| {
                let off = i * buf_size_bytes;

                let src_off_ptr = src_ptr.add(off as usize) as thoosa_shims::memcpy::CuDevicePtr;
                let dst_off_ptr: usize = dst_ptr.add(off as usize);

                let pool = Arc::clone(&bufs_pool);
                async move {
                    debug!("starting {i} thread_name={:?}", os_thread_id());
                    let buf_stream = pool.acquire().await;
                    let buf_ptr = buf_stream.0.as_mut_ptr() as usize;

                    let stream = buf_stream.1.clone();

                    tokio::task::spawn_blocking(move || {
                        let raw = thoosa_shims::memcpy::tho_cu_memcpy_dtoh_async(
                            buf_ptr as *mut c_void,
                            src_off_ptr as thoosa_shims::memcpy::CuDevicePtr,
                            c_buf_size,
                            stream.as_mut_ptr(),
                        );
                        let r: sys::CUresult = std::mem::transmute(raw);
                        if r != sys::CUresult::CUDA_SUCCESS {
                            return Err::<(), Cause>(Cause::SomethingElse(format!("tho_cu_memcpy_dtoh_async failed at chunk {i} with {r:?}. buf_ptr={buf_ptr} src_off_ptr={src_off_ptr} c_buf_size={c_buf_size}")).into());
                        }

                        match stream.synchronize() {
                            Ok(_) => {}
                            Err(e) => {
                                error!("stream.synchronize: e={e:?}")
                            }
                        }

                        std::ptr::copy_nonoverlapping(
                            buf_ptr as *mut _,
                            dst_off_ptr as *mut u8,
                            c_buf_size,
                        );
                        Ok(())
                    })
                        .await
                        .map_err(|e| Cause::External {
                            type_name: std::any::type_name::<tokio::task::JoinError>(),
                            source: Arc::new(e),
                        })??;

                    drop(buf_stream);
                    Ok(())
                }
            })
            .buffer_unordered(concurrency)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<(), Cause>>()?;
    }
    Ok(dst_t)
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="cuda_utils::copy_h2d"))]
pub async fn copy_h2d<R: AsRef<Tensor>>(
    init_type: Kind,
    src_t: &R,
    buf_size: usize,
    bufs_pool: &Arc<Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>>,
    dst_device_idx: usize,
) -> Result<Tensor, ThoError> {
    assert!(src_t.as_ref().device().eq(&Cpu));
    assert!(dst_device_idx > 0);

    let src_ptr = src_t.as_ref().data_ptr() as usize; // CPU mem
    let src_size = src_t.as_ref().size();

    copy_h2d_from_ptr(
        init_type,
        buf_size,
        bufs_pool,
        dst_device_idx,
        src_ptr,
        &src_size,
    )
    .await
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="cuda_utils::move_h2d"))]
pub async fn move_h2d(
    init_type: Kind,
    src_t: Tensor,
    buf_size: usize,
    bufs_pool: &Arc<Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>>,
    dst_device_idx: usize,
) -> Result<Tensor, ThoError> {
    assert!(src_t.as_ref().device().eq(&Cpu));
    assert!(dst_device_idx > 0);

    let src_ptr = src_t.data_ptr() as usize; // CPU mem
    let src_size = src_t.size();

    let res = copy_h2d_from_ptr(
        init_type,
        buf_size,
        bufs_pool,
        dst_device_idx,
        src_ptr,
        &src_size,
    )
    .await;
    drop(src_t); // keep source alive until the copy finishes
    res
}


#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="cuda_utils::copy_h2d_from_ptr"))]
pub async fn copy_h2d_from_ptr(
    init_type: Kind,
    buf_size: usize,
    bufs_pool: &Arc<Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>>,
    dst_device_idx: usize,
    src_ptr: usize,
    size: &[i64],
) -> Result<Tensor, ThoError> {
    let tensor_mem_size = tensor_mem_size(init_type, &size);
    debug!("copy_h2d_from_ptr: src_t.size={:?} tensor_mem_size={tensor_mem_size} buf_size={buf_size}",size);

    // chunk layout
    let chunks = tensor_mem_size / buf_size;
    let last = tensor_mem_size % buf_size;

    // destination tensor lives on GPU 0
    let dst_t = Tensor::f_empty(size, (init_type, Device::Cuda(dst_device_idx - 1)))?;

    unsafe {
        // base pointers
        let dst_ptr = dst_t.data_ptr() as usize; // GPU mem

        // CUdriver needs CUdeviceptr for the destination side
        let dst_base = dst_ptr as thoosa_shims::memcpy::CuDevicePtr;

        let chunk_ranges = (0..chunks)
            .map(|i| (i, buf_size))
            .chain((last != 0).then_some((chunks, last)))
            .collect::<Vec<_>>();

        let concurrency = 32; //chunk_ranges.len().max(1);

        stream::iter(chunk_ranges)
            .map(|(i, c_buf_size)| {
                let off = i * buf_size;

                // raw pointers for this chunk
                let src_off_ptr = src_ptr.add(off);
                let dst_off_ptr = dst_base + off as u64; // CUdeviceptr is a u64

                // clone the pool inside the async block
                let pool = Arc::clone(bufs_pool);

                async move {
                    debug!("copy_h2d_from_ptr: starting {i} thread_name={:?}", os_thread_id());

                    // grab (pinned slice, stream) from the pool
                    let buf_stream = pool.acquire().await;
                    let buf_ptr = buf_stream.0.as_mut_ptr() as usize;
                    // let stream = buf_stream.1.clone();

                    //----------------------------------------------------------
                    // 1. tokio::task::spawn_blocking(move || {copy   CPU-tensor  →  pinned host buffer   (memcpy)
                    //----------------------------------------------------------
                    debug!("copy_h2d_from_ptr: src_off_ptr={src_off_ptr} buf_ptr={buf_ptr} c_buf_size={c_buf_size} thread_name={:?}", os_thread_id());
                    tokio::task::spawn_blocking(move || {
                        std::ptr::copy_nonoverlapping(
                            src_off_ptr as *const u8,
                            buf_ptr as *mut u8,
                            c_buf_size,
                        );

                        //----------------------------------------------------------
                        // 2. copy   pinned host buffer  →  GPU tensor   (CUDA)
                        //----------------------------------------------------------
                        let raw = thoosa_shims::memcpy::tho_cu_memcpy_htod_async(
                            dst_off_ptr as thoosa_shims::memcpy::CuDevicePtr,
                            buf_ptr as *const c_void,
                            c_buf_size,
                            buf_stream.1.as_mut_ptr(),
                        );
                        let r: sys::CUresult = std::mem::transmute(raw);
                        if r != sys::CUresult::CUDA_SUCCESS {
                            panic!("cuMemcpyHtoDAsync_v2 failed at chunk {i} with {r:?}. buf_ptr={buf_ptr} dst_off_ptr={dst_off_ptr} c_buf_size={c_buf_size}");
                        }
                        let _ = buf_stream.1.synchronize();
                        drop(buf_stream); // return buffer/stream to pool
                    }).await.unwrap();

                    // wait for GPU work to finish before recycling buffer
                }
            })
            .buffer_unordered(concurrency)
            .collect::<Vec<_>>()
            .await;
    }

    Ok(dst_t)
}

//TODO OOM handling
pub fn pack_bits_bool(b: &Tensor) -> Result<Tensor, ThoError> {
    assert_eq!(b.kind(), Kind::Bool, "pack_bits_bool expects Bool");
    let dev = b.device();

    // 1) Flatten.
    let mut flat = b.f_reshape([-1])?;

    // 2) Pad to multiple of 8.
    let n = flat.numel();
    let pad: i64 = (8 - (n % 8)) as i64 % 8;
    if pad != 0 {
        let zeros = Tensor::f_zeros([pad], (Kind::Bool, dev))?;
        flat = Tensor::f_cat(&[flat, zeros], 0)?;
    }

    // 3) Group by 8 and cast {false,true}->{0u8,1u8}.
    let x_u8 = flat.f_view([-1, 8])?.f_to_kind(Kind::Uint8)?;

    // 4) Weights (MSB->LSB).
    let w_i64 = Tensor::from_slice(&[128u8, 64, 32, 16, 8, 4, 2, 1])
        .f_to_device(dev)?
        .f_to_kind(Kind::Int64)?;

    // 5) Dot(each row, weights) -> one byte per row.
    Ok(x_u8
        .f_to_kind(Kind::Int64)?
        .f_mul(&w_i64)?
        .f_sum_dim_intlist(1, /*keepdim=*/false, Kind::Int64)?
        .f_to_kind(Kind::Uint8)?
        .f_contiguous()?)
}

/// Unpack bytes back to Bool bits (MSB-first).
pub fn unpack_bits_u8(bytes: &Tensor, original_bits: Option<i64>) -> Tensor {
    assert_eq!(bytes.kind(), Kind::Uint8, "unpack_bits_u8 expects UInt8");
    let dev = bytes.device();

    // 1) Bit masks, MSB->LSB, shape [1,8] for broadcast.
    let masks = Tensor::from_slice(&[128u8, 64, 32, 16, 8, 4, 2, 1])
        .to_device(dev)
        .unsqueeze(0); // [1,8], u8

    // 2) Broadcast to [M,8] and do **bitwise** AND (Tensor–Tensor).
    let expanded = bytes.unsqueeze(1); // [M,1], u8
    let anded = expanded.bitwise_and_tensor(&masks); // [M,8], u8

    // 3) Convert to Bool: bit is set iff value != 0.
    //    Use ne_tensor with a scalar 0 (broadcasts fine).
    let zeros = Tensor::zeros([1], (Kind::Uint8, dev));
    let per_bit_bool = anded.ne_tensor(&zeros); // [M,8], Bool

    // 4) Flatten and trim optional padding.
    let bits = per_bit_bool.reshape([-1]); //.to_kind(Kind::Bool);
    if let Some(k) = original_bits {
        bits.narrow(0, 0, k).contiguous()
    } else {
        bits.contiguous()
    }
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

    unsafe { std::slice::from_raw_parts(pa, nbytes) == std::slice::from_raw_parts(pb, nbytes) }
}

#[cfg(test)]
#[allow(unused_imports, unused_variables, dead_code)]
mod tests {
    use crate::copy_ctx::CopyCtx;
    use crate::cuda_utils::{
        copy_d2h_from_ptr,
        copy_h2d_from_ptr,
        cuda_mem,
        equal_bytes,
        pack_bits_bool,
        speed_gb_per_s,
        unpack_bits_u8,
        PinnedHostBuffer,
    };
    use crate::utils::lin_for_size;
    use log::{debug, info};
    use rand::random_range;
    use std::cell::UnsafeCell;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering::Relaxed;
    use futures_util::StreamExt;
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::{Float, Int64, Uint8};
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::{Kind, Tensor};
    use thoosa_shared::ThoError;
    use tokio::task::JoinSet;
    use tokio::time::Instant;
    use tokio_utils::Pool;
    use thoosa_shared::applicables_runner::Output;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_shared::traits::SimpleId;
    use thoosa_shims::{context::ThoCudaContext, stream::ThoCudaStream};
    use crate::dims::AnyDims;
    use crate::tho_device::ThoDevice;
    use crate::tho_tensor::ThoTensor;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 64)]
    #[cfg_attr(miri, ignore)]
    #[allow(dead_code)]
    async fn heavy_test_copy_from_ptr() {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");

            let mut ctx: Option<UnsafeCell<crate::copy_ctx::CopyCtx>> = None;

            for mem_k in [0.1, 0.5, 1.0, 2.] {
                for buffs in [8, 16, 32, 64] {
                    if ctx.is_none() {
                        ctx = Some(UnsafeCell::new(CopyCtx::make_instance(mem_k, buffs)));
                    } else {
                        ctx.as_mut().unwrap().get_mut().rebuild(mem_k, buffs)
                    }

                    run_test_copy_from_ptr(&*ctx.as_ref().unwrap().get(), 3, 3).await;
                }
            }
        }
    }

    // #[tokio::test(flavor = "multi_thread", worker_threads = 64)]
    async fn test_copy_from_ptr_once() {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        let ctx:&'static CopyCtx = Box::leak(Box::new(CopyCtx::make_instance(0.2, 16)));

        run_test_copy_from_ptr(ctx, 3, 3).await;
    }

    #[allow(dead_code)]
    async fn run_test_copy_from_ptr(ctx: &'static CopyCtx, threads: u8, runs: u8) {
        let (free_mem, _total_mem) = cuda_mem();

        let init_type = Float;

        let total_size = (free_mem as f64 * 0.2) as u64 / init_type.elt_size_in_bytes() as u64; //123456; //25000; //123456; //
        let dim_1 = 4u64;
        let input_n = total_size / dim_1;

        let d2h_time = Arc::new(AtomicU64::new(0));
        let h2d_time = Arc::new(AtomicU64::new(0));


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
            .collect::<JoinSet<Result<(), ThoError>>>()
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
        input_n: i64,
        dim_1: i64,
        runs: u8,
        init_type: Kind,

        d2h_time: Arc<AtomicU64>,
        h2d_time: Arc<AtomicU64>,
        ctx: &CopyCtx,
    ) ->Result<(), ThoError>{
        let dim_0 = random_range(0.5 * input_n as f64..1.5 * input_n as f64) as i64;
        // println!("run_copy_: dim_0={dim_0} dim_1={dim_1}");
        let mut src_t = Tensor::rand([dim_0, dim_1], (Float, Cpu));

        let mut src_ptr = src_t.data_ptr() as usize;
        let mut src_size = src_t.size();
        for _t in 0..runs {
            // println!("run_copy_: thread_i={thread_i} t={t} src_t.size={:?}", src_t.size());
            let inst_0 = Instant::now();
            let tensor = ctx.copy_from_ptr(src_ptr, &src_size, 1, init_type).await?;
            h2d_time.fetch_add(inst_0.elapsed().as_millis() as u64, Relaxed);

            (src_ptr, src_size) = (tensor.data_ptr() as usize, tensor.size());

            // println!("run_copy_: thread_i={thread_i} t={t} copy_d2h done in {}", inst_0.elapsed().as_millis());

            let inst_0 = Instant::now();
            let tensor = ctx.copy_from_ptr(src_ptr, &src_size, 0, init_type).await?;
            d2h_time.fetch_add(inst_0.elapsed().as_millis() as u64, Relaxed);

            (src_ptr, src_size) = (tensor.data_ptr() as usize, tensor.size());

            assert!(equal_bytes(&tensor, &src_t));

            src_t = tensor;
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 64)]
    async fn test_small_one_copy_types() ->Result<(), ThoError>{
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        let cu_ctx = ThoCudaContext::new(0).expect("failed to create CUDA context");
        cu_ctx
            .bind_to_thread()
            .expect("failed to bind CUDA context for test_small_one_copy_types");

        let (free_mem, _total_mem) = cuda_mem();

        let init_type = Int64;

        let elts_in_mem = free_mem / init_type.elt_size_in_bytes() as usize; //123456; //25000; //123456; //

        let buffs = 64;
        let elts_in_buff = elts_in_mem / buffs;

        let buf_size = elts_in_buff * init_type.elt_size_in_bytes() + 17; //64 * 1024 * 1024; // total_size/ buffs as i64;
        let bufs_pool = Arc::new(Pool::from_iter((0..buffs).map(|_i| {
            (
                PinnedHostBuffer::new(buf_size as usize).unwrap(),
                Arc::new(ThoCudaStream::new(0)),
            )
        })));

        for tensor_kind in [
            Kind::Uint8,
            Kind::Int16,
            Kind::Int,
            Kind::Int64,
            Kind::Double,
        ] {
            let src_t = Tensor::rand(&[10, 1], (Float, Cuda(0)))
                .multiply_scalar(10.)
                .totype(tensor_kind);

            println!("src_t=\t{src_t}");

            let dst = copy_d2h_from_ptr(
                tensor_kind,
                buf_size,
                &bufs_pool,
                src_t.data_ptr() as usize,
                &src_t.size(),
            ).await?;

            println!("dst=\t{dst}");

            let dst = copy_h2d_from_ptr(
                tensor_kind,
                buf_size,
                &bufs_pool,
                1,
                dst.data_ptr() as usize,
                &dst.size(),
            ).await?;

            assert!(src_t.allclose(&dst, 0.01, 0.01, false));
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 64)]
    async fn heavy_one_copy_types() -> Result<(), ThoError>{
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        let cu_ctx = ThoCudaContext::new(0).expect("failed to create CUDA context");
        cu_ctx
            .bind_to_thread()
            .expect("failed to bind CUDA context for heavy_one_copy_types");

        let (free_mem, _total_mem) = cuda_mem();

        let init_type = Int64;

        let elts_in_mem = free_mem / init_type.elt_size_in_bytes() as usize; //123456; //25000; //123456; //
        let dim_1 = 4i64;
        let input_n = ((elts_in_mem as f64 * 0.1) as usize / dim_1 as usize) as i64;

        let buffs = 64;
        let elts_in_buff = elts_in_mem / buffs;

        let buf_size = elts_in_buff * init_type.elt_size_in_bytes() + 17; //64 * 1024 * 1024; // total_size/ buffs as i64;
        let bufs_pool = Arc::new(Pool::from_iter((0..buffs).map(|_i| {
            (
                PinnedHostBuffer::new(buf_size as usize).unwrap(),
                Arc::new(ThoCudaStream::new(0)),
            )
        })));

        for tensor_kind in [
            Kind::Uint8,
            Kind::Int16,
            Kind::Int,
            Kind::Int64,
            Kind::Double,
        ] {
            let src_t = lin_for_size(1., [input_n, dim_1], tensor_kind, &Cuda(0));

            println!("src_t=\t{src_t}");

            let dst = copy_d2h_from_ptr(
                tensor_kind,
                buf_size,
                &bufs_pool,
                src_t.data_ptr() as usize,
                &src_t.size(),
            ).await?;

            println!("dst=\t{dst}");

            let dst = copy_h2d_from_ptr(
                tensor_kind,
                buf_size,
                &bufs_pool,
                1,
                dst.data_ptr() as usize,
                &dst.size(),
            ).await?;

            assert!(src_t.allclose(&dst, 0.01, 0.01, false));
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_copy_d2h_from_ptr_reports_error() {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        let cu_ctx = ThoCudaContext::new(0).expect("failed to create CUDA context");
        cu_ctx
            .bind_to_thread()
            .expect("failed to bind CUDA context for error test");

        // Setup a small pool
        let buffs = 2usize;
        let buf_size = 1024usize;
        let bufs_pool = Arc::new(Pool::from_iter((0..buffs).map(|_| {
            (
                PinnedHostBuffer::new(buf_size).unwrap(),
                Arc::new(ThoCudaStream::new(0)),
            )
        })));

        // Use a CPU pointer as if it were a device pointer — should error
        let bogus_cpu_tensor = Tensor::zeros([4], (Kind::Int, Cpu));
        let bogus_device_ptr = bogus_cpu_tensor.data_ptr() as usize; // not a CUdeviceptr

        let res = super::copy_d2h_from_ptr(
            Kind::Int,
            buf_size,
            &bufs_pool,
            bogus_device_ptr,
            &bogus_cpu_tensor.size(),
        )
        .await;
        assert!(
            res.is_err(),
            "expected copy_d2h_from_ptr_result to error for invalid device ptr"
        );
    }

    // #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn compare_unfold() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        for mem_k in [0.1, 0.5, 1.] {
            for buffs in [2, 4, 8, 16, 32] {
                let copy_ctx = CopyCtx::make_instance(mem_k, buffs);

                let mut copy_unfold_time = 0;
                let mut unfold_copy_time = 0;

                let runs = 100;
                let size = [20, 1_000_000];
                let t = Tensor::rand(&size, FLOAT_CPU);

                for i in 0..runs * 2 {
                    if i % 2 == 0 {
                        let t_0 = Instant::now();

                        let cuda_t = copy_ctx
                            .copy_from_ptr(t.data_ptr() as usize, &t.size(), 1, t.kind())
                            .await?;
                        let unfolded = cuda_t.unfold(1, 100, 10).contiguous();

                        copy_unfold_time += t_0.elapsed().as_millis();
                    } else {
                        let t_0 = Instant::now();
                        let t = t.unfold(1, 100, 10).contiguous();

                        let cuda_t = copy_ctx
                            .copy_from_ptr(t.data_ptr() as usize, &t.size(), 1, t.kind())
                            .await;

                        unfold_copy_time += t_0.elapsed().as_millis();
                    }
                }

                debug!(
                    "mem_k={mem_k} buffs={buffs} copy_unfold_time={} unfold_copy_time={}",
                    copy_unfold_time / runs,
                    unfold_copy_time / runs
                );

                assert!(copy_unfold_time < unfold_copy_time);
            }
        }

        Ok(())
    }
    
    #[test]
    fn roundtrip_pack_unpack() -> Result<(), ThoError> {
        use tch::{Device, Kind, Tensor};

        // try CPU; switch to Device::Cuda(0) if you want to test on GPU
        let dev = Device::Cpu;

        let n = 1_00; //0_003i64;
        let mask = Tensor::randint( 2, [n], (Kind::Bool, dev));
        let packed = pack_bits_bool(&mask)?;

        let mask_size = mask.numel()*mask.kind().elt_size_in_bytes();
        let packed_size = packed.numel()*packed.kind().elt_size_in_bytes();

        let unpacked = unpack_bits_u8(&packed, Some(n));

        let unpacked_size = unpacked.numel()*unpacked.kind().elt_size_in_bytes();


        assert!(mask.allclose(&unpacked, 0.01, 0.01, false), "roundtrip mismatch");
        // assert!(mask.allclose(&unpack_bits_u8(&packed, None), 0.01, 0.01, false), "roundtrip mismatch");
        Ok(())
    }

    #[tokio::test]
    async fn roundtrip_pack_unpack_copy() -> Result<(), ThoError>{
        use tch::{Device, Kind, Tensor};

        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        GlobalCtx::init_devices(2);

        // try CPU; switch to Device::Cuda(0) if you want to test on GPU
        let dev = ThoDevice::from(Device::Cpu);

        let n = 12_345; //0_003i64;
        let channels = 20;
        let mask = Tensor::randint( 2, [channels * n], (Kind::Bool, Cuda(0))).view((channels, n));
        let size = mask.size();
        let packed_mask = pack_bits_bool(&mask)?;
        let packed = CopyCtx::instance(Cuda(0)).copy(&packed_mask, 0, Uint8).await?;

        let mask_size = mask.numel()*mask.kind().elt_size_in_bytes();
        let packed_size = packed.numel()*packed.kind().elt_size_in_bytes();

        let numelts = size.iter().fold(1, |mut a, v| {
            a *= v;
            a
        });
        let unpacked = unpack_bits_u8(&packed, Some(numelts)).view((channels, n));

        let unpacked_size = unpacked.numel()*unpacked.kind().elt_size_in_bytes();

        // println!("mask=\n{mask} unpacked=\n{unpacked} \n={numelts}");

        assert!(mask.to_device(Cpu).allclose(&unpacked, 0.01, 0.01, false), "roundtrip mismatch");
        // assert!(mask.allclose(&unpack_bits_u8(&packed, None), 0.01, 0.01, false), "roundtrip mismatch");

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_cuda_stream(){

    }

}

pub fn with_cuda_stream<F, R>(tho_device_idx:usize, mut f:F) -> (R, Option<ThoCudaStream>)
where
    F:FnMut() -> R
{

    if tho_device_idx > 0 {
        let motifs_stream = ThoCudaStream::new((tho_device_idx - 1) as i32); //TODO
        let _guard = ThoStreamGuard::set(&motifs_stream);

        (f(), Some(motifs_stream))
    } else {
        (f(), None)
    }
}

tokio::task_local! {
    pub static CURRENT_GUARD: ThoStreamGuard;
}