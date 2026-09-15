use crate as thoosa_common;
use crate::runner::global_ctx::GlobalCtx;
use crate::tho_device::ThoDevice;
use cust::sys::CU_MEMHOSTALLOC_WRITECOMBINED;
use log::debug;
use std::ffi::c_void;
use std::ops::Add;
use std::sync::Arc;
use std::time::Duration;
use tch::Device::Cpu;
use tch::Kind::Float;
use tch::nn::VarStore;
use tch::{Device, Kind, Tensor};
use thoosa_shared::futures_limits::{COPY_D2H, COPY_H2D};
use thoosa_shared::os_thread_id;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::{Cause, FuturesOrdering, ThoError, collect_futures_with_progress};
use thoosa_shims::context::ThoCudaContext;
use thoosa_shims::cuda_result::check_cuda_status;
use thoosa_shims::stream::{CUDA_SUCCESS, ThoCudaStream, ThoStreamGuard};
use tokio::task::{JoinError, spawn_blocking};
use tokio_utils::Pool;
use torch_sys::at_grad_set_enabled;
use tub::Guard;

// One contiguous byte range in the source/destination tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CopyChunk {
    pub(crate) index: usize,
    pub(crate) offset: usize,
    pub(crate) len: usize,
}

// Split a tensor byte range into chunks that fit in one pinned host buffer checkout.
pub(crate) fn make_chunks(tensor_mem_size: usize, buf_size: usize, label: &str) -> Result<Vec<CopyChunk>, ThoError> {
    if tensor_mem_size == 0 {
        return Ok(Vec::new());
    }
    if buf_size == 0 {
        return Err(Cause::SomethingElse(format!("{label} requires a non-zero copy buffer for {tensor_mem_size} bytes")).into());
    }

    let chunks = tensor_mem_size / buf_size;
    let last = tensor_mem_size % buf_size;

    Ok((0..chunks)
        .map(|i| CopyChunk {
            index: i,
            offset: i * buf_size,
            len: buf_size,
        })
        .chain((last != 0).then_some(CopyChunk {
            index: chunks,
            offset: chunks * buf_size,
            len: last,
        }))
        .collect())
}

// Preserve the existing ThoError shape when a blocking copy task panics or is cancelled.
fn join_error_to_tho_error(e: JoinError) -> ThoError {
    Cause::External {
        type_name: std::any::type_name::<JoinError>(),
        source: Arc::new(e),
    }.into()
}

pub struct PinnedHostBuffer {
    ptr: *mut u8,
    len: usize,
}

impl PinnedHostBuffer {
    pub fn new(len: usize, flags: u32) -> Result<Self, ThoError> {

        let mut raw: *mut c_void = std::ptr::null_mut();
        // let flags = CU_MEMHOSTALLOC_WRITECOMBINED;
        let res = unsafe {
            thoosa_shims::memcpy::tho_cu_mem_host_alloc(&mut raw, len, flags)
        };

        check_cuda_status(res, "cuMemHostAlloc")?;

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

        if self.ptr.is_null() {
            return;
        }

        let res = unsafe {
            thoosa_shims::memcpy::tho_cu_mem_free_host(self.ptr as *mut c_void)
        };

        if let Err(err) = check_cuda_status(res, "cuMemFreeHost") {
            log::error!("{err:?}");
        }
    }
}
// test-only imports moved into test modules


//free_mem, total_mem
pub fn cuda_mem_for_device(cuda_device: usize) -> (usize, usize) {
    use cust::sys::cuMemGetInfo_v2;

    let ctx = ThoCudaContext::new(cuda_device as i32).expect("failed to create CUDA context");
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

pub fn cuda_mem() -> (usize, usize) {
    cuda_mem_for_device(0)
}

///for cases when only size and type is known
pub fn tensor_mem_size(init_type: Kind, size: &[i64]) -> usize {//TODO rename to not confuse with one from thoosa_common::utils and move
    size.iter().fold(1, |a, v| a * (*v as usize)) * init_type.elt_size_in_bytes()
}

fn ensure_copy_d2h_source_is_device_memory(src_ptr: usize) -> Result<(), ThoError> {
    let mut is_device_memory = 0i32;
    let result = unsafe {
        thoosa_shims::memcpy::tho_cu_pointer_is_device_memory(
            &mut is_device_memory,
            src_ptr as thoosa_shims::memcpy::CuDevicePtr,
        )
    };


    check_cuda_status(result, "cuPointerGetAttribute")?;

    if is_device_memory == 0 {
        return Err(Cause::SomethingElse(format!(
            "copy_d2h_from_ptr expected CUDA device memory but src_ptr={src_ptr} is not device memory"
        )).into());
    }

    Ok(())
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

    assert!(src_t.as_ref().device().ne(&Cpu));

    let tensor_mem_size = tensor_mem_size(init_type, &src_t.as_ref().size());
    if tensor_mem_size < buf_size {
        return Ok(src_t.as_ref().to(Cpu))
    }


    let src_ptr: usize = src_t.as_ref().data_ptr() as usize;
    let src_size = src_t.as_ref().size();

    track_await(copy_d2h_from_ptr(init_type, buf_size, &bufs_pool, src_ptr, &src_size)).await
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="cuda_utils::move_d2h"))]
pub async fn move_d2h(
    init_type: Kind,
    src_t: Tensor,
    buf_size: usize,
    bufs_pool: &Arc<Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>>,
) -> Result<Tensor, ThoError> {

    assert!(src_t.as_ref().device().ne(&Device::Cpu));

    let tensor_mem_size = tensor_mem_size(init_type, &src_t.as_ref().size());
    if tensor_mem_size < buf_size {
        return Ok(src_t.to(Cpu))
    }


    let src_ptr: usize = src_t.data_ptr() as usize;
    let src_size = src_t.size();

    let res = track_await(copy_d2h_from_ptr(init_type, buf_size, &bufs_pool, src_ptr, &src_size)).await;
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
    let dst_t = Tensor::f_empty(src_size, (init_type, Cpu))?;
    track_await(copy_d2h_to_ptr(init_type, buf_size_bytes, bufs_pool, src_ptr, src_size, dst_t.data_ptr() as usize)).await?;
    debug!("copy_d2h_from_ptr: dst_t={dst_t:?} dst_t.device={:?}", dst_t.device());
    Ok(dst_t)
}

/// Copies a CUDA tensor into caller-owned CPU storage. The caller must keep both pointer ranges alive and guarantee that `dst_ptr` covers the byte size implied by `init_type` and `src_size`.
#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="cuda_utils::copy_d2h_to_ptr"))]
pub async fn copy_d2h_to_ptr(
    init_type: Kind,
    buf_size_bytes: usize,
    bufs_pool: &Arc<Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>>,
    src_ptr: usize,
    src_size: &[i64],
    dst_ptr: usize,
) -> Result<(), ThoError> {
    let tensor_mem_size = tensor_mem_size(init_type, src_size);
    let chunks = make_chunks(tensor_mem_size, buf_size_bytes, "copy_d2h_to_ptr")?;
    let chunk_len = chunks.len();
    debug!("copy_d2h_to_ptr: src_size={src_size:?} chunks={chunks:?} tensor_mem_size={tensor_mem_size} buf_size={buf_size_bytes}");

    track_await(collect_futures_with_progress(
        chunks.into_iter().map(|chunk| {
            let pool = Arc::clone(bufs_pool);
            async move {
                let buf_stream = track_await(pool.acquire()).await;
                let result = track_await(spawn_blocking(move || {
                    let buf_ptr = buf_stream.0.as_mut_ptr() as usize;
                    let stream = Arc::clone(&buf_stream.1);
                    let _guard = ThoStreamGuard::new(stream.as_ref())?;
                    let src_off_ptr = src_ptr.checked_add(chunk.offset).ok_or_else(|| ThoError::new("D2H source pointer offset overflow"))? as thoosa_shims::memcpy::CuDevicePtr;
                    let dst_off_ptr = dst_ptr.checked_add(chunk.offset).ok_or_else(|| ThoError::new("D2H destination pointer offset overflow"))?;
                    debug!("copy_d2h_to_ptr: starting {} thread_name={:?} src_off_ptr={src_off_ptr:?} off={}", chunk.index, os_thread_id(), chunk.offset);

                    let result = {
                        let span = tracing::debug_span!("copy_d2h_to_ptr", agg_perf = "cuda_utils::copy_d2h_to_ptr/tho_cu_memcpy_dtoh_async", len = chunk.len);
                        let _enter = span.enter();
                        unsafe { thoosa_shims::memcpy::tho_cu_memcpy_dtoh_async(buf_ptr as *mut c_void, src_off_ptr, chunk.len, stream.as_mut_ptr()) }
                    };
                    if result != CUDA_SUCCESS {
                        debug!("tho_cu_memcpy_dtoh_async failed at chunk {} with {result:?}. buf_ptr={buf_ptr} src_off_ptr={src_off_ptr} c_buf_size={}", chunk.index, chunk.len);
                        check_cuda_status(result, "tho_cu_memcpy_dtoh_async")?;
                    }
                    stream.synchronize()?;

                    let span = tracing::debug_span!("copy_d2h_to_ptr", agg_perf = "cuda_utils::copy_d2h_to_ptr/copy_nonoverlapping", len = chunk.len);
                    let _enter = span.enter();
                    unsafe {
                        // SAFETY: the caller guarantees the destination range, chunk offsets cover disjoint validated tensor ranges, and the pinned buffer remains leased here.
                        std::ptr::copy_nonoverlapping(buf_ptr as *const u8, dst_off_ptr as *mut u8, chunk.len);
                    }
                    drop(buf_stream);
                    Ok::<(), ThoError>(())
                })).await.map_err(join_error_to_tho_error).and_then(|result| result);
                Ok::<_, ThoError>(result)
            }
        }),
        chunk_len,
        FuturesOrdering::Unordered,
        COPY_D2H,
    )).await.into_iter().collect::<Result<(), ThoError>>()?;
    Ok(())
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="cuda_utils::copy_h2d"))]
pub async fn copy_h2d<R: AsRef<Tensor>>(
    init_type: Kind,
    src_t: &R,
    buf_size: usize,
    bufs_pool: &Arc<Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>>,
    dst_tho_device_idx: usize,
) -> Result<Tensor, ThoError> {
    assert!(src_t.as_ref().device().eq(&Cpu));
    assert!(dst_tho_device_idx > 0);

    let tensor_mem_size = tensor_mem_size(init_type, &src_t.as_ref().size());
    if tensor_mem_size < buf_size {
        return Ok(src_t.as_ref().to(ThoDevice::from(dst_tho_device_idx).into()))
    }


    let src_ptr = src_t.as_ref().data_ptr() as usize; // CPU mem
    let src_size = src_t.as_ref().size();

    track_await(copy_h2d_from_ptr(
        init_type,
        buf_size,
        bufs_pool,
        dst_tho_device_idx,
        src_ptr,
        &src_size,
    )).await
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="cuda_utils::move_h2d"))]
pub async fn move_h2d(
    init_type: Kind,
    src_t: Tensor,
    buf_size: usize,
    bufs_pool: &Arc<Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>>,
    dst_tho_device_idx: usize,
) -> Result<Tensor, ThoError> {
    assert!(src_t.as_ref().device().eq(&Cpu));
    assert!(dst_tho_device_idx > 0);

    let tensor_mem_size = tensor_mem_size(init_type, &src_t.as_ref().size());
    if tensor_mem_size < buf_size {
        return Ok(src_t.to(ThoDevice::from(dst_tho_device_idx).into()))
    }

    let src_ptr = src_t.data_ptr() as usize; // CPU mem
    let src_size = src_t.size();

    let res = track_await(copy_h2d_from_ptr(
        init_type,
        buf_size,
        bufs_pool,
        dst_tho_device_idx,
        src_ptr,
        &src_size,
    )).await;
    drop(src_t); // keep source alive until the copy finishes
    res
}


#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="cuda_utils::copy_h2d_from_ptr"))]
pub async fn copy_h2d_from_ptr(
    init_type: Kind,
    buf_size: usize,
    bufs_pool: &Arc<Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>>,
    dst_tho_device_idx: usize,
    src_ptr: usize,
    size: &[i64],
) -> Result<Tensor, ThoError> {
    let tensor_mem_size = tensor_mem_size(init_type, &size);
    debug!("copy_h2d_from_ptr: src_t.size={:?} tensor_mem_size={tensor_mem_size} buf_size={buf_size}",size);

    // Keep one future per chunk so a limit increase can start more copy work immediately.
    let chunks = make_chunks(tensor_mem_size, buf_size, "copy_h2d_from_ptr")?;
    let chunks_len = chunks.len();

    // destination tensor lives on GPU 0
    let dst_t = Tensor::f_empty(size, (init_type, Device::Cuda(dst_tho_device_idx - 1)))?;

    unsafe {
        // base pointers
        let dst_ptr = dst_t.data_ptr() as usize; // GPU mem

        // CUdriver needs CUdeviceptr for the destination side
        let dst_base = dst_ptr as thoosa_shims::memcpy::CuDevicePtr;

        track_await(collect_futures_with_progress(
            chunks.into_iter().map(|chunk| {
                let pool = Arc::clone(bufs_pool);

                async move {
                    let buf_stream = track_await(pool.acquire()).await;

                    let result = track_await(spawn_blocking(move || {

                            let buf_ptr = buf_stream.0.as_mut_ptr() as usize;
                            let stream = Arc::clone(&buf_stream.1);
                            let _guard = ThoStreamGuard::new(stream.as_ref())?;

                        // for chunk in chunk_group {
                            let src_off_ptr = src_ptr.add(chunk.offset);
                            let dst_off_ptr = dst_base + chunk.offset as u64;

                            debug!("copy_h2d_from_ptr: starting {} thread_name={:?}", chunk.index, os_thread_id());
                            debug!("copy_h2d_from_ptr: src_off_ptr={src_off_ptr} buf_ptr={buf_ptr} c_buf_size={} thread_name={:?}", chunk.len, os_thread_id());
                            {
                                let span = tracing::debug_span!(
                                    "copy_d2h_from_ptr",
                                    agg_perf = "cuda_utils::copy_h2d_from_ptr/copy_nonoverlapping",
                                    len = chunk.len,
                                );
                                let _enter = span.enter();
                                std::ptr::copy_nonoverlapping(
                                    src_off_ptr as *const u8,
                                    buf_ptr as *mut u8,
                                    chunk.len,
                                );
                            }

                            let result = {
                                let span = tracing::debug_span!(
                                    "copy_d2h_from_ptr",
                                    agg_perf = "cuda_utils::copy_h2d_from_ptr/tho_cu_memcpy_htod_async",
                                    len = chunk.len,
                                );
                                let _enter = span.enter();
                                thoosa_shims::memcpy::tho_cu_memcpy_htod_async(
                                    dst_off_ptr as thoosa_shims::memcpy::CuDevicePtr,
                                    buf_ptr as *const c_void,
                                    chunk.len,
                                    stream.as_mut_ptr(),
                                )
                            };
                            if result != CUDA_SUCCESS {
                                debug!("tho_cu_memcpy_htod_async failed at chunk {} with {result:?}. buf_ptr={buf_ptr} dst_off_ptr={dst_off_ptr} c_buf_size={}", chunk.index, chunk.len);
                                check_cuda_status(result, "tho_cu_memcpy_htod_async")?;
                            }
                            stream.synchronize()?;
                            debug!("copy_h2d_from_ptr: src_off_ptr={src_off_ptr} buf_ptr={buf_ptr} c_buf_size={} thread_name={:?} done", chunk.len, os_thread_id());
                        // }

                            drop(buf_stream); // return buffer/stream to pool

                            Ok::<(), ThoError>(())


                    })).await.map_err(join_error_to_tho_error).and_then(|result| result);

                    Ok::<_, ThoError>(result)
                }
            }),
            chunks_len,
            FuturesOrdering::Unordered,
            COPY_H2D,
        )).await
            .into_iter()
            .collect::<Result<(), ThoError>>()?;
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


pub fn fail_with_unrecoverable_error(){
    debug!("test_copy_ipc_cuda_worker_propagates_oom: do_run");
    // run with: CUDA_LAUNCH_BLOCKING=1 RUST_BACKTRACE=1 cargo run
    let device = Device::Cuda(0);

    // Shape: [batch, classes] => 3 valid classes: 0, 1, 2.
    let logits = Tensor::zeros([4, 3], (Float, device));

    // Invalid target: class id 3 is out of range for 3 classes.
    // Older tch versions may need Tensor::of_slice instead of from_slice.
    let targets = Tensor::from_slice(&[0i64, 1, 2, 3]).to_device(device);
    // let targets = Tensor::of_slice(&[0i64, 1, 2, 3]).to_device(device);

    let loss = logits.cross_entropy_for_logits(&targets);

    // Force synchronization / materialization.
    // The actual CUDA error is often reported here, not necessarily at launch,
    // because CUDA launches are asynchronous.
    println!("loss = {}", loss.double_value(&[]));

    // Usually this will also fail after the context is poisoned.
    let x = Tensor::zeros([1], (Float, device));
    println!("after poison = {}", x.double_value(&[0]));
}

#[cfg(test)]
#[allow(unused_imports, unused_variables, dead_code)]
mod tests {
    use crate::copy_ctx::CopyCtx;
    use crate::cuda_utils::{
        PinnedHostBuffer,
        copy_d2h_from_ptr,
        copy_h2d_from_ptr,
        cuda_mem,
        equal_bytes,
        pack_bits_bool,
        speed_gb_per_s,
        unpack_bits_u8,
    };
    use crate::dims::AnyDims;
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::tho_device::ThoDevice;
    use crate::tho_tensor::ThoTensor;
    use crate::utils::lin_for_size;
    use cust::sys::CU_MEMHOSTALLOC_WRITECOMBINED;
    use futures_util::StreamExt;
    use log::{debug, info};
    use rand::random_range;
    use std::cell::UnsafeCell;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering::Relaxed;
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::{Float, Int64, Uint8};
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::{Kind, Tensor};
    use thoosa_shared::ThoError;
    use thoosa_shared::applicables_runner::Output;
    use thoosa_shared::futures_limits::{COPY_D2H, COPY_H2D, futures_limits};
    use thoosa_shared::perf_logging::{PERF, format_perf_map};
    use thoosa_shared::traits::SimpleId;
    use thoosa_shims::cpp_log::init_cpp_logging;
    use thoosa_shims::{context::ThoCudaContext, stream::ThoCudaStream};
    use tokio::task::JoinSet;
    use tokio::time::Instant;
    use tokio_utils::Pool;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn copy_chunks_rejects_zero_buffer_for_nonempty_copy() {
        assert!(super::make_chunks(1, 0, "test").is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 64)]
    #[cfg_attr(miri, ignore)]
    #[allow(dead_code)]
    async fn heavy_test_copy_from_ptr() -> Result<(), ThoError> {
        let cu_ctx = ThoCudaContext::new(0 as i32).expect("failed to create CUDA context");
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");

            let mut ctx: Option<UnsafeCell<crate::copy_ctx::CopyCtx>> = None;

            for mem_k in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.] {
                for buffs in [2, 4, 8, 16, 32, 64] {
                    if ctx.is_none() {
                        ctx = Some(UnsafeCell::new(CopyCtx::make_instance_for_cuda_device(0, mem_k, buffs)?));
                    } else {
                        ctx.as_mut().unwrap().get_mut().rebuild(mem_k, buffs)?
                    }

                    let (d2h, h2d) = run_test_copy_from_ptr(&*ctx.as_ref().unwrap().get(), 10, 10).await;

                    info!("\n mem_k={mem_k}, buffs={buffs} ctx={ctx:?} d2h speed: {d2h:#2} Gb/s, h2d speed {h2d:#2} Gb/s");
                }
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 64)]
    async fn heavy_copy_from_ptr_once() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        futures_limits().force(COPY_H2D, 10);
        futures_limits().force(COPY_D2H, 10);

        let ctx:&'static CopyCtx = Box::leak(Box::new(CopyCtx::make_instance_for_cuda_device(0, 0.2, 16)?));

        let (d2h, h2d) = run_test_copy_from_ptr(ctx, 3, 10).await;
        info!("\n Copy d2h speed: {d2h:#2} GB/s, h2d speed {h2d:#2} GB/s");

        info!("\n{}", format_perf_map(&PERF));
        Ok(())
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="run_test_copy_from_ptr"))]
    async fn run_test_copy_from_ptr(ctx: &'static CopyCtx, threads: u8, runs: u8) -> (f64, f64) {
        let (_free_mem, total_mem) = cuda_mem();

        let init_type = Float;

        let total_size = (total_mem as f64 * 0.8) as u64 / init_type.elt_size_in_bytes() as u64; //123456; //25000; //123456; //
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

        (
            speed_gb_per_s(total_copied_mem_size_kb, d2h_time.load(Relaxed)),
            speed_gb_per_s(total_copied_mem_size_kb, h2d_time.load(Relaxed))
        )
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="run_copy_"))]
    async fn run_copy_(
        _thread_i: u8,
        input_n: i64,
        dim_1: i64,
        runs: u8,
        init_type: Kind,

        d2h_time: Arc<AtomicU64>,
        h2d_time: Arc<AtomicU64>,
        ctx: &CopyCtx,
    ) -> Result<(), ThoError> {

        for _t in 0..runs {
            let dim_0 = random_range(0.5 * input_n as f64..1.5 * input_n as f64) as i64;
            let mut src_t = Tensor::rand([dim_0, dim_1], (Float, Cpu));

            let mut src_ptr = src_t.data_ptr() as usize;
            let mut src_size = src_t.size();

            let inst_0 = Instant::now();
            let tensor = ctx.copy_from_ptr(src_ptr, &src_size, 1, init_type).await?;
            h2d_time.fetch_add(inst_0.elapsed().as_millis() as u64, Relaxed);

            (src_ptr, src_size) = (tensor.data_ptr() as usize, tensor.size());


            let inst_0 = Instant::now();
            let tensor = ctx.copy_from_ptr(src_ptr, &src_size, 0, init_type).await?;
            d2h_time.fetch_add(inst_0.elapsed().as_millis() as u64, Relaxed);

            (_, src_size) = (tensor.data_ptr() as usize, tensor.size());

            assert!(equal_bytes(&tensor, &src_t));

            src_t = tensor;
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 64)]
    async fn test_small_one_copy_types() ->Result<(), ThoError>{
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
            init_cpp_logging();
        }
        ();

        let cu_ctx = ThoCudaContext::new(0).expect("failed to create CUDA context");
        cu_ctx
            .bind_to_thread()
            .expect("failed to bind CUDA context for test_small_one_copy_types");

        let (free_mem, _total_mem) = cuda_mem();

        let init_type = Int64;

        let elts_in_mem = (0.1 * free_mem as f32) as usize / init_type.elt_size_in_bytes(); //123456; //25000; //123456; //

        let buffs = 16;
        let elts_in_buff = elts_in_mem / buffs;

        let buf_size = elts_in_buff * init_type.elt_size_in_bytes() + 17; //64 * 1024 * 1024; // total_size/ buffs as i64;


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
                &Arc::new(Pool::from_iter((0..buffs).map(|_i| {
                    (
                        PinnedHostBuffer::new(buf_size, 0).unwrap(),
                        Arc::new(ThoCudaStream::from_pool(0, true).unwrap()),
                    )
                }))),
                src_t.data_ptr() as usize,
                &src_t.size(),
            ).await?;

            println!("dst=\t{dst}");

            let dst = copy_h2d_from_ptr(
                tensor_kind,
                buf_size,
                &Arc::new(Pool::from_iter((0..buffs).map(|_i| {
                    (
                        PinnedHostBuffer::new(buf_size, CU_MEMHOSTALLOC_WRITECOMBINED).unwrap(),
                        Arc::new(ThoCudaStream::from_pool(0, true).unwrap()),
                    )
                }))),
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
        let input_n = ((elts_in_mem as f64 * 0.7) as usize / dim_1 as usize) as i64;

        let buffs = 16;
        let elts_in_buff = elts_in_mem / buffs;

        let buf_size = elts_in_buff * init_type.elt_size_in_bytes() + 17; //64 * 1024 * 1024; // total_size/ buffs as i64;

        for tensor_kind in [
            Kind::Uint8,
            Kind::Int16,
            Kind::Int,
            Kind::Int64,
            Kind::Double,
        ] {
            let src_t = lin_for_size(1., [input_n, dim_1], tensor_kind, &Cuda(0))?;

            println!("src_t=\t{src_t}");

            let dst = copy_d2h_from_ptr(
                tensor_kind,
                buf_size,
                &Arc::new(Pool::from_iter((0..buffs).map(|_i| {
                    (
                        PinnedHostBuffer::new(buf_size as usize, CU_MEMHOSTALLOC_WRITECOMBINED).unwrap(),
                        Arc::new(ThoCudaStream::from_pool(0, true).unwrap()),
                    )
                }))),
                src_t.data_ptr() as usize,
                &src_t.size(),
            ).await?;

            println!("dst=\t{dst}");

            let dst = copy_h2d_from_ptr(
                tensor_kind,
                buf_size,
                &Arc::new(Pool::from_iter((0..buffs).map(|_i| {
                    (
                        PinnedHostBuffer::new(buf_size as usize, 0).unwrap(),
                        Arc::new(ThoCudaStream::from_pool(0, true).unwrap()),
                    )
                }))),
                1,
                dst.data_ptr() as usize,
                &dst.size(),
            ).await?;

            assert!(src_t.allclose(&dst, 0.01, 0.01, false));
        }

        Ok(())
    }

    // #[tokio::test(flavor = "multi_thread", worker_threads = 8)] TODO add build config to enable asserts
    async fn test_copy_d2h_from_ptr_reports_error() {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
            // let _lib = libloading::Library::new("/home/bu/dvl/rust/thoosa/crates/target/debug/libthoosa_shims.so").expect("no libtorch_cuda.so");
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
                PinnedHostBuffer::new(buf_size, 0).unwrap(),
                Arc::new(ThoCudaStream::from_pool(0, true).unwrap()),
            )
        })));

        // Use a CPU pointer as if it were a device pointer — should error
        let bogus_cpu_tensor = Tensor::rand(&[1000, 1000], FLOAT_CPU);
        debug!("bogus_cpu_tensor={bogus_cpu_tensor:?} d={:?}", bogus_cpu_tensor.device());

        let bogus_device_ptr = bogus_cpu_tensor.data_ptr() as usize; // not a CUdeviceptr

        let res = super::copy_d2h_from_ptr(
            Kind::Int,
            buf_size,
            &bufs_pool,
            bogus_device_ptr,
            &bogus_cpu_tensor.size(),
        )
        .await;

        debug!("res={res:?} device={:?}", res.as_ref().map(|v|v.device()));

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
                let copy_ctx = CopyCtx::make_instance_for_cuda_device(0, mem_k, buffs)?;

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
        let global_mock_ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        GlobalCtx::init_devices(2)?;

        // try CPU; switch to Device::Cuda(0) if you want to test on GPU
        let dev = ThoDevice::from(Device::Cpu);

        let n = 12_345; //0_003i64;
        let channels = 20;
        let mask = Tensor::randint( 2, [channels * n], (Kind::Bool, Cuda(0))).view((channels, n));
        let size = mask.size();
        let packed_mask = pack_bits_bool(&mask)?;
        let packed = GlobalCtx::global().cuda_actor(Cuda(0)).copy(packed_mask.into(), 0.into()).await?.t();

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

}
