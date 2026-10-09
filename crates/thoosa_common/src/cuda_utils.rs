use crate as thoosa_common;
use crate::runner::global_ctx::GlobalCtx;
use crate::tho_device::ThoDevice;
use anyhow::Context as _;
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
use thoosa_shared::{cuda_cause_from_error_code, os_thread_id};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::{CudaActivityScope, FuturesOrdering, ThoError, collect_futures_with_progress};
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
        return Err(ThoError::new(format!("{label} requires a non-zero copy buffer for {tensor_mem_size} bytes")));
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

// Preserve both task failures and CUDA errors for the caller's recovery decision.
fn joined_copy_result<T>(joined: Result<Result<T, ThoError>, JoinError>) -> Result<T, ThoError> {
    Ok(joined.context("join blocking CUDA copy task")??)
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

        if res != 4 &&//CUDA error 4: CUDA runtime/driver is deinitializing; usually process shutdown or teardown ordering
            let Err(err) = check_cuda_status(res, "cuMemFreeHost") {
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
        return Err(ThoError::new(format!(
            "copy_d2h_from_ptr expected CUDA device memory but src_ptr={src_ptr} is not device memory"
        )));
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

// Quarantining a failed stream removes a pool entry permanently. Wake every
// waiter on failure so exhausting those entries cannot strand other transfers.
pub struct CopyBufferPool {
    buffers: Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>,
    failure: tokio::sync::watch::Sender<Option<ThoError>>,
}

impl CopyBufferPool {
    pub fn new(buffers: impl IntoIterator<Item = (PinnedHostBuffer, Arc<ThoCudaStream>)>) -> Self {
        Self { buffers: Pool::from_iter(buffers), failure: tokio::sync::watch::channel(None).0 }
    }

    pub fn remaining_capacity(&self) -> usize { self.buffers.remaining_capacity() }

    fn check(&self) -> Result<(), ThoError> {
        match self.failure.borrow().as_ref() { Some(error) => Err(error.clone()), None => Ok(()) }
    }

    fn fail(&self, error: ThoError) {
        self.failure.send_if_modified(|failure| {
            if failure.is_some() { false } else { *failure = Some(error); true }
        });
    }

    async fn acquire(&self) -> Result<Guard<(PinnedHostBuffer, Arc<ThoCudaStream>)>, ThoError> {
        let mut failure = self.failure.subscribe();
        self.check()?;
        let buffer = tokio::select! {
            buffer = self.buffers.acquire() => buffer,
            _ = failure.changed() => return Err(failure.borrow().clone().expect("copy pool failure")),
        };
        self.check()?;
        Ok(buffer)
    }
}

// Raw addresses are only usable while their allocations remain owned. Tensor is
// Send but not Sync, so hide its handle behind a mutex; chunks use disjoint byte
// ranges and never access the handle itself.
struct CopyStorage {
    address: usize,
    byte_len: usize,
    allocation: parking_lot::Mutex<CopyAllocation>,
}

enum CopyAllocation {
    Tensor(Tensor),
    External(Box<dyn Send>),
}

impl CopyStorage {
    fn tensor(tensor: Tensor) -> Result<Self, ThoError> {
        if !tensor.is_contiguous() { return Err(ThoError::new("chunked CUDA copy requires contiguous tensor storage")); }
        let byte_len = tensor.numel().checked_mul(tensor.kind().elt_size_in_bytes()).ok_or_else(|| ThoError::new("tensor copy byte size overflow"))?;
        let address = tensor.data_ptr() as usize;
        address.checked_add(byte_len).ok_or_else(|| ThoError::new("tensor copy address overflow"))?;
        Ok(Self { address, byte_len, allocation: parking_lot::Mutex::new(CopyAllocation::Tensor(tensor)) })
    }

    /// # Safety
    /// owner must keep the entire range allocated at this address until it is
    /// dropped, even after cancellation. Reads/writes must not race other users;
    /// for shared IPC storage the peer must also retain its allocation until the
    /// request completes or this process exits. A mapping alone does not prevent
    /// the peer from recycling an arena range.
    unsafe fn external(address: usize, byte_len: usize, owner: impl Send + 'static) -> Result<Self, ThoError> {
        if address == 0 && byte_len != 0 { return Err(ThoError::new("null copy allocation")); }
        address.checked_add(byte_len).ok_or_else(|| ThoError::new("copy allocation address overflow"))?;
        Ok(Self { address, byte_len, allocation: parking_lot::Mutex::new(CopyAllocation::External(Box::new(owner))) })
    }
}

// Every queued async chunk and every blocking closure retains this same object.
// Dropping the caller or its JoinSet cannot release either allocation early.
struct CopyTransfer {
    source: CopyStorage,
    destination: CopyStorage,
}

impl CopyTransfer {
    fn new(source: CopyStorage, destination: CopyStorage) -> Result<Arc<Self>, ThoError> {
        if source.byte_len != destination.byte_len { return Err(ThoError::new("copy allocation sizes differ")); }
        Ok(Arc::new(Self { source, destination }))
    }

    fn into_tensor(self: Arc<Self>) -> Result<Tensor, ThoError> {
        let transfer = Arc::try_unwrap(self).map_err(|_| ThoError::new("completed copy storage is still leased"))?;
        match transfer.destination.allocation.into_inner() {
            CopyAllocation::Tensor(tensor) => Ok(tensor),
            CopyAllocation::External(_) => Err(ThoError::new("copy destination is not a tensor")),
        }
    }
}

// An unwind must drain submitted DMA before releasing allocation owners. Only
// an unsuccessful drain quarantines storage; a host panic is not a CUDA fault.
struct CopyCompletion<T> {
    resources: Option<T>,
    submitted: bool,
    pool: Arc<CopyBufferPool>,
    stream: Arc<ThoCudaStream>,
}

impl<T> CopyCompletion<T> {
    fn finish(&mut self, completion: crate::copy_ctx::StreamCompletion) -> Result<(), ThoError> {
        self.submitted = false;
        if !completion.confirmed {
            self.pool.fail(completion.result.as_ref().unwrap_err().clone());
            // Keep all DMA allocations, including a worker mapping, alive when
            // neither synchronization attempt establishes physical completion.
            if let Some(resources) = self.resources.take() { std::mem::forget(resources); }
        }
        completion.result
    }
}

impl<T> Drop for CopyCompletion<T> {
    fn drop(&mut self) {
        if self.submitted {
            let completion = crate::copy_ctx::complete_cuda_stream(self.stream.synchronize(), || self.stream.synchronize());
            if let Err(error) = self.finish(completion) { log::debug!("CUDA copy unwind synchronization failed: {error:#}"); }
        }
    }
}

fn spawn_copy_chunk<R, F>(transfer: Arc<CopyTransfer>, resources: R, copy: F) -> tokio::task::JoinHandle<Result<(), ThoError>>
where
    R: Send + 'static,
    F: FnOnce(&Arc<CopyTransfer>, R) -> Result<(), ThoError> + Send + 'static,
{
    let activity = CudaActivityScope::current();
    spawn_blocking(move || {
        let _activity = activity;
        // Borrowing here forces the blocking closure to retain the whole owner,
        // rather than capturing only the raw address fields it happens to use.
        copy(&transfer, resources)
    })
}

#[derive(Clone, Copy)]
enum CopyDirection { ToCpu, ToCuda }

async fn copy_transfer(
    gate: &Arc<thoosa_shared::CudaActivityGate>,
    transfer: Arc<CopyTransfer>,
    buf_size: usize,
    pool: &Arc<CopyBufferPool>,
    direction: CopyDirection,
) -> Result<(), ThoError> {
    CudaActivityScope::for_operation(gate)?.run(async {
        let label = match direction { CopyDirection::ToCpu => COPY_D2H, CopyDirection::ToCuda => COPY_H2D };
        let chunks = make_chunks(transfer.source.byte_len, buf_size, label)?;
        let count = chunks.len();
        collect_futures_with_progress(chunks.into_iter().map(|chunk| {
            let transfer = Arc::clone(&transfer);
            let pool = Arc::clone(pool);
            async move {
                let buffer = pool.acquire().await?;
                let copy_pool = Arc::clone(&pool);
                let result = spawn_copy_chunk(transfer, buffer, move |transfer, buffer| {
                    let stream = Arc::clone(&buffer.1);
                    let mut completion = CopyCompletion { resources: Some((Arc::clone(transfer), buffer)), submitted: false, pool: copy_pool, stream: Arc::clone(&stream) };
                    let buffer = &completion.resources.as_ref().expect("copy resources").1;
                    if chunk.len > buffer.0.len() { return Err(ThoError::new("copy chunk exceeds pinned buffer")); }
                    let pinned = buffer.0.as_mut_ptr();
                    let _guard = ThoStreamGuard::new(stream.as_ref())?;
                    let source = transfer.source.address + chunk.offset;
                    let destination = transfer.destination.address + chunk.offset;
                    // Range sizes and address overflow were checked at construction;
                    // make_chunks partitions the destination into disjoint ranges.
                    let submitted = match direction {
                        CopyDirection::ToCpu => {
                            completion.submitted = true;
                            unsafe { thoosa_shims::memcpy::tho_cu_memcpy_dtoh_async(pinned.cast(), source as _, chunk.len, stream.as_mut_ptr()) }
                        }
                        CopyDirection::ToCuda => {
                            unsafe { std::ptr::copy_nonoverlapping(source as *const u8, pinned, chunk.len); }
                            completion.submitted = true;
                            unsafe { thoosa_shims::memcpy::tho_cu_memcpy_htod_async(destination as _, pinned.cast(), chunk.len, stream.as_mut_ptr()) }
                        }
                    };
                    // Drain even after a failed submission. Preserve the original
                    // CUDA action when a second synchronization establishes completion.
                    let sync = completion.finish(crate::copy_ctx::complete_cuda_stream(stream.synchronize(), || stream.synchronize()));
                    crate::copy_ctx::finish_cuda_run(check_cuda_status(submitted, "CUDA copy submission"), sync)?;
                    if matches!(direction, CopyDirection::ToCpu) {
                        unsafe { std::ptr::copy_nonoverlapping(pinned, destination as *mut u8, chunk.len); }
                    }
                    Ok(())
                }).await;
                // The collector drains all chunks and prioritizes fatal failures,
                // including a task panic alongside a failed stream completion.
                crate::copy_ctx::finish_cuda_run(joined_copy_result(result), pool.check())
            }
        }), count, FuturesOrdering::Unordered, label).await.map(|_| ())
    }).await
}

/// Own both tensors throughout the asynchronous transfer, including cancellation.
#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="cuda_utils::copy_d2h"))]
pub async fn copy_d2h(
    gate: &Arc<thoosa_shared::CudaActivityGate>,
    source: Tensor,
    buf_size: usize,
    pool: &Arc<CopyBufferPool>,
) -> Result<Tensor, ThoError> {
    CudaActivityScope::for_operation(gate)?.run(async move {
        pool.check()?;
        if source.device() != Device::Cuda(gate.device_idx() - 1) { return Err(ThoError::new("D2H source belongs to another device")); }
        // LibTorch handles small or strided tensors synchronously while this
        // frame owns the source. Only contiguous ranges enter the chunk engine.
        if !source.is_contiguous() || tensor_mem_size(source.kind(), &source.size()) < buf_size { return Ok(source.f_to(Cpu)?); }
        let destination = Tensor::f_empty(source.size(), (source.kind(), Cpu))?;
        let transfer = CopyTransfer::new(CopyStorage::tensor(source)?, CopyStorage::tensor(destination)?)?;
        copy_transfer(gate, Arc::clone(&transfer), buf_size, pool, CopyDirection::ToCpu).await?;
        transfer.into_tensor()
    }).await
}

/// Own both tensors throughout the asynchronous transfer, including cancellation.
#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="cuda_utils::copy_h2d"))]
pub async fn copy_h2d(
    gate: &Arc<thoosa_shared::CudaActivityGate>,
    source: Tensor,
    buf_size: usize,
    pool: &Arc<CopyBufferPool>,
    dst_tho_device_idx: usize,
) -> Result<Tensor, ThoError> {
    CudaActivityScope::for_operation(gate)?.run(async move {
        pool.check()?;
        if source.device() != Cpu || dst_tho_device_idx != gate.device_idx() { return Err(ThoError::new("invalid H2D source or destination device")); }
        let device = Device::Cuda(dst_tho_device_idx - 1);
        if !source.is_contiguous() || tensor_mem_size(source.kind(), &source.size()) < buf_size { return Ok(source.f_to(device)?); }
        let destination = Tensor::f_empty(source.size(), (source.kind(), device))?;
        let transfer = CopyTransfer::new(CopyStorage::tensor(source)?, CopyStorage::tensor(destination)?)?;
        copy_transfer(gate, Arc::clone(&transfer), buf_size, pool, CopyDirection::ToCuda).await?;
        transfer.into_tensor()
    }).await
}

/// # Safety
/// source_owner must retain the readable CPU range implied by size/kind, with no
/// concurrent writes, until it is dropped, including after future cancellation.
/// For IPC, the parent must retain its arena allocation until reply or worker exit.
#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="cuda_utils::copy_h2d_from_ptr"))]
pub(crate) async unsafe fn copy_h2d_from_ptr(
    gate: &Arc<thoosa_shared::CudaActivityGate>, kind: Kind, buf_size: usize,
    pool: &Arc<CopyBufferPool>,
    dst_tho_device_idx: usize, source: usize, size: &[i64], source_owner: impl Send + 'static,
) -> Result<Tensor, ThoError> {
    CudaActivityScope::for_operation(gate)?.run(async move {
        pool.check()?;
        if dst_tho_device_idx != gate.device_idx() { return Err(ThoError::new("H2D destination belongs to another device")); }
        let destination = CopyStorage::tensor(Tensor::f_empty(size, (kind, Device::Cuda(dst_tho_device_idx - 1)))?)?;
        // SAFETY: this function's caller supplies the source allocation contract.
        let source = unsafe { CopyStorage::external(source, destination.byte_len, source_owner)? };
        let transfer = CopyTransfer::new(source, destination)?;
        copy_transfer(gate, Arc::clone(&transfer), buf_size, pool, CopyDirection::ToCuda).await?;
        transfer.into_tensor()
    }).await
}

/// # Safety
/// destination_owner must retain a writable CPU range covering source's bytes,
/// without other accesses, until it is dropped, including after cancellation.
/// For IPC, the parent must retain its arena allocation until reply or worker exit.
#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="cuda_utils::copy_d2h_to_ptr"))]
pub(crate) async unsafe fn copy_d2h_to_ptr(
    gate: &Arc<thoosa_shared::CudaActivityGate>, source: Tensor, buf_size: usize,
    pool: &Arc<CopyBufferPool>,
    destination: usize, destination_owner: impl Send + 'static,
) -> Result<(), ThoError> {
    CudaActivityScope::for_operation(gate)?.run(async move {
        pool.check()?;
        if source.device() != Device::Cuda(gate.device_idx() - 1) { return Err(ThoError::new("D2H source belongs to another device")); }
        let source = CopyStorage::tensor(source)?;
        // SAFETY: this function's caller supplies the destination allocation contract.
        let destination = unsafe { CopyStorage::external(destination, source.byte_len, destination_owner)? };
        let transfer = CopyTransfer::new(source, destination)?;
        copy_transfer(gate, transfer, buf_size, pool, CopyDirection::ToCpu).await
    }).await
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
        copy_d2h,
        copy_h2d,
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
    use thoosa_shared::traits::DebugId;
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

            let inst_0 = Instant::now();
            let tensor = ctx.copy(src_t.f_alias()?, 1).await?;
            h2d_time.fetch_add(inst_0.elapsed().as_millis() as u64, Relaxed);


            let inst_0 = Instant::now();
            let tensor = ctx.copy(tensor, 0).await?;
            d2h_time.fetch_add(inst_0.elapsed().as_millis() as u64, Relaxed);

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

            let dst = copy_d2h(
                &Arc::new(thoosa_shared::CudaActivityGate::new(1)),
                src_t.f_alias()?,
                buf_size,
                &Arc::new(super::CopyBufferPool::new((0..buffs).map(|_i| {
                    (
                        PinnedHostBuffer::new(buf_size, 0).unwrap(),
                        Arc::new(ThoCudaStream::from_pool(0, true).unwrap()),
                    )
                }))),
            ).await?;

            println!("dst=\t{dst}");

            let dst = copy_h2d(
                &Arc::new(thoosa_shared::CudaActivityGate::new(1)),
                dst,
                buf_size,
                &Arc::new(super::CopyBufferPool::new((0..buffs).map(|_i| {
                    (
                        PinnedHostBuffer::new(buf_size, CU_MEMHOSTALLOC_WRITECOMBINED).unwrap(),
                        Arc::new(ThoCudaStream::from_pool(0, true).unwrap()),
                    )
                }))),
                1,
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

            let dst = copy_d2h(
                &Arc::new(thoosa_shared::CudaActivityGate::new(1)),
                src_t.f_alias()?,
                buf_size,
                &Arc::new(super::CopyBufferPool::new((0..buffs).map(|_i| {
                    (
                        PinnedHostBuffer::new(buf_size as usize, CU_MEMHOSTALLOC_WRITECOMBINED).unwrap(),
                        Arc::new(ThoCudaStream::from_pool(0, true).unwrap()),
                    )
                }))),
            ).await?;

            println!("dst=\t{dst}");

            let dst = copy_h2d(
                &Arc::new(thoosa_shared::CudaActivityGate::new(1)),
                dst,
                buf_size,
                &Arc::new(super::CopyBufferPool::new((0..buffs).map(|_i| {
                    (
                        PinnedHostBuffer::new(buf_size as usize, 0).unwrap(),
                        Arc::new(ThoCudaStream::from_pool(0, true).unwrap()),
                    )
                }))),
                1,
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
        let bufs_pool = Arc::new(super::CopyBufferPool::new((0..buffs).map(|_| {
            (
                PinnedHostBuffer::new(buf_size, 0).unwrap(),
                Arc::new(ThoCudaStream::from_pool(0, true).unwrap()),
            )
        })));

        // Use a CPU pointer as if it were a device pointer — should error
        let bogus_cpu_tensor = Tensor::rand(&[1000, 1000], FLOAT_CPU);
        debug!("bogus_cpu_tensor={bogus_cpu_tensor:?} d={:?}", bogus_cpu_tensor.device());

        let bogus_device_ptr = bogus_cpu_tensor.data_ptr() as usize; // not a CUdeviceptr

        let res = super::copy_d2h(
            &Arc::new(thoosa_shared::CudaActivityGate::new(1)),
            bogus_cpu_tensor,
            buf_size,
            &bufs_pool,
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
                            .copy(t.f_alias()?, 1)
                            .await?;
                        let unfolded = cuda_t.unfold(1, 100, 10).contiguous();

                        copy_unfold_time += t_0.elapsed().as_millis();
                    } else {
                        let t_0 = Instant::now();
                        let t = t.unfold(1, 100, 10).contiguous();

                        let cuda_t = copy_ctx
                            .copy(t.f_alias()?, 1)
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

pub fn maybe_cuda_mem(device: &Device) -> Option<(usize, usize)> {
    match device {
        Device::Cuda(_) if  Device::cuda_if_available().is_cuda() => Some(cuda_mem()),
        _ => None,
    }
}

#[cfg(test)]
mod transfer_lifetime_tests {
    use super::*;
    use futures_util::FutureExt;
    use std::sync::Weak;

    fn tensor_transfer() -> Result<Arc<CopyTransfer>, ThoError> {
        CopyTransfer::new(
            CopyStorage::tensor(Tensor::f_from_slice(&[1u8, 2, 3, 4, 5, 6, 7, 8])?)?,
            CopyStorage::tensor(Tensor::f_zeros([8], (Kind::Uint8, Cpu))?)?,
        )
    }

    async fn released(weak: &Weak<CopyTransfer>) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while weak.upgrade().is_some() { tokio::task::yield_now().await; }
        }).await.expect("completed chunks must release transfer allocations");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_cancelled_collector_retains_both_allocations_until_last_blocking_chunk() -> Result<(), ThoError> {
        let transfer = tensor_transfer()?;
        let weak = Arc::downgrade(&transfer);
        let (started, mut starts) = tokio::sync::mpsc::unbounded_channel();
        let mut releases = Vec::new();
        let mut completions = Vec::new();
        let mut chunks = Vec::new();
        for index in 0..2 {
            let (release, wait) = std::sync::mpsc::channel();
            let (done, finished) = tokio::sync::oneshot::channel();
            releases.push(release);
            completions.push(finished);
            let transfer = Arc::clone(&transfer);
            let started = started.clone();
            chunks.push(async move {
                let result = spawn_copy_chunk(transfer, (), move |transfer, ()| {
                    let _ = started.send(());
                    wait.recv_timeout(Duration::from_secs(5)).context("release cancelled copy chunk")?;
                    // Exercise the actual retained addresses after the collector
                    // has been cancelled. Each chunk owns a disjoint destination.
                    unsafe { std::ptr::copy_nonoverlapping((transfer.source.address + index * 4) as *const u8, (transfer.destination.address + index * 4) as *mut u8, 4); }
                    let _ = done.send(());
                    Ok(())
                }).await;
                Ok::<_, ThoError>(joined_copy_result(result))
            });
        }
        drop(transfer);
        let parent = tokio::spawn(collect_futures_with_progress(chunks.into_iter(), 2, FuturesOrdering::Unordered, "test cancelled copy chunks"));
        for _ in 0..2 { tokio::time::timeout(Duration::from_secs(5), starts.recv()).await.unwrap().unwrap(); }
        parent.abort();
        assert!(parent.await.unwrap_err().is_cancelled());
        assert!(weak.upgrade().is_some(), "cancellation must retain both tensor allocations");
        releases.remove(0).send(()).unwrap();
        completions.remove(0).await.unwrap();
        assert!(weak.upgrade().is_some(), "the second blocking chunk still needs both allocations");
        releases.remove(0).send(()).unwrap();
        completions.remove(0).await.unwrap();
        released(&weak).await;
        Ok(())
    }

    #[test]
    fn test_runtime_shutdown_retains_worker_mapping_and_tensor_until_blocking_copy_finishes() -> Result<(), ThoError> {
        // Test both IPC directions with a real mapping, without needing CUDA.
        for to_cpu in [false, true] {
            let mut mapping = memmap2::MmapOptions::new().len(8).map_anon()?;
            mapping.copy_from_slice(&[1u8, 2, 3, 4, 5, 6, 7, 8]);
            let address = mapping.as_mut_ptr() as usize;
            let mapping = Arc::new(mapping);
            let weak_mapping = Arc::downgrade(&mapping);
            // SAFETY: move the sole mapping owner into the transfer; only its
            // blocking chunk accesses these bytes until completion.
            let mapped = unsafe { CopyStorage::external(address, 8, mapping)? };
            let tensor = CopyStorage::tensor(Tensor::f_from_slice(&[1u8, 2, 3, 4, 5, 6, 7, 8])?)?;
            let transfer = if to_cpu { CopyTransfer::new(tensor, mapped)? } else { CopyTransfer::new(mapped, tensor)? };
            let weak_transfer = Arc::downgrade(&transfer);
            let (started, start) = tokio::sync::oneshot::channel();
            let (release, wait) = std::sync::mpsc::channel();
            let (done, finished) = std::sync::mpsc::channel();
            let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(1).build()?;
            runtime.spawn(async move {
                spawn_copy_chunk(transfer, (), move |transfer, ()| {
                    let _ = started.send(());
                    wait.recv_timeout(Duration::from_secs(5)).context("release shutdown copy chunk")?;
                    unsafe { std::ptr::copy_nonoverlapping(transfer.source.address as *const u8, transfer.destination.address as *mut u8, 8); }
                    let _ = done.send(());
                    Ok(())
                }).await
            });
            runtime.block_on(start).unwrap();
            runtime.shutdown_timeout(Duration::ZERO);
            assert!(weak_transfer.upgrade().is_some(), "runtime shutdown cannot free the tensor allocation");
            assert!(weak_mapping.upgrade().is_some(), "runtime shutdown cannot unmap worker storage");
            release.send(()).unwrap();
            finished.recv_timeout(Duration::from_secs(5)).unwrap();
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            while weak_transfer.upgrade().is_some() || weak_mapping.upgrade().is_some() {
                assert!(std::time::Instant::now() < deadline, "completed blocking copy must release all allocations");
                std::thread::yield_now();
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_copy_panic_drains_and_reuses_pool_without_restart() -> Result<(), ThoError> {
        let context = ThoCudaContext::new(0)?;
        context.bind_to_thread()?;
        let pool = Arc::new(CopyBufferPool::new([(PinnedHostBuffer::new(8, 0)?, Arc::new(ThoCudaStream::from_pool(0, true)?))]));
        let buffer = pool.acquire().await?;
        let waiting = pool.acquire();
        tokio::pin!(waiting);
        assert!(futures::poll!(&mut waiting).is_pending());
        let transfer = tensor_transfer()?;
        let weak = Arc::downgrade(&transfer);
        let failed_pool = Arc::clone(&pool);
        let result = spawn_copy_chunk(transfer, buffer, move |transfer, buffer| {
            // Unwinding after submission still owns both allocations until the
            // completion guard has drained the real CUDA stream.
            let stream = Arc::clone(&buffer.1);
            let _completion = CopyCompletion { resources: Some((Arc::clone(transfer), buffer)), submitted: true, pool: failed_pool, stream };
            panic!("copy panicked after submitting DMA")
        }).await;
        assert!(result.unwrap_err().is_panic());
        assert!(weak.upgrade().is_none(), "successful drain must release storage despite the host panic");
        drop(tokio::time::timeout(Duration::from_secs(5), waiting).await.unwrap()?);
        pool.check()?;
        assert_eq!(pool.remaining_capacity(), 1, "drained stream must return to the pool");
        Ok(())
    }

    #[tokio::test]
    async fn test_unconfirmed_copy_completion_quarantines_storage_and_wakes_pool_waiters() -> Result<(), ThoError> {
        let context = ThoCudaContext::new(0)?;
        context.bind_to_thread()?;
        let pool = Arc::new(CopyBufferPool::new([(PinnedHostBuffer::new(8, 0)?, Arc::new(ThoCudaStream::from_pool(0, true)?))]));
        let buffer = pool.acquire().await?;
        let waiting = pool.acquire();
        tokio::pin!(waiting);
        assert!(futures::poll!(&mut waiting).is_pending());
        let transfer = tensor_transfer()?;
        let weak = Arc::downgrade(&transfer);
        let stream = Arc::clone(&buffer.1);
        let mut completion = CopyCompletion { resources: Some((transfer, buffer)), submitted: true, pool: Arc::clone(&pool), stream };
        // Both drain attempts fail without a fatal code. Do not invent one, but
        // still retain potentially live storage and reject ordinary reuse.
        let drained = crate::copy_ctx::complete_cuda_stream(Err(ThoError::oom()), || Err(ThoError::oom()));
        let error = completion.finish(drained).unwrap_err();
        assert!(error.is_cuda_recovery_failed() && !error.needs_cuda_ctx_reset());
        drop(completion);
        assert!(weak.upgrade().is_some(), "unknown completion must not release storage");
        assert!(tokio::time::timeout(Duration::from_secs(5), waiting).await.unwrap().is_err());
        assert!(pool.check().unwrap_err().is_cuda_recovery_failed());
        assert_eq!(pool.remaining_capacity(), 0, "quarantined stream must never return to the pool");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_owning_copy_roundtrip_uses_multiple_chunks() -> Result<(), ThoError> {
        let context = ThoCudaContext::new(0)?;
        context.bind_to_thread()?;
        let gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
        // A deliberately unaligned chunk size exercises final partial chunks and
        // copies split inside an element without allocating a large test tensor.
        let buf_size = 17;
        let pool = Arc::new(CopyBufferPool::new((0..2).map(|_| {
            Ok((PinnedHostBuffer::new(buf_size, 0)?, Arc::new(ThoCudaStream::from_pool(0, true)?)))
        }).collect::<Result<Vec<_>, ThoError>>()?));
        for kind in [Kind::Uint8, Kind::Int64, Kind::Float, Kind::Double] {
            for size in [0, 1, 31, 257] {
                let source = Tensor::f_arange(size, (Kind::Int64, Cpu))?.f_to_kind(kind)?;
                let cuda = copy_h2d(&gate, source.f_alias()?, buf_size, &pool, 1).await?;
                let restored = copy_d2h(&gate, cuda, buf_size, &pool).await?;
                assert_eq!(restored.size(), source.size());
                assert_eq!(restored.kind(), kind);
                // Empty tensors may have null data pointers; don't form byte slices for them.
                assert!(size == 0 || equal_bytes(&source, &restored), "owning chunked roundtrip changed bytes for {kind:?}, size={size}");
            }
        }
        assert_eq!(pool.remaining_capacity(), 2, "completed chunks must return their buffers");
        Ok(())
    }

    #[test]
    fn test_completed_transfer_returns_destination_and_releases_source() -> Result<(), ThoError> {
        let transfer = tensor_transfer()?;
        let weak = Arc::downgrade(&transfer);
        let destination = transfer.into_tensor()?;
        assert!(weak.upgrade().is_none());
        assert_eq!(destination.size(), [8]);
        Ok(())
    }
}
