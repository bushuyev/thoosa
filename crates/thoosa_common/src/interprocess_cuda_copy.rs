//! Interprocess tensor staging backed by anonymous Linux memfd objects.
//!
//! The fd, together with [`IpcTensorTransferMetadata`], is the only shared-memory
//! identity sent between processes. Mapped addresses and tensor data pointers are
//! always process-local. The parent should retain an fd for every buffer that must
//! survive worker restarts and send that fd to each new worker with [`send_fd`].
//! GPU-to-GPU transfer through host is the composition of
//! [`ipc_d2h_producer_copy_tensor`] and [`ipc_h2d_consumer_copy`] using metadata
//! with [`IpcTransferDirection::GpuToGpuThroughHost`]. Concurrent bidirectional
//! transfers should use distinct buffers so each direction has independent ownership.

use std::ffi::{CString, c_void};
use std::fmt::{Debug, Formatter};
use std::io;
use std::marker::PhantomData;
use std::mem::{size_of, zeroed};
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
use std::ptr::NonNull;
use std::sync::Arc;

use cust::sys::CU_MEMHOSTREGISTER_PORTABLE;
use log::{debug, error};
use serde::{Deserialize, Serialize};
use tch::{Device, Kind, Tensor};
use tokio::net::UnixStream;
use tokio::task::{JoinError, spawn_blocking};
use tub::Pool;

use thoosa_shared::{Cause, FuturesOrdering, ThoError, collect_futures_with_progress};
use thoosa_shared::futures_limits::{COPY_D2H, COPY_H2D};
use thoosa_shared::tracked_await::track_await;
use thoosa_shims::context::ThoCudaContext;
use thoosa_shims::cuda_result::check_cuda_status;
use thoosa_shims::memcpy::{
    tho_cu_mem_host_register, tho_cu_mem_host_unregister, tho_cu_memcpy_dtoh_async,
    tho_cu_memcpy_htod_async,
};
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};

use crate::cuda_utils::{copy_chunks, tensor_mem_size};

pub type IpcBufsPool = Pool<(SharedMemfdBuffer, Arc<ThoCudaStream>)>;

fn invalid(message: impl Into<String>) -> ThoError {
    Cause::SomethingElse(message.into()).into()
}

fn join_error_to_tho_error(error: JoinError) -> ThoError {
    Cause::External {
        type_name: std::any::type_name::<JoinError>(),
        source: Arc::new(error),
    }.into()
}

fn page_aligned_len(len: usize) -> Result<usize, ThoError> {
    let page_size = unsafe {
        // SAFETY: sysconf has no pointer arguments and does not retain process state.
        libc::sysconf(libc::_SC_PAGESIZE)
    };
    if page_size <= 0 {
        return Err(io::Error::last_os_error().into());
    }
    let page_size = page_size as usize;
    let len = len.max(1);
    len.checked_add(page_size - 1)
        .map(|len| len / page_size * page_size)
        .ok_or_else(|| invalid(format!("memfd mapping length overflow for {len} bytes")))
}

/// An owned, process-local `MAP_SHARED` mapping of an anonymous memfd object.
pub struct SharedMemfdBuffer {
    fd: OwnedFd,
    ptr: NonNull<u8>,
    len: usize,
}

impl Debug for SharedMemfdBuffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedMemfdBuffer")
            .field("fd", &self.fd.as_raw_fd())
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}

impl SharedMemfdBuffer {
    /// Creates a new memfd, rounds its size up to a page boundary, and maps it.
    pub fn create(buffer_id: u64, requested_len: usize) -> Result<Self, ThoError> {
        let name = CString::new(format!("thoosa-tensor-{buffer_id}"))
            .map_err(|e| invalid(format!("invalid memfd name: {e}")))?;
        let mapped_len = page_aligned_len(requested_len)?;
        let raw_fd = unsafe {
            // SAFETY: name is a valid NUL-terminated C string. The returned fd is
            // uniquely adopted below and is never also owned elsewhere.
            libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC)
        };
        if raw_fd < 0 {
            return Err(io::Error::last_os_error().into());
        }
        let fd = unsafe {
            // SAFETY: memfd_create returned a new owned descriptor.
            OwnedFd::from_raw_fd(raw_fd)
        };
        if mapped_len > libc::off_t::MAX as usize {
            return Err(invalid(format!("memfd length {mapped_len} cannot be represented by off_t")));
        }
        let truncate_result = unsafe {
            // SAFETY: fd is valid and mapped_len was checked to fit Linux off_t.
            libc::ftruncate(fd.as_raw_fd(), mapped_len as libc::off_t)
        };
        if truncate_result != 0 {
            return Err(io::Error::last_os_error().into());
        }
        Self::map_owned_fd(fd, mapped_len)
    }

    /// Maps an fd received through inheritance or `SCM_RIGHTS` in this process.
    pub fn from_received_fd(fd: OwnedFd, mapped_len: usize) -> Result<Self, ThoError> {
        if mapped_len == 0 {
            return Err(invalid("cannot mmap a zero-length memfd"));
        }
        if page_aligned_len(mapped_len)? != mapped_len {
            return Err(invalid(format!("received memfd mapped_len={mapped_len} is not page-aligned")));
        }
        let mut stat = unsafe {
            // SAFETY: zero is a valid initial representation for libc::stat.
            zeroed::<libc::stat>()
        };
        let fstat_result = unsafe {
            // SAFETY: fd and the stat output pointer are valid for this call.
            libc::fstat(fd.as_raw_fd(), &mut stat)
        };
        if fstat_result != 0 {
            return Err(io::Error::last_os_error().into());
        }
        if stat.st_size < 0 || (stat.st_size as usize) < mapped_len {
            return Err(invalid(format!("received memfd is too small: file_len={} mapped_len={mapped_len}", stat.st_size)));
        }
        Self::map_owned_fd(fd, mapped_len)
    }

    fn map_owned_fd(fd: OwnedFd, mapped_len: usize) -> Result<Self, ThoError> {
        let raw_ptr = unsafe {
            // SAFETY: fd has at least mapped_len bytes and remains owned by Self for
            // the mapping lifetime. MAP_SHARED is required for interprocess visibility.
            libc::mmap(
                std::ptr::null_mut(),
                mapped_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if raw_ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error().into());
        }
        let ptr = NonNull::new(raw_ptr.cast::<u8>()).ok_or_else(|| invalid("mmap returned a null pointer"))?;
        debug!("SharedMemfdBuffer::map_owned_fd: fd={} local_mmap_ptr={:?} mapped_len={mapped_len}", fd.as_raw_fd(), ptr);
        Ok(Self { fd, ptr, len: mapped_len })
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }

    /// Duplicates the descriptor so ownership can be transferred independently.
    pub fn try_clone_fd(&self) -> io::Result<OwnedFd> {
        self.fd.try_clone()
    }

    fn checked_ptr(&self, offset: usize, byte_len: usize) -> Result<*const u8, ThoError> {
        let end = offset.checked_add(byte_len).ok_or_else(|| invalid("shared buffer range overflow"))?;
        if end > self.len {
            return Err(invalid(format!("shared buffer range [{offset}, {end}) exceeds mapped length {}", self.len)));
        }
        Ok(unsafe {
            // SAFETY: the checked range is within the live mmap allocation.
            self.ptr.as_ptr().add(offset)
        })
    }

    fn checked_mut_ptr(&mut self, offset: usize, byte_len: usize) -> Result<*mut u8, ThoError> {
        self.checked_ptr(offset, byte_len).map(|ptr| ptr.cast_mut())
    }
}

// The mapping and fd have stable ownership and may move between threads. Shared
// mutation is not exposed: writable access requires `&mut self`, while raw pointer
// use is confined to validated CUDA/copy operations and externally synchronized by
// the transfer ownership protocol.
unsafe impl Send for SharedMemfdBuffer {}
unsafe impl Sync for SharedMemfdBuffer {}

impl Drop for SharedMemfdBuffer {
    fn drop(&mut self) {
        let result = unsafe {
            // SAFETY: ptr/len are exactly the mapping returned by mmap and this is
            // their sole owner. CUDA registration guards borrow Self, so they must
            // have been dropped before this destructor can run.
            libc::munmap(self.ptr.as_ptr().cast::<c_void>(), self.len)
        };
        if result != 0 {
            error!("SharedMemfdBuffer::drop: munmap failed: {:?}", io::Error::last_os_error());
        }
    }
}

/// Registers this process's local mmap address with the CUDA Driver API.
///
/// The caller must make the intended CUDA context current before construction.
/// The lifetime prevents the mapping from being dropped before unregister.
pub struct CuMemHostRegistrationGuard<'a> {
    ptr: NonNull<c_void>,
    _buffer: PhantomData<&'a SharedMemfdBuffer>,
}

impl<'a> CuMemHostRegistrationGuard<'a> {
    pub fn new(buffer: &'a SharedMemfdBuffer) -> Result<Self, ThoError> {
        let ptr = NonNull::new(buffer.as_ptr().cast_mut().cast::<c_void>()).ok_or_else(|| invalid("cannot register a null mmap pointer"))?;
        let result = unsafe {
            // SAFETY: the mmap is live, writable, page-aligned, and remains alive
            // through the guard lifetime. The current thread owns a current context.
            tho_cu_mem_host_register(ptr.as_ptr(), buffer.len(), CU_MEMHOSTREGISTER_PORTABLE)
        };
        check_cuda_status(result, "cuMemHostRegister")?;
        Ok(Self { ptr, _buffer: PhantomData })
    }
}

impl Drop for CuMemHostRegistrationGuard<'_> {
    fn drop(&mut self) {
        let result = unsafe {
            // SAFETY: this pointer was successfully registered by the constructor
            // and the borrowed mapping is still live.
            tho_cu_mem_host_unregister(self.ptr.as_ptr())
        };
        if let Err(error) = check_cuda_status(result, "cuMemHostUnregister") {
            error!("CuMemHostRegistrationGuard::drop: {error:?}");
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum IpcTensorKind {
    Uint8,
    Int8,
    Int16,
    Int32,
    Int64,
    Half,
    Float,
    Double,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
    Bool,
    QInt8,
    QUInt8,
    QInt32,
    BFloat16,
    Float8e5m2,
    Float8e4m3fn,
    Float8e5m2fnuz,
    Float8e4m3fnuz,
}

impl From<Kind> for IpcTensorKind {
    fn from(value: Kind) -> Self {
        match value {
            Kind::Uint8 => Self::Uint8,
            Kind::Int8 => Self::Int8,
            Kind::Int16 => Self::Int16,
            Kind::Int => Self::Int32,
            Kind::Int64 => Self::Int64,
            Kind::Half => Self::Half,
            Kind::Float => Self::Float,
            Kind::Double => Self::Double,
            Kind::ComplexHalf => Self::ComplexHalf,
            Kind::ComplexFloat => Self::ComplexFloat,
            Kind::ComplexDouble => Self::ComplexDouble,
            Kind::Bool => Self::Bool,
            Kind::QInt8 => Self::QInt8,
            Kind::QUInt8 => Self::QUInt8,
            Kind::QInt32 => Self::QInt32,
            Kind::BFloat16 => Self::BFloat16,
            Kind::Float8e5m2 => Self::Float8e5m2,
            Kind::Float8e4m3fn => Self::Float8e4m3fn,
            Kind::Float8e5m2fnuz => Self::Float8e5m2fnuz,
            Kind::Float8e4m3fnuz => Self::Float8e4m3fnuz,
        }
    }
}

impl From<IpcTensorKind> for Kind {
    fn from(value: IpcTensorKind) -> Self {
        match value {
            IpcTensorKind::Uint8 => Kind::Uint8,
            IpcTensorKind::Int8 => Kind::Int8,
            IpcTensorKind::Int16 => Kind::Int16,
            IpcTensorKind::Int32 => Kind::Int,
            IpcTensorKind::Int64 => Kind::Int64,
            IpcTensorKind::Half => Kind::Half,
            IpcTensorKind::Float => Kind::Float,
            IpcTensorKind::Double => Kind::Double,
            IpcTensorKind::ComplexHalf => Kind::ComplexHalf,
            IpcTensorKind::ComplexFloat => Kind::ComplexFloat,
            IpcTensorKind::ComplexDouble => Kind::ComplexDouble,
            IpcTensorKind::Bool => Kind::Bool,
            IpcTensorKind::QInt8 => Kind::QInt8,
            IpcTensorKind::QUInt8 => Kind::QUInt8,
            IpcTensorKind::QInt32 => Kind::QInt32,
            IpcTensorKind::BFloat16 => Kind::BFloat16,
            IpcTensorKind::Float8e5m2 => Kind::Float8e5m2,
            IpcTensorKind::Float8e4m3fn => Kind::Float8e4m3fn,
            IpcTensorKind::Float8e5m2fnuz => Kind::Float8e5m2fnuz,
            IpcTensorKind::Float8e4m3fnuz => Kind::Float8e4m3fnuz,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum IpcTransferDirection {
    HostToDevice,
    DeviceToHost,
    GpuToGpuThroughHost,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum IpcProcessRole {
    Parent,
    Worker { worker_id: u64 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IpcTensorTransferMetadata {
    pub transfer_id: u64,
    pub buffer_id: u64,
    pub byte_len: usize,
    pub mapped_len: usize,
    pub offset: usize,
    pub kind: IpcTensorKind,
    pub shape: Vec<i64>,
    pub direction: IpcTransferDirection,
    pub producer: IpcProcessRole,
    pub consumer: IpcProcessRole,
}

impl IpcTensorTransferMetadata {
    pub fn new(
        transfer_id: u64,
        buffer_id: u64,
        mapped_len: usize,
        offset: usize,
        kind: Kind,
        shape: Vec<i64>,
        direction: IpcTransferDirection,
        producer: IpcProcessRole,
        consumer: IpcProcessRole,
    ) -> Result<Self, ThoError> {
        let byte_len = tensor_byte_len(kind, &shape)?;
        let metadata = Self {
            transfer_id,
            buffer_id,
            byte_len,
            mapped_len,
            offset,
            kind: kind.into(),
            shape,
            direction,
            producer,
            consumer,
        };
        metadata.validate_layout()?;
        Ok(metadata)
    }

    pub fn tch_kind(&self) -> Kind {
        self.kind.into()
    }

    pub fn validate_for_buffer(&self, buffer: &SharedMemfdBuffer) -> Result<(), ThoError> {
        self.validate_layout()?;
        if self.mapped_len != buffer.len() {
            return Err(invalid(format!("metadata mapped_len={} does not match local mapping length={}", self.mapped_len, buffer.len())));
        }
        buffer.checked_ptr(self.offset, self.byte_len)?;
        Ok(())
    }

    pub fn validate_tensor(&self, tensor: &Tensor) -> Result<(), ThoError> {
        self.validate_layout()?;
        if tensor.kind() != self.tch_kind() {
            return Err(invalid(format!("tensor kind {:?} does not match metadata kind {:?}", tensor.kind(), self.kind)));
        }
        if tensor.size() != self.shape {
            return Err(invalid(format!("tensor shape {:?} does not match metadata shape {:?}", tensor.size(), self.shape)));
        }
        if !tensor.is_contiguous() {
            return Err(invalid("interprocess copy requires a contiguous tensor"));
        }
        Ok(())
    }

    fn validate_layout(&self) -> Result<(), ThoError> {
        let expected = tensor_byte_len(self.tch_kind(), &self.shape)?;
        if expected == 0 {
            return Err(invalid("zero-byte tensors are not supported by the interprocess CUDA copy path"));
        }
        if expected != self.byte_len {
            return Err(invalid(format!("metadata byte_len={} but dtype/shape require {expected}", self.byte_len)));
        }
        let end = self.offset.checked_add(self.byte_len).ok_or_else(|| invalid("metadata offset + byte_len overflow"))?;
        if end > self.mapped_len {
            return Err(invalid(format!("metadata range [{}, {end}) exceeds mapped_len={}", self.offset, self.mapped_len)));
        }
        Ok(())
    }
}

fn tensor_byte_len(kind: Kind, shape: &[i64]) -> Result<usize, ThoError> {
    let elements = shape.iter().try_fold(1usize, |elements, &dimension| {
        if dimension < 0 {
            return Err(invalid(format!("negative tensor dimension {dimension}")));
        }
        elements.checked_mul(dimension as usize).ok_or_else(|| invalid("tensor element count overflow"))
    })?;
    elements.checked_mul(kind.elt_size_in_bytes()).ok_or_else(|| invalid("tensor byte length overflow"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum IpcBufferState {
    Free,
    Producing,
    Ready,
    Consuming,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IpcBufferStateMessage {
    pub transfer_id: u64,
    pub buffer_id: u64,
    pub state: IpcBufferState,
}

#[derive(Debug)]
pub struct IpcBufferOwnership {
    buffer_id: u64,
    transfer_id: Option<u64>,
    state: IpcBufferState,
}

impl IpcBufferOwnership {
    pub fn new(buffer_id: u64) -> Self {
        Self { buffer_id, transfer_id: None, state: IpcBufferState::Free }
    }

    pub fn from_ready(message: IpcBufferStateMessage) -> Result<Self, ThoError> {
        if message.state != IpcBufferState::Ready {
            return Err(invalid(format!("consumer expected READY, got {:?}", message.state)));
        }
        Ok(Self { buffer_id: message.buffer_id, transfer_id: Some(message.transfer_id), state: IpcBufferState::Ready })
    }

    pub fn state(&self) -> IpcBufferState {
        self.state
    }

    fn begin_producing(&mut self, metadata: &IpcTensorTransferMetadata) -> Result<(), ThoError> {
        self.require_buffer(metadata.buffer_id)?;
        if self.state != IpcBufferState::Free {
            return Err(invalid(format!("buffer {} cannot begin producing from {:?}", self.buffer_id, self.state)));
        }
        self.transfer_id = Some(metadata.transfer_id);
        self.state = IpcBufferState::Producing;
        Ok(())
    }

    fn publish_ready(&mut self) -> Result<IpcBufferStateMessage, ThoError> {
        if self.state != IpcBufferState::Producing {
            return Err(invalid(format!("buffer {} cannot publish READY from {:?}", self.buffer_id, self.state)));
        }
        self.state = IpcBufferState::Ready;
        Ok(self.message())
    }

    fn begin_consuming(&mut self, metadata: &IpcTensorTransferMetadata) -> Result<(), ThoError> {
        self.require_buffer(metadata.buffer_id)?;
        if self.transfer_id != Some(metadata.transfer_id) || self.state != IpcBufferState::Ready {
            return Err(invalid(format!("buffer {} cannot consume transfer {} from {:?}", self.buffer_id, metadata.transfer_id, self.state)));
        }
        self.state = IpcBufferState::Consuming;
        Ok(())
    }

    fn finish_consuming(&mut self) -> Result<IpcBufferStateMessage, ThoError> {
        if self.state != IpcBufferState::Consuming {
            return Err(invalid(format!("buffer {} cannot finish consuming from {:?}", self.buffer_id, self.state)));
        }
        let message = IpcBufferStateMessage {
            transfer_id: self.transfer_id.ok_or_else(|| invalid("consuming buffer has no transfer id"))?,
            buffer_id: self.buffer_id,
            state: IpcBufferState::Free,
        };
        self.state = IpcBufferState::Free;
        self.transfer_id = None;
        Ok(message)
    }

    pub fn acknowledge_consumed(&mut self, message: IpcBufferStateMessage) -> Result<(), ThoError> {
        self.require_buffer(message.buffer_id)?;
        if self.state != IpcBufferState::Ready || self.transfer_id != Some(message.transfer_id) || message.state != IpcBufferState::Free {
            return Err(invalid(format!("invalid consumed acknowledgement {message:?} for ownership {self:?}")));
        }
        self.state = IpcBufferState::Free;
        self.transfer_id = None;
        Ok(())
    }

    fn require_buffer(&self, buffer_id: u64) -> Result<(), ThoError> {
        if self.buffer_id != buffer_id {
            return Err(invalid(format!("ownership buffer_id={} does not match metadata buffer_id={buffer_id}", self.buffer_id)));
        }
        Ok(())
    }

    fn message(&self) -> IpcBufferStateMessage {
        IpcBufferStateMessage {
            transfer_id: self.transfer_id.expect("non-free ownership must have a transfer id"),
            buffer_id: self.buffer_id,
            state: self.state,
        }
    }
}

/// Same-process-shaped H2D entry point backed by cached memfd mappings.
///
/// The source pointer is process-local. Each chunk is first copied into a checked-out
/// shared mapping, then transferred to the local CUDA tensor with the Driver API.
pub async fn ipc_copy_h2d_from_ptr(
    kind: Kind,
    buf_size: usize,
    ipc_bufs_pool: &Arc<IpcBufsPool>,
    dst_tho_device_idx: usize,
    src_ptr: usize,
    size: &[i64],
) -> Result<Tensor, ThoError> {
    if dst_tho_device_idx == 0 {
        return Err(invalid("ipc_copy_h2d_from_ptr requires a CUDA destination"));
    }
    let byte_len = tensor_mem_size(kind, size);
    if byte_len != 0 && src_ptr == 0 {
        return Err(invalid("ipc_copy_h2d_from_ptr source pointer is null"));
    }
    let chunks = copy_chunks(byte_len, buf_size, "ipc_copy_h2d_from_ptr")?;
    let chunk_count = chunks.len();
    let destination = Tensor::f_empty(size, (kind, Device::Cuda(dst_tho_device_idx - 1)))?;
    let dst_ptr = destination.data_ptr() as u64;

    track_await(collect_futures_with_progress(
        chunks.into_iter().map(|chunk| {
            let pool = Arc::clone(ipc_bufs_pool);
            async move {
                let mut buf_stream = track_await(pool.acquire()).await;
                let result = track_await(spawn_blocking(move || {
                    tracing::info_span!(
                        "interprocess_cuda_copy::ipc_copy_h2d_from_ptr_chunk",
                        kind = "spawn_blocking",
                        file = file!(),
                        line = line!(),
                        chunk = chunk.index,
                    ).in_scope(|| {
                        if chunk.len > buf_stream.0.len() {
                            return Err(invalid(format!("H2D chunk length={} exceeds pooled mmap length={}", chunk.len, buf_stream.0.len())));
                        }
                        let source = src_ptr.checked_add(chunk.offset).ok_or_else(|| invalid("H2D source pointer overflow"))?;
                        let destination = dst_ptr.checked_add(chunk.offset as u64).ok_or_else(|| invalid("H2D destination pointer overflow"))?;
                        let shared_ptr = buf_stream.0.as_mut_ptr();
                        unsafe {
                            // SAFETY: the caller's local source tensor remains alive for
                            // the operation, copy_chunks bounds chunk.len to the tensor byte
                            // range, and the checked-out mmap is at least chunk.len bytes.
                            std::ptr::copy_nonoverlapping(source as *const u8, shared_ptr, chunk.len);
                        }
                        let stream = Arc::clone(&buf_stream.1);
                        let _thread_context = ThoCudaContext::new(stream.device())?;
                        let _stream_guard = ThoStreamGuard::new(stream.as_ref())?;
                        let _registration = CuMemHostRegistrationGuard::new(&buf_stream.0)?;
                        let result = unsafe {
                            // SAFETY: destination is a process-local CUDA allocation,
                            // shared_ptr is within the registered mmap, and the stream's
                            // CUDA context is current on this OS thread.
                            tho_cu_memcpy_htod_async(destination, shared_ptr.cast::<c_void>(), chunk.len, stream.as_mut_ptr())
                        };
                        check_cuda_status(result, "cuMemcpyHtoDAsync(memfd pool)")?;
                        stream.synchronize()?;
                        Ok::<(), ThoError>(())
                    })
                })).await.map_err(join_error_to_tho_error).and_then(|result| result);
                Ok::<_, ThoError>(result)
            }
        }),
        chunk_count,
        FuturesOrdering::Unordered,
        COPY_H2D,
    )).await.into_iter().collect::<Result<(), ThoError>>()?;

    Ok(destination)
}

/// Same-process-shaped D2H entry point backed by cached memfd mappings.
///
/// Each chunk is copied from the process-local CUDA pointer into a registered shared
/// mapping, synchronized, and then copied into owned CPU tensor storage.
pub async fn ipc_copy_d2h_from_ptr(
    kind: Kind,
    buf_size: usize,
    ipc_bufs_pool: &Arc<IpcBufsPool>,
    src_ptr: usize,
    size: &[i64],
) -> Result<Tensor, ThoError> {
    let byte_len = tensor_mem_size(kind, size);
    if byte_len != 0 && src_ptr == 0 {
        return Err(invalid("ipc_copy_d2h_from_ptr source pointer is null"));
    }
    let chunks = copy_chunks(byte_len, buf_size, "ipc_copy_d2h_from_ptr")?;
    let chunk_count = chunks.len();
    let destination = Tensor::f_empty(size, (kind, Device::Cpu))?;
    let dst_ptr = destination.data_ptr() as usize;

    track_await(collect_futures_with_progress(
        chunks.into_iter().map(|chunk| {
            let pool = Arc::clone(ipc_bufs_pool);
            async move {
                let mut buf_stream = track_await(pool.acquire()).await;
                let result = track_await(spawn_blocking(move || {
                    tracing::info_span!(
                        "interprocess_cuda_copy::ipc_copy_d2h_from_ptr_chunk",
                        kind = "spawn_blocking",
                        file = file!(),
                        line = line!(),
                        chunk = chunk.index,
                    ).in_scope(|| {
                        if chunk.len > buf_stream.0.len() {
                            return Err(invalid(format!("D2H chunk length={} exceeds pooled mmap length={}", chunk.len, buf_stream.0.len())));
                        }
                        let source = src_ptr.checked_add(chunk.offset).ok_or_else(|| invalid("D2H source pointer overflow"))? as u64;
                        let destination = dst_ptr.checked_add(chunk.offset).ok_or_else(|| invalid("D2H destination pointer overflow"))?;
                        let shared_ptr = buf_stream.0.as_mut_ptr();
                        let stream = Arc::clone(&buf_stream.1);
                        let _thread_context = ThoCudaContext::new(stream.device())?;
                        let _stream_guard = ThoStreamGuard::new(stream.as_ref())?;
                        let _registration = CuMemHostRegistrationGuard::new(&buf_stream.0)?;
                        let result = unsafe {
                            // SAFETY: source is a process-local CUDA allocation,
                            // shared_ptr is within the registered mmap, and the stream's
                            // CUDA context is current on this OS thread.
                            tho_cu_memcpy_dtoh_async(shared_ptr.cast::<c_void>(), source, chunk.len, stream.as_mut_ptr())
                        };
                        check_cuda_status(result, "cuMemcpyDtoHAsync(memfd pool)")?;
                        stream.synchronize()?;
                        unsafe {
                            // SAFETY: the Driver copy and stream synchronization completed,
                            // destination points into the live CPU tensor, and both ranges
                            // are validated for chunk.len bytes.
                            std::ptr::copy_nonoverlapping(shared_ptr, destination as *mut u8, chunk.len);
                        }
                        Ok::<(), ThoError>(())
                    })
                })).await.map_err(join_error_to_tho_error).and_then(|result| result);
                Ok::<_, ThoError>(result)
            }
        }),
        chunk_count,
        FuturesOrdering::Unordered,
        COPY_D2H,
    )).await.into_iter().collect::<Result<(), ThoError>>()?;

    Ok(destination)
}

pub fn ipc_h2d_producer_write_bytes(
    buffer: &mut SharedMemfdBuffer,
    metadata: &IpcTensorTransferMetadata,
    source: &[u8],
    ownership: &mut IpcBufferOwnership,
) -> Result<IpcBufferStateMessage, ThoError> {
    require_h2d(metadata)?;
    metadata.validate_for_buffer(buffer)?;
    if source.len() != metadata.byte_len {
        return Err(invalid(format!("H2D source byte length={} does not match metadata byte_len={}", source.len(), metadata.byte_len)));
    }
    ownership.begin_producing(metadata)?;
    let destination = buffer.checked_mut_ptr(metadata.offset, metadata.byte_len)?;
    unsafe {
        // SAFETY: source has exactly byte_len initialized bytes and the validated
        // destination range is writable and non-overlapping with source.
        std::ptr::copy_nonoverlapping(source.as_ptr(), destination, metadata.byte_len);
    }
    ownership.publish_ready()
}

/// Copies process-local source memory into a shared H2D staging buffer.
///
/// # Safety
/// `source` must be valid for reads of `metadata.byte_len` bytes and must not
/// overlap this buffer's mapped destination range.
pub unsafe fn ipc_h2d_producer_write_from_ptr(
    buffer: &mut SharedMemfdBuffer,
    metadata: &IpcTensorTransferMetadata,
    source: *const u8,
    ownership: &mut IpcBufferOwnership,
) -> Result<IpcBufferStateMessage, ThoError> {
    if source.is_null() && metadata.byte_len != 0 {
        return Err(invalid("H2D source pointer is null"));
    }
    let source = unsafe {
        // SAFETY: upheld by the caller contract above; no pointer is retained.
        std::slice::from_raw_parts(source, metadata.byte_len)
    };
    ipc_h2d_producer_write_bytes(buffer, metadata, source, ownership)
}

pub fn ipc_h2d_producer_write_tensor(
    buffer: &mut SharedMemfdBuffer,
    metadata: &IpcTensorTransferMetadata,
    source: &Tensor,
    ownership: &mut IpcBufferOwnership,
) -> Result<IpcBufferStateMessage, ThoError> {
    if source.device() != Device::Cpu {
        return Err(invalid(format!("H2D producer tensor must be on CPU, got {:?}", source.device())));
    }
    metadata.validate_tensor(source)?;
    unsafe {
        // SAFETY: a contiguous CPU tensor owns at least metadata.byte_len readable
        // bytes at data_ptr and remains borrowed for the entire copy.
        ipc_h2d_producer_write_from_ptr(buffer, metadata, source.data_ptr().cast::<u8>(), ownership)
    }
}

pub fn ipc_h2d_consumer_copy(
    buffer: &SharedMemfdBuffer,
    metadata: &IpcTensorTransferMetadata,
    cuda_device: usize,
    stream: &ThoCudaStream,
    ownership: &mut IpcBufferOwnership,
) -> Result<(Tensor, IpcBufferStateMessage), ThoError> {
    require_h2d(metadata)?;
    metadata.validate_for_buffer(buffer)?;
    if stream.device() != cuda_device as i32 {
        return Err(invalid(format!("CUDA stream device={} does not match destination device={cuda_device}", stream.device())));
    }
    ownership.begin_consuming(metadata)?;
    let _stream_guard = ThoStreamGuard::new(stream)?;
    let _registration = CuMemHostRegistrationGuard::new(buffer)?;
    let destination = Tensor::f_empty(metadata.shape.as_slice(), (metadata.tch_kind(), Device::Cuda(cuda_device)))?;
    let source = buffer.checked_ptr(metadata.offset, metadata.byte_len)?;
    let result = unsafe {
        // SAFETY: destination is a local CUDA tensor with exactly metadata.byte_len
        // bytes, source is within the registered live mmap, and stream is local.
        tho_cu_memcpy_htod_async(destination.data_ptr() as u64, source.cast::<c_void>(), metadata.byte_len, stream.as_mut_ptr())
    };
    check_cuda_status(result, "cuMemcpyHtoDAsync(memfd)")?;
    synchronize_driver_stream(stream)?;
    let consumed = ownership.finish_consuming()?;
    Ok((destination, consumed))
}

pub fn ipc_d2h_producer_copy_tensor(
    buffer: &mut SharedMemfdBuffer,
    metadata: &IpcTensorTransferMetadata,
    source: &Tensor,
    stream: &ThoCudaStream,
    ownership: &mut IpcBufferOwnership,
) -> Result<IpcBufferStateMessage, ThoError> {
    require_d2h(metadata)?;
    metadata.validate_for_buffer(buffer)?;
    metadata.validate_tensor(source)?;
    let Device::Cuda(cuda_device) = source.device() else {
        return Err(invalid(format!("D2H producer tensor must be on CUDA, got {:?}", source.device())));
    };
    if stream.device() != cuda_device as i32 {
        return Err(invalid(format!("CUDA stream device={} does not match source device={cuda_device}", stream.device())));
    }
    ownership.begin_producing(metadata)?;
    let destination = buffer.checked_mut_ptr(metadata.offset, metadata.byte_len)?;
    let _stream_guard = ThoStreamGuard::new(stream)?;
    let _registration = CuMemHostRegistrationGuard::new(buffer)?;
    // The caller must pass the stream that produced `source`, or a stream that has
    // already waited on the producer event. This synchronization makes all such
    // preceding work complete before the D2H transfer reads the device allocation.
    synchronize_driver_stream(stream)?;
    let result = unsafe {
        // SAFETY: source is a live local CUDA tensor, destination is within the
        // registered mmap, and both ranges contain metadata.byte_len bytes.
        tho_cu_memcpy_dtoh_async(destination.cast::<c_void>(), source.data_ptr() as u64, metadata.byte_len, stream.as_mut_ptr())
    };
    check_cuda_status(result, "cuMemcpyDtoHAsync(memfd)")?;
    synchronize_driver_stream(stream)?;
    ownership.publish_ready()
}

pub fn ipc_d2h_consumer_copy(
    buffer: &SharedMemfdBuffer,
    metadata: &IpcTensorTransferMetadata,
    ownership: &mut IpcBufferOwnership,
) -> Result<(Tensor, IpcBufferStateMessage), ThoError> {
    require_d2h(metadata)?;
    metadata.validate_for_buffer(buffer)?;
    ownership.begin_consuming(metadata)?;
    let destination = Tensor::f_empty(metadata.shape.as_slice(), (metadata.tch_kind(), Device::Cpu))?;
    let source = buffer.checked_ptr(metadata.offset, metadata.byte_len)?;
    unsafe {
        // SAFETY: metadata validates both ranges, destination owns independent CPU
        // storage, and the copy completes before the buffer may be acknowledged FREE.
        std::ptr::copy_nonoverlapping(source, destination.data_ptr().cast::<u8>(), metadata.byte_len);
    }
    // TODO: a torch::from_blob C++ shim could provide a zero-copy borrowed tensor,
    // but it would need to retain the mapping and ownership lease for tensor lifetime.
    let consumed = ownership.finish_consuming()?;
    Ok((destination, consumed))
}

fn require_h2d(metadata: &IpcTensorTransferMetadata) -> Result<(), ThoError> {
    if !matches!(metadata.direction, IpcTransferDirection::HostToDevice | IpcTransferDirection::GpuToGpuThroughHost) {
        return Err(invalid(format!("H2D operation cannot use direction {:?}", metadata.direction)));
    }
    Ok(())
}

fn require_d2h(metadata: &IpcTensorTransferMetadata) -> Result<(), ThoError> {
    if !matches!(metadata.direction, IpcTransferDirection::DeviceToHost | IpcTransferDirection::GpuToGpuThroughHost) {
        return Err(invalid(format!("D2H operation cannot use direction {:?}", metadata.direction)));
    }
    Ok(())
}

fn synchronize_driver_stream(stream: &ThoCudaStream) -> Result<(), ThoError> {
    stream.synchronize()
}

/// Sends one duplicated reference to `fd` over a connected Unix domain socket.
pub async fn send_fd(stream: &UnixStream, fd: BorrowedFd<'_>) -> io::Result<()> {
    loop {
        stream.writable().await?;
        match send_fd_once(stream.as_raw_fd(), fd.as_raw_fd()) {
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => continue,
            result => return result,
        }
    }
}

/// Receives one fd sent with [`send_fd`] and returns independent ownership of it.
pub async fn recv_fd(stream: &UnixStream) -> io::Result<OwnedFd> {
    loop {
        stream.readable().await?;
        match recv_fd_once(stream.as_raw_fd()) {
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => continue,
            result => return result,
        }
    }
}

fn send_fd_once(socket_fd: RawFd, fd: RawFd) -> io::Result<()> {
    let marker = [0x46u8];
    let mut iov = libc::iovec { iov_base: marker.as_ptr().cast_mut().cast::<c_void>(), iov_len: marker.len() };
    let control_len = unsafe {
        // SAFETY: CMSG_SPACE is a pure size calculation.
        libc::CMSG_SPACE(size_of::<libc::c_int>() as u32) as usize
    };
    let mut control = vec![0usize; control_len.div_ceil(size_of::<usize>())];
    let mut message = unsafe {
        // SAFETY: zero is a valid initial representation for msghdr.
        zeroed::<libc::msghdr>()
    };
    message.msg_iov = &mut iov;
    message.msg_iovlen = 1;
    message.msg_control = control.as_mut_ptr().cast::<c_void>();
    message.msg_controllen = control_len;
    unsafe {
        // SAFETY: control is suitably aligned and sized by CMSG_SPACE. The cmsghdr
        // and its one descriptor payload are initialized before sendmsg reads them.
        let header = libc::CMSG_FIRSTHDR(&message);
        if header.is_null() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "failed to allocate SCM_RIGHTS header"));
        }
        (*header).cmsg_level = libc::SOL_SOCKET;
        (*header).cmsg_type = libc::SCM_RIGHTS;
        (*header).cmsg_len = libc::CMSG_LEN(size_of::<libc::c_int>() as u32) as usize;
        std::ptr::write(libc::CMSG_DATA(header).cast::<libc::c_int>(), fd);
        let sent = libc::sendmsg(socket_fd, &message, libc::MSG_NOSIGNAL);
        if sent < 0 {
            return Err(io::Error::last_os_error());
        }
        if sent != marker.len() as isize {
            return Err(io::Error::new(io::ErrorKind::WriteZero, format!("SCM_RIGHTS marker write was partial: {sent}")));
        }
    }
    Ok(())
}

fn recv_fd_once(socket_fd: RawFd) -> io::Result<OwnedFd> {
    let mut marker = [0u8; 1];
    let mut iov = libc::iovec { iov_base: marker.as_mut_ptr().cast::<c_void>(), iov_len: marker.len() };
    let control_len = unsafe {
        // SAFETY: CMSG_SPACE is a pure size calculation.
        libc::CMSG_SPACE(size_of::<libc::c_int>() as u32) as usize
    };
    let mut control = vec![0usize; control_len.div_ceil(size_of::<usize>())];
    let mut message = unsafe {
        // SAFETY: zero is a valid initial representation for msghdr.
        zeroed::<libc::msghdr>()
    };
    message.msg_iov = &mut iov;
    message.msg_iovlen = 1;
    message.msg_control = control.as_mut_ptr().cast::<c_void>();
    message.msg_controllen = control_len;
    let received = unsafe {
        // SAFETY: all message buffers are valid and writable for their declared sizes.
        libc::recvmsg(socket_fd, &mut message, libc::MSG_CMSG_CLOEXEC)
    };
    if received < 0 {
        return Err(io::Error::last_os_error());
    }
    if received == 0 {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "socket closed while receiving memfd"));
    }
    if marker[0] != 0x46 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("invalid SCM_RIGHTS marker: {}", marker[0])));
    }
    if message.msg_flags & libc::MSG_CTRUNC != 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "SCM_RIGHTS control message was truncated"));
    }
    let mut received_fd = None;
    unsafe {
        // SAFETY: recvmsg initialized the ancillary buffer; CMSG_* walks only within
        // msg_controllen. Each received descriptor is adopted exactly once.
        let mut header = libc::CMSG_FIRSTHDR(&message);
        while !header.is_null() {
            if (*header).cmsg_level == libc::SOL_SOCKET && (*header).cmsg_type == libc::SCM_RIGHTS {
                let raw_fd = std::ptr::read(libc::CMSG_DATA(header).cast::<libc::c_int>());
                let owned = OwnedFd::from_raw_fd(raw_fd);
                if received_fd.replace(owned).is_some() {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "received more than one fd"));
                }
            }
            header = libc::CMSG_NXTHDR(&message, header);
        }
    }
    received_fd.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "SCM_RIGHTS message contained no fd"))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::process::Stdio;

    use tch::Device::Cpu;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixListener;
    use tokio::process::{Child, Command};

    use thoosa_shims::context::ThoCudaContext;

    use super::*;

    const TEST_SOCKET_ENV: &str = "THOOSA_MEMFD_COPY_TEST_SOCKET";
    const TEST_MODE_ENV: &str = "THOOSA_MEMFD_COPY_TEST_MODE";
    const TEST_FRAME_MAX: usize = 16 * 1024 * 1024;

    async fn write_frame(stream: &mut UnixStream, payload: &[u8]) -> Result<(), ThoError> {
        if payload.len() > u32::MAX as usize {
            return Err(invalid(format!("test frame too large: {}", payload.len())));
        }
        stream.write_all(&(payload.len() as u32).to_le_bytes()).await?;
        stream.write_all(payload).await?;
        stream.flush().await?;
        Ok(())
    }

    async fn read_frame(stream: &mut UnixStream) -> Result<Vec<u8>, ThoError> {
        let mut len = [0u8; 4];
        stream.read_exact(&mut len).await?;
        let len = u32::from_le_bytes(len) as usize;
        if len > TEST_FRAME_MAX {
            return Err(invalid(format!("test frame too large: {len}")));
        }
        let mut payload = vec![0u8; len];
        stream.read_exact(&mut payload).await?;
        Ok(payload)
    }

    async fn write_json<T: Serialize>(stream: &mut UnixStream, value: &T) -> Result<(), ThoError> {
        write_frame(stream, &serde_json::to_vec(value)?).await
    }

    async fn read_json<T: for<'de> Deserialize<'de>>(stream: &mut UnixStream) -> Result<T, ThoError> {
        Ok(serde_json::from_slice(&read_frame(stream).await?)?)
    }

    fn test_name() -> String {
        let module = module_path!();
        let module = module.split_once("::").map(|(_, module)| module).unwrap_or(module);
        format!("{module}::test_interprocess_h2d_d2h_and_worker_restart")
    }

    async fn spawn_worker(mode: &str) -> Result<(Child, UnixStream), ThoError> {
        let dir = tempfile::tempdir()?;
        let socket_path = dir.path().join("memfd-copy.sock");
        let listener = UnixListener::bind(&socket_path)?;
        let mut command = Command::new(std::env::current_exe()?);
        command
            .arg(test_name())
            .arg("--exact")
            .arg("--nocapture")
            .env(TEST_SOCKET_ENV, &socket_path)
            .env(TEST_MODE_ENV, mode)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
        let mut child = command.spawn()?;
        let accepted = tokio::time::timeout(std::time::Duration::from_secs(10), listener.accept()).await;
        let stream = match accepted {
            Ok(Ok((stream, _))) => stream,
            Ok(Err(error)) => return Err(error.into()),
            Err(_) => {
                let status = child.try_wait()?;
                let _ = child.kill().await;
                return Err(invalid(format!("memfd copy worker failed to connect; status={status:?}")));
            }
        };
        drop(dir);
        Ok((child, stream))
    }

    async fn wait_worker(mut child: Child) -> Result<(), ThoError> {
        let status = tokio::time::timeout(std::time::Duration::from_secs(30), child.wait())
            .await
            .map_err(|_| invalid("memfd copy worker did not exit"))??;
        if !status.success() {
            return Err(invalid(format!("memfd copy worker exited with {status}")));
        }
        Ok(())
    }

    fn f32_bytes(values: &[f32]) -> &[u8] {
        unsafe {
            // SAFETY: f32 has no invalid bit patterns and the returned slice borrows
            // values for exactly its initialized byte extent.
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        }
    }

    fn tensor_bytes(tensor: &Tensor, byte_len: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; byte_len];
        unsafe {
            // SAFETY: tests call this only for contiguous CPU tensors whose validated
            // metadata requires exactly byte_len bytes.
            std::ptr::copy_nonoverlapping(tensor.data_ptr().cast::<u8>(), bytes.as_mut_ptr(), byte_len);
        }
        bytes
    }

    async fn run_h2d_worker() -> Result<(), ThoError> {
        let socket = PathBuf::from(std::env::var_os(TEST_SOCKET_ENV).ok_or_else(|| invalid("missing worker socket"))?);
        let mut stream = UnixStream::connect(socket).await?;
        let metadata: IpcTensorTransferMetadata = read_json(&mut stream).await?;
        let ready: IpcBufferStateMessage = read_json(&mut stream).await?;
        let fd = recv_fd(&stream).await?;
        let buffer = SharedMemfdBuffer::from_received_fd(fd, metadata.mapped_len)?;
        debug!("run_h2d_worker: local_mmap_ptr={:?}", buffer.as_ptr());
        let _context = ThoCudaContext::new(0)?;
        let cuda_stream = ThoCudaStream::from_pool(0, true)?;
        let mut ownership = IpcBufferOwnership::from_ready(ready)?;
        let (cuda_tensor, consumed) = ipc_h2d_consumer_copy(&buffer, &metadata, 0, &cuda_stream, &mut ownership)?;
        let cpu_tensor = cuda_tensor.f_to_device(Cpu)?;
        write_json(&mut stream, &consumed).await?;
        write_frame(&mut stream, &tensor_bytes(&cpu_tensor, metadata.byte_len)).await?;
        Ok(())
    }

    async fn run_d2h_worker() -> Result<(), ThoError> {
        let socket = PathBuf::from(std::env::var_os(TEST_SOCKET_ENV).ok_or_else(|| invalid("missing worker socket"))?);
        let mut stream = UnixStream::connect(socket).await?;
        let metadata: IpcTensorTransferMetadata = read_json(&mut stream).await?;
        let fd = recv_fd(&stream).await?;
        let mut buffer = SharedMemfdBuffer::from_received_fd(fd, metadata.mapped_len)?;
        debug!("run_d2h_worker: local_mmap_ptr={:?}", buffer.as_ptr());
        let _context = ThoCudaContext::new(0)?;
        let cuda_stream = ThoCudaStream::from_pool(0, true)?;
        let element_count = metadata.byte_len / size_of::<f32>();
        let source = {
            let _stream_guard = ThoStreamGuard::new(&cuda_stream)?;
            Tensor::f_arange(element_count as i64, (Kind::Float, Device::Cuda(0)))?
        };
        let mut ownership = IpcBufferOwnership::new(metadata.buffer_id);
        let ready = ipc_d2h_producer_copy_tensor(&mut buffer, &metadata, &source, &cuda_stream, &mut ownership)?;
        write_json(&mut stream, &ready).await?;
        let consumed: IpcBufferStateMessage = read_json(&mut stream).await?;
        ownership.acknowledge_consumed(consumed)?;
        Ok(())
    }

    async fn run_h2d_parent_once(
        buffer: &mut SharedMemfdBuffer,
        ownership: &mut IpcBufferOwnership,
        transfer_id: u64,
        values: &[f32],
    ) -> Result<(), ThoError> {
        let metadata = IpcTensorTransferMetadata::new(
            transfer_id,
            7,
            buffer.len(),
            0,
            Kind::Float,
            vec![values.len() as i64],
            IpcTransferDirection::HostToDevice,
            IpcProcessRole::Parent,
            IpcProcessRole::Worker { worker_id: transfer_id },
        )?;
        let ready = ipc_h2d_producer_write_bytes(buffer, &metadata, f32_bytes(values), ownership)?;
        let (child, mut stream) = spawn_worker("h2d").await?;
        write_json(&mut stream, &metadata).await?;
        write_json(&mut stream, &ready).await?;
        send_fd(&stream, buffer.fd()).await?;
        let consumed: IpcBufferStateMessage = read_json(&mut stream).await?;
        let copied = read_frame(&mut stream).await?;
        ownership.acknowledge_consumed(consumed)?;
        if copied != f32_bytes(values) {
            return Err(invalid("interprocess H2D values did not round-trip"));
        }
        wait_worker(child).await
    }

    async fn run_parent() -> Result<(), ThoError> {
        let values = (0..1024).map(|value| value as f32).collect::<Vec<_>>();
        let mut buffer = SharedMemfdBuffer::create(7, std::mem::size_of_val(values.as_slice()))?;
        let parent_ptr = buffer.as_ptr();
        debug!("run_parent: local_mmap_ptr={parent_ptr:?}");
        let mut ownership = IpcBufferOwnership::new(7);

        run_h2d_parent_once(&mut buffer, &mut ownership, 1, &values).await?;
        // The parent still owns the original fd after the first worker exits. Passing
        // it again proves that a restarted worker can map the same anonymous memfd.
        run_h2d_parent_once(&mut buffer, &mut ownership, 2, &values).await?;

        let metadata = IpcTensorTransferMetadata::new(
            3,
            7,
            buffer.len(),
            0,
            Kind::Float,
            vec![values.len() as i64],
            IpcTransferDirection::DeviceToHost,
            IpcProcessRole::Worker { worker_id: 3 },
            IpcProcessRole::Parent,
        )?;
        let (child, mut stream) = spawn_worker("d2h").await?;
        write_json(&mut stream, &metadata).await?;
        send_fd(&stream, buffer.fd()).await?;
        let ready: IpcBufferStateMessage = read_json(&mut stream).await?;
        let mut consumer_ownership = IpcBufferOwnership::from_ready(ready)?;
        let (cpu_tensor, consumed) = ipc_d2h_consumer_copy(&buffer, &metadata, &mut consumer_ownership)?;
        write_json(&mut stream, &consumed).await?;
        if tensor_bytes(&cpu_tensor, metadata.byte_len) != f32_bytes(&values) {
            return Err(invalid("interprocess D2H values did not match"));
        }
        wait_worker(child).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_interprocess_h2d_d2h_and_worker_restart() {
        let result = match std::env::var(TEST_MODE_ENV).as_deref() {
            Ok("h2d") => run_h2d_worker().await,
            Ok("d2h") => run_d2h_worker().await,
            Ok(mode) => Err(invalid(format!("unknown memfd copy test mode {mode}"))),
            Err(_) => run_parent().await,
        };
        result.unwrap();
    }

    #[tokio::test]
    async fn test_fd_passing_and_metadata_validation_without_cuda() -> Result<(), ThoError> {
        let mut owner = SharedMemfdBuffer::create(11, 32)?;
        let metadata = IpcTensorTransferMetadata::new(
            12,
            11,
            owner.len(),
            8,
            Kind::Int64,
            vec![3],
            IpcTransferDirection::HostToDevice,
            IpcProcessRole::Parent,
            IpcProcessRole::Worker { worker_id: 1 },
        )?;
        let values = [1i64, 2, 3];
        let bytes = unsafe {
            // SAFETY: values is initialized and borrowed for its exact byte extent.
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(&values))
        };
        let mut ownership = IpcBufferOwnership::new(11);
        let ready = ipc_h2d_producer_write_bytes(&mut owner, &metadata, bytes, &mut ownership)?;
        let (left, right) = UnixStream::pair()?;
        send_fd(&left, owner.fd()).await?;
        let received = recv_fd(&right).await?;
        let mapped = SharedMemfdBuffer::from_received_fd(received, metadata.mapped_len)?;
        let mut consumer = IpcBufferOwnership::from_ready(ready)?;
        consumer.begin_consuming(&metadata)?;
        let source = mapped.checked_ptr(metadata.offset, metadata.byte_len)?;
        let copied = unsafe {
            // SAFETY: metadata validated the complete mapped range.
            std::slice::from_raw_parts(source, metadata.byte_len)
        };
        assert_eq!(copied, bytes);
        ownership.acknowledge_consumed(consumer.finish_consuming()?)?;
        assert_eq!(ownership.state(), IpcBufferState::Free);
        Ok(())
    }
}
