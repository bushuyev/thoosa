use thoosa_common::tch_error::tch_result;
use crate::dims::{AnyDims, ChannelsData, Dims, LocalSelect};
use anyhow::Context as _;
use std::ops::Range;
use thoosa_shared::traits::Select;
use crate::runner::common::cachable::ThoDataRef;
#[double]
use crate::runner::global_ctx::GlobalCtx;
use crate::tho_device::ThoDevice;
use async_trait::async_trait;
use log::{debug, warn};
use mockall_double::double;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::{Arc, OnceLock};
use parking_lot::{Mutex, ReentrantMutex, ReentrantMutexGuard};
use tch::{Device, Kind, Tensor};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::tracked_await::t;
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, Named, DebugId, ThoSelector, WithId};
use thoosa_shared::{CopyToDevice, ThoError};
use thoosa_shared::debug_id;
use torch_sys::IntList;
use thoosa_shared::comp_id::SynKey;
use crate::ipc::worker_msg::{CudaWorkerId, WorkerInput};
use thoosa_shared::ipc_types::{WorkerOutput};


// Synchronize the non-Sync Tensor handle instead of asserting Sync for arbitrary IDs.
// Reentrant access permits expressions that read the same tensor more than once.
/// A synchronous lease on local storage, shared by local and worker kernels.
/// It borrows existing storage without copying and cannot cross a Send await.
#[derive(Debug)]
pub struct LocalTensorLease<'a>(ReentrantMutexGuard<'a, Tensor>);

impl Display for LocalTensorLease<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { Display::fmt(&**self, f) }
}

impl Deref for LocalTensorLease<'_> {
    type Target = Tensor;
    fn deref(&self) -> &Tensor { &self.0 }
}

impl Borrow<Tensor> for LocalTensorLease<'_> {
    fn borrow(&self) -> &Tensor { self }
}

impl AsRef<Tensor> for LocalTensorLease<'_> {
    fn as_ref(&self) -> &Tensor { self }
}

impl<I: Named + Debug + Clone + Send + Sync, D: Dims> Input for ThoTensor<D, I> {}
impl<I: Named + Debug + Clone + Send + Sync, D: Dims> Input for &ThoTensor<D, I> {}

impl<D, Id> ThoTensor<D, Id>
where
    D: Dims,
    Id: Named + Debug + Clone
{

    pub fn kind(&self) -> Kind {
        match &self.inner {
            ThoTensorInner::Val { t } => t.lock().kind(),
            ThoTensorInner::Ref { kind, .. } => kind.clone(),
        }
    }

    pub fn clone_ref(&self) -> Result<Self, ThoError> {
        match &self.inner {
            ThoTensorInner::Val { .. } => {
                Err(ThoError::new("cloning worker references requires a remote tensor"))
            }
            ThoTensorInner::Ref { key, size, kind, selector } => {
                Ok(
                    Self {
                        id:self.id.clone(),
                        inner: ThoTensorInner::Ref {
                            key:key.clone(),
                            size:size.clone(),
                            kind:*kind,
                            selector:selector.as_ref().map(|v|v.clone_box()),
                        },
                        pd: PhantomData,
                    }
                )
            }
        }
    }

    pub fn strip_types(self)->ThoTensor {
        ThoTensor {
            id: (),
            inner: self.inner,
            pd: Default::default(),
        }
    }

    pub fn no_drop(&mut self) {

        if let ThoTensorInner::Ref { key, .. } = &mut self.inner {
            debug!("ThoTensor::no_drop: key={key:?}");
            key.do_drop = false;
        }
    }
}

impl ThoTensor {
    pub fn to_types<Df, Idf>(self, id: Idf) -> Result<ThoTensor<Df, Idf>, ThoError>
    where
        Df: Dims,
        Idf: Named + std::fmt::Debug + Clone,
    {

        match self.inner {
            ThoTensorInner::Val { t } => {
                ThoTensor::new(id, t.into_inner())
            }
            ThoTensorInner::Ref { key, size, kind, selector } => {
                Ok(ThoTensor::<Df, Idf> {
                    id,
                    inner: ThoTensorInner::Ref {
                        key,
                        size,
                        kind,
                        selector:selector.as_ref().map(|v|v.clone_box()),
                    },
                    pd: PhantomData,
                })
            }
        }

    }

}

// Logging a handle must not materialize data or fail because storage is remote.
// Callers that explicitly want tensor values can format a LocalTensorLease.
impl<D: Dims, Id: Named + Debug + Clone> Display for ThoTensor<D, Id> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ThoTensor(id={:?}, size={:?}, kind={:?}, device={})", self.id, self.size(), self.kind(), self.tho_device_idx())
    }
}

//drop can't be implemented directly on ThoTensor or ThoTensorInner to keep them movable
#[derive(Debug)]
struct KeyHolder {
    pub key:u64,
    pub worker_id: CudaWorkerId,
    count:Arc<RemoteAllocation>,
    pub do_drop:bool,
}

impl Clone for KeyHolder {
    fn clone(&self) -> Self {
        debug!("KeyHolder::clone: self={self:?} self.worker_id.device_idx={} strong_count={:?}", self.worker_id.device_idx, Arc::strong_count(&self.count) );
        KeyHolder {
            key:  self.key,
            worker_id: self.worker_id,
            count: Arc::clone(&self.count),
            do_drop: self.do_drop,
        }
    }
}

// The last Arc owner releases exactly once even when references are dropped on
// different threads. Namespacing by actor prevents independent runtimes sharing keys.
#[derive(Debug)]
struct RemoteAllocation {
    key: u64,
    worker: CudaWorkerId,
    release: Option<crate::ipc::cuda_process::TensorReleaseHandle>,
    enabled: std::sync::atomic::AtomicBool,
}
impl Drop for RemoteAllocation {
    fn drop(&mut self) {
        if self.enabled.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some(release) = &self.release { release.release(self.key, self.worker); }
        }
    }
}

impl KeyHolder {
    fn ownership(key: u64, worker: CudaWorkerId, release: Option<crate::ipc::cuda_process::TensorReleaseHandle>) -> Arc<RemoteAllocation> {
        static KEYS: OnceLock<Mutex<HashMap<(u64, CudaWorkerId, u64), std::sync::Weak<RemoteAllocation>>>> = OnceLock::new();
        let mut keys = KEYS.get_or_init(Default::default).lock();

        let identity = (
            release.as_ref().map_or(0, |owner| owner.owner_id),
            worker,
            key
        );
        if let Some(owner) = keys.get(&identity).and_then(std::sync::Weak::upgrade) {
            return owner;
        }
        // Remove dead metadata as new references arrive; the registry owns no tensor.
        keys.retain(|_, owner| owner.strong_count() != 0);
        let owner = Arc::new(RemoteAllocation { key, worker, release, enabled: std::sync::atomic::AtomicBool::new(false) });
        keys.insert(identity, Arc::downgrade(&owner));
        owner
    }
}
impl Drop for KeyHolder {
    fn drop(&mut self) {
        if self.do_drop { self.count.enabled.store(true, std::sync::atomic::Ordering::Relaxed); }
    }
}


enum ThoTensorInner {
    Val { t: ReentrantMutex<Tensor> },
    Ref { key: KeyHolder, size: Vec<i64>, kind: Kind, selector: Option<Box<dyn ThoSelector>> },
}

impl Debug for ThoTensorInner {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ThoTensorInner::Val { t } => f.debug_struct("Val")
                .field("t", t)
                .field("device", &t.lock().device())
                .finish(),
            ThoTensorInner::Ref { key, size, kind, selector } => {
                f.debug_struct("Ref")
                    .field("key", key)
                    .field("size", size)
                    .field("kind", kind)
                    .field("selector", selector)
                    .finish()
            },
        }
    }
}


// Only the IPC codec needs wire fields. This descriptor cannot change ownership.
pub(crate) struct RemoteTensorRef<'a> {
    pub key: u64,
    pub worker_id: CudaWorkerId,
    pub size: &'a [i64],
    pub kind: Kind,
    pub selector: &'a Option<Box<dyn ThoSelector>>,
}

impl <D, Id, S> Select<S> for ThoTensor<D, Id>
where
    S:ThoSelector + 'static,
    D:Dims,
    Id:Named + Debug + Clone,
    Self:LocalSelect<Id, S, Output = ThoTensor<D, Id>>,
{
    type Output = ThoTensor<D, Id>;

    fn subview(&self, selector: S) -> Result<Self::Output, ThoError> {
        match &self.inner {
            ThoTensorInner::Val { t } => {
                Self::local_subview(self.id(), &t.lock(), selector)
            }
            ThoTensorInner::Ref { key, size, kind, selector: parent } => {
                // Children select relative to the current view, while the worker indexes the original tensor.
                let (selector, selected_size) = TensorRangesSelector::compose(parent.as_deref(), size, &selector)?;
                debug!("ThoTensor::subview: key={key:?} size={size:?} kind={kind:?}");
                Ok(Self {
                    id:self.id().clone(),
                    inner: ThoTensorInner::Ref {
                        key: key.clone(),
                        size: selected_size,
                        kind: *kind,
                        selector:Some(Box::new(selector) as Box<dyn ThoSelector>),
                    },
                    pd: PhantomData,
                }
                )
            }
        }
    }
}

/// Absolute ranges keep nested IPC views serializable without retaining a chain of selectors.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct TensorRangesSelector(Vec<Range<i64>>);

#[typetag::serde(name = "TensorRangesSelector")]
impl ThoSelector for TensorRangesSelector {
    fn clone_box(&self) -> Box<dyn ThoSelector> { Box::new(self.clone()) }

    fn tensor_ranges(&self) -> Result<Vec<Range<i64>>, ThoError> { Ok(self.0.clone()) }
}

impl TensorRangesSelector {
    fn compose(parent: Option<&dyn ThoSelector>, size: &[i64], child: &dyn ThoSelector) -> Result<(Self, Vec<i64>), ThoError> {
        let parent = parent.map(ThoSelector::tensor_ranges).transpose()?.unwrap_or_default();
        let child = child.tensor_ranges()?;
        if parent.len() > size.len() || child.len() > size.len() {
            return Err(ThoError::new(format!("selector rank exceeds tensor rank {}", size.len())));
        }
        let mut ranges = Vec::with_capacity(size.len());
        let mut selected_size = Vec::with_capacity(size.len());
        for (dim, &length) in size.iter().enumerate() {
            let range = child.get(dim).cloned().unwrap_or(0..length);
            // Match worker narrowing: clamp the end, reject negative, reversed, or out-of-bounds starts.
            let end = range.end.min(length);
            if range.start < 0 || range.end < range.start || range.start > end {
                return Err(ThoError::new(format!("invalid selector range {range:?} for dimension {dim} with size {length}")));
            }
            let offset = parent.get(dim).map_or(0, |range| range.start);
            let start = offset.checked_add(range.start).context("composed selector start overflow")?;
            let absolute_end = offset.checked_add(end).context("composed selector end overflow")?;
            ranges.push(start..absolute_end);
            selected_size.push(end - range.start);
        }
        Ok((Self(ranges), selected_size))
    }
}

/// An opaque tensor handle. Metadata and selection work in either process;
/// tensor operations require `local()`, `local_mut()`, or owned `into_local()`.
/// Dimension markers describe shape without exposing the storage representation.
///
/// ```compile_fail
/// use thoosa_common::tho_tensor::ThoTensor;
/// fn inspect(handle: &ThoTensor) { let _ = &handle.inner; }
/// ```
/// ```compile_fail
/// use thoosa_common::tho_tensor::ThoTensor;
/// fn tensor_operation(handle: &ThoTensor) { handle.t_ref(); }
/// ```
pub struct ThoTensor<D = AnyDims, Id = ()>
where
    D: Dims,
    Id: Named + Debug + Clone,
{
    pub(crate) id: Id,
    inner: ThoTensorInner,

    pub(crate) pd: PhantomData<D>,
}

/// Dimension-erased access used by common model builders without depending on feature-specific markers.
pub trait TensorInitRef: Debug + Send + Sync {
    fn local(&self) -> Result<LocalTensorLease<'_>, ThoError>;
}

impl<D, Id> TensorInitRef for ThoTensor<D, Id>
where
    D: Dims,
    Id: Named + Debug + Clone + Send + Sync,
{
    fn local(&self) -> Result<LocalTensorLease<'_>, ThoError> {
        ThoTensor::local(self)
    }
}


impl<'a, D, Id> Deref for ThoDataRef<'a, ThoTensor<D, Id>>
where
    D: Dims,
    Id: Named + Debug + Clone,
{
    type Target = ThoTensor<D, Id>;

    fn deref(&self) -> &Self::Target {
        match self {
            ThoDataRef::Borrowed(v) => v,
            ThoDataRef::ReadLocked(v) => v,
            ThoDataRef::Owned(v) => v,
        }
    }
}

impl<D: Dims> From<Tensor> for ThoTensor<D, DebugId> {
    fn from(tensor: Tensor) -> Self {
        Self {
            id: debug_id!("1"),
            inner: ThoTensorInner::Val { t: ReentrantMutex::new(tensor) },
            pd: PhantomData,
        }
    }
}

impl From<Tensor> for ThoTensor {
    fn from(tensor: Tensor) -> Self {
        Self {
            id: (),
            inner: ThoTensorInner::Val { t: ReentrantMutex::new(tensor) },
            pd: PhantomData,
        }
    }
}

impl<D: Dims, Id: Named + Debug + Clone + Send + Sync> Adjustable for ThoTensor<D, Id> {}

#[async_trait]
impl<D, Id> Data for ThoTensor<D, Id>
where
    D: Dims,
    Id: Named + Debug + Clone + Send + Sync,
{
    type Metrics = D::Metrics;

    fn size(&self) -> Self::Metrics {
        // Dimension markers are advisory; metadata behaves identically in both modes.
        Self::Metrics::try_from(self.size()).unwrap_or_default()
    }

    fn tho_device_idx(&self) -> usize {

        match &self.inner {
            ThoTensorInner::Val { t } => ThoDevice::try_from(t.lock().device()).expect("CPU/CUDA tensor device").tho_device_idx(),
            ThoTensorInner::Ref { key, .. } => key.worker_id.device_idx,
        }
    }

    fn mem_size(&self) -> u64 {
        self.numel() as u64 * self.kind().elt_size_in_bytes() as u64
    }

}

impl<D: Dims, Id: Named + Debug + Clone> Debug for ThoTensor<D, Id> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThoTensor")
            .field("id", &self.id)
            .field("size", &self.size())
            .field("kind", &self.kind())
            .field("device", &self.tho_device_idx())
            .finish()
    }
}

impl<D: Dims, Id: Named + Debug + Clone> ThoTensor<D, Id> {
    /// Select ranges relative to this view. Remote views retain the original
    /// allocation owner, so dropping a parent cannot release a live child.
    pub fn select_ranges(&self, ranges: &[Range<i64>]) -> Result<Self, ThoError> {
        let child = TensorRangesSelector(ranges.to_vec());
        match &self.inner {
            ThoTensorInner::Val { t } => {
                let t = t.lock();
                let (ranges, _) = TensorRangesSelector::compose(None, &t.size(), &child)?;
                let mut view = tch_result(t.f_alias())?;
                for (dim, range) in ranges.0.into_iter().enumerate() {
                    view = tch_result(view.f_narrow(dim as i64, range.start, range.end - range.start))?;
                }
                Self::new(self.id.clone(), view)
            }
            ThoTensorInner::Ref { key, size, kind, selector } => {
                let (selector, size) = TensorRangesSelector::compose(selector.as_deref(), size, &child)?;
                Ok(Self { id: self.id.clone(), inner: ThoTensorInner::Ref { key: key.clone(), size, kind: *kind, selector: Some(Box::new(selector)) }, pd: PhantomData })
            }
        }
    }

    pub fn new(id: Id, tensor: Tensor) -> Result<Self, ThoError> {
        let t = if tensor.is_contiguous() {
            tensor
        } else {
            warn!("got a non-contiguous ThoTensor: id={id:?}");
            tch_result(tensor.f_contiguous())?
        };

        Ok(Self {
            id,
            inner: ThoTensorInner::Val { t: ReentrantMutex::new(t) },
            pd: PhantomData,
        })
    }

    pub fn new_ref(id: Id, key: u64, size: Vec<i64>, kind: tch::Kind, selector: Option<Box<dyn ThoSelector>>, worker_id: CudaWorkerId) -> Self {
        let tho_device_idx = worker_id.device_idx;

        debug!("KeyHolder::new_ref: id={id:?} key={key} size={size:?} kind={kind:?} selector={selector:?} tho_device_idx={tho_device_idx:?}");

        let release = crate::ipc::cuda_process::TensorReleaseHandle::current().or_else(|| {
            // Compatibility for explicit reference construction. IPC decoding
            // supplies the original actor without consulting the installed facade.
            if crate::runner::global_ctx::GlobalCtx::is_installed() {
                crate::runner::global_ctx::GlobalCtx::global().execution(worker_id.device_idx).ipc_actor().ok().map(|actor| actor.tensor_releaser())
            } else { None }
        });
        let count = KeyHolder::ownership(key, worker_id, release);

        debug!("KeyHolder::new_ref: id={id:?} key={key} size={size:?} kind={kind:?} selector={selector:?} tho_device_idx={tho_device_idx:?}");

        Self {
            id,
            inner: ThoTensorInner::Ref {
                key: KeyHolder {
                    key,
                    worker_id,
                    count,
                    do_drop: true,
                },
                size,
                kind,
                selector,
            },
            pd: PhantomData,
        }

    }

    pub(crate) fn remote_ref(&self) -> Option<RemoteTensorRef<'_>> {
        match &self.inner {
            ThoTensorInner::Val { .. } => None,
            ThoTensorInner::Ref { key, size, kind, selector } => Some(RemoteTensorRef { key: key.key, worker_id: key.worker_id, size, kind: *kind, selector }),
        }
    }

    pub(crate) fn set_release_handle(&mut self, release: crate::ipc::cuda_process::TensorReleaseHandle) {
        if let ThoTensorInner::Ref { key, .. } = &mut self.inner { key.count = KeyHolder::ownership(key.key, key.worker_id, Some(release)); }
    }

    pub(crate) fn ref_worker_id(&self) -> Option<CudaWorkerId> {
        match &self.inner {
            ThoTensorInner::Ref { key, .. } => Some(key.worker_id),
            ThoTensorInner::Val { .. } => None,
        }
    }

    pub(crate) fn ref_key(&self) -> Option<u64> {
        match &self.inner {
            ThoTensorInner::Val { .. } => None,
            ThoTensorInner::Ref { key, .. } => Some(key.key),
        }
    }

    pub fn zeros(id: Id, size: impl IntList, options: (Kind, Device)) -> Result<Self, ThoError> {
        Ok(Self {
            id,
            inner: ThoTensorInner::Val { t: ReentrantMutex::new(tch_result(Tensor::f_zeros(size, options))?) },
            pd: PhantomData,
        })
    }

    pub fn size(&self) -> Vec<i64> {
        match &self.inner {
            ThoTensorInner::Val { t } => t.lock().size(),
            ThoTensorInner::Ref { size, .. } => size.clone(),
        }
    }
    pub fn numel(&self) -> usize {
        self.size().iter().map(|&length| length as usize).product()
    }

    pub fn id(&self) -> &Id {
        &self.id
    }

    pub fn with_id<NewId>(self, id: NewId) -> ThoTensor<D, NewId>
    where
        NewId: Named + Debug + Clone,
    {
        let Self { inner, pd, .. } = self;
        ThoTensor { id, inner, pd }
    }

    /// Explicit local-storage capability for paths that may receive IPC references.
    /// Keep the returned guard within synchronous code, before the next await.
    pub fn local(&self) -> Result<LocalTensorLease<'_>, ThoError> {
        match &self.inner {
            ThoTensorInner::Val { t } => Ok(LocalTensorLease(t.lock())),
            ThoTensorInner::Ref { .. } => Err(ThoError::new("local tensor access requires materializing the worker reference")),
        }
    }

    /// Mutable local storage is available only through an exclusive lease.
    pub fn local_mut(&mut self) -> Result<LocalTensorMut<'_, D, Id>, ThoError> {
        match &mut self.inner {
            ThoTensorInner::Val { t } => Ok(LocalTensorMut { id: &self.id, t: t.get_mut(), pd: PhantomData }),
            ThoTensorInner::Ref { .. } => Err(ThoError::new("local mutation requires materializing the worker reference")),
        }
    }

    /// Transfer ownership of local storage. Remote storage must first be
    /// materialized by the worker or copied through the execution backend.
    pub fn into_local(self) -> Result<Tensor, ThoError> {
        match self.inner {
            ThoTensorInner::Val { t } => Ok(t.into_inner()),
            ThoTensorInner::Ref { .. } => Err(ThoError::new("local ownership requires materializing the worker reference")),
        }
    }

    pub fn tho_device_idx(&self) -> usize {
        match &self.inner {
            ThoTensorInner::Val { t } => ThoDevice::try_from(t.lock().device()).expect("CPU/CUDA tensor device").tho_device_idx(),
            ThoTensorInner::Ref { key, .. } => key.worker_id.device_idx,
        }
    }

    pub fn tho_device(&self) -> ThoDevice {
        match &self.inner {
            ThoTensorInner::Val { t } => ThoDevice::try_from(t.lock().device()).expect("CPU/CUDA tensor device"),
            ThoTensorInner::Ref { key, .. } => <usize as Into<ThoDevice>>::into(key.worker_id.device_idx),
        }
    }

    //selector is used to fetch a chunk of data in a data-specific way when handling OOM. Here it's only to
    pub(crate) fn selector(&self) -> Option<Box<dyn ThoSelector>> {
        match &self.inner {
            ThoTensorInner::Val { .. } => None,
            ThoTensorInner::Ref { selector, .. } => selector.as_ref().map(|v| v.clone_box()),
        }
    }


    ///for asserts only
    pub(crate) fn is_ref(&self) ->bool {
        match &self.inner {
            ThoTensorInner::Val { .. } => false,
            ThoTensorInner::Ref { .. } => true
        }
    }
}

impl<D: Dims, Id> WithId for ThoTensor<D, Id>
where
    Id: CachableIdTr + Send + Sync + Eq + std::hash::Hash + Clone + std::fmt::Debug + Named,
{
    type ID = Id;

    fn id(&self) -> &<Self as WithId>::ID {
        &self.id
    }
}

#[async_trait]
impl<D: Dims, Id> CopyToDevice for ThoTensor<D, Id>
where
    Id: Named + Debug + Clone + Send + Sync,
{
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        debug!("CopyToDevice for ThoTensor: self.id={:?} self={self:?} dst_device={dst_device}", self.id);


        let dst_device = ThoDevice::from(dst_device);
        let src_device = self.tho_device_idx().into();
        //should not copy device to itself
        assert!(!dst_device.eq(&src_device));

        let id = self.id.clone();

        let ctx_device = if dst_device.is_cuda() { dst_device } else { src_device };

        let cuda_actor = GlobalCtx::global().cuda_actor(ctx_device);
        let copy = match &self.inner {
            ThoTensorInner::Val { t: src } => {
                let src = {
                    let src = src.lock();
                    if src.is_contiguous() {
                        tch_result(src.f_alias())?
                    } else {
                        warn!("CopyToDevice for ThoTensor: tensor was not contiguous, self.id={:?}", self.id);
                        tch_result(src.f_contiguous())?
                    }
                };

                debug!("CopyToDevice for ThoTensor: self.id={:?} copy copy_from_ptr, dst_device.ordinal={:?}", self.id, dst_device.tho_device_idx());
                t!(cuda_actor.copy(src.into(), dst_device)).await?
            }
            ThoTensorInner::Ref { key, .. } => {
                debug!("CopyToDevice for ThoTensor: self.id={:?} copy out-of-process ref key={key:?} dst_device.ordinal={:?}", self.id, dst_device.tho_device_idx());
                t!(cuda_actor.copy(
                    self.clone_ref()?.strip_types(),
                    dst_device,
                ))
                .await?
            }
        };

        copy.to_types(id)
    }
}

impl<D: Dims, Id> Output for ThoTensor<D, Id> where Id: Named + Debug + Clone + Debug + Send + Sync + Eq + std::hash::Hash + 'static {}

pub struct LocalTensorMut<'a, D: Dims, Id: Named + Debug + Clone> {
    #[expect(dead_code, reason = "Keeps the typed identity borrow tied to the mutable tensor view.")]
    id: &'a Id,
    t: &'a mut Tensor,
    // Like a shared lease, mutable access belongs to a synchronous kernel.
    pd: PhantomData<(D, std::rc::Rc<()>)>,
}

impl<D: Dims, Id: Named + Debug + Clone> Deref for LocalTensorMut<'_, D, Id> {
    type Target = Tensor;
    fn deref(&self) -> &Tensor { self.t }
}

impl<D: Dims, Id: Named + Debug + Clone> std::ops::DerefMut for LocalTensorMut<'_, D, Id> {
    fn deref_mut(&mut self) -> &mut Tensor { self.t }
}

impl<'a, D: Dims, Id: Named + Debug + Clone> LocalTensorMut<'a, D, Id> {
    pub fn new(id: &'a Id, t: &'a mut Tensor) -> Self {
        Self { id, t, pd: Default::default() }
    }

    pub fn add_(&mut self, rhs: &ThoTensor<D, Id>) -> Result<(), ThoError> {
        let _ = tch_result(self.t.f_add_(&*rhs.local()?))?;

        Ok(())
    }

    pub fn copy_partial_from_(&mut self, dim: i64, start: i64, length: i64, rhs: &ThoTensor<D, Id>) -> Result<(), ThoError> {
        // Preserve typed CUDA failures at the tensor boundary for split assembly.
        tch_result(tch_result(self.t.f_narrow(dim, start, length))?.f_copy_(&*rhs.local()?))?;
        Ok(())
    }

}


#[typetag::serde(name = "ThoTensor_default")]
impl WorkerOutput for ThoTensor {}

#[typetag::serde(name = "ThoTensor_ChannelsData")]
impl WorkerOutput for ThoTensor<ChannelsData, SynKey> {}

// Postcard stage calls need concrete typetag registrations for tensor inputs used across the main workflow.
#[typetag::serde(name = "ThoTensor_ChannelsData_Input")]
impl WorkerInput for ThoTensor<ChannelsData, SynKey> {}

#[cfg(test)]
mod generation_tests {
    use super::*;
    use thoosa_shared::CudaDeviceGeneration;

    #[test]
    fn test_selector_composition_preserves_other_dimensions_and_clamps() -> anyhow::Result<()> {
        let parent = TensorRangesSelector(vec![2..8, 4..10]);
        let child = TensorRangesSelector(vec![1..20]);
        let (selected, size) = TensorRangesSelector::compose(Some(&parent), &[6, 6], &child)?;
        assert_eq!(selected.tensor_ranges()?, vec![3..8, 4..10]);
        assert_eq!(size, vec![5, 6]);
        let (unchanged, same_size) = TensorRangesSelector::compose(Some(&selected), &size, &TensorRangesSelector(vec![]))?;
        assert_eq!(unchanged.tensor_ranges()?, selected.tensor_ranges()?);
        assert_eq!(same_size, size);
        assert!(TensorRangesSelector::compose(None, &[6], &TensorRangesSelector(vec![0..1, 0..1])).is_err());
        Ok(())
    }

    #[test]
    fn test_storage_capabilities_return_errors_for_the_wrong_variant() -> Result<(), ThoError> {
        let installed = crate::runner::global_ctx::MockGlobalCtx::is_installed_context();
        installed.expect().return_const(false);
        let local: ThoTensor = tch_result(Tensor::f_from_slice(&[1f32]))?.into();
        assert!(local.clone_ref().is_err());
        assert!(local.local().is_ok());
        let mut remote = ThoTensor::<AnyDims, ()>::new_ref((), 999, vec![1], Kind::Float, None, CudaWorkerId { device_idx: 1, generation: CudaDeviceGeneration(0) });
        assert!(remote.local().is_err());
        assert!(remote.local_mut().is_err());
        assert!(remote.into_local().is_err());
        Ok(())
    }

    #[test]
    fn test_remote_metadata_and_selection_need_no_local_storage() -> Result<(), ThoError> {
        let installed = crate::runner::global_ctx::MockGlobalCtx::is_installed_context();
        installed.expect().return_const(false);
        let worker = CudaWorkerId { device_idx: 2, generation: CudaDeviceGeneration(7) };
        let remote = ThoTensor::<AnyDims, DebugId>::new_ref(debug_id!("remote metadata"), 1234, vec![8, 6], Kind::Float, None, worker);
        assert_eq!(remote.size(), vec![8, 6]);
        assert_eq!(remote.numel(), 48);
        assert_eq!(remote.kind(), Kind::Float);
        assert_eq!(remote.tho_device_idx(), 2);
        assert_eq!(Data::mem_size(&remote), 192);
        assert!(format!("{remote}").contains("size=[8, 6]"));
        assert!(remote.local().is_err());

        let parent = remote.select_ranges(&[2..7, 1..5])?;
        let child = parent.select_ranges(&[1..3])?;
        assert_eq!(child.size(), vec![2, 4]);
        assert_eq!(child.numel(), 8);
        assert_eq!(child.selector().ok_or_else(|| ThoError::new("missing selector"))?.tensor_ranges()?, vec![3..5, 1..5]);
        let owner = match &remote.inner { ThoTensorInner::Ref { key, .. } => Arc::downgrade(&key.count), _ => unreachable!() };
        assert_eq!(owner.strong_count(), 3, "selected handles must retain the original allocation owner");
        drop(remote);
        drop(parent);
        assert_eq!(owner.strong_count(), 1, "the child must survive its parents");
        assert!(child.select_ranges(&[3..4]).is_err());
        drop(child);
        assert_eq!(owner.strong_count(), 0);
        Ok(())
    }

    #[test]
    fn test_local_and_materialized_worker_inputs_use_the_same_lease() -> anyhow::Result<()> {
        use crate::ipc::cuda_wokrer::{AllTensors, WorkerStageInput};
        let installed = crate::runner::global_ctx::MockGlobalCtx::is_installed_context();
        installed.expect().return_const(false);
        // CPU worker storage exercises the wire/materialization boundary without a GPU.
        let tensors = AllTensors::new(CudaWorkerId { device_idx: 0, generation: CudaDeviceGeneration(0) });
        let make = || -> Result<ThoTensor, ThoError> { Ok(tch_result(tch_result(Tensor::f_arange(48, (Kind::Float, Device::Cpu)))?.f_reshape([8, 6]))?.into()) };
        let local = make()?;
        let pointer = local.local()?.data_ptr();
        assert_eq!(local.local()?.data_ptr(), pointer, "a lease must not copy tensor storage");
        let remote = make()?.worker_tensor_to_ref(&tensors, &ThoDevice::from(0usize))?;
        let expected = local.select_ranges(&[2..7, 1..5])?.select_ranges(&[1..3])?;
        let selected = remote.select_ranges(&[2..7, 1..5])?.select_ranges(&[1..3])?;
        let wire = postcard::to_allocvec(&selected)?;
        let decoded: ThoTensor = postcard::from_bytes(&wire)?;
        let materialized = decoded.ref_to_tensor(&tensors, false)?;
        assert_eq!(materialized.size(), selected.size());
        assert!(tch_result(materialized.local()?.f_equal(&*expected.local()?))?);
        // Invalid selections must fail the same way before a local kernel or IPC call.
        for ranges in [vec![-1..2], vec![3..2], vec![9..10], vec![0..1, 0..1, 0..1]] {
            assert!(local.select_ranges(&ranges).is_err());
            assert!(remote.select_ranges(&ranges).is_err());
        }
        let mut mutable = make()?;
        tch_result(mutable.local_mut()?.f_copy_(&*local.local()?))?;
        assert!(tch_result(mutable.into_local()?.f_equal(&*local.local()?))?);
        Ok(())
    }

    #[test]
    fn test_reference_ownership_is_scoped_to_device_and_generation() {
        let installed = crate::runner::global_ctx::MockGlobalCtx::is_installed_context();
        installed.expect().return_const(false);
        let worker_id = CudaWorkerId { device_idx: 1, generation: CudaDeviceGeneration(0) };
        let make = |worker| ThoTensor::<AnyDims, ()>::new_ref((), 15, vec![1], Kind::Float, None, worker);
        let first = make(worker_id);
        let same = make(worker_id);
        let replacement = make(CudaWorkerId { generation: CudaDeviceGeneration(1), ..worker_id });
        let other = make(CudaWorkerId { device_idx: 2, ..worker_id });
        let key = |tensor: ThoTensor| match tensor.inner {
            ThoTensorInner::Ref { key, .. } => key,
            _ => unreachable!(),
        };
        let first = key(first);
        assert!(Arc::ptr_eq(&first.count, &key(same).count));
        assert!(!Arc::ptr_eq(&first.count, &key(replacement).count));
        assert!(!Arc::ptr_eq(&first.count, &key(other).count));
    }
}

#[cfg(test)]
mod thread_safety_tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;
    use static_assertions::{assert_impl_all, assert_not_impl_any};
    use crate::nets::common_nets::ThoModuleWithStore;
    use crate::nets::nodes::{ThoBatchNorm1d, ThoClosureModule, ThoLinear};
    use crate::nets::tho_varstore::ThoTrainModule;
    use crate::torch_runner::ThoTensorView;
    use crate::ipc::cuda_wokrer::AllTensors;

    #[derive(Debug, Clone)]
    struct CellId(#[expect(dead_code, reason = "The field determines auto traits checked below.")] Cell<usize>);
    impl Named for CellId {}

    #[derive(Debug, Clone)]
    struct LocalId(#[expect(dead_code, reason = "The field determines auto traits checked below.")] Rc<Cell<usize>>);
    impl Named for LocalId {}

    // Negative assertions prevent an unconditional unsafe impl from reopening the safe-code holes.
    assert_impl_all!(ThoTensor: Send, Sync);
    assert_impl_all!(ThoTensor<AnyDims, CellId>: Send);
    assert_not_impl_any!(ThoTensor<AnyDims, CellId>: Sync);
    assert_not_impl_any!(ThoTensor<AnyDims, LocalId>: Send, Sync);
    assert_not_impl_any!(LocalTensorLease<'static>: Send, Sync);
    assert_not_impl_any!(LocalTensorMut<'static, AnyDims, ()>: Send, Sync);
    assert_impl_all!(ThoTensorView<'static, AnyDims, ()>: Send);
    assert_not_impl_any!(ThoTensorView<'static, AnyDims, ()>: Sync);
    assert_not_impl_any!(ThoTensorView<'static, AnyDims, CellId>: Send, Sync);
    assert_impl_all!(ThoClosureModule: Send, Sync);
    assert_impl_all!(ThoModuleWithStore<AnyDims, AnyDims>: Send, Sync);
    assert_impl_all!(ThoTrainModule<AnyDims, AnyDims>: Send);
    assert_not_impl_any!(ThoTrainModule<AnyDims, AnyDims>: Sync);
    assert_impl_all!(ThoLinear<AnyDims, AnyDims>: Send);
    assert_not_impl_any!(ThoLinear<AnyDims, AnyDims>: Sync);
    assert_impl_all!(ThoBatchNorm1d: Send);
    assert_not_impl_any!(ThoBatchNorm1d: Sync);
    assert_impl_all!(AllTensors: Send, Sync);
    assert_not_impl_any!(crate::stages::train_step::TrainStep<Rc<()>, AnyDims, AnyDims, crate::runner::run_ctx::RunCtxImpl>: Send, Sync);

    #[test]
    fn test_shared_model_serializes_module_access() -> Result<(), ThoError> {
        use crate::nets::nodes::ThoModule;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let active = Arc::new(AtomicUsize::new(0));
        let completed = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&completed);
        let module = ThoClosureModule::new(Arc::new(Box::new(move |t, _| {
            // Each caller has a different input, so only the model lock can serialize these calls.
            assert_eq!(active.fetch_add(1, Ordering::SeqCst), 0);
            std::thread::sleep(std::time::Duration::from_millis(1));
            let result = t.f_add_scalar(1.);
            active.fetch_sub(1, Ordering::SeqCst);
            count.fetch_add(1, Ordering::SeqCst);
            Ok(result?)
        })));
        let model = ThoModuleWithStore::<AnyDims, AnyDims>::new(SynKey::new([]), Box::new(module));
        std::thread::scope(|scope| -> Result<(), ThoError> {
            let handles = (0..4).map(|_| scope.spawn(|| -> Result<(), ThoError> {
                let input = ThoTensor::<AnyDims, DebugId>::new(debug_id!("model lock test"), tch_result(Tensor::f_from_slice(&[2f32]))?)?;
                for _ in 0..10 {
                    let output = model.tho_forward(&input, None, false)?;
                    assert_eq!(tch_result(output.local()?.f_double_value(&[0]))?, 3.);
                }
                Ok(())
            })).collect::<Vec<_>>();
            for handle in handles { handle.join().expect("model thread panicked")?; }
            Ok(())
        })?;
        assert_eq!(completed.load(Ordering::SeqCst), 40);
        Ok(())
    }

    #[test]
    fn test_tensor_guard_excludes_other_threads_and_allows_nested_reads() -> Result<(), ThoError> {
        let tensor = ThoTensor::<AnyDims, ()>::new((), tch_result(Tensor::f_from_slice(&[2f32, 3.]))?)?;
        std::thread::scope(|scope| -> Result<(), ThoError> {
            let guard = tensor.local()?;
            // Repeated reads of the same handle on one thread must not deadlock.
            assert_eq!(tch_result(tensor.local()?.f_double_value(&[0]))?, 2.);
            let tensor_ref = &tensor;
            scope.spawn(move || {
                let ThoTensorInner::Val { t } = &tensor_ref.inner else { unreachable!() };
                assert!(t.try_lock().is_none(), "another thread must not access a borrowed Tensor handle");
            }).join().expect("tensor access thread panicked");
            drop(guard);
            scope.spawn(|| -> Result<(), ThoError> {
                assert_eq!(tch_result(tensor.local()?.f_double_value(&[1]))?, 3.);
                Ok(())
            }).join().expect("tensor access thread panicked")?;
            Ok(())
        })
    }

    #[test]
    fn test_local_leases_allow_cross_thread_arithmetic() -> Result<(), ThoError> {
        let a = ThoTensor::<AnyDims, ()>::new((), tch_result(Tensor::f_from_slice(&[2f32]))?)?;
        let b = ThoTensor::<AnyDims, ()>::new((), tch_result(Tensor::f_from_slice(&[3f32]))?)?;
        // Release the first lease before acquiring another tensor in either order.
        fn add(a: &ThoTensor, b: &ThoTensor) -> Result<Tensor, ThoError> {
            let b = tch_result(b.local()?.f_alias())?;
            tch_result(a.local()?.f_add(&b))
        }
        std::thread::scope(|scope| -> Result<(), ThoError> {
            let forward = scope.spawn(|| -> Result<(), ThoError> {
                for _ in 0..100 { assert_eq!(tch_result(add(&a, &b)?.f_double_value(&[0]))?, 5.); }
                Ok(())
            });
            let reverse = scope.spawn(|| -> Result<(), ThoError> {
                for _ in 0..100 { assert_eq!(tch_result(add(&b, &a)?.f_double_value(&[0]))?, 5.); }
                Ok(())
            });
            forward.join().expect("tensor addition thread panicked")?;
            reverse.join().expect("tensor addition thread panicked")?;
            Ok(())
        })
    }
}
