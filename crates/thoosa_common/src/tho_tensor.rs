use thoosa_common::tch_error::tch_result;
use std::backtrace::Backtrace;
use crate::dims::{AnyDims, ChannelsData, Dims};
use crate::runner::common::cachable::ThoDataRef;
#[double]
use crate::runner::global_ctx::GlobalCtx;
use crate::tho_device::ThoDevice;
use async_trait::async_trait;
use log::{debug, error, warn};
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
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, Named, DebugId, ThoSelector, WithId};
use thoosa_shared::{CopyToDevice, ThoError};
use thoosa_shared::debug_id;
use torch_sys::IntList;
use thoosa_shared::comp_id::SynKey;
use crate::ipc::worker_msg::{CudaWorkerId, WorkerInput};
use thoosa_shared::ipc_types::{WorkerOutput};


// Synchronize the non-Sync Tensor handle instead of asserting Sync for arbitrary IDs.
// Reentrant access permits expressions that read the same tensor more than once.
#[derive(Debug)]
pub struct TensorRef<'a>(ReentrantMutexGuard<'a, Tensor>);

impl Display for TensorRef<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { Display::fmt(&**self, f) }
}

impl Deref for TensorRef<'_> {
    type Target = Tensor;
    fn deref(&self) -> &Tensor { &self.0 }
}

impl Borrow<Tensor> for TensorRef<'_> {
    fn borrow(&self) -> &Tensor { self }
}

impl AsRef<Tensor> for TensorRef<'_> {
    fn as_ref(&self) -> &Tensor { self }
}

impl<I: Named + Debug + Clone + Send + Sync, D: Dims> Input for ThoTensor<D, I> {}
impl<I: Named + Debug + Clone + Send + Sync, D: Dims> Input for &ThoTensor<D, I> {}

impl<D, Id> ThoTensor<D, Id>
where
    D: Dims,
    Id: Named + Debug + Clone
{

    pub fn shallow_clone<DO: Dims>(&self) -> ThoTensor<DO, Id> {
        ThoTensor {
            id: self.id.clone(),
            inner: match &self.inner{
                ThoTensorInner::Val { t } => {
                    ThoTensorInner::Val { t: ReentrantMutex::new(t.lock().shallow_clone()) }
                }
                ThoTensorInner::Ref { key, size, kind, selector } => {
                    ThoTensorInner::Ref {
                        key: key.clone(),
                        size: size.clone(),
                        kind: *kind,
                        selector: selector.as_ref().map(|v|v.clone_box()),
                    }
                }
            },
            pd: Default::default(),
        }
    }

    pub fn drop_inner(self) {
        drop(self.inner);
    }

    pub fn to_kind(&self, k: Kind) -> Result<ThoTensor<D, Id>, ThoError> {
        Ok(ThoTensor {
            id: self.id.clone(),
            inner: ThoTensorInner::Val { t: ReentrantMutex::new(tch_result(self.try_t_ref()?.f_to_kind(k))?) }, //TODO ref
            pd: PhantomData,
        })
    }

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

    pub fn clone_ref_fields(id: Id, key: &KeyHolder, size: &Vec<i64>, kind: &tch::Kind, selector: &Option<Box<dyn ThoSelector>>) -> Result<Self, ThoError> {
        Ok(ThoTensor {
            id,
            inner: ThoTensorInner::Ref {
                key:key.clone(),
                size:size.clone(),
                kind:kind.clone(),
                selector:selector.as_ref().map(|v|v.clone_box()),
            },
            pd: PhantomData,
        })
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

impl<D: Dims> Display for ThoTensor<D, DebugId> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&*self.t_ref(), f)
    }
}

//drop can't be implemented directly on ThoTensor or ThoTensorInner to keep them movable
#[derive(Debug)]
pub struct KeyHolder {
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


pub /*TODO*/ enum ThoTensorInner {
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


//D:Dims - is just a reminder of what dimensions a tensor has.
pub struct ThoTensor<D = AnyDims, Id = ()>
where
    D: Dims,
    Id: Named + Debug + Clone,
{
    pub(crate) id: Id,
    pub inner: ThoTensorInner,

    pub(crate) pd: PhantomData<D>,
}

/// Dimension-erased access used by common model builders without depending on feature-specific markers.
pub trait TensorInitRef: Debug + Send + Sync {
    fn tensor_ref(&self) -> TensorRef<'_>;
}

impl<D, Id> TensorInitRef for ThoTensor<D, Id>
where
    D: Dims,
    Id: Named + Debug + Clone + Send + Sync,
{
    fn tensor_ref(&self) -> TensorRef<'_> {
        self.t_ref()
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
        match &self.inner {
            ThoTensorInner::Val { t } => t.lock().size().try_into().unwrap_or_default(),
            ThoTensorInner::Ref { size, .. } => Self::Metrics::try_from(size.clone()).unwrap_or_else(|_e| panic!("Tho tensor size metrics conversion failed")),
        }
    }

    fn tho_device_idx(&self) -> usize {

        match &self.inner {
            ThoTensorInner::Val { t } => ThoDevice::try_from(t.lock().device()).expect("CPU/CUDA tensor device").tho_device_idx(),
            ThoTensorInner::Ref { key, .. } => key.worker_id.device_idx,
        }
    }

    fn mem_size(&self) -> u64 {
        match &self.inner {
            ThoTensorInner::Val { t } => { let t = t.lock(); t.size().iter().fold(1u64, |a, v| a * (*v as u64)) * t.kind().elt_size_in_bytes() as u64 },
            ThoTensorInner::Ref { key: _key, .. } => {
                1000 //TODO
            }
        }
    }

}

impl<D: Dims, Id: Named + Debug + Clone> Debug for ThoTensor<D, Id> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("")
            .field("id", &self.id)
            .field("inner", &self.inner)
            .finish()
    }
}

impl<D: Dims, Id: Named + Debug + Clone> ThoTensor<D, Id> {
    fn inner_tensor_ref(&self) -> TensorRef<'_> {
        match &self.inner {
            ThoTensorInner::Val { t } => TensorRef(t.lock()),
            ThoTensorInner::Ref { key, .. } => {
                error!("ThoTensor::inner_tensor_ref: {:?}", Backtrace::capture());
                panic!("can't get ref of a remote tensor key={key:?}")
            },
        }
    }

    fn into_inner_tensor(self) -> Tensor {
        match self.inner {
            ThoTensorInner::Val { t } => t.into_inner(),
            ThoTensorInner::Ref { key, .. } => {
                error!("ThoTensor::into_inner_tensor: {:?}", Backtrace::capture());
                panic!("can't get remote tensor key={key:?}")
            },
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

    pub(crate) fn set_release_handle(&mut self, release: crate::ipc::cuda_process::TensorReleaseHandle) {
        if let ThoTensorInner::Ref { key, .. } = &mut self.inner { key.count = KeyHolder::ownership(key.key, key.worker_id, Some(release)); }
    }

    pub fn ref_worker_id(&self) -> Option<CudaWorkerId> {
        match &self.inner {
            ThoTensorInner::Ref { key, .. } => Some(key.worker_id),
            ThoTensorInner::Val { .. } => None,
        }
    }

    pub fn ref_key(&self) -> Option<u64> {
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

    pub fn requires_grad(&self) -> bool {
        self.t_ref().requires_grad()
    }
    pub fn size(&self) -> Vec<i64> {
        match &self.inner {
            ThoTensorInner::Val { t } => t.lock().size(),
            ThoTensorInner::Ref { size, .. } => size.clone(),
        }
    }
    pub fn numel(&self) -> usize {
        self.t_ref().numel()
    }

    pub fn add(&self, rhs: &ThoTensor<D, Id>) -> Result<ThoTensor<D, Id>, ThoError> {
        // Do not acquire two shared tensor locks at once (including when self == rhs).
        let rhs = tch_result(rhs.t_ref().f_alias())?;
        ThoTensor::new(self.id.clone(), tch_result(self.t_ref().f_add(&rhs))?)
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
    pub fn try_t_ref(&self) -> Result<TensorRef<'_>, ThoError> {
        match &self.inner {
            ThoTensorInner::Val { t } => Ok(TensorRef(t.lock())),
            ThoTensorInner::Ref { .. } => Err(ThoError::new("local tensor access requires materializing the worker reference")),
        }
    }

    pub fn try_t_mut(&mut self) -> Result<&mut Tensor, ThoError> {
        match &mut self.inner {
            ThoTensorInner::Val { t } => Ok(t.get_mut()),
            ThoTensorInner::Ref { .. } => Err(ThoError::new("local mutation requires materializing the worker reference")),
        }
    }

    pub fn try_into_tensor(self) -> Result<Tensor, ThoError> {
        match self.inner {
            ThoTensorInner::Val { t } => Ok(t.into_inner()),
            ThoTensorInner::Ref { .. } => Err(ThoError::new("local ownership requires materializing the worker reference")),
        }
    }

    /// Locks this local Tensor handle. Keep the guard within synchronous code and
    /// avoid holding guards for multiple tensors in inconsistent orders.
    pub fn t_ref(&self) -> TensorRef<'_> {
        self.inner_tensor_ref()
    }

    /// Borrowing a local handle is synchronous: the guard must end before an await.
    pub fn as_ref(&self) -> TensorRef<'_> {
        self.t_ref()
    }

    pub fn t(self) -> Tensor {
        self.into_inner_tensor()
    }

    pub fn to_mut<'a>(&'a mut self) -> ThoTensorMut<'a, D, Id> {
        let Self { id, inner, .. } = self;
        let t = match inner {
            ThoTensorInner::Val { t } => t.get_mut(),
            ThoTensorInner::Ref { key, .. } => {
                error!("ThoTensor::to_mut: {:?}", Backtrace::capture());
                panic!("out-of-process ThoTensor ref is not mutable in this process, key={key:?}")
            },
        };
        ThoTensorMut { id, t, pd: Default::default() }
    }

    pub fn double_value(&self, idx: &[i64]) -> f64 {
        self.t_ref().double_value(idx)
    }

    pub fn tho_device_idx(&self) -> usize {
        match &self.inner {
            ThoTensorInner::Val { t: _t } => ThoDevice::try_from(self.t_ref().device()).expect("CPU/CUDA tensor device").tho_device_idx(),
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
    pub fn selector(&self) -> Option<Box<dyn ThoSelector>> {
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

impl From<ThoTensor> for Tensor {
    fn from(value: ThoTensor) -> Self {
        match value.inner {
            ThoTensorInner::Val { t } => t.into_inner(),
            ThoTensorInner::Ref { .. } => {
                error!("ThoTensor::from(value: ThoTensor): {:?}", Backtrace::capture());
                panic!("Can't obtain tensor from ref ThoTensor") //TODO
            }
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
                track_await!(cuda_actor.copy(src.into(), dst_device)).await?
            }
            ThoTensorInner::Ref { key, size, kind, selector, .. } => {
                debug!("CopyToDevice for ThoTensor: self.id={:?} copy out-of-process ref key={key:?} dst_device.ordinal={:?}", self.id, dst_device.tho_device_idx());
                track_await!(cuda_actor.copy(
                    ThoTensor::clone_ref_fields((), key, size, kind, selector)?, //TODO remove extra ThoTensor::new_ref
                    dst_device,
                ))
                .await?
            }
        };

        copy.to_types(id)
    }
}

impl<D: Dims, Id> Output for ThoTensor<D, Id> where Id: Named + Debug + Clone + Debug + Send + Sync + Eq + std::hash::Hash + 'static {}

pub struct ThoTensorMut<'a, D: Dims, Id: Named + Debug + Clone> {
    #[expect(dead_code, reason = "Keeps the typed identity borrow tied to the mutable tensor view.")]
    id: &'a Id,
    t: &'a mut Tensor,
    pd: PhantomData<D>,
}

impl<'a, D: Dims, Id: Named + Debug + Clone> ThoTensorMut<'a, D, Id> {
    pub fn new(id: &'a Id, t: &'a mut Tensor) -> Self {
        Self { id, t, pd: Default::default() }
    }

    pub fn add_(&mut self, rhs: &ThoTensor<D, Id>) -> Result<(), ThoError> {
        let _ = tch_result(self.t.f_add_(&rhs.t_ref()))?;

        Ok(())
    }

    pub fn copy_partial_from_(&mut self, dim: i64, start: i64, length: i64, rhs: &ThoTensor<D, Id>) -> Result<(), ThoError> {
        // Preserve typed CUDA failures at the tensor boundary for split assembly.
        tch_result(tch_result(self.t.f_narrow(dim, start, length))?.f_copy_(&rhs.t_ref()))?;
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
    fn test_storage_capabilities_return_errors_for_the_wrong_variant() -> Result<(), ThoError> {
        let installed = crate::runner::global_ctx::MockGlobalCtx::is_installed_context();
        installed.expect().return_const(false);
        let local: ThoTensor = tch_result(Tensor::f_from_slice(&[1f32]))?.into();
        assert!(local.clone_ref().is_err());
        assert!(local.try_t_ref().is_ok());
        let mut remote = ThoTensor::<AnyDims, ()>::new_ref((), 999, vec![1], Kind::Float, None, CudaWorkerId { device_idx: 1, generation: CudaDeviceGeneration(0) });
        assert!(remote.to_kind(Kind::Double).is_err());
        assert!(remote.try_t_ref().is_err());
        assert!(remote.try_t_mut().is_err());
        assert!(remote.try_into_tensor().is_err());
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
    assert_not_impl_any!(TensorRef<'static>: Send, Sync);
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
                    assert_eq!(tch_result(output.t_ref().f_double_value(&[0]))?, 3.);
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
            let guard = tensor.t_ref();
            // Repeated reads of the same handle on one thread must not deadlock.
            assert_eq!(tch_result(tensor.t_ref().f_double_value(&[0]))?, 2.);
            let tensor_ref = &tensor;
            scope.spawn(move || {
                let ThoTensorInner::Val { t } = &tensor_ref.inner else { unreachable!() };
                assert!(t.try_lock().is_none(), "another thread must not access a borrowed Tensor handle");
            }).join().expect("tensor access thread panicked");
            drop(guard);
            scope.spawn(|| -> Result<(), ThoError> {
                assert_eq!(tch_result(tensor.t_ref().f_double_value(&[1]))?, 3.);
                Ok(())
            }).join().expect("tensor access thread panicked")?;
            Ok(())
        })
    }

    #[test]
    fn test_shared_tensor_addition_in_both_orders() -> Result<(), ThoError> {
        let a = ThoTensor::<AnyDims, ()>::new((), tch_result(Tensor::f_from_slice(&[2f32]))?)?;
        let b = ThoTensor::<AnyDims, ()>::new((), tch_result(Tensor::f_from_slice(&[3f32]))?)?;
        // Concurrent callers must not hold A then B and B then A simultaneously.
        std::thread::scope(|scope| -> Result<(), ThoError> {
            let forward = scope.spawn(|| -> Result<(), ThoError> {
                for _ in 0..100 { assert_eq!(tch_result(a.add(&b)?.t_ref().f_double_value(&[0]))?, 5.); }
                Ok(())
            });
            let reverse = scope.spawn(|| -> Result<(), ThoError> {
                for _ in 0..100 { assert_eq!(tch_result(b.add(&a)?.t_ref().f_double_value(&[0]))?, 5.); }
                Ok(())
            });
            forward.join().expect("tensor addition thread panicked")?;
            reverse.join().expect("tensor addition thread panicked")?;
            Ok(())
        })
    }
}
