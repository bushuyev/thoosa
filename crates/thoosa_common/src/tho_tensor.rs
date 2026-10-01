use std::backtrace::Backtrace;
use crate::dims::{AnyDims, ChannelsData, Dims};
use crate::runner::common::cachable::ThoDataRef;
use thoosa_shared::traits::RunCtx;
#[double]
use crate::runner::global_ctx::GlobalCtx;
use crate::tho_device::ThoDevice;
use async_trait::async_trait;
use log::{debug, error, info, trace, warn};
use mockall_double::double;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::Deref;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::{Arc, OnceLock};
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering::Relaxed;
use parking_lot::Mutex;
use tch::{Device, Kind, Tensor};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, MonoMetrics, Named, DebugId, ThoSelector, WithId};
use thoosa_shared::{n_trace_line, CopyToDevice, ThoError};
use thoosa_shared::{panic_to_err, debug_id};
use torch_sys::IntList;
use thoosa_shared::comp_id::SynKey;
use crate::ipc::worker_msg::{CudaWorkerId, WorkerInput};
use thoosa_shared::ipc_types::{IpcTensorKind, WorkerOutput};


//TODO remove, it should be in DevicesWrap between async
unsafe impl<D: Dims, Id: Named + Debug + Clone> Sync for ThoTensor<D, Id> {}

impl<I: Named + Debug + Clone + Send + Sync, D: Dims> Input for ThoTensor<D, I> {}
impl<I: Named + Debug + Clone + Send + Sync, D: Dims> Input for &ThoTensor<D, I> {}

impl<D: Dims, Id: Named + Debug + Clone> AsRef<Tensor> for ThoTensor<D, Id> {
    fn as_ref(&self) -> &Tensor {
        self.t_ref()
    }
}

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
                    ThoTensorInner::Val { t: t.shallow_clone() }
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
            inner: ThoTensorInner::Val { t: self.t_ref().f_to_kind(k)? }, //TODO ref
            pd: PhantomData,
        })
    }

    pub fn kind(&self) -> Kind {
        match &self.inner {
            ThoTensorInner::Val { t } => t.kind(),
            ThoTensorInner::Ref { kind, .. } => kind.clone(),
        }
    }

    pub fn clone_ref(&self) -> Result<Self, ThoError> {
        match &self.inner {
            ThoTensorInner::Val { .. } => {
                panic!("can't clone val")
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
                ThoTensor::new(id, t)
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
        Display::fmt(self.t_ref(), f)
    }
}

//drop can't be implemented directly on ThoTensor or ThoTensorInner to keep them movable
#[derive(Debug)]
pub struct KeyHolder {
    pub key:u64,
    pub worker_id: CudaWorkerId,
    count:Arc<()>,
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

impl Drop for KeyHolder {
    fn drop(&mut self) {
        debug!("KeyHolder::drop: self={self:?} self.worker_id.device_idx={} strong_count={:?} GlobalCtx::is_installed()={}", self.worker_id.device_idx, Arc::strong_count(&self.count), GlobalCtx::is_installed() );
        if self.do_drop && self.worker_id.device_idx != 0 && GlobalCtx::is_installed() && Arc::strong_count(&self.count) == 2 {
            debug!("KeyHolder::drop: self.key={:?}: {}", self.key, n_trace_line(&Backtrace::capture(), 50) );

            let device_idx = self.worker_id.device_idx;
            let key = self.key;
            let worker_id = self.worker_id;
            tokio::spawn(async move {
                if let Err(e) = GlobalCtx::global().cuda_actor(device_idx).drop_tho_tensor(key, worker_id).await {
                    error!("Can't drop tensor key={key:?} e={e:?}");
                }
            });
        }
    }
}


pub /*TODO*/ enum ThoTensorInner {
    Val { t: Tensor },
    Ref { key: KeyHolder, size: Vec<i64>, kind: Kind, selector: Option<Box<dyn ThoSelector>> },
}

impl Debug for ThoTensorInner {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ThoTensorInner::Val { t } => f.debug_struct("Val")
                .field("t", t)
                .field("device", &t.device())
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
    fn tensor_ref(&self) -> &Tensor;
}

impl<D, Id> TensorInitRef for ThoTensor<D, Id>
where
    D: Dims,
    Id: Named + Debug + Clone + Send + Sync,
{
    fn tensor_ref(&self) -> &Tensor {
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

impl<D: Dims, Id: Named + Debug + Clone> Borrow<Tensor> for ThoTensor<D, Id> {
    fn borrow(&self) -> &Tensor {
        self.t_ref()
    }
}

impl<D: Dims> From<Tensor> for ThoTensor<D, DebugId> {
    fn from(tensor: Tensor) -> Self {
        Self {
            id: debug_id!("1"),
            inner: ThoTensorInner::Val { t: tensor },
            pd: PhantomData,
        }
    }
}

impl From<Tensor> for ThoTensor {
    fn from(tensor: Tensor) -> Self {
        Self {
            id: (),
            inner: ThoTensorInner::Val { t: tensor },
            pd: PhantomData,
        }
    }
}

impl<D: Dims> From<&Tensor> for ThoTensor<D, DebugId> {
    fn from(tensor: &Tensor) -> Self {
        Self {
            id: debug_id!("2"),
            inner: ThoTensorInner::Val { t: tensor.shallow_clone() },
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
            ThoTensorInner::Val { t } => t.size().try_into().unwrap_or_default(),
            ThoTensorInner::Ref { size, .. } => Self::Metrics::try_from(size.clone()).unwrap_or_else(|e| panic!("Tho tensor size metrics conversion failed")),
        }
    }

    fn tho_device_idx(&self) -> usize {

        match &self.inner {
            ThoTensorInner::Val { t } => ThoDevice::from(t.device()).tho_device_idx(),
            ThoTensorInner::Ref { key, .. } => key.worker_id.device_idx,
        }
    }

    fn mem_size(&self) -> u64 {
        match &self.inner {
            ThoTensorInner::Val { t } => t.size().iter().fold(1u64, |a, v| a * (*v as u64)) * t.kind().elt_size_in_bytes() as u64,
            ThoTensorInner::Ref { key, .. } => {
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
    fn inner_tensor_ref(&self) -> &Tensor {
        match &self.inner {
            ThoTensorInner::Val { t } => t,
            ThoTensorInner::Ref { key, .. } => {
                error!("ThoTensor::inner_tensor_ref: {:?}", Backtrace::capture());
                panic!("can't get ref of a remote tensor key={key:?}")
            },
        }
    }

    fn inner_tensor_mut(&mut self) -> &mut Tensor {
        match &mut self.inner {
            ThoTensorInner::Val { t } => t,
            ThoTensorInner::Ref { key, .. } => {
                error!("ThoTensor::inner_tensor_mut: {:?}", Backtrace::capture());
                panic!("can't get mut ref of a remote tensor key={key:?}")
            },
        }
    }

    fn into_inner_tensor(self) -> Tensor {
        match self.inner {
            ThoTensorInner::Val { t } => t,
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
            tensor.f_contiguous()?
        };

        Ok(Self {
            id,
            inner: ThoTensorInner::Val { t },
            pd: PhantomData,
        })
    }

    pub fn new_ref(id: Id, key: u64, size: Vec<i64>, kind: tch::Kind, selector: Option<Box<dyn ThoSelector>>, worker_id: CudaWorkerId) -> Self {
        let tho_device_idx = worker_id.device_idx;

        // #[cfg(tho_debug)]
        // let key_holder = if GlobalCtx::is_installed(){
        debug!("KeyHolder::new_ref: id={id:?} key={key} size={size:?} kind={kind:?} selector={selector:?} tho_device_idx={tho_device_idx:?}");

        // let trace = n_trace_line(&Backtrace::capture(), 200);
        static KEYS:OnceLock<Mutex<HashMap<(CudaWorkerId, u64), Arc<()>>>> = OnceLock::new();
        let count = Arc::clone(KEYS.get_or_init(||Mutex::new(HashMap::new())).lock().entry((worker_id, key)).or_insert(Arc::new(())));

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
            inner: ThoTensorInner::Val { t: Tensor::f_zeros(size, options)? },
            pd: PhantomData,
        })
    }

    pub fn requires_grad(&self) -> bool {
        self.t_ref().requires_grad()
    }
    pub fn size(&self) -> Vec<i64> {
        match &self.inner {
            ThoTensorInner::Val { t } => t.size(),
            ThoTensorInner::Ref { size, .. } => size.clone(),
        }
    }
    pub fn numel(&self) -> usize {
        self.t_ref().numel()
    }

    pub fn add(&self, rhs: &ThoTensor<D, Id>) -> Result<ThoTensor<D, Id>, ThoError> {
        ThoTensor::new(self.id.clone(), self.t_ref() + rhs.t_ref())
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

    pub fn t_ref(&self) -> &Tensor {
        self.inner_tensor_ref()
    }

    pub fn t(self) -> Tensor {
        self.into_inner_tensor()
    }

    pub fn to_mut<'a>(&'a mut self) -> ThoTensorMut<'a, D, Id> {
        let Self { id, inner, .. } = self;
        let t = match inner {
            ThoTensorInner::Val { t } => t,
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
            ThoTensorInner::Val { t } => ThoDevice::from(self.t_ref().device()).tho_device_idx(),
            ThoTensorInner::Ref { key, .. } => key.worker_id.device_idx,
        }
    }

    pub fn tho_device(&self) -> ThoDevice {
        match &self.inner {
            ThoTensorInner::Val { t } => ThoDevice::from(t.device()),
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
            ThoTensorInner::Val { t } => t,
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
                let src = if src.is_contiguous() {
                    src.shallow_clone()
                } else {
                    warn!("CopyToDevice for ThoTensor: tensor was not contiguous, self.id={:?}", self.id);
                    src.f_contiguous()?
                };

                debug!("CopyToDevice for ThoTensor: self.id={:?} copy copy_from_ptr, dst_device.ordinal={:?}", self.id, dst_device.tho_device_idx());
                track_await(cuda_actor.copy(src.into(), dst_device)).await?
            }
            ThoTensorInner::Ref { key, size, kind, selector, .. } => {
                debug!("CopyToDevice for ThoTensor: self.id={:?} copy out-of-process ref key={key:?} dst_device.ordinal={:?}", self.id, dst_device.tho_device_idx());
                track_await(cuda_actor.copy(
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
    id: &'a Id,
    t: &'a mut Tensor,
    pd: PhantomData<D>,
}

impl<'a, D: Dims, Id: Named + Debug + Clone> ThoTensorMut<'a, D, Id> {
    pub fn new(id: &'a Id, t: &'a mut Tensor) -> Self {
        Self { id, t, pd: Default::default() }
    }

    pub fn add_(&mut self, rhs: &ThoTensor<D, Id>) -> Result<(), ThoError> {
        let _ = self.t.f_add_(rhs.t_ref())?;

        Ok(())
    }

    pub fn copy_partial_from_(&mut self, dim: i64, start: i64, length: i64, rhs: &ThoTensor<D, Id>) {
        self.t.narrow(dim, start, length).copy_(rhs.t_ref());
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
