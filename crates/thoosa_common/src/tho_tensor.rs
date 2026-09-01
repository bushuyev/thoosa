use std::backtrace::Backtrace;
use crate::dims::{AnyDims, ChannelsData, ChannelsMotifsSlices, Dims, LabelProbability, MotifProbability, MotifSlices};
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
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::Deref;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering::Relaxed;
use tch::{Device, Kind, Tensor};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, MonoMetrics, Named, SimpleId, ThoSelector, WithId};
use thoosa_shared::{CopyToDevice, ThoError};
use thoosa_shared::{panic_to_err, simple_id};
use torch_sys::IntList;
use thoosa_shared::comp_id::SynKey;
use crate::ipc::worker_msg::{WorkerInput, WorkerOutput};


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
    //TODO remove
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
            tracking: Default::default(),
        }
    }

    pub fn drop_inner(self) {
        drop(self.inner);
    }

    pub fn to_kind(&self, k: Kind) -> Result<ThoTensor<D, Id>, ThoError> {
        Ok(ThoTensor {
            id: self.id.clone(),
            inner: ThoTensorInner::Val { t: self.t_ref().f_to_kind(k)? }, //TODO ref
            tracking: Default::default(),
            pd: PhantomData,
        })
    }

    pub fn kind(&self) -> Kind {
        match &self.inner {
            ThoTensorInner::Val { t } => t.kind(),
            ThoTensorInner::Ref { kind, .. } => kind.clone(),
        }
    }

    pub fn to_default(self) -> ThoTensor {
        ThoTensor {
            id: (),
            inner: self.inner,
            tracking: Default::default(),
            pd: Default::default(),
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
                        tracking: Default::default(),
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
            tracking: Default::default(),
            pd: Default::default(),
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
                    tracking: Default::default(),
                    pd: PhantomData,
                })
            }
        }

    }

}

impl<D: Dims> Display for ThoTensor<D, SimpleId> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.t_ref(), f)
    }
}

#[derive(Debug)]
pub struct KeyHolder {
    pub key:u64,
    pub device_idx:usize,
    count:Arc<()>
}

impl Clone for KeyHolder {
    fn clone(&self) -> Self {
        KeyHolder {
            key:  self.key,
            device_idx: self.device_idx,
            count: Arc::clone(&self.count),
        }
    }
}

impl Drop for KeyHolder {
    fn drop(&mut self) {
        debug!("KeyHolder::drop: self={self:?} self.device_idx={} strong_count={:?}", self.device_idx, Arc::strong_count(&self.count) );
        // if GlobalCtx::is_installed() && self.device_idx != 0 && Arc::strong_count(&self.count) == 1 {
        //     let device_idx = self.device_idx;
        //     let key = self.key;
        //     tokio::spawn(async move {
        //         if let Err(e) = GlobalCtx::global().cuda_actor(device_idx).drop_tho_tensor(key).await {
        //             error!("Can't drop tensor key={key:?}");
        //         }
        //     });
        // }
    }
}
//
// impl Deref for KeyHolder {
//     type Target = u64;
//
//     fn deref(&self) -> &Self::Target {
//         &self.key
//     }
// }
//
// impl From<u64> for KeyHolder {
//     fn from(key: u64) -> Self {
//         Self {
//             key
//         }
//     }
// }

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


// impl Drop for ThoTensorInner {
//     fn drop(&mut self) {
//         if let ThoTensorInner::Ref { key, tho_device_idx, .. } = self {
//             let device_idx = *tho_device_idx;
//             let key = *key;
//             tokio::spawn(async move {
//
//                 if let Err(e) = GlobalCtx::global().cuda_actor(device_idx).drop_tho_tensor(key).await {
//                     error!("Can't drop tensor key={key:?}");
//                 }
//             });
//         }
//     }
// }



//D:Dims - is just a reminder of what dimensions a tensor has.
pub struct ThoTensor<D = AnyDims, Id = ()>
where
    D: Dims,
    Id: Named + Debug + Clone,
{
    pub id: Id,
    pub inner: ThoTensorInner, //TODO pub

    #[allow(dead_code)]
    tracking: parking_lot::RwLock<Vec<&'static str>>,

    pd: PhantomData<D>,
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

impl<D: Dims> From<Tensor> for ThoTensor<D, SimpleId> {
    fn from(tensor: Tensor) -> Self {
        Self {
            id: simple_id!("1"),
            inner: ThoTensorInner::Val { t: tensor },
            pd: PhantomData,
            tracking: Default::default(),
        }
    }
}

impl From<Tensor> for ThoTensor {
    fn from(tensor: Tensor) -> Self {
        Self {
            id: (),
            inner: ThoTensorInner::Val { t: tensor },
            pd: PhantomData,
            tracking: Default::default(),
        }
    }
}

impl<D: Dims> From<&Tensor> for ThoTensor<D, SimpleId> {
    fn from(tensor: &Tensor) -> Self {
        Self {
            id: simple_id!("2"),
            inner: ThoTensorInner::Val { t: tensor.shallow_clone() },
            pd: PhantomData,
            tracking: Default::default(),
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
    type M = CpuOrGpu<ThoTensor<D, Id>>;

    fn size(&self) -> Self::Metrics {
        match &self.inner {
            ThoTensorInner::Val { t } => t.size().try_into().map_err(|_| ThoError::new("tensor size metrics conversion failed")).unwrap_or(Self::Metrics::default()),
            ThoTensorInner::Ref { size, .. } => Self::Metrics::try_from(size.clone()).unwrap_or_else(|e| panic!("Tho tensor size metrics conversion failed")),
        }
    }

    fn device(&self) -> usize {

        match &self.inner {
            ThoTensorInner::Val { t } => ThoDevice::from(t.device()).tho_device_idx(),
            ThoTensorInner::Ref { key, .. } => key.device_idx,
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

    async fn adrop(self)//remove
    where
        Self:Sized
    {
        // todo!()
        let tensor = self.to_default();
        let tho_device_idx_key = if let ThoTensorInner::Ref{ key, size, kind, selector } = &tensor.inner {
            Some((key.device_idx, key.key))

        } else {
            None
        };

        if let Some((tho_device_idx, key)) = tho_device_idx_key {
            if let Err(e) = GlobalCtx::global().cuda_actor(tho_device_idx).drop_tho_tensor(key).await {
                error!("Got error while dropping key={key:?} e={e:?} ");
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
            ThoTensorInner::Ref { key, .. } => todo!("out-of-process ThoTensor ref is not implemented yet, key={key:?}"),
        }
    }

    fn inner_tensor_mut(&mut self) -> &mut Tensor {
        match &mut self.inner {
            ThoTensorInner::Val { t } => t,
            ThoTensorInner::Ref { key, .. } => todo!("out-of-process ThoTensor ref is not mutable in this process, key={key:?}"),
        }
    }

    fn into_inner_tensor(self) -> Tensor {
        match self.inner {
            ThoTensorInner::Val { t } => t,
            ThoTensorInner::Ref { key, .. } => todo!("out-of-process ThoTensor ref cannot be consumed as a local tensor yet, key={key:?}"),
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
            tracking: Default::default(),
            pd: PhantomData,
        })
    }

    pub fn new_ref(id: Id, key: u64, size: Vec<i64>, kind: tch::Kind, selector: Option<Box<dyn ThoSelector>>, tho_device_idx: usize) -> Self {
        Self {
            id,
            inner: ThoTensorInner::Ref {
                key:KeyHolder{
                    key,
                    device_idx: tho_device_idx,
                    count: Arc::new(()),
                },
                size, kind, selector,
            },
            tracking: Default::default(),
            pd: PhantomData,
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
            /*                               at /rustc/b15a874aafe7eab9ea3ac2c1d59c7b03e1425027/library/core/src/panicking.rs:80:14
              16:     0x650bc6172c06 - core::result::unwrap_failed::hb3955d4a33f491ad
                                           at /rustc/b15a874aafe7eab9ea3ac2c1d59c7b03e1425027/library/core/src/result.rs:1862:5
              17:     0x650bc2986325 - core::result::Result<T,E>::unwrap::h084cb97a7d73525b
                                           at /home/bu/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib/rustlib/src/rust/library/core/src/result.rs:1233:23
              18:     0x650bc2986325 - tch::wrappers::tensor_generated::<impl tch::wrappers::tensor::Tensor>::zeros::h989c1bb7ff9d3639
                                           at /home/bu/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/tch-0.20.0/src/wrappers/tensor_generated.rs:19386:40
              19:     0x650bc2986325 - thoosa_common::tho_tensor::ThoTensor<D,Id>::zeros::hec7444fdf1238b68
                                           at /home/bu/dvl/rust/thoosa/crates/thoosa_common/src/tho_tensor.rs:233:16
              20:     0x650bc2986325 - thoosa_main::tasks::motif_loader::load_motif_slices::{{closure}}::{{closure}}::ha04a296d1395eb8b
                                           at /home/bu/dvl/rust/thoosa/crates/thoosa_main/src/tasks/motif_loader.rs:346:46
              21:     0x650bc29851cc - <tracing::instrument::Instrumented<T> as core::future::future::Future>::poll::h285e9afbf1ed55e9
                                           at /home/bu/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/tracing-0.1.43/src/instrument.rs:321:15
              22:     0x650bc29851cc - thoosa_main::tasks::motif_loader::load_motif_slices::{{closure}}::hcda8bc73e874ad61
                                           at /home/bu/dvl/rust/thoosa/crates/thoosa_main/src/tasks/motif_loader.rs:304:1
            */
            inner: ThoTensorInner::Val { t: Tensor::f_zeros(size, options)? },
            tracking: Default::default(),
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
        let Self { inner, tracking, pd, .. } = self;
        ThoTensor { id, inner, tracking, pd }
    }

    // #[cfg(test)]
    pub fn t_ref(&self) -> &Tensor {
        //TODO remove
        self.inner_tensor_ref()
    }

    pub fn t_mut(&mut self) -> &mut Tensor {
        self.inner_tensor_mut()
    }

    pub fn t(self) -> Tensor {
        self.into_inner_tensor()
    }

    // TODO remove
    pub fn run_inner<R, F>(&self, func: F) -> Result<R, ThoError>
    where
        F: Fn(&Tensor) -> Result<R, ThoError>,
    {
        func(self.t_ref())
    }

    pub fn _track_used_in(&self, token: &'static str) {
        self.tracking.write().push(token);
    }

    pub fn to_mut<'a>(&'a mut self) -> ThoTensorMut<'a, D, Id> {
        let Self { id, inner, .. } = self;
        let t = match inner {
            ThoTensorInner::Val { t } => t,
            ThoTensorInner::Ref { key, .. } => todo!("out-of-process ThoTensor ref is not mutable in this process, key={key:?}"),
        };
        ThoTensorMut { id, t, pd: Default::default() }
    }

    pub fn double_value(&self, idx: &[i64]) -> f64 {
        self.t_ref().double_value(idx)
    }

    pub fn tho_device_idx(&self) -> usize {
        match &self.inner {
            ThoTensorInner::Val { t } => ThoDevice::from(self.t_ref().device()).tho_device_idx(),
            ThoTensorInner::Ref { key, .. } => key.device_idx,
        }
    }

    pub fn tho_device(&self) -> ThoDevice {
        match &self.inner {
            ThoTensorInner::Val { t } => ThoDevice::from(t.device()),
            ThoTensorInner::Ref { key, .. } => <usize as Into<ThoDevice>>::into(key.device_idx),
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
        let src_device = self.device().into();
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
            ThoTensorInner::Ref { key, size, kind, selector } => {
                debug!("CopyToDevice for ThoTensor: self.id={:?} copy out-of-process ref key={key:?} dst_device.ordinal={:?}", self.id, dst_device.tho_device_idx());
                track_await(cuda_actor.copy(
                    ThoTensor::new_ref((), key.key, size.clone(), *kind, selector.as_ref().map(|v| v.clone_box()), key.device_idx), //TODO remove extra ThoTensor::new_ref
                    dst_device,
                ))
                .await?
            }
        };

        // match copy.inner {//restores id
        //     ThoTensorInner::Ref { key, size, kind, selector } => {
        //         // Keep the actor-provided metadata so selectors and later D2H copies describe the actual worker tensor.
        //         debug!("CopyToDevice for ThoTensor: self.id={:?} dst_device={dst_device:?} copied to out-of-process ref key={key}", self.id);
        //         Ok(ThoTensor::new_ref(id, key, size, kind, selector, tho_device_idx))
        //     }
        //     ThoTensorInner::Val { t } => {
        //         debug!("CopyToDevice for ThoTensor: self.id={:?} dst_device={dst_device:?} copy.device={:?}", self.id, t.device());
        //         ThoTensor::new(id, t)
        //     }
        // }
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
        Self { id: id, t, pd: Default::default() }
    }

    pub fn zero_(&mut self) {
        let _ = self.t.zero_();
    }
    pub fn add_(&mut self, rhs: &ThoTensor<D, Id>) -> Result<(), ThoError> {
        let _ = self.t.f_add_(rhs.t_ref())?;

        Ok(())
    }
    pub fn copy_from_(&mut self, rhs: &ThoTensor<D, Id>) {
        self.t.copy_(rhs.t_ref());
    }

    pub fn copy_partial_from_(&mut self, dim: i64, start: i64, length: i64, rhs: &ThoTensor<D, Id>) {
        self.t.narrow(dim, start, length).copy_(rhs.t_ref());
    }

    pub fn resize_(&mut self, new_shape: &[i64]) {
        let _ = self.t.resize_(new_shape);
    }
}


#[typetag::serde(name = "ThoTensor_default")]
impl WorkerOutput for ThoTensor {}

#[typetag::serde(name = "ThoTensor_ChannelsData")]
impl WorkerOutput for ThoTensor<ChannelsData, SynKey> {}

#[typetag::serde(name = "ThoTensor_ChannelsMotifsSlices")]
impl WorkerOutput for ThoTensor<ChannelsMotifsSlices, SynKey> {}

#[typetag::serde(name = "ThoTensor_MotifSlices")]
impl WorkerOutput for ThoTensor<MotifSlices, SynKey> {}

#[typetag::serde(name = "ThoTensor_LabelProbability")]
impl WorkerOutput for ThoTensor<LabelProbability, SynKey> {}


#[typetag::serde(name = "ThoTensor_MotifProbability")]
impl WorkerOutput for ThoTensor<MotifProbability, SynKey>{}

// Postcard stage calls need concrete typetag registrations for tensor inputs used across the main workflow.
#[typetag::serde(name = "ThoTensor_ChannelsData_Input")]
impl WorkerInput for ThoTensor<ChannelsData, SynKey> {}

#[typetag::serde(name = "ThoTensor_ChannelsMotifsSlices_Input")]
impl WorkerInput for ThoTensor<ChannelsMotifsSlices, SynKey> {}

#[typetag::serde(name = "ThoTensor_MotifSlices_Input")]
impl WorkerInput for ThoTensor<MotifSlices, SynKey> {}

#[typetag::serde(name = "ThoTensor_LabelProbability_Input")]
impl WorkerInput for ThoTensor<LabelProbability, SynKey> {}

#[typetag::serde(name = "ThoTensor_MotifProbability_Input")]
impl WorkerInput for ThoTensor<MotifProbability, SynKey> {}
