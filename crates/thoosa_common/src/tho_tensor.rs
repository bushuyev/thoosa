use std::borrow::Borrow;
use crate::tho_device::ThoDevice;
use crate::dims::{AnyDims, Dims};
use thoosa_shared::{panic_to_err, simple_id};
use async_trait::async_trait;
use std::fmt::{Debug, Display, Formatter};
use std::marker::PhantomData;
use tch::{Device, Kind, Tensor};
use thoosa_shared::{CopyToDevice, ThoError};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, MonoMetrics, Named, SimpleId, WithId};
use torch_sys::IntList;
use log::{debug, info, trace, warn};
use mockall_double::double;
use thoosa_shared::copy_mode::CpuOrGpu;
#[double]
use crate::runner::global_ctx::GlobalCtx;
use thoosa_shared::tracked_await::track_await;


//TODO remove, it should be in DevicesWrap between async
unsafe impl<D: Dims, Id: Named + Debug + Clone> Sync for ThoTensor<D, Id> {}

impl<I:Named + Debug + Clone + Send + Sync, D: Dims> Input for ThoTensor<D, I> {}
impl<I:Named + Debug + Clone + Send + Sync, D: Dims> Input for &ThoTensor<D, I> {}

impl<D: Dims> WithId for &ThoTensor<D, SimpleId> {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}



impl<D: Dims, Id: Named + Debug + Clone> AsRef<Tensor> for ThoTensor<D, Id> {
    fn as_ref(&self) -> &Tensor {
        self.t_ref()
    }
}

impl<D: Dims> ThoTensor<D, SimpleId> {
    //TODO remove
    pub fn shallow_clone<DO: Dims>(&self) -> ThoTensor<DO, SimpleId> {
        ThoTensor {
            id: self.id.clone(),
            inner: ThoTensorInner::Val { t: self.t_ref().shallow_clone() },
            pd: Default::default(),
            tracking: Default::default(),
        }
    }

    pub fn drop(self) {
        drop(self.inner);
    }


}

impl<D: Dims, Id: Named + Debug + Clone> ThoTensor<D, Id> {


    pub fn to_kind(&self, k: Kind) -> Result<ThoTensor<D, Id>, ThoError> {
        Ok(ThoTensor {
            id: self.id.clone(),
            inner: ThoTensorInner::Val { t: self.t_ref().f_to_kind(k)? },
            tracking: Default::default(),
            pd: PhantomData,
        })
    }

}

impl<D: Dims> Display for ThoTensor<D, SimpleId> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.t_ref(), f)
    }
}


enum ThoTensorInner {
    Val { t: Tensor },
    Ref { key: u64 },
}

impl Debug for ThoTensorInner {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ThoTensorInner::Val { t } => f.debug_struct("Val").field("t", t).finish(),
            ThoTensorInner::Ref { key } => f.debug_struct("Ref").field("key", key).finish(),
        }
    }
}


//D:Dims - is just a reminder of what dimensions a tensor has.
pub struct ThoTensor<D = AnyDims, Id = ()>
where
    D:Dims,
    Id:Named + Debug + Clone
{
    id: Id,
    inner: ThoTensorInner,

    #[allow(dead_code)]
    tracking: parking_lot::RwLock<Vec<&'static str>>,

    pd: PhantomData<D>,
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

impl <D:Dims, Id:Named  + Debug + Clone + Send + Sync> Adjustable for ThoTensor<D, Id> {}

impl<D, Id> Data for ThoTensor<D, Id>
where
    D: Dims,
    Id: Named + Debug + Clone + Send + Sync
{
    type Metrics = D::Metrics;
    type M = CpuOrGpu<ThoTensor<D, Id>>;

    fn size(&self) -> Self::Metrics {
        self.t_ref()
            .size()
            .try_into()
            .map_err(|_| ThoError::new("tensor size metrics conversion failed"))
            .unwrap_or(Self::Metrics::default())
    }

    fn device(&self) -> usize {
        ThoDevice::from(self.t_ref().device()).ordinal()
    }

    fn mem_size(&self) -> u64 {
        self.size().iter().fold(1u64, | a, v| a * (*v as u64)) * self.t_ref().kind().elt_size_in_bytes() as u64
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
            ThoTensorInner::Ref { key } => todo!("out-of-process ThoTensor ref is not implemented yet, key={key}"),
        }
    }

    fn inner_tensor_mut(&mut self) -> &mut Tensor {
        match &mut self.inner {
            ThoTensorInner::Val { t } => t,
            ThoTensorInner::Ref { key } => todo!("out-of-process ThoTensor ref is not mutable in this process, key={key}"),
        }
    }

    fn into_inner_tensor(self) -> Tensor {
        match self.inner {
            ThoTensorInner::Val { t } => t,
            ThoTensorInner::Ref { key } => todo!("out-of-process ThoTensor ref cannot be consumed as a local tensor yet, key={key}"),
        }
    }

    pub(crate) fn out_of_process_ref(id: Id, key: u64) -> Self {
        Self {
            id,
            inner: ThoTensorInner::Ref { key },
            tracking: Default::default(),
            pd: PhantomData,
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
            inner: ThoTensorInner::Val {
                t
            },
            tracking: Default::default(),
            pd: PhantomData,
        })
    }

    pub fn new_ref(id: Id, key:u64) -> Self {

        Self {
            id,
            inner: ThoTensorInner::Ref {
                key
            },
            tracking: Default::default(),
            pd: PhantomData,
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
        self.t_ref().size()
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

    // #[cfg(test)]
    pub fn t_ref(&self) -> &Tensor {
        self.inner_tensor_ref()
    }

    pub fn t_mut(&mut self) -> &mut Tensor {
        self.inner_tensor_mut()
    }

    pub fn t(self) -> Tensor {
        self.into_inner_tensor()
    }



    // No-params variant: transform tensor with a function that only needs &Tensor
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
            ThoTensorInner::Ref { key } => todo!("out-of-process ThoTensor ref is not mutable in this process, key={key}"),
        };
        ThoTensorMut {
            id,
            t,
            pd: Default::default(),
        }
    }

    pub fn double_value(&self, idx: &[i64]) -> f64 {
        self.t_ref().double_value(idx)
    }

    pub fn device_idx(&self) -> usize {
        ThoDevice::from(self.t_ref().device()).ordinal()
    }

    pub fn tho_device(&self) -> ThoDevice {
        ThoDevice::from(self.t_ref().device())
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
        debug!("CopyToDevice for ThoTensor: self.id={:?} self.t.size={:?} self.t.device={:?} dst_device={dst_device}", self.id, self.t_ref().size(), self.t_ref().device());

        //should not copy device to itself
        let dst_device = ThoDevice::from(dst_device);

        assert!(!dst_device.eq(&ThoDevice::from(self.t_ref().device())));
        let id = self.id.clone();

        let ctx_device = if dst_device.is_cuda() {
            dst_device
        }  else {
            self.tho_device()
        };

        let src_tensor = self.t_ref();
        let src = if src_tensor.is_contiguous() { None } else {
            warn!("CopyToDevice for ThoTensor: tensor was not contiguous, self.id={:?}", self.id);
            Some(src_tensor.f_contiguous()?)
        };
        let src_ref = src.as_ref().unwrap_or(src_tensor);
        debug!("CopyToDevice for ThoTensor: self.id={:?} copy copy_from_ptr, dst_device.ordinal={:?} src_contiguous={}", self.id, dst_device.ordinal(), src.is_none());


        let cuda_actor = GlobalCtx::global().cuda_actor(ctx_device);
        let copy = track_await(cuda_actor
            .copy(
                src_ref.shallow_clone().into(),
                dst_device,
            )).await?;

        debug!("CopyToDevice for ThoTensor: self.id={:?} self.t.size={:?} self.t.device={:?} dst_device={dst_device:?} copy.device={:?}",self.id,  self.t_ref().size(), self.t_ref().device(), copy.device());
        trace!("CopyToDevice for ThoTensor: self.id={:?} self.t=\n{} copy=\n{copy}", self.id, self.t_ref());

        Ok(ThoTensor::new(id, copy)?)
    }
}

impl<D: Dims, Id> Output for ThoTensor<D, Id>
where
    Id: Named + Debug + Clone + Debug + Send + Sync + Eq + std::hash::Hash + 'static
{}


pub struct ThoTensorMut<'a, D: Dims, Id: Named + Debug + Clone> {
    id: &'a Id,
    t: &'a mut Tensor,
    pd: PhantomData<D>,
}

impl<'a, D: Dims, Id: Named + Debug + Clone> ThoTensorMut<'a, D, Id> {
    pub fn new(id: &'a Id, t: &'a mut Tensor) -> Self {
        Self {
            id: id,
            t,
            pd: Default::default(),
        }
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

    pub fn copy_partial_from_(
        &mut self,
        dim: i64,
        start: i64,
        length: i64,
        rhs: &ThoTensor<D, Id>,
    ) {
        self.t.narrow(dim, start, length).copy_(rhs.t_ref());
    }

    pub fn resize_(&mut self, new_shape: &[i64]) {
        let _ = self.t.resize_(new_shape);
    }
}
