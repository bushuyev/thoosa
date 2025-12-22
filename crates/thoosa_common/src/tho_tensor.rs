use std::borrow::Borrow;
use crate::applicables::ThoDevice;
use crate::copy_ctx::CopyCtx;
use crate::cuda_utils::tensor_mem_size;
use crate::dims::{AnyDims, Dimensional, Dims};
use thoosa_shared::panic_to_err;
use ambassador::delegatable_trait;
use async_trait::async_trait;
use mockall::automock;
use std::fmt::{Debug, Display, Formatter};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;
use tch::{Device, IndexOp, Kind, Tensor};
use thoosa_shared::{CopyToDevice, ThoosaError};
use thoosa_shared::applicables_runner::{CachableId, Input, Output};
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, MonoMetrics, Named, Select, SimpleId, WithId};
use torch_sys::IntList;
use log::{debug, trace};


impl<D: Dims, Id: Named + Debug + Clone> Dimensional for ThoTensor<D, Id> {
    type Dimensions = D;
}

//TODO remove, it should be in DevicesWrap between async
unsafe impl<D: Dims, Id: Named + Debug + Clone> Sync for ThoTensor<D, Id> {}

impl<D: Dims> Input for ThoTensor<D, CachableId> {
}

impl<D: Dims> WithId for &ThoTensor<D, CachableId> {
    type ID = CachableId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<D: Dims> Input for &ThoTensor<D, CachableId> {
}

impl<D: Dims> AsRef<Tensor> for ThoTensor<D, CachableId> {
    fn as_ref(&self) -> &Tensor {
        &self.t
    }
}

impl<D: Dims> ThoTensor<D, CachableId> {
    //TODO remove
    pub fn shallow_clone<DO: Dims>(&self) -> ThoTensor<DO, CachableId> {
        ThoTensor {
            id: self.id.clone(),
            t: self.t.shallow_clone(),
            pd: Default::default(),
            tracking: Default::default(),
        }
    }

    pub fn drop(self) {
        drop(self.t);
    }

    // pub fn new(id: CachableId, tensor: Tensor) -> Self {
    //     Self {
    //         id,
    //         t: tensor, _pd: PhantomData,
    //         tracking: Default::default(),
    //     }
    // }
}

impl<D: Dims, Id: Named + Debug + Clone> ThoTensor<D, Id> {


    pub fn to_kind(&self, k: Kind) -> ThoTensor<D, Id> {
        ThoTensor {
            id: self.id.clone(),
            t: self.t.to_kind(k),
            tracking: Default::default(),
            pd: PhantomData,
        }
    }
    pub fn to_device(&self, d: Device) -> Result<ThoTensor<D, Id>, ThoosaError> {
        ThoTensor::new(self.id.clone(), self.t.to_device(d))
    }
    pub fn contiguous(&self) -> Result<ThoTensor<D, Id>, ThoosaError> {
        ThoTensor::new(self.id.clone(), self.t.contiguous())
    }


}

impl<D: Dims> Display for ThoTensor<D, CachableId> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.t, f)
    }
}

pub struct ThoTensorMut<'a, D: Dims, Id: Named + Debug + Clone> {
    _id: &'a Id,
    t: &'a mut Tensor,
    pd: PhantomData<D>,
}
unsafe impl<'a, D: Dims, Id: Named + Debug + Clone> Send for ThoTensorMut<'a, D, Id> {}
// (no Sync)

impl<'a, D: Dims, Id: Named + Debug + Clone> ThoTensorMut<'a, D, Id> {
    pub fn new(id: &'a Id, t: &'a mut Tensor) -> Self {
        Self {
            _id: id,
            t,
            pd: Default::default(),
        }
    }

    pub fn zero_(&mut self) {
        let _ = self.t.zero_();
    }
    pub fn add_(&mut self, rhs: &ThoTensor<D, Id>) -> Result<(), ThoosaError> {
        let _ = self.t.f_add_(&rhs.t)?;

        Ok(())
    }
    pub fn copy_from_(&mut self, rhs: &ThoTensor<D, Id>) {
        self.t.copy_(&rhs.t);
    }

    pub fn copy_partial_from_(
        &mut self,
        _dim: i64,
        start: i64,
        length: i64,
        rhs: &ThoTensor<D, Id>,
    ) {
        self.t.narrow(0, start, length).copy_(&rhs.t);
    }

    pub fn resize_(&mut self, new_shape: &[i64]) {
        let _ = self.t.resize_(new_shape);
    }
}

pub struct ThoTensor<D: Dims, Id: Named + Debug + Clone> {
    id: Id,//TODO remove
    pub t: Tensor,

    #[allow(dead_code)]
    tracking: parking_lot::RwLock<Vec<&'static str>>,

    pd: PhantomData<D>,
}

impl<D: Dims, Id: Named + Debug + Clone> Borrow<Tensor> for ThoTensor<D, Id> {
    fn borrow(&self) -> &Tensor {
        &self.t
    }
}

impl<D: Dims> From<Tensor> for ThoTensor<D, CachableId> {
    fn from(tensor: Tensor) -> Self {
        Self {
            id: "1".into(),
            t: tensor,
            pd: PhantomData,
            tracking: Default::default(),
        }
    }
}

impl<D: Dims> From<&Tensor> for ThoTensor<D, CachableId> {
    fn from(tensor: &Tensor) -> Self {
        Self {
            id: "2".into(),
            t: tensor.shallow_clone(),
            pd: PhantomData,
            tracking: Default::default(),
        }
    }
}



impl <D:Dims, Id:Named  + Debug + Clone + Send> Adjustable for ThoTensor<D, Id> {}

impl<D: Dims, Id: Named + Debug + Clone> Data for ThoTensor<D, Id> {
    type Metrics = D::Metrics;


    //TODO rename to metrics
    fn size(&self) -> Self::Metrics {
        self.t
            .size()
            .try_into()
            .map_err(|e| ThoosaError::new("".to_string()))
            .unwrap() //TODO
    }


    fn device(&self) -> usize {
        ThoDevice::from(self.t.device()).ordinal()
    }

    fn mem_size(&self) -> u64 {
        self.size().iter().fold(1u64, | a, v| (a * (*v as u64))) * self.t.kind().elt_size_in_bytes() as u64
    }
}

impl<D: Dims, Id: Named + Debug + Clone> Debug for ThoTensor<D, Id> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("")
            .field("id.name", &self.id.name())
            .field("t", &self.t)
            .finish()
    }
}

impl<D: Dims, Id: Named + Debug + Clone> ThoTensor<D, Id> {
    pub fn new(id: Id, tensor: Tensor) -> Result<Self, ThoosaError> {
        Ok(Self {
            id,
            t: if tensor.is_contiguous() {
                tensor
            } else {
                tensor.f_contiguous()?
            },
            tracking: Default::default(),
            pd: PhantomData,
        })
    }

    pub fn zeros(id: Id, size: impl IntList, options: (Kind, Device)) -> Self {
        Self {
            id,
            t: Tensor::zeros(size, options),
            tracking: Default::default(),
            pd: PhantomData,
        }
    }

    pub fn requires_grad(&self) -> bool {
        self.t.requires_grad()
    }
    pub fn size(&self) -> Vec<i64> {
        self.t.size()
    }
    pub fn numel(&self) -> usize {
        self.t.numel()
    }



    pub fn add(&self, rhs: &ThoTensor<D, Id>) -> Result<ThoTensor<D, Id>, ThoosaError> {
        ThoTensor::new(self.id.clone(), &self.t + &rhs.t)
    }

    pub fn mm<RD: Dims>(
        &self,
        rhs: &ThoTensor<RD, Id>,
    ) -> Result<ThoTensor<D, Id>, ThoosaError> {
        let mm_r = panic_to_err!(self.t.f_matmul(&rhs.t))?;
        debug!(
            "mm: self.t.size={:?}, rhs.t.size={:?} mm_r.size={:?}",
            self.t.size(),
            rhs.t.size(),
            mm_r.size()
        );

        Ok(ThoTensor::new(self.id.clone(), mm_r)?)
    }

    pub fn sum(&self, k: Kind) -> Result<ThoTensor<D, Id>, ThoosaError> {
        ThoTensor::new(self.id.clone(), self.t.sum(k))
    }

    pub fn sum_dim_intlist(&self, dim: i64) -> Result<ThoTensor<D, Id>, ThoosaError> {
        Ok(ThoTensor::new(
            self.id.clone(),
            panic_to_err!(self.t.f_sum_dim_intlist(dim, false, self.t.kind()))?,
        )?)
    }

    pub fn hadamard<RD: Dims>(
        &self,
        rhs: &ThoTensor<RD, Id>,
        to_id: Id,
    ) -> Result<ThoTensor<D, Id>, ThoosaError> {
        Ok(ThoTensor::new(to_id, self.t.f_multiply(&rhs.t)?)?)
    }

    pub fn mean(&self, k: Kind) -> Result<ThoTensor<D, Id>, ThoosaError> {
        ThoTensor::new(self.id.clone(), self.t.mean(k))
    }

    pub fn id(&self) -> &Id {
        &self.id
    }

    // #[cfg(test)]
    pub fn t_ref(&self) -> &Tensor {
        &self.t
    }

    pub fn t(self) -> Tensor {
        self.t
    }


    pub fn run_inner_with_params<
        F,
        OutDim: Dims,
        P: Dims,
    >(
        &self,
        id: Id,
        func: &F,
        params: &[&ThoTensor<P, Id>],
    ) -> Result<ThoTensor<OutDim, Id>, ThoosaError>
    where
        F: Fn(&Tensor, &[&Tensor]) -> Result<Tensor, ThoosaError> + ?Sized,
    {
        Ok(ThoTensor::new(
            id,
            func(
                &self.t,
                params
                    .into_iter()
                    .map(|p| &p.t)
                    .collect::<Vec<&Tensor>>()
                    .as_slice(),
            )?,
        )?)
    }

    // No-params variant: transform tensor with a function that only needs &Tensor
    pub fn run_inner<R, F>(&self, func: F) -> Result<R, ThoosaError>
    where
        F: Fn(&Tensor) -> Result<R, ThoosaError>,
    {
        func(&self.t)
    }

    pub fn _track_used_in(&self, token: &'static str) {
        self.tracking.write().push(token);
    }

    pub fn to_mut<'a>(&'a mut self) -> ThoTensorMut2<'a, D, Id> {
        ThoTensorMut2 {
            _id: &self.id,
            t: &mut self.t,
            pd: Default::default(),
        }
    }

    pub fn double_value(&self, idx: &[i64]) -> f64 {
        self.t.double_value(idx)
    }

    pub fn device(&self) -> usize {
        ThoDevice::from(self.t.device()).ordinal()
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
    Id: Named + Debug + Clone, //+ Send + Sync + Eq + std::hash::Hash + std::fmt::Debug //+ 'static
{
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoosaError> {
        debug!("CopyToDevice for ThoTensor: self.id={:?} self.t.size={:?} self.t.device={:?} dst_device={dst_device}", self.id, self.t.size(), self.t.device());
        //should not copy device to itself
        assert!(!ThoDevice::from(dst_device).eq(&ThoDevice::from(self.t.device())));

        let copy =
            if tensor_mem_size(self.t.kind(), &self.t.size()) < CopyCtx::instance().buf_size() {
                self.t.f_to_device(ThoDevice::from(dst_device).deref().clone())?
            } else {
                assert!(self.t.is_contiguous());
                CopyCtx::instance()
                    .copy_from_ptr(
                        self.t.data_ptr() as usize,
                        &self.t.size(),
                        dst_device,
                        self.t.kind(),
                    ).await?
            };

        trace!("CopyToDevice for ThoTensor: self.t.size={:?} self.t.device={:?} dst_device={dst_device} self.t=\n{} copy=\n{copy}", self.t.size(), self.t.device(), self.t);

        Ok(ThoTensor::new(self.id.clone(), copy)?)
    }
}

impl<D: Dims, Id> Output for ThoTensor<D, Id> where
    Id: Named + Debug + Clone + std::fmt::Debug + Send + Sync + Eq + std::hash::Hash + 'static
{
}


pub struct ThoTensorMut2<'a, D: Dims, Id: Named + Debug + Clone> {
    _id: &'a Id,
    t: &'a mut Tensor,
    pd: PhantomData<D>,
}

unsafe impl<'a, D: Dims, Id: Named + Debug + Clone> Send for ThoTensorMut2<'a, D, Id> {}

impl<'a, D: Dims, Id: Named + Debug + Clone> ThoTensorMut2<'a, D, Id> {
    pub fn new(id: &'a Id, t: &'a mut Tensor) -> Self {
        Self {
            _id: id,
            t,
            pd: Default::default(),
        }
    }

    pub fn zero_(&mut self) {
        let _ = self.t.zero_();
    }
    pub fn add_(&mut self, rhs: &ThoTensor<D, Id>) -> Result<(), ThoosaError> {
        let _ = self.t.f_add_(&rhs.t)?;

        Ok(())
    }
    pub fn copy_from_(&mut self, rhs: &ThoTensor<D, Id>) {
        self.t.copy_(&rhs.t);
    }

    pub fn copy_partial_from_(
        &mut self,
        dim: i64,
        start: i64,
        length: i64,
        rhs: &ThoTensor<D, Id>,
    ) {
        self.t.narrow(dim, start, length).copy_(&rhs.t);
    }

    pub fn resize_(&mut self, new_shape: &[i64]) {
        let _ = self.t.resize_(new_shape);
    }
}



#[derive(Clone, Debug)]
pub struct Labels;

impl Dims for Labels {
    type Metrics = MonoMetrics<0, 1>;

}
