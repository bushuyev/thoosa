use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::ops::Deref;
use caching_runner::common::{CopyToDevice, MyError};
use tch::{Device, Kind, Tensor};
use async_trait::async_trait;
use torch_sys::IntList;
use caching_runner::panic_to_err;
use thoosa_common::dims::Dims;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::traits::{CachableIdTr, Named, WithId};
use crate::{ThoDevice, ThoTensorView};
use crate::copy_ctx::CopyCtx;
use crate::cuda_utils::tensor_mem_size;

#[derive(Debug)]
pub struct ThoTensor<D: Dims, Id: Named + Clone> {
    id: Id,
    t: Tensor,
    //TODO move to a wrapper
    #[allow(dead_code)]
    tracking:parking_lot::RwLock<Vec<&'static str>>,
    _pd: PhantomData<D>,
}

impl <D:Dims, Id: Named + Clone> ThoTensor<D, Id> {
    pub fn new(id: Id, tensor: Tensor) -> Self {
        Self {
            id,
            t: if tensor.is_contiguous() {
                tensor
            } else {
                tensor.contiguous()
            },
            tracking: Default::default(),
            _pd: PhantomData::default(),
        }
    }

    pub fn zeros(id: Id, size: impl IntList, options: (Kind, Device)) -> Self {
        Self { id, t:Tensor::zeros(size, options), tracking: Default::default(), _pd: PhantomData::default() }
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

    // ---- Views / out-of-place ops (safe to share) ----
    pub fn narrow<'a>(&'a self, dim: i64, start: i64, len: i64) -> ThoTensorView<'a, D, Id> {
        ThoTensorView {
            t: self.t.narrow(dim, start, len),
            _owner: self,
        }
    }
    pub fn select<'a>(&'a self, dim: i64, index: i64) -> ThoTensorView<'a, D, Id> {
        ThoTensorView {
            t: self.t.select(dim, index),
            _owner: self,
        }
    }
    pub fn transpose<'a>(&'a self, d0: i64, d1: i64) -> ThoTensorView<'a, D, Id> {
        ThoTensorView {
            t: self.t.transpose(d0, d1),
            _owner: self,
        }
    }
    pub fn view<'a>(&'a self, shape: &[i64]) -> ThoTensorView<'a, D, Id> {
        ThoTensorView {
            t: self.t.view(shape),
            _owner: self,
        }
    }

    pub fn to_kind(&self, k: Kind) -> ThoTensor<D, Id> {
        ThoTensor {
            id: self.id.clone(),
            t: self.t.to_kind(k),
            tracking: Default::default(),
            _pd:PhantomData
        }
    }
    pub fn to_device(&self, d: Device) -> ThoTensor<D, Id> {
        ThoTensor::new(
            self.id.clone(),
             self.t.to_device(d),
        )
    }
    pub fn contiguous(&self) -> ThoTensor<D, Id> {
        ThoTensor::new(
            self.id.clone(),
            self.t.contiguous(),
        )
    }

    pub fn add(&self, rhs: &ThoTensor<D, Id>) -> ThoTensor<D, Id> {
        ThoTensor::new(
            self.id.clone(),
            &self.t + &rhs.t,
        )
    }

    pub fn mm<RD:Dims>(&self, rhs: &ThoTensor<RD, Id>) -> Result<ThoTensor<D, Id>, MyError> {
        let mm_r = panic_to_err!(self.t.f_matmul(&rhs.t)).map_err(|e| MyError::from(e))?;
        println!("mm: self.t.size={:?}, rhs.t.size={:?} mm_r.size={:?}", self.t.size(), rhs.t.size(), mm_r.size());

        Ok(ThoTensor::new(
            self.id.clone(),
            mm_r
        ))
    }

    pub fn sum(&self, k: Kind) -> ThoTensor<D, Id> {
        ThoTensor::new(self.id.clone(),self.t.sum(k))
    }


    pub fn sum_dim_intlist(&self, dim:i64) -> Result<ThoTensor<D, Id>, MyError> {
        Ok(ThoTensor::new(
            self.id.clone(),
            panic_to_err!(self.t.f_sum_dim_intlist(dim, false, self.t.kind()))?
        ))
    }


    pub fn hadamard<RD:Dims>(&self, rhs: &ThoTensor<RD, Id>, to_id: Id) -> Result<ThoTensor<D, Id>, MyError> {
        Ok(ThoTensor::new(
            to_id,
           self.t.f_multiply(&rhs.t)?
        ))
    }

    pub fn mean(&self, k: Kind) -> ThoTensor<D, Id> {
        ThoTensor::new(self.id.clone(),self.t.mean(k))
    }

    // explicit deep copy (no aliasing)
    pub fn deep_clone(&self) -> ThoTensor<D, Id> {
        ThoTensor::new(self.id.clone(), self.t.copy())

    }

    pub fn id(&self) -> &Id {
        &self.id
    }

    #[cfg(test)]
    pub fn t(&self) -> &Tensor {
        &self.t
    }

    pub fn run_inner_with_params<F, OutDim:Dims, P:Dims>(
        &self, id:Id,
        func:&F,
        params:&[&ThoTensor<P, Id>]
    ) -> Result<ThoTensor<OutDim, Id>, MyError>
    where F:Fn(&Tensor, &[&Tensor])->Result<Tensor, MyError> + ?Sized{


        Ok(ThoTensor::new(
            id,
            func(&self.t,  params.into_iter().map(|p|&p.t).collect::<Vec<&Tensor>>().as_slice())?
        ))
    }

    // No-params variant: transform tensor with a function that only needs &Tensor
    pub fn run_inner<OutDim: Dims, F>(
        &self,
        id: Id,
        func: F,
    ) -> Result<ThoTensor<OutDim, Id>, MyError>
    where
        F: Fn(&Tensor) -> Result<Tensor, MyError>,
    {
        Ok(ThoTensor::new(id, func(&self.t)?))
    }

    pub(crate) fn _track_used_in(&self, token: &'static str) {
        self.tracking.write().push(token);
    }

    pub fn to_mut<'a>(&'a mut self) -> ThoTensorMut<'a, D, Id> {
        ThoTensorMut {
            _id: &self.id,
            t: &mut self.t,
            pd: Default::default(),
        }
    }

    pub fn double_value(&self, idx: &[i64])->f64{
        self.t.double_value(idx)
    }

    pub fn device(&self)->usize {
        ThoDevice(self.t.device()).ordinal()
    }
}

impl<D: Dims, Id: Named + Clone> Display for ThoTensor<D, Id>{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.t.fmt(f)
    }
}

impl<D: Dims, Id> WithId for ThoTensor<D, Id>
where Id: CachableIdTr + Send + Sync + Eq + std::hash::Hash + Clone + std::fmt::Debug + Named
{
    type ID = Id;

    fn id(&self) -> &<Self as WithId>::ID {
        &self.id
    }
}

impl <D:Dims, Id> Input for ThoTensor<D, Id>
where Id: Named + Clone + std::fmt::Debug + Send + Sync + Eq + std::hash::Hash + 'static
{

}

#[async_trait]
impl <D:Dims, Id> CopyToDevice for ThoTensor<D, Id>
where Id: Named + Clone + Send + Sync + Eq + std::hash::Hash + std::fmt::Debug + 'static
{
    async fn copy_to_device(&self, dst_device: usize) ->Self {

        let copy = if tensor_mem_size(self.t.kind(), &self.t.size()) < CopyCtx::instance().buf_size() {
             self.t.to_device(ThoDevice::from(dst_device).deref().clone())
        }  else {
            assert!(self.t.is_contiguous());
            CopyCtx::instance().copy_from_ptr(self.t.data_ptr() as usize, &self.t.size(), dst_device, self.t.kind()).await
        };

        // println!("CopyToDevice for ThoTensor: self.t.size={:?} self.t.device={:?} dst_device={dst_device} self.t=\n{} copy=\n{copy}", self.t.size(), self.t.device(), self.t);

        ThoTensor::new(
            self.id.clone(),
            copy
        )
    }
}

impl <D:Dims, Id> Output for ThoTensor<D, Id>
where Id: Named + Clone + std::fmt::Debug + Send + Sync + Eq + std::hash::Hash + 'static
{

}

unsafe impl <D:Dims, Id: Named + Clone> Sync for ThoTensor<D, Id> {}


pub struct ThoTensorMut<'a, D:Dims, Id: Named + Clone>  {
    _id: &'a Id,
    t: &'a mut Tensor,
    pd:PhantomData<D>
}
unsafe impl<'a, D:Dims, Id: Named + Clone>  Send for ThoTensorMut<'a, D, Id> {}
// (no Sync)

impl<'a, D: Dims, Id: Named + Clone> ThoTensorMut<'a, D, Id> {
    pub fn new(id: &'a Id, t: &'a  mut Tensor) -> Self {
        Self {
            _id: id,
            t,
            pd: Default::default(),
        }
    }

    pub fn zero_(&mut self) {
        let _ = self.t.zero_();
    }
    pub fn add_(&mut self, rhs: &ThoTensor<D, Id>) -> Result<(), MyError> {
        let _ = self.t.f_add_(&rhs.t)?;

        Ok(())
    }
    pub fn copy_from_(&mut self, rhs: &ThoTensor<D, Id>) {
        self.t.copy_(&rhs.t);
    }

    pub fn copy_partial_from_(&mut self, _dim:i64, start: i64, length: i64, rhs: &ThoTensor<D, Id>) {
        self.t.narrow(0, start, length).copy_(&rhs.t);
    }


    pub fn resize_(&mut self, new_shape: &[i64]) {
        let _ = self.t.resize_(new_shape);
    }

}
