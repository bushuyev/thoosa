use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;
use tch::{Device, Kind, Tensor};
use async_trait::async_trait;
use ambassador::delegatable_trait;
use mockall::automock;
use torch_sys::IntList;
use thoosa_shared::applicables_runner::{CachableId, Input, Output};
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Named, WithId};
use crate::applicables::ThoDevice;
use crate::copy_ctx::CopyCtx;
use crate::cuda_utils::tensor_mem_size;
use crate::dims::{Dimensional, Dims};
use crate::panic_to_err;

#[derive(Debug)]
pub struct ThoTensor<D: Dims, Id: Named + Clone> {
    id: Id,//TODO move to wrappers (RawChannels etc)?
    pub t: Tensor, // attach the actual tensor
    pub _pd: PhantomData<D>,

    #[allow(dead_code)]
    tracking:parking_lot::RwLock<Vec<&'static str>>,
}

impl <D:Dims, Id: Named + Clone> Dimensional for ThoTensor<D, Id> {
    type Dimensions = D;
}

unsafe impl <D:Dims, Id: Named + Clone> Sync for ThoTensor<D, Id> {}


impl <D: Dims> From<Tensor> for ThoTensor<D, CachableId> {
    fn from(tensor: Tensor) -> Self {
        Self {
            id: "1".into(),
            t: tensor,
            _pd: PhantomData,
            tracking: Default::default(),
        }
    }
}

impl <D: Dims> From<&Tensor> for ThoTensor<D, CachableId> {
    fn from(tensor: &Tensor) -> Self {
        Self {
            id: "2".into(),
            t: tensor.shallow_clone(),
            _pd: PhantomData,
            tracking: Default::default(),
        }
    }
}

#[async_trait]
impl <D:Dims> Output for ThoTensor<D, CachableId> {

    async fn copy_to_device(self: Arc<Self>, device_ordinal: usize) -> Arc<Self> {
        if self.t.device().eq(&ThoDevice::from(device_ordinal).into()) {
            Arc::new(ThoTensor::new(
                self.id.clone(),
                self.t.shallow_clone()
            ))
        } else {
            Arc::new(ThoTensor::new(
                self.id.clone(),
                self.t.to_device(ThoDevice::from(device_ordinal).into())
            ))
        }

    }
}

impl <D:Dims> WithId for ThoTensor<D, CachableId> {
    type ID = CachableId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl <D:Dims> Input for ThoTensor<D, CachableId> {
    fn id(&self) -> &CachableId {
        &self.id
    }
}

impl <D:Dims> WithId for &ThoTensor<D, CachableId> {
    type ID = CachableId;

    fn id(&self) -> &Self::ID {
        todo!()
    }
}

impl <D:Dims> Input for &ThoTensor<D, CachableId> {
    fn id(&self) -> &CachableId {
        &self.id
    }
}

impl <D:Dims> AsRef<Tensor> for ThoTensor<D, CachableId>{
    fn as_ref(&self) -> &Tensor{
        &self.t
    }
}

impl <D:Dims> ThoTensor<D, CachableId> {


    //TODO remove
    pub fn shallow_clone<DO:Dims>(&self) -> ThoTensor<DO, CachableId> {
        ThoTensor {
            id: self.id.clone(),
            t: self.t.shallow_clone(),
            _pd: Default::default(),
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

    pub fn mm<RD:Dims>(&self, rhs: &ThoTensor<RD, Id>) -> Result<ThoTensor<D, Id>, ThoosaError> {
        let mm_r = panic_to_err!(self.t.f_matmul(&rhs.t))?;
        println!("mm: self.t.size={:?}, rhs.t.size={:?} mm_r.size={:?}", self.t.size(), rhs.t.size(), mm_r.size());

        Ok(ThoTensor::new(
            self.id.clone(),
            mm_r
        ))
    }

    pub fn sum(&self, k: Kind) -> ThoTensor<D, Id> {
        ThoTensor::new(self.id.clone(),self.t.sum(k))
    }


    pub fn sum_dim_intlist(&self, dim:i64) -> Result<ThoTensor<D, Id>, ThoosaError> {
        Ok(ThoTensor::new(
            self.id.clone(),
            panic_to_err!(self.t.f_sum_dim_intlist(dim, false, self.t.kind()))?
        ))
    }


    pub fn hadamard<RD:Dims>(&self, rhs: &ThoTensor<RD, Id>, to_id: Id) -> Result<ThoTensor<D, Id>, ThoosaError> {
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
    ) -> Result<ThoTensor<OutDim, Id>, ThoosaError>
    where F:Fn(&Tensor, &[&Tensor])->Result<Tensor, ThoosaError> + ?Sized{


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
    ) -> Result<ThoTensor<OutDim, Id>, ThoosaError>
    where
        F: Fn(&Tensor) -> Result<Tensor, ThoosaError>,
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
        ThoDevice::from(self.t.device()).ordinal()
    }
}

impl <D:Dims> Display for ThoTensor<D, CachableId> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.t, f)
    }
}

#[async_trait]
impl <D:Dims, Id> CopyToDevice for ThoTensor<D, Id>
where Id: Named + Clone //+ Send + Sync + Eq + std::hash::Hash + std::fmt::Debug //+ 'static
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

#[async_trait]
#[delegatable_trait]
#[automock]
pub trait CopyToDevice {
    async fn copy_to_device(&self, dst_device: usize) ->Self;
}



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
    pub fn add_(&mut self, rhs: &ThoTensor<D, Id>) -> Result<(), ThoosaError> {
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
