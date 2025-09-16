use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;
use tch::{Device, Tensor};
use async_trait::async_trait;
use ambassador::delegatable_trait;
use mockall::automock;
use thoosa_shared::applicables_runner::{CachableId, Input, Output};
use thoosa_shared::traits::{Named, WithId};
use crate::applicables::ThoDevice;
use crate::copy_ctx::CopyCtx;
use crate::cuda_utils::tensor_mem_size;
use crate::dims::{Dimensional, Dims};

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

unsafe impl <D:Dims, Id: Named + Clone> Send for ThoTensor<D, Id> {}

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


    pub fn size(&self) -> Vec<i64> {
        self.t.size()
    }

    pub fn device(&self)->Device{
        self.t.device()
    }

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