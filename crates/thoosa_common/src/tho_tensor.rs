use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use tch::{Device, Tensor};
use async_trait::async_trait;
use thoosa_shared::applicables_runner::{CachableId, Input, Output};
use thoosa_shared::traits::{Named, WithId};
use crate::applicables::ThoDevice;
use crate::dims::{Dimensional, Dims};

#[derive(Debug)]
pub struct ThoTensor<D: Dims, Id: Named + Clone> {
    id: Id,//TODO move to wrappers (RawChannels etc)?
    pub t: Tensor, // attach the actual tensor
    pub _pd: PhantomData<D>,
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
        }
    }
}

impl <D: Dims> From<&Tensor> for ThoTensor<D, CachableId> {
    fn from(tensor: &Tensor) -> Self {
        Self {
            id: "2".into(),
            t: tensor.shallow_clone(),
            _pd: PhantomData,
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

    pub fn shallow_clone<DO:Dims>(&self) -> ThoTensor<DO, CachableId> {
        ThoTensor {
            id: self.id.clone(),
            t: self.t.shallow_clone(),
            _pd: Default::default(),
        }
    }

    pub fn drop(self) {
        drop(self.t);
    }

    pub fn to_device(&self, device: Device) -> Self {
        Self{
            id: self.id.clone(),
            t:self.t.to_device(device),
            _pd: self._pd
        }
    }

    pub fn new(id: CachableId, tensor: Tensor) -> Self {
        Self {
            id,
            t: tensor, _pd: PhantomData
        }
    }
}

impl <D:Dims> Display for ThoTensor<D, CachableId> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.t, f)
    }
}