use std::fmt::{Debug, Display, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use tch::{Device, Tensor};
use thoosa_shared::applicables_runner::{CachableId, Input, Output};
use crate::applicables::ThoDevice;

pub trait Dims:Send +Sync+ Clone + Debug {}

#[derive(Clone, Debug)]
pub struct ChannelsData;
impl Dims for ChannelsData {}


#[derive(Clone, Debug)]
pub struct ChannelsDataFiltered;
impl Dims for ChannelsDataFiltered {}

#[derive(Clone, Debug)]
pub struct ChannelsSlicesData;
impl Dims for ChannelsSlicesData {}

#[derive(Clone, Debug)]
pub struct SlicesData;//sliced data [slices, slice_width]
impl Dims for SlicesData {}

#[derive(Clone, Debug)]
pub struct MotifData;//motif net applied to sliced data [slices, slice_width] - distances between motif and a slice
impl Dims for MotifData {}

#[derive(Clone, Debug)]
pub struct ChannelsMotifsData;
impl Dims for ChannelsMotifsData{}
impl LabelNetInputTensorDims for ChannelsMotifsData{}

#[derive(Clone, Debug)]
pub struct ChannelsMotifsFlattenedData;
impl Dims for ChannelsMotifsFlattenedData {}
impl LabelNetInputTensorDims for ChannelsMotifsFlattenedData {}

pub trait LabelNetInputTensorDims:Dims{}

#[derive(Clone, Debug)]
pub struct LabelProbability;
impl Dims for LabelProbability {}//[1] probability of an input belonging to certain labeled class

#[derive(Clone, Debug)]
pub struct LabelNetsProbabilities;
impl Dims for LabelNetsProbabilities {}//[num of label nets x 1] probabilities of an input belonging to corresponding label produces by all label nets

#[derive(Clone, Debug)]
pub struct LabelsProbabilities;
impl Dims for LabelsProbabilities {}//[num of labels, 1] probabilities of an input belonging to target labeled classes

#[derive(Clone, Debug)]
pub struct Final;
impl Dims for Final {}//[num of labels, 1] probabilities of an input belonging to target labeled classes

#[derive(Clone, Debug)]
pub enum MotifsNetsSelectionDims {
    ChannelsMotifsData(ChannelsMotifsData),
    ChannelsMotifsFlattenData(ChannelsMotifsFlattenedData)
}
impl Dims for MotifsNetsSelectionDims {}

#[derive(Clone, Debug)]
pub struct Batch<D:Dims>(D);
impl <D:Dims> Dims for Batch<D> {}


pub trait Dimensional {
    type Dimensions:Dims;
}


#[derive(Debug)]
pub struct ThoTensor<D: Dims> {
    id: CachableId,
    pub tensor: Tensor, // attach the actual tensor
    pub _pd: PhantomData<D>,
}

impl <D:Dims> Dimensional for ThoTensor<D> {
    type Dimensions = D;
}

impl <D:Dims> Clone for ThoTensor<D> {
    fn clone(&self) -> Self {
        ThoTensor {
            id: self.id.clone(),
            tensor: self.tensor.shallow_clone(),
            _pd: PhantomData,
        }
    }
}

unsafe impl <D:Dims> Sync for ThoTensor<D> {}
unsafe impl <D:Dims> Send for ThoTensor<D> {}


impl <D: Dims> From<Tensor> for ThoTensor<D> {
    fn from(tensor: Tensor) -> Self {
        Self {
            id: "1".to_string(),
            tensor,
            _pd: PhantomData,
        }
    }
}

impl <D: Dims> From<&Tensor> for ThoTensor<D> {
    fn from(tensor: &Tensor) -> Self {
        Self {
            id: "2".to_string(),
            tensor: tensor.shallow_clone(),
            _pd: PhantomData,
        }
    }
}

#[async_trait]
impl <D:Dims> Output for ThoTensor<D> {

    async fn copy_to_device(self: Arc<Self>, device_ordinal: usize) -> Arc<Self> {
        if self.tensor.device().eq(&ThoDevice::from(device_ordinal).into()) {
            Arc::new(ThoTensor::new(
                self.id.clone(),
                self.tensor.shallow_clone()
            ))
        } else {
            Arc::new(ThoTensor::new(
                self.id.clone(),
                self.tensor.to_device(ThoDevice::from(device_ordinal).into())
            ))
        }

    }
}

impl <D:Dims> Input for ThoTensor<D> {
    fn id(&self) -> &CachableId {
        &self.id
    }
}


impl <D:Dims> Input for &ThoTensor<D> {
    fn id(&self) -> &CachableId {
        &self.id
    }
}

pub type TCD = ThoTensor<ChannelsData>;
pub type TCDF = ThoTensor<ChannelsDataFiltered>;
pub type TCSD = ThoTensor<ChannelsSlicesData>;

impl <D:Dims> AsRef<Tensor> for ThoTensor<D>{
    fn as_ref(&self) -> &Tensor{
        &self.tensor
    }
}

impl <D:Dims> ThoTensor<D> {


    pub fn size(&self) -> Vec<i64> {
        self.tensor.size()
    }

    pub fn device(&self)->Device{
        self.tensor.device()
    }

    pub fn shallow_clone<DO:Dims>(&self) -> ThoTensor<DO> {
        ThoTensor {
            id: self.id.clone(),
            tensor: self.tensor.shallow_clone(),
            _pd: Default::default(),
        }
    }

    pub fn drop(self) {
        drop(self.tensor);
    }

    pub fn to_device(&self, device: Device) -> Self {
        Self{
            id: self.id.clone(),
            tensor:self.tensor.to_device(device),
            _pd: self._pd
        }
    }

    pub fn new(id: CachableId, tensor: Tensor) -> Self {
        Self { 
            id, tensor, _pd: PhantomData 
        }
    }
}

impl <D:Dims> Display for ThoTensor<D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.tensor, f)
    }
}


#[derive(Clone, Debug)]
pub struct AnyDims {}
impl Dims for AnyDims {}