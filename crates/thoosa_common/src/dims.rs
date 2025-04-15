use std::fmt::{Debug, Display, Formatter};
use std::marker::PhantomData;
use tch::{Device, Tensor};


pub trait Dims:Send + Clone + Debug {}

#[derive(Clone, Debug)]
pub struct ChannelsData;
impl Dims for ChannelsData {}

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
    pub tensor: Tensor, // attach the actual tensor
    pub _pd: PhantomData<D>,
}

impl <D:Dims> Dimensional for ThoTensor<D> {
    type Dimensions = D;
}

impl <D:Dims> Clone for ThoTensor<D> {
    fn clone(&self) -> Self {
        ThoTensor {
            tensor: self.tensor.shallow_clone(),
            _pd: PhantomData,
        }
    }
}



impl <D: Dims> From<Tensor> for ThoTensor<D> {
    fn from(tensor: Tensor) -> Self {
        Self {
            tensor,
            _pd: PhantomData,
        }
    }
}

impl <D: Dims> From<&Tensor> for ThoTensor<D> {
    fn from(tensor: &Tensor) -> Self {
        Self {
            tensor: tensor.shallow_clone(),
            _pd: PhantomData,
        }
    }
}



pub type TCD = ThoTensor<ChannelsData>;
pub type TCSD = ThoTensor<ChannelsSlicesData>;

impl <D:Dims> AsRef<Tensor> for ThoTensor<D>{
    fn as_ref(&self) -> &Tensor{
        self.tensor.as_ref()
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
            tensor: self.tensor.shallow_clone(),
            _pd: Default::default(),
        }
    }

    pub fn drop(self) {
        drop(self.tensor);
    }

    pub fn to_device(&self, device: Device) -> Self {
        Self{
            tensor:self.tensor.to_device(device),
            _pd: self._pd
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