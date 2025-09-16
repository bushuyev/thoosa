use std::fmt::{Debug, Display};
use thoosa_shared::applicables_runner::CachableId;
use crate::tho_tensor::ThoTensor;

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


pub type TCD = ThoTensor<ChannelsData, CachableId>;
pub type TCDF = ThoTensor<ChannelsDataFiltered, CachableId>;
pub type TCSD = ThoTensor<ChannelsSlicesData, CachableId>;
pub type ThoBatch<D> = ThoTensor<Batch<D>, CachableId>;


#[derive(Clone, Debug)]
pub struct AnyDims {}
impl Dims for AnyDims {}
