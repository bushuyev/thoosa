use crate::tho_tensor::{CopyToDevice, ThoTensor};
use async_trait::async_trait;
use std::fmt::{Debug, Display};
use std::pin::Pin;
use thoosa_shared::applicables_runner::CachableId;

pub trait Dims: Send + Sync + Clone + Debug {
    type Metrics: Debug + Clone + Send + Sync + Default + TryFrom<Vec<i64>>;

}

#[derive(Clone, Debug)]
pub struct ChannelsData;
impl Dims for ChannelsData {
    type Metrics = AnyMetrics;
}

#[derive(Clone, Debug)]
pub struct ChannelsDataFiltered;
impl Dims for ChannelsDataFiltered {
    type Metrics = AnyMetrics;
}

#[derive(Clone, Debug)]
pub struct ChannelsSlicesData;
impl Dims for ChannelsSlicesData {
    type Metrics = AnyMetrics;
}

#[derive(Clone, Debug)]
pub struct SlicesData; //sliced data [slices, slice_width]
impl Dims for SlicesData {
    type Metrics = AnyMetrics;
}

#[derive(Clone, Debug)]
pub struct MotifData; //motif net applied to sliced data [slices, slice_width] - distances between motif and a slice
impl Dims for MotifData {
    type Metrics = AnyMetrics;
}

#[derive(Clone, Debug)]
pub struct ChannelsMotifsData;
impl Dims for ChannelsMotifsData {
    type Metrics = AnyMetrics;
}
impl LabelNetInputTensorDims for ChannelsMotifsData {}

#[derive(Clone, Debug)]
pub struct ChannelsMotifsFlattenedData;
impl Dims for ChannelsMotifsFlattenedData {
    type Metrics = AnyMetrics;
}
impl LabelNetInputTensorDims for ChannelsMotifsFlattenedData {}

pub trait LabelNetInputTensorDims: Dims {}

#[derive(Clone, Debug)]
pub struct LabelProbability;
impl Dims for LabelProbability {
    type Metrics = AnyMetrics;
} //[1] probability of an input belonging to certain labeled class

#[derive(Clone, Debug)]
pub struct LabelNetsProbabilities;
impl Dims for LabelNetsProbabilities {
    type Metrics = AnyMetrics;
} //[num of label nets x 1] probabilities of an input belonging to corresponding label produces by all label nets

#[derive(Clone, Debug)]
pub struct LabelsProbabilities;
impl Dims for LabelsProbabilities {
    type Metrics = AnyMetrics;
} //[num of labels, 1] probabilities of an input belonging to target labeled classes

#[derive(Clone, Debug)]
pub struct Final;
impl Dims for Final {
    type Metrics = AnyMetrics;
} //[num of labels, 1] probabilities of an input belonging to target labeled classes

#[derive(Clone, Debug)]
pub enum MotifsNetsSelectionDims {
    ChannelsMotifsData(ChannelsMotifsData),
    ChannelsMotifsFlattenData(ChannelsMotifsFlattenedData),
}
impl Dims for MotifsNetsSelectionDims {
    type Metrics = AnyMetrics;
}

#[derive(Clone, Debug)]
pub struct Batch<D: Dims>(D);
impl<D: Dims> Dims for Batch<D> {
    type Metrics = AnyMetrics;
}

pub trait Dimensional {
    type Dimensions: Dims;
}

pub type TCD = ThoTensor<ChannelsData, CachableId>;
pub type TCDF = ThoTensor<ChannelsDataFiltered, CachableId>;
pub type TCSD = ThoTensor<ChannelsSlicesData, CachableId>;
pub type ThoBatch<D> = ThoTensor<Batch<D>, CachableId>;

#[derive(Clone, Debug)]
pub struct AnyDims {}
impl Dims for AnyDims {
    type Metrics = AnyMetrics;
}

pub struct AnySub {}

#[async_trait]
impl CopyToDevice for AnySub {
    async fn copy_to_device(&self, dst_device: usize) -> Self {
        todo!()
    }
}

#[derive(Clone, Debug, Default)]
pub struct AnyMetrics;

unsafe impl Sync for AnyMetrics {}
unsafe impl Send for AnyMetrics {}
impl From<Vec<i64>> for AnyMetrics {
    fn from(value: Vec<i64>) -> Self {
        AnyMetrics
    }
}
