use std::cmp::min;
use crate::tho_tensor::ThoTensor;
use async_trait::async_trait;
use std::fmt::Debug;
use std::ops::Range;
use tch::{IndexOp, Tensor};
use log::{debug, trace};
use thoosa_shared::{CopyToDevice, ThoError};
use thoosa_shared::traits::{Metrics, MonoMetrics, SimpleId};
use thoosa_named_derive::Named;
use thoosa_shared::comp_id::SynKey;

pub trait Dims: Send + Sync + Clone + Debug {
    type Metrics: Metrics + Debug + Clone + Send + Sync + Default + TryFrom<Vec<i64>>;
}


#[derive(Clone, Debug)]
pub struct ChannelsData;

impl Dims for ChannelsData {
    type Metrics = MonoMetrics<1, 2>;//channels x seq

}
impl ChannelsData {

    pub fn sub_single(t: &ThoTensor<ChannelsData, SynKey>, range: &Range<MonoMetrics<1, 2>>, size: &MonoMetrics<1, 2>) -> Result<Tensor, ThoError> {
        let channel_range = range.start.dim_0()..min(size.dim_0(), range.end.dim_0());
        let data_range = range.start.dim_1()..min(size.dim_1(), range.end.dim_1());

        debug!("ChannelsData2::subview: channel_range={channel_range:?} data_range={data_range:?} size={size:?}");
        trace!("ChannelsData2::subview: t=\n{}", t.t_ref());

        let sub_t = t.t_ref().f_i((channel_range.clone(), data_range.clone()))?;

        // assert_eq!(sub_t.size(), [])
        Ok(sub_t)
    }
}

#[derive(Clone, Debug)]
pub struct Slice; //sliced data [lice_width]
impl Dims for Slice {
    type Metrics = MonoMetrics<0, 1>;
}

#[derive(Clone, Debug)]
pub struct MotifProbability; //motif net applied to sliced data [slices, slice_width] - probabilities of a slice belonging to a motif
impl Dims for MotifProbability {
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
pub struct LabelsProbabilities;//TODO rename - name is too close to LabelProbability
impl Dims for LabelsProbabilities {
    type Metrics = AnyMetrics;
} //[num of labels, 1] probabilities of an input belonging to target labeled classes

#[derive(Clone, Debug)]
pub struct Labels;

impl Dims for Labels {
    type Metrics = MonoMetrics<0, 1>;

}

#[derive(Clone, Debug)]
pub struct Final;
impl Dims for Final {
    type Metrics = AnyMetrics;
} //[num of labels, 1] probabilities of an input belonging to target labeled classes

#[derive(Clone, Debug)]
pub struct Batch<D: Dims>(D);
impl<D: Dims> Dims for Batch<D> {
    type Metrics = MonoMetrics<0, 3>;//O
}


pub type ThoBatch<D> = ThoTensor<Batch<D>, SimpleId>;

#[derive(Clone, Debug, Named)]
#[name = "n/a"]
pub struct AnyDims {}
impl Dims for AnyDims {
    type Metrics = AnyMetrics;
}

#[derive(Clone, Debug, Default)]
pub struct AnyMetrics;

impl Metrics for AnyMetrics {}

unsafe impl Sync for AnyMetrics {}
unsafe impl Send for AnyMetrics {}
impl From<Vec<i64>> for AnyMetrics {
    fn from(_value: Vec<i64>) -> Self {
        AnyMetrics
    }
}

#[derive(Clone, Debug)]
pub struct MotifSlices;

impl Dims for MotifSlices {
    type Metrics = MonoMetrics<0, 2>; // [i64; 2];//slices x window - used to split motif's slices when calculating distances

}



#[derive(Clone, Debug)]
pub struct Spectorgram;

impl Dims for Spectorgram {
    type Metrics = MonoMetrics<0, 2>; // [i64; 2];//time x frequencies -

}
