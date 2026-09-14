use thoosa_common::dims::{AnyMetrics, Dims};
use thoosa_shared::traits::MonoMetrics;

// These markers describe tensors produced only by the motif-search and model-training workflow.
#[derive(Clone, Debug)]
pub struct MotifProbability;
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
}

#[derive(Clone, Debug)]
pub struct LabelNetsProbabilities;
impl Dims for LabelNetsProbabilities {
    type Metrics = AnyMetrics;
}

#[derive(Clone, Debug)]
pub struct LabelsProbabilities;
impl Dims for LabelsProbabilities {
    type Metrics = AnyMetrics;
}

#[derive(Clone, Debug)]
pub struct Labels;
impl Dims for Labels {
    type Metrics = MonoMetrics<0, 1>;
}

#[derive(Clone, Debug)]
pub struct Final;
impl Dims for Final {
    type Metrics = AnyMetrics;
}

#[derive(Clone, Debug)]
pub struct MotifSlices;
impl Dims for MotifSlices {
    type Metrics = MonoMetrics<0, 2>;
}

#[derive(Clone, Debug)]
pub struct Spectorgram;
impl Dims for Spectorgram {
    type Metrics = MonoMetrics<0, 2>;
}

#[derive(Clone, Debug)]
pub struct ChannelsMotifsSlices;
impl Dims for ChannelsMotifsSlices {
    type Metrics = MonoMetrics<1, 3>;
}

#[derive(Clone, Debug)]
pub struct ChannelsMotifsDists;
impl Dims for ChannelsMotifsDists {
    type Metrics = MonoMetrics<1, 2>;
}
