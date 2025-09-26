use thoosa_common::dims::Dims;
use thoosa_shared::traits::{DataId, SimpleId};
use torch_runner::data::ThoTensor2;
use torch_runner::metrics::TwoDimsMetrics;

#[derive(Clone, Debug)]
pub struct ChannelsData2;

impl Dims for ChannelsData2 {
    type Metrics = TwoDimsMetrics;
    type Sub = ThoTensor2<ChannelsData2, SimpleId>;
    type SubSelect = std::ops::Range<Self::Metrics>;
}

#[derive(Clone, Debug)]
pub struct ChannelsSlicesData2;

impl Dims for ChannelsSlicesData2 {
    type Metrics = [i64; 3];
    type Sub = ThoTensor2<ChannelsSlicesData2, SimpleId>;
    type SubSelect = std::ops::Range<Self::Metrics>;
}

#[derive(Clone, Debug)]
pub struct MotifSlices {

}

impl Dims for MotifSlices {
    type Metrics = [i64; 2];//slices x window

    type Sub = ThoTensor2<MotifSlices, DataId>;


    type SubSelect = std::ops::Range<Self::Metrics>;
}