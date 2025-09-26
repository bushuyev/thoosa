use std::cmp::min;
use log::debug;
use tch::IndexOp;
use thoosa_common::dims::Dims;
use thoosa_shared::{MyError, ThoosaError};
use thoosa_shared::traits::{DataId, SimpleId};
use torch_runner::data::ThoTensor2;
use torch_runner::metrics::TwoDimsMetrics;

#[derive(Clone, Debug)]
pub struct ChannelsData2;

#[derive(Clone, Debug, Default)]
pub struct ChannelsDataMetrics {
    channel:i64,
    seq:i64
}

impl ChannelsDataMetrics {
    pub fn new(channel: i64, seq: i64) -> Self {
        Self { channel, seq }
    }
}

impl From<[i64; 2]> for ChannelsDataMetrics {
    fn from([channel, seq]: [i64; 2]) -> Self {
        Self { channel, seq }
    }
}

impl TryFrom<Vec<i64>> for ChannelsDataMetrics {
    type Error = ThoosaError;

    fn try_from(value: Vec<i64>) -> Result<Self, Self::Error> {
        let &[channel, seq, ..] = value.as_slice() else {
            return Err(MyError::SomethingElse(format!(
                "can't make LoadedTensorMetrics from {value:?}"
            ))
                .into());
        };

        Ok(Self { channel, seq })
    }
}

impl ChannelsDataMetrics {
    pub fn seq_len(&self) -> i64 {
        self.seq
    }

}


impl Dims for ChannelsData2 {
    type Metrics = ChannelsDataMetrics;
    type Sub = ThoTensor2<ChannelsData2, SimpleId>;
    type SubSelect = std::ops::Range<Self::Metrics>;

    fn subview(t: &Self::Sub, range: &Self::SubSelect) -> Self::Sub {
        let size: ChannelsDataMetrics = t.size().try_into().unwrap();

        let channel_range = range.start.channel..min(size.channel, range.end.channel);
        let data_range = range.start.seq..min(size.seq, range.end.seq);

        debug!(
                "Data for LoadedTensor::subview: channel_range={channel_range:?} data_range={data_range:?} size={size:?}"
            );
        debug!("Data for LoadedTensor::subview: t=\n{}", t);

        let sub = ThoTensor2::new(
            SimpleId(format!("{}:{:?}:{:?}", t.id(), channel_range, data_range).leak()),//TODO display
            t.t().i((channel_range, data_range))
        );

        debug!("Data for LoadedTensor::subview: sub.t=\n{}",t.t());

        sub
    }
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