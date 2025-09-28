use std::cmp::min;
use std::ops::Range;
use log::{debug, trace};
use tch::{IndexOp, Tensor};
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

#[derive(Debug)]
pub enum ChannelsDataSubSelect{
    Single(std::ops::Range<ChannelsDataMetrics>),
    Vec(Vec<std::ops::Range<ChannelsDataMetrics>>)
}

impl Dims for ChannelsData2 {
    type Metrics = ChannelsDataMetrics;
    type Sub = ThoTensor2<ChannelsData2, SimpleId>;
    type SubSelect = ChannelsDataSubSelect;

    fn subview(t: &Self::Sub, sub_select: &Self::SubSelect) -> Self::Sub {
        match sub_select {
            ChannelsDataSubSelect::Single(range) => {
                let size: ChannelsDataMetrics = t.size().try_into().unwrap();

                let sub_t = Self::sub_single(t, range, &size);

                let sub = ThoTensor2::new(
                    SimpleId(format!("{}:{:?}", t.id(), range).leak()),//TODO display
                    sub_t
                );

                trace!("ChannelsData2::subview: Single sub.t=\n{}",sub.t());

                sub
            }
            ChannelsDataSubSelect::Vec(ranges) => {
                let size: ChannelsDataMetrics = t.size().try_into().unwrap();

                let subs = Tensor::vstack(
                    ranges.into_iter().map(|range| {
                        Self::sub_single(t, range, &size)
                    }).collect::<Vec<_>>().as_slice(),
                );

                let sub = ThoTensor2::new(
                    SimpleId(format!("{}:{:?}", t.id(), ranges).leak()),//TODO display
                    subs
                );

                trace!("ChannelsData2::subview: Vec sub.t=\n{}",sub.t());

                sub
            }
        }
    }
}

impl ChannelsData2 {
    fn sub_single(t: &ThoTensor2<ChannelsData2, SimpleId>, range: &Range<ChannelsDataMetrics>, size: &ChannelsDataMetrics) -> Tensor {
        let channel_range = range.start.channel..min(size.channel, range.end.channel);
        let data_range = range.start.seq..min(size.seq, range.end.seq);

        debug!("ChannelsData2::subview: channel_range={channel_range:?} data_range={data_range:?} size={size:?}");
        trace!("ChannelsData2::subview: t=\n{t}");

        let sub_t = t.t().i((channel_range.clone(), data_range.clone()));
        sub_t
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