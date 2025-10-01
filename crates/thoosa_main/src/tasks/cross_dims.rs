use std::cmp::min;
use std::ops::Range;
use log::{debug, trace};
use tch::{IndexOp, Tensor};
use caching_runner::common::Select;
use thoosa_common::dims::Dims;
use thoosa_shared::{MyError, ThoosaError};
use thoosa_shared::traits::{DataId, SimpleId};
use torch_runner::data::ThoTensor2;
use torch_runner::metrics::TwoDimsMetrics;

#[derive(Clone, Debug)]
pub struct ChannelsData2;


impl Dims for ChannelsData2 {
    type Metrics = ChannelsDataMetrics;

}

#[derive(Clone, Debug, Default)]
pub struct ChannelsDataVec {
    pub ranges:Vec<Range<ChannelsDataMetrics>>
}

impl Select<ChannelsDataVec> for ThoTensor2<ChannelsData2, SimpleId> {
    type Output = ThoTensor2<ChannelsData2, SimpleId>;

    fn select(&self, selector: ChannelsDataVec) -> Self::Output {
        let ChannelsDataVec { ranges } = selector;
        let size: ChannelsDataMetrics = self
            .size()
            .try_into()
            .expect("ChannelsData2::select unable to convert size");

        let mut subs = Vec::with_capacity(ranges.len());
        for range in &ranges {
            subs.push(ChannelsData2::sub_single(self, range, &size));
        }

        let stacked = Tensor::vstack(subs.as_slice());
        let sub = ThoTensor2::new(
            SimpleId::from(format!("{}:{ranges:?}", self.id().0)),
            stacked,
        );

        trace!("ChannelsData2::select: Vec sub.t=\n{}", sub.t_ref());

        sub
    }
}


impl ChannelsData2 {
    fn sub_single(t: &ThoTensor2<ChannelsData2, SimpleId>, range: &Range<ChannelsDataMetrics>, size: &ChannelsDataMetrics) -> Tensor {
        let channel_range = range.start.channel..min(size.channel, range.end.channel);
        let data_range = range.start.seq..min(size.seq, range.end.seq);

        debug!("ChannelsData2::subview: channel_range={channel_range:?} data_range={data_range:?} size={size:?}");
        trace!("ChannelsData2::subview: t=\n{t}");

        let sub_t = t.t_ref().i((channel_range.clone(), data_range.clone()));
        sub_t
    }
}


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
pub enum ChannelsDataSubOutput {
    Single(ThoTensor2<ChannelsData2, SimpleId>),
    Vec(ThoTensor2<MotifSlices, SimpleId>)
}


#[derive(Clone, Debug)]
pub struct ChannelsSlicesData2;

impl Dims for ChannelsSlicesData2 {
    type Metrics = [i64; 3];
}

#[derive(Clone, Debug)]
pub struct MotifSlices {

}

impl Dims for MotifSlices {
    type Metrics = [i64; 2];//slices x window

}
