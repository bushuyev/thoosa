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
    type Metrics = TwoDimsMetrics<1>;//channels x seq

}

#[derive(Clone, Debug, Default)]
pub struct ChannelsDataVec {
    pub ranges:Vec<Range<TwoDimsMetrics<1>>>
}

impl Select<ChannelsDataVec> for ThoTensor2<ChannelsData2, SimpleId> {
    type Output = ThoTensor2<ChannelsData2, SimpleId>;

    fn subview(&self, selector: ChannelsDataVec) -> Self::Output {
        let ChannelsDataVec { ranges } = selector;
        let size: TwoDimsMetrics<1> = self
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
    fn sub_single(t: &ThoTensor2<ChannelsData2, SimpleId>, range: &Range<TwoDimsMetrics<1>>, size: &TwoDimsMetrics<1>) -> Tensor {
        let channel_range = range.start.dim_0()..min(size.dim_0(), range.end.dim_0());
        let data_range = range.start.dim_1()..min(size.dim_1(), range.end.dim_1());

        debug!("ChannelsData2::subview: channel_range={channel_range:?} data_range={data_range:?} size={size:?}");
        trace!("ChannelsData2::subview: t=\n{t}");

        let sub_t = t.t_ref().i((channel_range.clone(), data_range.clone()));
        sub_t
    }
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
    type Metrics = TwoDimsMetrics<0>; //[i64;2]; // [i64; 2];//slices x window

}
