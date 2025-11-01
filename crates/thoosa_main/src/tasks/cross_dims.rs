use std::cmp::min;
use std::ops::Range;
use log::{debug, trace};
use tch::{IndexOp, Tensor};
use thoosa_common::dims::{AnyMetrics, Dims};
use thoosa_shared::{MyError, ThoosaError};
use thoosa_shared::traits::{DataId, Select, SimpleId};
use torch_runner::data::ThoTensor2;
use thoosa_shared::traits::MonoMetrics;

#[derive(Clone, Debug)]
pub struct ChannelsData2;


impl Dims for ChannelsData2 {
    type Metrics = MonoMetrics<1, 2>;//channels x seq

}

#[derive(Clone, Debug, Default)]
pub struct ChannelsDataVec {
    pub ranges:Vec<Range<MonoMetrics<1, 2>>>
}

impl Select<ChannelsDataVec> for ThoTensor2<ChannelsData2, SimpleId> {
    type Output = ThoTensor2<ChannelsData2, SimpleId>;

    fn subview(&self, selector: ChannelsDataVec) -> Result<Self::Output, ThoosaError> {
        let ChannelsDataVec { ranges } = selector;
        let size: MonoMetrics<1, 2> = self
            .size()
            .try_into()
            .expect("ChannelsData2::select unable to convert size");



        let mut subs = Vec::with_capacity(ranges.len());
        for range in &ranges {
            subs.push(ChannelsData2::sub_single(self, range, &size));
        }

        //TODO the size then should be checked to decide if tensor is valid. Should be a better solution
        let stacked = if subs.is_empty() {
            Tensor::empty(&[0], (self.t_ref().kind(), self.t_ref().device()))
        } else {
            Tensor::vstack(subs.as_slice())
        };

        let sub = ThoTensor2::new(
            SimpleId::from(format!("{}:{ranges:?}", self.id().0)),
            stacked,
        );

        trace!("ChannelsData2::select: Vec sub.t=\n{}", sub.t_ref());

        Ok(sub)
    }
}


impl ChannelsData2 {
    fn sub_single(t: &ThoTensor2<ChannelsData2, SimpleId>, range: &Range<MonoMetrics<1, 2>>, size: &MonoMetrics<1, 2>) -> Tensor {
        let channel_range = range.start.dim_0()..min(size.dim_0(), range.end.dim_0());
        let data_range = range.start.dim_1()..min(size.dim_1(), range.end.dim_1());

        debug!("ChannelsData2::subview: channel_range={channel_range:?} data_range={data_range:?} size={size:?}");
        trace!("ChannelsData2::subview: t=\n{t}");

        let sub_t = t.t_ref().i((channel_range.clone(), data_range.clone()));

        // assert_eq!(sub_t.size(), [])
        sub_t
    }
}




#[derive(Clone, Debug)]
pub struct MotifSlices;

impl Dims for MotifSlices {
    type Metrics = MonoMetrics<0, 2>; //[i64;2]; // [i64; 2];//slices x window - used to split motif's slices when calculating distances

}


#[derive(Clone, Debug)]
pub struct ChannelsMotifsSlices;

impl Dims for ChannelsMotifsSlices {
    type Metrics = MonoMetrics<1, 3>; //[i64;3]; // [i64; 3];//channels x slices x window

}


#[derive(Clone, Debug)]
pub struct ChannelsMotifsDists;

impl Dims for ChannelsMotifsDists {
    type Metrics = MonoMetrics<1, 2>; //[i64;3]; // [i64; 3];//channels x dists

}
