use std::cmp::min;
use std::ops::Range;
use log::{debug, trace};
use tch::{IndexOp, Tensor};
use thoosa_common::dims::{AnyMetrics, Batch, Dims};
use thoosa_shared::{key_elt, ThoosaError};
use thoosa_shared::traits::{Adjustable, DataId, Select};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::traits::MonoMetrics;
use thoosa_shared::comp_id::CompKey;

#[derive(Clone, Debug)]
pub struct ChannelsData2;


impl Dims for ChannelsData2 {
    type Metrics = MonoMetrics<1, 2>;//channels x seq

}

#[derive(Clone, Debug, Default)]
pub struct ChannelsDataVec {
    pub ranges:Vec<Range<MonoMetrics<1, 2>>>
}

impl Select<ChannelsDataVec> for ThoTensor<ChannelsData2, CompKey> {
    type Output = ThoTensor<ChannelsData2, CompKey>;

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

        let ranges_key = CompKey::new([key_elt!(format!("{ranges:?}"), "ranges")]);
        let sub = ThoTensor::new(self.id() + &ranges_key, stacked)?;

        trace!("ChannelsData2::select: Vec sub.t=\n{}", sub.t_ref());

        Ok(sub)
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChannelsData2Selector(pub Range<MonoMetrics<1, 2>>);

impl Select<ChannelsData2Selector> for ThoTensor<ChannelsData2, CompKey> {
    type Output = ThoTensor<ChannelsData2, CompKey>;

    fn subview(&self, selector: ChannelsData2Selector) -> Result<Self::Output, ThoosaError> {
        let size: MonoMetrics<1, 2> = self.size().try_into()?;

        let mut ranges = MonoMetrics::<1, 2>::map_to_ranges(&selector.0.start, &selector.0.end)?;
        debug!("ChannelsData2Selector::subview 1. ranges={ranges:?}, size={:?}", self.size());

        // if ranges[1].start > 48 {
        //     ranges[1].start -=48;
        // }
        // if ranges[1].end < self.size()[1]-100 {
        //     ranges[1].end +=100;
        // }
        // debug!("ChannelsData2Selector::subview 2. ranges={ranges:?}");

        let sub_tensor = self.t_ref().f_i((ranges[0].clone(), ranges[1].clone()))?;

        Ok(ThoTensor::new(self.id().clone(), sub_tensor)?)
    }
}

impl Adjustable for ChannelsData2Selector {}
impl From<Range<MonoMetrics<1, 2>>> for ChannelsData2Selector {
    fn from(value: Range<MonoMetrics<1, 2>>) -> Self {
        ChannelsData2Selector(value)
    }
}

#[derive(Debug, Clone, Default)]
pub struct BatchChannelsData2Selector(pub Range<MonoMetrics<0, 3>>);

impl Select<BatchChannelsData2Selector> for ThoTensor<Batch<ChannelsData2>, CompKey> {
    type Output = ThoTensor<Batch<ChannelsData2>, CompKey>;

    fn subview(&self, selector: BatchChannelsData2Selector) -> Result<Self::Output, ThoosaError> {
        todo!()
    }
}


impl ChannelsData2 {
    fn sub_single(t: &ThoTensor<ChannelsData2, CompKey>, range: &Range<MonoMetrics<1, 2>>, size: &MonoMetrics<1, 2>) -> Tensor {
        let channel_range = range.start.dim_0()..min(size.dim_0(), range.end.dim_0());
        let data_range = range.start.dim_1()..min(size.dim_1(), range.end.dim_1());

        debug!("ChannelsData2::subview: channel_range={channel_range:?} data_range={data_range:?} size={size:?}");
        trace!("ChannelsData2::subview: t=\n{}", t.t_ref());

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
