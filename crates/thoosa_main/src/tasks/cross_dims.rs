use std::cmp::min;
use std::ops::Range;
use log::{debug, trace};
use tch::{IndexOp, Tensor};
use tch::kind::FLOAT_CPU;
use thoosa_common::dims::{AnyMetrics, Batch, ChannelsData, Dims};
use thoosa_shared::{mono_key, ThoError};
use thoosa_shared::traits::{Adjustable, DataId, Select};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::traits::MonoMetrics;
use thoosa_shared::comp_id::SynKey;


#[derive(Clone, Debug, Default)]
pub struct ChannelsDataVec {
    pub ranges:Vec<Range<MonoMetrics<1, 2>>>
}

impl Select<ChannelsDataVec> for ThoTensor<ChannelsData, SynKey> {
    type Output = ThoTensor<ChannelsData, SynKey>;

    //TODO switch to index_select if lots of ranges
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="Select<ChannelsDataVec> for ThoTensor<ChannelsData2, SynKey>::subview"))]
    fn subview(&self, selector: ChannelsDataVec) -> Result<Self::Output, ThoError> {
        let ChannelsDataVec { ranges } = selector;
        let size: MonoMetrics<1, 2> = self.size().try_into()?;

        let mut subs = Vec::with_capacity(ranges.len());
        for range in &ranges {
            subs.push(ChannelsData::sub_single(self, range, &size)?);
        }

        let stacked = if subs.is_empty() {
            //TODO the size then should be checked to decide if tensor is valid. Should be a better solution
            Tensor::f_empty([0], (self.t_ref().kind(), self.t_ref().device()))?
        } else {
            Tensor::f_vstack(subs.as_slice())?
        };

        let sub = ThoTensor::new(self.id() + &mono_key!(0, format!("{ranges:?}")), stacked)?;

        trace!("ChannelsData2::select: Vec sub.t=\n{}", sub.t_ref());

        Ok(sub)
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChannelsData2Selector(pub Range<MonoMetrics<1, 2>>);

impl Select<ChannelsData2Selector> for ThoTensor<ChannelsData, SynKey> {
    type Output = ThoTensor<ChannelsData, SynKey>;

    fn subview(&self, selector: ChannelsData2Selector) -> Result<Self::Output, ThoError> {
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
