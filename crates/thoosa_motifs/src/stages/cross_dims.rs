use crate::stages::transforms::freq_filter::ChannelsDataSelector;
use log::{debug, trace};
use std::cmp::min;
use std::ops::Range;
use tch::kind::FLOAT_CPU;
use tch::{IndexOp, Tensor};
use thoosa_common::dims::{AnyMetrics, Batch, ChannelsData, Dims, LocalSelect};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::traits::MonoMetrics;
use thoosa_shared::traits::{Adjustable, DataId, Select, ThoSelector};
use thoosa_shared::{ThoError, mono_key};

#[derive(Clone, Debug, Default)]
pub struct ChannelsDataVec {
    pub ranges:Vec<Range<MonoMetrics<1, 2>>>
}

impl LocalSelect<SynKey, ChannelsDataVec> for ThoTensor<ChannelsData, SynKey> {
    type Output = ThoTensor<ChannelsData, SynKey>;

    //TODO switch to index_select if lots of ranges
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="Select<ChannelsDataVec> for ThoTensor<ChannelsData2, SynKey>::subview"))]
    fn local_subview(id:&SynKey, t:&Tensor, selector: ChannelsDataVec) -> Result<Self::Output, ThoError> {
        let ChannelsDataVec { ranges } = selector;
        let size: MonoMetrics<1, 2> = t.size().try_into()?;

        let mut subs = Vec::with_capacity(ranges.len());
        for range in &ranges {
            subs.push(ChannelsData::sub_single(t, range, &size)?);
        }

        let stacked = if subs.is_empty() {
            //TODO the size then should be checked to decide if tensor is valid. Should be a better solution
            Tensor::f_empty([0], (t.kind(), t.device()))?
        } else {
            Tensor::f_vstack(subs.as_slice())?
        };

        let sub = ThoTensor::new(id + &mono_key!(0, format!("{ranges:?}")), stacked)?;

        trace!("ChannelsData2::select: Vec sub.t=\n{}", sub.t_ref());

        Ok(sub)
    }
}

impl LocalSelect<SynKey, ChannelsDataSelector> for ThoTensor<ChannelsData, SynKey> {
    type Output = ThoTensor<ChannelsData, SynKey>;

    fn local_subview(id:&SynKey, t:&Tensor, selector: ChannelsDataSelector) -> Result<Self::Output, ThoError> {
        let size: MonoMetrics<1, 2> = t.size().try_into()?;

        let mut ranges = MonoMetrics::<1, 2>::map_to_ranges(&selector.0.start, &selector.0.end)?;
        debug!("ChannelsData2Selector::subview 1. ranges={ranges:?}, size={:?}", t.size());

        // if ranges[1].start > 48 {
        //     ranges[1].start -=48;
        // }
        // if ranges[1].end < self.size()[1]-100 {
        //     ranges[1].end +=100;
        // }
        // debug!("ChannelsData2Selector::subview 2. ranges={ranges:?}");

        let sub_tensor = t.f_i((ranges[0].clone(), ranges[1].clone()))?;

        Ok(ThoTensor::new(id.clone(), sub_tensor)?)
    }
}


