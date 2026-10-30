use thoosa_common::tch_error::tch_result;
use log::trace;
use std::ops::Range;
use tch::Tensor;
use thoosa_common::dims::{ChannelsData, LocalSelect};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::traits::MonoMetrics;
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
            tch_result(Tensor::f_empty([0], (t.kind(), t.device())))?
        } else {
            tch_result(Tensor::f_vstack(subs.as_slice()))?
        };

        let sub = ThoTensor::new(id + &mono_key!(0u32, format!("{ranges:?}")), stacked)?;

        trace!("ChannelsData2::select: Vec sub.t=\n{}", sub.t_ref());

        Ok(sub)
    }
}



