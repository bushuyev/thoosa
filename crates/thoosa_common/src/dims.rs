use crate::tho_tensor::{KeyHolder, ThoTensor, ThoTensorInner};
use async_trait::async_trait;
use log::{debug, trace};
use std::cmp::min;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Deref, Range};
use std::sync::Arc;
use tch::{IndexOp, Tensor};
use thoosa_named_derive::Named;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::traits::{Metrics, MonoMetrics, Named, RangeSelector, Select, SimpleId, ThoSelector};
use thoosa_shared::{CopyToDevice, ThoError};

pub trait Dims: Send + Sync + Clone + Debug {
    type Metrics: Metrics + Debug + Clone + Send + Sync + Default + TryFrom<Vec<i64>>;
}

impl <D, Id, S> Select<S> for ThoTensor<D, Id>
where
    S:ThoSelector + 'static,
    D:Dims,
    Id:Named + Debug + Clone,
    Self:LocalSelect<Id, S, Output = ThoTensor<D, Id>>,
{
    type Output = ThoTensor<D, Id>;

    fn subview(&self, selector: S) -> Result<Self::Output, ThoError> {
        match &self.inner {
            ThoTensorInner::Val { t } => {
                Self::local_subview(self.id(), t, selector)
            }
            ThoTensorInner::Ref { key, size, kind, .. } => {//TODO compose with 'parent' selector ?
                debug!("ThoTensor::subview: key={key:?} size={size:?} kind={kind:?}");
                Ok(Self {
                    id:self.id().clone(),
                    inner: ThoTensorInner::Ref {
                        key: key.clone(),
                        size: size.clone(),
                        kind: kind.clone(),
                        selector:Some(Box::new(selector) as Box<dyn ThoSelector>),
                    },
                    tracking: Default::default(),
                    pd: PhantomData,
                }
                )
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Batch<D: Dims>(D);
impl<D: Dims> Dims for Batch<D> {
    type Metrics = MonoMetrics<0, 3>;//O
}


pub type ThoBatch<D> = ThoTensor<Batch<D>, SimpleId>;

#[derive(Clone, Debug, Named)]
#[name = "n/a"]
pub struct AnyDims {}
impl Dims for AnyDims {
    type Metrics = AnyMetrics;
}

#[derive(Clone, Debug, Default)]
pub struct AnyMetrics;

impl Metrics for AnyMetrics {}

unsafe impl Sync for AnyMetrics {}
unsafe impl Send for AnyMetrics {}
impl From<Vec<i64>> for AnyMetrics {
    fn from(_value: Vec<i64>) -> Self {
        AnyMetrics
    }
}

pub trait LocalSelect<Id, S> {
    type Output;

    fn local_subview(id:&Id, t:&Tensor, selector: S) -> Result<Self::Output, ThoError>;
}

impl LocalSelect<u32, RangeSelector> for ThoTensor<AnyDims, u32> {
    type Output = ThoTensor<AnyDims, u32>;

    fn local_subview(id: &u32, t: &Tensor, selector: RangeSelector) -> Result<Self::Output, ThoError> {
        let start = i64::try_from(selector.start[0]).map_err(|_| ThoError::new("range selector start does not fit i64"))?;
        let end = i64::try_from(selector.end[0]).map_err(|_| ThoError::new("range selector end does not fit i64"))?;
        ThoTensor::new(*id, t.f_i(start..end)?)
    }
}

#[derive(Clone, Debug)]
pub struct ChannelsData;

impl Dims for ChannelsData {
    type Metrics = MonoMetrics<1, 2>;//channels x seq

}
impl ChannelsData {

    pub fn sub_single(t: &Tensor, range: &Range<MonoMetrics<1, 2>>, size: &MonoMetrics<1, 2>) -> Result<Tensor, ThoError> {
        let channel_range = range.start.dim_0()..min(size.dim_0(), range.end.dim_0());
        let data_range = range.start.dim_1()..min(size.dim_1(), range.end.dim_1());

        debug!("ChannelsData2::subview: channel_range={channel_range:?} data_range={data_range:?} size={size:?}");
        trace!("ChannelsData2::subview: t=\n{}", t);

        let sub_t = t.f_i((channel_range.clone(), data_range.clone()))?;

        // assert_eq!(sub_t.size(), [])
        Ok(sub_t)
    }
}

#[derive(Clone, Debug)]
pub struct Slice; //sliced data [lice_width]
impl Dims for Slice {
    type Metrics = MonoMetrics<0, 1>;
}
