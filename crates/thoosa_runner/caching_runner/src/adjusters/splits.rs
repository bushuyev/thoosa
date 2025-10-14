use crate::common::Data;
use crate::common::runnable::Stage;
use crate::oom_handler::{Adjustable, Adjuster, InputContext};
use ambassador::delegatable_trait;
use async_trait::async_trait;
use log::debug;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, DivAssign};
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Metrics, MonoMetrics};

#[derive(Debug)]
pub struct SplitsAdjuster<A, RA>
where
    A: Data + Adjustable,
    <A as Data>::Metrics: DivAssign<i64>,
{
    pd: PhantomData<(A, RA)>,
    min_split_len: <A as Data>::Metrics,
}

impl<A, RA> SplitsAdjuster<A, RA>
where
    A: Data + Adjustable,
    <A as Data>::Metrics: DivAssign<i64>,
    <A as Data>::Metrics: PartialOrd<<A as Data>::Metrics>,
{
    pub fn new(min_split_len: <A as Data>::Metrics) -> Self {
        debug!("SplitsAdjuster::new: min_split_len={min_split_len:?}");

        Self {
            pd: Default::default(),
            min_split_len,
        }
    }
}

#[async_trait]
impl <AC, AJ, RA:Debug> Adjuster<AC> for SplitsAdjuster<AJ, RA>
where
    AC:SplitsAdjusterContext<Metrics = <AJ as Data>::Metrics>,
    AJ: Data + Adjustable,
    <AJ as Data>::Metrics: DivAssign<i64>,
    <AJ as Data>::Metrics: PartialOrd<<AJ as Data>::Metrics>,
    for<'a, 'b> &'a <AJ as Data>::Metrics: Add<&'b <AJ as Data>::Metrics, Output = <AJ as Data>::Metrics>,
    RA:Adjustable + From<std::ops::Range<AJ::Metrics>>
{
    type A = RA;

    async fn adjust(
        &self,
        adjustable: &Self::A,
        adjust_context: &mut AC,
    ) -> Result<Self::A, ThoosaError> {
        let split_start = adjust_context.done_to();
        let mut split_end = (split_start + adjust_context.split_len());
        adjust_context.input_len().cap(&mut split_end);

        debug!("SplitsAdjuster::adjust: split_start={split_start:?} split_end={split_end:?} adjustable.size={:?}", adjust_context.input_len());

        // Ok((size_0.clone()..size_1).into())

        // Ok(MonoMetrics::<0,2>::map_to_ranges(&split_start, &split_end)?.into())
        Ok((split_start.clone()..split_end).into())
    }

    fn degrade(&self, adjust_context: &mut AC) -> bool {
        debug!("SplitsAdjuster::degrade: 0. adjust_context={adjust_context:?}");

        (*adjust_context.split_len_mut()) /= 2;
        debug!("SplitsAdjuster::degrade: 1. adjust_context={adjust_context:?}");

        adjust_context.split_len() > &self.min_split_len
    }
}

#[delegatable_trait]
pub trait SplitsAdjusterContext: Debug + Send + Sync {
    type Metrics: Debug + Send + Sync;

    fn input_len(&self) -> &Self::Metrics;
    fn split_len(&self) -> &Self::Metrics;
    fn done_to(&self) -> &Self::Metrics;
    fn split_len_mut(&mut self) -> &mut Self::Metrics;
}

#[derive(Debug)]
pub struct SplitsAdjusterContextImpl<'a, R>
where
    R: Stage + Adjustable + 'static,
    <R as Stage>::In<'a>: Data,
{
    done_to: <<R as Stage>::In<'a> as Data>::Metrics,
    split_len: <<R as Stage>::In<'a> as Data>::Metrics,
    input_len: <<R as Stage>::In<'a> as Data>::Metrics,

    output_builder: Box<dyn OutputBuilder<O = <R as Stage>::Out>>,
}

impl<'a, R: Stage + Adjustable + 'a> InputContext<'a, R> for SplitsAdjusterContextImpl<'a, R>
where
    <R as Stage>::In<'a>: Data,
    <<R as Stage>::In<'a> as Data>::Metrics: for<'x> AddAssign<&'x <<R as Stage>::In<'a> as Data>::Metrics>,
    <<R as Stage>::In<'a> as Data>::Metrics: PartialOrd,
{
    fn add_partial_output(&mut self, o: R::Out){
        self.output_builder.add_partial_output(o);

        self.done_to += &self.split_len.clone();

        debug!("SplitsAdjusterAcontextImpl::add_partial_output: self.done_to={:?}  self.split_len={:?}",self.done_to, self.split_len);
    }

    fn all_done(&self) -> bool {
        self.done_to >= self.input_len
    }

    fn input_len(&self) -> &<<R as Stage>::In<'a> as Data>::Metrics {
        &self.input_len
    }

    fn build_final_output(self) -> R::Out {
        self.output_builder.build()
    }
}

impl<'a, R: Stage + Adjustable + 'a> SplitsAdjusterContextImpl<'a, R>
where
    <R as Stage>::In<'a>: Data,
{
    pub fn new(
        input_len: <<R as Stage>::In<'a> as Data>::Metrics,
        output_builder: Box<dyn OutputBuilder<O = <R as Stage>::Out>>,
    ) -> Self {
        Self {
            done_to: input_len.default_end(),
            split_len: input_len.clone(),
            input_len,
            output_builder,
        }
    }
}

impl<'a, R: Stage + Adjustable + 'a> SplitsAdjusterContext for SplitsAdjusterContextImpl<'a, R>
where
    R: Stage + Adjustable,
    <R as Stage>::In<'a>: Data,
{
    type Metrics = <<R as Stage>::In<'a> as Data>::Metrics;

    fn split_len(&self) -> &Self::Metrics {
        &self.split_len
    }

    fn input_len(&self) -> &Self::Metrics {
        &self.input_len
    }


    fn done_to(&self) -> &Self::Metrics {
        &self.done_to
    }

    fn split_len_mut(&mut self) -> &mut Self::Metrics {
        &mut self.split_len
    }
}

pub trait OutputBuilder: Debug + Send + Sync {
    type O;
    fn add_partial_output(&mut self, o: Self::O);
    fn build(self: Box<Self>) -> Self::O;
}

#[cfg(test)]
mod tests {
    use mockall::mock;

    mock! {
        TestInputId {}
    }
    mock! {
        pub TestInput{
        }

    }

    #[test]
    fn test_0() {
        //TODO
    }
}
