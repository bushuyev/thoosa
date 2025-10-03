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
        _adjustable: &Self::A,
        adjust_context: &mut AC,
    ) -> Result<Self::A, ThoosaError> {
        let size_0 = adjust_context.done_to();
        let size_1 = size_0 + adjust_context.split_len(); //TODO cap on adjustable.size()

        debug!("SplitsAdjuster::adjust: size_0={size_0:?} size_1={size_1:?}");

        Ok((size_0.clone()..size_1).into())

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

    fn split_len(&self) -> &Self::Metrics;
    fn done_to(&self) -> &Self::Metrics;
    fn split_len_mut(&mut self) -> &mut Self::Metrics;
}

#[derive(Debug)]
pub struct SplitsAdjusterContextImpl<R>
where
    R: Stage + Adjustable + 'static,
    <R as Stage>::In<'static>: Data,
{
    done_to: <<R as Stage>::In<'static> as Data>::Metrics,
    split_len: <<R as Stage>::In<'static> as Data>::Metrics,
    input_len: <<R as Stage>::In<'static> as Data>::Metrics,

    output_builder: Box<dyn OutputBuilder<O = <R as Stage>::Out>>,
}

impl<R: Stage + Adjustable + 'static> InputContext<R> for SplitsAdjusterContextImpl<R>
where
    <R as Stage>::In<'static>: Data,
    <<R as Stage>::In<'static> as Data>::Metrics: for<'a> AddAssign<&'a <<R as Stage>::In<'static> as Data>::Metrics>,
    <<R as Stage>::In<'static> as Data>::Metrics: PartialOrd,
{
    fn add_partial_output(&mut self, o: R::Out) {
        self.output_builder.add_partial_output(o);

        self.done_to += &self.split_len;

        debug!(
            "SplitsAdjusterAcontextImpl::add_partial_output: self.done_to={:?}  self.split_len={:?}",
            self.done_to, self.split_len
        );
    }

    fn all_done(&self) -> bool {
        self.done_to >= self.input_len
    }

    fn input_len(&self) -> &<<R as Stage>::In<'static> as Data>::Metrics {
        &self.input_len
    }

    fn build_final_output(self) -> R::Out {
        self.output_builder.build()
    }
}

impl<R: Stage + Adjustable + 'static> SplitsAdjusterContextImpl<R>
where
    <R as Stage>::In<'static>: Data,
{
    pub fn new(
        input_len: <<R as Stage>::In<'static> as Data>::Metrics,
        output_builder: Box<dyn OutputBuilder<O = <R as Stage>::Out>>,
    ) -> Self {
        Self {
            done_to: Default::default(),
            split_len: input_len.clone(),
            input_len,
            output_builder,
        }
    }
}

impl<R: Stage + Adjustable + 'static> SplitsAdjusterContext for SplitsAdjusterContextImpl<R>
where
    R: Stage + Adjustable,
    <R as Stage>::In<'static>: Data,
{
    type Metrics = <<R as Stage>::In<'static> as Data>::Metrics;

    fn split_len(&self) -> &Self::Metrics {
        &self.split_len
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
        // use super::super::{
        //     RunnableContext, Adjustable, Adjuster, Adjusters, Contexts, InputContext,
        //     OOMHandler, RunCtxContext,
        // };
        // let x:RunnableContext = RunnableContext{};
        // let ctx = SplitsAdjusterAcontextImpl::new(100, "test");
    }
}
