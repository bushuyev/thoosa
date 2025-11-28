use crate::common::runnable::Stage;
use crate::oom_handler::{Adjuster, InputContext};
use ambassador::delegatable_trait;
use async_trait::async_trait;
use log::debug;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, DivAssign, Range};
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Adjustable, Data, Metrics, MonoMetrics};

///A - Selector (range, etc), M - its metrics
#[derive(Debug)]
pub struct SplitsAdjuster<A, M>
where
    A:Adjustable + From<Range<M>> + Debug,
    M: Metrics + Debug + Clone + Default + Send + Sync,
    M: DivAssign<i64>,//TODO replace i64 with num_traits type
    M: PartialOrd<M>,
    for<'a, 'b> &'a M: Add<&'b M, Output =M>
{
    pd: PhantomData<A>,
    min_split_len: M,
}

impl<A, M> SplitsAdjuster<A, M>
where
    A:Adjustable + From<Range<M>> + Debug,
    M: Metrics + Debug + Clone + Default + Send + Sync,
    M: DivAssign<i64>,//TODO replace i64 with num_traits type
    M: PartialOrd<M>,
    for<'a, 'b> &'a M: Add<&'b M, Output =M>
{
    pub fn new(min_split_len: M) -> Self {
        debug!("SplitsAdjuster::new: min_split_len={min_split_len:?}");

        Self {
            pd: Default::default(),
            min_split_len,
        }
    }
}

#[async_trait]
impl <AC,  A, M> Adjuster<AC> for SplitsAdjuster<A, M>
where
    AC:SplitsAdjusterContext<Metrics =M>,
    A:Adjustable + From<Range<M>> + Debug,
    M: Metrics + Debug + Clone + Default + Send + Sync,
    M: DivAssign<i64>,//TODO replace i64 with num_traits type
    M: PartialOrd<M>,
    for<'a, 'b> &'a M: Add<&'b M, Output =M>
{
    type A = A;

    async fn adjust(
        &self,
        adjustable: &A,
        adjust_context: &mut AC,
    ) -> Result<A, ThoosaError> {

        let split_start = adjust_context.done_to();

        let mut split_end = split_start + adjust_context.split_len();
        adjust_context.input_len().cap(&mut split_end);

        debug!("SplitsAdjuster::adjust: split_start={split_start:?} split_end={split_end:?} adjustable.size={:?}", adjust_context.input_len());


        Ok((split_start.clone()..split_end).into())
        // todo!()
    }

    fn degrade(&self, adjust_context: &mut AC) -> bool {
        debug!("SplitsAdjuster::degrade: 0. adjust_context={adjust_context:?}");

        (*adjust_context.split_len_mut()) /= 2; //TODO configure degrade step


        let can_degrade = adjust_context.split_len() > &self.min_split_len;

        debug!("SplitsAdjuster::degrade: 1. adjust_context={adjust_context:?} self.min_split_len={:?} can_degrade={can_degrade}", self.min_split_len);

        can_degrade
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

impl<'a, 'b, R: Stage + Adjustable + 'a> InputContext<'a, R> for SplitsAdjusterContextImpl<'b, R>
where
    <R as Stage>::In<'b>: Data,
    <<R as Stage>::In<'b> as Data>::Metrics: for<'x> AddAssign<&'x <<R as Stage>::In<'b> as Data>::Metrics>,
    <<R as Stage>::In<'b> as Data>::Metrics: PartialOrd,
{
    fn add_partial_output(&mut self, o: R::Out){
        self.output_builder.add_partial_output(o);

        self.done_to += &self.split_len.clone();

        debug!("SplitsAdjusterAcontextImpl::add_partial_output: self.done_to={:?}  self.split_len={:?}",self.done_to, self.split_len);
    }

    fn all_done(&self) -> bool {
        self.done_to >= self.input_len
    }

    // fn input_len(&self) -> &<<R as Stage>::In<'a> as Data>::Metrics {
    //     &self.input_len
    // }

    async fn build_final_output(self) -> Result<R::Out, ThoosaError> {
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
    fn build(self: Box<Self>) -> Result<Self::O, ThoosaError>;
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
