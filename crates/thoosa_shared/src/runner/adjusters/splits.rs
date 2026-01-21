
use thoosa_shared::runner::oom_handler::{Adjuster, InputContext};
use ambassador::delegatable_trait;
use async_trait::async_trait;
use log::debug;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{AddAssign, DivAssign, Range};
use thoosa_shared::ThoError;
use thoosa_shared::traits::{Adjustable, Data, Metrics};
use crate::runner::common::runnable::Stage;
use crate::runner::oom_handler::AdjusterError;

///A - Selector (range, etc), M - its metrics
#[derive(Debug)]
pub struct SplitsAdjuster<A, M>
where
    A:Adjustable + From<Range<M>> + Debug,
    M: Metrics + Debug + Clone + Default + Send + Sync,
    M: DivAssign<i64>,//TODO replace i64 with num_traits type
    M: PartialOrd<M>,
    M: AddAssign<M>,
{
    pd: PhantomData<A>,
    min_split_len: M,
    overlap_len: Option<M>, //for compensating lost slices when chunk needs to be sliced (unfolded)
}

impl<A, M> SplitsAdjuster<A, M>
where
    A:Adjustable + From<Range<M>> + Debug,
    M: Metrics + Debug + Clone + Default + Send + Sync,
    M: DivAssign<i64>,//TODO replace i64 with num_traits type
    M: PartialOrd<M>,
    M: AddAssign<M>,
{
    pub fn new(min_split_len: M) -> Self {
        debug!("SplitsAdjuster::new: min_split_len={min_split_len:?}");

        Self {
            pd: Default::default(),
            min_split_len,
            overlap_len:None
        }
    }

    pub fn new_with_overlap(min_split_len: M, overlap_len:M) -> Self {
        debug!("SplitsAdjuster::new_with_overlap: min_split_len={min_split_len:?}");

        Self {
            pd: Default::default(),
            min_split_len,
            overlap_len:Some(overlap_len)
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
    M: AddAssign<M>,
{
    type A = A;

    async fn adjust(
        &self,
        _adjustable: A,
        adjust_context: &mut AC,
    ) -> Result<A, AdjusterError<A>> {

        let split_start = adjust_context.done_to();

        let mut split_end = split_start.clone();
        split_end += adjust_context.split_len().clone();

        if let Some(overlap_len) = self.overlap_len.as_ref() {
            split_end += overlap_len.clone();
        }
        adjust_context.input_len().cap(&mut split_end);

        debug!("SplitsAdjuster::adjust: split_start={split_start:?} split_end={split_end:?} adjustable.size={:?}", adjust_context.input_len());


        Ok((split_start.clone()..split_end).into())
    }

    fn degrade(&self, adjust_context: &mut AC) -> bool {
        
        debug!("SplitsAdjuster::degrade: 0. adjust_context={adjust_context:?}");

        (*adjust_context.split_len_mut()) /= 2; //TODO configure degrade step

        let can_degrade = adjust_context.split_len() >= &self.min_split_len;

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
pub struct SplitsAdjusterContextImpl<'a, R, B>
where
    R: Stage + Adjustable + 'a,
    <R as Stage>::In<'a>: Data,
    B:OutputBuilder<O = <R as Stage>::Out>
{
    done_to: <<R as Stage>::In<'a> as Data>::Metrics,
    split_len: <<R as Stage>::In<'a> as Data>::Metrics,
    input_len: <<R as Stage>::In<'a> as Data>::Metrics,

    output_builder: B, //Box<dyn OutputBuilder<O = <R as Stage>::Out>>,
}

impl<'a, 'b, R, B> InputContext<'a, R> for SplitsAdjusterContextImpl<'b, R, B>
where
    R:Stage + Adjustable + 'a,
    B:OutputBuilder<O = <R as Stage>::Out>,
    <R as Stage>::In<'b>: Data,
    <<R as Stage>::In<'b> as Data>::Metrics: AddAssign<<<R as Stage>::In<'b> as Data>::Metrics>,
    <<R as Stage>::In<'b> as Data>::Metrics: PartialOrd,
{
    async fn add_partial_output(&mut self, o: R::Out){
        self.output_builder.add_partial_output(o).await;

        self.done_to += self.split_len.clone();

        debug!("SplitsAdjusterContextImpl::add_partial_output: self.done_to={:?}  self.split_len={:?}",self.done_to, self.split_len);
    }

    fn all_done(&self) -> bool {
        self.done_to >= self.input_len
    }


    async fn build_final_output(self) -> Result<R::Out, ThoError> {
        self.output_builder.build().await
    }
}

impl<'a, R, B> SplitsAdjusterContextImpl<'a, R, B>
where
    R:Stage + Adjustable + 'a,
    B:OutputBuilder<O = <R as Stage>::Out>,
    <R as Stage>::In<'a>: Data,
    <<R as Stage>::In<'a> as Data>::Metrics: AddAssign<<<R as Stage>::In<'a> as Data>::Metrics>,
    <<R as Stage>::In<'a> as Data>::Metrics: PartialOrd,
{
    pub fn new(
        input_len: <<R as Stage>::In<'a> as Data>::Metrics,
        output_builder: B,
    ) -> Self {
        Self {
            done_to: input_len.default_end(),
            split_len: input_len.clone(),
            input_len,
            output_builder,
        }
    }
}

impl<'a, R, B> SplitsAdjusterContext for SplitsAdjusterContextImpl<'a, R, B>
where
    R:Stage + Adjustable + 'a,
    B:OutputBuilder<O = <R as Stage>::Out>,
    <R as Stage>::In<'a>: Data,
    <<R as Stage>::In<'a> as Data>::Metrics: AddAssign<<<R as Stage>::In<'a> as Data>::Metrics>,
    <<R as Stage>::In<'a> as Data>::Metrics: PartialOrd,
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

#[allow(async_fn_in_trait)]
pub trait OutputBuilder: Debug + Send + Sync {
    type O;
    async fn add_partial_output(&mut self, o: Self::O);
    async fn build(self) -> Result<Self::O, ThoError>;
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::hash::{Hash, Hasher};
    use std::ops::{Deref, Range};
    use std::rc::Rc;
    use std::sync::Arc;
    use async_trait::async_trait;
    use log::debug;
    use mockall::{mock, Any};
    use mockall::predicate::always;
    use parking_lot::RwLock;
    use tokio::io::AsyncWriteExt;
    use thoosa_shared::ThoError;
    use crate::applicables_runner::{Input, Output};
    use crate::copy_mode::CpuOrGpu;
    use crate::CopyToDevice;
    use crate::runner::adjusters::splits::{OutputBuilder, SplitsAdjuster, SplitsAdjusterContextImpl};
    use crate::runner::common::MockRunCtx;
    use crate::runner::common::runnable::{Stage, SyncStage};
    use crate::runner::oom_handler::{Adjuster, InputContext};
    use crate::traits::{Adjustable, CachableIdTr, Data, WithId};
    use crate::traits::mock_cachable_id::MockCachableId;

    mock! {
        #[derive(Debug)]
        pub TestInput {
        }

        #[async_trait]
        impl CopyToDevice for TestInput {
            async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> where Self:Sized;
        }

        impl Data for TestInput {
              type Metrics = i64;

                type M = CpuOrGpu<MockTestInput>;

                fn size(&self) -> i64;
                fn device(&self) -> usize;

        }
        impl Input for TestInput {}
    }

    mock! {

        #[derive(Debug)]
        pub TestOutput{
        }

        #[async_trait]
        impl CopyToDevice for TestOutput {
            async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> where Self:Sized;
        }

        impl Data for TestOutput {
              type Metrics = i64;

                type M = CpuOrGpu<MockTestOutput>;

                fn size(&self) -> i64;
                fn device(&self) -> usize;

        }

        impl WithId for TestOutput {
            type ID =  MockCachableId;
            fn id(&self) -> & MockCachableId;
        }

        impl Output for TestOutput {
        }

    }

    mock! {
        #[derive(Clone, Debug)]
        pub TestTask {}

        impl Stage for TestTask {
            type In<'a> = MockTestInput;
            type Out = MockTestOutput;
            type Ctx = MockRunCtx;
        }

        impl SyncStage for TestTask {
            fn do_run<'f>(self: Arc<Self>, _input: &<Self as Stage>::In<'f>, _ctx: &<Self as Stage>::Ctx) -> Result<<Self as Stage>::Out, ThoError>;
        }

        impl Adjustable for TestTask {

        }

    }

    mock! {

        #[derive(Clone, Debug)]
        TestSelector {
        }

        impl Adjustable for TestSelector {
        }

        impl From<Range<i64>> for TestSelector {
            fn from(range: Range<i64>) -> Self {
                Self {
                    range,
                }
            }
        }
    }

    mock!{

        #[derive(Debug)]
        TestOutputBuilder {

        }

        impl OutputBuilder for TestOutputBuilder {
            type O = MockTestOutput;
            async fn add_partial_output(&mut self, o: MockTestOutput);
            async fn build(self) -> Result<MockTestOutput, ThoError>;
        }
    }

    pub struct MockTestOutputBuilderBuilder {
        mock:MockTestOutputBuilder
    }

    impl MockTestOutputBuilderBuilder {
        fn build() -> MockTestOutputBuilderBuilder {
            MockTestOutputBuilderBuilder {
                mock: MockTestOutputBuilder::new(),
            }
        }
        fn with_add_partial_output(mut self) -> Self {
            self.mock.expect_add_partial_output();

            self
        }

        fn finish(self)->MockTestOutputBuilder {
            self.mock
        }

        fn default()->MockTestOutputBuilder {
            MockTestOutputBuilderBuilder::build()
                .with_add_partial_output()
                .finish()
        }
    }

    pub struct MockTestSelectorBuilder {
        mock:MockTestSelector
    }



    impl MockTestSelectorBuilder {

        pub fn finish_with_from_ranges(ranges:Arc<parking_lot::RwLock<Vec<Range<i64>>>>) -> impl Default {
            let from_ctx = MockTestSelector::from_context();

            from_ctx.expect().returning( move |range|{
                ranges.write().push(range);
                Default::default()
            });

            from_ctx
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 64)]
    async fn test_degrade_once() -> Result<(), ThoError>{
        // let ranges:Arc<parking_lot::RwLock<Vec<Range<i64>>>> = Arc::new(RwLock::new(vec![]));
        //
        // let _from_ctx = MockTestSelectorBuilder::finish_with_from_ranges(Arc::clone(&ranges));
        //
        // let splits_adjuster: SplitsAdjuster<MockTestSelector, i64> = SplitsAdjuster::new(1);
        // let output_builder = MockTestOutputBuilderBuilder::default();
        // let mut context_impl:SplitsAdjusterContextImpl<MockTestTask, MockTestOutputBuilder>= SplitsAdjusterContextImpl::new(100, output_builder);
        //
        //
        // let mut selector = MockTestSelector::new();
        //
        //
        // let _adjusted = splits_adjuster.adjust(&mut selector, &mut context_impl).await?;
        //
        // assert!(splits_adjuster.degrade(&mut context_impl));//returns true, can degrade further
        //
        // let _adjusted = splits_adjuster.adjust(&mut selector, &mut context_impl).await?;
        //
        // context_impl.add_partial_output(MockTestOutput::new()).await;
        //
        // let _adjusted = splits_adjuster.adjust(&mut selector, &mut context_impl).await?;
        //
        // context_impl.add_partial_output(MockTestOutput::new()).await;
        //
        // assert!(context_impl.all_done());
        //
        // assert_eq!(
        //     &vec![
        //         0..100i64, //degrade ->
        //         0..50, 50..100
        //     ],
        //     &*ranges.read()
        // );

        Ok(())

    }
}
