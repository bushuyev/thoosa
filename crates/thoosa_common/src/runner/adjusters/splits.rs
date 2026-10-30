use crate::runner::oom_handler::AdjusterError;
use crate::runner::oom_handler::{Adjuster, InputContext};
use ambassador::delegatable_trait;
use async_trait::async_trait;
use log::{debug, info};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{AddAssign, DivAssign, Range};
use thoosa_shared::ThoError;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::Stage;
use thoosa_shared::traits::{Adjustable, Data, Metrics};

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
        info!("SplitsAdjuster::adjust, self={:?}", self);

        let split_start = adjust_context.done_to();

        let mut split_end = split_start.clone();
        split_end += adjust_context.split_len().clone();

        if let Some(overlap_len) = self.overlap_len.as_ref() {
            split_end += overlap_len.clone();
        }
        adjust_context.input_len().cap(&mut split_end);

        info!("SplitsAdjuster::adjust: split_start={split_start:?} split_end={split_end:?} adjustable.size={:?}", adjust_context.input_len());


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
    async fn add_partial_output(&mut self, o: R::Out) -> Result<(), ThoError> {
        // Commit split progress only after assembly succeeds.
        track_await!(self.output_builder.add_partial_output(o)).await?;

        self.done_to += self.split_len.clone();

        debug!("SplitsAdjusterContextImpl::add_partial_output: self.done_to={:?}  self.split_len={:?}",self.done_to, self.split_len);
        Ok(())
    }

    fn all_done(&self) -> bool {
        self.done_to >= self.input_len
    }


    async fn build_final_output(self) -> Result<R::Out, ThoError> {
        track_await!(self.output_builder.build()).await
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
        debug!("SplitsAdjusterContextImpl::new: input_len={input_len:?}");
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
    /// Complete assembly before reporting success; callers commit split progress only on Ok.
    async fn add_partial_output(&mut self, o: Self::O) -> Result<(), ThoError>;
    async fn build(self) -> Result<Self::O, ThoError>;
}

#[cfg(test)]
mod tests {
    use crate::runner::adjusters::splits::{OutputBuilder, SplitsAdjuster, SplitsAdjusterContextImpl};
    use crate::runner::common::MockRunCtx;
    use crate::runner::oom_handler::{Adjuster, InputContext};
    use async_trait::async_trait;
    use mockall::mock;
    use parking_lot::RwLock;
    use std::ops::Range;
    use std::sync::Arc;
    use thoosa_shared::CopyToDevice;
    use thoosa_shared::applicables_runner::{Input, Output};
    use thoosa_shared::traits::{Adjustable, Data, Stage};
    use thoosa_common::runner::sync_stage::{SyncStage};
    use thoosa_shared::ThoError;

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



            fn size(&self) -> i64;
            fn tho_device_idx(&self) -> usize;
            fn mem_size(&self) -> u64;
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



                fn size(&self) -> i64;
                fn tho_device_idx(&self) -> usize;
                fn mem_size(&self) -> u64;
        }

        impl Output for TestOutput {
        }

    }

    mock! {
        #[derive(Clone, Debug)]
        pub TestStage {}

        impl Stage for TestStage {
            type In<'a> = MockTestInput;
            type Out = MockTestOutput;
            type Ctx = MockRunCtx;
        }

        impl SyncStage for TestStage {
            fn do_run<'f>(self: Arc<Self>, _input: &<Self as Stage>::In<'f>, _ctx: &<Self as Stage>::Ctx) -> Result<<Self as Stage>::Out, ThoError>;
        }

        impl Adjustable for TestStage {

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
            async fn add_partial_output(&mut self, o: MockTestOutput) -> Result<(), ThoError>;
            async fn build(self) -> Result<MockTestOutput, ThoError>;
        }
    }

    #[allow(dead_code)]
    pub struct MockTestOutputBuilderBuilder {
        mock:MockTestOutputBuilder
    }

    #[allow(dead_code)]
    impl MockTestOutputBuilderBuilder {
        fn build() -> MockTestOutputBuilderBuilder {
            MockTestOutputBuilderBuilder {
                mock: MockTestOutputBuilder::new(),
            }
        }
        fn with_add_partial_output(mut self) -> Self {
            self.mock.expect_add_partial_output().returning(|_| Ok(()));

            self
        }

        fn with_partial_output_results(mut self, results: Vec<Result<(), ThoError>>) -> Self {
            // Script acceptance failures without involving GPU availability.
            for result in results {
                self.mock.expect_add_partial_output().times(1).return_once(move |_| result);
            }
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

    #[allow(dead_code)]
    pub struct MockTestSelectorBuilder {
        mock:MockTestSelector
    }



    #[allow(dead_code)]
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

    #[tokio::test]
    async fn test_assembly_error_stops_input_adjustment_before_build() {
        use crate::runner::oom_handler::{InputAdjustment, WithInputContext};

        type Context = SplitsAdjusterContextImpl<'static, MockTestStage, MockTestOutputBuilder>;
        type Adjustment = Box<dyn Adjuster<Context, A = Range<i64>>>;
        // No build expectation: the finalizer must never run after a rejected partial.
        let builder = MockTestOutputBuilderBuilder::build().with_partial_output_results(vec![Err(ThoError::oom())]).finish();
        let contexts = WithInputContext::new(Context::new(100, builder));
        let result = <Adjustment as InputAdjustment<'static, MockTestStage, _, Range<i64>>>::handle_success(MockTestOutput::new(), contexts).await;
        match result {
            Err(error) => assert!(error.is_retryable_oom()),
            Ok(_) => panic!("failed assembly must not return successful input adjustment"),
        }
    }

    #[tokio::test]
    async fn test_failed_assembly_preserves_cuda_error_and_split_progress() -> Result<(), ThoError> {
        for (message, oom, reset) in [
            ("CUDA out of memory", true, false),
            ("CUDA error: an illegal memory access was encountered", false, true),
        ] {
            let error = crate::tch_error::tch_error(tch::TchError::Torch(message.into()));
            let builder = MockTestOutputBuilderBuilder::build().with_partial_output_results(vec![Err(error), Ok(())]).finish();
            let mut context: SplitsAdjusterContextImpl<MockTestStage, MockTestOutputBuilder> = SplitsAdjusterContextImpl::new(100, builder);
            context.split_len = 50;

            let error = context.add_partial_output(MockTestOutput::new()).await.unwrap_err();
            assert_eq!((error.is_retryable_oom(), error.needs_cuda_restart()), (oom, reset));
            assert_eq!(context.done_to, 0);
            assert!(!context.all_done());
            let ThoError::Anyhow(source) = error else { panic!("expected original tensor error") };
            assert!(source.chain().any(|cause| cause.downcast_ref::<tch::TchError>().is_some()));

            context.add_partial_output(MockTestOutput::new()).await?;
            assert_eq!(context.done_to, 50);
            assert!(!context.all_done());
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 64)]
    async fn test_degrade_once() -> anyhow::Result<()>{
        let ranges:Arc<parking_lot::RwLock<Vec<Range<i64>>>> = Arc::new(RwLock::new(vec![]));

        let _from_ctx = MockTestSelectorBuilder::finish_with_from_ranges(Arc::clone(&ranges));

        let splits_adjuster: SplitsAdjuster<MockTestSelector, i64> = SplitsAdjuster::new(1);
        let output_builder = MockTestOutputBuilderBuilder::default();
        let mut context_impl:SplitsAdjusterContextImpl<MockTestStage, MockTestOutputBuilder>= SplitsAdjusterContextImpl::new(100, output_builder);


        let selector = MockTestSelector::new();


        let adjusted_selector = match splits_adjuster.adjust(selector, &mut context_impl).await {
            Ok(selector) => selector,
            Err(error) => anyhow::bail!("initial adjust failed: {error:?}"),
        };

        assert!(splits_adjuster.degrade(&mut context_impl));//returns true, can degrade further

        let adjusted_selector = match splits_adjuster.adjust(adjusted_selector, &mut context_impl).await {
            Ok(selector) => selector,
            Err(error) => anyhow::bail!("adjust after degrade failed: {error:?}"),
        };

        context_impl.add_partial_output(MockTestOutput::new()).await?;

        let _adjusted_selector = match splits_adjuster.adjust(adjusted_selector, &mut context_impl).await {
            Ok(selector) => selector,
            Err(error) => anyhow::bail!("adjust after partial output failed: {error:?}"),
        };

        context_impl.add_partial_output(MockTestOutput::new()).await?;

        assert!(context_impl.all_done());

        assert_eq!(
            &vec![
                0..100i64, //degrade ->
                0..50, 50..100
            ],
            &*ranges.read()
        );

        Ok(())

    }
}
