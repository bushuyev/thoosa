use crate::tasks::cross_batch_clc::{CrossClcResultMerge, CrossInput, CrossInputBatch1Selector, CrossUnfoldClcOutput};
#[cfg(test)]
use crate::tasks::cross_batch_clc::CrossUnfoldClc;
use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::RawTensorLoader;
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, DevicesWrapCopyMode, WithDataRef};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_shared::runner::common::{OutId, RunCtx};
use futures_util::future::BoxFuture;
use log::{debug, info};
use parking_lot::RwLock;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use futures_util::FutureExt;
use tch::Tensor;
use thoosa_common::dims::{AnyMetrics, AnySub, ChannelsData, ChannelsSlicesData, Dims};
use thoosa_common::runs::Simple;
use thoosa_motifs::find::FindMotifsSavableResult;
use thoosa_shared::{simple_id, ThoosaError};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_shared::runner::adjusters::permits::{PermitsAdjustContextImpl, PermitsAdjuster, PermitsAndDumpToCpuAdjustableContex};
use thoosa_shared::runner::adjusters::splits::{OutputBuilder, SplitsAdjuster, SplitsAdjusterContextImpl};
use thoosa_shared::resource::{Resource, ResourceStore, ResourcesConfig};
use thoosa_shared::runner::global_ctx::GlobalCtx;
use thoosa_shared::runner::oom_handler::{Adjusters, Contexts, NoopRunCtxContext, NoopRunnableContext, OOMHandler};
use thoosa_shared::runner::permits::{PermitStep, MAX_PERMITS};
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, Named, WithId};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::mono_key;
use thoosa_store::Store;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::traits::MonoMetrics;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_with_data::async_with_data;
use crate::tasks::slices_iter::SlicesIter;

#[derive(Debug)]
struct CrossUnfoldClcOutputBuilder<R: Debug> {
    partials: Vec<CrossUnfoldClcOutput<R>>,
}

impl<R: Debug> CrossUnfoldClcOutputBuilder<R> {
    fn new() -> Self {
        Self { partials: Vec::new() }
    }
}

impl<R> OutputBuilder for CrossUnfoldClcOutputBuilder<R>
where
    R: CrossClcResultMerge + Debug,
{
    type O = CrossUnfoldClcOutput<R>;

    fn add_partial_output(&mut self, o: Self::O) {
        self.partials.push(o);
    }

    async fn build(self) -> Result<Self::O, ThoosaError> {
        let mut iter = self.partials.into_iter();
        let first = iter
            .next()
            .ok_or_else(|| ThoosaError::new("CrossUnfoldClcOutputBuilder: no partial outputs"))?;
        let iter_0 = first.iter_0();
        let iter_1 = first.iter_1();

        let iter_0_step = iter_0.step;
        let iter_1_step = iter_1.step;
        let iter_0_resources = iter_0.resource_ids_sizes;
        let mut iter_1_resources = iter_1.resource_ids_sizes;

        let mut clc_result = first.clc_result();

        for partial in iter {
            let iter_1 = partial.iter_1();
            iter_1_resources.extend(iter_1.resource_ids_sizes);
            clc_result = clc_result.merge_batch_1(partial.clc_result())?;
        }

        Ok(CrossUnfoldClcOutput::new(
            clc_result,
            SlicesIter::new(iter_0_step, iter_0_resources),
            SlicesIter::new(iter_1_step, iter_1_resources),
        ))
    }
}

pub struct CrossLoader<ResC, L, CLC, RSV, R, C>
where
    C: RunCtx + Adjustable + 'static,
    ResC: ResourcesConfig<R = Vec<Resource>> + Debug,
    L: for<'i> Cached<
        In<'i> = Resource,
        Out = ThoTensor<ChannelsData2, SynKey>,
        Ctx=C
    > + OutId + 'static,

    R: Debug + Send + Sync + 'static,
    CLC: for<'i> SyncStage<
        In<'i> = CrossInput<'i, Arc<Resource>>,
        Out = CrossUnfoldClcOutput<R>,
        Ctx=C
    > + Adjustable + OutId + Send + Sync + 'static,

    RSV: for<'i> SyncStage<
        In<'i> = CrossUnfoldClcOutput<R>,
        Out = (),
        Ctx=C
    > + Stage<Ctx=C> + Send + Sync + 'static,
{
    loader: Arc<L>,
    store: Arc<RwLock<Box<dyn Store>>>,
    batch_clc: Arc<CLC>,
    receiver_factory: Box<dyn Fn(&[Arc<Resource>]) -> Arc<RSV> + Send + Sync>,
    chunk_size: usize,
    on_batch_done: Box<dyn Fn(R) -> BoxFuture<'static, R> + Sync + Send>,
    on_all_done: Box<dyn Fn(&Arc<RSV>) -> BoxFuture<'static, ()> + Sync + Send>,
    step:u32,
    pd: PhantomData<(ResC, R, C)>,
}

impl<ResC, L, CLC, RSV, R, C> Debug for CrossLoader<ResC, L, CLC, RSV, R, C>
where
    C: RunCtx + Adjustable + 'static,
    ResC: ResourcesConfig<R = Vec<Resource>> + Debug,
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx=C> + OutId + 'static,
    R: Debug + Send + Sync + 'static,
    CLC: for<'i> SyncStage<In<'i> = CrossInput<'i, Arc<Resource>>, Out = CrossUnfoldClcOutput<R>, Ctx=C, > + Adjustable + OutId + Send + Sync + 'static,
    RSV: for<'i> SyncStage<In<'i> = CrossUnfoldClcOutput<R>, Ctx=C> + Stage<Ctx=C, Out = ()> + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossLoader").finish()
    }
}

impl<ResC, L, CLC, RSV, R, C> CrossLoader<ResC, L, CLC, RSV, R, C>
where
    C: RunCtx + Adjustable + 'static,
    ResC: ResourcesConfig<R = Vec<Resource>> + Debug,
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx=C> + OutId + 'static,
    R: Debug + Send + Sync + 'static,
    CLC: for<'i> SyncStage<In<'i> = CrossInput<'i, Arc<Resource>>, Out = CrossUnfoldClcOutput<R>, Ctx=C, > + Adjustable + OutId + Send + Sync + 'static,
    RSV: for<'i> SyncStage<In<'i> = CrossUnfoldClcOutput<R>, Ctx=C> + Stage<Ctx=C, Out = ()> + Send + Sync + 'static,
{
    pub fn new<ReceiverFactory, OnAllDone, OnBatchDone>(
        loader: Arc<L>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        batch_clc: Arc<CLC>,
        receiver_factory: ReceiverFactory,
        chunk_size: usize,
        on_batch_done:OnBatchDone,
        on_all_done: OnAllDone,
        step: u32,
    ) -> Self
    where
        ReceiverFactory: Fn(&[Arc<Resource>]) -> Arc<RSV> + Sync + Send + 'static,
        OnBatchDone: Fn(R) -> BoxFuture<'static, R> + Sync + Send + 'static,
        OnAllDone: Fn(&Arc<RSV>) -> BoxFuture<'static, ()> + Sync + Send + 'static,
    {
        Self {
            loader,
            store: Arc::clone(store),
            batch_clc,
            receiver_factory: Box::new(receiver_factory),
            chunk_size,
            on_batch_done: Box::new(on_batch_done),
            on_all_done: Box::new(on_all_done),
            pd: Default::default(),
            step
        }
    }
}

// impl<ResC, L, CLC, RSV, R, C> OutId for CrossLoader<ResC, L, CLC, RSV, R, C>
// where
//     C:RunCtx+'static,
//     ResC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + 'static,
//     L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, CompKey>, Ctx=C> + OutId + 'static,
//     CLC: for<'i> SyncStage<In<'i> = CrossInput<'i, Arc<Resource>>, Out = CrossUnfoldClcOutput<R>, Ctx=C, > + Send + Sync + 'static,
//     RSV: for<'i> SyncStage<In<'i> = CrossUnfoldClcOutput<R>, Ctx=C> + Stage<Ctx=C, Out = ()> + Send + Sync + 'static,
//     R: Debug + Send + Sync + 'static,
//     C: RunCtx + 'static,
// {
//     fn out_id(&self, input_id: &<Self::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
//         CompKey::from(format!("cross_{}", input_id))
//     }
// }

//TODO move to lib
#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct ResourcesConfigInput<RC: ResourcesConfig + Debug + Send + Sync> {
    resources_config: RC,
}

impl<RC: ResourcesConfig + Debug + Send + Sync> ResourcesConfigInput<RC> {
    pub(crate) fn check(self, store: &Arc<RwLock<Box<dyn Store>>>) -> Result<Self, ThoosaError> {
        self.resources_config.check(store)?;

        Ok(self)
    }
}

impl<RC: ResourcesConfig + Debug + Send + Sync> ResourcesConfigInput<RC> {
    pub fn new(resources_config: RC) -> Self {
        Self {
            resources_config,
        }
    }


    pub fn resources_config(&self) -> &RC {
        &self.resources_config
    }
}

impl<ResC: ResourcesConfig + Debug + Send + Sync> WithId for ResourcesConfigInput<ResC> {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl<RC: ResourcesConfig + Debug + Send + Sync> Input for ResourcesConfigInput<RC> {}


impl<ResC, L, CLC, RSV, R, C> Stage for CrossLoader<ResC, L, CLC, RSV, R, C>
where
    C: RunCtx + Adjustable + 'static,
    ResC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + 'static,
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx=C> + OutId + 'static,
    CLC: for<'i> SyncStage<In<'i> = CrossInput<'i, Arc<Resource>>, Out = CrossUnfoldClcOutput<R>, Ctx=C, > + Adjustable + OutId + Send + Sync + 'static,
    RSV: for<'i> SyncStage<In<'i> = CrossUnfoldClcOutput<R>, Ctx=C> + Stage<Ctx=C, Out = ()> + Send + Sync + 'static,
    R: CrossClcResultMerge + Debug + Send + Sync + 'static,
{
    type Ctx=C;

    type In<'a> = ResourcesConfigInput<ResC>
    where
        Self: 'a;
    type Out = ();
}

// #[async_trait]
impl<ResC, L, CLC, RSV, R, C> AsyncStage for CrossLoader<ResC, L, CLC, RSV, R, C>
where
    C: RunCtx + Adjustable + 'static,
    for<'a> Self::In<'a>: Data + WithId + Send + Sync,
    ResourcesConfigInput<ResC>: Data<M=CpuOnly<ResourcesConfigInput<ResC>>> + WithId + Send + Sync,
    CpuOnly<ResourcesConfigInput<ResC>>: DevicesWrapCopyMode<ResourcesConfigInput<ResC>>,
    Self::Out: Data + WithId + Send + Sync,
    <Self::Out as Data>::M: DevicesWrapCopyMode<Self::Out>,
    ResC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + 'static,
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx=C> + OutId + 'static,
    CLC: for<'i> SyncStage<In<'i> = CrossInput<'i, Arc<Resource>>, Out = CrossUnfoldClcOutput<R>, Ctx=C, > + Adjustable + OutId + Send + Sync + 'static,
    RSV: for<'i> SyncStage<In<'i> = CrossUnfoldClcOutput<R>, Ctx=C> + Stage<Ctx=C, Out = ()> + Send + Sync + 'static,
    R: CrossClcResultMerge + Debug + Send + Sync + 'static,
{
    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        ctx: Arc<parking_lot::RwLock<Self::Ctx>>,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
        // let chunk_size = self.chunk_size;
        let max_batch = i64::try_from(self.chunk_size).unwrap_or(i64::MAX);
        let device_idx = ctx.read().device_idx();
        let loader_ctx = Arc::clone(&ctx).write().branch("LoaderForCrossLoader_TODO");//self id

        let resources = input.with_data_ref(0, Arc::clone(&ctx), None, |input: &ResourcesConfigInput<ResC>|{
                Ok(input
                    .resources_config
                    .to_resources(&self.store)?
                    .into_iter()
                    .filter(|r|r.length() > self.step)
                    .map(|r| Arc::new(r))
                    .collect::<Vec<_>>())
            }).await?;

        debug!("CrossLoader::run:  max_batch={max_batch} resources={resources:?}");

        let receiver = (self.receiver_factory)(resources.as_slice());

        let mut t_0_batch: Vec<_> = vec![];
        let mut t_0_batch_resources: Vec<_> = vec![];
        let mut t_0_batch_len = 0;

        let mut skip_0_i = 0;

        for (i_0, resource_0) in resources.iter().enumerate() {
            info!("CrossLoader::run: on i_0={i_0} resource_0={resource_0:?}");


            let t_0 = Arc::clone(&self.loader)
                .run_cached(Arc::new(DevicesWrap::wrap(resource_0.as_ref().clone(), 0)), Arc::clone(&loader_ctx))
                .await?;

            t_0_batch_len += t_0.size().dim_1();
            t_0_batch.push(t_0);
            t_0_batch_resources.push(Arc::clone(resource_0));

            if t_0_batch_len < max_batch && i_0 < resources.len() - 1 {
                continue;
            }

            let mut t_1_batch: Vec<_> = vec![];
            let mut t_1_batch_resources: Vec<_> = vec![];
            let mut t_1_batch_len = 0;

            for (i_1, resource_1) in resources
                .iter()
                .enumerate()
                .skip(skip_0_i)//skipping lower triangle
            {

                let t_1 = Arc::clone(&self.loader)
                    .run_cached(Arc::new(DevicesWrap::wrap(resource_1.as_ref().clone(), 0)), Arc::clone(&loader_ctx))
                    .await?;

                t_1_batch_len += t_1.size().dim_1();

                t_1_batch.push(t_1);
                t_1_batch_resources.push(Arc::clone(resource_1));

                if t_1_batch_len < max_batch && i_1 < resources.len() - 1 {
                    continue;
                }

                debug!("CrossLoader::run: t_0_batch={t_0_batch:?} t_1_batch={t_1_batch:?}");
                let batches_concated = [t_0_batch.clone(), t_1_batch].concat();
                let batch_clc = Arc::clone(&self.batch_clc);
                let ctx_for_clc = Arc::clone(&ctx);
                let t_0_batch_resources_cloned = t_0_batch_resources.clone();
                let t_1_batch_resources_cloned = t_1_batch_resources.clone();
                let mut cross_clc_output = async_with_data!(
                    Arc::clone(&ctx),
                    device_idx,
                    [batches_concated];
                    |data| async move {
                        let input = CrossInput::new(
                            data,
                            t_0_batch_resources_cloned.as_slice(),
                            t_1_batch_resources_cloned.as_slice(),
                        );

                        debug!("CrossLoader::run: input={input:?}");

                        let ctx_name = ctx_for_clc.read().name();
                        let required_ids: HashMap<&'static str, &HashSet<&dyn CachableIdTr>> = HashMap::new();
                        let input_len = input.size();
                        let out_key = Arc::new(batch_clc.out_id(input.id())) as Arc<dyn CachableIdTr>;
                        let out_key_for_permits = Arc::clone(&out_key);

                        let out = OOMHandler::new(
                            vec![
                                // Adjusters::RunContext(Box::new(PermitsAdjuster::new(
                                //     MAX_PERMITS,
                                //     *PermitStep::TWENTY,
                                //     move |n| {
                                //         GlobalCtx::global().acquire_many(
                                //             device_idx,
                                //             n,
                                //             Arc::new(simple_id!(ctx_name))
                                //         )
                                //     },
                                // ))),
                                // Adjusters::RunContext(Box::new(DumpToCpu::new())),
                                Adjusters::Input(Box::new(SplitsAdjuster::<CrossInputBatch1Selector, i64>::new(1))),
                            ],
                            None,
                        )
                        .run(
                            &batch_clc,
                            Arc::new(input),
                            Arc::clone(&ctx_for_clc),
                            Contexts::new(
                                SplitsAdjusterContextImpl::new(input_len, CrossUnfoldClcOutputBuilder::new()),
                                NoopRunnableContext {},
                                NoopRunCtxContext{},
                                // PermitsAndDumpToCpuAdjustableContex::new(
                                //     PermitsAdjustContextImpl::new(ctx_name),
                                //     DumpToCpuContextImpl::new(&required_ids),
                                // ),
                            ),
                        )
                        .await?;
                        Ok::<_, ThoosaError>(out)
                    }
                )?;

                let cross_clc_output = cross_clc_output.replace_clc_result(&self.on_batch_done).await;

                let receiver_arc = Arc::clone(&receiver);
                let ctx_arc = Arc::clone(&ctx);

                receiver_arc.consume(cross_clc_output, ctx_arc)?;


                t_1_batch = vec![];
                t_1_batch_resources = vec![];
                t_1_batch_len = 0;
            }

            skip_0_i = i_0;
            t_0_batch.clear();
            t_0_batch_resources.clear();
            t_0_batch_len = 0;


        }

        (self.on_all_done)(&receiver).await;

        Ok(Arc::new(DevicesWrap::wrap((), 0)))
    }
}


#[cfg(test)]
pub(crate) mod mock_cross_loader_clc {
    use thoosa_shared::runner::common::runnable::{Stage, SyncStage};
    use thoosa_shared::runner::common::{OutId, RunCtx};
    use crate::tasks::cross_batch_clc::{CrossInput, CrossUnfoldClcOutput};
    use parking_lot::RwLock;
    use std::fmt::Debug;
    use std::marker::PhantomData;
    use std::sync::Arc;
    use thoosa_shared::ThoosaError;
    use thoosa_shared::applicables_runner::{Input, Output};
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::{Adjustable, WithId};

    type RunBatchFn<Out, C> = dyn for<'a> Fn(
        &CrossInput<'a, Arc<Resource>>,
        Arc<RwLock<C>>,
    ) -> Result<CrossUnfoldClcOutput<Out>, ThoosaError>
    + Send
    + Sync
    + 'static;

    #[derive(Clone)]
    pub struct MockCrossLoaderClc<Out, C>
    where
        Out: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        run_batch_fn: Arc<RunBatchFn<Out, C>>,
        _ctx: PhantomData<C>,
    }

    impl<Out, C> std::fmt::Debug for MockCrossLoaderClc<Out, C>
    where
        Out: Input + Output + Send + Debug + 'static,
        C: RunCtx,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockCrossLoaderClc").finish()
        }
    }

    impl<Out, C> Adjustable for MockCrossLoaderClc<Out, C>
    where
        Out: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {}

    impl<Out, C> OutId for MockCrossLoaderClc<Out, C>
    where
        Out: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
            input_id.clone()
        }
    }

    impl<Out, C> Stage for MockCrossLoaderClc<Out, C>
    where
        Out: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        type Ctx = C;
        type In<'a> = CrossInput<'a, Arc<Resource>>;
        type Out = CrossUnfoldClcOutput<Out>;
    }

    impl<Out, C> SyncStage for MockCrossLoaderClc<Out, C>
    where
        Out: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        fn do_run<'s>(
            self: Arc<Self>,
            input: &Self::In<'s>,
            ctx: Arc<RwLock<Self::Ctx>>,
        ) -> Result<Self::Out, ThoosaError>
        where
            Self: Sized,
        {
            let f = Arc::clone(&self.run_batch_fn);
            f(input, ctx)
        }
    }

    pub struct MockCrossLoaderClcBuilder<Out, C>
    where
        Out: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        run_batch_fn: Option<Arc<RunBatchFn<Out, C>>>,
        _ctx: PhantomData<C>,
    }

    impl<Out, C> MockCrossLoaderClcBuilder<Out, C>
    where
        Out: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        pub fn with_run_batch_fn<F>(mut self, f: F) -> Self
        where
            F: for<'a> Fn(
                &CrossInput<'a, Arc<Resource>>,
                Arc<RwLock<C>>,
            ) -> Result<CrossUnfoldClcOutput<Out>, ThoosaError>
            + Send
            + Sync
            + 'static,
        {
            self.run_batch_fn = Some(Arc::new(f));
            self
        }

        pub fn finish(self) -> MockCrossLoaderClc<Out, C> {
            let run_batch_fn = self
                .run_batch_fn
                .expect("MockCrossLoaderClcBuilder requires run_batch_fn to be set");
            MockCrossLoaderClc {
                run_batch_fn,
                _ctx: PhantomData,
            }
        }
    }

    pub fn build<Out, C>() -> MockCrossLoaderClcBuilder<Out, C>
    where
        Out: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        MockCrossLoaderClcBuilder {
            run_batch_fn: None,
            _ctx: PhantomData,
        }
    }
}

#[cfg(test)]
pub(crate) mod mock_cross_loader_receiver {
    use thoosa_shared::runner::common::runnable::{Stage, SyncStage};
    use thoosa_shared::runner::common::RunCtx;
    use mockall::mock;
    use parking_lot::RwLock;
    use std::fmt::Debug;
    use std::sync::Arc;
    use thoosa_shared::ThoosaError;
    use thoosa_shared::applicables_runner::{Input, Output};

    mock! {
        #[derive(Debug)]
        pub CrossLoaderReceiver<In, C>
        where
            In: Input + Output + Send + Debug + 'static,
            C: RunCtx + 'static,
        {
        }

        impl<In, C> Stage for CrossLoaderReceiver<In, C>
        where
            In: Input + Output + Send + Debug + 'static,
            C: RunCtx + 'static,
        {
            type Ctx = C;
            type In<'a> = In where Self: 'a;
            type Out = ();
        }

        impl<In, C> SyncStage for CrossLoaderReceiver<In, C>
        where
            In: Input + Output + Send + Debug + 'static,
            C: RunCtx + 'static,
        {
            fn consume<'s>(self: Arc<Self>, out: <Self as Stage>::In<'s>, ctx: Arc<RwLock<<Self as Stage>::Ctx>>) -> Result<<Self as Stage>::Out, ThoosaError>;
        }
    }

    pub struct MockCrossLoaderReceiverBuilder<In, C>
    where
        In: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        mock: MockCrossLoaderReceiver<In, C>,
        consume_configured: bool,
    }

    impl<In, C> MockCrossLoaderReceiverBuilder<In, C>
    where
        In: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        pub fn with_receive_fn<F>(mut self, f: F) -> Self
        where
            F: Fn(In, Arc<RwLock<C>>) -> Result<(), ThoosaError> + Send + Sync + 'static,
        {
            let receive_fn = Arc::new(f);
            self.mock
                .expect_consume()
                .returning(move |out, ctx| receive_fn(out, ctx));
            self.consume_configured = true;
            self
        }

        pub fn finish(self) -> MockCrossLoaderReceiver<In, C> {
            assert!(
                self.consume_configured,
                "MockCrossLoaderReceiverBuilder requires receive_fn to be set"
            );
            self.mock
        }
    }

    pub fn build<In, C>() -> MockCrossLoaderReceiverBuilder<In, C>
    where
        In: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        MockCrossLoaderReceiverBuilder {
            mock: MockCrossLoaderReceiver::new(),
            consume_configured: false,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::mock_cross_loader_clc;
    use super::mock_cross_loader_receiver;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use super::*;
    use crate::tasks::cross_batch_clc::{CrossUnfoldClc, CrossUnfoldClcOutput, UnfoldParams};
    use crate::tasks::cross_loader::{CrossLoader, ResourcesConfigInput};
    use crate::tasks::dists_tracker::{DistsMask, DistsTracker};
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::RawTensorLoader;
    use crate::tasks::slices_iter::SlicesIter;
    use thoosa_shared::runner::common::{mock_run_ctx, MockRunCtx};
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use futures_util::FutureExt;
    use log::debug;
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tch::{Device, Kind, Tensor};
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::tho_device::ThoDevice;
    use thoosa_common::ws::FsWorkspace;
    use thoosa_motifs::find::FixedFindMotifsConfig;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::{Resource, ResourceStats};
    use thoosa_shared::{cdt_for_dir, ResourceType, ThoosaError};
    use thoosa_shared::mono_key;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_store::fs_store::ChunkedSlicesReader;
    use thoosa_store::{MockStoreBuilder, SlicesMatch, SqlStore};
    use tokio::task::yield_now;
    use tokio::time::sleep;
    use thoosa_common::tho_tensor::ThoTensor;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Observation<ID> {
        tensor_ids: Vec<ID>,
        batch_0_ids: Vec<u32>,
        batch_1_ids: Vec<u32>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, DummyData)]
    struct TestBatchOutput {
        id: SynKey,
        call_index: usize,
    }

    impl TestBatchOutput {
        fn new(call_index: usize) -> Self {
            Self {
                id: SynKey::new([mono_key!("batch_output", "batch_output")]),
                call_index,
            }
        }
    }

    impl WithId for TestBatchOutput {
        type ID = SynKey;

        fn id(&self) -> &Self::ID {
            &self.id
        }
    }

    impl Input for TestBatchOutput {}
    impl Output for TestBatchOutput {}

    impl CrossClcResultMerge for TestBatchOutput {
        fn merge_batch_1(self, _other: Self) -> Result<Self, ThoosaError> {
            Ok(self)
        }
    }

    fn slice_sum(base: f32, slice_idx: u32) -> f32 {
        2.0 * base + (2.0 * slice_idx as f32 + 1.0) / 10.0
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_cross_clc_without_oom() -> Result<(), ThoosaError> {
        test_cross_clc(None, 20).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_cross_clc_with_oom() -> Result<(), ThoosaError> {
        test_cross_clc(Some(20), 40).await
    }

    async fn test_cross_clc(oom_limit: Option<i64>, chunk_size: usize) -> Result<(), ThoosaError> {
        GlobalCtx::init_devices(2);

        let tensor_len = 10usize;
        let unfold_size = 2i64;
        let unfold_step = 1i64;

        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        fs::create_dir_all(&test_dir)?;
        let ws = Arc::new(RwLock::new(FsWorkspace::new(test_dir, 1, 1000, 32000)?));
        let store = Arc::new(RwLock::new(SqlStore::from_path(ws.read().store_path())?));

        let resource_keys: Vec<String> = vec!["111", "222", "333", "444", "555", "666"]
            .into_iter()
            .map(|v| v.to_string())
            .collect();

        let ctx = Arc::new(parking_lot::RwLock::new(RunCtx2::new(1, 0, "aa")));

        let resource_stats = resource_keys
            .iter()
            .map(|key| (key.clone(), (ResourceStats::new(vec![1, tensor_len as u32]), vec![])))
            .collect::<HashMap<_, _>>();

        {
            let store_guard = store.write();
            let mut conn_guard = store_guard.conn().lock();
            let t = conn_guard.transaction()?;
            let import_id = store_guard.save_import("test resources".to_string(), &t)?;
            store_guard.save_resources_labels(&resource_stats, import_id, &t)?;
            t.commit()?;
        }

        let run_id = store.write().save_run_config(
            None,
            &"find_motifs",
            "test_cross_clc",
            "",
            HashMap::new(),
        )?;
        let run_config = FindMotifsRunConfig::new(Some(run_id), 1, 2, None, None);
        let fixed_config = FixedFindMotifsConfig::new(None, 2, false, None);

        let resource_key_refs = resource_keys.iter().map(|v| v.as_str()).collect::<Vec<_>>();
        let resources = store.read().list_resources_by_file_stems(&resource_key_refs)?;
        let resources_to_move = resources.clone();
        let oom_triggered = Arc::new(AtomicBool::new(false));
        let oom_triggered_cloned = Arc::clone(&oom_triggered);

        let loader = Arc::new(RawTensorLoader::<RunCtx2>::new(
            move |key, _| {
                let i = resources_to_move
                    .iter()
                    .position(|v| v.eq(key))
                    .expect(format!("no {key:?}").as_str())
                    + 1;
                let v = (0..tensor_len)
                    .map(|v| i as f32 + v as f32 / 10.0)
                    .collect::<Vec<f32>>();
                debug!("executor: key={key:?} i={i} v={v:?}");

                Ok(ThoTensor::new(
                    SynKey::new([mono_key!(key.id(), format!("resource_id={}", key.id()))]),
                    Tensor::from_slice(v.as_slice()).view((1, tensor_len as i64)),
                )?)
            },
            ThoDevice::from(Device::Cpu),
        ));

        let batch_clc = Arc::new(CrossUnfoldClc::new(
            move |t_0, t_1, _input, _ctx, _| {
                debug!("test_cross_cls, batch_clc: t_0={:?}, t_1={:?}", t_0.size(), t_1.size());
                if let Some(limit) = oom_limit {
                    let batch_1_len = t_1.size()[1];
                    if batch_1_len > limit {
                        debug!(
                            "test_cross_clc: triggering OOM: batch_1_len={batch_1_len} limit={limit}"
                        );
                        oom_triggered_cloned.store(true, Ordering::SeqCst);
                        return Err(ThoosaError::oom());
                    }
                }
                let sum_0 = t_0.f_sum_dim_intlist(2, false, Kind::Float)?;
                let sum_1 = t_1.f_sum_dim_intlist(2, false, Kind::Float)?;
                let dists = sum_0.unsqueeze(2) + sum_1.unsqueeze(1);
                let mask = (sum_0.unsqueeze(2) - sum_1.unsqueeze(1)).lt(0.0);
                Ok(DistsMask::new(mask, Some(dists), false, None))
            },
            (),
            UnfoldParams::new(1, unfold_size, unfold_step),
        ));

        let cross_loader = Arc::new(CrossLoader::new(
            loader,
            &store,
            batch_clc,
            {
                let ws = Arc::clone(&ws);
                let store = Arc::clone(&store);
                let run_config = run_config.clone();
                let fixed_config = fixed_config.clone();
                move |_resources: &[Arc<Resource>]| {
                    // Integration-style test to exercise CrossLoader + DistsTracker end-to-end.
                    Arc::new(DistsTracker::<RunCtx2>::new(
                        &ws,
                        &store,
                        &run_config,
                        &fixed_config,
                        None,
                    ))
                }
            },
            chunk_size, //TODO config
            |batch| async { batch }.boxed(),
            |_receiver| async {}.boxed(),
            0,
        ));

        let cache_area = ctx.read().cache_area();
        let input = Arc::new(DevicesWrap::wrap(
            ResourcesConfigInput::new(Simple::new(ResourceType::List(resource_keys))),
            cache_area,
        ));

        let output = cross_loader
            .run(input, Arc::clone(&ctx))
            .await?;

        assert_eq!(output.id(), &());
        if oom_limit.is_some() {
            assert!(oom_triggered.load(Ordering::SeqCst));
        }

        let slices_per_resource = (tensor_len as i64 - unfold_size + 1) as u32;
        let mut expected = HashMap::new();

        for (i_0, resource_0) in resources.iter().enumerate() {
            let base_0 = (i_0 + 1) as f32;
            for (i_1, resource_1) in resources.iter().enumerate().skip(i_0 + 1) {
                let base_1 = (i_1 + 1) as f32;
                for slice_0 in 0..slices_per_resource {
                    for slice_1 in 0..slices_per_resource {
                        let sum_0 = slice_sum(base_0, slice_0);
                        let sum_1 = slice_sum(base_1, slice_1);
                        if sum_0 < sum_1 {
                            expected.insert(
                                (resource_0.id(), slice_0, resource_1.id(), slice_1),
                                sum_0 + sum_1,
                            );
                        }
                    }
                }
            }
        }

        let expected_len = expected.len();
        assert!(expected_len > 0);

        let slices_path = ws
            .read()
            .tmp_path()
            .join(format!("{run_id}_0_slices.zstd"));
        let reader = ChunkedSlicesReader::<SlicesMatch>::new(&slices_path)
            .map_err(|e| ThoosaError::new(format!("read slices file: {e}")))?;

        let mut remaining = expected;
        let mut actual_count = 0usize;
        for chunk in reader {
            let chunk = chunk.map_err(|e| ThoosaError::new(format!("read slices chunk: {e}")))?;
            for entry in chunk {
                actual_count += 1;
                let key = (entry.0.0, entry.0.1, entry.1.0, entry.1.1);
                let expected_dist = remaining
                    .remove(&key)
                    .unwrap_or_else(|| panic!("unexpected slice match {key:?}"));
                let delta = (entry.2 - expected_dist).abs();
                assert!(
                    delta < 1e-4,
                    "dist mismatch for {key:?}: got {}, expected {}",
                    entry.2,
                    expected_dist
                );
            }
        }

        assert_eq!(actual_count, expected_len);
        assert!(remaining.is_empty());

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_cross_loader() -> Result<(), ThoosaError> {
        GlobalCtx::init_devices(2);

        let resource_a: Resource = (1, 1, "resource_a", 2).into();
        let resource_b: Resource = (2, 1, "resource_b", 2).into();
        let resources = vec![resource_a.clone(), resource_b.clone()];



        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_run_cached([
                    (resource_a.clone(), Tensor::from_slice(&[1.0, 2.0]).view((1,2))),
                    (resource_b.clone(), Tensor::from_slice(&[3.0, 4.0]).view((1,2))),
                ])
                .finish(),
        );

        let observations = Arc::new(Mutex::new(Vec::<Observation<SynKey>>::new()));
        let produced_outputs = Arc::new(Mutex::new(Vec::<TestBatchOutput>::new()));
        let counter = Arc::new(AtomicUsize::new(0));

        let batch_observations = Arc::clone(&observations);
        let batch_outputs = Arc::clone(&produced_outputs);
        let batch_counter = Arc::clone(&counter);

        let batch_clc = Arc::new(
            mock_cross_loader_clc::build::<TestBatchOutput, MockRunCtx>()
                .with_run_batch_fn(move |input, _ctx| {
                    let observation = Observation {
                        tensor_ids: input.tensors().iter().map(|t| t.id().clone()).collect::<Vec<_>>(),
                        batch_0_ids: input.batch_0().iter().map(|r| r.as_ref().id()).collect(),
                        batch_1_ids: input.batch_1().iter().map(|r| r.as_ref().id()).collect(),
                    };

                    batch_observations.lock().unwrap().push(observation);

                    let call_index = batch_counter.fetch_add(1, Ordering::SeqCst);
                    let output = TestBatchOutput::new(call_index);
                    batch_outputs.lock().unwrap().push(output.clone());

                    Ok(CrossUnfoldClcOutput::new(
                        output,
                        SlicesIter::new(1, vec![]),
                        SlicesIter::new(1, vec![]),
                    ))
                })
                .finish(),
        );

        let received_outputs = Arc::new(Mutex::new(Vec::<TestBatchOutput>::new()));
        let receiver_observed = Arc::clone(&received_outputs);

        let ctx = Arc::new(RwLock::new(mock_run_ctx::default(0)));

        let mut mocked_store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
                .with_list_resources_by_file_stems(
                    vec!["resource_a".to_string(), "resource_b".to_string()],
                    Ok(resources.clone()),
                )
                .build() as Box<dyn Store>,
        ));

        let cross_loader = Arc::new(CrossLoader::new(
            loader,
            &mocked_store,
            batch_clc,
            {
                let receiver_observed = Arc::clone(&receiver_observed);
                move |_resources: &[Arc<Resource>]| {
                    Arc::new(
                        mock_cross_loader_receiver::build::<CrossUnfoldClcOutput<TestBatchOutput>, MockRunCtx>()
                            .with_receive_fn({
                                let receiver_observed = Arc::clone(&receiver_observed);
                                move |out, _ctx| {
                                    let result = out.clc_result();
                                    receiver_observed.lock().unwrap().push(result);
                                    Ok(())
                                }
                            })
                            .finish(),
                    )
                }
            },
            2,
            |batch|{
                async {batch}.boxed()
            },
            |p| {
                async {
                    // cross_loader.receiver;
                }
                .boxed()
            },
            0,
        ));

        let cache_area = ctx.read().cache_area();
        let input = Arc::new(DevicesWrap::wrap(
            ResourcesConfigInput::new(Simple::new(ResourceType::List(vec![
                "resource_a".to_string(),
                "resource_b".to_string(),
            ]))),
            cache_area,
        ));

        let _ = cross_loader
            .run(input, Arc::clone(&ctx))
            .await?;

        yield_now().await;
        sleep(Duration::from_millis(10)).await;

        let observed = observations.lock().unwrap().clone();

        let expected = vec![
            Observation {
                tensor_ids: vec![
                    SynKey::new([mono_key!("resource_a", "resource_a")]),
                    SynKey::new([mono_key!("resource_a", "resource_a")]),
                ],
                batch_0_ids: vec![resources[0].id()],
                batch_1_ids: vec![resources[0].id()],
            },
            Observation {
                tensor_ids: vec![
                    SynKey::new([mono_key!("resource_a", "resource_a")]),
                    SynKey::new([mono_key!("resource_b", "resource_b")]),
                ],
                batch_0_ids: vec![resources[0].id()],
                batch_1_ids: vec![resources[1].id()],
            },
            Observation {
                tensor_ids: vec![
                    SynKey::new([mono_key!("resource_b", "resource_b")]),
                    SynKey::new([mono_key!("resource_a", "resource_a")]),
                ],
                batch_0_ids: vec![resources[1].id()],
                batch_1_ids: vec![resources[0].id()],
            },
            Observation {
                tensor_ids: vec![
                    SynKey::new([mono_key!("resource_b", "resource_b")]),
                    SynKey::new([mono_key!("resource_b", "resource_b")]),
                ],
                batch_0_ids: vec![resources[1].id()],
                batch_1_ids: vec![resources[1].id()],
            },
        ];

        assert_eq!(observed, expected);

        let produced = produced_outputs.lock()?.clone();
        let received = received_outputs.lock()?.clone();

        let expected_outputs: Vec<TestBatchOutput> = (0..expected.len())
            .map(TestBatchOutput::new)
            .collect();

        assert_eq!(produced, expected_outputs);
        assert_eq!(produced, received);

        Ok(())
    }
}
