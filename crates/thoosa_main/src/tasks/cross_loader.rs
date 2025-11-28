use crate::tasks::cross_batch_clc::{CrossInput, CrossUnfoldClcOutput};
#[cfg(test)]
use crate::tasks::cross_batch_clc::CrossUnfoldClc;
use crate::tasks::cross_dims::{ChannelsData2};
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::{RawTensorLoader};
use async_trait::async_trait;
use caching_runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use caching_runner::common::runnable::{AsyncStage, Stage, SyncStage};
use caching_runner::common::{OutId, RunCtx};
use futures_util::future::BoxFuture;
use log::{debug, info};
use parking_lot::RwLock;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use tch::Tensor;
use thoosa_common::dims::{AnyMetrics, AnySub, ChannelsData, ChannelsSlicesData, Dims};
use thoosa_common::runs::Simple;
use thoosa_motifs::find::FindMotifsSavableResult;
use thoosa_shared::ThoosaError;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::resource::{Resource, ResourceStore, ResourcesConfig};
use thoosa_shared::traits::{Data, Named, SimpleId, WithId};
use thoosa_store::Store;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::traits::MonoMetrics;
use thoosa_dummy_data_derive::DummyData;

// #[derive(Debug, cached_derive::Cached)]
// #[cache_name = "LoadedTensorId"]
pub struct CrossLoader<ResC, L, CLC, RSV, R, C>
where
    C:RunCtx+'static,
    ResC: ResourcesConfig<R = Vec<Resource>> + Debug,
    L: for<'i> Cached<
        In<'i> = Resource,
        Out = ThoTensor<ChannelsData2, SimpleId>,
        Ctx=C
    > + OutId + 'static,

    R: Debug + Send + Sync + 'static,
    CLC: for<'i> SyncStage<
        In<'i> = CrossInput<'i, &'i Arc<Resource>>,
        Out = CrossUnfoldClcOutput<R>,
        Ctx=C
    > + Send + Sync + 'static,

    RSV: for<'i> SyncStage<
        In<'i> = CrossUnfoldClcOutput<R>,
        Out = (),
        Ctx=C
    > + Stage<Ctx=C> + Send + Sync + 'static,
    C: RunCtx + 'static,
{
    loader: Arc<L>,
    store: Arc<RwLock<Box<dyn Store>>>,
    batch_clc: Arc<CLC>,
    receiver: Arc<RSV>,
    chunk_size: usize,
    on_batch_done: Box<dyn Fn(R) -> BoxFuture<'static, R> + Sync + Send>,
    on_all_done: Box<dyn Fn(&Arc<RSV>) -> BoxFuture<'static, ()> + Sync + Send>,
    pd: PhantomData<(ResC, R, C)>,
}

impl<ResC, L, CLC, RSV, R, C> Debug for CrossLoader<ResC, L, CLC, RSV, R, C>
where
    C:RunCtx+'static,
    ResC: ResourcesConfig<R = Vec<Resource>> + Debug,
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
    R: Debug + Send + Sync + 'static,
    CLC: for<'i> SyncStage<In<'i> = CrossInput<'i, &'i Arc<Resource>>, Out = CrossUnfoldClcOutput<R>, Ctx=C, > + Send + Sync + 'static,
    RSV: for<'i> SyncStage<In<'i> = CrossUnfoldClcOutput<R>, Ctx=C> + Stage<Ctx=C, Out = ()> + Send + Sync + 'static,
    C: RunCtx + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossLoader").finish()
    }
}

impl<ResC, L, CLC, RSV, R, C> CrossLoader<ResC, L, CLC, RSV, R, C>
where
    C:RunCtx+'static,
    ResC: ResourcesConfig<R = Vec<Resource>> + Debug,
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
    R: Debug + Send + Sync + 'static,
    CLC: for<'i> SyncStage<In<'i> = CrossInput<'i, &'i Arc<Resource>>, Out = CrossUnfoldClcOutput<R>, Ctx=C, > + Send + Sync + 'static,
    RSV: for<'i> SyncStage<In<'i> = CrossUnfoldClcOutput<R>, Ctx=C> + Stage<Ctx=C, Out = ()> + Send + Sync + 'static,
    C: RunCtx + 'static,
{
    pub fn new<OnAllDone, OnBatchDone>(
        loader: Arc<L>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        batch_clc: Arc<CLC>,
        receiver: Arc<RSV>,
        chunk_size: usize,
        on_batch_done:OnBatchDone,
        on_all_done: OnAllDone,
    ) -> Self
    where
        OnBatchDone: Fn(R) -> BoxFuture<'static, R> + Sync + Send + 'static,
        OnAllDone: Fn(&Arc<RSV>) -> BoxFuture<'static, ()> + Sync + Send + 'static,
    {
        Self {
            loader,
            store: Arc::clone(store),
            batch_clc,
            receiver,
            chunk_size,
            on_batch_done: Box::new(on_batch_done),
            on_all_done: Box::new(on_all_done),
            pd: Default::default(),
        }
    }
}

impl<ResC, L, CLC, RSV, R, C> OutId for CrossLoader<ResC, L, CLC, RSV, R, C>
where
    C:RunCtx+'static,
    ResC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + 'static,
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
    CLC: for<'i> SyncStage<In<'i> = CrossInput<'i, &'i Arc<Resource>>, Out = CrossUnfoldClcOutput<R>, Ctx=C, > + Send + Sync + 'static,
    RSV: for<'i> SyncStage<In<'i> = CrossUnfoldClcOutput<R>, Ctx=C> + Stage<Ctx=C, Out = ()> + Send + Sync + 'static,
    R: Debug + Send + Sync + 'static,
    C: RunCtx + 'static,
{
    fn out_id(&self, input_id: &<Self::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        SimpleId::from(format!("cross_{}", input_id))
    }
}

//TODO move to lib
#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct ResourcesConfigInput<RC: ResourcesConfig + Debug + Send + Sync> {
    id: SimpleId,
    resources_config: RC,
}

impl<RC: ResourcesConfig + Debug + Send + Sync> ResourcesConfigInput<RC> {
    pub fn new(id: SimpleId, resources: RC) -> Self {
        Self {
            id,
            resources_config: resources, //: resources.to_resources().into_iter().map(|r| Arc::new(r)).collect(),
        }
    }

    pub fn id(&self) -> &SimpleId {
        &self.id
    }

    pub fn resources_config(&self) -> &RC {
        &self.resources_config
    }
}

impl<ResC: ResourcesConfig + Debug + Send + Sync> WithId for ResourcesConfigInput<ResC> {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<RC: ResourcesConfig + Debug + Send + Sync> Input for ResourcesConfigInput<RC> {}

#[derive(Debug, Clone, DummyData)]
#[dummy_data(clone)]
pub struct CrossResult {
    id: SimpleId,
}

impl WithId for CrossResult {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}
impl Output for CrossResult {}

impl CrossResult {
    pub fn new(id: SimpleId) -> Self {
        Self { id }
    }
}
impl<ResC, L, CLC, RSV, R, C> Stage for CrossLoader<ResC, L, CLC, RSV, R, C>
where
    C:RunCtx+'static,
    ResC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + 'static,
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
    CLC: for<'i> SyncStage<In<'i> = CrossInput<'i, &'i Arc<Resource>>, Out = CrossUnfoldClcOutput<R>, Ctx=C, > + Send + Sync + 'static,
    RSV: for<'i> SyncStage<In<'i> = CrossUnfoldClcOutput<R>, Ctx=C> + Stage<Ctx=C, Out = ()> + Send + Sync + 'static,
    R: Debug + Send + Sync + 'static,
    C: RunCtx + 'static,
{
    type Ctx=C;

    type In<'a> = ResourcesConfigInput<ResC>
    where
        Self: 'a;
    type Out = CrossResult;
}

// #[async_trait]
impl<ResC, L, CLC, RSV, R, C> AsyncStage for CrossLoader<ResC, L, CLC, RSV, R, C>
where
    C:RunCtx+'static,
    for<'a> Self::In<'a>: Data + WithId + Send + Sync,
    Self::Out: Data + WithId + Send + Sync,
    ResC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + 'static,
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
    CLC: for<'i> SyncStage<In<'i> = CrossInput<'i, &'i Arc<Resource>>, Out = CrossUnfoldClcOutput<R>, Ctx=C, > + Send + Sync + 'static,
    RSV: for<'i> SyncStage<In<'i> = CrossUnfoldClcOutput<R>, Ctx=C> + Stage<Ctx=C, Out = ()> + Send + Sync + 'static,
    R: Debug + Send + Sync + 'static,
    C: RunCtx + 'static,
{
    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        ctx: Arc<parking_lot::RwLock<Self::Ctx>>,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
        // let chunk_size = self.chunk_size;
        let max_batch = i64::try_from(self.chunk_size).unwrap_or(i64::MAX);
        let device_idx = ctx.read().device_idx();

        let resources = input.with_data_ref(0,
            Arc::clone(&ctx), None, |input: &ResourcesConfigInput<ResC>|{
                        Ok(input
                            .resources_config
                            .to_resources(&self.store)?
                            .into_iter()
                            .map(|r| Arc::new(r))
                            .collect::<Vec<_>>())
                    }).await?;

        debug!("CrossLoader::run  max_batch={max_batch} resources={}",resources.len());

        let mut t_0_batch: Vec<_> = vec![];
        let mut t_0_batch_resources: Vec<_> = vec![];
        let mut t_0_batch_len = 0;

        let mut skip_0_i = 0;

        for (i_0, resource_0) in resources.iter().enumerate() {
            info!("on i_0={i_0} resource_0={resource_0:?}");

            let t_0 = Arc::clone(&self.loader)
                .run_cached(Arc::new(DevicesWrap::wrap_cpu(resource_0.as_ref().clone())), Arc::clone(&ctx))
                .await?;

            t_0_batch_len += t_0.size().dim_1();
            t_0_batch.push(t_0);
            t_0_batch_resources.push(resource_0);

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
                    .run_cached(Arc::new(DevicesWrap::wrap_cpu(resource_1.as_ref().clone())), Arc::clone(&ctx))
                    .await?;

                t_1_batch_len += t_1.size().dim_1();

                t_1_batch.push(t_1);
                t_1_batch_resources.push(resource_1);

                if t_1_batch_len < max_batch && i_1 < resources.len() - 1 {
                    continue;
                }


                let mut cross_clc_output = DevicesWrap::with_data_array(//TODO replace with with_data!
                    [t_0_batch.clone(), t_1_batch].concat().as_slice(), //t_0_batch and t_1_batch are DataCacheWrappers
                    device_idx,
                    None,
                    |data| {

                        let input = CrossInput::new(
                            data,
                            t_0_batch_resources.as_slice(),
                            t_1_batch_resources.as_slice(),
                        );

                        debug!("input={input:?}");

                        let mut out = Arc::clone(&self.batch_clc).do_run(&input, Arc::clone(&ctx))?;

                        Ok(out)

                    },
                )
                .await?;

                let cross_clc_output = cross_clc_output.replace_clc_result(&self.on_batch_done).await;

                let receiver_arc = Arc::clone(&self.receiver);
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

        (self.on_all_done)(&self.receiver).await;

        Ok(Arc::new(DevicesWrap::wrap_cpu(CrossResult::new(self.out_id(WithId::id(input.as_ref()))))))
    }
}


#[cfg(test)]
pub(crate) mod mock_cross_loader_clc {
    use super::RunCtx2;
    use caching_runner::common::runnable::{Stage, SyncStage};
    use caching_runner::common::RunCtx;
    use crate::tasks::cross_batch_clc::{CrossInput, CrossUnfoldClcOutput};
    use parking_lot::RwLock;
    use std::fmt::Debug;
    use std::marker::PhantomData;
    use std::sync::Arc;
    use thoosa_shared::ThoosaError;
    use thoosa_shared::applicables_runner::{Input, Output};
    use thoosa_shared::resource::Resource;

    type RunBatchFn<Out, C> = dyn for<'a> Fn(
        &CrossInput<'a, &'a Arc<Resource>>,
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

    impl<Out, C> Stage for MockCrossLoaderClc<Out, C>
    where
        Out: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        type Ctx = C;
        type In<'a> = CrossInput<'a, &'a Arc<Resource>>;
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
                &CrossInput<'a, &'a Arc<Resource>>,
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
    use super::RunCtx2;
    use caching_runner::common::runnable::{Stage, SyncStage};
    use caching_runner::common::RunCtx;
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
    use crate::tasks::tensor_loader::{mock_tensor_loader};
    use super::*;
    use crate::cli::Commands::FindMotifs;
    use crate::tasks::cross_batch_clc::{CrossUnfoldClc, CrossUnfoldClcOutput, UnfoldParams};
    use crate::tasks::cross_loader::{CrossLoader, ResourcesConfigInput};
    use crate::tasks::dists_tracker::{DistsMask, DistsTracker};
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::RawTensorLoader;
    use crate::tasks::slices_iter::SlicesIter;
    use caching_runner::common::{mock_run_ctx, MockRunCtx, RunCtx};
    use caching_runner::global_ctx::GlobalCtx;
    use futures_util::FutureExt;
    use log::debug;
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, Once};
    use std::time::Duration;
    use futures_util::sink::With;
    use tch::{Device, Tensor};
    use thoosa_common::applicables::ThoDevice;
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::ws::FsWorkspace;
    use thoosa_motifs::distance::MaskMaker::Tril;
    use thoosa_motifs::find::FixedFindMotifsConfig;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::{cdt_for_dir, ResourceType, ThoosaError};
    use thoosa_store::{MockStore, MockStoreBuilder, SqlStore};
    use tokio::task::yield_now;
    use tokio::time::sleep;
    use thoosa_common::tho_tensor::ThoTensor;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Observation {
        tensor_ids: Vec<&'static str>,
        batch_0_ids: Vec<u32>,
        batch_1_ids: Vec<u32>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestBatchOutput {
        id: SimpleId,
        call_index: usize,
    }

    impl TestBatchOutput {
        fn new(call_index: usize) -> Self {
            Self {
                id: SimpleId::from("batch_output"),
                call_index,
            }
        }
    }

    impl WithId for TestBatchOutput {
        type ID = SimpleId;

        fn id(&self) -> &Self::ID {
            &self.id
        }
    }

    impl Input for TestBatchOutput {}
    impl Output for TestBatchOutput {}

    #[tokio::test]
    async fn test_cross_clc() -> Result<(), ThoosaError> {
        GlobalCtx::init_devices(2);

        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        fs::create_dir_all(&test_dir)?;
        let ws = Arc::new(RwLock::new(FsWorkspace::new(test_dir, 1, 1000, 32000)?));
        let mut store = Arc::new(RwLock::new(SqlStore::from_path(ws.read().store_path())?));

        let run_config = FindMotifsRunConfig::new(Some(0), 1, 2, None, None);
        let fixed_config = FixedFindMotifsConfig::new(None, 2, false, None);

        let resource_keys: Vec<String> = vec!["111", "222", "333", "444", "555", "666"]
            .into_iter()
            .map(|v| v.to_string())
            .collect();
        let resources: Vec<Resource> = resource_keys
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, 1u32, v.as_str(), 10).into())
            .collect();
        let resources_to_move = resources.clone();

        let ctx = Arc::new(parking_lot::RwLock::new(RunCtx2::new(0, 0, "aa")));

        let loader = Arc::new(RawTensorLoader::<RunCtx2>::new(
            move |key, _| {
                let i = resources_to_move
                    .iter()
                    .position(|v| v.eq(key))
                    .expect(format!("no {key:?}").as_str())
                    + 1;
                let v = (0..10)
                    .map(|v| i as f32 + v as f32 / 10.)
                    .collect::<Vec<f32>>();
                debug!("executor: key={key:?} i={i} v={v:?}");

                ThoTensor::new(
                    Input::id(key).clone(),
                    Tensor::from_slice(v.as_slice()).view((1, 10)),
                )
            },
            ThoDevice::from(Device::Cpu),
        ));

        let batch_clc = Arc::new(CrossUnfoldClc::new(
            |t_0, t_1, input, ctx, _| {
                let mut mask_makers = vec![];
                // if is_diag {
                //     mask_makers.push(Tril);
                // }
                let dists = thoosa_motifs::distance::clc_dist_d2_and_mask(
                    t_0,
                    t_1,
                    true,
                    &mut mask_makers,
                )?;

                debug!("CrossBatchClc::do_run:  dists={dists:?}");
                Ok(DistsMask::new(dists.0, dists.1, true, None))
            },
            (),
            UnfoldParams::new(1, 2, 1),
        ));

        let receiver = Arc::new(DistsTracker::<RunCtx2>::new(
            &ws,
            &store,
            &run_config,
            &fixed_config,
            None,
        ));

        let cross_loader = Arc::new(CrossLoader::new(
            loader,
            &store,
            batch_clc,
            receiver,
            20,
            |batch|{
                async {batch}.boxed()
            },
            |p| {
                async {
                    // cross_loader.receiver;
                }
                .boxed()
            },
        ));

        let cache_area = ctx.read().cache_area();
        let input = Arc::new(DevicesWrap::wrap(
            ResourcesConfigInput::new(
                "".into(),
                Simple::new(ResourceType::List(resource_keys)),
            ),
            cache_area,
        ));

        let _ = cross_loader
            .run(input, Arc::clone(&ctx))
            .await?;

        yield_now().await;
        sleep(Duration::from_millis(10)).await;

        //TODO
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

        let observations = Arc::new(Mutex::new(Vec::<Observation>::new()));
        let produced_outputs = Arc::new(Mutex::new(Vec::<TestBatchOutput>::new()));
        let counter = Arc::new(AtomicUsize::new(0));

        let batch_observations = Arc::clone(&observations);
        let batch_outputs = Arc::clone(&produced_outputs);
        let batch_counter = Arc::clone(&counter);

        let batch_clc = Arc::new(
            mock_cross_loader_clc::build::<TestBatchOutput, MockRunCtx>()
                .with_run_batch_fn(move |input, _ctx| {
                    let observation = Observation {
                        tensor_ids: input.tensors().iter().map(|t| WithId::id(t).0).collect(),
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

        let receiver = Arc::new(
            mock_cross_loader_receiver::build::<CrossUnfoldClcOutput<TestBatchOutput>, MockRunCtx>()
                .with_receive_fn(move |out, _ctx| {
                    let result = out.clc_result();
                    receiver_observed.lock().unwrap().push(result);
                    Ok(())
                })
                .finish(),
        );

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
            receiver,
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
        ));

        let cache_area = ctx.read().cache_area();
        let input = Arc::new(DevicesWrap::wrap(
            ResourcesConfigInput::new(
                SimpleId::from("resources"),
                Simple::new(ResourceType::List(vec![
                    "resource_a".to_string(),
                    "resource_b".to_string(),
                ])),
            ),
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
                tensor_ids: vec!["resource_a", "resource_a"],
                batch_0_ids: vec![resources[0].id()],
                batch_1_ids: vec![resources[0].id()],
            },
            Observation {
                tensor_ids: vec!["resource_a", "resource_b"],
                batch_0_ids: vec![resources[0].id()],
                batch_1_ids: vec![resources[1].id()],
            },
            Observation {
                tensor_ids: vec!["resource_b", "resource_a"],
                batch_0_ids: vec![resources[1].id()],
                batch_1_ids: vec![resources[0].id()],
            },
            Observation {
                tensor_ids: vec!["resource_b", "resource_b"],
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
