use crate::tasks::cross_batch_clc::CrossInput;
#[cfg(test)]
use crate::tasks::cross_batch_clc::CrossUnfoldClc;
use crate::tasks::cross_dims::{ChannelsData2};
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::{RawTensorLoader, TensorLoader};
use async_trait::async_trait;
use caching_runner::common::cachable::{Cached, DataCacheWrapper};
use caching_runner::common::runnable::Stage;
use caching_runner::common::{Data, OutId, RunCtx};
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
use thoosa_shared::CopyToDevice;
use thoosa_shared::ThoosaError;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::resource::{Resource, ResourceStore, ResourcesConfig};
use thoosa_shared::traits::{Named, SimpleId, WithId};
use thoosa_store::Store;
use torch_runner::data::ThoTensor2;
use thoosa_shared::traits::MonoMetrics;

/// Small adapter trait that hides the `Stage` GATs so `CrossLoader` can accept
/// any batched CLC stage while storing it behind an `Arc`.  Without this extra
/// trait the HRTB requirements on `Stage::In<'_>` make the `CrossLoader` type
/// ergonomics (and clone of the stage) far more tedious.
pub trait CrossLoaderClc: Send + Sync + std::fmt::Debug {
    type Out: Input + Output + Send + Debug + 'static;

    fn run_batch<'s>(
        self: Arc<Self>,
        input: &CrossInput<'s, &'s Arc<Resource>>,
        ctx: Arc<parking_lot::RwLock<RunCtx2>>,
    ) -> Result<Self::Out, ThoosaError>
    where
        Self: 's;
}

impl<T> CrossLoaderClc for T
where
    T: for<'s> Stage<In<'s> = CrossInput<'s, &'s Arc<Resource>>, Ctx = RunCtx2> + Send + Sync + 'static,
    T::Out: Input + Output + Send + Debug + 'static,
{
    type Out = T::Out;

    fn run_batch<'s>(
        self: Arc<Self>,
        input: &CrossInput<'s, &'s Arc<Resource>>,
        ctx: Arc<parking_lot::RwLock<RunCtx2>>,
    ) -> Result<Self::Out, ThoosaError>
    where
        Self: 's,
    {
        <Self as Stage>::do_run(self, input, ctx)
    }
}

#[cfg(test)]
pub(crate) mod mock_cross_loader_clc {
    use super::{CrossInput, CrossLoaderClc, RunCtx2};
    use parking_lot::RwLock;
    use std::fmt::Debug;
    use std::sync::Arc;
    use thoosa_shared::ThoosaError;
    use thoosa_shared::applicables_runner::{Input, Output};
    use thoosa_shared::resource::Resource;

    type RunBatchFn<Out> = dyn for<'a> Fn(
            &CrossInput<'a, &'a Arc<Resource>>,
            Arc<RwLock<RunCtx2>>,
        ) -> Result<Out, ThoosaError>
        + Send
        + Sync;

    #[derive(Clone)]
    pub struct MockCrossLoaderClc<Out>
    where
        Out: Input + Output + Send + Debug + 'static,
    {
        run_batch_fn: Arc<RunBatchFn<Out>>,
    }

    impl<Out> std::fmt::Debug for MockCrossLoaderClc<Out>
    where
        Out: Input + Output + Send + Debug + 'static,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockCrossLoaderClc").finish()
        }
    }

    impl<Out> CrossLoaderClc for MockCrossLoaderClc<Out>
    where
        Out: Input + Output + Send + Debug + 'static,
    {
        type Out = Out;

        fn run_batch<'s>(
            self: Arc<Self>,
            input: &CrossInput<'s, &'s Arc<Resource>>,
            ctx: Arc<RwLock<RunCtx2>>,
        ) -> Result<Self::Out, ThoosaError>
        where
            Self: 's,
        {
            let f = Arc::clone(&self.run_batch_fn);
            f(input, ctx)
        }
    }

    pub struct MockCrossLoaderClcBuilder<Out>
    where
        Out: Input + Output + Send + Debug + 'static,
    {
        run_batch_fn: Option<Arc<RunBatchFn<Out>>>,
    }

    impl<Out> MockCrossLoaderClcBuilder<Out>
    where
        Out: Input + Output + Send + Debug + 'static,
    {
        pub fn with_run_batch_fn<F>(mut self, f: F) -> Self
        where
            F: for<'a> Fn(
                    &CrossInput<'a, &'a Arc<Resource>>,
                    Arc<RwLock<RunCtx2>>,
                ) -> Result<Out, ThoosaError>
                + Send
                + Sync
                + 'static,
        {
            self.run_batch_fn = Some(Arc::new(f));
            self
        }

        pub fn finish(self) -> MockCrossLoaderClc<Out> {
            let run_batch_fn = self
                .run_batch_fn
                .expect("MockCrossLoaderClcBuilder requires run_batch_fn to be set");
            MockCrossLoaderClc { run_batch_fn }
        }
    }

    pub fn build<Out>() -> MockCrossLoaderClcBuilder<Out>
    where
        Out: Input + Output + Send + Debug + 'static,
    {
        MockCrossLoaderClcBuilder { run_batch_fn: None }
    }
}

pub trait CrossLoaderReceiver<In>: Send + Sync + Debug
where
    In: Input + Output + Send + Debug + 'static,
{
    fn receive(
        self: Arc<Self>,
        out: &In,
        ctx: Arc<parking_lot::RwLock<RunCtx2>>,
    ) -> Result<(), ThoosaError>;
}

impl<T, In> CrossLoaderReceiver<In> for T
//TODO added for mocks, should be for DistsTracker only
where
    T: for<'a> Stage<In<'a> = In, Ctx = RunCtx2> + Send + Sync + Debug + 'static,
    In: Input + Output + Send + Debug + 'static,
{
    fn receive(
        self: Arc<Self>,
        out: &In,
        ctx: Arc<parking_lot::RwLock<RunCtx2>>,
    ) -> Result<(), ThoosaError> {
        <Self as Stage>::do_run(self, out, ctx).map(|_| ())
    }
}

#[cfg(test)]
pub(crate) mod mock_cross_loader_receiver {
    use super::{CrossLoaderReceiver, RunCtx2};
    use parking_lot::RwLock;
    use std::fmt::Debug;
    use std::sync::Arc;
    use thoosa_shared::ThoosaError;
    use thoosa_shared::applicables_runner::{Input, Output};

    type ReceiveFn<In> = dyn Fn(&In, Arc<RwLock<RunCtx2>>) -> Result<(), ThoosaError> + Send + Sync;

    #[derive(Clone)]
    pub struct MockCrossLoaderReceiver<In>
    where
        In: Input + Output + Send + Debug + 'static,
    {
        receive_fn: Arc<ReceiveFn<In>>,
    }

    impl<In> std::fmt::Debug for MockCrossLoaderReceiver<In>
    where
        In: Input + Output + Send + Debug + 'static,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockCrossLoaderReceiver").finish()
        }
    }

    impl<In> CrossLoaderReceiver<In> for MockCrossLoaderReceiver<In>
    where
        In: Input + Output + Send + Debug + 'static,
    {
        fn receive(
            self: Arc<Self>,
            out: &In,
            ctx: Arc<RwLock<RunCtx2>>,
        ) -> Result<(), ThoosaError> {
            let f = Arc::clone(&self.receive_fn);
            f(out, ctx)
        }
    }

    pub struct MockCrossLoaderReceiverBuilder<In>
    where
        In: Input + Output + Send + Debug + 'static,
    {
        receive_fn: Option<Arc<ReceiveFn<In>>>,
    }

    impl<In> MockCrossLoaderReceiverBuilder<In>
    where
        In: Input + Output + Send + Debug + 'static,
    {
        pub fn with_receive_fn<F>(mut self, f: F) -> Self
        where
            F: Fn(&In, Arc<RwLock<RunCtx2>>) -> Result<(), ThoosaError> + Send + Sync + 'static,
        {
            self.receive_fn = Some(Arc::new(f));
            self
        }

        pub fn finish(self) -> MockCrossLoaderReceiver<In> {
            let receive_fn = self
                .receive_fn
                .expect("MockCrossLoaderReceiverBuilder requires receive_fn to be set");
            MockCrossLoaderReceiver { receive_fn }
        }
    }

    pub fn build<In>() -> MockCrossLoaderReceiverBuilder<In>
    where
        In: Input + Output + Send + Debug + 'static,
    {
        MockCrossLoaderReceiverBuilder { receive_fn: None }
    }
}

// #[derive(Debug, cached_derive::Cached)]
// #[cache_name = "LoadedTensorId"]
pub struct CrossLoader<ResC, L, CrossCLC, RSV>
where
    ResC: ResourcesConfig<R = Vec<Resource>> + Debug,
    L: TensorLoader + 'static,
    CrossCLC: CrossLoaderClc + 'static,
    CrossCLC::Out: Input + Output + Send + Debug + 'static,
    RSV: CrossLoaderReceiver<CrossCLC::Out> + 'static,
{
    loader: Arc<L>,
    store: Arc<RwLock<Box<dyn Store>>>,
    batch_clc: Arc<CrossCLC>,
    receiver: Arc<RSV>,
    chunk_size: usize,
    on_all_done: Box<dyn Fn(&Arc<RSV>) -> BoxFuture<'static, ()> + Sync + Send>,
    pd: PhantomData<ResC>,
}

impl<ResC, L, CrossCLC, RSV> Debug for CrossLoader<ResC, L, CrossCLC, RSV>
where
    ResC: ResourcesConfig<R = Vec<Resource>> + Debug,
    L: TensorLoader + 'static,
    CrossCLC: CrossLoaderClc + 'static,
    CrossCLC::Out: Input + Output + Send + Debug + 'static,
    RSV: CrossLoaderReceiver<CrossCLC::Out> + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossLoader").finish()
    }
}

impl<ResC, L, CrossCLC, RSV> CrossLoader<ResC, L, CrossCLC, RSV>
where
    ResC: ResourcesConfig<R = Vec<Resource>> + Debug,
    L: TensorLoader + 'static,
    CrossCLC: CrossLoaderClc + 'static,
    CrossCLC::Out: Input + Output + Send + Debug + 'static,
    RSV: CrossLoaderReceiver<CrossCLC::Out> + 'static,
{
    pub fn new<OnAllDone>(
        loader: Arc<L>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        batch_clc: Arc<CrossCLC>,
        receiver: Arc<RSV>,
        chunk_size: usize,
        on_all_done: OnAllDone,
    ) -> Self
    where
        OnAllDone: Fn(&Arc<RSV>) -> BoxFuture<'static, ()> + Sync + Send + 'static,
    {
        Self {
            loader,
            store: Arc::clone(store),
            batch_clc,
            receiver,
            chunk_size,
            on_all_done: Box::new(on_all_done),
            pd: Default::default(),
        }
    }
}

impl<ResC, L, CrossCLC, RSV> OutId for CrossLoader<ResC, L, CrossCLC, RSV>
where
    ResC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + 'static,
    L: TensorLoader + 'static,
    CrossCLC: CrossLoaderClc + 'static,
    CrossCLC::Out: Input + Output + Send + Debug + 'static,
    RSV: CrossLoaderReceiver<CrossCLC::Out> + 'static,
{
    fn out_id(&self, input_id: &<Self::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        SimpleId::from(format!("cross_{}", input_id))
    }
}

#[derive(Debug)]
pub struct ResourcesConfigInput<RC: ResourcesConfig> {
    id: SimpleId,
    resources_config: RC,
}

impl<RC: ResourcesConfig + Debug> ResourcesConfigInput<RC> {
    pub fn new(id: SimpleId, resources: RC) -> Self {
        Self {
            id,
            resources_config: resources, //: resources.to_resources().into_iter().map(|r| Arc::new(r)).collect(),
        }
    }
}

impl<ResC: ResourcesConfig> WithId for ResourcesConfigInput<ResC> {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<RC: ResourcesConfig + Debug> Input for ResourcesConfigInput<RC> {}

#[derive(Debug)]
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

#[async_trait]
impl<ResC, L, CrossCLC, RSV> Stage for CrossLoader<ResC, L, CrossCLC, RSV>
where
    ResC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + 'static,
    L: TensorLoader + 'static,
    CrossCLC: CrossLoaderClc + 'static,
    CrossCLC::Out: Input + Output + Send + Debug + 'static,
    RSV: CrossLoaderReceiver<CrossCLC::Out> + 'static,
{
    type In<'a>
        = ResourcesConfigInput<ResC>
    where
        Self: 'a;
    type Out = CrossResult;
    type Ctx = RunCtx2;

    async fn run(
        self: Arc<Self>,
        input: Arc<Self::In<'static>>,
        ctx: Arc<parking_lot::RwLock<Self::Ctx>>,
    ) -> Result<Self::Out, ThoosaError> {
        // let chunk_size = self.chunk_size;
        let max_batch = i64::try_from(self.chunk_size).unwrap_or(i64::MAX);
        let device_idx = ctx.read().device_idx();

        let resources = input
            .resources_config
            .to_resources(&self.store)?
            .into_iter()
            .map(|r| Arc::new(r))
            .collect::<Vec<_>>();
        debug!(
            "CrossLoader::run  max_batch={max_batch} resources={}",
            resources.len()
        );

        let mut t_0_batch: Vec<_> = vec![];
        let mut t_0_batch_resources: Vec<_> = vec![];
        let mut t_0_batch_len = 0;

        let mut skip_0_i = 0;

        for (i_0, resource_0) in resources.iter().enumerate() {
            info!("on resource_0={resource_0:?}");

            let t_0 = Arc::clone(&self.loader)
                .load(Arc::clone(resource_0), Arc::clone(&ctx))
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

            for (i_1, resource_1) in resources.iter().enumerate().skip(skip_0_i) {
                //skipping lower triangle

                let t_1 = Arc::clone(&self.loader)
                    .load(Arc::clone(resource_1), Arc::clone(&ctx))
                    .await?;

                t_1_batch_len += t_1.size().dim_1();

                t_1_batch.push(t_1);
                t_1_batch_resources.push(resource_1);

                if t_1_batch_len < max_batch && i_1 < resources.len() - 1 {
                    continue;
                }

                let _ = DataCacheWrapper::with_data_array(
                    [t_0_batch.clone(), t_1_batch].concat().as_slice(), //t_0_batch and t_1_batch are DataCacheWrappers
                    device_idx,
                    None,
                    |data| {
                        //t_0_batch + t_1_batch - loaded to device_idx

                        let input = CrossInput::new(
                            data,
                            t_0_batch_resources.as_slice(),
                            t_1_batch_resources.as_slice(),
                        );

                        debug!("input={input:?}");

                        let out = Arc::clone(&self.batch_clc).run_batch(&input, Arc::clone(&ctx))?;

                        let receiver_arc = Arc::clone(&self.receiver);
                        let ctx_arc = Arc::clone(&ctx);
                        tokio::task::block_in_place(|| {
                            receiver_arc.receive(&out, ctx_arc)
                        })?;

                        Ok(())
                    },
                )
                .await?;

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

        Ok(CrossResult::new(self.out_id(WithId::id(input.as_ref()))))
    }
}

#[cfg(test)]
mod tests {
    use super::mock_cross_loader_clc;
    use super::mock_cross_loader_receiver;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use super::*;
    use crate::cli::Commands::FindMotifs;
    use crate::tasks::cross_batch_clc::{CrossUnfoldClc, UnfoldParams};
    use crate::tasks::cross_loader::{CrossLoader, ResourcesConfigInput};
    use crate::tasks::dists_tracker::{DistWithMask, DistsTracker};
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::RawTensorLoader;
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
    use torch_runner::data::ThoTensor2;
    use crate::tasks::tensor_loader::mock_tensor_loader::make_wrapper;

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

        let loader = Arc::new(RawTensorLoader::new(
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

                ThoTensor2::new(
                    Input::id(key).clone(),
                    Tensor::from_slice(v.as_slice()).view((1, 10)),
                )
            },
            ThoDevice::from(Device::Cpu),
        ));

        let batch_clc = Arc::new(CrossUnfoldClc::new(
            |t_0, t_1, input, ctx| {
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
                Ok(DistWithMask::new(dists.0, dists.1))
            },
            (),
            UnfoldParams::new(1, 2, 1),
        ));

        let receiver = Arc::new(DistsTracker::new(
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
            |p| {
                async {
                    // cross_loader.receiver;
                }
                .boxed()
            },
        ));

        let _ = cross_loader
            .run(
                Arc::new(ResourcesConfigInput::new(
                    "".into(),
                    Simple::new(ResourceType::List(resource_keys)),
                )),
                ctx,
            )
            .await?;

        yield_now().await;
        sleep(Duration::from_millis(10)).await;

        Ok(())
    }



    #[tokio::test(flavor = "multi_thread")]
    async fn test_cross_loader_with_mocks() -> Result<(), ThoosaError> {
        let resource_a: Resource = (1, 1, "resource_a", 2).into();
        let resource_b: Resource = (2, 1, "resource_b", 2).into();
        let resources = vec![resource_a.clone(), resource_b.clone()];


        let loader_data: Arc<
            HashMap<u32, Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>>,
        > = Arc::new(HashMap::from([
            (resource_a.id(), make_wrapper("tensor_a", &[1.0, 2.0])),
            (resource_b.id(), make_wrapper("tensor_b", &[3.0, 4.0])),
        ]));

        let loader_map = Arc::clone(&loader_data);
        let loader = Arc::new(
            mock_tensor_loader::new()
                .with_load_fn(move |resource, _ctx| {
                    let map = Arc::clone(&loader_map);
                    async move {
                        Ok(map
                            .get(&resource.id())
                            .cloned()
                            .expect("mock loader data should be present"))
                    }
                })
                .build(),
        );

        let observations = Arc::new(Mutex::new(Vec::<Observation>::new()));
        let produced_outputs = Arc::new(Mutex::new(Vec::<TestBatchOutput>::new()));
        let counter = Arc::new(AtomicUsize::new(0));

        let batch_observations = Arc::clone(&observations);
        let batch_outputs = Arc::clone(&produced_outputs);
        let batch_counter = Arc::clone(&counter);

        let batch_clc = Arc::new(
            mock_cross_loader_clc::build::<TestBatchOutput>()
                .with_run_batch_fn(move |input, _ctx| {
                    let observation = Observation {
                        tensor_ids: input.tensors().iter().map(|t| t.id().0).collect(),
                        batch_0_ids: input.batch_0().iter().map(|r| r.id()).collect(),
                        batch_1_ids: input.batch_1().iter().map(|r| r.id()).collect(),
                    };

                    batch_observations.lock().unwrap().push(observation);

                    let call_index = batch_counter.fetch_add(1, Ordering::SeqCst);
                    let output = TestBatchOutput::new(call_index);
                    batch_outputs.lock().unwrap().push(output.clone());

                    Ok(output)
                })
                .finish(),
        );

        let received_outputs = Arc::new(Mutex::new(Vec::<TestBatchOutput>::new()));
        let receiver_observed = Arc::clone(&received_outputs);

        let receiver = Arc::new(
            mock_cross_loader_receiver::build::<TestBatchOutput>()
                .with_receive_fn(move |out, _ctx| {
                    receiver_observed.lock().unwrap().push(out.clone());
                    Ok(())
                })
                .finish(),
        );

        let ctx = Arc::new(RwLock::new(RunCtx2::new(0, 0, "mock")));

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
            |p| {
                async {
                    // cross_loader.receiver;
                }
                .boxed()
            },
        ));

        let _ = cross_loader
            .run(
                Arc::new(ResourcesConfigInput::new(
                    SimpleId::from("resources"),
                    Simple::new(ResourceType::List(vec![
                        "resource_a".to_string(),
                        "resource_b".to_string(),
                    ])),
                )),
                Arc::clone(&ctx),
            )
            .await?;

        yield_now().await;
        sleep(Duration::from_millis(10)).await;

        let observed = observations.lock().unwrap().clone();

        let expected_observations = vec![
            Observation {
                tensor_ids: vec!["tensor_a", "tensor_a"],
                batch_0_ids: vec![resources[0].id()],
                batch_1_ids: vec![resources[0].id()],
            },
            Observation {
                tensor_ids: vec!["tensor_a", "tensor_b"],
                batch_0_ids: vec![resources[0].id()],
                batch_1_ids: vec![resources[1].id()],
            },
            Observation {
                tensor_ids: vec!["tensor_b", "tensor_a"],
                batch_0_ids: vec![resources[1].id()],
                batch_1_ids: vec![resources[0].id()],
            },
            Observation {
                tensor_ids: vec!["tensor_b", "tensor_b"],
                batch_0_ids: vec![resources[1].id()],
                batch_1_ids: vec![resources[1].id()],
            },
        ];

        assert_eq!(observed, expected_observations);

        let produced = produced_outputs.lock().unwrap().clone();
        let received = received_outputs.lock().unwrap().clone();

        let expected_outputs: Vec<TestBatchOutput> = (0..expected_observations.len())
            .map(TestBatchOutput::new)
            .collect();

        assert_eq!(produced, expected_outputs);
        assert_eq!(produced, received);

        Ok(())
    }
}
