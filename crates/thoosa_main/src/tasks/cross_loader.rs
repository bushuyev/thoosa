use crate::tasks::cross_batch_clc::BatchInput;
#[cfg(test)]
use crate::tasks::cross_batch_clc::CrossBatchClc;
use crate::tasks::cross_dims::{ChannelsData2, ChannelsSlicesData2};
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::TensorLoader;
use async_trait::async_trait;
use caching_runner::common::cachable::{Cached, DataCacheWrapper};
use caching_runner::common::runnable::Stage;
use caching_runner::common::{Data, OutId, RunCtx};
use log::debug;
use parking_lot::RwLock;
use std::fmt::Debug;
use std::sync::Arc;
use tch::Tensor;
use thoosa_common::dims::{AnyMetrics, AnySub, ChannelsData, ChannelsSlicesData, Dims};
use thoosa_common::tho_tensor::CopyToDevice;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::resource::Resource;
use thoosa_shared::traits::{Named, SimpleId, WithId};
use thoosa_shared::ThoosaError;
use torch_runner::data::ThoTensor2;
use torch_runner::metrics::TwoDimsMetrics;

#[async_trait]
pub trait CrossLoaderTensorSource: Send + Sync + Debug {
    async fn load(
        self: Arc<Self>,
        resource: Arc<Resource>,
        ctx: Arc<parking_lot::RwLock<RunCtx2>>,
    ) -> Result<Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>, ThoosaError>;
}

#[async_trait]
impl CrossLoaderTensorSource for TensorLoader {
    async fn load(
        self: Arc<Self>,
        resource: Arc<Resource>,
        ctx: Arc<parking_lot::RwLock<RunCtx2>>,
    ) -> Result<Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>, ThoosaError> {
        <Self as Cached>::run_cached(self, resource, ctx).await
    }
}

#[cfg(test)]
pub(crate) mod mock_cross_loader_tensor_source {
    use super::ChannelsData2;
    use super::{CrossLoaderTensorSource, DataCacheWrapper, RunCtx2, SimpleId, ThoTensor2};
    use async_trait::async_trait;
    use parking_lot::RwLock;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::ThoosaError;

    type LoadFuture = Pin<
        Box<
            dyn Future<
                    Output = Result<
                        Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>,
                        ThoosaError,
                    >,
                > + Send,
        >,
    >;

    type LoadFn = dyn Fn(Arc<Resource>, Arc<RwLock<RunCtx2>>) -> LoadFuture + Send + Sync;

    #[derive(Clone)]
    pub struct MockCrossLoaderTensorSource {
        load_fn: Arc<LoadFn>,
    }

    impl std::fmt::Debug for MockCrossLoaderTensorSource {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockCrossLoaderTensorSource").finish()
        }
    }

    #[async_trait]
    impl CrossLoaderTensorSource for MockCrossLoaderTensorSource {
        async fn load(
            self: Arc<Self>,
            resource: Arc<Resource>,
            ctx: Arc<RwLock<RunCtx2>>,
        ) -> Result<Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>, ThoosaError>
        {
            (self.load_fn)(resource, ctx).await
        }
    }

    pub struct MockCrossLoaderTensorSourceBuilder {
        load_fn: Option<Arc<LoadFn>>,
    }

    impl MockCrossLoaderTensorSourceBuilder {
        pub fn with_load_fn<F, Fut>(mut self, f: F) -> Self
        where
            F: Fn(Arc<Resource>, Arc<RwLock<RunCtx2>>) -> Fut + Send + Sync + 'static,
            Fut: Future<
                    Output = Result<
                        Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>,
                        ThoosaError,
                    >,
                > + Send
                + 'static,
        {
            self.load_fn = Some(Arc::new(move |resource, ctx| Box::pin(f(resource, ctx))));
            self
        }

        pub fn finish(self) -> MockCrossLoaderTensorSource {
            let load_fn = self
                .load_fn
                .expect("MockCrossLoaderTensorSourceBuilder requires load_fn to be set");
            MockCrossLoaderTensorSource { load_fn }
        }
    }

    pub fn build() -> MockCrossLoaderTensorSourceBuilder {
        MockCrossLoaderTensorSourceBuilder { load_fn: None }
    }
}

/// Small adapter trait that hides the `Stage` GATs so `CrossLoader` can accept
/// any batched CLC stage while storing it behind an `Arc`.  Without this extra
/// trait the HRTB requirements on `Stage::In<'_>` make the `CrossLoader` type
/// ergonomics (and clone of the stage) far more tedious.
pub trait CrossLoaderClc: Send + Sync + std::fmt::Debug {
    type Out: Input + Output + Send + Debug + 'static;

    fn run_batch<'s>(
        self: Arc<Self>,
        input: &BatchInput<'s, &'s Arc<Resource>>,
        ctx: Arc<parking_lot::RwLock<RunCtx2>>,
    ) -> Result<Self::Out, ThoosaError>
    where
        Self: 's;
}

impl<T> CrossLoaderClc for T
where
    T: for<'s> Stage<In<'s> = BatchInput<'s, &'s Arc<Resource>>, Ctx = RunCtx2>
        + Send
        + Sync
        + 'static,
    T::Out: Input + Output + Send + Debug + 'static,
{
    type Out = T::Out;

    fn run_batch<'s>(
        self: Arc<Self>,
        input: &BatchInput<'s, &'s Arc<Resource>>,
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
    use super::{BatchInput, CrossLoaderClc, RunCtx2};
    use parking_lot::RwLock;
    use std::fmt::Debug;
    use std::sync::Arc;
    use thoosa_shared::applicables_runner::{Input, Output};
    use thoosa_shared::resource::Resource;
    use thoosa_shared::ThoosaError;

    type RunBatchFn<Out> = dyn for<'a> Fn(
            &BatchInput<'a, &'a Arc<Resource>>,
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
            input: &BatchInput<'s, &'s Arc<Resource>>,
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
                    &BatchInput<'a, &'a Arc<Resource>>,
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
    use thoosa_shared::applicables_runner::{Input, Output};
    use thoosa_shared::ThoosaError;

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
#[derive(Debug)]
// #[cache_name = "LoadedTensorId"]
pub struct CrossLoader<SRC, CLC, RSV> {
    loader: Arc<SRC>,
    batch_clc: Arc<CLC>,
    receiver: Arc<RSV>,
    chunk_size: usize,
    max_batch: usize,
}

impl<SRC, CLC, RSV> CrossLoader<SRC, CLC, RSV>
where
    SRC: CrossLoaderTensorSource + 'static,
    CLC: CrossLoaderClc + 'static,
    CLC::Out: Input + Output + Send + Debug + 'static,
    RSV: CrossLoaderReceiver<CLC::Out> + 'static,
{
    pub fn new(
        loader: Arc<SRC>,
        batch_clc: Arc<CLC>,
        receiver: Arc<RSV>,
        chunk_size: usize,
        max_batch: usize,
    ) -> Self {
        Self {
            loader,
            batch_clc,
            receiver,
            chunk_size,
            max_batch,
        }
    }
}

impl<SRC, CLC, RSV> OutId for CrossLoader<SRC, CLC, RSV>
where
    SRC: CrossLoaderTensorSource + 'static,
    CLC: CrossLoaderClc + 'static,
    CLC::Out: Input + Output + Send + Debug + 'static,
    RSV: CrossLoaderReceiver<CLC::Out> + 'static,
{
    fn out_id(&self, input_id: &<Self::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        SimpleId::from(format!("cross_{}", input_id))
    }
}

#[derive(Debug)]
pub struct ResourcesVec {
    id: SimpleId,
    resources: Vec<Arc<Resource>>,
}

impl ResourcesVec {
    pub fn new(id: SimpleId, resources: Vec<Resource>) -> Self {
        Self {
            id,
            resources: resources.into_iter().map(|r| Arc::new(r)).collect(),
        }
    }
}

impl WithId for ResourcesVec {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Input for ResourcesVec {}

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
impl<SRC, CLC, RSV> Stage for CrossLoader<SRC, CLC, RSV>
where
    SRC: CrossLoaderTensorSource + 'static,
    CLC: CrossLoaderClc + 'static,
    CLC::Out: Input + Output + Send + Debug + 'static,
    RSV: CrossLoaderReceiver<CLC::Out> + 'static,
{
    type In<'a>
        = ResourcesVec
    where
        Self: 'a;
    type Out = CrossResult;
    type Ctx = RunCtx2;

    async fn run(
        self: Arc<Self>,
        input: Arc<Self::In<'static>>,
        ctx: Arc<parking_lot::RwLock<Self::Ctx>>,
    ) -> Result<Self::Out, ThoosaError> {
        let chunk_size = self.chunk_size;
        let max_batch = i64::try_from(self.max_batch).unwrap_or(i64::MAX);
        let device_idx = ctx.read().device_idx();

        debug!(
            "CrossLoader::run chunk_size={chunk_size}  max_batch={max_batch} resources={}",
            input.resources.len()
        );

        let mut t_0_batch: Vec<_> = vec![];
        let mut t_0_batch_resources: Vec<_> = vec![];
        let mut t_0_batch_len = 0;

        let mut skip_0_i = 0;

        for (i_0, resource_0) in input.resources.iter().enumerate() {
            let t_0 = Arc::clone(&self.loader)
                .load(Arc::clone(resource_0), Arc::clone(&ctx))
                .await?;

            t_0_batch_len += t_0.size().seq_len();
            t_0_batch.push(t_0);
            t_0_batch_resources.push(resource_0);

            if t_0_batch_len < max_batch && i_0 < input.resources.len() - 1 {
                continue;
            }

            let mut t_1_batch: Vec<_> = vec![];
            let mut t_1_batch_resources: Vec<_> = vec![];
            let mut t_1_batch_len = 0;

            for (i_1, resource_1) in input.resources.iter().enumerate().skip(skip_0_i) {
                //skipping lower triangle

                let t_1 = Arc::clone(&self.loader)
                    .load(Arc::clone(resource_1), Arc::clone(&ctx))
                    .await?;

                t_1_batch_len += t_1.size().seq_len();

                t_1_batch.push(t_1);
                t_1_batch_resources.push(resource_1);

                if t_1_batch_len < max_batch && i_1 < input.resources.len() - 1 {
                    continue;
                }

                let _ = DataCacheWrapper::with_data_array(
                    [t_0_batch.clone(), t_1_batch].concat().as_slice(), //t_0_batch and t_1_batch are DataCacheWrappers
                    device_idx,
                    None,
                    |data| {
                        //t_0_batch + t_1_batch - loaded to device_idx

                        let input = BatchInput::new(
                            data,
                            t_0_batch_resources.as_slice(),
                            t_1_batch_resources.as_slice(),
                        );

                        debug!("input={input:?}");

                        let out =
                            Arc::clone(&self.batch_clc).run_batch(&input, Arc::clone(&ctx))?;

                        let receiver_arc = Arc::clone(&self.receiver);
                        let ctx_arc = Arc::clone(&ctx);
                        tokio::task::spawn_blocking(move || {
                            receiver_arc.receive(&out, ctx_arc).unwrap();
                        });

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

        Ok(CrossResult::new(self.out_id(WithId::id(input.as_ref()))))
    }
}

#[cfg(test)]
mod tests {
    use super::mock_cross_loader_clc;
    use super::mock_cross_loader_receiver;
    use super::mock_cross_loader_tensor_source;
    use super::*;
    use crate::tasks::cross_batch_clc::CrossBatchClc;
    use crate::tasks::cross_loader::{CrossLoader, ResourcesVec};
    use crate::tasks::dists_tracker::{DistWithMask, DistsTracker};
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::TensorLoader;
    use caching_runner::global_ctx::GlobalCtx;
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
    use thoosa_motifs::find::FixedFindMotifsConfig;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::{cdt_for_dir, ThoosaError};
    use tokio::task::yield_now;
    use tokio::time::sleep;
    use thoosa_motifs::distance::MaskMaker::Tril;
    use torch_runner::data::ThoTensor2;
    use crate::cli::Commands::FindMotifs;

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

        let run_config = FindMotifsRunConfig::new(Some(0), 1, 2, None, None);
        let fixed_config = FixedFindMotifsConfig::new(None, 2, false, None);

        let resources: Vec<Resource> = vec!["111", "222", "333", "444", "555", "666"]
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i as u32, 1u32, v, 10).into())
            .collect();
        let resources_to_move = resources.clone();

        let ctx = Arc::new(parking_lot::RwLock::new(RunCtx2::new(0, 0, "aa")));

        let loader = Arc::new(TensorLoader::new(
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

        let batch_clc = Arc::new(CrossBatchClc::new(|t_0, t_1, is_diag| {
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
        }));

        let receiver = Arc::new(DistsTracker::new_2(&ws, &run_config, &fixed_config, None));

        let cross_loader = Arc::new(CrossLoader::new(loader, batch_clc, receiver, 2, 20));

        let _ = cross_loader
            .run(Arc::new(ResourcesVec::new("".into(), resources)), ctx)
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

        fn make_wrapper(
            id: &'static str,
            data: &[f32],
        ) -> Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>> {
            let tensor = Tensor::from_slice(data).view((1, data.len() as i64));
            let tho_tensor = ThoTensor2::new(SimpleId::from(id), tensor);
            Arc::new(DataCacheWrapper::new(tho_tensor, 0, 1, 0))
        }

        let loader_data: Arc<
            HashMap<u32, Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>>,
        > = Arc::new(HashMap::from([
            (resource_a.id(), make_wrapper("tensor_a", &[1.0, 2.0])),
            (resource_b.id(), make_wrapper("tensor_b", &[3.0, 4.0])),
        ]));

        let loader_map = Arc::clone(&loader_data);
        let loader = Arc::new(
            mock_cross_loader_tensor_source::build()
                .with_load_fn(move |resource, _ctx| {
                    let map = Arc::clone(&loader_map);
                    async move {
                        Ok(map
                            .get(&resource.id())
                            .cloned()
                            .expect("mock loader data should be present"))
                    }
                })
                .finish(),
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

        let cross_loader = Arc::new(CrossLoader::new(loader, batch_clc, receiver, 1, 2));

        let _ = cross_loader
            .run(
                Arc::new(ResourcesVec::new(
                    SimpleId::from("resources"),
                    resources.clone(),
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
