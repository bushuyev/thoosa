use std::fmt::Debug;
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
use std::sync::Arc;
use tch::Tensor;
use thoosa_common::dims::{AnyMetrics, AnySub, ChannelsData, ChannelsSlicesData, Dims};
use thoosa_common::tho_tensor::CopyToDevice;
use thoosa_shared::ThoosaError;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::resource::Resource;
use thoosa_shared::traits::{Named, SimpleId, WithId};
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
    T: for<'s> Stage<
            In<'s> = BatchInput<'s, &'s Arc<Resource>>,
            Ctx = RunCtx2,
        > + Send + Sync + 'static,
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

// #[derive(Debug, cached_derive::Cached)]
#[derive(Debug)]
// #[cache_name = "LoadedTensorId"]
pub struct CrossLoader<SRC, CLC, RSV> {
    loader: Arc<SRC>,
    batch_clc: Arc<CLC>,
    receiver: Arc<RSV>,
    window: usize,
    stride: usize,
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
        window: usize,
        stride: usize,
        max_batch: usize,
    ) -> Self {
        Self {
            loader,
            batch_clc,
            receiver,
            window,
            stride,
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
    id:SimpleId
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
    type In<'a> = ResourcesVec where Self: 'a;
    type Out = CrossResult;
    type Ctx = RunCtx2;

    async fn run(
        self: Arc<Self>,
        input: Arc<Self::In<'static>>,
        ctx: Arc<parking_lot::RwLock<Self::Ctx>>,
    ) -> Result<Self::Out, ThoosaError> {
        let window = self.window;
        let stride = self.stride;
        let max_batch = i64::try_from(self.max_batch).unwrap_or(i64::MAX);
        let device_idx = ctx.read().device_idx();

        debug!(
            "CrossLoader::run window={window} stride={stride} max_batch={max_batch} resources={}",
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

            for (i_1, resource_1) in input.resources
                .iter()
                .enumerate()
                .skip(skip_0_i) {//skipping lower triangle

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
                    [t_0_batch.clone(), t_1_batch].concat().as_slice(),//t_0_batch and t_1_batch are DataCacheWrappers
                    device_idx,
                    None,
                    |data| {//t_0_batch + t_1_batch - loaded to device_idx

                        let input = BatchInput::new(
                            data,
                            t_0_batch_resources.as_slice(),
                            t_1_batch_resources.as_slice(),
                        );

                        debug!("input={input:?}");

                        let out = Arc::clone(&self.batch_clc).run_batch(&input, Arc::clone(&ctx))?;

                        let receiver_arc = Arc::clone(&self.receiver);
                        let ctx_arc = Arc::clone(&ctx);
                        tokio::task::spawn_blocking(move || {
                            receiver_arc.receive(&out, ctx_arc).unwrap();
                        });

                        Ok(())
                    }
                )
                .await?;

                t_1_batch = vec![];
                t_1_batch_resources = vec![];
                t_1_batch_len = 0;
            }

            skip_0_i = i_0.saturating_add(stride.saturating_sub(1));
            t_0_batch.clear();
            t_0_batch_resources.clear();
            t_0_batch_len = 0;
        }

        Ok(CrossResult::new(self.out_id(WithId::id(input.as_ref()))))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use super::*;
    use crate::tasks::cross_batch_clc::CrossBatchClc;
    use crate::tasks::cross_loader::{CrossLoader, ResourcesVec};
    use crate::tasks::dists_tracker::{DistWithMask, DistsTracker};
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::TensorLoader;
    use caching_runner::global_ctx::GlobalCtx;
    use log::debug;
    use std::sync::{Arc, Once};
    use std::time::Duration;
    use tch::{Device, Tensor};
    use thoosa_common::applicables::ThoDevice;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::{cdt_for_dir, ThoosaError};
    use tokio::task::yield_now;
    use tokio::time::sleep;
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::ws::FsWorkspace;
    use thoosa_motifs::find::FixedFindMotifsConfig;
    use torch_runner::data::ThoTensor2;

    static INIT: Once = Once::new();

    fn init_env() {
        INIT.call_once(|| {
            GlobalCtx::init_devices(2);
            thoosa_shared::test_utils::init_test_logging();
        });
    }

    #[tokio::test]
    async fn test_cross_clc() -> Result<(), ThoosaError> {
        init_env();
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

        let batch_clc = Arc::new(CrossBatchClc::new(|t_0, t_1| {
            let mut mask_makers = vec![];
            let dists = thoosa_motifs::distance::clc_dist_d2_and_mask(
                t_0.shallow_clone(),
                t_1,
                true,
                &mut mask_makers,
            )?;

            debug!("CrossBatchClc::do_run:  dists={dists:?}");
            Ok(DistWithMask::new(dists.0, dists.1))
        }));

        let receiver = Arc::new(DistsTracker::new_2(&ws, &run_config, &fixed_config, None));

        let cross_loader = Arc::new(CrossLoader::new(
            loader,
            batch_clc,
            receiver,
            2,
            1,
            20,
        ));

        let _ = cross_loader
            .run(
                Arc::new(ResourcesVec::new("".into(), resources)),
                ctx,
            )
            .await?;

        yield_now().await;
        sleep(Duration::from_millis(10)).await;

        Ok(())
    }

    #[derive(Clone, Copy)]
    struct CaseParams {
        window: usize,
        stride: usize,
        max_batch: usize,
    }

    async fn run_case(
        name: &str,
        specs: Vec<mock_util::TensorSpec>,
        params: CaseParams,
    ) -> Result<(), ThoosaError> {
        init_env();
        let fixtures = mock_util::build_fixtures(&specs);
        assert!(!fixtures.is_empty(), "fixtures required");

        let (batches, loader_calls) = mock_util::simulate_batches(&fixtures, params);
        let loader = mock_util::mock_loader(&fixtures, loader_calls);
        let (batch_clc, expected_ids) = mock_util::mock_batch_clc(&fixtures, &batches);
        let receiver = mock_util::mock_receiver(&expected_ids);

        let cross_loader = Arc::new(CrossLoader::new(
            loader,
            batch_clc,
            receiver,
            params.window,
            params.stride,
            params.max_batch,
        ));

        let ctx = Arc::new(parking_lot::RwLock::new(RunCtx2::new(0, 0, "mock")));
        let resources: Vec<Resource> = fixtures.iter().map(|f| f.resource.clone()).collect();
        let input_id = SimpleId::from(format!("case_{name}"));

        let _ = cross_loader
            .run(Arc::new(ResourcesVec::new(input_id, resources)), ctx)
            .await?;

        yield_now().await;
        sleep(Duration::from_millis(10)).await;

        Ok(())
    }

    #[tokio::test]
    async fn cross_loader_mock_single_batch() -> Result<(), ThoosaError> {
        run_case(
            "single_batch",
            vec![
                mock_util::TensorSpec { channels: 1, seq_len: 4 },
                mock_util::TensorSpec { channels: 2, seq_len: 5 },
                mock_util::TensorSpec { channels: 3, seq_len: 4 },
            ],
            CaseParams {
                window: 2,
                stride: 1,
                max_batch: 16,
            },
        )
        .await
    }

    #[tokio::test]
    async fn cross_loader_mock_small_batches() -> Result<(), ThoosaError> {
        run_case(
            "small_batches",
            vec![
                mock_util::TensorSpec { channels: 1, seq_len: 4 },
                mock_util::TensorSpec { channels: 2, seq_len: 6 },
                mock_util::TensorSpec { channels: 1, seq_len: 5 },
                mock_util::TensorSpec { channels: 3, seq_len: 4 },
            ],
            CaseParams {
                window: 2,
                stride: 1,
                max_batch: 7,
            },
        )
        .await
    }

    #[tokio::test]
    async fn cross_loader_mock_stride_two() -> Result<(), ThoosaError> {
        run_case(
            "stride_two",
            vec![
                mock_util::TensorSpec { channels: 1, seq_len: 4 },
                mock_util::TensorSpec { channels: 2, seq_len: 5 },
                mock_util::TensorSpec { channels: 2, seq_len: 4 },
                mock_util::TensorSpec { channels: 1, seq_len: 6 },
            ],
            CaseParams {
                window: 3,
                stride: 2,
                max_batch: 9,
            },
        )
        .await
    }

    mod mock_util {
        use super::*;
        use mockall::mock;
        use mockall::Sequence;
        use parking_lot::Mutex;
        use std::collections::HashMap;
        use thoosa_shared::applicables_runner::CachableId;

        #[derive(Clone, Debug)]
        pub struct TensorSpec {
            pub channels: i64,
            pub seq_len: i64,
        }

        #[derive(Clone, Debug)]
        pub struct LoaderFixture {
            pub resource: Resource,
            pub spec: TensorSpec,
        }

        pub fn build_fixtures(specs: &[TensorSpec]) -> Vec<LoaderFixture> {
            specs
                .iter()
                .enumerate()
                .map(|(idx, spec)| LoaderFixture {
                    resource: Resource::new(
                        idx as u32,
                        0,
                        format!("res_{idx}"),
                        spec.seq_len as u32,
                    ),
                    spec: spec.clone(),
                })
                .collect()
        }

        #[derive(Clone, Debug)]
        pub struct BatchCall {
            pub left: Vec<usize>,
            pub right: Vec<usize>,
        }

        pub fn simulate_batches(
            fixtures: &[LoaderFixture],
            params: super::CaseParams,
        ) -> (Vec<BatchCall>, usize) {
            assert!(params.max_batch > 0, "max_batch must be positive");
            let stride = params.stride.max(1);
            let mut batches = Vec::new();
            let mut loader_calls = 0usize;
            let mut t0_indices = Vec::new();
            let mut t0_len = 0usize;
            let mut skip_0_i = 0usize;

            for i0 in 0..fixtures.len() {
                loader_calls += 1;
                t0_indices.push(i0);
                t0_len += fixtures[i0].spec.seq_len as usize;
                let is_last_outer = i0 + 1 == fixtures.len();
                if t0_len < params.max_batch && !is_last_outer {
                    continue;
                }

                let mut t1_indices = Vec::new();
                let mut t1_len = 0usize;

                for i1 in skip_0_i..fixtures.len() {
                    loader_calls += 1;
                    t1_indices.push(i1);
                    t1_len += fixtures[i1].spec.seq_len as usize;
                    let is_last_inner = i1 + 1 == fixtures.len();
                    if t1_len < params.max_batch && !is_last_inner {
                        continue;
                    }

                    batches.push(BatchCall {
                        left: t0_indices.clone(),
                        right: t1_indices.clone(),
                    });

                    t1_indices.clear();
                    t1_len = 0;
                }

                skip_0_i = i0.saturating_add(stride.saturating_sub(1));
                t0_indices.clear();
                t0_len = 0;
            }

            (batches, loader_calls)
        }

        mock! {
            pub TensorLoaderCore {
                pub fn load(
                    &self,
                    resource: Arc<Resource>,
                ) -> Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>;
            }
        }

        #[derive(Clone)]
        pub struct TensorLoaderMock {
            core: Arc<Mutex<MockTensorLoaderCore>>,
        }

        impl Debug for TensorLoaderMock {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("TensorLoaderMock").finish()
            }
        }

        #[async_trait]
        impl CrossLoaderTensorSource for TensorLoaderMock {
            async fn load(
                self: Arc<Self>,
                resource: Arc<Resource>,
                _ctx: Arc<parking_lot::RwLock<RunCtx2>>,
            ) -> Result<Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>, ThoosaError> {
                Ok(self.core.lock().load(resource))
            }
        }

        pub fn mock_loader(
            fixtures: &[LoaderFixture],
            loader_calls: usize,
        ) -> Arc<TensorLoaderMock> {
            let device_count = GlobalCtx::global().device_count();
            let mut wrappers: HashMap<u32, Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>> =
                HashMap::new();

            for (idx, fixture) in fixtures.iter().enumerate() {
                let tensor = build_tensor(idx, &fixture.spec);
                let id = SimpleId::from(format!("tensor_{idx}"));
                let tho_tensor = ThoTensor2::new(id, tensor);
                wrappers.insert(
                    fixture.resource.id(),
                    Arc::new(DataCacheWrapper::new(tho_tensor, 0, device_count, 0)),
                );
            }

            let wrappers_arc = Arc::new(wrappers);
            let mut core = MockTensorLoaderCore::new();
            let map = Arc::clone(&wrappers_arc);
            core.expect_load()
                .times(loader_calls)
                .returning(move |resource: Arc<Resource>| {
                    let id = resource.id();
                    Arc::clone(
                        map.get(&id)
                            .expect("expected resource in mock loader"),
                    )
                });

            Arc::new(TensorLoaderMock {
                core: Arc::new(Mutex::new(core)),
            })
        }

        mock! {
            pub BatchClcCore {
                pub fn run_batch<'s>(
                    &self,
                    input: &BatchInput<'s, &'s Arc<Resource>>,
                ) -> DummyBatchOutput;
            }
        }

        #[derive(Clone)]
        pub struct BatchClcMock {
            core: Arc<Mutex<MockBatchClcCore>>,
        }

        impl Debug for BatchClcMock {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("BatchClcMock").finish()
            }
        }

        impl CrossLoaderClc for BatchClcMock {
            type Out = DummyBatchOutput;

            fn run_batch<'s>(
                self: Arc<Self>,
                input: &BatchInput<'s, &'s Arc<Resource>>,
                _ctx: Arc<parking_lot::RwLock<RunCtx2>>,
            ) -> Result<Self::Out, ThoosaError>
            where
                Self: 's,
            {
                Ok(self.core.lock().run_batch(input))
            }
        }

        pub fn mock_batch_clc(
            fixtures: &[LoaderFixture],
            batches: &[BatchCall],
        ) -> (Arc<BatchClcMock>, Vec<&'static str>) {
            let fixtures_arc = Arc::new(fixtures.to_vec());
            let mut core = MockBatchClcCore::new();
            let mut seq = Sequence::new();
            let mut output_ids = Vec::new();

            for (idx, batch) in batches.iter().enumerate() {
                let expected_left = batch.left.clone();
                let expected_right = batch.right.clone();
                let fixtures_ref = Arc::clone(&fixtures_arc);
                let output_id: &'static str = Box::leak(format!("batch_{idx}").into_boxed_str());
                output_ids.push(output_id);

                core.expect_run_batch()
                    .times(1)
                    .in_sequence(&mut seq)
                    .returning(move |input: &BatchInput<'_, &'_ Arc<Resource>>| {
                        assert_batch_input(fixtures_ref.as_ref(), input, &expected_left, &expected_right);
                        DummyBatchOutput::new(output_id)
                    });
            }

            let mock = BatchClcMock {
                core: Arc::new(Mutex::new(core)),
            };

            (Arc::new(mock), output_ids)
        }

        mock! {
            pub ReceiverCore {
                pub fn receive(
                    &self,
                    out: &DummyBatchOutput,
                ) -> Result<(), ThoosaError>;
            }
        }

        #[derive(Clone)]
        pub struct ReceiverMock {
            core: Arc<Mutex<MockReceiverCore>>,
        }

        impl Debug for ReceiverMock {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("ReceiverMock").finish()
            }
        }

        impl CrossLoaderReceiver<DummyBatchOutput> for ReceiverMock {
            fn receive(
                self: Arc<Self>,
                out: &DummyBatchOutput,
                _ctx: Arc<parking_lot::RwLock<RunCtx2>>,
            ) -> Result<(), ThoosaError> {
                self.core.lock().receive(out)
            }
        }

        pub fn mock_receiver(expected_ids: &[&'static str]) -> Arc<ReceiverMock> {
            let mut core = MockReceiverCore::new();
            let mut seq = Sequence::new();

            for &expected in expected_ids {
                core.expect_receive()
                    .times(1)
                    .in_sequence(&mut seq)
                    .returning(move |out: &DummyBatchOutput| {
                        assert_eq!(WithId::id(out).0, expected, "receiver saw unexpected id");
                        Ok(())
                    });
            }

            Arc::new(ReceiverMock {
                core: Arc::new(Mutex::new(core)),
            })
        }

        #[derive(Debug)]
        pub struct DummyBatchOutput {
            id: SimpleId,
        }

        impl DummyBatchOutput {
            pub fn new(id: &'static str) -> Self {
                Self {
                    id: SimpleId::from(id),
                }
            }
        }

        impl WithId for DummyBatchOutput {
            type ID = SimpleId;

            fn id(&self) -> &Self::ID {
                &self.id
            }
        }

        impl Input for DummyBatchOutput {
            fn id(&self) -> &CachableId {
                &self.id
            }
        }

        impl Output for DummyBatchOutput {}

        fn build_tensor(idx: usize, spec: &TensorSpec) -> Tensor {
            let mut data = Vec::with_capacity((spec.channels * spec.seq_len) as usize);
            for ch in 0..spec.channels {
                for pos in 0..spec.seq_len {
                    let val = idx as f32 + ch as f32 / 10.0 + pos as f32 / 100.0;
                    data.push(val);
                }
            }
            Tensor::from_slice(&data).view([spec.channels, spec.seq_len])
        }

        fn assert_batch_input(
            fixtures: &[LoaderFixture],
            input: &BatchInput<'_, &'_ Arc<Resource>>,
            expected_left: &[usize],
            expected_right: &[usize],
        ) {
            assert_eq!(input.batch_0().len(), expected_left.len(), "left resources mismatch");
            assert_eq!(input.batch_1().len(), expected_right.len(), "right resources mismatch");

            for (slot, idx) in expected_left.iter().enumerate() {
                assert_eq!(input.batch_0()[slot].id(), fixtures[*idx].resource.id());
            }
            for (slot, idx) in expected_right.iter().enumerate() {
                assert_eq!(input.batch_1()[slot].id(), fixtures[*idx].resource.id());
            }

            let expected_indices = expected_left
                .iter()
                .chain(expected_right.iter())
                .copied()
                .collect::<Vec<_>>();

            assert_eq!(
                input.tensors().len(),
                expected_indices.len(),
                "tensor count mismatch",
            );

            for (pos, idx) in expected_indices.into_iter().enumerate() {
                let tensor = input.tensors()[pos];
                let size = tensor.size();
                assert_eq!(size[0], fixtures[idx].spec.channels);
                assert_eq!(size[1], fixtures[idx].spec.seq_len);
            }
        }
    }
}
