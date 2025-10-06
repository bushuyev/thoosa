use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;
use async_trait::async_trait;
use log::debug;
use parking_lot::RwLock;
use tch::Device::Cpu;
use tch::kind::FLOAT_CPU;
use tch::Tensor;
use cached_derive::Cached;
use caching_runner::common::{Data, OutId, RunCtx, Select};
use caching_runner::common::runnable::Stage;
use caching_runner::tmp_cache::TmpCached;
use thoosa_common::dims::Dims;
use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::{CopyToDevice, ThoosaError};
use thoosa_shared::traits::{DataId, SimpleId, WithId};
use thoosa_store::Store;
use torch_runner::data::ThoTensor2;
use torch_runner::metrics::TwoDimsMetrics;
use crate::tasks::cross_dims::{ChannelsData2, ChannelsDataVec, MotifSlices};
use crate::tasks::load_motif_and_clc_dist::{MotifSlicesSelector, ClcMotifInput};
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::{RawTensorLoader, TensorLoader};
use ambassador::Delegate;
use rand::distr::{Alphanumeric, SampleString};
use rand::thread_rng;
use caching_runner::oom_handler::Adjustable;

#[derive(Debug, Cached)]
pub struct MotifLoader<SRC>
where
    SRC: TensorLoader + 'static
{
    store: Arc<RwLock<Box<dyn Store>>>,
    loader:Arc<SRC>
}

impl <SRC> MotifLoader<SRC>
    where SRC: TensorLoader + 'static
{
    pub fn new(
        store: Arc<RwLock<Box<dyn Store>>>,
        loader:Arc<SRC>
    ) -> Self {
        Self {
            store,
            loader
        }
    }
}

impl <SRC> OutId for MotifLoader<SRC>
where SRC: TensorLoader + 'static
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

#[derive(Debug)]//Delegate
// #[delegate(Select<MotifSlicesSelector>, t)]
pub struct MotifWithData {
    t:ThoTensor2<MotifSlices, DataId>,
    channel:i64,
    run_id: RunId
}

impl Select<MotifSlicesSelector> for MotifWithData {
    type Output = MotifWithData;

    fn subview(&self, selector: MotifSlicesSelector) -> Self::Output {
        MotifWithData {
            t: self.t.subview(selector),
            channel: self.channel,
            run_id: self.run_id
        }
    }

}

impl Adjustable for MotifWithData {}
impl ClcMotifInput for MotifWithData {}

impl MotifWithData {
    pub fn new(t: ThoTensor2<MotifSlices, DataId>, channel: i64, run_id: RunId) -> Self {
        Self { t, channel, run_id }
    }

    pub fn t(&self) -> &ThoTensor2<MotifSlices, DataId> {
        &self.t
    }

    pub fn channel(&self) -> i64 {
        self.channel
    }

    pub fn run_id(&self) -> RunId {
        self.run_id
    }
}

impl WithId for MotifWithData {
    type ID = DataId;

    fn id(&self) -> &Self::ID {
        self.t.id()
    }
}

#[async_trait]
impl CopyToDevice for MotifWithData {
    async fn copy_to_device(&self, dst_device: usize) -> Self{
        MotifWithData {
            t: self.t.copy_to_device(dst_device).await,
            channel: self.channel,
            run_id: self.run_id
        }
    }
}

impl Data for MotifWithData {
    type Metrics = <MotifSlices as Dims>::Metrics;

    fn size(&self) -> Self::Metrics {
        Data::size(&self.t)
    }

    fn device(&self) -> usize {
        self.t.device()
    }
}

impl Input for MotifWithData {}
impl Output for MotifWithData {}

#[async_trait]
impl <SRC> Stage for MotifLoader <SRC>
    where SRC: TensorLoader + 'static
{
    type In<'a> = DataId;
    type Out = MotifWithData;

    type Ctx = RunCtx2;

    async fn run(self: Arc<Self>, input: Arc<Self::In<'static>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    where
            for<'a> Self::In<'a>: Send + Sync
    {
        let (channel, run_id, slices, resources_map) = self.store.read().list_motif_slices(**input, None, None)?;

        let find_motifs_run_config = FindMotifsRunConfig::from_properties_map__(
            run_id,
            &self.store.read().list_run_config_values(run_id)?,
        )?;

        let resource_ids_slices = slices
            .iter()
            .fold(HashMap::new(), |mut a, slice| {
            a.entry(slice.0.clone()) //TODO clone
                .or_insert_with(|| vec![])
                .push(slice.1);
            a
        });

        let window = find_motifs_run_config.window;
        let (device_idx, cache_area) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area())
        };

        debug!("MotifLoader::run: window={window:?} device_idx={device_idx}");

        let tmp_cache:TmpCached<ThoTensor2<ChannelsData2, SimpleId>> = TmpCached::new(format!("motif_{}", Alphanumeric.sample_string(&mut rand::rng(), 16)).leak(), cache_area).await;

        for (resource_id, slices) in resource_ids_slices {
            let t = Arc::clone(&self.loader).load(Arc::new(resources_map.get(&resource_id).expect(&format!("no resource with id={resource_id}")).clone()), Arc::clone(&ctx)).await?;

            let selector = ChannelsDataVec {
                ranges:
                slices
                    .into_iter()
                    .map(|slice| {
                        TwoDimsMetrics::from([channel, slice as i64])..TwoDimsMetrics::from([channel + 1, (slice + window) as i64])
                    })
                    .collect(),
            };

            t.with_selected_data(device_idx, selector, |tt| {
                debug!("MotifLoader::run/with_data:  resource_id={resource_id}  t={}", tt.t_ref());
                tmp_cache.insert(tt);

                Ok(())
            }).await?;

        }

        Ok(MotifWithData::new(ThoTensor2::new(
            input.deref().clone(),
            Tensor::cat(
                tmp_cache.take_all(device_idx).await.into_iter().map(|tt|tt.t()).collect::<Vec<Tensor>>().as_slice(), 
                0
            )
        ), channel, run_id))
    }
}

#[cfg(test)]
pub mod mock_motif_loader {
    use std::fmt;
    use std::pin::Pin;
    use std::sync::Arc;
    use async_trait::async_trait;
    use parking_lot::RwLock;
    use caching_runner::common::cachable::DataCacheWrapper;
    use thoosa_shared::ThoosaError;
    use thoosa_shared::traits::DataId;
    use torch_runner::data::ThoTensor2;
    use crate::tasks::cross_dims::MotifSlices;
    use crate::tasks::load_motif_and_clc_dist::MotifLoaderStage;
    use crate::tasks::motif_loader::MotifWithData;
    use crate::tasks::run_ctx_2::RunCtx2;

    type RunCachedFuture = Pin<
        Box<
            dyn Future<
                Output = Result<Arc<DataCacheWrapper<MotifWithData>>, ThoosaError>,
            > + Send,
        >,
    >;

    type RunCachedFn = dyn Fn(Arc<DataId>, Arc<RwLock<RunCtx2>>) -> RunCachedFuture + Send + Sync;

    #[derive(Clone)]
    pub struct MockMotifLoader {
        run_cached_fn: Arc<RunCachedFn>,
    }

    impl fmt::Debug for MockMotifLoader {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("MockMotifLoader").finish()
        }
    }

    #[async_trait]
    impl MotifLoaderStage for MockMotifLoader {
        type Output = MotifWithData;

        async fn run_cached(
            self: Arc<Self>,
            input: Arc<DataId>,
            ctx: Arc<RwLock<RunCtx2>>,
        ) -> Result<Arc<DataCacheWrapper<Self::Output>>, ThoosaError> {
            (self.run_cached_fn)(input, ctx).await
        }
    }

    pub struct MockMotifLoaderBuilder {
        run_cached_fn: Option<Arc<RunCachedFn>>,
    }

    impl MockMotifLoaderBuilder {
        pub fn with_run_cached_fn<F, Fut>(mut self, f: F) -> Self
        where
            F: Fn(Arc<DataId>, Arc<RwLock<RunCtx2>>) -> Fut + Send + Sync + 'static,
            Fut: Future<
                Output = Result<Arc<DataCacheWrapper<MotifWithData>>, ThoosaError>,
            > + Send
            + 'static,
        {
            self.run_cached_fn = Some(Arc::new(move |input, ctx| Box::pin(f(input, ctx))));
            self
        }

        pub fn with_result(
            self,
            data: Arc<DataCacheWrapper<MotifWithData>>,
        ) -> Self {
            self.with_run_cached_fn(move |_input, _ctx| {
                let data = Arc::clone(&data);
                async move { Ok(data) }
            })
        }

        pub fn finish(self) -> Arc<MockMotifLoader> {
            let run_cached_fn = self
                .run_cached_fn
                .expect("MockMotifLoaderBuilder requires run_cached_fn to be set");
            Arc::new(MockMotifLoader { run_cached_fn })
        }
    }

    pub fn build() -> MockMotifLoaderBuilder {
        MockMotifLoaderBuilder { run_cached_fn: None }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use log::LevelFilter::Trace;
    use log::{set_max_level, trace};
    use parking_lot::RwLock;
    use caching_runner::common::cachable::Cached;
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::utils::lin_t;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::{TensorIdSliceIdx, ThoosaError};
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::DataId;
    use thoosa_store::MockStoreBuilder;
    use crate::tasks::motif_loader::MotifLoader;
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use crate::tasks::tensor_loader::mock_tensor_loader::make_wrapper_t;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn load_motifs() -> Result<(), ThoosaError> {
        set_max_level(Trace);
        GlobalCtx::init_devices(2);

        let run_id = 1;

        let mut store = MockStoreBuilder::new()
            .with_list_motif_slices(10u32, None, None, Ok((0i64, run_id, vec![
                TensorIdSliceIdx(1, 100),
                TensorIdSliceIdx(1, 200),
                TensorIdSliceIdx(2, 100)
            ], HashMap::from([
                (1, Resource::new(1, 1, "aaa".into(), 1000)),
                (2, Resource::new(2, 1, "bbb".into(), 1000))
            ]))))
            .with_list_run_config_values(
                run_id,
                Ok(FindMotifsRunConfig::new(Some(run_id), 100, 100, None, None).to_properties_iter(None).collect())
            )
            .build();

        let motif_loader = Arc::new(MotifLoader::new(
            Arc::new(RwLock::new(store)),

            Arc::new(
                mock_tensor_loader::new()
                .with_data([
                    (1, make_wrapper_t("tensor_a", lin_t(1000).view((1,-1)))),
                    (2, make_wrapper_t("tensor_b", lin_t(1000).view((1,-1)))),
                ])
                .build()
            )
        ));

        let ctx = Arc::new(RwLock::new(RunCtx2::new(0, 0, "test")));

        let motif = motif_loader.run_cached(Arc::new(DataId::from(10)), ctx).await?;

        let r = motif.with_data_ref(0, None, |t|{
            let dist = 0.; //average_euclidean_distance(t.t_ref()).unwrap();

            trace!("motif.t={} dist={dist}", t.t.t_ref());

            Ok(true)
        }).await?;

        assert_eq!(r, true);

        Ok(())
    }

}

