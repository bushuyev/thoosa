use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use async_trait::async_trait;
use log::debug;
use parking_lot::RwLock;
use tch::Device::Cpu;
use tch::kind::FLOAT_CPU;
use tch::Tensor;
use cached_derive::Cached;
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage};
use caching_runner::global_ctx::GlobalCtx;
use caching_runner::tmp_cache::TmpCached;
use thoosa_common::dims::Dims;
use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::{CopyToDevice, ThoosaError};
use thoosa_shared::traits::{Data, DataId, Select, SimpleId, WithId};
use thoosa_store::Store;
use torch_runner::data::ThoTensor2;
use thoosa_shared::traits::MonoMetrics;
use crate::tasks::cross_dims::{ChannelsData2, ChannelsDataVec, MotifSlices};
use crate::tasks::load_motif_and_clc_dist::{MotifSlicesSelector};
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::{RawTensorLoader};
use ambassador::Delegate;
use rand::distr::{Alphanumeric, SampleString};
use rand::thread_rng;
use caching_runner::common::cachable::{Cached, DevicesWrap};
use thoosa_shared::traits::Adjustable;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::Resource;
use crate::tasks::find_motifs::{FilteredLoaderStage, LoaderStageChoice};
use crate::tasks::stages_seq::{ABStage, StageChainOption};
use crate::tasks::transforms::freq_filter::FreqFilter;

#[derive(Debug, Cached)]
pub struct MotifLoader<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + 'static
{
    store: Arc<RwLock<Box<dyn Store>>>,
    loader:Arc<L>,
    pd:PhantomData<C>
}

impl <L, C> MotifLoader<L, C>
    where
        L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
        C: RunCtx + 'static
{
    pub fn new(
        store: Arc<RwLock<Box<dyn Store>>>,
        loader:Arc<L>
    ) -> Self {
        Self {
            store,
            loader,
            pd: Default::default(),
        }
    }
}

impl <L, C> OutId for MotifLoader<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + 'static
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

#[derive(Debug, Clone)]
pub struct MotifMeta {
    pub channel:i64,
    pub run_id: RunId
}

impl <L, C> Stage for MotifLoader <L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + 'static
{
    type In<'a> = DataId;
    type Out = DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>;

    type Ctx = RunCtx2;
}
#[async_trait]
impl <L, C> AsyncStage for MotifLoader <L, C>
    where
        L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
        C: RunCtx + 'static
{

    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError>
    where
        for<'a> Self::In<'a>: Send + Sync,
    {
        let motif_id = input
            .with_data_ref(0, None, |id| Ok(id.clone()))
            .await?;

        let motif_id_value = motif_id.clone().0;
        let (channel, find_motifs_run_id, slices, resources_map) =
            self.store
                .read()
                .list_motif_slices(motif_id_value, None, None)?;

        let find_motifs_run_config = FindMotifsRunConfig::from_properties_map__(
            find_motifs_run_id,
            &self.store.read().list_run_config_values(find_motifs_run_id)?,
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
        let num_devices = GlobalCtx::global().device_count();

        debug!("MotifLoader::run: find_motifs_run_id={find_motifs_run_id} window={window:?} device_idx={device_idx}");

        let tmp_cache:TmpCached<ThoTensor2<ChannelsData2, SimpleId>> = TmpCached::new("motifs", cache_area).await;

        let configurable_loader: Arc<LoaderStageChoice<L>> = match find_motifs_run_config.filter.as_ref() {
            None => {
                debug!("MotifLoader::run: find_motifs_run_id={find_motifs_run_id} no filter");
                Arc::new(StageChainOption::<L, FilteredLoaderStage<L>>::Single(Arc::clone(&self.loader)))
            },
            Some(filter_config) => {
                debug!("MotifLoader::run: find_motifs_run_id={find_motifs_run_id} filter_config={filter_config:?}");

                Arc::new(StageChainOption::<L, FilteredLoaderStage<L>>::Chained(Arc::new(ABStage::new(
                    SimpleId::from("ab"),
                    Arc::clone(&self.loader),
                    Arc::new(FreqFilter::new(filter_config)),
                ))))
            },
        };


        for (resource_id, slices) in resource_ids_slices {
            let resource = resources_map
                .get(&resource_id)
                .cloned()
                .expect(&format!("no resource with id={resource_id}"));
            let resource_cache = Arc::new(DevicesWrap::wrap_cpu(resource));

            let raw_tensor = Arc::clone(&self.loader)
                .run(resource_cache, Arc::clone(&ctx))
                .await?;

            let selector = ChannelsDataVec {
                ranges: slices
                    .into_iter()
                    .map(|slice| {
                        MonoMetrics::from([channel, slice as i64])..MonoMetrics::from([channel + 1, (slice + window) as i64])
                    })
                    .collect(),
            };

            raw_tensor.with_selected_data(device_idx, selector, |tt| {
                debug!("MotifLoader::run/with_data:  resource_id={resource_id}  t={}", tt.t_ref());
                tmp_cache.insert(tt);

                Ok(())
            }).await?;

        }

        let tensors: Vec<Tensor> = tmp_cache
            .take_all(device_idx)
            .await
            .into_iter()
            .map(|tt| tt.t())
            .collect();


        let output = DataWithExtra::new(
            ThoTensor2::new(
                motif_id,
                Tensor::cat(tensors.as_slice(), 0),
            ),
            MotifMeta{channel, run_id: find_motifs_run_id }
        );

        Ok(Arc::new(DevicesWrap::wrap(
            output,
            cache_area,
        )))
    }
}

#[cfg(test)]
pub mod mock_motif_loader {
    use std::fmt;
    use std::pin::Pin;
    use std::sync::Arc;
    use async_trait::async_trait;
    use parking_lot::RwLock;
    use cached_derive::Cached;
    use caching_runner::common::cachable::DevicesWrap;
    use caching_runner::common::OutId;
    use caching_runner::common::runnable::{AsyncStage, Stage};
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::ThoosaError;
    use thoosa_shared::traits::{DataId, SimpleId, WithId};
    use torch_runner::data::ThoTensor2;
    use crate::tasks::cross_dims::MotifSlices;
    // use crate::tasks::load_motif_and_clc_dist::MotifLoaderStage;
    use crate::tasks::motif_loader::{MotifMeta};
    use crate::tasks::run_ctx_2::RunCtx2;

    type RunCachedFuture = Pin<Box<dyn Future<Output=Result<Arc<DevicesWrap<DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>>>, ThoosaError>> + Send>>;

    type RunCachedFn = dyn Fn(
            Arc<DevicesWrap<DataId>>,
            Arc<RwLock<RunCtx2>>,
        ) -> RunCachedFuture
        + Send
        + Sync;

    #[derive(Clone, Cached)]
    pub struct MockMotifLoader {//TODO replace with mockallmock
        run_cached_fn: Arc<RunCachedFn>,
    }

    impl fmt::Debug for MockMotifLoader {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("MockMotifLoader").finish()
        }
    }

    impl Stage for MockMotifLoader {
        type Ctx = RunCtx2;
        type In<'a> where Self: 'a =DataId;
        type Out = DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>;
    }

    impl OutId for MockMotifLoader {
        fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
            DataId(0)
        }
    }

    #[async_trait]
    impl AsyncStage for MockMotifLoader {

        async fn run(
            self: Arc<Self>,
            input: Arc<DevicesWrap<DataId>>,
            ctx: Arc<RwLock<RunCtx2>>,
        ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
            (self.run_cached_fn)(input, ctx).await
        }
    }

    pub struct MockMotifLoaderBuilder {
        run_cached_fn: Option<Arc<RunCachedFn>>,
    }

    impl MockMotifLoaderBuilder {
        pub fn with_run_cached_fn<F, Fut>(mut self, f: F) -> Self
        where
            F: Fn(Arc<DevicesWrap<DataId>>, Arc<RwLock<RunCtx2>>) -> Fut + Send + Sync + 'static,
            Fut: Future<Output = Result<Arc<DevicesWrap<DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>>>, ThoosaError>, > + Send + 'static,
        {
            self.run_cached_fn = Some(Arc::new(move |input, ctx| Box::pin(f(input, ctx))));
            self
        }

        pub fn with_result(
            self,
            data: Arc<DevicesWrap<DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>>>,
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
    use caching_runner::common::RunCtx;
    use caching_runner::common::cachable::{Cached, DevicesWrap};
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
    use crate::tasks::tensor_loader::mock_tensor_loader::{make_wrapper_t, MockTensorLoader};

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

        let motif_loader = Arc::new(MotifLoader::<MockTensorLoader, RunCtx2>::new(
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

        let num_devices = GlobalCtx::global().device_count();
        let cache_area = ctx.read().cache_area();
        let input_id = Arc::new(DevicesWrap::wrap_cpu(DataId::from(&10u32)));

        let motif = motif_loader
            .run_cached(Arc::clone(&input_id), Arc::clone(&ctx))
            .await?;

        let r = motif.with_data_ref(0, None, |t|{
            let dist = 0.; //average_euclidean_distance(t.t_ref()).unwrap();

            trace!("motif.t={} dist={dist}", t.data().t_ref());

            Ok(true)
        }).await?;

        assert_eq!(r, true);

        Ok(())
    }
}