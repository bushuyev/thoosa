use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use async_trait::async_trait;
use log::{debug, trace, warn};
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
use thoosa_shared::{CopyToDevice, TensorIdSliceIdx, ThoosaError};
use thoosa_shared::traits::{Data, DataId, Select, SimpleId, WithId};
use thoosa_store::Store;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::traits::MonoMetrics;
use crate::tasks::cross_dims::{ChannelsData2, ChannelsDataVec, MotifSlices};
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::{RawTensorLoader};
use ambassador::Delegate;
use rand::distr::{Alphanumeric, SampleString};
use caching_runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::traits::Adjustable;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::{Resource, ResourceId};
use crate::tasks::find_motifs::{config_to_stage, FilteredLoaderStage, LoaderStageChoice};
use crate::tasks::stages_seq::{ABStage, StageChainOption};
use crate::tasks::transforms::freq_filter::FreqFilter;

#[derive(Debug, Cached)]
pub struct MotifLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{
    store: Arc<RwLock<Box<dyn Store>>>,
    loader: Arc<L>,
    pd: PhantomData<C>,
}

impl<L, C> MotifLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{
    pub fn new(
        store: Arc<RwLock<Box<dyn Store>>>,
        loader: Arc<L>,
    ) -> Self {
        Self {
            store,
            loader,
            pd: Default::default(),
        }
    }
}

impl<L, C> OutId for MotifLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

#[derive(Debug, Clone)]
pub struct MotifMeta {
    pub channel:i64,
    pub find_motifs_run_id: RunId,
    pub window:u32
}

impl<L, C> Stage for MotifLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{
    type In<'a> = DataId;
    type Out = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>;

    type Ctx = C;
}

#[async_trait]
impl<L, C> AsyncStage for MotifLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{

    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError>
    where
        for<'a> Self::In<'a>: Send + Sync,
    {
        let (device_idx, cache_area) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area())
        };

        let motif_id = input
            .with_data_ref(0, None, |id: &DataId| Ok(id.clone()))
            .await?;

        let motif_id_value = motif_id.clone().0;
        let (channel, find_motifs_run_id, slices, resources_map) =
            self.store
                .read()
                .list_motif_slices(motif_id_value, None, None)?;

        let (window, motif) = load_motif_slices(motif_id, Arc::clone(&self.store), Arc::clone(&self.loader), None, None, &ctx, find_motifs_run_id, slices, resources_map, channel, device_idx, cache_area).await?;


        Ok(Arc::new(DevicesWrap::wrap(
            DataWithExtra::new(
                motif,
                MotifMeta{channel, find_motifs_run_id: find_motifs_run_id, window },
            ),
            cache_area,
        )))
    }
}
///To load slices not belonging to a motif
#[derive(Debug)]
pub struct RandomSlicesLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{
    store: Arc<RwLock<Box<dyn Store>>>,
    loader: Arc<L>,
    pd: PhantomData<C>,
}

impl <L,C> RandomSlicesLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{
    pub fn new(store: Arc<RwLock<Box<dyn Store>>>, loader: Arc<L>) -> Self {
        Self { store, loader, pd: PhantomData::default() }
    }
}

#[derive(Debug, Clone, DummyData)]
pub struct SlicesLoaderInput {
    motif_id:DataId,
    limit:u32
}

impl SlicesLoaderInput {
    pub fn new(motif_id: DataId, limit: u32) -> Self {
        Self { motif_id, limit }
    }
}

impl WithId for SlicesLoaderInput {
    type ID = DataId;

    fn id(&self) -> &Self::ID {
        &self.motif_id
    }
}

impl Input for SlicesLoaderInput {}
impl<L, C> Stage for RandomSlicesLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{
    type In<'a> = SlicesLoaderInput;
    type Out = ThoTensor<MotifSlices, DataId>;

    type Ctx = C;
}


#[async_trait]
impl<L, C> AsyncStage for RandomSlicesLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{

    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<SlicesLoaderInput>>,
        ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError>
    where
            for<'a> Self::In<'a>: Send + Sync,
    {
        let (device_idx, cache_area) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area())
        };
        let SlicesLoaderInput { motif_id, limit } = input.clone_from_cpu().await?;

        let (channel, find_motifs_run_id) = self.store.read().get_motif_meta(*motif_id)?;

        let find_motifs_run_config = FindMotifsRunConfig::from_properties_map__(
            find_motifs_run_id,
            &self.store.read().list_run_config_values(find_motifs_run_id)?,
        )?;

        debug!("RandomSlicesLoader::run: motif_id={motif_id} channel={channel} find_motifs_run_config={find_motifs_run_config:?}");

        let motif_id_value = motif_id.clone().0;
        let (channel, find_motifs_run_id, slices, resources_map) =
            self.store
                .read()
                .list_random_slices(motif_id_value, find_motifs_run_config.window, None)?;

        let (window, motif) = load_motif_slices(motif_id, Arc::clone(&self.store), Arc::clone(&self.loader), Some(limit), Some((0, 0)), &ctx, find_motifs_run_id, slices, resources_map, channel, device_idx, cache_area).await?;


        Ok(Arc::new(DevicesWrap::wrap(
            motif,
            cache_area,
        )))
    }
}

async fn load_motif_slices<L, C>(motif_id: DataId, store: Arc<RwLock<Box<dyn Store>>>, loader: Arc<L>, limit: Option<u32>, all_but_in_run_id: Option<(RunId, u32)>, ctx: &Arc<RwLock<C>>, find_motifs_run_id: u32, slices: Vec<TensorIdSliceIdx>, resources_map: HashMap<ResourceId, Resource>, channel: i64, device_idx: usize, cache_area: u64) -> Result<( u32, ThoTensor<MotifSlices, DataId>), ThoosaError>
where
    C: 'static + RunCtx,
    L: 'static + Cached + OutId + for<'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C>,
{


    let find_motifs_run_config = FindMotifsRunConfig::from_properties_map__(
        find_motifs_run_id,
        &store.read().list_run_config_values(find_motifs_run_id)?,
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

    let num_devices = GlobalCtx::global().device_count();

    debug!("MotifLoader::run: find_motifs_run_id={find_motifs_run_id} window={window:?} device_idx={device_idx}");

    let tmp_cache: TmpCached<ThoTensor<ChannelsData2, SimpleId>> = TmpCached::new("motifs", cache_area).await;

    let configurable_loader = config_to_stage(&loader, &find_motifs_run_config);

    for (resource_id, slices) in resource_ids_slices {
        let resource = resources_map
            .get(&resource_id)
            .cloned()
            .expect(&format!("no resource with id={resource_id}"));
        let resource_cache = Arc::new(DevicesWrap::wrap_cpu(resource));

        let raw_tensor = Arc::clone(&configurable_loader)
            .run_cached(resource_cache, Arc::clone(&ctx))
            .await?;

        let size = raw_tensor.size();

        let selector = ChannelsDataVec {
            ranges: slices
                .into_iter()
                .filter(|slice| {
                    let r = MonoMetrics::from([channel + 1, (slice + window) as i64]) < *size;
                    if !r {
                        warn!("MotifLoader::run: slice={slice:?} is bigger than size={size:?} for resource_id={resource_id} in find_motifs_run_id={find_motifs_run_id}");
                    }
                    r
                })
                .map(|slice| {
                    MonoMetrics::from([channel, slice as i64])..MonoMetrics::from([channel + 1, (slice + window) as i64])
                })
                .collect(),
        };

        if !selector.ranges.is_empty() {
            raw_tensor.with_selected_data(device_idx, selector, |tt| {
                trace!("MotifLoader::run/with_data:  resource_id={resource_id}  t={}", tt.t_ref());
                if tt.size() == &[0] {
                    warn!("MotifLoader::run/with_data:  resource_id={resource_id}  got empty motif for motif_id={motif_id}");
                } else {
                    tmp_cache.insert(tt);
                }

                Ok(())
            }).await?;
        } else {
            warn!("Empty selector for resource_id={resource_id}  got empty motif for motif_id={motif_id}");
        }
    }

    let tensors: Vec<Tensor> = tmp_cache
        .take_all(device_idx)
        .await
        .into_iter()
        .map(|tt| tt.t())
        .collect();

    let motif = ThoTensor::new(
        motif_id,
        Tensor::cat(tensors.as_slice(), 0),
    );
    Ok((window, motif))
}


#[cfg(test)]
pub mod mock_motif_loader {
    use std::collections::HashMap;
    use async_trait::async_trait;
    use futures_util::future::{BoxFuture, Shared};
    use mockall::mock;
    use std::sync::Arc;
    use caching_runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
    use caching_runner::common::{MockRunCtx, OutId, RunCtx};
    use caching_runner::common::runnable::{AsyncStage, Stage, SyncStage};
    use thoosa_shared::{CopyToDevice, ThoosaError};
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::{DataId, Named, SimpleId};
    use thoosa_shared::traits::WithId;
    use thoosa_common::tho_tensor::ThoTensor;
    use dashmap::DashMap;
    use parking_lot::RwLock;
    use tch::kind::FLOAT_CPU;
    use tch::Tensor;
    use tokio::runtime::Handle;
    use tokio::task::block_in_place;
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use crate::tasks::cross_dims::{ChannelsData2, MotifSlices};
    use crate::tasks::motif_loader::{MotifLoader, MotifMeta};
    use crate::tasks::tensor_loader::mock_tensor_loader::MockTensorLoaderBuilder;

    mock! {
        #[derive(Debug)]
        pub MotifLoader {}


        impl Stage for MotifLoader {
            type Ctx = MockRunCtx;

            type In<'a> = DataId;
            type Out = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>;
        }

        impl OutId for MotifLoader {
            fn out_id<'a>(&self,input_id: &<<Self as Stage>::In<'a> as WithId>::ID,) -> <<MockMotifLoader as Stage>::Out as WithId>::ID;
        }

        #[async_trait]
        impl AsyncStage for MotifLoader {}

        #[async_trait]
        impl Cached for MotifLoader {

            async fn cache() -> &'static DashMap<<<Self as Stage>::Out as WithId>::ID, Shared<BoxFuture<'static, Result<Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>>>, ThoosaError>>>>;
            async fn run_cached(
                self: Arc<Self>,
                input: Arc<DevicesWrap<<MockMotifLoader as Stage>::In<'static>>>,
                ctx: Arc<RwLock<<MockMotifLoader as Stage>::Ctx>>,
            ) -> Result<Arc<DevicesWrap<<MockMotifLoader as Stage>::Out>>, ThoosaError>;
        }
    }

    pub struct MockMotifLoaderBuilder {
        mock:MockMotifLoader
    }

    impl MockMotifLoaderBuilder {
        pub fn with_run_cached<H>(mut self, h: H) -> Self
        where
            H: IntoIterator<Item=(u32, (MotifMeta, Tensor))>,
        {
            let data_map: Arc<HashMap<DataId, Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>>>>> = Arc::new(h.into_iter().map(|(k, v)| {
                (k.into(), Arc::new(DevicesWrap::wrap(DataWithExtra::new(ThoTensor::new(k.into(), v.1), v.0), 0)))
            }).collect());

            self.mock.expect_run_cached().returning(move |input, ctx| {
                block_in_place(|| {
                    Handle::current().block_on(async {
                        input.with_data_ref(0, None, |v: &DataId| {
                            Ok(Arc::clone(data_map
                                .get(v)
                                .expect(&format!("no mock data for {input:? } in MockMotifLoader"))))
                        }).await
                    })
                })
            });

            self
        }


        pub fn finish(self) -> MockMotifLoader {
            self.mock
        }

    }
    pub fn build()->MockMotifLoaderBuilder {
        MockMotifLoaderBuilder {
            mock: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use log::LevelFilter::Trace;
    use log::{set_max_level, trace};
    use parking_lot::RwLock;
    use tch::Tensor;
    use caching_runner::common::{mock_run_ctx, RunCtx};
    use caching_runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
    use caching_runner::common::runnable::AsyncStage;
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::utils::lin_t;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::{TensorIdSliceIdx, ThoosaError};
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::DataId;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_store::MockStoreBuilder;
    use crate::tasks::motif_loader::{MotifLoader, MotifMeta, RandomSlicesLoader, SlicesLoaderInput};
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::{mock_tensor_loader};
    use crate::tasks::cross_dims::MotifSlices;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_motifs::distance::average_euclidean_distance;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_load_motifs() -> Result<(), ThoosaError> {
        set_max_level(Trace);
        GlobalCtx::init_devices(2);

        let run_id = 1;

        let resource_a = Resource::new(1, 1, "tensor_a", 1000);
        let resource_b = Resource::new(2, 1, "tensor_b", 1000);
        let mut store = MockStoreBuilder::new()
            .with_list_motif_slices(10u32, None, None, Ok((0i64, run_id, vec![
                TensorIdSliceIdx(1, 100),
                TensorIdSliceIdx(1, 200),
                TensorIdSliceIdx(2, 100)
            ], HashMap::from([
                (1, resource_a.clone()),
                (2, resource_b.clone())
            ]))))
            .with_list_run_config_values(
                run_id,
                Ok(FindMotifsRunConfig::new(Some(run_id), 100, 100, None, None).to_properties_iter(None).collect())
            )
            .build();

        let motif_loader = Arc::new(MotifLoader::new(
            Arc::new(RwLock::new(store)),

            Arc::new(
                mock_tensor_loader::build()
                    .with_out_id()
                    .with_run_cached([
                        (resource_a, lin_t(1000).view((1,-1))),
                        (resource_b, lin_t(1000).view((1,-1)))
                    ])
                    .finish(),
            )
        ));

        let ctx = Arc::new(RwLock::new(mock_run_ctx::default(0)));

        let num_devices = GlobalCtx::global().device_count();
        let cache_area = ctx.read().cache_area();
        let input_id = Arc::new(DevicesWrap::wrap_cpu(DataId::from(&10u32)));

        let motif = motif_loader
            .run_cached(Arc::clone(&input_id), Arc::clone(&ctx))
            .await?;

        let r = motif.with_data_ref(0, None, |t: &DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>|{
            let dist = 0.; //average_euclidean_distance(t.t_ref()).unwrap();

            trace!("motif.t={} dist={dist}", t.data().t_ref());

            Ok(true)
        }).await?;

        assert_eq!(r, true);
        //TODO check data

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_load_random_slices() -> Result<(), ThoosaError> {
        set_max_level(Trace);
        GlobalCtx::init_devices(2);

        let run_id = 1;
        let motif_id = 10u32;
        let motif_width = 100;

        let resource_a = Resource::new(1, 1, "tensor_a", 1000);
        let resource_b = Resource::new(2, 1, "tensor_b", 1000);
        let mut store = MockStoreBuilder::new()
            .with_list_random_slices(motif_id, motif_width, None, Ok((0i64, run_id, vec![
                TensorIdSliceIdx(1, 100),
                TensorIdSliceIdx(1, 200),
                TensorIdSliceIdx(2, 100)
            ], HashMap::from([
                (1, resource_a.clone()),
                (2, resource_b.clone())
            ]))))
            .with_list_run_config_values(
                run_id,
                Ok(FindMotifsRunConfig::new(Some(run_id), 100, motif_width, None, None).to_properties_iter(None).collect())
            )
            .with_get_motif_meta( [(motif_id, Ok((0usize, run_id)))])
            .build();

        let random_slices_loader = Arc::new(RandomSlicesLoader::new(
            Arc::new(RwLock::new(store)),

            Arc::new(
                mock_tensor_loader::build()
                    .with_out_id()
                    .with_run_cached([
                        (resource_a, lin_t(1000).view((1,-1))),
                        (resource_b, lin_t(1000).view((1,-1)))
                    ])
                    .finish(),
            )
        ));

        let ctx = Arc::new(RwLock::new(mock_run_ctx::default(0)));

        let num_devices = GlobalCtx::global().device_count();
        let cache_area = ctx.read().cache_area();
        let input = Arc::new(DevicesWrap::wrap_cpu(SlicesLoaderInput{ motif_id: DataId::from(&motif_id), limit: 10 }));

        let slices = random_slices_loader
            .run(Arc::clone(&input), Arc::clone(&ctx))
            .await?;

        let r = slices.with_data_ref(0, None, |t: &ThoTensor<MotifSlices, DataId>|{
            let dist = average_euclidean_distance(t.t_ref()).unwrap();

            trace!("random slices: \t{} dist={dist}", t.t_ref());

            Ok(true)
        }).await?;
        // assert_eq!(r, true);
        //TODO check data

        Ok(())
    }

}
