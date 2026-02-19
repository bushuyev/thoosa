#![allow(unexpected_cfgs)]

use std::collections::HashMap;
use std::fmt::{format, Debug, Formatter};
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use async_trait::async_trait;
use log::{debug, error, info, trace, warn};
use parking_lot::RwLock;
use tch::Device::Cpu;
use tch::kind::FLOAT_CPU;
use tch::Tensor;
use cached_derive::Cached;
use thoosa_named_derive::Named;
use thoosa_shared::runner::common::{DropIf, OutId, RunCtx};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage};
use thoosa_shared::runner::global_ctx::GlobalCtx;
use thoosa_shared::runner::tmp_cache::TmpCached;
use thoosa_common::dims::Dims;
use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::{mono_key, TensorIdSliceIdx, ThoError};
use thoosa_shared::traits::{CachableIdTr, Data, DataId, Select, WithId};
use thoosa_store::Store;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::traits::MonoMetrics;
use thoosa_with_data::with_data;
use crate::tasks::cross_dims::{ChannelsData2, ChannelsDataVec, MotifSlices};
use crate::tasks::load_motif_and_clc_dist::MotifSlicesSelector;
use thoosa_shared::runner::run_ctx_2::RunCtxWithPermits;
use crate::tasks::tensor_loader::RawTensorLoader;
use ambassador::Delegate;
use rand::distr::{Alphanumeric, SampleString};
use thoosa_common::utils::plot_t;
use thoosa_common::ws::Workspace;
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use thoosa_shared::runner::permits::MAX_PERMITS;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::traits::Adjustable;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::{Resource, ResourceId};
use crate::tasks::find_motifs::{config_to_stage, FilteredLoaderStage, LoaderStageChoice};
use crate::tasks::stages_seq::{ABStage, StageChainOption};
use crate::tasks::transforms::freq_filter::FreqFilter;
use thoosa_shared::comp_id::SynKey;

#[derive(Debug, Clone, PartialEq, Eq, Hash, DummyData)]
pub struct MotifLoaderInput {
    select_motifs_run_id: Option<Vec<RunId>>,
    motif_id:DataId
}

/*
#[automatically_derived]
impl ::thoosa_shared::traits::Data for MotifLoaderInput
where
    Self: ::core::clone::Clone,
{
    type Metrics = ();
    fn size(&self) -> Self::Metrics { () }
    fn device(&self) -> usize { 0 }
}*/

#[cfg(test)]
impl From<u32> for MotifLoaderInput{
    fn from(value: u32) -> Self {
        MotifLoaderInput::find_motif_slices_only(DataId(value))
    }
}

impl MotifLoaderInput {
    pub fn all_slices(motif_id: DataId) -> Self {
        Self {
            motif_id,
            select_motifs_run_id: None //all slices loaded
        }
    }

    pub fn find_motif_slices_only(motif_id: DataId) -> Self {
        Self {
            motif_id,
            select_motifs_run_id: Some(vec![]) //only slices found during find_motifs loaded
        }
    }

    pub fn new(motif_id: DataId, select_motifs_run_id: Option<Vec<RunId>>) -> Self {
        Self { select_motifs_run_id, motif_id }// slices found during find_motifs + select_motifs_run_id
    }
}
impl Input for MotifLoaderInput {
}

impl WithId for MotifLoaderInput {
    type ID = DataId;

    fn id(&self) -> &Self::ID {
        &self.motif_id
    }
}

#[derive(Debug, Cached, Named)]
pub struct MotifLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{
    store: Arc<Box<dyn Store>>,
    loader: Arc<L>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    pd: PhantomData<C>,
}

impl<L, C> MotifLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{
    pub fn new(
        store: Arc<Box<dyn Store>>,
        loader: Arc<L>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
    ) -> Self {
        Self {
            store,
            loader,
            ws: Arc::clone(&ws),
            pd: Default::default(),
        }
    }
}

impl<L, C> OutId for MotifLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
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
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{
    type In<'a> = MotifLoaderInput;
    type Out = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>;

    type Ctx = C;
}

// #[async_trait]
impl<L, C> AsyncStage for MotifLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="MotifLoader::run"))]
    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        mut ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError>
    where
        for<'a> Self::In<'a>: Send + Sync,
    {
        let device_idx = ctx.device_idx();
        let cache_area = ctx.cache_area();

        let motif_id = with_data!(ctx.branch(Some("motif_loader_input"), false, false), input; |_ctx, input: &MotifLoaderInput| {
            Ok::<DataId, ThoError>(input.motif_id.clone())
        })?;



        let motif_id_value = motif_id.clone().0;
        //gpt TODO!! select_motifs_run_id from MotifLoaderInput is ignored; list_motif_slices(None, ...) loads slices from all runs.
        let (channel, find_motifs_run_id, slices, resources_map) = self
            .store
            .list_motif_slices(motif_id_value, None, Some(1000), None)
            .await?;//TODO config

        debug!("MotifLoader::run: motif_id={motif_id} channel={channel} ctx={ctx:?} find_motifs_run_id={find_motifs_run_id}");

        let (window, motif) = load_motif_slices(motif_id.clone(), find_motifs_run_id, channel, resources_map, slices, Arc::clone(&self.store), Arc::clone(&self.loader), ctx, &self.ws, "motif_loader").await?;


        #[allow(unexpected_cfgs)]
        #[cfg(tho_debug_plots)]
        {
            static COUNT: AtomicU64 = AtomicU64::new(0);

            plot_t(&motif.as_ref(), &format!("motif_{motif_id}__{COUNT:?}"), false)?; //.f_mul_scalar(100.)?

            COUNT.fetch_add(1, Relaxed);
        }

        Ok(Arc::new(DevicesWrap::wrap(
            DataWithExtra::new(
                motif,
                MotifMeta{channel, find_motifs_run_id, window },
            ),
            cache_area.into(),
        )))
    }
}
///To load slices not belonging to a motif
#[derive(Debug)]
pub struct RandomSlicesLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{
    store: Arc<Box<dyn Store>>,
    loader: Arc<L>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    pd: PhantomData<C>,
}

impl <L,C> RandomSlicesLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{
    pub fn new(store: Arc<Box<dyn Store>>, loader: Arc<L>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Self {
        Self { store, loader, pd: PhantomData::default(), ws:Arc::clone(&ws) }
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
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{
    type In<'a> = SlicesLoaderInput;
    type Out = ThoTensor<MotifSlices, DataId>;

    type Ctx = C;
}


// #[async_trait]
impl<L, C> AsyncStage for RandomSlicesLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + 'static,
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="RandomSlicesLoader::run"))]
    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<SlicesLoaderInput>>,
        mut ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError>
    where
            for<'a> Self::In<'a>: Send + Sync,
    {
        let (motif_id, limit) = with_data!(ctx.branch(Some("random_slices_input"), false, false), input; |_ctx, input: &SlicesLoaderInput| {
            Ok::<(DataId, u32), ThoError>((input.motif_id.clone(), input.limit))
        })?;


        let (channel, find_motifs_run_id) = self.store.get_motif_meta(*motif_id).await?;

        let find_motifs_run_config = FindMotifsRunConfig::from_properties_map_with_id(
            find_motifs_run_id,
            &self.store.list_run_config_values(find_motifs_run_id).await?,
        )?;


        debug!("RandomSlicesLoader::run: motif_id={motif_id} channel={channel} ctx={ctx:?} find_motifs_run_id={find_motifs_run_id}");

        let motif_id_value = motif_id.clone().0;
        let input_len_adjuster = find_motifs_run_config.input_len_adjuster();
        let (channel, find_motifs_run_id, slices, resources_map) = thoosa_shared::with_async_perf_log(
            || async {
                self.store
                    .list_random_motif_slices(motif_id_value, find_motifs_run_config.window, None, input_len_adjuster)
                    .await
            },
            "RandomSlicesLoader::run/store_read_list_random_motif_slices",
        )
        .await?;
        let cache_area = ctx.cache_area().into();
        let (window, motif) = load_motif_slices(motif_id.clone(), find_motifs_run_id, channel, resources_map, slices, Arc::clone(&self.store), Arc::clone(&self.loader), ctx, &self.ws, "random loader").await?;

        #[allow(unexpected_cfgs)]
        #[cfg(tho_debug_plots)]
        {
            static COUNT: AtomicU64 = AtomicU64::new(0);

            plot_t(&motif.as_ref(), &format!("random_motif_{motif_id}__{COUNT:?}"), false)?; //.f_mul_scalar(100.)?

            COUNT.fetch_add(1, Relaxed);
        }


        Ok(Arc::new(DevicesWrap::wrap(
            motif,
            cache_area,
        )))
    }
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="load_motif_slices"))]
pub(crate) async fn load_motif_slices<L, C>(
    motif_id: DataId,
    find_motifs_run_id: u32,
    channel: i64,
    resources_map: HashMap<ResourceId, Resource>,
    slices: Vec<TensorIdSliceIdx>,
    store: Arc<Box<dyn Store>>,
    loader: Arc<L>,
    mut ctx: C,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    descr: &str
) -> Result<(u32, ThoTensor<MotifSlices, DataId>), ThoError>
where
    C: 'static + RunCtx,
    L: 'static + Cached + OutId + for<'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C>,
{
    let device_idx = ctx.device_idx();
    debug!("load_motif_slices: motif_id={motif_id} slices={slices:?} descr={descr}");

    let find_motifs_run_config = FindMotifsRunConfig::from_properties_map_with_id(
        find_motifs_run_id,
        &store.list_run_config_values(find_motifs_run_id).await?,
    )?;

    let resource_ids_slices = slices
        .iter()
        // .take(100)//TODO input/config
        .fold(HashMap::new(), |mut a, slice| {
            a.entry(slice.0.clone()) //TODO clone
                .or_insert_with(|| vec![])
                .push(slice.1);
            a
        });

    assert!(!resource_ids_slices.is_empty());

    let window = find_motifs_run_config.window;

    debug!("load_motif_slices: find_motifs_run_id={find_motifs_run_id} window={window:?} device_idx={device_idx}");

    let tmp_cache: TmpCached<ThoTensor<ChannelsData2, SynKey>> = TmpCached::new("motifs", ctx.cache_area()).await;

    let configurable_loader = config_to_stage(&loader, &find_motifs_run_config, ws)?;

    for (resource_id, slices) in resource_ids_slices {

        let resource = resources_map
            .get(&resource_id)
            .cloned()
            .expect(&format!("no resource with id={resource_id}"));
        let resource_cache = Arc::new(DevicesWrap::wrap(resource, ctx.cache_area().into()));

        //into is needed because mock needs SynKey or &str
        let raw_tensor = Arc::clone(&configurable_loader)
            .run_cached(resource_cache, ctx.branch(Some(&format!("motif_loader_resource_{resource_id}")), false, false))
            .await?;

        let size = raw_tensor.size();

        let selector = ChannelsDataVec {
            ranges: slices
                .into_iter()
                .filter(|slice| {
                    let r = MonoMetrics::<1,2>::from([channel + 1, (slice + window) as i64]) <= *size;
                    if !r {
                        warn!("load_motif_slices: motif_id={motif_id} slice={slice:?} is bigger than size={size:?} for resource_id={resource_id} in find_motifs_run_id={find_motifs_run_id}  in {descr}");
                    }
                    r
                })
                .map(|slice| {
                    MonoMetrics::from([channel, slice as i64])..MonoMetrics::from([channel + 1, (slice + window) as i64])
                })
                .collect(),
        };

        if !selector.ranges.is_empty() {
            let mut ctx_for_data = ctx.branch(Some("motif_loader_selected_data"), false, false);
            let mut try_count = 0;
            loop {

                // let out_key = out_key.clone();
                let out_key = Arc::new(ctx_for_data.id().clone()); //out_key.clone();

                //TODO async OOM: move into OOMHandler as util fn
                match raw_tensor.with_selected_data(
                    device_idx,
                    ctx_for_data.branch(Some("motif_loader_selected_data_with_selected_data"), false, false),
                    selector.clone(),
                    |tt, _ctx| {
                        debug!("load_motif_slices:  motif_id={motif_id} resource_id={resource_id} try_count={try_count} resource_id={resource_id}  tt.size={:?}", tt.size());
                        trace!("load_motif_slices:  motif_id={motif_id} resource_id={resource_id}  try_count={try_count} resource_id={resource_id}  t={}", tt.t_ref());
                        if tt.size() == &[0] {
                            warn!("load_motif_slices: motif_id={motif_id} resource_id={resource_id}  got empty motif in {descr}");
                        } else {
                            tmp_cache.insert(tt);
                            debug!("load_motif_slices: motif_id={motif_id} resource_id={resource_id} added to tmp_cache");
                        }

                        Ok(())
                    }).await {
                        Ok(_) => {
                            break
                        }
                        Err(e) => {
                            warn!("load_motif_slices: motif_id={motif_id} error while loading raw tensor: {e:?}");

                            if try_count < 1 {
                                let acquired_permits = ctx_for_data.acquired_permits();
                                if acquired_permits < MAX_PERMITS {
                                    let result = GlobalCtx::global()
                                        .acquire_many(device_idx, MAX_PERMITS - acquired_permits, out_key)
                                        .await?;
                                    ctx_for_data.add_permit(result); //permits can be dropped only from OOM handler or when context is dropped
                                }
                                debug!("load_motif_slices:  motif_id={motif_id} resource_id={resource_id}  try_count={try_count} got {e:?}, acquired MAX_PERMITS, purging gpu({device_idx})");
                                tmp_cache.purge_gpu(device_idx).await?;

                                ctx_for_data.drop_permits(DropIf::Always);

                                debug!("load_motif_slices:  motif_id={motif_id} resource_id={resource_id} purge gpu done");

                                try_count += 1;
                            } else {
                                error!("Can't load for resource_id={resource_id} resource_id={resource_id}  for motif_id={motif_id} e={e:?}");
                                break
                            }
                        }
                };
            }
        } else {
            warn!("Empty selector for resource_id={resource_id}  got empty motif for motif_id={motif_id}  in {descr}");
        }
    }

    debug!("load_motif_slices: motif_id={motif_id} taking tmp_cache");

    let cache_name = tmp_cache.name;
    let tensors: Vec<Tensor> = tmp_cache
        .take_all(device_idx)
        .await?
        .into_iter()
        .map(|tt| tt.t())
        .collect();

    GlobalCtx::global().remove_cache(cache_name).await;//TODO add oneshot when cache is registered in GlobCtx and send when dropped?

    debug!("load_motif_slices: loaded {} tensors for motif_id={motif_id} descr={descr}", tensors.len());

    assert!(!tensors.is_empty(), "no slice tensors loaded for motif_id={motif_id} ");

    let motif = ThoTensor::new(
        motif_id.clone(),
        Tensor::f_cat(tensors.as_slice(), 0)?,
    )?;

    debug!("load_motif_slices: motif of size {:?} is built for motif_id={motif_id} descr={descr} device={:?}", motif.size(), motif.t.device());

    Ok((window, motif))
}


#[cfg(test)]
pub mod mock_motif_loader {
    use std::collections::HashMap;
    use async_trait::async_trait;
    use futures_util::future::{BoxFuture, Shared};
    use mockall::mock;
    use std::sync::Arc;
    use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
    use thoosa_shared::runner::cache_areas::CacheArea;
    use thoosa_shared::runner::common::{MockRunCtx, OutId, RunCtx};
    use thoosa_shared::runner::common::runnable::{AsyncStage, Stage, SyncStage};
    use thoosa_shared::ThoError;
    use thoosa_shared::traits::DataId;
    use thoosa_named_derive::Named;
    use thoosa_shared::traits::WithId;
    use thoosa_common::tho_tensor::ThoTensor;
    use dashmap::DashMap;
    use mockall::predicate::always;
    use parking_lot::RwLock;
    use tch::Tensor;
    use tokio::runtime::Handle;
    use tokio::task::block_in_place;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use crate::tasks::cross_dims::MotifSlices;
    use crate::tasks::motif_loader::{MotifLoaderInput, MotifMeta};
    mock! {
        #[derive(Debug, Named)]
        pub MotifLoader {}


        impl Stage for MotifLoader {
            type Ctx = MockRunCtx;

            type In<'a> = MotifLoaderInput;
            type Out = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>;
        }

        impl OutId for MotifLoader {
            fn out_id<'a>(&self,input_id: &<<Self as Stage>::In<'a> as WithId>::ID,) -> <<MockMotifLoader as Stage>::Out as WithId>::ID;
        }

        // #[async_trait]
        impl AsyncStage for MotifLoader {}

        #[async_trait]
        impl Cached for MotifLoader {

            async fn cache() -> &'static DashMap<<<Self as Stage>::Out as WithId>::ID, Shared<BoxFuture<'static, Result<Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>>>, ThoError >>>>;
            async fn run_cached(
                self: Arc<Self>,
                input: Arc<DevicesWrap<<MockMotifLoader as Stage>::In<'static>>>,
                ctx: <MockMotifLoader as Stage>::Ctx,
            ) -> Result<Arc<DevicesWrap<<MockMotifLoader as Stage>::Out>>, ThoError>;
        }
    }

    impl thoosa_shared::traits::Named for MockMotifLoader {
        fn name() -> &'static str { "MockMotifLoader" }
    }

    pub struct MockMotifLoaderBuilder {
        mock:MockMotifLoader
    }

    impl MockMotifLoaderBuilder {
        pub fn with_run_cached<H>(mut self, h: H) -> Self
        where
            H: IntoIterator<Item=(MotifLoaderInput, (MotifMeta, Tensor))>,
        {
            let data_map: Arc<HashMap<MotifLoaderInput, Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>>>>> = Arc::new(h.into_iter().map(|(k, v)| {
                (k.clone(), Arc::new(DevicesWrap::wrap(DataWithExtra::new(ThoTensor::new(k.motif_id.into(), v.1).unwrap(), v.0), CacheArea::default())))
            }).collect());

            self.mock.expect_run_cached().returning(move |input, ctx| {
                let mut ctx = ctx;
                block_in_place(|| {
                    Handle::current().block_on(async {
                        input.with_data_ref(0,
                        ctx, None, |v: &MotifLoaderInput, _ctx| {
                            Ok(Arc::clone(data_map
                                .get(v)
                                .expect(&format!("no mock data for v={v:? } in MockMotifLoader"))))
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
            mock: MockMotifLoader::new(),
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
    use thoosa_shared::runner::common::{mock_run_ctx, RunCtx};
    use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
    use thoosa_shared::runner::cache_areas::CacheArea;
    use thoosa_shared::runner::common::runnable::AsyncStage;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::utils::lin_t;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::{TensorIdSliceIdx, ThoError};
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::DataId;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_store::MockStoreBuilder;
    use crate::tasks::motif_loader::{MotifLoader, MotifMeta, RandomSlicesLoader, SlicesLoaderInput};
    use thoosa_shared::runner::run_ctx_2::RunCtxWithPermits;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use crate::tasks::cross_dims::MotifSlices;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_motifs::distance::average_euclidean_distance;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_load_motifs() -> Result<(), ThoError> {
        set_max_level(Trace);
        GlobalCtx::init_devices(2);
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 10000)));
        let run_id = 1;

        let resource_a = Resource::new(1, 1, "tensor_a", 1000);
        let resource_b = Resource::new(2, 1, "tensor_b", 1000);
        let mut store = MockStoreBuilder::new()
            .with_list_motif_slices(10u32, None, None, None, Ok((0i64, run_id, vec![
                TensorIdSliceIdx(1, 100),
                TensorIdSliceIdx(1, 200),
                TensorIdSliceIdx(2, 100)
            ], HashMap::from([
                (1, resource_a.clone()),
                (2, resource_b.clone())
            ]))))
            .with_list_run_config_values(
                run_id,
                Ok(FindMotifsRunConfig::new_with_running_mean(Some(run_id), 100, 100, None, None).to_properties_iter(None).collect())
            )
            .build();

        let motif_loader = Arc::new(MotifLoader::new(
            Arc::new(store),
            Arc::new(
                mock_tensor_loader::build()
                    .with_out_id()
                    .with_run_cached([
                        (resource_a, lin_t(1000).view((1,-1))),
                        (resource_b, lin_t(1000).view((1,-1)))
                    ])
                    .finish(),
            ),
            &ws
        ));

        let ctx = Arc::new(RwLock::new(mock_run_ctx::default(0)));

        let num_devices = GlobalCtx::global().device_count();
        let cache_area = thoosa_shared::with_perf_log(
            || ctx.try_read_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout").cache_area(),
            "MotifLoader::test_load_motif/ctx_read_cache_area",
        );
        let input_id = Arc::new(DevicesWrap::wrap(10u32.into(), CacheArea::default()));

        let motif = motif_loader
            .run_cached(
                Arc::clone(&input_id),
                thoosa_shared::with_perf_log(
                    || ctx.try_write_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
                    "MotifLoader::test_load_motif/ctx_write_for_run_cached",
                )
                .branch(Some("motif_loader_test"), false, false),
            )
            .await?;

        let mut ctx_for_data = thoosa_shared::with_perf_log(
            || ctx.try_write_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
            "MotifLoader::test_load_motif/ctx_write_for_data",
        )
        .branch(Some("motif_loader_test_data"), false, false);
        let r = motif.with_data_ref(0,
            ctx_for_data, None, |t: &DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>, _ctx|{
                        let dist = 0.; //average_euclidean_distance(t.t_ref()).unwrap();

                        trace!("motif.t={} dist={dist}", t.data().t_ref());

                        Ok(true)
                    }).await?;

        assert_eq!(r, true);
        //TODO check data

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_load_random_slices() -> Result<(), ThoError> {
        set_max_level(Trace);
        GlobalCtx::init_devices(2);
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 10000)));
        let run_id = 1;
        let motif_id = 10u32;
        let motif_width = 100;

        let resource_a = Resource::new(1, 1, "tensor_a", 1000);
        let resource_b = Resource::new(2, 1, "tensor_b", 1000);
        let mut store = MockStoreBuilder::new()
            .with_list_random_motif_slices(motif_id, motif_width, None, Ok((0i64, run_id, vec![
                TensorIdSliceIdx(1, 100),
                TensorIdSliceIdx(1, 200),
                TensorIdSliceIdx(2, 100)
            ], HashMap::from([
                (1, resource_a.clone()),
                (2, resource_b.clone())
            ]))))
            .with_list_run_config_values(
                run_id,
                Ok(FindMotifsRunConfig::new_with_running_mean(Some(run_id), 100, motif_width, None, None).to_properties_iter(None).collect())
            )
            .with_get_motif_meta( [(motif_id, Ok((0usize, run_id)))])
            .build();

        let random_slices_loader = Arc::new(RandomSlicesLoader::new(
            Arc::new(store),
            Arc::new(
                mock_tensor_loader::build()
                    .with_out_id()
                    .with_run_cached([
                        (resource_a, lin_t(1000).view((1,-1))),
                        (resource_b, lin_t(1000).view((1,-1)))
                    ])
                    .finish(),
            ),
            &ws
        ));

        let ctx = Arc::new(RwLock::new(mock_run_ctx::default(0)));

        let num_devices = GlobalCtx::global().device_count();
        let cache_area = thoosa_shared::with_perf_log(
            || ctx.try_read_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout").cache_area(),
            "RandomSlicesLoader::test_load_random_slices/ctx_read_cache_area",
        );
        let input = Arc::new(DevicesWrap::wrap(SlicesLoaderInput{ motif_id: DataId::from(&motif_id), limit: 10 }, CacheArea::default()));

        let slices:Arc<DevicesWrap<ThoTensor<MotifSlices, DataId>>> = random_slices_loader
            .run(
                Arc::clone(&input),
                thoosa_shared::with_perf_log(
                    || ctx.try_write_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
                    "RandomSlicesLoader::test_load_random_slices/ctx_write_for_run",
                )
                .branch(Some("random_slices_loader"), false, false),
            )
            .await?;

        let mut ctx_for_data = thoosa_shared::with_perf_log(
            || ctx.try_write_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
            "RandomSlicesLoader::test_load_random_slices/ctx_write_for_data",
        )
        .branch(Some("random_slices_loader_data"), false, false);
        let r = slices.with_data_ref(
            0,
            ctx_for_data, None, |t: &ThoTensor<MotifSlices, DataId>, _ctx| {
                let dist = average_euclidean_distance(t.t_ref()).unwrap();

                trace!("random slices: \t{} dist={dist}", t.t_ref());

                Ok(true)
            },
        ).await?;
        // assert_eq!(r, true);
        //TODO check data

        Ok(())
    }

}
