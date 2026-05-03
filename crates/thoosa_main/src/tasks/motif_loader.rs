#![allow(unexpected_cfgs)]

use std::collections::{HashMap, HashSet};
use std::fmt::{format, Debug, Formatter};
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;
use async_trait::async_trait;
use log::{debug, error, info, trace, warn};
use parking_lot::RwLock;
use tch::Device::Cpu;
use tch::kind::FLOAT_CPU;
use tch::{IndexOp, Tensor};
use cached_derive::Cached;
use thoosa_named_derive::Named;
use thoosa_common::runner::common::{DropIf, OutId, RunCtx};
use thoosa_common::runner::common::runnable::{AsyncStage, ConsumeResult, Stage, SyncStage};

use mockall_double::double;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::tmp_cache::TmpCached;
use thoosa_common::{copy_ctx, debug_plot};
use thoosa_common::dims::{AnyDims, ChannelsSlicesData, Dims, MotifSlices, Slice};
use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::{mono_key, TensorIdSliceIdx, ThoError};
use thoosa_shared::traits::{CachableIdTr, Data, DataId, WithId};
use thoosa_store::Store;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::traits::MonoMetrics;
use thoosa_with_data::with_data;
use crate::tasks::cross_dims::{ChannelsData2, ChannelsDataVec, ChannelsMotifsSlices};
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use crate::tasks::tensor_loader::RawTensorLoader;
use ambassador::Delegate;

use rand::distr::{Alphanumeric, SampleString};
use tch::Kind::Float;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::ws::Workspace;
use thoosa_common::runner::common::cachable::{Cached, ConsumeData, DevicesWrap, WithDataRef};
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::traits::Adjustable;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::{Resource, ResourceId};
use crate::tasks::find_motifs::{config_to_stage, FilteredLoaderStage, LoaderStageChoice};
use crate::tasks::stages_seq::{ABStage, StageChainOption};
use crate::tasks::transforms::freq_filter::FreqFilter;
use thoosa_shared::comp_id::{MonoKey, SynKey};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelStep, MaxParallelAndDumpToCpuAdjustableContex, MAX_PARALLEL};
use thoosa_common::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
use thoosa_common::utils::tensor_digest;
use crate::tasks::stage_ids::{MOTIF_LOADER, MOTIF_LOADER_SYNC_STAGE};

#[derive(Debug, Clone, PartialEq, Eq, Hash, DummyData)]
pub struct MotifLoaderInput {
    pub id:SynKey,
    select_motifs_run_id: Option<Vec<RunId>>,
    motif_id:SynKey
}


#[cfg(test)]
impl From<u32> for MotifLoaderInput{
    fn from(value: u32) -> Self {
        MotifLoaderInput::find_motif_slices_only(mono_key!(value, "motif_id").into())
    }
}

impl MotifLoaderInput {
    pub fn all_slices(motif_id: DataId) -> Self {
        Self {
            id: mono_key!(motif_id.0, "motif_id").into(),
            motif_id: mono_key!(motif_id.0, "motif_id").into(),
            select_motifs_run_id: None //all slices loaded
        }
    }

    pub fn find_motif_slices_only(motif_id: SynKey) -> Self {
        Self {
            id: motif_id.clone(),
            motif_id,
            select_motifs_run_id: Some(vec![]) //only slices found during find_motifs loaded
        }
    }

    pub fn new(motif_id: SynKey, select_motifs_run_id: Option<Vec<RunId>>) -> Self {
        Self {
            id: select_motifs_run_id.iter().flat_map(|v|v).fold(motif_id.clone(), |a, v:&RunId|{
                &a + mono_key!(*v, "select_motifs_run_id")
            }),
            select_motifs_run_id,
            motif_id
        }// slices found during find_motifs + select_motifs_run_id
    }
}
impl Input for MotifLoaderInput {
}

//TODO composite id with select_motifs_run_id
impl WithId for MotifLoaderInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug, Cached, Named)]
pub struct MotifLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    id: MonoKey,
    store: Arc<Box<dyn Store>>,
    loader: Arc<L>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    sync_motif_loader: Arc<SyncMotifLoader<C>>,
    pd: PhantomData<C>,
}

impl<L, C> MotifLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    pub fn new(
        store: Arc<Box<dyn Store>>,
        loader: Arc<L>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
    ) -> Self {
        Self {
            id: mono_key!(MOTIF_LOADER, "MotifLoader"),
            store,
            loader,
            ws: Arc::clone(&ws),
            sync_motif_loader: Arc::new(SyncMotifLoader::new()),
            pd: Default::default(),
        }
    }
}

impl<L, C> OutId for MotifLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

#[derive(Debug, Clone)]
pub struct MotifMeta {
    pub channel:i64,
    pub find_motifs_run_id: RunId,
    pub window:u32
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Named)]
struct LoadedMotifResourceId {
    resource_id: ResourceId,
}

impl LoadedMotifResourceId {
    fn new(resource_id: ResourceId) -> Self {
        Self { resource_id }
    }
}

#[derive(Debug)]
struct LoadedMotifResource {
    id: LoadedMotifResourceId,
    resource_id: ResourceId,
    slices: Vec<u32>,
    raw_tensor: ThoTensor<Slice, SynKey>,
}

impl WithId for LoadedMotifResource {
    type ID = LoadedMotifResourceId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Data for LoadedMotifResource {
    type Metrics = <ThoTensor<Slice, SynKey> as Data>::Metrics;
    type M = CpuOrGpu<Self>;

    fn size(&self) -> Self::Metrics {
        <ThoTensor<Slice, SynKey> as Data>::size(&self.raw_tensor)
    }

    fn device(&self) -> usize {
        self.raw_tensor.device()
    }

    fn mem_size(&self) -> u64 {
        self.raw_tensor.mem_size()
    }
}

#[async_trait]
impl thoosa_shared::CopyToDevice for LoadedMotifResource {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        Ok(Self {
            id: self.id.clone(),
            resource_id: self.resource_id,
            slices: self.slices.clone(),
            raw_tensor: self.raw_tensor.copy_to_device(dst_device).await?,
        })
    }
}

impl Output for LoadedMotifResource {}

#[derive(Debug)]
struct SyncMotifLoaderLoadedInput {
    id: SynKey,
    motif_id: DataId,
    select_motifs_run_id: Option<Vec<RunId>>,
    find_motifs_run_id: RunId,
    channel: i64,
    window: u32,
    loaded_resources: Vec<LoadedMotifResource>,
    descr: &'static str,
}

impl SyncMotifLoaderLoadedInput {
    fn new(
        id: SynKey,
        motif_id: DataId,
        select_motifs_run_id: Option<Vec<RunId>>,
        find_motifs_run_id: RunId,
        channel: i64,
        window: u32,
        loaded_resources: Vec<LoadedMotifResource>,
        descr: &'static str,
    ) -> Self {
        Self { id, motif_id, select_motifs_run_id, find_motifs_run_id, channel, window, loaded_resources, descr }
    }
}

impl Input for SyncMotifLoaderLoadedInput {}

impl WithId for SyncMotifLoaderLoadedInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Data for SyncMotifLoaderLoadedInput {
    type Metrics = MonoMetrics<0, 1>;
    type M = CpuOrGpu<Self>;

    fn size(&self) -> Self::Metrics {
        self.loaded_resources.len().into()
    }

    fn device(&self) -> usize {
        self.loaded_resources.first().map(|resource| resource.raw_tensor.device()).unwrap_or(0)
    }

    fn mem_size(&self) -> u64 {
        self.loaded_resources.iter().map(|resource| resource.raw_tensor.mem_size()).sum()
    }
}

#[async_trait]
impl thoosa_shared::CopyToDevice for SyncMotifLoaderLoadedInput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        let mut loaded_resources = Vec::with_capacity(self.loaded_resources.len());
        for resource in &self.loaded_resources {
            loaded_resources.push(LoadedMotifResource {
                id: resource.id.clone(),
                resource_id: resource.resource_id,
                slices: resource.slices.clone(),
                raw_tensor: resource.raw_tensor.copy_to_device(dst_device).await?,
            });
        }
        Ok(Self {
            id: self.id.clone(),
            motif_id: self.motif_id.clone(),
            select_motifs_run_id: self.select_motifs_run_id.clone(),
            find_motifs_run_id: self.find_motifs_run_id,
            channel: self.channel,
            window: self.window,
            loaded_resources,
            descr: self.descr,
        })
    }
}

#[derive(Debug)]
struct SyncMotifLoaderInput {
    id: SynKey,
    motif_id: DataId,
    select_motifs_run_id: Option<Vec<RunId>>,
    find_motifs_run_id: RunId,
    channel: i64,
    window: u32,
    loaded_resources: TmpCached<LoadedMotifResource>,
    descr: &'static str,
}

impl SyncMotifLoaderInput {
    fn new(
        id: SynKey,
        motif_id: DataId,
        select_motifs_run_id: Option<Vec<RunId>>,
        find_motifs_run_id: RunId,
        channel: i64,
        window: u32,
        loaded_resources: TmpCached<LoadedMotifResource>,
        descr: &'static str,
    ) -> Self {
        Self { id, motif_id, select_motifs_run_id, find_motifs_run_id, channel, window, loaded_resources, descr }
    }
}

impl WithId for SyncMotifLoaderInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl ConsumeData for SyncMotifLoaderInput {
    type O = SyncMotifLoaderLoadedInput;

    fn consume<R, F, P>(
        self,
        to_device_idx: usize,
        ctx: P,
        _trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> impl std::future::Future<Output = ConsumeResult<Self::O, R>> + Send
    where
        F: FnOnce(Self::O, P) -> ConsumeResult<Self::O, R> + Send,
        P: RunCtx,
    {
        async move {
            let SyncMotifLoaderInput {
                id,
                motif_id,
                select_motifs_run_id,
                find_motifs_run_id,
                channel,
                window,
                loaded_resources,
                descr,
            } = self;

            let loaded_resources = match loaded_resources.take_all_recovering(to_device_idx).await {
                Ok(loaded_resources) => loaded_resources,
                Err(err) => {
                    return ConsumeResult::Failed {
                        input: SyncMotifLoaderLoadedInput::new(
                            id,
                            motif_id,
                            select_motifs_run_id,
                            find_motifs_run_id,
                            channel,
                            window,
                            err.data,
                            descr,
                        ),
                        error: err.error,
                    };
                }
            };

            let input = SyncMotifLoaderLoadedInput::new(
                id,
                motif_id,
                select_motifs_run_id,
                find_motifs_run_id,
                channel,
                window,
                loaded_resources,
                descr,
            );

            if to_device_idx == 0 {
                do_run(input, ctx)
            } else {
                let stream = GlobalCtx::global().copy_ctx(to_device_idx).await.acquire_stream().await;
                let _guard = thoosa_shims::stream::ThoStreamGuard::new(&stream).unwrap();
                let r = do_run(input, ctx);
                stream.synchronize().unwrap();
                r
            }
        }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SyncMotifLoaderLoadedInput::burp"))]
    fn burp(o: Self::O) -> impl std::future::Future<Output = Self> + Send {
        async move {
            let loaded_resources = TmpCached::new(&format!("tmp_loaded_motif_resources_burp_{}", o.id)).await;
            for resource in o.loaded_resources {
                loaded_resources.insert(resource);
            }
            SyncMotifLoaderInput::new(
                o.id,
                o.motif_id,
                o.select_motifs_run_id,
                o.find_motifs_run_id,
                o.channel,
                o.window,
                loaded_resources,
                o.descr,
            )
        }
    }
}


#[derive(Debug, Named)]
struct SyncMotifLoader<C>
where
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    pd: PhantomData<C>,
}

impl<C> SyncMotifLoader<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn new() -> Self {
        Self {
            id: (MOTIF_LOADER_SYNC_STAGE, "SyncMotifLoader").into(),
            pd: PhantomData,
        }
    }
}

impl<C> Adjustable for SyncMotifLoader<C>
where
    C: RunCtx + Adjustable + 'static,
{}

impl<C> OutId for SyncMotifLoader<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<C> Stage for SyncMotifLoader<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type In<'a> = SyncMotifLoaderLoadedInput;
    type Out = DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>;
    type Ctx = C;
}

impl<C> SyncStage for SyncMotifLoader<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn consume<'a>(self: Arc<Self>, input: Self::In<'a>, ctx: Self::Ctx) -> ConsumeResult<Self::In<'a>, Self::Out> {
        let motif = match build_motif_from_loaded_resources(
            input.motif_id.clone(),
            input.select_motifs_run_id.as_ref(),
            input.find_motifs_run_id,
            input.channel,
            input.window,
            &input.loaded_resources,
            ctx.device_idx(),
            input.descr,
        ) {
            Ok(motif) => motif,
            Err(error) => return ConsumeResult::Failed { input, error },
        };
        ConsumeResult::Ok(DataWithExtra::new(
            motif,
            MotifMeta {
                channel: input.channel,
                find_motifs_run_id: input.find_motifs_run_id,
                window: input.window,
            },
        ))
    }
}

impl<L, C> Stage for MotifLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type In<'a> = MotifLoaderInput;
    type Out = DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>;

    type Ctx = C;
}

impl<L, C> AsyncStage for MotifLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
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
        let input_id = input.id().clone();

        let (motif_id, select_motifs_run_id) = with_data!(ctx.branch(Some("motif_loader_input"), false), input; |_ctx, input: &MotifLoaderInput| {
            Ok::<_, ThoError>((input.motif_id.clone(), input.select_motifs_run_id.clone()))
        })?;

        let motif_id_value = motif_id.vals[0].to_data_id();
        let select_motifs_run_id_for_key = select_motifs_run_id.clone();
        let (channel, find_motifs_run_id, slices, resources_map) = self
            .store
            .list_motif_slices(motif_id_value.0, select_motifs_run_id, Some(1000), None)
            .await?;

        let find_motifs_run_id = if self.store.run_config_descr(find_motifs_run_id).await?.r#type == "FindMotifsRunConfig" {
            find_motifs_run_id
        } else {
            let find_motifs_run_id = self.store.get_input_run_ids(find_motifs_run_id, Some("FindMotifsRunConfig")).await?;
            // TODO check if all find run configs have same properties - for merged motifs
            *find_motifs_run_id.first().ok_or_else(|| ThoError::new(format!("MotifLoader::run: no FindMotifsRunConfig input for motif_id={motif_id}")))? 
        };

        debug!("MotifLoader::run: motif_id={motif_id} channel={channel} ctx={ctx:?} find_motifs_run_id={find_motifs_run_id}");

        let find_motifs_run_config = FindMotifsRunConfig::from_properties_map_with_id(
            find_motifs_run_id,
            &self.store.list_run_config_values(find_motifs_run_id).await?,
        )?;

        let window = find_motifs_run_config.window;
        let resource_ids_slices = slices.iter().fold(HashMap::new(), |mut acc, slice| {
            acc.entry(slice.0).or_insert_with(Vec::new).push(slice.1);
            acc
        });
        assert!(!resource_ids_slices.is_empty());
        let configurable_loader = config_to_stage(&self.loader, &find_motifs_run_config, &self.ws)?;
        let loaded_resources = load_motif_resource_tensors(
            motif_id_value.clone(),
            channel,
            find_motifs_run_id,
            resources_map,
            resource_ids_slices,
            configurable_loader,
            ctx.branch(Some("motif_loader_load_resource_tensors"), false),
            "motif loader",
        )
        .await?;
        let loaded_resources_name = loaded_resources.registered_name();

        let sync_input = SyncMotifLoaderInput::new(
            input_id.clone(),
            motif_id_value,
            select_motifs_run_id_for_key,
            find_motifs_run_id,
            channel,
            window,
            loaded_resources,
            "motif loader",
        );
        let required_loaded_resource_ids = HashSet::<&dyn CachableIdTr>::new();
        let required_ids = HashMap::from([(loaded_resources_name, &required_loaded_resource_ids)]);
        let oom_handler: OOMHandler<'_, '_, _, _, _, _, ()> = OOMHandler::new(
            vec![
                Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::FIFTY))),
                Adjusters::RunContext(Box::new(DumpToCpu::new("motif_loader"))),
            ],
            None,
        );
        let motif = oom_handler
        .consume(
            &self.sync_motif_loader,
            sync_input,
            ctx.branch(Some("motif_loader_sync"), true),
            Contexts::new(
                NoopInputContext(None),
                NoopRunnableContext {},
                MaxParallelAndDumpToCpuAdjustableContex::new(MaxParallelAdjustContextImpl::new(), DumpToCpuContextImpl::new(&required_ids)),
            ),
        )
        .await?;

        Ok(Arc::new(DevicesWrap::wrap(motif)))
    }
}
///To load slices not belonging to a motif
#[derive(Debug)]
pub struct RandomSlicesLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    store: Arc<Box<dyn Store>>,
    loader: Arc<L>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    sync_motif_loader: Arc<SyncMotifLoader<C>>,
    pd: PhantomData<C>,
}

impl <L,C> RandomSlicesLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    pub fn new(store: Arc<Box<dyn Store>>, loader: Arc<L>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Self {
        Self { store, loader, ws:Arc::clone(&ws), sync_motif_loader: Arc::new(SyncMotifLoader::new()), pd: PhantomData::default() }
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
    C: RunCtx + Adjustable + 'static,
{
    type In<'a> = SlicesLoaderInput;
    type Out = ThoTensor<MotifSlices, SynKey>;

    type Ctx = C;
}



impl<L, C> AsyncStage for RandomSlicesLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
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

        let (motif_id, limit) = with_data!(ctx.branch(Some("random_slices_input"), false), input; |_ctx, input: &SlicesLoaderInput| {
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
        let (channel, find_motifs_run_id, slices, resources_map) = self.store.list_random_slices_out_of_motif(motif_id_value, find_motifs_run_config.window, None, input_len_adjuster, Some(limit as usize)).await?;


        let motif_id_value = motif_id.clone();
        let resource_ids_slices = slices.iter().fold(HashMap::new(), |mut acc, slice| {
            acc.entry(slice.0).or_insert_with(Vec::new).push(slice.1);
            acc
        });
        assert!(!resource_ids_slices.is_empty());

        debug!("RandomSlicesLoader::run: motif_id={motif_id} find_motifs_run_config={find_motifs_run_config:?} device_idx={}", ctx.device_idx());

        let configurable_loader = config_to_stage(&self.loader, &find_motifs_run_config, &self.ws)?;
        let loaded_resources = load_motif_resource_tensors(
            motif_id_value.clone(),
            channel,
            find_motifs_run_id,
            resources_map,
            resource_ids_slices,
            configurable_loader,
            ctx.branch(Some("random_loader_load_resource_tensors"), false),
            "random loader",
        )
        .await?;
        let loaded_resources_name = loaded_resources.registered_name();
        let sync_input = SyncMotifLoaderInput::new(
            mono_key!(*motif_id_value, "random_slices_loader_input").into(),
            motif_id_value,
            None,
            find_motifs_run_id,
            channel,
            find_motifs_run_config.window,
            loaded_resources,
            "random loader",
        );
        let required_loaded_resource_ids = HashSet::<&dyn CachableIdTr>::new();
        let required_ids = HashMap::from([(loaded_resources_name, &required_loaded_resource_ids)]);
        let oom_handler: OOMHandler<'_, '_, _, _, _, _, ()> = OOMHandler::new(
            vec![
                Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::FIFTY))),
                Adjusters::RunContext(Box::new(DumpToCpu::new("random_slices_loader"))),
            ],
            None,
        );
        let motif = oom_handler
        .consume(
            &self.sync_motif_loader,
            sync_input,
            ctx.branch(Some("random_slices_loader_sync"), true),
            Contexts::new(
                NoopInputContext(None),
                NoopRunnableContext {},
                MaxParallelAndDumpToCpuAdjustableContex::new(MaxParallelAdjustContextImpl::new(), DumpToCpuContextImpl::new(&required_ids)),
            ),
        )
        .await?;

        #[cfg(tho_debug_plots)]
        debug_plot!(motif.data().as_ref(), "random_motif_{motif_id}")?;

        let arc = Arc::new(DevicesWrap::wrap(ThoTensor::new(mono_key!("random", "random") + motif.id(), motif.data().t_ref().shallow_clone())?));

        Ok(arc)
    }
}

async fn load_motif_resource_tensors<L, C>(
    motif_id: DataId,
    channel: i64,
    find_motifs_run_id: u32,
    resources_map: HashMap<ResourceId, Resource>,
    resource_ids_slices: HashMap<ResourceId, Vec<u32>>,
    configurable_loader: Arc<LoaderStageChoice<L, C>>,
    mut ctx: C,
    descr: &str,
) -> Result<TmpCached<LoadedMotifResource>, ThoError>
where
    C: 'static + RunCtx + Adjustable,
    L: 'static + Cached + OutId + for<'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C>,
{
    let loaded_resources = TmpCached::new(&format!("tmp_loaded_motif_resources_{descr}")).await;
    let mut loaded_resources_count = 0usize;
    for (resource_id, slices) in resource_ids_slices {
        debug!("load_motif_resource_tensors: motif_id={motif_id} resource_id={resource_id} slices={slices:?} descr={descr}");

        let resource = resources_map.get(&resource_id).cloned().expect(&format!("no resource with id={resource_id}"));
        let resource_cache = Arc::new(DevicesWrap::wrap(resource));
        let raw_tensor = Arc::clone(&configurable_loader)
            .run_cached(resource_cache, ctx.branch(Some(&format!("motif_loader_resource_{resource_id}")), false))
            .await?;
        let channel_tensor = with_data!(ctx.branch(Some(&format!("motif_loader_resource_{resource_id}_clone")), false), raw_tensor; |_ctx, raw_tt:&ThoTensor<ChannelsData2, SynKey>| {

            let src = raw_tt.t_ref().i((channel, ..));
            let mut dst = src.f_zeros_like()?;
            dst.f_copy_(&src)?;

            ThoTensor::new(raw_tt.id().clone(), dst)
        })?;
        drop(raw_tensor);
        loaded_resources.insert(LoadedMotifResource {
            id: LoadedMotifResourceId::new(resource_id),
            resource_id,
            slices,
            raw_tensor: channel_tensor,
        });
        loaded_resources_count += 1;
    }
    debug!("load_motif_resource_tensors: motif_id={motif_id} find_motifs_run_id={find_motifs_run_id} loaded_resources={loaded_resources_count} descr={descr}");
    Ok(loaded_resources)
}

fn build_motif_from_loaded_resources(
    motif_id: DataId,
    select_motifs_run_id: Option<&Vec<RunId>>,
    find_motifs_run_id: u32,
    channel: i64,
    window: u32,
    loaded_resources: &[LoadedMotifResource],
    device_idx: usize,
    descr: &str,
) -> Result<ThoTensor<MotifSlices, SynKey>, ThoError> {
    let total_slices = loaded_resources.iter().map(|resource| resource.slices.len()).sum::<usize>() as i64;
    let motif_t = ThoTensor::<MotifSlices, DataId>::zeros(
        motif_id.clone(),
        [total_slices, window as i64],
        (Float, ThoDevice::from(device_idx).into()),
    )?;
    let mut slice_i = 0i64;

    for resource in loaded_resources {
        let size = resource.raw_tensor.t_ref().size();

        debug!("build_motif_slices_from_loaded_resources: motif_id={motif_id} raw_tensor={:?}", resource.resource_id);
        // debug!("build_motif_slices_from_loaded_resources: motif_id={motif_id} raw_tensor={:?} tensor_digest={:?}", resource.resource_id, tensor_digest(resource.raw_tensor.t_ref()));

        for &slice in &resource.slices {

            if (slice + window) as i64 > size[0] {
                warn!("build_motif_slices_from_loaded_resources: motif_id={motif_id} slice={slice:?}+{window} is bigger than size={size:?} for resource_id={} in find_motifs_run_id={find_motifs_run_id} in {descr}", resource.resource_id);

            } else {
                debug!("build_motif_slices_from_loaded_resources: motif_id={motif_id} slice_i={slice_i} slice={slice:?} size={size:?} resource_id={} descr={descr}", resource.resource_id);
                motif_t.t.f_get(slice_i)?.f_copy_(&resource.raw_tensor.t.f_i(( slice as i64..(slice + window) as i64))?)?;
                slice_i += 1;
            }
        }
    }
    /*
       select_motifs_run_id should not be used if SelectMotifsRunConfig doesn't contain extend_config
       let motif_id = select_motifs_run_id
        .map(|run_ids| {
            run_ids.iter().fold(mono_key!(motif_t.id().0, "motif").into(), |acc: SynKey, run_id| {
                &acc + mono_key!(*run_id, "select_motifs_run_id")
            })
        })
     */
    // TODO motif can contain either only slices found during initial 'find motifs' run or also slices found during 'select motifs' ran with 'extend' config. This should be reflected in motif id
    // let tensor = motif_t.t.f_i((0..slice_i, ..))?;
    // debug!("build_motif_slices_from_loaded_resources: motif_id={motif_id:?} tensor_digest={:?}", tensor_digest(&tensor));
    // ThoTensor::new(mono_key!(*motif_id, descr).into(), tensor) //f_i - if there were skipped slices. TODO error if it's emtpy?
    ThoTensor::new(mono_key!(*motif_id, descr).into(), motif_t.t.f_i((0..slice_i, ..))?) //f_i - if there were skipped slices. TODO error if it's emtpy?


}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="load_random_slices"))]
pub(crate) async fn load_random_slices<L, C>(
    find_motifs_run_id: u32,
    resources_map: HashMap<ResourceId, Resource>,
    slices: Vec<TensorIdSliceIdx>,
    store: Arc<Box<dyn Store>>,
    loader: Arc<L>,
    mut ctx: C,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    descr: &str
) -> Result<(u32, TmpCached<ThoTensor<ChannelsMotifsSlices, SynKey>>), ThoError>
where
    C: 'static + RunCtx + Adjustable,
    L: 'static + Cached + OutId + for<'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C>,
{
    let device_idx = ctx.device_idx();
    debug!("load_random_slices: find_motifs_run_id={find_motifs_run_id} slices={slices:?} descr={descr}");

    let find_motifs_run_config = FindMotifsRunConfig::from_properties_map_with_id(
        find_motifs_run_id,
        &store.list_run_config_values(find_motifs_run_id).await?,
    )?;

    let resource_ids_slices = slices
        .iter()
        .fold(HashMap::new(), |mut acc, slice| {
            acc.entry(slice.0.clone()).or_insert_with(Vec::new).push(slice.1);
            acc
        });

    assert!(!resource_ids_slices.is_empty());

    let window = find_motifs_run_config.window;
    let channels = thoosa_shared::with_perf_log(
        || ws.try_read_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
        "load_random_slices/read_ws",
    )
    .channels() as i64;
    assert!(channels > 0, "load_random_slices: channels must be > 0");

    debug!("load_random_slices:  find_motifs_run_config={find_motifs_run_config:?} device_idx={device_idx} channels={channels}");

    let tmp_cache = TmpCached::new(&format!("tmp_random_slices_{descr}")).await;

    let configurable_loader = config_to_stage(&loader, &find_motifs_run_config, ws)?;
    for (resource_id, slices) in resource_ids_slices {
        debug!("load_random_slices: resource_id={resource_id} slices={slices:?} in_loop_0");

        let resource = resources_map.get(&resource_id).cloned().expect(&format!("no resource with id={resource_id}"));
        let resource_cache = Arc::new(DevicesWrap::wrap(resource));
        let raw_tensor = Arc::clone(&configurable_loader)
            .run_cached(resource_cache, ctx.branch(Some(&format!("motif_loader_resource_{resource_id}")), false))
            .await?;
        let size = raw_tensor.size();
        debug!("load_random_slices: resource_id={resource_id} slices={slices:?} in_loop_1");

        let resource_slices = with_data!(ctx.branch(Some(&format!("motif_loader_resource_{resource_id}")), false), raw_tensor; |_ctx, raw_tt:&ThoTensor<ChannelsData2, SynKey>| {
            let resource_id = resource_id;
            let resource_slice_count = slices.len() as i64;
            let motif_t = ThoTensor::<AnyDims, DataId>::zeros(
                DataId(resource_id),
                [channels, resource_slice_count, window as i64],
                (Float, ThoDevice::from(ctx.device_idx()).into()),
            )?;
            let mut slice_i: i64 = 0;
            for slice in slices {
                let in_range = MonoMetrics::<1,2>::from([channels, (slice + window) as i64]) <= *size;
                if !in_range {
                    warn!("load_random_slices:  slice={slice:?}+{window} is bigger than size={size:?} for resource_id={resource_id} in find_motifs_run_id={find_motifs_run_id} in {descr}");
                } else {
                    debug!("load_random_slices:  slice_i={slice_i} slice={slice:?} size={size:?}");
                    let src = raw_tt.t.f_i((0..channels, slice as i64..(slice + window) as i64))?;
                    let mut dst = motif_t.t.f_i((0..channels, slice_i, ..))?;
                    dst.f_copy_(&src)?;
                    slice_i += 1;
                }
            }

            debug!("load_random_slices: resource_id={resource_id} with_data_done");

            if slice_i == 0 {
                Ok::<_, ThoError>(None)
            } else {
                Ok::<_, ThoError>(Some(ThoTensor::new(
                    mono_key!(resource_id, format!("random_slices_resource_id={resource_id}")).into(),
                    motif_t.t.f_i((.., 0..slice_i, ..))?,
                )?))
            }
        })?;

        if let Some(resource_slices) = resource_slices {
            tmp_cache.insert(resource_slices);
        }
    }
    debug!("load_random_slices: find_motifs_run_id={find_motifs_run_id} done_0");
    debug!("load_random_slices: find_motifs_run_id={find_motifs_run_id} done_1");

    Ok((window, tmp_cache))
}

#[cfg(test)]
pub mod mock_motif_loader {
    use std::collections::{HashMap, HashSet};
    use async_trait::async_trait;
    use futures_util::future::{BoxFuture, Shared};
    use mockall::mock;
    use std::sync::Arc;
    use thoosa_common::runner::common::cachable::{Cached, ConsumeData, DevicesWrap, WithDataRef};
    use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex};
use thoosa_common::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
    use thoosa_common::runner::common::{MockRunCtx, OutId, RunCtx};
    use thoosa_common::runner::common::runnable::{AsyncStage, ConsumeResult, Stage, SyncStage};
    use thoosa_shared::ThoError;
    use thoosa_shared::traits::DataId;
    use thoosa_shared::traits::WithId;
    use thoosa_common::tho_tensor::ThoTensor;
    use dashmap::DashMap;
    use tch::Tensor;
    use tokio::runtime::Handle;
    use tokio::task::block_in_place;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_common::dims::MotifSlices;
    use thoosa_shared::comp_id::SynKey;
    use crate::tasks::motif_loader::{MotifLoaderInput, MotifMeta};
    mock! {
        #[derive(Debug, Named)]
        pub MotifLoader {}


        impl Stage for MotifLoader {
            type Ctx = MockRunCtx;

            type In<'a> = MotifLoaderInput;
            type Out = DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>;
        }

        impl OutId for MotifLoader {
            fn out_id<'a>(&self,input_id: &<<Self as Stage>::In<'a> as WithId>::ID,) -> <<MockMotifLoader as Stage>::Out as WithId>::ID;
        }

        // #[async_trait]
        impl AsyncStage for MotifLoader {}

        #[async_trait]
        impl Cached for MotifLoader {

            async fn cache() -> &'static DashMap<<<Self as Stage>::Out as WithId>::ID, Shared<BoxFuture<'static, Result<Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>, ThoError >>>>;
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
            let data_map: Arc<HashMap<MotifLoaderInput, Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>>> = Arc::new(h.into_iter().map(|(k, v)| {
                (k.clone(), Arc::new(DevicesWrap::wrap(DataWithExtra::new(ThoTensor::new(k.motif_id, v.1).unwrap(), v.0))))
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
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use log::LevelFilter::Trace;
    use log::{set_max_level, trace};
    use parking_lot::RwLock;
    use tch::Tensor;
    use thoosa_common::runner::common::{mock_run_ctx, RunCtx};
    use thoosa_common::runner::common::cachable::{Cached, ConsumeData, DevicesWrap, WithDataRef};
    use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex};
use thoosa_common::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::global_ctx::{mock_global_ctx, GlobalCtx};
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::utils::lin_t;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::{TensorIdSliceIdx, ThoError};
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::DataId;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_store::MockStoreBuilder;
    use crate::tasks::motif_loader::{MotifLoader, MotifMeta, RandomSlicesLoader, SlicesLoaderInput};
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use thoosa_common::dims::MotifSlices;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_motifs::distance::average_euclidean_distance;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_store::RunConfigRow;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_load_motifs() -> Result<(), ThoError> {
        // set_max_level(Trace);
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 10000)));
        let find_motifs_run_id = 1;
        let motif_source_run_id = 2;

        let resource_a = Resource::new(1, 1, "tensor_a", 1000);
        let resource_b = Resource::new(2, 1, "tensor_b", 1000);
        let mut store = MockStoreBuilder::new()
            .with_list_motif_slices(10u32, Some(vec![]), None, None, Ok((0i64, motif_source_run_id, vec![
                TensorIdSliceIdx(1, 100),
                TensorIdSliceIdx(1, 200),
                TensorIdSliceIdx(2, 100)
            ], HashMap::from([
                (1, resource_a.clone()),
                (2, resource_b.clone())
            ]))))
            .with_run_config_descr([(
                motif_source_run_id,
                Ok(RunConfigRow {
                    id: motif_source_run_id,
                    r#type: "SelectMotifsRunConfig".to_string(),
                    descr: "test select".to_string(),
                    run_on: "2026-01-01 00:00:00".to_string(),
                }),
            )])
            .with_get_input_run_ids([(motif_source_run_id, Ok(vec![find_motifs_run_id]))])
            .with_list_run_config_values(
                find_motifs_run_id,
                Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_id), 100, 100, None, None).to_properties_iter(None).collect())
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

        // TODO: replace the test-only u32 -> MotifLoaderInput conversion with explicit constructors.
        let input_id = Arc::new(DevicesWrap::wrap(10u32.into()));

        let motif = motif_loader
            .run_cached(
                Arc::clone(&input_id),
                thoosa_shared::with_perf_log(
                    || ctx.try_write_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
                    "MotifLoader::test_load_motif/ctx_write_for_run_cached",
                )
                .branch(Some("motif_loader_test"), false),
            )
            .await?;

        let mut ctx_for_data = thoosa_shared::with_perf_log(
            || ctx.try_write_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
            "MotifLoader::test_load_motif/ctx_write_for_data",
        )
        .branch(Some("motif_loader_test_data"), false);
        let r = motif.with_data_ref(0, ctx_for_data, None, |t: &DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, _ctx|{
            let dist = 0.; //average_euclidean_distance(t.t_ref()).unwrap();

            trace!("motif.t={} dist={dist}", t.data().t_ref());

            Ok(true)
        }).await?;

        assert_eq!(r, true);
        //TODO check data

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 32)]
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
            .with_list_random_slices_out_of_motif(motif_id, motif_width, None, Ok((0i64, run_id, vec![
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
        let input = Arc::new(DevicesWrap::wrap(SlicesLoaderInput{ motif_id: DataId::from(&motif_id), limit: 10 }));

        let slices:Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>> = random_slices_loader
            .run(
                Arc::clone(&input),
                thoosa_shared::with_perf_log(
                    || ctx.try_write_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
                    "RandomSlicesLoader::test_load_random_slices/ctx_write_for_run",
                )
                .branch(Some("random_slices_loader"), false),
            )
            .await?;

        let mut ctx_for_data = thoosa_shared::with_perf_log(
            || ctx.try_write_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
            "RandomSlicesLoader::test_load_random_slices/ctx_write_for_data",
        )
        .branch(Some("random_slices_loader_data"), false);
        let r = slices.with_data_ref(
            0,
            ctx_for_data, None, |t: &ThoTensor<MotifSlices, SynKey>, _ctx| {
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
