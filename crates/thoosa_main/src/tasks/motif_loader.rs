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
use serde::{Deserialize, Serialize};
use tch::Device::Cpu;
use tch::kind::FLOAT_CPU;
use tch::{IndexOp, Tensor};
use cached_derive::Cached;
use thoosa_named_derive::Named;
use thoosa_common::runner::common::{DropIf, MaxParallelCtx, OutId, RunCtx};
use thoosa_common::runner::common::runnable::{AsyncStage, ConsumeResult, Stage, SyncStage};

use thoosa_common::runner::tmp_cache::TmpCached;
use thoosa_common::ipc::codec::{IpcCodec, JsonIpcCodec};
use thoosa_common::ipc::cuda_wokrer::WorkerStageOutput;
use thoosa_common::ipc::json_ser::SyncStageIpcWrapper;
use thoosa_common::{copy_ctx, debug_plot};
use thoosa_common::dims::{ChannelsData, Dims, MotifSlices};
use crate::tasks::find_motifs_run_config::FindMotifsRunConfig;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::{mono_key, TensorIdSliceIdx, ThoError};
use thoosa_shared::traits::{CachableIdTr, Data, DataId, WithId};
use thoosa_store::Store;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::traits::MonoMetrics;
use crate::tasks::cross_dims::{ChannelsDataVec, ChannelsMotifsSlices};
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use crate::tasks::tensor_loader::RawTensorLoader;
use ambassador::Delegate;

use rand::distr::{Alphanumeric, SampleString};
use tch::Kind::Float;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::ws::Workspace;
use thoosa_common::runner::common::cachable::{BorrowRef, SelectRef, Cached, ConsumeData, DevicesWrap, ThoDataRef};
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::traits::Adjustable;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::{Resource, ResourceId};
use crate::tasks::find_motifs::{config_to_stage, FilteredLoaderStage, LoaderStageChoice};
use thoosa_common::stages_seq::{ABStage, StageChainOption};
use crate::tasks::transforms::freq_filter::FreqFilter;
use thoosa_shared::comp_id::{MonoKey, SynKey};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep, MAX_PARALLEL};
use thoosa_common::runner::oom_handler::{NoContexts, NoInputAdjuster, NoRunCtxAdjuster, NoTaskAdjuster, run_ctx_adjuster, Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler, WithRunCtxContext};
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::utils::load_run_config;
use crate::tasks::stage_ids::{MOTIF_LOADER, MOTIF_LOADER_SYNC_STAGE, RAW_TENSOR_LOADER};
use crate::tasks::stft::SyncStftSelector;
use thoosa_shared::tracked_await::track_await;
use thoosa_proxy_derive::Proxy;

#[derive(Debug, Clone, PartialEq, Eq, Hash, DummyData)]
pub struct MotifLoaderInput {
    pub id:SynKey,
    select_motifs_run_config_id: Option<Vec<RunConfigId>>,
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
            select_motifs_run_config_id: None //all slices loaded
        }
    }

    pub fn find_motif_slices_only(motif_id: SynKey) -> Self {
        Self {
            id: motif_id.clone(),
            motif_id,
            select_motifs_run_config_id: Some(vec![]) //only slices found during find_motifs loaded
        }
    }

    pub fn new(motif_id: SynKey, select_motifs_run_config_id: Option<Vec<RunConfigId>>) -> Self {
        Self {
            id: select_motifs_run_config_id.iter().flat_map(|v|v).fold(motif_id.clone(), |a, v:&RunConfigId|{
                &a + mono_key!(*v, "select_motifs_run_config_id")
            }),
            select_motifs_run_config_id,
            motif_id
        }// slices found during find_motifs + select_motifs_run_config_id
    }
}
impl Input for MotifLoaderInput {
}

//TODO composite id with select_motifs_run_config_id
impl WithId for MotifLoaderInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug, Cached, Named)]
pub struct MotifLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    id: MonoKey,
    store: Arc<Box<dyn Store>>,
    loader: Arc<L>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    pd: PhantomData<C>,
}

impl<L, C> MotifLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
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
            pd: Default::default(),
        }
    }
}

impl<L, C> OutId for MotifLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifMeta {
    pub channel:i64,
    pub find_motifs_run_config_id: RunConfigId,
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
    raw_tensor: ThoTensor<MotifSlices, SynKey>,
}

impl WithId for LoadedMotifResource {
    type ID = LoadedMotifResourceId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Data for LoadedMotifResource {
    type Metrics = <ThoTensor<MotifSlices, SynKey> as Data>::Metrics;
    type M = CpuOrGpu<Self>;

    fn size(&self) -> Self::Metrics {
        <ThoTensor<MotifSlices, SynKey> as Data>::size(&self.raw_tensor)
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
            raw_tensor: track_await(self.raw_tensor.copy_to_device(dst_device)).await?,
        })
    }
}

impl Output for LoadedMotifResource {}

#[derive(Debug)]
pub struct SyncMotifLoaderInput {
    id: SynKey,
    motif_id: DataId,
    select_motifs_run_config_id: Option<Vec<RunConfigId>>,
    find_motifs_run_config_id: RunConfigId,
    channel: i64,
    window: u32,
    loaded_resources: Vec<LoadedMotifResource>,
    descr: String,
}

impl SyncMotifLoaderInput {
    fn new(
        id: SynKey,
        motif_id: DataId,
        select_motifs_run_config_id: Option<Vec<RunConfigId>>,
        find_motifs_run_config_id: RunConfigId,
        channel: i64,
        window: u32,
        loaded_resources: Vec<LoadedMotifResource>,
        descr: String,
    ) -> Self {
        Self { id, motif_id, select_motifs_run_config_id, find_motifs_run_config_id, channel, window, loaded_resources, descr }
    }
}

impl Input for SyncMotifLoaderInput {}

impl WithId for SyncMotifLoaderInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Data for SyncMotifLoaderInput {
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
impl thoosa_shared::CopyToDevice for SyncMotifLoaderInput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        let mut loaded_resources = Vec::with_capacity(self.loaded_resources.len());
        for resource in &self.loaded_resources {
            loaded_resources.push(LoadedMotifResource {
                id: resource.id.clone(),
                resource_id: resource.resource_id,
                slices: resource.slices.clone(),
                raw_tensor: track_await(resource.raw_tensor.copy_to_device(dst_device)).await?,
            });
        }
        Ok(Self {
            id: self.id.clone(),
            motif_id: self.motif_id.clone(),
            select_motifs_run_config_id: self.select_motifs_run_config_id.clone(),
            find_motifs_run_config_id: self.find_motifs_run_config_id,
            channel: self.channel,
            window: self.window,
            loaded_resources,
            descr: self.descr.clone(),
        })
    }
}

#[derive(Debug)]
struct SyncMotifLoaderInputWrapper {
    id: SynKey,
    motif_id: DataId,
    select_motifs_run_config_id: Option<Vec<RunConfigId>>,
    find_motifs_run_config_id: RunConfigId,
    channel: i64,
    window: u32,
    loaded_resources: TmpCached<LoadedMotifResource>,
    descr: String,
}

impl SyncMotifLoaderInputWrapper {
    fn new(
        id: SynKey,
        motif_id: DataId,
        select_motifs_run_config_id: Option<Vec<RunConfigId>>,
        find_motifs_run_config_id: RunConfigId,
        channel: i64,
        window: u32,
        loaded_resources: TmpCached<LoadedMotifResource>,
        descr: String,
    ) -> Self {
        Self { id, motif_id, select_motifs_run_config_id, find_motifs_run_config_id, channel, window, loaded_resources, descr }
    }
}

impl WithId for SyncMotifLoaderInputWrapper {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl ConsumeData for SyncMotifLoaderInputWrapper {
    type O = SyncMotifLoaderInput;

    fn consume<R: Send, F, P>(
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
            let SyncMotifLoaderInputWrapper {
                id,
                motif_id,
                select_motifs_run_config_id,
                find_motifs_run_config_id,
                channel,
                window,
                loaded_resources,
                descr,
            } = self;

            let loaded_resources = match track_await(loaded_resources.take_all_recovering(to_device_idx)).await {
                Ok(loaded_resources) => loaded_resources,
                Err(err) => {
                    return ConsumeResult::Failed {
                        input: SyncMotifLoaderInput::new(
                            id,
                            motif_id,
                            select_motifs_run_config_id,
                            find_motifs_run_config_id,
                            channel,
                            window,
                            err.data,
                            descr,
                        ),
                        error: err.error,
                    };
                }
            };

            let input = SyncMotifLoaderInput::new(
                id,
                motif_id,
                select_motifs_run_config_id,
                find_motifs_run_config_id,
                channel,
                window,
                loaded_resources,
                descr,
            );

            do_run(input, ctx)
        }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SyncMotifLoaderLoadedInput::burp"))]
    fn burp(o: Self::O) -> impl std::future::Future<Output = Self> + Send {
        async move {
            let loaded_resources = track_await(TmpCached::new(&format!("tmp_loaded_motif_resources_burp_{}", o.id))).await;
            for resource in o.loaded_resources {
                loaded_resources.insert(resource);
            }
            SyncMotifLoaderInputWrapper::new(
                o.id,
                o.motif_id,
                o.select_motifs_run_config_id,
                o.find_motifs_run_config_id,
                o.channel,
                o.window,
                loaded_resources,
                o.descr,
            )
        }
    }
}


pub(crate) const SYNC_MOTIF_LOADER_ENDPOINT: &str = "SyncMotifLoader";

#[derive(Debug, Serialize, Deserialize)]
pub struct SyncMotifLoaderOutput {
    motif: ThoTensor<MotifSlices, SynKey>,
    motif_meta: MotifMeta,
}

impl SyncMotifLoaderOutput {
    fn new(motif: ThoTensor<MotifSlices, SynKey>, motif_meta: MotifMeta) -> Self {
        Self { motif, motif_meta }
    }

    fn into_data_with_extra(self) -> DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta> {
        DataWithExtra::new(self.motif, self.motif_meta)
    }
}

impl WithId for SyncMotifLoaderOutput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        self.motif.id()
    }
}

impl Output for SyncMotifLoaderOutput {}

impl WorkerStageOutput for SyncMotifLoaderOutput {
    fn encode_from_worker<C: IpcCodec>(self, all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError> {
        //TODO fix encode/decode/encode
        let motif = C::decode::<ThoTensor<MotifSlices, SynKey>>(&<ThoTensor<MotifSlices, SynKey> as WorkerStageOutput>::encode_from_worker::<C>(self.motif, all_cuda_tensors, tho_device)?)?;
        C::encode(&Self::new(motif, self.motif_meta))
    }
}

#[derive(Debug, Serialize)]
struct SyncMotifLoaderResourceIpcRef<'a> {
    resource_id: ResourceId,
    slices: &'a [u32],
    raw_tensor: &'a ThoTensor<MotifSlices, SynKey>,
}

#[derive(Debug, Serialize)]
struct SyncMotifLoaderIpcInputRef<'a> {
    id: &'a SynKey,
    motif_id: u32,
    select_motifs_run_config_id: &'a Option<Vec<RunConfigId>>,
    find_motifs_run_config_id: RunConfigId,
    channel: i64,
    window: u32,
    loaded_resources: Vec<SyncMotifLoaderResourceIpcRef<'a>>,
    descr: &'a str,
    max_other_parallel: u8,
}

impl<'a> SyncMotifLoaderIpcInputRef<'a> {
    fn new(input: &'a SyncMotifLoaderInput, ctx: &RunCtxImpl) -> Self {
        Self {
            id: &input.id,
            motif_id: *input.motif_id,
            select_motifs_run_config_id: &input.select_motifs_run_config_id,
            find_motifs_run_config_id: input.find_motifs_run_config_id,
            channel: input.channel,
            window: input.window,
            loaded_resources: input.loaded_resources.iter().map(|resource| SyncMotifLoaderResourceIpcRef { resource_id: resource.resource_id, slices: resource.slices.as_slice(), raw_tensor: &resource.raw_tensor }).collect(),
            descr: input.descr.as_str(),
            max_other_parallel: ctx.max_other_parallel(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct SyncMotifLoaderResourceIpcInput {
    resource_id: ResourceId,
    slices: Vec<u32>,
    raw_tensor: ThoTensor<MotifSlices, SynKey>,
}

#[derive(Debug, Deserialize)]
struct SyncMotifLoaderIpcInput {
    id: SynKey,
    motif_id: u32,
    select_motifs_run_config_id: Option<Vec<RunConfigId>>,
    find_motifs_run_config_id: RunConfigId,
    channel: i64,
    window: u32,
    loaded_resources: Vec<SyncMotifLoaderResourceIpcInput>,
    descr: String,
    max_other_parallel: u8,
}

#[derive(Debug)]
struct SyncMotifLoaderIpcWrapper {
    stage: Arc<SyncMotifLoader<RunCtxImpl>>,
}

impl SyncStageIpcWrapper for SyncMotifLoaderIpcWrapper {
    fn do_run(&self, input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError> {
        let input = JsonIpcCodec::decode::<SyncMotifLoaderIpcInput>(input)?;
        let input_tensor_keys = input.loaded_resources.iter().filter_map(|resource| resource.raw_tensor.ref_key()).collect::<Vec<_>>();
        let loaded_resources = input.loaded_resources.into_iter().map(|resource| {
            Ok(LoadedMotifResource {
                id: LoadedMotifResourceId::new(resource.resource_id),
                resource_id: resource.resource_id,
                slices: resource.slices,
                raw_tensor: ThoTensor::ref_to_tensor(all_cuda_tensors, resource.raw_tensor)?,
            })
        }).collect::<Result<Vec<_>, ThoError>>()?;
        let stage_input = SyncMotifLoaderInput::new(
            input.id,
            input.motif_id.into(),
            input.select_motifs_run_config_id,
            input.find_motifs_run_config_id,
            input.channel,
            input.window,
            loaded_resources,
            input.descr,
        );
        let mut ctx = RunCtxImpl::new(tho_device.tho_device_idx(), SYNC_MOTIF_LOADER_ENDPOINT);
        ctx.set_max_other_parallel(input.max_other_parallel);
        let output = Arc::clone(&self.stage).do_run(&stage_input, &ctx)?;
        drop(stage_input);
        {
            let mut tensors = all_cuda_tensors.lock();
            for key in input_tensor_keys {
                tensors.remove(&key);
            }
        }
        output.encode_from_worker::<JsonIpcCodec>(all_cuda_tensors, tho_device)
    }
}

#[derive(Debug, Proxy)]
#[proxy(
    common = "thoosa_common",
    name = "ProxySyncMotifLoader",
    endpoint = "SYNC_MOTIF_LOADER_ENDPOINT",
    new_args = "tho_device_idx: usize",
    fields = "id: SynKey = (MOTIF_LOADER_SYNC_STAGE, \"SyncMotifLoader\").into()",
    out_id = "&self.id + input_id",
    run_input = "SyncMotifLoaderIpcInputRef::new(input, ctx)",
    endpoint_init = "sync_motif_loader_endpoint_init",
    wrapper = "SyncMotifLoaderIpcWrapper"
)]
struct SyncMotifLoaderProxyTarget {
    base: SyncMotifLoader<thoosa_common::runner::run_ctx_2::RunCtxImpl>,
}

impl SyncMotifLoaderProxyTarget {
    fn new() -> SyncMotifLoader<RunCtxImpl> {
        SyncMotifLoader::new()
    }
}

#[derive(Debug, Named)]
pub struct SyncMotifLoader<C>
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
    type In<'a> = SyncMotifLoaderInput;
    type Out = SyncMotifLoaderOutput;
    type Ctx = C;
}

impl<C> SyncStage for SyncMotifLoader<C>
where
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SyncMotifLoader::do_run"))]
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let motif = build_motif_from_loaded_resources(
            input.motif_id.clone(),
            input.select_motifs_run_config_id.as_ref(),
            input.find_motifs_run_config_id,
            input.channel,
            input.window,
            &input.loaded_resources,
            ctx.tho_device_idx(),
            &input.descr,
        )?;
        Ok(SyncMotifLoaderOutput::new(
            motif,
            MotifMeta {
                channel: input.channel,
                find_motifs_run_config_id: input.find_motifs_run_config_id,
                window: input.window,
            },
        ))
    }
}

#[derive(Debug)]
enum SyncMotifLoaderDynSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    Local(Arc<SyncMotifLoader<C>>),
    Proxy(Arc<ProxySyncMotifLoader>),
}

impl<C> SyncMotifLoaderDynSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    async fn new(tho_device_idx: usize) -> Result<Self, ThoError> {
        if GlobalCtx::global().is_ipc() {
            Ok(Self::Proxy(Arc::new(ProxySyncMotifLoader::new(tho_device_idx).await?)))
        } else {
            Ok(Self::Local(Arc::new(SyncMotifLoader::new())))
        }
    }
}

impl<C> Adjustable for SyncMotifLoaderDynSyncStage<C> where C: RunCtx + Adjustable + 'static {}

impl<C> OutId for SyncMotifLoaderDynSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        match self {
            Self::Local(stage) => stage.out_id(input_id),
            Self::Proxy(stage) => stage.out_id(input_id),
        }
    }
}

impl<C> Stage for SyncMotifLoaderDynSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type In<'a> = SyncMotifLoaderInput;
    type Out = SyncMotifLoaderOutput;
    type Ctx = C;
}

impl<C> SyncStage for SyncMotifLoaderDynSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn consume<'a>(self: Arc<Self>, input: Self::In<'a>, ctx: Self::Ctx) -> ConsumeResult<Self::In<'a>, Self::Out> {
        let result = match self.as_ref() {
            Self::Local(stage) => Arc::clone(stage).do_run(&input, &ctx),
            Self::Proxy(stage) => {
                let mut proxy_ctx = RunCtxImpl::new(ctx.tho_device_idx(), SYNC_MOTIF_LOADER_ENDPOINT);
                proxy_ctx.set_max_other_parallel(ctx.max_other_parallel());
                Arc::clone(stage).do_run(&input, &proxy_ctx)
            }
        };
        match result {
            Ok(output) => ConsumeResult::Ok(output),
            Err(error) => ConsumeResult::Failed { input, error },
        }
    }
}

impl<L, C> Stage for MotifLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type In<'a> = Arc<MotifLoaderInput>;
    type Out = Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>;

    type Ctx = C;
}

impl<L, C> AsyncStage for MotifLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="MotifLoader::run"))]
    async fn run(
        self: Arc<Self>,
        input: Self::In<'static>,
        mut ctx: Self::Ctx,
    ) -> Result<Self::Out, ThoError>
    where
        for<'a> Self::In<'a>: Send + Sync,
    {
        let input_id = input.id().clone();
        let (motif_id, select_motifs_run_config_id) = (input.motif_id.clone(), input.select_motifs_run_config_id.clone());

        let motif_id_value = motif_id.vals[0].to_data_id();
        let select_motifs_run_config_id_for_key = select_motifs_run_config_id.clone();
        // TODO(perf): cache motif slice lists/resource grouping by motif_id; repeated DB reads dominate MotifLoader elapsed time in traced runs.
        let (channel, find_motifs_run_config_id, slices, resources_map) = track_await(self
            .store
            .list_motif_slices(motif_id_value.0, select_motifs_run_config_id, Some(1000), None)).await?;

        let find_motifs_run_config: FindMotifsRunConfig = track_await(load_run_config(&self.store, find_motifs_run_config_id)).await?;
        let real_find_motifs_run_config_id = find_motifs_run_config.id.unwrap_or(find_motifs_run_config_id);

        debug!("MotifLoader::run: motif_id={motif_id} channel={channel} ctx={ctx:?} find_motifs_run_config_id={real_find_motifs_run_config_id}");

        let window = find_motifs_run_config.window;
        let resource_ids_slices = slices.iter().fold(HashMap::new(), |mut acc, slice| {
            acc.entry(slice.0).or_insert_with(Vec::new).push(slice.1);
            acc
        });
        assert!(!resource_ids_slices.is_empty());
        let configurable_loader = config_to_stage(&self.loader, &find_motifs_run_config, &self.ws, ctx.tho_device_idx()).await?;
        let loaded_resources = track_await(load_motif_resource_tensors(
            motif_id_value.clone(),
            channel,
            real_find_motifs_run_config_id,
            window,
            resources_map,
            resource_ids_slices,
            configurable_loader,
            ctx.branch(false),
            "motif loader",
        )).await?;
        let loaded_resources_name = loaded_resources.registered_name();
        let sync_motif_loader = Arc::new(SyncMotifLoaderDynSyncStage::new(ctx.tho_device_idx()).await?);

        let sync_input = SyncMotifLoaderInputWrapper::new(
            input_id.clone(),
            motif_id_value,
            select_motifs_run_config_id_for_key,
            real_find_motifs_run_config_id,
            channel,
            window,
            loaded_resources,
            "motif loader".to_string(),
        );
        let required_loaded_resource_ids = HashSet::<&dyn CachableIdTr>::new();
        let required_ids = HashMap::from([(loaded_resources_name, &required_loaded_resource_ids)]);
        let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoTaskAdjuster, _, ()> = OOMHandler::new(
            vec![
                Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::FIFTY))),
                Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("motif_loader"))),
            ],
            None,
        );
        let motif = track_await(oom_handler
        .consume(
            &sync_motif_loader,
            sync_input,
            ctx.branch(true),
            WithRunCtxContext::new(MaxParallelAndDumpToCpuAdjustableContex::new(MaxParallelAdjustContextImpl::new(), DumpToCpuContextImpl::new(&required_ids))),
        )).await?;

        Ok(Arc::new(DevicesWrap::wrap(motif.into_data_with_extra())))
    }
}
///To load slices not belonging to a motif
#[derive(cached_derive::Cached, Named, Debug)]
pub struct RandomSlicesLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    store: Arc<Box<dyn Store>>,
    loader: Arc<L>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    pd: PhantomData<C>,
}

impl <L,C> RandomSlicesLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    pub fn new(store: Arc<Box<dyn Store>>, loader: Arc<L>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Self {
        Self {
            id: (RAW_TENSOR_LOADER, "RandomSlicesLoader").into(),
            store,
            loader,
            ws:Arc::clone(&ws),
            pd: PhantomData::default()
        }
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
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type In<'a> = Arc<SlicesLoaderInput>;
    type Out = Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>;

    type Ctx = C;
}


impl<L, C> OutId for RandomSlicesLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,

{
    fn out_id(&self, input_id: &<Self::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + mono_key!(input_id.id().0, "random_slices")
    }
}



impl<L, C> AsyncStage for RandomSlicesLoader<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="RandomSlicesLoader::run"))]
    async fn run(
        self: Arc<Self>,
        input: Arc<SlicesLoaderInput>,
        mut ctx: Self::Ctx,
    ) -> Result<Self::Out, ThoError>
    where
            for<'a> Self::In<'a>: Send + Sync,
    {

        let (motif_id, limit) = (input.motif_id.clone(), input.limit);


        let (channel, find_motifs_run_config_id) = track_await(self.store.get_motif_meta(*motif_id)).await?;

        let find_motifs_run_config: FindMotifsRunConfig = track_await(load_run_config(&self.store, find_motifs_run_config_id)).await?;


        debug!("RandomSlicesLoader::run: motif_id={motif_id} channel={channel} ctx={ctx:?} find_motifs_run_config_id={find_motifs_run_config_id}");

        let motif_id_value = motif_id.clone().0;
        let input_len_adjuster = find_motifs_run_config.input_len_adjuster();
        // TODO(perf): precompute or cache random-slice pools for (motif_id, limit, input_len_adjuster) so each batch does not wait on fresh SQL selection.
        let (channel, find_motifs_run_config_id, slices, resources_map) = track_await(self.store.list_random_slices_out_of_motif(motif_id_value, find_motifs_run_config.window, None, input_len_adjuster, Some(limit as usize))).await?;


        let motif_id_value = motif_id.clone();
        let resource_ids_slices = slices.iter().fold(HashMap::new(), |mut acc, slice| {
            acc.entry(slice.0).or_insert_with(Vec::new).push(slice.1);
            acc
        });
        assert!(!resource_ids_slices.is_empty());

        debug!("RandomSlicesLoader::run: motif_id={motif_id} find_motifs_run_config={find_motifs_run_config:?} device_idx={}", ctx.tho_device_idx());

        let configurable_loader = config_to_stage(&self.loader, &find_motifs_run_config, &self.ws, ctx.tho_device_idx()).await?;
        let loaded_resources = track_await(load_motif_resource_tensors(
            motif_id_value.clone(),
            channel,
            find_motifs_run_config_id,
            find_motifs_run_config.window,
            resources_map,
            resource_ids_slices,
            configurable_loader,
            ctx.branch(false),
            "random loader",
        )).await?;
        let loaded_resources_name = loaded_resources.registered_name();
        let sync_motif_loader = Arc::new(SyncMotifLoaderDynSyncStage::new(ctx.tho_device_idx()).await?);
        let sync_input = SyncMotifLoaderInputWrapper::new(
            mono_key!(*motif_id_value, "random_slices_loader_input").into(),
            motif_id_value,
            None,
            find_motifs_run_config_id,
            channel,
            find_motifs_run_config.window,
            loaded_resources,
            "random loader".to_string(),
        );
        let required_loaded_resource_ids = HashSet::<&dyn CachableIdTr>::new();
        let required_ids = HashMap::from([(loaded_resources_name, &required_loaded_resource_ids)]);
        let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoTaskAdjuster, _, ()> = OOMHandler::new(
            vec![
                Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::FIFTY))),
                Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("random_slices_loader"))),
            ],
            None,
        );
        let motif = track_await(oom_handler
        .consume(
            &sync_motif_loader,
            sync_input,
            ctx.branch(true),
            WithRunCtxContext::new(MaxParallelAndDumpToCpuAdjustableContex::new(MaxParallelAdjustContextImpl::new(), DumpToCpuContextImpl::new(&required_ids))),
        )).await?;

        let DataWithExtra { data, ..} = motif.into_data_with_extra();

        #[cfg(tho_debug_plots)]
        debug_plot!(motif.data().as_ref(), "random_motif_{motif_id}")?;



        let arc: Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>> = Arc::new(DevicesWrap::wrap(data));

        Ok(arc)
    }
}

pub(crate) const SYNC_LOAD_MOTIF_RESOURCES_ENDPOINT: &str = "SyncLoadMotifResources";

#[derive(Debug)]
pub struct SyncLoadMotifResourcesInput<'a> {
    id: SynKey,
    raw_tensor: ThoDataRef<'a, ThoTensor<ChannelsData, SynKey>>,
    motif_id: DataId,
    resource_id: ResourceId,
    find_motifs_run_config_id: u32,
    channel: i64,
    window: u32,
    slices: Vec<u32>,
    descr: String,
}

impl Input for SyncLoadMotifResourcesInput<'_> {}

impl WithId for SyncLoadMotifResourcesInput<'_> {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug)]
struct SyncLoadMotifResourcesInputWrapper {
    id: SynKey,
    raw_tensor: Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>,
    motif_id: DataId,
    resource_id: ResourceId,
    find_motifs_run_config_id: u32,
    channel: i64,
    window: u32,
    slices: Vec<u32>,
    descr: String,
}

impl SyncLoadMotifResourcesInputWrapper {
    fn new(raw_tensor: Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, motif_id: DataId, resource_id: ResourceId, find_motifs_run_config_id: u32, channel: i64, window: u32, slices: Vec<u32>, descr: String) -> Self {
        Self { id: raw_tensor.id().clone(), raw_tensor, motif_id, resource_id, find_motifs_run_config_id, channel, window, slices, descr }
    }
}

impl WithId for SyncLoadMotifResourcesInputWrapper {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl BorrowRef for SyncLoadMotifResourcesInputWrapper {
    type Out<'a> = SyncLoadMotifResourcesInput<'a> where Self: 'a;

    fn borrow_to<'a>(&'a self, to_device_idx: usize) -> impl std::future::Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        async move {
            let raw_tensor = track_await(BorrowRef::borrow_to(&self.raw_tensor, to_device_idx)).await?;
            Ok(SyncLoadMotifResourcesInput {
                id: self.id.clone(),
                raw_tensor,
                motif_id: self.motif_id.clone(),
                resource_id: self.resource_id,
                find_motifs_run_config_id: self.find_motifs_run_config_id,
                channel: self.channel,
                window: self.window,
                slices: self.slices.clone(),
                descr: self.descr.clone(),
            })
        }
    }
}

#[derive(Debug, Serialize)]
struct SyncLoadMotifResourcesIpcInputRef<'a> {
    id: &'a SynKey,
    raw_tensor: &'a ThoTensor<ChannelsData, SynKey>,
    motif_id: u32,
    resource_id: ResourceId,
    find_motifs_run_config_id: u32,
    channel: i64,
    window: u32,
    slices: &'a [u32],
    descr: &'a str,
    max_other_parallel: u8,
}

impl<'a> SyncLoadMotifResourcesIpcInputRef<'a> {
    fn new(input: &'a SyncLoadMotifResourcesInput<'_>, ctx: &RunCtxImpl) -> Self {
        Self {
            id: &input.id,
            raw_tensor: input.raw_tensor.as_ref(),
            motif_id: *input.motif_id,
            resource_id: input.resource_id,
            find_motifs_run_config_id: input.find_motifs_run_config_id,
            channel: input.channel,
            window: input.window,
            slices: input.slices.as_slice(),
            descr: input.descr.as_str(),
            max_other_parallel: ctx.max_other_parallel(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct SyncLoadMotifResourcesIpcInput {
    id: SynKey,
    raw_tensor: ThoTensor<ChannelsData, SynKey>,
    motif_id: u32,
    resource_id: ResourceId,
    find_motifs_run_config_id: u32,
    channel: i64,
    window: u32,
    slices: Vec<u32>,
    descr: String,
    max_other_parallel: u8,
}

#[derive(Debug)]
struct SyncLoadMotifResourcesIpcWrapper {
    stage: Arc<SyncLoadMotifResources>,
}

impl SyncStageIpcWrapper for SyncLoadMotifResourcesIpcWrapper {
    fn do_run(&self, input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError> {
        let input = JsonIpcCodec::decode::<SyncLoadMotifResourcesIpcInput>(input)?;
        let raw_tensor = ThoTensor::ref_to_tensor(all_cuda_tensors, input.raw_tensor)?;
        let stage_input = SyncLoadMotifResourcesInput {
            id: input.id,
            raw_tensor: ThoDataRef::Borrowed(&raw_tensor),
            motif_id: input.motif_id.into(),
            resource_id: input.resource_id,
            find_motifs_run_config_id: input.find_motifs_run_config_id,
            channel: input.channel,
            window: input.window,
            slices: input.slices,
            descr: input.descr,
        };
        let mut ctx = RunCtxImpl::new(tho_device.tho_device_idx(), SYNC_LOAD_MOTIF_RESOURCES_ENDPOINT);
        ctx.set_max_other_parallel(input.max_other_parallel);
        let output = Arc::clone(&self.stage).do_run(&stage_input, &ctx)?;
        output.encode_from_worker::<JsonIpcCodec>(all_cuda_tensors, tho_device)
    }
}

#[derive(Debug, Proxy)]
#[proxy(
    common = "thoosa_common",
    name = "ProxySyncLoadMotifResources",
    endpoint = "SYNC_LOAD_MOTIF_RESOURCES_ENDPOINT",
    new_args = "tho_device_idx: usize",
    out_id = "input_id.clone()",
    run_input = "SyncLoadMotifResourcesIpcInputRef::new(input, ctx)",
    endpoint_init = "sync_load_motif_resources_endpoint_init",
    wrapper = "SyncLoadMotifResourcesIpcWrapper"
)]
pub struct SyncLoadMotifResources;

impl SyncLoadMotifResources {
    fn new() -> Self {
        Self
    }
}

impl Adjustable for SyncLoadMotifResources {}

impl OutId for SyncLoadMotifResources {
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

impl Stage for SyncLoadMotifResources {
    type Ctx = RunCtxImpl;
    type In<'a> = SyncLoadMotifResourcesInput<'a>;
    type Out = ThoTensor<MotifSlices, SynKey>;
}

impl SyncStage for SyncLoadMotifResources {
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SyncLoadMotifResources::do_run"))]
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let size = input.raw_tensor.size();
        let mut valid_slices = Vec::with_capacity(input.slices.len());

        for &slice in &input.slices {
            let Some(slice_end) = slice.checked_add(input.window) else {
                warn!("SyncLoadMotifResources::do_run: motif_id={} slice={slice:?}+{} overflows for resource_id={} in find_motifs_run_config_id={} in {}", input.motif_id, input.window, input.resource_id, input.find_motifs_run_config_id, input.descr);
                continue;
            };

            if slice_end as i64 > size[1] {
                warn!("SyncLoadMotifResources::do_run: motif_id={} slice={slice:?}+{} is bigger than size={size:?} for resource_id={} in find_motifs_run_config_id={} in {}", input.motif_id, input.window, input.resource_id, input.find_motifs_run_config_id, input.descr);
            } else {
                valid_slices.push(slice);
            }
        }

        let mut dst = Tensor::f_zeros([valid_slices.len() as i64, input.window as i64], (input.raw_tensor.t_ref().kind(), input.raw_tensor.t_ref().device()))?;
        for (slice_i, slice) in valid_slices.into_iter().enumerate() {
            let src = input.raw_tensor.t_ref().f_i((input.channel, slice as i64..(slice + input.window) as i64))?;
            dst.f_get(slice_i as i64)?.f_copy_(&src)?;
        }

        ThoTensor::new(input.raw_tensor.id().clone(), dst)
    }
}

#[derive(Debug)]
enum SyncLoadMotifResourcesStage {
    Local(Arc<SyncLoadMotifResources>),
    Proxy(Arc<ProxySyncLoadMotifResources>),
}

#[derive(Debug)]
struct SyncLoadMotifResourcesDynStage<C> {
    stage: SyncLoadMotifResourcesStage,
    pd: PhantomData<C>,
}

impl<C> SyncLoadMotifResourcesDynStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    async fn new(tho_device_idx: usize) -> Result<Self, ThoError> {
        let stage = if GlobalCtx::global().is_ipc() {
            SyncLoadMotifResourcesStage::Proxy(Arc::new(ProxySyncLoadMotifResources::new(tho_device_idx).await?))
        } else {
            SyncLoadMotifResourcesStage::Local(Arc::new(SyncLoadMotifResources::new()))
        };
        Ok(Self { stage, pd: PhantomData })
    }
}

impl<C> Adjustable for SyncLoadMotifResourcesDynStage<C> where C: RunCtx + Adjustable + 'static {}

impl<C> OutId for SyncLoadMotifResourcesDynStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

impl<C> Stage for SyncLoadMotifResourcesDynStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = SyncLoadMotifResourcesInput<'a>;
    type Out = ThoTensor<MotifSlices, SynKey>;
}

impl<C> SyncStage for SyncLoadMotifResourcesDynStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let mut stage_ctx = RunCtxImpl::new(ctx.tho_device_idx(), SYNC_LOAD_MOTIF_RESOURCES_ENDPOINT);
        stage_ctx.set_max_other_parallel(ctx.max_other_parallel());
        match &self.stage {
            SyncLoadMotifResourcesStage::Local(stage) => Arc::clone(stage).do_run(input, &stage_ctx),
            SyncLoadMotifResourcesStage::Proxy(stage) => Arc::clone(stage).do_run(input, &stage_ctx),
        }
    }
}

pub const SYNC_LOAD_RANDOM_SLICES_ENDPOINT: &str = "SyncLoadRandomSlices";

#[derive(Debug)]
pub struct SyncLoadRandomSlicesInput<'a> {
    id: SynKey,
    raw_tensor: ThoDataRef<'a, ThoTensor<ChannelsData, SynKey>>,
    resource_id: ResourceId,
    find_motifs_run_config_id: u32,
    channels: i64,
    window: u32,
    slices: Vec<u32>,
    descr: String,
}

impl Input for SyncLoadRandomSlicesInput<'_> {}

impl WithId for SyncLoadRandomSlicesInput<'_> {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug)]
struct SyncLoadRandomSlicesInputWrapper {
    id: SynKey,
    raw_tensor: Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>,
    resource_id: ResourceId,
    find_motifs_run_config_id: u32,
    channels: i64,
    window: u32,
    slices: Vec<u32>,
    descr: String,
}

impl SyncLoadRandomSlicesInputWrapper {
    fn new(
        raw_tensor: Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>,
        resource_id: ResourceId,
        find_motifs_run_config_id: u32,
        channels: i64,
        window: u32,
        slices: Vec<u32>,
        descr: String,
    ) -> Self {
        Self { id: raw_tensor.id().clone(), raw_tensor, resource_id, find_motifs_run_config_id, channels, window, slices, descr }
    }
}

impl WithId for SyncLoadRandomSlicesInputWrapper {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl BorrowRef for SyncLoadRandomSlicesInputWrapper {
    type Out<'a> = SyncLoadRandomSlicesInput<'a> where Self: 'a;

    fn borrow_to<'a>(&'a self, to_device_idx: usize) -> impl std::future::Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        async move {
            let raw_tensor = track_await(BorrowRef::borrow_to(&self.raw_tensor, to_device_idx)).await?;
            Ok(SyncLoadRandomSlicesInput {
                id: self.id.clone(),
                raw_tensor,
                resource_id: self.resource_id,
                find_motifs_run_config_id: self.find_motifs_run_config_id,
                channels: self.channels,
                window: self.window,
                slices: self.slices.clone(),
                descr: self.descr.clone(),
            })
        }
    }
}

#[derive(Debug, Serialize)]
struct SyncLoadRandomSlicesIpcInputRef<'a> {
    id: &'a SynKey,
    raw_tensor: &'a ThoTensor<ChannelsData, SynKey>,
    resource_id: ResourceId,
    find_motifs_run_config_id: u32,
    channels: i64,
    window: u32,
    slices: &'a [u32],
    descr: &'a str,
    max_other_parallel: u8,
}

impl<'a> SyncLoadRandomSlicesIpcInputRef<'a> {
    fn new(input: &'a SyncLoadRandomSlicesInput<'_>, ctx: &RunCtxImpl) -> Self {
        Self {
            id: &input.id,
            raw_tensor: input.raw_tensor.as_ref(),
            resource_id: input.resource_id,
            find_motifs_run_config_id: input.find_motifs_run_config_id,
            channels: input.channels,
            window: input.window,
            slices: input.slices.as_slice(),
            descr: input.descr.as_str(),
            max_other_parallel: ctx.max_other_parallel(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct SyncLoadRandomSlicesIpcInput {
    id: SynKey,
    raw_tensor: ThoTensor<ChannelsData, SynKey>,
    resource_id: ResourceId,
    find_motifs_run_config_id: u32,
    channels: i64,
    window: u32,
    slices: Vec<u32>,
    descr: String,
    max_other_parallel: u8,
}

#[derive(Debug)]
struct SyncLoadRandomSlicesIpcWrapper {
    stage: Arc<SyncLoadRandomSlices>,
}

impl SyncStageIpcWrapper for SyncLoadRandomSlicesIpcWrapper {
    fn do_run(&self, input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError> {
        let input = JsonIpcCodec::decode::<SyncLoadRandomSlicesIpcInput>(input)?;
        let raw_tensor = ThoTensor::ref_to_tensor(all_cuda_tensors, input.raw_tensor)?;
        let stage_input = SyncLoadRandomSlicesInput {
            id: input.id,
            raw_tensor: ThoDataRef::Borrowed(&raw_tensor),
            resource_id: input.resource_id,
            find_motifs_run_config_id: input.find_motifs_run_config_id,
            channels: input.channels,
            window: input.window,
            slices: input.slices,
            descr: input.descr,
        };
        let mut ctx = RunCtxImpl::new(tho_device.tho_device_idx(), SYNC_LOAD_RANDOM_SLICES_ENDPOINT);
        ctx.set_max_other_parallel(input.max_other_parallel);
        let output = Arc::clone(&self.stage).do_run(&stage_input, &ctx)?;
        output.encode_from_worker::<JsonIpcCodec>(all_cuda_tensors, tho_device)
    }
}

#[derive(Debug, Proxy)]
#[proxy(
    common = "thoosa_common",
    name = "ProxySyncLoadRandomSlices",
    endpoint = "SYNC_LOAD_RANDOM_SLICES_ENDPOINT",
    new_args = "tho_device_idx: usize",
    out_id = "input_id.clone()",
    run_input = "SyncLoadRandomSlicesIpcInputRef::new(input, ctx)",
    endpoint_init = "sync_load_random_slices_endpoint_init",
    wrapper = "SyncLoadRandomSlicesIpcWrapper"
)]
pub struct SyncLoadRandomSlices;

impl SyncLoadRandomSlices {
    fn new() -> Self {
        Self
    }
}

impl Adjustable for SyncLoadRandomSlices {}

impl OutId for SyncLoadRandomSlices {
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

impl Stage for SyncLoadRandomSlices {
    type Ctx = RunCtxImpl;
    type In<'a> = SyncLoadRandomSlicesInput<'a>;
    type Out = ThoTensor<ChannelsMotifsSlices, SynKey>;
}

impl SyncStage for SyncLoadRandomSlices {
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SyncLoadRandomSlices::do_run"))]
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let size = input.raw_tensor.size();
        let mut valid_slices = Vec::with_capacity(input.slices.len());
        for &slice in &input.slices {
            let Some(slice_end) = slice.checked_add(input.window) else {
                warn!("SyncLoadRandomSlices::do_run: slice={slice:?}+{} overflows for resource_id={} in find_motifs_run_config_id={} in {}", input.window, input.resource_id, input.find_motifs_run_config_id, input.descr);
                continue;
            };
            let in_range = size.len() >= 2 && input.channels <= size[0] && slice_end as i64 <= size[1];
            if in_range {
                valid_slices.push((slice, slice_end));
            } else {
                warn!("SyncLoadRandomSlices::do_run: slice={slice:?}+{} is bigger than size={size:?} for resource_id={} in find_motifs_run_config_id={} in {}", input.window, input.resource_id, input.find_motifs_run_config_id, input.descr);
            }
        }

        let mut output = Tensor::f_zeros(
            [input.channels, valid_slices.len() as i64, input.window as i64],
            (Float, input.raw_tensor.t_ref().device()),
        )?;
        for (slice_i, (slice, slice_end)) in valid_slices.into_iter().enumerate() {
            let src = input.raw_tensor.t_ref().f_i((0..input.channels, slice as i64..slice_end as i64))?;
            output.f_i((0..input.channels, slice_i as i64, ..))?.f_copy_(&src)?;
        }

        ThoTensor::new(mono_key!(input.resource_id, format!("random_slices_resource_id={}", input.resource_id)).into(), output)
    }
}

#[derive(Debug)]
enum SyncLoadRandomSlicesStage {
    Local(Arc<SyncLoadRandomSlices>),
    Proxy(Arc<ProxySyncLoadRandomSlices>),
}

#[derive(Debug)]
struct SyncLoadRandomSlicesDynStage<C> {
    stage: SyncLoadRandomSlicesStage,
    pd: PhantomData<C>,
}

impl<C> SyncLoadRandomSlicesDynStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    async fn new(tho_device_idx: usize) -> Result<Self, ThoError> {
        let stage = if GlobalCtx::global().is_ipc() {
            SyncLoadRandomSlicesStage::Proxy(Arc::new(ProxySyncLoadRandomSlices::new(tho_device_idx).await?))
        } else {
            SyncLoadRandomSlicesStage::Local(Arc::new(SyncLoadRandomSlices::new()))
        };
        Ok(Self { stage, pd: PhantomData })
    }
}

impl<C> Adjustable for SyncLoadRandomSlicesDynStage<C> where C: RunCtx + Adjustable + 'static {}

impl<C> OutId for SyncLoadRandomSlicesDynStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

impl<C> Stage for SyncLoadRandomSlicesDynStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = SyncLoadRandomSlicesInput<'a>;
    type Out = ThoTensor<ChannelsMotifsSlices, SynKey>;
}

impl<C> SyncStage for SyncLoadRandomSlicesDynStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let mut stage_ctx = RunCtxImpl::new(ctx.tho_device_idx(), SYNC_LOAD_RANDOM_SLICES_ENDPOINT);
        stage_ctx.set_max_other_parallel(ctx.max_other_parallel());
        match &self.stage {
            SyncLoadRandomSlicesStage::Local(stage) => Arc::clone(stage).do_run(input, &stage_ctx),
            SyncLoadRandomSlicesStage::Proxy(stage) => Arc::clone(stage).do_run(input, &stage_ctx),
        }
    }
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="load_motif_resource_tensors"))]
async fn load_motif_resource_tensors<L, C>(
    motif_id: DataId,
    channel: i64,
    find_motifs_run_config_id: u32,
    window: u32,
    resources_map: HashMap<ResourceId, Resource>,
    resource_ids_slices: HashMap<ResourceId, Vec<u32>>,
    configurable_loader: Arc<LoaderStageChoice<L, C>>,
    mut ctx: C,
    descr: &str,
) -> Result<TmpCached<LoadedMotifResource>, ThoError>
where
    C: 'static + RunCtx + Adjustable,
    L: 'static + Cached + OutId + for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx=C>,
{
    let loaded_resources = track_await(TmpCached::new(&format!("tmp_loaded_motif_resources_{descr}"))).await;
    let sync_load_motif_resources = Arc::new(SyncLoadMotifResourcesDynStage::new(ctx.tho_device_idx()).await?);
    let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoTaskAdjuster, NoRunCtxAdjuster, ()> = OOMHandler::new(vec![], None);
    let mut loaded_resources_count = 0usize;
    for (resource_id, slices) in resource_ids_slices {
        debug!("load_motif_resource_tensors: motif_id={motif_id} resource_id={resource_id} slices={slices:?} descr={descr}");

        // TODO(perf): load raw resource tensors with bounded concurrency; this loop is sequential and sits under the MotifLoader critical path.
        let resource = resources_map.get(&resource_id).cloned().expect(&format!("no resource with id={resource_id}"));
        let resource_cache = Arc::new(resource);
        let raw_tensor = track_await(Arc::clone(&configurable_loader).run_cached(resource_cache, ctx.branch(false))).await?;

        debug!("load_motif_resource_tensors: motif_id={motif_id} resource_id={resource_id} raw_tensor={raw_tensor:?} descr={descr}");

        let sync_input = Arc::new(SyncLoadMotifResourcesInputWrapper::new(
            Arc::clone(&raw_tensor),
            motif_id.clone(),
            resource_id,
            find_motifs_run_config_id,
            channel,
            window,
            slices.clone(),
            descr.to_string(),
        ));
        let channel_tensor = track_await(oom_handler.run(&sync_load_motif_resources, sync_input, ctx.branch(false), NoContexts)).await?;
        drop(raw_tensor);
        loaded_resources.insert(LoadedMotifResource {
            id: LoadedMotifResourceId::new(resource_id),
            resource_id,
            slices,
            raw_tensor: channel_tensor,
        });
        loaded_resources_count += 1;
    }
    debug!("load_motif_resource_tensors: motif_id={motif_id} find_motifs_run_config_id={find_motifs_run_config_id} loaded_resources={loaded_resources_count} descr={descr}");
    Ok(loaded_resources)
}

fn build_motif_from_loaded_resources(
    motif_id: DataId,
    select_motifs_run_config_id: Option<&Vec<RunConfigId>>,
    find_motifs_run_config_id: u32,
    channel: i64,
    window: u32,
    loaded_resources: &[LoadedMotifResource],
    device_idx: usize,
    descr: &str,
) -> Result<ThoTensor<MotifSlices, SynKey>, ThoError> {
    let total_slices = loaded_resources.iter().map(|resource| resource.raw_tensor.t_ref().size()[0]).sum::<i64>();
    debug!("build_motif_slices_from_loaded_resources: motif_id={motif_id} select_motifs_run_config_id={select_motifs_run_config_id:?} find_motifs_run_config_id={find_motifs_run_config_id} channel={channel} window={window} total_slices={total_slices} loaded_resources={} device_idx={device_idx} descr={descr}", loaded_resources.len());
    /*
       select_motifs_run_config_id should not be used if SelectMotifsRunConfig doesn't contain extend_config
       let motif_id = select_motifs_run_config_id
        .map(|run_config_ids| {
            run_config_ids.iter().fold(mono_key!(motif_t.id().0, "motif").into(), |acc: SynKey, run_config_id| {
                &acc + mono_key!(*run_config_id, "select_motifs_run_config_id")
            })
        })
     */
    // TODO motif can contain either only slices found during initial 'find motifs' run or also slices found during 'select motifs' ran with 'extend' config. This should be reflected in motif id
    let motif_refs = loaded_resources
        .iter()
        .filter(|resource| resource.raw_tensor.t_ref().size()[0] > 0)
        .map(|resource| resource.raw_tensor.t_ref())
        .collect::<Vec<_>>();

    let tensor = if motif_refs.is_empty() {
        Tensor::f_zeros([0, window as i64], (Float, ThoDevice::from(device_idx).into()))?
    } else {
        Tensor::f_cat(motif_refs.as_slice(), 0)?.f_totype(Float)?
    };

    ThoTensor::new(mono_key!(*motif_id, descr).into(), tensor)


}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="load_random_slices"))]
pub(crate) async fn load_random_slices<L, C>(
    find_motifs_run_config_id: u32,
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
    L: 'static + Cached + OutId + for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx=C>,
{
    let device_idx = ctx.tho_device_idx();
    debug!("load_random_slices: find_motifs_run_config_id={find_motifs_run_config_id} slices={slices:?} descr={descr}");

    let find_motifs_run_config: FindMotifsRunConfig = track_await(load_run_config(&store, find_motifs_run_config_id)).await?;

    let resource_ids_slices = slices
        .iter()
        .fold(HashMap::new(), |mut acc, slice| {
            acc.entry(slice.0.clone()).or_insert_with(Vec::new).push(slice.1);
            acc
        });

    assert!(!resource_ids_slices.is_empty());

    let window = find_motifs_run_config.window;
    let channels = thoosa_shared::perf_log!(thoosa_shared::sync_read_lock!(ws))
    .channels() as i64;
    assert!(channels > 0, "load_random_slices: channels must be > 0");

    debug!("load_random_slices:  find_motifs_run_config={find_motifs_run_config:?} device_idx={device_idx} channels={channels}");

    let tmp_cache = track_await(TmpCached::new(&format!("tmp_random_slices_{descr}"))).await;

    let configurable_loader = config_to_stage(&loader, &find_motifs_run_config, ws, ctx.tho_device_idx()).await?;
    let sync_load_random_slices = Arc::new(SyncLoadRandomSlicesDynStage::new(ctx.tho_device_idx()).await?);
    let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoTaskAdjuster, NoRunCtxAdjuster, ()> = OOMHandler::new(vec![], None);
    for (resource_id, slices) in resource_ids_slices {
        debug!("load_random_slices: resource_id={resource_id} slices={slices:?} in_loop_0");

        // TODO(perf): load_random_slices repeats the same sequential raw-resource loading pattern; use a bounded concurrent loader once memory pressure is modeled.
        let resource = resources_map.get(&resource_id).cloned().expect(&format!("no resource with id={resource_id}"));
        let resource_cache = Arc::new(resource);
        let raw_tensor = track_await(Arc::clone(&configurable_loader)
            .run_cached(resource_cache, ctx.branch(false))).await?;
        debug!("load_random_slices: resource_id={resource_id} slices={slices:?} in_loop_1");

        let sync_input = Arc::new(SyncLoadRandomSlicesInputWrapper::new(
            Arc::clone(&raw_tensor),
            resource_id,
            find_motifs_run_config_id,
            channels,
            window,
            slices,
            descr.to_string(),
        ));
        let resource_slices = track_await(oom_handler.run(&sync_load_random_slices, sync_input, ctx.branch(false), NoContexts)).await?;
        if resource_slices.size()[1] > 0 {
            tmp_cache.insert(resource_slices);
        }
    }
    debug!("load_random_slices: find_motifs_run_config_id={find_motifs_run_config_id} done_0");
    debug!("load_random_slices: find_motifs_run_config_id={find_motifs_run_config_id} done_1");

    Ok((window, tmp_cache))
}

#[cfg(test)]
pub mod mock_motif_loader {
    use std::collections::HashMap;
    use async_trait::async_trait;
    use futures_util::future::{BoxFuture, Shared};
    use mockall::mock;
    use std::sync::Arc;
    use thoosa_common::runner::common::cachable::{Cached, ConsumeData, DevicesWrap};
    use thoosa_common::runner::common::{MockRunCtx, OutId, RunCtx};
    use thoosa_common::runner::common::runnable::{AsyncStage, Stage, SyncStage};
    use thoosa_shared::ThoError;
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

            type In<'a> = Arc<MotifLoaderInput>;
            type Out = Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>;
        }

        impl OutId for MotifLoader {
            fn out_id<'a>(&self,input_id: &<<Self as Stage>::In<'a> as WithId>::ID,) -> <<MockMotifLoader as Stage>::Out as WithId>::ID;
        }

        // #[async_trait]
        impl AsyncStage for MotifLoader {
        }

        #[async_trait]
        impl Cached for MotifLoader {

            async fn cache() -> &'static DashMap<<<Self as Stage>::Out as WithId>::ID, Shared<BoxFuture<'static, Result<Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>, ThoError >>>>;
            async fn run_cached(
                self: Arc<Self>,
                input: <MockMotifLoader as Stage>::In<'static>,
                ctx: <MockMotifLoader as Stage>::Ctx,
            ) -> Result<<MockMotifLoader as Stage>::Out, ThoError>;
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
                        Ok(Arc::clone(data_map.get(input.as_ref()).expect(&format!("no mock data for v={input:? } in MockMotifLoader"))))
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
    use log::{debug, set_max_level, trace};
    use parking_lot::RwLock;
    use tch::Tensor;
    use thoosa_common::runner::common::{mock_run_ctx, RunCtx};
    use thoosa_common::runner::common::cachable::{SelectRef, Cached, ConsumeData, DevicesWrap};
    use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex};
use thoosa_common::runner::oom_handler::{run_ctx_adjuster, Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::global_ctx::{mock_global_ctx, GlobalCtx};
    use crate::tasks::find_motifs_run_config::FindMotifsRunConfig;
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
    use thoosa_shared::perf_logging::{format_perf_map, PERF};
    use thoosa_store::RunConfigRow;
    use crate::tasks::stft::SyncStftSelector;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_load_motifs() -> Result<(), ThoError> {
        // set_max_level(Trace);
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 10000)));
        let find_motifs_run_config_id = 1;
        let motif_source_run_config_id = 2;

        let resource_a = Resource::new(1, 1, "tensor_a", 1000);
        let resource_b = Resource::new(2, 1, "tensor_b", 1000);
        let mut store = MockStoreBuilder::new()
            .with_list_motif_slices(10u32, Some(vec![]), None, None, Ok((0i64, motif_source_run_config_id, vec![
                TensorIdSliceIdx(1, 100),
                TensorIdSliceIdx(1, 200),
                TensorIdSliceIdx(2, 100)
            ], HashMap::from([
                (1, resource_a.clone()),
                (2, resource_b.clone())
            ]))))
            .with_run_config_descr([(
                motif_source_run_config_id,
                Ok(RunConfigRow {
                    id: motif_source_run_config_id,
                    r#type: "SelectMotifsRunConfig".to_string(),
                    descr: "test select".to_string(),
                    run_on: "2026-01-01 00:00:00".to_string(),
                }),
            )])
            .with_get_input_run_config_ids([(motif_source_run_config_id, Ok(vec![find_motifs_run_config_id]))])
            .with_list_run_config_values([(
                find_motifs_run_config_id,
                Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_config_id), 100, 100, None, None).to_properties_iter(None).collect())
            )])
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


        // TODO: replace the test-only u32 -> MotifLoaderInput conversion with explicit constructors.
        let input_id = Arc::new(10u32.into());

        let motif = motif_loader
            .run_cached(
                Arc::clone(&input_id),
                thoosa_shared::perf_log!(thoosa_shared::sync_write_lock!(ctx))
                .branch(false),
            )
            .await?;

        let mut ctx_for_data = thoosa_shared::perf_log!(thoosa_shared::sync_write_lock!(ctx)).branch(false);

        let t = motif.borrow_to(0, Some(SyncStftSelector)).await?;

        let dist = 0.; //average_euclidean_distance(t.t_ref()).unwrap();

        trace!("motif.t={} dist={dist}", t.as_ref().data().t_ref());


        //TODO check data

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 32)]
    async fn test_load_random_slices() -> Result<(), ThoError> {
        set_max_level(Trace);
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 10000)));
        let run_config_id = 1;
        let motif_id = 10u32;
        let motif_width = 100;

        let resource_a = Resource::new(1, 1, "tensor_a", 1000);
        let resource_b = Resource::new(2, 1, "tensor_b", 1000);
        let mut store = MockStoreBuilder::new()
            .with_list_random_slices_out_of_motif(motif_id, motif_width, None, Ok((0i64, run_config_id, vec![
                TensorIdSliceIdx(1, 100),
                TensorIdSliceIdx(1, 200),
                TensorIdSliceIdx(2, 100)
            ], HashMap::from([
                (1, resource_a.clone()),
                (2, resource_b.clone())
            ]))))
            .with_run_config_descr([(
                run_config_id,
                Ok(RunConfigRow {
                    id: run_config_id,
                    r#type: "FindMotifsRunConfig".to_string(),
                    descr: "test find motifs".to_string(),
                    run_on: "2026-01-01 00:00:00".to_string(),
                }),
            )])
            .with_list_run_config_values([(
                run_config_id,
                Ok(FindMotifsRunConfig::new_with_running_mean(Some(run_config_id), 100, motif_width, None, None).to_properties_iter(None).collect())
            )])
            .with_get_motif_meta( [(motif_id, Ok((0usize, run_config_id)))])
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

        let input = Arc::new(SlicesLoaderInput{ motif_id: DataId::from(&motif_id), limit: 10 });

        let slices:Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>> = random_slices_loader
            .run(
                Arc::clone(&input),
                thoosa_shared::perf_log!(thoosa_shared::sync_write_lock!(ctx))
                .branch(false),
            )
            .await?;

        let mut ctx_for_data = thoosa_shared::perf_log!(thoosa_shared::sync_write_lock!(ctx))
        .branch(false);
        let t = slices.borrow_to(0, Some(SyncStftSelector)).await?;

        let dist = average_euclidean_distance(t.t_ref())?;

        trace!("random slices: \t{} dist={dist}", t.t_ref());

        // assert_eq!(r, true);
        //TODO check data

        Ok(())
    }

}
