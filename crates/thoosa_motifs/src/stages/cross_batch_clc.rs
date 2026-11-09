use crate::stages::dists_tracker::DistsMask;
use crate::stages::find_motifs::{CrossUnfoldMaskFn, MasksCtx};
use crate::stages::prepare_cross_batch::PreparedCrossBatch;
use crate::stages::resource_batch::{BorrowedResourceBatch, ResourceBatch, ResourceWindows, checked_batch_range};
use crate::stages::slices_iter::{ResourceSlices, SlicesIter};
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use log::{debug, warn};
use parking_lot::RwLock;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::future::Future;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;
use tch::Tensor;
use thoosa_common::dims::ChannelsData;
use thoosa_common::instance_cache::InstanceCache;
use thoosa_common::ipc::cuda_worker::{AllTensors, WokerRunCtx, WorkerStageInput};
use thoosa_common::ipc::proxy::{DynSyncStageEnum, ProxyStage};
use thoosa_common::ipc::serialization::json::JsonIpcCodec;
use thoosa_common::ipc::worker_msg::WorkerInput;
use thoosa_common::ipc::wraps::SyncStageIpcWrapper;
use thoosa_common::ipc::wraps::WorkerWrapper;
use thoosa_common::runner::common::cachable::{BorrowRef, SelectRef, ThoDataRef};
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::sync_stage::SyncStage;
use thoosa_common::tch_error::tch_result;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::utils::tensor_mem_size;
use thoosa_digest_id::DigestHash;
use thoosa_digest_id_derive::DigestHash as DigestHashDerive;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::ipc_types::WorkerOutput;
use thoosa_shared::resource::Resource;
use thoosa_shared::tracked_await::t;
use thoosa_shared::traits::{Adjustable, Data, OutId, RunCtx, Stage, ThoSelector, WithId};
use thoosa_shared::{ThoError, mono_key};

pub const SYNC_CROSS_UNFOLD_CLC_ENDPOINT: &str = "CrossUnfoldClc";

// A prepared outer batch is a different representation, never an offset into a
// flat tensor list. Only the checked constructors below may attach its resources.
#[derive(Debug)]
enum CrossOuterBatch<'a, R> {
    Raw(BorrowedResourceBatch<'a, R>),
    Prepared { data: ThoDataRef<'a, PreparedCrossBatch>, resources: Vec<R> },
}

#[derive(Debug)]
pub struct CrossInput<'a, R: AsRef<Resource>> {
    id: SynKey,
    outer: CrossOuterBatch<'a, R>,
    inner: BorrowedResourceBatch<'a, R>,
}

impl<'a, R: AsRef<Resource> + Clone> CrossInput<'a, R> {
    pub fn new(outer: BorrowedResourceBatch<'a, R>, inner: BorrowedResourceBatch<'a, R>, id: SynKey) -> Self {
        Self { id: &id + Self::resources_id(outer.resources(), inner.resources()), outer: CrossOuterBatch::Raw(outer), inner }
    }

    fn resources_id(b_0: &[R], b_1: &[R]) -> SynKey {
        SynKey::new(b_0.iter().chain(b_1.iter()).map(|resource| mono_key!(resource.as_ref().id(), format!("resource_id{}", resource.as_ref().id()))).collect::<Vec<_>>())
    }

    pub(crate) fn from_prepared(data: ThoDataRef<'a, PreparedCrossBatch>, resources: Vec<R>, inner: BorrowedResourceBatch<'a, R>, id: SynKey) -> Result<Self, ThoError> {
        data.as_ref().validate_resources(&resources)?;
        Ok(Self { id: &id + Self::resources_id(&resources, inner.resources()), outer: CrossOuterBatch::Prepared { data, resources }, inner })
    }

    pub(crate) fn raw_outer(&self) -> Result<&BorrowedResourceBatch<'a, R>, ThoError> {
        match &self.outer {
            CrossOuterBatch::Raw(batch) => Ok(batch),
            CrossOuterBatch::Prepared { .. } => Err(ThoError::new("outer batch is already prepared")),
        }
    }

    pub fn tensors(&self) -> Vec<&ThoTensor<ChannelsData, SynKey>> {
        let outer = match &self.outer { CrossOuterBatch::Raw(batch) => batch.tensors(), CrossOuterBatch::Prepared { .. } => &[] };
        outer.iter().chain(self.inner.tensors()).map(ThoDataRef::as_ref).collect()
    }

    pub fn batch_0(&self) -> &[R] {
        match &self.outer { CrossOuterBatch::Raw(batch) => batch.resources(), CrossOuterBatch::Prepared { resources, .. } => resources }
    }

    pub fn batch_1(&self) -> &[R] { self.inner.resources() }

    fn borrow_inner_range(&self, range: std::ops::Range<usize>) -> Result<CrossInput<'_, R>, ThoError> {
        let inner = self.inner.borrow_range(range)?;
        let outer = match &self.outer {
            CrossOuterBatch::Raw(batch) => CrossOuterBatch::Raw(batch.borrow_range(0..batch.len())?),
            CrossOuterBatch::Prepared { data, resources } => CrossOuterBatch::Prepared { data: ThoDataRef::Borrowed(data.as_ref()), resources: resources.clone() },
        };
        Ok(CrossInput { id: &self.id + Self::resources_id(self.batch_0(), inner.resources()), outer, inner })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossInputIpcResource {
    id: u32,
    import_id: u32,
    file_stem: String,
    length: u32,
}

impl CrossInputIpcResource {
    fn from_resource(resource: &Resource) -> Self {
        Self {
            id: resource.id(),
            import_id: resource.import_id(),
            file_stem: resource.file_stem().to_string(),
            length: resource.length(),
        }
    }

    fn into_resource(self) -> Resource {
        Resource::new(self.id, self.import_id, self.file_stem, self.length)
    }
}

// Each wire batch carries its own metadata. Decoding must pass through the
// same validation as local construction before any worker tensor is accessed.
#[derive(Serialize, Deserialize)]
struct RawBatchWire<T> {
    tensors: Vec<T>,
    resources: Vec<CrossInputIpcResource>,
}

#[derive(Serialize, Deserialize)]
enum OuterBatchWire<T, P> {
    Raw(RawBatchWire<T>),
    Prepared { data: P, resources: Vec<CrossInputIpcResource> },
}

#[derive(Serialize, Deserialize)]
struct CrossInputWire<T, P> {
    id: SynKey,
    outer: OuterBatchWire<T, P>,
    inner: RawBatchWire<T>,
}

impl<'a> RawBatchWire<&'a ThoTensor<ChannelsData, SynKey>> {
    fn from_batch(batch: &'a BorrowedResourceBatch<'_, Arc<Resource>>) -> Self {
        Self { tensors: batch.tensors().iter().map(ThoDataRef::as_ref).collect(), resources: batch.resources().iter().map(|r| CrossInputIpcResource::from_resource(r)).collect() }
    }
}

impl RawBatchWire<ThoTensor<ChannelsData, SynKey>> {
    fn into_batch(self) -> Result<BorrowedResourceBatch<'static, Arc<Resource>>, ThoError> {
        ResourceBatch::try_new(self.tensors.into_iter().map(ThoDataRef::Owned).collect(), self.resources.into_iter().map(|r| Arc::new(r.into_resource())).collect())
    }
}

impl Serialize for CrossInput<'_, Arc<Resource>> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let outer = match &self.outer {
            CrossOuterBatch::Raw(batch) => OuterBatchWire::Raw(RawBatchWire::from_batch(batch)),
            CrossOuterBatch::Prepared { data, resources } => OuterBatchWire::Prepared { data: data.as_ref(), resources: resources.iter().map(|r| CrossInputIpcResource::from_resource(r)).collect() },
        };
        CrossInputWire { id: self.id.clone(), outer, inner: RawBatchWire::from_batch(&self.inner) }.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CrossInput<'static, Arc<Resource>> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = CrossInputWire::<ThoTensor<ChannelsData, SynKey>, PreparedCrossBatch>::deserialize(deserializer)?;
        let inner = wire.inner.into_batch().map_err(serde::de::Error::custom)?;
        let outer = match wire.outer {
            OuterBatchWire::Raw(batch) => CrossOuterBatch::Raw(batch.into_batch().map_err(serde::de::Error::custom)?),
            OuterBatchWire::Prepared { data, resources } => {
                let resources = resources.into_iter().map(|r| Arc::new(r.into_resource())).collect::<Vec<_>>();
                data.validate_resources(&resources).map_err(serde::de::Error::custom)?;
                CrossOuterBatch::Prepared { data: ThoDataRef::Owned(data), resources }
            }
        };
        Ok(Self { id: wire.id, outer, inner })
    }
}

#[typetag::serde]
impl WorkerInput for CrossInput<'static, Arc<Resource>> {}

impl CrossInput<'_, Arc<Resource>> {
    fn map_worker_tensors<'o>(&self, map: impl Fn(&ThoTensor<ChannelsData, SynKey>) -> Result<ThoDataRef<'o, ThoTensor<ChannelsData, SynKey>>, ThoError>, map_prepared: impl FnOnce(&PreparedCrossBatch) -> Result<PreparedCrossBatch, ThoError>) -> Result<CrossInput<'o, Arc<Resource>>, ThoError> {
        let outer = match &self.outer {
            CrossOuterBatch::Raw(batch) => CrossOuterBatch::Raw(batch.map_tensors(&map)?),
            CrossOuterBatch::Prepared { data, resources } => {
                let data = map_prepared(data.as_ref())?;
                data.validate_resources(resources)?;
                CrossOuterBatch::Prepared { data: ThoDataRef::Owned(data), resources: resources.clone() }
            }
        };
        Ok(CrossInput { id: self.id.clone(), outer, inner: self.inner.map_tensors(map)? })
    }
}

impl WorkerStageInput for CrossInput<'_, Arc<Resource>> {
    fn clone_refs(&self) -> Result<Box<dyn WorkerInput>, ThoError> {
        Ok(Box::new(self.map_worker_tensors(|tensor| tensor.clone_ref().map(ThoDataRef::Owned), PreparedCrossBatch::clone_refs_for_worker)?))
    }

    fn ref_to_tensor(&self, all_cuda_tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {
        self.map_worker_tensors(|tensor| tensor.ref_to_tensor(all_cuda_tensors, remove).map(ThoDataRef::Owned), |prepared| prepared.ref_to_tensor(all_cuda_tensors, remove))
    }
}

impl<'a, R: AsRef<Resource> + Debug + Send + Sync> WithId for CrossInput<'a, R> {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<'a, R: AsRef<Resource> + Debug + Send + Sync> Input for CrossInput<'a, R> {}

impl<'a, R: AsRef<Resource> + Debug + Send + Sync> Data for CrossInput<'a, R> {
    type Metrics = i64;

    fn size(&self) -> Self::Metrics {
        self.inner.len() as i64
    }

    fn tho_device_idx(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        self.inner.tensors().iter().map(|tensor| tensor.as_ref().mem_size()).sum::<u64>() + match &self.outer {
            CrossOuterBatch::Raw(batch) => batch.tensors().iter().map(|tensor| tensor.as_ref().mem_size()).sum(),
            CrossOuterBatch::Prepared { data, .. } => data.as_ref().mem_size(),
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CrossInputBatch1Selector {
    resources_batch_range: Range<i64>,
}

#[typetag::serde]
impl ThoSelector for CrossInputBatch1Selector {
    fn clone_box(&self) -> Box<dyn ThoSelector> {
        Box::new(self.clone())
    }
}

impl From<Range<i64>> for CrossInputBatch1Selector {
    fn from(value: Range<i64>) -> Self {
        Self { resources_batch_range: value }
    }
}

impl Adjustable for CrossInputBatch1Selector {}

impl<'a, R> BorrowRef for CrossInput<'a, R>
where
    R: AsRef<Resource> + Clone + Debug + Send + Sync + 'static,
{
    fn protect(&self, _device: usize) -> thoosa_common::runner::residency::ResidencyLease {
        thoosa_common::runner::residency::ResidencyLease::default()
    }

    type Out<'o>
        = CrossInput<'o, R>
    where
        Self: 'o;

    fn borrow_ref<'o>(&'o self, _to_device_idx: usize) -> impl Future<Output = Result<Self::Out<'o>, ThoError>> + Send
    where
        Self: 'o,
    {
        async move { self.borrow_inner_range(0..self.inner.len()) }
    }
}

impl<'a, R> SelectRef<CrossInputBatch1Selector> for CrossInput<'a, R>
where
    R: AsRef<Resource> + Clone + Debug + Send + Sync + 'static,
{
    fn select_ref<'o>(&'o self, to_device_idx: usize, selector: Option<CrossInputBatch1Selector>) -> impl Future<Output = Result<<Self as BorrowRef>::Out<'o>, ThoError>> + Send
    where
        CrossInputBatch1Selector: 'o,
        Self: 'o,
    {
        async move {
            let Some(selector) = selector else {
                return BorrowRef::borrow_ref(self, to_device_idx).await;
            };
            self.borrow_inner_range(checked_batch_range(&selector.resources_batch_range, self.inner.len())?)
        }
    }
}

pub trait CrossClcResultMerge: Debug + Send + Sync {
    fn merge_batch_1(self, other: Self) -> Result<Self, ThoError>
    where
        Self: Sized;

    fn to_cpu(self) -> BoxFuture<'static, Result<Self, ThoError>>
    where
        Self: Sized + Send + 'static,
    {
        async move { Ok(self) }.boxed()
    }

    fn merge_batch_1_many(self, others: Vec<Self>) -> Result<Self, ThoError>
    where
        Self: Sized,
    {
        let mut acc = self;
        for other in others {
            acc = acc.merge_batch_1(other)?;
        }
        Ok(acc)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, DigestHashDerive)]
#[digest(tag = "thoosa.UnfoldParams:v1")]
pub struct UnfoldParams {
    //TODO use in FindMotifsRunConfig
    dimension: i64,
    size: i64,
    step: i64,
}

impl UnfoldParams {
    pub fn new(dimension: i64, size: i64, step: i64) -> Self {
        Self { dimension, size, step }
    }

    pub(crate) fn validate(&self) -> Result<(), ThoError> {
        // Cross batches are [channels, samples]; concatenation and slice indexing
        // require unfolding the sample dimension with positive window and step.
        if self.dimension != 1 || self.size <= 0 || self.step <= 0 {
            return Err(ThoError::new("invalid cross-batch unfolding parameters"));
        }
        Ok(())
    }

    pub(crate) fn windows(&self, samples: i64) -> Result<i64, ThoError> {
        self.validate()?;
        if samples < 0 { return Err(ThoError::new("negative resource sample length")); }
        Ok(if samples < self.size { 0 } else { 1 + (samples - self.size) / self.step })
    }

    pub fn dimension(&self) -> i64 {
        self.dimension
    }

    pub fn size(&self) -> i64 {
        self.size
    }

    pub fn step(&self) -> i64 {
        self.step
    }

    pub fn new_id(&self) -> SynKey {
        //TODO add trait for init types
        SynKey::new([
            mono_key!("UnfoldParams", "UnfoldParams"), //TODO id
            mono_key!(self.blake3_hash(), format!("Hash for {:?}", self)),
        ])
    }
}

pub trait CrossBatchFn<R, CUC, C>: Send + Sync {
    fn call(&self, t_0: Tensor, t_1: Tensor, input: &CrossInput<'_, Arc<Resource>>, cross_batch_ctx: &CUC, ctx: &C) -> Result<R, ThoError>;
}

impl<F, R, CUC, C> CrossBatchFn<R, CUC, C> for F
where
    F: for<'a, 'b, 'c, 'd> Fn(Tensor, Tensor, &'b CrossInput<'a, Arc<Resource>>, &'c CUC, &'d C) -> Result<R, ThoError> + Send + Sync,
{
    fn call(&self, t_0: Tensor, t_1: Tensor, input: &CrossInput<'_, Arc<Resource>>, cross_batch_ctx: &CUC, ctx: &C) -> Result<R, ThoError> {
        self(t_0, t_1, input, cross_batch_ctx, ctx)
    }
}

impl<R, CUC, C> CrossBatchFn<R, CUC, C> for Box<dyn CrossBatchFn<R, CUC, C> + 'static> {
    fn call(&self, t_0: Tensor, t_1: Tensor, input: &CrossInput<'_, Arc<Resource>>, cross_batch_ctx: &CUC, ctx: &C) -> Result<R, ThoError> {
        self.as_ref().call(t_0, t_1, input, cross_batch_ctx, ctx)
    }
}

pub const CROSS_UNFOLD_CLC_ENDPOINT: &str = "CrossUnfoldClc";

pub trait DynCrossUnfoldClcStage<R, C>: Debug + Send + Sync + 'static
where
    R: Debug + Output + Input + Send + 'static,
    C: RunCtx + 'static,
{
    fn do_run_dyn<'a>(self: Arc<Self>, input: &CrossInput<'a, Arc<Resource>>, ctx: &C) -> Result<CrossUnfoldClcOutput<R>, ThoError>;

    fn out_id_dyn(&self, input_id: &SynKey) -> SynKey;
}

impl<T, R, C> DynCrossUnfoldClcStage<R, C> for T
where
    T: for<'a> SyncStage<In<'a> = CrossInput<'a, Arc<Resource>>, Out = CrossUnfoldClcOutput<R>, Ctx = C> + OutId + Send + Sync + 'static,
    R: Debug + Output + Input + Send + 'static,
    C: RunCtx + 'static,
{
    fn do_run_dyn<'a>(self: Arc<Self>, input: &CrossInput<'a, Arc<Resource>>, ctx: &C) -> Result<CrossUnfoldClcOutput<R>, ThoError> {
        self.do_run(input, ctx)
    }

    fn out_id_dyn(&self, input_id: &SynKey) -> SynKey {
        self.out_id(input_id)
    }
}

pub struct CrossUnfoldClc<F, R, CUC, C>
where
    F: CrossBatchFn<R, CUC, C>,
    C: RunCtx + 'static,
{
    batch_clc_fn: F,
    cross_batch_ctx: Box<dyn Fn(&C) -> Result<CUC, ThoError> + Send + Sync + 'static>,
    pub unfold_params: UnfoldParams,
    _pd: PhantomData<(C, fn() -> R)>,
}

impl<F, R, CUC, C> Debug for CrossUnfoldClc<F, R, CUC, C>
where
    F: CrossBatchFn<R, CUC, C>,
    C: RunCtx + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossUnfoldClc").field("unfold_params", &self.unfold_params).finish()
    }
}

impl<F, R, CUC, C> CrossUnfoldClc<F, R, CUC, C>
where
    F: CrossBatchFn<R, CUC, C>,
    C: RunCtx + 'static,
{
    pub fn new_local<CrossBatchCtxFn>(batch_clc_fn: F, cross_batch_ctx: CrossBatchCtxFn, unfold_params: UnfoldParams) -> Self
    where
        CrossBatchCtxFn: Fn(&C) -> Result<CUC, ThoError> + Send + Sync + 'static,
    {
        Self {
            batch_clc_fn,
            cross_batch_ctx: Box::new(cross_batch_ctx),
            unfold_params,
            _pd: PhantomData,
        }
    }
}

static CROSS_UNFOLD_CLC_INSTANCES: InstanceCache = InstanceCache::new();

impl<C> CrossUnfoldClc<CrossUnfoldMaskFn<C>, DistsMask, MasksCtx, C>
where
    C: RunCtx + 'static,
{
    pub async fn new_dyn_for_find_motifs(init: crate::stages::find_motifs::FindMotifsCrossUnfoldClcInit, tho_device_idx: usize) -> Result<Arc<DynSyncStageEnum<CrossUnfoldClc<CrossUnfoldMaskFn<C>, DistsMask, MasksCtx, C>>>, ThoError> {
        let id = init.new_id().to_string();

        CROSS_UNFOLD_CLC_INSTANCES
            .get_or_try_insert_recreatable_with::<C, _, _, _, _>(tho_device_idx, GlobalCtx::global().cuda_activity_gate(tho_device_idx), GlobalCtx::global().is_ipc(), id.clone(), move || async move {
                if GlobalCtx::global().is_ipc() {
                    Ok(Arc::new(DynSyncStageEnum::Proxy {
                        stage: Arc::new(ProxyStage::proxy_recreatable(SYNC_CROSS_UNFOLD_CLC_ENDPOINT, Some(id), tho_device_idx, Box::new(init)).await?),
                        init: (),
                    }))
                } else {
                    Ok(Arc::new(DynSyncStageEnum::Local(Arc::new(Self::new_for_find_motifs(init)?))))
                }
            })
            .await
    }
}

#[typetag::serde(name = "CrossUnfoldClcOutput_DistsMask")]
impl WorkerOutput for CrossUnfoldClcOutput<DistsMask> {}

pub fn sync_cross_unfold_clc_endpoint_init<'a>(_stage_id: Option<String>, init: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, _all_cuda_tensors: &'a AllTensors, _tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let init: Box<dyn Any> = init;
        let Ok(init) = init.downcast::<crate::stages::find_motifs::FindMotifsCrossUnfoldClcInit>() else {
            return Err(ThoError::new("CrossUnfoldClc endpoint received an unexpected initializer type"));
        };
        let init = *init;
        let stage = CrossUnfoldClc::<CrossUnfoldMaskFn<WokerRunCtx>, DistsMask, MasksCtx, WokerRunCtx>::new_for_find_motifs(init)?;
        // WorkerStageInput owns the decoded resource lists and tensor references for this run.
        Ok(Box::new(WorkerWrapper::<JsonIpcCodec, _, _, _>::new(Arc::new(stage), PhantomData)) as Box<dyn SyncStageIpcWrapper>)
    }
    .boxed()
}

impl<F, R, CUC, C> Adjustable for CrossUnfoldClc<F, R, CUC, C>
where
    F: CrossBatchFn<R, CUC, C> + Send + Sync,
    CUC: Send + Sync,
    R: Debug,
    C: RunCtx + 'static,
{
}

impl<F, R, CUC, C> OutId for CrossUnfoldClc<F, R, CUC, C>
where
    F: CrossBatchFn<R, CUC, C>,
    R: Debug,
    C: RunCtx + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CrossUnfoldClcOutput<R: Debug> {
    id: SynKey,
    clc_result: R,
    iter_0: SlicesIter,
    iter_1: SlicesIter,
}

impl<R: Debug> CrossUnfoldClcOutput<R> {
    pub fn new(clc_result: R, iter_0: SlicesIter, iter_1: SlicesIter, id: SynKey) -> Self {
        Self { id, clc_result, iter_0, iter_1 }
    }

    pub fn id(&self) -> &SynKey {
        &self.id
    }

    pub fn clc_result(self) -> R {
        self.clc_result
    }

    pub fn iter_0(&self) -> SlicesIter {
        self.iter_0.clone()
    }

    pub fn iter_1(&self) -> SlicesIter {
        self.iter_1.clone()
    }

    pub async fn replace_clc_result<OnBatchDone>(mut self, f: &OnBatchDone) -> Result<Self, ThoError>
    where
        OnBatchDone: Fn(R) -> BoxFuture<'static, Result<R, ThoError>> + Sync + Send + 'static,
    {
        self.clc_result = t!(f(self.clc_result)).await?;
        Ok(self)
    }
}

impl thoosa_common::ipc::cuda_worker::WorkerStageOutput for CrossUnfoldClcOutput<DistsMask> {
    fn encode_from_worker(self, all_cuda_tensors: &AllTensors, tho_device: &ThoDevice) -> Result<Self, ThoError> {
        let Self { id, clc_result, iter_0, iter_1 } = self;

        Ok(CrossUnfoldClcOutput::new(clc_result.encode_from_worker(all_cuda_tensors, tho_device)?, iter_0, iter_1, id))
    }
}

impl<R: Debug> WithId for CrossUnfoldClcOutput<R> {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}
impl<R: Debug> Data for CrossUnfoldClcOutput<R> {
    type Metrics = ();

    fn size(&self) -> Self::Metrics {
        ()
    }

    fn tho_device_idx(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        0 //TODO delegate to Result which should be Data
    }
}
impl<R: Debug> Input for CrossUnfoldClcOutput<R> {}

impl<R: Debug> Output for CrossUnfoldClcOutput<R> {}

impl<F, R, CUC, C> Stage for CrossUnfoldClc<F, R, CUC, C>
where
    F: CrossBatchFn<R, CUC, C>,
    R: Debug,
    C: RunCtx + 'static,
{
    type In<'a> = CrossInput<'a, Arc<Resource>>;

    type Out = CrossUnfoldClcOutput<R>;
    type Ctx = C;
}

impl<F, R, CUC, C> SyncStage for CrossUnfoldClc<F, R, CUC, C>
//CUC Cross Unfold Context
where
    F: CrossBatchFn<R, CUC, C>,
    R: Debug + Output + Input + Send + 'static,
    for<'a> CUC: 'a,
    C: RunCtx + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "CrossUnfoldClc::do_run"))]
    fn do_run<'s>(self: Arc<Self>, input: &Self::In<'s>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError>
    where
        Self: Sized,
    {
        let unfold_params = &self.unfold_params;
        // Filtering and window metadata are one operation; no caller reconstructs
        // correspondence by zipping a filtered tensor list with source resources.
        let (batch_0_t, resource_ids_sizes_0) = match &input.outer {
            CrossOuterBatch::Prepared { data, .. } => {
                let prepared = data.as_ref();
                if prepared.params() != *unfold_params {
                    return Err(ThoError::new("prepared outer batch does not match cross comparison parameters"));
                }
                // This alias stays in the synchronous calculation; the registered
                // prepared batch owns storage between invocations.
                (tch_result(prepared.tensor().local()?.f_alias())?, prepared.slices())
            }
            CrossOuterBatch::Raw(batch) => {
                let (tensor, layout) = unfold_and_cat(batch, unfold_params)?;
                (tensor, resource_slices(layout))
            }
        };
        let (batch_1_t, layout) = unfold_and_cat(&input.inner, unfold_params)?;
        let resource_ids_sizes_1 = resource_slices(layout);

        debug!("CrossUnfoldClc::do_run: input.id={:?} batch_0_t={batch_0_t:?}  batch_1_t={batch_1_t:?} mem_0={} mem_1={}", input.id, tensor_mem_size(&batch_0_t), tensor_mem_size(&batch_1_t));

        let cross_batch_ctx = (self.cross_batch_ctx)(ctx)?;
        let result = self.batch_clc_fn.call(batch_0_t, batch_1_t, &input /*TODO it's only used to check if we're on diagonal*/, &cross_batch_ctx, ctx).map(|v| {
            CrossUnfoldClcOutput::new(
                v,
                SlicesIter::new_with_offsets(unfold_params.step as usize, resource_ids_sizes_0),
                SlicesIter::new_with_offsets(unfold_params.step as usize, resource_ids_sizes_1),
                input.id.clone(),
            )
        });

        debug!("CrossUnfoldClc::do_run: result={result:?}");

        result
    }
}

fn resource_slices<R: AsRef<Resource>>(layout: Vec<ResourceWindows<R>>) -> Vec<ResourceSlices> {
    layout.into_iter().filter(|entry| entry.windows > 0).map(|entry| ResourceSlices::new(entry.resource.as_ref().id(), entry.windows as usize)).collect()
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "cross_batch_clc/unfold_and_cat"))]
pub(crate) fn unfold_and_cat<R: Clone>(input: &BorrowedResourceBatch<'_, R>, params: &UnfoldParams) -> Result<(Tensor, Vec<ResourceWindows<R>>), ThoError> {
    params.validate()?;
    let mut tensors = Vec::with_capacity(input.len());
    let mut layout = Vec::with_capacity(input.len());
    for (resource, tensor, samples) in input.iter() {
        let windows = params.windows(samples)?;
        if windows > 0 {
            // Tensor failures must reach OOM handling, not silently drop a resource.
            tensors.push(tch_result(tensor.as_ref().local()?.f_unfold(params.dimension, params.size, params.step))?);
        } else {
            warn!("cross_batch_clc/unfold_and_cat: tensor={:?} samples={samples} params={params:?} skipped", tensor.as_ref().id());
        }
        layout.push(ResourceWindows { resource: resource.clone(), samples, windows });
    }
    if tensors.is_empty() {
        return Err(ThoError::new("batch is empty"));
    }
    Ok((tch_result(Tensor::f_concat(&tensors, 1))?, layout))
}

#[cfg(test)]
mod tests {
    use crate::stages::cross_batch_clc::{CrossBatchFn, CrossInput, CrossUnfoldClc, UnfoldParams};
    use crate::stages::dists_tracker::{DistsMask, DistsMaskType};
    use crate::stages::find_motifs::{MasksCtx, MeanDistsCtx};
    use crate::stages::resource_batch::ResourceBatch;
    use thoosa_common::runner::common::cachable::ThoDataRef;
    use log::debug;
    use std::sync::Arc;
    use tch::Device::{Cpu, Cuda};
    use tch::Tensor;
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::runner::run_ctx::RunCtxImpl;
    use thoosa_common::runner::sync_stage::SyncStage;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::utils::lin_t;
    use thoosa_shared::ThoError;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_shared::resource::Resource;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_cross_batch_clc() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        GlobalCtx::init_devices(2)?;

        let cross_batch_fn: Box<dyn CrossBatchFn<DistsMask, MasksCtx, RunCtxImpl>> = Box::new(|t_0: Tensor, t_1: Tensor, input: &CrossInput<Arc<Resource>>, _ctx: &MasksCtx, _: &RunCtxImpl| {
            debug!("t_0=\n{t_0}");
            debug!("t_1=\n{t_1}");
            debug!("input={input:?}");

            let tensor_00 = Tensor::from_slice(&[
                0.00, 0.01, 0.01, 0.02, 1.00, 1.01, 1.01, 1.02, 1.02, 1.03, /* 2.00, 2.01,*/
                0.10, 0.11, 0.11, 0.12, 1.10, 1.11, 1.11, 1.12, 1.12, 1.13, /* 2.10, 2.11,*/
                0.20, 0.21, 0.21, 0.22, 1.20, 1.21, 1.21, 1.22, 1.22, 1.23, /* 2.20, 2.21,*/
            ])
            .view((3, 5, 2));

            assert!(t_0.allclose(&tensor_00, 0.1, 0.1, false));

            let tensor_01 = Tensor::from_slice(&[2.00, 2.01, 2.10, 2.11, 2.20, 2.21]).view((3, 1, 2));
            assert!(t_1.allclose(&tensor_01, 0.1, 0.1, false));

            let channels = t_0.size()[0];
            let slices_0 = t_0.size()[1];
            let slices_1 = t_1.size()[1];
            let total = (channels * slices_0 * slices_1) as u32;
            let dists = lin_t(total).view((channels, slices_0, slices_1));
            let mask = dists.lt(3.0);

            let mask_type = if mask.device().eq(&Cpu) { DistsMaskType::AsIs } else { DistsMaskType::MaskPacked(mask.size(), Some(dists.size())) };
            Ok(DistsMask::new(mask, Some(dists), mask_type)?)
        });
        let cross_batch_clc = Arc::new(CrossUnfoldClc::new_local(cross_batch_fn, |_| Ok(MasksCtx::AccumulatedMeanDist { k: 1.0, ctx: MeanDistsCtx::new(3, Cuda(0))? }), UnfoldParams::new(1, 2, 1)));

        let tensor_00 = ThoTensor::new(SynKey::new([mono_key!(0u32, "tensor_00")]), Tensor::from_slice(&[0.00, 0.01, 0.02, 0.10, 0.11, 0.12, 0.20, 0.21, 0.22]).view((3, 3)))?;
        let tensor_01 = ThoTensor::new(SynKey::new([mono_key!(1u32, "tensor_01")]), Tensor::from_slice(&[1.00, 1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13, 1.20, 1.21, 1.22, 1.23]).view((3, 4)))?;

        let tensor_1 = ThoTensor::new(SynKey::new([mono_key!(2u32, "tensor_1")]), Tensor::from_slice(&[2.00, 2.01, 2.10, 2.11, 2.20, 2.21]).view((3, 2)))?;

        let resource_00 = Arc::new(Resource::new(0, 0, "file_00", 10));
        let resource_01 = Arc::new(Resource::new(1, 0, "file_01", 10));
        let chunk_0_resources = vec![Arc::clone(&resource_00), Arc::clone(&resource_01)];

        let resource_1 = Arc::new(Resource::new(2, 0, "file_1", 10));
        let chunk_1_resources = vec![Arc::clone(&resource_1)];

        let input = CrossInput::new(
            ResourceBatch::try_new(vec![ThoDataRef::Borrowed(&tensor_00), ThoDataRef::Borrowed(&tensor_01)], chunk_0_resources)?,
            ResourceBatch::try_new(vec![ThoDataRef::Borrowed(&tensor_1)], chunk_1_resources)?,
            "test".into(),
        );
        let ctx = RunCtxImpl::new(1, "test");
        let out = cross_batch_clc.do_run(&input, &ctx)?;

        let (mask, dists) = out.clc_result.unpack()?;
        let dists = dists.expect("dists");
        debug!("out=\n{mask}");

        assert_eq!(mask.size(), vec![3, 5, 1]);
        assert_eq!(dists.size(), vec![3, 5, 1]);

        let expected_dists = lin_t(15).view((3, 5, 1));
        let expected_mask = expected_dists.lt(3.0);
        assert!(dists.allclose(&expected_dists, 0.0, 0.0, false));
        assert!(mask.equal(&expected_mask));

        Ok(())
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    use tch::{Kind, kind::FLOAT_CPU};
    use thoosa_common::ipc::worker_msg::CudaWorkerId;

    fn resources() -> Vec<Arc<Resource>> {
        // Source lengths intentionally differ from post-filter tensor lengths.
        (10..13).map(|id| Arc::new(Resource::new(id, 1, format!("resource_{id}"), 100))).collect()
    }

    fn remote_batch() -> Result<BorrowedResourceBatch<'static, Arc<Resource>>, ThoError> {
        let worker = CudaWorkerId { device_idx: 1, generation: thoosa_shared::CudaDeviceGeneration(7) };
        ResourceBatch::try_new([5, 1, 7].into_iter().enumerate().map(|(i, len)| ThoDataRef::Owned(ThoTensor::new_ref(format!("tensor_{i}").into(), i as u64, vec![1, len], Kind::Float, None, worker))).collect(), resources())
    }

    #[test]
    fn test_unfold_filters_with_resource_boundaries() -> Result<(), ThoError> {
        let tensors = [5, 1, 7].into_iter().map(|len| Ok(ThoDataRef::Owned(ThoTensor::new(format!("len_{len}").into(), tch_result(Tensor::f_zeros([1, len], FLOAT_CPU))?)?))).collect::<Result<_, ThoError>>()?;
        let batch = ResourceBatch::try_new(tensors, resources())?;
        let (tensor, layout) = unfold_and_cat(&batch, &UnfoldParams::new(1, 3, 2))?;
        assert_eq!(tensor.size(), vec![1, 5, 3]);
        assert_eq!(layout.iter().map(|entry| (entry.resource.as_ref().id(), entry.samples, entry.windows)).collect::<Vec<_>>(), vec![(10, 5, 2), (11, 1, 0), (12, 7, 3)]);
        assert_eq!(resource_slices(layout), vec![ResourceSlices::new(10, 2), ResourceSlices::new(12, 3)]);
        assert!(unfold_and_cat(&batch, &UnfoldParams::new(0, 3, 2)).is_err());
        assert!(unfold_and_cat(&batch, &UnfoldParams::new(1, 3, 0)).is_err());
        assert!(unfold_and_cat(&batch, &UnfoldParams::new(1, 0, 2)).is_err());
        assert!(unfold_and_cat(&batch, &UnfoldParams::new(1, 8, 2)).is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_batch_selection_and_wire_validation() -> anyhow::Result<()> {
        let input = CrossInput::new(remote_batch()?, remote_batch()?, "cross".into());
        let selected = input.select_ref(1, Some((1..3).into())).await?;
        assert_eq!(selected.inner.resources().iter().map(|r| r.as_ref().id()).collect::<Vec<_>>(), vec![11, 12]);
        assert_eq!(selected.inner.lengths(), &[1, 7]);
        assert!(input.select_ref(1, Some((-1..1).into())).await.is_err());
        assert!(input.select_ref(1, Some((0..4).into())).await.is_err());
        let wire = serde_json::to_value(&selected)?;
        let decoded: CrossInput<'static, Arc<Resource>> = serde_json::from_value(wire.clone())?;
        assert_eq!(decoded.inner.resources().iter().map(|r| r.as_ref().id()).collect::<Vec<_>>(), vec![11, 12]);
        assert_eq!(decoded.inner.lengths(), &[1, 7]);
        assert_eq!(decoded.raw_outer()?.lengths(), &[5, 1, 7]);
        // Reference cloning uses the same paired transformation as worker materialization.
        let cloned = selected.clone_refs()?;
        assert_eq!(serde_json::to_value(cloned)?.get("inner"), wire.get("inner"));
        for outer in [false, true] {
            let mut malformed = wire.clone();
            let batch = if outer { &mut malformed["outer"]["Raw"] } else { &mut malformed["inner"] };
            batch["resources"].as_array_mut().expect("resource array").pop();
            assert!(serde_json::from_value::<CrossInput<'static, Arc<Resource>>>(malformed).is_err());
        }
        let mut malformed = wire;
        malformed["inner"]["tensors"][0]["size"] = serde_json::json!([1, -1]);
        assert!(serde_json::from_value::<CrossInput<'static, Arc<Resource>>>(malformed).is_err());
        Ok(())
    }
}
