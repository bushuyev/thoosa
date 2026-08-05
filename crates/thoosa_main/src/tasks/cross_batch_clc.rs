use crate::tasks::dists_tracker::DistsMask;
use crate::tasks::slices_iter::{ResourceSlices, SlicesIter};
use async_trait::async_trait;
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use log::{debug, trace, warn};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::future::Future;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;
use tch::kind::FLOAT_CPU;
use tch::{TchError, Tensor};
use thoosa_common::dims::ChannelsData;
use thoosa_common::ipc::cuda_wokrer::WokerRunCtx;
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_shared::traits::MaxParallelCtx;
use thoosa_common::runner::common::cachable::{BorrowRef, DevicesWrap, SelectRef, ThoDataRef};
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::utils::tensor_mem_size;
use thoosa_motifs::distance::clc_dist_d2;
use thoosa_proxy_derive::Proxy;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::{MonoKey, SynKey};
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::resource::Resource;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, Data, OutId, RunCtx, Stage, SyncStage, ThoSelector, WithId};
use thoosa_shared::{TensorIdSliceIdx, ThoError, mono_key};

pub(crate) const SYNC_CROSS_UNFOLD_CLC_ENDPOINT: &str = "CrossUnfoldClc";

#[derive(Debug)]
pub struct CrossInput<'a, R: AsRef<Resource>> {
    id: SynKey,
    tensors: Vec<ThoDataRef<'a, ThoTensor<ChannelsData, SynKey>>>,
    resources_0: &'a [R],
    resources_1: &'a [R],
}

impl<'a, R: AsRef<Resource>> CrossInput<'a, R> {

    pub fn new(
        tensors: &'a [&'a ThoTensor<ChannelsData, SynKey>],
        b_0: &'a [R],
        b_1: &'a [R],
        id: SynKey,
    ) -> Self {


        let batch_id = {
            let batch_0_ids = b_0.iter().map(|r| r.as_ref().id()).collect::<Vec<_>>();
            let batch_1_ids = b_1.iter().map(|r| r.as_ref().id()).collect::<Vec<_>>();
            mono_key!(//TODO
                format!("cross_input_b0={batch_0_ids:?}_b1={batch_1_ids:?}"),
                "CrossInput"
            )
        };

        Self {
            id: &id + batch_id,
            tensors: tensors.iter().map(|tensor| ThoDataRef::Borrowed(*tensor)).collect(),
            resources_0: b_0,
            resources_1: b_1,
        }
    }

    pub(crate) fn from_data_refs(tensors: Vec<ThoDataRef<'a, ThoTensor<ChannelsData, SynKey>>>, b_0: &'a [R], b_1: &'a [R], id: SynKey) -> Self {
        let batch_0_ids = b_0.iter().map(|r| r.as_ref().id()).collect::<Vec<_>>();
        let batch_1_ids = b_1.iter().map(|r| r.as_ref().id()).collect::<Vec<_>>();
        let batch_id = mono_key!(format!("cross_input_b0={batch_0_ids:?}_b1={batch_1_ids:?}"), "CrossInput");
        Self { id: &id + batch_id, tensors, resources_0: b_0, resources_1: b_1 }
    }

    pub(crate) fn from_decoded_parts(tensors: Vec<ThoDataRef<'a, ThoTensor<ChannelsData, SynKey>>>, b_0: &'a [R], b_1: &'a [R], id: SynKey) -> Self {
        Self { id, tensors, resources_0: b_0, resources_1: b_1 }
    }

    pub fn tensors(&self) -> Vec<&ThoTensor<ChannelsData, SynKey>> {
        self.tensors.iter().map(ThoDataRef::as_ref).collect()
    }

    pub fn batch_0(&self) -> &'a [R] {
        self.resources_0
    }

    pub fn batch_1(&self) -> &'a [R] {
        self.resources_1
    }


}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CrossInputIpcResource {
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

#[derive(Debug, Serialize)]
pub(crate) struct CrossInputFindMotifsIpcInputRef<'a> {
    id: &'a SynKey,
    tensors: Vec<&'a ThoTensor<ChannelsData, SynKey>>,
    resources_0: Vec<CrossInputIpcResource>,
    resources_1: Vec<CrossInputIpcResource>,
    max_other_parallel: Option<thoosa_common::runner::adjusters::max_parallel::MaxParallel>,
}

impl<'a> CrossInputFindMotifsIpcInputRef<'a> {
    pub(crate) fn new<C>(input: &'a CrossInput<'_, Arc<Resource>>, ctx: &C) -> Self
    where
        C: MaxParallelCtx,
    {
        Self {
            id: input.id(),
            tensors: input.tensors(),
            resources_0: input.resources_0.iter().map(|resource| CrossInputIpcResource::from_resource(resource.as_ref())).collect(),
            resources_1: input.resources_1.iter().map(|resource| CrossInputIpcResource::from_resource(resource.as_ref())).collect(),
            max_other_parallel: Some(ctx.max_other_parallel()),
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct CrossInputFindMotifsIpcInput {
    id: SynKey,
    tensors: Vec<ThoTensor<ChannelsData, SynKey>>,
    resources_0: Vec<CrossInputIpcResource>,
    resources_1: Vec<CrossInputIpcResource>,
    max_other_parallel: Option<thoosa_common::runner::adjusters::max_parallel::MaxParallel>,
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
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        self.resources_1.len() as i64
    }

    fn device(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        self.tensors.iter().map(|tensor| tensor.as_ref().mem_size()).sum()
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
    R: AsRef<Resource> + Debug + Send + Sync + 'static,
{
    type Out<'o> = CrossInput<'o, R> where Self: 'o;

    fn borrow_to<'o>(&'o self, _to_device_idx: usize) -> impl Future<Output = Result<Self::Out<'o>, ThoError>> + Send
    where
        Self: 'o,
    {
        async move {
            let tensors = self.tensors.iter().map(|tensor| ThoDataRef::Borrowed(tensor.as_ref())).collect();
            Ok(CrossInput::from_data_refs(tensors, self.resources_0, self.resources_1, self.id.clone()))
        }
    }
}

impl<'a, R> SelectRef<CrossInputBatch1Selector> for CrossInput<'a, R>
where
    R: AsRef<Resource> + Debug + Send + Sync + 'static,
{
    fn borrow_to<'o>(&'o self, to_device_idx: usize, selector: Option<CrossInputBatch1Selector>) -> impl Future<Output = Result<<Self as BorrowRef>::Out<'o>, ThoError>> + Send
    where
        CrossInputBatch1Selector: 'o,
        Self: 'o,
    {
        async move {
            let Some(selector) = selector else {
                return BorrowRef::borrow_to(self, to_device_idx).await;
            };
            let batch_0_len = self.resources_0.len();
            let range = selector.resources_batch_range.start as usize..selector.resources_batch_range.end as usize;
            let mut tensors = Vec::with_capacity(batch_0_len + range.len());
            tensors.extend(self.tensors[..batch_0_len].iter().map(|tensor| ThoDataRef::Borrowed(tensor.as_ref())));
            tensors.extend(self.tensors[batch_0_len + range.start..batch_0_len + range.end].iter().map(|tensor| ThoDataRef::Borrowed(tensor.as_ref())));
            Ok(CrossInput::from_data_refs(tensors, self.resources_0, &self.resources_1[range], self.id.clone()))
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct UnfoldParams { //TODO use in FindMotifsRunConfig
    dimension: i64,
    size: i64,
    step: i64,
}

impl UnfoldParams {
    pub fn new(dimension: i64, size: i64, step: i64) -> Self {
        Self {
            dimension,
            size,
            step,
        }
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
}

pub trait CrossBatchFn<R, CUC, C>: Send + Sync {
    fn call(
        &self,
        t_0: Tensor,
        t_1: Tensor,
        input: &CrossInput<'_, Arc<Resource>>,
        cross_batch_ctx: &CUC,
        ctx: &C,
    ) -> Result<R, ThoError>;
}

impl<F, R, CUC, C> CrossBatchFn<R, CUC, C> for F
where
    F: for<'a, 'b, 'c, 'd> Fn(Tensor, Tensor, &'b CrossInput<'a, Arc<Resource>>, &'c CUC, &'d C) -> Result<R, ThoError> + Send + Sync,
{
    fn call(
        &self,
        t_0: Tensor,
        t_1: Tensor,
        input: &CrossInput<'_, Arc<Resource>>,
        cross_batch_ctx: &CUC,
        ctx: &C,
    ) -> Result<R, ThoError> {
        self(t_0, t_1, input, cross_batch_ctx, ctx)
    }
}

impl<R, CUC, C> CrossBatchFn<R, CUC, C> for Box<dyn CrossBatchFn<R, CUC, C> + 'static> {
    fn call(
        &self,
        t_0: Tensor,
        t_1: Tensor,
        input: &CrossInput<'_, Arc<Resource>>,
        cross_batch_ctx: &CUC,
        ctx: &C,
    ) -> Result<R, ThoError> {
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

#[derive(Debug)]
pub struct CrossUnfoldClcDynSyncStage<R, C>
where
    R: Debug + Output + Input + Send + 'static,
    C: RunCtx + 'static,
{
    inner: Arc<dyn DynCrossUnfoldClcStage<R, C>>,
}

impl<R, C> CrossUnfoldClcDynSyncStage<R, C>
where
    R: Debug + Output + Input + Send + 'static,
    C: RunCtx + 'static,
{
    pub fn new(inner: Arc<dyn DynCrossUnfoldClcStage<R, C>>) -> Self {
        Self { inner }
    }
}

impl<R, C> Stage for CrossUnfoldClcDynSyncStage<R, C>
where
    R: Debug + Output + Input + Send + 'static,
    C: RunCtx + 'static,
{
    type In<'a> = CrossInput<'a, Arc<Resource>>;
    type Out = CrossUnfoldClcOutput<R>;
    type Ctx = C;
}

impl<R, C> SyncStage for CrossUnfoldClcDynSyncStage<R, C>
where
    R: Debug + Output + Input + Send + 'static,
    C: RunCtx + 'static,
{
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        Arc::clone(&self.inner).do_run_dyn(input, ctx)
    }
}

impl<R, C> OutId for CrossUnfoldClcDynSyncStage<R, C>
where
    R: Debug + Output + Input + Send + 'static,
    C: RunCtx + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        self.inner.out_id_dyn(input_id)
    }
}

impl<R, C> Adjustable for CrossUnfoldClcDynSyncStage<R, C>
where
    R: Debug + Output + Input + Send + 'static,
    C: RunCtx + 'static,
{
}

#[derive(Debug)]
struct CrossUnfoldClcFindMotifsIpcWrapper {
    stage: Arc<CrossUnfoldClc<crate::tasks::find_motifs::CrossUnfoldMaskFn<RunCtxImpl>, DistsMask, crate::tasks::find_motifs::MasksCtx, RunCtxImpl>>,
}

impl thoosa_common::ipc::json_ser::SyncStageIpcWrapper for CrossUnfoldClcFindMotifsIpcWrapper {
    fn do_run(
        &self,
        _input: &[u8],
        _all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>,
        _tho_device: &ThoDevice,
        _run_ctx: &WokerRunCtx,
    ) -> Result<Vec<u8>, ThoError> {
        type __ProxyIpcCodec = thoosa_common::ipc::codec::JsonIpcCodec;
        let input = <__ProxyIpcCodec as thoosa_common::ipc::codec::IpcCodec>::decode::<CrossInputFindMotifsIpcInput>(_input)?;

        debug!("CrossUnfoldClcFindMotifsIpcWrapper::do_run: input={input:?}");

        let tensors = input
            .tensors
            .into_iter()
            .map(|tensor| {
                ThoTensor::<ChannelsData, SynKey>::ref_to_tensor(&_all_cuda_tensors, tensor)

            })
            .collect::<Result<Vec<_>, ThoError>>()?;

        let resources_0 = input.resources_0.into_iter().map(|resource| Arc::new(resource.into_resource())).collect::<Vec<_>>();
        let resources_1 = input.resources_1.into_iter().map(|resource| Arc::new(resource.into_resource())).collect::<Vec<_>>();
        let tensor_refs = tensors.iter().map(ThoDataRef::Borrowed).collect::<Vec<_>>();
        let cross_input = CrossInput::from_decoded_parts(tensor_refs, resources_0.as_slice(), resources_1.as_slice(), input.id);

        debug!("CrossUnfoldClcFindMotifsIpcWrapper::do_run: cross_input={cross_input:?}");

        let mut ctx = RunCtxImpl::new(1, SYNC_CROSS_UNFOLD_CLC_ENDPOINT);
        if let Some(max_other_parallel) = input.max_other_parallel {
            ctx.set_max_other_parallel(max_other_parallel);
        }
        let out = Arc::clone(&self.stage).do_run(&cross_input, &ctx)?;

        debug!("CrossUnfoldClcFindMotifsIpcWrapper::do_run: out={out:?}");

        <CrossUnfoldClcOutput<DistsMask> as thoosa_common::ipc::cuda_wokrer::WorkerStageOutput>::encode_from_worker::<__ProxyIpcCodec>(out, _all_cuda_tensors, _tho_device)
    }
}

#[derive(Debug, Proxy)]
#[proxy(
    common = "thoosa_common",
    name = "ProxyCrossUnfoldClc",
    endpoint = "SYNC_CROSS_UNFOLD_CLC_ENDPOINT",
    new_args = "init: crate::tasks::find_motifs::FindMotifsCrossUnfoldClcInit, tho_device_idx: usize",
    init = "init.clone()",
    out_id = "input_id.clone()",
    run_input = "CrossInputFindMotifsIpcInputRef::new(input, ctx)",
    endpoint_init = "sync_cross_unfold_clc_endpoint_init",
    init_type = "crate::tasks::find_motifs::FindMotifsCrossUnfoldClcInit",
    init_stage = "CrossUnfoldClc::new_for_find_motifs(init)?",
    wrapper = "CrossUnfoldClcFindMotifsIpcWrapper"
)]
struct CrossUnfoldClcProxyTarget {
    base: CrossUnfoldClc<crate::tasks::find_motifs::CrossUnfoldMaskFn<thoosa_common::runner::run_ctx_2::RunCtxImpl>, crate::tasks::dists_tracker::DistsMask, crate::tasks::find_motifs::MasksCtx, thoosa_common::runner::run_ctx_2::RunCtxImpl>,
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
    pub fn new<CrossBatchCtxFn>(batch_clc_fn: F, cross_batch_ctx: CrossBatchCtxFn, unfold_params: UnfoldParams) -> CrossUnfoldClcDynSyncStage<R, C>
    where
        CrossBatchCtxFn: Fn(&C) -> Result<CUC, ThoError> + Send + Sync + 'static,
        F: Send + Sync + 'static,
        CUC: Send + Sync + 'static,
        R: Debug + Output + Input + Send + 'static,
    {
        CrossUnfoldClcDynSyncStage::new(Arc::new(Self::new_local(batch_clc_fn, cross_batch_ctx, unfold_params)))
    }

    pub(crate) fn new_local<CrossBatchCtxFn>(batch_clc_fn: F, cross_batch_ctx: CrossBatchCtxFn, unfold_params: UnfoldParams) -> Self
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

#[derive(Debug)]
struct CrossUnfoldClcProxyCtx<C>
where
    C: RunCtx + 'static,
{
    proxy: Arc<ProxyCrossUnfoldClc>,
    _pd: PhantomData<C>,
}

impl<C> Adjustable for CrossUnfoldClcProxyCtx<C>
where
    C: RunCtx + 'static,
{
}

impl<C> OutId for CrossUnfoldClcProxyCtx<C>
where
    C: RunCtx + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        self.proxy.out_id(input_id)
    }
}

impl<C> Stage for CrossUnfoldClcProxyCtx<C>
where
    C: RunCtx + 'static,
{
    type In<'a> = CrossInput<'a, Arc<Resource>>;
    type Out = CrossUnfoldClcOutput<DistsMask>;
    type Ctx = C;
}

impl<C> SyncStage for CrossUnfoldClcProxyCtx<C>
where
    C: RunCtx + 'static,
{
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let ctx = (ctx as &dyn Any).downcast_ref::<RunCtxImpl>().ok_or_else(|| ThoError::new("ProxyCrossUnfoldClc requires RunCtxImpl"))?;
        Arc::clone(&self.proxy).do_run(input, ctx)
    }
}

impl<C> CrossUnfoldClc<crate::tasks::find_motifs::CrossUnfoldMaskFn<C>, DistsMask, crate::tasks::find_motifs::MasksCtx, C>
where
    C: RunCtx + 'static,
{
    pub(crate) async fn new_dyn_for_find_motifs(init: crate::tasks::find_motifs::FindMotifsCrossUnfoldClcInit, tho_device_idx: usize) -> Result<CrossUnfoldClcDynSyncStage<DistsMask, C>, ThoError> {
        if GlobalCtx::global().is_ipc() {
            Ok(CrossUnfoldClcDynSyncStage::new(Arc::new(CrossUnfoldClcProxyCtx::<C> {
                proxy: Arc::new(ProxyCrossUnfoldClc::new(init, tho_device_idx).await?),
                _pd: PhantomData,
            })))
        } else {
            Ok(CrossUnfoldClcDynSyncStage::new(Arc::new(Self::new_for_find_motifs(init)?)))
        }
    }
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
        Self {
            id,
            clc_result,
            iter_0,
            iter_1,
        }
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
    where OnBatchDone: Fn(R) -> BoxFuture<'static, Result<R, ThoError>> + Sync + Send + 'static,{
        self.clc_result = track_await(f(self.clc_result)).await?;
        Ok(self)
    }
}

impl thoosa_common::ipc::cuda_wokrer::WorkerStageOutput for CrossUnfoldClcOutput<DistsMask> {
    fn encode_from_worker<C: thoosa_common::ipc::codec::IpcCodec>(
        self,
        all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>,
        tho_device: &ThoDevice,
    ) -> Result<Vec<u8>, ThoError> {
        let Self {
            id,
            clc_result,
            iter_0,
            iter_1,
        } = self;
        let clc_result = clc_result.into_worker_ipc_refs(all_cuda_tensors, tho_device)?;
        C::encode(&CrossUnfoldClcOutput::new(clc_result, iter_0, iter_1, id))
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
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        ()
    }

    fn device(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        0//TODO delegate to Result which should be Data
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


impl<F, R, CUC, C> SyncStage for CrossUnfoldClc<F, R, CUC, C>//CUC Cross Unfold Context
where
    F: CrossBatchFn<R, CUC, C>,
    R: Debug + Output + Input + Send + 'static,
    for<'a> CUC: 'a,
    C: RunCtx + 'static,
{


    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CrossUnfoldClc::do_run"))]
    fn do_run<'s>(self: Arc<Self>, input: &Self::In<'s>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError>
    where
        Self: Sized
    {
        assert!(!input.resources_0.is_empty(), "CrossUnfoldClc::do_run: no input resources");
        let unfold_params = &self.unfold_params;

        debug!("CrossUnfoldClc::do_run: input.id={:?} input={input:?} unfold_params={unfold_params:?}", input.id);
        debug!("CrossUnfoldClc::do_run: input.id={:?} input.tensors.len={} input.resources_0.len={} input.resources_1.len={}", input.id, input.tensors.len(), input.resources_0.len(), input.resources_1.len());

        //FIXME tensors which are too small (because of freq_filter) are already removed from batch_0_t and batch_1_t but still present in sizes (as 0) so that resources_0/1 can be filtered. Do it all in unfold_and_cat
        let tensors = input.tensors();
        let (batch_0_t, batch_0_sizes) =  unfold_and_cat(&tensors[0..input.resources_0.len()], &unfold_params)?;
        // let (batch_0_t, batch_0_sizes) =  cat_and_unfold(&input.tensors[0..input.resources_0.len()], &unfold_params)?; TODO doesn't work because sizes need to match splits to track mask, i.e. single spit can't come from more than one resource

        debug!("CrossUnfoldClc::do_run: input.id={:?} batch_0_sizes={batch_0_sizes:?}", input.id);


        //TODO if tensor is filtered because it's less than window
        // assert_eq!(batch_0_sizes.len(), input.resources_0.len());
        let resource_ids_sizes_0: Vec<_> = input
            .resources_0
            .into_iter()
            .zip(batch_0_sizes.into_iter())
            .filter(|v|v.1 != 0)
            .map(|(resource, size)| {
                ResourceSlices::new(resource.as_ref().id(), size as usize)
            })
            .collect();


        let (batch_1_t, batch_1_sizes) =  unfold_and_cat(&tensors[input.resources_0.len()..], &unfold_params)?;

        // assert_eq!(batch_1_sizes.len(), input.resources_1.len());
        debug!("CrossUnfoldClc::do_run: input.id={:?} batch_1_sizes={batch_1_sizes:?}", input.id);

        let resource_ids_sizes_1: Vec<_> = input
            .resources_1
            .into_iter()
            .zip(batch_1_sizes.into_iter())
            .filter(|v|v.1 != 0)
            .map(|(resource, size)| {
                ResourceSlices::new(resource.as_ref().id(), size as usize)
            })
            .collect();

        debug!("CrossUnfoldClc::do_run: input.id={:?} resource_ids_sizes_0={resource_ids_sizes_0:?}", input.id);
        debug!("CrossUnfoldClc::do_run: input.id={:?} resource_ids_sizes_1={resource_ids_sizes_1:?}", input.id);

        debug!("CrossUnfoldClc::do_run: input.id={:?} batch_0_t={batch_0_t:?}  batch_1_t={batch_1_t:?} mem_0={} mem_1={}", input.id, tensor_mem_size(&batch_0_t), tensor_mem_size(&batch_1_t));

        let cross_batch_ctx = (self.cross_batch_ctx)(ctx)?;
        let result = self.batch_clc_fn.call(batch_0_t, batch_1_t, &input/*TODO it's only used to check if we're on diagonal*/, &cross_batch_ctx, ctx).map(|v| {
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

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="cross_batch_clc/unfold_and_cat"))]
pub fn unfold_and_cat(input: &[&ThoTensor<ChannelsData, SynKey>], unfold_params: &UnfoldParams) -> Result<(Tensor, Vec<i64>), ThoError> {
    debug!("cross_batch_clc::unfold_and_cat: input={input:?} unfold_params={unfold_params:?}");

    assert!(!input.is_empty(), "cross_batch_clc/unfold_and_cat: input is empty");


    let (tensors, sizes):(Vec<Tensor>, Vec<i64>) = input.into_iter().fold((vec![], vec![]), |(mut unfolded_tensors, mut sizes), t|{
        if t.size().dim_1() >= unfold_params.size  {
            match  t.t_ref().f_unfold(
                unfold_params.dimension,
                unfold_params.size,
                unfold_params.step,
            ) {
                Ok(unfolded) => {

                    debug!("cross_batch_clc/unfold_and_cat: 0.  t.id={:?} t.size={:?} self.unfold_params={:?} unfolded.size={:?}", t.id(), t.size(), unfold_params, unfolded.size());

                    sizes.push(unfolded.size()[1]); //number of blocks

                    unfolded_tensors.push(unfolded);
                }
                Err(e) => {
                    warn!("cross_batch_clc/unfold_and_cat: 0.  t.id={:?} t.size={:?} self.unfold_params={:?} got e={e:?}", t.id(), t.size(), unfold_params);
                    sizes.push(0);
                }
            }

        } else {
            warn!("cross_batch_clc/unfold_and_cat: 0.  t.id={:?} t.size={:?} self.unfold_params={:?}  skipped", t.id(), t.size(), unfold_params);
            sizes.push(0);
        }
        (unfolded_tensors, sizes)
    });

    if tensors.is_empty() {
        return Err(ThoError::new("batch is empty"));
    }


    let batch_0_t = Tensor::f_concat(tensors.as_slice(), 1)?;
    Ok((batch_0_t, sizes))
}


#[cfg(test)]
mod tests {
    use crate::tasks::cross_batch_clc::{CrossBatchFn, CrossInput, CrossUnfoldClc, UnfoldParams, unfold_and_cat};
    use crate::tasks::dists_tracker::{DistsMask, DistsMaskType};
    use crate::tasks::find_motifs::{MasksCtx, MeanDistsCtx};
    use log::debug;
    use parking_lot::RwLock;
    use std::marker::PhantomData;
    use std::sync::Arc;
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::{Bool, Uint8};
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::{Kind, Tensor};
    use thoosa_common::dims::ChannelsData;
    use thoosa_common::runner::common::MockRunCtx;
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::utils::lin_t;
    use thoosa_shared::ThoError;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::{Stage, SyncStage};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_cross_batch_clc() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        GlobalCtx::init_devices(2);

        let cross_batch_fn: Box<dyn CrossBatchFn<DistsMask, MasksCtx, RunCtxImpl>> = Box::new(
            |t_0: Tensor, t_1: Tensor, input: &CrossInput<Arc<Resource>>, ctx: &MasksCtx, _: &RunCtxImpl| {
                debug!("t_0=\n{t_0}");
                debug!("t_1=\n{t_1}");
                debug!("input={input:?}");

                let tensor_00 = Tensor::from_slice(&[
                    0.00, 0.01,  0.01,0.02,     1.00, 1.01,  1.01, 1.02,  1.02,1.03,     /* 2.00, 2.01,*/
                    0.10, 0.11,  0.11,0.12,     1.10, 1.11,  1.11, 1.12,  1.12,1.13,     /* 2.10, 2.11,*/
                    0.20, 0.21,  0.21,0.22,     1.20, 1.21,  1.21, 1.22,  1.22,1.23,     /* 2.20, 2.21,*/
                ]).view((3,5,2));

                assert!(t_0.allclose(&tensor_00, 0.1, 0.1, false));

                let tensor_01 = Tensor::from_slice(&[
                    2.00, 2.01,
                    2.10, 2.11,
                    2.20, 2.21,
                ]).view((3,1,2));
                assert!(t_1.allclose(&tensor_01, 0.1, 0.1, false));

                let channels = t_0.size()[0];
                let slices_0 = t_0.size()[1];
                let slices_1 = t_1.size()[1];
                let total = (channels * slices_0 * slices_1) as u32;
                let dists = lin_t(total).view((channels, slices_0, slices_1));
                let mask = dists.lt(3.0);

                let mask_type = if mask.device().eq(&Cpu) {
                    DistsMaskType::AsIs
                } else {
                    DistsMaskType::MaskPacked(mask.size(), Some(dists.size()))
                };
                Ok(DistsMask::new(mask, Some(dists), mask_type)?)
            },
        );
        let cross_batch_clc = Arc::new(CrossUnfoldClc::new(
            cross_batch_fn,
            |_| Ok(MasksCtx::AccumulatedMeanDist {k:1.0, ctx: MeanDistsCtx::new(3, Cuda(0))?}),
            UnfoldParams::new(1, 2, 1),
        ));

        let tensor_00 = ThoTensor::new(SynKey::new([mono_key!(0u32, "tensor_00")]), Tensor::from_slice(&[
            0.00, 0.01, 0.02,
            0.10, 0.11, 0.12,
            0.20, 0.21, 0.22,
        ]).view((3,3)))?;
        let tensor_01 = ThoTensor::new(SynKey::new([mono_key!(1u32, "tensor_01")]), Tensor::from_slice(&[
            1.00, 1.01, 1.02, 1.03,
            1.10, 1.11, 1.12, 1.13,
            1.20, 1.21, 1.22, 1.23,
        ]).view((3,4)))?;

        let tensor_1 = ThoTensor::new(SynKey::new([mono_key!(2u32, "tensor_1")]), Tensor::from_slice(&[
            2.00, 2.01,
            2.10, 2.11,
            2.20, 2.21,
        ]).view((3,2)))?;

        let tensors = [&tensor_00, &tensor_01, &tensor_1];

        let resource_00 = Arc::new(Resource::new(0, 0, "file_00", 10));
        let resource_01 = Arc::new(Resource::new(1, 0, "file_01", 10));
        let chunk_0_resources = vec![Arc::clone(&resource_00), Arc::clone(&resource_01)];

        let resource_1 = Arc::new(Resource::new(2, 0, "file_1", 10));
        let chunk_1_resources = vec![Arc::clone(&resource_1)];

        let input = CrossInput::new(
            &tensors,
            chunk_0_resources.as_slice(),
            chunk_1_resources.as_slice(),
            "test".into()
        );
        let ctx = RunCtxImpl::new(1, "test");
        let out = cross_batch_clc.do_run(
            &input,
            &ctx,
        )?;

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
