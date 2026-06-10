use thoosa_common::dims::ChannelsData;
use crate::tasks::dists_tracker::DistsMask;
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use crate::tasks::slices_iter::{ResourceSlices, SlicesIter};
use async_trait::async_trait;
use thoosa_common::runner::common::cachable::{DevicesWrap, WithDataRef, WithSelectDataRef};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::common::runnable::{Stage, SyncStage};
use thoosa_common::runner::common::{OutId, MaxParallelCtx, RunCtx};
use log::{debug, trace, warn};
use parking_lot::RwLock;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;
use futures_util::future::BoxFuture;
use futures_util::FutureExt;
use tch::{TchError, Tensor};
use tch::kind::FLOAT_CPU;
use thoosa_common::utils::tensor_mem_size;
use thoosa_motifs::distance::clc_dist_d2;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::resource::Resource;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::traits::{Adjustable, Data, WithId};
use thoosa_shared::{mono_key, TensorIdSliceIdx, ThoError};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::comp_id::{MonoKey, SynKey};
use tracing::Instrument;
use thoosa_shared::tracked_await::track_await;

#[derive(Debug)]
pub struct CrossInput<'a, R: AsRef<Resource>> {
    id: SynKey,
    tensors: &'a [&'a ThoTensor<ChannelsData, SynKey>],
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
            tensors,
            resources_0: b_0,
            resources_1: b_1,
        }
    }

    pub fn tensors(&self) -> &'a [&'a ThoTensor<ChannelsData, SynKey>] {
        self.tensors
    }

    pub fn batch_0(&self) -> &'a [R] {
        self.resources_0
    }

    pub fn batch_1(&self) -> &'a [R] {
        self.resources_1
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
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        self.resources_1.len() as i64
    }

    fn device(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        self.tensors.iter().map(|tensor| tensor.mem_size()).sum()
    }
}

#[derive(Debug, Clone, Default)]
pub struct CrossInputBatch1Selector {
    resources_batch_range: Range<i64>,
}

impl From<Range<i64>> for CrossInputBatch1Selector {
    fn from(value: Range<i64>) -> Self {
        Self { resources_batch_range: value }
    }
}

impl Adjustable for CrossInputBatch1Selector {}

impl<'a, R> WithDataRef for CrossInput<'a, R>
where
    R: AsRef<Resource> + Debug + Send + Sync + 'static,
{
    type O<'o> = CrossInput<'o, R>;

    fn with_data_ref<'b, Rt: Send + 'b, F, P>(
        &'b self,
        _to_device_idx: usize,
        permits_ctx: P,
        _trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> impl std::future::Future<Output = Result<Rt, ThoError>> + Send + 'b
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<Rt, ThoError> + Send + 'b,
        P: RunCtx + 'b,
    {
        async move {
            debug!("CrossInput::with_data_ref: do_run");
            do_run(self, &permits_ctx)
        }
    }


}

impl<'a, R> WithSelectDataRef<CrossInputBatch1Selector> for CrossInput<'a, R>
where
    R: AsRef<Resource> + Debug + Send + Sync + 'static,
{
    fn with_selected_data_ref<'b, 'o, Rt: Send + 'b, F, P>(
        &'b self,
        _to_device_idx: usize,
        permits_ctx: P,
        selector: CrossInputBatch1Selector,
        _trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> impl std::future::Future<Output = Result<Rt, ThoError>> + Send + 'b
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<Rt, ThoError> + Send + 'b,
        P: RunCtx + 'b,
    {
        async move {
            let range = selector.resources_batch_range.start as usize..selector.resources_batch_range.end as usize;
            let batch_0_len = self.resources_0.len();
            let batch_1_offset = batch_0_len + range.start;
            let batch_1_end = batch_0_len + range.end;

            let mut selected_tensors = Vec::with_capacity(batch_0_len + (range.end - range.start));
            selected_tensors.extend_from_slice(&self.tensors[..batch_0_len]);
            selected_tensors.extend_from_slice(&self.tensors[batch_1_offset..batch_1_end]);

            let input = CrossInput::new(
                selected_tensors.as_slice(),
                self.resources_0,
                &self.resources_1[range.clone()],
                self.id.clone(), // + &mono_key!(0, "todo"),
            );

            do_run(&input, &permits_ctx)
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

#[derive(Debug)]
pub(crate) struct UnfoldParams { //TODO use in FindMotifsRunConfig
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

pub struct CrossUnfoldClc<F, R, CUC, C>
where
    F: CrossBatchFn<R, CUC, C>,
    C: RunCtx + 'static,
{
    batch_clc_fn: F,
    cross_batch_ctx: CUC,
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
    pub fn new(batch_clc_fn: F, cross_batch_ctx: CUC, unfold_params: UnfoldParams) -> Self {
        Self {
            batch_clc_fn,
            cross_batch_ctx,
            unfold_params,
            _pd: PhantomData,
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

#[derive(Debug)]
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
        self.clc_result = track_await(f(self.clc_result)).instrument(tracing::debug_span!("replace_clc_result/1", file = file!(), line = line!())).await?;
        Ok(self)
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
        let (batch_0_t, batch_0_sizes) =  unfold_and_cat(&input.tensors[0..input.resources_0.len()], &unfold_params)?;
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


        let (batch_1_t, batch_1_sizes) =  unfold_and_cat(&input.tensors[input.resources_0.len()..], &unfold_params)?;

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

        self.batch_clc_fn.call(batch_0_t, batch_1_t, &input/*TODO it's only used to check if we're on diagonal*/, &self.cross_batch_ctx, ctx).map(|v| {
            CrossUnfoldClcOutput::new(
                v,
                SlicesIter::new_with_offsets(unfold_params.step as usize, resource_ids_sizes_0),
                SlicesIter::new_with_offsets(unfold_params.step as usize, resource_ids_sizes_1),
                input.id.clone(),
            )
        })
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
    use std::marker::PhantomData;
    use crate::tasks::cross_batch_clc::{ unfold_and_cat, CrossBatchFn, CrossInput, CrossUnfoldClc, UnfoldParams};
    use crate::tasks::dists_tracker::{DistsMask, DistsMaskType};
    use crate::tasks::find_motifs::{MasksCtx, MeanDistsCtx};
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use thoosa_common::runner::common::MockRunCtx;
    use thoosa_common::runner::common::runnable::{Stage, SyncStage};
    use log::debug;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::{Bool, Uint8};
    use tch::{Kind, Tensor};
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use thoosa_common::utils::lin_t;
    use thoosa_shared::ThoError;
    use thoosa_shared::resource::Resource;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::dims::ChannelsData;

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
            MasksCtx::AccumulatedMeanDist {k:1.0, ctx: MeanDistsCtx::new(3, Cuda(0))?},
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
