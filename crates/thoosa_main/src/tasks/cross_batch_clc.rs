use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::dists_tracker::DistsMask;
use thoosa_shared::runner::run_ctx_2::RunCtxWithPermits;
use crate::tasks::slices_iter::{ResourceSlices, SlicesIter};
use async_trait::async_trait;
use thoosa_shared::runner::common::cachable::{DevicesWrap, WithDataRef, WithSelectDataRef};
use thoosa_shared::runner::cache_areas::CacheArea;
use thoosa_shared::runner::common::runnable::{Stage, SyncStage};
use thoosa_shared::runner::common::{OutId, PermitsCtx, RunCtx};
use log::{debug, trace};
use parking_lot::RwLock;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;
use futures_util::future::BoxFuture;
use futures_util::FutureExt;
use tch::Tensor;
use tch::kind::FLOAT_CPU;
use thoosa_common::utils::tensor_mem_size;
use thoosa_motifs::distance::{clc_dist_d2, clc_dist_d2_and_mask};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::resource::Resource;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::traits::{Adjustable, Data, WithId};
use thoosa_shared::{mono_key, TensorIdSliceIdx, ThoError};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::comp_id::{MonoKey, SynKey};

#[derive(Debug)]
pub struct CrossInput<'a, R: AsRef<Resource>> {
    id: MonoKey,
    tensors: &'a [&'a ThoTensor<ChannelsData2, SynKey>],
    resources_0: &'a [R],
    resources_1: &'a [R],
    cache_area: CacheArea,
}

impl<'a, R: AsRef<Resource>> CrossInput<'a, R> {

    pub fn new(
        tensors: &'a [&'a ThoTensor<ChannelsData2, SynKey>],
        b_0: &'a [R],
        b_1: &'a [R],
    ) -> Self {


        let id = {
            let batch_0_ids = b_0.iter().map(|r| r.as_ref().id()).collect::<Vec<_>>();
            let batch_1_ids = b_1.iter().map(|r| r.as_ref().id()).collect::<Vec<_>>();
            mono_key!(//TODO
                format!("cross_input_b0={batch_0_ids:?}_b1={batch_1_ids:?}"),
                "CrossInput"
            )
        };

        Self {
            id,
            tensors,
            resources_0: b_0,
            resources_1: b_1,
            cache_area: CacheArea::default(),
        }
    }

    pub fn tensors(&self) -> &'a [&'a ThoTensor<ChannelsData2, SynKey>] {
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
    type ID = MonoKey;

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

    async fn with_data_ref<Rt, F, P>(
        &self,
        _to_device_idx: usize,
        permits_ctx: P,
        _trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> Result<Rt, ThoError>
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<Rt, ThoError> + Send,
        P: RunCtx,
    {
        debug!("CrossInput::with_data_ref: do_run");
        do_run(self, &permits_ctx)
    }

    fn cache_area(&self) -> &CacheArea {
        &self.cache_area
    }
}

impl<'a, R> WithSelectDataRef<CrossInputBatch1Selector> for CrossInput<'a, R>
where
    R: AsRef<Resource> + Debug + Send + Sync + 'static,
{
    async fn with_selected_data_ref<'o, Rt, F, P>(
        &self,
        _to_device_idx: usize,
        permits_ctx: P,
        selector: CrossInputBatch1Selector,
        _trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> Result<Rt, ThoError>
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<Rt, ThoError>,
        P: RunCtx,
    {
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
        );

        do_run(&input, &permits_ctx)
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

pub struct CrossUnfoldClc<F, R, CUC, C>
where
    F: for<'a> Fn(Tensor, Tensor, &CrossInput<'a, Arc<Resource>>, &CUC, &C) -> Result<R, ThoError>,
    C: RunCtx + 'static,
{
    batch_clc_fn: F,
    cross_batch_ctx: CUC,
    pub unfold_params: UnfoldParams,
    _ctx: PhantomData<C>,
}

impl<F, R, CUC, C> Debug for CrossUnfoldClc<F, R, CUC, C>
where
    F: for<'a> Fn(Tensor, Tensor, &CrossInput<'a, Arc<Resource>>, &CUC, &C) -> Result<R, ThoError>,
    C: RunCtx + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossUnfoldClc").field("unfold_params", &self.unfold_params).finish()
    }
}


impl<F, R, CUC, C> CrossUnfoldClc<F, R, CUC, C>
where
    F: for<'a> Fn(Tensor, Tensor, &CrossInput<'a, Arc<Resource>>, &CUC, &C) -> Result<R, ThoError>,
    C: RunCtx + 'static,
{
    pub fn new(batch_clc_fn: F, cross_batch_ctx: CUC, unfold_params: UnfoldParams) -> Self {
        Self {
            batch_clc_fn,
            cross_batch_ctx,
            unfold_params,
            _ctx: PhantomData,
        }
    }
}

impl<F, R, CUC, C> Adjustable for CrossUnfoldClc<F, R, CUC, C>
where
    F: for<'a> Fn(Tensor, Tensor, &CrossInput<'a, Arc<Resource>>, &CUC, &C) -> Result<R, ThoError> + Send + Sync,
    CUC: Send + Sync,
    R: Debug,
    C: RunCtx + 'static,
{
}

impl<F, R, CUC, C> OutId for CrossUnfoldClc<F, R, CUC, C>
where
    F: for<'a> Fn(Tensor, Tensor, &CrossInput<'a, Arc<Resource>>, &CUC, &C) -> Result<R, ThoError>,
    R: Debug,
    C: RunCtx + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

#[derive(Debug)]
pub struct CrossUnfoldClcOutput<R: Debug> {
    id: MonoKey,
    clc_result: R,
    iter_0: SlicesIter,
    iter_1: SlicesIter,
}

impl<R: Debug> CrossUnfoldClcOutput<R> {
    pub fn new(clc_result: R, iter_0: SlicesIter, iter_1: SlicesIter) -> Self {
        Self {
            id: mono_key!("CrossUnfoldClcOutput", "stage"),
            clc_result,
            iter_0,
            iter_1,
        }
    }

    pub fn id(&self) -> &MonoKey {
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
        self.clc_result = f(self.clc_result).await?;
        Ok(self)
    }
}

impl<R: Debug> WithId for CrossUnfoldClcOutput<R> {
    type ID = MonoKey;

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
}
impl<R: Debug> Input for CrossUnfoldClcOutput<R> {}

impl<R: Debug> Output for CrossUnfoldClcOutput<R> {}


impl<F, R, CUC, C> Stage for CrossUnfoldClc<F, R, CUC, C>
where
    F: for<'a> Fn(Tensor, Tensor, &CrossInput<'a, Arc<Resource>>, &CUC, &C) -> Result<R, ThoError>,
    R: Debug,
    C: RunCtx + 'static,
{
    type In<'a> = CrossInput<'a, Arc<Resource>>;

    type Out = CrossUnfoldClcOutput<R>;
    type Ctx = C;
}


impl<F, R, CUC, C> SyncStage for CrossUnfoldClc<F, R, CUC, C>//CUC Cross Unfold Context
where
    for<'a> F: Fn(Tensor, Tensor, &CrossInput<'a, Arc<Resource>>, &CUC, &C) -> Result<R, ThoError> + 'a,
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

        debug!("CrossUnfoldClc::do_run: input={input:?} unfold_params={unfold_params:?}");
        debug!("CrossUnfoldClc::do_run: input.tensors.len={} input.resources_0.len={} input.resources_1.len={}", input.tensors.len(), input.resources_0.len(), input.resources_1.len());


        let (batch_0_t, batch_0_sizes) =  unfold_and_cat(&input.tensors[0..input.resources_0.len()], &unfold_params)?;

        debug!("CrossUnfoldClc::do_run: batch_0_sizes={batch_0_sizes:?}");


        //TODO if tensor is filtered because it's less than window
        // assert_eq!(batch_0_sizes.len(), input.resources_0.len());
        let resource_ids_sizes_0: Vec<_> = input
            .resources_0
            .into_iter()
            .zip(batch_0_sizes.into_iter())
            .map(|(resource, size)| {
                ResourceSlices::new(resource.as_ref().id(), size as usize)
            })
            .collect();


        let (batch_1_t, batch_1_sizes) =  unfold_and_cat(&input.tensors[input.resources_0.len()..], &unfold_params)?;

        // assert_eq!(batch_1_sizes.len(), input.resources_1.len());
        debug!("CrossUnfoldClc::do_run: batch_1_sizes={batch_1_sizes:?}");

        let resource_ids_sizes_1: Vec<_> = input
            .resources_1
            .into_iter()
            .zip(batch_1_sizes.into_iter())
            .map(|(resource, size)| {
                ResourceSlices::new(resource.as_ref().id(), size as usize)
            })
            .collect();

        debug!("CrossUnfoldClc::do_run: resource_ids_sizes_0={resource_ids_sizes_0:?}");
        debug!("CrossUnfoldClc::do_run: resource_ids_sizes_1={resource_ids_sizes_1:?}");

        debug!("CrossUnfoldClc::do_run:  batch_0_t={batch_0_t:?}  batch_1_t={batch_1_t:?} mem_0={} mem_1={}", tensor_mem_size(&batch_0_t), tensor_mem_size(&batch_1_t));

        (self.batch_clc_fn)(batch_0_t, batch_1_t, &input, &self.cross_batch_ctx, ctx).map(|v| {
            CrossUnfoldClcOutput::new(
                v,
                SlicesIter::new_with_offsets(unfold_params.step as usize, resource_ids_sizes_0),
                SlicesIter::new_with_offsets(unfold_params.step as usize, resource_ids_sizes_1),
            )
        })
    }
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="cross_batch_clc/unfold_and_cat"))]
pub fn unfold_and_cat(input: &[&ThoTensor<ChannelsData2, SynKey>], unfold_params: &UnfoldParams) -> Result<(Tensor, Vec<i64>), ThoError> {
    debug!("cross_batch_clc::unfold_and_cat: input={input:?} unfold_params={unfold_params:?}");

    assert!(!input.is_empty(), "cross_batch_clc/unfold_and_cat: input is empty");

    let mut batch_0_sizes = vec![];

    let batch_0_tensors = input
        .iter()
        //filter tensors smaller than motif window
        .filter(|t| t.size().dim_1() >= unfold_params.size) //TODO use metrics's size
        .map(|t| {

            let unfolded = t.t_ref().f_unfold(
                unfold_params.dimension,
                unfold_params.size,
                unfold_params.step,
            )?;
            debug!("cross_batch_clc/unfold_and_cat: 0.  t.id={:?} t.size={:?} self.unfold_params={:?} unfolded.size={:?}", t.id(), t.size(), unfold_params, unfolded.size());

            batch_0_sizes.push(unfolded.size()[1]); //number of blocks
            Ok(unfolded)
        })
        .collect::<Result<Vec<_>, ThoError>>()?;

    if batch_0_tensors.is_empty() {
        return Err(ThoError::new("batch 0 is empty"));
    }


    let batch_0_t = Tensor::f_concat(batch_0_tensors.as_slice(), 1)?;
    Ok((batch_0_t, batch_0_sizes))
}


#[cfg(test)]
mod tests {
    use crate::tasks::cross_batch_clc::{CrossInput, CrossUnfoldClc, UnfoldParams};
    use crate::tasks::dists_tracker::{DistsMask, DistsMaskType};
    use crate::tasks::find_motifs::{MasksCtx, MeanDistsCtx};
    use thoosa_shared::runner::run_ctx_2::RunCtxWithPermits;
    use thoosa_shared::runner::common::MockRunCtx;
    use thoosa_shared::runner::common::runnable::{Stage, SyncStage};
    use log::debug;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use polars::prelude::RevMapping::Global;
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::{Bool, Uint8};
    use tch::Tensor;
    use tch::kind::FLOAT_CPU;
    use thoosa_common::utils::lin_t;
    use thoosa_shared::ThoError;
    use thoosa_shared::resource::Resource;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_shared::runner::global_ctx::GlobalCtx;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_cross_batch_clc() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        GlobalCtx::init_devices(2);
        let cross_batch_clc = Arc::new(CrossUnfoldClc::new(
            |t_0, t_1, input, ctx, _| {
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
                Ok(DistsMask::new(mask, Some(dists), mask_type, None)?)
            },
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
        );
        let ctx = RunCtxWithPermits::new(1, "test");
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
