use crate::tasks::cross_dims::{ChannelsData2};
use crate::tasks::dists_tracker::DistsMask;
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::slices_iter::SlicesIter;
use async_trait::async_trait;
use caching_runner::common::cachable::DataCacheWrapper;
use caching_runner::common::runnable::{Stage, SyncStage};
use log::{debug, trace};
use parking_lot::RwLock;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use tch::Tensor;
use tch::kind::FLOAT_CPU;
use thoosa_common::utils::tensor_mem_size;
use thoosa_motifs::distance::{clc_dist_d2, clc_dist_d2_and_mask};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::resource::Resource;
use thoosa_shared::traits::{SimpleId, WithId};
use thoosa_shared::{TensorIdSliceIdx, ThoosaError};
use torch_runner::data::ThoTensor2;
use crate::tasks::cross_loader::CrossClc;

#[derive(Debug)]
pub struct CrossInput<'a, R: AsRef<Resource>> {
    tensors: &'a [&'a ThoTensor2<ChannelsData2, SimpleId>],
    resources_0: &'a [R],
    resources_1: &'a [R],
}

impl<'a, R: AsRef<Resource>> CrossInput<'a, R> {
    pub fn new(
        tensors: &'a [&'a ThoTensor2<ChannelsData2, SimpleId>],
        b_0: &'a [R],
        b_1: &'a [R],
    ) -> Self {
        Self { tensors, resources_0: b_0, resources_1: b_1 }
    }

    pub fn tensors(&self) -> &'a [&'a ThoTensor2<ChannelsData2, SimpleId>] {
        self.tensors
    }

    pub fn batch_0(&self) -> &'a [R] {
        self.resources_0
    }

    pub fn batch_1(&self) -> &'a [R] {
        self.resources_1
    }
}

impl<'a, R: AsRef<Resource> + Debug + Send + Sync> Input for CrossInput<'a, R> {}

#[derive(Debug)]
pub(crate) struct UnfoldParams {
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
}

pub struct CrossUnfoldClc<F, R, CUC>
where
    F: for<'a> Fn(Tensor, Tensor, &CrossInput<'a, &'a Arc<Resource>>, &CUC) -> Result<R, ThoosaError>,
{
    batch_clc_fn: F,
    cross_batch_ctx: CUC,
    unfold_params: UnfoldParams,
}

impl<F, R, CUC> Debug for CrossUnfoldClc<F, R, CUC>
where
    F: for<'a> Fn(Tensor, Tensor, &CrossInput<'a, &'a Arc<Resource>>, &CUC) -> Result<R, ThoosaError>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossBatchClc").finish()
    }
}


impl<F, R, CUC> CrossUnfoldClc<F, R, CUC>
where
    F: for<'a> Fn(Tensor, Tensor, &CrossInput<'a, &'a Arc<Resource>>, &CUC) -> Result<R, ThoosaError>,
{
    pub fn new(batch_clc: F, cross_batch_ctx: CUC, unfold_params: UnfoldParams) -> Self {
        Self {
            batch_clc_fn: batch_clc,
            cross_batch_ctx,
            unfold_params,
        }
    }
}

#[derive(Debug)]
pub struct CrossUnfoldClcOutput<R: Debug> {
    id: SimpleId,
    clc_result: R,
    iter_0: SlicesIter,
    iter_1: SlicesIter,
}

impl<R: Debug> CrossUnfoldClcOutput<R> {
    pub fn new(clc_result: R, iter_0: SlicesIter, iter_1: SlicesIter) -> Self {
        Self {
            id: "".into(),
            clc_result,
            iter_0,
            iter_1,
        }
    }

    pub fn id(&self) -> &SimpleId {
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
}

impl<R: Debug> WithId for CrossUnfoldClcOutput<R> {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}
impl<R: Debug> Input for CrossUnfoldClcOutput<R> {}

impl<R: Debug> Output for CrossUnfoldClcOutput<R> {}


impl<F, R, CUC> Stage for CrossUnfoldClc<F, R, CUC>
where
    F: for<'a> Fn(Tensor, Tensor, &CrossInput<'a, &'a Arc<Resource>>, &CUC) -> Result<R, ThoosaError>,
    R:Debug
{
    type In<'a> = CrossInput<'a, &'a Arc<Resource>> where F: 'a, R:'a, CUC:'a;

    type Out = CrossUnfoldClcOutput<R>;
    type Ctx = RunCtx2;
}


impl<F, R, CUC> SyncStage for CrossUnfoldClc<F, R, CUC>//CUC Cross Unfold Context
where
    for<'a> F: Fn(Tensor, Tensor, &CrossInput<'a, &'a Arc<Resource>>, &CUC) -> Result<R, ThoosaError> + 'a,
    R: Debug + Output + Input + Send + 'static,
    for<'a> CUC: 'a,
{


    fn do_run<'s>(self: Arc<Self>, input: &Self::In<'s>, _ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    where
        Self: Sized
    {
        let mut batch_0_sizes = vec![];

        let batch_0_tensors = input.tensors[0..input.resources_0.len()]
            .iter()
            .filter(|t| t.size()[1] >= self.unfold_params.size) //TODO use metrics's size
            .map(|t| {

                debug!("CrossUnfoldClc::do_run: 0. t.size={:?} self.unfold_params={:?}", t.size(), self.unfold_params);

                let unfolded = t.t_ref().unfold(
                    self.unfold_params.dimension,
                    self.unfold_params.size,
                    self.unfold_params.step,
                );
                batch_0_sizes.push(unfolded.size()[1]); //number of blocks
                unfolded
            })
            .collect::<Vec<_>>();

        if batch_0_tensors.is_empty() {
            return Err("batch 0 is empty".into());
        }


        let batch_0_t = Tensor::concat(batch_0_tensors.as_slice(), 1);

        let mut batch_1_sizes = vec![];

        let batch_1_tensors = input.tensors[input.resources_0.len()..]
            .iter()
            .filter(|t| t.size()[1] >= self.unfold_params.size) //TODO use metrics's size
            .map(|t| {

                debug!("CrossUnfoldClc::do_run: 1. t.size={:?} self.unfold_params={:?}", t.size(), self.unfold_params);

                let unfolded = t.t_ref().unfold(
                    self.unfold_params.dimension,
                    self.unfold_params.size,
                    self.unfold_params.step,
                );
                batch_1_sizes.push(unfolded.size()[1]); //number of blocks
                unfolded
            })
            .collect::<Vec<_>>();

        if batch_1_tensors.is_empty() {
            return Err("batch 0 is empty".into());
        }


        let batch_1_t = Tensor::concat(batch_1_tensors.as_slice(), 1);



        assert_eq!(batch_0_sizes.len(), input.resources_0.len());
        assert_eq!(batch_1_sizes.len(), input.resources_1.len());

        let resource_ids_sizes_0: Vec<_> = input
            .resources_0
            .into_iter()
            .zip(batch_0_sizes.into_iter())
            .map(|v| (v.0.as_ref().id(), v.1 as usize))
            .collect();

        let resource_ids_sizes_1: Vec<_> = input
            .resources_1
            .into_iter()
            .zip(batch_1_sizes.into_iter())
            .map(|v| (v.0.as_ref().id(), v.1 as usize))
            .collect();

        debug!("resource_ids_sizes={resource_ids_sizes_0:?}");

        debug!(
            "CrossBatchClc::do_run:  batch_0_t={batch_0_t:?}  batch_1_t={batch_1_t:?} mem_0={} mem_1={}",
            tensor_mem_size(&batch_0_t),
            tensor_mem_size(&batch_1_t)
        );

        (self.batch_clc_fn)(batch_0_t, batch_1_t, &input, &self.cross_batch_ctx).map(|v| {
            CrossUnfoldClcOutput::new(
                v,
                SlicesIter::new(self.unfold_params.step as usize, resource_ids_sizes_0),
                SlicesIter::new(self.unfold_params.step as usize, resource_ids_sizes_1),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::tasks::cross_batch_clc::{CrossInput, CrossUnfoldClc, UnfoldParams};
    use crate::tasks::dists_tracker::DistsMask;
    use crate::tasks::find_motifs::MeanDistsCtx;
    use crate::tasks::run_ctx_2::RunCtx2;
    use caching_runner::common::MockRunCtx;
    use caching_runner::common::runnable::{Stage, SyncStage};
    use log::debug;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use tch::Device::Cuda;
    use tch::Tensor;
    use tch::kind::FLOAT_CPU;
    use thoosa_common::utils::lin_t;
    use thoosa_shared::ThoosaError;
    use thoosa_shared::resource::Resource;
    use torch_runner::data::ThoTensor2;

    #[test]
    fn test_cross_batch_clc() -> Result<(), ThoosaError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        let cross_batch_clc = Arc::new(CrossUnfoldClc::new(
            |t_0, t_1, input, ctx| {
                debug!("t_0=\n{t_0}");
                debug!("t_1=\n{t_1}");

                Ok(DistsMask::new(
                    Tensor::rand(&[0], FLOAT_CPU),
                    Some(Tensor::rand(&[0], FLOAT_CPU)),
                ))
            },
            MeanDistsCtx::new(3, Cuda(0)),
            UnfoldParams::new(1, 1, 1),
        ));

        let tensor_00 = ThoTensor2::new("1".into(), Tensor::rand(&[3, 10], FLOAT_CPU));
        let tensor_01 = ThoTensor2::new("1".into(), Tensor::rand(&[3, 13], FLOAT_CPU));

        let tensor_1 = ThoTensor2::new("2".into(), Tensor::rand(&[3, 20], FLOAT_CPU));

        let tensors = [&tensor_00, &tensor_01, &tensor_1];

        let resource_00 = Arc::new(Resource::new(0, 0, "file_00".to_string(), 10));
        let resource_01 = Arc::new(Resource::new(1, 0, "file_01".to_string(), 10));
        let chunk_0_resources = vec![&resource_00, &resource_01];

        let resource_1 = Arc::new(Resource::new(2, 0, "file_1".to_string(), 10));
        let chunk_1_resources = vec![&resource_1];

        let input = CrossInput::new(
            &tensors,
            &chunk_0_resources.as_slice(),
            &chunk_1_resources.as_slice(),
        );
        let out = cross_batch_clc.do_run(&input, Arc::new(RwLock::new(RunCtx2::new(0, 0, "test"))))?;

        println!("{:?}", out);
        //TODO asserts

        Ok(())
    }
}
