use crate::tasks::cross_dims::{ChannelsData2, ChannelsSlicesData2};
use crate::tasks::dists_tracker::DistWithMask;
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::slices_iter::SlicesIter;
use async_trait::async_trait;
use caching_runner::common::cachable::DataCacheWrapper;
use caching_runner::common::runnable::Stage;
use log::debug;
use parking_lot::RwLock;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use tch::Tensor;
use tch::kind::FLOAT_CPU;
use thoosa_motifs::distance::{clc_dist_d2, clc_dist_d2_and_mask};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::resource::Resource;
use thoosa_shared::traits::{SimpleId, WithId};
use thoosa_shared::{TensorIdSliceIdx, ThoosaError};
use torch_runner::data::ThoTensor2;

#[derive(Debug)]
pub struct BatchInput<'a, R: AsRef<Resource>> {
    tensors: &'a [&'a ThoTensor2<ChannelsData2, SimpleId>],
    b_0: &'a [R],
    b_1: &'a [R],
}

impl<'a, R: AsRef<Resource>> BatchInput<'a, R> {
    pub fn new(
        tensors: &'a [&'a ThoTensor2<ChannelsData2, SimpleId>],
        b_0: &'a [R],
        b_1: &'a [R],
    ) -> Self {
        Self { tensors, b_0, b_1 }
    }

    pub fn tensors(&self) -> &'a [&'a ThoTensor2<ChannelsData2, SimpleId>] {
        self.tensors
    }

    pub fn batch_0(&self) -> &'a [R] {
        self.b_0
    }

    pub fn batch_1(&self) -> &'a [R] {
        self.b_1
    }
}

impl<'a, R: AsRef<Resource> + Debug + Send + Sync> Input for BatchInput<'a, R> {}

pub struct CrossBatchClc<F, R>
where
    F: Fn(Tensor, Tensor) -> Result<R, ThoosaError>,
{
    batch_clc: F,
}

impl<F, R> Debug for CrossBatchClc<F, R>
where
    F: Fn(Tensor, Tensor) -> Result<R, ThoosaError>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossBatchClc").finish()
    }
}

impl<F, R> CrossBatchClc<F, R>
where
    F: Fn(Tensor, Tensor) -> Result<R, ThoosaError>,
{
    pub fn new(batch_clc: F) -> Self {
        Self { batch_clc }
    }
}

#[derive(Debug)]
pub struct CrossBatchClcOutput<R: Debug> {
    id: SimpleId,
    clc_result: R,
    iter: SlicesIter,
}

impl<R: Debug> CrossBatchClcOutput<R> {
    pub fn new(clc_result: R, iter: SlicesIter) -> Self {
        Self {
            id: "".into(),
            clc_result,
            iter,
        }
    }

    pub fn id(&self) -> &SimpleId {
        &self.id
    }

    pub fn clc_result(&self) -> &R {
        &self.clc_result
    }

    pub fn iter(&self) -> SlicesIter {
        self.iter.clone()
    }
}

impl<R: Debug> WithId for CrossBatchClcOutput<R> {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}
impl<R: Debug> Input for CrossBatchClcOutput<R> {}

impl<R: Debug> Output for CrossBatchClcOutput<R> {}

#[async_trait]
impl<F, R> Stage for CrossBatchClc<F, R>
where
    F: Fn(Tensor, Tensor) -> Result<R, ThoosaError> + 'static,
    R: Debug + Output + Input + Send + 'static,
{
    type In<'a> = BatchInput<'a, &'a Arc<Resource>>; //where Self:'a, R:'a, F: 'a;
    type Out = CrossBatchClcOutput<R>;
    type Ctx = RunCtx2;

    fn do_run<'s>(
        self: Arc<Self>,
        input: &Self::In<'s>,
        _ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Self::Out, ThoosaError>
    where
        Self: Sized,
    {
        let mut batch_sizes = vec![];

        let batch_0_t = Tensor::concat(
            input.tensors[0..input.b_0.len()]
                .iter()
                .map(|t| {
                    let unfolded = t.t().unfold(1, 2, 1);
                    batch_sizes.push(unfolded.size()[1]); //number of blocks
                    unfolded
                })
                .collect::<Vec<_>>()
                .as_slice(),
            1,
        );

        // let mut batch_1_sizes = vec![];

        let batch_1_t = Tensor::concat(
            input.tensors[input.b_0.len()..]
                .iter()
                .map(|t| {
                    let unfolded = t.t().unfold(1, 2, 1);
                    batch_sizes.push(unfolded.size()[1]); //number of blocks
                    unfolded
                })
                .collect::<Vec<_>>()
                .as_slice(),
            1,
        );

        assert_eq!(batch_sizes.len(), input.b_0.len() + input.b_1.len());

        let resource_ids_sizes: Vec<_> = input
            .b_0
            .into_iter()
            .chain(input.b_1.into_iter())
            .zip(batch_sizes.into_iter())
            .map(|v| (v.0.id(), v.1 as usize))
            .collect();

        debug!("resource_ids_sizes={resource_ids_sizes:?}");

        debug!("CrossBatchClc::do_run:  batch_0_t={batch_0_t:?}  batch_1_t={batch_1_t:?}");

        let iter = SlicesIter::new(2, resource_ids_sizes);

        (self.batch_clc)(batch_0_t.shallow_clone(), batch_1_t)
            .map(|v| CrossBatchClcOutput::new(v, iter))
    }
}

#[cfg(test)]
mod tests {
    use crate::tasks::cross_batch_clc::{BatchInput, CrossBatchClc};
    use crate::tasks::dists_tracker::DistWithMask;
    use crate::tasks::run_ctx_2::RunCtx2;
    use caching_runner::common::MockRunCtx;
    use caching_runner::common::runnable::Stage;
    use log::debug;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use tch::Tensor;
    use tch::kind::FLOAT_CPU;
    use thoosa_common::utils::lin_t;
    use thoosa_shared::ThoosaError;
    use thoosa_shared::resource::Resource;
    use torch_runner::data::ThoTensor2;

    #[test]
    fn test_0() -> Result<(), ThoosaError> {
        let cross_batch_clc = Arc::new(CrossBatchClc::new(|t_0, t_1| {
            debug!("t_0=\n{t_0}");
            debug!("t_1=\n{t_1}");

            Ok(DistWithMask::new(
                Tensor::rand(&[0], FLOAT_CPU),
                Some(Tensor::rand(&[0], FLOAT_CPU)),
            ))
        }));

        let tensor_00 = ThoTensor2::new("1".into(), Tensor::rand(&[3, 10], FLOAT_CPU));
        let tensor_01 = ThoTensor2::new("1".into(), Tensor::rand(&[3, 13], FLOAT_CPU));

        let tensor_1 = ThoTensor2::new("2".into(), Tensor::rand(&[3, 20], FLOAT_CPU));

        let tensors = [&tensor_00, &tensor_01, &tensor_1];

        let resource_00 = Arc::new(Resource::new(0, 0, "file_00".to_string(), 10));
        let resource_01 = Arc::new(Resource::new(1, 0, "file_01".to_string(), 10));
        let chunk_0_resources = vec![&resource_00, &resource_01];

        let resource_1 = Arc::new(Resource::new(2, 0, "file_1".to_string(), 10));
        let chunk_1_resources = vec![&resource_1];

        let input = BatchInput::new(
            &tensors,
            &chunk_0_resources.as_slice(),
            &chunk_1_resources.as_slice(),
        );
        cross_batch_clc.do_run(&input, Arc::new(RwLock::new(RunCtx2::new(0, 0, "test"))))?;

        Ok(())
    }
}
