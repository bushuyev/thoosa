use std::fmt::Debug;
use crate::tasks::cross_batch_clc::BatchInput;
#[cfg(test)]
use crate::tasks::cross_batch_clc::CrossBatchClc;
use crate::tasks::cross_dims::ChannelsSlicesData2;
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::TensorLoader;
use async_trait::async_trait;
use caching_runner::common::cachable::{Cached, DataCacheWrapper};
use caching_runner::common::runnable::Stage;
use caching_runner::common::{Data, OutId, RunCtx};
use log::debug;
use parking_lot::RwLock;
use std::sync::Arc;
use tch::Tensor;
use tch::kind::FLOAT_CPU;
use thoosa_common::dims::{AnyMetrics, AnySub, ChannelsData, ChannelsSlicesData, Dims};
use thoosa_common::tho_tensor::CopyToDevice;
use thoosa_shared::ThoosaError;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::resource::Resource;
use thoosa_shared::traits::{Named, SimpleId, WithId};
use torch_runner::data::ThoTensor2;
use torch_runner::metrics::TwoDimsMetrics;

pub trait CrossLoaderClc: Send + Sync + std::fmt::Debug {

    type Out;

    fn run_batch<'s>(
        self: Arc<Self>,
        input: &BatchInput<'s, &'s Arc<Resource>>,
        ctx: Arc<parking_lot::RwLock<RunCtx2>>,
    ) -> Result<Self::Out, ThoosaError>
    where
        Self: 's;
}

impl<T> CrossLoaderClc for T
where
    T: for<'s> Stage<
            In<'s> = BatchInput<'s, &'s Arc<Resource>>,
            Ctx = RunCtx2,
        > + 'static,
{
    type Out = T::Out;

    fn run_batch<'s>(
        self: Arc<Self>,
        input: &BatchInput<'s, &'s Arc<Resource>>,
        ctx: Arc<parking_lot::RwLock<RunCtx2>>,
    ) -> Result<Self::Out, ThoosaError>
    where
        Self: 's,
    {
        <Self as Stage>::do_run(self, input, ctx)
    }
}

// #[derive(Debug, cached_derive::Cached)]
#[derive(Debug)]
// #[cache_name = "LoadedTensorId"]
pub struct CrossLoader<CLC, RSV> {
    loader: Arc<TensorLoader>,
    batch_clc: Arc<CLC>,
    receiver: Arc<RSV>,
}

impl<CLC, RSV> CrossLoader<CLC, RSV>
where
    CLC: CrossLoaderClc + 'static,
{
    pub fn new(loader: Arc<TensorLoader>, cross_unfold: Arc<CLC>, receiver:Arc<RSV>) -> Self {
        Self {
            loader,
            batch_clc: cross_unfold,
            receiver
        }
    }
}

impl<CLC, RSV> OutId for CrossLoader<CLC, RSV>
where
    CLC: CrossLoaderClc + 'static,
    RSV: for<'a> Stage<In<'a>=CLC::Out, Ctx=RunCtx2> + Debug + Send + Sync + 'static
{
    fn out_id(&self, input_id: &<Self::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        SimpleId::from(format!("cross_{}", input_id))
    }
}

#[derive(Debug)]
pub struct ResourcesVec {
    id: SimpleId,
    resources: Vec<Arc<Resource>>,
}

impl ResourcesVec {
    pub fn new(id: SimpleId, resources: Vec<Resource>) -> Self {
        Self {
            id,
            resources: resources.into_iter().map(|r| Arc::new(r)).collect(),
        }
    }
}

impl WithId for ResourcesVec {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Input for ResourcesVec {}

#[async_trait]
impl<CLC, RSV> Stage for CrossLoader<CLC, RSV>
where
    CLC: CrossLoaderClc + 'static,
    RSV: for<'a> Stage<In<'a>=CLC::Out,Ctx=RunCtx2> + Debug + Send + Sync + 'static,
{
    type In<'a> = ResourcesVec where Self: 'a;
    type Out = ThoTensor2<ChannelsSlicesData2, SimpleId>;
    type Ctx = RunCtx2;

    async fn run(
        self: Arc<Self>,
        input: Arc<Self::In<'static>>,
        ctx: Arc<parking_lot::RwLock<Self::Ctx>>,
    ) -> Result<Self::Out, ThoosaError> {
        let window = 2;
        let stride = 1;
        let device_idx = ctx.read().device_idx();
        let max_batch = 20;

        let mut t_0_batch: Vec<_> = vec![];
        let mut t_0_batch_resources: Vec<_> = vec![];
        let mut t_0_batch_len = 0;

        let mut skip_0_i = 0;


        for (i_0, resource_0) in input.resources.iter().enumerate() {
            let t_0 = Arc::clone(&self.loader)
                .run_cached(Arc::clone(resource_0), Arc::clone(&ctx))
                .await?;

            t_0_batch_len += t_0.size().seq_len();
            t_0_batch.push(t_0);
            t_0_batch_resources.push(resource_0);

            if t_0_batch_len < max_batch && i_0 < input.resources.len() - 1 {
                continue;
            }

            let mut t_1_batch: Vec<_> = vec![];
            let mut t_1_batch_resources: Vec<_> = vec![];
            let mut t_1_batch_len = 0;

            for (i_1, resource_1) in input.resources
                .iter()
                .enumerate()
                .skip(skip_0_i) {//skipping lower triangle

                let t_1 = Arc::clone(&self.loader)
                    .run_cached(Arc::clone(resource_1), Arc::clone(&ctx))
                    .await?;

                t_1_batch_len += t_1.size().seq_len();

                t_1_batch.push(t_1);
                t_1_batch_resources.push(resource_1);

                if t_1_batch_len < max_batch && i_1 < input.resources.len() - 1 {
                    continue;
                }

                let _ = DataCacheWrapper::with_data_array(
                    [t_0_batch.clone(), t_1_batch].concat().as_slice(),
                    device_idx,
                    None,
                    |data| {

                        let input = BatchInput::new(
                            data,
                            t_0_batch_resources.as_slice(),
                            t_1_batch_resources.as_slice(),
                        );

                        debug!("input={input:?}");

                        let out = Arc::clone(&self.batch_clc).run_batch(&input, Arc::clone(&ctx))?;

                        Arc::clone(&self.receiver).do_run(&out, Arc::clone(&ctx))?;

                        Ok(())
                    },
                )
                .await?;

                t_1_batch = vec![];
                t_1_batch_resources = vec![];
                t_1_batch_len = 0;
            }

            skip_0_i = i_0;
            t_0_batch.clear();
            t_0_batch_resources.clear();
            t_0_batch_len = 0;
        }

        Ok(ThoTensor2::new(
            "id".into(),
            Tensor::zeros(&[10, 0], FLOAT_CPU),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::tasks::cross_batch_clc::CrossBatchClc;
    use crate::tasks::cross_loader::{CrossLoader, ResourcesVec};
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::TensorLoader;
    use caching_runner::common::cachable::Cached;
    use caching_runner::common::runnable::Stage;
    use caching_runner::global_ctx::GlobalCtx;
    use log::debug;
    use std::sync::Arc;
    use tch::kind::FLOAT_CPU;
    use tch::{Device, Tensor};
    use thoosa_common::applicables::ThoDevice;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_motifs::distance::clc_dist_d2;
    use thoosa_shared::ThoosaError;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::Resource;
    use torch_runner::data::ThoTensor2;
    use crate::tasks::dists_tracker::DistsTracker;

    #[ctor::ctor]
    pub fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test]
    async fn test_cross_clc() -> Result<(), ThoosaError> {
        GlobalCtx::init_devices(2);
        let resources: Vec<Resource> = vec!["111", "222", "333", "444", "555", "666"]
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i as u32, 1u32, v, 10).into())
            .collect();
        let resources_to_move = resources.clone();

        let ctx = Arc::new(parking_lot::RwLock::new(RunCtx2::new(0, 0, "aa")));

        let cl = Arc::new(CrossLoader::new(
            Arc::new(TensorLoader::new(
                move |key, _| {
                    let i = resources_to_move
                        .iter()
                        .position(|v| v.eq(key))
                        .expect(format!("no {key:?}").as_str())
                        + 1;
                    let v = (0..10)
                        .map(|v| i as f32 + v as f32 / 10.)
                        .collect::<Vec<f32>>();
                    debug!("executor: key={key:?} i={i} v={v:?}");

                    ThoTensor2::new(
                        Input::id(key).clone(),
                        Tensor::from_slice(v.as_slice()).view((1, 10)),
                    )
                },
                ThoDevice::from(Device::Cpu),
            )),
            Arc::new(CrossBatchClc::new()),
            Arc::new(DistsTracker::new())
        ));

        let r = cl
            .run(Arc::new(ResourcesVec::new("".into(), resources)), ctx)
            .await?;

        Ok(())
    }

    #[test]
    fn test_batch_unfold() -> Result<(), ThoosaError> {
        let t_0_inp = dummy_t(0, 3, 5);

        let t_1_inp = dummy_t(1, 3, 4);

        let t_2_inp = dummy_t(2, 3, 6);

        println!("t_0=\n{t_0_inp}");

        let tensors = vec![t_0_inp, t_1_inp, t_2_inp];
        let window = 2;
        let stride = 1;

        let max_batch = 20; //5+4+6=15 < 20

        let mut t_0_batch: Vec<&Tensor> = vec![];
        let mut t_0_batch_len = 0;

        let mut skip_0_i = 0;

        for (i_0, t_0) in tensors.iter().enumerate() {
            t_0_batch.push(t_0);
            t_0_batch_len += t_0.size()[1];

            if t_0_batch_len < max_batch && i_0 < tensors.len() - 1 {
                continue;
            }

            let batch_0_t = Tensor::concat(
                t_0_batch
                    .iter()
                    .map(|t| t.unfold(1, 2, 1))
                    .collect::<Vec<_>>()
                    .as_slice(),
                1,
            );

            let mut t_1_batch: Vec<&Tensor> = vec![];
            let mut t_1_batch_len = 0;

            for (i_1, t_1) in tensors.iter().enumerate().skip(skip_0_i) {
                t_1_batch.push(t_1);

                t_1_batch_len += t_1.size()[1];

                if t_1_batch_len < max_batch && i_1 < tensors.len() - 1 {
                    continue;
                }

                let batch_1_t = Tensor::concat(
                    t_1_batch
                        .iter()
                        .map(|t| t.unfold(1, 2, 1))
                        .collect::<Vec<_>>()
                        .as_slice(),
                    1,
                );

                println!(
                    "t_1_batch.len={} batch_0_t={batch_0_t:?}  batch_1_t={batch_1_t:?}",
                    t_1_batch.len()
                );

                let dists = clc_dist_d2(batch_0_t.shallow_clone(), batch_1_t, 1)?;

                println!("i_0={i_0} i_1={i_1} dists=\n{dists}");
                t_1_batch.clear();
                t_1_batch_len = 0;
            }
            skip_0_i = i_0;
            t_0_batch.clear();
            t_0_batch_len = 0;
        }

        Ok(())
    }

    fn dummy_t(n: usize, channels: usize, seq_len: usize) -> Tensor {
        let mut data = Vec::with_capacity(channels * seq_len);
        for ch in 0..channels {
            for pos in 0..seq_len {
                let val = n as f32 + (ch as f32) / 10.0 + (pos as f32) / 100.0;
                data.push(val);
            }
        }
        Tensor::from_slice(&data).view([channels as i64, seq_len as i64])
    }
}
