use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use log::debug;
use parking_lot::RwLock;
use tch::kind::FLOAT_CPU;
use tch::Tensor;
use caching_runner::common::{Data, OutId};
use caching_runner::common::cachable::Cached;
use caching_runner::common::runnable::Stage;
use thoosa_common::dims::{AnyMetrics, AnySub, ChannelsData, ChannelsSlicesData, Dims};
use thoosa_common::tho_tensor::CopyToDevice;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::resource::Resource;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Named, SimpleId, WithId};
use torch_runner::data::ThoTensor2;
use torch_runner::metrics::TwoDimsMetrics;
use crate::tasks::cross_unfold::{CrossUnfold, TensorsPair};
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::{TensorLoader};

// #[derive(Debug, cached_derive::Cached)]
#[derive(Debug)]
// #[cache_name = "LoadedTensorId"]
pub struct CrossLoader {
    loader: Arc<TensorLoader>,
    cross_unfold: Arc<CrossUnfold>
}


impl CrossLoader {
    pub fn new(
        loader:Arc<TensorLoader>,
        cross_unfold: Arc<CrossUnfold>
    ) -> Self {
        Self { loader, cross_unfold }
    }
}

impl OutId for CrossLoader {
    fn out_id(&self, input_id: &<Self::In as WithId>::ID) -> <Self::Out as WithId>::ID {
        SimpleId::from(format!("cross_{}", input_id))
    }
}
#[derive(Debug)]
pub struct ResourcesVec {
    id:SimpleId,
    resources:Vec<Arc<Resource>>
}

impl ResourcesVec {
    pub fn new(id: SimpleId, resources: Vec<Resource>) -> Self {
        Self { id, resources: resources.into_iter().map(|r|Arc::new(r)).collect() }
    }


}

impl WithId for ResourcesVec {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Input for ResourcesVec{}


#[derive(Clone, Debug)]
pub struct ChannelsSlicesData2;
impl Dims for ChannelsSlicesData2 {
    type Metrics = TwoDimsMetrics;
    type Sub = ThoTensor2<ChannelsSlicesData2, SimpleId>;
    type SubSelect = std::ops::Range<Self::Metrics>;
}

#[async_trait]
impl  Stage for CrossLoader {
    type In = ResourcesVec;
    type Out = ThoTensor2<ChannelsSlicesData2, SimpleId>;
    type Ctx = RunCtx2;

    async fn run(self: Arc<Self>, _input: Arc<Self::In>, _ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    {
        // unimplemented!()
        for (i_0, resource_0) in _input.resources.iter().enumerate() {
            let t_0 = Arc::clone(&self.loader).run_cached(
                Arc::clone(resource_0),
                Arc::clone(&_ctx)
            ).await?;

            for (i_1, resource_1) in _input.resources.iter().enumerate() {
                if i_1 >= i_0 {
                    let t_1 = Arc::clone(&self.loader).run_cached(
                        Arc::clone(resource_1),
                        Arc::clone(&_ctx)
                    ).await?;

                    // let result = Arc::clone(&self.cross_unfold).run(
                    //     Arc::new(TensorsPair::new("".into(), Arc::clone(&t_0), t_1)),
                    //     Arc::clone(&_ctx)
                    // ).await?;


                    let result = Arc::clone(&self.cross_unfold).do_run(
                        &TensorsPair::new("".into(), Arc::clone(&t_0), t_1),
                        Arc::clone(&_ctx)
                    )?;

                    debug!("resource_0={resource_0:?} resource_1={resource_1:?}: result={result:?}");


                }
            }

        }

        Ok(ThoTensor2::new("id".into(), Tensor::zeros(&[10,0], FLOAT_CPU)))
    }


}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use log::debug;
    use tch::{Device, Tensor};
    use tch::kind::FLOAT_CPU;
    use caching_runner::common::cachable::Cached;
    use caching_runner::common::runnable::Stage;
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_common::applicables::ThoDevice;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_motifs::distance::clc_dist_d2;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::ThoosaError;
    use torch_runner::data::ThoTensor2;
    use crate::tasks::cross_loader::{CrossLoader, ResourcesVec};
    use crate::tasks::cross_unfold::CrossUnfold;
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::{TensorLoader};

    #[ctor::ctor]
    pub fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test]
    async fn test_0() -> Result<(), ThoosaError> {
        GlobalCtx::init_devices(2);
        let resources: Vec<Resource> = vec!["111", "222", "333", "444", "555",  "666"].into_iter().enumerate().map(|(i, v)| (i as u32, 1u32, v, 10).into()).collect();
        let resources_to_move = resources.clone();

        let ctx = Arc::new(parking_lot::RwLock::new(RunCtx2::new(0, 0, "aa")));

        let  cl = Arc::new(CrossLoader::new(
            Arc::new(TensorLoader::new(
                move |key, _| {

                    let i = resources_to_move.iter().position(|v| v.eq(key)).expect(format!("no {key:?}").as_str()) + 1;
                    let v = (0..10).map(|v| i as f32 + v as f32 / 10.).collect::<Vec<f32>>();
                    debug!("executor: key={key:?} i={i} v={v:?}");


                    ThoTensor2::new(Input::id(key).clone(), Tensor::from_slice(v.as_slice()).view((1, 10)))
                },
                ThoDevice::from(Device::Cpu)
            )),
            Arc::new(CrossUnfold::new())
        ));

        let r = cl.run(
            Arc::new(ResourcesVec::new("".into(), resources)),
            ctx
        ).await?;

        Ok(())
    }

    #[test]
    fn test_batch_unfold() -> Result<(), ThoosaError>{

        let t_0_inp = dummy_t(0, 3, 5);

        let t_1_inp = dummy_t(1, 3, 4);

        let t_2_inp = dummy_t(2, 3, 6);

        println!("t_0=\n{t_0_inp}");



        let tensors = vec![t_0_inp, t_1_inp, t_2_inp];
        let window = 2;
        let stride = 1;

        let batch_size = 3;
        let  max_batch = 20;//5+4+6=15 < 20

        let mut t_0_batch:Vec<&Tensor> = vec![];
        let mut t_0_batch_len = 0;

        let mut skip_0_i =0;

        for (i_0, t_0) in tensors.iter().enumerate(){
            t_0_batch.push(t_0);
            t_0_batch_len+=t_0.size()[1];

            if t_0_batch_len < max_batch && i_0 < tensors.len()-1 {
                continue
            }

            let batch_0_t = Tensor::concat(t_0_batch.iter().map(|t|t.unfold(1, 2, 1)).collect::<Vec<_>>().as_slice(), 1);


            let mut t_1_batch:Vec<&Tensor> = vec![];
            let mut t_1_batch_len = 0;

            for (i_1, t_1) in tensors.iter().enumerate().skip(skip_0_i){
                t_1_batch.push(t_1);

                t_1_batch_len+=t_1.size()[1];

                if t_1_batch_len < max_batch && i_1 < tensors.len()-1 {
                    continue
                }

                let batch_1_t = Tensor::concat(t_1_batch.iter().map(|t|t.unfold(1, 2, 1)).collect::<Vec<_>>().as_slice(), 1);

                println!("t_1_batch.len={} batch_0_t={batch_0_t:?}  batch_1_t={batch_1_t:?}", t_1_batch.len());

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