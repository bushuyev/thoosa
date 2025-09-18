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
use thoosa_common::dims::{ChannelsData, ChannelsSlicesData, Dims};
use thoosa_common::tho_tensor::CopyToDevice;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::resource::Resource;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Named, SimpleId, WithId};
use torch_runner::data::ThoTensor2;
use torch_runner::metrics::TwoDimsMetrics;
use crate::tasks::cross_unfold::{CrossUnfold, TensorsPair};
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::{RawChannelsData, TensorLoader};

#[derive(Debug)]
pub struct Slice {
    id:SimpleId,
    t:ThoTensor2<ChannelsSlicesData, SimpleId>,
}

impl Slice {
    pub fn new(id: SimpleId, t: ThoTensor2<ChannelsSlicesData, SimpleId>) -> Self {
        Self { id, t }
    }
}
impl WithId for Slice {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for Slice {}

#[async_trait]
impl CopyToDevice for Slice {
    async fn copy_to_device(&self, dst_device: usize) ->Self {
        Slice::new(self.id.clone(), self.t.copy_to_device(dst_device).await)
    }
}

impl  Data for Slice {
    type Metrics = TwoDimsMetrics;
    type Sub = Self;
    type SubSelect = std::ops::Range<Self::Metrics>;

    fn size(&self) -> Self::Metrics {
        self.t.size().try_into().expect("Data for CrossSlices::size failed")
    }

    fn device(&self) -> usize {
        self.t.device()
    }
}

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

#[async_trait]
impl  Stage for CrossLoader {
    type In = ResourcesVec;
    type Out = Slice;
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

        Ok(Slice::new("".into(), ThoTensor2::new("id".into(), Tensor::zeros(&[10,0], FLOAT_CPU))))
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
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::ThoosaError;
    use torch_runner::data::ThoTensor2;
    use crate::tasks::cross_loader::{CrossLoader, ResourcesVec};
    use crate::tasks::cross_unfold::CrossUnfold;
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::{RawChannelsData, TensorLoader};

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


                    RawChannelsData::new(ThoTensor2::new(Input::id(key).clone(), Tensor::from_slice(v.as_slice()).view((1, 10))))
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
    fn test_batch_unfold() {

        let t_0 = Tensor::from_slice(&[
            0.00, 0.01, 0.02, 0.03, 0.04,   //channel 0
            0.10, 0.11, 0.12, 0.13, 0.14,   //channel 1
            0.20, 0.21, 0.22, 0.23, 0.24,   //channel 2
        ]).view((3,-1));

        let t_1 = Tensor::from_slice(&[
            1.00, 1.01, 1.02, 1.03,   //channel 0
            1.10, 1.11, 1.12, 1.13,   //channel 1
            1.20, 1.21, 1.22, 1.23,   //channel 2
        ]).view((3,-1));

        let t_2 = Tensor::from_slice(&[
            2.00, 2.01, 2.02, 2.03, 2.04, 2.05,    //channel 0
            2.10, 2.11, 2.12, 2.13, 2.14, 2.15,    //channel 1
            2.20, 2.21, 2.22, 2.23, 2.24, 2.25,    //channel 2
        ]).view((3,-1));

        println!("t_0=\n{t_0}");
        let t_0_u = t_0.unfold(1, 2, 1);
        println!("t_0.unfold=\n{}", t_0_u);
        let t_1_u = t_1.unfold(1, 2, 1);
        println!("t_1.unfold=\n{}", t_1_u);
        let t_2_u = t_2.unfold(1, 2, 1);
        println!("t_2.unfold=\n{}", t_2_u);

        let batch = Tensor::concat(&[t_0_u, t_1_u, t_2_u], 1);

        println!("batch=\n{batch}");

        // let tensors = vec![t_0, t_1, t_2];
        // let window = 2;
        // let stride = 1;
        //
        // let batch_size = 3;
        //
        // for (i_0, t) in tensors.iter().enumerate(){
        //     for (i_1, t) in tensors.iter().enumerate(){
        //         if i_1 >= i_0 {
        //
        //         }
        //     }
        // }

    }
}