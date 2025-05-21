use std::fmt;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use async_trait::async_trait;
use crossbeam_utils::sync::WaitGroup;
use futures_util::FutureExt;
use log::debug;
use parking_lot::{Mutex, RwLock};
use rand::prelude::SliceRandom;
use rand::thread_rng;
use tch::{Device, Tensor};
use tch::Kind::Float;
use thoosa_shared::resource::Resource;
use crate::applicables::{ThoRunCtx, ThoRunner};
use crate::dims::{Batch, Dims, ThoTensor};
use crate::loader::{LoadRunner, Loader};
use crate::utils::{tensor_mem_size, Splittable, SplittableDataBatch};

#[async_trait]
pub trait AsyncDataBatch<D:Dims>  {
    /// returns (inputs, labels)
    async fn train_batch(&self) -> (ThoTensor<Batch<D>>, Tensor);
    async fn valid_batch(&self) -> (ThoTensor<Batch<D>>, Tensor);
}


/**
* returns num labels * items_per_label x channels x motifs x motifs batch size
* TODO rename
*/
#[derive(Clone)]
pub struct AsyncLoadingTensorsDataBatch<'b, DI:Dims, DO:Dims + Send, F: (Fn(ThoTensor<DI>)->ThoTensor<DO>) + Send + Sync + Clone> {
    train_resources: Arc<Vec<(i64, Vec <Resource>)>>,
    valid_resources: Arc<Vec<(i64, Vec <Resource>)>>,

    batch_size: usize,
    runner: Arc<ThoRunner>,
    device:Device,
    transform_fn:Option<Arc<RwLock<Box<F>>>>, //for single tensor, wrap into BatchTransform for per-batch transform
    _pd_b:PhantomData<&'b DI>,
    _pd_do:PhantomData<DO>,
}


impl <'b, DI:Dims, DO:Dims + Send, F: Fn(ThoTensor<DI>)->ThoTensor<DO> + Send + Sync + Clone> Debug for AsyncLoadingTensorsDataBatch<'_, DI, DO, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("")
            .field("train", &self.train_resources)
            .field("val", &self.valid_resources)
            .field("batch_size", &self.batch_size)
            .field("device", &self.device)
            .finish()
    }
}


impl <'b, DI:Dims, DO:Dims + Send, F: Fn(ThoTensor<DI>)->ThoTensor<DO> + Send + Sync + Clone> AsyncLoadingTensorsDataBatch<'b, DI, DO, F> {

    pub fn new(
        train: Arc<Vec<(i64, Vec <Resource>)>>,
        val: Arc<Vec<(i64, Vec <Resource>)>>,
        batch_size: usize,
        loader: Arc<ThoRunner>,
        device:Device,
        transform_fn:Option<Box<F>>
    ) -> Self {

        Self {
            train_resources: train,
            valid_resources: val,
            batch_size,
            runner: loader,
            device,
            transform_fn: transform_fn.map(|f|Arc::new(RwLock::new(f))),

            _pd_b:PhantomData,
            _pd_do:PhantomData,
        }
    }

    async fn choose_for_batch(
        class_resources: &Vec<(i64, Vec<Resource>)>,
        batch_size: usize,
        loader: Arc<ThoRunner>,
        device: &Device,
        transform_fn: Option<Arc<RwLock<Box<F>>>>,
    ) -> (ThoTensor<Batch<DO>>, Tensor) {
        debug!("LoadingTensorsDataBatch::choose_for_batch: batch_size={batch_size} class_resources.len={}", class_resources.len());


        // let wg = WaitGroup::new();
        let area = loader.next_ctx();
        let mut x_y_acc:Vec<(ThoTensor<DO>, i64)> = vec![];

        if batch_size <  class_resources.len() {
            debug!("LoadingTensorsDataBatch::choose_for_batch: batch_size <  class_resources.len");

            for _ in 0..batch_size {
                // let x_y_acc_in_exec = Arc::clone(&x_y_acc);
                // let wg_exec = wg.clone();
                let transform_fn = transform_fn.as_ref().map(|v|Arc::clone(v));

                let (class, resources) = class_resources.choose(&mut thread_rng()).expect("class_resources.choose failed");
                let resource_id = resources.choose(&mut thread_rng()).expect("resources.choose failed").clone();
                let class = *class;
                Self::load_tensor(&loader, area.clone(), &mut x_y_acc, transform_fn, resource_id, class).await;

            }
            // wg.wait();
            loader.purge_all_gpu_caches(Arc::into_inner(area).unwrap().into_inner(), *device).await;
            return Self::vec_to_tensors(device, x_y_acc);
        }

        let per_class_len = batch_size / class_resources.len();
        debug!("LoadingTensorsDataBatch::choose_for_batch: class_tensors={} per_class_len={per_class_len}", class_resources.len());

        let mut x_y_vecs = vec![];
        for (class, resources) in class_resources{
            assert!(resources.len() > 0, "no tensors for class={class} in ClassTensorsDataBatch");

            let class = *class;
            let map = resources.choose_multiple(&mut thread_rng(), per_class_len).map(|key| key.clone());
            for resource_id in map {

                let transform_fn = transform_fn.as_ref().map(|v|Arc::clone(v));

                Self::load_tensor(&loader, area.clone(), &mut x_y_vecs, transform_fn, resource_id, class).await;
            }
        }

        // let mut x_y_vecs_mx = class_resources.iter().fold(Arc::new(Mutex::new(vec![])), |mut x_y_acc, (class, resources)| {
        //     assert!(resources.len() > 0, "no tensors for class={class} in ClassTensorsDataBatch");
        //
        //     let class = *class;
        //     for resource_id in resources.choose_multiple(&mut rng, per_class_len).map(|key|key.clone()) {
        //
        //         let x_y_acc_in_exec = Arc::clone(&x_y_acc);
        //         let wg_exec = wg.clone();
        //         let transform_fn = transform_fn.as_ref().map(|v|Arc::clone(v));
        //
        //         Self::load_tensor(&loader, area, x_y_acc_in_exec, wg_exec, transform_fn, resource_id, class).await;
        //     }
        //
        //     x_y_acc
        // });

        // wg.wait();


        debug!("LoadingTensorsDataBatch::choose_for_batch: x_y_vecs.1.len={} batch_size={batch_size}", x_y_vecs.len());
        // loader.read().debug("LoadingTensorsDataBatch::choose_for_batch");

        //TODO throw if not enough tensors for batch?
        (x_y_vecs.len()..batch_size).for_each(|i|{
            x_y_vecs.push(
                (
                    x_y_vecs[i- x_y_vecs.len()].0.shallow_clone(),
                    x_y_vecs[i- x_y_vecs.len()].1
                )
            );

        });
        // loader.write().purge_cache(area);
        Self::vec_to_tensors(device, x_y_vecs)
    }

    async fn load_tensor(
        loader: &Arc<ThoRunner>,
        area: Arc<RwLock<ThoRunCtx>>,
        x_y_acc_in_exec: &mut Vec<(ThoTensor<DO>, i64)>,
        // wg_exec: WaitGroup,
        transform_fn: Option<Arc<RwLock<Box<F>>>>,
        resource_id: Resource,
        class: i64
    ) {
        debug!("LoadingTensorsDataBatch::load_tensor: 0 resource_id={resource_id:?} area={area:?}");

        // loader.write().exec_scoped(move |future_id, loader, keys_purge, done_sender| {
        //     async move {
                debug!("LoadingTensorsDataBatch::load_tensor: 1 resource_id={resource_id:?} area={area:?}");
                let resource_id_debug = resource_id.clone();
                // let t = Loader::<Tensor>::load_future(Arc::clone(&loader), resource_id, area, future_id, done_sender).await;
                let t = loader.run_loader(&resource_id, &Device::Cpu, Some(area.clone())).await;//TODO device

                debug!("LoadingTensorsDataBatch::load_tensor: 2 resource_id={resource_id_debug:?} area={area:?}");

                let t = if let Some(transform) = transform_fn {
                    transform.read()(t.shallow_clone().into())//TODO move load_tensor into Batch args to return correct DO
                } else {
                    t.shallow_clone().into()
                };
                debug!("LoadingTensorsDataBatch::load_tensor: 3 resource_id={resource_id_debug:?} area={area:?}");
                x_y_acc_in_exec.push((t.into(), class));
                debug!("LoadingTensorsDataBatch::load_tensor: 4 resource_id={resource_id_debug:?} area={area:?}");
                // drop(wg_exec);
        //     }.boxed()
        // });
    }

    fn vec_to_tensors(device: &Device, mut x_y_vecs: Vec<(ThoTensor<DO>, i64)>) -> (ThoTensor<Batch<DO>>, Tensor) {
        let classes = Tensor::from_slice(x_y_vecs.iter().map(|e| e.1).collect::<Vec<_>>().as_slice()).to_device(*device);

        let tensor = Tensor::stack(
            x_y_vecs.into_iter().map(|e| e.0.as_ref().shallow_clone()).collect::<Vec<_>>().as_slice(),
            0,
        ).to_device(*device).set_requires_grad(true);

        debug!("LoadingTensorsDataBatch::choose_for_batch: tensor.size={:?} tensor_size_in_bytes={:?} tensor.device()={:?}", tensor.size(), tensor_mem_size(&tensor), tensor.device());

        (
            tensor.totype(Float).into(),
            classes.totype(Float)
        )
    }
}


#[async_trait]
impl <'b, DI:Dims + Sync, DO:Dims + Send + Sync, F: Fn(ThoTensor<DI>)->ThoTensor<DO> + Send + Sync + Clone> AsyncDataBatch<DO> for AsyncLoadingTensorsDataBatch<'_, DI, DO, F> {

    async fn train_batch(&self) -> (ThoTensor<Batch<DO>>, Tensor){
        tch::no_grad(|| {
            Self::choose_for_batch(&self.train_resources, self.batch_size, Arc::clone(&self.runner), &self.device, self.transform_fn.clone())
        }).await
    }

    async fn valid_batch(&self) -> (ThoTensor<Batch<DO>>, Tensor){
        tch::no_grad(|| {
            Self::choose_for_batch(&self.valid_resources, self.batch_size, Arc::clone(&self.runner), &self.device, self.transform_fn.clone())
        }).await
    }
}


impl <'b, DI:Dims, DO:Dims + Send, F: Fn(ThoTensor<DI>)->ThoTensor<DO> + Send + Sync + Clone> Splittable<1> for AsyncLoadingTensorsDataBatch<'_, DI, DO, F> {
    fn split(&self) -> Option<[Self; 1]>
    where
        Self: Sized
    {
        let mut batch = self.clone();
        batch.batch_size = batch.batch_size/2;//TODO param

        if batch.batch_size > 0 {
            debug!("Splittable<1> for LoadingTensorsDataBatch: batch.batch_size={}", batch.batch_size);
            Some([batch])
        } else {
            None
        }

    }
}

// impl <'b, DI:Dims, DO:Dims + Send, F: Fn(ThoTensor<DI>)->ThoTensor<DO> + Send + Sync + Clone> SplittableDataBatch<DO> for AsyncLoadingTensorsDataBatch<'_, DI, DO, F> {}


#[cfg(test)]
mod tests {
    #[test]
    fn test_0(){

    }
}