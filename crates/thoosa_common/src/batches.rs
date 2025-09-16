use crate::applicables::{ThoRunCtx, ThoRunner};
use crate::dims::{AnyDims, Batch, Dims, ThoTensor, ThoBatch};
use thoosa_shared::applicables_runner::CachableId;
use crate::loader::{LoadRunner, Loader};
use crate::utils::{tensor_mem_size, DataBatch, Splittable, SplittableDataBatch};
use async_trait::async_trait;
use crossbeam_utils::sync::WaitGroup;
use futures_util::FutureExt;
use log::debug;
use parking_lot::{Mutex, RwLock};
use rand::seq::{SliceRandom, IndexedRandom};
use rand::Rng;
use std::cell::RefCell;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;
use std::{fmt, thread};
use tch::Kind::Float;
use tch::{Device, Tensor};
use thoosa_shared::resource::Resource;

#[async_trait]
pub trait AsyncDataBatch<D: Dims>: Send + Sync {
    /// returns (inputs, labels)
    async fn train_batch(&self) -> (ThoBatch<D>, Tensor);
    async fn valid_batch(&self) -> (ThoBatch<D>, Tensor);
}

/**
* returns num labels * items_per_label x channels x motifs x motifs batch size
* TODO rename
*/
// #[derive()]
pub struct AsyncLoadingTensorsDataBatch<
    DI: Dims,
    DO: Dims + Send,
    F: (Fn(ThoTensor<DI, CachableId>) -> ThoTensor<DO, CachableId>) + Send + Sync,
> {
    train_resources: Arc<Vec<(i64, Vec<Resource>)>>,
    valid_resources: Arc<Vec<(i64, Vec<Resource>)>>,

    batch_size: usize,
    runner: Arc<ThoRunner>,
    device: Device,
    transform_fn: Option<Arc<RwLock<Box<F>>>>, //for single tensor, wrap into BatchTransform for per-batch transform
    _pd_di: PhantomData<DI>,
    _pd_do: PhantomData<DO>,

    last_cache_area: Vec<ThoRunCtx>,
}

impl<DI: Dims, DO: Dims + Send, F: Fn(ThoTensor<DI, CachableId>) -> ThoTensor<DO, CachableId> + Send + Sync> Debug
    for AsyncLoadingTensorsDataBatch<DI, DO, F>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("")
            .field("train", &self.train_resources)
            .field("val", &self.valid_resources)
            .field("batch_size", &self.batch_size)
            .field("device", &self.device)
            .finish()
    }
}

impl<DI: Dims, DO: Dims + Send, F: Fn(ThoTensor<DI, CachableId>) -> ThoTensor<DO, CachableId> + Send + Sync>
    AsyncLoadingTensorsDataBatch<DI, DO, F>
{
    pub fn new(
        train: Arc<Vec<(i64, Vec<Resource>)>>,
        val: Arc<Vec<(i64, Vec<Resource>)>>,
        batch_size: usize,
        runner: Arc<ThoRunner>,
        device: Device,
        transform_fn: Option<Box<F>>,
    ) -> Self {
        Self {
            train_resources: train,
            valid_resources: val,
            batch_size,
            runner,
            device,
            transform_fn: transform_fn.map(|f| Arc::new(RwLock::new(f))),

            _pd_di: PhantomData,
            _pd_do: PhantomData,
            last_cache_area: vec![],
        }
    }

    async fn choose_for_batch(
        class_resources: &Vec<(i64, Vec<Resource>)>,
        batch_size: usize,
        runner: Arc<ThoRunner>,
        device: &Device,
        transform_fn: Option<Arc<RwLock<Box<F>>>>,
        area: Option<Arc<RwLock<ThoRunCtx>>>,
    ) -> (ThoBatch<DO>, Tensor) {
        debug!("AsyncLoadingTensorsDataBatch::choose_for_batch: batch_size={batch_size} class_resources.len={}", class_resources.len());

        // let wg = WaitGroup::new();
        let mut x_y_acc: Vec<(ThoTensor<DO, CachableId>, i64)> = vec![];

        if batch_size < class_resources.len() {
            debug!(
                "AsyncLoadingTensorsDataBatch::choose_for_batch: batch_size <  class_resources.len"
            );

            for _ in 0..batch_size {
                // let x_y_acc_in_exec = Arc::clone(&x_y_acc);
                // let wg_exec = wg.clone();
                let transform_fn = transform_fn.as_ref().map(|v| Arc::clone(v));

                let (class, resources) = class_resources
                    .choose(&mut rand::rng())
                    .expect("class_resources.choose failed");
                let resource_id = resources
                    .choose(&mut rand::rng())
                    .expect("resources.choose failed")
                    .clone();
                x_y_acc.push((
                    load_tensor(
                        &runner,
                        area.as_ref().expect("no area"),
                        /* &mut x_y_acc, */ transform_fn,
                        resource_id, /*, class*/
                    )
                    .await,
                    *class,
                ));
            }
            // wg.wait();
            // runner.purge_all_gpu_caches(Arc::into_inner(area).unwrap().into_inner(), *device).await;
            return Self::vec_to_tensors(device, x_y_acc);
        }

        let per_class_len = batch_size / class_resources.len();
        debug!("AsyncLoadingTensorsDataBatch::choose_for_batch: class_tensors={} per_class_len={per_class_len}", class_resources.len());

        let mut x_y_vecs = vec![];
        for (class, resources) in class_resources {
            assert!(
                resources.len() > 0,
                "no tensors for class={class} in ClassTensorsDataBatch"
            );

            // let class = *class;
            let map = resources
                .choose_multiple(&mut rand::rng(), per_class_len)
                .map(|key| key.clone());
            for resource_id in map {
                let transform_fn = transform_fn.as_ref().map(|v| Arc::clone(v));

                x_y_vecs.push((
                    load_tensor(
                        &runner,
                        &area.as_ref().expect("no area"),
                        /*&mut x_y_vecs, */ transform_fn,
                        resource_id, /*, class*/
                    )
                    .await,
                    *class,
                ));
            }
        }

        debug!("AsyncLoadingTensorsDataBatch::choose_for_batch: x_y_vecs.1.len={} batch_size={batch_size}", x_y_vecs.len());
        // loader.read().debug("AsyncLoadingTensorsDataBatch::choose_for_batch");

        //TODO throw if not enough tensors for batch?
        (x_y_vecs.len()..batch_size).for_each(|i| {
            x_y_vecs.push((
                x_y_vecs[i - x_y_vecs.len()].0.shallow_clone(),
                x_y_vecs[i - x_y_vecs.len()].1,
            ));
        });
        // loader.write().purge_cache(area);

        Self::vec_to_tensors(device, x_y_vecs)
    }

    fn vec_to_tensors(
        device: &Device,
        mut x_y_vecs: Vec<(ThoTensor<DO, CachableId>, i64)>,
    ) -> (ThoBatch<DO>, Tensor) {
        let classes =
            Tensor::from_slice(x_y_vecs.iter().map(|e| e.1).collect::<Vec<_>>().as_slice())
                .to_device(*device);

        let tensor = Tensor::stack(
            x_y_vecs
                .into_iter()
                .map(|e| e.0.as_ref().shallow_clone())
                .collect::<Vec<_>>()
                .as_slice(),
            0,
        )
        .to_device(*device)
        .set_requires_grad(true);

        debug!("AsyncLoadingTensorsDataBatch::choose_for_batch: tensor.size={:?} tensor_size_in_bytes={:?} tensor.device()={:?}", tensor.size(), tensor_mem_size(&tensor), tensor.device());

        (tensor.totype(Float).into(), classes.totype(Float))
    }

    pub fn purge_gpu_caches(&mut self) {
        // let last_area = self.last_cache_area.replace(self.runner.next_ctx()).unwrap();
        // self.runner.purge_all_gpu_caches(Arc::)
    }
}

pub async fn load_tensor<
    DI: Dims + Send + Sync,
    DO: Dims + Send + Sync,
    F: Fn(ThoTensor<DI, CachableId>) -> ThoTensor<DO, CachableId> + Send + Sync,
>(
    loader: &Arc<ThoRunner>,
    area: &Arc<RwLock<ThoRunCtx>>,
    // x_y_acc_in_exec: &mut Vec<(ThoTensor<DO>, i64)>,
    // wg_exec: WaitGroup,
    transform_fn: Option<Arc<RwLock<Box<F>>>>,
    resource_id: Resource,
    // class: i64
) -> ThoTensor<DO, CachableId> {
    debug!(
        "AsyncLoadingTensorsDataBatch::load_tensor: 0 resource_id={resource_id:?} area={area:?} "
    );

    debug!(
        "AsyncLoadingTensorsDataBatch::load_tensor: 1 resource_id={resource_id:?} area={area:?}  "
    );
    let resource_id_debug = resource_id.clone();
    // let t = Loader::<Tensor>::load_future(Arc::clone(&loader), resource_id, area, future_id, done_sender).await;
    let t = loader
        .run_loader(&resource_id, &Device::Cpu, Some(Arc::clone(&area)))
        .await; //TODO device

    debug!("AsyncLoadingTensorsDataBatch::load_tensor: 2 resource_id={resource_id_debug:?} area={area:?} ");

    let t = if let Some(transform) = transform_fn {
        transform.read()(t.shallow_clone().into()) //TODO move load_tensor into Batch args to return correct DO
    } else {
        t.shallow_clone().into()
    };
    debug!("AsyncLoadingTensorsDataBatch::load_tensor: 3 resource_id={resource_id_debug:?} area={area:?} ");
    // x_y_acc_in_exec.push((t.into(), class));
    debug!("AsyncLoadingTensorsDataBatch::load_tensor: 4 resource_id={resource_id_debug:?} area={area:?} ");
    t
}

#[async_trait]
impl<
        DI: Dims + Sync,
        DO: Dims + Send + Sync,
        F: Fn(ThoTensor<DI, CachableId>) -> ThoTensor<DO, CachableId> + Send + Sync,
    > AsyncDataBatch<DO> for AsyncLoadingTensorsDataBatch<DI, DO, F>
{
    async fn train_batch(&self) -> (ThoBatch<DO>, Tensor) {
        let next_ctx = self.runner.next_ctx();
        let batch = tch::no_grad(|| {
            Self::choose_for_batch(
                &self.train_resources,
                self.batch_size,
                Arc::clone(&self.runner),
                &self.device,
                self.transform_fn.clone(),
                Some(Arc::clone(&next_ctx)),
            )
        })
        .await;

        //TODO keep list of ctx and purge one in n
        Arc::clone(&self.runner)
            .purge_all_gpu_caches(Arc::into_inner(next_ctx).unwrap().into_inner(), self.device)
            .await;

        batch
    }

    async fn valid_batch(&self) -> (ThoBatch<DO>, Tensor) {
        let next_ctx = self.runner.next_ctx();
        let batch = tch::no_grad(|| {
            let batch = Self::choose_for_batch(
                &self.valid_resources,
                self.batch_size,
                Arc::clone(&self.runner),
                &self.device,
                self.transform_fn.clone(),
                Some(Arc::clone(&next_ctx)),
            );

            batch
        })
        .await;

        //TODO keep list of ctx and purge one in n
        Arc::clone(&self.runner)
            .purge_all_gpu_caches(Arc::into_inner(next_ctx).unwrap().into_inner(), self.device)
            .await;

        batch
    }
}

impl<DI: Dims, DO: Dims + Send, F: Fn(ThoTensor<DI, CachableId>) -> ThoTensor<DO, CachableId> + Send + Sync> Splittable<1>
    for AsyncLoadingTensorsDataBatch<DI, DO, F>
{
    //TODO change to spelt(self)?
    fn split(&self) -> Option<[Self; 1]>
    where
        Self: Sized,
    {
        let mut batch = AsyncLoadingTensorsDataBatch {
            train_resources: self.train_resources.clone(),
            valid_resources: self.valid_resources.clone(),
            batch_size: self.batch_size,
            runner: Arc::clone(&self.runner),
            device: self.device,
            transform_fn: self.transform_fn.as_ref().map(|arc| Arc::clone(arc)),
            _pd_di: Default::default(),
            _pd_do: Default::default(),
            last_cache_area: vec![], //TODO purge caches?
        };

        batch.batch_size = batch.batch_size / 2; //TODO param

        if batch.batch_size > 0 {
            debug!(
                "Splittable<1> for LoadingTensorsDataBatch: batch.batch_size={}",
                batch.batch_size
            );
            Some([batch])
        } else {
            None
        }
    }
}

// impl <DI:Dims, DO:Dims + Send, F: Fn(ThoTensor<DI>)->ThoTensor<DO> + Send + Sync + Clone> SplittableDataBatch<DO> for AsyncLoadingTensorsDataBatch<DI, DO, F> {}

///////////////////////////////////
pub struct AsyncClassTensorsLoaderData<
    DI: Dims,
    DO: Dims, /*, F: (Fn(ThoTensor<DI>)->ThoTensor<DO>) + Send + Sync*/
> {
    train_resources: Arc<Vec<(Resource, Vec<f32>)>>,
    valid_resources: Arc<Vec<(Resource, Vec<f32>)>>,

    batch_size: usize,
    runner: Arc<ThoRunner>,
    device: Device,
    // transform_fn:Option<Arc<RwLock<Box<F>>>>,

    // _pd_f:PhantomData<F>,
    _pd_di: PhantomData<DI>,
    _pd_do: PhantomData<DO>,
}

impl<DI: Dims, DO: Dims /*, F: (Fn(ThoTensor<DI>)->ThoTensor<DO>) + Send + Sync*/> Debug
    for AsyncClassTensorsLoaderData<DI, DO /*, F*/>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClassTensorsLoaderData")
            .field("train_resources", &self.train_resources)
            .field("valid_resources", &self.valid_resources)
            .field("batch_size", &self.batch_size)
            .field("device", &self.device)
            .finish()
    }
}

impl<DI: Dims, DO: Dims /*F: Fn(ThoTensor<DI>)->ThoTensor<DO> + Send + Sync*/>
    AsyncClassTensorsLoaderData<DI, DO /*, F*/>
{
    pub fn new(
        train_loader: Arc<Vec<(Resource, Vec<f32>)>>,
        valid_loader: Arc<Vec<(Resource, Vec<f32>)>>,
        batch_size: usize,
        loader: Arc<ThoRunner>,
        device: Device,
        // transform_fn:Option<Box<F>>
    ) -> Self {
        Self {
            train_resources: train_loader,
            valid_resources: valid_loader,
            batch_size,
            runner: loader,
            device,
            // transform_fn: transform_fn.map(|f|Arc::new(RwLock::new(f))),

            // _pd_f:PhantomData,
            _pd_di: PhantomData,
            _pd_do: PhantomData,
        }
    }
}
#[async_trait]
impl<
        DI: Dims + Sync,
        DO: Dims + Send + Sync, /*, F: Fn(ThoTensor<DI>)->ThoTensor<DO> + Send + Sync*/
    > AsyncDataBatch<DO> for AsyncClassTensorsLoaderData<DI, DO /*, F*/>
{
    async fn train_batch(&self) -> (ThoBatch<DO>, Tensor) {
        let next_ctx = self.runner.next_ctx();
        let batch = tch::no_grad(|| {
            Self::choose_for_batch(
                &self.train_resources,
                self.batch_size,
                Arc::clone(&self.runner),
                &self.device,
                None,
            ) //self.transform_fn.as_ref())
        })
        .await;

        //TODO keep list of ctx and purge one in n
        Arc::clone(&self.runner)
            .purge_all_gpu_caches(Arc::into_inner(next_ctx).unwrap().into_inner(), self.device)
            .await;

        batch
    }

    async fn valid_batch(&self) -> (ThoBatch<DO>, Tensor) {
        let next_ctx = self.runner.next_ctx();

        let batch = tch::no_grad(|| {
            Self::choose_for_batch(
                &self.valid_resources,
                self.batch_size,
                Arc::clone(&self.runner),
                &self.device,
                None,
            ) //, self.transform_fn.as_ref())
        })
        .await;

        //TODO keep list of ctx and purge one in n
        Arc::clone(&self.runner)
            .purge_all_gpu_caches(Arc::into_inner(next_ctx).unwrap().into_inner(), self.device)
            .await;

        batch
    }
}

impl<DI: Dims, DO: Dims /* F: Fn(ThoTensor<DI>)->ThoTensor<DO> + Send + Sync*/>
    AsyncClassTensorsLoaderData<DI, DO /*, F*/>
{
    async fn choose_for_batch(
        resources: &Vec<(Resource, Vec<f32>)>,
        batch_size: usize,
        loader: Arc<ThoRunner>,
        device: &Device,
        area: Option<Arc<RwLock<ThoRunCtx>>>,
    ) -> (ThoBatch<DO>, Tensor) {
        debug!("ClassTensorsLoaderData.batch: resources={resources:?} batch_size={batch_size}");

        let x_y_vecs_mx = Arc::new(Mutex::new((vec![], vec![])));

        let x_y_acc_in_exec = Arc::clone(&x_y_vecs_mx); //TODO join all

        // let mut transform_fn = transform_fn.as_ref().map(|v|v.clone());

        let area = loader.next_ctx();
        let rnd_resoruces = resources
            .choose_multiple(&mut rand::rng(), batch_size)
            .map(|v| v.clone())
            .collect::<Vec<_>>();

        let mut resources_to_purge = Vec::with_capacity(batch_size);

        for (resource_id, labels_p) in rnd_resoruces {
            resources_to_purge.push(resource_id.clone());

            // let t = AsyncLoadingTensorsDataBatch::load_tensor(&loader, &area, transform_fn.map(|v|Arc::clone(v)), resource_id).await;
            let t = loader
                .run_loader(&resource_id, device, Some(Arc::clone(&area)))
                .await;

            let mut guard = x_y_acc_in_exec.lock();

            guard.0.push(t);
            guard.1.push(Tensor::from_slice(labels_p.as_slice()));
        }
        // loader.purge_all_gpu_caches(Arc::into_inner(area).unwrap().into_inner(), *device).await;

        debug!(
            "ClassTensorsLoaderData::choose_for_batch: x_y_vecs.1.len={} batch_size={batch_size}",
            x_y_vecs_mx.lock().0.len()
        );

        //TODO throw if not enough tensors for batch?
        let mut guard = x_y_vecs_mx.lock();
        // let mut inputs = guard.0.drain(..).collect::<Vec<_>>();
        let mut labels = guard.1.drain(..).collect::<Vec<_>>();

        assert!(labels.len() > 0, "No tensors for batch");

        /*
        let mut inputs = if let Some(ref transform) = transform_fn {
            guard.0.drain(..).collect::<Vec<_>>().iter().map(|t| transform(t.into()).as_ref().shallow_clone()).collect() //TODO par on devices
        } else {
            guard.0.drain(..).collect::<Vec<_>>()
        };*/

        let mut inputs = guard
            .0
            .drain(..)
            .map(|tt| tt.t.shallow_clone())
            .collect::<Vec<Tensor>>(); //TODO remove shallow clone

        drop(guard);
        drop(x_y_vecs_mx);

        assert_eq!(labels.len(), inputs.len(), "classes.len() != inputs.len()");

        (inputs.len()..batch_size).for_each(|i| {
            let idx = i - inputs.len();
            let tensor_x = inputs[idx].shallow_clone();
            let tensor_y = labels[idx].shallow_clone();
            inputs.push(tensor_x);
            labels.push(tensor_y);
        });

        (
            Tensor::stack(inputs.as_slice(), 0)
                .to_device(*device)
                .set_requires_grad(true)
                .into(),
            Tensor::stack(labels.as_slice(), 0).to_device(*device),
        )
    }
}

///////////////////////
//batch transform, for the single tensor transform pass transform_fn to the batch impl
pub struct AsyncBatchTransform<DI: Dims, DO: Dims, B, BF>
where
    B: AsyncDataBatch<DI> + Debug,
    BF: FnMut(ThoBatch<DI>) -> ThoBatch<DO> + Sync + Send,
{
    inner: Arc<Box<B>>,
    transform_fn: Arc<RwLock<BF>>,
    _pd_di: PhantomData<DI>,
    _pd_do: PhantomData<DO>,
}

impl<DI: Dims, DO: Dims, B, F> Debug for AsyncBatchTransform<DI, DO, B, F>
where
    B: AsyncDataBatch<DI> + Debug,
    F: FnMut(ThoBatch<DI>) -> ThoBatch<DO> + Sync + Send,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchTransform")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<DI: Dims, DO: Dims, B, F> AsyncBatchTransform<DI, DO, B, F>
where
    B: AsyncDataBatch<DI> + Debug,
    F: FnMut(ThoBatch<DI>) -> ThoBatch<DO> + Sync + Send,
{
    pub fn new(inner: Arc<Box<B>>, transform_fn: F) -> Self {
        Self {
            inner,
            transform_fn: Arc::new(RwLock::new(transform_fn)),
            _pd_di: PhantomData,
            _pd_do: PhantomData,
        }
    }
    pub fn from_rc(inner: Arc<Box<B>>, transform_fn: Arc<RwLock<F>>) -> Self {
        Self {
            inner,
            transform_fn,
            _pd_di: PhantomData,
            _pd_do: PhantomData,
        }
    }
}

#[async_trait]
impl<DI: Dims, DO: Dims, B, F> AsyncDataBatch<DO> for AsyncBatchTransform<DI, DO, B, F>
where
    B: AsyncDataBatch<DI> + Debug,
    F: FnMut(ThoBatch<DI>) -> ThoBatch<DO> + Sync + Send,
{
    async fn train_batch(&self) -> (ThoBatch<DO>, Tensor) {
        let (inputs, labels) = self.inner.train_batch().await;
        (self.transform_fn.write()(inputs), labels)
    }

    async fn valid_batch(&self) -> (ThoBatch<DO>, Tensor) {
        let (inputs, labels) = self.inner.valid_batch().await;
        (self.transform_fn.write()(inputs), labels)
    }
}

impl<DI: Dims, DO: Dims + Send, B, F> Splittable<1> for AsyncBatchTransform<DI, DO, B, F>
where
    B: SplittableDataBatch<DI> + Debug + AsyncDataBatch<DI>,
    F: FnMut(ThoBatch<DI>) -> ThoBatch<DO> + Sync + Send,
{
    fn split(&self) -> Option<[Self; 1]>
    where
        Self: Sized,
    {
        let Some([split]) = self.inner.split() else {
            panic!("BatchTransform::split() called on empty inner");
        };
        Some([AsyncBatchTransform::from_rc(
            Arc::new(Box::new(split)),
            Arc::clone(&self.transform_fn),
        )])
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_0() {}
}
