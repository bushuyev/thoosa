use crate::builder::{Node, SeqBuilder};
use crate::loader::{LoadRunner, Loader};
use crate::my_tch::ThoModule;
use crate::parzen::optimizer::{AsyncTPEOptimizer, ConfigurationSpace, TPEOptimizer};
use crate::runs::{Empty, Labeled, Logits, NetResources, SavableResult};
use crate::utils::{
    cuda_mem, run_split_on_oom, tensor_mem_size, train_2, DummyListener, Reducer,
    SplitQueue, Splittable, SplittableDataBatch, SplittableTensor,
};
use crate::utils::{pad_to, plot_series, train, DataBatch, ListenerResponse, TrainListener};
use crate::ws::Workspace;
use clap::clap_derive::Args;
use crossbeam_utils::sync::WaitGroup;
use futures::FutureExt;
use std::cell::RefCell;
use thoosa_shared::{panic_to_err, panic_to_err_async, CopyToDevice};

use log::{debug, info, trace, warn, Log};
use parking_lot::{Mutex, RwLock};
use plotters::prelude::BLACK;
use rand::seq::{IndexedRandom, SliceRandom};

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::{fmt, fs};

use std::marker::PhantomData;
use std::ops::Deref;
use std::path::PathBuf;
use std::rc::Rc;

use tch::nn::{init, AdamW, ConvConfig, LinearConfig, OptimizerConfig, VarStore};
use tch::Device::{Cpu, Cuda};
use tch::Kind::Float;
use tch::{nn, Device, IndexOp, Reduction, Tensor};

use thoosa_shared::{clc_step, slices_len};

use thoosa_shared::resource::Resource;
use thoosa_shared::val::{NamedRange, NamedVal, Val};
use thoosa_shared::{make_combos, RangeOrValue, ThoosaError};

use async_trait::async_trait;
use futures_util::future::join_all;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use thoosa_dummy_data_derive::DummyData;
use thoosa_store::Store;

use crate::tho_device::{ThoDevice};
use crate::batches::AsyncDataBatch;
use crate::dims::{
    AnyDims, AnyMetrics, AnySub, Batch, ChannelsData, ChannelsMotifsData,
    ChannelsMotifsFlattenedData, Dims, Final, LabelNetInputTensorDims, LabelNetsProbabilities,
    LabelProbability, LabelsProbabilities, MotifProbability, MotifsNetsSelectionDims, Slice, ThoBatch,
    TCD, TCSD,
};
use crate::find_motifs_run_config::*;
use crate::tho_tensor::ThoTensor;
use thoosa_properties_map::ToHashmap;
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::applicables_runner::{CachableId, Input, Output, Runner, ToOrdinal};
use thoosa_shared::copy_mode::{CpuOnly, CpuOrGpu};
use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};
use thoosa_shared::run_configs::{RunConfig};
use thoosa_shared::run_configs::{OptRunConfigs, RunId};
use thoosa_shared::traits::{Data, DataId, SimpleId, WithId};

#[derive(Args, Debug, Clone)]
pub struct TrainLabelNetsArgs {
    #[arg(long)]
    pub select_motifs_run_id: Vec<RunId>,

    #[arg(long)]
    pub train_resource_group: String,

    #[arg(long)]
    valid_resource_group: String,

    #[command(flatten)]
    pub fixed_config: OptConfig,

    #[arg(long)]
    pub run_config_cnt: String, //TrainLabelNetsRunConfig
}

impl TrainLabelNetsArgs {
    pub fn resource_config(&self) -> NetResources<Labeled> {
        NetResources::new(
            Labeled::new(self.train_resource_group.clone()),
            Labeled::new(self.valid_resource_group.clone()),
        )
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
#[run_config_opt]
pub struct TrainLabelNetsRunConfig {
    #[in_map]
    pub id: Option<RunId>,

    #[in_map]
    #[run_config]
    pub common_training_args: CommonTrainingRunConfig,

    #[in_map]
    #[run_config]
    pub net: LabelNetTypeArgs,
}

impl TrainLabelNetsRunConfig {
    pub fn new(
        id: Option<RunId>,
        common_training_args: CommonTrainingRunConfig,
        net: LabelNetTypeArgs,
    ) -> Self {
        Self {
            id,
            common_training_args,
            net,
        }
    }
}


//batch transform, for single tensor transform pass transform_fn to the batch impl
struct BatchTransform<DI: Dims, DO: Dims, B, F>
where
    B: DataBatch<DI> + Debug,
    F: FnMut(ThoBatch<DI>) -> ThoBatch<DO> + Sync + Send,
{
    inner: Rc<Box<B>>,
    transform_fn: Rc<RefCell<F>>,
    _pd_di: PhantomData<DI>,
    _pd_do: PhantomData<DO>,
}

impl<DI: Dims, DO: Dims, B, F> Debug for BatchTransform<DI, DO, B, F>
where
    B: DataBatch<DI> + Debug,
    F: FnMut(ThoBatch<DI>) -> ThoBatch<DO> + Sync + Send,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchTransform")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<DI: Dims, DO: Dims, B, F> BatchTransform<DI, DO, B, F>
where
    B: DataBatch<DI> + Debug,
    F: FnMut(ThoBatch<DI>) -> ThoBatch<DO> + Sync + Send,
{
    pub fn new(inner: Rc<Box<B>>, transform_fn: F) -> Self {
        Self {
            inner,
            transform_fn: Rc::new(RefCell::new(transform_fn)),
            _pd_di: PhantomData,
            _pd_do: PhantomData,
        }
    }
    pub fn from_rc(inner: Rc<Box<B>>, transform_fn: Rc<RefCell<F>>) -> Self {
        Self {
            inner,
            transform_fn,
            _pd_di: PhantomData,
            _pd_do: PhantomData,
        }
    }
}

impl<DI: Dims, DO: Dims, B, F> DataBatch<DO> for BatchTransform<DI, DO, B, F>
where
    B: DataBatch<DI> + Debug,
    F: FnMut(ThoBatch<DI>) -> ThoBatch<DO> + Sync + Send,
{
    fn train_batch(&self) -> (ThoBatch<DO>, Tensor) {
        let (inputs, labels) = self.inner.train_batch();
        (self.transform_fn.borrow_mut()(inputs), labels)
    }

    fn valid_batch(&self) -> (ThoBatch<DO>, Tensor) {
        let (inputs, labels) = self.inner.valid_batch();
        (self.transform_fn.borrow_mut()(inputs), labels)
    }
}

impl<DI: Dims, DO: Dims, B, F> Splittable<1> for BatchTransform<DI, DO, B, F>
where
    B: SplittableDataBatch<DI> + Debug,
    F: FnMut(ThoBatch<DI>) -> ThoBatch<DO> + Sync + Send,
{
    fn split(&self) -> Option<[Self; 1]>
    where
        Self: Sized,
    {
        let Some([split]) = self.inner.split() else {
            panic!("BatchTransform::split() called on empty inner");
        };
        Some([BatchTransform::from_rc(
            Rc::new(Box::new(split)),
            Rc::clone(&self.transform_fn),
        )])
    }
}

/**
* returns num labels * items_per_label x channels x motifs x motifs batch size
* TODO rename
*/
#[derive(Clone)]
struct LoadingTensorsDataBatch<
    'b,
    DI: Dims,
    DO: Dims + Send,
    F: (Fn(ThoTensor<DI, CachableId>) -> ThoTensor<DO, CachableId>) + Send + Sync + Clone,
> {
    train_resources: Rc<Vec<(i64, Vec<Resource>)>>,
    valid_resources: Rc<Vec<(i64, Vec<Resource>)>>,

    batch_size: usize,
    loader: Arc<RwLock<LoadRunner<Tensor>>>,
    device: Device,
    transform_fn: Option<Arc<RwLock<Box<F>>>>, //for single tensor, wrap into BatchTransform for per-batch transform
    _pd_b: PhantomData<&'b DI>,
    _pd_do: PhantomData<DO>,
}

impl<
    'b,
    DI: Dims,
    DO: Dims + Send,
    F: Fn(ThoTensor<DI, CachableId>) -> ThoTensor<DO, CachableId> + Send + Sync + Clone,
> Debug for LoadingTensorsDataBatch<'_, DI, DO, F>
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

impl<
    'b,
    DI: Dims,
    DO: Dims + Send,
    F: Fn(ThoTensor<DI, CachableId>) -> ThoTensor<DO, CachableId> + Send + Sync + Clone,
> LoadingTensorsDataBatch<'b, DI, DO, F>
{
    pub fn new(
        train: Rc<Vec<(i64, Vec<Resource>)>>,
        val: Rc<Vec<(i64, Vec<Resource>)>>,
        batch_size: usize,
        loader: Arc<RwLock<LoadRunner<Tensor>>>,
        device: Device,
        transform_fn: Option<Box<F>>,
    ) -> Self {
        Self {
            train_resources: train,
            valid_resources: val,
            batch_size,
            loader,
            device,
            transform_fn: transform_fn.map(|f| Arc::new(RwLock::new(f))),

            _pd_b: PhantomData,
            _pd_do: PhantomData,
        }
    }

    fn choose_for_batch(
        class_resources: &Vec<(i64, Vec<Resource>)>,
        batch_size: usize,
        loader: Arc<RwLock<LoadRunner<Tensor>>>,
        device: &Device,
        transform_fn: Option<Arc<RwLock<Box<F>>>>,
    ) -> (ThoBatch<DO>, Tensor) {
        debug!(
            "LoadingTensorsDataBatch::choose_for_batch: batch_size={batch_size} class_resources.len={}",
            class_resources.len()
        );

        let mut rng = rand::rng();
        let wg = WaitGroup::new();
        let area = loader.write().get_available_cache_area();
        let x_y_acc: Arc<Mutex<Vec<(ThoTensor<DO, CachableId>, i64)>>> =
            Arc::new(Mutex::new(vec![]));

        if batch_size < class_resources.len() {
            debug!("LoadingTensorsDataBatch::choose_for_batch: batch_size <  class_resources.len");

            for _ in 0..batch_size {
                let x_y_acc_in_exec = Arc::clone(&x_y_acc);
                let wg_exec = wg.clone();
                let transform_fn = transform_fn.as_ref().map(|v| Arc::clone(v));

                let (class, resources) = class_resources
                    .choose(&mut rng)
                    .expect("class_resources.choose failed");
                let resource_id = resources
                    .choose(&mut rng)
                    .expect("resources.choose failed")
                    .clone();
                let class = *class;
                Self::load_tensor(
                    &loader,
                    area,
                    x_y_acc_in_exec,
                    wg_exec,
                    transform_fn,
                    resource_id,
                    class,
                );
            }
            wg.wait();
            loader.write().purge_cache(area);
            return Self::vec_to_tensors(device, x_y_acc.lock().drain(..).collect()); //TODO drain
        }

        let per_class_len = batch_size / class_resources.len();
        debug!(
            "LoadingTensorsDataBatch::choose_for_batch: class_tensors={} per_class_len={per_class_len}",
            class_resources.len()
        );

        let mut x_y_vecs_mx = class_resources.iter().fold(
            Arc::new(Mutex::new(vec![])),
            |mut x_y_acc, (class, resources)| {
                assert!(
                    resources.len() > 0,
                    "no tensors for class={class} in ClassTensorsDataBatch"
                );

                let class = *class;
                resources
                    .choose_multiple(&mut rng, per_class_len)
                    .map(|key| key.clone())
                    .for_each(|resource_id| {
                        let x_y_acc_in_exec = Arc::clone(&x_y_acc);
                        let wg_exec = wg.clone();
                        let transform_fn = transform_fn.as_ref().map(|v| Arc::clone(v));

                        Self::load_tensor(
                            &loader,
                            area,
                            x_y_acc_in_exec,
                            wg_exec,
                            transform_fn,
                            resource_id,
                            class,
                        );
                    });

                x_y_acc
            },
        );

        wg.wait();

        let mut x_y_vecs: Vec<(ThoTensor<DO, CachableId>, i64)> =
            x_y_vecs_mx.lock().drain(..).collect();

        debug!(
            "LoadingTensorsDataBatch::choose_for_batch: x_y_vecs.1.len={} batch_size={batch_size}",
            x_y_vecs.len()
        );
        loader
            .read()
            .debug("LoadingTensorsDataBatch::choose_for_batch");

        //TODO throw if not enough tensors for batch?
        (x_y_vecs.len()..batch_size).for_each(|i| {
            x_y_vecs.push((
                x_y_vecs[i - x_y_vecs.len()].0.shallow_clone(),
                x_y_vecs[i - x_y_vecs.len()].1,
            ));
        });
        loader.write().purge_cache(area);
        Self::vec_to_tensors(device, x_y_vecs)
    }

    fn load_tensor(
        loader: &Arc<RwLock<LoadRunner<Tensor>>>,
        area: u64,
        x_y_acc_in_exec: Arc<Mutex<Vec<(ThoTensor<DO, CachableId>, i64)>>>,
        wg_exec: WaitGroup,
        transform_fn: Option<Arc<RwLock<Box<F>>>>,
        resource_id: Resource,
        class: i64,
    ) {
        debug!("LoadingTensorsDataBatch::load_tensor: 0 resource_id={resource_id:?} area={area}");

        loader.write().exec_scoped(move |future_id, loader, keys_purge, done_sender| {
            async move {
                debug!("LoadingTensorsDataBatch::load_tensor: 1 resource_id={resource_id:?} area={area}");
                let resource_id_debug = resource_id.clone();
                let t = Loader::<Tensor>::load_future(Arc::clone(&loader), resource_id, area, future_id, done_sender).await;

                debug!("LoadingTensorsDataBatch::load_tensor: 2 resource_id={resource_id_debug:?} area={area}");

                let t = if let Some(transform) = transform_fn {
                    transform.read()(t.into())
                } else {
                    t.into()
                };
                debug!("LoadingTensorsDataBatch::load_tensor: 3 resource_id={resource_id_debug:?} area={area}");
                x_y_acc_in_exec.lock().push((t.into(), class));
                debug!("LoadingTensorsDataBatch::load_tensor: 4 resource_id={resource_id_debug:?} area={area}");
                drop(wg_exec);
            }.boxed()
        });
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

        debug!(
            "LoadingTensorsDataBatch::choose_for_batch: tensor.size={:?} tensor_size_in_bytes={:?} tensor.device()={:?}",
            tensor.size(),
            tensor_mem_size(&tensor),
            tensor.device()
        );

        (tensor.totype(Float).into(), classes.totype(Float))
    }
}

impl<
    'b,
    DI: Dims,
    DO: Dims + Send,
    F: Fn(ThoTensor<DI, CachableId>) -> ThoTensor<DO, CachableId> + Send + Sync + Clone,
> DataBatch<DO> for LoadingTensorsDataBatch<'_, DI, DO, F>
{
    fn train_batch(&self) -> (ThoBatch<DO>, Tensor) {
        tch::no_grad(|| {
            Self::choose_for_batch(
                &self.train_resources,
                self.batch_size,
                Arc::clone(&self.loader),
                &self.device,
                self.transform_fn.clone(),
            )
        })
    }

    fn valid_batch(&self) -> (ThoBatch<DO>, Tensor) {
        tch::no_grad(|| {
            Self::choose_for_batch(
                &self.valid_resources,
                self.batch_size,
                Arc::clone(&self.loader),
                &self.device,
                self.transform_fn.clone(),
            )
        })
    }
}

impl<
    'b,
    DI: Dims,
    DO: Dims + Send,
    F: Fn(ThoTensor<DI, CachableId>) -> ThoTensor<DO, CachableId> + Send + Sync + Clone,
> Splittable<1> for LoadingTensorsDataBatch<'_, DI, DO, F>
{
    fn split(&self) -> Option<[Self; 1]>
    where
        Self: Sized,
    {
        let mut batch = self.clone();
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

impl<
    'b,
    DI: Dims,
    DO: Dims + Send,
    F: Fn(ThoTensor<DI, CachableId>) -> ThoTensor<DO, CachableId> + Send + Sync + Clone,
> SplittableDataBatch<DO> for LoadingTensorsDataBatch<'_, DI, DO, F>
{
}

struct ClassTensorsLoaderData<DI: Dims, DO: Dims> {
    train_resources: Vec<(Resource, Vec<f32>)>,
    valid_resources: Vec<(Resource, Vec<f32>)>,

    batch_size: usize,
    loader: Arc<RwLock<LoadRunner<Tensor>>>,
    device: Device,
    transform_fn:
        Option<Box<dyn Fn(ThoTensor<DI, CachableId>) -> ThoTensor<DO, CachableId> + Sync + Send>>,
}

impl<DI: Dims, DO: Dims> Debug for ClassTensorsLoaderData<DI, DO> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClassTensorsLoaderData")
            .field("train_resources", &self.train_resources)
            .field("valid_resources", &self.valid_resources)
            .field("batch_size", &self.batch_size)
            .field("device", &self.device)
            .finish()
    }
}

impl<DI: Dims, DO: Dims> ClassTensorsLoaderData<DI, DO> {
    pub fn new(
        train_loader: Vec<(Resource, Vec<f32>)>,
        valid_loader: Vec<(Resource, Vec<f32>)>,
        batch_size: usize,
        loader: Arc<RwLock<LoadRunner<Tensor>>>,
        device: Device,
        transform_fn: Option<
            Box<dyn Fn(ThoTensor<DI, CachableId>) -> ThoTensor<DO, CachableId> + Sync + Send>,
        >,
    ) -> Self {
        Self {
            train_resources: train_loader,
            valid_resources: valid_loader,
            batch_size,
            loader,
            device,
            transform_fn, // transform_fn.map(|f|Arc::new(RwLock::new(f)))
        }
    }
}

impl<DI: Dims, DO: Dims> DataBatch<DO> for ClassTensorsLoaderData<DI, DO> {
    fn train_batch(&self) -> (ThoBatch<DO>, Tensor) {
        Self::batch(
            &self.train_resources,
            self.batch_size,
            Arc::clone(&self.loader),
            &self.device,
            self.transform_fn.as_ref(),
        )
    }

    fn valid_batch(&self) -> (ThoBatch<DO>, Tensor) {
        Self::batch(
            &self.valid_resources,
            self.batch_size,
            Arc::clone(&self.loader),
            &self.device,
            self.transform_fn.as_ref(),
        )
    }
}

impl<DI: Dims, DO: Dims> ClassTensorsLoaderData<DI, DO> {
    fn batch(
        resources: &Vec<(Resource, Vec<f32>)>,
        batch_size: usize,
        loader: Arc<RwLock<LoadRunner<Tensor>>>,
        device: &Device,
        transform_fn: Option<
            &Box<dyn Fn(ThoTensor<DI, CachableId>) -> ThoTensor<DO, CachableId> + Sync + Send>,
        >,
    ) -> (ThoBatch<DO>, Tensor) {
        debug!("ClassTensorsLoaderData.batch: resources={resources:?} batch_size={batch_size}");

        let x_y_vecs_mx = Arc::new(Mutex::new((vec![], vec![])));
        let wg = WaitGroup::new();

        let x_y_acc_in_exec = Arc::clone(&x_y_vecs_mx);
        let wg_exec = wg.clone();

        // let mut transform_fn = transform_fn.as_ref().map(|v|v.clone());

        let area = loader.write().get_available_cache_area();
        let mut rng = rand::rng();
        let rnd_resoruces = resources
            .choose_multiple(&mut rng, batch_size)
            .map(|v| v.clone())
            .collect::<Vec<_>>();
        loader
            .write()
            .exec(move |future_id, loader, keys_purge, done_sender| {
                async move {
                    let mut resources_to_purge = Vec::with_capacity(batch_size);

                    for (resource_id, labels_p) in rnd_resoruces {
                        resources_to_purge.push(resource_id.clone());

                        let t = Loader::<Tensor>::load_future(
                            Arc::clone(&loader),
                            resource_id.clone(),
                            0,
                            future_id,
                            done_sender.clone(),
                        )
                        .await;

                        let mut guard = x_y_acc_in_exec.lock();

                        guard.0.push(t);
                        guard.1.push(Tensor::from_slice(labels_p.as_slice()));
                    }
                    keys_purge(&resources_to_purge, area); //TODO not on each batch
                    drop(wg_exec);
                }
                .boxed()
            });
        wg.wait();

        debug!(
            "ClassTensorsLoaderData::choose_for_batch: x_y_vecs.1.len={} batch_size={batch_size}",
            x_y_vecs_mx.lock().0.len()
        );

        //TODO throw if not enough tensors for batch?
        let mut guard = x_y_vecs_mx.lock();
        // let mut inputs = guard.0.drain(..).collect::<Vec<_>>();
        let mut labels = guard.1.drain(..).collect::<Vec<_>>();

        assert!(labels.len() > 0, "No tensors for batch");

        let mut inputs = if let Some(ref transform) = transform_fn {
            guard
                .0
                .drain(..)
                .collect::<Vec<_>>()
                .iter()
                .map(|t| transform(t.into()).as_ref().shallow_clone())
                .collect() //TODO par on devices
        } else {
            guard.0.drain(..).collect::<Vec<_>>()
        };

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

#[derive(Debug)]
pub struct TrainLabelNetsRunConfigResult {
    select_motifs_for_label_run_ids: Vec<RunId>,
    pub label: String,
    pub loss: f32,
    label_node_params: LabelNetParams,
    train_label_nets_run_config: TrainLabelNetsRunConfig,
    var_store: VarStore,
}

impl WithId for Box<TrainLabelNetsRunConfigResult> {
    type ID = CachableId;

    fn id(&self) -> &Self::ID {
        todo!()
    }
}


impl TrainLabelNetsRunConfigResult {
    pub fn new(
        select_motifs_for_label_run_ids: Vec<RunId>,
        label: String,
        loss: f32,
        label_node_params: LabelNetParams,
        train_motifs_nodes_values: TrainLabelNetsRunConfig,
        var_store: VarStore,
    ) -> Self {
        Self {
            select_motifs_for_label_run_ids,
            label,
            loss,
            label_node_params,
            train_label_nets_run_config: train_motifs_nodes_values,
            var_store,
        }
    }
}

impl SavableResult for TrainLabelNetsRunConfigResult {
    fn value(&self) -> f32 {
        self.loss
    }

    fn save(
        &mut self,
        store: &Arc<RwLock<Box<dyn Store>>>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
    ) -> Result<u32, ThoosaError> {
        let run_config_id = store.write().save_run_config(
            Some(&self.select_motifs_for_label_run_ids),
            &TrainLabelNetsRunConfig::type_name(),
            "Label",
            "todo",
            self.train_label_nets_run_config
                .to_properties_iter(None)
                .collect(),
        )?;

        ws.read()
            .save_label_model(
                run_config_id,
                &self.label,
                &self.var_store,
                &self.label_node_params,
            )
            .expect("save_label_model failed");

        Ok(run_config_id)
    }
}





#[derive(Debug)]
pub struct TrainFinalNetsRunConfigResult {
    train_label_nodes_run_ids: Vec<RunId>,
    loss: f32,
    var_store: VarStore,
    final_node_params: FinalNetParams,
    train_final_net_run_config: TrainFinalNetRunConfig,
}

impl TrainFinalNetsRunConfigResult {
    pub fn new(
        train_label_nodes_run_ids: Option<&Vec<RunId>>,
        loss: f32,
        var_store: VarStore,
        final_node_params: FinalNetParams,
        train_final_nodes_values: TrainFinalNetRunConfig,
    ) -> Self {
        Self {
            train_label_nodes_run_ids: train_label_nodes_run_ids
                .expect("train_label_nodes_run_ids")
                .clone(),
            loss,
            var_store,
            final_node_params,
            train_final_net_run_config: train_final_nodes_values,
        }
    }
}

impl SavableResult for TrainFinalNetsRunConfigResult {
    fn value(&self) -> f32 {
        self.loss
    }

    fn save(
        &mut self,
        store: &Arc<RwLock<Box<dyn Store>>>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
    ) -> Result<u32, ThoosaError> {
        let run_config_id = store.write().save_run_config(
            Some(&self.train_label_nodes_run_ids),
            &TrainFinalNetRunConfig::type_name(),
            "Final",
            "todo",
            self.train_final_net_run_config
                .to_properties_iter(None)
                .collect(),
        )?;

        ws.read()
            .save_final_net(run_config_id, &self.var_store, &self.final_node_params)
            .expect("save_final_model failed");

        Ok(run_config_id)
    }
}


pub fn train_final_net(
    run_config_values: &TrainFinalNetRunConfig,
    label_nets_num: u32,
    labels_num: u32,
    batch: &mut dyn DataBatch<LabelNetsProbabilities>,
    var_store: &VarStore,
    device: &Device,
) -> Result<(f32, FinalNetParams), ThoosaError> {
    let final_node_params =
        FinalNetParams::new(label_nets_num, labels_num, run_config_values.net.num_layers);
    debug!("train_final_node: final_node_params={final_node_params:?}");

    let mut optimizer = nn::Adam::default();
    /* if let Some(adam_w) = run_config_values.adam_w {
        debug!("train_final_nodes: setting adam_w={adam_w:?}");
        optimizer = optimizer.beta1(adam_w.beta1).beta2(adam_w.beta2);
    }*/

    let loss = train(
        device,
        batch,
        &final_net(&final_node_params)
            .build_tch_module::<Batch<LabelNetsProbabilities>, Batch<LabelsProbabilities>>(
                &var_store.root(),
            )?,
        |pred: &Tensor, target: &Tensor| {
            debug!(
                "train_final: pred.size={:?} target.size={:?}",
                pred.size(),
                target.size()
            );
            trace!("train_final: pred=\n{pred} target=\n{target}");
            let pred = pad_to(&pred, &target.size()).expect("pad_to failed");
            debug!("train_final: padded pred.size={:?}", pred.size());

            Ok(pred.cross_entropy_loss::<Tensor>(&target, None, Reduction::Mean, -100, 0.0))
        },
        run_config_values.common_training_args.cycles as usize,
        optimizer.build(&var_store, run_config_values.common_training_args.lr)?,
        |t: ThoBatch<LabelNetsProbabilities>| {
            trace!("train_final: t=\n{t}");
            debug!("train_final: t.size={:?} ", t.size());

            t.as_ref().squeeze().into() //TODO move squeeze
        },
        None,
    )?;

    Ok((loss, final_node_params))
}

type ModuleFactory<DI, DO> = dyn Fn(&nn::Path) -> Result<Box<dyn ThoModule<DI, DO>>, ThoosaError>;

pub struct ThoModuleWithStore<DI: Dims + Send + 'static + ?Sized, DO: Dims + Send> {
    id:SimpleId,
    module: Box<dyn ThoModule<DI, DO>>,
    var_store: VarStore,
    module_factory: Option<Arc<ModuleFactory<DI, DO>>>,
}

impl <DI, DO> WithId for ThoModuleWithStore<DI, DO>
where
    DI: Dims + Send + 'static + ?Sized,
    DO: Dims + Send, {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}


impl <DI, DO> Output for ThoModuleWithStore<DI, DO>
where
    DI: Dims + Send + 'static + ?Sized,
    DO: Dims + Send + 'static, {

}

impl<DI, DO> Debug for ThoModuleWithStore<DI, DO>
where
    DI: Dims + Send + 'static + ?Sized,
    DO: Dims + Send,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ThoModuleWithStore")
            .field("module", &self.module)
            .field("var_store", &self.var_store)
            .field("has_factory", &self.module_factory.is_some())
            .finish()
    }
}

//TODO
unsafe impl<DI: Dims + Send, DO: Dims + Send> Send for ThoModuleWithStore<DI, DO> {}
unsafe impl<DI: Dims + Send, DO: Dims + Send> Sync for ThoModuleWithStore<DI, DO> {}

impl<DI: Dims + Send + 'static, DO: Dims + Send + 'static> ThoModule<DI, DO>
    for ThoModuleWithStore<DI, DO>
{
    fn tho_forward(&self, t: &ThoTensor<DI, CachableId>, train: bool) -> Result<ThoTensor<DO, CachableId>, ThoosaError> {
        self.module.tho_forward(t, train)
    }
}

impl<DI: Dims + Send + 'static, DO: Dims + Send + 'static> ThoModuleWithStore<DI, DO> {
    // #[cfg(test)]
    pub fn new(id: SimpleId, module: Box<dyn ThoModule<DI, DO>>) -> Self {
        Self::with_factory(id, module, VarStore::new(Cpu), None)
    }

    pub fn with_factory(
        id: SimpleId,
        module: Box<dyn ThoModule<DI, DO>>,
        var_store: VarStore,
        module_factory: Option<Arc<ModuleFactory<DI, DO>>>,
    ) -> Self {
        Self {
            id,
            module,
            var_store,
            module_factory,
        }
    }

    pub fn load<F>(id: SimpleId, builder: F, model: PathBuf, device: Device) -> Result<Self, ThoosaError>
    where
        F: Fn(&nn::Path) -> Result<Box<dyn ThoModule<DI, DO>>, ThoosaError> + Send + Sync + 'static,
    {
        let builder: Arc<ModuleFactory<DI, DO>> = Arc::new(builder);
        let mut vs = VarStore::new(device);
        let module = builder(&vs.root())?;
        vs.load(&model)?;
        vs.freeze();

        Ok(Self::with_factory(id, module, vs, Some(builder)))
    }

    fn mem_total(&self) -> usize {
        self.var_store
            .variables()
            .iter()
            .map(|t| tensor_mem_size(t.1))
            .sum()
    }

    fn debug(&self) {
        self.module.debug();
    }

    pub fn infer(&self, t: &ThoTensor<DI, CachableId>) -> ThoTensor<DO, CachableId> {
        self.module.tho_forward(t, false).unwrap()
    }

    pub fn var_store(&self) -> &VarStore {
        &self.var_store
    }
}

#[async_trait]
impl <DI, DO> CopyToDevice for ThoModuleWithStore<DI, DO>
where
DI: Dims + Send + 'static,
DO: Dims + Send + 'static {

    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoosaError> {

        debug!("ThoModuleWithStore::copy_to_device: has module_factory={}", self.module_factory.is_some());

        let builder = self
            .module_factory
            .as_ref()
            .unwrap_or_else(|| panic!("copy_to_device requires module_factory; construct modules via ThoModuleWithStore::load to enable copying"));

        let mut var_store = VarStore::new(ThoDevice::from(dst_device).into());

        let module = builder(&var_store.root()).expect("module builder failed while copying to device");

        var_store.copy(&self.var_store).expect("failed to copy VarStore weights");

        var_store.freeze();

        Ok(Self::with_factory(self.id.clone(), module, var_store, Some(Arc::clone(builder))))
    }
}


impl <DI, DO> Data for ThoModuleWithStore<DI, DO>
where
    DI: Dims + Send + 'static,
    DO: Dims + Send + 'static{
    type Metrics = ();
    type M = CpuOrGpu<ThoModuleWithStore<DI, DO>>;

    fn size(&self) -> Self::Metrics {
        ()
    }

    fn device(&self) -> usize {
        ThoDevice::from(self.var_store.device()).ordinal()
    }
}

//TODO move to ws
pub fn load_final_model(
    run_id: RunId,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    device: &Device,
) -> Result<
    (
        FinalNetParams,
        ThoModuleWithStore<Batch<LabelProbability>, Batch<LabelsProbabilities>>,
    ),
    ThoosaError,
> {
    let (model_path, params_path) = ws.read().final_model(run_id)?;
    debug!("load_final_model: params_path={model_path:?} run_id={run_id}");

    let final_node_params = FinalNetParams::from_file(&params_path)?;
    debug!("load_final_model:  final_node_params={final_node_params:?}");

    let final_params_for_builder = final_node_params.clone();
    let node = ThoModuleWithStore::load("".into(),
                                        move |path: &nn::Path| final_net(&final_params_for_builder).build_tch_module(path),
                                        model_path,
                                        *device,
    )?;

    Ok((final_node_params, node))
}

#[derive(Args, Debug, Clone)]
pub struct TrainMotifsNetsArgs {
    #[arg(long)]
    pub select_motifs_run_ids: Vec<RunId>,

    #[command(flatten)]
    pub fixed_config: OptConfig,

    #[arg(long)]
    pub run_config_cnt: String, //TrainMotifsNetsRunConfig
}

impl TrainMotifsNetsArgs {
    pub fn resource_config(&self) -> Empty {
        Empty {}
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
#[run_config_opt]
pub struct TrainMotifsNetsRunConfig {
    #[in_map]
    pub id: Option<RunId>,

    #[in_map]
    #[run_config]
    pub common_training_args: CommonTrainingRunConfig,

    #[in_map]
    #[run_config]
    pub num_layers: u32,

    #[in_map]
    #[run_config]
    pub step: u32,
}

impl TrainMotifsNetsRunConfig {
    pub fn new(id: Option<RunId>) -> Self {
        Self {
            id,
            common_training_args: CommonTrainingRunConfig::new(1, 1, 0.1),
            num_layers: 1,
            step: 1,
        }
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
pub struct CommonTrainingRunConfig {
    #[in_map]
    #[run_config]
    pub cycles: u32,

    #[in_map]
    #[run_config]
    pub batch_size: u32, //number of the motif slices in a batch

    #[in_map]
    #[run_config]
    pub lr: f64,

}

impl CommonTrainingRunConfig {
    pub fn new(
        cycles: u32,
        batch_size: u32,
        lr: f64,
    ) -> Self {
        Self {
            cycles,
            batch_size,
            lr,
        }
    }
}

#[derive(Args, Debug, Clone)]
pub struct OptConfig {
    #[arg(long, default_value_t = 100)]
    pub opt_max_evals: u32,

    #[arg(long, default_value_t = 10)]
    pub opt_num_init: u32,
}

impl OptConfig {
    pub fn new(max_evals: u32, n_init: u32) -> Self {
        Self {
            opt_max_evals: max_evals,
            opt_num_init: n_init,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct MotifNetParams {
    pub num_features: u32,
    pub num_layers: u32,
    pub stride:u32
}

impl MotifNetParams {
    pub fn new(num_features: u32, num_layers: u32, stride: u32) -> Self {
        Self {
            num_features,
            num_layers,
            stride
        }
    }

    pub fn to_properties_map(&self) -> HashMap<String, Val> {
        HashMap::from([("num_features".into(), self.num_features.into())])
    }

    pub fn from_properties_map(properties: HashMap<String, Val>) -> Self {
        MotifNetParams::new(
            properties
                .get("num_features")
                .expect(format!("no num_features in {:?}", properties).as_str())
                .try_into()
                .expect("num_features as_i64"),
            1,
            100,
        )
    }

    pub fn from_file(path: &PathBuf) -> Result<Self, ThoosaError> {
        serde_json5::from_str::<MotifNetParams>(fs::read_to_string(&path)?.as_str()).map_err(
            |serde_err| ThoosaError::new(format!("err={}, path={path:?}", serde_err.to_string())),
        )
    }

    pub fn num_features(&self) -> u32 {
        self.num_features
    }

}


pub fn motif_node_2(params: &MotifNetParams) -> Node {
    debug!("motif_node: params={params:?}");
    let mut builder = SeqBuilder::new(params.num_features, None);
    let nd = ConvConfig {
        stride: params.stride as i64,
        ..ConvConfig::default()
    };

    let mut net = builder
        .closure(
            Arc::new(Box::new(move |t: &Tensor| {

                let r = t.unsqueeze(1);

                debug!("motif_node_2: 0. input.size={:?}", r.size());

                Ok(r)
            })),
            None,
        )
        // .batch_norm()
        .conv_1d("conv_1", 10, 1, 4, nd, None)
        .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
        .adaptive_max_pool(1, 4)
        .closure(
            Arc::new(Box::new(move |t: &Tensor| {

                let r = t.squeeze();

                debug!("motif_node_2: 1. input.size={:?}", r);

                Ok(r)
            })),
            None,
        )

        .linear_s(
            4,
            #[cfg(debug_assertions)]
            "motif_node first layer".to_string(),
        )
        .func(|t|Tensor::f_relu(t).map_err(|e|e.into()));

    for i in 0..params.num_layers as usize {
        net = net
            .linear_s(
                4,
                #[cfg(debug_assertions)]
                format!("motif_node  {i}th layer"),
            )
            .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
    }

    net
        // .linear_s(180)
        .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
        .linear_s(
            1,
            #[cfg(debug_assertions)]
            "motif_node last layer".to_string(),
        )
        .func(|t|Tensor::f_sigmoid(t).map_err(|e|e.into()))
        .end_seq()
}


pub fn motif_node(params: &MotifNetParams) -> Node {
    debug!("motif_node: params={params:?}");
    let mut builder = SeqBuilder::new(params.num_features, None);
    let mut net = builder
        // .batch_norm()
        // .conv_1d(10, 1, 1, ConvConfig::default())
        .linear_s(
            params.num_features * 2,
            #[cfg(debug_assertions)]
            "motif_node first layer".to_string(),
        )
        .func(|t|Tensor::f_relu(t).map_err(|e|e.into()));

    for i in 0..params.num_layers as usize {
        net = net
            .linear_s(
                params.num_features * 2,
                #[cfg(debug_assertions)]
                format!("motif_node  {i}th layer"),
            )
            .func(|t|Tensor::f_relu(t).map_err(|e|e.into()));
    }

    net
        // .linear_s(180)
        // .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
        .linear_s(
            1,
            #[cfg(debug_assertions)]
            "motif_node last layer".to_string(),
        )
        .func(|t|Tensor::f_sigmoid(t).map_err(|e|e.into()))
        .end_seq()
}

#[derive(Debug, Copy, Clone)]
pub enum NetsPadding {
    Max,
    Fixed(u32),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabelNetParams {
    channels: u32,
    motif_nets: u32,
    motif_layers: u32,
    pub slices_len: u32,
    pub net_type: LabelNetTypeParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LabelNetTypeParams {
    MaxPadding(MaxPaddingNetParams),

    FixedNumSlices(FixedNumSlicesNetParams),
}

impl Default for LabelNetTypeParams {
    fn default() -> Self {
        LabelNetTypeParams::MaxPadding(MaxPaddingNetParams::default())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaxPaddingNetParams {}

impl MaxPaddingNetParams {
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedNumSlicesNetParams {}

impl FixedNumSlicesNetParams {
    pub fn new() -> Self {
        Self {}
    }
}

impl LabelNetParams {
    pub fn new(
        channels: u32,
        motif_nets: u32,
        motif_layers: u32,
        slices_len: u32,
        net_type: LabelNetTypeParams,
    ) -> Self {
        Self {
            channels,
            motif_nets,
            slices_len,
            motif_layers,
            net_type,
        }
    }

    pub fn from_file(path: &PathBuf) -> Result<Self, ThoosaError> {
        serde_json5::from_str::<LabelNetParams>(fs::read_to_string(&path)?.as_str())
            .map_err(|serde_err| ThoosaError::new(serde_err.to_string()))
    }

    pub fn label_net(&self) -> Node {
        match &self.net_type {
            LabelNetTypeParams::MaxPadding(max_padding_params) => {
                LabelNetParams::label_net_max_padding(&self, max_padding_params)
            }
            LabelNetTypeParams::FixedNumSlices(fixed_num_slices_params) => {
                LabelNetParams::label_net_fixed_num_slices(&self, fixed_num_slices_params)
            }
        }
    }

    fn label_net_fixed_num_slices(
        params: &LabelNetParams,
        fixed_num_slices_params: &FixedNumSlicesNetParams,
    ) -> Node {
        debug!("label_net_fixed_num_slices: params={:?}", params);

        //sum of motif nets in all channels
        let motif_nodes = params.motif_nets;
        SeqBuilder::new(params.slices_len, None) //Some(2)
            .closure(
                Arc::new(Box::new(move |t: &Tensor| {
                    debug!("label_net_fixed_num_slices: input.size={:?}", t.size());
                    Ok(t.shallow_clone())
                })),
                None,
            )
            .split(
                (0..motif_nodes)
                    .map(|_i| {
                        let num_features = params.slices_len;
                        let mut builder = SeqBuilder::new(num_features, None);
                        let mut net = builder.linear_s(
                            num_features,
                            #[cfg(debug_assertions)]
                            "label_net_fixed_num_slices: motif nets first layer".to_string(),
                        );

                        for i in 0..params.motif_layers as usize {
                            //params.num_layers
                            net = net
                                .linear_s(
                                    num_features,
                                    #[cfg(debug_assertions)]
                                    format!("label_net_fixed_num_slices: motif nets {i}th layer"),
                                )
                                .func(|t|Tensor::f_relu(t).map_err(|e|e.into()));
                        }

                        net.func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
                            .linear_s(
                                1,
                                #[cfg(debug_assertions)]
                                "label_net_fixed_num_slices: motif nets last layer".to_string(),
                            )
                            // .func(|t|Tensor::f_sigmoid(t).map_err(|e|e.into()))
                            .end_seq()
                    })
                    .collect(),
                Box::new(move |t: &Tensor| {
                    Ok((0..motif_nodes)
                        .map(|net_i| {
                            debug!(
                                "label_net_fixed_num_slices: split, t.size={:?}  net_i={net_i}",
                                t.size()
                            );
                            t.i((.., net_i as i64, ..))
                        })
                        .collect())
                }), //label batch, channel, max_motif_nets, max slices
                Box::new(move |t: Vec<Tensor>| {
                    debug!("label_net_fixed_num_slices: join t={t:?}");

                    let out = Tensor::f_concat(t.as_slice(), 1)?; //.mean_dim(1, false, Float);
                    debug!("label_net_fixed_num_slices: join out={:?}", out.size());

                    Ok(out)
                }),
            )
            .closure(
                Arc::new(Box::new(move |t: &Tensor| {
                    debug!(
                        "label_net_fixed_num_slices: merged final in flatten t.size={:?}",
                        t.size()
                    );

                    Ok(t.shallow_clone())
                })),
                Some(motif_nodes),
            )
            .linear(
                #[cfg(debug_assertions)]
                "label_net_fixed_num_slices: first main layer".to_string(),
            )
            // .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
            .linear_s(
                1,
                #[cfg(debug_assertions)]
                "label_net_fixed_num_slices: final layer".to_string(),
            )
            .func(|t|Tensor::f_sigmoid(t).map_err(|e|e.into()))
            .end_seq()
    }

    fn label_net_max_padding(
        params: &LabelNetParams,
        max_padding_params: &MaxPaddingNetParams,
    ) -> Node {
        debug!("label_node_max_padding: params={:?}", params);

        let motif_nodes = params.motif_nets;
        SeqBuilder::new(params.slices_len, None) // batch x channel x max motif_nets x slices //TODO split?
            .closure(
                Arc::new(Box::new(move |t: &Tensor| {
                    debug!("label_node_max_padding: input.size={:?}", t.size());
                    Ok(t.shallow_clone())
                })),
                None,
            )
            .split(
                (0..motif_nodes)
                    .map(|_i| {
                        let num_features = params.slices_len;
                        let mut builder = SeqBuilder::new(num_features, None);
                        let mut net = builder.linear_s(
                            num_features,
                            #[cfg(debug_assertions)]
                            "label_net_max_padding: motif nets first layer".to_string(),
                        );

                        for i in 0..params.motif_layers as usize {
                            //params.num_layers
                            net = net
                                .linear_s(
                                    num_features,
                                    #[cfg(debug_assertions)]
                                    format!("label_net_max_padding: motif nets {i}-th layer"),
                                )
                                .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
                        }

                        net.func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
                            .linear_s(
                                1,
                                #[cfg(debug_assertions)]
                                "label_net_max_padding: motif nets last layer".to_string(),
                            )
                            // .func(|t|Tensor::f_sigmoid(t).map_err(|e|e.into()))
                            .end_seq()
                    })
                    .collect(),
                Box::new(move |t: &Tensor| {
                    Ok((0..motif_nodes)
                        .map(|net_i| {
                            debug!(
                                "label_net_max_padding: split, t.size={:?} net_i={net_i}",
                                t.size()
                            );
                            t.i((.., .., net_i as i64, ..))
                        })
                        .collect())
                }), //label batch, channel, max_motif_nets, max slices
                Box::new(move |t: Vec<Tensor>| {
                    debug!("label_net_max_padding: join t={t:?}");

                    let out = Tensor::f_stack(t.as_slice(), 1)?.f_mean_dim(1, false, Float)?;
                    debug!("label_net_max_padding: join out={:?}", out.size());

                    Ok(out)
                }),
            )
            .closure(
                Arc::new(Box::new(move |t: &Tensor| {
                    debug!(
                        "label_net_max_padding: merged final in flatten t.size()={:?}",
                        t.size()
                    );

                    let t_0 = t.squeeze_dims(2);
                    debug!(
                        "label_net_max_padding: merged final out flatten t_0.size()={:?}",
                        t_0.size()
                    );

                    Ok(t_0)
                })),
                Some(params.channels),
            )
            .linear(
                #[cfg(debug_assertions)]
                "label_net_max_padding: first main layer".to_string(),
            )
            // .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
            .linear_s(
                1,
                #[cfg(debug_assertions)]
                "label_net_max_padding: final layer".to_string(),
            )
            .func(|t|Tensor::f_sigmoid(t).map_err(|e|e.into()))
            .end_seq()
    }
}

pub fn any_len_label_node(params: &LabelNetParams) -> Node {
    let num_features = params.slices_len;//params.motif_nets;
    debug!("any_len_label_node: num_features={num_features}");

    SeqBuilder::new(num_features/*30000*/, None)
        .closure(
            Arc::new(Box::new(move |t: &Tensor| {
                debug!("any_len_label_node: input={t:?}");
                Ok(t.shallow_clone())
            })),
            None,
        )
        .split1(

        {
            let mut seq_builder = SeqBuilder::new(num_features, None);
            let mut builder = seq_builder
            .closure(
                Arc::new(Box::new(move |t: &Tensor| {
                    debug!("any_len_label_node: motif input={t:?}");
                    Ok(t.shallow_clone())
                })),
                None,
            );
            for _ in 0..params.motif_layers {
                builder = builder.linear_s(
                    num_features,
                    #[cfg(debug_assertions)]
                    "any_len_label_node 0".to_string(),
                )
                    .func(|t|Tensor::f_relu(t).map_err(|e|e.into()));
            }
            builder
                .end_seq()
            },

            Box::new(move |t: &Tensor| {
                debug!("any_len_label_node: t.size={:?}", t.size());
                let batch_size = t.size()[0];
                let motif_nets = t.size()[1];
                let dim_size = t.size()[2] as u32;
                let ranges = (0..dim_size).step_by(num_features as usize).map(|v|v as i64).collect::<Vec<_>>();
                debug!("any_len_label_node: ranges={ranges:?}");


                Ok(ranges.into_iter().map(|start| {

                    let t = if start as u32 + num_features > dim_size  {
                        debug!("any_len_label_node: start as u32 + features={:?} dim_size={dim_size} t.size={:?} padding to {:?}", start as u32 + num_features, t.size(), [batch_size, motif_nets, num_features as i64]);
                        pad_to(&t.f_narrow(2, start, (dim_size as i64 - start))?, &[batch_size, motif_nets, num_features as i64])?
                    } else {
                        debug!("any_len_label_node: start as u32 + features={:?} dim_size={dim_size} t.size={:?}", start as u32 + num_features, t.size());
                        t.f_narrow(2, start, num_features as i64)?
                    };

                    Ok(t)
                }).collect::<Result<Vec<Tensor>, ThoosaError>>()?)
            }),
            Box::new(move |res: Vec<Tensor>| {
                debug!("any_len_label_node: res={res:?}");

                let all = Tensor::f_concat(res.as_slice(), 1)?;
                debug!("any_len_label_node: all.size={:?}", all.size());

                let r = all.f_mean_dim(1, false, None)?;//.i((0, ..num_features as i64)); //
                debug!("any_len_label_node: r.size={:?}", r.size());

                Ok(r)

            })
        )
        .closure(
            Arc::new(Box::new(move |t: &Tensor| {
                debug!("any_len_label_node: final input={t:?}");
                Ok(t.shallow_clone())
            })),
            None,
        )
        // .conv_1d("any_len_label_node conv", 1000)
        .linear_s(
            num_features,
            #[cfg(debug_assertions)]
            "any_len_label_node 0".to_string(),
        )
        .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
        .linear_s(
            num_features,
            #[cfg(debug_assertions)]
            "any_len_label_node 1".to_string(),
        )
        .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
        .linear_s(
            1,
            #[cfg(debug_assertions)]
            "any_len_label_node 2".to_string(),
        )
        .func(|t|Tensor::f_sigmoid(t).map_err(|e|e.into()))
        .end_seq()
}


pub fn flat_label_node(params: &LabelNetParams) -> Node {
    debug!("flat_label_node: params={params:?}");
    let num_features = params.motif_nets * params.slices_len; //params.motif_nets * slices_len;
    let mut builder = SeqBuilder::new(0, None);

    builder
        .closure(
            Arc::new(Box::new(move |t: &Tensor| {
                debug!("flat_label_node before flatten t.size={:?}", t.size());
                //batch x motifs x dists
                let t_0 = t.flatten(1, 2);// - channels removed
                debug!("flat_label_node flatten t.size={:?} t_0.size={:?}",t.size(),t_0.size());

                Ok(t_0)
            })),
            Some(num_features),
        )
        .linear_s(
            num_features,
            #[cfg(debug_assertions)]
            "flat_label_node 0".to_string(),
        )
        .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
        .linear_s(
            num_features,
            #[cfg(debug_assertions)]
            "flat_label_node 1".to_string(),
        )
        .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
        .linear_s(
            1,
            #[cfg(debug_assertions)]
            "flat_label_node 2".to_string(),
        )
        .func(|t|Tensor::f_sigmoid(t).map_err(|e|e.into()))
        .end_seq()
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct FinalNetParams {
    in_features: u32,
    out_features: u32,
    num_layers: u32,
}

impl FinalNetParams {
    pub fn new(in_features: u32, out_features: u32, num_layers: u32) -> Self {
        Self {
            in_features,
            out_features,
            num_layers,
        }
    }

    pub fn from_file(path: &PathBuf) -> Result<Self, ThoosaError> {
        serde_json5::from_str::<FinalNetParams>(fs::read_to_string(&path)?.as_str())
            .map_err(|serde_err| ThoosaError::new(serde_err.to_string()))
    }
}

pub fn final_net(params: &FinalNetParams) -> Node {
    debug!("final_node: params={:?}", params);

    let mut builder = SeqBuilder::new(params.in_features, None);
    let mut net = builder
        .linear_s(
            2 * params.in_features,
            #[cfg(debug_assertions)]
            "final_net: first layer".to_string(),
        )
        .func(|t|Tensor::f_relu(t).map_err(|e|e.into()));

    for i in 0..params.num_layers {
        net = net
            .linear_s(
                params.in_features,
                #[cfg(debug_assertions)]
                format!("final_net: {i}th layer"),
            )
            .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
    }

    net
        // .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
        .linear_s(
            params.out_features,
            #[cfg(debug_assertions)]
            "final_net: last layer".to_string(),
        )
        // .func(|t|t.softmax(1, Float))
        .end_seq()
}

pub struct PlotVecf32 {
    id: String,
    data: Vec<f32>,
}

impl PlotVecf32 {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.into(),
            data: vec![],
        }
    }

    fn plot(&self) {
        plot_series(
            vec![(
                self.id.clone(),
                BLACK,
                self.data
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i as f32, *v as f32))
                    .collect::<Vec<(f32, f32)>>(),
            )],
            self.id.clone().as_str(),
            (2000, 1000),
            &format!("/tmp/plots/motif_train_{}.png", self.id),
        )
        .unwrap();
    }
}

impl TrainListener for PlotVecf32 {
    fn on_valid_cycle(&mut self, cycle: u32, loss: f32) -> ListenerResponse {
        self.data.push(loss);

        //TODO move into another listener
        if self.data.len() > 10 {
            let len_half = self.data.len() / 2;
            if self.data[0..len_half].iter().sum::<f32>()
                < self.data[len_half..].iter().sum::<f32>()
            {
                debug!(
                    "TrainListener for PlotVecf32::on_valid_cycle stopping self.data={:?}",
                    self.data
                );
                return ListenerResponse::Stop;
            }
        }

        ListenerResponse::Nope
    }
}

#[derive(Debug)]
pub struct TrainMotifsNetResult {
    pub loss: f32, //TODO pub
    channel: u32,
    motif_id: u32,
    vs: VarStore,
    params: MotifNetParams,
    run_config_values: TrainMotifsNetsRunConfig,
    find_motifs_run_id: RunId,
}

impl TrainMotifsNetResult {
    pub fn new(
        loss: f32,
        channel: u32,
        motif_id: u32,
        vs: VarStore,
        find_motifs_run_id: RunId,
        params: MotifNetParams,
        run_config_values: TrainMotifsNetsRunConfig,
    ) -> Self {
        Self {
            loss,
            channel,
            motif_id,
            vs,
            params,
            run_config_values,
            find_motifs_run_id,
        }
    }

}

#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct TrainMotifsNetsResult {
    id:RunId,
    pub train_motif_nets_results:Vec<TrainMotifsNetResult>
}

impl TrainMotifsNetsResult {
    pub fn new(id: RunId, train_motif_nets_results: Vec<TrainMotifsNetResult>) -> Self {
        Self { id, train_motif_nets_results }
    }
}

impl SavableResult for TrainMotifsNetsResult {
    fn value(&self) -> f32 {
        0.0
    }

    fn save(&mut self, store: &Arc<RwLock<Box<dyn Store>>>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<u32, ThoosaError> {
        for r in self.train_motif_nets_results.iter_mut() {
            r.save(store, ws)?;
        }

        Ok(0)
    }
}

//TODO remove
impl WithId for Box<TrainMotifsNetsResult> {
    type ID = RunId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}


impl WithId for TrainMotifsNetsResult {
    type ID = RunId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[async_trait]
impl Output for TrainMotifsNetsResult {
}


impl SavableResult for TrainMotifsNetResult {
    fn value(&self) -> f32 {
        self.loss
    }

    fn save(
        &mut self,
        store: &Arc<RwLock<Box<dyn Store>>>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
    ) -> Result<u32, ThoosaError> {
        //TODO save loss to choose model
        debug!("TrainMotifsNetsRunConfigResult::save: self={:?}", self);

        let run_config_id = store.write().save_run_config(
            Some(&[self.find_motifs_run_id]),
            &TrainMotifsNetsRunConfig::type_name(),
            "TrainMotifsNetsRunConfigResult2",
            "todo",
            self.run_config_values.to_properties_iter(None).collect(),
        )?;

        ws.read().save_motif_net(
            &self.channel,
            &run_config_id,
            &self.motif_id,
            &self.vs,
            &self.params,
        )?;

        Ok(run_config_id)
    }
}

pub async fn opt_train_2<P, R, F, N: Debug + Default>(
    opt_config: Arc<OptConfig>, //N node parameters
    train: F,
    run_config_source: Arc<OptRunConfigs<Vec<NamedVal<P>>, Vec<NamedRange<P>>, Vec<Hyperparameter<P>>, P> >,
    device: Device,
    descr: &str,
) -> Result<(VarStore, N, R, f32), ThoosaError>
where
    P: FieldName,
    R: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone,
    F: for<'a, 'b> AsyncFnMut(
            &VarStore,
            R,
            &'a mut (dyn TrainListener + 'b),
        ) -> Result<(f32, N), ThoosaError>
        + Clone,
{
    let const_vals = run_config_source
        .grid_run_config_source()
        .const_params()
        .clone();
    let min_loss = Arc::new(RwLock::new(f32::MAX));

    let min_var_store: Arc<RwLock<Option<(VarStore, N, R, f32)>>> = Arc::new(RwLock::new(None));

    let grid_vals_sets = run_config_source.grid_to_vals();

    info!("opt_train_2: running opt train {descr} grid_vals_sets={grid_vals_sets:?}");

    for grid_vals in grid_vals_sets.into_iter() {
        let const_vals = const_vals.clone();
        let grid_vals = grid_vals.clone();
        let mut train_clone = train.clone();

        let min_var_store_clone = Arc::clone(&min_var_store);
        let min_loss_moved = Arc::clone(&min_loss);

        let mut opt = AsyncTPEOptimizer::new(
            async move |opt_vals| {
                debug!("opt_train: opt_vals={opt_vals:?}  "); //descr={descr}

                let run_config: R = [const_vals.as_slice(), grid_vals.as_slice(), opt_vals]
                    .concat()
                    .into_iter()
                    .into();

                let vs = VarStore::new(device);

                let mut listener = PlotVecf32::new(
                    format!("_params_{:?}", "id" /*TODO run_config.get_id()*/).as_str(),
                ); //motif_id ?

                let (loss, node_params) = panic_to_err_async!(
                    train_clone
                        .async_call_mut((
                            &vs,
                            run_config.clone(),
                            &mut listener /*Arc::new(Mutex::new(Box::new(listener)))*/
                        ))
                        .await
                ).unwrap_or_else(|e| {
                    warn!("opt_train failed with {e:?} on {run_config:?} ");
                    (f32::MAX, Default::default())
                });
                listener.plot();

                debug!("opt_train: run_config_values={run_config:?} loss={loss} "); //descr={descr}
                let min_loss_val = *min_loss_moved.read();
                if loss < min_loss_val {
                    *min_loss_moved.write() = loss;
                    min_var_store_clone
                        .write()
                        .replace((vs, node_params, run_config, loss));
                }
                print!(".");
                loss
            },
            ConfigurationSpace::new(run_config_source.opt_params().clone()),
            opt_config.opt_max_evals,
            opt_config.opt_num_init,
        );
        info!("opt_train_2: wait on optimize ");
        let best_config = opt.optimize().await;

        debug!("best_config={:?}", best_config);
        trace!("done opt {descr}");
        info!("opt_train: done opt, descr={descr} best_config={best_config:?} min_loss={} min_var_store={min_var_store:?}", min_loss.read());
    }
    Arc::into_inner(min_var_store)
        .unwrap()
        .into_inner()
        .ok_or_else(|| ThoosaError::new("opt was not run"))
}

pub fn opt_train<P, R, F, N: Debug + Default>(
    opt_config: &OptConfig, //N node parameters
    mut train: F,
    run_config_source: &OptRunConfigs<
        Vec<NamedVal<P>>,
        Vec<NamedRange<P>>,
        Vec<Hyperparameter<P>>,
        P,
    >,
    device: Device,
    descr: &str,
) -> Result<(VarStore, N, R, f32), ThoosaError>
where
    P: FieldName,
    R: RunConfig + From<std::vec::IntoIter<NamedVal<P>>>,
    F: FnMut(&VarStore, &R, &mut dyn TrainListener) -> Result<(f32, N), ThoosaError>,
{
    let const_vals = run_config_source.grid_run_config_source().const_params();
    let mut min_loss = f32::MAX;
    let mut min_var_store: Option<(VarStore, N, R, f32)> = None;
    info!("opt_train: running opt train {descr} ");

    run_config_source.grid_to_vals().into_iter().for_each(|grid_vals| {

        let mut opt = TPEOptimizer::new(|opt_vals| {
            let run_config = [
                const_vals.as_slice(),
                grid_vals.as_slice(),
                opt_vals
            ].concat().into_iter().into();

            let vs = VarStore::new(device);

            let mut listener = PlotVecf32::new(format!("_params_{:?}", "id"/*TODO run_config.get_id()*/).as_str()); //motif_id ?
            let (loss, node_params) = match panic_to_err!(train(&vs, &run_config, &mut listener)) {
                Ok(v) => {
                    v
                }
                Err(e) => {
                    warn!("opt_train failed with {e:?} on {run_config:?}");
                    (f32::MAX, Default::default())
                }
            };
            listener.plot();

            debug!("opt_train: descr={descr} run_config_values={run_config:?} loss={loss}");
            if loss < min_loss {
                min_loss = loss;
                min_var_store.replace((vs, node_params, run_config, min_loss));
            }
            print!(".");
            loss
        },
            ConfigurationSpace::new(run_config_source.opt_params().clone()),
            opt_config.opt_max_evals,
            opt_config.opt_num_init,
        );

        let best_config = opt.optimize();

        trace!("done opt {descr}");
        info!("opt_train: done opt, descr={descr} best_config={best_config:?} min_loss={min_loss} min_var_store={min_var_store:?}");

        // min_var_store.ok_or_else(||ThoosaError::new("opt was not run"))
    });

    min_var_store.ok_or_else(|| ThoosaError::new("opt was not run"))
}

pub fn run_train_motif_net(
    motif_width: u32,
    run_config: &TrainMotifsNetsRunConfig,
    device: &Device,
    batch: &mut dyn DataBatch<Slice>,
    vs: &VarStore,
    listener: &mut dyn TrainListener,
) -> Result<(f32, MotifNetParams), ThoosaError> {
    let node_params = MotifNetParams::new(motif_width, run_config.num_layers, 100);
    debug!("run_train_motif_net: node_params={node_params:?}");

    let loss = train(
        &device,
        batch,
        &motif_node(&node_params)
            .build_tch_module::<Batch<Slice>, Batch<MotifProbability>>(&vs.root())
            .unwrap(),
        |pred: &Tensor, target: &Tensor| {
            let pred_y = pred.i((.., 0));

            debug!("run_train_motif_net: pred_y ={pred_y} target={target}");

            let loss = pred_y.binary_cross_entropy::<Tensor>(&target.totype(Float), None, Reduction::Mean);
            debug!("run_train_motif_net: loss={loss}");

            Ok(loss)
        },
        run_config.common_training_args.cycles as usize,
        AdamW::default()
            .build(&vs, run_config.common_training_args.lr)
            .unwrap(),
        |t| t,
        Some(listener),
    )?;

    Ok((loss, node_params))
}

pub async fn run_train_motif_net_2(
    motif_width: u32,
    run_config: &TrainMotifsNetsRunConfig,
    device: &Device,
    batch: &dyn AsyncDataBatch<Slice, MotifProbability>,
    vs: &VarStore,
    listener: &mut dyn TrainListener,
) -> Result<(f32, MotifNetParams), ThoosaError> {
    let node_params = MotifNetParams::new(motif_width, run_config.num_layers, run_config.step);
    debug!("run_train_motif_net: node_params={node_params:?}");

    let loss = train_2(
        &device,
        batch,
        motif_node_2(&node_params).build_tch_module::<Batch<Slice>, Batch<MotifProbability>>(&vs.root())?,
        |pred: &Tensor, target: &Tensor| {
            let pred_y = pred; //.i((.., 0));

            let loss = pred_y.binary_cross_entropy::<Tensor>(&target.totype(Float), None, Reduction::Mean).set_requires_grad(true);
            debug!("run_train_motif_net: pred_y ={pred_y} target={target} loss={loss}");

            Ok(loss)
        },
        run_config.common_training_args.cycles as usize,
        AdamW::default()
            .build(&vs, run_config.common_training_args.lr)
            .unwrap(),
        |t| t,
        Some(listener),
    ).await?;

    Ok((loss, node_params))
}


pub fn slices_in_all_resources<ResourceId>(slices: Vec<(ResourceId, Vec<Tensor>)>) -> Vec<Tensor> {
    slices
        .into_iter()
        .flat_map(|(_resource_key, slices)| slices.into_iter())
        .collect::<Vec<Tensor>>()
}

#[derive(Debug)]
pub struct TensorsDataBatch<D: Dims> {
    train: Vec<Vec<Tensor>>, //index - class
    val: Vec<Vec<Tensor>>,
    batch_size: u32,
    device: Device,
    _pd: PhantomData<D>,
}

impl<D: Dims> TensorsDataBatch<D> {
    pub fn new(
        classes_tensors: Vec<Vec<Tensor>>,
        batch_size: u32,
        train_valid_split: f32,
        device: Device,
    ) -> Self {
        let classes_len = classes_tensors.len();
        assert!(classes_len > 0);
        // assert!(batch_size % classes_len == 0);

        let (train, val) = split_train_val(classes_tensors, train_valid_split);

        Self {
            train,
            val,
            batch_size,
            device,
            _pd: PhantomData,
        }
    }

    fn choose_for_batch(
        class_tensors: &Vec<Vec<Tensor>>,
        batch_size: u32,
        device: &Device,
    ) -> (ThoBatch<D>, Tensor) {
        let mut rng = rand::rng();

        let per_class_len = batch_size / class_tensors.len() as u32; //TODO make balanced
        debug!(
            "TensorsDataBatch::choose_for_batch: class_tensors={} per_class_len={per_class_len}",
            class_tensors.len()
        );

        let mut x_y_vecs =
            class_tensors
                .iter()
                .enumerate()
                .fold(vec![], |mut x_y_acc, (class, tensors)| {
                    assert!(
                        tensors.len() > 0,
                        "no tensors for class={class} in ClassTensorsDataBatch"
                    );

                    tensors
                        .choose_multiple(&mut rng, per_class_len as usize)
                        .for_each(|t| {
                            x_y_acc.push((t.shallow_clone(), class as i64));
                        });

                    x_y_acc
                });

        x_y_vecs.shuffle(&mut rng);

        debug!(
            "TensorsDataBatch::choose_for_batch: x_y_vecs.1.len()={} batch_size={batch_size}",
            x_y_vecs.len()
        );
        assert!(x_y_vecs.len() > 0);

        //TODO throw if not enough tensors for batch?
        (x_y_vecs.len()..batch_size as usize).for_each(|i| {
            x_y_vecs.push((
                x_y_vecs[i - x_y_vecs.len()].0.shallow_clone(),
                x_y_vecs[i - x_y_vecs.len()].1,
            ));
        });

        let classes =
            Tensor::from_slice(x_y_vecs.iter().map(|e| e.1).collect::<Vec<_>>().as_slice())
                .to_device(*device);

        let tensor = Tensor::stack(
            x_y_vecs
                .into_iter()
                .map(|e| e.0)
                .collect::<Vec<_>>()
                .as_slice(),
            0,
        )
        .view((batch_size as i64, -1))
        .to_device(*device)
        .set_requires_grad(true);

        debug!(
            "TensorsDataBatch::choose_for_batch: tensor.size()={:?} tensor.device()={:?}",
            tensor.size(),
            tensor.device()
        );

        (tensor.into(), classes)
    }
}

impl<D: Dims> DataBatch<D> for TensorsDataBatch<D> {
    fn train_batch(&self) -> (ThoBatch<D>, Tensor) {
        Self::choose_for_batch(&self.train, self.batch_size, &self.device)
    }

    fn valid_batch(&self) -> (ThoBatch<D>, Tensor) {
        Self::choose_for_batch(&self.val, self.batch_size, &self.device)
    }
}

fn split_train_val<T>(
    classes_tensors: Vec<Vec<T>>,
    train_valid_split: f32,
) -> (Vec<Vec<T>>, Vec<Vec<T>>) {
    let classes_len = classes_tensors.len();
    let (train, val): (Vec<Vec<T>>, Vec<Vec<T>>) = classes_tensors.into_iter().fold(
        (
            Vec::with_capacity(classes_len),
            Vec::with_capacity(classes_len),
        ),
        |mut class_train_val, tensors| {
            let tensors_len = tensors.len();
            let split_i = (tensors_len as f32 * train_valid_split) as usize;

            let train_val = tensors.into_iter().enumerate().fold(
                (
                    Vec::with_capacity(split_i),
                    Vec::with_capacity(tensors_len - split_i),
                ),
                |mut train_val, (i, t)| {
                    if i < split_i {
                        train_val.0.push(t);
                    } else {
                        train_val.1.push(t);
                    }
                    train_val
                },
            );

            class_train_val.0.push(train_val.0); //train
            class_train_val.1.push(train_val.1); //validation
            class_train_val
        },
    );
    (train, val)
}




#[derive(Debug)]
pub struct LoadMotifNetOutput {
    channel: u32,
    nets: Vec<(MotifNetParams, ThoModuleWithStore<Batch<Slice>, MotifProbability>)>, //single motif's nets (of different types)
}

impl LoadMotifNetOutput {
    pub fn new(
        channel: u32,
        nets: Vec<(MotifNetParams, ThoModuleWithStore<Batch<Slice>, MotifProbability>)>,
    ) -> Self {
        Self { channel, nets }
    }
}

impl WithId for LoadMotifNetOutput {
    type ID = CachableId;

    fn id(&self) -> &Self::ID {
        todo!()
    }
}





#[derive(Args, Debug, Clone)]
pub struct TrainFinalNetArgs {
    #[arg(long)]
    pub train_labels_run_ids: Vec<RunId>,

    #[arg(long)]
    train_resource_group: String,

    #[arg(long)]
    valid_resource_group: String,

    #[command(flatten)]
    pub fixed_config: OptConfig,

    #[arg(long)]
    pub run_config_cnt: String, //TrainLabelNetsRunConfig
}

impl TrainFinalNetArgs {
    pub fn resource_config(&self) -> NetResources<Logits> {
        NetResources::new(
            Logits::new(self.train_resource_group.clone()),
            Logits::new(self.valid_resource_group.clone()),
        )
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
#[run_config_opt]
pub struct TrainFinalNetRunConfig {
    #[in_map]
    pub id: Option<RunId>,

    #[in_map]
    #[run_config]
    pub common_training_args: CommonTrainingRunConfig,

    #[in_map]
    #[run_config]
    pub net: FinalNetArgs,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
pub struct FinalNetArgs {
    #[in_map]
    #[run_config]
    pub num_layers: u32,
}

impl FinalNetArgs {
    #[cfg(test)]
    pub fn new(num_layers: u32) -> Self {
        Self { num_layers }
    }
}

impl TrainFinalNetRunConfig {
    pub fn new(
        id: Option<RunId>,
        common_training_args: CommonTrainingRunConfig,
        net: FinalNetArgs,
    ) -> Self {
        Self {
            id,
            common_training_args,
            net,
        }
    }
}

#[derive(Debug, Args, Clone)]
pub struct InferArgs {
    #[arg(long)]
    pub final_run_id: RunId,

    #[arg(long, default_value_t = 1000)]
    pub input_slice_step: usize,

    #[arg(long, default_value = "../../data_local/test_eegs")]
    pub input_dir: String,

    #[arg(long, default_value = "submission.csv")]
    pub output_dir: String,
}

impl InferArgs {
    pub fn input_data(&self) -> &str {
        &self.input_dir
    }
    pub fn output(&self) -> &str {
        &self.output_dir
    }

    pub fn new(
        final_run_id: RunId,
        input_data: String,
        output: String,
        input_slice_step: usize,
    ) -> Self {
        Self {
            final_run_id,
            input_slice_step,
            input_dir: input_data,
            output_dir: output,
        }
    }
}

pub fn is_nonzero(t_1: &Tensor) -> Result<bool, ThoosaError> {
    Ok(t_1
        .flatten(0, (t_1.size().len() - 1) as i64)
        .iter::<f64>()?
        .find(|v| v > &0.)
        .is_some())
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum LabelNetTypeArgs {
    MaxPadding(MaxPaddingNetArgs),

    FixedNumSlices(FixedNumSlicesNetArgs),
}

//TODO remove
impl Default for LabelNetTypeArgs {
    fn default() -> LabelNetTypeArgs {
        LabelNetTypeArgs::MaxPadding(MaxPaddingNetArgs { num_layers: 0 })
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MaxPaddingNetArgs {
    #[in_map]
    #[run_config]
    pub num_layers: u32,
}

impl MaxPaddingNetArgs {
    pub fn new(num_layers: u32) -> Self {
        Self { num_layers }
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct FixedNumSlicesNetArgs {
    #[in_map]
    #[run_config]
    pub num_layers: u32,

    #[in_map]
    #[run_config]
    pub slices_len: u32,
}

impl FixedNumSlicesNetArgs {
    pub fn new(num_layers: u32, slices_len: u32) -> Self {
        Self {
            num_layers,
            slices_len,
        }
    }
}

///sums and multiplies inputs by k
pub fn dummy_label_net<DI: Dims + Send + 'static, DO: Dims + Send + 'static>(
    num_features: u32,
    k: f64,
    device: Device,
    descr: &str,
) -> Result<Box<dyn ThoModule<DI, DO>>, ThoosaError> {
    let vs = VarStore::new(device);

    SeqBuilder::new(num_features, None)
        .closure(
            Arc::new(Box::new(move |t: &Tensor| {
                debug!(
                        "test_label_net: merged final in flatten t.size()={:?}",
                        t.size()
                    );

                let t_0 = t.sum_dim_intlist(&[1, 2][..], false, None);
                debug!(
                        "test_label_net: merged final out flatten t_0.size()={:?}",
                        t_0.size()
                    );

                Ok(t_0)
            })),
            None,
        )
        .linear_s_with_conf(
            1,
            LinearConfig {
                ws_init: init::Init::Const(k), //conflicts with ctor's init
                bs_init: None,
                bias: false,
            },
            #[cfg(debug_assertions)]
            descr,
        )
        .end_seq()
        .build_tch_module(&vs.root())
}


#[cfg(test)]
mod train_tests {
    use crate::builder::{Node, SeqBuilder};
    use crate::dims::{
        AnyDims, Batch, Dims, LabelProbability, MotifProbability, MotifsNetsSelectionDims, Slice,
        ThoBatch,
    };
    use crate::find_motifs_run_config::FindMotifsRunConfig;
    use crate::loader::{LoadRunner, OfMemSize, SyncWrapper};
    use crate::my_tch::ThoModule;
    use crate::simp::DummyDs;
    use crate::tho_tensor::ThoTensor;
    use crate::train_nets::{dummy_label_net, motif_node, opt_train, BatchTransform, ClassTensorsLoaderData, CommonTrainingRunConfig, CommonTrainingRunConfigRangeOrValue, FinalNetArgs, FinalNetParams, FindMotifsRunConfigRangeOrValue, FixedNumSlicesNetArgs, FixedNumSlicesNetParams, InputFilterRangeOrValue, LabelNetTypeArgs, LabelNetTypeArgsRangeOrValue, LabelNetTypeParams, LoadMotifNetOutput, LoadingTensorsDataBatch, MaxPaddingNetArgs, MaxPaddingNetArgsRangeOrValue, MaxPaddingNetParams, NetsPadding, OptConfig, PlotVecf32, RangeOrValue, TrainFinalNetRunConfig, TrainLabelNetsRunConfig, TrainLabelNetsRunConfigRangeOrValue};
    use crate::train_nets::{
        InferArgs, LabelNetParams, MotifNetParams, TensorsDataBatch, ThoModuleWithStore,
    };
    use crate::transforms::filters::BandPassFilterParamsRangeOrValue;
    use crate::utils::{
        cuda_mem, from, optimizer_mem_size, set_requires_grad_in_subpath, train,
        vs_mem_size, SplittableDataBatch, TrainListener,
    };
    use crate::utils::{
        lin_t, pad_to, reset_print_options, run_split_on_oom, t_0_0, tensor_mem_size, DataBatch,
        Reducer, SplitQueue, Splittable, SplittableTensor,
    };
    use crate::ws::mock_ws_simp;
    use crate::ws::{FsWorkspace, Workspace, WsMockBuilder};
    use crate::{train_nets, utils};
    use async_trait::async_trait;
    use futures::executor::block_on;
    use futures_util::FutureExt;
    use log::LevelFilter::{Debug, Info, Trace};
    use log::{debug, error, info, set_max_level, trace};
    use mockall::automock;
    use mockall_double::double;
    use parking_lot::{Mutex, RwLock};
    use rand::Rng;
    use rayon::iter::IntoParallelRefMutIterator;
    use std::collections::HashMap;
    use std::ops::Deref;
    use std::path::{Path, PathBuf};
    use std::pin::Pin;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::{fs, mem};
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::nn::{init, AdamW, BatchNormConfig, LinearConfig, Module, OptimizerConfig, VarStore};
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::{Float, Half};
    use tch::{nn, Device, Tensor};
    use tch::{IndexOp, Kind, Reduction};
    use thoosa_properties_map::ToHashmap;
    use thoosa_run_config_args::OptRunConfigSource;
    use thoosa_shared::applicables_runner::{CachableId, Input};
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::{OptRunConfigs, RunId};
    use thoosa_shared::test_utils::set_panic_hook;
    use thoosa_shared::val::Val;
    use thoosa_shared::{batch_windows, cdt_for_dir, clc_num_batches, pulse_in_rand, sin_v, slices_len, CopyToDevice, ThoosaError};
    use thoosa_store::{mock_store_with_find_motifs_run_config, MockStore, SqlStore};
    use torch_sys::at_autocast_clear_cache;

    #[ctor::ctor]
    fn test_init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_class_tensors_data_batch() -> Result<(), ThoosaError> {
        let mut batch: TensorsDataBatch<AnyDims> = TensorsDataBatch::new(
            vec![
                vec![
                    Tensor::from_slice(&[0., 0., 0.]),
                    Tensor::from_slice(&[0., 0., 1.]),
                    Tensor::from_slice(&[0., 0., 2.]),
                    Tensor::from_slice(&[0., 1., 0.]),
                    Tensor::from_slice(&[0., 1., 1.]),
                    Tensor::from_slice(&[0., 1., 2.]),
                    Tensor::from_slice(&[0., 2., 0.]),
                    Tensor::from_slice(&[0., 2., 1.]),
                    Tensor::from_slice(&[0., 2., 2.]),
                    Tensor::from_slice(&[0., 3., 0.]),
                    Tensor::from_slice(&[0., 3., 1.]),
                    Tensor::from_slice(&[0., 3., 2.]),
                ],
                vec![
                    Tensor::from_slice(&[1., 0., 0.]),
                    Tensor::from_slice(&[1., 0., 1.]),
                    Tensor::from_slice(&[1., 0., 2.]),
                    Tensor::from_slice(&[1., 1., 0.]),
                    Tensor::from_slice(&[1., 1., 1.]),
                    Tensor::from_slice(&[1., 1., 2.]),
                    Tensor::from_slice(&[1., 2., 0.]),
                    Tensor::from_slice(&[1., 2., 1.]),
                    Tensor::from_slice(&[1., 2., 2.]),
                    Tensor::from_slice(&[1., 3., 0.]),
                    Tensor::from_slice(&[1., 3., 1.]),
                    Tensor::from_slice(&[1., 3., 2.]),
                ],
            ],
            10,
            0.5,
            Cpu,
        );

        let train_batch = batch.train_batch();

        debug!(
            "train_batch: x={:?}",
            from::<Vec<f32>>(&train_batch.0.as_ref().i((.., 0)))
        );
        debug!("train_batch: y={:?}", from::<Vec<f32>>(&train_batch.1));

        assert_eq!(
            from::<Vec<f32>>(&train_batch.0.as_ref().i((.., 0))),
            from::<Vec<f32>>(&train_batch.1)
        );

        // debug!("train_batch: x={:?}, y={:?}", train_batch.0.size(), train_batch.1.size());

        assert_eq!(10, train_batch.0.size()[0]);
        assert_eq!(5, train_batch.1.iter::<i64>()?.filter(|v| *v == 1).count());
        assert_eq!(5, train_batch.1.iter::<i64>()?.filter(|v| *v == 0).count());

        let valid_batch = batch.valid_batch();
        assert_eq!(10, valid_batch.0.size()[0]);
        assert_eq!(5, valid_batch.1.iter::<i64>()?.filter(|v| *v == 1).count());
        assert_eq!(5, valid_batch.1.iter::<i64>()?.filter(|v| *v == 0).count());

        Ok(())
    }


    #[test]
    fn test_test_net() -> Result<(), ThoosaError> {
        let t_0 = lin_test_tensor(1000, 5);
        debug!("t_0.size={:?}", t_0.size());

        let t: ThoTensor<AnyDims, CachableId> = t_0.unfold(1, 100, 10).into();

        debug!("t.size={:?}", t.size());

        //sums and multiplies by k
        let r_0: ThoTensor<AnyDims, CachableId> =
            utils::test_net(100, 1., Cpu, "test_net")?.tho_forward(&t, true)?;
        debug!("r_0.size={:?}", r_0.size());

        assert_eq!(Tensor::from_slice(&[4950.]), r_0.as_ref().i((0, 0, ..)));
        assert_eq!(
            Tensor::from_slice(&[9900.]),
            utils::test_net::<AnyDims, AnyDims>(100, 2., Cpu, "test_net")?
                .tho_forward(&t, true)?
                .as_ref()
                .i((0, 0, ..))
        );

        Ok(())
    }
    

    fn label_test_nets(
        num_features: u32,
        k: f64,
        params: LabelNetTypeParams,
        device: Device,
    ) -> Vec<(
        RunId,
        Box<dyn Fn() -> LabelNetParams + Sync + Send>,
        Box<
            dyn Fn() -> ThoModuleWithStore<MotifsNetsSelectionDims, Batch<LabelProbability>>
                + Sync
                + Send,
        >,
    )> {
        vec![(
            0u32,
            Box::new(move || LabelNetParams::new(2, 3, 3, 100, params.clone())),
            Box::new(move || {
                ThoModuleWithStore::new("".into(),
                                        dummy_label_net(num_features, k, device, "test_label_net").unwrap(),
                )
            }),
        )]
    }

    //5 ch
    fn lin_test_tensor(input_n: i64, channels: u32) -> Tensor {
        Tensor::vstack(
            (1..=channels)
                .map(|c| {
                    Tensor::from_slice(
                        (0..input_n)
                            .map(|i| i as f32 * c as f32)
                            .collect::<Vec<_>>()
                            .as_slice(),
                    )
                })
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    #[tokio::test]
    async fn test_save_load_label_model_fixed_num_slices() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_dir={test_dir:?}");
        let ws = Arc::new(RwLock::new(FsWorkspace::new(test_dir, 3, 1000, 32000)?));

        let params = LabelNetParams::new(
            3,
            19,
            2,
            100,
            train_nets::LabelNetTypeParams::FixedNumSlices(FixedNumSlicesNetParams::new()),
        );

        let mut vs_0 = VarStore::new(Cpu);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> =
            LabelNetParams::label_net(&params).build_tch_module(&vs_0.root())?;
        vs_0.freeze();

        let _ = net.tho_forward(&Tensor::rand([11, 3 * 19, 100], FLOAT_CPU).into(), false);

        let model_path = ws
            .write()
            .save_label_model(1, &"test_label", &vs_0, &params)?;

        debug!(
            "model_path={model_path:?} vs_0.variables().len()={}",
            vs_0.variables().len()
        );

        let params_for_builder = params.clone();
        let loaded_module: ThoModuleWithStore<AnyDims, AnyDims> = ThoModuleWithStore::load(
            "".into(),
           move |path: &nn::Path| LabelNetParams::label_net(&params_for_builder).build_tch_module(path),
           thoosa_shared::replace_extension(&model_path, "safetensors").1.into(),
           Cpu,
        )?;

        assert!(vs_0.variables().len() > 0);
        assert_eq!(vs_0.variables(), loaded_module.var_store().variables());


        let cuda_module = loaded_module.copy_to_device(1).await?;
        let _ = cuda_module.tho_forward(
            &Tensor::rand([11, 3 * 19, 100], FLOAT_CUDA).into(),
            false,
        );


        Ok(())
    }


    #[test]
    fn test_save_load_label_model_max_padding() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_dir={test_dir:?}");
        let channels = 3;
        let max_motif_nodes = 19;
        let slices_len = 23;

        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            test_dir,
            channels as u32,
            1000,
            32000,
        )?));
        let params = LabelNetParams::new(
            channels,
            max_motif_nodes,
            3,
            slices_len,
            LabelNetTypeParams::MaxPadding(MaxPaddingNetParams::new()),
        );

        let mut vs_0 = VarStore::new(Cpu);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> =
            LabelNetParams::label_net(&params).build_tch_module(&vs_0.root())?;
        vs_0.freeze();

        let batch = 11i64;
        let _ = net.tho_forward(
            &Tensor::rand(
                [
                    batch,
                    channels as i64,
                    max_motif_nodes as i64,
                    slices_len as i64,
                ],
                FLOAT_CPU,
            )
            .into(),
            false,
        );

        let model_path = ws
            .write()
            .save_label_model(1, &"test_label", &vs_0, &params)?;

        debug!(
            "model_path={model_path:?} vs_0.variables().len()={}",
            vs_0.variables().len()
        );

        let params_for_builder = params.clone();
        let loaded_module: ThoModuleWithStore<AnyDims, AnyDims> = ThoModuleWithStore::load("".into(),
                                                                                           move |path: &nn::Path| LabelNetParams::label_net(&params_for_builder).build_tch_module(path),
                                                                                           thoosa_shared::replace_extension(&model_path, "safetensors")
                                                                                               .1
                                                                                               .into(),
                                                                                           Cpu,
        )?;

        assert!(vs_0.variables().len() > 0);
        assert_eq!(vs_0.variables(), loaded_module.var_store().variables());

        Ok(())
    }



    fn dummy_0(data: &mut [f64; 11]) -> (Tensor, f32) {
        (Tensor::from_slice(data).to_kind(Float), 0.0)
    }

    fn dummy_1(data: &mut [f64; 11]) -> (Tensor, f32) {
        (Tensor::from_slice(data).to_kind(Float) + 1.0, 1.0)
    }

    fn asc_t<const S: usize>(mut data: [f32; S]) -> Tensor {
        Tensor::from_slice(
            (0..S)
                .map(|i| data[i] + i as f32 * 2.)
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .to_kind(Float)
    }

    fn desc_t<const S: usize>(mut data: [f32; S]) -> Tensor {
        Tensor::from_slice(
            (0..S)
                .rev()
                .map(|i| data[i] + i as f32 * 2.)
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .to_kind(Float)
    }



    fn dummy_labels_net_input<'a>(
        labels: &[bool],
        channels: usize,
        max_motif_nets: usize,
        slices_len: usize,
        tr: Option<Box<dyn Fn(&Tensor) -> Tensor + 'a>>,
    ) -> Tensor {
        //batch size x channels x max motif nets x slices_len
        let mut labels_iter = labels.iter();
        let t = Tensor::stack(
            labels
                .iter()
                .map(|label| {
                    Tensor::stack(
                        (0..channels)
                            .map(|mb| {
                                Tensor::stack(
                                    (0..max_motif_nets)
                                        .map(|mmn| {
                                            if *label {
                                                motifs_1(slices_len)
                                            } else {
                                                motifs_0(slices_len)
                                            }
                                        })
                                        .collect::<Vec<_>>()
                                        .as_slice(),
                                    0,
                                )
                            })
                            .collect::<Vec<_>>()
                            .as_slice(),
                        0,
                    )
                })
                .map(|t| {
                    debug!("dummy_labels_net_input: tr.size={:?}", t.size());
                    tr.as_ref().map(|tr| tr(&t)).unwrap_or(t)
                })
                .collect::<Vec<_>>()
                .as_slice(),
            0,
        );
        debug!("dummy_labels_net_input: t.size={:?}", t.size());

        t
    }

    fn dummy_motif_net_input<'a>(
        labels: &[bool],
        channels: usize,
        n: usize,
        tr: Option<Box<dyn Fn(&Tensor) -> Tensor + 'a>>,
    ) -> Tensor {
        //batch size x channels x max motif nets x slices_len
        let batch = Tensor::stack(
            labels
                .iter()
                .map(|label| {
                    Tensor::stack(
                        (0..channels)
                            .map(|channel| if *label { motifs_1(n) } else { motifs_0(n) })
                            .collect::<Vec<_>>()
                            .as_slice(),
                        0,
                    )
                })
                .map(|t| {
                    debug!("dummy_motif_net_input: tr.size={:?}", t.size());
                    tr.as_ref().map(|tr| tr(&t)).unwrap_or(t)
                })
                .collect::<Vec<_>>()
                .as_slice(),
            0,
        );
        debug!("dummy_motif_net_input: t.size={:?}", batch.size());

        batch
    }

    fn motifs_1(n: usize) -> Tensor {
        // Tensor::rand(n as i64, FLOAT_CPU).multiply_scalar(0.1).f_add_scalar(0.8).unwrap()
        Tensor::ones(n as i64, FLOAT_CPU)
    }

    fn motifs_0(n: usize) -> Tensor {
        // Tensor::rand(n as i64, FLOAT_CPU).multiply_scalar(0.01)
        Tensor::zeros(n as i64, FLOAT_CPU)
    }

    const fn build_labels<const BATCH_SIZE: usize>() -> [bool; BATCH_SIZE] {
        let mut arr = [false; BATCH_SIZE];
        let mut i = 0;
        while i < BATCH_SIZE {
            arr[i] = i % 2 == 0;
            i += 1;
        }
        arr
    }

    #[test]
    fn test_train_final() -> Result<(), ThoosaError> {
        set_max_level(Info);

        struct DummyLabelBatch {}

        impl DummyLabelBatch {
            pub fn new() -> Self {
                Self {}
            }
        }

        const BATCH_SIZE: usize = 10;
        const LABEL_NETS_SIZE: usize = 9;
        const LABELS_SIZE: usize = 6;

        impl<D: Dims> DataBatch<D> for DummyLabelBatch {
            fn train_batch(&self) -> (ThoBatch<D>, Tensor) {
                let train_batch = dummy_final_net_input::<LABEL_NETS_SIZE, Batch<D>>(BATCH_SIZE);
                trace!("train_batch=\n{} \n{}", train_batch.0, train_batch.1);
                train_batch
            }

            fn valid_batch(&self) -> (ThoBatch<D>, Tensor) {
                let valid_batch = dummy_final_net_input::<LABEL_NETS_SIZE, Batch<D>>(BATCH_SIZE);
                trace!("valid_batch={} \n{}", valid_batch.0, valid_batch.1);
                valid_batch
            }
        }

        let (loss, final_params) = super::train_final_net(
            &TrainFinalNetRunConfig::new(
                Some(1),
                CommonTrainingRunConfig::new(10, 10, 0.1),
                FinalNetArgs::new(3),
            ),
            LABEL_NETS_SIZE as u32,
            LABELS_SIZE as u32,
            &mut DummyLabelBatch::new(),
            &VarStore::new(Cpu),
            &Cpu,
        )?;

        assert!(loss < 0.5);
        debug!("test_train_final: loss={loss:?} final_params={final_params:?}");
        // fs::remove_dir_all(dir)?;

        Ok(())
    }

    fn dummy_final_net_input<const LABELS_SIZE: usize, D: Dims>(
        batch_size: usize,
    ) -> (ThoTensor<D, CachableId>, Tensor) {
        let mut rng = rand::rng();
        let data = (0..batch_size)
            .map(|label| {
                let label = rng.random_range(0..LABELS_SIZE);

                let x = random_tensor_with_dominant(LABELS_SIZE as i64, label as i64, 0.99, Cpu);
                let mut y = [0.; LABELS_SIZE];
                y[label] = 1.0;
                let y = Tensor::from_slice(y.as_slice());

                (x, y)
            })
            .collect::<Vec<_>>();

        (
            Tensor::stack(
                data.iter()
                    .map(|d| d.0.shallow_clone())
                    .collect::<Vec<_>>()
                    .as_slice(),
                0,
            )
            .into(),
            Tensor::stack(
                data.iter()
                    .map(|d| d.1.shallow_clone())
                    .collect::<Vec<_>>()
                    .as_slice(),
                0,
            ),
        )
    }

    /// Generate an n-dimensional tensor whose elements sum to 1,
    /// with index m taking `fraction` of the total (e.g., 0.8),
    /// and the remaining (1 - fraction) spread among the other n-1 entries.
    ///
    /// # Arguments
    /// * `n`         - Size of the tensor (number of elements).
    /// * `m`         - The index that will hold the dominant fraction.
    /// * `fraction`  - The fraction of the total sum to assign to the m-th element (e.g., 0.8).
    /// * `device`    - tch::Device (e.g., Device::Cpu).
    ///
    /// # Panics
    /// If `m >= n` or `fraction <= 0.0` or `fraction >= 1.0`.
    pub fn random_tensor_with_dominant(n: i64, m: i64, fraction: f64, device: Device) -> Tensor {
        assert!(m < n, "m must be < n");
        assert!(
            fraction > 0.0 && fraction < 1.0,
            "fraction must be in (0,1)"
        );

        let remainder = 1.0 - fraction;

        // Generate n-1 random values in [0,1]
        let other_values = Tensor::rand(&[n - 1], (Kind::Float, device));
        // Normalize them so they sum up to `remainder`
        let sum_others = other_values.sum(Kind::Float);
        let scaled_others = &other_values / &sum_others * remainder;

        // Create an output tensor of zeros
        let mut out = Tensor::zeros(&[n], (Kind::Float, device));

        // Fill the m-th entry with the dominant fraction
        let _ = out.get(m).fill_(fraction);

        // Prepare indices for all entries except m
        let indices: Vec<i64> = (0..n).filter(|&idx| idx != m).collect();

        // Put the scaled_others values into out at positions other than m
        let _ = out.index_put_(
            &[Some(Tensor::from_slice(indices.as_slice()))],
            &scaled_others,
            false,
        );

        out
    }

    //6 corrs 4 motif models
    //TODO remove?
    fn save_test_models(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        base_run_id: RunId,
        channels: u32,
        max_motif_batches: u32,
        max_batch_size: u32,
        motif_node_params_run_id: RunId,
        window: u32,
    ) -> Result<(), ThoosaError> {
        // fs::create_dir_all(models_path)?;

        // fs::write(models_path.join("motif_model.json"), serde_json5::to_string(&MotifNodeParams::new(window))?)?;

        save_test_label_model(
            &ws,
            base_run_id + 1,
            "seizure",
            3,
            4,
            max_motif_batches,
            max_batch_size,
        )?;
        // fs::write(models_path.join("motif_model.json"), serde_json5::to_string(&MotifNodeParams::new(window))?)?;

        save_test_motif_model(ws, "0", 0, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "0", 1, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "0", 2, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "0", 3, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "1", 10, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "1", 11, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "2", 20, window, motif_node_params_run_id)?;

        save_test_label_model(
            ws,
            base_run_id + 1,
            "lpd",
            1,
            1,
            max_motif_batches,
            max_batch_size,
        )?;

        save_test_motif_model(ws, "0", 0, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "1", 10, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "2", 20, window, motif_node_params_run_id)?;

        save_test_label_model(
            ws,
            base_run_id + 1,
            "gpd",
            1,
            1,
            max_motif_batches,
            max_batch_size,
        )?;

        save_test_motif_model(ws, "0", 0, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "1", 10, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "2", 20, window, motif_node_params_run_id)?;

        save_test_label_model(
            ws,
            base_run_id + 1,
            "lrda",
            1,
            1,
            max_motif_batches,
            max_batch_size,
        )?;

        save_test_motif_model(ws, "0", 0, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "1", 10, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "2", 20, window, motif_node_params_run_id)?;

        save_test_label_model(
            ws,
            base_run_id + 1,
            "grda",
            1,
            1,
            max_motif_batches,
            max_batch_size,
        )?;

        save_test_motif_model(ws, "0", 0, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "1", 10, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "2", 20, window, motif_node_params_run_id)?;

        save_test_label_model(
            ws,
            base_run_id + 1,
            "other",
            1,
            1,
            max_motif_batches,
            max_batch_size,
        )?;

        save_test_motif_model(ws, "0", 0, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "1", 10, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "2", 20, window, motif_node_params_run_id)?;
        Ok(())
    }

    pub fn save_test_tensor(input_path: &PathBuf, name: &str) {
        fs::create_dir_all(input_path).unwrap();
        fs::write(&input_path.join(name), "lalalal").unwrap();
    }

    fn save_test_final_model(models_path: &PathBuf, features: u32) -> Result<(), ThoosaError> {
        fs::create_dir_all(models_path.clone())?;

        let mut vs_0 = VarStore::new(Cpu);
        let _: Box<dyn ThoModule<AnyDims, AnyDims>> =
            train_nets::final_net(&FinalNetParams::new(features, 0, 1))
                .build_tch_module(&vs_0.root())?;

        vs_0.freeze();
        debug!(
            "save_test_final_model: features={} vs_0.variables()={:?}",
            features,
            vs_0.variables()
        );
        vs_0.save(models_path.join("final.safetensors"))?;
        Ok(())
    }

    fn save_test_label_model(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        run_id: RunId,
        label: &str,
        channels: u32,
        max_motif_nets: u32,
        max_motif_batches: u32,
        motif_batch_size: u32,
    ) -> Result<(), ThoosaError> {
        let label_node_params = LabelNetParams::new(
            channels,
            max_motif_nets,
            0,
            0,
            LabelNetTypeParams::MaxPadding(MaxPaddingNetParams::new()),
        ); //TODO slices_len, motif_layers

        let mut vs_0 = VarStore::new(Cpu);
        let _: Box<dyn ThoModule<AnyDims, AnyDims>> =
            LabelNetParams::label_net(&label_node_params).build_tch_module(&vs_0.root())?;
        vs_0.freeze();
        debug!(
            "save_test_corr_model: channels={}, max_motif_nets={}, max_motif_batches={}, motif_batch_size={} vs_0.variables()={:?}",
            channels,
            max_motif_nets,
            max_motif_batches,
            motif_batch_size,
            vs_0.variables()
        );

        ws.write()
            .save_label_model(run_id, &label, &vs_0, &label_node_params)?;
        Ok(())
    }

    fn save_test_motif_model(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        channel: &str,
        motif_id: u32,
        window: u32,
        motif_node_params_run_id: RunId,
    ) -> Result<(), ThoosaError> {
        let mut vs_0 = VarStore::new(Cpu);
        let _test_motif_model: Box<dyn ThoModule<AnyDims, AnyDims>> =
            train_nets::motif_node(&MotifNetParams::new(window, 1, 100))
                .build_tch_module(&vs_0.root())?;

        let test_motif_model: Box<dyn ThoModule<AnyDims, AnyDims>> = SeqBuilder::new(100, None)
            .linear_s_with_conf(
                1,
                LinearConfig {
                    ws_init: init::Init::Const(1.0),
                    bs_init: None,
                    bias: false,
                },
                #[cfg(debug_assertions)]
                "test_motif_model",
            )
            .end_seq()
            .build_tch_module(&vs_0.root())?;

        vs_0.freeze();
        debug!(
            "save_test_motif_model: channel={channel} vs_0.variables()={:?}",
            vs_0.variables()
        );
        ws.write().save_motif_net(
            &channel,
            &motif_node_params_run_id,
            &motif_id,
            &vs_0,
            &MotifNetParams::new(100, 1, 100),
        )?;
        Ok(())
    }

    #[test]
    fn test_is_nonzero() -> Result<(), ThoosaError> {
        assert!(super::is_nonzero(&Tensor::from_slice(&[0., 1., 0.]))?);
        assert!(super::is_nonzero(&Tensor::stack(
            &[
                Tensor::from_slice(&[0., 1., 0.]),
                Tensor::from_slice(&[0., 0., 0.])
            ],
            0
        ))?);
        assert!(!super::is_nonzero(&Tensor::from_slice(&[0., 0., 0.]))?);

        Ok(())
    }

    #[test]
    fn test_loading_tensors_data_batch() {
        let loader = Arc::new(RwLock::new(LoadRunner::new(
            move |key: &Resource| {
                SyncWrapper::wrap_tensor({
                    match key.file_stem() {
                        "aaa" => {
                            let t_0_0 = Tensor::from_slice(&[
                                0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                            ]); //t 0 channel 0
                            let t_0_1 = Tensor::from_slice(&[
                                0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19,
                            ]); //t 0 channel 1
                            let t_0 = Tensor::stack(&[t_0_0, t_0_1], 0);
                            t_0
                        }
                        "bbb" => {
                            let t_1_0 = Tensor::from_slice(&[1.0, 1.01, 1.02, 1.03, 1.04, 1.05]); //t 1 channel 0
                            let t_1_1 = Tensor::from_slice(&[1.1, 1.11, 1.12, 1.13, 1.14, 1.15]); //t 1 channel 1
                            let t_1 = Tensor::stack(&[t_1_0, t_1_1], 0);
                            t_1
                        }
                        "ccc" => {
                            let t_2_0 = Tensor::from_slice(&[
                                2.0, 2.01, 2.02, 2.03, 2.04, 2.05, 2.06, 2.07,
                            ]); //t 2 channel 0
                            let t_2_1 = Tensor::from_slice(&[
                                2.1, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17,
                            ]); //t 2 channel 1
                            let t_2 = Tensor::stack(&[t_2_0, t_2_1], 0);
                            t_2
                        }
                        _ => {
                            panic!("unexpected key={key:?}")
                        }
                    }
                })
            },
            1,
            10,
            2,
        )));
        /*
        TODO
                let mut data_batch:LoadingTensorsDataBatch<AnyDims, AnyDims> = LoadingTensorsDataBatch::new(
                    Rc::new(vec![
                        (0, vec![
                            Resource::new(0, 1, "aaa".to_string(), 10),
                            Resource::new(1, 1, "bbb".to_string(), 6),
                        ]),
                        (1, vec![
                            Resource::new(2, 1, "ccc".to_string(), 8),
                        ])
                    ]),
                    Rc::new(vec![]),
                    3,
                    loader,
                    Cpu,
                    Some(Box::new(|t:ThoTensor<AnyDims, CachableId>|{
                        t.as_ref().pad([0,10-t.size()[1]], "constant", 0.).into()
                    }))
                );

                let batch = data_batch.train_batch();


                // debug!("data={}", batch.0);
                // debug!("labels={:?}", batch.1);

                assert_eq!(
                    [3_i64,2,10],
                    batch.0.size()[..]
                );*/
    }

    #[test]
    fn test_batch_transform() {
        let n = 1000;
        let t_0 = Tensor::from_slice(pulse_in_rand(n, &[], 0.05, 1., 0.1).as_slice());
        let t_1 = Tensor::from_slice(sin_v(n, 22.).as_slice());

        let batch = Rc::new(Box::new(DummyDs::new(
            t_0.shallow_clone(),
            t_1.shallow_clone(),
        )));

        let mut some_v = 0;

        let batch = BatchTransform::new(
            Rc::clone(&batch),
            Box::new(|t| {
                some_v += 1;
                t
            }),
        );

        //TODO
    }

    #[test]
    fn test_0() {
        struct A;

        fn func(a: &A) {}

        let a_rc = Rc::new(A);

        func(&a_rc);

        let t = Tensor::rand(&[0], FLOAT_CPU);
        // t.interpolate

        //     struct Outer {
        //         inn:RwLock<Inner>
        //     }
        //
        //     #[async_trait]
        //     trait AsyncOuter {
        //         async fn run_inner(&self);
        //     }
        //
        //     #[async_trait::async_trait(?Send)]
        //     impl AsyncOuter for Outer {
        //         async fn run_inner(&self) {
        //             Mutex::new(self.inn.read()).lock().some_func_2().await;
        //         }
        //     }
        //     struct Inner {
        //
        //     }
        //
        //     impl Inner {
        //         async fn some_func(&self){
        //             println!("some_func");
        //         }
        //
        //         fn some_func_2<'a>(&'a self)-> Pin<Box<dyn std::future::Future<Output = ()> + Send + 'a>> {
        //             async {
        //                 println!("some_func 2");
        //             }.boxed()
        //         }
        //
        //     }
        //
        //     block_on(async {
        //         let outer_rw = Outer {
        //             inn: RwLock::new(Inner{}),
        //         };
        //
        //
        //         outer_rw.run_inner().await;
        //
        //
        //     })
    }

}
