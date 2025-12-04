use crate::builder::{Node, SeqBuilder};
use crate::loader::{LoadRunner, Loader};
use crate::my_tch::ThoModule;
use crate::parzen::optimizer::{AsyncTPEOptimizer, ConfigurationSpace, TPEOptimizer};
use crate::runs::{Empty, Labeled, Logits, NetResources, SavableResult};
use crate::utils::{pad_to, plot_series, train, DataBatch, ListenerResponse, TrainListener};
use crate::utils::{
    cuda_mem, run_split_on_oom, tensor_mem_size, train_2, DummyListener, Reducer,
    SplitQueue, Splittable, SplittableDataBatch, SplittableTensor,
};
use crate::ws::Workspace;
use thoosa_shared::{panic_to_err, panic_to_err_async, CopyToDevice};
use clap::clap_derive::Args;
use crossbeam_utils::sync::WaitGroup;
use futures::FutureExt;
use std::cell::RefCell;

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

use tch::Device::{Cpu, Cuda};
use tch::Kind::Float;
use tch::nn::{AdamW, OptimizerConfig, VarStore};
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

use crate::applicables::{ThoDevice, ThoRunCtx, ThoRunner};
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
use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};
use thoosa_shared::run_configs::{Cachable, RunConfig, Runnable};
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

pub enum LabelNetTypes {
    MaxPadding { params: LabelNetParams },
}

#[derive(Debug)]
pub enum LabelNetApplicableAndNetE {
    MaxPadding {
        applicable: Arc<MotifsNetsSelectionApplicable>,
        params: LabelNetParams, // module: ThoModuleWithStore<Batch<ChannelsMotifsData>, Batch<LabelProbability>>
    },
    FixedNumSlices {
        applicable: Arc<MotifsNetsSelectionApplicable>,
        params: LabelNetParams, // module: ThoModuleWithStore<Batch<ChannelsMotifsFlattenedData>, Batch<LabelProbability>>
    },
}

#[derive(Clone, Debug)]
pub enum LabelNetInputTensors {
    MaxPadding(ChannelsMotifsData),
    FixedNumSlices(ChannelsMotifsFlattenedData),
}

impl Dims for LabelNetInputTensors {
    type Metrics = AnyMetrics;
}

impl From<ThoTensor<ChannelsMotifsData, CachableId>>
    for ThoTensor<LabelNetInputTensors, CachableId>
{
    fn from(value: ThoTensor<ChannelsMotifsData, CachableId>) -> Self {
        ThoTensor::from(value.t)
    }
}

impl From<ThoTensor<ChannelsMotifsFlattenedData, CachableId>>
    for ThoTensor<LabelNetInputTensors, CachableId>
{
    fn from(value: ThoTensor<ChannelsMotifsFlattenedData, CachableId>) -> Self {
        ThoTensor::from(value.t)
    }
}

impl LabelNetApplicableAndNetE {
    pub fn new(
        nettype_args: &LabelNetTypeArgs,
        motifs_nets_selection_applicable: &Arc<MotifsNetsSelectionApplicable>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
    ) -> Self {
        match &nettype_args {
            LabelNetTypeArgs::MaxPadding(params) => {
                debug!("LabelNetApplicableAndNetE::MaxPadding new: params={params:?}");

                let slices_len = slices_len(
                    ws.read().max_input(),
                    motifs_nets_selection_applicable
                        .find_motifs_applicable
                        .run_config()
                        .window(),
                    motifs_nets_selection_applicable
                        .find_motifs_applicable
                        .run_config()
                        .step(),
                );

                LabelNetApplicableAndNetE::MaxPadding {
                    applicable: Arc::clone(&motifs_nets_selection_applicable),
                    // module: LabelNetParams::new(ws.channels(), motifs_nets_selection_applicable.max_nets(), params.num_layers, slices_len, LabelNetTypeParams::MaxPadding(MaxPaddingNetParams {})).label_net().build_tch_module_2::<Batch<ChannelsMotifsData>, Batch<LabelProbability>>(&vs.root()).unwrap()
                    params: LabelNetParams::new(
                        ws.read().channels(),
                        motifs_nets_selection_applicable.max_nets(),
                        params.num_layers,
                        slices_len,
                        LabelNetTypeParams::MaxPadding(MaxPaddingNetParams {}),
                    ),
                }
            }
            LabelNetTypeArgs::FixedNumSlices(params) => {
                debug!("LabelNetApplicableAndNetE::FixedNumSlices new: params={params:?}");

                LabelNetApplicableAndNetE::FixedNumSlices {
                    applicable: Arc::clone(&motifs_nets_selection_applicable),
                    // module: LabelNetParams::new(ws.channels(), motifs_nets_selection_applicable.nets_count(), params.num_layers, params.slices_len, LabelNetTypeParams::FixedNumSlices(FixedNumSlicesNetParams{})).label_net().build_tch_module_2::<Batch<ChannelsMotifsFlattenedData>, Batch<LabelProbability>>(&vs.root()).unwrap()
                    params: LabelNetParams::new(
                        ws.read().channels(),
                        motifs_nets_selection_applicable.nets_count(),
                        params.num_layers,
                        params.slices_len,
                        LabelNetTypeParams::FixedNumSlices(FixedNumSlicesNetParams {}),
                    ),
                }
            }
        }
    }

    pub fn apply(
        &self,
        t: &TCD,
    ) -> Result<ThoTensor<LabelNetInputTensors, CachableId>, ThoosaError> {
        match self {
            LabelNetApplicableAndNetE::MaxPadding { applicable, params } => {
                debug!("LabelNetApplicableAndNetE::MaxPadding apply: params={params:?}");
                applicable
                    .run(t.shallow_clone().into(), ())
                    .map(|t| t.into())
            }
            LabelNetApplicableAndNetE::FixedNumSlices { applicable, params } => {
                debug!("LabelNetApplicableAndNetE::FixedNumSlices apply: params={params:?}");
                applicable
                    .run(t.shallow_clone().into(), params.slices_len)
                    .map(|t| t.into())
            }
        }
    }

    pub fn module(
        &self,
        vs: &VarStore,
    ) -> Result<Box<dyn ThoModule<Batch<LabelNetInputTensors>, Batch<LabelProbability>>>, ThoosaError>
    {
        Ok(Box::new(match self {
            LabelNetApplicableAndNetE::MaxPadding { params, .. } => {
                debug!("LabelNetApplicableAndNetE::MaxPadding module: params={params:?}");
                params
                    .label_net()
                    .build_tch_module_2::<Batch<LabelNetInputTensors>, Batch<LabelProbability>>(
                        &vs.root(),
                    )?
            }
            LabelNetApplicableAndNetE::FixedNumSlices { params, .. } => {
                debug!("LabelNetApplicableAndNetE::FixedNumSlices module: params={params:?}");
                params
                    .label_net()
                    .build_tch_module_2::<Batch<LabelNetInputTensors>, Batch<LabelProbability>>(
                        &vs.root(),
                    )?
            }
        }))
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
    select_motifs_for_label_run_id: u32,
    label: String,
    loss: f32,
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
        find_motifs_run_id: RunId,
        label: String,
        loss: f32,
        label_node_params: LabelNetParams,
        train_motifs_nodes_values: TrainLabelNetsRunConfig,
        var_store: VarStore,
    ) -> Self {
        Self {
            select_motifs_for_label_run_id: find_motifs_run_id,
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
            Some(&[self.select_motifs_for_label_run_id]),
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

pub fn run_opt_train_label_nets(
    run_configs: &Arc<
        OptRunConfigs<
            Vec<NamedVal<TrainLabelNetsRunConfigFields>>,
            Vec<NamedRange<TrainLabelNetsRunConfigFields>>,
            Vec<Hyperparameter<TrainLabelNetsRunConfigFields>>,
            TrainLabelNetsRunConfigFields,
        >,
    >,
    train_labels_resources: Rc<Vec<(i64, Vec<Resource>)>>, //TODO i64 to 1/0 enum for binary class
    valid_labels_resources: Rc<Vec<(i64, Vec<Resource>)>>,
    motifs_nets_selection_applicable_arc: Arc<MotifsNetsSelectionApplicable>,
    label: &String,
    fixed_config: &OptConfig,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    loader: &Arc<RwLock<LoadRunner<Tensor>>>,
    device: &Device,
) -> Result<Box<dyn SavableResult>, ThoosaError> {
    debug!(
        "run_opt_train_label_nodes: label={label:?} run_configs={run_configs:?} channels={} cuda_mem={:?}",
        ws.read().channels(),
        cuda_mem()
    );

    opt_train(
        fixed_config,
        |vs: &VarStore, run_config: &TrainLabelNetsRunConfig, _listener: &mut dyn TrainListener| { //TODO add _listener to train_label_nodes_2
            debug!("run_opt_train_label_nodes:MaxPadding run_config={run_config:?}");


            let mut result = None::<(f32, LabelNetParams)>;

            let applicable_and_net_enum = LabelNetApplicableAndNetE::new(
                &run_config.net, &motifs_nets_selection_applicable_arc,  ws
            );

            run_split_on_oom(
                Reducer::new(
                    LoadingTensorsDataBatch::<ChannelsData, LabelNetInputTensors, _>::new(
                     train_labels_resources.clone(),
                     valid_labels_resources.clone(),
                     run_config.common_training_args.batch_size as usize,
                     Arc::clone(&loader),
                     device.clone(),
                     Some(Box::new( |t:TCD|{
                         applicable_and_net_enum.apply(&t).expect("applicable_and_net_enum.apply failed")
                     }))
                    )

                ),
                |batch| {
                    train_label_net(&applicable_and_net_enum.module(vs)?, run_config, batch, vs, ws, &device)
                        .map(|v| (v,
                              match &applicable_and_net_enum {

                                  LabelNetApplicableAndNetE::MaxPadding { applicable, params } => {
                                      debug!("run_opt_train_label_nodes:MaxPadding bet params={params:?}");

                                      params.clone()
                                  }
                                  LabelNetApplicableAndNetE::FixedNumSlices { applicable, params } => {
                                      debug!("run_opt_train_label_nodes:FixedNumSlices bet params={params:?}");

                                      params.clone()
                                  }
                              }
                        ))
                },
                |v, i| {
                    debug!("v={v:?}");
                    result.replace(v);
                },
                3, //TODO config
            )?;

            Ok(result.expect("no result from train_label_nodes"))
        },
        run_configs,
        *device,
        "label",
    ).map(|(var_store, label_node_params, run_config, min_loss)| {
        Box::new(TrainLabelNetsRunConfigResult::new(motifs_nets_selection_applicable_arc.select_motifs_run_id, label.clone(), min_loss, label_node_params, run_config, var_store)) as Box<dyn SavableResult>
    })
}


pub fn train_label_net<DI: Dims + Send + 'static, DO: Dims + Send + 'static, B: DataBatch<DI>>(
    net: &Box<dyn ThoModule<Batch<DI>, DO>>,
    run_config: &TrainLabelNetsRunConfig,
    mut batch: &mut B,
    var_store: &VarStore,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    device: &Device,
) -> Result<f32, ThoosaError> {
    let device = device.clone();

    let mut optimizer = AdamW::default().amsgrad(true);

    //TODO
    // if let Some(adam_w) = run_config.common_training_args.adam_w {
    //     debug!("train_label_nodes: setting adam_w={adam_w:?}");
    //     optimizer = optimizer.beta1(adam_w.beta1).beta2(adam_w.beta2);
    // }

    debug!("train_label_net: cuda_mem={:?}", cuda_mem());

    let loss = train(
        &device,
        batch as &mut dyn DataBatch<DI>,
        net,
        |pred: &Tensor, target: &Tensor| {
            debug!("train_label_net: pred.size={:?}", pred.size());
            let pred_y = pred.squeeze();
            debug!(
                "train_label_net: pred_y={:?} target={:?} kind={:?}",
                pred_y.size(),
                target.size(),
                target.kind()
            );
            pred_y.binary_cross_entropy::<Tensor>(&target, None, Reduction::Mean)
        },
        run_config.common_training_args.cycles as usize,
        optimizer.build(&var_store, run_config.common_training_args.lr)?,
        |t| {
            debug!(
                "train_label_net: t.size={:?} cuda_mem={:?}",
                t.size(),
                cuda_mem()
            );
            t
        },
        None,
    )?;

    Ok(loss)
}

// pub fn run_opt_train_final_net(
//     train_label_nodes_run_ids: Option<&Vec<RunId>>,
//     train_final_run_config: &OptRunConfigs<Vec<NamedVal<TrainFinalNetRunConfigFields>>, Vec<NamedRange<TrainFinalNetRunConfigFields>>, Vec<Hyperparameter<TrainFinalNetRunConfigFields>>, TrainFinalNetRunConfigFields>,
//     train_resources:Vec<(Resource, Vec<f32>)>,
//     valid_resources: Vec<(Resource, Vec<f32>)>,
//     fixed_config: &OptConfig,
//     ws: &Arc<RwLock<Box<dyn Workspace>>>,
//     store: &Arc<RwLock<Box<dyn Store>>>,
//     loader: &Arc<RwLock< LoadRunner<Tensor>>>,
//     device: &Device,
// ) -> Result<Box<dyn SavableResult>, ThoosaError> {
//
//     let device = device.clone();
//
//     let labels = store.read().list_labels()?;
//
//     debug!("run_opt_train_final: labels={labels:?}");
//
//
//     let label_nets_applicable = Arc::new(LabelNetsApplicable::load_all_filtered_by_run_ids(train_label_nodes_run_ids.map(|v|v.as_slice()), labels, store, ws, device));
//
//     opt_train(
//         fixed_config,
//          |vs:&VarStore, run_config_values:&TrainFinalNetRunConfig, listener: &mut dyn TrainListener|{
//              let label_nets_applicable_rc = Arc::clone(&label_nets_applicable);
//
//             let  batch = Rc::new(Box::new(ClassTensorsLoaderData::<ChannelsData, ChannelsData>::new(
//                 train_resources.clone(),
//                 valid_resources.clone(),
//                 run_config_values.common_training_args.batch_size as usize,
//                 Arc::clone(&loader),
//                 device.clone(),
//                 None,
//             )));
//
//             train_final_net(
//                 run_config_values,
//                 label_nets_applicable.label_nets_num(),
//                 label_nets_applicable.labels_num(),
//                 &mut BatchTransform::new(Rc::clone(&batch), |t:ThoBatch<ChannelsData>|{
//                     label_nets_applicable_rc.run(t.as_ref().into(), ()).expect("label_nets_applicable.apply failed")
//                 }),
//                 vs,
//                 &device
//             )
//         },
//         &train_final_run_config,
//         device,
//         "final"
//     ).map(|(var_store, final_node_params, train_final_nodes_values, min_loss)| {
//         debug!("run_opt_train_final: done, min_loss={min_loss} final_node_params={final_node_params:?}");
//         Box::new(TrainFinalNetsRunConfigResult::new(train_label_nodes_run_ids, min_loss, var_store, final_node_params, train_final_nodes_values)) as Box<dyn SavableResult>
//     })
//
// }

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

pub async fn train_final_net_2<B: AsyncDataBatch<LabelNetsProbabilities, LabelsProbabilities>>(
    run_config_values: &TrainFinalNetRunConfig,
    label_nets_num: u32,
    labels_num: u32,
    batch: &mut B,
    var_store: &VarStore,
    device: &Device,
) -> Result<(f32, FinalNetParams), ThoosaError> {
    let final_node_params =
        FinalNetParams::new(label_nets_num, labels_num, run_config_values.net.num_layers);
    debug!("train_final_node: final_node_params={final_node_params:?}");

    let optimizer = nn::Adam::default();
    /* if let Some(adam_w) = run_config_values.adam_w {
        debug!("train_final_nodes: setting adam_w={adam_w:?}");
        optimizer = optimizer.beta1(adam_w.beta1).beta2(adam_w.beta2);
    }*/

    let loss = train_2(
        device,
        batch,
        final_net(&final_node_params)
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

            pred.cross_entropy_loss::<Tensor>(&target, None, Reduction::Mean, -100, 0.0)
        },
        run_config_values.common_training_args.cycles as usize,
        optimizer.build(&var_store, run_config_values.common_training_args.lr)?,
        |t: ThoBatch<LabelNetsProbabilities>| {
            trace!("train_final: t=\n{t}");
            debug!("train_final: t.size={:?} ", t.size());

            t.as_ref().squeeze().into() //TODO move squeeze
        },
        None,
    )
    .await?;

    Ok((loss, final_node_params))
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

            pred.cross_entropy_loss::<Tensor>(&target, None, Reduction::Mean, -100, 0.0)
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
    DO: Dims + Send, {

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
    fn tho_forward(&self, t: &ThoTensor<DI, CachableId>, train: bool) -> ThoTensor<DO, CachableId> {
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
        self.module.tho_forward(t, false)
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
    num_layers: u32,
}

impl TrainMotifsNetsRunConfig {
    pub fn new(id: Option<RunId>) -> Self {
        Self {
            id,
            common_training_args: CommonTrainingRunConfig::new(1, 1, 0.1, 0.1, 1, 1),
            num_layers: 1,
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

    #[in_map]
    #[run_config]
    pub max_loss: f32,

    #[in_map]
    #[run_config]
    pub opt_max_evals: u32,

    #[in_map]
    #[run_config]
    pub opt_n_init: u32,
}

impl CommonTrainingRunConfig {
    pub fn new(
        cycles: u32,
        batch_size: u32,
        lr: f64,
        max_loss: f32,
        opt_max_evals: u32,
        opt_n_init: u32,
    ) -> Self {
        Self {
            cycles,
            batch_size,
            lr,
            max_loss,
            opt_max_evals,
            opt_n_init,
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
    num_features: u32,
    num_layers: u32,
}

impl MotifNetParams {
    pub fn new(num_features: u32, num_layers: u32) -> Self {
        Self {
            num_features,
            num_layers,
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

    pub fn with_num_layer(&self, num_layers: u32) -> Self {
        Self {
            num_features: self.num_features,
            num_layers,
        }
    }
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
        .func(Tensor::relu);

    for i in 0..params.num_layers as usize {
        net = net
            .linear_s(
                params.num_features * 2,
                #[cfg(debug_assertions)]
                format!("motif_node  {i}th layer"),
            )
            .func(Tensor::relu);
    }

    net
        // .linear_s(180)
        // .func(Tensor::relu)
        .linear_s(
            1,
            #[cfg(debug_assertions)]
            "motif_node last layer".to_string(),
        )
        .func(Tensor::sigmoid)
        .end_seq()
}

#[derive(Debug)]
pub struct MotifsNetsSelectionApplicable {
    pub select_motifs_run_id: RunId,
    pub find_motifs_applicable: FindMotifsApplicable,
    pub channels_motifs: Vec<Vec<Arc<LoadMotifNetOutput>>>,
    pub device: Device,
    pub max_input: u32,
}

impl MotifsNetsSelectionApplicable {
    pub fn new(
        select_motifs_run_id: RunId,
        store: &Arc<RwLock<Box<dyn Store>>>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        device: Device,
        channels_motifs: Vec<Vec<Arc<LoadMotifNetOutput>>>,
    ) -> Self {
        debug!(
            "MotifsNetsSelectionApplicable::new: select_motifs_run_id={select_motifs_run_id}"
        );

        let find_motifs_run_config_ids = store.read().get_input_run_ids(select_motifs_run_id, Some(&FindMotifsRunConfig::type_name())).expect(format!("FindMotifsRunConfig for select_motifs_run_id={select_motifs_run_id:?} not found").as_str());
        assert_eq!(find_motifs_run_config_ids.len(), 1);

        let find_motifs_run_config = FindMotifsRunConfig::from_properties_map__(
            find_motifs_run_config_ids[0],
            &store
                .read()
                .list_run_config_values(find_motifs_run_config_ids[0])
                .expect("list_run_params"),
        )
        .unwrap();

        debug!(
            "MotifsNetsSelectionApplicable::new: find_motifs_run_config={find_motifs_run_config:?}"
        );

        debug!(
            "MotifsNetsSelectionApplicable::new: channels_motifs={:?}",
            channels_motifs
                .iter()
                .map(|c| (c, c.len()))
                .collect::<Vec<_>>()
        );

        MotifsNetsSelectionApplicable {
            select_motifs_run_id,
            find_motifs_applicable: find_motifs_run_config.to_applicable(&ws.read(), device),
            channels_motifs,
            device,
            max_input: ws.read().max_input(),
        }
    }

    //TODO calculate in costr
    pub fn max_nets(&self) -> u32 {
        self.channels_motifs
            .iter()
            .map(|motifs| motifs.len() as u32)
            .max()
            .expect("max ")
    }

    pub fn nets_count(&self) -> u32 {
        self.channels_motifs
            .iter()
            .map(|motifs| motifs.len() as u32)
            .sum()
    }
}

#[derive(Debug, Copy, Clone)]
pub enum NetsPadding {
    Max,
    Fixed(u32),
}

#[async_trait]
impl Runnable<TCD, ThoTensor<ChannelsMotifsData, CachableId>, ()>
    for MotifsNetsSelectionApplicable
{
    type R = ThoRunner;

    fn run(
        &self,
        input: TCD,
        params: (),
    ) -> Result<ThoTensor<ChannelsMotifsData, CachableId>, ThoosaError> {
        let motif_width = self.find_motifs_applicable.run_config().window;

        let unfolded = self
            .find_motifs_applicable
            .run(input, FindMotifsRunConfigOptions::FilterUnfold)?;

        trace!("Applicable for MotifsNetsSelectionApplicable::apply:Max: unfolded=\n{unfolded}");
        debug!(
            "Applicable for MotifsNetsSelectionApplicable::apply:Max: unfolded={unfolded:?} device={:?}",
            unfolded.device()
        );

        let max_num_slices = slices_len(
            self.max_input,
            motif_width,
            self.find_motifs_applicable.run_config().step,
        );

        debug!(
            "Applicable for MotifsNetsSelectionApplicable::apply:Max: motif_width={motif_width} self.max_input={} max_num_slices={max_num_slices}",
            self.max_input
        );

        let tensor = Tensor::stack(
            self.channels_motifs
                .iter()
                .enumerate()
                .map(|(channel, motif_nets)| {
                    debug!("Applicable for MotifsNetsSelectionApplicable::apply:Max: channel={channel} motif_nodes.len()={}", motif_nets.len());

                    if motif_nets.is_empty() {
                        Ok(Tensor::zeros([self.max_nets() as i64, max_num_slices as i64], (Float, Cpu)))
                    } else {
                        // let unfolded_ref = unfolded.as_ref();
                        // debug!("Applicable for MotifsNetsSelectionApplicable::apply:Max: unfolded_ref: {unfolded:?} device={:?}", unfolded.device());

                        let unfolded_channel = match unfolded.t.f_i((channel as i64, .., ..)) {
                            Ok(v) => v,
                            Err(e) => {
                                debug!("Applicable for MotifsNetsSelectionApplicable::apply:Max:  f_i failed: {:?}", e);
                                panic!("Applicable for MotifsNetsSelectionApplicable::apply:Max:  failed: {:?}", e);
                            }
                        }; //channel, slices, slice

                        debug!("Applicable for MotifsNetsSelectionApplicable::apply:Max: unfolded.size={:?}", unfolded_channel.size());
                        trace!("Applicable for MotifsNetsSelectionApplicable::apply:Max: unfolded_channel=\n{unfolded_channel}");

                        let motif_dists = motif_nets.iter().map(|load_motif_output| {
                            // net.set_device(device.clone());
                            debug!("Applicable for MotifsNetsSelectionApplicable::apply:Max: net={load_motif_output:?}");

                            let mut motif_distance_splits = vec![];
                            run_split_on_oom(
                                SplitQueue::<2, _>::new(SplittableTensor::<0/*batch dimension*/>(unfolded_channel.shallow_clone())),
                                |t| {
                                    debug!("Applicable for MotifsNetsSelectionApplicable::apply:Max: forward: t.size={:?}", t.as_ref().size());
                                    let r = load_motif_output.nets[0].1.infer(
                                        &t.as_ref().to_device(self.device.clone()).into()
                                    ).as_ref().flatten(0, 1); //flatten - removes always 1-sized distance dimension [ slices, slices_len]->[slices, [distance]]
                                    debug!("Applicable for MotifsNetsSelectionApplicable::apply:Max: forward:  r.size={:?}", r.size());

                                    Ok(r.to_device(Cpu))
                                },
                                |v, t| {
                                    debug!("Applicable for MotifsNetsSelectionApplicable::apply:Max: on_ok: v={v:?} device={:?}", v.device());
                                    motif_distance_splits.push(v);
                                },
                                3 //TODO config
                            ).unwrap(); //TODO unwrap

                            // net.set_device(Cpu);

                            debug!("Applicable for MotifsNetsSelectionApplicable::apply:Max: motif_distance_splits={motif_distance_splits:?}");
                            if motif_distance_splits.len() > 0 {
                                Ok(Tensor::concat(&motif_distance_splits, 0))
                            } else if motif_distance_splits.len() == 1 {
                                Ok(motif_distance_splits.pop().unwrap())
                            } else {
                                Err(ThoosaError::new("no motif distances "))
                            }
                        }).collect::<Result<Vec<_>, ThoosaError>>()?;


                        debug!("Applicable for MotifsNetsSelectionApplicable::apply:Max: motif_dists={motif_dists:?}");
                        //TODO pad_sequence?
                        let motif_nets_out = Tensor::stack(motif_dists.as_slice(), 0);

                        trace!("Applicable for MotifsNetsSelectionApplicable::apply:Max: motif_nets_out=\n{motif_nets_out}");
                        debug!("Applicable for MotifsNetsSelectionApplicable::apply:Max: motif_dists.len={} motif_nets_out.size={:?}", motif_dists.len(), motif_nets_out.size());

                        let r = pad_to(&motif_nets_out, &[self.max_nets() as i64, max_num_slices as i64]);

                        debug!("Applicable for MotifsNetsSelectionApplicable::apply:Max: r.size={:?}", r.as_ref().map(|r|r.size()));

                        r
                    }
                })
                .collect::<Result<Vec<_>, ThoosaError>>().unwrap().as_slice(),
            0
        );
        debug!(
            "Applicable for MotifsNetsSelectionApplicable::apply:Max: t.size={:?}",
            tensor.size()
        );
        Ok(tensor.into())
    }

    async fn async_run(
        self: Arc<Self>,
        input: Arc<TCD>,
        runner: Arc<Self::R>,
        ctx: Option<Arc<RwLock<<<Self as Runnable<TCD, ThoTensor<ChannelsMotifsData, CachableId>, ()>>::R as Runner>::Ctx>>>,
    ) -> Arc<ThoTensor<ChannelsMotifsData, CachableId>> {
        Arc::new(self.run(input.deref().shallow_clone(), ()).unwrap())
    }

    fn device(&self) -> impl ToOrdinal {
        ThoDevice::from(Cpu)
    }
}

#[async_trait]
impl Cachable<TCD, ThoTensor<ChannelsMotifsData, CachableId>, ()>
    for MotifsNetsSelectionApplicable
{
    fn applied_id(&self, input_id: &CachableId) -> CachableId {
        <MotifsNetsSelectionApplicable as Cachable<
            TCD,
            ThoTensor<ChannelsMotifsData, CachableId>,
            (),
        >>::format_0(input_id.0)
    }

    fn name(&self) -> String {
        "MotifsNetsSelectionApplicable".to_string()
    }

    fn run_id(&self) -> String {
        self.select_motifs_run_id.to_string()
    }
}

#[async_trait]
impl Runnable<TCD, ThoTensor<ChannelsMotifsFlattenedData, CachableId>, u32>
    for MotifsNetsSelectionApplicable
{
    type R = ThoRunner;

    fn run(
        &self,
        input: TCD,
        slices_len: u32,
    ) -> Result<ThoTensor<ChannelsMotifsFlattenedData, CachableId>, ThoosaError> {
        let motif_width = self.find_motifs_applicable.run_config().window;

        let filtered = self
            .find_motifs_applicable
            .run(input, FindMotifsRunConfigOptions::Filter)?;
        debug!(
            "Applicable for MotifsNetsSelectionApplicable::apply:Fixed: filtered.size={:?} motif_width={motif_width} slices_len={slices_len}",
            filtered.as_ref().size()
        );

        let (input_slice_step, pad) =
            clc_step(filtered.as_ref().size()[1] as u32, motif_width, slices_len); //(channel.size()[0]-motif_width)/100;
        debug!(
            "Applicable for MotifsNetsSelectionApplicable::apply:Fixed: input_slice_step={input_slice_step} pad={pad}"
        );

        let unfolded = filtered
            .as_ref()
            .pad([0, pad as i64], "constant", 0.)
            .unfold(1, motif_width as i64, input_slice_step as i64);

        Ok(Tensor::concat(
            self.channels_motifs
                .iter()
                .enumerate()
                .filter(|(_, motif_nodes)| !motif_nodes.is_empty())
                .map(|(channel, motif_nets)| {
                    // let channel = motif_nets.channel;
                    let unfolded_channel = unfolded.i((channel as i64, .., ..)); //channel, slices, slice

                    debug!("Applicable for MotifsNetsSelectionApplicable::apply:Fixed: unfolded.size={:?}", unfolded_channel.size());
                    trace!("Applicable for MotifsNetsSelectionApplicable::apply:Fixed: unfolded_channel=\n{unfolded_channel}");

                    let motif_dists = motif_nets.iter().map(|load_motif_net_output| {
                        // net.set_device(device.clone());
                        debug!("Applicable for MotifsNetsSelectionApplicable::apply:Fixed: load_motif_net_output={load_motif_net_output:?}");

                        let mut motif_distance_splits = vec![];
                        run_split_on_oom(
                            SplitQueue::<2, _>::new(SplittableTensor::<0/*batch dimension*/>(unfolded_channel.shallow_clone())),
                            |t| {
                                debug!("Applicable for MotifsNetsSelectionApplicable::apply:Fixed: forward: t.size={:?}", t.as_ref().size());
                                let r = load_motif_net_output.nets[0].1.infer(
                                    &t.as_ref().to_device(self.device.clone()).into()
                                ).as_ref().flatten(0, 1); //flatten - removes always 1-sized distance dimension [ slices, slices_len]->[slices, [distance]]
                                debug!("Applicable for MotifsNetsSelectionApplicable::apply:Fixed: forward:  r.size={:?}", r.size());

                                Ok(r.to_device(Cpu))
                            },
                            |v, t| {
                                debug!("Applicable for MotifsNetsSelectionApplicable::apply:Fixed: on_ok: v={v:?} device={:?}", v.device());
                                motif_distance_splits.push(v);
                            },
                            10 //TODO config
                        ).unwrap(); //TODO unwrap

                        // net.set_device(Cpu);

                        debug!("Applicable for MotifsNetsSelectionApplicable::apply:Fixed: motif_distance_splits={motif_distance_splits:?}");
                        if motif_distance_splits.len() > 0 {
                            Ok(Tensor::concat(&motif_distance_splits, 0))
                        } else if motif_distance_splits.len() == 1 {
                            Ok(motif_distance_splits.pop().unwrap())
                        } else {
                            Err(ThoosaError::new("no motif distances "))
                        }
                    }).collect::<Result<Vec<_>, ThoosaError>>()?;


                    debug!("Applicable for MotifsNetsSelectionApplicable::apply:Fixed: motif_dists={motif_dists:?}");

                    let motif_nets_out = Tensor::stack(motif_dists.as_slice(), 0);

                    trace!("Applicable for MotifsNetsSelectionApplicable::apply:Fixed: motif_nets_out=\n{motif_nets_out}");
                    debug!("Applicable for MotifsNetsSelectionApplicable::apply:Fixed: motif_dists.len={} motif_nets_out.size={:?}", motif_dists.len(), motif_nets_out.size());

                    Ok(motif_nets_out)
                })
                .collect::<Result<Vec<_>, ThoosaError>>()?.as_slice(),
            0
        ).into())
    }

    async fn async_run(
        self: Arc<Self>,
        input: Arc<TCD>,
        runner: Arc<Self::R>,
        ctx: Option<
            Arc<
                RwLock<
                    <<Self as Runnable<
                        TCD,
                        ThoTensor<ChannelsMotifsFlattenedData, CachableId>,
                        u32,
                    >>::R as Runner>::Ctx,
                >,
            >,
        >,
    ) -> Arc<ThoTensor<ChannelsMotifsFlattenedData, CachableId>> {
        todo!()
    }

    fn device(&self) -> impl ToOrdinal {
        ThoDevice::from(Cpu)
    }
}

#[async_trait]
impl Cachable<TCD, ThoTensor<ChannelsMotifsFlattenedData, CachableId>, u32>
    for MotifsNetsSelectionApplicable
{
    fn applied_id(&self, input_id: &CachableId) -> CachableId {
        todo!()
    }

    fn name(&self) -> String {
        todo!()
    }

    fn run_id(&self) -> String {
        todo!()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabelNetParams {
    channels: u32,
    motif_nets: u32,
    motif_layers: u32,
    slices_len: u32,
    net_type: LabelNetTypeParams,
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
                    t.shallow_clone()
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
                                .func(Tensor::relu);
                        }

                        net.func(Tensor::relu)
                            .linear_s(
                                1,
                                #[cfg(debug_assertions)]
                                "label_net_fixed_num_slices: motif nets last layer".to_string(),
                            )
                            // .func(Tensor::sigmoid)
                            .end_seq()
                    })
                    .collect(),
                Box::new(move |t: &Tensor| {
                    (0..motif_nodes)
                        .map(|net_i| {
                            debug!(
                                "label_net_fixed_num_slices: split, t.size={:?}  net_i={net_i}",
                                t.size()
                            );
                            t.i((.., net_i as i64, ..))
                        })
                        .collect()
                }), //label batch, channel, max_motif_nets, max slices
                Box::new(move |t: Vec<Tensor>| {
                    debug!("label_net_fixed_num_slices: join t={t:?}");

                    let out = Tensor::concat(t.as_slice(), 1); //.mean_dim(1, false, Float);
                    debug!("label_net_fixed_num_slices: join out={:?}", out.size());

                    out
                }),
            )
            .closure(
                Arc::new(Box::new(move |t: &Tensor| {
                    debug!(
                        "label_net_fixed_num_slices: merged final in flatten t.size={:?}",
                        t.size()
                    );

                    t.shallow_clone()
                })),
                Some(motif_nodes),
            )
            .linear(
                #[cfg(debug_assertions)]
                "label_net_fixed_num_slices: first main layer".to_string(),
            )
            // .func(Tensor::relu)
            .linear_s(
                1,
                #[cfg(debug_assertions)]
                "label_net_fixed_num_slices: final layer".to_string(),
            )
            .func(Tensor::sigmoid)
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
                    t.shallow_clone()
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
                                .func(Tensor::relu);
                        }

                        net.func(Tensor::relu)
                            .linear_s(
                                1,
                                #[cfg(debug_assertions)]
                                "label_net_max_padding: motif nets last layer".to_string(),
                            )
                            // .func(Tensor::sigmoid)
                            .end_seq()
                    })
                    .collect(),
                Box::new(move |t: &Tensor| {
                    (0..motif_nodes)
                        .map(|net_i| {
                            debug!(
                                "label_net_max_padding: split, t.size={:?} net_i={net_i}",
                                t.size()
                            );
                            t.i((.., .., net_i as i64, ..))
                        })
                        .collect()
                }), //label batch, channel, max_motif_nets, max slices
                Box::new(move |t: Vec<Tensor>| {
                    debug!("label_net_max_padding: join t={t:?}");

                    let out = Tensor::stack(t.as_slice(), 1).mean_dim(1, false, Float);
                    debug!("label_net_max_padding: join out={:?}", out.size());

                    out
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

                    t_0
                })),
                Some(params.channels),
            )
            .linear(
                #[cfg(debug_assertions)]
                "label_net_max_padding: first main layer".to_string(),
            )
            // .func(Tensor::relu)
            .linear_s(
                1,
                #[cfg(debug_assertions)]
                "label_net_max_padding: final layer".to_string(),
            )
            .func(Tensor::sigmoid)
            .end_seq()
    }
}

pub fn flat_label_node(params: &LabelNetParams, slices_len: u32) -> Node {
    debug!("motif_node: params={params:?}");
    let num_features = params.channels * params.motif_nets * slices_len;
    let mut builder = SeqBuilder::new(slices_len, None);
    builder
        .closure(
            Arc::new(Box::new(move |t: &Tensor| {
                let t_0 = t.flatten(1, 3);
                debug!(
                    "label_node flatten t.size={:?} t_0.size={:?}",
                    t.size(),
                    t_0.size()
                );

                t_0
            })),
            Some(num_features),
        )
        .linear_s(
            num_features,
            #[cfg(debug_assertions)]
            "flat_label_node 0".to_string(),
        )
        .func(Tensor::relu)
        .linear_s(
            num_features,
            #[cfg(debug_assertions)]
            "flat_label_node 1".to_string(),
        )
        .func(Tensor::relu)
        .linear_s(
            1,
            #[cfg(debug_assertions)]
            "flat_label_node 2".to_string(),
        )
        .func(Tensor::sigmoid)
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
        .func(Tensor::relu);

    for i in 0..params.num_layers {
        net = net
            .linear_s(
                params.in_features,
                #[cfg(debug_assertions)]
                format!("final_net: {i}th layer"),
            )
            .func(Tensor::relu)
    }

    net
        // .func(Tensor::relu)
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
    fn on_valid_cycle(&mut self, cycle: usize, loss: f32) -> ListenerResponse {
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

#[async_trait]
impl Output for Box<TrainMotifsNetsResult> {
    async fn copy_to_device(self: Arc<Self>, device_ordinal: usize) -> Arc<Self> {
        todo!() //TODO not needed for not cachable Outputs
    }
}

impl WithId for TrainMotifsNetsResult {
    type ID = RunId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[async_trait]
impl Output for TrainMotifsNetsResult {}


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

    info!("opt_train_2: running opt train {descr} ");

    for grid_vals in run_config_source.grid_to_vals().into_iter() {
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

                let (loss, node_params) = match panic_to_err_async!(
                    train_clone
                        .async_call_mut((
                            &vs,
                            run_config.clone(),
                            &mut listener /*Arc::new(Mutex::new(Box::new(listener)))*/
                        ))
                        .await
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        warn!("opt_train failed with {e:?} on {run_config:?} ");
                        (f32::MAX, Default::default())
                    }
                };
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
    let node_params = MotifNetParams::new(motif_width, run_config.num_layers);
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

            loss
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
    let node_params = MotifNetParams::new(motif_width, run_config.num_layers);
    debug!("run_train_motif_net: node_params={node_params:?}");

    let loss = train_2(
        &device,
        batch,
        motif_node(&node_params).build_tch_module::<Batch<Slice>, Batch<MotifProbability>>(&vs.root())?,
        |pred: &Tensor, target: &Tensor| {
            let pred_y = pred; //.i((.., 0));

            let loss = pred_y.binary_cross_entropy::<Tensor>(&target.totype(Float), None, Reduction::Mean).set_requires_grad(true);
            debug!("run_train_motif_net: pred_y ={pred_y} target={target} loss={loss}");

            loss
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
pub struct LabelNetApplicable {
    pub motifs_nets_selection_applicable: MotifsNetsSelectionApplicable,
    pub label_net_params: LabelNetParams,
    pub module: ThoModuleWithStore<MotifsNetsSelectionDims, Batch<LabelProbability>>,
    device: Device,
}

impl LabelNetApplicable {
    pub fn motifs_nets_selection_applicable_options(&self) -> NetsPadding {
        match self.label_net_params.net_type {
            LabelNetTypeParams::MaxPadding(_) => NetsPadding::Max,
            LabelNetTypeParams::FixedNumSlices(_) => {
                NetsPadding::Fixed(self.label_net_params.slices_len)
            }
        }
    }
}

#[derive(Debug)]
pub struct LabelNetsApplicable {
    labels_applicables: HashMap<String, Vec<LabelNetApplicable>>,
    labels: Vec<String>,
}

#[derive(Debug)]
pub struct LoadMotifNetCachable {
    device: ThoDevice,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
}

impl LoadMotifNetCachable {
    pub fn new(device: ThoDevice, ws: Arc<RwLock<Box<dyn Workspace>>>) -> Self {
        Self { device, ws }
    }
}

#[derive(Debug)]
pub struct LoadMotifNetInput {
    motif_id: u32,
    cache_id: CachableId,
    channel: u32,
}

impl LoadMotifNetInput {
    pub fn new(motif_id: u32, channel: u32) -> Self {
        Self {
            motif_id,
            cache_id: format!("MotifNets_{}", motif_id).into(),
            channel,
        }
    }
}

impl WithId for LoadMotifNetInput {
    type ID = CachableId;

    fn id(&self) -> &Self::ID {
        todo!()
    }
}

impl Input for LoadMotifNetInput {
    fn id(&self) -> &CachableId {
        &self.cache_id
    }
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

#[async_trait]
impl Output for LoadMotifNetOutput {
    async fn copy_to_device(self: Arc<Self>, device_ordinal: usize) -> Arc<Self> {
        Arc::new(LoadMotifNetOutput {
            channel: self.channel,
            nets: join_all(self
                .nets
                .iter()
                .map(|(params, net)|  {
                    async {
                        (
                            params.clone(),
                            net.copy_to_device(0).await.unwrap()
                        )
                    }
                })
            ).await
        })
    }
}

#[async_trait]
impl Runnable<LoadMotifNetInput, LoadMotifNetOutput, ()> for LoadMotifNetCachable {
    type R = ThoRunner;

    async fn async_run(
        self: Arc<Self>,
        input: Arc<LoadMotifNetInput>,
        runner: Arc<Self::R>,
        ctx: Option<Arc<RwLock<<<Self as Runnable<LoadMotifNetInput, LoadMotifNetOutput, ()>>::R as Runner>::Ctx>>>,
    ) -> Arc<LoadMotifNetOutput> {
        /* let channel_motif_models_paths = self.ws.read().motif_models(input.channel as u32, input.motif_id).unwrap();
        Arc::new(LoadMotifNetOutput::new(input.channel, async {

            channel_motif_models_paths.into_iter().map(|(model, config)| {
                let motif_node_params = MotifNetParams::from_file(&config).unwrap();

                debug!("load_motif_models_ws: model={model:?} config={config:?} motif_node_params={motif_node_params:?}");

                //TODO ChannelsMotifsFlattenData
                let params_for_builder = motif_node_params.clone();
                let module: ThoModuleWithStore<SlicesData, MotifData> = ThoModuleWithStore::load(
                    move |path: &nn::Path| motif_node(&params_for_builder).build_tch_module(path),
                    model,
                    self.device.clone().into(),
                )
                .unwrap();

                (motif_node_params, module)
            }).collect::<Vec<_>>()

        }.await))*/

        //TODO make loading async - ws return
        Arc::new(LoadMotifNetOutput::new(
            input.channel,
            self.ws
                .read()
                .load_motif_nets(input.channel, input.motif_id, &self.device)
                .unwrap(),
        ))
    }

    fn device(&self) -> impl ToOrdinal {
        self.device.clone()
    }
}

#[async_trait]
impl Cachable<LoadMotifNetInput, LoadMotifNetOutput, ()> for LoadMotifNetCachable {
    fn applied_id(&self, input_id: &CachableId) -> CachableId {
        <LoadMotifNetCachable as Cachable<LoadMotifNetInput, LoadMotifNetOutput, ()>>::format_0(
            input_id.0,
        )
    }

    fn name(&self) -> String {
        "MotifNet".to_string()
    }

    fn run_id(&self) -> String {
        todo!()
    }
}

///returns vec[channels][motifs][LoadMotifNetOutput(motif's net)]
pub async fn load_channels_motifs(
    select_motifs_run_id: RunId,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    device: Device,
    runner: &Arc<ThoRunner>,
    ctx: Option<Arc<RwLock<ThoRunCtx>>>,
) -> Vec<Vec<Arc<LoadMotifNetOutput>>> {
    let load_motif_net_cachable = Arc::new(LoadMotifNetCachable::new(device.into(), ws.clone()));

    join_all(
        store
            .read()
            .list_selected_motifs(select_motifs_run_id, ws.read().channels())
            .expect("list_selected_motifs") //TODO move out
            .into_iter()
            .enumerate()
            .flat_map(|(channel, channels_motifs)| {
                channels_motifs
                    .into_iter()
                    .map(move |(motif_id, _)| (channel, motif_id))
            })
            .map(async |(channel, motif_id)| {
                debug!("load_channels_motifs: channel={channel} motif_id={motif_id}");
                runner
                    .run_load_motif(
                        &Arc::new(LoadMotifNetInput::new(motif_id, channel as u32)),
                        &load_motif_net_cachable,
                        ctx.as_ref().map(|v| Arc::clone(v)),
                    )
                    .await
            }),
    )
    .await
    .into_iter()
    .fold(
        vec![vec![]; ws.read().channels() as usize],
        |mut acc, output| {
            acc[output.channel as usize].push(output);
            acc
        },
    )
}

impl LabelNetsApplicable {
    pub async fn load_all_filtered_by_run_ids(
        train_label_nodes_run_ids: Option<&[RunId]>,
        labels: Vec<String>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        device: Device,
        runner: &Arc<ThoRunner>,
    ) -> LabelNetsApplicable {
        let label_models = ws
            .read()
            .load_label_nets(train_label_nodes_run_ids, &device)
            .expect("load_all_label_models_ws");

        LabelNetsApplicable {

            labels_applicables: join_all(label_models.into_iter().map( | (label, label_run_id_nodes)| {
                async {
                    let label_applicables = join_all(label_run_id_nodes.into_iter().map(async |(label_run_id, label_net_params, module)| {
                        let select_motifs_run_ids = store
                            .read()
                            .get_input_run_ids(label_run_id, Some(&SelectMotifsRunConfig::type_name()))
                            .expect("get_input_run_ids(motif_node_params_selection_run_ids")
                            .into_iter()
                            .collect::<HashSet<RunId>>(); //TODO distinct in sql?

                        debug!("load_all_label_and_motif_models: label={label} label_run_id={label_run_id} select_motifs_run_ids={select_motifs_run_ids:?}");

                        assert_eq!(select_motifs_run_ids.len(), 1);
                        let select_motifs_for_label_run_id = select_motifs_run_ids.into_iter().next().unwrap();

                        let channels_motifs: Vec<Vec<Arc<LoadMotifNetOutput>>> = load_channels_motifs(
                            select_motifs_for_label_run_id,
                            ws,
                            store,
                            device,
                            runner,
                            None
                        ).await;


                        LabelNetApplicable {
                            motifs_nets_selection_applicable: MotifsNetsSelectionApplicable::new(select_motifs_for_label_run_id, store, ws, device, channels_motifs),
                            label_net_params,
                            module,
                            device
                        }
                    }).collect::<Vec<_>>()).await;
                    (label, label_applicables)
                }
            })).await.into_iter().fold(HashMap::new(), |mut a, (label, label_applicables)|{
                a.insert(label, label_applicables);
                a
            }),
            labels
        }
    }

    pub fn labels_num(&self) -> u32 {
        self.labels_applicables.len() as u32
    }

    pub fn label_nets_num(&self) -> u32 {
        self.labels_applicables
            .iter()
            .map(|(_, nets)| nets.len() as u32)
            .sum::<u32>()
    }
}

#[async_trait]
impl Runnable<ThoBatch<ChannelsData>, ThoBatch<LabelProbability>, ()> for LabelNetApplicable {
    type R = ThoRunner;

    fn run(
        &self,
        input: ThoBatch<ChannelsData>,
        _: (),
    ) -> Result<ThoBatch<LabelProbability>, ThoosaError> {
        debug!(
            "Applicable for LabelNetApplicable::apply: batch.size={:?}",
            input.size()
        );

        Ok(match &self.label_net_params.net_type {
            LabelNetTypeParams::MaxPadding(params) => {
                let out = self.module.infer(
                    //label net accepts batch, motif net - single tensor
                    &Tensor::stack(
                        (0..input.size()[0])
                            .map(|i| {
                                let t: ThoTensor<ChannelsMotifsData, CachableId> =
                                    self.motifs_nets_selection_applicable.run(
                                        input.as_ref().i((i, .., ..)).into(),
                                        (), //NetsPadding::Max
                                    )?;

                                debug!("Applicable for LabelNetApplicable::apply: i={i} t={t:?}");

                                Ok::<Tensor, ThoosaError>(t.as_ref().shallow_clone())
                            })
                            .collect::<Result<Vec<_>, _>>()?
                            .as_slice(),
                        0,
                    )
                    .to_device(self.device)
                    .into(),
                );
                debug!(
                    "Applicable for LabelNetApplicable MaxPadding::apply:  out.size()={:?}",
                    out.size()
                );
                //batch_size x 1
                assert_eq!(out.size().len(), 2);
                assert_eq!(out.size()[1], 1);

                out
            }
            LabelNetTypeParams::FixedNumSlices(params) => {
                let out = self.module.infer(
                    //label net accepts batch, motif net - single tensor
                    &Tensor::stack(
                        (0..input.size()[0])
                            .map(|i| {
                                let t: ThoTensor<ChannelsMotifsFlattenedData, CachableId> =
                                    self.motifs_nets_selection_applicable.run(
                                        input.as_ref().i((i, .., ..)).into(),
                                        self.label_net_params.slices_len, //TODO dims
                                    )?;

                                debug!("Applicable for LabelNetApplicable::apply: i={i} t={t:?}");

                                Ok::<Tensor, ThoosaError>(t.as_ref().shallow_clone())
                            })
                            .collect::<Result<Vec<_>, _>>()?
                            .as_slice(),
                        0,
                    )
                    .to_device(self.device)
                    .into(),
                );

                //batch_size x 1
                assert_eq!(out.size().len(), 2);
                assert_eq!(out.size()[1], 1);

                debug!(
                    "Applicable for LabelNetApplicable FixedNumSlices    ::apply:  out.size()={:?}",
                    out.size()
                );

                out
            }
        })
    }

    async fn async_run(
        self: Arc<Self>,
        input: Arc<ThoBatch<ChannelsData>>,
        runner: Arc<Self::R>,
        ctx: Option<Arc<RwLock<<<Self as Runnable<ThoBatch<ChannelsData>, ThoBatch<LabelProbability>, ()>>::R as Runner>::Ctx>>>,
    ) -> Arc<ThoBatch<LabelProbability>> {
        todo!()
    }

    fn device(&self) -> impl ToOrdinal {
        ThoDevice::from(Cpu)
    }
}

#[async_trait]
impl Cachable<ThoBatch<ChannelsData>, ThoBatch<LabelProbability>, ()> for LabelNetApplicable {
    fn applied_id(&self, input_id: &CachableId) -> CachableId {
        todo!()
    }

    fn name(&self) -> String {
        todo!()
    }

    fn run_id(&self) -> String {
        todo!()
    }
}

#[async_trait]
impl Runnable<ThoBatch<ChannelsData>, ThoBatch<LabelNetsProbabilities>, ()>
    for LabelNetsApplicable
{
    type R = ThoRunner;

    fn run(
        &self,
        input: ThoBatch<ChannelsData>,
        _: (),
    ) -> Result<ThoBatch<LabelNetsProbabilities>, ThoosaError> {
        debug!(
            "Applicable for LabelNetsApplicable::apply: self.labels_applicables={:?} input.size={:?}",
            self.labels_applicables,
            input.size()
        );

        let t = Tensor::stack(
            self.labels.iter()
                .map(|label| (label, self.labels_applicables.get(label).expect(format!("no {label} in labels applicable").as_str())))
                .flat_map(|(label, label_applicables)| {
                    debug!("Applicable for LabelNetsApplicable::apply: label={label}");

                    label_applicables.iter().map(|label_applicable| {
                        label_applicable.run(input.shallow_clone(), ()).map(|v| {
                            let label_applicable_out = v.as_ref().shallow_clone();

                            debug!("Applicable for LabelNetsApplicable::apply: label_applicable_out.size={label_applicable_out:?}");
                            label_applicable_out
                        })
                    })
                }).collect::<Result<Vec<_>, _>>()?.as_slice(),
            1
        );

        debug!(
            "Applicable for LabelNetsApplicable::apply: t.size={:?}",
            t.size()
        );

        Ok(t.into())
    }

    async fn async_run(
        self: Arc<Self>,
        input: Arc<ThoBatch<ChannelsData>>,
        runner: Arc<Self::R>,
        ctx: Option<
            Arc<
                RwLock<
                    <<Self as Runnable<
                        ThoBatch<ChannelsData>,
                        ThoBatch<LabelNetsProbabilities>,
                        (),
                    >>::R as Runner>::Ctx,
                >,
            >,
        >,
    ) -> Arc<ThoBatch<LabelNetsProbabilities>> {
        todo!()
    }
    fn device(&self) -> impl ToOrdinal {
        ThoDevice::from(Cpu)
    }
}

#[async_trait]
impl Cachable<ThoBatch<ChannelsData>, ThoBatch<LabelNetsProbabilities>, ()>
    for LabelNetsApplicable
{
    fn applied_id(&self, input_id: &CachableId) -> CachableId {
        todo!()
    }

    fn name(&self) -> String {
        todo!()
    }

    fn run_id(&self) -> String {
        todo!()
    }
}

#[derive(Debug)]
pub struct FinalNetApplicable {
    label_nets_applicable: LabelNetsApplicable,
    final_net_params: FinalNetParams,
    final_net_module: ThoModuleWithStore<Batch<LabelProbability>, Batch<LabelsProbabilities>>,
    labels: Vec<String>,
}

impl FinalNetApplicable {
    pub fn new(
        final_run_id: RunId,
        labels: Vec<String>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        device: Device,
        label_nets_applicable: LabelNetsApplicable,
    ) -> FinalNetApplicable {
        let (final_net_params, final_net_module) =
            load_final_model(final_run_id, ws, &device).expect("load_final_model failed");

        FinalNetApplicable {
            label_nets_applicable,
            final_net_params,
            final_net_module,
            labels,
        }
    }
}
#[async_trait]
impl Runnable<TCD, ThoTensor<Final, CachableId>, ()> for FinalNetApplicable {
    type R = ThoRunner;

    fn run(&self, input: TCD, options: ()) -> Result<ThoTensor<Final, CachableId>, ThoosaError> {
        assert_eq!(input.t.size().len(), 2); //single tensor - channels x data

        Ok(self
            .final_net_module
            .infer(
                &self
                    .label_nets_applicable
                    .run(input.as_ref().unsqueeze(0).into(), options)?
                    .as_ref()
                    .view((1, -1))
                    .into(),
            )
            .as_ref()
            .shallow_clone()
            .into())
    }

    async fn async_run(
        self: Arc<Self>,
        input: Arc<TCD>,
        runner: Arc<Self::R>,
        ctx: Option<
            Arc<
                RwLock<
                    <<Self as Runnable<TCD, ThoTensor<Final, CachableId>, ()>>::R as Runner>::Ctx,
                >,
            >,
        >,
    ) -> Arc<ThoTensor<Final, CachableId>> {
        todo!()
    }

    fn device(&self) -> impl ToOrdinal {
        ThoDevice::from(Cpu)
    }
}

#[async_trait]
impl Cachable<TCD, ThoTensor<Final, CachableId>, ()> for FinalNetApplicable {
    fn applied_id(&self, input_id: &CachableId) -> CachableId {
        todo!()
    }

    fn name(&self) -> String {
        todo!()
    }

    fn run_id(&self) -> String {
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
    pub(crate) num_layers: u32,
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

#[cfg(test)]
mod train_tests {
    use crate::applicables::ThoRunner;
    use crate::builder::SeqBuilder;
    use crate::dims::{
        AnyDims, Batch, Dims, LabelProbability, MotifProbability, MotifsNetsSelectionDims, Slice,
        ThoBatch,
    };
    use crate::find_motifs_run_config::FindMotifsRunConfig;
    use crate::loader::{LoadRunner, OfMemSize, SyncWrapper};
    use crate::my_tch::ThoModule;
    use crate::simp::DummyDs;
    use crate::tho_tensor::ThoTensor;
    use crate::{train_nets, utils};
    use crate::train_nets::{
        load_channels_motifs, opt_train, BatchTransform,
        ClassTensorsLoaderData, CommonTrainingRunConfig, CommonTrainingRunConfigRangeOrValue,
        FinalNetArgs, FinalNetParams, FindMotifsRunConfigRangeOrValue,
        FixedNumSlicesNetArgs, FixedNumSlicesNetParams, InputFilterRangeOrValue,
        LabelNetApplicable, LabelNetApplicableAndNetE, LabelNetInputTensors, LabelNetTypeArgs,
        LabelNetTypeArgsRangeOrValue, LabelNetTypeParams, LabelNetsApplicable, LoadMotifNetOutput,
        LoadingTensorsDataBatch, MaxPaddingNetArgs, MaxPaddingNetArgsRangeOrValue,
        MaxPaddingNetParams, MotifsNetsSelectionApplicable, NetsPadding, OptConfig, PlotVecf32,
        RangeOrValue, TrainFinalNetRunConfig, TrainLabelNetsRunConfig,
        TrainLabelNetsRunConfigRangeOrValue,
    };
    use crate::train_nets::{
        InferArgs, LabelNetParams, MotifNetParams, TensorsDataBatch, ThoModuleWithStore,
    };
    use crate::transforms::filters::BandPassFilterParamsRangeOrValue;
    use crate::utils::{
        lin_t, pad_to, reset_print_options, run_split_on_oom, t_0_0, tensor_mem_size, DataBatch,
        Reducer, SplitQueue, Splittable, SplittableTensor,
    };
    use crate::utils::{
        cuda_mem, from, optimizer_mem_size, set_requires_grad_in_subpath, train,
        vs_mem_size, SplittableDataBatch, TrainListener,
    };
    use crate::ws::mock_ws_simp;
    use crate::ws::{FsWorkspace, Workspace, WsMockBuilder};
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
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::{Float, Half};
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::nn::{init, AdamW, BatchNormConfig, LinearConfig, Module, OptimizerConfig, VarStore};
    use tch::{nn, Device, Tensor};
    use tch::{IndexOp, Kind, Reduction};
    use thoosa_properties_map::ToHashmap;
    use thoosa_run_config_args::OptRunConfigSource;
    use thoosa_shared::applicables_runner::{CachableId, Input};
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::Runnable;
    use thoosa_shared::run_configs::{Cachable, OptRunConfigs, RunId};
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

    ///sums and multiplies inputs by k
    pub fn test_label_net<DI: Dims + Send + 'static, DO: Dims + Send + 'static>(
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

                    t_0
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

    #[test]
    fn test_test_net() -> Result<(), ThoosaError> {
        let t_0 = lin_test_tensor(1000, 5);
        debug!("t_0.size={:?}", t_0.size());

        let t: ThoTensor<AnyDims, CachableId> = t_0.unfold(1, 100, 10).into();

        debug!("t.size={:?}", t.size());

        //sums and multiplies by k
        let r_0: ThoTensor<AnyDims, CachableId> =
            utils::test_net(100, 1., Cpu, "test_net")?.tho_forward(&t, true);
        debug!("r_0.size={:?}", r_0.size());

        assert_eq!(Tensor::from_slice(&[4950.]), r_0.as_ref().i((0, 0, ..)));
        assert_eq!(
            Tensor::from_slice(&[9900.]),
            utils::test_net::<AnyDims, AnyDims>(100, 2., Cpu, "test_net")?
                .tho_forward(&t, true)
                .as_ref()
                .i((0, 0, ..))
        );

        Ok(())
    }

    // #[test] TODO
    fn test_apply_motif_nets_padding_max() -> Result<(), ThoosaError> {
        set_max_level(Trace);
        set_panic_hook();
        reset_print_options();

        let input_n = 1000;
        let motif_width = 100u32;
        let step = (motif_width / 1) as usize; //1 - step==motif_width slices do not overlap so net output - sum(slice)*k

        //batch 2 x channels 5 x n 1000
        // let batch = Tensor::stack(&[
        let t_0 = lin_test_tensor(input_n, 5);
        let t_1 = lin_test_tensor(input_n / 2, 5);
        // ], 0);

        let store = Arc::new(RwLock::new(mock_store_with_find_motifs_run_config(
            step as i32,
            motif_width as i32,
            vec![
                vec![(0, 1.0), (1, 1.0)],
                vec![(0, 1.0)], //motif id == index in mock vec
            ],
        )));

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 2, 100, input_n as u32)
                .with_load_motif_nets(vec![
                    vec![
                        //channel 0
                        utils::motif_test_net(motif_width, 1., Cpu),
                        utils::motif_test_net(motif_width, 2., Cpu),
                    ],
                    vec![
                        //channel 1
                        utils::motif_test_net(motif_width, 10., Cpu),
                    ],
                ])
                .build(),
        ));
        let channels_motifs: Vec<Vec<Arc<LoadMotifNetOutput>>> = block_on(load_channels_motifs(
            1,
            &ws,
            &store,
            Cuda(0),
            &dummy_runner(),
            None,
        ));
        let applicable = MotifsNetsSelectionApplicable::new(1, &store, &ws, Cpu, channels_motifs);

        let t_0_applied = applicable.run(t_0.into(), ())?;

        debug!("t_0_applied.size={:?}", t_0_applied.size());
        trace!("t_0_0_applied=\n{}", t_0_applied.as_ref().i((0, ..)));
        trace!("t_0_1_applied=\n{}", t_0_applied.as_ref().i((1, ..)));

        let channel_0_motifs = Tensor::stack(
            &[
                //sum 0-100  100-200 ...
                Tensor::from_slice(&[
                    4950., 14950., 24950., 34950., 44950., 54950., 64950., 74950., 84950., 94950.,
                ]), //ch 0 net 0 (ResourceId=1)
                //sum 0-100  100-200 ... * 2 - k in second net in channel 0/**/
                Tensor::from_slice(&[
                    4950., 14950., 24950., 34950., 44950., 54950., 64950., 74950., 84950., 94950.,
                ]) * 2., // ch. 0 net 1 (ResourceId=2)
            ],
            0,
        )
        .totype(Float)
        .view((1, 2, 10));

        assert_eq!(
            channel_0_motifs,
            t_0_applied.as_ref().i((0, ..)).view((1, 2, 10))
        );

        let channel_1_motifs = Tensor::stack(
            &[
                //sum 0-100  100-200 ...
                //sum 0-100  100-200 ... * 10 - k in second net in channel 0/**/
                Tensor::from_slice(&[
                    4950., 14950., 24950., 34950., 44950., 54950., 64950., 74950., 84950., 94950.,
                ]) * 2.
                    * 10., // ch. 0 net 1 (ResourceId=2) first * 2. because lin_test_tensor multiplies each chanel by its ordinal
                Tensor::from_slice(&[
                    4950., 14950., 24950., 34950., 44950., 54950., 64950., 74950., 84950., 94950.,
                ]) * 0., // ch. 0 net 1 (ResourceId=2)
            ],
            0,
        )
        .totype(Float)
        .view((1, 2, 10));

        assert_eq!(
            channel_1_motifs,
            t_0_applied.as_ref().i((1, ..)).view((1, 2, 10))
        );

        let t_1_applied = applicable.run(t_1.into(), ())?;

        debug!("t_1_applied.size={:?}", t_1_applied.size());
        trace!("t_1_applied=\n{}", t_1_applied.as_ref().i((0, ..)));
        trace!("t_1_applied=\n{}", t_1_applied.as_ref().i((1, ..)));

        let channel_0_motifs = Tensor::stack(
            &[
                //sum 0-100  100-200 ...
                Tensor::from_slice(&[4950., 14950., 24950., 34950., 44950., 0., 0., 0., 0., 0.]), //ch 0 net 0 (ResourceId=1)
                //sum 0-100  100-200 ... * 2 - k in second net in channel 0/**/
                Tensor::from_slice(&[4950., 14950., 24950., 34950., 44950., 0., 0., 0., 0., 0.])
                    * 2., // ch. 0 net 1 (ResourceId=2)
            ],
            0,
        )
        .totype(Float)
        .view((1, 2, 10));

        assert_eq!(
            channel_0_motifs,
            t_1_applied.as_ref().i((0, ..)).view((1, 2, 10))
        );

        let channel_1_motifs = Tensor::stack(
            &[
                //sum 0-100  100-200 ...
                //sum 0-100  100-200 ... * 10 - k in second net in channel 0/**/
                Tensor::from_slice(&[4950., 14950., 24950., 34950., 44950., 0., 0., 0., 0., 0.])
                    * 2.
                    * 10., // ch. 0 net 1 (ResourceId=2) first * 2. because lin_test_tensor multiplies each chanel by its ordinal
                Tensor::from_slice(&[4950., 14950., 24950., 34950., 44950., 0., 0., 0., 0., 0.])
                    * 0., // ch. 0 net 1 (ResourceId=2)
            ],
            0,
        )
        .totype(Float)
        .view((1, 2, 10));

        assert_eq!(
            channel_1_motifs,
            t_1_applied.as_ref().i((1, ..)).view((1, 2, 10))
        );

        Ok(())
    }

    #[test]
    fn test_apply_motif_nets_padding_fixed() -> Result<(), ThoosaError> {
        set_max_level(Trace);
        reset_print_options();
        set_panic_hook();

        let input_n = 1000;
        let motif_width = 100u32;
        let step = (motif_width / 1) as usize; //1 - step==motif_width slices do not overlap so net output - sum(slice)*k

        //batch 2 x channels 5 x n 1000
        let t_0 = lin_test_tensor(input_n, 5);
        let t_1 = lin_test_tensor(input_n / 2, 5);

        let store = Arc::new(RwLock::new(mock_store_with_find_motifs_run_config(
            step as i32,
            motif_width as i32,
            vec![
                vec![(0, 1.0), (1, 1.0)],
                vec![(2, 1.0)], //2 used as index in with_label_nets
            ],
        )));

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 5, 100, input_n as u32)
                .with_load_motif_nets(vec![
                    vec![
                        //channel 0
                        utils::motif_test_net(motif_width, 1., Cpu),
                        utils::motif_test_net(motif_width, 2., Cpu),
                    ],
                    vec![
                        //channel 1
                        utils::motif_test_net(motif_width, 0., Cpu), //dummy
                        utils::motif_test_net(motif_width, 0., Cpu), //dummy
                        utils::motif_test_net(motif_width, 10., Cpu), //taken in with_label_nets by motif_id as index
                    ],
                ])
                .build(),
        ));
        let channels_motifs = block_on(load_channels_motifs(
            1,
            &ws,
            &store,
            Cuda(0),
            &dummy_runner(),
            None,
        ));
        let applicable = MotifsNetsSelectionApplicable::new(1, &store, &ws, Cpu, channels_motifs);

        let t_0_applied = applicable.run(t_0.into(), 11)?;

        debug!("t_0_applied.size={:?}", t_0_applied.size());
        trace!("t_0_applied=\n{}", t_0_applied);

        let channel_0_motifs = Tensor::from_slice(&[
            4950., 13950., 22950., 31950., 40950., 49950., 58950., 67950., 76950., 85950., 94950.,
            9900., 27900., 45900., 63900., 81900., 99900., 117900., 135900., 153900., 171900.,
            189900., 99000., 279000., 459000., 639000., 819000., 999000., 1179000., 1359000.,
            1539000., 1719000.,
            1899000., // lin_test_tensor multiplies each chanel by its ordinal + *10 by net
        ])
        .totype(Float)
        .view((3, 11));

        assert_eq!(&channel_0_motifs, t_0_applied.as_ref());

        let t_1_applied = applicable.run(t_1.into(), 11)?;

        debug!("t_1_applied.size={:?}", t_1_applied.size());
        trace!("t_1_applied=\n{}", t_1_applied);

        let channel_1_motifs = Tensor::from_slice(&[
            4950., 8950., 12950., 16950., 20950., 24950., 28950., 32950., 36950., 40950., 44950.,
            9900., 17900., 25900., 33900., 41900., 49900., 57900., 65900., 73900., 81900., 89900.,
            99000., 179000., 259000., 339000., 419000., 499000., 579000., 659000., 739000.,
            819000., 899000.,
        ])
        .totype(Float)
        .view((3, 11));

        assert_eq!(&channel_1_motifs, t_1_applied.as_ref());

        Ok(())
    }

    // #[test] TODO
    fn test_apply_label_nets_padding_fixed() -> Result<(), ThoosaError> {
        set_panic_hook();
        set_max_level(Trace);
        reset_print_options();

        let input_n = 1000;
        let motif_width = 100u32;
        let step = 10; //1 - step==motif_width slices do not overlap so net output - sum(slice)*k

        //batch 2 x channels 5 x n 1000
        let t_0 = lin_test_tensor(input_n, 5);
        let t_1 = lin_test_tensor(input_n / 2, 5);
        // let batch = Tensor::stack(&[t_0, t_1], 0);

        let runner = dummy_runner();

        let applicable = block_on(LabelNetsApplicable::load_all_filtered_by_run_ids(
            Some(&[1, 2]),
            vec!["Label_0".to_string(), "Label_1".to_string()],
            &Arc::new(RwLock::new(mock_store_with_find_motifs_run_config(
                step as i32,
                motif_width as i32,
                vec![vec![(0, 1.0), (1, 1.0)], vec![(2, 1.0)]],
            ))),
            &Arc::new(RwLock::new(
                WsMockBuilder::new(None, 5, 100, input_n as u32)
                    .with_load_motif_nets(vec![
                        vec![
                            //channel 0
                            utils::motif_test_net(motif_width, 1., Cpu),
                            utils::motif_test_net(motif_width, 2., Cpu),
                        ],
                        vec![
                            //channel 1
                            utils::motif_test_net(motif_width, 0., Cpu), //dummy
                            utils::motif_test_net(motif_width, 0., Cpu), //dummy
                            utils::motif_test_net(motif_width, 10., Cpu), //taken in with_label_nets by motif_id as index
                        ],
                    ])
                    .with_label_nets(HashMap::from([
                        (
                            "Label_0".to_string(),
                            label_test_nets(
                                100,
                                1.,
                                LabelNetTypeParams::FixedNumSlices(FixedNumSlicesNetParams::new()),
                                Cpu,
                            ),
                        ),
                        (
                            "Label_1".to_string(),
                            label_test_nets(
                                100,
                                10.,
                                LabelNetTypeParams::FixedNumSlices(FixedNumSlicesNetParams::new()),
                                Cpu,
                            ),
                        ),
                    ]))
                    .build(),
            )),
            Cpu,
            &dummy_runner(),
        ));

        let t_0_applied = applicable.run(t_0.unsqueeze(0).into(), ())?;

        trace!("t_0_applied=\n{t_0_applied}");
        assert!(
            t_0_applied.as_ref().allclose(
                &Tensor::from_slice(&[
                    4.9810e7_f32,
                    9.9620e7,
                    9.9619e8,
                    4.9810e6,
                    9.9620e6,
                    9.9620e7
                ])
                .view((1, 2, 3, 1)),
                1e2,
                1e2,
                false
            )
        );
        Ok(())
    }

    fn dummy_runner() -> Arc<ThoRunner> {
        Arc::new(ThoRunner::new(&[1, 1], move |key, _| {
            ThoTensor::new(Input::id(key).clone(), Tensor::rand([100, 3], FLOAT_CPU))
        }))
    }

    // #[test] TODO
    fn test_apply_label_nets_max_padding() -> Result<(), ThoosaError> {
        set_max_level(Debug);
        reset_print_options();

        let input_n = 1000;
        let motif_width = 100u32;
        let step = 10; //1 - step==motif_width slices do not overlap so net output - sum(slice)*k
        let slices = (input_n - motif_width) / step + 1;

        //batch 2 x channels 2 x n 1000
        let t_0 = lin_test_tensor(input_n as i64, 2);
        // let t_1 = lin_test_tensor(input_n as i64 / 2, 2);
        // let batch = Tensor::stack(&[t_0, t_1], 0);

        let applicable = block_on(LabelNetsApplicable::load_all_filtered_by_run_ids(
            Some(&[1, 2]),
            vec!["Label_0".to_string(), "Label_1".to_string()],
            &Arc::new(RwLock::new(mock_store_with_find_motifs_run_config(
                step as i32,
                motif_width as i32,
                vec![vec![(0, 1.0), (1, 1.0)], vec![(2, 1.0)]],
            ))),
            &Arc::new(RwLock::new(
                WsMockBuilder::new(None, 2, 100, input_n)
                    .with_load_motif_nets(vec![
                        vec![
                            //channel 0
                            utils::motif_test_net(motif_width, 1., Cpu),
                            utils::motif_test_net(motif_width, 2., Cpu),
                        ],
                        vec![
                            //channel 1
                            utils::motif_test_net(motif_width, 0., Cpu), //dummy
                            utils::motif_test_net(motif_width, 0., Cpu), //dummy
                            utils::motif_test_net(motif_width, 10., Cpu), //taken in with_label_nets by motif_id as index
                        ],
                    ])
                    .with_label_nets(HashMap::from([
                        (
                            "Label_0".to_string(),
                            label_test_nets(
                                slices,
                                1.,
                                LabelNetTypeParams::MaxPadding(MaxPaddingNetParams::new()),
                                Cpu,
                            ),
                        ),
                        (
                            "Label_1".to_string(),
                            label_test_nets(
                                slices,
                                10.,
                                LabelNetTypeParams::MaxPadding(MaxPaddingNetParams::new()),
                                Cpu,
                            ),
                        ),
                    ]))
                    .build(),
            )),
            Cpu,
            &dummy_runner(),
        ));

        let tho_tensor = applicable.run(t_0.unsqueeze(0).into(), ())?;
        let t_0_applied = tho_tensor.as_ref();
        debug!("t_0_applied={t_0_applied:?}");
        trace!("t_0_applied=\n{t_0_applied}");

        assert!(t_0_applied.view((1, -1)).allclose(
            &Tensor::from_slice(&[
                4.5454e6_f32,
                9.0909e6,
                9.0909e7,
                0.0000e0,
                4.5454e7,
                9.0909e7,
                9.0909e8,
                0.0000e0
            ]),
            1e3,
            1e3,
            false
        ));

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
                                        test_label_net(num_features, k, device, "test_label_net").unwrap(),
                )
            }),
        )]
    }

    #[test]
    fn s_low_test_apply_motif_models() -> Result<(), ThoosaError> {
        // log::set_max_level(Trace);
        // let _device = CudaDevice::new(0)?;
        unsafe {
            let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        let (free_mem, total_mem) = cuda_mem();
        debug!("Device 0 free_mem={free_mem} total_mem={total_mem}");

        let input_n = (free_mem + free_mem / 2) as u32;
        debug!("input_n={input_n}");
        let motif_width = 20;
        let input_slice_step = 10;
        let input_slices_motif_batch_size = 5;

        let max_dists_len = slices_len(input_n, motif_width, input_slice_step);

        let channel = lin_test_tensor(input_n as i64, 1).to_device(Cpu);
        let unfolded = channel.unfold(1, motif_width as i64, input_slice_step as i64);
        let sums = unfolded
            .sum_dim_intlist(2, true, Float)
            .view(max_dists_len as i64); //sums of the slices should be what test_net produces
        debug!(
            "tensor_size_in_bytes={:?}M ",
            tensor_mem_size(&unfolded) / 1024 / 1024
        );
        trace!("unfolded=\n{unfolded} \nsums=\n{}", sums);

        let store = Arc::new(RwLock::new(mock_store_with_find_motifs_run_config(
            input_slice_step as i32,
            motif_width as i32,
            vec![vec![(0, 1.0)]],
        )));

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 5, 100, input_n as u32)
                .with_load_motif_nets(vec![vec![
                    //channel 0
                    utils::motif_test_net(motif_width, 1., Cuda(0)),
                ]])
                .build(),
        ));

        let channels_motifs = block_on(load_channels_motifs(
            1,
            &ws,
            &store,
            Cuda(0),
            &dummy_runner(),
            None,
        ));

        let applicable =
            MotifsNetsSelectionApplicable::new(1, &store, &ws, Cuda(0), channels_motifs);

        let out = applicable.run(channel.into(), ())?;

        // let channel_1_t = apply_channel_motifs(&mut motif_nets, channel, motif_width, 1, max_dists_len, &Cuda(0))?.view(max_dists_len as i64);

        debug!("sums=\n{sums:?}");
        trace!("sums=\n{sums}");

        debug!("out=\n{out:?}");
        trace!("out=\n{out}");

        assert!(sums.allclose(&out.as_ref().i((0, 0, ..)), 100.0, 100.0, true));

        Ok(())
    }

    // #[test] TODO
    fn test_apply_motif_models_diff_len() -> Result<(), ThoosaError> {
        let input_n_0 = 1000;
        let motif_width = 120;
        let t_0 = lin_test_tensor(input_n_0, 3); //.unsqueeze(0);
        let t_1 = lin_test_tensor(831, 3); //.unsqueeze(0);

        debug!(" t_0.size()={:?}, t_1.size()={:?}", t_0.size(), t_1.size());

        let motif_batch_size = 13;
        let step = 100;

        let store = Arc::new(RwLock::new(mock_store_with_find_motifs_run_config(
            step,
            motif_width as i32,
            vec![vec![(0, 1.0), (1, 1.0)], vec![(0, 1.0)]],
        )));

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 3, 100, input_n_0 as u32)
                .with_load_motif_nets(vec![
                    vec![
                        //channel 0
                        utils::motif_test_net(motif_width, 1., Cpu),
                        utils::motif_test_net(motif_width, 1., Cpu),
                    ],
                    vec![
                        //channel 1
                        utils::motif_test_net(motif_width, 0., Cpu),
                        utils::motif_test_net(motif_width, 0., Cpu),
                        utils::motif_test_net(motif_width, 10., Cpu),
                    ],
                    vec![],
                ])
                .build(),
        ));

        let channels_motifs = block_on(load_channels_motifs(
            1,
            &ws,
            &store,
            Cuda(0),
            &dummy_runner(),
            None,
        ));
        let applicable = MotifsNetsSelectionApplicable::new(1, &store, &ws, Cpu, channels_motifs);

        let channels_motifs_0 = applicable.run(t_0.into(), ())?;

        // let channels_motifs_0 = apply_motif_nets(&t_0, Arc::clone(&motif_nets), &Cpu, input_n_0 as u32);

        debug!("channels_motifs_0.size={:?}", channels_motifs_0.size());
        let channels_motifs_1 = applicable.run(t_1.into(), ())?;

        debug!("channels_motifs_1.size={:?}", channels_motifs_1.size());

        assert_eq!([2, 2, 9], channels_motifs_0.size()[..]); //channel, motif nets, batches, batch size
        assert_eq!([2, 2, 9], channels_motifs_1.size()[..]); //channel, motif nets, batches, batch size

        Ok(())
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


    //TODO split into tests for FixedNumSlices and MaxPadding
    #[test]
    fn test_train_label() -> Result<(), ThoosaError> {
        set_max_level(Debug);
        const CHANNELS: usize = 2;

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, CHANNELS as u32, 100, 1100u32)
                .with_load_motif_nets(vec![
                    vec![
                        //channel 0
                        utils::motif_test_net(100, 1., Cpu),
                        utils::motif_test_net(100, 1., Cpu),
                        utils::motif_test_net(100, 1., Cpu),
                        utils::motif_test_net(100, 1., Cpu),
                        utils::motif_test_net(100, 1., Cpu),
                    ],
                    vec![
                        //channel 0
                        utils::motif_test_net(100, 1., Cpu),
                        utils::motif_test_net(100, 1., Cpu),
                        utils::motif_test_net(100, 1., Cpu),
                    ],
                ])
                .build(),
        ));

        const BATCH_SIZE: usize = 10;
        const INPUT_SIZE: usize = 1100;

        const MAX_MOTIF_NETS: usize = 5;
        const TRAIN_LABELS: [bool; BATCH_SIZE] = build_labels::<BATCH_SIZE>();
        const VALID_LABELS: [bool; BATCH_SIZE] = build_labels::<BATCH_SIZE>();

        const MAX_SLICES_LEN: usize = slices_len(INPUT_SIZE as u32, 100, 100) as usize;

        #[derive(Debug)]
        struct DummyLabelBatch {
            applicable: LabelNetApplicableAndNetE,
        }

        impl DummyLabelBatch {
            pub fn new(applicable: LabelNetApplicableAndNetE) -> Self {
                Self { applicable }
            }
        }

        impl DataBatch<LabelNetInputTensors> for DummyLabelBatch {
            fn train_batch(&self) -> (ThoBatch<LabelNetInputTensors>, Tensor) {
                (
                    dummy_motif_net_input(
                        &TRAIN_LABELS,
                        CHANNELS,
                        INPUT_SIZE,
                        Some(Box::new(|t| {
                            self.applicable
                                .apply(&t.into())
                                .unwrap()
                                .as_ref()
                                .shallow_clone()
                        })),
                    )
                    .into(),
                    Tensor::from_slice(TRAIN_LABELS.as_slice()).totype(Float),
                )
            }

            fn valid_batch(&self) -> (ThoBatch<LabelNetInputTensors>, Tensor) {
                (
                    dummy_motif_net_input(&TRAIN_LABELS, CHANNELS, INPUT_SIZE, None).into(),
                    Tensor::from_slice(VALID_LABELS.as_slice()).totype(Float),
                )
            }
        }

        let store = Arc::new(RwLock::new(mock_store_with_find_motifs_run_config(
            100,
            100,
            vec![
                vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)],
                vec![(0, 1.0), (1, 1.0), (2, 1.0)],
            ],
        )));
        let channels_motifs = block_on(load_channels_motifs(
            1,
            &ws,
            &store,
            Cuda(0),
            &dummy_runner(),
            None,
        ));
        let motifs_nets_selection_applicable_arc = Arc::new(MotifsNetsSelectionApplicable::new(
            1,
            &store,
            &ws,
            Cpu,
            channels_motifs,
        ));

        impl Splittable<1> for DummyLabelBatch {
            fn split(&self) -> Option<[Self; 1]>
            where
                Self: Sized,
            {
                None //Some([DummyLabelBatch::new()])
            }
        }

        impl SplittableDataBatch<LabelNetInputTensors> for DummyLabelBatch {}

        let run_config = TrainLabelNetsRunConfig::new(
            Some(1),
            CommonTrainingRunConfig::new(10, 10, 0.1, 0.1, 10, 3),
            LabelNetTypeArgs::MaxPadding(MaxPaddingNetArgs::new(3)),
        );

        let applicable_and_net_enum = LabelNetApplicableAndNetE::new(
            &run_config.net,
            &motifs_nets_selection_applicable_arc,
            &ws,
        );

        let vs = VarStore::new(Cpu);
        let loss = super::train_label_net(
            &applicable_and_net_enum.module(&vs)?,
            &run_config,
            &mut DummyLabelBatch::new(applicable_and_net_enum),
            &vs,
            &ws,
            &Cpu,
        )?;

        debug!("test_train_label: MaxPadding loss={loss:?} run_config={run_config:?}");

        assert!(loss < 1.0);

        let run_config = TrainLabelNetsRunConfig::new(
            Some(1),
            CommonTrainingRunConfig::new(10, 10, 0.1, 0.1, 10, 3),
            LabelNetTypeArgs::FixedNumSlices(FixedNumSlicesNetArgs {
                num_layers: 3,
                slices_len: MAX_SLICES_LEN as u32,
            }), //MAX_SLICES_LEN to match DummyLabelBatch
        );

        let applicable_and_net_enum = LabelNetApplicableAndNetE::new(
            &run_config.net,
            &motifs_nets_selection_applicable_arc,
            &ws,
        );

        let vs = VarStore::new(Cpu);
        let loss = super::train_label_net(
            &applicable_and_net_enum.module(&vs)?,
            &run_config,
            &mut DummyLabelBatch::new(applicable_and_net_enum),
            &vs,
            &ws,
            &Cpu,
        )?;

        debug!("test_train_label: FixedNumSlices loss={loss:?} run_config={run_config:?}");

        assert!(loss < 1.0);

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

    #[test]
    fn test_train_basic_oom() -> Result<(), ThoosaError> {
        reset_print_options();
        set_max_level(Debug);
        // let _device = CudaDevice::new(0)?;
        let cudarc_lib = unsafe {
            let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        };
        // let batch_size = 10000;
        // const FEATURES:i64 = 2i64.pow(10);

        let batch_size = 10;
        const FEATURES: u32 = 10;

        let res_seq_0 = SeqBuilder::new(FEATURES, None)
            // .func(|t|{
            //     debug!("res_seq_0: t={t:?}");
            //     t.shallow_clone()
            // })
            .batch_norm(BatchNormConfig::default())
            .linear_s(
                FEATURES * 20,
                #[cfg(debug_assertions)]
                "test".to_string(),
            )
            .func(Tensor::relu)
            .linear_s(
                FEATURES * 16,
                #[cfg(debug_assertions)]
                "test".to_string(),
            )
            .func(Tensor::relu)
            .linear_s(
                10,
                #[cfg(debug_assertions)]
                "test".to_string(),
            )
            // .func(Tensor::sigmoid)
            .end_seq();

        let res_seq_1 = SeqBuilder::new(FEATURES, None)
            // .func(|t|{
            //     debug!("res_seq_1: t={t:?}");
            //     t.shallow_clone()
            // })
            .batch_norm(BatchNormConfig::default())
            .linear_s(
                FEATURES * 20,
                #[cfg(debug_assertions)]
                "test".to_string(),
            )
            .func(Tensor::relu)
            .linear_s(
                FEATURES * 16,
                #[cfg(debug_assertions)]
                "test".to_string(),
            )
            .func(Tensor::relu)
            .linear_s(
                10,
                #[cfg(debug_assertions)]
                "test".to_string(),
            )
            // .func(Tensor::sigmoid)
            .end_seq();

        let full = SeqBuilder::new(FEATURES, None)
            .split(
                vec![res_seq_0, res_seq_1],
                Box::new(move |t: &Tensor| {
                    debug!("split: t={t:?}");
                    vec![t.i((.., 0, ..)), t.i((.., 1, ..))]
                }), //label batch, channel, max_motif_nets, max slices
                Box::new(move |t: Vec<Tensor>| {
                    debug!("join: in t={t:?}");
                    let t = Tensor::concat(t.as_slice(), 1);
                    debug!("join: t={t:?}");
                    t
                }),
            )
            .linear_feats(
                20,
                20,
                #[cfg(debug_assertions)]
                "test".to_string(),
            )
            .func(Tensor::relu)
            .linear_feats(
                20,
                1,
                #[cfg(debug_assertions)]
                "test".to_string(),
            )
            .func(Tensor::sigmoid)
            .end_seq();

        let mut vs_0 = VarStore::new(Cuda(0));

        let root = vs_0.root();
        let net_full: Box<dyn ThoModule<AnyDims, AnyDims>> = full.build_tch_module(&root)?; //&(root / "vs_0"))?;

        debug!("vs_0={vs_0:?}");

        let kind = Kind::BFloat16;
        // let kind = Kind::Double;
        vs_0.root().set_kind(kind);

        // vs_0.root() /"seq_0"
        let mut vs_opt_main = VarStore::new(Cuda(0));
        let mut vs_opt_split_0 = VarStore::new(Cuda(0));
        let mut vs_opt_split_1 = VarStore::new(Cuda(0));
        for (name, t) in vs_0.variables() {
            debug!("adding name={name}");
            if name.contains("split_0") {
                _ = vs_opt_split_0
                    .root()
                    .add(&name.replace(".", "/"), t, !name.contains("batch"));
            } else if name.contains("split_1") {
                _ = vs_opt_split_1
                    .root()
                    .add(&name.replace(".", "/"), t, !name.contains("batch"));
            } else {
                _ = vs_opt_main
                    .root()
                    .add(&name.replace(".", "/"), t, !name.contains("batch"));
            }
        }

        debug!("vs_opt_main={vs_opt_main:?}");
        debug!("vs_opt_split_0={vs_opt_split_0:?}");
        debug!("vs_opt_split_1={vs_opt_split_1:?}");

        // let mut opt_0 = AdamW::default().beta1(0.9).beta2(0.999).build(&(vs_0.root() /"seq_0").into(), 1e-3)?;
        let mut opt_main = AdamW::default()
            .beta1(0.9)
            .beta2(0.999)
            .build(&vs_opt_main, 1e-3)?;
        let mut opt_split_0 = AdamW::default()
            .beta1(0.9)
            .beta2(0.999)
            .build(&vs_opt_split_0, 1e-3)?;
        let mut opt_split_1 = AdamW::default()
            .beta1(0.9)
            .beta2(0.999)
            .build(&vs_opt_split_1, 1e-3)?;

        // vs_0.freeze();
        // set_requires_grad_in_subpath(&vs_0, "seq_0", false);
        // set_requires_grad_in_subpath(&vs_0, "seq_1", false);
        // set_requires_grad_in_subpath(&vs_0, "seq_3", false);
        // set_requires_grad_in_subpath(&vs_0, "seq_0.split_1", false);
        let mut split_i = 0;
        for epoch in 0..1000 {
            // if epoch % 2 == 0 {
            //     set_requires_grad_in_subpath(&vs_0, "seq_0.split_1", false);
            //     set_requires_grad_in_subpath(&vs_0, "seq_0.split_0", true);
            // } else {
            //     set_requires_grad_in_subpath(&vs_0, "seq_0.split_1", true);
            //     set_requires_grad_in_subpath(&vs_0, "seq_0.split_0", false);
            // }

            let mut rng = rand::rng();

            let (input_0, input_1, target): (Vec<Tensor>, Vec<Tensor>, Vec<f32>) = (0..batch_size)
                .map(|i| {
                    let mut data = [0.; FEATURES as usize];
                    rng.fill(&mut data);

                    if rng.random_bool(0.5) {
                        (asc_t(data), desc_t(data), 0.)
                    } else {
                        (desc_t(data), asc_t(data), 1.)
                    }
                })
                .fold(
                    (Vec::new(), Vec::new(), Vec::new()),
                    |mut a, (i_0, i_1, label)| {
                        a.0.push(i_0);
                        a.1.push(i_1);
                        a.2.push(label);
                        a
                    },
                );

            let input_0 = Tensor::stack(input_0.as_slice(), 0).to_device(Cuda(0));
            let input_1 = Tensor::stack(input_1.as_slice(), 0).to_device(Cuda(0));
            let input = Tensor::stack(&[input_0, input_1], 0).transpose(0, 1);

            trace!("input=\n{input}");

            let target = Tensor::from_slice(target.as_slice())
                .view((-1, 1))
                .to_device(Cuda(0));

            trace!("target=\n{target}");

            let pred = net_full.tho_forward(&input.totype(kind).into(), true);

            let loss = pred.as_ref().binary_cross_entropy(
                &target.totype(kind),
                None::<Tensor>,
                Reduction::Mean,
            );
            if split_i == 0 {
                //TODO off-load to CPU
                opt_split_0.backward_step(&loss);
            } else if split_i == 1 {
                opt_split_1.backward_step(&loss);
            } else {
                opt_main.backward_step(&loss);
            }

            // debug!("input=\n{input}");
            // debug!("target=\n{target}");
            // debug!("pred=\n{pred}");
            if epoch % 50 == 0 {
                info!("loss={loss} cuda_mem={:?}", cuda_mem());
            }
            split_i += 1;
            if split_i > 2 {
                split_i = 0;
            }
        }
        debug!(
            "opt_0 optimizer_mem_size: {:?}",
            optimizer_mem_size(&opt_main)
        );
        // debug!("cuda_mem: {:?}", tch::Cuda::memory_allocated());

        vs_0.freeze();
        vs_0.set_kind(Float);

        // debug!("opt_1 optimizer_mem_size: {:?}", optimizer_mem_size(&opt_1));

        let mut rng = rand::rng();

        // let mut results = vec![];
        for i in 0..100 {
            let mut data = [0.; FEATURES as usize];
            rng.fill(&mut data);

            let pred_0 = net_full.tho_forward(
                &Tensor::stack(
                    &[
                        Tensor::stack(&[asc_t(data), desc_t(data)], 0), //0
                                                                        // Tensor::stack(&[desc_t(data), asc_t(data)], 0), //1
                    ],
                    1,
                )
                .transpose(0, 1)
                .to_device(Cuda(0))
                .into(),
                false,
            ); //0
            info!("pred_0=\n{pred_0}");

            let pred_1 = net_full.tho_forward(
                &Tensor::stack(
                    &[
                        // Tensor::stack(&[asc_t(data), desc_t(data)], 0), //0
                        Tensor::stack(&[desc_t(data), asc_t(data)], 0), //1
                    ],
                    1,
                )
                .transpose(0, 1)
                .to_device(Cuda(0))
                .into(),
                false,
            ); //1
            info!("pred_1=\n{pred_1}");

            assert!(pred_0.as_ref().squeeze().double_value(&[]) < 0.5);
            assert!(pred_1.as_ref().squeeze().double_value(&[]) > 0.5);

            // results.push((
            //     pred_0.squeeze().double_value(&[]) < 0.5,
            //     pred_1.squeeze().double_value(&[]) > 0.5
            // ));
        }
        // info!("results={results:?}");

        // net_0
        /* let mut data = [0.; FEATURES as usize];
        for _ in 0..10 {
            rng.fill(&mut data);
            let pred_input = Tensor::stack(&[
                Tensor::stack(&[asc_t(data), desc_t(data)], 0), //0
                // Tensor::stack(&[desc_t(data), asc_t(data)], 0), //1
            ], 1).transpose(0, 1).to_device(Cuda(0));

            debug!("pred_input={pred_input:?}");
            let pred_batch = net_full.tho_forward(&pred_input, false);
            info!("pred_batch=\n{pred_batch}");
        }*/
        // for i in 0..10 {
        //
        // }

        // let features_num = 400_000_000;
        // let vs = VarStore::new(Cuda(0));
        // debug!("cuda_mem={:?}", cuda_mem());
        //
        // let net = test_net_vs(features_num, 1., &vs)?;
        // debug!("pred={net:?} vs_mem_size={:?} cuda_mem={:?}", vs_mem_size(&vs), cuda_mem());
        //
        // let input_t = Tensor::rand([10, features_num], FLOAT_CUDA);
        // let mut pred = net.forward(&input_t);
        //
        // set_max_level(Trace);
        // trace!("pred=\n{pred}");
        // debug!("input_t mem_size={:?} cuda_mem={:?}", input_t.mem_size(), cuda_mem());
        //
        // let loss = pred.sigmoid().binary_cross_entropy(&Tensor::rand([10, 1], FLOAT_CUDA), None::<Tensor>, Reduction::Mean);
        //
        // drop(input_t);
        // debug!("input_t dropped cuda_mem={:?}", cuda_mem());
        //
        // debug!("pred={loss:?} mem_size={:?}", loss.mem_size());
        //
        // let vs_mem_size_0 = vs_mem_size(&vs);
        //
        // debug!("vs_mem_size_0={vs_mem_size_0} vs.kind()={:?}", vs.kind());
        //
        // let mut opt = AdamW::default().beta1(0.9).beta2(0.999).amsgrad(true).build(&vs, 1e-10)?;
        //
        // debug!("0: cuda_mem={:?}  optimizer_size={}", cuda_mem(),  optimizer_mem_size(&opt));
        // opt.backward_step(&loss);
        //
        // debug!("1: cuda_mem={:?}  optimizer_size={}", cuda_mem(),  optimizer_mem_size(&opt));

        Ok(())
    }

    // #[test] TODO
    fn test_train_oom() -> Result<(), ThoosaError> {
        let cudarc_lib = unsafe {
            let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        };

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, CHANNELS as u32, 100, 1000u32)
                .with_load_motif_nets(vec![
                    vec![
                        //channel 0
                        utils::motif_test_net(100, 1., Cpu),
                        utils::motif_test_net(100, 1., Cpu),
                        utils::motif_test_net(100, 1., Cpu),
                        utils::motif_test_net(100, 1., Cpu),
                        utils::motif_test_net(100, 1., Cpu),
                    ],
                    vec![
                        //channel 0
                        utils::motif_test_net(100, 1., Cpu),
                        utils::motif_test_net(100, 1., Cpu),
                        utils::motif_test_net(100, 1., Cpu),
                    ],
                ])
                .build(),
        ));

        const BATCH_SIZE: usize = 10;
        const CHANNELS: usize = 2;
        const MAX_MOTIF_NETS: usize = 5;

        const TRAIN_LABELS: [bool; BATCH_SIZE] = build_labels::<BATCH_SIZE>();
        const VALID_LABELS: [bool; BATCH_SIZE] = build_labels::<BATCH_SIZE>();

        const MAX_SLICES_LEN: usize = slices_len(1000, 100, 100) as usize;

        #[derive(Clone, Debug)]
        struct DummyLabelBatch {}

        impl DummyLabelBatch {
            pub fn new() -> Self {
                Self {}
            }
        }

        impl<D: Dims> DataBatch<D> for DummyLabelBatch {
            fn train_batch(&self) -> (ThoBatch<D>, Tensor) {
                (
                    dummy_labels_net_input(
                        &TRAIN_LABELS,
                        CHANNELS,
                        MAX_MOTIF_NETS,
                        MAX_SLICES_LEN,
                        None,
                    )
                    .to_device(Cuda(0))
                    .into(),
                    Tensor::from_slice(TRAIN_LABELS.as_slice())
                        .totype(Float)
                        .to_device(Cuda(0)),
                )
            }

            fn valid_batch(&self) -> (ThoBatch<D>, Tensor) {
                (
                    dummy_labels_net_input(
                        &TRAIN_LABELS,
                        CHANNELS,
                        MAX_MOTIF_NETS,
                        MAX_SLICES_LEN,
                        None,
                    )
                    .to_device(Cuda(0))
                    .into(),
                    Tensor::from_slice(VALID_LABELS.as_slice())
                        .totype(Float)
                        .to_device(Cuda(0)),
                )
            }
        }

        impl Splittable<1> for DummyLabelBatch {
            fn split(&self) -> Option<[Self; 1]>
            where
                Self: Sized,
            {
                Some([DummyLabelBatch::new()])
            }
        }

        impl<D: Dims> SplittableDataBatch<D> for DummyLabelBatch {}

        let batch = DummyLabelBatch::new();

        let store = Arc::new(RwLock::new(mock_store_with_find_motifs_run_config(
            100,
            100,
            vec![
                vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)],
                vec![(0, 1.0), (1, 1.0), (2, 1.0)],
            ],
        )));
        let channels_motifs = block_on(load_channels_motifs(
            1,
            &ws,
            &store,
            Cuda(0),
            &dummy_runner(),
            None,
        ));
        let motifs_nets_selection_applicable_arc = Arc::new(MotifsNetsSelectionApplicable::new(
            1,
            &store,
            &ws,
            Cpu,
            channels_motifs,
        ));

        let run_config = TrainLabelNetsRunConfig::new(
            Some(1),
            CommonTrainingRunConfig::new(10, BATCH_SIZE as u32, 0.1, 0.1, 10, 3),
            LabelNetTypeArgs::MaxPadding(MaxPaddingNetArgs::new(3)),
        );

        let applicable_and_net_enum = LabelNetApplicableAndNetE::new(
            &run_config.net,
            &motifs_nets_selection_applicable_arc,
            &ws,
        );

        let r = opt_train(
            &OptConfig::new(10, 3),
            |vs: &VarStore,
             run_config: &TrainLabelNetsRunConfig,
             listener: &mut dyn TrainListener| {
                let mut r = None::<(f32, LabelNetParams)>;
                run_split_on_oom(
                    Reducer::new(batch.clone()),
                    |batch| {
                        super::train_label_net(
                            &applicable_and_net_enum.module(&vs)?,
                            run_config,
                            batch, //max_nets
                            &VarStore::new(Cuda(0)),
                            &ws,
                            &Cuda(0),
                        )
                    },
                    |v, i| {
                        debug!("v={v:?}");
                        r.replace((v, LabelNetParams::default()));
                    },
                    3,
                )?;

                Ok(r.unwrap())
            },
            &TrainLabelNetsRunConfigRangeOrValue {
                id: None,
                common_training_args: CommonTrainingRunConfigRangeOrValue {
                    cycles: 10.into(),
                    batch_size: 10.into(),
                    lr: (0.01..0.1).into(),
                    max_loss: 0.1f32.into(),
                    opt_max_evals: 10.into(),
                    opt_n_init: 3.into(),
                },
                net: LabelNetTypeArgsRangeOrValue::MaxPadding(MaxPaddingNetArgsRangeOrValue {
                    num_layers: 3.into(),
                })
                .into(),
            }
            .opt_run_configs(),
            Cuda(0),
            "label",
        );

        debug!("test_train_oom: r={r:?}");

        assert!(r.is_ok());

        Ok(())
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
                CommonTrainingRunConfig::new(10, 10, 0.1, 0.1, 10, 3),
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
            train_nets::motif_node(&MotifNetParams::new(window, 1))
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
            &MotifNetParams::new(100, 1),
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
