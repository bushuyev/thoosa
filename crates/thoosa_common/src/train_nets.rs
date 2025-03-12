use std::cell::RefCell;
use crate::builder::{Node, SeqBuilder};
use crate::loader::{LoadRunner, Loader};
use crate::my_tch::ThoModule;
use crate::parzen::optimizer::ConfigurationSpace;
use crate::runs::{Empty, Labeled, Logits, NetResources, Params, RunConfig, SavableOptResult, Simple};
use crate::{panic_to_err, train_nets};
use crate::utils::{cuda_mem, from, run_split_on_oom, tensor_mem_size, Reducer, SplitQueue, Splittable, SplittableDataBatch, SplittableTensor};
use crate::utils::{pad_to, plot_series, train, DataBatch, ListenerResponse, TrainListener};
use crate::ws::{FsWorkspace, Workspace};
use clap::clap_derive::Args;
use crossbeam_utils::sync::WaitGroup;
use futures::FutureExt;
use futures::StreamExt;
use log::{debug, info, trace, warn, Log};
use parking_lot::{Mutex, RawRwLock, RwLock};
use plotters::prelude::BLACK;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use std::{fmt, fs};
use std::hash::Hash;
use std::iter::Map;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;
use tch::nn::{AdamW, ConvConfig, Module, OptimizerConfig, VarStore};
use tch::Device::{Cpu, Cuda};
use tch::Kind::Float;
use tch::{nn, Device, IndexOp, Kind, Reduction, Tensor};
use clap::Subcommand;
use thoosa_shared::{clc_step, parse_key_val, slices_len};
use thoosa_shared::parse_range_or_value;
use thoosa_shared::resource::{Resource, ResourceId};
use thoosa_shared::val::{NamedVal, Val};
use thoosa_shared::val::{prefix_properties, unprefix_properties};
use thoosa_shared::{batch_windows, clc_num_batches, make_combos, RangeOrValue, ThoosaError};
use thoosa_shared::val::Val::UInt;
use thoosa_store::Store;
use std::panic::AssertUnwindSafe;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::run_configs::RunId;

#[derive(Debug, Args, Clone)]
pub struct TrainLabelNetsParams {

    #[arg(long)]
    pub select_motifs_for_label_run_ids: Vec<RunId>,

    #[arg(long)]
    train_resource_group: String,

    #[arg(long)]
    valid_resource_group: String,

    #[command(flatten)]
    common: CommonTrainingArgs,

    #[command(subcommand)]
    pub net: LabelNetType,

}

impl TrainLabelNetsParams {


    pub fn dummy(motif_node_params_selection_run_ids: Vec<RunId>, train_resource_group: String, valid_resource_group: String, batch_size: usize, cycles: usize, lr: impl Into<RangeOrValue<f64>>, num_layers:impl Into<RangeOrValue<u32>>, opt_max_evals: u32, opt_n_init: u32) ->Self {
        TrainLabelNetsParams {
            select_motifs_for_label_run_ids: motif_node_params_selection_run_ids,

            train_resource_group,
            valid_resource_group,

            net: LabelNetType::FixedNumSlices(FixedNumSlicesNetParams {
                slices_len: 100.into(),
                num_layers: 3.into(),
            }),

            common: CommonTrainingArgs {
                batch_size,
                opt_max_evals,
                opt_n_init,
                cycles,
                lr: lr.into(),
                max_loss: 0.1
            }
        }
    }

    //TODO add explicit flag to enable optimization
    pub fn to_config_space(&self) -> Option<ConfigurationSpace<String>> {
        [
            &[
                ("lr".into(), &self.common.lr).try_into().ok(),
                // (&self.num_layers).try_into().ok(),
                // self.net.to_config_space()
            ],
            adam_w_hyperparams().into_iter().map(|v|Some(v)).collect::<Vec<Option<Hyperparameter<String>>>>().as_slice(),
            // &[[LabelNodeType::MaxPadding, LabelNodeType::FixedNumSlices{slices_len:100}][..].try_into().ok()]//TODO LabelNodeType
        ].concat().into_iter()
            .collect::<Option<Vec<Hyperparameter<String>>>>()
            .map(|v|ConfigurationSpace::new(v))
    }
}

#[derive(Debug)]
pub struct TrainLabelNodeRunConfig {
    id:Option<u32>,
    run_config_or_opt_space: ValuesOrOptSpace<TrainLabelNodesValues>
}

impl TrainLabelNodeRunConfig {

    pub fn from_values(id: Option<u32>, lr: f64, num_layers: u32, node_type: LabelNetType) -> Self {
        Self {
            id,
            run_config_or_opt_space: ValuesOrOptSpace::Values(TrainLabelNodesValues::new(node_type, TrainNodesValues {lr, num_layers, adam_w: None})),
        }
    }

    pub fn from_opt_space(id: Option<u32>, opt_space:ConfigurationSpace<String>, opt_config: OptConfig) -> Self {
        Self {
            id,
            run_config_or_opt_space: ValuesOrOptSpace::OptSpace {opt_space, opt_config},
        }
    }
}

impl RunConfig for TrainLabelNodeRunConfig {
    fn to_properties_map(&self) -> HashMap<String, Val> {
        match &self.run_config_or_opt_space {
            ValuesOrOptSpace::Values(values) => {
                // values.to_properties_map()
                todo!()
            },
            ValuesOrOptSpace::OptSpace { .. } => {
                todo!()
            }
        }

    }

    fn from_properties_map(run_id: RunId, properties: HashMap<String, Val>) -> Self {
        todo!()
    }

    fn set_id(&mut self, id: RunId) {
        self.id.replace(id);
    }

    fn type_name() -> &'static str {
        "LabelNode"
    }
}


#[derive(Debug, Clone)]
pub struct TrainLabelNodesValues {
    label_node_type: LabelNetType,//TODO use enum with LabelNodeParams for each ndoe type
    train_nodes_values: TrainNodesValues
}

impl TrainLabelNodesValues {
    pub fn new(label_node_type: LabelNetType, train_nodes_values:TrainNodesValues) -> Self {
        Self{label_node_type, train_nodes_values}
    }
}

impl TryFrom<&[NamedVal<String>]> for TrainLabelNodesValues {
    type Error = ();

    fn try_from(value: &[NamedVal<String>]) -> Result<Self, Self::Error> {
        todo!()
    }
}
impl TryFrom<&Vec<Val>> for TrainLabelNodesValues {
    type Error =ThoosaError;

    fn try_from(value: &Vec<Val>) -> Result<Self, Self::Error> {

        Ok(Self {
            label_node_type: LabelNetType::MaxPadding(MaxPaddingNetParams{ num_layers: 10.into() }),
            train_nodes_values: (&value[..value.len()-1].to_vec()).try_into()?,//TODO to_vec
        })
    }
}

impl Display for TrainLabelNodesValues {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.label_node_type, self.train_nodes_values)
    }
}


#[derive(Debug)]
pub struct CommonTrainLabelNodeRunConfig {
    pub label_batch_size:usize,
    pub cycles:usize,
}

impl CommonTrainLabelNodeRunConfig {

    pub fn new(label_batch_size:usize, cycles:usize) -> Self {
        Self {  label_batch_size, cycles}
    }
}

impl Params<TrainLabelNodeRunConfig, NetResources<Labeled>, CommonTrainLabelNodeRunConfig> for TrainLabelNetsParams {

    fn to_run_configs(&self) -> Vec<TrainLabelNodeRunConfig> {

        if let Some(config_space) = self.to_config_space() {

            debug!("TrainLabelNodeParams::to_run_configs: config_space={config_space:?}");
            vec![
                TrainLabelNodeRunConfig::from_opt_space(None, config_space, OptConfig::new(self.common.opt_max_evals, self.common.opt_n_init))
            ]
        } else {
            //TODO avoid collecting
            let v = make_combos([
                self.common.lr.clone().into_iter().map(|v| v.into()).collect::<Vec<Val>>().into_iter(),
                // self.num_layers.clone().into_iter().map(|v| v.into()).collect::<Vec<Val>>().into_iter(),
            ].into_iter()).into_iter().map(|mut config_combo:Vec<Val>|{
                debug!("TrainLabelNodeParams::to_run_configs: config_combo={config_combo:?} ");
                let num_layers = config_combo.pop().unwrap().try_into().unwrap();
                let lr = config_combo.pop().unwrap().try_into().unwrap();
                TrainLabelNodeRunConfig::from_values(
                    None,
                    lr,
                    num_layers,
                    LabelNetType::FixedNumSlices(FixedNumSlicesNetParams{ num_layers: 3.into(), slices_len:100.into()}), //TODO LabelNodeType
                )
            }).collect();

            debug!("TrainLabelNodeParams::to_run_configs: self.lr={:?} self.num_layers={:?} v={v:?}", self.common.lr, self.net);

            v
        }
    }

    fn resource_config(&self) -> NetResources<Labeled> {
        NetResources::new(
            Labeled::new(self.train_resource_group.clone()), Labeled::new(self.valid_resource_group.clone())
        )
    }

    fn common_config(&self) -> CommonTrainLabelNodeRunConfig {
        CommonTrainLabelNodeRunConfig::new(
            self.common.batch_size,
            self.common.cycles
        )
    }
}

struct BatchTransform<B, F> where B:DataBatch+Debug, F:FnMut(Tensor) -> Tensor + Sync + Send{
    inner: Rc<Box<B>>,
    transform_fn:Rc<RefCell<F>>
}

impl <B, F>  Debug for BatchTransform<B, F> where B:DataBatch+Debug, F:FnMut(Tensor) -> Tensor + Sync + Send {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchTransform")
                 .field("inner", &self.inner)
                 .finish()
    }
}

impl <B, F> BatchTransform<B, F> where B:DataBatch+Debug, F:FnMut(Tensor) -> Tensor + Sync + Send {
    pub fn new(inner: Rc<Box<B>>, transform_fn: F) -> Self {
        Self {
            inner,
            transform_fn:Rc::new(RefCell::new(transform_fn))
        }
    }
    pub fn from_rc(inner:Rc<Box<B>>, transform_fn: Rc<RefCell<F>>) -> Self {
        Self {
            inner,
            transform_fn
        }
    }
}

impl <B, F> DataBatch for BatchTransform<B, F> where B:DataBatch+Debug, F:FnMut(Tensor) -> Tensor + Sync + Send {
    fn train_batch(&self) -> (Tensor, Tensor) {
        let (inputs, labels) = self.inner.train_batch();
        ((self.transform_fn.borrow_mut())(inputs), labels)
    }

    fn valid_batch(&self) -> (Tensor, Tensor) {
        let (inputs, labels) = self.inner.valid_batch();
        ((self.transform_fn.borrow_mut())(inputs), labels)
    }
}

impl <B, F> Splittable<1> for BatchTransform<B, F>  where B:SplittableDataBatch+Debug, F:FnMut(Tensor) -> Tensor + Sync + Send {
    fn split(&self) -> Option<[Self; 1]>
    where
        Self: Sized
    {
        let Some([split]) = self.inner.split() else {
            panic!("BatchTransform::split() called on empty inner");
        };
        Some([BatchTransform::from_rc(Rc::new(Box::new(split)), Rc::clone(&self.transform_fn))])
    }
}

/**
* returns num labels * items_per_label x channels x motifs x motifs batch size
*/
#[derive(Clone)]
struct LoadingTensorsDataBatch  {
    train: Rc<Vec<(i64, Vec <Resource>)>>,
    val: Rc<Vec<(i64, Vec <Resource>)>>,
    batch_size: usize,
    executor: Arc<RwLock<LoadRunner<Box<Tensor>>>>,
    device:Device,
    transform_fn:Option<Arc<RwLock<Box<dyn Fn(Box<Tensor>)->Box<Tensor> + Sync + Send >>>>
}

impl Debug for LoadingTensorsDataBatch {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("")
            .field("train", &self.train)
            .field("val", &self.val)
            .field("batch_size", &self.batch_size)
            .field("device", &self.device)
            .finish()
    }
}

impl LoadingTensorsDataBatch {

    pub fn new(
        train: Rc<Vec<(i64, Vec <Resource>)>>,
        val: Rc<Vec<(i64, Vec <Resource>)>>,
        batch_size: usize,
        executor: Arc<RwLock< LoadRunner<Box<Tensor>>>>,
        device:Device,
        transform_fn:Option<Box<dyn Fn(Box<Tensor>)->Box<Tensor> + Sync + Send >>
    ) -> Self {

        Self {
            train,
            val,
            batch_size,
            executor,
            device,
            transform_fn: transform_fn.map(|f|Arc::new(RwLock::new(f)))
        }
    }

    fn choose_for_batch(class_resources: &Vec<(i64, Vec<Resource>)>, batch_size: usize, loader: Arc<RwLock<LoadRunner<Box<Tensor>>>>, device: &Device, transform_fn:Option<Arc<RwLock<Box<dyn Fn(Box<Tensor>)->Box<Tensor> + Sync + Send >>>>) -> (Tensor, Tensor) {
        debug!("LoadingTensorsDataBatch::choose_for_batch: batch_size={batch_size} class_resources.len={}", class_resources.len());

        let mut rng = thread_rng();
        let wg = WaitGroup::new();
        let area = loader.write().get_available_cache_area();
        let x_y_acc:Arc<Mutex<Vec<(Box<Tensor>, i64)>>> = Arc::new(Mutex::new(vec![]));

        if batch_size <  class_resources.len() {
            debug!("LoadingTensorsDataBatch::choose_for_batch: batch_size <  class_resources.len");

            for _ in 0..batch_size {
                let x_y_acc_in_exec = Arc::clone(&x_y_acc);
                let wg_exec = wg.clone();
                let transform_fn = transform_fn.as_ref().map(|v|Arc::clone(v));

                let (class, resources) = class_resources.choose(&mut rng).expect("class_resources.choose failed");
                let resource_id = resources.choose(&mut rng).expect("resources.choose failed").clone();
                let class = *class;
                Self::load_tensor(&loader, area, x_y_acc_in_exec, wg_exec, transform_fn, resource_id, class);
            }
            wg.wait();
            loader.write().purge_cache(area);
            return Self::vec_to_tensors(device, x_y_acc.lock().drain(..).collect());//TODO drain
        }

        let per_class_len = batch_size / class_resources.len();
        debug!("LoadingTensorsDataBatch::choose_for_batch: class_tensors={} per_class_len={per_class_len}", class_resources.len());


        let mut x_y_vecs_mx = class_resources.iter().fold(Arc::new(Mutex::new(vec![])), |mut x_y_acc, (class, resources)| {
            assert!(resources.len() > 0, "no tensors for class={class} in ClassTensorsDataBatch");

            let class = *class;
            resources.choose_multiple(&mut rng, per_class_len).map(|key|key.clone()).for_each(|resource_id| {

                let x_y_acc_in_exec = Arc::clone(&x_y_acc);
                let wg_exec = wg.clone();
                let transform_fn = transform_fn.as_ref().map(|v|Arc::clone(v));

                Self::load_tensor(&loader, area, x_y_acc_in_exec, wg_exec, transform_fn, resource_id, class);
            });

            x_y_acc
        });

        wg.wait();

        let mut x_y_vecs:Vec<(Box<Tensor>, i64)> = x_y_vecs_mx.lock().drain(..).collect();

        debug!("LoadingTensorsDataBatch::choose_for_batch: x_y_vecs.1.len={} batch_size={batch_size}", x_y_vecs.len());
        loader.read().debug("LoadingTensorsDataBatch::choose_for_batch");

        //TODO throw if not enough tensors for batch?
        (x_y_vecs.len()..batch_size).for_each(|i|{
            x_y_vecs.push(
                (
                    Box::new(x_y_vecs[i- x_y_vecs.len()].0.shallow_clone()),
                    x_y_vecs[i- x_y_vecs.len()].1
                )
            );

        });
        loader.write().purge_cache(area);
        Self::vec_to_tensors(device, x_y_vecs)
    }

    fn load_tensor(loader: &Arc<RwLock<LoadRunner<Box<Tensor>>>>, area: u64, x_y_acc_in_exec: Arc<Mutex<Vec<(Box<Tensor>, i64)>>>, wg_exec: WaitGroup, transform_fn: Option<Arc<RwLock<Box<dyn Fn(Box<Tensor>) -> Box<Tensor> + Sync + Send>>>>, resource_id: Resource, class: i64) {
        loader.write().exec(move |loader, keys_purge| {
            async move {
                let t = Loader::<Box<Tensor>>::load_future(Arc::clone(&loader), resource_id, area).await; //TODO cache area
                let t = if let Some(transform) = transform_fn {
                    transform.read()(t)
                } else {
                    t
                };

                x_y_acc_in_exec.lock().push((t, class));
                drop(wg_exec);
            }.boxed()
        });
    }

    fn vec_to_tensors(device: &Device, mut x_y_vecs: Vec<(Box<Tensor>, i64)>) -> (Tensor, Tensor) {
        let classes = Tensor::from_slice(x_y_vecs.iter().map(|e| e.1).collect::<Vec<_>>().as_slice()).to_device(*device);

        let tensor = Tensor::stack(
            x_y_vecs.into_iter().map(|e| e.0).collect::<Vec<_>>().as_slice(),
            0,
        ).to_device(*device).set_requires_grad(true);

        debug!("LoadingTensorsDataBatch::choose_for_batch: tensor.size={:?} tensor_size_in_bytes={:?} tensor.device()={:?}", tensor.size(), tensor_mem_size(&tensor), tensor.device());

        (
            tensor.totype(Float),
            classes.totype(Float)
        )
    }
}

impl DataBatch for LoadingTensorsDataBatch {

    fn train_batch(&self) -> (Tensor, Tensor){
        tch::no_grad(|| {
            Self::choose_for_batch(&self.train, self.batch_size, Arc::clone(&self.executor), &self.device, self.transform_fn.clone())
        })
    }

    fn valid_batch(&self) -> (Tensor, Tensor){
        tch::no_grad(|| {
            Self::choose_for_batch(&self.val, self.batch_size, Arc::clone(&self.executor), &self.device, self.transform_fn.clone())
        })
    }
}


impl Splittable<1> for LoadingTensorsDataBatch {
    fn split(&self) -> Option<[Self; 1]>
    where
        Self: Sized
    {
        let mut batch = self.clone();
        batch.batch_size = batch.batch_size/2;

        if batch.batch_size > 0 {
            debug!("Splittable<1> for LoadingTensorsDataBatch: batch.batch_size={}", batch.batch_size);
            Some([batch])
        } else {
            None
        }
    }
}

impl SplittableDataBatch for LoadingTensorsDataBatch {}


struct ClassTensorsLoaderData  {
    train_resources: Vec<(Resource, Vec<f32>)>,
    valid_resources: Vec<(Resource, Vec<f32>)>,
    executor: Arc<RwLock< LoadRunner<Box<Tensor>>>>,
    batch_size: usize,
    device:Device,
    transform_fn:Option<Box<dyn Fn(Box<Tensor>)->Box<Tensor> + Sync + Send >>
}

impl Debug for ClassTensorsLoaderData {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClassTensorsLoaderData")
                 .field("train_resources", &self.train_resources)
                 .field("valid_resources", &self.valid_resources)
                 .field("batch_size", &self.batch_size)
                 .field("device", &self.device)
                 .finish()
    }
}

impl ClassTensorsLoaderData  {

    pub fn new(
        train_loader: Vec<(Resource, Vec<f32>)>,
        valid_loader: Vec<(Resource, Vec<f32>)>,
        batch_size: usize,
        executor: Arc<RwLock< LoadRunner<Box<Tensor>>>>,
        device:Device,
        transform_fn:Option<Box<dyn Fn(Box<Tensor>)->Box<Tensor> + Sync + Send >>
    ) -> Self {
        Self {
            train_resources: train_loader,
            valid_resources: valid_loader,
            batch_size,
            executor,
            device,
            transform_fn// transform_fn.map(|f|Arc::new(RwLock::new(f)))
        }
    }

}

impl   DataBatch for ClassTensorsLoaderData {

    fn train_batch(&self) -> (Tensor, Tensor){
        Self::batch(&self.train_resources, self.batch_size, Arc::clone(&self.executor), &self.device, self.transform_fn.as_ref())
    }

    fn valid_batch(&self) -> (Tensor, Tensor){
        Self::batch(&self.valid_resources, self.batch_size, Arc::clone(&self.executor), &self.device, self.transform_fn.as_ref())
    }
}

impl ClassTensorsLoaderData {

    fn batch(resources: &Vec<(Resource, Vec<f32>)>, batch_size: usize, executor: Arc<RwLock< LoadRunner<Box<Tensor>>>>,device: &Device, transform_fn:Option<&Box<dyn Fn(Box<Tensor>)->Box<Tensor> + Sync + Send >>) -> (Tensor, Tensor) {
        debug!("ClassTensorsLoaderData.batch: resources={resources:?} batch_size={batch_size}");

        let x_y_vecs_mx = Arc::new(Mutex::new((vec![], vec![])));
        let wg = WaitGroup::new();

        let x_y_acc_in_exec = Arc::clone(&x_y_vecs_mx);
        let wg_exec = wg.clone();

        // let mut transform_fn = transform_fn.as_ref().map(|v|v.clone());

        let area = executor.write().get_available_cache_area();
        let mut rng = thread_rng();
        let rnd_resoruces = resources.choose_multiple(&mut rng, batch_size).map(|v|v.clone()).collect::<Vec<_>>();
        executor.write().exec(move |loader, keys_purge| {
            async move {
                let mut resources_to_purge = Vec::with_capacity(batch_size);

                for (resource_id, labels_p) in rnd_resoruces {
                    resources_to_purge.push(resource_id.clone());

                    let t =  Loader::<Box<Tensor>>::load_future(Arc::clone(&loader), resource_id.clone(), 0).await;

                    let mut guard = x_y_acc_in_exec.lock();

                    guard.0.push(t);
                    guard.1.push(
                        Box::new(Tensor::from_slice(labels_p.as_slice()))
                    );

                }
                keys_purge(&resources_to_purge, area);//TODO not on each batch
                drop(wg_exec);
            }.boxed()
        });
        wg.wait();

        debug!("ClassTensorsLoaderData::choose_for_batch: x_y_vecs.1.len={} batch_size={batch_size}", x_y_vecs_mx.lock().0.len());

        //TODO throw if not enough tensors for batch?
        let mut guard = x_y_vecs_mx.lock();
        let mut inputs = guard.0.drain(..).collect::<Vec<_>>();
        let mut labels = guard.1.drain(..).collect::<Vec<_>>();
        drop(guard);
        drop(x_y_vecs_mx);

        assert!(labels.len() > 0, "No tensors for batch");
        assert_eq!(labels.len(), inputs.len(), "classes.len() != inputs.len()");

        inputs = if let Some(ref transform) = transform_fn {
            inputs.iter().map(|t| transform(Box::new(t.shallow_clone()))).collect() //TODO par on devices
        } else {
            inputs
        };

        (inputs.len()..batch_size).for_each(|i|{

            let idx = i - inputs.len();
            let tensor_x = inputs[idx].shallow_clone();
            let tensor_y = labels[idx].shallow_clone();
            inputs.push(Box::new(tensor_x));
            labels.push(Box::new(tensor_y));

        });

        (
            Tensor::stack(
                inputs.as_slice(),
                0,
            ).to_device(*device).set_requires_grad(true),
            Tensor::stack(
                labels.as_slice(),
                0,
            ).to_device(*device),
        )
    }
}

#[derive(Debug)]
pub struct TrainLabelNodesRunConfigResult {
    train_label_nodes_run_config_id:u32,
    label:String,
    loss: f32,
    label_node_params: LabelNodeParams,
    train_label_nodes_values: Option<TrainLabelNodesValues>,
    var_store: VarStore
}

impl TrainLabelNodesRunConfigResult {
    pub fn new(train_label_nodes_run_config_id:u32, label:String, loss: f32, label_node_params: LabelNodeParams, train_motifs_nodes_values: Option<TrainLabelNodesValues>, var_store: VarStore) -> Self {
        Self {
            train_label_nodes_run_config_id,
            label,
            loss,
            label_node_params,
            train_label_nodes_values: train_motifs_nodes_values,
            var_store,
        }
    }
}

impl SavableOptResult for TrainLabelNodesRunConfigResult {
    fn value(&self) -> f32 {
        self.loss
    }

    fn save(&mut self, store: &Arc<Mutex<Box<dyn Store>>>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<u32, ThoosaError> {
        let run_config_id = match self.train_label_nodes_values.as_ref(){
            None => self.train_label_nodes_run_config_id,

            Some(train_motifs_nodes_values) => {
                store.lock().save_run_config(
                    Some(&[self.train_label_nodes_run_config_id]),
                    &TrainLabelNodeRunConfig::type_name(),
                    "Label",
                    "todo",
                    TrainLabelNodeRunConfig::from_values(Some(self.train_label_nodes_run_config_id), train_motifs_nodes_values.train_nodes_values.lr, train_motifs_nodes_values.train_nodes_values.num_layers, train_motifs_nodes_values.label_node_type.clone()).to_properties_map()//TODO make from_values to take train_motifs_nodes_values
                )?
            }
        };

        ws.read().save_label_model(run_config_id, &self.label, &self.var_store, &self.label_node_params).expect("save_label_model failed");

        Ok(run_config_id)
    }
}

pub fn run_opt_train_label_nodes (
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    run_config:&TrainLabelNodeRunConfig,
    comon_label_node_params:&CommonTrainLabelNodeRunConfig,
    train_labels_resources:Rc<Vec<(i64, Vec <Resource>)>>, //TODO i64 to 1/0 enum for binary class
    valid_labels_resources: Rc<Vec<(i64, Vec <Resource>)>>,
    device: &Device,
    executor: &Arc<RwLock< LoadRunner<Box<Tensor>>>>,
    channels_motifs_nodes:Arc<RwLock<Vec<(FindMotifsRunConfig, Vec<(MotifNodeParams, ThoModuleWithStore)>)>>>,
    max_nets:i64,
    label: &String,
    slices_len: i64
) -> Result<Box<dyn SavableOptResult>, ThoosaError> {

    let max_nets = channels_motifs_nodes.read().iter().map(|nets| nets.1.len()).sum::<usize>() as i64;
    if max_nets <= 0 {
        panic!("no motif nets found");//TODO skip?
    }

    let channels = channels_motifs_nodes.read().len() as i64;
    let device = device.clone();
    let max_input = ws.read().max_input();

    let batch = Rc::new(Box::new(LoadingTensorsDataBatch::new(
        train_labels_resources,
        valid_labels_resources,

        comon_label_node_params.label_batch_size,
        Arc::clone(&executor),
        device.clone(),
        None

    )));
    debug!("run_opt_train_label_nodes: label={label:?} run_config={run_config:?} slices_len={slices_len} comon_label_node_params={comon_label_node_params:?} channels={channels} cuda_mem={:?}", cuda_mem());

    opt_train(
          |vs:&VarStore, train_node_values:&TrainLabelNodesValues, listener: &mut PlotVecf32|{

             let mut result = None::<(f32, LabelNodeParams)>;
             run_split_on_oom(
                 Reducer::new(BatchTransform::new(Rc::clone(&batch), |t|{
                     match &train_node_values.label_node_type {
                         LabelNetType::MaxPadding(params) => {
                             apply_motif_nets_with_padding(&t, Arc::clone(&channels_motifs_nodes), &device, max_input as u32)
                         }
                         LabelNetType::FixedNumSlices(params) => {
                             apply_motif_nets(&t, Arc::clone(&channels_motifs_nodes), &device, *params.slices_len.expect_value().unwrap())
                         }
                     }

                 })),
                 |b|{
                     train_label_nodes(
                         ws, train_node_values, comon_label_node_params, b, &device, channels, max_nets, vs, slices_len
                     )
                 },
                 |v, i|{
                     debug!("v={v:?}");
                     result.replace(v);
                 },
                 3
             )?;

             Ok(result.expect("no result from train_label_nodes"))
        },
        &run_config.run_config_or_opt_space,
        device,
        "label"
    ).map(|(var_store, label_node_params, train_label_nodes_values, min_loss)| {
        Box::new(TrainLabelNodesRunConfigResult::new(run_config.id.expect("run_config.id"), label.clone(), min_loss, label_node_params, train_label_nodes_values, var_store)) as Box<dyn SavableOptResult>
    })
}

pub fn train_label_nodes<B:DataBatch> (
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    train_node_values:&TrainLabelNodesValues,
    comon_label_node_params:&CommonTrainLabelNodeRunConfig,
    mut batch:&mut B,
    device: &Device,
    channels:i64,
    max_nets:i64,
    vs_0: &VarStore,
    slices_len: i64
) ->  Result<(f32, LabelNodeParams), ThoosaError> {

    let label_node_params = LabelNodeParams::new(channels, max_nets, slices_len, train_node_values.train_nodes_values.num_layers as i64);
    debug!("train_label_nodes: label_node_params={label_node_params:?} cuda_mem={:?}", cuda_mem());

    let device = device.clone();

    let mut optimizer = AdamW::default().amsgrad(true);

    if let Some(adam_w) = train_node_values.train_nodes_values.adam_w {
        debug!("train_label_nodes: setting adam_w={adam_w:?}");
        optimizer = optimizer.beta1(adam_w.beta1).beta2(adam_w.beta2);
    }

    debug!("train_label_nodes: cuda_mem={:?}", cuda_mem());

    let loss = train(
        &device,
        batch as &mut dyn DataBatch,
        &match &train_node_values.label_node_type {
            LabelNetType::MaxPadding(params) => {
                label_node_with_padding(&label_node_params).build_tch_module(&vs_0.root())?
            }
            LabelNetType::FixedNumSlices(params) => {
                label_node(&label_node_params).build_tch_module(&vs_0.root())?
            }
        },

        |pred: &Tensor, target: &Tensor| {
            debug!("train_on_label: pred.size={:?}", pred.size());
            let pred_y = pred.squeeze();
            debug!("train_on_label: pred_y={:?} target={:?} kind={:?}", pred_y.size(), target.size(), target.kind());
            pred_y.binary_cross_entropy::<Tensor>(&target, None, Reduction::Mean)
        },
        comon_label_node_params.cycles,
        optimizer.build(&vs_0, train_node_values.train_nodes_values.lr)?,

        |t: Tensor| {
            debug!("train_on_label: t.size={:?} cuda_mem={:?}", t.size(), cuda_mem());
            t
        },
        None,
    )?;

    Ok((loss, label_node_params))
}

//t - input label tensors batch size x channels x max motif nets x motif batches x motif batch size
fn view_for_label_model(t: Tensor, channels: i64, max_motif_nets: i64, motif_batch_size: i64, max_motif_batches: usize) -> Tensor {
    assert_eq!(t.size().len(), 5, "view_for_label_model: t.size()={:?}, should be 5 dims", t.size());

    debug!("view_for_label_model: t.size={:?} channels={channels}, max_motif_nets={max_motif_nets}, max_motif_batches={max_motif_batches} motif_batch_size={motif_batch_size} max_motif_batches as i64 * motif_batch_size={}", t.size(), max_motif_batches as i64 * motif_batch_size);

    // let t = t.view([-1/*items_per_label as i64 * 2*/, channels, max_nodes, max_motif_batches as i64 * motif_batch_size]);
    let t = t.flatten(3,4);

    debug!("view_for_label_model: 2. t.size={:?}", t.size());

    t
}

pub fn run_opt_train_final (
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Mutex<Box<dyn Store>>>,
    train_final_run_config: &TrainFinalNodeRunConfig,
    common_train_final_run_config: &CommonTrainFinalNodeRunConfig,
    train_resources:Vec<(Resource, Vec<f32>)>,
    valid_resources: Vec<(Resource, Vec<f32>)>,
    device: &Device,
    loader: &Arc<RwLock< LoadRunner<Box<Tensor>>>>,
    mut label_models: HashMap<String, Vec<(RunId, LabelNodeParams, ThoModuleWithStore)>>,
    labels_channels_motifs_models:HashMap<String, Vec<(FindMotifsRunConfig, Vec<(MotifNodeParams, ThoModuleWithStore)>)>>,

) -> Result<Box<dyn SavableOptResult>, ThoosaError> {
    let labels_len = labels_channels_motifs_models.len();

    let labels_channels_motifs_models_rw = labels_channels_motifs_models.into_iter().fold(HashMap::new(), |mut a, v|{
        a.insert(v.0, Arc::new(RwLock::new(v.1)));
        a
    });
    let device = device.clone();

    let labels = store.lock().list_labels()?;
    let max_input = ws.read().max_input() as u32;
    let batch = Rc::new(Box::new(ClassTensorsLoaderData::new(
        train_resources,
        valid_resources,

        common_train_final_run_config.batch_size,
        Arc::clone(&loader),
        device.clone(),
        // Some( Box::new(move |t:Box<Tensor>| {
        //     apply_motif_and_label_nets(&t, &label_models, &labels_channels_motifs_models_rw, &device, &labels, max_input)
        //
        // }))
        None
    )));

    opt_train(
        |vs:&VarStore, run_config_values:&TrainNodesValues, listener: &mut PlotVecf32|{
            train_final_node(
                ws, run_config_values, common_train_final_run_config, &mut BatchTransform::new(Rc::clone(&batch), |t|{
                    apply_motif_and_label_nets(&Box::new(t), &mut label_models, &labels_channels_motifs_models_rw, &device, &labels, max_input)
                }), &device, labels_len, vs
            )
        },
        &train_final_run_config.run_config_or_opt_space,
        device,
        "final"
    ).map(|(var_store, final_node_params, train_final_nodes_values, min_loss)| {
        debug!("run_opt_train_final: done, min_loss={min_loss} final_node_params={final_node_params:?}");
        Box::new(TrainFinalNodesRunConfigResult::new(train_final_run_config.id.expect("run_config.id."), min_loss, var_store, final_node_params, train_final_nodes_values)) as Box<dyn SavableOptResult>
    })

}


/**
slices single input tensor's channel into batches of input_slices_motif_batch_size with input_slice_step which were used during motif node training.
all motif nodes are to be from same run so have same input params.
Arranges output into max_motif_nodes x max_motif_batches  (max_ - to be same for all channels)

input_slices_motif_batch_size - both for motif node input batch size and output (input for label node) batch

 # returns Tensor nodes x batches x batch size
*/
pub fn apply_channel_motifs(
    motif_nodes: &mut [(MotifNodeParams, ThoModuleWithStore)],
    batch_channel: Tensor,
    motif_width: i64,
    sum_motif_nodes: i64,
    slices_len: i64,
    device: &Device
) -> Result<Tensor, ThoosaError> {

    debug!("apply_channel_motifs_2: motif_nodes={motif_nodes:?} batch_channel={batch_channel:?} motif_width={motif_width} max_motif_nodes={sum_motif_nodes} max_dists_len={slices_len} device={device:?}");

    if motif_nodes.is_empty() {
        return Ok(Tensor::zeros([batch_channel.size()[0], sum_motif_nodes, slices_len], (Float, *device)));
    }

    let (input_slice_step, pad) = clc_step(batch_channel.size()[1], motif_width, slices_len); //(channel.size()[0]-motif_width)/100;

    let unfolded = batch_channel
        .pad([0, pad], "constant", 0.)//pad: last dimension [0 - padding left, pad - padding right]
        .unfold(1, motif_width, input_slice_step);

    debug!("apply_channel_motifs_2: channel.size={:?} motif_width={motif_width} input_slice_step={input_slice_step} pad={pad}  unfolded.size()={:?}", batch_channel.size(), unfolded.size());

    let motif_dists = motif_nodes.iter_mut().map(|(_params, net)| {
        net.set_device(device.clone());

        let mut motif_distance_splits = vec![];
        run_split_on_oom(
            SplitQueue::<2, _>::new(SplittableTensor::<0>(unfolded.shallow_clone())),
            |t|{
                let t = t.as_ref();
                trace!("apply_channel_motifs_2: forward: t=\n{t}");
                let r = net.infer(&t.to_device(device.clone()));
                debug!("apply_channel_motifs_2: forward: size={:?} r.size={:?}", t.as_ref().size(), r.size());
                trace!("apply_channel_motifs_2: forward: r=\n{r}");

                Ok(r)
            },
            |v, t|{
                debug!("v={v:?}");
                motif_distance_splits.push(v);
            },
            10//TODO config
        ).unwrap();//TODO unwrap

        net.set_device(Cpu);

        debug!("apply_channel_motifs_2: motif_distance_splits={motif_distance_splits:?}");
        if motif_distance_splits.len() > 0 {
            Ok(Tensor::concat(&motif_distance_splits, 0))

        } else if motif_distance_splits.len() == 1 {
            Ok(motif_distance_splits.pop().unwrap())

        } else {
            Err(ThoosaError::new("no motif distances "))

        }
    }).collect::<Result<Vec<_>, ThoosaError>>()?;

    Ok(Tensor::stack(motif_dists.as_slice(), 1).view((batch_channel.size()[0], motif_nodes.len() as i64, -1)))
}

pub fn apply_channel_motifs_with_padding(input_slice_step: usize,
                            motif_nodes: &mut [(MotifNodeParams, ThoModuleWithStore)],
                            channel: Tensor,
                            motif_width: i64,
                            max_motif_nodes: i64,
                            slices_len: i64,
                            device: &Device
) -> Result<Tensor, ThoosaError> {

    debug!("apply_channel_motifs_2: motif_nodes={motif_nodes:?} channel={channel:?} motif_width={motif_width} max_motif_nodes={max_motif_nodes} max_dists_len={slices_len} device={device:?}");

    if motif_nodes.is_empty() {
        return Ok(Tensor::zeros([max_motif_nodes, slices_len], (Float, *device)));
    }

    let unfolded = channel.unfold(0, motif_width, input_slice_step as i64);

    let motif_dists = motif_nodes.iter_mut().map(|(_params, net)| {
        net.set_device(device.clone());

        let mut motif_distance_splits = vec![];
        run_split_on_oom(
            SplitQueue::<2, _>::new(SplittableTensor::<0>(unfolded.shallow_clone())),
            |t|{

                let r = net.infer(&t.as_ref().to_device(device.clone()));
                debug!("apply_channel_motifs_2: forward: size={:?} r.size={:?}", t.as_ref().size(), r.size());

                Ok(r)
            },
            |v, t|{
                debug!("v={v:?}");
                motif_distance_splits.push(v);
            },
            10
        ).unwrap();//TODO unwrap

        net.set_device(Cpu);
        debug!("apply_channel_motifs_2: motif_distance_splits={motif_distance_splits:?}");
        if motif_distance_splits.len() > 0 {
            Ok(Tensor::concat(&motif_distance_splits, 0))
        } else if motif_distance_splits.len() == 1 {
            Ok(motif_distance_splits.pop().unwrap())
        } else {
            Err(ThoosaError::new("no motif distances "))
        }
    }).collect::<Result<Vec<_>, ThoosaError>>()?;

    //TODO pad_sequence?
    pad_to(&Tensor::stack(motif_dists.as_slice(), 0).view((motif_nodes.len() as i64, -1)), &[max_motif_nodes, slices_len])
}


pub fn apply_motif_and_label_nets(
    batch_t: &Tensor,
    label_models: &mut HashMap<String, Vec<(RunId, LabelNodeParams, ThoModuleWithStore)>>,
    labels_channels_motifs_models_rw: &HashMap<String, Arc<RwLock<Vec<(FindMotifsRunConfig, Vec<(MotifNodeParams, ThoModuleWithStore)>)>>>>,
    device: &Device,
    labels: &Vec<String>,
    max_input: u32
) -> Tensor {
    let t_0 = Instant::now();
    let mut labels_tensors = vec![];
    for label in labels {
        let label_nodes = label_models.get_mut(label).expect(format!("no label model for {label}").as_str());
        if label_nodes.len() == 0 {
            warn!("no label models for {label}");
            continue;
        }
        debug!("apply_motif_and_label_nets: label={label} device={device:?} label_nodes.len={:?} label_nodes[0].1={:?} t.size()={:?}", label_nodes.len(), label_nodes[0].1, batch_t.size());

        let channels_motifs_models = labels_channels_motifs_models_rw.get(label).expect(format!("no {label} label in labels_channels_motifs_models").as_str());
        debug!("apply_motif_and_label_nets: len={} channels_motifs_models={channels_motifs_models:?}", channels_motifs_models.read().len());

        let motif_nets_t = apply_motif_nets(&batch_t, Arc::clone(channels_motifs_models), &device, 100);


        for (label_node_run_id, label_node_params, label_node) in label_nodes.iter_mut() {
            debug!("apply_motif_and_label_nets: label_node_params={label_node_params:?} label_node_run_id={label_node_run_id} t_net.size()={:?}", motif_nets_t.size());

            let t_net_view = motif_nets_t.shallow_clone(); //view_for_label_model(motif_nets_t.unsqueeze(0), channels, label_node_params.max_motif_nodes, 0, label_nodes[0].1.slices_len as u32); //.view((t.size()[0], -1));

            debug!("apply_motif_and_label_nets: label={label} label_node_run_id={label_node_run_id:?} t_net_view.size()={:?} t_net_view.device()={:?}", t_net_view.size(), t_net_view.device());
            trace!("apply_motif_and_label_nets: label={label} t_net_view=\n{t_net_view}");
            label_node.set_device(*device);
            debug!("apply_motif_and_label_nets: label={label} label_node={label_node:?}");
            // label_node.debug();
            let t_out = label_node.infer(&t_net_view.to_device(*device));
            label_node.set_device(Cpu);
            debug!("apply_motif_and_label_nets: label={label} label_node.forward t_out.size={:?} t_out.device={:?}", t_out.size(), t_out.device());

            labels_tensors.push(t_out);
        }
    }
    let final_input_t = Tensor::stack(labels_tensors.as_slice(), 1);
    debug!("apply_motif_and_label_nets: final_input_t.size={:?} elapsed={}", final_input_t.size(), t_0.elapsed().as_millis());

    final_input_t
}

#[derive(Debug)]
struct TrainFinalNodesRunConfigResult {
    train_final_nodes_run_config_id: u32,
    loss: f32,
    var_store: VarStore,
    final_node_params: FinalNodeParams,
    train_final_nodes_values: Option<TrainNodesValues>
}

impl TrainFinalNodesRunConfigResult {
    pub fn new(train_final_nodes_run_config_id: u32, loss: f32, var_store: VarStore, final_node_params: FinalNodeParams, train_final_nodes_values: Option<TrainNodesValues>) -> Self {
        Self {
            train_final_nodes_run_config_id,
            loss,
            var_store,
            final_node_params,
            train_final_nodes_values,
        }
    }
}

impl SavableOptResult for TrainFinalNodesRunConfigResult {
    fn value(&self) -> f32 {
        self.loss
    }

    fn save(&mut self, store: &Arc<Mutex<Box<dyn Store>>>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<u32, ThoosaError> {
        let run_config_id = match self.train_final_nodes_values.as_ref(){
            None => self.train_final_nodes_run_config_id,

            Some(train_final_nodes_values) => {
                store.lock().save_run_config(
                    Some(&[self.train_final_nodes_run_config_id]),
                    &TrainLabelNodeRunConfig::type_name(),
                    "Final",
                    "todo",
                    TrainLabelNodeRunConfig::from_values(Some(self.train_final_nodes_run_config_id), train_final_nodes_values.lr, train_final_nodes_values.num_layers, LabelNetType::FixedNumSlices(FixedNumSlicesNetParams{slices_len: 100.into(), num_layers: 3.into()})).to_properties_map()
                )?
            }
        };

        ws.read().save_final_model(run_config_id, &self.var_store, &self.final_node_params).expect("save_final_model failed");

        Ok(run_config_id)
    }
}

pub fn train_final_node(ws: &Arc<RwLock<Box<dyn Workspace>>>, run_config_values: &TrainNodesValues, common_train_final_run_config: &CommonTrainFinalNodeRunConfig, batch:&mut dyn DataBatch, device: &Device, labels_len: usize, vs_0: &VarStore) ->  Result<(f32, FinalNodeParams), ThoosaError>{
    let final_node_params= FinalNodeParams::new(labels_len as i64, run_config_values.num_layers);

    let mut optimizer = nn::Adam::default();
    if let Some(adam_w) = run_config_values.adam_w {
        debug!("train_final_nodes: setting adam_w={adam_w:?}");
        optimizer = optimizer.beta1(adam_w.beta1).beta2(adam_w.beta2);
    }

    let loss = train(
        device,
        batch,
        &final_node(&final_node_params).build_tch_module(&vs_0.root())?, //TODO features?
        |pred: &Tensor, target: &Tensor| {



            debug!("train_final: pred.size={:?} target.size={:?}", pred.size(), target.size());
            trace!("train_final: pred=\n{pred} target=\n{target}");
            let pred= pad_to(&pred, &target.size()).expect("pad_to failed");
            debug!("train_final: padded pred.size={:?}", pred.size());

            pred.cross_entropy_loss::<Tensor>(&target, None, Reduction::Mean, -100, 0.0)

        },
        common_train_final_run_config.cycles,
        optimizer.build(&vs_0, run_config_values.lr)?,
        |t:Tensor|{
            trace!("train_final: t=\n{t}");
            debug!("train_final: t.size={:?} ", t.size());

            t.squeeze()
        },
        None
    )?;

    Ok((loss, final_node_params))
}

#[derive(Debug)]
struct LabelModels {
    label:String,
    label_model:Mutex<Box<dyn Module>>,
    motif_models:Vec<Vec<Mutex<Box<dyn Module>>>>,
    #[allow(dead_code)]
    motif_node_params:MotifNodeParams,
    #[allow(dead_code)]
    label_node_params:LabelNodeParams
}

impl LabelModels {

    pub fn new(label:String, corr_model: Box<dyn Module>, motif_models: Vec<Vec<Box<dyn Module>>>,  motif_node_params:MotifNodeParams, label_node_params:LabelNodeParams) -> Self {
        Self {
            label,
            label_model: Mutex::new(corr_model),
            motif_models: motif_models.into_iter().map(|v|v.into_iter().map(|m|Mutex::new(m)).collect::<Vec<_>>()).collect::<Vec<_>>(),
            motif_node_params,
            label_node_params
        }
    }

}

#[derive(Debug)]
pub struct ThoModuleWithStore {
    module:Box<dyn ThoModule>,
    var_store:VarStore
}

unsafe impl Sync for ThoModuleWithStore {}

impl ThoModuleWithStore {

    #[cfg(test)]
    pub fn new(module: Box<dyn ThoModule>) -> Self {
        Self {
            module,
            var_store: VarStore::new(Cpu)//dummy
        }
    }

    pub fn load(node: Node, model:PathBuf, device:Device) -> Result<Self, ThoosaError> {
        let mut vs = VarStore::new(device.clone());
        let module = node.build_tch_module(&vs.root())?;
        vs.load(&model)?;
        vs.freeze();

        Ok(Self {
            module,
            var_store: vs
        })
    }

    pub fn set_device(&mut self, device: Device) {
        self.var_store.set_device(device);
    }


    fn mem_total(&self) -> usize {
        self.var_store.variables()
            .iter()
            .map(|t| tensor_mem_size(t.1))
            .sum()
    }

    fn debug(&self){
        self.module.debug();
    }

    pub fn infer(&self, t: &Tensor) -> Tensor {
        self.module.tho_forward(t, false)
    }

    pub fn var_store(&self) -> &VarStore {
        &self.var_store
    }
}


pub fn load_motif_models(ws:&Arc<RwLock<Box<dyn Workspace>>>, channel:usize, motif_id:usize, device: &Device) -> Result<Vec<(MotifNodeParams, ThoModuleWithStore)>, ThoosaError> {
    debug!("load_motif_models_ws: channel={channel} motif_selection_run_id={motif_id}");

    let channel_motif_models_paths = ws.read().motif_models(&channel, motif_id)?;

    channel_motif_models_paths.into_iter().map(|(model, config)|{

        // let mut vs = VarStore::new(device.clone());
        let motif_node_params = MotifNodeParams::from_file(&config)?;

        debug!("load_motif_models_ws: model={model:?} config={config:?} motif_node_params={motif_node_params:?}");

        // let module = SyncModule::new(motif_node(&motif_node_params).build_tch_module(&vs.root())?);
        // vs.load(&model)?;
        // vs.freeze();

        let module = ThoModuleWithStore::load(motif_node(&motif_node_params), model, device.clone())?;

        Ok((motif_node_params, module))

    }).collect()

}

pub fn load_label_models_ws(ws:&Arc<RwLock<Box<dyn Workspace>>>, train_label_nodes_run_ids: Option<&Vec<RunId>>, device: &Device) -> Result<HashMap<String, Vec<(RunId, LabelNodeParams, ThoModuleWithStore)>>, ThoosaError> {
    let label_models_paths = ws.read().label_models()?;

    debug!("load_all_label_models_ws: label_models_paths={label_models_paths:?} train_label_nodes_run_ids={train_label_nodes_run_ids:?}");

    let train_label_nodes_run_ids_hash = train_label_nodes_run_ids.map(|v|v.into_iter().collect::<HashSet<_>>());

    debug!("load_all_label_models_ws: train_label_nodes_run_ids_hash={train_label_nodes_run_ids_hash:?}");


    label_models_paths.into_iter().try_fold(HashMap::new(), |mut a, (label, paths)|{

        let models:Vec<(RunId, LabelNodeParams, ThoModuleWithStore)>  =  paths.into_iter().filter_map(|(model_path, params_path)|{
            let run_id = FsWorkspace::model_path_to_run_id(&model_path);
            if train_label_nodes_run_ids_hash.as_ref().map(|set|!set.contains(&run_id)).unwrap_or(true) {
                debug!("load_all_label_models_ws: skipping run_id={run_id}");
                return None;
            }

            let label_node_params = LabelNodeParams::from_file(&params_path).expect(format!("no {params_path:?}").as_str());

            debug!("load_all_label_models_ws: model={model_path:?} run_id={run_id} label_node_params={label_node_params:?}");

            let node = ThoModuleWithStore::load(label_node(&label_node_params), model_path.clone(), *device).expect(format!("can't load {model_path:?}").as_str());
            // vs.load(&model_path).expect(format!("no {model_path:?}").as_str());
            // vs.freeze();

            Some((run_id, label_node_params, node))
        }).collect();

        a.insert(
            label,
            models
        );

        Ok(a)

    })
}

pub fn load_final_model(run_id: RunId, ws:&Arc<RwLock<Box<dyn Workspace>>>, device: &Device) ->Result<(RunId, FinalNodeParams, ThoModuleWithStore), ThoosaError> {
    let (model_path, params_path) = ws.read().final_model(run_id)?;
    debug!("load_final_model: params_path={model_path:?} run_id={run_id}");


    let final_node_params = FinalNodeParams::from_file(&params_path)?;
    debug!("load_final_model:  final_node_params={final_node_params:?}");

    let node = ThoModuleWithStore::load(final_node(&final_node_params), model_path, *device)?;

    Ok((run_id,  final_node_params, node))
}


#[derive(Args, Debug, Clone)]
struct CommonTrainingArgs {
    #[arg(long, default_value_t = 1000)]
    pub cycles: usize,

    #[arg(long, default_value_t = 10)]
    pub batch_size: usize,//number of the motif slices in a batch

    #[arg(long, value_parser=parse_range_or_value::<f64>)]
    pub lr: RangeOrValue<f64>,

    #[arg(long, default_value_t = 0.2)]
    pub max_loss: f32,

    #[arg(long, default_value_t = 10)]
    pub opt_max_evals: u32,

    #[arg(long, default_value_t = 3)] //TODO validate opt_n_init < opt_max_evals
    pub opt_n_init: u32,
}

#[derive(Args, Debug, Clone)]
pub struct TrainMotifsNetsParams {

    #[arg(long)]
    pub motif_selection_run_id: Vec<RunId>,

    #[arg(long, value_parser=parse_range_or_value::<u32>)]
    pub num_layers: RangeOrValue<u32>,

    #[command(flatten)]
    common: CommonTrainingArgs,
}

#[derive(Debug, Clone)]
pub struct OptConfig {
    pub max_evals:u32,
    pub n_init:u32
}

impl OptConfig {
    pub fn new(max_evals: u32, n_init: u32) -> Self {
        Self { max_evals, n_init }
    }
}


#[derive(Debug, Clone)]
pub enum ValuesOrOptSpace<V:Display> {
    Values (V),
    OptSpace{opt_space:ConfigurationSpace<String>,opt_config:OptConfig}
}

#[derive(Clone, Debug)]
pub struct TrainMotifsNodesRunConfig {
    id:Option<RunId>,

    run_config_or_opt_space: ValuesOrOptSpace<TrainNodesValues>

}

// unsafe impl Sync for TrainMotifsNodesRunConfig{}
pub fn adam_w_hyperparams() -> Vec<Hyperparameter<String>>{
    vec![
        Hyperparameter::UniformFloat {name: "beta1".into(), lower:0.5, upper: 0.99},
        Hyperparameter::UniformFloat {name: "beta2".into(), lower:0.9, upper: 0.9999}
    ]
}

pub fn basic_hyperparams() -> Vec<Hyperparameter<String>>{
    vec![
        Hyperparameter::UniformFloat {name: "lr".into(), lower:1e-6, upper: 1e-2},
        Hyperparameter::UniformInt {name: "num_layers".into(), lower:0, upper: 6}
    ]
}

#[derive(Debug, Clone)]
pub struct TrainNodesValues {
    lr: f64,
    num_layers: u32,
    adam_w: Option<AdamW>
}

impl TrainNodesValues {
    pub fn new(lr: f64, num_layers: u32, adam_w: Option<AdamW>) -> Self {
        Self { lr, num_layers, adam_w }
    }
}

impl Display for TrainNodesValues {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.lr, self.num_layers)
    }
}

impl TryFrom<&[NamedVal<String>]> for TrainNodesValues {
    type Error = ThoosaError;

    fn try_from(value: &[NamedVal<String>]) -> Result<Self, Self::Error> {
        // let params_map = value.into_iter().try_fold(HashMap::new(), |mut a, x| {
        //     a.insert(x.name().as_str(), x.val());
        //     Ok::<HashMap<&str, &Val>, ThoosaError>(a)
        // })?;
        //
        // Ok(TrainNodesValues::new(
        //     (*params_map.get("lr").expect("lr")).try_into()?,
        //     (*params_map.get("num_layers").expect("num_layers")).try_into()?,
        //     None//TODO
        // ))
        todo!()//replace with builder
    }
}
impl TryFrom<&Vec<Val>> for TrainNodesValues {
    type Error =ThoosaError;

    fn try_from(value: &Vec<Val>) -> Result<Self, Self::Error> {
        debug!("TrainNodesValues::try_from: value={:?}", value);

        let num_layers:f32 = (&value[1]).try_into()?;//TODO opt can't produce ints
        Ok(Self {
            lr: (&value[0]).try_into()?,
            num_layers: num_layers as u32,
            adam_w: if value.len() > 2 {Some(AdamW::default()
                .beta1((&value[2]).try_into()?)
                .beta2((&value[3]).try_into()?)
            )} else {
                None
            },
        })
    }
}

impl From<TrainNodesValues> for HashMap<String, Val>   {
    fn from(v: TrainNodesValues) -> Self {

        let mut h = HashMap::from([

                ("TrainNodesValues.lr".into(), (v.lr).into()),
                ("TrainNodesValues.num_layers".into(), (v.num_layers).into()),
            ]
        );

        if let Some(adam_w) = v.adam_w {
            h.extend(HashMap::from([
                ("AdamW.beta1".into(), adam_w.beta1.into()),
                ("AdamW.beta2".into(), adam_w.beta2.into()),
                ("AdamW.wd".into(), adam_w.wd.into()),
                ("AdamW.eps".into(), adam_w.eps.into()),
                ("AdamW.amsgrad".into(), adam_w.amsgrad.into())
            ]));
        }
        h
    }
}

#[derive(Debug)]
pub struct CommonTrainMotifsNodesRunConfig {
    pub batch_size: usize,
    pub cycles: usize,
    pub max_loss:f32,
}

impl TrainMotifsNodesRunConfig {
    pub fn from_values(id: Option<RunId>, lr: f64, num_layers: u32) -> Self {
        Self {
            id,
            run_config_or_opt_space: ValuesOrOptSpace::Values(TrainNodesValues {lr, num_layers, adam_w: None}),
        }
    }

    pub fn from_opt_space(id: Option<RunId>, opt_space:ConfigurationSpace<String>, opt_config: OptConfig) -> Self {
        Self {
            id,
            run_config_or_opt_space: ValuesOrOptSpace::OptSpace {opt_space, opt_config},
        }
    }
}

impl RunConfig for TrainMotifsNodesRunConfig{

    fn to_properties_map(&self) -> HashMap<String, Val> {
        match &self.run_config_or_opt_space {
            ValuesOrOptSpace::Values(TrainNodesValues { lr, num_layers , ..}) => {
                 HashMap::from([
                    ("TrainMotifsNodesRunConfig.lr".into(), (*lr).into()),
                    ("TrainMotifsNodesRunConfig.num_layers".into(), (*num_layers).into()),
                ])
            }
            ValuesOrOptSpace::OptSpace{opt_space, opt_config} => {
                prefix_properties("TrainMotifsNodesRunConfig_opt_space", opt_space.into())
                //     .into_iter().chain(prefix_properties("TrainMotifsNodesRunConfig_opt_config", opt_config.into())
                // ).collect()

            }
        }
    }

    fn from_properties_map(run_id: RunId, properties: HashMap<String, Val>) -> Self {
        let properties = unprefix_properties("TrainMotifsNodesRunConfig", &properties);

        if properties.contains_key("config_space") {
            todo!()
        } else {
            TrainMotifsNodesRunConfig {
                id:Some(run_id),

                run_config_or_opt_space: ValuesOrOptSpace::Values(TrainNodesValues {
                    lr: <&Val as TryInto<f64>>::try_into(properties.get("lr").expect("no lr")).expect("lr as f32"),
                    num_layers: <&Val as TryInto<u32>>::try_into(properties.get("num_layers").expect("no num_layers")).expect("num_layers as u32"),
                    adam_w: None,
                }),
            }
        }
    }

    fn set_id(&mut self, id: RunId) {
        self.id.replace(id);
    }

    fn type_name() -> &'static str {
        "MotifNode"
    }
}

impl Params<TrainMotifsNodesRunConfig, Empty, CommonTrainMotifsNodesRunConfig> for TrainMotifsNetsParams {

    fn to_run_configs(&self) -> Vec<TrainMotifsNodesRunConfig> {

        if let Some(config_space) = self.to_config_space() {

            debug!("TrainMotifsNodesParams::to_run_configs: config_space={config_space:?}");
            vec![
                TrainMotifsNodesRunConfig::from_opt_space(None, config_space, OptConfig::new(self.common.opt_max_evals, self.common.opt_n_init))
            ]

        } else {
            //TODO avoid collecting
            let v = make_combos([
                self.common.lr.clone().into_iter().map(|v| v.into()).collect::<Vec<Val>>().into_iter(),
                self.num_layers.clone().into_iter().map(|v| v.into()).collect::<Vec<Val>>().into_iter(),
            ].into_iter()).into_iter().map(|mut config_combo:Vec<Val>|{
                debug!("TrainMotifsNodesParams::to_run_configs: config_combo={config_combo:?} ");
                let num_layers = config_combo.pop().unwrap().try_into().unwrap();
                let lr = config_combo.pop().unwrap().try_into().unwrap();
                TrainMotifsNodesRunConfig::from_values(
                    None,
                    lr,
                    num_layers,
                )
            }).collect();

            debug!("TrainMotifsNodesParams::to_run_configs: self.lr={:?} self.num_layers={:?} v={v:?}", self.common.lr, self.num_layers);

            v
        }
    }

    fn resource_config(&self) -> Empty {
        Empty{}
    }

    fn common_config(&self) -> CommonTrainMotifsNodesRunConfig {
        CommonTrainMotifsNodesRunConfig {
            batch_size: self.common.batch_size,
            cycles: self.common.cycles,
            max_loss: self.common.max_loss
        }
    }
}

impl TrainMotifsNetsParams {

    pub fn dummy(motif_selection_run_id:Vec<RunId>, cycles: usize, max_loss: f32, batch_size:usize, lr: impl Into<RangeOrValue<f64>>, num_layers:impl Into<RangeOrValue<u32>>, opt_max_evals: u32, opt_n_init: u32) ->Self{

        let lr = lr.into();

        if let RangeOrValue::Range{range, step} = &lr{
            assert!(range.end > range.start);
        }

        TrainMotifsNetsParams {
            motif_selection_run_id,

            num_layers: num_layers.into(),

            common: CommonTrainingArgs {
                cycles,

                lr,

                max_loss,

                batch_size,

                opt_max_evals,

                opt_n_init,
            }
        }
    }

    //TODO add explicit flag to enable optimization
    pub fn to_config_space(&self) -> Option<ConfigurationSpace<String>> {
        [
            ("lr".into(), &self.common.lr).try_into().ok(),
            ("num_layers".into(), &self.num_layers).try_into().ok(),
        ].into_iter()
            .collect::<Option<Vec<Hyperparameter<String>>>>()
            .map(|v|ConfigurationSpace::new(v))
    }
}
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct MotifNodeParams {
    num_features: i64,
    num_layers:i64
}

impl MotifNodeParams {

    pub fn new(num_features: i64, num_layers: i64) -> Self {
        Self {
            num_features,
            num_layers
        }
    }

    pub fn to_properties_map(&self) -> HashMap<String, Val> {
        HashMap::from([
            ("num_features".into(),  self.num_features.into()),
        ])
    }

    pub fn from_properties_map(properties:HashMap<String, Val>) -> Self {

        MotifNodeParams::new(
            properties.get("num_features").expect(format !("no num_features in {:?}", properties).as_str()).try_into().expect("num_features as_i64"), 1,
        )
    }

    pub fn from_file(path:&PathBuf) -> Result<Self, ThoosaError> {
        serde_json5::from_str::<MotifNodeParams>(
            fs::read_to_string(&path)?.as_str()
        ).map_err(|serde_err|ThoosaError::new(format!("err={}, path={path:?}", serde_err.to_string())))
    }

    pub fn num_features(&self) -> i64 {
        self.num_features
    }

    pub fn with_num_layer(&self, num_layers:i64) -> Self {
        Self {
            num_features: self.num_features,
            num_layers,
        }
    }
}

pub fn motif_node(params: &MotifNodeParams) -> Node {
    debug!("motif_node: params={params:?}");
    let mut builder = SeqBuilder::new(params.num_features, None);
    let mut net = builder
        // .batch_norm()
        // .conv_1d(10, 1, 1, ConvConfig::default())
        .linear_s(params.num_features * 2)
        .func(Tensor::relu);

        for i in 0..params.num_layers as usize {
            net = net.linear_s(params.num_features * 2)
                .func(Tensor::relu);
        }


    net
        // .linear_s(180)
        // .func(Tensor::relu)
        .linear_s(1)
        .func(Tensor::sigmoid)
        .end_seq()
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabelNodeParams {
    channels:i64,
    motif_nodes:i64,
    motif_layers:i64,
    slices_len:i64
}

impl LabelNodeParams {
    pub fn new(channels: i64, max_motif_nodes: i64, slices_len:i64, motif_layers:i64) -> Self {
        Self { channels, motif_nodes: max_motif_nodes, slices_len, motif_layers }
    }

    pub fn to_properties_map(&self) -> HashMap<String, Val> {
        let map = HashMap::from([
            ("LabelNodeParams.channels".into(),  self.channels.into()),
            ("LabelNodeParams.max_motif_nets".into(),  self.motif_nodes.into()),
            ("LabelNodeParams.slices_len".into(),  self.slices_len.into()),
        ]);

        map
    }

    pub fn from_properties_map(properties:HashMap<String, Val>) -> Self {
        let properties = unprefix_properties("LabelNodeParams", &properties);
        LabelNodeParams::new(
            properties.get("channels").expect("channels").try_into().expect("channels as_i64"),
            properties.get("motif_nets").expect("motif_nets").try_into().expect("motif_nets as_i64"),
            properties.get("motif_layers").expect("motif_layers").try_into().expect("motif_layers as_i64"),
            properties.get("slices_len").expect("slices_len").try_into().expect("slices_len as_i64")
        )
    }

    pub fn from_file(path:&PathBuf) -> Result<Self, ThoosaError> {
        serde_json5::from_str::<LabelNodeParams>(
            fs::read_to_string(&path)?.as_str()
        ).map_err(|serde_err|ThoosaError::new(serde_err.to_string()))
    }
}

pub fn flat_label_node(params: &LabelNodeParams, slices_len:i64) -> Node {
    debug!("motif_node: params={params:?}");
    let num_features = params.channels * params.motif_nodes * slices_len;
    let mut builder = SeqBuilder::new(slices_len, None);
    builder
        .closure(Box::new(move |t: &Tensor| {
            let t_0 = t.flatten(1, 3);
            debug!("label_node flatten t.size={:?} t_0.size={:?}", t.size(), t_0.size());

            t_0
        }), Some(num_features))
        .linear_s(num_features)
        .func(Tensor::relu)
        .linear_s(num_features)
        .func(Tensor::relu)
        .linear_s(1)

        .func(Tensor::sigmoid)
        .end_seq()
}

pub fn label_node(params:&LabelNodeParams) -> Node {
    debug!("label_node: params={:?}", params);

    let max_motif_nets=  params.motif_nodes;
    SeqBuilder::new(params.slices_len, None)//Some(2)

        .split(
            (0..params.motif_nodes).map( |_i| {
                let num_features = params.slices_len;
                let mut builder = SeqBuilder::new(num_features, None);
                let mut net = builder
                    .linear_s(num_features);

                for i in 0..params.motif_layers as usize {//params.num_layers
                    net = net
                        .linear_s(num_features)
                        .func(Tensor::relu);
                }

                net
                    .func(Tensor::relu)
                    .linear_s(1)
                    // .func(Tensor::sigmoid)
                    .end_seq()

            }).collect(),

            Box::new(
                move |t:&Tensor|
                    (0..max_motif_nets)
                    .map(|net_i|{
                        t.i((.., net_i, ..))
                    })
                .collect()
            ),//label batch, channel, max_motif_nets, max slices

            Box::new(move |t:Vec<Tensor>|{
                debug!("label_node merging t={t:?}");
                // let out = Tensor::stack(t.as_slice(), 1).mean_dim(1, false, Float);
                let out = Tensor::concat(t.as_slice(), 1); //.mean_dim(1, false, Float);
                debug!("label_node merging out={:?}", out.size());

                out
            })

        )
        .closure(Box::new(move |t:&Tensor|{
            debug!("label_node merged final in flatten t.size()={:?}", t.size());

            // let t_0 = t.squeeze_dims(1);
            // debug!("label_node merged final out flatten t_0.size()={:?}", t_0.size());
            // t_0
            t.shallow_clone()

        }), Some(params.motif_nodes))

        .linear()
        // .func(Tensor::relu)
        .linear_s(1)
        .func(Tensor::sigmoid)
        .end_seq()

}


pub fn label_node_with_padding(params:&LabelNodeParams) -> Node {
    debug!("label_node: params={:?}", params);

    let max_motif_nets=  params.motif_nodes;
    SeqBuilder::new(params.slices_len, Some(3))

        .split(
            (0..params.motif_nodes).map( |_i| {
                let num_features = params.slices_len;
                let mut builder = SeqBuilder::new(num_features, None);
                let mut net = builder
                    .linear_s(num_features);

                for i in 0..params.motif_layers as usize {//params.num_layers
                    net = net
                        .linear_s(num_features)
                        .func(Tensor::relu);
                }

                net
                    .func(Tensor::relu)
                    .linear_s(1)
                    // .func(Tensor::sigmoid)
                    .end_seq()

            }).collect(),

            Box::new(
                move |t:&Tensor|
                    (0..max_motif_nets)
                        .map(|net_i|{
                            t.i((.., .., net_i, ..))
                        })
                        .collect()
            ),//label batch, channel, max_motif_nets, max slices

            Box::new(move |t:Vec<Tensor>|Tensor::stack(t.as_slice(), 1).mean_dim(1, false, Float))

        )
        .closure(Box::new(move |t:&Tensor|{
            debug!("label_node merged final in flatten t.size()={:?}", t.size());

            let t_0 = t.squeeze_dims(2);
            debug!("label_node merged final out flatten t_0.size()={:?}", t_0.size());

            t_0

        }), Some(params.channels))

        .linear()
        // .func(Tensor::relu)
        .linear_s(1)
        .func(Tensor::sigmoid)
        .end_seq()

}


#[derive(Debug, Serialize, Deserialize, Default)]
pub struct FinalNodeParams {
    in_features: i64,
    num_layers: u32
}

impl FinalNodeParams {
    pub fn new(in_features: i64, num_layers: u32) -> Self {
        Self { in_features, num_layers }
    }

    pub fn to_properties_map(&self) -> HashMap<String, Val> {
        HashMap::from([
            ("FinalNodeParams.classes_num".into(), self.in_features.into()),
            ("FinalNodeParams.num_layers".into(), self.num_layers.into()),
        ])
    }

    pub fn from_file(path:&PathBuf) -> Result<Self, ThoosaError> {
        serde_json5::from_str::<FinalNodeParams>(
            fs::read_to_string(&path)?.as_str()
        ).map_err(|serde_err|ThoosaError::new(serde_err.to_string()))
    }


}

pub fn final_node(params:&FinalNodeParams) -> Node {
    debug!("final_node: params={:?}", params);

    let mut builder = SeqBuilder::new(params.in_features, None);
    let mut net = builder
        .linear_s(2*params.in_features)
        .func(Tensor::relu);

    for i in 0..params.num_layers {
        net = net.linear_s(params.in_features)
             .func(Tensor::relu)
    }

    net
        // .func(Tensor::relu)
        .linear_s(params.in_features)
        // .func(|t|t.softmax(1, Float))
        .end_seq()
}

struct PlotVecf32 {
    id:String,
    data:Vec<f32>
}

impl PlotVecf32 {
    pub fn new(id:&str) -> Self {
        Self {id: id.into(), data: vec![] }
    }

    fn plot(&self){
        plot_series(
            vec![
                (
                    self.id.clone(),
                    BLACK, self.
                    data.iter().enumerate().map(|(i, v)| (i as f32, *v as f32)).collect::<Vec<(f32, f32)>>()
                )
            ],
            self.id.clone().as_str(),
            (2000, 1000),
            &format!("/tmp/plots/motif_train_{}.png", self.id)
        ).unwrap();
    }
}

impl TrainListener for PlotVecf32 {
    fn on_valid_cycle(&mut self, cycle: usize, loss: f32) -> ListenerResponse {
        self.data.push(loss);

        //TODO move into another listener
        if self.data.len() > 10 {
            let len_half = self.data.len() / 2;
            if self.data[0..len_half].iter().sum::<f32>() < self.data[len_half..].iter().sum::<f32>()  {
                debug!("TrainListener for PlotVecf32::on_valid_cycle stopping self.data={:?}", self.data);
                return ListenerResponse::Stop;
            }
        }

        ListenerResponse::Nope
    }
}

#[derive(Debug)]
pub struct TrainMotifsNodesRunConfigResult {
    loss:f32,
    channel:u32,
    motif_id:usize,
    vs:VarStore,
    params:MotifNodeParams,
    run_config_values: Option<TrainNodesValues>,
    train_motifs_nodes_run_config_id: RunId
}

impl TrainMotifsNodesRunConfigResult {
    pub fn new(loss:f32, channel:u32, motif_id:usize, vs: VarStore, train_motifs_nodes_run_config_id: RunId, params: MotifNodeParams, run_config_values: Option<TrainNodesValues>) -> Self {
        Self { loss, channel, motif_id, vs, params, run_config_values,train_motifs_nodes_run_config_id }
    }
}

impl SavableOptResult for TrainMotifsNodesRunConfigResult {

    fn value(&self) -> f32 {
        self.loss
    }

    fn save(&mut self, store:&Arc<Mutex<Box<dyn Store>>>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<u32, ThoosaError> {
        let run_config_id = match self.run_config_values.as_ref() {
            None => self.train_motifs_nodes_run_config_id,
            Some(run_config_values) => {
                store.lock().save_run_config(
                    Some(&[self.train_motifs_nodes_run_config_id]),
                    &TrainMotifsNodesRunConfig::type_name(),
                    "descr",
                    "todo",
                    TrainMotifsNodesRunConfig::from_values(Some(self.train_motifs_nodes_run_config_id), run_config_values.lr, run_config_values.num_layers).to_properties_map()
                )?
            }
        };

        ws.read().save_motif_model(&self.channel, &run_config_id, &self.motif_id, &self.vs, &self.params)?;

        Ok(run_config_id)
    }
}

pub fn run_opt_train_motif(
    motif_id:usize,
    channel: u32,
    all_slices_0:Vec<Box<Tensor>>,
    all_slices_1:Vec<Box<Tensor>>,
    train_motifs_run_config: &TrainMotifsNodesRunConfig,
    motif_width: i64,
    common_config: &CommonTrainMotifsNodesRunConfig,
    device: Device
) -> Result<Box<dyn SavableOptResult>, ThoosaError> {


    /*plot_series(
       all_slices_0.iter().map(|t| {
           (format!("0"), BLACK, t.iter::<f64>().unwrap().enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>())
       }).collect(),
       "all_slices_0",
       (2000, 1000),
       &"/tmp/plots/all_slices_0.png"
   ).unwrap();

   plot_series(
       all_slices_1.iter().map(|t| {
           (format!("0"), BLACK, t.iter::<f64>().unwrap().enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>())
       }).collect(),
       "all_slices_0",
       (2000, 1000),
       &"/tmp/plots/all_slices_1.png"
   ).unwrap();*/

    let mut batch = TensorsDataBatch::new(
        vec![
            all_slices_0,
            all_slices_1
        ],
        common_config.batch_size,
        0.8,//TODO config
        device
    );

    opt_train(
        |vs:&VarStore, run_config_values:&TrainNodesValues, listener: &mut PlotVecf32|{
            run_train_motif(motif_width, common_config, &device, &mut batch, vs, run_config_values, listener)
        },
        &train_motifs_run_config.run_config_or_opt_space,
        device,
        "motifs"
    ).map(|(var_store, motif_node_params, train_motifs_nodes_values, min_loss)|{
        Box::new(TrainMotifsNodesRunConfigResult::new(min_loss, channel, motif_id, var_store, train_motifs_run_config.id.expect("id"), motif_node_params, train_motifs_nodes_values)) as Box<dyn SavableOptResult>
    })

}


pub fn opt_train<F, V, N:Debug+Default>(mut train: F, run_config: &ValuesOrOptSpace<V>, device: Device, descr: &str) -> Result<(VarStore, N, Option<V>, f32), ThoosaError>//Option<V> - Some if run config from opt
where
    F: FnMut(&VarStore, &V, &mut PlotVecf32) -> Result<(f32, N), ThoosaError>,
    for<'a> <V as TryFrom<&'a [NamedVal<String>]>>::Error: Debug,//TODO remove Vec
    V: Clone + Display + Debug + for<'a> TryFrom<&'a [NamedVal<String>]>,
{

    match &run_config {

        ValuesOrOptSpace::Values(v) => {
            let vs = VarStore::new(device);

            debug!("opt_train no opt: v={v:?}");
            let mut listener = PlotVecf32::new(format!("_params_{}", v).as_str()); //motif_id ?
            let (loss, node_params) = train(&vs, v, &mut listener)?;
            listener.plot();

            debug!("opt_train: done no opt, descr={descr} loss={loss}");
            Ok((vs, node_params, None, loss))
        }

        ValuesOrOptSpace::OptSpace{opt_space,opt_config} => {
            let mut min_loss = f32::MAX;
            let mut min_var_store: Option<(VarStore, N, Option<V>, f32)> = None;
            info!("running opt train {descr}");

            let mut opt = crate::parzen::optimizer::TPEOptimizer::new(
                |params| {
                    let run_config_values: V = params.try_into().expect("can't convert params into TrainNodesValues"); //TODO handle

                    let vs = VarStore::new(device);

                    let mut listener = PlotVecf32::new(format!("_params_{}", run_config_values).as_str()); //motif_id ?
                    let (loss, node_params) =  match panic_to_err!(train(&vs, &run_config_values, &mut listener)){
                        Ok(v) => {
                           v
                        }
                        Err(e) => {
                            warn!("opt_train failed with {e:?} on {run_config_values:?}");
                            (f32::MAX, Default::default())
                        }
                    };
                    listener.plot();

                    debug!("opt_train: descr={descr} params={params:?} run_config_values={run_config_values:?} loss={loss}");
                    if loss < min_loss {
                        min_loss = loss;
                        min_var_store.replace((vs, node_params, Some(run_config_values), min_loss));
                    }
                    print!(".");
                    loss
                },
                opt_space.clone(),
                opt_config.max_evals as usize, //common_config.opt_max_evals as usize,
                opt_config.n_init as usize, //common_config.opt_n_init as usize,

            );

            let best_config = opt.optimize();
            trace!("done opt {descr}");
            info!("opt_train: done opt, descr={descr} best_config={best_config:?} min_loss={min_loss} min_var_store={min_var_store:?}");

            min_var_store.ok_or_else(||ThoosaError::new("opt was not run"))
            // min_var_store.unwrap_or_else(|| (VarStore::new(Cpu), Default::default(), Some(vec![]), f32::MAX))
        }
    }

}

fn run_train_motif(motif_width: i64, common_config: &CommonTrainMotifsNodesRunConfig, device: &Device, mut batch: &mut TensorsDataBatch, vs: &VarStore, run_config_vals:&TrainNodesValues, listener: &mut PlotVecf32) -> Result<(f32, MotifNodeParams), ThoosaError> {
    let node_params = MotifNodeParams::new(motif_width, run_config_vals.num_layers as i64);

    let loss=  train(
        &device,
        batch as &mut dyn DataBatch,
        &motif_node(&node_params).build_tch_module(&vs.root()).unwrap(),
        |pred: &Tensor, target: &Tensor| {
            let pred_y = pred.i((.., 0));

            let loss = pred_y.binary_cross_entropy::<Tensor>(&target.totype(Float), None, Reduction::Mean);
            debug!("train_on_motif: pred_y ={pred_y} target={target} loss={loss}");

            loss
        },
        common_config.cycles,
        AdamW::default().build(&vs, run_config_vals.lr).unwrap(),
        |t| t,
        Some(listener),
    )?;

    Ok((loss, node_params))
}

pub fn slices_in_all_resources <ResourceId>(slices: Vec<(ResourceId, Vec<Box<Tensor>>)>) -> Vec<Box<Tensor>> {
    slices.into_iter().flat_map(|(_resource_key, slices)| slices.into_iter()).collect::<Vec<Box<Tensor>>>()
}


#[derive(Debug)]
pub struct TensorsDataBatch {
    train :Vec<Vec<Box<Tensor>>>,//index - class
    val :Vec<Vec<Box<Tensor>>>,
    batch_size: usize,
    device:Device
}

impl TensorsDataBatch {

    pub fn new(classes_tensors: Vec<Vec<Box<Tensor>>>, batch_size: usize, train_valid_split:f32, device:Device) -> Self {

        let classes_len = classes_tensors.len();
        assert!(classes_len > 0);
        // assert!(batch_size % classes_len == 0);

        let (train, val) = split_train_val(classes_tensors, train_valid_split);

        Self {
            train,
            val,
            batch_size,
            device
        }
    }


    fn choose_for_batch(class_tensors: &Vec<Vec<Box<Tensor>>>, batch_size: usize, device:&Device) -> (Tensor, Tensor) {
        let mut rng = thread_rng();

        let per_class_len = batch_size / class_tensors.len();//TODO make balanced
        debug!("TensorsDataBatch::choose_for_batch: class_tensors={} per_class_len={per_class_len}", class_tensors.len());

        let mut x_y_vecs = class_tensors.iter().enumerate().fold(vec![], |mut x_y_acc, (class, tensors)| {
            assert!(tensors.len() > 0, "no tensors for class={class} in ClassTensorsDataBatch");

            tensors.choose_multiple(&mut rng, per_class_len).for_each(|t| {

                x_y_acc.push((t.shallow_clone(), class as i64));
            });

            x_y_acc
        });

        x_y_vecs.shuffle(&mut rng);

        debug!("TensorsDataBatch::choose_for_batch: x_y_vecs.1.len()={} batch_size={batch_size}", x_y_vecs.len());
        assert!(x_y_vecs.len() > 0);

        //TODO throw if not enough tensors for batch?
        (x_y_vecs.len()..batch_size).for_each(|i|{
            x_y_vecs.push(
                (
                    x_y_vecs[i-x_y_vecs.len()].0.shallow_clone(),
                    x_y_vecs[i-x_y_vecs.len()].1
                )
            );

        });

        let classes = Tensor::from_slice(x_y_vecs.iter().map(|e| e.1).collect::<Vec<_>>().as_slice()).to_device(*device);

        let tensor = Tensor::stack(
            x_y_vecs.into_iter().map(|e|e.0).collect::<Vec<_>>().as_slice(),
            0,
        ).view((batch_size as i64, -1)).to_device(*device).set_requires_grad(true);



        debug!("TensorsDataBatch::choose_for_batch: tensor.size()={:?} tensor.device()={:?}", tensor.size(), tensor.device());

        (
            tensor,
            classes
        )
    }
}

impl DataBatch for TensorsDataBatch {

    fn train_batch(&self) -> (Tensor, Tensor) {
        Self::choose_for_batch(&self.train, self.batch_size, &self.device)
    }

    fn valid_batch(&self) -> (Tensor, Tensor) {
        Self::choose_for_batch(&self.val, self.batch_size, &self.device)
    }
}

fn split_train_val<T>(classes_tensors: Vec<Vec<T>>, train_valid_split: f32) -> (Vec<Vec<T>>, Vec<Vec<T>>) {
    let classes_len = classes_tensors.len();
    let (train, val): (Vec<Vec<T>>, Vec<Vec<T>>) = classes_tensors
        .into_iter()
        .fold((Vec::with_capacity(classes_len), Vec::with_capacity(classes_len)), |mut class_train_val, tensors| {
            let tensors_len = tensors.len();
            let split_i = (tensors_len as f32 * train_valid_split) as usize;

            let train_val = tensors.into_iter().enumerate().fold((Vec::with_capacity(split_i), Vec::with_capacity(tensors_len - split_i)), |mut train_val, (i, t)| {
                if i < split_i {
                    train_val.0.push(t);
                } else {
                    train_val.1.push(t);
                }
                train_val
            });


            class_train_val.0.push(train_val.0); //train
            class_train_val.1.push(train_val.1); //validation
            class_train_val
        });
    (train, val)
}


/// returns Tensor channels x nets x batches x batch size
fn apply_motif_nets(
    batch_t: &Tensor,
    channels_motifs_nodes: Arc<RwLock<Vec<(FindMotifsRunConfig, Vec<(MotifNodeParams, ThoModuleWithStore)>)>>>,
    device: &Device,
    slices_len: u32
) -> Tensor {
    let max_motif_nodes = channels_motifs_nodes.read().iter().map(|nets| nets.1.len()).max().unwrap_or(0);

    debug!("apply_motif_nets: input_t.size={:?} max_motif_nodes={max_motif_nodes} ", batch_t.size());

    let channels_motifs = Tensor::concat(
        channels_motifs_nodes.write()
            .iter_mut()
            .enumerate()
            .map(|(channel, (find_motifs_cfg, motif_nodes))| {
                // let slices_len = 100; //slices_len(input_t.size()[1], find_motifs_cfg.window as i64, find_motifs_cfg.step as usize);
                debug!("apply_motif_nets: slices_len={slices_len} slices_len={slices_len} motif_width={} step={}", find_motifs_cfg.window, find_motifs_cfg.step);
                let r = apply_channel_motifs(
                    motif_nodes,
                    batch_t.i((.., channel as i64, ..)),
                    find_motifs_cfg.window as i64,
                    max_motif_nodes as i64,
                    slices_len as i64,
                    device
                );
                debug!("apply_motif_nets: r={:?}", r);
                r
            })
            .collect::<Result<Vec<_>, ThoosaError>>().unwrap().as_slice(),
        1
    );

    // let padded = pad_to(&channels_motifs, &[5, max_motif_nodes as i64, max_motif_batches as i64, 7]).unwrap();
    // debug!("apply_label_motifs: channels_motifs.size={:?} padded.size()={:?}", channels_motifs.size(), padded.size());
    // padded

    debug!("apply_motif_nets: channels_motifs.size={:?}", channels_motifs.size());

    channels_motifs
}

fn apply_motif_nets_with_padding(
    input_t: &Tensor,
    channels_motifs_nodes: Arc<RwLock<Vec<(FindMotifsRunConfig, Vec<(MotifNodeParams, ThoModuleWithStore)>)>>>,
    device: &Device,
    input_n: u32
) -> Tensor {
    let max_motif_nodes = channels_motifs_nodes.read().iter().map(|nets| nets.1.len()).max().unwrap_or(0);

    debug!("apply_motif_nets: input_t.size={:?} max_motif_nodes={max_motif_nodes} ", input_t.size());

    let channels_motifs = Tensor::stack(
        channels_motifs_nodes.write()
            .iter_mut()
            .enumerate()
            .map(|(channel, (find_motifs_cfg, motif_nodes))| {
                let slices_len = slices_len(input_n as i64, find_motifs_cfg.window as i64, find_motifs_cfg.step as usize);
                debug!("apply_motif_nets: slices_len={slices_len} input_n={input_n} motif_width={} step={}", find_motifs_cfg.window, find_motifs_cfg.step);
                apply_channel_motifs_with_padding(
                    find_motifs_cfg.step as usize,
                    motif_nodes,
                    input_t.i((channel as i64, ..)),
                    find_motifs_cfg.window as i64,
                    max_motif_nodes as i64,
                    slices_len,
                    device
                )})
            .collect::<Result<Vec<_>, ThoosaError>>().unwrap().as_slice(),
        0
    );

    // let padded = pad_to(&channels_motifs, &[5, max_motif_nodes as i64, max_motif_batches as i64, 7]).unwrap();
    // debug!("apply_label_motifs: channels_motifs.size={:?} padded.size()={:?}", channels_motifs.size(), padded.size());
    // padded

    debug!("apply_motif_nets: channels_motifs.size={:?}", channels_motifs.size());

    channels_motifs
}

#[derive(Debug, Args, Clone)]
pub struct TrainFinalNetParams {
    #[arg(long)]
    pub train_labels_run_ids: Vec<RunId>,

    #[arg(long,  default_value_t = 100)]
    batch_size: usize,

    #[arg(long, default_value_t = 1000)]
    cycles: usize,

    #[arg(long, value_parser=parse_range_or_value::<f64>)]
    pub lr: RangeOrValue<f64>,

    #[arg(long, value_parser=parse_range_or_value::<u32>)]
    pub num_layers: RangeOrValue<u32>,

    #[arg(long)]
    train_resource_group: String,

    #[arg(long)]
    valid_resource_group: String,

    #[arg(long, default_value_t = 10)]
    pub opt_max_evals: u32,

    #[arg(long, default_value_t = 3)] //TODO validate opt_n_init < opt_max_evals
    pub opt_n_init: u32,
}

impl TrainFinalNetParams {

    pub fn dummy(train_labels_run_ids: Vec<RunId>, batch_size: usize, train_resource_group:String, valid_resource_group:String, cycles: usize, lr: impl Into<RangeOrValue<f64>>, num_layers:impl Into<RangeOrValue<u32>>, opt_max_evals: u32, opt_n_init: u32) ->Self {
        TrainFinalNetParams {
            train_labels_run_ids,
            cycles,
            batch_size,
            lr:lr.into(),
            num_layers:num_layers.into(),
            train_resource_group,
            valid_resource_group,
            opt_max_evals,
            opt_n_init,
        }
    }

    //TODO add explicit flag to enable optimization
    pub fn to_config_space(&self) -> Option<ConfigurationSpace<String>> {
        [
            vec![
                ("lr".into(), &self.lr).try_into().ok(),
                ("num_layers".into(), &self.num_layers).try_into().ok(),
            ],
            adam_w_hyperparams().into_iter().map(|v|Some(v)).collect::<Vec<Option<Hyperparameter<String>>>>(),
        ].concat().into_iter()
            .collect::<Option<Vec<Hyperparameter<String>>>>()
            .map(|v|ConfigurationSpace::new(v))
    }


    pub fn train_resource_group(&self) -> &str {
        &self.train_resource_group
    }

    pub fn valid_resource_group(&self) -> &str {
        &self.valid_resource_group
    }

}

#[derive(Debug)]
pub struct TrainFinalNodeRunConfig {
    id:Option<RunId>,
    run_config_or_opt_space: ValuesOrOptSpace<TrainNodesValues>
}

impl TrainFinalNodeRunConfig {

    pub fn from_values(id: Option<RunId>, lr: f64, num_layers: u32) -> Self {
        Self {
            id,
            run_config_or_opt_space: ValuesOrOptSpace::Values(TrainNodesValues {lr, num_layers, adam_w: None}),
        }
    }

    pub fn from_opt_space(id: Option<RunId>, opt_space:ConfigurationSpace<String>, opt_config: OptConfig) -> Self {
        Self {
            id,
            run_config_or_opt_space: ValuesOrOptSpace::OptSpace {opt_space, opt_config},
        }
    }
}

impl RunConfig for crate::train_nets::TrainFinalNodeRunConfig {
    fn from_properties_map(run_id: RunId, properties: HashMap<String, Val>) -> Self {
        todo!()
    }

    fn set_id(&mut self, id: RunId) {
        self.id.replace(id);
    }

    fn type_name() -> &'static str {
        "FinalNode"
    }
}


#[derive(Debug)]
pub struct CommonTrainFinalNodeRunConfig {
    pub cycles:usize,
    pub batch_size:usize,
}

impl CommonTrainFinalNodeRunConfig {
    pub fn new(cycles: usize, batch_size:usize) -> Self {
        Self { cycles, batch_size }
    }
}

impl Params<TrainFinalNodeRunConfig, NetResources<Logits>, CommonTrainFinalNodeRunConfig> for TrainFinalNetParams {

    fn to_run_configs(&self) -> Vec<TrainFinalNodeRunConfig> {

        if let Some(config_space) = self.to_config_space() {

            debug!("TrainFinalNodeParams::to_run_configs: config_space={config_space:?}");
            vec![
                TrainFinalNodeRunConfig::from_opt_space(None, config_space, OptConfig::new(self.opt_max_evals, self.opt_n_init))
            ]

        } else {
            //TODO avoid collecting
            let v = make_combos([
                self.lr.clone().into_iter().map(|v| v.into()).collect::<Vec<Val>>().into_iter(),
                self.num_layers.clone().into_iter().map(|v| v.into()).collect::<Vec<Val>>().into_iter(),
            ].into_iter()).into_iter().map(|mut config_combo:Vec<Val>|{
                debug!("TrainFinalNodeParams::to_run_configs: config_combo={config_combo:?} ");
                let num_layers = config_combo.pop().unwrap().try_into().unwrap();
                let lr = config_combo.pop().unwrap().try_into().unwrap();
                TrainFinalNodeRunConfig::from_values(
                    None,
                    lr,
                    num_layers,
                )
            }).collect();

            debug!("TrainFinalNodeParams::to_run_configs: self.lr={:?} self.num_layers={:?} v={v:?}", self.lr, self.num_layers);

            v
        }
    }

    fn resource_config(&self) -> NetResources<Logits> {
        NetResources::new(
            Logits::new(self.train_resource_group.clone()),
            Logits::new(self.valid_resource_group.clone())
        )
    }

    fn common_config(&self) -> CommonTrainFinalNodeRunConfig {
        CommonTrainFinalNodeRunConfig::new(
            self.cycles,
            self.batch_size
        )
    }
}


#[derive(Debug, Args, Clone)]
pub struct InferParams {
    #[arg(long)]
    pub final_run_id: RunId,

    #[arg(long,  default_value_t = 1000)]
    pub input_slice_step: usize,

    #[arg(long, default_value = "../../data_local/test_eegs")]
    pub input_dir: String,

    #[arg(long, default_value = "submission.csv")]
    pub output_dir: String,
}

impl InferParams {


    pub fn input_data(&self) -> &str {
        &self.input_dir
    }
    pub fn output(&self) -> &str {
        &self.output_dir
    }

    pub fn new(final_run_id: RunId, input_data: String, output: String, input_slice_step: usize) -> Self {
        Self { final_run_id, input_slice_step, input_dir: input_data, output_dir: output }
    }
}

#[allow(dead_code)]
fn build_multidim<'a>( builder:&'a mut SeqBuilder, dims:&'a [i64])  {

    let builder = dims.windows(2).into_iter().fold(builder, | b, d|{
        debug!("d={:?}", d);

        b.linear_s(d[0])
            // .func(Tensor::relu)
            .linear_s(1)
            // .func(Tensor::sigmoid)
            .closure(Box::new(|t:&Tensor|t.squeeze()), Some(d[1]))

    });

    builder.func(Tensor::sigmoid);
}

pub fn is_nonzero(t_1: &Tensor) -> Result<bool, ThoosaError> {
    Ok(t_1.flatten(0, (t_1.size().len() - 1) as i64).iter::<f64>()?.find(|v| v > &0.).is_some())
}




#[derive(Clone, PartialEq, Debug)]
pub struct FindMotifsRunConfig {
    pub id:Option<RunId>,
    pub step: u32,
    pub window: u32,
    pub dist_quantile: f32,
    pub min_tracker_step: Option<u32>,
    pub filter: Option<InputFilter>,
    pub descr: String,
}


impl RunConfig for FindMotifsRunConfig {
    fn to_properties_map(&self) -> HashMap<String, Val> {
        let mut map = HashMap::from([
            ("FindMotifsRunConfig.step".into(), UInt(self.step)),
            ("FindMotifsRunConfig.window".into(), UInt(self.window)),
            ("FindMotifsRunConfig.dist_quantile".into(), self.dist_quantile.into()),
            ("FindMotifsRunConfig.descr".into(), self.descr.as_str().into()),
        ]);

        if let Some(filter) = self.filter.as_ref() {
            map.extend(thoosa_shared::val::prefix_properties("FindMotifsRunConfig.filter", filter.to_properties_map()).into_iter());
        }

        map
    }

    //TODO set id
    fn from_properties_map(run_id: RunId, properties: HashMap<String, Val>) -> Self {
        debug!("FindMotifsRunConfig::from_properties_map: run_id={run_id} properties={properties:?}");

        let properties = unprefix_properties("FindMotifsRunConfig", &properties);
        let find_motifs_run_config = FindMotifsRunConfig::new(
            None,
            properties.get("step").expect("no step").try_into().expect("step as_u32"),
            properties.get("window").expect("no window").try_into().expect("window as_u32"),
            properties.get("dist_quantile").expect("no dist_quantile").try_into().expect("dist_quantile as_f32"),
            <&Val as TryInto<String>>::try_into(properties.get("descr").expect("no descr")).expect("descr as_str"),
            InputFilter::from_properties_map(&thoosa_shared::val::unprefix_properties("filter", &properties)),
        );

        debug!("FindMotifsRunConfig::from_properties_map: find_motifs_run_config={find_motifs_run_config:?}");

        find_motifs_run_config
    }

    fn set_id(&mut self, id: RunId) {
        self.id.replace(id);
    }

    fn type_name() -> &'static str {
        "FindMotifs"
    }
}

impl FindMotifsRunConfig {
    pub fn new(id: Option<u32>,
               step: u32,
               window: u32,
               dist_quantile: f32, //TODO move to common
               descr: impl Into<String>,
               filter: Option<InputFilter>,
    ) -> Self {
        Self {
            id,
            step,
            window,
            dist_quantile,
            filter,
            min_tracker_step: None,//TODO
            descr: descr.into(),
        }
    }

    pub fn window(&self) -> i64 {
        self.window as i64//TODO
    }

    pub fn step(&self) -> usize {
        self.step as usize //TODO
    }
}

#[derive(ThoosaRunConfig, Deserialize, Debug, Subcommand, PartialEq, Clone)]
pub enum InputFilter {
    Conv1(Conv1Params)
}

#[derive(ThoosaRunConfig, Deserialize, Debug, Args, PartialEq, Clone)] //PartialEq - for tests
pub struct Conv1Params {
    #[arg(long)]
    #[run_config]
    kernel: i64,

    #[arg(long, default_value_t = 1)]
    #[run_config]
    stride: i64,

    #[arg(long, default_value_t = 0)]
    #[run_config]
    padding: i64,
}

impl InputFilter {
    pub fn to_properties_map(&self) -> HashMap<String, Val> {
        match self {
            Self::Conv1(conv1_params) => {
                conv1_params.to_properties_map().into_iter().fold(HashMap::new(), |mut a, v| {
                    a.insert(format!("conv_1.{}", v.0), v.1);
                    a
                })
            }
        }
    }

    pub fn from_properties_map(properties: &HashMap<String, Val>) -> Option<Self> {
        let conv_1_props = thoosa_shared::val::unprefix_properties("conv_1", &properties);
        if conv_1_props.len() > 0 {
            assert!(conv_1_props.len() == 3);
            Conv1Params::from_properties_map(conv_1_props).map(|props| InputFilter::Conv1(props)).ok()
        } else {
            None
        }
    }
}

impl Conv1Params {
    pub fn new(kernel: i64, stride: i64, padding: i64) -> Self {
        Self { kernel, stride, padding }
    }

    pub fn to_properties_map(&self) -> HashMap<String, Val> {
        HashMap::from([
            ("Conv1Params.kernel".into(), self.kernel.into()),
            ("Conv1Params.stride".into(), self.stride.into()),
            ("Conv1Params.padding".into(), self.padding.into()),
        ])
    }

    pub fn from_properties_map(properties: HashMap<String, Val>) -> Result<Self, ThoosaError> {
        let properties = thoosa_shared::val::unprefix_properties("Conv1Params", &properties);

        Ok(Conv1Params {
            kernel: properties.get("kernel").expect("kernel").try_into().expect("kernel as_i64"),
            stride: properties.get("stride").expect("stride").try_into().expect("stride as_i64"),
            padding: properties.get("padding").expect("padding").try_into().expect("padding as_i64"),
        })
    }
}

#[derive(Debug, Subcommand, Clone)]
pub enum LabelNetType {
    //to produce same size tensors for label net
    MaxPadding(MaxPaddingNetParams),
    //...input tensors are padded to max nets in a channel/max input length over all inputs
    FixedNumSlices(FixedNumSlicesNetParams),
    //... motif nodes are ran with calculated step to produce equally sized (slices_len) output tensors (shared/clc_step) for inputs of different size
}

#[derive(Debug, Args, Clone)]
pub struct MaxPaddingNetParams {
    #[arg(long, value_parser=parse_range_or_value::<u32>)]
    num_layers: RangeOrValue<u32>,
}

#[derive(Debug, Args, Clone)]
pub struct FixedNumSlicesNetParams {
    #[arg(long, value_parser=parse_range_or_value::<u32>)]
    pub num_layers: RangeOrValue<u32>,

    #[arg(long, value_parser=parse_range_or_value::<u32>)]
    pub slices_len: RangeOrValue<u32>,
}

impl LabelNetType {


}

impl Display for LabelNetType {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            LabelNetType::MaxPadding(_) => {
                Display::fmt(&"MaxPadding", f)
            }
            LabelNetType::FixedNumSlices(_) => {
                Display::fmt(&"FixedNumSlices", f)
            }
        }
    }
}

impl LabelNetType {

}

#[cfg(test)]
mod train_tests {
    use crate::builder::SeqBuilder;
    use crate::loader::{LoadRunner, OfMemSize, SyncWrapper};
    use crate::my_tch::ThoModule;
    use crate::simp::DummyDs;
    use crate::train_nets;
    use crate::train_nets::{apply_channel_motifs, apply_channel_motifs_with_padding, apply_motif_and_label_nets, apply_motif_nets, apply_motif_nets_with_padding, label_node, opt_train, BatchTransform, ClassTensorsLoaderData, CommonTrainFinalNodeRunConfig, FinalNodeParams, FindMotifsRunConfig, FixedNumSlicesNetParams, LabelNetType, LoadingTensorsDataBatch, PlotVecf32, TrainLabelNodesValues, TrainMotifsNetsParams, TrainNodesValues, ValuesOrOptSpace};
    use crate::train_nets::{load_label_models_ws, load_motif_models, CommonTrainLabelNodeRunConfig, InferParams, LabelNodeParams, MotifNodeParams, TensorsDataBatch, ThoModuleWithStore, TrainFinalNetParams, TrainLabelNetsParams, TrainLabelNodeRunConfig};
    use crate::utils::{cuda_mem, from, optimizer_mem_size, set_requires_grad_in_subpath, train, vs_mem_size, SplittableDataBatch};
    use crate::utils::{lin_t, pad_to, reset_print_options, run_split_on_oom, t_0_0, tensor_mem_size, DataBatch, Reducer, SplitQueue, Splittable, SplittableTensor};
    use crate::ws::{DummyWorkspace, FsWorkspace, Workspace};
    use cudarc::driver::CudaDevice;
    use log::LevelFilter::{Debug, Info, Trace};
    use log::{debug, info, set_max_level, trace};
    use parking_lot::{Mutex, RwLock};
    use rand::{thread_rng, Rng};
    use rayon::iter::IntoParallelRefMutIterator;
    use std::collections::HashMap;
    use std::{fs, mem};
    use std::ops::Deref;
    use std::path::{Path, PathBuf};
    use std::rc::Rc;
    use std::sync::Arc;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::nn::{init, AdamW, BatchNormConfig, LinearConfig, Module, OptimizerConfig, VarStore};
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::{Float, Half};
    use tch::{nn, Device, Tensor};
    use tch::{IndexOp, Kind, Reduction};
    use torch_sys::at_autocast_clear_cache;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::{batch_windows, cdt_for_dir, clc_num_batches, pulse_in_rand, sin_v, slices_len, ThoosaError};
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::val::Val;
    use thoosa_store::SqlStore;
    use crate::runs::Params;

    #[ctor::ctor]
    fn test_init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_class_tensors_data_batch()-> Result<(), ThoosaError>{

        let mut batch = TensorsDataBatch::new(
            vec![
                vec![
                    Box::new(Tensor::from_slice(&[0., 0., 0.])),Box::new(Tensor::from_slice(&[0., 0., 1.])),Box::new(Tensor::from_slice(&[0., 0., 2.])),
                    Box::new(Tensor::from_slice(&[0., 1., 0.])),Box::new(Tensor::from_slice(&[0., 1., 1.])),Box::new(Tensor::from_slice(&[0., 1., 2.])),
                    Box::new(Tensor::from_slice(&[0., 2., 0.])),Box::new(Tensor::from_slice(&[0., 2., 1.])),Box::new(Tensor::from_slice(&[0., 2., 2.])),
                    Box::new(Tensor::from_slice(&[0., 3., 0.])),Box::new(Tensor::from_slice(&[0., 3., 1.])),Box::new(Tensor::from_slice(&[0., 3., 2.]))
                ],

                vec![
                    Box::new(Tensor::from_slice(&[1., 0., 0.])),Box::new(Tensor::from_slice(&[1., 0., 1.])),Box::new(Tensor::from_slice(&[1., 0., 2.])),
                    Box::new(Tensor::from_slice(&[1., 1., 0.])),Box::new(Tensor::from_slice(&[1., 1., 1.])),Box::new(Tensor::from_slice(&[1., 1., 2.])),
                    Box::new(Tensor::from_slice(&[1., 2., 0.])),Box::new(Tensor::from_slice(&[1., 2., 1.])),Box::new(Tensor::from_slice(&[1., 2., 2.])),
                    Box::new(Tensor::from_slice(&[1., 3., 0.])),Box::new(Tensor::from_slice(&[1., 3., 1.])),Box::new(Tensor::from_slice(&[1., 3., 2.]))
                ]

            ],
            10,
            0.5,
            Cpu
        );

        let train_batch = batch.train_batch();

        debug!("train_batch: x={:?}", from::<Vec<f32>>(&train_batch.0.i((.., 0))));
        debug!("train_batch: y={:?}", from::<Vec<f32>>(&train_batch.1));

        assert_eq!(from::<Vec<f32>>(&train_batch.0.i((.., 0))), from::<Vec<f32>>(&train_batch.1));

        // debug!("train_batch: x={:?}, y={:?}", train_batch.0.size(), train_batch.1.size());

        assert_eq!(10, train_batch.0.size()[0]);
        assert_eq!(5, train_batch.1.iter::<i64>()?.filter(|v|*v==1).count());
        assert_eq!(5, train_batch.1.iter::<i64>()?.filter(|v|*v==0).count());

        let valid_batch = batch.valid_batch();
        assert_eq!(10, valid_batch.0.size()[0]);
        assert_eq!(5, valid_batch.1.iter::<i64>()?.filter(|v|*v==1).count());
        assert_eq!(5, valid_batch.1.iter::<i64>()?.filter(|v|*v==0).count());


        Ok(())
    }

    ///sums and multiplies inputs by k
    pub fn test_net(num_features: i64, k: f64, device: Device, ) -> Result<Box<dyn ThoModule>, ThoosaError> {
        let vs = VarStore::new(device);

        SeqBuilder::new(num_features, None)
            .linear_s_with_conf(1, LinearConfig {
                ws_init: init::Init::Const(k),//conflicts with ctor's init
                bs_init: None,
                bias: false
            })
            .end_seq().build_tch_module(&vs.root())
    }

    ///sums and multiplies inputs by k
    pub fn test_net_vs(num_features: i64, k: f64, vs:&VarStore) -> Result<Box<dyn ThoModule>, ThoosaError> {

        SeqBuilder::new(num_features, None)
            //.linear_s_with_conf(num_features, LinearConfig {
            //     ws_init: init::Init::Const(k),//conflicts with ctor's init
            //     bs_init: None,
            //     bias: false
            // })

            .linear_s_with_conf(1, LinearConfig {
                ws_init: init::Init::Const(k),//conflicts with ctor's init
                bs_init: None,
                bias: false
            })
            .end_seq().build_tch_module(&vs.root())
    }


    #[test]
    fn test_test_net() -> Result<(), ThoosaError>{
        let t_0 = lin_test_tensor(1000);
        debug!("t_0.size={:?}", t_0.size());

        let t = t_0.unfold(1, 100, 10);

        debug!("t.size={:?}", t.size());

        //sums and multiplies by k
        let r_0 = test_net(100, 1., Cpu)?.tho_forward(&t, true);
        debug!("r_0.size={:?}", r_0.size());

        assert_eq!(Tensor::from_slice(&[4950.]), r_0.i((0, 0, ..)));
        assert_eq!(Tensor::from_slice(&[9900.]), test_net(100, 2., Cpu)?.tho_forward(&t, true).i((0, 0, ..)));

        Ok(())
    }


    #[test]
    fn test_apply_motif_models() -> Result<(), ThoosaError>{
        set_max_level(Trace);
        reset_print_options();

        let input_n = 1000;
        let motif_width = 100;
        let step = (motif_width / 1) as usize;//1 - step==motif_width slices do not overlap so net output - sum(slice)*k

        //batch 2 x channels 5 x n 1000
        let batch = Tensor::stack(&[
            lin_test_tensor(input_n),
            lin_test_tensor(input_n)
        ], 0);

        debug!(" batch.size()={:?}", batch.size());

        let slices_len = 11;

        let mut motif_nets: Vec<(FindMotifsRunConfig, Vec<(MotifNodeParams, ThoModuleWithStore)>)> = vec![
            (//channel 0 nets
             FindMotifsRunConfig::new(Some(1), step as u32, motif_width as u32, 0.1, "test", None),
             vec![
                 (MotifNodeParams::new(motif_width, 1), ThoModuleWithStore::new(test_net(motif_width, 1., Cpu)?)),
                 (MotifNodeParams::new(motif_width, 1), ThoModuleWithStore::new(test_net(motif_width, 2., Cpu)?))
             ]
            ),
            (//channel 1 nets&mut DummyLabelBatch::new(),
             FindMotifsRunConfig::new(Some(1), step as u32, motif_width as u32, 0.1, "test", None),
             vec![
                 (MotifNodeParams::new(motif_width, 1), ThoModuleWithStore::new(test_net(motif_width, 10., Cpu)?))
             ]
            )
        ];

        let max_motif_nets_in_channel = motif_nets.iter().map(|nets|nets.1.len()).max().unwrap_or(0);
        assert_eq!(2, max_motif_nets_in_channel);



        // let channel_0_t = apply_channel_motifs(step, batch_size as i64, &mut motif_nets[0], t.i((0, ..)), max_motif_nets_in_channel as i64, max_motif_batches, &Cpu);
        let channel_0_t = apply_channel_motifs(&mut motif_nets[0].1, batch.i((.., 0, ..)), motif_width, max_motif_nets_in_channel as i64, slices_len, &Cpu)?;

        // debug!("t.i((0, ..))=\n{}", batch.i((0, ..)));
        debug!("channel_0_t.size={:?}", channel_0_t.size());
        trace!("channel_0_t=\n{channel_0_t}");

        let channel_0_motifs = Tensor::stack(&[
            //sum 0-100  100-200 ...
            Tensor::from_slice(&[4950.,  13950.,  22950.,  31950.,  40950.,  49950.,  58950.,  67950.,  76950.,  85950.,  94950.]),  //ch 0 net 0 (ResourceId=1)
            //sum 0-100  100-200 ... * 2 - k in second net in channel 0
            Tensor::from_slice(&[9900.,  27900.,  45900.,  63900.,  81900.,  99900., 117900., 135900., 153900., 171900., 189900.]) // ch. 0 net 1 (ResourceId=2)
        ], 0).totype(Float).view((1, 2, 11));

        assert_eq!(channel_0_motifs, channel_0_t.i((0, ..)).view((1,2,11)));

        // let channel_1_t = apply_channel_motifs(step, batch_size as i64, &mut motif_nets[1], t.i((1, ..)), max_motif_nets_in_channel as i64, max_motif_batches, &Cpu);
        let channel_1_t = apply_channel_motifs(&mut motif_nets[1].1, batch.i((.., 1, ..)), motif_width, max_motif_nets_in_channel as i64, slices_len, &Cpu)?;

        debug!("channel_1_t.size={:?}", channel_1_t.size());
        trace!("channel_1_t=\n{channel_1_t}");

        let channel_1_motifs = Tensor::stack(&[
            Tensor::from_slice(&[99000.,  279000.,  459000.,  639000.,  819000.,  999000., 1179000., 1359000., 1539000., 1719000., 1899000.,]),  //ch 1 net 0 (ResourceId=1)
        ], 0).totype(Float).view((1, 1, 11));

        assert_eq!(channel_1_motifs, channel_1_t.view((2, -1)).i((0, ..)).view((1,1,11)));

        debug!("test_apply_motif_models: channel_0_motifs.size()={:?}, channel_1_motifs.size()={:?}", channel_0_motifs.size(), channel_1_motifs.size());

        let channels_motifs = apply_motif_nets(&batch, Arc::new(RwLock::new(motif_nets)), &Cpu, 11u32).squeeze();

        let sample = Tensor::concat(&[channel_0_motifs, channel_1_motifs], 1);

        debug!("test_apply_motif_models: channels_motifs.size()={:?}, sample.size()={:?}", channels_motifs.size(), sample.size());

        assert_eq!(
            sample,
            channels_motifs.i((0, .., ..)).view((1,3,11))
        );

        Ok(())
    }

    #[test]
    fn test_apply_motif_models_with_padding() -> Result<(), ThoosaError>{
        reset_print_options();
        let input_n = 1000;
        let motif_width = 100;
        let step = (motif_width / 1) as usize;//1 - step==motif_width slices do not overlap so net output - sum(slice)*k

        let t = lin_test_tensor(input_n);

        let slices_len = slices_len(input_n, motif_width, step);

        let mut motif_nets: Vec<(FindMotifsRunConfig, Vec<(MotifNodeParams, ThoModuleWithStore)>)> = vec![
            (//channel 0 nets
             FindMotifsRunConfig::new(Some(1), step as u32, motif_width as u32, 0.1, "test", None),
             vec![
                 (MotifNodeParams::new(motif_width, 1), ThoModuleWithStore::new(test_net(motif_width, 1., Cpu)?)),
                 (MotifNodeParams::new(motif_width, 1), ThoModuleWithStore::new(test_net(motif_width, 2., Cpu)?))
             ]
            ),
            (//channel 1 nets&mut DummyLabelBatch::new(),
             FindMotifsRunConfig::new(Some(1), step as u32, motif_width as u32, 0.1, "test", None),
             vec![
                 (MotifNodeParams::new(motif_width, 1), ThoModuleWithStore::new(test_net(motif_width, 10., Cpu)?))
             ]
            )
        ];

        let max_motif_nets_in_channel = motif_nets.iter().map(|nets|nets.1.len()).max().unwrap_or(0);
        assert_eq!(2, max_motif_nets_in_channel);

        let batch_size = 10;
        let motif_batches = batch_windows(input_n as usize, motif_width as usize, step, batch_size);
        debug!("motif_batches={motif_batches:?}");
        let max_motif_batches = motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0};

        // let channel_0_t = apply_channel_motifs(step, batch_size as i64, &mut motif_nets[0], t.i((0, ..)), max_motif_nets_in_channel as i64, max_motif_batches, &Cpu);
        let channel_0_t = apply_channel_motifs_with_padding(step, &mut motif_nets[0].1, t.i((0, ..)), motif_width, max_motif_nets_in_channel as i64, slices_len, &Cpu)?;

        debug!("t.i((0, ..))=\n{}", t.i((0, ..)));
        debug!("channel_0_t.size={:?} channel_0_t=\n{channel_0_t}", channel_0_t.size());

        let channel_0_motifs = Tensor::stack(&[
            //sum 0-100  100-200 ...
            Tensor::from_slice(&[4950., 14950., 24950., 34950., 44950., 54950., 64950., 74950., 84950., 94950.]),  //ch 0 net 0 (ResourceId=1)
            //sum 0-100  100-200 ... * 2 - k in second net in channel 0
            Tensor::from_slice(&[9900., 29900., 49900., 69900., 89900., 109900., 129900., 149900., 169900., 189900.]) // ch. 0 net 1 (ResourceId=2)
        ], 0
        ).totype(Float).view((2, 10));
        assert_eq!(channel_0_motifs, channel_0_t.view((2, 10)));

        // let channel_1_t = apply_channel_motifs(step, batch_size as i64, &mut motif_nets[1], t.i((1, ..)), max_motif_nets_in_channel as i64, max_motif_batches, &Cpu);
        let channel_1_t = apply_channel_motifs_with_padding(step, &mut motif_nets[1].1, t.i((1, ..)), motif_width, max_motif_nets_in_channel as i64, slices_len, &Cpu)?;

        debug!("test_apply_motif_models: max_motif_batches={:?}", max_motif_batches);

        let channel_1_motifs = Tensor::stack(&[
            Tensor::from_slice(&[99000., 299000., 499000., 699000., 899000., 1.099e+06, 1.299e+06, 1.499e+06, 1.699e+06, 1.899e+06]),  //ch 1 net 0 (ResourceId=1)
            Tensor::from_slice(&[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),  //ch 1 net 0 (ResourceId=1)
        ], 0
        ).totype(Float).view((2, 10));
        assert_eq!(channel_1_motifs, channel_1_t);

        debug!("test_apply_motif_models: channel_0_motifs.size()={:?}, channel_1_motifs.size()={:?}", channel_0_motifs.size(), channel_1_motifs.size());

        let channels_motifs = apply_motif_nets_with_padding(&t, Arc::new(RwLock::new(motif_nets)), &Cpu, input_n as u32).squeeze();

        assert_eq!(
            Tensor::stack(&[channel_0_motifs, channel_1_motifs], 0),
            channels_motifs
        );

        Ok(())
    }

    #[test]
    fn slow_test_apply_motif_models_2() -> Result<(), ThoosaError> {
        unsafe {
            let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        // set_max_level(Trace);
        let input_n = 2_000_000_000;//TODO calculated from free mem
        let motif_width = 20;
        let input_slice_step = 10;
        let input_slices_motif_batch_size = 5;

        let max_dists_len = slices_len(input_n, motif_width, input_slice_step);

        let channel = lin_t(input_n).to_device(Cuda(0));
        let unfolded = channel.unfold(0, motif_width, input_slice_step as i64);
        let sums = unfolded.sum_dim_intlist(1, true, Float).view(max_dists_len);//sums of the slices should be what test_net produces
        debug!("tensor_size_in_bytes={:?}M ", tensor_mem_size(&unfolded)/1024/1024);
        trace!("unfolded=\n{unfolded} \nsums=\n{}", sums);

        let mut motif_nets = vec![
            (MotifNodeParams::new(motif_width, 1), ThoModuleWithStore::new(test_net(motif_width, 1., Cuda(0))?))
        ];

        let channel_1_t = apply_channel_motifs(&mut motif_nets, channel, motif_width, 1, max_dists_len, &Cuda(0))?.view(max_dists_len);

        trace!("channel_1_t=\n{channel_1_t}");

        assert!(sums.allclose(&channel_1_t, 0.1, 0.1, true));


        Ok(())
    }

    #[test]
    fn test_apply_motif_models_diff_len() -> Result<(), ThoosaError>{
        let input_n_0 = 1000;
        let motif_width= 120;
        let t_0 = lin_test_tensor(input_n_0).unsqueeze(0);
        let t_1 = lin_test_tensor(831).unsqueeze(0);

        debug!(" t_0.size()={:?}, t_1.size()={:?}", t_0.size(), t_1.size());

        let motif_batch_size = 13;
        let step= 100;

        let motif_nets = Arc::new(RwLock::new(vec![
            (FindMotifsRunConfig::new(Some(1), step as u32, motif_width as u32, 0.1, "test", None),
             vec![
                 (MotifNodeParams::new(motif_width, 1), ThoModuleWithStore::new(test_net(motif_width, 1., Cpu)?)), (MotifNodeParams::new(motif_width, 1), ThoModuleWithStore::new(test_net(motif_width, 1., Cpu)?))
             ]),
            (FindMotifsRunConfig::new(Some(1), step as u32, motif_width as u32, 0.1, "test", None),
             vec![(MotifNodeParams::new(motif_width, 1), ThoModuleWithStore::new(test_net(motif_width, 10., Cpu)?))], //channel 1 nets
            ),
            (FindMotifsRunConfig::new(Some(1), step as u32, motif_width as u32, 0.1, "test", None),
             vec![]
            )
        ]));

        let max_dists_len = (input_n_0-motif_width)/step as i64 + 1;


        let channels_motifs_0 = apply_motif_nets(&t_0, Arc::clone(&motif_nets), &Cpu, input_n_0 as u32);

        debug!("channels_motifs_0.size={:?}", channels_motifs_0.size());

        let channels_motifs_1 = apply_motif_nets(&t_1, Arc::clone(&motif_nets), &Cpu, input_n_0 as u32);

        debug!("channels_motifs_1.size={:?}", channels_motifs_1.size());

        // assert_eq!(vec![3, 2, max_dists_len], channels_motifs_0.size());//channel, motif nets, batches, batch size
        assert_eq!(t_0.size(), channels_motifs_0.size());//channel, motif nets, batches, batch size
        // assert_eq!(vec![3, 2,  (831-motif_width)/step as i64 + 1], channels_motifs_1.size());//TODO
        assert_eq!(vec![1, 5,  input_n_0], channels_motifs_1.size());//TODO

        Ok(())
    }

    //5 ch
    fn lin_test_tensor(input_n: i64) -> Tensor {
        Tensor::vstack(&[
            Tensor::from_slice((0..input_n).map(|i| i as f32).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((0..input_n).map(|i| i as f32 * 2.).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((0..input_n).map(|i| i as f32 * 3.).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((0..input_n).map(|i| i as f32 * 4.).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((0..input_n).map(|i| i as f32 * 5.).collect::<Vec<_>>().as_slice())
        ])
    }

    fn load_test_tensor(id:&Resource, input_n: i64) ->Box<Tensor> {
        debug!("test_load_tensor: {id:?}");

        Box::new(lin_test_tensor(input_n))
    }

    #[test]
    fn test_train_label() -> Result<(), ThoosaError> {
        set_max_level(Debug);
        let ws = Arc::new(RwLock::new(DummyWorkspace::new(2, 1000)?));

        const BATCH_SIZE:usize = 10;
        const CHANNELS:usize = 2;
        const MAX_MOTIF_NETS:usize = 5;
        let channel_motif_nets = [3,2,1];//TODO

        // let mut rnd = thread_rng();

        const TRAIN_LABELS: [bool; BATCH_SIZE] = build_labels::<BATCH_SIZE>();
        const VALID_LABELS: [bool; BATCH_SIZE] = build_labels::<BATCH_SIZE>();

        const MAX_SLICES_LEN:usize = slices_len(1100, 100, 100) as usize;


        #[derive(Debug)]
        struct DummyLabelBatch {}

        impl DummyLabelBatch {
            pub fn new() -> Self {
                Self {}
            }
        }

        impl DataBatch for DummyLabelBatch {

            fn train_batch(&self) -> (Tensor, Tensor) {
                (
                    dummy_labels_net_input(&TRAIN_LABELS, MAX_MOTIF_NETS, MAX_SLICES_LEN),
                    Tensor::from_slice(TRAIN_LABELS.as_slice()).totype(Float)
                )
            }

            fn valid_batch(&self) -> (Tensor, Tensor) {
                (
                    dummy_labels_net_input(&TRAIN_LABELS, MAX_MOTIF_NETS, MAX_SLICES_LEN),
                    Tensor::from_slice(VALID_LABELS.as_slice()).totype(Float)
                )
            }
        }

        impl Splittable<1> for DummyLabelBatch {
            fn split(&self) -> Option<[Self; 1]>
            where
                Self: Sized
            {
                Some([DummyLabelBatch::new()])
            }
        }

        impl SplittableDataBatch for DummyLabelBatch {}

        let (loss, params) = super::train_label_nodes(
            &ws,
            &TrainLabelNodesValues::new(LabelNetType::FixedNumSlices(FixedNumSlicesNetParams{
                num_layers: 3.into(),
                slices_len: 100.into(),
            }), TrainNodesValues::new(1e-3, 3, Some(AdamW::default().amsgrad(true)))),
            &CommonTrainLabelNodeRunConfig::new(BATCH_SIZE, 10),
            &mut DummyLabelBatch::new(),
            &Cpu,
            CHANNELS as i64,
            MAX_MOTIF_NETS as i64, //max_nets
            &VarStore::new(Cpu),
            MAX_SLICES_LEN as i64
        )?;

        debug!("test_train_label: loss={loss:?} params={params:?}");

        assert!(loss < 1.0);

        Ok(())
    }


    fn dummy_0(data: &mut [f64; 11]) -> (Tensor, f32) {
        (Tensor::from_slice(data).to_kind(Float), 0.0)
    }

    fn dummy_1(data: &mut [f64; 11]) -> (Tensor, f32) {
        (Tensor::from_slice(data).to_kind(Float) + 1.0, 1.0)
    }

    #[test]
    fn test_gpt()-> Result<(), ThoosaError> {
        let _device = CudaDevice::new(0)?;
        let cudarc_lib = unsafe {
            let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        };
        // Choose your device: either CPU or CUDA(0) if you have a GPU.
        let device = Device::Cuda(0); // or Device::Cpu

        // Create a variable store for the model parameters.
        let vs = nn::VarStore::new(device);

        // Define a simple feed-forward network:
        //   Input(11) -> Linear(11->16) -> ReLU -> Linear(16->8) -> ReLU
        //   -> Linear(8->1) -> Sigmoid
        let net = nn::seq()
            .add(nn::linear(&vs.root(), 11, 16, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(&vs.root(), 16, 8, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(&vs.root(), 8, 1, Default::default()))
            .add_fn(|x| x.sigmoid());

        // Generate some synthetic data for binary classification.
        // We'll say: 50% chance we add 100.0 to the inputs and label = 1,
        // otherwise label = 0.
        let mut rng = thread_rng();
        let (input_vec, target_vec): (Vec<Tensor>, Vec<f32>) = (0..20)
            .map(|_| {
                // each "row" is 11 features
                let mut data = [0.0; 11];
                rng.fill(&mut data);

                if rng.gen_bool(0.5) {
                    dummy_1(&mut data)
                } else {
                    dummy_0(&mut data)
                }
            })
            .unzip();

        // Stack the data along dimension 0 => shape [20, 11]
        let input = Tensor::stack(&input_vec, 0).to_device(device);
        // Convert targets to a Tensor => shape [20]
        let target = Tensor::from_slice(&target_vec).to_device(device);

        // Create an optimizer, e.g., SGD with learning rate 1e-2
        let mut opt = nn::Sgd::default().build(&vs, 1e-2)?;

        // Training loop
        let epochs = 1000;
        for epoch in 0..epochs {
            // Forward pass
            let logits = net.forward(&input); // shape [20, 1]
            // Binary cross-entropy expects (preds, target),
            // so we unsqueeze target to match shape [20, 1].
            let loss = logits.binary_cross_entropy(
                &target.unsqueeze(-1),
                None::<Tensor>,
                Reduction::Mean,
            );

            // Backprop + update weights
            opt.backward_step(&loss);

            // Print the loss occasionally
            if epoch % 50 == 0 {
                println!("Epoch {:3}, loss: {:8.6}", epoch, loss.double_value(&[]));
            }
        }

        let mut data = [0.0; 11];
        rng.fill(&mut data);

        // Example: do a forward pass on the same data after training
        println!("Predictions 1:\n{:?}", net.forward(&Tensor::stack(&[
            dummy_1(&mut data).0
        ], 0).to_device(device)).squeeze().to_device(Cpu));

        println!("Predictions 0:\n{:?}", net.forward(&Tensor::stack(&[
            dummy_0(&mut data).0
        ], 0).to_device(device)).squeeze().to_device(Cpu));

        Ok(())
    }

    fn asc_t<const S:usize>(mut data: [f32; S]) -> Tensor {
        Tensor::from_slice((0..S).map(|i| data[i] + i as f32 * 2.).collect::<Vec<_>>().as_slice()).to_kind(Float)
    }

    fn desc_t<const S:usize>(mut data: [f32; S]) -> Tensor {
        Tensor::from_slice((0..S).rev().map(|i| data[i] + i as f32 * 2.).collect::<Vec<_>>().as_slice()).to_kind(Float)
    }

    #[test]
    fn test_train_basic_oom()->Result<(), ThoosaError> {
        reset_print_options();
        set_max_level(Debug);
        let _device = CudaDevice::new(0)?;
        let cudarc_lib = unsafe {
            let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        };
        // let batch_size = 10000;
        // const FEATURES:i64 = 2i64.pow(10);

        let batch_size = 10;
        const FEATURES:i64 = 10;

        let res_seq_0 = SeqBuilder::new(FEATURES, None)
            // .func(|t|{
            //     debug!("res_seq_0: t={t:?}");
            //     t.shallow_clone()
            // })
            .batch_norm(BatchNormConfig::default())
            .linear_s(FEATURES*20)
            .func(Tensor::relu)
            .linear_s(FEATURES*16)
            .func(Tensor::relu)
            .linear_s(10)
            // .func(Tensor::sigmoid)
            .end_seq();

        let res_seq_1 = SeqBuilder::new(FEATURES, None)
            // .func(|t|{
            //     debug!("res_seq_1: t={t:?}");
            //     t.shallow_clone()
            // })
            .batch_norm(BatchNormConfig::default())
            .linear_s(FEATURES*20)
            .func(Tensor::relu)
            .linear_s(FEATURES*16)
            .func(Tensor::relu)
            .linear_s(10)
            // .func(Tensor::sigmoid)
            .end_seq();


        let full = SeqBuilder::new(FEATURES, None)

            .split(
                vec![
                    res_seq_0,
                    res_seq_1
                ],

                Box::new(
                    move |t:&Tensor| {
                        debug!("split: t={t:?}");
                        vec![
                            t.i((.., 0, ..)),
                            t.i((.., 1, ..))
                        ]
                    }
                ),//label batch, channel, max_motif_nets, max slices

                Box::new(move |t:Vec<Tensor>| {
                    debug!("join: in t={t:?}");
                    let t = Tensor::concat(t.as_slice(), 1);
                    debug!("join: t={t:?}");
                    t
                })

            )
            .linear_feats(20, 20)
            .func(Tensor::relu)
            .linear_feats(20, 1)
            .func(Tensor::sigmoid)
            .end_seq();

        let mut vs_0 = VarStore::new(Cuda(0));


        let root = vs_0.root();
        let net_full = full.build_tch_module(&root)?; //&(root / "vs_0"))?;

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
                _ = vs_opt_split_0.root().add(&name.replace(".", "/"), t, !name.contains("batch"));
            } else if name.contains("split_1") {
                _ = vs_opt_split_1.root().add(&name.replace(".", "/"), t, !name.contains("batch"));
            } else {
                _ = vs_opt_main.root().add(&name.replace(".", "/"), t, !name.contains("batch"));
            }
        }

        debug!("vs_opt_main={vs_opt_main:?}");
        debug!("vs_opt_split_0={vs_opt_split_0:?}");
        debug!("vs_opt_split_1={vs_opt_split_1:?}");

        // let mut opt_0 = AdamW::default().beta1(0.9).beta2(0.999).build(&(vs_0.root() /"seq_0").into(), 1e-3)?;
        let mut opt_main = AdamW::default().beta1(0.9).beta2(0.999).build(&vs_opt_main, 1e-3)?;
        let mut opt_split_0 = AdamW::default().beta1(0.9).beta2(0.999).build(&vs_opt_split_0, 1e-3)?;
        let mut opt_split_1 = AdamW::default().beta1(0.9).beta2(0.999).build(&vs_opt_split_1, 1e-3)?;

        // vs_0.freeze();
        // set_requires_grad_in_subpath(&vs_0, "seq_0", false);
        // set_requires_grad_in_subpath(&vs_0, "seq_1", false);
        // set_requires_grad_in_subpath(&vs_0, "seq_3", false);
        // set_requires_grad_in_subpath(&vs_0, "seq_0.split_1", false);
        let mut split_i =0;
        for epoch in 0..1000 {
            // if epoch % 2 == 0 {
            //     set_requires_grad_in_subpath(&vs_0, "seq_0.split_1", false);
            //     set_requires_grad_in_subpath(&vs_0, "seq_0.split_0", true);
            // } else {
            //     set_requires_grad_in_subpath(&vs_0, "seq_0.split_1", true);
            //     set_requires_grad_in_subpath(&vs_0, "seq_0.split_0", false);
            // }

            let mut rng = thread_rng();

            let (input_0, input_1, target):(Vec<Tensor>, Vec<Tensor>, Vec<f32>) = (0..batch_size).map(|i| {

                let mut data = [0.; FEATURES as usize];
                rng.fill(&mut data);

                if rng.gen_bool(0.5) {
                    (
                        asc_t(data),
                        desc_t(data),
                        0.
                    )
                } else {
                    (
                        desc_t(data),
                        asc_t(data),
                        1.
                    )
                }
            }).fold((Vec::new(), Vec::new(), Vec::new()), |mut a, (i_0, i_1, label)| {
                a.0.push(i_0);
                a.1.push(i_1);
                a.2.push(label);
                a
            });

            let input_0 = Tensor::stack(input_0.as_slice(), 0).to_device(Cuda(0));
            let input_1 = Tensor::stack(input_1.as_slice(), 0).to_device(Cuda(0));
            let input = Tensor::stack(&[input_0, input_1], 0).transpose(0,1);

            trace!("input=\n{input}");

            let target = Tensor::from_slice(target.as_slice()).view((-1,1)).to_device(Cuda(0));

            trace!("target=\n{target}");

            let pred = net_full.tho_forward(&input.totype(kind), true);

            let loss = pred.binary_cross_entropy(&target.totype(kind), None::<Tensor>, Reduction::Mean);

            if split_i == 0 {
                //TODO off-load to CPU
                opt_split_0.backward_step(&loss);

            } else if split_i ==1 {
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
            split_i+=1;
            if split_i > 2 {
                split_i = 0;
            }
        }
        debug!("opt_0 optimizer_mem_size: {:?}", optimizer_mem_size(&opt_main));
        // debug!("cuda_mem: {:?}", tch::Cuda::memory_allocated());

        vs_0.freeze();
        vs_0.set_kind(Float);

        // debug!("opt_1 optimizer_mem_size: {:?}", optimizer_mem_size(&opt_1));

        let mut rng = thread_rng();

        // let mut results = vec![];
        for i in 0..100 {
            let mut data = [0.; FEATURES as usize];
            rng.fill(&mut data);

            let pred_0 = net_full.tho_forward(&Tensor::stack(&[
                Tensor::stack(&[asc_t(data), desc_t(data)], 0), //0
                // Tensor::stack(&[desc_t(data), asc_t(data)], 0), //1
            ], 1).transpose(0, 1).to_device(Cuda(0)), false); //0
            info!("pred_0=\n{pred_0}");

            let pred_1 = net_full.tho_forward(&Tensor::stack(&[
                // Tensor::stack(&[asc_t(data), desc_t(data)], 0), //0
                Tensor::stack(&[desc_t(data), asc_t(data)], 0), //1
            ], 1).transpose(0, 1).to_device(Cuda(0)), false); //1
            info!("pred_1=\n{pred_1}");

            assert!(pred_0.squeeze().double_value(&[]) < 0.5);
            assert!(pred_1.squeeze().double_value(&[]) > 0.5);

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



    #[test]
    fn test_train_oom()->Result<(), ThoosaError>{

        let _device = CudaDevice::new(0)?;
        let cudarc_lib = unsafe {
            let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        };

        let ws = Arc::new(RwLock::new(DummyWorkspace::new(2, 1000)?));

        const BATCH_SIZE:usize = 1;
        const CHANNELS:usize = 2;
        const MAX_MOTIF_NETS:usize = 5;

        let channel_motif_nets = [3,2,1];//TODO

        // let mut rnd = thread_rng();

        const TRAIN_LABELS: [bool; BATCH_SIZE] = build_labels::<BATCH_SIZE>();
        const VALID_LABELS: [bool; BATCH_SIZE] = build_labels::<BATCH_SIZE>();

        const MAX_SLICES_LEN:usize = slices_len(1000, 100, 100) as usize;

        #[derive(Clone, Debug)]
        struct DummyLabelBatch {}

        impl DummyLabelBatch {
            pub fn new() -> Self {
                Self {}
            }
        }

        impl DataBatch for DummyLabelBatch {

            fn train_batch(&self) -> (Tensor, Tensor) {
                (
                    dummy_labels_net_input(&TRAIN_LABELS, MAX_MOTIF_NETS, MAX_SLICES_LEN).to_device(Cuda(0)),
                    Tensor::from_slice(TRAIN_LABELS.as_slice()).totype(Float).to_device(Cuda(0))
                )
            }

            fn valid_batch(&self) -> (Tensor, Tensor) {
                (
                    dummy_labels_net_input(&TRAIN_LABELS, MAX_MOTIF_NETS, MAX_SLICES_LEN).to_device(Cuda(0)),
                    Tensor::from_slice(VALID_LABELS.as_slice()).totype(Float).to_device(Cuda(0))
                )
            }
        }

        impl Splittable<1> for DummyLabelBatch {
            fn split(&self) -> Option<[Self; 1]>
            where
                Self: Sized
            {
                Some([DummyLabelBatch::new()])
            }
        }

        impl SplittableDataBatch for DummyLabelBatch{}

        let batch = DummyLabelBatch::new();
        let common_train_label_node_run_config = CommonTrainLabelNodeRunConfig::new(BATCH_SIZE, 10);

        let r= opt_train(

            |vs:&VarStore, run_config_values:&TrainLabelNodesValues, listener: &mut PlotVecf32|{
                let mut r = None::<(f32, LabelNodeParams)>;
                run_split_on_oom(
                    Reducer::new(batch.clone()),
                    |batch|{
                        super::train_label_nodes(
                            &ws,
                            run_config_values,
                            &common_train_label_node_run_config,
                            batch,
                            &Cuda(0),
                            CHANNELS as i64,
                            MAX_MOTIF_NETS as i64, //max_nets
                            &VarStore::new(Cuda(0)),
                            MAX_SLICES_LEN as i64
                        )
                    },
                    |v, i|{
                        debug!("v={v:?}");
                        r.replace(v);
                    },
                    3
                )?;

                Ok(r.unwrap())
            },
            &ValuesOrOptSpace::Values(TrainLabelNodesValues::new(LabelNetType::FixedNumSlices(FixedNumSlicesNetParams{
                num_layers: 3.into(),
                slices_len: 100.into(),
            }), TrainNodesValues {lr:0.01, num_layers:1, adam_w: None})),
            Cpu,
            "label"
        );

        debug!("test_train_oom: r={r:?}");


        Ok(())
    }

    #[test]
    fn test_train_train_motifs_nodes_params() {
        let train_motifs_nodes_params = TrainMotifsNetsParams::dummy(vec![1], 10, 0.1, 10, 0.1, 1, 10, 30);

        let run_config = train_motifs_nodes_params.to_run_configs();

        debug!("test_train_motifs_nodes_params: run_config={run_config:?}");

        // let train_nodes_values = TrainNodesValues::new(0.1, 3, Some(AdamW::default()));
        // let train_nodes_values_map:HashMap<String, Val> = train_nodes_values.into();

        // debug!("test_train_nodes_values_to_hashmap:{train_nodes_values_map:#?}");
    }

    fn dummy_labels_net_input(labels:&[bool], max_motif_nets:usize, slices_len:usize) ->Tensor{
        //batch size x channels x max motif nets x slices_len
        let mut labels_iter = labels.iter();
        Tensor::stack(
            labels.iter().map(|label|
                                  // Tensor::stack((0..channels).map(|mb|
                                  Tensor::stack((0..max_motif_nets).map(|mmn|
                                      if *label {
                                          motifs_1(slices_len)
                                      } else {
                                          motifs_0(slices_len)
                                      }
                                  ).collect::<Vec<_>>().as_slice(), 0)
                              // ).collect::<Vec<_>>().as_slice(), 0)
            ).collect::<Vec<_>>().as_slice(),
            0,
        )
    }

    fn motifs_1(n: usize) -> Tensor {
        // Tensor::rand(n as i64, FLOAT_CPU).multiply_scalar(0.1).f_add_scalar(0.8).unwrap()
        Tensor::ones(n as i64, FLOAT_CPU)
    }

    fn motifs_0(n: usize) -> Tensor {
        // Tensor::rand(n as i64, FLOAT_CPU).multiply_scalar(0.01)
        Tensor::zeros(n as i64, FLOAT_CPU)
    }

    const fn build_labels<const BATCH_SIZE:usize>() -> [bool; BATCH_SIZE] {
        let mut arr = [false; BATCH_SIZE];
        let mut i = 0;
        while i < BATCH_SIZE {
            arr[i] = i % 2 == 0;
            i += 1;
        }
        arr
    }

    fn load_test_channels_motifs_models(ws1: &Arc<RwLock<Box<dyn Workspace>>>, channels_motif_ids: Vec<(usize, usize)>) -> Vec<(MotifNodeParams, ThoModuleWithStore)> {

        channels_motif_ids.into_iter().flat_map(|(channel, motif_id)| load_motif_models(ws1, channel, motif_id, &Cpu).expect("load_motif_models")).collect()
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

        const BATCH_SIZE:usize = 10;
        const LABELS_SIZE:usize = 6;


        impl DataBatch for DummyLabelBatch {

            fn train_batch(&self) -> (Tensor, Tensor) {
                let train_batch = dummy_final_net_input::<LABELS_SIZE>(BATCH_SIZE);
                trace!("train_batch=\n{} \n{}", train_batch.0, train_batch.1);
                train_batch
            }

            fn valid_batch(&self) -> (Tensor, Tensor) {
                let valid_batch= dummy_final_net_input::<LABELS_SIZE>(BATCH_SIZE);
                trace!("valid_batch={} \n{}", valid_batch.0, valid_batch.1);
                valid_batch
            }
        }

        let (loss, final_params) = super::train_final_node(
            &Arc::new(RwLock::new(DummyWorkspace::new(0, 100)?)),
            &TrainNodesValues ::new(
                1.0e-2,
                1u32,
                None
            ),
            &CommonTrainFinalNodeRunConfig::new(300, BATCH_SIZE),
            &mut DummyLabelBatch::new(),
            &Cpu,
            LABELS_SIZE,
            &VarStore::new(Cpu)
        )?;

        assert!(loss < 0.5);
        debug!("test_train_final: loss={loss:?} final_params={final_params:?}");
        // fs::remove_dir_all(dir)?;

        Ok(())
    }

    fn dummy_final_net_input<const LABELS_SIZE:usize>(batch_size: usize) -> (Tensor, Tensor) {
        let mut rng = thread_rng();
        let data = (0..batch_size).map(|label| {
            let label = rng.gen_range(0..LABELS_SIZE);

            let x = random_tensor_with_dominant(LABELS_SIZE as i64, label as i64, 0.99, Cpu);
            let mut y = [0.; LABELS_SIZE];
            y[label] = 1.0;
            let y = Tensor::from_slice(y.as_slice());

            (x, y)
        }).collect::<Vec<_>>();

        (
            Tensor::stack(data.iter().map(|d|d.0.shallow_clone()).collect::<Vec<_>>().as_slice(), 0),
            Tensor::stack(data.iter().map(|d|d.1.shallow_clone()).collect::<Vec<_>>().as_slice(), 0),
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
    pub fn random_tensor_with_dominant(
        n: i64,
        m: i64,
        fraction: f64,
        device: Device
    ) -> Tensor {
        assert!(m < n, "m must be < n");
        assert!(fraction > 0.0 && fraction < 1.0, "fraction must be in (0,1)");

        let remainder = 1.0 - fraction;

        // Generate n-1 random values in [0,1]
        let other_values = Tensor::rand(&[n - 1], (Kind::Float, device));
        // Normalize them so they sum up to `remainder`
        let sum_others = other_values.sum(Kind::Float);
        let scaled_others = &other_values / &sum_others * remainder;

        // Create an output tensor of zeros
        let mut out = Tensor::zeros(&[n], (Kind::Float, device));

        // Fill the m-th entry with the dominant fraction
        out.get(m).fill_(fraction);

        // Prepare indices for all entries except m
        let indices: Vec<i64> = (0..n).filter(|&idx| idx != m).collect();

        // Put the scaled_others values into out at positions other than m
        out.index_put_(&[Some(Tensor::from_slice(indices.as_slice()))], &scaled_others, false);

        out
    }

    //6 corrs 4 motif models
    //TODO remove?
    fn save_test_models(ws: &Arc<RwLock<Box<dyn Workspace>>>, base_run_id: RunId, channels: usize, max_motif_batches: i64, max_batch_size: i64, motif_node_params_run_id: usize, window: i64) -> Result<(), ThoosaError> {
        // fs::create_dir_all(models_path)?;

        // fs::write(models_path.join("motif_model.json"), serde_json5::to_string(&MotifNodeParams::new(window))?)?;


        save_test_label_model(&ws, base_run_id + 1, "seizure", 3, 4, max_motif_batches, max_batch_size)?;
        // fs::write(models_path.join("motif_model.json"), serde_json5::to_string(&MotifNodeParams::new(window))?)?;


        save_test_motif_model(ws, "0", 0, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "0", 1, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "0", 2, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "0", 3, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "1", 10, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "1", 11, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "2", 20, window, motif_node_params_run_id)?;

        save_test_label_model(ws, base_run_id + 1, "lpd", 1, 1, max_motif_batches, max_batch_size)?;

        save_test_motif_model(ws, "0", 0, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "1", 10, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "2", 20, window, motif_node_params_run_id)?;


        save_test_label_model(ws, base_run_id + 1, "gpd", 1, 1, max_motif_batches, max_batch_size)?;

        save_test_motif_model(ws, "0", 0, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "1", 10, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "2", 20, window, motif_node_params_run_id)?;


        save_test_label_model(ws, base_run_id + 1, "lrda", 1, 1, max_motif_batches, max_batch_size)?;

        save_test_motif_model(ws, "0", 0, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "1", 10, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "2", 20, window, motif_node_params_run_id)?;


        save_test_label_model(ws, base_run_id + 1, "grda", 1, 1, max_motif_batches, max_batch_size)?;

        save_test_motif_model(ws, "0", 0, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "1", 10, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "2", 20, window, motif_node_params_run_id)?;


        save_test_label_model(ws, base_run_id + 1, "other", 1, 1, max_motif_batches, max_batch_size)?;

        save_test_motif_model(ws, "0", 0, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "1", 10, window, motif_node_params_run_id)?;
        save_test_motif_model(ws, "2", 20, window, motif_node_params_run_id)?;
        Ok(())
    }


    pub fn save_test_tensor(input_path: &PathBuf, name: &str) {
        fs::create_dir_all(input_path).unwrap();
        fs::write(&input_path.join(name), "lalalal").unwrap();
    }

    fn save_test_final_model(models_path: &PathBuf, features: i64) -> Result<(), ThoosaError> {
        fs::create_dir_all(models_path.clone())?;

        let mut vs_0 = nn::VarStore::new(Cpu);
        let _ = train_nets::final_node(&FinalNodeParams::new(features, 1)).build_tch_module(&vs_0.root())?;

        vs_0.freeze();
        debug!("save_test_final_model: features={} vs_0.variables()={:?}", features, vs_0.variables());
        vs_0.save(models_path.join("final.safetensors"))?;
        Ok(())
    }

    fn save_test_label_model(ws: &Arc<RwLock<Box<dyn Workspace>>>, run_id: RunId, label:&str, channels: usize, max_motif_nets: i64, max_motif_batches: i64, motif_batch_size: i64) -> Result<(), ThoosaError> {
        let label_node_params = LabelNodeParams::new(channels as i64, max_motif_nets, 0, 0);//TODO slices_len, motif_layers

        let mut vs_0 = nn::VarStore::new(Cpu);
        let _ = train_nets::label_node(&label_node_params).build_tch_module(&vs_0.root())?;
        vs_0.freeze();
        debug!("save_test_corr_model: channels={}, max_motif_nets={}, max_motif_batches={}, motif_batch_size={} vs_0.variables()={:?}", channels, max_motif_nets, max_motif_batches, motif_batch_size, vs_0.variables());

        ws.write().save_label_model(run_id, &label, &vs_0, &label_node_params );
        Ok(())
    }

    fn save_test_motif_model(ws: &Arc<RwLock<Box<dyn Workspace>>>, channel: &str, motif_id: usize, window: i64, motif_node_params_run_id: usize) -> Result<(), ThoosaError> {

        let mut vs_0 = nn::VarStore::new(Cpu);
        let _test_motif_model = train_nets::motif_node(&MotifNodeParams::new(window, 1)).build_tch_module(&vs_0.root())?;

        let test_motif_model = SeqBuilder::new(100, None)
            .linear_s_with_conf(1, LinearConfig {
                ws_init: init::Init::Const(1.0),
                bs_init: None,
                bias: false
            })
            .end_seq().build_tch_module(&vs_0.root())?;


        vs_0.freeze();
        debug!("save_test_motif_model: channel={channel} vs_0.variables()={:?}", vs_0.variables());
        ws.write().save_motif_model(&channel, &motif_node_params_run_id, &motif_id, &vs_0, &MotifNodeParams::new(100, 1))?;
        Ok(())
    }

    #[test]
    fn test_is_nonzero() -> Result<(), ThoosaError> {

        assert!(super::is_nonzero(&Tensor::from_slice(&[0., 1., 0.]))?);
        assert!(super::is_nonzero(&Tensor::stack(&[Tensor::from_slice(&[0., 1., 0.]), Tensor::from_slice(&[0., 0., 0.])], 0))?);
        assert!(!super::is_nonzero(&Tensor::from_slice(&[0., 0., 0.]))?);

        Ok(())
    }

    #[test]
    fn test_batch_transform(){
        let n = 1000;
        let t_0 = Tensor::from_slice(pulse_in_rand(n, &[], 0.05, 1., 0.1).as_slice());
        let t_1 = Tensor::from_slice(sin_v(n, 22.).as_slice());



        let batch = Rc::new(Box::new(DummyDs::new(
            t_0.shallow_clone(),
            t_1.shallow_clone(),
        )));

        let mut some_v = 0;

        let batch = BatchTransform::new(Rc::clone(&batch), Box::new(|t|{
            some_v+=1;
            t
        }));

        //TODO
    }

}