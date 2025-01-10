use parking_lot::{Mutex, RawRwLock, RwLock};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use std::fs;
use std::hash::Hash;
use std::iter::Map;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;
use clap::clap_derive::Args;
use crossbeam_utils::sync::WaitGroup;
use futures::FutureExt;
use futures::StreamExt;
use log::{debug, info, trace, warn, Log};
use plotters::prelude::BLACK;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tch::nn::{AdamW, ConvConfig, Module, OptimizerConfig, VarStore};
use tch::Kind::Float;
use tch::{nn, Device, IndexOp, Kind, Reduction, Tensor};
use tch::Device::{Cpu, Cuda};
use crate::builder::{Node, SeqBuilder};
use crate::loader::{LoadRunner, Loader};
use crate::my_tch::{apply_channel_motifs, ThoModule};
use crate::parzen::optimizer::{ConfigurationSpace, Hyperparameter, Primitives};
use crate::runs::{Empty, Labeled, Logits, NetResources, Params, RunConfig, SavableOptResult, Simple};
use crate::train_nets;
use crate::utils::{cuda_mem, from, tensor_size_in_bytes};
use crate::utils::{pad_to, plot_series, train, DataBatch, ListenerResponse, TrainListener};
use crate::ws::{FsWorkspace, Workspace};
use thoosa_shared::parse_key_val;
use thoosa_shared::parse_range_or_value;
use thoosa_shared::resource::{Resource, ResourceId};
use thoosa_shared::val::Val;
use thoosa_shared::val::{prefix_properties, unprefix_properties};
use thoosa_shared::{batch_windows, clc_num_batches, make_combos, RangeOrValue, ThoosaError};
use thoosa_store::Store;


#[derive(Debug, Args, Clone)]
pub struct TrainLabelNodeParams {


    #[arg(long)]
    pub select_motifs_for_label_run_ids: Vec<usize>,

    #[arg(long)]
    train_resource_group: String,

    #[arg(long)]
    valid_resource_group: String,

    #[arg(long,  default_value_t = 100)]
    input_slice_step: usize,

    #[arg(long, default_value_t = 10)]
    cycles: usize,

    #[arg(long, value_parser=parse_range_or_value::<f64>)]
    pub lr: RangeOrValue<f64>,

    #[arg(long, value_parser=parse_range_or_value::<u32>)]
    pub num_layers: RangeOrValue<u32>,

    #[arg(long, default_value_t = 10)]
    motif_batch_size:i64,

    #[arg(long, default_value_t = 5)]
    label_batch_size:usize,

    #[arg(long, default_value_t = 10)]
    pub opt_max_evals: u32,

    #[arg(long, default_value_t = 3)] //TODO validate opt_n_init < opt_max_evals
    pub opt_n_init: u32,
}

impl TrainLabelNodeParams {


    pub fn dummy(step: usize, motif_node_params_selection_run_ids: Vec<usize>, train_resource_group: String, valid_resource_group: String, motif_batch_size: i64, label_batch_size: usize, cycles: usize, lr: impl Into<RangeOrValue<f64>>, num_layers:impl Into<RangeOrValue<u32>>, opt_max_evals: u32, opt_n_init: u32) ->Self {
        TrainLabelNodeParams {
            select_motifs_for_label_run_ids: motif_node_params_selection_run_ids,
            input_slice_step: step,
            cycles,
            lr: lr.into(),
            num_layers:num_layers.into(),

            train_resource_group,
            valid_resource_group,
            motif_batch_size,
            label_batch_size,
            opt_max_evals,
            opt_n_init
        }
    }

    //TODO add explicit flag to enable optimization
    pub fn to_config_space(&self) -> Option<ConfigurationSpace> {
        [
            vec![
                (&self.lr).try_into().ok(),
                (&self.num_layers).try_into().ok(),
            ],
            adam_w_hyperparams().into_iter().map(|v|Some(v)).collect::<Vec<Option<Hyperparameter>>>(),
        ].concat().into_iter()
            .collect::<Option<Vec<Hyperparameter>>>()
            .map(|v|ConfigurationSpace::new(v))
    }
}


#[derive(Debug)]
pub struct TrainLabelNodeRunConfig {
    id:Option<usize>,
    run_config_or_opt_space: ValuesOrOptSpace<TrainNodesValues>
}

impl TrainLabelNodeRunConfig {

    pub fn from_values(id: Option<usize>, lr: f64, num_layers: u32) -> Self {
        Self {
            id,
            run_config_or_opt_space: ValuesOrOptSpace::Values(TrainNodesValues {lr, num_layers, adam_w: None}),
        }
    }

    pub fn from_opt_space(id: Option<usize>, opt_space:ConfigurationSpace, opt_config: OptConfig) -> Self {
        Self {
            id,
            run_config_or_opt_space: ValuesOrOptSpace::OptSpace {opt_space, opt_config},
        }
    }
}

impl RunConfig for TrainLabelNodeRunConfig {
    fn from_properties_map(run_id: usize, properties: HashMap<String, Val>) -> Self {
        todo!()
    }

    fn set_id(&mut self, id: usize) {
        self.id.replace(id);
    }

    fn type_name() -> &'static str {
        "LabelNode"
    }
}

#[derive(Debug)]
pub struct CommonTrainLabelNodeRunConfig {
    pub input_slice_step:usize,
    pub motif_batch_size:usize,
    pub label_batch_size:usize,
    pub cycles:usize,
}

impl CommonTrainLabelNodeRunConfig {

    pub fn new(input_slice_step: usize, motif_batch_size: usize,  label_batch_size:usize, cycles:usize) -> Self {
        Self { input_slice_step, motif_batch_size, label_batch_size, cycles}
    }
}

impl Params<TrainLabelNodeRunConfig, NetResources<Labeled>, CommonTrainLabelNodeRunConfig> for TrainLabelNodeParams {

    fn to_run_configs(&self) -> Vec<TrainLabelNodeRunConfig> {

        if let Some(config_space) = self.to_config_space() {

            debug!("TrainLabelNodeParams::to_run_configs: config_space={config_space:?}");
            vec![
                TrainLabelNodeRunConfig::from_opt_space(None, config_space, OptConfig::new(self.opt_max_evals, self.opt_n_init))
            ]
        } else {
            //TODO avoid collecting
            let v = make_combos(vec![
                self.lr.clone().into_iter().map(|v| v.into()).collect::<Vec<Val>>().into_iter(),
                self.num_layers.clone().into_iter().map(|v| v.into()).collect::<Vec<Val>>().into_iter(),
            ]).into_iter().map(|mut config_combo:Vec<Val>|{
                debug!("TrainLabelNodeParams::to_run_configs: config_combo={config_combo:?} ");
                let num_layers = config_combo.pop().unwrap().try_into().unwrap();
                let lr = config_combo.pop().unwrap().try_into().unwrap();
                TrainLabelNodeRunConfig::from_values(
                    None,
                    lr,
                    num_layers,
                )
            }).collect();

            debug!("TrainLabelNodeParams::to_run_configs: self.lr={:?} self.num_layers={:?} v={v:?}", self.lr, self.num_layers);

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
            self.input_slice_step,
            self.motif_batch_size as usize,
            self.label_batch_size,
            self.cycles
        )
    }
}
/**
* returns num labels * items_per_label x channels x motifs x motifs batch size
*/
struct LoadingTensorsDataBatch  {
    train: Rc<Vec<(i64, Vec <Resource>)>>,
    val: Rc<Vec<(i64, Vec <Resource>)>>,
    batch_size: usize,
    executor: Arc<parking_lot::RwLock< LoadRunner<Box<Tensor>>>>,
    device:Device,
    transform_fn:Option<Arc<RwLock<Box<dyn Fn(Box<Tensor>)->Box<Tensor> + Sync + Send >>>>
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

        let x_y_acc:Arc<Mutex<Vec<(Box<Tensor>, i64)>>> = Arc::new(Mutex::new(vec![]));

        if batch_size <  class_resources.len() {
            debug!("LoadingTensorsDataBatch::choose_for_batch: batch_size <  class_resources.len");

            for _ in 0..batch_size {
                let x_y_acc_in_exec = Arc::clone(&x_y_acc);
                let wg_exec = wg.clone();
                let transform_fn = transform_fn.as_ref().map(|v|Arc::clone(v));

                let ( class, resources) = class_resources.choose(&mut rng).expect("class_resources.choose failed");
                let resource_id = resources.choose(&mut rng).expect("resources.choose failed").clone();
                let class = *class;
                loader.write().exec(move |loader, _keys_purge| {
                    async move {
                        let t = Loader::<Box<Tensor>>::load_future(Arc::clone(&loader), resource_id, 0).await; //TODO cache area
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
            wg.wait();

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

                loader.write().exec(move |loader, _keys_purge| {
                    async move {
                        let t = Loader::<Box<Tensor>>::load_future(Arc::clone(&loader), resource_id, 0).await; //TODO cache area
                        let t = if let Some(transform) = transform_fn {
                            transform.read()(t)
                        } else {
                            t
                        };

                        x_y_acc_in_exec.lock().push((t, class));
                        drop(wg_exec);
                    }.boxed()
                });
            });

            x_y_acc
        });

        wg.wait();

        let mut x_y_vecs:Vec<(Box<Tensor>, i64)> = x_y_vecs_mx.lock().drain(..).collect();
        // x_y_vecs.shuffle(&mut rng);

        debug!("LoadingTensorsDataBatch::choose_for_batch: x_y_vecs.1.len={} batch_size={batch_size}", x_y_vecs.len());

        //TODO throw if not enough tensors for batch?
        (x_y_vecs.len()..batch_size).for_each(|i|{
            x_y_vecs.push(
                (
                    Box::new(x_y_vecs[i- x_y_vecs.len()].0.shallow_clone()),
                    x_y_vecs[i- x_y_vecs.len()].1
                )
            );

        });

        Self::vec_to_tensors(device, x_y_vecs)
    }

    fn vec_to_tensors(device: &Device, mut x_y_vecs: Vec<(Box<Tensor>, i64)>) -> (Tensor, Tensor) {
        let classes = Tensor::from_slice(x_y_vecs.iter().map(|e| e.1).collect::<Vec<_>>().as_slice()).to_device(*device);

        let tensor = Tensor::stack(
            x_y_vecs.into_iter().map(|e| e.0).collect::<Vec<_>>().as_slice(),
            0,
        ).to_device(*device).set_requires_grad(true);

        debug!("LoadingTensorsDataBatch::choose_for_batch: tensor.size={:?} tensor_size_in_bytes={:?} tensor.device()={:?}", tensor.size(), tensor_size_in_bytes(&tensor), tensor.device());

        (
            tensor.totype(Float),
            classes.totype(Float)
        )
    }
}

impl DataBatch for LoadingTensorsDataBatch {

    fn train_batch(&mut self) -> (Tensor, Tensor){
        tch::no_grad(|| {
            Self::choose_for_batch(&self.train, self.batch_size, Arc::clone(&self.executor), &self.device, self.transform_fn.clone())
        })
    }

    fn valid_batch(&mut self) -> (Tensor, Tensor){
        tch::no_grad(|| {
            Self::choose_for_batch(&self.val, self.batch_size, Arc::clone(&self.executor), &self.device, self.transform_fn.clone())
        })
    }
}


struct ClassTensorsLoaderData  {
    train_resources: Vec<(Resource, Vec<f32>)>,
    valid_resources: Vec<(Resource, Vec<f32>)>,
    executor: Arc<RwLock< LoadRunner<Box<Tensor>>>>,
    batch_size: usize,
    device:Device,
    transform_fn:Option<Box<dyn FnMut(Box<Tensor>)->Box<Tensor> + Sync + Send >>
}

impl ClassTensorsLoaderData  {

    pub fn new(
        train_loader: Vec<(Resource, Vec<f32>)>,
        valid_loader: Vec<(Resource, Vec<f32>)>,
        batch_size: usize,
        executor: Arc<RwLock< LoadRunner<Box<Tensor>>>>,
        device:Device,
        transform_fn:Option<Box<dyn FnMut(Box<Tensor>)->Box<Tensor> + Sync + Send >>
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

    fn train_batch(&mut self) -> (Tensor, Tensor){
        Self::batch(&mut self.train_resources, self.batch_size, Arc::clone(&self.executor), &self.device, self.transform_fn.as_mut())
    }

    fn valid_batch(&mut self) -> (Tensor, Tensor){
        Self::batch(&mut self.valid_resources, self.batch_size, Arc::clone(&self.executor), &self.device, self.transform_fn.as_mut())
    }
}

impl ClassTensorsLoaderData {

    fn batch(resources: &Vec<(Resource, Vec<f32>)>, batch_size: usize, executor: Arc<RwLock< LoadRunner<Box<Tensor>>>>,device: &Device, mut transform_fn:Option<&mut Box<dyn FnMut(Box<Tensor>)->Box<Tensor> + Sync + Send >>) -> (Tensor, Tensor) {
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

        inputs = if let Some(ref mut transform) = transform_fn {
            inputs.iter_mut().map(|t| transform(Box::new(t.shallow_clone()))).collect() //TODO par on devices
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
    train_label_nodes_run_config_id:usize,
    label:String,
    loss: f32,
    label_node_params: LabelNodeParams,
    train_motifs_nodes_values: Option<TrainNodesValues>,
    var_store: VarStore
}

impl TrainLabelNodesRunConfigResult {
    pub fn new(train_label_nodes_run_config_id:usize, label:String, loss: f32, label_node_params: LabelNodeParams, train_motifs_nodes_values: Option<TrainNodesValues>, var_store: VarStore) -> Self {
        Self {
            train_label_nodes_run_config_id,
            label,
            loss,
            label_node_params,
            train_motifs_nodes_values,
            var_store,
        }
    }
}

impl SavableOptResult for TrainLabelNodesRunConfigResult {
    fn value(&self) -> f32 {
        self.loss
    }

    fn save(&mut self, store: &Arc<Mutex<Box<dyn Store>>>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<usize, ThoosaError> {
        let run_config_id = match self.train_motifs_nodes_values.as_ref(){
            None => self.train_label_nodes_run_config_id,

            Some(train_motifs_nodes_values) => {
                store.lock().save_run_config(
                    Some(&[self.train_label_nodes_run_config_id]),
                    &TrainLabelNodeRunConfig::type_name(),
                    "Label",
                    "todo",
                    TrainLabelNodeRunConfig::from_values(Some(self.train_label_nodes_run_config_id), train_motifs_nodes_values.lr, train_motifs_nodes_values.num_layers).to_properties_map()
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
    train_labels_resources:Rc<Vec<(i64, Vec <Resource>)>>,//TODO i64 to 1/0 enum for binary class
    valid_labels_resources: Rc<Vec<(i64, Vec <Resource>)>>,
    device: &Device,
    executor: &Arc<RwLock< LoadRunner<Box<Tensor>>>>,
    channels_motifs_nodes:Arc<RwLock<Vec<Vec<(MotifNodeParams, SyncModule)>>>>,
    max_motif_batches:usize,
    max_nets:i64,
    label: &String
) -> Result<Box<dyn SavableOptResult>, ThoosaError> {

    let input_slice_step = comon_label_node_params.input_slice_step;
    let input_slices_motif_batch_size = comon_label_node_params.motif_batch_size as i64;
    let channels = channels_motifs_nodes.read().len() as i64;
    let device = device.clone();

    let mut batch = LoadingTensorsDataBatch::new(
        train_labels_resources,
        valid_labels_resources,

        comon_label_node_params.label_batch_size,
        Arc::clone(&executor),
        device.clone(),
        Some( Box::new(move |t:Box<Tensor>| {
            Box::new(apply_motif_nets(&t, Arc::clone(&channels_motifs_nodes), input_slice_step, input_slices_motif_batch_size, max_motif_batches, &device))
        }))

    );

    debug!("run_opt_train_label_nodes: label={label:?} run_config={run_config:?} comon_label_node_params={comon_label_node_params:?} channels={channels} cuda_mem={:?}", cuda_mem());

    opt_train(
        |vs:&VarStore, run_config_values:&TrainNodesValues, listener: &mut PlotVecf32|{
            train_label_nodes(
                ws, run_config_values, comon_label_node_params, &mut batch, &device, channels, max_motif_batches, max_nets, vs
            )
        },
        &run_config.run_config_or_opt_space,
        device,
        "label"
    ).map(|(var_store, label_node_params, train_label_nodes_values, min_loss)| {
        Box::new(TrainLabelNodesRunConfigResult::new(run_config.id.expect("run_config.id"), label.clone(), min_loss, label_node_params, train_label_nodes_values, var_store)) as Box<dyn SavableOptResult>
    })
}

pub fn train_label_nodes (
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    run_config_values:&TrainNodesValues,
    comon_label_node_params:&CommonTrainLabelNodeRunConfig,
    batch:&mut dyn DataBatch,
    device: &Device,
    channels:i64,
    max_motif_batches:usize,
    max_nets:i64,
    vs_0: &VarStore
) ->  Result<(f32, LabelNodeParams), ThoosaError> {

    let label_node_params = LabelNodeParams::new(channels, max_nets, max_motif_batches as i64, comon_label_node_params.motif_batch_size as i64);
    debug!("train_label_nodes: label_node_params={label_node_params:?} cuda_mem={:?}", cuda_mem());

    let device = device.clone();
    let channels = ws.read().channels();

    let motif_batch_size = comon_label_node_params.motif_batch_size as i64;
    let mut optimizer = AdamW::default().amsgrad(true);

    if let Some(adam_w) = run_config_values.adam_w {
        debug!("train_label_nodes: setting adam_w={adam_w:?}");
        optimizer = optimizer.beta1(adam_w.beta1).beta2(adam_w.beta2);
    }

    debug!("train_label_nodes: cuda_mem={:?}", cuda_mem());

    let loss = train(
        &device,
        batch,
        // &flat_label_node(&label_node_params).build_tch_module(&vs_0.root())?,
        &label_node(&label_node_params).build_tch_module(&vs_0.root())?,
        |pred: &Tensor, target: &Tensor| {
            debug!("train_on_label: pred.size={:?}", pred.size());
            let pred_y = pred.squeeze();
            //pred.i((.., 0, 0));
            debug!("train_on_label: pred_y={:?} target={:?} kind={:?}", pred_y.size(), target.size(), target.kind());
            pred_y.binary_cross_entropy::<Tensor>(&target, None, Reduction::Mean)
        },
        comon_label_node_params.cycles,
        // nn::AdamW::default().build(&vs_0, run_config.lr)?,
        optimizer.build(&vs_0, run_config_values.lr)?,

        |t: Tensor| {
            debug!("train_on_label: t.size={:?} cuda_mem={:?}", t.size(), cuda_mem());
            //merge max_motif_batches * motif_batch_size
            let t_out = view_for_label_model(t, channels as i64, label_node_params.max_motif_nodes, motif_batch_size, label_node_params.max_motif_batches as usize); //.view((t.size()[0], -1));
            //t.size()[0] - batch size
            debug!("train_on_label: t_out.size={:?} cuda_mem={:?}",  t_out.size(), cuda_mem());
            t_out
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
    mut label_models: HashMap<String, Vec<(usize, LabelNodeParams, SyncModule)>>,
    labels_channels_motifs_models:HashMap<String, Vec<Vec<(MotifNodeParams, SyncModule)>>>,

) -> Result<Box<dyn SavableOptResult>, ThoosaError> {
    let labels_len = labels_channels_motifs_models.len();

    let labels_channels_motifs_models_rw = labels_channels_motifs_models.into_iter().fold(HashMap::new(), |mut a, v|{
        a.insert(v.0, Arc::new(RwLock::new(v.1)));
        a
    });
    let input_slice_step = common_train_final_run_config.input_slice_step;
    let device = device.clone();

    let labels = store.lock().list_labels()?;

    let mut batch = ClassTensorsLoaderData::new(
        train_resources,
        valid_resources,

        common_train_final_run_config.batch_size,
        Arc::clone(&loader),
        device.clone(),
        Some( Box::new(move |t:Box<Tensor>| {
            apply_motif_and_label_nets(&t, &mut label_models, &labels_channels_motifs_models_rw, input_slice_step, &device, &labels)

        }))
    );

    opt_train(
        |vs:&VarStore, run_config_values:&TrainNodesValues, listener: &mut PlotVecf32|{
            train_final_node(
                ws, run_config_values, common_train_final_run_config, &mut batch, &device, labels_len, vs
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

pub fn apply_motif_and_label_nets(
    input_t: &Box<Tensor>,
    label_models: &mut HashMap<String, Vec<(usize, LabelNodeParams, SyncModule)>>,
    labels_channels_motifs_models_rw: &HashMap<String, Arc<RwLock<Vec<Vec<(MotifNodeParams, SyncModule)>>>>>,
    input_slice_step: usize,
    device: &Device,
    labels: &Vec<String>
) -> Box<Tensor> {
    let t_0 = Instant::now();
    let mut labels_tensors = vec![];
    for label in labels {
        let label_nodes = label_models.get_mut(label).expect(format!("no label model for {label}").as_str());
        debug!("apply_motif_and_label_nets: label={label} device={device:?} label_nodes.len={:?} label_nodes[0].1={:?} t.size()={:?}", label_nodes.len(), label_nodes[0].1, input_t.size());

        let channels_motifs_models = labels_channels_motifs_models_rw.get(label).expect(format!("no {label} label in labels_channels_motifs_models").as_str());
        debug!("apply_motif_and_label_nets: len={} channels_motifs_models={channels_motifs_models:?}", channels_motifs_models.read().len());

        let motif_nets_t = apply_motif_nets(&input_t, Arc::clone(channels_motifs_models), input_slice_step, label_nodes[0].1.motif_batch_size, label_nodes[0].1.max_motif_batches as usize, &device);
        let channels = label_nodes[0].1.channels;
        for (label_node_run_id, label_node_params, label_node) in label_nodes.iter_mut() {
            debug!("apply_motif_and_label_nets: label_node_params={label_node_params:?} label_node_run_id={label_node_run_id} t_net.size()={:?}", motif_nets_t.size());

            let t_net_view = view_for_label_model(motif_nets_t.unsqueeze(0), channels, label_node_params.max_motif_nodes, label_node_params.motif_batch_size, label_node_params.max_motif_batches as usize); //.view((t.size()[0], -1));

            debug!("apply_motif_and_label_nets: label={label} label_node_run_id={label_node_run_id:?} t_net_view.size()={:?} t_net_view.device()={:?}", t_net_view.size(), t_net_view.device());
            trace!("apply_motif_and_label_nets: label={label} t_net_view=\n{t_net_view}");
            label_node.set_device(*device);
            debug!("apply_motif_and_label_nets: label={label} label_node={label_node:?}");
            // label_node.debug();
            let t_out = label_node.forward(&t_net_view.to_device(*device));
            label_node.set_device(Cpu);
            debug!("apply_motif_and_label_nets: label={label} label_node.forward t_out.size={:?} t_out.device={:?}", t_out.size(), t_out.device());

            labels_tensors.push(t_out);
        }
    }
    let final_input_t = Tensor::stack(labels_tensors.as_slice(), 0);
    debug!("apply_motif_and_label_nets: final_input_t.size={:?} elapsed={}", final_input_t.size(), t_0.elapsed().as_millis());

    Box::new(final_input_t)
}

#[derive(Debug)]
struct TrainFinalNodesRunConfigResult {
    train_final_nodes_run_config_id: usize,
    loss: f32,
    var_store: VarStore,
    final_node_params: FinalNodeParams,
    train_final_nodes_values: Option<TrainNodesValues>
}

impl TrainFinalNodesRunConfigResult {
    pub fn new(train_final_nodes_run_config_id: usize, loss: f32, var_store: VarStore, final_node_params: FinalNodeParams, train_final_nodes_values: Option<TrainNodesValues>) -> Self {
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

    fn save(&mut self, store: &Arc<Mutex<Box<dyn Store>>>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<usize, ThoosaError> {
        let run_config_id = match self.train_final_nodes_values.as_ref(){
            None => self.train_final_nodes_run_config_id,

            Some(train_final_nodes_values) => {
                store.lock().save_run_config(
                    Some(&[self.train_final_nodes_run_config_id]),
                    &TrainLabelNodeRunConfig::type_name(),
                    "Final",
                    "todo",
                    TrainLabelNodeRunConfig::from_values(Some(self.train_final_nodes_run_config_id), train_final_nodes_values.lr, train_final_nodes_values.num_layers).to_properties_map()
                )?
            }
        };

        ws.read().save_final_model(run_config_id, &self.var_store, &self.final_node_params).expect("save_final_model failed");

        Ok(run_config_id)
    }
}

pub fn train_final_node(ws: &Arc<RwLock<Box<dyn Workspace>>>, run_config_values: &TrainNodesValues, common_train_final_run_config: &CommonTrainFinalNodeRunConfig, batch:&mut dyn DataBatch, device: &Device, labels_len: usize, vs_0: &VarStore) ->  Result<(f32, FinalNodeParams), ThoosaError>{
    let final_node_params= FinalNodeParams::new(labels_len as i64, labels_len as i64, run_config_values.num_layers);

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

            let pred_y = pred; //pred.i((.., 0, 0));
            debug!("train_final: pred_y.size={:?} target.size={:?}", pred_y.size(), target.size());
            trace!("train_final: pred_y=\n{pred_y} target=\n{target}");

            pred_y.cross_entropy_loss::<Tensor>(&target, None, Reduction::Mean, -100, 0.0)

        },
        common_train_final_run_config.cycles,
        optimizer.build(&vs_0, run_config_values.lr)?,
        |t:Tensor|{
            trace!("train_final: t=\n{t} t.squeeze()={}", t.squeeze());
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
pub struct SyncModule {
    module:Box<dyn ThoModule>,
    var_store:VarStore
}

unsafe impl Sync for SyncModule {}

impl SyncModule {
    pub fn new(module: Box<dyn ThoModule>) -> Self {
        Self {
            module,
            var_store: VarStore::new(Cpu)//dummy TODO remove
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
            .map(|t|tensor_size_in_bytes(t.1))
            .sum()
    }

    fn debug(&self){
        self.module.debug();
    }
}

impl Module for SyncModule {
    fn forward(&self, t: &Tensor) -> Tensor {
        self.module.forward(t)
    }
}

pub fn load_motif_models(ws:&Arc<RwLock<Box<dyn Workspace>>>, channel:usize, motif_id:usize, device: &Device) -> Result<Vec<(MotifNodeParams, SyncModule)>, ThoosaError> {
    debug!("load_motif_models_ws: channel={channel} motif_selection_run_id={motif_id}");

    let channel_motif_models_paths = ws.read().motif_models(&channel, motif_id)?;

    channel_motif_models_paths.into_iter().map(|(model, config)|{

        // let mut vs = VarStore::new(device.clone());
        let motif_node_params = MotifNodeParams::from_file(&config)?;

        debug!("load_motif_models_ws: model={model:?} config={config:?} motif_node_params={motif_node_params:?}");

        // let module = SyncModule::new(motif_node(&motif_node_params).build_tch_module(&vs.root())?);
        // vs.load(&model)?;
        // vs.freeze();

        let module = SyncModule::load(motif_node(&motif_node_params), model, device.clone())?;

        Ok((motif_node_params, module))

    }).collect()

}

pub fn load_label_models_ws(ws:&Arc<RwLock<Box<dyn Workspace>>>, train_label_nodes_run_ids: Option<&Vec<usize>>, device: &Device) -> Result<HashMap<String, Vec<(usize, LabelNodeParams, SyncModule)>>, ThoosaError> {
    let label_models_paths = ws.read().label_models()?;

    debug!("load_all_label_models_ws: label_models_paths={label_models_paths:?} train_label_nodes_run_ids={train_label_nodes_run_ids:?}");

    let train_label_nodes_run_ids = train_label_nodes_run_ids.map(|v|v.into_iter().collect::<HashSet<_>>());

    label_models_paths.into_iter().try_fold(HashMap::new(), |mut a, (label, paths)|{

        let models:Vec<(usize, LabelNodeParams, SyncModule)>  =  paths.into_iter().filter_map(|(model_path, params_path)|{
            let run_id = FsWorkspace::model_path_to_run_id(&model_path);
            if train_label_nodes_run_ids.as_ref().map(|set|set.contains(&run_id)).unwrap_or(true) {
                debug!("load_all_label_models_ws: skipping run_id={run_id}");
                return None;
            }

            let mut vs = nn::VarStore::new(device.clone());
            let label_node_params = LabelNodeParams::from_file(&params_path).expect(format!("no {params_path:?}").as_str());

            debug!("load_all_label_models_ws: model={model_path:?} run_id={run_id} label_node_params={label_node_params:?}");

            let node = SyncModule::load(label_node(&label_node_params), model_path.clone(), *device).expect(format!("can't load {model_path:?}").as_str());
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

pub fn load_final_model(run_id:usize, ws:&Arc<RwLock<Box<dyn Workspace>>>, device: &Device)->Result<(usize, FinalNodeParams, SyncModule), ThoosaError> {
    let (model_path, params_path) = ws.read().final_model(run_id)?;
    debug!("load_final_model: params_path={model_path:?} run_id={run_id}");

    let mut vs = nn::VarStore::new(device.clone());
    let final_node_params = FinalNodeParams::from_file(&params_path)?;

    debug!("load_final_model:  final_node_params={final_node_params:?}");

    let node = SyncModule::load(final_node(&final_node_params), model_path, *device)?;
    // let label_node_params = FinalNodeParams::from_file(&config)?;

    Ok((run_id,  final_node_params, node))
}


#[derive(Debug, Args, Clone)]
pub struct TrainMotifsNodesParams {

    #[arg(long)]
    pub motif_selection_run_id: Vec<usize>,

    #[arg(long, default_value_t = 1000)]
    pub cycles: usize,

    #[arg(long, default_value_t = 10)]
    pub batch_size: usize,//number of the motif slices in a batch

    #[arg(long, value_parser=parse_range_or_value::<f64>)]
    pub lr: RangeOrValue<f64>,

    #[arg(long, value_parser=parse_range_or_value::<u32>)]
    pub num_layers: RangeOrValue<u32>,

    #[arg(long, default_value_t = 0.2)]
    pub max_loss: f32,

    #[arg(long, default_value_t = 10)]
    pub opt_max_evals: u32,

    #[arg(long, default_value_t = 3)] //TODO validate opt_n_init < opt_max_evals
    pub opt_n_init: u32,

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
    OptSpace{opt_space:ConfigurationSpace,opt_config:OptConfig}
}

#[derive(Clone, Debug)]
pub struct TrainMotifsNodesRunConfig {
    id:Option<usize>,

    run_config_or_opt_space: ValuesOrOptSpace<TrainNodesValues>

}

// unsafe impl Sync for TrainMotifsNodesRunConfig{}
pub fn adam_w_hyperparams() -> Vec<Hyperparameter>{
    vec![
        Hyperparameter::UniformFloat {name: "beta1".to_string(), lower:0.5, upper: 0.99},
        Hyperparameter::UniformFloat {name: "beta2".to_string(), lower:0.9, upper: 0.9999}
    ]
}

pub fn basic_hyperparams() -> Vec<Hyperparameter>{
    vec![
        Hyperparameter::UniformFloat {name: "lr".to_string(), lower:1e-6, upper: 1e-2},
        Hyperparameter::UniformInt {name: "num_layers".to_string(), lower:0, upper: 6}
    ]
}

#[derive(Debug, Clone)]
pub struct TrainNodesValues {
    lr: f64, num_layers: u32,
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

impl TryFrom<&Vec<Primitives>> for TrainNodesValues {
    type Error =ThoosaError;

    fn try_from(value: &Vec<Primitives>) -> Result<Self, Self::Error> {
        let num_layers:f32 = value[1].try_into()?;//TODO opt can't produce ints
        Ok(Self {
            lr: value[0].try_into()?,
            num_layers: num_layers as u32,
            adam_w: if value.len() > 2 {Some(AdamW::default()
                .beta1(value[2].into())
                .beta2(value[3].into())
            )} else {
                None
            },
        })
    }
}

#[derive(Debug)]
pub struct CommonTrainMotifsNodesRunConfig {
    pub batch_size: usize,
    pub cycles: usize,
    pub max_loss:f32,
}

impl TrainMotifsNodesRunConfig {
    pub fn from_values(id: Option<usize>, lr: f64, num_layers: u32) -> Self {
        Self {
            id,
            run_config_or_opt_space: ValuesOrOptSpace::Values(TrainNodesValues {lr, num_layers, adam_w: None}),
        }
    }

    pub fn from_opt_space(id: Option<usize>, opt_space:ConfigurationSpace, opt_config: OptConfig) -> Self {
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

    fn from_properties_map(run_id: usize, properties: HashMap<String, Val>) -> Self {
        let properties = unprefix_properties("TrainMotifsNodesRunConfig", &properties);

        if properties.contains_key("config_space") {
            todo!()
        } else {
            TrainMotifsNodesRunConfig {
                id:Some(run_id),

                run_config_or_opt_space: ValuesOrOptSpace::Values(TrainNodesValues {
                    lr: properties.get("lr").expect("no lr").as_f32().expect("lr as_f32") as f64,
                    num_layers: properties.get("num_layers").expect("no num_layers").as_usize().expect("num_layers as usize") as u32,
                    adam_w: None,
                }),
            }
        }
    }

    fn set_id(&mut self, id: usize) {
        self.id.replace(id);
    }

    fn type_name() -> &'static str {
        "MotifNode"
    }
}

impl Params<TrainMotifsNodesRunConfig, Empty, CommonTrainMotifsNodesRunConfig> for TrainMotifsNodesParams {

    fn to_run_configs(&self) -> Vec<TrainMotifsNodesRunConfig> {

        if let Some(config_space) = self.to_config_space() {

            debug!("TrainMotifsNodesParams::to_run_configs: config_space={config_space:?}");
            vec![
                TrainMotifsNodesRunConfig::from_opt_space(None, config_space, OptConfig::new(self.opt_max_evals, self.opt_n_init))
            ]

        } else {
            //TODO avoid collecting
            let v = make_combos(vec![
                self.lr.clone().into_iter().map(|v| v.into()).collect::<Vec<Val>>().into_iter(),
                self.num_layers.clone().into_iter().map(|v| v.into()).collect::<Vec<Val>>().into_iter(),
            ]).into_iter().map(|mut config_combo:Vec<Val>|{
                debug!("TrainMotifsNodesParams::to_run_configs: config_combo={config_combo:?} ");
                let num_layers = config_combo.pop().unwrap().try_into().unwrap();
                let lr = config_combo.pop().unwrap().try_into().unwrap();
                TrainMotifsNodesRunConfig::from_values(
                    None,
                    lr,
                    num_layers,
                )
            }).collect();

            debug!("TrainMotifsNodesParams::to_run_configs: self.lr={:?} self.num_layers={:?} v={v:?}", self.lr, self.num_layers);

            v
        }
    }

    fn resource_config(&self) -> Empty {
        Empty{}
    }

    fn common_config(&self) -> CommonTrainMotifsNodesRunConfig {
        CommonTrainMotifsNodesRunConfig {
            batch_size: self.batch_size,
            cycles: self.cycles,
            max_loss: self.max_loss
        }
    }
}

impl TrainMotifsNodesParams {

    pub fn dummy(motif_selection_run_id:Vec<usize>, cycles: usize, max_loss: f32, batch_size:usize, lr: impl Into<RangeOrValue<f64>>, num_layers:impl Into<RangeOrValue<u32>>, opt_max_evals: u32, opt_n_init: u32) ->Self{

        let lr = lr.into();

        if let RangeOrValue::Range{range, step} = &lr{
            assert!(range.end > range.start);
        }

        TrainMotifsNodesParams {
            motif_selection_run_id,

            cycles,

            lr,

            num_layers: num_layers.into(),

            max_loss,

            batch_size,

            opt_max_evals,

            opt_n_init,
        }
    }

    //TODO add explicit flag to enable optimization
    pub fn to_config_space(&self) -> Option<ConfigurationSpace> {
        [
            (&self.lr).try_into().ok(),
            (&self.num_layers).try_into().ok(),
        ].into_iter()
            .collect::<Option<Vec<Hyperparameter>>>()
            .map(|v|ConfigurationSpace::new(v))
    }
}
#[derive(Debug, Serialize, Deserialize, Clone)]
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
            properties.get("num_features").expect(format !("no num_features in {:?}", properties).as_str()).as_i64().expect("num_features as_i64"), 1,
        )
    }

    pub fn from_file(path:&PathBuf) -> Result<Self, ThoosaError> {
        serde_json::from_str::<MotifNodeParams>(
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
    let mut builder = SeqBuilder::new(params.num_features);
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelNodeParams {
    channels:i64,
    max_motif_nodes:i64,
    max_motif_batches:i64,
    motif_batch_size:i64
}

impl LabelNodeParams {
    pub fn new(channels: i64, max_motif_nodes: i64, max_motif_batches: i64, motif_batch_size: i64) -> Self {
        Self { channels, max_motif_nodes, max_motif_batches, motif_batch_size }
    }

    pub fn to_properties_map(&self) -> HashMap<String, Val> {
        let map = HashMap::from([
            ("LabelNodeParams.channels".into(),  self.channels.into()),
            ("LabelNodeParams.max_motif_nets".into(),  self.max_motif_nodes.into()),
            ("LabelNodeParams.max_motif_batches".into(),  self.max_motif_batches.into()),
            ("LabelNodeParams.motif_batch_size".into(),  self.motif_batch_size.into()),
        ]);

        map

    }

    pub fn from_properties_map(properties:HashMap<String, Val>) -> Self {
        let properties = unprefix_properties("LabelNodeParams", &properties);
        LabelNodeParams::new(
            properties.get("channels").expect("channels").as_i64().expect("channels as_i64"),
            properties.get("max_motif_nets").expect("max_motif_nets").as_i64().expect("max_motif_nets as_i64"),
            properties.get("max_motif_batches").expect("max_motif_batches").as_i64().expect("max_motif_batches as_i64"),
            properties.get("motif_batch_size").expect("motif_batch_size").as_i64().expect("motif_batch_size as_i64"),
        )
    }

    pub fn from_file(path:&PathBuf) -> Result<Self, ThoosaError> {
        serde_json::from_str::<LabelNodeParams>(
            fs::read_to_string(&path)?.as_str()
        ).map_err(|serde_err|ThoosaError::new(serde_err.to_string()))
    }
}

pub fn flat_label_node(params: &LabelNodeParams) -> Node {
    debug!("motif_node: params={params:?}");
    let num_features = params.channels * params.max_motif_nodes * params.max_motif_batches * params.motif_batch_size;
    let mut builder = SeqBuilder::new(params.max_motif_batches * params.motif_batch_size);
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
    /*
    let features = max_motif_batches as i64 * motif_batch_size;
    let dims = [max_motif_batches as i64 * motif_batch_size, max_nets, channels, 1];

    let mut builder = SeqBuilder::new(features);

    build_multidim(&mut builder, &dims);
    builder.end_seq()
    */

    /*
        SeqBuilder::new(features)
            .func(Tensor::relu)
            .linear_s(features)
            .func(Tensor::relu)
            .linear_s(1)
            .func(Tensor::sigmoid)
            .end_seq()
    */
    let max_motif_nets=  params.max_motif_nodes;
    SeqBuilder::new(params.max_motif_batches * params.motif_batch_size)
        // .batch_norm()
        .closure(Box::new(move |t:&Tensor|{
            //batch x max_motif_nets
            debug!("label_node:  t.size()={:?}", t.size());

            let t_0 = t.shallow_clone();

            t_0

        }), None)
        .split(
            (0..params.max_motif_nodes).map( |_i| {
                let num_features = params.max_motif_batches * params.motif_batch_size;
                let mut builder = SeqBuilder::new(num_features);
                let mut net = builder
                    .closure(Box::new(move |t: &Tensor| {
                        debug!("label_node in split in t.size={:?} device={:?} t.is_nonzero()={}", t.size(), t.device(), is_nonzero(&t).unwrap());
                        // let t = t.view((-1, channels, max_motif_batches as i64 * motif_batch_size));
                        // debug!(">>>>>>>>>>>>>> 0 {:?}", t.size());
                        t.shallow_clone()
                    }), None);

                for i in 0..3 as usize {//params.num_layers
                    net = net.linear_s(num_features * 2)
                        .func(Tensor::relu);
                }

                net
                    // .conv_1d(10, params.channels, params.channels, ConvConfig::default())
                    // .max_pool(10, 1, 0)
                    // .func(Tensor::relu)
                    // .linear_s(channels)
                    // .func(Tensor::sigmoid)
                    // .closure(Box::new(|t:&Tensor|{
                    //     debug!(">>>>>>>>>>>>>> 1 {:?}", t.size());
                    //     t.squeeze()
                    // }), Some(channels))
                    // .func(Tensor::relu)
                    // .linear_s(100)
                    // .func(Tensor::sigmoid)
                    // .closure(Box::new(|t:&Tensor|{
                    //     debug!(">>>>>>>>>>>>>> 2 {:?}", t.size());
                    //     t.squeeze()
                    // }), Some(channels))

                    .func(Tensor::relu)
                    // .linear_f(0.5)
                    // .linear_f(0.5)
                    .linear_s(1)
                    .func(Tensor::sigmoid)
                    .closure(
                        Box::new(|t: &Tensor| {
                            debug!("label_node in split out t.size={:?}", t.size());
                            t.shallow_clone()
                        }),
                        Some(1)
                    ).end_seq()

            }).collect(),

            Box::new(
                move |t:&Tensor|  (0..max_motif_nets)
                // .zip((0..channels))
                // .map(|(net_i, channel_id)|t.i((.., channelmehrzweckzeitreihenanalysegerät_id, net_i, ..)))
                .map(|net_i|{
                    debug!("label_node split merged in t.size={:?} net_i={:?}", t.size(), net_i);
                    let x = t.i((.., .., net_i, ..));

                    debug!("label_node split merged out 4 x.size={:?}", x.size());

                    x
                })
                // .filter(|t|{
                //     let nonzero = is_nonzero(&t).unwrap();
                //     if !nonzero {
                //         info!("zero filter");
                //     }
                //     nonzero
                // })
                .collect()
            ),//label batch, channel, max_motif_nets, max motif_net batches x motif_net batch size

            Box::new(move |t:Vec<Tensor>|Tensor::stack(t.as_slice(), 1).mean_dim(1, false, Float))

        )
        .closure(Box::new(move |t:&Tensor|{
            //batch x max_motif_nets
            debug!("label_node merged final in flatten t.size()={:?}", t.size());
            // let t_size = t.size();

            // assert!(t_size.len()==4, "t_size.len() {}, not 4", t_size.len());
            // assert!(t_size[1] == max_motif_nets, "unexpected t_size[1]={}, should be max_motif_nets={}", t_size[1], max_motif_nets);
            // assert!(t_size[2] == 1, "unexpected t_size[2]={}, should be 1", t_size[2]); //.conv_1d - channels -> 1
            // assert!(t_size[3] == 1, "unexpected t_size[3]={}, should be 1", t_size[2]);
            // let t_0 = t.squeeze_dims(&[2,3]);
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

#[derive(Debug, Serialize, Deserialize)]
pub struct FinalNodeParams {
    features: i64,
    classes_num: i64,
    num_layers: u32
}

impl FinalNodeParams {
    pub fn new(features: i64, classes_num: i64, num_layers: u32) -> Self {
        Self { features, classes_num, num_layers }
    }

    pub fn to_properties_map(&self) -> HashMap<String, Val> {
        HashMap::from([
            ("FinalNodeParams.kernel".into(), self.features.into()),
            ("FinalNodeParams.stride".into(), self.classes_num.into()),
            ("FinalNodeParams.num_layers".into(), self.num_layers.into()),
        ])
    }

    pub fn from_file(path:&PathBuf) -> Result<Self, ThoosaError> {
        serde_json::from_str::<FinalNodeParams>(
            fs::read_to_string(&path)?.as_str()
        ).map_err(|serde_err|ThoosaError::new(serde_err.to_string()))
    }


}

pub fn final_node(params:&FinalNodeParams) -> Node {
    debug!("final_node: params={:?}", params);

    let mut builder = SeqBuilder::new(params.features);
    let mut net = builder
        .linear_s(2*params.features)
        .func(Tensor::relu);

    for i in 0..params.num_layers {
        net = net.linear_s(params.features)
             .func(Tensor::relu)
    }

    net
        // .func(Tensor::relu)
        .linear_s(params.classes_num)
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
    train_motifs_nodes_run_config_id:usize
}

impl TrainMotifsNodesRunConfigResult {
    pub fn new(loss:f32, channel:u32, motif_id:usize, vs: VarStore, train_motifs_nodes_run_config_id:usize, params: MotifNodeParams, run_config_values: Option<TrainNodesValues>) -> Self {
        Self { loss, channel, motif_id, vs, params, run_config_values,train_motifs_nodes_run_config_id }
    }
}

impl SavableOptResult for TrainMotifsNodesRunConfigResult {

    fn value(&self) -> f32 {
        self.loss
    }

    fn save(&mut self, store:&Arc<Mutex<Box<dyn Store>>>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<usize, ThoosaError> {
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


pub fn opt_train<F, V, N:Debug>(mut train: F, run_config: &ValuesOrOptSpace<V>, device: Device, descr: &str) -> Result<(VarStore, N, Option<V>, f32), ThoosaError>//Option<V> - Some if run config from opt
where
    F: FnMut(&VarStore, &V, &mut PlotVecf32) -> Result<(f32, N), ThoosaError>,
    for<'a> <V as TryFrom<&'a Vec<Primitives>>>::Error: Debug,
    V: Clone + Display + Debug + for<'a> TryFrom<&'a Vec<Primitives>>,
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

            let mut opt = crate::parzen::optimizer::TPEOptimizer::new(
                |params| {
                    let run_config_values: V = params.try_into().expect("can't convert params into TrainNodesValues"); //TODO handle

                    let vs = VarStore::new(device);

                    let mut listener = PlotVecf32::new(format!("_params_{}", run_config_values).as_str()); //motif_id ?
                    let (loss, node_params) = train(&vs, &run_config_values, &mut listener).expect("tain failed");
                    listener.plot();

                    debug!("opt_train: descr={descr} params={params:?} loss={loss}");
                    if loss < min_loss {
                        min_loss = loss;
                        min_var_store.replace((vs, node_params, Some(run_config_values), min_loss));
                    }
                    loss
                },
                opt_space.clone(),
                opt_config.max_evals as usize, //common_config.opt_max_evals as usize,
                opt_config.n_init as usize, //common_config.opt_n_init as usize,
            );

            let best_config = opt.optimize();
            trace!("opt_train: done opt, descr={descr} min_var_store={min_var_store:?}");
            info!("opt_train: done opt, descr={descr} best_config={best_config:?} min_loss={min_loss}");

            min_var_store.ok_or_else(||ThoosaError::new("opt was not run"))
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

    fn train_batch(&mut self) -> (Tensor, Tensor) {
        Self::choose_for_batch(&self.train, self.batch_size, &self.device)
    }

    fn valid_batch(&mut self) -> (Tensor, Tensor) {
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
    input_t: &Tensor,
    channels_motifs_nodes: Arc<RwLock<Vec<Vec<(MotifNodeParams, SyncModule)>>>>,
    input_slice_step: usize,
    input_slices_motif_batch_size: i64,
    max_motif_batches: usize,
    device: &Device
) -> Tensor {
    let max_motif_nodes = channels_motifs_nodes.read().iter().map(|nets| nets.len()).max().unwrap_or(0);

    debug!("apply_motif_nets: input_t.size={:?} max_motif_nodes={max_motif_nodes} input_slices_motif_batch_size={input_slices_motif_batch_size} max_motif_batches={max_motif_batches}", input_t.size());

    let channels_motifs = Tensor::stack(
        channels_motifs_nodes.write()
            .iter_mut()
            .enumerate()
            .map(|(channel, motif_nodes)| apply_channel_motifs(
                input_slice_step,
                input_slices_motif_batch_size,
                motif_nodes,
                input_t.i((channel as i64, ..)),
                max_motif_nodes as i64,
                max_motif_batches,
                device
            ))
            .collect::<Vec<_>>().as_slice(),
        0
    );

    // let padded = pad_to(&channels_motifs, &[5, max_motif_nodes as i64, max_motif_batches as i64, 7]).unwrap();
    // debug!("apply_label_motifs: channels_motifs.size={:?} padded.size()={:?}", channels_motifs.size(), padded.size());
    // padded

    debug!("apply_motif_nets: channels_motifs.size={:?}", channels_motifs.size());

    channels_motifs
}

#[derive(Debug, Args, Clone)]
pub struct TrainFinalNodeParams {
    #[arg(long)]
    pub train_labels_run_ids: Vec<usize>,

    #[arg(long,  default_value_t = 1000)]
    input_slice_step: usize,

    #[arg(long,  default_value_t = 100)]
    motif_batch_size: i64,

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

impl TrainFinalNodeParams {

    pub fn dummy(train_labels_run_ids: Vec<usize>, input_slice_step: usize, batch_size: usize, train_resource_group:String, valid_resource_group:String, cycles: usize, motif_batch_size: i64, lr: impl Into<RangeOrValue<f64>>, num_layers:impl Into<RangeOrValue<u32>>, opt_max_evals: u32, opt_n_init: u32) ->Self {
        TrainFinalNodeParams {
            train_labels_run_ids,
            input_slice_step,
            cycles,
            batch_size,
            lr:lr.into(),
            num_layers:num_layers.into(),
            train_resource_group,
            valid_resource_group,
            motif_batch_size,
            opt_max_evals,
            opt_n_init,
        }
    }

    //TODO add explicit flag to enable optimization
    pub fn to_config_space(&self) -> Option<ConfigurationSpace> {
        [
            vec![
                (&self.lr).try_into().ok(),
                (&self.num_layers).try_into().ok(),
            ],
            adam_w_hyperparams().into_iter().map(|v|Some(v)).collect::<Vec<Option<Hyperparameter>>>(),
        ].concat().into_iter()
            .collect::<Option<Vec<Hyperparameter>>>()
            .map(|v|ConfigurationSpace::new(v))
    }


    pub fn train_resource_group(&self) -> &str {
        &self.train_resource_group
    }

    pub fn valid_resource_group(&self) -> &str {
        &self.valid_resource_group
    }

    pub fn step(&self) -> usize {
        self.input_slice_step
    }

    pub fn motif_batch_size(&self) -> i64 {
        self.motif_batch_size
    }
}

#[derive(Debug)]
pub struct TrainFinalNodeRunConfig {
    id:Option<usize>,
    run_config_or_opt_space: ValuesOrOptSpace<TrainNodesValues>
}

impl TrainFinalNodeRunConfig {

    pub fn from_values(id: Option<usize>, lr: f64, num_layers: u32) -> Self {
        Self {
            id,
            run_config_or_opt_space: ValuesOrOptSpace::Values(TrainNodesValues {lr, num_layers, adam_w: None}),
        }
    }

    pub fn from_opt_space(id: Option<usize>, opt_space:ConfigurationSpace, opt_config: OptConfig) -> Self {
        Self {
            id,
            run_config_or_opt_space: ValuesOrOptSpace::OptSpace {opt_space, opt_config},
        }
    }
}

impl RunConfig for crate::train_nets::TrainFinalNodeRunConfig {
    fn from_properties_map(run_id: usize, properties: HashMap<String, Val>) -> Self {
        todo!()
    }

    fn set_id(&mut self, id: usize) {
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
    pub input_slice_step: usize
}

impl CommonTrainFinalNodeRunConfig {
    pub fn new(cycles: usize, batch_size:usize, input_slice_step: usize) -> Self {
        Self { cycles, batch_size,input_slice_step }
    }
}

impl Params<TrainFinalNodeRunConfig, NetResources<Logits>, CommonTrainFinalNodeRunConfig> for TrainFinalNodeParams {

    fn to_run_configs(&self) -> Vec<TrainFinalNodeRunConfig> {

        if let Some(config_space) = self.to_config_space() {

            debug!("TrainFinalNodeParams::to_run_configs: config_space={config_space:?}");
            vec![
                TrainFinalNodeRunConfig::from_opt_space(None, config_space, OptConfig::new(self.opt_max_evals, self.opt_n_init))
            ]

        } else {
            //TODO avoid collecting
            let v = make_combos(vec![
                self.lr.clone().into_iter().map(|v| v.into()).collect::<Vec<Val>>().into_iter(),
                self.num_layers.clone().into_iter().map(|v| v.into()).collect::<Vec<Val>>().into_iter(),
            ]).into_iter().map(|mut config_combo:Vec<Val>|{
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
            self.batch_size,
            self.input_slice_step
        )
    }
}


#[derive(Debug, Args, Clone)]
pub struct InferParams {
    #[arg(long)]
    pub final_run_id:usize,

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

    pub fn new(final_run_id: usize, input_data: String, output: String, input_slice_step: usize) -> Self {
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



#[cfg(test)]
mod train_tests {
    use parking_lot::{Mutex, RwLock};
    use std::collections::HashMap;
    use std::fs;
    use std::ops::Deref;
    use std::path::{Path, PathBuf};
    use std::rc::Rc;
    use std::sync::Arc;
    use cudarc::driver::CudaDevice;
    use log::{debug, info, set_max_level, trace};
    use log::LevelFilter::{Info, Trace};
    use rand::{thread_rng, Rng};
    use rayon::iter::IntoParallelRefMutIterator;
    use tch::nn::{init, AdamW, LinearConfig, Module, VarStore};
    use tch::Device::{Cpu, Cuda};
    use tch::{IndexOp, Kind, Reduction};
    use tch::Kind::Float;
    use tch::{nn, Device, Tensor};
    use tch::kind::FLOAT_CPU;
    use crate::builder::SeqBuilder;
    use crate::loader::{LoadRunner, SyncWrapper};
    use crate::my_tch::{apply_channel_motifs, ThoModule};
    use crate::train_nets;
    use crate::train_nets::{apply_motif_and_label_nets, apply_motif_nets, label_node, ClassTensorsLoaderData, CommonTrainFinalNodeRunConfig, FinalNodeParams, LoadingTensorsDataBatch, TrainNodesValues};
    use crate::train_nets::{load_label_models_ws, load_motif_models, CommonTrainLabelNodeRunConfig, InferParams, LabelNodeParams, MotifNodeParams, SyncModule, TensorsDataBatch, TrainFinalNodeParams, TrainLabelNodeParams, TrainLabelNodeRunConfig};
    use crate::utils::{lin_t, pad_to, reset_print_options, run_split_on_oom, t_0_0, tensor_size_in_bytes, DataBatch, Reducer, SplitQueue, Splittable, SplittableTensor};
    use crate::utils::from;
    use crate::ws::{DummyWorkspace, FsWorkspace, Workspace};
    use thoosa_shared::resource::Resource;
    use thoosa_shared::{batch_windows, cdt_for_dir, clc_num_batches, ThoosaError};
    use thoosa_store::SqlStore;
    use crate::simp::DummyDs;

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

        SeqBuilder::new(num_features)
            .linear_s_with_conf(1, LinearConfig {
                ws_init: init::Init::Const(k),//conflicts with ctor's init
                bs_init: None,
                bias: false
            })
            .end_seq().build_tch_module(&vs.root())
    }

    #[test]
    fn test_test_net() -> Result<(), ThoosaError>{
        let t = lin_test_tensor(100).i((0, ..));

        //sums and multiplies by k
        assert_eq!(Tensor::from_slice(&[4950.]), test_net(100, 1., Cpu)?.forward(&t));
        assert_eq!(Tensor::from_slice(&[9900.]), test_net(100, 2., Cpu)?.forward(&t));

        Ok(())
    }


    #[test]
    fn test_apply_motif_models() -> Result<(), ThoosaError>{
        reset_print_options();
        let input_n = 1000;
        let motif_width = 100;
        let step = (motif_width / 1) as usize;//1 - step==motif_width slices do not overlap so net output - sum(slice)*k

        let t = lin_test_tensor(input_n);

        let max_dists_len = (input_n-motif_width)/step as i64 + 1;

        let mut motif_nets = vec![
            vec![
                (MotifNodeParams::new(motif_width, 1), SyncModule::new(test_net(motif_width, 1., Cpu)?)),
                (MotifNodeParams::new(motif_width, 1), SyncModule::new(test_net(motif_width, 2., Cpu)?))
            ], //channel 0 nets
            vec![
                (MotifNodeParams::new(motif_width, 1), SyncModule::new(test_net(motif_width, 10., Cpu)?))
            ] //channel 1 nets
        ];

        let max_motif_nets_in_channel = motif_nets.iter().map(|nets|nets.len()).max().unwrap_or(0);
        assert_eq!(2, max_motif_nets_in_channel);

        let batch_size = 10;
        let motif_batches = batch_windows(input_n as usize, motif_width as usize, step, batch_size);
        debug!("motif_batches={motif_batches:?}");
        let max_motif_batches = motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0};

        // let channel_0_t = apply_channel_motifs(step, batch_size as i64, &mut motif_nets[0], t.i((0, ..)), max_motif_nets_in_channel as i64, max_motif_batches, &Cpu);
        let channel_0_t = apply_channel_motifs_2(step,  &mut motif_nets[0], t.i((0, ..)), motif_width, max_motif_nets_in_channel as i64, max_dists_len, &Cpu)?;

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
        let channel_1_t = apply_channel_motifs_2(step,  &mut motif_nets[1], t.i((1, ..)), motif_width, max_motif_nets_in_channel as i64, max_dists_len, &Cpu)?;

        debug!("test_apply_motif_models: max_motif_batches={:?}", max_motif_batches);

        let channel_1_motifs = Tensor::stack(&[
                Tensor::from_slice(&[99000., 299000., 499000., 699000., 899000., 1.099e+06, 1.299e+06, 1.499e+06, 1.699e+06, 1.899e+06]),  //ch 1 net 0 (ResourceId=1)
                Tensor::from_slice(&[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),  //ch 1 net 0 (ResourceId=1)
            ], 0
        ).totype(Float).view((2, 10));
        assert_eq!(channel_1_motifs, channel_1_t);

        debug!("test_apply_motif_models: channel_0_motifs.size()={:?}, channel_1_motifs.size()={:?}", channel_0_motifs.size(), channel_1_motifs.size());

        let channels_motifs = apply_motif_nets(&t, Arc::new(RwLock::new(motif_nets)), step, batch_size as i64, max_motif_batches, &Cpu).squeeze();

        assert_eq!(
            Tensor::stack(&[channel_0_motifs, channel_1_motifs], 0),
            channels_motifs
        );

        Ok(())
    }

    fn apply_channel_motifs_2(input_slice_step: usize,
                              motif_nodes: &mut [(MotifNodeParams, SyncModule)],
                              channel: Tensor,
                              motif_width: i64,
                              max_motif_nodes: i64,
                              max_dists_len: i64,
                              device: &Device
    ) -> Result<Tensor, ThoosaError> {

        let unfolded = channel.unfold(0, motif_width, input_slice_step as i64);

        let motif_dists = motif_nodes.iter_mut().map(|(_params, net)| {
            net.set_device(device.clone());

            let mut motif_distance_splits = vec![];
            run_split_on_oom(
                SplitQueue::<2, _>::new(SplittableTensor::<0>(unfolded.shallow_clone())),
                |t|{

                    let r = net.forward(t.as_ref());
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

        pad_to(&Tensor::stack(motif_dists.as_slice(), 0).view((motif_nodes.len() as i64, -1)), &[max_motif_nodes, max_dists_len])
    }

    #[test]
    fn test_apply_motif_models_2() -> Result<(), ThoosaError> {
        unsafe {
            let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        set_max_level(Trace);
        let input_n = 2_000_000_000;//TODO calculated from free mem
        // let input_n = 10000;//TODO calculated from free mem
        let motif_width = 20;
        let input_slice_step = 10;
        let input_slices_motif_batch_size = 5;
        let max_dists_len = (input_n-motif_width)/input_slice_step as i64 + 1;

        let channel = lin_t(input_n).to_device(Cuda(0));
        let unfolded = channel.unfold(0, motif_width, input_slice_step as i64);
        let sums = unfolded.sum_dim_intlist(1, true, Float).view(max_dists_len);//sums of the slices should be what test_net produces
        debug!("tensor_size_in_bytes={:?}M ", tensor_size_in_bytes(&unfolded)/1024/1024);
        trace!("unfolded=\n{unfolded} \nsums=\n{}", sums);

        let mut motif_nets = vec![
            (MotifNodeParams::new(motif_width, 1), SyncModule::new(test_net(motif_width, 1., Cuda(0))?))
        ];

        let channel_1_t = apply_channel_motifs_2(input_slice_step,  &mut motif_nets, channel, motif_width, 1, max_dists_len, &Cuda(0))?.view(max_dists_len);

        trace!("channel_1_t=\n{channel_1_t}");

        assert!(sums.allclose(&channel_1_t, 0.1, 0.1, true));


        Ok(())
    }

    #[test]
    fn test_apply_motif_models_diff_len() -> Result<(), ThoosaError>{
        let input_n_0 = 1000;
        let motif_width= 120;
        let t_0 = lin_test_tensor(input_n_0);
        let t_1 = lin_test_tensor(831);

        let motif_batch_size = 13;

        let motif_nets = Arc::new(RwLock::new(vec![
            vec![(MotifNodeParams::new(motif_width, 1), SyncModule::new(test_net(motif_width, 1., Cpu)?)), (MotifNodeParams::new(motif_width, 1), SyncModule::new(test_net(motif_width, 1., Cpu)?))], //channel 0 nets
            vec![(MotifNodeParams::new(motif_width, 1), SyncModule::new(test_net(motif_width, 10., Cpu)?))], //channel 1 nets
            vec![]
        ]));

        let step = 11;
        let batch_size = 10;
        let motif_batches = batch_windows(input_n_0 as usize, motif_width as usize, step, batch_size);
        let max_motif_batches = motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0};

        debug!("max_motif_batches={}", max_motif_batches);

        let channels_motifs_0 = apply_motif_nets(&t_0, Arc::clone(&motif_nets), step, motif_batch_size, max_motif_batches, &Cpu);

        debug!("channels_motifs_0.size={:?}", channels_motifs_0.size());

        let channels_motifs_1 = apply_motif_nets(&t_1, Arc::clone(&motif_nets), step, motif_batch_size, max_motif_batches, &Cpu);

        debug!("channels_motifs_1.size={:?}", channels_motifs_1.size());

        assert_eq!(vec![3, 2, 9, 13], channels_motifs_0.size());//channel, motif nets, batches, batch size
        assert_eq!(vec![3, 2, 9, 13], channels_motifs_1.size());
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
    fn test_label_node() -> Result<(), ThoosaError> {
        unsafe {
            let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        let mut vs = nn::VarStore::new(Device::Cpu);

        let node = label_node(&LabelNodeParams::new(20, 3, 69, 10)).build_tch_module(&vs.root())?;
        //
        vs.load("/tmp/ws_test/models/labels/other/430.safetensors")?;
        vs.freeze();
        node.debug();
        debug!("---------------");

        vs.set_device(Cuda(0));
        node.debug();
        Ok(())
    }

    fn test_train_label() -> Result<(), ThoosaError> {
        set_max_level(Info);
        let ws = Arc::new(RwLock::new(DummyWorkspace::new(2, 1000)?));

        const BATCH_SIZE:usize = 10;
        const CHANNELS:usize = 2;
        const MAX_MOTIF_NETS:usize = 5;
        const MOTIF_BATCHES:usize = 3;
        const MOTIF_BATCH_SIZE:usize = 10;
        let channel_motif_nets = [3,2,1];//TODO

        // let mut rnd = thread_rng();

        const TRAIN_LABELS: [bool; BATCH_SIZE] = build_labels::<BATCH_SIZE>();
        const VALID_LABELS: [bool; BATCH_SIZE] = build_labels::<BATCH_SIZE>();

        const MAX_MOTIF_BATCHES:usize = clc_num_batches(1000, 100, 100, BATCH_SIZE);

        struct DummyLabelBatch {}

        impl DummyLabelBatch {
            pub fn new() -> Self {
                Self {}
            }
        }

        impl DataBatch for DummyLabelBatch {

            fn train_batch(&mut self) -> (Tensor, Tensor) {
                (
                    dummy_labels_net_input(&TRAIN_LABELS, CHANNELS, MAX_MOTIF_NETS, MAX_MOTIF_BATCHES, MOTIF_BATCH_SIZE),
                    Tensor::from_slice(TRAIN_LABELS.as_slice()).totype(Float)
                )
            }

            fn valid_batch(&mut self) -> (Tensor, Tensor) {
                (
                    dummy_labels_net_input(&TRAIN_LABELS,  CHANNELS, MAX_MOTIF_NETS, MAX_MOTIF_BATCHES, MOTIF_BATCH_SIZE),
                    Tensor::from_slice(VALID_LABELS.as_slice()).totype(Float)
                )
            }
        }

        let (loss, params) = super::train_label_nodes(
            &ws,
            &TrainNodesValues::new(1e-3, 3, Some(AdamW::default().amsgrad(true))),
            &CommonTrainLabelNodeRunConfig::new(100,  MOTIF_BATCH_SIZE, BATCH_SIZE, 300),
            &mut DummyLabelBatch::new(),
            &Cpu,
            CHANNELS as i64,
            MAX_MOTIF_BATCHES as usize,
            MAX_MOTIF_NETS as i64, //max_nets
            &VarStore::new(Cpu)
        )?;

        debug!("test_train_label: loss={loss:?} params={params:?}");

        assert!(loss < 1.0);

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
        const MOTIF_BATCHES:usize = 3;
        const MOTIF_BATCH_SIZE:usize = 10;
        let channel_motif_nets = [3,2,1];//TODO

        // let mut rnd = thread_rng();

        const TRAIN_LABELS: [bool; BATCH_SIZE] = build_labels::<BATCH_SIZE>();
        const VALID_LABELS: [bool; BATCH_SIZE] = build_labels::<BATCH_SIZE>();

        const MAX_MOTIF_BATCHES:usize = clc_num_batches(1000000, 100, 100, BATCH_SIZE);

        #[derive(Debug)]
        struct DummyLabelBatch {}

        impl DummyLabelBatch {
            pub fn new() -> Self {
                Self {}
            }
        }

        impl DataBatch for DummyLabelBatch {

            fn train_batch(&mut self) -> (Tensor, Tensor) {
                (
                    dummy_labels_net_input(&TRAIN_LABELS, CHANNELS, MAX_MOTIF_NETS, MAX_MOTIF_BATCHES, MOTIF_BATCH_SIZE).to_device(Cuda(0)),
                    Tensor::from_slice(TRAIN_LABELS.as_slice()).totype(Float).to_device(Cuda(0))
                )
            }

            fn valid_batch(&mut self) -> (Tensor, Tensor) {
                (
                    dummy_labels_net_input(&TRAIN_LABELS,  CHANNELS, MAX_MOTIF_NETS, MAX_MOTIF_BATCHES, MOTIF_BATCH_SIZE).to_device(Cuda(0)),
                    Tensor::from_slice(VALID_LABELS.as_slice()).totype(Float).to_device(Cuda(0))
                )
            }
        }

        impl Splittable<1> for DummyLabelBatch {
            fn split(&self) -> [Self;1]
            where
                Self: Sized
            {
                [DummyLabelBatch::new()]
            }
        }

        run_split_on_oom(
            Reducer::new(DummyLabelBatch::new()),
            |i|{
                super::train_label_nodes(
                    &ws,
                    &TrainNodesValues::new(1e-3, 3, Some(AdamW::default().amsgrad(true))),
                    &CommonTrainLabelNodeRunConfig::new(100,  MOTIF_BATCH_SIZE, BATCH_SIZE, 300),
                    &mut DummyLabelBatch::new(),
                    &Cuda(0),
                    CHANNELS as i64,
                    MAX_MOTIF_BATCHES as usize,
                    MAX_MOTIF_NETS as i64, //max_nets
                    &VarStore::new(Cuda(0))
                )
            },
            |v, i|{
                debug!("v={v:?}");
                v
            },
            3
        )?;

        Ok(())
    }


    fn dummy_labels_net_input(labels:&[bool], channels:usize, max_motif_nets:usize, motif_batches:usize, motif_batch_size:usize ) ->Tensor{
        //batch size x channels x max motif nets x motif batches x motif batch size
        let mut labels_iter = labels.iter();
        Tensor::stack(
            labels.iter().map(|label|
                Tensor::stack((0..channels).map(|mb|
                    Tensor::stack((0..max_motif_nets).map(|mmn|
                          Tensor::stack((0..motif_batches).map(|c| if *label {
                              motifs_1(motif_batch_size)
                          } else {
                              motifs_0(motif_batch_size)
                          }).collect::<Vec<_>>().as_slice(), 0),
                    ).collect::<Vec<_>>().as_slice(), 0)
                ).collect::<Vec<_>>().as_slice(), 0)
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

    fn load_test_channels_motifs_models(ws1: &Arc<RwLock<Box<dyn Workspace>>>, channels_motif_ids: Vec<(usize, usize)>) -> Vec<(MotifNodeParams, SyncModule)> {

        channels_motif_ids.into_iter().flat_map(|(channel, motif_id)| load_motif_models(ws1, channel, motif_id, &Cpu).expect("load_motif_models")).collect()
    }

    #[test]
    fn test_train_final() -> Result<(), ThoosaError> {

        // set_max_level(Trace);

        let final_node_params = FinalNodeParams::new(6, 6, 1);

        struct DummyLabelBatch {}

        impl DummyLabelBatch {
            pub fn new() -> Self {
                Self {}
            }
        }

        const BATCH_SIZE:usize = 10;
        const LABELS_SIZE:usize = 6;


        impl DataBatch for DummyLabelBatch {

            fn train_batch(&mut self) -> (Tensor, Tensor) {
                let train_batch = dummy_final_net_input::<LABELS_SIZE>(BATCH_SIZE);
                debug!("train_batch=\n{} \n{}", train_batch.0, train_batch.1);
                train_batch
            }

            fn valid_batch(&mut self) -> (Tensor, Tensor) {
                let valid_batch= dummy_final_net_input::<LABELS_SIZE>(BATCH_SIZE);
                debug!("valid_batch={} \n{}", valid_batch.0, valid_batch.1);
                valid_batch
            }
        }

        let (loss, final_params) = super::train_final_node(
            &Arc::new(RwLock::new(DummyWorkspace::new(0, 100)?)),
            &TrainNodesValues ::new(
                1.0e-2,
                3u32, None
            ),
            &CommonTrainFinalNodeRunConfig::new(300, BATCH_SIZE, 10),
            &mut DummyLabelBatch::new(),
            &Cpu,
            LABELS_SIZE,
            &VarStore::new(Cpu)
        )?;

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
            y[label as usize] = 1.0;
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
    fn save_test_models(ws: &Arc<RwLock<Box<dyn Workspace>>>, base_run_id: usize, channels: usize, max_motif_batches: i64, max_batch_size: i64, motif_node_params_run_id: usize, window: i64) -> Result<(), ThoosaError> {
        // fs::create_dir_all(models_path)?;

        // fs::write(models_path.join("motif_model.json"), serde_json::to_string(&MotifNodeParams::new(window))?)?;


        save_test_label_model(&ws, base_run_id + 1, "seizure", 3, 4, max_motif_batches, max_batch_size)?;
        // fs::write(models_path.join("motif_model.json"), serde_json::to_string(&MotifNodeParams::new(window))?)?;


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
        let _ = train_nets::final_node(&FinalNodeParams::new(features, features, 1)).build_tch_module(&vs_0.root())?;

        vs_0.freeze();
        debug!("save_test_final_model: features={} vs_0.variables()={:?}", features, vs_0.variables());
        vs_0.save(models_path.join("final.safetensors"))?;
        Ok(())
    }

    fn save_test_label_model(ws: &Arc<RwLock<Box<dyn Workspace>>>, run_id: usize, label:&str, channels: usize, max_motif_nets: i64, max_motif_batches: i64, motif_batch_size: i64) -> Result<(), ThoosaError> {
        let label_node_params = LabelNodeParams::new(channels as i64, max_motif_nets, max_motif_batches, motif_batch_size);

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

        let test_motif_model = SeqBuilder::new(100)
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


}
