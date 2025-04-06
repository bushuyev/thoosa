use std::cell::RefCell;
use crate::builder::{Node, SeqBuilder};
use crate::loader::{LoadRunner, Loader, SyncWrapper};
use crate::my_tch::ThoModule;
use crate::parzen::optimizer::{ConfigurationSpace, TPEOptimizer};
use crate::runs::{Empty, Labeled, Logits, NetResources, SavableResult, Simple};
use crate::{panic_to_err, train_nets};
use crate::utils::{cuda_mem, from, run_split_on_oom, tensor_mem_size, Reducer, SplitQueue, Splittable, SplittableDataBatch, SplittableTensor};
use crate::utils::{pad_to, plot_series, train, DataBatch, ListenerResponse, TrainListener};
use crate::ws::{FsWorkspace, Workspace};
use clap::clap_derive::Args;
use crossbeam_utils::sync::WaitGroup;
use futures::FutureExt;
use futures::StreamExt;
use log::{debug, info, log_enabled, trace, warn, Log};
use parking_lot::{Mutex, RawRwLock, RwLock, RwLockReadGuard};
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
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::Instant;
use tch::nn::{AdamW, ConvConfig, Module, OptimizerConfig, VarStore};
use tch::Device::{Cpu, Cuda};
use tch::Kind::Float;
use tch::{nn, Device, IndexOp, Kind, Reduction, Tensor};
use clap::Subcommand;
use thoosa_shared::{clc_step, parse_key_val, slices_len};
use thoosa_shared::parse_range_or_value;
use thoosa_shared::resource::{Resource, ResourceId};
use thoosa_shared::val::{NamedRange, NamedVal, Val};
use thoosa_shared::{batch_windows, clc_num_batches, make_combos, RangeOrValue, ThoosaError};
use thoosa_shared::val::Val::UInt;
use thoosa_store::Store;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use log::Level::Debug as DebugLevel;
use thoosa_properties_map::{FromHashmap, ToHashmap};
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_shared::run_configs::{Applicable, RunConfig};
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};
use thoosa_shared::run_configs::{OptRunConfigs, RunId};
use crate::find_motifs_run_config::*;
use crate::transforms::filters::*;

#[derive(Args, Debug, Clone)]
pub struct TrainLabelNetsArgs {
    #[arg(long)]
    pub motif_selection_run_id: Vec<RunId>,

    #[arg(long)]
    train_resource_group: String,

    #[arg(long)]
    valid_resource_group: String,

    #[command(flatten)]
    pub fixed_config:OptConfig,

    #[arg(long)]
    pub run_config_cnt:String,//TrainLabelNetsRunConfig
}

impl TrainLabelNetsArgs {

    pub fn resource_config(&self) -> NetResources<Labeled> {
        NetResources::new(
            Labeled::new(self.train_resource_group.clone()), Labeled::new(self.valid_resource_group.clone())
        )
    }

}


#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
#[run_config_opt]
pub struct TrainLabelNetsRunConfig {
    #[in_map]
    pub id:Option<RunId>,

    #[in_map]
    #[run_config]
    pub common_training_args:CommonTrainingRunConfig,

    #[in_map]
    #[run_config]
    pub net: LabelNetTypeArgs,
}

impl TrainLabelNetsRunConfig {
    pub fn new(id: Option<RunId>, common_training_args: CommonTrainingRunConfig, net: LabelNetTypeArgs) -> Self {
        Self { id, common_training_args, net }
    }
}

//batch transform, for single tensor transform pass transform_fn to the batch impl
struct BatchTransform<B, F>
where
    B: DataBatch + Debug,
    F: FnMut(Tensor) -> Tensor + Sync + Send,
{
    inner: Rc<Box<B>>,
    transform_fn:Rc<RefCell<F>>
}

impl<B, F> Debug for BatchTransform<B, F>
where
    B: DataBatch + Debug,
    F: FnMut(Tensor) -> Tensor + Sync + Send,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchTransform")
                 .field("inner", &self.inner)
                 .finish()
    }
}

impl <B, F> BatchTransform<B, F>
where
    B: DataBatch + Debug,
    F: FnMut(Tensor) -> Tensor + Sync + Send,
{
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

impl<B, F> DataBatch for BatchTransform<B, F>
where
    B: DataBatch + Debug,
    F: FnMut(Tensor) -> Tensor + Sync + Send,
{
    fn train_batch(&self) -> (Tensor, Tensor) {
        let (inputs, labels) = self.inner.train_batch();
        ((self.transform_fn.borrow_mut())(inputs), labels)
    }

    fn valid_batch(&self) -> (Tensor, Tensor) {
        let (inputs, labels) = self.inner.valid_batch();
        ((self.transform_fn.borrow_mut())(inputs), labels)
    }
}

impl<B, F> Splittable<1> for BatchTransform<B, F>
where
    B: SplittableDataBatch + Debug,
    F: FnMut(Tensor) -> Tensor + Sync + Send,
{
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
* TODO rename
*/
#[derive(Clone)]
struct LoadingTensorsDataBatch<'b>  {
    train: Rc<Vec<(i64, Vec <Resource>)>>,
    val: Rc<Vec<(i64, Vec <Resource>)>>,
    batch_size: usize,
    loader: Arc<RwLock<LoadRunner<Box<Tensor>>>>,
    device:Device,
    transform_fn:Option<Arc<RwLock<Box<dyn Fn(Box<Tensor>)->Box<Tensor> + Sync + Send  + 'b>>>>//for single tensor, wrap into BatchTransform for per-batch transform
}

impl Debug for LoadingTensorsDataBatch<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("")
            .field("train", &self.train)
            .field("val", &self.val)
            .field("batch_size", &self.batch_size)
            .field("device", &self.device)
            .finish()
    }
}

impl <'b> LoadingTensorsDataBatch<'b> {

    pub fn new(
        train: Rc<Vec<(i64, Vec <Resource>)>>,
        val: Rc<Vec<(i64, Vec <Resource>)>>,
        batch_size: usize,
        loader: Arc<RwLock< LoadRunner<Box<Tensor>>>>,
        device:Device,
        transform_fn:Option<Box<dyn Fn(Box<Tensor>)->Box<Tensor> + Sync + Send  + 'b>>
    ) -> Self {

        Self {
            train,
            val,
            batch_size,
            loader,
            device,
            transform_fn: transform_fn.map(|f|Arc::new(RwLock::new(f)))
        }
    }

    fn choose_for_batch(class_resources: &Vec<(i64, Vec<Resource>)>, batch_size: usize, loader: Arc<RwLock<LoadRunner<Box<Tensor>>>>, device: &Device, transform_fn: Option<Arc<RwLock<Box<dyn Fn(Box<Tensor>) -> Box<Tensor> + Sync + Send + 'b>>>>) -> (Tensor, Tensor) {
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

    fn load_tensor(loader: &Arc<RwLock<LoadRunner<Box<Tensor>>>>, area: u64, x_y_acc_in_exec: Arc<Mutex<Vec<(Box<Tensor>, i64)>>>, wg_exec: WaitGroup, transform_fn: Option<Arc<RwLock<Box<dyn Fn(Box<Tensor>) -> Box<Tensor> + Sync + Send + 'b>>>>, resource_id: Resource, class: i64) {
        debug!("LoadingTensorsDataBatch::load_tensor: 0 resource_id={resource_id:?} area={area}");

        loader.write().exec_scoped(move |loader, keys_purge| {
            async move {
                debug!("LoadingTensorsDataBatch::load_tensor: 1 resource_id={resource_id:?} area={area}");
                let resource_id_debug = resource_id.clone();
                let t = Loader::<Box<Tensor>>::load_future(Arc::clone(&loader), resource_id, area).await;

                debug!("LoadingTensorsDataBatch::load_tensor: 2 resource_id={resource_id_debug:?} area={area}");

                let t = if let Some(transform) = transform_fn {
                    transform.read()(t)
                } else {
                    t
                };
                debug!("LoadingTensorsDataBatch::load_tensor: 3 resource_id={resource_id_debug:?} area={area}");
                x_y_acc_in_exec.lock().push((t, class));
                debug!("LoadingTensorsDataBatch::load_tensor: 4 resource_id={resource_id_debug:?} area={area}");
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

impl DataBatch for LoadingTensorsDataBatch<'_> {

    fn train_batch(&self) -> (Tensor, Tensor){
        tch::no_grad(|| {
            Self::choose_for_batch(&self.train, self.batch_size, Arc::clone(&self.loader), &self.device, self.transform_fn.clone())
        })
    }

    fn valid_batch(&self) -> (Tensor, Tensor){
        tch::no_grad(|| {
            Self::choose_for_batch(&self.val, self.batch_size, Arc::clone(&self.loader), &self.device, self.transform_fn.clone())
        })
    }
}


impl Splittable<1> for LoadingTensorsDataBatch<'_> {
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

impl SplittableDataBatch for LoadingTensorsDataBatch<'_> {}


struct ClassTensorsLoaderData  {
    train_resources: Vec<(Resource, Vec<f32>)>,
    valid_resources: Vec<(Resource, Vec<f32>)>,
    loader: Arc<RwLock< LoadRunner<Box<Tensor>>>>,
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
        loader: Arc<RwLock< LoadRunner<Box<Tensor>>>>,
        device:Device,
        transform_fn:Option<Box<dyn Fn(Box<Tensor>)->Box<Tensor> + Sync + Send >>
    ) -> Self {
        Self {
            train_resources: train_loader,
            valid_resources: valid_loader,
            batch_size,
            loader,
            device,
            transform_fn// transform_fn.map(|f|Arc::new(RwLock::new(f)))
        }
    }

}

impl DataBatch for ClassTensorsLoaderData {

    fn train_batch(&self) -> (Tensor, Tensor){
        Self::batch(&self.train_resources, self.batch_size, Arc::clone(&self.loader), &self.device, self.transform_fn.as_ref())
    }

    fn valid_batch(&self) -> (Tensor, Tensor){
        Self::batch(&self.valid_resources, self.batch_size, Arc::clone(&self.loader), &self.device, self.transform_fn.as_ref())
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
pub struct TrainLabelNetsRunConfigResult {
    select_motifs_for_label_run_id:u32,
    label:String,
    loss: f32,
    label_node_params: LabelNetParams,
    train_label_nets_run_config: TrainLabelNetsRunConfig,
    var_store: VarStore
}

impl TrainLabelNetsRunConfigResult {
    pub fn new(find_motifs_run_id:RunId, label:String, loss: f32, label_node_params: LabelNetParams, train_motifs_nodes_values: TrainLabelNetsRunConfig, var_store: VarStore) -> Self {
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

    fn save(&mut self, store: &Arc<Mutex<Box<dyn Store>>>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<u32, ThoosaError> {
        let run_config_id =  store.lock().save_run_config(
            Some(&[self.select_motifs_for_label_run_id]),
            &TrainLabelNetsRunConfig::type_name(),
            "Label",
            "todo",
            self.train_label_nets_run_config.to_properties_iter(None).collect()
        )?;

        ws.read().save_label_model(run_config_id, &self.label, &self.var_store, &self.label_node_params).expect("save_label_model failed");

        Ok(run_config_id)
    }
}

pub fn run_opt_train_label_nets(
    select_motifs_for_label_run_id:&RunId,
    run_configs: &Arc<OptRunConfigs<Vec<NamedVal<TrainLabelNetsRunConfigFields>>, Vec<NamedRange<TrainLabelNetsRunConfigFields>>, Vec<Hyperparameter<TrainLabelNetsRunConfigFields>>, TrainLabelNetsRunConfigFields>>,
    train_labels_resources:Rc<Vec<(i64, Vec <Resource>)>>, //TODO i64 to 1/0 enum for binary class
    valid_labels_resources: Rc<Vec<(i64, Vec <Resource>)>>,
    channels_motifs_nets:Arc<MotifsNetsSelectionApplicable>,
    label: &String,
    fixed_config: &OptConfig,
    slices_len: i64,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    loader: &Arc<RwLock< LoadRunner<Box<Tensor>>>>,
    device: &Device
) -> Result<Box<dyn SavableResult>, ThoosaError> {

    // let nets_count = channels_motifs_nets.read().iter().map(|nets| nets.1.len()).sum::<usize>() as i64;
    // if nets_count <= 0 {
    //     panic!("no motif nets found");//TODO skip?
    // }
    // TODO to nets in some channel ok?

    let channels = ws.read().channels();
    let device = device.clone();
    let max_input = ws.read().max_input();
    let nets_count = channels_motifs_nets.nets_count();
    debug!("run_opt_train_label_nodes: label={label:?} run_configs={run_configs:?} channels={channels} cuda_mem={:?}", cuda_mem());
    let channels_motifs_nets_for_batch = Arc::clone(&channels_motifs_nets);
    opt_train(
        fixed_config,
        |vs:&VarStore, run_config:&TrainLabelNetsRunConfig, _listener: &mut dyn TrainListener|{//TODO add _listener to train_label_nodes_2

            // let batch = Box::new(LoadingTensorsDataBatch::new(
            //     train_labels_resources.clone(),
            //     valid_labels_resources.clone(),
            //     run_config.common_training_args.batch_size as usize,
            //     Arc::clone(&loader),
            //     device.clone(),
            //     Some(Box::new(|t|{
            //         match &run_config.net {
            //             LabelNetType::MaxPadding(params) => {
            //                 // apply_motif_nets_with_padding(&t, Arc::clone(&channels_motifs_nets), &device, max_input as u32)
            //                 channels_motifs_nets.write().apply(&Box::new(t.shallow_clone()), MotifsNetsPadding::Max).expect("channels_motifs_nets/apply failed")//TODO Box
            //             }
            //             LabelNetType::FixedNumSlices(params) => {
            //                 // apply_motif_nets(&t, Arc::clone(&channels_motifs_nets), &device, params.slices_len)
            //                 channels_motifs_nets.write().apply(&Box::new(t.shallow_clone()), MotifsNetsPadding::Fixed(params.slices_len)).expect("channels_motifs_nets/apply failed")//TODO Box
            //             }
            //         }
            //     }))
            //
            // ));

            let mut result = None::<(f32, LabelNetParams)>;
            // let net_type = run_config.net.clone();
            let channels_motifs_nets_for_batch = Arc::clone(&channels_motifs_nets);
            run_split_on_oom(
                Reducer::new(LoadingTensorsDataBatch::new(
                    train_labels_resources.clone(),
                    valid_labels_resources.clone(),
                    run_config.common_training_args.batch_size as usize,
                    Arc::clone(&loader),
                    device.clone(),
                    Some(Box::new(move |t|{
                        match &run_config.net {
                            LabelNetTypeArgs::MaxPadding(params) => {
                                // apply_motif_nets_with_padding(&t, Arc::clone(&channels_motifs_nets), &device, max_input as u32)
                                channels_motifs_nets_for_batch.apply(&Box::new(t.shallow_clone()), NetsPadding::Max).expect("channels_motifs_nets/apply failed")//TODO Box
                            }
                            LabelNetTypeArgs::FixedNumSlices(params) => {
                                // apply_motif_nets(&t, Arc::clone(&channels_motifs_nets), &device, params.slices_len)
                                channels_motifs_nets_for_batch.apply(&Box::new(t.shallow_clone()), NetsPadding::Fixed(params.slices_len)).expect("channels_motifs_nets/apply failed")//TODO Box
                            }
                        }
                    }))

                )),
                |b|{
                    train_label_net(
                        ws, run_config, b, &device, channels, nets_count, vs, &channels_motifs_nets
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
        run_configs,
        device,
        "label"
    ).map(|(var_store, label_node_params, run_config, min_loss)| {
        Box::new(TrainLabelNetsRunConfigResult::new(channels_motifs_nets.motif_selection_run_id, label.clone(), min_loss, label_node_params, run_config, var_store)) as Box<dyn SavableResult>
    })
}

pub fn train_label_net<B:DataBatch> (
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    run_config:&TrainLabelNetsRunConfig,
    // comon_label_node_params:&CommonTrainLabelNodeRunConfig,
    mut batch:&mut B,
    device: &Device,
    channels:u32,
    max_nets:u32,
    vs_0: &VarStore,
    channels_motifs_nets:&Arc<MotifsNetsSelectionApplicable>,
) ->  Result<(f32, LabelNetParams), ThoosaError> {

    let guard = channels_motifs_nets;

    let label_node_params = match &run_config.net {//TODO return net params from net builder
        LabelNetTypeArgs::MaxPadding(config) => {
            debug!("train_label_net: LabelNetType::MaxPadding(config={config:?})");


            let find_motifs_run_config = guard.find_motifs_applicable.run_config();
            let slices_len = slices_len(ws.read().max_input(), find_motifs_run_config.window(), find_motifs_run_config.step());

            debug!("train_label_net: LabelNetType::MaxPadding max_input={} find_motifs_run_config={find_motifs_run_config:?} slices_len={slices_len}", ws.read().max_input());

            LabelNetParams::new(channels, max_nets, config.num_layers, slices_len, LabelNetTypeParams::MaxPadding(MaxPaddingNetParams::new(config.num_layers)))
        }
        LabelNetTypeArgs::FixedNumSlices(config) => {
            debug!("train_label_net: LabelNetType::FixedNumSlices(config={config:?})");



            LabelNetParams::new(channels, max_nets, config.num_layers, config.slices_len, LabelNetTypeParams::FixedNumSlices(FixedNumSlicesNetParams::new(config.num_layers, guard.nets_count())))

        }
    };

    debug!("train_label_net: label_node_params={label_node_params:?} cuda_mem={:?}", cuda_mem());

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
        batch as &mut dyn DataBatch,
        &label_node_params.label_net().build_tch_module(&vs_0.root())?,

        |pred: &Tensor, target: &Tensor| {
            debug!("train_label_net: pred.size={:?}", pred.size());
            let pred_y = pred.squeeze();
            debug!("train_label_net: pred_y={:?} target={:?} kind={:?}", pred_y.size(), target.size(), target.kind());
            pred_y.binary_cross_entropy::<Tensor>(&target, None, Reduction::Mean)
        },
        run_config.common_training_args.cycles as usize,
        optimizer.build(&vs_0, run_config.common_training_args.lr)?,

        |t: Tensor| {
            debug!("train_label_net: t.size={:?} cuda_mem={:?}", t.size(), cuda_mem());
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

pub fn run_opt_train_final_net(
    train_label_nodes_run_ids: Option<&Vec<RunId>>,
    train_final_run_config: &OptRunConfigs<Vec<NamedVal<TrainFinalNetRunConfigFields>>, Vec<NamedRange<TrainFinalNetRunConfigFields>>, Vec<Hyperparameter<TrainFinalNetRunConfigFields>>, TrainFinalNetRunConfigFields>,
    train_resources:Vec<(Resource, Vec<f32>)>,
    valid_resources: Vec<(Resource, Vec<f32>)>,
    mut label_models: HashMap<String, Vec<(RunId, LabelNetParams, ThoModuleWithStore)>>,
    labels_channels_motifs_models:HashMap<String, Vec<(FindMotifsRunConfig, Vec<(MotifNetParams, ThoModuleWithStore)>)>>,
    fixed_config: &OptConfig,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Mutex<Box<dyn Store>>>,
    loader: &Arc<RwLock< LoadRunner<Box<Tensor>>>>,
    device: &Device,
) -> Result<Box<dyn SavableResult>, ThoosaError> {
    let labels_len = labels_channels_motifs_models.len() as u32;

    let labels_channels_motifs_models_rw = labels_channels_motifs_models.into_iter().fold(HashMap::new(), |mut a, v|{
        a.insert(v.0, Arc::new(RwLock::new(v.1)));
        a
    });
    let device = device.clone();

    let labels = store.lock().list_labels()?;

    debug!("run_opt_train_final: labels={labels:?}");

    assert_eq!(labels.len() as u32, labels_len);

    let max_input = ws.read().max_input();

    let label_nets_applicable = Arc::new(LabelNetsApplicable::load_all_filtered_by_run_ids(train_label_nodes_run_ids.map(|v|v.as_slice()), store, ws, device));



    opt_train(
        fixed_config,
         |vs:&VarStore, run_config_values:&TrainFinalNetRunConfig, listener: &mut dyn TrainListener|{
             let label_nets_applicable_rc = Arc::clone(&label_nets_applicable);

            let  batch = Rc::new(Box::new(ClassTensorsLoaderData::new(
                train_resources.clone(),
                valid_resources.clone(),
                run_config_values.common_training_args.batch_size as usize,
                Arc::clone(&loader),
                device.clone(),

                None,
                // Some(Box::new(move |t|{
                //     label_nets_applicable_rc.apply(&t, ()).expect("label_nets_applicable.apply failed")
                // }))
            )));

            train_final_node(
                ws,
                run_config_values,
                &mut BatchTransform::new(Rc::clone(&batch), |t|{
                    *label_nets_applicable_rc.apply(&Box::new(t.shallow_clone()), ()).expect("label_nets_applicable.apply failed")
                }),
                &device,
                labels_len,
                vs
            )
        },
        &train_final_run_config,
        device,
        "final"
    ).map(|(var_store, final_node_params, train_final_nodes_values, min_loss)| {
        debug!("run_opt_train_final: done, min_loss={min_loss} final_node_params={final_node_params:?}");
        Box::new(TrainFinalNodesRunConfigResult::new(train_label_nodes_run_ids, min_loss, var_store, final_node_params, train_final_nodes_values)) as Box<dyn SavableResult>
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
    motif_nodes: &mut [(MotifNetParams, ThoModuleWithStore)],
    batch_channel: Tensor,
    motif_width: u32,
    sum_motif_nodes: u32,
    slices_len: u32,
    device: &Device
) -> Result<Tensor, ThoosaError> {

    trace!("apply_channel_motifs: motif_nodes={motif_nodes:?}");
    debug!("apply_channel_motifs: batch_channel={batch_channel:?} motif_width={motif_width} max_motif_nodes={sum_motif_nodes} slices_len={slices_len} device={device:?}");

    if motif_nodes.is_empty() {
        return Ok(Tensor::zeros([batch_channel.size()[0], sum_motif_nodes as i64, slices_len as i64], (Float, *device)));
    }

    let (input_slice_step, pad) = clc_step(batch_channel.size()[1] as u32, motif_width, slices_len); //(channel.size()[0]-motif_width)/100;

    let unfolded = batch_channel
        .pad([0, pad as i64], "constant", 0.)//pad: last dimension [0 - padding left, pad - padding right]
        .unfold(1, motif_width as i64, input_slice_step as i64);

    debug!("apply_channel_motifs: channel.size={:?} motif_width={motif_width} input_slice_step={input_slice_step} pad={pad}  unfolded.size()={:?}", batch_channel.size(), unfolded.size());

    let motif_dists = motif_nodes.iter_mut().map(|(_params, net)| {
        net.set_device(device.clone());

        let mut motif_distance_splits = vec![];
        run_split_on_oom(
            SplitQueue::<2, _>::new(SplittableTensor::<0>(unfolded.shallow_clone())),
            |t|{
                let t = t.as_ref();
                trace!("apply_channel_motifs: forward: t=\n{t}");
                let r = net.infer(&t.to_device(device.clone()));
                debug!("apply_channel_motifs: forward: size={:?} r.size={:?}", t.as_ref().size(), r.size());
                trace!("apply_channel_motifs: forward: r=\n{r}");

                Ok(r)
            },
            |v, t|{
                debug!("v={v:?}");
                motif_distance_splits.push(v);
            },
            10//TODO config
        ).unwrap();//TODO unwrap

        net.set_device(Cpu);

        debug!("apply_channel_motifs: motif_distance_splits={motif_distance_splits:?}");
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

pub fn apply_channel_motifs_with_padding(input_slice_step: u32,
                                         motif_nodes: &mut [(MotifNetParams, ThoModuleWithStore)],
                                         channel: Tensor, //batch , data
                                         motif_width: u32,
                                         max_motif_nodes: u32,
                                         slices_len: u32,
                                         device: &Device
) -> Result<Tensor, ThoosaError> {

    trace!("apply_channel_motifs_with_padding: motif_nodes={motif_nodes:?}");
    debug!("apply_channel_motifs_with_padding: channel={channel:?} motif_width={motif_width} input_slice_step={input_slice_step} max_motif_nodes={max_motif_nodes} motif_nodes.len={} max_dists_len={slices_len} device={device:?}", motif_nodes.len());

    if motif_nodes.is_empty() {
        return Ok(Tensor::zeros([channel.size()[0], max_motif_nodes as i64, slices_len as i64], (Float, *device)));
    }

    let unfolded = channel.unfold(1, motif_width as i64, input_slice_step as i64);

    debug!("apply_channel_motifs_with_padding: unfolded.size={:?}", unfolded.size());

    let motif_dists = motif_nodes.iter_mut().map(|(_params, net)| {
        net.set_device(device.clone());
        debug!("apply_channel_motifs: net={net:?}");

        let mut motif_distance_splits = vec![];
        run_split_on_oom(
            SplitQueue::<2, _>::new(SplittableTensor::<0/*batch dimension*/>(unfolded.shallow_clone())),
            |t|{

                debug!("apply_channel_motifs_with_padding: forward: t.size={:?}", t.as_ref().size());
                let r = net.infer(&t.as_ref().to_device(device.clone())).flatten(1,2);//flatten - removes always 1-sized distance dimension [batch, slices, slices_len]->[batch, slices, [distance]]
                debug!("apply_channel_motifs_with_padding: forward:  r.size={:?}", r.size());

                Ok(r)
            },
            |v, t|{
                debug!("v={v:?}");
                motif_distance_splits.push(v);
            },
            10//TODO config
        ).unwrap();//TODO unwrap

        net.set_device(Cpu);

        debug!("apply_channel_motifs: motif_distance_splits={motif_distance_splits:?}");
        if motif_distance_splits.len() > 0 {
            Ok(Tensor::concat(&motif_distance_splits, 0))

        } else if motif_distance_splits.len() == 1 {
            Ok(motif_distance_splits.pop().unwrap())

        } else {
            Err(ThoosaError::new("no motif distances "))

        }
    }).collect::<Result<Vec<_>, ThoosaError>>()?;


    debug!("apply_channel_motifs_with_padding: motif_dists={motif_dists:?}");
    //TODO pad_sequence?
    let motif_nets_out = Tensor::stack(motif_dists.as_slice(), 1);

    debug!("apply_channel_motifs_with_padding: motif_dists.len={} motif_nets_out.size={:?}", motif_dists.len(), motif_nets_out.size());

    let r = pad_to(&motif_nets_out, &[channel.size()[0], max_motif_nodes as i64, slices_len as i64]);

    debug!("apply_channel_motifs_with_padding: r.size={:?}", r.as_ref().map(|r|r.size()));

    r
}


pub fn apply_motif_and_label_nets(
    batch_t: &Tensor,
    label_models: &mut HashMap<String, Vec<(RunId, LabelNetParams, ThoModuleWithStore)>>,
    labels_channels_motifs_models_rw: &HashMap<String, Arc<RwLock<Vec<(FindMotifsRunConfig, Vec<(MotifNetParams, ThoModuleWithStore)>)>>>>,
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

        // let motif_nets_t = apply_motif_nets(&batch_t, Arc::clone(channels_motifs_models), &device, 100);

        for (label_node_run_id, label_node_params, label_node) in label_nodes.iter_mut() {
            let motif_nets_t = match &label_node_params.net_type {
                LabelNetTypeParams::MaxPadding(params) => {
                    apply_motif_nets_with_padding(&batch_t, Arc::clone(&channels_motifs_models), &device, max_input)
                }
                LabelNetTypeParams::FixedNumSlices(params) => {
                    apply_motif_nets(&batch_t, Arc::clone(&channels_motifs_models), &device, label_node_params.slices_len)
                }
            };

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
    train_label_nodes_run_ids: Vec<RunId>,
    loss: f32,
    var_store: VarStore,
    final_node_params: FinalNetParams,
    train_final_net_run_config: TrainFinalNetRunConfig
}

impl TrainFinalNodesRunConfigResult {
    pub fn new(train_label_nodes_run_ids: Option<&Vec<RunId>>, loss: f32, var_store: VarStore, final_node_params: FinalNetParams, train_final_nodes_values: TrainFinalNetRunConfig) -> Self {
        Self {
            train_label_nodes_run_ids: train_label_nodes_run_ids.expect("train_label_nodes_run_ids").clone(),
            loss,
            var_store,
            final_node_params,
            train_final_net_run_config: train_final_nodes_values,
        }
    }
}

impl SavableResult for TrainFinalNodesRunConfigResult {
    fn value(&self) -> f32 {
        self.loss
    }

    fn save(&mut self, store: &Arc<Mutex<Box<dyn Store>>>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<u32, ThoosaError> {
        let run_config_id =                 store.lock().save_run_config(
            Some(&self.train_label_nodes_run_ids),
            &TrainFinalNetRunConfig::type_name(),
            "Final",
            "todo",
            self.train_final_net_run_config.to_properties_iter(None).collect()
        )?;

        ws.read().save_final_model(run_config_id, &self.var_store, &self.final_node_params).expect("save_final_model failed");

        Ok(run_config_id)
    }
}

pub fn train_final_node(ws: &Arc<RwLock<Box<dyn Workspace>>>, run_config_values: &TrainFinalNetRunConfig, batch:&mut dyn DataBatch, device: &Device, labels_len: u32, vs_0: &VarStore) ->  Result<(f32, FinalNetParams), ThoosaError>{
    let final_node_params= FinalNetParams::new(labels_len, run_config_values.net.num_layers);

    let mut optimizer = nn::Adam::default();
   /* if let Some(adam_w) = run_config_values.adam_w {
        debug!("train_final_nodes: setting adam_w={adam_w:?}");
        optimizer = optimizer.beta1(adam_w.beta1).beta2(adam_w.beta2);
    }*/

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
        run_config_values.common_training_args.cycles as usize,
        optimizer.build(&vs_0, run_config_values.common_training_args.lr)?,
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
    motif_node_params: MotifNetParams,
    #[allow(dead_code)]
    label_node_params: LabelNetParams
}

impl LabelModels {

    pub fn new(label:String, corr_model: Box<dyn Module>, motif_models: Vec<Vec<Box<dyn Module>>>, motif_node_params: MotifNetParams, label_node_params: LabelNetParams) -> Self {
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


pub fn load_final_model(run_id: RunId, ws:&Arc<RwLock<Box<dyn Workspace>>>, device: &Device) ->Result<(FinalNetParams, ThoModuleWithStore), ThoosaError> {
    let (model_path, params_path) = ws.read().final_model(run_id)?;
    debug!("load_final_model: params_path={model_path:?} run_id={run_id}");


    let final_node_params = FinalNetParams::from_file(&params_path)?;
    debug!("load_final_model:  final_node_params={final_node_params:?}");

    let node = ThoModuleWithStore::load(final_node(&final_node_params), model_path, *device)?;

    Ok((final_node_params, node))
}


#[derive(Args, Debug, Clone)]
pub struct CommonTrainingArgs {
    #[arg(long, default_value_t = 1000)]
    pub cycles: usize,

    #[arg(long, default_value_t = 10)]
    pub batch_size: usize,//number of the motif slices in a batch

    //TODO remove
    #[arg(long, value_parser=parse_range_or_value::<f64>)]
    pub lr: RangeOrValue<f64>,

    #[arg(long, default_value_t = 0.2)]
    pub max_loss: f32,

    //TODO move
    #[arg(long, default_value_t = 10)]
    pub opt_max_evals: u32,

    //TODO move
    #[arg(long, default_value_t = 3)] //TODO validate opt_n_init < opt_max_evals
    pub opt_n_init: u32,
}


#[derive(Args, Debug, Clone)]
pub struct TrainMotifsNetsArgs {

    #[arg(long)]
    pub motif_selection_run_id: Vec<RunId>,

    #[command(flatten)]
    pub fixed_config: OptConfig,

    #[arg(long)]
    pub run_config_cnt:String,//TrainMotifsNetsRunConfig

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
    pub id:Option<RunId>,

    #[in_map]
    #[run_config]
    pub common_training_args:CommonTrainingRunConfig,

    #[in_map]
    #[run_config]
    num_layers:u32


}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
pub struct CommonTrainingRunConfig {

    #[in_map]
    #[run_config]
    pub cycles: u32,

    #[in_map]
    #[run_config]
    pub batch_size: u32,//number of the motif slices in a batch

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
    pub fn new(cycles: u32, batch_size: u32, lr: f64, max_loss: f32, opt_max_evals: u32, opt_n_init: u32) -> Self {
        Self { cycles, batch_size, lr, max_loss, opt_max_evals, opt_n_init }
    }
}

#[derive(Args, Debug, Clone)]
pub struct OptConfig {

    #[arg(long, default_value_t = 100)]
    pub opt_max_evals:u32,

    #[arg(long, default_value_t = 10)]
    pub opt_num_init:u32
}

impl OptConfig {
    pub fn new(max_evals: u32, n_init: u32) -> Self {
        Self { opt_max_evals: max_evals, opt_num_init: n_init }
    }
}


#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct MotifNetParams {
    num_features: u32,
    num_layers:u32
}

impl MotifNetParams {

    pub fn new(num_features: u32, num_layers: u32) -> Self {
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

        MotifNetParams::new(
            properties.get("num_features").expect(format !("no num_features in {:?}", properties).as_str()).try_into().expect("num_features as_i64"), 1,
        )
    }

    pub fn from_file(path:&PathBuf) -> Result<Self, ThoosaError> {
        serde_json5::from_str::<MotifNetParams>(
            fs::read_to_string(&path)?.as_str()
        ).map_err(|serde_err|ThoosaError::new(format!("err={}, path={path:?}", serde_err.to_string())))
    }

    pub fn num_features(&self) -> u32 {
        self.num_features
    }

    pub fn with_num_layer(&self, num_layers:u32) -> Self {
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

#[derive(Debug)]
pub struct MotifsNetsSelectionApplicable {
    pub motif_selection_run_id:RunId,
    pub find_motifs_applicable: FindMotifsApplicable,
    pub channels_motifs: Vec<Vec<(MotifNetParams, ThoModuleWithStore)>>,
    pub device:Device,
    pub max_input:u32
}

impl MotifsNetsSelectionApplicable {

    pub fn new(motif_selection_run_id:RunId, store:&Arc<Mutex<Box<dyn Store>>>, ws: &Arc<RwLock<Box<dyn Workspace>>>, device:Device) -> Self {
        let channels = ws.read().channels();

        debug!("MotifsNetsSelectionApplicable::new: motif_selection_run_id={motif_selection_run_id} channels={channels:?}");

        let find_motifs_run_config_ids = store.lock().get_input_run_ids(motif_selection_run_id, Some(&FindMotifsRunConfig::type_name())).expect(format!("FindMotifsRunConfig for motif_selection_run_id={motif_selection_run_id:?} not found").as_str());
        assert_eq!(find_motifs_run_config_ids.len(), 1);


        let find_motifs_run_config = FindMotifsRunConfig::from_properties_map__(find_motifs_run_config_ids[0], &store.lock().list_run_config_values(find_motifs_run_config_ids[0]).expect("list_run_params")).unwrap();

        debug!("MotifsNetsSelectionApplicable::new: find_motifs_run_config={find_motifs_run_config:?}");

        let channels_motifs = store
            .lock()
            .list_selected_motifs(motif_selection_run_id, channels)
            .expect("list_selected_motifs") //TODO move out
            .into_iter()
            .enumerate()
            .map(|(channel, channels_motifs)| {
                debug!("MotifsNetsSelectionApplicable::new: channels_motifs: channel={channel} channels_motifs={channels_motifs:?}");

                channels_motifs
                    .into_iter()
                    .flat_map(move |(motif_id, corr)| {
                        debug!("MotifsNetsSelectionApplicable::new: channels_motifs: channel={channel} motif_id={motif_id}");
                        ws.read().load_motif_models(channel as u32, motif_id, &device)
                            .expect("load_motif_models")
                    })
                    .collect::<Vec<_>>()

            })
            .collect::<Vec<Vec<_>>>();

        debug!("MotifsNetsSelectionApplicable::new: channels_motifs={:?}", channels_motifs.iter().map(|c|c.len()).collect::<Vec<_>>());

        MotifsNetsSelectionApplicable {
            motif_selection_run_id,
            find_motifs_applicable: find_motifs_run_config.to_applicable(&ws.read(), device),
            channels_motifs,
            device,
            max_input: ws.read().max_input()
        }
    }

    pub fn nets_count(&self) -> u32 {
        self.channels_motifs.iter().map(|motifs| motifs.len() as u32).max().expect("No motif count")
    }
}

#[derive(Debug, Copy, Clone)]
pub enum NetsPadding {
    Max,
    Fixed(u32)
}

impl Applicable<&Box<Tensor>, Box<Tensor>, NetsPadding> for MotifsNetsSelectionApplicable {

    fn apply(&self, t: &Box<Tensor>, options: NetsPadding) -> Result<Box<Tensor>, ThoosaError> {
        debug!("Applicable for MotifsNetsSelectionApplicable::apply: t.size={:?} options={options:?}", t.size());
        let motif_width= self.find_motifs_applicable.run_config().window;

        match options {
            NetsPadding::Max => {

                let unfolded = self.find_motifs_applicable.apply(t, FindMotifsRunConfigOptions::FilterUnfold)?;

                trace!("Applicable for MotifsNetsSelectionApplicable::apply/Max: unfolded=\n{unfolded}");

                let max_num_slices = slices_len(self.max_input, motif_width, self.find_motifs_applicable.run_config().step);

                debug!("Applicable for MotifsNetsSelectionApplicable::apply/Max: motif_width={motif_width} self.max_input={} max_num_slices={max_num_slices}", self.max_input);

                Ok(Box::new(Tensor::stack(
                    self.channels_motifs
                        .iter()
                        .enumerate()
                        .map(|(channel, motif_nodes)| {
                            debug!("Applicable for MotifsNetsSelectionApplicable::apply/Max: channel={channel} motif_nodes.len()={}", motif_nodes.len());

                            if motif_nodes.is_empty() {
                                Ok(Tensor::zeros([self.nets_count() as i64, max_num_slices as i64], (Float, self.device)))
                            } else {

                                let unfolded_channel = unfolded.i((channel as i64, .., ..));//channel, slices, slice

                                debug!("Applicable for MotifsNetsSelectionApplicable::apply/Max: unfolded.size={:?}", unfolded_channel.size());
                                trace!("Applicable for MotifsNetsSelectionApplicable::apply/Max: unfolded_channel=\n{unfolded_channel}");

                                let motif_dists = motif_nodes.iter().map(|(_params, net)| {
                                    // net.set_device(device.clone());
                                    debug!("Applicable for MotifsNetsSelectionApplicable::apply/Max: net={net:?}");

                                    let mut motif_distance_splits = vec![];
                                    run_split_on_oom(
                                        SplitQueue::<2, _>::new(SplittableTensor::<0/*batch dimension*/>(unfolded_channel.shallow_clone())),
                                        |t| {
                                            debug!("Applicable for MotifsNetsSelectionApplicable::apply/Max: forward: t.size={:?}", t.as_ref().size());
                                            let r = net.infer(&t.as_ref().to_device(self.device.clone())).flatten(0, 1); //flatten - removes always 1-sized distance dimension [ slices, slices_len]->[slices, [distance]]
                                            debug!("Applicable for MotifsNetsSelectionApplicable::apply/Max: forward:  r.size={:?}", r.size());

                                            Ok(r)
                                        },
                                        |v, t| {
                                            debug!("v={v:?}");
                                            motif_distance_splits.push(v);
                                        },
                                        10 //TODO config
                                    ).unwrap(); //TODO unwrap

                                    // net.set_device(Cpu);

                                    debug!("Applicable for MotifsNetsSelectionApplicable::apply/Max: motif_distance_splits={motif_distance_splits:?}");
                                    if motif_distance_splits.len() > 0 {
                                        Ok(Tensor::concat(&motif_distance_splits, 0))
                                    } else if motif_distance_splits.len() == 1 {
                                        Ok(motif_distance_splits.pop().unwrap())
                                    } else {
                                        Err(ThoosaError::new("no motif distances "))
                                    }
                                }).collect::<Result<Vec<_>, ThoosaError>>()?;


                                debug!("Applicable for MotifsNetsSelectionApplicable::apply/Max: motif_dists={motif_dists:?}");
                                //TODO pad_sequence?
                                let motif_nets_out = Tensor::stack(motif_dists.as_slice(), 0);

                                trace!("Applicable for MotifsNetsSelectionApplicable::apply/Max: motif_nets_out=\n{motif_nets_out}");
                                debug!("Applicable for MotifsNetsSelectionApplicable::apply/Max: motif_dists.len={} motif_nets_out.size={:?}", motif_dists.len(), motif_nets_out.size());

                                let r = pad_to(&motif_nets_out, &[self.nets_count() as i64, max_num_slices as i64]);

                                debug!("Applicable for MotifsNetsSelectionApplicable::apply/Max: r.size={:?}", r.as_ref().map(|r|r.size()));

                                r


                            }


                         })
                        .collect::<Result<Vec<_>, ThoosaError>>().unwrap().as_slice(),
                    0
                )))
            }
            //returns sum of motifs in all channels x slices_len
            NetsPadding::Fixed(slices_len) => {
                let (input_slice_step, pad) = clc_step(t.size()[1] as u32, motif_width, slices_len); //(channel.size()[0]-motif_width)/100;
                debug!("Applicable for MotifsNetsSelectionApplicable::apply/Fixed: input_slice_step={input_slice_step} pad={pad}");

                let filtered = self.find_motifs_applicable.apply(
                    &Box::new(t.pad([0, pad as i64], "constant", 0.)), FindMotifsRunConfigOptions::Filter
                )?;

                let unfolded = filtered.unfold(1, motif_width as i64, input_slice_step as i64);

                let nets_count = self.nets_count();

                Ok(Box::new(Tensor::concat(
                    self.channels_motifs
                        .iter()
                        .enumerate()
                        .filter(|(_, motif_nodes)|!motif_nodes.is_empty())
                        .map(|(channel, motif_nets)| {

                            let unfolded_channel = unfolded.i((channel as i64, .., ..));//channel, slices, slice

                            debug!("Applicable for MotifsNetsSelectionApplicable::apply/Fixed: unfolded.size={:?}", unfolded_channel.size());
                            trace!("Applicable for MotifsNetsSelectionApplicable::apply/Fixed: unfolded_channel=\n{unfolded_channel}");

                            let motif_dists = motif_nets.iter().map(|(_params, net)| {
                                // net.set_device(device.clone());
                                debug!("Applicable for MotifsNetsSelectionApplicable::apply/Fixed: net={net:?}");

                                let mut motif_distance_splits = vec![];
                                run_split_on_oom(
                                    SplitQueue::<2, _>::new(SplittableTensor::<0/*batch dimension*/>(unfolded_channel.shallow_clone())),
                                    |t| {
                                        debug!("Applicable for MotifsNetsSelectionApplicable::apply/Fixed: forward: t.size={:?}", t.as_ref().size());
                                        let r = net.infer(&t.as_ref().to_device(self.device.clone())).flatten(0, 1); //flatten - removes always 1-sized distance dimension [ slices, slices_len]->[slices, [distance]]
                                        debug!("Applicable for MotifsNetsSelectionApplicable::apply/Fixed: forward:  r.size={:?}", r.size());

                                        Ok(r)
                                    },
                                    |v, t| {
                                        debug!("v={v:?}");
                                        motif_distance_splits.push(v);
                                    },
                                    10 //TODO config
                                ).unwrap(); //TODO unwrap

                                // net.set_device(Cpu);

                                debug!("Applicable for MotifsNetsSelectionApplicable::apply/Fixed: motif_distance_splits={motif_distance_splits:?}");
                                if motif_distance_splits.len() > 0 {
                                    Ok(Tensor::concat(&motif_distance_splits, 0))
                                } else if motif_distance_splits.len() == 1 {
                                    Ok(motif_distance_splits.pop().unwrap())
                                } else {
                                    Err(ThoosaError::new("no motif distances "))
                                }
                            }).collect::<Result<Vec<_>, ThoosaError>>()?;


                            debug!("Applicable for MotifsNetsSelectionApplicable::apply/Fixed: motif_dists={motif_dists:?}");

                            let motif_nets_out = Tensor::stack(motif_dists.as_slice(), 0);

                            trace!("Applicable for MotifsNetsSelectionApplicable::apply/Fixed: motif_nets_out=\n{motif_nets_out}");
                            debug!("Applicable for MotifsNetsSelectionApplicable::apply/Fixed: motif_dists.len={} motif_nets_out.size={:?}", motif_dists.len(), motif_nets_out.size());

                            Ok(motif_nets_out)

                        })
                        .collect::<Result<Vec<_>, ThoosaError>>()?.as_slice(),
                    0
                )))
            }
        }


    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabelNetParams {
    channels:u32,
    motif_nodes:u32,
    motif_layers:u32,
    slices_len:u32,
    net_type: LabelNetTypeParams
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
pub struct MaxPaddingNetParams {
    num_layers:u32
}

impl MaxPaddingNetParams {
    pub fn new(num_layers:u32) -> Self {
        Self {num_layers}
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedNumSlicesNetParams {
    num_layers: u32,
    nets_count: u32,
}

impl FixedNumSlicesNetParams {
    pub fn new(num_layers: u32, nets_count: u32) -> Self {
        Self { num_layers, nets_count }
    }
}

impl LabelNetParams {
    pub fn new(channels: u32, max_motif_nodes: u32, motif_layers:u32, slices_len:u32, net_type: LabelNetTypeParams) -> Self {
        Self { channels, motif_nodes: max_motif_nodes, slices_len, motif_layers, net_type }
    }

    pub fn from_file(path:&PathBuf) -> Result<Self, ThoosaError> {
        serde_json5::from_str::<LabelNetParams>(
            fs::read_to_string(&path)?.as_str()
        ).map_err(|serde_err|ThoosaError::new(serde_err.to_string()))
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

    fn label_net_fixed_num_slices(params:&LabelNetParams, fixed_num_slices_params: &FixedNumSlicesNetParams) -> Node {
        debug!("label_node_fixed_num_slices: params={:?}", params);

        let nets_count =  fixed_num_slices_params.nets_count;//sum of motif nets in all channels
        SeqBuilder::new(params.slices_len, None)//Some(2)
            .closure(Box::new(move |t:&Tensor|{
                debug!("label_node_fixed_num_slices: input.size={:?}", t.size());
                t.shallow_clone()

            }), None)
            .split(
                (0..nets_count).map( |_i| {
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
                        (0..nets_count)
                            .map(|net_i|{
                                t.i((.., net_i as i64, ..))
                            })
                            .collect()
                ),//label batch, channel, max_motif_nets, max slices

                Box::new(move |t:Vec<Tensor>|{
                    debug!("label_node_fixed_num_slices: merging t={t:?}");
                    // let out = Tensor::stack(t.as_slice(), 1).mean_dim(1, false, Float);
                    let out = Tensor::concat(t.as_slice(), 1); //.mean_dim(1, false, Float);
                    debug!("label_node_fixed_num_slices: merging out={:?}", out.size());

                    out
                })

            )
            .closure(Box::new(move |t:&Tensor|{
                debug!("label_node_fixed_num_slices: merged final in flatten t.size()={:?}", t.size());

                // let t_0 = t.squeeze_dims(1);
                // debug!("label_node merged final out flatten t_0.size()={:?}", t_0.size());
                // t_0
                t.shallow_clone()

            }), Some(nets_count))

            .linear()
            // .func(Tensor::relu)
            .linear_s(1)
            .func(Tensor::sigmoid)
            .end_seq()

    }

    fn label_net_max_padding(params:&LabelNetParams, max_padding_params: &MaxPaddingNetParams) -> Node {
        debug!("label_node_max_padding: params={:?}", params);

        let max_motif_nets=  params.motif_nodes;
        SeqBuilder::new(params.slices_len, None)// batch x channel x max motif_nets x slices //TODO split?
            .closure(Box::new(move |t:&Tensor|{
                debug!("label_node_max_padding: input.size={:?}", t.size());
                t.shallow_clone()

            }), None)

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
                                debug!("label_node_max_padding: split, t.size={:?}", t.size());
                                t.i((.., .., net_i as i64, ..))
                            })
                            .collect()
                ),//label batch, channel, max_motif_nets, max slices

                Box::new(move |t:Vec<Tensor>|Tensor::stack(t.as_slice(), 1).mean_dim(1, false, Float))

            )
            .closure(Box::new(move |t:&Tensor|{
                debug!("label_node_max_padding: merged final in flatten t.size()={:?}", t.size());

                let t_0 = t.squeeze_dims(2);
                // let t_0 = t..flatten(1, 2);
                debug!("label_node_max_padding: merged final out flatten t_0.size()={:?}", t_0.size());

                t_0

            }), Some(params.channels))

            .linear()
            // .func(Tensor::relu)
            .linear_s(1)
            .func(Tensor::sigmoid)
            .end_seq()

    }

}

pub fn flat_label_node(params: &LabelNetParams, slices_len:u32) -> Node {
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


#[derive(Debug, Serialize, Deserialize, Default)]
pub struct FinalNetParams {
    in_features: u32,
    num_layers: u32
}

impl FinalNetParams {
    pub fn new(in_features: u32, num_layers: u32) -> Self {
        Self { in_features, num_layers }
    }

    pub fn to_properties_map(&self) -> HashMap<String, Val> {
        HashMap::from([
            ("FinalNodeParams.classes_num".into(), self.in_features.into()),
            ("FinalNodeParams.num_layers".into(), self.num_layers.into()),
        ])
    }

    pub fn from_file(path:&PathBuf) -> Result<Self, ThoosaError> {
        serde_json5::from_str::<FinalNetParams>(
            fs::read_to_string(&path)?.as_str()
        ).map_err(|serde_err|ThoosaError::new(serde_err.to_string()))
    }


}

pub fn final_node(params:&FinalNetParams) -> Node {
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

pub struct PlotVecf32 {
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
pub struct TrainMotifsNetsRunConfigResult {
    loss:f32,
    channel:u32,
    motif_id:u32,
    vs:VarStore,
    params: MotifNetParams,
    run_config_values: TrainMotifsNetsRunConfig,
    find_motifs_run_id: RunId
}

impl TrainMotifsNetsRunConfigResult {
    pub fn new(loss:f32, channel:u32, motif_id:u32, vs: VarStore, find_motifs_run_id: RunId, params: MotifNetParams, run_config_values: TrainMotifsNetsRunConfig) -> Self {
        Self { loss, channel, motif_id, vs, params, run_config_values, find_motifs_run_id }
    }
}

impl SavableResult for TrainMotifsNetsRunConfigResult {

    fn value(&self) -> f32 {
        self.loss
    }

    fn save(&mut self, store:&Arc<Mutex<Box<dyn Store>>>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<u32, ThoosaError> {
        let run_config_id = store.lock().save_run_config(
            Some(&[self.find_motifs_run_id]),
            &TrainMotifsNetsRunConfig::type_name(),
            "TrainMotifsNetsRunConfigResult2",
            "todo",
            self.run_config_values.to_properties_iter(None).collect()
        )?;

        ws.read().save_motif_model(&self.channel, &run_config_id, &self.motif_id, &self.vs, &self.params)?;

        Ok(run_config_id)
    }
}

pub fn opt_train<P, R, F, N: Debug + Default>(
    opt_config: &OptConfig, //N node parameters
    mut train: F,
    run_config_source: &OptRunConfigs<Vec<NamedVal<P>>, Vec<NamedRange<P>>, Vec<Hyperparameter<P>>, P>,
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
    info!("running opt train {descr}");

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
                                        opt_config.opt_max_evals,//TODO
                                        opt_config.opt_num_init,
        );

        let best_config = opt.optimize();

        debug!("best_config={:?}", best_config);
        let best_config = opt.optimize();
        trace!("done opt {descr}");
        info!("opt_train: done opt, descr={descr} best_config={best_config:?} min_loss={min_loss} min_var_store={min_var_store:?}");

        // min_var_store.ok_or_else(||ThoosaError::new("opt was not run"))
    });

    min_var_store.ok_or_else(||ThoosaError::new("opt was not run"))
}

pub fn run_train_motif_net(motif_width: u32, run_config: &TrainMotifsNetsRunConfig, device: &Device, mut batch: &mut TensorsDataBatch, vs: &VarStore, listener: &mut dyn TrainListener) -> Result<(f32, MotifNetParams), ThoosaError> {
    let node_params = MotifNetParams::new(motif_width, run_config.num_layers);

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
        run_config.common_training_args.cycles as usize,
        AdamW::default().build(&vs, run_config.common_training_args.lr).unwrap(),
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
    batch_size: u32,
    device:Device
}

impl TensorsDataBatch {

    pub fn new(classes_tensors: Vec<Vec<Box<Tensor>>>, batch_size: u32, train_valid_split:f32, device:Device) -> Self {

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


    fn choose_for_batch(class_tensors: &Vec<Vec<Box<Tensor>>>, batch_size: u32, device:&Device) -> (Tensor, Tensor) {
        let mut rng = thread_rng();

        let per_class_len = batch_size / class_tensors.len() as u32;//TODO make balanced
        debug!("TensorsDataBatch::choose_for_batch: class_tensors={} per_class_len={per_class_len}", class_tensors.len());

        let mut x_y_vecs = class_tensors.iter().enumerate().fold(vec![], |mut x_y_acc, (class, tensors)| {
            assert!(tensors.len() > 0, "no tensors for class={class} in ClassTensorsDataBatch");

            tensors.choose_multiple(&mut rng, per_class_len as usize).for_each(|t| {

                x_y_acc.push((t.shallow_clone(), class as i64));
            });

            x_y_acc
        });

        x_y_vecs.shuffle(&mut rng);

        debug!("TensorsDataBatch::choose_for_batch: x_y_vecs.1.len()={} batch_size={batch_size}", x_y_vecs.len());
        assert!(x_y_vecs.len() > 0);

        //TODO throw if not enough tensors for batch?
        (x_y_vecs.len()..batch_size as usize).for_each(|i|{
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
    channels_motifs_nodes: Arc<RwLock<Vec<(FindMotifsRunConfig, Vec<(MotifNetParams, ThoModuleWithStore)>)>>>,
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
                    find_motifs_cfg.window,
                    max_motif_nodes as u32,
                    slices_len,
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
/// returns batch x channels x motif_nets x slices
fn apply_motif_nets_with_padding(
    input_t: &Tensor,
    channels_motifs_nodes: Arc<RwLock<Vec<(FindMotifsRunConfig, Vec<(MotifNetParams, ThoModuleWithStore)>)>>>,
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
                let slices_len = slices_len(input_n, find_motifs_cfg.window, find_motifs_cfg.step);
                debug!("apply_motif_nets: slices_len={slices_len} input_n={input_n} motif_width={} step={}", find_motifs_cfg.window, find_motifs_cfg.step);
                apply_channel_motifs_with_padding(
                    find_motifs_cfg.step,
                    motif_nodes,
                    input_t.i((.., channel as i64, ..)), //batch, channel, data
                    find_motifs_cfg.window,
                    max_motif_nodes as u32,
                    slices_len,
                    device
                )})
            .collect::<Result<Vec<_>, ThoosaError>>().unwrap().as_slice(),
        1
    );

    // let padded = pad_to(&channels_motifs, &[5, max_motif_nodes as i64, max_motif_batches as i64, 7]).unwrap();
    // debug!("apply_label_motifs: channels_motifs.size={:?} padded.size()={:?}", channels_motifs.size(), padded.size());
    // padded

    debug!("apply_motif_nets: channels_motifs.size={:?}", channels_motifs.size());

    channels_motifs
}

#[derive(Debug)]
pub struct LabelNetApplicable {

    pub motifs_nets_selection_applicable: MotifsNetsSelectionApplicable,
    pub label_net_params:LabelNetParams,
    pub module:ThoModuleWithStore
}

impl LabelNetApplicable {
    pub fn motifs_nets_selection_applicable_options(&self)->NetsPadding {
        match self.label_net_params.net_type {
            LabelNetTypeParams::MaxPadding(_) => {
                NetsPadding::Max
            }
            LabelNetTypeParams::FixedNumSlices(FixedNumSlicesNetParams{ num_layers, nets_count }) => {
                NetsPadding::Fixed(self.label_net_params.slices_len)
            }
        }
    }
}

#[derive(Debug)]
pub struct LabelNetsApplicable {
    label_applicables:HashMap<String, Vec<LabelNetApplicable>>
}

impl LabelNetsApplicable {

    pub fn load_all_filtered_by_run_ids(train_label_nodes_run_ids: Option<&[RunId]>, store: &Arc<Mutex<Box<dyn Store>>>, ws: &Arc<RwLock<Box<dyn Workspace>>>, device: Device) -> LabelNetsApplicable {
        let label_models = ws.read().load_label_models_ws(train_label_nodes_run_ids, &device).expect("load_all_label_models_ws");

        LabelNetsApplicable {
            label_applicables: label_models.into_iter().fold(HashMap::new(), |mut a, (label, label_run_id_nodes)| {
                let label_applicables = label_run_id_nodes.into_iter().map(|(label_run_id, label_net_params, module)| {
                    let motif_selection_run_ids = store
                        .lock()
                        .get_input_run_ids(label_run_id, Some(&SelectMotifsRunConfig::type_name()))
                        .expect("get_input_run_ids(motif_node_params_selection_run_ids")
                        .into_iter()
                        .collect::<HashSet<RunId>>(); //TODO distinct in sql?

                    debug!("load_all_label_and_motif_models: label={label} label_run_id={label_run_id} motif_selection_run_ids={motif_selection_run_ids:?}");

                    assert_eq!(motif_selection_run_ids.len(), 1);

                    LabelNetApplicable {
                        motifs_nets_selection_applicable: MotifsNetsSelectionApplicable::new(motif_selection_run_ids.into_iter().next().unwrap(), store, ws, device),
                        label_net_params,
                        module,
                    }
                }).collect::<Vec<_>>();

                a.insert(label, label_applicables);
                a
            })
        }
    }
}

impl Applicable<&Box<Tensor>, Box<Tensor>, ()> for LabelNetApplicable {
    fn apply(&self, batch: &Box<Tensor>, _: ()) -> Result<Box<Tensor>, ThoosaError> {
        Ok(
            Box::new(self.module.infer(
                //label net accepts batch, motif net - single tensor
                &Tensor::stack(
                    (0..batch.size()[0])
                        .map(|i| self.motifs_nets_selection_applicable.apply(&Box::new(batch.i((i, .., ..))), self.motifs_nets_selection_applicable_options()))
                        .collect::<Result<Vec<_>, _>>()?.as_slice(),
                    0,
                )
            ))
        )
    }
}

impl Applicable<&Box<Tensor>, Box<Tensor>, ()> for LabelNetsApplicable {
    fn apply(&self, subject: &Box<Tensor>, _: ()) -> Result<Box<Tensor>, ThoosaError> {

        Ok(
            Box::new(
                Tensor::stack(
                    self.label_applicables.iter().map(|(label, label_applicable)|{
                        label_applicable[0].apply(subject, ())
                    }).collect::<Result<Vec<_>, _>>()?.as_slice(),

                    1
                )
            )


        )
    }

}


#[derive(Debug)]
pub struct FinalNetApplicable {
    label_nets_applicable: LabelNetsApplicable,
    final_net_params: FinalNetParams,
    final_net_module:ThoModuleWithStore
}

impl FinalNetApplicable {

    pub fn new(final_run_id:RunId, store: &Arc<Mutex<Box<dyn Store>>>, ws: &Arc<RwLock<Box<dyn Workspace>>>, device: Device) -> FinalNetApplicable {

        let train_label_node_run_ids = store.lock().get_input_run_ids(
            final_run_id,
            Some(&TrainLabelNetsRunConfig::type_name())
        ).expect(format!("FinalNetApplicable: final_run_id={final_run_id} not found").as_str());

        let (final_net_params, final_net_module) = load_final_model(final_run_id, ws, &device).expect("load_final_model failed");

        FinalNetApplicable {
            label_nets_applicable: LabelNetsApplicable::load_all_filtered_by_run_ids(Some(&train_label_node_run_ids), store, ws, device),
            final_net_params,
            final_net_module
        }
    }
}

impl Applicable<&Box<Tensor>, Box<Tensor>, ()> for FinalNetApplicable {

    fn apply(&self, subject: &Box<Tensor>, options: ()) -> Result<Box<Tensor>, ThoosaError> {
        assert_eq!(subject.size().len(), 2);//single tensor - channels x data

        Ok(Box::new(self.final_net_module.infer(
            &self.label_nets_applicable.apply(
                &Box::new(subject.unsqueeze(0)),
                options
            )?.view((1,-1))
        )))
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
    pub run_config_cnt:String,//TrainLabelNetsRunConfig
}

impl TrainFinalNetArgs {

    pub fn resource_config(&self) -> NetResources<Logits> {
        NetResources::new(
            Logits::new(self.train_resource_group.clone()), Logits::new(self.valid_resource_group.clone())
        )
    }

}


#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
#[run_config_opt]
pub struct TrainFinalNetRunConfig {
    #[in_map]
    pub id:Option<RunId>,

    #[in_map]
    #[run_config]
    pub common_training_args:CommonTrainingRunConfig,

    #[in_map]
    #[run_config]
    pub net: FinalNetArgs,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
struct FinalNetArgs {
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
    pub fn new(id: Option<RunId>, common_training_args: CommonTrainingRunConfig, net: FinalNetArgs) -> Self {
        Self { id, common_training_args, net }
    }
}





#[derive(Debug, Args, Clone)]
pub struct InferArgs {
    #[arg(long)]
    pub final_run_id: RunId,

    #[arg(long,  default_value_t = 1000)]
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

    pub fn new(final_run_id: RunId, input_data: String, output: String, input_slice_step: usize) -> Self {
        Self { final_run_id, input_slice_step, input_dir: input_data, output_dir: output }
    }
}

#[allow(dead_code)]
fn build_multidim<'a>( builder:&'a mut SeqBuilder, dims:&'a [i64])  {

    let builder = dims.windows(2).into_iter().fold(builder, | b, d|{
        debug!("d={:?}", d);

        b.linear_s(d[0] as u32)
            // .func(Tensor::relu)
            .linear_s(1)
            // .func(Tensor::sigmoid)
            .closure(Box::new(|t:&Tensor|t.squeeze()), Some(d[1] as u32))

    });

    builder.func(Tensor::sigmoid);
}

pub fn is_nonzero(t_1: &Tensor) -> Result<bool, ThoosaError> {
    Ok(t_1.flatten(0, (t_1.size().len() - 1) as i64).iter::<f64>()?.find(|v| v > &0.).is_some())
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



#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize,  Debug, Clone, PartialEq)]
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
        Self { num_layers, slices_len }
    }
}

#[cfg(test)]
mod train_tests {
    use crate::builder::SeqBuilder;
    use crate::loader::{LoadRunner, OfMemSize, SyncWrapper};
    use crate::my_tch::ThoModule;
    use crate::simp::DummyDs;
    use crate::train_nets;
    use crate::train_nets::{apply_channel_motifs, apply_channel_motifs_with_padding, apply_motif_and_label_nets, apply_motif_nets, apply_motif_nets_with_padding, opt_train, BatchTransform, ClassTensorsLoaderData, CommonTrainingRunConfig, CommonTrainingRunConfigRangeOrValue, FinalNetParams, FindMotifsRunConfigRangeOrValue, FixedNumSlicesNetArgs, InputFilterRangeOrValue, LabelNetTypeArgs, RangeOrValue, LoadingTensorsDataBatch, MaxPaddingNetArgs, MaxPaddingNetArgsRangeOrValue, MotifsNetsSelectionApplicable, OptConfig, PlotVecf32, TrainFinalNetRunConfig, TrainLabelNetsRunConfig, TrainLabelNetsRunConfigRangeOrValue, NetsPadding, LabelNetTypeArgsRangeOrValue, MaxPaddingNetParams, FixedNumSlicesNetParams, LabelNetApplicable, LabelNetsApplicable, FinalNetArgs};
    use crate::train_nets::{ InferArgs, LabelNetParams, MotifNetParams, TensorsDataBatch, ThoModuleWithStore};
    use crate::utils::{cuda_mem, from, optimizer_mem_size, set_requires_grad_in_subpath, train, vs_mem_size, SplittableDataBatch, TrainListener};
    use crate::utils::{lin_t, pad_to, reset_print_options, run_split_on_oom, t_0_0, tensor_mem_size, DataBatch, Reducer, SplitQueue, Splittable, SplittableTensor};
    use crate::ws::{mock_workspace_with_motif_nets, FsWorkspace, Workspace};
    use cudarc::driver::CudaDevice;
    use log::LevelFilter::{Debug, Info, Trace};
    use log::{debug, error, info, set_max_level, trace};
    use parking_lot::{Mutex, RwLock};
    use rand::{thread_rng, Rng};
    use rayon::iter::IntoParallelRefMutIterator;
    use std::collections::HashMap;
    use std::{fs, mem};
    use std::ops::Deref;
    use std::path::{Path, PathBuf};
    use std::rc::Rc;
    use std::sync::Arc;
    use mockall::automock;
    use mockall_double::double;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::nn::{init, AdamW, BatchNormConfig, LinearConfig, Module, OptimizerConfig, VarStore};
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::{Float, Half};
    use tch::{nn, Device, Tensor};
    use tch::{IndexOp, Kind, Reduction};
    use torch_sys::at_autocast_clear_cache;
    use thoosa_properties_map::ToHashmap;
    use thoosa_run_config_args::OptRunConfigSource;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::{batch_windows, cdt_for_dir, clc_num_batches, pulse_in_rand, sin_v, slices_len, ThoosaError};
    use thoosa_shared::run_configs::{Applicable, OptRunConfigs, RunId};
    use thoosa_shared::val::Val;
    use thoosa_store::{mock_store_with_find_motifs_run_config, MockStore, SqlStore};
    use crate::find_motifs_run_config::FindMotifsRunConfig;
    use crate::transforms::filters::BandPassFilterParamsRangeOrValue;
    use crate::ws::mock_workspace_simp;

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
    pub fn test_net(num_features: u32, k: f64, device: Device, ) -> Result<Box<dyn ThoModule>, ThoosaError> {
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
    pub fn test_net_vs(num_features: u32, k: f64, vs:&VarStore) -> Result<Box<dyn ThoModule>, ThoosaError> {

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
    fn test_apply_motif_models_padding_max() -> Result<(), ThoosaError>{
        set_max_level(Trace);
        reset_print_options();

        let input_n = 1000;
        let motif_width = 100u32;
        let step = (motif_width / 1) as usize;//1 - step==motif_width slices do not overlap so net output - sum(slice)*k

        //batch 2 x channels 5 x n 1000
        // let batch = Tensor::stack(&[
        let t_0 =    lin_test_tensor(input_n);
        let t_1 =     lin_test_tensor(input_n/2);
        // ], 0);


        let applicable = MotifsNetsSelectionApplicable::new(
            1,
            &Arc::new(Mutex::new(mock_store_with_find_motifs_run_config(step as i32, motif_width as i32, vec![
                vec![(0, 1.0), (1, 1.0)],
                vec![(0, 1.0)]
            ]))),
            &Arc::new(RwLock::new(mock_workspace_with_motif_nets(5, 100, input_n as u32, vec![
                vec![//channel 0
                    (Box::new(move ||MotifNetParams::new(motif_width, 1)), Box::new(move ||ThoModuleWithStore::new(test_net(motif_width, 1., Cpu).unwrap()))),
                    (Box::new(move ||MotifNetParams::new(motif_width, 1)), Box::new(move ||ThoModuleWithStore::new(test_net(motif_width, 2., Cpu).unwrap())))
                ],
                vec![//channel 1
                    (Box::new(move ||MotifNetParams::new(motif_width, 1)), Box::new(move ||ThoModuleWithStore::new(test_net(motif_width, 10., Cpu).unwrap())))
                ]
            ]))),
            Cpu
        );

        let t_0_applied = applicable.apply(&Box::new(t_0), NetsPadding::Max)?;

        debug!("t_0_applied.size={:?}", t_0_applied.size());
        trace!("t_0_0_applied=\n{}", t_0_applied.i((0, ..)));
        trace!("t_0_1_applied=\n{}", t_0_applied.i((1, ..)));

        let channel_0_motifs = Tensor::stack(&[
            //sum 0-100  100-200 ...
            Tensor::from_slice(&[4950., 14950., 24950., 34950., 44950., 54950., 64950., 74950., 84950., 94950.,]),  //ch 0 net 0 (ResourceId=1)
            //sum 0-100  100-200 ... * 2 - k in second net in channel 0/**/
            Tensor::from_slice(&[4950., 14950., 24950., 34950., 44950., 54950., 64950., 74950., 84950., 94950.]) *2. // ch. 0 net 1 (ResourceId=2)
        ], 0).totype(Float).view((1, 2, 10));

        assert_eq!(channel_0_motifs, t_0_applied.i((0, ..)).view((1, 2, 10)));


        let channel_1_motifs = Tensor::stack(&[
            //sum 0-100  100-200 ...
            //sum 0-100  100-200 ... * 10 - k in second net in channel 0/**/
            Tensor::from_slice(&[4950., 14950., 24950., 34950., 44950., 54950., 64950., 74950., 84950., 94950.]) *2. * 10., // ch. 0 net 1 (ResourceId=2) first * 2. because lin_test_tensor multiplies each chanel by its ordinal
            Tensor::from_slice(&[4950., 14950., 24950., 34950., 44950., 54950., 64950., 74950., 84950., 94950.]) *0. // ch. 0 net 1 (ResourceId=2)
        ], 0).totype(Float).view((1, 2, 10));

        assert_eq!(channel_1_motifs, t_0_applied.i((1, ..)).view((1, 2, 10)));

        let t_1_applied = applicable.apply(&Box::new(t_1), NetsPadding::Max)?;

        debug!("t_1_applied.size={:?}", t_1_applied.size());
        trace!("t_1_applied=\n{}", t_1_applied.i((0, ..)));
        trace!("t_1_applied=\n{}", t_1_applied.i((1, ..)));

        let channel_0_motifs = Tensor::stack(&[
            //sum 0-100  100-200 ...
            Tensor::from_slice(&[4950., 14950., 24950., 34950., 44950., 0., 0., 0., 0., 0.,]),  //ch 0 net 0 (ResourceId=1)
            //sum 0-100  100-200 ... * 2 - k in second net in channel 0/**/
            Tensor::from_slice(&[4950., 14950., 24950., 34950., 44950., 0., 0., 0., 0., 0.]) *2. // ch. 0 net 1 (ResourceId=2)
        ], 0).totype(Float).view((1, 2, 10));

        assert_eq!(channel_0_motifs, t_1_applied.i((0, ..)).view((1, 2, 10)));


        let channel_1_motifs = Tensor::stack(&[
            //sum 0-100  100-200 ...
            //sum 0-100  100-200 ... * 10 - k in second net in channel 0/**/
            Tensor::from_slice(&[4950., 14950., 24950., 34950., 44950., 0., 0., 0., 0., 0.]) *2. * 10., // ch. 0 net 1 (ResourceId=2) first * 2. because lin_test_tensor multiplies each chanel by its ordinal
            Tensor::from_slice(&[4950., 14950., 24950., 34950., 44950., 0., 0., 0., 0., 0.]) *0. // ch. 0 net 1 (ResourceId=2)
        ], 0).totype(Float).view((1, 2, 10));

        assert_eq!(channel_1_motifs, t_1_applied.i((1, ..)).view((1, 2, 10)));

        Ok(())
    }

    #[test]
    fn test_apply_motif_models_padding_fixed() -> Result<(), ThoosaError>{
        set_max_level(Trace);
        reset_print_options();

        let input_n = 1000;
        let motif_width = 100u32;
        let step = (motif_width / 1) as usize;//1 - step==motif_width slices do not overlap so net output - sum(slice)*k

        //batch 2 x channels 5 x n 1000
        let t_0 =    lin_test_tensor(input_n);
        let t_1 =     lin_test_tensor(input_n/2);


        let applicable = MotifsNetsSelectionApplicable::new(
            1,
            &Arc::new(Mutex::new(mock_store_with_find_motifs_run_config(step as i32, motif_width as i32, vec![
                vec![(0, 1.0), (1, 1.0)],
                vec![(0, 1.0)]
            ]))),
            &Arc::new(RwLock::new(mock_workspace_with_motif_nets(5, 100, input_n as u32, vec![
                vec![//channel 0
                     (Box::new(move ||MotifNetParams::new(motif_width, 1)), Box::new(move ||ThoModuleWithStore::new(test_net(motif_width, 1., Cpu).unwrap()))),
                     (Box::new(move ||MotifNetParams::new(motif_width, 1)), Box::new(move ||ThoModuleWithStore::new(test_net(motif_width, 2., Cpu).unwrap())))
                ],
                vec![//channel 1
                     (Box::new(move ||MotifNetParams::new(motif_width, 1)), Box::new(move ||ThoModuleWithStore::new(test_net(motif_width, 10., Cpu).unwrap())))
                ]
            ]))),
            Cpu
        );

        let t_0_applied = applicable.apply(&Box::new(t_0), NetsPadding::Fixed(11))?;

        debug!("t_0_applied.size={:?}", t_0_applied.size());
        trace!("t_0_applied=\n{}", t_0_applied);

        let channel_0_motifs = Tensor::from_slice(&[
             4950.,  13950.,  22950.,  31950.,  40950.,  49950.,  58950.,  67950.,  76950.,  85950.,  94950.,
             9900.,  27900.,  45900.,  63900.,  81900.,  99900., 117900., 135900., 153900., 171900., 189900.,
             99000.,  279000.,  459000.,  639000.,  819000.,  999000., 1179000., 1359000., 1539000., 1719000., 1899000.// lin_test_tensor multiplies each chanel by its ordinal + *10 by net
        ]).totype(Float).view((3, 11));

        assert_eq!(channel_0_motifs, *t_0_applied);

        let t_1_applied = applicable.apply(&Box::new(t_1), NetsPadding::Fixed(11))?;

        debug!("t_1_applied.size={:?}", t_1_applied.size());
        trace!("t_1_applied=\n{}", t_1_applied);

        let channel_1_motifs = Tensor::from_slice(&[
            4950.,   8950.,  12950.,  16950.,  20950.,  24950.,  28950.,  32950.,  36950.,  40950.,  44950.,
            9900.,  17900.,  25900.,  33900.,  41900.,  49900.,  57900.,  65900.,  73900.,  81900.,  89900.,
            99000., 179000., 259000., 339000., 419000., 499000., 579000., 659000., 739000., 819000., 899000.
        ]).totype(Float).view((3, 11));

        assert_eq!(channel_1_motifs, *t_1_applied);

        Ok(())
    }

    #[test]
    fn test_apply_label_models_padding_fixed() -> Result<(), ThoosaError> {
        set_max_level(Trace);
        reset_print_options();

        let input_n = 1000;
        let motif_width = 100u32;
        let step = (motif_width / 1) as usize; //1 - step==motif_width slices do not overlap so net output - sum(slice)*k

        //batch 2 x channels 5 x n 1000
        let t_0 = lin_test_tensor(input_n);
        let t_1 = lin_test_tensor(input_n / 2);


        let applicable = LabelNetsApplicable::load_all_filtered_by_run_ids(
            Some(&[1, 2]),
            &Arc::new(Mutex::new(mock_store_with_find_motifs_run_config(step as i32, motif_width as i32, vec![
                vec![(0, 1.0), (1, 1.0)],
                vec![(0, 1.0)]
            ]))),
            &Arc::new(RwLock::new(mock_workspace_with_motif_nets(5, 100, input_n as u32, vec![
                vec![ //channel 0
                      (Box::new(move || MotifNetParams::new(motif_width, 1)), Box::new(move || ThoModuleWithStore::new(test_net(motif_width, 1., Cpu).unwrap()))),
                      (Box::new(move || MotifNetParams::new(motif_width, 1)), Box::new(move || ThoModuleWithStore::new(test_net(motif_width, 2., Cpu).unwrap())))
                ],
                vec![ //channel 1
                      (Box::new(move || MotifNetParams::new(motif_width, 1)), Box::new(move || ThoModuleWithStore::new(test_net(motif_width, 10., Cpu).unwrap())))
                ]
            ]))),
            Cpu);


        Ok(())
    }


    #[test]
    fn slow_test_apply_motif_models() -> Result<(), ThoosaError> {
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
        let unfolded = channel.unfold(0, motif_width as i64, input_slice_step as i64);
        let sums = unfolded.sum_dim_intlist(1, true, Float).view(max_dists_len as i64);//sums of the slices should be what test_net produces
        debug!("tensor_size_in_bytes={:?}M ", tensor_mem_size(&unfolded)/1024/1024);
        trace!("unfolded=\n{unfolded} \nsums=\n{}", sums);

        let mut motif_nets = vec![
            (MotifNetParams::new(motif_width, 1), ThoModuleWithStore::new(test_net(motif_width, 1., Cuda(0))?))
        ];

        let channel_1_t = apply_channel_motifs(&mut motif_nets, channel, motif_width, 1, max_dists_len, &Cuda(0))?.view(max_dists_len as i64);

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
            (FindMotifsRunConfig::new(Some(1), step as u32, motif_width as u32, None, None),
             vec![
                 (MotifNetParams::new(motif_width, 1), ThoModuleWithStore::new(test_net(motif_width, 1., Cpu)?)), (MotifNetParams::new(motif_width, 1), ThoModuleWithStore::new(test_net(motif_width, 1., Cpu)?))
             ]),
            (FindMotifsRunConfig::new(Some(1), step as u32, motif_width as u32, None, None),
             vec![(MotifNetParams::new(motif_width, 1), ThoModuleWithStore::new(test_net(motif_width, 10., Cpu)?))], //channel 1 nets
            ),
            (FindMotifsRunConfig::new(Some(1), step as u32, motif_width as u32, None, None),
             vec![]
            )
        ]));



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
    fn test_save_load_label_model_fixed_num_slices() -> Result<(), ThoosaError>{
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_dir={test_dir:?}");
        let ws = Arc::new(RwLock::new(FsWorkspace::new(test_dir, 3, 1000, 32000)?));

        let params = LabelNetParams::new(3, 19, 2, 100, train_nets::LabelNetTypeParams::FixedNumSlices(FixedNumSlicesNetParams::new(10, 100)));

        let mut vs_0 = VarStore::new(Cpu);
        let net = LabelNetParams::label_net(&params).build_tch_module(&vs_0.root())?;
        vs_0.freeze();

        let _ = net.tho_forward(&Tensor::rand([11, 3 * 19, 100], FLOAT_CPU), false);

        let model_path = ws.write().save_label_model(1, &"test_label", &vs_0, &params)?;

        debug!("model_path={model_path:?} vs_0.variables().len()={}", vs_0.variables().len());


        let loaded_module = ThoModuleWithStore::load(LabelNetParams::label_net(&params), thoosa_shared::replace_extension(&model_path, "safetensors").1.into(), Cpu)?;

        assert!(vs_0.variables().len() > 0);
        assert_eq!(vs_0.variables(),loaded_module.var_store().variables());

        Ok(())
    }

    #[test]
    fn test_save_load_label_model_max_padding() -> Result<(), ThoosaError>{
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_dir={test_dir:?}");
        let channels = 3;
        let max_motif_nodes = 19;
        let slices_len = 23;

        let ws = Arc::new(RwLock::new(FsWorkspace::new(test_dir, channels as u32, 1000, 32000)?));
        let params = LabelNetParams::new(channels, max_motif_nodes, 3, slices_len, train_nets::LabelNetTypeParams::MaxPadding(MaxPaddingNetParams::new(3)));

        let mut vs_0 = VarStore::new(Cpu);
        let net = LabelNetParams::label_net(&params).build_tch_module(&vs_0.root())?;
        vs_0.freeze();

        let batch = 11i64;
        let _ = net.tho_forward(&Tensor::rand([batch, channels as i64, max_motif_nodes as i64, slices_len as i64], FLOAT_CPU), false);


        let model_path = ws.write().save_label_model(1, &"test_label", &vs_0, &params)?;

        debug!("model_path={model_path:?} vs_0.variables().len()={}", vs_0.variables().len());


        let loaded_module = ThoModuleWithStore::load(LabelNetParams::label_net(&params), thoosa_shared::replace_extension(&model_path, "safetensors").1.into(), Cpu)?;

        assert!(vs_0.variables().len() > 0);
        assert_eq!(vs_0.variables(),loaded_module.var_store().variables());

        Ok(())
    }

    #[test]
    fn test_x(){
        let rc = FindMotifsRunConfig::new(Some(1), 100, 100, None, Some(0.1));

        let map:HashMap<String, Val> = rc.to_properties_iter(None).collect();


        debug!("map={map:?}");
    }

    #[test]
    fn test_train_label() -> Result<(), ThoosaError> {
        set_max_level(Debug);
        let ws = Arc::new(RwLock::new(mock_workspace_simp(None, 2, 1000, 1100)));

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
                    dummy_labels_net_input(&TRAIN_LABELS, CHANNELS, MAX_MOTIF_NETS, MAX_SLICES_LEN),
                    Tensor::from_slice(TRAIN_LABELS.as_slice()).totype(Float)
                )
            }

            fn valid_batch(&self) -> (Tensor, Tensor) {
                (
                    dummy_labels_net_input(&TRAIN_LABELS, CHANNELS, MAX_MOTIF_NETS, MAX_SLICES_LEN),
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

        let (loss, params) = super::train_label_net(
            &ws,
            &TrainLabelNetsRunConfig::new(
                Some(1),
                CommonTrainingRunConfig::new(10, 10, 0.1, 0.1, 10, 3),
                LabelNetTypeArgs::MaxPadding(MaxPaddingNetArgs::new(3))
                // LabelNetType::FixedNumSlices(FixedNumSlicesNetParams { num_layers: 1, slices_len: MAX_SLICES_LEN as u32 })
            ),
            &mut DummyLabelBatch::new(),
            &Cpu,
            CHANNELS as u32,
            MAX_MOTIF_NETS as u32, //max_nets
            &VarStore::new(Cpu),
            &Arc::new(MotifsNetsSelectionApplicable::new(
                1,
                &Arc::new(Mutex::new(mock_store_with_find_motifs_run_config(100, 100, vec![vec![(0, 0.0)]]))),
                &ws,
                Cpu
            ))
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
        const FEATURES:u32 = 10;

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

        let ws = Arc::new(RwLock::new(mock_workspace_simp(None, 2, 1000, 1000)));

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
                    dummy_labels_net_input(&TRAIN_LABELS, CHANNELS, MAX_MOTIF_NETS, MAX_SLICES_LEN).to_device(Cuda(0)),
                    Tensor::from_slice(TRAIN_LABELS.as_slice()).totype(Float).to_device(Cuda(0))
                )
            }

            fn valid_batch(&self) -> (Tensor, Tensor) {
                (
                    dummy_labels_net_input(&TRAIN_LABELS, CHANNELS, MAX_MOTIF_NETS, MAX_SLICES_LEN).to_device(Cuda(0)),
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

        let r= opt_train(
            &OptConfig::new(10, 3),
            |vs:&VarStore, run_config:&TrainLabelNetsRunConfig, listener: &mut dyn TrainListener|{
                let mut r = None::<(f32, LabelNetParams)>;
                run_split_on_oom(
                    Reducer::new(batch.clone()),
                    |batch|{
                        super::train_label_net(
                            &ws,
                            run_config,
                            batch,
                            &Cuda(0),
                            CHANNELS as u32,
                            MAX_MOTIF_NETS as u32, //max_nets
                            &VarStore::new(Cuda(0)),
                            &Arc::new(MotifsNetsSelectionApplicable::new(
                                1,
                                &Arc::new(Mutex::new(SqlStore::from_path(ws.read().store_path())?)),
                                &ws,
                                Cpu
                            ))

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
                net: LabelNetTypeArgsRangeOrValue::MaxPadding(MaxPaddingNetArgsRangeOrValue{ num_layers: 3.into() }).into(),
            }.opt_run_configs(),
            Cpu,
            "label"
        );

        debug!("test_train_oom: r={r:?}");


        Ok(())
    }

    fn dummy_labels_net_input(labels:&[bool], channels:usize,  max_motif_nets:usize, slices_len:usize) ->Tensor{
        //batch size x channels x max motif nets x slices_len
        let mut labels_iter = labels.iter();
        let t = Tensor::stack(
            labels.iter().map(|label|
                          Tensor::stack((0..channels).map(|mb|
                                  Tensor::stack((0..max_motif_nets).map(|mmn|
                                      if *label {
                                          motifs_1(slices_len)
                                      } else {
                                          motifs_0(slices_len)
                                      }
                                  ).collect::<Vec<_>>().as_slice(), 0)
                          ).collect::<Vec<_>>().as_slice(), 0)
            ).collect::<Vec<_>>().as_slice(),
            0,
        );
        debug!("dummy_labels_net_input: t.size={:?}", t.size());

        t
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
            &Arc::new(RwLock::new(mock_workspace_simp(None, 0, 1000, 100))),
            &TrainFinalNetRunConfig::new(Some(1), CommonTrainingRunConfig::new(10, 10, 0.1, 0.1, 10, 3), FinalNetArgs::new(3)),
            &mut DummyLabelBatch::new(),
            &Cpu,
            LABELS_SIZE as u32,
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
    fn save_test_models(ws: &Arc<RwLock<Box<dyn Workspace>>>, base_run_id: RunId, channels: u32, max_motif_batches: u32, max_batch_size: u32, motif_node_params_run_id: RunId, window: u32) -> Result<(), ThoosaError> {
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

    fn save_test_final_model(models_path: &PathBuf, features: u32) -> Result<(), ThoosaError> {
        fs::create_dir_all(models_path.clone())?;

        let mut vs_0 = VarStore::new(Cpu);
        let _ = train_nets::final_node(&FinalNetParams::new(features, 1)).build_tch_module(&vs_0.root())?;

        vs_0.freeze();
        debug!("save_test_final_model: features={} vs_0.variables()={:?}", features, vs_0.variables());
        vs_0.save(models_path.join("final.safetensors"))?;
        Ok(())
    }

    fn save_test_label_model(ws: &Arc<RwLock<Box<dyn Workspace>>>, run_id: RunId, label:&str, channels: u32, max_motif_nets: u32, max_motif_batches: u32, motif_batch_size: u32) -> Result<(), ThoosaError> {
        let label_node_params = LabelNetParams::new(channels, max_motif_nets, 0, 0, train_nets::LabelNetTypeParams::MaxPadding(MaxPaddingNetParams { num_layers: 0 }));//TODO slices_len, motif_layers

        let mut vs_0 = VarStore::new(Cpu);
        let _ = LabelNetParams::label_net(&label_node_params).build_tch_module(&vs_0.root())?;
        vs_0.freeze();
        debug!("save_test_corr_model: channels={}, max_motif_nets={}, max_motif_batches={}, motif_batch_size={} vs_0.variables()={:?}", channels, max_motif_nets, max_motif_batches, motif_batch_size, vs_0.variables());

        ws.write().save_label_model(run_id, &label, &vs_0, &label_node_params )?;
        Ok(())
    }

    fn save_test_motif_model(ws: &Arc<RwLock<Box<dyn Workspace>>>, channel: &str, motif_id: u32, window: u32, motif_node_params_run_id: RunId) -> Result<(), ThoosaError> {

        let mut vs_0 = VarStore::new(Cpu);
        let _test_motif_model = train_nets::motif_node(&MotifNetParams::new(window, 1)).build_tch_module(&vs_0.root())?;

        let test_motif_model = SeqBuilder::new(100, None)
            .linear_s_with_conf(1, LinearConfig {
                ws_init: init::Init::Const(1.0),
                bs_init: None,
                bias: false
            })
            .end_seq().build_tch_module(&vs_0.root())?;


        vs_0.freeze();
        debug!("save_test_motif_model: channel={channel} vs_0.variables()={:?}", vs_0.variables());
        ws.write().save_motif_model(&channel, &motif_node_params_run_id, &motif_id, &vs_0, &MotifNetParams::new(100, 1))?;
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
    fn test_loading_tensors_data_batch(){
        let loader = Arc::new(RwLock::new(LoadRunner::new(
            move |key: &Resource| {
                SyncWrapper::new_tb(Box::new({
                    match key.file_stem() {
                        "aaa"=>  {
                            let t_0_0 = Tensor::from_slice(&[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]); //t 0 channel 0
                            let t_0_1 = Tensor::from_slice(&[0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]); //t 0 channel 1
                            let t_0 = Tensor::stack(&[t_0_0, t_0_1], 0);
                            t_0
                        }
                        "bbb"=>  {
                            let t_1_0 = Tensor::from_slice(&[1.0, 1.01, 1.02, 1.03, 1.04, 1.05]);//t 1 channel 0
                            let t_1_1 = Tensor::from_slice(&[1.1, 1.11, 1.12, 1.13, 1.14, 1.15]);//t 1 channel 1
                            let t_1 = Tensor::stack(&[t_1_0, t_1_1], 0);
                            t_1
                        }
                        "ccc"=>  {
                            let t_2_0 = Tensor::from_slice(&[2.0, 2.01, 2.02, 2.03, 2.04, 2.05, 2.06, 2.07]);//t 2 channel 0
                            let t_2_1 = Tensor::from_slice(&[2.1, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16, 2.17]);//t 2 channel 1
                            let t_2 = Tensor::stack(&[t_2_0, t_2_1], 0);
                            t_2
                        }
                        _ => {
                            panic!("unexpected key={key:?}")
                        }
                    }

                }
                ))
            },
            1,
            10,
        )));

        let mut data_batch = LoadingTensorsDataBatch::new(
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
            Some(Box::new(|t|{
                Box::new(t.pad([0,10-t.size()[1]], "constant", 0.))
            }))
        );

        let batch = data_batch.train_batch();


        // debug!("data={}", batch.0);
        // debug!("labels={:?}", batch.1);

        assert_eq!(
            [3_i64,2,10],
            batch.0.size()[..]
        );
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
