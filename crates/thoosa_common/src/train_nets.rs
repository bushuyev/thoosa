use std::collections::HashMap;
use std::fs;
use std::iter::Map;
use std::path::{Path, PathBuf};
use std::sync::{Arc};
use parking_lot::{Mutex, RwLock};

use clap::clap_derive::Args;
use crossbeam_utils::sync::WaitGroup;
use futures::StreamExt;
use log::{debug, info, warn};
use plotters::prelude::BLACK;
use rand::prelude::SliceRandom;
use rand::thread_rng;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tch::{Device, IndexOp, Kind, nn, Reduction, Tensor};
use tch::Kind::Float;
use tch::nn::{ConvConfig, Module, OptimizerConfig, VarStore};
use futures::FutureExt;

use thoosa_shared::{batch_windows, RangeOrValue, ThoosaError};
use thoosa_shared::parse_key_val;
use thoosa_shared::val::unprefix_properties;
use thoosa_shared::val::Val;
use thoosa_shared::parse_range_or_value;

use crate::builder::{Node, SeqBuilder};
use crate::loader::{LoadRunner, Loader};
use crate::my_tch::apply_channel_motifs;
use crate::parzen::optimizer::{ConfigurationSpace, Hyperparameter};
use crate::train_nets;
use crate::utils::{DataBatch, train, plot_series, TrainListener, ListenerResponse};
use crate::utils::from;
use crate::ws::Workspace;

enum BinaryLabel {

}

#[derive(Debug, Args, Clone)]
pub struct TrainOnLabelParams {

    #[arg(long,  default_value_t = 1000)]
    step: usize,


    #[arg(long)]
    train_resource_group: Option<String>,

    #[arg(long)]
    valid_resource_group: Option<String>,


    #[arg(long, value_parser=parse_key_val::<String, String>)]
    train_labels_datasets: Vec<(String, String)>,

    #[arg(long, value_parser=parse_key_val::<String, String>)]
    valid_labels_datasets: Vec<(String, String)>,


    #[arg(long, default_value_t = 10)]
    cycles: usize,

    #[arg(long, default_value_t = 1.0e-5)]
    lr: f64,

    #[arg(long, default_value_t = 10)]
    motif_batch_size:i64,

    #[arg(long, default_value_t = 5)]
    label_batch_size:usize
}

impl TrainOnLabelParams {

    pub fn cycles(&self) -> usize {
        self.cycles
    }
    pub fn lr(&self) -> f64 {
        self.lr
    }

    #[allow(dead_code)]
    pub fn dummy(step: usize, _motif_models_dir:String, _models_dir:String, train_labels_datasets: Vec<(String, String)>, valid_labels_datasets: Vec<(String, String)>, motif_batch_size: i64, items_per_label: usize, cycles: usize, lr: f64) ->Self {
        TrainOnLabelParams {
            step,
            cycles,
            lr,
            train_labels_datasets,
            valid_labels_datasets,
            train_resource_group: None,
            valid_resource_group: None,
            motif_batch_size,
            label_batch_size: items_per_label
        }
    }

    pub fn train_labels_datasets(&self) -> &Vec<(String, String)> {
        &self.train_labels_datasets
    }

    pub fn valid_labels_datasets(&self) -> &Vec<(String, String)> {
        &self.valid_labels_datasets
    }

    pub fn with_labels_datasets(&self, train_labels_datasets: Vec<(String, String)>, valid_labels_datasets: Vec<(String, String)>) -> Self {
        let mut clone = self.clone();
        clone.train_labels_datasets = train_labels_datasets;
        clone.valid_labels_datasets = valid_labels_datasets;

        clone
    }

    pub fn train_resource_group(&self) -> Option<&String> {
        self.train_resource_group.as_ref()
    }

    pub fn valid_resource_group(&self) -> Option<&String> {
        self.valid_resource_group.as_ref()
    }

    pub fn motif_batch_size(&self) -> i64 {
        self.motif_batch_size
    }


    pub fn step(&self) -> usize {
        self.step
    }

    pub fn dsp_filter(&self) ->Box<impl Fn(Tensor)->Tensor> {
        Box::new(|t:Tensor|t)
    }
}

/**
* returns num labels * items_per_label x channels x motifs x motifs batch size
*/
struct LoadingTensorsDataBatch {
    train: Vec<(i64, Vec<String>)>,
    val: Vec<(i64, Vec<String>)>,
    batch_size: usize,
    executor: Arc<parking_lot::RwLock<LoadRunner<String, Box<Tensor>>>>,
    device:Device,
    transform_fn:Option<Arc<RwLock<Box<dyn Fn(Box<Tensor>)->Box<Tensor> + Sync + Send >>>>
}

impl LoadingTensorsDataBatch {

    pub fn new(
        train: Vec<(i64, Vec<String>)>,
        val: Vec<(i64, Vec<String>)>,
        batch_size: usize,
        executor: Arc<parking_lot::RwLock<LoadRunner<String, Box<Tensor>>>>,
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

    fn choose_for_batch(class_resources: &Vec<(i64, Vec<String>)>, batch_size: usize, executor: Arc<parking_lot::RwLock<LoadRunner<String, Box<Tensor>>>>, device: &Device, transform_fn:Option<Arc<RwLock<Box<dyn Fn(Box<Tensor>)->Box<Tensor> + Sync + Send >>>>) -> (Tensor, Tensor) {
        let mut rng = thread_rng();

        let per_class_len = batch_size / class_resources.len();
        debug!("LoadingTensorsDataBatch::to_x_y: class_tensors={} per_class_len={per_class_len}", class_resources.len());


        let wg = WaitGroup::new();

        let mut x_y_vecs_mx = class_resources.iter().fold(Arc::new(Mutex::new(vec![])), |mut x_y_acc, (class, resources)| {
            assert!(resources.len() > 0, "no tensors for class={class} in ClassTensorsDataBatch");
            let class = *class;
            resources.choose_multiple(&mut rng, per_class_len).map(|key|key.clone()).for_each(|resource_id| {

                let x_y_acc_in_exec = Arc::clone(&x_y_acc);
                let wg_exec = wg.clone();
                let transform_fn = transform_fn.as_ref().map(|v|Arc::clone(v));

                executor.write().exec(move |loader, _keys_purge| {
                    async move {

                        let t =  Loader::<String, Box<Tensor>>::load_future(Arc::clone(&loader), resource_id, 0).await;
                        let t = if let Some(transform) = transform_fn{
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
        x_y_vecs.shuffle(&mut rng);

        debug!("LoadingTensorsDataBatch::choose_for_batch: x_y_vecs.1.len()={} batch_size={batch_size}", x_y_vecs.len());

        //TODO throw if not enough tensors for batch?
        (x_y_vecs.len()..batch_size).for_each(|i|{
            x_y_vecs.push(
                (
                    Box::new(x_y_vecs[i- x_y_vecs.len()].0.shallow_clone()),
                    x_y_vecs[i- x_y_vecs.len()].1
                )
            );

        });

        let classes = Tensor::from_slice(x_y_vecs.iter().map(|e| e.1).collect::<Vec<_>>().as_slice()).to_device(*device);

        let tensor = Tensor::stack(
            x_y_vecs.into_iter().map(|e|e.0).collect::<Vec<_>>().as_slice(),
            0,
        );



        debug!("LoadingTensorsDataBatch::choose_for_batch: tensor.size()={:?} tensor.device()={:?}", tensor.size(), tensor.device());

        (
            tensor.totype(Float),
            classes.totype(Float)
        )
    }
}

impl  DataBatch for LoadingTensorsDataBatch {

    fn train_batch(&mut self) -> (Tensor, Tensor){
        Self::choose_for_batch(&self.train, self.batch_size, Arc::clone(&self.executor), &self.device, self.transform_fn.clone())
    }

    fn valid_batch(&mut self) -> (Tensor, Tensor){
        Self::choose_for_batch(&self.val, self.batch_size, Arc::clone(&self.executor), &self.device, self.transform_fn.clone()) //TODO replacement, just pull next
    }
}


struct ClassTensorsLoaderData<I> {
    train_loader: I,
    valid_loader: I,
}

impl <I> ClassTensorsLoaderData<I> where I:Iterator<Item=Result<(Tensor, Tensor), ThoosaError>> + Send + Sync {

    pub fn new(train_loader: I, valid_loader: I) -> Self {
        Self {
            train_loader,
            valid_loader,
        }
    }

}


impl <I> DataBatch for ClassTensorsLoaderData<I> where I:Iterator<Item=Result<(Tensor, Tensor), ThoosaError>> + Send + Sync {

    fn train_batch(&mut self) -> (Tensor, Tensor){
        Self::batch(&mut self.train_loader)
    }

    fn valid_batch(&mut self) -> (Tensor, Tensor){
        Self::batch(&mut self.valid_loader) //no replacement, just pull next
    }
}

impl<I> ClassTensorsLoaderData<I> where I: Iterator<Item=Result<(Tensor, Tensor), ThoosaError>> + Send + Sync {

    fn batch(i: &mut I) -> (Tensor, Tensor) {
        debug!("ClassTensorsLoaderData.batch:");

        let t = i.next().expect("next").expect("result");//TODO expects

        (
            t.0,
            t.1
        )
    }
}

#[allow(dead_code)]
pub fn train_on_label(
    ws: Arc<RwLock<Workspace>>,
    motif_node_params:&MotifNodeParams,
    label_node_params:&LabelNodeParams,
    train_on_label_params: &TrainOnLabelParams,
    train_labels_loaders:Vec<(i64, Vec<String>)>,
    valid_labels_loaders: Vec<(i64, Vec<String>)>,
    channels: i64,
    device: &Device,
    executor: Arc<RwLock<LoadRunner<String, Box<Tensor>>>>,
    channels_motifs_models:Arc<RwLock<Vec<Vec<SyncModule>>>>
) -> Result<VarStore, ThoosaError>
    {


    let vs_0 = nn::VarStore::new(device.clone());


    // let apply_motifs = |t:Box<Tensor>| Box::new(apply_label_motifs(motif_node_params.num_features, step, &t, &motif_nets, label_node_params.motif_batch_size, label_node_params.max_motif_batches as usize, device));
    //TODO pass params to LoadingTensorsDataBatch ?
    let num_features = motif_node_params.num_features;
    let motif_batch_size = label_node_params.motif_batch_size;
    let max_motif_batches = label_node_params.max_motif_batches as usize;
    let device = device.clone();
        let step = train_on_label_params.step();
    train(
        &device,
        &mut LoadingTensorsDataBatch::new(

            train_labels_loaders,
            valid_labels_loaders,

            train_on_label_params.label_batch_size,
            Arc::clone(&executor),
            device.clone(),
            Some( Box::new(move |t:Box<Tensor>| Box::new(apply_label_motifs(num_features, step, &t, Arc::clone(&channels_motifs_models), motif_batch_size, max_motif_batches, &device))))

        ) as &mut dyn DataBatch,
        &label_node(&label_node_params).build_tch_module(&vs_0.root())?,
        |pred: &Tensor, target: &Tensor| {
            debug!("train_on_label: pred.size={:?}", pred.size());
            let pred_y = pred.squeeze();
            //pred.i((.., 0, 0));
            debug!("train_on_label: pred_y={:?} target={:?} kind={:?}", pred_y.size(), target.size(), target.kind());
            pred_y.binary_cross_entropy::<Tensor>(&target, None, Reduction::Mean)
        },
        train_on_label_params.cycles,
        nn::AdamW::default().build(&vs_0, train_on_label_params.lr)?,

        |t: Tensor| {
            debug!("train_on_label: t.size()={:?} ", t.size());
            //merge max_motif_batches * motif_batch_size
            let t_out = view_for_label_model(t, channels, label_node_params.max_motif_nets, motif_batch_size, label_node_params.max_motif_batches as usize); //.view((t.size()[0], -1));
            //t.size()[0] - batch size
            debug!("train_on_label: t_out.size()={:?}",  t_out.size());
            t_out
        },
        None,
    )?;

    Ok(vs_0)
}

//t - input tensors batch size x channels x nets x motif batches x motif match size
fn view_for_label_model(t: Tensor, channels: i64, max_nets: i64, motif_batch_size: i64, max_motif_batches: usize) -> Tensor {
    assert!(t.size().len()==5, "view_for_label_model: t.size()={:?}, should be 5 dims", t.size());
    debug!("view_for_label_model: t.size()={:?} channels={}, max_nets={}, max_motif_batches as i64 * motif_batch_size={}", t.size(), channels, max_nets, max_motif_batches as i64 * motif_batch_size);
    t.view([-1/*items_per_label as i64 * 2*/, channels, max_nets, max_motif_batches as i64 * motif_batch_size])
}

fn apply_motifs_to_loaders<I, F>(
    mut labels_loaders: HashMap<i64, Vec<I>>,
    labels: &mut Vec<i64>,
    apply_motifs:F
) -> Vec<Vec<Map<I, F>>> where I: Iterator<Item=Box<Tensor>> + Send + Sync, F: Fn(Box<Tensor>) -> Box<Tensor> + Copy {

    labels.into_iter()
        .map(
            |label| labels_loaders
                .remove(&label).expect(format!("no {}", label).as_str())
                .into_iter()
                .map( |l| l.map(apply_motifs))
                .collect::<Vec<_>>())
        .collect::<Vec<Vec<_>>>()
}

pub fn train_final<I>(_ws_dir: impl AsRef<Path>, params: &TrainFinalParams, train_tensors_loader:I, valid_tensors_loader: I, step: usize, motif_batch_size: i64, channels: i64, max_input:usize, device: &Device, labels: Vec<String>) -> Result<VarStore, ThoosaError>
    where I:Iterator<Item=(Box<Tensor>, Box<Tensor>)> + Send + Sync {
/*
    // let models = Path::new(ws_dir.as_ref()).join("label_models").iter().flat_map(|dir| {
    //     load_label_and_motif_models(dir.as_str(), channels as usize, device).expect(format!("load_corr_and_motif_models: {}", dir).as_str())
    // }).collect::<Vec<LabelModels>>();

    let models: Vec<LabelModels> = vec![];


    let labels_models = models.into_iter().fold(HashMap::new(), |mut a, corr_models| {
        a.entry(corr_models.label.clone()).or_insert_with(||vec![]).push(corr_models);
        a
    });

    debug!("train_final: models.size={:?} models={:?}", labels_models.len(), labels_models);

    let apply_final_nets_f =  |t:(Box<Tensor>, Box<Tensor>)| {
        debug!("train_final: tensors_loader.map: t.0.size={:?} t.1.size={:?}", t.0.size(), t.1.size());

        let apply_nets = |label| {
            match labels_models.get(label) {
                Some(label_models) => {//label_models in one of the model dirs
                    Tensor::stack(label_models.iter().map(|model| {

                        let motif_batches = batch_windows(max_input, model.motif_node_params.num_features as usize, step, motif_batch_size as usize);
                        let max_motif_batches = motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0};

                        debug!("train_final: model.motif_node_params={:?} motif_batches={}", model.motif_node_params, max_motif_batches);

                        let t = apply_label_motifs(model.motif_node_params.num_features, step, &t.0, &model.motif_models, motif_batch_size, max_motif_batches, device);

                        //TODO move into CorrModels
                        let max_nets =  model.motif_models.iter().map(|nets|nets.len()).max().unwrap_or(0) as i64;
                        debug!("train_final: motif models model.label={} t.size()={:?}", model.label, t.size());
                        debug!("train_final: t.size()={:?} ", t.size());

                        let t_x = view_for_label_model(t.reshape([1, channels, max_nets, motif_batch_size, max_motif_batches as i64]), channels, max_nets, motif_batch_size, max_motif_batches);
                        debug!("train_final: t_x.size()={:?} max_nets={}", t_x.size(), max_nets);

                        let t = model.label_model.lock().forward(&t_x);
                        debug!("train_final: corr_model out t.size()={:?}", t.size());
                        t
                    }).collect::<Vec<Tensor>>().as_slice(), 0)
                }
                None => {
                    debug!("train_final: no model for label {:?}", label);

                    Tensor::zeros(&[params.models_dir.len() as i64, 1, 1], (Float, *device))
                }
            }
        };
        Ok(
            (
                Tensor::stack(labels.iter().map(apply_nets).collect::<Vec<Tensor>>().as_slice(), 0),
                t.1.shallow_clone()
            )
        )
    };

    let vs_0 = nn::VarStore::new(device.clone());

    train(
        device,
        &mut ClassTensorsLoaderData::new(
            train_tensors_loader.map(apply_final_nets_f),
            valid_tensors_loader.map(apply_final_nets_f)
        ) as &mut dyn DataBatch,
        &final_node(&FinalNodeParams::new(params.models_dir.len() as i64* 6, 6)).build_tch_module(&vs_0.root())?,
        |pred: &Tensor, target: &Tensor| {

            let pred_y = pred; //pred.i((.., 0, 0));
            debug!("pred_y.size ={:?} target.size={:?}", pred_y.size(), target.size());
            debug!("xxxxx: target={:?}", from::<Vec<f32>>(&target));
            debug!("xxxxx: pred_y={:?} ", from::<Vec<f32>>(&pred_y));
            pred_y.cross_entropy_loss::<Tensor>(&target, None, Reduction::Mean, -100, 0.0)

        },
        params.cycles,
        nn::AdamW::default().build(&vs_0, params.lr)?,
        |t:Tensor|{
            debug!("t.size()={:?} ", t.size());

            debug!("xxxxx: 0 ={:?}",  from::<Vec<f32>>(&t.i((.., 0)).squeeze()));
            // debug!("xxxxx: 1 ={:?}",  from::<Vec<f32>>(&t.i((.., 1)).squeeze()));
            // debug!("xxxxx: 2 ={:?}",  from::<Vec<f32>>(&t.i((.., 2)).squeeze()));

            let t_out = t.view(t.size()[0] * t.size()[1]); //model_dirs * labels
            debug!(" t_out.size()={:?}",  t_out.size());
            t_out
        },
        None
    )?;

    Ok(vs_0)*/

    todo!()
}

pub fn infer<F>(params: &InferParams, path_to_tensor:F, motif_n: i64, step: usize, motif_batch_size: i64, channels: i64, max_input:usize, device: &Device, labels: Vec<String>, motif_node_params: MotifNodeParams) -> Result<(Vec<String>, Tensor), ThoosaError>
    where F:Fn(&Path) -> Result<Tensor, ThoosaError> + Sync{


   /* let models = load_label_and_motif_models(params.models_dir.as_str(), channels as usize, device, motif_node_params)?
        .into_iter().fold(HashMap::new(), |mut a, v|{
        a.insert(v.label.clone(), v);
        a
    });

    let mut vs = nn::VarStore::new(*device);
    let final_model = Mutex::new(train_nets::final_node(&FinalNodeParams::new(6, 6)).build_tch_module(&vs.root())?);
    vs.load(params.final_model.as_ref().unwrap_or(&Path::new(&params.models_dir).join("final.safetensors").to_str().unwrap().to_string()))?;
    vs.freeze();

    debug!("infer: params.input_data={:?} models={:?}", params.input_data, models);
    debug!("infer: labels={:?}", labels);

    let tensors_ids = fs::read_dir(&params.input_data)?.par_bridge().map(|dir_entry| {
        let dir_entry = dir_entry.expect("file: ");

        let input_t = path_to_tensor(dir_entry.path().as_path()).expect(format!("can't load{dir_entry:?}").as_str());
        debug!("infer: t.size={:?}", input_t.size());

        let t = Tensor::stack(labels.iter().map(|label|{
            match models.get(label) {
                Some(model) => {
                    let motif_batches = batch_windows(max_input, model.motif_node_params.num_features as usize, step, motif_batch_size as usize);
                    let max_motif_batches = motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0};
                    let max_nets =  model.motif_models.iter().map(|nets|nets.len()).max().unwrap_or(0) as i64;
                    debug!("infer: channels={:?}, max_motif_batches={:?} motif_batch_size={:?} features={:?} motif_n={:?}", channels, max_motif_batches, motif_batch_size, channels *  max_motif_batches as i64 * motif_batch_size, motif_n);


                    let t = apply_label_motifs(model.motif_node_params.num_features, step, &input_t, &model.motif_models, motif_batch_size, max_motif_batches, device);
                    let t_x = view_for_label_model(t.reshape([1, channels, max_nets, motif_batch_size, max_motif_batches as i64]), channels, max_nets, motif_batch_size, max_motif_batches);
                    // debug!("train_final: t_x.size()={:?} max_nets={}", t_x.size(), max_nets);
                    debug!("infer: motif models model={:?} model.label={} t.size()={:?} t_x.size()={:?}", model, model.label, t.size(), t_x.size());

                    let t = model.label_model.lock().forward(&t_x);
                    debug!("infer: label_model out .size()={:?}", t.size());
                    t.view(1)
                }
                None => {
                    Tensor::zeros(&[1], (Float, *device))
                }
            }

        }).collect::<Vec<Tensor>>().as_slice(), 0);

        debug!("infer: t.size()={:?}", t.size());

        let pred = final_model.lock().forward(&t.transpose(0, 1)); //for single eeg

        let pred= pred.softmax(1, Float);
        debug!("infer: pred={:?}", pred.size());
        // debug!("infer: pred={:?}", from::<Vec<f32>>(&pred));
        (
            dir_entry.path().file_stem().unwrap().to_str().unwrap().to_string(),
            pred
        )
    }).collect::<Vec<(String, Tensor)>>();

    let (ids, tensors) = tensors_ids.into_iter().fold((vec![], vec![]), |mut a, v|{
        a.0.push(v.0);
        a.1.push(v.1);
        a
    });
    Ok((ids, Tensor::vstack(tensors.as_slice())))*/
    todo!()
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
/**
* features without x motif nets
*/
fn load_label_and_motif_models(motif_models_path: impl AsRef<Path>, channels: usize, device: &Device, motif_node_params: MotifNodeParams) -> Result<Vec<LabelModels>, ThoosaError> {
    let re = Regex::new(r"^(?<label>[a-zA-Z0-9]+?)_(?<corr_run>\d+).safetensors").unwrap();//TODO lazy_static

    debug!("load_corr_and_motif_models: motif_models_path={:?}", motif_models_path.as_ref());

    let mut models = vec![];
    for e in fs::read_dir(motif_models_path.as_ref())? {
        let entry = e?;

        if entry.file_type()?.is_file() {
            let file_name = entry.file_name();
            let file_name = file_name.to_str().expect(format!("to_str {:?}", file_name).as_str());
            debug!("load_label_and_motif_models: file_name={:?}", file_name);

            let Some(caps) = re.captures(file_name) else {
                warn!("load_label_and_motif_models: unexpected file {:?} in models {:?}", file_name, motif_models_path.as_ref());
                continue
            };

            let label = caps["label"].parse::<String>()?;
            let corr_run = caps["corr_run"].parse::<String>()?;
            debug!("load_label_and_motif_models: label={} corr_run={}", label, corr_run);



            let channels_motifs_models = load_motif_models(motif_models_path.as_ref(), label.clone(), channels, &device, &motif_node_params)?;
            assert!(channels_motifs_models.len() > 0);
            let _max_nets =  channels_motifs_models.iter().map(|nets|nets.len()).max().unwrap_or(0) as i64;

            let label_node_params = serde_json::from_str::<LabelNodeParams>(
                fs::read_to_string(&motif_models_path.as_ref().join(format!("{}_{}.json", label, corr_run)))?.as_str()
            )?;

            let mut vs = nn::VarStore::new(device.clone());
            let corr_model = train_nets::label_node(&label_node_params).build_tch_module(&vs.root())?;
            debug!("load_label_and_motif_models: entry={:?}", entry);
            vs.load(entry.path())?;
            vs.freeze();

            models.push(crate::train_nets::LabelModels::new(label, corr_model, channels_motifs_models, motif_node_params.clone(), label_node_params))
        }

    }
    //TODO should be sorted as labels and same len
    models.sort_by(|a, b|a.label.cmp(&b.label));

    Ok(models)
}

#[derive(Debug)]
pub struct SyncModule {
    module:Box<dyn Module>
}

unsafe impl Sync for SyncModule {}

impl SyncModule {
    pub fn new(module: Box<dyn Module>) -> Self {
        Self { module }
    }
}

impl Module for SyncModule {
    fn forward(&self, t: &Tensor) -> Tensor {
        self.module.forward(t)
    }
}

pub fn load_motif_models_ws(ws:&Workspace, channels:usize, device: &Device) -> Result<Vec<Vec<SyncModule>>, ThoosaError> {
    let channel_motif_models_paths = ws.motif_models(channels)?;

    channel_motif_models_paths.into_iter().map(|paths|{
        paths.into_iter().map(|(model, config)|{
            let mut vs = nn::VarStore::new(device.clone());
            let node = SyncModule::new(motif_node(&MotifNodeParams::from_file(&config)?).build_tch_module(&vs.root())?);
            vs.load(&model)?;
            vs.freeze();

            Ok(node)
        }).collect()
    }).collect()

}

//TODO remove
pub fn load_motif_models(motif_models_path: impl AsRef<Path>, label:String, channels:usize, device: &Device, motif_node_params: &MotifNodeParams) -> Result<Vec<Vec<Box<dyn Module>>>, ThoosaError> {
    debug!("load_motif_models: motif_models_path={:?} label={label:?} motif_node_params={motif_node_params:?}", motif_models_path.as_ref());

    let mut channels_motifs_models = vec![];

    let corr_run_motif_models_path = motif_models_path.as_ref().join(label); //format!("corr_run_{}", corr_run));

    for channel in 0..channels {

        let motif_path = corr_run_motif_models_path.join(channel.to_string());
        let mut channels_motifs:Vec<Box<dyn Module>> = vec![];

        debug!("load_motif_models: motif_path={:?}, motif_path.is_dir()={:?}", motif_path, motif_path.is_dir());

        if motif_path.is_dir() {
            let mut channel_motifs = fs::read_dir(&motif_path)?
                .map(|res| res.map(|e| e.file_name()))
                .collect::<Result<Vec<_>, std::io::Error>>()?;

            channel_motifs.sort();

            debug!("load_motif_models: channel_motifs={:?}", channel_motifs);

            for motif in channel_motifs {
                debug!("load_motif_models: motif={:?}", motif);

                let mut vs = nn::VarStore::new(device.clone());
                let node = motif_node(&motif_node_params).build_tch_module(&vs.root())?;//TODO window
                vs.load(&motif_path.join(motif))?;
                vs.freeze();
                debug!("load_motif_models: len={:?}", vs.variables().len());
                for (key, t) in  vs.variables() {
                    debug!("load_motif_models: key={:?} kind={:?}", key, t.kind());
                }

                channels_motifs.push(node);
            }
        }

        channels_motifs_models.push(channels_motifs);
    }
    debug!("load_motif_models: channels_motifs_models sizes={:?}", channels_motifs_models.iter().map(|n|n.len()).collect::<Vec<usize>>());

    Ok(channels_motifs_models)

}

#[derive(Debug, Args, Clone)]
pub struct TrainOnMotifsParams {
    #[arg(long)]
    pub motif_ids: MotifsBy,

    #[arg(long)]
    pub motif_selection_ids: Vec<usize>,

    #[arg(long, default_value_t = 1000)]
    pub cycles: usize,

    #[arg(long, default_value_t = 10)]
    pub batch_size: usize,

    #[arg(long, value_parser=parse_range_or_value::<f64>, default_value_t = 1.0e-5.into())]
    pub lr: RangeOrValue<f64>,

    #[arg(long, value_parser=parse_range_or_value::<i64>, default_value_t = 1i64.into())]
    pub num_layers: RangeOrValue<i64>,

    #[arg(long, default_value_t = 0.2)]
    pub max_loss: f32,

    #[arg(long, default_value_t = 0.05)]
    pub min_corr: f64,
}

impl TrainOnMotifsParams {

    pub fn dummy(corr_run_ids:Vec<usize>, cycles: usize, max_loss: f32, batch_size:usize, lr: impl Into<RangeOrValue<f64>>, num_layers:impl Into<RangeOrValue<i64>>) ->Self{

        let lr = lr.into();

        if let RangeOrValue::Range(r) = &lr{
            assert!(r.end > r.start);
        }

        TrainOnMotifsParams {
            motif_ids: MotifsBy::CorrRunId(0),

            motif_selection_ids: corr_run_ids,

            cycles,

            lr,

            num_layers: num_layers.into(),

            max_loss,

            batch_size,

            min_corr: 0.05,
        }
    }

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

    pub fn new(num_features: i64) -> Self {
        Self {
            num_features,
            num_layers: 1
        }
    }

    pub fn to_properties_map(&self) -> HashMap<String, Val> {
        HashMap::from([
            ("num_features".into(),  self.num_features.into()),
        ])
    }

    pub fn from_properties_map(properties:HashMap<String, Val>) -> Self {

        MotifNodeParams::new(
            properties.get("num_features").expect(format !("no num_features in {:?}", properties).as_str()).as_i64().expect("num_features as_i64"),
        )
    }

    pub fn from_file(path:&PathBuf) -> Result<Self, ThoosaError> {
        serde_json::from_str::<MotifNodeParams>(
            fs::read_to_string(&path)?.as_str()
        ).map_err(|serde_err|ThoosaError::new(serde_err.to_string()))
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
        .linear_s(100)
        .func(Tensor::relu);


        for i in 0..params.num_layers as usize {
            net = net.linear_s(100)
                .func(Tensor::relu);
        }


    net
        // .linear_s(180)
        // .func(Tensor::relu)
        .linear_s(1)
        .func(Tensor::sigmoid)
        .end_seq()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LabelNodeParams {
    channels:i64,
    max_motif_nets:i64,
    max_motif_batches:i64,
    motif_batch_size:i64
}

impl LabelNodeParams {
    pub fn new(channels: i64, max_motif_nets: i64, max_motif_batches: i64, motif_batch_size: i64) -> Self {
        Self { channels, max_motif_nets, max_motif_batches, motif_batch_size }
    }

    pub fn to_properties_map(&self) -> HashMap<String, Val> {
        let map = HashMap::from([
            ("LabelNodeParams.channels".into(),  self.channels.into()),
            ("LabelNodeParams.max_motif_nets".into(),  self.max_motif_nets.into()),
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
}

pub fn label_node(params:&LabelNodeParams) -> Node {
    debug!("corr_full_node: params={:?}", params);
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
    let max_motif_nets=  params.max_motif_nets;
    SeqBuilder::new(params.max_motif_batches as i64 * params.motif_batch_size)
        // .batch_norm()
        .closure(Box::new(move |t:&Tensor|{
            //batch x max_motif_nets
            debug!("000.  t.size()={:?}", t.size());

            let t_0 = t.shallow_clone();

            t_0

        }), None)
        .split(
            (0..params.max_motif_nets).map( |_i|
                SeqBuilder::new(params.max_motif_batches as i64 * params.motif_batch_size)
                    .closure(Box::new(move |t:&Tensor|{
                        debug!(">>>>>>>>>>>>>> 0 {:?} t.is_nonzero()={}", t.size(), is_nonzero(&t).unwrap());
                        // let t = t.view((-1, channels, max_motif_batches as i64 * motif_batch_size));
                        // debug!(">>>>>>>>>>>>>> 0 {:?}", t.size());
                        t.shallow_clone()
                    }), None)
                    .conv_1d(10, params.channels, params.channels, ConvConfig::default())
                    .max_pool(10, 1, 0)
                    // .func(Tensor::relu)
                    // .linear_s(channels)
                    // .func(Tensor::sigmoid)
                    // .closure(Box::new(|t:&Tensor|{
                    //     debug!(">>>>>>>>>>>>>> 1 {:?}", t.size());
                    //     t.squeeze()
                    // }), Some(channels))
                    // .func(Tensor::relu)
                    // .linear_s(1)
                    // .func(Tensor::sigmoid)
                    // .closure(Box::new(|t:&Tensor|{
                    //     debug!(">>>>>>>>>>>>>> 2 {:?}", t.size());
                    //     t.squeeze()
                    // }), Some(channels))

                    .func(Tensor::relu)
                    .linear_f(0.5)
                    .linear_f(0.5)
                    .linear_s(1)
                    .func(Tensor::sigmoid)
                    .closure(Box::new(|t:&Tensor|{
                        debug!(">>>>>>>>>>>>>> 3 {:?}", t.size());
                        // t.view((-1, 1))
                        t.shallow_clone()
                    }), Some(1))

                    .end_seq()
            ).collect(),

            Box::new(
                move |t:&Tensor|  (0..max_motif_nets)
                // .zip((0..channels))
                // .map(|(net_i, channel_id)|t.i((.., channelmehrzweckzeitreihenanalysegerät_id, net_i, ..)))
                .map(|net_i|{
                    debug!(">>>>>>>>>>>>>> 3 net_i={:?}", net_i);
                    t.i((.., .., net_i, ..))
                })
                // .filter(|t|{
                //     let nonzero = is_nonzero(&t).unwrap();
                //     if !nonzero {
                //         info!("zero filter");
                //     }
                //     nonzero
                // })
                .collect()
            ),//batch, channel, max_motif_nets, max motif_net batches x motif_net batch size

            Box::new(move |t:Vec<Tensor>|Tensor::stack(t.as_slice(), 1).mean_dim(1, false, Float))

        )
        .closure(Box::new(move |t:&Tensor|{
            //batch x max_motif_nets
            debug!("0. corr_full_node: flatten t.size()={:?}", t.size());
            // let t_size = t.size();

            // assert!(t_size.len()==4, "t_size.len() {}, not 4", t_size.len());
            // assert!(t_size[1] == max_motif_nets, "unexpected t_size[1]={}, should be max_motif_nets={}", t_size[1], max_motif_nets);
            // assert!(t_size[2] == 1, "unexpected t_size[2]={}, should be 1", t_size[2]); //.conv_1d - channels -> 1
            // assert!(t_size[3] == 1, "unexpected t_size[3]={}, should be 1", t_size[2]);
            // let t_0 = t.squeeze_dims(&[2,3]);
            let t_0 = t.squeeze_dims(2);
            debug!("1. corr_full_node: flatten t_0.size()={:?}", t_0.size());

            t_0

        }), Some(params.channels))

        .linear()
        // .func(Tensor::relu)
        .linear_s(1)
        .func(Tensor::sigmoid)
        .end_seq()

}

#[derive(Debug)]
pub struct FinalNodeParams {
    features: i64,
    classes_num: i64
}

impl FinalNodeParams {
    pub fn new(features: i64, classes_num: i64) -> Self {
        Self { features, classes_num }
    }
}

pub fn final_node(params:&FinalNodeParams) -> Node {
    SeqBuilder::new(params.features)
        // .linear_s(features)
        // .func(Tensor::relu)
        // .linear_s(2*features)
        // .func(Tensor::relu)
        // .linear_s(features)
        // .func(Tensor::relu)
        .linear_s(params.classes_num)
        // .func(|t|t.softmax(0, Float))
        .end_seq()
}

pub fn train_on_motif(motif_id:usize, all_slices_0:Vec<Box<Tensor>>, all_slices_1:Vec<Box<Tensor>>, train_on_motif_params: &TrainOnMotifsParams, device: Device, node_params: &MotifNodeParams, batch_size: usize) -> Result<(f32, (VarStore, MotifNodeParams)), ThoosaError> {
    info!("train_on_motif: device={:?} training on {} slices_0 {} slices_1", device, all_slices_0.len(), all_slices_1.len());

    assert!(all_slices_0.len() > 0, "all_slices_0 len");
    assert!(all_slices_1.len() > 0, "all_slices_1 len");


    plot_series(
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
    ).unwrap();


    let mut min_loss = f32::MAX;
    let mut min_var_store : Option<(VarStore, MotifNodeParams)> = None;
    let mut batch = TensorsDataBatch::new(
        vec![
            all_slices_0,
            all_slices_1
        ],
        batch_size,
        0.8,
        device
    );

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


    match train_on_motif_params.to_config_space() {
        None => {
            let vs = VarStore::new(device);
            let lr = *train_on_motif_params.lr.expect_value()?;
            let node_params = node_params.with_num_layer(*train_on_motif_params.num_layers.expect_value()?);

            debug!("train_on_motif no opt: train_on_motif_params={train_on_motif_params:?}");

            let mut listener = PlotVecf32::new(format!("_motif_{}_params_{:?}", motif_id, train_on_motif_params).as_str());

            let loss = train(
                &device,
                &mut batch as &mut dyn DataBatch,

                &motif_node(&node_params).build_tch_module(&vs.root()).unwrap(),

                |pred: &Tensor, target: &Tensor| {
                    let pred_y = pred.i((.., 0, 0));

                    let loss = pred_y.binary_cross_entropy::<Tensor>(&target.totype(Float), None, Reduction::Mean);
                    debug!("train_on_motif: pred_y ={pred_y} target={target} loss={loss}");

                    loss
                },
                train_on_motif_params.cycles,
                nn::AdamW::default().build(&vs, lr).unwrap(),
                |t| t,
                Some(&mut listener),

            ).unwrap();

            listener.plot();

            if loss < min_loss {
                min_loss = loss;
                min_var_store.replace((vs, node_params));
            }

            println!("train_on_motif:  min_loss={min_loss}");
            // let best_config = opt.optimize();


            Ok((min_loss, min_var_store.expect("min_var_store")))
        }
        Some(cs) => {
            let mut opt = crate::parzen::optimizer::TPEOptimizer::new(
                |params| {
                    let vs = VarStore::new(device);
                    // let lr = 1. / 10_f64.powf(params[0].into());
                    let lr = params[0].into();
                    let node_params = node_params.with_num_layer(params[1].into());

                    debug!("train_on_motif in opt: params={params:?}");

                    let mut listener = PlotVecf32::new(format!("_motif_{}_params_{:?}", motif_id, params).as_str());

                    let loss = train(
                        &device,
                        &mut batch as &mut dyn DataBatch,

                        &motif_node(&node_params).build_tch_module(&vs.root()).unwrap(),

                        |pred: &Tensor, target: &Tensor| {
                            let pred_y = pred.i((.., 0, 0));

                            let loss = pred_y.binary_cross_entropy::<Tensor>(&target.totype(Float), None, Reduction::Mean);
                            debug!("train_on_motif: pred_y ={pred_y} target={target} loss={loss}");

                            loss
                        },
                        train_on_motif_params.cycles,
                        nn::AdamW::default().build(&vs, lr).unwrap(),
                        |t| t,
                        Some(&mut listener),

                    ).unwrap();

                    listener.plot();

                    debug!("train_on_motif: params={params:?} loss={loss}");
                    if loss < min_loss {
                        min_loss = loss;
                        min_var_store.replace((vs, node_params));
                    }

                    loss
                },
                cs,
                100,
                10,
            );
            let best_config = opt.optimize();
            println!("train_on_motif: best_config={best_config:?} min_loss={min_loss}");
            // let best_config = opt.optimize();


            Ok((min_loss, min_var_store.expect("min_var_store")))
        }
    }

}

pub fn slices_in_all_resources(slices: Vec<(String, Vec<Box<Tensor>>)>) -> Vec<Box<Tensor>> {
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
        ).view((batch_size as i64, 1, -1));



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


#[allow(dead_code)]
#[derive(Clone, Debug )]
pub enum MotifsBy {
    CorrRunId(usize),

    MotifsIds(Vec<usize>)
}

impl From<&str> for MotifsBy {
    fn from(v: &str) -> MotifsBy {
        debug!("From<&str> for MotifsBy: v={:?}", v);

        MotifsBy::CorrRunId(0)
    }
}

/// returns Tensor channels x nets x batches x batch size
fn apply_label_motifs(motif_n: i64, step: usize, input_t: &Tensor, channels_motifs_models: Arc<RwLock<Vec<Vec<SyncModule>>>>, motif_batch_size: i64, max_motif_batches: usize, device: &Device) -> Tensor {
    let max_nets = channels_motifs_models.read().iter().map(|nets| nets.len()).max().unwrap_or(0);//TODO remove

    debug!("apply_label_motifs: input_t.size()={:?} max_nets={max_nets} motif_batch_size={motif_batch_size} max_motif_batches={max_motif_batches}", input_t.size());

    let channels_motifs = Tensor::stack(
        channels_motifs_models.read()
            .iter()
            .enumerate()
            .map(|(channel, net)| apply_channel_motifs(motif_n, step, &net, input_t.i((channel as i64, ..)), motif_batch_size, max_nets as i64, max_motif_batches, device))
            .collect::<Vec<_>>().as_slice(),
        0
    );
    debug!("apply_label_motifs: channels_motifs.size={:?}", channels_motifs.size());
    // pad_to(&channels_motifs, &[eeg_t.size()[0], max_nets as i64, max_motif_batches as i64, motif_batch_size as i64]).unwrap()
    channels_motifs
}

#[derive(Debug, Args, Clone)]
pub struct TrainFinalParams {
    #[arg(long,  default_value_t = 1000)]
    step: usize,

    #[arg(long,  default_value_t = 1000)]
    window: usize,

    #[arg(long,  default_value_t = 100)]
    motif_batch_size: i64,

    #[arg(long, default_value_t = 1000)]
    cycles: usize,

    #[arg(long, default_value_t = 1.0e-5)]
    lr: f64,

    #[arg(long)]
    models_dir: Vec<String>,

    #[arg(long)]
    final_model: String,

    #[arg(long)]
    train_resource_group: String,

    #[arg(long)]
    valid_resource_group: String,

    #[arg(long, default_value_t = 5)]
    items_per_label:usize //TODO eeg batch size?
}

impl TrainFinalParams {

    pub fn cycles(&self) -> usize {
        self.cycles
    }

    pub fn lr(&self) -> f64 {
        self.lr
    }


    pub fn final_model(&self) -> &String {
        &self.final_model
    }

    pub fn dummy(step: usize, window: usize, models_dir:Vec<String>, final_model:String, train_resource_group:String, valid_resource_group:String, cycles: usize, motif_batch_size: i64, items_per_label: usize) ->Self {
        TrainFinalParams {
            step,
            window,
            cycles,
            lr: 1.0e-5,
            models_dir,
            final_model,
            train_resource_group,
            valid_resource_group,
            motif_batch_size,
            items_per_label
        }
    }
    pub fn train_resource_group(&self) -> &str {
        &self.train_resource_group
    }
    pub fn valid_resource_group(&self) -> &str {
        &self.valid_resource_group
    }


    pub fn step(&self) -> usize {
        self.step
    }
    pub fn window(&self) -> usize {
        self.window
    }

    pub fn motif_batch_size(&self) -> i64 {
        self.motif_batch_size
    }

    pub fn items_per_label(&self) -> usize {
        self.items_per_label
    }
}

#[derive(Debug, Args, Clone)]
pub struct InferParams {
    #[arg(long,  default_value_t = 1000)]
    step: usize,

    #[arg(long,  default_value_t = 1000)]
    window: usize,

    #[arg(long,  default_value_t = 100)]
    motif_batch_size: i64,

    #[arg(long, default_value = "/tmp/")]
    models_dir: String,

    #[arg(long, default_value = "/tmp/")]
    final_model: Option<String>,

    #[arg(long, default_value = "../../data_local/test_eegs")]
    input_data: String,

    #[arg(long, default_value = "submission.csv")]
    output: String,
}

impl InferParams {
    pub fn models_dir(&self) -> &str {
        &self.models_dir
    }
    pub fn final_model(&self) -> &Option<String> {
        &self.final_model
    }
    pub fn input_data(&self) -> &str {
        &self.input_data
    }
    pub fn output(&self) -> &str {
        &self.output
    }

    pub fn step(&self) -> usize {
        self.step
    }
    pub fn window(&self) -> usize {
        self.window
    }
    pub fn motif_batch_size(&self) -> i64 {
        self.motif_batch_size
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
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc};
    use parking_lot::{Mutex, RwLock};

    use log::debug;
    use tch::{Device, nn, Tensor};
    use tch::Device::Cpu;
    use tch::IndexOp;
    use tch::Kind::Float;
    use tch::nn::{init, LinearConfig, Module};

    use thoosa_shared::{batch_windows, ThoosaError};

    use crate::builder::SeqBuilder;
    use crate::loader::{LoadRunner, SyncWrapper};
    use crate::my_tch::apply_channel_motifs;
    use crate::train_nets::{TensorsDataBatch, InferParams, LabelNodeParams, MotifNodeParams, TrainFinalParams, TrainOnLabelParams, SyncModule, load_motif_models_ws};
    use crate::train_nets;
    use crate::utils::{cdt_for_dir, from};
    use crate::utils::DataBatch;
    use crate::train_nets::FinalNodeParams;
    use crate::ws::Workspace;

    #[ctor::ctor]
    fn test_init() {
        thoosa_shared::init_test_logging();
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

        debug!("train_batch: x={:?}", from::<Vec<f32>>(&train_batch.0.i((.., 0, 0))));
        debug!("train_batch: y={:?}", from::<Vec<f32>>(&train_batch.1));

        assert_eq!(from::<Vec<f32>>(&train_batch.0.i((.., 0, 0))), from::<Vec<f32>>(&train_batch.1));

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

    pub fn test_net(num_features: i64, k: f64, ) -> Result<Box<dyn Module>, ThoosaError> {
        let vs = nn::VarStore::new(Device::Cpu);

        SeqBuilder::new(num_features)
            .linear_s_with_conf(1, LinearConfig {
                ws_init: tch::nn::init::Init::Const(k),//conflicts with ctor's init
                bs_init: None,
                bias: false
            })
            .end_seq().build_tch_module(&vs.root())
    }


    #[test]
    fn test_apply_motif_models() -> Result<(), ThoosaError>{
        let input_n = 1000;
        let motif_n = 100;
        let step = (motif_n / 1) as usize;

        let t = make_test_tensor(input_n);

        let motif_nets = vec![
            vec![SyncModule::new(test_net(motif_n, 1.)?), SyncModule::new(test_net(motif_n, 2.)?)], //channel 0 nets
            vec![SyncModule::new(test_net(motif_n, 10.)?)] //channel 1 nets
        ];

        let max_motif_nets_in_channel = motif_nets.iter().map(|nets|nets.len()).max().unwrap_or(0);
        let batch_size = 10;
        let motif_batches = batch_windows(input_n as usize, motif_n as usize, step, batch_size);
        let max_motif_batches = motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0};

        let channel_0_t = apply_channel_motifs(motif_n, step, &motif_nets[0], t.i((0, ..)), batch_size as i64, max_motif_nets_in_channel as i64, max_motif_batches, &Cpu);


        let channel_0_motifs = Tensor::stack(&[
                Tensor::from_slice(&[4950., 14950., 24950., 34950., 44950., 54950., 64950., 74950., 84950., 94950.]),  //ch 0 net 0 (k=1)
                Tensor::from_slice(&[9900., 29900., 49900., 69900., 89900., 109900., 129900., 149900., 169900., 189900.]) // ch. 0 net 1 (k=2)
            ], 0
        ).totype(Float).view((2, 1, 10));
        assert_eq!(channel_0_motifs, channel_0_t);

        let channel_1_t = apply_channel_motifs(motif_n, step, &motif_nets[1], t.i((1, ..)), batch_size as i64, max_motif_nets_in_channel as i64, max_motif_batches, &Cpu);

        debug!("test_apply_motif_models: max_motif_batches={:?}", max_motif_batches);

        let channel_1_motifs = Tensor::stack(&[
                Tensor::from_slice(&[99000., 299000., 499000., 699000., 899000., 1.099e+06, 1.299e+06, 1.499e+06, 1.699e+06, 1.899e+06]),  //ch 1 net 0 (k=1)
                Tensor::from_slice(&[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),  //ch 1 net 0 (k=1)
            ], 0
        ).totype(Float).view((2, 1, 10));
        assert_eq!(channel_1_motifs, channel_1_t);

        debug!("test_apply_motif_models: channel_0_motifs.size()={:?}, channel_1_motifs.size()={:?}", channel_0_motifs.size(), channel_1_motifs.size());

        let channels_motifs = super::apply_label_motifs(motif_n, step, &t, Arc::new(RwLock::new(motif_nets)), batch_size as i64, max_motif_batches, &Cpu);

        assert_eq!(
            Tensor::stack(&[channel_0_motifs, channel_1_motifs], 0),
            channels_motifs
        );

        Ok(())
    }


    #[test]
    fn test_apply_motif_models_diff_len() -> Result<(), ThoosaError>{
        let input_n_0 = 1000;
        let motif_n= 120;
        let t_0 = make_test_tensor(input_n_0);
        let t_1 = make_test_tensor(831);

        let motif_batch_size = 13;

        let motif_nets = Arc::new(RwLock::new(vec![
            vec![SyncModule::new(test_net(motif_n, 1.)?), SyncModule::new(test_net(motif_n, 1.)?)], //channel 0 nets
            vec![SyncModule::new(test_net(motif_n, 10.)?)], //channel 1 nets
            vec![]
        ]));

        let step = 11;
        let batch_size = 10;
        let motif_batches = batch_windows(input_n_0 as usize, motif_n as usize, step, batch_size);
        let max_motif_batches = motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0};

        debug!("max_motif_batches={}", max_motif_batches);

        let channels_motifs_0 = super::apply_label_motifs(motif_n, step, &t_0, Arc::clone(&motif_nets), motif_batch_size, max_motif_batches, &Cpu);

        debug!("channels_motifs_0.size={:?}", channels_motifs_0.size());

        let channels_motifs_1 = super::apply_label_motifs(motif_n, step, &t_1, Arc::clone(&motif_nets), motif_batch_size, max_motif_batches, &Cpu);

        debug!("channels_motifs_1.size={:?}", channels_motifs_1.size());

        assert_eq!(vec![3, 2, 9, 13], channels_motifs_0.size());//channel, motif nets, batches, batch size
        assert_eq!(vec![3, 2, 9, 13], channels_motifs_1.size());
        Ok(())
    }

    //5 ch
    fn make_test_tensor(input_n: i64) -> Tensor {
        Tensor::vstack(&[
            Tensor::from_slice((0..input_n).map(|i| i as f32).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((0..input_n).map(|i| i as f32 * 2.).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((0..input_n).map(|i| i as f32 * 3.).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((0..input_n).map(|i| i as f32 * 4.).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((0..input_n).map(|i| i as f32 * 5.).collect::<Vec<_>>().as_slice())
        ])
    }

    fn load_test_tensor(id:&String) ->Box<Tensor> {
        debug!("test_load_tensor: {}", id);

        Box::new(make_test_tensor(1000))
    }


    #[test]
    fn test_train_on_label() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_train_on_label: test_dir={test_dir:?}");

        let ws = Workspace::new(test_dir)?;


        let params = TrainOnLabelParams {
            step: 100,
            cycles: 3,
            lr: 1.0e-5,
            train_labels_datasets: vec![],
            valid_labels_datasets: vec![],
            train_resource_group: None,
            valid_resource_group: None,
            label_batch_size: 13,
            motif_batch_size: 7
        };

        let motifs_path = ws.motifs_models_path();
        save_test_motif_model(&ws, "seizure", "0", "00", 100)?;
        save_test_motif_model(&ws, "seizure", "0", "01", 100)?;
        save_test_motif_model(&ws, "lpd", "0", "02", 100)?;
        save_test_motif_model(&ws, "gpd", "0", "03", 100)?;
        save_test_motif_model(&ws, "lrda", "1", "10", 100)?;
        save_test_motif_model(&ws, "grda", "1", "11", 100)?;
        save_test_motif_model(&ws, "other", "2", "20", 100)?;

        let resource_keys = |(label, kind)| {
            (label, (0..13).map(move |id| {
                format!("{}-{}-{}", label, kind, id)
            }).collect())
        };

        let executor_rc = Arc::new(parking_lot::RwLock::new(LoadRunner::new(
            move |key: &String| {

                SyncWrapper::new(
                    load_test_tensor(key),
                    |e| Box::new(e.shallow_clone())
                )
            }, 4, 100,
        )));

        let channels_motifs_models = load_motif_models_ws(&ws,  3,  &Cpu).expect("load_motif_models");//TODO num channels, motifs to util
        let channels_motifs_models: Arc<RwLock<Vec<Vec<SyncModule>>>> = Arc::new(RwLock::new(channels_motifs_models));

        for channel_models in channels_motifs_models.read().iter(){
            debug!("test_train_on_label: channel_models={}", channel_models.len());
        }

        super::train_on_label(
            Arc::new(RwLock::new(ws)),
            &MotifNodeParams::new(100),//TODO take window/num_features from saved motif model or motif run id
            &LabelNodeParams::new(3, 4, 20, 7), //max motif nets - 2 for seizure
            &params,
            vec![
                (1, "seizure_vote".to_string()),
                (0, "other".to_string()),
                (0, "gpd_vote".to_string()),
                (0, "lrda_vote".to_string()),
                (0, "grda_vote".to_string()),
                (0, "other_vote".to_string())

            ].into_iter().map(resource_keys).collect::<Vec<_>>(),
            vec![
                (1, "seizure_vote".to_string()),
                (0, "other".to_string()),
                (0, "gpd_vote".to_string()),
                (0, "lrda_vote".to_string()),
                (0, "grda_vote".to_string()),
                (0, "other_vote".to_string())

            ].into_iter().map(resource_keys).collect::<Vec<_>>(), 3,
            &Cpu,
            executor_rc,
            channels_motifs_models
        )?;

        Ok(())
    }

    #[test]
    fn test_train_final() -> Result<(), ThoosaError> {
        let dir = "/tmp/test_train_final";
        let test_base = Path::new(dir).join(cdt_for_dir());
        debug!("test_train_final: test_base: {:?}", test_base);

        let models_path_0 = test_base.join("0");
        let models_path_1 = test_base.join("1");

        let params = TrainFinalParams {
            step:100,
            window:100,
            motif_batch_size:100,
            cycles: 3,
            lr: 1.0e-5,
            models_dir: vec![
                models_path_0.to_str().unwrap().to_string(),
                models_path_1.to_str().unwrap().to_string()
            ],
            final_model: test_base.join("final.safetensors").to_str().expect("final_model path").to_string(),
            train_resource_group: "zzz".into(),
            valid_resource_group: "zzz".into(),
            items_per_label: 5
        };
        let channels = 5;

        let motif_batch_size = 7;
        let max_motif_batches = 978;//from 684400 in test_train_on_label


        let ws = Workspace::new(test_base)?;
        save_test_models(&ws, &models_path_0, channels, max_motif_batches, motif_batch_size)?;
        save_test_models(&ws, &models_path_1, channels, max_motif_batches, motif_batch_size)?;

        fs::write(&models_path_1.join("unexpected.file"), "lalala")?;


        let load_cl = move |_id| {
            (Box::new(make_test_tensor(1000)), Box::new(Tensor::from_slice(&[0.1, 0.1, 0.1, 0.1, 0.3, 0.3])))
        };
        super::train_final(
            &"/tmp/",
            &params,
            (0..13).map(load_cl),
            (0..13).map(load_cl),
            100,
            motif_batch_size,
            channels,
            684400,
            &Cpu,
            vec!["other", "lpd", "gpd", "lrda", "grda", "seizure"].into_iter().map(|v|v.into()).collect(),
        )?;

        fs::remove_dir_all(dir)?;

        Ok(())
    }

    //6 corrs 4 motif models
    fn save_test_models(ws: &Workspace, models_path: &PathBuf, channels: i64, max_motif_batches: i64, max_batch_size: i64) -> Result<(), ThoosaError> {
        fs::create_dir_all(models_path)?;

        let window = 100;
        fs::write(models_path.join("motif_model.json"), serde_json::to_string(&MotifNodeParams::new(window))?)?;


        save_test_label_model(&ws, "seizure", channels, 4, max_motif_batches, max_batch_size)?;
        fs::write(models_path.join("motif_model.json"), serde_json::to_string(&MotifNodeParams::new(window))?)?;


        save_test_motif_model(ws, "seizure", "0", "00", window)?;
        save_test_motif_model(ws, "seizure", "0", "01", window)?;
        save_test_motif_model(ws, "seizure", "0", "02", window)?;
        save_test_motif_model(ws, "seizure", "0", "03", window)?;
        save_test_motif_model(ws, "seizure", "1", "10", window)?;
        save_test_motif_model(ws, "seizure", "1", "11", window)?;
        save_test_motif_model(ws, "seizure", "2", "20", window)?;

        save_test_label_model(ws, "lpd", channels, 1, max_motif_batches, max_batch_size)?;

        save_test_motif_model(ws, "lpd", "0", "00", window)?;
        save_test_motif_model(ws, "lpd", "1", "10", window)?;
        save_test_motif_model(ws, "lpd", "2", "20", window)?;


        save_test_label_model(ws, "gpd", channels, 1, max_motif_batches, max_batch_size)?;

        save_test_motif_model(ws, "gpd", "0", "00", window)?;
        save_test_motif_model(ws, "gpd", "1", "10", window)?;
        save_test_motif_model(ws, "gpd", "2", "20", window)?;


        save_test_label_model(ws, "lrda", channels, 1, max_motif_batches, max_batch_size)?;

        save_test_motif_model(ws, "lrda", "0", "00", window)?;
        save_test_motif_model(ws, "lrda", "1", "10", window)?;
        save_test_motif_model(ws, "lrda", "2", "20", window)?;


        save_test_label_model(ws, "grda", channels, 1, max_motif_batches, max_batch_size)?;

        save_test_motif_model(ws, "grda", "0", "00", window)?;
        save_test_motif_model(ws, "grda", "1", "10", window)?;
        save_test_motif_model(ws, "grda", "2", "20", window)?;


        save_test_label_model(ws, "other", channels, 1, max_motif_batches, max_batch_size)?;

        save_test_motif_model(ws, "other", "0", "00", window)?;
        save_test_motif_model(ws, "other", "1", "10", window)?;
        save_test_motif_model(ws, "other", "2", "20", window)?;
        Ok(())
    }

    #[test]
    fn test_infer() -> Result<(), ThoosaError> {
        let channels = 5;
        let motif_batch_size = 7;
        let max_input = 684400;//TODO
        let motif_n = 100;
        let step = 100;

        // let motif_batches = batch_windows(max_input, motif_n as usize, step, motif_batch_size as usize);
        let max_motif_batches = 978; //(motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0}) as i64;

        let cdt = cdt_for_dir();
        let models_path = Path::new("/tmp/test_infer").join(&cdt);


        save_test_models(&Workspace::new(&cdt)?, &models_path, channels, max_motif_batches, motif_batch_size)?;
        save_test_final_model(&models_path, 6)?;

        let input_path = Path::new("/tmp/test_infer_input").join(&cdt);
        for i in 0..10 {
            save_test_tensor(&input_path, format!("123123213{}.parquet", i).as_str());
        }


        fn test_path_to_t(path:&Path) -> Result<Tensor, ThoosaError> {
            debug!("test_eeg_to_t: path={:?}", path);

            let t = Tensor::vstack(&[
                Tensor::rand(600, (Float, Cpu)),
                Tensor::rand(600, (Float, Cpu)),
                Tensor::rand(600, (Float, Cpu)),
                Tensor::rand(600, (Float, Cpu)),
                Tensor::rand(600, (Float, Cpu)),
            ]);
            // debug!("test_eeg_to_t:");
            // t.print();
            Ok(t)
        }


        let t = super::infer(
            &InferParams {
                step:100,
                window:100,
                motif_batch_size,
                models_dir: models_path.as_path().to_str().unwrap().to_string(),
                final_model: Some(models_path.join("final.safetensors").to_str().unwrap().to_string()),
                input_data: input_path.to_str().unwrap().to_string(),
                output: "/tmp/out.csv".to_string()
            },
            test_path_to_t,
            motif_n,
            step,
            motif_batch_size,
            channels,
            max_input,
            &Cpu,
            vec![
                "other".into(),
                "grda".into(),
                "lrda".into(),
                "gpd".into(),
                "lpd".into(),
                "seizure".into()
            ],
            MotifNodeParams::new(100)
        )?;

        debug!("t={:?}", t.1.size());

        assert_eq!(vec![10,6], t.1.size());

        Ok(())
    }

    fn save_test_tensor(input_path: &PathBuf, name: &str) {
        fs::create_dir_all(input_path).unwrap();
        fs::write(&input_path.join(name), "lalalal").unwrap();
    }

    fn save_test_final_model(models_path: &PathBuf, features: i64) -> Result<(), ThoosaError> {
        fs::create_dir_all(models_path.clone())?;

        let mut vs_0 = nn::VarStore::new(Device::Cpu);
        let _ = train_nets::final_node(&FinalNodeParams::new(features, features)).build_tch_module(&vs_0.root())?;

        vs_0.freeze();
        debug!("save_test_final_model: features={} vs_0.variables()={:?}", features, vs_0.variables());
        vs_0.save(models_path.join("final.safetensors"))?;
        Ok(())
    }

    fn save_test_label_model(ws: &Workspace, label:&str, channels: i64, max_motif_nets: i64, max_motif_batches: i64, motif_batch_size: i64) -> Result<(), ThoosaError> {
        // fs::create_dir_all(models_path.clone())?;
        let label_node_params = LabelNodeParams::new(channels, max_motif_nets, max_motif_batches, motif_batch_size);

        let mut vs_0 = nn::VarStore::new(Device::Cpu);
        let _ = train_nets::label_node(&label_node_params).build_tch_module(&vs_0.root())?;

        //TODO use label path from ws
        fs::write(ws.models_path().join(format!("{}_0.json", label)), serde_json::to_string(&label_node_params)?)?;

        vs_0.freeze();
        debug!("save_test_corr_model: channels={}, max_motif_nets={}, max_motif_batches={}, motif_batch_size={} vs_0.variables()={:?}", channels, max_motif_nets, max_motif_batches, motif_batch_size, vs_0.variables());
        vs_0.save(ws.models_path().join(format!("{}_0.safetensors", label)))?;
        Ok(())
    }

    fn save_test_motif_model(ws: &Workspace, label: &str, channel: &str, motif_id: &str, window: i64) -> Result<(), ThoosaError> {
        // let channel_0 = ws/*.join(format!("{}", label))*/.join(channel);
        // fs::create_dir_all(channel_0.clone())?;

        let mut vs_0 = nn::VarStore::new(Device::Cpu);
        let _test_motif_model = train_nets::motif_node(&MotifNodeParams::new(window)).build_tch_module(&vs_0.root())?;

        let test_motif_model = SeqBuilder::new(100)
            .linear_s_with_conf(1, LinearConfig {
                ws_init: init::Init::Const(1.0),
                bs_init: None,
                bias: false
            })
            .end_seq().build_tch_module(&vs_0.root())?;


        vs_0.freeze();
        debug!("save_test_motif_model: channel={channel} vs_0.variables()={:?}", vs_0.variables());
        // vs_0.save(channel_0.join(format!("test_motif_{}.safetensors", motif_id)))?;
        ws.save_motif_model(channel, 1, motif_id, vs_0, MotifNodeParams::new(100));
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
    fn test_test() -> Result<(), ThoosaError> {
        let t_1 = Tensor::stack(&[Tensor::from_slice(&[0., 0., 0.]), Tensor::from_slice(&[0., 0., 0.])], 0);


        debug!("size={:?} t={}", t_1.size(), super::is_nonzero(&Tensor::from_slice(&[0., 1., 0.]))?);

        // debug!("t={}", t.is_nonzero());

        // let mut vs_0 = nn::VarStore::new(Cpu);
        //
        // let batch_len = 11;
        // let d_0 = 3;
        // let d_1 = 7;
        // let d_2 = 5;
        //
        // let mut builder = SeqBuilder::new(d_2);
        //
        //
        // super::build_multidim(&mut builder, &[d_2, d_1, d_0, 1]);
        //
        //
        // let net = builder.end_seq().build_tch_module(&vs_0.root())?;
        //
        // let tensor = Tensor::zeros(&[batch_len, d_0, d_1, d_2], (Float, Cpu));
        // let out = net.forward(&tensor);
        //
        // debug!("out.size={:?}", out.size());


        // let t =
        //     Tensor::stack(&[
        //         Tensor::vstack(&[
        //             Tensor::from_slice(&[1., 1., 1.]),
        //             Tensor::from_slice(&[2., 2., 2.]),
        //             Tensor::from_slice(&[3., 3., 3.])
        //         ]),
        //         Tensor::vstack(&[
        //             Tensor::from_slice(&[4., 4., 4.]),
        //             Tensor::from_slice(&[5., 5., 5.]),
        //             Tensor::from_slice(&[6., 6., 6.])
        //         ])
        //     ], 0);


        // let t1 = Instant::now();
        //
        // let mut t = Tensor::stack((0..2).map(|_|
        //     Tensor::stack((0..10).map(|_|
        //         Tensor::stack((0..100).map(|_|
        //                          Tensor::zeros(&[1000], (Double, Cpu)) //batches
        //         ).collect::<Vec<Tensor>>().as_slice(), 0)
        //     ).collect::<Vec<Tensor>>().as_slice(), 0)
        // ).collect::<Vec<Tensor>>().as_slice(), 0);
        //
        //
        // debug!("t.size()={:?} elapsed={}", t.size(), t1.elapsed().as_millis());
        // let t1 = Instant::now();
        // for c in 0..2 {
        //     for n in 0..10 {
        //         for b in 0..100 {
        //             t = t.index_put(&[Some(Tensor::from_slice(&[c])), Some(Tensor::from_slice(&[n])), Some(Tensor::from_slice(&[b]))], &Tensor::ones(&[1000], (Double, Cpu)), false);
        //         }
        //     }
        // }

        // debug!("t.size()={:?} elapsed={}", t.size(), t1.elapsed().as_millis());

        // t.index_put(&[Some(Tensor::from_slice(&[1])), Some(Tensor::from_slice(&[2]))], &Tensor::ones(&[3], (Double, Cpu)), false).print();


        Ok(())
    }

}
