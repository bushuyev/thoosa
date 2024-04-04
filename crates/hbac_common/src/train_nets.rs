use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::iter::Map;
use std::marker::PhantomData;
use hbac_shared::{HbacError, batch_windows};
use tch::nn::{ConvConfig, Module, OptimizerConfig, VarStore};
use tch::{Device, IndexOp, Kind, nn, Reduction, Tensor};
use clap::clap_derive::Args;
use log::{debug, info, warn, trace};
use rand::{Rng, thread_rng};
use crate::builder::{clc_conv1d_out, Node, SeqBuilder};
use crate::utils::{DataBatch, pad_to, train};
use crate::utils::from;
use rand::prelude::SliceRandom;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;
use crate::train_nets;
use std::path::Path;
use tch::Device::Cpu;
use tch::Kind::{Double, Float};
use std::sync::Mutex;
use crate::my_tch::apply_channel_motifs;
use regex::Regex;
use serde::{Deserialize, Serialize};
use hbac_shared::parse_key_val;


#[derive(Debug, Args, Clone)]
pub struct TrainOnLabelParams {

    #[arg(long,  default_value_t = 1000)]
    step: usize,

    #[arg(long,  default_value_t = 1000)]
    window: usize,

    #[arg(long)]
    train_resource_group: Option<String>,

    #[arg(long)]
    valid_resource_group: Option<String>,


    #[arg(long, value_parser=parse_key_val::<String, String>)]
    train_labels_datasets: Vec<(String, String)>,

    #[arg(long, value_parser=parse_key_val::<String, String>)]
    valid_labels_datasets: Vec<(String, String)>,


    #[arg(long)]
    corr_run: usize,

    #[arg(long, default_value = "/tmp/")]
    motif_models_dir: String,

    #[arg(long, default_value = "/tmp/")]
    models_dir: String,

    #[arg(long, default_value_t = 10)]
    cycles: usize,

    #[arg(long, default_value_t = 1.0e-5)]
    lr: f64,

    #[arg(long, default_value_t = 10)]
    motif_batch_size:i64,

    #[arg(long, default_value_t = 5)]
    items_per_label:usize //TODO eeg batch size?
}

impl TrainOnLabelParams {
    pub fn corr_run(&self) -> usize {
        self.corr_run
    }
    pub fn motif_models_dir(&self) -> &str {
        &self.motif_models_dir
    }

    pub fn cycles(&self) -> usize {
        self.cycles
    }
    pub fn lr(&self) -> f64 {
        self.lr
    }
    pub fn models_dir(&self) -> &str {
        &self.models_dir
    }

    pub fn dummy(step: usize, window: usize, motif_models_dir:String, models_dir:String, train_labels_datasets: Vec<(String, String)>, valid_labels_datasets: Vec<(String, String)>, motif_batch_size: i64, items_per_label: usize, cycles: usize, lr: f64) ->Self {
        TrainOnLabelParams {
            step,
            window,
            corr_run:0,
            motif_models_dir,
            models_dir,
            cycles,
            lr,
            train_labels_datasets,
            valid_labels_datasets,
            train_resource_group: None,
            valid_resource_group: None,
            motif_batch_size,
            items_per_label
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


    pub fn items_per_label(&self) -> usize {
        self.items_per_label
    }


    pub fn step(&self) -> usize {
        self.step
    }
    pub fn window(&self) -> usize {
        self.window
    }
}

/**
* returns num labels * items_per_label x channels x motifs x motifs batch size
*/
struct LabeledTensorsLoaderDataBatch<I> {
    train_labels_tensor_loaders: Vec<Vec<I>>,
    valid_labels_tensor_loaders: Vec<Vec<I>>,
    items_per_label: usize
}

impl <I> DataBatch for LabeledTensorsLoaderDataBatch<I> where I:Iterator<Item=Box<Tensor>> + Send + Sync {

    fn train_batch(&mut self) -> (Tensor, Tensor){
        Self::batch(&mut self.train_labels_tensor_loaders, self.items_per_label as f32)
    }

    fn valid_batch(&mut self) -> (Tensor, Tensor){
        Self::batch(&mut self.valid_labels_tensor_loaders, self.items_per_label as f32) //TODO replacement, just pull next
    }
}

impl<I> LabeledTensorsLoaderDataBatch<I> where I: Iterator<Item=Box<Tensor>> + Send + Sync {

    pub fn new(train_labels_tensor_loaders: Vec<Vec<I>>, valid_labels_tensor_loaders: Vec<Vec<I>>, items_per_label: usize) -> Self {
        Self {
            train_labels_tensor_loaders,
            valid_labels_tensor_loaders,
            items_per_label,
        }
    }

    fn batch(loaders: &mut Vec<Vec<I>>, items_per_label: f32) -> (Tensor, Tensor) {
        let label_tensors = loaders.iter_mut().enumerate().map(|(label, loaders)| {
            let take = (items_per_label / loaders.len() as f32).ceil() as usize;

            debug!("train_batch: label={:?}, self.batch_size={:?}, loaders.len()={:?}, take={:?}", label, items_per_label, loaders.len(), take);

            (label as i64, loaders.iter_mut()
                .flat_map(|loader| loader.take(take))
                .collect::<Vec<Box<Tensor>>>())
        }).collect::<Vec<(i64, Vec<Box<Tensor>>)>>();

        debug!("train_batch: label_tensors={} label_tensors={:?}", label_tensors.len(), label_tensors);

        let x_y = label_tensors.into_iter().fold((vec![], vec![]), |mut a, v| {
            (0..v.1.len()).for_each(|_| { a.1.push(v.0); });//label  - y
            a.0.extend(v.1);//tensox - x

            a
        });

        (
            Tensor::stack(x_y.0.as_slice(), 0).totype(Float),
            Tensor::from_slice(x_y.1.as_slice()).totype(Float).to_device(x_y.0[0].device())//TODO device
        )
    }
}


struct ClassTensorsLoaderData<I> {
    train_loader: I,
    valid_loader: I,
}

impl <I> ClassTensorsLoaderData<I> where I:Iterator<Item=Result<(Tensor, Tensor), HbacError>> + Send + Sync {

    pub fn new(train_loader: I, valid_loader: I) -> Self {
        Self {
            train_loader,
            valid_loader,
        }
    }

}


impl <I> DataBatch for ClassTensorsLoaderData<I> where I:Iterator<Item=Result<(Tensor, Tensor), HbacError>> + Send + Sync {

    fn train_batch(&mut self) -> (Tensor, Tensor){
        Self::batch(&mut self.train_loader)
    }

    fn valid_batch(&mut self) -> (Tensor, Tensor){
        Self::batch(&mut self.valid_loader) //no replacement, just pull next
    }
}

impl<I> ClassTensorsLoaderData<I> where I: Iterator<Item=Result<(Tensor, Tensor), HbacError>> + Send + Sync {

    fn batch(i: &mut I) -> (Tensor, Tensor) {
        debug!("ClassTensorsLoaderData.batch:");

        let t = i.next().expect("next").expect("result");//TODO expects

        (
            t.0,
            t.1
        )
    }
}

pub fn train_on_label<I>(params: &TrainOnLabelParams, train_labels_loaders:Vec<(i64, I)>, valid_labels_loaders: Vec<(i64, I)>, items_per_label: usize, motif_n: i64, step: usize, motif_batch_size: i64, channels: i64, max_input:usize, device: &Device, label: &String) -> Result<(VarStore, LabelNodeParams), HbacError>
    where I:Iterator<Item=Box<Tensor>> + Send + Sync {

    let mut train_labels_loaders_map = train_labels_loaders.into_iter().fold(HashMap::new(), |mut a, (label, loader)|{
        a.entry(label).or_insert_with(||vec![]).push(loader);
        a
    });

    let mut valid_labels_loaders_map = valid_labels_loaders.into_iter().fold(HashMap::new(), |mut a, (label, loader)|{
        a.entry(label).or_insert_with(||vec![]).push(loader);
        a
    });

    let mut keys = train_labels_loaders_map.keys().map(|v|*v).collect::<Vec<i64>>();
    keys.sort();

    let mut valid_keys = valid_labels_loaders_map.keys().map(|v|*v).collect::<Vec<i64>>();
    valid_keys.sort();

    assert!(keys == valid_keys);

    debug!("train_on_eeg: keys={:?} labels_loaders={:?}", keys, train_labels_loaders_map.keys());

    let channels_motifs_models = load_motif_models(params.motif_models_dir.as_str(), label.clone(), channels as usize, device)?;//TODO num channels
    // debug!("train_on_eeg: models.size={:?} models={:?}", channels_motifs_models.len(), channels_motifs_models);
    debug!("train_on_eeg: models.size={:?}", channels_motifs_models.len());

    //TODO remove mutexes
    let motif_nets: Vec<Vec<Mutex<Box<dyn Module>>>> = channels_motifs_models.into_iter().map(|v|v.into_iter().map(|m|Mutex::new(m)).collect::<Vec<_>>()).collect::<Vec<_>>();

    let max_nets =  motif_nets.iter().map(|nets|nets.len()).max().unwrap_or(0) as i64;
    if max_nets <= 0 {
        return Err(HbacError::new("no motif nets found"));
    }

    let motif_batches = batch_windows(max_input, motif_n as usize, step, motif_batch_size as usize);
    let max_motif_batches = motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0};

    let mut vs_0 = nn::VarStore::new(device.clone());

    debug!("train_on_eeg: channels={:?}  max_nets={:?}  max_motif_batches={:?} motif_batch_size={:?}", channels,  max_nets,  max_motif_batches, motif_batch_size);
    debug!("train_on_eeg: [items_per_label as i64* 2={:?}, channels={:?}, max_nets={:?}, max_motif_batches as i64 * motif_batch_size={:?}]", items_per_label as i64* 2, channels, max_nets, max_motif_batches as i64 * motif_batch_size);
    let apply_motifs = |t:Box<Tensor>| Box::new(apply_label_motifs(motif_n, step, &t, &motif_nets, motif_batch_size, max_motif_batches, device));
    let node_params = LabelNodeParams::new(channels, max_nets, max_motif_batches as i64, motif_batch_size);
    train(
        device,
        &mut LabeledTensorsLoaderDataBatch::new(
            apply_motifs_to_loaders(
                motif_n, step, motif_batch_size, device, train_labels_loaders_map, &mut keys, &motif_nets, max_motif_batches,
                apply_motifs,
            ),

            apply_motifs_to_loaders(
                motif_n, step, motif_batch_size, device, valid_labels_loaders_map, &mut keys, &motif_nets, max_motif_batches,
                apply_motifs,
            ),

            items_per_label

        ) as &mut dyn DataBatch,

        // &corr_full_node(channels * max_nets * max_motif_batches as i64 * motif_batch_size).build_tch_module(&vs_0.root())?,
        &label_node(&node_params).build_tch_module(&vs_0.root())?,
        |pred: &Tensor, target: &Tensor| {
            let pred_y = pred.squeeze();
            //pred.i((.., 0, 0));
            debug!("train_on_eeg: pred_y={:?} target={:?} kind={:?}", pred_y.size(), target.size(), target.kind());
            pred_y.binary_cross_entropy::<Tensor>(&target, None, Reduction::Mean)
        },
        params.cycles,
        nn::AdamW::default().build(&vs_0, params.lr)?,
        |t: Tensor| {
            debug!("train_on_eeg: t.size()={:?} ", t.size());
            //merge max_motif_batches * motif_batch_size
            let t_out = view_for_label_model(t, channels, max_nets, motif_batch_size, max_motif_batches); //.view((t.size()[0], -1));
            //t.size()[0] - batch size
            debug!("train_on_eeg: t_out.size()={:?}",  t_out.size());
            t_out
        },
    )?;



    Ok((vs_0, node_params))
}

fn view_for_label_model(t: Tensor, channels: i64, max_nets: i64, motif_batch_size: i64, max_motif_batches: usize) -> Tensor {
    assert!(t.size().len()==5, "view_for_label_model: t.size()={:?}, should be 5 dims", t.size());
    t.view([-1/*items_per_label as i64 * 2*/, channels, max_nets, max_motif_batches as i64 * motif_batch_size])
}

fn apply_motifs_to_loaders<I, F>(motif_n: i64, step: usize, motif_batch_size: i64, device: &Device, mut labels_loaders: HashMap<i64, Vec<I>>, keys: &mut Vec<i64>, motif_nets: &Vec<Vec<Mutex<Box<dyn Module>>>>, max_motif_batches: usize,
apply_motifs:F) -> Vec<Vec<Map<I, F>>> where I: Iterator<Item=Box<Tensor>> + Send + Sync, F: Fn(Box<Tensor>)->Box<Tensor> +Copy{
    keys.into_iter()
        .map(
            |label| labels_loaders
                .remove(&label).expect(format!("no {}", label).as_str())
                .into_iter()
                .map( |l| l.map(apply_motifs))
                .collect::<Vec<_>>())
        .collect::<Vec<Vec<_>>>()
}

pub fn train_final<I>(params: &TrainFinalParams, train_tensors_loader:I, valid_tensors_loader: I, motif_n: i64, step: usize, motif_batch_size: i64, channels: i64, max_input:usize, device: &Device, labels: Vec<String>) -> Result<VarStore, HbacError>
    where I:Iterator<Item=(Box<Tensor>, Box<Tensor>)> + Send + Sync {


    let motif_batches = batch_windows(max_input, motif_n as usize, step, motif_batch_size as usize);
    let max_motif_batches = motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0};

    debug!("train_final: channels={:?}, max_motif_batches={:?} motif_batch_size={:?} features={:?} motif_n={:?}", channels, max_motif_batches, motif_batch_size, channels *  max_motif_batches as i64 * motif_batch_size, motif_n);

    let models = params.models_dir.iter().flat_map(|dir|
            load_label_and_motif_models(dir.as_str(), channels as usize, motif_n, max_motif_batches as i64, motif_batch_size, device).expect(format!("load_corr_and_motif_models: {}", dir).as_str())
    ).collect::<Vec<LabelModels>>();


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
                        let t = apply_label_motifs(motif_n, step, &t.0, &model.motif_models, motif_batch_size, max_motif_batches, device);

                        //TODO move into CorrModels
                        let max_nets =  model.motif_models.iter().map(|nets|nets.len()).max().unwrap_or(0) as i64;
                        debug!("train_final: motif models model.label={} t.size()={:?}", model.label, t.size());
                        debug!("train_final: t.size()={:?} ", t.size());

                        let t_x = view_for_label_model(t.reshape([1, channels, max_nets, motif_batch_size, max_motif_batches as i64]), channels, max_nets, motif_batch_size, max_motif_batches);
                        debug!("train_final: t_x.size()={:?} max_nets={}", t_x.size(), max_nets);

                        let t = model.label_model.lock().unwrap().forward(&t_x);
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

    let mut vs_0 = nn::VarStore::new(device.clone());

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

            let t_out = t.view((t.size()[0] * t.size()[1])); //model_dirs * labels
            debug!(" t_out.size()={:?}",  t_out.size());
            t_out
        }
    )?;

    Ok(vs_0)
}

pub fn infer<F>(params: &InferParams, eeg_to_tensor:F, motif_n: i64, step: usize, motif_batch_size: i64, channels: i64, max_input:usize, device: &Device, labels: Vec<String>) -> Result<(Vec<String>, Tensor), HbacError>
    where F:Fn(&Path) -> Result<Tensor, HbacError> + Sync{

    let motif_batches = batch_windows(max_input, motif_n as usize, step, motif_batch_size as usize);
    let max_motif_batches = motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0};

    debug!("infer: channels={:?}, max_motif_batches={:?} motif_batch_size={:?} features={:?} motif_n={:?}", channels, max_motif_batches, motif_batch_size, channels *  max_motif_batches as i64 * motif_batch_size, motif_n);

    let models = load_label_and_motif_models(params.models_dir.as_str(), channels as usize, motif_n, max_motif_batches as i64, motif_batch_size, device)?
        .into_iter().fold(HashMap::new(), |mut a, v|{
        a.insert(v.label.clone(), v);
        a
    });

    let mut vs = nn::VarStore::new(*device);
    let final_model = Mutex::new(train_nets::final_node(&FinalNodeParams::new(6, 6)).build_tch_module(&vs.root())?);
    vs.load(params.final_model.as_ref().unwrap_or(&Path::new(&params.models_dir).join("final.safetensors").to_str().unwrap().to_string()))?;
    vs.freeze();

    debug!("infer: params.input_data={:?} models={:?}", params.input_data, models);

    let tensors_ids = fs::read_dir(&params.input_data)?.par_bridge().map(|eeg| {
        let eeg = eeg.expect("file: ");

        let t = eeg_to_tensor(eeg.path().as_path()).expect("eeg");
        debug!("infer: t.size={:?}", t.size());

        let t = Tensor::stack(labels.iter().map(|label|{
            match models.get(label) {
                Some(model)=>{
                    let t = apply_label_motifs(motif_n, step, &t, &model.motif_models, motif_batch_size, max_motif_batches, device);
                    let t_x = t.view((1, -1));
                    debug!("infer: motif models model={:?} model.label={} t.size()={:?} t_x.size()={:?}", model, model.label, t.size(), t_x.size());

                    let t = model.label_model.lock().unwrap().forward(&t_x);
                    debug!("corr_model out .size()={:?}", t.size());
                    t.view(1)
                }
                None => {
                    Tensor::zeros(&[1], (Float, *device))
                }
            }

        }).collect::<Vec<Tensor>>().as_slice(), 0);

        debug!("infer: t.size()={:?}", t.size());

        let pred = final_model.lock().expect("lock").forward(&t.transpose(0, 1)); //for single eeg

        let pred= pred.softmax(1, Float);
        debug!("infer: pred={:?}", pred.size());
        // debug!("infer: pred={:?}", from::<Vec<f32>>(&pred));
        (
            eeg.path().file_stem().unwrap().to_str().unwrap().to_string(),
            pred
        )
    }).collect::<Vec<(String, Tensor)>>();

    let (ids, tensors) = tensors_ids.into_iter().fold((vec![], vec![]), |mut a, v|{
        a.0.push(v.0);
        a.1.push(v.1);
        a
    });
    Ok((ids, Tensor::vstack(tensors.as_slice())))

}

#[derive(Debug)]
struct LabelModels {
    label:String,
    label_model:Mutex<Box<dyn Module>>,
    motif_models:Vec<Vec<Mutex<Box<dyn Module>>>>
}

impl LabelModels {

    pub fn new(label:String, corr_model: Box<dyn Module>, motif_models: Vec<Vec<Box<dyn Module>>>) -> Self {
        Self {
            label,
            label_model: Mutex::new(corr_model),
            motif_models: motif_models.into_iter().map(|v|v.into_iter().map(|m|Mutex::new(m)).collect::<Vec<_>>()).collect::<Vec<_>>()
        }
    }

}

/**
* features without x motif nets
*/
fn load_label_and_motif_models(motif_models_path: impl AsRef<Path>, channels: usize, motif_n: i64, max_motif_batches: i64, motif_batch_size: i64, device: &Device) -> Result<Vec<LabelModels>, HbacError> {
    let re = Regex::new(r"^(?<label>[a-zA-Z0-9]+?)_(?<corr_run>\d+).safetensors").unwrap();//TODO lazy_static

    debug!("load_corr_and_motif_models: channels={:?}, motif_n={:?}, max_motif_batches={:?}, motif_batch_size={:?}", channels, motif_n, max_motif_batches, motif_batch_size);

    let mut models = vec![];
    for e in fs::read_dir(motif_models_path.as_ref())? {
        let entry = e?;

        if entry.file_type()?.is_file() {
            let file_name = entry.file_name();
            let file_name = file_name.to_str().expect(format!("to_str {:?}", file_name).as_str());
            debug!("file_name={:?}", file_name);

            let Some(caps) = re.captures(file_name) else {
                warn!("unexpected file {:?} in models {:?}", file_name, motif_models_path.as_ref());
                continue
            };

            let label = caps["label"].parse::<String>()?;
            let corr_run = caps["corr_run"].parse::<String>()?;
            debug!("label={} corr_run={}", label, corr_run);

            let mut vs = nn::VarStore::new(device.clone());

            let channels_motifs_models = load_motif_models(motif_models_path.as_ref(), label.clone(), channels, &device)?;
            assert!(channels_motifs_models.len() > 0);
            let max_nets =  channels_motifs_models.iter().map(|nets|nets.len()).max().unwrap_or(0) as i64;

            let label_node_params = serde_json::from_str::<LabelNodeParams>(
                fs::read_to_string(&motif_models_path.as_ref().join(format!("{}_{}.json", label, corr_run)))?.as_str()
            )?;

            let corr_model = train_nets::label_node(&label_node_params).build_tch_module(&vs.root())?;
            vs.load(entry.path())?;
            vs.freeze();

            models.push(crate::train_nets::LabelModels::new(label, corr_model, channels_motifs_models))
        }

    }
    //TODO should be sorted as labels and same len
    models.sort_by(|a, b|a.label.cmp(&b.label));

    Ok(models)
}

pub fn load_motif_models(motif_models_path: impl AsRef<Path>, label:String, channels:usize, device: &Device) -> Result<Vec<Vec<Box<dyn Module>>>, HbacError> {
    debug!("load_motif_models: motif_models_path={:?} label={:?}", motif_models_path.as_ref(), label);

    let motif_node_params = serde_json::from_str::<MotifNodeParams>(
        fs::read_to_string(&motif_models_path.as_ref().join("motif_model.json"))?.as_str()
    )?;

    debug!("train_on_eeg: motif_node_params={:?}", motif_node_params);

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
                let node = train_nets::motif_node(&motif_node_params).build_tch_module(&vs.root())?;//TODO window
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
    Ok(channels_motifs_models)

}

#[derive(Debug, Args, Clone)]
pub struct TrainOnMotifsParams {
    #[arg(long)]
    motif_ids: MotifsBy,

    #[arg(long)]
    corr_run_ids: Vec<usize>,

    #[arg(long,  default_value_t = 1000)]
    window: usize,//should be same as one used when finding motifs or TODO center bigger/smaller window

    #[arg(long, default_value = "/tmp/")]
    output_dir: String,

    #[arg(long, default_value_t = 1000)]
    cycles: usize,

    #[arg(long, default_value_t = 1.0e-5)]
    lr: f64,

    #[arg(long, default_value_t = 0.2)]
    max_loss: f32,

    #[arg(long, default_value_t = 0.05)]
    min_corr: f64,
}

impl TrainOnMotifsParams {

    pub fn output_dir(&self) -> &str {
        &self.output_dir
    }

    pub fn cycles(&self) -> usize {
        self.cycles
    }
    pub fn lr(&self) -> f64 {
        self.lr
    }

    pub fn window(&self) -> usize {
        self.window
    }

    pub fn motif_ids(&self) -> &MotifsBy {
        &self.motif_ids
    }

    pub fn corr_run_ids(&self) -> &Vec<usize> {
        &self.corr_run_ids
    }

    pub fn max_loss(&self) -> f32 {
        self.max_loss
    }

    pub fn min_corr(&self) -> f64 {
        self.min_corr
    }

    pub fn dummy(output_dir: &String, corr_run_ids:Vec<usize>, window: usize, cycles: usize) ->Self{
        TrainOnMotifsParams {
            motif_ids: MotifsBy::CorrRunId(0),

            corr_run_ids,

            window,//should be same as one used when finding motifs or TODO center bigger/smaller window

            output_dir: output_dir.clone(),

            cycles,

            lr: 1.0e-5,

            max_loss: 0.2,

            min_corr: 0.05,
        }
    }
}
#[derive(Debug, Serialize, Deserialize)]
pub struct MotifNodeParams {
    window: i64
}

impl MotifNodeParams {
    pub fn new(window: i64) -> Self {
        Self { window }
    }
}

pub fn motif_node(params: &MotifNodeParams) -> Node {
    SeqBuilder::new(params.window)
        .batch_norm()
        .conv_1d(10, 1, 1, ConvConfig::default())
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
            (0..params.max_motif_nets).map( |i|
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
            let t_size = t.size();

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
struct FinalNodeParams {
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

pub fn train_on_motif(all_slices_0:Vec<Box<Tensor>>, all_slices_1:Vec<Box<Tensor>>, train_on_motif_params: &TrainOnMotifsParams, device: Device, node_params: &MotifNodeParams, batch_size: usize) -> Result<(f32, VarStore), HbacError> {
    info!("train_on_motif: device={:?} training on {} slices_0 {} slices_1", device, all_slices_0.len(), all_slices_1.len());

    assert!(all_slices_0.len() > 0, "all_slices_0 len");
    assert!(all_slices_1.len() > 0, "all_slices_1 len");

    let vs = nn::VarStore::new(device);

    let net = train_nets::motif_node(node_params).build_tch_module(&vs.root())?;

    let valid_loss = crate::utils::train(
        &device,
        &mut ClassTensorsDataBatch::new(
            vec![
                all_slices_0,
                all_slices_1
            ],
            batch_size,
            0.8,
            device
        ) as &mut dyn DataBatch,
        &net,
        |pred: &Tensor, target: &Tensor| {
            // pred.cross_entropy_loss::<Tensor>(&target.totype(Kind::Int64), None, Reduction::Mean, -100, 0.0)
            let pred_y = pred.i((.., 0, 0));
            debug!("train_on_motif: pred_y.device={:?}, target.device()={:?} pred_y ={:?} ", pred_y.device(), target.device(), from::<Vec<f32>>(&pred_y));
            pred_y.binary_cross_entropy::<Tensor>(&target.totype(Kind::Float), None, Reduction::Mean)
        },
        train_on_motif_params.cycles,
        nn::AdamW::default().build(&vs, train_on_motif_params.lr)?,
        |t|t
    )?;
    Ok((valid_loss, vs))
}

pub fn flat_map(slices: Vec<(String, Vec<Box<Tensor>>)>) -> Vec<Box<Tensor>> {
    slices.into_iter().flat_map(|(eeg_id, slices)| slices.into_iter()).collect::<Vec<Box<Tensor>>>()
}


#[derive(Debug)]
pub struct ClassTensorsDataBatch {
    train :Vec<Vec<Box<Tensor>>>,//index - class
    val :Vec<Vec<Box<Tensor>>>,
    batch_size: usize,
    device:Device
}

impl ClassTensorsDataBatch {

    pub fn new(classes_tensors: Vec<Vec<Box<Tensor>>>, batch_size: usize, train_valid_split:f32, device:Device) -> Self {
        /*
        TODO impl - shuffle first
        let mut rng = thread_rng();

        let mut x_y:Vec<(Tensor, usize)> = classes_tensors.into_iter().flat_map(|(label, tensors)|tensors.into_iter().map(move |t|(t, label))).collect();
        x_y.shuffle(&mut rng);

        let split_i = (x_y.len() as f32 * split) as usize;

        println!("split_i={}", split_i);
        let train = &x_y[..split_i];
        let valid = &x_y[split_i..];
        */

        let classes_len = classes_tensors.len();
        assert!(classes_len > 0);
        assert!(batch_size % classes_len == 0);

        let (train, val):(Vec<Vec<Box<Tensor>>>, Vec<Vec<Box<Tensor>>>) = classes_tensors
            .into_iter()
            .fold((Vec::with_capacity(classes_len), Vec::with_capacity(classes_len)), |mut class_train_val, tensors|{

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


            class_train_val.0.push(train_val.0);//train
            class_train_val.1.push(train_val.1); //validation
            class_train_val
        });

        Self {
            train,
            val,
            batch_size,
            device
        }
    }

    fn to_x_y(class_tensors: &Vec<Vec<Box<Tensor>>>, batch_size: usize, device:&Device) -> (Tensor, Tensor) {
        let mut rng = thread_rng();

        let per_class_len = batch_size / class_tensors.len();//TODO make balanced
        debug!("ClassTensorsDataBatch::to_x_y: per_class_len={} device={:?}", per_class_len, device);

        let mut x_y_vecs = class_tensors.iter().enumerate().fold(vec![], |mut x_y_acc, (class, tensors)| {
            assert!(tensors.len() > 0, "no tensors for class={} in ClassTensorsDataBatch", class);

            tensors.choose_multiple(&mut rng, per_class_len).for_each(|t| {

                x_y_acc.push((t.shallow_clone(), class as i64));
            });

            x_y_acc
        });

        x_y_vecs.shuffle(&mut rng);

        debug!("ClassTensorsDataBatch::to_x_y: x_y_vecs.1.len()={} ", x_y_vecs.len());

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



        debug!("ClassTensorsDataBatch::to_x_y: tensor.size()={:?} tensor.device()={:?}", tensor.size(), tensor.device());

        (
            tensor,
            classes
        )
    }
}

impl DataBatch for ClassTensorsDataBatch {

    fn train_batch(&mut self) -> (Tensor, Tensor) {
        Self::to_x_y(&self.train, self.batch_size, &self.device)
    }

    fn valid_batch(&mut self) -> (Tensor, Tensor) {
        Self::to_x_y(&self.val, self.batch_size, &self.device)
    }
}


#[derive(Clone, Debug )]
enum MotifsBy {
    CorrRunId(usize),

    MotifsIds(Vec<usize>)
}

impl From<&str> for MotifsBy {
    fn from(v: &str) -> MotifsBy {
        debug!("From<&str> for MotifsBy: v={:?}", v);

        MotifsBy::CorrRunId(0)
    }
}

fn apply_label_motifs(motif_n: i64, step: usize, eeg_t: &Tensor, motif_nets: &Vec<Vec<Mutex<Box<dyn Module>>>>, motif_batch_size: i64, max_motif_batches: usize, device: &Device) -> Tensor {
    let max_nets = motif_nets.iter().map(|nets| nets.len()).max().unwrap_or(0);

    debug!("apply_label_motifs: max_motif_nets_in_channel={:?} motif_batch_size={:?} max_motif_batches={:?}, ", max_nets, motif_batch_size, max_motif_batches);

    let channels_motifs = Tensor::stack(
        motif_nets
            .iter()
            .enumerate()
            .map(|(i, net)| apply_channel_motifs(motif_n, step, &net, eeg_t.i((i as i64, ..)), motif_batch_size, max_nets as i64, max_motif_batches, device))
            // .map(|t|pad_to(&t, &[max_motif_batches as i64, max_in_fold as i64]).unwrap())
            .collect::<Vec<_>>().as_slice(),
        0
    );
    // pad_to(&channels_motifs, &[max_motif_nets_in_channel as i64, max_folds as i64, max_in_fold as i64]).unwrap()
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

    pub fn models_dir(&self) -> &Vec<String> {
        &self.models_dir
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

fn build_multidim<'a>( builder:&'a mut SeqBuilder, dims:&'a [i64])  {

    let builder = dims.windows(2).into_iter().fold(builder, |mut b, d|{
        debug!("d={:?}", d);

        b.linear_s(d[0])
            // .func(Tensor::relu)
            .linear_s(1)
            // .func(Tensor::sigmoid)
            .closure(Box::new(|t:&Tensor|t.squeeze()), Some(d[1]))

    });

    builder.func(Tensor::sigmoid);
}

pub fn is_nonzero(t_1: &Tensor) -> Result<bool, HbacError> {
    Ok(t_1.flatten(0, (t_1.size().len() - 1) as i64).iter::<f64>()?.find(|v| v > &0.).is_some())
}



#[cfg(test)]
mod train_tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use tch::{Device, nn, Tensor};
    use hbac_shared::{HbacError, batch_windows};
    use crate::train_nets::{ClassTensorsDataBatch, load_motif_models, motif_node, TrainOnLabelParams, TrainFinalParams, InferParams, LabelNodeParams, FinalNodeParams, MotifNodeParams};
    use crate::utils::{cdt_for_dir, from};
    use crate::utils::DataBatch;
    use tch::IndexOp;
    use crate::train_nets;
    use log::debug;
    use tch::Kind::{Double, Float};
    use tch::nn::{ConvConfig, LinearConfig, Module, VarStore};
    use crate::builder::SeqBuilder;
    use std::sync::Mutex;
    use std::thread::Builder;
    use std::time::Instant;
    use tch::Device::Cpu;
    use crate::my_tch::apply_channel_motifs;
    use tch::nn::init;

    #[test]
    fn test_class_tensors_data_batch()-> Result<(), HbacError>{

        let mut batch = ClassTensorsDataBatch::new(
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

        println!("train_batch: x={:?}", from::<Vec<f32>>(&train_batch.0.i((.., 0, 0))));
        println!("train_batch: y={:?}", from::<Vec<f32>>(&train_batch.1));

        assert_eq!(from::<Vec<f32>>(&train_batch.0.i((.., 0, 0))), from::<Vec<f32>>(&train_batch.1));

        // println!("train_batch: x={:?}, y={:?}", train_batch.0.size(), train_batch.1.size());

        assert_eq!(10, train_batch.0.size()[0]);
        assert_eq!(5, train_batch.1.iter::<i64>()?.filter(|v|*v==1).count());
        assert_eq!(5, train_batch.1.iter::<i64>()?.filter(|v|*v==0).count());

        let valid_batch = batch.valid_batch();
        assert_eq!(10, valid_batch.0.size()[0]);
        assert_eq!(5, valid_batch.1.iter::<i64>()?.filter(|v|*v==1).count());
        assert_eq!(5, valid_batch.1.iter::<i64>()?.filter(|v|*v==0).count());


        Ok(())
    }

    pub fn test_net(num_features: i64, k: f64, ) -> Result<Box<dyn Module>, HbacError> {
        let mut vs = nn::VarStore::new(Device::Cpu);

        SeqBuilder::new(num_features)
            .linear_s_with_conf(1, LinearConfig {
                ws_init: init::Init::Const(k),
                bs_init: None,
                bias: false
            })
            .end_seq().build_tch_module(&vs.root())
    }


    #[test]
    fn test_apply_motif_models() -> Result<(), HbacError>{
        let input_n = 1000;
        let motif_n = 100;
        let step = (motif_n / 1) as usize;

        let eeg_t = make_test_eeg(input_n);

        let motif_nets = vec![
            vec![Mutex::new(test_net(motif_n, 1.)?), Mutex::new(test_net(motif_n, 2.)?)], //channel 0 nets
            vec![Mutex::new(test_net(motif_n, 10.)?)] //channel 1 nets
        ];

        let max_motif_nets_in_channel = motif_nets.iter().map(|nets|nets.len()).max().unwrap_or(0);
        let batch_size = 10;
        let motif_batches = batch_windows(input_n as usize, motif_n as usize, step, batch_size);
        let max_motif_batches = motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0};

        let channel_0_t = apply_channel_motifs(motif_n, step, &motif_nets[0], eeg_t.i((0, ..)), batch_size as i64, max_motif_nets_in_channel as i64, max_motif_batches, &Cpu);


        let channel_0_motifs = Tensor::stack(&[
                Tensor::from_slice(&[4950., 14950., 24950., 34950., 44950., 54950., 64950., 74950., 84950., 94950.]),  //ch 0 net 0 (k=1)
                Tensor::from_slice(&[9900., 29900., 49900., 69900., 89900., 109900., 129900., 149900., 169900., 189900.]) // ch. 0 net 1 (k=2)
            ], 0
        ).totype(Float).view((2, 1, 10));
        assert_eq!(channel_0_motifs, channel_0_t);

        let channel_1_t = apply_channel_motifs(motif_n, step, &motif_nets[1], eeg_t.i((1, ..)), batch_size as i64, max_motif_nets_in_channel as i64, max_motif_batches, &Cpu);

        println!("max_motif_batches={:?}", max_motif_batches);

        let channel_1_motifs = Tensor::stack(&[
                Tensor::from_slice(&[99000., 299000., 499000., 699000., 899000., 1.099e+06, 1.299e+06, 1.499e+06, 1.699e+06, 1.899e+06]),  //ch 1 net 0 (k=1)
                Tensor::from_slice(&[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),  //ch 1 net 0 (k=1)
            ], 0
        ).totype(Float).view((2, 1, 10));
        assert_eq!(channel_1_motifs, channel_1_t);

        println!("channel_0_motifs.size()={:?}, channel_1_motifs.size()={:?}", channel_0_motifs.size(), channel_1_motifs.size());

        let channels_motifs = super::apply_label_motifs(motif_n, step, &eeg_t, &motif_nets, batch_size as i64, max_motif_batches, &Cpu);

        assert_eq!(
            Tensor::stack(&[channel_0_motifs, channel_1_motifs], 0),
            channels_motifs
        );

        Ok(())
    }


    #[test]
    fn test_apply_motif_models_diff_len() -> Result<(), HbacError>{
        let input_n_0 = 1000;
        let motif_n= 120;
        let eeg_t_0 = make_test_eeg(input_n_0);
        let eeg_t_1 = make_test_eeg(831);

        let motif_batch_size = 13;

        let motif_nets = vec![
            vec![Mutex::new(test_net(motif_n, 1.)?), Mutex::new(test_net(motif_n, 1.)?)], //channel 0 nets
            vec![Mutex::new(test_net(motif_n, 10.)?)], //channel 1 nets
            vec![]
        ];

        let step = 11;
        let batch_size = 10;
        let motif_batches = batch_windows(input_n_0 as usize, motif_n as usize, step, batch_size);
        let max_motif_batches = motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0};

        debug!("max_motif_batches={}", max_motif_batches);

        let channels_motifs_0 = super::apply_label_motifs(motif_n, step, &eeg_t_0, &motif_nets, motif_batch_size, max_motif_batches, &Cpu);

        println!("channels_motifs_0.size={:?}", channels_motifs_0.size());

        let channels_motifs_1 = super::apply_label_motifs(motif_n, step, &eeg_t_1, &motif_nets, motif_batch_size, max_motif_batches, &Cpu);

        println!("channels_motifs_1.size={:?}", channels_motifs_1.size());

        assert_eq!(vec![3, 2, 9, 13], channels_motifs_0.size());//channel, motif nets, batches, batch size
        assert_eq!(vec![3, 2, 9, 13], channels_motifs_1.size());
        Ok(())
    }

    fn make_test_eeg(input_n: i64) -> Tensor {
        Tensor::vstack(&[
            Tensor::from_slice((0..input_n).map(|i| i as f32).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((0..input_n).map(|i| i as f32 * 2.).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((0..input_n).map(|i| i as f32 * 3.).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((0..input_n).map(|i| i as f32 * 4.).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((0..input_n).map(|i| i as f32 * 5.).collect::<Vec<_>>().as_slice())
        ])
    }

    fn load_test_tensor(id:String) ->Box<Tensor> {
        debug!("test_load_tensor: {}", id);

        Box::new(make_test_eeg(1000))
    }


    #[test]
    fn test_train_on_label() -> Result<(), HbacError> {
        let models_path = Path::new("/tmp/test_train_on_eeg").join(cdt_for_dir());
        fs::create_dir_all(&models_path);
        fs::write(models_path.join("motif_model.json"), serde_json::to_string(&MotifNodeParams::new(100))?)?;

        let params = TrainOnLabelParams {
            corr_run: 0,
            step: 1000,
            window: 1000,
            motif_models_dir: models_path.to_str().unwrap().to_string(),
            models_dir: models_path.to_str().unwrap().to_string(),
            cycles: 3,
            lr: 1.0e-5,
            train_labels_datasets: vec![],
            valid_labels_datasets: vec![],
            train_resource_group: None,
            valid_resource_group: None,
            items_per_label: 5,
            motif_batch_size: 5
        };

        save_test_motif_model(&models_path, "seizure", "0", "00", 100)?;
        save_test_motif_model(&models_path, "seizure", "0", "01", 100)?;
        save_test_motif_model(&models_path, "lpd", "0", "02", 100)?;
        save_test_motif_model(&models_path, "gpd", "0", "03", 100)?;
        save_test_motif_model(&models_path, "lrda", "1", "10", 100)?;
        save_test_motif_model(&models_path, "grda", "1", "11", 100)?;
        save_test_motif_model(&models_path, "other", "2", "20", 100)?;

        let loader = |(label, kind)| {
            (label, (0..13).map(move |id| {
                load_test_tensor(format!("{}-{}-{}", label, kind, id))
            }))
        };
        super::train_on_label(
            &params,
            vec![
                (1, "seizure_vote".to_string()),
                (0, "other".to_string()),
                (0, "gpd_vote".to_string()),
                (0, "lrda_vote".to_string()),
                (0, "grda_vote".to_string()),
                (0, "other_vote".to_string())

            ].into_iter().map(loader).collect::<Vec<_>>(),
            vec![
                (1, "seizure_vote".to_string()),
                (0, "other".to_string()),
                (0, "gpd_vote".to_string()),
                (0, "lrda_vote".to_string()),
                (0, "grda_vote".to_string()),
                (0, "other_vote".to_string())

            ].into_iter().map(loader).collect::<Vec<_>>(),
            13, 100, 100, 7, 5,
            684400,
            &Cpu,
            &"seizure".to_string()
        )?;

        Ok(())
    }

    #[test]
    fn test_train_final() -> Result<(), HbacError> {
        let test_base = Path::new("/tmp/test_train_final").join(cdt_for_dir());
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
        let max_motif_batches = 978;//from 684400 in test_train_on_eeg

        let max_nets = 4;

        save_test_models(&models_path_0, channels, max_motif_batches, motif_batch_size)?;
        save_test_models(&models_path_1, channels, max_motif_batches, motif_batch_size)?;

        fs::write(&models_path_1.join("unexpected.file"), "lalala");


        let load_cl = move |id| {
            (Box::new(make_test_eeg(1000)), Box::new(Tensor::from_slice(&[0.1, 0.1, 0.1, 0.1, 0.3, 0.3])))
        };
        super::train_final(
            &params,
            (0..13).map(load_cl),
            (0..13).map(load_cl),
            100,
            100,
            motif_batch_size,
            channels,
            684400,
            &Cpu,
            vec!["other", "lpd", "gpd", "lrda", "grda", "seizure"].into_iter().map(|v|v.into()).collect(),
        )?;

        Ok(())
    }

    //6 corrs 4 motif models
    fn save_test_models(models_path: &PathBuf, channels: i64, max_motif_batches: i64, max_batch_size: i64) -> Result<(), HbacError> {
        fs::create_dir_all(models_path);

        let window = 100;
        fs::write(models_path.join("motif_model.json"), serde_json::to_string(&MotifNodeParams::new(window))?)?;


        save_test_label_model(&models_path, "seizure", channels, 4, max_motif_batches, max_batch_size);
        fs::write(models_path.join("motif_model.json"), serde_json::to_string(&MotifNodeParams::new(window))?)?;


        save_test_motif_model(&models_path, "seizure", "0", "00", window)?;
        save_test_motif_model(&models_path, "seizure", "0", "01", window)?;
        save_test_motif_model(&models_path, "seizure", "0", "02", window)?;
        save_test_motif_model(&models_path, "seizure", "0", "03", window)?;
        save_test_motif_model(&models_path, "seizure", "1", "10", window)?;
        save_test_motif_model(&models_path, "seizure", "1", "11", window)?;
        save_test_motif_model(&models_path, "seizure", "2", "20", window)?;

        save_test_label_model(&models_path, "lpd", channels, 1, max_motif_batches, max_batch_size);

        save_test_motif_model(&models_path, "lpd", "0", "00", window)?;
        save_test_motif_model(&models_path, "lpd", "1", "10", window)?;
        save_test_motif_model(&models_path, "lpd", "2", "20", window)?;


        save_test_label_model(&models_path, "gpd", channels, 1, max_motif_batches, max_batch_size);

        save_test_motif_model(&models_path, "gpd", "0", "00", window)?;
        save_test_motif_model(&models_path, "gpd", "1", "10", window)?;
        save_test_motif_model(&models_path, "gpd", "2", "20", window)?;


        save_test_label_model(&models_path, "lrda", channels, 1, max_motif_batches, max_batch_size);

        save_test_motif_model(&models_path, "lrda", "0", "00", window)?;
        save_test_motif_model(&models_path, "lrda", "1", "10", window)?;
        save_test_motif_model(&models_path, "lrda", "2", "20", window)?;


        save_test_label_model(&models_path, "grda", channels, 1, max_motif_batches, max_batch_size);

        save_test_motif_model(&models_path, "grda", "0", "00", window)?;
        save_test_motif_model(&models_path, "grda", "1", "10", window)?;
        save_test_motif_model(&models_path, "grda", "2", "20", window)?;


        save_test_label_model(&models_path, "other", channels, 1, max_motif_batches, max_batch_size);

        save_test_motif_model(&models_path, "other", "0", "00", window)?;
        save_test_motif_model(&models_path, "other", "1", "10", window)?;
        save_test_motif_model(&models_path, "other", "2", "20", window)?;
        Ok(())
    }

    #[test]
    fn test_infer() -> Result<(), HbacError> {
        let channels = 5;
        let motif_batch_size = 7;
        let max_input = 700;
        let motif_n = 100;
        let step = 100;

        let motif_batches = batch_windows(max_input, motif_n as usize, step, motif_batch_size as usize);
        let max_motif_batches = (motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0}) as i64;

        let models_path = Path::new("/tmp/test_train_final").join(cdt_for_dir());
        let max_nets = 4;

        save_test_models(&models_path, channels, max_motif_batches, 100)?;
        save_test_final_model(&models_path, 6);

        let input_path = Path::new("/tmp/test_train_final_input").join(cdt_for_dir());
        for i in 0..10 {
            save_fake_eeg(&input_path, format!("123123213{}.parquet", i).as_str());
        }


        fn test_eeg_to_t(path:&Path) -> Result<Tensor, HbacError> {
            debug!("test_eeg_to_t: path={:?}", path);

            let t = Tensor::vstack(&[
                Tensor::rand((600), (Float, Cpu)),
                Tensor::rand((600), (Float, Cpu)),
                Tensor::rand((600), (Float, Cpu)),
                Tensor::rand((600), (Float, Cpu)),
                Tensor::rand((600), (Float, Cpu)),
            ]);
            // println!("test_eeg_to_t:");
            // t.print();
            Ok(t)
        }


        let t = super::infer(
            &InferParams {
                step:1000,
                window:1000,
                motif_batch_size:100,
                models_dir: models_path.as_path().to_str().unwrap().to_string(),
                final_model: Some(models_path.join("final.safetensors").to_str().unwrap().to_string()),
                input_data: input_path.to_str().unwrap().to_string(),
                output: "/tmp/out.csv".to_string()
            },
            test_eeg_to_t,
            motif_n,
            step,
            motif_batch_size,
            channels,
            max_input,
            &Cpu,
            vec!["label_a".into(), "label_b".into()],
        )?;

        debug!("t={:?}", t.1.size());

        assert_eq!(vec![10,6], t.1.size());

        Ok(())
    }

    fn save_fake_eeg(input_path: &PathBuf, name: &str) {
        fs::create_dir_all(input_path);
        fs::write(&input_path.join(name), "lalalal");
    }

    fn save_test_final_model(models_path: &PathBuf, features: i64) -> Result<(), HbacError> {
        fs::create_dir_all(models_path.clone());

        let mut vs_0 = nn::VarStore::new(Device::Cpu);
        let test_final_model = train_nets::final_node(&FinalNodeParams::new(features, features)).build_tch_module(&vs_0.root())?;

        vs_0.freeze();
        debug!("save_test_final_model: features={} vs_0.variables()={:?}", features, vs_0.variables());
        vs_0.save(models_path.join("final.safetensors"))?;
        Ok(())
    }

    fn save_test_label_model(models_path: &PathBuf, label:&str, channels: i64, max_motif_nets: i64, max_motif_batches: i64, motif_batch_size: i64) -> Result<(), HbacError> {
        fs::create_dir_all(models_path.clone());
        let label_node_params = LabelNodeParams::new(channels, max_motif_nets, max_motif_batches, motif_batch_size);

        let mut vs_0 = nn::VarStore::new(Device::Cpu);
        let test_corr_model = train_nets::label_node(&label_node_params).build_tch_module(&vs_0.root())?;


        fs::write(models_path.join(format!("{}_0.json", label)), serde_json::to_string(&label_node_params)?)?;

        vs_0.freeze();
        debug!("save_test_corr_model: channels={}, max_motif_nets={}, max_motif_batches={}, motif_batch_size={} vs_0.variables()={:?}", channels, max_motif_nets, max_motif_batches, motif_batch_size, vs_0.variables());
        vs_0.save(models_path.join(format!("{}_0.safetensors", label)))?;
        Ok(())
    }

    fn save_test_motif_model(models_path: &PathBuf, label: &str, channel: &str, motif_id: &str, window: i64) -> Result<(), HbacError> {
        let channel_0 = models_path.join(format!("{}", label)).join(channel);
        fs::create_dir_all(channel_0.clone());

        let mut vs_0 = nn::VarStore::new(Device::Cpu);
        let test_motif_model = train_nets::motif_node(&super::MotifNodeParams::new(window)).build_tch_module(&vs_0.root())?;

        // let test_motif_model = SeqBuilder::new(100)
        //     .linear_s_with_conf(1, LinearConfig {
        //         ws_init: init::Init::Const(1.0),
        //         bs_init: None,
        //         bias: false
        //     })
        //     .end_seq().build_tch_module(&vs_0.root())?;


        vs_0.freeze();
        debug!("save_test_motif_model: vs_0.variables()={:?}", vs_0.variables());
        vs_0.save(channel_0.join(format!("test_motif_{}.safetensors", motif_id)))?;
        Ok(())
    }

    #[test]
    fn test_is_nonzero() -> Result<(), HbacError> {

        assert!(super::is_nonzero(&Tensor::from_slice(&[0., 1., 0.]))?);
        assert!(super::is_nonzero(&Tensor::stack(&[Tensor::from_slice(&[0., 1., 0.]), Tensor::from_slice(&[0., 0., 0.])], 0))?);
        assert!(!super::is_nonzero(&Tensor::from_slice(&[0., 0., 0.]))?);

        Ok(())
    }

    #[test]
    fn test_test() -> Result<(), HbacError> {
        let t_1 = Tensor::stack(&[Tensor::from_slice(&[0., 0., 0.]), Tensor::from_slice(&[0., 0., 0.])], 0);


        println!("size={:?} t={}", t_1.size(), super::is_nonzero(&Tensor::from_slice(&[0., 1., 0.]))?);

        // println!("t={}", t.is_nonzero());

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
        // println!("t.size()={:?} elapsed={}", t.size(), t1.elapsed().as_millis());
        // let t1 = Instant::now();
        // for c in 0..2 {
        //     for n in 0..10 {
        //         for b in 0..100 {
        //             t = t.index_put(&[Some(Tensor::from_slice(&[c])), Some(Tensor::from_slice(&[n])), Some(Tensor::from_slice(&[b]))], &Tensor::ones(&[1000], (Double, Cpu)), false);
        //         }
        //     }
        // }

        // println!("t.size()={:?} elapsed={}", t.size(), t1.elapsed().as_millis());

        // t.index_put(&[Some(Tensor::from_slice(&[1])), Some(Tensor::from_slice(&[2]))], &Tensor::ones(&[3], (Double, Cpu)), false).print();


        Ok(())
    }

}
