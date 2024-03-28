use std::collections::{HashMap, VecDeque};
use std::fs;
use std::iter::Map;
use std::marker::PhantomData;
use hbac_shared::{HbacError, batch_windows};
use tch::nn::{ConvConfig, Module, OptimizerConfig, VarStore};
use tch::{Device, IndexOp, Kind, nn, Reduction, Tensor};
use clap::clap_derive::Args;
use log::{debug, info, warn};
use rand::{Rng, thread_rng};
use crate::builder::{Node, SeqBuilder};
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
use hbac_shared::parse_key_val;

#[derive(Debug, Args, Clone)]
pub struct TrainOnEEGParams {


    #[arg(long, value_parser=parse_key_val::<String, String>, default_value="seizure=seizure/100,lpd=lpd/10")]
    labels_datasets: Vec<(String, String)>,

    #[arg(long)]
    corr_run: usize,

    #[arg(long, default_value = "/tmp/")]
    motif_models_dir: String,

    #[arg(long, default_value = "/tmp/")]
    eeg_models_dir: String,

    #[arg(long, default_value_t = 10)]
    cycles: usize,

    #[arg(long, default_value_t = 1.0e-5)]
    lr: f64,
}

impl TrainOnEEGParams {
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
    pub fn eeg_models_dir(&self) -> &str {
        &self.eeg_models_dir
    }



    pub fn dummy(motif_models_dir:String, eeg_models_dir:String, labels_datasets: Vec<(String, String)>) ->Self {
        TrainOnEEGParams {
            corr_run:0,
            motif_models_dir,
            eeg_models_dir,
            cycles: 100,
            lr:1.0e-5,
            labels_datasets
        }
    }
    pub fn labels_datasets(&self) -> &Vec<(String, String)> {
        &self.labels_datasets
    }
}

fn run_motif_nets<I>(loader: I) where I:Iterator<Item=Result<Tensor, HbacError>>, I: std::marker::Send {
    for eeg in loader {

    }
}

/**
* returns num labels * items_per_label x channels x motifs x motifs batch size
*/
struct LabeledTensorsLoaderDataBatch<I> {
    classes_tensor_loaders: Vec<Vec<I>>,
    items_per_label: usize
}

impl <I> DataBatch for LabeledTensorsLoaderDataBatch<I> where I:Iterator<Item=Box<Tensor>> + Send + Sync {

    fn train_batch(&mut self) -> (Tensor, Tensor){
        self.batch()
    }

    fn valid_batch(&mut self) -> (Tensor, Tensor){
        self.batch() //TODO replacement, just pull next
    }
}

impl<I> LabeledTensorsLoaderDataBatch<I> where I: Iterator<Item=Box<Tensor>> + Send + Sync {

    pub fn new(labels_tensor_loaders: Vec<Vec<I>>, items_per_label: usize, train_valid_split: f32) -> Self {
        Self {
            classes_tensor_loaders: labels_tensor_loaders,
            items_per_label: items_per_label //TODO it should be batch_size
        }
    }

    fn batch(&mut self) -> (Tensor, Tensor) {
        let label_tensors = self.classes_tensor_loaders.iter_mut().enumerate().map(|(label, loaders)| {
            let take = (self.items_per_label as f32 / loaders.len() as f32).ceil() as usize;

            debug!("train_batch: label={:?}, self.batch_size={:?}, loaders.len()={:?}, take={:?}", label, self.items_per_label, loaders.len(), take);

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
    loader: I,
}

impl <I> ClassTensorsLoaderData<I> where I:Iterator<Item=Result<(Tensor, Tensor), HbacError>> + Send + Sync {

    pub fn new(loader: I, train_valid_split: f32) -> Self {
        Self {
            loader,
        }
    }

}


impl <I> DataBatch for ClassTensorsLoaderData<I> where I:Iterator<Item=Result<(Tensor, Tensor), HbacError>> + Send + Sync {

    fn train_batch(&mut self) -> (Tensor, Tensor){
        self.batch()
    }

    fn valid_batch(&mut self) -> (Tensor, Tensor){
        self.batch() //no replacement, just pull next
    }
}

impl<I> ClassTensorsLoaderData<I> where I: Iterator<Item=Result<(Tensor, Tensor), HbacError>> + Send + Sync {

    fn batch(&mut self) -> (Tensor, Tensor) {
        debug!("ClassTensorsLoaderData.batch:");

        let t = self.loader.next().expect("next").expect("result");//TODO expects

        (
            t.0,
            t.1
        )
    }
}

pub fn train_on_eeg<I>(params: &TrainOnEEGParams, labels_loaders:Vec<(i64, I)>, items_per_label: usize, motif_n: i64, step: usize, motif_batch_size: i64, channels: i64, max_input:usize, device: &Device, label: &String) -> Result<VarStore, HbacError>
    where I:Iterator<Item=Box<Tensor>> + Send + Sync {

    //TODO it should be either Vec or HashMap for labels - in ClassTensorsDataBatch etc
    let mut labels_loaders = labels_loaders.into_iter().fold(HashMap::new(), |mut a, (label, loader)|{
        a.entry(label).or_insert_with(||vec![]).push(loader);
        a
    });

    let mut keys = labels_loaders.keys().map(|v|*v).collect::<Vec<i64>>();
    keys.sort();

    debug!("train_on_eeg: keys={:?} labels_loaders={:?}", keys, labels_loaders.keys());
    let models = load_motif_models(params.motif_models_dir.as_str(), label.clone(), channels as usize, motif_n, device)?;//TODO num channels
    debug!("train_on_eeg: models.size={:?} models={:?}", models.len(), models);

    let motif_nets: Vec<Vec<Mutex<Box<dyn Module>>>> = models.into_iter().map(|v|v.into_iter().map(|m|Mutex::new(m)).collect::<Vec<_>>()).collect::<Vec<_>>();

    let max_nets =  motif_nets.iter().map(|nets|nets.len()).max().unwrap_or(0) as i64;
    if max_nets <= 0 {
        return Err(HbacError::new("no motif nets found"));
    }

    let motif_batches = batch_windows(max_input, motif_n as usize, step, motif_batch_size as usize);
    let max_motif_batches = motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0};

    let loaders_vec = keys.into_iter()
        .map(
            |label| labels_loaders
                .remove(&label).expect(format!("no {}", label).as_str())
                .into_iter()
                .map(|l|{
                    l.map(|t|{
                        Box::new(apply_motifs(motif_n, step, &t, &motif_nets, motif_batch_size, max_motif_batches, device))
                    })
                })
                .collect::<Vec<_>>())
        .collect::<Vec<Vec<_>>>();

    let mut vs_0 = nn::VarStore::new(device.clone());

    debug!("train_on_eeg: channels={:?}  max_nets={:?}  max_motif_batches={:?}", channels,  max_nets,  max_motif_batches);
    train(
        device,
        &mut LabeledTensorsLoaderDataBatch::new(loaders_vec, items_per_label, 0.8) as &mut dyn DataBatch,
        &corr_full_node(channels * max_nets *  max_motif_batches as i64 * motif_batch_size).build_tch_module(&vs_0.root())?,
        |pred: &Tensor, target: &Tensor| {

            let pred_y = pred; //pred.i((.., 0, 0));
            debug!("pred_y ={:?} target={:?}", pred_y.size(), target.size());
            pred_y.binary_cross_entropy::<Tensor>(&target.totype(Kind::Float), None, Reduction::Mean)
        },
        params.cycles,
        nn::AdamW::default().build(&vs_0, 1.0e-4)?,
        |t:Tensor|{
            let t_out = t.view((t.size()[0], -1));//t.size()[0] - batch size
            debug!("t.size()={:?} t_out.size()={:?}", t.size(), t_out.size());
            t_out
        }
    )?;

    Ok(vs_0)
}

pub fn train_final<I>(params: &TrainFinalParams, tensors_loader:I, motif_n: i64, step: usize, motif_batch_size: i64, channels: i64, max_input:usize, device: &Device, labels: Vec<String>) -> Result<VarStore, HbacError>
    where I:Iterator<Item=(Box<Tensor>, Box<Tensor>)> + Send + Sync {


    let motif_batches = batch_windows(max_input, motif_n as usize, step, motif_batch_size as usize);
    let max_motif_batches = motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0};

    debug!("train_final: channels={:?}, max_motif_batches={:?} motif_batch_size={:?} features={:?} motif_n={:?}", channels, max_motif_batches, motif_batch_size, channels *  max_motif_batches as i64 * motif_batch_size, motif_n);

    let models = params.models_dir.iter().flat_map(|dir|
            load_corr_and_motif_models(dir.as_str(), channels *  max_motif_batches as i64 * motif_batch_size, motif_n, channels as usize, device).expect(format!("load_corr_and_motif_models: {}", dir).as_str())
    ).collect::<Vec<CorrModels>>();


    let labels_models = models.into_iter().fold(HashMap::new(), |mut a, corr_models| {
        a.entry(corr_models.label.clone()).or_insert_with(||vec![]).push(corr_models);
        a
    });


    debug!("train_final: models.size={:?} models={:?}", labels_models.len(), labels_models);

    let tensors_loader =  tensors_loader.map( move |t|{
        debug!("train_final: tensors_loader.map: t.0.size={:?} t.1.size={:?}", t.0.size(), t.1.size());

        Ok(
            (
                Tensor::stack(labels.iter().map(|label|{
                    match labels_models.get(label) {
                        Some(label_models)=>{

                            Tensor::stack(label_models.iter().map(|model|{
                                let t = apply_motifs(motif_n, step, &t.0, &model.motif_models, motif_batch_size, max_motif_batches, device);
                                let t_x = t.view((1, -1));
                                debug!("train_final: motif models model.label={} t.size()={:?} t_x.size()={:?}", model.label, t.size(), t_x.size());

                                let t = model.corr_model.lock().unwrap().forward(&t_x);
                                debug!("train_final: corr_model out .size()={:?}", t.size());
                                t
                            }).collect::<Vec<Tensor>>().as_slice(), 0)
                        }
                        None => {
                            debug!("train_final: no model for label {:?}", label);


                            Tensor::zeros(&[params.models_dir.len() as i64, 1], (Float, *device))
                        }
                    }

                }).collect::<Vec<Tensor>>().as_slice(), 0),

                t.1.shallow_clone()
            )
        )
    });

    let mut vs_0 = nn::VarStore::new(device.clone());

    train(
        device,
        &mut ClassTensorsLoaderData::new(tensors_loader, 0.8) as &mut dyn DataBatch,
        &final_node(params.models_dir.len() as i64* 6, 6).build_tch_module(&vs_0.root())?,
        |pred: &Tensor, target: &Tensor| {

            let pred_y = pred; //pred.i((.., 0, 0));
            debug!("pred_y ={:?} target={:?}", pred_y.size(), target.size());
            pred_y.cross_entropy_loss::<Tensor>(&target, None, Reduction::Mean, -100, 0.0)

        },
        params.cycles,
        nn::AdamW::default().build(&vs_0, 1.0e-4)?,
        |t:Tensor|{
            debug!("t.size()={:?} ", t.size());
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

    let models = load_corr_and_motif_models(params.models_dir.as_str(), channels *  max_motif_batches as i64 * motif_batch_size, motif_n, channels as usize, device)?
        .into_iter().fold(HashMap::new(), |mut a, v|{
        a.insert(v.label.clone(), v);
        a
    });

    let mut vs = nn::VarStore::new(*device);
    let final_model = Mutex::new(train_nets::final_node(6, 6).build_tch_module(&vs.root())?);
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
                    let t = apply_motifs(motif_n, step, &t, &model.motif_models, motif_batch_size, max_motif_batches, device);
                    let t_x = t.view((1, -1));
                    debug!("infer: motif models model={:?} model.label={} t.size()={:?} t_x.size()={:?}", model, model.label, t.size(), t_x.size());

                    let t = model.corr_model.lock().unwrap().forward(&t_x);
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
struct CorrModels {
    label:String,
    corr_model:Mutex<Box<dyn Module>>,
    motif_models:Vec<Vec<Mutex<Box<dyn Module>>>>
}

impl CorrModels {

    pub fn new(label:String, corr_model: Box<dyn Module>, motif_models: Vec<Vec<Box<dyn Module>>>) -> Self {
        Self {
            label,
            corr_model: Mutex::new(corr_model),
            motif_models: motif_models.into_iter().map(|v|v.into_iter().map(|m|Mutex::new(m)).collect::<Vec<_>>()).collect::<Vec<_>>()
        }
    }

}

/**
* features without x motif nets
*/
fn load_corr_and_motif_models(motif_models_path: impl AsRef<Path>, features: i64, motif_n: i64, channels: usize, device: &Device) -> Result<Vec<CorrModels>, HbacError> {
    let re = Regex::new(r"^(?<label>[a-zA-Z0-9]+?)_(?<corr_run>\d+).safetensors").unwrap();//TODO lazy_static

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
            debug!("corr_run={} ", label);

            let mut vs = nn::VarStore::new(device.clone());

            let motif_nets = load_motif_models(motif_models_path.as_ref(), label.clone(), channels, motif_n, &device)?;

            let max_nets =  motif_nets.iter().map(|nets|nets.len()).max().unwrap_or(0) as i64;
            assert!(motif_nets.len() > 0);

            let features = max_nets * features;

            debug!("load_corr_and_motif_models: corr_full_node max_nets={} features={:?}", max_nets, features);
            let corr_model = train_nets::corr_full_node(features).build_tch_module(&vs.root())?;
            vs.load(entry.path())?;
            vs.freeze();

            models.push(crate::train_nets::CorrModels::new(label, corr_model, motif_nets))
        }

    }
    //TODO should be sorted as labels and same len
    models.sort_by(|a, b|a.label.cmp(&b.label));

    Ok(models)
}

fn load_motif_models(motif_models_path: impl AsRef<Path>, label:String, channels:usize, motif_n: i64, device: &Device) -> Result<Vec<Vec<Box<dyn Module>>>, HbacError> {
    debug!("load_motif_models: motif_models_path={:?} label={:?}, channels={:?}, motif_n={:?}", motif_models_path.as_ref(), label, channels, motif_n);


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
                let node = train_nets::motif_node(motif_n).build_tch_module(&vs.root())?;//TODO window
                vs.load(&motif_path.join(motif))?;
                vs.freeze();
                // debug!("load_models: len={:?}", vs.variables().len());
                // for key in  vs.variables().keys() {
                //     debug!("load_models: key={:?}", key);
                // }

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

    pub fn dummy(output_dir: &String, corr_run_ids:Vec<usize>) ->Self{
        TrainOnMotifsParams {
            motif_ids: MotifsBy::CorrRunId(0),

            corr_run_ids,

            window: 1000,//should be same as one used when finding motifs or TODO center bigger/smaller window

            output_dir: output_dir.clone(),

            cycles: 1000,

            lr: 1.0e-5,

            max_loss: 0.2,

            min_corr: 0.05,
        }
    }
}

pub fn motif_node(window: i64) -> Node {
    SeqBuilder::new(window)
        .batch_norm()
        .conv(10, 1, 1, ConvConfig::default())
        .linear_s(1)
        .func(Tensor::sigmoid)
        .end_seq()
}

pub fn corr_full_node(features: i64) -> Node {
    SeqBuilder::new(features)
        .linear_s(1)
        .func(Tensor::sigmoid)
        .end_seq()
}

pub fn final_node(features: i64, classes_num: i64) -> Node {
    SeqBuilder::new(features)
        .linear_s(classes_num)
        .func(Tensor::relu)
        .linear_s(classes_num)
        // .func(|t|t.softmax(0, Float))
        .end_seq()
}

pub fn train_on_motif(all_slices_0:Vec<Box<Tensor>>, all_slices_1:Vec<Box<Tensor>>, train_on_motif_params: &TrainOnMotifsParams, device: Device, node: &Node) -> Result<(f32, VarStore), HbacError> {
    info!("train_on_motif: device={:?} training on {} slices", device, all_slices_0.len());

    assert!(all_slices_0.len() > 0, "all_slices_0 len");
    assert!(all_slices_1.len() > 0, "all_slices_1 len");

    let vs = nn::VarStore::new(device);

    let net = node.build_tch_module(&vs.root())?;

    let valid_loss = crate::utils::train(
        &device,
        &mut ClassTensorsDataBatch::new(
            vec![
                all_slices_0,
                all_slices_1
            ],
            4,
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

fn apply_motifs(motif_n: i64, step: usize, eeg_t: &Tensor, motif_nets: &Vec<Vec<Mutex<Box<dyn Module>>>>, motif_batch_size: i64, max_motif_batches: usize, device: &Device) -> Tensor {
    let max_nets = motif_nets.iter().map(|nets| nets.len()).max().unwrap_or(0);

    debug!("apply_motifs: max_motif_nets_in_channel={:?} motif_batch_size={:?} max_motif_batches={:?}, ", max_nets, motif_batch_size, max_motif_batches);

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

    #[arg(long, default_value_t = 1000)]
    cycles: usize,

    #[arg(long, default_value_t = 1.0e-5)]
    lr: f64,

    #[arg(long)]
    models_dir: Vec<String>,

    #[arg(long, default_value = "../../data_local/corr_motif_models_on_3_pure/final.safetensors")]
    final_model: String,//TODO

    #[arg(long, default_value = "pure group 10")]
    group: String,
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

    pub fn group(&self) -> &str {
        &self.group
    }

    pub fn dummy(models_dir:Vec<String>, final_model:String, group:String, cycles: usize) ->Self {
        TrainFinalParams {
            cycles,
            lr: 1.0e-5,
            models_dir,
            final_model,
            group
        }
    }
}

#[derive(Debug, Args, Clone)]
pub struct InferParams {

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
}


#[cfg(test)]
mod train_tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use tch::{Device, nn, Tensor};
    use hbac_shared::{HbacError, batch_windows};
    use crate::train_nets::{ClassTensorsDataBatch, load_motif_models, motif_node, TrainOnEEGParams, TrainFinalParams, InferParams};
    use crate::utils::{cdt_for_dir, from};
    use crate::utils::DataBatch;
    use tch::IndexOp;
    use crate::train_nets;
    use log::debug;
    use tch::Kind::{Double, Float};
    use tch::nn::{ConvConfig, LinearConfig, Module, VarStore};
    use crate::builder::SeqBuilder;
    use std::sync::Mutex;
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
        assert_eq!(
            channel_0_motifs, channel_0_t);

        let channel_1_t = apply_channel_motifs(motif_n, step, &motif_nets[1], eeg_t.i((1, ..)), batch_size as i64, max_motif_nets_in_channel as i64, max_motif_batches, &Cpu);

        println!("max_motif_batches={:?}", max_motif_batches);

        let channel_1_motifs = Tensor::stack(&[
                Tensor::from_slice(&[99000., 299000., 499000., 699000., 899000., 1.099e+06, 1.299e+06, 1.499e+06, 1.699e+06, 1.899e+06]),  //ch 1 net 0 (k=1)
                Tensor::from_slice(&[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),  //ch 1 net 0 (k=1)
            ], 0
        ).totype(Float).view((2, 1, 10));
        assert_eq!(
            channel_1_motifs, channel_1_t
        );

        println!("channel_0_motifs.size()={:?}, channel_1_motifs.size()={:?}", channel_0_motifs.size(), channel_1_motifs.size());

        let channels_motifs = super::apply_motifs(motif_n, step, &eeg_t, &motif_nets, batch_size as i64, max_motif_batches, &Cpu);

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

        let channels_motifs_0 = super::apply_motifs(motif_n, step, &eeg_t_0, &motif_nets, motif_batch_size, max_motif_batches, &Cpu);

        println!("channels_motifs_0.size={:?}", channels_motifs_0.size());

        let channels_motifs_1 = super::apply_motifs(motif_n, step, &eeg_t_1, &motif_nets, motif_batch_size, max_motif_batches, &Cpu);

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
    fn test_train_on_eeg() -> Result<(), HbacError> {
        let models_path = Path::new("/tmp/test_train_on_eeg").join(cdt_for_dir());

        let params = TrainOnEEGParams {
            corr_run: 0,

            motif_models_dir: models_path.to_str().unwrap().to_string(),
            eeg_models_dir: models_path.to_str().unwrap().to_string(),
            cycles: 3,
            lr: 1.0e-5,
            labels_datasets: vec![]
        };

        save_test_motif_model(&models_path, "seizure", "0", "00", 100)?;
        save_test_motif_model(&models_path, "seizure", "0", "01", 100)?;
        save_test_motif_model(&models_path, "lpd", "0", "02", 100)?;
        save_test_motif_model(&models_path, "gpd", "0", "03", 100)?;
        save_test_motif_model(&models_path, "lrda", "1", "10", 100)?;
        save_test_motif_model(&models_path, "grda", "1", "11", 100)?;
        save_test_motif_model(&models_path, "other", "2", "20", 100)?;

        super::train_on_eeg(
            &params,
            vec![
                (1, "seizure_vote".to_string()),
                (0, "other".to_string()),
                (0, "gpd_vote".to_string()),
                (0, "lrda_vote".to_string()),
                (0, "grda_vote".to_string()),
                (0, "other_vote".to_string())

            ].into_iter().map(|(label, kind)|{

                (label, (0..13).map( move |id|{
                    load_test_tensor(format!("{}-{}-{}", label, kind, id))
                }))

            }).collect::<Vec<_>>(),
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
            cycles: 3,
            lr: 1.0e-5,
            models_dir: vec![
                models_path_0.to_str().unwrap().to_string(),
                models_path_1.to_str().unwrap().to_string()
            ],
            final_model: test_base.join("final.safetensors").to_str().expect("final_model path").to_string(),
            group: "zzz".into()
        };
        let channels = 5;

        let motif_batch_size = 7;
        let max_motif_batches = 978;//from 684400 in test_train_on_eeg

        let max_nets = 4;
        let features = channels *  max_nets * max_motif_batches as i64 * motif_batch_size;

        debug!("features={}", features);


        save_test_models(&models_path_0, channels, motif_batch_size, max_motif_batches, features)?;
        save_test_models(&models_path_1, channels, motif_batch_size, max_motif_batches, features)?;

        fs::write(&models_path_1.join("unexpected.file"), "lalala");


        super::train_final(
            &params,
            (0..13).map( move |id|{
                (Box::new(make_test_eeg(1000)), Box::new(Tensor::from_slice(&[0.1, 0.1, 0.1, 0.1, 0.3, 0.3])))
            }),
            100,
            100,
            motif_batch_size,
            channels,
            684400,
            &Cpu,
            vec!["other", "lpd", "gpd", "lrda", "grda", "seizure"].into_iter().map(|v|v.into()).collect()
        )?;

        Ok(())
    }

    //6 corrs 4 motif models
    fn save_test_models(models_path: &PathBuf, channels: i64, motif_batch_size: i64, max_motif_batches: i32, features: i64) -> Result<(), HbacError> {
        save_test_corr_model(&models_path, "seizure", features);

        save_test_motif_model(&models_path, "seizure", "0", "00", 100)?;
        save_test_motif_model(&models_path, "seizure", "0", "01", 100)?;
        save_test_motif_model(&models_path, "seizure", "0", "02", 100)?;
        save_test_motif_model(&models_path, "seizure", "0", "03", 100)?;
        save_test_motif_model(&models_path, "seizure", "1", "10", 100)?;
        save_test_motif_model(&models_path, "seizure", "1", "11", 100)?;
        save_test_motif_model(&models_path, "seizure", "2", "20", 100)?;

        let max_nets = 1;
        let features = channels * max_nets * max_motif_batches as i64 * motif_batch_size;
        save_test_corr_model(&models_path, "lpd", features);

        save_test_motif_model(&models_path, "lpd", "0", "00", 100)?;
        save_test_motif_model(&models_path, "lpd", "1", "10", 100)?;
        save_test_motif_model(&models_path, "lpd", "2", "20", 100)?;


        save_test_corr_model(&models_path, "gpd", features);

        save_test_motif_model(&models_path, "gpd", "0", "00", 100)?;
        save_test_motif_model(&models_path, "gpd", "1", "10", 100)?;
        save_test_motif_model(&models_path, "gpd", "2", "20", 100)?;


        save_test_corr_model(&models_path, "lrda", features);

        save_test_motif_model(&models_path, "lrda", "0", "00", 100)?;
        save_test_motif_model(&models_path, "lrda", "1", "10", 100)?;
        save_test_motif_model(&models_path, "lrda", "2", "20", 100)?;


        save_test_corr_model(&models_path, "grda", features);

        save_test_motif_model(&models_path, "grda", "0", "00", 100)?;
        save_test_motif_model(&models_path, "grda", "1", "10", 100)?;
        save_test_motif_model(&models_path, "grda", "2", "20", 100)?;


        save_test_corr_model(&models_path, "other", features);

        save_test_motif_model(&models_path, "other", "0", "00", 100)?;
        save_test_motif_model(&models_path, "other", "1", "10", 100)?;
        save_test_motif_model(&models_path, "other", "2", "20", 100)?;
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
        let max_motif_batches = motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0};

        let models_path = Path::new("/tmp/test_train_final").join(cdt_for_dir());
        let max_nets = 4;
        let features = channels *  max_nets * max_motif_batches as i64 * motif_batch_size;

        debug!("features={}", features);

        save_test_models(&models_path, channels, motif_batch_size, max_motif_batches as i32, features)?;
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
        let test_final_model = train_nets::final_node(features, features).build_tch_module(&vs_0.root())?;

        vs_0.freeze();
        debug!("save_test_final_model: features={} vs_0.variables()={:?}", features, vs_0.variables());
        vs_0.save(models_path.join("final.safetensors"))?;
        Ok(())
    }

    fn save_test_corr_model(models_path: &PathBuf, label:&str, features: i64) -> Result<(), HbacError> {
        fs::create_dir_all(models_path.clone());

        let mut vs_0 = nn::VarStore::new(Device::Cpu);
        let test_corr_model = train_nets::corr_full_node(features).build_tch_module(&vs_0.root())?;

        vs_0.freeze();
        debug!("save_test_corr_model: features={} vs_0.variables()={:?}", features, vs_0.variables());
        vs_0.save(models_path.join(format!("{}_0.safetensors", label)))?;
        Ok(())
    }

    fn save_test_motif_model(models_path: &PathBuf, label: &str, channel: &str, motif_id: &str, window: i64) -> Result<(), HbacError> {
        let channel_0 = models_path.join(format!("{}", label)).join(channel);
        fs::create_dir_all(channel_0.clone());

        let mut vs_0 = nn::VarStore::new(Device::Cpu);
        let test_motif_model = train_nets::motif_node(window).build_tch_module(&vs_0.root())?;

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
    fn test_test(){
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


        let t1 = Instant::now();

        let mut t = Tensor::stack((0..2).map(|_|
            Tensor::stack((0..10).map(|_|
                Tensor::stack((0..100).map(|_|
                                 Tensor::zeros(&[1000], (Double, Cpu)) //batches
                ).collect::<Vec<Tensor>>().as_slice(), 0)
            ).collect::<Vec<Tensor>>().as_slice(), 0)
        ).collect::<Vec<Tensor>>().as_slice(), 0);


        println!("t.size()={:?} elapsed={}", t.size(), t1.elapsed().as_millis());
        let t1 = Instant::now();
        for c in 0..2 {
            for n in 0..10 {
                for b in 0..100 {
                    t = t.index_put(&[Some(Tensor::from_slice(&[c])), Some(Tensor::from_slice(&[n])), Some(Tensor::from_slice(&[b]))], &Tensor::ones(&[1000], (Double, Cpu)), false);
                }
            }
        }

        println!("t.size()={:?} elapsed={}", t.size(), t1.elapsed().as_millis());

        // t.index_put(&[Some(Tensor::from_slice(&[1])), Some(Tensor::from_slice(&[2]))], &Tensor::ones(&[3], (Double, Cpu)), false).print();


    }

}
