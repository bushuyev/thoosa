use std::collections::{HashMap, VecDeque};
use std::fs;
use std::marker::PhantomData;
use hbac_shared::HbacError;
use tch::nn::{ConvConfig, Module, OptimizerConfig, VarStore};
use tch::{Device, IndexOp, Kind, nn, Reduction, Tensor};
use clap::clap_derive::Args;
use log::{debug, info};
use rand::{Rng, thread_rng};
use crate::builder::{Node, SeqBuilder};
use crate::utils::{DataBatch, train};
use crate::utils::from;
use rand::prelude::SliceRandom;
use rayon::iter::ParallelBridge;
use rayon::iter::ParallelIterator;
use crate::train_nets;
use std::path::Path;
use tch::Device::Cpu;
use tch::Kind::{Double, Float};

#[derive(Debug, Args)]
pub struct TrainOnEEGParams {

    #[arg(long, default_value = "/tmp/motif_models")]
    motif_models_dir: String,

    #[arg(long, default_value_t = 1000)]
    cycles: usize,

    #[arg(long, default_value_t = 1000)]
    features: i64,

    #[arg(long, default_value_t = 1.0e-5)]
    lr: f64,
}


fn run_motif_nets<I>(loader: I) where I:Iterator<Item=Result<Tensor, HbacError>>, I: std::marker::Send {
    for eeg in loader {

    }
}

struct ClassTensorsLoaderDataBatch<I> {
    classes_tensor_loaders: Vec<Vec<I>>,
    // cache:VecDeque<(Tensor, Tensor)>, TODO
    batch_size: usize
}

impl <I> ClassTensorsLoaderDataBatch<I> where I:Iterator<Item=Result<Tensor, HbacError>> + Send + Sync {

    pub fn new(classes_tensor_loaders: Vec<Vec<I>>, batch_size: usize, train_valid_split: f32) -> Self {
        Self {
            classes_tensor_loaders,
            // cache: VecDeque::with_capacity(batch_size),
            batch_size
        }
    }

}

impl <I> DataBatch for ClassTensorsLoaderDataBatch<I> where I:Iterator<Item=Result<Tensor, HbacError>> + Send + Sync {

    fn train_batch(&mut self) -> (Tensor, Tensor){

        let label_tensors = self.classes_tensor_loaders.iter_mut().enumerate().map(|(label, loaders)|{
            let take = self.batch_size/loaders.len() + 1;
            (label as i64, loaders.iter_mut()
                .flat_map(|loader|loader.take(take))
                .map(|t|t.unwrap())
                .collect::<Vec<Tensor>>())
        }).collect::<Vec<(i64, Vec<Tensor>)>>();

        debug!("label_tensors={} label_tensors={:?}", label_tensors.len(), label_tensors);

        let x_y = label_tensors.into_iter().fold((vec![], vec![]), |mut a, v|{

            (0..v.1.len()).for_each(|_|{a.1.push(v.0);});//label  - y
            a.0.extend(v.1);//tensox - x

            a
        });

            (
                Tensor::stack(x_y.0.as_slice(), 0).totype(Float),
                Tensor::from_slice(x_y.1.as_slice()).totype(Float)
            )
    }

    fn valid_batch(&mut self) -> (Tensor, Tensor){
        todo!()
    }
}

fn apply_motif_models(eeg:Tensor, motif_models:&Vec<Vec<Box<dyn Module>>>) -> Tensor {

    todo!()
}

pub fn train_on_eeg<I>(params: &TrainOnEEGParams, labels_loaders:Vec<(i64, I)>)-> Result<(), HbacError>
    where I:Iterator<Item=Result<Tensor, HbacError>> + Send + Sync {
    let models = load_models(params, 20)?;//TODO num channels

    let mut labels_loaders = labels_loaders.into_iter().fold(HashMap::new(), |mut a, (label, loader)|{
        a.entry(label).or_insert_with(||vec![]).push(loader);
        a
    });

    let mut keys = labels_loaders.keys().map(|v|*v).collect::<Vec<i64>>();
    keys.sort();

    debug!("train_on_eeg: keys={:?} labels_loaders={:?}", keys, labels_loaders.keys());

    let loaders_vec = keys.into_iter().map(|label|labels_loaders.remove(&label).expect(format!("no {}", label).as_str())).collect::<Vec<Vec<I>>>();

    let mut ds = ClassTensorsLoaderDataBatch::new(loaders_vec, 3, 0.8);

    // let tb = ds.train_batch();
    // println!("tb={:?} tb={:?}", tb.0.size(), tb.1.size());
    let mut vs_0 = nn::VarStore::new(Device::Cpu);

    train(
        &Cpu, //TODO
        &mut ds as &mut dyn DataBatch,
        &full_node(3).build_tch_module(&vs_0.root())?,
        |pred: &Tensor, target: &Tensor| {

            let pred_y = pred.i((.., 0, 0));
            debug!("pred_y ={:?} ", from::<Vec<f32>>(&pred_y));
            pred_y.binary_cross_entropy::<Tensor>(&target.totype(Kind::Float), None, Reduction::Mean)
        },
        3,
        nn::AdamW::default().build(&vs_0, 1.0e-4)?
    )?;

    // labels_loaders.into_iter().par_bridge().try_for_each(|(label, loader)|{
    //     // run_motif_nets(loader);
    //     for eeg in loader {
    //
    //     }
    //
    //     Ok::<(), HbacError>(())
    // });


    Ok(())
}

fn load_models(params: &TrainOnEEGParams, channels:usize) -> Result<Vec<Vec<Box<dyn Module>>>, HbacError> {
    debug!("load_models: channel_motifs_dir={:?}", params.motif_models_dir);

    let motif_models_path  = Path::new(params.motif_models_dir.as_str());

    let mut channels_motifs_models = vec![];

    for channel in 0..channels {

        let motif_path = motif_models_path.join(channel.to_string());
        let mut channels_motifs:Vec<Box<dyn Module>> = vec![];

        if motif_path.is_dir() {
            let mut channel_motifs = fs::read_dir(&motif_path)?
                .map(|res| res.map(|e| e.file_name()))
                .collect::<Result<Vec<_>, std::io::Error>>()?;

            channel_motifs.sort();

            for motif in channel_motifs {
                debug!("load_models: motif={:?}", motif);

                let mut vs = nn::VarStore::new(Device::Cpu);
                let node = train_nets::motif_node(params.features).build_tch_module(&vs.root())?;//TODO window

                vs.load(&motif_path.join(motif))?;

                channels_motifs.push(node);
            }
        }

        channels_motifs_models.push(channels_motifs);
    }
    Ok(channels_motifs_models)

}

#[derive(Debug, Args)]
pub struct TrainOnMotifsParams {
    #[arg(long)]
    motif_ids: MotifsBy,

    #[arg(long,  default_value_t = 1000)]
    window: usize,//should be same as one used when finding motifs or TODO center bigger/smaller window

    #[arg(long, default_value = "/tmp/motif_models")]
    output_dir: String,

    #[arg(long, default_value_t = 1000)]
    cycles: usize,

    #[arg(long, default_value_t = 1.0e-5)]
    lr: f64,
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
}

pub fn motif_node(window: i64) -> Node {
    SeqBuilder::new(window)
        .batch_norm()
        .conv(10, 1, 10, ConvConfig::default())
        .linear_s(1)
        .func(Tensor::sigmoid)
        .end_seq()
}

pub fn full_node(nets_len: i64) -> Node {
    SeqBuilder::new(nets_len)
        .linear_s(1)
        .func(Tensor::sigmoid)
        .end_seq()
}

pub fn train_on_motif(all_slices_0:Vec<Box<Tensor>>, all_slices_1:Vec<Box<Tensor>>, train_on_motif_params: &TrainOnMotifsParams, device: Device, node: &Node) -> Result<(f32, VarStore), HbacError> {


    let vs = nn::VarStore::new(device);

    let net = node.build_tch_module(&vs.root())?;

    info!("train_on_motif: training on {} slices", all_slices_0.len());

    let valid_loss = crate::utils::train(
        &Device::Cpu,
        &mut ClassTensorsDataBatch::new(
            vec![
                all_slices_0,
                all_slices_1
            ],
            10,
            0.8
        ) as &mut dyn DataBatch,
        &net,
        |pred: &Tensor, target: &Tensor| {
            // pred.cross_entropy_loss::<Tensor>(&target.totype(Kind::Int64), None, Reduction::Mean, -100, 0.0)
            let pred_y = pred.i((.., 0, 0));
            debug!("pred_y ={:?} ", from::<Vec<f32>>(&pred_y));
            pred_y.binary_cross_entropy::<Tensor>(&target.totype(Kind::Float), None, Reduction::Mean)
        },
        train_on_motif_params.cycles,
        nn::AdamW::default().build(&vs, train_on_motif_params.lr)?
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
    batch_size: usize
}

impl ClassTensorsDataBatch {

    pub fn new(classes_tensors: Vec<Vec<Box<Tensor>>>, batch_size: usize, train_valid_split:f32) -> Self {
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
        }
    }

    fn to_x_y(class_tensors: &Vec<Vec<Box<Tensor>>>, batch_size: usize) -> (Tensor, Tensor) {
        let mut rng = thread_rng();

        let per_class_len = batch_size / class_tensors.len();//TODO make balanced

        debug!("ClassTensorsDataBatch::to_x_y: per_class_len={}", per_class_len);

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

        let classes = Tensor::from_slice(x_y_vecs.iter().map(|e| e.1).collect::<Vec<_>>().as_slice());

        let tensor = Tensor::stack(
            x_y_vecs.into_iter().map(|e|e.0).collect::<Vec<_>>().as_slice(),
            0,
        ).view((batch_size as i64, 1, -1));

        debug!("ClassTensorsDataBatch::to_x_y: tensor.size()={:?}", tensor.size());

        (
            tensor,
            classes
        )
    }
}

impl DataBatch for ClassTensorsDataBatch {

    fn train_batch(&mut self) -> (Tensor, Tensor) {
        Self::to_x_y(&self.train, self.batch_size)
    }

    fn valid_batch(&mut self) -> (Tensor, Tensor) {
        Self::to_x_y(&self.val, self.batch_size)
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


#[cfg(test)]
mod train_tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use tch::{Device, nn, Tensor};
    use hbac_shared::HbacError;
    use crate::train_nets::{ClassTensorsDataBatch, motif_node, TrainOnEEGParams};
    use crate::utils::{cdt_for_dir, from};
    use crate::utils::DataBatch;
    use tch::IndexOp;
    use crate::train_nets;
    use log::debug;
    use tch::Kind::Double;
    use tch::nn::{ConvConfig, Module, VarStore};
    use crate::builder::SeqBuilder;

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
            0.5
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

    fn test_net(num_features: i64) -> Result<Box<dyn Module>, HbacError> {
        let mut vs = nn::VarStore::new(Device::Cpu);

        SeqBuilder::new(num_features)
            .batch_norm()
            .conv(num_features/10, 1, 1, ConvConfig::default())
            .linear_s(1)
            .func(Tensor::sigmoid)
            .end_seq().build_tch_module(&vs.root())
    }


    #[test]
    fn test_apply_motif_models() -> Result<(), HbacError>{
        let input_n = 1000;
        let motif_n = 100;
        let eeg = Tensor::vstack(&[
            Tensor::from_slice((0..input_n).map(|i|i as f32).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((0..input_n).map(|i|i as f32 * 2.).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((0..input_n).map(|i|i as f32 * 3.).collect::<Vec<_>>().as_slice())
        ]);

        // let channel_0_models = vec![
        //     test_net(motif_n)
        // ];
        // apply_motif_models(
        //
        // )
        let step = (motif_n / 4) as usize;
        let chunks = (0..=(input_n - motif_n) as usize).step_by(step ).array_chunks::<10>();

        // test_net(motif_n)?.forward(&)

        Ok(())
    }

    #[test]
    fn test_train_on_eeg() -> Result<(), HbacError>{

        fn test_load_tensor(id:String)->Result<Tensor, HbacError> {
            debug!("test_load_tensor: {}", id);
            Ok(Tensor::from_slice(&[0., 0., 0., 1.]))
        }

        let models_path = Path::new("/tmp/test_train_on_eeg").join(cdt_for_dir());
        let channel_0 = models_path.join("0");
        fs::create_dir_all(channel_0.clone());

        let mut vs_0 = nn::VarStore::new(Device::Cpu);
        let test_motif_model = train_nets::motif_node(1000).build_tch_module(&vs_0.root())?;
        vs_0.freeze();
        vs_0.save(channel_0.join("test_motif.safetensors"))?;

        super::train_on_eeg(

            &TrainOnEEGParams {
                motif_models_dir: models_path.to_str().unwrap().to_string(),
                cycles: 10,
                features: 1000,
                lr: 1.0e-5
            },

            vec![
                (1, "seizure_vote".to_string()),
                (0, "lpd_vote".to_string()),
                (0, "gpd_vote".to_string()),
                (0, "lrda_vote".to_string()),
                (0, "grda_vote".to_string()),
                (0, "other_vote".to_string())

            ].into_iter().map(|(label, kind)|{

                (label, vec!["123".to_string(), "234".to_string()].into_iter().map(move |id|{

                    test_load_tensor(format!("{}-{}-{}", label, kind, id))
                }))

            }).collect::<Vec<_>>()

        )?;

        /*

        vec![
                ( vec!["id1", "id2"].into_iter().map(|id|
                    eeg_ids.into_iter().map(|eeg_id| {//TODO CacheOnThreads
                        eeg_to_tensor(base_path, eeg_id.as_str(), *device)
                    }))
            ]
         */
        Ok(())
    }
}
