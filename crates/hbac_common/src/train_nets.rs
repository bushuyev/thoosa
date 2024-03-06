use hbac_shared::HbacError;
use tch::nn::{ConvConfig, OptimizerConfig, VarStore};
use tch::{Device, IndexOp, Kind, nn, Reduction, Tensor};
use clap::clap_derive::Args;
use log::{debug, info};
use rand::{Rng, thread_rng};
use crate::builder::{Node, SeqBuilder};
use crate::utils::DataBatch;
use crate::utils::from;
use rand::prelude::SliceRandom;

#[derive(Debug, Args)]
pub struct TrainOnEEGParams {

    #[arg(long, default_value = "/tmp/motif_models")]
    motif_models_dir: String,

    #[arg(long, default_value_t = 1000)]
    cycles: usize,

    #[arg(long, default_value_t = 1.0e-5)]
    lr: f64,
}

pub fn train_on_eeg<F>(loader:F, params: &TrainOnEEGParams)-> Result<(), HbacError>
    where F:Fn(&String) -> Box<Tensor> + Sync + Send + 'static {

    Ok(())
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

pub fn motif_node(train_on_motif_params: &TrainOnMotifsParams) -> Node {
    SeqBuilder::new(train_on_motif_params.window() as i64)
        .batch_norm()
        .conv(10, 1, 10, ConvConfig::default())
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
        &ClassTensorsDataBatch::new(
            vec![
                all_slices_0,
                all_slices_1
            ],
            10,
            0.8
        ) as &dyn DataBatch,
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

    fn train_batch(&self) -> (Tensor, Tensor) {
        Self::to_x_y(&self.train, self.batch_size)

    }

    fn valid_batch(&self) -> (Tensor, Tensor) {
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
    use tch::Tensor;
    use hbac_shared::HbacError;
    use crate::train_nets::{ClassTensorsDataBatch, TrainOnEEGParams};
    use crate::utils::from;
    use crate::utils::DataBatch;
    use tch::IndexOp;

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

    #[test]
    fn test_train_on_eeg() -> Result<(), HbacError>{

        super::train_on_eeg(

            |id| Box::new(Tensor::from_slice(&[1., 0., 2.])),
            &TrainOnEEGParams {
                motif_models_dir: "/tmp/dummy".into(),
                cycles: 10,
                lr: 1.0e-5
            }
        )?;

        Ok(())
    }
}
