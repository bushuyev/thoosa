use core::fmt;
use std::borrow::Borrow;
use std::fmt::{Debug, Formatter};
use std::sync::Mutex;

use log::{debug, error, warn};
use tch::{Device, IndexOp, Kind, Tensor};
use tch::Kind::Float;
use tch::nn::{Init, LinearConfig, Module, Path};
use tch::nn::ConvConfig;

use hbac_shared::*;
use hbac_shared::HbacError;

use crate::builder::Node;
// use crate::{data, utils};
use crate::builder::SeqBuilder;
use crate::utils::*;

#[derive(Debug)]
pub struct MySequential {
    layers: Vec<Box<dyn Module>>,
}

impl MySequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }
}

impl Module for MySequential {
    fn forward(&self, xs: &Tensor) -> Tensor {
        if self.layers.is_empty() {
            xs.shallow_clone()
        } else {
            debug!("MySequential::forward: xs.size={:?}", xs.size());
            let xs = self.layers[0].forward(xs);
            self.layers.iter().skip(1).fold(xs, |xs, layer| layer.forward(&xs))
        }
    }
}

pub struct MyParallel {
    layers: Vec<Box<dyn Module>>,
    split:Box<dyn Fn(&Tensor)->Vec<Tensor> + Send + Sync>,
    join:Box<dyn Fn(Vec<Tensor>)->Tensor + Send + Sync>,
}

impl Debug for MyParallel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("")
            .field("layers", &self.layers)
            .finish()
    }
}

impl crate::my_tch::MyParallel {

    pub fn new(layers: Vec<Box<dyn Module>>, split:Box<dyn Fn(&Tensor)->Vec<Tensor> + Send + Sync>, join:Box<dyn Fn(Vec<Tensor>)->Tensor + Send + Sync>) -> Self {

        assert!(layers.len() > 0, "layers len > 0");

        Self {
            layers,
            split,
            join
        }
    }
}

impl Module for crate::my_tch::MyParallel {
    fn forward(&self, xs: &Tensor) -> Tensor {
        debug!("MyParallel::forward: xs={:?}", xs.size());
        let t = (self.join)(
            (self.split)(xs).into_iter().zip(self.layers.iter()).map(|(t, net)| {
                net.forward(&t)
            }).collect::<Vec<Tensor>>()
        );

        debug!("MyParallel::forward: t={:?}", t.size());

        t

    }
}

#[derive(Debug)]
pub struct MyMaxPool1d {
    kernel_size: i64,
    stride: i64,
    padding: i64,
    dilation: i64,
    ceil_mode: bool,
}

impl MyMaxPool1d {
    pub fn new(kernel_size: i64, stride: i64, padding: i64) -> Self {
        Self {
            kernel_size,
            stride, padding,
            dilation: 1,
            ceil_mode: false
        }
    }
}

impl Module for MyMaxPool1d {

    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.max_pool1d(self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)
    }
}



#[derive(Debug)]
pub struct MyBatchNorm1d {

}

impl crate::my_tch::MyBatchNorm1d {
    pub fn new() -> Self {
        Self {

        }
    }
}

impl Module for crate::my_tch::MyBatchNorm1d {

    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.batch_norm::<Tensor>(None, None, None, None, true, 0.1, 1.0e-05, false)
    }
}



#[derive(Debug)]
pub struct MyAvgPool1d {
    kernel_size: i64,
    stride: i64,
    padding: i64,
    ceil_mode: bool,
    count_include_pad: bool
}

impl crate::my_tch::MyAvgPool1d {
    pub fn new(kernel_size: i64, stride: i64, padding: i64) -> Self {
        Self {
            kernel_size,
            stride, padding,
            ceil_mode: false,
            count_include_pad: false
        }
    }
}

impl Module for MyAvgPool1d {

    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.avg_pool1d(self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
    }
}


pub type Fp = fn(&Tensor)->Tensor;
pub type LossFp = fn(&Tensor, &Tensor)->Tensor;


pub struct MyFuncModule {
    func: Fp
}

impl Debug for MyFuncModule {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MyFuncModule")
            .finish()
    }
}

impl  MyFuncModule {

    pub fn new(func: Fp) -> Self{
        Self { func }
    }

}

impl Module for MyFuncModule {
    fn forward(&self, xs: &Tensor) -> Tensor {
        (self.func)(xs)
    }
}


pub struct MyClosureModule {
    func: Box<dyn Fn(&Tensor)->Tensor + Send>
}

impl MyClosureModule {
    pub fn new(func: Box<impl Fn(&Tensor) -> Tensor  + Send + 'static>) -> Self {
        Self { func }
    }
}

impl Debug for MyClosureModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("MyClosureModule")
            .finish()
    }
}


impl Module for MyClosureModule {
    fn forward(&self, xs: &Tensor) -> Tensor {
        (self.func)(xs)
    }
}

#[derive(Debug)]
pub struct MyLinear {
    pub ws: Tensor,
    pub bs: Option<Tensor>,
}

impl Module for MyLinear {
    fn forward(&self, xs: &Tensor) -> Tensor {
        match xs.f_linear(&self.ws, self.bs.as_ref()) {//TODO error
            Ok(t) => t,
            Err(e) => {
                error!("{:?}", e);
                panic!("{:?}", e);
            }
        }
    }
}

pub fn my_linear<'a, T: Borrow<Path<'a>>>(
    vs: T,
    in_dim: i64,
    out_dim: i64,
    c: LinearConfig,
) -> Result<MyLinear, HbacError> {
    assert!(in_dim > 0);
    assert!(out_dim > 0);

    let vs = vs.borrow();
    let bs = if c.bias {
        let bs_init = c.bs_init.unwrap_or_else(|| {
            let bound = 1.0 / (in_dim as f64).sqrt();
            Init::Uniform { lo: -bound, up: bound }
        });
        Some(vs.f_var("bias", &[out_dim], bs_init)?)
    } else {
        None
    };

    Ok(MyLinear { ws: vs.f_var("weight", &[out_dim, in_dim], c.ws_init)?, bs })
}

pub fn configure_dense(num_features: i64, num_blocks: usize, num_layers_in_block: usize, kernel: i64, channels: i64) -> Node  {
    let mut builder = SeqBuilder::new(num_features);

    let dense_block = |b:&mut SeqBuilder| b
        .n_blocks(
            |b:&mut SeqBuilder, _i| b
                .func(Tensor::elu)
                .conv_1d(kernel, channels, channels, ConvConfig::default())
                .func(Tensor::elu)
                .conv_1d(kernel, channels, channels, ConvConfig::default())
                .end(),
            num_layers_in_block)
        .end();


    builder
        .conv_1d(kernel, channels, channels, ConvConfig::default())
        .func(|t|t.elu())
        .max_pool( 3, 2, 1)
        .n_blocks(|b:&mut SeqBuilder, _i| b
            .block(dense_block)
            .block(|b:&mut SeqBuilder| b
                .func(Tensor::elu)
                .conv_1d(kernel, channels, channels, ConvConfig::default())
                .func(Tensor::elu)
                .conv_1d(kernel, channels, channels, ConvConfig::default())
                .end())
            .end(), num_blocks)
        .block(dense_block)
        .func(Tensor::relu)
        .avg_pool(7, 3, 0)
        .dropout()
        .linear_s(6)
        .end_seq()

}


// fn train(base_path:&str, device:Device) -> Result<(), HbacError> {
//
//
//     let good_eegs = EEGs::good_eegs(base_path)?;
//
//     let t1 = Instant::now();
//     let eegs = EEGs::load_eegs (
//         base_path,
//         // None,
//         // Some(col("patient_id").is_in(lit(Series::new("eegs_", &[44615, 8033, 2641, 10111, 30539, 57503, 13252, 17637, 38015, 58327])))),
//         Some(col("eeg_id").is_in(lit(good_eegs))),
//         Some(data::ByPatientIdSplitter::new(0.7)),
//     )?;
//
//
//     debug!("eegs={:?}", eegs.len());
//
//     debug!("done in {}", t1.elapsed().as_millis());
//     let vs = nn::VarStore::new(device.clone());
//
//
//
//     for num_blocks in [16, 32, 64] { //[2, 4, 6, 8, 16]
//
//         for kernel in [3, 5, 10, 20, 50, 100] { //[5, 7, 10, 20, 50, 100]
//         debug!("---------------------------------------num_blocks={:?} kernel={:?}", num_blocks, kernel);
//
//             match configure_dense(50 * 200, num_blocks, 4, kernel, 20).build_tch_module(&vs.root()) {
//                 Ok(model) => {
//                     match utils::train(
//                         &device,
//                         eegs.as_ref() as &dyn DataBatch,
//                         model,
//                         &vs,
//                         |pred: &Tensor, target: &Tensor| {
//                             pred.cross_entropy_loss::<Tensor>(target, None, Reduction::Mean, -100, 0.0)
//                         }
//                     ){
//                         Ok(_) => {
//                             debug!("num_blocks={:?} kernel={:?} done", num_blocks, kernel);
//                         }
//                         Err(e) => {
//                             debug!("num_blocks={:?} kernel={:?} train failed: {:?}", num_blocks, kernel, e);
//                         }
//                     }
//                 }
//                 Err(e) => {
//                     debug!("num_blocks={:?} kernel={:?} build_tch_module failed: {:?}", num_blocks, kernel, e);
//                 }
//             };
//
//         }
//     }
//
//     Ok(())
//
// }

#[allow(dead_code)]
fn load_batch(device: &Device, batch_size:i64) -> (Tensor, Tensor) {
    (
        tch::Tensor::randn([batch_size, 50 * 200, 20], (Kind::Float, device.clone())), //batch, 20 EEG feats, 50 sec * 200 samples per second,
        tch::Tensor::randn([batch_size, 6], (Kind::Float, device.clone())) //batch, seizure_vote,lpd_vote,gpd_vote, lrda_vote, grda_vote,other_vote, 50 sec * 200 samples per second,
    )
}

pub fn apply_channel_motifs(motif_n: i64, step: usize, nets: &Vec<Mutex<Box<dyn Module>>>, eeg_t_channel: Tensor, motif_batch_size: i64, max_nets: i64, max_motif_batches: usize, device: &Device) -> Tensor {//pad_motifs_to- to have num motifs in all channels same

    let input_t_size = eeg_t_channel.size();
    debug!("apply_channel_motifs: eeg_t_channel.size()={:?} input_t_size={:?} nets.len()={:?}", eeg_t_channel.size(), input_t_size, nets.len());

    assert!(input_t_size.len() == 1);


    let input_n = input_t_size[0];
    let folds = batch_windows(input_n as usize, motif_n as usize, step, motif_batch_size as usize);

    debug!("nput_n={:?}, folds.0.len()={:?}, folds.1.len()={:?}", input_n, folds.0.len(), folds.1.len());


    if nets.len() == 0 {
        //if no 'main' folds build motifs on remainder
        return Tensor::zeros([max_nets as i64, max_motif_batches as i64, motif_batch_size as i64], (Float, *device));
    }

    let nets_len = nets.len();
    let mut net_batches = (0..nets_len).map(|_| Vec::with_capacity(((input_n - motif_n) as usize) / step)).collect::<Vec<Vec<Tensor>>>();


    for slice_0_idx in folds.0 {
        debug!("apply_channel_motifs: slice_0_idx={:?}", slice_0_idx);

        let batch = Tensor::stack(slice_0_idx.into_iter().map(|i| eeg_t_channel.i(i as i64..i as i64 + motif_n)).collect::<Vec<Tensor>>().as_slice(), 0);
        debug!("apply_channel_motifs: batch.size={:?} motif_batch_size={:?} motif_n={:?}", batch.size(), motif_batch_size, motif_n);

        let batch = batch.view((motif_batch_size, 1, motif_n));

        for ((_net_i, net), net_batch) in nets.iter().enumerate().zip(net_batches.iter_mut()) {

            let net_rs = net.lock().unwrap().forward(&batch).squeeze();
            debug!("apply_channel_motifs: batch.size()={:?} net_rs size={:?} net_rs_vals={:?}", batch.size(), net_rs.size(), from::<Vec<f32>>(&net_rs));
            add_net_batch(motif_batch_size, net_batch, net_rs);
        };
    }

    if folds.1.len() > 0 {
        debug!("apply_channel_motifs: reminder={:?}", folds.1);

        let batch = Tensor::stack(folds.1.into_iter().map(|i| eeg_t_channel.i(i as i64..i as i64 + motif_n)).collect::<Vec<Tensor>>().as_slice(), 0);
        debug!("apply_channel_motifs: reminder, batch.size={:?} motif_batch_size={:?} motif_n={:?}", batch.size(), motif_batch_size, motif_n);

        let batch = batch.view((-1, 1, motif_n));

        for ((net_i, net), net_batch) in nets.iter().enumerate().zip(net_batches.iter_mut()) {

            let net_rs = net.lock().unwrap().forward(&batch).set_requires_grad(false).squeeze();

            if net_rs.size().len() > 0 {
                add_net_batch(motif_batch_size, net_batch, net_rs);

            } else {
                warn!("apply_channel_motifs: reminder, no rs for net_i={} batch.size()={:?}", net_i, batch.size());
            }
        };
    }

    //nets x slices in tensor/motif_batch_size x motif_batch_size
    net_batches.iter().enumerate().for_each(|(i, net)| {
        debug!("apply_channel_motifs: net={:?} sizes={:?}", i, net.iter().map(|t|t.size()).collect::<Vec<Vec<i64>>>());
    });
    let net_tensors = net_batches.into_iter().map(|net_batches| Tensor::stack(net_batches.as_slice(), 0)).collect::<Vec<Tensor>>();
    debug!("apply_channel_motifs: net_tensors.len()={:?} sizes={:?}", net_tensors.len(), net_tensors.iter().map(|t|t.size()).collect::<Vec<Vec<i64>>>());

    let nets_t = Tensor::stack(
        net_tensors.as_slice(),
        0
    );
    debug!("apply_channel_motifs: 2.2 nets_t.zie={:?}", nets_t.size());

    // let nets_t = nets_t.view((-1, nets_t.size()[2])); //(x, 1, y) -> (x,y)


    debug!("apply_channel_motifs: 1. nets_t={:?} max_nets={}, max_motif_batches={} motif_batch_size={}", nets_t.size(), max_nets, max_motif_batches, motif_batch_size);

    let nets_t = pad_to(&nets_t, &[max_nets as i64, max_motif_batches as i64, motif_batch_size as i64]).unwrap();

    debug!("apply_channel_motifs: 2. nets_t={:?}", nets_t.size());
    nets_t.set_requires_grad(false)
}

fn add_net_batch(motif_batch_size: i64, net_batch: &mut Vec<Tensor>, net_rs: Tensor) {
    if net_rs.size()[0] < motif_batch_size as i64 {
        net_batch.push(
            pad_to(&net_rs, &[motif_batch_size as i64]).expect("pad_to failed")
        );
    } else {
        net_batch.push(
            net_rs
        );
    }
}


#[cfg(test)]
mod tch_tests {
    use std::fs;
    use std::sync::Arc;
    use std::sync::Mutex;

    use chrono::Utc;
    use log::{debug, LevelFilter};
    use plotters::prelude::{BLACK, GREEN, RED};
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    use serde::{Deserialize, Serialize};
    use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger};
    use tch::{Device, IndexOp, Kind, nn, Reduction, Tensor};
    use tch::Device::Cpu;
    use tch::nn::{ConvConfig, Module, OptimizerConfig, VarStore};

    use hbac_shared::*;
    use hbac_shared::HbacError;

    use crate::builder::clc_conv1d_out;
    use crate::builder::SeqBuilder;
    use crate::utils;
    use crate::utils::*;

    #[ctor::ctor]
    fn init() {
        CombinedLogger::init(
            vec![
                TermLogger::new(LevelFilter::Debug, Config::default(), TerminalMode::Mixed, ColorChoice::Auto)
            ]
        ).unwrap_or_else(|e|{
            println!("CombinedLogger::init failed: {:?}", e);
        });


        std::env::set_var("POLARS_FMT_MAX_COLS", "100");
        std::env::set_var("POLARS_FMT_MAX_ROWS", "100");
    }

    #[test]
    pub fn test_conv1_out(){
        let l_out = clc_conv1d_out(50 * 200, 5, ConvConfig::default());

        assert_eq!(l_out, 9996);
    }

    #[test]
    pub fn test_seq_linear_same_size(){
        let seq = SeqBuilder::new(2)
            .linear()
            .end_seq();

        let vs = nn::VarStore::new(Device::Cpu);

        let net = seq.build_tch_module(&vs.root()).unwrap();

        let y_p = net.forward(&tch::Tensor::randn([10, 2], (Kind::Float, Device::Cpu)));

        assert_eq!(vec![10,2], y_p.size())
    }

    #[test]
    pub fn test_split_seq() -> Result<(), HbacError>{
        let res_seq = SeqBuilder::new(10)
            .linear_s(1)
            .end_seq();

        let conv_seq = SeqBuilder::new(10)
            .conv_1d(5, 1, 1, ConvConfig::default())
            .linear_s(1)
            .end_seq();

        let main = SeqBuilder::new(10)
            .split(
                vec![res_seq, conv_seq],
                Box::new(|t:&Tensor|vec![t.i((.., 0, ..)).view((10, 1, 10)), t.i((.., 1, ..)).view((10, 1, 10))]),
                Box::new(|t:Vec<Tensor>|Tensor::stack(t.as_slice(), 1).squeeze())

            ).end_seq();

        let vs = nn::VarStore::new(Device::Cpu);

        let net = main.build_tch_module(&vs.root())?;


        let y_p = net.forward(&tch::Tensor::randn([10, 2, 10], (Kind::Float, Device::Cpu)));


        assert_eq!(vec![10,2], y_p.size());

        Ok(())
    }


    #[test]
    pub fn test_simp()-> Result<(), HbacError>{

        let mut vs_0 = nn::VarStore::new(Device::Cpu);

        let n = 1000;
        let net_0 = test_net(&vs_0, n)?;



        // let t_0 = Tensor::from_slice(sin_v(1000, 20.).as_slice());
        let t_0 = Tensor::from_slice(pulse_in_rand(n, &[], 0.05, 1., 0.1).as_slice());
        let t_1 = Tensor::from_slice(sin_v(n, 22.).as_slice());


        utils::train(
            &Device::Cpu,
            &mut crate::simp::DummyDs::new(
                t_0.shallow_clone(),
                t_1.shallow_clone(),
            ) as &mut dyn DataBatch,
            &net_0,
            |pred: &Tensor, target: &Tensor| {
                // pred.cross_entropy_loss::<Tensor>(&target.totype(Kind::Int64), None, Reduction::Mean, -100, 0.0)

                let pred_y = pred.i((.., 0, 0));
                debug!("pred_y ={:?} ", from::<Vec<f32>>(&pred_y));
                pred_y.binary_cross_entropy::<Tensor>(&target.totype(Kind::Float), None, Reduction::Mean)
            },
            n as usize,
            nn::AdamW::default().build(&vs_0, 1.0e-4)?,
            |t|t
        )?;



        let batch_0 = Tensor::stack((0..10).map(|_| t_0.shallow_clone()).collect::<Vec<Tensor>>().as_slice(), 0).view((10, 1, n));
        let batch_1 = Tensor::stack((0..10).map(|_| t_1.shallow_clone()).collect::<Vec<Tensor>>().as_slice(), 0).view((10, 1, n));
        println!("net t_0={:?}, t_1={:?}",
             from::<Vec<f32>>(&net_0.forward(&batch_0).view(-1)),
             from::<Vec<f32>>(&net_0.forward(&batch_1).view(-1))
        );

        vs_0.freeze();
        vs_0.save("/tmp/test_vars.safetensors")?;

        let mut vs_1 = nn::VarStore::new(Device::Cpu);

        let net_1 = test_net(&vs_1, n)?;
        vs_1.load("/tmp/test_vars.safetensors")?;

        println!("net 1 t_0={:?}, t_1={:?}",
             from::<Vec<f32>>(&net_1.forward(&batch_0).view(-1)),
             from::<Vec<f32>>(&net_1.forward(&batch_1).view(-1))
        );

        assert_eq!(from::<Vec<f32>>(&net_0.forward(&batch_0).view(-1)), from::<Vec<f32>>(&net_1.forward(&batch_0).view(-1)));
        assert_eq!(from::<Vec<f32>>(&net_0.forward(&batch_1).view(-1)), from::<Vec<f32>>(&net_1.forward(&batch_1).view(-1)));


        Ok(())
    }

    fn test_net(vs: &VarStore, num_features: i64) -> Result<Box<dyn Module>, HbacError> {
        SeqBuilder::new(num_features)
            .batch_norm()
            .conv_1d(num_features/10, 1, 1, ConvConfig::default())
            .linear_s(1)
            .func(Tensor::sigmoid)
            .end_seq().build_tch_module(&vs.root())
    }

    #[test]
    pub fn test_plot_eeg() -> Result<(), HbacError>{

        let g_n = 5;
        // let g = gauss_i(2.5, 2.0, g_n);
        let g = gauss_i( (g_n as f32 - 1.0) /2.0, 1.0, g_n);
        let _g_v = g.map(|v|v * 0.3).collect::<Vec<f32>>();

        let t = load_tensor("/media/bu/KINGSTON/all_eegs.pt/all/", "1221753362", Cpu)?;//2259539799

        println!("t={:?}", t.size());
        let t = t.i((1..3, ..)).view((2, -1));//.view((1, -1));

        println!("t={:?}", t.size());

        let x = lin_t(t.size()[1]);

        // let t_f1 = t.conv1d(&Tensor::from_slice(g_v.as_slice()).totype(Float).view((1, 1, -1)), None::<Tensor>, 1, 0, 1, 1);
        let t_f1 = t.avg_pool1d(10, 1, 0, false, false);
        let t_f100 = t.avg_pool1d(20, 1, 0, false, false);

        println!("t_f1={:?}", t_f1.size());
        let n = t.size()[1] as u32;//11; //

        // let channels_motifs_models = crate::train_nets::load_motif_models(
        //     "/home/bu/dvl/kaggle/HMS_Harmful_Brain_Activity_Classification/data_down/200_model_08_max_ratio_dist/0_200/models/",
        //     "gpd".into(),
        //     20,
        //     1000,
        //     &Cpu
        // )?;
        // let motif_nets: Vec<Vec<Mutex<Box<dyn Module>>>> = channels_motifs_models.into_iter().map(|v|v.into_iter().map(|m|Mutex::new(m)).collect::<Vec<_>>()).collect::<Vec<_>>();

        // println!("nets_in_channel={:?}", motif_nets.iter().map(|nets_in_channel|nets_in_channel.len()).collect::<Vec<usize>>());

        // let motif_batches = batch_windows(684400, 1000, 100, 100);
        // let max_motif_batches = motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0};



        // let motifs_t = super::apply_channel_motifs(
        //     1000,
        //     100,
        //     &motif_nets[0],
        //     t_f1.i((0,..) ),
        //     100,
        //     1,
        //     max_motif_batches,
        //     &Cpu
        // );

        // println!("motifs_t.size()={:?}", motifs_t.squeeze().flatten(0,1).size());

        // let motifs_vec = vec_f32_from(&motifs_t.squeeze());

        // println!("motifs_vec={:?}", motifs_vec);
        // let motif_len = motifs_vec.len();
        plot_series(
            vec![
                (format!("t"), BLACK, vec_f32_from(&x).into_iter().zip(vec_f32_from(&t.i((0, ..)).squeeze())).collect::<Vec<(f32, f32)>>()),
                (format!("t_f1"), RED, vec_f32_from(&x).into_iter().zip(vec_f32_from(&t_f1.i((0, ..)).squeeze())).collect::<Vec<(f32, f32)>>()),
                (format!("t_f1"), GREEN, vec_f32_from(&x).into_iter().zip(vec_f32_from(&t_f100.i((0, ..)).squeeze())).collect::<Vec<(f32, f32)>>()),
                // (format!("motif"), RED, (0..motif_len).into_iter().map(|i|i as f32).zip(motifs_vec).collect::<Vec<(f32, f32)>>()),
                // (format!("G"), RED, (0..g_n).map(|i| 10. * i as f32).zip(g_v.into_iter()).collect::<Vec<(f32, f32)>>()),

            ], "EEG", (n as u32, 480), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
            // ], "EEG", ((g_n*30) as u32, 480), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        )?;




        Ok(())
    }

    #[test]
    pub fn test_full() -> Result<(), HbacError>{
        let motif_n =  1000;
        let input_n =  10000;

        // let motif_0_slice = comp_sin_v(motif_n);
        let motif_0_slice = sin_v(motif_n, 10.);
        let motif_0_v = pulse_in_rand(
            input_n,
            &[(((input_n-motif_n)/2) as usize, motif_0_slice.clone()), ], 0.05, 1., 0.1
        );

        // let motif_1_slice = comp_sin_k_v(motif_n, vec![30., 20., 10., 5., 3.]);
        let motif_1_slice = sin_v(motif_n, 20.);
        let motif_1_v = pulse_in_rand(
            input_n,
            &[(((input_n-motif_n)/2) as usize, motif_1_slice.clone()), ], 0.05, 1., 0.1
        );

        let rnd_v_slice = comp_sin_k_v(motif_n, vec![6., 3., 2.], 0.5);
        let rnd_v = pulse_in_rand(
            input_n,
            &[(((input_n-motif_n)/2) as usize, rnd_v_slice.clone()), ], 0.05, 1., 0.1
        );

        let x = lin_t(input_n);
        plot_series(
            vec![
                (format!("Motif 0"), RED, vec_f32_from(&x).into_iter().zip(motif_0_v.clone().into_iter()).collect::<Vec<(f32, f32)>>()),
                (format!("Motif 1"), GREEN, vec_f32_from(&x).into_iter().zip(motif_1_v.clone().into_iter()).collect::<Vec<(f32, f32)>>()),
                (format!("Rand"), BLACK,  vec_f32_from(&x).into_iter().zip(rnd_v.clone().into_iter()).collect::<Vec<(f32, f32)>>()),

            ], "EEG", (input_n as u32, 480), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        )?;
        let motif_0_t = Tensor::from_slice(motif_0_slice.as_slice());
        let motif_1_t = Tensor::from_slice(motif_1_slice.as_slice());
        let rnd_t = Tensor::from_slice(rnd_v_slice.as_slice());


        debug!("rnd_t.size()={:?}, motif_0_t.size()={:?}, motif_1_t.size()={:?}", rnd_t.size(), motif_0_t.size(), motif_1_t.size());

        let vs_0 = nn::VarStore::new(Device::Cpu);
        let net_0 = test_net(&vs_0, motif_n)?;

        utils::train(
            &Device::Cpu,
            &mut crate::simp::DummyDs::new(
                rnd_t.shallow_clone(),
                motif_0_t.shallow_clone(),
            ) as &mut dyn DataBatch,
            &net_0,
            |pred: &Tensor, target: &Tensor| {
                // pred.cross_entropy_loss::<Tensor>(&target.totype(Kind::Int64), None, Reduction::Mean, -100, 0.0)

                let pred_y = pred.i((.., 0, 0));
                debug!("pred_y ={:?} ", from::<Vec<f32>>(&pred_y));
                pred_y.binary_cross_entropy::<Tensor>(&target.totype(Kind::Float), None, Reduction::Mean)
            },
            100,
            nn::AdamW::default().build(&vs_0, 1.0e-4)?,
            |t|t
        )?;

        let vs_1 = nn::VarStore::new(Device::Cpu);
        let net_1 = test_net(&vs_1, motif_n)?;
        utils::train(
            &Device::Cpu,
            &mut crate::simp::DummyDs::new(
                rnd_t.shallow_clone(),
                motif_1_t.shallow_clone(),
            ) as &mut dyn DataBatch,
            &net_1,
            |pred: &Tensor, target: &Tensor| {
                // pred.cross_entropy_loss::<Tensor>(&target.totype(Kind::Int64), None, Reduction::Mean, -100, 0.0)

                let pred_y = pred.i((.., 0, 0));
                debug!("pred_y ={:?} ", from::<Vec<f32>>(&pred_y));
                pred_y.binary_cross_entropy::<Tensor>(&target.totype(Kind::Float), None, Reduction::Mean)
            },
            100,
            nn::AdamW::default().build(&vs_1, 1.0e-4)?,
            |t|t
        )?;

        let batch_rnd = Tensor::stack((0..10).map(|_| rnd_t.shallow_clone()).collect::<Vec<Tensor>>().as_slice(), 0).view((10, 1, motif_n));
        let batch_0 = Tensor::stack((0..10).map(|_| motif_0_t.shallow_clone()).collect::<Vec<Tensor>>().as_slice(), 0).view((10, 1, motif_n));
        let batch_1 = Tensor::stack((0..10).map(|_| motif_1_t.shallow_clone()).collect::<Vec<Tensor>>().as_slice(), 0).view((10, 1, motif_n));

        debug!("net_0 rnd_t={:?}, motif_0={:?}",
             from::<Vec<f32>>(&net_0.forward(&batch_rnd).view(-1)),
             from::<Vec<f32>>(&net_0.forward(&batch_0).view(-1))
        );

        debug!("net_1 rnd_t={:?}, motif_1={:?}",
             from::<Vec<f32>>(&net_1.forward(&batch_rnd).view(-1)),
             from::<Vec<f32>>(&net_1.forward(&batch_1).view(-1))
        );

        debug!("net_1x0 0-1={:?}, 1-0={:?}",
             from::<Vec<f32>>(&net_0.forward(&batch_1).view(-1)),
             from::<Vec<f32>>(&net_1.forward(&batch_0).view(-1))
        );

        let motif_0_t = Tensor::from_slice(motif_0_v.as_slice());
        debug!("motif_0_t={:?}", motif_0_t.size());

        let motif_1_t = Tensor::from_slice(motif_1_v.as_slice());
        debug!("motif_1_t={:?}", motif_1_t.size());

        // motif_0_t.split(motif_n, 0).into_iter().array_chunks::<10>().for_each(|slice|{
        let step = (motif_n / 4) as usize;


        let nets = vec![Mutex::new(net_0), Mutex::new(net_1)];


        let eeg_tensors = vec![motif_0_t, motif_1_t];

        let eeg_tensors_len = eeg_tensors.len();
        let nets_len = nets.len();

        let eeg_tensors = Arc::new(Mutex::new(eeg_tensors));

        let motif_batch_size = 10;

        let max_folds = batch_windows(input_n as usize, motif_n as usize, step, motif_batch_size as usize);
        let max_folds = max_folds.0.get(0).map(|v|v.len()).unwrap_or(max_folds.1.len());



        let full_batch_vec = (0..eeg_tensors_len).into_par_iter().map(|i|{

            let eeg_t = {
                let guard = eeg_tensors.lock().unwrap();
                guard.get(i).expect(format!("can't get tensor at {}", i).as_str()).shallow_clone()
            };


            super::apply_channel_motifs(motif_n, step, &nets, eeg_t, motif_batch_size, 0, max_folds, &Cpu)

        }).collect::<Vec<Tensor>>();

        let node = SeqBuilder::new(nets_len as i64)
            .linear_s(1)
            .func(Tensor::sigmoid)
            .end_seq();

        debug!("full_net_batch={:?}", full_batch_vec);

        let full_batch = Tensor::stack(full_batch_vec.as_slice(), 0);

        debug!("full_batch.size()={:?}", full_batch.size());

        let vs = nn::VarStore::new(Cpu);

        let net = node.build_tch_module(&vs.root())?;

        let _ = net.forward(&full_batch.transpose(1, 2));


        // for n in net_0_out {
        //     debug!("net_0_out={:?}", n);
        // }
        //
        // for n in net_1_out {
        //     debug!("net_1_out={:?}", n);
        // }

        // debug!("net_rnd_out={:?}", from::<Vec<f32>>(&net_0.forward(&Tensor::stack((0..10).map(|_| Tensor::from_slice(sig_in_rand(motif_n, &[], 0.05, 1.).as_slice())).collect::<Vec<Tensor>>().as_slice(), 0).view((10, 1, 1000))).i((.., 0, 0))));



        Ok(())
    }

    //TODO ignores values when decreasing length
    fn scale_x(series: &[f32], new_length: usize) -> Vec<f32> {
        let mut stretched_series = Vec::with_capacity(new_length);
        let old_length = series.len();
        let factor = (old_length - 1) as f32 / (new_length - 1) as f32;

        for new_index in 0..new_length {

            let old_index = (new_index as f32 * factor) as usize;
            let remainder = new_index as f32 * factor - old_index as f32;

            debug!("new_index={}, old_index={} remainder={}", new_index, old_index, remainder);

            if old_index == old_length - 1 {
                stretched_series.push(series[old_index]);
            } else {
                let value = series[old_index] * (1.0 - remainder) + series[old_index + 1] * remainder;
                stretched_series.push(value);
            }
        }

        stretched_series
    }


    #[allow(unused)]
    #[test]
    pub fn test_scale_x(){
        let v_5 = (0..5).map(|i| i as f32).collect::<Vec<f32>>();
        let t_5 = Tensor::from_slice(v_5.as_slice());

        let v_10 = (0..10).map(|i| i as f32).collect::<Vec<f32>>();
        let t_10 = Tensor::from_slice(v_10.as_slice());

        let v_20 = (0..20).map(|i| i as f32).collect::<Vec<f32>>();
        let t_20 = Tensor::from_slice(v_20.as_slice());

        // (0..10).zip(v_5.into_iter()).map(|v|)

        debug!("v_5={:?} x={:?}",  v_5, scale_x(v_5.as_slice(), 10));
        debug!("v_10={:?} x={:?}", v_10, scale_x(v_10.as_slice(), 10));
        debug!("v_20 ={:?} x={:?}", v_20, scale_x(v_20.as_slice(), 10));

        // let v_20_x = [0.0, 100.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
        // debug!("v_20_x={:?} x={:?}", v_20_x, scale_horizontally(v_20_x.as_slice(), 10));

        //[1, 2, 10, 4, 5] -> [1, 1.5, 2, 2.5, 10., 3.5, 4., 4.5. 5.]
        //[1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 11, 12, 13, 14, 15, 16, 17, 18, 19] -> [1,3,5,7,9,100,13,15,17,19]
        //TODO
    }

    #[allow(dead_code)]
    #[test]
    pub fn test_save_load() -> Result<(), HbacError> {
        let vs_0 = nn::VarStore::new(Device::Cpu);

        let node = SeqBuilder::new(123)
            .linear_s(10)
            .conv_1d(3, 1, 2, ConvConfig::default())
            .conv_1d(5, 1, 2, ConvConfig::default())
            .linear_s(20)
            .func(Tensor::sigmoid)
            .end_seq();
        let _my_mod = node.build_tch_module(&vs_0.root())?;

        println!("vs_0.variables()={:?}", vs_0.variables());

        // let x = tch::jit::CModule::
        // let x = tch::CModule::save(my_mod);

        // x.save("/tmp/test_mod.pt");

        vs_0.save("/tmp/test.pt")?;

        let mut vs_1 = nn::VarStore::new(Device::Cpu);
        vs_1.load("/tmp/test.pt")?;

        println!("vs_1.variables()={:?}", vs_1.variables());
        #[derive(Debug, Serialize, Deserialize)]
        struct T {
            i: i32,
            f: f64,
        }

        let t = T { i: 1, f: 321.456 };

        fs::write("/tmp/model.json", serde_json::to_string(&t)?)?;

        println!("{:?}", serde_json::from_str::<T>(&fs::read_to_string("/tmp/model.json").unwrap()).unwrap());
        //TODO
        Ok(())
    }

    #[test]
    pub fn test_test() -> Result<(), HbacError> {

        // SeqBuilder::new(num_features)
        //     .linear_s(1)
        //     .func(Tensor::sigmoid)
        //     .end_seq().build_tch_module(&vs.root());

        // let folds = batch_windows(684400, 1000, 100, 1000);
        //
        // println!("folds.0.len={:?} folds.1.len={:?}", folds.0.len(), folds.1.len());
        // println!("folds={:?}", folds);

        // let t = Tensor::from_slice(&[1., 2., 3., 4.]);
        //
        // t.i(2..3).print();
        // t.i(0..10).print();

        // println!("chunks_for_eeg={:?}", chunks_for_eeg);

        // let t_0_1 = Tensor::stack(&[Tensor::from_slice(&[0.]), Tensor::from_slice(&[0.])], 0);
        //
        // println!("t_0_1={:?}", t_0_1.size());
        //
        // let t_1_1 = Tensor::stack(&[Tensor::from_slice(&[0., 1., 1., 0.]), Tensor::from_slice(&[0., 1., 2., 0.,])], 0);
        // let t_1_2 = Tensor::stack(&[Tensor::from_slice(&[0., 2., 1., 0.]), Tensor::from_slice(&[0., 2., 2., 0.,])], 0);
        // let t_1_3 = Tensor::stack(&[Tensor::from_slice(&[0., 2., 1., 0.]), Tensor::from_slice(&[0., 2., 2., 0.,])], 0);
        // let t_0_1_2 = Tensor::stack(&[t_1_1, t_1_2, t_1_3], 0);
        //
        // println!("t_0_1_2={:?}", t_0_1_2.size());
        //
        // let x = t_0_1_2.linear::<Tensor>(&t_0_1, None);
        //
        // println!("x.size()={:?}", x.size());

        // let mut fifo = [0.0; 3];
        // for i in 0..100{
        //     fifo[i % fifo.len() as usize] = i as f32;
        //     println!("fifo={:?} i={}", fifo, i % fifo.len(), )
        // }

        // let t_0 = Tensor::from_slice(&[1., 2., 3.]).view((3, 1, 1));
        // let t_1 = Tensor::from_slice(&[10., 20., 30.]).view((3, 1, 1));
        //
        // println!("t_0={:?}", t_0.size());
        // println!("t_1={:?}", t_1.size());
        //
        // let t_v_1 = Tensor::vstack(&[t_0, t_1]);
        //
        // println!("t_v_1.size={:?}", t_v_1.size());
        //
        // let t_0 = Tensor::from_slice(&[1., 2., 3.]).view((3, 1, 1));
        // let t_1 = Tensor::from_slice(&[10., 20., 30.]).view((3, 1, 1));
        //
        // // println!("t_0={:?}", t_0.size());
        // // println!("t_1={:?}", t_1.size());
        //
        // let t_v_2 = Tensor::vstack(&[t_0, t_1]);
        //
        // println!("t_v_2.size={:?}", t_v_2.size());
        //
        // let t = Tensor::stack(&[t_v_1, t_v_2], 0);
        //
        // println!("t={:?}", t.size());

        // t.print();

        // let input_n = 10_000usize;
        // let motif_n = 1_000usize;
        //
        // let mut chunks = (0..=input_n - motif_n).step_by(motif_n/2 ).array_chunks::<10>();
        // while let Some(slice) = chunks.next() {
        //     println!("slices={:?}", slice);
        // }
        //
        // let rem_option = chunks.into_remainder();
        // println!("is_some={}", rem_option.is_some());
        // if let Some(mut reminder) = rem_option {
        //     println!("reminder={:?}", reminder.len());
        //
        // }

        // let vs = nn::VarStore::new(Device::Cpu);
        //
        // let mod_0 = SeqBuilder::new(1000).linear_s(1).end_seq().build_tch_module(&vs.root())?;


        // debug!("debug: {:?}", (||{println!(">>>debug"); "debug"})());
        // trace!("trace: {:?}", (||{println!(">>>trace"); "trace"})());

        // let t1 = Tensor::from_slice(&[4., 6.]);
        // let t2 = Tensor::from_slice(&[15., 5.]);
        //
        // let t_stack = Tensor::stack(&[t1.copy(), t2.copy()], 0);
        // let t_column_stack = Tensor::column_stack(&[t1.copy(), t2.copy()]);
        // let t_vstack = Tensor::vstack(&[t1.copy(), t2.copy()]);
        //
        //
        // debug!("t1={:?}, t2={:?}, t_stack={:?} t_column_stack={:?} t_vstack={:?}", t1, t2, t_stack, t_column_stack, t_vstack);
        // t_stack.print();
        // debug!("t_stack mean={:?}", t_stack.mean_dim(1, false, kind::Kind::Float));
        //
        // t_column_stack.print();
        // debug!("t_column_stack mean={:?}", t_column_stack.mean_dim(1, false, kind::Kind::Float));
        //
        // t_vstack.print();
        // debug!("t_vstack mean={:?}", t_vstack.mean_dim(1, false, kind::Kind::Float));

        Ok(())
    }

}

