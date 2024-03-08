use std::borrow::Borrow;
use std::env;
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::io::Write;
use std::sync::Mutex;
use std::time::Instant;

use log::{debug, info};
use plotters::prelude::*;
// use polars::prelude::{col, lit, NamedFrom, Series};
use simplelog::*;
use tch::{Device, IndexOp, kind, Kind, nn, no_grad, Reduction, Tensor};
use tch::nn::{Init, Linear, LinearConfig, Module, Path, Sequential};
use tch::nn::{ConvConfig, OptimizerConfig};

// use crate::{data, utils};
use crate::builder::{NthConst, SeqBuilder};
use crate::builder::Node;
// use crate::data::EEGs;
use hbac_shared::HbacError;
use crate::utils::DataBatch;
use hbac_shared::*;
// use linfa::prelude::*;

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

pub fn my_linear<'a, T: Borrow<Path<'a>>>(
    vs: T,
    in_dim: i64,
    out_dim: i64,
    c: LinearConfig,
) -> Result<Linear, HbacError> {
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

    Ok(Linear { ws: vs.f_var("weight", &[out_dim, in_dim], c.ws_init)?, bs })
}

pub fn configure_dense(num_features: i64, num_blocks: usize, num_layers_in_block: usize, kernel: i64, channels: i64) -> Node  {
    let mut builder = SeqBuilder::new(num_features);

    let dense_block = |b:&mut SeqBuilder| b
        .n_blocks(
            |b:&mut SeqBuilder, i| b
                .func(Tensor::elu)
                .conv(kernel, channels, channels, ConvConfig::default())
                .func(Tensor::elu)
                .conv(kernel, channels, channels, ConvConfig::default())
                .end(),
            num_layers_in_block)
        .end();


    builder
        .conv(kernel, channels, channels, ConvConfig::default())
        .func(|t|t.elu())
        .max_pool( 3, 2, 1)
        .n_blocks(|b:&mut SeqBuilder, i| b
            .block(dense_block)
            .block(|b:&mut SeqBuilder| b
                .func(Tensor::elu)
                .conv(kernel, channels, channels, ConvConfig::default())
                .func(Tensor::elu)
                .conv(kernel, channels, channels, ConvConfig::default())
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

fn load_batch(device: &Device, batch_size:i64) -> (Tensor, Tensor) {
    (
        tch::Tensor::randn([batch_size, 50 * 200, 20], (Kind::Float, device.clone())), //batch, 20 EEG feats, 50 sec * 200 samples per second,
        tch::Tensor::randn([batch_size, 6], (Kind::Float, device.clone())) //batch, seizure_vote,lpd_vote,gpd_vote, lrda_vote, grda_vote,other_vote, 50 sec * 200 samples per second,
    )
}



pub fn apply_channel_motifs(motif_n: i64, step: usize, nets: &Vec<Mutex<Box<dyn Module>>>, eeg_t_channel: Tensor, motif_batch_size: i64, pad_motifs_to: Option<i64>) -> Tensor {//pad_motifs_to- to have num motifs in all channels same

    debug!("apply_channel_motifs: eeg_t_channel.size()={:?}", eeg_t_channel.size());

    let input_t_size = eeg_t_channel.size();
    assert!(input_t_size.len() == 1);

    let input_n = input_t_size[0];
    let nets_len = nets.len();
    let mut net_tensors_vec = (0..nets_len).map(|_| Vec::with_capacity(((input_n - motif_n) as usize) / step)).collect::<Vec<Vec<Tensor>>>();

    let folds = slice_windows(input_n as usize, motif_n as usize, step, motif_batch_size as usize);

    for slice_0_idx in folds.0 {
        debug!("apply_channel_motifs: slice_0_idx={:?}", slice_0_idx);

        let batch = Tensor::stack(slice_0_idx.into_iter().map(|i| eeg_t_channel.i(i as i64..i as i64 + motif_n)).collect::<Vec<Tensor>>().as_slice(), 0);
        debug!("apply_channel_motifs: batch.size={:?} motif_batch_size={:?}", batch.size(), motif_batch_size);

        let batch = batch.view((motif_batch_size, 1, motif_n));

        for ((net_i, net), net_tensors_vec) in nets.iter().enumerate().zip(net_tensors_vec.iter_mut()) {

            let net_rs = net.lock().unwrap().forward(&batch);
            debug!("apply_channel_motifs: batch.size()={:?} net_rs size={:?}", batch.size(), net_rs.size());
            net_tensors_vec.push(net_rs.view((motif_batch_size)));
        };
    }


    if folds.1.len() > 0 {
        debug!("apply_channel_motifs: reminder={:?}", folds.1);
        //TODO process remainder?
    }

    //nets x slices in tensor/motif_batch_size x motif_batch_size
    let mut net_tensors = net_tensors_vec.into_iter().map(|slices_vec| Tensor::vstack(slices_vec.as_slice())).collect::<Vec<Tensor>>();
    let zero_t = Tensor::zeros([motif_batch_size as i64], (Kind::Float, Device::Cpu)).view((1, motif_batch_size as i64));

    if let Some(pad_to) = pad_motifs_to {
        (net_tensors.len()..pad_to as usize).for_each(|_|{
            net_tensors.push(zero_t.shallow_clone());
        })
    }

    Tensor::stack(
        net_tensors.as_slice(),
        0
    ).view((pad_motifs_to.unwrap_or(nets_len as i64), -1))
}


#[cfg(test)]
mod tch_tests {
    use log::{debug, trace, LevelFilter};
    use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger};
    use tch::{Device, IndexOp, kind, Kind, nn, Reduction, Tensor};
    use tch::nn::{ConvConfig, Module, OptimizerConfig, Sequential, VarStore};
    use hbac_shared::HbacError;

    use crate::builder::clc_conv1d_out;
    use crate::builder::SeqBuilder;
    use crate::my_tch::{configure_dense};
    use crate::utils;
    use crate::utils::*;
    use hbac_shared::*;
    use core::f32::consts::PI;
    use std::iter::{ArrayChunks, StepBy};
    use std::ops::RangeInclusive;
    use chrono::Utc;
    use plotters::prelude::{BLACK, GREEN, RED};
    use rayon::prelude::*;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    use std::sync::Arc;
    use std::sync::Mutex;
    use tch::Device::Cpu;

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
    pub fn test_seq_linear(){
        let seq = SeqBuilder::new(2)
            .linear()
            .end_seq();

        let vs = nn::VarStore::new(Device::Cpu);

        let net = seq.build_tch_module(&vs.root()).unwrap();

        let y_p = net.forward(&tch::Tensor::randn([10, 2], (Kind::Float, Device::Cpu)));

        y_p.print()
    }

    // #[test]
    // pub fn test_configure_dense(){
    //     let vs = nn::VarStore::new(Device::Cpu);
    //
    //     let model = configure_dense(50 * 200, 4, 4, 100, 20).build_tch_module(&vs.root()).unwrap();
    //
    //
    //     let eegs = EEGs::load_all_eegs("../../data").unwrap();
    //
    //     let mut opt = nn::AdamW::default().build(&vs, 1e-2).unwrap();
    //
    //
    //     // // let tensor = Tensor::randn([10, 20, 10000], (Kind::Float, Device::Cpu));
    //     let (tensor, y_t) = eegs.train_batch(10);
    //     // debug!("size:   {:?}", tensor.size());
    //     let pred_y = model.forward(
    //         &tensor.totype(Kind::Float) //batch, channels, L in
    //     );
    //     let pred_y_mean = pred_y.mean_dim(1, false, kind::Kind::Float);
    //     debug!("1. pred size:   pred_y.size()={:?}     y_t.size()={:?}", pred_y_mean.size(), y_t.size());
    //     // let loss = y_t.mse_loss(&pred_y_mean, Reduction::Mean); //.sum(kind::Kind::Float);;
    //     let loss = y_t.cross_entropy_loss::<Tensor>(&pred_y_mean, None, Reduction::Mean, -100, 0.0);
    //     // let loss_1 = (pred_y_mean - y_t).pow_tensor_scalar(2).sum(kind::Kind::Float);
    //     debug!(":::::::::::::::::::::::::::: pred_y.size()={:?} y_t.size()={:?}, pred_y_mean.size()={:?},  loss.size()={:?}", pred_y.size(), y_t.size(), pred_y_mean.size(),  loss.size());
    //     loss.print();
    //     debug!("::::::::::::::::::::::::::::: loss={:?}", loss);
    //     opt.backward_step(&loss);
    //     //
    //     // // let tensor = Tensor::randn([10, 20, 10000], (Kind::Float, Device::Cpu));
    //     // let (tensor, y_t) = eegs.batch_tensor(10);
    //     // debug!("size:   {:?}", tensor.size());
    //     // let pred_y = model.forward(
    //     //     &tensor //batch, channels, L in
    //     // );
    //     // debug!("2. pred size:   {:?}", pred_y.size());
    //     // let pred_y_mean = pred_y.mean_dim(1, false, kind::Kind::Float);
    //     // // let loss = (pred_y_mean - y_t).pow_tensor_scalar(2).sum(kind::Kind::Float);
    //     // let loss = y_t.mse_loss(&pred_y_mean, Reduction::None);
    //     // opt.backward_step(&loss);
    //     // pred_y.print();
    //
    //
    //
    //
    // }

    #[test]
    pub fn test_res_seq(){
        let res_seq = SeqBuilder::new(10)
            .linear()
            .end_seq();

        let conv_seq = SeqBuilder::new(10)
            .conv(5, 5, 5, ConvConfig::default())
            .end_seq();

        let main = SeqBuilder::new(10)
            .split(
                res_seq,
                conv_seq,
                |t|t.copy(),
                |t|t.copy()
            ).end_seq();

    }


    #[test]
    pub fn test_simp()-> Result<(), HbacError>{

        let mut vs_0 = nn::VarStore::new(Device::Cpu);

        let n = 1000;
        let net_0 = test_net(&vs_0, n)?;



        // let t_0 = Tensor::from_slice(sin_v(1000, 20.).as_slice());
        let t_0 = Tensor::from_slice(sig_in_rand(n, &[], 0.05, 1.).as_slice());
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
            nn::AdamW::default().build(&vs_0, 1.0e-4)?
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
            .conv(num_features/10, 1, 1, ConvConfig::default())
            .linear_s(1)
            .func(Tensor::sigmoid)
            .end_seq().build_tch_module(&vs.root())
    }

    #[test]
    pub fn test_full() -> Result<(), HbacError>{
        let motif_n =  1000;
        let input_n =  10000;

        // let motif_0_slice = comp_sin_v(motif_n);
        let motif_0_slice = sin_v(motif_n, 10.);
        let motif_0_v = sig_in_rand(
            input_n,
            &[(((input_n-motif_n)/2) as usize, motif_0_slice.clone()), ], 0.05, 1.
        );

        // let motif_1_slice = comp_sin_k_v(motif_n, vec![30., 20., 10., 5., 3.]);
        let motif_1_slice = sin_v(motif_n, 20.);
        let motif_1_v = sig_in_rand(
            input_n,
            &[(((input_n-motif_n)/2) as usize, motif_1_slice.clone()), ], 0.05, 1.
        );

        let rnd_v_slice = comp_sin_k_v(motif_n, vec![6., 3., 2.], 0.5);
        let rnd_v = sig_in_rand(
            input_n,
            &[(((input_n-motif_n)/2) as usize, rnd_v_slice.clone()), ], 0.05, 1.
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

        let mut vs_0 = nn::VarStore::new(Device::Cpu);
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
            nn::AdamW::default().build(&vs_0, 1.0e-4)?
        )?;

        let mut vs_1 = nn::VarStore::new(Device::Cpu);
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
            nn::AdamW::default().build(&vs_1, 1.0e-4)?
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

        let full_batch_vec = (0..eeg_tensors_len).into_par_iter().map(|i|{

            let eeg_t = {
                let guard = eeg_tensors.lock().unwrap();
                guard.get(i).expect(format!("can't get tensor at {}", i).as_str()).shallow_clone()
            };


            super::apply_channel_motifs(motif_n, step, &nets, eeg_t, motif_batch_size, None)

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

        net.forward(&full_batch.transpose(1, 2));


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


    #[test]
    pub fn test_test() -> Result<(), HbacError> {


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

