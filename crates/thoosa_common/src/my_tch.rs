use core::fmt;
use std::borrow::Borrow;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use parking_lot::{Mutex, RwLock};

use log::{debug, error, trace, warn};
use tch::{Device, IndexOp, Tensor};
use tch::Device::Cpu;
use tch::Kind::Float;
use tch::kind::FLOAT_CPU;
use tch::nn::{batch_norm1d, BatchNorm, BatchNormConfig, Conv1D, Conv2D, Init, LinearConfig, Module, ModuleT, Optimizer, Path};
use tch::nn::ConvConfig;

use thoosa_shared::*;
use thoosa_shared::ThoosaError;

use crate::builder::Node;
// use crate::{data, utils};
use crate::builder::SeqBuilder;
use crate::train_nets::{MotifNodeParams, ThoModuleWithStore};
use crate::utils::*;

#[derive(Debug)]
pub struct MySequential {
    layers: Vec<Box<dyn ThoModule>>,
    split_input:Option<(i64, i64)> //dim features
}

impl MySequential {

    pub fn new(layers: Vec<Box<dyn ThoModule>>, split_input: Option<(i64, i64)>) -> Self {
        Self {
            layers ,
            split_input
        }
    }

    fn run_layers(&self, t: &Tensor, train: bool) -> Tensor {
        debug!("MySequential::forward: xs.size={:?} device={:?} self.layers[0]={:?}", t.size(), t.device(), self.layers[0]);

        let xs = self.layers[0].tho_forward(t, train);
        self.layers.iter().skip(1).enumerate().fold(xs, |xs, (i, layer)| {
            debug!("MySequential::layer={i} xs.size={:?} device={:?} layer={:?}", xs.size(), xs.device(), layer);

            layer.tho_forward(&xs, train)
        })
    }
}

impl ThoModule for MySequential {
    fn debug(&self) {
        for l in &self.layers {
            l.debug();
        }
    }

    fn tho_forward(&self, t: &Tensor, train: bool) -> Tensor {
        if self.layers.is_empty() {
            t.shallow_clone()
        } else {
            match self.split_input {
                None => {
                    self.run_layers(t, train)
                }
                Some((dim, num_features)) => {
                    let splits= t.split(num_features, dim);
                    let split_r = Tensor::concat(splits.into_iter().enumerate().filter_map(| (i, split) | {
                        if split.size()[dim as usize] == num_features {
                            let r = self.run_layers(&split, train);
                            debug!("MySequential::tho_forward i={i} split={:?} r.size()={:?}", split.size(), r.size());
                            Some(r)
                        } else {
                            debug!("MySequential::tho_forward skipping i={i} split.size={:?}", split.size());

                            None
                        }

                    }).collect::<Vec<_>>().as_slice(), 1);

                    let mean_r = split_r.mean_dim(1, false, None).unsqueeze(1);

                    debug!("MySequential::tho_forward split_r.size()={:?} mean_r.size()={:?}", split_r.size(), mean_r.size());

                    mean_r
                }
            }

        }
    }
}

pub struct MyParallel {
    layers: Vec<Box<dyn ThoModule>>,
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


impl MyParallel {

    pub fn new(layers: Vec<Box<dyn ThoModule>>, split:Box<dyn Fn(&Tensor)->Vec<Tensor> + Send + Sync>, join:Box<dyn Fn(Vec<Tensor>)->Tensor + Send + Sync>) -> Self {

        assert!(layers.len() > 0, "layers len > 0");

        Self {
            layers,
            split,
            join
        }
    }
}

impl ThoModule for MyParallel{
    fn debug(&self) {
        for l in &self.layers{
            l.debug();
        }
    }

    fn tho_forward(&self, t: &Tensor, train: bool) -> Tensor {
        debug!("MyParallel::forward: xs={:?}", t.size());
        let t = (self.join)(
            (self.split)(t).into_iter().zip(self.layers.iter()).map(|(t, net)| {
                net.tho_forward(&t, train)
            }).collect::<Vec<Tensor>>()
        );

        debug!("MyParallel::forward: t={:?}", t.size());

        t
    }
}

impl ThoModule for Conv1D {
    fn tho_forward(&self, t: &Tensor, train: bool) -> Tensor {
        self.forward(t)
    }
}

impl ThoModule for Conv2D {
    fn tho_forward(&self, t: &Tensor, train: bool) -> Tensor {
        self.forward(t)
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


impl ThoModule for MyMaxPool1d {
    fn tho_forward(&self, t: &Tensor, train: bool) -> Tensor {
        t.max_pool1d(self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode)
    }
}

#[derive(Debug)]
pub struct MyBatchNorm1d {
    batch_norm:BatchNorm
}

impl MyBatchNorm1d {
    pub fn new<'a, P: Borrow<Path<'a>>>(path:P, num_features:i64, config: BatchNormConfig) -> Self {
        Self {
            batch_norm:batch_norm1d(path, num_features, config)
        }
    }
}

impl ThoModule for MyBatchNorm1d{
    fn tho_forward(&self, t: &Tensor, train: bool) -> Tensor {
        self.batch_norm.forward_t(t, train)
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

impl MyAvgPool1d {
    pub fn new(kernel_size: i64, stride: i64, padding: i64) -> Self {
        Self {
            kernel_size,
            stride, padding,
            ceil_mode: false,
            count_include_pad: false
        }
    }
}


impl ThoModule for MyAvgPool1d{
    fn tho_forward(&self, t: &Tensor, train: bool) -> Tensor {
        t.avg_pool1d(self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
    }
}

pub trait ThoModule:Debug  + Send  {
    fn debug(&self){}
    fn mem_size(&self) -> usize {//TODO
        0
    }

    fn tho_forward(&self, t:&Tensor, train: bool) -> Tensor;

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


impl ThoModule for MyFuncModule{
    fn tho_forward(&self, t: &Tensor, train: bool) -> Tensor {
        (self.func)(t)
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


impl ThoModule for MyClosureModule{
    fn tho_forward(&self, t: &Tensor, train: bool) -> Tensor {
        (self.func)(t)
    }
}


#[derive(Debug)]
pub struct MyLinear {
    pub ws: Tensor,
    pub bs: Option<Tensor>,
}


impl ThoModule for MyLinear{
    fn debug(&self) {
        debug!("ThoModule for MyLinear::debug: ws.device={:?} bs.device={:?}", self.ws.device(), self.bs.as_ref().map(|bs|bs.device()));
    }

    fn tho_forward(&self, t: &Tensor, train: bool) -> Tensor {
        debug!("Module for MyLinear::forward: ws={:?} bs={:?} xs={:?}", self.ws.device(), self.bs.as_ref().map(|bs|bs.device()), t.device());
        match t.f_linear(&self.ws, self.bs.as_ref()) {//TODO error
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
) -> Result<MyLinear, ThoosaError> {
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
    let mut builder = SeqBuilder::new(num_features, None);

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

fn add_node_output_batch(motif_batch_size: i64, net_batch: &mut Vec<Tensor>, motifs_p_t: Tensor) {
    debug!("add_node_output_batch: motif_batch_size={motif_batch_size} net_batch.len={} motifs_p_t.size={:?}", net_batch.len(), motifs_p_t.size());

    if motifs_p_t.size()[0] < motif_batch_size {
        net_batch.push(
            pad_to(&motifs_p_t, &[motif_batch_size]).expect("pad_to failed")
        );
    } else {
        net_batch.push(
            motifs_p_t
        );
    }
}


#[cfg(test)]
mod tch_tests {

    use std::sync::Arc;
    use parking_lot::Mutex;

    use chrono::Utc;
    use log::{debug, set_max_level};
    use log::LevelFilter::Info;
    use plotters::prelude::{BLACK, GREEN, RED};
    use rayon::iter::{IntoParallelIterator, ParallelIterator};


    use tch::{Device, IndexOp, Kind, nn, Reduction, Tensor};
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::Float;
    use tch::kind::FLOAT_CPU;
    use tch::nn::{BatchNormConfig, ConvConfig, Init, Module, OptimizerConfig, VarStore};

    use thoosa_shared::*;
    use thoosa_shared::ThoosaError;

    use crate::builder::{clc_conv1d_out, Node};
    use crate::builder::SeqBuilder;
    use crate::my_tch::ThoModule;
    use crate::train_nets::{MotifNodeParams, ThoModuleWithStore};
    use crate::utils;
    use crate::utils::*;

    #[test]
    pub fn test_conv1_out(){
        let l_out = clc_conv1d_out(50 * 200, 5, ConvConfig::default());

        assert_eq!(l_out, 9996);
    }

    #[test]
    pub fn test_seq_linear_same_size(){
        let seq = SeqBuilder::new(2, None)
            .linear()
            .end_seq();

        let vs = nn::VarStore::new(Device::Cpu);

        let net = seq.build_tch_module(&vs.root()).unwrap();

        let y_p = net.tho_forward(&tch::Tensor::randn([10, 2], (Kind::Float, Device::Cpu)), true);

        assert_eq!(vec![10,2], y_p.size())
    }

    #[test]
    pub fn test_split_seq() -> Result<(), ThoosaError>{
        // unsafe {
        //     let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        // }

        let res_seq = SeqBuilder::new(10, None)
            .linear_s(1)
            .end_seq();

        let conv_seq = SeqBuilder::new(10, None)
            .conv_1d(5, 1, 1, ConvConfig::default())
            .linear_s(1)
            .end_seq();

        let main = SeqBuilder::new(10, None)
            .split(
                vec![res_seq, conv_seq],
                Box::new(|t:&Tensor|vec![t.i((.., 0, ..)).view((10, 1, 10)), t.i((.., 1, ..)).view((10, 1, 10))]),
                Box::new(|t:Vec<Tensor>|Tensor::stack(t.as_slice(), 1).squeeze())

            ).end_seq();

        let mut vs = nn::VarStore::new(Cpu);

        let net = main.build_tch_module(&vs.root())?;


        let y_p = net.tho_forward(&tch::Tensor::randn([10, 2, 10], (Kind::Float, Device::Cpu)), true);


        assert_eq!(vec![10,2], y_p.size());

        net.debug();
        for v in vs.trainable_variables() {
            debug!("device={:?}", v.device());
        }

        // vs.set_device(Cuda(0));
        //
        // net.debug();
        // for v in vs.trainable_variables() {
        //     debug!("device={:?}", v.device());
        // }

        Ok(())
    }

    #[test]
    pub fn test_save_load_net() -> Result<(), ThoosaError>{
        set_max_level(Info);
        let mut vs_0 = VarStore::new(Cpu);

        let n = 200;
        let net_0 = test_net(&vs_0, n)?;



        // let t_0 = Tensor::from_slice(sin_v(1000, 20.).as_slice());
        let t_0 = Tensor::from_slice(pulse_in_rand(n, &[], 0.05, 1., 0.1).as_slice());
        let t_1 = Tensor::from_slice(sin_v(n, 22.).as_slice());


        train(
            &Cpu,
            &mut crate::simp::DummyDs::new(
                t_0.shallow_clone(),
                t_1.shallow_clone(),
            ) as &mut dyn DataBatch,
            &net_0,
            |pred: &Tensor, target: &Tensor| {
                // pred.cross_entropy_loss::<Tensor>(&target.totype(Kind::Int64), None, Reduction::Mean, -100, 0.0)

                let pred_y = pred.i((.., 0, 0));
                let loss = pred_y.binary_cross_entropy::<Tensor>(&target.totype(Kind::Float), None, Reduction::Mean);
                debug!("pred_y ={pred_y} ");
                debug!("target ={target} ");

                loss
            },
            100,
            nn::AdamW::default().build(&vs_0, 1.0e-4)?,
            |t|t,
            None
        )?;



        let batch_0 = Tensor::stack((0..10).map(|_| t_0.shallow_clone()).collect::<Vec<Tensor>>().as_slice(), 0).view((10, n));
        let batch_1 = Tensor::stack((0..10).map(|_| t_1.shallow_clone()).collect::<Vec<Tensor>>().as_slice(), 0).view((10, n));

        debug!("net t_0={:?}, t_1={:?}",
             from::<Vec<f32>>(&net_0.tho_forward(&batch_0, true).view(-1)),
             from::<Vec<f32>>(&net_0.tho_forward(&batch_1, true).view(-1))
        );

        vs_0.freeze();
        vs_0.save("/tmp/test_vars.safetensors")?;

        let mut vs_1 = nn::VarStore::new(Device::Cpu);

        let net_1 = test_net(&vs_1, n)?;
        vs_1.load("/tmp/test_vars.safetensors")?;

        debug!("net 1 t_0={:?}, t_1={:?}",
             from::<Vec<f32>>(&net_1.tho_forward(&batch_0, true).view(-1)),
             from::<Vec<f32>>(&net_1.tho_forward(&batch_1, true).view(-1))
        );

        assert_eq!(from::<Vec<f32>>(&net_0.tho_forward(&batch_0, true).view(-1)), from::<Vec<f32>>(&net_1.tho_forward(&batch_0, true).view(-1)));
        assert_eq!(from::<Vec<f32>>(&net_0.tho_forward(&batch_1, true).view(-1)), from::<Vec<f32>>(&net_1.tho_forward(&batch_1, true).view(-1)));


        Ok(())
    }

    fn test_net(vs: &VarStore, num_features: i64) -> Result<Box<dyn ThoModule>, ThoosaError> {
        SeqBuilder::new(num_features, None)
            .batch_norm(BatchNormConfig::default())
            .closure(Box::new(|t:&Tensor|{
                t.view((t.size()[0], 1, t.size()[1]))//channel for convolution ("Given groups=1, weight of size [1, 1, 100], expected input[1, 10, 1000] to have 1 channels, but got 10 channels instead)
            }), None)
            .conv_1d(num_features/10, 1, 1, ConvConfig::default())
            .linear_s(1)
            .func(Tensor::sigmoid)
            .end_seq().build_tch_module(&vs.root())
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

        let v_10 = (0..10).map(|i| i as f32).collect::<Vec<f32>>();

        let v_20 = (0..20).map(|i| i as f32).collect::<Vec<f32>>();


        debug!("v_5={:?} x={:?}",  v_5, scale_x(v_5.as_slice(), 10));
        debug!("v_10={:?} x={:?}", v_10, scale_x(v_10.as_slice(), 10));
        debug!("v_20 ={:?} x={:?}", v_20, scale_x(v_20.as_slice(), 10));


        assert!(
            Tensor::from_slice(
                scale_x(v_5.as_slice(), 10).as_slice()
            ).allclose(
                &Tensor::from_slice(&[0.0, 0.44444445, 0.8888889, 1.3333334, 1.7777778, 2.2222223, 2.6666667, 3.1111112, 3.5555556, 4.0]).totype(Float),
                0.02, 0.02, true
            )
        );

        assert!(
            Tensor::from_slice(
                scale_x(v_10.as_slice(), 10).as_slice()
            ).allclose(
                &Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).totype(Float),
                0.02, 0.02, true
            )
        );

        assert!(
            Tensor::from_slice(
                scale_x(v_20.as_slice(), 10).as_slice()
            ).allclose(
                &Tensor::from_slice(&[0.0, 2.1111112, 4.2222223, 6.3333335, 8.444445, 10.555555, 12.666667, 14.777779, 16.88889, 19.0]).totype(Float),
                0.02, 0.02, true
            )
        );
    }

    #[allow(dead_code)]
    #[test]
    pub fn test_save_load() -> Result<(), ThoosaError> {
        let vs_0 = VarStore::new(Cpu);

        let node = test_node();
        let _my_mod = node.build_tch_module(&vs_0.root())?;

        debug!("variables={:?}", vs_0.variables());

        assert_eq!(
            26,
            vs_0.variables().len()
        );

        vs_0.save("/tmp/test.safetensors")?;

        let loaded_module = ThoModuleWithStore::load(test_node(), "/tmp/test.safetensors".into(), Cpu)?;

        debug!("loaded variables={:?}", loaded_module.var_store().variables());

        assert_eq!(
            vs_0.variables(),
            loaded_module.var_store().variables()
        );


        fn test_node() -> Node {
            SeqBuilder::new(123, None)
                .split(
                    vec![
                        SeqBuilder::new(123, None)
                            .batch_norm(BatchNormConfig::default())//3 weight bias running_var
                            .linear()//2 weight bias
                            .func(Tensor::relu)
                            .linear()//2 weight bias
                            .func(Tensor::relu)
                            .linear_s(1)//2 weight bias
                            .end_seq(),
                        SeqBuilder::new(321, None)
                            .batch_norm(BatchNormConfig::default())//3 weight bias running_var
                            .linear()//2 weight bias
                            .func(Tensor::relu)
                            .linear_s(1)//2 weight bias
                            .end_seq()
                    ],

                    Box::new(
                        move |t:&Tensor| {
                            vec![
                                t.i((.., 0, ..)),
                                t.i((.., 1, ..))
                            ]
                        }
                    ),

                    Box::new(move |t:Vec<Tensor>| {
                        Tensor::concat(t.as_slice(), 1)
                    })

                )
                .linear_s(10)//2 weight bias
                .conv_1d(3, 1, 2, ConvConfig::default())//2 weight bias
                .conv_1d(5, 1, 2, ConvConfig::default())//2 weight bias
                .linear_s(20)//2 weight bias
                .func(Tensor::sigmoid)
                .end_seq()
        }


        Ok(())
    }

}

