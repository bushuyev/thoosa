use core::fmt;
use std::borrow::Borrow;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use parking_lot::{Mutex, RwLock};

use log::{debug, error, warn};
use tch::{Device, IndexOp, Tensor};
use tch::Device::Cpu;
use tch::Kind::Float;
use tch::nn::{Init, LinearConfig, Module, Path};
use tch::nn::ConvConfig;

use thoosa_shared::*;
use thoosa_shared::ThoosaError;

use crate::builder::Node;
// use crate::{data, utils};
use crate::builder::SeqBuilder;
use crate::train_nets::{MotifNodeParams, SyncModule};
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

/**
slices single input tensor's channel into batches of input_slices_motif_batch_size with input_slice_step which were used during motif node training.
all motif nodes are to be from same run so have same input params.
Arranges output into max_motif_nodes x max_motif_batches  (max_ - to be same for all channels)

input_slices_motif_batch_size - both for motif node input batch size and output (input for label node) batch

 # returns Tensor nodes x batches x batch size
*/
pub fn apply_channel_motifs(
    input_slice_step: usize,
    input_slices_motif_batch_size: i64,
    motif_nodes: &mut [(MotifNodeParams, SyncModule)],
    channel: Tensor,
    max_motif_nodes: i64,
    max_motif_batches: usize,
    device: &Device
) -> Tensor {

    if motif_nodes.is_empty() {
        return Tensor::zeros([max_motif_nodes, max_motif_batches as i64, input_slices_motif_batch_size], (Float, *device));
    }

    for motif_node in motif_nodes.iter_mut() {
        motif_node.1.set_device(device.clone());
    }


    let motif_width = motif_nodes[0].0.num_features();//TODO different window lengths in same selection?

    let input_t_size = channel.size();
    debug!("apply_channel_motifs: channel.size={:?} input_t_size={:?} nodes.len={:?} motif_width={motif_width} input_slices_motif_batch_size={input_slices_motif_batch_size}", channel.size(), input_t_size, motif_nodes.len());

    assert_eq!(input_t_size.len(), 1);//single channel

    let input_n = input_t_size[0];


    let folds = batch_windows(input_n as usize, motif_width as usize, input_slice_step, input_slices_motif_batch_size as usize);

    debug!("apply_channel_motifs: input_n={:?}, folds.0.len={:?}, folds.1.len={:?}", input_n, folds.0.len(), folds.1.len());

    let nodes_len = motif_nodes.len();
    let mut node_batches = (0..nodes_len).map(|_| Vec::with_capacity(((input_n - motif_width) as usize) / input_slice_step)).collect::<Vec<Vec<Tensor>>>();


    for slice_0_idx in folds.0 {
        debug!("apply_channel_motifs: slice_0_idx={:?}", slice_0_idx);

        let input_slices_batch = Tensor::stack(slice_0_idx.into_iter().map(|i| channel.i(i as i64..i as i64 + motif_width)).collect::<Vec<Tensor>>().as_slice(), 0);//TODO unfold?
        debug!("apply_channel_motifs: input_slices_batch={:?} input_slices_motif_batch_size={:?} motif_width={:?}", input_slices_batch.size(), input_slices_motif_batch_size, motif_width);

        for ((_node_i, node), node_output_batches) in motif_nodes.iter().enumerate().zip(node_batches.iter_mut()) {
            let motifs_p_t = node.1.forward(&input_slices_batch.to_device(*device)).view(input_slices_batch.size()[0]);//squeeze is to remove motif width dimension: batch x width -> batch x 1 (probability) -> batch of probabilities that slice matches the motif

            debug!("apply_channel_motifs: motifs_p_t.size={:?} motifs_p_t_vals={motifs_p_t}", motifs_p_t.size());
            add_node_output_batch(input_slices_motif_batch_size, node_output_batches, motifs_p_t);
        };
    }

    if folds.1.len() > 0 {
        debug!("apply_channel_motifs: reminder={:?}", folds.1);

        let input_slices_batch = Tensor::stack(folds.1.into_iter().map(|i| channel.i(i as i64..i as i64 + motif_width)).collect::<Vec<Tensor>>().as_slice(), 0);
        debug!("apply_channel_motifs: reminder, input_slices_batch={:?} input_slices_motif_batch_size={:?} motif_width={:?}", input_slices_batch.size(), input_slices_motif_batch_size, motif_width);

        for ((node_i, node), node_batch) in motif_nodes.iter().enumerate().zip(node_batches.iter_mut()) {
            let tensor = node.1.forward(&input_slices_batch.to_device(*device));
            debug!("apply_channel_motifs: reminder, tensor.size={:?} input_slices_motif_batch_size={input_slices_motif_batch_size}", tensor.size());

            let motifs_p_t = tensor.view(input_slices_batch.size()[0]);

            debug!("apply_channel_motifs: reminder, motifs_p_t.size={:?} motifs_p_t_vals={motifs_p_t}", motifs_p_t.size());

            if motifs_p_t.size().len() > 0 {
                add_node_output_batch(input_slices_motif_batch_size, node_batch, motifs_p_t);

            } else {
                warn!("apply_channel_motifs: reminder, no results for node_i={} batch.size()={:?}", node_i, input_slices_batch.size());
            }
        };
    }

    //nodes x batches in tensor/motif_batch_size x motif_batch_size
    if log::log_enabled!(log::Level::Debug) {
        node_batches.iter().enumerate().for_each(|(i, node)| {
            debug!("apply_channel_motifs: i={i:?} node sizes={:?}",  node.iter().map(|t|t.size()).collect::<Vec<Vec<i64>>>());
        });
    }
    let node_tensors = node_batches.into_iter().map(|node_batches| Tensor::stack(node_batches.as_slice(), 0)).collect::<Vec<Tensor>>();
    debug!("apply_channel_motifs: node_tensors.len()={:?} sizes={:?}", node_tensors.len(), node_tensors.iter().map(|t|t.size()).collect::<Vec<Vec<i64>>>());

    let nodes_t = Tensor::stack(
        node_tensors.as_slice(),
        0
    );
    debug!("apply_channel_motifs: 2.2 nodes_t.size={:?}", nodes_t.size());
    debug!("apply_channel_motifs: 1. nodes_t.size={:?} max_motif_nodes={max_motif_nodes}, max_motif_batches={max_motif_batches} motif_batch_size={input_slices_motif_batch_size}", nodes_t.size());

    //pad_motifs_to- to have same num motifs in all channels
    let nodes_t = pad_to(&nodes_t, &[max_motif_nodes, max_motif_batches as i64, input_slices_motif_batch_size]).unwrap();

    for motif_node in motif_nodes {
        motif_node.1.set_device(Cpu);
    }

    debug!("apply_channel_motifs: 2. nodes_t={:?}", nodes_t.size());
    nodes_t
}

fn add_node_output_batch(motif_batch_size: i64, net_batch: &mut Vec<Tensor>, motifs_p_t: Tensor) {
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
    use log::debug;
    use plotters::prelude::{BLACK, GREEN, RED};
    use rayon::iter::{IntoParallelIterator, ParallelIterator};


    use tch::{Device, IndexOp, Kind, nn, Reduction, Tensor};
    use tch::Device::Cpu;
    use tch::Kind::Float;
    use tch::nn::{ConvConfig, Init, Module, OptimizerConfig, VarStore};

    use thoosa_shared::*;
    use thoosa_shared::ThoosaError;

    use crate::builder::{clc_conv1d_out};
    use crate::builder::SeqBuilder;
    use crate::train_nets::{MotifNodeParams, SyncModule};
    use crate::utils;
    use crate::utils::*;

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
    pub fn test_split_seq() -> Result<(), ThoosaError>{
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
    pub fn test_simp()-> Result<(), ThoosaError>{

        let mut vs_0 = VarStore::new(Cpu);

        let n = 1000;
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
            n as usize,
            nn::AdamW::default().build(&vs_0, 1.0e-4)?,
            |t|t,
            None
        )?;



        let batch_0 = Tensor::stack((0..10).map(|_| t_0.shallow_clone()).collect::<Vec<Tensor>>().as_slice(), 0).view((10, n));
        let batch_1 = Tensor::stack((0..10).map(|_| t_1.shallow_clone()).collect::<Vec<Tensor>>().as_slice(), 0).view((10, n));

        debug!("net t_0={:?}, t_1={:?}",
             from::<Vec<f32>>(&net_0.forward(&batch_0).view(-1)),
             from::<Vec<f32>>(&net_0.forward(&batch_1).view(-1))
        );

        vs_0.freeze();
        vs_0.save("/tmp/test_vars.safetensors")?;

        let mut vs_1 = nn::VarStore::new(Device::Cpu);

        let net_1 = test_net(&vs_1, n)?;
        vs_1.load("/tmp/test_vars.safetensors")?;

        debug!("net 1 t_0={:?}, t_1={:?}",
             from::<Vec<f32>>(&net_1.forward(&batch_0).view(-1)),
             from::<Vec<f32>>(&net_1.forward(&batch_1).view(-1))
        );

        assert_eq!(from::<Vec<f32>>(&net_0.forward(&batch_0).view(-1)), from::<Vec<f32>>(&net_1.forward(&batch_0).view(-1)));
        assert_eq!(from::<Vec<f32>>(&net_0.forward(&batch_1).view(-1)), from::<Vec<f32>>(&net_1.forward(&batch_1).view(-1)));


        Ok(())
    }

    fn test_net(vs: &VarStore, num_features: i64) -> Result<Box<dyn Module>, ThoosaError> {
        SeqBuilder::new(num_features)
            .batch_norm()
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
        let vs_0 = nn::VarStore::new(Device::Cpu);

        let node = SeqBuilder::new(123)
            .linear_s(10)
            .conv_1d(3, 1, 2, ConvConfig::default())
            .conv_1d(5, 1, 2, ConvConfig::default())
            .linear_s(20)
            .func(Tensor::sigmoid)
            .end_seq();
        let _my_mod = node.build_tch_module(&vs_0.root())?;

        vs_0.save("/tmp/test.safetensors")?;

        let mut vs_1 = nn::VarStore::new(Device::Cpu);
        for (key, t) in vs_0.variables() {
            let path = key.split(".").collect::<Vec<&str>>();
            let _ = (vs_1.root() / path[0] ).var(&path[1], &t.size(), Init::Const(0.0));
        }

        vs_1.load("/tmp/test.safetensors")?;

        assert_eq!(
            vs_0.variables().get("layer1.weight"),
            vs_1.variables().get("layer1.weight")
        );
        assert_eq!(
            vs_0.variables().get("layer1.weight__7"),
            vs_1.variables().get("layer1.weight__7")
        );

        assert_eq!(
            vs_0.variables().get("layer1.bias__6"),
            vs_1.variables().get("layer1.bias__6")
        );
        assert_eq!(
            vs_0.variables().get("layer1.bias"),
            vs_1.variables().get("layer1.bias")
        );

        assert_eq!(
            vs_0.variables().get("cnn.weight__5"),
            vs_1.variables().get("cnn.weight__5")
        );
        assert_eq!(
            vs_0.variables().get("cnn.bias__4"),
            vs_1.variables().get("cnn.bias__4")
        );


        Ok(())
    }

}

