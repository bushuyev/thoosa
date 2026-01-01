use core::fmt;
use std::borrow::Borrow;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use log::{debug, error};
use tch::Tensor;

use tch::nn::ConvConfig;
use tch::nn::{
    BatchNorm, BatchNormConfig, Conv, Init, LinearConfig, ModuleT, Path, batch_norm1d,
};

use thoosa_shared::ThoError;
use thoosa_shared::traits::SimpleId;
use crate::builder::Node;
// use crate::{data, utils};
use crate::builder::SeqBuilder;
use crate::dims::{AnyDims, Dims};
use crate::tho_tensor::ThoTensor;
use crate::utils::*;

#[derive(Debug)]
pub struct MySequential<DI: Dims, DO: Dims> {
    layers: Vec<Box<dyn ThoModule<AnyDims, AnyDims>>>,
    split_input: Option<(i64, i64)>, //dim features
    _pd_do: PhantomData<DO>,
    _pd_di: PhantomData<DI>,
}

impl<DI: Dims, DO: Dims> MySequential<DI, DO> {
    pub fn new(
        layers: Vec<Box<dyn ThoModule<AnyDims, AnyDims>>>,
        split_input: Option<(i64, i64)>,
    ) -> Self {
        Self {
            layers,
            split_input,
            _pd_do: PhantomData,
            _pd_di: PhantomData,
        }
    }

    fn run_layers(&self, t: &ThoTensor<DI, SimpleId>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        debug!("MySequential::run_layers: t.size={:?} device={:?} self.layers[0]={:?}",t.as_ref().size(),t.device_idx(),self.layers[0]);

        let output_t = self.layers[0].tho_forward(&t.as_ref().into(), train)?;

        Ok(self.layers
            .iter()
            .skip(1)
            .enumerate()
            .try_fold(output_t, |output_t, (i, layer)| {
                debug!("MySequential::run_layers={i} output_t.size={:?} output_t.device={:?} layer={:?}",output_t.size(),output_t.device_idx(),layer);

                layer.tho_forward(&output_t, train)
            })?
            .as_ref()
            .into())
    }
}

impl<DI: Dims + Send + Sync, DO: Dims + Send + Sync> ThoModule<DI, DO> for MySequential<DI, DO> {
    fn debug(&self) {
        for l in &self.layers {
            l.debug();
        }
    }

    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        if self.layers.is_empty() {
            Ok(t.shallow_clone())
        } else {
            match self.split_input {
                None => self.run_layers(t, train),
                Some((dim, num_features)) => {
                    let splits = t.as_ref().f_split(num_features, dim)?;
                    debug!("MySequential::tho_forward splits.sizes={:?} num_features={num_features} dim={dim}",splits.iter().map(|t| t.size()).collect::<Vec<_>>());

                    let split_r = Tensor::f_concat(
                        splits
                            .into_iter()
                            .enumerate()
                            .map(|(i, split)| {
                                if split.size()[dim as usize] == num_features {
                                    debug!("MySequential::tho_forward i={i} split={:?}",split.size());

                                    let r = self.run_layers(&split.into(), train)?;
                                    debug!("MySequential::tho_forward i={i} r.size()={:?}",r.size());

                                    Ok(r)
                                } else {
                                    //TODO ok skipping?
                                    debug!("MySequential::tho_forward skipping i={i} split.size={:?}",split.size());

                                    Err(ThoError::new(format!("Unexpected split size: {:?} dim={dim}", split.size())))
                                }
                            })
                            .collect::<Result<Vec<_>, ThoError>>()?
                            .as_slice(),
                        1,
                    )?;

                    let mean_r = split_r.f_mean_dim(1, false, None)?.f_unsqueeze(1)?;//TODO why mean?

                    debug!("MySequential::tho_forward split_r.size()={:?} mean_r.size()={:?}",split_r.size(),mean_r.size());

                    Ok(mean_r.into())
                }
            }
        }
    }
}

pub struct MyParallel<DI: Dims, DO: Dims> {
    layers: Vec<Box<dyn ThoModule<DI, DO>>>,
    split: Arc<dyn Fn(&Tensor) -> Result<Vec<Tensor>, ThoError> + Send + Sync>,
    join: Arc<dyn Fn(Vec<Tensor>) -> Result<Tensor, ThoError> + Send + Sync>,
}

impl<DI: Dims, DO: Dims> Debug for MyParallel<DI, DO> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("").field("layers", &self.layers).finish()
    }
}

impl<DI: Dims, DO: Dims> MyParallel<DI, DO> {
    pub fn new(
        layers: Vec<Box<dyn ThoModule<DI, DO>>>,
        split: Arc<dyn Fn(&Tensor) -> Result<Vec<Tensor>, ThoError> + Send + Sync>,
        join: Arc<dyn Fn(Vec<Tensor>) -> Result<Tensor, ThoError> + Send + Sync>,
    ) -> Self {
        assert!(layers.len() > 0, "layers len == 0");

        Self {
            layers,
            split,
            join,
        }
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for MyParallel<DI, DO> {
    fn debug(&self) {
        for l in &self.layers {
            l.debug();
        }
    }

    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        debug!("MyParallel::forward: xs={:?}", t.size());
        let t = (self.join)(
            (self.split)(t.as_ref())?
                .into_iter()
                .zip(self.layers.iter())
                .map(|(t, net)| net.tho_forward(&t.into(), train).map(|v|v.as_ref().shallow_clone()))
                .collect::<Result<Vec<Tensor>, ThoError>>()?,
        )?;

        debug!("MyParallel::forward: t={:?}", t.size());

        Ok(t.into())
    }
}
/////
pub struct Split1<DI: Dims, DO: Dims> {
    layer: Box<dyn ThoModule<DI, DO>>,
    split: Arc<dyn Fn(&Tensor) -> Result<Vec<Tensor>, ThoError> + Send + Sync>,
    join: Arc<dyn Fn(Vec<Tensor>) -> Result<Tensor, ThoError> + Send + Sync>,
}

impl<DI: Dims, DO: Dims> Debug for Split1<DI, DO> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MySeq").finish()//TODO
    }
}

impl<DI: Dims, DO: Dims> Split1<DI, DO> {
    pub fn new(
        layer: Box<dyn ThoModule<DI, DO>>,
        split: Arc<dyn Fn(&Tensor) -> Result<Vec<Tensor>, ThoError> + Send + Sync>,
        join: Arc<dyn Fn(Vec<Tensor>) -> Result<Tensor, ThoError> + Send + Sync>,
    ) -> Self {

        Self {
            layer,
            split,
            join,
        }
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for Split1<DI, DO> {
    fn debug(&self) {
       self.layer.debug()
    }

    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        debug!("MyParallel::forward: xs={:?}", t.size());
        let t = (self.join)(
            (self.split)(t.as_ref())?
                .into_iter()
                .map(|split_t| self.layer.tho_forward(&split_t.into(), train).map(|v|v.as_ref().shallow_clone()))
                .collect::<Result<Vec<Tensor>, ThoError>>()?,
        )?;

        debug!("MyParallel::forward: t={:?}", t.size());

        Ok(t.into())
    }
}
/*
impl <DI:Dims, DO:Dims> ThoModule<DI, DO> for Conv1D {
    fn tho_forward(&self, t: &ThoTensor<DI>, train: bool) -> ThoTensor<DO> {
        self.forward(t.as_ref()).into()
    }
}

impl <DI:Dims, DO:Dims> ThoModule<DI, DO> for Conv2D {
    fn tho_forward(&self, t: &ThoTensor<DI>, train: bool) -> ThoTensor<DO> {
        self.forward(t.as_ref()).into()
    }
}*/

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
            stride,
            padding,
            dilation: 1,
            ceil_mode: false,
        }
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for MyMaxPool1d {
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        Ok(t.as_ref()
            .f_max_pool1d(
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.ceil_mode,
            )?
            .into())
    }
}

#[derive(Debug)]
pub struct AdaptiveMaxPool1d {
    output_size: u32
}

impl AdaptiveMaxPool1d {
    pub fn new(output_size: u32) -> Self {
        Self { output_size }
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for AdaptiveMaxPool1d {
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        Ok(t.as_ref()
            .f_adaptive_avg_pool1d(self.output_size as i64)?
            .into())
    }
}

#[derive(Debug)]
pub struct MyBatchNorm1d {
    batch_norm: BatchNorm,
}

impl MyBatchNorm1d {
    pub fn new<'a, P: Borrow<Path<'a>>>(
        path: P,
        num_features: u32,
        config: BatchNormConfig,
    ) -> Self {
        Self {
            batch_norm: batch_norm1d(path, num_features as i64, config),
        }
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for MyBatchNorm1d {
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        Ok(self.batch_norm.forward_t(t.as_ref(), train).into())
    }
}

unsafe impl Send for MyBatchNorm1d {}
unsafe impl Sync for MyBatchNorm1d {}

#[derive(Debug)]
pub struct MyConv<ND> {
    conv: Conv<ND>,
}

impl<ND> MyConv<ND> {
    pub fn new(conv: Conv<ND>) -> Self {
        Self { conv }
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for MyConv<[i64; 1]> {
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        Ok(self.conv.forward_t(t.as_ref(), train).into())//TODO custom conv for errors
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for MyConv<[i64; 2]> {
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        Ok(self.conv.forward_t(t.as_ref(), train).into())//TODO custom conv for errors
    }
}

unsafe impl<ND> Send for MyConv<ND> {}
unsafe impl<ND> Sync for MyConv<ND> {}

#[derive(Debug)]
pub struct MyAvgPool1d {
    kernel_size: i64,
    stride: i64,
    padding: i64,
    ceil_mode: bool,
    count_include_pad: bool,
}

impl MyAvgPool1d {
    pub fn new(kernel_size: i64, stride: i64, padding: i64) -> Self {
        Self {
            kernel_size,
            stride,
            padding,
            ceil_mode: false,
            count_include_pad: false,
        }
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for MyAvgPool1d {
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        Ok(t.as_ref()
            .f_avg_pool1d(
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
            )?
            .into())
    }
}

pub trait ThoModule<DI: Dims, DO: Dims>: Debug + /*Sync +*/ Send {
    fn debug(&self) {}
    fn mem_size(&self) -> usize {
        //TODO
        0
    }

    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError>;
}

// unsafe impl <DI:Dims, DO:Dims>  Send for ThoModuleWithStore<DI, DO>{}
// unsafe impl <DI:Dims, DO:Dims>  Sync for ThoModuleWithStore<DI, DO>{}

pub type Fp = fn(&Tensor) -> Result<Tensor, ThoError>;
pub type LossFp = fn(&Tensor, &Tensor) -> Result<Tensor, ThoError>;

pub struct MyFuncModule {
    func: Fp,
}

impl Debug for MyFuncModule {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MyFuncModule").finish()
    }
}

impl MyFuncModule {
    pub fn new(func: Fp) -> Self {
        Self { func }
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for MyFuncModule {
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        (self.func)(t.as_ref()).map(|t|t.into())
    }
}

pub struct MyClosureModule {
    func: Arc<Box<dyn Fn(&Tensor) -> Result<Tensor, ThoError>>>,
}

unsafe impl Send for MyClosureModule {}
unsafe impl Sync for MyClosureModule {}

impl MyClosureModule {
    pub fn new(func: Arc<Box<dyn Fn(&Tensor) -> Result<Tensor, ThoError>>>) -> Self {
        Self { func }
    }
}

impl Debug for MyClosureModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("MyClosureModule").finish()
    }
}

impl<DI: Dims + Send + Sync, DO: Dims + Send + Sync> ThoModule<DI, DO> for MyClosureModule {
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        (self.func)(t.as_ref()).map(|v|v.into())
    }
}

#[derive(Debug)]
pub struct MyLinear<DI: Dims, DO: Dims> {
    pub ws: Tensor,
    pub bs: Option<Tensor>,
    _pd_di: PhantomData<DI>,
    _pd_do: PhantomData<DO>,

    #[cfg(debug_assertions)]
    descr: String,
}

unsafe impl<DI: Dims, DO: Dims> Send for MyLinear<DI, DO> {}
unsafe impl<DI: Dims, DO: Dims> Sync for MyLinear<DI, DO> {}

impl<DI: Dims + Send + Sync, DO: Dims + Send + Sync> ThoModule<DI, DO> for MyLinear<DI, DO> {
    fn debug(&self) {
        debug!(
            "ThoModule for MyLinear::debug: ws.device={:?} bs.device={:?}",
            self.ws.device(),
            self.bs.as_ref().map(|bs| bs.device())
        );
    }

    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        #[cfg(debug_assertions)]
        {
            debug!("Module for MyLinear::forward: descr={}", self.descr);
            debug!("Module for MyLinear::forward: devices ws={:?} bs={:?} t={:?}",self.ws.device(),self.bs.as_ref().map(|bs| bs.device()),t.device_idx());
            debug!("Module for MyLinear::forward: sizes ws={:?} bs={:?} t={:?}",self.ws.size(),self.bs.as_ref().map(|bs| bs.size()),t.size());
        }
        match t.as_ref().f_linear(&self.ws, self.bs.as_ref()) {
            //TODO error
            Ok(t) => Ok(t.into()),
            Err(e) => {
                error!("{:?}", e);
                #[cfg(debug_assertions)]
                error!("failed layer descr: {} ", self.descr);

                Err(e.into())
            }
        }
    }
}

pub fn my_linear<'a, T: Borrow<Path<'a>>, DI: Dims, DO: Dims>(
    vs: T,
    in_dim: i64,
    out_dim: i64,
    c: LinearConfig,
    #[cfg(debug_assertions)] descr: String,
) -> Result<MyLinear<DI, DO>, ThoError> {
    assert!(in_dim > 0);
    assert!(out_dim > 0);

    let vs = vs.borrow();
    let bs = if c.bias {
        let bs_init = c.bs_init.unwrap_or_else(|| {
            let bound = 1.0 / (in_dim as f64).sqrt();
            Init::Uniform {
                lo: -bound,
                up: bound,
            }
        });
        Some(vs.f_var("bias", &[out_dim], bs_init)?)
    } else {
        None
    };

    Ok(MyLinear {
        ws: vs.f_var("weight", &[out_dim, in_dim], c.ws_init)?,
        bs,
        _pd_di: Default::default(),
        _pd_do: Default::default(),
        #[cfg(debug_assertions)]
        descr,
    })
}

pub fn configure_dense(
    num_features: u32,
    num_blocks: usize,
    num_layers_in_block: usize,
    kernel: u32,
    channels: u32,
) -> Node {
    let mut builder = SeqBuilder::new(num_features, None);

    let dense_block = |b: &mut SeqBuilder| {
        b.n_blocks(
            |b: &mut SeqBuilder, _i| {
                b.func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
                    .conv_1d("conv", kernel, channels, channels, ConvConfig::default(), None)
                    .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
                    .conv_1d("conv", kernel, channels, channels, ConvConfig::default(), None)
                    .end()
            },
            num_layers_in_block,
        )
        .end()
    };

    builder
        .conv_1d("conv", kernel, channels, channels, ConvConfig::default(), None)
        .func(|t| t.f_elu().map_err(|e|e.into()))
        .max_pool(3, 2, 1)
        .n_blocks(
            |b: &mut SeqBuilder, _i| {
                b.block(dense_block)
                    .block(|b: &mut SeqBuilder| {
                        b.func(|t|Tensor::f_elu(t).map_err(|e|e.into()))
                            .conv_1d("conv", kernel, channels, channels, ConvConfig::default(), None)
                            .func(|t|Tensor::f_elu(t).map_err(|e|e.into()))
                            .conv_1d("conv", kernel, channels, channels, ConvConfig::default(), None)
                            .end()
                    })
                    .end()
            },
            num_blocks,
        )
        .block(dense_block)
        .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
        .avg_pool(7, 3, 0)
        .dropout()
        .linear_s(
            6,
            #[cfg(debug_assertions)]
            "test".to_string(),
        )
        .end_seq()
}

#[allow(dead_code)]
fn add_node_output_batch(motif_batch_size: i64, net_batch: &mut Vec<Tensor>, motifs_p_t: Tensor) {
    debug!(
        "add_node_output_batch: motif_batch_size={motif_batch_size} net_batch.len={} motifs_p_t.size={:?}",
        net_batch.len(),
        motifs_p_t.size()
    );

    if motifs_p_t.size()[0] < motif_batch_size {
        net_batch.push(pad_to(&motifs_p_t, &[motif_batch_size]).expect("pad_to failed"));
    } else {
        net_batch.push(motifs_p_t);
    }
}

#[cfg(test)]
mod tch_tests {
    use std::sync::Arc;
    use log::LevelFilter::Info;
    use log::{debug, set_max_level};

    use tch::Device::Cpu;
    use tch::Kind::Float;
    use tch::{Device, IndexOp, Kind, Reduction, Tensor, nn};

    use tch::nn::{BatchNormConfig, ConvConfig, OptimizerConfig, VarStore};

    use thoosa_shared::ThoError;
    use thoosa_shared::*;

    use crate::builder::SeqBuilder;
    use crate::builder::{Node, clc_conv1d_out};
    use crate::dims::{AnyDims, Batch};
    use crate::my_tch::ThoModule;
    use crate::train_nets::ThoModuleWithStore;

    use crate::utils::*;

    #[test]
    pub fn test_conv1_out() {
        let l_out = clc_conv1d_out(50 * 200, 5, ConvConfig::default());

        assert_eq!(l_out, 9996);
    }

    #[test]
    pub fn test_seq_linear_same_size() {
        let seq = SeqBuilder::new(2, None)
            .linear(
                #[cfg(debug_assertions)]
                "test".to_string(),
            )
            .end_seq();

        let vs = nn::VarStore::new(Device::Cpu);

        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = seq.build_tch_module(&vs.root()).unwrap();

        let y_p = net.tho_forward(&Tensor::randn([10, 2], (Float, Cpu)).into(), true).unwrap();

        assert_eq!(vec![10, 2], y_p.size())
    }

    #[test]
    pub fn test_split_seq() -> Result<(), ThoError> {
        // unsafe {
        //     let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        // }

        let res_seq = SeqBuilder::new(10, None)
            .linear_s(
                1,
                #[cfg(debug_assertions)]
                "test".to_string(),
            )
            .end_seq();

        let conv_seq = SeqBuilder::new(10, None)
            .conv_1d("conv", 5, 1, 1, ConvConfig::default(), None)
            .linear_s(
                1,
                #[cfg(debug_assertions)]
                "test".to_string(),
            )
            .end_seq();

        let main = SeqBuilder::new(10, None)
            .split(
                vec![res_seq, conv_seq],
                Box::new(|t: &Tensor| {
                    Ok(vec![
                        t.i((.., 0, ..)).view((10, 1, 10)),
                        t.i((.., 1, ..)).view((10, 1, 10)),
                    ])
                }),
                Box::new(|t: Vec<Tensor>| Ok(Tensor::stack(t.as_slice(), 1).squeeze())),
            )
            .end_seq();

        let vs = nn::VarStore::new(Cpu);

        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = main.build_tch_module(&vs.root())?;

        let y_p = net.tho_forward(
            &Tensor::randn([10, 2, 10], (Kind::Float, Device::Cpu)).into(),
            true,
        ).unwrap();

        assert_eq!(vec![10, 2], y_p.size());

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
    pub fn test_save_load_net() -> Result<(), ThoError> {
        set_max_level(Info);
        let mut vs_0 = VarStore::new(Cpu);

        let n = 200;
        let net_0 = test_net(&vs_0, n)?;

        // let t_0 = Tensor::from_slice(sin_v(1000, 20.).as_slice());
        let t_0 = Tensor::from_slice(pulse_in_rand(n, &[], 0.05, 1., 0.1).as_slice());
        let t_1 = Tensor::from_slice(sin_v(n, 22.).as_slice());

        train(
            &Cpu,
            &mut crate::simp::DummyDs::new(t_0.shallow_clone(), t_1.shallow_clone())
                as &mut dyn DataBatch<AnyDims>,
            &net_0,
            |pred: &Tensor, target: &Tensor| {
                // pred.cross_entropy_loss::<Tensor>(&target.totype(Kind::Int64), None, Reduction::Mean, -100, 0.0)

                let pred_y = pred.i((.., 0, 0));
                let loss = pred_y.binary_cross_entropy::<Tensor>(
                    &target.totype(Kind::Float),
                    None,
                    Reduction::Mean,
                );
                debug!("pred_y ={pred_y} ");
                debug!("target ={target} ");

               Ok(loss)
            },
            100,
            nn::AdamW::default().build(&vs_0, 1.0e-4)?,
            |t| t,
            None,
        )?;

        let batch_0 = Tensor::stack(
            (0..10)
                .map(|_| t_0.shallow_clone())
                .collect::<Vec<Tensor>>()
                .as_slice(),
            0,
        )
        .view((10, n as i64))
        .into();
        let batch_1 = Tensor::stack(
            (0..10)
                .map(|_| t_1.shallow_clone())
                .collect::<Vec<Tensor>>()
                .as_slice(),
            0,
        )
        .view((10, n as i64))
        .into();

        debug!(
            "net t_0={:?}, t_1={:?}",
            from::<Vec<f32>>(&net_0.tho_forward(&batch_0, true)?.as_ref().view(-1)),
            from::<Vec<f32>>(&net_0.tho_forward(&batch_1, true)?.as_ref().view(-1))
        );

        vs_0.freeze();
        vs_0.save("/tmp/test_vars.safetensors")?;

        let mut vs_1 = nn::VarStore::new(Device::Cpu);

        let net_1 = test_net(&vs_1, n)?;
        vs_1.load("/tmp/test_vars.safetensors")?;

        debug!(
            "net 1 t_0={:?}, t_1={:?}",
            from::<Vec<f32>>(&net_1.tho_forward(&batch_0, true)?.as_ref().view(-1)),
            from::<Vec<f32>>(&net_1.tho_forward(&batch_1, true)?.as_ref().view(-1))
        );

        assert_eq!(
            from::<Vec<f32>>(&net_0.tho_forward(&batch_0, true)?.as_ref().view(-1)),
            from::<Vec<f32>>(&net_1.tho_forward(&batch_0, true)?.as_ref().view(-1))
        );
        assert_eq!(
            from::<Vec<f32>>(&net_0.tho_forward(&batch_1, true)?.as_ref().view(-1)),
            from::<Vec<f32>>(&net_1.tho_forward(&batch_1, true)?.as_ref().view(-1))
        );

        Ok(())
    }

    fn test_net(
        vs: &VarStore,
        num_features: u32,
    ) -> Result<Box<dyn ThoModule<Batch<AnyDims>, AnyDims>>, ThoError> {
        SeqBuilder::new(num_features, None)
            .batch_norm(BatchNormConfig::default())
            .closure(
                Arc::new(Box::new(|t: &Tensor| {
                    Ok(t.view((t.size()[0], 1, t.size()[1]))) //channel for convolution ("Given groups=1, weight of size [1, 1, 100], expected input[1, 10, 1000] to have 1 channels, but got 10 channels instead)
                })),
                None,
            )
            .conv_1d("conv", num_features / 10, 1, 1, ConvConfig::default(), None)
            .linear_s(
                1,
                #[cfg(debug_assertions)]
                "test".to_string(),
            )
            .func(|t|Tensor::f_sigmoid(t).map_err(|e|e.into()))
            .end_seq()
            .build_tch_module(&vs.root())
    }

    //TODO ignores values when decreasing length
    fn scale_x(series: &[f32], new_length: usize) -> Vec<f32> {
        let mut stretched_series = Vec::with_capacity(new_length);
        let old_length = series.len();
        let factor = (old_length - 1) as f32 / (new_length - 1) as f32;

        for new_index in 0..new_length {
            let old_index = (new_index as f32 * factor) as usize;
            let remainder = new_index as f32 * factor - old_index as f32;

            debug!(
                "new_index={}, old_index={} remainder={}",
                new_index, old_index, remainder
            );

            if old_index == old_length - 1 {
                stretched_series.push(series[old_index]);
            } else {
                let value =
                    series[old_index] * (1.0 - remainder) + series[old_index + 1] * remainder;
                stretched_series.push(value);
            }
        }

        stretched_series
    }

    #[allow(unused)]
    #[test]
    pub fn test_scale_x() {
        let v_5 = (0..5).map(|i| i as f32).collect::<Vec<f32>>();

        let v_10 = (0..10).map(|i| i as f32).collect::<Vec<f32>>();

        let v_20 = (0..20).map(|i| i as f32).collect::<Vec<f32>>();

        debug!("v_5={:?} x={:?}", v_5, scale_x(v_5.as_slice(), 10));
        debug!("v_10={:?} x={:?}", v_10, scale_x(v_10.as_slice(), 10));
        debug!("v_20 ={:?} x={:?}", v_20, scale_x(v_20.as_slice(), 10));

        assert!(
            Tensor::from_slice(scale_x(v_5.as_slice(), 10).as_slice()).allclose(
                &Tensor::from_slice(&[
                    0.0, 0.44444445, 0.8888889, 1.3333334, 1.7777778, 2.2222223, 2.6666667,
                    3.1111112, 3.5555556, 4.0
                ])
                .totype(Float),
                0.02,
                0.02,
                true
            )
        );

        assert!(
            Tensor::from_slice(scale_x(v_10.as_slice(), 10).as_slice()).allclose(
                &Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                    .totype(Float),
                0.02,
                0.02,
                true
            )
        );

        assert!(
            Tensor::from_slice(scale_x(v_20.as_slice(), 10).as_slice()).allclose(
                &Tensor::from_slice(&[
                    0.0, 2.1111112, 4.2222223, 6.3333335, 8.444445, 10.555555, 12.666667,
                    14.777779, 16.88889, 19.0
                ])
                .totype(Float),
                0.02,
                0.02,
                true
            )
        );
    }

    #[allow(dead_code)]
    #[test]
    pub fn test_save_load() -> Result<(), ThoError> {
        let vs_0 = VarStore::new(Cpu);

        let node = test_node();
        let _my_mod: Box<dyn ThoModule<AnyDims, AnyDims>> = node.build_tch_module(&vs_0.root())?;

        debug!("variables={:?}", vs_0.variables());

        assert_eq!(26, vs_0.variables().len());

        vs_0.save("/tmp/test.safetensors")?;

        let loaded_module: ThoModuleWithStore<AnyDims, AnyDims> = ThoModuleWithStore::load("".into(),
           |path: &nn::Path| test_node().build_tch_module(path),
           "/tmp/test.safetensors".into(),
           Cpu,
        )?;

        debug!(
            "loaded variables={:?}",
            loaded_module.var_store().variables()
        );

        assert_eq!(vs_0.variables(), loaded_module.var_store().variables());

        fn test_node() -> Node {
            SeqBuilder::new(123, None)
                .split(
                    vec![
                        SeqBuilder::new(123, None)
                            .batch_norm(BatchNormConfig::default()) //3 weight bias running_var
                            .linear(
                                #[cfg(debug_assertions)]
                                "test_node".to_string(),
                            ) //2 weight bias
                            .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
                            .linear(
                                #[cfg(debug_assertions)]
                                "test_node".to_string(),
                            ) //2 weight bias
                            .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
                            .linear_s(
                                1,
                                #[cfg(debug_assertions)]
                                "test".to_string(),
                            ) //2 weight bias
                            .end_seq(),
                        SeqBuilder::new(321, None)
                            .batch_norm(BatchNormConfig::default()) //3 weight bias running_var
                            .linear(
                                #[cfg(debug_assertions)]
                                "test_node".to_string(),
                            ) //2 weight bias
                            .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
                            .linear_s(
                                1,
                                #[cfg(debug_assertions)]
                                "test".to_string(),
                            ) //2 weight bias
                            .end_seq(),
                    ],
                    Box::new(move |t: &Tensor| Ok(vec![t.i((.., 0, ..)), t.i((.., 1, ..))])),
                    Box::new(move |t: Vec<Tensor>| Ok(Tensor::concat(t.as_slice(), 1))),
                )
                .linear_s(
                    10,
                    #[cfg(debug_assertions)]
                    "test".to_string(),
                ) //2 weight bias
                .conv_1d("conv", 3, 1, 2, ConvConfig::default(), None) //2 weight bias
                .conv_1d("conv", 5, 1, 2, ConvConfig::default(), None) //2 weight bias
                .linear_s(
                    20,
                    #[cfg(debug_assertions)]
                    "test".to_string(),
                ) //2 weight bias
                .func(|t|Tensor::f_sigmoid(t).map_err(|e|e.into()))
                .end_seq()
        }

        Ok(())
    }
}
