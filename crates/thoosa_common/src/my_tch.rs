use core::fmt;
use std::borrow::Borrow;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use log::{debug, error};
use tch::{nn, Kind, TchError, Tensor};

use tch::nn::{ConvConfig, Module};
use tch::nn::{
    BatchNorm, BatchNormConfig, Conv, Init, LinearConfig, ModuleT, Path, batch_norm1d,
};

use thoosa_shared::ThoError;
use thoosa_shared::traits::SimpleId;
use crate::builder::{clc_avg_pool1d_out, clc_conv1d_out, clc_max_pool1d_out, Node};
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

    fn run_layers(&self, t: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut [i64]>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        debug!("MySequential::run_layers: t.size={:?} device={:?} self.layers[0]={:?}", t.as_ref().size(), t.device_idx(), self.layers[0]);

        let output_t = self.layers[0].tho_forward(&t.as_ref().into(), lengths.as_deref_mut(), train)?;

        Ok(self.layers
            .iter()
            .skip(1)
            .enumerate()
            .try_fold(output_t, |output_t, (i, layer)| {
                debug!("MySequential::run_layers={i} output_t.size={:?} output_t.device={:?} layer={:?}", output_t.size(), output_t.device_idx(), layer);

                layer.tho_forward(&output_t, lengths.as_deref_mut(), train)
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

    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut [i64]>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        if self.layers.is_empty() {
            Ok(t.shallow_clone())
        } else {
            match self.split_input {
                None => self.run_layers(t, lengths.as_deref_mut(), train),
                Some((dim, num_features)) => {
                    let splits = t.as_ref().f_split(num_features, dim)?;
                    debug!("MySequential::tho_forward splits.sizes={:?} num_features={num_features} dim={dim}",splits.iter().map(|t| t.size()).collect::<Vec<_>>());

                    let original_lengths = lengths.as_deref().map(|vals| vals.to_vec());
                    let mut updated_lengths: Option<Vec<i64>> = None;

                    let split_r = Tensor::f_concat(
                        splits
                            .into_iter()
                            .enumerate()
                            .map(|(i, split)| {
                                if split.size()[dim as usize] == num_features {
                                    debug!("MySequential::tho_forward i={i} split={:?}",split.size());

                                    let mut lengths_run = original_lengths.as_ref().map(|vals| vals.clone());
                                    let r = self.run_layers(&split.into(), lengths_run.as_deref_mut(), train)?;
                                    if updated_lengths.is_none() {
                                        updated_lengths = lengths_run;
                                    }
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

                    if let (Some(updated), Some(lengths_out)) = (updated_lengths, lengths.as_deref_mut()) {
                        lengths_out.copy_from_slice(updated.as_slice());
                    }

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

    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut [i64]>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        debug!("MyParallel::forward: xs={:?}", t.size());
        let original_lengths = lengths.as_deref().map(|vals| vals.to_vec());
        let mut updated_lengths: Option<Vec<i64>> = None;
        let t = (self.join)(
            (self.split)(t.as_ref())?
                .into_iter()
                .zip(self.layers.iter())
                .map(|(t, net)| {
                    let mut lengths_run = original_lengths.as_ref().map(|vals| vals.clone());
                    let out = net.tho_forward(&t.into(), lengths_run.as_deref_mut(), train).map(|v|v.as_ref().shallow_clone())?;
                    if updated_lengths.is_none() {
                        updated_lengths = lengths_run;
                    }
                    Ok(out)
                })
                .collect::<Result<Vec<Tensor>, ThoError>>()?,
        )?;

        if let (Some(updated), Some(lengths_out)) = (updated_lengths, lengths.as_deref_mut()) {
            lengths_out.copy_from_slice(updated.as_slice());
        }

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

    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut [i64]>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        debug!("MyParallel::forward: xs={:?}", t.size());
        let original_lengths = lengths.as_deref().map(|vals| vals.to_vec());
        let mut updated_lengths: Option<Vec<i64>> = None;
        let t = (self.join)(
            (self.split)(t.as_ref())?
                .into_iter()
                .map(|split_t| {
                    let mut lengths_run = original_lengths.as_ref().map(|vals| vals.clone());
                    let out = self.layer.tho_forward(&split_t.into(), lengths_run.as_deref_mut(), train).map(|v|v.as_ref().shallow_clone())?;
                    if updated_lengths.is_none() {
                        updated_lengths = lengths_run;
                    }
                    Ok(out)
                })
                .collect::<Result<Vec<Tensor>, ThoError>>()?,
        )?;

        if let (Some(updated), Some(lengths_out)) = (updated_lengths, lengths.as_deref_mut()) {
            lengths_out.copy_from_slice(updated.as_slice());
        }

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
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut [i64]>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        if let Some(lengths) = lengths.as_deref_mut() {
            for len in lengths.iter_mut() {
                if *len < 0 {
                    return Err(ThoError::new(format!("MyMaxPool1d::tho_forward length must be >= 0, got {len}")));
                }
                *len = clc_max_pool1d_out(*len as u32, self.kernel_size as u32, self.padding as u32, self.dilation as u32, self.stride as u32)? as i64;
            }
        }
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
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut [i64]>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        if let Some(lengths) = lengths.as_deref_mut() {
            for len in lengths.iter_mut() {
                *len = self.output_size as i64;
            }
        }
        Ok(t.as_ref()
            .f_adaptive_max_pool1d(self.output_size as i64)?
            .0.into())
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
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, _lengths: Option<&mut [i64]>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        Ok(self.batch_norm.forward_t(t.as_ref(), train).into())
    }
}

unsafe impl Send for MyBatchNorm1d {}
unsafe impl Sync for MyBatchNorm1d {}

#[derive(Debug)]
pub struct MyConv<ND> {
    conv: Conv<ND>,
    kernel: u32,
    config: ConvConfig,
}

impl<ND> MyConv<ND> {
    pub fn new(conv: Conv<ND>, kernel: u32, conv_config: ConvConfig) -> Self {
        Self { conv, kernel, config: conv_config }
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for MyConv<[i64; 1]> {
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut [i64]>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        debug!("MyConv::tho_forward: lengths={lengths:?}");
        if let Some(lengths) = lengths.as_deref_mut() {
            for len in lengths.iter_mut() {
                if *len < 0 {
                    return Err(ThoError::new(format!("MyConv::tho_forward length must be >= 0, got {len}")));
                }
                *len = clc_conv1d_out(*len as u32, self.kernel, self.config)? as i64;
            }
        }
        debug!("MyConv::tho_forward lengths={lengths:?}");
        Ok(self.conv.forward_t(t.as_ref(), train).into())//TODO custom conv for errors
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for MyConv<[i64; 2]> {
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, _lengths: Option<&mut [i64]>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        Ok(self.conv.forward_t(t.as_ref(), train).into())//TODO custom conv for errors
    }
}


#[derive(Debug)]
pub struct MyCosineConv1d {
    pub w: Tensor,       // [K, C, L] learnable filters
    ones: Tensor,        // [1, 1, L] buffer for sliding energy
    k: i64,              // out_channels (num filters)
    c: i64,              // in_channels
    l: i64,
    config: ConvConfig
}

impl MyCosineConv1d {
    /// Create a cosine-conv bank with K filters, C input channels, kernel length L.
    /// Weights are N(0, 0.02).
    pub fn new(vs: &nn::Path, c: i64, l: i64, config: ConvConfig) -> Result<Self, TchError> {
        let k = 1;//for single motif
        let device = vs.device();
        let w = vs.var("W", &[1, c, l], nn::Init::Randn { mean: 0.0, stdev: 0.02 });
        // fixed boxcar (sum over L) to build window norms; kept as a (non-trainable) buffer
        let ones = Tensor::ones([1, 1, l], (Kind::Float, device));
        Ok(Self { w, ones, k, c, l, config })
    }

    /// Ensure input is 3D [B, C, T] (adds singleton channel if needed).
    pub fn ensure_3d(x: &Tensor) -> Tensor {
        match x.size().as_slice() {
            // [B, T] -> [B, 1, T]
            [b, t] => x.f_unsqueeze(1).unwrap_or_else(|_| x.shallow_clone()).f_view([*b, 1, *t]).unwrap(),
            _      => x.shallow_clone(),
        }
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for MyCosineConv1d {
    fn tho_forward(&self, x_in: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut [i64]>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        if let Some(lengths) = lengths.as_deref_mut() {
            for len in lengths.iter_mut() {
                if *len < 0 {
                    return Err(ThoError::new(format!("MyCosineConv1d::tho_forward length must be >= 0, got {len}")));
                }
                *len = clc_conv1d_out(*len as u32, self.l as u32, self.config)? as i64;
            }
        }

        let x = Self::ensure_3d(x_in.as_ref());

        // ---- Normalize filters per out-channel: Wn = W / ||W||_2 (over C,L) ----
        // denom: [K, 1, 1]
        let w2      = self.w.f_mul(&self.w)?;                                  // [K,C,L]
        let sum_cl  = w2.f_sum_dim_intlist(&[1, 2][..], true, Kind::Float)?;       // [K,1,1]
        let denom_w = sum_cl.f_sqrt()?.f_clamp_min(1e-12)?;                    // [K,1,1]
        let w_norm  = self.w.f_div(&denom_w)?;                                  // [K,C,L]

        // ---- Numerator: standard cross-correlation with normalized kernels ----
        // y_num: [B, K, T-L+1]
        let y_num = x.f_conv1d(
            &w_norm,              // weight [K,C,L]
            None::<Tensor>,                // bias
            &[1],                 // stride
            &[0],                 // padding
            &[1],                 // dilation
            1,                    // groups
        )?;

        // ---- Denominator: sliding L2 norm of the window across *all* channels ----
        // energy over channels: sum_c x^2 -> [B,1,T]
        let x2        = x.f_mul(&x)?;                                          // [B,C,T]
        let x2_sum_c  = x2.f_sum_dim_intlist(1, true, Kind::Float)?;        // [B,1,T]
        // convolve with ones to get sum of squares per window: [B,1,T-L+1]
        let win_energy = x2_sum_c.f_conv1d(&self.ones, None::<Tensor>, &[1], &[0], &[1], 1)?;
        let denom_x    = win_energy.f_sqrt()?.f_clamp_min(1e-12)?;             // [B,1,T-L+1]

        // ---- Cosine similarity: divide per-position ----
        // broadcasting: [B,K,T’] / [B,1,T’]
        let y = y_num.f_div(&denom_x)?;
        Ok(y.into())
    }
}

#[derive(Debug)]
pub struct MySharpeningHead1d {
    cls: nn::Conv1D,
    k_mu: i64,
    k_hat: i64,
    p: f64,
    temperature: f64,
}

impl MySharpeningHead1d {
    pub fn new(vs: &nn::Path, in_channels: i64, out_channels: i64, k_mu: i64, k_hat: i64, p: f64, temperature: f64) -> Result<Self, ThoError> {
        if in_channels <= 0 {
            return Err(ThoError::new(format!("MySharpeningHead1d::new in_channels must be > 0, got {in_channels}")));
        }
        if out_channels <= 0 {
            return Err(ThoError::new(format!("MySharpeningHead1d::new out_channels must be > 0, got {out_channels}")));
        }
        if k_mu <= 0 || k_mu % 2 == 0 {
            return Err(ThoError::new(format!("MySharpeningHead1d::new k_mu must be an odd positive number, got {k_mu}")));
        }
        if k_hat <= 0 || k_hat % 2 == 0 {
            return Err(ThoError::new(format!("MySharpeningHead1d::new k_hat must be an odd positive number, got {k_hat}")));
        }
        if p <= 0.0 {
            return Err(ThoError::new(format!("MySharpeningHead1d::new p must be > 0, got {p}")));
        }
        if temperature <= 0.0 {
            return Err(ThoError::new(format!("MySharpeningHead1d::new temperature must be > 0, got {temperature}")));
        }

        let cfg = ConvConfig { bias: true, ..ConvConfig::default() };
        let cls = nn::conv1d(vs / "cls", in_channels, out_channels, 1, cfg);

        Ok(Self { cls, k_mu, k_hat, p, temperature })
    }

    fn same_pad(kernel: i64) -> (i64, i64) {
        (kernel / 2, kernel - 1 - kernel / 2)
    }

    fn same_avgpool1d(x: &Tensor, kernel: i64) -> Result<Tensor, ThoError> {
        let (pad_left, pad_right) = Self::same_pad(kernel);
        let padded = x.f_pad([pad_left, pad_right], "replicate", 0.0)?;
        padded.f_avg_pool1d(kernel, 1, 0, false, true).map_err(|e| e.into())
    }

    fn morph_opening1d(x: &Tensor, kernel: i64) -> Result<Tensor, ThoError> {
        let (pad_left, pad_right) = Self::same_pad(kernel);
        let erode = x
            .f_neg()?
            .f_pad([pad_left, pad_right], "replicate", 0.0)?
            .f_max_pool1d(kernel, 1, 0, 1, false)?
            .f_neg()?;
        erode
            .f_pad([pad_left, pad_right], "replicate", 0.0)?
            .f_max_pool1d(kernel, 1, 0, 1, false)
            .map_err(|e| e.into())
    }

    pub fn forward_with_thp(&self, sims: &Tensor) -> Result<(Tensor, Tensor), ThoError> {
        let sims_shape = sims.size();
        if sims_shape.len() != 3 {
            return Err(ThoError::new(format!("MySharpeningHead1d::forward_with_thp expected [B,C,T], got {:?}", sims_shape)));
        }

        let mu = Self::same_avgpool1d(sims, self.k_mu)?;
        let mu2 = Self::same_avgpool1d(&sims.f_mul(sims)?, self.k_mu)?;
        let var = mu2.f_sub(&mu.f_mul(&mu)?)?.f_clamp_min(1e-12)?;
        let z = sims.f_sub(&mu)?.f_div(&var.f_sqrt()?)?;

        let opening = Self::morph_opening1d(&z, self.k_hat)?;
        let th = z.f_sub(&opening)?;
        let mut thp = th.f_relu()?;
        if (self.p - 1.0).abs() > f64::EPSILON {
            thp = thp.f_pow_tensor_scalar(self.p)?;
        }

        let logits = self.cls.forward(&thp) / self.temperature;
        Ok((logits, thp))
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for MySharpeningHead1d {
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, _lengths: Option<&mut [i64]>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        self.forward_with_thp(t.as_ref()).map(|(logits, _thp)| logits.into())
    }
}

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
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut [i64]>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        if let Some(lengths) = lengths.as_deref_mut() {
            for len in lengths.iter_mut() {
                if *len < 0 {
                    return Err(ThoError::new(format!("MyAvgPool1d::tho_forward length must be >= 0, got {len}")));
                }
                *len = clc_avg_pool1d_out(*len as u32, self.kernel_size as u32, self.padding as u32, self.stride as u32) as i64;
            }
        }
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

    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, lengths: Option<&mut [i64]>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError>;
}

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
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, _lengths: Option<&mut [i64]>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        (self.func)(t.as_ref()).map(|t|t.into())
    }
}

pub struct MyClosureModule {
    //TODO Tensor->ThoTensor
    func: Arc<Box<dyn Fn(&Tensor, Option<&[i64]>) -> Result<Tensor, ThoError>>>,
}

unsafe impl Send for MyClosureModule {}
unsafe impl Sync for MyClosureModule {}

impl MyClosureModule {
    pub fn new(func: Arc<Box<dyn Fn(&Tensor, Option<&[i64]>) -> Result<Tensor, ThoError>>>) -> Self {
        Self { func }
    }
}

impl Debug for MyClosureModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("MyClosureModule").finish()
    }
}

impl<DI: Dims + Send + Sync, DO: Dims + Send + Sync> ThoModule<DI, DO> for MyClosureModule {
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, lengths: Option<&mut [i64]>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        (self.func)(
            t.as_ref(),
            lengths.as_deref()
        ).map(|v|v.into())
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

    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, _lengths: Option<&mut [i64]>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
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
    path: T,
    in_dim: i64,
    out_dim: i64,
    c: LinearConfig,
    #[cfg(debug_assertions)] descr: String,
) -> Result<MyLinear<DI, DO>, ThoError> {
    assert!(in_dim > 0);
    assert!(out_dim > 0);

    let path = path.borrow();
    let bs = if c.bias {
        let bs_init = c.bs_init.unwrap_or_else(|| {
            let bound = 1.0 / (in_dim as f64).sqrt();
            Init::Uniform {
                lo: -bound,
                up: bound,
            }
        });
        Some(path.f_var("bias", &[out_dim], bs_init)?)
    } else {
        None
    };

    Ok(MyLinear {
        ws: path.f_var("weight", &[out_dim, in_dim], c.ws_init)?,
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
) -> Result<Node, ThoError> {
    let mut builder = SeqBuilder::new(num_features, None)?;

    let dense_block = |b: &mut SeqBuilder| -> Result<(), ThoError> {
        b.n_blocks(
            |b: &mut SeqBuilder, _i| {
                b.func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?
                    .conv_1d("conv", kernel, channels, channels, ConvConfig::default(), None)?
                    .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?
                    .conv_1d("conv", kernel, channels, channels, ConvConfig::default(), None)?
                    .end()
            },
            num_layers_in_block,
        )?;
        b.end()
    };

    builder
        .conv_1d("conv", kernel, channels, channels, ConvConfig::default(), None)?
        .func(|t| t.f_elu().map_err(|e|e.into()))?
        .max_pool(3, 2, 1)?
        .n_blocks(
            |b: &mut SeqBuilder, _i| {
                b.block(dense_block)?
                    .block(|b: &mut SeqBuilder| {
                        b.func(|t|Tensor::f_elu(t).map_err(|e|e.into()))?
                            .conv_1d("conv", kernel, channels, channels, ConvConfig::default(), None)?
                            .func(|t|Tensor::f_elu(t).map_err(|e|e.into()))?
                            .conv_1d("conv", kernel, channels, channels, ConvConfig::default(), None)?
                            .end()
                    })?
                    .end()
            },
            num_blocks,
        )?
        .block(dense_block)?
        .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?
        .avg_pool(7, 3, 0)?
        .dropout()?
        .linear_s(
            6,
            #[cfg(debug_assertions)]
            "test".to_string(),
            "todo",
        )?
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


#[derive(Debug)]
pub struct MyAttnPool1d {
    proj: nn::Conv1D,
    v: nn::Conv1D,
}

impl MyAttnPool1d {
    pub fn new(vs: &nn::Path, c: i64, attn_hidden: i64) -> Self {
        let mut cfg = nn::ConvConfig { ..Default::default() };
        cfg.bias = true;
        let proj = nn::conv1d(vs / "proj", c, attn_hidden, 1, cfg);

        let mut cfg2 = nn::ConvConfig { ..Default::default() };
        cfg2.bias = false;
        let v = nn::conv1d(vs / "v", attn_hidden, 1, 1, cfg2);

        Self { proj, v }
    }

}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for MyAttnPool1d {
    /// h: [B, C, T]
    /// mask: Option<[B, T] bool> where true=valid
    /// returns: (pooled [B,C], attn [B,T])

    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, lengths: Option<&mut [i64]>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        let mask = lengths.and_then(|lengths| mask_from_lengths(lengths, *lengths.iter().max().unwrap(), t.as_ref().device()).ok());

        let h = t.as_ref();

        let p = self.proj.forward(&h);
        let t = p.tanh();

        let scores = self.v.forward(&t).squeeze_dim(1);

        debug!("MyAttnPool1d::tho_forward: p.size={:?}, t.size={:?}, scores.size={:?}", p.size(), t.size(), scores.size());


        let scores = if let Some(m) = mask {
            debug!("MyAttnPool1d::tho_forward:  after mask.size={:?} ", m.size());
            // set padding to large negative so softmax weight ~0 there
            scores.f_masked_fill(&m.squeeze_dim(1).logical_not(), -1e9)?
        } else {
            scores
        };

        debug!("MyAttnPool1d::tho_forward:  after mask scores.size={:?}", scores.size());

        let attn = scores.softmax(-1, Kind::Float);               // [B,T]
        let attn_unsqueezed = attn.unsqueeze(1);
        debug!("MyAttnPool1d::tho_forward: scores.size={:?} attn.size={:?} h.size={:?}, attn_unsqueezed.size={:?}",scores.size(), attn.size(), h.size(), attn_unsqueezed.size());
        let pooled = (h * attn_unsqueezed).sum_dim_intlist(2, false, Kind::Float); // [B,C]

        debug!("MyAttnPool1d::tho_forward: pooled.size={:?}", pooled.size());
        Ok(pooled.into())
    }
}

#[cfg(test)]
mod tch_tests {
    use std::sync::Arc;
    use log::debug;

    use tch::Device::Cpu;
    use tch::Kind::Float;
    use tch::{Device, IndexOp, Kind, Tensor, nn};

    use tch::nn::{BatchNormConfig, ConvConfig, VarStore};

    use thoosa_shared::ThoError;

    use crate::builder::SeqBuilder;
    use crate::builder::{Node, clc_conv1d_out};
    use crate::dims::{AnyDims, Batch};
    use crate::my_tch::ThoModule;
    use crate::train_nets::ThoModuleWithStore;

    #[test]
    pub fn test_conv1_out() {
        let l_out = clc_conv1d_out(50 * 200, 5, ConvConfig::default()).unwrap();

        assert_eq!(l_out, 9996);
    }

    #[test]
    pub fn test_seq_linear_same_size() -> Result<(), ThoError> {
        let seq = SeqBuilder::new(2, None)?
            .linear(
                "todo",
                #[cfg(debug_assertions)]
                "test".to_string(),
            )?
            .end_seq()?;

        let vs = nn::VarStore::new(Device::Cpu);

        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = seq.build_tch_module(&vs.root())?;

        let y_p = net.tho_forward(&Tensor::randn([10, 2], (Float, Cpu)).into(), None, true)?;

        assert_eq!(vec![10, 2], y_p.size());
        Ok(())
    }

    #[test]
    pub fn test_split_seq() -> Result<(), ThoError> {
        // unsafe {
        //     let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        // }

        let res_seq = SeqBuilder::new(10, None)?
            .linear_s(
                1,
                #[cfg(debug_assertions)]
                "test".to_string(),
                "todo",
            )?
            .end_seq()?;

        let conv_seq = SeqBuilder::new(10, None)?
            .conv_1d("conv", 5, 1, 1, ConvConfig::default(), None)?
            .linear_s(
                1,
                #[cfg(debug_assertions)]
                "test".to_string(),
                "todo",
            )?
            .end_seq()?;

        let main = SeqBuilder::new(10, None)?
            .split(
                vec![res_seq, conv_seq],
                Box::new(|t: &Tensor| {
                    Ok(vec![
                        t.i((.., 0, ..)).view((10, 1, 10)),
                        t.i((.., 1, ..)).view((10, 1, 10)),
                    ])
                }),
                Box::new(|t: Vec<Tensor>| Ok(Tensor::stack(t.as_slice(), 1).squeeze())),
            )?
            .end_seq()?;

        let vs = nn::VarStore::new(Cpu);

        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = main.build_tch_module(&vs.root())?;

        let y_p = net.tho_forward(
            &Tensor::randn([10, 2, 10], (Kind::Float, Device::Cpu)).into(),
            None,
            true,
        )?;

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
    pub fn test_cosine_conv_1d_output_bounds() -> Result<(), ThoError> {
        let node = SeqBuilder::new(10, None)?
            .closure(
                Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&[i64]>| {
                    Ok(t.unsqueeze(1))
                })),
                None,
            )?
            .cosine_conv_1d("cosine", 3, 1, 2, ConvConfig { stride: 1, bias: false, ..ConvConfig::default() })?
            .end_seq()?;

        let vs = nn::VarStore::new(Cpu);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = node.build_tch_module(&vs.root())?;
        let out = net.tho_forward(&Tensor::randn([4, 10], (Kind::Float, Device::Cpu)).into(), None, true)?;

        assert_eq!(vec![4, 2, 8], out.size());
        assert!(out.as_ref().max().double_value(&[]) <= 1.0001);
        assert!(out.as_ref().min().double_value(&[]) >= -1.0001);

        Ok(())
    }

    #[test]
    pub fn test_sharpening_head_1d_shapes() -> Result<(), ThoError> {
        let vs = nn::VarStore::new(Cpu);
        let head = super::MySharpeningHead1d::new(&vs.root(), 4, 2, 5, 5, 1.7, 0.5)?;
        let sims = Tensor::randn([3, 4, 16], (Kind::Float, Device::Cpu));

        let (logits, thp) = head.forward_with_thp(&sims)?;

        assert_eq!(vec![3, 2, 16], logits.size());
        assert_eq!(vec![3, 4, 16], thp.size());
        assert!(thp.min().double_value(&[]) >= -1e-7);

        Ok(())
    }


    #[allow(dead_code)]
    fn test_net(
        vs: &VarStore,
        num_features: u32,
    ) -> Result<Box<dyn ThoModule<Batch<AnyDims>, AnyDims>>, ThoError> {
        SeqBuilder::new(num_features, None)?
            .batch_norm(BatchNormConfig::default())?
            .closure(
                Arc::new(Box::new(|t: &Tensor, _lengths: Option<&[i64]>| {
                    Ok(t.view((t.size()[0], 1, t.size()[1]))) //channel for convolution ("Given groups=1, weight of size [1, 1, 100], expected input[1, 10, 1000] to have 1 channels, but got 10 channels instead)
                })),
                None,
            )?
            .conv_1d("conv", num_features / 10, 1, 1, ConvConfig::default(), None)?
            .linear_s(
                1,
                #[cfg(debug_assertions)]
                "test".to_string(),
                "todo",
            )?
            .func(|t|Tensor::f_sigmoid(t).map_err(|e|e.into()))?
            .end_seq()?
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

        let node = test_node()?;
        let _my_mod: Box<dyn ThoModule<AnyDims, AnyDims>> = node.build_tch_module(&vs_0.root())?;

        debug!("variables={:?}", vs_0.variables());

        assert_eq!(26, vs_0.variables().len());

        vs_0.save("/tmp/test.safetensors")?;

        let loaded_module: ThoModuleWithStore<AnyDims, AnyDims> = ThoModuleWithStore::load("".into(),
           |path: &nn::Path| test_node()?.build_tch_module(path),
           "/tmp/test.safetensors".into(),
           Cpu,
        )?;

        debug!(
            "loaded variables={:?}",
            loaded_module.var_store().variables()
        );

        assert_eq!(vs_0.variables(), loaded_module.var_store().variables());

        fn test_node() -> Result<Node, ThoError> {
            SeqBuilder::new(123, None)?
                .split(
                    vec![
                        SeqBuilder::new(123, None)?
                            .batch_norm(BatchNormConfig::default())? //3 weight bias running_var
                            .linear(
                                "todo",
                                #[cfg(debug_assertions)]
                                "test_node".to_string(),
                            )? //2 weight bias
                            .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?
                            .linear(
                                "todo",
                                #[cfg(debug_assertions)]
                                "test_node".to_string(),
                            )? //2 weight bias
                            .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?
                            .linear_s(
                                1,
                                #[cfg(debug_assertions)]
                                "test".to_string(),
                                "todo",
                            )? //2 weight bias
                            .end_seq()?,
                        SeqBuilder::new(321, None)?
                            .batch_norm(BatchNormConfig::default())? //3 weight bias running_var
                            .linear(
                                "todo",
                                #[cfg(debug_assertions)]
                                "test_node".to_string(),
                            )? //2 weight bias
                            .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?
                            .linear_s(
                                1,
                                #[cfg(debug_assertions)]
                                "test".to_string(),
                                "todo",
                            )? //2 weight bias
                            .end_seq()?,
                    ],
                    Box::new(move |t: &Tensor| Ok(vec![t.i((.., 0, ..)), t.i((.., 1, ..))])),
                    Box::new(move |t: Vec<Tensor>| Ok(Tensor::concat(t.as_slice(), 1))),
                )?
                .linear_s(
                    10,
                    #[cfg(debug_assertions)]
                    "test".to_string(),
                    "todo",
                )? //2 weight bias
                .conv_1d("conv", 3, 1, 2, ConvConfig::default(), None)? //2 weight bias
                .conv_1d("conv", 5, 1, 2, ConvConfig::default(), None)? //2 weight bias
                .linear_s(
                    20,
                    #[cfg(debug_assertions)]
                    "test".to_string(),
                    "todo",
                )? //2 weight bias
                .func(|t|Tensor::f_sigmoid(t).map_err(|e|e.into()))?
                .end_seq()
        }

        Ok(())
    }
}
