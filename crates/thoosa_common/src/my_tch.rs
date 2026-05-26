use core::fmt;
use std::any::Any;
use std::borrow::Borrow;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;
use log::{debug, error, info, trace};
use serde::{Deserialize, Serialize};
use tch::{nn, Kind, TchError, Tensor};
use tch::Kind::Float;
use tch::kind::FLOAT_CPU;
use tch::nn::{ConvConfig, Module};
use tch::nn::{
    BatchNorm, BatchNormConfig, Conv, Init, LinearConfig, ModuleT, Path, batch_norm1d,
};
use torch_sys::C_tensor;
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::{panic_to_err, ThoError};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::traits::SimpleId;
use thoosa_shims::quantized_linear::FastQuantizedLinear;
use crate::batches::BatchLengths;
use crate::builder::{clc_avg_pool1d_out, clc_conv1d_out, clc_max_pool1d_out, Node};
// use crate::{data, utils};
use crate::builder::SeqBuilder;
use crate::dims::{AnyDims, Dims, MotifSlices};
use crate::my_tch::MotifInitMode::{CenteredMean, L2NormalizedMean, ZScoreMean};
use crate::tho_tensor::ThoTensor;
use crate::utils::*;

#[derive(Debug)]
pub struct MySequential<DI: Dims, DO: Dims> {
    pub layers: Vec<Box<dyn ThoModule<AnyDims, AnyDims>>>,
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

    fn run_layers(&self, t: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut BatchLengths>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
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

impl<DI: Dims + Send + Sync + 'static, DO: Dims + Send + Sync + 'static> ThoModule<DI, DO> for MySequential<DI, DO> {
    fn debug(&self) {
        for l in &self.layers {
            l.debug();
        }
    }

    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut BatchLengths>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        if self.layers.is_empty() {
            Ok(t.shallow_clone())
        } else {
            match self.split_input {
                None => self.run_layers(t, lengths.as_deref_mut(), train),
                Some((dim, num_features)) => {
                    let splits = t.as_ref().f_split(num_features, dim)?;
                    debug!("MySequential::tho_forward splits.sizes={:?} num_features={num_features} dim={dim}",splits.iter().map(|t| t.size()).collect::<Vec<_>>());

                    let original_lengths = lengths.as_deref().cloned();
                    let mut updated_lengths: Option<BatchLengths> = None;

                    let split_r = Tensor::f_concat(
                        splits
                            .into_iter()
                            .enumerate()
                            .map(|(i, split)| {
                                if split.size()[dim as usize] == num_features {
                                    debug!("MySequential::tho_forward i={i} split={:?}",split.size());

                                    let mut lengths_run = original_lengths.as_ref().map(|vals| vals.clone());
                                    let r = self.run_layers(&split.into(), lengths_run.as_mut(), train)?;
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
                        *lengths_out = updated;
                    }

                    let mean_r = split_r.f_mean_dim(1, false, None)?.f_unsqueeze(1)?;//TODO why mean?

                    debug!("MySequential::tho_forward split_r.size()={:?} mean_r.size()={:?}",split_r.size(),mean_r.size());

                    Ok(mean_r.into())
                }
            }
        }
    }

    fn quantize(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.quantize();
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

impl<DI: Dims + 'static, DO: Dims + 'static> ThoModule<DI, DO> for MyParallel<DI, DO> {
    fn debug(&self) {
        for l in &self.layers {
            l.debug();
        }
    }

    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut BatchLengths>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        debug!("MyParallel::forward: xs={:?}", t.size());
        let original_lengths = lengths.as_deref().cloned();
        let mut updated_lengths: Option<BatchLengths> = None;
        let t = (self.join)(
            (self.split)(t.as_ref())?
                .into_iter()
                .zip(self.layers.iter())
                .map(|(t, net)| {
                    let mut lengths_run = original_lengths.as_ref().map(|vals| vals.clone());
                    let out = net.tho_forward(&t.into(), lengths_run.as_mut(), train).map(|v|v.as_ref().shallow_clone())?;
                    if updated_lengths.is_none() {
                        updated_lengths = lengths_run;
                    }
                    Ok(out)
                })
                .collect::<Result<Vec<Tensor>, ThoError>>()?,
        )?;

        if let (Some(updated), Some(lengths_out)) = (updated_lengths, lengths.as_deref_mut()) {
            *lengths_out = updated;
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

impl<DI: Dims + 'static, DO: Dims + 'static> ThoModule<DI, DO> for Split1<DI, DO> {
    fn debug(&self) {
       self.layer.debug()
    }

    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut BatchLengths>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        debug!("MyParallel::forward: xs={:?}", t.size());
        let original_lengths = lengths.as_deref().cloned();
        let mut updated_lengths: Option<BatchLengths> = None;
        let t = (self.join)(
            (self.split)(t.as_ref())?
                .into_iter()
                .map(|split_t| {
                    let mut lengths_run = original_lengths.as_ref().map(|vals| vals.clone());
                    let out = self.layer.tho_forward(&split_t.into(), lengths_run.as_mut(), train).map(|v|v.as_ref().shallow_clone())?;
                    if updated_lengths.is_none() {
                        updated_lengths = lengths_run;
                    }
                    Ok(out)
                })
                .collect::<Result<Vec<Tensor>, ThoError>>()?,
        )?;

        if let (Some(updated), Some(lengths_out)) = (updated_lengths, lengths.as_deref_mut()) {
            *lengths_out = updated;
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
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut BatchLengths>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        if let Some(lengths) = lengths.as_deref_mut() {
            for len in lengths.as_mut_slice().iter_mut() {
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
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut BatchLengths>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        if let Some(lengths) = lengths.as_deref_mut() {
            for len in lengths.as_mut_slice().iter_mut() {
                *len = self.output_size as i64;
            }
        }
        Ok(t.as_ref()
            .f_adaptive_max_pool1d(self.output_size as i64)?
            .0.into())
    }
}

#[derive(Debug)]
pub struct AdaptiveAvgPool1d {
    output_size: u32
}

impl AdaptiveAvgPool1d {
    pub fn new(output_size: u32) -> Self {
        Self { output_size }
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for AdaptiveAvgPool1d {
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut BatchLengths>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        if let Some(lengths) = lengths.as_deref_mut() {
            for len in lengths.as_mut_slice().iter_mut() {
                *len = self.output_size as i64;
            }
        }
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
        debug!("MyBatchNorm1d::tho_forward: path={:?} num_features={num_features}", path.borrow());

        Self {
            batch_norm: batch_norm1d(path, num_features as i64, config),
        }
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for MyBatchNorm1d {
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, _lengths: Option<&mut BatchLengths>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        debug!("MyBatchNorm1d::tho_forward: t.size={:?}", t.size());
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
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut BatchLengths>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        debug!("MyConv::tho_forward: lengths={lengths:?}");
        if let Some(lengths) = lengths.as_deref_mut() {
            for len in lengths.as_mut_slice().iter_mut() {
                if *len < 0 {
                    return Err(ThoError::new(format!("MyConv::tho_forward length must be >= 0, got {len}")));
                }
                *len = clc_conv1d_out(*len as u32, self.kernel, self.config, "")? as i64;
            }
        }
        debug!("MyConv::tho_forward lengths={lengths:?}");
        panic_to_err!(Ok::<_, ThoError>(self.conv.forward_t(t.as_ref(), train).into()))//TODO custom conv for errors
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for MyConv<[i64; 2]> {
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, _lengths: Option<&mut BatchLengths>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        Ok(self.conv.forward_t(t.as_ref(), train).into())//TODO custom conv for errors
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct InitDummy {}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TrimmedMean {
    #[in_map]
    #[run_config]
    trim_frac: f64
}

#[derive(Serialize, Deserialize, ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug)]
pub enum MotifInitMode {
    Mean(InitDummy),
    Median(InitDummy),
    TrimmedMean (TrimmedMean),
    CenteredMean(InitDummy),
    L2NormalizedMean(InitDummy),
    ZScoreMean(InitDummy),
    CenteredL2NormalizedMean(InitDummy),
}

impl MotifInitMode {
    pub fn list(trim_frac: f64) ->Vec<MotifInitMode> {
        vec![
            MotifInitMode::Mean(InitDummy{}),
            MotifInitMode::Median(InitDummy{}),
            MotifInitMode::TrimmedMean(TrimmedMean{ trim_frac }),
            CenteredMean(InitDummy{}),
            L2NormalizedMean(InitDummy{}),
            ZScoreMean(InitDummy{}),
            CenteredMean(InitDummy{})
        ]
    }
}

impl Default for MotifInitMode {
    fn default() -> MotifInitMode {
        MotifInitMode::CenteredMean(InitDummy{})
    }
}

#[derive(Debug)]
pub struct MyCosineConv1d {
    /// Learnable kernel, shape [1, 1, L]
    pub kernel: Tensor,

    /// Learnable logit scale
    pub alpha: Tensor,

    /// Learnable logit bias
    pub beta: Tensor,

    l: i64,
}

impl MyCosineConv1d {
    pub fn new(
        vs: &Path,
        l: i64,
        init: Option<Arc<ThoTensor<MotifSlices, SynKey>>>,
        init_mode: MotifInitMode,
        a: f64,
        b: f64,
    ) -> Result<Self, TchError> {
        debug!("MyCosineConv1d::new: l={l}, init_mode={init_mode:?}");

        let kernel = match init {
            None => {
                // No motif slices provided: start from zeros.
                vs.f_var("kernel", &[1, 1, l], nn::Init::Const(0.0))?
            }
            Some(motif_slices) => {
                let motif_slices = motif_slices.as_ref().as_ref(); // [N, L]
                let sz = motif_slices.size();
                if sz.len() != 2 {
                    return Err(TchError::Torch(
                        format!(
                            "motif_slices must have shape [num_slices, motif_width], got {:?}",
                            sz
                        )
                            .into(),
                    ));
                }

                let motif_l = sz[1];
                if motif_l != l {
                    return Err(TchError::Torch(
                        format!(
                            "MyCosineConv1d::new: provided l={} but motif_slices width={}",
                            l, motif_l
                        )
                            .into(),
                    ));
                }

                let proto = Self::build_motif_prototype(motif_slices, init_mode)?;
                debug!("MyCosineConv1d::new: proto={proto:?}");

                // Final normalization for cosine matching:
                // zero-mean + unit-L2, then reshape to [1,1,L]
                let kernel_init = Self::finalize_prototype(&proto, true, true)?; // [1,1,L]

                let mut kernel = vs.f_var("kernel", &[1, 1, l], nn::Init::Const(0.0))?;

                tch::no_grad(|| -> Result<(), TchError> {
                    kernel.f_copy_(&kernel_init)?;
                    Ok(())
                })?;

                debug!("MyCosineConv1d::new: kernel={kernel:?}");
                kernel
            }
        };

        let alpha = vs.f_var("alpha", &[], nn::Init::Const(a))?;
        let beta = vs.f_var("beta", &[], nn::Init::Const(b))?;

        Ok(Self {
            kernel,
            alpha,
            beta,
            l,
        })
    }

    /// Accept [B, L] or [B, 1, L].
    pub fn ensure_3d(x: &Tensor) -> Result<Tensor, ThoError> {
        match x.size().as_slice() {
            [b, t] => Ok(x.f_unsqueeze(1)?.f_view([*b, 1, *t])?),
            [_, 1, _] => Ok(x.shallow_clone()),
            other => Err(ThoError::new(format!("input must have shape [B,L] or [B,1,L], got {:?}", other))),
        }
    }

    /// Forward:
    /// x: [B, L] or [B,1,L]
    /// returns logits: [B, 1]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, ThoError> {
        let x = Self::ensure_3d(&x)?;
        let sz = x.size();

        if sz[2] != self.l {
            return Err(ThoError::new(format!("MyCosineConv1d::forward: expected input width {}, got {}", self.l, sz[2])));
        }

        // Normalize kernel each forward: [1,1,L] / ||kernel||
        let k_sq = self.kernel.f_mul(&self.kernel)?;
        let k_norm = k_sq
            .f_sum_dim_intlist(&[1i64, 2i64][..], true, Kind::Float)?
            .f_sqrt()?
            .f_clamp_min(1e-12)?;
        let k_unit = self.kernel.f_div(&k_norm)?;

        // Full-width conv -> [B,1,1]
        let num = x.f_conv1d(
            &k_unit,
            Option::<&Tensor>::None,
            [1],
            [0],
            [1],
            1,
        )?;

        // ||x|| over channel+time -> [B,1,1]
        let x_sq = x.f_mul(&x)?;
        let x_norm = x_sq
            .f_sum_dim_intlist(&[1i64, 2i64][..], true, Kind::Float)?
            .f_sqrt()?
            .f_clamp_min(1e-12)?;

        // cosine similarity in [-1, 1]
        let score = num.f_div(&x_norm)?; // [B,1,1]

        // logit = alpha * score + beta
        Ok(score
            .f_mul(&self.alpha)?
            .f_add(&self.beta)?
            .f_squeeze_dim(2)?)// [B,1]
    }

    fn build_motif_prototype(
        motif_slices: &Tensor, // [N, L]
        mode: MotifInitMode,
    ) -> Result<Tensor, TchError> {
        match mode {
            MotifInitMode::Mean(..) => Self::motif_proto_mean(motif_slices),
            MotifInitMode::Median(..) => Self::motif_proto_median(motif_slices),
            MotifInitMode::TrimmedMean(TrimmedMean{trim_frac}) => {
                Self::motif_proto_trimmed_mean(motif_slices, trim_frac)
            }
            MotifInitMode::CenteredMean(..) => Self::motif_proto_centered_mean(motif_slices),
            MotifInitMode::L2NormalizedMean(..) => {
                Self::motif_proto_l2_normalized_mean(motif_slices)
            }
            MotifInitMode::ZScoreMean(..) => Self::motif_proto_zscore_mean(motif_slices),
            MotifInitMode::CenteredL2NormalizedMean(..) => {
                Self::motif_proto_centered_l2_normalized_mean(motif_slices)
            }
        }
    }

    fn finalize_prototype(
        proto: &Tensor, // [L]
        zero_mean: bool,
        unit_l2: bool,
    ) -> Result<Tensor, TchError> {
        let mut p = proto.shallow_clone();

        if zero_mean {
            let m = p.f_mean(Kind::Float)?;
            p = p.f_sub(&m)?;
        }

        if unit_l2 {
            let p2 = p.f_mul(&p)?;
            let norm = p2.f_sum(Kind::Float)?.f_sqrt()?.f_clamp_min(1e-12)?;
            p = p.f_div(&norm)?;
        }

        p.f_view([1, 1, p.size()[0]])
    }

    fn motif_proto_mean(motif_slices: &Tensor) -> Result<Tensor, TchError> {
        motif_slices.f_mean_dim(&[0i64][..], false, Kind::Float)
    }

    fn motif_proto_median(motif_slices: &Tensor) -> Result<Tensor, TchError> {
        let (values, _indices) = motif_slices.f_median_dim(0, false)?;
        Ok(values)
    }

    fn motif_proto_trimmed_mean(
        motif_slices: &Tensor,
        trim_frac: f64,
    ) -> Result<Tensor, TchError> {
        let sz = motif_slices.size();
        let n = sz[0];

        if !(0.0..0.5).contains(&trim_frac) {
            return Err(TchError::Torch(
                format!("trim_frac must be in [0.0, 0.5), got {trim_frac}").into(),
            ));
        }

        let trim = ((n as f64) * trim_frac).floor() as i64;
        let keep = n - 2 * trim;
        if keep <= 0 {
            return Err(TchError::Torch(
                format!(
                    "trim_frac={} is too large for number of slices n={}",
                    trim_frac, n
                )
                    .into(),
            ));
        }

        let (sorted, _idx) = motif_slices.f_sort(0, false)?;
        let trimmed = sorted.f_narrow(0, trim, keep)?;
        trimmed.f_mean_dim(&[0i64][..], false, Kind::Float)
    }

    fn motif_proto_centered_mean(motif_slices: &Tensor) -> Result<Tensor, TchError> {
        let row_mean = motif_slices.f_mean_dim(&[1i64][..], true, Kind::Float)?; // [N,1]
        let centered = motif_slices.f_sub(&row_mean)?;
        centered.f_mean_dim(&[0i64][..], false, Kind::Float)
    }

    fn motif_proto_l2_normalized_mean(motif_slices: &Tensor) -> Result<Tensor, TchError> {
        let x2 = motif_slices.f_mul(motif_slices)?;
        let row_norm = x2
            .f_sum_dim_intlist(&[1i64][..], true, Kind::Float)?
            .f_sqrt()?
            .f_clamp_min(1e-12)?;
        let normed = motif_slices.f_div(&row_norm)?;
        normed.f_mean_dim(&[0i64][..], false, Kind::Float)
    }

    fn motif_proto_zscore_mean(motif_slices: &Tensor) -> Result<Tensor, TchError> {
        let row_mean = motif_slices.f_mean_dim(&[1i64][..], true, Kind::Float)?;
        let centered = motif_slices.f_sub(&row_mean)?;

        let c2 = centered.f_mul(&centered)?;
        let row_var = c2.f_mean_dim(&[1i64][..], true, Kind::Float)?;
        let row_std = row_var.f_sqrt()?.f_clamp_min(1e-12)?;

        let z = centered.f_div(&row_std)?;
        z.f_mean_dim(&[0i64][..], false, Kind::Float)
    }

    fn motif_proto_centered_l2_normalized_mean(
        motif_slices: &Tensor,
    ) -> Result<Tensor, TchError> {
        let row_mean = motif_slices.f_mean_dim(&[1i64][..], true, Kind::Float)?;
        let centered = motif_slices.f_sub(&row_mean)?;

        let c2 = centered.f_mul(&centered)?;
        let row_norm = c2
            .f_sum_dim_intlist(&[1i64][..], true, Kind::Float)?
            .f_sqrt()?
            .f_clamp_min(1e-12)?;

        let normed = centered.f_div(&row_norm)?;
        normed.f_mean_dim(&[0i64][..], false, Kind::Float)
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for MyCosineConv1d {
    fn tho_forward(&self, x_in: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut BatchLengths>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
       Ok(self.forward(x_in.as_ref())?.into())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum BinaryHeadPool {
    Max,
    Mean,
    Lse { tau: f64 },
}

#[derive(Debug)]
pub struct BinaryHead {
    alpha: Tensor,
    beta: Tensor,
    pool: BinaryHeadPool,
}

impl BinaryHead {
    pub fn new(vs: &nn::Path, pool: BinaryHeadPool) -> Self {
        let alpha = vs.var("alpha", &[], nn::Init::Const(5.0));
        let beta = vs.var("beta", &[], nn::Init::Const(0.0));
        Self { alpha, beta, pool }
    }

    pub fn forward_max(&self, sims: &Tensor) -> Result<Tensor, ThoError> {
        let pooled = pool_time_max(sims).map_err(|e| ThoError::from_error(format!("BinaryHead::forward_max failed, e={e:?}"), e))?;
        pooled.f_mul(&self.alpha)?.f_add(&self.beta).map_err(|e| e.into())
    }

    pub fn forward_lse(&self, sims: &Tensor, tau: f64) -> Result<Tensor, ThoError> {
        debug!("BinaryHead::forward_lse: sims={sims:?}");

        if tau <= 0.0 {
            return Err(ThoError::new(format!("BinaryHead::forward_lse tau must be > 0, got {tau}")));
        }
        let pooled = pool_time_lse(sims, tau).map_err(|e| ThoError::from_error(format!("BinaryHead::forward_lse failed, e={e:?}"), e))?;
        pooled.f_mul(&self.alpha)?.f_add(&self.beta).map_err(|e| e.into())
    }

    pub fn forward_mean(&self, sims: &Tensor) -> Result<Tensor, ThoError> {
        let pooled = pool_time_mean(sims).map_err(|e| ThoError::from_error(format!("BinaryHead::forward_mean failed, e={e:?}"), e))?;
        pooled.f_mul(&self.alpha)?.f_add(&self.beta).map_err(|e| e.into())
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for BinaryHead {
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, _lengths: Option<&mut BatchLengths>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        let logits = match self.pool {
            BinaryHeadPool::Max => self.forward_max(t.as_ref())?,
            BinaryHeadPool::Mean => self.forward_mean(t.as_ref())?,
            BinaryHeadPool::Lse { tau } => self.forward_lse(t.as_ref(), tau)?,
        };
        Ok(logits.into())
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
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, _lengths: Option<&mut BatchLengths>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
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
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut BatchLengths>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        if let Some(lengths) = lengths.as_deref_mut() {
            for len in lengths.as_mut_slice().iter_mut() {
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

#[derive(Debug)]
pub struct MyDropout {
    p: f64,
}

impl MyDropout {
    pub fn new(p: f64) -> Self {
        Self { p }
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for MyDropout {
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, _lengths: Option<&mut BatchLengths>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        Ok(t.as_ref().f_dropout(self.p, train)?.into())
    }
}




pub trait ThoModuleDowncast {
    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn type_name(&self) -> &'static str;
}

impl<T: Any> ThoModuleDowncast for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn type_name(&self) -> &'static str {
        std::any::type_name::<T>()
    }
}

pub trait ThoModule<DI: Dims, DO: Dims>: Debug + /*Sync +*/ Send + ThoModuleDowncast {
    fn debug(&self) {}

    //TODO should consume t: &ThoTensor<DI, SimpleId>?
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, lengths: Option<&mut BatchLengths>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError>;

    fn quantize(&mut self) {

    }
}

pub type Fp = fn(&Tensor) -> Result<Tensor, ThoError>;
pub type LossFp = fn(&Tensor, &Tensor) -> Result<Tensor, ThoError>;//TODO add RC as param and make loss function net-specific

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
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, _lengths: Option<&mut BatchLengths>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        (self.func)(t.as_ref()).map(|t|t.into())
    }
}

pub struct MyClosureModule {
    //TODO Tensor->ThoTensor
    func: Arc<Box<dyn Fn(&Tensor, Option<&BatchLengths>) -> Result<Tensor, ThoError>>>,
}

unsafe impl Send for MyClosureModule {}
unsafe impl Sync for MyClosureModule {}

impl MyClosureModule {
    pub fn new(func: Arc<Box<dyn Fn(&Tensor, Option<&BatchLengths>) -> Result<Tensor, ThoError>>>) -> Self {
        Self { func }
    }
}

impl Debug for MyClosureModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("MyClosureModule").finish()
    }
}

impl<DI: Dims + Send + Sync, DO: Dims + Send + Sync> ThoModule<DI, DO> for MyClosureModule {
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, lengths: Option<&mut BatchLengths>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
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

    quantized:Option<FastQuantizedLinear>,

    #[cfg(any(debug_assertions, tho_debug))]
    descr: String,
}

unsafe impl<DI: Dims, DO: Dims> Send for MyLinear<DI, DO> {}
unsafe impl<DI: Dims, DO: Dims> Sync for MyLinear<DI, DO> {}


impl<DI, DO> MyLinear<DI, DO>
where
    DI: Dims + Send + Sync + 'static,
    DO: Dims + Send + Sync + 'static
{
    pub fn new(
        ws: Tensor, bs: Option<Tensor>, _pd_di: PhantomData<DI>, _pd_do: PhantomData<DO>,
        #[cfg(any(debug_assertions, tho_debug))]
        descr: String
    ) -> Self {

        Self {
            quantized: None,
            ws, bs, _pd_di, _pd_do,

            #[cfg(any(debug_assertions, tho_debug))]
            descr
        }
    }
}


impl<DI, DO> ThoModule<DI, DO> for MyLinear<DI, DO>
where
    DI: Dims + Send + Sync + 'static,
    DO: Dims + Send + Sync + 'static
{
    fn debug(&self) {
        debug!("ThoModule for MyLinear::debug: ws.device={:?} bs.device={:?}",self.ws.device(),self.bs.as_ref().map(|bs| bs.device()));
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="MyLinear::tho_forward"))]
    fn tho_forward(&self, in_t: &ThoTensor<DI, SimpleId>, _lengths: Option<&mut BatchLengths>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        #[cfg(any(debug_assertions, tho_debug))]
        {
            debug!("Module for MyLinear::forward: descr={} in_t={in_t:?} self.ws={:?} self.bs={:?}", self.descr, self.ws, self.bs);
            debug!("Module for MyLinear::forward: devices ws={:?} bs={:?} t={:?}",self.ws.device(),self.bs.as_ref().map(|bs| bs.device()),in_t.device_idx());
            debug!("Module for MyLinear::forward: sizes ws={:?} bs={:?} t={:?}",self.ws.size(),self.bs.as_ref().map(|bs| bs.size()),in_t.size());
        }
        match &self.quantized {
            None => {
                match in_t.as_ref().f_linear(&self.ws, self.bs.as_ref()) {
                    Ok(t) => Ok(t.into()),
                    Err(e) => {
                        error!("{:?}", e);
                        #[cfg(any(debug_assertions, tho_debug))]
                        error!("failed layer descr: {} ", self.descr);

                        Err(e.into())
                    }
                }
            }
            Some(quantized) => {

                match quantized.try_forward(&in_t.as_ref().to_kind(in_t.as_ref().kind())) {
                    Ok(t) => Ok(t.into()),
                    Err(e) => {
                        error!("{:?}", e);
                        #[cfg(any(debug_assertions, tho_debug))]
                        error!("failed layer descr: {} ", self.descr);

                        Err(e.into())
                    }
                }
            }
        }
    }

    fn quantize(&mut self) {
        let ws_fl = self.ws.as_ref().to_kind(Float);
        let bs_fl = self.bs.as_ref().map(|t| t.to_kind(Float));

        debug!("MyLinear::new: ws={:?} ws_fl={ws_fl:?} bs={:?} bs_fl={bs_fl:?} ", self.ws, self.bs);

        self.quantized =  Some(
            FastQuantizedLinear::from_float_weights(
                // &Tensor::rand(ws.as_ref().size(), FLOAT_CPU),
                &ws_fl,
                // bs.as_ref().map(|t| Tensor::rand(t.size(), FLOAT_CPU)).as_ref()
                bs_fl.as_ref()
            ).expect("can't quantize")
        );
    }
}

pub fn my_linear<'a, T: Borrow<Path<'a>>, DI: Dims + 'static, DO: Dims + 'static>(
    path: T,
    in_dim: i64,
    out_dim: i64,
    c: LinearConfig,
    #[cfg(any(debug_assertions, tho_debug))]
    descr: String,
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

    Ok(MyLinear::new(
        path.f_var("weight", &[out_dim, in_dim], c.ws_init)?,
        bs,
        Default::default(),
        Default::default(),
        #[cfg(any(debug_assertions, tho_debug))]
        descr,
    ))
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
        .dropout(0.5)?
        .linear_s(
            6,
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
        net_batch.push(pad_to(&motifs_p_t, &[motif_batch_size], 0.0).expect("pad_to failed"));
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

    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, lengths: Option<&mut BatchLengths>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        let mask = lengths.and_then(|lengths| mask_from_lengths(lengths.as_slice(), *lengths.as_slice().iter().max().unwrap(), t.as_ref().device()).ok());

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

fn valid_time_mask(lengths: &[i64], batch: i64, max_l: i64, device: tch::Device) -> Result<Tensor, ThoError> {
    if lengths.len() != batch as usize {
        return Err(ThoError::new(format!(
            "SelfAttention::tho_forward lengths mismatch: batch={batch} lengths.len={}",
            lengths.len()
        )));
    }
    let pos = Tensor::f_arange(max_l, (Kind::Int64, device))?.f_unsqueeze(0)?;
    let len = Tensor::f_from_slice(lengths)?.f_to_device(device)?.f_view([batch, 1])?;
    pos.f_lt_tensor(&len).map_err(|e| e.into())
}

fn self_attention_forward_t(
    x: Tensor,
    key_valid: Tensor,
    w_q: &nn::Linear,
    w_k: &nn::Linear,
    w_v: &nn::Linear,
    w_o: &nn::Linear,
    n_heads: i64,
    head_dim: i64,
    dropout: f64,
    train: bool,
) -> Result<Tensor, ThoError> {
    let (b, l, _d) = {
        let s = x.size();
        (s[0], s[1], s[2])
    };

    trace!("SelfAttention::tho_forward: key_valid=\n{key_valid}");

    debug!(
        "SelfAttention::tho_forward: x={:?}, key_valid={:?}, w_q.ws={:?}, w_q.bs={:?}, w_k.ws={:?}, w_k.bs={:?}, w_v.ws={:?}, w_v.bs={:?}, w_o.ws={:?}, w_o.bs={:?}, n_heads={:?}, head_dim={:?}, dropout={:?}",
        x,key_valid,w_q.ws,w_q.bs.as_ref().map(|v| v),w_k.ws,w_k.bs.as_ref().map(|v| v),w_v.ws,w_v.bs.as_ref().map(|v| v),w_o.ws,w_o.bs.as_ref().map(|v| v),n_heads,head_dim,dropout
    );

    let q = x
        .f_linear(&w_q.ws, w_q.bs.as_ref())?
        .f_view([b, l, n_heads, head_dim])?
        .f_transpose(1, 2)?;

    let k = x
        .f_linear(&w_k.ws, w_k.bs.as_ref())?
        .f_view([b, l, n_heads, head_dim])?
        .f_transpose(1, 2)?;

    let v = x
        .f_linear(&w_v.ws, w_v.bs.as_ref())?
        .f_view([b, l, n_heads, head_dim])?
        .f_transpose(1, 2)?;

    drop(x);

    let scale = (head_dim as f64).sqrt();

    let attn = {
        let scores = {
            let mut raw_scores = panic_to_err!(q.f_matmul(&k.f_transpose(-2, -1)?)?.f_div_scalar(scale))?;

            let masked_scores = {
                let invalid_mask = key_valid
                    .f_logical_not()?
                    .f_unsqueeze(1)?
                    .f_unsqueeze(2)?;

                drop(key_valid);

                panic_to_err!(raw_scores.f_masked_fill_(&invalid_mask, -1e4))?
            };

            drop(raw_scores);
            masked_scores
        };

        let attn = scores
            .f_softmax(-1, scores.kind())?
            .f_dropout(dropout, train)?;

        drop(scores);
        attn
    };

    drop(q);
    drop(k);

    let y = attn
        .f_matmul(&v)?
        .f_transpose(1, 2)?
        .f_contiguous()?
        .f_view([b, l, n_heads * head_dim])?;

    drop(attn);
    drop(v);

    let out = y.f_linear(&w_o.ws, w_o.bs.as_ref())?;
    drop(y);

    out.f_dropout(dropout, train).map_err(Into::into)
}

fn attn_block_forward_t(
    x: &Tensor,
    key_valid: Tensor,
    ln1: &nn::LayerNorm,
    ln2: &nn::LayerNorm,
    w_q: &nn::Linear,
    w_k: &nn::Linear,
    w_v: &nn::Linear,
    w_o: &nn::Linear,
    ff1: &nn::Linear,
    ff2: &nn::Linear,
    n_heads: i64,
    head_dim: i64,
    dropout: f64,
    train: bool,
) -> Result<Tensor, ThoError> {

    debug!("attn_block_forward_t: x.kind={:?} ln1.ws.kind={:?}", x.kind(), ln1.ws.as_ref().map(|ws|ws.kind()));

    let h = x.f_layer_norm(
        ln1.normalized_shape.as_slice(),
        ln1.ws.as_ref(), //.map(|v|v.to_kind(Float)),//TODO it needs Float only on CPU
        ln1.bs.as_ref(), //.map(|v|v.to_kind(Float)),
        1e-5,
        true,
    )?;//.to_kind(x.kind())

    let x = x.f_add(
        &self_attention_forward_t(h, key_valid, w_q, w_k, w_v, w_o, n_heads, head_dim, dropout, train)?
    )?;

    debug!("attn_block_forward_t: x.kind={:?} ln2.ws.kind={:?}", x.kind(), ln2.ws.as_ref().map(|ws|ws.kind()));

    let f = x.f_layer_norm(//x.to_kind(Float).
        ln2.normalized_shape.as_slice(),
        ln2.ws.as_ref(), //.map(|v|v.to_kind(Float)),//TODO it needs Float only on CPU
        ln2.bs.as_ref(), //.map(|v|v.to_kind(Float)),
        1e-5,
        true,
    )?;//.to_kind(x.kind())

    debug!("attn_block_forward_t: f={f:?}");

    let f = f.f_linear(&ff1.ws, ff1.bs.as_ref())?
    .f_gelu("none")?
    .f_dropout(dropout, train)?
    .f_linear(&ff2.ws, ff2.bs.as_ref())?
    .f_dropout(dropout, train)?;

    let out = x.f_add(&f)?;

    debug!("attn_block_forward_t: out={out:?}");

    Ok(out)
}

#[derive(Debug)]
pub struct SelfAttention {
    ln1: nn::LayerNorm,
    ln2: nn::LayerNorm,
    w_q: nn::Linear,
    w_k: nn::Linear,
    w_v: nn::Linear,
    w_o: nn::Linear,
    ff1: nn::Linear,
    ff2: nn::Linear,
    n_heads: i64,
    head_size: i64,
    dropout: f64,
}

impl SelfAttention {
    pub fn new(vs: &nn::Path, d_model: i64, n_heads: i64, ff_mult: i64, dropout: f64) -> Result<Self, ThoError> {
        debug!("SelfAttention::new: d_model={d_model}, n_heads={n_heads}, ff_mult={ff_mult}, dropout={dropout} ");

        if d_model <= 0 {
            return Err(ThoError::new(format!("SelfAttention::new d_model must be > 0, got {d_model}")));
        }
        if n_heads <= 0 {
            return Err(ThoError::new(format!("SelfAttention::new n_heads must be > 0, got {n_heads}")));
        }
        if d_model % n_heads != 0 {
            return Err(ThoError::new(format!("SelfAttention::new d_model must be divisible by n_heads, got d_model={d_model} n_heads={n_heads}")));
        }
        if ff_mult <= 0 {
            return Err(ThoError::new(format!("SelfAttention::new ff_mult must be > 0, got {ff_mult}")));
        }
        if !(0.0..1.0).contains(&dropout) {
            return Err(ThoError::new(format!("SelfAttention::new dropout must be in [0,1), got {dropout}")));
        }

        let head_size = d_model / n_heads;
        let lin = |name: &str| nn::linear(vs / name, d_model, d_model, Default::default());
        let ln_cfg = nn::LayerNormConfig { eps: 1e-5, ..Default::default() };

        Ok(Self {
            ln1: nn::layer_norm(vs / "ln1", vec![d_model], ln_cfg),
            ln2: nn::layer_norm(vs / "ln2", vec![d_model], ln_cfg),
            w_q: lin("q"),
            w_k: lin("k"),
            w_v: lin("v"),
            w_o: lin("o"),
            ff1: nn::linear(vs / "ff1", d_model, d_model * ff_mult, Default::default()),
            ff2: nn::linear(vs / "ff2", d_model * ff_mult, d_model, Default::default()),
            n_heads,
            head_size,
            dropout,
        })
    }
}

fn sinusoidal_positional_encoding(length: i64, d_model: i64, device: tch::Device, kind: Kind) -> Result<Tensor, ThoError> {
    let half = (d_model + 1) / 2;
    let pos = Tensor::f_arange(length, (Kind::Float, device))?.f_unsqueeze(1)?;
    let exponent = Tensor::f_arange(half, (Kind::Float, device))?.f_mul_scalar(-2.0 * 10000_f64.ln() / d_model as f64)?;
    let div_term = exponent.f_exp()?.f_unsqueeze(0)?;
    let angles = pos.f_mul(&div_term)?;
    Tensor::f_stack(&[angles.f_sin()?, angles.f_cos()?], -1)?
        .f_flatten(1, 2)?
        .f_narrow(1, 0, d_model)?
        .f_to_kind(kind)
        .map_err(|e| e.into())
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for SelfAttention {
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, lengths: Option<&mut BatchLengths>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        debug!("SelfAttention::tho_forward:  t.id={:?} t.size={:?} t.kind={:?} lengths={lengths:?}", t.id(), t.size(), t.as_ref().kind());

        let s = t.size();
        if s.len() != 3 {
            return Err(ThoError::new(format!("SelfAttention::tho_forward expected [B,L,D], got {:?}", s)));
        }

        let b = s[0];
        let l = s[1];
        let d = s[2];
        if d != self.n_heads * self.head_size {
            return Err(ThoError::new(format!("SelfAttention::tho_forward d_model mismatch, expected {}, got {d}", self.n_heads * self.head_size)));
        }

        let x = t.as_ref();
        let pos = sinusoidal_positional_encoding(l, d, x.device(), x.kind())?.f_unsqueeze(0)?;
        let mut h = x.f_add(&pos)?;

        let key_valid = if let Some(lengths) = lengths.as_deref() {
            valid_time_mask(lengths.as_slice(), b, l, x.device())?
        } else {
            Tensor::f_ones([b, l], (Kind::Bool, x.device()))?
        };

        let mask_f = key_valid.f_to_kind(t.as_ref().kind())?.f_unsqueeze(-1)?;
        h = h.f_mul(&mask_f)?;
        h = attn_block_forward_t(
            &h,
            key_valid,
            &self.ln1,
            &self.ln2,
            &self.w_q,
            &self.w_k,
            &self.w_v,
            &self.w_o,
            &self.ff1,
            &self.ff2,
            self.n_heads,
            self.head_size,
            self.dropout,
            train,
        )?;

        Ok(h.into())
    }
}

#[cfg(test)]
mod tch_tests {
    use std::ptr;
    use std::sync::Arc;
    use std::time::Instant;
    use log::debug;

    use crate::batches::BatchLengths;
    use tch::Device::Cpu;
    use tch::Kind::{Bool, Float};
    use tch::{Device, IndexOp, Kind, Tensor, nn};
    use tch::kind::FLOAT_CPU;
    use tch::nn::{BatchNormConfig, ConvConfig, VarStore};
    use torch_sys::C_tensor;
    use thoosa_shared::ThoError;
    use thoosa_shims::quantized_linear::FastQuantizedLinear;
    use crate::builder::SeqBuilder;
    use crate::builder::{Node, clc_conv1d_out};
    use crate::dims::{AnyDims, Batch, Dims};
    use crate::my_tch::{valid_time_mask, InitDummy, MotifInitMode, MyLinear, ThoModule};
    use crate::train_nets::ThoModuleWithStore;

    #[test]
    pub fn test_conv1_out() {
        let l_out = clc_conv1d_out(50 * 200, 5, ConvConfig::default(), "").unwrap();

        assert_eq!(l_out, 9996);
    }

    #[test]
    pub fn test_seq_linear_same_size() -> Result<(), ThoError> {
        let seq = SeqBuilder::new(2, None)?
            .linear("todo")?
            .end_seq()?;

        let vs = nn::VarStore::new(Device::Cpu);

        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = seq.build_tch_module(&vs.root())?;

        let y_p = net.tho_forward(&Tensor::randn([10, 2], (Float, Cpu)).into(), None, true)?;

        assert_eq!(vec![10, 2], y_p.size());
        Ok(())
    }

    #[test]
    pub fn test_adaptive_avg_pool_1d_builder() -> Result<(), ThoError> {
        let seq = SeqBuilder::new(20, None)?
            .adaptive_avg_pool(5, 5)?
            .end_seq()?;

        let vs = nn::VarStore::new(Device::Cpu);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = seq.build_tch_module(&vs.root())?;

        let mut lengths = BatchLengths::new(vec![20i64]);
        let y_p = net.tho_forward(&Tensor::randn([1, 1, 20], (Float, Cpu)).into(), Some(&mut lengths), true)?;

        assert_eq!(vec![1, 1, 5], y_p.size());
        assert_eq!(vec![5], lengths.as_slice());

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
                "todo",
            )?
            .end_seq()?;

        let conv_seq = SeqBuilder::new(10, None)?
            .conv_1d("conv", 5, 1, 1, ConvConfig::default(), None)?
            .linear_s(
                1,
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

    //TODO motif nets should fail on unexpected input?
    // #[test]
    // pub fn test_cosine_conv_1d_output_bounds() -> Result<(), ThoError> {
    //     let node = SeqBuilder::new(10, None)?
    //         .closure(
    //             Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
    //                 Ok(t.unsqueeze(1))
    //             })),
    //             None,
    //         )?
    //         .cosine_conv_1d("cosine", 3, None, 0.5, 0.1, MotifInitMode::Mean(InitDummy{}))?
    //         .end_seq()?;
    //
    //     let vs = nn::VarStore::new(Cpu);
    //     let net: Box<dyn ThoModule<AnyDims, AnyDims>> = node.build_tch_module(&vs.root())?;
    //     let out = net.tho_forward(&Tensor::randn([4, 10], (Kind::Float, Device::Cpu)).into(), None, true)?;
    //
    //     assert_eq!(vec![4, 1, 8], out.size());
    //     assert!(out.as_ref().max().double_value(&[]) <= 1.0001);
    //     assert!(out.as_ref().min().double_value(&[]) >= -1.0001);
    //
    //     Ok(())
    // }

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
            .batch_norm(BatchNormConfig::default(), None)?
            .closure(
                Arc::new(Box::new(|t: &Tensor, _lengths: Option<&BatchLengths>| {
                    Ok(t.view((t.size()[0], 1, t.size()[1]))) //channel for convolution ("Given groups=1, weight of size [1, 1, 100], expected input[1, 10, 1000] to have 1 channels, but got 10 channels instead)
                })),
                None,
            )?
            .conv_1d("conv", num_features / 10, 1, 1, ConvConfig::default(), None)?
            .linear_s(
                1,
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
                                                                                           false,
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
                            .batch_norm(BatchNormConfig::default(), None)? //3 weight bias running_var
                            .linear("todo")? //2 weight bias
                            .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?
                            .linear("todo")? //2 weight bias
                            .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?
                            .linear_s(
                                1,
                                "todo",
                            )? //2 weight bias
                            .end_seq()?,
                        SeqBuilder::new(321, None)?
                            .batch_norm(BatchNormConfig::default(), None)? //3 weight bias running_var
                            .linear("todo")? //2 weight bias
                            .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?
                            .linear_s(
                                1,
                                "todo",
                            )? //2 weight bias
                            .end_seq()?,
                    ],
                    Box::new(move |t: &Tensor| Ok(vec![t.i((.., 0, ..)), t.i((.., 1, ..))])),
                    Box::new(move |t: Vec<Tensor>| Ok(Tensor::concat(t.as_slice(), 1))),
                )?
                .linear_s(
                    10,
                    "todo",
                )? //2 weight bias
                .conv_1d("conv", 3, 1, 2, ConvConfig::default(), None)? //2 weight bias
                .conv_1d("conv", 5, 1, 2, ConvConfig::default(), None)? //2 weight bias
                .linear_s(
                    20,
                    "todo",
                )? //2 weight bias
                .func(|t|Tensor::f_sigmoid(t).map_err(|e|e.into()))?
                .end_seq()
        }

        Ok(())
    }

    #[test]
    fn test_valid_time_mask() -> Result<(), ThoError>{

        let mask =valid_time_mask(
            &[5, 11,3,7],
            4,
            11,
            Cpu
        )?;

        assert_eq!(
            mask,
            Tensor::from_slice(&[
                1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,1,
                1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0
            ]).view((4, 11)).totype(Bool)
        );

        Ok(())
    }

    fn call_generated_1<F>(f: F) -> Result<Tensor, ThoError>
    where
        F: FnOnce(*mut *mut C_tensor),
    {
        let mut out: *mut C_tensor = ptr::null_mut();

        f(&mut out as *mut *mut C_tensor);

        // check_torch_err()?;

        if out.is_null() {
            Err(ThoError::new("generated libtorch op returned a null Tensor pointer"))
        } else {
            // Takes ownership of the returned libtorch Tensor pointer.
            Ok(unsafe { Tensor::from_ptr(out) })
        }
    }

    fn as_mut_c_tensor(t: &Tensor) -> *mut C_tensor {
        // The generated libtorch API takes *mut C_tensor even for logical inputs.
        // We do not mutate these tensors from Rust.
        t.as_ptr() as *mut C_tensor
    }

    fn scalar_f32(v: f64) -> Tensor {
        Tensor::from_slice(&[v as f32])
            .to_device(Device::Cpu)
            .to_kind(Kind::Float)
    }

    fn symmetric_i8_scale(t: &Tensor) -> f64 {
        let max_abs = t.abs().max().double_value(&[]);
        if max_abs == 0.0 || !max_abs.is_finite() {
            1.0
        } else {
            max_abs / 127.0
        }
    }

    fn scalar_i64(v: i64) -> Tensor {
        Tensor::from_slice(&[v])
            .to_device(Device::Cpu)
            .to_kind(Kind::Int64)
    }

    //TODO https://chatgpt.com/g/g-p-6802177ab924819193b7de5c74c7014c-rust/c/6a0308a4-c734-83eb-9983-9d324d9cda56
    #[test]
    fn test_quantized_linear() -> Result<(), ThoError>{
        let weight = Tensor::rand(&[3, 100], FLOAT_CPU);
        let bias = Tensor::rand(&[3], FLOAT_CPU);

        let weight_scale = scalar_f32(symmetric_i8_scale(&weight));
        let weight_zero_point = scalar_i64(0);

        let packed_weight = call_generated_1(|out| unsafe {
            torch_sys::c_generated::atg__wrapped_linear_prepack(
                out,
                as_mut_c_tensor(&weight),
                as_mut_c_tensor(&weight_scale),
                as_mut_c_tensor(&weight_zero_point),
                as_mut_c_tensor(&bias),
            );
        })?;


        Ok(())
    }

    #[test]
    fn test_quantized() -> Result<(), ThoError>{
        let vs = nn::VarStore::new(Device::Cpu);

        let linear = nn::linear(
            vs.root(),
            1024,
            512,
            Default::default(),
        );

        let qlinear = FastQuantizedLinear::from_float_linear(&linear)?;

        let xs = Tensor::randn([256, 1024], FLOAT_CPU);

        // Calls quantized::linear_dynamic through C++ shim.
        let y_fast = qlinear.try_forward(&xs)?;

        // Float reference.
        let y_ref = xs.f_linear(&linear.ws, linear.bs.as_ref())?;

        let max_abs_err = (&y_fast - &y_ref)
            .abs()
            .max()
            .double_value(&[]);

        println!("y_fast.size = {:?}", y_fast.size());
        println!("max_abs_err = {max_abs_err}");

        Ok(())
    }

    #[test]
    fn bench_linear() -> Result<(), Box<dyn std::error::Error>> {
        let vs = nn::VarStore::new(Device::Cpu);

        let linear = nn::linear(
            vs.root(),
            4096,
            4096,
            Default::default(),
        );

        let qlinear = FastQuantizedLinear::from_float_linear(&linear)?;

        let xs = Tensor::randn([128, 4096], FLOAT_CPU);

        // Warmup.
        for _ in 0..10 {
            let _ = xs.f_linear(&linear.ws, linear.bs.as_ref())?;
            let _ = qlinear.try_forward(&xs)?;
        }

        let n = 100;

        let t0 = Instant::now();
        for _ in 0..n {
            let _ = xs.f_linear(&linear.ws, linear.bs.as_ref())?;
        }
        let float_elapsed = t0.elapsed();

        let t1 = Instant::now();
        for _ in 0..n {
            let _ = qlinear.try_forward(&xs)?;
        }
        let q_elapsed = t1.elapsed();

        println!("float elapsed: {:?}", float_elapsed);
        println!("quant elapsed: {:?}", q_elapsed);
        println!(
            "speedup: {:.2}x",
            float_elapsed.as_secs_f64() / q_elapsed.as_secs_f64()
        );

        Ok(())
    }
}
