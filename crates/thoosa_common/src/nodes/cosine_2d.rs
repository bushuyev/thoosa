use std::sync::Arc;
use log::debug;
use tch::{nn, Kind, Tensor};
use tch::nn::Path;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::ThoError;
use thoosa_shared::traits::SimpleId;
use crate::batches::BatchLengths;
use crate::debug_plot;
use crate::dims::{Dims, MotifSlices, Spectorgram};
use crate::my_tch::{MotifInitMode, MyCosineConv1d, ThoModule};
use crate::tho_tensor::ThoTensor;
use crate::utils::z_norm_dims;

#[derive(Debug)]
pub struct  MyCosineConv2d {
    kh:i64,//frequencies
    kw:i64,//time
    kernel: Tensor,
    /// Learnable logit scale
    pub alpha: Tensor,

    /// Learnable logit bias
    pub beta: Tensor,

    centered_input_norm:bool
}

impl MyCosineConv2d {
    pub fn new(
        vs: &Path,
        init: Option<Arc<ThoTensor<Spectorgram, SynKey>>>,
        init_mode: MotifInitMode,
        a: f64,
        b: f64,
    ) -> Result<Self, ThoError> {
        let Some(proto) = init else {
            panic!("no kernel")
        };

        debug_plot!(proto.as_ref().as_ref(), "kernel_", plot_repeats=11)?;

        debug!("MyCosineConv2d::new: 0. proto={proto:?}");
        let patches = Self::ensure_motif_patches_4d(&proto.as_ref().as_ref())?; // [N,1,KH,KW]
        let size = patches.size();

        // if sz[2] != kh || sz[3] != kw {
        //     return Err(ThoError::new(format!("expected motif patches [N,1,{kh},{kw}], got {:?}", sz)));
        // }
        let kh = size[2];//frequencies
        let kw = size[3];//time

        let proto = z_norm_dims(&patches.f_sum_dim_intlist(0, false, patches.kind())?, &[1,2], 1e-8)?; // [1,KH,KW]

        // let kernel_init = Self::finalize_prototype_2d(&proto, true, true)?; // [1,1,KH,KW]

        let mut kernel = vs.f_var("kernel", &[1, 1, kh, kw], nn::Init::Const(0.0))?;


        debug!("MyCosineConv2d::new: 1. kernel={kernel:?}");

        tch::no_grad(|| -> Result<(), ThoError> {
            kernel.f_copy_(&proto)?;
            Ok(())
        })?;


        let alpha = vs.f_var("alpha", &[], nn::Init::Const(a))?;
        let beta = vs.f_var("beta", &[], nn::Init::Const(b))?;
        Ok(Self {
            kh,
            kw,
            kernel,
            alpha,
            beta,
            centered_input_norm: true,//TODO
        })
    }

    /// Accept [B, H, W] or [B, 1, H, W].
    pub fn ensure_4d(x: &Tensor) -> Result<Tensor, ThoError> {
        match x.size().as_slice() {
            [b, h, w] => Ok(x.f_view([*b, 1, *h, *w])?),
            [_, 1, _, _] => Ok(x.shallow_clone()),
            other => Err(ThoError::new(format!("input must have shape [B,H,W] or [B,1,H,W], got {:?}", other))),
        }
    }

    /// Accept [N, KH, KW] or [N, 1, KH, KW].
    fn ensure_motif_patches_4d(x: &Tensor) -> Result<Tensor, ThoError> {
        match x.size().as_slice() {
            [n, h, w] => Ok(x.f_view([*n, 1, *h, *w])?),
            [_, 1, _, _] => Ok(x.shallow_clone()),
            other => Err(ThoError::new(format!("motif patches must have shape [N,KH,KW] or [N,1,KH,KW], got {:?}", other))),
        }
    }

    fn finalize_prototype_2d(
        proto: &Tensor,
        zero_mean: bool,
        unit_l2: bool,
    ) -> Result<Tensor, ThoError> {
        let mut p = match proto.size().as_slice() {
            [h, w] => proto.f_view([1, *h, *w])?,
            [1, _, _] => proto.shallow_clone(),
            other => {
                return Err(ThoError::new(format!("expected proto shape [H,W] or [1,H,W], got {:?}", other)));
            }
        };

        if zero_mean {
            let m = p.f_mean(Kind::Float)?;
            p = p.f_sub(&m)?;
        }

        if unit_l2 {
            let norm = p
                .f_mul(&p)?
                .f_sum(Kind::Float)?
                .f_sqrt()?
                .f_clamp_min(1e-12)?;

            p = p.f_div(&norm)?;
        }

        let sz = p.size(); // [1,H,W]
        Ok(p.f_view([1, 1, sz[1], sz[2]])?)
    }

    /// x: [B,H,W] or [B,1,H,W]
    /// returns similarity map: [B,1,H_out,W_out]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, ThoError> {
        debug!("MyCosineConv2d::forward: x={x:?}");

        let x = Self::ensure_4d(x)?;
        let sz = x.size();

        let h = sz[2];
        let w = sz[3];

        if h < self.kh || w < self.kw {
            return Err(ThoError::new(format!("input spectrogram is smaller than kernel: input=({h},{w}), kernel=({},{})", self.kh, self.kw)));
        }

        // Normalize kernel every forward.
        let k_sq = self.kernel.f_mul(&self.kernel)?;
        let k_norm = k_sq
            .f_sum_dim_intlist(&[1i64, 2i64, 3i64][..], true, Kind::Float)?
            .f_sqrt()?
            .f_clamp_min(1e-12)?;

        let k_unit = self.kernel.f_div(&k_norm)?; // [1,1,KH,KW]

        // Numerator:
        //
        // [B,1,H_out,W_out]
        let num = x.f_conv2d(
            &k_unit,
            Option::<&Tensor>::None,
            [1, 1],
            [0, 0],
            [1, 1],
            1,
        )?;

        // Denominator: local ||x_patch||.
        let ones = Tensor::ones(
            &[1, 1, self.kh, self.kw],
            (x.kind(), x.device()),
        );

        let x_sq = x.f_mul(&x)?;

        let local_norm_sq = if self.centered_input_norm {
            // Local sum and sum of squares over each KH x KW patch.
            let local_sum = x.f_conv2d(
                &ones,
                Option::<&Tensor>::None,
                [1, 1],
                [0, 0],
                [1, 1],
                1,
            )?;

            let local_sum_sq = x_sq.f_conv2d(
                &ones,
                Option::<&Tensor>::None,
                [1, 1],
                [0, 0],
                [1, 1],
                1,
            )?;

            let area = (self.kh * self.kw) as f64;

            // sum((x - mean)^2) = sum(x^2) - sum(x)^2 / area
            let correction = local_sum.f_mul(&local_sum)?.f_div_scalar(area)?;
            local_sum_sq.f_sub(&correction)?
        } else {
            x_sq.f_conv2d(
                &ones,
                Option::<&Tensor>::None,
                [1, 1],
                [0, 0],
                [1, 1],
                1,
            )?
        };

        let x_norm = local_norm_sq
            .f_clamp_min(1e-12)?
            .f_sqrt()?;

        let score = num.f_div(&x_norm)?; // cosine-like score

        let out = score
            .f_mul(&self.alpha)?
            .f_add(&self.beta)?;

        debug!("MyCosineConv2d::forward: out={out:?}");

        Ok(out)
    }

}


impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for MyCosineConv2d {

    /// For full-size matching where input H/W equals kernel H/W.
    /// Returns [B,1].
    fn tho_forward(&self, x_in: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut BatchLengths>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        let y = self.forward(x_in.as_ref())?; // [B,1,1,1] if full match
        Ok(y.f_squeeze_dim(3)?.f_squeeze_dim(2)?.into())
    }
}
