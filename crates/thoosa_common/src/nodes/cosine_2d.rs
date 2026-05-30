use std::sync::Arc;
use log::{debug, trace};
use tch::{nn, IndexOp, Kind, Tensor};
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
use crate::utils::plot_spectrogram;


#[derive(Debug, Clone, Copy)]
pub struct CosineConv2dBankConfig {
    pub stride: [i64; 2],
    pub padding: [i64; 2],
    pub dilation: [i64; 2],

    /// If true, subtract mean from every kernel during forward.
    ///
    /// For exact equivalence with your old MyCosineConv2d, keep this false.
    pub center_kernel_forward: bool,

    /// If true, input patch norm is computed as:
    ///
    /// sqrt(sum((x_patch - mean(x_patch))^2))
    ///
    /// For exact equivalence with your old MyCosineConv2d, keep this false.
    pub center_input_norm: bool,

    pub eps: f64,
}

impl Default for CosineConv2dBankConfig {
    fn default() -> Self {
        Self {
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            center_kernel_forward: false,
            center_input_norm: false,
            eps: 1e-12,
        }
    }
}

/// Shared primitive.
///
/// x:
///     [B, C, H, W]
///
/// kernels:
///     [M, C, KH, KW]
///
/// alpha:
///     [M] or scalar if M = 1
///
/// beta:
///     [M] or scalar if M = 1
///
/// returns:
///     [B, M, H_out, W_out]
pub fn cosine_conv2d_bank_logits(
    x: &Tensor,
    kernels: &Tensor,//TODO remove &
    alpha: &Tensor,
    beta: &Tensor,
    cfg: CosineConv2dBankConfig,
) -> Result<Tensor, ThoError> {
    debug!("cosine_conv2d_bank_logits: x={x:?} kernels={kernels:?} alpha={alpha:?} beta={beta:?} cfg={cfg:?}  x.isnan={}", x.isnan().any());
    debug!("cosine_conv2d_bank_logits: devices x={:?} kernels={:?} alpha={:?} beta={:?} ",  x.device(), kernels.device(), alpha.device(), beta.device());
    trace!("cosine_conv2d_bank_logits: x=\n{x} kernels=\n{kernels} alpha=\n{alpha} beta=\n{beta}");

    let x_sz = x.size();
    if x_sz.len() != 4 {
        return Err(ThoError::new(
            format!("x must have shape [B,C,H,W], got {:?}", x_sz),
        ));
    }

    let b = x_sz[0];
    let c = x_sz[1];
    let h = x_sz[2];
    let w = x_sz[3];

    let k_sz = kernels.size();
    if k_sz.len() != 4 {
        return Err(ThoError::new(format!("kernels must have shape [M,C,KH,KW], got {:?}", k_sz)));
    }

    let m = k_sz[0];
    let kc = k_sz[1];
    let kh = k_sz[2];
    let kw = k_sz[3];

    if kc != c {
        return Err(ThoError::new(format!("kernel input channels must match x channels: x C={}, kernel C={}", c, kc)));
    }

    if h < kh || w < kw {
        return Err(ThoError::new(format!("input spatial size ({h},{w}) smaller than kernel size ({kh},{kw})")));
    }

    // alpha/beta: allow scalar for M=1, or [M].
    let alpha = alpha.f_view([m])?;
    let beta = beta.f_view([m])?;

    let mut k = kernels.shallow_clone(); // [M,C,KH,KW]

    if cfg.center_kernel_forward {
        let k_mean = k.f_mean_dim(&[1i64, 2i64, 3i64][..], true, Kind::Float)?;
        k = k.f_sub(&k_mean)?;
    }

    // Normalize each kernel independently:
    //
    // k_unit[m] = k[m] / ||k[m]||
    let k_norm = k
        .f_mul(&k)?
        .f_sum_dim_intlist(&[1i64, 2i64, 3i64][..], true, Kind::Float)?
        .f_sqrt()?
        .f_clamp_min(cfg.eps)?;

    debug!("cosine_conv2d_bank_logits: k_norm={k_norm:?} k_norm.isnan={}", k_norm.isnan().any());

    let k_unit = k.f_div(&k_norm)?; // [M,C,KH,KW]

    // Numerator:
    //
    // num[b,m,y,t] = dot(x_patch[b,y,t], k_unit[m])
    let num = x.f_conv2d(
        &k_unit,
        Option::<&Tensor>::None,
        cfg.stride,
        cfg.padding,
        cfg.dilation,
        1,
    )?; // [B,M,H_out,W_out]

    // Local input norm. Same spatial sliding window as numerator.
    let ones = Tensor::ones(
        &[1, c, kh, kw],
        (x.kind(), x.device()),
    );

    let x_sq = x.f_mul(x)?;

    let local_norm_sq = if cfg.center_input_norm {
        let local_sum = x.f_conv2d(
            &ones,
            Option::<&Tensor>::None,
            cfg.stride,
            cfg.padding,
            cfg.dilation,
            1,
        )?; // [B,1,H_out,W_out]

        let local_sum_sq = x_sq.f_conv2d(
            &ones,
            Option::<&Tensor>::None,
            cfg.stride,
            cfg.padding,
            cfg.dilation,
            1,
        )?; // [B,1,H_out,W_out]
        debug!("cosine_conv2d_bank_logits: local_sum_sq={local_sum_sq:?} local_sum_sq.isnan={}", local_sum_sq.isnan().any());

        let area = (c * kh * kw) as f64;

        // sum((x - mean)^2) = sum(x^2) - sum(x)^2 / area
        let correction = local_sum.f_mul(&local_sum)?.f_div_scalar(area)?;

        local_sum_sq.f_sub(&correction)?
    } else {
        x_sq.f_conv2d(
            &ones,
            Option::<&Tensor>::None,
            cfg.stride,
            cfg.padding,
            cfg.dilation,
            1,
        )? // [B,1,H_out,W_out]
    };
    debug!("cosine_conv2d_bank_logits: local_norm_sq={local_norm_sq:?} local_norm_sq.isnan={}", local_norm_sq.isnan().any());

    let x_norm = local_norm_sq
        .f_clamp_min(cfg.eps)?
        .f_sqrt()?; // [B,1,H_out,W_out]

    debug!("cosine_conv2d_bank_logits: x_norm={x_norm:?} x_norm.isnan={}", x_norm.isnan().any());

    // Broadcast x_norm over motif dimension M.
    let score = num.f_div(&x_norm)?; // [B,M,H_out,W_out]

    debug!("cosine_conv2d_bank_logits: score={score:?} score.isnan={}", score.isnan().any());

    let alpha = alpha.f_view([1, m, 1, 1])?;
    let beta = beta.f_view([1, m, 1, 1])?;

    Ok(score
        .f_mul(&alpha)?
        .f_add(&beta)?)
}

#[derive(Debug)]
pub struct MyCosineConv2d {
    pub kernel: Tensor,

    /// Learnable logit scale
    pub alpha: Tensor,

    /// Learnable logit bias
    pub beta: Tensor,

    kh:i64,//frequencies
    kw:i64,//time

    cfg: CosineConv2dBankConfig,
}

impl MyCosineConv2d {
    pub fn new(
        vs: &Path,
        kh: i64,
        kw: i64,
        init_patches: Option<Arc<ThoTensor<Spectorgram, SynKey>>>, //Option<&Tensor>, // [N,KH,KW] or [N,1,KH,KW]
        a: f64,
        b: f64,
    ) -> Result<Self, ThoError> {
        debug!("MyCosineConv2d::new: init_patches={init_patches:?} kh={kh} kw={kw} a={a} b={b}");
        let kernel = match init_patches {
            None => {
                vs.f_var("kernel", &[1, 1, kh, kw], nn::Init::Const(0.0))?
            }

            Some(patches) => {
                let patches = patches.as_ref().as_ref();
                let patches = Self::ensure_motif_patches_4d(patches)?; // [N,1,KH,KW]
                let psz = patches.size();

                // if psz[2] != kh || psz[3] != kw {
                //     return Err(ThoError::new(format!("init patches must have spatial size [{kh},{kw}], got {psz:?}")));
                // }
                let kh_freq = psz[2];//frequencies
                let kw_time = psz[3];//time
                debug!("MyCosineConv2d::new: kh_freq={kh_freq:?} kw_time={kw_time:?}");

                // Simple mean prototype. You can replace this with your
                // existing Mean / Median / TrimmedMean / CenteredL2NormalizedMean logic.
                let proto = patches.f_mean_dim(&[0i64][..], false, Kind::Float)?;
                // proto: [1,KH,KW]

                let kernel_init = Self::finalize_prototype_2d(&proto, true, true)?;
                // kernel_init: [1,1,KH,KW]

                let mut kernel = vs.f_var("kernel", &[1, 1, kh_freq, kw_time], nn::Init::Const(0.0))?;

                tch::no_grad(|| -> Result<(), ThoError> {
                    kernel.f_copy_(&kernel_init)?;
                    Ok(())
                })?;

                debug!("MyCosineConv2d::new: kernel={kernel:?}  kernel.isnan().any={}", kernel.isnan().any());

                kernel
            }
        };

        let alpha = vs.f_var("alpha", &[], nn::Init::Const(a))?;
        let beta = vs.f_var("beta", &[], nn::Init::Const(b))?;

        Ok(Self {
            kernel,
            alpha,
            beta,
            kh,
            kw,
            cfg: CosineConv2dBankConfig {
                center_kernel_forward: false,
                center_input_norm: false,
                ..Default::default()
            },
        })
    }

    /// Optional constructor if you want Pearson-like matching from the beginning.
    ///
    /// Do not use this to combine already-trained old models unless they were trained
    /// with the same settings.
    pub fn with_forward_normalization(
        mut self,
        center_kernel_forward: bool,
        center_input_norm: bool,
    ) -> Self {
        self.cfg.center_kernel_forward = center_kernel_forward;
        self.cfg.center_input_norm = center_input_norm;
        self
    }

    /// Accept [B,H,W] or [B,1,H,W].
    pub fn ensure_4d_1ch(x: &Tensor) -> Result<Tensor, ThoError> {
        match x.size().as_slice() {
            [b, h, w] => Ok(x.f_view([*b, 1, *h, *w])?),
            [_, 1, _, _] => Ok(x.shallow_clone()),
            other => Err(ThoError::new(format!("input must have shape [B,H,W] or [B,1,H,W], got {other:?}"))),
        }
    }

    /// Accept [N,KH,KW] or [N,1,KH,KW].
    fn ensure_motif_patches_4d(x: &Tensor) -> Result<Tensor, ThoError> {
        match x.size().as_slice() {
            [n, h, w] => Ok(x.f_view([*n, 1, *h, *w])?),
            [_, 1, _, _] => Ok(x.shallow_clone()),
            other => Err(ThoError::new(format!("motif patches must have shape [N,KH,KW] or [N,1,KH,KW], got {:?}", other))),
        }
    }

    /// proto: [H,W], [1,H,W], or [1,1,H,W]
    /// returns: [1,1,H,W]
    fn finalize_prototype_2d(
        proto: &Tensor,
        zero_mean: bool,
        unit_l2: bool,
    ) -> Result<Tensor, ThoError> {
        let mut p = match proto.size().as_slice() {
            [h, w] => proto.f_view([1, *h, *w])?,
            [1, _, _] => proto.shallow_clone(),
            [1, 1, h, w] => proto.f_view([1, *h, *w])?,
            other => {
                return Err(ThoError::new(format!("expected proto shape [H,W], [1,H,W], or [1,1,H,W], got {:?}", other)));
            }
        };

        if zero_mean {
            let mean = p.f_mean(Kind::Float)?;
            p = p.f_sub(&mean)?;
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

    /// Returns full similarity/logit map:
    ///
    /// input:
    ///     [B,H,W] or [B,1,H,W]
    ///
    /// output:
    ///     [B,1,H_out,W_out]
    pub fn forward_map(&self, x: &Tensor) -> Result<Tensor, ThoError> {
        let x = Self::ensure_4d_1ch(x)?; // [B,1,H,W]

        cosine_conv2d_bank_logits(
            &x,
            &self.kernel, // [1,1,KH,KW]
            &self.alpha,  // scalar
            &self.beta,   // scalar
            self.cfg,
        )
    }

    /// For exact full-size matching.
    ///
    /// Requires output spatial size to be [1,1].
    ///
    /// Returns:
    ///     [B,1]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor, ThoError> {
        debug!("MyCosineConv2d::forward: x={x:?}");
        let y = self.forward_map(x)?; // [B,1,H_out,W_out]
        debug!("MyCosineConv2d::forward: y={y:?}");

        let sz = y.size();

        if sz[2] != 1 || sz[3] != 1 {
            return Err(ThoError::new(format!("MyCosineConv2d::forward expects full-size match with output [B,1,1,1], got {sz:?}. Use forward_map or forward_anywhere instead.")));
        }

        Ok(y.view((x.size()[0], -1))) // [B,1]
        // Ok(y.f_squeeze_dim(3)?.f_squeeze_dim(2)?) // [B,1]
    }

    /// Position-independent detection.
    ///
    /// Returns:
    ///     [B,1]
    ///
    /// Meaning:
    ///     strongest occurrence of this motif anywhere in the input.
    pub fn forward_anywhere(&self, x: &Tensor) -> Result<Tensor, ThoError> {
        let y = self.forward_map(x)?; // [B,1,H_out,W_out]

        let (y, _idx_t) = y.f_max_dim(3, false)?; // [B,1,H_out]
        let (y, _idx_h) = y.f_max_dim(2, false)?; // [B,1]

        Ok(y)
    }
}

impl<DI: Dims, DO: Dims> ThoModule<DI, DO> for MyCosineConv2d {

    /// For full-size matching where input H/W equals kernel H/W.
    /// Returns [B,1].
    fn tho_forward(&self, x_in: &ThoTensor<DI, SimpleId>, mut lengths: Option<&mut BatchLengths>, _train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        debug!("MyCosineConv2d::tho_forward: x_in={x_in:?}");

        let y = self.forward_anywhere(x_in.as_ref())?; // [B,1,1,1] if full match
        debug!("MyCosineConv2d::tho_forward: y={y:?}");

        Ok(y.into()) //.f_squeeze_dim(3)?.f_squeeze_dim(2)?.into())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use log::debug;
    use tch::Device::Cpu;
    use tch::kind::FLOAT_CPU;
    use tch::nn::VarStore;
    use tch::Tensor;
    use thoosa_shared::ThoError;
    use crate::my_tch::{InitDummy, MotifInitMode};
    use crate::nodes::cosine_2d::MyCosineConv2d;
    use crate::tho_tensor::ThoTensor;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_check_dims() -> Result<(), ThoError> {
        let slices = 3;
        let frequencies = 8000;
        let time = 4;
        let batch =  7;

        let vs = VarStore::new(Cpu);
        let cosine_2d = MyCosineConv2d::new(
            &vs.root(),
            0, 0,
            Some(Arc::new(ThoTensor::new("".into(), Tensor::rand(&[slices,frequencies,time], FLOAT_CPU))?)),
             1.0, 0.1,
        )?;//MotifInitMode::Mean(InitDummy{}),


        let y = cosine_2d.forward(&Tensor::rand(&[batch, 1, frequencies, time], FLOAT_CPU))?;

        debug!("test_check_dims: y={y:?}");

        Ok(())
    }

}