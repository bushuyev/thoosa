use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;
use log::debug;
use tch::nn::{BatchNormConfig, ConvConfig, LinearConfig};
use tch::{Tensor, nn, no_grad};

use crate::batches::BatchLengths;
use crate::dims::{AnyDims, Dims, MotifSlices, Spectorgram};
use crate::my_tch::{AdaptiveAvgPool1d, AdaptiveMaxPool1d, BinaryHead, BinaryHeadPool, Fp, MyAttnPool1d, MyAvgPool1d, MyBatchNorm1d, MyClosureModule, MyConv, MyCosineConv1d, MyDropout, MyFuncModule, MyMaxPool1d, MyParallel, MySequential, MySharpeningHead1d, SelfAttention, Split1, ThoModule, my_linear, MotifInitMode};
use thoosa_shared::{panic_to_err, ThoError};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::traits::SimpleId;
use crate::nodes::cosine_2d::MyCosineConv2d;
use crate::tho_tensor::ThoTensor;

//https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
pub fn clc_conv1d_out<P:Display + ?Sized>(l_in: u32, kernel: u32, c: ConvConfig, path: &P) -> Result<u32, ThoError> {
    debug!("clc_conv1d_out: path={path} l_in={l_in} kernel={kernel} padding={} dilation={} stride={}", c.padding, c.dilation, c.stride);

    if l_in == 0 { //convolution was first layer, num_features is not known
        return  Ok(0);
    }
    if kernel == 0 {// if kernel is calculated in runtime
        return Err(ThoError::new(format!("kernel={kernel} should be bigger than 0")))
    }
    if l_in + 2 * (c.padding as u32) < c.dilation as u32  * (kernel - 1) {
        return Err(ThoError::new(format!("clc_conv1d_out: l_in + 2 * padding < dilation * (kernel - 1) - 1 : l_in={l_in} kernel={} dilation={} padding={}", kernel, c.dilation, c.padding)))
    }

    if l_in + 2 * c.padding as u32 - c.dilation as u32 * (kernel - 1) < 1{

        return Err(ThoError::new(format!("l_in + 2 * c.padding as u32 - c.dilation as u32 * (kernel - 1) < 1 : l_in={l_in} kernel={} dilation={}", kernel, c.dilation)))
    }

    let r = (l_in + 2 * c.padding as u32 - c.dilation as u32 * (kernel - 1) - 1) / c.stride as u32 + 1;

    debug!("clc_conv1d_out: path={path}  r={r}");

    Ok(r)
}

pub fn clc_max_pool1d_out(l_in: u32, kernel: u32, padding: u32, dilation: u32, stride: u32) -> Result<u32, ThoError> {
    // throwing error should cause f32::MAX loss in optimizer
    if kernel < 2 {// if kernel is calculated in runtime
        return Err(ThoError::new(format!("kernel={kernel} should be bigger than 2")))
    }

    if l_in + 2 * padding < dilation * (kernel - 1) - 1 {
        return Err(ThoError::new(format!("clc_max_pool1d_out: l_in + 2 * padding < dilation * (kernel - 1) - 1 : l_in={l_in} kernel={kernel} dilation={dilation} padding={padding}")))
    }

    debug!("clc_max_pool1d_out: l_in={l_in} kernel={kernel} padding={padding} dilation={dilation} stride={stride}");
    let r= (l_in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;

    debug!("clc_max_pool1d_out: r={r}");

    Ok(r)
}

pub fn clc_avg_pool1d_out(l_in: u32, kernel: u32, padding: u32, stride: u32) -> u32 {
    let r= (l_in + 2 * padding - kernel) / stride + 1;

    debug!("clc_avg_pool1d_out: l_in={l_in} kernel={kernel} padding={padding} stride={stride} r={r}");

    r
}

#[derive(Clone)]
pub enum Node {
    Seq {
        elts: Vec<Node>,
        split_input: Option<(i64, i64)>,
    },
    Par {
        elts: Vec<Node>,
        split: Arc<dyn Fn(&Tensor) -> Result<Vec<Tensor>, ThoError> + Send + Sync>,
        join: Arc<dyn Fn(Vec<Tensor>) -> Result<Tensor, ThoError> + Send + Sync>,
    },
    Split1 {
        node: Box<Node>,
        split: Arc<dyn Fn(&Tensor) -> Result<Vec<Tensor>, ThoError> + Send + Sync>,
        join: Arc<dyn Fn(Vec<Tensor>) -> Result<Tensor, ThoError> + Send + Sync>,
    },
    Conv1d {
        path:String,
        kernel: u32,
        in_channels: u32,
        out_channels: u32,
        config: ConvConfig,
        init: Option<Arc<ThoTensor<AnyDims, SimpleId>>>
    },
    CosineConv1d {
        path:String,
        kernel: u32,
        a: f64,
        b: f64,
        init: Option<Arc<ThoTensor<MotifSlices, SynKey>>>,
        init_mode:MotifInitMode
    },
    CosineConv2d {
        path:String,
        kw_time:i64,
        kh_freq:i64,
        a: f64,
        b: f64,
        init: Option<Arc<ThoTensor<Spectorgram, SynKey>>>,
        init_mode:MotifInitMode
    },
    SharpeningHead1d {
        path: String,
        in_channels: u32,
        out_channels: u32,
        k_mu: u32,
        k_hat: u32,
        p: f32,
        temperature: f32,
    },
    BinaryHead {
        path: String,
        pool: BinaryHeadPool,
    },
    Conv2d {
        kernel: u32,
        in_channels: u32,
        out_channels: u32,
        config: ConvConfig,
    },
    BatchNorm {
        num_features: u32,
        config: BatchNormConfig,
    },
    Linear {
        path:String,
        in_feats: u32,
        out_feats: u32,
        config: Option<LinearConfig>
    },
    Func {
        func: Fp,
    },
    MaxPool {
        kernel_size: u32,
        stride: u32,
        padding: u32,
    },
    AdaptiveMaxPool {
        output_size: u32,
    },
    AdaptiveAvgPool1d {
        output_size: u32,
    },
    AttnPool1d {
        path:String,
        channels: u32,
        attn_hidden: u32
    },
    AvgPool {
        kernel_size: u32,
        stride: u32,
        padding: u32,
    },
    SelfAttention {
        path: String,
        d_model: u32,
        n_heads: u32,
        ff_mult: u32,
        dropout: f64,
    },
    Dropout {
        p: f64,
    },
    Closure {
        cl: Arc<Box<dyn Fn(&Tensor, Option<&BatchLengths>) -> Result<Tensor, ThoError>>>
    },
}

impl Debug for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Node").finish()
    }
}

pub trait NthParams {
    type Params;
    fn on_nth_layer(&self, i: usize) -> Self::Params;
}

pub struct NthConst {
    n: i64,
}

impl NthConst {
    pub fn new(n: i64) -> Self {
        Self { n }
    }
}

impl NthParams for NthConst {
    type Params = i64;

    fn on_nth_layer(&self, _i: usize) -> i64 {
        self.n
    }
}

impl Node {

    pub fn build_tch_module<DI: Dims + Send + 'static, DO: Dims + Send + 'static>(
        self,
        vs: &nn::Path,
    ) -> Result<Box<dyn ThoModule<DI, DO>>, ThoError> {
        Ok(match self {
            Node::Seq { elts, split_input } => {
                let layers: Vec<Box<dyn ThoModule<AnyDims, AnyDims>>> =
                    elts.into_iter().enumerate().try_fold(
                        Vec::new(),
                        |mut kids: Vec<Box<dyn ThoModule<AnyDims, AnyDims>>>, (i, node)| {
                            kids.push(node.build_tch_module(&(vs / format!("seq_{i}")))?);

                            Ok::<Vec<Box<dyn ThoModule<AnyDims, AnyDims>>>, ThoError>(kids)
                        },
                    )?;

                Box::new(MySequential::new(layers, split_input))
            }

            Node::Par { elts, split, join } => {
                let layers: Vec<Box<dyn ThoModule<DI, DO>>> =
                    elts.into_iter().enumerate().try_fold(
                        Vec::new(),
                        |mut kids: Vec<Box<dyn ThoModule<DI, DO>>>, (i, node)| {
                            kids.push(node.build_tch_module(&(vs / format!("split_{i}")))?);
                            Ok::<Vec<Box<dyn ThoModule<DI, DO>>>, ThoError>(kids)
                        },
                    )?;

                Box::new(MyParallel::new(layers, split, join))
            }

            Node::Split1 { node, split, join } => {
                Box::new(Split1::new(
                    node.build_tch_module(&(vs / "split_1"))?,//TODO name
                    split,
                    join
                ))
            }

            Node::Conv1d {
                path,
                kernel,
                in_channels,
                out_channels,
                config,
                init
            } => {
                let mut conv = panic_to_err!(Ok::<_, ThoError>(nn::conv1d(
                    vs / path,
                    in_channels as i64,
                    out_channels as i64,
                    kernel as i64,
                    config,
                )))?;
                if let Some(init) = init {
                    no_grad(|| {
                        conv.ws.copy_(&init.t.view((1, 1, kernel as i64)));
                    });

                }
                Box::new(MyConv::new(conv, kernel, config))
            },

            Node::Conv2d {
                kernel,
                in_channels,
                out_channels,
                config,
            } => Box::new(MyConv::new(nn::conv2d(
                vs / "cnn_2d",
                in_channels as i64,
                out_channels as i64,
                kernel as i64,
                config,
            ),
          0,
            ConvConfig::default())),

            Node::CosineConv1d {
                path,
                kernel,
                a, b,
                init,
                init_mode
            } => Box::new(MyCosineConv1d::new(
                &(vs / path),
                kernel as i64,
                init,
                init_mode,
                a, b
            )?),
            Node::CosineConv2d {
                path,
                kw_time, kh_freq,
                a, b,
                init,
                init_mode
            } => Box::new(MyCosineConv2d::new(
                &(vs / path),
                init,
                init_mode,
                a, b,
                kw_time,
                kh_freq
            )?),


            Node::SharpeningHead1d {
                path,
                in_channels,
                out_channels,
                k_mu,
                k_hat,
                p,
                temperature,
            } => Box::new(MySharpeningHead1d::new(
                &(vs / path),
                in_channels as i64,
                out_channels as i64,
                k_mu as i64,
                k_hat as i64,
                p as f64,
                temperature as f64,
            )?),

            Node::BinaryHead { path, pool } => Box::new(BinaryHead::new(&(vs / path), pool)),

            Node::BatchNorm {
                num_features,
                config,
            } => Box::new(MyBatchNorm1d::new(vs / "batch_norm", num_features, config)),

            Node::Linear {
                path,
                in_feats,
                out_feats,
                config,

            } => {

                debug!("Node::Linear: path={path} in_feats={:?}, out_feats={:?}",in_feats, out_feats);

                assert!(in_feats > 0, "in_feats > 0: in_feats={in_feats} path={path}");
                assert!(out_feats > 0, "out_feats > 0: in_feats={in_feats} path={path}");

                #[cfg(any(debug_assertions, tho_debug))]
                let decr = path.clone();

                Box::new(my_linear(
                    vs / path,
                    in_feats as i64,
                    out_feats as i64,
                    config.unwrap_or(LinearConfig::default()),
                    #[cfg(any(debug_assertions, tho_debug))]
                    decr,
                )?)
            }

            Node::MaxPool {
                kernel_size,
                stride,
                padding,
            } => Box::new(MyMaxPool1d::new(
                kernel_size as i64,
                stride as i64,
                padding as i64,
            )),

            Node::AdaptiveMaxPool { output_size } => Box::new(AdaptiveMaxPool1d::new(
                output_size
            )),

            Node::AdaptiveAvgPool1d { output_size } => Box::new(AdaptiveAvgPool1d::new(
                output_size
            )),

            Node::AvgPool {
                kernel_size,
                stride,
                padding,
            } => Box::new(MyAvgPool1d::new(
                kernel_size as i64,
                stride as i64,
                padding as i64,
            )),

            Node::SelfAttention { path, d_model, n_heads, ff_mult, dropout } => {
                Box::new(SelfAttention::new(
                    &(vs / path),
                    d_model as i64,
                    n_heads as i64,
                    ff_mult as i64,
                    dropout,
                )?)
            }

            Node::Dropout { p } => Box::new(MyDropout::new(p)),

            Node::Func { func } => Box::new(MyFuncModule::new(func)),

            Node::Closure { cl: closure } => {
                Box::new(MyClosureModule::new(closure))
            },

            Node::AttnPool1d { path, channels, attn_hidden } => {
                Box::new(MyAttnPool1d::new(&(vs / path), channels as i64, attn_hidden as i64))
            }
        })
    }
}

#[derive(Debug)]
pub struct SeqBuilder {
    num_features: u32,
    kernel: Option<u32>,
    elts: Vec<Node>,
    split_input: Option<(i64, i64)>,
}

impl SeqBuilder {
    pub fn new(num_features: u32, split_input_dim: Option<i64>) -> Result<Self, ThoError> {
        Ok(Self {
            num_features,
            kernel: None,
            elts: vec![],
            split_input: split_input_dim.map(|dim| (dim, num_features as i64)),
        })
    }

    pub fn split(
        &mut self,
        elts: Vec<Node>,
        split: Box<dyn Fn(&Tensor) -> Result<Vec<Tensor>, ThoError> + Send + Sync>,
        join: Box<dyn Fn(Vec<Tensor>) -> Result<Tensor, ThoError> + Send + Sync>,
    ) -> Result<&mut Self, ThoError> {
        self.elts.push(Node::Par {
            elts,
            split: Arc::from(split),
            join: Arc::from(join),
        });

        Ok(self)
    }

    pub fn split1(
        &mut self,
        node: Node,
        split: Box<dyn Fn(&Tensor) -> Result<Vec<Tensor>, ThoError> + Send + Sync>,
        join: Box<dyn Fn(Vec<Tensor>) -> Result<Tensor, ThoError> + Send + Sync>,
    ) -> Result<&mut Self, ThoError> {
        self.elts.push(Node::Split1 {
            node: Box::new(node),
            split: Arc::from(split),
            join: Arc::from(join),
        });

        Ok(self)
    }


    pub fn end_seq(&mut self) -> Result<Node, ThoError> {
        let mut elts = Vec::new();

        std::mem::swap(&mut elts, &mut self.elts);

        Ok(Node::Seq {
            elts,
            split_input: self.split_input,
        })
    }

    pub fn batch_norm(&mut self, config: BatchNormConfig, option: Option<u32>) -> Result<&mut Self, ThoError> {
        self.elts.push(Node::BatchNorm {
            num_features: option.unwrap_or(self.num_features),
            config,
        });
        Ok(self)
    }

    pub fn conv_1d<P:ToString>(
        &mut self,
        path: P,
        kernel: u32,
        in_channels: u32,
        out_channels: u32,
        config: ConvConfig,
        init: Option<Arc<ThoTensor<AnyDims, SimpleId>>>,
    ) -> Result<&mut Self, ThoError> {
        self.elts.push(Node::Conv1d {
            path:path.to_string(),
            kernel,
            in_channels,
            out_channels,
            config,
            init,
        });

        self.num_features = clc_conv1d_out(self.num_features, kernel, config, &path.to_string())?;
        self.kernel = Some(kernel);

        Ok(self)
    }

    pub fn cosine_conv_1d<P:ToString>(
        &mut self,
        path: P,
        kernel: u32,
        init: Option<Arc<ThoTensor<MotifSlices, SynKey>>>,
        a: f64,
        b: f64,
        init_mode: MotifInitMode
    ) -> Result<&mut Self, ThoError> {

        self.elts.push(Node::CosineConv1d {
            path:path.to_string(),
            kernel,
            a, b,
            init,
            init_mode
        });

        self.num_features = clc_conv1d_out(self.num_features, kernel, ConvConfig::default(),  "")?;
        self.kernel = Some(kernel);

        Ok(self)
    }

    pub fn cosine_conv_2d<P:ToString>(
        &mut self,
        path: P,
        kw_time: u32,
        kh_freq: u32,
        init: Option<Arc<ThoTensor<Spectorgram, SynKey>>>,
        a: f64,
        b: f64,
        init_mode: MotifInitMode
    ) -> Result<&mut Self, ThoError> {

        self.elts.push(Node::CosineConv2d {
            path:path.to_string(),
            kw_time: kw_time as i64,
            kh_freq: kh_freq as i64,
            a, b,
            init,
            init_mode
        });

        self.num_features = clc_conv1d_out(self.num_features, kw_time, ConvConfig::default(), "")?;
        self.kernel = Some(kw_time);

        Ok(self)
    }


    pub fn sharpening_head_1d<P:ToString>(&mut self, path: P, in_channels: u32, out_channels: u32, k_mu: u32, k_hat: u32, p: f32, temperature: f32) -> Result<&mut Self, ThoError> {
        self.elts.push(Node::SharpeningHead1d {
            path: path.to_string(),
            in_channels,
            out_channels,
            k_mu,
            k_hat,
            p,
            temperature,
        });
        Ok(self)
    }

    pub fn binary_head<P:ToString>(&mut self, path: P, pool: BinaryHeadPool) -> Result<&mut Self, ThoError> {
        self.elts.push(Node::BinaryHead {
            path: path.to_string(),
            pool,
        });
        Ok(self)
    }

    pub fn conv_2d(
        &mut self,
        _kernel: u32,
        _in_channels: u32,
        _out_channels: u32,
        _config: ConvConfig,
    ) -> Result<&mut Self, ThoError> {
        todo!()
    }

    pub fn max_pool(&mut self, kernel: u32, stride: u32, padding: u32) -> Result<&mut Self, ThoError> {
        self.elts.push(Node::MaxPool {
            kernel_size: kernel,
            stride,
            padding,
        });

        self.num_features = clc_max_pool1d_out(self.num_features, kernel, padding, 1, stride)?; //TODO config?

        Ok(self)
    }

    pub fn adaptive_max_pool(&mut self, output_size:u32, num_features:u32) -> Result<&mut Self, ThoError> {
        self.num_features = num_features;

        self.elts.push(Node::AdaptiveMaxPool {
            output_size
        });

        Ok(self)
    }

    pub fn adaptive_avg_pool(&mut self, output_size:u32, num_features:u32) -> Result<&mut Self, ThoError> {
        self.num_features = num_features;

        self.elts.push(Node::AdaptiveAvgPool1d {
            output_size
        });

        Ok(self)
    }

    pub fn attn_pool1d<P:ToString>(&mut self, path:P, channels:u32, attn_hidden:u32) -> Result<&mut Self, ThoError> {

        self.elts.push(Node::AttnPool1d {
            path: path.to_string(),
            channels,
            attn_hidden,
        });

        Ok(self)
    }

    pub fn self_attention<P: ToString>(&mut self, path: P, d_model: u32, n_heads: u32, ff_mult: u32, dropout: f64) -> Result<&mut Self, ThoError> {
        if d_model == 0 {
            return Err(ThoError::new(format!("self_attention, d_model > 0: path={}", path.to_string())));
        }
        if n_heads == 0 {
            return Err(ThoError::new(format!("self_attention, n_heads > 0: path={}", path.to_string())));
        }
        if ff_mult == 0 {
            return Err(ThoError::new(format!("self_attention, ff_mult > 0: path={}", path.to_string())));
        }

        self.elts.push(Node::SelfAttention {
            path: path.to_string(),
            d_model,
            n_heads,
            ff_mult,
            dropout,
        });
        self.num_features = d_model;
        Ok(self)
    }

    pub fn avg_pool(&mut self, kernel: u32, stride: u32, padding: u32) -> Result<&mut Self, ThoError> {
        self.elts.push(Node::AvgPool {
            kernel_size: kernel,
            stride,
            padding,
        });

        self.num_features = clc_avg_pool1d_out(self.num_features, kernel, padding, stride);

        Ok(self)
    }

    pub fn dropout(&mut self, p: f64) -> Result<&mut Self, ThoError> {
        self.elts.push(Node::Dropout { p });
        Ok(self)
    }

    pub fn linear<P:ToString>(&mut self, path: P) -> Result<&mut Self, ThoError> {
        if self.num_features == 0 {
            return Err(ThoError::new(format!("linear, self.num_features > 0: path={}", path.to_string())));
        }

        self.elts.push(Node::Linear {
            path: path.to_string(),
            in_feats: self.num_features,
            out_feats: self.num_features,
            config: None,
        });
        Ok(self)
    }

    pub fn linear_feats<P:ToString>(
        &mut self,
        path: P,
        num_features: u32,
        out_feats: u32,
    ) -> Result<&mut Self, ThoError> {
        if num_features == 0 {
            return Err(ThoError::new(format!("linear_feats, num_features > 0: path={}", path.to_string())));
        }

        self.elts.push(Node::Linear {
            path: path.to_string(),
            in_feats: num_features,
            out_feats,
            config: None,
        });
        self.num_features = out_feats;
        Ok(self)
    }

    pub fn linear_f<P:ToString>(&mut self, path: P, f: f32) -> Result<&mut Self, ThoError> {
        if self.num_features == 0 {
            return Err(ThoError::new(format!("linear_f, self.num_features > 0: path={}", path.to_string())));
        }

        let out_feats = (self.num_features as f32 * f) as u32;
        self.elts.push(Node::Linear {
            path: path.to_string(),
            in_feats: self.num_features,
            out_feats,
            config: None,
        });
        self.num_features = out_feats;

        Ok(self)
    }

    pub fn linear_s<P:ToString>(
        &mut self,
        out_feats: u32,
        path: P,
    ) -> Result<&mut Self, ThoError> {
        if self.num_features == 0 {
            return Err(ThoError::new(format!("linear_f, self.num_features > 0: path={}", path.to_string())));
        }
        if out_feats == 0 {
            return Err(ThoError::new(format!("linear_f, out_feats > 0: path={}", path.to_string())));
        }

        self.elts.push(Node::Linear {
            path: path.to_string(),
            in_feats: self.num_features,
            out_feats,
            config: None,
        });
        self.num_features = out_feats;

        Ok(self)
    }

    pub fn linear_s_with_conf<P:ToString>(
        &mut self,
        path: P,
        out_feats: u32,
        config: LinearConfig,
    ) -> Result<&mut Self, ThoError> {
        if self.num_features == 0 {
            return Err(ThoError::new(format!("linear_s_with_conf, self.num_features > 0: path={}", path.to_string())));
        }
        if out_feats == 0 {
            return Err(ThoError::new(format!("linear_s_with_conf, out_feats > 0: path={}", path.to_string())));
        }

        self.elts.push(Node::Linear {
            path: path.to_string(),
            in_feats: self.num_features,
            out_feats,
            config: Some(config),

        });
        self.num_features = out_feats;

        Ok(self)
    }

    pub fn linear_p<P:ToString>(&mut self, path: P, n: usize, p: &impl NthParams<Params = u32>) -> Result<&mut Self, ThoError> {

        let out_feats = p.on_nth_layer(n);

        if self.num_features == 0 {
            return Err(ThoError::new(format!("linear_p, self.num_features > 0: path={}", path.to_string())));
        }
        if out_feats == 0 {
            return Err(ThoError::new(format!("linear_p, out_feats > 0: path={}", path.to_string())));
        }

        self.elts.push(Node::Linear {
            path: path.to_string(),
            in_feats: self.num_features,
            out_feats,
            config: None,

        });
        self.num_features = out_feats;

        Ok(self)
    }

    pub fn func(&mut self, func: Fp) -> Result<&mut Self, ThoError> {
        self.elts.push(Node::Func { func });
        Ok(self)
    }

    pub fn closure(
        &mut self,
        cl: Arc<Box<dyn Fn(&Tensor, Option<&BatchLengths>) -> Result<Tensor, ThoError>>>,
        num_features: Option<u32>,
    ) -> Result<&mut Self, ThoError> {
        self.elts.push(Node::Closure { cl });
        if let Some(num_features) = num_features {
            self.num_features = num_features;
        }

        Ok(self)
    }

    pub fn n_blocks<F>(&mut self, mut block: F, n: usize) -> Result<&mut Self, ThoError>
    where
        F: for<'a> FnMut(&'a mut SeqBuilder, usize) -> Result<(), ThoError>,
    {
        for i in 0..n {
            block(self, i)?;
        }

        Ok(self)
    }

    pub fn block<F>(&mut self, mut block: F) -> Result<&mut Self, ThoError>
    where
        F: for<'a> FnMut(&'a mut SeqBuilder) -> Result<(), ThoError>,
    {
        block(self)?;

        Ok(self)
    }

    pub fn end(&mut self) -> Result<(), ThoError> {
        Ok(())
    }
}
