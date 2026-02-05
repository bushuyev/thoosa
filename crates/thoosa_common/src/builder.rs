use std::fmt::{format, Debug, Formatter};
use std::sync::Arc;
use log::debug;
use tch::nn::{BatchNormConfig, ConvConfig, LinearConfig};
use tch::{Tensor, nn, no_grad};

use crate::dims::{AnyDims, Dims};
use crate::my_tch::{AdaptiveMaxPool1d, Fp, MyAttnPool1d, MyAvgPool1d, MyBatchNorm1d, MyClosureModule, MyConv, MyCosineConv1d, MyFuncModule, MyMaxPool1d, MyParallel, MySequential, MySharpeningHead1d, Split1, ThoModule, my_linear};
use thoosa_shared::ThoError;
use thoosa_shared::traits::SimpleId;
use crate::tho_tensor::ThoTensor;

//https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
pub fn clc_conv1d_out(l_in: u32, kernel: u32, c: ConvConfig) -> Result<u32, ThoError> {
    if l_in == 0 { //convolution was first layer, num_features is not known
        return  Ok(0);
    }
    if kernel < 2 {// if kernel is calculated in runtime
        return Err(ThoError::new(format!("kernel={kernel} should be bigger than 2")))
    }
    if l_in + 2 * (c.padding as u32) < c.dilation as u32  * (kernel - 1) {
        return Err(ThoError::new(format!("l_in + 2 * padding < dilation * (kernel - 1) - 1 : l_in={l_in} kernel={} dilation={} should be bigger than 2", kernel, c.dilation)))
    }

    if l_in + 2 * c.padding as u32 - c.dilation as u32 * (kernel - 1) < 1{

        return Err(ThoError::new(format!("l_in + 2 * c.padding as u32 - c.dilation as u32 * (kernel - 1) < 1 : l_in={l_in} kernel={} dilation={} should be bigger than 1", kernel, c.dilation)))
    }

    debug!("clc_conv1d_out: l_in={l_in} kernel={kernel} padding={} dilation={} stride={}", c.padding, c.dilation, c.stride);

    let r = (l_in + 2 * c.padding as u32 - c.dilation as u32 * (kernel - 1) - 1) / c.stride as u32 + 1;

    debug!("clc_conv1d_out: r={r}");

    Ok(r)
}

pub fn clc_max_pool1d_out(l_in: u32, kernel: u32, padding: u32, dilation: u32, stride: u32) -> Result<u32, ThoError> {
    // throwing error should cause f32::MAX loss in optimizer
    if kernel < 2 {// if kernel is calculated in runtime
        return Err(ThoError::new(format!("kernel={kernel} should be bigger than 2")))
    }
    if l_in + 2 * padding < dilation * (kernel - 1) - 1 {
        return Err(ThoError::new(format!("l_in + 2 * padding < dilation * (kernel - 1) - 1 : l_in={l_in} kernel={kernel} dilation={dilation} should be bigger than 2")))
    }

    debug!("clc_max_pool1d_out: l_in={l_in} kernel={kernel} padding={padding} dilation={dilation} stride={stride}");
    let r= (l_in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;

    debug!("clc_max_pool1d_out: r={r}");

    Ok(r)
}

pub fn clc_avg_pool1d_out(l_in: u32, kernel: u32, padding: u32, stride: u32) -> u32 {
    (l_in + 2 * padding - kernel) / stride + 1
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
        in_channels: u32,
        out_channels: u32,
        config: ConvConfig,
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
        config: Option<LinearConfig>,
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
    Closure {
        cl: Arc<Box<dyn Fn(&Tensor, Option<&[i64]>) -> Result<Tensor, ThoError>>>
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
                let mut conv = nn::conv1d(
                    vs / path,
                    in_channels as i64,
                    out_channels as i64,
                    kernel as i64,
                    config,
                );
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
                in_channels,
                out_channels,
                config,
            } => Box::new(MyCosineConv1d::new(
                &(vs / path),
                in_channels as i64,
                kernel as i64,
                config,
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

                #[cfg(debug_assertions)]
                let decr = path.clone();

                Box::new(my_linear(
                    vs / path,
                    in_feats as i64,
                    out_feats as i64,
                    config.unwrap_or(LinearConfig::default()),
                    #[cfg(debug_assertions)]
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

            Node::AvgPool {
                kernel_size,
                stride,
                padding,
            } => Box::new(MyAvgPool1d::new(
                kernel_size as i64,
                stride as i64,
                padding as i64,
            )),

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
    pub fn new(num_features: u32, split_input_dim: Option<i64>) -> Self {
        Self {
            num_features,
            kernel: None,
            elts: vec![],
            split_input: split_input_dim.map(|dim| (dim, num_features as i64)),
        }
    }

    pub fn split(
        &mut self,
        elts: Vec<Node>,
        split: Box<dyn Fn(&Tensor) -> Result<Vec<Tensor>, ThoError> + Send + Sync>,
        join: Box<dyn Fn(Vec<Tensor>) -> Result<Tensor, ThoError> + Send + Sync>,
    ) -> &mut Self {
        self.elts.push(Node::Par {
            elts,
            split: Arc::from(split),
            join: Arc::from(join),
        });

        self
    }

    pub fn split1(
        &mut self,
        node: Node,
        split: Box<dyn Fn(&Tensor) -> Result<Vec<Tensor>, ThoError> + Send + Sync>,
        join: Box<dyn Fn(Vec<Tensor>) -> Result<Tensor, ThoError> + Send + Sync>,
    ) -> &mut Self {
        self.elts.push(Node::Split1 {
            node: Box::new(node),
            split: Arc::from(split),
            join: Arc::from(join),
        });

        self
    }


    pub fn end_seq(&mut self) -> Node {
        let mut elts = Vec::new();

        std::mem::swap(&mut elts, &mut self.elts);

        Node::Seq {
            elts,
            split_input: self.split_input,
        }
    }

    pub fn batch_norm(&mut self, config: BatchNormConfig) -> &mut Self {
        self.elts.push(Node::BatchNorm {
            num_features: self.num_features,
            config,
        });
        self
    }

    pub fn conv_1d<P:ToString>(
        &mut self,
        path: P,
        kernel: u32,
        in_channels: u32,
        out_channels: u32,
        config: ConvConfig,
        init: Option<Arc<ThoTensor<AnyDims, SimpleId>>>,
    ) -> &mut Self {
        self.elts.push(Node::Conv1d {
            path:path.to_string(),
            kernel,
            in_channels,
            out_channels,
            config,
            init,
        });

        self.num_features = clc_conv1d_out(self.num_features, kernel, config).unwrap();
        self.kernel = Some(kernel);

        self
    }

    pub fn cosine_conv_1d<P:ToString>(
        &mut self,
        path: P,
        kernel: u32,
        in_channels: u32,
        out_channels: u32,
        config: ConvConfig,
    ) -> &mut Self {
        self.elts.push(Node::CosineConv1d {
            path:path.to_string(),
            kernel,
            in_channels,
            out_channels,
            config,
        });

        self.num_features = clc_conv1d_out(self.num_features, kernel, config).unwrap();
        self.kernel = Some(kernel);

        self
    }

    pub fn sharpening_head_1d<P:ToString>(&mut self, path: P, in_channels: u32, out_channels: u32, k_mu: u32, k_hat: u32, p: f32, temperature: f32) -> &mut Self {
        self.elts.push(Node::SharpeningHead1d {
            path: path.to_string(),
            in_channels,
            out_channels,
            k_mu,
            k_hat,
            p,
            temperature,
        });
        self
    }

    pub fn conv_2d(
        &mut self,
        kernel: u32,
        in_channels: u32,
        out_channels: u32,
        config: ConvConfig,
    ) -> &mut Self {
        todo!()
    }

    pub fn max_pool(&mut self, kernel: u32, stride: u32, padding: u32) -> &mut Self {
        self.elts.push(Node::MaxPool {
            kernel_size: kernel,
            stride,
            padding,
        });

        self.num_features = clc_max_pool1d_out(self.num_features, kernel, padding, 1, stride).unwrap(); //TODO config?

        self
    }

    pub fn adaptive_max_pool(&mut self, output_size:u32, num_features:u32) -> &mut Self {
        self.num_features = num_features;

        self.elts.push(Node::AdaptiveMaxPool {
            output_size
        });

        self
    }

    pub fn attn_pool1d<P:ToString>(&mut self, path:P, channels:u32, attn_hidden:u32) -> &mut Self {

        self.elts.push(Node::AttnPool1d {
            path: path.to_string(),
            channels,
            attn_hidden,
        });

        self
    }



    pub fn avg_pool(&mut self, kernel: u32, stride: u32, padding: u32) -> &mut Self {
        self.elts.push(Node::AvgPool {
            kernel_size: kernel,
            stride,
            padding,
        });

        self.num_features = clc_avg_pool1d_out(self.num_features, kernel, padding, stride);

        self
    }

    pub fn dropout(&mut self) -> &mut Self {
        self
    }

    pub fn linear<P:ToString>(&mut self, path: P, #[cfg(debug_assertions)] descr: String) -> &mut Self {
        assert!(self.num_features > 0, "linear, self.num_features > 0: path={}", path.to_string());

        self.elts.push(Node::Linear {
            path: path.to_string(),
            in_feats: self.num_features,
            out_feats: self.num_features,
            config: None,
        });
        self
    }

    pub fn linear_feats<P:ToString>(
        &mut self,
        path: P,
        num_features: u32,
        out_feats: u32,
        #[cfg(debug_assertions)] descr: String,
    ) -> &mut Self {
        assert!(num_features > 0, "linear_feats, num_features > 0: path={}", path.to_string());

        self.elts.push(Node::Linear {
            path: path.to_string(),
            in_feats: num_features,
            out_feats,
            config: None,
        });
        self.num_features = out_feats;
        self
    }

    pub fn linear_f<P:ToString>(&mut self, path: P, f: f32, #[cfg(debug_assertions)] descr: String) -> &mut Self {
        assert!(self.num_features > 0, "linear_f, self.num_features > 0: path={}", path.to_string());

        let out_feats = (self.num_features as f32 * f) as u32;
        self.elts.push(Node::Linear {
            path: path.to_string(),
            in_feats: self.num_features,
            out_feats,
            config: None,
        });
        self.num_features = out_feats as u32;

        self
    }

    pub fn linear_s<P:ToString>(
        &mut self,
        out_feats: u32,
        #[cfg(debug_assertions)] descr: String,
        path: P,
    ) -> &mut Self {
        assert!(self.num_features > 0, "linear_f, self.num_features > 0: path={}", path.to_string());
        assert!(out_feats > 0, "linear_f, out_feats > 0: path={}", path.to_string());

        self.elts.push(Node::Linear {
            path: path.to_string(),
            in_feats: self.num_features,
            out_feats,
            config: None,
        });
        self.num_features = out_feats;

        self
    }

    pub fn linear_s_with_conf<P:ToString>(
        &mut self,
        path: P,
        out_feats: u32,
        config: LinearConfig,
        #[cfg(debug_assertions)] descr: &str,
    ) -> &mut Self {
        assert!(self.num_features > 0, "linear_s_with_conf, self.num_features > 0: path={}", path.to_string());
        assert!(out_feats > 0, "linear_s_with_conf, out_feats > 0: path={}", path.to_string());

        self.elts.push(Node::Linear {
            path: path.to_string(),
            in_feats: self.num_features,
            out_feats,
            config: Some(config),

        });
        self.num_features = out_feats;

        self
    }

    pub fn linear_p<P:ToString>(&mut self, path: P, n: usize, p: &impl NthParams<Params = u32>) -> &mut Self {

        let out_feats = p.on_nth_layer(n);

        assert!(self.num_features > 0, "linear_p, self.num_features > 0: path={}", path.to_string());
        assert!(out_feats > 0, "linear_p, out_feats > 0: path={}", path.to_string());

        self.elts.push(Node::Linear {
            path: path.to_string(),
            in_feats: self.num_features,
            out_feats,
            config: None,

        });
        self.num_features = out_feats;

        self
    }

    pub fn func(&mut self, func: Fp) -> &mut Self {
        self.elts.push(Node::Func { func });
        self
    }

    pub fn closure(
        &mut self,
        cl: Arc<Box<dyn Fn(&Tensor, Option<&[i64]>) -> Result<Tensor, ThoError>>>,
        num_features: Option<u32>,
    ) -> &mut Self {
        self.elts.push(Node::Closure { cl });
        if let Some(num_features) = num_features {
            self.num_features = num_features;
        }

        self
    }

    pub fn n_blocks<F>(&mut self, mut block: F, n: usize) -> &mut Self
    where
        F: for<'a> FnMut(&'a mut SeqBuilder, usize),
    {
        for i in 0..n {
            block(self, i);
        }

        self
    }

    pub fn block<F>(&mut self, mut block: F) -> &mut Self
    where
        F: for<'a> FnMut(&'a mut SeqBuilder),
    {
        block(self);

        self
    }

    pub fn end(&mut self) {}
}
