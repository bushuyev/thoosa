use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use log::debug;
use tch::nn::{BatchNormConfig, ConvConfig, LinearConfig, Module};
use tch::{Tensor, nn, no_grad};

use crate::dims::{AnyDims, Dims};
use crate::my_tch::{Fp, MyAvgPool1d, MyBatchNorm1d, MyClosureModule, MyConv, MyFuncModule, MyMaxPool1d, MyParallel, MySequential, ThoModule, my_linear, AdaptiveMaxPool1d, Split1};
use crate::train_nets::ThoModuleWithStore;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::SimpleId;
use crate::runs::Simple;
use crate::tho_tensor::ThoTensor;

//https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
pub fn clc_conv1d_out(l_in: u32, kernel: u32, c: ConvConfig) -> u32 {
    debug!("clc_conv1d_out: l_in={l_in} kernel={kernel} padding={} dilation={} stride={}", c.padding, c.dilation, c.stride);

    let r = (l_in + 2 * c.padding as u32 - c.dilation as u32 * (kernel - 1) - 1) / c.stride as u32 + 1;

    debug!("clc_conv1d_out: r={r}");

    r
}

pub fn clc_max_pool1d_out(l_in: u32, kernel: u32, padding: u32, dilation: u32, stride: u32) -> u32 {
    debug!("clc_max_pool1d_out: l_in={l_in} kernel={kernel} padding={padding} dilation={dilation} stride={stride}");
    let r= (l_in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;

    debug!("clc_max_pool1d_out: r={r}");

    r
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
        split: Arc<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>,
        join: Arc<dyn Fn(Vec<Tensor>) -> Tensor + Send + Sync>,
    },
    Split1 {
        node: Box<Node>,
        split: Arc<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>,
        join: Arc<dyn Fn(Vec<Tensor>) -> Tensor + Send + Sync>,
    },
    Conv1d {
        path:&'static str,
        kernel: u32,
        in_channels: u32,
        out_channels: u32,
        config: ConvConfig,
        init: Option<Arc<ThoTensor<AnyDims, SimpleId>>>
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
        in_feats: u32,
        out_feats: u32,
        config: Option<LinearConfig>,
        #[cfg(debug_assertions)]
        descr: String,
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
    AvgPool {
        kernel_size: u32,
        stride: u32,
        padding: u32,
    },
    Closure {
        cl: Arc<Box<dyn Fn(&Tensor) -> Tensor>>
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
    pub fn build_tch_module_2<DI: Dims + Send + 'static, DO: Dims + Send + 'static>(
        self,
        vs: &nn::Path,
    ) -> Result<ThoModuleWithStore<DI, DO>, ThoosaError> {
        Ok(ThoModuleWithStore::new("".into(), self.build_tch_module(vs)?))
    }

    pub fn build_tch_module<DI: Dims + Send + 'static, DO: Dims + Send + 'static>(
        self,
        vs: &nn::Path,
    ) -> Result<Box<dyn ThoModule<DI, DO>>, ThoosaError> {
        Ok(match self {
            Node::Seq { elts, split_input } => {
                let layers: Vec<Box<dyn ThoModule<AnyDims, AnyDims>>> =
                    elts.into_iter().enumerate().try_fold(
                        Vec::new(),
                        |mut kids: Vec<Box<dyn ThoModule<AnyDims, AnyDims>>>, (i, node)| {
                            kids.push(node.build_tch_module(&(vs / format!("seq_{i}")))?);

                            Ok::<Vec<Box<dyn ThoModule<AnyDims, AnyDims>>>, ThoosaError>(kids)
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
                            Ok::<Vec<Box<dyn ThoModule<DI, DO>>>, ThoosaError>(kids)
                        },
                    )?;

                Box::new(MyParallel::new(layers, split, join))
            }

            Node::Split1 { node, split, join } => {
                Box::new(Split1::new(
                    node.build_tch_module(&(vs / format!("Split1")))?,//TODO name
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
                Box::new(MyConv::new(conv))
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
            ))),

            Node::BatchNorm {
                num_features,
                config,
            } => Box::new(MyBatchNorm1d::new(vs / "batch_norm", num_features, config)),

            Node::Linear {
                in_feats,
                out_feats,
                config,
                #[cfg(debug_assertions)]
                descr,
            } => {
                assert!(in_feats > 0);
                assert!(out_feats > 0);

                #[cfg(debug_assertions)]
                debug!(
                    "Node::Linear: descr={descr} in_feats={:?}, out_feats={:?}",
                    in_feats, out_feats
                );

                Box::new(my_linear(
                    vs / "linear",
                    in_feats as i64,
                    out_feats as i64,
                    config.unwrap_or(LinearConfig::default()),
                    #[cfg(debug_assertions)]
                    descr,
                )?) //TODO rename layer1
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
        split: Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>,
        join: Box<dyn Fn(Vec<Tensor>) -> Tensor + Send + Sync>,
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
        split: Box<dyn Fn(&Tensor) -> Vec<Tensor> + Send + Sync>,
        join: Box<dyn Fn(Vec<Tensor>) -> Tensor + Send + Sync>,
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

    pub fn conv_1d(
        &mut self,
        path: &'static str,
        kernel: u32,
        in_channels: u32,
        out_channels: u32,
        config: ConvConfig,
        init: Option<Arc<ThoTensor<AnyDims, SimpleId>>>,
    ) -> &mut Self {
        self.elts.push(Node::Conv1d {
            path,
            kernel,
            in_channels,
            out_channels,
            config,
            init,
        });

        self.num_features = clc_conv1d_out(self.num_features, kernel, config) as u32;
        self.kernel = Some(kernel);

        self
    }

    pub fn conv_2d(
        &mut self,
        kernel: u32,
        in_channels: u32,
        out_channels: u32,
        config: ConvConfig,
    ) -> &mut Self {
        self.elts.push(Node::Conv2d {
            kernel,
            in_channels,
            out_channels,
            config,
        });

        self.num_features = clc_conv1d_out(self.num_features, kernel, config) as u32; //TODO 2d
        self.kernel = Some(kernel);

        self
    }

    pub fn max_pool(&mut self, kernel: u32, stride: u32, padding: u32) -> &mut Self {
        self.elts.push(Node::MaxPool {
            kernel_size: kernel,
            stride,
            padding,
        });

        self.num_features = clc_max_pool1d_out(self.num_features, kernel, padding, 1, stride); //TODO config?

        self
    }

    pub fn adaptive_max_pool(&mut self, output_size:u32, num_features:u32) -> &mut Self {
        self.num_features = num_features;

        self.elts.push(Node::AdaptiveMaxPool {
            output_size
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

    pub fn linear(&mut self, #[cfg(debug_assertions)] descr: String) -> &mut Self {
        self.elts.push(Node::Linear {
            in_feats: self.num_features,
            out_feats: self.num_features,
            config: None,
            #[cfg(debug_assertions)]
            descr,
        });
        self
    }

    pub fn linear_feats(
        &mut self,
        num_features: u32,
        out_feats: u32,
        #[cfg(debug_assertions)] descr: String,
    ) -> &mut Self {
        self.elts.push(Node::Linear {
            in_feats: num_features,
            out_feats,
            config: None,
            #[cfg(debug_assertions)]
            descr,
        });
        self.num_features = out_feats;
        self
    }

    pub fn linear_f(&mut self, f: f32, #[cfg(debug_assertions)] descr: String) -> &mut Self {
        let out_feats = (self.num_features as f32 * f) as u32;
        self.elts.push(Node::Linear {
            in_feats: self.num_features,
            out_feats,
            config: None,
            #[cfg(debug_assertions)]
            descr,
        });
        self.num_features = out_feats as u32;

        self
    }

    pub fn linear_s(
        &mut self,
        out_feats: u32,
        #[cfg(debug_assertions)] descr: String,
    ) -> &mut Self {
        assert!(out_feats > 0);
        self.elts.push(Node::Linear {
            in_feats: self.num_features,
            out_feats,
            config: None,
            #[cfg(debug_assertions)]
            descr,
        });
        self.num_features = out_feats;

        self
    }

    pub fn linear_s_with_conf(
        &mut self,
        out_feats: u32,
        config: LinearConfig,
        #[cfg(debug_assertions)] descr: &str,
    ) -> &mut Self {
        self.elts.push(Node::Linear {
            in_feats: self.num_features,
            out_feats,
            config: Some(config),
            #[cfg(debug_assertions)]
            descr: descr.into(),
        });
        self.num_features = out_feats;

        self
    }

    pub fn linear_p(&mut self, n: usize, p: &impl NthParams<Params = u32>) -> &mut Self {
        let out_feats = p.on_nth_layer(n);
        self.elts.push(Node::Linear {
            in_feats: self.num_features,
            out_feats,
            config: None,
            #[cfg(debug_assertions)]
            descr: "zzz".to_string(),
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
        cl: Arc<Box<dyn Fn(&Tensor) -> Tensor>>,
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
