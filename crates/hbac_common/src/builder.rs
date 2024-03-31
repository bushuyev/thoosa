use std::fmt;
use std::fmt::{Debug, Formatter};
use std::pin::Pin;
use log::debug;
use tch::nn::{ConvConfig, LinearConfig, Module};
use tch::{nn, Tensor};
use hbac_shared::HbacError;
use crate::my_tch::{Fp, my_linear, MyAvgPool1d, MyBatchNorm1d, MyClosureModule, MyFuncModule, MyMaxPool1d, MySequential};

//https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
pub fn clc_conv1d_out(l_in: i64, k: i64, c: ConvConfig) -> i64 {
    (l_in + 2 * c.padding - c.dilation * (k -1 ) -1)/c.stride +1
}


pub fn clc_maxpool1d_out(l_in: i64, kernel: i64, padding:i64, dilation:i64, stride:i64) -> i64 {
    (l_in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
}

pub fn clc_avgpool1d_out(l_in: i64, kernel: i64, padding:i64, stride:i64) -> i64 {
    (l_in + 2 * padding - kernel) / stride + 1
}

type TFN = dyn Fn(&Tensor) -> Tensor;

pub enum Node {
    Seq {elts: Vec<Node>},
    Conv_1d {kernel: i64, in_channels: i64, out_channels:i64, config: ConvConfig},
    Conv_2d {kernel: i64, in_channels: i64, out_channels:i64, config: ConvConfig},
    BatchNorm {num_features: i64},
    Linear { in_feats: i64, out_feats: i64, config:Option<LinearConfig>},
    Func { func:Fp},
    MaxPool { kernel_size:i64, stride:i64, padding:i64},
    AvgPool { kernel_size:i64, stride:i64, padding:i64},
    Flatten { from_dim:i64, to_dim:i64},
    Closure { cl:Box<dyn Fn(&Tensor)->Tensor + Send + Sync>}
}

impl Debug for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Node")
            .finish()
    }
}

trait NthParams {
    type Params;
    fn on_nth_layer(&self, i:usize)->Self::Params;
}

pub struct NthConst {
    n:i64
}

impl NthConst {
    pub fn new(n: i64) -> Self {
        Self { n }
    }
}


impl NthParams for NthConst {
    type Params = i64;

    fn on_nth_layer(&self, i: usize) -> i64 {
        self.n
    }
}


impl Node {

    pub fn build_tch_module(self, vs: &nn::Path) -> Result<Box<dyn Module>, HbacError>  {
        Ok(match self {
            Node::Seq { elts } => {

                let layers:Vec<Box<dyn Module>> = elts.into_iter().try_fold(Vec::new(), |mut kids:Vec<Box<dyn Module>>, node| {
                    kids.push(node.build_tch_module(vs)?);
                    Ok::<Vec<Box<dyn Module>>, HbacError>(kids)
                })?;

                Box::new(MySequential::new(layers))

            }
            Node::Conv_1d { kernel, in_channels, out_channels, config} => {
                Box::new(nn::conv1d(vs/ "cnn", in_channels, out_channels, kernel, config))
            }
            Node::Conv_2d { kernel, in_channels, out_channels, config} => {
                Box::new(nn::conv2d(vs/ "cnn", in_channels, out_channels, kernel, config))
            }
            Node::BatchNorm { .. } => {
                Box::new(MyBatchNorm1d::new())
            }
            Node::Linear {in_feats, out_feats, config} => {
                assert!(in_feats > 0);
                assert!(out_feats > 0);
                debug!("Node::Linear: in_feats={:?}, out_feats={:?}", in_feats, out_feats);
                Box::new(my_linear(vs/ "layer1", in_feats, out_feats, config.unwrap_or(LinearConfig::default()))?)
            }
            Node::MaxPool { kernel_size, stride, padding } => {
                Box::new(MyMaxPool1d::new(kernel_size, stride, padding))
            }
            Node::AvgPool { kernel_size, stride, padding } => {
                Box::new(MyAvgPool1d::new(kernel_size, stride, padding))
            }
            Node::Func { func } => {
                Box::new(MyFuncModule::new(func))
            }
            Node::Flatten { from_dim, to_dim } => {
                let from_dim = from_dim;
                let to_dim = to_dim;
                Box::new(MyClosureModule::new(Box::new(move |t:&Tensor|t.flatten(from_dim, to_dim))))
            }
            Node::Closure { cl: closure } => {
                Box::new(MyClosureModule::new(Box::new(closure)))
            }
        })
    }
}


#[derive(Debug)]
pub struct SeqBuilder {
    num_features:i64,
    kernel:Option<i64>,
    elts: Vec<Node>
}

impl SeqBuilder {
    pub fn new(num_features:i64) -> Self {
        Self {
            num_features,
            kernel: None,
            elts: vec![]
        }
    }

    pub fn split<F, G>(&mut self, seq_1:Node, seq_2:Node, split_f:F, glue_f:G) -> &mut Self
        where F: for<'a> FnMut(&Tensor) -> Tensor,
              G: for<'a> FnMut(&Tensor) -> Tensor {
        self
    }

    pub fn end_seq(&mut self) -> Node {
        let mut elts = Vec::new();

        std::mem::swap(&mut elts, &mut self.elts);

        Node::Seq {
            elts
        }
    }

    pub fn batch_norm(&mut self) -> &mut Self {
        self.elts.push(
            Node::BatchNorm {num_features:self.num_features}
        );
        self
    }

    pub fn conv_1d(&mut self, kernel: i64, in_channels: i64, out_channels: i64, config: ConvConfig) -> &mut Self {

        self.elts.push(
            Node::Conv_1d {kernel, in_channels, out_channels, config },
        );

        self.num_features = clc_conv1d_out(self.num_features, kernel, config);
        self.kernel = Some(kernel);

        self
    }

    pub fn conv_2d(&mut self, kernel: i64, in_channels: i64, out_channels: i64, config: ConvConfig) -> &mut Self {

        self.elts.push(
            Node::Conv_2d {kernel, in_channels, out_channels, config },
        );

        self.num_features = clc_conv1d_out(self.num_features, kernel, config);//TODO 2d
        self.kernel = Some(kernel);

        self
    }


    pub fn max_pool(&mut self, kernel:i64, stride:i64, padding:i64) -> &mut Self {
        self.elts.push(
            Node::MaxPool {
                kernel_size: kernel,
                stride,
                padding
            }
        );

        self.num_features = clc_maxpool1d_out(self.num_features, kernel, padding, 1, stride);//TODO config?

        self
    }

    pub fn avg_pool(&mut self, kernel:i64, stride:i64, padding:i64) -> &mut Self {
        self.elts.push(
            Node::AvgPool {
                kernel_size: kernel,
                stride,
                padding
            }
        );

        self.num_features = clc_avgpool1d_out(self.num_features, kernel, padding, stride);

        self
    }

    pub fn dropout(&mut self) -> &mut Self {
        self
    }

    pub fn linear(&mut self) -> &mut Self {
        self.elts.push(
            Node::Linear { in_feats:self.num_features, out_feats:self.num_features, config:None}
        );
        self
    }

    pub fn linear_f(&mut self, f:f32) -> &mut Self {
        let out_feats = (self.num_features as f32 * f) as i64;
        self.elts.push(
            Node::Linear { in_feats:self.num_features, out_feats, config:None}
        );
        self.num_features = out_feats;

        self
    }

    pub fn linear_s(&mut self, out_feats:i64) -> &mut Self {
        assert!(out_feats > 0);
        self.elts.push(
            Node::Linear { in_feats:self.num_features, out_feats, config:None}
        );
        self.num_features = out_feats;

        self
    }

    pub fn linear_s_with_conf(&mut self, out_feats:i64, config:LinearConfig) -> &mut Self {
        self.elts.push(
            Node::Linear { in_feats:self.num_features, out_feats, config:Some(config)}
        );
        self.num_features = out_feats;

        self
    }

    pub fn linear_p(&mut self, n:usize, p:&impl NthParams<Params=i64>) -> &mut Self {
        let out_feats = p.on_nth_layer(n);
        self.elts.push(
            Node::Linear { in_feats:self.num_features, out_feats, config:None}
        );
        self.num_features = out_feats;

        self
    }

    pub fn func(&mut self, func: Fp) -> &mut Self {
        self.elts.push(
            Node::Func { func }
        );
        self
    }

    pub fn closure(&mut self, cl: Box<dyn Fn(&Tensor)->Tensor + Send + Sync>, i: i64) -> &mut Self {
        self.elts.push(
            Node::Closure { cl }
        );
        self.num_features = i;
        self
    }

    pub fn n_blocks<F>(&mut self, mut block: F, n:usize) -> &mut Self
        where F: for <'a> FnMut(&'a mut SeqBuilder, usize) {

        for i in 0..n {
            block(self, i);
        }

        self
    }

    pub fn block<F>(&mut self, mut block:F) -> &mut Self
        where F: for <'a> FnMut(&'a mut SeqBuilder) {

        block(self);

        self
    }



    pub fn end(&mut self) {

    }
}
