use std::borrow::Borrow;
use std::fmt::{Debug, Formatter};
use tch::nn::{Init, Linear, LinearConfig, Module, Path, Sequential};
use tch::Tensor;
use torch_sys::IntList;
use crate::data::GenError;

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
            let xs = self.layers[0].forward(xs);
            self.layers.iter().skip(1).fold(xs, |xs, layer| layer.forward(&xs))
        }
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



pub fn my_linear<'a, T: Borrow<Path<'a>>>(
    vs: T,
    in_dim: i64,
    out_dim: i64,
    c: LinearConfig,
) -> Result<Linear, GenError> {
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

    Ok(Linear { ws: vs.f_var("weight", &[out_dim, in_dim], c.ws_init)?, bs })
}
