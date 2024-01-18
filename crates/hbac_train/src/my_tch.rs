use tch::nn::{Module, Sequential};
use tch::Tensor;

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
