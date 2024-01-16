use tch::nn::{Module, OptimizerConfig};
use tch::{kind, nn, Device, Tensor};

fn my_module(p: nn::Path, dim: i64) -> impl nn::Module {
    let x1 = p.randn("x1", &[dim], 1.0, 5.0);
    let x2 = p.randn("x2", &[dim], 1.0, 5.0);
    nn::func(move |xs| xs * &x1 + xs.exp() * &x2)
}

fn gradient_descent() {
    let vs = nn::VarStore::new(Device::Cuda(0));
    let my_module = my_module(vs.root(), 7);
    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();
    for idx in 1..500000 {
        // Dummy mini-batches made of zeros.
        let xs = Tensor::zeros(&[7], kind::FLOAT_CUDA);
        let ys = Tensor::zeros(&[7], kind::FLOAT_CUDA);
        let loss = (my_module.forward(&xs) - ys).pow_tensor_scalar(2).sum(kind::Kind::Float);
        opt.backward_step(&loss);

        if idx % 1000 == 0 {
            println!("loss={:?}", loss);
        }
    }
}

fn main() {
    gradient_descent();


    println!("Ok")

}