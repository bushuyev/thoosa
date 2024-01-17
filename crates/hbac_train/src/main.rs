use tch::nn::{Module, OptimizerConfig};
use tch::{kind, nn, Device, Tensor, Kind};
use tch::data::TextData;
use tch::TchError::Torch;

fn my_module(p: nn::Path, dim: i64) -> impl nn::Module {
    let x1 = p.randn("x1", &[dim], 1.0, 5.0);
    let x2 = p.randn("x2", &[dim], 1.0, 5.0);
    nn::func(move |xs| xs * &x1 + xs.exp() * &x2)
}

fn net(vs: &nn::Path) -> impl Module {

    nn::seq()
        .add(
            nn::conv1d(
                vs / "layer1",
                 20,
                20,
                5,
                Default::default(),
            )
        )
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, 9996, 6, Default::default()))
}


fn main() {
    let device = if tch::utils::has_cuda() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    println!("device={:?}", device);
    const BATCH_SIZE: i64 = 10i64;


    let vs = nn::VarStore::new(device);
    let net = net(&vs.root());

    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();

    for idx in 1..2 {
        // Dummy mini-batches made of zeros.
        let (x_t, y_t) = load_batch(device, BATCH_SIZE);

        let pred_y = net.forward(&x_t.transpose(1, 2));
        println!("pred_y: size={:?}", pred_y.size());

        let pred_y_mean = pred_y.mean_dim(1, false, kind::Kind::Float);


        println!("pred_y_mean: size={:?}", pred_y_mean);

        let loss = (pred_y_mean - y_t).pow_tensor_scalar(2).sum(kind::Kind::Float);
        opt.backward_step(&loss);

        if idx % 1000 == 0 {
            println!("loss={:?}", loss);
        }
    }


    // gradient_descent();


    println!("Ok")

}

fn load_batch(device: Device, batch_size:i64) -> (Tensor, Tensor) {
    (
    tch::Tensor::randn([batch_size, 50 * 200, 20], (Kind::Float, device)), //batch, 20 EEG feats, 50 sec * 200 samples per second,
    tch::Tensor::randn([batch_size, 6], (Kind::Float, device)) //batch, seizure_vote,lpd_vote,gpd_vote, lrda_vote, grda_vote,other_vote, 50 sec * 200 samples per second,
    )
}