mod my_tch;

use tch::nn::{ConvConfig, LinearConfig, Module, OptimizerConfig, Sequential};
use tch::{kind, nn, Device, Tensor, Kind, Reduction, no_grad};
use tch::data::TextData;
use tch::TchError::Torch;
use crate::my_tch::MySequential;
use plotters::prelude::*;

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


fn train_x2(device:&Device) {
    println!("test_x2device={:?}", device);
    const BATCH_SIZE: i64 = 10i64;


    let vs = nn::VarStore::new(device.clone());

    let nth_params = NthConst::new(100);

    let lin_block = |b: &mut SeqBuilder, i| b
        .linear_p(i, &nth_params)
        .func(|t|t.elu())
        .end();

    let seq = SeqBuilder::new(1000)
        .n_blocks(lin_block, 3)
        .linear_s(1000)
        .end_seq();
    let model = seq.build_tch_module(&vs.root());

    let mut opt = nn::AdamW::default().build(&vs, 1e-2).unwrap();

    let x_t = Tensor::from_slice(&(0..1000).map(|v|v as f32).collect::<Vec<f32>>()[..]).to_device(*device);
    let y_t = x_t.pow_tensor_scalar(2.);

    println!("x_tx_t.device={:?}", x_t.device());
    println!("y_t.device={:?}", y_t.device());

    // x_t.print();
    // y_t.print();

    for idx in 1..10_000 {
        // let (x_t, y_t) = load_batch(device, BATCH_SIZE);

        let pred_y = model.forward(&x_t);

        // println!("pred_y.device={:?}", pred_y.device());

        let loss = pred_y.huber_loss(&y_t, Reduction::Mean, 1.0).sum(kind::Kind::Float);

        opt.backward_step(&loss);

        if idx % 100 == 0 {
            print!(".");
        }
    }



    no_grad(|| {
        let y_pred = model.forward(&x_t);

        let pred = x_t.iter::<f64>().unwrap().zip(
            y_pred.iter::<f64>().unwrap()
        ).map(|(x, y)|(x as f32, y as f32)).collect::<Vec<(f32, f32)>>();

        let test = x_t.iter::<f64>().unwrap().zip(
            y_t.iter::<f64>().unwrap()
        ).map(|(x, y)|(x as f32, y as f32)).collect::<Vec<(f32, f32)>>();

        // println!("test={:?}", test);
        // println!("pred={:?}", pred);

        plot(
            vec![
                ("pred".into(), RED, pred),
                ("test".into(), GREEN, test),
            ]
        ).unwrap();
    })

}

//
// plot(
//     vec![
//         ("x2".into(), RED,  (-50..=50).map(|x| x as f32 / 50.0).map(|x| (x, x * x)).collect::<Vec<(f32, f32)>>()),
//         ("x3".into(), GREEN, (-50..=50).map(|x| x as f32 / 50.0).map(|x| (x, x * x * x)).collect::<Vec<(f32, f32)>>())
//     ]
// ).unwrap();

fn train_dense(device:&Device){

    println!("device={:?}", device);
    const BATCH_SIZE: i64 = 10i64;


    let vs = nn::VarStore::new(device.clone());

    // let nth_params = NthConst::new(1000);
    //
    // let lin_block = |b: &mut SeqBuilder, i| b
    //     .linear_p(i, &nth_params)
    //     .end();
    //
    // let seq = SeqBuilder::new(10000)
    //     .n_blocks(lin_block, 3)
    //     .linear_s(6)
    //     .end_seq();
    let net = configure_dense(10, 4, 4, 5).build_tch_module(&vs.root());

    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();

    for idx in 1..2 {
        let (x_t, y_t) = load_batch(device, BATCH_SIZE);

        let pred_y = net.forward(&x_t.transpose(1, 2));
        println!("pred_y: size={:?}", pred_y.size());

        let pred_y_mean = pred_y.mean_dim(1, false, kind::Kind::Float);


        println!("pred_y_mean: size={:?}", pred_y_mean);

        let loss = (pred_y_mean - y_t).pow_tensor_scalar(2).sum(kind::Kind::Float);
        opt.backward_step(&loss);
        //
        // if idx % 1000 == 0 {
        //     println!("loss={:?}", loss);
        // }
    }
}

fn plot(name_x_y_vec: Vec<(String, RGBColor, Vec<(f32, f32)>)>) -> Result<(), Box<dyn std::error::Error>> {
    
    let (min_x, max_x, min_y, max_y) = name_x_y_vec.iter().flat_map(|name_x_y_vec| name_x_y_vec.2.iter()).fold((f32::MAX, f32::MIN, f32::MAX, f32::MIN, ), |mut max_min, (x, y)| {
        if max_min.0 > *x {
            max_min.0 = *x
        }
        if max_min.1 < *x {
            max_min.1 = *x
        }

        if max_min.2 > *y {
            max_min.2 = *y
        }
        if max_min.3 < *y {
            max_min.3 = *y
        }

        max_min
    });


    let root = BitMapBackend::new("0.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("pred/test", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart.configure_mesh().draw()?;

    for (name, color, x_y) in name_x_y_vec {
        chart
            .draw_series(LineSeries::new(
                x_y,
                &color,
            ))?
            .label(name)
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}


fn main() {
    let device = if tch::utils::has_cuda() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    // test_1(&device);
    train_x2(&device);

    // gradient_descent();


    println!("Ok")

}

fn load_batch(device: &Device, batch_size:i64) -> (Tensor, Tensor) {
    (
    tch::Tensor::randn([batch_size, 50 * 200, 20], (Kind::Float, device.clone())), //batch, 20 EEG feats, 50 sec * 200 samples per second,
    tch::Tensor::randn([batch_size, 6], (Kind::Float, device.clone())) //batch, seizure_vote,lpd_vote,gpd_vote, lrda_vote, grda_vote,other_vote, 50 sec * 200 samples per second,
    )
}


#[derive(Debug)]
pub enum Node {
    Seq {elts: Vec<Node>},
    Conv {kernel: i64, in_channels: i64, out_channels:i64, config: ConvConfig},
    BatchNorm {num_features: i64},
    Linear { in_feats: i64, out_feats: i64},
    MaxPool { kernel_size:i64, stride:i64, padding:i64}
}

trait NthParams {
    type Params;
    fn on_nth_layer(&self, i:usize)->Self::Params;
}

struct NthConst {
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

    pub fn build_tch_module(&self, vs: &nn::Path) -> Box<dyn Module>  {
        match self {
            Node::Seq { elts } => {

                let layers:Vec<Box<dyn Module>> = elts.iter().fold(Vec::new(), |mut kids, node| {
                    kids.push(node.build_tch_module(vs));
                    kids
                });

                Box::new(MySequential::new(layers))

            }
            Node::Conv { kernel, in_channels, out_channels, config} => {
                Box::new(nn::conv1d(vs/ "cnn", *in_channels, *out_channels, *kernel, *config))
            }
            Node::BatchNorm { .. } => {
                todo!()
            }
            Node::Linear {in_feats, out_feats} => {
                Box::new(nn::linear(vs/ "layer1", *in_feats, *out_feats, LinearConfig::default()))
            }
            Node::MaxPool { .. } => {
                todo!()
            }
        }
    }
}


#[derive(Debug)]
struct SeqBuilder {
    num_features:i64,
    elts: Vec<Node>
}

impl SeqBuilder {
    pub fn new(num_features:i64) -> Self {
        Self {
            num_features,
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

    pub fn conv(&mut self, kernel: i64, in_channels: i64, out_channels: i64, config: ConvConfig) -> &mut Self {

        self.elts.push(
            Node::Conv {kernel, in_channels, out_channels, config },
        );

        self.num_features = clc_conv1d_out(self.num_features, kernel, config);

        self
    }

    pub fn max_pool(&mut self) -> &mut Self {
        self
    }

    pub fn avg_pool(&mut self) -> &mut Self {
        self
    }

    pub fn dropout(&mut self) -> &mut Self {
        self
    }

    pub fn linear(&mut self) -> &mut Self {
        self.elts.push(
            Node::Linear { in_feats:self.num_features, out_feats:self.num_features}
        );
        self
    }

    pub fn linear_s(&mut self, out_feats:i64) -> &mut Self {
        self.elts.push(
            Node::Linear { in_feats:self.num_features, out_feats}
        );
        self.num_features = out_feats;

        self
    }

    pub fn linear_p(&mut self, n:usize, p:&impl NthParams<Params=i64>) -> &mut Self {
        let out_feats = p.on_nth_layer(n);
        self.elts.push(
            Node::Linear { in_feats:self.num_features, out_feats}
        );
        self.num_features = out_feats;

        self
    }

    pub fn func<F>(&mut self, func: F) -> &mut Self
        where F: for<'a> FnMut(&Tensor) -> Tensor {
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

pub fn configure_dense(num_features: i64, num_blocks: usize, num_layers_in_block: usize, kernel: i64) -> Node {
    let mut builder = crate::SeqBuilder::new(num_features);

    let dense_block = |b:&mut SeqBuilder| b
        .n_blocks(
    |b:&mut SeqBuilder, i| b
                .func(|t|t.elu())
                .conv(kernel, kernel, kernel, ConvConfig::default())
                .func(|t|t.elu())
                .conv(kernel, kernel, kernel, ConvConfig::default())
                .end(),
    num_layers_in_block)
        .end();


    builder
        .conv(kernel, kernel, kernel, ConvConfig::default())
        .func(|t|t.elu())
        .max_pool()
        .n_blocks(|b:&mut SeqBuilder, i| b
            .block(dense_block)
            .block(|b:&mut SeqBuilder| b
                .func(|t|t.elu())
                .conv(kernel, kernel, kernel, ConvConfig::default())
                .func(|t|t.elu())
                .conv(kernel, kernel, kernel, ConvConfig::default())
                .end())
            .end(), num_blocks)
        .block(dense_block)
        .func(|t|t.relu())
        .avg_pool()
        .dropout()
        .linear()
        .end_seq()

}

//https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
pub fn clc_conv1d_out(l_in: i64, k: i64, c: ConvConfig) -> i64 {
    (l_in + 2 * c.padding - c.dilation * (k -1 ) -1)/c.stride +1
}



#[cfg(test)]
mod tests {
    use tch::{Device, kind, Kind, nn, Tensor};
    use tch::nn::{ConvConfig, Module};
    use crate::{configure_dense, clc_conv1d_out, SeqBuilder};

    #[test]
    pub fn test_conv1_out(){
        let l_out = clc_conv1d_out(50 * 200, 5, ConvConfig::default());

        assert_eq!(l_out, 9996);
    }

    #[test]
    pub fn test_seq_linear(){
        let seq = SeqBuilder::new(2)
            .linear()
            .end_seq();

        let vs = nn::VarStore::new(Device::Cpu);

        let net = seq.build_tch_module(&vs.root());

        let y_p = net.forward(&tch::Tensor::randn([10, 2], (Kind::Float, Device::Cpu)));

        y_p.print()
    }

    #[test]
    pub fn test_configure_dense(){
        let vs = nn::VarStore::new(Device::Cpu);

        let model = configure_dense(1000, 4, 4, 20).build_tch_module(&vs.root());

        let _ = model.forward(
            &Tensor::randn([10, 20, 1000], (Kind::Float, Device::Cpu)) //batch, channels, L in
        );
    }

    #[test]
    pub fn test_res_seq(){
        let res_seq = SeqBuilder::new(10)
                .linear()
                .end_seq();

        let conv_seq = SeqBuilder::new(10)
            .conv(5, 5, 5, ConvConfig::default())
            .end_seq();

        let main = SeqBuilder::new(10)
            .split(
                res_seq,
                conv_seq,
                |t|t.copy(),
                |t|t.copy()
            ).end_seq();

    }

    #[test]
    pub fn test_test(){
        println!("ZZZZZZZZZZZZZZZZZZB");

        let t1 = Tensor::from_slice(&[4., 6.]);
        let t2 = Tensor::from_slice(&[15., 5.]);

        let t_stack = Tensor::stack(&[t1.copy(), t2.copy()], 0);
        let t_column_stack = Tensor::column_stack(&[t1.copy(), t2.copy()]);
        let t_vstack = Tensor::vstack(&[t1.copy(), t2.copy()]);


        println!("t1={:?}, t2={:?}, t_stack={:?} t_column_stack={:?} t_vstack={:?}", t1, t2, t_stack, t_column_stack, t_vstack);
        t_stack.print();
        println!("t_stack mean={:?}", t_stack.mean_dim(1, false, kind::Kind::Float));

        t_column_stack.print();
        println!("t_column_stack mean={:?}", t_column_stack.mean_dim(1, false, kind::Kind::Float));

        t_vstack.print();
        println!("t_vstack mean={:?}", t_vstack.mean_dim(1, false, kind::Kind::Float));
    }
}