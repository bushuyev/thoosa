mod my_tch;
mod data;
mod builder;
mod utils;
mod simp;

use std::time::Instant;
use tch::nn::{ConvConfig, Module, OptimizerConfig};
use tch::{Device, kind, Kind, nn, no_grad, Reduction, Tensor};
use plotters::prelude::*;
use crate::data::{EEGs, GenError};
use std::env;
use builder::{NthConst, SeqBuilder};
use crate::builder::Node;


fn train_x2(device:&Device) {
    println!("test_x2device={:?}", device);
    const BATCH_SIZE: i64 = 10i64;


    let vs = nn::VarStore::new(device.clone());

    let nth_params = NthConst::new(100);

    let lin_block = |b: &mut SeqBuilder, i| b
        .linear_p(i, &nth_params)
        .func(Tensor::elu)
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

        utils::plot(
            vec![
                ("pred".into(), RED, pred),
                ("test".into(), GREEN, test),
            ], "pred/test", (640, 480)
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

pub fn configure_dense(num_features: i64, num_blocks: usize, num_layers_in_block: usize, kernel: i64, channels: i64) -> Node  {
    let mut builder = crate::SeqBuilder::new(num_features);

    let dense_block = |b:&mut SeqBuilder| b
        .n_blocks(
            |b:&mut SeqBuilder, i| b
                .func(Tensor::elu)
                .conv(kernel, channels, channels, ConvConfig::default())
                .func(Tensor::elu)
                .conv(kernel, channels, channels, ConvConfig::default())
                .end(),
            num_layers_in_block)
        .end();


    builder
        .conv(kernel, channels, channels, ConvConfig::default())
        .func(|t|t.elu())
        .max_pool( 3, 2, 1)
        .n_blocks(|b:&mut SeqBuilder, i| b
            .block(dense_block)
            .block(|b:&mut SeqBuilder| b
                .func(Tensor::elu)
                .conv(kernel, channels, channels, ConvConfig::default())
                .func(Tensor::elu)
                .conv(kernel, channels, channels, ConvConfig::default())
                .end())
            .end(), num_blocks)
        .block(dense_block)
        .func(Tensor::relu)
        .avg_pool(7, 3, 0)
        .dropout()
        .linear_s(6)
        .end_seq()

}


fn main() -> Result<(), GenError> {
    let args: Vec<String> = env::args().collect();
    let base_path = args.get(1).map(|s|s.as_str()).unwrap_or("../../data");

    let device = if tch::utils::has_cuda() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    println!("device={:?}, base_path={:?}", device, base_path);


    let t1 = Instant::now();
    let eegs = EEGs::load_all_eegs(base_path)?;


    println!("eegs={:?}", eegs.len());

    println!("done in {}", t1.elapsed().as_millis());
    let vs = nn::VarStore::new(device.clone());


    utils::train(
        &device,
        eegs,
        configure_dense(50 * 200, 4, 4, 100, 20).build_tch_module(&vs.root()),
        &vs,
         |pred:&Tensor, target:&Tensor|{
             pred.cross_entropy_loss::<Tensor>(target, None, Reduction::Mean, -100, 0.0)
        }
    );

    Ok(())

}

fn load_batch(device: &Device, batch_size:i64) -> (Tensor, Tensor) {
    (
    tch::Tensor::randn([batch_size, 50 * 200, 20], (Kind::Float, device.clone())), //batch, 20 EEG feats, 50 sec * 200 samples per second,
    tch::Tensor::randn([batch_size, 6], (Kind::Float, device.clone())) //batch, seizure_vote,lpd_vote,gpd_vote, lrda_vote, grda_vote,other_vote, 50 sec * 200 samples per second,
    )
}


#[cfg(test)]
mod tests {
    use tch::{Device, kind, Kind, nn, Reduction, Tensor};
    use tch::nn::{ConvConfig, Module, OptimizerConfig};
    use crate::builder::SeqBuilder;
    use crate::builder::clc_conv1d_out;
    use crate::configure_dense;
    use crate::data::EEGs;
    use crate::utils::DataBatch;

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

        let model = configure_dense(50 * 200, 4, 4, 100, 20).build_tch_module(&vs.root());


        let eegs = EEGs::load_all_eegs("../../data").unwrap();

        let mut opt = nn::AdamW::default().build(&vs, 1e-2).unwrap();


        // // let tensor = Tensor::randn([10, 20, 10000], (Kind::Float, Device::Cpu));
        let (tensor, y_t) = eegs.batch_tensor(10);
        // println!("size:   {:?}", tensor.size());
        let pred_y = model.forward(
            &tensor.totype(Kind::Float) //batch, channels, L in
        );
        let pred_y_mean = pred_y.mean_dim(1, false, kind::Kind::Float);
        println!("1. pred size:   pred_y.size()={:?}     y_t.size()={:?}", pred_y_mean.size(), y_t.size());
        // let loss = y_t.mse_loss(&pred_y_mean, Reduction::Mean); //.sum(kind::Kind::Float);;
        let loss = y_t.cross_entropy_loss::<Tensor>(&pred_y_mean, None, Reduction::Mean, -100, 0.0);
        // let loss_1 = (pred_y_mean - y_t).pow_tensor_scalar(2).sum(kind::Kind::Float);
        println!(":::::::::::::::::::::::::::: pred_y.size()={:?} y_t.size()={:?}, pred_y_mean.size()={:?},  loss.size()={:?}", pred_y.size(), y_t.size(), pred_y_mean.size(),  loss.size());
        loss.print();
        println!("::::::::::::::::::::::::::::: loss={:?}", loss);
        opt.backward_step(&loss);
        //
        // // let tensor = Tensor::randn([10, 20, 10000], (Kind::Float, Device::Cpu));
        // let (tensor, y_t) = eegs.batch_tensor(10);
        // println!("size:   {:?}", tensor.size());
        // let pred_y = model.forward(
        //     &tensor //batch, channels, L in
        // );
        // println!("2. pred size:   {:?}", pred_y.size());
        // let pred_y_mean = pred_y.mean_dim(1, false, kind::Kind::Float);
        // // let loss = (pred_y_mean - y_t).pow_tensor_scalar(2).sum(kind::Kind::Float);
        // let loss = y_t.mse_loss(&pred_y_mean, Reduction::None);
        // opt.backward_step(&loss);
        // pred_y.print();




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

