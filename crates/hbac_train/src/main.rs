mod my_tch;
mod data;
mod builder;
mod utils;
mod simp;
mod version;
mod motifs;

use std::time::Instant;
use tch::nn::{ConvConfig, Module, OptimizerConfig};
use tch::{Device, kind, Kind, nn, no_grad, Reduction, Tensor};
use plotters::prelude::*;
use crate::data::{EEGs, GenError};
use std::env;
use std::fs::File;
use std::io::Write;
use log::{debug, info};
use polars::prelude::{col, lit, NamedFrom, Series};
use simplelog::*;
use builder::{NthConst, SeqBuilder};
use crate::builder::Node;
use crate::motifs::find_motifs;
use crate::utils::DataBatch;
use crate::version::VERSION;




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
    CombinedLogger::init(
        vec![
            TermLogger::new(LevelFilter::Debug, Config::default(), TerminalMode::Mixed, ColorChoice::Auto),
            WriteLogger::new(LevelFilter::Debug, Config::default(), File::create("hbac.log")?),
        ]
    )?;

    info!("VERSION: {}", VERSION);
    std::io::stdout().flush()?;

    let args: Vec<String> = env::args().collect();
    let base_path = args.get(1).map(|s|s.as_str()).unwrap_or("../../data");

    let device = if tch::utils::has_cuda() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    debug!("device={:?}, base_path={:?}", device, base_path);

    let good_eegs = EEGs::good_eegs(base_path)?;

    let t1 = Instant::now();
    let eegs = EEGs::load_eegs (
        base_path,
        // None,
        // Some(col("patient_id").is_in(lit(Series::new("eegs_", &[44615, 8033, 2641, 10111, 30539, 57503, 13252, 17637, 38015, 58327])))),
        Some(col("eeg_id").is_in(lit(good_eegs))),
        Some(data::ByPatientIdSplitter::new(0.7)),
    )?;

    find_motifs();

    debug!("eegs={:?}", eegs.len());

    debug!("done in {}", t1.elapsed().as_millis());
    let vs = nn::VarStore::new(device.clone());



    for num_blocks in [16, 32, 64] { //[2, 4, 6, 8, 16]

        for kernel in [3, 5, 10, 20, 50, 100] { //[5, 7, 10, 20, 50, 100]
            debug!("---------------------------------------num_blocks={:?} kernel={:?}", num_blocks, kernel);

            match configure_dense(50 * 200, num_blocks, 4, kernel, 20).build_tch_module(&vs.root()) {
                Ok(model) => {
                    match utils::train(
                        &device,
                        eegs.as_ref() as &dyn DataBatch,
                        model,
                        &vs,
                        |pred: &Tensor, target: &Tensor| {
                            pred.cross_entropy_loss::<Tensor>(target, None, Reduction::Mean, -100, 0.0)
                        }
                    ){
                        Ok(_) => {
                            debug!("num_blocks={:?} kernel={:?} done", num_blocks, kernel);
                        }
                        Err(e) => {
                            debug!("num_blocks={:?} kernel={:?} train failed: {:?}", num_blocks, kernel, e);
                        }
                    }
                }
                Err(e) => {
                    debug!("num_blocks={:?} kernel={:?} build_tch_module failed: {:?}", num_blocks, kernel, e);
                }
            };

        }
    }

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
    use log::debug;
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

        let net = seq.build_tch_module(&vs.root()).unwrap();

        let y_p = net.forward(&tch::Tensor::randn([10, 2], (Kind::Float, Device::Cpu)));

        y_p.print()
    }

    #[test]
    pub fn test_configure_dense(){
        let vs = nn::VarStore::new(Device::Cpu);

        let model = configure_dense(50 * 200, 4, 4, 100, 20).build_tch_module(&vs.root()).unwrap();


        let eegs = EEGs::load_all_eegs("../../data").unwrap();

        let mut opt = nn::AdamW::default().build(&vs, 1e-2).unwrap();


        // // let tensor = Tensor::randn([10, 20, 10000], (Kind::Float, Device::Cpu));
        let (tensor, y_t) = eegs.train_batch(10);
        // debug!("size:   {:?}", tensor.size());
        let pred_y = model.forward(
            &tensor.totype(Kind::Float) //batch, channels, L in
        );
        let pred_y_mean = pred_y.mean_dim(1, false, kind::Kind::Float);
        debug!("1. pred size:   pred_y.size()={:?}     y_t.size()={:?}", pred_y_mean.size(), y_t.size());
        // let loss = y_t.mse_loss(&pred_y_mean, Reduction::Mean); //.sum(kind::Kind::Float);;
        let loss = y_t.cross_entropy_loss::<Tensor>(&pred_y_mean, None, Reduction::Mean, -100, 0.0);
        // let loss_1 = (pred_y_mean - y_t).pow_tensor_scalar(2).sum(kind::Kind::Float);
        debug!(":::::::::::::::::::::::::::: pred_y.size()={:?} y_t.size()={:?}, pred_y_mean.size()={:?},  loss.size()={:?}", pred_y.size(), y_t.size(), pred_y_mean.size(),  loss.size());
        loss.print();
        debug!("::::::::::::::::::::::::::::: loss={:?}", loss);
        opt.backward_step(&loss);
        //
        // // let tensor = Tensor::randn([10, 20, 10000], (Kind::Float, Device::Cpu));
        // let (tensor, y_t) = eegs.batch_tensor(10);
        // debug!("size:   {:?}", tensor.size());
        // let pred_y = model.forward(
        //     &tensor //batch, channels, L in
        // );
        // debug!("2. pred size:   {:?}", pred_y.size());
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
        debug!("ZZZZZZZZZZZZZZZZZZB");

        let t1 = Tensor::from_slice(&[4., 6.]);
        let t2 = Tensor::from_slice(&[15., 5.]);

        let t_stack = Tensor::stack(&[t1.copy(), t2.copy()], 0);
        let t_column_stack = Tensor::column_stack(&[t1.copy(), t2.copy()]);
        let t_vstack = Tensor::vstack(&[t1.copy(), t2.copy()]);


        debug!("t1={:?}, t2={:?}, t_stack={:?} t_column_stack={:?} t_vstack={:?}", t1, t2, t_stack, t_column_stack, t_vstack);
        t_stack.print();
        debug!("t_stack mean={:?}", t_stack.mean_dim(1, false, kind::Kind::Float));

        t_column_stack.print();
        debug!("t_column_stack mean={:?}", t_column_stack.mean_dim(1, false, kind::Kind::Float));

        t_vstack.print();
        debug!("t_vstack mean={:?}", t_vstack.mean_dim(1, false, kind::Kind::Float));
    }
}

