use std::io::Write;
use std::mem;
use log::{debug, info, trace};
use plotters::prelude::{BLACK, Color, IntoFont, LineSeries, RED, RGBColor, WHITE};
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::element::PathElement;
use tch::{Device, nn, Reduction, TchError, Tensor};
use tch::nn::{Module, OptimizerConfig, VarStore};
use plotters::drawing::IntoDrawingArea;
use crate::data::{EEGs};
use crate::my_tch::{Fp, LossFp};
use hbac_common::HbacError;

#[derive(Debug)]
pub enum MyError {
    TchError(TchError)
}

impl From<TchError> for MyError {
    fn from(value: TchError) -> Self {
        MyError::TchError(value)
    }
}

pub trait DataBatch {
    fn train_batch(&self, batch_size: usize) -> (Tensor, Tensor);
    fn valid(&self, batch_size:Option<usize>) -> (Tensor, Tensor);
}

pub fn collect_to_vec(t:&Tensor)-> Result<Vec<f64>, MyError> {
    Ok(t.iter::<f64>()?.collect::<Vec<f64>>())
}

pub fn train(device:&Device, eegs:&dyn DataBatch, model:Box<dyn Module>, vs:&VarStore, loss_fn:LossFp) -> Result<(), HbacError>{

    debug!("device={:?}", device);
    const BATCH_SIZE: usize = 10usize;

    let mut opt = nn::AdamW::default().build(vs, 6.25e-5).unwrap();

    for idx in 1..100 {
        trace!("idx={}", idx);
        std::io::stdout().flush()?;

        let (x_t, y_t) = eegs.train_batch(BATCH_SIZE);// load_batch(device, BATCH_SIZE);

        let x_t = x_t.to_device(*device);
        let y_t = y_t.to_device(*device);

        let pred_y = model.forward(&x_t);


        let pred_y_mean = pred_y.mean_dim(1, false, tch::Kind::Float);

        let loss = loss_fn(&pred_y_mean, &y_t);

        opt.backward_step(&loss);

        tch::no_grad(||{
            if idx % 10 == 0 {
                let (x_t, y_t) = eegs.valid(Some(BATCH_SIZE));

                let pred_y = model.forward(&x_t.to_device(*device));


                let pred_y_mean = pred_y.mean_dim(1, false, tch::Kind::Float);
                let valid_loss = loss_fn(&pred_y_mean, &y_t.to_device(*device));
                info!("valid loss={:?}", valid_loss);
                std::io::stdout().flush().unwrap();
            }

        });

        if idx % 10 == 0 {
            info!("loss={:?}", loss);
            std::io::stdout().flush()?;
        }

        drop(x_t);
        drop(y_t);
        drop(pred_y_mean);
        drop(loss);
    }

    Ok(())
}