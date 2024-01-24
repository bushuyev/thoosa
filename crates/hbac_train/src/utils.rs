use plotters::prelude::{BLACK, Color, IntoFont, LineSeries, RED, RGBColor, WHITE};
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::element::PathElement;
use tch::{Device, nn, Reduction, TchError, Tensor};
use tch::nn::{Module, OptimizerConfig, VarStore};
use plotters::drawing::IntoDrawingArea;
use crate::data::EEGs;
use crate::my_tch::{Fp, LossFp};

#[derive(Debug)]
enum MyError {
    TchError(TchError)
}

impl From<TchError> for MyError {
    fn from(value: TchError) -> Self {
        MyError::TchError(value)
    }
}

pub trait DataBatch {
    fn batch_tensor(&self, batch_size: usize) -> (Tensor, Tensor);
}

pub fn collect_to_vec(t:&Tensor)-> Result<Vec<f64>, MyError> {
    Ok(t.iter::<f64>()?.collect::<Vec<f64>>())
}

pub fn train(device:&Device, eegs:Box<dyn DataBatch>, net:Box<dyn Module>, vs:&VarStore, loss:LossFp){

    println!("device={:?}", device);
    const BATCH_SIZE: usize = 10usize;

    let mut opt = nn::AdamW::default().build(vs, 0.00001).unwrap();

    for idx in 1..100 {
        print!(".");

        let (x_t, y_t) = eegs.batch_tensor(BATCH_SIZE);// load_batch(device, BATCH_SIZE);

        let pred_y = net.forward(&x_t.to_device(*device));


        let pred_y_mean = pred_y.mean_dim(1, false, tch::Kind::Float);
        let loss = loss(&pred_y_mean, &y_t.to_device(*device)); //.sum(kind::Kind::Float);;

        opt.backward_step(&loss);

        if idx % 10 == 0 {
            println!("loss={:?}", loss);
        }
    }
}

pub fn plot(name_x_y_vec: Vec<(String, RGBColor, Vec<(f32, f32)>)>, caption: &str, w_h: (u32, u32)) -> Result<(), Box<dyn std::error::Error>> {

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


    let root = BitMapBackend::new("0.png", w_h).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 50).into_font())
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
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}
