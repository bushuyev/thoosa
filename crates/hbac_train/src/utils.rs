use plotters::prelude::{BLACK, Color, IntoFont, LineSeries, RED, RGBColor, WHITE};
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::element::PathElement;
use tch::{Device, nn, Reduction, Tensor};
use tch::nn::{Module, OptimizerConfig, VarStore};
use plotters::drawing::IntoDrawingArea;
use crate::data::EEGs;

pub fn train(device:&Device, eegs:Box<EEGs>, net:Box<dyn Module>, vs:&VarStore){

    println!("device={:?}", device);
    const BATCH_SIZE: usize = 10usize;

    let mut opt = nn::AdamW::default().build(vs, 1e-5).unwrap();

    for idx in 1..100 {
        println!("--------------------------idx={}", idx);

        let (x_t, y_t) = eegs.batch_tensor(BATCH_SIZE);// load_batch(device, BATCH_SIZE);
        // println!("size:  x_t  {:?} y_t ={:?}", x_t.size(), y_t.size());

        // x_t.print();
        let pred_y = net.forward(&x_t);


        let pred_y_mean = pred_y.mean_dim(1, false, tch::Kind::Float);


        // println!("pred_y_mean: size={:?}", pred_y_mean);

        // let loss = (pred_y_mean - y_t).pow_tensor_scalar(2).sum(kind::Kind::Float);
        let loss = y_t.cross_entropy_loss::<Tensor>(&pred_y_mean, None, Reduction::Mean, -100, 0.0); //.sum(kind::Kind::Float);;
        opt.backward_step(&loss);

        if idx % 10 == 0 {
            println!("loss={:?}", loss);
        }
    }
}

pub fn plot(name_x_y_vec: Vec<(String, RGBColor, Vec<(f32, f32)>)>) -> Result<(), Box<dyn std::error::Error>> {

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
