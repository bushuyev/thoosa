use std::any::Any;
use std::collections::VecDeque;
use std::io::Write;
use chrono::SecondsFormat;
use plotters::prelude::*;
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::element::PathElement;
use tch::{Device, Kind, nn, Tensor};
use tch::nn::{Module, OptimizerConfig, VarStore};
use plotters::drawing::IntoDrawingArea;
use hbac_shared::{gauss_i, HbacError, pulse_v, pulses_in_const, pulse_in_rand};
use std::error::Error;
use std::{fs, mem};
use std::ops::Deref;
use std::time::Instant;
use log::{debug, info, trace, warn};
use crate::my_tch::{Fp, LossFp};
use tch::nn::Optimizer;
use tch::IndexOp;
use tch::Kind::Float;
use torch_sys::*;
use torch_sys::cuda::atc_cuda_device_count;

pub fn plot_heatmap(path:&str, t:&Tensor)  -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .x_label_area_size(10)
        .y_label_area_size(10)
        .build_cartesian_2d(-2.1f64..0.6f64, -1.2f64..1.2f64)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()?;

    let plotting_area = chart.plotting_area();

    let range = plotting_area.get_pixel_range();

    let (pw, ph) = (range.0.end - range.0.start, range.1.end - range.1.start);
    let (xr, yr) = (chart.x_range(), chart.y_range());



    // for (x, y, c) in mandelbrot_set(xr, yr, (pw as usize, ph as usize), 100) {
    //     if c != 100 {
    //         plotting_area.draw_pixel((x, y), &MandelbrotHSL::get_color(c as f64 / 100.0))?;
    //     } else {
    //         plotting_area.draw_pixel((x, y), &BLACK)?;
    //     }
    // }

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", path);


    Ok(())
}

pub fn list_devices() ->Vec<Device> {
    let devices = unsafe {atc_cuda_device_count()};
    if devices == 0 {
        vec![Device::Cpu]
    } else {
        (0..devices).map(|i|Device::Cuda(i.try_into().unwrap())).collect()
    }
}

pub fn plot_series(name_x_y_vec: Vec<(String, RGBColor, Vec<(f32, f32)>)>, caption: &str, w_h: (u32, u32), path: &impl AsRef<std::path::Path>) -> Result<(), HbacError> {
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)?;
    }


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


    let root = BitMapBackend::new(path, w_h).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(100)
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



    // let motif_area = |x: i32, dx: i32| {
    //     return EmptyElement::at((x as f32, 10.))
    //         + Rectangle::new([(0, 0), ((dx as f32 *0.963) as i32, w_h.1 as i32 - 80)], &BLUE)//TODO 963?
    //         + Text::new(
    //         format!("({:.2},{:.2})", x, dx),
    //         (5, 5),
    //         ("sans-serif", 15.0).into_font(),
    //     );
    // };
    //
    //
    // chart.plotting_area().draw(&motif_area(100, 100)).expect("Drawing Failure");

    root.present()?;

    Ok(())
}

//from tch-0.14.0/tests/test_utils.rs
pub fn from<'a, T>(t: &'a Tensor) -> T
    where
        <T as TryFrom<&'a tch::Tensor>>::Error: std::fmt::Debug,
        T: TryFrom<&'a Tensor>,
{
    T::try_from(t).unwrap()
}


pub fn vec_f32_from(t: &Tensor) -> Vec<f32> {
    from::<Vec<f32>>(&t.reshape(-1))
}

pub fn vec_f64_from(t: &Tensor) -> Vec<f64> {
    from::<Vec<f64>>(&t.reshape(-1))
}

pub fn cdt() -> String {
    chrono::offset::Local::now().to_rfc3339_opts(SecondsFormat::Secs, true)
}

pub fn cdt_for_dir() ->String {//TODO move to shared
    cdt().replace(':', "_").replace('+', "_")
}

pub fn pad_to(input:&Tensor, to_size:&[i64])-> Result<Tensor, HbacError>{
    if input.size().len() != to_size.len() {
        HbacError::new(format!("dimensions don't match: {:?} {:?}", input.size(), to_size));
    }
    let mut to_size = to_size.to_vec();


    let mut input_size = input.size();
    input_size.reverse();
    to_size.reverse();
    let pad_idx = input_size.into_iter().zip(
        to_size.into_iter()
    )
        .enumerate()
        .flat_map(|(i, (from, to))| {
            if from > to {
                panic!("from {} > to {} at {}", from, to, i);
                [0, 0]
            } else {
                [0, to - from]
            }
        })
        .collect::<Vec<i64>>();

    Ok(
        input.pad(
            pad_idx.as_slice(),
            "constant",
            0.
        )
    )

}

pub fn load_tensor(base_path: &str, eeg_id: &str, device: Device) -> Result<Tensor, HbacError> {
    Tensor::load(std::path::Path::new(base_path).join(format!("{}.pt", eeg_id))).map(|t|t.to_device(device)).map_err(|e|HbacError::new(e.to_string())).map(|t|{
        debug!("load_tensor: t.kind()={:?}", t.kind());
        t.avg_pool1d(10, 1, 0, false, false)
    })
}


#[cfg(test)]
pub mod tests {
    use std::fs;
    use tch::Device::Cpu;
    use tch::Kind::Float;
    use tch::{Device, Tensor};
    use hbac_shared::HbacError;

    #[test]
    fn test_cdt_for_dir(){
        println!("{}", super::cdt_for_dir() );
        assert_eq!(false, super::cdt_for_dir().contains(":"))
    }

    #[test]
    fn test_pad_to() -> Result<(), HbacError>{
        let t = Tensor::vstack(&[
            Tensor::from_slice(&[1., 2.]),
            Tensor::from_slice(&[3., 4.]),
        ]);


        assert_eq!(
            Tensor::vstack(&[
                Tensor::from_slice(&[1., 2., 0.]),
                Tensor::from_slice(&[3., 4., 0.]),
                Tensor::from_slice(&[0., 0., 0.]),
            ]),
            super::pad_to(
                &t,
                &[3,3]
            )?
        );

        Ok(())
    }

    #[test]
    fn test_pad_to_same() -> Result<(), HbacError>{
        let t = Tensor::vstack(&[
            Tensor::from_slice(&[1., 2.]),
            Tensor::from_slice(&[3., 4.]),
        ]);


        assert_eq!(
            Tensor::vstack(&[
                Tensor::from_slice(&[1., 2.]),
                Tensor::from_slice(&[3., 4.]),
            ]),
            super::pad_to(
                &t,
                &[2,2]
            )?
        );

        Ok(())
    }

    #[test]
    fn test_pad_to_one_dim() -> Result<(), HbacError>{
        let t = Tensor::ones([1, 3], (Float, Cpu));


        t.pad([0, 0, 0, 1], "constant", 0.).print();
        assert_eq!(
            Tensor::vstack(&[
                Tensor::from_slice(&[1., 1., 1.]),
                Tensor::from_slice(&[0., 0., 0.]),
            ]),

        super::pad_to(
            &t,
            &[2,3]
        )?);


        Ok(())
    }


    #[test]
    fn test_save_and_load()  -> Result<(), HbacError>{

        let base_path = format!("/tmp/test_test_save_and_load/{}", super::cdt_for_dir());
        fs::create_dir_all(&base_path)?;

        let id = "1000913311";

        let sample = Tensor::rand(&[3, 3], (Float, Cpu));

        sample.save(format!("{}/{}.pt", base_path.as_str(), id));

        let loaded = super::load_tensor(base_path.as_str(), id, Cpu)?;

        assert_eq!(sample, loaded);

        Ok(())
    }

    #[test]
    fn test_devices(){
        let devices = vec!["dev1", "dev2"];
        let classes = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"];

        for (device, class) in devices.into_iter().cycle().zip(classes.into_iter()) {
            println!("device={} class={}", device, class);
        }
    }
}

pub fn z_norm(t:&Tensor, dim: i64) -> Tensor {

    let mean_t = t.mean_dim(dim, false, Kind::Float);
    let std_t = t.std(false);
    // println!("mean={:?}", vec_f32_from(&mean_t));
    // println!("std_t={:?}", vec_f32_from(&std_t));
    let z_t = (t - mean_t) / std_t;
    // println!("z_t={:?}", vec_f32_from(&z_t));
    z_t
}

pub fn z_norm_2(t_0:&Tensor, dim:i64) -> Result<Tensor, HbacError>{
    let std = t_0.std_dim(1, true, true);
    //TODO std - 0
    Ok(((t_0 -t_0.mean_dim(1, true, Kind::Float))/ std))
}


pub fn dot_norm(dim: i64, slice_01: &Tensor, slice_02: &Tensor) -> Result<Tensor, HbacError> {
    Ok(norm_2(slice_01, dim).tensordot(&norm_2(slice_02, dim), [dim], [dim]).diag(0))
}


pub fn self_dot(t1: &Tensor, i_0: i64, i_1: i64, dim:i64, step:i64, d_i: i64) -> Result<Tensor, HbacError> {
    let slice_1 = t1.f_slice(dim, Some(i_0), Some(i_0+d_i), step)?;
    let slice_2 = t1.f_slice(dim, Some(i_1), Some(i_1+d_i), step)?;

    Ok(slice_1.tensordot(&slice_2, [dim], [dim]).diag(0))
}

pub fn norm(t: &Tensor, dim: &[i64]) -> Tensor {
    t.view(dim) / t.dot(&t).sqrt()
}

pub fn norm_2(t: &Tensor, dim: i64) -> Tensor {
    t/ t.tensordot(&t, dim, dim).diag(0).sqrt().view([1, -1]).transpose(0, 1)
}

pub fn lin_t(n: i64) -> Tensor {
    Tensor::from((0..n).map(|x| x as f32).collect::<Vec<f32>>().as_slice())
}


pub trait DataBatch {
    fn train_batch(&mut self) -> (Tensor, Tensor);
    fn valid_batch(&mut self) -> (Tensor, Tensor);
}

pub fn train<F>(device:&Device, data:&mut dyn DataBatch, model:&Box<dyn Module>, loss_fn:LossFp, cycles: usize, mut opt: Optimizer, x_f:F) -> Result<f32, HbacError>
    where F:Fn(Tensor)->Tensor {

    debug!("train: device={:?} cycles={}", device, cycles);


    let mut last_valid_loss = [0f32;3];

    for idx in 1..cycles {

        let (x_t_0, y_t) = data.train_batch();
        debug!("train: device={:?} x_t_0.size()={:?} ", device, x_t_0.size());

        let x_t = x_f(x_t_0);

        debug!("train: device={:?}  x_t.size()={:?} y_t.size()={:?}", device, x_t.size(), y_t.size());


        let pred_y = model.forward(&x_t);

        mem::drop(x_t);
        unsafe { at_autocast_clear_cache(); }

        debug!("train: device={:?} pred_y.size()={:?}", device, pred_y.size());
        debug!("train: device={:?}   y_t ={:?}", device, from::<Vec<f32>>(&y_t));

        // let pred_y_mean = pred_y.view((10, 1)); //pred_y.mean_dim(1, false, tch::Kind::Float);

        let loss = loss_fn(&pred_y, &y_t);

        let t1 = Instant::now();



        opt.backward_step(&loss);

        debug!("train: device={:?}  backward_step done in {}", device, t1.elapsed().as_millis());



        if idx % 10 == 0 {

            tch::no_grad(||{
                let (x_t_0, y_t) = data.valid_batch();

                let x_t = x_f(x_t_0);

                debug!("train: device={:?}  vaid    x_t.size()={:?}",  device, x_t.size());

                let pred_y = model.forward(&x_t);

                debug!("train: device={:?}  vaid pred_y.size()={:?} y_t.size()={:?}", device, pred_y.size(), y_t.size());
                debug!("train: device={:?}  vaid    y_t.size()={:?}",  device, from::<Vec<f32>>(&y_t));
                // debug!("vaid: pred_y.size()={:?} ", from::<Vec<f32>>(&pred_y));

                // let pred_y_mean = pred_y.view((10, 1)); // pred_y.mean_dim(1, false, tch::Kind::Float);

                let valid_loss = loss_fn(&pred_y, &y_t);
                last_valid_loss[idx % last_valid_loss.len()] = valid_loss.double_value(&[]) as f32;
                info!("train: device={:?}  loss={:?}, valid loss={:?} avg loss={}", device, loss, valid_loss, last_valid_loss.into_iter().sum::<f32>() / last_valid_loss.len() as f32);
                std::io::stdout().flush().unwrap();
            });
        }



        debug!("train: idx={} loss={:?}", idx, loss);
        std::io::stdout().flush()?;


        // drop(x_t);
        drop(y_t);
        drop(pred_y);
        drop(loss);
    }

    Ok(last_valid_loss.into_iter().sum::<f32>() / last_valid_loss.len() as f32)
}



pub fn two_sigs(n: i64) -> (Tensor, Tensor) {
    let sig_1 = Tensor::from_slice(
        pulses_in_const(n, &[
            // (10, pulse_v(8., 3., 20, 2.)),
            // (30, pulse_v(8., 3., 20, 2.))
            (100, pulse_v(80., 30., 200, 20.)),
            (300, pulse_v(80., 30., 200, 20.))
        ]).as_slice()
    );

    let sig_2 = Tensor::from_slice(
        pulse_in_rand(n, &[], 10., 0.001, 0.1).as_slice()
    );
    (sig_1, sig_2)
}


pub fn gauss_t(μ: f32, σ: f32, n: i64) -> Tensor {
    Tensor::from(gauss_i(μ, σ, n).collect::<Vec<_>>().as_slice())
}

pub fn pulse_t(μ: f32, σ: f32, n: i64, f: f32) -> Tensor {
    Tensor::from(pulse_v(μ, σ, n, f).as_slice())
}
