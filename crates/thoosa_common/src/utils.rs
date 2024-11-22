use std::{fs, mem};
use std::ffi::c_void;
use std::io::Write;
use std::time::Instant;

use chrono::SecondsFormat;
use log::{debug, info};
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::drawing::IntoDrawingArea;
use plotters::element::PathElement;
use plotters::prelude::*;
use rand::prelude::ThreadRng;
use rand::Rng;
use rand::thread_rng;
use tch::{Device, Kind, Tensor};
use tch::Kind::Float;
use tch::kind::FLOAT_CPU;
use tch::nn::Module;
use tch::nn::Optimizer;
use torch_sys::*;
use torch_sys::cuda::atc_cuda_device_count;

use thoosa_shared::{gauss_i, ThoosaError, pulse_in_rand, pulse_v, pulses_in_const, cdt, saw};

use crate::my_tch::LossFp;
use crate::parzen::opt_rng::opt_rnd;

pub fn plot_heatmap(path:&str, _t:&Tensor) -> Result<(), Box<dyn std::error::Error>> {
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

    let (_pw, _ph) = (range.0.end - range.0.start, range.1.end - range.1.start);
    let (_xr, _yr) = (chart.x_range(), chart.y_range());



    // for (x, y, c) in mandelbrot_set(xr, yr, (pw as usize, ph as usize), 100) {
    //     if c != 100 {
    //         plotting_area.draw_pixel((x, y), &MandelbrotHSL::get_color(c as f64 / 100.0))?;
    //     } else {
    //         plotting_area.draw_pixel((x, y), &BLACK)?;
    //     }
    // }

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    debug!("Result has been saved to {}", path);


    Ok(())
}

pub fn list_devices() ->Vec<Device> {
    let cuda_if_available = Device::cuda_if_available();
    debug!("cuda_if_available={cuda_if_available:?}");

    let devices = unsafe {atc_cuda_device_count()};
    if devices == 0 {
        vec![Device::Cpu]
    } else {
        (0..devices).map(|i|Device::Cuda(i.try_into().unwrap())).collect()
    }
}

pub fn plot_series<S>(name_x_y_vec: Vec<(String, S, Vec<(f32, f32)>)>, caption: &str, w_h: (u32, u32), path: &impl AsRef<std::path::Path>) -> Result<(), ThoosaError> where S:Into<ShapeStyle> + Clone{
    plot_series_with_points(name_x_y_vec, caption, w_h, path, None)
}

pub fn plot_series_with_points<S>(name_x_y_vec: Vec<(String, S, Vec<(f32, f32)>)>, caption: &str, w_h: (u32, u32), path: &impl AsRef<std::path::Path>, points:Option<&Vec<(f32, f32)>>) -> Result<(), ThoosaError> where S:Into<ShapeStyle> + Clone{
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
                color.clone(),
            ))?
            .label(name)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.clone()));
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

    if let Some(points) = points{
        for point in points {
            root.draw(&Circle::new(chart.backend_coord(point), 10, ShapeStyle {
                color: BLUE. mix(0.6),
                filled: false,
                stroke_width: 2,
            }))?;
        }
    }

    root.present()?;

    Ok(())
}



pub fn plot_3d(){//TODO
    // let root = BitMapBackend::gif(crate::bfgs::opt_tests::OUT_FILE_NAME, (600, 400), 100)?.into_drawing_area();
    //
    // for pitch in 9..10 {
    //     root.fill(&WHITE)?;
    //
    //     let mut chart = ChartBuilder::on(&root)
    //         .caption("2D Gaussian PDF", ("sans-serif", 20))
    //         .build_cartesian_3d(-100.0..100.0, -100.0..100.0, -100.0..100.0)?;
    //     chart.with_projection(|mut p| {
    //         p.pitch = 1.57 - (1.57 - pitch as f64 / 50.0).abs();
    //         p.scale = 0.7;
    //         p.into_matrix() // build the projection matrix
    //     });
    //
    //     chart
    //         .configure_axes()
    //         .light_grid_style(BLACK.mix(0.15))
    //         .max_light_lines(3)
    //         .draw()?;
    //
    //     chart.draw_series(
    //         SurfaceSeries::xoz(
    //             (-100..=100).map(|x| x as f64 /5.),
    //             (-100..=100).map(|x| x as f64 /5.),
    //             crate::bfgs::opt_tests::pdf,
    //         )
    //             .style_func(&|&v| (VulcanoHSL::get_color(v / 5.0)).into()),
    //     )?;
    //
    //     root.present()?;
    // }
    //
    // // To avoid the IO failure being ignored silently, we manually call the present function
    // root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    // debug!("Result has been saved to {}", crate::bfgs::opt_tests::OUT_FILE_NAME);
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



pub fn pad_to(input:&Tensor, to_size:&[i64])-> Result<Tensor, ThoosaError>{
    if input.size().len() != to_size.len() {
        ThoosaError::new(format!("dimensions don't match: {:?} {:?}", input.size(), to_size));
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

pub fn load_tensor(base_path: &str, tensor_id: &str, device: Device) -> Result<Tensor, ThoosaError> {
    Tensor::load(std::path::Path::new(base_path).join(format!("{}.pt", tensor_id)))
        .map(|t|t.to_device(device)).map_err(|e| ThoosaError::new(e.to_string()))
}

pub fn clone(from:&Tensor) -> Tensor {
    let to = Tensor::zeros(from.size(), FLOAT_CPU);//TODO clone
    let _ = from.clone(&to);

    to
}

#[cfg(test)]
pub mod tests {
    use std::collections::HashSet;
    use std::{fs, mem};

    use clap::{arg, Args, command, Parser, Subcommand};
    use log::debug;
    use tch::Device::Cpu;
    use tch::Kind::Float;
    use tch::Tensor;

    use thoosa_shared::{cdt_for_dir, ThoosaError};
    use crate::bool_iter::BoolTensorIterator;

    #[test]
    fn test_cdt_for_dir(){
        debug!("{}", cdt_for_dir() );
        assert_eq!(false, cdt_for_dir().contains(":"))
    }

    #[test]
    fn test_pad_to() -> Result<(), ThoosaError>{
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
    fn test_pad_to_same() -> Result<(), ThoosaError>{
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
    fn test_pad_to_one_dim() -> Result<(), ThoosaError>{
        let t = Tensor::ones([1, 3], (Float, Cpu));

        assert_eq!(
            Tensor::vstack(&[
                Tensor::from_slice(&[1., 1., 1.]),
                Tensor::from_slice(&[0., 0., 0.]),
            ]),

            super::pad_to(
                &t,
                &[2,3]
            )?
        );


        Ok(())
    }


    #[test]
    fn test_save_and_load()  -> Result<(), ThoosaError>{

        let base_path = format!("/tmp/test_test_save_and_load/{}", cdt_for_dir());
        fs::create_dir_all(&base_path)?;

        let id = "1000913311";

        let sample = Tensor::rand(&[3, 3], (Float, Cpu));

        sample.save(format!("{}/{}.pt", base_path.as_str(), id))?;

        let loaded = super::load_tensor(base_path.as_str(), id, Cpu)?;

        assert_eq!(sample, loaded);

        Ok(())
    }

    #[test]
    fn test_devices(){
        let devices = vec!["dev1", "dev2"];
        let classes = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"];


        let chunk_len = classes.len()/devices.len();
        let devices_classes = devices.into_iter().cycle().zip(classes.chunks(chunk_len).into_iter()).collect::<Vec<(&str, &[&str])>>();
        let mut processed_classes:HashSet<&str> = HashSet::new();
        devices_classes.into_iter().for_each( |(device, classes_chunk)| {
            debug!("2. device={:?} class={:?}", device, classes_chunk);
            processed_classes.extend(classes_chunk)
        });

        assert_eq!(processed_classes.len(), 6);
        //TODO devices

        debug!("processed_classes={:?}", processed_classes);

    }

    #[test]
    fn test_param_ranges(){

        #[derive(Parser, Clone)]
        #[command(version, about, long_about = None)]
        struct TestCli {
            #[arg(long)]
            some_field: String,

            #[command(subcommand)]
            command: Option<Commands>,
        }

        #[derive(Debug, Subcommand, Clone)]
        enum Commands {
            TestCommand(TestCommandParams),
        }

        #[derive(Debug, Args, Clone)]
        struct TestCommandParams{
            #[arg(long)]
            window:usize,

            #[arg(long)]
            step:usize,

            #[arg(long)]
            command_param:String,

            #[command(subcommand)]
            input_filter: InputFilter
        }

        #[derive(Debug, Subcommand, Clone)]
        enum InputFilter {
            Conv1(Conv1Params)
        }


        #[derive(Debug, Args, Clone)]
        struct Conv1Params {
            #[arg(long)]
            kernel:usize
        }

        let cli = TestCli::parse_from(["", "--some-field", "zzz", "test-command", "--command-param", "command-param-value", "--window", "11", "--step", "22", "conv1", "--kernel", "10"]);
        debug!("cli.some_field={:?}", cli.some_field);

        match cli.command.unwrap() {
            Commands::TestCommand(cmd)=>{
                debug!("cmd.command_param={:?}", cmd.command_param);

                match cmd.input_filter {
                    InputFilter::Conv1(filter_params) => {
                        debug!("filter_params={:?}", filter_params);
                    }
                }
            }
        }
    }

    #[test]
    fn test_bool_tensor_iterator(){

        let t = Tensor::from_slice((0..100).map(|v|v % 2 == 0).collect::<Vec<_>>().as_slice());


        for (i, b) in (0..100).zip::<BoolTensorIterator>((&t).into()){
            assert_eq!(i % 2 == 0, b);
        }

    }

}

pub fn z_norm(t:&Tensor, dim: i64) -> Tensor {

    let mean_t = t.mean_dim(dim, false, Float);
    let std_t = t.std(false);
    let z_t = (t - mean_t) / std_t;
    z_t
}

pub fn z_norm_2(t_0:&Tensor, _dim:i64) -> Result<Tensor, ThoosaError>{
    let std = t_0.std_dim(1, true, true);
    //TODO std - 0
    Ok((t_0 -t_0.mean_dim(1, true, Float))/ std)
}


pub fn dot_norm(dim: i64, slice_01: &Tensor, slice_02: &Tensor) -> Result<Tensor, ThoosaError> {
    Ok(norm_2(slice_01, dim).tensordot(&norm_2(slice_02, dim), [dim], [dim]).diag(0))
}


pub fn self_dot(t1: &Tensor, i_0: i64, i_1: i64, dim:i64, step:i64, d_i: i64) -> Result<Tensor, ThoosaError> {
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

pub trait TrainListener {
    fn on_cycle(&mut self, cycle:usize, loss:f32) -> ListenerResponse {
        ListenerResponse::Nope
    }
    fn on_valid_cycle(&mut self, cycle:usize, loss:f32) -> ListenerResponse;
}

pub enum ListenerResponse {
    Stop,
    UpdateLR(f64),
    Nope
}

pub fn train<F>(
    device:&Device,
    data:&mut dyn DataBatch,
    model:&Box<dyn Module>,
    loss_fn:LossFp,
    cycles: usize,
    mut opt: Optimizer,
    x_f:F,
    mut listener: Option<&mut dyn TrainListener>
) -> Result<f32, ThoosaError>
    where F:Fn(Tensor)->Tensor {

    debug!("train: device={:?} cycles={}", device, cycles);


    let mut last_valid_loss = [0f32;3];

    for idx in 1..cycles {

        let (x_t_0, y_t) = data.train_batch();
        debug!("train: device={:?} x_t_0.size()={:?} ", device, x_t_0.size());

        let x_t = x_f(x_t_0);

        debug!("train: device={:?}  x_t.size()={:?} y_t.size()={:?}", device, x_t.size(), y_t.size());


        let pred_y = model.forward(&x_t);

        drop(x_t);
        unsafe { at_autocast_clear_cache(); }

        // debug!("train: device={:?} pred_y.size()={:?}", device, pred_y.size());
        // debug!("train: device={:?}   y_t ={:?}", device, from::<Vec<f32>>(&y_t));

        // let pred_y_mean = pred_y.view((10, 1)); //pred_y.mean_dim(1, false, tch::Kind::Float);

        let loss = loss_fn(&pred_y, &y_t);

        let t1 = Instant::now();

        opt.backward_step(&loss);

        debug!("train: device={:?}  backward_step done in {}", device, t1.elapsed().as_millis());


        if idx % 10 == 0 {

            if let Some(rs) = tch::no_grad(||{
                let (x_t_0, y_t) = data.valid_batch();
                // debug!("train: x_t_0.size={:?}", x_t_0.size());

                let x_t = x_f(x_t_0);


                let pred_y = model.forward(&x_t);

                let valid_loss = loss_fn(&pred_y, &y_t);
                let valid_loss_f32 = valid_loss.double_value(&[]) as f32;


                last_valid_loss[idx % last_valid_loss.len()] = valid_loss_f32;
                info!("train: last_valid_loss={last_valid_loss:?}  loss={loss:?}, valid loss={valid_loss:?} avg loss={}", last_valid_loss.into_iter().sum::<f32>() / last_valid_loss.len() as f32);

                listener.as_mut().map(|l|l.on_valid_cycle(idx, valid_loss_f32))

            }).take() {
                match rs {
                    ListenerResponse::Stop => {
                        break;
                    }
                    ListenerResponse::UpdateLR(lr) => {
                        opt.set_lr(lr);
                    }
                    ListenerResponse::Nope => {}
                }
            }
        }

        debug!("train: idx={} loss={:?}", idx, loss);
        std::io::stdout().flush()?;
    }
    debug!("train: final valid last_valid_loss={last_valid_loss:?} avgvalid_loss={}",  last_valid_loss.into_iter().sum::<f32>() / last_valid_loss.len() as f32);
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

//f(x)=-sin(3x)+x^2+0.7 *x
pub fn sin3_pow2(n:i64, min: f32, max: f32, with_noise: Option<f32>) -> (Tensor, Tensor, Vec<f32>) {

    let mut rng = thread_rng();
    let x = (0..n).map(|i| min + (i as f32 / (n-1) as f32) * (max-min)).into_iter().collect::<Vec<f32>>();

    let y_train_full = Tensor::from_slice(x.iter().map(|x|{
        sin3_pow2_with_noise_f(with_noise, &mut rng, *x)
    } ).collect::<Vec<f32>>().as_slice()).view((1, -1)).totype(Float); //.transpose(0,1);

    let x_t = Tensor::from_slice(x.as_slice()).totype(Float).view((-1, 1));


    (y_train_full, x_t, x)
}

fn sin3_pow2_with_noise_f(with_noise: Option<f32>, rng: &mut ThreadRng, x: f32) -> f32 {
    sin3_pow2_f(x) + with_noise.map(|v| v * rng.gen::<f32>()).unwrap_or(0.)
}

pub fn sin3_pow2_f(x: f32) -> f32 {
    -(3. * x).sin() + x.powi(2) + 0.7 * x
}

pub fn sin3_pow2_t(t:&Tensor)->Tensor {
    t.multiply_scalar(-3.).sin() + t.pow_tensor_scalar(2) + t.multiply_scalar(0.7) //TODO noise?
}


pub fn gauss_t(μ: f32, σ: f32, n: i64) -> Tensor {
    Tensor::from(gauss_i(μ, σ, n).collect::<Vec<_>>().as_slice())
}

pub fn pulse_t(μ: f32, σ: f32, n: i64, f: f32) -> Tensor {
    Tensor::from(pulse_v(μ, σ, n, f).as_slice())
}

pub fn three_sigs_tensor(n: i64, sig_0_v: &Vec<f32>, sig_1_v: &Vec<f32>, sig_2_v: &Vec<f32>) -> (Tensor, Tensor, Tensor, Tensor) {
    let t_0_0 = Tensor::vstack(&[
        Tensor::from_slice([sig_0_v.clone(), saw(n), saw(n)].concat().as_slice()),
        Tensor::from_slice([saw(n), saw(n), saw(n)].concat().as_slice()),
        Tensor::from_slice([saw(n), sig_2_v.clone(), saw(n)].concat().as_slice()),
    ]);

    // debug!("t_0_0 mean={} max-min={}", t_0_0.mean(None).double_value(&[]), t_0_0.max() - t_0_0.min());

    let t_0_1 = Tensor::vstack(&[
        Tensor::from_slice([saw(n), saw(n), sig_0_v.clone()].concat().as_slice()),
        Tensor::from_slice([saw(n), saw(n), saw(n)].concat().as_slice()),
        Tensor::from_slice([sig_2_v.clone(), saw(n), saw(n)].concat().as_slice()),
    ]);

    // debug!("t_0_1 mean={} max-min={}", t_0_1.mean(None).double_value(&[]), t_0_1.max() - t_0_1.min());


    let t_1_0 = Tensor::vstack(&[
        Tensor::from_slice([saw(n), saw(n), saw(n)].concat().as_slice()),
        Tensor::from_slice([saw(n), sig_1_v.clone(), saw(n)].concat().as_slice()),
        Tensor::from_slice([saw(n), sig_2_v.clone(), saw(n)].concat().as_slice())
    ]);


    // debug!("t_1_0 mean={} max-min={}", t_1_0.mean(None).double_value(&[]), t_1_0.max() - t_1_0.min());

    let t_1_1 = Tensor::vstack(&[
        Tensor::from_slice([saw(n), saw(n), saw(n)].concat().as_slice()),
        Tensor::from_slice([saw(n), saw(n), sig_1_v.clone()].concat().as_slice()),
        Tensor::from_slice([saw(n), saw(n), sig_2_v.clone()].concat().as_slice()),
    ]);
    (t_0_0, t_0_1, t_1_0, t_1_1)
}

pub fn copy_cudarc(t:&Tensor) -> *mut c_void {
    use cudarc::driver::sys::CUdeviceptr_v2;
    unsafe {
        let mut host_buffer: *mut c_void = std::ptr::null_mut();
        let size_in_bytes = t.numel() * size_of::<f32>();
        let cudarc_lib = cudarc::driver::sys::lib();
        let r = cudarc_lib.cuMemAllocHost_v2(
            &mut host_buffer,
            size_in_bytes
        );

        let device = cudarc::driver::CudaDevice::new(0).expect("cuda 0");

        let r = cudarc_lib.cuMemcpyAsync(
            host_buffer as CUdeviceptr_v2,
            t.data_ptr() as CUdeviceptr_v2,
            size_in_bytes,
            *device.cu_stream()
        );
        cudarc_lib.cuStreamSynchronize(*device.cu_stream());

        host_buffer
    }


}

pub fn free_cudarc(buf : *mut f32){
    unsafe {
        cudarc::driver::sys::lib().cuMemFreeHost(buf as *mut c_void);
    }
}
