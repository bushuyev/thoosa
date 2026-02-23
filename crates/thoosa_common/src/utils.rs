#![allow(non_camel_case_types)]

use std::collections::HashMap;
use std::fmt::Display;
use std::iter::Map;

use std::ops::Range;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::{fs, ptr};
use log::{debug, error, trace};
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::drawing::IntoDrawingArea;
use plotters::element::PathElement;
use plotters::prelude::*;
use rand::prelude::ThreadRng;
use rand::{Rng, RngExt};
use std::sync::Arc;
use tch::display::PrinterOptions;
use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
use tch::nn::{LinearConfig, Optimizer};
use tch::nn::VarStore;
use tch::Kind::Float;
use tch::{nn, Device, IndexOp, Kind, TchError, Tensor};
use thoosa_shared::{cdt_for_dir, gauss_i, panic_to_err, pulse_in_rand, pulse_v, pulses_in_const, saw, ThoError};
use torch_sys::cuda::atc_cuda_device_count;

use crate::builder::SeqBuilder;
use crate::dims::{Batch, Dims, MotifProbability, Slice};
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use lazy_static::lazy_static;
use thoosa_shared::colors_iter::DistinctColorIter;

#[macro_export]
macro_rules! debug_plot {
    ($tensor:expr, $fmt:literal, $($arg:expr),+ $(,)?) => {{
        #[allow(unexpected_cfgs)]
        #[cfg(tho_debug_plots)]
        {
            $crate::utils::debug_plot_impl($tensor, &format!($fmt, $($arg),+))
        }
        #[allow(unexpected_cfgs)]
        #[cfg(not(tho_debug_plots))]
        {
            Ok::<(), thoosa_shared::ThoError>(())
        }
    }};
    ($tensor:expr, $fmt:literal $(,)?) => {{
        #[allow(unexpected_cfgs)]
        #[cfg(tho_debug_plots)]
        {
            $crate::utils::debug_plot_impl($tensor, &format!($fmt))
        }
        #[allow(unexpected_cfgs)]
        #[cfg(not(tho_debug_plots))]
        {
            Ok::<(), thoosa_shared::ThoError>(())
        }
    }};
    ($tensor:expr, $label:expr $(,)?) => {{
        #[allow(unexpected_cfgs)]
        #[cfg(tho_debug_plots)]
        {
            $crate::utils::debug_plot_impl($tensor, &format!("{}", $label))
        }
        #[allow(unexpected_cfgs)]
        #[cfg(not(tho_debug_plots))]
        {
            Ok::<(), thoosa_shared::ThoError>(())
        }
    }};
}

pub fn plot_heatmap(path: &str, _t: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
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

pub fn list_devices() -> Vec<Device> {
    let cuda_if_available = Device::cuda_if_available();
    debug!("cuda_if_available={cuda_if_available:?}");

    let devices = unsafe { atc_cuda_device_count() };
    if devices == 0 {
        vec![Device::Cpu]
    } else {
        (0..devices)
            .map(|i| Device::Cuda(i.try_into().unwrap()))
            .collect()
    }
}

lazy_static! {
    static ref DEBUG_PLOT_LABEL_COUNTS: Mutex<HashMap<String, u64>> = Mutex::new(HashMap::new());
}

static DEBUG_PLOT_DIR: OnceLock<Result<PathBuf, String>> = OnceLock::new();

fn debug_plot_dir() -> Result<&'static PathBuf, ThoError> {
    let path = DEBUG_PLOT_DIR.get_or_init(|| {
        let path = std::env::temp_dir().join("plots").join(cdt_for_dir());
        fs::create_dir_all(&path).map_err(|e| format!("debug_plot: create_dir_all({path:?}) failed: {e}"))?;
        Ok(path)
    });
    match path {
        Ok(path) => Ok(path),
        Err(e) => Err(ThoError::new(e.clone())),
    }
}

fn sanitize_debug_plot_label(label: &str) -> String {
    let sanitized = label
        .chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' | '.' => c,
            _ => '_',
        })
        .collect::<String>();
    if sanitized.is_empty() { "plot".to_string() } else { sanitized }
}

#[doc(hidden)]
pub fn debug_plot_impl(t_0: &Tensor, label: &str) -> Result<(), ThoError> {
    let file_stem = sanitize_debug_plot_label(label);
    let file_name = {
        let mut counts = DEBUG_PLOT_LABEL_COUNTS
            .lock()
            .map_err(|e| ThoError::new(format!("debug_plot: lock failed: {e}")))?;
        let seen = counts.entry(file_stem.clone()).or_insert(0);
        let file_name = if *seen == 0 {
            format!("{file_stem}.png")
        } else {
            format!("{file_stem}_{}.png", *seen - 1)
        };
        *seen += 1;
        file_name
    };
    let path = debug_plot_dir()?.join(file_name);

    plot_series(
        (0..t_0.size()[0])
            .into_iter()
            .zip(DistinctColorIter::new())
            .map(|(channel, color)| {
                (
                    format!("channel {channel}"),
                    color,
                    t_0.i((channel, ..))
                        .iter::<f64>()
                        .unwrap()
                        .enumerate()
                        .map(|(i, v)| (i as f32, v as f32))
                        .collect::<Vec<(f32, f32)>>(),
                )
            })
            .collect(),
        label,
        (2000, 1000),
        &path,
    )?;
    Ok(())
}


pub fn plot_series<S>(
    name_x_y_vec: Vec<(String, S, Vec<(f32, f32)>)>,
    caption: &str,
    w_h: (u32, u32),
    path: &impl AsRef<std::path::Path>,
) -> Result<(), ThoError>
where
    S: Into<ShapeStyle> + Clone,
{
    _plot_series(name_x_y_vec, caption, w_h, path, None, None)
}

pub fn plot_series_with_points<S>(
    name_x_y_vec: Vec<(String, S, Vec<(f32, f32)>)>,
    caption: &str,
    w_h: (u32, u32),
    path: &impl AsRef<std::path::Path>,
    points: &Vec<(f32, f32)>,
) -> Result<(), ThoError>
where
    S: Into<ShapeStyle> + Clone,
{
    _plot_series(name_x_y_vec, caption, w_h, path, Some(points), None)
}

pub fn plot_series_with_lanes<S>(
    name_x_y_vec: Vec<(String, S, Vec<(f32, f32)>)>,
    caption: &str,
    w_h: (u32, u32),
    path: &impl AsRef<std::path::Path>,
    lanes: &[(i32, i32)],
) -> Result<(), ThoError>
where
    S: Into<ShapeStyle> + Clone,
{
    _plot_series(name_x_y_vec, caption, w_h, path, None, Some(lanes))
}


fn _plot_series<S>(
    name_x_y_vec: Vec<(String, S, Vec<(f32, f32)>)>,
    caption: &str,
    w_h: (u32, u32),
    path: &impl AsRef<std::path::Path>,
    points: Option<&Vec<(f32, f32)>>,
    _lanes: Option<&[(i32, i32)]>,
) -> Result<(), ThoError>
where
    S: Into<ShapeStyle> + Clone,
{
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)?;
    }

    let (min_x, max_x, min_y, max_y) = name_x_y_vec
        .iter()
        .flat_map(|name_x_y_vec| name_x_y_vec.2.iter())
        .fold(
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN),
            |mut max_min, (x, y)| {
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
            },
        );

    let scale_x = w_h.0 as f32 / (max_x - min_x);

    debug!("scale_x={scale_x:?} w_h={w_h:?} max_x={max_x} min_x={min_x}");

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
            .draw_series(LineSeries::new(x_y, color.clone()))?
            .label(name)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.clone()));
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    // if let Some(lanes) = lanes {
    //     for lane in lanes {
    //         let motif_area = |x_0: i32, x_1: i32| {
    //             return EmptyElement::at((x_0 as f32 * scale_x, 0.))
    //                 + Rectangle::new(
    //                 [
    //                      (0, 0),
    //                      (((x_1-x_0) as f32 * scale_x) as i32, w_h.1 as i32 )
    //                  ],
    //                 &BLUE)
    //                 + Text::new(format!("({:.2},{:.2})", x_0, x_1), (5, 5), ("sans-serif", 15.0).into_font(),
    //             );
    //         };
    //         chart.plotting_area().draw(&motif_area(lane.0, lane.1)).expect("Drawing Failure");
    //     }
    // }



    // let motif_area = |x_0: i32, x_1: i32| {
    //     return EmptyElement::at((1000., 0.))
    //         + Rectangle::new(
    //         [
    //             (0, 0),
    //             (1000, w_h.1 as i32 )
    //         ],
    //         &BLUE)
    //         + Text::new(format!("({:.2},{:.2})", x_0, x_1), (5, 5), ("sans-serif", 15.0).into_font(),
    //     );
    // };
    // chart.plotting_area().draw(&motif_area(1000, 2000)).expect("Drawing Failure");




    if let Some(points) = points {
        for point in points {
            root.draw(&Circle::new(
                chart.backend_coord(point),
                10,
                ShapeStyle {
                    color: BLUE.mix(0.6),
                    filled: false,
                    stroke_width: 2,
                },
            ))?;
        }
    }

    root.present()?;

    Ok(())
}

pub fn draw_function_plot(
    output_path: &impl AsRef<std::path::Path>,
    w_h: (u32, u32),
    xy: &[(f32, f32)],
    bands: &[(f32, f32)],
    title: &str,
) -> Result<(), ThoError> {
    let (width, height) = w_h;

    // ----- Determine ranges (include bands in X range) -----
    let mut xs = xy.iter().map(|(x, _)| *x).collect::<Vec<_>>();
    xs.extend(bands.iter().flat_map(|(a, b)| [*a, *b]));

    let (mut xmin, mut xmax) = match (xs.iter().cloned().fold(f32::INFINITY, f32::min),
                                      xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max)) {
        (a, b) if a.is_finite() && b.is_finite() && a < b => (a, b),
        _ => (0.0, 1.0),
    };

    let (mut ymin, mut ymax) = match (
        xy.iter().map(|&(_, y)| y).fold(f32::INFINITY, f32::min),
        xy.iter().map(|&(_, y)| y).fold(f32::NEG_INFINITY, f32::max),
    ) {
        (a, b) if a.is_finite() && b.is_finite() && a < b => (a, b),
        _ => (-1.0, 1.0),
    };

    // Add a little padding so the plot isn't glued to edges
    let pad_x = ((xmax - xmin).abs()).max(1e-6) * 0.05;
    let pad_y = ((ymax - ymin).abs()).max(1e-6) * 0.10;
    xmin -= pad_x; xmax += pad_x;
    ymin -= pad_y; ymax += pad_y;

    // ----- Backend & chart -----
    let root = BitMapBackend::new(output_path.as_ref(), (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(35)
        .y_label_area_size(45)
        .build_cartesian_2d(xmin..xmax, ymin..ymax)?;

    chart.configure_mesh()
        .x_labels(20)
        .y_labels(10)
        .light_line_style(&TRANSPARENT) // grid only on major lines
        .draw()?;

    // ----- Draw vertical highlight bands behind the line -----
    let plot_area = chart.plotting_area();
    for &(x0, x1) in bands {
        let (a, b) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
        // Slightly translucent orange fill
        plot_area.draw(&Rectangle::new(
            [(a, ymin), (b, ymax)],
            RGBAColor(255, 165, 0, 0.20).filled(),
        ))?;
    }

    // ----- Draw the function line -----
    // Filter out NaNs just in case.
    let clean = xy.iter().copied().filter(|(x, y)| x.is_finite() && y.is_finite());
    chart
        .draw_series(LineSeries::new(clean, &BLUE))?
        .label("f(x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}


pub fn plot_3d() { //TODO
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

pub fn to_vec_f32(t: &Tensor) -> Vec<f32> {
    from::<Vec<f32>>(&t.reshape(-1))
}

pub fn to_vec_f64(t: &Tensor) -> Vec<f64> {
    from::<Vec<f64>>(&t.reshape(-1))
}

pub fn to_vec_i64(t: &Tensor) -> Vec<i64> {
    from::<Vec<i64>>(&t.reshape(-1))
}

pub fn pad_to(input: &Tensor, to_size: &[i64], pad_with: f64) -> Result<Tensor, ThoError> {
    debug!("pad_to: input.size={:?}, to_size{:?}",input.size(),to_size);

    if input.size().len() != to_size.len() {
        ThoError::new(format!(
            "dimensions don't match: {:?} {:?}",
            input.size(),
            to_size
        ));
    }
    let mut to_size = to_size.to_vec();

    let mut input_size = input.size();
    input_size.reverse();
    to_size.reverse();

    let pad_idx = input_size
        .into_iter()
        .zip(to_size.into_iter())
        .enumerate()
        .flat_map(|(i, (from, to))| {
            debug!("pad_to: i={i} from={from} to={to}");
            if from > to {
                error!("from {from} > to {to} at {i}"); //panic doesn't show up in test
                panic!("from {from} > to {to} at {i}");
            } else {
                [0, to - from]
            }
        })
        .collect::<Vec<i64>>();

    input
        .f_pad(pad_idx.as_slice(), "constant", pad_with)
        .map_err(ThoError::from)
}

pub fn pad_to_2(input: &Tensor, to_size: &[i64], truncate: bool, with_mask: bool) -> Result<(Tensor, Option<Tensor>), ThoError> {
    debug!("pad_to: input.size={:?}, to_size={:?}", input.size(), to_size);

    if input.size().len() != to_size.len() {
        return Err(ThoError::new(format!(
            "dimensions don't match: {:?} vs {:?}",
            input.size(),
            to_size
        )));
    }

    let mut result = input.shallow_clone();
    let mut cur_size = result.size();

    if truncate {
        // for each dimension where from > to, keep only the first `to` elements.
        for (dim, (&from, &to)) in cur_size.iter().zip(to_size.iter()).enumerate() {
            if from > to {
                debug!("pad_to: truncating dim {dim} from {from} to {to}");
                result = result.f_narrow(dim as i64, 0, to)?;
            }
        }
        cur_size = result.size();
    } else {
        // in pure pad mode treat from > to as an error.
        for (dim, (&from, &to)) in cur_size.iter().zip(to_size.iter()).enumerate() {
            if from > to {
                return Err(ThoError::new(format!(
                    "from {from} > to {to} at dim {dim}"
                )));
            }
        }
    }

    // build pad spec: last dimension first, [left, right, left, right, ...].
    let mut from_rev = cur_size.clone();
    from_rev.reverse();
    let mut to_rev = to_size.to_vec();
    to_rev.reverse();

    let mut pad_idx = Vec::with_capacity(from_rev.len() * 2);
    for (i, (&from, &to)) in from_rev.iter().zip(to_rev.iter()).enumerate() {
        if from > to {
            // should not happen if truncate == true and the above logic ran.
            return Err(ThoError::new(format!(
                "after truncation: from {from} > to {to} at reversed dim {i}"
            )));
        }
        pad_idx.push(0);          // pad on the "left"
        pad_idx.push(to - from);  // pad on the "right"
    }

    let padded = result.f_pad(pad_idx.as_slice(), "constant", 0.0)?;
    let mask = if with_mask {
        let base_mask = Tensor::ones(cur_size.as_slice(), (Kind::Uint8, result.device()));
        let padded_mask = base_mask.f_pad(pad_idx.as_slice(), "constant", 0.0)?;
        Some(padded_mask.to_kind(Kind::Bool))
    } else {
        None
    };

    Ok((padded, mask))
}


pub fn clone(from: &Tensor) -> Tensor {
    let to = Tensor::zeros(from.size(), FLOAT_CPU); //TODO clone
    let _ = from.clone(&to);

    to
}

pub fn hist_1_dim(tensor: &Tensor, bins: u32) -> Tensor {
    assert_eq!(
        tensor.dim(),
        2,
        "hist_1_dim: input tensor must have exactly two dimensions."
    );

    let min_vals = tensor.min_dim(1, true).0; //channels x 1
    let max_vals = tensor.max_dim(1, true).0; //channels x 1

    let norm = (tensor - &min_vals) / &(&max_vals - &min_vals).clamp_min(1e-8);

    trace!("norm={norm}");

    let bin_indices = (norm * bins as f64)
        .floor()
        .clamp(0.0, (bins - 1) as f64)
        .to_kind(Kind::Int64); // channels x dists

    trace!("bin_indices={bin_indices}");

    let one_hot = bin_indices.one_hot(bins as i64);
    trace!("one_hot={one_hot}");

    one_hot.sum_dim_intlist(1, false, Kind::Int)
}

//TODO replace with same from cuda_utils free_mem, total_mem
pub fn cuda_mem() -> (usize, usize) {
    use cudarc::driver::CudaContext;
    use cust::sys::cuMemGetInfo_v2;

    let _ctx = CudaContext::new(0).unwrap();
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        let _res = cuMemGetInfo_v2(&mut free, &mut total);
        // assert_eq!(res, sys::CUresult::CUDA_SUCCESS);
    }
    (free, total)
}


pub fn kick_register_cuda_hooks() {
    unsafe {
        //TODO to kick REGISTER_CUDA_HOOKS(CUDAHooks); in
        // ATen   cuda
        //     │ │ │   detail
        //     │ │ │ │ │ 󰙲 CUDAHooks.cpp
        let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        // let func: libloading::Symbol<unsafe extern fn() -> u32> = lib.get(b"my_func")?;
        // Ok(func())
        println!("lib={lib:?}"); //logging is not set up yet
        let t = Tensor::rand(1_000_000, FLOAT_CUDA);
        let _buf = Vec::<f32>::try_from(&t).unwrap();
    };
}

pub struct DummyListener;

impl TrainListener for DummyListener {
    fn on_valid_cycle(&mut self, _cycle: u32, _loss: f32) -> ListenerResponse {
        ListenerResponse::Nope
    }
}

/// x:    [B, C, T] float
/// mask: [B, 1, T] bool (true = keep, false = padding)
/// out:  [B, C]
pub fn masked_mean(x: &Tensor, mask: &Tensor) -> Result<Tensor, ThoError> {
    // float mask in {0,1}
    let mask_f = mask.f_to_kind(Kind::Float)?; // [B,1,T]

    // sum(x * mask) over T
    let xm = x.f_mul(&mask_f)?; // [B,C,T] (broadcast over C)
    let sum = xm.f_sum_dim_intlist(2, false, Kind::Float)?; // [B,C]

    // denom = count(valid) over T, clamp to avoid div-by-zero
    let denom = mask_f
        .f_sum_dim_intlist(2, false, Kind::Float)? // [B,1]
        .f_clamp_min(1.0)?;                           // [B,1]

    // broadcast denom over C
    let mean = sum.f_div(&denom)?; // [B,C]
    Ok(mean)
}

/// x:    [B, C, T] float
/// mask: [B, 1, T] bool
/// out:  [B, C]
///
/// Uses a large negative fill for masked timesteps (good if x is e.g. probabilities).
pub fn masked_max(x: &Tensor, mask: &Tensor) -> Result<Tensor, ThoError> {
    let mask_f = mask.f_to_kind(Kind::Float)?; // [B,1,T]
    let neg: f64 = -1e9;

    // x_keep = x * mask
    let x_keep = x.f_mul(&mask_f)?; // [B,C,T]

    // inv = (1 - mask) * neg
    let ones = mask_f.f_ones_like()?;                 // [B,1,T]
    let inv = ones.f_subtract(&mask_f)?;              // [B,1,T]
    let inv_scaled = inv.f_multiply_scalar(neg)?;     // [B,1,T]

    // x_masked = x_keep + inv_scaled  (broadcast over C)
    let x_masked = x_keep.f_add(&inv_scaled)?;        // [B,C,T]

    let (mx, _idx) = x_masked.f_max_dim(2, false)?;   // mx: [B,C]
    Ok(mx)
}

pub fn mask_from_lengths(lengths: &[i64], max_len: i64, device: Device) -> Result<Tensor, ThoError> {
    if lengths.is_empty() {
        return Err(ThoError::new("mask_from_lengths: empty lengths"));
    }
    if max_len < 0 {
        return Err(ThoError::new(format!("mask_from_lengths: max_len must be >= 0, got {max_len}")));
    }
    let lengths_t = Tensor::f_from_slice(lengths)?.f_to_device(device)?.f_view([lengths.len() as i64, 1, 1])?;
    let range = Tensor::f_arange(max_len, (Kind::Int64, device))?.f_view([1, 1, max_len])?;
    Ok(range.f_lt_tensor(&lengths_t)?.to_kind(Kind::Bool))
}


pub fn z_norm(t: &Tensor, dim: i64) -> Tensor {
    let mean_t = t.mean_dim(dim, true, Float);
    let std_t = t.std(false);
    let z_t = (t - mean_t) / std_t;
    z_t
}

pub fn z_norm_2(t_0: &Tensor, _dim: i64) -> Result<Tensor, ThoError> {
    let std = t_0.f_std_dim(1, true, true)?;
    //TODO std - 0
    Ok(t_0.f_sub(&t_0.f_mean_dim(1, true, Float)?)?.f_div(&std)?)
}

pub fn z_norm_dims(x: &Tensor, dims: &[i64], eps: f64) -> Result<Tensor, ThoError> {
    // Mean over dims (keepdim=true so it broadcasts back)
    let mean = x.f_mean_dim(dims, /*keepdim=*/true, Kind::Float)?;

    // Variance over dims (unbiased=false = "population" variance, typical for normalization)
    let var = x.f_var_dim(dims, /*unbiased=*/false, /*keepdim=*/true)?;

    // std = sqrt(var + eps)
    // Prefer f_add_scalar if available in your tch version.
    let std = var.f_add_scalar(eps)?.f_sqrt()?;

    // (x - mean) / std
    let centered = x.f_subtract(&mean)?;
    Ok(centered.f_divide(&std)?)
}

pub fn skewness_kurtosis_1_dim(dists: &Tensor) -> (Tensor, Tensor) {
    let mean = dists.mean_dim(1, true, Kind::Float);

    let std = dists.std_dim(1, true, true);

    let std = &std + 1e-6;

    //z score = (x - mean) / std
    let z = (dists - &mean) / &std;

    // Compute skewness: skewness = mean(z^3)
    let skewness = z.pow_tensor_scalar(3).mean_dim(1, false, Float);

    // Compute kurtosis: kurtosis = mean(z^4) - 3
    let kurtosis = z.pow_tensor_scalar(4).mean_dim(1, false, Float) - 3.0;

    (skewness, kurtosis)
}

pub fn dot_norm(dim: i64, slice_01: &Tensor, slice_02: &Tensor) -> Result<Tensor, ThoError> {
    let norm_01 = norm_2(slice_01, dim)?;
    let norm_02 = norm_2(slice_02, dim)?;
    Ok(norm_01.f_tensordot(&norm_02, [dim], [dim])?.f_diag(0)?)
}

pub fn self_dot(
    t1: &Tensor,
    i_0: i64,
    i_1: i64,
    dim: i64,
    step: i64,
    d_i: i64,
) -> Result<Tensor, ThoError> {
    let slice_1 = t1.f_slice(dim, Some(i_0), Some(i_0 + d_i), step)?;
    let slice_2 = t1.f_slice(dim, Some(i_1), Some(i_1 + d_i), step)?;

    Ok(slice_1.f_tensordot(&slice_2, [dim], [dim])?.f_diag(0)?)
}

pub fn norm(t: &Tensor, dim: &[i64]) -> Tensor {
    t.view(dim) / t.dot(&t).sqrt()
}

pub fn norm_2(t: &Tensor, dim: i64) -> Result<Tensor, ThoError> {
    let denom = t
        .f_tensordot(&t, dim, dim)?
        .f_diag(0)?
        .f_sqrt()?
        .f_view([1, -1])?
        .f_transpose(0, 1)?;
    Ok(t.f_div(&denom)?)
}

pub fn lin_t(n: u32) -> Tensor {
    // Tensor::from(lin_iter(n).collect::<Vec<f32>>().as_slice())
    Tensor::arange(n as i64, FLOAT_CPU)
}

pub fn lin_for_size(k: f64, size: [i64; 2], kind: Kind, device: &Device) -> Tensor {
    let end = size.iter().fold(1, |a, v| a * *v);

    debug!("lin_for_size: end={end}");

    let mut t = panic_to_err!(Tensor::f_arange(end, (kind, *device)))
        .expect("f_arange failed")
        .view(&size[..]);

    t = t.set_requires_grad(false);
    match kind {
        Kind::Uint8 | Kind::Int8 | Kind::Int16 | Kind::Int | Kind::Int64 => {
            t *= k as i32;
        }
        _ => {
            t *= k;
        }
    }

    trace!("lin_for_size:t=\n{t}");
    debug!("lin_for_size: t.size()={:?}", t.size());

    t
}

pub fn lin_iter(n: u32) -> Map<Range<u32>, fn(u32) -> f32> {
    (0..n).map(|x| x as f32)
}

use crate::dims::ThoBatch;
use crate::train_nets::{MotifNetParams, ThoModuleWithStore};

//TODO remove?
pub trait DataBatch<D: Dims> {
    /// returns (inputs, labels)
    fn train_batch(&self) -> (ThoBatch<D>, Tensor);
    fn valid_batch(&self) -> (ThoBatch<D>, Tensor);
}


pub trait TrainListener: Send + Sync {
    fn on_cycle(&mut self, _cycle: usize, _loss: f32) -> BoxFuture<'_, ListenerResponse> {
        async {ListenerResponse::Nope}.boxed()
    }
    fn on_valid_cycle(&mut self, cycle: u32, loss: f32) -> ListenerResponse;
}

pub enum ListenerResponse {
    Stop,
    UpdateLR(f64),
    Nope,
}


pub fn two_sigs(n: u32) -> (Tensor, Tensor) {
    let sig_1 = Tensor::from_slice(
        pulses_in_const(
            n,
            &[
                // (10, pulse_v(8., 3., 20, 2.)),
                // (30, pulse_v(8., 3., 20, 2.))
                (100, pulse_v(80., 30., 200, 20.)),
                (300, pulse_v(80., 30., 200, 20.)),
            ],
        )
            .as_slice(),
    );

    let sig_2 = Tensor::from_slice(pulse_in_rand(n, &[], 10., 0.001, 0.1).as_slice());
    (sig_1, sig_2)
}

//f(x)=-sin(3x)+x^2+0.7 *x
pub fn sin3_pow2(
    n: i64,
    min: f32,
    max: f32,
    with_noise: Option<f32>,
) -> (Tensor, Tensor, Vec<f32>) {
    let mut rng = rand::rng();
    let x = (0..n)
        .map(|i| min + (i as f32 / (n - 1) as f32) * (max - min))
        .into_iter()
        .collect::<Vec<f32>>();

    let y_train_full = Tensor::from_slice(
        x.iter()
            .map(|x| sin3_pow2_with_noise_f(with_noise, &mut rng, *x))
            .collect::<Vec<f32>>()
            .as_slice(),
    )
        .view((1, -1))
        .totype(Float); //.transpose(0,1);

    let x_t = Tensor::from_slice(x.as_slice()).totype(Float).view((-1, 1));

    (y_train_full, x_t, x)
}

fn sin3_pow2_with_noise_f(with_noise: Option<f32>, rng: &mut ThreadRng, x: f32) -> f32 {
    sin3_pow2_f(x) + with_noise.map(|v| v * rng.random::<f32>()).unwrap_or(0.)
}

pub fn sin3_pow2_f(x: f32) -> f32 {
    -(3. * x).sin() + x.powi(2) + 0.7 * x
}

pub fn sin3_pow2_t(t: &Tensor) -> Tensor {
    t.multiply_scalar(-3.).sin() + t.pow_tensor_scalar(2) + t.multiply_scalar(0.7) //TODO noise?
}

pub fn gauss_t(μ: f32, σ: f32, n: i64) -> Tensor {
    Tensor::from(gauss_i(μ, σ, n).collect::<Vec<_>>().as_slice())
}

pub fn pulse_t(μ: f32, σ: f32, n: i64, f: f32) -> Tensor {
    Tensor::from(pulse_v(μ, σ, n, f).as_slice())
}

pub fn three_sigs_tensor(
    n: u32,
    sig_0_v: &Vec<f32>,
    sig_1_v: &Vec<f32>,
    sig_2_v: &Vec<f32>,
) -> (Tensor, Tensor, Tensor, Tensor) {
    let t_0_0 = t_0_0(n, sig_0_v, sig_2_v);

    // debug!("t_0_0 mean={} max-min={}", t_0_0.mean(None).double_value(&[]), t_0_0.max() - t_0_0.min());

    let t_0_1 = t_0_1(n, sig_0_v, sig_2_v);

    // debug!("t_0_1 mean={} max-min={}", t_0_1.mean(None).double_value(&[]), t_0_1.max() - t_0_1.min());

    let t_1_0 = t_1_0(n, sig_1_v, sig_2_v);

    // debug!("t_1_0 mean={} max-min={}", t_1_0.mean(None).double_value(&[]), t_1_0.max() - t_1_0.min());

    let t_1_1 = t_1_1(n, sig_1_v, sig_2_v);
    (t_0_0, t_0_1, t_1_0, t_1_1)
}

pub fn t_0_0(n: u32, sig_0_v: &Vec<f32>, sig_2_v: &Vec<f32>) -> Tensor {
    Tensor::vstack(&[
        Tensor::from_slice([sig_0_v.clone(), saw(n), saw(n)].concat().as_slice()),
        Tensor::from_slice([saw(n), saw(n), saw(n)].concat().as_slice()),
        Tensor::from_slice([saw(n), sig_2_v.clone(), saw(n)].concat().as_slice()),
    ])
}

pub fn t_0_1(n: u32, sig_0_v: &Vec<f32>, sig_2_v: &Vec<f32>) -> Tensor {
    Tensor::vstack(&[
        Tensor::from_slice([saw(n), sig_0_v.clone(), saw(n)].concat().as_slice()),
        Tensor::from_slice([saw(n), saw(n), saw(n)].concat().as_slice()),
        Tensor::from_slice([sig_2_v.clone(), saw(n), saw(n)].concat().as_slice()),
    ])
}

pub fn t_1_0(n: u32, sig_1_v: &Vec<f32>, sig_2_v: &Vec<f32>) -> Tensor {
    Tensor::vstack(&[
        Tensor::from_slice([saw(n), saw(n), saw(n)].concat().as_slice()),
        Tensor::from_slice([saw(n), sig_1_v.clone(), saw(n)].concat().as_slice()),
        Tensor::from_slice([saw(n), sig_2_v.clone(), saw(n)].concat().as_slice()),
    ])
}

pub fn t_1_1(n: u32, sig_1_v: &Vec<f32>, sig_2_v: &Vec<f32>) -> Tensor {
    Tensor::vstack(&[
        Tensor::from_slice([saw(n), saw(n), saw(n)].concat().as_slice()),
        Tensor::from_slice([sig_1_v.clone(), saw(n), saw(n)].concat().as_slice()),
        Tensor::from_slice([sig_2_v.clone(), saw(n), saw(n)].concat().as_slice()),
    ])
}

pub fn reset_print_options() {
    let mut printer_options = PrinterOptions::default();

    unsafe {
        let printer_options_ptr: *mut PrinterOptions = &mut printer_options;

        let threshold_ptr =
            (printer_options_ptr as *mut u8).add(size_of::<usize>() * 1) as *mut usize;
        let line_width_ptr =
            (printer_options_ptr as *mut u8).add(size_of::<usize>() * 3) as *mut usize;

        ptr::write(threshold_ptr, usize::MAX);
        ptr::write(line_width_ptr, 10_000);
    }
    tch::display::set_print_options(printer_options);
}

pub fn tensor_mem_size(t: &Tensor) -> usize {
    t.numel() * t.kind().elt_size_in_bytes()
}

pub fn vs_mem_size(opt: &VarStore) -> usize {
    opt.trainable_variables()
        .iter()
        .map(|t| tensor_mem_size(t))
        .sum()
}

pub fn optimizer_mem_size(opt: &Optimizer) -> usize {
    opt.trainable_variables()
        .iter()
        .map(|t| tensor_mem_size(t))
        .sum()
}

pub fn set_requires_grad_in_subpath(vsd: &VarStore, subpath: &str, requires_grad: bool) {
    for (name, tensor) in vsd.variables() {
        if name.contains("batch_norm") {
            continue;
        }
        let freezing = name.starts_with(subpath);
        debug!("freeze_subpath: name={name:?} freezing={freezing}");

        if freezing {
            _ = tensor.set_requires_grad(requires_grad);
        }
    }
}

//TODO check GPT
pub fn mix_x_binary_with_labels(x_1: &Tensor, lengths_1: Option<&[i64]>, x_0: &Tensor, lengths_0: Option<&[i64]>) -> Result<(Tensor, Option<Vec<i64>>, Tensor), ThoError> {
    let [ma, _na, ..] = x_1.size()[0..2] else {panic!("todo")}; //.expect("X1 must be 2D [m, n]");
    let [mb, _nb, ..] = x_0.size()[0..2] else {panic!("todo")}; //.expect("X0 must be 2D [m, n]");
    // assert_eq!((na, nb), (na, na), "X1 and X0 must have same n");
    // assert_eq!(ma, mb, "X1 and X0 must have same m");
    assert_eq!(x_0.device(), x_1.device(), "A and B must be on the same device");

    let device = x_0.device();
    let two_m = ma * 2;

    // X0 = [A; B]
    let x0 = Tensor::f_cat(&[x_1, x_0], 0)?;

    // Choose a reasonable dtype for Y (float if A is floating, else Float)
    let y_kind = match x_0.kind() {
        Kind::Double | Kind::Float | Kind::Half | Kind::BFloat16 => x_0.kind(),
        _ => Kind::Float,
    };

    // Y0 = [ones(m,1); zeros(m,1)]
    let y0 = Tensor::f_cat(
        &[
            Tensor::f_ones(&[ma, 1], (y_kind, device))?,
            Tensor::f_zeros(&[mb, 1], (y_kind, device))?,
        ],
        0,
    )?;

    // Single random permutation applied to both X and Y
    let perm = Tensor::f_randperm(two_m, (Kind::Int64, device))?;
    let x = x0.f_index_select(0, &perm)?;
    let y = y0.f_index_select(0, &perm)?;

    let lengths = match (lengths_1, lengths_0) {
        (None, None) => None,
        (Some(lengths_1), Some(lengths_0)) => {
            if lengths_1.len() != ma as usize {
                return Err(ThoError::new(format!("lengths_1 len mismatch: expected={ma} got={}", lengths_1.len())));
            }
            if lengths_0.len() != mb as usize {
                return Err(ThoError::new(format!("lengths_0 len mismatch: expected={mb} got={}", lengths_0.len())));
            }
            let mut lengths = Vec::with_capacity((ma + mb) as usize);
            lengths.extend_from_slice(lengths_1);
            lengths.extend_from_slice(lengths_0);
            let perm_cpu = perm.to_device(Device::Cpu);
            let perm_vec: Vec<i64> = (&perm_cpu).try_into().map_err(|e| ThoError::from_error("mix_x_binary_with_labels: perm to vec", e))?;
            let mut lengths_perm = Vec::with_capacity(lengths.len());
            for idx in perm_vec {
                lengths_perm.push(lengths[idx as usize]);
            }
            Some(lengths_perm)
        }
        _ => return Err(ThoError::new("lengths_1 and lengths_0 must both be Some or None")),
    };

    Ok((x, lengths, y))
}


///sums and multiplies inputs by k
pub fn test_net<DI: Dims + Send + 'static, DO: Dims + Send + 'static>(
    num_features: u32,
    k: f64,
    device: Device,
    _descr: &str,
) -> Result<Box<ThoModuleWithStore<DI, DO>>, ThoError> {
    let vs = VarStore::new(device);

    let builder = Arc::new(move |path: &nn::Path| -> Result<Box<dyn crate::my_tch::ThoModule<DI, DO>>, ThoError> {
        SeqBuilder::new(num_features, None)?
            .linear_s_with_conf(
                "todo",
                1,
                LinearConfig {
                    ws_init: nn::Init::Const(k), //conflicts with ctor's init
                    bs_init: None,
                    bias: false,
                },
                #[cfg(debug_assertions)]
                "test",
            )?
            .end_seq()?
            .build_tch_module(path)
    });

    Ok(Box::new(ThoModuleWithStore::with_factory(
        "test_net",
        Arc::clone(&builder)(&vs.root())?,
        vs,
        Some(builder)
    )))

}


pub fn motif_test_net(
    motif_width: u32,
    k: f64,
    device: Device,
) -> (
    Box<dyn Fn() -> MotifNetParams + Sync + Send>,
    Box<dyn Fn() -> ThoModuleWithStore<Batch<Slice>, MotifProbability> + Sync + Send>,
) {
    (
        Box::new(move || MotifNetParams::new(motif_width, 1, 100, 32, 10)),
        Box::new(move || {
            *test_net(motif_width, k, device, "test_motif_net").unwrap()
        }),
    )
}

pub fn assert_close<A:Into<f64> + Display, B:Into<f64> + Display>(val_a: A, val_b: B, max_diff: f64) {
    let a = <A as Into<f64>>::into(val_a);
    let b = <B as Into<f64>>::into(val_b);
    let diff = (a - b).abs();
    assert!(
        diff <= max_diff,
        "values not close enough: {a} vs {b} (diff={diff}, max_diff={max_diff})"
    );
}

pub fn pad_stack_with_mask<T:AsRef<Tensor>>(tensors: &[&T], channels: i64, max_input: i64) -> Result<(Tensor, Vec<i64>), ThoError> {
    let mut stack = Vec::with_capacity(tensors.len());
    let mut lengths = Vec::with_capacity(tensors.len());

    for v in tensors {
        let t = v.as_ref();
        let size = t.size();
        if size.len() != 2 {
            return Err(ThoError::new(format!("pad_stack_with_mask expects 2D tensors, got size={size:?}")));
        }
        if size[0] != channels {
            return Err(ThoError::new(format!("pad_stack_with_mask channels mismatch: expected={channels} got={}", size[0])));
        }

        let mut t = t.shallow_clone();
        let len = size[1].min(max_input);
        if size[1] > max_input {
            t = t.f_narrow(1, 0, max_input)?;
        }
        let pad_right = max_input - len;
        if pad_right > 0 {
            t = t.f_pad(&[0, pad_right, 0, 0], "constant", 0.0)?;
        }
        stack.push(t);
        lengths.push(len);
    }

    let stack = Tensor::f_stack(stack.as_slice(), 0)?;

    Ok((stack, lengths))
}

/// 1D grayscale morphology using max-pool (GPU friendly).
/// Expected input shape: [B, C, L] (batch, channels, length).
///
/// Note:
/// - Prefer odd `k` so padding = k/2 keeps length unchanged and centered.
/// - These `f_*` versions propagate CUDA/OOM/ATen errors instead of panicking.
#[inline]
pub fn dilate1d(x: &Tensor, k: i64) -> Result<Tensor, TchError> {
    let pad = k / 2;
    x.f_max_pool1d(k, 1, pad, 1, false)
}

#[inline]
pub fn erode1d(x: &Tensor, k: i64) -> Result<Tensor, TchError> {
    let pad = k / 2;
    // erosion(x) = -dilation(-x)
    let nx = x.f_neg()?;
    let pooled = nx.f_max_pool1d(k, 1, pad, 1, false)?;
    pooled.f_neg()
}

#[inline]
pub fn opening1d(x: &Tensor, k: i64) -> Result<Tensor, TchError> {
    // opening = erosion then dilation
    let e = erode1d(x, k)?;
    dilate1d(&e, k)
}

#[inline]
pub fn white_tophat1d(x: &Tensor, k: i64) -> Result<Tensor, TchError> {
    // white top-hat = x - opening(x)
    let opened = opening1d(x, k)?;
    x.f_sub(&opened)
}


/// Local-maximum mask along the last dimension (expects xs shape [B, C, L]).
///
/// A position is a local max if it equals the max over a sliding window of size `kernel`.
/// Note: flat plateaus produce multiple `true`s (all equal-to-max positions).
pub fn local_max_1d_mask(xs: &Tensor, kernel: i64) -> Result<Tensor, TchError> {
    if xs.dim() != 3 {
        return Err(TchError::Shape(format!(
            "f_local_max_1d_mask: expected xs dim=3 ([B,C,L]), got shape {:?}",
            xs.size()
        )));
    }
    if kernel <= 0 {
        return Err(TchError::Shape(format!(
            "f_local_max_1d_mask: kernel must be > 0, got {kernel}"
        )));
    }
    if kernel % 2 == 0 {
        return Err(TchError::Shape(format!(
            "f_local_max_1d_mask: kernel must be odd to preserve alignment, got {kernel}"
        )));
    }

    let pad = (kernel - 1) / 2;

    // max_pool1d with stride=1 + padding keeps length L.
    // Signature in tch is (kernel, stride, padding, dilation, ceil_mode) for 1D pooling.
    let pooled = xs.f_max_pool1d(kernel, 1, pad, 1, false)?;

    // Since max_pool returns values taken from xs, exact equality is typically safe here.
    let mask = xs.f_eq_tensor(&pooled)?;
    Ok(mask)
}

/// Returns a tensor with only local maxima kept (everything else becomes 0),
/// same shape and dtype as `xs`.
pub fn local_max_1d(xs: &Tensor, kernel: i64) -> Result<Tensor, TchError> {
    let mask = local_max_1d_mask(xs, kernel)?;
    let mask_f = mask.f_to_kind(xs.kind())?;
    xs.f_mul(&mask_f)
}

/// Global-max over time: [B,1,T’] -> [B,1]
pub fn pool_time_max(sims: &Tensor) -> Result<Tensor, TchError> {
    // dim=2 is time; keepdim=false to drop it
    let (pooled, _idx) = sims.f_max_dim(2, /*keepdim=*/false)?;
    Ok(pooled) // [B,1]
}

/// LogSumExp (soft-max) pooling with temperature tau: [B,1,T’] -> [B,1]
pub fn pool_time_lse(sims: &Tensor, tau: f64) -> Result<Tensor, TchError> {
    // Stable LSE: m + τ*log(Σ exp((x-m)/τ))
    let (m, _idx) = sims.f_max_dim(2, /*keepdim=*/true)?;           // [B,1,1]
    let x_centered = sims.f_sub(&m)?.f_div_scalar(tau)?;            // [B,1,T’]
    let ex = x_centered.f_exp()?;                                   // [B,1,T’]
    let s = ex.f_sum_dim_intlist(2, /*keepdim=*/false, Kind::Float)?; // [B,1]
    let lse = s.f_log()?.f_mul_scalar(tau)?;                         // [B,1]
    Ok(lse.f_add(&m.f_squeeze_dim(2)?)?)                             // [B,1]
}

/// Mean over time (simple baseline): [B,1,T’] -> [B,1]
pub fn pool_time_mean(sims: &Tensor) -> Result<Tensor, TchError> {
    let tprime = sims.size()[2] as f64;
    let sum = sims.f_sum_dim_intlist(2, /*keepdim=*/false, Kind::Float)?; // [B,1]
    sum.f_div_scalar(tprime)
}


#[cfg(test)]
#[allow(dead_code, unused_variables, unused_assignments, non_snake_case)]
pub mod tests {
    use crate::bool_iter::BoolTensorIterator;

    use crate::utils::{hist_1_dim, local_max_1d, local_max_1d_mask, masked_max, masked_mean, mix_x_binary_with_labels, pad_to_2, to_vec_f32, to_vec_i64, TrainListener};
    use clap::{Args, Parser, Subcommand};

    use log::debug;
    use std::collections::{HashMap, HashSet};

    use std::fmt::Debug;
    use serde::Deserialize;
    use tch::kind::FLOAT_CPU;
    use tch::nn::VarStore;
    use tch::Device::Cpu;
    use tch::Kind::Float;
    use tch::{Device, Kind, Tensor};
    use thoosa_properties_map::{FromHashmap, ToHashmap};
    use thoosa_properties_map_derive::ThoosaPropertiesMap;
    use thoosa_run_config_args::OptRunConfigSource;
    use thoosa_run_config_args_derive::ThoosaRunConfig;

    use thoosa_shared::hyperparameter::Hyperparameter;

    use thoosa_shared::val::Val::{Int, UInt};
    use thoosa_shared::val::{NamedVal, Val};
    use thoosa_shared::{cdt_for_dir, clc_step, RangeOrValue, ThoError};

    use thoosa_shared::run_configs::RunId;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    fn assert_allclose(a: &Tensor, b: &Tensor, atol: f64) {
        let ok = a.allclose(b, /*rtol=*/1e-5, /*atol=*/atol, /*equal_nan=*/false);
        assert!(
            ok,
            "tensors not close:\n  a={:?}\n  b={:?}",
            a,
            b
        );
    }

    #[test]
    fn test_masked_mean_and_max_basic() {
        // Shapes: B=2, C=2, T=4
        // x[b,c,:]
        // b0,c0: [1,2,3,4]   b0,c1: [10,20,30,40]
        // b1,c0: [5,6,7,8]   b1,c1: [50,60,70,80]
        let x = Tensor::from_slice(&[
            1f32, 2., 3., 4.,      10., 20., 30., 40.,
            5.,   6., 7., 8.,      50., 60., 70., 80.,
        ])
            .view([2i64, 2i64, 4i64])
            .to_kind(Kind::Float);

        // mask[b,1,:]
        // b0 valid first 3: [1,1,1,0]
        // b1 valid first 2: [1,1,0,0]
        let mask = Tensor::from_slice(&[
            1i64, 1, 1, 0,
            1,    1, 0, 0,
        ])
            .view([2i64, 1i64, 4i64])
            .to_kind(Kind::Bool);

        // Expected masked mean:
        // b0: c0=(1+2+3)/3=2,   c1=(10+20+30)/3=20
        // b1: c0=(5+6)/2=5.5,   c1=(50+60)/2=55
        let expected_mean = Tensor::from_slice(
            &[
                2f32, 20.,
                5.5, 55.
            ])
            .view([2i64, 2i64]);

        // Expected masked max:
        // b0: c0=max(1,2,3)=3,   c1=max(10,20,30)=30
        // b1: c0=max(5,6)=6,     c1=max(50,60)=60
        let expected_max = Tensor::from_slice(&[3f32, 30., 6., 60.])
            .view([2i64, 2i64]);

        let mean = masked_mean(&x, &mask).unwrap();
        let mx = masked_max(&x, &mask).unwrap();

        assert_allclose(&mean, &expected_mean, 1e-5);
        assert_allclose(&mx, &expected_max, 1e-5);
    }

    #[test]
    fn test_all_masked_behavior() {
        // If everything is masked:
        // - masked_mean_f returns 0 (because x*mask = 0 and denom is clamped to 1)
        // - masked_max_f returns neg fill value (-1e9) for every channel
        let x = Tensor::from_slice(&[
            1f32, 2., 3.,
            10., 20., 30.,
        ])
            .view([1i64, 2i64, 3i64])
            .to_kind(Kind::Float);

        let mask = Tensor::zeros([1i64, 1i64, 3i64], (Kind::Bool, x.device()));

        let mean = masked_mean(&x, &mask).unwrap();
        let mx = masked_max(&x, &mask).unwrap();

        let expected_mean = Tensor::zeros([1i64, 2i64], (Kind::Float, x.device()));
        let expected_max = Tensor::full([1i64, 2i64], -1e9f64, (Kind::Float, x.device()));

        assert_allclose(&mean, &expected_mean, 1e-5);
        assert_allclose(&mx, &expected_max, 1e-5);
    }

    #[test]
    fn test_clc_step() {
        let n = 10_000u32;
        let w = 1000;
        let t = Tensor::rand(&[1, n as i64], FLOAT_CPU);
        let slices_num = 100;
        let (step, pad) = clc_step(n, w, slices_num); //(n-w)/(k-1);
        let u = t
            .pad([0i64, pad as i64], "constant", 0.)
            .unfold(1, w as i64, step as i64);

        debug!("u={u:?}");

        assert_eq!(slices_num, u.size()[1] as u32);
    }


    #[test]
    fn test_cdt_for_dir() {
        debug!("{}", cdt_for_dir());
        assert_eq!(false, cdt_for_dir().contains(":"))
    }

    #[test]
    fn test_pad_to() -> Result<(), ThoError> {
        let t = Tensor::vstack(&[Tensor::from_slice(&[1., 2.]), Tensor::from_slice(&[3., 4.])]);
        let expected = Tensor::vstack(&[
            Tensor::from_slice(&[1., 2., 0.]),
            Tensor::from_slice(&[3., 4., 0.]),
            Tensor::from_slice(&[0., 0., 0.]),
        ]);
        let expected_mask = Tensor::from_slice(&[1u8, 1, 0, 1, 1, 0, 0, 0, 0]).view([3, 3]).to_kind(Kind::Bool);

        let (actual, mask) = super::pad_to_2(&t, &[3, 3], false, true)?;
        assert_eq!(expected, actual);
        assert_eq!(expected_mask, mask.expect("mask"));

        let (actual, mask) = super::pad_to_2(&t, &[3, 3], true, true)?;
        assert_eq!(expected, actual);
        assert_eq!(expected_mask, mask.expect("mask"));

        Ok(())
    }

    #[test]
    fn test_pad_to_truncate_and_pad() -> Result<(), ThoError> {
        let t = Tensor::vstack(&[
            Tensor::from_slice(&[1., 2., 3.]),
            Tensor::from_slice(&[4., 5., 6.]),
        ]);

        let expected = Tensor::vstack(&[
            Tensor::from_slice(&[1., 2.]),
            Tensor::from_slice(&[4., 5.]),
            Tensor::from_slice(&[0., 0.]),
        ]);

        // from shape [2, 3] to [3, 2] with truncation enabled
        let (actual, mask) = pad_to_2(&t, &[3, 2], true, true)?;
        let expected_mask = Tensor::from_slice(&[1u8, 1, 1, 1, 0, 0]).view([3, 2]).to_kind(Kind::Bool);

        assert_eq!(expected, actual);
        assert_eq!(expected_mask, mask.expect("mask"));
        Ok(())
    }

    #[test]
    fn test_pad_to_same() -> Result<(), ThoError> {
        let t = Tensor::vstack(&[Tensor::from_slice(&[1., 2.]), Tensor::from_slice(&[3., 4.])]);

        assert_eq!(
            Tensor::vstack(&[Tensor::from_slice(&[1., 2.]), Tensor::from_slice(&[3., 4.]),]),
            super::pad_to(&t, &[2, 2], 0.)?
        );

        Ok(())
    }

    #[test]
    fn test_pad_to_one_dim() -> Result<(), ThoError> {
        let t = Tensor::ones([1, 3], (Float, Cpu));

        assert_eq!(
            Tensor::vstack(&[
                Tensor::from_slice(&[1., 1., 1.]),
                Tensor::from_slice(&[0., 0., 0.]),
            ]),
            super::pad_to(&t, &[2, 3], 0.)?
        );

        Ok(())
    }
    
    #[test]
    fn test_devices() {
        let devices = vec!["dev1", "dev2"];
        let classes = [
            "seizure_vote",
            "lpd_vote",
            "gpd_vote",
            "lrda_vote",
            "grda_vote",
            "other_vote",
        ];

        let chunk_len = classes.len() / devices.len();
        let devices_classes = devices
            .into_iter()
            .cycle()
            .zip(classes.chunks(chunk_len).into_iter())
            .collect::<Vec<(&str, &[&str])>>();
        let mut processed_classes: HashSet<&str> = HashSet::new();
        devices_classes
            .into_iter()
            .for_each(|(device, classes_chunk)| {
                debug!("2. device={:?} class={:?}", device, classes_chunk);
                processed_classes.extend(classes_chunk)
            });

        assert_eq!(processed_classes.len(), 6);
        //TODO devices

        debug!("processed_classes={:?}", processed_classes);
    }

    #[test]
    fn test_run_config_args() {
        #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Debug)]
        struct SomeSubTrainConfig {
            #[in_map]
            #[run_config]
            lr: i32,
        }

        #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Debug)]
        struct CommonTrainConfig {
            #[in_map]
            #[run_config]
            beta: Option<i32>,

            #[in_map]
            #[run_config]
            sub_config: SomeSubTrainConfig,
        }

        #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Debug)]
        enum NetType {
            NetOne(NetOneConfig),
            NetTwo(NetTwoConfig),
        }

        #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Debug)]
        struct NetOneConfig {
            #[in_map]
            #[run_config]
            num_aaa: u32,
        }

        #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Debug)]
        struct NetTwoConfig {
            #[in_map]
            #[run_config]
            num_bbb: u32,
        }

        #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Debug)]
        #[run_config_opt]
        struct TrainLabelsRunConfig {
            #[in_map]
            pub id: Option<RunId>,

            #[in_map]
            #[run_config]
            some_prop: u32,

            #[in_map]
            #[run_config]
            common_config: CommonTrainConfig,

            #[in_map]
            #[run_config]
            net_type_0: NetType,

            #[in_map]
            #[run_config]
            net_type_1: Option<NetType>,
        }

        // impl RunConfig for TrainLabelsRunConfig {
        //     fn from_properties_map(run_id: RunId, properties: HashMap<String, Val>) -> Self {
        //         todo!()
        //     }
        //
        //     fn set_id(&mut self, id: RunId) {
        //         todo!()
        //     }
        //
        //     fn type_name() -> &'static str {
        //         todo!()
        //     }
        // }

        #[derive(Parser, Clone)]
        #[command(version, about, long_about = None)]
        struct TestCli {
            #[arg(long)]
            some_field: String,

            #[command(subcommand)]
            command: Commands,
        }

        #[derive(Debug, Subcommand, Clone)]
        enum Commands {
            StandardCommand { number: u32 },

            TrainLabels(TraneLabelsArgs),
        }

        #[derive(Debug, Args, Clone)]
        struct TraneLabelsArgs {
            #[arg(long)]
            resources: String,

            #[arg(long)]
            run_config_cnt: String,
        }

        let cli = TestCli::parse_from([
            "",
            "--some-field",
            "zzz",
            "train-labels",
            "--resources",
            "zzzz",
            "--run-config-cnt",
            r#"
                    {
                        some_prop:'1',
                        common_config:{
                            beta:'1..10',
                            sub_config:{
                                lr:'0..10:5,1'
                            }
                        },
                        net_type_0:{
                            NetOne:{num_aaa:'1'}
                        },
                        net_type_1:{
                            NetTwo:{num_bbb:'10..100:50,1'}
                        }
                    }
                "#,
        ]);

        #[derive(Debug, Default)]
        struct TestNodeParams {
            layers: u32,
        }

        fn test_train_label(
            vs: &VarStore,
            run_config: &TrainLabelsRunConfig,
            ls: &mut dyn TrainListener,
        ) -> Result<(f32, TestNodeParams), ThoError> {
            debug!("test_train_label: run_config={:?}", run_config);

            Ok((0.1, TestNodeParams { layers: 0 }))
        }

        let opt_sources = match &cli.command {
            Commands::TrainLabels(train_labels_args) => {
                debug!("Commands::TrainLabels: train_labels_args={train_labels_args:?}");

                let v: TrainLabelsRunConfigRangeOrValue =
                    serde_json5::from_str(&train_labels_args.run_config_cnt).unwrap();

                debug!("TrainLabelsRunConfigRangeOrValue: v={v:?}");

                assert_eq!(RangeOrValue::Value(1), v.some_prop);
                assert_eq!(
                    Some(RangeOrValue::Range {
                        range: 1..10,
                        step: None
                    }),
                    v.common_config.beta
                );
                assert_eq!(
                    RangeOrValue::Range {
                        range: 0..10,
                        step: Some((5, 1))
                    },
                    v.common_config.sub_config.lr
                );

                let NetTypeRangeOrValue::NetOne(NetOneConfigRangeOrValue {
                    num_aaa: ref num_aaa_range_or_value,
                }) = v.net_type_0
                else {
                    panic!("no NetOne/num_aaa")
                };
                assert_eq!(&RangeOrValue::Value(1), num_aaa_range_or_value);

                let Some(NetTypeRangeOrValue::NetTwo(NetTwoConfigRangeOrValue {
                    num_bbb: ref num_bbb_range_or_value,
                })) = v.net_type_1
                else {
                    panic!("no NetOne/num_bbb")
                };
                assert_eq!(
                    &RangeOrValue::Range {
                        range: 10..100,
                        step: Some((50, 1))
                    },
                    num_bbb_range_or_value
                );

                v.opt_run_configs().expect("TrainLabelsRunConfigRangeOrValue::opt_run_configs should not fail")
            }

            _ => {
                panic!("Subcommand not implemented");
            }
        };

        debug!(
            "const_params={:?}",
            opt_sources.grid_run_config_source().const_params()
        );
        assert_eq!(
            &vec![
                NamedVal::new(TrainLabelsRunConfigFields::SomeProp, UInt(1)),
                NamedVal::new(
                    TrainLabelsRunConfigFields::NetType0(NetTypeFields::NetOne(
                        NetOneConfigFields::NumAaa
                    )),
                    UInt(1)
                ),
                // NamedVal::new(TrainLabelsRunConfigFields::NetType1(NetTypeFields::NetTwo(NetTwoConfigFields::NumBbb)), UInt(2)),
            ],
            opt_sources.grid_run_config_source().const_params()
        );

        debug!(
            "       grid_to_vals={:?}",
            opt_sources.grid_run_config_source().grid_to_vals()
        );
        assert_eq!(
            (10..100)
                .step_by(50)
                .flat_map(|num_bbb| (0..10).step_by(5).map(move |lr| {
                    vec![
                        NamedVal::<TrainLabelsRunConfigFields>::new(
                            TrainLabelsRunConfigFields::CommonConfig(
                                CommonTrainConfigFields::SubConfig(SomeSubTrainConfigFields::Lr),
                            ),
                            Int(lr),
                        ),
                        NamedVal::<TrainLabelsRunConfigFields>::new(
                            TrainLabelsRunConfigFields::NetType1(NetTypeFields::NetTwo(
                                NetTwoConfigFields::NumBbb,
                            )),
                            UInt(num_bbb),
                        ),
                    ]
                }))
                .collect::<Vec<_>>(),
            opt_sources.grid_run_config_source().grid_to_vals()
        );

        debug!("opt_params={:?}", opt_sources.opt_params());
        assert_eq!(
            &vec![Hyperparameter::UniformInt {
                name: TrainLabelsRunConfigFields::CommonConfig(CommonTrainConfigFields::Beta),
                lower: 1,
                upper: 10
            }],
            opt_sources.opt_params()
        );

        // opt_train_2(test_train_label, &opt_sources, Cpu, "test").unwrap();
    }

    #[test]
    fn test_run_config_option() {
        #[derive(ThoosaRunConfig, Clone, PartialEq, Debug, Deserialize)]
        #[run_config_opt]
        pub struct TestRunConfig {
            pub id: Option<u32>,

            #[run_config]
            pub step: u32,

            #[run_config]
            pub window: u32,

            #[run_config]
            pub dist_quantile: f32,

            #[run_config]
            pub keep_slice_matches: bool,

            #[run_config]
            pub filter: Option<TestFilter>,

            pub descr: String,
        }

        #[derive(ThoosaRunConfig, Clone, PartialEq, Debug, Deserialize)]
        enum TestFilter {
            Filter1(Filter1RunConfig),
            Filter2(Filter2RunConfig),
        }

        #[derive(ThoosaRunConfig, Clone, PartialEq, Debug, Deserialize)]
        struct Filter1RunConfig {
            #[run_config]
            pub param_a: u32,
        }

        #[derive(ThoosaRunConfig, Clone, PartialEq, Debug, Deserialize)]
        struct Filter2RunConfig {
            #[run_config]
            pub param_b: u32,
        }

        let run_config_range_or_value: TestRunConfigRangeOrValue = serde_json5::from_str(
            r#" {
            step:'1',
            window:'10',
            keep_slice_matches: 'true',
            dist_quantile: '0.1',
            descr: 'test'
        } "#,
        )
        .unwrap();

        assert_eq!(
            TestRunConfig {
                id: None,
                step: 1,
                window: 10,
                dist_quantile: 0.1,
                keep_slice_matches: true,
                descr: "test".to_string(),
                filter: None
            },
            build_test_run_config(run_config_range_or_value)
        );

        let run_config_range_or_value: TestRunConfigRangeOrValue = serde_json5::from_str(
            r#" {
            id:111,
            step:'1',
            window:'10',
            keep_slice_matches: 'true',
            dist_quantile: '0.1',
            descr: 'test',
            filter: {
                Filter1: {
                    param_a: '1'
                }
            }
        } "#,
        )
        .unwrap();

        debug!("run_config_range_or_value={run_config_range_or_value:?}");
        assert_eq!(
            TestRunConfig {
                id: Some(111),
                step: 1,
                window: 10,
                dist_quantile: 0.1,
                keep_slice_matches: true,
                descr: "test".to_string(),
                filter: Some(TestFilter::Filter1(Filter1RunConfig { param_a: 1 })),
            },
            build_test_run_config(run_config_range_or_value)
        );

        fn build_test_run_config(run_config_range_or_value: TestRunConfigRangeOrValue) -> TestRunConfig {
            let mut const_vals = Vec::new();
            let mut grid_vals = Vec::new();
            let mut opt_params = Vec::new();
            run_config_range_or_value.fill_run_configs(|v| v, &mut const_vals, &mut grid_vals, &mut opt_params);

            let grid_vals: Vec<NamedVal<TestRunConfigFields>> = grid_vals.into_iter().flat_map(|range| range.into_iter()).collect();
            let opt_vals: Vec<NamedVal<TestRunConfigFields>> = opt_params.iter().map(|opt_param| opt_param.try_into().unwrap()).collect();

            [const_vals.as_slice(), grid_vals.as_slice(), opt_vals.as_slice()].concat().into_iter().into()
        }
    }

    #[test]
    fn test_build_from_named_vals() {
        #[derive(ThoosaRunConfig, Debug, Clone)]
        struct SubStruct0 {
            #[run_config]
            field_a: u32,
        }

        #[derive(ThoosaRunConfig, Debug, PartialEq, Clone)]
        struct SubStruct1 {
            #[run_config]
            field_b: u32,
        }

        #[derive(ThoosaRunConfig, Debug, Clone)]
        enum SubEnum {
            Sub0(SubStruct0),
            Sub1(SubStruct1),
        }

        #[derive(ThoosaRunConfig, Debug)]
        struct SubStructB {
            #[run_config]
            field_a: u32,
            #[run_config]
            field_b: u32,
        }

        #[derive(ThoosaRunConfig, Debug)]
        #[run_config_opt]
        struct SomeStruct {
            pub id: Option<RunId>,

            #[run_config]
            field_00: u32,

            #[run_config]
            field_01: f32,

            #[run_config]
            sub_struct: SubStructB,

            #[run_config]
            sub_struct_op: Option<SubStructB>,

            #[run_config]
            sub_enum: SubEnum,
        }

        let path_vals = [
            NamedVal::new(SomeStructFields::Field00, Int(1)),
            NamedVal::new(SomeStructFields::Field01, Val::Real(2.0)),
            NamedVal::new(
                SomeStructFields::SubStruct(SubStructBFields::FieldA),
                Int(3),
            ),
            NamedVal::new(
                SomeStructFields::SubStruct(SubStructBFields::FieldB),
                Int(4),
            ),
            NamedVal::new(
                SomeStructFields::SubStructOp(SubStructBFields::FieldA),
                Int(5),
            ),
            NamedVal::new(
                SomeStructFields::SubStructOp(SubStructBFields::FieldB),
                Int(6),
            ),
            NamedVal::new(
                SomeStructFields::SubEnum(SubEnumFields::Sub0(SubStruct0Fields::FieldA)),
                UInt(1),
            ),
        ];

        let some_struct: SomeStruct = path_vals.into_iter().into(); // SomeStructBuilder::add_path_vals(path_vals);

        debug!("some_struct={some_struct:?}");

        assert_eq!(some_struct.field_00, 1);
        assert_eq!(some_struct.field_01, 2.0);
        assert_eq!(some_struct.sub_struct.field_a, 3);
        assert_eq!(some_struct.sub_struct.field_b, 4);

        assert_eq!(
            some_struct.sub_struct_op.as_ref().map(|v| v.field_a),
            Some(5)
        );
        assert_eq!(
            some_struct.sub_struct_op.as_ref().map(|v| v.field_b),
            Some(6)
        );

        let SubEnum::Sub0(sub_struct_0) = some_struct.sub_enum else {
            panic!("some_struct.sub_enum should be Sub0");
        };
        assert_eq!(sub_struct_0.field_a, 1);

        let path_vals = [
            NamedVal::new(SomeStructFields::Field00, Val::Int(1)),
            NamedVal::new(SomeStructFields::Field01, Val::Real(2.0)),
            NamedVal::new(
                SomeStructFields::SubStruct(SubStructBFields::FieldA),
                Val::Int(3),
            ),
            NamedVal::new(
                SomeStructFields::SubStruct(SubStructBFields::FieldB),
                Val::Int(4),
            ),
            NamedVal::new(
                SomeStructFields::SubEnum(SubEnumFields::Sub1(SubStruct1Fields::FieldB)),
                Val::UInt(2),
            ),
        ];

        let some_struct: SomeStruct = path_vals.into_iter().into(); // SomeStructBuilder_::add_path_vals(path_vals);
        assert_eq!(some_struct.field_00, 1);
        assert_eq!(some_struct.field_01, 2.0);
        assert_eq!(some_struct.sub_struct.field_a, 3);
        assert_eq!(some_struct.sub_struct.field_b, 4);
        let SubEnum::Sub1(sub_struct_1) = some_struct.sub_enum else {
            panic!("some_struct.sub_enum should be Sub0");
        };
        assert_eq!(sub_struct_1.field_b, 2);
    }

    #[test]
    fn test_bool_tensor_iterator() {
        let t = Tensor::from_slice((0..100).map(|v| v % 2 == 0).collect::<Vec<_>>().as_slice());

        for (i, b) in (0..100).zip::<BoolTensorIterator>((&t).into()) {
            assert_eq!(i % 2 == 0, b);
        }
    }

    #[test]
    fn test_hist_1_dim() {
        // log::set_max_level(Trace);
        let t = Tensor::vstack(&[
            Tensor::from_slice(&[1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            Tensor::from_slice(&[1., 2., 4., 4., 5., 6., 6., 8., 9.]),
            Tensor::from_slice(&[1., 1., 1., 1., 1., 1., 1., 1., 1.]),
        ]);

        // debug!("t={t}");

        let hist = hist_1_dim(&t, 3);
        // debug!("hist={hist}");

        assert_eq!(
            Tensor::from_slice(&[3, 3, 3, 2, 5, 2, 9, 0, 0]).view((3, 3)),
            hist
        )
    }

    #[test]
    fn test_to_from_hashmap() {
        #[derive(ThoosaPropertiesMap, Debug, PartialEq)]
        struct TestTopStruct {
            #[in_map]
            field_a: u32,

            #[in_map]
            field_b: i32,

            #[in_map]
            field_o: Option<i32>,

            #[in_map]
            sub_struct: AnotherTestStruct,

            #[in_map]
            sub_struct_o: Option<AnotherTestStruct>,

            #[in_map]
            sub_enum: SomeEnum,

            #[in_map]
            sub_enum_o: Option<SomeEnum>,
        }

        #[derive(ThoosaPropertiesMap, Debug, PartialEq)]
        enum SomeEnum {
            VariantA(VariantA),
            VariantB(VariantB),
        }

        #[derive(ThoosaPropertiesMap, Debug, PartialEq)]
        struct VariantA {
            #[in_map]
            some_field: u32,
        }

        #[derive(ThoosaPropertiesMap, Debug, PartialEq)]
        struct VariantB {
            #[in_map]
            another_field: u32,
        }

        #[derive(ThoosaPropertiesMap, Debug, PartialEq)]
        struct AnotherTestStruct {
            #[in_map]
            field_c: u32,
            #[in_map]
            field_d: i32,
        }

        let run_config = TestTopStruct {
            field_a: 1,
            field_b: 2,
            field_o: Some(22),
            sub_struct: AnotherTestStruct {
                field_c: 3,
                field_d: 4,
            },
            sub_struct_o: Some(AnotherTestStruct {
                field_c: 30,
                field_d: 40,
            }),
            sub_enum: SomeEnum::VariantA(VariantA { some_field: 5 }),
            sub_enum_o: Some(SomeEnum::VariantB(VariantB { another_field: 50 })),
        };

        let props: HashMap<String, Val> = run_config.to_properties_iter(None).collect();
        debug!("props={props:?}");
        assert_eq!(
            run_config,
            TestTopStruct::from_properties_map(None, &props).unwrap()
        );

        let run_config = TestTopStruct {
            field_a: 1,
            field_b: 2,
            field_o: None,
            sub_struct: AnotherTestStruct {
                field_c: 3,
                field_d: 4,
            },
            sub_struct_o: None,
            sub_enum: SomeEnum::VariantA(VariantA { some_field: 5 }),
            sub_enum_o: None,
        };

        let props: HashMap<String, Val> = run_config.to_properties_iter(None).collect();

        assert_eq!(
            run_config,
            TestTopStruct::from_properties_map(None, &props).unwrap()
        );
    }

    #[test]
    fn mix_basic_shapes_and_labels() -> Result<(), ThoError> {
        let m = 4i64;
        let n = 3i64;

        // Build A with values [0..m*n), B with values [1000..1000+m*n)
        let va: Vec<f32> = (0..(m * n)).map(|x| x as f32).collect();
        let vb: Vec<f32> = (0..(m * n)).map(|x| 1000.0 + x as f32).collect();

        let a = Tensor::from_slice(&va).reshape(&[m, n]);
        let b = Tensor::from_slice(&vb).reshape(&[m, n]);

        let (x, _lengths, y) = mix_x_binary_with_labels(&a, None, &b, None)?;
        debug!("mix_basic_shapes_and_labels: x=\n{x} y=\n{y}");

        // Shapes
        assert_eq!(x.size(), [2 * m, n], "X must be [2m, n]");
        assert_eq!(y.size(), [2 * m, 1], "Y must be [2m, 1]");

        // Label counts: expect m ones and m zeros
        let y_i64 = y.to_kind(Kind::Int64);
        let ones = y_i64.sum(Kind::Int64).int64_value(&[]);
        assert_eq!(ones, m, "There must be exactly m ones in Y");
        let zeros = 2 * m - ones;
        assert_eq!(zeros, m, "There must be exactly m zeros in Y");

        // Row-by-row sanity: since A rows start < 1000 and B rows >= 1000,
        // the first element of each row tells us the source deterministically.
        let x_cpu = x.to_device(Device::Cpu);
        let y_cpu = y_i64.to_device(Device::Cpu);
        for i in 0..(2 * m) {
            let first_val = x_cpu.double_value(&[i, 0]);
            let label = y_cpu.int64_value(&[i, 0]);
            if first_val >= 1000.0 {
                assert_eq!(label, 0, "Row {i} came from B, expected label 0");
            } else {
                assert_eq!(label, 1, "Row {i} came from A, expected label 1");
            }
        }

        Ok(())
    }

    #[test]
    fn mix_masks_align_with_rows() -> Result<(), ThoError> {
        let a = Tensor::from_slice(&[1., 2., 3., 4., 5., 6.]).reshape(&[2, 3]);
        let b = Tensor::from_slice(&[101., 102., 103., 104., 105., 106.]).reshape(&[2, 3]);
        let lengths_a = vec![2, 3];
        let lengths_b = vec![1, 2];

        let (x, lengths, y) = mix_x_binary_with_labels(&a, Some(lengths_a.as_slice()), &b, Some(lengths_b.as_slice()))?;
        let lengths = lengths.expect("lengths");

        assert_eq!(x.size(), [4, 3], "X must be [2m, n]");
        assert_eq!(y.size(), [4, 1], "Y must be [2m, 1]");

        let x_cpu = x.to_device(Device::Cpu);
        let y_cpu = y.to_kind(Kind::Int64).to_device(Device::Cpu);

        for i in 0..4usize {
            let first_val = x_cpu.double_value(&[i as i64, 0]) as i64;
            let label = y_cpu.int64_value(&[i as i64, 0]);

            match first_val {
                1 => { assert_eq!(lengths[i], lengths_a[0]); assert_eq!(label, 1); }
                4 => { assert_eq!(lengths[i], lengths_a[1]); assert_eq!(label, 1); }
                101 => { assert_eq!(lengths[i], lengths_b[0]); assert_eq!(label, 0); }
                104 => { assert_eq!(lengths[i], lengths_b[1]); assert_eq!(label, 0); }
                _ => panic!("unexpected first_val={first_val}"),
            }
        }

        Ok(())
    }

    #[test]
    fn test_local_max_mask_basic() {
        // [0, 1, 0, 2, 2, 1, 0]
        // With kernel=3 => local maxima at 1 and both 2's (plateau).
        let x = Tensor::from_slice(&[0f32, 1., 0., 2., 2., 1., 0.]).view([1, 1, 7]);

        let m = local_max_1d_mask(&x, 3).unwrap();
        let m_i64 = m.to_kind(Kind::Int64).view([-1]);
        let v = to_vec_i64(&m_i64);
        assert_eq!(v, vec![0, 1, 0, 1, 1, 0, 0]);

        let peaks = local_max_1d(&x, 3).unwrap().view([-1]);
        let pv = to_vec_f32(&peaks);
        assert_eq!(pv, vec![0., 1., 0., 2., 2., 0., 0.]);
    }
}
