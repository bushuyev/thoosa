use std::cell::RefMut;
use std::collections::VecDeque;

use std::fmt::{Debug, Display};
use std::io::Write;
use std::iter::Map;

use std::ops::Range;
use std::time::Instant;
use std::{fs, ptr};

use log::{debug, error, info, trace, warn};
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::drawing::IntoDrawingArea;
use plotters::element::PathElement;
use plotters::prelude::*;
use rand::prelude::ThreadRng;
use rand::Rng;
use std::panic::AssertUnwindSafe;
use std::pin::pin;
use parking_lot::lock_api::{RawRwLock, RwLock};
use std::sync::Arc;
use tch::display::PrinterOptions;
use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
use tch::nn::{LinearConfig, Optimizer};
use tch::nn::VarStore;
use tch::Device::{Cpu, Cuda};
use tch::Kind::Float;
use tch::{nn, Device, Kind, TchError, Tensor};
use thoosa_shared::{gauss_i, panic_to_err, panic_to_err_async, pulse_in_rand, pulse_v, pulses_in_const, saw, MotifKind, ThoosaError};
use torch_sys::cuda::atc_cuda_device_count;
use thoosa_shared::traits::{Named, SimpleId};
use crate::my_tch::{LossFp, ThoModule};

use crate::batches::AsyncDataBatch;
use crate::builder::SeqBuilder;
use crate::dims::{AnyDims, Batch, Dims, MotifProbability, Slice};
use futures_util::{FutureExt, StreamExt};
use futures_util::future::BoxFuture;

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

pub fn plot_series<S>(
    name_x_y_vec: Vec<(String, S, Vec<(f32, f32)>)>,
    caption: &str,
    w_h: (u32, u32),
    path: &impl AsRef<std::path::Path>,
) -> Result<(), ThoosaError>
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
) -> Result<(), ThoosaError>
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
) -> Result<(), ThoosaError>
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
    lanes: Option<&[(i32, i32)]>,
) -> Result<(), ThoosaError>
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



    let motif_area = |x_0: i32, x_1: i32| {
        return EmptyElement::at((1000., 0.))
            + Rectangle::new(
            [
                (0, 0),
                (1000, w_h.1 as i32 )
            ],
            &BLUE)
            + Text::new(format!("({:.2},{:.2})", x_0, x_1), (5, 5), ("sans-serif", 15.0).into_font(),
        );
    };
    chart.plotting_area().draw(&motif_area(1000, 2000)).expect("Drawing Failure");




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
) -> Result<(), ThoosaError> {
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

pub fn vec_f32_from(t: &Tensor) -> Vec<f32> {
    from::<Vec<f32>>(&t.reshape(-1))
}

pub fn vec_f64_from(t: &Tensor) -> Vec<f64> {
    from::<Vec<f64>>(&t.reshape(-1))
}

pub fn pad_to(input: &Tensor, to_size: &[i64]) -> Result<Tensor, ThoosaError> {
    debug!("pad_to: input.size={:?}, to_size{:?}",input.size(),to_size);

    if input.size().len() != to_size.len() {
        ThoosaError::new(format!(
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

    input.f_pad(pad_idx.as_slice(), "constant", 0.).map_err(|e|ThoosaError::new(e.to_string()))
}

pub fn pad_to_2(input: &Tensor, to_size: &[i64], truncate: bool) -> Result<Tensor, ThoosaError> {
    debug!("pad_to: input.size={:?}, to_size={:?}", input.size(), to_size);

    if input.size().len() != to_size.len() {
        return Err(ThoosaError::new(format!(
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
                result = result.narrow(dim as i64, 0, to);
            }
        }
        cur_size = result.size();
    } else {
        // in pure pad mode treat from > to as an error.
        for (dim, (&from, &to)) in cur_size.iter().zip(to_size.iter()).enumerate() {
            if from > to {
                return Err(ThoosaError::new(format!(
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
            return Err(ThoosaError::new(format!(
                "after truncation: from {from} > to {to} at reversed dim {i}"
            )));
        }
        pad_idx.push(0);          // pad on the "left"
        pad_idx.push(to - from);  // pad on the "right"
    }

    Ok(result.pad(pad_idx.as_slice(), "constant", 0.0))
}

pub fn load_tensor(
    base_path: &str,
    tensor_id: &str,
    device: Device,
) -> Result<Tensor, ThoosaError> {
    Tensor::load(std::path::Path::new(base_path).join(format!("{}.pt", tensor_id)))
        .map(|t| t.to_device(device))
        .map_err(|e| ThoosaError::new(e.to_string()))
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

pub trait Splittable<const SIZE: usize> {
    fn split(&self) -> Option<[Self; SIZE]>
    where
        Self: Sized;
}

impl<R, S> Splittable<1> for RwLock<R, S>
where
    R: RawRwLock,
{
    fn split(&self) -> Option<[Self; 1]>
    where
        Self: Sized,
    {
        todo!()
    }
}

pub trait Splitter<const SIZE: usize> {
    type S: Splittable<SIZE>;
    fn next(&mut self) -> Option<Self::S>;
    fn split(&mut self, split: Self::S) -> Result<(), ThoosaError>;
}

#[derive(Debug)]
pub struct SplitQueue<const SIZE: usize, T> {
    queue: VecDeque<T>,
}

impl<const SIZE: usize, T> SplitQueue<SIZE, T>
where
    T: Debug + Splittable<SIZE>,
{
    pub fn new(chunk: T) -> Self {
        Self {
            queue: VecDeque::from([chunk]),
        }
    }
}

impl<const SIZE: usize, T> Splitter<SIZE> for SplitQueue<SIZE, T>
where
    T: Debug + Splittable<SIZE>,
{
    type S = T;

    fn next(&mut self) -> Option<T> {
        let v = self.queue.pop_front();

        debug!(
            "SplitQueue::next: self.queue.len={} pop_front={v:?}",
            self.queue.len()
        );

        v
    }

    fn split(&mut self, chunk: T) -> Result<(), ThoosaError> {
        debug!("SplitQueue::split: chunk={chunk:?}");

        let splits = chunk.split();
        if let Some(splits) = splits {
            for split in splits.into_iter().rev() {
                debug!("SplitQueue::split: push_front chunk={split:?}");

                self.queue.push_front(split);
            }
            Ok(())
        } else {
            Err(ThoosaError::new("Can't split anymore".to_owned()))
        }
    }
}

#[derive(Debug)]
pub struct Reducer<T> {
    value: Option<T>,
}

impl<T> Reducer<T> {
    pub fn new(value: T) -> Self {
        Self { value: Some(value) }
    }
}

impl<T> Splitter<1> for Reducer<T>
where
    T: Debug + Splittable<1>,
{
    type S = T;

    fn next(&mut self) -> Option<Self::S> {
        self.value.take()
    }

    fn split(&mut self, v: Self::S) -> Result<(), ThoosaError> {
        match v.split() {
            None => Err(ThoosaError::new("Can't split anymore".to_owned())),
            Some(a) => {
                let [v_0] = a;
                self.value = Some(v_0);

                Ok(())
            }
        }
    }
}

// impl <R, T> Splitter<1> for Reducer<RwLock<R, T>> where R:RawRwLock, T:Debug + Splittable<1> {
//     type S = RwLock<R, T>;
//
//     fn next(&mut self) -> Option<Self::S> {
//         self.value.take()
//     }
//
//     fn split(&mut self, v: Self::S) -> Result<(), ThoosaError> {
//         match v.split() {
//             None => {
//                 Err(ThoosaError::new("Can't split anymore".to_owned()))
//             }
//             Some(a) => {
//                 let [v_0] = a;
//                 self.value = Some(v_0);
//
//                 Ok(())
//             }
//         }
//     }
// }

pub async fn run_split_on_oom_2<
    's,
    'v,
    'fr,
    const SIZE: usize,
    S: 's,
    SV: 'v,
    F,
    FR: 'fr + Debug,
    OkF,
    OkR,
>(
    mut splitter: S,
    mut f: F,
    on_ok: OkF,
    max_attempts: i32,
) -> Result<(), ThoosaError>
where
    SV: Splittable<SIZE>,
    S: for<'x> Splitter<SIZE, S = SV>,
    //F: for <'f> AsyncFnMut(&'f mut V) -> Result<FR, ThoosaError>,
    // F: for<'a, 'b> AsyncFnMut(&'a mut (dyn Splittable<SIZE> + 'b)) -> Result<FR, ThoosaError>,
    F: AsyncFnMut(Arc<SV>) -> Result<FR, ThoosaError>,
    OkF: AsyncFn(FR, Arc<SV>) -> OkR,
{
    'out: while let Some(split_0) = splitter.next() {
        let split_0_arc = Arc::new(split_0);
        for attempt in 0..max_attempts {
            match panic_to_err_async!(f(Arc::clone(&split_0_arc)).await) {
                Ok(fr) => {
                    on_ok(fr, Arc::clone(&split_0_arc)).await;
                    continue 'out;
                }
                Err(e) if e.is_oom() => {
                    debug!(
                        "run_split_on_oom: splitting on oom, max_attempts={max_attempts} attempt={attempt}"
                    );
                }
                Err(e) => {
                    debug!("run_split_on_oom: failed on oom {e:?}");
                    return Err(e);
                }
            }
        }
        if let Err(e) = splitter.split(Arc::into_inner(split_0_arc).unwrap()) {
            return Err(e);
        }
    }

    Ok(())
}

pub fn run_split_on_oom<const SIZE: usize, F, FR, V, S, OkF, OkR>(
    mut splitter: S,
    mut f: F,
    mut on_ok: OkF,
    max_attempts: i32,
) -> Result<(), ThoosaError>
where
    F: FnMut(&mut V) -> Result<FR, ThoosaError>,
    S: Splitter<SIZE, S = V>,
    OkF: FnMut(FR, &mut V) -> OkR,
{
    'out: while let Some(mut split_0) = splitter.next() {
        for attempt in 0..max_attempts {
            match panic_to_err!(f(&mut split_0)) {
                Ok(fr) => {
                    on_ok(fr, &mut split_0);
                    continue 'out;
                }
                Err(e) if e.is_oom() => {
                    debug!(
                        "run_split_on_oom: splitting on oom, max_attempts={max_attempts} attempt={attempt}"
                    );
                }
                Err(e) => {
                    debug!("run_split_on_oom: failed on oom {e:?}");
                    return Err(e);
                }
            }
        }
        if let Err(e) = splitter.split(split_0) {
            return Err(e);
        }
    }

    Ok(())
}

#[derive(Debug)]
pub struct SplittableTensor<const D: i64>(pub Tensor); //TODO more than one dimension?

impl<const S: usize, const D: i64> Splittable<S> for SplittableTensor<D> {
    fn split(&self) -> Option<[Self; S]>
    where
        Self: Sized,
    {
        let split_size = self.0.size()[D as usize] / S as i64 + 1; //TODO if self.0.size()[D as usize] < 2

        debug!(
            "SplittableTensor::split: self.0.size()={:?} S={S} D={D} split_size={split_size}",
            self.0.size()
        );

        let splits = self
            .0
            .split(split_size, D)
            .into_iter()
            .map(|t| SplittableTensor::<D>(t))
            .collect::<Vec<_>>();

        debug!("SplittableTensor::split: splits={splits:?}");

        // let splits_len = splits.len();

        splits.try_into().ok()
    }
}

//TODO for batch + degrade Kind: Double->Float->BFloat16
pub struct CompositeReducer {
    _reducers: Vec<Box<dyn Splittable<1>>>,
}

impl CompositeReducer {
    pub fn new(_reducers: Vec<Box<dyn Splittable<1>>>) -> Self {
        Self { _reducers }
    }
}

impl<const D: i64> AsRef<Tensor> for SplittableTensor<D> {
    fn as_ref(&self) -> &Tensor {
        &self.0
    }
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
    fn on_valid_cycle(&mut self, cycle: u32, loss: f32) -> ListenerResponse {
        ListenerResponse::Nope
    }
}

#[cfg(test)]
pub mod tests {
    use crate::bool_iter::BoolTensorIterator;

    use crate::utils::{hist_1_dim, mix_x_binary_with_labels, pad_to_2, CompositeReducer, Splittable, SplittableTensor, TrainListener};
    use clap::{arg, command, Args, Parser, Subcommand};

    use log::debug;
    use std::collections::{HashMap, HashSet};

    use std::fmt::Debug;
    use std::fs;

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

    use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};

    use thoosa_shared::val::Val::{Int, UInt};
    use thoosa_shared::val::{NamedVal, Val};
    use thoosa_shared::{cdt_for_dir, clc_step, RangeOrValue, ThoosaError};

    use thoosa_shared::run_configs::RunId;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_splittable_tensor() {
        let t = Tensor::rand(&[3, 3], FLOAT_CPU);
        debug!("t=\n{t}");

        let sp_t = SplittableTensor::<0>(t.shallow_clone());

        debug!("sp_t=\n{sp_t:?}");

        let splits: [_; 2] = sp_t.split().expect("some splits");

        debug!("splits={splits:?}");

        assert_eq!(
            t,
            Tensor::concat(&[splits[0].as_ref(), splits[1].as_ref()], 0)
        )
    }

    #[test]
    fn test_composite_reducer() {
        let _reducer = CompositeReducer::new(vec![]);
        //TODO
    }

    // #[test]
    // fn test_run_split_on_oom_2(){
    //
    //
    //     fn run_step<ST>(step:ST)
    //     where ST:AsyncFnOnce()->Result<(), ThoosaError> + Send + 'static {
    //
    //
    //         rayon::scope(|scope|{
    //             scope.spawn_future(
    //                 step
    //             );
    //         });
    //     }
    //
    //     run_step(async || {
    //         Ok(())
    //     });
    //
    // }

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
    fn test_splittable_tensor_cant_split() {
        let t = Tensor::rand(&[3, 3], FLOAT_CPU);
        debug!("t=\n{t}");

        let sp_t = SplittableTensor::<0>(t.shallow_clone());
        let splits: Option<[_; 4]> = sp_t.split();

        assert!(splits.is_none())
    }

    #[test]
    fn test_cdt_for_dir() {
        debug!("{}", cdt_for_dir());
        assert_eq!(false, cdt_for_dir().contains(":"))
    }

    #[test]
    fn test_pad_to() -> Result<(), ThoosaError> {
        let t = Tensor::vstack(&[Tensor::from_slice(&[1., 2.]), Tensor::from_slice(&[3., 4.])]);

        assert_eq!(
            Tensor::vstack(&[
                Tensor::from_slice(&[1., 2., 0.]),
                Tensor::from_slice(&[3., 4., 0.]),
                Tensor::from_slice(&[0., 0., 0.]),
            ]),
            super::pad_to_2(&t, &[3, 3], false)?
        );

        assert_eq!(
            Tensor::vstack(&[
                Tensor::from_slice(&[1., 2., 0.]),
                Tensor::from_slice(&[3., 4., 0.]),
                Tensor::from_slice(&[0., 0., 0.]),
            ]),
            super::pad_to_2(&t, &[3, 3], true)?
        );

        Ok(())
    }

    #[test]
    fn test_pad_to_truncate_and_pad() -> Result<(), ThoosaError> {
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
        let actual = pad_to_2(&t, &[3, 2], true)?;

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn test_pad_to_same() -> Result<(), ThoosaError> {
        let t = Tensor::vstack(&[Tensor::from_slice(&[1., 2.]), Tensor::from_slice(&[3., 4.])]);

        assert_eq!(
            Tensor::vstack(&[Tensor::from_slice(&[1., 2.]), Tensor::from_slice(&[3., 4.]),]),
            super::pad_to(&t, &[2, 2])?
        );

        Ok(())
    }

    #[test]
    fn test_pad_to_one_dim() -> Result<(), ThoosaError> {
        let t = Tensor::ones([1, 3], (Float, Cpu));

        assert_eq!(
            Tensor::vstack(&[
                Tensor::from_slice(&[1., 1., 1.]),
                Tensor::from_slice(&[0., 0., 0.]),
            ]),
            super::pad_to(&t, &[2, 3])?
        );

        Ok(())
    }

    #[test]
    fn test_save_and_load() -> Result<(), ThoosaError> {
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
        ) -> Result<(f32, TestNodeParams), ThoosaError> {
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

                v.opt_run_configs()
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

        fn build_test_run_config<P, S, R>(run_config_range_or_value: S) -> R
        where
            P: FieldName,
            S: OptRunConfigSource<P = P>,
            R: From<std::vec::IntoIter<NamedVal<P>>>,
        {
            let run_config_source = run_config_range_or_value.opt_run_configs();

            let const_vals: Vec<NamedVal<P>> = run_config_source
                .grid_run_config_source()
                .const_params()
                .clone();
            let grid_vals: Vec<NamedVal<P>> = run_config_source.grid_to_vals()[0].clone();
            let opt_vals: Vec<NamedVal<P>> = run_config_source
                .opt_params()
                .into_iter()
                .map(|opt_param| opt_param.try_into().unwrap())
                .collect();

            [
                const_vals.as_slice(),
                grid_vals.as_slice(),
                opt_vals.as_slice(),
            ]
            .concat()
            .into_iter()
            .into()
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
    fn mix_basic_shapes_and_labels() -> Result<(), ThoosaError> {
        let m = 4i64;
        let n = 3i64;

        // Build A with values [0..m*n), B with values [1000..1000+m*n)
        let va: Vec<f32> = (0..(m * n)).map(|x| x as f32).collect();
        let vb: Vec<f32> = (0..(m * n)).map(|x| 1000.0 + x as f32).collect();

        let a = Tensor::from_slice(&va).reshape(&[m, n]);
        let b = Tensor::from_slice(&vb).reshape(&[m, n]);

        let (x, y) = mix_x_binary_with_labels(&a, &b)?;
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
}

pub fn z_norm(t: &Tensor, dim: i64) -> Tensor {
    let mean_t = t.mean_dim(dim, true, Float);
    let std_t = t.std(false);
    let z_t = (t - mean_t) / std_t;
    z_t
}

pub fn z_norm_2(t_0: &Tensor, _dim: i64) -> Result<Tensor, ThoosaError> {
    let std = t_0.std_dim(1, true, true);
    //TODO std - 0
    Ok((t_0 - t_0.mean_dim(1, true, Float)) / std)
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

pub fn dot_norm(dim: i64, slice_01: &Tensor, slice_02: &Tensor) -> Result<Tensor, ThoosaError> {
    Ok(norm_2(slice_01, dim)
        .tensordot(&norm_2(slice_02, dim), [dim], [dim])
        .diag(0))
}

pub fn self_dot(
    t1: &Tensor,
    i_0: i64,
    i_1: i64,
    dim: i64,
    step: i64,
    d_i: i64,
) -> Result<Tensor, ThoosaError> {
    let slice_1 = t1.f_slice(dim, Some(i_0), Some(i_0 + d_i), step)?;
    let slice_2 = t1.f_slice(dim, Some(i_1), Some(i_1 + d_i), step)?;

    Ok(slice_1.tensordot(&slice_2, [dim], [dim]).diag(0))
}

pub fn norm(t: &Tensor, dim: &[i64]) -> Tensor {
    t.view(dim) / t.dot(&t).sqrt()
}

pub fn norm_2(t: &Tensor, dim: i64) -> Tensor {
    t / t
        .tensordot(&t, dim, dim)
        .diag(0)
        .sqrt()
        .view([1, -1])
        .transpose(0, 1)
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
use crate::tho_tensor::ThoTensor;
use crate::train_nets::{LabelNetParams, MotifNetParams, ThoModuleWithStore};

pub trait DataBatch<D: Dims> {
    /// returns (inputs, labels)
    fn train_batch(&self) -> (ThoBatch<D>, Tensor);
    fn valid_batch(&self) -> (ThoBatch<D>, Tensor);
}

pub trait SplittableDataBatch<D: Dims>: DataBatch<D> + Splittable<1> + Debug {}

pub trait TrainListener: Send + Sync {
    fn on_cycle(&mut self, _cycle: usize, _loss: f32) -> BoxFuture<ListenerResponse> {
        async {ListenerResponse::Nope}.boxed()
    }
    fn on_valid_cycle(&mut self, cycle: u32, loss: f32) -> ListenerResponse;
}

pub enum ListenerResponse {
    Stop,
    UpdateLR(f64),
    Nope,
}

pub fn train<F, DI: Dims, DO: Dims>(
    device: &Device,
    data: &mut dyn DataBatch<DI>,
    net: &Box<dyn ThoModule<Batch<DI>, DO>>,
    loss_fn: LossFp,
    cycles: usize,
    mut opt: Optimizer,
    x_f: F,
    mut listener: Option<&mut dyn TrainListener>,
) -> Result<f32, ThoosaError>
where
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI>,
{
    debug!("train: device={:?} cycles={}", device, cycles);
    debug!("train: opt={opt:?} optimizer_size={}",optimizer_mem_size(&opt));

    let mut last_valid_loss = [0f32; 3];

    for idx in 1..cycles {
        debug!("train:  cuda_mem={:?}", cuda_mem());
        let (x_t_0, expected_y) = data.train_batch();
        debug!("train: device={:?} x_t_0.size={:?} expected_y.size={:?}", device, x_t_0.size(), expected_y.size());
        debug!("train: 0. cuda_mem={:?}", cuda_mem());

        let x_t = x_f(x_t_0);

        debug!("train: device={:?}  x_t.size={:?}",device,x_t.size());

        let predicted_y = net.tho_forward(&x_t, true)?;
        debug!("train: 1. cuda_mem={:?}", cuda_mem());
        trace!("train: x_t=\n{x_t}");
        x_t.drop();
        // unsafe { at_autocast_clear_cache(); }

        let loss = loss_fn(predicted_y.as_ref(), &expected_y)?;

        debug!("train: 2. cuda_mem={:?}", cuda_mem());
        trace!("train: loss={loss} predicted_y=\n{predicted_y} y_t=\n{expected_y} optimizer_size={}",optimizer_mem_size(&opt));

        predicted_y.drop();
        drop(expected_y);
        debug!("train: 2.1 cuda_mem={:?}", cuda_mem());
        let t1 = Instant::now();

        opt.backward_step(&loss);

        debug!(
            "train: 3. cuda_mem={:?}  optimizer_size={}",
            cuda_mem(),
            optimizer_mem_size(&opt)
        );
        debug!(
            "train: device={:?}  backward_step done in {}",
            device,
            t1.elapsed().as_millis()
        );

        if idx % 10 == 0 {
            if let Some(rs) = tch::no_grad(||{
                let (x_t_0, y_t) = data.valid_batch();
                // debug!("train: x_t_0.size={:?}", x_t_0.size());

                let x_t = x_f(x_t_0);

                let pred_y = net.tho_forward(&x_t, true)?;

                let valid_loss = loss_fn(&pred_y.as_ref(), &y_t)?;
                let valid_loss_f32 = valid_loss.double_value(&[]) as f32;

                last_valid_loss[idx % last_valid_loss.len()] = valid_loss_f32;
                debug!("train: last_valid_loss={last_valid_loss:?}  loss={loss:?}, valid loss={valid_loss:?} avg loss={}", last_valid_loss.into_iter().sum::<f32>() / last_valid_loss.len() as f32);

                Ok::<_, ThoosaError>(listener.as_mut().map(|l|l.on_valid_cycle(idx as u32, valid_loss_f32)))

            })?.take() {
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
    debug!(
        "train: final valid last_valid_loss={last_valid_loss:?} avgvalid_loss={}",
        last_valid_loss.into_iter().sum::<f32>() / last_valid_loss.len() as f32
    );
    Ok(last_valid_loss.into_iter().sum::<f32>() / last_valid_loss.len() as f32)
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

        ptr::write(threshold_ptr, 100_000);
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
pub fn mix_x_binary_with_labels(x_1: &Tensor, x_0: &Tensor) -> Result<(Tensor, Tensor), ThoosaError> {
    let [ma, na, ..] = x_1.size()[0..2] else {panic!("todo")}; //.expect("X1 must be 2D [m, n]");
    let [mb, nb, ..] = x_0.size()[0..2] else {panic!("todo")}; //.expect("X0 must be 2D [m, n]");
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

    Ok((x, y))
}



#[inline]
pub fn os_thread_id() -> u64 {
    #[cfg(target_os = "linux")]
    {
        // Works even if glibc doesn’t export gettid: call the syscall directly.
        unsafe { libc::syscall(libc::SYS_gettid) as u64 }
    }
}

///sums and multiplies inputs by k
pub fn test_net<DI: Dims + Send + 'static, DO: Dims + Send + 'static>(
    num_features: u32,
    k: f64,
    device: Device,
    descr: &str,
) -> Result<Box<ThoModuleWithStore<DI, DO>>, ThoosaError> {
    let vs = VarStore::new(device);


    let builder = Arc::new( move |path:&nn::Path|SeqBuilder::new(num_features, None)
        .linear_s_with_conf(
            1,
            LinearConfig {
                ws_init: nn::Init::Const(k), //conflicts with ctor's init
                bs_init: None,
                bias: false,
            },
            #[cfg(debug_assertions)]
            "test",
        )
        .end_seq().build_tch_module(path));

    Ok(Box::new(ThoModuleWithStore::with_factory(
        "test_net".into(),
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
        Box::new(move || MotifNetParams::new(motif_width, 1, 100)),
        Box::new(move || {
            *test_net(motif_width, k, device, "test_motif_net").unwrap()
        }),
    )
}

pub fn assert_close<A:Into<f64> + Display, B:Into<f64> + Display>(val_0: A, val_1: B, max_diff: f64) {
    let a = <A as Into<f64>>::into(val_0);
    let b = <B as Into<f64>>::into(val_1);
    let diff = (a - b).abs();
    assert!(
        diff <= max_diff,
        "values not close enough: {a} vs {b} (diff={diff}, max_diff={max_diff})"
    );
}
