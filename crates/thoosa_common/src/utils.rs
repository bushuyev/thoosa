use std::cell::{RefCell, RefMut};
use std::collections::VecDeque;
use std::ffi::c_void;
use std::fmt::{Debug, Display};
use std::io::Write;
use std::iter::Map;
use std::ops::ControlFlow::Continue;
use std::ops::{Add, Div, Mul, Range};
use std::time::Instant;
use std::{fs, mem, ptr};

use chrono::SecondsFormat;
use cudarc::driver::CudaDevice;
use log::{debug, error, info, trace, warn};
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::drawing::IntoDrawingArea;
use plotters::element::PathElement;
use plotters::prelude::*;
use rand::prelude::ThreadRng;
use rand::thread_rng;
use rand::Rng;
use std::panic::AssertUnwindSafe;
use std::rc::Rc;
use tch::display::PrinterOptions;
use tch::kind::FLOAT_CPU;
use tch::nn::Optimizer;
use tch::nn::{Module, VarStore};
use tch::Device::{Cpu, Cuda};
use tch::Kind::Float;
use tch::{Device, IndexOp, Kind, TchError, Tensor};
use thoosa_shared::{cdt, gauss_i, pulse_in_rand, pulse_v, pulses_in_const, saw, ThoosaError};
use torch_sys::cuda::atc_cuda_device_count;
use torch_sys::*;

use crate::my_tch::{LossFp, ThoModule};
use thoosa_shared::opt_rng::opt_rnd;

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
    _plot_series_with_points(name_x_y_vec, caption, w_h, path, None)
}

pub fn plot_series_with_points<S>(name_x_y_vec: Vec<(String, S, Vec<(f32, f32)>)>, caption: &str, w_h: (u32, u32), path: &impl AsRef<std::path::Path>, points:&Vec<(f32, f32)>) -> Result<(), ThoosaError> where S:Into<ShapeStyle> + Clone{
    _plot_series_with_points(name_x_y_vec, caption, w_h, path, Some(points))
}

fn _plot_series_with_points<S>(name_x_y_vec: Vec<(String, S, Vec<(f32, f32)>)>, caption: &str, w_h: (u32, u32), path: &impl AsRef<std::path::Path>, points:Option<&Vec<(f32, f32)>>) -> Result<(), ThoosaError> where S:Into<ShapeStyle> + Clone{
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
    debug!("pad_to: input.size={:?}, to_size{:?}", input.size(), to_size);

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
            debug!("pad_to: i={i} from={from} to={to}");
            if from > to {
                error!("from {from} > to {to} at {i}");//panic doesn't show up in test
                panic!("from {from} > to {to} at {i}");
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

pub fn hist_1_dim(tensor: &Tensor, bins: u32) -> Tensor {
    assert_eq!(tensor.dim(), 2, "hist_1_dim: input tensor must have exactly two dimensions.");


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


//free_mem, total_mem
pub fn cuda_mem() -> (usize, usize){
    let mut free_mem: usize = 0;
    let mut total_mem: usize = 0;
    unsafe {
        CudaDevice::new(0);//TODO device

        let cudarc_lib = cudarc::driver::sys::lib();
        cudarc_lib.cuMemGetInfo_v2(&mut free_mem as *mut usize, &mut total_mem as *mut usize);

        (free_mem, total_mem)
    }
}

#[macro_export]
macro_rules! panic_to_err {

    ($expr:expr) => {{
        match std::panic::catch_unwind(AssertUnwindSafe(|| {
            $expr
        })) {
            //no panic
            Ok(inner_result) => match inner_result {
                Ok(val) => Ok(val),
                Err(err) => Err(ThoosaError::from(err))
            },
            //panic
            Err(e) => {
                if let Some(msg) = e.downcast_ref::<String>() {
                    debug!("thoosa_err String: {msg}");
                    Err(ThoosaError::from(msg))
                //TODO
                // } else if let Some(thoosa_error) = e.downcast_ref::<ThoosaError>() {
                //     Err(*thoosa_error)

                } else if let Some(msg) = e.downcast_ref::<&str>() {
                    debug!("thoosa_err str: {msg}");
                    Err(ThoosaError::from(*msg))
                } else {
                    debug!("thoosa_err unknown: {e:?}");
                    Err(ThoosaError::from(format!("{e:?}")))
                }
            }
        }
    }};
}

pub trait Splittable<const SIZE: usize> {
    fn split(&self) -> Option<[Self; SIZE]> where Self: Sized;
}


pub trait Splitter<const SIZE: usize> {
    type S:Splittable<SIZE>;
    fn next(&mut self) -> Option<Self::S>;
    fn split(&mut self, split: Self::S) -> Result<(), ThoosaError>;
}

#[derive(Debug)]
pub struct SplitQueue<const SIZE: usize, T> {
    queue:VecDeque<T>
}

impl <const SIZE: usize, T> SplitQueue<SIZE, T> where T:Debug + Splittable<SIZE>{
    pub fn new(chunk: T) -> Self {
        Self {
            queue: VecDeque::from([chunk])
        }
    }
}

impl <const SIZE: usize, T> Splitter<SIZE> for SplitQueue<SIZE, T> where T:Debug + Splittable<SIZE> {
    type S=T;

    fn next(&mut self) -> Option<T> {
        let v = self.queue.pop_front();

        debug!("SplitQueue::next: self.queue.len={} pop_front={v:?}", self.queue.len());

        v
    }

    fn split(&mut self, chunk:T) -> Result<(), ThoosaError> {
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
    value:Option<T>
}

impl <T> Reducer<T> {

    pub fn new(value: T) -> Self {
        Self { value:Some(value) }
    }
}

impl <T> Splitter<1> for Reducer<T> where T:Debug + Splittable<1> {
    type S = T;

    fn next(&mut self) -> Option<Self::S> {
        self.value.take()
    }

    fn split(&mut self, v: Self::S) -> Result<(), ThoosaError> {
        match v.split() {
            None => {
                Err(ThoosaError::new("Can't split anymore".to_owned()))
            }
            Some(a) => {
                let [v_0] = a;
                self.value = Some(v_0);

                Ok(())
            }
        }
    }
}

pub fn run_split_on_oom<const SIZE: usize, F, FR, V, S, OkF, OkR>(mut splitter: S, mut f: F, mut on_ok: OkF, max_attempts: i32) -> Result<(), ThoosaError>
where
    F: FnMut(&mut V) -> Result<FR, ThoosaError>,
    S: Splitter<SIZE, S=V>,
    OkF: FnMut(FR, &mut V) -> OkR,
{

    'out: while let Some(mut split_0) = splitter.next() {
        for attempt in 0..max_attempts {
            match panic_to_err!(f(&mut split_0)) {
                Ok(fr) => {
                    on_ok(fr, &mut split_0);
                    break 'out;
                }
                Err(e) if e.is_oom() => {
                    debug!("run_split_on_oom: splitting on oom, max_attempts={max_attempts} attempt={attempt}");
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
pub struct SplittableTensor<const D:i64>(pub Tensor);//TODO more than one dimension?

impl <const S:usize, const D:i64> Splittable<S> for SplittableTensor<D> {
    fn split(&self) -> Option<[Self; S]>
    where
        Self: Sized,
    {

        let split_size = self.0.size()[D as usize] / S as i64 + 1; //TODO if self.0.size()[D as usize] < 2

        debug!("SplittableTensor::split: self.0.size()={:?} S={S} D={D} split_size={split_size}", self.0.size());

        let splits = self.0.split(split_size, D).into_iter().map(|t|SplittableTensor::<D>(t)).collect::<Vec<_>>();

        debug!("SplittableTensor::split: splits={splits:?}");

        // let splits_len = splits.len();

        splits.try_into().ok()
    }
}

//TODO for batch + degrade Kind: Double->Float->BFloat16
pub struct CompositeReducer {
    reducers:Vec<Box<dyn Splittable<1>>>
}

impl CompositeReducer {
    pub fn new(reducers:Vec<Box<dyn Splittable<1>>>) -> Self {
        Self { reducers }
    }
}

impl <const D:i64> AsRef<Tensor> for SplittableTensor<D> {
    fn as_ref(&self) -> &Tensor {
        &self.0
    }
}

#[cfg(test)]
pub mod tests {
    use clap::{Arg, ArgAction, ArgGroup, ArgMatches, Command};
    use crate::bool_iter::BoolTensorIterator;
    use crate::parzen::optimizer::{ConfigurationSpace, TPEOptimizer};
    use crate::runs::Simple;
    use crate::train_nets::{motif_node, opt_train_2, InputFilter, MotifNodeParams, ThoModuleWithStore, TrainLabelNodeRunConfig, TrainNodesValues};
    use crate::utils::{hist_1_dim, CompositeReducer, ListenerResponse, Splittable, SplittableTensor, TrainListener};
    use clap::{arg, command, Args, Parser, Subcommand};
    use cudarc::driver::CudaDevice;
    use log::LevelFilter::{Info, Trace};
    use log::{debug, error, info, set_max_level};
    use std::collections::{HashMap, HashSet};
    use std::path::PathBuf;
    use std::time::Instant;
    use std::{fs, mem};
    use std::fmt::{Debug, Display, Formatter};
    use std::marker::PhantomData;
    use std::str::FromStr;
    use clap::builder::Str;
    use serde::{Deserialize, Serialize};
    use tch::kind::FLOAT_CPU;
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::Float;
    use tch::{IndexOp, Tensor};
    use tch::nn::VarStore;
    use thoosa_run_config_args::{OptRunConfigSource, RunConfig};
    use thoosa_run_config_args_derive::ThoosaRunConfig;

    use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};
    use thoosa_shared::resource::ResourcesConfig;
    use thoosa_shared::val::Val::{Int, UInt};
    use thoosa_shared::val::{NamedRange, NamedVal, Val};
    use thoosa_shared::{add_prefix, add_range_or_value, cdt_for_dir, clc_step, RangeOrValue, ResourceType, ThoosaError};
    use thoosa_shared::run_configs::{ConstConfigs, GridRunConfigs, OptRunConfigs, RunId};

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

        let splits:[_; 2] = sp_t.split().expect("some splits");

        debug!("splits={splits:?}");

        assert_eq!(
            t,
            Tensor::concat(&[splits[0].as_ref(), splits[1].as_ref()], 0)
        )
    }

    #[test]
    fn test_composite_reducer(){
        let reducer = CompositeReducer::new(vec![

        ]);
        //TODO
    }

    #[test]
    fn test_clc_step(){
        let n = 10_000;
        let w = 1000;
        let t = Tensor::rand(&[1, n], FLOAT_CPU);
        let k = 100;
        let (step, pad) = clc_step(n, w, k); //(n-w)/(k-1);
        let u = t.pad([0, pad], "constant", 0.).unfold(1, w, step);

        debug!("u={u:?}");

        assert_eq!(k as i64, u.size()[1]);
    }

    #[test]
    fn test_splittable_tensor_cant_split(){
        let t = Tensor::rand(&[3, 3], FLOAT_CPU);
        debug!("t=\n{t}");

        let sp_t = SplittableTensor::<0>(t.shallow_clone());
        let splits:Option<[_; 4]> = sp_t.split();

        assert!(
            splits.is_none()
        )
    }


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
    fn test_run_config_args() {

        #[derive(ThoosaRunConfig, Debug)]
        struct SomeSubTrainConfig {
            #[run_config]
            lr: i32,
        }

        #[derive(ThoosaRunConfig, Debug)]
        struct CommonTrainConfig {
            #[run_config]
            beta: Option<i32>,

            #[run_config]
            sub_config:SomeSubTrainConfig
        }

        #[derive(ThoosaRunConfig, Debug)]
        enum NetType {
            NetOne(NetOneConfig),
            NetTwo(NetTwoConfig)
        }


        #[derive(ThoosaRunConfig, Debug)]
        struct NetOneConfig {
            #[run_config]
            num_aaa: u32,
        }

        #[derive(ThoosaRunConfig, Debug)]
        struct NetTwoConfig {
            #[run_config]
            num_bbb: u32,
        }

        #[derive(ThoosaRunConfig, Debug)]
        #[run_config_opt]
        struct TrainLabelsRunConfig {
            #[run_config]
            some_prop:u32,

            #[run_config]
            common_config:CommonTrainConfig,

            #[run_config]
            net_type_0: NetType,

            #[run_config]
            net_type_1: Option<NetType>

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
            StandardCommand {number:u32},

            TrainLabels(TraneLabelsArgs)
        }

        #[derive(Debug, Args, Clone)]
        struct TraneLabelsArgs{
            #[arg(long)]
            resources: String,

            #[arg(long)]
            run_config_cnt:String,
        }

        let cli = TestCli::parse_from([
            "",
            "--some-field", "zzz",
            "train-labels",
                "--resources", "zzzz",
                "--run-config-cnt", r#"
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
                "#
        ]);


        #[derive(Debug, Default)]
        struct TestNodeParams {
            layers:u32
        }

        struct DummyListener;

        impl TrainListener for DummyListener {
            fn on_valid_cycle(&mut self, cycle: usize, loss: f32) -> ListenerResponse {
                ListenerResponse::Nope
            }
        }

        fn test_train_label(vs:&VarStore, run_config:&TrainLabelsRunConfig, ls:&mut dyn TrainListener)->Result<(f32, TestNodeParams), ThoosaError>{
            debug!("test_train_label: run_config={:?}", run_config);

            Ok((0.1, TestNodeParams{ layers: 0 }))
        }


        let opt_sources = match &cli.command {

            Commands::TrainLabels(train_labels_args) =>{
                debug!("Commands::TrainLabels: train_labels_args={train_labels_args:?}");

                let v: TrainLabelsRunConfigRangeOrValue = serde_json5::from_str(&train_labels_args.run_config_cnt).unwrap();

                debug!("TrainLabelsRunConfigRangeOrValue: v={v:?}");

                assert_eq!(RangeOrValue::Value(1), v.some_prop);
                assert_eq!(Some(RangeOrValue::Range{range: 1..10, step: None}), v.common_config.beta);
                assert_eq!(RangeOrValue::Range{range: 0..10, step: Some((5,1))}, v.common_config.sub_config.lr);

                let NetTypeRangeOrValue::NetOne(NetOneConfigRangeOrValue {num_aaa: ref num_aaa_range_or_value}) = v.net_type_0 else {panic!("no NetOne/num_aaa")};
                assert_eq!(&RangeOrValue::Value(1), num_aaa_range_or_value);

                let Some(NetTypeRangeOrValue::NetTwo(NetTwoConfigRangeOrValue {num_bbb: ref num_bbb_range_or_value})) = v.net_type_1 else {panic!("no NetOne/num_bbb")};
                assert_eq!(&RangeOrValue::Range{range: 10..100, step: Some((50,1))}, num_bbb_range_or_value);


                v.opt_run_configs()
            }

            _ => {
                panic!("Subcommand not implemented");
            }
        };

        debug!("const_params={:?}", opt_sources.grid_run_config_source().const_params());
        assert_eq!(
            &vec![
                NamedVal::new(TrainLabelsRunConfigFields::SomeProp, UInt(1)),
                NamedVal::new(TrainLabelsRunConfigFields::NetType0(NetTypeFields::NetOne(NetOneConfigFields::NumAaa)), UInt(1)),
                // NamedVal::new(TrainLabelsRunConfigFields::NetType1(NetTypeFields::NetTwo(NetTwoConfigFields::NumBbb)), UInt(2)),
            ],
            opt_sources.grid_run_config_source().const_params()
        );


        debug!("       grid_to_vals={:?}", opt_sources.grid_run_config_source().grid_to_vals());
        assert_eq!((10..100).step_by(50).flat_map(|num_bbb| (0..10).step_by(5).map(move |lr| {

            vec![
                NamedVal::<TrainLabelsRunConfigFields>::new(TrainLabelsRunConfigFields::CommonConfig(CommonTrainConfigFields::SubConfig(SomeSubTrainConfigFields::Lr)), Int(lr)),
                NamedVal::<TrainLabelsRunConfigFields>::new(TrainLabelsRunConfigFields::NetType1(NetTypeFields::NetTwo(NetTwoConfigFields::NumBbb)), UInt(num_bbb))
            ]

        })).collect::<Vec<_>>(), opt_sources.grid_run_config_source().grid_to_vals());

        debug!("opt_params={:?}", opt_sources.opt_params());
        assert_eq!(
            &vec![Hyperparameter::UniformInt { name: TrainLabelsRunConfigFields::CommonConfig(CommonTrainConfigFields::Beta), lower: 1, upper: 10 }],
            opt_sources.opt_params()
        );

        // opt_train_2(test_train_label, &opt_sources, Cpu, "test").unwrap();

    }

    #[test]
    fn test_run_config_option() {

        #[derive(ThoosaRunConfig, Clone, PartialEq, Debug, Deserialize)]
        #[run_config_opt]
        pub struct TestRunConfig {
            pub id:Option<u32>,

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
            Filter2(Filter2RunConfig)
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

        let run_config_range_or_value:TestRunConfigRangeOrValue = serde_json5::from_str(r#" {
            step:'1',
            window:'10',
            keep_slice_matches: 'true',
            dist_quantile: '0.1',
            descr: 'test'
        } "#).unwrap();

        assert_eq!(
            TestRunConfig {
                id:None,
                step: 1,
                window: 10,
                dist_quantile: 0.1,
                keep_slice_matches: true,
                descr: "test".to_string(),
                filter:None
            },
            build_test_run_config(run_config_range_or_value)
        );


        let run_config_range_or_value:TestRunConfigRangeOrValue = serde_json5::from_str(r#" {
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
        } "#).unwrap();

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
        where P:FieldName, S:OptRunConfigSource<P=P>,
            R:From<std::vec::IntoIter<NamedVal<P>>>

        {
            let run_config_source = run_config_range_or_value.opt_run_configs();

            let const_vals: Vec<NamedVal<P>> = run_config_source.grid_run_config_source().const_params().clone();
            let grid_vals: Vec<NamedVal<P>> = run_config_source.grid_to_vals()[0].clone();
            let opt_vals: Vec<NamedVal<P>> = run_config_source.opt_params().into_iter().map(|opt_param| opt_param.try_into().unwrap()).collect();

            [
                const_vals.as_slice(),
                grid_vals.as_slice(),
                opt_vals.as_slice()
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
            #[run_config]
            field_00: u32,

            #[run_config]
            field_01: f32,

            #[run_config]
            sub_struct: SubStructB,

            #[run_config]
            sub_struct_op: Option<SubStructB>,

            #[run_config]
            sub_enum: SubEnum
        }

        let path_vals = [
            NamedVal::new(SomeStructFields::Field00, Int(1)), NamedVal::new(SomeStructFields::Field01, Val::Real(2.0)),
            NamedVal::new(SomeStructFields::SubStruct(SubStructBFields::FieldA), Int(3)), NamedVal::new(SomeStructFields::SubStruct(SubStructBFields::FieldB), Int(4)),
            NamedVal::new(SomeStructFields::SubStructOp(SubStructBFields::FieldA), Int(5)), NamedVal::new(SomeStructFields::SubStructOp(SubStructBFields::FieldB), Int(6)),
            NamedVal::new(SomeStructFields::SubEnum(SubEnumFields::Sub0(SubStruct0Fields::FieldA)), UInt(1)),
        ];

        let some_struct: SomeStruct = path_vals.into_iter().into();// SomeStructBuilder::add_path_vals(path_vals);

        debug!("some_struct={some_struct:?}");

        assert_eq!(some_struct.field_00, 1);
        assert_eq!(some_struct.field_01, 2.0);
        assert_eq!(some_struct.sub_struct.field_a, 3);
        assert_eq!(some_struct.sub_struct.field_b, 4);

        assert_eq!(some_struct.sub_struct_op.as_ref().map(|v|v.field_a), Some(5));
        assert_eq!(some_struct.sub_struct_op.as_ref().map(|v|v.field_b), Some(6));

        let SubEnum::Sub0(sub_struct_0) = some_struct.sub_enum else {
            panic!("some_struct.sub_enum should be Sub0");
        };
        assert_eq!(sub_struct_0.field_a, 1);

        let path_vals = [
            NamedVal::new(SomeStructFields::Field00, Val::Int(1)), NamedVal::new(SomeStructFields::Field01, Val::Real(2.0)),
            NamedVal::new(SomeStructFields::SubStruct(SubStructBFields::FieldA), Val::Int(3)), NamedVal::new(SomeStructFields::SubStruct(SubStructBFields::FieldB), Val::Int(4)),
            NamedVal::new(SomeStructFields::SubEnum(SubEnumFields::Sub1(SubStruct1Fields::FieldB)), Val::UInt(2)),
        ];

        let some_struct: SomeStruct = path_vals.into_iter().into();// SomeStructBuilder_::add_path_vals(path_vals);
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
    fn test_bool_tensor_iterator(){

        let t = Tensor::from_slice((0..100).map(|v|v % 2 == 0).collect::<Vec<_>>().as_slice());


        for (i, b) in (0..100).zip::<BoolTensorIterator>((&t).into()){
            assert_eq!(i % 2 == 0, b);
        }

    }

    #[test]
    fn test_hist_1_dim(){
        // log::set_max_level(Trace);
        let t = Tensor::vstack(&[
            Tensor::from_slice(&[1., 2., 3., 4., 5., 6., 7., 8., 9.,]),
            Tensor::from_slice(&[1., 2., 4., 4., 5., 6., 6., 8., 9.,]),
            Tensor::from_slice(&[1., 1., 1., 1., 1., 1., 1., 1., 1.,]),
        ]);

        // debug!("t={t}");

        let hist = hist_1_dim(&t, 3);
        // debug!("hist={hist}");

        assert_eq!(
            Tensor::from_slice(&[
               3, 3, 3,
               2, 5, 2,
               9, 0, 0
            ]).view((3,3)),
            hist
        )
    }


    // #[test]
    fn test_mem_track()->Result<(), ThoosaError>{//TODO util
        let mut free_mem_0: usize = 0;
        let mut free_mem_1: usize = 0;
        let mut total_mem: usize = 0;
        let _device = CudaDevice::new(0)?;
        let cudarc_lib = unsafe {
            let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");

            let cudarc_lib = cudarc::driver::sys::lib();
            cudarc_lib.cuMemGetInfo_v2(&mut free_mem_0 as *mut usize, &mut total_mem as *mut usize);
            debug!("Device 0 free_mem: {free_mem_0}, total_mem: {total_mem}");
            cudarc_lib.cuMemGetInfo_v2(&mut free_mem_0 as *mut usize, &mut total_mem as *mut usize);

            // cudarc_lib.

            cudarc_lib
        };

        set_max_level(Info);
        let motif_node_params = MotifNodeParams::from_file(&PathBuf::from("/tmp/ws_test/models/motifs/0/31211_366.json"))?;

        let inst_0 = Instant::now();
        let mut module = ThoModuleWithStore::load(motif_node(&motif_node_params), "/tmp/ws_test/models/motifs/0/31211_366.safetensors".into(), Cpu)?;
        for i in 0..100 {
            module.set_device(Cuda(0));
            module.set_device(Cpu);
        }

        info!("cuda/cpu done in {}", inst_0.elapsed().as_millis());

        let inst_0 = Instant::now();

        for i in 0..100 {
            let mut module = ThoModuleWithStore::load(motif_node(&motif_node_params), "/tmp/ws_test/models/motifs/0/31211_366.safetensors".into(), Cuda(0))?;
        }

        info!("cuda load done in {}", inst_0.elapsed().as_millis());

        unsafe {
            cudarc_lib.cuMemGetInfo_v2(&mut free_mem_1 as *mut usize, &mut total_mem as *mut usize);
            debug!("Device 0 free_mem: {free_mem_1}, total_mem: {total_mem}");
        }

        debug!("free_mem_1 diff={}", (free_mem_0 - free_mem_1)/1024/1024);

        Ok(())
    }

    #[test]
    fn test_to_from_hashmap(){
        #[derive(Debug, PartialEq, Clone)]
        pub enum TestVal {
            Real(f64),
            Int(i32),
            UInt(u32),
            Str(String),
            Bool(bool)
        }

        impl TryFrom<&TestVal> for u32 {
            type Error = ThoosaError;

            fn try_from(value: &TestVal) -> Result<Self, Self::Error> {
                match value {
                    TestVal::UInt(v) => Ok(*v),
                    _=> panic!("can't convert")
                }
            }
        }

        impl TryFrom<&TestVal> for i32 {
            type Error = ThoosaError;

            fn try_from(value: &TestVal) -> Result<Self, Self::Error> {
                match value {
                    TestVal::Int(v) => Ok(*v),
                    _=> panic!("can't convert")
                }
            }
        }

        //TODO implement TryFrom for other types

        trait ToHashmap {
            fn to_properties_iter(&self, parent: Option<String>,) -> Box<dyn Iterator<Item=(String, TestVal)>>;
        }

        trait FromHashmap {
            fn from_properties_map(parent:Option<String>, properties: &HashMap<String, TestVal>) -> Result<Self, ThoosaError> where Self:Sized;
        }

        #[derive(Debug, PartialEq)]
        struct TestTopStruct {
            field_a:u32,
            field_b:i32,
            field_o:Option<i32>,
            sub_struct:AnotherTestStruct,
            sub_struct_o:Option<AnotherTestStruct>,
            sub_enum:SomeEnum
        }

        impl ToHashmap for TestTopStruct {
            fn to_properties_iter(&self, parent: Option<String>,) -> Box<dyn Iterator<Item=(String, TestVal)>> {
                Box::new([
                    ("TestTopStruct::field_a".to_string(), Some(TestVal::UInt(self.field_a))),
                    ("TestTopStruct::field_b".to_string(), Some(TestVal::Int(self.field_b))),
                    ("TestTopStruct::field_o".to_string(), self.field_o.map(|v|TestVal::Int(v))),
                ]
                    .into_iter()
                    .filter_map(|v|v.1.map(|val|(v.0, val)))
                    .chain(self.sub_struct.to_properties_iter(Some("TestTopStruct::sub_struct.".to_string())))
                    .chain(self.sub_struct_o.as_ref().map(|s|s.to_properties_iter(Some("TestTopStruct::sub_struct_o.".to_string()))).unwrap_or(Box::new(std::iter::empty::<(String, TestVal)>())))
                    .chain(self.sub_enum.to_properties_iter(Some("TestTopStruct::sub_enum.".to_string())))
                )
            }
        }

        impl FromHashmap for TestTopStruct {
            fn from_properties_map(parent:Option<String>, properties: &HashMap<String, TestVal>) -> Result<Self, ThoosaError> {
                let prefix = parent.unwrap_or("TestTopStruct::".to_string());
                Ok(TestTopStruct {
                    field_a: properties.get(&(thoosa_shared::conc_str(&prefix, "field_a"))).expect("todo msg").try_into().expect("..."),
                    field_b: properties.get(&(prefix.clone()+"field_b")).expect("todo msg").try_into().expect("..."),
                    field_o: properties.get(&(prefix.clone()+"field_o")).map(|v|v.try_into().expect("...")),
                    sub_struct: AnotherTestStruct::from_properties_map(Some("TestTopStruct::sub_struct.".to_string()), &properties)?,
                    sub_struct_o: AnotherTestStruct::from_properties_map(Some("TestTopStruct::sub_struct_o.".to_string()), &properties).ok(),
                    sub_enum: SomeEnum::from_properties_map(Some("TestTopStruct::sub_enum.".to_string()), &properties)?,
                })
            }
        }


        #[derive(Debug, PartialEq)]
        enum SomeEnum {
            VariantA(VariantA),
            VariantB(VariantB)
        }

        impl ToHashmap for SomeEnum {
            fn to_properties_iter(&self, parent: Option<String>) -> Box<dyn Iterator<Item=(String, TestVal)>> {
                let prefix = parent.unwrap_or("SomeEnum::".to_string());
                match &self {
                    SomeEnum::VariantA(v) => {
                        Box::new([
                            ((prefix.clone() + "variant_a").to_string(), TestVal::Bool(true)),
                        ]
                            .into_iter()
                            .chain(
                        v.to_properties_iter(Some((prefix.clone() + "variant_a.").to_string()))))
                    }
                    SomeEnum::VariantB(v) => {
                        Box::new([
                            ((prefix.clone() + "variant_b").to_string(), TestVal::Bool(true)),
                        ]
                            .into_iter()
                            .chain(
                                v.to_properties_iter(Some((prefix.clone() + "variant_b.").to_string()))))
                    }
                }
            }
        }

        impl FromHashmap for SomeEnum {

            fn from_properties_map(parent:Option<String>, properties: &HashMap<String, TestVal>) -> Result<Self, ThoosaError>{
                let prefix = parent.unwrap_or("TestTopStruct::".to_string());

                if properties.get(&(prefix.clone() + "variant_a")).is_some() {
                    Ok(SomeEnum::VariantA(VariantA::from_properties_map(Some(prefix.clone() + "variant_a."), properties)?))

                } else if properties.get(&(prefix.clone() + "variant_b")).is_some() {
                    Ok(SomeEnum::VariantB(VariantB::from_properties_map(Some(prefix.clone() + "variant_b."), properties)?))

                } else {
                    unreachable!()
                }
            }
        }

        #[derive(Debug, PartialEq)]
        struct VariantA {
            some_field:u32
        }
        impl ToHashmap for VariantA {
            fn to_properties_iter(&self, parent: Option<String>) -> Box<dyn Iterator<Item=(String, TestVal)>> {
                let prefix = parent.unwrap_or("VariantA::".to_string());
                Box::new([
                    ((prefix.clone()+"some_field").to_string(), TestVal::UInt(self.some_field)),
                ].into_iter())
            }
        }

        impl FromHashmap for VariantA {
            fn from_properties_map(parent:Option<String>, properties: &HashMap<String, TestVal>) -> Result<Self, ThoosaError> {
                let prefix = parent.unwrap_or("VariantA::".to_string());

                Ok(VariantA {
                    some_field: properties.get(&(prefix.clone() + "some_field")).expect("todo msg").try_into()?,
                })
            }

        }

        #[derive(Debug, PartialEq)]
        struct VariantB {
            another_field:u32
        }

        impl ToHashmap for VariantB {
            fn to_properties_iter(&self, parent: Option<String>) -> Box<dyn Iterator<Item=(String, TestVal)>> {
                let prefix = parent.unwrap_or("VariantB::".to_string());
                Box::new([
                    ((prefix.clone()+"another_field").to_string(), TestVal::UInt(self.another_field)),
                ].into_iter())
            }
        }

        impl FromHashmap for VariantB {
            fn from_properties_map(parent:Option<String>, properties: &HashMap<String, TestVal>) -> Result<Self, ThoosaError>{
                let prefix = parent.unwrap_or("VariantB::".to_string());

                Ok(VariantB {
                    another_field: properties.get(&(prefix.clone() + "another_field")).expect("todo msg").try_into()?,
                })
            }

        }


        #[derive(Debug, PartialEq)]
        struct AnotherTestStruct {
            field_c:u32,
            field_d:i32,
        }

        impl ToHashmap for AnotherTestStruct {
            fn to_properties_iter(&self, parent: Option<String>,) -> Box<dyn Iterator<Item=(String, TestVal)>> {
                let prefix = parent.unwrap_or("AnotherTestStruct::".to_string());
                Box::new([
                    ((prefix.clone()+"field_c").to_string(), TestVal::UInt(self.field_c)),
                    ((prefix.clone()+"field_d").to_string(), TestVal::Int(self.field_d))
                ].into_iter())
            }
        }

        impl FromHashmap for AnotherTestStruct {
            fn from_properties_map(parent:Option<String>, properties: &HashMap<String, TestVal>) -> Result<Self, ThoosaError> {
                let prefix = parent.unwrap_or("AnotherTestStruct::".to_string());

                debug!("AnotherTestStruct::from_properties_map: prefix={prefix:?}");

                Ok(AnotherTestStruct {
                    field_c: properties.get(&(prefix.clone() + "field_c")).map(|v|v.try_into().expect("zzz")).ok_or(ThoosaError::new("zzz"))?,
                    field_d: properties.get(&(prefix.clone() + "field_d")).map(|v|v.try_into().expect("zzz")).ok_or(ThoosaError::new("zzz"))?,
                })
            }

        }

        let run_config = TestTopStruct {
            field_a: 1,
            field_b: 2,
            field_o: Some(22),
            sub_struct: AnotherTestStruct { field_c: 3, field_d: 4 },
            sub_struct_o: Some(AnotherTestStruct { field_c: 30, field_d: 40 }),
            sub_enum: SomeEnum::VariantA(VariantA{ some_field: 5 }),
        };

        let props:HashMap<String, TestVal> = run_config.to_properties_iter(None).collect();
        debug!("props={props:#?}");
        assert_eq!(run_config, TestTopStruct::from_properties_map(None, &props).unwrap());


        let run_config = TestTopStruct {
            field_a: 1,
            field_b: 2,
            field_o: None,
            sub_struct: AnotherTestStruct { field_c: 3, field_d: 4 },
            sub_struct_o: None,
            sub_enum: SomeEnum::VariantA(VariantA{ some_field: 5 }),
        };

        let props:HashMap<String, TestVal> = run_config.to_properties_iter(None).collect();

        assert_eq!(run_config, TestTopStruct::from_properties_map(None, &props).unwrap());

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
    Tensor::from(lin_iter(n).collect::<Vec<f32>>().as_slice())
}

pub fn lin_iter(n: i64) -> Map<Range<i64>, fn(i64) -> f32> {
    (0..n).map(|x| x as f32)
}

pub trait DataBatch {
    /// returns (inputs, labels)
    fn train_batch(&self) -> (Tensor, Tensor);//TODO change to Box<Tensor>
    fn valid_batch(&self) -> (Tensor, Tensor);
}

pub trait SplittableDataBatch: DataBatch + Splittable<1> + Debug {

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
    model:&Box<dyn ThoModule>,
    loss_fn:LossFp,
    cycles: usize,
    mut opt: Optimizer,
    x_f:F,
    mut listener: Option<&mut dyn TrainListener>
) -> Result<f32, ThoosaError>
    where F:Fn(Tensor)->Tensor {

    debug!("train: device={:?} cycles={}", device, cycles);
    debug!("train: opt={opt:?} optimizer_size={}", optimizer_mem_size(&opt));

    let mut last_valid_loss = [0f32;3];

    for idx in 1..cycles {
        debug!("train:  cuda_mem={:?}", cuda_mem());
        let (x_t_0, expected_y) = data.train_batch();
        debug!("train: device={:?} x_t_0.size()={:?}", device, x_t_0.size());
        debug!("train: 0. cuda_mem={:?}", cuda_mem());

        let x_t = x_f(x_t_0);

        debug!("train: device={:?}  x_t.size()={:?} y_t.size()={:?}", device, x_t.size(), expected_y.size());

        let predicted_y = model.tho_forward(&x_t, true);
        debug!("train: 1. cuda_mem={:?}",  cuda_mem());
        trace!("train: x_t=\n{x_t}");
        drop(x_t);
        // unsafe { at_autocast_clear_cache(); }

        let loss = loss_fn(&predicted_y, &expected_y);

        debug!("train: 2. cuda_mem={:?}",  cuda_mem());
        trace!("train: loss={loss} predicted_y=\n{predicted_y} y_t=\n{expected_y} optimizer_size={}", optimizer_mem_size(&opt));
        drop(predicted_y);
        drop(expected_y);
        debug!("train: 2.1 cuda_mem={:?}",  cuda_mem());
        let t1 = Instant::now();

        opt.backward_step(&loss);

        debug!("train: 3. cuda_mem={:?}  optimizer_size={}", cuda_mem(),  optimizer_mem_size(&opt));
        debug!("train: device={:?}  backward_step done in {}", device, t1.elapsed().as_millis());


        if idx % 10 == 0 {

            if let Some(rs) = tch::no_grad(||{
                let (x_t_0, y_t) = data.valid_batch();
                // debug!("train: x_t_0.size={:?}", x_t_0.size());

                let x_t = x_f(x_t_0);

                let pred_y = model.tho_forward(&x_t, true);

                let valid_loss = loss_fn(&pred_y, &y_t);
                let valid_loss_f32 = valid_loss.double_value(&[]) as f32;


                last_valid_loss[idx % last_valid_loss.len()] = valid_loss_f32;
                debug!("train: last_valid_loss={last_valid_loss:?}  loss={loss:?}, valid loss={valid_loss:?} avg loss={}", last_valid_loss.into_iter().sum::<f32>() / last_valid_loss.len() as f32);

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
    let t_0_0 = t_0_0(n, sig_0_v, sig_2_v);

    // debug!("t_0_0 mean={} max-min={}", t_0_0.mean(None).double_value(&[]), t_0_0.max() - t_0_0.min());

    let t_0_1 = t_0_1(n, sig_0_v, sig_2_v);

    // debug!("t_0_1 mean={} max-min={}", t_0_1.mean(None).double_value(&[]), t_0_1.max() - t_0_1.min());


    let t_1_0 = t_1_0(n, sig_1_v, sig_2_v);


    // debug!("t_1_0 mean={} max-min={}", t_1_0.mean(None).double_value(&[]), t_1_0.max() - t_1_0.min());

    let t_1_1 = t_1_1(n, sig_1_v, sig_2_v);
    (t_0_0, t_0_1, t_1_0, t_1_1)
}

pub fn t_0_0(n: i64, sig_0_v: &Vec<f32>, sig_2_v: &Vec<f32>) -> Tensor {
    Tensor::vstack(&[
        Tensor::from_slice([sig_0_v.clone(), saw(n), saw(n)].concat().as_slice()),
        Tensor::from_slice([saw(n), saw(n), saw(n)].concat().as_slice()),
        Tensor::from_slice([saw(n), sig_2_v.clone(), saw(n)].concat().as_slice()),
    ])
}

pub fn t_0_1(n: i64, sig_0_v: &Vec<f32>, sig_2_v: &Vec<f32>) -> Tensor {
    Tensor::vstack(&[
        Tensor::from_slice([saw(n), sig_0_v.clone(), saw(n)].concat().as_slice()),
        Tensor::from_slice([saw(n), saw(n), saw(n)].concat().as_slice()),
        Tensor::from_slice([sig_2_v.clone(), saw(n), saw(n)].concat().as_slice()),
    ])
}

pub fn t_1_0(n: i64, sig_1_v: &Vec<f32>, sig_2_v: &Vec<f32>) -> Tensor {
    Tensor::vstack(&[
        Tensor::from_slice([saw(n), saw(n), saw(n)].concat().as_slice()),
        Tensor::from_slice([saw(n), sig_1_v.clone(), saw(n)].concat().as_slice()),
        Tensor::from_slice([saw(n), sig_2_v.clone(), saw(n)].concat().as_slice())
    ])
}

pub fn t_1_1(n: i64, sig_1_v: &Vec<f32>, sig_2_v: &Vec<f32>) -> Tensor {
    Tensor::vstack(&[
        Tensor::from_slice([saw(n), saw(n), saw(n)].concat().as_slice()),
        Tensor::from_slice([sig_1_v.clone(), saw(n), saw(n)].concat().as_slice()),
        Tensor::from_slice([sig_2_v.clone(), saw(n), saw(n)].concat().as_slice()),
    ])
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

pub fn reset_print_options(){
    let mut printer_options = PrinterOptions::default();

    unsafe {
        let printer_options_ptr: *mut PrinterOptions = &mut printer_options;

        let threshold_ptr = (printer_options_ptr as *mut u8).add(size_of::<usize>()*1) as *mut usize;
        let line_width_ptr = (printer_options_ptr as *mut u8).add(size_of::<usize>()*3) as *mut usize;

        ptr::write(threshold_ptr, 100_000);
        ptr::write(line_width_ptr, 10_000);
    }
    tch::display::set_print_options(printer_options);
}

pub fn tensor_mem_size(t:&Tensor) -> usize {
    t.numel() * t.kind().elt_size_in_bytes()
}

pub fn vs_mem_size(opt:&VarStore) -> usize {
    opt.trainable_variables().iter().map(|t| tensor_mem_size(t)).sum()
}

pub fn optimizer_mem_size(opt:&Optimizer) -> usize {
    opt.trainable_variables().iter().map(|t| tensor_mem_size(t)).sum()
}


pub fn set_requires_grad_in_subpath(vsd: &VarStore, subpath: &str, requires_grad: bool) {

    for (name, mut tensor) in vsd.variables() {
        if name.contains("batch_norm") {
            continue
        }
        let freezing = name.starts_with(subpath);
        debug!("freeze_subpath: name={name:?} freezing={freezing}");

        if freezing {
            _ = tensor.set_requires_grad(requires_grad);
        }
    }

}