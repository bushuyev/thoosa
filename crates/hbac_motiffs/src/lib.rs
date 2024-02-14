extern crate core;

use std::fmt::Display;
use std::fs::{File, OpenOptions};
use std::sync::{Arc, Mutex};

use log::debug;
use tch::{Device, Kind, kind, Tensor};
use rand::Rng;
use rayon::iter::{ParallelBridge, ParallelIterator};
use crate::utils::cdt;

pub mod utils;
mod tensor_iter;
mod cache_on_threads;
use std::io::Write;

use tensor_iter::*;

#[derive(Debug)]
pub struct GenError {
    msg:String
}

impl GenError {
    pub fn new(msg:String) -> Self {
        Self {
            msg
        }
    }
}


impl<D> From<D> for GenError where D: Display {
    fn from(value: D) -> Self {
        debug!("value={}", value);
        GenError::new(value.to_string())
    }
}

fn main() {

    let t = Tensor::from_slice(&[1., 2., 3.]);
    let t_0 = Tensor::zeros([3], (Kind::Float, Device::Cpu));
    // let t = Tensor::pad_sequence()

    println!("xxx h={:?}", Tensor::hstack(&[t.view([-1]), t_0.view([-1])]));
    println!("xxx t={:?} t_0={:?}", t.flip([0]), t_0);
}

fn z_norm(t:&Tensor, dim: i64) -> Tensor {

    let mean_t = t.mean_dim(dim, false, Kind::Float);
    let std_t = t.std(false);
    // println!("mean={:?}", vec_f32_from(&mean_t));
    // println!("std_t={:?}", vec_f32_from(&std_t));
    let z_t = (t - mean_t) / std_t;
    // println!("z_t={:?}", vec_f32_from(&z_t));
    z_t
}

fn self_dot_norm(dim: i64, slice_01: &Tensor, slice_02: &Tensor) -> Result<Tensor, GenError> {
    Ok(norm_2(slice_01).tensordot(&norm_2(slice_02), [dim], [dim]).diag(0))
}


pub fn find_motiffs_iter(loader: fn(&String) -> Box<Tensor>, keys: Vec<String>, m: i64, step: i64, min_dot: Vec<f32>, motiffs_dir:&str){

    let ts_iter = crate::tensor_iter::TensorsIter::new(
        Box::new(crate::tensor_iter::CacheOnThreadsStore::new(loader, keys)),
        1, m, step
    );
    let dir = format!("{}/{}", motiffs_dir, cdt());
    match std::fs::create_dir_all(dir.clone()){
        Ok(_) => {}
        Err(_) => {}
    }

    let all_mt = min_dot
        .into_iter()
        .enumerate()
        .map(|(i, min_dot)|Mutex::new(MotifTracker::new(min_dot, format!("{}/motifs_{}.csv", dir.clone(), i).as_str())))
        .collect::<Vec<Mutex<MotifTracker>>>();

    // ts_iter.par_bridge().map(|r|(Arc::clone(&all_mt), r)).for_each(|(mt, r)| {
    ts_iter.par_bridge().for_each(|r| {
        match r {
            Ok((t_s_0, t_s_1)) => {
                // v.0.tensordot(&v.1, 1, 1)
                let dp = self_dot_norm(1, &t_s_0.1, &t_s_1.1).unwrap();

                for (i, mt) in all_mt.iter().enumerate() {
                    mt.lock().expect("mt").track(t_s_0.0, t_s_1.0, dp.double_value(&[i as i64]) as f32);
                }

            }
            Err(e) => {
                panic!("v={:?}", e);
            }
        };
    });

    println!("saved to {:?}", dir);
}

fn norm(t: &Tensor, dim: &[i64]) -> Tensor {
    t.view(dim) / t.dot(&t).sqrt()
}

fn norm_2(t: &Tensor) -> Tensor {
    t/ t.tensordot(&t, 1, 1).diag(0).sqrt().view([1, -1]).transpose(0, 1)
}


#[derive(Debug, PartialEq, Copy, Clone)]
pub struct TensorSliceIdx(usize, i64);

impl From<(usize, i64)> for TensorSliceIdx {
    fn from(value: (usize, i64)) -> Self {
        TensorSliceIdx(value.0, value.1)
    }
}

impl Display for TensorSliceIdx {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{},{}", self.0, self.1)
    }
}

#[derive(Debug)]
pub(crate) struct MotifTracker {
    motifs:Vec<(TensorSliceIdx, TensorSliceIdx, TensorSliceIdx, f32)>,//i_0, match_0, match_1, dot
    motif_start_op:Option<TensorSliceIdx>,//tensor, slice
    min_dot: f32,
    i_0: TensorSliceIdx,
    file:File,
    written_len: usize,
    last_dot: f32
}

impl MotifTracker {
    pub fn new(min_dot:f32, path: &str) -> Self {
        match std::fs::remove_file(path){
            Ok(_) => {}
            Err(_) => {}
        }

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(path)
            .unwrap();

        Self {
            motif_start_op: None,
            min_dot,
            motifs: vec![],
            i_0: TensorSliceIdx(0, 0),
            file,
            written_len: 0,
            last_dot: 0.
        }
    }

    pub fn track(&mut self, i_0: TensorSliceIdx, i: TensorSliceIdx, dot:f32) {
        match self.motif_start_op {
            None => {
                if dot > self.min_dot {
                    self.motif_start_op = Some(i);
                    self.last_dot = dot;
                }
            }
            Some(motif_start) => {
                if dot <= self.min_dot {
                    self.motifs.push((i_0, motif_start, i, self.last_dot));
                    self.motif_start_op = None;
                    self.last_dot = 0.0;
                }
            }
        }

        if self.i_0 != i_0 {
            self.i_0 = i_0;
            self.dump();

        }

    }

    pub fn next(&mut self) {
        self.last_dot = 0.0;
        self.motif_start_op = None;
    }

    fn dump(&mut self) {
        for motif in self.motifs.iter().skip(self.written_len) {
            writeln!(self.file, "{}, {}, {},  {:.3}", motif.0, motif.1, motif.2, motif.3);
        }
        self.file.flush();

        self.written_len = self.motifs.len();
    }

    pub fn motifs(&self) -> &Vec<(TensorSliceIdx, TensorSliceIdx, TensorSliceIdx, f32)> {
        &self.motifs
    }
}

mod tests {
    use std::collections::VecDeque;
    use std::f64::consts::PI;
    use std::fs::{File, OpenOptions};
    use std::iter::Map;
    use std::mem::size_of_val;
    use std::ops::Range;
    use std::time::Instant;
    use std::io;
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    use log::{debug, LevelFilter, SetLoggerError};
    use plotters::prelude::{BLACK, BLUE, MAGENTA};
    use plotters::style::{GREEN, RED};
    use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger};
    use tch::{Device, Kind, kind, Tensor};
    use chrono::Utc;
    use rand::Rng;
    use rayon::prelude::ParallelBridge;
    use crate::{GenError, z_norm};
    use rayon::iter::ParallelIterator;
    use rayon::iter::IntoParallelIterator;
    use crate::*;
    use crate::utils::*;

    #[ctor::ctor]
    fn init() {
        match CombinedLogger::init(
            vec![
                TermLogger::new(LevelFilter::Debug, Config::default(), TerminalMode::Mixed, ColorChoice::Auto)
            ]
        ){
            Ok(_) => {}
            Err(_) => {}
        };
    }


    #[test]
    pub fn test_filt(){
        //20 x 200_000 in files 5000 files 20_000_000_000
        // let dots = [1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1];
        let dots = [0.5, 0.9, 0.9, 0.5, 0.5, 0.9, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5];

        let mut mt = crate::tests::MotifTracker::new(0.8, "/tmp/test_motifs.csv");

        for (i, dot) in dots.into_iter().enumerate(){
            mt.track(TensorSliceIdx(0, 0), TensorSliceIdx(0, i as i64), dot as f32);
        }
        assert_eq!([(TensorSliceIdx(0, 0), TensorSliceIdx(0, 1), TensorSliceIdx(0, 3), 0.9f32), (TensorSliceIdx(0, 0), TensorSliceIdx(0, 5), TensorSliceIdx(0, 8), 0.9f32)], mt.motifs()[0..2]);

        // let t1 = Instant::now();
        //
        // let mut sum = 0i64;
        // let n = 20_000_000_000i64; //_000
        // for i in (0..n) {
        //     sum+1;
        //     if i % 1_000_000 == 0 {
        //         print!(".");
        //         std::io::stdout().flush().unwrap();
        //     }
        // }
        // debug!("done {} in {}", n, t1.elapsed().as_millis());

    }

    #[test]
    pub fn   test_z_norm() {
        assert_eq!(vec![-1.2247449, 0.0, 1.2247449], vec_f32_from(&z_norm(&Tensor::from_slice(&[1., 2., 3.]), 0i64)));
    }

    #[test]
    pub fn test_norm() -> Result<(), GenError>{
        let t = Tensor::from_slice(&[1., 2., 3.]);
        let norm = norm(&t, &[-1]);

        assert_eq!(1.0, vec_f32_from(&norm.dot(&norm))[0]);


        let t = Tensor::vstack(&[
            Tensor::from_slice(&[1., 2., 3.]),
            Tensor::from_slice(&[1., 2., 3.])
        ]);

        let norm = norm_2(&t);

        assert_eq!(Tensor::from_slice(&[1., 1.]),norm.tensordot(&norm, 1, 1).diag(0));

        Ok(())
    }



    #[test]
    pub fn test_pulses_in_rand() -> Result<(), GenError>{
        let n = 1000i64;
        let x = lin_t(n);
        // let gauss_1 = gauss_t(400., 100., n);

        let to_pulses_in_random = pulses_in_rand(
            n,
            &[
                (100, pulse_v(80., 20., 200, 20.)),
                (100, pulse_v(80., 20., 200, 20.))
            ]
        );

        let sig = Tensor::from_slice(
            to_pulses_in_random.as_slice()
        );

        // // let t_10 = Tensor::from_slice((0..10).map(|i| i as f64).collect::<Vec<_>>().as_slice());
        // let t_1_snippet = gauss_1.f_slice(0, Some(0i64), Some(100i64), 1)?;
        // let t_2_snippet = gauss_1.f_slice(0, Some(0i64), Some(100i64), 1)?;
        // let t_2_1_snippet = gauss_1.f_slice(0, Some(10i64), Some(110i64), 1)?;
        //
        // let dot_0 = norm(&t_1_snippet).dot(&norm(&t_2_snippet));
        // let dot_1 = norm(&t_1_snippet).dot(&norm(&t_2_1_snippet));
        // //
        // println!("dot_0={:?}", dot_0);
        println!("sig={:?}", sig.size());

        crate::utils::plot_series(
            vec![
                (format!("G1"), BLACK, vec_f32_from(&x).into_iter().zip(vec_f32_from(&sig)).collect::<Vec<(f32, f32)>>()),

            ], "EEG", (1000, 500), format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        ).unwrap();

        Ok(())
    }

    // #[test]
    fn test_motifs_match(){
        let t1 = Instant::now();

        let n = 10_000;

        let mut t = Box::new(Tensor::zeros(&[n, n], (Kind::Float, Device::Cpu)));

        for i in (0..n) {
            if i % 100 == 0{
                debug!("i={}", i);
            }

            // std::io::stdout().flush().expect("flush");


            for j in (0..n) {

                t.index_put_(
                    &[
                        Some(Tensor::from_slice(&[i])),//row
                        Some(Tensor::from_slice(&[j])) //col
                    ],
                    &Tensor::from_slice(&[ (i * j) as f32  ]),
                    false
                );
            }
        }


        // debug!("mem size {}", size_of_val(&t.as_ptr()));
        debug!("done {} in {}", n, t1.elapsed().as_millis());
    }



    fn self_dot(t1: &Tensor, i_0: i64, i_1: i64, dim:i64, step:i64, d_i: i64) -> Result<Tensor, GenError> {
        let slice_1 = t1.f_slice(dim, Some(i_0), Some(i_0+d_i), step)?;
        let slice_2 = t1.f_slice(dim, Some(i_1), Some(i_1+d_i), step)?;

        // slice_t.print();

        Ok(slice_1.tensordot(&slice_2, [dim], [dim]).diag(0))
    }



    #[test]
    fn test_self_dot() -> Result<(), GenError>{
        let t1 = Tensor::from_slice2(&[
            &[1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
            &[10., 20., 30., 40., 50., 60., 70., 80., 90.],
            &[100., 200., 300., 400., 500., 600., 700., 800., 900.]
        ]);

        //(1,2,3) x
        //(1,2,3)
        let d_0 = self_dot(&t1, 0, 0, 1, 1, 3)?;
        assert_eq!(Tensor::from_slice(&[14., 1400., 140000.]), d_0);

        //(1,2,3) x
        //(2,3,4)
        let d_1 = self_dot(&t1, 0, 1, 1, 1, 3)?;
        assert_eq!(Tensor::from_slice(&[20., 2000., 200000.]), d_1);

        //(1,2,3) x
        //(3,4,5)
        let d_2 = self_dot(&t1, 0, 2, 1, 1, 3)?;
        assert_eq!(Tensor::from_slice(&[26., 2600., 260000.]), d_2);

        //(1,2,3) x
        //(4,5,6) = 26-(1*3+2*3+3*3)+(1*4+2*4+3*4)
        let d_3 = self_dot(&t1, 0, 3, 1, 1, 3)?;
        assert_eq!(Tensor::from_slice(&[32., 3200., 320000.]), d_3);


        Ok(())
    }

    #[test]
    fn test_motifs_match_vec(){
        let t1 = Instant::now();

        let n = 1_0;

        // let mut rows_vec:Box<Vec<Vec<f32>>> = Box::new(Vec::with_capacity(n));
        let mut t = Box::new(Tensor::zeros(&[n, n], (Kind::Uint8, Device::Cpu)));

        for i in (0..n) {
            if i % 1000 == 0 {
                debug!("i={}", i);
            }

            // std::io::stdout().flush().expect("flush");
            let mut row_vec:Vec<u8> = Vec::with_capacity(n as usize);

            for j in (0..n) {
                row_vec.push((i * j) as u8)

            }
            // rows_vec.push(row_vec);
            t.index_put_(
                &[
                    Some(Tensor::from_slice(&[i])),//row
                ],
                &Tensor::from_slice(row_vec.as_slice()),
                false
            );
        }

        t.print();

        // debug!("mem size {}", size_of_val(&t.as_ptr()));
        debug!("done {} in {}", n, t1.elapsed().as_millis());

        // let t1_vec = from::<Vec<Vec<f32>>>(&t);
        // debug!("t1_vec={:?}", t1_vec.len());
        for i in (0..n){
            for j in (0..n) {
                print!("{:?} ", t.int64_value(&[i, j]));
            }
            println!();
        }


        // let mut t = Box::new(Tensor::from_slice(rows_vec.as_slice()));
        // debug!("done t {:?} in {}", t.size(), t1.elapsed().as_millis());
    }


    fn pulses_in_rand(n: i64, pulses: &[(usize, Vec<f64>)]) -> Vec<f64> {
        let random = random_v(n, 10., 1.);
        let pulses_len:usize = pulses.iter().map(|(offset, pulse)| *offset + pulse.len()).sum();

        random.into_iter().zip(
            pulses
                .into_iter()
                .flat_map(
                    |(offset, pulse)|
                        std::iter::repeat(0f64)
                            .take(*offset)
                            .chain(pulse.into_iter().map(|v|*v))
                )
                .chain(std::iter::repeat(0f64).take(n as usize-pulses_len))
        ).map(|(v1, v2)| 0.001 * v1 + v2).collect::<Vec<f64>>()
    }

    #[test]
    pub fn test_find_motifs_fft() -> Result<(), GenError>{
        let n = 1000i64;
        let x = lin_t(n);
        let gauss_1 = gauss_t(400., 100., n); //f.flip([0]);
        let gauss_2 = gauss_t(600., 100., n); //.flip([0]);

        let m = 200;

        let zeros = Tensor::zeros([m], (Kind::Float, Device::Cpu));

        let t1 = Instant::now();
        let mut corr:Vec<f32> = vec![];
        let mut dots:Vec<f32> = vec![];
        let mut dots_ff:Vec<f32> = vec![];
        // for d_1 in (0..n-m) {
        let d_1 = 0; {
            let t_1_snippet = gauss_1.f_slice(0, Some(d_1), Some((d_1+m) as i64), 1)?;
            let t_1_snippet_zn = z_norm(&t_1_snippet, 0);
            let t_1_snippet_ff = t_1_snippet.fft_fft(None, 0, "backward");

            for d_2 in (0..n-m) {
                let t_2_snippet = gauss_2.f_slice(0, Some(d_2 as i64), Some((d_2+m) as i64), 1)?;
                let t_2_snippet_zn = z_norm(&t_2_snippet, 0);
                let t_2_snippet_ff = t_2_snippet.fft_fft(None, 0, "backward");

                let dp = t_1_snippet_zn.dot(&t_2_snippet_zn);
                let dp_ff = t_1_snippet_ff.dot(&t_2_snippet_ff);
                // let t_2_snippet_zp = Tensor::hstack(&[t_2_snippet.view([-1]), zeros.view([-1])]);

                // corr.push(vec_f32_from(&d.abs()));
                dots.push(vec_f32_from(&dp)[0]/100.0);
                dots_ff.push(vec_f32_from(&dp_ff)[0]/200.0);
                // debug!("d.size()={:?}", dp.size());
            }
        }
        debug!("done in {}", t1.elapsed().as_millis());
        debug!("1. dots.len= {}", dots.len());
        dots.extend((0..n as usize -dots.len()).map(|_|0.).collect::<Vec<f32>>());
        debug!("2. dots.len= {}", dots.len());


        // let gauss_1_0 = gauss_1.f_slice(0, Some(300), Some(500), 1)?;

        crate::utils::plot_series(
            vec![
                (format!("G1"), BLACK, vec_f32_from(&x).into_iter().zip(vec_f32_from(&gauss_1)).collect::<Vec<(f32, f32)>>()),
                // (format!("G1"), BLACK, vec_f32_from(&lin_t(200)).into_iter().zip(vec_f32_from(&gauss_1_0)).collect::<Vec<(f32, f32)>>()),
                (format!("G2"), BLUE, vec_f32_from(&x).into_iter().zip(vec_f32_from(&gauss_2)).collect::<Vec<(f32, f32)>>()),
                (format!("dots"), GREEN, dots.into_iter().enumerate().map(|(i, v)|(i as f32, v)).collect::<Vec<(f32, f32)>>()),
                (format!("dots_ff"), RED, dots_ff.into_iter().enumerate().map(|(i, v)|(i as f32, v)).collect::<Vec<(f32, f32)>>())

            ], "EEG", (1000, 480), format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        ).unwrap();
        // find_motifs();
        println!("ok");

        Ok(())
    }

    struct TensorsSource {
        tensors:Vec<Tensor>
    }

    impl TensorsSource {
        pub fn new(tensors: Vec<Tensor>) -> Self {
            Self { tensors }
        }

        fn get_slices(&self,  i_0: i64, i_1: i64, m: i64, dim: i64) -> Result<(Tensor, Tensor), GenError> {
            Ok((
                self.tensors[0].f_slice(dim, Some(i_0), Some(i_0 + m), 1)?,
                self.tensors[0].f_slice(dim, Some(i_1), Some(i_1 + m), 1)?
            ))
        }

    }

    #[test]
    pub fn test_find_motifs() -> Result<(), GenError>{
        let n = 1_000i64;
        let x = lin_t(n);
        let sig_num = 2;
        // let sig_1 = gauss_t(400., 100., n); //f.flip([0]);
        // let sig_2 = gauss_t(600., 100., n); //.flip([0]);

        let (sig_0_1, sig_0_2) = two_sigs(1_000);
        println!("sig_1.size()={:?}, sig_2.size()={:?}", sig_0_1.size(), sig_0_2.size());
        let sig_1_c = sig_0_1.copy();
        let sig_2_c = sig_0_2.copy();
        let t_0 = Tensor::vstack(&[sig_0_1, sig_0_2]);
        println!("t.size={:?}", t_0.size());

        let (sig_1_1, sig_1_2) = two_sigs(1_000);
        let t_1 = Tensor::vstack(&[sig_1_1, sig_1_2]);



        let m = 100;
        let step = 10;


        let t1 = Instant::now();


        let mut mt = crate::tests::MotifTracker::new(0.9, "/tmp/test_motifs.csv");

        let dim = 1;


        let ts = crate::tests::TensorsSource::new(vec![t_0, t_1]);
        for i_0 in (0..n-m).step_by(step) {
            // let d_1 = 200; {
            // let t_1_snippet = sig_1.f_slice(0, Some(d_1), Some((d_1+m) as i64), 1)?;

            for i_1 in (i_0..n-m).step_by(10).skip(1) {
                debug!("i_0={:?}, i_1={:?} m={}", i_0, i_1, m);

                // let t_2_snippet = norm(&sig_2.f_slice(0, Some(d_2 as i64), Some((d_2+m) as i64), 1)?);

                // let dp = norm(&t_1_snippet).dot(&t_2_snippet);


                let (slice_01, slice_02) = ts.get_slices(i_0, i_1, m, dim)?;

                let dp = self_dot_norm(dim, &slice_01, &slice_02)?;

                // dp.print();

                // println!("i_0={:?}, i_1={:?}, dp={:?} ", i_0, i_1, dp.double_value(&[0]));
                // row_dots.push(dp.double_value(&[0]));
                mt.track(TensorSliceIdx(0, i_0), TensorSliceIdx(0, i_1), dp.double_value(&[0]) as f32);

            }
            mt.next();
            // dots.push(row_dots);
        }
        mt.dump();
        debug!("done in {}", t1.elapsed().as_millis());
        // dots.extend((0..n as usize -dots.len()).map(|_|0.).collect::<Vec<f32>>());
        // debug!("2. dots.len= {}", dots.len());


        // let gauss_1_0 = gauss_1.f_slice(0, Some(300), Some(500), 1)?;

        crate::utils::plot_series(
            vec![
                (format!("G1"), BLACK, vec_f32_from(&x).into_iter().zip(vec_f32_from(&sig_1_c)).collect::<Vec<(f32, f32)>>()),
                // (format!("G1"), BLACK, vec_f32_from(&lin_t(200)).into_iter().zip(vec_f32_from(&gauss_1_0)).collect::<Vec<(f32, f32)>>()),
                // (format!("G2"), BLUE, vec_f32_from(&x).into_iter().zip(vec_f32_from(&sig_2_c)).collect::<Vec<(f32, f32)>>()),
                // (format!("dots"), MAGENTA, dots.into_iter().enumerate().map(|(i, v)|(i as f32, v)).collect::<Vec<(f32, f32)>>())

            ], "EEG", (1000, 480), format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        ).unwrap();
        // find_motifs();
        println!("ok");

        Ok(())
    }

    #[test]
    pub fn test_motiffs_iter(){
        fn load_sig_tensor(key:&String) ->Box<Tensor>{
            let (sig_1_1, sig_1_2) = two_sigs(1000);
            Box::new(Tensor::vstack(&[sig_1_1, sig_1_2]))
        }

        let ts_iter = crate::tensor_iter::TensorsIter::new(
            Box::new(crate::tensor_iter::CacheOnThreadsStore::new(load_sig_tensor, vec!["sig_0".into()])),
            1, 100, 10
        );


        let mt = Arc::new(Mutex::new(MotifTracker::new(0.9, "/tmp/test_motifs.csv")));

        ts_iter.par_bridge().map(|r|(Arc::clone(&mt), r)).for_each(|(mt, r)| {
            match r {
                Ok((t_s_0, t_s_1)) => {
                    // v.0.tensordot(&v.1, 1, 1)
                    let dp = self_dot_norm(1, &t_s_0.1, &t_s_1.1).unwrap();
                    mt.lock().expect("mt").track(t_s_0.0, t_s_1.0, dp.double_value(&[0]) as f32);
                }
                Err(e) => {
                    panic!("v={:?}", e);
                }
            };
        });

        println!("mt={:?}", mt);
    }


    pub fn two_sigs(n: i64) -> (Tensor, Tensor) {
        let sig_1 = Tensor::from_slice(
            pulses_in_rand(n, &[
                // (10, pulse_v(8., 3., 20, 2.)),
                // (30, pulse_v(8., 3., 20, 2.))
                (100, pulse_v(80., 30., 200, 20.)),
                (300, pulse_v(80., 30., 200, 20.))
            ]).as_slice()
        );

        let sig_2 = Tensor::from_slice(
            pulses_in_rand(n, &[]).as_slice()
        );
        (sig_1, sig_2)
    }

    fn gauss_t(μ: f64, σ: f64, n: i64) -> Tensor {
        Tensor::from(gauss_i(μ, σ, n).collect::<Vec<_>>().as_slice())
    }

    fn gauss_i(μ: f64, σ: f64, n: i64) -> impl Iterator<Item=f64> {
        (0..n).map(move |x| (-0.5 * ((x as f64 - μ) / σ).powi(2)).exp())
    }

    fn pulse_t(μ: f64, σ: f64, n: i64, f: f64) -> Tensor {
        Tensor::from(pulse_v(μ, σ, n, f).as_slice())
    }

    fn pulse_v(μ: f64, σ: f64, n: i64, f: f64) -> Vec<f64> {
        gauss_i(μ, σ, n).enumerate().map(|(i, x)| (f * 2. * PI *  i as f64 / n as f64).sin() * x).collect::<Vec<_>>()
    }


    fn random_v(n: i64, amplitude: f64, max_slope: f64) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut signal = Vec::with_capacity(n as usize);

        let mut current_value = 0.0;
        let mut prev_value = 0.0;

        for _ in 0..n {
            let slope = rng.gen_range(-max_slope..max_slope);
            let next_value = current_value + slope;

            // Check if the slope exceeds the maximum allowed slope
            if (next_value - prev_value).abs() > max_slope {
                let direction = if next_value > prev_value { -1.0 } else { 1.0 };
                current_value = prev_value + direction * max_slope;
            } else {
                current_value = next_value;
            }

            signal.push(current_value * amplitude);
            prev_value = current_value;
        }

        signal
    }

    fn lin_t(n: i64) -> Tensor {
        Tensor::from((0..n).map(|x| x as f32).collect::<Vec<f32>>().as_slice())
    }



}