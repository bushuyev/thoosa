use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::Mutex;
use std::time::Instant;

use clap::clap_derive::{Args, Parser, Subcommand};
use log::{debug, info};
use rayon::iter::{ParallelBridge, ParallelIterator};
use tch::Tensor;

use hbac_common::HbacError;
use hbac_common::utils::*;

use crate::TensorSliceIdx;

#[derive(Debug, Args)]
pub struct FindMotifsParams {
    #[arg(long, default_value_t = 1000)]
    step: i64,

    #[arg(long, default_value_t = 1000)]
    window: i64,

    #[arg(long, default_value_t = 0.9)]
    min_dot: f32,

    #[arg(long, default_value = "seizure_vote")]
    kind: String,

    #[arg(long, default_value = "../../data_local/motifs")]
    motifs_dir: String,

    #[arg(long, default_value_t = 10)]
    preload_d: usize,

    #[arg(long, default_value_t = 10)]
    max_keys_fifo: usize,

}

impl FindMotifsParams {
    pub fn kind(&self) -> &str {
        &self.kind
    }
}

pub fn find_motifs<F>(loader:F, cli: &FindMotifsParams, records_num:i64, eegs: impl Iterator<Item=Option<i64>>) -> Result<(), HbacError> where F:Fn(&String) -> Box<Tensor> + Sync + Send + 'static {

    let slots_to_compare = (records_num - cli.window) / cli.step;
    let slots_to_compare = slots_to_compare.pow(2) / 2 - slots_to_compare;

    let motifs_dir = format!("{}/{}", cli.motifs_dir, cdt());

    if let Err(e) = std::fs::create_dir_all(motifs_dir.clone()) {
        panic!("can't create {}: {:?}", motifs_dir, e);
    }


    let eeg_ids_str = eegs.map(|v| v.unwrap().to_string()).collect::<Vec<String>>();
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(false)
        .open(format!("{}/{}_eegs.txt", motifs_dir, cli.kind))
        .unwrap();

    write!(file, "{:?}", eeg_ids_str);

    find_motifs_iter(
        loader,
        eeg_ids_str,
        cli.window,
        cli.step,
        (0..20).map(|_| cli.min_dot).collect(),
        slots_to_compare as usize,
        motifs_dir.as_str(),
        cli.preload_d,
        cli.max_keys_fifo,
    );
    Ok(())
}

pub fn find_motifs_iter<F>(loader: F, keys: Vec<String>, m: i64, step: i64, min_dot: Vec<f32>, slots_to_process:usize, motifs_dir:&str, preload_d:usize, max_keys_fifo: usize) where F:Fn(&String) -> Box<Tensor> + Sync + Send + 'static {

    let all_mt = min_dot
        .into_iter()
        .enumerate()
        .map(|(i, min_dot)|Mutex::new(SimpMotifTracker::new(min_dot, format!("{}/motifs_{}.csv", motifs_dir.clone(), i).as_str())))
        .collect::<Vec<Mutex<SimpMotifTracker>>>();

    let ts_iter = crate::tensor_iter::TensorsIter::new(
        Box::new(crate::tensor_iter::CacheOnThreadsStore::new(loader, keys, preload_d, max_keys_fifo)),
        1, m, step
    );

    let t1 = Instant::now();

    ts_iter.par_bridge().for_each(|r| {
        match r {
            Ok((t_s_0, t_s_1, slots_processed)) => {

                let dp = dot_norm(1, &t_s_0.1, &t_s_1.1).unwrap();

                for (i, mt) in all_mt.iter().enumerate() {
                    mt.lock().expect("mt").track(t_s_0.0, t_s_1.0, dp.double_value(&[i as i64]) as f32);
                }

                if slots_processed > 0 && slots_processed % 10_000 == 0 {
                    for (i, mt) in all_mt.iter().enumerate() {
                        mt.lock().expect("mt").dump();
                    }

                    let slots_per_sec = (slots_processed as u64 / t1.elapsed().as_secs()) as usize;
                    info!("pace: {} num_threads: {} slots_processed: {} out of {}, remaining {} min", slots_per_sec, rayon::current_num_threads(), slots_processed, slots_to_process, ((slots_to_process-slots_processed as usize)/(slots_per_sec+1))/60)
                }

            }
            Err(e) => {
                panic!("v={:?}", e);
            }
        };
    });

    for (i, mt) in all_mt.iter().enumerate() {
        mt.lock().expect("mt").dump();
    }

    debug!("saved to {:?}", motifs_dir);
}


#[derive(Debug)]
pub(crate) struct SimpMotifTracker {
    motifs:Vec<(TensorSliceIdx, TensorSliceIdx, f32)>,//i_0, match_0, match_1, dot
    min_dot: f32,
    i_0: TensorSliceIdx,
    file:File,
}

impl SimpMotifTracker {
    pub fn new(min_dot:f32, path: &str) -> Self {
        match std::fs::remove_file(path) {
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
            min_dot,
            motifs: vec![],
            i_0: TensorSliceIdx(0, 0),
            file,
        }
    }

    pub fn track(&mut self, i_0: TensorSliceIdx, i_1: TensorSliceIdx, dot:f32) {
        if dot > self.min_dot {
            self.motifs.push((i_0, i_1, dot));
        }
    }

    fn dump(&mut self) {
        for motif in self.motifs.iter() {
            writeln!(self.file, "{}, {},  {:.3}", motif.0, motif.1, motif.2);
        }
        self.file.flush();
        self.motifs.clear();

    }


    pub fn motifs(&self) -> &Vec<(TensorSliceIdx, TensorSliceIdx, f32)> {
        &self.motifs
    }
}

#[derive(Debug)]
pub(crate) struct MotifTracker {
    motifs:Vec<(TensorSliceIdx, TensorSliceIdx, TensorSliceIdx, f32)>,//i_0, match_0, match_1, dot
    motif_start_op:Option<TensorSliceIdx>,//tensor, slice
    min_dot: f32,
    i_0: TensorSliceIdx,
    file:File,
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
        for motif in self.motifs.iter() {
            writeln!(self.file, "{}, {}, {},  {:.3}", motif.0, motif.1, motif.2, motif.3);
        }
        self.file.flush();
        self.motifs.clear();

    }

    pub fn motifs(&self) -> &Vec<(TensorSliceIdx, TensorSliceIdx, TensorSliceIdx, f32)> {
        &self.motifs
    }
}

#[cfg(test)]
mod motif_find_tests {
    use std::collections::{HashMap, VecDeque};
    use std::f64::consts::PI;
    use std::fs::{File, OpenOptions};
    use std::{fs, io};
    use std::io::Write;
    use std::iter::Map;
    use std::mem::size_of_val;
    use std::ops::Mul;
    use std::ops::Range;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;
    use plotters::prelude::*;
    use tch::IndexOp;

    use chrono::Utc;
    use log::{debug, LevelFilter, SetLoggerError};
    use rand::Rng;
    use rayon::iter::IntoParallelIterator;
    use rayon::iter::ParallelIterator;
    use rayon::prelude::ParallelBridge;
    use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger};
    use tch::{Device, Kind, kind, Tensor};
    use tch::Kind::{Float, Int};


    use crate::*;
    use crate::find::MotifTracker;
    use hbac_common::utils::*;
    use crate::fragments::{list_motifs, write_motifs_slices};

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

        let mut mt = super::MotifTracker::new(0.8, "/tmp/test_motifs.csv");

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
    pub fn test_norm() -> Result<(), HbacError>{
        let t = Tensor::from_slice(&[1., 2., 3.]);
        let norm = norm(&t, &[-1]);

        assert_eq!(1.0, vec_f32_from(&norm.dot(&norm))[0]);


        let t = Tensor::vstack(&[
            Tensor::from_slice(&[1., 2., 3.]),
            Tensor::from_slice(&[1., 2., 3.])
        ]);

        let norm = norm_2(&t, 1);

        assert_eq!(Tensor::from_slice(&[1., 1.]),norm.tensordot(&norm, 1, 1).diag(0));

        Ok(())
    }



    #[test]
    pub fn test_pulses_in_rand() -> Result<(), HbacError>{
        let n = 1000i64;
        let x = lin_t(n);
        // let gauss_1 = gauss_t(400., 100., n);

        let to_pulses_in_random = pulses_in_rand(
            n,
            &[
                (100, pulse_v(80., 20., 200, 20.)),
                (100, pulse_v(80., 20., 200, 20.))
            ], 10.
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

        plot_series(
            vec![
                (format!("G1"), BLACK, vec_f32_from(&x).into_iter().zip(vec_f32_from(&sig)).collect::<Vec<(f32, f32)>>()),

            ], "EEG", (1000, 500), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
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

    #[test]
    fn test_self_dot() -> Result<(), HbacError>{
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
    fn test_self_dot_norm() -> Result<(), HbacError> {
        let t1 = Tensor::from_slice2(&[
            &[1., 2., 3., 4., 5., 6., 7., 8., 9.],
            &[10., 20., 30., 40., 50., 60., 70., 80., 90.],
            &[100., 200., 300., 400., 500., 600., 700., 800., 900.]
        ]);

        //(1,2,3) x
        //(1,2,3)
        let slice_1 = t1.f_slice(1, Some(0), Some(3), 1)?; //1, 2, 3 - normilzed -  (0.267261, 0.534522, 0.801784)
        let slice_2 = t1.f_slice(1, Some(3), Some(6), 1)?; //4, 5, 6 - (0.455842, 0.569803, 0.683763)

        let d_0 = dot_norm(1, &slice_1, &slice_2)?;

        assert_eq!(format!("{:.4}", d_0.double_value(&[0])), "0.9746");
        assert_eq!(format!("{:.4}", d_0.double_value(&[1])), "0.9746");
        assert_eq!(format!("{:.4}", d_0.double_value(&[2])), "0.9746");

        Ok(())
    }

    fn pulses_in_const(n: i64, pulses: &[(usize, Vec<f64>)]) -> Vec<f64> {
        let pulses_len:usize = pulses.iter().map(|(offset, pulse)| *offset + pulse.len()).sum();

        std::iter::repeat(0.).take(n as usize).zip(
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


    fn pulses_in_rand(n: i64, pulses: &[(usize, Vec<f64>)], amplitude: f64) -> Vec<f64> {
        let random = random_v(n, amplitude, 1.);
        let pulses_len:usize = pulses.iter().map(|(offset, pulse)| *offset ).sum();

        join_with_offset::<f64>(n, 0., 0.001, pulses, random)
    }

    fn join_with_offset<T>(n: i64, def:T, scale:T, pulses: &[(usize, Vec<T>)], main: Vec<T>) -> Vec<T> where T:Copy + Clone + Mul<Output = T> + std::ops::Add<Output = T>{
        main.into_iter().zip(
            pulses
                .into_iter()
                .flat_map(
                    |(offset, pulse)|
                        std::iter::repeat(def)
                            .take(*offset)
                            .chain(pulse.into_iter().map(|v| *v))
                )
                .chain(std::iter::repeat(def).take(n as usize - pulses.len()))
        ).map(|(v1, v2)| (scale * v1) + v2).collect::<Vec<T>>()
    }

    #[test]
    pub fn test_find_motifs_fft() -> Result<(), HbacError>{
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

        plot_series(
            vec![
                (format!("G1"), BLACK, vec_f32_from(&x).into_iter().zip(vec_f32_from(&gauss_1)).collect::<Vec<(f32, f32)>>()),
                // (format!("G1"), BLACK, vec_f32_from(&lin_t(200)).into_iter().zip(vec_f32_from(&gauss_1_0)).collect::<Vec<(f32, f32)>>()),
                (format!("G2"), BLUE, vec_f32_from(&x).into_iter().zip(vec_f32_from(&gauss_2)).collect::<Vec<(f32, f32)>>()),
                (format!("dots"), GREEN, dots.into_iter().enumerate().map(|(i, v)|(i as f32, v)).collect::<Vec<(f32, f32)>>()),
                (format!("dots_ff"), RED, dots_ff.into_iter().enumerate().map(|(i, v)|(i as f32, v)).collect::<Vec<(f32, f32)>>())

            ], "EEG", (1000, 480), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        ).unwrap();
        // find_motifs();
        println!("ok");

        Ok(())
    }


    pub struct TensorsSource {
        tensors:Vec<Tensor>
    }

    impl TensorsSource {
        pub fn new(tensors: Vec<Tensor>) -> Self {
            Self { tensors }
        }

        fn get_slices(&self,  i_0: i64, i_1: i64, m: i64, dim: i64) -> Result<(Tensor, Tensor), HbacError> {
            Ok((
                self.tensors[0].f_slice(dim, Some(i_0), Some(i_0 + m), 1)?,
                self.tensors[0].f_slice(dim, Some(i_1), Some(i_1 + m), 1)?
            ))
        }

    }

    #[test]
    pub fn test_eeg_select_by_corrcoef() -> Result<(), HbacError> {
        save_good_motifs(
            "../../data_local/motif_tensors/seizure_vote_eegs.pt",
            HashMap::from([
                ("/media/bu/KINGSTON/eegs.pt/seizure_vote_eegs.txt/", 1.),
                ("/media/bu/KINGSTON/eegs.pt/gpd_vote_eegs.txt/", 0.),
                ("/media/bu/KINGSTON/eegs.pt/grda_vote_eegs.txt/", 0.),
                ("/media/bu/KINGSTON/eegs.pt/lpd_vote_eegs.txt/", 0.),
                ("/media/bu/KINGSTON/eegs.pt/lrda_vote_eegs.txt/", 0.),
                ("/media/bu/KINGSTON/eegs.pt/other_vote_eegs.txt/", 0.)
            ]), 10,
            "../../data_local/motif_tensors/seizure_vote_eegs_good_idx.pt"
        )
    }

    fn save_good_motifs(motifs_path: &str, tensor_paths_labels:HashMap<&str, f32>, i1: i32, save_to: &str) -> Result<(), HbacError> {
        let all_zero = Tensor::zeros(1000, (tch::Kind::Float, Device::Cpu));

        // for f in fs::read_dir("/tmp/eegs_for_seizure_fragments_3_1/")? {
        let motifs = Tensor::load(motifs_path)?;
        println!("motifs={:?}", motifs.size());

        let channels_num = motifs.size()[0];
        let motifs_num = motifs.size()[1];

        let mut channels_motifs_classes: Vec<Vec<(Vec<f64>, Vec<f64>)>> = Vec::with_capacity(channels_num); // (Vec<f64>, Vec<f64>) - conv. value, class label
        for i in 0..channels_num {

            let mut motifs_classes = Vec::with_capacity(motifs_num);
            for i in 0..motifs_num {
                motifs_classes.push((Vec::with_capacity(400 * tensor_paths_labels.len()), Vec::with_capacity(400 * tensor_paths_labels.len())));//num of eegs of all classes
            }
            channels_motifs_classes.push(motifs_classes);
        }


        for (path, label) in tensor_paths_labels {
            dot_motifs(&motifs_num, &all_zero, &mut channels_motifs_classes, path, label, limit)?;
        }

        // println!("clc_motifs: {:?}", channels_motifs_classes);

        let mut channels_good_motifs = (0..channels_num).map(|_| vec![]).collect::<Vec<Vec<i32>>>();

        for motif_i in 0..motifs_num {
            for channel_i in 0..channels_num {
                let vals_labels_t = Tensor::vstack(&[
                    Tensor::from_slice(channels_motifs_classes[channel_i][motif_i].0.as_slice()),
                    Tensor::from_slice(channels_motifs_classes[channel_i][motif_i].1.as_slice())
                ]);

                // println!("vals_labels_t={:?}", channels_motifs_classes[channel_i][motif_i]);
                let corr = vals_labels_t.corrcoef().double_value(&[0, 1]);

                if corr > 0. {
                    println!("+channel_i={}, motif_i={}, v={:.5}", channel_i, motif_i, corr);

                    channels_good_motifs[channel_i].push(motif_i as i32);
                } else if corr < 0. {
                    println!("-channel_i={}, motif_i={}, v={:.5}", channel_i, motif_i, corr);
                }
            }
        }

        let Some(max_motifs) = channels_good_motifs.iter().map(|c| c.len()).max() else {
            return Err("No good motifs".into());
        };

        let good_motifs = Tensor::vstack(channels_good_motifs.into_iter().map(|mut good_motifs_idxs| {
            for _ in good_motifs_idxs.len()..max_motifs {
                good_motifs_idxs.push(-1);
            }
            Tensor::from_slice(good_motifs_idxs.as_slice())
        }).collect::<Vec<Tensor>>().as_slice());
        // good_motifs.print();

        good_motifs.save(save_to);

        //

        Ok(())
    }

    fn dot_motifs(motifs: &Tensor, all_zero: &Tensor, channels_motifs_classes: &mut Vec<Vec<(Vec<f64>, Vec<f64>)>>, eeg_path: &str, class_label: f64, limit: i32) -> Result<(), HbacError> {
        let mut count = 0;
        for f in fs::read_dir(eeg_path)? {
            let e = f?;
            let eeg = Tensor::load(e.path())?;

            let matches = clc_dot_matches(&eeg, &motifs, 1000, 0.97)?;

            for motif_i in 0..120 {//TODO
                for channel_i in 0..20 {
                    let motif_t = motifs.i((channel_i as i64, motif_i as i64));

                    let v_label = if motif_t.equal(&all_zero) {
                        (0., class_label)
                    } else {
                        let v = matches.i((channel_i as i64, motif_i as i64));
                        let conv_val = v.double_value(&[]);
                        // println!("channel_i={:?}, motif_i={:?}, v={:?}", channel_i,motif_i, conv_val);

                        (conv_val, class_label)
                    };

                    channels_motifs_classes[channel_i][motif_i].0.push(v_label.0);
                    channels_motifs_classes[channel_i][motif_i].1.push(v_label.1);
                }
            }

            if count > limit {
                break
            }
            count +=1;

        }
        Ok(())
    }


    fn conv_motifs(motifs: &Tensor, all_zero: &Tensor, channels_motifs_classes: &mut Vec<Vec<(Vec<f64>, Vec<f64>)>>, eeg_path: &str, class_label: f64) -> Result<(), HbacError> {
        let motifs_flat = motifs.view((20*120, 1, 1000)); //.transpose(0, 1);
        println!("motifs={:?}", motifs_flat.size());

        let mut count = 0;
        for f in fs::read_dir(eeg_path)? {
            let e = f?;
            let eeg = Tensor::load(e.path())?;

            let conv_t = eeg.conv1d(&motifs_flat, None::<Tensor>, 10, 0, 1, 20);
            let pool_t = conv_t.max_pool1d(10, 1, 0, 1, false);

            let val_t = pool_t.view((20, 120, -1));

            for motif_i in 0..120 {//TODO
                for channel_i in 0..20 {
                    let motif_t = motifs.i((channel_i as i64, motif_i as i64));

                    let v_label = if motif_t.equal(&all_zero) {
                        (0., class_label)
                    } else {
                        let eeg_x_motif = val_t.i((channel_i as i64, motif_i as i64, ..));
                        let conv_val = eeg_x_motif.max().double_value(&[]);

                        (conv_val, class_label)
                    };

                    channels_motifs_classes[channel_i][motif_i].0.push(v_label.0);
                    channels_motifs_classes[channel_i][motif_i].1.push(v_label.1);
                }
            }

            if count > 100 {
                break
            }
            count +=1;

        }
        Ok(())
    }

    /**
    * series - [channel x data]
    * motifs = [channel x motif x data slice]
    */
    fn clc_dot_matches(series: &Tensor, motifs: &Tensor, step:usize, threshold: f32) -> Result<Tensor, HbacError> {

        let channels_num = series.size()[0];
        let series_len = series.size()[1];
        let motifs_num = motifs.size()[1];
        let motif_len = motifs.size()[2];

        //channels, motif, motif x series dots
        let mut matches:Vec<Vec<Vec<f32>>> = (0..channels_num).fold(Vec::new(), |mut a, v|{
            a.push((0..motifs_num).fold(Vec::new(), |mut a, _| {
                a.push(Vec::new());
                a
            }));
            a
        });

        for i in (0..=series_len-motif_len).step_by(step) {
            for m in 0..motifs_num {

                let dp = dot_norm(
                    1,
                    &series.i((.., i..i + motif_len)),
                    &motifs.i((.., m, ..))
                )?;
                for (channel, dot) in from::<Vec<f32>>(&dp).into_iter().enumerate() {
                    matches[channel][m as usize].push(dot);
                }

            }
        }

        let matches = Tensor::vstack(matches.into_iter().map(|channel_motifs| {

            Tensor::from_slice(channel_motifs.into_iter().map(|motifs_dots| {

                motifs_dots.into_iter().map(|v| if v> threshold { 1 } else {0}).sum()

            }).collect::<Vec<i32>>().as_slice())

        }).collect::<Vec<Tensor>>().as_slice());

        Ok(matches)
    }

    fn pool_eeg_vs_fragment(eeg: String, pulse: String) -> Result<Vec<f32>, HbacError> {
        let t = Tensor::load(eeg)?;
        let pulse = Tensor::load(pulse)?;
        // println!("t={:?}", t.size());
        // println!("pulse={:?}", pulse.size());

        let t_0 = t.get(3).view([1, -1]);

        // println!("t_0={:?}", t_0.size());

        let conv_t = t_0.view([1, -1]).conv1d(&pulse.view([1, 1, -1]), None::<Tensor>, 1000, 0, 1, 1);

        // println!("conv_t.size = {:?}", conv_t.size());

        let pool_t = conv_t.max_pool1d(10, 1, 0, 1, false);

        // println!("pool.size = {:?}", pool_t.size());

        let n = t_0.size()[1];
        let x = lin_t(n);
        Ok(vec_f32_from(&pool_t))
    }

    #[test]
    pub fn test_check_with_conv() -> Result<(), HbacError> {
        let n = 1_000i64;

        let pulse = Tensor::from_slice(pulse_v(80., 30., 200, 20.).as_slice());

        let sig_0_1  = Tensor::from_slice(
            pulses_in_rand(n, &[
                // (10, pulse_v(8., 3., 20, 2.)),
                // (30, pulse_v(8., 3., 20, 2.))
                (100, pulse_v(80., 30., 200, 20.)),
                (300, pulse_v(40., 30., 200, 20.))
            ], 50.).as_slice()
        );

        println!("sig_0_1.size = {:?}", sig_0_1.size());
        println!("pulse.size = {:?}", pulse.size());

        let sig_1_c = sig_0_1.copy();

        let conv_t = sig_1_c.view([1, -1]).conv1d(&pulse.view([1, 1, -1]), None::<Tensor>, 1, 0, 1, 1);

        println!("conv_t.size = {:?}", conv_t.size());

        let pool_t = conv_t.max_pool1d(10, 1, 0, 1, false);

        println!("pool.size = {:?}", pool_t.size());

        let x = lin_t(n);

        let conv_v = join_with_offset::<f32>(n, 0.0, 1., &[(((n-conv_t.size()[1])/2) as usize, vec_f32_from(&conv_t))], (0..1000).map(|x|0.).collect::<Vec<f32>>());
        let pool_v = join_with_offset::<f32>(n, 0.0, 1., &[(((n-pool_t.size()[1])/2) as usize, vec_f32_from(&pool_t))], (0..1000).map(|x|0.).collect::<Vec<f32>>());
        plot_series(
            vec![
                (format!("Sig"), BLACK, vec_f32_from(&x).into_iter().zip(vec_f32_from(&sig_1_c.g_mul_scalar(10.))).collect::<Vec<(f32, f32)>>()),
                (format!("CONV"), RED,  vec_f32_from(&x).into_iter().zip(conv_v).collect::<Vec<(f32, f32)>>()),
                (format!("POOL"), GREEN,  vec_f32_from(&x).into_iter().zip(pool_v).collect::<Vec<(f32, f32)>>()),

            ], "EEG", (n as u32, 480), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        )?;

        Ok(())
    }

    #[test]
    pub fn test_check_with_conv_eeg() -> Result<(), HbacError> {
        let channel = 0;
        let t_0 = Tensor::load("/media/bu/KINGSTON/eegs.pt/seizure_vote_eegs.txt/2088807520.pt")?.i((channel)).view((1, -1));
        let t_1 = Tensor::load("/media/bu/KINGSTON/eegs.pt/seizure_vote_eegs.txt/2088807520.pt")?.i((channel)).view((1, -1));
        // let t_1 = Tensor::load("/media/bu/KINGSTON/eegs.pt/other_vote_eegs.txt/155224651.pt")?.i((channel)).view((1, -1));


        let gauss = gauss_t(0., 10., 100).totype(Float);
        println!("t_0.size={:?} t_1.size={:?}", t_0.size(), t_1.size());


        let range = 1000..2000;
        let pulse = t_0.i((0, range.clone()));
        let pulse_x = t_0.i((0, range));

        let pulse_x_m = pulse_x.median().view([1]).f_double_value(&[0])?;
        let t_0_m = t_0.median().view([1]).f_double_value(&[0])?;

        println!("pulse_x_m={:?} t_0_m={}", pulse_x_m, t_0_m);

        // let pulse_x = z_norm(&pulse_x, 0); //Tensor::from_slice((0..1000).map(|_|t_0_m).collect::<Vec<f64>>().as_slice()).totype(Float).view([1, 1000]);
        let pulse_x = pulse_x.view([1, -1]).conv1d(&gauss.view([1, 1, -1]), None::<Tensor>, 1, 50, 1, 1); //Tensor::from_slice((0..1000).map(|_|t_0_m).collect::<Vec<f64>>().as_slice()).totype(Float).view([1, 1000]);
        let pulse_x = pulse_x./*g_mul_scalar(0.05).*/i((..,0..1000));

        let sig  = t_1;

        println!("pulse.size = {:?} pulse_x.size()={:?}", pulse.size(), pulse_x.size());


        println!("sig.size = {:?}", sig.size());

        let sig_n = z_norm(&sig.copy(), 1);

        // let n = pulse.size()[0] as i64;
        let n = sig.size()[1] as i64;

        let conv_t = z_norm(&sig, 1).view([1, -1]).conv1d(&pulse.view([1, 1, -1]), None::<Tensor>, 1, 0, 1, 1);
        let conv_x_t = z_norm(&sig, 1).view([1, -1]).conv1d(&pulse_x.view([1, 1, -1]), None::<Tensor>, 1, 0, 1, 1);
        let d_conv = (n - conv_t.size()[1]) / 2 ;

        // let conv_t_p = Tensor::hstack(&[
        //     Tensor::zeros([1, d_conv], (Kind::Float, Device::Cpu)),
        //     conv_t.view(conv_t.size().as_slice()),
        //     Tensor::zeros([1, d_conv + 1], (Kind::Float, Device::Cpu))
        // ]);
        // let deconv_t = conv_t.conv_transpose1d(&)
        // println!("conv_t.size = {:?} conv_t_p.size={:?}", conv_t.size(), conv_t_p.size());

        let conv_diff_t = conv_t - conv_x_t;



        // let pool_t = conv_t.max_pool1d(10, 1, 0, 1, false);

        // println!("pool.size = {:?}", pool_t.size());

        let x = lin_t(n);


        // let conv_v = join_with_offset::<f32>(n, 0.0, 1., &[(d_conv as usize, vec_f32_from(&conv_t))], (0..n).map(|x|0.).collect::<Vec<f32>>());
        // let pool_v = join_with_offset::<f32>(n, 0.0, 1., &[(((n-pool_t.size()[1])/2) as usize, vec_f32_from(&pool_t))], (0..n).map(|x|0.).collect::<Vec<f32>>());
        plot_series(
            vec![
                // (format!("Pulse"), BLACK, vec_f32_from(&x).into_iter().zip(vec_f32_from(&pulse)).collect::<Vec<(f32, f32)>>()),
                // (format!("Pulse X"), RED, vec_f32_from(&x).into_iter().zip(vec_f32_from(&pulse_x)).collect::<Vec<(f32, f32)>>()),
                // (format!("Sig"), BLACK, vec_f32_from(&x).into_iter().zip(vec_f32_from(&sig.g_mul_scalar(10.))).collect::<Vec<(f32, f32)>>()),
                // (format!("Sig Norm"), RED, vec_f32_from(&x).into_iter().zip(vec_f32_from(&sig_n.g_mul_scalar(10.))).collect::<Vec<(f32, f32)>>()),
                // (format!("CONV"), RED,  vec_f32_from(&x).into_iter().zip(conv_v).collect::<Vec<(f32, f32)>>()),
                // (format!("POOL"), GREEN,  vec_f32_from(&x).into_iter().zip(pool_v).collect::<Vec<(f32, f32)>>()),
                (format!("POOL"), GREEN,  vec_f32_from(&x).into_iter().zip(vec_f32_from(&conv_diff_t)).collect::<Vec<(f32, f32)>>()),

            ], "EEG", (n as u32, 480), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        )?;

        Ok(())
    }



    #[test]
    pub fn test_find_motifs() -> Result<(), HbacError>{
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
        let step = 100;


        let t1 = Instant::now();


        let mut mt = super::SimpMotifTracker::new(0.9, "/tmp/test_motifs.csv");

        let dim = 1;


        let ts = TensorsSource::new(vec![t_0, t_1]);
        for i_0 in (0..n-m).step_by(step) {

            for i_1 in (i_0..n-m).step_by(step).skip(1) {
                debug!("i_0={:?}, i_1={:?} m={}", i_0, i_1, m);

                let (slice_01, slice_02) = ts.get_slices(i_0, i_1, m, dim)?;

                let dp = dot_norm(dim, &slice_01, &slice_02)?;

                mt.track(TensorSliceIdx(0, i_0), TensorSliceIdx(0, i_1), dp.double_value(&[0]) as f32);

            }
            // mt.next();
            // dots.push(row_dots);
        }

        debug!("done in {}", t1.elapsed().as_millis());
        // dots.extend((0..n as usize -dots.len()).map(|_|0.).collect::<Vec<f32>>());
        // debug!("2. dots.len= {}", dots.len());


        // let gauss_1_0 = gauss_1.f_slice(0, Some(300), Some(500), 1)?;

        let motifs = mt.motifs();
        assert_eq!((TensorSliceIdx(0, 100), TensorSliceIdx(0, 600)), (motifs[0].0, motifs[0].1));
        assert_eq!((TensorSliceIdx(0, 200), TensorSliceIdx(0, 700)), (motifs[1].0, motifs[1].1));

        // mt.dump();

        plot_series(
            vec![
                (format!("G1"), BLACK, vec_f32_from(&x).into_iter().zip(vec_f32_from(&sig_1_c)).collect::<Vec<(f32, f32)>>()),
                // (format!("G1"), BLACK, vec_f32_from(&lin_t(200)).into_iter().zip(vec_f32_from(&gauss_1_0)).collect::<Vec<(f32, f32)>>()),
                // (format!("G2"), BLUE, vec_f32_from(&x).into_iter().zip(vec_f32_from(&sig_2_c)).collect::<Vec<(f32, f32)>>()),
                // (format!("dots"), MAGENTA, dots.into_iter().enumerate().map(|(i, v)|(i as f32, v)).collect::<Vec<(f32, f32)>>())

            ], "EEG", (1000, 480), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        )?;
        // find_motifs();
        println!("ok");

        Ok(())
    }

    #[test]
    fn test_conv(){

        let t_0 =Tensor::vstack(&[
            Tensor::from_slice(&[1.,  1.,  1.,  0.,  0.,  0. ]),
            Tensor::from_slice(&[2.,  2.,  2.,  0.,  0.,  0. ])
        ]); //view((2, 1, 6))
        let t_1 = Tensor::vstack(&[
            Tensor::from_slice(&[1.,  1.,  1.]),
            Tensor::from_slice(&[2.,  2.,  2.])
        ]).view((2, 1, 3)); //view((2, 1, 3))



        let c = t_0.conv1d(&t_1, None::<Tensor>, 1, 3, 1, 2);

        println!("t_0={:?} t_1={:?} c={:?}", t_0.size(), t_1.size(), c.size());
        c.print();

        assert_eq!(
            c,
            Tensor::vstack(&[
                Tensor::from_slice(&[ 0.,   1.,   2.,   3.,   2.,   1.,   0.,   0.,   0.,   0.,]),
                Tensor::from_slice(&[ 0.,   4.,   8.,  12.,   8.,   4.,   0.,   0.,   0.,   0.,])
            ])
        )

        //https://www.rapidtables.com/calc/math/convolution-calculator.html
        //1 1 1 0 0 0 * 1 1 1 = 1  2  3  2  1 0 0 0
        //2 2 2 0 0 0 * 2 2 2 = 4  8  12 8  4 0 0 0
    }

    #[test]
    fn test_conv_2(){

        let t_0 =Tensor::vstack(&[
            Tensor::from_slice(&[1.,  1.,  1.,  0.,  0.,  0. ]),// first channel
            Tensor::from_slice(&[2.,  2.,  2.,  0.,  0.,  0. ]) // second channel
        ]); //view((2, 1, 6))

        let t_1 = Tensor::vstack(&[
            Tensor::vstack(&[
                Tensor::from_slice(&[1.,  1.,  1.]),// 1 first channel kernel
                Tensor::from_slice(&[3.,  3.,  3.]) // 2 first channel kenrel
            ]),
            Tensor::vstack(&[
                Tensor::from_slice(&[2.,  2.,  2.]),// 1 second channel kernel
                Tensor::from_slice(&[4.,  4.,  4.]) // 2 second channel kernel
            ])
        ]).view((4, 1, 3));

        println!("t_0={:?} t_1={:?}", t_0.size(), t_1.size());

        let c = t_0.conv1d(&t_1, None::<Tensor>, 1, 3, 1, 2);

        println!("c={:?}", c.size());
        c.print();

        assert_eq!(
            c,
            Tensor::vstack(&[
                Tensor::from_slice(&[ 0.,   1.,   2.,   3.,   2.,   1.,   0.,   0.,   0.,   0.,]), // first channel * 1 kernel
                Tensor::from_slice(&[ 0.,   3.,   6.,   9.,   6.,   3.,   0.,   0.,   0.,   0.,]), // first channel * 2 kernel
                Tensor::from_slice(&[ 0.,   4.,   8.,  12.,   8.,   4.,   0.,   0.,   0.,   0.,]), // second channel * 1 kernel
                Tensor::from_slice(&[ 0.,   8.,  16.,  24.,  16.,   8.,   0.,   0.,   0.,   0.,])  // second channel * 2 kernel
            ])
        );

        let c_by_channel = c.view((2, -1, 10));

        println!("c={:?}", c_by_channel.size());
        c_by_channel.print();

        //https://www.rapidtables.com/calc/math/convolution-calculator.html
        //1 1 1 0 0 0 * 1 1 1 = 1  2  3  2  1 0 0 0
        //2 2 2 0 0 0 * 2 2 2 = 4  8  12 8  4 0 0 0
    }

    #[test]
    fn test_clc_dot_matches() -> Result<(), HbacError>{
        let t_0 = Tensor::vstack(&[
            Tensor::from_slice(&[1.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  1.]),//channel 0
            Tensor::from_slice(&[10., 10., 10., 20., 20., 20., 30., 30., 10.])//channel 1
        ]);

        let motifs_t = Tensor::vstack(&[
            Tensor::vstack(&[ //channel 0
                Tensor::from_slice(&[1.,  1.,  1.]),//motif 0
                Tensor::from_slice(&[1.,  2.,  2.]),//motif 1
                Tensor::from_slice(&[3.,  3.,  1.]),//motif 2
                Tensor::from_slice(&[1.,  7.,  1.]),//motif 3
            ]),
            Tensor::vstack(&[//channel 1
                Tensor::from_slice(&[10.,  10.,  10.]),//motif 0
                Tensor::from_slice(&[10.,  20.,  20.]),//motif 1
                Tensor::from_slice(&[30.,  30.,  10.]),//motif 2
                Tensor::from_slice(&[10.,  70.,  10.]),//motif 3
            ])
        ]).view((2, 4, -1));

        let matches_t = Tensor::vstack(&[
            Tensor::from_slice(&[ //channel 0
                2,//motif 0
                2,//motif 1
                1,//motif 2
                0,//motif 3
            ]),
            Tensor::from_slice(&[//channel 1
                2,//motif 0
                2,//motif 1
                1,//motif 2
                0,//motif 3
            ])
        ]).totype(Int);

        assert_eq!(matches_t, clc_dot_matches(&t_0, &motifs_t, 1, 0.99)?);

        println!("Ok");
        Ok(())
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
            pulses_in_rand(n, &[], 10.).as_slice()
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
}
