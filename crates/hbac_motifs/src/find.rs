use std::collections::HashMap;
use std::fmt::Debug;
use std::fs;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::Mutex;
use std::time::Instant;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::path::Path;

use clap::value_parser;
use clap::parser::*;
use clap::clap_derive::{Args, Parser, Subcommand};
use hbac_shared::parse_key_val;

use log::{debug, info, warn};
use rayon::iter::{ParallelBridge, ParallelIterator};
use rayon::iter::IntoParallelIterator;
use tch::{Device, IndexOp, Tensor};
use hbac_common::utils::*;

use hbac_shared::HbacError;
use hbac_common::utils::*;
use rayon::iter::IntoParallelRefIterator;
use torch_sys::C_tensor;
use hbac_shared::TensorSliceIdx;
use hbac_store::Store;
use hbac_store::StoreDumper;
use std::cell::RefCell;
use crate::cache_on_threads::CacheOnThreads;
use crate::distance::slice_t;
use crate::distance::dist_d2;

#[derive(Debug, Args)]
pub struct FindMotifsParams {
    #[arg(long, default_value_t = 1000)]
    step: usize,

    #[arg(long, default_value_t = 1000)]
    window: usize,

    #[arg(long, default_value_t = 0.9)]
    min_dot: f32,

    #[arg(long, default_value = "seizure_vote")]
    kind: String,

    #[arg(long, default_value = "../../data_local/motifs")]
    descr: String,

    #[arg(long, default_value_t = 10)]
    preload_d: usize,

    #[arg(long, default_value_t = 100)]
    max_keys_fifo: usize,

}

impl FindMotifsParams {
    pub fn kind(&self) -> &str {
        &self.kind
    }
}


pub fn find_motifs_euc<F>(loader:F, find_motifs_params: &FindMotifsParams, records_num:usize, eegs: impl Iterator<Item=Option<i64>>, store:Arc<Mutex<impl Store + 'static>>, channels: i64) -> Result<(), HbacError>
    where F:Fn(&String) -> Box<Tensor> + Sync + Send + 'static {


    let slots_to_process = (records_num - find_motifs_params.window) / find_motifs_params.step;
    let slots_to_process = slots_to_process.pow(2) / 2 - slots_to_process; // n x n - diagonal


    let ids = eegs.filter(|e|e.is_some()).map(|e|e.unwrap()).collect::<Vec<i64>>();
    let eeg_ids_str = ids.iter().map(|v| v.to_string()).collect::<Vec<String>>();//TODO check if all eeg exist


    let min_dot = (0..channels).map(|_| find_motifs_params.min_dot).collect::<Vec<f32>>();
    let run_id = { store.lock()?.save_run(format!("find motifs in {} with min dots: {:?}", find_motifs_params.kind, min_dot).as_str(), &eeg_ids_str)}?;

    let mut all_motif_trackers = min_dot
        .into_iter()
        .enumerate()
        // .map(|(i, min_dot)|Mutex::new(SimpMotifTracker::<StoreDumper>::new(min_dot, StoreDumper::new(Arc::clone(&store), run_id, i))))//TODO name
        .map(|(i, min_dot)| SimpMotifTracker::<StoreDumper>::new(min_dot, StoreDumper::new(Arc::clone(&store), run_id, i)))//TODO name
        .collect::<Vec<SimpMotifTracker<_>>>();


    let mut cache = CacheOnThreads::new(loader, find_motifs_params.max_keys_fifo, Some(Box::new(move ||{
        // let preload_d = preload_d_rc_in_cache.fetch_add(1, Ordering::Relaxed);

        // warn!("on blocking: {:?}", preload_d)
    })));


    let v_w_0 = eeg_ids_str.chunks(20);
    let v_w_1 = eeg_ids_str.chunks(20);
    let mut c = 0;

    let t1 = Instant::now();
    let mut slots_processed = 0;

    v_w_0.enumerate().for_each(|(chunk_i_0, vw_0)|{
        let t_0_tensors = cache.blocking_get_1(vw_0).into_iter().map(|t|{
            debug!("t_0 size={:?}", t.size());
            let sliced = slice_t(&t, find_motifs_params.step as i32, find_motifs_params.window as i64);

            debug!("t_0 sliced size={:?}", sliced.size());

            sliced
        }).collect::<Vec<Tensor>>();
        let sizes_0 = t_0_tensors.iter().map(|t|t.size()[0]).collect::<Vec<_>>();
        // let tensor_ids_0 = sizes_0.iter().zip(vw_0.into_iter()).flat_map(|(size, id)| std::iter::repeat(id).take(*size as usize)).collect::<Vec<&String>>();

        // let t_0_sliced = z_norm(&Tensor::vstack(&t_0_tensors), 2);
        let t_0_sliced = Tensor::vstack(&t_0_tensors);

        debug!("t_0_sliced i_0={} merged={:?} sizes_0={:?}", chunk_i_0, t_0_sliced.size(), sizes_0);

        <std::slice::Chunks<'_, String> as Clone>::clone(&v_w_1).skip(chunk_i_0).enumerate().for_each(|(chunk_i_1, vw_1)| {
            let t_1_tensors = cache.blocking_get_1(vw_1).into_iter().map(|t|{
                debug!("t_0 size={:?}", t.size());

                let sliced = slice_t(&t, find_motifs_params.step as i32, find_motifs_params.window as i64);

                debug!("t_0 sliced size={:?}", sliced.size());

                sliced
            }).collect::<Vec<Tensor>>();
            let sizes_1 = t_1_tensors.iter().map(|t|t.size()[0]).collect::<Vec<_>>();
            // let tensor_ids_1 = sizes_1.iter().zip(vw_1.into_iter()).flat_map(|(size, id)| std::iter::repeat(id).take(*size as usize)).collect::<Vec<&String>>();

            // let t_1_sliced = z_norm(&Tensor::vstack(&t_1_tensors), 2);
            let t_1_sliced = Tensor::vstack(&t_1_tensors);
            debug!("t_1_sliced merged={:?} sizes_1={:?}", t_1_sliced.size(), sizes_1);

            for (c, mt) in all_motif_trackers.iter_mut().enumerate() {
                let tensor_0 = t_0_sliced.i((.., c as i64, ..)).squeeze();
                let tensor_1 = t_1_sliced.i((.., c as i64, ..)).squeeze();
                debug!("tensor_0.size={:?} tensor_1.size()={:?}", tensor_0.size(), tensor_1.size());
                let t = dist_d2(1, &tensor_0, &tensor_1).expect(format!("dist_d2 on channel={}, chunk_0={} chunk_1={}", c, chunk_i_0, chunk_i_1).as_str());
                // t.print();

                let mean = 0.05; // * t.mean(None).double_value(&[]) as f32;//TODO k
                // t.quantile()
                debug!("dist_d2: t={:?} mean={:?}", t.size(), mean);
                // for i_0 in 0..t.size()[0] {
                //     for i_1 in 0..t.size()[1] {
                //         debug!("t_0={:?} s_0={} t_1={:?} s_1:{} v={}", tensor_ids_0[i_0 as usize], i_0 * find_motifs_params.step as i64,  tensor_ids_1[i_1 as usize], i_1 * find_motifs_params.step as i64, t.double_value(&[i_0, i_1]));
                //     }
                // }
                let mut i_0 = 0;
                let mut i_1 = 0;

                // debug!("tensor_ids_0={:?}", tensor_ids_0);
                // debug!("tensor_ids_1={:?}", tensor_ids_1);

                for (t_idx_0, t_size_0) in sizes_0.iter().enumerate() {
                    //TODO clc i_1 to skip - sum tensor sizes
                    let skip_idx_1 = if chunk_i_0 == chunk_i_1 {
                        //if first chunk both in outer and inner cycles
                        let skip_ = (0..t_idx_0).map(|i|sizes_0[i]).sum();
                        debug!("skipping t_idx_0={} skip_={}", t_idx_0, skip_);
                        skip_
                    } else {
                        0
                    };

                    for s_0 in  0..*t_size_0 {
                        i_1 = skip_idx_1 as i64;

                        for (t_idx_1, t_size_1) in sizes_1.iter().enumerate().skip(t_idx_0) {

                            for s_1 in  0..*t_size_1 {
                                let dist = t.double_value(&[i_0, i_1]);

                                debug!("dist val: t_idx_0={} chunk_i_0={}, chunk_i_1={} t_0={:?} s_0={:3} t_1={:?} s_1:{:3} i_0={}, i_1={} v={}", t_idx_0,chunk_i_0, chunk_i_1,  vw_0[t_idx_0 as usize], s_0 * find_motifs_params.step as i64, vw_1[t_idx_1 as usize], s_1 * find_motifs_params.step as i64, i_0, i_1, dist);

                                i_1+= 1;
                                if  (t_idx_0 != t_idx_1 || s_0 != s_1) && (dist as f32) < mean {
                                    mt.track((vw_0[t_idx_0 as usize].clone(), s_0 * find_motifs_params.step as i64).into(), (vw_1[t_idx_1 as usize].clone(), s_1 * find_motifs_params.step as i64).into(), dist as f32)
                                }
                                slots_processed+=1;
                            }
                        }
                        i_0+= 1;
                    }

                    if slots_processed > 0 && slots_processed % 10_000 == 0 {

                        let slots_per_sec = (slots_processed as u64 / t1.elapsed().as_secs()) as usize;
                        let (remaining, overflow) = if slots_to_process > slots_processed {
                            ((slots_to_process - slots_processed)/(slots_per_sec +1), false)
                        }  else {
                            ((slots_processed - slots_to_process )/(slots_per_sec +1), true)
                        };
                        info!("pace: {} num_threads: {} slots_processed: {} out of {}, remaining {} min {}", slots_per_sec, rayon::current_num_threads(), slots_processed, slots_to_process, remaining/60, if overflow {"overflow"} else {""} )
                    }
                }
                mt.dump();
            }
        })
    });

    for i in 0..channels { //TODO channels len
        { store.lock()?.set_motifs(run_id, i as i64)}?;
    }

    info!("done {}/{}", slots_to_process, slots_processed);

    Ok(())
}

pub fn find_motifs<F>(loader:F, cli: &FindMotifsParams, records_num:usize, eegs: impl Iterator<Item=Option<i64>>, store:Arc<Mutex<impl Store + 'static>>, channels: i32) -> Result<(), HbacError>
    where F:Fn(&String) -> Box<Tensor> + Sync + Send + 'static {

    let slots_to_process = (records_num - cli.window) / cli.step;
    let slots_to_process = slots_to_process.pow(2) / 2 - slots_to_process; // n x n - diagonale


    let eeg_ids_str = eegs.map(|v| v.unwrap().to_string()).collect::<Vec<String>>();//TODO check if all eeg exist


    let min_dot = (0..channels).map(|_| cli.min_dot).collect::<Vec<f32>>();
    let run_id = { store.lock()?.save_run(format!("find motifs in {} with min dots: {:?}", cli.kind, min_dot).as_str(), &eeg_ids_str)}?;

    let all_motif_trackers = min_dot
        .into_iter()
        .enumerate()
        .map(|(i, min_dot)|Mutex::new(SimpMotifTracker::<StoreDumper>::new(min_dot, StoreDumper::new(Arc::clone(&store), run_id, i))))//TODO name
        .collect::<Vec<Mutex<SimpMotifTracker<_>>>>();

    let ts_iter = crate::tensor_iter::TensorsIter::new(
        Box::new(crate::tensor_iter::CacheOnThreadsStore::new(loader, eeg_ids_str, cli.preload_d, cli.max_keys_fifo)),
        1, cli.window, cli.step
    );

    let t1 = Instant::now();

    ts_iter.array_chunks::<3>().par_bridge().for_each(|rr| {
        debug!("find_motifs: in array_chunks, rr.len()={}", rr.len());
        for r in rr {
            match r {
                Ok((t_s_0, t_s_1, slots_processed)) => {
                    let dp = dot_norm(1, &t_s_0.1, &t_s_1.1).unwrap();

                    debug!("find_motifs: t_s_0.1.size()={:?}, t_s_0.1.device()={:?}, t_s_1.1.size()={:?}, t_s_1.1.device()={:?} dp.size()={:?}, dp.device()={:?}", t_s_0.1.size(), t_s_0.1.device(), t_s_1.1.size(), t_s_1.1.device(), dp.size(), dp.device());

                    for (channel, mt) in all_motif_trackers.iter().enumerate() {
                        mt.lock().expect("mt").track(t_s_0.0.clone(), t_s_1.0.clone(), dp.double_value(&[channel as i64]) as f32);
                    }

                    if slots_processed > 0 && slots_processed % 10_000 == 0 {
                        for (i, mt) in all_motif_trackers.iter().enumerate() {
                            mt.lock().expect("mt").dump();
                        }

                        let slots_per_sec = (slots_processed as u64 / t1.elapsed().as_secs()) as usize;
                        info!("pace: {} num_threads: {} slots_processed: {} out of {}, remaining {} min", slots_per_sec, rayon::current_num_threads(), slots_processed, slots_to_process, ((slots_to_process-slots_processed)/(slots_per_sec+1))/60)
                    }
                }
                Err(e) => {
                    panic!("v={:?}", e);
                }
            };
        }
    });

    for (i, mt) in all_motif_trackers.iter().enumerate() {
        mt.lock().expect("mt").dump();
    }

    for i in 0..channels { //TODO channels len
        { store.lock()?.set_motifs(run_id, i as i64)}?;
    }

    Ok(())
}


pub trait MotifDumper {
    fn dump(&mut self, motifs:&mut Vec<(TensorSliceIdx<String>, TensorSliceIdx<String>, f32)>) -> Result<(), HbacError>;
}

#[derive(Debug)]
pub struct FileDumper {
    file:File,
}

impl FileDumper {
    pub fn new(path: &str) -> Self {
        /*match std::fs::remove_file(path) {
            Ok(_) => {}
            Err(_) => {}
        }*/

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(path)
            .unwrap();


        Self {
            file
        }
    }
}

impl MotifDumper for FileDumper {
    fn dump(&mut self, motifs:&mut Vec<(TensorSliceIdx<String>, TensorSliceIdx<String>, f32)>) -> Result<(), HbacError> {
        for motif in motifs.iter() {
            writeln!(self.file, "{}, {},  {:.3}", motif.0, motif.1, motif.2);
        }
        self.file.flush();

        Ok(())

    }
}


impl MotifDumper for StoreDumper {

    fn dump(&mut self, motifs:&mut Vec<(TensorSliceIdx<String>, TensorSliceIdx<String>, f32)>) -> Result<(), HbacError>{

        self.store.lock()?.save_slices(self.run_id, self.channel as i64,  motifs.into_iter().map(|v|((v.0.0.clone(), v.0.1), (v.1.0.clone(), v.1.1), v.2)).collect::<Vec<_>>())?;//TODO move TensorSliceIdx to shared

        Ok(())

    }
}


#[derive(Debug)]
pub(crate) struct SimpMotifTracker<D:MotifDumper+Debug> {
    motifs:Vec<(TensorSliceIdx<String>, TensorSliceIdx<String>, f32)>,//i_0, match_0, match_1, dot
    min_dot: f32,
    dumper:D
}

impl <D:MotifDumper+Debug> SimpMotifTracker<D> {

    pub fn new(min_dot:f32, dumper:D) -> SimpMotifTracker<D> {
        Self {
            min_dot,
            motifs: vec![],
            dumper
        }
    }

    pub fn track(&mut self, i_0: TensorSliceIdx<String>, i_1: TensorSliceIdx<String>, dot:f32) {

        // if dot > self.min_dot {
        //     debug!("SimpMotifTracker::track: i_0={:?}, i_1={:?}, dot={:?}", i_0, i_1, dot);

            self.motifs.push((i_0, i_1, dot));
        // }
    }

    fn dump(&mut self) {
        // debug!("SimpMotifTracker::dump: self.motifs={}", self.motifs.len());
        if self.motifs.len() > 0 {
            self.dumper.dump(&mut self.motifs).expect("dumper");
            self.motifs.clear();
        }
    }


    pub fn motifs(&self) -> &Vec<(TensorSliceIdx<String>, TensorSliceIdx<String>, f32)> {
        &self.motifs
    }
}

#[derive(Debug)]
pub(crate) struct MotifTracker {
    motifs:Vec<(TensorSliceIdx<String>, TensorSliceIdx<String>, TensorSliceIdx<String>, f32)>,//i_0, match_0, match_1, dot
    motif_start_op:Option<TensorSliceIdx<String>>,//tensor, slice
    min_dot: f32,
    i_0: TensorSliceIdx<String>,
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
            i_0: TensorSliceIdx("".to_string(), 0),
            file,
            last_dot: 0.
        }
    }

    pub fn track(&mut self, i_0: TensorSliceIdx<String>, i: TensorSliceIdx<String>, dot:f32) {
        match &self.motif_start_op {
            None => {
                if dot > self.min_dot {
                    self.motif_start_op = Some(i);
                    self.last_dot = dot;
                }
            }
            Some(motif_start) => {
                if dot <= self.min_dot {
                    self.motifs.push((i_0.clone(), motif_start.clone(), i, self.last_dot));
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

    pub fn motifs(&self) -> &Vec<(TensorSliceIdx<String>, TensorSliceIdx<String>, TensorSliceIdx<String>, f32)> {
        &self.motifs
    }
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


fn dot_motifs<I>(motifs: &Tensor, all_zero: &Tensor, channels_motifs_classes: Arc<Mutex<Vec<Vec<(Vec<f64>, Vec<i64>)>>>>, eegs_iter_f: I, class_label: i64, limit: Option<usize>, step: usize, threshold: f32) -> Result<(), HbacError>
    where I:Iterator<Item=Result<Tensor, HbacError>>, I: std::marker::Send {

    let mut count = AtomicU64::new(0);

    let channels_num = motifs.size()[0] as usize;
    let motifs_num = motifs.size()[1] as usize;

    debug!("dot_motifs: channels_num={} motifs_num={}", channels_num, motifs_num);

    for eeg in eegs_iter_f {

        let matches = clc_dot_matches(&eeg?, &motifs, step, threshold)?;

        for motif_i in 0..motifs_num {

            for channel_i in 0..channels_num {
                let motif_t = motifs.i((channel_i as i64, motif_i as i64));

                let num_matches__label = if motif_t.equal(&all_zero) {
                    (0., class_label)
                } else {
                    let v = matches.i((channel_i as i64, motif_i as i64));
                    let conv_val = v.double_value(&[]);

                     (conv_val, class_label)
                };

                if let Ok(mut channels_motifs_classes_guard) =  channels_motifs_classes.lock(){
                    channels_motifs_classes_guard[channel_i][motif_i].0.push(num_matches__label.0);
                    channels_motifs_classes_guard[channel_i][motif_i].1.push(num_matches__label.1);
                }
            }
        }

        if limit.map(|limit| count.load(Ordering::Relaxed) > limit as u64).unwrap_or(false) {
            return Ok(())
        }

        let count_ = count.fetch_add(1, Ordering::Relaxed);
        if count_ % 100 == 0 {
            info!("dot_motifs: count={}", count_);
        }
        debug!("dot_motifs: count={}", count_);
    }
    Ok(())
}

#[derive(Debug, Args)]
pub struct SelectMotifsParams {

    #[arg(long)]
    run_id: usize,

    #[arg(long,  default_value_t = 1000)]
    step: usize,

    #[arg(long,  default_value_t = 1000)]
    window: usize,

    #[arg(long, value_parser=parse_key_val::<i64, String>)]
    label_tensors: Vec<(i64, String)>,//TODO rename to label_kinds

}

impl SelectMotifsParams {

    pub fn label_tensors(&self) -> &Vec<(i64, String)> {
        &self.label_tensors
    }

    pub fn run_id(&self) -> usize {
        self.run_id
    }
    pub fn step(&self) -> usize {
        self.step
    }
    pub fn window(&self) -> usize {
        self.window
    }
}

pub fn select_motifs<I>(motifs: Tensor, labels_loaders:Vec<(i64, I)>, limit: Option<usize>, step: usize, device: Device) -> Result<Vec<Vec<(i64, f64)>>, HbacError>
    where I:Iterator<Item=Result<Tensor, HbacError>> + Send + Sync {
    let motifs_size = motifs.size();

    debug!("select_motifs: find_good_motifs: motifs={:?}", motifs_size);

    let channels_num = motifs_size[0];
    let motifs_num = motifs_size[1];
    let slice_len = motifs_size[2];

    let mut channels_motifs_classes: Vec<Vec<(Vec<f64>, Vec<i64>)>> = Vec::with_capacity(channels_num as usize); // (Vec<f64>, Vec<f64>) - conv. value, class label
    for i in 0..channels_num {

        let mut motifs_classes = Vec::with_capacity(motifs_num as usize);
        for i in 0..motifs_num {
            motifs_classes.push((Vec::with_capacity(400 * labels_loaders.len()), Vec::with_capacity(400 * labels_loaders.len())));//num of eegs of all classes
        }
        channels_motifs_classes.push(motifs_classes);
    }


    let motifs_ptr = motifs.as_ptr() as usize;
    let channels_motifs_classes_lock = Arc::new(Mutex::new(channels_motifs_classes));


    labels_loaders.into_iter().par_bridge().try_for_each(|(label, loader)|{
        dot_motifs(
            &unsafe {Tensor::clone_from_ptr(motifs_ptr as *mut C_tensor)},
            &Tensor::zeros(slice_len, (tch::Kind::Float, device)), //TODO
            Arc::clone(&channels_motifs_classes_lock),
            loader, label, limit, step, 0.97 //TODO step - k * slice_len
        )
    });

    let mut channels_good_motifs = (0..channels_num).map(|_| vec![]).collect::<Vec<Vec<(i64, f64)>>>();

    let Ok(channels_motifs_classes_guard) = channels_motifs_classes_lock.lock() else {
        panic!("select_motifs: can't lock channels_motifs_classes_lock");
    };

    debug!("select_motifs: channels_motifs_classes_guard={:?}", channels_motifs_classes_guard.len());

    for motif_i in 0..motifs_num {
        for channel_i in 0..channels_num {

            let vals_labels_t = Tensor::vstack(&[
                Tensor::from_slice(channels_motifs_classes_guard[channel_i as usize][motif_i as usize].0.as_slice()),
                Tensor::from_slice(channels_motifs_classes_guard[channel_i as usize][motif_i as usize].1.as_slice())
            ]);

            let corr = vals_labels_t.corrcoef().double_value(&[0, 1]);

            if corr > 0. {
                debug!("select_motifs: +channel_i={}, motif_i={}, v={:.5}", channel_i, motif_i, corr);

                channels_good_motifs[channel_i as usize].push((motif_i, corr));
            } else {
                debug!("select_motifs: -channel_i={}, motif_i={}, v={:.5}", channel_i, motif_i, corr);
            }
        }
    }

    let Some(max_motifs) = channels_good_motifs.iter().map(|c| c.len()).max() else {
        return Err("No good motifs".into());
    };

    Ok(channels_good_motifs)
    // Ok(Tensor::vstack(channels_good_motifs.into_iter().map(|mut good_motifs_idxs| {
    //     for _ in good_motifs_idxs.len()..max_motifs {
    //         good_motifs_idxs.push(-1);
    //     }
    //     Tensor::from_slice(good_motifs_idxs.as_slice())
    // }).collect::<Vec<Tensor>>().as_slice()))
}

pub fn iter_all_in_dir<P: AsRef<std::path::Path>>(path:P) -> Result<impl Iterator<Item=Result<Tensor, HbacError>>, HbacError> {

    fs::read_dir(path.as_ref()).map(|r|
        r.map(|e_r| e_r.and_then(|e| {
                debug!("select_motifs: load_all_in_dir");
                Tensor::load(e.path()).map_err(|e| std::io::Error::other(e.to_string()))
            }).map_err(|e| HbacError::new(e.to_string()))
        )
    ).map_err(|e| HbacError::new(format!("got {} while reading {:?} ", e, path.as_ref())))
}





#[cfg(test)]
mod motif_find_tests {
    use std::collections::{HashMap, VecDeque};
    use std::f64::consts::PI;
    use std::fs::{File, OpenOptions, ReadDir};
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
    use std::path::Path;

    use chrono::Utc;
    use log::{debug, LevelFilter, SetLoggerError};
    use rand::prelude::ThreadRng;
    use rand::Rng;
    use rayon::iter::IntoParallelIterator;
    use rayon::iter::ParallelIterator;
    use rayon::prelude::ParallelBridge;
    use rayon::iter::IntoParallelRefIterator;
    use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger};
    use tch::{Device, Kind, kind, Tensor};
    use tch::Kind::{Float, Int};
    use tch::TchError;
    use torch_sys::C_tensor;

    use crate::*;
    use crate::find::{FileDumper, FindMotifsParams, MotifTracker};
    use hbac_common::utils::*;
    use crate::find::select_motifs;
    use crate::fragments::{list_motifs, write_motifs_slices};
    use crate::find::iter_all_in_dir;
    use hbac_common::utils::cdt_for_dir;
    use hbac_shared::*;
    use hbac_store::{SqlStore, Store};

    #[ctor::ctor]
    fn init() {
        match CombinedLogger::init(
            vec![
                TermLogger::new(LevelFilter::Debug, Config::default(), TerminalMode::Mixed, ColorChoice::Auto)
            ]
        ){
            Ok(_) => {}
            Err(e) => {
                println!("logger error: {:?}", e);
            }
        };
    }


    #[test]
    pub fn test_filt(){
        //20 x 200_000 in files 5000 files 20_000_000_000
        // let dots = [1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1];
        let dots = [0.5, 0.9, 0.9, 0.5, 0.5, 0.9, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5];

        let mut mt = super::MotifTracker::new(0.8, "/tmp/test_motifs.csv");

        for (i, dot) in dots.into_iter().enumerate(){
            mt.track(TensorSliceIdx(0.to_string(), 0), TensorSliceIdx(0.to_string(), i as i64), dot as f32);
        }
        assert_eq!([(TensorSliceIdx(0.to_string(), 0), TensorSliceIdx(0.to_string(), 1), TensorSliceIdx(0.to_string(), 3), 0.9f32), (TensorSliceIdx(0.to_string(), 0), TensorSliceIdx(0.to_string(), 5), TensorSliceIdx(0.to_string(), 8), 0.9f32)], mt.motifs()[0..2]);

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
    pub fn test_composite_sin() -> Result<(), HbacError>{
        let n = 300i64;
        let x = lin_t(n);
        // let gauss_1 = gauss_t(400., 100., n);

        let pulses_in_random = pulse_in_rand(
            n,
            &[
                (100, comp_sin_v(100)),
                // (100, pulse_v(80., 20., 200, 20.))
            ], 2., 1., 0.1
        );

        let sig = Tensor::from_slice(
            pulses_in_random.as_slice()
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

    #[test]
    pub fn test_pulses_in_rand() -> Result<(), HbacError>{
        let n = 1000i64;
        let x = lin_t(n);
        // let gauss_1 = gauss_t(400., 100., n);

        let to_pulses_in_random = pulse_in_rand(
            n,
            &[
                (100, pulse_v(80., 20., 200, 20.)),
                (100, pulse_v(80., 20., 200, 20.))
            ], 10., 0.001
            , 0.1);

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

    #[test]
    fn test_self_dot_norm_x() -> Result<(), HbacError> {
        let t1 = Tensor::stack(&[
           Tensor::from_slice2(&[
               &[1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1],
               &[10.1, 20.1, 30.1, 40.1, 50.1, 60.1, 70.1, 80.1, 90.1],
               &[100.1, 200.1, 300.1, 400.1, 500.1, 600.1, 700.1, 800.1, 900.1]
           ]),
           Tensor::from_slice2(&[
               &[1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2],
               &[10.2, 20.2, 30.2, 40.2, 50.2, 60.2, 70.2, 80.2, 90.2],
               &[100.2, 200.2, 300.2, 400.2, 500.2, 600.2, 700.2, 800.2, 900.2]
           ]),
           Tensor::from_slice2(&[
               &[1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 9.3],
               &[10.3, 20.3, 30.3, 40.3, 50.3, 60.3, 70.3, 80.3, 90.3],
               &[100.3, 200.3, 300.3, 400.3, 500.3, 600.3, 700.3, 800.3, 900.3]
           ])
        ], 0);



        //(1,2,3) x
        //(1,2,3)
        let slice_1 = t1.f_slice(2, Some(0), Some(3), 1)?; //1, 2, 3 - normilzed -  (0.267261, 0.534522, 0.801784)
        let slice_2 = t1.f_slice(2, Some(3), Some(6), 1)?; //4, 5, 6 - (0.455842, 0.569803, 0.683763)

        println!("slice_1={:?}", slice_1.size());
        println!("slice_2={:?}", slice_2.size());

        // let dot_t = slice_1.dot(&slice_2);
        let dot_t = slice_1.tensordot(&slice_2, 2, 2);

        println!("dot_t={:?}", dot_t.size());

        dot_t.print();

        // let d_0 = dot_norm(2, &slice_1, &slice_2)?;
        //
        // assert_eq!(format!("{:.4}", d_0.double_value(&[0, 0])), "0.9746");
        // assert_eq!(format!("{:.4}", d_0.double_value(&[0, 1])), "0.9746");
        // assert_eq!(format!("{:.4}", d_0.double_value(&[0, 2])), "0.9746");

        Ok(())
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
    pub fn test_select_motifs() -> Result<(), HbacError> {//TODO dummy tensors

        let test_dir = Path::new("/tmp").join(cdt_for_dir());

        fs::create_dir_all(test_dir.clone());
        fs::create_dir_all(test_dir.clone().join("0"));
        fs::create_dir_all(test_dir.clone().join("1"));

        //as build in motifs_to_tensor - channels x  motif  x slice
        let motifs_tensor = //Tensor::vstack(&[
            Tensor::vstack(&[//channel 0
                Tensor::vstack( &[ //
                    Tensor::from_slice(&[1., 0., 1.]),//motif 0 first slice - good motif, matches tensors in "0" and does not matches tensors in 1
                    Tensor::from_slice(&[1., 2., 3.]) //motif 1 first slice - bad motif, matches tensors in "1" and does not matches tensors in 0
                ]).view((2, 3)),
                Tensor::vstack( &[//channel 1
                    Tensor::from_slice(&[4., 5., 6.]),//motif 0 first slice - bad motif,
                    Tensor::from_slice(&[2., 2., 0.]) //motif 1 first slice - good motif,
                ]).view((2, 3)),
            ]).view((2, 2, 3));


        // tensors with matches
        let t_0_0 = Tensor::vstack( &[
            Tensor::from_slice(&[0., 0., 1., 0., 1., 0.]),//channel 0
            Tensor::from_slice(&[1., 1., 0., 1., 0., 0.]) //channel 1
        ]).view((2, 6));

        t_0_0.save(test_dir.join("0").join("t_0_0.pt"));

        let t_0_1 = Tensor::vstack( &[
            Tensor::from_slice(&[0., 0., 1., 0., 1., 0.]),//channel 0
            Tensor::from_slice(&[0., 2., 2., 0., 0., 0.]) //channel 1
        ]).view((2, 6));

        t_0_1.save(test_dir.join("0").join("t_0_1.pt"));

        // tensors without matches
        let t_1_0 = Tensor::vstack( &[
            Tensor::from_slice(&[1., 2., 3., 4., 5., 6.]),//channel 0
            Tensor::from_slice(&[1., 2., 3., 4., 5., 6.]) //channel 1
        ]).view((2, 6));

        t_1_0.save(test_dir.join("1").join("t_1_0.pt"));

        let t_1_1 = Tensor::vstack( &[
            Tensor::from_slice(&[1., 2., 3., 4., 5., 6.]),//channel 0
            Tensor::from_slice(&[1., 2., 3., 4., 5., 6.]) //channel 1
        ]).view((2, 6));

        t_1_1.save(test_dir.join("1").join("t_1_1.pt"));


        let channels_good_motifs = select_motifs(
            // Tensor::load("../../data_local/motif_tensors/seizure_vote_eegs.pt")?,
            motifs_tensor,
            vec![
                (1, iter_all_in_dir(test_dir.join("0"))?),
                (0, iter_all_in_dir(test_dir.join("1"))?),

                // (1, iter_all_in_dir("/media/bu/KINGSTON/eegs.pt/seizure_vote_eegs.txt/")?),
                // (0, iter_all_in_dir("/media/bu/KINGSTON/eegs.pt/lpd_vote_eegs.txt/")?),
                // (0, iter_all_in_dir("/media/bu/KINGSTON/eegs.pt/gpd_vote_eegs.txt/")?),
                // (0, iter_all_in_dir("/media/bu/KINGSTON/eegs.pt/lrda_vote_eegs.txt/")?),
                // (0, iter_all_in_dir("/media/bu/KINGSTON/eegs.pt/grda_vote_eegs.txt/")?),
                // (0, iter_all_in_dir("/media/bu/KINGSTON/eegs.pt/other_vote_eegs.txt/")?),
            ],
            Some(3),
            1,
            Device::Cpu
        )?;

        println!("t={:?}", channels_good_motifs);

        assert_eq!(0, channels_good_motifs[0][0].0); //motif 0 is good for channel 0
        assert_eq!(1, channels_good_motifs[1][0].0); //motif 1 is good for channel 1

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
            pulse_in_rand(n, &[
                // (10, pulse_v(8., 3., 20, 2.)),
                // (30, pulse_v(8., 3., 20, 2.))
                (100, pulse_v(80., 30., 200, 20.)),
                (300, pulse_v(40., 30., 200, 20.))
            ], 50., 0.001, 0.1).as_slice()
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
        let eeg_id_series =  [Some(1111i64), Some(2222), Some(3333)];
        let store = Arc::new(Mutex::new(SqlStore::new(format!("/tmp/test_store_{}.db3", cdt_for_dir()))?));

        let pulse = pulse_v(80., 30., 200, 20.);
        let N = 1000i64;

        let x = lin_t(N);
        let sig_0 = pulse_in_rand(N, &[(100, pulse.clone())], 0.1, 0.1, 1.);
        let sig_1 = pulse_in_rand(N, &[(200, pulse.clone())], 0.1, 0.1, 1.);
        let sig_2 = pulse_in_rand(N, &[(300, pulse.clone())], 0.1, 0.1, 1.);
        plot_series(
            vec![
                (format!("Sig0"), RED, vec_f32_from(&x).into_iter().zip(sig_0.clone()).collect::<Vec<(f32, f32)>>()),
                (format!("Sig1"), GREEN, vec_f32_from(&x).into_iter().zip(sig_1.clone()).collect::<Vec<(f32, f32)>>()),
                (format!("Sig2"), BLACK, vec_f32_from(&x).into_iter().zip(sig_2.clone()).collect::<Vec<(f32, f32)>>()),

            ], "EEG", (1000, 500), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        ).unwrap();

        super::find_motifs(
            move |key:&String| {
                let sig = match key.as_str() {
                    "1111" => &sig_0,
                    "2222" => &sig_1,
                    "3333" => &sig_2,
                    _=> panic!("no")
                };

                Box::new(Tensor::vstack(&[
                    Tensor::from_slice(
                        sig.as_slice()
                    ), //ch 0
                    // Tensor::from_slice(
                    //     sig_in_rand(N, &[], 10., 0.1, 0.1).as_slice()
                    // )  //ch 1, not used
                ]))
            },
            &FindMotifsParams {
                step: 50,
                window: 100,
                min_dot: 0.99,
                kind: "seizure_vote".to_string(),
                descr: "test".to_string(),
                preload_d: 3,
                max_keys_fifo: 3,
            },
            3 * N as usize,
            eeg_id_series.into_iter(),
            Arc::clone(&store), 1
        );

        debug!("{:?}", store.lock()?.list_run_resources(1)?);
        let eeg_slices= store.lock()?.motifs_to_eeg_slices(1)?;
        debug!("0: {:?}", eeg_slices.get(&"1111".to_string()));
        debug!("1: {:?}", eeg_slices.get(&"2222".to_string()));
        debug!("2: {:?}", eeg_slices.get(&"3333".to_string()));

        Ok(())
    }


    #[test]
    pub fn test_find_motifs_euc() -> Result<(), HbacError>{
        let eeg_id_series =  [Some(1111i64), Some(2222), Some(3333)];
        let store_path = format!("/tmp/test_store_{}.db3", cdt_for_dir());
        debug!("store_path={}", store_path);
        let store = Arc::new(Mutex::new(SqlStore::new(store_path)?));

        let pulse = pulse_v(50., 10., 100, 20.);
        let N = 1000i64;

        let x = lin_t(N);

        let sig_0=  join_with_offset::<f32>(N, 0., 10.0, &[(100, pulse.clone())], random_saw());
        let sig_1=  join_with_offset::<f32>(N, 0., 10.0, &[(500, pulse.clone())], random_saw());
        let sig_2=  join_with_offset::<f32>(N, 0., 10.0, &[(800, pulse.clone())], random_saw());

       /*
       plot_series(
            vec![
                (format!("Sig0"), RED, vec_f32_from(&x).into_iter().zip(sig_0.clone()).collect::<Vec<(f32, f32)>>()),
                (format!("Sig1"), GREEN, vec_f32_from(&x).into_iter().zip(sig_1.clone()).collect::<Vec<(f32, f32)>>()),
                (format!("Sig2"), BLACK, vec_f32_from(&x).into_iter().zip(sig_2.clone()).collect::<Vec<(f32, f32)>>()),

            ], "EEG", (1000, 500), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        ).unwrap();
        */

        super::find_motifs_euc(
            move |key:&String| {
                let sig = match key.as_str() {
                    "1111" => &sig_0,
                    "2222" => &sig_1,
                    "3333" => &sig_2,
                    _=> panic!("no")
                };

                Box::new(Tensor::vstack(&[
                    Tensor::from_slice(
                        sig.as_slice()
                    ), //ch 0
                ]))
            },
            &FindMotifsParams {
                step: 100,
                window: 100,
                min_dot: 0.99,
                kind: "seizure_vote".to_string(),
                descr: "test".to_string(),
                preload_d: 3,
                max_keys_fifo: 3,
            },
            3 * N as usize,
            eeg_id_series.into_iter(),
            Arc::clone(&store),
            1
        );

        debug!("{:?}", store.lock()?.list_run_resources(1)?);
        let eeg_slices= store.lock()?.motifs_to_eeg_slices(1)?;
        debug!("0: {:?}", eeg_slices.get(&"1111".to_string()));
        debug!("1: {:?}", eeg_slices.get(&"2222".to_string()));
        debug!("2: {:?}", eeg_slices.get(&"3333".to_string()));

        assert_eq!(&MotifSlice { motif: 1, slise_idx: 100, channel_idx: 0 }, eeg_slices.get(&"1111".to_string()).and_then(|v|v.get(0)).unwrap());
        assert_eq!(&MotifSlice { motif: 1, slise_idx: 500, channel_idx: 0 }, eeg_slices.get(&"2222".to_string()).and_then(|v|v.get(0)).unwrap());
        assert_eq!(&MotifSlice { motif: 1, slise_idx: 800, channel_idx: 0 }, eeg_slices.get(&"3333".to_string()).and_then(|v|v.get(0)).unwrap());

        Ok(())
    }

    fn random_saw() -> Vec<f32> {
        let mut rng = rand::thread_rng();

        (0..20)
            .map(|i| rng.gen_range(-10. .. 10.))
            .flat_map(
                |v| (0..25).map(move |i| v * i as f32).chain((0..25).map(move |i| 25. * v - v * i as f32))
            ).collect::<Vec<f32>>()
    }

    #[test]
    pub fn test_simple_motif_tracker() -> Result<(), HbacError>{
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


        let mut mt = super::SimpMotifTracker::new(0.9, FileDumper::new("/tmp/test_motifs.csv"));

        let dim = 1;


        let ts = TensorsSource::new(vec![t_0, t_1]);
        for i_0 in (0..n-m).step_by(step) {

            for i_1 in (i_0..n-m).step_by(step).skip(1) {
                debug!("i_0={:?}, i_1={:?} m={}", i_0, i_1, m);

                let (slice_01, slice_02) = ts.get_slices(i_0, i_1, m, dim)?;

                let dp = dot_norm(dim, &slice_01, &slice_02)?;

                mt.track(TensorSliceIdx(0.to_string(), i_0), TensorSliceIdx(0.to_string(), i_1), dp.double_value(&[0]) as f32);

            }
            // mt.next();
            // dots.push(row_dots);
        }

        debug!("done in {}", t1.elapsed().as_millis());
        // dots.extend((0..n as usize -dots.len()).map(|_|0.).collect::<Vec<f32>>());
        // debug!("2. dots.len= {}", dots.len());


        // let gauss_1_0 = gauss_1.f_slice(0, Some(300), Some(500), 1)?;

        let motifs = mt.motifs();
        assert_eq!((TensorSliceIdx(0.to_string(), 100), TensorSliceIdx(0.to_string(), 600)), (motifs[0].0.clone(), motifs[0].1.clone()));
        assert_eq!((TensorSliceIdx(0.to_string(), 200), TensorSliceIdx(0.to_string(), 700)), (motifs[1].0.clone(), motifs[1].1.clone()));

        // mt.dump();

        // plot_series(
        //     vec![
        //         (format!("G1"), BLACK, vec_f32_from(&x).into_iter().zip(vec_f32_from(&sig_1_c)).collect::<Vec<(f32, f32)>>()),
        //         // (format!("G1"), BLACK, vec_f32_from(&lin_t(200)).into_iter().zip(vec_f32_from(&gauss_1_0)).collect::<Vec<(f32, f32)>>()),
        //         // (format!("G2"), BLUE, vec_f32_from(&x).into_iter().zip(vec_f32_from(&sig_2_c)).collect::<Vec<(f32, f32)>>()),
        //         // (format!("dots"), MAGENTA, dots.into_iter().enumerate().map(|(i, v)|(i as f32, v)).collect::<Vec<(f32, f32)>>())
        //
        //     ], "EEG", (1000, 480), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        // )?;
        // find_motifs();

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
                Tensor::from_slice(&[1.,  1.,  1.]),//motif 0  - matches 1.,  1.,  1. and  2.,  2.,  2.,
                Tensor::from_slice(&[1.,  2.,  2.]),//motif 1  - matches  1.,  2.,  2. and  2.,  3.,  3.,
                Tensor::from_slice(&[3.,  3.,  1.]),//motif 2  - matches 3.,  3.,  1.
                Tensor::from_slice(&[1.,  7.,  1.]),//motif 3  - no matches
            ]),
            Tensor::vstack(&[//channel 1
                Tensor::from_slice(&[10.,  70.,  10.]),//motif 3
                Tensor::from_slice(&[30.,  30.,  10.]),//motif 2
                Tensor::from_slice(&[10.,  20.,  20.]),//motif 1
                Tensor::from_slice(&[10.,  10.,  10.]),//motif 0
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
                0,//motif 3
                1,//motif 2
                2,//motif 1
                2,//motif 0
            ])
        ]).totype(Int);

        assert_eq!(matches_t, super::clc_dot_matches(&t_0, &motifs_t, 1, 0.99)?);

        Ok(())
    }

    #[test]
    fn test_par(){
        let t = Tensor::from_slice(&[1., 2., 3.]);
        let ptr = t.as_ptr() as usize;

        vec![1., 2., 3.].par_iter().for_each(|v|{
            unsafe {
                println!("v={}", v * Tensor::clone_from_ptr(ptr as *mut C_tensor).double_value(&[1]));
            }
        });
    }

    #[test]
    fn test_par_0(){
        let t = Arc::new(Mutex::new(Tensor::from_slice(&[1., 2., 3.])));

        vec![1., 2., 3.].par_iter().for_each(|v|{
            if let Ok(lock) = t.lock() {
                println!("v={}", v * lock.double_value(&[1]));
            }

        });
    }

    #[test]
    fn test_test(){
        let v = vec!["a1a2a3a4", "b1b2b3b4", "c1c2c3c4", "d1d2d3d4", "e1e2e3e4", "f1f2f3f4", "d1d2d3d4"];
        let v_w_1 = v.chunks(2);
        let v_w_2 = v.chunks(2);
        let mut c = 0;
        v_w_1.enumerate().for_each(|(i_0, vw1)|{

            <std::slice::Chunks<'_, &str> as Clone>::clone(&v_w_2).skip(i_0).for_each(|vw2| {
                println!("vw1={:?} vw2={:?}", vw1, vw2);

                vw1.chunks(2).flat_map(|v|v.into_iter()).for_each(|w1|{
                    w1.chars().collect::<Vec<char>>().chunks(2).for_each(|c1|{

                        vw2.chunks(2).flat_map(|v|v.into_iter()).for_each(|w2|{

                            w2.chars().collect::<Vec<char>>().chunks(2).for_each(|c2| {
                                println!("c1={:?} c2={:?}", c1.into_iter().collect::<String>(), c2.into_iter().collect::<String>());
                                c+=1;
                            });
                        })
                    })

                });
            });
        });

        println!("c={}", c);
    }
}
