use async_trait::async_trait;
use clap::clap_derive::{Args, Subcommand};
use crossbeam_channel::SendError;
use crossbeam_utils::sync::WaitGroup;
use futures::executor::block_on;
use futures::future::join_all;
use futures::{FutureExt, StreamExt};
use futures_util::future::{BoxFuture, try_join_all};
use log::*;
use log::{debug, error, info, trace, warn};
use parking_lot::Mutex;
use parking_lot::RwLock;
use rand::Rng;
use rand::seq::SliceRandom;
use std::cell::Cell;
use std::cmp::max;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::fs::{File, OpenOptions};
use std::future::Future;
use std::hash::Hash;
use std::io::Write;
use std::marker::PhantomData;
use std::num::NonZero;
use std::ops::{Add, AddAssign, Deref, Range};
use std::panic::AssertUnwindSafe;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::thread::current;
use std::time::{Duration, Instant};
use std::{f64, fmt, fs, mem, thread};
use tch::{Device, IndexOp, Kind, Tensor};
use thoosa_common::utils::*;
use thoosa_shared::{CopyToDevice, TensorIdSliceIdx};
use thoosa_shared::val::Val;
use thoosa_shared::{RangeOrValue, ResourceType, ThoosaError, make_combos};
use thoosa_shared::{avg_hist, parse_key_val};
use thoosa_store::{IdSlice, MotifTrackerStore, SqlStore, Store};
use thoosa_store::{SlicesMatch, SqlMotifTrackerStore};

use crate::distance::MaskMaker::{Diagonal, Le, LeQuantile, LeRunningMean, LocalMin, Tril};
use crate::distance::{MaskMaker, average_euclidean_distance, clc_dist_d2, slice_t};
use crate::distance::{clc_dist_d2_and_mask, dist_d2};
use plotters::prelude::BLACK;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelBridge};
use serde::Deserialize;
use tch::Device::{Cpu, Cuda};
use tch::Kind::{Bool, Float};
use tch::kind::{FLOAT_CPU, FLOAT_CUDA, INT64_CPU};
use thoosa_common::tho_device::{ThoDevice};
use thoosa_common::bool_iter::BoolTensorIterator;
use thoosa_common::dims::{Dims, TCD, TCSD};
use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig, MotifSelector, SelectMotifsRunConfig, SlicesFilter};
use thoosa_common::loader::Loader;
use thoosa_common::loader::{LoadRunner, SyncWrapper};
use thoosa_common::parzen::optimizer::ConfigurationSpace;
use thoosa_common::runs::{Labeled, SavableResult, Simple};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::*;
use thoosa_common::transforms::filters::clc_frequencies;
use thoosa_common::ws::{FsWorkspace, Workspace};
use thoosa_common::{loader};
use thoosa_dummy_data_derive::DummyData;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::run_configs::{RunConfig};

use thoosa_properties_map::{FromHashmap, ToHashmap};
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_shared::applicables_runner::{CachableId, Input, Output};
use thoosa_shared::colors_iter::DistinctColorIter;
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::parse_range_or_value;
use thoosa_shared::parse_resources;
use thoosa_shared::resource::{Resource, ResourceId, ResourceStore, ResourcesConfig};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::traits::{Data, WithId};
use thoosa_shared::val::Val::UInt;
use thoosa_shared::panic_to_err;
use thoosa_store::fs_store::FileMotifTrackerStore;

#[derive(Debug, Args, Clone)]
pub struct FindMotifsArgs {
    #[arg(long,value_parser=parse_resources)]
    pub resources: ResourceType,

    #[command(flatten)]
    pub fixed_config: FixedFindMotifsConfig,

    #[arg(long)]
    pub run_config_cnt: String, //FindMotifsRunConfig
}

impl FindMotifsArgs {
    pub fn resource_config(&self) -> Simple {
        Simple::new(self.resources.clone())
    }
}

#[derive(Debug, Args, Clone)]
pub struct FixedFindMotifsConfig {
    //TODO rename to Args
    #[arg(long)]
    pub min_tracker_step: Option<usize>, //TODO usize

    #[arg(long, default_value_t = 10_000)]
    pub max_chunk_size: u32, //max number of entries in butched tensors

    #[arg(long, default_value_t = false)]
    pub keep_slice_matches: bool, //sql

    #[arg(long)]
    pub max_slices_k: Option<f32>,
}

impl FixedFindMotifsConfig {
    pub fn new(
        min_tracker_step: Option<usize>,
        max_chunk_size: u32,
        keep_slice_matches: bool,
        max_slices_k: Option<f32>,
    ) -> Self {
        Self {
            min_tracker_step,
            max_chunk_size,
            keep_slice_matches,
            max_slices_k,
        }
    }
}


#[derive(Debug, Clone, DummyData)]
#[dummy_data(clone)]
pub struct FindMotifsSavableResult {
    run_config: Arc<FindMotifsRunConfig>,
}



impl FindMotifsSavableResult {
    pub fn new(run_config: Arc<FindMotifsRunConfig>) -> Self {
        Self { run_config }
    }
}

impl SavableResult for FindMotifsSavableResult {
    fn value(&self) -> f32 {
        0.0
    }

    fn save(
        &mut self,
        store: &Arc<RwLock<Box<dyn Store>>>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
    ) -> Result<u32, ThoosaError> {
        Ok(self.run_config.id.expect("id"))
    }
}

impl Output for FindMotifsSavableResult {}
impl WithId for FindMotifsSavableResult {
    type ID = RunId;

    fn id(&self) -> &Self::ID {
        self.run_config.id.as_ref().unwrap()
    }
}



//TODO trait for MotifTracker
#[derive(Debug)]
pub struct SimpleMotifTracker<'a> {
    last: Option<SlicesMatch>,
    motifs: Vec<SlicesMatch>,                //i_0, match_0, match_1, dist
    dumper: Box<dyn MotifTrackerStore + 'a>, //lifetime is for DummyDumper
    min_step: Option<usize>,                 //TODO remove, all filtering is in gpu
    max_slices_counts: Option<u32>,
}

impl<'a> SimpleMotifTracker<'a> {
    pub fn new(
        dumper: Box<dyn MotifTrackerStore + 'a>,
        min_step: Option<usize>,
        max_slices_counts: Option<u32>,
    ) -> SimpleMotifTracker {
        Self {
            last: None,
            motifs: vec![],
            dumper,
            min_step,
            max_slices_counts,
        }
    }

    pub fn track(&mut self, i_0: TensorIdSliceIdx, i_1: TensorIdSliceIdx, dist: f32) {
        trace!("SimpMotifTracker::track: i_0={i_0:?} i_1={i_1:?} dist={dist}");

        match self.min_step {
            None => {
                self.motifs.push(SlicesMatch(i_0, i_1, dist));
            }

            Some(min_step) => {
                let last = self.last.take();
                if let Some(last) = last {
                    //TODO
                    if last.2 <= dist &&
                        last.0.0 == i_0.0 && last.1.0 == i_1.0 && //still same chunks
                        (i_0.1 - last.0.1  > min_step as u32 || i_1.1 - last.1.1  > min_step as u32)
                    {
                        //i_. .1 < last.1.1 if next tensor
                        self.motifs.push(last);
                    }
                }
                self.last.replace(SlicesMatch(i_0, i_1, dist));
            }
        }
    }

    pub fn dump(&mut self) {
        debug!("SimpMotifTracker::dump: self.motifs={}", self.motifs.len());
        if let Some(last) = self.last.take() {
            self.motifs.push(last);
        }

        if self.motifs.len() > 0 {
            self.dumper
                .save_slices(mem::take(&mut self.motifs))
                .expect("dumper");
        }
    }

    pub fn build_groups(&mut self, store: &Arc<RwLock<Box<dyn Store>>>) -> Result<(), ThoosaError> {
        self.dumper.load_slices_into_groups(store, self.max_slices_counts)
    }

    #[cfg(test)]
    pub fn motifs(&self) -> &Vec<SlicesMatch> {
        &self.motifs
    }
}

/**
TODO  selects motifs based on correlation to a label only
*/
#[derive(Debug, Args, Clone)]
pub struct SelectMotifsParams {
    #[arg(long)]
    find_motifs_run_ids: Vec<RunId>,

    #[arg(long)]
    pub resources_group: String,
}

#[derive(Debug, Args, Clone)]
pub struct SelectMotifsArgs {
    #[arg(long)]
    pub find_motifs_run_ids: Vec<RunId>,

    #[arg(long)]
    pub resources_group: String,

    #[arg(long)]
    pub run_config_cnt: String, //SelectMotifsRunConfig
}

impl SelectMotifsArgs {
    pub fn resource_config(&self) -> Simple {
        Simple::new(ResourceType::NotInGroup(self.resources_group.clone()))
    }
}


#[derive(Debug, Clone, DummyData)]
#[dummy_data(clone)]
pub struct SelectMotifsSavableResult {
    dummy_id:RunId,
    run_config: Vec<(SelectMotifsRunConfig, Vec<(u32, f64)>)>,
    // good_motif_ids_rates: Vec<(u32, f64)>,
}

impl Output for SelectMotifsSavableResult {}
impl WithId for SelectMotifsSavableResult {
    type ID = RunId;

    fn id(&self) -> &Self::ID {
        &self.dummy_id
    }
}

impl SelectMotifsSavableResult {
    pub fn new(
        find_motif_run_id: &[RunId],
        run_config: Vec<SelectMotifsRunConfig>,
        good_motif_ids_rates: Vec<(u32, f64)>,
    ) {
        todo!() //remove it, used from old code
    }
    pub fn new2(
        run_config_good_motif_ids: Vec<(SelectMotifsRunConfig, Vec<(u32, f64)>)>,
    ) -> Self {
        Self {
            dummy_id: 0,
            run_config: run_config_good_motif_ids,
        }
    }
}

impl SavableResult for SelectMotifsSavableResult {
    fn value(&self) -> f32 {
        0.1
    }

    fn save(
        &mut self,
        store: &Arc<RwLock<Box<dyn Store>>>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
    ) -> Result<u32, ThoosaError> {
        for (run_config, good_motif_ids_rates) in &self.run_config {
            store
                .write()
                .save_selected_motifs(run_config.id.expect("should be pre set in run_grid"), good_motif_ids_rates.clone())?; //TODO mem::replace?

        }
        Ok(0)//TODO not used
    }
}
#[cfg(test)]
mod motif_find_tests {
    use std::{fs, mem, thread};

    use crossbeam_channel::internal::SelectHandle;
    use crossbeam_utils::sync::WaitGroup;
    use futures::executor::block_on;
    use log::LevelFilter::Trace;
    use log::{debug, set_max_level, trace};
    use parking_lot::RwLock;
    use parking_lot::{Mutex, RawMutex};
    use plotters::prelude::full_palette::ORANGE_800;
    use plotters::prelude::{Color, RGBAColor, ShapeStyle};
    use plotters::style::full_palette::{GREY_A100, YELLOW_A100};
    use rayon::iter::ParallelBridge;
    use rayon::iter::ParallelIterator;
    use std::collections::{HashMap, HashSet};
    use std::fmt::{Debug, Display};
    use std::fs::create_dir_all;
    use std::ops::{Add, AddAssign, Deref};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::atomic::{AtomicU32, AtomicU64};
    use tch::Device::Cpu;
    use tch::Kind::{Bool, Float, Int};
    use tch::kind::DOUBLE_CPU;
    use tch::{Device, IndexOp, Kind, Tensor};
    use thoosa_common::find_motifs_run_config::{Conv1Params, FindMotifsRunConfig, InputFilter};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::utils::*;
    use thoosa_shared::*;

    use crate::distance::clc_dist_d2;
    use crate::find::{
        FixedFindMotifsConfig, MotifTrackerStore,
    };
    use thoosa_common::loader::{LoadRunner, Loader, SyncWrapper};
    use thoosa_common::runs::Simple;
    use thoosa_common::ws::{FsWorkspace, mock_ws_simp};
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::colors_iter::DistinctColorIter;
    use thoosa_shared::resource::{Resource, ResourceId, ResourceStats};
    use thoosa_shared::run_configs::RunConfig;
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::val::Val;
    use thoosa_shared::val::Val::UInt;
    use thoosa_store::{IdSlice, SlicesMatch, SqlStore, Store};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_z_norm() {
        assert_eq!(
            vec![-1.2247449, 0.0, 1.2247449],
            vec_f32_from(&z_norm(&Tensor::from_slice(&[1., 2., 3.]), 0i64))
        );
    }

    #[test]
    fn test_norm() -> Result<(), ThoosaError> {
        let t = Tensor::from_slice(&[1., 2., 3.]);
        let norm = norm(&t, &[-1]);

        assert_eq!(1.0, vec_f32_from(&norm.dot(&norm))[0]);

        let t = Tensor::vstack(&[
            Tensor::from_slice(&[1., 2., 3.]),
            Tensor::from_slice(&[1., 2., 3.]),
        ]);

        let norm = norm_2(&t, 1);

        assert_eq!(
            Tensor::from_slice(&[1., 1.]),
            norm.tensordot(&norm, 1, 1).diag(0)
        );

        Ok(())
    }

    #[test]
    fn test_self_dot() -> Result<(), ThoosaError> {
        let t1 = Tensor::from_slice2(&[
            &[1., 2., 3., 4., 5., 6., 7., 8., 9.],
            &[10., 20., 30., 40., 50., 60., 70., 80., 90.],
            &[100., 200., 300., 400., 500., 600., 700., 800., 900.],
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
    fn test_self_dot_norm() -> Result<(), ThoosaError> {
        let t1 = Tensor::from_slice2(&[
            &[1., 2., 3., 4., 5., 6., 7., 8., 9.],
            &[10., 20., 30., 40., 50., 60., 70., 80., 90.],
            &[100., 200., 300., 400., 500., 600., 700., 800., 900.],
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

    pub struct TensorsSource {
        tensors: Vec<Tensor>,
    }

    impl TensorsSource {
        pub fn new(tensors: Vec<Tensor>) -> Self {
            Self { tensors }
        }

        fn get_slices(
            &self,
            i_0: i64,
            i_1: i64,
            m: i64,
            dim: i64,
        ) -> Result<(Tensor, Tensor), ThoosaError> {
            Ok((
                self.tensors[0].f_slice(dim, Some(i_0), Some(i_0 + m), 1)?,
                self.tensors[0].f_slice(dim, Some(i_1), Some(i_1 + m), 1)?,
            ))
        }
    }


    //TODO replace with mockall
    #[derive(Debug)]
    struct DummyDumper<'a> {
        motifs: &'a mut Vec<SlicesMatch>,
    }

    impl<'a> DummyDumper<'a> {
        pub fn new(motifs: &'a mut Vec<SlicesMatch>) -> Self {
            Self { motifs }
        }
    }

    impl<'a> MotifTrackerStore for DummyDumper<'a> {
        fn save_slices(&mut self, motifs: Vec<SlicesMatch>) -> Result<(), ThoosaError> {
            self.motifs
                .extend(motifs.into_iter().collect::<Vec<SlicesMatch>>());

            Ok(())
        }

        fn load_slices_into_groups(
            &mut self,
            store: &Arc<RwLock<Box<dyn Store>>>,
            max_slices_counts: Option<u32>,
        ) -> Result<(), ThoosaError> {
            todo!()
        }

        fn run_id(&self) -> RunId {
            todo!()
        }

        fn channel(&self) -> u32 {
            todo!()
        }

        fn path(&self) -> &PathBuf {
            todo!()
        }
    }

    #[test]
    pub fn test_simple_motif_tracker() -> Result<(), ThoosaError> {
        let mut motifs = vec![];
        {
            DummyDumper::new(&mut motifs);
            let mut mt = super::SimpleMotifTracker::new(
                Box::new(DummyDumper::new(&mut motifs)),
                None,
                Some(100),
            );

            mt.track(TensorIdSliceIdx(0, 10), TensorIdSliceIdx(1, 10), 1.0);
            mt.track(TensorIdSliceIdx(0, 20), TensorIdSliceIdx(1, 20), 1.0);
            mt.track(TensorIdSliceIdx(0, 30), TensorIdSliceIdx(1, 30), 1.0);

            mt.dump();
        }
        assert_eq!(motifs.len(), 3);
        assert_eq!(
            SlicesMatch(TensorIdSliceIdx(0, 10), TensorIdSliceIdx(1, 10), 1.0),
            motifs[0]
        );
        assert_eq!(
            SlicesMatch(TensorIdSliceIdx(0, 20), TensorIdSliceIdx(1, 20), 1.0),
            motifs[1]
        );
        assert_eq!(
            SlicesMatch(TensorIdSliceIdx(0, 30), TensorIdSliceIdx(1, 30), 1.0),
            motifs[2]
        );

        let mut motifs = vec![];
        {
            let mut mt = super::SimpleMotifTracker::new(
                Box::new(DummyDumper::new(&mut motifs)),
                Some(10),
                Some(100),
            );

            mt.track(TensorIdSliceIdx(0, 10), TensorIdSliceIdx(1, 10), 1.0);
            mt.track(TensorIdSliceIdx(0, 20), TensorIdSliceIdx(1, 20), 1.0);
            mt.track(TensorIdSliceIdx(0, 31), TensorIdSliceIdx(1, 30), 1.0);

            mt.dump();
        }

        assert_eq!(motifs.len(), 2);
        assert_eq!(
            SlicesMatch(TensorIdSliceIdx(0, 20), TensorIdSliceIdx(1, 20), 1.0),
            motifs[0]
        );
        assert_eq!(
            SlicesMatch(TensorIdSliceIdx(0, 31), TensorIdSliceIdx(1, 30), 1.0),
            motifs[1]
        );

        Ok(())
    }

    #[test]
    pub fn test_simple_motif_tracker_with_tensorssource() -> Result<(), ThoosaError> {
        let n = 1_000i64;

        let (sig_0_1, sig_0_2) = two_sigs(1_000);
        debug!(
            "sig_1.size()={:?}, sig_2.size()={:?}",
            sig_0_1.size(),
            sig_0_2.size()
        );
        let _sig_0_c = sig_0_1.copy();

        let t_0 = Tensor::vstack(&[sig_0_1, sig_0_2]);
        debug!("t.size={:?}", t_0.size());

        let (sig_1_1, sig_1_2) = two_sigs(1_000);
        let _sig_1_c = sig_1_1.copy();
        let t_1 = Tensor::vstack(&[sig_1_1, sig_1_2]);

        let m = 100;
        let step = 100;

        let mut motifs = vec![];
        let mut mt = super::SimpleMotifTracker::new(
            Box::new(DummyDumper::new(&mut motifs)),
            None,
            Some(100),
        );

        let dim = 1;

        let ts = TensorsSource::new(vec![t_0, t_1]);
        for i_0 in (0..n - m).step_by(step) {
            for i_1 in (i_0..n - m).step_by(step).skip(1) {
                debug!("i_0={:?}, i_1={:?} m={}", i_0, i_1, m);

                let (slice_01, slice_02) = ts.get_slices(i_0, i_1, m, dim)?;

                let dp = dot_norm(dim, &slice_01, &slice_02)?;

                mt.track(
                    TensorIdSliceIdx(0, i_0 as u32),
                    TensorIdSliceIdx(0, i_1 as u32),
                    dp.double_value(&[0]) as f32,
                );
            }
        }

        /*
        let x = lin_t(n);
        plot_series(
            vec![
                (format!("G1"), BLACK, vec_f32_from(&x).into_iter().zip(vec_f32_from(&_sig_0_c)).collect::<Vec<(f32, f32)>>()),
                // (format!("G1"), BLACK, vec_f32_from(&lin_t(200)).into_iter().zip(vec_f32_from(&gauss_1_0)).collect::<Vec<(f32, f32)>>()),
                (format!("G2"), BLUE, vec_f32_from(&x).into_iter().zip(vec_f32_from(&_sig_1_c)).collect::<Vec<(f32, f32)>>()),
                // (format!("dots"), MAGENTA, dots.into_iter().enumerate().map(|(i, v)|(i as f32, v)).collect::<Vec<(f32, f32)>>())

            ], "test_simple_motif_tracker", (1000, 480), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        )?;
        */

        let motifs: Vec<&SlicesMatch> = mt.motifs().iter().filter(|v| v.2 > 0.9).collect();

        assert_eq!(
            (TensorIdSliceIdx(0, 100), TensorIdSliceIdx(0, 600)),
            (motifs[0].0.clone(), motifs[0].1.clone())
        );
        assert_eq!(
            (TensorIdSliceIdx(0, 200), TensorIdSliceIdx(0, 700)),
            (motifs[1].0.clone(), motifs[1].1.clone())
        );

        Ok(())
    }

    #[test]
    fn test_conv() {
        let t_0 = Tensor::vstack(&[
            Tensor::from_slice(&[1., 1., 1., 0., 0., 0.]),
            Tensor::from_slice(&[2., 2., 2., 0., 0., 0.]),
        ]); //view((2, 1, 6))
        let t_1 = Tensor::vstack(&[
            Tensor::from_slice(&[1., 1., 1.]),
            Tensor::from_slice(&[2., 2., 2.]),
        ])
        .view((2, 1, 3)); //view((2, 1, 3))

        let c = t_0.conv1d(&t_1, None::<Tensor>, 1, 3, 1, 2);

        debug!("t_0={:?} t_1={:?} c={:?}", t_0.size(), t_1.size(), c.size());

        assert_eq!(
            c,
            Tensor::vstack(&[
                Tensor::from_slice(&[0., 1., 2., 3., 2., 1., 0., 0., 0., 0.,]),
                Tensor::from_slice(&[0., 4., 8., 12., 8., 4., 0., 0., 0., 0.,])
            ])
        )

        //https://www.rapidtables.com/calc/math/convolution-calculator.html
        //1 1 1 0 0 0 * 1 1 1 = 1  2  3  2  1 0 0 0
        //2 2 2 0 0 0 * 2 2 2 = 4  8  12 8  4 0 0 0
    }

    #[test]
    fn test_conv_2() {
        let t_0 = Tensor::vstack(&[
            Tensor::from_slice(&[1., 1., 1., 0., 0., 0.]), // first channel
            Tensor::from_slice(&[2., 2., 2., 0., 0., 0.]), // second channel
        ]); //view((2, 1, 6))

        let t_1 = Tensor::vstack(&[
            Tensor::vstack(&[
                Tensor::from_slice(&[1., 1., 1.]), // 1 first channel kernel
                Tensor::from_slice(&[3., 3., 3.]), // 2 first channel kenrel
            ]),
            Tensor::vstack(&[
                Tensor::from_slice(&[2., 2., 2.]), // 1 second channel kernel
                Tensor::from_slice(&[4., 4., 4.]), // 2 second channel kernel
            ]),
        ])
        .view((4, 1, 3));

        debug!("t_0={:?} t_1={:?}", t_0.size(), t_1.size());

        let c = t_0.conv1d(&t_1, None::<Tensor>, 1, 3, 1, 2);

        debug!("c={:?}", c.size());

        assert_eq!(
            c,
            Tensor::vstack(&[
                Tensor::from_slice(&[0., 1., 2., 3., 2., 1., 0., 0., 0., 0.,]), // first channel * 1 kernel
                Tensor::from_slice(&[0., 3., 6., 9., 6., 3., 0., 0., 0., 0.,]), // first channel * 2 kernel
                Tensor::from_slice(&[0., 4., 8., 12., 8., 4., 0., 0., 0., 0.,]), // second channel * 1 kernel
                Tensor::from_slice(&[0., 8., 16., 24., 16., 8., 0., 0., 0., 0.,]) // second channel * 2 kernel
            ])
        );

        let c_by_channel = c.view((2, -1, 10));

        debug!("c={:?}", c_by_channel.size());

        //https://www.rapidtables.com/calc/math/convolution-calculator.html
        //1 1 1 0 0 0 * 1 1 1 = 1  2  3  2  1 0 0 0
        //2 2 2 0 0 0 * 2 2 2 = 4  8  12 8  4 0 0 0
    }

    #[test]
    fn test_find_motifs_run_config_to_from_map() {
        let params = FindMotifsRunConfig::new(
            Some(0),
            10,
            10,
            Some(InputFilter::Conv1(Conv1Params::new(10, 1, 1))),
            Some(0.01),
        );

        let properties: HashMap<String, Val> = params.to_properties_iter(None).collect();

        let loaded_params = FindMotifsRunConfig::from_properties_map_with_id(0, &properties).unwrap();

        debug!("loaded_params={:?}", loaded_params);

        assert_eq!(params, loaded_params);
    }

}

pub fn plot_resource(
    resource_config: Simple,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
) -> Result<(), ThoosaError> {
    let tensors_path = ws.read().tensors_path().clone();
    let channel = 0; //TODO

    let resources = resource_config.to_resources(store)?;
    let all_tensors = resources.iter().map(|r| {
        let t = Tensor::load(
            // tensors_path.join(format!("{}_{}.pt", key.import_id(), key.file_stem())),
            FsWorkspace::resource_tensor_path(&tensors_path, r),
        )
        .unwrap()
        .i((channel, ..));

        let freqs = clc_frequencies(&t, t.size()[0], 32000f64, 100);
        debug!("plot_resource: r={r:?} freqs={freqs:?}");

        let freqs_bins = avg_hist(&freqs.iter().map(|v| v.0).collect::<Vec<f64>>(), 10);

        debug!("plot_resource: freqs_bins={freqs_bins:?}");

        t
    });
    let mut color_iter = DistinctColorIter::new();
    plot_series(
        all_tensors
            .into_iter()
            .enumerate()
            .map(|(i, t)| {
                (
                    format!("{i}"),
                    color_iter.next().unwrap_or(BLACK),
                    t.iter::<f64>()
                        .unwrap()
                        .enumerate()
                        .map(|(i, v)| (i as f32, v as f32))
                        .collect::<Vec<(f32, f32)>>(),
                )
            })
            .collect(),
        format!("Resources={resource_config:?}").as_str(),
        (2000, 1000),
        &ws.read().plots_path().join(format!(
            "resources_{:?}.png",
            resources.iter().map(|r| r.file_stem()).collect::<String>()
        )),
    )?;

    Ok(())
}
