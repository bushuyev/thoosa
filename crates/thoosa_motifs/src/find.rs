use clap::clap_derive::{Args, Subcommand};
use crossbeam_channel::SendError;
use crossbeam_utils::sync::WaitGroup;
use futures::{FutureExt, StreamExt};
use log::*;
use log::{debug, error, info, trace, warn};
use parking_lot::Mutex;
use parking_lot::RwLock;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::cell::Cell;
use std::cmp::max;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::fs::{File, OpenOptions};
use std::hash::Hash;
use std::io::Write;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Deref, Range};
use std::rc::Rc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::current;
use std::time::{Duration, Instant};
use std::{fmt, fs, mem, thread};
use tch::{Device, IndexOp, Kind, Tensor};
use std::panic::AssertUnwindSafe;

use thoosa_common::utils::*;
use thoosa_shared::parse_key_val;
use thoosa_shared::val::Val;
use thoosa_shared::TensorIdSliceIdx;
use thoosa_shared::{make_combos, RangeOrValue, ResourceType, ThoosaError};
use thoosa_store::{IdSlice, MotifTrackerStore, SqlStore, Store};
use thoosa_store::{SlicesMatch, SqlMotifTrackerStore};


use crate::chunks::{ChunkOfSlicedTensors, ChunkOfSlicedTensorsIter};
use crate::distance::MaskMaker::{Diagonal, LeQuantile, LeMean, LocalMin, Tril, LE};
use crate::distance::{average_euclidean_distance, slice_t};
use crate::distance::{clc_dist_d2_and_mask, dist_d2};
use crate::fragments::{load_motif_slices, motifs_to_tensor, MotifsTensors};
use plotters::prelude::BLACK;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelBridge};
use tch::kind::{FLOAT_CPU, FLOAT_CUDA, INT64_CPU};
use tch::Device::{Cpu, Cuda};
use thoosa_common::bool_iter::BoolTensorIterator;
use thoosa_common::{loader, panic_to_err};
use thoosa_common::loader::Loader;
use thoosa_common::loader::{LoadRunner, SyncWrapper};
use thoosa_common::parzen::optimizer::Primitives::Float;
use thoosa_common::parzen::optimizer::{ConfigurationSpace, Hyperparameter};
use thoosa_common::runs::{Labeled, Params, ResourcesConfig, RunConfig, SavableOptResult, Simple};
use thoosa_common::ws::{FsWorkspace, Workspace};
use thoosa_shared::colors_iter::DistinctColorIter;
use thoosa_shared::parse_range_or_value;
use thoosa_shared::parse_resources;
use thoosa_shared::resource::{Resource, ResourceId, ResourceStore};
use thoosa_shared::val::Val::UInt;
use thoosa_store::fs_store::FileMotifTrackerStore;

#[derive(Debug, Args, Clone)]
pub struct FindMotifsParams {//TODO pub
    #[arg(long, value_parser=parse_range_or_value::<u32>, default_value = "1000")]
    pub step: RangeOrValue<u32>,

    #[arg(long, value_parser=parse_range_or_value::<u32>, default_value = "1000")]
    pub window: RangeOrValue<u32>,

    #[arg(long,value_parser=parse_resources)]
    pub resources: ResourceType,

    #[arg(long)]
    pub min_tracker_step: Option<usize>,

    #[arg(long, default_value_t = 100)]
    pub chunk_size: u32,


    #[arg(long, default_value_t = false)]
    pub keep_slice_matches: bool,

    #[arg(long)]
    pub max_slices_k: Option<f32>,


    #[command(subcommand)]
    pub input_filter: Option<InputFilter>,


    #[arg(long, default_value_t = 0.01)]
    pub dist_quantile: f32,//TODO Option

    #[arg(long)]
    pub mean_k: Option<f32>,


}

#[derive(Debug)]
pub struct CommonFindMotifsConfig {
    min_tracker_step:Option<usize>,
    chunk_size:u32,
    keep_slice_matches:bool, //sql
    max_slices_k:Option<f32>,
    mean_k: f32,
}

impl CommonFindMotifsConfig {

    pub fn new(min_tracker_step: Option<usize>, chunk_size: u32, keep_slice_matches: bool, max_slices_k: Option<f32>, mean_k: Option<f32>) -> Self {
        Self {
            min_tracker_step,
            chunk_size,
            keep_slice_matches,
            max_slices_k,
            mean_k: mean_k.unwrap_or(0.01),
        }
    }
}

impl Params<FindMotifsRunConfig, Simple, CommonFindMotifsConfig> for FindMotifsParams {
    fn to_run_configs(&self) -> Vec<FindMotifsRunConfig> {

        debug!("FindMotifsParams::to_run_configs: self.step={:?} self.window={:?}", self.step, self.window);

        make_combos(vec![
            self.step.clone().into_iter(),
            self.window.clone().into_iter(),
        ]).into_iter().map(|config_combo|{
            debug!("FindMotifsParams::to_run_configs: config_combo={config_combo:?} ");
            FindMotifsRunConfig::new(None, config_combo[0], config_combo[1], self.dist_quantile, "".to_string(), None)
        }).collect()
    }

    fn resource_config(&self) -> Simple {
        Simple::new(
            self.resources.clone()
        )
    }

    fn common_config(&self) -> CommonFindMotifsConfig {

        CommonFindMotifsConfig {
            chunk_size: self.chunk_size,
            min_tracker_step: self.min_tracker_step,
            keep_slice_matches: self.keep_slice_matches,
            max_slices_k: self.max_slices_k,
            mean_k: self.mean_k.unwrap_or(0.01),
        }
    }
}

#[derive(Debug, Subcommand, PartialEq, Clone)]
pub enum InputFilter {
    Conv1(Conv1Params)
}

#[derive(Debug, Args, PartialEq, Clone)] //PartialEq - for tests
pub struct Conv1Params {
    #[arg(long)]
    kernel: i64,

    #[arg(long, default_value_t = 1)]
    stride: i64,

    #[arg(long, default_value_t = 0)]
    padding: i64,
}

impl InputFilter {
    pub fn to_properties_map(&self) -> HashMap<String, Val> {
        match self {
            Self::Conv1(conv1_params) => {
                conv1_params.to_properties_map().into_iter().fold(HashMap::new(), |mut a, v| {
                    a.insert(format!("conv_1.{}", v.0), v.1);
                    a
                })
            }
        }
    }

    pub fn from_properties_map(properties: &HashMap<String, Val>) -> Option<Self> {
        let conv_1_props = thoosa_shared::val::unprefix_properties("conv_1", &properties);
        if conv_1_props.len() > 0 {
            assert!(conv_1_props.len() == 3);
            Conv1Params::from_properties_map(conv_1_props).map(|props| InputFilter::Conv1(props)).ok()
        } else {
            None
        }
    }
}

impl Conv1Params {
    pub fn new(kernel: i64, stride: i64, padding: i64) -> Self {
        Self { kernel, stride, padding }
    }

    pub fn to_properties_map(&self) -> HashMap<String, Val> {
        HashMap::from([
            ("Conv1Params.kernel".into(), self.kernel.into()),
            ("Conv1Params.stride".into(), self.stride.into()),
            ("Conv1Params.padding".into(), self.padding.into()),
        ])
    }

    pub fn from_properties_map(properties: HashMap<String, Val>) -> Result<Self, ThoosaError> {
        let properties = thoosa_shared::val::unprefix_properties("Conv1Params", &properties);

        Ok(Conv1Params {
            kernel: properties.get("kernel").expect("kernel").as_i64().expect("kernel as_i64"),
            stride: properties.get("stride").expect("stride").as_i64().expect("stride as_i64"),
            padding: properties.get("padding").expect("padding").as_i64().expect("padding as_i64"),
        })
    }
}


impl FindMotifsParams {
    // #[cfg(test)]
    pub fn dummy(step: u32, window: impl Into<RangeOrValue<u32>>, group: String, chunk_size: u32, dist_quantile: f32, input_filter: Option<&InputFilter>, min_tracker_step: Option<usize>, keep_slice_matches: bool, max_slices_k: Option<f32>, mean_k: Option<f32>) -> Self {
        FindMotifsParams {
            step:step.into(),

            window:window.into(),

            resources: ResourceType::Group(group),

            min_tracker_step,

            chunk_size,

            dist_quantile,

            keep_slice_matches,

            max_slices_k,

            mean_k,

            input_filter: input_filter.cloned(),
        }
    }

    pub fn resources(&self) -> &ResourceType {
        &self.resources
    }

    pub fn max_dist_k(&self) -> f32 {
        self.dist_quantile
    }

}

pub fn find_motifs_by_euc_dist(
    _: Option<&Vec<usize>>,//first step, no input run ids
    mut run_configs: Vec<FindMotifsRunConfig>,
    resource_config: &Simple,
    common_config: &CommonFindMotifsConfig,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Mutex<Box<dyn Store>>>,
    loader: &Arc<RwLock< LoadRunner<Box<Tensor>>>>,
    devices:Vec<Device>
) -> Result<Option<Vec<Box<dyn SavableOptResult>>>, ThoosaError> {

    let mut resource_keys = resource_config.to_resources(store)?;
    // resource_keys.shuffle(&mut thread_rng());

    let run_config_ids:Vec<usize> = run_configs.iter().map(|run_config| run_config.id.expect("id")).collect();

    debug!("find_motifs_by_euc_dist: resource_keys={resource_keys:?} run_config_ids={run_config_ids:?} channels={}", ws.read().channels());

    let run_config__channels_trackers: HashMap<usize, Vec<Mutex<SimpleMotifTracker>>> = run_configs.iter().fold(HashMap::new(), |mut a, run_config| {

        let slices_counts = resource_keys.iter().map(|r|(r.length()-run_config.window)/run_config.step + 1).sum::<u32>();
        let max_slices_counts = common_config.max_slices_k.map(|k| (k * slices_counts as f32) as u32);

        debug!("find_motifs_by_euc_dist: run_config={run_config:?} slices_counts={slices_counts} max_slices_counts={max_slices_counts:?}");

        assert!(a.insert(
            run_config.id.expect("id"),
            (0..ws.read().channels()).map(|channel| {
                Mutex::new(SimpleMotifTracker::new(
                    if common_config.keep_slice_matches {
                        SqlMotifTrackerStore::new(
                            ws.read().tmp_run_store_path(run_config.id.expect("id") as u32, channel as u32),
                            run_config.id.expect("id"),
                            channel
                        ).expect("can't create tmp store")
                    } else {
                        FileMotifTrackerStore::new(
                            ws.read().tmp_path(),
                            run_config.id.expect("id"),
                            channel
                        ).expect("can't create tmp store")
                    },
                    common_config.min_tracker_step,
                    max_slices_counts
                ))
            }).collect::<Vec<Mutex<SimpleMotifTracker>>>()
        ).is_none());

        a
    });

    let run_configs_arcs = run_configs.into_iter()
        .map(|rc|Arc::new(rc))
        .collect::<Vec<_>>();

    // debug!("find_motifs_by_euc_dist_: records_num={records_num} resource_keys={resource_keys:?} run_config_ids={run_config_ids:?}");
    let devices_2x = devices.clone();

    let mut means_acc = Mutex::new(Tensor::zeros([ws.read().channels() as i64, 1], (Kind::Float, devices[0])));
    let mut total_samples = Mutex::new(Tensor::zeros([ws.read().channels() as i64, 1], (Kind::Int64, devices[0])));

    chunk_and_slice_2x(
        resource_keys,
        common_config.chunk_size as usize,
        &run_configs_arcs,
        move |chunk_sliced_0: &Arc<ChunkOfSlicedTensors>, chunk_sliced_1: &ChunkOfSlicedTensors, find_motifs_run_config:&Arc<FindMotifsRunConfig>| {

            debug!("find_motifs_by_euc_dist/gpu_fn: \nchunk_sliced_0={chunk_sliced_0:?} \nchunk_sliced_1={chunk_sliced_1:?}");

            let mask_tril = chunk_sliced_0.resource_ids[0].eq(&chunk_sliced_1.resource_ids[0]) && !chunk_sliced_1.is_split();
            let mask_makers = if mask_tril {
                &mut [LocalMin, LeMean { k: common_config.mean_k, avg: &mut means_acc, total_samples: &mut total_samples }, Tril][..]
            } else {
                &mut [LocalMin, LeMean { k: common_config.mean_k, avg: &mut means_acc, total_samples: &mut total_samples }][..]
            };

            let (mask, _) = clc_dist_d2_and_mask(
                //TODO devices[0], should be in chunk_and_slice_2x
                chunk_sliced_0.t.to_device(devices_2x[0]),
                chunk_sliced_1.t.to_device(devices_2x[0]),
                false,
                mask_makers,
                //remove diagonal if same chunk_0 and chunk_1 point to same tensors
            )?;

            let mask_cpu = SyncWrapper::new_tb(Box::new(mask.to_device(Cpu)));
            drop(mask);

            trace!("find_motifs_by_euc_dist: mask_cpu=\n{mask_cpu} mask_tril=\n{mask_tril}");
            if log_enabled!(Level::Debug) {
                debug!("find_motifs_by_euc_dist: count_nonzero={}", mask_cpu.count_nonzero_dim_intlist(&[1, 2]));
            }

            Ok(mask_cpu)

        },
        &|run_config_id, mask_cpu, chunk_0, chunk_1| {
            let channel_trackers = run_config__channels_trackers.get(&run_config_id).expect("channel_trackers");

            debug!("find_motifs_by_euc_dist: mask_cpu.size={:?}", mask_cpu.size());

            channel_trackers.iter().enumerate()/*.par_bridge()*/.for_each(|(channel, mt)| {
                let mask_cpu_flattened = mask_cpu.i((channel as i64, .., ..)).flatten(0, 1);
                let mut channel_mask:BoolTensorIterator = (&mask_cpu_flattened).into();
                debug!("find_motifs_by_euc_dist: channel={channel} mask_cpu_flattened.size={:?}", mask_cpu_flattened.size());
                trace!("find_motifs_by_euc_dist: mask_cpu_flattened={mask_cpu_flattened}");

                chunk_0.iter().for_each(|slice_0|{
                    chunk_1.iter().for_each(|slice_1|{

                        let mask = channel_mask.next().unwrap();
                        trace!("find_motifs_by_euc_dist: run_config_id={run_config_id} slice_0={slice_0:?} slice_1={slice_1:?} channel={channel} mask={mask}");

                        if mask
                            && (slice_0.0 != slice_1.0 || slice_0.1 != slice_0.1
                        ) {
                            mt.lock().track(
                                slice_0.clone(),
                                slice_1.clone(),
                                0.0,
                            );
                        }
                    });
                });
                mt.lock().dump();
            });

        },
        loader,
        &devices
    )?;

    //TODO
    // for tr in run_config__channels_trackers.iter().flat_map(|(_, v)|v.iter()){
    //     // tr.lock().finish();
    // }

    let total_trackers  = run_config__channels_trackers.values().map(|v| v.len()).sum::<usize>();
    info!("setting motifs in {} configs,  {total_trackers} trackers on {} threads", run_config_ids.len(), rayon::current_num_threads());
    let num_trackers_per_thread = max(1, total_trackers/rayon::current_num_threads());
    let run_config__channels_trackers_vec = run_config__channels_trackers.values().flat_map(|v| v.into_iter()).collect::<Vec<_>>();

    let mut trackers_done = AtomicU32::new(0);
    run_config__channels_trackers_vec.chunks(num_trackers_per_thread).par_bridge().for_each(|chunk| {
        for mt in chunk {
            mt.lock().build_groups(store).expect("build_groups");
            info!(" {} trackers done ", trackers_done.fetch_add(1, Relaxed));
        }
    });

    Ok(None)
}

#[derive(Clone, PartialEq, Debug)]
pub struct FindMotifsRunConfig {
    id:Option<usize>,
    step: u32,
    window: u32,
    dist_quantile: f32,
    min_tracker_step: Option<usize>,
    filter: Option<InputFilter>,
    descr: String,
}

impl RunConfig for FindMotifsRunConfig {
    fn to_properties_map(&self) -> HashMap<String, Val> {
        let mut map = HashMap::from([
            ("FindMotifsRunConfig.step".into(), UInt(self.step)),
            ("FindMotifsRunConfig.window".into(), UInt(self.window)),
            ("FindMotifsRunConfig.dist_quantile".into(), self.dist_quantile.into()),
            ("FindMotifsRunConfig.descr".into(), self.descr.as_str().into()),
        ]);

        if let Some(filter) = self.filter.as_ref() {
            map.extend(thoosa_shared::val::prefix_properties("FindMotifsRunConfig.filter", filter.to_properties_map()).into_iter());
        }

        map
    }

    //TODO set id
    fn from_properties_map(run_id: usize, properties: HashMap<String, Val>) -> Self {
        let properties = thoosa_shared::val::unprefix_properties("FindMotifsRunConfig", &properties);
        FindMotifsRunConfig::new(
            None,
            properties.get("step").expect("no step").as_u32().expect("step as_u32"),
            properties.get("window").expect("no window").as_u32().expect("window as_u32"),
            properties.get("dist_quantile").expect("no dist_quantile").as_f32().expect("dist_quantile as_f32"),
            properties.get("descr").expect("no descr").as_str().expect("descr as_str"),
            InputFilter::from_properties_map(&thoosa_shared::val::unprefix_properties("filter", &properties)),
        )
    }

    fn set_id(&mut self, id:usize) {
        self.id.replace(id);
    }

    fn type_name() -> &'static str {
        "FindMotifs"
    }
}

impl FindMotifsRunConfig {
    pub fn new(id: Option<usize>,
               step: u32,
               window: u32,
               dist_quantile: f32, //TODO move to common
               descr: impl Into<String>,
               filter: Option<InputFilter>,
    ) -> Self {
        Self {
            id,
            step,
            window,
            dist_quantile,
            filter,
            min_tracker_step: None,//TODO
            descr: descr.into(),
        }
    }

    pub fn window(&self) -> i64 {
        self.window as i64//TODO
    }

    pub fn step(&self) -> usize {
        self.step as usize //TODO
    }
}

fn chunk_and_slice<G, CTX>(
    resources: Vec <Resource>,
    chunk_size: usize,
    run_configs:&mut Vec<(FindMotifsRunConfig, CTX)>,
    mut callback: G,
    exec: &Arc<RwLock< LoadRunner<Box<Tensor>>>>,
    devices:&[Device]
) -> Result<(), ThoosaError>
where
    CTX: Send + Sync,
    G: Fn(&ChunkOfSlicedTensors, &mut (FindMotifsRunConfig, CTX)) -> Result<(), ThoosaError> + Send + Sync
{
    let mut resource_chunks = resources.chunks(chunk_size)
        .map(|v| v.iter().map(|k| k.clone()).collect::<Vec<Resource>>())//TODO k.clone()
        .collect::<Vec<Vec<Resource>>>();//[Resource]

    resource_chunks.push(vec![]); //to get last window processed


    let area_0 = exec.write().get_available_cache_area();

    exec.write().preload(&resource_chunks[0], area_0);

    if resource_chunks.len() > chunk_size {
        exec.write().preload(&resource_chunks[chunk_size], area_0);
    }

    //first element in each window to be loaded and processed, second - preloaded
    for (chunk_i_0, window_0) in resource_chunks.windows(2).enumerate() {
        //window_0 - current,preload windows
        debug!("chunk_and_slice: chunk_i_0={chunk_i_0} window_0={window_0:?}");

        if window_0[1].len() > 0 {
            exec.write().preload(&window_0[1],  area_0);
        }

        let wg_0 = WaitGroup::new();

        let window_0_0 = window_0[0].clone();
        let window_0_0_clone = window_0_0.iter().map(|r|r.id()).collect::<Vec<ResourceId>>();

        let wg_0_clone = wg_0.clone();
        let window_0_tensors = Arc::new(RwLock::new(vec![]));
        let window_0_tensors_arc = Arc::clone(&window_0_tensors);

        exec.write().exec(|loader, keys_purge| {
            async move {

                for key in &window_0_0 {
                    debug!("chunk_and_slice:  exec:callback_arc: key={key:?}");

                    let sync_wrapper = SyncWrapper::new_tb(
                        Loader::<Box<Tensor>>::load_future(Arc::clone(&loader), key.clone(), area_0).await
                    );
                    window_0_tensors_arc.write().push(sync_wrapper);
                }
                drop(wg_0_clone);

            }.boxed()
        });

        wg_0.wait();

        run_configs.iter_mut().zip(devices.into_iter().cycle())
            // .par_bridge() //TODO spread among devices
            .for_each(|(run_config_with_ctx, device)| {
                let mut split_queue = ChunkOfSlicedTensors::start_split_queue(ChunkOfSlicedTensors::slice_tensors_chunk(
                    run_config_with_ctx.0.step,
                    run_config_with_ctx.0.window,
                    &window_0_tensors.read(),
                    window_0_0_clone.clone(),
                    Cpu
                ));
                while let Some(mut chunk_0_split) = split_queue.next() {
                    match callback(
                        &chunk_0_split,
                        run_config_with_ctx,
                    ) {
                        Ok(_) => {
                            chunk_0_split.drop_tensor();

                        }
                        Err(e) => {
                            debug!("got e={e:?}, splitting");

                            split_queue.split(chunk_0_split);
                        }
                    }

                }
            });


        let i_1 = Instant::now();

        exec.write().purge_cache_keys(window_0[0].clone(), area_0);
        info!("done {chunk_i_0} chunks out of {} in {} mls", resource_chunks.len(), i_1.elapsed().as_millis());
    }
    Ok(())
}


fn chunk_and_slice_2x<G, C>(
    resources: Vec <Resource>,
    chunk_size: usize,
    run_configs:&Vec<Arc<FindMotifsRunConfig>>,
    mut gpu_fn: G,
    mut cpu_fn: &C,
    loader: &Arc<RwLock< LoadRunner<Box<Tensor>>>>,
    devices:&[Device]
) -> Result<(), ThoosaError>
where
    G: FnMut(&Arc<ChunkOfSlicedTensors>, &ChunkOfSlicedTensors, &Arc<FindMotifsRunConfig>) -> Result<SyncWrapper<Box<Tensor>>, ThoosaError> + Send + Sync , //&mut Arc<Vec<Mutex<SimpleMotifTracker<SqlMotifTrackerStore>>>>,
    C: Fn(usize, SyncWrapper<Box<Tensor>>, Vec<TensorIdSliceIdx>, Vec<TensorIdSliceIdx>) + Send + Sync,
{
    debug!("chunk_and_slice_2x: chunk_size={chunk_size:?} run_configs={run_configs:?}");
    trace!("chunk_and_slice_2x: resources={resources:?}");

    rayon::scope(|s| {
        let mut resource_chunks = resources.chunks(chunk_size)
            .map(|v| v.iter().map(|k| k.clone()).collect::<Vec<Resource>>()) //TODO k.clone()
            .collect::<Vec<Vec<Resource>>>(); //[Resource]

        resource_chunks.push(vec![]); //to get last window processed

        let (tx, rx) = crossbeam_channel::unbounded::<(usize, SyncWrapper<Box<Tensor>>, Vec<TensorIdSliceIdx>, Vec<TensorIdSliceIdx>)>();
        let trackers_pool = rayon::ThreadPoolBuilder::new().num_threads(32).spawn_handler(|thread_builder| {
            let b = thread::Builder::new();
            b.name(format!("{}/{:?}", "trackers_pool", thread_builder.index()))
                .spawn(|| thread_builder.run())?;

            Ok(())
        }).build().expect("trackers_pool");

        let area_0 = loader.write().get_available_cache_area();
        let area_1 = loader.write().get_available_cache_area();

        loader.write().preload(&resource_chunks[0], area_0);

        if resource_chunks.len() > chunk_size {
            loader.write().preload(&resource_chunks[chunk_size], area_0);
        }

        let mut sleept_times = Arc::new(AtomicU64::new(0));
        let mut sleept_times_arc = Arc::clone(&sleept_times);
        let mut num_tasks = Arc::new(AtomicU64::new(0));
        let mut num_tasks_arc = Arc::clone(&num_tasks);

        s.spawn(move |scope| {

            //first element in each window to be loaded and processed, second - preloaded

            for (chunk_i_0, window_0) in resource_chunks.windows(2).enumerate() {
                //window_0 - current,preload windows
                debug!("chunk_and_slice: chunk_i_0={chunk_i_0} window_0={window_0:?}");

                if window_0[1].len() > 0 {
                    loader.write().preload(&window_0[1], area_0);
                }

                let wg_0 = WaitGroup::new();

                let window_0_0 = window_0[0].clone();
                let window_0_0_clone = window_0_0.iter().map(|r| r.id()).collect::<Vec<ResourceId>>();

                let wg_0_clone = wg_0.clone();
                let window_0_tensors = Arc::new(RwLock::new(vec![]));
                let window_0_tensors_arc = Arc::clone(&window_0_tensors);

                loader.write().exec(|loader, keys_purge| {
                    async move {
                        for key in &window_0_0 {
                            debug!("0. exec:callback_arc: key={key:?}");

                            let sync_wrapper = SyncWrapper::new_tb( //TODO remove SyncWrapper? it goes only to run_configs_applied_to_chunk_0
                                Loader::<Box<Tensor>>::load_future(Arc::clone(&loader), key.clone(), area_0).await
                            );
                            window_0_tensors_arc.write().push(sync_wrapper);
                        }
                        drop(wg_0_clone);
                    }.boxed()
                });

                wg_0.wait();

                let run_configs_applied_to_chunk_0:Vec<Arc<ChunkOfSlicedTensors>> = run_configs.par_iter().enumerate()
                    .map(|(i, run_config)| {
                        Arc::new(ChunkOfSlicedTensors::slice_tensors_chunk(
                            run_config.step,
                            run_config.window,
                            &window_0_tensors.read(),
                            window_0_0_clone.clone(),
                            Cpu,
                        ))//.into()
                    }).collect::<Vec<_>>();

                drop(window_0_tensors);

                let i_1 = Instant::now();
                for (chunk_i_1, window_1) in resource_chunks.windows(2).into_iter().skip(chunk_i_0).enumerate() {
                    debug!("chunk_and_slice: chunk_i_1={chunk_i_1} window_1={window_1:?}");

                    if window_1[1].len() > 0 {
                        loader.write().preload(&window_1[1], area_1);
                    }
                    trace!("chunk_and_slice: area_1={area_1:064b}");

                    let wg_1 = WaitGroup::new();

                    let window_1_0 = window_1[0].clone();
                    let window_1_0_clone = window_1_0.iter().map(|r| r.id()).collect::<Vec<ResourceId>>();

                    let wg_1_clone = wg_1.clone();
                    let window_1_tensors = Arc::new(RwLock::new(vec![]));
                    let window_1_tensors_arc = Arc::clone(&window_1_tensors);

                    loader.write().exec(|loader, keys_purge| {
                        async move {
                            for key in &window_1_0 {
                                debug!("chunk_and_slice: 1. exec:callback_arc: key={key:?}");

                                let sync_wrapper = SyncWrapper::new_tb(Loader::<Box<Tensor>>::load_future(Arc::clone(&loader), key.clone(), area_1).await);
                                window_1_tensors_arc.write().push(sync_wrapper);
                            }
                            drop(wg_1_clone);
                        }.boxed()
                    });
                    wg_1.wait();

                    run_configs.iter()
                        .zip(run_configs_applied_to_chunk_0.iter())
                        // .par_bridge() //TODO spread among devices
                        .for_each(|(run_config, chunk_0)| {

                            run_split_on_oom(
                                ChunkOfSlicedTensors::start_split_queue(ChunkOfSlicedTensors::slice_tensors_chunk(
                                    run_config.step,
                                    run_config.window,
                                    &window_1_tensors.read(),
                                    window_1_0_clone.clone(),
                                    Cpu
                                )),
                                |chunk_1_split|{
                                    let num_tasks_v = num_tasks_arc.load(Relaxed);
                                    if num_tasks_v > 100 {//TODO config
                                        debug!("chunk_and_slice: num_tasks_v={num_tasks_v}");
                                        sleept_times_arc.fetch_add(1, Relaxed);
                                        thread::sleep(Duration::from_millis(3000));//TODO config
                                    }
                                    gpu_fn(
                                        &chunk_0,
                                        &chunk_1_split,
                                        run_config,
                                    )
                                },
                                |cpu_t, chunk_1_split|{
                                    chunk_1_split.drop_tensor();

                                    match tx.send((
                                        run_config.id.expect("id"),
                                        cpu_t,
                                        chunk_0.to_vec(),
                                        chunk_1_split.to_vec())
                                    ) {
                                        Ok(_) => {}
                                        Err(e) => {
                                            error!("e:{e:?}"); //TODO panic?
                                        }
                                    }
                                },
                                3 //TODO config
                            );
                        });

                    loader.write().purge_cache_keys(window_1[0].clone(), area_1);
                    print!(".");
                    std::io::stdout().flush().unwrap();
                }
                loader.write().purge_cache_keys(window_0[0].clone(), area_0);
                info!("done {chunk_i_0} chunks out of {} in {} mls, slept {sleept_times_arc:?} times", resource_chunks.len(), i_1.elapsed().as_millis());
            }
            drop(tx);
        });


        trackers_pool.scope(|s| {
            for (run_config_it, t, chunk_0, chunk_1) in rx.iter() {

                num_tasks.fetch_add(1, Relaxed);
                let num_tasks = Arc::clone(&num_tasks);
                s.spawn(move |scope| {
                    cpu_fn(run_config_it, t, chunk_0, chunk_1);
                    num_tasks.fetch_sub(1, Relaxed);
                });
            }
        });

    });
    Ok(())
}

fn collect_dists<'a, F, C>(
    chunk_0: Arc<RwLock<ChunkOfSlicedTensors>>,
    chunk_1: Arc<RwLock<ChunkOfSlicedTensors>>,
    distances:&Vec<f32>,
    step: i64,
    mut listener:F,
    condition:C
) where F:FnMut((TensorIdSliceIdx, TensorIdSliceIdx, f32)),
        C:Fn(&f32)->bool,
{

    let mut distances_i = 0;

    let chunk_0_read_guard = chunk_0.read();
    let chunk_1_read_guard = chunk_1.read();

    chunk_0_read_guard.sizes.iter().enumerate().for_each(|(t_idx_0, t_size_0)| {
        for (slice_0_idx, s_0) in (0..*t_size_0).enumerate() {

            for (t_idx_1, t_size_1) in chunk_1_read_guard.sizes.iter().enumerate() {
                for (slice_1_idx, s_1) in (0..*t_size_1).enumerate() {
                    //println!("t_idx_0={t_idx_0} slice_0_idx={slice_0_idx} t_idx_1={t_idx_1} slice_1_idx={slice_1_idx} d={}", distances[distances_i]);

                    if condition(&distances[distances_i])
                    //if not on diagonal
                    && (
                        chunk_0_read_guard.resource_ids.get(t_idx_0).and_then(|k_0| chunk_1_read_guard.resource_ids.get(t_idx_1).map(|k_1| k_1 != k_0)).unwrap_or(false)
                        ||
                        slice_0_idx != slice_1_idx
                    ) {
                        listener(
                            (
                                TensorIdSliceIdx(chunk_0_read_guard.resource_ids[t_idx_0].clone(), (slice_0_idx * step as usize) as u32),
                                TensorIdSliceIdx(chunk_1_read_guard.resource_ids[t_idx_1].clone(), (slice_1_idx * step as usize) as u32),
                                distances[distances_i]
                            ),
                        );
                    }

                    distances_i += 1;
                }
            }
        }
    });

}



//TODO trait for MotifTracker
#[derive(Debug)]
pub(crate) struct SimpleMotifTracker<'a> {
    last:Option<SlicesMatch>,
    motifs: Vec<SlicesMatch>, //i_0, match_0, match_1, dist
    dumper: Box<dyn MotifTrackerStore + 'a>, //lifetime is for DummyDumper
    min_step:Option<usize>,//TODO remove, all filtering is in gpu
    max_slices_counts: Option<u32>,
}

impl<'a> SimpleMotifTracker<'a> {

    pub fn new(dumper: Box<dyn MotifTrackerStore+'a>, min_step:Option<usize>, max_slices_counts: Option<u32>) -> SimpleMotifTracker {
        Self {
            last:None,
            motifs: vec![],
            dumper,
            min_step,
            max_slices_counts
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
                if let Some(last) = last {//TODO
                    if
                        last.2 <= dist &&
                        last.0.0 == i_0.0 && last.1.0 == i_1.0 && //still same chunks
                        (i_0.1 - last.0.1  > min_step as u32 || i_1.1 - last.1.1  > min_step as u32) {//i_. .1 < last.1.1 if next tensor
                        self.motifs.push(last);
                    }
                }
                self.last.replace(SlicesMatch(i_0, i_1, dist));
            }
        }

    }

    fn dump(&mut self) {
        debug!("SimpMotifTracker::dump: self.motifs={}", self.motifs.len());
        if let Some(last) = self.last.take(){
            self.motifs.push(last);
        }

        if self.motifs.len() > 0 {
            self.dumper.save_slices(mem::take(&mut self.motifs)).expect("dumper");
        }
    }

    fn build_groups(&mut self, store: &Arc<Mutex<Box<dyn Store>>>) -> Result<(), ThoosaError>{
        self.dumper.load_slices_into_groups(store, self.max_slices_counts)
    }

    #[cfg(test)]
    pub fn motifs(&self) -> &Vec<SlicesMatch> {
        &self.motifs
    }
}

/**
 * series - [channel x data]
 * motifs = [channel x motif x data slice]
 */
#[allow(dead_code)]
fn clc_dot_matches(series: &Tensor, motifs: &Tensor, step: usize, threshold: f32) -> Result<Tensor, ThoosaError> {
    let channels_num = series.size()[0];
    let series_len = series.size()[1];
    let motifs_num = motifs.size()[1];
    let motif_len = motifs.size()[2];

    //channels, motif, motif x series dots
    let mut matches: Vec<Vec<Vec<f32>>> = (0..channels_num).fold(Vec::new(), |mut a, _v| {
        a.push((0..motifs_num).fold(Vec::new(), |mut a, _| {
            a.push(Vec::new());
            a
        }));
        a
    });

    for i in (0..=series_len - motif_len).step_by(step) {
        for m in 0..motifs_num {
            let dp = dot_norm(
                1,
                &series.i((.., i..i + motif_len)),
                &motifs.i((.., m, ..)),
            )?;

            for (channel, dot) in from::<Vec<f32>>(&dp).into_iter().enumerate() {
                matches[channel][m as usize].push(dot);
            }
        }
    }

    let matches = Tensor::vstack(matches.into_iter().map(|channel_motifs| {
        Tensor::from_slice(channel_motifs.into_iter().map(|motifs_dots| {
            motifs_dots.into_iter().map(|v| if v > threshold { 1 } else { 0 }).sum()
        }).collect::<Vec<i32>>().as_slice())
    }).collect::<Vec<Tensor>>().as_slice());

    Ok(matches)
}


/**
 * series - [channel x data]
 * motifs = [channel x motif x data slice]
 * channel_motif_ids - max distances for motifs
 */
fn clc_dist_matches(target: &Tensor, motifs: Arc<RwLock<SyncWrapper<Box<Tensor>>>>, step: usize, motifs_distances: Arc<RwLock<HashMap<usize, f32>>>, channel_motif_ids: Arc<RwLock<Vec<Vec<usize>>>>) -> Result<Tensor, ThoosaError> {
    let channels_num = target.size()[0];

    let motifs_num = motifs.read().size()[1];
    let motif_len = motifs.read().size()[2];


    //channels, motif, motif x series dots
    let mut channels_matches: Vec<Vec<Vec<f32>>> = (0..channels_num).map(|_| Vec::new()).collect();

    let slices = slice_t(&target, step as u32, motif_len as u32);//TODO as

    for c in 0..channels_num {
        let t_0 = slices.i((.., c, ..));

        let t_1 = motifs.read().i((c, .., ..));

        let dt = dist_d2(&t_0, &t_1).expect(format!("dist_d2 on channel={}", c).as_str());
        debug!("clc_dist_matches: t_0={:?} t_1={:?} dt.size={:?}", t_0.size(), t_1.size(), dt.size());

        for motif in 0..motifs_num {
            let vec = from::<Vec<f32>>(&dt.i((.., motif)));
            debug!("clc_dist_matches: c={} motif={} vec={:?}", c, motif,  vec);

            channels_matches[c as usize].push(vec);
        }
    }


    debug!("clc_dist_matches: channels_matches_lens={:?}", channels_matches.iter().map(|c| c.len()).collect::<Vec<usize>>());
    debug!("clc_dist_matches: channel_motif_ids_lens={:?}", channel_motif_ids.read().iter().map(|c| c.len()).collect::<Vec<usize>>());

    let channel_motif_ids: Vec<Vec<Option<usize>>> = channel_motif_ids.read().iter().map(|ids_in_c| {
        let mut x: Vec<Option<usize>> = ids_in_c.iter().map(|v| Some(*v)).collect();

        for _ in x.len()..motifs_num as usize {
            x.push(None)
        }

        x
    }).collect();

    // assert!(channels_matches_lens.eq(&channel_motif_ids_lens), "channels_matches_lens={:?}, channel_motif_ids_lens={:?}", channels_matches_lens, channel_motif_ids_lens);


    let matches = Tensor::vstack(channels_matches.into_iter().zip(channel_motif_ids.iter()).map(|(channel_motifs, motif_ids)| {
        debug!("clc_dist_matches: motif_ids={:?}", motif_ids);
        debug!("clc_dist_matches: channel_motifs={:?}", channel_motifs);
        debug!("clc_dist_matches: motifs_distances={:?}", motifs_distances);
        Tensor::from_slice(channel_motifs.into_iter().zip(motif_ids.into_iter()).map(|(motifs_dists, motif_id_op)| {
            // debug!("motifs_dists={:?}", motifs_dists);

            motifs_dists.into_iter().map(|motif_dist| {

                // debug!("clc_dist_matches: motif_id={:?} motifs_distances.read().get(motif_id)={:?} motif_dist={}", motif_id_op, motifs_distances.read().get(&motif_id_op.unwrap_or(0)), motif_dist);

                //= in >=  for tests where avg motif dist 0.0
                if motif_id_op.as_ref().map(|motif_id| motifs_distances.read().get(motif_id).expect(format!("no motif with motif_id={}", motif_id).as_str()) >= &motif_dist).unwrap_or(false) {
                    1
                } else {
                    0
                }
            }).sum()
        }).collect::<Vec<i32>>().as_slice())
    }).collect::<Vec<Tensor>>().as_slice());

    Ok(matches)
}

fn clc_motifs_matches_for_label<L>(
    motifs: Arc<RwLock<SyncWrapper<Box<Tensor>>>>,
    all_zero: SyncWrapper<Box<Tensor>>,
    channels_motifs_matches_labels: Arc<Mutex<Vec<Vec<(Vec<f64>, Vec<L>)>>>>, //output TODO return it
    resources: Vec<Resource>,
    executor: Arc<RwLock<LoadRunner<Box<Tensor>>>>,
    label: L,
    step: usize,
    motifs_distances: Arc<RwLock<HashMap<usize, f32>>>,
    channel_motif_ids: Arc<RwLock<Vec<Vec<usize>>>>,
) -> Result<(), ThoosaError>
where
    L: Clone + Send + Display + Debug + 'static,
{
    let count = AtomicU64::new(0);

    let channels_num = motifs.read().size()[0] as usize; //TODO read
    let motifs_num = motifs.read().size()[1] as usize;

    debug!("clc_motifs_matches_for_label: label={label} channels_num={} motifs_num={} motifs.read().size()={:?} step={step}", channels_num, motifs_num, motifs.read().size());


    let wg = WaitGroup::new();
    let wg_in_loader = wg.clone();
    let area = executor.write().get_available_cache_area();

    executor.write().exec(move |loader, keys_purge| {
        async move {
            for resource in &resources {
                debug!("clc_motifs_matches_for_label: resource={resource:?}");

                // let matches = clc_dot_matches(&eeg?, &motifs, step, threshold)?;

                let target = &*Loader::<Box<Tensor>>::load_future(Arc::clone(&loader), resource.clone(), area).await;
                let matches = clc_dist_matches(
                    target,
                    Arc::clone(&motifs), step, Arc::clone(&motifs_distances), Arc::clone(&channel_motif_ids)
                ).unwrap();

                for motif_i in 0..motifs_num {
                    for channel_i in 0..channels_num {
                        let motif_t = motifs.read().i((channel_i as i64, motif_i as i64));

                        let num_matches_label = if motif_t.equal(&all_zero) { //TODO this is to handle zero motifs which were added to pad up to max motifs num per channel
                            (0., label.clone())
                        } else {
                            let matches_t = matches.i((channel_i as i64, motif_i as i64));
                            let num_matches = matches_t.double_value(&[]);
                            debug!("clc_motifs_matches_for_label: channel_i={channel_i} motif_i={motif_i} matches_t={matches_t} num_matches={num_matches}");

                            (num_matches, label.clone())
                        };

                        let mut channels_motifs_classes_guard = channels_motifs_matches_labels.lock();
                        channels_motifs_classes_guard[channel_i][motif_i].0.push(num_matches_label.0);
                        channels_motifs_classes_guard[channel_i][motif_i].1.push(num_matches_label.1);
                    }
                }

                let count_ = count.fetch_add(1, Ordering::Relaxed);
                if count_ % 100 == 0 {
                    info!("dot_motifs: count={}", count_);
                }
                debug!("clc_motifs_matches_for_label: dot_motifs: count={}", count_);
            };

            debug!("clc_motifs_matches_for_label: channels_motifs_classes_guard[channel_i][motif_i]={:?}", channels_motifs_matches_labels.lock());
            keys_purge(&resources, area);
            drop(wg_in_loader);
        }.boxed()
    });

    wg.wait();


    Ok(())
}

/**
TODO  selects motifs based on correlation to a label only
*/
#[derive(Debug, Args, Clone)]
pub struct SelectMotifsParams {
    #[arg(long)]
    find_motifs_run_ids: Vec<usize>,

    #[arg(long)]
    pub resources_group: String,

}

impl SelectMotifsParams {

    pub fn find_motifs_run_ids(&self) -> &Vec<usize> {
        &self.find_motifs_run_ids
    }

    pub fn dummy(run_ids: Vec<usize>, resources_group: String) -> Self {
        SelectMotifsParams {
            find_motifs_run_ids: run_ids,
            resources_group,
        }
    }


}

impl Params<SelectMotifsRunConfig, Labeled, ()> for SelectMotifsParams {
    fn to_run_configs(&self) -> Vec<SelectMotifsRunConfig> {
        vec![SelectMotifsRunConfig::new()]
    }

    fn resource_config(&self) -> Labeled {
        Labeled::new(self.resources_group.clone())
    }

    fn common_config(&self) -> () {
        ()
    }
}

#[derive(Clone, PartialEq, Debug)]
enum MotifSelector {
    ByLabelVariance//TODO
}

#[derive(Clone, PartialEq, Debug)]
pub struct SelectMotifsRunConfig{
    id:Option<usize>,
    motif_selector: MotifSelector
}

impl SelectMotifsRunConfig {
    pub fn new() -> Self {
        Self {
            id:None,
            motif_selector: MotifSelector::ByLabelVariance
        }
    }
}

impl RunConfig for SelectMotifsRunConfig {

    fn from_properties_map(run_id: usize, properties: HashMap<String, Val>) -> Self {
        Self {
            id: Some(run_id),
            motif_selector: MotifSelector::ByLabelVariance,
        }
    }

    fn set_id(&mut self, id: usize) {
        assert!(self.id.replace(id).is_none());
    }

    fn type_name() -> &'static str {
        "SelectMotifs"
    }
}

fn select_motifs_by_label_variance(labels_channels_motifs_counts: &Tensor, mask: &Tensor) -> (Tensor, Tensor) {
    debug!("select_motifs_by_label_variance: labels_channels_motifs_counts={:?}", labels_channels_motifs_counts.size());

    let var = labels_channels_motifs_counts.var_dim(0, true, false);
    let mean =labels_channels_motifs_counts.mean_dim(0, false, Kind::Float).f_add_scalar(1e-7).unwrap();

    trace!("select_motifs_by_label_variance: mask=\n{mask}");
    trace!("select_motifs_by_label_variance: labels_channels_motifs_counts=\n{labels_channels_motifs_counts}");
    trace!("select_motifs_by_label_variance: var=\n{var}");
    trace!("select_motifs_by_label_variance: mean=\n{mean}");
    let idx_dispersion= var/mean;

    trace!("select_motifs_by_label_variance: idx_dispersion=\n{idx_dispersion}");

    //idx_dispersion has 0 where mask 0 cause counts dists filled with -f32::MIN so no need to
    //let idx_dispersion_masked = &idx_dispersion * &mask_float;
    let idx_dispersion_sums = idx_dispersion.sum_dim_intlist(1, false, Kind::Float);
    let counts = mask.sum_dim_intlist(1, false, Kind::Float);

    let idx_dispersion_mean = (idx_dispersion_sums/counts)
        .unsqueeze(1)//by channel
        .clamp_min(1.); //under-dispersed

    trace!("select_motifs_by_label_variance: idx_dispersion_mean=\n{idx_dispersion_mean}");

    let max = idx_dispersion.amax(1, true);//per channel
    trace!("select_motifs_by_label_variance: max=\n{max}");

    (idx_dispersion.ge_tensor(&idx_dispersion_mean), idx_dispersion/ max)
}

 pub fn select_motifs_for_label(
     input_run_ids:Option<&Vec<usize>>,
     select_motif_run_configs: Vec<SelectMotifsRunConfig>,
     resource_config: &Labeled,
     common_config: &(),
     ws: &Arc<RwLock<Box<dyn Workspace>>>,
     store: &Arc<Mutex<Box<dyn Store>>>,
     loader: &Arc<RwLock< LoadRunner<Box<Tensor>>>>,
     devices:Vec<Device>
) -> Result<Option<Vec<Box<dyn SavableOptResult>>>, ThoosaError>{

    let motifs_area = loader.write().get_available_cache_area();
    let labels_resources = resource_config.to_resources(store)?;

    debug!("select_motifs_for_label_2: select_motif_run_configs={select_motif_run_configs:?}");
    trace!("select_motifs_for_label_2: labels_resources={labels_resources:?}");

    let mut find_motifs_run_config_motifs_tensors = input_run_ids.expect("input_run_ids").iter().map(|find_motif_run_id|{

        let resource_id_motif_slices = store.lock().list_motifs_slices(*find_motif_run_id)?;
        trace!("select_motifs_for_label_2: resource_id_motif_slices={resource_id_motif_slices:?}");

        let find_motifs_run_config = FindMotifsRunConfig::from_properties_map(*find_motif_run_id, store.lock().list_run_config_values(*find_motif_run_id)?);

        motifs_to_tensor(
            resource_id_motif_slices,
            loader,
            find_motifs_run_config.window(),
            ws.read().channels() as i64,
            Cpu,
            motifs_area,
        ).map(|motifs_tensors| {
            let size = &motifs_tensors.channels_motifs_slices.size()[0..=1];//channels x motifs for motif sums

            debug!("select_motifs_for_label_2: {find_motifs_run_config:?} motifs_tensor.size()={:?}", motifs_tensors.channels_motifs_slices.size());
            trace!("select_motifs_for_label_2: motifs_tensors.channels_motifs_ids=\n{} channels_motifs_mask=\n{} motifs_tensors.channels_motifs_avg_dists=\n{}", motifs_tensors.channels_motifs_ids, motifs_tensors.channels_motifs_mask, motifs_tensors.channels_motifs_avg_dists);

            (
                find_motifs_run_config,
                (
                    motifs_tensors,
                    HashMap::from_iter(
                        labels_resources
                            .iter()
                            .map(|(label, _)|
                                (
                                    label.clone(),
                                    SyncWrapper::new_tb(Box::new(Tensor::zeros(size.clone(), INT64_CPU)))//motif counts
                                )
                            )
                    )
                )
            )
        })

    }).collect::<Result<Vec<(FindMotifsRunConfig, (MotifsTensors, HashMap<String, SyncWrapper<Box<Tensor>>>))>, ThoosaError>>()?;
    loader.write().purge_cache(motifs_area);


    debug!("select_motifs_for_label_2: motifs_tensors loaded");

    // let mut find_motifs_run_configs = find_motifs_run_config_motifs_tensors.iter().map(|v|v.0.clone()).collect::<Vec<_>>();

    for (label, resources) in labels_resources {
        debug!("select_motifs_for_label_2: label={label:?} resources={resources:?}");
        chunk_and_slice(
            resources,
            2,//TODO params
            &mut find_motifs_run_config_motifs_tensors,

            |chunk, (cfg, (motif_tensors, label_motif_matches_sums))|{

                let broadcast_to_size = [motif_tensors.channels_motifs_slices.size()[0], motif_tensors.channels_motifs_slices.size()[1], chunk.t.size()[1]];

                debug!("select_motifs_for_label_2: label={label} resources cfg={cfg:?} chunk.t.size={:?} motifs_tensors={:?} broadcast_to_size={broadcast_to_size:?}", chunk.t.size(), motif_tensors); //chunk.t

                let (mask, _) = clc_dist_d2_and_mask(
                    motif_tensors.channels_motifs_slices.to_device(devices[0]),
                    chunk.t.to_device(devices[0]),
                    false,
                    //compare distances between tensor and motifs to motif's average distance
                    &mut [LE {v: motif_tensors.channels_motifs_avg_dists.unsqueeze(2).broadcast_to(broadcast_to_size).to_device(devices[0])}]
                )?;
                let mask = mask.to_device(Cpu);
                trace!("select_motifs_for_label_2: label={label} mask=\n{mask}");

                let sums = mask.sum_dim_intlist(2, false, Kind::Int64);//channels x motif matches
                trace!("select_motifs_for_label_2: label={label} sums=\n{sums}");

                label_motif_matches_sums.get_mut(&label).expect(format!("no motif sums tensor for {label}").as_str()).add_assign(sums.into());

                if log_enabled!(Level::Trace) {
                    for (label, motif_sums) in label_motif_matches_sums.iter() {
                        trace!("select_motifs_for_label_2: label={label} motif_sums=\n{motif_sums}");
                    }
                }

                Ok(())
            },
            loader,
            &devices
        )?;
    }

    find_motifs_run_config_motifs_tensors.iter().try_fold(vec![],  |mut a, (find_motif_run_cfg, (motifs_tensors, label_motif_matches_sums))|{
        debug!("select_motifs_for_label_2: cfg.window={} channels_motif_ids=\n{:?}", find_motif_run_cfg.window, motifs_tensors.channels_motifs_ids);
        trace!("select_motifs_for_label_2:  avg_dists_t=\n{}", motifs_tensors.channels_motifs_avg_dists);
        if log_enabled!(Level::Trace) {
            for (label, motif_sums) in label_motif_matches_sums.iter() {
                trace!("select_motifs_for_label_2: label={label} final motif_sums=\n{motif_sums}");
            }
        }
        if label_motif_matches_sums.len() > 0 {
            let labels_channels_motifs_counts = Tensor::f_stack(
                label_motif_matches_sums.into_iter().map(|t| &***t.1).collect::<Vec<_>>().as_slice(),
                0,
            )?.totype(Kind::Float);

            for run_config in &select_motif_run_configs {

                match run_config.motif_selector {
                    //TODO other selectors
                    MotifSelector::ByLabelVariance => {

                        let (good_motifs, motifs_rates) = select_motifs_by_label_variance(&labels_channels_motifs_counts, &motifs_tensors.channels_motifs_mask);
                        debug!("select_motifs_for_label_2: good_motifs=\n{good_motifs} motifs_tensors.channels_motifs_ids\n{} motifs_tensors.channels_motifs_avg_dists=\n{}", motifs_tensors.channels_motifs_ids, motifs_tensors.channels_motifs_avg_dists);

                        let good_motif_ids_rates = from::<Vec<Vec<i64>>>(&motifs_tensors.channels_motifs_ids).into_iter().flat_map(|v| v.into_iter()).zip(
                            from::<Vec<Vec<bool>>>(&good_motifs)
                                .into_iter()
                                .flat_map(|v| v.into_iter())
                                .zip(from::<Vec<Vec<f32>>>(&motifs_rates).into_iter().flat_map(|v| v.into_iter())))
                            .filter_map(|(id, (b, r))| if b { Some((id as usize, r as f64)) } else { None })
                            .collect::<Vec<(usize, f64)>>();

                        info!("num good motifs: {} good_motif_ids_rates={good_motif_ids_rates:?}", good_motifs.sum(Kind::Int));
                        a.push(store.lock().save_selected_motifs(run_config.id.expect("no cfg id"), good_motif_ids_rates)?);
                    }
                }
            }
        } else {
            warn!("no label motif matches for cfg={find_motif_run_cfg:?}");
        }
        Ok::<Vec<usize>, ThoosaError>(a)
    })?;

    Ok(None)

}

fn clc_motif_matches(
    motifs: &Tensor,         // channels x motifs x motif_len
    data: &Tensor,           // channels x num_slices x motif_len
    motif_mask: &Tensor,     // channels x motifs 1/0
    mean_distances: &Tensor, // channels x motifs
) -> Tensor {
    debug!("clc_motif_matches:  motifs.size()={:?}, data.size()={:?}, motif_mask.size()={:?}, mean_distances.size()={:?}", motifs.size(), data.size(), motif_mask.size(), mean_distances.size());

    let dists = Tensor::cdist(motifs, data, 2.0, 0);
    debug!("clc_motif_matches: dists={dists}");

    let motifs_expanded = motifs.unsqueeze(2);     // channels x motifs x 1 x motif_len
    let data_expanded = data.unsqueeze(1);         // channels x 1 x num_slices x motif_len

    debug!("clc_motif_matches: motifs_expanded.size={:?}", motifs_expanded.size());
    debug!("clc_motif_matches: data_expanded.size={:?}", data_expanded.size());

    // Compute the Euclidean distances
    let diff = motifs_expanded - data_expanded;    // [channels x motifs x num_slices x motif_len]
    let squared_diff = diff.pow_tensor_scalar(2);
    let squared_distances = squared_diff.sum_dim_intlist(3, false, Kind::Float);
    let distances = squared_distances.sqrt();      // [channels x motifs x num_slices]

    debug!("clc_motif_matches: distances.size={:?}", distances.size());
    debug!("clc_motif_matches: distances={distances}");

    // Compare distances to mean distances
    let thresholds = mean_distances.unsqueeze(2);  // [channels x motifs x 1]
    let matches = distances.le_tensor(&thresholds); // [channels x motifs x num_slices], bool tensor

    // Count the number of matches
    let num_matches = matches.sum_dim_intlist(2, false, Kind::Int64); // [channels x motifs]

    // Apply the motif mask to zero out counts for padded motifs
    let num_matches = num_matches * motif_mask.to_kind(Kind::Int64);

    num_matches
}

pub fn select_motifs_by_ratio (motifs: Tensor, labels_resources: Vec<(String, Vec <Resource>)>, executor: &Arc<RwLock< LoadRunner<Box<Tensor>>>>, step: usize, device: Device, min_ratio: f32, motifs_distances: Arc<RwLock<HashMap<usize, f32>>>, channel_motif_ids: Arc<RwLock<Vec<Vec<usize>>>>) -> Result<HashMap<String, Vec<Vec<(i64, f64)>>>, ThoosaError> {

    let mut labels_channels_good_motifs = labels_resources.iter().fold(HashMap::new(), |mut a, v| {
        a.insert(v.0.clone(), (0..motifs.size()[0]).map(|_| vec![]).collect::<Vec<Vec<(i64, f64)>>>()); //motifs.size()[0] - channels_num
        a
    });

    let (channels_num, motifs_num, channels_motifs_counts_labels_arc) = clc_channels_motifs_labels(
        Arc::new(RwLock::new(SyncWrapper::new_tb(Box::new(motifs)))),
        labels_resources,
        executor,
        step,
        &device,
        motifs_distances,
        channel_motif_ids
    );

    let channels_motifs_counts_labels_guard = channels_motifs_counts_labels_arc.lock();

    debug!("select_motifs_by_ratio: channels_motifs_counts_labels_guard={:?}", channels_motifs_counts_labels_guard);

    for motif_i in 0..motifs_num {
        for channel_i in 0..channels_num {
            debug!("select_motifs_by_ratio: channels_motifs_classes_guard[channel_i={}][motif_i={}]={:?}", channel_i, motif_i, channels_motifs_counts_labels_guard[channel_i as usize][motif_i as usize]);

            let counts_labels = &channels_motifs_counts_labels_guard[channel_i as usize][motif_i as usize];
            //TODO move into clc_channels_motifs_labels
            let labels_counts: HashMap<String, usize> = counts_labels.0.iter().zip(counts_labels.1.iter()).fold(HashMap::new(), |mut a, v| {
                *a.entry(v.1.clone()).or_insert_with(|| 0) += *v.0 as usize;
                a
            });

            debug!("select_motifs_by_ratio: labels_counts={:?}", labels_counts);
            let sum: f32 = labels_counts.values().map(|v| *v as f32).sum::<f32>();
            if sum > 0. {
                for (label, counts) in labels_counts {
                    let ratio = counts as f32 / sum;
                    if ratio > min_ratio {
                        labels_channels_good_motifs.get_mut(&label).unwrap()[channel_i as usize].push((motif_i, ratio as f64));
                    }
                }
            }

            debug!("select_motifs_by_ratio: labels_channels_good_motifs={:?}", labels_channels_good_motifs);
        }
    }

    // let Some(max_motifs) = channels_good_motifs.iter().map(|c| c.len()).max() else {
    //     return Err("No good motifs".into());
    // };

    Ok(labels_channels_good_motifs)
    // Ok(HashMap::new())
}

pub fn select_motifs_by_corr (
    motifs: Tensor,
    labels_loaders: Vec<(i64, Vec <Resource>)>,
    executor: &Arc<RwLock< LoadRunner<Box<Tensor>>>>,
    step: usize,
    device: Device,
    min_corr: f64,
    motifs_distances: Arc<RwLock<HashMap<usize, f32>>>,
    channel_motif_ids: Arc<RwLock<Vec<Vec<usize>>>>
) -> Result<Vec<Vec<(i64, f64)>>, ThoosaError> {

    let (channels_num, motifs_num, channels_motifs_classes_lock) = clc_channels_motifs_labels(
        Arc::new(RwLock::new(SyncWrapper::new(Box::new(motifs), |e| Box::new(e.shallow_clone())))),
        labels_loaders,
        executor,
        step,
        &device,
        motifs_distances,
        channel_motif_ids
    );

    let mut channels_good_motifs = (0..channels_num).map(|_| vec![]).collect::<Vec<Vec<(i64, f64)>>>();

    let channels_motifs_classes_guard = channels_motifs_classes_lock.lock();

    debug!("select_motifs: channels_motifs_classes_guard={:?}", channels_motifs_classes_guard.len());

    for motif_i in 0..motifs_num {
        for channel_i in 0..channels_num {
            debug!("select_motifs: channels_motifs_classes_guard[channel_i={}][motif_i={}]={:?}", channel_i, motif_i, channels_motifs_classes_guard[channel_i as usize][motif_i as usize]);

            let vals_labels_t = Tensor::vstack(&[
                Tensor::from_slice(channels_motifs_classes_guard[channel_i as usize][motif_i as usize].0.as_slice()),
                Tensor::from_slice(channels_motifs_classes_guard[channel_i as usize][motif_i as usize].1.as_slice())
            ]);

            let corr = vals_labels_t.corrcoef().double_value(&[0, 1]);

            if corr > min_corr {
                debug!("select_motifs: +channel_i={}, motif_i={}, v={:.5}", channel_i, motif_i, corr);

                channels_good_motifs[channel_i as usize].push((motif_i, corr));
            } else {
                debug!("select_motifs: -channel_i={}, motif_i={}, v={:.5}", channel_i, motif_i, corr);
            }
        }
    }

    let Some(_max_motifs) = channels_good_motifs.iter().map(|c| c.len()).max() else {
        return Err("No good motifs".into());
    };

    Ok(channels_good_motifs)
}

fn clc_channels_motifs_labels<L>(
    motifs: Arc<RwLock<SyncWrapper<Box<Tensor>>>>,
    labels_resources: Vec<(L, Vec <Resource>)>,
    executor: &Arc<RwLock< LoadRunner<Box<Tensor>>>>,
    step: usize,
    device: &Device,
    motifs_distances: Arc<RwLock<HashMap<usize, f32>>>,
    channel_motif_ids: Arc<RwLock<Vec<Vec<usize>>>>
) -> (i64, i64, Arc<Mutex<Vec<Vec<(Vec<f64>, Vec<L>)>>>>)
where
    L: Clone + Send + Sync + Display + Debug + 'static,
{
    let motifs_size = motifs.read().size();

    debug!("clc_channels_motifs_labels: motifs_size={motifs_size:?}");
    trace!("clc_channels_motifs_labels: motifs={}", motifs.read());

    let channels_num = motifs_size[0];
    let motifs_num = motifs_size[1];
    let slice_len = motifs_size[2];

    let mut channels_motifs_counts_labels: Vec<Vec<(Vec<f64>, Vec<L>)>> = Vec::with_capacity(channels_num as usize); // (Vec<f64>, Vec<f64>) - conv. value, class label

    for _i in 0..channels_num {
        let mut motifs_classes = Vec::with_capacity(motifs_num as usize);
        for _i in 0..motifs_num {
            motifs_classes.push((vec![], vec![]));//TODO with_capacity?
        }
        channels_motifs_counts_labels.push(motifs_classes);
    }

    let channels_motifs_matches_labels_arc = Arc::new(Mutex::new(channels_motifs_counts_labels));


    labels_resources.into_iter().par_bridge().try_for_each(|(label, resources)| {

        clc_motifs_matches_for_label(
            Arc::clone(&motifs),
            SyncWrapper::new_tb(Box::new(Tensor::zeros(slice_len, (tch::Kind::Float, *device)))),
            Arc::clone(&channels_motifs_matches_labels_arc), //output,
            resources,
            Arc::clone(&executor),
            label,
            step, //TODO step - k * slice_len ?
            Arc::clone(&motifs_distances),
            Arc::clone(&channel_motif_ids)
        )
    }).unwrap();

    (channels_num, motifs_num, channels_motifs_matches_labels_arc)
}


//TODO for tests only?
pub fn iter_all_in_dir<P: AsRef<std::path::Path>>(path: P) -> impl Iterator<Item=Box<Tensor>> {
    fs::read_dir(path.as_ref()).map(|r|
        r.map(|e_r|

            e_r.and_then(|e| {
                debug!("select_motifs: load_all_in_dir");
                Ok(Box::new(Tensor::load(e.path()).expect(format!("can't load {:?}", e.path()).as_str())))
            }).expect("")
        )
    ).expect("iter_all_in_dir")
}

#[cfg(test)]
mod motif_find_tests {
    use std::{fs, mem, thread};

    use crossbeam_channel::internal::SelectHandle;
    use crossbeam_utils::sync::WaitGroup;
    use parking_lot::{Mutex, RawMutex};
    use std::collections::{HashMap, HashSet};
    use std::fmt::{Debug, Display};
    use std::ops::{Add, AddAssign, Deref};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::atomic::{AtomicU32, AtomicU64};
    use std::sync::Arc;


    use log::LevelFilter::Trace;
    use log::{debug, trace};
    use parking_lot::RwLock;
    use parking_lot_mpsc::SendError;
    use plotters::prelude::full_palette::ORANGE_800;
    use plotters::prelude::{Color, RGBAColor, ShapeStyle};
    use plotters::style::full_palette::{GREY_A100, YELLOW_A100};
    use rayon::iter::ParallelBridge;
    use rayon::iter::ParallelIterator;
    use tch::kind::DOUBLE_CPU;
    use tch::Device::Cpu;
    use tch::Kind::{Bool, Float, Int};
    use tch::{Device, IndexOp, Kind, Tensor};
    use thoosa_common::utils::*;
    use thoosa_shared::*;

    use crate::chunks::ChunkOfSlicedTensors;
    use crate::distance::clc_dist_d2;
    use crate::find::select_motifs_by_corr;
    use crate::find::{clc_motif_matches, collect_dists, find_motifs_by_euc_dist, plot_motif_slices, select_motifs_by_label_variance, select_motifs_by_ratio, select_motifs_for_label, CommonFindMotifsConfig, Conv1Params, FindMotifsParams, FindMotifsRunConfig, InputFilter, MotifTrackerStore};
    use crate::fragments::motifs_to_tensor;
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::runs::{Params, RunConfig, Simple};
    use thoosa_common::ws::{DummyWorkspace, FsWorkspace};
    use thoosa_shared::colors_iter::DistinctColorIter;
    use thoosa_shared::resource::{Resource, ResourceId, ResourceStats};
    use thoosa_shared::val::Val;
    use thoosa_shared::val::Val::UInt;
    use thoosa_store::{IdSlice, SlicesMatch, SqlStore, Store};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }


    #[test]
    fn test_z_norm() {
        assert_eq!(vec![-1.2247449, 0.0, 1.2247449], vec_f32_from(&z_norm(&Tensor::from_slice(&[1., 2., 3.]), 0i64)));
    }

    #[test]
    fn test_norm() -> Result<(), ThoosaError> {
        let t = Tensor::from_slice(&[1., 2., 3.]);
        let norm = norm(&t, &[-1]);

        assert_eq!(1.0, vec_f32_from(&norm.dot(&norm))[0]);


        let t = Tensor::vstack(&[
            Tensor::from_slice(&[1., 2., 3.]),
            Tensor::from_slice(&[1., 2., 3.])
        ]);

        let norm = norm_2(&t, 1);

        assert_eq!(Tensor::from_slice(&[1., 1.]), norm.tensordot(&norm, 1, 1).diag(0));

        Ok(())
    }


    #[test]
    fn test_self_dot() -> Result<(), ThoosaError> {
        let t1 = Tensor::from_slice2(&[
            &[1., 2., 3., 4., 5., 6., 7., 8., 9.],
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
    fn test_self_dot_norm() -> Result<(), ThoosaError> {
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


    pub struct TensorsSource {
        tensors: Vec<Tensor>,
    }

    impl TensorsSource {
        pub fn new(tensors: Vec<Tensor>) -> Self {
            Self { tensors }
        }

        fn get_slices(&self, i_0: i64, i_1: i64, m: i64, dim: i64) -> Result<(Tensor, Tensor), ThoosaError> {
            Ok((
                self.tensors[0].f_slice(dim, Some(i_0), Some(i_0 + m), 1)?,
                self.tensors[0].f_slice(dim, Some(i_1), Some(i_1 + m), 1)?
            ))
        }
    }

    #[test]
    fn test_select_motifs() -> Result<(), ThoosaError> { //TODO dummy tensors

        let test_dir = Path::new("/tmp").join(cdt_for_dir());

        fs::create_dir_all(test_dir.clone())?;

        //as built in motifs_to_tensor - channels x  motif  x slice
        //for label xxx:
        let motifs_tensor = //Tensor::vstack(&[
            Tensor::vstack(&[ //channel 0
                Tensor::vstack(&[ //
                    Tensor::from_slice(&[1., 0., 1.]), //motif 0 first slice - good motif, matches tensors in "0" and does not match tensors in 1
                    Tensor::from_slice(&[1., 2., 3.]) //motif 1 first slice - bad motif, matches tensors in "1" and does not match tensors in 0
                ]).view((2, 3)),
                Tensor::vstack(&[ //channel 1
                    Tensor::from_slice(&[4., 5., 6.]), //motif 0 first slice - bad motif,
                    Tensor::from_slice(&[2., 2., 0.]) //motif 1 first slice - good motif,
                ]).view((2, 3)),
            ]).view((2, 2, 3));


        // tensors with matches
        let t_0_0 = Tensor::vstack(&[
            Tensor::from_slice(&[0., 0., 1., 0., 1., 0.]), //channel 0
            Tensor::from_slice(&[1., 1., 0., 1., 0., 0.]) //channel 1
        ]).view((2, 6));

        t_0_0.save(test_dir.join("t_0_0.pt"))?;

        let t_0_1 = Tensor::vstack(&[
            Tensor::from_slice(&[0., 0., 1., 0., 1., 0.]), //channel 0
            Tensor::from_slice(&[0., 2., 2., 0., 0., 0.]) //channel 1
        ]).view((2, 6));

        t_0_1.save(test_dir.join("t_0_1.pt"))?; //XXX


        // tensors without matches
        let t_1_0 = Tensor::vstack(&[
            Tensor::from_slice(&[1., 2., 3., 4., 5., 6.]), //channel 0
            Tensor::from_slice(&[1., 2., 3., 4., 5., 6.]) //channel 1
        ]).view((2, 6));

        t_1_0.save(test_dir.join("t_1_0.pt"))?;

        let t_1_1 = Tensor::vstack(&[
            Tensor::from_slice(&[1., 2., 3., 4., 5., 6.]), //channel 0
            Tensor::from_slice(&[1., 2., 3., 4., 5., 6.]) //channel 1
        ]).view((2, 6));

        t_1_1.save(test_dir.join("t_1_1.pt"))?; //YYY


        let t_2_0 = Tensor::vstack(&[
            Tensor::from_slice(&[2., 2., 2., 2., 2., 2.]), //channel 0
            Tensor::from_slice(&[0., 2., 2., 1., 1., 1.]) //channel 1
        ]).view((2, 6));

        t_2_0.save(test_dir.join("t_2_0.pt"))?;
        let t_2_1 = Tensor::vstack(&[
            Tensor::from_slice(&[2., 2., 2., 2., 2., 2.]), //channel 0
            Tensor::from_slice(&[1., 2., 2., 1., 0., 0.]) //channel 1
        ]).view((2, 6));

        t_2_1.save(test_dir.join("t_2_1.pt"))?; //ZZZ

        let channel_motif_ids = vec![
            vec![5, 7],
            vec![8, 9],
        ];


        let motifs_avg_distances = HashMap::from([
            (5, 0.5),
            (7, 0.5),
            (8, 0.5),
            (9, 0.5)
        ]);

        let executor_rc = Arc::new(parking_lot::RwLock::new(LoadRunner::new(
            move |key: &Resource| {
                debug!("LoadRunner: test_dir={test_dir:?} key={key:?}");
                let tensor = Tensor::load(test_dir.join(key.file_stem())).expect(format!("can't load {:?}", key).as_str());
                debug!("LoadRunner: tensor={tensor}");

                SyncWrapper::new_tb(Box::new(tensor))
            }, 4, 100,
        )));
        let motifs_avg_distances_arc = Arc::new(RwLock::new(motifs_avg_distances));
        let channel_motif_ids_arc = Arc::new(RwLock::new(channel_motif_ids));

        let channels_good_motifs = select_motifs_by_corr(
            motifs_tensor.shallow_clone(),
            vec![
                (1, vec![(1,1,"t_0_0.pt",6).into(), (2,1,"t_0_1.pt",6).into()]),
                (0, vec![(3,1,"t_1_0.pt",6).into(), (4,1,"t_1_1.pt",6).into()]),
            ],
            &executor_rc,
            1,
            Device::Cpu,
            0.1,
            Arc::clone(&motifs_avg_distances_arc),
            Arc::clone(&channel_motif_ids_arc),
        )?;

        assert_eq!(0, channels_good_motifs[0][0].0); //motif 0 is good for channel 0
        assert_eq!(1, channels_good_motifs[1][0].0); //motif 1 is good for channel 1

        let labels_channels_good_motifs = select_motifs_by_ratio(
            motifs_tensor,
            vec![
                ("xxx".into(), vec![(1,1, "t_0_0.pt",6).into(), (2,1, "t_0_1.pt",6).into()]),
                ("yyy".into(), vec![(3,1, "t_1_0.pt",6).into(), (4,1, "t_1_1.pt",6).into()]),
                ("zzz".into(), vec![(5,1, "t_2_0.pt",6).into(), (6,1, "t_2_1.pt",6).into()]),
            ],
            &executor_rc,
            1,
            Device::Cpu,
            0.5,
            Arc::clone(&motifs_avg_distances_arc),
            Arc::clone(&channel_motif_ids_arc),
        )?;


        debug!("2. t={:?}", labels_channels_good_motifs);
        let channels_good_motifs = labels_channels_good_motifs.get(&String::from("xxx")).unwrap();
        assert_eq!(0, channels_good_motifs[0][0].0); //motif 0 is good for channel 0
        assert_eq!(1, channels_good_motifs[1][0].0); //motif 1 is good for channel 1


        Ok(())
    }

    #[test]
    fn test_find_motifs_by_euc_dist() -> Result<(), ThoosaError> {
        let eeg_id_series:Vec<String> = vec!["1111".into(), "2222".into(), "3333".into()];

        // std::fs::remove_file("/tmp/test_db.db3").unwrap_or(());
        // let mut store = Arc::new(Mutex::new(SqlStore::from_path("/tmp/test_db.db3")?));
        // let store = Arc::new(Mutex::new(SqlStore::in_mem()?));

        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        fs::create_dir_all(&test_dir)?;
        let mut store = Arc::new(Mutex::new(SqlStore::from_path(test_dir.join("test_find_motifs_by_euc_dist.db3"))?));
        
        
        let ws = Arc::new(RwLock::new(DummyWorkspace::new(1, 1000)?));

        let pulse = pulse_v(50., 10., 100, 20.);
        let n = 1000i64;

        let sig_0 = join_with_offset::<f32>(0., 10.0, &[(100, pulse.clone())], random_saw(1000));
        let sig_1 = join_with_offset::<f32>(0., 10.0, &[(500, pulse.clone())], random_saw(1000));
        let sig_2 = join_with_offset::<f32>(0., 10.0, &[(800, pulse.clone())], random_saw(1000));

        /*
        let x = lin_t(n);
        plot_series(
             vec![
                 (format!("Sig0"), plotters::prelude::RED, vec_f32_from(&x).into_iter().zip(sig_0.clone()).collect::<Vec<(f32, f32)>>()),
                 (format!("Sig1"), plotters::prelude::GREEN, vec_f32_from(&x).into_iter().zip(sig_1.clone()).collect::<Vec<(f32, f32)>>()),
                 (format!("Sig2"), plotters::prelude::BLACK, vec_f32_from(&x).into_iter().zip(sig_2.clone()).collect::<Vec<(f32, f32)>>()),

             ], "test_find_motifs_by_euc_dist", (1000, 500), &format!("/tmp/plots/0_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str()
        ).unwrap();
        */


        let  executor_rc = Arc::new(RwLock::new(LoadRunner::new(
            move |key: &Resource| {
                let sig = match key.file_stem() {
                    "1111" => &sig_0,
                    "2222" => &sig_1,
                    "3333" => &sig_2,
                    _ => panic!("no")
                };

                SyncWrapper::new_tb(
                    Box::new(Tensor::vstack(&[
                        Tensor::from_slice(
                            sig.as_slice()
                        ), //ch 0
                    ]))
                )
            }, 4, 100,
        )));

        let resources = test_init_store(&store, &["label_0", "label_1"], HashMap::from([
            ("1111".to_string(), (ResourceStats::new(vec![1,n as u32]), vec![("label_0".to_string(), 1.0)])),
            ("2222".to_string(), (ResourceStats::new(vec![1,n as u32]), vec![("label_0".to_string(), 1.0)])),
            ("3333".to_string(), (ResourceStats::new(vec![1,n as u32]), vec![("label_0".to_string(), 1.0)])),
        ]))?;


        let params = FindMotifsParams {
            step: 100.into(),
            window: 100.into(),
            resources: ResourceType::List(eeg_id_series),
            min_tracker_step: Some(3),
            chunk_size: 2,
            dist_quantile: 0.01,
            keep_slice_matches: false,
            max_slices_k: None,
            mean_k: None,
            input_filter: None,
        };

        let (common_config, run_configs) = to_test_run_config(&store, &params, false);
        let run_ids = run_configs.iter().map(|run_config|run_config.id.expect("id")).collect::<Vec<_>>();
        find_motifs_by_euc_dist(
            None,
            run_configs,
            &Simple::new(params.resources),
            &common_config,
            &ws,
            &store,
            &executor_rc,
            vec![Cpu]
        )?;

        // debug!("{:?}", store.lock().list_run_resources(1)?);
        let slices = store.lock().list_motifs_slices(run_ids[0])?;

        debug!("run_ids={run_ids:?} slices: {slices:?}");
        debug!("0: {:?}", slices.get(&resources[0]));
        debug!("1: {:?}", slices.get(&resources[1]));
        debug!("2: {:?}", slices.get(&resources[2]));

        assert!(slices.get(&resources[0]).and_then(|v| v.iter().find(|s|s.slice_idx == 100)).is_some());
        assert!(slices.get(&resources[1]).and_then(|v| v.iter().find(|s|s.slice_idx == 500)).is_some());
        assert!(slices.get(&resources[2]).and_then(|v| v.iter().find(|s|s.slice_idx == 800)).is_some());

        Ok(())
    }


    #[test]
    fn test_find_motifs_by_euc_dist_two_motifs() -> Result<(), ThoosaError> {
        let eeg_id_series:Vec<String> = vec!["1111".into(), "2222".into(), "3333".into()];

        // std::fs::remove_file("/tmp/test_db.db3").unwrap_or(());
        // let mut store = Arc::new(Mutex::new(SqlStore::from_path("/tmp/test_db.db3")?));
        // let store = Arc::new(Mutex::new(SqlStore::in_mem()?));

        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        fs::create_dir_all(&test_dir)?;



        // let ws = Arc::new(RwLock::new(DummyWorkspace::new(1, 1000)?));
        let ws = Arc::new(RwLock::new(FsWorkspace::new(test_dir, 1, 1000)?));
        let mut store = Arc::new(Mutex::new(SqlStore::from_path(ws.read().store_path())?));

        // let pulse = pulse_v(50., 10., 100, 20.);
        let pulse_0 = sig_one_imp(100, 30.0, None, Some(2. ..2.01), 0.0);
        let pulse_1 = sig_one_imp(100, 30.0, None, Some(20. ..20.01), 0.0);
        let n = 1000i64;

        let scale = 10.0;
        // dists between sample pulses and joined are too big
        // let sig_0 = join_with_offset::<f32>(0., scale, &[(100, pulse_0.clone()), (100, pulse_1.clone())], saw_with_steps(n as usize, 10, 250.));
        // let sig_1 = join_with_offset::<f32>(0., scale, &[(500, pulse_0.clone()), (100, pulse_1.clone())], saw_with_steps(n as usize, 10, 250.));
        // let sig_2 = join_with_offset::<f32>(0., scale, &[(200, pulse_1.clone()), (500, pulse_0.clone())], saw_with_steps(n as usize, 10, 250.));
        let sig_0 = join_with_offset::<f32>(0., scale, &[(100, pulse_0.clone()), (100, pulse_1.clone())], random_saw(1000));
        let sig_1 = join_with_offset::<f32>(0., scale, &[(500, pulse_0.clone()), (100, pulse_1.clone())], random_saw(1000));
        let sig_2 = join_with_offset::<f32>(0., scale, &[(200, pulse_1.clone()), (500, pulse_0.clone())], random_saw(1000));



        let x = lin_t(n);
        plot_series(
             vec![
                 (format!("Sig0"), plotters::prelude::RED, vec_f32_from(&x).into_iter().zip(sig_0.clone()).collect::<Vec<(f32, f32)>>()),
                 (format!("Sig1"), plotters::prelude::GREEN, vec_f32_from(&x).into_iter().zip(sig_1.clone()).collect::<Vec<(f32, f32)>>()),
                 (format!("Sig2"), plotters::prelude::BLACK, vec_f32_from(&x).into_iter().zip(sig_2.clone()).collect::<Vec<(f32, f32)>>()),

             ], "test_find_motifs_by_euc_dist", (1000, 500), &ws.read().plots_path().join(format!("0_{:?}.png", chrono::Utc::now().to_rfc3339()))
        ).unwrap();


        let loader = Arc::new(RwLock::new(LoadRunner::new(
            move |key: &Resource| {
                let sig = match key.file_stem() {
                    "1111" => &sig_0,
                    "2222" => &sig_1,
                    "3333" => &sig_2,
                    _ => panic!("no")
                };

                SyncWrapper::new_tb(
                    Box::new(Tensor::vstack(&[
                        Tensor::from_slice(
                            sig.as_slice()
                        ), //ch 0
                    ]))
                )
            }, 4, 100,
        )));

        let resources = test_init_store(&store, &["label_0", "label_1"], HashMap::from([
            ("1111".to_string(), (ResourceStats::new(vec![1,n as u32]), vec![("label_0".to_string(), 1.0)])),
            ("2222".to_string(), (ResourceStats::new(vec![1,n as u32]), vec![("label_0".to_string(), 1.0)])),
            ("3333".to_string(), (ResourceStats::new(vec![1,n as u32]), vec![("label_0".to_string(), 1.0)])),
        ]))?;


        let params = FindMotifsParams {
            step: 100.into(),
            window: 100.into(),
            resources: ResourceType::List(eeg_id_series),
            min_tracker_step: Some(3),
            chunk_size: 2,
            dist_quantile: 0.1,
            keep_slice_matches: false,
            max_slices_k: None,
            mean_k: None,
            input_filter: None,
        };

        let (common_config, run_configs) = to_test_run_config(&store, &params, false);
        let run_ids = run_configs.iter().map(|run_config|run_config.id.expect("id")).collect::<Vec<_>>();


        find_motifs_by_euc_dist(
            None,
            run_configs,
            &Simple::new(params.resources),
            &common_config,
            &ws,
            &store,
            &loader,
            vec![Cpu]
        )?;

        let slices = store.lock().list_motifs_slices(run_ids[0])?;

        let motif_id_set = slices
            .iter()
            .flat_map(|(resource, motifs)| motifs.into_iter().map(|m| m.motif_id).collect::<HashSet<_>>())
            .collect::<HashSet<_>>();

        for motif_id in motif_id_set {
            plot_motif_slices(&ws, &store, &loader, motif_id, &Cpu)?;
        }
        debug!("run_ids={run_ids:?} slices: {slices:?}");
        debug!("0: {:?}", slices.get(&resources[0]));
        debug!("1: {:?}", slices.get(&resources[1]));
        debug!("2: {:?}", slices.get(&resources[2]));


        let motifs_tensors = motifs_to_tensor(
            slices.clone(),
            &loader,
            100,
            ws.read().channels() as i64,
            Cpu,
            0,
        )?;
        let pulse_0_t = Tensor::from_slice(pulse_0.as_slice()).multiply_scalar(scale as f64).view((1, -1));
        let pulse_1_t = Tensor::from_slice(pulse_1.as_slice()).multiply_scalar(scale as f64).view((1, -1));

        let mut mean_motifs = vec![];
        let mut colors_iter = DistinctColorIter::new();

        let mut pulse_0_found = None::<i64>;
        let mut pulse_1_found = None::<i64>;
        let zero = Tensor::from_slice(&[0.]).totype(Float);

        for (motif_i, (motif_id, avg_dists)) in motifs_tensors.channels_motifs_ids.i((0, ..)).iter::<i64>()?.zip(motifs_tensors.channels_motifs_avg_dists.i((0, ..)).iter::<f64>()?).enumerate() {
            let motif_t = motifs_tensors.channels_motifs_slices.i((0, motif_i as i64, ..)).view((1, -1));

            debug!("Size of motif_t tensor: {:?} pulse_0_t.size()={:?}", motif_t.size(), pulse_0_t.size());


            let dist_pulse_0 = clc_dist_d2(motif_t.shallow_clone(), pulse_0_t.shallow_clone(), 0)?;

            if dist_pulse_0.allclose(&zero, 0.1, 0.2, false) {
                pulse_0_found.replace(motif_id);
            }

            let dist_pulse_1 = clc_dist_d2(motif_t.shallow_clone(), pulse_1_t.shallow_clone(), 0)?;
            if dist_pulse_1.allclose(&zero, 0.1, 0.2, false) {
                pulse_1_found.replace(motif_id);
            }

            debug!("motif_id={motif_id:?}, avg_dists={avg_dists:?} dist_pulse_0={dist_pulse_0} dist_pulse_1={dist_pulse_1}");

            let rgb = colors_iter.next().unwrap().rgb();
            mean_motifs.push((format!("Motif {motif_i}"), ShapeStyle {
                color: RGBAColor(rgb.0, rgb.1, rgb.2, 1.0),
                filled: false,
                stroke_width: 1,
            }, vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&motif_t.i((0, ..)))).collect::<Vec<(f32, f32)>>()),)
        }


        let rgb = YELLOW_A100; //colors_iter.next().unwrap().rgb();
        mean_motifs.push(("pulse 0".to_string(), ShapeStyle {
            color: RGBAColor(rgb.0, rgb.1, rgb.2, 0.5),
            filled: true,
            stroke_width: 5,
        }, vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&pulse_0_t.i((0, ..)))).collect::<Vec<(f32, f32)>>()));

        let rgb = GREY_A100; //colors_iter.next().unwrap().rgb();
        mean_motifs.push(("pulse 1".to_string(), ShapeStyle {
            color: RGBAColor(rgb.0, rgb.1, rgb.2, 0.5),
            filled: true,
            stroke_width: 5,
        }, vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&pulse_1_t.i((0, ..)))).collect::<Vec<(f32, f32)>>()));

        plot_series(
            mean_motifs, "test_find_motifs_by_euc_dist", (1000, 500), &ws.read().plots_path().join(format!("motifs_{:?}.png", chrono::Utc::now().to_rfc3339()))
        ).unwrap();

        let pulse_0_motif = pulse_0_found.expect("pulse_0 not found");
        let pulse_1_motif = pulse_1_found.expect("pulse_1 not found");


        let mut pulse_0_slices: Vec<_> = slices
            .values()
            .flat_map(|slice_list| slice_list.iter())
            .filter(|slice| slice.motif_id == pulse_0_motif as usize)
            .map(|slice| slice.slice_idx)
            .collect();

        let mut pulse_1_slices: Vec<_> = slices
            .values()
            .flat_map(|slice_list| slice_list.iter())
            .filter(|slice| slice.motif_id == pulse_1_motif as usize)
            .map(|slice| slice.slice_idx)
            .collect();

        pulse_0_slices.sort();
        pulse_1_slices.sort();

        debug!("pulse_0_slices={pulse_0_slices:?} pulse_1_slices={pulse_1_slices:?}");

        assert_eq!(pulse_0_slices, vec![100, 500, 800]);
        assert_eq!(pulse_1_slices, vec![200, 300, 700]);


        Ok(())
    }


    fn test_init_store(store: &Arc<Mutex<Box<dyn Store>>>, labels: &[&str], resources: HashMap<String, (ResourceStats, Vec<(String, f32)>)>) -> Result<Vec<Resource>, ThoosaError> {
        {
            let mut store_guard = store.lock();

            let mut conn_guard = store_guard.conn().lock();
            let t = conn_guard.transaction()?;

            for label in labels {
                store_guard.save_label(label, &t)?;
            }

            let import_id = store_guard.save_import(format!("test on {}", cdt_for_dir()), &t)?;

            store_guard.save_resources_labels(&resources, 1, &t)?;
            t.commit().expect("commit");
        }

        store.lock().list_resources_by_file_stems(&vec!["1111", "2222", "3333"])
    }

    fn to_test_run_config(store: &Arc<Mutex<Box<dyn Store>>>, params: &FindMotifsParams, file_motif_tracker: bool) -> (CommonFindMotifsConfig, Vec<FindMotifsRunConfig>) {
        let mut common_config = params.common_config();
        common_config.keep_slice_matches = file_motif_tracker;
        let mut run_configs = params.to_run_configs();
        run_configs.iter_mut().for_each(|c| {
            let id = store.lock().save_run_config(None, &FindMotifsRunConfig::type_name(), "test", "test", c.to_properties_map()).unwrap();
            c.set_id(id);
        });
        (common_config, run_configs)
    }

    #[test]
    fn test_find_motifs_by_euc_dist_with_smooth_kernel() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = Arc::new(RwLock::new(DummyWorkspace::new(1, 1000)?));

        let eeg_id_series:Vec<String> = vec!["1111".into(), "2222".into(), "3333".into()];
        let store = Arc::new(Mutex::new(SqlStore::in_mem()?));

        let pulse = pulse_v(50., 10., 100, 20.);
        let n = 1000i64;

        let sig_0 = join_with_offset::<f32>(0., 10.0, &[(100, pulse.clone())], random_saw(1000));
        let sig_1 = join_with_offset::<f32>(0., 10.0, &[(500, pulse.clone())], random_saw(1000));
        let sig_2 = join_with_offset::<f32>(0., 10.0, &[(800, pulse.clone())], random_saw(1000));

        /*
        let x = lin_t(N);

        plot_series(
             vec![
                 (format!("Sig0"), RED, vec_f32_from(&x).into_iter().zip(sig_0.clone()).collect::<Vec<(f32, f32)>>()),
                 (format!("Sig1"), GREEN, vec_f32_from(&x).into_iter().zip(sig_1.clone()).collect::<Vec<(f32, f32)>>()),
                 (format!("Sig2"), BLACK, vec_f32_from(&x).into_iter().zip(sig_2.clone()).collect::<Vec<(f32, f32)>>()),

             ], "test_find_motifs_by_euc_dist_with_smooth_kernel", (1000, 500), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
         ).unwrap();
         */

        let executor_rc = Arc::new(RwLock::new(LoadRunner::new(
            move |key: &Resource| {
                let sig = match key.file_stem() {
                    "1111" => &sig_0,
                    "2222" => &sig_1,
                    "3333" => &sig_2,
                    _ => panic!("no")
                };

                SyncWrapper::new(
                    Box::new(Tensor::vstack(&[
                        Tensor::from_slice(
                            sig.as_slice()
                        ), //ch 0
                    ])),
                    |e| Box::new(e.shallow_clone())
                )
            }, 4, 100,
        )));


        let resources = test_init_store(&store, &["label_0", "label_1"], HashMap::from([
            ("1111".to_string(), (ResourceStats::new(vec![1,n as u32]), vec![("label_0".to_string(), 1.0)])),
            ("2222".to_string(), (ResourceStats::new(vec![1,n as u32]), vec![("label_0".to_string(), 1.0)])),
            ("3333".to_string(), (ResourceStats::new(vec![1,n as u32]), vec![("label_0".to_string(), 1.0)])),
        ]))?;

        let params = FindMotifsParams {
            step: 100.into(),
            window: 100.into(),
            resources: ResourceType::List(eeg_id_series),
            min_tracker_step: Some(3),
            chunk_size: 2,
            dist_quantile: 0.0001,
            keep_slice_matches: false,
            max_slices_k: None,
            mean_k: None,
            input_filter: Some(InputFilter::Conv1(Conv1Params::new(10, 1, 1))),
        };

        let (common_config, run_configs) = to_test_run_config(&store, &params, false);

        let run_ids = run_configs.iter().map(|run_config|run_config.id.expect("id")).collect::<Vec<_>>();

        find_motifs_by_euc_dist(
            None,
            run_configs,
            &Simple::new(params.resources),
            &common_config,
            &ws,
            &store,
            &executor_rc,
            vec![Cpu]
        )?;


        debug!("run_ids={:?}", run_ids);

        debug!("{:?}", store.lock().list_run_resources(1)?);
        let slices = store.lock().list_motifs_slices(run_ids[0])?;
        debug!("0: {:?}", slices.get(&resources[0]));
        debug!("1: {:?}", slices.get(&resources[1]));
        debug!("2: {:?}", slices.get(&resources[2]));


        assert!(slices.get(&resources[0]).and_then(|v| v.iter().find(|s|s.slice_idx == 100)).is_some());
        assert!(slices.get(&resources[1]).and_then(|v| v.iter().find(|s|s.slice_idx == 500)).is_some());
        assert!(slices.get(&resources[2]).and_then(|v| v.iter().find(|s|s.slice_idx == 800)).is_some());

        Ok(())
    }


    #[test]
    fn test_find_motifs_by_euc_dist_diff_sizes() -> Result<(), ThoosaError> {
        let eeg_id_series:Vec<String> = vec!["0000".into(), "1111".into(), "2222".into()];
        // let store_path = format!("/tmp/test_store_{}.db3", cdt_for_dir());
        // let store = Arc::new(Mutex::new(SqlStore::from_path(store_path)?));
        // let store = Arc::new(Mutex::new(SqlStore::in_mem()?));
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        fs::create_dir_all(&test_dir)?;
        let mut store = Arc::new(Mutex::new(SqlStore::from_path(test_dir.join("test_find_motifs_by_euc_dist_diff_sizes.db3"))?));


        let ws = Arc::new(RwLock::new(DummyWorkspace::new(1, 1000)?));

        let pulse = pulse_v(50., 10., 100, 20.);

        let sig_0 = join_with_offset::<f32>(0., 10.0, &[(100, pulse.clone())], random_saw(550));
        let sig_1 = join_with_offset::<f32>(0., 10.0, &[(600, pulse.clone())], random_saw(1100));
        let sig_2 = join_with_offset::<f32>(0., 10.0, &[(1000, pulse.clone())], random_saw(1950));

        let n = 3000i64;
        /*
        plot_series(
            vec![
                (format!("Sig0"), RED, (0..N).map(|x|x as f32).zip(sig_0.clone()).collect::<Vec<(f32, f32)>>()),
                (format!("Sig1"), GREEN, (0..N).map(|x|x as f32).zip(sig_1.clone()).collect::<Vec<(f32, f32)>>()),
                (format!("Sig2"), BLACK, (0..N).map(|x|x as f32).zip(sig_2.clone()).collect::<Vec<(f32, f32)>>()),

            ], "test_find_motifs_by_euc_dist_diff_sizes", (3000, 500), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        ).unwrap();
        */
        let loader = Arc::new(RwLock::new(LoadRunner::new(
            move |key: &Resource| {
                let sig = match key.file_stem() {
                    "0000" => &sig_0,
                    "1111" => &sig_1,
                    "2222" => &sig_2,
                    _ => panic!("no")
                };

                SyncWrapper::new_tb(
                    Box::new(Tensor::vstack(&[
                        Tensor::from_slice(
                            sig.as_slice()
                        ), //ch 0
                    ]))
                )
            }, 4, 100,
        )));

        let resources = test_init_store(&store, &["label_0", "label_1"], HashMap::from([
            ("1111".to_string(), (ResourceStats::new(vec![1,n as u32]), vec![("label_0".to_string(), 1.0)])),
            ("2222".to_string(), (ResourceStats::new(vec![1,n as u32]), vec![("label_0".to_string(), 1.0)])),
            ("3333".to_string(), (ResourceStats::new(vec![1,n as u32]), vec![("label_0".to_string(), 1.0)])),
        ]))?;

        let params = FindMotifsParams {
            step: 50.into(),
            window: 100.into(),
            resources: ResourceType::List(eeg_id_series),
            min_tracker_step: Some(3),
            chunk_size: 2,
            dist_quantile: 0.0001,
            keep_slice_matches: false,
            max_slices_k: None,
            mean_k: None,
            input_filter: None,
        };


        let (common_config, run_configs) = to_test_run_config(&store, &params, false);

        let run_ids = find_motifs_by_euc_dist(
            None,
            run_configs,
            &Simple::new(params.resources),
            &common_config,
            &ws,
            &store,
            &loader,
            vec![Cpu]
        )?;

        debug!("{:?}", store.lock().list_run_resources(1)?);
        let eeg_slices = store.lock().list_motifs_slices(1)?;

        debug!("eeg_slices={eeg_slices:?}");
        //TODO
        // assert!(eeg_slices.get(&resources[0]).and_then(|v| v.iter().find(|s|s.slise_idx == 100)).is_some());
        assert!(eeg_slices.get(&resources[0]).and_then(|v| v.iter().find(|s|s.slice_idx == 600)).is_some());
        assert!(eeg_slices.get(&resources[1]).and_then(|v| v.iter().find(|s|s.slice_idx == 1000)).is_some());

        Ok(())
    }


    #[test]
    pub fn test_find_motifs_by_euc_dist_wrong_index() -> Result<(), ThoosaError> {
        // let store_path = format!("/tmp/test_store_{}.db3", cdt_for_dir());
        // let store = Arc::new(Mutex::new(SqlStore::from_path(store_path)?));
        let store = Arc::new(Mutex::new(SqlStore::in_mem()?));
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = Arc::new(RwLock::new(DummyWorkspace::new(1, 1000)?));

        let pulse = pulse_v(50., 10., 100, 20.);


        let tensor_sizes = [
            1000, 1000, 1000,
            500, 500, 500,
        ];

        let tensors = tensor_sizes.into_iter()
            .map(|size| join_with_offset::<f32>(0., 10.0, &[(100, pulse.clone())], random_saw(size as usize)))
            .collect::<Vec<Vec<f32>>>();

        let n = 684400i64;
        let tensors_len = tensors.len();

        let loader = Arc::new(RwLock::new(LoadRunner::new(
            move |key: &Resource| {
                let sig = tensors.get(
                    key.file_stem().parse::<usize>().expect(format!("parse {}", key.file_stem()).as_str())
                ).expect(format!("tensor {key:?}", ).as_str());

                SyncWrapper::new_tb(
                    Box::new(Tensor::vstack(&[
                        Tensor::from_slice(
                            sig.as_slice()
                        ), //ch 0
                    ]))
                )
            }, 4, 100,
        )));

        let resources = test_init_store(&store, &["label_0", "label_1"], (0..tensors_len).fold(HashMap::new(), |mut a, v |{
            a.insert(format!("{}", v), (ResourceStats::new(vec![1,n as u32]), vec![]));
            a
        }))?;

        let params = FindMotifsParams {
            step: 100.into(),
            window: 100.into(),
            resources: ResourceType::List((0..tensors_len).map(|v| format!("{}", v)).collect()),
            min_tracker_step: None,
            chunk_size: 3,
            dist_quantile: 0.01,
            keep_slice_matches: false,
            max_slices_k: None,
            mean_k: None,
            input_filter: None,
        };

        let (common_config, run_configs) = to_test_run_config(&store, &params, false);

        let run_ids = find_motifs_by_euc_dist(
            None,
            run_configs,
            &Simple::new(params.resources),
            &common_config,
            &ws,
            &store,
            &loader,
            vec![Cpu]
        )?;

        debug!("{:?}", store.lock().list_run_resources(1)?);
        let eeg_slices = store.lock().list_motifs_slices(1)?;
        debug!("eeg_slices:{:?}", eeg_slices);

        assert_eq!(6, eeg_slices.len());

        Ok(())
    }

    #[derive(Debug)]
    struct DummyDumper<'a> {
        motifs:  &'a mut Vec<SlicesMatch>
    }

    impl <'a> DummyDumper<'a> {
        pub fn new(motifs: &'a mut Vec<SlicesMatch>) -> Self {
            Self {
                motifs
            }
        }
    }

    impl <'a> MotifTrackerStore for DummyDumper<'a> {

        fn save_slices(&mut self, motifs: Vec<SlicesMatch>) -> Result<(), ThoosaError>{

            self.motifs.extend(motifs.into_iter().collect::<Vec<SlicesMatch>>());

            Ok(())
        }

        fn load_slices_into_groups(&mut self, store: &Arc<Mutex<Box<dyn Store>>>, max_slices_counts: Option<u32>) -> Result<(), ThoosaError> {
            todo!()
        }

        fn run_id(&self) -> usize {
            todo!()
        }

        fn channel(&self) -> usize {
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
            let mut mt = super::SimpleMotifTracker::new(Box::new(DummyDumper::new(&mut motifs)), None, Some(100));

            mt.track(TensorIdSliceIdx(0, 10), TensorIdSliceIdx(1, 10), 1.0);
            mt.track(TensorIdSliceIdx(0, 20), TensorIdSliceIdx(1, 20), 1.0);
            mt.track(TensorIdSliceIdx(0, 30), TensorIdSliceIdx(1, 30), 1.0);

            mt.dump();
        }
        assert_eq!(motifs.len(), 3);
        assert_eq!(SlicesMatch(TensorIdSliceIdx(0, 10), TensorIdSliceIdx(1, 10), 1.0), motifs[0]);
        assert_eq!(SlicesMatch(TensorIdSliceIdx(0, 20), TensorIdSliceIdx(1, 20), 1.0), motifs[1]);
        assert_eq!(SlicesMatch(TensorIdSliceIdx(0, 30), TensorIdSliceIdx(1, 30), 1.0), motifs[2]);

        let mut motifs = vec![];
        {
            let mut mt = super::SimpleMotifTracker::new(Box::new(DummyDumper::new(&mut motifs)), Some(10), Some(100));

            mt.track(TensorIdSliceIdx(0, 10), TensorIdSliceIdx(1, 10), 1.0);
            mt.track(TensorIdSliceIdx(0, 20), TensorIdSliceIdx(1, 20), 1.0);
            mt.track(TensorIdSliceIdx(0, 31), TensorIdSliceIdx(1, 30), 1.0);

            mt.dump();
        }


        assert_eq!(motifs.len(), 2);
        assert_eq!(SlicesMatch(TensorIdSliceIdx(0, 20), TensorIdSliceIdx(1, 20), 1.0), motifs[0]);
        assert_eq!(SlicesMatch(TensorIdSliceIdx(0, 31), TensorIdSliceIdx(1, 30), 1.0), motifs[1]);

        Ok(())
    }

    #[test]
    pub fn test_simple_motif_tracker_with_tensorssource() -> Result<(), ThoosaError> {
        let n = 1_000i64;


        let (sig_0_1, sig_0_2) = two_sigs(1_000);
        debug!("sig_1.size()={:?}, sig_2.size()={:?}", sig_0_1.size(), sig_0_2.size());
        let _sig_0_c = sig_0_1.copy();

        let t_0 = Tensor::vstack(&[sig_0_1, sig_0_2]);
        debug!("t.size={:?}", t_0.size());

        let (sig_1_1, sig_1_2) = two_sigs(1_000);
        let _sig_1_c = sig_1_1.copy();
        let t_1 = Tensor::vstack(&[sig_1_1, sig_1_2]);

        let m = 100;
        let step = 100;

        let mut motifs = vec![];
        let mut mt = super::SimpleMotifTracker::new(Box::new(DummyDumper::new(&mut motifs)), None, Some(100));

        let dim = 1;


        let ts = TensorsSource::new(vec![t_0, t_1]);
        for i_0 in (0..n - m).step_by(step) {
            for i_1 in (i_0..n - m).step_by(step).skip(1) {
                debug!("i_0={:?}, i_1={:?} m={}", i_0, i_1, m);

                let (slice_01, slice_02) = ts.get_slices(i_0, i_1, m, dim)?;

                let dp = dot_norm(dim, &slice_01, &slice_02)?;

                mt.track(TensorIdSliceIdx(0, i_0 as u32), TensorIdSliceIdx(0, i_1 as u32), dp.double_value(&[0]) as f32);
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

        assert_eq!((TensorIdSliceIdx(0, 100), TensorIdSliceIdx(0, 600)), (motifs[0].0.clone(), motifs[0].1.clone()));
        assert_eq!((TensorIdSliceIdx(0, 200), TensorIdSliceIdx(0, 700)), (motifs[1].0.clone(), motifs[1].1.clone()));


        Ok(())
    }

    #[test]
    fn test_conv() {
        let t_0 = Tensor::vstack(&[
            Tensor::from_slice(&[1., 1., 1., 0., 0., 0.]),
            Tensor::from_slice(&[2., 2., 2., 0., 0., 0.])
        ]); //view((2, 1, 6))
        let t_1 = Tensor::vstack(&[
            Tensor::from_slice(&[1., 1., 1.]),
            Tensor::from_slice(&[2., 2., 2.])
        ]).view((2, 1, 3)); //view((2, 1, 3))


        let c = t_0.conv1d(&t_1, None::<Tensor>, 1, 3, 1, 2);

        debug!("t_0={:?} t_1={:?} c={:?}", t_0.size(), t_1.size(), c.size());


        assert_eq!(
            c,
            Tensor::vstack(&[
                Tensor::from_slice(&[0., 1., 2., 3., 2., 1., 0., 0., 0., 0., ]),
                Tensor::from_slice(&[0., 4., 8., 12., 8., 4., 0., 0., 0., 0., ])
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
            Tensor::from_slice(&[2., 2., 2., 0., 0., 0.]) // second channel
        ]); //view((2, 1, 6))

        let t_1 = Tensor::vstack(&[
            Tensor::vstack(&[
                Tensor::from_slice(&[1., 1., 1.]), // 1 first channel kernel
                Tensor::from_slice(&[3., 3., 3.]) // 2 first channel kenrel
            ]),
            Tensor::vstack(&[
                Tensor::from_slice(&[2., 2., 2.]), // 1 second channel kernel
                Tensor::from_slice(&[4., 4., 4.]) // 2 second channel kernel
            ])
        ]).view((4, 1, 3));

        debug!("t_0={:?} t_1={:?}", t_0.size(), t_1.size());

        let c = t_0.conv1d(&t_1, None::<Tensor>, 1, 3, 1, 2);

        debug!("c={:?}", c.size());


        assert_eq!(
            c,
            Tensor::vstack(&[
                Tensor::from_slice(&[0., 1., 2., 3., 2., 1., 0., 0., 0., 0., ]), // first channel * 1 kernel
                Tensor::from_slice(&[0., 3., 6., 9., 6., 3., 0., 0., 0., 0., ]), // first channel * 2 kernel
                Tensor::from_slice(&[0., 4., 8., 12., 8., 4., 0., 0., 0., 0., ]), // second channel * 1 kernel
                Tensor::from_slice(&[0., 8., 16., 24., 16., 8., 0., 0., 0., 0., ])  // second channel * 2 kernel
            ])
        );

        let c_by_channel = c.view((2, -1, 10));

        debug!("c={:?}", c_by_channel.size());


        //https://www.rapidtables.com/calc/math/convolution-calculator.html
        //1 1 1 0 0 0 * 1 1 1 = 1  2  3  2  1 0 0 0
        //2 2 2 0 0 0 * 2 2 2 = 4  8  12 8  4 0 0 0
    }

    #[test]
    fn test_clc_dot_matches() -> Result<(), ThoosaError> {
        let t_0 = Tensor::vstack(&[
            Tensor::from_slice(&[1., 1., 1., 2., 2., 2., 3., 3., 1.]), //channel 0
            Tensor::from_slice(&[10., 10., 10., 20., 20., 20., 30., 30., 10.]) //channel 1
        ]);

        let motifs_t = Tensor::vstack(&[
            Tensor::vstack(&[ //channel 0
                Tensor::from_slice(&[1., 1., 1.]), //motif 0  - matches 1.,  1.,  1. and  2.,  2.,  2.,
                Tensor::from_slice(&[1., 2., 2.]), //motif 1  - matches  1.,  2.,  2. and  2.,  3.,  3.,
                Tensor::from_slice(&[3., 3., 1.]), //motif 2  - matches 3.,  3.,  1.
                Tensor::from_slice(&[1., 7., 1.]), //motif 3  - no matches
            ]),
            Tensor::vstack(&[ //channel 1
                Tensor::from_slice(&[10., 70., 10.]), //motif 3
                Tensor::from_slice(&[30., 30., 10.]), //motif 2
                Tensor::from_slice(&[10., 20., 20.]), //motif 1
                Tensor::from_slice(&[10., 10., 10.]), //motif 0
            ])
        ]).view((2, 4, -1));

        let matches_t = Tensor::vstack(&[
            Tensor::from_slice(&[ //channel 0
                2, //motif 0
                2, //motif 1
                1, //motif 2
                0, //motif 3
            ]),
            Tensor::from_slice(&[ //channel 1
                0, //motif 3
                1, //motif 2
                2, //motif 1
                2, //motif 0
            ])
        ]).totype(Int);

        assert_eq!(matches_t, super::clc_dot_matches(&t_0, &motifs_t, 1, 0.99)?);

        Ok(())
    }

    #[test]
    fn test_clc_dist_matches() -> Result<(), ThoosaError> {
        let t_0 = Tensor::vstack(&[
            Tensor::from_slice(&[1., 1., 1., 1., 2., 2., 3., 3., 1.]), //channel 0
            Tensor::from_slice(&[10., 10., 10., 20., 20., 20., 30., 30., 10.]) //channel 1
        ]);

        let motifs_t = Tensor::vstack(&[
            Tensor::vstack(&[ //channel 0
                Tensor::from_slice(&[1., 1., 1.]), //motif 0  - matches twice
                Tensor::from_slice(&[1., 2., 2.]), //motif 1  - matches once
                Tensor::from_slice(&[3., 3., 1.]), //motif 2  - matches once
                Tensor::from_slice(&[1., 7., 1.]), //motif 3  - no matches
            ]),
            Tensor::vstack(&[ //channel 1
                Tensor::from_slice(&[10., 70., 10.]), //no matches
                Tensor::from_slice(&[30., 30., 10.]), //matches once
                Tensor::from_slice(&[10., 20., 20.]), //matches once
                Tensor::from_slice(&[10., 10., 10.]), //matches once
            ])
        ]).view((2, 4, -1));

        let matches_t = Tensor::vstack(&[
            Tensor::from_slice(&[ //channel 0
                2, //motif 0
                1, //motif 1
                1, //motif 2
                0, //motif 3
            ]),
            Tensor::from_slice(&[ //channel 1
                0, //motif 3
                1, //motif 2
                1, //motif 1
                1, //motif 0
            ])
        ]).totype(Int);

        let _threshold = 2.;
        let tensor = super::clc_dist_matches(&t_0, Arc::new(RwLock::new(SyncWrapper::new(Box::new(motifs_t), |e|Box::new(e.shallow_clone())))), 1, Arc::new(RwLock::new(HashMap::from([
            (1, 0.01),
            (2, 0.01),
            (3, 0.01),
            (4, 0.01),
            (10, 0.01),
            (20, 0.01),
            (30, 0.01),
            (40, 0.01),
        ]))), Arc::new(RwLock::new(vec![
            vec![1, 2, 3, 4],
            vec![10, 20, 30, 40],
        ])))?;


        assert_eq!(matches_t, tensor);

        Ok(())
    }

    #[test]
    fn test_filter_and_slicer_params() {
        let params = FindMotifsRunConfig::new(None, 10, 10, 1.0, "test", Some(InputFilter::Conv1(Conv1Params::new(10, 1, 1))));

        let properties: HashMap<String, Val> = params.to_properties_map();

        debug!("properties={:?}", properties);

        let sample_properties: HashMap<String, Val> = HashMap::from([
            ("FindMotifsRunConfig.window".into(), UInt(10)),
            ("FindMotifsRunConfig.step".into(), UInt(10)),
            ("FindMotifsRunConfig.dist_quantile".into(), 1.0.into()),
            ("FindMotifsRunConfig.descr".into(), "test".into()),
            ("FindMotifsRunConfig.filter.conv_1.Conv1Params.padding".into(), 1.into()),
            ("FindMotifsRunConfig.filter.conv_1.Conv1Params.stride".into(), 1.into()),
            ("FindMotifsRunConfig.filter.conv_1.Conv1Params.kernel".into(), 10.into()),
        ]);

        //
        // assert_eq!(sample_properties.len(), properties.len(), "len is not same");
        // for (key, val) in sample_properties.iter() {
        //     if properties.get(key).expect(key).ne(val) {
        //         debug!("wrong val key={key:?} sample_val={val:?} properties_val={:?} ", properties.get(key));
        //     }
        // }
        assert_eq!(sample_properties, properties);

        let loaded_params = crate::find::FindMotifsRunConfig::from_properties_map(0, properties);

        debug!("loaded_params={:?}", loaded_params);

        assert_eq!(params, loaded_params);
    }

    #[test]
    fn test_chunk_and_slice() -> Result<(), ThoosaError> {

        let resources: Vec<Resource> = vec!["111", "222", "333", "444", "555"].into_iter().enumerate().map(|(i, v)| (i as u32, 1u32, v, 10).into()).collect();
        let resources_to_move = resources.clone();

        let chunks_0_test: Arc<Mutex<Vec<Tensor>>> = Arc::new(Mutex::new(vec![]));
        let chunks_0_test_rc = Arc::clone(&chunks_0_test);

        let chunks_1_test: Arc<Mutex<Vec<Tensor>>> = Arc::new(Mutex::new(vec![]));
        let chunks_1_test_rc = Arc::clone(&chunks_1_test);

        let executor = LoadRunner::new(
            move |key| {
                let i = resources_to_move.iter().position(|v| v.eq(key)).expect(format!("no {key:?}").as_str()) + 1;
                let v = (0..10).map(|v| i as f32 + v as f32 / 10.).collect::<Vec<f32>>();
                debug!("executor: key={key:?} i={i} v={v:?}");

                SyncWrapper::new_tb(Box::new(Tensor::from_slice(v.as_slice()).view((1, 10))))
            }, 4, 100,
        );
        let executor_rc = Arc::new(RwLock::new(executor));

        crate::find::chunk_and_slice_2x(
            resources,
            2,
            &mut vec![Arc::new(FindMotifsRunConfig::new(Some(1), 3, 5, 1.0, "", None))],
            // HashMap::new(),
            move |chunk_0, chunk_1, _find_motifs_run_config| {
                debug!("chunk_0={chunk_0:?} chunk_1={chunk_1:?}");

                trace!("callback: chunk_0.t=\n{}", chunk_0.t.i((0, .., ..)).squeeze());
                trace!("callback: chunk_1.t=\n{}", chunk_1.t.i((0, .., ..)).squeeze());

                chunks_0_test_rc.lock().push(chunk_0.t.i(( 0, .., ..)).squeeze());
                chunks_1_test_rc.lock().push(chunk_1.t.i(( 0, .., ..)).squeeze());
                Ok(SyncWrapper::new_tb(Box::new(Tensor::from_slice(&[0.1]))))
            },
            &|run_config_id, t, chunk_0, chunk_1|{

            },
            &executor_rc,
            &[Cpu]
        )?;

        let mut chunks_0 = vec![];
        let mut chunks_1 = vec![];

        chunks_0.push(Tensor::from_slice(&[
            1.0000, 1.1000, 1.2000, 1.3000, 1.4000,
            1.3000, 1.4000, 1.5000, 1.6000, 1.7000,
            2.0000, 2.1000, 2.2000, 2.3000, 2.4000,
            2.3000, 2.4000, 2.5000, 2.6000, 2.7000
        ]).view((4, 5)).totype(Float));

        chunks_1.push(Tensor::from_slice(&[
            1.0000, 1.1000, 1.2000, 1.3000, 1.4000,
            1.3000, 1.4000, 1.5000, 1.6000, 1.7000,
            2.0000, 2.1000, 2.2000, 2.3000, 2.4000,
            2.3000, 2.4000, 2.5000, 2.6000, 2.7000
        ]).view((4, 5)).totype(Float));


        chunks_0.push(Tensor::from_slice(&[
            1.0000, 1.1000, 1.2000, 1.3000, 1.4000,
            1.3000, 1.4000, 1.5000, 1.6000, 1.7000,
            2.0000, 2.1000, 2.2000, 2.3000, 2.4000,
            2.3000, 2.4000, 2.5000, 2.6000, 2.7000
        ]).view((4, 5)).totype(Float));

        chunks_1.push(Tensor::from_slice(&[
            3.0000, 3.1000, 3.2000, 3.3000, 3.4000,
            3.3000, 3.4000, 3.5000, 3.6000, 3.7000,
            4.0000, 4.1000, 4.2000, 4.3000, 4.4000,
            4.3000, 4.4000, 4.5000, 4.6000, 4.7000,
        ]).view((4, 5)).totype(Float));


        chunks_0.push(Tensor::from_slice(&[
            1.0000, 1.1000, 1.2000, 1.3000, 1.4000,
            1.3000, 1.4000, 1.5000, 1.6000, 1.7000,
            2.0000, 2.1000, 2.2000, 2.3000, 2.4000,
            2.3000, 2.4000, 2.5000, 2.6000, 2.7000
        ]).view((4, 5)).totype(Float));

        chunks_1.push(Tensor::from_slice(&[
            5.0000, 5.1000, 5.2000, 5.3000, 5.4000,
            5.3000, 5.4000, 5.5000, 5.6000, 5.7000,
        ]).view((2, 5)).totype(Float));


        //////////////////
        chunks_0.push(Tensor::from_slice(&[
            3.0000, 3.1000, 3.2000, 3.3000, 3.4000,
            3.3000, 3.4000, 3.5000, 3.6000, 3.7000,
            4.0000, 4.1000, 4.2000, 4.3000, 4.4000,
            4.3000, 4.4000, 4.5000, 4.6000, 4.7000,
        ]).view((4, 5)).totype(Float));
        chunks_1.push(Tensor::from_slice(&[
            3.0000, 3.1000, 3.2000, 3.3000, 3.4000,
            3.3000, 3.4000, 3.5000, 3.6000, 3.7000,
            4.0000, 4.1000, 4.2000, 4.3000, 4.4000,
            4.3000, 4.4000, 4.5000, 4.6000, 4.7000,
        ]).view((4, 5)).totype(Float));

        chunks_0.push(Tensor::from_slice(&[
            3.0000, 3.1000, 3.2000, 3.3000, 3.4000,
            3.3000, 3.4000, 3.5000, 3.6000, 3.7000,
            4.0000, 4.1000, 4.2000, 4.3000, 4.4000,
            4.3000, 4.4000, 4.5000, 4.6000, 4.7000,
        ]).view((4, 5)).totype(Float));
        chunks_1.push(Tensor::from_slice(&[
            5.0000, 5.1000, 5.2000, 5.3000, 5.4000,
            5.3000, 5.4000, 5.5000, 5.6000, 5.7000,
        ]).view((2, 5)).totype(Float));
        ///////////////////

        chunks_0.push(Tensor::from_slice(&[
            5.0000, 5.1000, 5.2000, 5.3000, 5.4000,
            5.3000, 5.4000, 5.5000, 5.6000, 5.7000,
        ]).view((2, 5)).totype(Float));
        chunks_1.push(Tensor::from_slice(&[
            5.0000, 5.1000, 5.2000, 5.3000, 5.4000,
            5.3000, 5.4000, 5.5000, 5.6000, 5.7000,
        ]).view((2, 5)).totype(Float));


        assert_eq!(chunks_0, *chunks_0_test.lock());
        assert_eq!(chunks_1, *chunks_1_test.lock());

        Ok(())
    }

    // /**
    //
    //
    // //ChunkOfSicedTensors is a structure holding information about stacked Tensors
    // struct ChunkOfSicedTensors<'a> {
    //     keys: &'a Vec<String>, //Tensor's keys (each tensor is identified by a String key
    //     sizes: Vec<i64>,  //Tensors sizes -
    //     t: Tensor, //
    // }
    //
    //
    // //TensorSliceIdx is a structure to identify a slice in a tensor by tensor's key and slice index
    // pub struct TensorSliceIdx<T>(pub T, pub i64) where T:Eq+Display+Debug;
    // */
    //
    // #[test]
    // fn test_collect_distances(){
    //     //x is made by stacking slices of tensors "aaa" and "bbb"
    //     let aaa_bbb_t = [
    //         10.9, 11.8, //slice 0 of tensor "aaa"
    //         12.7, 13.6, //slice 1 of tensor "aaa"
    //         14.5, 15.4, //slice 2 of tensor "aaa"
    //         16.3, 17.2, //slice 0 of tensor "bbb"
    //         18.1, 19.0, //slice 1 of tensor "bbb"
    //     ];
    //
    //     let ccc_ddd_t = [
    //         20.9, 21.8, //slice 0 of tensor "ccc"
    //         24.5, 25.4, //slice 2 of tensor "ddd"
    //         26.3, 27.2, //slice 0 of tensor "ddd"
    //     ];
    //
    //
    //     let slices_chunk_0 = Tensor::from_slice(&aaa_bbb_t).view((5, 2));
    //     let slices_chunk_1 = Tensor::from_slice(&ccc_ddd_t).view((3, 2));
    //
    //     let dist_t = super::dist_d2(&slices_chunk_0, &slices_chunk_1).unwrap();
    //     // let dist_t =  Tensor::cdist(&slices_chunk_0, &slices_chunk_1, 2.0, 0);
    //
    //     let mut distance_pairs = Vec::new();
    //     let chunk_0 = ChunkOfSlicedTensors::new(
    //         vec![0, 1],
    //         vec![3, 2],
    //         slices_chunk_0
    //     );
    //     let chunk_1 = ChunkOfSlicedTensors::new(
    //         vec![2, 3],
    //         vec![1, 2],
    //         slices_chunk_1
    //     );
    //     collect_dists(
    //         &chunk_0,
    //         &chunk_1,
    //         &from::<Vec<f32>>(&dist_t.view(-1)),
    //         1,
    //         |pair|{
    //             distance_pairs.push(pair);
    //         },
    //         |dist|{
    //             true
    //         }
    //     );
    //
    //     println!("distance_pairs={distance_pairs:?}");
    //
    //     assert_eq!(aaa_bbb_t.len() / 2 * ccc_ddd_t.len() / 2,  distance_pairs.len());
    //     assert_eq!(
    //        distance_pairs[0],
    //        (TensorSliceIdx(0, 0), TensorSliceIdx(2, 0), test_dist(&aaa_bbb_t[0..2], &ccc_ddd_t[0..2])),// dist between slices 0 and 1 of tensor "aaa" - [0.9000, 1.8000], [2.7000, 3.6000] is 2.5456
    //     );
    //
    //     assert_eq!(
    //        distance_pairs[1],
    //        (TensorSliceIdx(0, 0), TensorSliceIdx(3, 0), test_dist(&aaa_bbb_t[0..2], &ccc_ddd_t[2..4])),// dist between slices 0 and 1 of tensor "aaa" - [0.9000, 1.8000], [2.7000, 3.6000] is 2.5456
    //     );
    //
    //     assert_eq!(
    //        distance_pairs[2],
    //        (TensorSliceIdx(0, 0), TensorSliceIdx(3, 1), test_dist(&aaa_bbb_t[0..2], &ccc_ddd_t[4..6])),// dist between slices 0 and 1 of tensor "aaa" - [0.9000, 1.8000], [2.7000, 3.6000] is 2.5456
    //     );
    //
    //     assert_eq!(
    //        distance_pairs[3],
    //        (TensorSliceIdx(0, 1), TensorSliceIdx(2, 0), test_dist(&aaa_bbb_t[2..4], &ccc_ddd_t[0..2])),// dist between slices 0 and 1 of tensor "aaa" - [0.9000, 1.8000], [2.7000, 3.6000] is 2.5456
    //     );
    //
    //
    // }

    fn test_dist(x:&[f32], y:&[f32]) -> f32 {
        ((x[0]-y[0]) * (x[0]-y[0]) + (x[1]-y[1]) * (x[1]-y[1])).sqrt()
    }


    #[test]
    fn test_clc_motif_matches() -> Result<(), ThoosaError>{
        let n = 1000;

        let max_y = 30.0;

        let sig_0_v = sig_one_imp(n, max_y, None, None, 0.1);
        let sig_1_v = sig_two_imp(n, max_y, None, None, 0.1);
        let sig_2_v = sig_three_imp(n, max_y, None, None);

        let x = lin_t(3 * n);
        //each [3, 3000]
        let (t_0_0, t_0_1, t_1_0, t_1_1) = three_sigs_tensor(n, &sig_0_v, &sig_1_v, &sig_2_v);

        let motif_0 = Tensor::from_slice(sig_0_v.as_slice());
        let motif_1 = Tensor::from_slice(sig_1_v.as_slice());
        let motif_2 = Tensor::from_slice(sig_2_v.as_slice());

        //[3, 1, 1000] = 3 channels, 1 motif in each, 1000 motif length
        let motifs = Tensor::stack(&[
            Tensor::stack(&[motif_0],0),//ch 0
            Tensor::stack(&[motif_1],0),//ch 1
            Tensor::stack(&[motif_2],0) //ch 2
        ], 0);
        // let t = Tensor::vstack(&[
        //     Tensor::from_slice(&[1.1, 1.2, 1.3]),
        //     Tensor::from_slice(&[2.1, 2.2, 2.3]),
        //     Tensor::from_slice(&[3.1, 3.2, 13.3])
        // ]);
        debug!("motifs.size()={:?}", motifs.size());
        debug!("t_0_0.size()={:?}", t_0_0.size());
        debug!("t_0_1.size()={:?}", t_0_1.size());
        debug!("t_1_0.size()={:?}", t_1_0.size());
        debug!("t_1_1.size()={:?}", t_1_1.size());

        // plot_series(
        //     vec![
        //         (format!("ch 0"), plotters::prelude::RED, vec_f32_from(&x).into_iter().zip(vec_f32_from(&t_0_0.i((0, ..)))).collect::<Vec<(f32, f32)>>()),
        //         (format!("ch 1"), plotters::prelude::RED, vec_f32_from(&x).into_iter().zip(vec_f32_from(&t_0_0.i((1, ..)))).collect::<Vec<(f32, f32)>>()),
        //         (format!("ch 2"), plotters::prelude::RED, vec_f32_from(&x).into_iter().zip(vec_f32_from(&t_0_0.i((2, ..)))).collect::<Vec<(f32, f32)>>()),
        //
        //     ], "test_clc_motif_matches t_0_0", (2000, 1000), &format!("/tmp/plots/0_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str()
        // )?;

        let t_0= clc_motif_matches(
            &motifs,
            &ChunkOfSlicedTensors::slice_tensors_chunk(
                100,
                1000,
                &vec![t_0_0.into(), t_0_1.into()],
                    vec![0, 1],
                    Cpu
            ).t,
            &Tensor::from_slice(&[1, 1, 1]).view((3, 1)),
            &Tensor::from_slice(&[1.0, 1.0, 1.0]).view((3, 1)),
        );

        debug!("t_0={t_0}");

        let t_1 = clc_motif_matches(
            &motifs,
            &ChunkOfSlicedTensors::slice_tensors_chunk(
                100,
                1000,
                &vec![t_1_0.into(), t_1_1.into()],
                vec![0, 1],
                Cpu
            ).t,
            &Tensor::from_slice(&[1, 1, 1]).view((3, 1)),
            &Tensor::from_slice(&[1.0, 1.0, 1.0]).view((3, 1)),
        );

        debug!("t_1={t_1}");
        //TODO

        Ok(())
    }

    #[test]
    fn test_select_motifs_by_label_variance(){
        let mask = Tensor::from_slice(&[
            1, 1, 1, 1, 1, 1, 1, // channel_0
            0, 0, 0, 0, 0, 0, 0, // channel_1
            1, 1, 1, 1, 1, 1, 1, // channel_2
        ]).totype(Bool).view([3, 7]);

        log::set_max_level(Trace);
        let matches = Tensor::from_slice(&[
            0,   0,   0,   0,  0,  0,  0, // label_1, channel_0
            0,   0,   0,   0,  0,  0,  0, // label_1, channel_0
            11,  12,  13,  14, 15, 16, 17, // label_1, channel_1

            1,   2,   3,   4,  5,  6,  7, // label_1, channel_0
            1,   2,   3,   4,  5,  6,  7, // label_1, channel_0
            11,  12,  13,  14, 15, 16, 17, // label_1, channel_1
        ]).totype(Float).view([2, 3, 7]); // 2 labels x 3 channels x 7 motifs


        let (good_motifs, motifs_rates) = select_motifs_by_label_variance(&matches, &mask);
        assert_eq!(
            Tensor::from_slice(&[
                // idx_dispersion_mean = 4
                false, false, false, true , true , true , true,
                //masked
                false, false, false, false, false, false, false,
                // idx_dispersion_mean = 1
                false, false, false, false, false, false, false
            ]).view((3,7)),
            good_motifs
        );

        let good_rates = from::<Vec<Vec<bool>>>(&good_motifs)
            .into_iter()
            .flat_map(|v|v.into_iter())
            .zip(from::<Vec<Vec<f32>>>(&motifs_rates).into_iter().flat_map(|v|v.into_iter()))
            .filter(|(b, r)|*b)
            .map(|(b, r)|r).collect::<Vec<_>>();

        debug!("motifs_rates=\n{motifs_rates}");
        debug!("good_rates=\n{good_rates:?}");

        assert_eq!(good_rates, vec![0.5714286, 0.71428573, 0.85714287, 1.0]);

    }
}

pub fn plot_motif_slices(ws: &Arc<RwLock<Box<dyn Workspace>>>, store: &Arc<Mutex<Box<dyn Store>>>, executor: &Arc<RwLock<LoadRunner<Box<Tensor>>>>, motif_id: usize, device: &Device) -> Result<(), ThoosaError> {
    let (channel, run_id, slices, resources_map) = store.lock().list_motif_slices(motif_id, None, None)?;
    let slices_len = slices.len();
    // if slices_len < 3 {
    //     return Ok(())
    // }

    let find_motifs_run_config = FindMotifsRunConfig::from_properties_map(run_id, store.lock().list_run_config_values(run_id)?);

    let resources_slices = load_motif_slices(channel, find_motifs_run_config.window(), slices.into_iter().map(|s| TensorIdSliceIdx(s.0.into(), s.1)).collect(), resources_map, executor); //TODO Into for TensorSliceIdx

    let all_slices = resources_slices.iter().flat_map(|(_id, slices)| slices.iter()).collect::<Vec<&Box<Tensor>>>();

    let all_slices= if all_slices.len() > 3_000 {
        warn!("Too many slices ({}) in motif {motif_id} to plot, truncating", all_slices.len());
        &all_slices[..3_000]
    } else {
        debug!("plot_motif_slices: all_slices.len()={}", all_slices.len());
        &all_slices
    };

    let mut slices_t = Tensor::vstack(all_slices.iter().map(|s| s.shallow_clone()).collect::<Vec<Tensor>>().as_slice()).to_device(*device);
    
    let motif_avg_dist = loop {
        match panic_to_err!(average_euclidean_distance(&slices_t)){
            Ok(v) => {
                break Ok(v);
            }
            Err(e) if e.is_oom()=> {
                let size = slices_t.size()[0]/2;
                slices_t = slices_t.i((..size, ..));
                debug!("plot_motif_slices cut slices_t.size={:?}", slices_t.size());
            }
            Err(e) => {
                break Err(ThoosaError::from(e))
            }
        }
    }?;

    // check_slices(&all_sices, 0.9);
    let mut color_iter = DistinctColorIter::new();
    plot_series(
        all_slices.into_iter().enumerate().map(|(i, t)| {
            (format!("{i}"), color_iter.next().unwrap_or(BLACK), t.iter::<f64>().unwrap().enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>())
        }).collect(),
        format!("Motif: {motif_id} run id: {run_id} channel: {channel} number of slices: {slices_len} motif_avg_dist={motif_avg_dist}").as_str(),
        (2000, 1000),
        &ws.read().plots_path().join(run_id.to_string()).join(format!("motif_{}.png", motif_id))
    )?;

    Ok(())
}