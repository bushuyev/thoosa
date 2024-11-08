use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::{fmt, fs, mem};
use std::cell::Cell;
use std::fs::{File, OpenOptions};
use std::hash::Hash;
use std::io::Write;
use std::marker::PhantomData;
use std::ops::{Deref, Range};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::Mutex;
use std::time::{Instant};

use clap::clap_derive::{Args, Subcommand};
use crossbeam_utils::sync::WaitGroup;
use futures::FutureExt;
use log::{debug, info, trace, warn};
use parking_lot::RwLock;
use tch::{Device, IndexOp, Tensor};


use thoosa_common::utils::*;
use thoosa_shared::{make_combos, RangeOrValue, Resources, ThoosaError};
use thoosa_shared::parse_key_val;
use thoosa_shared::TensorSliceIdx;
use thoosa_shared::val::Val;
use thoosa_store::{SqlStore, Store};
use thoosa_store::StoreDumper;


use thoosa_common::loader::{LoadRunner, SyncWrapper};
use thoosa_common::loader::Loader;
use crate::distance::dist_d2;
use crate::distance::slice_t;
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelBridge};
use rayon::iter::ParallelIterator;
use rayon::iter::IndexedParallelIterator;
use tch::Device::{Cpu, Cuda};
use thoosa_common::loader;
use thoosa_common::parzen::optimizer::{ConfigurationSpace, Hyperparameter};
use thoosa_common::runs::{Params, ResourcesConfig, RunConfig, Simple};
use thoosa_common::ws::{FsWorkspace, Workspace};
use thoosa_shared::parse_range_or_value;
use thoosa_shared::parse_resources;
use thoosa_shared::resource::Resource;

#[derive(Debug, Args, Clone)]
pub struct FindMotifsParams {//TODO pub
    #[arg(long, value_parser=parse_range_or_value::<u32>, default_value = "1000")]
    pub step: RangeOrValue<u32>,

    #[arg(long, value_parser=parse_range_or_value::<u32>, default_value = "1000")]
    pub window: RangeOrValue<u32>,

    #[arg(long, default_value_t = 0.9)]
    pub min_dot: f32,

    #[arg(long,value_parser=parse_resources::<Resources>)]
    pub resources: Resources,

    #[arg(long, default_value = "10")]
    pub min_tracker_step: Option<usize>,

    #[arg(long, default_value_t = 100)]
    pub max_keys_fifo: usize,

    #[arg(long, default_value_t = 100)]
    pub chunk_size: u32,

    #[arg(long, default_value_t = 0.01)]
    pub max_dist: f32,

    #[command(subcommand)]
    pub input_filter: Option<InputFilter>,
}

#[derive(Debug)]
pub struct CommonFindMotifsConfig {
    min_tracker_step:Option<usize>,
    chunk_size:u32
}

impl CommonFindMotifsConfig {
    pub fn new(min_tracker_step: Option<usize>, chunk_size: u32) -> Self {
        Self { min_tracker_step, chunk_size }
    }
}

impl Params<FindMotifsRunConfig, Simple, CommonFindMotifsConfig> for FindMotifsParams {
    fn to_run_configs(&self) -> Vec<FindMotifsRunConfig> {
        // vec![
        //     FindMotifsRunConfig::new(10, 10, "", None)
        // ]

        debug!("FindMotifsParams::to_run_configs: self.step={:?} self.window={:?}", self.step, self.window);

        make_combos(vec![
            self.step.clone().into_iter(),
            self.window.clone().into_iter(),
        ]).into_iter().map(|config_combo|{
            debug!("FindMotifsParams::to_run_configs: config_combo={config_combo:?} ");
            FindMotifsRunConfig::new(config_combo[0], config_combo[1], 1.0, "".to_string(), None)
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
            min_tracker_step: self.min_tracker_step
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

    pub fn from_properties_map(properties: &HashMap<String, Val>) -> Result<Self, ThoosaError> {
        let conv_1_props = thoosa_shared::val::unprefix_properties("conv_1", &properties);
        if conv_1_props.len() > 0 {
            assert!(conv_1_props.len() == 3);
            return Ok(InputFilter::Conv1(Conv1Params::from_properties_map(conv_1_props)?));
        }

        Err(ThoosaError::new(format!("can't build InputFilter from {:?}", properties)))
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
    pub fn dummy(step: u32, window: Range<u32>, resource_set: String, chunk_size: u32, max_dist_k: f32, input_filter: Option<&InputFilter>) -> Self {
        FindMotifsParams {
            step:step.into(),

            window:window.into(),

            min_dot: 0.1,

            resources:Resources::Set(resource_set),

            min_tracker_step: Some(10),

            max_keys_fifo: 1000,

            chunk_size,

            max_dist: max_dist_k,

            input_filter: input_filter.cloned(),
        }
    }

    pub fn min_dot(&self) -> f32 {
        self.min_dot
    }
    pub fn resources(&self) -> &Resources {
        &self.resources
    }
    pub fn max_keys_fifo(&self) -> usize {
        self.max_keys_fifo
    }
    pub fn max_dist_k(&self) -> f32 {
        self.max_dist
    }

    pub fn to_filter_and_slice_params(&self, expander: &dyn Expander<T=FindMotifsRunConfig>) -> Vec<FindMotifsRunConfig> {
        expander.expand(FindMotifsRunConfig::new(*self.step.expect_value().unwrap(), *self.window.expect_value().unwrap(), 1.0, "no filter", self.input_filter.clone()))
    }
}

pub trait Expander {
    type T: Clone;
    fn expand(&self, t: Self::T) -> Vec<Self::T>;
}

pub struct LinearExpander {
    n: usize,
}

impl LinearExpander {
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl Expander for LinearExpander {
    type T = FindMotifsRunConfig;

    fn expand(&self, t: FindMotifsRunConfig) -> Vec<FindMotifsRunConfig> {
        (0..self.n as i64).map(|i| {
            let mut params = t.clone();
            // t.window * i;
            // params.filter = Some(InputFilter::Conv1(params.filter.map(|InputFilter::Conv1(f)| {
            //     Conv1Params::new(f.kernel + f.kernel * i, f.stride, f.padding)
            // }).unwrap_or(Conv1Params::new(10, 10, 10))));
            params
        }).collect()
    }
}

fn resource_config_to_keys(store:&Arc<Mutex<Box<dyn Store>>>, resource_config:&Simple) -> Vec<Resource>{
    match resource_config.resources() {
        Resources::List(list) => {
            store.lock().list_resources_by_file_stems(&list.into_iter().map(|v|v.as_str()).collect()).expect("list_resources_by_file_stems")
        }
        Resources::Set(set) => {
            store.lock().list_resource_set_resources_by_descr(set).expect("list_resource_set_resources")
        }
        Resources::Group(group) => {
            let mut store_guard = store.lock();
            //TODO move into store
            store_guard.list_resource_set_group_sets(group).expect("list_resource_set_group_sets").into_iter().flat_map(|v|
                store_guard.list_resource_set_resources_by_descr(&v.1).expect("list_resource_set_resources_by_descr")
            ).collect()
        }
    }
}

pub fn find_motifs_by_euc_dist<K>(
    run_configs: Vec<FindMotifsRunConfig>,
    resource_config: &Simple,
    common_config: &CommonFindMotifsConfig,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Mutex<Box<dyn Store>>>,
    loader: &Arc<RwLock<LoadRunner<K, Box<Tensor>>>>,
    devices:Vec<Device>
) -> Result<Vec<usize>, ThoosaError>
where K: Clone + Hash + Eq + From<String>+ Send + Sync + Debug + Display,
      Vec<K>: FromIterator<Arc<String>>  {

    let resource_keys = resource_config_to_keys(store, resource_config)
        .into_iter()
        .map(|key|Arc::new(key))
        .collect::<Vec<K>>();

    let run_config_ids:Vec<usize> = run_configs.iter().map(|run_config| run_config.id.expect("id")).collect();

    let run_config__trackers: Vec<Vec<Arc<RwLock<SimpMotifTracker<StoreDumper, K>>>>> = run_configs.iter().map(|run_config| {
        (0..ws.read().channels()).map(|channel|
                Arc::new(RwLock::new(SimpMotifTracker::<StoreDumper, K>::new(
                    StoreDumper::new(Arc::clone(&store), run_config.id.expect("id"), channel),
                    common_config.min_tracker_step,
            )))).collect::<Vec<Arc<RwLock<SimpMotifTracker<StoreDumper, K>>>>>()

    }).collect();

    // debug!("find_motifs_by_euc_dist_: records_num={records_num} resource_keys={resource_keys:?} run_config_ids={run_config_ids:?}");
    chunk_and_slice(
        resource_keys,
        common_config.chunk_size as usize,
        &run_configs,
        run_config__trackers,
        move |chunk_0: ChunkOfSlicedTensors<K>, chunk_1: ChunkOfSlicedTensors<K>, params_idx: usize, find_motifs_run_config:&FindMotifsRunConfig, channels_motif_trackers:&Vec<Arc<RwLock<SimpMotifTracker<StoreDumper, K>>>>| {
            debug!("find_motifs_by_euc_dist: tensor_0.chunk_0={:?} chunk_1={:?} find_motifs_run_config={find_motifs_run_config:?} params_idx={:?}", chunk_0, chunk_1 ,params_idx);

            //TODO spread channels over devices
            channels_motif_trackers.iter().enumerate().par_bridge().for_each(|(channel, mt)| {
                let tensor_0 = chunk_0.t.i((.., channel as i64, ..)).squeeze();
                let tensor_1 = chunk_1.t.i((.., channel as i64, ..)).squeeze();
                debug!("find_motifs_by_euc_dist: tensor_0.size={:?} tensor_1.size()={:?}", tensor_0.size(), tensor_1.size());

                let euc_dist_t = dist_d2(
                    &tensor_0.to_device(devices[0]),
                    &tensor_1.to_device(devices[0])
                ).expect(format!("dist_d2 on channel={}", channel, ).as_str());

                collect_dists(
                    &chunk_0,
                    &chunk_1,
                    &euc_dist_t,
                    find_motifs_run_config.step as i64,
                    |(slice_0, slice_1, dist)| {
                        if dist < find_motifs_run_config.max_dist {
                            mt.write().track(
                                slice_0,
                                slice_1,
                                dist,
                            );
                        }
                    },
                    |dist: &f32| {
                        dist < &find_motifs_run_config.max_dist
                    }
                );
                debug!("find_motifs_by_euc_dist: channel={channel} done");
                mt.write().dump();
            });

            Ok(())
        },
        loader,
    )?;

    for run_config_id in run_config_ids.iter() {
        for channel in 0..ws.read().channels() {
            { store.lock().set_motifs(*run_config_id, channel as i64) }?;
        }

        { store.lock().delete_slices_without_matches(*run_config_id) }?;
    }

    // info!("{} done {}/{}", find_motifs_params.resource_set, slots_to_process, slots_processed);
    // info!("{} done {}/{}", find_motifs_params.resource_set, slots_to_process, slots_processed);


    Ok(run_config_ids)//TODO return only successful run_id (window size check)?
}


#[derive(Debug)]
struct ChunkOfSlicedTensors<K> where K: Clone + Hash + Eq + From<String>+ Send + Sync + Debug + 'static {
    keys:Vec<K>,
    sizes: Vec<i64>,
    t: SyncWrapper<Tensor>,
}

impl  <K> ChunkOfSlicedTensors<K> where K: Clone + Hash + Eq + From<String>+ Send + Sync + Debug + 'static {
    pub fn new(keys:  Vec<K>, sizes: Vec<i64>, t: Tensor) -> Self {
        Self {
            keys,
            sizes,
            t:SyncWrapper::new_t(t),
        }
    }

    pub fn slice_tensors_chunk(step: u32, window: u32, tensors_chunk: &Vec<SyncWrapper<Box<Tensor>>>, keys: Vec<K>) -> ChunkOfSlicedTensors<K> {

        let tensors_sliced = tensors_chunk.into_iter().map(|t| {

            let sliced = slice_t(
                &t,
                step,
                window,
            );

            debug!("slice_tensors_chunk: sliced size={:?}", sliced.size());

            sliced
        }).collect::<Vec<Tensor>>();


        ChunkOfSlicedTensors::new(
            keys,
            tensors_sliced.iter().map(|t| t.size()[0]).collect::<Vec<_>>(), //num slices in each tensor,
            Tensor::vstack(&tensors_sliced),
        )
    }

}

#[derive(Clone, PartialEq, Debug)]
pub struct FindMotifsRunConfig {
    id:Option<usize>,
    step: u32,
    window: u32,
    max_dist: f32,
    min_tracker_step: Option<usize>,
    filter: Option<InputFilter>,
    descr: String,
}

impl RunConfig for FindMotifsRunConfig {
    fn to_properties_map(&self) -> HashMap<String, Val> {
        let mut map = HashMap::from([
            ("FilterAndSlicerParams.step".into(), self.step.into()),
            ("FilterAndSlicerParams.window".into(), self.window.into()),
            ("FilterAndSlicerParams.max_dist".into(), self.max_dist.into()),
            ("FilterAndSlicerParams.descr".into(), self.descr.as_str().into()),
        ]);

        if let Some(filter) = self.filter.as_ref() {
            map.extend(thoosa_shared::val::prefix_properties("FilterAndSlicerParams.filter", filter.to_properties_map()).into_iter());
        }

        map
    }

    fn from_properties_map(properties: HashMap<String, Val>) -> Self {
        let properties = thoosa_shared::val::unprefix_properties("FilterAndSlicerParams", &properties);
        FindMotifsRunConfig::new(
            properties.get("step").expect("no step").as_usize().expect("step as_usize") as u32,
            properties.get("window").expect("no window").as_i64().expect("window as_i64") as u32,
            properties.get("max_dist").expect("no max_dist").as_f32().expect("window as_f32") as f32,
            properties.get("descr").expect("no descr").as_str().expect("descr as_str"),
            InputFilter::from_properties_map(&thoosa_shared::val::unprefix_properties("filter", &properties)).ok(),
        )
    }

    fn set_id(&mut self, id:usize) {
        self.id.replace(id);
    }
}

impl FindMotifsRunConfig {
    pub fn new(
        step: u32,
        window: u32,
        max_dist: f32,
        descr: impl Into<String>,
        filter: Option<InputFilter>,
    ) -> Self {
        Self {
            id:None,
            step,
            window,
            max_dist,
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

fn chunk_and_slice<G, K>(
    resources_ids: Vec<K>,
    chunk_size: usize,
    run_configs:&Vec<FindMotifsRunConfig>,
    mut find_motifs_run_configs_trackers: Vec<Vec<Arc<RwLock<SimpMotifTracker<StoreDumper, K>>>>>,
    mut callback: G,
    exec: &Arc<RwLock<LoadRunner<K, Box<Tensor>>>>,
) -> Result<(), ThoosaError>
where
    G: Fn(ChunkOfSlicedTensors<K>, ChunkOfSlicedTensors<K>, usize, &FindMotifsRunConfig, &Vec<Arc<RwLock<SimpMotifTracker<StoreDumper, K>>>>) -> Result<(), ThoosaError> + Send + Sync,
    K: Clone + Hash + Eq + From<String>+ Send + Sync + Display + Debug,
{
    let mut resources_ids_chunks = resources_ids.chunks(chunk_size)
        .map(|v| v.iter().map(|k| k.clone()).collect::<Vec<K>>())
        .collect::<Vec<Vec<K>>>();

    resources_ids_chunks.push(vec![]); //to get last window processed


    let area_0 = exec.write().get_available_cache_area();
    let area_1 = exec.write().get_available_cache_area();

    exec.write().preload(&resources_ids_chunks[0], area_0);

    if resources_ids_chunks.len() > chunk_size {
        exec.write().preload(&resources_ids_chunks[chunk_size], area_0);
    }


    //first element in each window to be loaded and processed, second - preloaded
    for (chunk_i_0, window_0) in resources_ids_chunks.windows(2).enumerate() {
        //window_0 - current,preload windows
        debug!("chunk_and_slice: chunk_i_0={chunk_i_0} window_0={window_0:?}");

        if window_0[1].len() > 0 {
            exec.write().preload(&window_0[1],  area_0);
        }


        let wg_1 = WaitGroup::new();


        let window_0_0 = window_0[0].clone();
        let window_0_0_clone = window_0_0.clone();

        let wg_1_clone = wg_1.clone();
        let window_0_tensors = Arc::new(RwLock::new(vec![]));
        let window_0_tensors_arc = Arc::clone(&window_0_tensors);

        // let window_0_0_clone = window_0_0.clone();

        exec.write().exec(|loader, keys_purge| {
            async move {

                for key in &window_0_0 {
                    debug!("0. exec:callback_arc: key={key:?}");

                    let sync_wrapper = SyncWrapper::new_tb(Loader::<K, Box<Tensor>>::load_future(Arc::clone(&loader), key.clone(), area_0).await);
                    window_0_tensors_arc.write().push(sync_wrapper);
                }
                drop(wg_1_clone);

            }.boxed()
        });

        wg_1.wait();

        for (chunk_i_1, window_1) in resources_ids_chunks.windows(2).into_iter().skip(chunk_i_0).enumerate() {
            debug!("chunk_and_slice: chunk_i_1={chunk_i_1} window_1={window_1:?}");

            if window_1[1].len() > 0 {
                exec.write().preload(&window_1[1], area_1);
            }
            trace!("chunk_and_slice: area_1={area_1:064b}");

            let wg_1 = WaitGroup::new();


            let window_1_0 = window_1[0].clone();
            let window_1_0_clone = window_1_0.clone();

            let wg_1_clone = wg_1.clone();
            let window_1_tensors = Arc::new(RwLock::new(vec![]));
            let window_1_tensors_arc = Arc::clone(&window_1_tensors);

            exec.write().exec(|loader, keys_purge| {
                async move {

                    for key in &window_1_0 {
                        debug!("1. exec:callback_arc: key={key:?}");

                        let sync_wrapper = SyncWrapper::new_tb(Loader::<K, Box<Tensor>>::load_future(Arc::clone(&loader), key.clone(), area_1).await);
                        window_1_tensors_arc.write().push(sync_wrapper);
                    }


                    drop(wg_1_clone);

                }.boxed()
            });
            wg_1.wait();

            // let callback_clone = Cell::new(callback_clone.clone());
            run_configs.iter().zip(find_motifs_run_configs_trackers.iter())
                .par_bridge()
                .for_each(|(run_config, mt)| {

                    callback(
                        ChunkOfSlicedTensors::slice_tensors_chunk(
                            run_config.step,
                            run_config.window,
                            &window_0_tensors.read(),
                            window_0_0_clone.clone()
                        ),
                        ChunkOfSlicedTensors::slice_tensors_chunk(
                            run_config.step,
                            run_config.window,
                            &window_1_tensors.read(),
                            window_1_0_clone.clone()
                        ),
                        0,
                        run_config,
                        mt
                    ).expect("callback_arc failed");
                });

            exec.write().purge_cache_keys(window_1[0].clone(), area_1);

        }
        exec.write().purge_cache_keys(window_0[0].clone(), area_0);

    }
    Ok(())
}

fn collect_dists<'a, F, C, K>(
    chunk_0: &'a ChunkOfSlicedTensors<K>,
    chunk_1: &'a ChunkOfSlicedTensors<K>,
    dist_t:&Tensor, step: i64,
    mut listener:F,
    condition:C
) where F:FnMut((TensorSliceIdx<K>, TensorSliceIdx<K>, f32)),
        C:Fn(&f32)->bool,
        K: Clone + Hash + Eq + From<String>+ Send + Sync + Debug + Display + 'static
{


    let distances = from::<Vec<f32>>(&dist_t.view(-1));
    let mut distances_i = 0;

    chunk_0.sizes.iter().enumerate().for_each(|(t_idx_0, t_size_0)| {
        for (slice_0_idx, s_0) in (0..*t_size_0).enumerate() {

            for (t_idx_1, t_size_1) in chunk_1.sizes.iter().enumerate() {
                for (slice_1_idx, s_1) in (0..*t_size_1).enumerate() {
                    //println!("t_idx_0={t_idx_0} slice_0_idx={slice_0_idx} t_idx_1={t_idx_1} slice_1_idx={slice_1_idx} d={}", distances[distances_i]);

                    if chunk_0.keys[t_idx_0].ne(&chunk_1.keys[t_idx_1]) || slice_0_idx != slice_1_idx && condition(&distances[distances_i]) {
                        listener(
                            (
                                TensorSliceIdx(chunk_0.keys[t_idx_0].clone(), slice_0_idx as i64 * step),
                                TensorSliceIdx(chunk_1.keys[t_idx_1].clone(), slice_1_idx as i64 * step),
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



pub trait MotifDumper<K:Eq+Display+Debug> {
    fn dump(&mut self, motifs: Vec<(TensorSliceIdx<K>, TensorSliceIdx<K>, f32)>) -> Result<(), ThoosaError>;
}

impl <K:Eq+Display+Debug> MotifDumper<K> for StoreDumper {
    fn dump(&mut self, motifs: Vec<(TensorSliceIdx<K>, TensorSliceIdx<K>, f32)>) -> Result<(), ThoosaError> {
        debug!("MotifDumper::dump: motifs.len()={:?}, self.run_id={:?}", motifs.len(), self.run_id);

        self.store.lock().save_slices(self.run_id, self.channel as i64, motifs.into_iter().map(|v| ((v.0.0.to_string(), v.0.1), (v.1.0.to_string(), v.1.1), v.2)).collect::<Vec<_>>())?; //TODO move TensorSliceIdx to shared

        Ok(())
    }
}

//TODO trait for MotifTracker
#[derive(Debug)]
pub(crate) struct SimpMotifTracker<D: MotifDumper<K> + Debug, K:Eq + Display + Debug> {
    last:Option<(TensorSliceIdx<K>, TensorSliceIdx<K>, f32)>,
    motifs: Vec<(TensorSliceIdx<K>, TensorSliceIdx<K>, f32)>, //i_0, match_0, match_1, dist
    dumper: D,
    min_step:Option<usize>
}

impl<'a, D: MotifDumper<K> + Debug, K:Eq + Display + Debug> SimpMotifTracker<D, K> {

    pub fn new(dumper: D, min_step:Option<usize>) -> SimpMotifTracker<D, K> {
        Self {
            last:None,
            motifs: vec![],
            dumper,
            min_step
        }
    }

    pub fn track(&'a mut self, i_0: TensorSliceIdx<K>, i_1: TensorSliceIdx<K>, dist: f32) {
        debug!("SimpMotifTracker::track: i_0={i_0:?} i_1={i_1:?} dist={dist}");

        match self.min_step {
            None => {
                self.motifs.push((i_0, i_1, dist));
            }

            Some(min_step) => {

                let last = self.last.take();
                if let Some(last) = last {
                    if i_0.1 - last.0.1  > min_step as i64 || i_1.1 - last.1.1  > min_step as i64 {//i_. .1 < last.1.1 if next tensor
                        self.motifs.push(last);
                    }
                }
                self.last.replace((i_0, i_1, dist));
            }
        }

    }

    fn dump(&mut self) {
        debug!("SimpMotifTracker::dump: self.motifs={}", self.motifs.len());
        if let Some(last) = self.last.take(){
            self.motifs.push(last);
        }

        if self.motifs.len() > 0 {
            self.dumper.dump(mem::take(&mut self.motifs)).expect("dumper");
        }
    }

    #[cfg(test)]
    pub fn motifs(&self) -> &Vec<(TensorSliceIdx<K>, TensorSliceIdx<K>, f32)> {
        &self.motifs
    }
}
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct MotifTracker {
    motifs: Vec<(TensorSliceIdx<String>, TensorSliceIdx<String>, TensorSliceIdx<String>, f32)>, //i_0, match_0, match_1, dot
    motif_start_op: Option<TensorSliceIdx<String>>, //tensor, slice
    min_dot: f32,
    i_0: TensorSliceIdx<String>,
    file: File,
    last_dot: f32,
}
#[allow(dead_code)]
impl MotifTracker {
    pub fn new(min_dot: f32, path: &str) -> Self {
        match std::fs::remove_file(path) {
            Ok(_) => {}
            Err(_) => {}
        }

        let file = OpenOptions::new()
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
            last_dot: 0.,
        }
    }

    pub fn track(&mut self, i_0: TensorSliceIdx<String>, i: TensorSliceIdx<String>, dot: f32) {
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
            writeln!(self.file, "{}, {}, {},  {:.3}", motif.0, motif.1, motif.2, motif.3).unwrap();
        }
        self.file.flush().unwrap();
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

fn clc_motifs_matches_for_label<L, K: Clone + Hash + Eq + From<String>+ Send + Sync + Debug + Display  + 'static>(
    motifs: Arc<RwLock<SyncWrapper<Box<Tensor>>>>,
    all_zero: SyncWrapper<Box<Tensor>>,
    channels_motifs_matches_labels: Arc<Mutex<Vec<Vec<(Vec<f64>, Vec<L>)>>>>, //output TODO return it
    resources: Vec<K>,
    executor: Arc<RwLock<LoadRunner<K, Box<Tensor>>>>,
    label: L,
    step: usize,
    motifs_distances: Arc<RwLock<HashMap<usize, f32>>>,
    channel_motif_ids: Arc<RwLock<Vec<Vec<usize>>>>,
) -> Result<(), ThoosaError>
where L: Clone + Send + Display + Debug + 'static,
{
    let count = AtomicU64::new(0);

    let channels_num = motifs.read().size()[0] as usize;//TODO read
    let motifs_num = motifs.read().size()[1] as usize;

    debug!("clc_motifs_matches_for_label: label={label} channels_num={} motifs_num={} motifs.read().size()={:?} step={step}", channels_num, motifs_num, motifs.read().size());



    let wg = WaitGroup::new();
    let wg_in_loader = wg.clone();

    executor.write().exec(move |loader, _keys_purge| {
        async move {

            for resource_id in resources {

                debug!("clc_motifs_matches_for_label: resource_id={resource_id}");

                // let matches = clc_dot_matches(&eeg?, &motifs, step, threshold)?;
                let matches = clc_dist_matches(
                    &*Loader::<K, Box<Tensor>>::load_future(Arc::clone(&loader), resource_id, 0).await,
                    Arc::clone(&motifs), step, Arc::clone(&motifs_distances), Arc::clone(&channel_motif_ids)
                ).unwrap();

                for motif_i in 0..motifs_num {
                    for channel_i in 0..channels_num {
                        let motif_t = motifs.read().i((channel_i as i64, motif_i as i64));

                        let num_matches_label = if motif_t.equal(&all_zero) {//TODO this is to handle zero motifs which were added to pad up to max motifs num per channel
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

    #[arg(long, value_parser = parse_key_val::< String, String >, default_value = "xxx=yyy")]
    labels_datasets: Vec<(String, String)>,

    #[arg(long)]
    train_resource_group: Option<String>, //TODO enum with labels_datasets?

    #[arg(long, default_value_t = 0.5)]
    min_ratio: f32,
}

impl SelectMotifsParams {

    pub fn find_motifs_run_ids(&self) -> &Vec<usize> {
        &self.find_motifs_run_ids
    }

    pub fn dummy(run_ids: Vec<usize>, labels_datasets: Vec<(String, String)>, min_ratio: f32) -> Self {
        SelectMotifsParams {
            find_motifs_run_ids: run_ids,
            train_resource_group: None,
            labels_datasets,
            min_ratio,
        }
    }
    pub fn labels_datasets(&self) -> &Vec<(String, String)> {
        &self.labels_datasets
    }

    pub fn with_labels_datasets(&self, labels_datasets: Vec<(String, String)>) -> Self {
        let mut clone = self.clone();
        clone.labels_datasets = labels_datasets;

        clone
    }

    pub fn min_ratio(&self) -> f32 {
        self.min_ratio
    }

    pub fn train_resource_group(&self) -> Option<&String> {
        self.train_resource_group.as_ref()
    }
}

pub fn select_motifs_by_ratio<K: Clone + Hash + Eq + From<String>+ Send + Sync + Debug + Display  + 'static>(motifs: Tensor, labels_resources: Vec<(String, Vec<K>)>, executor: &Arc<RwLock<LoadRunner<K, Box<Tensor>>>>, step: usize, device: Device, min_ratio: f32, motifs_distances: Arc<RwLock<HashMap<usize, f32>>>, channel_motif_ids: Arc<RwLock<Vec<Vec<usize>>>>) -> Result<HashMap<String, Vec<Vec<(i64, f64)>>>, ThoosaError> {

    let mut labels_channels_good_motifs = labels_resources.iter().fold(HashMap::new(), |mut a, v| {
        a.insert(v.0.clone(), (0..motifs.size()[0]).map(|_| vec![]).collect::<Vec<Vec<(i64, f64)>>>()); //motifs.size()[0] - channels_num
        a
    });

    let (channels_num, motifs_num, channels_motifs_counts_labels_arc) = clc_channels_motifs_labels(
        Arc::new(RwLock::new(SyncWrapper::new(Box::new(motifs), |e| Box::new(e.shallow_clone())))),
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
}

pub fn select_motifs_by_corr<K: Clone + Hash + Eq + From<String>+ Send + Sync + Debug + Display  + 'static>(
    motifs: Tensor,
    labels_loaders: Vec<(i64, Vec<K>)>,
    executor: &Arc<RwLock<LoadRunner<K, Box<Tensor>>>>,
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

fn clc_channels_motifs_labels<L, K: Clone + Hash + Eq + From<String>+ Send + Sync + Debug + Display  + 'static>(
    motifs: Arc<RwLock<SyncWrapper<Box<Tensor>>>>,
    labels_resources: Vec<(L, Vec<K>)>,
    executor: &Arc<RwLock<LoadRunner<K, Box<Tensor>>>>,
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

    //TODO par_bridge
    labels_resources.into_iter()/*.par_bridge()*/.try_for_each(|(label, resources)| {
        clc_motifs_matches_for_label(
            Arc::clone(&motifs),
            SyncWrapper::new(Box::new(Tensor::zeros(slice_len, (tch::Kind::Float, *device))), |e| Box::new(e.shallow_clone())), //TODO
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
    use std::{fs, mem};

    use std::collections::{HashMap};
    use std::fmt::{Debug, Display};
    use std::ops::Deref;
    use std::path::Path;
    use std::sync::{Arc};
    use parking_lot::{Mutex, RawMutex};


    use log::{debug, trace};
    use parking_lot::RwLock;
    use rayon::iter::ParallelBridge;
    use rayon::iter::ParallelIterator;
    use tch::{Device, IndexOp, Tensor};
    use tch::Device::Cpu;
    use tch::Kind::{Float, Int};

    use thoosa_common::utils::*;
    use thoosa_common::utils::cdt_for_dir;
    use thoosa_shared::*;

    use thoosa_shared::val::Val;
    use thoosa_store::{SqlStore, Store};
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::runs::{Params, RunConfig, Simple};
    use thoosa_common::ws::{DummyWorkspace, FsWorkspace};
    use thoosa_shared::resource::ResourceId;
    use crate::find::{Conv1Params, /*FileDumper, */FindMotifsRunConfig, FindMotifsParams, InputFilter, select_motifs_by_ratio, MotifDumper, find_motifs_by_euc_dist, CommonFindMotifsConfig, ChunkOfSlicedTensors, collect_dists};

    use crate::find::select_motifs_by_corr;


    #[ctor::ctor]
    fn init() {
        init_test_logging();
    }

    #[test]
    fn test_filt() {
        //20 x 200_000 in files 5000 files 20_000_000_000
        // let dots = [1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1];
        let dots = [0.5, 0.9, 0.9, 0.5, 0.5, 0.9, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5];

        let mut mt = super::MotifTracker::new(0.8, "/tmp/test_motifs.csv");

        for (i, dot) in dots.into_iter().enumerate() {
            mt.track(TensorSliceIdx(0.to_string(), 0), TensorSliceIdx(0.to_string(), i as i64), dot as f32);
        }
        assert_eq!(
            [
                (TensorSliceIdx(0.to_string(), 0), TensorSliceIdx(0.to_string(), 1), TensorSliceIdx(0.to_string(), 3), 0.9f32),
                (TensorSliceIdx(0.to_string(), 0), TensorSliceIdx(0.to_string(), 5), TensorSliceIdx(0.to_string(), 8), 0.9f32)
            ],
            mt.motifs()[0..2]
        );
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
            move |key: &String| {
                debug!("LoadRunner: test_dir={test_dir:?} key={key}");
                let tensor = Tensor::load(test_dir.join(key)).expect(format!("can't load {:?}", key).as_str());
                debug!("LoadRunner: tensor={tensor}");

                SyncWrapper::new(
                    Box::new(tensor),
                    |e| Box::new(e.shallow_clone())
                )
            }, 4, 100,
        )));
        let motifs_avg_distances_arc = Arc::new(RwLock::new(motifs_avg_distances));
        let channel_motif_ids_arc = Arc::new(RwLock::new(channel_motif_ids));

        let channels_good_motifs = select_motifs_by_corr(
            motifs_tensor.shallow_clone(),
            vec![
                (1, vec!["t_0_0.pt".to_string(), "t_0_1.pt".to_string()]),
                (0, vec!["t_1_0.pt".to_string(), "t_1_1.pt".to_string()]),
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
                ("xxx".into(), vec!["t_0_0.pt".to_string(), "t_0_1.pt".to_string()]),
                ("yyy".into(), vec!["t_1_0.pt".to_string(), "t_1_1.pt".to_string()]),
                ("zzz".into(), vec!["t_2_0.pt".to_string(), "t_2_1.pt".to_string()]),
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
        // let store_path = format!("/tmp/test_store_{}.db3", cdt_for_dir());
        // let store = Arc::new(Mutex::new(SqlStore::from_path(store_path)?));
        let store = Arc::new(Mutex::new(SqlStore::in_mem()?));
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = Arc::new(RwLock::new(DummyWorkspace::new(1)?));

        let pulse = pulse_v(50., 10., 100, 20.);
        let n = 1000i64;

        let sig_0 = join_with_offset::<f32>(0., 10.0, &[(100, pulse.clone())], random_saw(1000));
        let sig_1 = join_with_offset::<f32>(0., 10.0, &[(500, pulse.clone())], random_saw(1000));
        let sig_2 = join_with_offset::<f32>(0., 10.0, &[(800, pulse.clone())], random_saw(1000));


        let x = lin_t(n);
        plot_series(
             vec![
                 (format!("Sig0"), plotters::prelude::RED, vec_f32_from(&x).into_iter().zip(sig_0.clone()).collect::<Vec<(f32, f32)>>()),
                 (format!("Sig1"), plotters::prelude::GREEN, vec_f32_from(&x).into_iter().zip(sig_1.clone()).collect::<Vec<(f32, f32)>>()),
                 (format!("Sig2"), plotters::prelude::BLACK, vec_f32_from(&x).into_iter().zip(sig_2.clone()).collect::<Vec<(f32, f32)>>()),

             ], "test_find_motifs_by_euc_dist", (1000, 500), &format!("/tmp/plots/0_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str()
        ).unwrap();


        let  executor_rc = Arc::new(RwLock::new(LoadRunner::new(
            move |key: &AsRef<ResourceId>| {
                let sig = match key.as_str() {
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


        let params = FindMotifsParams {
            step: 100.into(),
            window: 100.into(),
            min_dot: 0.99,
            resources: Resources::List(eeg_id_series),
            min_tracker_step: Some(3),
            max_keys_fifo: 3,
            chunk_size: 2,
            max_dist: 0.0001,
            input_filter: None,
        };

        let (common_config, run_configs) = to_test_run_config(&store, &params);

        let run_ids = find_motifs_by_euc_dist(
            run_configs,
            &Simple::new(params.resources),
            &common_config,
            &ws,
            &store,
            &executor_rc,
            vec![Cpu]
        )?;


        debug!("{:?}", store.lock().list_run_resources(1)?);
        let eeg_slices = store.lock().list_motifs_slices(run_ids[0])?;
        debug!("0: {:?}", eeg_slices.get(&"1111".to_string()));
        debug!("1: {:?}", eeg_slices.get(&"2222".to_string()));
        debug!("2: {:?}", eeg_slices.get(&"3333".to_string()));

        assert_eq!(&MotifSlice { motif_id: 1, slise_idx: 100, channel_idx: 0 }, eeg_slices.get(&"1111".to_string()).and_then(|v| v.get(0)).unwrap());
        assert_eq!(&MotifSlice { motif_id: 1, slise_idx: 500, channel_idx: 0 }, eeg_slices.get(&"2222".to_string()).and_then(|v| v.get(0)).unwrap());
        assert_eq!(&MotifSlice { motif_id: 1, slise_idx: 800, channel_idx: 0 }, eeg_slices.get(&"3333".to_string()).and_then(|v| v.get(0)).unwrap());

        Ok(())
    }

    fn to_test_run_config(store: &Arc<Mutex<Box<dyn Store>>>, params: &FindMotifsParams) -> (CommonFindMotifsConfig, Vec<FindMotifsRunConfig>) {
        let common_config = params.common_config();
        let mut run_configs = params.to_run_configs();
        run_configs.iter_mut().for_each(|c| {
            let id = store.lock().save_tree_run(None, "test", "test", HashMap::new()).unwrap();
            c.set_id(id);
        });
        (common_config, run_configs)
    }

    #[test]
    fn test_find_motifs_by_euc_dist_with_smooth_kernel() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = Arc::new(RwLock::new(DummyWorkspace::new(1)?));

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
            move |key: &Arc<String>| {
                let sig = match key.as_str() {
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


        let params = FindMotifsParams {
            step: 100.into(),
            window: 100.into(),
            min_dot: 0.99,
            resources: Resources::List(eeg_id_series),
            min_tracker_step: Some(3),
            max_keys_fifo: 3,
            chunk_size: 2,
            max_dist: 0.0001,
            input_filter: Some(InputFilter::Conv1(Conv1Params::new(10, 1, 1))),
        };

        let (common_config, run_configs) = to_test_run_config(&store, &params);

        let run_ids = find_motifs_by_euc_dist(
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
        let eeg_slices = store.lock().list_motifs_slices(run_ids[0])?;
        debug!("0: {:?}", eeg_slices.get(&"1111".to_string()));
        debug!("1: {:?}", eeg_slices.get(&"2222".to_string()));
        debug!("2: {:?}", eeg_slices.get(&"3333".to_string()));

        assert_eq!(&MotifSlice { motif_id: 1, slise_idx: 100, channel_idx: 0 }, eeg_slices.get(&"1111".to_string()).and_then(|v| v.get(0)).unwrap());
        assert_eq!(&MotifSlice { motif_id: 1, slise_idx: 500, channel_idx: 0 }, eeg_slices.get(&"2222".to_string()).and_then(|v| v.get(0)).unwrap());
        assert_eq!(&MotifSlice { motif_id: 1, slise_idx: 800, channel_idx: 0 }, eeg_slices.get(&"3333".to_string()).and_then(|v| v.get(0)).unwrap());

        Ok(())
    }


    #[test]
    fn test_find_motifs_by_euc_dist_diff_sizes() -> Result<(), ThoosaError> {
        let eeg_id_series:Vec<String> = vec!["0000".into(), "1111".into(), "2222".into()];
        // let store_path = format!("/tmp/test_store_{}.db3", cdt_for_dir());
        // let store = Arc::new(Mutex::new(SqlStore::from_path(store_path)?));
        let store = Arc::new(Mutex::new(SqlStore::in_mem()?));
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = Arc::new(RwLock::new(DummyWorkspace::new(1)?));

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
            move |key: &Arc<String>| {
                let sig = match key.as_str() {
                    "0000" => &sig_0,
                    "1111" => &sig_1,
                    "2222" => &sig_2,
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

        let params = FindMotifsParams {
            step: 50.into(),
            window: 100.into(),
            min_dot: 0.99,
            resources: Resources::List(eeg_id_series),
            min_tracker_step: Some(3),
            max_keys_fifo: 3,
            chunk_size: 2,
            max_dist: 0.0001,
            input_filter: None,
        };


        let (common_config, run_configs) = to_test_run_config(&store, &params);

        let run_ids = find_motifs_by_euc_dist(
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
        debug!("0: {:?}", eeg_slices.get(&"0000".to_string()));
        debug!("1: {:?}", eeg_slices.get(&"1111".to_string()));
        debug!("2: {:?}", eeg_slices.get(&"2222".to_string()));

        //TODO
        assert_eq!(&MotifSlice { motif_id: 1, slise_idx: 100, channel_idx: 0 }, eeg_slices.get(&"0000".to_string()).and_then(|v| v.get(0)).unwrap());
        assert_eq!(&MotifSlice { motif_id: 1, slise_idx: 600, channel_idx: 0 }, eeg_slices.get(&"1111".to_string()).and_then(|v| v.get(0)).unwrap());
        assert_eq!(&MotifSlice { motif_id: 1, slise_idx: 1000, channel_idx: 0 }, eeg_slices.get(&"2222".to_string()).and_then(|v| v.get(0)).unwrap());

        Ok(())
    }


    #[test]
    pub fn test_find_motifs_by_euc_dist_wrong_index() -> Result<(), ThoosaError> {
        // let store_path = format!("/tmp/test_store_{}.db3", cdt_for_dir());
        // let store = Arc::new(Mutex::new(SqlStore::from_path(store_path)?));
        let store = Arc::new(Mutex::new(SqlStore::in_mem()?));
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = Arc::new(RwLock::new(DummyWorkspace::new(1)?));

        let pulse = pulse_v(50., 10., 100, 20.);


        let tensor_sizes = [
            1000, 1000, 1000,
            500, 500, 500,
        ];

        let tensors = tensor_sizes.into_iter().map(|s| join_with_offset::<f32>(0., 10.0, &[(100, pulse.clone())], random_saw(s as usize))).collect::<Vec<Vec<f32>>>();

        let n = 684400i64;
        let tensors_len = tensors.len();

        let loader = Arc::new(RwLock::new(LoadRunner::new(
            move |key: &Arc<String>| {
                let sig = tensors.get(key.parse::<usize>().expect(format!("parse {}", key).as_str())).expect(format!("tensor {}", key).as_str());

                SyncWrapper::new(
                    Box::new(Tensor::vstack(&[
                        Tensor::from_slice(
                            sig.as_slice()
                        ), //ch 0
                    ]))
,
                    |e| Box::new(e.shallow_clone())
                )
            }, 4, 100,
        )));

        let params = FindMotifsParams {
            step: 100.into(),
            window: 100.into(),
            min_dot: 0.99,
            resources: Resources::List((0..tensors_len).map(|v| format!("{}", v)).collect()),
            min_tracker_step: None,
            max_keys_fifo: 1000,
            chunk_size: 3,
            max_dist: 0.001,
            input_filter: None,
        };

        let (common_config, run_configs) = to_test_run_config(&store, &params);

        let run_ids = find_motifs_by_euc_dist(
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

    #[test]
    pub fn test_simple_motif_tracker() -> Result<(), ThoosaError> {
       #[derive(Debug)]
        struct DummyDumper {
            motifs:  Vec<(TensorSliceIdx<String>, TensorSliceIdx<String>, f32)>
        }

        impl DummyDumper {
            pub fn new() -> Self {
                Self {
                    motifs:vec![]
                }
            }
        }

        impl <K:Eq+Display+Debug> MotifDumper<K> for DummyDumper {
            fn dump(&mut self, motifs: Vec<(TensorSliceIdx<K>, TensorSliceIdx<K>, f32)>) -> Result<(), ThoosaError>{

                self.motifs.extend(motifs.into_iter().map(|m|{
                    (
                        TensorSliceIdx("t_0".to_string(), m.0.1),
                        TensorSliceIdx("t_1".to_string(), m.1.1),
                        1.0
                    )
                }).collect::<Vec<(TensorSliceIdx<String>, TensorSliceIdx<String>, f32)>>());

                Ok(())
            }
        }

        let mut mt = super::SimpMotifTracker::new(DummyDumper::new(), None);

        mt.track(TensorSliceIdx("t_0".to_string(), 10), TensorSliceIdx("t_1".to_string(), 10), 1.0);
        mt.track(TensorSliceIdx("t_0".to_string(), 20), TensorSliceIdx("t_1".to_string(), 20), 1.0);
        mt.track(TensorSliceIdx("t_0".to_string(), 30), TensorSliceIdx("t_1".to_string(), 30), 1.0);

        mt.dump();
        assert_eq!(mt.dumper.motifs.len(), 3);
        assert_eq!((TensorSliceIdx("t_0".to_string(), 10), TensorSliceIdx("t_1".to_string(), 10), 1.0), mt.dumper.motifs[0]);
        assert_eq!((TensorSliceIdx("t_0".to_string(), 20), TensorSliceIdx("t_1".to_string(), 20), 1.0), mt.dumper.motifs[1]);
        assert_eq!((TensorSliceIdx("t_0".to_string(), 30), TensorSliceIdx("t_1".to_string(), 30), 1.0), mt.dumper.motifs[2]);


        let mut mt = super::SimpMotifTracker::new(DummyDumper::new(), Some(10));

        mt.track(TensorSliceIdx("t_0".to_string(), 10), TensorSliceIdx("t_1".to_string(), 10), 1.0);
        mt.track(TensorSliceIdx("t_0".to_string(), 20), TensorSliceIdx("t_1".to_string(), 20), 1.0);
        mt.track(TensorSliceIdx("t_0".to_string(), 31), TensorSliceIdx("t_1".to_string(), 30), 1.0);

        mt.dump();
        assert_eq!(mt.dumper.motifs.len(), 2);
        assert_eq!((TensorSliceIdx("t_0".to_string(), 20), TensorSliceIdx("t_1".to_string(), 20), 1.0), mt.dumper.motifs[0]);
        assert_eq!((TensorSliceIdx("t_0".to_string(), 31), TensorSliceIdx("t_1".to_string(), 30), 1.0), mt.dumper.motifs[1]);

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


        // let mut mt = super::SimpMotifTracker::new(FileDumper::new("/tmp/test_motifs.csv"), None);

        let dim = 1;


        let ts = TensorsSource::new(vec![t_0, t_1]);
        for i_0 in (0..n - m).step_by(step) {
            for i_1 in (i_0..n - m).step_by(step).skip(1) {
                debug!("i_0={:?}, i_1={:?} m={}", i_0, i_1, m);

                let (slice_01, slice_02) = ts.get_slices(i_0, i_1, m, dim)?;

                let dp = dot_norm(dim, &slice_01, &slice_02)?;

                // mt.track(TensorSliceIdx(&0.to_string(), i_0), TensorSliceIdx(&0.to_string(), i_1), dp.double_value(&[0]) as f32);
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

        // let motifs: Vec<&(TensorSliceIdx<String>, TensorSliceIdx<String>, f32)> = mt.motifs().iter().filter(|v| v.2 > 0.9).collect();
        //
        // assert_eq!((TensorSliceIdx(0.to_string(), 100), TensorSliceIdx(0.to_string(), 600)), (motifs[0].0.clone(), motifs[0].1.clone()));
        // assert_eq!((TensorSliceIdx(0.to_string(), 200), TensorSliceIdx(0.to_string(), 700)), (motifs[1].0.clone(), motifs[1].1.clone()));


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
        let params = FindMotifsRunConfig::new(10, 10, 1.0, "test", Some(InputFilter::Conv1(Conv1Params::new(10, 1, 1))));

        let properties: HashMap<String, Val> = params.to_properties_map();

        debug!("properties={:?}", properties);

        let sample_properties: HashMap<String, Val> = HashMap::from([
            ("FilterAndSlicerParams.window".into(), 10.into()),
            ("FilterAndSlicerParams.step".into(), 10.into()),
            ("FilterAndSlicerParams.max_dist".into(), 1.0.into()),
            ("FilterAndSlicerParams.descr".into(), "test".into()),
            ("FilterAndSlicerParams.filter.conv_1.Conv1Params.padding".into(), 1.into()),
            ("FilterAndSlicerParams.filter.conv_1.Conv1Params.stride".into(), 1.into()),
            ("FilterAndSlicerParams.filter.conv_1.Conv1Params.kernel".into(), 10.into()),
        ]);

        assert_eq!(sample_properties, properties);

        let loaded_params = crate::find::FindMotifsRunConfig::from_properties_map(properties);

        debug!("loaded_params={:?}", loaded_params);

        assert_eq!(params, loaded_params);
    }

    #[test]
    fn test_chunk_and_slice() -> Result<(), ThoosaError> {

        let resources: Vec<String> = vec!["111", "222", "333", "444", "555"].into_iter().map(|v| v.into()).collect();
        let resources_to_move = resources.clone();

        let chunks_0_test: Arc<Mutex<Vec<Tensor>>> = Arc::new(Mutex::new(vec![]));
        let chunks_0_test_rc = Arc::clone(&chunks_0_test);

        let chunks_1_test: Arc<Mutex<Vec<Tensor>>> = Arc::new(Mutex::new(vec![]));
        let chunks_1_test_rc = Arc::clone(&chunks_1_test);

        let executor = LoadRunner::new(
            move |key| {
                let i = resources_to_move.iter().position(|v| v.eq(key)).expect(format!("no {key}").as_str()) + 1;
                let v = (0..10).map(|v| i as f32 + v as f32 / 10.).collect::<Vec<f32>>();
                debug!("executor: key={key:?} i={i} v={v:?}");

                SyncWrapper::new_tb(Box::new(Tensor::from_slice(v.as_slice()).view((1, 10))))
            }, 4, 100,
        );
        let executor_rc = Arc::new(RwLock::new(executor));

        crate::find::chunk_and_slice(
            resources,
            2,
            &vec![FindMotifsRunConfig::new(3, 5, 1.0, "", None)],
            vec![vec![]],
             move |chunk_0, chunk_1, _size, _find_motifs_run_config, _channels_mt| {
                debug!("chunk_0={chunk_0:?} chunk_1={chunk_1:?}");

                trace!("callback: chunk_0.t=\n{}", chunk_0.t.i((.., 0, ..)).squeeze());
                trace!("callback: chunk_1.t=\n{}", chunk_1.t.i((.., 0, ..)).squeeze());

                chunks_0_test_rc.lock().push(chunk_0.t.i((.., 0, ..)).squeeze());
                chunks_1_test_rc.lock().push(chunk_1.t.i((.., 0, ..)).squeeze());
                Ok(())
            },
            &executor_rc,
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

    /**


    //ChunkOfSicedTensors is a structure holding information about stacked Tensors
    struct ChunkOfSicedTensors<'a> {
        keys: &'a Vec<String>, //Tensor's keys (each tensor is identified by a String key
        sizes: Vec<i64>,  //Tensors sizes -
        t: Tensor, //
    }


    //TensorSliceIdx is a structure to identify a slice in a tensor by tensor's key and slice index
    pub struct TensorSliceIdx<T>(pub T, pub i64) where T:Eq+Display+Debug;
    */

    #[test]
    fn test_collect_distances(){
        //x is made by stacking slices of tensors "aaa" and "bbb"
        let aaa_bbb_t = [
            10.9, 11.8, //slice 0 of tensor "aaa"
            12.7, 13.6, //slice 1 of tensor "aaa"
            14.5, 15.4, //slice 2 of tensor "aaa"
            16.3, 17.2, //slice 0 of tensor "bbb"
            18.1, 19.0, //slice 1 of tensor "bbb"
        ];

        let ccc_ddd_t = [
            20.9, 21.8, //slice 0 of tensor "ccc"
            24.5, 25.4, //slice 2 of tensor "ddd"
            26.3, 27.2, //slice 0 of tensor "ddd"
        ];


        let slices_chunk_0 = Tensor::from_slice(&aaa_bbb_t).view((5, 2));
        let slices_chunk_1 = Tensor::from_slice(&ccc_ddd_t).view((3, 2));

        let dist_t = super::dist_d2(&slices_chunk_0, &slices_chunk_1).unwrap();
        // let dist_t =  Tensor::cdist(&slices_chunk_0, &slices_chunk_1, 2.0, 0);

        let mut distance_pairs = Vec::new();
        let chunk_0 = ChunkOfSlicedTensors::new(
            vec!["aaa".to_string(), "bbb".to_string()],
            vec![3, 2],
            slices_chunk_0
        );
        let chunk_1 = ChunkOfSlicedTensors::new(
            vec!["ccc".to_string(), "ddd".to_string()],
            vec![1, 2],
            slices_chunk_1
        );
        collect_dists(
            &chunk_0,
            &chunk_1,
            &dist_t,
            1,
            |pair|{
                distance_pairs.push(pair);
            },
            |dist|{
                true
            }
        );

        println!("distance_pairs={distance_pairs:?}");

        assert_eq!(aaa_bbb_t.len() / 2 * ccc_ddd_t.len() / 2,  distance_pairs.len());
        assert_eq!(
           distance_pairs[0],
           (TensorSliceIdx("aaa".to_string(), 0), TensorSliceIdx("ccc".to_string(), 0), test_dist(&aaa_bbb_t[0..2], &ccc_ddd_t[0..2])),// dist between slices 0 and 1 of tensor "aaa" - [0.9000, 1.8000], [2.7000, 3.6000] is 2.5456
        );

        assert_eq!(
           distance_pairs[1],
           (TensorSliceIdx("aaa".to_string(), 0), TensorSliceIdx("ddd".to_string(), 0), test_dist(&aaa_bbb_t[0..2], &ccc_ddd_t[2..4])),// dist between slices 0 and 1 of tensor "aaa" - [0.9000, 1.8000], [2.7000, 3.6000] is 2.5456
        );

        assert_eq!(
           distance_pairs[2],
           (TensorSliceIdx("aaa".to_string(), 0), TensorSliceIdx("ddd".to_string(), 1), test_dist(&aaa_bbb_t[0..2], &ccc_ddd_t[4..6])),// dist between slices 0 and 1 of tensor "aaa" - [0.9000, 1.8000], [2.7000, 3.6000] is 2.5456
        );

        assert_eq!(
           distance_pairs[3],
           (TensorSliceIdx("aaa".to_string(), 1), TensorSliceIdx("ccc".to_string(), 0), test_dist(&aaa_bbb_t[2..4], &ccc_ddd_t[0..2])),// dist between slices 0 and 1 of tensor "aaa" - [0.9000, 1.8000], [2.7000, 3.6000] is 2.5456
        );


    }

    fn test_dist(x:&[f32], y:&[f32]) -> f32 {
        ((x[0]-y[0]) * (x[0]-y[0]) + (x[1]-y[1]) * (x[1]-y[1])).sqrt()
    }

}
