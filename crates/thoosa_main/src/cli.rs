use clap::{Args, Parser, Subcommand};
use std::sync::Arc;
use parking_lot::{Mutex};
use std::collections::HashMap;
use tch::{Device, Tensor};
use log::{debug, error, info, warn, LevelFilter};
use std::cmp::max;
use std::fmt::{Debug, Display};
use std::path::{Path, PathBuf};
use tch::nn::Module;
use std::fs;
use plotters::prelude::BLACK;
use simplelog::{ColorChoice, CombinedLogger, Config, TermLogger, TerminalMode, WriteLogger};
use std::fs::File;
use std::hash::Hash;
use std::str::FromStr;
use std::io::Write;
use futures_util::{AsyncReadExt, StreamExt};
use parking_lot::RwLock;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tch::Device::{Cpu, Cuda};
use lazy_static::lazy_static;
use thoosa_common::loader::{LoadRunner, SyncWrapper};
use thoosa_common::{train_nets, ResourceLoader};
use thoosa_common::parzen::optimizer::{ConfigurationSpace, Hyperparameter};
use thoosa_common::runs::{Params, ResourcesConfig, RunConfig, Simple};
use thoosa_common::train_nets::{infer, load_all_label_models_ws, load_motif_models_ws, train_final, train_label_nodes, FinalNodeParams, InferParams, LabelNodeParams, MotifNodeParams, SyncModule, TrainFinalParams, TrainLabelNodeParams, TrainMotifsNodesParams};
use thoosa_common::utils::{list_devices, plot_series};
use thoosa_motifs::find;
use thoosa_motifs::find::{select_motifs_by_ratio, FindMotifsRunConfig, FindMotifsParams, InputFilter, SelectMotifsParams, find_motifs_by_euc_dist, CommonFindMotifsConfig};
use thoosa_motifs::fragments::{check_slices, load_motif_slices, motifs_to_tensor, MotifsToTensorParams, PlotMotifSlicesParams};
use thoosa_shared::{batch_windows, clc_num_batches, make_combos, RangeOrValue, Resources, TensorSliceIdx, ThoosaError};
use thoosa_shared::val::{prefix_properties, unprefix_properties, Val};
use thoosa_store::{SqlStore, Store};
use crate::data::{clc_records, labeled_tensors_by_descr, load_to_logits_tensors, ParquetLoader};
use crate::data;
use crate::cli::Commands::{FindMotifs, Infer, InitResources, MotifsToTensor, PlotMotifSlices, RunAll, SayHello, SelectMotifs, TrainFinal, TrainOnLabel, TrainOnMotifs};
use thoosa_shared::parse_range_or_value;

use crate::version::VERSION;
use thoosa_common::ws::{DummyWorkspace, FsWorkspace, Workspace};

#[derive(Parser, Clone)]
#[command(version, about, long_about = None)]
pub struct Cli {//TODO pub
    #[arg(long)]
    pub base_path: String,//TODO remove

    #[arg(long)]
    pub resource_pattern: String,//TODO remove, add Import

    #[arg(long, default_value=".")]
    pub ws_path: String,

    #[arg(long)]
    pub tensors_path: Option<String>,

    #[arg(long, default_value="thoosa.db3")]
    pub store_path: String,

    #[arg(long, default_value = "INFO")]
    pub term_logger_level: String,

    #[arg(long, default_value = "INFO")]
    pub write_logger_level: String,

    #[arg(long, default_value = "thoosa.log")]
    pub logfile: String,

    #[arg(long)]
    pub limit: Option<usize>,

    #[arg(long)]
    pub num_threads: Option<usize>,

    #[command(subcommand)]
    pub command: Option<Commands>,

    #[arg(long)]
    pub input_filters_kernel: Option<i64>,//TODO all common filters configs
}



#[derive(Debug, Args, Clone)]
pub struct RunAllParams {

    #[arg(long)]
    train_resource_group: String,

    #[arg(long)]
    valid_resource_group: String,

    #[arg(long, default_value_t = false)]
    init:bool,

    #[arg(long, value_parser=parse_range_or_value::<u32>, default_value = "1000")]
    pub step: RangeOrValue<u32>,

    #[arg(long, value_parser=parse_range_or_value::<u32>, default_value = "1000")]
    pub window: RangeOrValue<u32>,

    #[arg(long, default_value = "/tmp")]
    models_dir: String,

    #[arg(long, default_value_t = 1000)]
    find_motifs_chunk_size: usize,

    #[command(subcommand)]
    find_motifs_input_filter: Option<InputFilter>,

    #[arg(long, default_value_t = 0.3)]
    find_motifs_max_dist_k: f32,

    #[arg(long, default_value_t = 0.3)]
    select_motifs_min_ratio: f32,

    #[arg(long, default_value_t = 100)]
    train_final_cycles:usize,

    #[arg(long,  default_value_t = 100)]
    motif_batch_size: i64,

}

impl RunAllParams {

    pub fn train_resource_group(&self) -> &str {
        &self.train_resource_group
    }

    pub fn init(&self) -> bool {
        self.init
    }

    fn find_motifs_params(&self, resource_set: String) -> FindMotifsParams {
        // FindMotifsParams::dummy(self.step as u32, self.window..self.window, resource_set, self.find_motifs_chunk_size as u32, self.find_motifs_max_dist_k, self.find_motifs_input_filter.as_ref())
        todo!()
    }
    // #[cfg(test)]
    pub fn dummy(train_resource_group: &str, valid_resource_group: &str) -> Self {
        Self {
            train_resource_group: train_resource_group.to_string(),
            valid_resource_group: valid_resource_group.to_string(),
            init: false,
            step: 1000.into(),
            window: 1000.into(),
            models_dir: "".to_string(),
            find_motifs_chunk_size: 1000,
            find_motifs_input_filter: None,
            find_motifs_max_dist_k: 0.3,
            select_motifs_min_ratio: 0.3,
            train_final_cycles: 1000,
            motif_batch_size: 100

        }
    }
    pub fn valid_resource_group(&self) -> &str {
        &self.valid_resource_group
    }
    pub fn models_dir(&self) -> &str {
        &self.models_dir
    }
}


impl Params<FindMotifsRunConfig, Simple, CommonFindMotifsConfig> for RunAllParams {
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
            Resources::Group(self.train_resource_group.clone())
        )
    }

    fn common_config(&self) -> CommonFindMotifsConfig {
        CommonFindMotifsConfig::new(Some(100), 100)//TODO
    }
}


#[derive(Debug, Args, Clone, Default)]
pub struct InitResourcesParams {

}

#[derive(Debug, Args, Clone)]
pub struct SayHelloParams {

    #[arg(long, value_parser=parse_range_or_value::<f64>, default_value = "1.0e-5")]
    pub lr: RangeOrValue<f64>,

    #[arg(long, value_parser=parse_range_or_value::<u32>, default_value = "1000")]
    pub step: RangeOrValue<u32>,

    #[arg(long, value_parser=parse_range_or_value::<u32>, default_value = "1000")]
    pub window: RangeOrValue<u32>
}

#[derive(Debug)]
pub struct SayHelloRunConfig {
    step: u32,
    window: u32,
    cs: Option<ConfigurationSpace>,
}

impl Params<SayHelloRunConfig, Simple, ()> for SayHelloParams {
    fn to_run_configs(&self) -> Vec<SayHelloRunConfig> {

        let tpe_config_space = [
            (&self.lr).try_into().ok(),
        ].into_iter()
            .collect::<Option<Vec<Hyperparameter>>>()
            .map(|v| ConfigurationSpace::new(v));

        make_combos(vec![
            self.window.clone().into_iter(),
            self.step.clone().into_iter()
        ]).into_iter().map(|config_combo|{
            SayHelloRunConfig {
                window: config_combo[0],
                step: config_combo[1],
                cs: tpe_config_space.as_ref().map(|v|v.clone())
            }
        }).collect()
    }

    fn resource_config(&self) -> Simple {
        Simple::new(Resources::List(vec![]))
    }

    fn common_config(&self) -> () {
        ()
    }
}

impl RunConfig for SayHelloRunConfig {
    fn config_space(&self) -> Option<&ConfigurationSpace> {
        self.cs.as_ref()
    }

    fn from_properties_map(properties: HashMap<String, Val>) -> Self {
        todo!()
    }

    // fn to_config_space(&self) -> Option<ConfigurationSpace> {
    //     [
    //         (&self.lr).try_into().ok(),
    //         (&self.num_layers).try_into().ok(),
    //     ].into_iter()
    //         .collect::<Option<Vec<Hyperparameter>>>()
    //         .map(|v|ConfigurationSpace::new(v))
    //
    //
    // }

}

#[derive(Debug, Subcommand, Clone)]
pub enum Commands {
    InitResources (InitResourcesParams),
    FindMotifs (FindMotifsParams),
    PlotMotifSlices(PlotMotifSlicesParams),
    MotifsToTensor(MotifsToTensorParams),
    SelectMotifs(SelectMotifsParams),
    TrainOnMotifs(TrainMotifsNodesParams),
    TrainOnLabel(TrainLabelNodeParams),
    TrainFinal(TrainFinalParams),
    RunAll(RunAllParams),
    Infer(InferParams),
    SayHello(SayHelloParams)
}

pub fn run_cli<F>(
    labeled_resourses:F,
    labels:Vec<&str>,
    channels: usize, max_input: usize
) -> Result<(), ThoosaError>
    where F:Fn(&dyn AsRef<Path>) -> Result<HashMap<String, Vec<(String, f32)>>, ThoosaError>
{

    let cli = Cli::parse();


    //TODO make optional
    let logfile = match File::create(&cli.logfile){
        Ok(file) => {
            file
        }
        Err(err) => {
            panic!("Can't crreate logfile {:?}", cli.logfile);
        }
    };
    CombinedLogger::init(
        vec![
            TermLogger::new(LevelFilter::from_str(cli.term_logger_level.as_str())?, Config::default(), TerminalMode::Mixed, ColorChoice::Auto),
            WriteLogger::new(LevelFilter::from_str(cli.write_logger_level.as_str())?, Config::default(), logfile),
        ]
    ).expect("failed to init logging");


    let ws = Arc::new(RwLock::new(FsWorkspace::new(
        cli.base_path.clone(),
        channels,
        max_input,
        cli.tensors_path.as_ref().map(|p|p.into()))?
    ));//TODO channels


    let store = Arc::new(Mutex::new(SqlStore::from_path(&cli.store_path)?));//TODO ws


    let devices  = list_devices();


    let cpu_num = std::thread::available_parallelism()?;
    let num_threads = cli.num_threads.unwrap_or_else(||max(2, cpu_num.get()/2));

    info!("VERSION: {VERSION} devices={devices:?}, base_path={:?} cpus={cpu_num} num_threads={num_threads} limit={:?}",  cli.base_path, cli.limit);
    std::io::stdout().flush()?;

    //TODO optional for benchmark
    rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global()?;
    let base_path = cli.base_path.clone();
    let resource_pattern = cli.resource_pattern.clone();

    //TODO import parquets into tensors
    let exec_loader = ParquetLoader::new(
        cli.base_path.clone(),
        cli.tensors_path.clone(),
        None,
        Box::new(move |_, resource_id| resource_path(&base_path, &resource_pattern, resource_id))
    );

    let loader = Arc::new(RwLock::new(LoadRunner::new(
        move |key: &Arc<String>| {

            SyncWrapper::new(
                exec_loader.load(key, Device::Cpu),
                |e| Box::new(e.shallow_clone().to(Cuda(0)))
            )
        }, 4, 100,
    )));


    match &cli.command {

        Some(InitResources(init_resources_params)) => {
            debug!("InitResources: init_resources_params={:?}", init_resources_params);
            init_resources(
                Arc::clone(&store),
                labels,
                ||labeled_resourses(&cli.base_path)
            )?;
        }

        Some(FindMotifs(find_motifs_params)) => {
            info!("find_motifs_params={:?} devices={:?}", find_motifs_params, devices);

            // let run_id = run_find_motifs_on_set(find_motifs_params, &ws, &store, &loader);
            //
            // info!("run_find_motifs_on_set done, run_id={:?}", run_id);


            let run_ids = run_step(
                None,
                find_motifs_params,
                find_motifs_by_euc_dist,
                &ws,
                &store,
                &loader,
                devices
            )?;

            info!("run_find_motifs_on_set done, run_id={:?}", run_ids);
        }

        Some(SelectMotifs(select_motifs_params)) => {
            info!("select_motifs_params={:?} devices={:?}", select_motifs_params, devices);

            let select_motifs_params = if let Some(train_resource_group) = select_motifs_params.train_resource_group() {
                let train_labels_datasets = { store.lock().list_group_labeled_resources_sets(train_resource_group.as_str())? };
                select_motifs_params.with_labels_datasets(train_labels_datasets)
            } else {
                select_motifs_params.clone()
            };

            let corr_ids = run_select_motifs(&ws, &store, devices, &select_motifs_params, &loader, channels as i64);

            info!("run_select_motifs done, corr_ids={:?}", corr_ids);
        }

        Some(TrainOnMotifs(train_on_motif_params)) => {
            debug!("train_on_motif: train_on_motif_params={:?}", train_on_motif_params);

            run_train_motif_nodes_on_motif_selection(&ws, &store, devices, train_on_motif_params, &loader)?;
        }

        Some(TrainOnLabel(train_on_label_params)) => {
            debug!("train_on_label_params={:?}", train_on_label_params);

            let train_on_label_params = if let Some(train_resource_group) = train_on_label_params.train_resource_group() {
                let train_labels_datasets = { store.lock().list_group_labeled_resources_sets(train_resource_group.as_str())? };
                let valid_labels_datasets = { store.lock().list_group_labeled_resources_sets(train_on_label_params.valid_resource_group().expect("valid_resource_group").as_str())? };
                train_on_label_params.with_labels_datasets(train_labels_datasets, valid_labels_datasets)

            } else {
                train_on_label_params.clone()
            };

            run_train_label_nodes(&ws, &store, devices, &train_on_label_params, &loader)?;
        }

        Some(TrainFinal(train_final_params)) => {

            debug!("train_final_params={:?}", train_final_params);

            run_train_final(&ws, &store, &devices[0], train_final_params, &loader)?;
        }

        Some(RunAll(run_all_params)) => {
            debug!("RunAll: run_all_params={:?}", run_all_params);

            if run_all_params.init {
                debug!("RunAll: init_resources");
                init_resources(
                    Arc::clone(&store),
                    labels,
                    ||labeled_resourses(&cli.base_path)
                )?;
            }

            //run_id - params set
            // let find_motifs_run_ids = run_find_motifs_on_group(&ws, &store, devices.clone(), run_all_params, &loader)?;

            let find_motifs_run_ids = run_step(
                None,
                run_all_params,
                find_motifs_by_euc_dist,
                &ws,
                &store,
                &loader,
                devices.clone()
            )?;


            info!("find_motifs done: find_motifs_run_ids={:?}", find_motifs_run_ids);

            let train_labels_datasets = {store.lock().list_group_labeled_resources_sets(run_all_params.train_resource_group())?};
            let valid_labels_datasets = {store.lock().list_group_labeled_resources_sets(run_all_params.valid_resource_group())?};

            info!("run all: train_labels_datasets={:?}", train_labels_datasets);
            info!("run all: valid_labels_datasets={:?}", valid_labels_datasets);

            let motif_selection_run_ids = run_select_motifs(
                &ws,
                &store,
                devices.clone(),
                &SelectMotifsParams::dummy(find_motifs_run_ids, train_labels_datasets.clone(), run_all_params.select_motifs_min_ratio),
                &loader,
                ws.read().channels() as i64
            )?;

            info!("select_motifs done: motif_selection_run_ids={:?}", motif_selection_run_ids);

            let motif_node_selection_run_id = run_train_motif_nodes_on_motif_selection(&ws, &store, devices.clone(), &TrainMotifsNodesParams::dummy(motif_selection_run_ids, 300, 0.2, 10, 1.0e-5, 1i32), &loader)?;

            info!("run_train_on_motifs done");

            run_train_label_nodes(
                &ws,
                &store,
                devices.clone(),
                &TrainLabelNodeParams::dummy(
                    *run_all_params.step.expect_value().unwrap() as usize,
                    motif_node_selection_run_id,
                    train_labels_datasets.clone(),
                    valid_labels_datasets.clone(),
                    run_all_params.motif_batch_size,
                    5,
                    100,
                    1.0e-5,
                ),
                &loader
            )?;

            info!("run_train_on_label done");

            run_train_final(
                &ws,
                &store,
                &devices[0],
                &TrainFinalParams::dummy(
                    *run_all_params.step.expect_value().unwrap() as usize,
                    *run_all_params.window.expect_value().unwrap() as usize, //FIXME not batch size
                    run_all_params.train_resource_group.clone(),
                    run_all_params.valid_resource_group.clone(),
                    run_all_params.train_final_cycles,
                    run_all_params.motif_batch_size,
                    5,
                ),
                &loader
            )?;

            info!("run_train_final done");
        }

        Some(Infer(infer_params)) => {

            let (labels,  motif_node_params) =  {
                let mut store_guard = store.lock();
                (
                    store_guard.list_labels().expect("store_guard"),
                    MotifNodeParams::from_properties_map(store_guard.list_run_params(777, None).expect("list_run_params"))
                )
            };

            let t = infer(
                &ws,
                &infer_params,
                |path:&Path|{
                    // data::parquet_to_tensor_by_path(<Path as AsRef<Path>>::as_ref(path), *devices.first().expect("first device"))
                    // parquet_to_tensor_by_path(path, *devices.first().expect("first device"))
                    // Box::new(move |_, resource_id| resource_path(&base_path, &resource_pattern, resource_id))
                    Ok(Tensor::from_slice(&[0.]))
                }
            //     infer_params.window() as i64,
            //     infer_params.step(),
            //     infer_params.motif_batch_size(),
            //     channels,
            //     max_input, //TODO
            //     &devices[0],
            //     labels,
            //     motif_node_params
            )?;

            data::save_tensor(
                &labels.iter().map(|v|v.as_str()).collect::<Vec<&str>>(),
                t.0,
                &t.1,
                infer_params.output()
            )

        }
        Some(PlotMotifSlices(plot_motif_slices_params)) => {
            plot_motif_slices(&cli, store, *devices.first().expect("first device"), &plot_motif_slices_params, &loader)?;
        }

        Some(MotifsToTensor(choose_fragments_params)) => {
            motifs_to_tensor_cmd(&cli, &store, *devices.first().expect("first device"), choose_fragments_params, &loader)?;
        }

        //gdb --args target/x86_64-unknown-linux-gnu/debug/thoosa_hbac --term-logger-level=DEBUG --base-path=/tmp/ --resource-pattern=xxx say-hello
        Some(SayHello(params)) => {
            debug!("params={:?}", params);
            let run_configs = params.to_run_configs();

            debug!("run_configs={:?}", run_configs);
        }

        None => {
            warn!("No command");
        }

    }

    Ok(())
}

pub fn run_train_final<K: Clone + Hash + Eq + From<String>+ Send + Sync + Debug + Display  + 'static>(ws: &Arc<RwLock<Box<dyn Workspace>>>, store: &Arc<Mutex<Box<dyn Store>>>, device: &Device, train_final_params: &TrainFinalParams, executor: &Arc<RwLock<LoadRunner<K, Box<Tensor>>>>) -> Result<(), ThoosaError> {

    let labels = {store.lock().list_labels()? };

    let (label_models, labels_channels_motifs_models) = load_all_label_and_motif_models(&ws, store, &device);

    let labels_len = labels_channels_motifs_models.len() as i64;
    let final_node_params = FinalNodeParams::new(labels_len, labels_len);

    let final_run_id = store.lock().save_tree_run(Some(label_models.values().flat_map(|v|v.iter().map(|w|w.0)).collect::<Vec<usize>>().as_slice()), "final", "".into(), prefix_properties("final_node", final_node_params.to_properties_map())).expect("save_tree_run");

    debug!("run_train_final: labels={:?}", labels);



    let vs = train_final(
        &ws,
        train_final_params,
        load_to_logits_tensors(
            train_final_params.train_resource_group(),
            &device,
            Arc::clone(&store),
        ),
        load_to_logits_tensors(
            train_final_params.valid_resource_group(),
            &device,
            Arc::clone(&store),

        ),
        train_final_params.step(),
        train_final_params.motif_batch_size(),
        &device,
        labels,
        executor,
        label_models,
        labels_channels_motifs_models,
        &final_node_params
    )?;


    ws.read().save_final_model(final_run_id, vs, final_node_params);

    Ok(())
}

fn load_all_label_and_motif_models(ws: &Arc<RwLock<Box<dyn Workspace>>>, store: &Arc<Mutex<Box<dyn Store>>>, device: &&Device) -> (
    HashMap<String, Vec<(usize, LabelNodeParams, SyncModule)>>,
    HashMap<String, Vec<Vec<(MotifNodeParams, SyncModule)>>>
) {
    let label_models = load_all_label_models_ws(&ws, &Cpu).expect("load_all_label_models_ws");

    let mut labels_channels_motifs_models: HashMap<String, Vec<Vec<(MotifNodeParams, SyncModule)>>> = HashMap::new();

    for (label, label_run_id_nodes) in label_models.iter() {
        for (label_run_id, label_node_params, node) in label_run_id_nodes {

            let motif_node_params_selection_run_ids = store.lock().get_input_run_ids(*label_run_id).expect("get_input_run_id(label_run_id)");
            assert_eq!(motif_node_params_selection_run_ids.len(), 1);

            let motif_selection_run_ids = store.lock().get_input_run_ids(motif_node_params_selection_run_ids[0]).expect("get_input_run_ids(motif_node_params_selection_run_ids");
            assert_eq!(motif_selection_run_ids.len(), 1);


            let channels_motifs_models = store.lock()
                .list_selected_motifs(motif_selection_run_ids[0]).expect("list_selected_motifs").into_iter().enumerate().flat_map(
                |(channel, channels_motifs)| {
                    let ws_moved = Arc::clone(&ws);
                    channels_motifs.into_iter().map(move |(motif_id, corr)| load_motif_models_ws(&ws_moved, channel, motif_id, &device).expect("load_motif_models"))
                }).collect();


            labels_channels_motifs_models.insert(label.clone(), channels_motifs_models);
        }
    }
    (label_models, labels_channels_motifs_models)
}

pub fn run_train_label_nodes<K: Clone + Hash + Eq + From<String>+ Send + Sync + Debug + Display  + 'static>(ws: &Arc<RwLock<Box<dyn Workspace>>>, store: &Arc<Mutex<Box<dyn Store>>>, devices: Vec<Device>, train_label_node_params: &TrainLabelNodeParams, executor: &Arc<RwLock<LoadRunner<K, Box<Tensor>>>>) -> Result<(), ThoosaError> {
    debug!("run_train_label_nodes: train_on_label_params={:?}", train_label_node_params);


    let valid_datasets_map = train_label_node_params.valid_labels_datasets().into_iter().fold(HashMap::new(), |mut a, v|{
        a.insert(v.0.clone(), v.1.clone());
        a
    });

    if valid_datasets_map.len() != train_label_node_params.train_labels_datasets().len() {
        error!("valid_datasets_map={:?}", valid_datasets_map);
        error!("train_label_node_params.train_labels_datasets()={:?}", train_label_node_params.train_labels_datasets());
        panic!("valid_datasets_map.len() != train_on_label_params.train_labels_datasets().len()");
    } //TODO check keys


    //each motif_selection_id trained with label datasets
    let label_datasets__motif_node_params_selection_run_ids = train_label_node_params.train_labels_datasets()
        .into_iter()
        .flat_map( |ds| train_label_node_params.motif_node_params_selection_run_ids().iter().map(|motif_node_params_selection_run_id|((ds.0.clone(), ds.1.clone()), *motif_node_params_selection_run_id)))
        .collect::<Vec<((String, String), usize)>>();

    let chunk_len = max(1, label_datasets__motif_node_params_selection_run_ids.len()/devices.len());

    let devices__label_datasets__motif_node_params_selection_run_ids_chunks = devices.into_iter().cycle().zip(label_datasets__motif_node_params_selection_run_ids.chunks(chunk_len).into_iter()).collect::<Vec<(Device, &[((String, String), usize)])>>();
    let store_in_threads =  Arc::clone(&store);


    devices__label_datasets__motif_node_params_selection_run_ids_chunks.into_par_iter().for_each(move |(device_i, devices__label_datasets__motif_node_params_selection_run_ids_chunk)| {

        for ((label, resource_set),  motif_node_params_selection_run_id) in devices__label_datasets__motif_node_params_selection_run_ids_chunk {
            let motif_selection_run_ids = store_in_threads.lock().get_input_run_ids(*motif_node_params_selection_run_id).expect("get_input_run_id(*motif_node_selection_run_id)"); //TODO format err
            assert_eq!(motif_selection_run_ids.len(), 1);

            let channels_motifs_models = store_in_threads.lock().list_selected_motifs(motif_selection_run_ids[0]).expect("list_selected_motifs").into_iter().enumerate().flat_map(

                |(channel, channels_motifs)| {
                    let ws_moved = Arc::clone(&ws);
                    channels_motifs.into_iter().map(move |(motif_id, corr)| load_motif_models_ws(&ws_moved, channel, motif_id, &device_i).expect("load_motif_models"))
                }).collect();

            let channels_motifs_models: Arc<RwLock<Vec<Vec<(MotifNodeParams, SyncModule)>>>> = Arc::new(RwLock::new(channels_motifs_models)); //channels_motifs_models.into_iter().map(|v|v.into_iter().map(|m|Mutex::new(m)).collect::<Vec<_>>()).collect::<Vec<_>>();


            let max_nets = channels_motifs_models.read().iter().map(|nets| nets.len()).max().unwrap_or(0) as i64;
            if max_nets <= 0 {
                panic!("no motif nets found");
            }

            let motif_node_params_run_ids = store_in_threads.lock().list_output_run_ids(*motif_node_params_selection_run_id).expect("list_output_run_ids")[0]; //TODO motif_node_params_run_ids len > 1

            let motif_node_params = {
                MotifNodeParams::from_properties_map(unprefix_properties("motifs_node", &store_in_threads.lock().list_run_params(motif_node_params_run_ids, None).expect("list_run_params")))
            };


            let max_motif_batches = clc_num_batches(ws.read().max_input(), motif_node_params.num_features() as usize, train_label_node_params.input_slice_step(), train_label_node_params.motif_batch_size() as usize);
            debug!("run_train_label_nodes: max_motif_batches={max_motif_batches}");

            let (label_node_run_ids_params) = {
                let mut store_guard = store.lock();

                (0..1).fold((vec![]), |mut a, _| { //TODO expand
                    let label_node_params = LabelNodeParams::new(ws.read().channels() as i64, max_nets, max_motif_batches as i64, train_label_node_params.motif_batch_size());

                    a.push(
                        (
                            store_guard.save_tree_run(Some(&[*motif_node_params_selection_run_id]), "todo", "".into(), prefix_properties("label_node", label_node_params.to_properties_map())).expect("save_tree_run"),
                            label_node_params
                        )
                    );

                    a
                })
            };

            info!("run_train_label_nodes: training {} on {} device={:?}", label, resource_set, device_i);
            for (label_node_run_id, label_node_params) in label_node_run_ids_params {
                match train_label_nodes(
                    ws,
                    &motif_node_params,
                    &label_node_params,
                    train_label_node_params,
                    labeled_tensors_by_descr(
                        label,
                        resource_set,
                        Arc::clone(&store),
                    ),
                    labeled_tensors_by_descr(
                        label,
                        valid_datasets_map.get(label).expect(format!("valid dataset for {}", label).as_str()),
                        Arc::clone(&store),
                    ),
                    &device_i,
                    &executor,
                    Arc::clone(&channels_motifs_models),
                ) {
                    Ok((loss, (vs, params))) => {
                        debug!("run_train_label_nodes: save_label_model label_node_run_id={label_node_run_id} label={label} motif_node_params_run_ids={motif_node_params_run_ids}");
                        ws.read().save_label_model(label_node_run_id, label, vs, params).expect("save_label_model failed");
                    }
                    Err(e) => {
                        //should be no motif nets found only
                        warn!("e:{:?}", e);
                    }
                }
            }
        }

    });
    Ok(())
}

/**
returns motif_node_params_selection_run_id
*/
fn run_train_motif_nodes_on_motif_selection<K: Clone + Hash + Eq + From<String>+ Send + Sync + Debug + Display  + 'static>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>, store: &Arc<Mutex<Box<dyn Store>>>, devices: Vec<Device>, train_on_motif_params: &TrainMotifsNodesParams,
    executor: &Arc<RwLock<LoadRunner<K, Box<Tensor>>>>) -> Result<Vec<usize>, ThoosaError> {
    let motif_selection_run_ids = &train_on_motif_params.motif_selection_run_id;


    let chunk_len = motif_selection_run_ids.len()/devices.len();

    debug!("run_train_motif_nodes_on_motif_selection: motif_selection_run_ids={motif_selection_run_ids:?} devices={devices:?}");

    let devices_motif_selection = devices
        .into_iter()
        .cycle()
        .zip(
            motif_selection_run_ids.chunks(chunk_len).into_iter()
        ).collect::<Vec<(Device, &[(usize)])>>();


    let run_ids = Arc::new(Mutex::new(vec![]));
    let run_ids_for_threads = Arc::clone(&run_ids);

    devices_motif_selection.into_par_iter().for_each( |(device_i, motif_selection_run_ids_chunk)| {

        for (motif_selection_run_id) in motif_selection_run_ids_chunk {
            let find_motifs_run_ids = store.lock().get_input_run_ids(*motif_selection_run_id).expect("motif_selection_id failed");
            assert_eq!(find_motifs_run_ids.len(), 1);
            debug!("run_train_motif_nodes_on_motif_selection: find_motifs_run_ids={find_motifs_run_ids:?} motif_selection_id={motif_selection_run_id}");

            let motif_node_params_selection_run_id = run_train_motif_nodes(
                ws,
                find_motifs_run_ids[0],
                *motif_selection_run_id,
                Arc::clone(store),
                device_i,
                train_on_motif_params,
                executor
            ).expect("train_on_motifs");

            run_ids_for_threads.lock().push(motif_node_params_selection_run_id);

            info!("TrainOnMotifs: motif_node_params_selection_run_id={motif_node_params_selection_run_id} done ");
        }


    });


    let guard = run_ids.lock();
    Ok(guard.to_vec())

}


pub fn run_select_motifs<K: Clone + Hash + Eq + From<String>+ Send + Sync + Debug + Display  + 'static>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Mutex<Box<dyn Store>>>,
    devices: Vec<Device>,
    select_motifs_params: &SelectMotifsParams,
    executor: &Arc<RwLock<LoadRunner<K, Box<Tensor>>>>, channels: i64
) -> Result<Vec<usize>, ThoosaError>{

    let chunk_len = select_motifs_params.find_motifs_run_ids().len()/devices.len();

    // let cli_clone = cli.clone();
    let motif_selection_run_ids_arc = Arc::new(Mutex::new(vec![]));
    let motif_selection_run_ids_arc_iter =  Arc::clone(&motif_selection_run_ids_arc);

    devices
        .into_iter()
        .cycle()
        .zip(select_motifs_params.find_motifs_run_ids().chunks(chunk_len).into_iter())
        .collect::<Vec<(Device, &[usize])>>()
        .into_par_iter()
        .for_each(move |(device_i, find_motifs_run_ids)| {

        for find_motifs_run_id in find_motifs_run_ids {
            let corr_ids = select_motifs_for_label(
                ws,
                Arc::clone(store),
                device_i,
                &select_motifs_params,
                *find_motifs_run_id,
                executor
            ).expect("select_motifs_in_class");

            motif_selection_run_ids_arc_iter.lock().extend(corr_ids);
        }
    });

    let motif_selection_run_ids = motif_selection_run_ids_arc.lock().clone();
    Ok(motif_selection_run_ids)
}

fn motifs_to_tensor_cmd<K: Clone + Hash + Eq + From<String>+ Send + Sync + Debug + Display  + 'static>(cli: &Cli, store: &Arc<Mutex<Box<dyn Store>>>, device: Device, choose_fragments_params: &MotifsToTensorParams, executor: &Arc<RwLock<LoadRunner<K, Box<Tensor>>>>) -> Result<(), ThoosaError> {
    let mut store_guard = store.lock();

    let t = motifs_to_tensor(
        store_guard.list_motifs_slices(choose_fragments_params.run_id())?,
        executor,
        choose_fragments_params.window(),
        20,
        device
    )?;

    let out = Path::new(&cli.base_path).join("out");
    fs::create_dir_all(&out)?;
    t.0.save(out.join(format!("motifs_in_{}", choose_fragments_params.run_id())))?;

    Ok(())
}


pub fn init_resources<F>(store:Arc<Mutex<Box<dyn Store>>>, labels:Vec<&str>, labeled_resourses:F) -> Result<(), ThoosaError>
    where F:Fn() -> Result<HashMap<String, Vec<(String, f32)>>, ThoosaError>
{
    let mut store_guard = store.lock();
    let labeled_resources = labeled_resourses()?;

    for label in labels {
        store_guard.save_label(label)?;//TODO merge with save_resources_labels
    }
    // store_guard.save_label("seizure")?;
    // store_guard.save_label("lpd")?;
    // store_guard.save_label("gpd")?;
    // store_guard.save_label("lrda")?;
    // store_guard.save_label("grda")?;
    // store_guard.save_label("other")?;

    store_guard.save_resources_labels(labeled_resources)?;

    store_guard.save_pure_sets(1000, true)?;
    store_guard.save_pure_sets(100, true)?;
    store_guard.save_pure_sets(10, true)?;
    store_guard.save_pure_sets(3, true)?;

    store_guard.save_groups_by_label_cnt(100)?;
    /*
    sqlite> select count(*), lb.label from (select count(*), label_id from labeled_resources group by resource_id having sum(p)=max(p)) as cnt left join labels lb on lb.id=cnt.label_id group by label_id;
    2227|seizure
    1181|lpd
    806|gpd
    384|lrda
    1154|grda
    4208|other
    */

    Ok(())
}

fn plot_motif_slices<K: Clone + Hash + Eq + From<String>+ Send + Sync + Debug + Display  + 'static>(cli: &Cli, store: Arc<Mutex<Box<dyn Store>>>, device: Device, plot_motif_slices_params: &PlotMotifSlicesParams, executor: &Arc<RwLock<LoadRunner<K, Box<Tensor>>>>) -> Result<(), ThoosaError> {
    let mut store_guard = store.lock();

    let (channel, _run_id, slices) = store_guard.list_motif_slices(plot_motif_slices_params.motif_id(), None, None)?;


    let slices = load_motif_slices(channel, plot_motif_slices_params.window as i64, slices.into_iter().map(|s|TensorSliceIdx(s.0.into(), s.1)).collect(), executor);//TODO Into for TensorSliceIdx

    let all_sices = slices.iter().flat_map(|(_id, slices)| slices.iter()).collect::<Vec<&Box<Tensor>>>();

    // check_slices(&all_sices, 0.9);

    let plots_dir = Path::new(cli.base_path.as_str()).join("plots");

    fs::create_dir_all(&plots_dir)?;

    plot_series(
        all_sices.into_iter().map(|t| {
            (format!("0"), BLACK, t.iter::<f64>().unwrap().enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>())
        }).collect(),
        format!("Motif {}", plot_motif_slices_params.motif_id()).as_str(),
        (2000, 1000),
        &plots_dir.join(format!("motif_{}.png", plot_motif_slices_params.motif_id()))
    ).unwrap();
    Ok(())
}

fn run_train_motif_nodes<K: Clone + Hash + Eq + From<String>+ Send + Sync + Debug + Display  + 'static>(ws: &Arc<RwLock<Box<dyn Workspace>>>, find_motifs_run_id: usize, motif_selection_run_id: usize, store:Arc<Mutex<Box<dyn Store>>>, device: Device, train_on_motif_params: &TrainMotifsNodesParams, executor: &Arc<RwLock<LoadRunner<K, Box<Tensor>>>>) -> Result<usize, ThoosaError> {
    debug!("train_on_motifs: device={device:?}, find_motifs_run_id={find_motifs_run_id} motif_selection_run_id={motif_selection_run_id:?}");

    let (motifs, params) = {
        let mut store_guard = store.lock();

        (
            store_guard.list_selected_motifs(motif_selection_run_id)?//TODO param
                .into_iter()
                .flat_map(|channel_motifs| channel_motifs.into_iter())
                .filter(|(_motif_id, corr)| *corr > train_on_motif_params.min_corr)
                .map(|(motif_id, _corr)| motif_id).collect::<Vec<usize>>(),

            FindMotifsRunConfig::from_properties_map(store_guard.list_run_params(find_motifs_run_id, None)?)
        )
    };

    debug!("train_on_motifs: motifs={motifs:?}");

    // let filter_and_slice:FilterAndSlicer = params.into();


    let (motif_node_params_run_ids, motif_node_params_selection_run_id) = {
        let mut store_guard = store.lock();

        let motif_node_selection_run_id =  store_guard
            //TODO motif_node_selection params - max error
            .save_tree_run(Some(&[motif_selection_run_id]), "motifs_node", "".into(), prefix_properties("motifs_node", HashMap::new()))
            .expect("motif_node_selection_run_id");

        ((0..1).fold((vec![]), |mut a, _| {//TODO expand grid search params
            let node_params = MotifNodeParams::new(params.window());

            a.push(
                (
                 store_guard
                     .save_tree_run(Some(&[motif_node_selection_run_id]), "todo", "".into(), prefix_properties("motifs_node", node_params.to_properties_map()))
                     .expect("save_tree_run"),
                 node_params
                )
            );

            a
        }), motif_node_selection_run_id)

    };


    for motif_id in motifs.iter() {
        let (all_sices_0, all_sices_1, channel) = {
            let mut store_guard = store.lock();

            let (channel, run_id, slices) = store_guard.list_motif_slices(*motif_id, None, None)?;

            assert_eq!(run_id, find_motifs_run_id);

            //TODO move load to DataBatch to avoid preload?
            debug!("train_on_motif: 1. channel={}, find_motifs_run_id={}, slices={}", channel, find_motifs_run_id, slices.len());

            let all_sices_1 = train_nets::slices_in_all_resources(
                load_motif_slices(channel, params.window(), slices.into_iter().map(|s|TensorSliceIdx(s.0.into(), s.1)).collect(), executor)//TODO slices
            );

            //list motifs which are NOT motif_id and NOT from run_id
            let (channel, run_id, slices) = store_guard.list_motif_slices(*motif_id, Some(all_sices_1.len()), Some((find_motifs_run_id, channel as usize)))?;

            debug!("train_on_motif: 0. channel={}, run_id={}, slices={}", channel, run_id, slices.len());

            let all_sices_0 = train_nets::slices_in_all_resources(
                load_motif_slices(channel, params.window(), slices.into_iter().map(|s|TensorSliceIdx(s.0.into(), s.1)).collect(), executor) //TODO slices
            );

            debug!("train_on_motif: 0. run_id={}, done", run_id);

            (all_sices_0, all_sices_1, channel)
        };

        assert!(all_sices_1.len() > 0, "all_sices_1.len(), motif={motif_id}");
        assert!(all_sices_0.len() > 0, "all_sices_0.len(), motif={motif_id}");

        let batch_size = train_on_motif_params.batch_size; //TODO, purge small motifs
        if all_sices_1.len() < batch_size {
            warn!("num of motif_id {motif_id} all_sices_1.len()={} in channel {channel} is less than {batch_size}", all_sices_1.len());
            continue;
        }

        if all_sices_0.len() < batch_size {
            warn!("num of not motif_id {motif_id} all_sices_0.len()={} in channel {channel} is less than {batch_size}", all_sices_0.len());
            continue;
        }

        debug!("train_on_motif:  all_sices_0.len()={:?}, all_sices_0[0].size()={:?}, all_sices_1.len()={:?}, all_sices_1[0].size()={:?}", all_sices_0.len(), all_sices_0[0].size(), all_sices_1.len(), all_sices_1[0].size());



        for (motif_node_params_run_id, node_params) in &motif_node_params_run_ids {
            debug!("train_on_motif: node_params={node_params:?}");

            let (loss,  (vs, params)) = train_nets::train_motif_nodes(
                *motif_id,
                all_sices_0.iter().map(|v|Box::new(v.shallow_clone())).collect(),//TODO change ClassTensorsDataBatch to accept refs
                all_sices_1.iter().map(|v|Box::new(v.shallow_clone())).collect(),
                train_on_motif_params,
                device,
                node_params,
                batch_size
            )?;

            debug!("train_on_motif: motif_id={motif_id} loss={loss}");

            if loss < train_on_motif_params.max_loss {
                //TODO save to run_id <<<<<<<<<<<<<<<<<<<<<<
                ws.read().save_motif_model(&channel, motif_node_params_run_id, motif_id, vs, params)?;

            }
        }
    }

    Ok(motif_node_params_selection_run_id)
}

/**
return selected_motif_run_ids
*/
fn select_motifs_for_label<K: Clone + Hash + Eq + From<String>+ Send + Sync + Debug + Display  + 'static>(ws: &Arc<RwLock<Box<dyn Workspace>>>, store:Arc<Mutex<Box<dyn Store>>>, device: Device, select_motifs_params: &SelectMotifsParams, find_motifs_run_id:usize, executor: &Arc<RwLock<LoadRunner<K, Box<Tensor>>>>) -> Result<Vec<usize>, ThoosaError> {
    debug!("select_motifs_for_label: device={:?}", device);

    let (resource_id_motif_slices, motifs_avg_distances, params) = {
        let mut store_guard = store.lock();

        (
            store_guard.list_motifs_slices(find_motifs_run_id)?,
            store_guard.list_motifs_distances(find_motifs_run_id)?,
            FindMotifsRunConfig::from_properties_map(store_guard.list_run_params(find_motifs_run_id, None)?)
        )
    };

    assert!(resource_id_motif_slices.len() > 0, "FAILED: no slices for run_id={}", find_motifs_run_id);


    let (motifs, channel_motif_ids) = motifs_to_tensor(
        resource_id_motif_slices,

        executor,
        params.window(),
        ws.read().channels() as i64,
        device
    ).unwrap();

    info!("select_motifs_for_label: motifs tensor built, size={:?}", motifs.size());


    let channel_motif_ids = Arc::new(RwLock::new(channel_motif_ids));
    let labels_channels_good_motif_idxs:HashMap<String, Vec<Vec<(i64, f64)>>> = select_motifs_by_ratio(
        motifs,
        select_motifs_params.labels_datasets().into_iter().map(|label_dataset|{
            (
                label_dataset.0.clone(),
                store.lock().list_resource_set_resources_by_descr(label_dataset.1.as_str()).unwrap().into_iter().map(|r|r.into()).collect::<Vec<K>>()
            )

        }).collect(),
        executor,
        params.step(),
        device,
        select_motifs_params.min_ratio(),
        Arc::new(RwLock::new(motifs_avg_distances)), //TODO keeps motif indexes with motif tensors
        Arc::clone(&channel_motif_ids)
    )?;

    Ok(labels_channels_good_motif_idxs.into_iter().map(|(label, channels_good_motif_idxs)| {

        let motifs_corrs = channel_motif_ids.read().iter()
            .zip(channels_good_motif_idxs.into_iter())
            .map(|(motif_ids, motif_idxs_corss)| {

                motif_idxs_corss.into_iter().map(|(motif_idx, motif_idx_corr)| {
                    (motif_ids[motif_idx as usize], motif_idx_corr)
                }).collect::<Vec<(usize, f64)>>()

            }).collect::<Vec<Vec<(usize, f64)>>>();

        debug!("select_motifs: motifs_corrs={:?}", motifs_corrs);

        store.lock().save_selected_motifs(find_motifs_run_id, format!("{:?}", select_motifs_params.labels_datasets()).as_str(), motifs_corrs).expect("save_motifs_corrs")
    }).collect::<Vec<usize>>())

}


fn resource_path(base_path: &str, resource_pattern: &str, resource_key: &str) -> String { //TODO change to impl AsRef<Path> + in ParquetLoader
    Path::new(base_path).join(resource_pattern.replace("{}", resource_key)).into_os_string().into_string().unwrap()
}

//P - params
// RC - RunConfig
//R  - Resources Config
//CC  - CommonConfig - same for all run configurations
pub fn run_step<'a, P, RC, R, CC, F, K>(
    input_run_ids:Option<Vec<usize>>,
    params:&P,
    step:F,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Mutex<Box<dyn Store>>>,
    exec: &Arc<RwLock<LoadRunner<K, Box<Tensor>>>>,
    devices:Vec<Device>
) -> Result<Vec<usize>, ThoosaError>
where
    F:FnOnce(Vec<RC>, &R, &CC, &Arc<RwLock<Box<dyn Workspace>>>, &Arc<Mutex<Box<dyn Store>>>, &Arc<RwLock<LoadRunner<K, Box<Tensor>>>>, Vec<Device>) -> Result<Vec<usize>, ThoosaError>,
    P:Params<RC, R, CC>,
    RC:RunConfig,
    R:ResourcesConfig,
    K: Clone + Hash + Eq + From<String>+ Send + Sync + Debug + Display  + 'static
{
    let run_configs = params.to_run_configs();
    let resource_config = params.resource_config();
    let resources_serialized = resource_config.to_string();

    let ran_ids = step(

        run_configs.into_iter().map(|mut run_config|{
            let run_config_id = store.lock().save_tree_run(
                input_run_ids.as_deref(),
                "zzz",//TODO
                &resources_serialized,
                run_config.to_properties_map()
            ).unwrap();

            run_config.set_id(run_config_id);

            run_config
        }).collect::<Vec<RC>>(),

        &resource_config,
        &params.common_config(),
        ws,
        store,
        exec,
        devices
    )?;

    Ok(ran_ids)
}



#[cfg(test)]
mod main_tests {
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::vec::IntoIter;
    use log::debug;
    use parking_lot::{Mutex, RawMutex, RwLock};
    use parking_lot::lock_api::MutexGuard;
    use plotters::prelude::{IntoLinspace, BLACK};
    use polars::export::arrow::array::clone;
    use polars::export::chrono;
    use rayon::iter::ParallelIterator;
    use rayon::prelude::IntoParallelIterator;
    use tch::{nn, Device, IndexOp, Tensor};
    use tch::Device::Cpu;
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::runs::{Params, ResourcesConfig};
    use thoosa_common::train_nets::{motif_node, MotifNodeParams, TrainFinalParams, TrainLabelNodeParams, TrainMotifsNodesParams};
    use thoosa_common::utils::{cdt_for_dir, from, lin_t, plot_series, vec_f32_from};
    use thoosa_motifs::find::{FindMotifsRunConfig, FindMotifsParams, LinearExpander, SelectMotifsParams, find_motifs_by_euc_dist};
    use thoosa_motifs::fragments::PlotMotifSlicesParams;
    use thoosa_shared::{recurse_files, saw, sig_one_imp, sig_two_imp, sig_three_imp, ThoosaError, Resources};
    use thoosa_shared::val::unprefix_properties;
    use thoosa_store::{SqlStore, Store};
    use crate::cli::{init_resources, plot_motif_slices, resource_path, run_step, run_train_final, run_train_label_nodes, run_train_motif_nodes_on_motif_selection, select_motifs_for_label, Cli, RunAllParams};
    use crate::cli::Commands::FindMotifs;
    use crate::data::ParquetLoader;
    use thoosa_common::ws::{FsWorkspace, Workspace};
    use thoosa_common::runs::RunConfig;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::init_test_logging();
    }


    #[test]
    fn test_resource_path() {
        assert_eq!(
            resource_path(
                "/xxx/yyy",
                "zzz/{}.parquet",
                "aaa"
            ),
            "/xxx/yyy/zzz/aaa.parquet".to_string()
        )
    }


    #[test]
    fn test_motifs_corrs() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());

        fs::create_dir_all(&test_dir)?;
        let base_path = test_dir.to_string_lossy().to_string();

        let n = 1000i64;

        let tensors_path = Some(test_dir.join("tensors"));
        let ws = Arc::new(RwLock::new(FsWorkspace::new(test_dir.clone(), 3, n as usize, tensors_path)?));
        let ws_rc = Arc::clone(&ws);
        let max_y = 30.0;
        let sig_0_v = sig_one_imp(n, max_y, None, None);
        let sig_1_v = sig_two_imp(n, max_y, None, None);
        let sig_2_v = sig_three_imp(n, max_y, None, None);


        let x = lin_t(n);
        // let sig_0_v_00 = sig_one_imp(n, max_y, Some(10. .. 100.), Some(10. .. 100.));
        // let sig_0_v_01 = sig_two_imp(n, max_y, Some(10. .. 100.), Some(10. .. 100.));
        // let sig_0_v_02 = sig_one_imp(n, max_y, Some(10. .. 100.), Some(10. .. 100.));

        plot_series(
            vec![
                (format!("Sig0"), plotters::prelude::RED, vec_f32_from(&x).into_iter().zip(sig_0_v.clone()).collect::<Vec<(f32, f32)>>()),
                (format!("Sig1"), plotters::prelude::GREEN, vec_f32_from(&x).into_iter().zip(sig_1_v.clone()).collect::<Vec<(f32, f32)>>()),
                (format!("Sig2"), plotters::prelude::BLACK, vec_f32_from(&x).into_iter().zip(sig_2_v.clone()).collect::<Vec<(f32, f32)>>()),

            ], "test_find_motifs_euc", (2000, 1000), &format!("/tmp/plots/0_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str()
        ).unwrap();


        /*
        let sig_0 = sig_0(n);

        let sig_1 = sig_1(n);

        let sig_2 = sig_2(n);

        */

        let t_0_0 = Tensor::vstack(&[
            Tensor::from_slice([sig_0_v.clone(), saw(n), saw(n)].concat().as_slice()),
            Tensor::from_slice([sig_2_v.clone(),          saw(n), saw(n)].concat().as_slice()),
            Tensor::from_slice([saw(n),          sig_2_v.clone(), saw(n)].concat().as_slice()),
        ]);
        t_0_0.save(ws.read().tensors_path().join(format!("t_0_0.pt")))?;

        // debug!("t_0_0 mean={} max-min={}", t_0_0.mean(None).double_value(&[]), t_0_0.max() - t_0_0.min());

        let t_0_1 = Tensor::vstack(&[
            Tensor::from_slice([saw(n), sig_0_v.clone(), saw(n) ].concat().as_slice()),
            Tensor::from_slice([saw(n),          sig_2_v.clone(), saw(n)].concat().as_slice()),
            Tensor::from_slice([sig_2_v.clone(),          saw(n), saw(n)].concat().as_slice()),
        ]);
        t_0_1.save(ws.read().tensors_path().join(format!("t_0_1.pt")))?;

        // debug!("t_0_1 mean={} max-min={}", t_0_1.mean(None).double_value(&[]), t_0_1.max() - t_0_1.min());


        let t_1_0 = Tensor::vstack(&[
            Tensor::from_slice([saw(n), sig_1_v.clone(), saw(n)].concat().as_slice()),
            Tensor::from_slice([saw(n),          saw(n), sig_2_v.clone()].concat().as_slice()),
            Tensor::from_slice([saw(n),          sig_2_v.clone(), saw(n)].concat().as_slice())
        ]);
        t_1_0.save(ws.read().tensors_path().join(format!("t_1_0.pt")))?;

        // debug!("t_1_0 mean={} max-min={}", t_1_0.mean(None).double_value(&[]), t_1_0.max() - t_1_0.min());

        let t_1_1 = Tensor::vstack(&[
            Tensor::from_slice([saw(n), saw(n), sig_1_v.clone()].concat().as_slice()),
            Tensor::from_slice([saw(n),          saw(n), sig_2_v.clone()].concat().as_slice()),
            Tensor::from_slice([sig_2_v.clone(),          sig_2_v.clone(), saw(n)].concat().as_slice()),
        ]);
        t_1_1.save(ws.read().tensors_path().join(format!("t_1_1.pt")))?;

        // debug!("t_1_1 mean={} max-min={}", t_1_1.mean(None).double_value(&[]), t_1_1.max() - t_1_1.min());
        let channels = 3;

        let loader = Arc::new(RwLock::new(LoadRunner::new(
            move |key: &Arc<String>| {

                SyncWrapper::new(
                    Box::new(Tensor::load(ws.read().tensors_path().join(format!("{key}.pt"))).unwrap()),
                    |e| Box::new(e.shallow_clone())
                )
            }, 4, 100,
        )));

        let store = Arc::new(Mutex::new(SqlStore::in_mem()?));
        // std::fs::remove_file("/tmp/test_main_db.db3").unwrap_or(());
        // let store = Arc::new(Mutex::new(SqlStore::from_path("/tmp/test_main_db.db3")?));

        init_resources(
            Arc::clone(&store),
            vec!["label_0", "label_1"],
            ||Ok(HashMap::from([
                ("t_0_0".to_string(), vec![("label_0".to_string(), 1.0)]),
                ("t_0_1".to_string(), vec![("label_0".to_string(), 1.0)]),
                ("t_1_0".to_string(), vec![("label_1".to_string(), 1.0)]),
                ("t_1_1".to_string(), vec![("label_1".to_string(), 1.0)])
            ]))
        )?;
        let run_ids = run_step(
            None,
            &FindMotifsParams {
                step: 100.into(),
                window: 800.into(),
                min_dot: 0.0,
                resources: Resources::List(vec!["t_0_0".into(), "t_0_1".into(), "t_1_0".into(), "t_1_1".into()]),
                min_tracker_step: Some(3),
                max_keys_fifo: 3,
                chunk_size: 2,
                max_dist: 0.5,
                input_filter: None,
            },
            find_motifs_by_euc_dist,
            &ws_rc,
            &store,
            &loader,
            vec![Cpu]
        )?;


        let slices = store.lock().list_motifs_slices(run_ids[0])?;
        debug!("test_motifs_with_tpe: slices={:?}", slices);
        assert!(slices.len() >= 4);


        // assert_eq!((110, 0), slices.get(&"t_0".to_string()).and_then(|v| v.get(0).map(|v|(v.slise_idx, v.channel_idx))).unwrap());
        // assert_eq!(&MotifSlice { motif_id: 1, slise_idx: 1100, channel_idx: 0 }, slices.get(&"t_1".to_string()).and_then(|v| v.get(0)).unwrap());

        let select_motifs_run_ids = select_motifs_for_label(
            &ws_rc,
            Arc::clone(&store),
            Cpu,
            &SelectMotifsParams::dummy(
                run_ids.clone(),
                vec![
                    ("label_0".to_string(), "1/label_0".to_string()),
                    ("label_1".to_string(), "1/label_1".to_string())
                ], 0.1
            ),
            run_ids[0],
            &loader
        ).expect("select_motifs_in_class");

        debug!("test_motifs_with_tpe: corr_ids={select_motifs_run_ids:?}");
        let motif_selection_id = store.lock().list_input_run_ids(&select_motifs_run_ids)?;
        debug!("test_motifs_with_tpe: motif_selection_id={motif_selection_id:?}");


        // store.lock().list_selected_motifs();

        //there are motifs with corr 1.0 for each label
        // assert!(store.lock().list_selected_motifs(
        //     corrs.iter().find(|(id, run_id)|label.eq("label_0")).expect("label_0").1
        //     //0 channel
        // )?[0].iter().find(|(motif_id, corr)|corr == &1.0).is_some());
        //
        // assert!(store.lock().list_selected_motifs(
        //     corrs.iter().find(|(id, run_id)|label.eq("label_1")).expect("label_1").1
        // )?[0].iter().find(|(motif_id, corr)|corr == &1.0).is_some());

        Ok(())

    }

    #[test]
    fn test_run_all() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_run_train_on_motifs: test_dir={test_dir:?}");
        let n = 1000i64;

        let tensors_path = Some(test_dir.join("tensors"));
        let ws = Arc::new(RwLock::new(FsWorkspace::new(test_dir, 1, 2 * n as usize, tensors_path)?));
        let ws_runner = Arc::clone(&ws);


        let max_y = 30.0;


        (0..10).for_each(|i|{
            label_0_t(n, max_y).save(ws.read().tensors_path().join(format!("t_0_{i}.pt"))).unwrap();
        });

        (0..10).for_each(|i|{
            label_1_t(n, max_y).save(ws.read().tensors_path().join(format!("t_1_{i}.pt"))).unwrap();
        });

        /*
        let sig_0_v = sig_0(n, max_y);
        let sig_1_v = sig_1(n, max_y);
        let sig_2_v = sig_2(n, max_y);
        let x = lin_t(n);
        plot_series(
            vec![
                (format!("Sig0"), plotters::prelude::RED, vec_f32_from(&x).into_iter().zip(sig_0_v.clone()).collect::<Vec<(f32, f32)>>()),
                (format!("Sig1"), plotters::prelude::GREEN, vec_f32_from(&x).into_iter().zip(sig_1_v.clone()).collect::<Vec<(f32, f32)>>()),
                // (format!("Sig2"), plotters::prelude::BLACK, vec_f32_from(&x).into_iter().zip(sig_2_v.clone()).collect::<Vec<(f32, f32)>>()),

            ], "test_find_motifs_euc", (2000, 1000), &format!("/tmp/plots/0_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str()
        )?;
        */

        let models_dir = ws.read().base_path().join("models").to_string_lossy().to_string();
        let cli = Cli {
            base_path: ws.read().base_path_as_str(),
            resource_pattern: "{}.parquet".to_string(),
            ws_path: models_dir.clone(),
            tensors_path: None,
            store_path: "".to_string(),
            term_logger_level: "".to_string(),
            write_logger_level: "".to_string(),
            logfile: "".to_string(),
            limit: None,
            num_threads: None,
            command: None, //Some(FindMotifs(FindMotifsParams::dummy(10, 10, "resource_set_1".into(), 10, 0.3, None))),
            input_filters_kernel: None,
        };

        // let store = Arc::new(Mutex::new(SqlStore::in_mem()?));
        // fs::remove_file("/tmp/test_main_db.db3").unwrap_or(());
        // let store = Arc::new(Mutex::new(SqlStore::from_path("/tmp/test_main_db.db3")?));
        let store = Arc::new(Mutex::new(SqlStore::in_mem()?));
        init_test_store(store.lock());

        ///////////////////////////

        let find_motifs_params = FindMotifsParams::dummy(100, 800..800, "test set".into(), 10, 1.0, None);
        let param_set = find_motifs_params.to_filter_and_slice_params(&LinearExpander::new(1));

        let find_motifs_run_id_0 = save_test_motifs(Arc::clone(&store), find_motifs_params.clone(), param_set.clone(), "t_0_")?;
        let find_motifs_run_id_1 = save_test_motifs(Arc::clone(&store), find_motifs_params, param_set, "t_1_")?;

        let motifs_slices:Vec<Vec<(usize, f64)>> = store.lock().list_motifs_slices(find_motifs_run_id_1)?.into_values().map(|v| v.into_iter().map(|s| (s.motif_id, 1.0f64)).collect()).collect();
        debug!("test_run_train_on_motifs: run_id_1={find_motifs_run_id_1} motifs_slices={motifs_slices:?}");
        //0 channel
        let motif_ids = motifs_slices[0].iter().map(|(motif, corr)| *motif).collect::<HashSet<usize>>();


        let motif_selection_run_id = store.lock().save_selected_motifs(find_motifs_run_id_1, "test", motifs_slices)?;

        debug!("test_run_train_on_motifs: motif_ids={motif_ids:?} motif_selection_run_id={motif_selection_run_id}");

        let executor_rc = Arc::new(RwLock::new(LoadRunner::new(
            move |key: &String| {
                let path = ws_runner.read().tensor_path(key);
                debug!("test_run_train_on_motifs: path={path:?}");
                SyncWrapper::new(
                    Box::new(Tensor::load(path).unwrap()),
                    |e| Box::new(e.shallow_clone())
                )
            }, 4, 100,
        )));

        // plot_motif_slices(&cli, Arc::clone(&store), Cpu, &PlotMotifSlicesParams::new(motif_ids.into_iter().next().unwrap(), 800), Arc::clone(&executor_rc));

        let motif_node_params_selection_run_ids = run_train_motif_nodes_on_motif_selection(
            &ws,
            &store,
            vec![Cpu],
            // &TrainOnMotifsParams::dummy(vec![corr_run_id], 100, 100.0, 10, 10e-5..10e-3, 1),
            // &TrainOnMotifsParams::dummy(vec![motif_selection_id], 10, 100.0, 10, 10e-9 .. 10e-5, 2..5),
            &TrainMotifsNodesParams::dummy(vec![motif_selection_run_id], 3, 100.0, 10, 10e-5, 2),
            &executor_rc,
        )?;

        debug!("test_run_train_on_motifs: motif_node_params_selection_run_ids={motif_node_params_selection_run_ids:?}");

        /*
        let models = ws.read().motif_models(1)?;
        debug!("test_run_train_on_motifs: train_run_ids={train_run_ids:?} models={models:?}");

        let motif_node_params = MotifNodeParams::from_file(&models[0][0].1)?; //from_properties_map(unprefix_properties("motifs_node", &store.lock().list_run_params(train_run_ids[0], None).expect("list_run_params")));

        let mut vs = nn::VarStore::new(Cpu);
        let node = RwLock::new(
            SyncWrapper::new(
            motif_node(&motif_node_params).build_tch_module(&vs.root())?
                , |v| todo!()
            )
        );//TODO window
        vs.load(&models[0][0].0)?;
        vs.freeze();


        let oks = (0..10).into_par_iter().map(|i| {
            let t_0 = label_0_t(n, max_y).i((.., 100..900));
            plot_t(&t_0, "t_0_{i}").unwrap();
            let label_0_p = node.read().forward(&t_0.view((1, 1, -1)));

            let t_1 = label_1_t(n, max_y).i((.., 100..900));
            plot_t(&t_1, "t_1_{i}").unwrap();
            let label_1_p = node.read().forward(&t_1.view((1, 1, -1)));

            debug!("test_run_train_on_motifs: label_0_p={label_0_p} label_1_p={label_1_p}");

            label_1_p.double_value(&[0]) > label_0_p.double_value(&[0])
        }).filter(|r|*r).count();

        debug!("test_run_train_on_motifs: oks={oks:?}");
        assert!(oks > 7);
         */


        /////////////////////LABEL
        //apply_channel_motifs - input 2000 - motif_width=800, step=100 batch_size=10
        //                     - two batches of slices- one full - [0, 100, 200, 300, 400, 500, 600, 700, 800, 900] + one reminder 1000, 1100, 1200
        //
        run_train_label_nodes(
            &ws,
            &store,
            vec![Cpu],
            &TrainLabelNodeParams::dummy(
                100,
                motif_node_params_selection_run_ids,
                vec![
                    ("Label_0".into(), "Label_0/6".into()),
                    ("Label_1".into(), "Label_1/6".into())
                ], vec![
                    ("Label_0".into(), "Label_0/4".into()),
                    ("Label_1".into(), "Label_1/4".into())
                ],
                10, 10, 3, 0.4
            ),
            &executor_rc,
        )?;

        /////////////////////FINAL
        run_train_final(
            &ws,
            &store,
            &Cpu,
            &TrainFinalParams::dummy(100, 3, "pure group 6".into(), "pure group 4".into(), 10, 10, 10),
            &executor_rc,
        )?;


        Ok(())
    }

    fn init_test_store(mut store_guard: MutexGuard<RawMutex, Box<dyn Store>>) -> Result<(), ThoosaError>{

        store_guard.save_label("Label_0")?;
        store_guard.save_label("Label_1")?;

        store_guard.save_resources_labels(HashMap::from([
            ("t_0_0".into(), vec![("Label_0".into(), 1.0)]),
            ("t_0_1".into(), vec![("Label_0".into(), 1.0)]),
            ("t_0_2".into(), vec![("Label_0".into(), 1.0)]),
            ("t_0_3".into(), vec![("Label_0".into(), 1.0)]),
            ("t_0_4".into(), vec![("Label_0".into(), 1.0)]),
            ("t_0_5".into(), vec![("Label_0".into(), 1.0)]),
            ("t_0_6".into(), vec![("Label_0".into(), 1.0)]),
            ("t_0_7".into(), vec![("Label_0".into(), 1.0)]),
            ("t_0_8".into(), vec![("Label_0".into(), 1.0)]),
            ("t_0_9".into(), vec![("Label_0".into(), 1.0)]),

            ("t_1_0".into(), vec![("Label_1".into(), 1.0)]),
            ("t_1_1".into(), vec![("Label_1".into(), 1.0)]),
            ("t_1_2".into(), vec![("Label_1".into(), 1.0)]),
            ("t_1_3".into(), vec![("Label_1".into(), 1.0)]),
            ("t_1_4".into(), vec![("Label_1".into(), 1.0)]),
            ("t_1_5".into(), vec![("Label_1".into(), 1.0)]),
            ("t_1_6".into(), vec![("Label_1".into(), 1.0)]),
            ("t_1_7".into(), vec![("Label_1".into(), 1.0)]),
            ("t_1_8".into(), vec![("Label_1".into(), 1.0)]),
            ("t_1_9".into(), vec![("Label_1".into(), 1.0)]),

        ]))?;

        store_guard.save_pure_sets(4, true)?;
        store_guard.save_pure_sets(6, true)?;

        Ok(())
    }

    fn plot_t(t_0: &Tensor, label:&str) -> Result<(), ThoosaError> {
        plot_series(
            (0..1).into_iter().map(|channel| {
                (format!("0"), BLACK, t_0.i((channel, ..)).iter::<f64>().unwrap().enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>())
            }).collect(),
            label, (2000, 1000), &format!("/tmp/plots/{label}_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str()
        )?;
        Ok(())
    }

    fn label_1_t(n: i64, max_y: f32) -> Tensor {
        Tensor::vstack(&[
            Tensor::from_slice([sig_one_imp(n, max_y, Some(10. .. 12.), Some(10. .. 12.)), sig_one_imp(n, max_y, Some(10. .. 12.), Some(10. .. 12.))].concat().as_slice()),
        ])
    }

    fn label_0_t(n: i64, max_y: f32) -> Tensor {
        Tensor::vstack(&[
            Tensor::from_slice([sig_three_imp(n, max_y, Some(10. .. 12.), Some(10. .. 12.)), sig_two_imp(n, max_y, Some(10. .. 12.), Some(10. .. 12.))].concat().as_slice()),
        ])
    }

    fn save_test_motifs(store: Arc<Mutex<Box<dyn Store>>>, find_motifs_params: FindMotifsParams, param_set: Vec<FindMotifsRunConfig>, resource_label_key: &str) -> Result<usize, ThoosaError> {
        let resource_0_ids:Vec<String> = (0..10).map(|i| format!("{resource_label_key}{i}")).collect();
        let run_id = store.lock().save_tree_run(
            None,
            format!("find motifs with params {:?}/{:?}", find_motifs_params, param_set[0]).as_str(),
            "".into(), //&resource_0_ids,
            param_set[0].to_properties_map(),
        )?;

        store.lock().save_slices(
            run_id,
            0i64,
            resource_0_ids.windows(2).into_iter().flat_map(|resource_id| {
                [
                    ((resource_id[0].clone(), 100), (resource_id[0].clone(), 1100), 1.0),
                    ((resource_id[0].clone(), 100), (resource_id[1].clone(), 100), 1.0),
                    ((resource_id[0].clone(), 100), (resource_id[1].clone(), 1100), 1.0),
                    ((resource_id[1].clone(), 100), (resource_id[1].clone(), 1100), 1.0)
                ]
            }).collect::<Vec<_>>()
        )?; //all tensors contain two signals - [0-1000) and [1000-2000)


        for i in 0..1 { //single channel
            { store.lock().set_motifs(run_id, i) }?;
        }

        { store.lock().delete_slices_without_matches(run_id) }?;
        Ok(run_id)
    }

    #[test]
    fn test_rust_step() -> Result<(), ThoosaError> {

        let n = 1000i64;
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let tensors_path = Some(test_dir.join("tensors"));
        let ws = Arc::new(RwLock::new(FsWorkspace::new(test_dir, 1, 2 * n as usize, tensors_path)?));
        let ws_runner = Arc::clone(&ws);

        (0..10).for_each(|i|{
            label_0_t(n, 100.).save(ws.read().tensors_path().join(format!("t_0_{i}.pt"))).unwrap();
        });

        // fs::remove_file("/tmp/test_main_db.db3").unwrap_or(());
        // let store = Arc::new(Mutex::new(SqlStore::from_path("/tmp/test_main_db.db3")?));
        let store = Arc::new(Mutex::new(SqlStore::in_mem()?));
        init_test_store(store.lock());

        let loader = Arc::new(RwLock::new(LoadRunner::new(
            move |key: &Arc<String>| {
                let path = ws_runner.read().tensor_path(key);
                debug!("test_run_train_on_motifs: path={path:?}");
                SyncWrapper::new(
                    Box::new(Tensor::load(path).unwrap()),
                    |e| Box::new(e.shallow_clone())
                )
            }, 4, 100,
        )));

        let devices = vec![Cpu, Cpu];
        let run_ids = run_step(
            None,
            &FindMotifsParams::dummy(10, 10..20, "Label_0/4".into(), 10, 0.1, None),
            find_motifs_by_euc_dist,
            &ws,
            &store,
            &loader,
            devices
        )?;

        println!("run_ids={run_ids:?}");

        assert_eq!(vec![1,2],run_ids);

        Ok(())
    }

    #[test]
    fn test_cuda(){
        let cuda = Device::cuda_if_available();

        println!("cuda={cuda:?}");
    }

}