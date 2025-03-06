use crate::cli::Commands::{
    FindMotifs, ImportResources, Infer, MotifsToTensor, PlotMotifSlices, RunAll, SayHello,
    SelectMotifs, TrainFinalNet, TrainLabelNets, TrainMotifsNets,
};
use crate::data;
use crate::data::{clc_records, labeled_tensors_by_descr, load_to_logits_tensors, ParquetLoader};
use clap::{Args, Parser, Subcommand};
use futures_util::{AsyncReadExt, StreamExt};
use lazy_static::lazy_static;
use log::{debug, error, info, trace, warn, Level, LevelFilter};
use parking_lot::{Mutex, RwLock};
use plotters::prelude::BLACK;
use polars::export::arrow::pushable::Pushable;
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use simplelog::{ColorChoice, CombinedLogger, Config, TermLogger, TerminalMode, WriteLogger};
use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Debug, Display};
use std::fs::{read_dir, File, OpenOptions};
use std::hash::Hash;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::str::FromStr;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;
use std::{fs, process};
use tch::nn::{Module, VarStore};
use tch::Device::{Cpu, Cuda};
use tch::Kind::Float;
use tch::{Device, Kind, Tensor};
use thoosa_common::input_probs::InputProbs;
use thoosa_common::loader::{LoadRunner, SyncWrapper};
use thoosa_common::parzen::optimizer::ConfigurationSpace;
use thoosa_common::runs::{DummySavableOptResult, Empty, Labeled, Logits, NetResources, Params, RunConfig, SavableOptResult, Simple};
use thoosa_common::train_nets::{apply_motif_and_label_nets, load_final_model, load_label_models_ws, load_motif_models, run_opt_train_final, run_opt_train_label_nodes, train_label_nodes, CommonTrainFinalNodeRunConfig, CommonTrainLabelNodeRunConfig, CommonTrainMotifsNodesRunConfig, FinalNodeParams, FindMotifsRunConfig, FixedNumSlicesNetParams, InferParams, InputFilter, LabelNetType, LabelNodeParams, MotifNodeParams, OptConfig, ThoModuleWithStore, TrainFinalNetParams, TrainFinalNodeRunConfig, TrainLabelNetsParams, TrainLabelNodeRunConfig, TrainMotifsNetsParams, TrainMotifsNodesRunConfig, TrainNodesValues};
use thoosa_common::utils::{cuda_mem, from, list_devices, plot_series};
use thoosa_common::{train_nets, ResourceLoader};
use thoosa_motifs::find;
use thoosa_motifs::find::{
    find_motifs_by_euc_dist, select_motifs_by_ratio, select_motifs_for_label,
    CommonFindMotifsConfig, FindMotifsParams, SelectMotifsParams,
    SelectMotifsRunConfig,
};
use thoosa_motifs::fragments::{
    load_motif_slices, motifs_to_tensor, MotifsBy, MotifsToTensorParams, PlotMotifSlicesParams,
};
use thoosa_shared::{parse_range_or_value, slices_len};
use thoosa_shared::val::{prefix_properties, unprefix_properties, Val};
use thoosa_shared::{
    batch_windows, cdt, cdt_for_dir, clc_num_batches, make_combos, RangeOrValue, ResourceType,
    TensorIdSliceIdx, ThoosaError,
};
use thoosa_store::{SqlStore, Store};
use torch_sys::at_grad_set_enabled;

use crate::version::VERSION;
use thoosa_common::ws::{DummyWorkspace, FsWorkspace, Workspace};
use thoosa_motifs::distance::average_euclidean_distance;
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::resource::{Resource, ResourceId, ResourceStats, ResourcesConfig};

#[derive(Parser, Clone, Debug)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[arg(long, default_value = ".")]
    pub ws_path: String,

    #[arg(long, default_value = "INFO")]
    pub term_logger_level: LevelFilter,

    #[arg(long, default_value = "INFO")]
    pub write_logger_level: LevelFilter,

    #[arg(long)]
    pub loader_threads: Option<usize>,

    #[arg(long, default_value_t = 100)]
    pub futures_channel_bound: usize,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Debug, Args, Clone)]
pub struct RunAllParams {
    #[arg(long)]
    train_resource_group: String,

    #[arg(long)]
    valid_resource_group: String,

    #[arg(long)]
    pub import_source_path: Option<String>,

    #[arg(long, value_parser=parse_range_or_value::<u32>, default_value = "1000")]
    pub step: RangeOrValue<u32>,

    #[arg(long, value_parser=parse_range_or_value::<u32>, default_value = "1000")]
    pub window: RangeOrValue<u32>,

    #[arg(long, default_value_t = 2)]
    find_motifs_chunk_size: usize,

    #[command(subcommand)]
    find_motifs_input_filter: Option<InputFilter>,

    #[arg(long, default_value_t = 0.3)]
    find_motifs_max_dist_k: f32,

    #[arg(long, default_value_t = 0.3)]
    select_motifs_min_ratio: f32,

    #[arg(long, default_value_t = 10)]
    batch_size: i64,

    #[arg(long, default_value_t = 0.2)]
    pub max_loss: f32,

    #[arg(long, default_value_t = 100)]
    cycles: usize,

    #[arg(long, value_parser=parse_range_or_value::<f64>)]
    pub lr: RangeOrValue<f64>,

    #[arg(long, value_parser=parse_range_or_value::<u32>)]
    pub num_layers: RangeOrValue<u32>,

    #[arg(long, default_value_t = 10)]
    pub opt_max_evals: u32,

    #[arg(long, default_value_t = 3)] //TODO validate opt_n_init < opt_max_evals
    pub opt_n_init: u32,
}

impl RunAllParams {
    pub fn train_resource_group(&self) -> &str {
        &self.train_resource_group
    }

    pub fn dummy(train_resource_group: &str, valid_resource_group: &str) -> Self {
        Self {
            train_resource_group: train_resource_group.to_string(),
            valid_resource_group: valid_resource_group.to_string(),
            import_source_path: None,
            step: 1000.into(),
            window: 1000.into(),
            find_motifs_chunk_size: 1000,
            find_motifs_input_filter: None,
            find_motifs_max_dist_k: 0.3,
            select_motifs_min_ratio: 0.3,
            cycles: 1000,
            lr: 1e-5.into(),
            num_layers: 3.into(),

            batch_size: 10,
            max_loss: 0.1,
            opt_n_init: 1,
            opt_max_evals: 3,
        }
    }

    pub fn valid_resource_group(&self) -> &str {
        &self.valid_resource_group
    }

    pub fn to_net_config_space(&self) -> Option<ConfigurationSpace<String>> {
        [
            ("lr".into(), &self.lr).try_into().ok(),
            ("num_layers".into(), &self.num_layers).try_into().ok(),
        ]
        .into_iter()
        .collect::<Option<Vec<Hyperparameter<String>>>>()
        .map(|v| ConfigurationSpace::new(v))
    }
}

impl Params<FindMotifsRunConfig, Simple, CommonFindMotifsConfig> for RunAllParams {
    fn to_run_configs(&self) -> Vec<FindMotifsRunConfig> {
        debug!(
            "FindMotifsParams::to_run_configs: self.step={:?} self.window={:?}",
            self.step, self.window
        );

        make_combos([
            self.step.clone().into_iter(),
            self.window.clone().into_iter(),
        ].into_iter())
        .into_iter()
        .map(|config_combo| {
            debug!("FindMotifsParams::to_run_configs: config_combo={config_combo:?} ");
            FindMotifsRunConfig::new(
                None,
                config_combo[0],
                config_combo[1],
                1.0,
                "".to_string(),
                None,
            )
        })
        .collect()
    }

    fn resource_config(&self) -> Simple {
        Simple::new(ResourceType::Group(self.train_resource_group.clone()))
    }

    fn common_config(&self) -> CommonFindMotifsConfig {
        CommonFindMotifsConfig::new(
            None,
            self.find_motifs_chunk_size as u32,
            false,
            Some(0.5),  //TODO config
            Some(0.01), //TODO config
        )
    }
}

impl Params<SelectMotifsRunConfig, Labeled, ()> for RunAllParams {
    fn to_run_configs(&self) -> Vec<SelectMotifsRunConfig> {
        debug!(
            "FindMotifsParams::to_run_configs: self.step={:?} self.window={:?}",
            self.step, self.window
        );

        vec![SelectMotifsRunConfig::new()]
    }

    fn resource_config(&self) -> Labeled {
        Labeled::new(self.train_resource_group.clone())
    }

    fn common_config(&self) -> () {
        ()
    }
}

impl Params<TrainMotifsNodesRunConfig, Empty, CommonTrainMotifsNodesRunConfig> for RunAllParams {
    fn to_run_configs(&self) -> Vec<TrainMotifsNodesRunConfig> {
        debug!(
            "FindMotifsParams::to_run_configs: self.step={:?} self.window={:?}",
            self.step, self.window
        );
        if let Some(config_space) = self.to_net_config_space() {
            debug!("TrainMotifsNodesParams::to_run_configs: config_space={config_space:?}");
            vec![TrainMotifsNodesRunConfig::from_opt_space(
                None,
                config_space,
                OptConfig::new(self.opt_max_evals, self.opt_n_init),
            )]
        } else {
            //TODO avoid collecting
            let v = make_combos([
                self.lr
                    .clone()
                    .into_iter()
                    .map(|v| v.into())
                    .collect::<Vec<Val>>()
                    .into_iter(),
                self.num_layers
                    .clone()
                    .into_iter()
                    .map(|v| v.into())
                    .collect::<Vec<Val>>()
                    .into_iter(),
            ].into_iter())
            .into_iter()
            .map(|mut config_combo: Vec<Val>| {
                debug!("TrainMotifsNodesParams::to_run_configs: config_combo={config_combo:?} ");
                let num_layers = config_combo.pop().unwrap().try_into().unwrap();
                let lr = config_combo.pop().unwrap().try_into().unwrap();
                TrainMotifsNodesRunConfig::from_values(None, lr, num_layers)
            })
            .collect();

            debug!(
                "TrainMotifsNodesParams::to_run_configs: self.lr={:?} self.num_layers={:?} v={v:?}",
                self.lr, self.num_layers
            );

            v
        }
    }

    fn resource_config(&self) -> Empty {
        Empty {}
    }

    fn common_config(&self) -> CommonTrainMotifsNodesRunConfig {
        CommonTrainMotifsNodesRunConfig {
            batch_size: self.batch_size as usize,
            cycles: self.cycles,
            max_loss: 0.1,
        }
    }
}

impl Params<TrainLabelNodeRunConfig, NetResources<Labeled>, CommonTrainLabelNodeRunConfig>
    for RunAllParams
{
    fn to_run_configs(&self) -> Vec<TrainLabelNodeRunConfig> {
        if let Some(config_space) = self.to_net_config_space() {
            debug!("TrainLabelNodeParams::to_run_configs: config_space={config_space:?}");
            vec![TrainLabelNodeRunConfig::from_opt_space(
                None,
                config_space,
                OptConfig::new(self.opt_max_evals, self.opt_n_init),
            )]
        } else {
            //TODO avoid collecting
            let v = make_combos([
                self.lr
                    .clone()
                    .into_iter()
                    .map(|v| v.into())
                    .collect::<Vec<Val>>()
                    .into_iter(),
                self.num_layers
                    .clone()
                    .into_iter()
                    .map(|v| v.into())
                    .collect::<Vec<Val>>()
                    .into_iter(),
            ].into_iter())
            .into_iter()
            .map(|mut config_combo: Vec<Val>| {
                debug!("TrainLabelNodeParams::to_run_configs: config_combo={config_combo:?} ");
                let num_layers = config_combo.pop().unwrap().try_into().unwrap();
                let lr = config_combo.pop().unwrap().try_into().unwrap();
                TrainLabelNodeRunConfig::from_values(None, lr, num_layers, LabelNetType::FixedNumSlices(FixedNumSlicesNetParams{ num_layers: 3.into(), slices_len: 100.into() }))
            })
            .collect();

            debug!(
                "TrainLabelNodeParams::to_run_configs: self.lr={:?} self.num_layers={:?} v={v:?}",
                self.lr, self.num_layers
            );

            v
        }
    }

    fn resource_config(&self) -> NetResources<Labeled> {
        NetResources::new(
            Labeled::new(self.train_resource_group.clone()),
            Labeled::new(self.valid_resource_group.clone()),
        )
    }

    fn common_config(&self) -> CommonTrainLabelNodeRunConfig {
        CommonTrainLabelNodeRunConfig::new(
            self.batch_size as usize,
            self.cycles,
        )
    }
}

impl Params<TrainFinalNodeRunConfig, NetResources<Logits>, CommonTrainFinalNodeRunConfig>
    for RunAllParams
{
    fn to_run_configs(&self) -> Vec<TrainFinalNodeRunConfig> {
        if let Some(config_space) = self.to_net_config_space() {
            debug!("TrainFinalNodeParams::to_run_configs: config_space={config_space:?}");
            vec![TrainFinalNodeRunConfig::from_opt_space(
                None,
                config_space,
                OptConfig::new(self.opt_max_evals, self.opt_n_init),
            )]
        } else {
            //TODO avoid collecting
            let v = make_combos([
                self.lr
                    .clone()
                    .into_iter()
                    .map(|v| v.into())
                    .collect::<Vec<Val>>()
                    .into_iter(),
                self.num_layers
                    .clone()
                    .into_iter()
                    .map(|v| v.into())
                    .collect::<Vec<Val>>()
                    .into_iter(),
            ].into_iter())
            .into_iter()
            .map(|mut config_combo: Vec<Val>| {
                debug!("TrainFinalNodeParams::to_run_configs: config_combo={config_combo:?} ");
                let num_layers = config_combo.pop().unwrap().try_into().unwrap();
                let lr = config_combo.pop().unwrap().try_into().unwrap();
                TrainFinalNodeRunConfig::from_values(None, lr, num_layers)
            })
            .collect();

            debug!(
                "TrainFinalNodeParams::to_run_configs: self.lr={:?} self.num_layers={:?} v={v:?}",
                self.lr, self.num_layers
            );

            v
        }
    }

    fn resource_config(&self) -> NetResources<Logits> {
        NetResources::new(
            Logits::new(self.train_resource_group.clone()),
            Logits::new(self.valid_resource_group.clone()),
        )
    }

    fn common_config(&self) -> CommonTrainFinalNodeRunConfig {
        CommonTrainFinalNodeRunConfig::new(
            self.cycles,
            self.batch_size as usize,
        )
    }
}

/**
export LD_LIBRARY_PATH=/home/bu/dvl/torch/libtorch-2.5.0/lib/ && cargo run --release -- \
--term-logger-level=TRACE --write-logger-level=INFO  --ws-path="/tmp/ws_import_test"  \
--logfile=/tmp/ws_1/thoosa.log import-resources --source-path=/media/bu/KINGSTON/hbac_orig

*/
#[derive(Debug, Args, Clone, Default)]
pub struct ImportResourcesParams {
    #[arg(long)]
    pub source_path: String,
}

#[derive(Debug, Args, Clone)]
pub struct SayHelloParams {
    #[arg(long, value_parser=parse_range_or_value::<f64>, default_value = "1.0e-5")]
    pub lr: RangeOrValue<f64>,

    #[arg(long, value_parser=parse_range_or_value::<u32>, default_value = "1000")]
    pub step: RangeOrValue<u32>,

    #[arg(long, value_parser=parse_range_or_value::<u32>, default_value = "1000")]
    pub window: RangeOrValue<u32>,
}

#[derive(Debug)]
pub struct SayHelloRunConfig {
    step: u32,
    window: u32,
    cs: Option<ConfigurationSpace<String>>,
}

impl Params<SayHelloRunConfig, Simple, ()> for SayHelloParams {
    fn to_run_configs(&self) -> Vec<SayHelloRunConfig> {
        let tpe_config_space = [("lr".into(), &self.lr).try_into().ok()]
            .into_iter()
            .collect::<Option<Vec<Hyperparameter<String>>>>()
            .map(|v| ConfigurationSpace::new(v));

        make_combos([
            self.window.clone().into_iter(),
            self.step.clone().into_iter(),
        ].into_iter())
        .into_iter()
        .map(|config_combo| SayHelloRunConfig {
            window: config_combo[0],
            step: config_combo[1],
            cs: tpe_config_space.as_ref().map(|v| v.clone()),
        })
        .collect()
    }

    fn resource_config(&self) -> Simple {
        Simple::new(ResourceType::List(vec![]))
    }

    fn common_config(&self) -> () {
        ()
    }
}

impl RunConfig for SayHelloRunConfig {
    fn config_space(&self) -> Option<&ConfigurationSpace<String>> {
        self.cs.as_ref()
    }

    fn from_properties_map(run_id: usize, properties: HashMap<String, Val>) -> Self {
        todo!()
    }

    fn set_id(&mut self, id: usize) {}

    fn type_name() -> &'static str {
        "Hello"
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
    ImportResources(ImportResourcesParams),
    FindMotifs(FindMotifsParams),
    PlotMotifSlices(PlotMotifSlicesParams),
    MotifsToTensor(MotifsToTensorParams),
    SelectMotifs(SelectMotifsParams),
    TrainMotifsNets(TrainMotifsNetsParams),
    TrainLabelNets(TrainLabelNetsParams),
    TrainFinalNet(TrainFinalNetParams),
    RunAll(RunAllParams),
    Infer(InferParams),
    SayHello(SayHelloParams),
}

fn set_oom_score_adj(score: i32) {
    let pid = process::id();
    let path = format!("/proc/{}/oom_score_adj", pid);

    if let Ok(mut file) = OpenOptions::new().write(true).open(&path) {
        if let Err(e) = write!(file, "{}", score) {
            eprintln!("Failed to write to {}: {}", path, e);
        }
    } else {
        eprintln!("Failed to open {}", path);
    }
}

pub fn run_cli<F, R>(
    labeled_resources: F,
    resource_to_tensor: R,
    labels: Vec<&str>,
    channels: usize,
    max_input: usize,
) -> Result<(), ThoosaError>
where
    F: Fn(&dyn AsRef<Path>) -> Result<HashMap<String, Vec<(String, f32)>>, ThoosaError>,
    R: Fn(&dyn AsRef<Path>, &str) -> Result<Tensor, ThoosaError> + Sync + Send,
{
    // set_oom_score_adj(-1000);

    unsafe {
        at_grad_set_enabled(0);
    };

    let cli = Cli::parse();

    let ws = Arc::new(RwLock::new(FsWorkspace::new(
        cli.ws_path.clone(),
        channels,
        max_input,
    )?));

    let logfile = match File::create(
        ws.read()
            .log_path()
            .join(format!("thoosa_{}.log", cdt_for_dir()).to_owned()),
    ) {
        Ok(file) => file,
        Err(err) => {
            panic!("Can't create logfile in {:?}", cli.ws_path);
        }
    };
    CombinedLogger::init(vec![
        TermLogger::new(
            LevelFilter::from_str(cli.term_logger_level.as_str())?,
            Config::default(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        ),
        WriteLogger::new(
            LevelFilter::from_str(cli.write_logger_level.as_str())?,
            Config::default(),
            logfile,
        ),
    ])
    .expect("failed to init logging");

    let store = Arc::new(Mutex::new(SqlStore::from_path(ws.read().store_path())?));

    let devices = list_devices();

    let cpu_num = std::thread::available_parallelism()?;
    let loader_threads = cli
        .loader_threads
        .unwrap_or_else(|| max(2, cpu_num.get() / 8));

    //TODO torch.backends.cuda.matmul.allow_tf32 = True
    info!("VERSION: {VERSION} devices={devices:?}, base_path={:?} cpus={cpu_num} loader_threads={loader_threads} tch::get_num_interop_threads={}, tch::get_num_threads={}",  cli.ws_path, tch::get_num_interop_threads(), tch::get_num_threads());
    std::io::stdout().flush()?;

    let tensors_path = ws.read().tensors_path().clone();

    let loader = Arc::new(RwLock::new(LoadRunner::new(
        move |key: &Resource| {
            let tensor = Tensor::load_multi_with_device(
                tensors_path.join(format!("{}_{}.pt", key.import_id(), key.file_stem())),
                Cpu,
            ) //Cuda(0))
            .map_err(|e| ThoosaError::new(e.to_string()))
            .expect(format!("can't load {key:?}").as_str())
            .pop()
            .unwrap()
            .1
            .totype(Float);
            if tensor.requires_grad() || tensor.retains_grad() {
                warn!("tensor requires grad!");
            }
            debug!("key={key:?} tensor.size={:?}", tensor.size());
            SyncWrapper::new_tb(Box::new(tensor))
        },
        loader_threads,
        cli.futures_channel_bound,
    )));

    match &cli.command {
        Some(ImportResources(import_resources_params)) => {
            debug!(
                "ImportResources: import_resources_params={:?}",
                import_resources_params
            );
            import_resources(
                &ws,
                &store,
                labels,
                || labeled_resources(&import_resources_params.source_path),
                |key: String| {
                    resource_to_tensor(&import_resources_params.source_path, key.as_str())
                },
                true,
            )?;

            info!("ImportResources done");
        }

        Some(FindMotifs(find_motifs_params)) => {
            debug!(
                "find_motifs_params={:?} devices={:?}",
                find_motifs_params, devices
            );

            let run_ids = run_step(
                None,
                find_motifs_params,
                find_motifs_by_euc_dist,
                &ws,
                &store,
                &loader,
                devices,
                "FindMotifs",
            )?;

            info!("FindMotifs done, run_ids={:?}", run_ids);
        }

        Some(SelectMotifs(select_motifs_params)) => {
            debug!(
                "SelectMotifs: select_motifs_params={:?} devices={:?}",
                select_motifs_params, devices
            );

            let run_ids = run_step(
                Some(select_motifs_params.find_motifs_run_ids().clone()),
                select_motifs_params,
                select_motifs_for_label,
                &ws,
                &store,
                &loader,
                devices.clone(),
                "SelectMotifs",
            )?;

            info!("SelectMotifs done, run_ids={:?}", run_ids);
        }

        Some(TrainMotifsNets(train_on_motif_params)) => {
            debug!(
                "TrainOnMotifs: train_on_motif: train_on_motif_params={:?}",
                train_on_motif_params
            );
            unsafe {
                at_grad_set_enabled(1);
            }
            let run_ids = run_step(
                Some(train_on_motif_params.motif_selection_run_id.clone()),
                train_on_motif_params,
                run_train_motif_nodes_on_motif_selection,
                &ws,
                &store,
                &loader,
                devices.clone(),
                "TrainFinalNode",
            )?;

            info!("TrainOnMotifs done, run_ids={:?}", run_ids);
        }

        Some(TrainLabelNets(train_on_label_params)) => {
            debug!(
                "TrainOnLabel: train_on_label_params={:?} cuda_mem={:?}",
                train_on_label_params,
                cuda_mem()
            );
            unsafe {
                at_grad_set_enabled(1);
            }

            let run_ids = run_step(
                Some(
                    train_on_label_params
                        .select_motifs_for_label_run_ids
                        .clone(),
                ),
                train_on_label_params,
                run_train_label_nodes,
                &ws,
                &store,
                &loader,
                devices.clone(),
                "TrainOnLabel",
            )?;

            info!("TrainOnLabel done, run_ids={:?}", run_ids);
        }

        Some(TrainFinalNet(train_final_params)) => {
            debug!("TrainFinal: train_final_params={:?}", train_final_params);
            unsafe {
                at_grad_set_enabled(1);
            }

            let run_ids = run_step(
                Some(train_final_params.train_labels_run_ids.clone()),
                train_final_params,
                run_train_final_node,
                &ws,
                &store,
                &loader,
                devices.clone(),
                "TrainFinalNode",
            )?;

            info!("TrainFinal done, run_ids={run_ids:?}");
        }

        Some(RunAll(run_all_params)) => {
            debug!("RunAll: run_all_params={:?}", run_all_params);

            if let Some(source_path) = run_all_params.import_source_path.as_ref() {
                debug!("RunAll: init_resources");
                import_resources(
                    &ws,
                    &store,
                    labels,
                    || labeled_resources(source_path),
                    |key: String| resource_to_tensor(&source_path, key.as_str()),
                    true,
                )?;
            }

            //run_id - params set
            //let find_motifs_run_ids = run_find_motifs_on_group(&ws, &store, devices.clone(), run_all_params, &loader)?;

            let find_motifs_run_ids = run_step(
                None,
                run_all_params,
                find_motifs_by_euc_dist,
                &ws,
                &store,
                &loader,
                devices.clone(),
                "RunAll/FindMotifs",
            )?;

            info!(
                "find_motifs done: find_motifs_run_ids={:?}",
                find_motifs_run_ids
            );

            let motif_selection_run_ids = run_step(
                Some(find_motifs_run_ids),
                run_all_params,
                select_motifs_for_label,
                &ws,
                &store,
                &loader,
                devices.clone(),
                "RunAll/SelectMotifs",
            )?;

            info!(
                "select_motifs done: motif_selection_run_ids={:?}",
                motif_selection_run_ids
            );
            unsafe {
                at_grad_set_enabled(1);
            }
            let train_motif_nodes_run_ids = run_step(
                Some(motif_selection_run_ids.clone()),
                run_all_params,
                run_train_motif_nodes_on_motif_selection,
                &ws,
                &store,
                &loader,
                devices.clone(),
                "Test/TrainMotifsNodes",
            )?;

            info!("run_train_on_motifs done: train_motif_nodes_run_ids={train_motif_nodes_run_ids:?}");

            let train_label_nodes_run_ids = run_step(
                Some(motif_selection_run_ids),
                run_all_params,
                run_train_label_nodes,
                &ws,
                &store,
                &loader,
                devices.clone(),
                "Test/TrainLabelsNodes",
            )?;

            info!("run_train_on_label done: train_label_nodes_run_ids={train_label_nodes_run_ids:?}");

            let train_final_nodes_run_ids = run_step(
                Some(train_label_nodes_run_ids),
                run_all_params,
                run_train_final_node,
                &ws,
                &store,
                &loader,
                devices.clone(),
                "Test/TrainFinalNode",
            )?;

            info!("run_train_final done: train_final_nodes_run_ids={train_final_nodes_run_ids:?}");
        }

        Some(Infer(infer_params)) => {
            run_infer(&ws, &store, infer_params, devices[0], |path| {
                // Ok(Tensor::load(path)?) TODO config
                data::parquet_to_tensor_by_path(
                    <Path as AsRef<Path>>::as_ref(path),
                    *devices.first().expect("first device"),
                )
            })?;
        }

        Some(PlotMotifSlices(plot_motif_slices_params)) => {
            run_plot_motif_slices(&ws, &store, &loader, &plot_motif_slices_params, &devices[0])?;
        }

        Some(MotifsToTensor(choose_fragments_params)) => {
            motifs_to_tensor_cmd(
                &cli,
                &store,
                *devices.first().expect("first device"),
                choose_fragments_params,
                &loader,
            )?;
        }

        //gdb --args target/x86_64-unknown-linux-gnu/debug/thoosa_hbac --term-logger-level=DEBUG --base-path=/tmp/ --resource-pattern=xxx say-hello
        Some(SayHello(params)) => {
            debug!("params={:?}", params);
            let run_configs = params.to_run_configs();
            debug!("run_configs={:?}", run_configs);

            let dir = read_dir(ws.read().tensors_path())?;

            let mut lengths = vec![];

            for tensorfile in dir {
                let path_buf = tensorfile?.path();
                let t = Tensor::load(&path_buf);
                let t_size = t?.size();
                debug!("path_buf={path_buf:?} size={:?}", t_size);
                lengths.push(t_size[1]);
            }

            info!(
                "tensors: {} max={:?} avg={}",
                lengths.len(),
                lengths.iter().max(),
                lengths.iter().sum::<i64>() / lengths.len() as i64
            );
        }

        None => {
            warn!("No command");
        }
    }

    Ok(())
}

pub fn run_train_final_node(
    train_label_nodes_run_ids: Option<&Vec<usize>>, //TODO use to filter label nets
    run_configs: Vec<TrainFinalNodeRunConfig>,
    resource_config: &NetResources<Logits>,
    common_config: &CommonTrainFinalNodeRunConfig,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Mutex<Box<dyn Store>>>,
    loader: &Arc<RwLock<LoadRunner<Box<Tensor>>>>,
    devices: Vec<Device>,
) -> Result<Option<Vec<Box<dyn SavableOptResult>>>, ThoosaError> {
    debug!("run_train_final_node: run_configs.len={} resource_config={resource_config:?}", run_configs.len());
    let (train_resource_keys, valid_resource_keys) = resource_config.to_resources(store)?;

    run_configs
        .into_iter()
        .map(|train_final_run_config| {
            let (label_models, labels_channels_motifs_models) =
                load_label_and_motif_models(&ws, store, &Cpu, train_label_nodes_run_ids); //TODO move out of run_configs cycle

            debug!("label_models={label_models:?} train_label_nodes_run_ids={train_label_nodes_run_ids:?} labels_channels_motifs_models={labels_channels_motifs_models:?}");

            run_opt_train_final(
                ws,
                store,
                &train_final_run_config,
                &common_config,
                train_resource_keys.clone(),
                valid_resource_keys.clone(),
                &devices[0],
                loader,
                label_models,
                labels_channels_motifs_models,
            )
        })
        .collect::<Result<Vec<Box<dyn SavableOptResult>>, ThoosaError>>()
        .map(|v| v.into_iter().collect::<Vec<Box<dyn SavableOptResult>>>())
        .map(|v| Some(v))
}

pub fn run_infer<F>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Mutex<Box<dyn Store>>>,
    infer_params: &InferParams,
    device: Device,
    load_tensor: F,
) -> Result<InputProbs, ThoosaError>
where
    F: Fn(&PathBuf) -> Result<Tensor, ThoosaError> + Sync,
{
    let final_model = load_final_model(infer_params.final_run_id, ws, &device)?;

    let train_label_node_run_ids = store.lock().get_input_run_ids(
        infer_params.final_run_id,
        Some(&TrainLabelNodeRunConfig::type_name())
    ).expect(format!("TrainLabelNodeRunConfig for infer_params.final_run_id={:?} not found", infer_params.final_run_id).as_str());

    let (mut label_models, labels_channels_motifs_models) =
        load_label_and_motif_models(
            &ws,
            store,
            &device,
            Some(
                &train_label_node_run_ids
            )
        );

    let labels_channels_motifs_models_rw =
        labels_channels_motifs_models
            .into_iter()
            .fold(HashMap::new(), |mut a, v| {
                a.insert(v.0, Arc::new(RwLock::new(v.1)));
                a
            });

    let labels = store.lock().list_labels()?;
    let mut infer_result = InputProbs::new(labels.clone());

    for entry in read_dir(&infer_params.input_dir)? {
        let path = entry?.path();

        let t = apply_motif_and_label_nets(
            &load_tensor(&path)?.unsqueeze(0),//for batch dim
            &mut label_models,
            &labels_channels_motifs_models_rw,
            &device,
            &labels,
            ws.read().max_input() as u32
        );

        let logits = final_model.2.infer(&t.view((1, -1))); //batch(1 for infer) x classes
        let probs = logits.softmax(-1, Float);

        debug!(
            "run_infer:={path:?} label=\n{t} t.view((1,-1))=\n{} logits=\n{logits} probs=\n{probs}",
            t.view((1, -1))
        );

        infer_result.add(
            path.file_stem()
                .expect(format!("no stam for {path:?}").as_str())
                .to_string_lossy()
                .to_string(),
            from::<Vec<f32>>(&probs.squeeze()),
        );

        info!(
            "run_infer: entry={path:?} result=\n{logits} argmax={:?}",
            logits.argmax(-1, false)
        );
    }

    Ok(infer_result)
}


fn load_label_and_motif_models(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Mutex<Box<dyn Store>>>,
    device: &Device,
    train_label_nodes_run_ids: Option<&Vec<usize>>,
) -> (
    HashMap<String, Vec<(usize, LabelNodeParams, ThoModuleWithStore)>>,
    HashMap<String, Vec<(FindMotifsRunConfig, Vec<(MotifNodeParams, ThoModuleWithStore)>)>>,
) {
    debug!("load_label_and_motif_models: train_label_nodes_run_ids={train_label_nodes_run_ids:?}");

    let label_models = load_label_models_ws(&ws, train_label_nodes_run_ids, device).expect("load_all_label_models_ws");

    debug!("load_label_and_motif_models: label_models={label_models:?}");

    let mut labels_channels_motifs_models: HashMap<
        String,
        Vec<(FindMotifsRunConfig, Vec<(MotifNodeParams, ThoModuleWithStore)>)>,
    > = HashMap::new();

    for (label, label_run_id_nodes) in label_models.iter() {

        debug!("load_label_and_motif_models: label={label:?} label_run_id_nodes={label_run_id_nodes:?}");

        for (label_run_id, label_node_params, node) in label_run_id_nodes {
            // let motif_node_config_run_ids = store.lock().get_input_run_ids(*label_run_id, Some(&TrainMotifsNodesRunConfig::type_name())).expect("get_input_run_id(label_run_id)");
            // debug!("load_all_label_and_motif_models: label={label} label_run_id={label_run_id:?} motif_node_config_run_ids={motif_node_config_run_ids:?}");
            // assert_eq!(motif_node_config_run_ids.len(), 1);
            debug!("load_label_and_motif_models: label_run_id={label_run_id:?}");

            let motif_selection_run_ids = store
                .lock()
                .get_input_run_ids(*label_run_id, Some(&SelectMotifsRunConfig::type_name()))
                .expect("get_input_run_ids(motif_node_params_selection_run_ids")
                .into_iter()
                .collect::<HashSet<usize>>(); //TODO distinct in sql?

            debug!("load_all_label_and_motif_models: label={label} label_run_id={label_run_id} motif_selection_run_ids={motif_selection_run_ids:?}");

            assert_eq!(motif_selection_run_ids.len(), 1);

            let find_motifs_for_label_run_ids = store.lock().get_input_run_ids(*motif_selection_run_ids.iter().next().unwrap(), Some(&FindMotifsRunConfig::type_name())).expect(format!("FindMotifsRunConfig for motif_selection_run_ids={motif_selection_run_ids:?} not found").as_str());
            assert_eq!(find_motifs_for_label_run_ids.len(), 1);
            let find_motifs_run_config = FindMotifsRunConfig::from_properties_map(find_motifs_for_label_run_ids[0], store.lock().list_run_config_values(find_motifs_for_label_run_ids[0]).expect("list_run_params"));


            let channels_motifs_models = store
                .lock()
                .list_selected_motifs(
                    motif_selection_run_ids.into_iter().next().unwrap(),
                    ws.read().channels() as u32,
                )
                .expect("list_selected_motifs")
                .into_iter()
                .enumerate()
                .map(|(channel, channels_motifs)| {
                    let ws_moved = Arc::clone(&ws);

                    (
                        find_motifs_run_config.clone(),
                    channels_motifs
                        .into_iter()
                        //TODO more than one model for a motif?
                        .flat_map(move |(motif_id, rate)| {
                            load_motif_models(&ws_moved, channel, motif_id, device).expect("load_motif_models")
                        })
                        .collect::<Vec<_>>()
                    )
                })
                .collect::<Vec<(FindMotifsRunConfig, Vec<_>)>>();

            debug!("load_all_label_and_motif_models: label={label} channels_motifs_models={channels_motifs_models:?}");

            labels_channels_motifs_models.insert(label.clone(), channels_motifs_models);
        }
    }
    (label_models, labels_channels_motifs_models)
}

pub fn run_train_label_nodes(
    select_motifs_for_label_run_ids: Option<&Vec<usize>>,
    run_configs: Vec<TrainLabelNodeRunConfig>,
    resource_config: &NetResources<Labeled>,
    common_config: &CommonTrainLabelNodeRunConfig,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Mutex<Box<dyn Store>>>,
    loader: &Arc<RwLock<LoadRunner<Box<Tensor>>>>,
    devices: Vec<Device>,
) -> Result<Option<Vec<Box<dyn SavableOptResult>>>, ThoosaError> {
    debug!("run_train_label_nodes: run_configs={run_configs:?} resource_config={resource_config:?} cuda_mem={:?}", cuda_mem());

    let (train_labels_resource_keys, valid_label_resource_keys) = resource_config.to_resources(store)?;

    debug!(
        "run_train_label_nodes: train_resource_keys={:?} valid_resource_keys={:?}",
        train_labels_resource_keys
            .iter()
            .map(|(label, keys)| (label, keys.len()))
            .collect::<Vec<_>>(),
        valid_label_resource_keys
            .iter()
            .map(|(label, keys)| (label, keys.len()))
            .collect::<Vec<_>>()
    );
    trace!("run_train_label_nodes: train_labels_resource_keys={train_labels_resource_keys:?}");
    trace!("run_train_label_nodes: valid_label_resource_keys={valid_label_resource_keys:?}");

    let valid_datasets_map = valid_label_resource_keys
        .into_iter()
        .fold(HashMap::new(), |mut a, v| {
            assert!(v.1.len() > 0, "no resources for {}", v.0);
            a.insert(v.0.clone(), v.1.clone());
            a
        });

    let select_motifs_for_label_run_ids: &Vec<usize> =
        select_motifs_for_label_run_ids.expect("select_motifs_for_label_run_ids");

    let chunk_len = max(1, train_labels_resource_keys.len() / devices.len());

    debug!("run_train_label_nodes: select_motifs_for_label_run_ids={select_motifs_for_label_run_ids:?} run_configs={run_configs:?} devices={devices:?} chunk_len={chunk_len}");


    let channels = ws.read().channels() as u32;

    let devices__resources = devices
        .into_iter()
        .cycle()
        .zip(train_labels_resource_keys.chunks(chunk_len).into_iter())
        .collect::<Vec<(Device, &[(String, Vec<Resource>)])>>();

    let run_configs_arc = Arc::new(run_configs); //.into_iter().map(|c|Arc::new(c)).collect::<Vec<_>>());
    let select_motifs_for_label_run_ids__channels_motifs_models = select_motifs_for_label_run_ids
        .iter()
        .map(|select_motifs_for_label_run_id| {

            let find_motifs_run_config_ids = store.lock().get_input_run_ids(*select_motifs_for_label_run_id, Some(&FindMotifsRunConfig::type_name())).expect(format!("FindMotifsRunConfig for select_motifs_for_label_run_id={select_motifs_for_label_run_id:?} not found").as_str());
            assert_eq!(find_motifs_run_config_ids.len(), 1);
            let find_motifs_run_config = FindMotifsRunConfig::from_properties_map(find_motifs_run_config_ids[0], store.lock().list_run_config_values(find_motifs_run_config_ids[0]).expect("list_run_params"));

            (
                *select_motifs_for_label_run_id,

                Arc::new(RwLock::new(

                    store
                        .lock()
                        .list_selected_motifs(*select_motifs_for_label_run_id, channels)
                        .expect("list_selected_motifs") //TODO move out
                        .into_iter()
                        .enumerate()
                        .map(|(channel, channels_motifs)| {
                            let ws_moved = Arc::clone(&ws);
                            (
                            find_motifs_run_config.clone(),
                            channels_motifs
                                .into_iter()
                                .flat_map(move |(motif_id, corr)| {
                                    load_motif_models(&ws_moved, channel, motif_id, &Cpu)
                                        .expect("load_motif_models")
                                })
                                .collect()
                            )
                        })
                        .collect()

                )),
            )
        })
        .collect::<Vec<(usize, Arc<RwLock<Vec<(FindMotifsRunConfig, Vec<(MotifNodeParams, ThoModuleWithStore)>)>>>)>>();

    devices__resources
        .into_par_iter()
        .map(move |(device_i, train_resource_keys_chunk)| {

            let run_configs_arc = Arc::clone(&run_configs_arc);
            let valid_datasets_map_clone = valid_datasets_map.clone(); //TODO


            train_resource_keys_chunk.iter().flat_map( |(label, resource_keys)| {
                debug!("run_train_label_nodes: label={label} resource_keys={resource_keys:?} cuda_mem={:?}", cuda_mem());

                let train_resource_keys_len = resource_keys.len();
                let valid_resource_keys = valid_datasets_map_clone.get(label).expect(format!("valid dataset for {}", label).as_str()).clone();
                let valid_resource_keys_len = valid_resource_keys.len();


                let train_labels_resource_rc = Rc::new(vec![
                    (1, resource_keys.clone()),
                    (0, store.lock().list_resource_random_not_label(label.as_str(), train_resource_keys_len as i64).expect("list_resource_random_not_label for train"))
                ]);
                let valid_labels_resources_rc = Rc::new(vec![
                    (1, valid_resource_keys),
                    (0, store.lock().list_resource_random_not_label(label.as_str(), valid_resource_keys_len as i64).expect("list_resource_random_not_label for valid"))
                ]);

                debug!("run_train_label_nodes: label={label} \ntrain_labels_resource_rc[0]={:?} \ntrain_labels_resource_rc[1]={:?} cuda_mem={:?}", train_labels_resource_rc[0], train_labels_resource_rc[1], cuda_mem());


                let run_configs_arc = Arc::clone(&run_configs_arc);

                select_motifs_for_label_run_ids__channels_motifs_models.iter().flat_map(move |(select_motifs_for_label_run_id, channels_motifs_models)| {
                    let find_motifs_for_label_run_ids = store.lock().get_input_run_ids(*select_motifs_for_label_run_id, Some(&FindMotifsRunConfig::type_name())).expect(format!("FindMotifsRunConfig for select_motifs_for_label_run_id={select_motifs_for_label_run_id} not found").as_str());
                    assert_eq!(find_motifs_for_label_run_ids.len(), 1);
                    let find_motifs_run_config = FindMotifsRunConfig::from_properties_map(find_motifs_for_label_run_ids[0], store.lock().list_run_config_values(find_motifs_for_label_run_ids[0]).expect("list_run_params"));


                    let motif_width = find_motifs_run_config.window(); //channels_motifs_models.read().iter().flat_map(|v|v.iter().map(|v|v.0.num_features())).next().expect("no models"); //all channels motifs come from same selection/find_motifs
                    let slices_len = slices_len(ws.read().max_input() as i64, motif_width, find_motifs_run_config.step());
                    if log::log_enabled!(Level::Debug){
                        debug!("run_train_label_nodes: label={label} channels={channels} channels_motifs_models.len={:?}", channels_motifs_models.read().iter().map(|m|m.1.len()).collect::<Vec<_>>());
                    }
                    trace!("run_train_label_nodes: label={label} select_motifs_for_label_run_id={select_motifs_for_label_run_id} channels_motifs_models={channels_motifs_models:?}");

                    let channels_motifs_models_arc = Arc::clone(&channels_motifs_models);

                    let max_nets = channels_motifs_models.read().iter().map(|nets| nets.1.len()).sum::<usize>() as i64;
                    if max_nets <= 0 {
                        panic!("no motif nets found");//TODO skip?
                    }

                    debug!("run_train_label_nodes: label={label} cuda_mem={:?}", cuda_mem());

                    //TODO Rc
                    let train_labels_resources_clone = Rc::clone(&train_labels_resource_rc);
                    let valid_labels_resources_clone = Rc::clone(&valid_labels_resources_rc);

                    let mut results = Vec::with_capacity(run_configs_arc.len());
                    for run_config in run_configs_arc.iter() {
                        debug!("run_train_label_nodes: run_config={run_config:?}");

                        match run_opt_train_label_nodes(
                            ws,
                            &run_config,
                            common_config,
                            Rc::clone(&train_labels_resources_clone),
                            Rc::clone(&valid_labels_resources_clone),
                            &device_i,
                            &loader,
                            Arc::clone(&channels_motifs_models_arc),
                            max_nets,
                            label,
                            100, //slices_len/3
                        ) {
                            Ok(result) => {
                                results.push(Ok(result));
                            }
                            Err(e) => {
                                warn!("run_train_label_nodes: run error={e:?}");
                                results.push(Ok(DummySavableOptResult::new()));
                            }
                        };

                    }
                    results
                })
            }).collect::<Result<Vec<_>, ThoosaError>>()
        })
        .collect::<Result<Vec<Vec<Box<dyn SavableOptResult>>>, ThoosaError>>().map(|v| v.into_iter()
        .flat_map(|v|v.into_iter())
        .collect::<Vec<Box<dyn SavableOptResult>>>()).map(|v|Some(v))
}

/**
returns motif_node_params_selection_run_id
*/
fn run_train_motif_nodes_on_motif_selection(
    motif_selection_run_ids: Option<&Vec<usize>>,
    train_motifs_nodes_run_configs: Vec<TrainMotifsNodesRunConfig>,
    _resource_config: &Empty,
    common_config: &CommonTrainMotifsNodesRunConfig,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Mutex<Box<dyn Store>>>,
    loader: &Arc<RwLock<LoadRunner<Box<Tensor>>>>,
    devices: Vec<Device>,
) -> Result<Option<Vec<Box<dyn SavableOptResult>>>, ThoosaError> {
    let motif_selection_run_ids: &Vec<usize> =
        motif_selection_run_ids.expect("motif_selection_run_ids");

    let chunk_len = max(1, motif_selection_run_ids.len() / devices.len());

    debug!("run_train_motif_nodes_on_motif_selection: motif_selection_run_ids={motif_selection_run_ids:?} train_motifs_nodes_run_config={train_motifs_nodes_run_configs:?} devices={devices:?} chunk_len={chunk_len}");

    let devices_motif_selection = devices
        .into_iter()
        .cycle()
        .zip(motif_selection_run_ids.chunks(chunk_len).into_iter())
        .collect::<Vec<(Device, &[usize])>>();

    devices_motif_selection
        .into_par_iter()
        .map( |(device_i, motif_selection_run_ids_chunk)| {

        motif_selection_run_ids_chunk.iter().flat_map(|motif_selection_run_id| {

            let find_motifs_run_ids = store.lock().get_input_run_ids(*motif_selection_run_id, None).expect("motif_selection_id failed");
            assert_eq!(find_motifs_run_ids.len(), 1);//selection has single input
            let find_motifs_run_id = find_motifs_run_ids[0];

            let motifs = store.lock().list_selected_motifs(*motif_selection_run_id, ws.read().channels() as u32).expect("list_selected_motifs")
                .into_iter()
                .flat_map(|channel_motifs| channel_motifs.into_iter())
                .map(|(motif_id, _corr)| motif_id).collect::<Vec<usize>>();

            let find_motifs_run_config = FindMotifsRunConfig::from_properties_map(find_motifs_run_id, store.lock().list_run_config_values(find_motifs_run_id).expect("list_run_params"));

            train_motifs_nodes_run_configs.iter().map(move | train_motifs_nodes_run_config | {

                debug!("run_train_motif_nodes_on_motif_selection: find_motifs_run_ids={find_motifs_run_ids:?} motif_selection_id={motif_selection_run_id}");

                run_train_motif_nodes(
                    &motifs,
                    find_motifs_run_ids[0],
                    train_motifs_nodes_run_config,
                    common_config,
                    &find_motifs_run_config,
                    ws,
                    Arc::clone(store),
                    loader,
                    device_i
                )
            })
        }).collect::<Result<Vec<Vec<Box<dyn SavableOptResult>>>, ThoosaError>>()

    })
    .collect::<Result<Vec<Vec<Vec<Box<dyn SavableOptResult>>>>, ThoosaError>>()
    .map(|v| v.into_iter()
            .flat_map(|v|v.into_iter()
                .flat_map(|v|v.into_iter()))
            .collect::<Vec<Box<dyn SavableOptResult>>>()).map(|v|Some(v)
    )
}

fn motifs_to_tensor_cmd(
    cli: &Cli,
    store: &Arc<Mutex<Box<dyn Store>>>,
    device: Device,
    choose_fragments_params: &MotifsToTensorParams,
    executor: &Arc<RwLock<LoadRunner<Box<Tensor>>>>,
) -> Result<(), ThoosaError> {
    let mut store_guard = store.lock();

    let t = motifs_to_tensor(
        store_guard.list_motifs_slices(choose_fragments_params.run_id())?,
        executor,
        choose_fragments_params.window(),
        20,
        device,
        0,
    )?;

    let out = Path::new(&cli.ws_path).join("out");
    fs::create_dir_all(&out)?;
    t.channels_motifs_slices
        .save(out.join(format!("motifs_in_{}", choose_fragments_params.run_id())))?;

    Ok(())
}

pub fn import_resources<F, R>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Mutex<Box<dyn Store>>>,
    labels: Vec<&str>,
    labeled_resources: F,
    resource_to_tensor: R, //None for tests - do not import parquet files
    insert_label: bool,
) -> Result<Vec<Resource>, ThoosaError>
where
    F: Fn() -> Result<HashMap<String, Vec<(String, f32)>>, ThoosaError>,
    R: Fn(String) -> Result<Tensor, ThoosaError> + Sync + Send,
{
    let mut store_guard = store.lock();

    let mut conn_guard = store_guard.conn().lock();
    let t = conn_guard.transaction()?;

    if insert_label {
        for label in labels {
            store_guard.save_label(label, &t)?; //TODO merge with save_resources_labels
        }
    }

    let labeled_resources_map = labeled_resources()?;
    let ws_guard = ws.write();
    let import_id = store_guard.save_import(format!("Import on {}", cdt()), &t)?;

    let (imported_tensors_stats, failed_tensors): (
        HashMap<String, (ResourceStats, Vec<(String, f32)>)>,
        Vec<String>,
    ) = labeled_resources_map
        .into_iter()
        .par_bridge()
        .fold(
            || (HashMap::new(), vec![]),
            |mut a, (resource_key, probs)| {
                trace!(
                    "import_resources: resource_key={resource_key} ok={} err={}",
                    a.0.len(),
                    a.1.len()
                );

                match resource_to_tensor(resource_key.clone())
                    .and_then(|t| ws_guard.import_tensor(t, import_id, resource_key.as_str()))
                {
                    Ok(stats) => {
                        assert!(a.0.insert(resource_key.clone(), (stats, probs)).is_none());
                        //resource key should not repeat
                    }
                    Err(err) => {
                        error!("Can't load {resource_key}: {err:?}");
                        a.1.push(resource_key.clone());
                    }
                }

                a
            },
        )
        .reduce_with(|mut v1, v2| {
            v1.0.extend(v2.0); //TODO assert no keys overlaps
            v1.1.extend(v2.1);
            v1
        })
        .expect("imported_tensors");

    let resource_ids = store_guard.save_resources_labels(&imported_tensors_stats, import_id, &t)?;

    let imported_resources: Vec<Resource> = imported_tensors_stats
        .keys()
        .zip(resource_ids.into_iter())
        .map(|(resource_key, resource_id)| {
            Resource::new(resource_id, import_id, resource_key.clone(), 0)
        })
        .collect();

    trace!("import_resources: imported_resources={imported_resources:?}, failed_tensors={failed_tensors:?}");

    info!(
        "Import id: {}, imported {} tensors to {:?}, failed {}",
        import_id,
        imported_resources.len(),
        ws_guard.tensors_path(),
        failed_tensors.len()
    );

    store_guard.save_pure_sets(3, true, &t)?;
    store_guard.save_pure_sets(10, true, &t)?;
    store_guard.save_pure_sets(100, true, &t)?;
    store_guard.save_pure_sets(200, true, &t)?;
    store_guard.save_pure_sets(400, true, &t)?;

    store_guard.save_groups_by_label_cnt(100, &t)?;
    t.commit()?;

    Ok(imported_resources) //for tests
}

fn run_plot_motif_slices(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Mutex<Box<dyn Store>>>,
    loader: &Arc<RwLock<LoadRunner<Box<Tensor>>>>,
    plot_motif_slices_params: &PlotMotifSlicesParams,
    device: &Device,
) -> Result<(), ThoosaError> {
    match plot_motif_slices_params.motifs_by() {
        MotifsBy::MotifId(motif_id) => {
            debug!("run_plot_motif_slices: MotifsBy::MotifId run_id={motif_id}");

            find::plot_motif_slices(ws, store, loader, motif_id, device)?;
        }
        MotifsBy::RunId(run_id) => {
            debug!("run_plot_motif_slices: MotifsBy::RunId run_id={run_id}");
            fs::create_dir_all(ws.read().plots_path().join(run_id.to_string()))?;

            //TODO add list for  motif_ids only without grouping by  slice_matches
            let motif_distances = store.lock().list_motifs(run_id)?;
            let motifs_len = motif_distances.len();
            let mut done_cnt = AtomicU32::new(0);
            motif_distances /*.into_iter()*/
                .into_par_iter()
                .for_each(|motif_id| {
                    //TODO par in devices
                    debug!("run_plot_motif_slices: motif_id={motif_id}");
                    find::plot_motif_slices(ws, store, loader, motif_id, device).unwrap();
                    let done = done_cnt.fetch_add(1, Relaxed);
                    if done % 10 == 0 {
                        info!("done {done} from  {motifs_len}");
                    }
                });
        }
    }

    Ok(())
}

fn run_train_motif_nodes(
    motifs: &Vec<usize>,
    find_motifs_run_id: usize,
    train_on_motif_run_config: &TrainMotifsNodesRunConfig,
    common_config: &CommonTrainMotifsNodesRunConfig,
    find_motifs_run_config: &FindMotifsRunConfig,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<Mutex<Box<dyn Store>>>,
    loader: &Arc<RwLock<LoadRunner<Box<Tensor>>>>,
    device: Device,
) -> Result<Vec<Box<dyn SavableOptResult>>, ThoosaError> {
    debug!("train_on_motifs: device={device:?}, find_motifs_run_id={find_motifs_run_id}");

    motifs.into_iter().map(|motif_id| {
        let (all_slices_0, all_slices_1, channel) = {
            let mut store_guard = store.lock();

            let (channel, run_id, slices, resources_map) = store_guard.list_motif_slices(*motif_id, None, None)?;

            assert_eq!(run_id, find_motifs_run_id);

            debug!("train_on_motif: 1. channel={channel}, run_id={run_id}, slices.len={}", slices.len());
            trace!("train_on_motif: 1. slices={slices:?}, resources_map={resources_map:?}");

            let all_slices_1 = train_nets::slices_in_all_resources(
                load_motif_slices(channel, find_motifs_run_config.window(), slices, resources_map, loader)
            );

            //list motifs which are NOT motif_id and NOT from run_id
            let (channel, run_id, slices, resources_map) = store_guard.list_motif_slices(*motif_id, Some(all_slices_1.len()), Some((find_motifs_run_id, channel as usize)))?;

            debug!("train_on_motif: 0. channel={channel}, run_id={run_id}, slices.len={}", slices.len());
            trace!("train_on_motif: 0. slices={slices:?}, resources_map={resources_map:?}");

            let all_slices_0 = train_nets::slices_in_all_resources(
                load_motif_slices(channel, find_motifs_run_config.window(), slices, resources_map, loader)
            );

            debug!("train_on_motif: 0. run_id={}, done", run_id);

            (all_slices_0, all_slices_1, channel)
        };

        assert!(all_slices_1.len() > 0, "all_slices_1.len(), motif={motif_id}");
        assert!(all_slices_0.len() > 0, "all_slices_0.len(), motif={motif_id}");

        let batch_size = common_config.batch_size; //TODO, purge small motifs
        if all_slices_1.len() < batch_size {
            warn!("num of motif_id {motif_id} all_slices_1.len()={} in channel {channel} is less than {batch_size}", all_slices_1.len());
            return Err(ThoosaError::new("skip"));
        }

        if all_slices_0.len() < batch_size {
            warn!("num of not motif_id {motif_id} all_slices_0.len()={} in channel {channel} is less than {batch_size}", all_slices_0.len());
            return Err(ThoosaError::new("skip"));
        }

        debug!("train_on_motif:  all_slices_0.len()={:?}, all_slices_0[0].size()={:?}, all_slices_1.len()={:?}, all_slices_1[0].size()={:?}", all_slices_0.len(), all_slices_0[0].size(), all_slices_1.len(), all_slices_1[0].size());


        train_nets::run_opt_train_motif(
            *motif_id,
            channel as u32,
            all_slices_0.iter().map(|v|Box::new(v.shallow_clone())).collect(), //TODO change ClassTensorsDataBatch to accept refs
            all_slices_1.iter().map(|v|Box::new(v.shallow_clone())).collect(),
            train_on_motif_run_config,
            find_motifs_run_config.window(),
            common_config,
            device,
        )

    }).filter(|v|v.is_ok()).collect()
}

fn resource_path(base_path: &str, resource_pattern: &str, resource_key: &str) -> String {
    //TODO change to impl AsRef<Path> + in ParquetLoader
    Path::new(base_path)
        .join(resource_pattern.replace("{}", resource_key))
        .into_os_string()
        .into_string()
        .unwrap()
}

// P - params
// RC - RunConfig
// R  - Resources Config
// CC  - CommonConfig - same for all run configurations
// ST - Step Fn
// run each own run config with all input run configs
pub fn run_step<'a, P, RC, R, CC, ST>(
    input_run_ids: Option<Vec<usize>>,
    params: &P,
    step: ST,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Mutex<Box<dyn Store>>>,
    exec: &Arc<RwLock<LoadRunner<Box<Tensor>>>>,
    devices: Vec<Device>,
    descr: &str,
) -> Result<Vec<usize>, ThoosaError>
where
    ST: FnOnce(
        Option<&Vec<usize>>,
        Vec<RC>,
        &R,
        &CC,
        &Arc<RwLock<Box<dyn Workspace>>>,
        &Arc<Mutex<Box<dyn Store>>>,
        &Arc<RwLock<LoadRunner<Box<Tensor>>>>,
        Vec<Device>,
    ) -> Result<Option<Vec<Box<dyn SavableOptResult>>>, ThoosaError>,
    P: Params<RC, R, CC>,
    RC: RunConfig,
    R: ResourcesConfig,
{
    let run_configs = params.to_run_configs();
    let resource_config = params.resource_config();
    let resources_serialized = resource_config.serialize();

    let mut run_config_ids = vec![];

    let run_configs = run_configs
        .into_iter()
        .map(|mut run_config| {
            let run_config_id = store
                .lock()
                .save_run_config(
                    input_run_ids.as_deref(),
                    &RC::type_name(),
                    descr,
                    &resources_serialized,
                    run_config.to_properties_map(),
                )
                .unwrap();

            run_config_ids.push(run_config_id);
            run_config.set_id(run_config_id);

            run_config
        })
        .collect::<Vec<RC>>();

    debug!("run_step: run_configs={run_configs:?}");

    let mut opt_run_configs_opt = step(
        input_run_ids.as_ref(),
        run_configs,
        &resource_config,
        &params.common_config(),
        ws,
        store,
        exec,
        devices,
    )?;

    if let Some(mut opt_run_configs) = opt_run_configs_opt {
        debug!(
            "run_step: saving opt_run_configs.len={}",
            opt_run_configs.len()
        );
        let mut opt_run_config_ids = vec![];
        for mut opt_run_config in opt_run_configs {
            debug!("run_step: saving opt_run_config={opt_run_config:?}");

            let opt_run_config_id = opt_run_config.save(store, ws)?;

            opt_run_config_ids.push(opt_run_config_id);
        }
        Ok(opt_run_config_ids)
    } else {
        Ok(run_config_ids)
    }
}

#[cfg(test)]
mod cli_tests {
    use crate::cli::Commands::FindMotifs;
    use crate::cli::{
        import_resources, resource_path, run_infer, run_plot_motif_slices, run_step,
        run_train_final_node, run_train_label_nodes, run_train_motif_nodes_on_motif_selection, Cli,
        RunAllParams,
    };
    use crate::data::ParquetLoader;
    use log::LevelFilter::{Debug, Trace, Warn};
    use log::{debug, info, trace, LevelFilter};
    use parking_lot::lock_api::MutexGuard;
    use parking_lot::{Mutex, RawMutex, RwLock};
    use plotters::prelude::{IntoLinspace, BLACK};
    use polars::export::arrow::array::clone;
    use polars::export::chrono;
    use rand::Rng;
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    use rayon::prelude::IntoParallelIterator;
    use simplelog::CombinedLogger;
    use std::cmp::{max, min};
    use std::collections::{HashMap, HashSet};
    use std::io::Read;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::vec::IntoIter;
    use std::{fs, ptr};
    use clap::Parser;
    use tch::display::{set_print_options, PrinterOptions};
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::Device::{Cpu, Cuda};
    use tch::{nn, Device, IndexOp, Tensor};
    use thoosa_common::input_probs::{compare_input_probs, InputProbs};
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::runs::RunConfig;
    use thoosa_common::runs::Params;
    use thoosa_common::train_nets::{motif_node, FindMotifsRunConfig, InferParams, MotifNodeParams, TrainFinalNetParams, TrainLabelNetsParams, TrainMotifsNetsParams};
    use thoosa_common::utils::{
        from, lin_t, plot_series, reset_print_options, t_0_0, t_0_1, t_1_0, t_1_1,
        three_sigs_tensor, vec_f32_from,
    };
    use thoosa_common::ws::{FsWorkspace, Workspace};
    use thoosa_motifs::distance::clc_dist_d2_and_mask;
    use thoosa_motifs::distance::MaskMaker::LeQuantile;
    use thoosa_motifs::find::{
        find_motifs_by_euc_dist, select_motifs_for_label, FindMotifsParams,
        SelectMotifsParams,
    };
    use thoosa_motifs::fragments::{MotifsBy, PlotMotifSlicesParams};
    use thoosa_shared::resource::{Resource, ResourceId, ResourceStats, ResourcesConfig};
    use thoosa_shared::val::unprefix_properties;
    use thoosa_shared::{
        cdt, cdt_for_dir, random_saw, recurse_files, saw, saw_with_steps, sig_one_imp,
        sig_three_imp, sig_two_imp, ResourceType, TensorIdSliceIdx, ThoosaError,
    };
    use thoosa_store::{MotifTrackerStore, SlicesMatch, SqlMotifTrackerStore, SqlStore, Store};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_resource_path() {
        assert_eq!(
            resource_path("/xxx/yyy", "zzz/{}.parquet", "aaa"),
            "/xxx/yyy/zzz/aaa.parquet".to_string()
        )
    }

    #[test]
    fn test_motifs() -> Result<(), ThoosaError> {
        // log::set_max_level(LevelFilter::Trace);

        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_motifs: test_dir={test_dir:?}");

        fs::create_dir_all(&test_dir)?;
        let base_path = test_dir.to_string_lossy().to_string();

        let n = 1000i64;

        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            test_dir.clone(),
            3,
            n as usize,
        )?));
        let ws_rc = Arc::clone(&ws);
        let max_y = 30.0;

        let sig_0_v = sig_one_imp(n, max_y, None, None, 0.1);
        let sig_1_v = sig_two_imp(n, max_y, None, None, 0.1);
        let sig_2_v = sig_three_imp(n, max_y, None, None);

        /*
        let x = lin_t(n);
        plot_series(
            vec![
                (format!("Sig0"), plotters::prelude::RED, vec_f32_from(&x).into_iter().zip(sig_0_v.clone()).collect::<Vec<(f32, f32)>>()),
                (format!("Sig1"), plotters::prelude::GREEN, vec_f32_from(&x).into_iter().zip(sig_1_v.clone()).collect::<Vec<(f32, f32)>>()),
                (format!("Sig2"), plotters::prelude::BLACK, vec_f32_from(&x).into_iter().zip(sig_2_v.clone()).collect::<Vec<(f32, f32)>>()),

            ], "test_find_motifs_euc", (2000, 1000), &format!("/tmp/plots/0_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str()
        )?;*/

        let (t_0_0, t_0_1, t_1_0, t_1_1) = three_sigs_tensor(n, &sig_0_v, &sig_1_v, &sig_2_v);
        debug!(
            "t_0_0.size()={:?}, t_0_1.size()={:?}, t_1_0.size()={:?}, t_1_1.size()={:?}",
            t_0_0.size(),
            t_0_1.size(),
            t_1_0.size(),
            t_1_1.size()
        );

        t_0_0.save(ws.read().tensors_path().join(format!("t_0_0.pt")))?;
        t_0_1.save(ws.read().tensors_path().join(format!("t_0_1.pt")))?;
        t_1_0.save(ws.read().tensors_path().join(format!("t_1_0.pt")))?;
        t_1_1.save(ws.read().tensors_path().join(format!("t_1_1.pt")))?;

        // debug!("t_1_1 mean={} max-min={}", t_1_1.mean(None).double_value(&[]), t_1_1.max() - t_1_1.min());
        let channels = 3;
        let tensors_path = ws.read().tensors_path().clone();
        let tensors_path_for_import = ws.read().tensors_path().clone();

        let loader = Arc::new(RwLock::new(LoadRunner::new(
            move |key: &Resource| {
                SyncWrapper::new_tb(Box::new(
                    Tensor::load(tensors_path.join(format!("{}.pt", key.file_stem()))).unwrap(),
                ))
            },
            4,
            100,
        )));

        let mut store = Arc::new(Mutex::new(SqlStore::from_path(ws.read().store_path())?));

        let resources = import_resources(
            &ws,
            &store,
            vec!["label_0", "label_1"],
            || {
                Ok(HashMap::from([
                    ("t_0_0".to_string(), vec![("label_0".to_string(), 1.0)]),
                    ("t_0_1".to_string(), vec![("label_0".to_string(), 1.0)]),
                    ("t_1_0".to_string(), vec![("label_1".to_string(), 1.0)]),
                    ("t_1_1".to_string(), vec![("label_1".to_string(), 1.0)]),
                ]))
            },
            // None::<Box<dyn Fn(String) -> Result<Tensor, ThoosaError> + Sync + Send>>,
            |key| {
                Tensor::load(tensors_path_for_import.join(format!("{}.pt", key)))
                    .map_err(|e| ThoosaError::new(e.to_string()))
            },
            true,
        )?;

        let resources_0_0 = resources
            .iter()
            .find(|r| r.file_stem().eq("t_0_0"))
            .expect("t_0_0");
        let resources_0_1 = resources
            .iter()
            .find(|r| r.file_stem().eq("t_0_1"))
            .expect("t_0_1");
        let resources_1_0 = resources
            .iter()
            .find(|r| r.file_stem().eq("t_1_0"))
            .expect("t_1_0");
        let resources_1_1 = resources
            .iter()
            .find(|r| r.file_stem().eq("t_1_1"))
            .expect("t_1_1");

        debug!("test_motifs: resources_0_0={resources_0_0:?}");
        let run_ids = run_step(
            None,
            &FindMotifsParams {
                step: 200.into(),
                window: (n as u32).into(),
                resources: ResourceType::List(vec![
                    "t_0_0".into(),
                    "t_0_1".into(),
                    "t_1_0".into(),
                    "t_1_1".into(),
                ]),
                min_tracker_step: Some(3),
                chunk_size: 2,
                dist_quantile: 0.01,
                keep_slice_matches: false,
                max_slices_k: Some(0.1),
                mean_k: Some(0.1),
                input_filter: None,
            },
            find_motifs_by_euc_dist,
            &ws_rc,
            &store,
            &loader,
            vec![Cpu],
            "Test/FindMotifs",
        )?;

        let mut slices = store.lock().list_motifs_slices(run_ids[0])?;

        debug!("slices={slices:?}");

        // debug!("test_motifs: len={:?} slices_0_0={:?}", slices.get(resources_0_0).map(|v|v.len()), slices.get(resources_0_0));
        // debug!("test_motifs: len={:?} slices_0_1={:?}", slices.get(resources_0_1).map(|v|v.len()), slices.get(resources_0_1));
        // debug!("test_motifs: len={:?} slices_1_0={:?}", slices.get(resources_1_0).map(|v|v.len()), slices.get(resources_1_0));
        // debug!("test_motifs: len={:?} slices_1_1={:?}", slices.get(resources_1_1).map(|v|v.len()), slices.get(resources_1_1));
        assert!(slices.len() >= 4);

        let mut resource_0_0_slices = slices.remove(resources_0_0).expect("resources_0_0 slices");
        resource_0_0_slices.sort_by(|a, b| a.slice_idx.cmp(&b.slice_idx));

        let mut resource_0_1_slices = slices.remove(resources_0_1).expect("resources_0_1 slices");
        resource_0_1_slices.sort_by(|a, b| a.slice_idx.cmp(&b.slice_idx));

        let mut resource_1_0_slices = slices.remove(resources_1_0).expect("resources_0_1 slices");
        resource_1_0_slices.sort_by(|a, b| a.slice_idx.cmp(&b.slice_idx));

        let mut resource_1_1_slices = slices.remove(resources_1_1).expect("resources_0_1 slices");
        resource_1_1_slices.sort_by(|a, b| a.slice_idx.cmp(&b.slice_idx));

        debug!("resource_0_0_slices={resource_0_0_slices:?}");
        debug!("resource_0_1_slices={resource_0_1_slices:?}");
        debug!("resource_1_0_slices={resource_1_0_slices:?}");
        debug!("resource_1_1_slices={resource_1_1_slices:?}");
        //TODO

        // let slice_0_0 = slices.get(resources_0_0).unwrap().into_iter().find(|s|s.channel == 0).expect("0_0");
        // assert_eq!(slice_0_0.slice_idx, 0);
        // let slice_0_1 = slices.get(resources_0_1).unwrap().into_iter().find(|s|s.channel == 0).expect("0_1");
        // assert_eq!(slice_0_1.slice_idx, 1000);
        //
        //
        // let slice_1_0 = slices.get(resources_1_0).unwrap().into_iter().find(|s|s.channel == 1).expect("1_0");
        // assert_eq!(slice_1_0.slice_idx, 1000);
        // let slice_1_1 = slices.get(resources_1_1).unwrap().into_iter().find(|s|s.channel == 1).expect("1_1");
        // assert_eq!(slice_1_1.slice_idx, 2000);
        //
        // let motif_dists = store.lock().list_motifs(run_ids[0])?;
        // debug!("test_motifs: motif_dists={motif_dists:?}");
        //
        // for motif_id in motif_dists.keys() {
        //     let (channel, run_id, motif_slice) = store.lock().list_motif_slices(*motif_id, None, None)?;
        //     debug!("test_motifs: motif_id={motif_id} channel={channel} motif_slice={motif_slice:?}");
        // }

        // assert_eq!((110, 0), slices.get(&"t_0".to_string()).and_then(|v| v.get(0).map(|v|(v.slise_idx, v.channel_idx))).unwrap());
        // assert_eq!(&MotifSlice { motif_id: 1, slise_idx: 1100, channel_idx: 0 }, slices.get(&"t_1".to_string()).and_then(|v| v.get(0)).unwrap());

        /* let select_motifs_run_ids = select_motifs_for_label(
            &ws_rc,
            Arc::clone(&store),
            Cpu,
            &SelectMotifsParams::dummy(
                run_ids.clone(),
                "group".into(),
                0.1
            ),
            run_ids[0],
            &loader
        ).expect("select_motifs_in_class");

        debug!("test_motifs_with_tpe: select_motifs_run_ids={select_motifs_run_ids:?}");
        let motif_selection_id = store.lock().list_input_run_ids(&select_motifs_run_ids)?;
        debug!("test_motifs_with_tpe: motif_selection_id={motif_selection_id:?}");

        // store.lock().list_selected_motifs();

        // there are motifs with corr 1.0 for each label
        let selected_motifs_0 = store.lock().list_selected_motifs(
            select_motifs_run_ids[0]
            // select_motifs_run_ids.iter().find(|(id, run_id)|label.eq("label_0")).expect("label_0").1
            //0 channel
        )?;
        debug!("selected_motifs_0={selected_motifs_0:?}");
        //TODO
        // assert!(selected_motifs_0[0].iter().find(|(motif_id, corr)|corr == &1.0).is_some());

        let selected_motifs_1 = store.lock().list_selected_motifs(
            select_motifs_run_ids[1]
            // corrs.iter().find(|(id, run_id)|label.eq("label_1")).expect("label_1").1
        )?;
        debug!("selected_motifs_1={selected_motifs_1:?}");
        // assert!(selected_motifs_1[0].iter().find(|(motif_id, corr)|corr == &1.0).is_some());*/

        Ok(())
    }

    fn init_test_store(
        mut store: &Arc<Mutex<Box<dyn Store>>>,
        tensors_per_label: i32,
    ) -> Result<Vec<ResourceId>, ThoosaError> {
        let store_guard = store.lock();
        let mut mutex_guard = store_guard.conn().lock();
        let t = mutex_guard.transaction()?;

        store_guard.save_label("Label_0", &t)?;
        store_guard.save_label("Label_1", &t)?;

        let import_id = store_guard.save_import(format!("test on {}", cdt()), &t)?;

        let resources = store_guard.save_resources_labels(
            &(0..tensors_per_label)
                .map(|i| {
                    (
                        format!("t_0_{i}"),
                        (
                            ResourceStats::new(vec![1, 1000]),
                            vec![("Label_0".into(), 1.0)],
                        ),
                    )
                })
                .chain((0..tensors_per_label).map(|i| {
                    (
                        format!("t_1_{i}"),
                        (
                            ResourceStats::new(vec![1, 1000]),
                            vec![("Label_1".into(), 1.0)],
                        ),
                    )
                }))
                .collect(),
            import_id,
            &t,
        )?;

        store_guard.save_pure_sets(tensors_per_label as usize / 10, true, &t)?;
        store_guard.save_pure_sets(tensors_per_label as usize, true, &t)?;
        // store_guard.save_pure_sets(4, true, &t)?;
        // store_guard.save_pure_sets(6, true, &t)?;

        t.commit()?;

        Ok(resources)
    }

    fn plot_t(t_0: &Tensor, label: &str) -> Result<(), ThoosaError> {
        plot_series(
            (0..1)
                .into_iter()
                .map(|channel| {
                    (
                        format!("0"),
                        BLACK,
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
            &format!(
                "/tmp/plots/{label}_{:?}.png",
                chrono::Utc::now().to_rfc3339()
            )
            .as_str(),
        )?;
        Ok(())
    }

    fn label_1_t(n: i64, max_y: f32) -> Tensor {
        Tensor::vstack(&[Tensor::from_slice(
            [
                sig_one_imp(n, max_y, Some(10. ..12.), Some(10. ..12.), 0.1),
                sig_one_imp(n, max_y, Some(10. ..12.), Some(10. ..12.), 0.1),
            ]
            .concat()
            .as_slice(),
        )])
    }

    fn label_0_t(n: i64, max_y: f32) -> Tensor {
        Tensor::vstack(&[Tensor::from_slice(
            [
                sig_three_imp(n, max_y, Some(10. ..12.), Some(10. ..12.)),
                sig_two_imp(n, max_y, Some(10. ..12.), Some(10. ..12.), 0.1),
            ]
            .concat()
            .as_slice(),
        )])
    }

    fn save_test_motifs(
        resource_0_ids: &Vec<ResourceId>,
        store: Arc<Mutex<Box<dyn Store>>>,
        find_motifs_params: FindMotifsParams,
        param_set: Vec<FindMotifsRunConfig>,
        resource_label_key: &str,
    ) -> Result<usize, ThoosaError> {
        // let resource_0_ids:Vec<ResourceId> = (0..10).map(|i| i as ResourceId).collect();
        let run_id = store.lock().save_run_config(
            None,
            &"todo",
            format!(
                "find motifs with params {:?}/{:?}",
                find_motifs_params, param_set[0]
            )
            .as_str(),
            "".into(), //&resource_0_ids,
            param_set[0].to_properties_map(),
        )?;

        let slices = resource_0_ids
            .windows(2)
            .into_iter()
            .flat_map(|resource_id| {
                [
                    SlicesMatch(
                        TensorIdSliceIdx(resource_id[0], 100),
                        TensorIdSliceIdx(resource_id[0], 1100),
                        1.0,
                    ),
                    SlicesMatch(
                        TensorIdSliceIdx(resource_id[0], 100),
                        TensorIdSliceIdx(resource_id[1], 100),
                        1.0,
                    ),
                    SlicesMatch(
                        TensorIdSliceIdx(resource_id[0], 100),
                        TensorIdSliceIdx(resource_id[1], 1100),
                        1.0,
                    ),
                    SlicesMatch(
                        TensorIdSliceIdx(resource_id[1], 100),
                        TensorIdSliceIdx(resource_id[1], 1100),
                        1.0,
                    ),
                ]
            })
            .collect::<Vec<_>>();
        // store.lock().save_slices(
        //     run_id,
        //     0i64,
        //     slices
        // )?; //all tensors contain two signals - [0-1000) and [1000-2000)

        let parent_path = Path::new("/tmp/").join(cdt_for_dir());
        fs::create_dir_all(&parent_path);
        let test_db = parent_path.join("slices_db.db3");
        let mut slices_store =
            SqlMotifTrackerStore::new(&test_db, run_id, 0).expect("can't create tmp store");

        slices_store.save_slices(slices.clone())?;

        let groups = slices_store.load_slices_into_groups(&store, None)?;

        Ok(run_id)
    }

    #[test]
    fn slow_test_run_step() -> Result<(), ThoosaError> {
        unsafe {
            let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        // log::set_max_level(Warn);
        let max_y = 1f32;

        // let (tensors_per_label, cycles, opt_max_evals, opt_n_init, motif_batch_size) = (20, 50, 20, 10, 10);
        let (tensors_per_label, cycles, opt_max_evals, opt_n_init, motif_batch_size) = (10, 10, 3, 1, 3);
        // debug!("test_run_step: tensors_per_label={tensors_per_label}, cycles={cycles}, opt_max_evals={opt_max_evals}, opt_n_init={opt_n_init}, motif_batch_size={motif_batch_size}");

        fn _saw(n: i64) -> Vec<f32> {
            saw_with_steps(n as usize, 10, 5.)
        }

        fn label_0_t_rs(n: i64, sig_0_v: &Vec<f32>, sig_2_v: &Vec<f32>, max_y: f32) -> Tensor {
            Tensor::vstack(&[
                Tensor::from_slice(
                    [sig_0_v.clone(), _saw(n), sig_0_v.clone()]
                        .concat()
                        .as_slice(),
                ),
                // Tensor::from_slice([sig_2_v.clone(), _saw(n),          sig_2_v.clone()].concat().as_slice()),
                // Tensor::from_slice([_saw(n),          sig_2_v.clone(), _saw(n)].concat().as_slice()),
                Tensor::from_slice([_saw(n), _saw(n), _saw(n)].concat().as_slice()),
                Tensor::from_slice([_saw(n), _saw(n), _saw(n)].concat().as_slice()),
            ])
        }

        fn label_1_t_rs(n: i64, sig_1_v: &Vec<f32>, sig_2_v: &Vec<f32>, max_y: f32) -> Tensor {
            Tensor::vstack(&[
                Tensor::from_slice(
                    [sig_1_v.clone(), _saw(n), sig_1_v.clone()]
                        .concat()
                        .as_slice(),
                ),
                // Tensor::from_slice([sig_2_v.clone(), _saw(n),          sig_2_v.clone()].concat().as_slice()),
                // Tensor::from_slice([_saw(n),          sig_2_v.clone(), _saw(n)].concat().as_slice())
                Tensor::from_slice([_saw(n), _saw(n), _saw(n)].concat().as_slice()),
                Tensor::from_slice([_saw(n), _saw(n), _saw(n)].concat().as_slice()),
            ])
        }

        let n = 1000i64;
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_run_step: test_dir={test_dir:?}");
        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            test_dir.clone(),
            3,
            3 * n as usize,
        )?));
        let ws_runner = Arc::clone(&ws);

        //t_label_tensor
        //0 label - sig_0 in channel 0
        //1 label - sig_1 in channel 1
        //0,1 label - sig_2 in channel 2
        let sig_0 = sig_one_imp(n, 30.0, None, Some(20. ..20.1), 0.1);
        let sig_1 = sig_one_imp(n, 30.0, None, Some(1. ..1.01), 0.1);

        (0..tensors_per_label).for_each(|i| {
            label_0_t_rs(n, &sig_0, &sig_three_imp(n, 30.0, None, None), max_y)
                .save(ws.read().tensors_path().join(format!("1_t_0_{i}.pt")))
                .unwrap(); //first 1_ for import_id
                           // label_1_t_rs(n, &sig_one_imp(n, 30.0, None, Some(1. ..1.01), 0.1), &sig_three_imp(n, 30.0, None, None), max_y).save(ws.read().tensors_path().join(format!("1_t_0_{i}.pt"))).unwrap();
        });

        (0..tensors_per_label).for_each(|i| {
            label_1_t_rs(n, &sig_1, &sig_three_imp(n, 30.0, None, None), max_y)
                .save(ws.read().tensors_path().join(format!("1_t_1_{i}.pt")))
                .unwrap();
            // label_0_t_rs(n, &sig_one_imp(n, 30.0, None, Some(20. ..20.1), 0.1), &sig_three_imp(n, 30.0, None, None), max_y).save(ws.read().tensors_path().join(format!("1_t_1_{i}.pt"))).unwrap();//first 1_ for import_id
        });

        /*
        let x = lin_t(3*n);
        plot_series(
            (0..tensors_per_label).map(|i| {
                let t = Tensor::load(&ws.read().tensors_path().join(format!("1_t_0_{i}.pt"))).expect(format!("can't open 1_t_0_{i}.pt").as_str());

                (format!("Sig0"), plotters::prelude::BLUE, vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&t.i((0,..)))).collect::<Vec<(f32, f32)>>())

            }).collect::<Vec<_>>(),
            "test_find_motifs_by_euc_dist", (1000, 500), &ws.read().plots_path().join("0.png").to_path_buf(),
        )?;
        plot_series(
            (0..tensors_per_label).map(|i| {
                let t = Tensor::load(&ws.read().tensors_path().join(format!("1_t_1_{i}.pt"))).expect(format!("can't open 1_t_1_{i}.pt").as_str());

                (format!("Sig1"), plotters::prelude::GREEN, vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&t.i((0,..)))).collect::<Vec<(f32, f32)>>())


            }).collect::<Vec<_>>(),
            "test_find_motifs_by_euc_dist", (1000, 500), &ws.read().plots_path().join("1.png").to_path_buf(),
        )?;*/

        // let store = Arc::new(Mutex::new(SqlStore::in_mem()?));
        let store = Arc::new(Mutex::new(SqlStore::from_path(ws.read().store_path())?));

        init_test_store(&store, tensors_per_label)?;
        let group_name = format!("pure group {tensors_per_label}");

        let loader = Arc::new(RwLock::new(LoadRunner::new(
            move |key: &Resource| {
                let path = ws_runner.read().tensor_path(key);
                debug!("test_run_step: path={path:?}");
                SyncWrapper::new_tb(Box::new(
                    Tensor::load(&path).expect(format!("can't open {path:?}").as_str()),
                ))
            },
            4,
            100,
        )));

        // let devices = vec![Cpu, Cpu];
        let devices = vec![Cuda(0)];

        // log::set_max_level(Trace);
        reset_print_options();

        let find_motifs_run_ids = run_step(
            None,
            &FindMotifsParams::dummy(
                100,
                800,
                group_name.clone(),
                2,
                0.5,
                None,
                None,
                false,
                None,
                Some(0.2),
            ),
            find_motifs_by_euc_dist,
            &ws,
            &store,
            &loader,
            devices.clone(),
            "Test/FindMotifs",
        )?;

        run_plot_motif_slices(
            &ws,
            &store,
            &loader,
            &PlotMotifSlicesParams::new(MotifsBy::RunId(find_motifs_run_ids[0])),
            &devices[0],
        )?;
        debug!("test_run_step: run_ids={find_motifs_run_ids:?}");

        let select_motifs_for_label_run_ids = run_step(
            Some(find_motifs_run_ids),
            &SelectMotifsParams::dummy(vec![], group_name.clone()),
            select_motifs_for_label,
            &ws,
            &store,
            &loader,
            devices.clone(),
            "Test/SelectMotifs",
        )?;

        assert_eq!(select_motifs_for_label_run_ids.len(), 1);

        let selected_motifs = store.lock().list_selected_motifs(
            select_motifs_for_label_run_ids[0],
            ws.read().channels() as u32,
        )?;

        debug!(
            "test_run_step: selected_motifs[0].len={} selected_motifs={selected_motifs:?}",
            selected_motifs[0].len()
        );

        assert!(selected_motifs[0].len() > 2);
        //no motifs in channels 1 and 2
        assert!(selected_motifs[1].is_empty());
        assert!(selected_motifs[2].is_empty());
        //TODO check if motifs match signals

        let train_motif_nodes_run_ids = run_step(
            Some(select_motifs_for_label_run_ids.clone()),
            &TrainMotifsNetsParams::dummy(
                select_motifs_for_label_run_ids.clone(),
                cycles,
                0.01,
                motif_batch_size,
                1e-5..1e-3,
                3..4,
                opt_max_evals,
                opt_n_init,
            ),
            run_train_motif_nodes_on_motif_selection,
            &ws,
            &store,
            &loader,
            devices.clone(),
            "Test/TrainMotifsNodes",
        )?;
        debug!("train_motif_nodes_run_ids={train_motif_nodes_run_ids:?}");

        let train_label_nodes_run_ids = run_step(
            Some(select_motifs_for_label_run_ids.clone()),
            &TrainLabelNetsParams::dummy(
                select_motifs_for_label_run_ids,
                group_name.clone(),
                group_name.clone(),
                5,
                2 * cycles,
                1e-3..1e-2,
                3..4,
                opt_max_evals,
                opt_n_init,
            ),
            run_train_label_nodes,
            &ws,
            &store,
            &loader,
            devices.clone(),
            "Test/TrainLabelsNodes",
        )?;

        debug!("train_label_nodes_run_ids={train_label_nodes_run_ids:?}");

        let train_final_nodes_run_ids = run_step(
            Some(train_label_nodes_run_ids.clone()),
            &TrainFinalNetParams::dummy(train_label_nodes_run_ids,
                                        10,
                                        group_name.clone(),
                                        group_name.clone(),
                                        2 * cycles,
                                        1e-5..1e-2,
                                        0..3,
                                        opt_max_evals,
                                        opt_n_init,
            ),
            run_train_final_node,
            &ws,
            &store,
            &loader,
            devices.clone(),
            "Test/TrainFinalNode",
        )?;

        debug!("train_final_nodes_run_ids={train_final_nodes_run_ids:?}");

        let input_dir = test_dir.join("infer_input");

        debug!("run_infer: input_dir={input_dir:?}");

        fs::create_dir_all(input_dir.as_path())?;
        let mut expected = InputProbs::new(vec!["Label_0".to_string(), "Label_1".to_string()]);

        (0..10).for_each(|i| {
            label_0_t_rs(n, &sig_0, &sig_three_imp(n, 30.0, None, None), max_y)
                .save(input_dir.join(format!("t_0_{i}.pt")))
                .unwrap();
            expected.add(format!("t_0_{i}"), vec![1.0, 0.0]);
        });

        (0..10).for_each(|i| {
            label_1_t_rs(n, &sig_1, &sig_three_imp(n, 30.0, None, None), max_y)
                .save(input_dir.join(format!("t_1_{i}.pt")))
                .unwrap();
            expected.add(format!("t_1_{i}"), vec![0.0, 1.0]);
        });

        let inferred = run_infer(
            &ws,
            &store,
            &InferParams::new(
                *train_final_nodes_run_ids
                    .last()
                    .expect("train_final_nodes_run_ids"),
                input_dir.to_string_lossy().to_string(),
                test_dir
                    .join("infer_output")
                    .as_path()
                    .to_string_lossy()
                    .to_string(),
                100,
            ),
            // &InferParams::new(*train_final_nodes_run_ids.last().expect("train_final_nodes_run_ids"), ws.read().tensors_path().to_string_lossy().to_string(), test_dir.join("infer_output").as_path().to_string_lossy().to_string(), 100),
            devices[0],
            |path| Ok(Tensor::load(path)?),
        )?;

        info!("inferred: labels={:?}", inferred.labels());

        for (file, probs) in inferred.inputs_probs() {
            info!("file={file} probs={probs:?}")
        }

        let (mse, _) = compare_input_probs(&expected, &inferred);

        assert!(mse < 0.1);

        Ok(())
    }

    #[test]
    fn test_import_resources() -> Result<(), ThoosaError> {
        let n = 1000i64;
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = Arc::new(RwLock::new(FsWorkspace::new(test_dir, 1, 2 * n as usize)?));

        let store = Arc::new(Mutex::new(SqlStore::in_mem()?));
        // std::fs::remove_file("/tmp/test_main_db.db3").unwrap_or(());
        // let store = Arc::new(Mutex::new(SqlStore::from_path("/tmp/test_main_db.db3")?));

        let imported = import_resources(
            &ws,
            &store,
            vec!["label_0", "label_1"],
            || {
                Ok(HashMap::from([
                    ("t_0_0".to_string(), vec![("label_0".to_string(), 1.0)]),
                    ("t_0_1".to_string(), vec![("label_0".to_string(), 1.0)]),
                    ("t_1_0".to_string(), vec![("label_1".to_string(), 1.0)]),
                    ("t_1_1".to_string(), vec![("label_1".to_string(), 1.0)]),
                ]))
            },
            |_: String| Ok(Tensor::from_slice(&[0., 1., 2., 0., 10., 20.]).view((2, -1))),
            true,
        )?;

        debug!("imported={imported:?}");
        let mut imported_keys = imported
            .into_iter()
            .map(|r| r.file_stem().to_string())
            .collect::<Vec<_>>();
        imported_keys.sort();

        assert_eq!(vec!["t_0_0", "t_0_1", "t_1_0", "t_1_1",], imported_keys);

        Ok(())
    }

    #[test]
    fn test_cli_train_label_nets() -> Result<(), ThoosaError> {
        let cli = Cli::parse_from([
            "",
            "--term-logger-level", "INFO",
            "--write-logger-level", "DEBUG",
            "--ws-path", "/tmp/ws_test/",
            "train-label-nets",
                "--train-resource-group", "pure group 100",
                "--valid-resource-group", "group 1 labels/100/0-100",
                "--select-motifs-for-label-run-ids", "254",
                "--lr", "0.00001..0.001",
                "--opt-max-evals", "20",
                "--opt-n-init", "10",
                "--batch-size", "6",
                "--cycles", "100",
                "fixed-num-slices",
                    "--num-layers", "3",
                    "--slices-len", "100",

        ]);

        debug!("cli={cli:?}");
        //TODO
        Ok(())
    }
}