
use std::backtrace::Backtrace;
use std::{env, fs};
use std::cmp::max;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::str::FromStr;
use log::{debug, error, info, LevelFilter, warn};
use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger, WriteLogger};
use tch::{Device, IndexOp, Kind, nn, Reduction, Tensor};
use hbac_motifs::find::{FilterAndSlicer, FilterAndSlicerParams, FindMotifsParams, InputFilter, iter_all_in_dir, prefix_properties, select_motifs_by_corr, select_motifs_by_ratio, SelectMotifsParams};
use crate::data::{clc_records, EEGs, eegs_to_labels_tensors, load_to_logits_tensors, parquet_to_tensor, read_eeg_index, read_motif_to_iter};
use crate::version::VERSION;
use clap::{Args, Parser, Subcommand};
use polars::export::rayon;
use polars::prelude::{LazyCsvReader, Series};
use hbac_common::utils::{cdt, DataBatch, dot_norm, from, list_devices, load_tensor, norm_2, plot_series};
use hbac_shared::HbacError;
use hbac_motifs::{find, fragments};
use hbac_motifs::fragments::{check_slices, list_motifs, load_motif_slices, motifs_to_tensor, MotifsToFragmentsParams, MotifsToTensorParams, PlotMotifSlicesParams, write_motifs_slices};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use plotters::prelude::BLACK;
use polars::export::rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelBridge};
use tch::Device::Cpu;
use tch::nn::{ConvConfig, OptimizerConfig, VarStore};

use hbac_common::builder::{Node, SeqBuilder};
use hbac_common::train_nets::{ClassTensorsDataBatch, infer, InferParams, train_final, train_on_label, TrainFinalParams, TrainOnLabelParams, TrainOnMotifsParams};
use hbac_common::simp::DummyDs;
use hbac_common::{ResourceLoader, train_nets};
use hbac_store::{SqlStore, Store};
use hbac_shared::*;
use hbac_shared::TensorSliceIdx;
use crate::Commands::*;
use polars::export::rayon::iter::ParallelIterator;
use hbac_common::train_nets::MotifNodeParams;
use hbac_shared::val::Val;

mod data;
mod utils;

mod version;


#[derive(Parser, Clone)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(long, default_value="../../data")]
    base_path: String,

    #[arg(long)]
    tensors_path: Option<String>,

    #[arg(long, default_value="hbac.db3")]
    store_path: String,

    #[arg(long, default_value = "INFO")]
    term_logger_level: String,

    #[arg(long, default_value = "INFO")]
    write_logger_level: String,

    #[arg(long, default_value = "hbac.log")]
    logfile: String,

    #[arg(long)]
    limit: Option<usize>,

    #[arg(long)]
    num_threads: Option<usize>,

    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(long)]
    input_filters_kernel: Option<i64>,//TODO all common filters configs
}


#[derive(Debug, Args, Clone)]
pub struct RunAllParams {

    #[arg(long)]
    train_resource_group: String,

    #[arg(long)]
    valid_resource_group: String,

    #[arg(long, default_value_t = false)]
    init:bool,

    #[arg(long, default_value_t = 1000)]
    step: usize,

    #[arg(long, default_value_t = 1000)]
    window: usize,

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
    pub fn step(&self) -> usize {
        self.step
    }
    pub fn window(&self) -> usize {
        self.window
    }

    fn find_motifs_params(&self, resource_set: String) -> FindMotifsParams {
        FindMotifsParams::dummy(self.step, self.window, resource_set, self.find_motifs_chunk_size, self.find_motifs_max_dist_k, self.find_motifs_input_filter.as_ref())
    }

    fn dummy()-> Self {
        Self {
            train_resource_group: "".to_string(),
            valid_resource_group: "".to_string(),
            init: false,
            step: 1000,
            window: 1000,
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


#[derive(Debug, Args, Clone, Default)]
pub struct InitResourcesParams {

}

#[derive(Debug, Subcommand, Clone)]
enum Commands {
    InitResources (InitResourcesParams),
    FindMotifs (FindMotifsParams),
    PlotMotifSlices(PlotMotifSlicesParams),
    MotifsToTensor(MotifsToTensorParams),
    SelectMotifs(SelectMotifsParams),
    TrainOnMotifs(TrainOnMotifsParams),
    TrainOnEEG(TrainOnLabelParams),
    TrainFinal(TrainFinalParams),
    RunAll(RunAllParams),
    Infer(InferParams),
}


fn main()  {
    let t1 = Instant::now();
    match _main() {

        Ok(_) => {
            info!("All done in {} mls", t1.elapsed().as_millis());
        }

        Err(e) => {
            error!("failed with {:?} in {} mls", e, t1.elapsed().as_millis());
        }
    }
    std::process::exit(0x0000);//for jupyter
}

struct EEGLoader {
    base_path:String,
    tensors_path:Option<String>,
    dsp_filters:Option<Box<dyn Fn(Tensor)->Tensor +Send + Sync >>
}

impl EEGLoader {
    pub fn new(base_path: String, tensors_path: Option<String>, dsp_filters: Option<Box<dyn Fn(Tensor)->Tensor +Send + Sync>>) -> Self {
        Self { base_path, tensors_path, dsp_filters }
    }

}

impl ResourceLoader for EEGLoader {
    type K = String;
    type V = Tensor;

    fn load(&self, key: &String, device:Device) -> Box<Tensor> {

        let t = if let Some(tensors_path) = self.tensors_path.as_ref() {
            load_tensor(tensors_path.as_str(), key.as_str(), device)
        } else {
            parquet_to_tensor(self.base_path.as_str(), key.as_str(), device)
        }.map(|t|{
            match self.dsp_filters.as_ref() {
                None => t,
                Some(f) => f(t)
            }
        }).expect(format!("can't load {}", key).as_str());

        Box::new(t)
    }
}

fn _main() -> Result<(), HbacError> {
    let cli = Cli::parse();

    CombinedLogger::init(
        vec![
            TermLogger::new(LevelFilter::from_str(cli.term_logger_level.as_str())?, Config::default(), TerminalMode::Mixed, ColorChoice::Auto),
            WriteLogger::new(LevelFilter::from_str(cli.write_logger_level.as_str())?, Config::default(), File::create(&cli.logfile)?),
        ]
    ).expect("failed to init logging");

    let store = Arc::new(Mutex::new(SqlStore::new(&cli.store_path)?));
    let loader = Arc::new(Box::new(EEGLoader::new(cli.base_path.clone(), cli.tensors_path.clone(), Some(Box::new(move |t:Tensor|{

        match cli.input_filters_kernel {
            None => {
                t
            }
            Some(kernel) => {
                debug!("main: loading with kernel={}", kernel);
                t.avg_pool1d(kernel, 1, 0, false, false)
            }
        }

    })))));

    let devices = list_devices();

    let cpu_num = std::thread::available_parallelism()?;
    let num_threads = cli.num_threads.unwrap_or_else(||max(2, cpu_num.get()/2));

    info!("VERSION: {} devices={:?}, base_path={:?} cpus={} num_threads={}", VERSION, devices, cli.base_path, cpu_num, num_threads);
    info!("limit_eeg: {:?}", cli.limit);
    std::io::stdout().flush()?;


    rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global()?;

    match &cli.command {

        Some(InitResources(init_resources_params)) => {
            debug!("InitResources: init_resources_params={:?}", init_resources_params);
            init_resources(
                &cli,
                Some(init_resources_params),
                Arc::clone(&store)
            )?;
        }

        Some(FindMotifs(find_motifs_params)) => {
            info!("find_motifs_params={:?} devices={:?}", find_motifs_params, devices);

            let run_id = run_find_motifs_on_set(&cli, &store, devices[0], find_motifs_params, Arc::clone(&loader));

            info!("run_find_motifs_on_set done, run_id={:?}", run_id);
        }

        Some(SelectMotifs(select_motifs_params)) => {
            info!("select_motifs_params={:?} devices={:?}", select_motifs_params, devices);

            let select_motifs_params = if let Some(train_resource_group) = select_motifs_params.train_resource_group() {
                let train_labels_datasets = { store.lock().expect("store lock").list_group_labeled_resources_sets(train_resource_group.as_str())? };
                select_motifs_params.with_labels_datasets(train_labels_datasets)
            } else {
                select_motifs_params.clone()
            };

            let corr_ids = run_select_motifs(&cli, &store, devices, &select_motifs_params, Arc::clone(&loader));

            info!("run_select_motifs done, corr_ids={:?}", corr_ids);
        }

        Some(TrainOnMotifs(train_on_motif_params)) => {
            debug!("train_on_motif: train_on_motif_params={:?}", train_on_motif_params);

            run_train_on_motifs(&cli, &store, devices, train_on_motif_params, Arc::clone(&loader))?;
        }

        Some(TrainOnEEG(train_on_eeg_params)) => {
            debug!("train_on_eeg_params={:?}", train_on_eeg_params);

            let train_on_eeg_params = if let Some(train_resource_group) = train_on_eeg_params.train_resource_group() {
                let train_labels_datasets = { store.lock().expect("store lock").list_group_labeled_resources_sets(train_resource_group.as_str())? };
                let valid_labels_datasets = { store.lock().expect("store lock").list_group_labeled_resources_sets(train_on_eeg_params.valid_resource_group().expect("valid_resource_group").as_str())? };
                train_on_eeg_params.with_labels_datasets(train_labels_datasets, valid_labels_datasets)

            } else {
                train_on_eeg_params.clone()
            };

            run_train_on_label(vec![], &store, devices, &train_on_eeg_params, Arc::clone(&loader))?;
        }

        Some(TrainFinal(train_final_params)) => {

            debug!("train_final_params={:?}", train_final_params);

            run_train_final(&cli, &store, &devices[0], train_final_params, Arc::clone(&loader))?;
        }

        Some(RunAll(run_all_params)) => {
            debug!("RunAll: run_all_params={:?}", run_all_params);

            if run_all_params.init {
                debug!("RunAll: init_resources");
                init_resources(
                    &cli,
                    None,
                    Arc::clone(&store)
                )?;
            }

            //run_id - params set
            let run_ids = run_find_motifs_on_group(&cli, &store, devices.clone(), run_all_params, Arc::clone(&loader))?;

            info!("find_motifs done: run_ids={:?}", run_ids);

            let train_labels_datasets = {store.lock().expect("store lock").list_group_labeled_resources_sets(run_all_params.train_resource_group())?};
            let valid_labels_datasets = {store.lock().expect("store lock").list_group_labeled_resources_sets(run_all_params.valid_resource_group())?};

            info!("run all: train_labels_datasets={:?}", train_labels_datasets);
            info!("run all: valid_labels_datasets={:?}", valid_labels_datasets);

            let corr_ids = run_select_motifs(
                &cli,
                &store,
                devices.clone(),
                &SelectMotifsParams::dummy(run_ids, train_labels_datasets.clone(), run_all_params.select_motifs_min_ratio),
                Arc::clone(&loader)
            )?;

            info!("select_motifs done: corr_ids={:?}", corr_ids);

            let run_ids = run_train_on_motifs(&cli, &store, devices.clone(), &TrainOnMotifsParams::dummy(&run_all_params.models_dir, corr_ids, 300), Arc::clone(&loader))?;

            info!("run_train_on_motifs done");

            run_train_on_label(
                run_ids,
                &store,
                devices.clone(),
               &TrainOnLabelParams::dummy(
                   run_all_params.step,
                   run_all_params.window,
                   run_all_params.models_dir.clone(),
                   run_all_params.models_dir.clone(),
                   train_labels_datasets.clone(),
                   valid_labels_datasets.clone(),
                   run_all_params.motif_batch_size,
                   5,
                   100,
                   1.0e-5,
               ),
               Arc::clone(&loader),
            )?;

            info!("run_train_on_eeg done");

            run_train_final(
                &cli, &store,
                &devices[0],
                &TrainFinalParams::dummy(
                    run_all_params.step,
                    run_all_params.window,
                    vec![run_all_params.models_dir.clone()],
                    format!("{}/final.safetensors", run_all_params.models_dir),
                    run_all_params.train_resource_group.clone(),
                    run_all_params.valid_resource_group.clone(),
                    run_all_params.train_final_cycles,
                    run_all_params.motif_batch_size,
                    5,
                ),
                Arc::clone(&loader)
            )?;

            info!("run_train_final done");
        }

        Some(Infer(infer_params)) => {

            let labels = store.lock().expect("store").list_labels()?;

            let t = infer(
                &infer_params,
                |path:&Path|{
                    data::parquet_to_tensor_by_path(<Path as AsRef<Path>>::as_ref(path), *devices.first().expect("first device"))
                },
                infer_params.window() as i64,
                infer_params.step(),
                infer_params.motif_batch_size(),
                20,
                684400,
                &devices[0],
                labels
            )?;

            data::save_tensor(
                &["eeg_id","seizure_vote","lpd_vote","gpd_vote","lrda_vote","grda_vote","other_vote"],
                t.0,
                &t.1,
                infer_params.output()
            )

        }
        Some(PlotMotifSlices(plot_motif_slices_params)) => {
            plot_motif_slices(&cli, &store, *devices.first().expect("first device"), &plot_motif_slices_params, Arc::clone(&loader))?;
        }

        Some(MotifsToTensor(choose_fragments_params)) => {
            motifs_to_tensor_cmd(&cli, &store, *devices.first().expect("first device"), choose_fragments_params, Arc::clone(&loader))?;
        }
        None => {
            warn!("No command");
        }
    }

    Ok(())
}

fn run_train_final(cli: &Cli, store: &Arc<Mutex<SqlStore>>, device: &Device, train_final_params: &TrainFinalParams, loader: Arc<Box<impl ResourceLoader<K=String, V=Tensor>>>) -> Result<(), HbacError> {

    let labels = {store.lock().expect("mutex_guard").list_labels()? };

    debug!("run_train_final: labels={:?}", labels);

    let vs = train_final(
        train_final_params,
        load_to_logits_tensors(
            cli.base_path.as_str(),
            cli.tensors_path.clone(),
            train_final_params.train_resource_group(),
            &device,
            Arc::clone(&store), //cli.limit.as_ref()
            //684400 - TODO cut at max_input
            Arc::clone(&loader)
        ),
        load_to_logits_tensors(
            cli.base_path.as_str(),
            cli.tensors_path.clone(),
            train_final_params.valid_resource_group(),
            &device,
            Arc::clone(&store), //cli.limit.as_ref()
            //684400 - TODO cut at max_input
            Arc::clone(&loader)
        ),
        train_final_params.step(),
        train_final_params.motif_batch_size(),
        20,
        684400,
        &device,
        labels,
    )?;

    vs.save(train_final_params.final_model())?;
    Ok(())
}

fn run_train_on_label(input_run_ids: Vec<usize>, store: &Arc<Mutex<SqlStore>>, devices: Vec<Device>, train_on_eeg_params: &TrainOnLabelParams, loader: Arc<Box<impl ResourceLoader<K=String, V=Tensor>>>) -> Result<(), HbacError> {
    debug!("run_train_on_eeg: train_on_eeg_params={:?}", train_on_eeg_params);
    // let labels_kinds = store_guard.corrs_descr(train_on_eeg_params.corr_run())?;


    let valid_datasets_map = train_on_eeg_params.valid_labels_datasets().into_iter().fold(HashMap::new(), |mut a, v|{
        a.insert(v.0.clone(), v.1.clone());
        a
    });

    if valid_datasets_map.len() != train_on_eeg_params.train_labels_datasets().len() {
        error!("valid_datasets_map={:?}", valid_datasets_map);
        error!("train_on_eeg_params.train_labels_datasets()={:?}", train_on_eeg_params.train_labels_datasets());
        panic!("valid_datasets_map.len() != train_on_eeg_params.train_labels_datasets().len()");
    } //TODO check keys

    let label_datasets_with_run_id = train_on_eeg_params.train_labels_datasets().flat_map(|ds| input_run_ids.iter().map(|input_run_id|(ds, *input_run_id))).collect::<Vec<((String, String), usize)>>();
    let chunk_len = max(1, label_datasets_with_run_id.len()/devices.len());

    let devices_train_labels_datasets = devices.into_iter().cycle().zip(label_datasets_with_run_id.chunks(chunk_len).into_iter()).collect::<Vec<(Device, &[((String, String), usize)])>>();//corrs_run(id, label)
    devices_train_labels_datasets.into_par_iter().for_each(move |(device_i, chunk_resource_sets)| {

        for ((label, resource_set), input_run_id) in chunk_resource_sets {

            info!("training {} on {} device={:?}", label, resource_set, device_i);

            match train_on_label(
                *input_run_id,
                train_on_eeg_params,
                eegs_to_labels_tensors(
                    label,
                    resource_set,
                    &device_i,
                    Arc::clone(&store),
                    Arc::clone(&loader)
                ),
                eegs_to_labels_tensors(
                    label,
                    valid_datasets_map.get(label).expect(format!("valid dataset for {}", label).as_str()),
                    &device_i,
                    Arc::clone(&store),
                    Arc::clone(&loader)
                ),
                train_on_eeg_params.items_per_label(),
                train_on_eeg_params.window() as i64,
                train_on_eeg_params.step(),
                train_on_eeg_params.motif_batch_size(),
                20,
                684400,
                &device_i,
                label
            ) {
                Ok((vs,node_params)) => {
                    vs.save(format!("{}/{}_{}.safetensors", train_on_eeg_params.models_dir(), label, train_on_eeg_params.corr_run())).expect("save vs");
                    fs::write(Path::new(train_on_eeg_params.models_dir()).join(format!("{}_{}.json", label, train_on_eeg_params.corr_run())), serde_json::to_string(&node_params).expect("save params"));
                }
                Err(e) => {//should be no motif nets found only
                    warn!("e:{:?}", e);
                }
            }
        }

    });
    Ok(())
}

fn run_train_on_motifs(cli: &Cli, store: &Arc<Mutex<SqlStore>>, devices: Vec<Device>, train_on_motif_params: &TrainOnMotifsParams, loader: Arc<Box<impl ResourceLoader<K=String, V=Tensor>>>) -> Result<Vec<usize>, HbacError> {
    let corr_run_ids = train_on_motif_params.corr_run_ids();

    let chunk_len = corr_run_ids.len()/devices.len();
    let corr_run_id_labels = { store.lock()?.list_corrs(corr_run_ids)? }; //(1..=36).collect::<Vec<usize>>();

    let devices_corr_run = devices.into_iter().cycle().zip(corr_run_id_labels.chunks(chunk_len).into_iter()).collect::<Vec<(Device, &[(usize, usize, String)])>>();//corrs_run(id, label)

    let cli_clone = cli.clone();
    let mut run_ids = Arc::new(Mutex::new(vec![]));
    let mut run_ids_for_threads = Arc::clone(&run_ids);

    devices_corr_run.into_par_iter().for_each( |(device_i, corrs_run)| {
        info!("TrainOnMotifs: corrs_run: {:?} device_i={:?}", corrs_run, device_i);

        for (corr_run_id, run_id, label) in corrs_run {
            let run_ids = train_on_motifs(
                &cli_clone,
                *run_id,
                *corr_run_id,
                Arc::clone(&store),
                device_i,
                train_on_motif_params,
                label,
                Arc::clone(&loader)
            ).expect("train_on_motifs");

            run_ids_for_threads.lock().expect("run_ids lock").extend(run_ids);

            // fs::write(Path::new(train_on_motif_params.output_dir()).join(format!("motif_model_{}.json", run_id)), serde_json::to_string(&node_params).expect("serde_json::to_string"));
        }

        info!("TrainOnMotifs: corrs_run: {:?} device_i={:?} done ", corrs_run, device_i);
    });


    let guard = run_ids.lock().expect("run_ids lock on return");
    Ok(guard.to_vec())

}

pub fn run_select_motifs(cli: &Cli, store: &Arc<Mutex<SqlStore>>, devices: Vec<Device>, select_motifs_params: &SelectMotifsParams, loader: Arc<Box<impl ResourceLoader<K=String, V=Tensor>>>) -> Result<Vec<usize>, HbacError>{

    let chunk_len = select_motifs_params.run_ids().len()/devices.len();

    let cli_clone = cli.clone();
    let corr_ids = Arc::new(Mutex::new(vec![]));
    let corr_ids_for_iter =  Arc::clone(&corr_ids);

    devices.into_iter().cycle().zip(select_motifs_params.run_ids().chunks(chunk_len).into_iter()).collect::<Vec<(Device, &[usize])>>().into_par_iter().for_each(move |(device_i, run_ids)| {
        for run_id in run_ids {
            let corr_ids = select_motifs_for_label(
                &cli_clone,
                Arc::clone(&store),
                device_i,
                &select_motifs_params,
                *run_id,
                Arc::clone(&loader)
            ).expect("select_motifs_in_class");

            corr_ids_for_iter.lock().expect("corr_ids lock").extend(corr_ids);
        }
    });

    let corr_ids = corr_ids.lock().expect("corr_ids lock ret").clone();
    Ok(corr_ids)
}

fn run_find_motifs_on_group(cli: &Cli, store: &Arc<Mutex<SqlStore>>, devices: Vec<Device>, run_all_params: &RunAllParams, loader: Arc<Box<impl ResourceLoader<K=String, V=Tensor>+ 'static>>) -> Result<Vec<usize>, HbacError>{

    let resource_sets = { store.lock().unwrap().list_resource_set_group_sets(run_all_params.train_resource_group()).expect("list_resource_set_group_sets")};

    assert!(resource_sets.len() > 0, "{} is empty", run_all_params.train_resource_group());

    let chunk_len = resource_sets.len()/devices.len();
    let devices_resource_sets = devices.into_iter().cycle().zip(resource_sets.chunks(chunk_len).into_iter()).collect::<Vec<(Device, &[(usize, String)])>>();

    let mut run_ids = Arc::new(Mutex::new(vec![]));
    let mut run_ids_for_threads = Arc::clone(&run_ids);

    devices_resource_sets.into_par_iter().for_each(move |(device_i, chunk_resource_sets)| {

        for resource_set in chunk_resource_sets {
            let find_motifs_params = &run_all_params.find_motifs_params(resource_set.1.clone());

            match run_find_motifs_on_set(cli, &store, device_i, find_motifs_params, Arc::clone(&loader)) {
                Ok(run_id) => {
                    run_ids_for_threads.lock().expect("run_ids lock").push(run_id);
                }
                Err(e) => {
                    error!("run_find_motifs_on_group: resource_set={:?}, e={:?}", resource_set, e);
                }
            }
        }
    });

    let guard = run_ids.lock().expect("run_ids lock on return");
    Ok(guard.iter().flat_map(|v|v.iter().map(|v:&usize|*v)).collect())

}

fn run_find_motifs_on_set(cli: &Cli, store: &Arc<Mutex<SqlStore>>, device: Device, find_motifs_params: &FindMotifsParams, loader: Arc<Box<impl ResourceLoader<K=String, V=Tensor> + 'static>>) -> Result<Vec<usize>, HbacError> {
    let eeg_id_series = { store.lock().unwrap().list_resource_set_resources_by_descr(find_motifs_params.resource_set()).expect("list_resource_set_resources") };

    let records_num = clc_records(&eeg_id_series, cli.base_path.as_str()).unwrap();
    debug!("records_num={}", records_num);

    find::find_motifs_euc(
        move |key: &String| {
            loader.load(key, device)
        },
        find_motifs_params,
        records_num,
        eeg_id_series,
        Arc::clone(&store),
        20,
    )

}

fn motifs_to_tensor_cmd(cli: &Cli, store: &Arc<Mutex<SqlStore>>, device: Device, choose_fragments_params: &MotifsToTensorParams, loader: Arc<Box<impl ResourceLoader<K=String, V=Tensor>>>) -> Result<(), HbacError> {
    let mut store_guard = store.lock()?;
    let base_path_1 = cli.base_path.clone();
    let tensors_path = cli.tensors_path.clone();
    let t = motifs_to_tensor(
        store_guard.list_motifs_slices(choose_fragments_params.run_id())?,
        move |key: &String| {

            loader.load(&key, device)
        },
        choose_fragments_params.window(),
        20,
        device
    )?;

    if let Some(out) = choose_fragments_params.output_dir() {
        t.0.save(Path::new(out).join(format!("motifs_in_{}", choose_fragments_params.run_id())));
    }
    Ok(())
}

fn init_resources(cli:&Cli, params:Option<&InitResourcesParams>, store:Arc<Mutex<impl Store + 'static>>) -> Result<(), HbacError> {
    let mut store_guard = store.lock()?;

    store_guard.save_label("seizure");
    store_guard.save_label("lpd");
    store_guard.save_label("gpd");
    store_guard.save_label("lrda");
    store_guard.save_label("grda");
    store_guard.save_label("other");


    store_guard.save_resources_labels(data::labeled_resourses(Path::new(&cli.base_path).join("train.csv"))?)?;

    store_guard.save_pure_sets(1000, true);
    store_guard.save_pure_sets(100, true);
    store_guard.save_pure_sets(10, true);
    store_guard.save_pure_sets(3, true);

    store_guard.save_groups_by_label_cnt(100);
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


fn plot_motif_slices(cli: &Cli, store: &Arc<Mutex<SqlStore>>, device: Device, plot_motif_slices_params: &&PlotMotifSlicesParams, loader: Arc<Box<impl ResourceLoader<K=String, V=Tensor>>>) -> Result<(), HbacError> {
    let mut store_guard = store.lock()?;

    let (channel, run_id, slices) = store_guard.list_motif_slices(plot_motif_slices_params.motif_id(), None, None)?;
    let base_path = cli.base_path.as_str();

    let slices = load_motif_slices(
        cli.base_path.as_str(), channel, 1000, slices,
        move |eeg_id| {
            loader.load(&eeg_id.to_string(), device)
        },
        device
    );

    let all_sices = slices.iter().flat_map(|(eeg_id, slices)| slices.iter()).collect::<Vec<&Box<Tensor>>>();

    check_slices(&all_sices, 0.9);

    plot_series(
        all_sices.into_iter().map(|t| {
            (format!("0"), BLACK, t.iter::<f64>().unwrap().enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>())
        }).collect(),
        format!("Motif {}", plot_motif_slices_params.motif_id()).as_str(),
        (10000, 10000),
        &Path::new(plot_motif_slices_params.output_dir()).join(format!("motif_{}.png", plot_motif_slices_params.motif_id()))
    ).unwrap();
    Ok(())
}

fn train_on_motifs(cli: &Cli, input_run_id: usize, corr_run: usize, store:Arc<Mutex<impl Store + 'static>>, device: Device, train_on_motif_params: &TrainOnMotifsParams, label: &String, loader: Arc<Box<impl ResourceLoader<K=String, V=Tensor>>>) -> Result<Vec<usize>, HbacError> {
    debug!("train_on_motifs: device={:?}, corr_run={:?}", device, corr_run);

    let (motifs, params) = {
        let mut store_guard = store.lock()?;

        // for motif_id in train_on_motif_params.motif_ids() {

        (
            store_guard.list_corr_motifs(corr_run)?//TODO param
                .into_iter()
                .flat_map(|channel_motifs| channel_motifs.into_iter())
                .filter(|(motif_id, corr)| *corr > train_on_motif_params.min_corr())
                .map(|(motif_id, corr)| motif_id).collect::<Vec<usize>>(),

            FilterAndSlicerParams::from_properties_map(store_guard.list_run_params(input_run_id, None)?)
        )
    };


    let output_dir = Path::new(train_on_motif_params.output_dir()).join(input_run_id.to_string()).join(label);//TODO corr_run

    fs::create_dir_all(&output_dir);


    let filter_and_slice:FilterAndSlicer = params.into();
    let filter_op = filter_and_slice.filter();

    let (node_params_vec, run_ids) = {
        let mut store_guard = store.lock()?;

        // store_guard.save_params_value(
        //     prefix_properties("motifs_node", node_params.to_properties_map()),
        //     run_id,
        //     Some(label.as_str())
        // )?;

        (0..1).fold((vec![], vec![]), |mut a, _| {//TODO expand
            let node_params = MotifNodeParams::new(filter_and_slice.window());

            a.1.push(store_guard.save_tree_run(Some(input_run_id), "todo", &vec![], prefix_properties("motifs_node", node_params.to_properties_map())).expect("save_tree_run"));
            a.0.push(node_params);

            a
        })

    };


    for motif_id in motifs.iter() {
        let (all_sices_0, all_sices_1, channel) = {
            let mut store_guard = store.lock()?;

            let (channel, run_id, slices) = store_guard.list_motif_slices(*motif_id, None, None)?;

            debug!("train_on_motif: 1. channel={}, run_id={}, slices={}", channel, run_id, slices.len());
            let loader_m = Arc::clone(&loader);
            let all_sices_1 = train_nets::slices_in_all_resources(
                load_motif_slices(
                    cli.base_path.as_str(), channel, filter_and_slice.window(), slices,
                    move |eeg_id| {
                        // loader_m.load(&eeg_id.to_string(), device)

                        let t = loader_m.load(&eeg_id.to_string(), device);

                        if let Some(filter) = filter_op {
                            Box::new(filter(&t))
                        } else {
                            t
                        }
                    },
                    device
                )
            );
            debug!("train_on_motif: 1. run_id={}, done", run_id);

            let (channel, run_id, slices) = store_guard.list_motif_slices(*motif_id, Some(all_sices_1.len()), Some((run_id, channel as usize)))?;

            debug!("train_on_motif: 0. channel={}, run_id={}, slices={}", channel, run_id, slices.len());

            let loader_m = Arc::clone(&loader);
            let all_sices_0 = train_nets::slices_in_all_resources(
                load_motif_slices(
                    cli.base_path.as_str(), channel, filter_and_slice.window(), slices,
                    move |eeg_id| {
                        loader_m.load(&eeg_id.to_string(), device)
                    },
                    device
                )
            );

            debug!("train_on_motif: 0. run_id={}, done", run_id);

            (all_sices_0, all_sices_1, channel)
        };

        assert!(all_sices_1.len() > 0, "all_sices_1.len(), motif={}", motif_id);
        assert!(all_sices_0.len() > 0, "all_sices_0.len(), motif={}", motif_id);

        let batch_size = 2; //TODO, purge small motifs
        if all_sices_1.len() < batch_size {
            warn!("num of motif_id {} {} in channel {} is less than {}", motif_id, all_sices_1.len(), channel, batch_size);
            continue;
        }

        if all_sices_0.len() < batch_size {
            warn!("num of not motif_id {} {} in channel {} is less than {}", motif_id, all_sices_0.len(),  channel, batch_size);
            continue;
        }

        debug!("train_on_motif: all_sices_1.len()={}, all_sices_0.len()={} all_sices_1[0].size()={:?}", all_sices_1.len(), all_sices_0.len(), all_sices_1[0].size());


        for node_params in &node_params_vec {
            let (loss, mut vs) = train_nets::train_on_motif(
                all_sices_0.iter().map(|v|Box::new(v.shallow_clone())).collect(),//TODO change ClassTensorsDataBatch to accept refs
                all_sices_1.iter().map(|v|Box::new(v.shallow_clone())).collect(),
                train_on_motif_params,
                device,
                node_params,
                batch_size
            )?;

            debug!("train_on_motif: motif_id={} loss={}", motif_id, loss);

            if loss < train_on_motif_params.max_loss() {
                info!("train_on_motif: saving motif_id={} loss={}", motif_id, loss);

                let channel_models = output_dir.join(channel.to_string());
                fs::create_dir_all(&channel_models);

                vs.save(channel_models.join(format!("{}_{}_{:3}.safetensors", corr_run, motif_id, loss)))?;
            }
        }
    }

    Ok(run_ids)
}

fn select_motifs_for_label(cli: &Cli, store:Arc<Mutex<impl Store + 'static>>, device: Device, select_motifs_params: &SelectMotifsParams, run_id:usize, loader: Arc<Box<impl ResourceLoader<K=String, V=Tensor>>>) -> Result<Vec<usize>, HbacError> {
    debug!("select_motifs_in_class: device={:?} cli.tensors_path={:?}", device, cli.tensors_path);

    let (resource_id_motif_slices, motifs_avg_distances, params) = {
        let mut store_guard = store.lock()?;

        (store_guard.list_motifs_slices(run_id)?, store_guard.list_motifs_distances(run_id)?, FilterAndSlicerParams::from_properties_map(store_guard.list_run_params(run_id, None)?))
    };


    assert!(resource_id_motif_slices.len() > 0, "no slices for run_id={}", run_id);
    let loader_m = Arc::clone(&loader);

    let filter_and_slice:FilterAndSlicer = params.into();
    let filter_op = filter_and_slice.filter();

    let (motifs, channel_motif_ids, motifs_slices_map) = motifs_to_tensor(
        resource_id_motif_slices,

        move |key: &String| {
            let t = loader_m.load(key, device);

            if let Some(filter) = filter_op {
                Box::new(filter(&t))
            } else {
                t
            }

        },
        filter_and_slice.window() as i64,
        20,//TODO
        device
    )?;

    info!("select_motifs: motifs tensor built, size={:?}", motifs.size());

    let labels_channels_good_motif_idxs = select_motifs_by_ratio(
        motifs,
        select_motifs_params.labels_datasets().into_iter().map(|(label_dataset)|{
            let loader_m = Arc::clone(&loader);
            (
                label_dataset.0.clone(),
                store.lock().expect("store").list_resource_set_resources_by_descr(label_dataset.1.as_str()).unwrap().into_iter().map(move |eeg_id| {
                    loader_m.load(&eeg_id, device)
                })
            )
        }).collect(),
        cli.limit,
        filter_and_slice.step(),
        device,
        select_motifs_params.min_ratio(),
        motifs_avg_distances,//TODO keeps motif indexes with motif tensors
        &channel_motif_ids
    )?;

    Ok(labels_channels_good_motif_idxs.into_iter().map(|(label, channels_good_motif_idxs)| {

        let motifs_corrs = channel_motif_ids.iter()
            .zip(channels_good_motif_idxs.into_iter())
            .map(|(motif_ids, motif_idxs_corss)| {

                motif_idxs_corss.into_iter().map(|(motif_idx, motif_idx_corr)| {
                    (motif_ids[motif_idx as usize], motif_idx_corr)
                }).collect::<Vec<(usize, f64)>>()

            }).collect::<Vec<Vec<(usize, f64)>>>();

        debug!("select_motifs: motifs_corrs={:?}", motifs_corrs);

        store.lock().expect("store").save_motifs_corrs(run_id, format!("{:?}", select_motifs_params.labels_datasets()).as_str(), motifs_corrs, label).expect("save_motifs_corrs")
    }).collect::<Vec<usize>>())

}


#[cfg(test)]
mod main_tests {
    use std::fs;
    use std::sync::{Arc, Mutex};
    use log::LevelFilter;
    use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger};
    use tch::Device;
    use tch::Device::Cpu;
    use hbac_common::train_nets::{TrainFinalParams, TrainOnLabelParams};
    use hbac_motifs::find::{FindMotifsParams, SelectMotifsParams};
    use hbac_shared::HbacError;
    use hbac_store::{SqlStore, Store};
    use crate::{Cli, Commands, EEGLoader, run_find_motifs_on_group, run_select_motifs, run_train_final, run_train_on_label, RunAllParams};
    use crate::Commands::{FindMotifs, SelectMotifs, TrainFinal, TrainOnEEG};
    use log::debug;
    // use crate::data::{clc_records, read_eeg_index, read_motif_to_iter_imp};

    #[ctor::ctor]
    fn init() {
        CombinedLogger::init(
            vec![
                TermLogger::new(LevelFilter::Debug, Config::default(), TerminalMode::Mixed, ColorChoice::Auto)
            ]
        ).unwrap_or_else(|e|{
            println!("CombinedLogger::init failed: {:?}", e);
        });


        std::env::set_var("POLARS_FMT_MAX_COLS", "100");
        std::env::set_var("POLARS_FMT_MAX_ROWS", "100");
    }

    #[test]
    fn test_it() -> Result<(), HbacError>{

        // let x = FindMotifsParams::;
        // println!("x={:?}", x);
        // clc_records()
        // debug!("XXXXXXXXXXXXXXXXX");
        // main();
        // EEGs::good_eegs_of("../../data/", "seizure_vote", Some(10))?.i64()?.into_iter().enumerate().map(|(i, v):(usize, Option<i64>)|{
        //     // String::from(v.expect(format!("no value in {}", i))).as_str();
        //
        //     data::eeg_to_tensor("../../data/", v.expect(format!("no value in {}", i).as_str()).to_string().as_str(), Device::Cpu)
        // }).try_for_each(|t|{
        //     println!("t={:?}", t?.size());
        //
        //     Ok::<(), HbacError>(())
        // })?;

        // let ce = Tensor::from_slice(&[1., 1., 0., 0.]).binary_cross_entropy::<Tensor>(&Tensor::from_slice(&[1., 1., 0., 0.1]), None, Reduction::Mean);
        // ce.print();

        Ok(())
    }

    #[test]
    fn test_run_find_motifs() -> Result<(), HbacError>{
        let cli = Cli{
            base_path: "../../data/".to_string(),
            tensors_path: None,
            store_path: "".to_string(),
            term_logger_level: "".to_string(),
            write_logger_level: "".to_string(),
            logfile: "".to_string(),
            limit: None,
            num_threads: None,
            command: Some(FindMotifs(FindMotifsParams::dummy(10, 10, "xxx".into(), 10, 0.3, None))),
            input_filters_kernel: None,
        };

        fs::remove_file("/tmp/test_main_db.db3").unwrap_or(());
        let store = Arc::new(Mutex::new(SqlStore::new("/tmp/test_main_db.db3")?));
        let loader = Arc::new(Box::new(EEGLoader::new("../../data/".into(), Some("/media/bu/KINGSTON/all_eegs.pt/all/".into()), None)));

        run_find_motifs_on_group(
            &cli,
            &store,
            vec![Cpu],
            &RunAllParams::dummy(),
            Arc::clone(&loader)
        )?;

        Ok(())
    }

    #[test]
    fn test_run_select_motifs() -> Result<(), HbacError>{
        let cli = Cli{
            base_path: "../../data/".to_string(),
            tensors_path: None,
            store_path: "".to_string(),
            term_logger_level: "".to_string(),
            write_logger_level: "".to_string(),
            logfile: "".to_string(),
            limit: None,
            num_threads: None,
            command: Some(SelectMotifs(SelectMotifsParams::dummy(vec![0, 1], vec![], 0.1))),
            input_filters_kernel: None,
        };

        let store = Arc::new(Mutex::new(SqlStore::new("/tmp/test_main_db.db3")?));
        let loader = Arc::new(Box::new(EEGLoader::new("../../data/".into(), Some("/media/bu/KINGSTON/all_eegs.pt/all/".into()), None)));
        let corr_ids = run_select_motifs(
            &cli,
            &store,
            vec![Cpu],
            &SelectMotifsParams::dummy(vec![0, 1], vec![("seizure".into(), "seizure/10".into())], 0.1),
            Arc::clone(&loader)
        )?;

        debug!("corr_ids={:?}", corr_ids);

        Ok(())
    }



    #[test]
    fn test_run_train_on_label() -> Result<(), HbacError>{
        let train_eeg_params = TrainOnLabelParams::dummy(
            1000,
            1000,
            "../../data_local/test_models/".into(),
            "/tmp".into(),
            vec![("grda".into(), "grda/3".into())],
            vec![("grda".into(), "grda/3".into())], 100, 2, 2, 1.0e-5,
        );

        let loader = Arc::new(Box::new(EEGLoader::new("../../data/".into(), Some("/media/bu/KINGSTON/all_eegs.pt/all/".into()), None)));

        let cli = Cli{
            base_path: "../../data/".into(),
            tensors_path: Some("/media/bu/KINGSTON/all_eegs.pt/all/".into()),
            store_path: "../../data_down/group_1_labels_100_100_200/hbac-model/hbac_all.db3".to_string(),
            term_logger_level: "DEBUG".into(),
            write_logger_level: "DEBUG".into(),
            logfile: "/tmp/hbac_test.log".into(),
            limit: None,
            num_threads: None,
            command: Some(TrainOnEEG(train_eeg_params.clone())),
            input_filters_kernel: None,
        };

        let store = Arc::new(Mutex::new(SqlStore::new(cli.store_path.clone())?));

        run_train_on_label(
            vec![],
            &store,
            vec![Cpu],
            &train_eeg_params,
            Arc::clone(&loader),
        )?;

        Ok(())
    }

    #[test]
    fn test_run_train_final() -> Result<(), HbacError>{

        let train_final_params = TrainFinalParams::dummy(
            1000,
            1000,
            vec![
                "../../data_down/group_1_labels_100_0-100/0_100_models".into(),
                "../../data_down/group_1_labels_100_100_200/100_200_models".into()
            ],
            "/tmp/final.safetensors".into(),
            "pure group 3".into(),
            "pure group 3".into(),
            10,
            100,
            5
        );

        let loader = Arc::new(Box::new(EEGLoader::new("../../data/".into(), Some("/media/bu/KINGSTON/all_eegs.pt/all/".into()), None)));

        let cli = Cli{
            base_path: "../../data/".into(),
            tensors_path: Some("/media/bu/KINGSTON/all_eegs.pt/all/".into()),
            store_path: "../../data_down/group_1_labels_100_100_200/hbac-model/hbac_all.db3".to_string(),
            term_logger_level: "DEBUG".into(),
            write_logger_level: "DEBUG".into(),
            logfile: "/tmp/hbac_test.log".into(),
            limit: None,
            num_threads: None,
            command: Some(TrainFinal(train_final_params.clone())),
            input_filters_kernel: None,
        };

        let store = Arc::new(Mutex::new(SqlStore::new(cli.store_path.clone())?));

         run_train_final(
             &cli,
             &store,
             &Cpu,
             &train_final_params,
             Arc::clone(&loader)
         )?;

        Ok(())
    }

    // #[test]
    // fn motifs_to_store() -> Result<(), HbacError>{
    //     let runs = vec![
    //         ("2024-02-14T17:36:36Z", "seizure"),
    //         ("2024-02-15T17:14:09Z", "lpd"),
    //         ("2024-02-16T06:34:04Z", "gpd"),
    //         ("2024-02-16T11:31:30Z", "lrda"),
    //         ("2024-02-16T16:29:11Z", "grda"),
    //         ("2024-02-19T07:12:01Z", "other"),
    //     ];
    //
    //     let mut store = SqlStore::new("hbac_imp.db3")?;
    //
    //     for (dir, desc) in runs {
    //         let (_, eeg_ids) = read_eeg_index(format!("../../data_down/14_02_results/{}", dir).as_str())?;
    //
    //         let run_id = store.save_run(format!("{}/csv motifs import", desc).as_str(), &eeg_ids)?;
    //
    //         for channel in 0..20 {
    //             let r = read_motif_to_iter_imp(format!("../../data_down/14_02_results/{}/motifs_{}.csv", dir, channel).as_str())?;
    //
    //             for chunk in r.chunks(500) {
    //                 store.save_slices(run_id, channel, chunk.to_vec());
    //             }
    //
    //             store.set_motifs(run_id, channel);
    //
    //             println!("desc={}, channel={}, run_id={}, r.len()={}", desc, channel, run_id, r.len());
    //         }
    //     }
    //     Ok(())
    // }
}