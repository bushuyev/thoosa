use clap::{Args, Parser, Subcommand};
use std::sync::Arc;
use parking_lot::{Mutex};
use std::collections::HashMap;
use tch::{Device, Tensor};
use log::{debug, error, info, warn, LevelFilter};
use std::cmp::max;
use std::path::Path;
use tch::nn::Module;
use std::fs;
use plotters::prelude::BLACK;
use simplelog::{ColorChoice, CombinedLogger, Config, TermLogger, TerminalMode, WriteLogger};
use std::fs::File;
use std::str::FromStr;
use std::io::Write;
use parking_lot::lock_api::RwLock;
use polars::export::rayon;
use polars::export::rayon::iter::{IntoParallelIterator, ParallelIterator};
use thoosa_common::loader::{LoadRunner, SyncWrapper};
use thoosa_common::{train_nets, ResourceLoader};
use thoosa_common::train_nets::{infer, load_motif_models, train_final, train_on_label, InferParams, LabelNodeParams, MotifNodeParams, TrainFinalParams, TrainOnLabelParams, TrainOnMotifsParams};
use thoosa_common::utils::{list_devices, plot_series};
use thoosa_motifs::find;
use thoosa_motifs::find::{select_motifs_by_ratio, FilterAndSlicer, FilterAndSlicerParams, FindMotifsParams, InputFilter, SelectMotifsParams};
use thoosa_motifs::fragments::{check_slices, load_motif_slices, motifs_to_tensor, MotifsToTensorParams, PlotMotifSlicesParams};
use thoosa_shared::{batch_windows, ThoosaError};
use thoosa_shared::val::{prefix_properties, unprefix_properties};
use thoosa_store::{SqlStore, Store};
use crate::data::{clc_records, labeled_tensors_by_descr, load_to_logits_tensors, ParquetLoader};
use crate::data;
use crate::cli::Commands::{FindMotifs, Infer, InitResources, MotifsToTensor, PlotMotifSlices, RunAll, SelectMotifs, TrainFinal, TrainOnLabel, TrainOnMotifs};


use crate::version::VERSION;

#[derive(Parser, Clone)]
#[command(version, about, long_about = None)]
pub struct Cli {//TODO pub
    #[arg(long)]
    pub base_path: String,

    #[arg(long)]
    pub resource_pattern: String,

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
    // #[cfg(test)]
    pub fn dummy(train_resource_group: &str, valid_resource_group: &str) -> Self {
        Self {
            train_resource_group: train_resource_group.to_string(),
            valid_resource_group: valid_resource_group.to_string(),
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
pub enum Commands {
    InitResources (InitResourcesParams),
    FindMotifs (FindMotifsParams),
    PlotMotifSlices(PlotMotifSlicesParams),
    MotifsToTensor(MotifsToTensorParams),
    SelectMotifs(SelectMotifsParams),
    TrainOnMotifs(TrainOnMotifsParams),
    TrainOnLabel(TrainOnLabelParams),
    TrainFinal(TrainFinalParams),
    MotifsWithTpe(RunAllParams),
    RunAll(RunAllParams),
    Infer(InferParams),
}

pub fn run_cli<F, L>(
    labeled_resourses:F,
    labels:Vec<&str>,
    parquet_to_tensor_by_path:L,
    channels: i64
) -> Result<(), ThoosaError>
    where F:Fn(&dyn AsRef<Path>) -> Result<HashMap<String, Vec<(String, f32)>>, ThoosaError>,
            L:Fn(&Path,Device) -> Result<Tensor, ThoosaError> + Sync
{
    let cli = Cli::parse();

    CombinedLogger::init(
        vec![
            TermLogger::new(LevelFilter::from_str(cli.term_logger_level.as_str())?, Config::default(), TerminalMode::Mixed, ColorChoice::Auto),
            WriteLogger::new(LevelFilter::from_str(cli.write_logger_level.as_str())?, Config::default(), File::create(&cli.logfile)?),
        ]
    ).expect("failed to init logging");

    let store = Arc::new(Mutex::new(SqlStore::from_path(&cli.store_path)?));
    let base_path = cli.base_path.clone();
    let resource_pattern = cli.resource_pattern.clone();

    //TODO replace with executor_rc
    let loader = Arc::new(Box::new(ParquetLoader::new(
        cli.base_path.clone(),
        cli.tensors_path.clone(),
        Some(Box::new(move |t: Tensor| {
            match cli.input_filters_kernel {
                None => {
                    t
                }
                Some(kernel) => {
                    debug!("main: loading with kernel={}", kernel);
                    t.avg_pool1d(kernel, 1, 0, false, false)
                }
            }
        })),
        Box::new(move |_, resource_id| resource_path(&base_path, &resource_pattern, resource_id))
    )));

    let devices = list_devices();

    let cpu_num = std::thread::available_parallelism()?;
    let num_threads = cli.num_threads.unwrap_or_else(||max(2, cpu_num.get()/2));

    info!("VERSION: {VERSION} devices={devices:?}, base_path={:?} cpus={cpu_num} num_threads={num_threads} limit={:?}",  cli.base_path, cli.limit);
    std::io::stdout().flush()?;

    //TODO optional for benchmark
    rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global()?;
    let base_path = cli.base_path.clone();
    let resource_pattern = cli.resource_pattern.clone();
    let exec_loader = ParquetLoader::new(
        cli.base_path.clone(),
        cli.tensors_path.clone(),
        None,
        Box::new(move |_, resource_id| resource_path(&base_path, &resource_pattern, resource_id))
    );

    let executor_rc = Arc::new(parking_lot::RwLock::new(LoadRunner::new(
        move |key: &String| {

            SyncWrapper::new(
                exec_loader.load(key, Device::Cpu),
                |e| Box::new(e.shallow_clone())
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

            let run_id = run_find_motifs_on_set(&cli, &store, devices[0], find_motifs_params, Arc::clone(&executor_rc));

            info!("run_find_motifs_on_set done, run_id={:?}", run_id);
        }

        Some(SelectMotifs(select_motifs_params)) => {
            info!("select_motifs_params={:?} devices={:?}", select_motifs_params, devices);

            let select_motifs_params = if let Some(train_resource_group) = select_motifs_params.train_resource_group() {
                let train_labels_datasets = { store.lock().list_group_labeled_resources_sets(train_resource_group.as_str())? };
                select_motifs_params.with_labels_datasets(train_labels_datasets)
            } else {
                select_motifs_params.clone()
            };

            let corr_ids = run_select_motifs(&cli, &store, devices, &select_motifs_params, Arc::clone(&executor_rc), channels);

            info!("run_select_motifs done, corr_ids={:?}", corr_ids);
        }

        Some(TrainOnMotifs(train_on_motif_params)) => {
            debug!("train_on_motif: train_on_motif_params={:?}", train_on_motif_params);

            run_train_on_motifs(&cli, &store, devices, train_on_motif_params, Arc::clone(&loader))?;
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

            run_train_on_label(&cli, vec![], &store, devices, &train_on_label_params, Arc::clone(&loader))?;
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
                    Arc::clone(&store),
                    labels,
                    ||labeled_resourses(&cli.base_path)
                )?;
            }

            //run_id - params set
            let run_ids = run_find_motifs_on_group(&cli, &store, devices.clone(), run_all_params, Arc::clone(&executor_rc))?;

            info!("find_motifs done: run_ids={:?}", run_ids);

            let train_labels_datasets = {store.lock().list_group_labeled_resources_sets(run_all_params.train_resource_group())?};
            let valid_labels_datasets = {store.lock().list_group_labeled_resources_sets(run_all_params.valid_resource_group())?};

            info!("run all: train_labels_datasets={:?}", train_labels_datasets);
            info!("run all: valid_labels_datasets={:?}", valid_labels_datasets);

            let corr_ids = run_select_motifs(
                &cli,
                &store,
                devices.clone(),
                &SelectMotifsParams::dummy(run_ids, train_labels_datasets.clone(), run_all_params.select_motifs_min_ratio),
                Arc::clone(&executor_rc), channels
            )?;

            info!("select_motifs done: corr_ids={:?}", corr_ids);

            let run_ids = run_train_on_motifs(&cli, &store, devices.clone(), &TrainOnMotifsParams::dummy(corr_ids, 300), Arc::clone(&loader))?;

            info!("run_train_on_motifs done");

            run_train_on_label(
                &cli,
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

            info!("run_train_on_label done");

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

            let (labels,  motif_node_params) =  {
                let mut store_guard = store.lock();
                (
                    store_guard.list_labels().expect("store_guard"),
                    MotifNodeParams::from_properties_map(store_guard.list_run_params(777, None).expect("list_run_params"))
                )
            };

            let t = infer(
                &infer_params,
                |path:&Path|{
                    // data::parquet_to_tensor_by_path(<Path as AsRef<Path>>::as_ref(path), *devices.first().expect("first device"))
                    parquet_to_tensor_by_path(path, *devices.first().expect("first device"))
                },
                infer_params.window() as i64,
                infer_params.step(),
                infer_params.motif_batch_size(),
                channels,
                684400,//TODO
                &devices[0],
                labels,
                motif_node_params
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
            motifs_to_tensor_cmd(&cli, &store, *devices.first().expect("first device"), choose_fragments_params, Arc::clone(&executor_rc))?;
        }

        Some(Commands::MotifsWithTpe(motifs_with_tpe_params)) => {
            debug!("MotifsWithTPE: motifs_with_tpe_params={:?}", motifs_with_tpe_params);

            motifs_with_tpe(&cli, motifs_with_tpe_params, &store, devices[0], Arc::clone(&executor_rc));
        }

        None => {
            warn!("No command");
        }
    }

    Ok(())
}

pub fn run_train_final(cli: &Cli, store: &Arc<Mutex<SqlStore>>, device: &Device, train_final_params: &TrainFinalParams, loader: Arc<Box<impl ResourceLoader<K=String, V=Tensor>>>) -> Result<(), ThoosaError> {

    let labels = {store.lock().list_labels()? };

    debug!("run_train_final: labels={:?}", labels);

    let vs = train_final(
        &cli.ws_path,
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

pub fn run_train_on_label(cli: &Cli, input_run_ids: Vec<usize>, store: &Arc<Mutex<SqlStore>>, devices: Vec<Device>, train_on_label_params: &TrainOnLabelParams, loader: Arc<Box<impl ResourceLoader<K=String, V=Tensor>>>) -> Result<(), ThoosaError> {
    debug!("run_train_on_label: train_on_label_params={:?}", train_on_label_params);
    // let labels_kinds = store_guard.corrs_descr(train_on_label_params.corr_run())?;


    let valid_datasets_map = train_on_label_params.valid_labels_datasets().into_iter().fold(HashMap::new(), |mut a, v|{
        a.insert(v.0.clone(), v.1.clone());
        a
    });

    if valid_datasets_map.len() != train_on_label_params.train_labels_datasets().len() {
        error!("valid_datasets_map={:?}", valid_datasets_map);
        error!("train_on_label_params.train_labels_datasets()={:?}", train_on_label_params.train_labels_datasets());
        panic!("valid_datasets_map.len() != train_on_label_params.train_labels_datasets().len()");
    } //TODO check keys

    let label_datasets_with_run_id = train_on_label_params.train_labels_datasets().into_iter().flat_map( |ds| input_run_ids.iter().map(|input_run_id|((ds.0.clone(), ds.1.clone()), *input_run_id))).collect::<Vec<((String, String), usize)>>();
    let chunk_len = max(1, label_datasets_with_run_id.len()/devices.len());

    let devices_train_labels_datasets = devices.into_iter().cycle().zip(label_datasets_with_run_id.chunks(chunk_len).into_iter()).collect::<Vec<(Device, &[((String, String), usize)])>>();//corrs_run(id, label)
    let store_in_threads =  Arc::clone(&store);

    devices_train_labels_datasets.into_par_iter().for_each(move |(device_i, chunk_resource_sets)| {

        for ((label, resource_set), input_run_id) in chunk_resource_sets {

            let motif_node_params =  {
                let mut store_guard = store_in_threads.lock();
                MotifNodeParams::from_properties_map(unprefix_properties("motifs_node", &store_guard.list_run_params(*input_run_id, None).expect("list_run_params")))
            };

            let channels_motifs_models = load_motif_models(Path::new(&cli.ws_path).join("motifs"), label.clone(), 10, &device_i, &motif_node_params).expect("load_motif_models");//TODO num channels, motifs to util
            debug!("train_on_label: models.size={:?}", channels_motifs_models.len());


            //TODO remove mutexes
            let motif_nets: Vec<Vec<Mutex<Box<dyn Module>>>> = channels_motifs_models.into_iter().map(|v|v.into_iter().map(|m|Mutex::new(m)).collect::<Vec<_>>()).collect::<Vec<_>>();

            let max_nets =  motif_nets.iter().map(|nets|nets.len()).max().unwrap_or(0) as i64;
            if max_nets <= 0 {
                panic!("no motif nets found");
            }

            let motif_batches = batch_windows(684400, motif_node_params.num_features() as usize, train_on_label_params.step(), train_on_label_params.motif_batch_size() as usize);
            let max_motif_batches = motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0};


            let (label_node_params_vec, _run_ids) = {
                let mut store_guard = store.lock();

                (0..1).fold((vec![], vec![]), |mut a, _| {//TODO expand
                    let label_node_params = LabelNodeParams::new(20, max_nets, max_motif_batches as i64, train_on_label_params.motif_batch_size());

                    a.1.push(store_guard.save_tree_run(Some(*input_run_id), "todo", &vec![], prefix_properties("motifs_node", label_node_params.to_properties_map())).expect("save_tree_run"));
                    a.0.push(label_node_params);

                    a
                })

            };

            info!("training {} on {} device={:?}", label, resource_set, device_i);
            for label_node_params in label_node_params_vec {
                match train_on_label(
                    &cli.ws_path,
                    &motif_node_params,
                    &label_node_params,
                    train_on_label_params,
                    labeled_tensors_by_descr(
                        label,
                        resource_set,
                        &device_i,
                        Arc::clone(&store),
                        Arc::clone(&loader)
                    ),
                    labeled_tensors_by_descr(
                        label,
                        valid_datasets_map.get(label).expect(format!("valid dataset for {}", label).as_str()),
                        &device_i,
                        Arc::clone(&store),
                        Arc::clone(&loader)
                    ),
                    train_on_label_params.items_per_label(),
                    train_on_label_params.window() as i64,
                    train_on_label_params.step(),
                    train_on_label_params.motif_batch_size(),
                    20,
                    684400,
                    &device_i,
                    label
                ) {
                    Ok(vs) => {
                        vs.save(Path::new(&cli.ws_path).join("label_models").join(format!("{}_{}.safetensors", label, train_on_label_params.corr_run()))).expect("save vs")//TODO corr_run-> run_id
                        // fs::write(Path::new(train_on_label_params.models_dir()).join(format!("{}_{}.json", label, train_on_label_params.corr_run())), serde_json::to_string(&node_params).expect("save params"));
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

fn run_train_on_motifs(cli: &Cli, store: &Arc<Mutex<SqlStore>>, devices: Vec<Device>, train_on_motif_params: &TrainOnMotifsParams, loader: Arc<Box<impl ResourceLoader<K=String, V=Tensor>>>) -> Result<Vec<usize>, ThoosaError> {
    let corr_run_ids = train_on_motif_params.corr_run_ids();

    let chunk_len = corr_run_ids.len()/devices.len();
    let corr_run_id_labels = { store.lock().list_corrs(corr_run_ids)? }; //(1..=36).collect::<Vec<usize>>();

    let devices_corr_run = devices.into_iter().cycle().zip(corr_run_id_labels.chunks(chunk_len).into_iter()).collect::<Vec<(Device, &[(usize, usize, String)])>>();//corrs_run(id, label)

    let cli_clone = cli.clone();
    let run_ids = Arc::new(Mutex::new(vec![]));
    let run_ids_for_threads = Arc::clone(&run_ids);

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

            run_ids_for_threads.lock().extend(run_ids);

            // fs::write(Path::new(train_on_motif_params.output_dir()).join(format!("motif_model_{}.json", run_id)), serde_json::to_string(&node_params).expect("serde_json::to_string"));
        }

        info!("TrainOnMotifs: corrs_run: {:?} device_i={:?} done ", corrs_run, device_i);
    });


    let guard = run_ids.lock();
    Ok(guard.to_vec())

}


pub fn run_select_motifs(cli: &Cli, store: &Arc<Mutex<SqlStore>>, devices: Vec<Device>, select_motifs_params: &SelectMotifsParams, executor: Arc<parking_lot::RwLock<LoadRunner<String, Box<Tensor>>>>, channels: i64) -> Result<Vec<usize>, ThoosaError>{

    let chunk_len = select_motifs_params.run_ids().len()/devices.len();

    let cli_clone = cli.clone();
    let corr_ids = Arc::new(Mutex::new(vec![]));
    let corr_ids_for_iter =  Arc::clone(&corr_ids);

    devices
        .into_iter()
        .cycle()
        .zip(select_motifs_params.run_ids().chunks(chunk_len).into_iter())
        .collect::<Vec<(Device, &[usize])>>()
        .into_par_iter()
        .for_each(move |(device_i, run_ids)| {

        for run_id in run_ids {
            let corr_ids = select_motifs_for_label(
                &cli_clone,
                Arc::clone(&store),
                device_i,
                &select_motifs_params,
                *run_id,
                Arc::clone(&executor),
                channels
            ).expect("select_motifs_in_class");

            corr_ids_for_iter.lock().extend(corr_ids);
        }
    });

    let corr_ids = corr_ids.lock().clone();
    Ok(corr_ids)
}


pub fn run_find_motifs_on_group(cli: &Cli, store: &Arc<Mutex<SqlStore>>, devices: Vec<Device>, run_all_params: &RunAllParams, executor: Arc<parking_lot::RwLock<LoadRunner<String, Box<Tensor>>>>) -> Result<Vec<usize>, ThoosaError>{

    let resource_sets = { store.lock().list_resource_set_group_sets(run_all_params.train_resource_group()).expect("list_resource_set_group_sets")};

    assert!(resource_sets.len() > 0, "{} is empty", run_all_params.train_resource_group());

    let chunk_len = resource_sets.len()/devices.len();
    let devices_resource_sets = devices.into_iter().cycle().zip(resource_sets.chunks(chunk_len).into_iter()).collect::<Vec<(Device, &[(usize, String)])>>();

    let run_ids = Arc::new(Mutex::new(vec![]));
    let run_ids_for_threads = Arc::clone(&run_ids);

    devices_resource_sets.into_par_iter().for_each(move |(device, chunk_resource_sets)| {

        for resource_set in chunk_resource_sets {
            let find_motifs_params = &run_all_params.find_motifs_params(resource_set.1.clone());

            match run_find_motifs_on_set(cli, &store, device, find_motifs_params, Arc::clone(&executor)) {
                Ok(run_id) => {
                    run_ids_for_threads.lock().push(run_id);
                }
                Err(e) => {
                    error!("run_find_motifs_on_group: resource_set={:?}, e={:?}", resource_set, e);
                }
            }
        }
    });

    let guard = run_ids.lock();
    Ok(guard.iter().flat_map(|v|v.iter().map(|v:&usize|*v)).collect())

}

fn run_find_motifs_on_set(cli: &Cli, store: &Arc<Mutex<SqlStore>>, _device: Device, find_motifs_params: &FindMotifsParams, executor: Arc<parking_lot::RwLock<LoadRunner<String, Box<Tensor>>>>) -> Result<Vec<usize>, ThoosaError> {

    let resource_ids = { store.lock().list_resource_set_resources_by_descr(find_motifs_params.resource_set()).expect("list_resource_set_resources") };

    let records_num = clc_records(&resource_ids, cli.base_path.as_str()).unwrap();
    debug!("run_find_motifs_on_set: records_num={records_num} resource_set={} resource_ids={resource_ids:?}", find_motifs_params.resource_set());

    find::find_motifs_by_euc_dist(
        find_motifs_params,
        records_num,
        resource_ids,
        Arc::clone(&store),
        20,
        executor
    )

}

fn motifs_to_tensor_cmd(cli: &Cli, store: &Arc<Mutex<SqlStore>>, device: Device, choose_fragments_params: &MotifsToTensorParams, executor: Arc<parking_lot::RwLock<LoadRunner<String, Box<Tensor>>>>) -> Result<(), ThoosaError> {
    let mut store_guard = store.lock();

    let t = motifs_to_tensor(
        store_guard.list_motifs_slices(choose_fragments_params.run_id())?,
        executor,
        choose_fragments_params.window(),
        20,
        device
    )?;

    let out = Path::new(&cli.ws_path).join("out");
    fs::create_dir_all(&out)?;
    t.0.save(out.join(format!("motifs_in_{}", choose_fragments_params.run_id())))?;

    Ok(())
}

pub fn motifs_with_tpe(cli: &Cli, run_all_params:&RunAllParams, store: &Arc<Mutex<SqlStore>>, device: Device, executor: Arc<parking_lot::RwLock<LoadRunner<String, Box<Tensor>>>>)-> Result<(), ThoosaError>{

    let run_ids = run_find_motifs_on_group(&cli, &store, vec![device], run_all_params, Arc::clone(&executor))?;

    info!("find_motifs done: run_ids={:?}", run_ids);

    let train_labels_datasets = {store.lock().list_group_labeled_resources_sets(run_all_params.train_resource_group())?};
    let valid_labels_datasets = {store.lock().list_group_labeled_resources_sets(run_all_params.valid_resource_group())?};

    info!("run all: train_labels_datasets={:?}", train_labels_datasets);
    info!("run all: valid_labels_datasets={:?}", valid_labels_datasets);

    let corr_ids = run_select_motifs(
        &cli,
        &store,
        vec![device],
        &SelectMotifsParams::dummy(run_ids, train_labels_datasets.clone(), run_all_params.select_motifs_min_ratio),
        Arc::clone(&executor), 20
    )?;

    Ok(())
}

pub fn init_resources<F>(store:Arc<Mutex<impl Store + 'static>>, labels:Vec<&str>, labeled_resourses:F) -> Result<(), ThoosaError>
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

fn plot_motif_slices(cli: &Cli, store: &Arc<Mutex<SqlStore>>, device: Device, plot_motif_slices_params: &&PlotMotifSlicesParams, loader: Arc<Box<impl ResourceLoader<K=String, V=Tensor>>>) -> Result<(), ThoosaError> {
    let mut store_guard = store.lock();

    let (channel, _run_id, slices) = store_guard.list_motif_slices(plot_motif_slices_params.motif_id(), None, None)?;
    let _base_path = cli.base_path.as_str();

    let slices = load_motif_slices(
        cli.base_path.as_str(), channel, 1000, slices,
        move |resource_id| {
            loader.load(&resource_id.to_string(), device)
        },
        device
    );

    let all_sices = slices.iter().flat_map(|(_id, slices)| slices.iter()).collect::<Vec<&Box<Tensor>>>();

    check_slices(&all_sices, 0.9);

    let plots_dir = Path::new(cli.ws_path.as_str()).join("plots");

    fs::create_dir_all(&plots_dir)?;

    plot_series(
        all_sices.into_iter().map(|t| {
            (format!("0"), BLACK, t.iter::<f64>().unwrap().enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>())
        }).collect(),
        format!("Motif {}", plot_motif_slices_params.motif_id()).as_str(),
        (10000, 10000),
        &plots_dir.join(format!("motif_{}.png", plot_motif_slices_params.motif_id()))
    ).unwrap();
    Ok(())
}

fn train_on_motifs(cli: &Cli, input_run_id: usize, corr_run: usize, store:Arc<Mutex<impl Store + 'static>>, device: Device, train_on_motif_params: &TrainOnMotifsParams, label: &String, loader: Arc<Box<impl ResourceLoader<K=String, V=Tensor>>>) -> Result<Vec<usize>, ThoosaError> {
    debug!("train_on_motifs: device={:?}, corr_run={:?}", device, corr_run);

    let (motifs, params) = {
        let mut store_guard = store.lock();

        // for motif_id in train_on_motif_params.motif_ids() {

        (
            store_guard.list_corr_motifs(corr_run)?//TODO param
                .into_iter()
                .flat_map(|channel_motifs| channel_motifs.into_iter())
                .filter(|(_motif_id, corr)| *corr > train_on_motif_params.min_corr())
                .map(|(motif_id, _corr)| motif_id).collect::<Vec<usize>>(),

            FilterAndSlicerParams::from_properties_map(store_guard.list_run_params(input_run_id, None)?)
        )
    };


    let output_dir = Path::new(&cli.ws_path).join("motifs").join(input_run_id.to_string()).join(label);//TODO corr_run

    fs::create_dir_all(&output_dir)?;


    let filter_and_slice:FilterAndSlicer = params.into();
    let filter_op = filter_and_slice.filter();

    let (node_params_vec, run_ids) = {
        let mut store_guard = store.lock();

        (0..1).fold((vec![], vec![]), |mut a, _| {//TODO expand
            let node_params = MotifNodeParams::new(filter_and_slice.window());

            a.1.push(store_guard.save_tree_run(Some(input_run_id), "todo", &vec![], prefix_properties("motifs_node", node_params.to_properties_map())).expect("save_tree_run"));
            a.0.push(node_params);

            a
        })

    };


    for motif_id in motifs.iter() {
        let (all_sices_0, all_sices_1, channel) = {
            let mut store_guard = store.lock();

            let (channel, run_id, slices) = store_guard.list_motif_slices(*motif_id, None, None)?;

            debug!("train_on_motif: 1. channel={}, run_id={}, slices={}", channel, run_id, slices.len());
            let loader_m = Arc::clone(&loader);
            let all_sices_1 = train_nets::slices_in_all_resources(
                load_motif_slices(
                    cli.base_path.as_str(), channel, filter_and_slice.window(), slices,
                    move |resource_id| {
                        // loader_m.load(&eeg_id.to_string(), device)

                        let t = loader_m.load(&resource_id.to_string(), device);

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
                    move |resource_id| {
                        loader_m.load(&resource_id.to_string(), device)
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
            let (loss,  vs) = train_nets::train_on_motif(
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
                fs::create_dir_all(&channel_models)?;

                vs.save(channel_models.join(format!("{}_{}_{:3}.safetensors", corr_run, motif_id, loss)))?;
            }
        }
    }

    Ok(run_ids)
}

fn select_motifs_for_label(cli: &Cli, store:Arc<Mutex<impl Store + 'static>>, device: Device, select_motifs_params: &SelectMotifsParams, run_id:usize, executor: Arc<parking_lot::RwLock<LoadRunner<String, Box<Tensor>>>>, channels: i64) -> Result<Vec<usize>, ThoosaError> {
    debug!("select_motifs_in_class: device={:?} cli.tensors_path={:?}", device, cli.tensors_path);

    let (resource_id_motif_slices, motifs_avg_distances, params) = {
        let mut store_guard = store.lock();

        (
            store_guard.list_motifs_slices(run_id)?,
            store_guard.list_motifs_distances(run_id)?,
            FilterAndSlicerParams::from_properties_map(store_guard.list_run_params(run_id, None)?)
        )
    };

    assert!(resource_id_motif_slices.len() > 0, "FAILED: no slices for run_id={}", run_id);

    let filter_and_slice:FilterAndSlicer = params.into();

    let (motifs, channel_motif_ids) = motifs_to_tensor(
        resource_id_motif_slices,

        Arc::clone(&executor),
        filter_and_slice.window() as i64,
        channels,
        device
    ).unwrap();

    info!("select_motifs_for_label: motifs tensor built, size={:?}", motifs.size());


    let channel_motif_ids = Arc::new(RwLock::new(channel_motif_ids));
    let labels_channels_good_motif_idxs:HashMap<String, Vec<Vec<(i64, f64)>>> = select_motifs_by_ratio(
        motifs,
        select_motifs_params.labels_datasets().into_iter().map(|label_dataset|{
            (
                label_dataset.0.clone(),
                store.lock().list_resource_set_resources_by_descr(label_dataset.1.as_str()).unwrap()
            )

        }).collect(),
        Arc::clone(&executor),
        cli.limit,
        filter_and_slice.step(),
        device,
        select_motifs_params.min_ratio(),
        Arc::new(RwLock::new(motifs_avg_distances)),//TODO keeps motif indexes with motif tensors
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

        store.lock().save_motifs_corrs(run_id, format!("{:?}", select_motifs_params.labels_datasets()).as_str(), motifs_corrs, label).expect("save_motifs_corrs")
    }).collect::<Vec<usize>>())

}


fn resource_path(base_path: &str, resource_pattern: &str, resource_key: &str) -> String { //TODO change to impl AsRef<Path> + in ParquetLoader
    Path::new(base_path).join(resource_pattern.replace("{}", resource_key)).into_os_string().into_string().unwrap()
}

#[cfg(test)]
mod main_tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;
    use log::debug;
    use parking_lot::{Mutex, RwLock};
    use polars::export::chrono;
    use tch::{Device, Tensor};
    use tch::Device::Cpu;
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::utils::{cdt_for_dir, lin_t, plot_series, vec_f32_from};
    use thoosa_motifs::find::{find_motifs_by_euc_dist, FindMotifsParams, SelectMotifsParams};
    use thoosa_shared::{saw, sig_0, sig_1, sig_2, ThoosaError};
    use thoosa_store::{SqlStore, Store};
    use crate::cli::{init_resources, motifs_with_tpe, resource_path, select_motifs_for_label, Cli, RunAllParams};
    use crate::cli::Commands::FindMotifs;
    use crate::data::ParquetLoader;

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
    fn test_motifs_with_tpe() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        fs::create_dir_all(&test_dir)?;
        debug!("test_dir={test_dir:?}");
        let base_path = test_dir.to_string_lossy().to_string();

        let n = 100i64;
        let max_y = 30.0;
        let sig_0_v = sig_0(n, max_y);
        let sig_1_v = sig_1(n, max_y);
        let sig_2_v = sig_2(n, max_y);

        let x = lin_t(n);

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
            // Tensor::from_slice([saw(n), saw(n), saw(n)].concat().as_slice()),
            Tensor::from_slice([sig_0_v.clone(), saw(n), saw(n)].concat().as_slice()),
            // Tensor::from_slice([saw(n),          sig_2_v.clone(), saw(n)].concat().as_slice()),
            // Tensor::from_slice([saw(n),          saw(n), saw(n)].concat().as_slice())
        ]);
        t_0_0.save(test_dir.join(format!("t_0_0.pt")))?;

        // debug!("t_0_0={t_0_0}");
        debug!("t_0_0 mean={} max-min={}", t_0_0.mean(None).double_value(&[]), t_0_0.max() - t_0_0.min());

        let t_0_1 = Tensor::vstack(&[
            Tensor::from_slice([saw(n), saw(n), sig_0_v.clone()].concat().as_slice()),
            // Tensor::from_slice([sig_0_v.clone(), saw(n), saw(n)].concat().as_slice()),
            // Tensor::from_slice([saw(n),          sig_2_v.clone(), saw(n)].concat().as_slice()),
            // Tensor::from_slice([saw(n),          saw(n), saw(n)].concat().as_slice()),
        ]);
        t_0_1.save(test_dir.join(format!("t_0_1.pt")))?;

        // debug!("t_0_1={t_0_1}");
        debug!("t_0_1 mean={} max-min={}", t_0_1.mean(None).double_value(&[]), t_0_1.max() - t_0_1.min());


        let t_1_0 = Tensor::vstack(&[
            Tensor::from_slice([saw(n), saw(n), saw(n)].concat().as_slice()),
            // Tensor::from_slice([sig_1_v.clone(), saw(n), saw(n)].concat().as_slice()),
            // Tensor::from_slice([saw(n),          sig_2_v.clone(), saw(n)].concat().as_slice()),
            // Tensor::from_slice([saw(n),          saw(n), saw(n)].concat().as_slice())
        ]);
        t_1_0.save(test_dir.join(format!("t_1_0.pt")))?;

        // debug!("t_1_0={t_1_0}");
        debug!("t_1_0 mean={} max-min={}", t_1_0.mean(None).double_value(&[]), t_1_0.max() - t_1_0.min());

        let t_1_1 = Tensor::vstack(&[
            Tensor::from_slice([saw(n), saw(n), saw(n)].concat().as_slice()),
            // Tensor::from_slice([sig_1_v.clone(), saw(n), saw(n)].concat().as_slice()),
            // Tensor::from_slice([saw(n),          sig_2_v.clone(), saw(n)].concat().as_slice()),
            // Tensor::from_slice([saw(n),          saw(n), saw(n)].concat().as_slice())
        ]);
        t_1_1.save(test_dir.join(format!("t_1_1.pt")))?;

        // debug!("t_1_1={t_1_1}");
        debug!("t_1_1 mean={} max-min={}", t_1_1.mean(None).double_value(&[]), t_1_1.max() - t_1_1.min());
        let channels = 1;

        let executor_rc = Arc::new(RwLock::new(LoadRunner::new(
            move |key: &String| {

                SyncWrapper::new(
                    Box::new(Tensor::load(test_dir.join(format!("{key}.pt"))).unwrap()),
                    |e| Box::new(e.shallow_clone())
                )
            }, 4, 100,
        )));

        // let store = Arc::new(Mutex::new(SqlStore::in_mem()?));
        std::fs::remove_file("/tmp/test_main_db.db3").unwrap_or(());
        let store = Arc::new(Mutex::new(SqlStore::from_path("/tmp/test_main_db.db3")?));

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

        let run_ids = find_motifs_by_euc_dist(

            &FindMotifsParams {
                step: 10,
                window: 80,
                min_dot: 0.0,
                resource_set: "test".to_string(),
                preload_d: 3,
                max_keys_fifo: 3,
                chunk_size: 2,
                max_dist: 0.1,
                input_filter: None,
            },
            3 * n as usize,
            vec!["t_0_0".into(), "t_0_1".into(), "t_1_0".into(), "t_1_1".into()],
            Arc::clone(&store),
            channels,
            Arc::clone(&executor_rc)
        )?;

        debug!("{:?}", store.lock().list_run_resources(1)?);
        let slices = store.lock().list_motifs_slices(run_ids[0])?;
        debug!("slices: {:?}", slices);

        // assert_eq!((110, 0), slices.get(&"t_0".to_string()).and_then(|v| v.get(0).map(|v|(v.slise_idx, v.channel_idx))).unwrap());
        // assert_eq!(&MotifSlice { motif_id: 1, slise_idx: 1100, channel_idx: 0 }, slices.get(&"t_1".to_string()).and_then(|v| v.get(0)).unwrap());


        /*
        let cli = Cli{
            base_path,
            resource_pattern: "{}.parquet".to_string(),
            ws_path: "/tmp".to_string(),
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

        let corr_ids = select_motifs_for_label(
            &cli,
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
            Arc::clone(&executor_rc),
            channels
        ).expect("select_motifs_in_class");

        println!("corr_ids={corr_ids:?}");
        */

        Ok(())

    }


}