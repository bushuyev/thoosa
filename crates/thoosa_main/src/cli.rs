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
use futures_util::{AsyncReadExt, StreamExt};
use parking_lot::RwLock;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tch::Device::Cpu;
use thoosa_common::loader::{LoadRunner, SyncWrapper};
use thoosa_common::{train_nets, ResourceLoader};
use thoosa_common::train_nets::{infer, load_all_label_models_ws, load_motif_models_ws, train_final, train_on_label, InferParams, LabelNodeParams, MotifNodeParams, SyncModule, TrainFinalParams, TrainOnLabelParams, TrainOnMotifsParams};
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
use thoosa_common::ws::Workspace;

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
    channels: usize
) -> Result<(), ThoosaError>
    where F:Fn(&dyn AsRef<Path>) -> Result<HashMap<String, Vec<(String, f32)>>, ThoosaError>,
            L:Fn(&Path,Device) -> Result<Tensor, ThoosaError> + Sync
{
    let cli = Cli::parse();

    let ws = Arc::new(RwLock::new(Workspace::new(cli.base_path.clone(), 20)?));//TODO channels

    CombinedLogger::init(
        vec![
            TermLogger::new(LevelFilter::from_str(cli.term_logger_level.as_str())?, Config::default(), TerminalMode::Mixed, ColorChoice::Auto),
            WriteLogger::new(LevelFilter::from_str(cli.write_logger_level.as_str())?, Config::default(), File::create(&cli.logfile)?),
        ]
    ).expect("failed to init logging");

    let store = Arc::new(Mutex::new(SqlStore::from_path(&cli.store_path)?));
    // let base_path = ws.bas.clone();
    let resource_pattern = cli.resource_pattern.clone();

    let ws_loader = Arc::clone(&ws);
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
        Box::new(move |_, resource_id| resource_path(ws_loader.read().base_path().as_os_str().to_str().unwrap(), &resource_pattern, resource_id))//TODO move resource_path into ws
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

    let executor_rc = Arc::new(RwLock::new(LoadRunner::new(
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

            let corr_ids = run_select_motifs(&cli, &store, devices, &select_motifs_params, Arc::clone(&executor_rc), channels as i64);

            info!("run_select_motifs done, corr_ids={:?}", corr_ids);
        }

        Some(TrainOnMotifs(train_on_motif_params)) => {
            debug!("train_on_motif: train_on_motif_params={:?}", train_on_motif_params);

            run_train_on_motifs(&Arc::clone(&ws), &store, devices, train_on_motif_params, Arc::clone(&executor_rc))?;
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

            run_train_on_label_on_single_motif_selection(Arc::clone(&ws), &store, devices, &train_on_label_params, Arc::clone(&executor_rc))?;
        }

        Some(TrainFinal(train_final_params)) => {

            debug!("train_final_params={:?}", train_final_params);

            run_train_final(Arc::clone(&ws), &store, &devices[0], train_final_params, Arc::clone(&executor_rc))?;
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

            let selected_motifs_run_ids = run_select_motifs(
                &cli,
                &store,
                devices.clone(),
                &SelectMotifsParams::dummy(run_ids, train_labels_datasets.clone(), run_all_params.select_motifs_min_ratio),
                Arc::clone(&executor_rc), ws.read().channels() as i64
            )?;

            info!("select_motifs done: selected_motifs_run_ids={:?}", selected_motifs_run_ids);

            let moitif_run_ids = run_train_on_motifs(&ws, &store, devices.clone(), &TrainOnMotifsParams::dummy(selected_motifs_run_ids, 300, 0.2, 10, 1.0e-5, 1), Arc::clone(&executor_rc))?;

            info!("run_train_on_motifs done");

            run_train_on_label_on_single_motif_selection(
                Arc::clone(&ws),
                &store,
                devices.clone(),
                &TrainOnLabelParams::dummy(
                    run_all_params.step,
                    moitif_run_ids,
                    train_labels_datasets.clone(),
                    valid_labels_datasets.clone(),
                    run_all_params.motif_batch_size,
                    5,
                    100,
                    1.0e-5,
                ),
                Arc::clone(&executor_rc)
            )?;

            info!("run_train_on_label done");

            // run_train_final(
            //     &cli, &store,
            //     &devices[0],
            //     &TrainFinalParams::dummy(
            //         run_all_params.step,
            //         run_all_params.window,
            //         vec![run_all_params.models_dir.clone()],
            //         format!("{}/final.safetensors", run_all_params.models_dir),
            //         run_all_params.train_resource_group.clone(),
            //         run_all_params.valid_resource_group.clone(),
            //         run_all_params.train_final_cycles,
            //         run_all_params.motif_batch_size,
            //         5,
            //     ),
            //     Arc::clone(&loader)
            // )?;
            //
            // info!("run_train_final done");
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
            plot_motif_slices(&cli, store, *devices.first().expect("first device"), &plot_motif_slices_params, Arc::clone(&executor_rc))?;
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

pub fn run_train_final(ws: Arc<RwLock<Workspace>>, store: &Arc<Mutex<SqlStore>>, device: &Device, train_final_params: &TrainFinalParams, executor: Arc<RwLock<LoadRunner<String, Box<Tensor>>>>) -> Result<(), ThoosaError> {

    let labels = {store.lock().list_labels()? };

    let label_models = load_all_label_models_ws(&ws.read(),  &Cpu).expect("load_all_label_models_ws");

    debug!("run_train_final: labels={:?}", labels);

    let vs = train_final(
        Arc::clone(&ws),
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
        20,
        684400,
        &device,
        labels,
        executor,
        label_models
    )?;

    vs.save(ws.read().final_models_path())?;
    Ok(())
}

//TODO implement training label model on multiple motif selection
pub fn run_train_on_label_on_single_motif_selection(ws: Arc<RwLock<Workspace>>, store: &Arc<Mutex<SqlStore>>, devices: Vec<Device>, train_on_label_params: &TrainOnLabelParams, executor: Arc<RwLock<LoadRunner<String, Box<Tensor>>>>) -> Result<(), ThoosaError> {
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


    //each motif_selection_id trained with label datasets
    let label_datasets_with_motif_model_run_id = train_on_label_params.train_labels_datasets()
        .into_iter()
        .flat_map( |ds| train_on_label_params.motif_run_ids().iter().map(|motif_run_id|((ds.0.clone(), ds.1.clone()), *motif_run_id)))
        .collect::<Vec<((String, String), usize)>>();

    let chunk_len = max(1, label_datasets_with_motif_model_run_id.len()/devices.len());

    let devices_train_labels_datasets_motif_models_run_ids = devices.into_iter().cycle().zip(label_datasets_with_motif_model_run_id.chunks(chunk_len).into_iter()).collect::<Vec<(Device, &[((String, String), usize)])>>();
    let store_in_threads =  Arc::clone(&store);


    devices_train_labels_datasets_motif_models_run_ids.into_par_iter().for_each(move |(device_i, chunk_resource_sets)| {

        for ((label, resource_set), motif_model_run_id) in chunk_resource_sets {

            let channels_motifs_models = load_motif_models_ws(&ws.read(), *motif_model_run_id, &Cpu).expect("load_motif_models");//TODO num channels, motifs to util
            let channels_motifs_models: Arc<RwLock<Vec<Vec<SyncModule>>>> = Arc::new(RwLock::new(channels_motifs_models)); //channels_motifs_models.into_iter().map(|v|v.into_iter().map(|m|Mutex::new(m)).collect::<Vec<_>>()).collect::<Vec<_>>();


            let motif_node_params =  {
                let mut store_guard = store_in_threads.lock();
                MotifNodeParams::from_properties_map(unprefix_properties("motifs_node", &store_guard.list_run_params(*motif_model_run_id, None).expect("list_run_params")))
            };

            // let channels_motifs_models = load_motif_models(ws.read().models_path(), label.clone(), 10, &device_i, &motif_node_params).expect("load_motif_models");//TODO num channels, motifs to util
            // debug!("train_on_label: models.size={:?}", channels_motifs_models.len());


            let max_nets =  channels_motifs_models.read().iter().map(|nets|nets.len()).max().unwrap_or(0) as i64;
            if max_nets <= 0 {
                panic!("no motif nets found");
            }

            let motif_batches = batch_windows(684400, motif_node_params.num_features() as usize, train_on_label_params.step(), train_on_label_params.motif_batch_size() as usize);
            let max_motif_batches = motif_batches.0.len() + if motif_batches.1.len() > 0 {1} else {0};


            let (run_ids_label_node_params) = {
                let mut store_guard = store.lock();

                (0..1).fold((vec![]), |mut a, _| {//TODO expand
                    let label_node_params = LabelNodeParams::new(ws.read().channels() as i64, max_nets, max_motif_batches as i64, train_on_label_params.motif_batch_size());

                    a.push(
                        (
                            store_guard.save_tree_run(Some(*motif_model_run_id), "todo", &vec![], prefix_properties("label_node", label_node_params.to_properties_map())).expect("save_tree_run"),
                            label_node_params
                        )
                    );

                    a
                })
            };

            info!("training {} on {} device={:?}", label, resource_set, device_i);
            for (run_id, label_node_params) in run_ids_label_node_params {

                match train_on_label(
                    Arc::clone(&ws),
                    &motif_node_params,
                    &label_node_params,
                    train_on_label_params,
                    labeled_tensors_by_descr(
                        label,
                        resource_set,
                        Arc::clone(&store)
                    ),
                    labeled_tensors_by_descr(
                        label,
                        valid_datasets_map.get(label).expect(format!("valid dataset for {}", label).as_str()),
                        Arc::clone(&store)
                    ),
                    &device_i,
                    Arc::clone(&executor),
                    Arc::clone(&channels_motifs_models)
                ) {
                    Ok((loss, (vs, params))) => {
                        ws.read().save_label_model(run_id, label, vs, params).expect("save_label_model failed");
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

fn run_train_on_motifs(ws: &Arc<RwLock<Workspace>>, store: &Arc<Mutex<SqlStore>>, devices: Vec<Device>, train_on_motif_params: &TrainOnMotifsParams, executor: Arc<parking_lot::RwLock<LoadRunner<String, Box<Tensor>>>>) -> Result<Vec<usize>, ThoosaError> {
    let selected_motifs_run_ids = &train_on_motif_params.motif_selection_run_id;


    let chunk_len = selected_motifs_run_ids.len()/devices.len();
    // let motif_selection_ids_labels = { store.lock().list_motif_selections(motif_selection_ids)? }; //(1..=36).collect::<Vec<usize>>();

    debug!("run_train_on_motifs: selected_motifs_run_ids={selected_motifs_run_ids:?} devices={devices:?}");

    let devices_motif_selection = devices
        .into_iter()
        .cycle()
        .zip(
            selected_motifs_run_ids.chunks(chunk_len).into_iter()
        ).collect::<Vec<(Device, &[(usize)])>>();


    let run_ids = Arc::new(Mutex::new(vec![]));
    let run_ids_for_threads = Arc::clone(&run_ids);

    devices_motif_selection.into_par_iter().for_each( |(device_i, selected_motifs_run_ids_chunk)| {

        for (selected_motifs_run_id) in selected_motifs_run_ids_chunk {
            let input_run_id = store.lock().get_input_run_id(*selected_motifs_run_id).expect("motif_selection_id failed");
            debug!("run_train_on_motifs: input_run_id={input_run_id} motif_selection_id={selected_motifs_run_id}");
            let run_ids = train_on_motifs(
                Arc::clone(&ws),
                input_run_id,
                *selected_motifs_run_id,
                Arc::clone(&store),
                device_i,
                train_on_motif_params,
                Arc::clone(&executor)
            ).expect("train_on_motifs");

            run_ids_for_threads.lock().extend(run_ids);

        }

        info!("TrainOnMotifs: selected_motifs_run_id: {:?} run_ids_for_threads={:?} device_i={:?} done ", selected_motifs_run_ids_chunk, run_ids_for_threads, device_i);
    });


    let guard = run_ids.lock();
    Ok(guard.to_vec())

}


pub fn run_select_motifs(cli: &Cli, store: &Arc<Mutex<SqlStore>>, devices: Vec<Device>, select_motifs_params: &SelectMotifsParams, executor: Arc<RwLock<LoadRunner<String, Box<Tensor>>>>, channels: i64) -> Result<Vec<usize>, ThoosaError>{

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
        executor,
        Some(10)//TODO to params
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

fn plot_motif_slices(cli: &Cli, store: Arc<Mutex<SqlStore>>, device: Device, plot_motif_slices_params: &PlotMotifSlicesParams, executor: Arc<parking_lot::RwLock<LoadRunner<String, Box<Tensor>>>>) -> Result<(), ThoosaError> {
    let mut store_guard = store.lock();

    let (channel, _run_id, slices) = store_guard.list_motif_slices(plot_motif_slices_params.motif_id(), None, None)?;


    let slices = load_motif_slices(
        cli.base_path.as_str(), channel, plot_motif_slices_params.window as i64, slices,
        executor,
        device
    );

    let all_sices = slices.iter().flat_map(|(_id, slices)| slices.iter()).collect::<Vec<&Box<Tensor>>>();

    // check_slices(&all_sices, 0.9);

    let plots_dir = Path::new(cli.ws_path.as_str()).join("plots");

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

fn train_on_motifs(ws: Arc<RwLock<Workspace>>, find_motifs_run_id: usize, selected_motifs_run_id: usize, store:Arc<Mutex<impl Store + 'static>>, device: Device, train_on_motif_params: &TrainOnMotifsParams, executor: Arc<RwLock<LoadRunner<String, Box<Tensor>>>>) -> Result<Vec<usize>, ThoosaError> {
    debug!("train_on_motifs: device={device:?}, find_motifs_run_id={find_motifs_run_id} selected_motifs_run_id={selected_motifs_run_id:?}");

    let (motifs, params) = {
        let mut store_guard = store.lock();

        (
            store_guard.list_selected_motifs(selected_motifs_run_id)?//TODO param
                .into_iter()
                .flat_map(|channel_motifs| channel_motifs.into_iter())
                .filter(|(_motif_id, corr)| *corr > train_on_motif_params.min_corr)
                .map(|(motif_id, _corr)| motif_id).collect::<Vec<usize>>(),

            FilterAndSlicerParams::from_properties_map(store_guard.list_run_params(find_motifs_run_id, None)?)
        )
    };

    debug!("train_on_motifs: motifs={motifs:?}");

    let filter_and_slice:FilterAndSlicer = params.into();

    let (run_ids_node_params) = {
        let mut store_guard = store.lock();

        (0..1).fold((vec![]), |mut a, _| {//TODO expand grid search params
            let node_params = MotifNodeParams::new(filter_and_slice.window());

            a.push(
                (
                 store_guard
                     .save_tree_run(Some(find_motifs_run_id), "todo", &vec![], prefix_properties("motifs_node", node_params.to_properties_map()))
                     .expect("save_tree_run"),
                 node_params
                )
            );

            a
        })

    };


    for motif_id in motifs.iter() {
        let (all_sices_0, all_sices_1, channel) = {
            let mut store_guard = store.lock();

            let (channel, run_id, slices) = store_guard.list_motif_slices(*motif_id, None, None)?;

            assert_eq!(run_id, find_motifs_run_id);

            //TODO move load to DataBatch to avoid preload?
            debug!("train_on_motif: 1. channel={}, find_motifs_run_id={}, slices={}", channel, find_motifs_run_id, slices.len());
            let all_sices_1 = train_nets::slices_in_all_resources(
                load_motif_slices(
                    ws.read().base_path_as_str().as_str(), channel, filter_and_slice.window(), slices,
                    Arc::clone(&executor),
                    device
                )
            );


            //list motifs which are NOT motif_id and NOT from run_id
            let (channel, run_id, slices) = store_guard.list_motif_slices(*motif_id, Some(all_sices_1.len()), Some((find_motifs_run_id, channel as usize)))?;

            debug!("train_on_motif: 0. channel={}, run_id={}, slices={}", channel, run_id, slices.len());

            let all_sices_0 = train_nets::slices_in_all_resources(
                load_motif_slices(
                    ws.read().base_path_as_str().as_str(), channel, filter_and_slice.window(), slices,
                    Arc::clone(&executor),
                    device
                )
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



        for (run_id, node_params) in &run_ids_node_params {
            debug!("train_on_motif: node_params={node_params:?}");

            let (loss,  (vs, params)) = train_nets::train_on_motif(
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
                ws.read().save_motif_model(channel, run_id, motif_id, vs, params)?;

            }
        }
    }

    Ok(run_ids_node_params.into_iter().map(|run_id_node_params|run_id_node_params.0).collect())
}

fn select_motifs_for_label(cli: &Cli, store:Arc<Mutex<impl Store + 'static>>, device: Device, select_motifs_params: &SelectMotifsParams, run_id:usize, executor: Arc<parking_lot::RwLock<LoadRunner<String, Box<Tensor>>>>, channels: i64) -> Result<Vec<usize>, ThoosaError> {
    debug!("select_motifs_for_label: device={:?}", device);

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

    debug!("select_motifs_for_label: filter_and_slice={filter_and_slice:?}");

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

        store.lock().save_selected_motifs(run_id, format!("{:?}", select_motifs_params.labels_datasets()).as_str(), motifs_corrs).expect("save_motifs_corrs")
    }).collect::<Vec<usize>>())

}


fn resource_path(base_path: &str, resource_pattern: &str, resource_key: &str) -> String { //TODO change to impl AsRef<Path> + in ParquetLoader
    Path::new(base_path).join(resource_pattern.replace("{}", resource_key)).into_os_string().into_string().unwrap()
}

#[cfg(test)]
mod main_tests {
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use log::debug;
    use parking_lot::{Mutex, RawMutex, RwLock};
    use parking_lot::lock_api::MutexGuard;
    use plotters::prelude::BLACK;
    use polars::export::chrono;
    use rayon::iter::ParallelIterator;
    use rayon::prelude::IntoParallelIterator;
    use tch::{nn, Device, IndexOp, Tensor};
    use tch::Device::Cpu;
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::train_nets::{motif_node, MotifNodeParams, TrainFinalParams, TrainOnLabelParams, TrainOnMotifsParams};
    use thoosa_common::utils::{cdt_for_dir, lin_t, plot_series, vec_f32_from};
    use thoosa_motifs::find::{find_motifs_by_euc_dist, FilterAndSlicerParams, FindMotifsParams, LinearExpander, SelectMotifsParams};
    use thoosa_motifs::fragments::PlotMotifSlicesParams;
    use thoosa_shared::{recurse_files, saw, sig_one_imp, sig_two_imp, sig_three_imp, ThoosaError};
    use thoosa_shared::val::unprefix_properties;
    use thoosa_store::{SqlStore, Store};
    use crate::cli::{init_resources, motifs_with_tpe, plot_motif_slices, resource_path, run_train_final, run_train_on_label_on_single_motif_selection, run_train_on_motifs, select_motifs_for_label, Cli, RunAllParams};
    use crate::cli::Commands::FindMotifs;
    use crate::data::ParquetLoader;
    use thoosa_common::ws::Workspace;

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
        t_0_0.save(test_dir.join(format!("t_0_0.pt")))?;

        // debug!("t_0_0 mean={} max-min={}", t_0_0.mean(None).double_value(&[]), t_0_0.max() - t_0_0.min());

        let t_0_1 = Tensor::vstack(&[
            Tensor::from_slice([saw(n), sig_0_v.clone(), saw(n) ].concat().as_slice()),
            Tensor::from_slice([saw(n),          sig_2_v.clone(), saw(n)].concat().as_slice()),
            Tensor::from_slice([sig_2_v.clone(),          saw(n), saw(n)].concat().as_slice()),
        ]);
        t_0_1.save(test_dir.join(format!("t_0_1.pt")))?;

        // debug!("t_0_1 mean={} max-min={}", t_0_1.mean(None).double_value(&[]), t_0_1.max() - t_0_1.min());


        let t_1_0 = Tensor::vstack(&[
            Tensor::from_slice([saw(n), sig_1_v.clone(), saw(n)].concat().as_slice()),
            Tensor::from_slice([saw(n),          saw(n), sig_2_v.clone()].concat().as_slice()),
            Tensor::from_slice([saw(n),          sig_2_v.clone(), saw(n)].concat().as_slice())
        ]);
        t_1_0.save(test_dir.join(format!("t_1_0.pt")))?;

        // debug!("t_1_0 mean={} max-min={}", t_1_0.mean(None).double_value(&[]), t_1_0.max() - t_1_0.min());

        let t_1_1 = Tensor::vstack(&[
            Tensor::from_slice([saw(n), saw(n), sig_1_v.clone()].concat().as_slice()),
            Tensor::from_slice([saw(n),          saw(n), sig_2_v.clone()].concat().as_slice()),
            Tensor::from_slice([sig_2_v.clone(),          sig_2_v.clone(), saw(n)].concat().as_slice()),
        ]);
        t_1_1.save(test_dir.join(format!("t_1_1.pt")))?;

        // debug!("t_1_1 mean={} max-min={}", t_1_1.mean(None).double_value(&[]), t_1_1.max() - t_1_1.min());
        let channels = 3;

        let executor_rc = Arc::new(RwLock::new(LoadRunner::new(
            move |key: &String| {

                SyncWrapper::new(
                    Box::new(Tensor::load(test_dir.join(format!("{key}.pt"))).unwrap()),
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

        let run_ids = find_motifs_by_euc_dist(

            &FindMotifsParams {
                step: 100,
                window: 800,
                min_dot: 0.0,
                resource_set: "test".to_string(),
                preload_d: 3,
                max_keys_fifo: 3,
                chunk_size: 2,
                max_dist: 0.5,
                input_filter: None,
            },
            3 * n as usize,
            vec!["t_0_0".into(), "t_0_1".into(), "t_1_0".into(), "t_1_1".into()],
            Arc::clone(&store),
            channels,
            Arc::clone(&executor_rc),
            Some(100)
        )?;

        let slices = store.lock().list_motifs_slices(run_ids[0])?;
        debug!("test_motifs_with_tpe: slices={:?}", slices);
        assert!(slices.len() >= 4);


        // assert_eq!((110, 0), slices.get(&"t_0".to_string()).and_then(|v| v.get(0).map(|v|(v.slise_idx, v.channel_idx))).unwrap());
        // assert_eq!(&MotifSlice { motif_id: 1, slise_idx: 1100, channel_idx: 0 }, slices.get(&"t_1".to_string()).and_then(|v| v.get(0)).unwrap());


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

        let select_motifs_run_ids = select_motifs_for_label(
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

        let ws = Arc::new(RwLock::new(Workspace::new(test_dir, 1)?));
        let ws_runner = Arc::clone(&ws);


        let n = 1000i64;
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
        fs::remove_file("/tmp/test_main_db.db3").unwrap_or(());
        let store = Arc::new(Mutex::new(SqlStore::from_path("/tmp/test_main_db.db3")?));
        init_test_store(store.lock());

        ///////////////////////////

        let find_motifs_params = FindMotifsParams::dummy(100, 800, "test set".into(), 10, 1.0, None);
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

        plot_motif_slices(&cli, Arc::clone(&store), Cpu, &PlotMotifSlicesParams::new(motif_ids.into_iter().next().unwrap(), 800), Arc::clone(&executor_rc));

        let motif_models_run_ids = run_train_on_motifs(
            &ws,
            &store,
            vec![Cpu],
            // &TrainOnMotifsParams::dummy(vec![corr_run_id], 100, 100.0, 10, 10e-5..10e-3, 1),
            // &TrainOnMotifsParams::dummy(vec![motif_selection_id], 10, 100.0, 10, 10e-9 .. 10e-5, 2..5),
            &TrainOnMotifsParams::dummy(vec![motif_selection_run_id], 10, 100.0, 10, 10e-5, 2),
            Arc::clone(&executor_rc),
        )?;

        debug!("test_run_train_on_motifs: motif_models_run_ids={motif_models_run_ids:?}");

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
        run_train_on_label_on_single_motif_selection(
            Arc::clone(&ws),
            &store,
            vec![Cpu],
            &TrainOnLabelParams::dummy(100, motif_models_run_ids, vec![
                ("Label_0".into(), "Label_0/6".into()),
                ("Label_1".into(), "Label_1/6".into())
            ], vec![
                ("Label_0".into(), "Label_0/4".into()),
                ("Label_1".into(), "Label_1/4".into())
            ],
100, 10, 10, 0.4),
            Arc::clone(&executor_rc),
        )?;

        /////////////////////FINAL
        run_train_final(
            Arc::clone(&ws),
            &store,
            &Cpu,
            &TrainFinalParams::dummy(100, 3, "pure group 6".into(), "pure group 4".into(), 10, 100, 10),
            Arc::clone(&executor_rc),
        )?;


        Ok(())
    }

    fn init_test_store(mut store_guard: MutexGuard<RawMutex, SqlStore>) -> Result<(), ThoosaError>{

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

    fn save_test_motifs(store: Arc<Mutex<SqlStore>>, find_motifs_params: FindMotifsParams, param_set: Vec<FilterAndSlicerParams>, resource_label_key: &str) -> Result<usize, ThoosaError> {
        let resource_0_ids = (0..10).map(|i| format!("{resource_label_key}{i}")).collect();
        let run_id = store.lock().save_tree_run(
            None,
            format!("find motifs with params {:?}/{:?}", find_motifs_params, param_set[0]).as_str(),
            &resource_0_ids,
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
}