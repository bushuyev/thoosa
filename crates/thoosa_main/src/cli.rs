use crate::cli::Commands::{
    FindMotifs, ImportResources, Infer, MotifsToTensor, PlotMotifSlices, RunAll, SayHello,
    SelectMotifs, TrainFinalNet, TrainMotifsNets,
};
use crate::data;
use crate::data::{ParquetLoader, clc_records, labeled_tensors_by_descr, load_to_logits_tensors};
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::{RawTensorLoader};
use crate::version::VERSION;
use async_trait::async_trait;
use caching_runner::common::runnable::Stage;
use caching_runner::global_ctx::GlobalCtx;
use clap::{Args, Parser, Subcommand};
use futures::executor::block_on;
use futures_util::future::{join_all, try_join_all};
use log::{Level, LevelFilter, debug, error, info, trace, warn};
use parking_lot::RwLock;
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use simplelog::{ColorChoice, CombinedLogger, Config, TermLogger, TerminalMode, WriteLogger};
use std::cmp::max;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::{File, OpenOptions, read_dir};
use std::io::Write;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use std::{fs, process};
use plotters::prelude::BLACK;
use tch::Device::Cpu;
use tch::Kind::Float;
use tch::nn::VarStore;
use tch::{Device, Tensor};
use thoosa_common::applicables::{ThoDevice, ThoRunCtx, ThoRunner};
use thoosa_common::batches::{
    AsyncBatchTransform, AsyncClassTensorsLoaderData, AsyncLoadingTensorsDataBatch,
};
use thoosa_common::dims::{Batch, ChannelsData, TCD, ThoBatch};
use thoosa_common::find_motifs_run_config::{
    FindMotifsApplicable, FindMotifsRunConfig, FindMotifsRunConfigRangeOrValue,
    SelectMotifsRunConfigRangeOrValue,
};
use thoosa_common::input_probs::InputProbs;
use thoosa_common::loader::{LoadRunner, SyncWrapper};
use thoosa_common::runs::{Empty, Labeled, Logits, NetResources, SavableResult, Simple};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets;
use thoosa_common::train_nets::{
    CommonTrainingRunConfig, FinalNetApplicable, FinalNetParams, InferArgs,
    LabelNetApplicableAndNetE, LabelNetInputTensors, LabelNetParams, LabelNetsApplicable,
    MotifNetParams, MotifsNetsSelectionApplicable, OptConfig, PlotVecf32, TensorsDataBatch,
    ThoModuleWithStore, TrainFinalNetArgs, TrainFinalNetRunConfig, TrainFinalNetRunConfigFields,
    TrainFinalNetRunConfigRangeOrValue, TrainFinalNetsRunConfigResult, TrainLabelNetsArgs,
    TrainLabelNetsRunConfig, TrainLabelNetsRunConfigFields, TrainLabelNetsRunConfigRangeOrValue,
    TrainLabelNetsRunConfigResult, TrainMotifsNetsArgs, TrainMotifsNetsRunConfig,
    TrainMotifsNetsRunConfigFields, TrainMotifsNetsRunConfigRangeOrValue,
    TrainMotifsNetsRunConfigResult, load_channels_motifs, load_final_model, opt_train, opt_train_2,
    run_opt_train_label_nets, run_train_motif_net, train_final_net, train_final_net_2,
    train_label_net_2,
};
use thoosa_common::utils::{DummyListener, TrainListener, cuda_mem, from, list_devices, lin_t, plot_series};
use thoosa_common::ws::{FsWorkspace, Workspace};
use thoosa_motifs::find;
use thoosa_motifs::find::{
    FindMotifsArgs, FixedFindMotifsConfig, SelectMotifsArgs, find_motifs_by_euc_dist_2,
    select_motifs_for_label,
};
use thoosa_motifs::fragments::{
    MotifsBy, MotifsToTensorArgs, PlotMotifSlicesArgs, load_motif_slices, load_motif_slices_2,
    motifs_to_tensor,
};
use thoosa_properties_map::ToHashmap;
use thoosa_run_config_args::{GridRunConfigSource, OptRunConfigSource};
use thoosa_shared::applicables_runner::{Input, Runner, ToOrdinal};
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::resource::{Resource, ResourceId, ResourceStats, ResourcesConfig};
use thoosa_shared::run_configs::{OptRunConfigs, RunConfig, RunId, Runnable};
use thoosa_shared::traits::{DataId, SimpleId};
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_shared::{ResourceType, ThoosaError, cdt, cdt_for_dir};
use thoosa_store::{SqlStore, Store};
use tokio::runtime::Runtime;
use torch_runner::data::ThoTensor2;
use torch_sys::at_grad_set_enabled;
use caching_runner::common::cachable::Cached;
use thoosa_motifs::distance::average_euclidean_distance;
use thoosa_shared::colors_iter::DistinctColorIter;
use crate::tasks::motif_loader::MotifLoader;

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

#[derive(Args, Debug, Clone)]
pub struct RunAllArgs {
    #[arg(long)]
    train_resource_group: String,

    #[arg(long)]
    valid_resource_group: String,

    #[arg(long)]
    pub run_config_cnt: String, //TrainLabelNetsRunConfig
}

/**
export LD_LIBRARY_PATH=/home/bu/dvl/torch/libtorch-2.5.0/lib/ && cargo run --release -- \
--term-logger-level=TRACE --write-logger-level=INFO  --ws-path="/tmp/ws_import_test"  \
--logfile=/tmp/ws_1/thoosa.log import-resources --source-path=/media/bu/KINGSTON/hbac_orig

*/
#[derive(Debug, Args, Clone, Default)]
pub struct ImportResourcesArgs {
    #[arg(long)]
    pub source_path: String,
}

#[derive(Debug, Args, Clone)]
pub struct SayHelloArgs {}

#[derive(Debug, Subcommand, Clone)]
pub enum Commands {
    ImportResources(ImportResourcesArgs),
    FindMotifs(FindMotifsArgs),
    PlotMotifSlices(PlotMotifSlicesArgs),
    MotifsToTensor(MotifsToTensorArgs),
    SelectMotifs(SelectMotifsArgs),
    TrainMotifsNets(TrainMotifsNetsArgs),
    TrainLabelNets(TrainLabelNetsArgs),
    TrainFinalNet(TrainFinalNetArgs),
    RunAll(RunAllArgs),
    Infer(InferArgs),
    SayHello(SayHelloArgs),
}

#[allow(dead_code)]
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

pub async fn run_cli<F, R>(
    labeled_resources: F,
    resource_to_tensor: R,
    labels: &[&str],
    channels: u32,
    max_input: u32,
    sample_rate: u32,
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
        sample_rate,
    )?));

    let logfile = match File::create(
        ws.read()
            .log_path()
            .join(format!("thoosa_{}.log", cdt_for_dir()).to_owned()),
    ) {
        Ok(file) => file,
        Err(_err) => {
            panic!("Can't create logfile in {:?}", cli.ws_path);
        }
    };
    if let Err(e) = CombinedLogger::init(vec![
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
    ]) {
        //attempted to set a logger after the logging system was already initialized - no #[cfg(test)] on tests mod
        panic!("Can't init log: {:?}", e.to_string());
    }

    let store = Arc::new(RwLock::new(SqlStore::from_path(ws.read().store_path())?));

    let devices = list_devices();

    let cpu_num = std::thread::available_parallelism()?;
    let loader_threads = cli
        .loader_threads
        .unwrap_or_else(|| max(2, cpu_num.get() / 2));

    //TODO torch.backends.cuda.matmul.allow_tf32 = True
    info!(
        "VERSION: {VERSION} devices={devices:?}, base_path={:?} cpus={cpu_num} loader_threads={loader_threads} tch::get_num_interop_threads={}, tch::get_num_threads={}",
        cli.ws_path,
        tch::get_num_interop_threads(),
        tch::get_num_threads()
    );
    std::io::stdout().flush()?;

    let tensors_path = ws.read().tensors_path().clone();
    let tensors_path_2 = tensors_path.clone();
    let tensors_path_runner = ws.read().tensors_path().clone();

    // let loader = Arc::new(RwLock::new(LoadRunner::new(
    //     move |key: &Resource| {
    //         let tensor = Tensor::load_multi_with_device(
    //             FsWorkspace::resource_tensor_path(&tensors_path, key),
    //             Cpu,
    //         )
    //         .map_err(|e| ThoosaError::new(e.to_string()))
    //         .expect(format!("can't load {key:?}").as_str())
    //         .pop()
    //         .unwrap()
    //         .1
    //         .totype(Float);
    //         if tensor.requires_grad() || tensor.retains_grad() {
    //             warn!("tensor requires grad!");
    //         }
    //         debug!("key={key:?} tensor.size={:?}", tensor.size());
    //         tensor.into()
    //     },
    //     loader_threads,
    //     cli.futures_channel_bound,
    //     2,
    // )));
    GlobalCtx::init_devices(2);

    let loader_2 = Arc::new(RawTensorLoader::new(
        move |key, _| {
            ThoTensor2::new(
                Input::id(key).clone(),
                Tensor::load(tensors_path_2.join(format!(
                    "{}_{}.pt",
                    key.import_id(),
                    key.file_stem()
                )))
                .unwrap(),
            )
        },
        ThoDevice::from(Device::Cpu),
    ));

    // let runner = Arc::new(ThoRunner::new(&[10, 2], move |key, device| {
    //     debug!("loader_fn: thread={:?}", std::thread::current().name());
    //     let tensor = Tensor::load_multi_with_device(
    //         FsWorkspace::resource_tensor_path(&tensors_path_runner, key),
    //         *device,
    //     )
    //     .map_err(|e| ThoosaError::new(e.to_string()))
    //     .expect(format!("can't load {key:?}").as_str())
    //     .pop()
    //     .unwrap()
    //     .1
    //     .totype(Float);
    //     if tensor.requires_grad() || tensor.retains_grad() {
    //         warn!("tensor requires grad!");
    //     }
    //     debug!("key={key:?} tensor.size={:?}", tensor.size());
    //     tensor.into()
    // }));

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

        Some(FindMotifs(find_motifs_args)) => {
            debug!(
                "FindMotifs2: find_motifs_args={:?} devices={:?}",
                find_motifs_args, devices
            );

            let find_motifs_run_config_range_or_value: FindMotifsRunConfigRangeOrValue =
                serde_json5::from_str(&find_motifs_args.run_config_cnt)
                    .expect("FindMotifsRunConfig2");

            debug!(
                "FindMotifs2: find_motifs_run_config_range_or_value={find_motifs_run_config_range_or_value:?}"
            );




            let run_ids = run_grid::<Simple, _, _, _, _, _>(
                None,
                find_motifs_run_config_range_or_value,
                Arc::new(ResourcesConfigInput::new(
                    SimpleId::from("resources"),
                    find_motifs_args.resource_config(),
                )),
                &store,
                true,
                |run_configs| {
                    Arc::new(crate::tasks::find_motifs::FindMotifsStage::new(
                        &ws,
                        &store,
                        loader_2,
                        run_configs,
                        FixedFindMotifsConfig::new(None, 2, false, None),
                        ThoDevice::from(1),
                    ))
                },
                || Arc::new(parking_lot::RwLock::new(RunCtx2::new(1, 0, "aa"))),
            )
            .await?;

            info!("FindMotifs done, run_ids={run_ids:?}");

        }

        Some(SelectMotifs(select_motifs_args)) => {
            // debug!(
            //     "SelectMotifs2: select_motifs_args.run_config_cnt={:?}",
            //     select_motifs_args.run_config_cnt
            // );
            //
            // let select_motifs_run_config_range_or_value: SelectMotifsRunConfigRangeOrValue =
            //     serde_json5::from_str(&select_motifs_args.run_config_cnt).expect("SelectMotifs2");
            //
            // let run_ids = run_all_grid_step(
            //     Some(select_motifs_args.find_motifs_run_ids.clone()),
            //     select_motifs_run_config_range_or_value,
            //     &(),
            //     &select_motifs_args.resource_config(),
            //     select_motifs_for_label,
            //     &ws,
            //     &store,
            //     &loader,
            //     devices.clone(),
            //     "SelectMotifs",
            //     false,
            // )?;
            //
            // info!("SelectMotifs done, run_ids={run_ids:?}");
        }

        Some(TrainMotifsNets(train_motif_nets_args)) => {
            // debug!(
            //     "TrainMotifsNets2: train_motif_nets_args.run_config_cnt={:?}",
            //     train_motif_nets_args.run_config_cnt
            // );
            // let train_motif_nets_run_config_range_or_value: TrainMotifsNetsRunConfigRangeOrValue =
            //     serde_json5::from_str(&train_motif_nets_args.run_config_cnt)
            //         .expect("TrainMotifsNets2");
            //
            // let run_ids = run_opt_step::<
            //     TrainMotifsNetsRunConfig,
            //     TrainMotifsNetsRunConfigRangeOrValue,
            //     OptConfig,
            //     Empty,
            //     _,
            // >(
            //     Some(train_motif_nets_args.motif_selection_run_id.clone()),
            //     train_motif_nets_run_config_range_or_value,
            //     &train_motif_nets_args.fixed_config,
            //     &train_motif_nets_args.resource_config(),
            //     run_train_motif_nodes_on_motif_selection,
            //     &ws,
            //     &store,
            //     &loader,
            //     devices.clone(),
            //     "TrainMotifsNets2",
            // )?;

            // info!("TrainMotifsNets done, run_ids={run_ids:?}");
        }

        Some(Commands::TrainLabelNets(train_label_nets_args)) => {
            debug!(
                "TrainMotifsNets2: train_motif_nets_args.run_config_cnt={:?}",
                train_label_nets_args.run_config_cnt
            );
            // let train_label_nets_run_config_range_or_value: TrainLabelNetsRunConfigRangeOrValue =
            //     serde_json5::from_str(&train_label_nets_args.run_config_cnt).expect(
            //         "TrainLabelNets2\
            // ",
            //     );
            //
            // run_opt_step::<
            //     TrainLabelNetsRunConfig,
            //     TrainLabelNetsRunConfigRangeOrValue,
            //     OptConfig,
            //     NetResources<Labeled>,
            //     _,
            // >(
            //     Some(train_label_nets_args.motif_selection_run_id.clone()),
            //     train_label_nets_run_config_range_or_value,
            //     &train_label_nets_args.fixed_config,
            //     &train_label_nets_args.resource_config(),
            //     run_train_label_nets,
            //     &ws,
            //     &store,
            //     &loader,
            //     devices.clone(),
            //     "TrainMotifsNets2",
            // )?;

            info!("TrainMotifsNets done");
        }

        Some(TrainFinalNet(train_final_net_args)) => {
            debug!(
                "TrainFinalNet: train_motif_nets_args.run_config_cnt={:?}",
                train_final_net_args.run_config_cnt
            );
            // let train_final_net_run_config_range_or_value: TrainFinalNetRunConfigRangeOrValue = serde_json5::from_str(&train_final_net_args.run_config_cnt).expect("TrainFinalNet");
            //
            // run_opt_step::<TrainFinalNetRunConfig, TrainFinalNetRunConfigRangeOrValue, OptConfig, NetResources<Logits>, _>(
            //     Some(train_final_net_args.train_labels_run_ids.clone()),
            //     train_final_net_run_config_range_or_value,
            //     &train_final_net_args.fixed_config,
            //     &train_final_net_args.resource_config(),
            //     run_train_final_nets,
            //     &ws,
            //     &store,
            //     &loader,
            //     devices.clone(),
            //     "TrainFinalNet",
            // )?;

            info!("TrainFinalNet done");
        }

        Some(RunAll(run_all_params)) => {
            debug!("RunAll: run_all_params={:?}", run_all_params);

            /*if let Some(source_path) = run_all_params.import_source_path.as_ref() {
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

            info!("run_train_final done: train_final_nodes_run_ids={train_final_nodes_run_ids:?}");*/
        }

        Some(Infer(infer_params)) => {
            // block_on(run_infer(
            //     &ws,
            //     &store,
            //     infer_params,
            //     devices[0],
            //     |path| {
            //         data::parquet_to_tensor_by_path(
            //             //TODO move load_tensor to project params
            //             <Path as AsRef<Path>>::as_ref(path),
            //             *devices.first().expect("first device"),
            //         )
            //     },
            //     &runner,
            // ))?;
        }

        Some(PlotMotifSlices(plot_motif_slices_params)) => {
            run_plot_motif_slices(&ws, &store, &loader_2, &plot_motif_slices_params, &devices[0]).await?;
        }

        Some(MotifsToTensor(choose_fragments_params)) => {
            // motifs_to_tensor_cmd(
            //     &cli,
            //     &store,
            //     *devices.first().expect("first device"),
            //     choose_fragments_params,
            //     &ws,
            //     &loader,
            // )?;
        }

        //gdb --args target/x86_64-unknown-linux-gnu/debug/thoosa_hbac --term-logger-level=DEBUG --base-path=/tmp/ --resource-pattern=xxx say-hello
        Some(SayHello(args)) => {
            debug!("args={:?}", args);

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

fn save_run_configs<R: RunConfig + thoosa_properties_map::ToHashmap>(
    input_run_ids: Option<&[RunId]>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    run_configs: &mut Vec<R>,
    descr: &str,
) -> Vec<RunId> {
    run_configs
        .into_iter()
        .map(|run_config| {
            let run_config_id = store
                .write()
                .save_run_config(
                    input_run_ids,
                    &<R>::type_name(),
                    descr,
                    &"todo",
                    run_config.to_properties_iter(None).collect(),
                )
                .unwrap();

            // run_config_ids.push(run_config_id);
            run_config.set_id(run_config_id);

            run_config_id
        })
        .collect()
}

#[derive(Debug)]
pub struct TrainFinalNetApplicable {
    // labels:Vec<String>,
    device: Device,
    label_nets_applicable: Arc<LabelNetsApplicable>,
}

impl TrainFinalNetApplicable {
    pub fn new(
        device: Device,
        /*labels:Vec<String>, */ label_nets_applicable: Arc<LabelNetsApplicable>,
    ) -> Self {
        Self {
            device,
            /*labels,*/ label_nets_applicable,
        }
    }
}

#[derive(Debug)]
pub struct TrainFinalNetInput {
    train_resources: Arc<Vec<(Resource, Vec<f32>)>>,
    valid_resources: Arc<Vec<(Resource, Vec<f32>)>>,
    train_label_nodes_run_ids: Option<Vec<RunId>>,
    run_configs: Arc<
        OptRunConfigs<
            Vec<NamedVal<TrainFinalNetRunConfigFields>>,
            Vec<NamedRange<TrainFinalNetRunConfigFields>>,
            Vec<Hyperparameter<TrainFinalNetRunConfigFields>>,
            TrainFinalNetRunConfigFields,
        >,
    >,
    fixed_config: Arc<OptConfig>,
}

#[async_trait]
impl Runnable<TrainFinalNetInput, Box<TrainFinalNetsRunConfigResult>, ()>
    for TrainFinalNetApplicable
{
    type R = ThoRunner;

    async fn async_run(
        self: Arc<Self>,
        input: Arc<TrainFinalNetInput>,
        runner: Arc<Self::R>,
        _ctx: Option<
            Arc<
                RwLock<
                    <<Self as Runnable<
                        TrainFinalNetInput,
                        Box<TrainFinalNetsRunConfigResult>,
                        (),
                    >>::R as Runner>::Ctx,
                >,
            >,
        >,
    ) -> Arc<Box<TrainFinalNetsRunConfigResult>> {
        let device = self.device.clone();

        let input_for_train = Arc::clone(&input);

        let label_nets_applicable = Arc::clone(&self.label_nets_applicable);
        opt_train_2(
            Arc::clone(&input.fixed_config),
            async move |vs:&VarStore, run_config_values:TrainFinalNetRunConfig, _listener|{


                let  batch = Arc::new(Box::new(AsyncClassTensorsLoaderData::<ChannelsData, ChannelsData>::new(
                    Arc::clone(&input_for_train.train_resources),
                    Arc::clone(&input_for_train.valid_resources),
                    run_config_values.common_training_args.batch_size as usize,
                    Arc::clone(&runner),
                    device.clone(),
                    // None,
                )));

                train_final_net_2(
                    &run_config_values,
                    label_nets_applicable.label_nets_num(),
                    label_nets_applicable.labels_num(),
                    &mut AsyncBatchTransform::new(Arc::clone(&batch), |t:ThoBatch<ChannelsData>|{
                        label_nets_applicable.run(t.as_ref().into(), ()).expect("label_nets_applicable.apply failed")
                    }),
                    vs,
                    &device
                ).await
                // Ok((0.0, FinalNetParams::new(0, 0, 0)))
            },
            Arc::clone(&input.run_configs),
            device,
            "final"
        ).await.map(|(var_store, final_node_params, train_final_nodes_values, min_loss)| {
            debug!("run_opt_train_final: done, min_loss={min_loss} final_node_params={final_node_params:?}");

            Arc::new(Box::new(TrainFinalNetsRunConfigResult::new(input.train_label_nodes_run_ids.as_ref(), min_loss, var_store, final_node_params, train_final_nodes_values.clone())))
        }).expect("TODO")
    }

    fn device(&self) -> impl ToOrdinal {
        ThoDevice::from(self.device)
    }
}

pub async fn run_train_final_nets_2(
    train_label_nodes_run_ids: Option<&Vec<RunId>>, //TODO use to filter label nets
    run_configs: &Arc<
        OptRunConfigs<
            Vec<NamedVal<TrainFinalNetRunConfigFields>>,
            Vec<NamedRange<TrainFinalNetRunConfigFields>>,
            Vec<Hyperparameter<TrainFinalNetRunConfigFields>>,
            TrainFinalNetRunConfigFields,
        >,
    >,
    fixed_config: &Arc<OptConfig>,
    resource_config: &NetResources<Logits>,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    runner: &Arc<ThoRunner>,
    devices: Vec<Device>,
) -> Result<Vec<Box<dyn SavableResult>>, ThoosaError> {
    debug!(
        "run_train_final_node_2: run_configs.len={run_configs:?} resource_config={resource_config:?}"
    );
    let (_train_resource_keys, _valid_resource_keys) = resource_config.to_resources(store)?;

    debug!("run_train_final_node_2: train_label_nodes_run_ids={train_label_nodes_run_ids:?}");

    let labels = store.read().list_labels()?;
    let label_nets_applicable = Arc::new(
        LabelNetsApplicable::load_all_filtered_by_run_ids(
            train_label_nodes_run_ids.map(|v| v.as_slice()),
            labels,
            store,
            ws,
            devices[0],
            &runner,
        )
        .await,
    );
    let train_label_net_applicable = Arc::new(TrainFinalNetApplicable::new(
        devices[0],
        label_nets_applicable,
    ));

    let (train_resource_keys, valid_resource_keys) = resource_config.to_resources(store)?;

    async move {
        Ok(runner
            .run(
                Arc::new(TrainFinalNetInput {
                    train_resources: Arc::new(train_resource_keys),
                    valid_resources: Arc::new(valid_resource_keys),
                    train_label_nodes_run_ids: train_label_nodes_run_ids.map(|v| v.to_vec()),
                    run_configs: Arc::clone(run_configs),
                    fixed_config: Arc::clone(&fixed_config),
                }),
                &train_label_net_applicable,
            )
            .await as Box<dyn SavableResult>)
    }
    .await
    .map(|v| vec![v])
}

// pub fn run_train_final_nets(
//     train_label_nodes_run_ids: Option<&Vec<RunId>>, //TODO use to filter label nets
//     run_configs: OptRunConfigs<Vec<NamedVal<TrainFinalNetRunConfigFields>>, Vec<NamedRange<TrainFinalNetRunConfigFields>>, Vec<Hyperparameter<TrainFinalNetRunConfigFields>>, TrainFinalNetRunConfigFields>,
//     fixed_config: &OptConfig,
//     resource_config: &NetResources<Logits>,
//     ws: &Arc<RwLock<Box<dyn Workspace>>>,
//     store: &Arc<RwLock<Box<dyn Store>>>,
//     loader: &Arc<RwLock<LoadRunner<Tensor>>>,
//     devices: Vec<Device>,
// ) -> Result<Option<Vec<Box<dyn SavableResult>>>, ThoosaError> {
//     debug!("run_train_final_node_2: run_configs.len={run_configs:?} resource_config={resource_config:?}");
//     let (train_resource_keys, valid_resource_keys) = resource_config.to_resources(store)?;
//
//     debug!("run_train_final_node_2: train_label_nodes_run_ids={train_label_nodes_run_ids:?}");
//
//     run_opt_train_final_net(
//         train_label_nodes_run_ids,
//         &run_configs,
//         train_resource_keys.clone(),
//         valid_resource_keys.clone(),
//         fixed_config,
//         ws,
//         store,
//         loader,
//         &devices[0],
//     ).map(|v| Some(vec![v]))
// }

pub async fn run_infer<F>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    infer_params: &InferArgs,
    device: Device,
    load_tensor: F,
    runner: &Arc<ThoRunner>,
) -> Result<InputProbs, ThoosaError>
where
    F: Fn(&PathBuf) -> Result<Tensor, ThoosaError> + Sync,
{
    let labels = store.read().list_labels()?;
    let train_label_node_run_ids = store
        .read()
        .get_input_run_ids(
            infer_params.final_run_id,
            Some(&TrainLabelNetsRunConfig::type_name()),
        )
        .expect(&format!(
            "FinalNetApplicable: final_run_id={} not found",
            infer_params.final_run_id
        ));

    let final_applicable = FinalNetApplicable::new(
        infer_params.final_run_id,
        labels.clone(),
        store,
        ws,
        device,
        LabelNetsApplicable::load_all_filtered_by_run_ids(
            Some(&train_label_node_run_ids),
            labels.clone(),
            store,
            ws,
            device,
            &runner,
        )
        .await,
    );

    let mut infer_result = InputProbs::new(labels);

    for entry in read_dir(&infer_params.input_dir)? {
        let path = entry?.path();

        let t = load_tensor(&path)?;

        debug!(
            "run_infer:={path:?} label=\n{t} t.view((1,-1))=\n{}",
            t.view((1, -1))
        );

        let logits = final_applicable.run(t.into(), ())?; //batch(1 for infer) x classes
        let probs = logits.t.softmax(-1, Float);

        debug!("run_infer:={path:?} logits=\n{logits} probs=\n{probs}");

        infer_result.add(
            path.file_stem()
                .expect(format!("no stam for {path:?}").as_str())
                .to_string_lossy()
                .to_string(),
            from::<Vec<f32>>(&probs.squeeze()),
        );

        info!(
            "run_infer: entry={path:?} result=\n{logits} argmax={:?}",
            logits.t.argmax(-1, false)
        );
    }

    Ok(infer_result)
}

#[derive(Debug)]
pub struct TrainLabelNetApplicable {
    device: Device,
}

impl TrainLabelNetApplicable {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

#[derive(Debug)]
pub struct TrainLabelNetInput {
    run_configs: Arc<
        OptRunConfigs<
            Vec<NamedVal<TrainLabelNetsRunConfigFields>>,
            Vec<NamedRange<TrainLabelNetsRunConfigFields>>,
            Vec<Hyperparameter<TrainLabelNetsRunConfigFields>>,
            TrainLabelNetsRunConfigFields,
        >,
    >,
    train_labels_resources: Arc<Vec<(i64, Vec<Resource>)>>, //TODO i64 to 1/0 enum for binary class
    valid_labels_resources: Arc<Vec<(i64, Vec<Resource>)>>,
    motifs_nets_selection_applicable_arc: Arc<MotifsNetsSelectionApplicable>,
    label: String,
    fixed_config: Arc<OptConfig>,        //TODO move to applicable?
    ws: Arc<RwLock<Box<dyn Workspace>>>, //TODO remove, replace with sizes copy
}

#[async_trait]
impl Runnable<TrainLabelNetInput, Box<TrainLabelNetsRunConfigResult>, ()>
    for TrainLabelNetApplicable
{
    type R = ThoRunner;

    async fn async_run(
        self: Arc<Self>,
        input: Arc<TrainLabelNetInput>,
        runner: Arc<Self::R>,
        _ctx: Option<
            Arc<
                RwLock<
                    <<Self as Runnable<
                        TrainLabelNetInput,
                        Box<TrainLabelNetsRunConfigResult>,
                        (),
                    >>::R as Runner>::Ctx,
                >,
            >,
        >,
    ) -> Arc<Box<TrainLabelNetsRunConfigResult>> {
        let input_for_train = Arc::clone(&input);
        let device_for_train = self.device.clone();
        //TODO use run_split_on_oom_2
        opt_train_2(
            Arc::clone(&input.fixed_config),
            async move |vs: &VarStore, run_config: TrainLabelNetsRunConfig, _listener| { //TODO add _listener to train_label_nodes_2
                debug!("run_opt_train_label_nodes:MaxPadding run_config={run_config:?}  ");

                let applicable_and_net_enum = LabelNetApplicableAndNetE::new(
                    &run_config.net, &input_for_train.motifs_nets_selection_applicable_arc, &input_for_train.ws
                );

                let mut batch = AsyncLoadingTensorsDataBatch::<ChannelsData, LabelNetInputTensors, _>::new(
                    Arc::clone(&input_for_train.train_labels_resources),
                    Arc::clone(&input_for_train.valid_labels_resources),
                    run_config.common_training_args.batch_size as usize,
                    Arc::clone(&runner),
                    device_for_train,
                    Some(Box::new(|t: TCD| {
                        debug!("Runnable for TrainLabelNetApplicable t={t:?}");
                        applicable_and_net_enum.apply(&t).expect("applicable_and_net_enum.apply failed")
                    }))
                );

                let result = train_label_net_2(&applicable_and_net_enum.module(vs).unwrap(), &run_config, &mut batch, vs, &device_for_train).await
                    .map(|v| (v,
                          match &applicable_and_net_enum {
                    LabelNetApplicableAndNetE::MaxPadding { applicable: _, params } => {
                                  debug!("run_opt_train_label_nodes:MaxPadding bet params={params:?}  ");

                                  params.clone()
                              }
                    LabelNetApplicableAndNetE::FixedNumSlices { applicable: _, params } => {
                                  debug!("run_opt_train_label_nodes:FixedNumSlices bet params={params:?}  ");

                                  params.clone()
                              }
                          }
                    ));
                Ok(result.expect("no result from train_label_nodes"))
            },
            Arc::clone(&input.run_configs),
            self.device.clone(),
            "label",
        ).await.map(|(var_store, label_node_params, run_config, min_loss)| {
            Arc::new(Box::new(TrainLabelNetsRunConfigResult::new(input.motifs_nets_selection_applicable_arc.motif_selection_run_id, input.label.clone(), min_loss, label_node_params, run_config, var_store)))
        }).expect("TODO")
    }
    fn device(&self) -> impl ToOrdinal {
        ThoDevice::from(self.device)
    }
}

pub async fn run_train_label_nets_2(
    selected_motifs_for_label_run_ids: Option<&Vec<RunId>>,
    run_configs: &Arc<
        OptRunConfigs<
            Vec<NamedVal<TrainLabelNetsRunConfigFields>>,
            Vec<NamedRange<TrainLabelNetsRunConfigFields>>,
            Vec<Hyperparameter<TrainLabelNetsRunConfigFields>>,
            TrainLabelNetsRunConfigFields,
        >,
    >,
    fixed_config: &Arc<OptConfig>,
    resource_config: &NetResources<Labeled>,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    runner: &Arc<ThoRunner>,
    devices: Vec<Device>,
) -> Result<Vec<Box<dyn SavableResult>>, ThoosaError> {
    debug!(
        "run_train_label_nodes: run_configs={run_configs:?} resource_config={resource_config:?} cuda_mem={:?}",
        cuda_mem()
    );

    let (train_labels_resource_keys, valid_label_resource_keys) =
        resource_config.to_resources(store)?;
    let ctx_for_motif_nets = runner.next_ctx();
    let runner_for_motif_nets_purge = Arc::clone(&runner);
    // let ctx_for_motif_nets_purge = Arc::clone(&ctx_for_motif_nets);

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

    let valid_datasets_map =
        valid_label_resource_keys
            .into_iter()
            .fold(HashMap::new(), |mut a, v| {
                assert!(v.1.len() > 0, "no resources for {}", v.0);
                a.insert(v.0.clone(), v.1.clone());
                a
            });

    let selected_motifs_for_label_run_ids: &Vec<RunId> =
        selected_motifs_for_label_run_ids.expect("select_motifs_for_label_run_ids");

    debug!(
        "run_train_label_nodes: select_motifs_for_label_run_ids={selected_motifs_for_label_run_ids:?} run_configs={run_configs:?} devices={devices:?}"
    );

    let selected_motifs_for_label_applicable = Arc::new(
        join_all(selected_motifs_for_label_run_ids.iter().map(
            async |select_motifs_for_label_run_id| {
                let channels_motifs = load_channels_motifs(
                    *select_motifs_for_label_run_id,
                    ws,
                    store,
                    devices[0],
                    &runner,
                    Some(Arc::clone(&ctx_for_motif_nets)),
                )
                .await;

                Arc::new(MotifsNetsSelectionApplicable::new(
                    *select_motifs_for_label_run_id,
                    store,
                    ws,
                    devices[0],
                    channels_motifs,
                ))
            },
        ))
        .await,
    );

    let chunk_len = max(1, train_labels_resource_keys.len() / devices.len());

    debug!("run_train_label_nodes:  chunk_len={chunk_len}");
    let device_for_purge = devices[0].clone();

    let devices_resources = devices
        .into_iter()
        .cycle()
        .zip(train_labels_resource_keys.chunks(chunk_len).into_iter())
        .collect::<Vec<(Device, &[(String, Vec<Resource>)])>>();

    let savables_res = try_join_all(devices_resources
        .into_iter()
        .flat_map(move |(device_i, train_resource_keys_chunk)| {

            let valid_datasets_map_clone = valid_datasets_map.clone(); //TODO
            let selected_motifs_for_label_applicable = selected_motifs_for_label_applicable.clone(); //TODO
            let train_label_net_applicable = Arc::new(TrainLabelNetApplicable::new(device_i));

            train_resource_keys_chunk.iter().flat_map( move |(label, resource_keys)| {
                debug!("run_train_label_nodes: label={label} resource_keys={resource_keys:?} cuda_mem={:?}", cuda_mem());

                let train_resource_keys_len = resource_keys.len();
                let valid_resource_keys = valid_datasets_map_clone.get(label).expect(format!("valid dataset for {}", label).as_str()).clone();
                let valid_resource_keys_len = valid_resource_keys.len();
                let train_label_net_applicable = Arc::clone(&train_label_net_applicable);

                let train_labels_resources = Arc::new(vec![
                    (1, resource_keys.clone()),
                    (0, store.read().list_resource_random_not_label(label.as_str(), train_resource_keys_len as i64).expect("list_resource_random_not_label for train"))
                ]);
                let valid_labels_resources = Arc::new(vec![
                    (1, valid_resource_keys),
                    (0, store.read().list_resource_random_not_label(label.as_str(), valid_resource_keys_len as i64).expect("list_resource_random_not_label for valid"))
                ]);

                debug!("run_train_label_nodes: label={label} \ntrain_labels_resource_rc[0]={:?} \ntrain_labels_resource_rc[1]={:?} cuda_mem={:?}", train_labels_resources[0], valid_labels_resources[1], cuda_mem());


                //TODO avoid vec cloning
                <Vec<Arc<MotifsNetsSelectionApplicable>> as Clone>::clone(&selected_motifs_for_label_applicable).into_iter().map( move |motifs_nets_selection_applicable| {
                // selected_motifs_for_label_applicable.iter().map( move |motifs_nets_selection_applicable| {

                    let train_labels_resources =  Arc::clone(&train_labels_resources);
                    let valid_labels_resources = Arc::clone(&valid_labels_resources);
                    let motifs_nets_selection_applicable_arc = Arc::clone(&motifs_nets_selection_applicable);
                    
                    let train_label_net_applicable = Arc::clone(&train_label_net_applicable);
                    async move {
                        Ok(runner.run(
                            Arc::new(TrainLabelNetInput {
                                run_configs: Arc::clone(run_configs),
                                train_labels_resources,
                                valid_labels_resources,
                                motifs_nets_selection_applicable_arc,
                                label: label.clone(),
                                fixed_config: Arc::clone(fixed_config),
                                ws: Arc::clone(ws),
                            }),
                            &train_label_net_applicable,
                        ).await as Box<dyn SavableResult>)
                    }
                })
            })
        })).await;

    let arc_count = Arc::strong_count(&ctx_for_motif_nets);
    runner_for_motif_nets_purge
        .purge_all_gpu_caches(
            Arc::into_inner(ctx_for_motif_nets)
                .expect(&format!(
                    "Arc::into_inner(ctx_for_purge) in train label nets: arc_count={arc_count}"
                ))
                .into_inner(),
            device_for_purge,
        )
        .await;

    savables_res
}

pub fn run_train_label_nets(
    selected_motifs_for_label_run_ids: Option<&Vec<RunId>>,
    run_configs: OptRunConfigs<
        Vec<NamedVal<TrainLabelNetsRunConfigFields>>,
        Vec<NamedRange<TrainLabelNetsRunConfigFields>>,
        Vec<Hyperparameter<TrainLabelNetsRunConfigFields>>,
        TrainLabelNetsRunConfigFields,
    >,
    fixed_config: &OptConfig,
    resource_config: &NetResources<Labeled>,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    loader: &Arc<RwLock<LoadRunner<Tensor>>>,
    devices: Vec<Device>,
) -> Result<Option<Vec<Box<dyn SavableResult>>>, ThoosaError> {
    debug!(
        "run_train_label_nodes: run_configs={run_configs:?} resource_config={resource_config:?} cuda_mem={:?}",
        cuda_mem()
    );

    let (train_labels_resource_keys, valid_label_resource_keys) =
        resource_config.to_resources(store)?;

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

    let valid_datasets_map =
        valid_label_resource_keys
            .into_iter()
            .fold(HashMap::new(), |mut a, v| {
                assert!(v.1.len() > 0, "no resources for {}", v.0);
                a.insert(v.0.clone(), v.1.clone());
                a
            });

    let selected_motifs_for_label_run_ids: &Vec<RunId> =
        selected_motifs_for_label_run_ids.expect("select_motifs_for_label_run_ids");

    debug!(
        "run_train_label_nodes: select_motifs_for_label_run_ids={selected_motifs_for_label_run_ids:?} run_configs={run_configs:?} devices={devices:?}"
    );

    let channels = ws.read().channels();

    let run_configs_arc = Arc::new(run_configs); //.into_iter().map(|c|Arc::new(c)).collect::<Vec<_>>());
    let selected_motifs_for_label_applicable = selected_motifs_for_label_run_ids
        .iter()
        .map(|select_motifs_for_label_run_id| {
            Arc::new(
                MotifsNetsSelectionApplicable::new(
                    *select_motifs_for_label_run_id,
                    store,
                    ws,
                    devices[0],
                    vec![],
                ), //TODO if not removed
            )
        })
        .collect::<Vec<Arc<MotifsNetsSelectionApplicable>>>();

    let chunk_len = max(1, train_labels_resource_keys.len() / devices.len());

    debug!("run_train_label_nodes:  chunk_len={chunk_len}");

    let devices_resources = devices
        .into_iter()
        .cycle()
        .zip(train_labels_resource_keys.chunks(chunk_len).into_iter())
        .collect::<Vec<(Device, &[(String, Vec<Resource>)])>>();

    devices_resources
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
                    (0, store.read().list_resource_random_not_label(label.as_str(), train_resource_keys_len as i64).expect("list_resource_random_not_label for train"))
                ]);
                let valid_labels_resources_rc = Rc::new(vec![
                    (1, valid_resource_keys),
                    (0, store.read().list_resource_random_not_label(label.as_str(), valid_resource_keys_len as i64).expect("list_resource_random_not_label for valid"))
                ]);

                debug!("run_train_label_nodes: label={label} \ntrain_labels_resource_rc[0]={:?} \ntrain_labels_resource_rc[1]={:?} cuda_mem={:?}", train_labels_resource_rc[0], train_labels_resource_rc[1], cuda_mem());


                let run_configs_arc = Arc::clone(&run_configs_arc);

                selected_motifs_for_label_applicable.iter().map(move |motifs_nets_selection_applicable| {

                    if log::log_enabled!(Level::Debug){
                        debug!("run_train_label_nodes: label={label} channels={channels} motifs_nets_selection_applicable.len={:?}", motifs_nets_selection_applicable.max_nets());
                    }
                    trace!("run_train_label_nodes: label={label} channels_motifs_models={motifs_nets_selection_applicable:?}");
                    debug!("run_train_label_nodes: label={label} cuda_mem={:?}", cuda_mem());

                    //TODO Rc
                    let train_labels_resources_clone = Rc::clone(&train_labels_resource_rc);
                    let valid_labels_resources_clone = Rc::clone(&valid_labels_resources_rc);

                    run_opt_train_label_nets(
                        &run_configs_arc,
                        Rc::clone(&train_labels_resources_clone),
                        Rc::clone(&valid_labels_resources_clone),
                        Arc::clone(&motifs_nets_selection_applicable),
                        label,
                        fixed_config, //TODO slices_len,
                        ws,
                        &loader,
                        &device_i,
                    )
                })
            }).collect::<Result<Vec<_>, ThoosaError>>()
        })
        .collect::<Result<Vec<Vec<Box<dyn SavableResult>>>, ThoosaError>>().map(|v| v.into_iter()
        .flat_map(|v|v.into_iter())
        .collect::<Vec<Box<dyn SavableResult>>>()).map(|v|Some(v))
}

/**
returns motif_node_params_selection_run_id
*/
#[allow(dead_code)]
async fn run_train_motif_nodes_on_motif_selection_2(
    motif_selection_run_ids: Option<&Vec<RunId>>,
    train_motifs_nodes_run_configs: &Arc<
        OptRunConfigs<
            Vec<NamedVal<TrainMotifsNetsRunConfigFields>>,
            Vec<NamedRange<TrainMotifsNetsRunConfigFields>>,
            Vec<Hyperparameter<TrainMotifsNetsRunConfigFields>>,
            TrainMotifsNetsRunConfigFields,
        >,
    >,
    fixed_config: &Arc<OptConfig>,
    _resource_config: &Empty,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    runner: &Arc<ThoRunner>,
    devices: Vec<Device>,
) -> Result<Vec<Box<dyn SavableResult>>, ThoosaError> {
    let motif_selection_run_ids: &Vec<RunId> =
        motif_selection_run_ids.expect("motif_selection_run_ids");

    let chunk_len = max(1, motif_selection_run_ids.len() / devices.len());

    debug!(
        "run_train_motif_nodes_on_motif_selection: motif_selection_run_ids={motif_selection_run_ids:?} train_motifs_nodes_run_config={train_motifs_nodes_run_configs:?} devices={devices:?} chunk_len={chunk_len}"
    );
    let train_motifs_nodes_run_configs_arc = Arc::new(train_motifs_nodes_run_configs);
    let device_for_purge = devices[0].clone();

    let devices_motif_selection = devices
        .into_iter()
        .cycle()
        .zip(motif_selection_run_ids.chunks(chunk_len).into_iter())
        .collect::<Vec<(Device, &[RunId])>>();

    let ctx = runner.next_ctx();
    let runner_for_purge = Arc::clone(&runner);
    let ctx_for_purge = Arc::clone(&ctx);

    let savables_res = try_join_all(devices_motif_selection
        .into_iter()
        .flat_map(move |(device, motif_selection_run_ids_chunk)| {
            debug!("run_train_motif_nodes_on_motif_selection: motif_selection_run_ids_chunk={motif_selection_run_ids_chunk:?}");
            let train_motifs_nodes_run_configs_arc = Arc::clone(&train_motifs_nodes_run_configs_arc);
            let ctx = Arc::clone(&ctx);

            motif_selection_run_ids_chunk.iter().flat_map(move |motif_selection_run_id| {
                let find_motifs_run_ids = store.read().get_input_run_ids(*motif_selection_run_id, None).expect("motif_selection_id failed");
                debug!("run_train_motif_nodes_on_motif_selection: find_motifs_run_ids={find_motifs_run_ids:?}");


                let selected_motifs = store.read().list_selected_motifs(*motif_selection_run_id, ws.read().channels()).expect("list_selected_motifs");

                selected_motifs.into_iter()
                    .flat_map(|channel_motifs| channel_motifs.into_iter())
                    .flat_map({
                        let train_motifs_nodes_run_configs_arc = Arc::clone(&train_motifs_nodes_run_configs_arc);
                        let ctx = Arc::clone(&ctx);

                        move |(motif_id, _corr)| {
                            let train_motifs_nodes_run_configs_arc = Arc::clone(&train_motifs_nodes_run_configs_arc);
                            let ctx = Arc::clone(&ctx);

                            <Vec<u32> as Clone>::clone(&find_motifs_run_ids).into_iter().map( move |find_motifs_run_id| {
                                let find_motifs_run_config_applicable = FindMotifsRunConfig::from_properties_map__(find_motifs_run_id, &store.read().list_run_config_values(find_motifs_run_id).expect("list_run_params")).unwrap()
                                    .to_applicable(ws.read().deref(), device);


                                debug!("run_train_motif_nodes_on_motif_selection: find_motifs_run_id={find_motifs_run_id:?} motif_selection_id={motif_selection_run_id}");

                                run_train_motif_nets_2(
                                    motif_id,
                                    find_motifs_run_id,
                                    Arc::clone(&train_motifs_nodes_run_configs_arc),
                                    find_motifs_run_config_applicable,
                                    fixed_config,
                                    ws,
                                    Arc::clone(store),
                                    runner,
                                    device,
                                    Arc::clone(&ctx)
                                )
                            })

                        }})
            })

        })).await;

    let arc_count = Arc::strong_count(&ctx_for_purge);
    runner_for_purge
        .purge_all_gpu_caches(
            Arc::into_inner(ctx_for_purge)
                .expect(&format!(
                    "Arc::into_inner(ctx_for_purge) in train motif nets: arc_count={arc_count}"
                ))
                .into_inner(),
            device_for_purge,
        )
        .await;

    savables_res
}

/**
returns motif_node_params_selection_run_id
*/
fn run_train_motif_nodes_on_motif_selection(
    motif_selection_run_ids: Option<&Vec<RunId>>,
    train_motifs_nodes_run_configs: OptRunConfigs<
        Vec<NamedVal<TrainMotifsNetsRunConfigFields>>,
        Vec<NamedRange<TrainMotifsNetsRunConfigFields>>,
        Vec<Hyperparameter<TrainMotifsNetsRunConfigFields>>,
        TrainMotifsNetsRunConfigFields,
    >,
    fixed_config: &OptConfig,
    _resource_config: &Empty,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    loader: &Arc<RwLock<LoadRunner<Tensor>>>,
    devices: Vec<Device>,
) -> Result<Option<Vec<Box<dyn SavableResult>>>, ThoosaError> {
    let motif_selection_run_ids: &Vec<RunId> =
        motif_selection_run_ids.expect("motif_selection_run_ids");

    let chunk_len = max(1, motif_selection_run_ids.len() / devices.len());

    debug!(
        "run_train_motif_nodes_on_motif_selection: motif_selection_run_ids={motif_selection_run_ids:?} train_motifs_nodes_run_config={train_motifs_nodes_run_configs:?} devices={devices:?} chunk_len={chunk_len}"
    );

    let devices_motif_selection = devices
        .into_iter()
        .cycle()
        .zip(motif_selection_run_ids.chunks(chunk_len).into_iter())
        .collect::<Vec<(Device, &[RunId])>>();

    devices_motif_selection
        .into_par_iter()
        .map( |(device, motif_selection_run_ids_chunk)| {
            debug!("run_train_motif_nodes_on_motif_selection: motif_selection_run_ids_chunk={motif_selection_run_ids_chunk:?}");

            motif_selection_run_ids_chunk.iter().map(|motif_selection_run_id| {

                let find_motifs_run_ids = store.read().get_input_run_ids(*motif_selection_run_id, None).expect("motif_selection_id failed");
                debug!("run_train_motif_nodes_on_motif_selection: find_motifs_run_ids={find_motifs_run_ids:?}");

                let motifs = store.read().list_selected_motifs(*motif_selection_run_id, ws.read().channels()).expect("list_selected_motifs")
                    .into_iter()
                    .flat_map(|channel_motifs| channel_motifs.into_iter())
                    .map(|(motif_id, _corr)| motif_id).collect::<Vec<u32>>();

                debug!("run_train_motif_nodes_on_motif_selection:  motifs={motifs:?}");

                find_motifs_run_ids.into_iter().map(|find_motifs_run_id| {
                    // assert_eq!(find_motifs_run_ids.len(), 1, "expected single find_motifs_run_id:  input motif_selection_run_id={motif_selection_run_id} find_motifs_run_ids={find_motifs_run_ids:?}"); //selection has single input
                    // let find_motifs_run_id = find_motifs_run_ids[0];

                    let find_motifs_run_config_applicable = FindMotifsRunConfig::from_properties_map__(find_motifs_run_id, &store.read().list_run_config_values(find_motifs_run_id).expect("list_run_params"))?
                        .to_applicable(ws.read().deref(), device);


                    debug!("run_train_motif_nodes_on_motif_selection: find_motifs_run_id={find_motifs_run_id:?} motif_selection_id={motif_selection_run_id}");

                    run_train_motif_nets(
                        &motifs,
                        find_motifs_run_id,
                        &train_motifs_nodes_run_configs,
                        &find_motifs_run_config_applicable,
                        fixed_config,
                        ws,
                        Arc::clone(store),
                        loader,
                        device
                    )
                }).collect::<Result<Vec<Vec<Box<dyn SavableResult>>>, ThoosaError>>()//TODO remove extra collect

            }).collect::<Result<Vec<Vec<Vec<Box<dyn SavableResult>>>>, ThoosaError>>()

        })
        .collect::<Result<Vec<Vec<Vec<Vec<Box<dyn SavableResult>>>>>, ThoosaError>>()
        .map(|v| v.into_iter()
            .flat_map(|v|v.into_iter()
                .flat_map(|v|v.into_iter()
                    .flat_map(|v|v.into_iter())))
            .collect::<Vec<Box<dyn SavableResult>>>()).map(|v|Some(v)
    )
}

fn motifs_to_tensor_cmd(
    cli: &Cli,
    store: &Arc<RwLock<Box<dyn Store>>>,
    device: Device,
    choose_fragments_params: &MotifsToTensorArgs,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    loader: &Arc<RwLock<LoadRunner<Tensor>>>,
) -> Result<(), ThoosaError> {
    let find_motif_run_id = choose_fragments_params.run_id();

    let find_motifs_run_config = FindMotifsRunConfig::from_properties_map__(
        find_motif_run_id,
        &store.read().list_run_config_values(find_motif_run_id)?,
    )?;

    let t = motifs_to_tensor(
        store.read().list_motifs_slices(find_motif_run_id)?,
        loader,
        &find_motifs_run_config,
        20,
        device,
        0,
        ws,
    )?;

    let out = Path::new(&cli.ws_path).join("out");
    fs::create_dir_all(&out)?;
    t.channels_motifs_slices
        .save(out.join(format!("motifs_in_{}", choose_fragments_params.run_id())))?;

    Ok(())
}

pub fn import_resources<F, R>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    labels: &[&str],
    labeled_resources: F,
    resource_to_tensor: R, //None for tests - do not import parquet files
    insert_label: bool,
) -> Result<Vec<ResourceId>, ThoosaError>
where
    F: Fn() -> Result<HashMap<String, Vec<(String, f32)>>, ThoosaError>,
    R: Fn(String) -> Result<Tensor, ThoosaError> + Sync + Send,
{
    let store_guard = store.write();

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
                        //resource keys should not repeat
                        assert!(a.0.insert(resource_key.clone(), (stats, probs)).is_none());
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

    info!(
        "Import id: {}, imported {} tensors to {:?}, failed {}",
        import_id,
        resource_ids.len(),
        ws_guard.tensors_path(),
        failed_tensors.len()
    );

    store_guard.save_pure_sets(3, true, &t)?;
    store_guard.save_pure_sets(10, true, &t)?;
    store_guard.save_pure_sets(50, true, &t)?;
    store_guard.save_pure_sets(100, true, &t)?;
    store_guard.save_pure_sets(150, true, &t)?;
    store_guard.save_pure_sets(200, true, &t)?;

    store_guard.save_groups_by_label_cnt(100, &t)?;
    t.commit()?;

    trace!("import_resources: resource_ids={resource_ids:?}, failed_tensors={failed_tensors:?}");

    Ok(resource_ids) //for tests
}

async fn run_plot_motif_slices(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    loader: &Arc<RawTensorLoader>,
    plot_motif_slices_params: &PlotMotifSlicesArgs,
    device: &Device,
) -> Result<(), ThoosaError> {
    let ctx = Arc::new(parking_lot::RwLock::new(RunCtx2::new(1, 0, "aa")));

    match plot_motif_slices_params.motifs_by() {
        MotifsBy::MotifId(motif_id) => {
            debug!("run_plot_motif_slices: MotifsBy::MotifId run_id={motif_id}");


            let motif_loader = Arc::new(MotifLoader::new(
                Arc::clone(store),
                Arc::clone(loader)
            ));
            let run_id = 0;

            let motif = motif_loader.run_cached(Arc::new(DataId::from(motif_id as u32)), ctx).await?;

            let r = motif.with_data_ref(0, None, |t|{
                let dist = 0.0; //average_euclidean_distance(t.t_ref()).unwrap();

                println!("motif.t={} dist={dist}", t.t_ref());
                let all_slices= t.t_ref().unbind(0);

                let mut color_iter = DistinctColorIter::new();
                plot_series(
                    all_slices.into_iter().enumerate().map(|(i, t)| {
                        (format!("{i}"), color_iter.next().unwrap_or(BLACK), t.iter::<f64>().unwrap().enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>())
                    }).collect(),
                    "zzz",// format!("Motif: {motif_id} run id: {run_id} channel: {channel} number of slices: {slices_len} motif_avg_dist={motif_avg_dist}").as_str(),
                    (2000, 1000),
                    &ws.read().plots_path().join(run_id.to_string()).join(format!("motif_{}.png", motif_id))
                )?;


                Ok(true)
            }).await?;


            // find::plot_motif_slices(
            //     ws,
            //     store,
            //     loader,
            //     motif_id as u32,
            //     plot_motif_slices_params.limit,
            //     device,
            // )?;
        }
        MotifsBy::RunId(run_id) => {
            debug!("run_plot_motif_slices: MotifsBy::RunId run_id={run_id}");
            fs::create_dir_all(ws.read().plots_path().join(run_id.to_string()))?;

            //TODO add list for  motif_ids only without grouping by  slice_matches
            let motif_distances = store.read().list_motifs(run_id)?;
            let motifs_len = motif_distances.len();
            let done_cnt = AtomicU32::new(0);
            motif_distances /*.into_iter()*/
                .into_par_iter()
                .for_each(|motif_id| {
                    //TODO par in devices
                    debug!("run_plot_motif_slices: motif_id={motif_id}");
                    // find::plot_motif_slices(
                    //     ws,
                    //     store,
                    //     loader,
                    //     motif_id as u32,
                    //     plot_motif_slices_params.limit,
                    //     device,
                    // )
                    // .unwrap();
                    let done = done_cnt.fetch_add(1, Relaxed);
                    if done % 10 == 0 {
                        info!("done {done} from  {motifs_len}");
                    }
                });
        }
    }

    Ok(())
}

#[allow(dead_code)]
async fn run_train_motif_nets_2(
    motif_id: u32,
    find_motifs_run_id: RunId,
    train_motifs_nodes_run_configs: Arc<
        OptRunConfigs<
            Vec<NamedVal<TrainMotifsNetsRunConfigFields>>,
            Vec<NamedRange<TrainMotifsNetsRunConfigFields>>,
            Vec<Hyperparameter<TrainMotifsNetsRunConfigFields>>,
            TrainMotifsNetsRunConfigFields,
        >,
    >,
    find_motifs_run_config_applicable: FindMotifsApplicable,
    fixed_config: &Arc<OptConfig>,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<RwLock<Box<dyn Store>>>,
    runner: &Arc<ThoRunner>,
    device: Device,
    ctx: Arc<RwLock<ThoRunCtx>>,
) -> Result<Box<dyn SavableResult>, ThoosaError> {
    debug!(
        "train_on_motifs: device={device:?}, find_motifs_run_id={find_motifs_run_id} motifs={motif_id:?}"
    );

    let find_motifs_run_config_applicable = Arc::new(find_motifs_run_config_applicable);

    let find_motifs_run_config_applicable = Arc::clone(&find_motifs_run_config_applicable);

    let (all_slices_0, all_slices_1, channel) = {
        debug!(
            "train_on_motif: 1. before store lock on thread={:?}",
            std::thread::current().name()
        );

        let (channel, run_id, slices, resources_map) =
            store.read().list_motif_slices(motif_id, None, None)?;

        assert_eq!(run_id, find_motifs_run_id);

        debug!(
            "train_on_motif: 1. channel={channel}, run_id={run_id}, slices.len={}",
            slices.len()
        );
        trace!("train_on_motif: 1. slices={slices:?}, resources_map={resources_map:?}");

        let all_slices_1 = train_nets::slices_in_all_resources(
            load_motif_slices_2(
                channel,
                &find_motifs_run_config_applicable,
                slices,
                resources_map,
                runner,
                &device,
                Arc::clone(&ctx),
            )
            .await,
        );

        let (channel, run_id, slices, resources_map) = store.read().list_motif_slices(
            motif_id,
            Some(all_slices_1.len() as u32),
            Some((find_motifs_run_id, channel as u32)),
        )?;

        debug!(
            "train_on_motif: 0. channel={channel}, run_id={run_id}, slices.len={}",
            slices.len()
        );
        trace!("train_on_motif: 0. slices={slices:?}, resources_map={resources_map:?}");

        let all_slices_0 = train_nets::slices_in_all_resources(
            load_motif_slices_2(
                channel,
                &find_motifs_run_config_applicable,
                slices,
                resources_map,
                runner,
                &device,
                Arc::clone(&ctx),
            )
            .await,
        );

        debug!("train_on_motif: 0. run_id={}, done", run_id);

        (all_slices_0, all_slices_1, channel)
    };

    assert!(
        all_slices_1.len() > 0,
        "all_slices_1.len(), motif={motif_id}"
    );
    assert!(
        all_slices_0.len() > 0,
        "all_slices_0.len(), motif={motif_id}"
    );

    debug!(
        "train_on_motif:  all_slices_0.len()={:?}, all_slices_0[0].size()={:?}, all_slices_1.len()={:?}, all_slices_1[0].size()={:?}",
        all_slices_0.len(),
        all_slices_0[0].size(),
        all_slices_1.len(),
        all_slices_1[0].size()
    );

    Ok(runner
        .run(
            Arc::new(TrainMotifNetApplicableInput {
                motif_id,
                fixed_config: Arc::clone(&fixed_config),
                all_slices_0: Arc::new(all_slices_0.into_iter().map(|s| s.into()).collect()),
                all_slices_1: Arc::new(all_slices_1.into_iter().map(|s| s.into()).collect()),
                find_motifs_run_config: find_motifs_run_config_applicable.run_config().clone(),
                train_motifs_nodes_run_configs,
                channel: channel as u32,
            }),
            &Arc::new(TrainMotifNetApplicable { device }),
        )
        .await as Box<dyn SavableResult>) //TODO unwrap
}

fn run_train_motif_nets(
    motifs: &Vec<u32>,
    find_motifs_run_id: RunId,
    train_motifs_nodes_run_configs: &OptRunConfigs<
        Vec<NamedVal<TrainMotifsNetsRunConfigFields>>,
        Vec<NamedRange<TrainMotifsNetsRunConfigFields>>,
        Vec<Hyperparameter<TrainMotifsNetsRunConfigFields>>,
        TrainMotifsNetsRunConfigFields,
    >,
    find_motifs_run_config_applicable: &FindMotifsApplicable,
    fixed_config: &OptConfig,
    _ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<RwLock<Box<dyn Store>>>,
    loader: &Arc<RwLock<LoadRunner<Tensor>>>,
    device: Device,
) -> Result<Vec<Box<dyn SavableResult>>, ThoosaError> {
    debug!(
        "train_on_motifs: device={device:?}, find_motifs_run_id={find_motifs_run_id} motifs={motifs:?}"
    );

    motifs.into_iter().map(|motif_id| {
        let (all_slices_0, all_slices_1, channel) = {

            let (channel, run_id, slices, resources_map) = store.read().list_motif_slices(*motif_id, None, None)?;

            assert_eq!(run_id, find_motifs_run_id);

            debug!("train_on_motif: 1. channel={channel}, run_id={run_id}, slices.len={}", slices.len());
            trace!("train_on_motif: 1. slices={slices:?}, resources_map={resources_map:?}");

            let all_slices_1 = train_nets::slices_in_all_resources(
                load_motif_slices(channel, find_motifs_run_config_applicable, slices, resources_map, loader)
            );

            //list motifs which are NOT motif_id and NOT from run_id
            let (channel, run_id, slices, resources_map) = store.read().list_motif_slices(*motif_id, Some(all_slices_1.len() as u32), Some((find_motifs_run_id, channel as u32)))?;

            debug!("train_on_motif: 0. channel={channel}, run_id={run_id}, slices.len={}", slices.len());
            trace!("train_on_motif: 0. slices={slices:?}, resources_map={resources_map:?}");

            let all_slices_0 = train_nets::slices_in_all_resources(
                load_motif_slices(channel, find_motifs_run_config_applicable, slices, resources_map, loader)
            );

            debug!("train_on_motif: 0. run_id={}, done", run_id);

            (all_slices_0, all_slices_1, channel)
        };

        assert!(all_slices_1.len() > 0, "all_slices_1.len(), motif={motif_id}");
        assert!(all_slices_0.len() > 0, "all_slices_0.len(), motif={motif_id}");


        debug!("train_on_motif:  all_slices_0.len()={:?}, all_slices_0[0].size()={:?}, all_slices_1.len()={:?}, all_slices_1[0].size()={:?}", all_slices_0.len(), all_slices_0[0].size(), all_slices_1.len(), all_slices_1[0].size());


        opt_train(
            fixed_config,
            move |vs:&VarStore, run_config:&TrainMotifsNetsRunConfig, listener: &mut dyn TrainListener|{
                //TODO batch size
                let all_slices_0_cl = all_slices_0.iter().map(|v|v.shallow_clone()).collect::<Vec<_>>();
                let all_slices_1_cl = all_slices_1.iter().map(|v|v.shallow_clone()).collect::<Vec<_>>();


                let mut batch = TensorsDataBatch::new(
                    vec![
                        all_slices_0_cl,
                        all_slices_1_cl
                    ],
                    run_config.common_training_args.batch_size,
                    0.8,//TODO config
                    device
                );
                run_train_motif_net(find_motifs_run_config_applicable.run_config().window(), run_config, &device, &mut batch, vs, listener)
            },
            train_motifs_nodes_run_configs,
            device,
            "motifs"
        ).map(|(var_store, motif_node_params, train_motifs_nets_run_config, min_loss)|{
            Box::new(TrainMotifsNetsRunConfigResult::new(min_loss, channel as u32, *motif_id, var_store, find_motifs_run_id/* TODO train_motifs_run_config.id.expect("id")*/, motif_node_params, train_motifs_nets_run_config)) as Box<dyn SavableResult>

        })

    }).filter(|v|v.is_ok()).collect()
}

#[derive(Debug)]
#[allow(dead_code)]
struct TrainMotifNetApplicable {
    device: Device,
}

#[derive(Debug)]
#[allow(dead_code)]
struct TrainMotifNetApplicableInput {
    motif_id: u32,
    fixed_config: Arc<OptConfig>,
    all_slices_0: Arc<Vec<SyncWrapper<Tensor>>>,
    all_slices_1: Arc<Vec<SyncWrapper<Tensor>>>,
    find_motifs_run_config: FindMotifsRunConfig,
    train_motifs_nodes_run_configs: Arc<
        OptRunConfigs<
            Vec<NamedVal<TrainMotifsNetsRunConfigFields>>,
            Vec<NamedRange<TrainMotifsNetsRunConfigFields>>,
            Vec<Hyperparameter<TrainMotifsNetsRunConfigFields>>,
            TrainMotifsNetsRunConfigFields,
        >,
    >,
    channel: u32,
}

#[async_trait]
impl Runnable<TrainMotifNetApplicableInput, Box<TrainMotifsNetsRunConfigResult>, ()>
    for TrainMotifNetApplicable
{
    type R = ThoRunner;

    async fn async_run(
        self: Arc<Self>,
        input: Arc<TrainMotifNetApplicableInput>,
        _runner: Arc<Self::R>,
        _ctx: Option<
            Arc<
                RwLock<
                    <<Self as Runnable<
                        TrainMotifNetApplicableInput,
                        Box<TrainMotifsNetsRunConfigResult>,
                        (),
                    >>::R as Runner>::Ctx,
                >,
            >,
        >,
    ) -> Arc<Box<TrainMotifsNetsRunConfigResult>> {
        let train_motifs_nodes_run_configs = Arc::clone(&input.train_motifs_nodes_run_configs);
        let all_slices_0_cl = Arc::clone(&input.all_slices_0);
        let all_slices_1_cl = Arc::clone(&input.all_slices_1);
        let window = input.find_motifs_run_config.window();
        let device = self.device;

        opt_train_2(
            Arc::clone(&input.fixed_config),
            async move |vs: &VarStore, run_config: TrainMotifsNetsRunConfig, _listener| {
                //TODO batch size
                let all_slices_0_cl = all_slices_0_cl.iter().map(|v| v.shallow_clone()).collect::<Vec<_>>();
                let all_slices_1_cl = all_slices_1_cl.iter().map(|v| v.shallow_clone()).collect::<Vec<_>>();

                let mut batch = TensorsDataBatch::new(
                    vec![
                        all_slices_0_cl,
                        all_slices_1_cl
                    ],
                    run_config.common_training_args.batch_size,
                    0.8, //TODO config
                    device
                );

                run_train_motif_net(window, &run_config, &self.device, &mut batch, vs, &mut DummyListener/*, listener*/)
            },
            train_motifs_nodes_run_configs,
            device,
            "motifs"
        ).await.map(|(var_store, motif_node_params, train_motifs_nets_run_config, min_loss)| {
            Arc::new(Box::new(TrainMotifsNetsRunConfigResult::new(min_loss, input.channel, input.motif_id, var_store, input.find_motifs_run_config.id.unwrap()/* TODO train_motifs_run_config.id.expect("id")*/, motif_node_params, train_motifs_nets_run_config)))
        }).expect("TODO opt_train failed")
    }
    fn device(&self) -> impl ToOrdinal {
        ThoDevice::from(self.device)
    }
}

#[allow(dead_code)]
fn resource_path(base_path: &str, resource_pattern: &str, resource_key: &str) -> String {
    //TODO change to impl AsRef<Path> + in ParquetLoader
    Path::new(base_path)
        .join(resource_pattern.replace("{}", resource_key))
        .into_os_string()
        .into_string()
        .unwrap()
}

pub fn run_opt_step_2<RC, RCRV, CC, R, ST /* F*/>(
    input_run_ids: Option<Vec<RunId>>,
    rcrv: &Arc<RCRV>,
    common_config: Arc<CC>,
    resource_config: &R,
    opt_step: ST,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    runner: &Arc<ThoRunner>,
    devices: Vec<Device>,
    _descr: &str,
) -> Result<Vec<RunId>, ThoosaError>
where
    ST: AsyncFnOnce(
        Option<&Vec<RunId>>,
        &Arc<
            OptRunConfigs<
                Vec<NamedVal<<RCRV as OptRunConfigSource>::P>>,
                Vec<NamedRange<<RCRV as OptRunConfigSource>::P>>,
                Vec<Hyperparameter<<RCRV as OptRunConfigSource>::P>>,
                <RCRV as OptRunConfigSource>::P,
            >,
        >,
        &Arc<CC>,
        &R,
        &Arc<RwLock<Box<dyn Workspace>>>,
        &Arc<RwLock<Box<dyn Store>>>,
        &Arc<ThoRunner>,
        Vec<Device>,
    ) -> Result<Vec<Box<dyn SavableResult>>, ThoosaError>, //TODO remove Option & Vec
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<<RCRV as OptRunConfigSource>::P>>> + ToHashmap,
    RCRV: OptRunConfigSource,
    R: ResourcesConfig,
{
    let opt_run_configs = block_on(opt_step.async_call_once((
        input_run_ids.as_ref(),
        &Arc::new(rcrv.opt_run_configs()),
        &common_config,
        resource_config,
        ws,
        store,
        runner,
        devices,
    )))?;

    //TODO choose best?
    let mut opt_run_config_ids = vec![];
    for mut opt_run_config in opt_run_configs {
        debug!("run_step: saving opt_run_config={opt_run_config:?}");

        let opt_run_config_id = opt_run_config.save(store, ws)?;

        opt_run_config_ids.push(opt_run_config_id);
    }
    Ok(opt_run_config_ids)
}

pub fn run_opt_step<RC, RCRV, CC, R, ST>(
    input_run_ids: Option<Vec<RunId>>,
    rcrv: RCRV,
    common_config: &CC,
    resource_config: &R,
    opt_step: ST,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    loader: &Arc<RwLock<LoadRunner<Tensor>>>,
    devices: Vec<Device>,
    _descr: &str,
) -> Result<Vec<RunId>, ThoosaError>
where
    ST: FnOnce(
        Option<&Vec<RunId>>,
        OptRunConfigs<
            Vec<NamedVal<<RCRV as OptRunConfigSource>::P>>,
            Vec<NamedRange<<RCRV as OptRunConfigSource>::P>>,
            Vec<Hyperparameter<<RCRV as OptRunConfigSource>::P>>,
            <RCRV as OptRunConfigSource>::P,
        >,
        &CC,
        &R,
        &Arc<RwLock<Box<dyn Workspace>>>,
        &Arc<RwLock<Box<dyn Store>>>,
        &Arc<RwLock<LoadRunner<Tensor>>>,
        Vec<Device>,
    ) -> Result<Option<Vec<Box<dyn SavableResult>>>, ThoosaError>, //TODO remove Option & Vec
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<<RCRV as OptRunConfigSource>::P>>> + ToHashmap,
    RCRV: OptRunConfigSource,
    R: ResourcesConfig,
{
    let opt_run_configs_opt = opt_step(
        input_run_ids.as_ref(),
        rcrv.opt_run_configs(),
        common_config,
        resource_config,
        ws,
        store,
        loader,
        devices,
    )?;

    if let Some(opt_run_configs) = opt_run_configs_opt {
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
        panic!("todo")
    }
}

pub fn run_all_grid_step_2<RC: Send, RCRV, R: Send + Sync, FC: Send + Sync, ST>(
    input_run_ids: Option<Vec<RunId>>,
    rcrv: RCRV,
    fixed_config: &FC,
    resource_config: &R,
    grid_step: ST,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    runner: &Arc<ThoRunner>,
    devices: Vec<Device>,
    _descr: &str,
    presave: bool,
) -> Result<Vec<RunId>, ThoosaError>
where
    ST: AsyncFnOnce(
            Option<&Vec<RunId>>,
            Vec<RC>,
            &R,
            &FC,
            &Arc<RwLock<Box<dyn Workspace>>>,
            &Arc<RwLock<Box<dyn Store>>>,
            &Arc<ThoRunner>,
            Vec<Device>,
        ) -> Result<Option<Vec<Box<dyn SavableResult>>>, ThoosaError>
        + Send,
    RC: RunConfig
        + From<std::vec::IntoIter<NamedVal<<RCRV as GridRunConfigSource>::P>>>
        + thoosa_properties_map::ToHashmap,
    RCRV: GridRunConfigSource,
    R: ResourcesConfig + Send,
{
    let grid_run_configs = rcrv.grid_run_configs();
    debug!("run_all_grid_step: grid_run_configs={grid_run_configs:?}");

    let mut run_configs: Vec<RC> = grid_run_configs.to_run_configs();
    if presave {
        save_run_configs(
            input_run_ids.as_ref().map(|v| v.as_slice()),
            &store,
            &mut run_configs,
            "find motifs",
        );
    }
    // let run_config_ids = save_run_configs(input_run_ids.as_ref().map(|v|v.as_slice()), &store, &mut run_configs, "find motifs");

    debug!("FindMotifs2: run_configs={run_configs:?}");

    // let mut opt_run_configs_opt = block_on(rayon::scope(move |scope|scope.spawn_future(
    let opt_run_configs_opt = block_on(grid_step.async_call_once((
        input_run_ids.as_ref(),
        run_configs,
        resource_config,
        fixed_config,
        ws,
        store,
        runner,
        devices,
    )))?;

    if let Some(opt_run_configs) = opt_run_configs_opt {
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
        panic!("todo")
    }
}

pub fn run_all_grid_step<RC, RCRV, R, FC, ST>(
    input_run_ids: Option<Vec<RunId>>,
    rcrv: RCRV,
    fixed_config: &FC,
    resource_config: &R,
    grid_step: ST,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    loader: &Arc<RwLock<LoadRunner<Tensor>>>,
    devices: Vec<Device>,
    _descr: &str,
    presave: bool,
) -> Result<Vec<RunId>, ThoosaError>
where
    ST: FnOnce(
        Option<&Vec<RunId>>,
        Vec<RC>,
        &R,
        &FC,
        &Arc<RwLock<Box<dyn Workspace>>>,
        &Arc<RwLock<Box<dyn Store>>>,
        &Arc<RwLock<LoadRunner<Tensor>>>,
        Vec<Device>,
    ) -> Result<Option<Vec<Box<dyn SavableResult>>>, ThoosaError>,
    RC: RunConfig
        + From<std::vec::IntoIter<NamedVal<<RCRV as GridRunConfigSource>::P>>>
        + thoosa_properties_map::ToHashmap,
    RCRV: GridRunConfigSource,
    R: ResourcesConfig,
{
    let grid_run_configs = rcrv.grid_run_configs();
    debug!("run_all_grid_step: grid_run_configs={grid_run_configs:?}");

    let mut run_configs: Vec<RC> = grid_run_configs.to_run_configs();
    if presave {
        save_run_configs(
            input_run_ids.as_ref().map(|v| v.as_slice()),
            &store,
            &mut run_configs,
            "find motifs",
        );
    }
    // let run_config_ids = save_run_configs(input_run_ids.as_ref().map(|v|v.as_slice()), &store, &mut run_configs, "find motifs");

    debug!("FindMotifs2: run_configs={run_configs:?}");

    let opt_run_configs_opt = grid_step(
        input_run_ids.as_ref(),
        run_configs,
        resource_config,
        fixed_config,
        ws,
        store,
        loader,
        devices,
    )?;

    if let Some(opt_run_configs) = opt_run_configs_opt {
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
        panic!("todo")
    }
}

pub async fn run_grid<ResConf, RCRV, F, RC, ST, CtxF>(
    input_run_ids: Option<Vec<RunId>>,
    rcrv: RCRV,
    // fixed_config: FC,
    input: Arc<ST::In<'static>>,
    // ctx:Arc<parking_lot::RwLock<ST::Ctx>>,
    // stage: Arc<ST>,
    // ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    // loader: &Arc<RwLock<LoadRunner<Tensor>>>,
    // devices: Vec<Device>,
    // _descr: &str,
    presave: bool,
    f: F,
    ctx_f: CtxF,
) -> Result<Vec<RunId>, ThoosaError>
where
    ResConf: ResourcesConfig<R = Vec<Resource>> + Debug,
    F: FnOnce(Vec<RC>) -> Arc<ST>,
    ST: Stage + Send + Sync, // + From<(Arc<RwLock<Box<dyn Workspace>>>, Arc<RwLock<Box<dyn Store>>>, Arc<RwLock<LoadRunner<Tensor>>>, Vec<RC>, FC)>,
    RC: RunConfig
        + From<std::vec::IntoIter<NamedVal<<RCRV as GridRunConfigSource>::P>>>
        + thoosa_properties_map::ToHashmap,
    RCRV: GridRunConfigSource,
    // ResConf: ResourcesConfig,
    for<'c> <ST as Stage>::In<'c>: Send + Sync,
    CtxF: FnOnce() -> Arc<RwLock<<ST as Stage>::Ctx>>,
{
    let grid_run_configs = rcrv.grid_run_configs();
    debug!("run_all_grid_step: grid_run_configs={grid_run_configs:?}");

    let mut run_configs: Vec<RC> = grid_run_configs.to_run_configs();
    let mut run_ids = if presave {
        save_run_configs(
            input_run_ids.as_ref().map(|v| v.as_slice()),
            &store,
            &mut run_configs,
            "find motifs",
        )
    } else {
        vec![]
    };

    let stage = f(run_configs);

    let ctx = Arc::new(parking_lot::RwLock::new(
        crate::tasks::run_ctx_2::RunCtx2::new(0, 0, "aa"),
    ));

    stage.run(input, ctx_f()).await?;

    Ok(run_ids)
}

#[cfg(test)]
mod cli_tests {
    use crate::cli::{
        Cli, import_resources, resource_path, run_all_grid_step, run_all_grid_step_2, run_infer,
        run_opt_step, run_opt_step_2, run_plot_motif_slices, run_train_final_nets_2,
        run_train_label_nets, run_train_label_nets_2, run_train_motif_nodes_on_motif_selection,
        run_train_motif_nodes_on_motif_selection_2,
    };
    use crate::data::ParquetLoader;
    use clap::Parser;
    use futures::executor::block_on;
    use log::LevelFilter::{Debug, Error, Trace, Warn};
    use log::{LevelFilter, debug, error, info, set_max_level, trace};
    use parking_lot::lock_api::MutexGuard;
    use parking_lot::{Mutex, RawMutex, RawRwLock, RwLock};
    use plotters::prelude::{BLACK, IntoLinspace};
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
    use tch::Device::{Cpu, Cuda};
    use tch::display::{PrinterOptions, set_print_options};
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::{Device, IndexOp, Tensor, nn};
    use thoosa_common::applicables::ThoRunner;
    use thoosa_common::find_motifs_run_config::{
        FindMotifsRunConfig, FindMotifsRunConfigRangeOrValue, SelectMotifsRunConfigRangeOrValue,
    };
    use thoosa_common::input_probs::{InputProbs, compare_input_probs};
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::runs::{Empty, Labeled, Logits, NetResources, Simple};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{
        CommonTrainingRunConfig, InferArgs, MotifNetParams, OptConfig, TrainFinalNetRunConfig,
        TrainFinalNetRunConfigRangeOrValue, TrainLabelNetsRunConfig, TrainLabelNetsRunConfigFields,
        TrainLabelNetsRunConfigRangeOrValue, TrainMotifsNetsRunConfig,
        TrainMotifsNetsRunConfigRangeOrValue, motif_node,
    };
    use thoosa_common::utils::{
        from, lin_t, plot_series, reset_print_options, t_0_0, t_0_1, t_1_0, t_1_1,
        three_sigs_tensor, vec_f32_from,
    };
    use thoosa_common::ws::{FsWorkspace, Workspace};
    use thoosa_motifs::distance::MaskMaker::LeQuantile;
    use thoosa_motifs::distance::clc_dist_d2_and_mask;
    use thoosa_motifs::find::{
        FixedFindMotifsConfig, SelectMotifsParams, find_motifs_by_euc_dist,
        find_motifs_by_euc_dist_2, select_motifs_for_label, select_motifs_for_label_2,
    };
    use thoosa_motifs::fragments::{MotifsBy, PlotMotifSlicesArgs};
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::{Resource, ResourceId, ResourceStats, ResourcesConfig};
    use thoosa_shared::run_configs::RunConfig;
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::test_utils::set_panic_hook;
    use thoosa_shared::{
        ResourceType, TensorIdSliceIdx, ThoosaError, cdt, cdt_for_dir, random_saw, recurse_files,
        saw, saw_with_steps, sig_one_imp, sig_three_imp, sig_two_imp,
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

        let n = 1000;

        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            test_dir.clone(),
            3,
            n as u32,
            32000,
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
                SyncWrapper::wrap_tensor(
                    Tensor::load(tensors_path.join(format!("{}.pt", key.file_stem()))).unwrap(),
                )
            },
            4,
            100,
            2,
        )));

        let mut store = Arc::new(RwLock::new(SqlStore::from_path(ws.read().store_path())?));

        let imported_ids = import_resources(
            &ws,
            &store,
            &["label_0", "label_1"],
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

        let resources = store.read().list_resources_by_ids(&imported_ids)?;

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

        let run_ids = run_all_grid_step(
            None,
            serde_json5::from_str::<FindMotifsRunConfigRangeOrValue>(
                &r#"{
                step:"200",
                window:"1000",
                slices_filter: {'DistanceRunningMeanK':{'mean':'0.2'}}

            }"#,
            )
            .unwrap(),
            &FixedFindMotifsConfig::new(None, 2, false, None),
            &Simple::new(ResourceType::List(vec![
                "t_0_0".into(),
                "t_0_1".into(),
                "t_1_0".into(),
                "t_1_1".into(),
            ])),
            find_motifs_by_euc_dist,
            &ws,
            &store,
            &loader,
            vec![Cpu],
            "Test/FindMotifs",
            true,
        )?;

        let mut slices = store.read().list_motifs_slices(run_ids[0])?;

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
        // let motif_dists = store.read().list_motifs(run_ids[0])?;
        // debug!("test_motifs: motif_dists={motif_dists:?}");
        //
        // for motif_id in motif_dists.keys() {
        //     let (channel, run_id, motif_slice) = store.read().list_motif_slices(*motif_id, None, None)?;
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
        let motif_selection_id = store.read().list_input_run_ids(&select_motifs_run_ids)?;
        debug!("test_motifs_with_tpe: motif_selection_id={motif_selection_id:?}");

        // store.read().list_selected_motifs();

        // there are motifs with corr 1.0 for each label
        let selected_motifs_0 = store.read().list_selected_motifs(
            select_motifs_run_ids[0]
            // select_motifs_run_ids.iter().find(|(id, run_id)|label.eq("label_0")).expect("label_0").1
            //0 channel
        )?;
        debug!("selected_motifs_0={selected_motifs_0:?}");
        //TODO
        // assert!(selected_motifs_0[0].iter().find(|(motif_id, corr)|corr == &1.0).is_some());

        let selected_motifs_1 = store.read().list_selected_motifs(
            select_motifs_run_ids[1]
            // corrs.iter().find(|(id, run_id)|label.eq("label_1")).expect("label_1").1
        )?;
        debug!("selected_motifs_1={selected_motifs_1:?}");
        // assert!(selected_motifs_1[0].iter().find(|(motif_id, corr)|corr == &1.0).is_some());*/

        Ok(())
    }

    fn init_test_store(
        mut store: &Arc<RwLock<Box<dyn Store>>>,
        tensors_per_label: i32,
    ) -> Result<Vec<ResourceId>, ThoosaError> {
        let store_guard = store.write();
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

    fn label_1_t(n: u32, max_y: f32) -> Tensor {
        Tensor::vstack(&[Tensor::from_slice(
            [
                sig_one_imp(n, max_y, Some(10. ..12.), Some(10. ..12.), 0.1),
                sig_one_imp(n, max_y, Some(10. ..12.), Some(10. ..12.), 0.1),
            ]
            .concat()
            .as_slice(),
        )])
    }

    fn label_0_t(n: u32, max_y: f32) -> Tensor {
        Tensor::vstack(&[Tensor::from_slice(
            [
                sig_three_imp(n, max_y, Some(10. ..12.), Some(10. ..12.)),
                sig_two_imp(n, max_y, Some(10. ..12.), Some(10. ..12.), 0.1),
            ]
            .concat()
            .as_slice(),
        )])
    }

    /*

       input: batch x channels x n ->
       unfold: -> batch x channels x slices x slice len
       motif net: -> batch x channels x slices x 1(distance)
       label net: batch x channels * motif nets x slices x 1(distance, flattened) -> batch x labels(p)
       final net: batch x labels x labels(p) -> batch x labels(p)

    */
    #[test]
    fn heavy_test_run_all_fixed_slices_basic() -> Result<(), ThoosaError> {
        set_panic_hook();

        let test_dir = Path::new("/tmp").join(cdt_for_dir());

        let n = 1000;
        let max_y = 1f32;

        let (ws, store, group_name, loader, runner, devices) =
            prepare_test_data(n, max_y, &test_dir)?;

        let train_final_nodes_run_ids = run_all_steps(
            r#"{
                            step:"100",
                            window:"800",
                            slices_filter: {'DistanceRunningMeanK':{'mean':'0.3'}}
                        }"#,
            r#"{
                            motif_selector:{
                                ByLabelVariance: {_dummy: 0}
                            }
                        }"#,
            r#"{
                            common_training_args: {
                                    cycles: "100",
                                    batch_size: "11",
                                    lr: "0.0001..0.001",
                                    max_loss: "0.1",
                                    opt_max_evals: "10",
                                    opt_n_init: "3"
                            },
                            num_layers: "3"
                        }"#,
            r#"{
                            common_training_args: {
                                    cycles: "100",
                                    batch_size: "11",
                                    lr: "0.0001..0.001",
                                    max_loss: "0.1",
                                    opt_max_evals: "10",
                                    opt_n_init: "3"
                            },
                            net: {
                                FixedNumSlices: {
                                    num_layers: "4",
                                    slices_len: "333"
                                }
                            }
                        }"#,
            r#"{
                            common_training_args: {
                                cycles: "100",
                                batch_size: "11",
                                lr: "0.0001..0.001",
                                max_loss: "0.1",
                                opt_max_evals: "10",
                                opt_n_init: "3"
                            },
                            net: {
                                num_layers: "2"
                            }
                        }"#,
            &group_name,
            OptConfig::new(50, 10),
            &ws,
            &store,
            &loader,
            &devices,
            &runner,
        )?;

        debug!("train_final_nodes_run_ids={train_final_nodes_run_ids:?}");

        verify_infer(
            &test_dir,
            n,
            max_y,
            &ws,
            &store,
            devices,
            train_final_nodes_run_ids,
            0.2,
            &runner,
        )?;

        Ok(())
    }

    #[test]
    fn heavy_test_run_all_max_padding_basic() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());

        let n = 1000;
        let max_y = 1f32;

        let (ws, store, group_name, loader, runner, devices) =
            prepare_test_data(n, max_y, &test_dir)?;

        let train_final_nodes_run_ids = run_all_steps(
            r#"{
                step:"100",
                window:"800",
                slices_filter: {'DistanceRunningMeanK':{'mean':'0.1'}}
            }"#,
            r#"{
                motif_selector:{
                    ByLabelVariance: {_dummy: 0}
                }
            }"#,
            r#"{
                common_training_args: {
                        cycles: "100",
                        batch_size: "11",
                        lr: "0.0001..0.001",
                        max_loss: "0.1",
                        opt_max_evals: "10",
                        opt_n_init: "3"
                },
                num_layers: "3"
            }"#,
            r#"{
                common_training_args: {
                        cycles: "100",
                        batch_size: "11",
                        lr: "0.0001..0.001",
                        max_loss: "0.1",
                        opt_max_evals: "10",
                        opt_n_init: "3"
                },
                net: {
                    MaxPadding: {
                        num_layers: "2"
                    }
                }
            }"#,
            r#"{
                common_training_args: {
                    cycles: "100",
                    batch_size: "11",
                    lr: "0.0001..0.001",
                    max_loss: "0.1",
                    opt_max_evals: "10",
                    opt_n_init: "3"
                },
                net: {
                    num_layers: "2"
                }
            }"#,
            &group_name,
            OptConfig::new(10, 3),
            &ws,
            &store,
            &loader,
            &devices,
            &runner,
        )?;

        debug!("train_final_nodes_run_ids={train_final_nodes_run_ids:?}");

        verify_infer(
            &test_dir,
            n,
            max_y,
            &ws,
            &store,
            devices,
            train_final_nodes_run_ids,
            0.2,
            &runner,
        )?;

        Ok(())
    }

    #[test]
    fn heavy_test_run_all_max_padding_with_mean_grid_lr_opt() -> Result<(), ThoosaError> {
        set_panic_hook();

        let test_dir = Path::new("/tmp").join(cdt_for_dir());

        let n = 1000;
        let max_y = 1f32;

        let (ws, store, group_name, loader, runner, devices) =
            prepare_test_data(n, max_y, &test_dir)?;

        let train_final_nodes_run_ids = run_all_steps(
            r#"{
                step:"100",
                window:"800",
                slices_filter: {'DistanceRunningMeanK':{'mean':'0.1..0.2:0.05,2'}}
            }"#,
            r#"{
                motif_selector:{
                    ByLabelVariance: {_dummy: 0}
                }
            }"#,
            r#"{
                common_training_args: {
                        cycles: "100",
                        batch_size: "11",
                        lr: "0.0001..0.001",
                        max_loss: "0.1",
                        opt_max_evals: "10",
                        opt_n_init: "3"
                },
                num_layers: "3"
            }"#,
            r#"{
                common_training_args: {
                        cycles: "100",
                        batch_size: "11",
                        lr: "0.0001..0.001",
                        max_loss: "0.1",
                        opt_max_evals: "10",
                        opt_n_init: "3"
                },
                net: {
                    MaxPadding: {
                        num_layers: "2"
                    }
                }
            }"#,
            r#"{
                common_training_args: {
                    cycles: "100",
                    batch_size: "11",
                    lr: "0.0001..0.001",
                    max_loss: "0.1",
                    opt_max_evals: "10",
                    opt_n_init: "3"
                },
                net: {
                    num_layers: "2"
                }
            }"#,
            &group_name,
            OptConfig::new(10, 3),
            &ws,
            &store,
            &loader,
            &devices,
            &runner,
        )?;

        debug!("train_final_nodes_run_ids={train_final_nodes_run_ids:?}");

        verify_infer(
            &test_dir,
            n,
            max_y,
            &ws,
            &store,
            devices,
            train_final_nodes_run_ids,
            0.2,
            &runner,
        )?;

        Ok(())
    }

    #[test]
    fn heavy_test_run_all_max_padding_band_filter() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());

        let n = 1000;
        let max_y = 1f32;

        let (ws, store, group_name, loader, runner, devices) =
            prepare_test_data(n, max_y, &test_dir)?;

        let train_final_nodes_run_ids = run_all_steps(
            r#"{
                            step:"100",
                            window:"800",
                            filter:{
                                BandPassFilter:{
                                    low_hz:"1",
                                    band_hz:"10",
                                    kernel_size:"1000",
                                    stride:"1",
                                    padding:"1"
                                }
                            },
                            slices_filter: {'DistanceRunningMeanK':{'mean':'0.2'}}
                        }"#,
            r#"{
                            motif_selector:{
                                ByLabelVariance: {_dummy: 0}
                            }
                        }"#,
            r#"{
                            common_training_args: {
                                    cycles: "100",
                                    batch_size: "11",
                                    lr: "0.0001..0.001",
                                    max_loss: "0.1",
                                    opt_max_evals: "10",
                                    opt_n_init: "3"
                            },
                            num_layers: "3"
                        }"#,
            r#"{
                            common_training_args: {
                                    cycles: "100",
                                    batch_size: "11",
                                    lr: "0.0001..0.001",
                                    max_loss: "0.1",
                                    opt_max_evals: "10",
                                    opt_n_init: "3"
                            },
                            net: {
                                MaxPadding: {
                                    num_layers: "2"
                                }
                            }
                        }"#,
            r#"{
                            common_training_args: {
                                cycles: "100",
                                batch_size: "11",
                                lr: "0.0001..0.001",
                                max_loss: "0.1",
                                opt_max_evals: "10",
                                opt_n_init: "3"
                            },
                            net: {
                                num_layers: "2"
                            }
                        }"#,
            &group_name,
            OptConfig::new(10, 3),
            &ws,
            &store,
            &loader,
            &devices,
            &runner,
        )?;

        debug!("train_final_nodes_run_ids={train_final_nodes_run_ids:?}");

        verify_infer(
            &test_dir,
            n,
            max_y,
            &ws,
            &store,
            devices,
            train_final_nodes_run_ids,
            0.2,
            &runner,
        )?;

        Ok(())
    }

    #[test]
    fn heavy_test_run_all_fixed_slices_band_filter() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());

        let n = 1000;
        let max_y = 1f32;

        let (ws, store, group_name, loader, runner, devices) =
            prepare_test_data(n, max_y, &test_dir)?;

        let train_final_nodes_run_ids = run_all_steps(
            r#"{
                            step:"100",
                            window:"800",
                            filter:{
                                BandPassFilter:{
                                    low_hz:"1",
                                    band_hz:"10",
                                    kernel_size:"1000",
                                    stride:"1",
                                    padding:"1"
                                }
                            },
                            slices_filter: {'DistanceRunningMeanK':{'mean':'0.2'}}
                        }"#,
            r#"{
                            motif_selector:{
                                ByLabelVariance: {_dummy: 0}
                            }
                        }"#,
            r#"{
                            common_training_args: {
                                    cycles: "100",
                                    batch_size: "11",
                                    lr: "0.0001..0.001",
                                    max_loss: "0.1",
                                    opt_max_evals: "10",
                                    opt_n_init: "3"
                            },
                            num_layers: "3"
                        }"#,
            r#"{
                            common_training_args: {
                                    cycles: "100",
                                    batch_size: "11",
                                    lr: "0.0001..0.001",
                                    max_loss: "0.1",
                                    opt_max_evals: "10",
                                    opt_n_init: "3"
                            },
                            net: {
                                FixedNumSlices: {
                                    num_layers: "2",
                                    slices_len: "333"
                                }
                            }
                        }"#,
            r#"{
                            common_training_args: {
                                cycles: "100",
                                batch_size: "11",
                                lr: "0.0001..0.001",
                                max_loss: "0.1",
                                opt_max_evals: "10",
                                opt_n_init: "3"
                            },
                            net: {
                                num_layers: "2"
                            }
                        }"#,
            &group_name,
            OptConfig::new(10, 3),
            &ws,
            &store,
            &loader,
            &devices,
            &runner,
        )?;

        debug!("train_final_nodes_run_ids={train_final_nodes_run_ids:?}");

        verify_infer(
            &test_dir,
            n,
            max_y,
            &ws,
            &store,
            devices,
            train_final_nodes_run_ids,
            0.2,
            &runner,
        )?;

        Ok(())
    }

    fn verify_infer(
        test_dir: &PathBuf,
        n: u32,
        max_y: f32,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        devices: Vec<Device>,
        train_final_nodes_run_ids: Vec<RunId>,
        max_mse: f64,
        runner: &Arc<ThoRunner>,
    ) -> Result<(), ThoosaError> {
        let input_dir = test_dir.join("infer_input");

        debug!("run_infer: input_dir={input_dir:?}");

        fs::create_dir_all(input_dir.as_path())?;
        let mut expected = InputProbs::new(vec!["Label_0".to_string(), "Label_1".to_string()]);
        let sig_0 = sig_one_imp(n, 30.0, None, Some(20. ..20.1), 0.1);
        let sig_1 = sig_one_imp(n, 30.0, None, Some(1. ..1.01), 0.1);

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

        let inferred = block_on(run_infer(
            &ws,
            &store,
            &InferArgs::new(
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
            runner,
        ))?;

        info!("inferred: labels={:?}", inferred.labels());

        for (file, probs) in inferred.inputs_probs() {
            info!("file={file} probs={probs:?}")
        }

        let (mse, _) = compare_input_probs(&expected, &inferred);

        assert!(mse < max_mse);
        Ok(())
    }

    fn _saw(n: u32) -> Vec<f32> {
        saw_with_steps(n as usize, 10, 5.)
    }

    fn label_0_t_rs(n: u32, sig_0_v: &Vec<f32>, sig_2_v: &Vec<f32>, max_y: f32) -> Tensor {
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

    fn label_1_t_rs(n: u32, sig_1_v: &Vec<f32>, sig_2_v: &Vec<f32>, max_y: f32) -> Tensor {
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

    fn prepare_test_data(
        n: u32,
        max_y: f32,
        test_dir: &PathBuf,
    ) -> Result<
        (
            Arc<RwLock<Box<dyn Workspace>>>,
            Arc<RwLock<Box<dyn Store>>>,
            String,
            Arc<RwLock<LoadRunner<Tensor>>>,
            Arc<ThoRunner>,
            Vec<Device>,
        ),
        ThoosaError,
    > {
        unsafe {
            let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        // log::set_max_level(Warn);

        let tensors_per_label = 10;

        debug!("test_run_step: test_dir={test_dir:?}");
        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            test_dir.clone(),
            3,
            3 * n as u32,
            32000,
        )?));
        let ws_loader = Arc::clone(&ws);
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
        });

        (0..tensors_per_label).for_each(|i| {
            label_1_t_rs(n, &sig_1, &sig_three_imp(n, 30.0, None, None), max_y)
                .save(ws.read().tensors_path().join(format!("1_t_1_{i}.pt")))
                .unwrap();
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

        let store = Arc::new(RwLock::new(SqlStore::from_path(ws.read().store_path())?));

        init_test_store(&store, tensors_per_label)?;
        let group_name = format!("pure group {tensors_per_label}");

        let loader = Arc::new(RwLock::new(LoadRunner::new(
            move |key: &Resource| {
                let path = ws_loader.read().tensor_path(key);
                debug!("test_run_step: path={path:?}");
                SyncWrapper::wrap_tensor(
                    Tensor::load(&path).expect(format!("can't open {path:?}").as_str()),
                )
            },
            4, //std::thread::available_parallelism as usize/2,
            100,
            3,
        )));

        let runner = Arc::new(ThoRunner::new(&[20, 2], move |key, device| {
            let path = ws_runner.read().tensor_path(key);
            debug!("test_run_step: path={path:?}");

            ThoTensor::new(
                Input::id(key).clone(),
                Tensor::load(&path).expect(format!("can't open {path:?}").as_str()),
            )
        }));

        // let devices = vec![Cpu, Cpu];
        let devices = vec![Cuda(0)];

        // log::set_max_level(Trace);
        reset_print_options();
        Ok((ws, store, group_name, loader, runner, devices))
    }

    fn run_all_steps(
        find_motifs_run_config: &str,
        select_motifs_run_config: &str,
        train_motif_nodes_run_config: &str,
        train_label_nodes_run_config: &str,
        train_final_nodes_run_config: &str,
        group_name: &String,
        opt_config: OptConfig,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        loader: &Arc<RwLock<LoadRunner<Tensor>>>,
        devices: &Vec<Device>,
        runner: &Arc<ThoRunner>,
    ) -> Result<Vec<RunId>, ThoosaError> {
        let opt_config_arc = Arc::new(opt_config.clone());

        let find_motifs_run_ids = run_all_grid_step_2(
            None,
            serde_json5::from_str::<FindMotifsRunConfigRangeOrValue>(&find_motifs_run_config)
                .unwrap(),
            &FixedFindMotifsConfig::new(None, 2, false, None),
            &Simple::new(ResourceType::Group(group_name.clone())),
            find_motifs_by_euc_dist_2,
            ws,
            store,
            runner,
            devices.clone(),
            "Test/FindMotifs",
            true,
        )?;

        // run_plot_motif_slices(
        //     &ws,
        //     &store,
        //     &loader,
        //     &PlotMotifSlicesArgs::new(MotifsBy::RunId(find_motifs_run_ids[0])),
        //     &devices[0],
        // )?;
        // debug!("test_run_step: find_motifs_run_ids={find_motifs_run_ids:?}");

        let select_motifs_for_label_run_ids = run_all_grid_step_2(
            Some(find_motifs_run_ids),
            serde_json5::from_str::<SelectMotifsRunConfigRangeOrValue>(&select_motifs_run_config)
                .unwrap(),
            &(),
            &Labeled::new(group_name.clone()),
            select_motifs_for_label_2,
            ws,
            store,
            runner,
            devices.clone(),
            "Test/SelectMotifs",
            false,
        )?;

        debug!(
            "test_run_step: select_motifs_for_label_run_ids={select_motifs_for_label_run_ids:?}"
        );

        for select_motifs_for_label_run_id in &select_motifs_for_label_run_ids {
            let selected_motifs = store.read().list_selected_motifs(
                *select_motifs_for_label_run_id,
                ws.read().channels() as u32,
            )?;

            debug!(
                "test_run_step: select_motifs_for_label_run_id={select_motifs_for_label_run_id} selected_motifs[0].len={} selected_motifs={selected_motifs:?}",
                selected_motifs[0].len()
            );
        }

        let train_motif_nodes_run_ids = run_opt_step_2::<
            TrainMotifsNetsRunConfig,
            TrainMotifsNetsRunConfigRangeOrValue,
            OptConfig,
            Empty,
            _,
        >(
            Some(select_motifs_for_label_run_ids.clone()),
            &Arc::new(
                serde_json5::from_str::<TrainMotifsNetsRunConfigRangeOrValue>(
                    &train_motif_nodes_run_config,
                )
                .unwrap(),
            ),
            Arc::clone(&opt_config_arc),
            &Empty {},
            run_train_motif_nodes_on_motif_selection_2,
            ws,
            store,
            runner,
            devices.clone(),
            "Test/TrainMotifsNets",
        )?;
        debug!("train_motif_nodes_run_ids={train_motif_nodes_run_ids:?}");

        let train_label_nodes_run_ids = run_opt_step_2::<
            TrainLabelNetsRunConfig,
            TrainLabelNetsRunConfigRangeOrValue,
            OptConfig,
            NetResources<Labeled>,
            _,
        >(
            Some(select_motifs_for_label_run_ids.clone()),
            &Arc::new(
                serde_json5::from_str::<TrainLabelNetsRunConfigRangeOrValue>(
                    &train_label_nodes_run_config,
                )
                .unwrap(),
            ),
            Arc::clone(&opt_config_arc),
            &NetResources::new(
                Labeled::new(group_name.clone()),
                Labeled::new(group_name.clone()),
            ),
            run_train_label_nets_2,
            &ws,
            &store,
            runner,
            devices.clone(),
            "Test/TrainLabelsNets",
        )?;

        debug!("train_label_nodes_run_ids={train_label_nodes_run_ids:?}");

        let train_final_nodes_run_ids = run_opt_step_2::<
            TrainFinalNetRunConfig,
            TrainFinalNetRunConfigRangeOrValue,
            OptConfig,
            NetResources<Logits>,
            _,
        >(
            Some(train_label_nodes_run_ids.clone()),
            &Arc::new(
                serde_json5::from_str::<TrainFinalNetRunConfigRangeOrValue>(
                    &train_final_nodes_run_config,
                )
                .unwrap(),
            ),
            Arc::clone(&opt_config_arc),
            &NetResources::new(
                Logits::new(group_name.clone()),
                Logits::new(group_name.clone()),
            ),
            run_train_final_nets_2,
            &ws,
            &store,
            runner,
            devices.clone(),
            "Test/TrainFinalNets",
        )?;

        log::logger().flush();
        info!(
            "runner.statistics: {:?}",
            runner.futures_runner().statistics()
        );
        Ok(train_final_nodes_run_ids)
    }

    #[test]
    fn test_import_resources() -> Result<(), ThoosaError> {
        let n = 1000i64;
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            test_dir,
            1,
            2 * n as u32,
            32000,
        )?));

        let store = Arc::new(RwLock::new(SqlStore::in_mem()?));
        // std::fs::remove_file("/tmp/test_main_db.db3").unwrap_or(());
        // let store = Arc::new(RwLock::new(SqlStore::from_path("/tmp/test_main_db.db3")?));

        let imported_ids = import_resources(
            &ws,
            &store,
            &["label_0", "label_1"],
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

        let imported_resources = store.read().list_resources_by_ids(&imported_ids)?;

        debug!("imported={imported_resources:?}");
        let mut imported_keys = imported_resources
            .into_iter()
            .map(|r| r.file_stem().to_string())
            .collect::<Vec<_>>();
        imported_keys.sort();

        assert_eq!(vec!["t_0_0", "t_0_1", "t_1_0", "t_1_1",], imported_keys);

        Ok(())
    }
}
