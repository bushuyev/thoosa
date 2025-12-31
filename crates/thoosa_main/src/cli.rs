use crate::cli::Commands::{
    FindMotifs, ImportResources, Infer, MotifsToTensor, PlotMotifSlices, RunAll, SayHello,
    SelectMotifs, TrainFinalNet, TrainMotifsNets,
};
use crate::data;
use crate::data::{clc_records, labeled_tensors_by_descr, load_to_logits_tensors};
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::RawTensorLoader;
use crate::version::VERSION;
use async_trait::async_trait;
use thoosa_shared::runner::common::runnable::Stage;
use clap::{Args, Parser, Subcommand};
use futures::executor::block_on;
use futures_util::future::{join_all, try_join_all};
use futures_util::{StreamExt, TryStreamExt};
use log::{debug, error, info, trace, warn, Level, LevelFilter};
use parking_lot::{RawRwLock, RwLock};
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use simplelog::{ColorChoice, CombinedLogger, Config, TermLogger, TerminalMode, WriteLogger};
use std::cmp::max;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::{read_dir, File, OpenOptions};
use std::io::Write;
use std::ops::{Deref, Range};
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
use tch::kind::FLOAT_CPU;
use thoosa_common::tho_device::{ThoDevice};
use thoosa_common::dims::{Batch, ChannelsData, ThoBatch, TCD};
use thoosa_common::find_motifs_run_config::{ByLabelVarianceConfig, ByLabelVarianceConfigRangeOrValue, DistanceRunningMeanK, DistanceRunningMeanKRangeOrValue, FindMotifsRunConfig, FindMotifsRunConfigRangeOrValue, MotifSelector, MotifSelectorRangeOrValue, SelectMotifsRunConfigRangeOrValue, SlicesFilterRangeOrValue};
use thoosa_common::input_probs::InputProbs;
use thoosa_common::loader::{LoadRunner, SyncWrapper};
use thoosa_common::runs::{Empty, Labeled, Logits, NetResources, SavableResult, Simple};
use thoosa_common::train_nets;
use thoosa_common::train_nets::{load_final_model, opt_train, opt_train_2, CommonTrainingRunConfig, CommonTrainingRunConfigRangeOrValue, FinalNetParams, FixedNumSlicesNetArgsRangeOrValue, InferArgs, LabelNetParams, LabelNetTypeArgsRangeOrValue, MotifNetParams, OptConfig, PlotVecf32, TensorsDataBatch, ThoModuleWithStore, TrainFinalNetArgs, TrainFinalNetRunConfig, TrainFinalNetRunConfigFields, TrainFinalNetRunConfigRangeOrValue, TrainFinalNetsRunConfigResult, TrainLabelNetsArgs, TrainLabelNetsRunConfig, TrainLabelNetsRunConfigFields, TrainLabelNetsRunConfigRangeOrValue, TrainLabelNetsRunConfigResult, TrainMotifsNetResult, TrainMotifsNetsArgs, TrainMotifsNetsResult, TrainMotifsNetsRunConfig, TrainMotifsNetsRunConfigFields, TrainMotifsNetsRunConfigRangeOrValue};
use thoosa_common::utils::{cuda_mem, from, lin_t, list_devices, plot_series, reset_print_options, DummyListener, TrainListener};
use thoosa_common::ws::{FsWorkspace, Workspace};
use thoosa_motifs::find;
use thoosa_motifs::find::{FindMotifsArgs, FixedFindMotifsConfig, SelectMotifsArgs};
use thoosa_motifs::fragments::{
    MotifsBy, MotifsToTensorArgs, PlotMotifSlicesArgs};
use thoosa_properties_map::ToHashmap;
use thoosa_run_config_args::{GridRunConfigSource, OptRunConfigSource};
use thoosa_shared::applicables_runner::{Input};
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::resource::{Resource, ResourceId, ResourceStats, ResourceStore, ResourcesConfig};
use thoosa_shared::run_configs::{OptRunConfigs, RunConfig, RunId};
use thoosa_shared::traits::{Data, DataId, SimpleId, WithId};
use thoosa_shared::{mono_key, parse_resources, simple_id, RangeOrValue};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_shared::{cdt, cdt_for_dir, ResourceType, ThoosaError};
use thoosa_store::{SqlStore, Store};
use tokio::runtime::Runtime;
use thoosa_common::tho_tensor::ThoTensor;
use torch_sys::at_grad_set_enabled;
use thoosa_common::find_motifs_run_config::InputFilterRangeOrValue::BandPassFilter;
use thoosa_common::transforms::filters::BandPassFilterParamsRangeOrValue;
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, DevicesWrapCopyMode, WithDataRef};
use thoosa_shared::runner::common::RunCtx;
use thoosa_shared::runner::common::runnable::AsyncStage;
use thoosa_shared::runner::global_ctx::GlobalCtx;
use thoosa_motifs::distance::average_euclidean_distance;
use thoosa_shared::colors_iter::DistinctColorIter;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::perf_logging::{format_perf_map, PERF};
use crate::tasks::average_euclidean_distance::{AverageDistanceValue, AverageEuclideanDistanceStage};
use crate::tasks::load_motif_and_clc_dist::{LoadMotifAndClcDistanceStage, MotifSlicesSelector};
use crate::tasks::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta};
use crate::tasks::select_motifs::SelectMotifsStage;
use crate::tasks::train_label_nets::{new_train_label_nets_with_ml, TrainLabelNets, TrainLabelNetsInput};
use crate::tasks::train_motif_nets::{TrainMotifNetInput, TrainMotifNets};

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
    find_resources: String,

    #[arg(long)]
    select_resources: String,

    #[arg(long, default_value = "XXX")]
    train_resource_group: String,

    #[arg(long, default_value = "XXX")]
    valid_resource_group: String,

    #[arg(long)]
    pub run_config_cnt: String, //TrainLabelNetsRunConfig
}

impl RunAllArgs {

    pub fn find_resource_config(&self) -> Result<Simple, ThoosaError> {
        Ok(Simple::new(parse_resources(&self.find_resources)?))
    }

    pub fn select_resource_config(&self) -> Result<Simple, ThoosaError> {
        Ok(Simple::new(parse_resources(&self.select_resources)?))
    }

    pub fn train_resource_config(&self) -> NetResources<Labeled> {
        NetResources::new(
            Labeled::new(self.train_resource_group.clone()),
            Labeled::new(self.valid_resource_group.clone()),
        )

    }

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
    if log::log_enabled!(log::Level::Trace) {
        reset_print_options();
    }
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
    thoosa_shared::test_utils::maybe_init_perf_tracing_stdout("info");

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

    GlobalCtx::init_devices(2);

    let loader_2 = Arc::new(RawTensorLoader::<RunCtx2>::new(
        move |key, _| {
            ThoTensor::new(
                SynKey::new([mono_key!(key.id(), format!("resource_id={}", key.id()))]),
                Tensor::load(tensors_path_2.join(format!(
                    "{}_{}.pt",
                    key.import_id(),
                    key.file_stem()
                )))
                .unwrap(),
                // Tensor::zeros(&[20, 10_000], FLOAT_CPU)
            )
        },
        ThoDevice::from(Device::Cpu),
    ));

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
            debug!("FindMotifs: find_motifs_args={:?} devices={:?}",find_motifs_args, devices);

            let find_motifs_run_config_range_or_value: FindMotifsRunConfigRangeOrValue =
                serde_json5::from_str(&find_motifs_args.run_config_cnt)
                    .expect("FindMotifsRunConfig2");

            debug!("FindMotifs: find_motifs_run_config_range_or_value={find_motifs_run_config_range_or_value:?}");

            let input_wrapper = Arc::new(DevicesWrap::wrap(
                ResourcesConfigInput::new(find_motifs_args.resource_config()),
                0
            ));

            let run_ids = run_grid::</*Simple,*/ _, _, _, _, _>(
                None,
                find_motifs_run_config_range_or_value,
                input_wrapper,
                &ws,
                &store,
                true,
                |run_configs| {
                    Arc::new(
                        crate::tasks::find_motifs::FindMotifsStage::new(
                            &ws,
                            &store,
                            loader_2,
                            run_configs,
                            find_motifs_args.fixed_config.clone(), //FixedFindMotifsConfig::new(None, 10_000, false, None),
                            ThoDevice::from(1),
                        )
                        .expect("FindMotifsStage::new failed"),
                    )
                },
                || Arc::new(parking_lot::RwLock::new(RunCtx2::new(1, 0, "find motifs"))),
            ).await?;

            info!("FindMotifs done, run_ids={run_ids:?}");

        }

        Some(SelectMotifs(select_motifs_args)) => {
            debug!("SelectMotifs2: select_motifs_args.run_config_cnt={:?}",select_motifs_args.run_config_cnt);

            let select_motifs_run_config_range_or_value: SelectMotifsRunConfigRangeOrValue = serde_json5::from_str(&select_motifs_args.run_config_cnt).expect("SelectMotifs2");

            let input_wrapper = Arc::new(DevicesWrap::wrap(
                ResourcesConfigInput::new(select_motifs_args.resource_config()),
                0
            ));
            //TODO move it into SelectMotifsStage
            let motif_loader = Arc::new(MotifLoader::new(
                Arc::clone(&store),
                Arc::clone(&loader_2)
            ));

            let run_ids = run_grid::</*Labeled,*/ _, _, _, _, _>(
                None,
                select_motifs_run_config_range_or_value,
                input_wrapper,
                &ws,
                &store,
                true,
                |run_configs| {

                    Arc::new(SelectMotifsStage::<RawTensorLoader<RunCtx2>, MotifLoader<RawTensorLoader<RunCtx2>, RunCtx2>, RunCtx2>::new(
                        &ws,
                        &store,
                        loader_2,
                        motif_loader,
                        run_configs,
                        select_motifs_args.find_motifs_run_ids.clone(),
                        100_000,//TODO config
                        #[cfg(test)]
                        None
                    ))
                },
                || Arc::new(parking_lot::RwLock::new(RunCtx2::new(1, 0, "select motifs"))),
            ).await?;

            info!("SelectMotifs done, run_ids={run_ids:?}");
        }

        Some(TrainMotifsNets(train_motif_nets_args)) => {
            debug!("TrainMotifsNets2: train_motif_nets_args.run_config_cnt={:?}",train_motif_nets_args.run_config_cnt);

            let train_motif_nets_run_config_range_or_value: TrainMotifsNetsRunConfigRangeOrValue = serde_json5::from_str(&train_motif_nets_args.run_config_cnt).expect("TrainMotifsNets2");

            let input_wrapper = Arc::new(DevicesWrap::wrap(
                TrainMotifNetInput::new(train_motif_nets_args.select_motifs_run_ids.clone(), Arc::new(train_motif_nets_args.fixed_config.clone())),
                0
            ));

            let run_ids = run_opt::</*Labeled,*/ _, _, TrainMotifsNetsRunConfig, _, _>(
                None,
                train_motif_nets_run_config_range_or_value,
                input_wrapper,
                &ws,
                &store,
                true,
                async |run_configs| {

                    Ok(Arc::new(TrainMotifNets::<_, RunCtx2>::new(
                        loader_2,
                        &ws,
                        &store,
                        run_configs,
                    )))
                },
                || Arc::new(parking_lot::RwLock::new(RunCtx2::new(1, 0, "train motif nets"))),
            ).await?;

        }

        Some(Commands::TrainLabelNets(train_label_nets_args)) => {
            debug!("TrainLabelNets: train_label_nets_args.run_config_cnt={:?}",train_label_nets_args.run_config_cnt);

            let train_motif_nets_run_config_range_or_value: TrainLabelNetsRunConfigRangeOrValue = serde_json5::from_str(&train_label_nets_args.run_config_cnt).expect("TrainLabelNets");

            let input_wrapper = Arc::new(DevicesWrap::wrap(
                ResourcesConfigInput::new( //TODO the group is needed to load resouces not belonging to label being trained
                                           train_label_nets_args.resource_config()),
                0
            ));

            let opt_config = Arc::new(train_label_nets_args.fixed_config.clone());


            let run_ids = run_opt::</*Labeled,*/ _, _, TrainLabelNetsRunConfig, _, _>(
                None,
                train_motif_nets_run_config_range_or_value,
                input_wrapper,
                &ws,
                &store,
                true,
                async |run_configs| {

                    new_train_label_nets_with_ml(
                        train_label_nets_args.select_motifs_run_id.clone(),
                        loader_2,
                        run_configs,
                        &ws,
                        &store,
                        Arc::new(parking_lot::RwLock::new(RunCtx2::new(1, GlobalCtx::global().next_cache_area()?, "init label nets"))),
                        opt_config
                    ).await.map(|v|Arc::new(v))
                },
                || Arc::new(parking_lot::RwLock::new(RunCtx2::new(1, 0, "train label nets"))),
            ).await?;



            info!("TrainMotifsNets done");
        }

        Some(TrainFinalNet(train_final_net_args)) => {
            debug!("TrainFinalNet: train_motif_nets_args.run_config_cnt={:?}",train_final_net_args.run_config_cnt);
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

            let find_motifs_input = Arc::new(DevicesWrap::wrap(
                ResourcesConfigInput::new(run_all_params.find_resource_config()?).check(&store)?,
                0
            ));

            let select_motifs_input = Arc::new(DevicesWrap::wrap(
                ResourcesConfigInput::new(run_all_params.select_resource_config()?).check(&store)?,
                0
            ));

            let train_labels_input = Arc::new(DevicesWrap::wrap(
                ResourcesConfigInput::new( //TODO the group is needed to load resouces not belonging to label being trained
                                           run_all_params.train_resource_config()).check(&store)?,
                0
            ));


            let find_motifs_run_config_range_or_value:FindMotifsRunConfigRangeOrValue = FindMotifsRunConfigRangeOrValue{
                id: None,
                step: RangeOrValue::Value(100), //RangeOrValue::Range { range: Range { start: 50, end: 100 }, step: Some((100, 0)) },
                window: RangeOrValue::Value(1000), //RangeOrValue::Range { range: Range { start: 1000, end: 2001 }, step: Some((1000, 1)) },
                min_tracker_step: None,
                // filter: Some(BandPassFilter(BandPassFilterParamsRangeOrValue{
                //     low_hz: RangeOrValue::Range { range: Range { start: 0., end: 201. }, step: Some((100., 1)) },
                //     band_hz: RangeOrValue::Value(100.),
                //     kernel_size: RangeOrValue::Value(1000),
                //     stride: RangeOrValue::Value(1),
                //     padding: RangeOrValue::Value(0),
                // })),
                filter: None,
                slices_filter: SlicesFilterRangeOrValue::DistanceRunningMeanK(DistanceRunningMeanKRangeOrValue{ mean: RangeOrValue::Value(0.02)}),
            };

            let find_motif_run_ids = run_grid::</*Simple,*/ _, _, _, _, _>(
                None,
                find_motifs_run_config_range_or_value,
                find_motifs_input,
                &ws,
                &store,
                true,
                |run_configs| {
                    Arc::new(
                        crate::tasks::find_motifs::FindMotifsStage::new(
                            &ws,
                            &store,
                            Arc::clone(&loader_2),
                            run_configs,
                            FixedFindMotifsConfig::new(None, 10_000, false, None),
                            ThoDevice::from(1),
                        )
                        .expect("FindMotifsStage::new failed"),
                    )
                },
                || Arc::new(parking_lot::RwLock::new(RunCtx2::new(1, 0, "find motifs"))),
            ).await?;

            info!("find motifs done, find_motif_run_ids={find_motif_run_ids:?}");


            let select_motifs_run_ids = run_grid::</*Labeled,*/ _, _, _, _, _>(
                Some(find_motif_run_ids.clone()),
                SelectMotifsRunConfigRangeOrValue{
                    id: None,
                    motif_selector: MotifSelectorRangeOrValue::ByLabelVariance(ByLabelVarianceConfigRangeOrValue{ _dummy: 0 })
                },
                select_motifs_input,
                &ws,
                &store,
                true,
                |run_configs| {

                    Arc::new(SelectMotifsStage::<RawTensorLoader<RunCtx2>, MotifLoader<RawTensorLoader<RunCtx2>, RunCtx2>, RunCtx2>::new(
                        &ws,
                        &store,
                        Arc::clone(&loader_2),
                        Arc::new(MotifLoader::new(
                            Arc::clone(&store),
                            Arc::clone(&loader_2)
                        )),
                        run_configs,
                        find_motif_run_ids.clone(),
                        100_000, //TODO config
                        #[cfg(test)]
                        None
                    ))
                },
                || Arc::new(parking_lot::RwLock::new(RunCtx2::new(1, 0, "select motifs"))),
            ).await?;

            info!("select motifs done: find_motif_run_ids={find_motif_run_ids:?} select_motifs_run_ids={select_motifs_run_ids:?}");



            let common_training_args = CommonTrainingRunConfigRangeOrValue {
                cycles: RangeOrValue::Value(100),
                batch_size: RangeOrValue::Value(20),
                lr: RangeOrValue::Value(0.0001),
            };
            let opt_config = Arc::new(OptConfig::new(100, 20));

            let train_motif_nets_run_config_range_or_value: TrainMotifsNetsRunConfigRangeOrValue = TrainMotifsNetsRunConfigRangeOrValue {
                id: None,
                common_training_args: common_training_args.clone(),
                num_layers: RangeOrValue::Range { range: Range { start: 3, end: 10 }, step: None },
                step: RangeOrValue::Value(10),
            };


            let input_wrapper = Arc::new(DevicesWrap::wrap(
                TrainMotifNetInput::new(select_motifs_run_ids.clone(), Arc::clone(&opt_config)),
                0
            ));

            let run_ids = run_opt::</*Labeled,*/ _, _, TrainMotifsNetsRunConfig, _, _>(
                None,
                train_motif_nets_run_config_range_or_value,
                input_wrapper,
                &ws,
                &store,
                true,
                async |run_configs| {

                    Ok(Arc::new(TrainMotifNets::<_, RunCtx2>::new(
                        Arc::clone(&loader_2),
                        &ws,
                        &store,
                        run_configs,
                    )))
                },
                || Arc::new(parking_lot::RwLock::new(RunCtx2::new(1, 0, "train motif nets"))),
            ).await?;

            info!("train motif nets done: run_ids={run_ids:?}");

            ///////////////////
            let train_label_nets_run_config_range_or_value: TrainLabelNetsRunConfigRangeOrValue = TrainLabelNetsRunConfigRangeOrValue{
                id: None,
                common_training_args,
                net: LabelNetTypeArgsRangeOrValue::FixedNumSlices(FixedNumSlicesNetArgsRangeOrValue{
                    num_layers: RangeOrValue::Range { range: Range { start: 3, end: 10 }, step: None },
                    slices_len: RangeOrValue::Value(1000)
                }),
            };

            let select_motifs_run_ids = select_motifs_run_ids.clone();

            let run_ids = run_opt::</*Labeled,*/ _, _, TrainLabelNetsRunConfig, _, _>(
                None,
                train_label_nets_run_config_range_or_value,
                train_labels_input,
                &ws,
                &store,
                true,
                async |run_configs| {

                    new_train_label_nets_with_ml(
                        select_motifs_run_ids,
                        loader_2,
                        run_configs,
                        &ws,
                        &store,
                        Arc::new(parking_lot::RwLock::new(RunCtx2::new(1, 0, "init label nets"))),
                        Arc::clone(&opt_config)
                    ).await.map(|v|Arc::new(v))
                },
                || Arc::new(parking_lot::RwLock::new(RunCtx2::new(1, 0, "train label nets"))),
            ).await?;

            info!("train label nets done: run_ids={run_ids:?}");

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

            let select_motifs_run_ids = run_step(
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
                "select_motifs done: select_motifs_run_ids={:?}",
                select_motifs_run_ids
            );
            unsafe {
                at_grad_set_enabled(1);
            }
            let train_motif_nodes_run_ids = run_step(
                Some(select_motifs_run_ids.clone()),
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
                Some(select_motifs_run_ids),
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

    info!("\n{}", format_perf_map(&PERF));

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
pub struct TrainLabelNetApplicable {
    device: Device,
}

impl TrainLabelNetApplicable {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
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
                trace!("import_resources: resource_key={resource_key} ok={} err={}",a.0.len(),a.1.len());

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
    loader: &Arc<RawTensorLoader<RunCtx2>>,
    plot_motif_slices_params: &PlotMotifSlicesArgs,
    device: &Device,
) -> Result<(), ThoosaError> {
    let ctx = Arc::new(parking_lot::RwLock::new(RunCtx2::new(1, 0, "aa")));

    let motif_loader_impl = Arc::new(MotifLoader::<RawTensorLoader<RunCtx2>, RunCtx2>::new(
        Arc::clone(store),
        Arc::clone(loader)
    ));
    let clc_dist = Arc::new(LoadMotifAndClcDistanceStage::new(
        Arc::clone(&motif_loader_impl), // as Arc<dyn MotifLoaderStage<Output = <AverageEuclideanDistanceStage as Stage>::In<'static>>>,
        Arc::new(AverageEuclideanDistanceStage::<RunCtx2>::new()),
    ));

    match plot_motif_slices_params.motifs_by() {
        MotifsBy::MotifId(motif_id) => {
            debug!("run_plot_motif_slices: MotifsBy::MotifId run_id={motif_id}");

            plot_motif(&ws, ctx, motif_id, &motif_loader_impl, clc_dist, None).await?;

        }

        MotifsBy::RunId { find_motifs_run_id, select_motifs_run_ids } => {
            debug!(
                "run_plot_motif_slices: MotifsBy::RunId find_run_id={find_motifs_run_id} select_run_ids={select_motifs_run_ids:?}"
            );

            let motifs = store.read().list_motifs(find_motifs_run_id)?;
            let motifs_len = motifs.len();
            let done_cnt = Arc::new(AtomicU32::new(0));

            futures::stream::iter(motifs)
                .map(|motif_id| {
                    let done_cnt = Arc::clone(&done_cnt);
                    let clc_dist = Arc::clone(&clc_dist);
                    let motif_loader_impl = Arc::clone(&motif_loader_impl);
                    let ctx = Arc::clone(&ctx);
                    async move {
                        debug!("run_plot_motif_slices: motif_id={motif_id}");
                        if let Err(e) = plot_motif(&ws, ctx, motif_id, &motif_loader_impl, clc_dist, None).await {
                            error!("motif {motif_id} failed: e={e:?}");
                        };
                        MotifLoader::<RawTensorLoader<RunCtx2>, RunCtx2>::cache().await.clear();

                        let done = done_cnt.fetch_add(1, Relaxed);
                        if done % 32 == 0 {
                            info!("done {done} from  {motifs_len}");
                        }
                    }})
                .buffer_unordered(10) // <- bound parallelism
                .collect::<Vec<()>>()
                .await;

        }

        MotifsBy::GoodMotifs(selection_run_id) => {
            let channels = ws.read().channels();
            debug!("run_plot_motif_slices: MotifsBy::GoodMotifs selection_run_id={selection_run_id} channels={channels}");

            let motifs = store
                .read()
                .list_selected_motifs(selection_run_id, channels)?;

            let motif_ids: Vec<usize> = motifs
                .into_iter()
                .flat_map(|channel| channel.into_iter().map(|(motif_id, _)| motif_id as usize))
                .collect();

            if motif_ids.is_empty() {
                warn!("run_plot_motif_slices: no motifs found for selection_run_id={selection_run_id}");
                return Ok(());
            }

            let motifs_len = motif_ids.len();
            let done_cnt = Arc::new(AtomicU32::new(0));
            let suffix = Some(vec![selection_run_id]);
            futures::stream::iter(motif_ids)
                .map(|motif_id| {
                    let done_cnt = Arc::clone(&done_cnt);
                    let clc_dist = Arc::clone(&clc_dist);
                    let motif_loader_impl = Arc::clone(&motif_loader_impl);
                    let ctx = Arc::clone(&ctx);
                    let ws = Arc::clone(&ws);
                    let suffix = suffix.clone();
                    async move {
                        debug!("run_plot_motif_slices: motif_id={motif_id}");
                        if let Err(e) = plot_motif(&ws, ctx, motif_id, &motif_loader_impl, clc_dist, suffix).await {
                            error!("motif {motif_id} failed: e={e:?}");
                        };
                        MotifLoader::<RawTensorLoader<RunCtx2>, RunCtx2>::cache().await.clear();

                        let done = done_cnt.fetch_add(1, Relaxed);
                        if done % 32 == 0 {
                            info!("done {done} from  {motifs_len}");
                        }
                    }})
                .buffer_unordered(10)
                .collect::<Vec<()>>()
                .await;
        }
    }

    Ok(())
}

async fn plot_motif(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    ctx: Arc<RwLock<RunCtx2>>,
    motif_id: usize,
    motif_loader_impl: &Arc<MotifLoader<RawTensorLoader<RunCtx2>, RunCtx2>>,
    clc_dist: Arc<LoadMotifAndClcDistanceStage<MotifLoader<RawTensorLoader<RunCtx2>, RunCtx2>, AverageEuclideanDistanceStage<RunCtx2>, RunCtx2>>,
    select_motif_run_ids: Option<Vec<RunId>>,
) -> Result<(), ThoosaError> {
    let cache_area = ctx.read().cache_area();
    let num_devices = GlobalCtx::global().device_count();
    let select_motif_run_ids_sfx = select_motif_run_ids.as_ref().map(|v|v.iter().map(|v|format!("_{}", v)).collect::<String>()).unwrap_or("".to_string());
    let motif_id_arc = Arc::new(DevicesWrap::wrap(MotifLoaderInput::new(DataId(motif_id as u32), select_motif_run_ids), 0));

    let motif = Arc::clone(&motif_loader_impl).run_cached(
        Arc::clone(&motif_id_arc),
        Arc::clone(&ctx),
    )
    .await?;

    let dist = clc_dist
        .run_cached(Arc::clone(&motif_id_arc), Arc::clone(&ctx))
        .await?
        .with_data_ref(0, Arc::clone(&ctx), None, |dist_val: &AverageDistanceValue| Ok(dist_val.value()))
        .await?;



    let r = motif.with_data_ref(0, Arc::clone(&ctx), None, |motif: &DataWithExtra<ThoTensor<_, DataId>, MotifMeta>| {
            let run_id = motif.extra().find_motifs_run_id;
            let channel = motif.extra().channel;

            trace!("motif.t={} dist={dist}", motif.data().t_ref());
            let all_slices = motif.data().t_ref().unbind(0);
            let slices_len = all_slices.len();
            let mut color_iter = DistinctColorIter::new();
            plot_series(
                all_slices.into_iter().enumerate().map(|(i, t)| {
                    (format!("{i}"), color_iter.next().unwrap_or(BLACK), t.iter::<f64>().unwrap().enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>())
                }).collect(),
                format!("Motif: {motif_id} run id: {run_id} channel: {channel} number of slices: {slices_len} motif_avg_dist={dist}").as_str(),
                (2000, 1000),
                &ws.read().plots_path().join(format!("{run_id}{select_motif_run_ids_sfx}")).join(format!("motif_{}.png", motif_id))
            )?;


            Ok(true)
        }).await?;
    Ok(())
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

pub async fn run_grid</*ResConf, */RCRV, F, RC, ST, CtxF>(

    input_run_ids: Option<Vec<RunId>>,
    rcrv: RCRV,
    // fixed_config: FC,
    input: Arc<DevicesWrap<ST::In<'static>>>,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    presave: bool,
    f: F,
    ctx_f: CtxF,
) -> Result<Vec<RunId>, ThoosaError>
where
    F: FnOnce(Vec<RC>) -> Arc<ST>,//TODO return Result
    ST: Stage + AsyncStage + Send + Sync, // + From<(Arc<RwLock<Box<dyn Workspace>>>, Arc<RwLock<Box<dyn Store>>>, Arc<RwLock<LoadRunner<Tensor>>>, Vec<RC>, FC)>,
    RC: RunConfig
        + From<std::vec::IntoIter<NamedVal<<RCRV as GridRunConfigSource>::P>>>
        + thoosa_properties_map::ToHashmap,
    RCRV: GridRunConfigSource,
    for<'c> <ST as Stage>::In<'c>: Data + WithId + Send + Sync,
    for<'c> <<ST as Stage>::In<'c> as Data>::M: DevicesWrapCopyMode<<ST as Stage>::In<'c>>,
    <ST as Stage>::Out: Data + WithId + Send + Sync + SavableResult,
    <<ST as Stage>::Out as Data>::M: DevicesWrapCopyMode<<ST as Stage>::Out>,
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

    let mut savable = Arc::into_inner(stage.run(input, ctx_f()).await?).expect("").take_data(0).await?;

    savable.save(store, ws)?;


    Ok(run_ids)
}

pub async fn run_opt</*ResConf, */RCRV, F, RC, ST, CtxF>(

    input_run_ids: Option<Vec<RunId>>,
    rcrv: RCRV,
    // fixed_config: FC,
    input: Arc<DevicesWrap<ST::In<'static>>>,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    presave: bool,
    init_stage: F,
    ctx_f: CtxF,
) -> Result<Vec<RunId>, ThoosaError>
where
    F: AsyncFnOnce(OptRunConfigs<
        Vec<NamedVal<<RCRV as OptRunConfigSource>::P>>,
        Vec<NamedRange<<RCRV as OptRunConfigSource>::P>>,
        Vec<Hyperparameter<<RCRV as OptRunConfigSource>::P>>,
        <RCRV as OptRunConfigSource>::P,
    >) -> Result<Arc<ST>, ThoosaError>,
    ST: Stage + AsyncStage + Send + Sync, // + From<(Arc<RwLock<Box<dyn Workspace>>>, Arc<RwLock<Box<dyn Store>>>, Arc<RwLock<LoadRunner<Tensor>>>, Vec<RC>, FC)>,
    RC: RunConfig
    + From<std::vec::IntoIter<NamedVal<<RCRV as OptRunConfigSource>::P>>>
    + thoosa_properties_map::ToHashmap,
    RCRV: OptRunConfigSource,
    for<'c> <ST as Stage>::In<'c>: Data + WithId + Send + Sync,
    for<'c> <<ST as Stage>::In<'c> as Data>::M: DevicesWrapCopyMode<<ST as Stage>::In<'c>>,
    <ST as Stage>::Out: Data + WithId + Send + Sync + SavableResult,
    <<ST as Stage>::Out as Data>::M: DevicesWrapCopyMode<<ST as Stage>::Out>,
    CtxF: FnOnce() -> Arc<RwLock<<ST as Stage>::Ctx>>,
{
    let opt_run_configs = rcrv.opt_run_configs();
    debug!("run_all_grid_step: grid_run_configs={opt_run_configs:?}");

    // let mut run_configs: Vec<RC> = opt_run_configs.to_run_configs();


    let stage = init_stage(opt_run_configs).await?;

    let mut savable = Arc::into_inner(stage.run(input, ctx_f()).await?).expect("").take_data(0).await?;

    savable.save(store, ws)?;


    Ok(vec![])

}


#[cfg(test)]
mod cli_tests {
    use crate::cli::{
        import_resources, run_all_grid_step, run_opt_step,
        run_plot_motif_slices, Cli,
    };
    use clap::Parser;
    use futures::executor::block_on;
    use log::LevelFilter::{Debug, Error, Trace, Warn};
    use log::{debug, error, info, set_max_level, trace, LevelFilter};
    use parking_lot::lock_api::MutexGuard;
    use parking_lot::{Mutex, RawMutex, RawRwLock, RwLock};
    use plotters::prelude::{IntoLinspace, BLACK};
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
    use tch::display::{set_print_options, PrinterOptions};
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::{nn, Device, IndexOp, Tensor};
    use thoosa_common::find_motifs_run_config::{
        FindMotifsRunConfig, FindMotifsRunConfigRangeOrValue, SelectMotifsRunConfigRangeOrValue,
    };
    use thoosa_common::input_probs::{compare_input_probs, InputProbs};
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::runs::{Empty, Labeled, Logits, NetResources, Simple};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{
        motif_node, CommonTrainingRunConfig, InferArgs, MotifNetParams, OptConfig,
        TrainFinalNetRunConfig, TrainFinalNetRunConfigRangeOrValue, TrainLabelNetsRunConfig,
        TrainLabelNetsRunConfigFields, TrainLabelNetsRunConfigRangeOrValue,
        TrainMotifsNetsRunConfig, TrainMotifsNetsRunConfigRangeOrValue,
    };
    use thoosa_common::utils::{
        from, lin_t, plot_series, reset_print_options, t_0_0, t_0_1, t_1_0, t_1_1,
        three_sigs_tensor, vec_f32_from,
    };
    use thoosa_common::ws::{FsWorkspace, Workspace};
    use thoosa_motifs::distance::MaskMaker::LeQuantile;
    use thoosa_motifs::distance::clc_dist_d2_and_mask;
    use thoosa_motifs::find::{
        FixedFindMotifsConfig, SelectMotifsParams};
    use thoosa_motifs::fragments::{MotifsBy, PlotMotifSlicesArgs};
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::{Resource, ResourceId, ResourceStats, ResourcesConfig};
    use thoosa_shared::run_configs::RunConfig;
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::test_utils::set_panic_hook;
    use thoosa_shared::{
        cdt, cdt_for_dir, random_saw, recurse_files, saw, saw_with_steps, sig_one_imp,
        sig_three_imp, sig_two_imp, ResourceType, TensorIdSliceIdx, ThoosaError,
    };
    use thoosa_store::{MotifTrackerStore, SlicesMatch, SqlMotifTrackerStore, SqlStore, Store};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
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
