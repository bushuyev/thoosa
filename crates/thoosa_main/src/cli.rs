use crate::cli::Commands::{
    FindMotifs, ImportResources, Infer, MotifsToTensor, PlotMotifSlices, RunAll, SayHello,
    SelectMotifs, TrainFinalNet, TrainMotifsNets,
};
use crate::data;
use crate::data::{clc_records, labeled_tensors_by_descr, load_to_logits_tensors};
use crate::tasks::cross_loader::ResourcesConfigInput;
use thoosa_shared::runner::run_ctx_2::RunCtxWithPermits;
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
use std::cmp::max;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::{read_dir, File, OpenOptions};
use std::io::Write;
use std::ops::{Deref, Range};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use std::{fs, process};
use std::marker::PhantomData;
use std::time::Instant;
use plotters::prelude::BLACK;
use tch::Device::{Cpu, Cuda};
use tch::Kind::Float;
use tch::nn::VarStore;
use tch::{Device, Tensor};
use tch::kind::FLOAT_CPU;
use thoosa_common::tho_device::{ThoDevice};
use thoosa_common::dims::{Batch, ChannelsData, ThoBatch, TCD};
use thoosa_common::find_motifs_run_config::{AggrRangeOrValue, ByAvgEucDist, ByAvgEucDistRangeOrValue, AccumulateStatsRangeOrValue, CoherenceRunConfigRangeOrValue, ExtendMotifsConfigRangeOrValue, FindMotifsRunConfig, FindMotifsRunConfigRangeOrValue, InlierRatioRunConfigRangeOrValue, MeanAggrRangeOrValue, MotifCriteriaRangeOrValue, MotifSelector, MotifSelectorRangeOrValue, NormThicknessRunConfigRangeOrValue, PrecalculateStatsRangeOrValue, SelectMotifsRunConfigRangeOrValue, SlicesFilterRangeOrValue, QuantileAggrRangeOrValue, ComparisonMethodRangeOrValue, PearsonCorrConfigRangeOrValue, EucDistConfigRangeOrValue, MinAggrRangeOrValue, ExactStatsRangeOrValue};
use thoosa_common::input_probs::InputProbs;
use thoosa_common::loader::{LoadRunner, SyncWrapper};
use thoosa_common::runs::{Empty, Labeled, Logits, NetResources, SavableResult, Simple};
use thoosa_common::train_nets;
use thoosa_common::train_nets::{load_final_model, opt_train, AttnPool1dArgsRangeOrValue, CommonTrainingRunConfig, CommonTrainingRunConfigRangeOrValue, ConvNetArgsRangeOrValue, ConvNetJointArgsRangeOrValue, FinalNetArgs, FinalNetArgsRangeOrValue, FinalNetParams, InferArgs, LabelNetParams, LabelNetTypeArgsRangeOrValue, LinNetArgs, LinNetArgsRangeOrValue, MinMaxArgsRangeOrValue, MotifCosineNetArgs, MotifCosineNetArgsRangeOrValue, MotifNetParams, OptConfig, PlotVecf32, TensorsDataBatch, ThoModuleWithStore, TrainFinalNetArgs, TrainFinalNetRunConfig, TrainFinalNetRunConfigFields, TrainFinalNetRunConfigRangeOrValue, TrainFinalNetsRunConfigResult, TrainLabelNetsArgs, TrainLabelNetsRunConfig, TrainLabelNetsRunConfigFields, TrainLabelNetsRunConfigRangeOrValue, TrainLabelNetsRunConfigResult, TrainMotifNetResult, TrainMotifsNetsArgs, TrainMotifsNetsResult, TrainMotifsNetsRunConfig, TrainMotifsNetsRunConfigFields, TrainMotifsNetsRunConfigRangeOrValue};
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
use thoosa_shared::traits::{Adjustable, Data, DataId, SimpleId, WithId};
use thoosa_shared::{collect_futures_with_progress, mono_key, parse_resources, simple_id, CopyToDevice, RangeOrValue, StreamBuffering};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_shared::{cdt, cdt_for_dir, ResourceType, ThoError};
use thoosa_store::{SqlStore, Store};
use tokio::runtime::Runtime;
use thoosa_common::tho_tensor::ThoTensor;
use torch_sys::at_grad_set_enabled;
use thoosa_common::copy_ctx::CopyCtx;
use thoosa_common::find_motifs_run_config::InputFilterRangeOrValue::{BandPassFilter, NotchBandFilter};
use thoosa_common::transforms::filters::{BandPassFilterParamsRangeOrValue, NotchBandFilterParamsRangeOrValue};
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, DevicesWrapCopyMode, WithDataRef};
use thoosa_shared::runner::cache_areas::CacheArea;
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::runner::common::runnable::AsyncStage;
use thoosa_shared::runner::global_ctx::GlobalCtx;
use thoosa_motifs::distance::average_euclidean_distance;
use thoosa_shared::colors_iter::DistinctColorIter;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::perf_logging::{format_perf_map, maybe_init_perf_tracing_stdout_simplelog_like, PERF};
use thoosa_shared::runner::oom_handler::{Adjuster, AdjusterError, Adjusters, AsyncOOMHandler, OOMHandler};
use thoosa_shared::test_utils::maybe_init_perf_tracing_stdout;
use crate::tasks::slices_stats::{Aggr, ScalarStatsValue, Metrics, SlicesStats};
use crate::tasks::cross_dims::ChannelsMotifsDists;
use crate::tasks::load_motif_and_clc_dist::{LoadMotifAndClcDistanceInput, LoadMotifAndClcDistanceStage, MotifSlicesSelector};
use crate::tasks::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta};
use crate::tasks::select_motifs::SelectMotifsStage;
use crate::tasks::train_final_net::new_train_final_net_with_ml;
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

    #[arg(long, default_value_t = false)]
    pub term_logger_extra_fields: bool,

    #[arg(long, default_value_t = false)]
    pub write_logger_extra_fields: bool,

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

    #[arg(long, default_value_t = 20_000)]
    pub max_chunk_size: u32, //FixedFindMotifsConfig::max_chunk_size

    #[arg(long, default_value = "XXX")]
    train_resource_group: String,

    #[arg(long, default_value = "XXX")]
    valid_resource_group: String,

    #[arg(long)]
    pub run_config_cnt: String, //TrainLabelNetsRunConfig
}

impl RunAllArgs {

    pub fn find_resource_config(&self) -> Result<Simple, ThoError> {
        Ok(Simple::new(
            parse_resources(&self.find_resources)
                .map_err(|e| ThoError::from_boxed_error("parse find_resources", e))?,
        ))
    }

    pub fn select_resource_config(&self) -> Result<Simple, ThoError> {
        Ok(Simple::new(
            parse_resources(&self.select_resources)
                .map_err(|e| ThoError::from_boxed_error("parse select_resources", e))?,
        ))
    }

    pub fn train_resource_config(&self) -> NetResources<Labeled> {
        NetResources::new(
            Labeled::new(self.train_resource_group.clone()),
            Labeled::new(self.valid_resource_group.clone()),
        )

    }

    pub fn train_final_resource_config(&self) -> NetResources<Logits> {
        NetResources::new(
            Logits::new(self.train_resource_group.clone()),
            Logits::new(self.valid_resource_group.clone()),
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
) -> Result<(), ThoError>
where
    F: Fn(&dyn AsRef<Path>) -> Result<HashMap<String, Vec<(String, f32)>>, ThoError>,
    R: Fn(&dyn AsRef<Path>, &str) -> Result<Tensor, ThoError> + Sync + Send,
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

    let cdt_for_dir = cdt_for_dir();
    let logfile = match File::create(
        ws.read()
            .log_path()
            .join(format!("thoosa_{}.log", cdt_for_dir).to_owned()),
    ) {
        Ok(file) => file,
        Err(_err) => {
            panic!("Can't create logfile in {:?}", cli.ws_path);
        }
    };

    let perf_path =  ws.read()
        .log_path()
        .join(format!("thoosa_perf_{}.log", cdt_for_dir).to_owned());


    let _logging_guards = maybe_init_perf_tracing_stdout_simplelog_like(
        cli.term_logger_level,
        cli.term_logger_extra_fields,
        cli.write_logger_level,
        cli.write_logger_extra_fields,
        logfile,
        "info",
        perf_path,
    )?;

    let store = Arc::new(RwLock::new(SqlStore::from_path(ws.read().store_path())?));

    let devices = list_devices();

    let cpu_num = std::thread::available_parallelism()?;

    //TODO torch.backends.cuda.matmul.allow_tf32 = True
    info!(
        "VERSION: {VERSION} devices={devices:?}, base_path={:?} cpus={cpu_num} tch::get_num_interop_threads={}, tch::get_num_threads={}",
        cli.ws_path,
        tch::get_num_interop_threads(),
        tch::get_num_threads()
    );
    std::io::stdout().flush()?;

    let tensors_path = ws.read().tensors_path().clone();
    let tensors_path_for_loader = tensors_path.clone();

    GlobalCtx::init_devices(2);

    let loader = Arc::new(RawTensorLoader::<RunCtxWithPermits>::new(
        move |key, _| {
            ThoTensor::new(
                SynKey::new([mono_key!(key.id(), format!("resource_id={}", key.id()))]),
                Tensor::load(tensors_path_for_loader.join(format!(
                    "{}_{}.pt",
                    key.import_id(),
                    key.file_stem()
                )))?,
            )
        },
        ThoDevice::from(Device::Cpu),//it's always cpu
    ));

    let mut root_ctx = RunCtxWithPermits::new(1, "root_ctx");

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
                true,
            )?;

            info!("ImportResources done");
        }

        Some(FindMotifs(find_motifs_args)) => {
            debug!("FindMotifs: find_motifs_args={:?} devices={:?}",find_motifs_args, devices);

            let find_motifs_run_config_range_or_value: FindMotifsRunConfigRangeOrValue =
                serde_json5::from_str(&find_motifs_args.run_config_cnt)
                    .expect("FindMotifsRunConfig2");

            debug!("FindMotifs: find_motifs_run_config_range_or_value={find_motifs_run_config_range_or_value:?} find_motifs_args.fixed_config={:?}", find_motifs_args.fixed_config);

            let input_wrapper = Arc::new(DevicesWrap::wrap(
                ResourcesConfigInput::new(find_motifs_args.resource_config()),
                CacheArea::default()
            ));

            let init_ctx = root_ctx.branch( Some("find motifs init ctx"), true, false);
            let init_loader = Arc::clone(&loader);
            let init_ws = Arc::clone(&ws);
            let init_store = Arc::clone(&store);
            let run_ids = run_grid (
                None,
                find_motifs_run_config_range_or_value,
                input_wrapper,
                &ws,
                &store,
                true,
                async move |run_configs| {
                    Arc::new(
                        crate::tasks::find_motifs::FindMotifsStage::new(
                            &init_ws,
                            &init_store,
                            init_loader,
                            run_configs,
                            find_motifs_args.fixed_config.clone(), //FixedFindMotifsConfig::new(None, 10_000, false, None),
                            ThoDevice::from(1),
                            init_ctx
                        )
                        .await.expect("FindMotifsStage::new failed"),
                    )
                },
                || root_ctx.branch( Some("find motifs"), true, false),
            ).await?;

            info!("FindMotifs done, run_ids={run_ids:?}");

        }

        Some(SelectMotifs(select_motifs_args)) => {
            debug!("SelectMotifs2: select_motifs_args.run_config_cnt={:?}",select_motifs_args.run_config_cnt);

            let select_motifs_run_config_range_or_value: SelectMotifsRunConfigRangeOrValue = serde_json5::from_str(&select_motifs_args.run_config_cnt).expect("SelectMotifs2");

            let input_wrapper = Arc::new(DevicesWrap::wrap(
                ResourcesConfigInput::new(select_motifs_args.resource_config()),
                CacheArea::default()
            ));
            //TODO move it into SelectMotifsStage
            let motif_loader = Arc::new(MotifLoader::new(
                Arc::clone(&store),
                Arc::clone(&loader),
                &ws
            ));

            let run_ids = run_grid::</*Labeled,*/ _, _, _, _, _>(
                None,
                select_motifs_run_config_range_or_value,
                input_wrapper,
                &ws,
                &store,
                true,
                |run_configs| async {

                    Arc::new(SelectMotifsStage::<RawTensorLoader<RunCtxWithPermits>, MotifLoader<RawTensorLoader<RunCtxWithPermits>, RunCtxWithPermits>, RunCtxWithPermits>::new(
                        &ws,
                        &store,
                        loader,
                        motif_loader,
                        run_configs,
                        select_motifs_args.find_motifs_run_ids.clone(),
                        100_000, //TODO config
                        #[cfg(test)]
                        None
                    ))
                },
                || root_ctx.branch( Some("select motifs"), true, false),
            ).await?;

            info!("SelectMotifs done, run_ids={run_ids:?}");
        }

        Some(TrainMotifsNets(train_motif_nets_args)) => {
            debug!("TrainMotifsNets2: train_motif_nets_args.run_config_cnt={:?}",train_motif_nets_args.run_config_cnt);

            let train_motif_nets_run_config_range_or_value: TrainMotifsNetsRunConfigRangeOrValue = serde_json5::from_str(&train_motif_nets_args.run_config_cnt).expect("TrainMotifsNets2");

            let input_wrapper = Arc::new(DevicesWrap::wrap(
                TrainMotifNetInput::new(train_motif_nets_args.select_motifs_run_ids.clone(), Arc::new(train_motif_nets_args.fixed_config.clone())),
                CacheArea::default()
            ));

            let run_ids = run_opt::</*Labeled,*/ _, _, TrainMotifsNetsRunConfig, _, _>(
                None,
                train_motif_nets_run_config_range_or_value,
                input_wrapper,
                &ws,
                &store,
                true,
                async |run_configs| {

                    Ok(Arc::new(TrainMotifNets::<_, RunCtxWithPermits>::new(
                        loader,
                        &ws,
                        &store,
                        run_configs,
                    )))
                },
                || root_ctx.branch( Some("train motif nets"), true, false),
            ).await?;

        }

        Some(Commands::TrainLabelNets(train_label_nets_args)) => {
            debug!("TrainLabelNets: train_label_nets_args.run_config_cnt={:?}",train_label_nets_args.run_config_cnt);

            let train_motif_nets_run_config_range_or_value: TrainLabelNetsRunConfigRangeOrValue = serde_json5::from_str(&train_label_nets_args.run_config_cnt).expect("TrainLabelNets");

            let input_wrapper = Arc::new(DevicesWrap::wrap(
                ResourcesConfigInput::new( //TODO the group is needed to load resources not belonging to label being trained
                   train_label_nets_args.resource_config()),
                CacheArea::default()
            ));

            let opt_config = Arc::new(train_label_nets_args.fixed_config.clone());
            let init_ctx = root_ctx.branch(Some("init label nets"), true, false);

            let run_ids = run_opt::</*Labeled,*/ _, _, TrainLabelNetsRunConfig, _, _>(
                None,
                train_motif_nets_run_config_range_or_value,
                input_wrapper,
                &ws,
                &store,
                true,
                async |run_configs| {

                    new_train_label_nets_with_ml(
                        train_label_nets_args.train_motif_nets_run_ids.clone(),
                        loader,
                        run_configs,
                        &ws,
                        &store,
                        init_ctx,
                        opt_config
                    ).await.map(|v|Arc::new(v))
                },
                ||root_ctx.branch( Some("train label nets"), true, false),
            ).await?;



            info!("TrainMotifsNets done");
        }

        Some(TrainFinalNet(train_final_net_args)) => {
            debug!("TrainFinalNet: train_motif_nets_args.run_config_cnt={:?}",train_final_net_args.run_config_cnt);
            let train_final_net_run_config_range_or_value: TrainFinalNetRunConfigRangeOrValue = serde_json5::from_str(&train_final_net_args.run_config_cnt).expect("TrainFinalNet");

            let input_wrapper = Arc::new(DevicesWrap::wrap(
                ResourcesConfigInput::new(
                   train_final_net_args.resource_config()),
                CacheArea::default()
            ));

            let opt_config = Arc::new(train_final_net_args.fixed_config.clone());
            let init_ctx = root_ctx.branch(Some("init final nets"), true, false);
            let run_ids = run_opt::</*Labeled,*/ _, _, TrainFinalNetRunConfig, _, _>(
                None,
                train_final_net_run_config_range_or_value,
                input_wrapper,
                &ws,
                &store,
                true,
                async |run_configs| {

                    new_train_final_net_with_ml(
                        train_final_net_args.train_label_nets_run_ids.clone(),
                        loader,
                        run_configs,
                        &ws,
                        &store,
                        init_ctx,
                        opt_config
                    ).await.map(|v|Arc::new(v))

                },
                || root_ctx.branch( Some("train final nets"), true, false),
            ).await?;


            info!("TrainFinalNet done");
        }

        Some(RunAll(run_all_args)) => {
            debug!("RunAll: run_all_params={:?}", run_all_args);

            let (find_motifs_run_config_range_or_value,
                fixed_find_motifs_config,
                select_motifs_run_config,
                opt_config,
                train_motif_nets_run_config_range_or_value,
                train_label_nets_run_config_range_or_value,
                train_final_net_run_config_range_or_value)
                = run_all_config_0(run_all_args);

            ///////////////////////////////

            run_all(
                &ws,
                &store,
                &loader,
                root_ctx,
                run_all_args,
                find_motifs_run_config_range_or_value,
                fixed_find_motifs_config,
                select_motifs_run_config,
                &opt_config,
                train_motif_nets_run_config_range_or_value,
                train_label_nets_run_config_range_or_value,
                train_final_net_run_config_range_or_value
            ).await?;

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
            run_plot_motif_slices(&ws, &store, &loader, &plot_motif_slices_params, &devices[0]).await?;
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

            // let mut lengths = vec![];
            // for tensorfile in dir {
            //     let path_buf = tensorfile?.path();
            //     let t = Tensor::load(&path_buf);
            //     let t_size = t?.size();
            //     debug!("path_buf={path_buf:?} size={:?}", t_size);
            //     lengths.push(t_size[1]);
            // }
            //
            // info!("tensors: {} max={:?} avg={}",lengths.len(),lengths.iter().max(),lengths.iter().sum::<i64>() / lengths.len() as i64);

            perf_unfold_concat_cuda_vs_cpu().await?;

            info!("Hello");
        }

        None => {
            warn!("No command");
        }
    }

    info!("\n{}", format_perf_map(&PERF));

    GlobalCtx::global().shutdown().await;

    Ok(())
}

#[allow(unreachable_code)]
async fn run_all(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    loader: &Arc<RawTensorLoader<RunCtxWithPermits>>,
    mut root_ctx: RunCtxWithPermits,
    run_all_args: &RunAllArgs,
    find_motifs_run_config_range_or_value: FindMotifsRunConfigRangeOrValue,
    fixed_find_motifs_config: FixedFindMotifsConfig,
    select_motifs_run_config: SelectMotifsRunConfigRangeOrValue,
    opt_config: &Arc<OptConfig>,
    train_motif_nets_run_config_range_or_value: TrainMotifsNetsRunConfigRangeOrValue,
    train_label_nets_run_config_range_or_value: TrainLabelNetsRunConfigRangeOrValue,
    train_final_net_run_config_range_or_value: TrainFinalNetRunConfigRangeOrValue
) -> Result<(), ThoError> {

    //gpt Build the input for FindMotifs: wrap resource config in DevicesWrap and allocate a fresh cache area.
    let find_motifs_input = Arc::new(DevicesWrap::wrap(
        ResourcesConfigInput::new(run_all_args.find_resource_config()?).check(&store)?,
        CacheArea::default()
    ));

    //gpt Build the input for SelectMotifs: separate resource config so selection can use its own dataset.
    let select_motifs_input = Arc::new(DevicesWrap::wrap(
        ResourcesConfigInput::new(run_all_args.select_resource_config()?).check(&store)?,
        CacheArea::default()
    ));

    //gpt Build the input for label training: this is typically a labeled resource set.
    let train_labels_input = Arc::new(DevicesWrap::wrap(
        ResourcesConfigInput::new( //TODO the group is needed to load resources not belonging to label being trained
                                   run_all_args.train_resource_config()).check(&store)?,
        CacheArea::default()
    ));

    //gpt Build the input for final net training: usually a higher-level resource config (e.g., logits inputs).
    let train_final_input = Arc::new(DevicesWrap::wrap(
        ResourcesConfigInput::new(
            run_all_args.train_final_resource_config()).check(&store)?,
        CacheArea::default()
    ));

    //gpt Pre-create context and shared handles for FindMotifs stage construction (kept outside the run_grid closure).
    let init_ctx_find_motifs = root_ctx.branch(Some("init find motifs"), true, false);
    let store_in_find_motifs = Arc::clone(&store);
    let ws_in_find_motifs = Arc::clone(&ws);
    let loader_in_find_motifs = Arc::clone(&loader);

    //gpt Run a grid over FindMotifs configs, constructing a stage per grid point and executing it.
    let find_motifs_run_ids = run_grid::</*Simple,*/ _, _, _, _, _>(
        None,
        find_motifs_run_config_range_or_value,
        find_motifs_input,
        &ws,
        &store,
        true,
        async move |run_configs| {
            Arc::new(
                crate::tasks::find_motifs::FindMotifsStage::new(
                    &ws_in_find_motifs,
                    &store_in_find_motifs,
                    loader_in_find_motifs,
                    run_configs,
                    fixed_find_motifs_config,
                    ThoDevice::from(1),
                    //gpt TODO!! hard-coded device index assumes GPU 1 exists; consider selecting from args or ctx.
                    init_ctx_find_motifs
                ).await.expect("FindMotifsStage::new failed"),
                //gpt TODO!! `expect` will panic and bypass error propagation; consider returning the error instead.
            )
        },
        || root_ctx.branch(Some("find motifs"), true, false),
    ).await?;

    info!("find motifs done, find_motif_run_ids={find_motifs_run_ids:?}");

    //gpt Plot slices for each FindMotifs run (CPU plotting); useful for debugging but potentially expensive.
    for find_motifs_run_id in &find_motifs_run_ids {
        run_plot_motif_slices(&ws, &store, &loader, &PlotMotifSlicesArgs::new(MotifsBy::RunId { find_motifs_run_id: *find_motifs_run_id, select_motifs_run_ids: None }), &Cpu).await?;
    }


    debug!("starting select motifs: debug_semaphore={:?}", GlobalCtx::global().debug_semaphore(1));
    //gpt The select stage depends on the FindMotifs run ids; keep the ids to join results across runs.
    //gpt TODO!! if find_motifs_run_ids is empty, downstream stages will still run; should we short-circuit?

    //gpt Run a grid over SelectMotifs configs; each run consumes FindMotifs outputs and produces selected motifs.
    let select_motifs_run_ids = run_grid::</*Labeled,*/ _, _, _, _, _>(
        Some(find_motifs_run_ids.clone()),
        select_motifs_run_config,
        select_motifs_input,
        &ws,
        &store,
        true,
        async |run_configs| {
            Arc::new(SelectMotifsStage::<RawTensorLoader<RunCtxWithPermits>, MotifLoader<RawTensorLoader<RunCtxWithPermits>, RunCtxWithPermits>, RunCtxWithPermits>::new(
                &ws,
                &store,
                Arc::clone(&loader),
                Arc::new(MotifLoader::new(
                    Arc::clone(&store),
                    Arc::clone(&loader),
                    &ws
                )),
                run_configs,
                find_motifs_run_ids.clone(),
                run_all_args.max_chunk_size as i64,
                #[cfg(test)]
                None
            ))
        },
        || root_ctx.branch(Some("select motifs"), true, false),
    ).await?;

    //gpt Plot "good motifs" from each selection run (CPU plotting; large runs can be slow).
    for select_motifs_run_id in &select_motifs_run_ids {
        run_plot_motif_slices(&ws, &store, &loader, &PlotMotifSlicesArgs::new(MotifsBy::GoodMotifs (*select_motifs_run_id)), &Cpu).await?;
    }

    info!("select motifs done: find_motifs_run_ids={find_motifs_run_ids:?} select_motifs_run_ids={select_motifs_run_ids:?}");

    //gpt Plot motifs with selection overlays per FindMotifs run.
    for find_motifs_run_id in &find_motifs_run_ids {
        run_plot_motif_slices(&ws, &store, &loader, &PlotMotifSlicesArgs::new(MotifsBy::RunId { find_motifs_run_id: *find_motifs_run_id, select_motifs_run_ids: Some(select_motifs_run_ids.clone()) }), &Cpu).await?;
    }
    return Ok(());

    debug!("starting train motif nets: debug_semaphore={:?}", GlobalCtx::global().debug_semaphore(1));

    //gpt Train motif nets needs a custom input wrapper (motif ids + opt config) rather than a resource config.
    let input_wrapper = Arc::new(DevicesWrap::wrap(
        TrainMotifNetInput::new(select_motifs_run_ids.clone(), Arc::clone(&opt_config)),
        CacheArea::default()
    ));

    //gpt Run a grid over motif-net training configs; output is a set of run ids for trained motif nets.
    let train_motif_nets_run_ids = run_opt::</*Labeled,*/ _, _, TrainMotifsNetsRunConfig, _, _>(
        None,
        train_motif_nets_run_config_range_or_value,
        input_wrapper,
        &ws,
        &store,
        true,
        async |run_configs| {
            Ok(Arc::new(TrainMotifNets::<_, RunCtxWithPermits>::new(
                Arc::clone(&loader),
                &ws,
                &store,
                run_configs,
            )))
        },
        || root_ctx.branch(Some("train motif nets"), true, false),
    ).await?;

    info!("train motif nets done: train_motif_nets_run_ids={train_motif_nets_run_ids:?}");

    ///////////////////
    debug!("starting train label nets: debug_semaphore={:?}", GlobalCtx::global().debug_semaphore(1));

    //gpt Pre-create context and shared handles for label-net stage construction.
    let init_ctx_train_labels = root_ctx.branch(Some("init label nets"), true, false);
    let store_in_train_labels = Arc::clone(&store);
    let ws_in_train_labels = Arc::clone(&ws);
    let loader_in_train_labels = Arc::clone(&loader);
    let opt_config_in_train_labels = Arc::clone(&opt_config);
    //gpt Run a grid over label-net training configs; consumes motif nets from previous stage.
    let train_label_nets_run_ids = run_opt::</*Labeled,*/ _, _, TrainLabelNetsRunConfig, _, _>(
        None,
        train_label_nets_run_config_range_or_value,
        train_labels_input,
        &ws,
        &store,
        true,
        async move |run_configs| {
            new_train_label_nets_with_ml(
                train_motif_nets_run_ids,
                loader_in_train_labels,
                run_configs,
                &ws_in_train_labels,
                &store_in_train_labels,
                init_ctx_train_labels,
                opt_config_in_train_labels
            ).await.map(|v| Arc::new(v))
        },
        || root_ctx.branch(Some("train label nets"), true, false),
    ).await?;

    info!("train label nets done: run_ids={train_label_nets_run_ids:?}");

    ////////////////////////
    debug!("starting train final nets: debug_semaphore={:?}", GlobalCtx::global().debug_semaphore(1));


    //gpt Pre-create context and shared handles for final-net stage construction.
    let init_ctx_in_train_final = root_ctx.branch(Some("init final nets"), true, false);
    let store_in_train_final = Arc::clone(&store);
    let ws_in_train_final = Arc::clone(&ws);
    let loader_in_train_final = Arc::clone(&loader);

    //gpt Run a grid over final-net training configs; consumes label-net run ids and produces final models.
    let _run_ids = run_opt::</*Labeled,*/ _, _, TrainFinalNetRunConfig, _, _>(
        None,
        train_final_net_run_config_range_or_value,
        train_final_input,
        &ws,
        &store,
        true,
        async move |run_configs| {
            new_train_final_net_with_ml(
                train_label_nets_run_ids,
                loader_in_train_final,
                run_configs,
                &ws_in_train_final,
                &store_in_train_final,
                init_ctx_in_train_final,
                Arc::clone(opt_config)
            ).await.map(|v| Arc::new(v))
        },
        || root_ctx.branch(Some("train final nets"), true, false),
    ).await?;
    Ok(())
}

fn run_all_config_0(run_all_params: &RunAllArgs) -> (FindMotifsRunConfigRangeOrValue, FixedFindMotifsConfig, SelectMotifsRunConfigRangeOrValue, Arc<OptConfig>, TrainMotifsNetsRunConfigRangeOrValue, TrainLabelNetsRunConfigRangeOrValue, TrainFinalNetRunConfigRangeOrValue) {
    let find_motifs_run_config_range_or_value: FindMotifsRunConfigRangeOrValue = FindMotifsRunConfigRangeOrValue {
        id: None,
        step: RangeOrValue::Value(100), //RangeOrValue::Range { range: Range { start: 50, end: 100 }, step: Some((100, 0)) },
        window: RangeOrValue::Value(1000),
        freq_filter: None,
        slices_filter: SlicesFilterRangeOrValue::AccumulateStats(AccumulateStatsRangeOrValue { aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k:  RangeOrValue::Range{range: 0.05..0.5, step: Some((0.05, 2))}  }) }),
    };
    let fixed_find_motifs_config = FixedFindMotifsConfig::new(run_all_params.max_chunk_size, false);
    //////////////////////select
    let select_motifs_run_config = SelectMotifsRunConfigRangeOrValue {
        id: None,
        // extend_config: Some(ExtendMotifsConfigRangeOrValue {
        //     slices_selector_metric: ComparisonMethodRangeOrValue::EucDist(EucDistConfigRangeOrValue {  }),
        //     slices_selector_aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k: RangeOrValue::Value(1.0) }),
        // }),
        extend_config: None,
        motif_criteria: MotifCriteriaRangeOrValue {
            norm_thickness: Some(NormThicknessRunConfigRangeOrValue { eps: RangeOrValue::Value(1e-8) }),
            inlier_ratioRunConfig: Some(InlierRatioRunConfigRangeOrValue {
                k: RangeOrValue::Value(3.0),
                eps: RangeOrValue::Value(1e-8),
            }),
            coherenceRunConfig: Some(CoherenceRunConfigRangeOrValue {
                k: RangeOrValue::Value(10.0),
                eps: RangeOrValue::Value(1e-8),
            }),
        },
        threshold: RangeOrValue::Value(0.9),
    };
    /////////////////////train_motifs
    let common_training_args = CommonTrainingRunConfigRangeOrValue {
        cycles: RangeOrValue::Value(100),
        batch_size: RangeOrValue::Value(20),
        lr: RangeOrValue::Value(0.0001),
        patience: RangeOrValue::Value(100),
        min_delta: RangeOrValue::Value(1e-4_f32),

    };
    let opt_config = Arc::new(OptConfig::new(200, 10));

    let train_motif_nets_run_config_range_or_value: TrainMotifsNetsRunConfigRangeOrValue = TrainMotifsNetsRunConfigRangeOrValue {
        id: None,
        common_training_args: common_training_args.clone(),
        // net: train_nets::MotifNetTypeArgsRangeOrValue::ConvNet(train_nets::MotifConvNetArgsRangeOrValue {
        //     num_layers: RangeOrValue::Range { range: Range { start: 3, end: 5 }, step: None },
        //     step: RangeOrValue::Value(10),
        //     channels: RangeOrValue::Value(32),
        //     kernel: RangeOrValue::Value(10),
        // }),
        net: train_nets::MotifNetTypeArgsRangeOrValue::CosineNet(MotifCosineNetArgsRangeOrValue{
            step:  RangeOrValue::Value(10),
            channels: RangeOrValue::Range { range: Range { start: 2, end: 32 }, step: None },
            kernel: RangeOrValue::Value(10),
        }),

    };
    //////////////////////train labels
    let train_label_nets_run_config_range_or_value: TrainLabelNetsRunConfigRangeOrValue = TrainLabelNetsRunConfigRangeOrValue {
        id: None,
        common_training_args: common_training_args.clone(),

        net: LabelNetTypeArgsRangeOrValue::ConvNet(ConvNetArgsRangeOrValue {
            num_conv_layers: RangeOrValue::Range { range: Range { start: 3, end: 5 }, step: None },
            hidden_conv_channels: RangeOrValue::Value(64),
            num_head_layers: RangeOrValue::Value(10),
            num_head_features: RangeOrValue::Value(256),
            kernel: RangeOrValue::Value(10),
            // kernel: RangeOrValue::Range { range: Range { start: 10, end: 100 }, step: None },
            joint: ConvNetJointArgsRangeOrValue::AttnPool1d(AttnPool1dArgsRangeOrValue{ _dummy: 0 }),
        }),

    };
    //////////////////////train final
    let train_final_net_run_config_range_or_value: TrainFinalNetRunConfigRangeOrValue = TrainFinalNetRunConfigRangeOrValue {
        id: None,
        common_training_args,
        net: FinalNetArgsRangeOrValue {
            num_layers: RangeOrValue::Range { range: Range { start: 3, end: 5 }, step: None },
        },
    };
    (find_motifs_run_config_range_or_value, fixed_find_motifs_config, select_motifs_run_config, opt_config, train_motif_nets_run_config_range_or_value, train_label_nets_run_config_range_or_value, train_final_net_run_config_range_or_value)
}

fn test_run_all_config_1(run_all_params: &RunAllArgs) -> (FindMotifsRunConfigRangeOrValue, FixedFindMotifsConfig, SelectMotifsRunConfigRangeOrValue, Arc<OptConfig>, TrainMotifsNetsRunConfigRangeOrValue, TrainLabelNetsRunConfigRangeOrValue, TrainFinalNetRunConfigRangeOrValue) {
    let find_motifs_run_config_range_or_value: FindMotifsRunConfigRangeOrValue = FindMotifsRunConfigRangeOrValue {
        id: None,
        step: RangeOrValue::Value(2), //RangeOrValue::Range { range: Range { start: 50, end: 100 }, step: Some((100, 0)) },
        window: RangeOrValue::Value(100),
        freq_filter: None,
        // freq_filter: Some(vec![BandPassFilter(BandPassFilterParamsRangeOrValue {
        //     // low_hz: RangeOrValue::Range { range: Range { start: 0., end: 201. }, step: Some((50., 1)) },
        //     low_hz:  RangeOrValue::Value(0.0) ,
        //     band_hz: RangeOrValue::Value(100.),
        //     kernel_size: RangeOrValue::Value(1000),
        //     stride: RangeOrValue::Value(1),
        //     padding: RangeOrValue::Value(0),
        // })]),
        // slices_filter: SlicesFilterRangeOrValue::AccumulateStats(AccumulateStatsRangeOrValue { aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue{ k: RangeOrValue::Value(0.025) }) }),
        // slices_filter: SlicesFilterRangeOrValue::AccumulateStats(AccumulateStatsRangeOrValue { aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue{ k: RangeOrValue::Range{range: 0.005..0.1, step: Some((0.005, 3))} }) }),
        // slices_filter: SlicesFilterRangeOrValue::AccumulateStats(AccumulateStatsRangeOrValue { aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue{ q: RangeOrValue::Value(0.01) })}),
        // slices_filter: SlicesFilterRangeOrValue::AccumulateStats(AccumulateStatsRangeOrValue { aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue{ q: RangeOrValue::Range{range: 0.005..0.1, step: Some((0.005, 3))}})}),
        // slices_filter: SlicesFilterRangeOrValue::PrecalculateStats(PrecalculateStatsRangeOrValue { aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue{ q: RangeOrValue::Range{range: 0.05..1.0, step: Some((0.05, 2))} }), slices: RangeOrValue::Value(1000) }),
        // slices_filter: SlicesFilterRangeOrValue::PrecalculateStats(PrecalculateStatsRangeOrValue { aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue{ q: RangeOrValue::Value(0.25) }), slices: RangeOrValue::Value(100) }),
        // slices_filter: SlicesFilterRangeOrValue::PrecalculateStats(PrecalculateStatsRangeOrValue{aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k: RangeOrValue::Range{range: 0.005..0.05, step: Some((0.005, 4))} }), slices: RangeOrValue::Value(100)}),
        // slices_filter: SlicesFilterRangeOrValue::PrecalculateStats(PrecalculateStatsRangeOrValue{aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k:  RangeOrValue::Value(0.025) }), slices: RangeOrValue::Value(1000)}),
        // slices_filter: SlicesFilterRangeOrValue::PrecalculateStats(PrecalculateStatsRangeOrValue{aggr: AggrRangeOrValue::Min(MinAggrRangeOrValue { k:  RangeOrValue::Range{range: 0.095..1.2, step: Some((0.005, 3))} }), slices: RangeOrValue::Value(1000)}),
        slices_filter: SlicesFilterRangeOrValue::ExactStats(ExactStatsRangeOrValue{ euc_dist:  RangeOrValue::Value(2.0)  })
    };
    let fixed_find_motifs_config = FixedFindMotifsConfig::new(run_all_params.max_chunk_size, false);
    //////////////////////select
    let select_motifs_run_config = SelectMotifsRunConfigRangeOrValue {
        id: None,
        // extend_config: Some(ExtendMotifsConfigRangeOrValue {
        //     // slices_selector_metric: ComparisonMethodRangeOrValue::EucDist(ComparisonMethodConfigRangeOrValue {  }),
        //     slices_selector_metric: ComparisonMethodRangeOrValue::PearsonCorr(PearsonCorrConfigRangeOrValue {  }),
        //     slices_selector_aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k: RangeOrValue::Value(1.0) }),
        //     // slices_selector_aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k: RangeOrValue::Range{range: 0.8..1.2, step: Some((0.1, 1))} }),
        // }),
        extend_config:None,
        motif_criteria: MotifCriteriaRangeOrValue {
            norm_thickness: Some(NormThicknessRunConfigRangeOrValue { eps: RangeOrValue::Value(1e-8) }),
            inlier_ratioRunConfig: Some(InlierRatioRunConfigRangeOrValue {
                k: RangeOrValue::Value(3.0),
                eps: RangeOrValue::Value(1e-8),
            }),
            coherenceRunConfig: Some(CoherenceRunConfigRangeOrValue {
                k: RangeOrValue::Value(10.0),
                eps: RangeOrValue::Value(1e-8),
            }),
        },
        threshold: RangeOrValue::Value(0.91),
        // threshold: RangeOrValue::Range{range: 0.4..1.1, step: Some((0.1, 1))},
    };
    /////////////////////train_motifs
    let common_training_args = CommonTrainingRunConfigRangeOrValue {
        cycles: RangeOrValue::Value(100),
        batch_size: RangeOrValue::Value(10),
        // lr: RangeOrValue::Value(0.0001),
        lr: RangeOrValue::Range{ range: Range { start: 0.00001, end: 0.001 }, step: None },
        patience: RangeOrValue::Value(10),
        min_delta: RangeOrValue::Value(1e-4_f32),
    };
    let opt_config = Arc::new(OptConfig::new(100, 10));

    let train_motif_nets_run_config_range_or_value: TrainMotifsNetsRunConfigRangeOrValue = TrainMotifsNetsRunConfigRangeOrValue {
        id: None,
        common_training_args: common_training_args.clone(),
        // net: train_nets::MotifNetTypeArgsRangeOrValue::ConvNet(train_nets::MotifConvNetArgsRangeOrValue { _dummy: 0 }),
        net: train_nets::MotifNetTypeArgsRangeOrValue::CosineNet(MotifCosineNetArgsRangeOrValue{
            step: RangeOrValue::Value(10),
            channels: RangeOrValue::Value(8),
            kernel: RangeOrValue::Range { range: Range { start: 3, end: 10 }, step: None },
        }),
        // num_layers: RangeOrValue::Range { range: Range { start: 2, end: 8 }, step: None },

    };
    //////////////////////train labels
    let train_label_nets_run_config_range_or_value: TrainLabelNetsRunConfigRangeOrValue = TrainLabelNetsRunConfigRangeOrValue {
        id: None,
        common_training_args: common_training_args.clone(),

        net: LabelNetTypeArgsRangeOrValue::ConvNet(ConvNetArgsRangeOrValue {
            // num_conv_layers: RangeOrValue::Range { range: Range { start: 3, end: 5 }, step: None },
            num_conv_layers: RangeOrValue::Value(8),
            hidden_conv_channels: RangeOrValue::Value(64),
            num_head_layers: RangeOrValue::Value(10),
            num_head_features: RangeOrValue::Value(256),
            kernel: RangeOrValue::Value(10),
            // kernel: RangeOrValue::Range { range: Range { start: 10, end: 100 }, step: None },
            // joint: ConvNetJointArgsRangeOrValue::MinMax(MinMaxArgsRangeOrValue{ _dummy: 0 }),
            joint: ConvNetJointArgsRangeOrValue::AttnPool1d(AttnPool1dArgsRangeOrValue{_dummy:0}),
        }),
    };
    //////////////////////train final
    let train_final_net_run_config_range_or_value: TrainFinalNetRunConfigRangeOrValue = TrainFinalNetRunConfigRangeOrValue {
        id: None,
        common_training_args,
        net: FinalNetArgsRangeOrValue {
            num_layers: RangeOrValue::Range { range: Range { start: 3, end: 5 }, step: None },
        },
    };
    (find_motifs_run_config_range_or_value, fixed_find_motifs_config, select_motifs_run_config, opt_config, train_motif_nets_run_config_range_or_value, train_label_nets_run_config_range_or_value, train_final_net_run_config_range_or_value)
}



fn run_all_config(run_all_params: &RunAllArgs) -> (FindMotifsRunConfigRangeOrValue, FixedFindMotifsConfig, SelectMotifsRunConfigRangeOrValue, Arc<OptConfig>, TrainMotifsNetsRunConfigRangeOrValue, TrainLabelNetsRunConfigRangeOrValue, TrainFinalNetRunConfigRangeOrValue) {
    let find_motifs_run_config_range_or_value: FindMotifsRunConfigRangeOrValue = FindMotifsRunConfigRangeOrValue {
        id: None,
        step: RangeOrValue::Value(100), //RangeOrValue::Range { range: Range { start: 50, end: 100 }, step: Some((100, 0)) },
        // window: RangeOrValue::Value(1000),
        window: RangeOrValue::Range { range: Range { start: 1000, end: 5001 }, step: Some((1000, 1)) },
        freq_filter: Some(vec![
            NotchBandFilter(NotchBandFilterParamsRangeOrValue {
                low_hz: RangeOrValue::Value(58.0),
                high_hz: RangeOrValue::Value(62.0),
            }),
            BandPassFilter(BandPassFilterParamsRangeOrValue {
                low_hz: RangeOrValue::Range { range: Range { start: 0., end: 201. }, step: Some((50., 1)) },
                band_hz: RangeOrValue::Value(100.),
                kernel_size: RangeOrValue::Value(1000),
                stride: RangeOrValue::Value(1),
                padding: RangeOrValue::Value(0),
            }),
        ]),
        // freq_filter: None,
        slices_filter: SlicesFilterRangeOrValue::AccumulateStats(AccumulateStatsRangeOrValue { aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k: RangeOrValue::Value(0.02) }) }),
    };
    let fixed_find_motifs_config = FixedFindMotifsConfig::new(run_all_params.max_chunk_size, false);
    //////////////////////select
    let select_motifs_run_config = SelectMotifsRunConfigRangeOrValue {
        id: None,
        extend_config: Some(ExtendMotifsConfigRangeOrValue {
            slices_selector_metric: ComparisonMethodRangeOrValue::EucDist(EucDistConfigRangeOrValue {  }),
            slices_selector_aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k: RangeOrValue::Value(1.0) }),
        }),
        motif_criteria: MotifCriteriaRangeOrValue {
            norm_thickness: Some(NormThicknessRunConfigRangeOrValue { eps: RangeOrValue::Value(1e-8) }),
            inlier_ratioRunConfig: None,
            coherenceRunConfig: None,
        },
        threshold: RangeOrValue::Value(0.5),
    };
    /////////////////////train_motifs
    let common_training_args = CommonTrainingRunConfigRangeOrValue {
        cycles: RangeOrValue::Value(100),
        batch_size: RangeOrValue::Value(20),
        // batch_size: RangeOrValue::Range { range: Range { start: 5, end: 25 }, step: Some((5, 1)) },
        lr: RangeOrValue::Range { range: Range { start: 0.00001, end: 0.0001 }, step: None },
        // lr: RangeOrValue::Value(0.0001),
        patience: RangeOrValue::Value(100),
        min_delta: RangeOrValue::Value(1e-4_f32),

    };
    let opt_config = Arc::new(OptConfig::new(200, 10));

    let train_motif_nets_run_config_range_or_value: TrainMotifsNetsRunConfigRangeOrValue = TrainMotifsNetsRunConfigRangeOrValue {
        id: None,
        common_training_args: common_training_args.clone(),
        net: train_nets::MotifNetTypeArgsRangeOrValue::ConvNet(train_nets::MotifConvNetArgsRangeOrValue {
            num_layers: RangeOrValue::Range { range: Range { start: 3, end: 10 }, step: None },
            step: RangeOrValue::Value(10),
            channels: RangeOrValue::Value(32),
            kernel: RangeOrValue::Value(10),
        }),

    };
    //////////////////////train labels
    let train_label_nets_run_config_range_or_value: TrainLabelNetsRunConfigRangeOrValue = TrainLabelNetsRunConfigRangeOrValue {
        id: None,
        common_training_args: common_training_args.clone(),
        // net: LabelNetTypeArgsRangeOrValue::LinNet(LinNetArgsRangeOrValue{
        //     num_layers: RangeOrValue::Range { range: Range { start: 3, end: 10 }, step: None },
        //     slices_len: RangeOrValue::Value(1000)
        // }),
        net: LabelNetTypeArgsRangeOrValue::ConvNet(ConvNetArgsRangeOrValue {
            num_conv_layers: RangeOrValue::Range { range: Range { start: 3, end: 12 }, step: None },
            hidden_conv_channels: RangeOrValue::Value(64),
            num_head_layers: RangeOrValue::Value(10),
            // kernel: RangeOrValue::Value(10),
            num_head_features: RangeOrValue::Value(256),
            kernel: RangeOrValue::Range { range: Range { start: 10, end: 100 }, step: None },
            joint: ConvNetJointArgsRangeOrValue::MinMax(MinMaxArgsRangeOrValue{ _dummy: 0 }),
        }),
    };
    //////////////////////train final
    let train_final_net_run_config_range_or_value: TrainFinalNetRunConfigRangeOrValue = TrainFinalNetRunConfigRangeOrValue {
        id: None,
        common_training_args,
        net: FinalNetArgsRangeOrValue {
            num_layers: RangeOrValue::Range { range: Range { start: 3, end: 20 }, step: None },
        },
    };
    (find_motifs_run_config_range_or_value, fixed_find_motifs_config, select_motifs_run_config, opt_config, train_motif_nets_run_config_range_or_value, train_label_nets_run_config_range_or_value, train_final_net_run_config_range_or_value)
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

pub fn import_resources<F, R>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    labels: &[&str],
    labeled_resources: F,
    resource_to_tensor: R, //None for tests - do not import parquet files
    insert_label: bool,
    save_pure_sets: bool,
) -> Result<Vec<ResourceId>, ThoError>
where
    F: Fn() -> Result<HashMap<String, Vec<(String, f32)>>, ThoError>,
    R: Fn(String) -> Result<Tensor, ThoError> + Sync + Send,
{
    import_resources_with_pure_sets(ws, store, labels, labeled_resources, resource_to_tensor, insert_label, Some(&[(3, true), (10, true), (50, true), (100, true), (150, true), (10000, false)]))
}

 pub fn import_resources_with_pure_sets<F, R>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    labels: &[&str],
    labeled_resources: F,
    resource_to_tensor: R, //None for tests - do not import parquet files
    insert_label: bool,
    pure_sets_lens: Option<&[(usize, bool)]>,
) -> Result<Vec<ResourceId>, ThoError>
where
    F: Fn() -> Result<HashMap<String, Vec<(String, f32)>>, ThoError>,
    R: Fn(String) -> Result<Tensor, ThoError> + Sync + Send,
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

    if let Some(pure_sets_lens) = pure_sets_lens {
        for pure_sets_len in pure_sets_lens {
            store_guard.save_pure_sets(pure_sets_len.0, pure_sets_len.1, &t)?;//
        }
    }


    // store_guard.save_pure_balanced_sets("A10", Some(10), 0.6, 0.2, &t)?;
    // store_guard.save_pure_balanced_sets("B10", Some(10), 0.6, 0.2, &t)?;
    // store_guard.save_pure_balanced_sets("C10", Some(10), 0.6, 0.2, &t)?;
    // store_guard.save_pure_balanced_sets("A100", Some(100), 0.6, 0.2, &t)?;
    // store_guard.save_pure_balanced_sets("B100", Some(100), 0.6, 0.2, &t)?;
    // store_guard.save_pure_balanced_sets("C100", Some(100), 0.6, 0.2, &t)?;
    // store_guard.save_pure_balanced_sets("All", None,  0.6, 0.2, &t)?;

    store_guard.save_groups_by_label_cnt(100, &t)?;
    t.commit()?;

    trace!("import_resources: resource_ids={resource_ids:?}, failed_tensors={failed_tensors:?}");

    Ok(resource_ids) //for tests
}

async fn run_plot_motif_slices(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    loader: &Arc<RawTensorLoader<RunCtxWithPermits>>,
    plot_motif_slices_params: &PlotMotifSlicesArgs,
    device: &Device,
) -> Result<(), ThoError> {
    let mut ctx = RunCtxWithPermits::new(1, "run_plot_motif_slices");

    let motif_loader_impl = Arc::new(MotifLoader::<RawTensorLoader<RunCtxWithPermits>, RunCtxWithPermits>::new(
        Arc::clone(store),
        Arc::clone(loader),
        ws
    ));
    let clc_dist = Arc::new(LoadMotifAndClcDistanceStage::new(
        Arc::clone(&motif_loader_impl), // as Arc<dyn MotifLoaderStage<Output = <SlicesStats as Stage>::In<'static>>>,
        Arc::new(SlicesStats::<RunCtxWithPermits>::new()),
    ));

    match plot_motif_slices_params.motifs_by() {
        MotifsBy::MotifId(motif_id) => {
            debug!("run_plot_motif_slices: MotifsBy::MotifId run_id={motif_id}");

            plot_motif(&ws, ctx, motif_id, &motif_loader_impl, clc_dist, None).await?;

        }

        MotifsBy::RunId { find_motifs_run_id, select_motifs_run_ids } => {
            debug!("run_plot_motif_slices: MotifsBy::RunId find_run_id={find_motifs_run_id} select_run_ids={select_motifs_run_ids:?}");

            let motifs = store.read().list_motifs(find_motifs_run_id)?;
            let motifs_len = motifs.len();
            let done_cnt = Arc::new(AtomicU32::new(0));

            let futures = motifs
                .into_iter()
                .map(|motif_id| {
                    let done_cnt = Arc::clone(&done_cnt);
                    let clc_dist = Arc::clone(&clc_dist);
                    let motif_loader_impl = Arc::clone(&motif_loader_impl);
                    let ctx = ctx.branch(Some(&format!("motif_{motif_id}")), true, false);
                    async move {
                        debug!("run_plot_motif_slices: motif_id={motif_id}");

                        //gpt TODO!! select_motifs_run_ids is not forwarded to plot_motif here; plots ignore selection filtering even when provided.
                        if let Err(e) = plot_motif(&ws, ctx, motif_id as usize, &motif_loader_impl, clc_dist, None).await {
                            error!("motif {motif_id} failed: e={e:?}");
                        };
                        MotifLoader::<RawTensorLoader<RunCtxWithPermits>, RunCtxWithPermits>::cache().await.clear();

                        let done = done_cnt.fetch_add(1, Relaxed);
                        if done % 32 == 0 {
                            info!("done {done} from  {motifs_len}");
                        }

                        Ok::<_, ThoError>(())
                    }
                })
                .collect::<Vec<_>>();

            let _ = collect_futures_with_progress(
                futures,
                crate::buffer_limits::CLI_RUN_PLOT_MOTIF_SLICES_BUFFER_UNORDERED_LIMIT,
                StreamBuffering::Unordered,
                "run_plot_motif_slices",
            )
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
            let futures = motif_ids
                .into_iter()
                .map(|motif_id| {
                    let done_cnt = Arc::clone(&done_cnt);
                    let clc_dist = Arc::clone(&clc_dist);
                    let motif_loader_impl = Arc::clone(&motif_loader_impl);

                    let ws = Arc::clone(&ws);
                    let suffix = suffix.clone();
                    let ctx = ctx.branch(Some(&format!("motif_{motif_id}")), true, false);
                    async move {
                        debug!("run_plot_motif_slices: motif_id={motif_id}");
                        if let Err(e) = plot_motif(&ws, ctx, motif_id, &motif_loader_impl, clc_dist, suffix).await {
                            error!("motif {motif_id} failed: e={e:?}");
                        };
                        MotifLoader::<RawTensorLoader<RunCtxWithPermits>, RunCtxWithPermits>::cache().await.clear();

                        let done = done_cnt.fetch_add(1, Relaxed);
                        if done % 32 == 0 {
                            info!("done {done} from  {motifs_len}");
                        }

                        Ok::<_, ThoError>(())
                    }
                })
                .collect::<Vec<_>>();

            let _ = collect_futures_with_progress(
                futures,
                crate::buffer_limits::CLI_RUN_PLOT_MOTIF_SLICES_BUFFER_UNORDERED_LIMIT,
                StreamBuffering::Unordered,
                "run_plot_motif_slices",
            )
            .await;
        }
    }

    Ok(())
}

async fn plot_motif(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    mut ctx: RunCtxWithPermits,
    motif_id: usize,
    motif_loader_impl: &Arc<MotifLoader<RawTensorLoader<RunCtxWithPermits>, RunCtxWithPermits>>,
    clc_dist: Arc<LoadMotifAndClcDistanceStage<MotifLoader<RawTensorLoader<RunCtxWithPermits>, RunCtxWithPermits>, SlicesStats<RunCtxWithPermits>, RunCtxWithPermits>>,
    select_motif_run_ids: Option<Vec<RunId>>,
) -> Result<(), ThoError> {
    let select_motif_run_ids_sfx = select_motif_run_ids.as_ref().map(|v|v.iter().map(|v|format!("_{}", v)).collect::<String>()).unwrap_or("".to_string());
    let motif_loader_input = MotifLoaderInput::new(DataId(motif_id as u32), select_motif_run_ids);
    let motif_id_arc = Arc::new(DevicesWrap::wrap(motif_loader_input.clone(), CacheArea::default()));
    let clc_input = Arc::new(DevicesWrap::wrap(LoadMotifAndClcDistanceInput::new(motif_loader_input, Metrics::EucDist, Aggr::Mean(1.0)), CacheArea::default()));

    let motif = Arc::clone(&motif_loader_impl).run_cached(
        Arc::clone(&motif_id_arc),
        ctx.branch(Some("plot_motif_loader"), false, false),
    )
    .await?;

    let mut ctx_for_dist = ctx.branch(Some("plot_motif_dist_value"), false, false);
    let dist = clc_dist
        .run/*_cached*/(clc_input, ctx.branch(Some("plot_motif_clc_dist"), false, false))
        .await?
        .with_data_ref(0, &mut ctx_for_dist, None, |dist_val: &ScalarStatsValue, _ctx| Ok(dist_val.value()))
        .await?;


    let r = motif.with_data_ref(0, &mut ctx.branch(Some("plot_motif_data"), false, false), None, |motif: &DataWithExtra<ThoTensor<_, DataId>, MotifMeta>, _ctx| {
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
) -> Result<Vec<RunId>, ThoError>
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
    ) -> Result<Option<Vec<Box<dyn SavableResult>>>, ThoError>, //TODO remove Option & Vec
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
) -> Result<Vec<RunId>, ThoError>
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
    ) -> Result<Option<Vec<Box<dyn SavableResult>>>, ThoError>,
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

pub async fn run_grid<RCRV, F, RC, ST, CtxF>(

    input_run_ids: Option<Vec<RunId>>,
    rcrv: RCRV,
    input: Arc<DevicesWrap<ST::In<'static>>>,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<RwLock<Box<dyn Store>>>,
    presave: bool,
    f: F,
    ctx_f: CtxF,
) -> Result<Vec<RunId>, ThoError>
where
    F: AsyncFnOnce(Vec<RC>) -> Arc<ST>,//TODO return Result
    ST: Stage + AsyncStage + Send + Sync, // + From<(Arc<RwLock<Box<dyn Workspace>>>, Arc<RwLock<Box<dyn Store>>>, Arc<RwLock<LoadRunner<Tensor>>>, Vec<RC>, FC)>,
    RC: RunConfig
        + From<std::vec::IntoIter<NamedVal<<RCRV as GridRunConfigSource>::P>>>
        + thoosa_properties_map::ToHashmap,
    RCRV: GridRunConfigSource,
    for<'c> <ST as Stage>::In<'c>: Data + WithId + Send + Sync,
    for<'c> <<ST as Stage>::In<'c> as Data>::M: DevicesWrapCopyMode<<ST as Stage>::In<'c>>,
    <ST as Stage>::Out: Data + WithId + Send + Sync + SavableResult,
    <<ST as Stage>::Out as Data>::M: DevicesWrapCopyMode<<ST as Stage>::Out>,
    CtxF: FnOnce() -> <ST as Stage>::Ctx,
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
    //gpt TODO!! run_ids only reflect pre-saved run_config ids; if presave is false this stays empty even though the stage output is saved below.

    let stage = f(run_configs).await;

    //gpt TODO!! Arc::into_inner will panic if the output Arc has multiple owners; consider a fallible path or cloning the inner data.
    let mut savable = Arc::into_inner(stage.run(input, ctx_f()).await?).expect("").take_data(0).await?;

    //gpt TODO!! savable.save returns a run_id, but it's ignored here; downstream uses presaved ids only.
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
) -> Result<Vec<RunId>, ThoError>
where
    F: AsyncFnOnce(OptRunConfigs<
        Vec<NamedVal<<RCRV as OptRunConfigSource>::P>>,
        Vec<NamedRange<<RCRV as OptRunConfigSource>::P>>,
        Vec<Hyperparameter<<RCRV as OptRunConfigSource>::P>>,
        <RCRV as OptRunConfigSource>::P,
    >) -> Result<Arc<ST>, ThoError>,
    ST: Stage + AsyncStage + Send + Sync, // + From<(Arc<RwLock<Box<dyn Workspace>>>, Arc<RwLock<Box<dyn Store>>>, Arc<RwLock<LoadRunner<Tensor>>>, Vec<RC>, FC)>,
    RC: RunConfig
    + From<std::vec::IntoIter<NamedVal<<RCRV as OptRunConfigSource>::P>>>
    + thoosa_properties_map::ToHashmap,
    RCRV: OptRunConfigSource,
    for<'c> <ST as Stage>::In<'c>: Data + WithId + Send + Sync,
    for<'c> <<ST as Stage>::In<'c> as Data>::M: DevicesWrapCopyMode<<ST as Stage>::In<'c>>,
    <ST as Stage>::Out: Data + WithId + Send + Sync + SavableResult,
    <<ST as Stage>::Out as Data>::M: DevicesWrapCopyMode<<ST as Stage>::Out>,
    CtxF: FnOnce() -> <ST as Stage>::Ctx,
{
    let opt_run_configs = rcrv.opt_run_configs();
    debug!("run_all_grid_step: grid_run_configs={opt_run_configs:?}");

    // let mut run_configs: Vec<RC> = opt_run_configs.to_run_configs();
    //gpt TODO!! presave flag is currently unused in run_opt; run_ids always come from the stage output.
    let stage = init_stage(opt_run_configs).await?;

    //gpt TODO!! Arc::into_inner will panic if the output Arc has multiple owners; consider a fallible path or cloning the inner data.
    let mut savable = Arc::into_inner(stage.run(input, ctx_f()).await?).expect("").take_data(0).await?;

    let run_id = savable.save(store, ws)?;


    Ok(vec![run_id])

}

async fn perf_unfold_concat_cuda_vs_cpu() -> Result<(), ThoError>{
    async fn load_tensors(dim_1: &i64, num_tensors: usize, cuda_first:bool) -> Result<u128, ThoError> {
        let mut tensors = Vec::with_capacity(num_tensors);

        let ctx = CopyCtx::instance(Cuda(0));

        let t_0 = Instant::now();
        for i in 0..num_tensors {


            let t = Tensor::stack(&[
                make_t(&dim_1, &(i as i64), 0.1, ctx, cuda_first).await?,
                make_t(&dim_1, &(i as i64), 0.2, ctx, cuda_first).await?,
                make_t(&dim_1, &(i as i64), 0.3, ctx, cuda_first).await?,
                make_t(&dim_1, &(i as i64), 0.4, ctx, cuda_first).await?,
                make_t(&dim_1, &(i as i64), 0.5, ctx, cuda_first).await?,
            ], 0)
                .unfold(1, 1000, 100);

            tensors.push(t);
        }

        let batch = Tensor::f_concat(tensors.as_slice(), 1)?;
        let batch= if !cuda_first {
            ctx.copy(&batch, 1, Float).await?
        } else {
            batch
        };

        assert_eq!(batch.device(), Cuda(0));

        Ok(t_0.elapsed().as_millis())
    }

    async fn make_t(dim_1: &i64, i: &i64, x: f64, copy_ctx: &CopyCtx, cuda_first:bool) -> Result<Tensor, ThoError> {
        let t = x + Tensor::arange_start(i * dim_1, (i + 1) * dim_1, FLOAT_CPU);

        if cuda_first {
            copy_ctx.copy(&t, 1, Float).await
        } else {
            Ok(t)
        }
    }

    // log::set_max_level(Info);
    // GlobalCtx::init_devices(2);

    let num_tensors = 1000;

    let mut cuda_first_time = 0;
    let mut cpu_first_time = 0;

    let rounds = 20;

    for r in 0..rounds*2 {
        if r % 2 == 0 {
            cuda_first_time += load_tensors(&100_000, num_tensors, true).await?;
        } else {
            cpu_first_time += load_tensors(&100_000, num_tensors, false).await?;
        }

    }


    println!("cuda_first_time in {} cpu_first_time={}", cuda_first_time/rounds, cpu_first_time/rounds);

    Ok(())
}



#[cfg(test)]
mod cli_tests {
    use crate::cli::{import_resources, import_resources_with_pure_sets, run_all, run_all_config_0, test_run_all_config_1, run_all_grid_step, run_grid, run_opt_step, run_plot_motif_slices, Cli, RunAllArgs};
    use clap::Parser;
    use futures::executor::block_on;
    use log::LevelFilter::{Debug, Error, Info, Trace, Warn};
    use log::{debug, error, info, set_max_level, trace, LevelFilter};
    use parking_lot::lock_api::MutexGuard;
    use parking_lot::{Mutex, RawMutex, RawRwLock, RwLock};
    use plotters::prelude::{IntoLinspace, BLACK};
    use rand::Rng;
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    use rayon::prelude::IntoParallelIterator;
    use std::cmp::{max, min};
    use std::collections::{HashMap, HashSet};
    use std::io::Read;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::vec::IntoIter;
    use std::{fs, ptr};
    use std::time::Duration;
    use tch::Device::{Cpu, Cuda};
    use tch::display::{set_print_options, PrinterOptions};
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::{nn, Device, IndexOp, Tensor};
    use tch::Kind::Float;
    use thoosa_common::find_motifs_run_config::{
        FindMotifsRunConfig, FindMotifsRunConfigRangeOrValue, SelectMotifsRunConfigRangeOrValue,
    };
    use thoosa_common::input_probs::{compare_input_probs, InputProbs};
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::my_tch::ThoModule;
    use thoosa_common::runs::{Empty, Labeled, Logits, NetResources, Simple};
    use thoosa_common::tch_test_utils;
    use thoosa_common::tch_test_utils::{resource_tensor_two_easy_motifs, Interval};
    use thoosa_common::tho_device::ThoDevice;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{lin_motif_node, CommonTrainingRunConfig, InferArgs, MotifCosineNetParams, MotifNetParams, MotifNetTypeParams, OptConfig, TrainFinalNetRunConfig, TrainFinalNetRunConfigRangeOrValue, TrainLabelNetsRunConfig, TrainLabelNetsRunConfigFields, TrainLabelNetsRunConfigRangeOrValue, TrainMotifsNetsRunConfig, TrainMotifsNetsRunConfigRangeOrValue};
    use thoosa_common::utils::{from, lin_t, local_max_1d, plot_series, plot_t, reset_print_options, t_0_0, t_0_1, t_1_0, t_1_1, three_sigs_tensor, to_vec_f32, white_tophat1d};
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
    use thoosa_shared::test_utils::{set_panic_hook, MotifKind};
    use thoosa_shared::{cdt, cdt_for_dir, mono_key, panic_to_err_async, random_saw, recurse_files, saw, saw_with_steps, sig_one_imp, sig_three_imp, sig_two_imp, ResourceType, TensorIdSliceIdx, ThoError};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::perf_logging::{format_perf_map, PERF};
    use thoosa_shared::runner::cache_areas::CacheArea;
    use thoosa_shared::runner::common::{PermitsCtx, RunCtx};
    use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
    use thoosa_shared::runner::common::runnable::AsyncStage;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_shared::runner::run_ctx_2::RunCtxWithPermits;
    use thoosa_shared::traits::{Data, DataId};
    use thoosa_store::{MotifTrackerStore, SlicesMatch, SqlMotifTrackerStore, SqlStore, Store};
    use crate::tasks::slices_stats::{Aggr, ScalarStatsValue, Metrics, SlicesStats};
    use crate::tasks::cross_dims::MotifSlices;
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::find_motifs::FindMotifsStage;
    use crate::tasks::load_motif_and_clc_dist::{LoadMotifAndClcDistanceInput, LoadMotifAndClcDistanceStage};
    use crate::tasks::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta};
    use crate::tasks::tensor_loader::RawTensorLoader;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    fn init_test_store(
        mut store: &Arc<RwLock<Box<dyn Store>>>,
        tensors_per_label: i32,
    ) -> Result<Vec<ResourceId>, ThoError> {
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


    #[test]
    fn test_import_resources() -> Result<(), ThoError> {
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
            false,
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

    #[test]
    fn test_resource_tensor_two_easy_motifs() -> Result<(), ThoError>{

        let (t, intervals, _) = resource_tensor_two_easy_motifs(
            1000,
            1,
            100,
            2,
            MotifKind::Pulse,
            2,
            MotifKind::Triangle,
            100,
            0.05,
            3.0,
            None
        )?;

        plot_t(&t, "two_easy_motifs", true)?;

        Ok(())
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 32)]
    async fn heavy_test_run_all() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        // log::set_max_level(log::LevelFilter::Trace);
        let global_ctx = GlobalCtx::init_devices(2);

        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_motifs: test_dir={test_dir:?}");

        fs::create_dir_all(&test_dir)?;
        let base_path = test_dir.to_string_lossy().to_string();


        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            test_dir.clone(),
            1,
            0,
            100,
        )?));
        let ws_rc = Arc::clone(&ws);
        let max_y = 30.0;


        // let x = lin_t(n);
        // plot_series(
        //     vec![
        //         (
        //             format!("Sig0"),
        //             plotters::prelude::RED,
        //             vec_f32_from(&x)
        //                 .into_iter()
        //                 .zip(sig_0_v.clone())
        //                 .collect::<Vec<(f32, f32)>>(),
        //         ),
        //         (
        //             format!("Sig1"),
        //             plotters::prelude::GREEN,
        //             vec_f32_from(&x)
        //                 .into_iter()
        //                 .zip(sig_1_v.clone())
        //                 .collect::<Vec<(f32, f32)>>(),
        //         ),
        //         (
        //             format!("Sig2"),
        //             plotters::prelude::BLACK,
        //             vec_f32_from(&x)
        //                 .into_iter()
        //                 .zip(sig_2_v.clone())
        //                 .collect::<Vec<(f32, f32)>>(),
        //         ),
        //     ],
        //     "test_find_motifs_euc",
        //     (2000, 1000),
        //     &format!("/tmp/plots/0_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str(),
        // )?;
        let mut import_hash = HashMap::new();
        for i in 0..30 {
            let (t, _intervals) = tensor_0()?;

            let stem = format!("t_0_{i}");

            plot_t(&t, &stem, true);
            t.save(ws.read().tensors_path().join(&format!("{stem}.pt")))?;
            t.save(ws.read().tensors_path().join(&stem))?;

            import_hash.insert(stem, vec![("label_0".to_string(), 1.0)]);

            let (t, _intervals) = tensor_1()?;

            let stem = format!("t_1_{i}");

            plot_t(&t, &stem, true);

            t.save(ws.read().tensors_path().join(&format!("{stem}.pt")))?;
            import_hash.insert(stem, vec![("label_1".to_string(), 1.0)]);
        }


        let channels = 3;
        let tensors_path = ws.read().tensors_path().clone();
        let tensors_path_for_import = ws.read().tensors_path().clone();

        let mut store = Arc::new(RwLock::new(SqlStore::from_path(ws.read().store_path())?));

        debug!("test_run_all: import_hash={import_hash:?}");

        let imported_ids = import_resources_with_pure_sets(
            &ws,
            &store,
            &["label_0", "label_1"],
            move || {
                Ok(import_hash.clone())
            },
            // None::<Box<dyn Fn(String) -> Result<Tensor, ThoosaError> + Sync + Send>>,
            |key| {
                Tensor::load(tensors_path_for_import.join(format!("{}.pt", key)))
                    .map_err(ThoError::from)
            },
            true,
            Some(&[(20, true), (10, true)])
        )?;

        let resources = store.read().list_resources_by_ids(&imported_ids)?;

        let loader = Arc::new(RawTensorLoader::<RunCtxWithPermits>::new(
            move |key, _| {
                let tensor = Tensor::load(tensors_path.join(format!("{}.pt", key.file_stem())))?;
                debug!("test_motifs: key={key:?} loaded tensor={tensor:?}");
                Ok(ThoTensor::new(SynKey::new([mono_key!(key.id(), format!("resource_id={}", key.id()))]), tensor)?)
            },
            ThoDevice::from(Device::Cpu),
        ));

        let mut ctx = RunCtxWithPermits::new(1, "test_run_all");
        ctx.add_permit(global_ctx.acquire_many(0, 1, Arc::new(SynKey::new([mono_key!("test", "test")]))).await?);


        let run_all_args = RunAllArgs{
            find_resources: "group:pure group 10".to_string(),
            select_resources: "group:pure group 20".to_string(),
            max_chunk_size: 10,
            train_resource_group: "pure group 20".to_string(),
            valid_resource_group: "pure group 10".to_string(),
            run_config_cnt: "".to_string(),
        };

        let (find_motifs_run_config_range_or_value,
            fixed_find_motifs_config,
            select_motifs_run_config,
            opt_config,
            train_motif_nets_run_config_range_or_value,
            train_label_nets_run_config_range_or_value,
            train_final_net_run_config_range_or_value)
            = test_run_all_config_1(&run_all_args);

        match panic_to_err_async!({run_all(
            &ws,
            &store,
            &loader,
            ctx,
            &run_all_args,
            find_motifs_run_config_range_or_value,
            fixed_find_motifs_config,
            select_motifs_run_config,
            &opt_config,
            train_motif_nets_run_config_range_or_value,
            train_label_nets_run_config_range_or_value,
            train_final_net_run_config_range_or_value
        ).await}){
            Ok(_) => {
                debug!("ok");
            }
            Err(e) => {
                debug!("e={e:?}");
            }
        };

        global_ctx.shutdown().await;


        Ok(())
    }

    fn tensor_1() -> Result<(Tensor, Vec<Vec<Vec<Interval>>>), ThoError> {
        // tch_test_utils::resource_tensor(1000usize, &[
        //     vec![(3, 100, MotifKind::Sin)],
        // ], 100, 10.)
        let r = resource_tensor_two_easy_motifs(
            1000,
            1,
            100,
            2,
            MotifKind::Sin,
            2,
            MotifKind::Triangle,
            100,
            0.05,
            3.0,
            None
        )?;
        Ok((r.0, r.1))
    }

    fn tensor_0() -> Result<(Tensor, Vec<Vec<Vec<Interval>>>), ThoError> {
        // tch_test_utils::resource_tensor(
        //     1000usize,
        //     &[
        //         vec![(3, 100, MotifKind::Pulse)
        //         ],
        //     ], 100, 10.0)

        let r = resource_tensor_two_easy_motifs(
            1000,
            1,
            100,
            2,
            MotifKind::Pulse,
            2,
            MotifKind::Triangle,
            100,
            0.05,
            3.0,
            None
        )?;
        Ok((r.0, r.1))
    }
    /*
        #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="test_tracing"))]
        async fn test_tracing(){
            tokio::time::sleep(Duration::from_millis(100)).await
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_log_trace() {
            log::set_max_level(Trace);

            trace!("some trace");
            debug!("some debug");
            info!("some info");

            test_tracing().await;
            info!("test_train_label_net_3x3: PERF=\n{}", format_perf_map(&PERF));
            println!("test_log_trace");
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_log_debug() {
            log::set_max_level(Debug);//should be default

            trace!("some trace");
            debug!("some debug");
            info!("some info");

            test_tracing().await;
            info!("test_train_label_net_3x3: PERF=\n{}", format_perf_map(&PERF));
            println!("test_log_debug");
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn test_log_info() {
            log::set_max_level(Info);

            trace!("some trace");
            debug!("some debug");
            info!("some info");

            test_tracing().await;
            info!("test_train_label_net_3x3: PERF=\n{}", format_perf_map(&PERF));
            println!("test_log_info");
        }
         */

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_apply_moitf_net() -> Result<(), ThoError>{
        GlobalCtx::init_devices(2);
        let ws = Arc::new(RwLock::new(FsWorkspace::new("/home/bu/tmp/ws_test_hbac", 20, 684400, 200)?));
        let tensors_path = ws.read().tensors_path().clone();
        let store = Arc::new(RwLock::new(SqlStore::from_path(ws.read().store_path())?));
        let mut ctx = RunCtxWithPermits::new(0, "run_plot_motif_slices");

        let loader = Arc::new(RawTensorLoader::<RunCtxWithPermits>::new(
            move |key, _| {
                ThoTensor::new(
                    SynKey::new([mono_key!(key.id(), format!("resource_id={}", key.id()))]),
                    Tensor::load(tensors_path.join(format!(
                        "{}_{}.pt",
                        key.import_id(),
                        key.file_stem()
                    )))?,
                )
            },
            ThoDevice::from(Device::Cpu),//it's always cpu
        ));


        let motif_loader_impl = Arc::new(MotifLoader::<RawTensorLoader<RunCtxWithPermits>, RunCtxWithPermits>::new(
            Arc::clone(&store),
            Arc::clone(&loader),
            &ws
        ));


        let clc_dist = Arc::new(LoadMotifAndClcDistanceStage::new(
            Arc::clone(&motif_loader_impl), // as Arc<dyn MotifLoaderStage<Output = <SlicesStats as Stage>::In<'static>>>,
            Arc::new(SlicesStats::<RunCtxWithPermits>::new()),
        ));

        let resource = "1315636310"; //3879494837";

        let t = Tensor::load(format!("/home/bu/tmp/ws_test_hbac/tensors/1_{resource}.pt"))?;

        let channel = 13;
        let motif_id= 22127;
        let run_id = 7777772;

        let motif = Arc::clone(&motif_loader_impl).run(
            Arc::new(DevicesWrap::wrap(MotifLoaderInput::new(DataId(motif_id as u32), None), CacheArea::default())),
            ctx.branch(Some("plot_motif_loader"), false, false),
        ).await?;

        let motif_t = Arc::into_inner(motif).expect("no motif").take_data(0).await?;


        let channel_t = t.i((channel, ..));
        plot_t(&channel_t.view((1, -1)), &format!("resource_1_{resource}_{channel}"), false)?;
        debug!("motif_t.data.size={:?}", motif_t.data().size());
        plot_t(&motif_t.data().as_ref().view((1, -1)), &format!("motif_{motif_id}"), false)?;



        /*let t_1 = white_tophat1d(&channel_16_t, 101)?;
        debug!("t_1={:?}", t_1.size());
        let mask = local_max_1d(&t_1.view((1,1,-1)), 101)?;
        let t_1_masked = t_1 * mask;
        plot_t(&t_1_masked.squeeze_dim(1), "1_2722809500_t_1_masked")?;*/



        let (params, module)  = ws.read().load_motif_net(&PathBuf::from(format!("{channel}/{motif_id}_{run_id}/")), &Cpu)?;
        match params.net_type {
            MotifNetTypeParams::LinNetParams(_) => {
                debug!("LinNetParams");
                todo!()
            }
            MotifNetTypeParams::ConvNetParams(_) => {
                debug!("ConvNetParams");
                todo!()
            }
            MotifNetTypeParams::CosineNetParams(MotifCosineNetParams{ num_features, stride, channels, kernel }) => {
                debug!("CosineNetParams: num_features={num_features} stride={stride} channels={channels} kernel={kernel}");

                let out = module.tho_forward(&channel_t.unfold(0, num_features as i64, stride as i64).into(), None, false)?;

                debug!("out.size={:?}", out.size());

                // plot_t(&out.t.softmax(0, Float).view((1, -1)), &format!("{motif_id}_applied_to_{resource}"))?;
                plot_t(&out.t.view((1, -1)), &format!("{motif_id}_applied_to_{resource}"), false)?;
            }
        }



        Ok(())
    }

}
