use crate::cli::Commands::{
     ConvertPrecision, Export, FindMotifs, ImportResources, Infer, MotifsToTensor, PlotData, PurgeModels, RunAll,
    SayHello, SelectMotifs, SeqScan, TrainFinalNet, TrainMotifsNets,
};
use crate::data;
use crate::data::{clc_records, labeled_tensors_by_descr, load_to_logits_tensors};
use crate::progress_socket::CollectEtaProgressSocketServer;
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::convert_precision::{ConvertPrecisionArgs, ConvertPrecisionInput, ConvertPrecisionStage};
use crate::tasks::export::{ExportArgs, ExportInput, Export as ExportStage};
use crate::tasks::purge_models::{PurgeModelsArgs, PurgeModelsInput, PurgeModelsStage};
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use crate::tasks::tensor_loader::RawTensorLoader;
use crate::version::VERSION;
use async_trait::async_trait;
use thoosa_common::runner::common::runnable::Stage;
use clap::{Args, Parser, Subcommand};
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
use std::sync::{Arc, OnceLock};
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use std::{fs, process};
use std::marker::PhantomData;
use std::time::Instant;
use mockall_double::double;
use plotters::prelude::BLACK;
use plotters::style::ShapeStyle;
use serde::Serialize;
use tch::Device::{Cpu, Cuda};
use tch::Kind::Float;
use tch::nn::VarStore;
use tch::{Device, IndexOp, Tensor};
use tch::kind::FLOAT_CPU;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::dims::{Batch, ChannelsData, ThoBatch, TCD};
use crate::tasks::find_motifs_run_config::{AccumulateStats, AccumulateStatsRangeOrValue, AggrRangeOrValue, CoherenceRunConfigRangeOrValue, CrossLoaderRunConfigRangeOrValue, ExactStatsRangeOrValue, ExtendMotifsConfig, ExtendMotifsConfigRangeOrValue, FinderKindRangeOrValue, FindMotifsRunConfig, FindMotifsRunConfigFields, FindMotifsRunConfigRangeOrValue, InlierRatioRunConfigRangeOrValue, InterestingAmp01RunConfig, InterestingAmp01RunConfigRangeOrValue, InterestingTurningPointsRatioRunConfig, InterestingTurningPointsRatioRunConfigRangeOrValue, MaxAggr, MaxSlicesAbsRangeOrValue, MaxSlicesRangeOrValue, MeanAggrRangeOrValue, MetricsRangeOrValue, MinAggrRangeOrValue, MotifCriterionConstraintRangeOrValue, NormThicknessRunConfigRangeOrValue, PeriodicityAcfRunConfigRangeOrValue, PrecalculateStatsRangeOrValue, QuantileAggrRangeOrValue, SelectMotifsRunConfigFields, SelectMotifsRunConfigRangeOrValue, SlicesFilter, SlicesFilterRangeOrValue};
use thoosa_common::input_probs::InputProbs;
use thoosa_common::runs::{Empty, Labeled, Logits, NetResources, SavableResult, Simple};

use thoosa_common::seq_scan_run_config::SeqScanRunConfigRangeOrValue;
use thoosa_common::train_nets;
use thoosa_common::train_nets::{load_final_model, opt_train, AdamWRangeOrValue, AttnPool1dArgsRangeOrValue, BinaryHeadLse, BinaryHeadLseRangeOrValue, BinaryHeadPoolArgs, BinaryHeadPoolArgsRangeOrValue, CommonTrainingRunConfig, CommonTrainingRunConfigRangeOrValue, ConvNetArgsRangeOrValue, ConvNetJointArgsRangeOrValue, FinalNetArgs, FinalNetArgsRangeOrValue, FinalNetParams, InferArgs, LabelNetParams, LabelNetTypeArgsRangeOrValue, LinNetArgs, LinNetArgsRangeOrValue, MinMaxArgsRangeOrValue, MotifCosineNetArgs, MotifCosineNetArgsRangeOrValue, MotifNetParams, MotifsBy as TrainMotifsBy, OptConfig, OptimizersRangeOrValue, PlotVecf32, SelfAttentionNetArgs, SelfAttentionNetArgsRangeOrValue, ThoModuleWithStore, TrainFinalNetArgs, TrainFinalNetRunConfig, TrainFinalNetRunConfigFields, TrainFinalNetRunConfigRangeOrValue, TrainFinalNetsRunConfigResult, TrainLabelNetsArgs, TrainLabelNetRunConfig, TrainLabelNetRunConfigFields, TrainLabelNetRunConfigRangeOrValue, TrainLabelNetRunConfigResult, TrainMotifNetResult, TrainMotifsNetsArgs, TrainMotifsNetsResult, TrainMotifNetRunConfig, TrainMotifNetRunConfigFields, TrainMotifNetRunConfigRangeOrValue};
use thoosa_common::utils::{from, kick_register_cuda_hooks, lin_t, list_devices, load_run_config, plot_series, plot_series_with_sidebar_values, plot_spectrogram, reset_print_options, DummyListener, TrainListener};
use thoosa_common::ws::{FsWorkspace, Workspace};
use thoosa_common::my_tch::ThoModule;
use thoosa_motifs::find;
use thoosa_motifs::find::{FindMotifsArgs, FixedFindMotifsConfig, SelectMotifsArgs};
use thoosa_motifs::fragments::{
    DataBy, MotifsToTensorArgs, PlotDataArgs, Resolution};
use thoosa_properties_map::ToHashmap;
use thoosa_run_config_args::{GridRunConfigSource, OptRunConfigSource};
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};
use thoosa_shared::resource::{Resource, ResourceId, ResourceStats, ResourceStore, ResourcesConfig};
use thoosa_shared::run_configs::{run_id, OptRunConfigs, RunConfig, RunConfigId, RUN_ID};
use thoosa_shared::traits::{Adjustable, Data, DataId, SimpleId, WithId};
use thoosa_shared::{collect_futures_with_progress, mono_key, panic_to_err_async, parse_resources, simple_id, CopyToDevice, RangeOrValue, StreamBuffering};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_shared::{cdt, cdt_for_dir, ResourceType, ThoError};
use thoosa_store::{SqlStore, Store};
use tokio::runtime::{Handle, Runtime};
use tokio::{select, sync::watch, time::{self, Duration}};
use tokio::time::MissedTickBehavior;
use thoosa_common::tho_tensor::ThoTensor;
use torch_sys::at_grad_set_enabled;
use thoosa_common::copy_ctx::CopyCtx;

use thoosa_common::transforms::filters::{AverageFilterParams, AverageFilterParamsRangeOrValue, BandPassFilterParamsRangeOrValue, InputFilter, MedianFilterParamsRangeOrValue, NotchBandFilterParamsRangeOrValue, Padding, PaddingRangeOrValue, SavGolFilterParams, SavGolFilterParamsRangeOrValue, ZeroPaddingRangeOrValue};
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap, DevicesWrapCopyMode, WithDataRef};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::common::{OutId, RunCtx};
use thoosa_common::runner::common::runnable::AsyncStage;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_motifs::distance::{average_euclidean_distance, EucDistConfig, EucDistConfigRangeOrValue, SimilarityMetric, SimilarityMetricRangeOrValue};
use thoosa_shared::colors_iter::DistinctColorIter;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::perf_logging::{format_perf_map, init_tracing, install_panic_hook, SyncMakeWriter, PERF};
use thoosa_common::runner::oom_handler::{Adjuster, AdjusterError, Adjusters, OOMHandler};
use thoosa_common::transforms::filters::InputFilterRangeOrValue::AverageFilter;
use thoosa_shared::collect_futures_eta::{collect_eta_record_completion, CollectEtaInvocationGuard};
use thoosa_shared::test_utils::maybe_init_perf_tracing_stdout;
use thoosa_store::rusqlite::Transaction;
use crate::tasks::slices_stats::{Aggr, Metrics};
use crate::tasks::cross_dims::{ChannelsData2, ChannelsMotifsDists};
use crate::tasks::apply_motif_nets::apply_motif_net_forward;
use crate::tasks::find_and_select_motifs::{FindAndSelectMotifsInput, FindAndSelectMotifsRunConfig, FindAndSelectMotifsRunConfigFields, FindAndSelectMotifsRunConfigRangeOrValue, FindAndSelectMotifsStage};
use crate::tasks::load_motif_net::MotifNetMeta;
use crate::tasks::bucketing::{BucketRunConfig, BucketRunConfigRangeOrValue, BucketingStage, BucketingInput};
use crate::tasks::merge_selected_motifs::{MergeSelectedMotifs, MergeSelectedMotifsArgs, MergeSelectedMotifsInput};
use crate::tasks::motif_stats::{MotifStats, MotifStatsInput, MotifStatsOutput};
use crate::tasks::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta};
use crate::tasks::select_buckets::{SelectBucketsRunConfigRangeOrValue};
use crate::tasks::select_motifs::{SelectMotifsInput, SelectMotifsStage};
use crate::tasks::seq_scan::SeqScanStage;
use crate::tasks::train_final_net::new_train_final_net;
use crate::tasks::train_label_nets::{new_train_label_nets, TrainLabelNets, TrainLabelNetsInput};
use crate::tasks::train_motif_nets::{TrainMotifNetInput, TrainMotifNets};
use crate::tasks::transforms::freq_filter::FreqFilter;

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

    #[arg(long, default_value_t = false)]
    pub write_perf_log_file: bool,

    #[arg(long)]
    pub loader_threads: Option<usize>,

    #[arg(long, default_value_t = 100)]
    pub futures_channel_bound: usize,

    #[arg(long)]
    pub run_id:Option<u32>,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Args, Debug, Clone)]
pub struct RunAllArgs {
    #[arg(long)]
    pub find_resources: String,

    #[arg(long)]
    pub select_resources: String,

    #[arg(long, default_value_t = 20_000)]
    pub max_chunk_size: u32,

    #[arg(long, default_value = "XXX")]
    pub train_resource_group: String,

    #[arg(long, default_value = "XXX")]
    pub valid_resource_group: String,

    #[arg(long, default_value_t = 10)]
    pub train_motifs_epoch_size: u32,

    #[arg(long, default_value = "0.3")]
    pub max_motif_net_loss: f32, //filters motif nets for label training TODO move to train labels args

    #[arg(long)]
    pub selected_motif_nets_limit: Option<usize>,//filters motif nets for label training TODO move to train labels args

    #[arg(long, default_value = "100")]
    pub motif_slices_step: u32,

    #[arg(long, default_value = "false")]
    pub with_plots: bool,

    #[arg(long, value_enum, default_value_t = Resolution::Normal)]
    pub resolution: Resolution,

    #[arg(long)]
    pub labels: Option<String>,
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

    fn parse_labels_csv(raw: &str) -> Vec<String> {
        raw.split(',').map(|label| label.trim().to_string()).filter(|label| !label.is_empty()).collect()
    }

    pub fn labels_filter(&self) -> Option<Vec<String>> {
        self.labels.as_deref().map(Self::parse_labels_csv).filter(|labels| !labels.is_empty())
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

    #[arg(long)]
    pub freq_filters: Vec<InputFilter>,
}

#[derive(Debug, Args, Clone)]
pub struct SayHelloArgs {}

#[derive(Debug, Args, Clone)]
pub struct SeqScanArgs {
    #[arg(long, value_parser=parse_resources)]
    pub resources: ResourceType,

    #[arg(long)]
    pub run_config_cnt: String,

    #[arg(long, default_value_t = 10_000)]
    pub max_chunk_size: usize, //max number of entries in tensor batches
}

impl SeqScanArgs {
    pub fn resource_config(&self) -> Simple {
        Simple::new(self.resources.clone())
    }
}

#[derive(Debug, Args, Clone)]
pub struct BucketArgs {
    #[arg(long, value_parser=parse_resources)]
    pub resources: ResourceType,

    #[arg(long)]
    pub run_config_cnt: String,
}

impl BucketArgs {
    pub fn resource_config(&self) -> Simple {
        Simple::new(self.resources.clone())
    }
}

#[derive(Debug, Args, Clone)]
pub struct BucketOptArgs {
    #[arg(long, value_parser=parse_resources)]
    pub resources: ResourceType,

    #[arg(long)]
    pub run_config_cnt: String,

    #[command(flatten)]
    pub opt_config: OptConfig,
}

impl BucketOptArgs {
    pub fn resource_config(&self) -> Simple {
        Simple::new(self.resources.clone())
    }
}

#[derive(Debug, Subcommand, Clone)]
pub enum Commands {
    ImportResources(ImportResourcesArgs),
    FindMotifs(FindMotifsArgs),
    SeqScan(SeqScanArgs),
    PlotData(PlotDataArgs),
    MotifsToTensor(MotifsToTensorArgs),
    SelectMotifs(SelectMotifsArgs),
    MergeSelectedMotifs(MergeSelectedMotifsArgs),
    TrainMotifsNets(TrainMotifsNetsArgs),
    TrainLabelNets(TrainLabelNetsArgs),
    TrainFinalNet(TrainFinalNetArgs),
    ConvertPrecision(ConvertPrecisionArgs),
    Export(ExportArgs),
    PurgeModels(PurgeModelsArgs),
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

pub async fn run_cli<F, R, A, S, FI>(
    labeled_resources: F,
    resource_to_tensor: R,
    init_data:S,
    run_all:A,
    labels: &[&str],
    channels: u32,
    max_input: u32,
    sample_rate: u32,
    infer: Option<FI>,
) -> Result<(), ThoError>
where
    F: Fn(&dyn AsRef<Path>) -> Result<HashMap<String, Vec<(String, f32)>>, ThoError>,
    R: Fn(&dyn AsRef<Path>, &str) -> Result<Tensor, ThoError> + Sync + Send,
    A: AsyncFn(
        &Arc<RwLock<Box<dyn Workspace>>>,
        &Arc<Box<dyn Store>>,
        &Arc<RawTensorLoader<RunCtxImpl>>,
        RunCtxImpl,
        &RunAllArgs,
    ) -> Result<(), ThoError>,
    S: FnOnce(&Transaction) -> Result<(), ThoError>,
    FI: AsyncFnOnce(
        &InferArgs,
        &Arc<RwLock<Box<dyn Workspace>>>,
        &Arc<Box<dyn Store>>,
    ) -> Result<(), ThoError>
{
    // console_subscriber::init();
    if log::log_enabled!(log::Level::Trace) {
        reset_print_options();
    }
    //set_oom_score_adj(-1000);//TODO conf, needs sudo

    //gpt Navigation note: project binaries such as thoosa_bird_clef provide the closures above.
    //gpt This shared function handles CLI parsing, workspace/store setup, tensor cache loading, and command dispatch.
    unsafe {
        at_grad_set_enabled(0);
    };

    let cli = Cli::parse();


    let ws = match &cli.command {
        Some(Infer(InferArgs{ models_dir: Some(models_dir), .. })) => {
            info!("FsWorkspace: models_dir={models_dir:?}");
            Arc::new(RwLock::new(FsWorkspace::new_with_models_dir(
                cli.ws_path.clone(),
                channels,
                max_input,
                sample_rate,
                models_dir
            )?))
        }
        _=> {
            Arc::new(RwLock::new(FsWorkspace::new(
                cli.ws_path.clone(),
                channels,
                max_input,
                sample_rate,
            )?))
        }
    };

    let cdt_for_dir = cdt_for_dir();
    let log_filename = ws.read()
        .log_path()
        .join(format!("thoosa_{}.log", cdt_for_dir).to_owned());


    let perf_path = if cli.write_perf_log_file {
        Some(
            ws.read()
                .log_path()
                .join(format!("thoosa_perf_{}.log", cdt_for_dir).to_owned()),
        )
    } else {
        None
    };

    install_panic_hook(SyncMakeWriter::new(OpenOptions::new()
        .create(true)
        .append(true)
        .open( ws.read()
            .log_path()
            .join(format!("thoosa_panic_{}.log", cdt_for_dir)))?));


    let _logging_guards = init_tracing(
        cli.term_logger_level,
        cli.term_logger_extra_fields,
        cli.write_logger_level,
        cli.write_logger_extra_fields,
        log_filename,
        "info",
        perf_path,
    )?;

    let _collect_eta_progress_server = match CollectEtaProgressSocketServer::start_default() {
        Ok(server) => Some(server),
        Err(err) => {
            warn!("run_cli: failed to start collect ETA progress socket server: {err}");
            None
        }
    };

    /*
    let store = match &cli.command {
        Some(Infer(InferArgs{ sql_store_file: Some(sql_store_file), .. })) => {
            info!("reqd-only SqlStore: sql_store_file={sql_store_file:?}");
            Arc::new(SqlStore::from_path_readonly(sql_store_file).await?)
        }
        _=> {
            Arc::new(SqlStore::from_path(ws.read().store_path()).await?)
        }
    };
     */
    let store = Arc::new(SqlStore::from_path(ws.read().store_path()).await?);

    RUN_ID.set(match cli.run_id{
        None => {

            if let Some(Infer(_) | Export(_)) = &cli.command {
                panic!("No run_id")
            }
            store.save_run(format!("{cli:?}")).await?
        }
        Some(run_id) => run_id
    }).expect("can't set run_id");

    let devices = list_devices();

    let cpu_num = std::thread::available_parallelism()?;

    //TODO torch.backends.cuda.matmul.allow_tf32 = True
    info!(
        "VERSION: {VERSION} devices={devices:?}, base_path={:?} cpus={cpu_num} tch::get_num_interop_threads={}, tch::get_num_threads={} run_id={}",
        cli.ws_path,
        tch::get_num_interop_threads(),
        tch::get_num_threads(),
        run_id()
    );
    std::io::stdout().flush()?;

    let tensors_path = ws.read().tensors_path().clone();
    let tensors_path_for_loader = tensors_path.clone();

    // Creates CPU plus GPU device wrappers for the async runner. Big tensors crossing awaits should remain in run_cached/TmpCached so OOM handling can dump them to CPU.
    if tch::Cuda::is_available() {
        kick_register_cuda_hooks();
        thoosa_common::runner::global_ctx::GlobalCtx::init_devices(2);
    } else {
        thoosa_common::runner::global_ctx::GlobalCtx::init_devices(1);
    }

    let (stop_cpu_purge_tx, mut rx) = watch::channel(false);
    let metrics = Handle::current().metrics();
    let workers = metrics.num_workers();
    let mut prev_poll = vec![0u64; workers];
    let mut prev_park = vec![0u64; workers];
    let mut prev_busy = vec![std::time::Duration::ZERO; workers];

    let purge_cpu_worker = tokio::spawn(async move {
        let mut ticker = time::interval(Duration::from_secs(60));
        ticker.tick().await; //
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            select! {
                _ = ticker.tick() => {

                    #[cfg(tho_debug)]
                    {
                        info!(
                            "[tokio] alive_tasks={} global_q={} blocking_q={}",
                            metrics.num_alive_tasks(),
                            metrics.global_queue_depth(),
                            metrics.blocking_queue_depth(),
                        );
                    }

                    for w in 0..workers {
                           let poll = metrics.worker_poll_count(w);
                            let park = metrics.worker_park_count(w);
                            let busy = metrics.worker_total_busy_duration(w);

                            debug!(
                                "  w{w}: poll+{} park+{} busy+{:?}",
                                poll - prev_poll[w],
                                park - prev_park[w],
                                busy - prev_busy[w],
                            );

                            prev_poll[w] = poll;
                            prev_park[w] = park;
                            prev_busy[w] = busy;
                    }
                    debug!("run_cli: purge_cpu tick");
                    GlobalCtx::global().purge_cpu(10).await;
                    debug!("run_cli: purge_cpu done");
                }
                _ = rx.changed() => {
                    if *rx.borrow() {
                        debug!("stopping cpu purger");
                        break;
                    }
                }
            }
        }
    });


    let loader = Arc::new(RawTensorLoader::<RunCtxImpl>::new(
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

    let mut root_ctx = RunCtxImpl::new(1, "root_ctx");

    // ImportResources is the only command that calls the project OGG decoder directly;
    // later commands load the persisted tensor files through RawTensorLoader.
    match &cli.command {
        Some(ImportResources(import_resources_args)) => {
            debug!("ImportResources: import_resources_params={:?}",import_resources_args);

            let freq_filter = if import_resources_args.freq_filters.is_empty() {
                None
            } else {
                Some(Arc::new(FreqFilter::<RunCtxImpl>::new(
                    import_resources_args.freq_filters.as_slice(),
                    &ws,
                    0,
                )?))
            };
            import_resources(
                &ws,
                &store,
                labels,
                || labeled_resources(&import_resources_args.source_path),
                |key: String| {
                    resource_to_tensor(&import_resources_args.source_path, key.as_str())
                },
                freq_filter,
                true,
                init_data
            ).await?;

            info!("ImportResources done");
        }

        Some(FindMotifs(find_motifs_args)) => {
            debug!("FindMotifs: find_motifs_args={:?} devices={:?}",find_motifs_args, devices);

            let find_motifs_run_config_range_or_value: FindMotifsRunConfigRangeOrValue =
                serde_json5::from_str(&find_motifs_args.run_config_cnt)
                    .expect("FindMotifsRunConfig2");

            debug!("FindMotifs: find_motifs_run_config_range_or_value={find_motifs_run_config_range_or_value:?} find_motifs_args.fixed_config={:?}", find_motifs_args.fixed_config);

            let input_wrapper = Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(find_motifs_args.resource_config())));

            let init_ctx = root_ctx.branch(Some("find motifs init ctx"), true);
            let init_loader = Arc::clone(&loader);
            let init_ws = Arc::clone(&ws);
            let init_store = Arc::clone(&store);
            let run_config_ids = run_grid (
                None,
                find_motifs_run_config_range_or_value,
                input_wrapper,
                &ws,
                &store,
                async move |run_configs| {
                    Ok(Arc::new(
                        crate::tasks::find_motifs::FindMotifsStage::new(
                            &init_ws,
                            &init_store,
                            init_loader,
                            run_configs,
                            find_motifs_args.fixed_config.clone(), //FixedFindMotifsConfig::new(None, 10_000, false, None),
                            ThoDevice::from(1),
                            init_ctx
                        )
                        .await?,
                    ))
                },
                || root_ctx.branch(Some("find motifs"), true),
                "find motifs",
            ).await?;

            info!("FindMotifs done, run_config_ids={run_config_ids:?}");

        }

        Some(SeqScan(seq_scan_args)) => {
            debug!("SeqScan: seq_scan_args={seq_scan_args:?}");

            let seq_scan_run_config_range_or_value: SeqScanRunConfigRangeOrValue =
                serde_json5::from_str(&seq_scan_args.run_config_cnt).expect("SeqScanRunConfig");

            let input_wrapper = Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(seq_scan_args.resource_config())));

            let run_config_ids = run_grid(
                None,
                seq_scan_run_config_range_or_value,
                input_wrapper,
                &ws,
                &store,
                |run_configs| async {
                    Ok(Arc::new(SeqScanStage::<Simple, RawTensorLoader<RunCtxImpl>, RunCtxImpl>::new_with_chunk_size(
                        &ws,
                        &store,
                        loader,
                        run_configs,
                        seq_scan_args.max_chunk_size
                    )?))
                },
                || root_ctx.branch(Some("seq scan"), true), "find motifs",
            ).await?;

            info!("SeqScan done, run_config_ids={run_config_ids:?}");
        }

        Some(SelectMotifs(select_motifs_args)) => {
            debug!("SelectMotifs: select_motifs_args.run_config_cnt={:?}",select_motifs_args.run_config_cnt);

            let select_motifs_run_config_range_or_value: SelectMotifsRunConfigRangeOrValue = serde_json5::from_str(&select_motifs_args.run_config_cnt).expect("SelectMotifs2");

            let input_wrapper = Arc::new(DevicesWrap::wrap(SelectMotifsInput::standalone(ResourcesConfigInput::new(select_motifs_args.resource_config()))));
            //TODO move it into SelectMotifsStage
            let motif_loader = Arc::new(MotifLoader::new(
                Arc::clone(&store),
                Arc::clone(&loader),
                &ws
            ));

            let run_config_ids = run_grid::</*Labeled,*/ _, _, _, _, _>(
                None,
                select_motifs_run_config_range_or_value,
                input_wrapper,
                &ws,
                &store,
                |run_configs| async {

                    Ok(Arc::new(SelectMotifsStage::<RawTensorLoader<RunCtxImpl>, MotifLoader<RawTensorLoader<RunCtxImpl>, RunCtxImpl>, RunCtxImpl>::new(
                        &ws,
                        &store,
                        loader,
                        motif_loader,
                        run_configs,
                        select_motifs_args.find_motifs_run_config_ids.clone(), //TODO config
                        // #[cfg(test)]
                        None
                    ).await?))
                },
                || root_ctx.branch(Some("select motifs"), true), "find motifs",
            ).await?;

            info!("SelectMotifs done, run_config_ids={run_config_ids:?}");
        }

        Some(Commands::MergeSelectedMotifs(merge_selected_motifs_args)) => {
            debug!("MergeSelectedMotifs: merge_selected_motifs_args{:?}",merge_selected_motifs_args);
            let motif_stats = Arc::new(MotifStats::new(
                Arc::clone(&store),
                Arc::clone(&loader),
                Arc::clone(&ws),
                Arc::new(MotifLoader::new(
                    Arc::clone(&store),
                    Arc::clone(&loader),
                    &ws
                ))
            ));

            let output_wrap = Arc::new(MergeSelectedMotifs::new(&store, &ws, motif_stats)).run(
                Arc::new(DevicesWrap::wrap(MergeSelectedMotifsInput::new(merge_selected_motifs_args.select_motifs_run_config_ids.clone()))),
                root_ctx.branch(Some("select motifs"), true)
            ).await?;

            let output  = Arc::into_inner(output_wrap).unwrap().take_data(0).await?;

            info!("MergeSelectedMotifs done, output={output:?}");
        }

        Some(TrainMotifsNets(train_motif_nets_args)) => {
            debug!("TrainMotifsNets: train_motif_nets_args.run_config_cnt={:?}",train_motif_nets_args.run_config_cnt);

            let train_motif_nets_run_config_range_or_value: TrainMotifNetRunConfigRangeOrValue = serde_json5::from_str(&train_motif_nets_args.run_config_cnt).expect("TrainMotifsNets2");
            let motifs_by = train_motif_nets_args.motifs_by()?;

            let input_wrapper = Arc::new(DevicesWrap::wrap(TrainMotifNetInput::new(motifs_by, Arc::new(train_motif_nets_args.opt_config.clone()))));

            let run_config_ids = run_opt::<_, _, TrainMotifNetRunConfig, _, _>(
                train_motif_nets_run_config_range_or_value,
                input_wrapper,
                &ws,
                &store,
                async |run_configs| {

                    Ok(Arc::new(TrainMotifNets::<_, RunCtxImpl>::new(
                        loader,
                        &ws,
                        &store,
                        run_configs,
                        train_motif_nets_args.epoch_size,
                        None,
                    )?))
                },
                || root_ctx.branch(Some("train motif nets"), true),
            ).await?;

            info!("TrainMotifsNets: run_config_ids={run_config_ids:?}");
        }

        Some(Commands::TrainLabelNets(train_label_nets_args)) => {
            debug!("TrainLabelNets: train_label_nets_args.run_config_cnt={:?}",train_label_nets_args.run_config_cnt);

            let train_motif_nets_run_config_range_or_value: TrainLabelNetRunConfigRangeOrValue = serde_json5::from_str(&train_label_nets_args.run_config_cnt).expect("TrainLabelNets");

            let input_wrapper = Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(train_label_nets_args.resource_config())));

            let opt_config = Arc::new(train_label_nets_args.opt_config.clone());
            let init_ctx = root_ctx.branch(Some("init label nets"), true);
            let labels_filter = train_label_nets_args.labels_filter();

            let run_config_ids = run_opt::</*Labeled,*/ _, _, TrainLabelNetRunConfig, _, _>(
                train_motif_nets_run_config_range_or_value,
                input_wrapper,
                &ws,
                &store,
                async |run_configs| {

                    new_train_label_nets(
                        train_label_nets_args.train_motif_nets_run_config_ids.clone(),
                        loader,
                        run_configs,
                        &ws,
                        &store,
                        init_ctx,
                        opt_config,
                        labels_filter.clone(),
                        train_label_nets_args.max_motif_net_loss,
                        train_label_nets_args.selected_motif_nets_limit,
                        train_label_nets_args.motif_slices_step.unwrap() //TODO or take from find motifs
                    ).await.map(|v|Arc::new(v))
                },
                ||root_ctx.branch(Some("train label nets"), true),
            ).await?;



            info!("TrainMotifsNets done");
        }

        Some(TrainFinalNet(train_final_net_args)) => {
            debug!("TrainFinalNet: train_motif_nets_args.run_config_cnt={:?}",train_final_net_args.run_config_cnt);
            let train_final_net_run_config_range_or_value: TrainFinalNetRunConfigRangeOrValue = serde_json5::from_str(&train_final_net_args.run_config_cnt).expect("TrainFinalNet");

            let input_wrapper = Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(
               train_final_net_args.resource_config())));

            let opt_config = Arc::new(train_final_net_args.opt_config.clone());
            let init_ctx = root_ctx.branch(Some("init final nets"), true);
            let run_config_ids = run_opt::</*Labeled,*/ _, _, TrainFinalNetRunConfig, _, _>(
                train_final_net_run_config_range_or_value,
                input_wrapper,
                &ws,
                &store,
                async |run_configs| {

                    new_train_final_net(
                        train_final_net_args.train_label_nets_run_config_ids.clone(),
                        loader,
                        run_configs,
                        &ws,
                        &store,
                        init_ctx,
                        opt_config,
                        10,//TODO
                        train_final_net_args.selected_motif_nets_limit,
                    ).await.map(|v|Arc::new(v))

                },
                || root_ctx.branch(Some("train final nets"), true),
            ).await?;


            info!("TrainFinalNet done");
        }

        Some(ConvertPrecision(convert_precision_args)) => {
            debug!("ConvertPrecision: convert_precision_args={convert_precision_args:?}");
            let output = Arc::new(ConvertPrecisionStage::<RunCtxImpl>::new(&ws, convert_precision_args.clone()))
                .run(
                    Arc::new(DevicesWrap::wrap(ConvertPrecisionInput::new())),
                    root_ctx.branch(Some("convert_precision"), true),
                )
                .await?;
            let output = Arc::into_inner(output).expect("top level stage should not have more than one reference").take_data(0).await?;
            info!("ConvertPrecision done, converted={}", output.converted().len());
        }

        Some(Export(export_args)) => {
            debug!("Export: export_args={export_args:?}");
            let output = Arc::new(ExportStage::<RunCtxImpl>::new(&ws, &store, export_args.clone()))
                .run(
                    Arc::new(DevicesWrap::wrap(ExportInput::new())),
                    root_ctx.branch(Some("export"), true),
                )
                .await?;
            let output = Arc::into_inner(output).expect("top level stage should not have more than one reference").take_data(0).await?;
            info!("Export done, target_dir={:?} label_nets={} motif_nets={} copied_models={} converted={}", output.target_dir(), output.label_net_run_config_ids().len(), output.motif_net_run_config_ids().len(), output.copied_models().len(), output.converted().len());
        }

        Some(PurgeModels(purge_models_args)) => {
            debug!("PurgeModels: purge_models_args={purge_models_args:?}");
            let output = Arc::new(PurgeModelsStage::<RunCtxImpl>::new(&ws, &store, purge_models_args.clone()))
                .run(
                    Arc::new(DevicesWrap::wrap(PurgeModelsInput::new())),
                    root_ctx.branch(Some("purge_models"), true),
                )
                .await?;
            let output = Arc::into_inner(output).expect("top level stage should not have more than one reference").take_data(0).await?;
            info!("PurgeModels done, purged={}", output.purged().len());
        }

        Some(RunAll(run_all_args)) => {
            debug!("RunAll: run_all_params={:?}", run_all_args);

            // Project-specific callback supplied by thoosa_projects/thoosa_bird_clef/src/main.rs.
            run_all(
                &ws,
                &store,
                &loader,
                root_ctx,
                run_all_args,
            ).await?;

        }

        Some(Infer(infer_params)) => {
            if let Some(infer) = infer {
                infer(infer_params, &ws, &store).await?;
            }
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

        Some(PlotData(plot_slices_params)) => {
            run_plot(&ws, &store, &loader, &plot_slices_params, plot_slices_params.resolution(), &devices[0], root_ctx).await?;
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

            // let dir = read_dir(ws.read().tensors_path())?;
            //
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
            // perf_unfold_concat_cuda_vs_cpu().await?;

            info!("Hello");
        }

        None => {
            warn!("No command");
        }
    }

    info!("\n{}", format_perf_map(&PERF));

    GlobalCtx::global().shutdown().await;

    let _ = stop_cpu_purge_tx.send(true);
    let _ = purge_cpu_worker.await;


    info!("run_id {} done", run_id());

    Ok(())
}


fn resolve_median_window<P: FieldName + PartialEq>(named_vals: &[NamedVal<P>], window_path: &P) -> RangeOrValue<u32> {
    match named_vals.iter().find(|field| field.path() == window_path) {
        None => RangeOrValue::Value(100),
        Some(value) => {
            let window = value.val().to_i64(0) as u32;
            info!("Window: window={window}");
            // RangeOrValue::Value(window)
            RangeOrValue::Range{ range: Range { start: window/2, end: window * 2 }, step: None }
        }
    }
}

fn resolve_median_window_grid<P: FieldName + PartialEq>(named_vals: &[NamedVal<P>], window_path: &P) -> RangeOrValue<u32> {
    match named_vals.iter().find(|field| field.path() == window_path) {
        None => RangeOrValue::Value(100),
        Some(value) => {
            let window = value.val().to_i64(0) as u32;
            info!("Window: window={window}");
            // RangeOrValue::Value(window)
            RangeOrValue::Range{ range: Range { start: window/2, end: window * 2 }, step: Some((window/4, 0)) }
        }
    }
}


pub fn median_window_from_find_motifs(named_vals: &[NamedVal<FindMotifsRunConfigFields>]) -> RangeOrValue<u32> {
    resolve_median_window(named_vals, &FindMotifsRunConfigFields::Window)
}

pub fn median_window_from_find_motifs_grid(named_vals: &[NamedVal<FindMotifsRunConfigFields>]) -> RangeOrValue<u32> {
    resolve_median_window_grid(named_vals, &FindMotifsRunConfigFields::Window)
}


pub fn median_window_from_find_and_select(named_vals: &[NamedVal<FindAndSelectMotifsRunConfigFields>]) -> RangeOrValue<u32> {
    resolve_median_window(
        named_vals,
        &FindAndSelectMotifsRunConfigFields::FindMotifsRunConfig(FindMotifsRunConfigFields::Window),
    )
}

fn run_all_find_motifs_config_grid_0(min_k:u32, max_k:u32, steps:u32) -> FindMotifsRunConfigRangeOrValue<FindMotifsRunConfigFields> {

    let median_window = move |named_vals: &[NamedVal<FindMotifsRunConfigFields>]| {
        match named_vals.iter().find(|field| field.path().eq(&FindMotifsRunConfigFields::Window)) {
            None => RangeOrValue::Value(100),
            Some(value) => {
                let window = value.val().to_i64(0) as u32;
                // info!("Window: window={window}");
                let end = window.saturating_mul(max_k);
                let start = window / min_k;
                RangeOrValue::Range { range: Range { start, end }, step: Some(((end-start)/steps, 0)) }
            }
        }
    };
    FindMotifsRunConfigRangeOrValue {
        id: None,
        step: RangeOrValue::Value(10),
        // window: RangeOrValue::Value(200),
        window: RangeOrValue::Range { range: Range { start: 10, end: 501 }, step: Some((50, 0)) },
        freq_filter: Some(vec![
            AverageFilter(AverageFilterParamsRangeOrValue {
                window: RangeOrValue::dynamic(median_window),
                step: RangeOrValue::Value(10),
            }),
            // SavGolFilter(SavGolFilterParamsRangeOrValue {
            //     window_len: RangeOrValue::dynamic(median_window),//RangeOrValue::Range { range: Range { start: 100, end: 200 }, step: None },
            //     poly_order: RangeOrValue::Range { range: Range { start: 1, end: 10 }, step: Some((1, 0)) },
            //     stride: RangeOrValue::Value(10),
            //     padding: PaddingRangeOrValue::Zero(ZeroPaddingRangeOrValue::default()),
            // }),
        ]),
        finder_kind: FinderKindRangeOrValue::CrossLoader(CrossLoaderRunConfigRangeOrValue {
            similarity_metric: SimilarityMetricRangeOrValue::EucDist(EucDistConfigRangeOrValue { z_normalize: RangeOrValue::Value(false) }),
            slices_filter: SlicesFilterRangeOrValue::Accumulate(AccumulateStatsRangeOrValue {
                aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue {
                    q: RangeOrValue::Value(0.05),
                    // q: RangeOrValue::Range { range: 0.0..0.1, step: Some((0.05, 2)) },
                }),
            }),
        }),
        max_slices: Some(MaxSlicesRangeOrValue::Abs(MaxSlicesAbsRangeOrValue { count: RangeOrValue::Value(1000) })),
    }
}


fn dyn_step<P: FieldName + PartialEq>(named_vals: &[NamedVal<P>], window_path: &P) -> RangeOrValue<u32> {
    match named_vals.iter().find(|field| field.path() == window_path) {
        None => RangeOrValue::Value(10),
        Some(value) => {
            let window = value.val().to_i64(0) as u32;
            // info!("Window: window={window}");

            if window <= 10 {
                RangeOrValue::Value(5)
            // } else if window <= 500 {
            //     RangeOrValue::Value(10)
            } else {
                RangeOrValue::Value(10)
            }
        }
    }
}

pub fn dyn_step_find_and_select(named_vals: &[NamedVal<FindAndSelectMotifsRunConfigFields>]) -> RangeOrValue<u32> {
    dyn_step(
        named_vals,
        &FindAndSelectMotifsRunConfigFields::FindMotifsRunConfig(FindMotifsRunConfigFields::Window),
    )
}

async fn save_run_configs<R: RunConfig + thoosa_properties_map::ToHashmap>(
    input_run_config_ids: Option<&[RunConfigId]>,
    store: &Arc<Box<dyn Store>>,
    run_configs: &mut Vec<R>,
    descr: &str,
) -> Result<Vec<RunConfigId>, ThoError> {
    let mut run_config_ids = Vec::with_capacity(run_configs.len());
    for run_config in run_configs.iter_mut() {
        let run_config_id = store
            .save_run_config(
                input_run_config_ids,
                &<R>::type_name(),
                descr,
                run_config.to_properties_iter(None).collect(),
            )
            .await?;
        run_config.set_id(run_config_id);
        run_config_ids.push(run_config_id);
    }
    Ok(run_config_ids)
}

pub async fn import_resources<F, R, S>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    labels: &[&str],
    labeled_resources: F,
    resource_to_tensor: R, //None for tests - do not import parquet files
    freq_filter: Option<Arc<FreqFilter<RunCtxImpl>>>,
    insert_label: bool,
    init_data:S,
) -> Result<Vec<ResourceId>, ThoError>
where
    F: Fn() -> Result<HashMap<String, Vec<(String, f32)>>, ThoError>,
    R: Fn(String) -> Result<Tensor, ThoError> + Sync + Send,
    S: FnOnce(&Transaction) -> Result<(), ThoError>
{
    // import_resources_with_pure_sets(ws, store, labels, labeled_resources, resource_to_tensor, freq_filter, insert_label, Some(&[(3, true), (10, true), (50, true), (100, true), (150, true), (10000, false)])).await
    let mut conn = store.open_external_writer_conn()?;
    let t = conn.transaction()?;

    if insert_label {
        for label in labels {
            SqlStore::save_label(label, &t)?; //TODO not all labels may have resources, still save all labels?
        }
    }

    let labeled_resources_map = labeled_resources()?;
    let total_count = labeled_resources_map.iter().map(|e|e.1.len()).sum::<usize>();
    debug!("import_resources_with_pure_sets: labeled_resources_map={labeled_resources_map:?}");

    let ws_guard = ws.write();
    let import_descr = format!("Import on {}", cdt());
    let import_id = SqlStore::save_import(import_descr.as_str(), &t)?;

    let import_eta_guard = CollectEtaInvocationGuard::new("import", total_count, 1);
    let import_eta_guard_id = import_eta_guard.id();
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
                    .and_then(|t| apply_import_freq_filter(freq_filter.as_deref(), resource_key.as_str(), t))
                    .and_then(|t| {
                        ws_guard.import_tensor(t, import_id, resource_key.as_str())
                    })
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
                collect_eta_record_completion(import_eta_guard_id);

                a
            },
        )
        .reduce_with(|mut v1, v2| {
            v1.0.extend(v2.0); //TODO assert no keys overlaps
            v1.1.extend(v2.1);
            v1
        })
        .expect("imported_tensors");

    let resource_ids = SqlStore::save_resources_labels(
        imported_tensors_stats
            .iter()
            .map(|(resource_key, (stats, labels))| (resource_key.clone(), stats.length(), labels.clone()))
            .collect(),
        import_id,
        &t,
    )?;

    init_data(&t)?;


    t.commit()?;

    info!(
        "Import id: {}, imported {} tensors to {:?}, failed {}",
        import_id,
        resource_ids.len(),
        ws_guard.tensors_path(),
        failed_tensors.len()
    );

    Ok(resource_ids)
}


fn apply_import_freq_filter(
    freq_filter: Option<&FreqFilter<RunCtxImpl>>,
    resource_key: &str,
    tensor: Tensor,
) -> Result<Tensor, ThoError> {
    match freq_filter {
        Some(freq_filter) => Ok(freq_filter
            .apply_tho_tensor(
                &ThoTensor::new(
                    SynKey::new([mono_key!(resource_key.to_string(), format!("resource_key={resource_key}"))]),
                    tensor,
                )?
                // .to_device(freq_filter.input_device())?,
            )?
            // .to_device(Cpu)?
            .t()),
        None => Ok(tensor),
    }
}

pub async fn run_plot<C>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    loader: &Arc<RawTensorLoader<C>>,
    plot_params: &PlotDataArgs,
    resolution: Resolution,
    _device: &Device,
    mut ctx:C
) -> Result<(), ThoError>
where
    C:RunCtx + Adjustable{

    let print_to_stdout = plot_params.print_to_stdout();

    let motif_loader_impl = Arc::new(MotifLoader::<RawTensorLoader<C>, C>::new(
        Arc::clone(store),
        Arc::clone(loader),
        ws
    ));
    let clc_dist = Arc::new(MotifStats::new(
        Arc::clone(store),
        Arc::clone(loader),
        Arc::clone(ws),
        Arc::clone(&motif_loader_impl),
    ));

    match plot_params.data_by() {
        DataBy::Resource { resource_id, channels, freq_filters } => {
            debug!("run_plot: MotifsBy::Resource resource_id={resource_id} channels={channels:?} freq_filters={freq_filters:?}");
            plot_resource(ws, store, loader, resource_id, channels, freq_filters, resolution, print_to_stdout, ctx, "").await?;
            return Ok(());
        }
        DataBy::MotifNetApplied { resource_id, net_path } => {
            debug!("run_plot: MotifsBy::MotifNetApplied resource_id={resource_id} net_path={net_path:?}");
            plot_motif_net_applied(ws, store, loader, resource_id, net_path, resolution, print_to_stdout, ctx).await?;
            return Ok(());
        }
        DataBy::MotifId(motif_ids) => {
            debug!("run_plot: MotifsBy::MotifId motif_ids={motif_ids:?}");
            for motif_id in motif_ids {
                plot_motif(&ws, ctx.branch(Some(&format!("motif_{motif_id}")), true), motif_id, &motif_loader_impl, Arc::clone(&clc_dist), None, resolution, print_to_stdout).await?;
            }
        }
        DataBy::RunId {
            find_motifs_run_config_ids,
            select_motifs_run_config_ids
        } => {

            debug!("run_plot: MotifsBy::RunId find_run_config_ids={find_motifs_run_config_ids:?} select_run_config_ids={select_motifs_run_config_ids:?}");
            for find_motifs_run_config_id in find_motifs_run_config_ids {
                plot_found_motifs(
                    ws,
                    store,
                    find_motifs_run_config_id,
                    &motif_loader_impl,
                    &clc_dist,
                    select_motifs_run_config_ids.clone(),
                    resolution,
                    print_to_stdout,
                    ctx.branch(Some(&format!("find_motifs_run_{find_motifs_run_config_id}")), true),
                )
                .await?;
            }
        }

        DataBy::SelectedMotifs { select_motifs_run_config_ids } => {
            for select_motifs_run_config_id in select_motifs_run_config_ids {
                plot_selected_motifs(
                    ws,
                    store,
                    select_motifs_run_config_id,
                    &motif_loader_impl,
                    &clc_dist,
                    resolution,
                    print_to_stdout,
                    ctx.branch(Some(&format!("selected_motifs_run_{select_motifs_run_config_id}")), true),
                )
                .await?;
            }
        }

    }

    Ok(())
}

#[derive(Serialize)]
#[serde(untagged)]
enum StdoutPlotNumber {
    Finite(f32),
    Special(String),
}

impl From<f32> for StdoutPlotNumber {
    fn from(value: f32) -> Self {
        if value.is_finite() {
            Self::Finite(value)
        } else {
            Self::Special(value.to_string())
        }
    }
}

#[derive(Serialize)]
struct StdoutPlotPoint {
    x: StdoutPlotNumber,
    y: StdoutPlotNumber,
}

#[derive(Serialize)]
struct StdoutPlotSeries {
    name: String,
    points: Vec<StdoutPlotPoint>,
}

#[derive(Serialize)]
struct StdoutPlotSidebarRow {
    name: String,
    value: StdoutPlotNumber,
}

#[derive(Serialize)]
struct StdoutPlotSidebar {
    title: String,
    rows: Vec<StdoutPlotSidebarRow>,
}

#[derive(Serialize)]
struct StdoutPlotDump {
    format: &'static str,
    caption: String,
    target_path: String,
    size: (u32, u32),
    series: Vec<StdoutPlotSeries>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sidebar: Option<StdoutPlotSidebar>,
}

fn print_plot_data_to_stdout<S>(
    plot_data: Vec<(String, S, Vec<(f32, f32)>)>,
    caption: &str,
    size: (u32, u32),
    path: &impl AsRef<Path>,
    sidebar: Option<(&str, &[(String, f32)])>,
) -> Result<(), ThoError> {
    let dump = StdoutPlotDump {
        format: "thoosa_plot_data_v1",
        caption: caption.to_string(),
        target_path: path.as_ref().display().to_string(),
        size,
        series: plot_data
            .into_iter()
            .map(|(name, _, points)| StdoutPlotSeries {
                name,
                points: points
                    .into_iter()
                    .map(|(x, y)| StdoutPlotPoint { x: x.into(), y: y.into() })
                    .collect(),
            })
            .collect(),
        sidebar: sidebar.map(|(title, rows)| StdoutPlotSidebar {
            title: title.to_string(),
            rows: rows
                .iter()
                .map(|(name, value)| StdoutPlotSidebarRow {
                    name: name.clone(),
                    value: (*value).into(),
                })
                .collect(),
        }),
    };

    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &dump).map_err(|e| ThoError::from_error("print plot data to stdout", e))?;
    writeln!(&mut stdout).map_err(|e| ThoError::from_error("finish plot data stdout line", e))?;
    Ok(())
}

fn emit_or_plot_series<S>(
    print_to_stdout: bool,
    plot_data: Vec<(String, S, Vec<(f32, f32)>)>,
    caption: &str,
    size: (u32, u32),
    path: &impl AsRef<Path>,
) -> Result<(), ThoError>
where
    S: Into<ShapeStyle> + Clone,
{
    if print_to_stdout {
        print_plot_data_to_stdout(plot_data, caption, size, path, None)
    } else {
        plot_series(plot_data, caption, size, path)
    }
}

fn emit_or_plot_series_with_sidebar_values<S>(
    print_to_stdout: bool,
    plot_data: Vec<(String, S, Vec<(f32, f32)>)>,
    caption: &str,
    size: (u32, u32),
    path: &impl AsRef<Path>,
    sidebar_title: &str,
    sidebar_rows: &[(String, f32)],
) -> Result<(), ThoError>
where
    S: Into<ShapeStyle> + Clone,
{
    if print_to_stdout {
        print_plot_data_to_stdout(plot_data, caption, size, path, Some((sidebar_title, sidebar_rows)))
    } else {
        plot_series_with_sidebar_values(plot_data, caption, size, path, sidebar_title, sidebar_rows)
    }
}

async fn plot_found_motifs<C>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    find_motifs_run_config_id: RunConfigId,
    motif_loader_impl: &Arc<MotifLoader<RawTensorLoader<C>, C>>,
    clc_dist: &Arc<MotifStats<MotifLoader<RawTensorLoader<C>, C>, RawTensorLoader<C>, C>>,
    select_motifs_run_config_ids: Option<Vec<RunConfigId>>,
    resolution: Resolution,
    print_to_stdout: bool,
    mut ctx: C,
) -> Result<(), ThoError>
where
    C: RunCtx + Adjustable,
{
    let motifs = store.list_motifs(find_motifs_run_config_id).await?;
    let motifs_len = motifs.len();
    let done_cnt = Arc::new(AtomicU32::new(0));

    let futures = motifs
        .into_iter()
        .map(|motif_id| {
            let done_cnt = Arc::clone(&done_cnt);
            let clc_dist = Arc::clone(clc_dist);
            let motif_loader_impl = Arc::clone(motif_loader_impl);
            let ctx = ctx.branch(Some(&format!("motif_{motif_id}")), true);
            let select_motifs_run_config_ids = select_motifs_run_config_ids.clone();
            async move {
                debug!("run_plot: motif_id={motif_id}");

                if let Err(e) = plot_motif(&ws, ctx, motif_id as usize, &motif_loader_impl, clc_dist, select_motifs_run_config_ids, resolution, print_to_stdout).await {
                    error!("motif {motif_id} failed: e={e:?}");
                };
                MotifLoader::<RawTensorLoader<RunCtxImpl>, RunCtxImpl>::cache().await.clear();

                let done = done_cnt.fetch_add(1, Relaxed);
                if done % 32 == 0 {
                    info!("done {done} from  {motifs_len}");
                }

                Ok::<_, ThoError>(())
            }
        });

    let _ = collect_futures_with_progress(
        futures,
        motifs_len,
        crate::buffer_limits::CLI_RUN_PLOT_MOTIF_SLICES_BUFFER_UNORDERED_LIMIT,
        StreamBuffering::Unordered,
        &format!("plot_found_motifs_for_run_config_id_{find_motifs_run_config_id}"),
    )
    .await;

    Ok(())
}

async fn plot_selected_motifs<C>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    select_motifs_run_config_id: RunConfigId,
    motif_loader_impl: &Arc<MotifLoader<RawTensorLoader<C>, C>>,
    clc_dist: &Arc<MotifStats<MotifLoader<RawTensorLoader<C>, C>, RawTensorLoader<C>, C>>,
    resolution: Resolution,
    print_to_stdout: bool,
    mut ctx: C,
) -> Result<(), ThoError>
where
    C: RunCtx + Adjustable,
{
    let channels = ws.read().channels();
    debug!("plot_selected_motifs: select_motifs_run_config_id={select_motifs_run_config_id} channels={channels}");

    let motifs = store
        .list_selected_motifs(select_motifs_run_config_id, channels)
        .await?;

    debug!("plot_selected_motifs: motifs={motifs:?}");

    let motif_ids: Vec<usize> = motifs
        .into_iter()
        .flat_map(|channel| channel.into_iter().map(|(motif_id, _)| motif_id as usize))
        .collect();

    if motif_ids.is_empty() {
        warn!("plot_selected_motifs: no motifs found for select_motifs_run_config_id={select_motifs_run_config_id}");
        return Ok(());
    }

    let motifs_len = motif_ids.len();
    let done_cnt = Arc::new(AtomicU32::new(0));
    let suffix = Some(vec![select_motifs_run_config_id]);
    let futures = motif_ids
        .into_iter()
        .map(|motif_id| {
            let done_cnt = Arc::clone(&done_cnt);
            let clc_dist = Arc::clone(clc_dist);
            let motif_loader_impl = Arc::clone(motif_loader_impl);
            let ws = Arc::clone(ws);
            let suffix = suffix.clone();
            let ctx = ctx.branch(Some(&format!("motif_{motif_id}")), true);
            async move {
                debug!("plot_selected_motifs: motif_id={motif_id}");
                if let Err(e) = plot_motif(&ws, ctx, motif_id, &motif_loader_impl, clc_dist, suffix, resolution, print_to_stdout).await {
                    error!("motif {motif_id} failed: e={e:?}");
                };
                //MotifLoader::<RawTensorLoader<RunCtxImpl>, RunCtxImpl>::cache().await.clear();

                let done = done_cnt.fetch_add(1, Relaxed);
                if done % 32 == 0 {
                    info!("done {done} from  {motifs_len}");
                }

                Ok::<_, ThoError>(())
            }
        });

    let _ = collect_futures_with_progress(
        futures,
        motifs_len,
        crate::buffer_limits::CLI_RUN_PLOT_MOTIF_SLICES_BUFFER_UNORDERED_LIMIT,
        StreamBuffering::Unordered,
        &format!("plot_selected_motifs_for_run_config_id_{select_motifs_run_config_id}"),
    )
    .await;

    Ok(())
}

async fn load_resource_tensor_for_plot<C>(
    store: &Arc<Box<dyn Store>>,
    loader: &Arc<RawTensorLoader<C>>,
    resource_id: ResourceId,
    mut ctx: C,
) -> Result<Arc<DevicesWrap<ThoTensor<ChannelsData2, SynKey>>>, ThoError>
where
    C: RunCtx + Adjustable,
{
    let resource = store
        .list_resources_by_ids(&vec![resource_id])
        .await?
        .into_iter()
        .next()
        .ok_or_else(|| ThoError::new(format!("resource_id={resource_id} not found")))?;

    Arc::clone(loader)
        .run_cached(
            Arc::new(DevicesWrap::wrap(resource)),
            ctx.branch(Some("plot_resource_load"), false),
        )
        .await
}

fn sanitize_plot_path_segment(v: &str) -> String {
    let s = v
        .chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' | '.' => c,
            _ => '_',
        })
        .collect::<String>();
    if s.is_empty() { "plot".to_string() } else { s }
}

fn plot_resource_channels<C>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    resource_id: ResourceId,
    prefix: &str,
    suffix: &str,
    caption: &str,
    tensor: &ThoTensor<ChannelsData2, SynKey>,
    channels: Option<&Vec<u32>>,
    resolution: Resolution,
    print_to_stdout: bool,
) -> Result<(), ThoError>
where
    C: RunCtx + Adjustable,
{
    let total_channels = tensor.size()[0] as u32;
    let channels_to_plot = channels.cloned().unwrap_or_else(|| (0..total_channels).collect());
    if channels_to_plot.is_empty() {
        return Err(ThoError::new(format!("plot_resource_channels: no channels requested for resource_id={resource_id}")));
    }
    for channel in &channels_to_plot {
        if *channel >= total_channels {
            return Err(ThoError::new(format!("plot_resource_channels: channel {channel} out of range [0,{total_channels}) for resource_id={resource_id}")));
        }
    }

    let mut color_iter = DistinctColorIter::new();
    let plot_data = channels_to_plot
        .into_iter()
        .map(|channel| {
            let channel_t = tensor.t_ref().f_i((channel as i64, ..))?;
            Ok::<_, ThoError>((
                format!("channel_{channel}"),
                color_iter.next().unwrap_or(BLACK),
                channel_t.iter::<f64>()?.enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>(),
            ))
        })
        .collect::<Result<Vec<_>, ThoError>>()?;

    emit_or_plot_series(
        print_to_stdout,
        plot_data,
        caption,
        resolution.plot_size((2000, 1000)),
        &ws.read().plots_path().join("resource").join(format!("{prefix}resource_{resource_id}_{suffix}.png")),
    )?;

    Ok(())
}

fn plot_single_line_tensor(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    path_suffix: &str,
    caption: &str,
    line_t: &Tensor,
    resolution: Resolution,
    print_to_stdout: bool,
) -> Result<(), ThoError> {
    let line_t = line_t.f_view([-1])?;
    let points = line_t
        .iter::<f64>()?
        .enumerate()
        .map(|(i, v)| (i as f32, v as f32))
        .collect::<Vec<_>>();

    emit_or_plot_series(
        print_to_stdout,
        vec![("0".to_string(), BLACK, points)],
        caption,
        resolution.plot_size((2000, 1000)),
        &ws.read().plots_path().join("resource").join(format!("{path_suffix}.png")),
    )?;

    Ok(())
}

pub async fn plot_resource<C>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    loader: &Arc<RawTensorLoader<C>>,
    resource_id: ResourceId,
    channels: Option<Vec<u32>>,
    freq_filters: Option<Vec<InputFilter>>,
    resolution: Resolution,
    print_to_stdout: bool,
    mut ctx: C,
    prefix: &str,
) -> Result<(), ThoError>
where
    C: RunCtx + Adjustable,
{
    let raw = load_resource_tensor_for_plot(store, loader, resource_id, ctx.branch(Some("plot_resource_load"), false)).await?;

    let raw_channels = channels.clone();
    let channels_frag = sanitize_plot_path_segment(&format!("{channels:?}", ));

    raw.with_data_ref(0, ctx.branch(Some("plot_resource_raw"), false), None, |tensor: &ThoTensor<ChannelsData2, SynKey>, _ctx| {
        plot_resource_channels::<C>(ws, resource_id, prefix, &format!("raw_{channels_frag}",), &format!("Resource {resource_id} raw channels: {raw_channels:?}"), tensor, raw_channels.as_ref(), resolution, print_to_stdout)
    }).await?;

    if let Some(freq_filters) = freq_filters {
        let filter_frag = sanitize_plot_path_segment(&freq_filters.iter().map(|f|format!("{:?}", f)).collect::<String>());//TODO remove "*Params"
        let filtered = Arc::new(FreqFilter::<C>::new(freq_filters.as_slice(), ws, resource_id)?)
            .run(
                Arc::clone(&raw),
                ctx.branch(Some("plot_resource_freq_filter"), false),
            )
            .await?;

        let filtered_channels = channels.clone();
        filtered.with_data_ref(0, ctx.branch(Some("plot_resource_freq_filter_plot"), false), None, |tensor: &ThoTensor<ChannelsData2, SynKey>, _ctx| {
            plot_resource_channels::<C>(ws, resource_id, prefix, &format!("freq_filtered_{channels_frag}_{filter_frag}"), &format!("Resource {resource_id} channels: {filtered_channels:?} filters {freq_filters:?}"), tensor, filtered_channels.as_ref(), resolution, print_to_stdout)
        }).await?;
    }

    Ok(())
}

async fn plot_motif_net_applied<C>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    loader: &Arc<RawTensorLoader<C>>,
    resource_id: ResourceId,
    net_path: PathBuf,
    resolution: Resolution,
    print_to_stdout: bool,
    mut ctx: C,
) -> Result<(), ThoError>
where
    C: RunCtx + Adjustable,
{
    let raw = load_resource_tensor_for_plot(store, loader, resource_id, ctx.branch(Some("plot_resource_motif_net_load"), false)).await?;

    // raw.with_data_ref(0, ctx.branch(Some("plot_resource_motif_net_raw_all"), false), None, |tensor: &ThoTensor<ChannelsData2, SynKey>, _ctx| {
    //     plot_resource_channels::<C>(ws, resource_id, "raw_all_channels", &format!("Resource {resource_id} raw (all channels)"), tensor, None)
    // }).await?;

    let (path_channel, motif_id) = FsWorkspace::parse_motif_path(&net_path)?;
    let (store_channel, find_motifs_run_config_id) = store.get_motif_meta(motif_id).await?;

    let find_motifs_run_config:FindMotifsRunConfig = load_run_config(store, find_motifs_run_config_id).await?;

    if path_channel as usize != store_channel {
        warn!("plot_motif_net_applied_variant: channel mismatch path_channel={path_channel} store_channel={store_channel} motif_id={motif_id} net_path={net_path:?}");
    }
    let channel = store_channel as u32;
    let channel_only = vec![channel];
    let net_path_sfx = sanitize_plot_path_segment(&net_path.to_string_lossy());

    raw.with_data_ref(0, ctx.branch(Some("plot_resource_motif_net_raw_channel"), false), None, |tensor: &ThoTensor<ChannelsData2, SynKey>, _ctx| {
        plot_resource_channels::<C>(ws, resource_id, "", &format!("motif_net_input_raw_ch_{channel}"), &format!("Resource {resource_id} motif-net input raw channel={channel}"), tensor, Some(&channel_only), resolution, print_to_stdout)
    }).await?;



    let filtered_or_raw = if let Some(freq_filters) = find_motifs_run_config.freq_filter.clone() {
        let filtered = Arc::new(FreqFilter::<C>::new(freq_filters.as_slice(), ws, find_motifs_run_config_id)?)
            .run(
                Arc::clone(&raw),
                ctx.branch(Some("plot_resource_motif_net_prefilter"), false),
            )
            .await?;

        let channel_only_filtered = vec![channel];
        filtered.with_data_ref(0, ctx.branch(Some("plot_resource_motif_net_prefilter_channel"), false), None, |tensor: &ThoTensor<ChannelsData2, SynKey>, _ctx| {
            plot_resource_channels::<C>(ws, resource_id, "", &format!("motif_net_input_prefiltered_ch_{channel}_{net_path_sfx}"), &format!("Resource {resource_id} motif-net prefiltered channel={channel} find_run_config_id={find_motifs_run_config_id}"), tensor, Some(&channel_only_filtered), resolution, print_to_stdout)
        }).await?;

        filtered
    } else {
        Arc::clone(&raw)
    };

    let (motif_net_params, motif_net) = {
        let ws_guard = ws.read();
        ws_guard.load_motif_net(&net_path, &Cpu, false)?
    };
    let motif_slices_step = 5u32; //motif_net_params.stride() as u32;
    let motif_net = Arc::new(DataWithExtra::new(motif_net, MotifNetMeta::new(motif_id, channel, motif_net_params)));

    filtered_or_raw.with_data_ref(0, ctx.branch(Some("plot_resource_motif_net_apply"), false), None, {
        let motif_net = Arc::clone(&motif_net);
        move |tensor: &ThoTensor<ChannelsData2, SynKey>, _ctx| {
            let out = apply_motif_net_forward::<C>(
                mono_key!(format!("plot_motif_net_applied_resource_{resource_id}"), format!("motif_{motif_id}_channel_{channel}")) + tensor.id(),
                tensor,
                motif_net.as_ref(),
                motif_slices_step,
                tensor.id(),
                None,
            )?;

            plot_single_line_tensor(
                ws,
                &format!("resource_{resource_id}_motif_net_applied_{motif_id}_{channel}_{net_path_sfx}"),
                &format!("Resource {resource_id} motif-net output motif_id={motif_id} channel={channel} find_run_config_id={find_motifs_run_config_id} net={net_path:?}"),
                out.t_ref(),
                resolution,
                print_to_stdout,
            )?;

            Ok(())
        }
    }).await?;

    Ok(())
}

pub async fn plot_motif<C>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    mut ctx: C,
    motif_id: usize,
    motif_loader_impl: &Arc<MotifLoader<RawTensorLoader<C>, C>>,
    motif_stats: Arc<MotifStats<MotifLoader<RawTensorLoader<C>, C>, RawTensorLoader<C>, C>>,
    select_motif_run_config_ids: Option<Vec<RunConfigId>>,
    resolution: Resolution,
    print_to_stdout: bool,
) -> Result<(), ThoError>
where
    C:RunCtx + Adjustable{
    let select_motif_run_config_ids_sfx = select_motif_run_config_ids.as_ref().map(|v|v.iter().map(|v|format!("_{}", v)).collect::<String>()).unwrap_or("".to_string());
    let motif_loader_input = MotifLoaderInput::new(((motif_id as u32), "motif_id").into(), select_motif_run_config_ids);
    let motif_id_arc = Arc::new(DevicesWrap::wrap(motif_loader_input.clone()));
    let metrics = Metrics::all();
    let clc_input = Arc::new(DevicesWrap::wrap(MotifStatsInput::from_motif_input(motif_loader_input, metrics.clone())));

    let motif = Arc::clone(&motif_loader_impl).run_cached(
        Arc::clone(&motif_id_arc),
        ctx.branch(Some("plot_motif_loader"), false),
    )
    .await?;

    let metrics_for_plot: Vec<(String, f32)> = motif_stats
        .run/*_cached*/(clc_input, ctx.branch(Some("plot_motif_clc_dist"), false))
        .await?
        .with_data_ref(
            0,
            ctx.branch(Some("plot_motif_metrics_values"), false),
            None,
            move |stats: &MotifStatsOutput, _ctx| {
                metrics
                    .iter()
                    .map(|metric| Ok((format!("{metric:?}"), stats.channel_value(metric, 0)?)))
                    .collect::<Result<Vec<_>, ThoError>>()
            },
        )
        .await.unwrap_or(vec![]);

    let metrics_for_plot = Arc::new(metrics_for_plot);


    let metrics_for_plot_ref = Arc::clone(&metrics_for_plot);
    let r = motif.with_data_ref(ctx.device_idx(), ctx.branch(Some("plot_motif_data"), false), None, move |motif: &DataWithExtra<ThoTensor<_, SynKey>, MotifMeta>, _ctx| {
        let run_config_id = motif.extra().find_motifs_run_config_id;
        let channel = motif.extra().channel;

        trace!("plot_motif: motif.t={}", motif.data().t_ref());
        let all_slices = motif.data().t_ref().unbind(0);
        let slices_len = all_slices.len();
        let mut color_iter = DistinctColorIter::new();

        let target_dit = ws.read().plots_path().join(format!("{run_config_id}{select_motif_run_config_ids_sfx}"));

        fs::create_dir_all(&target_dit)?;
        let mut specs = vec![];
        for (i, slice) in all_slices.iter().enumerate() {
            let spec = slice.stft(
                16000,
                4000,
                16000,
                None::<Tensor>,
                false,
                true,
                true,
                true
            );
            let spec_abs =spec.squeeze_dim(0).abs().transpose(0,1);


            plot_spectrogram(
                &spec_abs,
                &target_dit.join(format!("spec_{i}_{}.png", motif_id)),
                1200,
                600,
            )?;
            specs.push(spec_abs);
        }
        let all_spects_t = Tensor::stack(&specs, 0).sum_dim_intlist(0, false, None);

        plot_spectrogram(
            &all_spects_t,
            &target_dit.join(format!("spec_all_{}.png", motif_id)),
            1200,
            600,
        )?;

        let plot_data = all_slices
            .into_iter()
            .enumerate()
            .map(|(i, t)| {
                Ok::<_, ThoError>((
                    format!("{i}"),
                    color_iter.next().unwrap_or(BLACK),
                    t.iter::<f64>()?.enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>(),
                ))
            })
            .collect::<Result<Vec<_>, ThoError>>()?;

        emit_or_plot_series_with_sidebar_values(
            print_to_stdout,
            plot_data,
            format!("Motif: {motif_id} run id: {run_config_id} channel: {channel} number of slices: {slices_len}").as_str(),
            resolution.plot_size((2600, 1200)),
            &ws.read().plots_path().join(format!("{run_config_id}{select_motif_run_config_ids_sfx}")).join(format!("motif_{}.png", motif_id)),
            "Motif Metrics",
            metrics_for_plot_ref.as_slice(),
        )?;


        Ok(true)
    }).await?;
    Ok(())
}

pub async fn run_grid<RCRV, F, RC, ST, CtxF>(
    input_run_config_ids: Option<Vec<RunConfigId>>,
    rcrv: RCRV,
    input: Arc<DevicesWrap<ST::In<'static>>>,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    f: F,
    ctx_f: CtxF,
    descr: &str,
) -> Result<Vec<RunConfigId>, ThoError>
where
    F: AsyncFnOnce(Vec<RC>) -> Result<Arc<ST>, ThoError>,
    ST: Stage + AsyncStage<InWrap = DevicesWrap<<ST as Stage>::In<'static>>, OutWrap = DevicesWrap<<ST as Stage>::Out>> + Send + Sync, // + From<(Arc<RwLock<Box<dyn Workspace>>>, Arc<Box<dyn Store>>, Arc<RwLock<LoadRunner<Tensor>>>, Vec<RC>, FC)>,
    RC: RunConfig
        + From<std::vec::IntoIter<NamedVal<<RCRV as GridRunConfigSource>::P>>>
        + thoosa_properties_map::ToHashmap,
    RCRV: GridRunConfigSource + Debug,
    for<'c> <ST as Stage>::In<'c>: Data + WithId + Send + Sync,
    for<'c> <<ST as Stage>::In<'c> as Data>::M: DevicesWrapCopyMode<<ST as Stage>::In<'c>>,
    for<'a> <ST as Stage>::Out: Data + WithId + Send + Sync + SavableResult<P<'a>:Default>,
    <<ST as Stage>::Out as Data>::M: DevicesWrapCopyMode<<ST as Stage>::Out>,
    CtxF: FnOnce() -> <ST as Stage>::Ctx,
{
    debug!("run_all_grid_step: rcrv={rcrv:?}");

    let grid_run_configs = rcrv.grid_run_configs()?;
    debug!("run_grid: grid_run_configs={grid_run_configs:?}");

    let mut run_configs: Vec<RC> = grid_run_configs.to_run_configs();
    info!("run_grid: run_configs.len={} ", run_configs.len());
    debug!("run_grid: run_configs={run_configs:?}");

    let mut run_config_ids = save_run_configs(
        input_run_config_ids.as_ref().map(|v| v.as_slice()),
        &store,
        &mut run_configs,
        descr,
    ).await?;

    let stage = f(run_configs).await?;

    debug!("run_grid: stage={stage:?}");

    let mut savable = Arc::into_inner(stage.run(input, ctx_f()).await?).expect("top level stage should not have more than one reference").take_data(0).await?;

    debug!("run_grid: savable={savable:?}");

    savable.save(store, ws, <<ST as Stage>::Out as SavableResult>::P::default()).await?;


    Ok(run_config_ids)
}

pub async fn run_opt<RCRV, F, RC, ST, CtxF>(
    rcrv: RCRV,
    input: Arc<DevicesWrap<ST::In<'static>>>,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    init_stage: F,
    ctx_f: CtxF,
) -> Result<Vec<RunConfigId>, ThoError>
where
    F: AsyncFnOnce(OptRunConfigs<
        Vec<NamedVal<<RCRV as OptRunConfigSource>::P>>,
        Vec<NamedRange<<RCRV as OptRunConfigSource>::P>>,
        Vec<Hyperparameter<<RCRV as OptRunConfigSource>::P>>,
        <RCRV as OptRunConfigSource>::P,
    >) -> Result<Arc<ST>, ThoError>,
    ST: Stage + AsyncStage<InWrap = DevicesWrap<<ST as Stage>::In<'static>>, OutWrap = DevicesWrap<<ST as Stage>::Out>> + Send + Sync,
    RC: RunConfig
    + From<std::vec::IntoIter<NamedVal<<RCRV as OptRunConfigSource>::P>>>
    + ToHashmap,
    RCRV: OptRunConfigSource,
    for<'c> <ST as Stage>::In<'c>: Data + WithId + Send + Sync,
    for<'c> <<ST as Stage>::In<'c> as Data>::M: DevicesWrapCopyMode<<ST as Stage>::In<'c>>,
    for<'a> <ST as Stage>::Out: Data + WithId + Send + Sync + SavableResult<P<'a>:Default, R=Vec<RunConfigId>>,
    <<ST as Stage>::Out as Data>::M: DevicesWrapCopyMode<<ST as Stage>::Out>,
    CtxF: FnOnce() -> <ST as Stage>::Ctx,
{
    let opt_run_configs = rcrv.opt_run_configs()?;
    debug!("run_all_grid_step: run_opt={opt_run_configs:?}");

    let stage = init_stage(opt_run_configs).await?;

    let mut savable = Arc::into_inner(stage.run(input, ctx_f()).await?).expect("").take_data(0).await?;

    let run_config_id = panic_to_err_async!(savable.save(store, ws, <<ST as Stage>::Out as SavableResult>::P::default()))?;

    Ok(run_config_id)
}

async fn perf_unfold_concat_cuda_vs_cpu() -> Result<(), ThoError>{
    async fn load_tensors(dim_1: &i64, num_tensors: usize, cuda_first:bool) -> Result<u128, ThoError> {
        let mut tensors = Vec::with_capacity(num_tensors);

        let ctx = thoosa_common::runner::global_ctx::GlobalCtx::global().copy_ctx(Cuda(0)).await;

        let t_0 = Instant::now();
        for i in 0..num_tensors {


            let t = Tensor::f_stack(&[
                make_t(&dim_1, &(i as i64), 0.1, &ctx, cuda_first).await?,
                make_t(&dim_1, &(i as i64), 0.2, &ctx, cuda_first).await?,
                make_t(&dim_1, &(i as i64), 0.3, &ctx, cuda_first).await?,
                make_t(&dim_1, &(i as i64), 0.4, &ctx, cuda_first).await?,
                make_t(&dim_1, &(i as i64), 0.5, &ctx, cuda_first).await?,
            ], 0)?
                .f_unfold(1, 1000, 100)?;

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
        let t = x + Tensor::f_arange_start(i * dim_1, (i + 1) * dim_1, FLOAT_CPU)?;

        if cuda_first {
            copy_ctx.copy(&t, 1, Float).await
        } else {
            Ok(t)
        }
    }

    // log::set_max_level(Info);
    // GlobalCtx::init_devices(2);

    let num_tensors = 3;

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

//to check if there are no `#field_ident field value Real(87.0) is not valid: ThoosaError { msg: "can't convert Real(87.0)", frame_0
fn check_opt_run_config<RCRV, RC>(find_and_select_run_config: &RCRV) -> Result<(), ThoError>
where
    RCRV: OptRunConfigSource,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<<RCRV as OptRunConfigSource>::P>>> + ToHashmap,
    <RCRV as OptRunConfigSource>::P: 'static,
{
    let opt_run_config = find_and_select_run_config.opt_run_configs()?;
    let const_vals = opt_run_config.grid_run_config_source().const_params();
    let grid_runs = opt_run_config.grid_to_vals_and_opt_params();

    for (grid_vals, opt_hps) in grid_runs {
        let opt_vals = opt_hps.iter().map(|hp| NamedVal::new(hp.name().clone(), hp.random_sample())).collect::<Vec<_>>();

        for i in 0..100 {
            let mut run_config: RC = [const_vals.as_slice(), grid_vals.as_slice(), &opt_vals].concat().into_iter().into();

            debug!("run_config={run_config:?}");
        }
    }
    Ok(())
}



#[cfg(test)]
mod cli_tests {
    use crate::cli::{import_resources, median_window_from_find_and_select, run_all_find_motifs_config_grid_0, run_grid, run_plot, Cli, RunAllArgs};
    use clap::Parser;
    use futures::executor::block_on;
    use log::LevelFilter::{Debug, Error, Info, Trace, Warn};
    use log::{debug, error, info, set_max_level, trace, LevelFilter};
    use parking_lot::lock_api::MutexGuard;
    use parking_lot::{Mutex, RawMutex, RawRwLock, RwLock};
    use plotters::prelude::{IntoLinspace, BLACK};
    use rand::{rng, Rng, RngExt};
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    use rayon::prelude::IntoParallelIterator;
    use std::cmp::{max, min};
    use std::collections::{HashMap, HashSet};
    use std::io::Read;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::vec::IntoIter;
    use std::{fs, ptr};
    use std::ops::Range;
    use std::time::Duration;
    use mockall_double::double;
    use tch::Device::{Cpu, Cuda};
    use tch::display::{set_print_options, PrinterOptions};
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::{nn, Device, IndexOp, Tensor};
    use tch::Kind::Float;
    use crate::tasks::find_motifs_run_config::{AccumulateStatsRangeOrValue, AggrRangeOrValue, CoherenceRunConfigRangeOrValue, CrossLoaderRunConfigRangeOrValue, ExtendMotifsConfigRangeOrValue, FinderKindRangeOrValue, FindMotifsRunConfig, FindMotifsRunConfigFields, FindMotifsRunConfigRangeOrValue, InlierRatioRunConfigRangeOrValue, MaxSlicesAbsRangeOrValue, MaxSlicesRangeOrValue, MetricsRangeOrValue, MotifCriterionConstraintRangeOrValue, NormThicknessRunConfigRangeOrValue, PrecalculateStatsRangeOrValue, QuantileAggrRangeOrValue, SelectMotifsRunConfigRangeOrValue, SlicesFilterRangeOrValue};
    use thoosa_common::input_probs::{compare_input_probs, InputProbs};
    use thoosa_common::my_tch::ThoModule;
    use thoosa_common::runs::{Empty, Labeled, Logits, NetResources, Simple};
    use thoosa_common::tch_test_utils;
    use thoosa_common::tch_test_utils::{resource_tensor_two_easy_motifs, Interval};
    use thoosa_common::tho_device::ThoDevice;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{lin_motif_node, CommonTrainingRunConfig, InferArgs, MotifCosineNetParams, MotifNetParams, MotifNetTypeParams, OptConfig, TrainFinalNetRunConfig, TrainFinalNetRunConfigRangeOrValue, TrainLabelNetRunConfig, TrainLabelNetRunConfigFields, TrainLabelNetRunConfigRangeOrValue, TrainMotifNetRunConfig, TrainMotifNetRunConfigRangeOrValue};
    use thoosa_common::transforms::filters::{AverageFilterParams, AverageFilterParamsRangeOrValue, BandPassFilterParams, BandPassFilterParamsRangeOrValue, InputFilter, InputFilterRangeOrValue, MedianFilterParams, NotchBandFilterParamsRangeOrValue, Padding};
    use thoosa_common::debug_plot;
    use thoosa_common::dims::MotifSlices;
    use thoosa_common::utils::{from, lin_t, local_max_1d, plot_series, reset_print_options, t_0_0, t_0_1, t_1_0, t_1_1, three_sigs_tensor, to_vec_f32, white_tophat1d};
    use thoosa_common::ws::{mock_ws_simp, FsWorkspace, Workspace};
    use thoosa_motifs::distance::MaskMaker::LeQuantile;
    use thoosa_motifs::fragments::Resolution;
    use thoosa_motifs::find::{
        FixedFindMotifsConfig, SelectMotifsParams};
    use thoosa_motifs::fragments::{DataBy, PlotDataArgs};
    use thoosa_properties_map::ToHashmap;
    use thoosa_run_config_args::OptRunConfigSource;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::{Resource, ResourceId, ResourceStats, ResourcesConfig};
    use thoosa_shared::run_configs::RunConfig;
    use thoosa_shared::run_configs::RunConfigId;
    use thoosa_shared::test_utils::MotifKind;
    use thoosa_shared::{cdt, cdt_for_dir, mono_key, panic_to_err_async, random_saw, recurse_files, saw, saw_with_steps, sig_one_imp, sig_three_imp, sig_two_imp, RangeOrValue, ResourceType, TensorIdSliceIdx, ThoError};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::perf_logging::{format_perf_map, set_panic_hook, PERF};
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::common::{mock_run_ctx, MaxParallelCtx, MockRunCtx, RunCtx};
    use thoosa_common::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
    use thoosa_common::runner::common::mock_run_ctx::MockRunCtxBuilder;
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::global_ctx::{mock_global_ctx, GlobalCtx};
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use thoosa_common::transforms::filters::InputFilterRangeOrValue::AverageFilter;
    use thoosa_motifs::distance::{EucDistConfigRangeOrValue, SimilarityMetricRangeOrValue};
    use thoosa_shared::traits::{Data, DataId};
    use thoosa_shared::val::{NamedVal, Val};
    use thoosa_store::{MotifTrackerStore, SlicesMatch, SqlMotifTrackerStore, SqlStore, Store};
    use thoosa_store::rusqlite::Transaction;
    use thoosa_with_data::with_data;
    use crate::tasks::slices_stats::{Aggr, Metrics};
    use crate::tasks::cross_dims::ChannelsData2;
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::find_and_select_motifs::{FindAndSelectMotifsRunConfig, FindAndSelectMotifsRunConfigFields, FindAndSelectMotifsRunConfigRangeOrValue};
    use crate::tasks::find_motifs::FindMotifsStage;
    use crate::tasks::motif_stats::MotifStats;
    use crate::tasks::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta};
    use crate::tasks::tensor_loader::RawTensorLoader;
    use crate::tasks::transforms::freq_filter::FreqFilter;

    #[ctor::ctor]
    pub fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_write_perf_log_file_default_is_false() {
        let cli = Cli::parse_from(["thoosa_main"]);
        assert!(!cli.write_perf_log_file);
    }

    #[test]
    fn test_write_perf_log_file_flag_enables_perf_file_logging() {
        let cli = Cli::parse_from(["thoosa_main", "--write-perf-log-file"]);
        assert!(cli.write_perf_log_file);
    }

    #[test]
    fn test_plot_data_resolution_flag_parses_high() {
        let cli = Cli::parse_from(["thoosa_main", "plot-data", "--data-by", "motif_id:7", "--resolution", "high"]);

        match cli.command {
            Some(crate::cli::Commands::PlotData(args)) => assert_eq!(args.resolution(), Resolution::High),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn test_plot_data_stdout_flag_parses_true() {
        let cli = Cli::parse_from(["thoosa_main", "plot-data", "--data-by", "motif_id:7", "--stdout"]);

        match cli.command {
            Some(crate::cli::Commands::PlotData(args)) => assert!(args.print_to_stdout()),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn test_plot_data_motif_id_parses_multiple_ids() {
        let cli = Cli::parse_from(["thoosa_main", "plot-data", "--data-by", "motif_id:1,2,3"]);

        match cli.command {
            Some(crate::cli::Commands::PlotData(args)) => assert_eq!(args.data_by(), DataBy::MotifId(vec![1, 2, 3])),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn test_plot_data_motif_id_parses_spaced_quoted_ids() {
        let cli = Cli::parse_from(["thoosa_main", "plot-data", "--data-by", "motif_id:'1, 2'"]);

        match cli.command {
            Some(crate::cli::Commands::PlotData(args)) => assert_eq!(args.data_by(), DataBy::MotifId(vec![1, 2])),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn test_convert_precision_command_parses() {
        let cli = Cli::parse_from(["thoosa_main", "convert-precision", "--nets-by", "run_config_id:1,2", "--target-precision", "bf16"]);

        match cli.command {
            Some(crate::cli::Commands::ConvertPrecision(args)) => {
                assert_eq!(args.nets_by(), crate::tasks::convert_precision::ConvertPrecisionBy::RunId(vec![1, 2]));
                assert_eq!(args.target_precision(), crate::tasks::convert_precision::ConvertPrecisionTarget::Bf16);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn test_merge_selected_motifs_ids_parse_single_spaced_and_csv_values() {
        let single = Cli::parse_from(["thoosa_main", "merge-selected-motifs", "--select-motifs-run-config-ids=1"]);
        match single.command {
            Some(crate::cli::Commands::MergeSelectedMotifs(args)) => assert_eq!(args.select_motifs_run_config_ids, vec![1]),
            other => panic!("unexpected command: {other:?}"),
        }

        let spaced = Cli::parse_from(["thoosa_main", "merge-selected-motifs", "--select-motifs-run-config-ids='1, 2'"]);
        match spaced.command {
            Some(crate::cli::Commands::MergeSelectedMotifs(args)) => assert_eq!(args.select_motifs_run_config_ids, vec![1, 2]),
            other => panic!("unexpected command: {other:?}"),
        }

        let csv = Cli::parse_from(["thoosa_main", "merge-selected-motifs", "--select-motifs-run-config-ids=1,2,3"]);
        match csv.command {
            Some(crate::cli::Commands::MergeSelectedMotifs(args)) => assert_eq!(args.select_motifs_run_config_ids, vec![1, 2, 3]),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn test_run_all_resolution_flag_parses_high() {
        let cli = Cli::parse_from([
            "thoosa_main",
            "run-all",
            "--find-resources", "group:find",
            "--select-resources", "group:select",
            "--resolution", "high",
        ]);

        match cli.command {
            Some(crate::cli::Commands::RunAll(args)) => assert_eq!(args.resolution, Resolution::High),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn test_import_resources_freq_filters_parse() {
        let cli = Cli::parse_from([
            "thoosa_main",
            "import-resources",
            "--source-path",
            "/tmp/input",
            "--freq-filters",
            "{MedianFilter:{window:5}}",
            "--freq-filters",
            "{AverageFilter:{window:7,step:1}}",
        ]);

        match cli.command {
            Some(crate::cli::Commands::ImportResources(args)) => {
                assert_eq!(
                    args.freq_filters,
                    vec![
                        InputFilter::MedianFilter(MedianFilterParams { window: 5 }),
                        InputFilter::AverageFilter(AverageFilterParams { window: 7, step: 1 }),
                    ],
                );
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn test_seq_scan_command_parses() {
        let cli = Cli::parse_from([
            "thoosa_main",
            "seq-scan",
            "--resources", "group:find",
            "--run-config-cnt", "{step:10,window:100,metric:{MeanAbs:{center:false}},buckets_count:4}",
        ]);

        match cli.command {
            Some(crate::cli::Commands::SeqScan(args)) => {
                assert_eq!(args.resources, ResourceType::Group("find".to_owned()));
                assert_eq!(args.run_config_cnt, "{step:10,window:100,metric:{MeanAbs:{center:false}},buckets_count:4}");
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn test_run_all_config_0_preserves_dynamic_find_and_select_fields() {
        let config =     FindMotifsRunConfigRangeOrValue {
            id: None,
            step: RangeOrValue::Value(10),

            window: RangeOrValue::Range { range: Range { start: 10, end: 501 }, step: Some((50, 0)) },
            freq_filter: Some(vec![
                AverageFilter(AverageFilterParamsRangeOrValue {
                    window: RangeOrValue::dynamic(median_window_from_find_and_select),
                    step: RangeOrValue::Value(10),
                }),

            ]),
            finder_kind: FinderKindRangeOrValue::CrossLoader(CrossLoaderRunConfigRangeOrValue {
                similarity_metric: SimilarityMetricRangeOrValue::EucDist(EucDistConfigRangeOrValue { z_normalize: RangeOrValue::Value(false) }),
                slices_filter: SlicesFilterRangeOrValue::Accumulate(AccumulateStatsRangeOrValue {
                    aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue {
                        q: RangeOrValue::Range { range: 0.0..0.05, step: None },
                    }),
                }),
            }),
            max_slices: Some(MaxSlicesRangeOrValue::Abs(MaxSlicesAbsRangeOrValue { count: RangeOrValue::Value(1000) })),
        };
        let freq_filter = config.freq_filter.as_ref().expect("find-and-select sample config should include freq_filter");

        let median_window = match &freq_filter[0] {
            AverageFilter(params) => &params.window,
            other => panic!("expected median filter in find-and-select sample config, got {other:?}"),
        };

        let resolve = match median_window {
            RangeOrValue::Dynamic(resolve) => resolve,
            other => panic!("expected dynamic median window in find-and-select sample config, got {other:?}"),
        };

        let resolved = resolve(&[NamedVal::new(
            FindAndSelectMotifsRunConfigFields::FindMotifsRunConfig(FindMotifsRunConfigFields::Window),
            Val::UInt(123),
        )]);

        assert_eq!(resolved, RangeOrValue::Range { range: Range { start: 61, end: 246 }, step: None });
    }

    #[test]
    fn test_run_all_find_motifs_config_grid_0_uses_captured_window_bounds() {
        let median_window = move |named_vals: &[NamedVal<FindMotifsRunConfigFields>]| {
            match named_vals.iter().find(|field| field.path().eq(&FindMotifsRunConfigFields::Window)) {
                None => RangeOrValue::Value(100),
                Some(value) => {
                    let window = value.val().to_i64(0) as u32;
                    let end = window.saturating_mul(3);
                    let start = window / 4;
                    RangeOrValue::Range { range: Range { start, end }, step: Some(((end-start)/7, 0)) }
                }
            }
        };
        let config = FindMotifsRunConfigRangeOrValue {
            id: None,
            step: RangeOrValue::Value(10),
            // window: RangeOrValue::Value(200),
            window: RangeOrValue::Range { range: Range { start: 10, end: 501 }, step: Some((50, 0)) },
            freq_filter: Some(vec![
                AverageFilter(AverageFilterParamsRangeOrValue {
                    window: RangeOrValue::dynamic(median_window),
                    step: RangeOrValue::Value(10),
                }),
            ]),
            finder_kind: FinderKindRangeOrValue::CrossLoader(CrossLoaderRunConfigRangeOrValue {
                similarity_metric: SimilarityMetricRangeOrValue::EucDist(EucDistConfigRangeOrValue { z_normalize: RangeOrValue::Value(false) }),
                slices_filter: SlicesFilterRangeOrValue::Accumulate(AccumulateStatsRangeOrValue {
                    aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue {
                        q: RangeOrValue::Value(0.05),
                        // q: RangeOrValue::Range { range: 0.0..0.1, step: Some((0.05, 2)) },
                    }),
                }),
            }),
            max_slices: Some(MaxSlicesRangeOrValue::Abs(MaxSlicesAbsRangeOrValue { count: RangeOrValue::Value(1000) })),
        };


        // let config = run_all_find_motifs_config_grid_0(3, 4, 7);
        let freq_filter = config.freq_filter.as_ref().expect("grid sample config should include freq_filter");

        let median_window = match &freq_filter[0] {
            InputFilterRangeOrValue::AverageFilter(params) => &params.window,
            other => panic!("expected median filter in grid sample config, got {other:?}"),
        };

        let resolve = match median_window {
            RangeOrValue::Dynamic(resolve) => resolve,
            other => panic!("expected dynamic median window in grid sample config, got {other:?}"),
        };

        let resolved = resolve(&[NamedVal::new(FindMotifsRunConfigFields::Window, Val::UInt(120))]);

        assert_eq!(resolved, RangeOrValue::Range { range: Range { start: 30, end: 360 }, step: Some((47, 0)) });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_import_resources() -> Result<(), ThoError> {
        let n = 1000i64;
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            test_dir,
            1,
            2 * n as u32,
            32000,
        )?));

        let store = Arc::new(SqlStore::in_mem().await?);
        // std::fs::remove_file("/tmp/test_main_db.db3").unwrap_or(());
        // let store = Arc::new(SqlStore::from_path("/tmp/test_main_db.db3")?);

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
            None,
            true,
            |t: &Transaction|{
                Ok(())
            },
        ).await?;

        let imported_resources: Vec<Resource> = store.list_resources_by_ids(&imported_ids).await?;

        debug!("imported={imported_resources:?}");
        let mut imported_keys = imported_resources
            .into_iter()
            .map(|r| r.file_stem().to_string())
            .collect::<Vec<_>>();
        imported_keys.sort();

        assert_eq!(vec!["t_0_0", "t_0_1", "t_1_0", "t_1_1",], imported_keys);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_import_resources_applies_freq_filter() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            test_dir,
            1,
            100,
            32000,
        )?));
        let store = Arc::new(SqlStore::in_mem().await?);
        let freq_filter = Arc::new(FreqFilter::<RunCtxImpl>::new(
            &[InputFilter::MedianFilter(MedianFilterParams { window: 5 })],
            &ws,
            0,
        )?);

        let imported_ids = import_resources(
            &ws,
            &store,
            &["label_0"],
            || Ok(HashMap::from([("t_0_0".to_string(), vec![("label_0".to_string(), 1.0)])])),
            |_: String| Ok(Tensor::from_slice(&[0., 100., 0.]).view((1, -1))),
            Some(freq_filter),
            true,
            |t: &Transaction|{
                Ok(())
            }
        ).await?;

        let imported_resources: Vec<Resource> = store.list_resources_by_ids(&imported_ids).await?;
        let imported = &imported_resources[0];
        let imported_tensor = Tensor::load(
            ws.read()
                .tensors_path()
                .join(format!("{}_{}.pt", imported.import_id(), imported.file_stem())),
        )?;
        let expected = Tensor::zeros([1, 3], (imported_tensor.kind(), imported_tensor.device()));

        assert!(imported_tensor.allclose(&expected, 1e-5, 1e-5, false));

        Ok(())
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_resource_tensor_two_easy_motifs() -> Result<(), ThoError>{

        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let (pulse_triangle_t, intervals, _) = resource_tensor_two_easy_motifs(
            1000,
            1,
            100,
            2,
            MotifKind::Pulse,
            2,
            MotifKind::Triangle,
            100,
            0.01,
            0.2,
            None
        )?;

        debug_plot!(&pulse_triangle_t, "pulse_triangle")?;

        let (pulse_sin_t, intervals, _) = resource_tensor_two_easy_motifs(
            1000,
            1,
            100,
            2,
            MotifKind::Pulse,
            2,
            MotifKind::Sin,
            100,
            0.01,
            0.2,
            None
        )?;
        debug_plot!(&pulse_sin_t, "pulse_sin")?;

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 10000)));

        for freq in 0..100 {
            let freq_filter = Arc::new(FreqFilter::<MockRunCtx>::new(&[
                            InputFilter::BandPassFilter(BandPassFilterParams {
                                low_hz: freq as f64,
                                band_hz: 1.0,
                                kernel_size: 100,
                                stride: 1,
                                padding: Padding::zero(),
                            })
                        ], &ws,
                                                                     0)?);

            let mut ctx = mock_run_ctx::default(1);

            let pulse_triangle_filtered_t = Arc::clone(&freq_filter).run(Arc::new(DevicesWrap::wrap(ThoTensor::new("".into(), pulse_triangle_t.shallow_clone())?)), ctx.branch(Some("filter"), false)).await?;
            let pulse_sin_filtered_t = freq_filter.run(Arc::new(DevicesWrap::wrap(ThoTensor::new("".into(), pulse_sin_t.shallow_clone())?)), ctx.branch(Some("filter"), false)).await?;

            with_data!(ctx.branch(Some("filter"), false), pulse_triangle_filtered_t, pulse_sin_filtered_t; |ctx, pulse_triangle_t:&ThoTensor<ChannelsData2, SynKey>, pulse_sin_t:&ThoTensor<ChannelsData2, SynKey> |{

                debug_plot!(pulse_triangle_t.as_ref(), "pulse_triangle_{freq}_filtered_t")?;
                debug_plot!(pulse_sin_t.as_ref(), "pulse_sin_{freq}_filtered_t")?;

                Ok::<_, ThoError>(())
            })?;
        }


        Ok(())
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

}
