use crate::cli::Commands::{
    FindMotifs, ImportResources, Infer, MotifsToTensor, PlotData, RunAll, SayHello,
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
use tch::{Device, IndexOp, Tensor};
use tch::kind::FLOAT_CPU;
use thoosa_common::tho_device::{ThoDevice};
use thoosa_common::dims::{Batch, ChannelsData, ThoBatch, TCD};
use thoosa_common::find_motifs_run_config::{AggrRangeOrValue, ByAvgEucDist, ByAvgEucDistRangeOrValue, AccumulateStatsRangeOrValue, CoherenceRunConfigRangeOrValue, ExtendMotifsConfigRangeOrValue, FindMotifsRunConfig, FindMotifsRunConfigRangeOrValue, InlierRatioRunConfigRangeOrValue, MeanAggrRangeOrValue, MotifCriteriaRangeOrValue, MotifSelector, MotifSelectorRangeOrValue, NormThicknessRunConfigRangeOrValue, PrecalculateStatsRangeOrValue, SelectMotifsRunConfigRangeOrValue, SlicesFilterRangeOrValue, QuantileAggrRangeOrValue, ComparisonMethodRangeOrValue, PearsonCorrConfigRangeOrValue, EucDistConfigRangeOrValue, MinAggrRangeOrValue, ExactStatsRangeOrValue, MaxSlicesRangeOrValue, MaxSlicesAbsRangeOrValue, InterestingAmp01RunConfig, InterestingAmp01RunConfigRangeOrValue, InterestingTurningPointsRatioRunConfig, InterestingTurningPointsRatioRunConfigRangeOrValue};
use thoosa_common::input_probs::InputProbs;
use thoosa_common::loader::{LoadRunner, SyncWrapper};
use thoosa_common::runs::{Empty, Labeled, Logits, NetResources, SavableResult, Simple};
use thoosa_common::train_nets;
use thoosa_common::train_nets::{load_final_model, opt_train, AttnPool1dArgsRangeOrValue, CommonTrainingRunConfig, CommonTrainingRunConfigRangeOrValue, ConvNetArgsRangeOrValue, ConvNetJointArgsRangeOrValue, FinalNetArgs, FinalNetArgsRangeOrValue, FinalNetParams, InferArgs, LabelNetParams, LabelNetTypeArgsRangeOrValue, LinNetArgs, LinNetArgsRangeOrValue, MinMaxArgsRangeOrValue, MotifCosineNetArgs, MotifCosineNetArgsRangeOrValue, MotifNetParams, MotifsBy as TrainMotifsBy, OptConfig, PlotVecf32, SelfAttentionNetArgs, SelfAttentionNetArgsRangeOrValue, TensorsDataBatch, ThoModuleWithStore, TrainFinalNetArgs, TrainFinalNetRunConfig, TrainFinalNetRunConfigFields, TrainFinalNetRunConfigRangeOrValue, TrainFinalNetsRunConfigResult, TrainLabelNetsArgs, TrainLabelNetsRunConfig, TrainLabelNetsRunConfigFields, TrainLabelNetsRunConfigRangeOrValue, TrainLabelNetsRunConfigResult, TrainMotifNetResult, TrainMotifsNetsArgs, TrainMotifsNetsResult, TrainMotifsNetsRunConfig, TrainMotifsNetsRunConfigFields, TrainMotifsNetsRunConfigRangeOrValue};
use thoosa_common::utils::{cuda_mem, from, lin_t, list_devices, plot_series, plot_series_with_sidebar_values, reset_print_options, DummyListener, TrainListener};
use thoosa_common::ws::{FsWorkspace, Workspace};
use thoosa_common::my_tch::ThoModule;
use thoosa_motifs::find;
use thoosa_motifs::find::{FindMotifsArgs, FixedFindMotifsConfig, SelectMotifsArgs};
use thoosa_motifs::fragments::{
    DataBy, MotifsToTensorArgs, PlotDataArgs};
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
use tokio::runtime::{Handle, Runtime};
use tokio::{select, sync::watch, time::{self, Duration}};
use tokio::time::MissedTickBehavior;
use thoosa_common::tho_tensor::ThoTensor;
use torch_sys::at_grad_set_enabled;
use thoosa_common::copy_ctx::CopyCtx;
use thoosa_common::find_motifs_run_config::InputFilterRangeOrValue::{BandPassFilter, SavGolFilter, NotchBandFilter};
use thoosa_common::transforms::filters::{BandPassFilterParamsRangeOrValue, NotchBandFilterParamsRangeOrValue, SavGolFilterParams, SavGolFilterParamsRangeOrValue, Padding, SgZeroPaddingRangeOrValue, PaddingRangeOrValue};
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, DevicesWrapCopyMode, WithDataRef};
use thoosa_shared::runner::cache_areas::CacheArea;
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::runner::common::runnable::AsyncStage;
use thoosa_shared::runner::global_ctx::GlobalCtx;
use thoosa_motifs::distance::average_euclidean_distance;
use thoosa_shared::colors_iter::DistinctColorIter;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::perf_logging::{format_perf_map, init_tracing, PERF};
use thoosa_shared::runner::oom_handler::{Adjuster, AdjusterError, Adjusters, AsyncOOMHandler, OOMHandler};
use thoosa_shared::test_utils::maybe_init_perf_tracing_stdout;
use crate::tasks::slices_stats::{Aggr, Metrics};
use crate::tasks::cross_dims::{ChannelsData2, ChannelsMotifsDists};
use crate::tasks::apply_motif_nets::apply_motif_net_forward;
use crate::tasks::find_and_select_motifs::{FindAndSelectMotifsInput, FindAndSelectMotifsRunConfig, FindAndSelectMotifsRunConfigRangeOrValue, FindAndSelectMotifsStage};
use crate::tasks::load_motif_net::MotifNetMeta;
use crate::tasks::motif_stats::{MotifStats, MotifStatsInput, MotifStatsOutput};
use crate::tasks::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta};
use crate::tasks::select_motifs::SelectMotifsStage;
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

    #[arg(long, default_value = "0.3")]
    pub max_motif_net_loss: f32, //FixedFindMotifsConfig::max_chunk_size

    #[arg(long, default_value = "100")]
    pub motif_slices_step: u32,

    #[arg(long, default_value = "false")]
    pub with_plots: bool,

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
}

#[derive(Debug, Args, Clone)]
pub struct SayHelloArgs {}

#[derive(Debug, Subcommand, Clone)]
pub enum Commands {
    ImportResources(ImportResourcesArgs),
    FindMotifs(FindMotifsArgs),
    PlotData(PlotDataArgs),
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
    // console_subscriber::init();
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

    let perf_path = if cli.write_perf_log_file {
        Some(
            ws.read()
                .log_path()
                .join(format!("thoosa_perf_{}.log", cdt_for_dir).to_owned()),
        )
    } else {
        None
    };


    let _logging_guards = init_tracing(
        cli.term_logger_level,
        cli.term_logger_extra_fields,
        cli.write_logger_level,
        cli.write_logger_extra_fields,
        logfile,
        "info",
        perf_path,
    )?;

    let store = Arc::new(SqlStore::from_path(ws.read().store_path()).await?);

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

                    info!(
                        "[tokio] alive_tasks={} global_q={} blocking_q={}",
                        metrics.num_alive_tasks(),
                        metrics.global_queue_depth(),
                        metrics.blocking_queue_depth(),
                    );

                    for w in 0..workers {
                           let poll = metrics.worker_poll_count(w);
                            let park = metrics.worker_park_count(w);
                            let busy = metrics.worker_total_busy_duration(w);

                            info!(
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
            ).await?;

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
                    ).await.expect("SelectMotifsStage::new failed"))
                },
                || root_ctx.branch( Some("select motifs"), true, false),
            ).await?;

            info!("SelectMotifs done, run_ids={run_ids:?}");
        }

        Some(TrainMotifsNets(train_motif_nets_args)) => {
            debug!("TrainMotifsNets: train_motif_nets_args.run_config_cnt={:?}",train_motif_nets_args.run_config_cnt);

            let train_motif_nets_run_config_range_or_value: TrainMotifsNetsRunConfigRangeOrValue = serde_json5::from_str(&train_motif_nets_args.run_config_cnt).expect("TrainMotifsNets2");
            let motifs_by = train_motif_nets_args.motifs_by()?;

            let input_wrapper = Arc::new(DevicesWrap::wrap(
                TrainMotifNetInput::new(motifs_by, Arc::new(train_motif_nets_args.fixed_config.clone())),
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

            info!("TrainMotifsNets: run_ids={run_ids:?}");
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
            let labels_filter = train_label_nets_args.labels_filter();

            let run_ids = run_opt::</*Labeled,*/ _, _, TrainLabelNetsRunConfig, _, _>(
                None,
                train_motif_nets_run_config_range_or_value,
                input_wrapper,
                &ws,
                &store,
                true,
                async |run_configs| {

                    new_train_label_nets(
                        train_label_nets_args.train_motif_nets_run_ids.clone(),
                        loader,
                        run_configs,
                        &ws,
                        &store,
                        init_ctx,
                        opt_config,
                        labels_filter.clone(),
                        train_label_nets_args.max_motif_net_loss,
                        train_label_nets_args.motif_slices_step.unwrap() //TODO or take from find motifs
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

                    new_train_final_net(
                        train_final_net_args.train_label_nets_run_ids.clone(),
                        loader,
                        run_configs,
                        &ws,
                        &store,
                        init_ctx,
                        opt_config,
                        10//TODO
                    ).await.map(|v|Arc::new(v))

                },
                || root_ctx.branch( Some("train final nets"), true, false),
            ).await?;


            info!("TrainFinalNet done");
        }

        Some(RunAll(run_all_args)) => {
            debug!("RunAll: run_all_params={:?}", run_all_args);

            let all_configs = run_all_config_0(run_all_args);

            ///////////////////////////////

            run_all(
                &ws,
                &store,
                &loader,
                root_ctx,
                run_all_args,
                all_configs
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

        Some(PlotData(plot_slices_params)) => {
            run_plot(&ws, &store, &loader, &plot_slices_params, &devices[0], root_ctx).await?;
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


    Ok(())
}

#[allow(unreachable_code)]
async fn run_all(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    loader: &Arc<RawTensorLoader<RunCtxWithPermits>>,
    mut root_ctx: RunCtxWithPermits,
    run_all_args: &RunAllArgs,
    all_config:AllConfigs
) -> Result<(), ThoError> {

    //gpt Build the input for FindMotifs: wrap resource config in DevicesWrap and allocate a fresh cache area.
    let find_motifs_input = Arc::new(DevicesWrap::wrap(
        ResourcesConfigInput::new(run_all_args.find_resource_config()?).check(&store).await?,
        CacheArea::default()
    ));

    //gpt Build the input for SelectMotifs: separate resource config so selection can use its own dataset.
    let select_motifs_input = Arc::new(DevicesWrap::wrap(
        ResourcesConfigInput::new(run_all_args.select_resource_config()?).check(&store).await?,
        CacheArea::default()
    ));

    //gpt Build the input for label training: this is typically a labeled resource set.
    let train_labels_input = Arc::new(DevicesWrap::wrap(
        ResourcesConfigInput::new( //TODO the group is needed to load resources not belonging to label being trained
               run_all_args.train_resource_config()
        ).check(&store).await?,
        CacheArea::default()
    ));

    //gpt Build the input for final net training: usually a higher-level resource config (e.g., logits inputs).
    let train_final_input = Arc::new(DevicesWrap::wrap(
        ResourcesConfigInput::new(
            run_all_args.train_final_resource_config()).check(&store).await?,
        CacheArea::default()
    ));

    let find_motifs_run_config_range_or_value_for_opt = all_config.find_motifs_run_config_range_or_value.clone();
    let select_motifs_run_config_for_opt = all_config.select_motifs_run_config.clone();
    let fixed_find_motifs_config_for_opt = all_config.fixed_find_motifs_config.clone();

    //gpt Pre-create context and shared handles for FindMotifs stage construction (kept outside the run_grid closure).
    let init_ctx_find_motifs = root_ctx.branch(Some("init find motifs"), true, false);
    let store_in_find_motifs = Arc::clone(&store);
    let ws_in_find_motifs = Arc::clone(&ws);
    let loader_in_find_motifs = Arc::clone(&loader);

    let find_and_select_input = Arc::new(DevicesWrap::wrap(
        FindAndSelectMotifsInput::new(
            run_all_args.find_resources.clone(),
            run_all_args.select_resources.clone(),
        ),
        CacheArea::default(),
    ));

    let find_and_select_run_config = FindAndSelectMotifsRunConfigRangeOrValue {
        id: None,
        find_motifs_run_config: find_motifs_run_config_range_or_value_for_opt,
        select_motifs_run_config: select_motifs_run_config_for_opt,
    };

    //TODO check if it's opt or grid and then check
    // check_opt_run_config::<_, FindAndSelectMotifsRunConfig>(&find_and_select_run_config)?;

    let select_motifs_run_ids = match run_opt::</*Simple,*/ _, _, FindAndSelectMotifsRunConfig, _, _>(
        None,
        find_and_select_run_config,
        find_and_select_input,
        &ws,
        &store,
        false,
        async |run_configs| {
            Ok(Arc::new(FindAndSelectMotifsStage::new(
                Arc::clone(&loader),
                &ws,
                &store,
                run_configs,
                fixed_find_motifs_config_for_opt,
                Arc::clone(&all_config.find_and_select_motifs_opt_config),
                run_all_args.max_chunk_size as i64,
            )))
        },
        || root_ctx.branch(Some("opt find and select motifs"), true, false),
    ).await {
        Ok(best_find_and_select_result_ids) => {
            info!("opt FindAndSelectMotifsRunConfig done: find_and_select_run_ids={best_find_and_select_result_ids:?}");
            assert_eq!(best_find_and_select_result_ids.len(), 1); //should be one from run_opt

            let read_guard = store;
            let select_motifs_run_ids = read_guard
                .list_input_run_ids(&best_find_and_select_result_ids)
                .await?
                .into_iter().map(
                    |opt_select_result_ids| opt_select_result_ids.1
                ).collect::<Vec<_>>();

            info!("opt FindAndSelectMotifsRunConfig done: select_motifs_run_ids={select_motifs_run_ids:?}");

            select_motifs_run_ids
        }
        Err(e) if matches!(e.what(), thoosa_shared::Cause::SomethingElse(msg) if msg == "no opt params in run config") => {
            info!("find and select motifs opt skipped: no opt params in run config, falling back to grid search");

            //gpt Run a grid over FindMotifs configs, constructing a stage per grid point and executing it.
            let find_motifs_run_ids = run_grid::</*Simple,*/ _, _, _, _, _>(
                None,
                all_config.find_motifs_run_config_range_or_value,
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
                            all_config.fixed_find_motifs_config,
                            ThoDevice::from(1),
                            //gpt TODO!! hard-coded device index assumes GPU 1 exists; consider selecting from args or ctx.
                            init_ctx_find_motifs
                        ).await.expect("FindMotifsStage::new failed"),
                        //gpt TODO!! `expect` will panic and bypass error propagation; consider returning the error instead.
                    )
                },
                || root_ctx.branch(Some("find_motifs"), true, false),
            ).await?;

            info!("find motifs done, find_motif_run_ids={find_motifs_run_ids:?}");

            //gpt Plot slices for each FindMotifs run (CPU plotting); useful for debugging but potentially expensive.
            // for find_motifs_run_id in &find_motifs_run_ids {
            //     run_plot_slices(&ws, &store, &loader, &PlotSlicesArgs::new(MotifsBy::RunId { find_motifs_run_id: *find_motifs_run_id, select_motifs_run_ids: None }), &Cpu).await?;
            // }

            debug!("starting select motifs: debug_semaphore={:?}", GlobalCtx::global().debug_semaphore(1));
            //gpt The select stage depends on the FindMotifs run ids; keep the ids to join results across runs.
            //gpt TODO!! if find_motifs_run_ids is empty, downstream stages will still run; should we short-circuit?

            //gpt Run a grid over SelectMotifs configs; each run consumes FindMotifs outputs and produces selected motifs.
            let mut select_motifs_run_ids = vec![];
            for find_motifs_run_id in &find_motifs_run_ids {
                select_motifs_run_ids.extend( run_grid::</*Labeled,*/ _, _, _, _, _>(
                    Some(vec![*find_motifs_run_id]),
                    all_config.select_motifs_run_config.clone(),
                    select_motifs_input.clone(),
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
                            vec![*find_motifs_run_id],
                            run_all_args.max_chunk_size as i64,
                            #[cfg(test)]
                            None
                        ).await.expect("SelectMotifsStage::new failed"))
                    },
                    || root_ctx.branch(Some("select_motifs"), true, false),
                ).await?);

            }

            info!("select motifs done: find_motifs_run_ids={find_motifs_run_ids:?} select_motifs_run_ids={select_motifs_run_ids:?}");

            select_motifs_run_ids
        }
        Err(e) => return Err(e),
    };

    // #[cfg(tho_debug_plots)]
    if run_all_args.with_plots {
        for select_motifs_run_id in &select_motifs_run_ids {
            run_plot(&ws, &store, &loader, &PlotDataArgs::new(DataBy::SelectedMotifs { select_motifs_run_ids: vec![*select_motifs_run_id] }), &Cpu, root_ctx.branch("plot".into(), false, false)).await?;
        }
    }

    debug!("starting train motif nets: debug_semaphore={:?}", GlobalCtx::global().debug_semaphore(1));

    info!("find and select motifs  done: select_motifs_run_ids={select_motifs_run_ids:?}");
    // return Ok(());
    //gpt Train motif nets needs a custom input wrapper (motif ids + opt config) rather than a resource config.
    let input_wrapper = Arc::new(DevicesWrap::wrap(
        TrainMotifNetInput::new(TrainMotifsBy::SelectMotifsRunIds { select_motifs_run_ids: select_motifs_run_ids.clone() }, Arc::clone(&all_config.train_opt_config)),
        CacheArea::default()
    ));

    //gpt Run a grid over motif-net training configs; output is a set of run ids for trained motif nets.
    let train_motif_nets_run_ids = run_opt::</*Labeled,*/ _, _, TrainMotifsNetsRunConfig, _, _>(
        None,
        all_config.train_motif_nets_run_config_range_or_value,
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
    let opt_config_in_train_labels = Arc::clone(&all_config.train_opt_config);
    let labels_filter = run_all_args.labels_filter();

    //gpt Run a grid over label-net training configs; consumes motif nets from previous stage.
    let train_label_nets_run_ids = run_opt::</*Labeled,*/ _, _, TrainLabelNetsRunConfig, _, _>(
        None,
        all_config.train_label_nets_run_config_range_or_value,
        train_labels_input,
        &ws,
        &store,
        true,
        async move |run_configs| {
            new_train_label_nets(
                train_motif_nets_run_ids,
                loader_in_train_labels,
                run_configs,
                &ws_in_train_labels,
                &store_in_train_labels,
                init_ctx_train_labels,
                opt_config_in_train_labels,
                labels_filter.clone(),
                Some(run_all_args.max_motif_net_loss), //to RunAll args,
                run_all_args.motif_slices_step
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
        all_config.train_final_net_run_config_range_or_value,
        train_final_input,
        &ws,
        &store,
        true,
        async move |run_configs| {
            new_train_final_net(
                train_label_nets_run_ids,
                loader_in_train_final,
                run_configs,
                &ws_in_train_final,
                &store_in_train_final,
                init_ctx_in_train_final,
                Arc::clone(&all_config.train_opt_config),
                10 //TODO
            ).await.map(|v| Arc::new(v))
        },
        || root_ctx.branch(Some("train final nets"), true, false),
    ).await?;
    Ok(())
}

struct AllConfigs {
    find_motifs_run_config_range_or_value: FindMotifsRunConfigRangeOrValue,
    fixed_find_motifs_config:FixedFindMotifsConfig,
    select_motifs_run_config:SelectMotifsRunConfigRangeOrValue,
    find_and_select_motifs_opt_config:Arc<OptConfig>,
    train_motif_nets_run_config_range_or_value:TrainMotifsNetsRunConfigRangeOrValue,
    train_label_nets_run_config_range_or_value:TrainLabelNetsRunConfigRangeOrValue,
    train_final_net_run_config_range_or_value:TrainFinalNetRunConfigRangeOrValue,
    train_opt_config:Arc<OptConfig>,
}

fn run_all_config_0(run_all_params: &RunAllArgs) -> AllConfigs {
    let find_motifs_run_config_range_or_value: FindMotifsRunConfigRangeOrValue = FindMotifsRunConfigRangeOrValue {
        id: None,
        step: RangeOrValue::Value(10), //RangeOrValue::Range { range: Range { start: 50, end: 100 }, step: Some((100, 0)) },
        window: RangeOrValue::Range { range: Range { start: 100, end: 2100 }, step: Some((200, 0)) }, //RangeOrValue::Value(1000),
        // window: RangeOrValue::Value(300), //RangeOrValue::Value(1000),
        // freq_filter: None,
        freq_filter: Some(vec![
            // BandPassFilter(BandPassFilterParamsRangeOrValue {
            //     // low_hz: RangeOrValue::Range { range: Range { start: 0., end: 201. }, step: Some((50., 1)) },
            //     low_hz:  RangeOrValue::Range { range: Range { start: 0., end: 200. }, step: None },//Some((10., 1))
            //     // band_hz: RangeOrValue::Value(10.),
            //     band_hz:  RangeOrValue::Value(1.0),
            //     // kernel_size: RangeOrValue::Value(100),
            //     kernel_size: RangeOrValue::Range { range: Range { start: 10, end: 1000 }, step: None },
            //     stride: RangeOrValue::Value(10),
            //     // stride: RangeOrValue::Range { range: Range { start: 2, end: 10 }, step: None },
            //     padding: SgPaddingRangeOrValue::Zero(SgZeroPaddingRangeOrValue {}),
            // }),
            SavGolFilter(SavGolFilterParamsRangeOrValue {
                window_len: RangeOrValue::Value(111),
                // poly_order: RangeOrValue::Range { range: Range { start: 1, end: 11 }, step: Some((3, 0)) },
                poly_order: RangeOrValue::Value(3),
                stride: RangeOrValue::Value(5),
                padding: PaddingRangeOrValue::Zero(SgZeroPaddingRangeOrValue {}),
            }),
            NotchBandFilter(NotchBandFilterParamsRangeOrValue {
                low_hz: RangeOrValue::Value(59.0),
                high_hz: RangeOrValue::Value(62.0),
            })
        ]),
        // slices_filter: SlicesFilterRangeOrValue::Precalculate(PrecalculateStatsRangeOrValue { aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue{ q: RangeOrValue::Value(0.0005) }), slices: RangeOrValue::Value(1000) }),//Some((0.05, 2))
        // slices_filter: SlicesFilterRangeOrValue::Precalculate(PrecalculateStatsRangeOrValue { aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue{ q: RangeOrValue::Range{range: 0.0..0.1, step: None} }), slices: RangeOrValue::Value(1000) }),//Some((0.05, 2))
        // slices_filter: SlicesFilterRangeOrValue::Precalculate(PrecalculateStatsRangeOrValue { aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue{ q: RangeOrValue::Range{range: 0.0..0.1, step: Some((0.05, 2))} }), slices: RangeOrValue::Value(1000) }),//Some((0.05, 2))
        // slices_filter: SlicesFilterRangeOrValue::Precalculate(PrecalculateStatsRangeOrValue { aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k:  RangeOrValue::Range{range: 0.01..0.3, step: None}  }), slices: RangeOrValue::Value(1000) }),//Some((0.05, 3))
        // slices_filter: SlicesFilterRangeOrValue::Accumulate(AccumulateStatsRangeOrValue { aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k:  RangeOrValue::Range{range: 0.01..0.3, step: None}  }) }),//Some((0.05, 2))
        // slices_filter: SlicesFilterRangeOrValue::Accumulate(AccumulateStatsRangeOrValue { aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k:  RangeOrValue::Value(0.02)  }) }),
        // slices_filter: SlicesFilterRangeOrValue::Accumulate(AccumulateStatsRangeOrValue { aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue{ q: RangeOrValue::Range{range: 0.0..0.1, step: None} }) }),//Some((0.05, 2))
        slices_filter: SlicesFilterRangeOrValue::Accumulate(AccumulateStatsRangeOrValue { aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue{ q: RangeOrValue::Value(0.07) }) }),//Some((0.05, 2))
        max_slices: Some(MaxSlicesRangeOrValue::Abs(MaxSlicesAbsRangeOrValue{count: RangeOrValue::Value(1000)})),
    };
    let fixed_find_motifs_config = FixedFindMotifsConfig::new(run_all_params.max_chunk_size, false);
    //////////////////////select
    let select_motifs_run_config = SelectMotifsRunConfigRangeOrValue {
        id: None,
        // extend_config: Some(ExtendMotifsConfigRangeOrValue {
        //     slices_selector_metric: ComparisonMethodRangeOrValue::EucDist(EucDistConfigRangeOrValue {  }),
        //     slices_selector_aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k: RangeOrValue::Value(1.0) }),
        // }),
        // extend_config: Some(ExtendMotifsConfigRangeOrValue {
        //     slices_selector_metric: ComparisonMethodRangeOrValue::EucDist(EucDistConfigRangeOrValue {  }),
        //     // slices_selector_metric: ComparisonMethodRangeOrValue::PearsonCorr(PearsonCorrConfigRangeOrValue {  }),
        //     // slices_selector_aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k: RangeOrValue::Value(0.5) }),
        //     slices_selector_aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue { q: RangeOrValue::Value(0.01) }),
        //     // slices_selector_aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k: RangeOrValue::Range{range: 0.8..1.2, step: Some((0.1, 1))} }),
        // }),
        extend_config: None,
        motif_criteria: MotifCriteriaRangeOrValue {
            norm_thickness: Some(NormThicknessRunConfigRangeOrValue {
                k: RangeOrValue::Value(0.0), //TODO not used RangeOrValue::Range{range: 1.0..10.0, step: None},
                eps: RangeOrValue::Value(1e-8)
            }),
            inlier_ratioRunConfig:None,
            // inlier_ratioRunConfig: Some(InlierRatioRunConfigRangeOrValue {
            //     k: RangeOrValue::Value(3.0),
            //     eps: RangeOrValue::Value(1e-8),
            // }),
            coherenceRunConfig:None,
            // coherenceRunConfig: Some(CoherenceRunConfigRangeOrValue {
            //     k: RangeOrValue::Value(10.0),
            //     eps: RangeOrValue::Value(1e-8),
            // }),
            interesting_amp_01: None,
            // interesting_amp_01: Some(InterestingAmp01RunConfigRangeOrValue {
            //     q_lo: RangeOrValue::Value(0.05),
            //     q_hi: RangeOrValue::Value(0.95),
            //     a0: RangeOrValue::Value(0.5),
            //     eps: RangeOrValue::Value(1e-8),
            // }),
            interesting_tv_norm_01: None,
            interesting_curvature_norm_01: None,
            interesting_turning_points_ratio: None,
            // interesting_turning_points_ratio: Some(InterestingTurningPointsRatioRunConfigRangeOrValue{ peak: RangeOrValue::Value(4), }),
            interesting_band_energy_ratio_01: None,
            interesting_entropy_gauss_01: None,
            interesting_entropy_band_01: None,
        },
        threshold: RangeOrValue::Value(0.73), //RangeOrValue::Range{range: 0.8..0.95, step: None}, //RangeOrValue::Value(0.8),
        min_slices: Some(RangeOrValue::Value(5)),
        max_slices: Some(RangeOrValue::Value(100)),
        extend_max_slices: Some(MaxSlicesRangeOrValue::Abs(MaxSlicesAbsRangeOrValue{count: RangeOrValue::Value(1000)})),
    };
    let find_and_select_motifs_opt_config = Arc::new(OptConfig::new(100, 10));

    /////////////////////train_motifs
    let common_training_args = CommonTrainingRunConfigRangeOrValue {
        cycles: RangeOrValue::Value(100),
        batch_size: RangeOrValue::Value(10),
        // lr: RangeOrValue::Value(0.0001),
        lr: RangeOrValue::Range{ range: Range { start: 0.00001, end: 0.001 }, step: None },
        patience: RangeOrValue::Value(10),
        min_delta: RangeOrValue::Value(1e-4_f32),
    };

    let train_motif_nets_run_config_range_or_value: TrainMotifsNetsRunConfigRangeOrValue = TrainMotifsNetsRunConfigRangeOrValue {
        id: None,
        common_training_args: common_training_args.clone(),
        // net: train_nets::MotifNetTypeArgsRangeOrValue::ConvNet(train_nets::MotifConvNetArgsRangeOrValue {
        //     num_layers: RangeOrValue::Range { range: Range { start: 3, end: 5 }, step: None },
        //     step: RangeOrValue::Value(10),
        //     channels: RangeOrValue::Value(32),
        //     kernel: RangeOrValue::Value(10),
        // }),

        // net: train_nets::MotifNetTypeArgsRangeOrValue::CosineNet(MotifCosineNetArgsRangeOrValue{
        //     step: RangeOrValue::Value(1),
        //     channels: RangeOrValue::Range { range: Range { start: 4, end: 32 }, step: None },
        //     kernel: RangeOrValue::Range { range: Range { start: 10, end: 100 }, step: None },
        // })

        net: train_nets::MotifNetTypeArgsRangeOrValue::ConvNet(train_nets::MotifConvNetArgsRangeOrValue {
            conv_layers: RangeOrValue::Range { range: Range { start: 2, end: 24 }, step: None },
            head_layers: RangeOrValue::Range { range: Range { start: 2, end: 24 }, step: None },
            step: RangeOrValue::Value(10),
            channels: RangeOrValue::Range{ range: Range {start: 8, end: 256}, step: None },
            // kernel: RangeOrValue::Value(10),
            kernel: RangeOrValue::Range{ range: Range {start: 10, end: 100}, step: None },
        })

    };

    //////////////////////train labels
    let train_label_nets_run_config_range_or_value: TrainLabelNetsRunConfigRangeOrValue = TrainLabelNetsRunConfigRangeOrValue {
        id: None,
        common_training_args: common_training_args.clone(),

        // net: LabelNetTypeArgsRangeOrValue::ConvNet(ConvNetArgsRangeOrValue {
        //     num_conv_layers: RangeOrValue::Range { range: Range { start: 4, end: 32 }, step: None },
        //     hidden_conv_channels: RangeOrValue::Range { range: Range { start: 8, end: 32 }, step: None },
        //     // hidden_conv_channels: RangeOrValue::Value(64),
        //     // num_head_layers: RangeOrValue::Value(10),
        //     num_head_layers: RangeOrValue::Range { range: Range { start: 8, end: 32 }, step: None },
        //     num_head_features: RangeOrValue::Range { range: Range { start: 8, end: 256 }, step: None },
        //     // num_head_features: RangeOrValue::Value(256),
        //     kernel: RangeOrValue::Value(100),
        //     // kernel: RangeOrValue::Range { range: Range { start: 10, end: 100 }, step: None },
        //     joint: ConvNetJointArgsRangeOrValue::AttnPool1d(AttnPool1dArgsRangeOrValue{ _dummy: 0 }),
        // }),
        net: LabelNetTypeArgsRangeOrValue::SelfAttentionNet(SelfAttentionNetArgsRangeOrValue {
            // d_model: RangeOrValue::Value(64),
            d_model_heads: RangeOrValue::Range { range: Range { start: 2, end: 4 }, step: None },
            n_heads: RangeOrValue::Range { range: Range { start: 4, end: 32 }, step: None },
            ff_mult: RangeOrValue::Value(4),
            // dropout: RangeOrValue::Value(0.1),
            dropout: RangeOrValue::Range { range: Range { start: 0.01, end: 0.1 }, step: None },
        })

    };
    //////////////////////train final
    let train_final_net_run_config_range_or_value: TrainFinalNetRunConfigRangeOrValue = TrainFinalNetRunConfigRangeOrValue {
        id: None,
        common_training_args,
        net: FinalNetArgsRangeOrValue {
            num_layers: RangeOrValue::Range { range: Range { start: 4, end: 32 }, step: None },
        },
    };
    let train_opt_config = Arc::new(OptConfig::new(100, 10));
    AllConfigs{find_motifs_run_config_range_or_value, fixed_find_motifs_config, select_motifs_run_config, find_and_select_motifs_opt_config, train_motif_nets_run_config_range_or_value, train_label_nets_run_config_range_or_value, train_final_net_run_config_range_or_value, train_opt_config}
}

fn run_all_config_test_1(run_all_params: &RunAllArgs) -> AllConfigs {
    let find_motifs_run_config_range_or_value: FindMotifsRunConfigRangeOrValue = FindMotifsRunConfigRangeOrValue {
        id: None,
        step: RangeOrValue::Value(1), //RangeOrValue::Range { range: Range { start: 50, end: 100 }, step: Some((100, 0)) },
        window: RangeOrValue::Value(100),
        // freq_filter: None,
        freq_filter: Some(vec![BandPassFilter(BandPassFilterParamsRangeOrValue {
            // low_hz: RangeOrValue::Range { range: Range { start: 0., end: 201. }, step: Some((50., 1)) },
            low_hz:  RangeOrValue::Range { range: Range { start: 0., end: 100. }, step: Some((10., 1)) },//
            // low_hz:  RangeOrValue::Value(0.0),//
            band_hz: RangeOrValue::Value(2.),
            // band_hz:  RangeOrValue::Range { range: Range { start: 5., end: 20. }, step: Some((5., 1)) },
            kernel_size: RangeOrValue::Value(10),
            stride: RangeOrValue::Value(1),
            padding: PaddingRangeOrValue::Zero(SgZeroPaddingRangeOrValue {}),
        })]),
        // slices_filter: SlicesFilterRangeOrValue::Accumulate(AccumulateStatsRangeOrValue { aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue{ k: RangeOrValue::Value(0.025) }) }),
        // slices_filter: SlicesFilterRangeOrValue::Accumulate(AccumulateStatsRangeOrValue { aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue{ k: RangeOrValue::Range{range: 0.005..0.1, step: Some((0.005, 3))} }) }),
        // slices_filter: SlicesFilterRangeOrValue::Accumulate(AccumulateStatsRangeOrValue { aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue{ q: RangeOrValue::Value(0.01) })}),
        // slices_filter: SlicesFilterRangeOrValue::Accumulate(AccumulateStatsRangeOrValue { aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue{ q: RangeOrValue::Range{range: 0.005..0.1, step: Some((0.005, 3))}})}),
        // slices_filter: SlicesFilterRangeOrValue::Precalculate(PrecalculateStatsRangeOrValue { aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue{ q: RangeOrValue::Range{range: 0.05..1.0, step: Some((0.05, 2))} }), slices: RangeOrValue::Value(100) }),
        slices_filter: SlicesFilterRangeOrValue::Precalculate(PrecalculateStatsRangeOrValue { aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue{ q: RangeOrValue::Value(0.05) }), slices: RangeOrValue::Value(100) }),
        // slices_filter: SlicesFilterRangeOrValue::Precalculate(PrecalculateStatsRangeOrValue{aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k: RangeOrValue::Range{range: 0.005..0.05, step: Some((0.005, 4))} }), slices: RangeOrValue::Value(100)}),
        // slices_filter: SlicesFilterRangeOrValue::Precalculate(PrecalculateStatsRangeOrValue{aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k:  RangeOrValue::Value(0.025) }), slices: RangeOrValue::Value(1000)}),
        // slices_filter: SlicesFilterRangeOrValue::Precalculate(PrecalculateStatsRangeOrValue{aggr: AggrRangeOrValue::Min(MinAggrRangeOrValue { k:  RangeOrValue::Range{range: 0.095..1.2, step: Some((0.005, 3))} }), slices: RangeOrValue::Value(1000)}),
        // slices_filter: SlicesFilterRangeOrValue::Exact(ExactStatsRangeOrValue{ euc_dist:  RangeOrValue::Range{range: 0.001..0.1, step: Some((0.001, 3))}  }),//Some((0.01, 2))
        // slices_filter: SlicesFilterRangeOrValue::Exact(ExactStatsRangeOrValue{ euc_dist:  RangeOrValue::Range{range: 0.001..0.2, step: None}  }),//Some((0.01, 2))
        // slices_filter: SlicesFilterRangeOrValue::Exact(ExactStatsRangeOrValue{ euc_dist:  RangeOrValue::Value(0.01)  }),
        max_slices: Some(MaxSlicesRangeOrValue::Abs(MaxSlicesAbsRangeOrValue{count: RangeOrValue::Value(100)})),
    };
    let fixed_find_motifs_config = FixedFindMotifsConfig::new(run_all_params.max_chunk_size, false);
    //////////////////////select
    let select_motifs_run_config = SelectMotifsRunConfigRangeOrValue {
        id: None,
        // extend_config: Some(ExtendMotifsConfigRangeOrValue {
        //     slices_selector_metric: ComparisonMethodRangeOrValue::EucDist(EucDistConfigRangeOrValue {  }),
        //     // slices_selector_metric: ComparisonMethodRangeOrValue::PearsonCorr(PearsonCorrConfigRangeOrValue {  }),
        //     slices_selector_aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k: RangeOrValue::Value(0.5) }),
        //     // slices_selector_aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k: RangeOrValue::Range{range: 0.8..1.2, step: Some((0.1, 1))} }),
        // }),
        extend_config:None,
        motif_criteria: MotifCriteriaRangeOrValue {
            norm_thickness: Some(NormThicknessRunConfigRangeOrValue { k: RangeOrValue::Value(5.0), eps: RangeOrValue::Value(1e-8) }),
            inlier_ratioRunConfig: Some(InlierRatioRunConfigRangeOrValue {
                k: RangeOrValue::Value(3.0),
                eps: RangeOrValue::Value(1e-8),
            }),
            coherenceRunConfig: Some(CoherenceRunConfigRangeOrValue {
                k: RangeOrValue::Value(10.0),
                eps: RangeOrValue::Value(1e-8),
            }),
            interesting_amp_01: None,
            interesting_tv_norm_01: None,
            interesting_curvature_norm_01: None,
            interesting_turning_points_ratio: None,
            interesting_band_energy_ratio_01: None,
            interesting_entropy_gauss_01: None,
            interesting_entropy_band_01: None,
        },
        threshold: RangeOrValue::Value(0.8),
        // threshold: RangeOrValue::Range{range: 0.4..1.1, step: Some((0.1, 1))},
        min_slices: None,
        max_slices: None,
        extend_max_slices: Some(MaxSlicesRangeOrValue::Abs(MaxSlicesAbsRangeOrValue{count: RangeOrValue::Value(100)})),
    };
    /////////////////////train_motifs
    let common_training_args = CommonTrainingRunConfigRangeOrValue {
        cycles: RangeOrValue::Value(150),
        batch_size: RangeOrValue::Value(10),
        // lr: RangeOrValue::Value(0.0001),
        lr: RangeOrValue::Range{ range: Range { start: 0.00001, end: 0.001 }, step: None },
        patience: RangeOrValue::Value(10),
        min_delta: RangeOrValue::Value(1e-4_f32),
    };
    let find_and_select_motifs_opt_config = Arc::new(OptConfig::new(100, 10));


    let train_motif_nets_run_config_range_or_value: TrainMotifsNetsRunConfigRangeOrValue = TrainMotifsNetsRunConfigRangeOrValue {
        id: None,
        common_training_args: common_training_args.clone(),
        // net: train_nets::MotifNetTypeArgsRangeOrValue::ConvNet(train_nets::MotifConvNetArgsRangeOrValue { _dummy: 0 }),
        net: train_nets::MotifNetTypeArgsRangeOrValue::CosineNet(MotifCosineNetArgsRangeOrValue{
            step: RangeOrValue::Value(1),
            channels: RangeOrValue::Range { range: Range { start: 4, end: 32 }, step: None },
            kernel: RangeOrValue::Range { range: Range { start: 10, end: 100 }, step: None },
        }),
        // num_layers: RangeOrValue::Range { range: Range { start: 2, end: 8 }, step: None },

    };
    //////////////////////train labels
    let train_label_nets_run_config_range_or_value: TrainLabelNetsRunConfigRangeOrValue = TrainLabelNetsRunConfigRangeOrValue {
        id: None,
        common_training_args: common_training_args.clone(),

        // net: LabelNetTypeArgsRangeOrValue::ConvNet(ConvNetArgsRangeOrValue {
        //     // num_conv_layers: RangeOrValue::Range { range: Range { start: 3, end: 5 }, step: None },
        //     num_conv_layers: RangeOrValue::Value(8),
        //     hidden_conv_channels: RangeOrValue::Value(64),
        //     num_head_layers: RangeOrValue::Value(10),
        //     num_head_features: RangeOrValue::Value(256),
        //     kernel: RangeOrValue::Value(10),
        //     // kernel: RangeOrValue::Range { range: Range { start: 10, end: 100 }, step: None },
        //     // joint: ConvNetJointArgsRangeOrValue::MinMax(MinMaxArgsRangeOrValue{ _dummy: 0 }),
        //     joint: ConvNetJointArgsRangeOrValue::AttnPool1d(AttnPool1dArgsRangeOrValue{_dummy:0}),
        // }),

        net: LabelNetTypeArgsRangeOrValue::SelfAttentionNet( SelfAttentionNetArgsRangeOrValue {
            d_model_heads: RangeOrValue::Value(2),
            n_heads: RangeOrValue::Value(4),
            ff_mult: RangeOrValue::Value(2),
            dropout: RangeOrValue::Value(0.1),
        })
    };
    //////////////////////train final
    let train_final_net_run_config_range_or_value: TrainFinalNetRunConfigRangeOrValue = TrainFinalNetRunConfigRangeOrValue {
        id: None,
        common_training_args,
        net: FinalNetArgsRangeOrValue {
            num_layers: RangeOrValue::Range { range: Range { start: 3, end: 5 }, step: None },
        },
    };
    let train_opt_config = Arc::new(OptConfig::new(60, 10));

    AllConfigs{find_motifs_run_config_range_or_value, fixed_find_motifs_config, select_motifs_run_config, find_and_select_motifs_opt_config, train_motif_nets_run_config_range_or_value, train_label_nets_run_config_range_or_value, train_final_net_run_config_range_or_value, train_opt_config}
}

async fn save_run_configs<R: RunConfig + thoosa_properties_map::ToHashmap>(
    input_run_ids: Option<&[RunId]>,
    store: &Arc<Box<dyn Store>>,
    run_configs: &mut Vec<R>,
    descr: &str,
) -> Result<Vec<RunId>, ThoError> {
    let mut run_ids = Vec::with_capacity(run_configs.len());
    for run_config in run_configs.iter_mut() {
        let run_config_id = store
            .save_run_config(
                input_run_ids,
                &<R>::type_name(),
                descr,
                &"todo",
                run_config.to_properties_iter(None).collect(),
            )
            .await?;
        run_config.set_id(run_config_id);
        run_ids.push(run_config_id);
    }
    Ok(run_ids)
}

pub async fn import_resources<F, R>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
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
    import_resources_with_pure_sets(ws, store, labels, labeled_resources, resource_to_tensor, insert_label, Some(&[(3, true), (10, true), (50, true), (100, true), (150, true), (10000, false)])).await
}

pub async fn import_resources_with_pure_sets<F, R>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
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
    let store_guard = store;

    if insert_label {
        for label in labels {
            store_guard.save_label(label).await?; //TODO merge with save_resources_labels
        }
    }

    let labeled_resources_map = labeled_resources()?;
    let ws_guard = ws.write();
    let import_id = store_guard.save_import(format!("Import on {}", cdt())).await?;

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

    let resource_ids = store_guard.save_resources_labels(&imported_tensors_stats, import_id).await?;

    info!(
        "Import id: {}, imported {} tensors to {:?}, failed {}",
        import_id,
        resource_ids.len(),
        ws_guard.tensors_path(),
        failed_tensors.len()
    );

    if let Some(pure_sets_lens) = pure_sets_lens {
        for pure_sets_len in pure_sets_lens {
            store_guard.save_pure_sets(pure_sets_len.0, pure_sets_len.1).await?;//
        }
    }


    // store_guard.save_pure_balanced_sets("A10", Some(10), 0.6, 0.2, &t)?;
    // store_guard.save_pure_balanced_sets("B10", Some(10), 0.6, 0.2, &t)?;
    // store_guard.save_pure_balanced_sets("C10", Some(10), 0.6, 0.2, &t)?;
    // store_guard.save_pure_balanced_sets("A100", Some(100), 0.6, 0.2, &t)?;
    // store_guard.save_pure_balanced_sets("B100", Some(100), 0.6, 0.2, &t)?;
    // store_guard.save_pure_balanced_sets("C100", Some(100), 0.6, 0.2, &t)?;
    // store_guard.save_pure_balanced_sets("All", None,  0.6, 0.2, &t)?;

    store_guard.save_groups_by_label_cnt(100).await?;

    trace!("import_resources: resource_ids={resource_ids:?}, failed_tensors={failed_tensors:?}");

    Ok(resource_ids) //for tests
}

pub async fn run_plot<C>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    loader: &Arc<RawTensorLoader<C>>,
    plot_params: &PlotDataArgs,
    _device: &Device,
    mut ctx:C
) -> Result<(), ThoError>
where
    C:RunCtx + Adjustable{


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

    match plot_params.motifs_by() {
        DataBy::Resource { resource_id, channels, freq_filters } => {
            debug!("run_plot: MotifsBy::Resource resource_id={resource_id} channels={channels:?} freq_filters={freq_filters:?}");
            plot_resource(ws, store, loader, resource_id, channels, freq_filters, ctx).await?;
            return Ok(());
        }
        DataBy::MotifNetApplied { resource_id, net_path } => {
            debug!("run_plot: MotifsBy::MotifNetApplied resource_id={resource_id} net_path={net_path:?}");
            plot_motif_net_applied(ws, store, loader, resource_id, net_path, ctx).await?;
            return Ok(());
        }
        DataBy::MotifId(motif_id) => {
            debug!("run_plot: MotifsBy::MotifId run_id={motif_id}");
            plot_motif(&ws, ctx, motif_id, &motif_loader_impl, clc_dist, None).await?;
        }
        DataBy::RunId {
            find_motifs_run_ids,
            select_motifs_run_ids
        } => {

            debug!("run_plot: MotifsBy::RunId find_run_ids={find_motifs_run_ids:?} select_run_ids={select_motifs_run_ids:?}");
            for find_motifs_run_id in find_motifs_run_ids {
                plot_found_motifs(
                    ws,
                    store,
                    find_motifs_run_id,
                    &motif_loader_impl,
                    &clc_dist,
                    select_motifs_run_ids.clone(),
                    ctx.branch(Some(&format!("find_motifs_run_{find_motifs_run_id}")), true, false),
                )
                .await?;
            }
        }

        DataBy::SelectedMotifs { select_motifs_run_ids } => {
            for select_motifs_run_id in select_motifs_run_ids {
                plot_selected_motifs(
                    ws,
                    store,
                    select_motifs_run_id,
                    &motif_loader_impl,
                    &clc_dist,
                    ctx.branch(Some(&format!("selected_motifs_run_{select_motifs_run_id}")), true, false),
                )
                .await?;
            }
        }

    }

    Ok(())
}

async fn plot_found_motifs<C>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    find_motifs_run_id: RunId,
    motif_loader_impl: &Arc<MotifLoader<RawTensorLoader<C>, C>>,
    clc_dist: &Arc<MotifStats<MotifLoader<RawTensorLoader<C>, C>, RawTensorLoader<C>, C>>,
    select_motifs_run_ids: Option<Vec<RunId>>,
    mut ctx: C,
) -> Result<(), ThoError>
where
    C: RunCtx + Adjustable,
{
    let motifs = store.list_motifs(find_motifs_run_id).await?;
    let motifs_len = motifs.len();
    let done_cnt = Arc::new(AtomicU32::new(0));

    let futures = motifs
        .into_iter()
        .map(|motif_id| {
            let done_cnt = Arc::clone(&done_cnt);
            let clc_dist = Arc::clone(clc_dist);
            let motif_loader_impl = Arc::clone(motif_loader_impl);
            let ctx = ctx.branch(Some(&format!("motif_{motif_id}")), true, false);
            let select_motifs_run_ids = select_motifs_run_ids.clone();
            async move {
                debug!("run_plot: motif_id={motif_id}");

                if let Err(e) = plot_motif(&ws, ctx, motif_id as usize, &motif_loader_impl, clc_dist, select_motifs_run_ids).await {
                    error!("motif {motif_id} failed: e={e:?}");
                };
                MotifLoader::<RawTensorLoader<RunCtxWithPermits>, RunCtxWithPermits>::cache().await.clear();

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
        "run_plot_slices",
    )
    .await;

    Ok(())
}

async fn plot_selected_motifs<C>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    select_motifs_run_id: RunId,
    motif_loader_impl: &Arc<MotifLoader<RawTensorLoader<C>, C>>,
    clc_dist: &Arc<MotifStats<MotifLoader<RawTensorLoader<C>, C>, RawTensorLoader<C>, C>>,
    mut ctx: C,
) -> Result<(), ThoError>
where
    C: RunCtx + Adjustable,
{
    let channels = ws.read().channels();
    debug!("run_plot: MotifsBy::GoodMotifs select_motifs_run_id={select_motifs_run_id} channels={channels}");

    let motifs = store
        .list_selected_motifs(select_motifs_run_id, channels)
        .await?;

    let motif_ids: Vec<usize> = motifs
        .into_iter()
        .flat_map(|channel| channel.into_iter().map(|(motif_id, _)| motif_id as usize))
        .collect();

    if motif_ids.is_empty() {
        warn!("run_plot: no motifs found for select_motifs_run_id={select_motifs_run_id}");
        return Ok(());
    }

    let motifs_len = motif_ids.len();
    let done_cnt = Arc::new(AtomicU32::new(0));
    let suffix = Some(vec![select_motifs_run_id]);
    let futures = motif_ids
        .into_iter()
        .map(|motif_id| {
            let done_cnt = Arc::clone(&done_cnt);
            let clc_dist = Arc::clone(clc_dist);
            let motif_loader_impl = Arc::clone(motif_loader_impl);
            let ws = Arc::clone(ws);
            let suffix = suffix.clone();
            let ctx = ctx.branch(Some(&format!("motif_{motif_id}")), true, false);
            async move {
                debug!("run_plot: motif_id={motif_id}");
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
        });

    let _ = collect_futures_with_progress(
        futures,
        motifs_len,
        crate::buffer_limits::CLI_RUN_PLOT_MOTIF_SLICES_BUFFER_UNORDERED_LIMIT,
        StreamBuffering::Unordered,
        "run_plot_slices",
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
            Arc::new(DevicesWrap::wrap(resource, CacheArea::default())),
            ctx.branch(Some("plot_resource_load"), false, false),
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
    suffix: &str,
    caption: &str,
    tensor: &ThoTensor<ChannelsData2, SynKey>,
    channels: Option<&Vec<u32>>,
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

    plot_series(
        plot_data,
        caption,
        (2000, 1000),
        &ws.read().plots_path().join("resource").join(format!("resource_{resource_id}_{suffix}.png")),
    )?;

    Ok(())
}

fn plot_single_line_tensor(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    path_suffix: &str,
    caption: &str,
    line_t: &Tensor,
) -> Result<(), ThoError> {
    let line_t = line_t.f_view([-1])?;
    let points = line_t
        .iter::<f64>()?
        .enumerate()
        .map(|(i, v)| (i as f32, v as f32))
        .collect::<Vec<_>>();

    plot_series(
        vec![("0".to_string(), BLACK, points)],
        caption,
        (2000, 1000),
        &ws.read().plots_path().join("resource").join(format!("{path_suffix}.png")),
    )?;

    Ok(())
}

async fn plot_resource<C>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    loader: &Arc<RawTensorLoader<C>>,
    resource_id: ResourceId,
    channels: Option<Vec<u32>>,
    freq_filters: Option<Vec<thoosa_common::find_motifs_run_config::InputFilter>>,
    mut ctx: C,
) -> Result<(), ThoError>
where
    C: RunCtx + Adjustable,
{
    let raw = load_resource_tensor_for_plot(store, loader, resource_id, ctx.branch(Some("plot_resource_load"), false, false)).await?;

    let raw_channels = channels.clone();
    let channels_frag = sanitize_plot_path_segment(&format!("{channels:?}", ));

    raw.with_data_ref(0, ctx.branch(Some("plot_resource_raw"), false, false), None, |tensor: &ThoTensor<ChannelsData2, SynKey>, _ctx| {
        plot_resource_channels::<C>(ws, resource_id, &format!("raw_{channels_frag}",), &format!("Resource {resource_id} raw channels: {raw_channels:?}"), tensor, raw_channels.as_ref())
    }).await?;

    if let Some(freq_filters) = freq_filters {
        let filter_frag = sanitize_plot_path_segment(&freq_filters.iter().map(|f|format!("{:?}", f)).collect::<String>());//TODO remove "*Params"
        let filtered = Arc::new(FreqFilter::<C>::new(freq_filters.as_slice(), ws, resource_id)?)
            .run(
                Arc::clone(&raw),
                ctx.branch(Some("plot_resource_freq_filter"), false, false),
            )
            .await?;

        let filtered_channels = channels.clone();
        filtered.with_data_ref(0, ctx.branch(Some("plot_resource_freq_filter_plot"), false, false), None, |tensor: &ThoTensor<ChannelsData2, SynKey>, _ctx| {
            plot_resource_channels::<C>(ws, resource_id, &format!("freq_filtered_{channels_frag}_{filter_frag}"), &format!("Resource {resource_id} channels: {filtered_channels:?} filters {freq_filters:?}"), tensor, filtered_channels.as_ref())
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
    mut ctx: C,
) -> Result<(), ThoError>
where
    C: RunCtx + Adjustable,
{
    let raw = load_resource_tensor_for_plot(store, loader, resource_id, ctx.branch(Some("plot_resource_motif_net_load"), false, false)).await?;

    // raw.with_data_ref(0, ctx.branch(Some("plot_resource_motif_net_raw_all"), false, false), None, |tensor: &ThoTensor<ChannelsData2, SynKey>, _ctx| {
    //     plot_resource_channels::<C>(ws, resource_id, "raw_all_channels", &format!("Resource {resource_id} raw (all channels)"), tensor, None)
    // }).await?;

    let (path_channel, motif_id) = FsWorkspace::parse_motif_path(&net_path)?;
    let (store_channel, find_motifs_run_id) = store.get_motif_meta(motif_id).await?;
    if path_channel as usize != store_channel {
        warn!("plot_motif_net_applied_variant: channel mismatch path_channel={path_channel} store_channel={store_channel} motif_id={motif_id} net_path={net_path:?}");
    }
    let channel = store_channel as u32;
    let channel_only = vec![channel];
    let net_path_sfx = sanitize_plot_path_segment(&net_path.to_string_lossy());

    raw.with_data_ref(0, ctx.branch(Some("plot_resource_motif_net_raw_channel"), false, false), None, |tensor: &ThoTensor<ChannelsData2, SynKey>, _ctx| {
        plot_resource_channels::<C>(ws, resource_id, &format!("motif_net_input_raw_ch_{channel}_{net_path_sfx}"), &format!("Resource {resource_id} motif-net input raw channel={channel} net={net_path:?}"), tensor, Some(&channel_only))
    }).await?;

    let find_motifs_run_config = FindMotifsRunConfig::from_properties_map_with_id(
        find_motifs_run_id,
        &store.list_run_config_values(find_motifs_run_id).await?,
    )?;

    let filtered_or_raw = if let Some(freq_filters) = find_motifs_run_config.freq_filter.clone() {
        let filtered = Arc::new(FreqFilter::<C>::new(freq_filters.as_slice(), ws, find_motifs_run_id)?)
            .run(
                Arc::clone(&raw),
                ctx.branch(Some("plot_resource_motif_net_prefilter"), false, false),
            )
            .await?;

        let channel_only_filtered = vec![channel];
        filtered.with_data_ref(0, ctx.branch(Some("plot_resource_motif_net_prefilter_channel"), false, false), None, |tensor: &ThoTensor<ChannelsData2, SynKey>, _ctx| {
            plot_resource_channels::<C>(ws, resource_id, &format!("motif_net_input_prefiltered_ch_{channel}_{net_path_sfx}"), &format!("Resource {resource_id} motif-net prefiltered channel={channel} find_run_id={find_motifs_run_id}"), tensor, Some(&channel_only_filtered))
        }).await?;

        filtered
    } else {
        Arc::clone(&raw)
    };

    let (motif_net_params, motif_net) = {
        let ws_guard = ws.read();
        ws_guard.load_motif_net(&net_path, &Cpu)?
    };
    let motif_slices_step = 5u32; //motif_net_params.stride() as u32;
    let motif_net = Arc::new(DataWithExtra::new(motif_net, MotifNetMeta::new(motif_id, channel, motif_net_params)));

    filtered_or_raw.with_data_ref(0, ctx.branch(Some("plot_resource_motif_net_apply"), false, false), None, {
        let motif_net = Arc::clone(&motif_net);
        move |tensor: &ThoTensor<ChannelsData2, SynKey>, _ctx| {
            let out = apply_motif_net_forward(
                mono_key!(format!("plot_motif_net_applied_resource_{resource_id}"), format!("motif_{motif_id}_channel_{channel}")) + tensor.id(),
                tensor,
                motif_net.as_ref(),
                motif_slices_step,
                tensor.id(),
            )?;

            plot_single_line_tensor(
                ws,
                &format!("resource_{resource_id}_motif_net_applied_{motif_id}_{channel}_{net_path_sfx}"),
                &format!("Resource {resource_id} motif-net output motif_id={motif_id} channel={channel} find_run_id={find_motifs_run_id} net={net_path:?}"),
                out.t_ref(),
            )?;

            Ok(())
        }
    }).await?;

    Ok(())
}

async fn plot_motif<C>(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    mut ctx: C,
    motif_id: usize,
    motif_loader_impl: &Arc<MotifLoader<RawTensorLoader<C>, C>>,
    clc_dist: Arc<MotifStats<MotifLoader<RawTensorLoader<C>, C>, RawTensorLoader<C>, C>>,
    select_motif_run_ids: Option<Vec<RunId>>,
) -> Result<(), ThoError>
where
    C:RunCtx + Adjustable{
    let select_motif_run_ids_sfx = select_motif_run_ids.as_ref().map(|v|v.iter().map(|v|format!("_{}", v)).collect::<String>()).unwrap_or("".to_string());
    let motif_loader_input = MotifLoaderInput::new(DataId(motif_id as u32), select_motif_run_ids);
    let motif_id_arc = Arc::new(DevicesWrap::wrap(motif_loader_input.clone(), CacheArea::default()));
    let metrics = Metrics::all();
    let clc_input = Arc::new(DevicesWrap::wrap(
        MotifStatsInput::from_motif_input(motif_loader_input, metrics.clone(), Aggr::Mean(1.0)),
        CacheArea::default(),
    ));

    let motif = Arc::clone(&motif_loader_impl).run_cached(
        Arc::clone(&motif_id_arc),
        ctx.branch(Some("plot_motif_loader"), false, false),
    )
    .await?;

    let metrics_for_plot: Vec<(String, f32)> = clc_dist
        .run/*_cached*/(clc_input, ctx.branch(Some("plot_motif_clc_dist"), false, false))
        .await?
        .with_data_ref(
            0,
            ctx.branch(Some("plot_motif_metrics_values"), false, false),
            None,
            move |stats: &MotifStatsOutput, _ctx| {
                metrics
                    .iter()
                    .map(|metric| Ok((format!("{metric:?}"), stats.channel_value(*metric, 0)?)))
                    .collect::<Result<Vec<_>, ThoError>>()
            },
        )
        .await?;
    let metrics_for_plot = Arc::new(metrics_for_plot);


    let metrics_for_plot_ref = Arc::clone(&metrics_for_plot);
    let r = motif.with_data_ref(0, ctx.branch(Some("plot_motif_data"), false, false), None, move |motif: &DataWithExtra<ThoTensor<_, DataId>, MotifMeta>, _ctx| {
            let run_id = motif.extra().find_motifs_run_id;
            let channel = motif.extra().channel;

            trace!("plot_motif: motif.t={}", motif.data().t_ref());
            let all_slices = motif.data().t_ref().unbind(0);
            let slices_len = all_slices.len();
            let mut color_iter = DistinctColorIter::new();
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

            plot_series_with_sidebar_values(
                plot_data,
                format!("Motif: {motif_id} run id: {run_id} channel: {channel} number of slices: {slices_len}").as_str(),
                (2600, 1200),
                &ws.read().plots_path().join(format!("{run_id}{select_motif_run_ids_sfx}")).join(format!("motif_{}.png", motif_id)),
                "Motif Metrics",
                metrics_for_plot_ref.as_slice(),
            )?;


            Ok(true)
        }).await?;
    Ok(())
}

pub async fn run_opt_step<RC, RCRV, CC, R, ST>(
    input_run_ids: Option<Vec<RunId>>,
    rcrv: RCRV,
    common_config: &CC,
    resource_config: &R,
    opt_step: ST,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
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
        &Arc<Box<dyn Store>>,
        &Arc<RwLock<LoadRunner<Tensor>>>,
        Vec<Device>,
    ) -> Result<Option<Vec<Box<dyn SavableResult>>>, ThoError>, //TODO remove Option & Vec
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<<RCRV as OptRunConfigSource>::P>>> + ToHashmap,
    RCRV: OptRunConfigSource,
    R: ResourcesConfig,
{
    let opt_run_configs_opt = opt_step(
        input_run_ids.as_ref(),
        rcrv.opt_run_configs()?,
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

            let opt_run_config_id = opt_run_config.save(store, ws).await?;

            opt_run_config_ids.push(opt_run_config_id);
        }
        Ok(opt_run_config_ids)
    } else {
        panic!("todo")
    }
}


pub async fn run_all_grid_step<RC, RCRV, R, FC, ST>(
    input_run_ids: Option<Vec<RunId>>,
    rcrv: RCRV,
    fixed_config: &FC,
    resource_config: &R,
    grid_step: ST,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
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
        &Arc<Box<dyn Store>>,
        &Arc<RwLock<LoadRunner<Tensor>>>,
        Vec<Device>,
    ) -> Result<Option<Vec<Box<dyn SavableResult>>>, ThoError>,
    RC: RunConfig
        + From<std::vec::IntoIter<NamedVal<<RCRV as GridRunConfigSource>::P>>>
        + thoosa_properties_map::ToHashmap,
    RCRV: GridRunConfigSource,
    R: ResourcesConfig,
{
    let grid_run_configs = rcrv.grid_run_configs()?;
    debug!("run_all_grid_step: grid_run_configs={grid_run_configs:?}");

    let mut run_configs: Vec<RC> = grid_run_configs.to_run_configs();
    if presave {
        save_run_configs(
            input_run_ids.as_ref().map(|v| v.as_slice()),
            &store,
            &mut run_configs,
            "find motifs",
        ).await?;
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

            let opt_run_config_id = opt_run_config.save(store, ws).await?;

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
    store: &Arc<Box<dyn Store>>,
    presave: bool,
    f: F,
    ctx_f: CtxF,
) -> Result<Vec<RunId>, ThoError>
where
    F: AsyncFnOnce(Vec<RC>) -> Arc<ST>,//TODO return Result
    ST: Stage + AsyncStage + Send + Sync, // + From<(Arc<RwLock<Box<dyn Workspace>>>, Arc<Box<dyn Store>>, Arc<RwLock<LoadRunner<Tensor>>>, Vec<RC>, FC)>,
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
    let grid_run_configs = rcrv.grid_run_configs()?;
    debug!("run_all_grid_step: grid_run_configs={grid_run_configs:?}");

    let mut run_configs: Vec<RC> = grid_run_configs.to_run_configs();
    info!("run_all_grid_step: run_configs.len={}", run_configs.len());
    debug!("run_all_grid_step: run_configs={run_configs:?}");

    let mut run_ids = if presave {
        save_run_configs(
            input_run_ids.as_ref().map(|v| v.as_slice()),
            &store,
            &mut run_configs,
            "find motifs",
        ).await?
    } else {
        vec![]
    };
    //gpt TODO!! run_ids only reflect pre-saved run_config ids; if presave is false this stays empty even though the stage output is saved below.

    let stage = f(run_configs).await;

    //gpt TODO!! Arc::into_inner will panic if the output Arc has multiple owners; consider a fallible path or cloning the inner data.
    let mut savable = Arc::into_inner(stage.run(input, ctx_f()).await?).expect("").take_data(0).await?;

    //gpt TODO!! savable.save returns a run_id, but it's ignored here; downstream uses presaved ids only.
    savable.save(store, ws).await?;


    Ok(run_ids)
}

pub async fn run_opt</*ResConf, */RCRV, F, RC, ST, CtxF>(

    input_run_ids: Option<Vec<RunId>>,
    rcrv: RCRV,
    // fixed_config: FC,
    input: Arc<DevicesWrap<ST::In<'static>>>,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
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
    ST: Stage + AsyncStage + Send + Sync, // + From<(Arc<RwLock<Box<dyn Workspace>>>, Arc<Box<dyn Store>>, Arc<RwLock<LoadRunner<Tensor>>>, Vec<RC>, FC)>,
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
    let opt_run_configs = rcrv.opt_run_configs()?;
    debug!("run_all_grid_step: run_opt={opt_run_configs:?}");

    // let mut run_configs: Vec<RC> = opt_run_configs.to_run_configs();
    //gpt TODO!! presave flag is currently unused in run_opt; run_ids always come from the stage output.
    let stage = init_stage(opt_run_configs).await?;

    //gpt TODO!! Arc::into_inner will panic if the output Arc has multiple owners; consider a fallible path or cloning the inner data.
    let mut savable = Arc::into_inner(stage.run(input, ctx_f()).await?).expect("").take_data(0).await?;

    let run_id = savable.save(store, ws).await?;


    Ok(vec![run_id])//TODO

}

async fn perf_unfold_concat_cuda_vs_cpu() -> Result<(), ThoError>{
    async fn load_tensors(dim_1: &i64, num_tensors: usize, cuda_first:bool) -> Result<u128, ThoError> {
        let mut tensors = Vec::with_capacity(num_tensors);

        let ctx = CopyCtx::instance(Cuda(0));

        let t_0 = Instant::now();
        for i in 0..num_tensors {


            let t = Tensor::f_stack(&[
                make_t(&dim_1, &(i as i64), 0.1, ctx, cuda_first).await?,
                make_t(&dim_1, &(i as i64), 0.2, ctx, cuda_first).await?,
                make_t(&dim_1, &(i as i64), 0.3, ctx, cuda_first).await?,
                make_t(&dim_1, &(i as i64), 0.4, ctx, cuda_first).await?,
                make_t(&dim_1, &(i as i64), 0.5, ctx, cuda_first).await?,
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

//to check if there are no `#field_ident field value Real(87.0) is not valid: ThoosaError { msg: "can't convert Real(87.0)", frame_0
fn check_opt_run_config<RCRV, RC>(find_and_select_run_config: &RCRV) -> Result<(), ThoError>
where
    RCRV: OptRunConfigSource,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<<RCRV as OptRunConfigSource>::P>>> + ToHashmap
{
    let opt_run_config = find_and_select_run_config.opt_run_configs()?;
    let const_vals = opt_run_config.grid_run_config_source().const_params();
    let grid_vals_vec = opt_run_config.grid_run_config_source().grid_to_vals();
    let opt_hps = opt_run_config.opt_params();

    for grid_vals in grid_vals_vec {
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
    use crate::cli::{import_resources, import_resources_with_pure_sets, run_all, run_all_config_0, run_all_config_test_1, run_all_grid_step, run_grid, run_opt_step, run_plot, Cli, RunAllArgs};
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
    use tch::Device::{Cpu, Cuda};
    use tch::display::{set_print_options, PrinterOptions};
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::{nn, Device, IndexOp, Tensor};
    use tch::Kind::Float;
    use thoosa_common::find_motifs_run_config::{AggrRangeOrValue, CoherenceRunConfigRangeOrValue, ComparisonMethodRangeOrValue, EucDistConfigRangeOrValue, ExtendMotifsConfigRangeOrValue, FindMotifsRunConfig, FindMotifsRunConfigRangeOrValue, InlierRatioRunConfigRangeOrValue, InputFilter, MaxSlicesAbsRangeOrValue, MaxSlicesRangeOrValue, MotifCriteriaRangeOrValue, NormThicknessRunConfigRangeOrValue, PrecalculateStatsRangeOrValue, QuantileAggrRangeOrValue, SelectMotifsRunConfigRangeOrValue, SlicesFilterRangeOrValue};
    use thoosa_common::find_motifs_run_config::InputFilterRangeOrValue::{BandPassFilter, NotchBandFilter};
    use thoosa_common::input_probs::{compare_input_probs, InputProbs};
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::my_tch::ThoModule;
    use thoosa_common::runs::{Empty, Labeled, Logits, NetResources, Simple};
    use thoosa_common::tch_test_utils;
    use thoosa_common::tch_test_utils::{resource_tensor_two_easy_motifs, Interval};
    use thoosa_common::tho_device::ThoDevice;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{lin_motif_node, CommonTrainingRunConfig, InferArgs, MotifCosineNetParams, MotifNetParams, MotifNetTypeParams, OptConfig, TrainFinalNetRunConfig, TrainFinalNetRunConfigRangeOrValue, TrainLabelNetsRunConfig, TrainLabelNetsRunConfigFields, TrainLabelNetsRunConfigRangeOrValue, TrainMotifsNetsRunConfig, TrainMotifsNetsRunConfigRangeOrValue};
    use thoosa_common::transforms::filters::{BandPassFilterParams, BandPassFilterParamsRangeOrValue, NotchBandFilterParamsRangeOrValue, Padding};
    use thoosa_common::debug_plot;
    use thoosa_common::utils::{from, lin_t, local_max_1d, plot_series, reset_print_options, t_0_0, t_0_1, t_1_0, t_1_1, three_sigs_tensor, to_vec_f32, white_tophat1d};
    use thoosa_common::ws::{mock_ws_simp, FsWorkspace, Workspace};
    use thoosa_motifs::distance::MaskMaker::LeQuantile;
    use thoosa_motifs::distance::clc_dist_d2_and_mask;
    use thoosa_motifs::find::{
        FixedFindMotifsConfig, SelectMotifsParams};
    use thoosa_motifs::fragments::{DataBy, PlotDataArgs};
    use thoosa_properties_map::ToHashmap;
    use thoosa_run_config_args::OptRunConfigSource;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::{Resource, ResourceId, ResourceStats, ResourcesConfig};
    use thoosa_shared::run_configs::RunConfig;
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::test_utils::{set_panic_hook, MotifKind};
    use thoosa_shared::{cdt, cdt_for_dir, mono_key, panic_to_err_async, random_saw, recurse_files, saw, saw_with_steps, sig_one_imp, sig_three_imp, sig_two_imp, RangeOrValue, ResourceType, TensorIdSliceIdx, ThoError};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::perf_logging::{format_perf_map, PERF};
    use thoosa_shared::runner::cache_areas::CacheArea;
    use thoosa_shared::runner::common::{mock_run_ctx, MockRunCtx, PermitsCtx, RunCtx};
    use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
    use thoosa_shared::runner::common::mock_run_ctx::MockRunCtxBuilder;
    use thoosa_shared::runner::common::runnable::AsyncStage;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_shared::runner::run_ctx_2::RunCtxWithPermits;
    use thoosa_shared::traits::{Data, DataId};
    use thoosa_shared::val::NamedVal;
    use thoosa_store::{MotifTrackerStore, SlicesMatch, SqlMotifTrackerStore, SqlStore, Store};
    use thoosa_with_data::with_data;
    use crate::tasks::slices_stats::{Aggr, Metrics};
    use crate::tasks::cross_dims::{ChannelsData2, MotifSlices};
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::find_and_select_motifs::{FindAndSelectMotifsRunConfig, FindAndSelectMotifsRunConfigRangeOrValue};
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
            true,
            false,
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
    async fn test_resource_tensor_two_easy_motifs() -> Result<(), ThoError>{

        GlobalCtx::init_devices(2);

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

            let pulse_triangle_filtered_t = Arc::clone(&freq_filter).run(Arc::new(DevicesWrap::wrap(ThoTensor::new("".into(), pulse_triangle_t.shallow_clone())?, CacheArea::default())), ctx.branch(Some("filter"), false, false)).await?;
            let pulse_sin_filtered_t = freq_filter.run(Arc::new(DevicesWrap::wrap(ThoTensor::new("".into(), pulse_sin_t.shallow_clone())?, CacheArea::default())), ctx.branch(Some("filter"), false, false)).await?;

            with_data!(ctx.branch(Some("filter"), false, false), pulse_triangle_filtered_t, pulse_sin_filtered_t; |ctx, pulse_triangle_t:&ThoTensor<ChannelsData2, SynKey>, pulse_sin_t:&ThoTensor<ChannelsData2, SynKey> |{

                debug_plot!(pulse_triangle_t.as_ref(), "pulse_triangle_{freq}_filtered_t")?;
                debug_plot!(pulse_sin_t.as_ref(), "pulse_sin_{freq}_filtered_t")?;

                Ok::<_, ThoError>(())
            })?;
        }


        Ok(())
    }


    fn tensor_1() -> Result<(Tensor, Vec<Vec<Vec<Interval>>>), ThoError> {
        // tch_test_utils::resource_tensor(1000usize, &[
        //     vec![(3, 100, MotifKind::Sin)],
        // ], 100, 10.)
        let r = resource_tensor_two_easy_motifs(
            rng().random_range(1000..2000),
            1,
            100,
            2,
            MotifKind::Sin,
            2,
            MotifKind::Triangle,
            100,
            0.01,
            0.2,
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
            rng().random_range(1000..2000),
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
        Ok((r.0, r.1))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 32)]
    async fn heavy_test_run_all() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        // log::set_max_level(log::LevelFilter::Trace);
        let global_ctx = GlobalCtx::init_devices(2);

        // let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let test_dir = Path::new("/tmp/heavy_test_run_all");
        if test_dir.is_dir() {
            fs::remove_dir_all(test_dir)?;
        }

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

            debug_plot!(&t, &stem)?;
            let tensors_path = ws.read().tensors_path().clone();
            t.save(tensors_path.join(&format!("{stem}.pt")))?;
            t.save(tensors_path.join(&stem))?;

            import_hash.insert(stem, vec![("label_0".to_string(), 1.0)]);

            let (t, _intervals) = tensor_1()?;

            let stem = format!("t_1_{i}");

            debug_plot!(&t, &stem)?;

            t.save(tensors_path.join(&format!("{stem}.pt")))?;
            import_hash.insert(stem, vec![("label_1".to_string(), 1.0)]);
        }


        let channels = 3;
        let tensors_path = ws.read().tensors_path().clone();
        let tensors_path_for_import = ws.read().tensors_path().clone();

        let mut store = Arc::new(SqlStore::from_path(ws.read().store_path()).await?);

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
        ).await?;

        let resources = store.list_resources_by_ids(&imported_ids).await?;

        let loader = Arc::new(RawTensorLoader::<RunCtxWithPermits>::new(
            move |key, _| {
                let tensor = Tensor::load(tensors_path.join(format!("{}.pt", key.file_stem())))?;
                debug!("test_motifs: key={key:?} loaded tensor={tensor:?}");
                Ok(ThoTensor::new(SynKey::new([mono_key!(key.id(), format!("resource_id={}", key.id()))]), tensor)?)
            },
            ThoDevice::from(Device::Cpu),
        ));

        let mut ctx = RunCtxWithPermits::new(1, "test_run_all");
        // ctx.add_permit(global_ctx.acquire_many(0, 1, Arc::new(SynKey::new([mono_key!("test", "test")]))).await?);


        let run_all_args = RunAllArgs{
            find_resources: "group:pure group 10".to_string(),
            select_resources: "group:pure group 20".to_string(),
            max_chunk_size: 100000,
            train_resource_group: "pure group 20".to_string(),
            valid_resource_group: "pure group 10".to_string(),
            run_config_cnt: "".to_string(),
            max_motif_net_loss: 0.4,
            motif_slices_step: 10,
            with_plots: false,
            labels: None,
        };

        let all_configs = run_all_config_test_1(&run_all_args);

        match panic_to_err_async!({run_all(
            &ws,
            &store,
            &loader,
            ctx,
            &run_all_args,
            all_configs,
        ).await}){
            Ok(_) => {
                debug!("ok");
            }
            Err(e) => {
                debug!("e={e:?}");
                panic!("test failed")
            }
        };

        info!("\n{}", format_perf_map(&PERF));

        global_ctx.shutdown().await;


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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_apply_moitf_net() -> Result<(), ThoError>{
        GlobalCtx::init_devices(2);
        let ws = Arc::new(RwLock::new(FsWorkspace::new("/home/bu/tmp/ws_test_hbac", 20, 684400, 200)?));
        let tensors_path = ws.read().tensors_path().clone();
        let store = Arc::new(SqlStore::from_path(ws.read().store_path()).await?);
        let mut ctx = RunCtxWithPermits::new(0, "run_plot_slices");

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


        let clc_dist = Arc::new(MotifStats::new(
            Arc::clone(&store),
            Arc::clone(&loader),
            Arc::clone(&ws),
            Arc::clone(&motif_loader_impl),
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
        debug_plot!(&channel_t.view((1, -1)), "resource_1_{resource}_{channel}")?;
        debug!("motif_t.data.size={:?}", motif_t.data().size());
        debug_plot!(&motif_t.data().as_ref().view((1, -1)), "motif_{motif_id}")?;



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
                debug_plot!(&out.t.view((1, -1)), "{motif_id}_applied_to_{resource}")?;
            }
        }



        Ok(())
    }

}
