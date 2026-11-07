use crate::dims::MotifSlices;
use crate::motifs_store::MotifsStore;
use crate::stages::convert_precision::{ConvertPrecisionArgs, ConvertPrecisionInput, ConvertPrecisionStage};
use crate::stages::cross_loader::ResourcesConfigInput;
use crate::stages::export::{Export as ExportStage, ExportArgs, ExportInput};
use crate::stages::find_motifs_run_config::{FindMotifsRunConfig, FindMotifsRunConfigFields, FindMotifsRunConfigRangeOrValue, SelectMotifsRunConfigRangeOrValue};
use crate::stages::purge_models::{PurgeModelsArgs, PurgeModelsInput, PurgeModelsStage};
use crate::stages::tensor_loader::RawTensorLoader;
use anyhow::Context as _;
use async_trait::async_trait;
use clap::{Args, Subcommand};
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use log::{debug, error, info, trace, warn};
use parking_lot::RwLock;
use plotters::prelude::BLACK;
use plotters::style::ShapeStyle;
use rayon::iter::{ParallelBridge, ParallelIterator};
use serde::Serialize;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::OpenOptions;
use std::io::Write;
use std::marker::PhantomData;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use tch::Device::Cpu;
use tch::{Device, IndexOp, Tensor};
use thoosa_common::dims::ChannelsData;
use thoosa_common::resources::{Labeled, Logits, NetResources, SavableResult, Simple};
use thoosa_common::runner::run_ctx::RunCtxImpl;
use thoosa_common::tch_error::tch_result;
use thoosa_common::tho_device::ThoDevice;
use thoosa_shared::traits::Stage;

use crate::find::{FindMotifsArgs, SelectMotifsArgs};
use crate::fragments::{DataBy, PlotDataArgs, Resolution};
use crate::motifs_ws::parse_motif_path;
use crate::stages::apply_motif_nets::apply_motif_net_forward;
use crate::stages::apply_motif_nets::{APPLY_MOTIF_NET_ENDPOINT, ApplyMotifNets, STACK_APPLIED_MOTIF_NETS_ENDPOINT, apply_motif_net_endpoint_init, stack_applied_motif_nets_init};
use crate::stages::find_and_select_motifs::FindAndSelectMotifsRunConfigFields;
use crate::stages::load_motif_net::MotifNetMeta;
use crate::stages::merge_selected_motifs::{MergeSelectedMotifs, MergeSelectedMotifsArgs, MergeSelectedMotifsInput};
use crate::stages::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta};
use crate::stages::motif_stats::{MOTIF_STATS_CONSUME_ENDPOINT, MOTIF_STATS_SYNC_ENDPOINT, MotifStats, MotifStatsInput, motif_stats_consume_stage_endpoint_init, motif_stats_sync_stage_endpoint_init};
use crate::stages::select_motifs::{SelectMotifsInput, SelectMotifsStage};
use crate::stages::seq_scan::SeqScanStage;
use crate::stages::slices_stats::Metrics;
use crate::stages::train_final_net::{SYNC_LOGITS_BATCH_ENDPOINT, new_train_final_net, sync_logits_batch_init, train_final_step_endpoint_inits};
use crate::stages::train_label_nets::{SYNC_LABEL_BATCH_ENDPOINT, TrainLabelNetRunConfig, TrainLabelNetRunConfigRangeOrValue, TrainLabelNetsArgs, new_train_label_nets, sync_label_batch_endpoint_init, train_label_step_endpoint_inits};
use crate::stages::train_motif_nets::{TrainMotifNetInput, TrainMotifNets, train_motif_step_endpoint_inits};
use crate::train_nets::{TrainFinalNetArgs, TrainFinalNetRunConfig, TrainFinalNetRunConfigRangeOrValue, TrainMotifNetRunConfig, TrainMotifNetRunConfigRangeOrValue, TrainMotifsNetsArgs};
use thoosa_common::args::StageCommand;
use thoosa_common::data_with_extra::DataWithExtra;
use thoosa_common::runner::common::cachable::{BorrowRef, Cached, DevicesWrap};
use thoosa_common::stages::freq_filter::{FreqFilter, SYNC_FREQ_FILTER_ENDPOINT, SyncFreqFilter, SyncFreqFilterIpcInit};
use thoosa_common::stages::seq_scan::SeqScanRunConfigRangeOrValue;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::utils::{load_run_config, plot_series, plot_series_with_sidebar_values};
use thoosa_properties_map::ToHashmap;

type MotifsInitFn = InitFn<dyn crate::motifs_ws::MotifsWorkspace>;
use crate::stages::apply_label_nets::{APPLY_LABEL_NET_ENDPOINT, STACK_APPLIED_LABEL_NETS_ENDPOINT, apply_label_net_endpoint_init, stack_applied_label_nets_init};
use crate::stages::cross_batch_clc::{CROSS_UNFOLD_CLC_ENDPOINT, sync_cross_unfold_clc_endpoint_init};
use crate::stages::prepare_cross_batch::{PREPARE_CROSS_BATCH_ENDPOINT, prepare_cross_batch_endpoint_init};
use thoosa_common::ipc::cuda_wokrer::{AllTensors, InitFn, WokerRunCtx};
use thoosa_common::ipc::serialization::json::JsonIpcCodec;
use thoosa_common::ipc::worker_msg::WorkerInput;
use thoosa_common::ipc::wraps::SyncStageIpcWrapper;
use thoosa_common::ipc::wraps::WorkerWrapperWithState;
use thoosa_common::nets::common_nets::OptConfig;
use thoosa_common::transforms::filters::InputFilter;
use thoosa_run_config_args::{GridRunConfigSource, OptRunConfigSource};
use thoosa_shared::collect_futures_eta::{CollectEtaInvocationGuard, collect_eta_record_completion};
use thoosa_shared::colors_iter::DistinctColorIter;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::hyperparameter::FieldName;
use thoosa_shared::resource::{ResourceId, ResourceStats};
use thoosa_shared::run_configs::{RunConfig, RunConfigId};
use thoosa_shared::tracked_await::t;
use thoosa_shared::traits::AsyncStage;
use thoosa_shared::traits::{Adjustable, MaxParallelCtx, RunCtx};
use thoosa_shared::val::NamedVal;
use thoosa_shared::{FuturesOrdering, RangeOrValue, collect_futures_best_effort_with_progress, mono_key, panic_to_err_async, parse_resources};
use thoosa_shared::{ResourceType, ThoError, cdt};
use thoosa_store::rusqlite::Transaction;
use thoosa_store::{SqlStore, Store};

// Keep malformed user configuration on the command's Result path and retain
// the parser's typed source together with the configuration being decoded.
fn parse_cli_config<T: serde::de::DeserializeOwned>(input: &str, name: &str) -> anyhow::Result<T> {
    serde_json5::from_str(input).with_context(|| format!("invalid {name} configuration"))
}

#[derive(Debug, Subcommand, Clone)]
pub enum MotifCommands {
    FindMotifs(FindMotifsArgs),
    SeqScan(SeqScanArgs),
    SelectMotifs(SelectMotifsArgs),
    MergeSelectedMotifs(MergeSelectedMotifsArgs),
    TrainMotifsNets(TrainMotifsNetsArgs),
    TrainLabelNets(TrainLabelNetsArgs),
    TrainFinalNet(TrainFinalNetArgs),
    Export(ExportArgs),
    ConvertPrecision(ConvertPrecisionArgs),
    PurgeModels(PurgeModelsArgs),
    PlotData(PlotDataArgs),
}

#[async_trait]
impl StageCommand for MotifCommands {
    type Store = Arc<dyn MotifsStore>;
    type Workspace = dyn crate::motifs_ws::MotifsWorkspace;
    type Ctx = RunCtxImpl;

    async fn run(self, store: &Arc<dyn MotifsStore>, ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, mut ctx: RunCtxImpl) -> anyhow::Result<()> {
        match self {
            MotifCommands::FindMotifs(find_motifs_args) => {
                debug!("MotifCommands::FindMotifs: args={find_motifs_args:?}");

                let find_motifs_run_config_range_or_value: FindMotifsRunConfigRangeOrValue = parse_cli_config(&find_motifs_args.run_config_cnt, "FindMotifsRunConfig")?;

                debug!("FindMotifs: find_motifs_run_config_range_or_value={find_motifs_run_config_range_or_value:?} find_motifs_args.fixed_config={:?}", find_motifs_args.fixed_config);

                let loader = init_loader(ws);

                let input_wrapper = Arc::new(ResourcesConfigInput::new(find_motifs_args.resource_config()));

                let init_ctx = ctx.branch(true);
                let init_loader = Arc::clone(&loader);
                let init_ws = Arc::clone(&ws);
                let init_store = Arc::clone(&store);
                let run_config_ids = t!(run_grid(
                    None,
                    find_motifs_run_config_range_or_value,
                    input_wrapper,
                    &ws,
                    &store,
                    async move |run_configs| { Ok(Arc::new(t!(crate::stages::find_motifs::FindMotifsStage::new(&init_ws, &init_store, init_loader, run_configs, find_motifs_args.fixed_config.clone(), init_ctx)).await?,)) },
                    || ctx.branch(true),
                    "find motifs",
                ))
                .await?;

                info!("FindMotifs done, run_config_ids={run_config_ids:?}");
                Ok(())
            }

            MotifCommands::SeqScan(seq_scan_args) => {
                debug!("SeqScan: seq_scan_args={seq_scan_args:?}");
                let loader = init_loader(ws);

                let seq_scan_run_config_range_or_value: SeqScanRunConfigRangeOrValue = parse_cli_config(&seq_scan_args.run_config_cnt, "SeqScanRunConfig")?;

                let input_wrapper = Arc::new(ResourcesConfigInput::new(seq_scan_args.resource_config()));

                let run_config_ids = t!(run_grid(
                    None,
                    seq_scan_run_config_range_or_value,
                    input_wrapper,
                    &ws,
                    &store,
                    |run_configs| async { Ok(Arc::new(SeqScanStage::<Simple, RawTensorLoader<RunCtxImpl>, RunCtxImpl>::new_with_chunk_size(&ws, &store, loader, run_configs, seq_scan_args.max_chunk_size).await?)) },
                    || ctx.branch(true),
                    "find motifs",
                ))
                .await?;

                info!("SeqScan done, run_config_ids={run_config_ids:?}");
                Ok(())
            }
            MotifCommands::SelectMotifs(select_motifs_args) => {
                debug!("SelectMotifs: select_motifs_args.run_config_cnt={:?}", select_motifs_args.run_config_cnt);
                let loader = init_loader(ws);

                let select_motifs_run_config_range_or_value: SelectMotifsRunConfigRangeOrValue = parse_cli_config(&select_motifs_args.run_config_cnt, "SelectMotifs2")?;

                let input_wrapper = Arc::new(SelectMotifsInput::standalone(ResourcesConfigInput::new(select_motifs_args.resource_config())));
                //TODO move it into SelectMotifsStage
                let motif_loader = MotifLoader::new(Arc::clone(&store), Arc::clone(&loader), &ws).await?;

                let run_config_ids = t!(run_grid::</*Labeled,*/ _, _, _, _, _>(
                    None,
                    select_motifs_run_config_range_or_value,
                    input_wrapper,
                    &ws,
                    &store,
                    |run_configs| async {
                        Ok(Arc::new(
                            t!(SelectMotifsStage::<RawTensorLoader<RunCtxImpl>, MotifLoader<RawTensorLoader<RunCtxImpl>, RunCtxImpl>, RunCtxImpl>::new(
                                &ws,
                                &store,
                                loader,
                                motif_loader,
                                run_configs,
                                select_motifs_args.find_motifs_run_config_ids.clone(), //TODO config
                                // #[cfg(test)]
                                None
                            ))
                            .await?,
                        ))
                    },
                    || ctx.branch(true),
                    "find motifs",
                ))
                .await?;

                info!("SelectMotifs done, run_config_ids={run_config_ids:?}");
                Ok(())
            }
            MotifCommands::MergeSelectedMotifs(merge_selected_motifs_args) => {
                debug!("MergeSelectedMotifs: merge_selected_motifs_args{:?}", merge_selected_motifs_args);
                let loader = init_loader(ws);

                let motif_stats = Arc::new(MotifStats::new(Arc::clone(&store), Arc::clone(&loader), Arc::clone(&ws), MotifLoader::new(Arc::clone(&store), Arc::clone(&loader), &ws).await?));

                let output = t!(Arc::new(MergeSelectedMotifs::new(&store, &ws, motif_stats)).run(Arc::new(MergeSelectedMotifsInput::new(merge_selected_motifs_args.select_motifs_run_config_ids.clone())), ctx.branch(true))).await?;

                info!("MergeSelectedMotifs done, output={output:?}");
                Ok(())
            }
            MotifCommands::TrainMotifsNets(train_motif_nets_args) => {
                debug!("TrainMotifsNets: train_motif_nets_args.run_config_cnt={:?}", train_motif_nets_args.run_config_cnt);

                let loader = init_loader(ws);

                let train_motif_nets_run_config_range_or_value: TrainMotifNetRunConfigRangeOrValue = parse_cli_config(&train_motif_nets_args.run_config_cnt, "TrainMotifsNets")?;
                let motifs_by = train_motif_nets_args.motifs_by()?;

                let input_wrapper = TrainMotifNetInput::new(motifs_by, Arc::new(train_motif_nets_args.opt_config.clone()), train_motif_nets_args.motifs_limit);

                let run_config_ids = t!(run_opt::<_, _, TrainMotifNetRunConfig, _, _>(
                    train_motif_nets_run_config_range_or_value,
                    input_wrapper,
                    &ws,
                    &store,
                    async |run_configs| { Ok(Arc::new(TrainMotifNets::<_, RunCtxImpl>::new(loader, &ws, &store, run_configs, train_motif_nets_args.epoch_size, None,).await?)) },
                    || ctx.branch(true),
                ))
                .await?;

                info!("TrainMotifsNets: run_config_ids={run_config_ids:?}");
                Ok(())
            }

            MotifCommands::TrainLabelNets(train_label_nets_args) => {
                debug!("TrainLabelNets: train_label_nets_args.run_config_cnt={:?}", train_label_nets_args.run_config_cnt);
                let loader = init_loader(ws);
                let train_motif_nets_run_config_range_or_value: TrainLabelNetRunConfigRangeOrValue = parse_cli_config(&train_label_nets_args.run_config_cnt, "TrainLabelNets")?;

                let input_wrapper = Arc::new(ResourcesConfigInput::new(train_label_nets_args.resource_config()));

                let opt_config = Arc::new(train_label_nets_args.opt_config.clone());
                let mut init_ctx = ctx.branch(true);
                let labels_filter = train_label_nets_args.labels_filter();

                let _run_config_ids = t!(run_opt::</*Labeled,*/ _, _, TrainLabelNetRunConfig, _, _>(
                    train_motif_nets_run_config_range_or_value,
                    input_wrapper,
                    &ws,
                    &store,
                    async |run_configs| {
                        let apply_motif_nets = Arc::new(
                            t!(ApplyMotifNets::new(
                                &ws,
                                &store,
                                train_label_nets_args.train_motif_nets_run_config_ids.clone(),
                                init_ctx.branch(false),
                                train_label_nets_args.max_motif_net_loss,
                                train_label_nets_args.selected_motif_nets_limit,
                                train_label_nets_args.motif_slices_step.unwrap_or(100), //TODO or take from find motifs
                                None,
                                false,
                            ))
                            .await?,
                        );

                        Ok(Arc::new(
                            t!(new_train_label_nets(
                                train_label_nets_args.train_motif_nets_run_config_ids.clone(),
                                loader,
                                apply_motif_nets,
                                run_configs,
                                &ws,
                                &store,
                                opt_config,
                                labels_filter.clone(),
                                init_ctx.branch(false)
                            ))
                            .await?,
                        ))
                    },
                    || ctx.branch(true),
                ))
                .await?;

                info!("TrainMotifsNets done");
                Ok(())
            }

            MotifCommands::TrainFinalNet(train_final_net_args) => {
                debug!("TrainFinalNet: train_motif_nets_args.run_config_cnt={:?}", train_final_net_args.run_config_cnt);
                let loader = init_loader(ws);
                let train_final_net_run_config_range_or_value: TrainFinalNetRunConfigRangeOrValue = parse_cli_config(&train_final_net_args.run_config_cnt, "TrainFinalNet")?;

                let input_wrapper = Arc::new(ResourcesConfigInput::new(train_final_net_args.resource_config()));

                let opt_config = Arc::new(train_final_net_args.opt_config.clone());
                let init_ctx = ctx.branch(true);
                let _run_config_ids = t!(run_opt::<_, _, TrainFinalNetRunConfig, _, _>(
                    train_final_net_run_config_range_or_value,
                    input_wrapper,
                    &ws,
                    &store,
                    async |run_configs| {
                        Ok(Arc::new(
                            t!(new_train_final_net(train_final_net_args.train_label_nets_run_config_ids.clone(), loader, run_configs, &ws, &store, init_ctx, opt_config, train_final_net_args.motif_slices_step,)).await?,
                        ))
                    },
                    || ctx.branch(true),
                ))
                .await?;

                info!("TrainFinalNet done");

                Ok(())
            }

            MotifCommands::Export(export_args) => {
                debug!("Export: export_args={export_args:?}");
                let output = t!(Arc::new(ExportStage::<RunCtxImpl>::new(&ws, store, export_args.clone())).run(ExportInput::new(), ctx.branch(true),)).await?;
                // let output = track_await(CachedStageOutput::take_data(output, 0)).await?;
                info!(
                    "Export done, target_dir={:?} label_nets={} motif_nets={} copied_models={} converted={}",
                    output.target_dir(),
                    output.label_net_run_config_ids().len(),
                    output.motif_net_run_config_ids().len(),
                    output.copied_models().len(),
                    output.converted().len()
                );

                Ok(())
            }
            MotifCommands::ConvertPrecision(convert_precision_args) => {
                debug!("ConvertPrecision: convert_precision_args={convert_precision_args:?}");
                let output = t!(Arc::new(ConvertPrecisionStage::<RunCtxImpl>::new(&ws, convert_precision_args.clone())).run(ConvertPrecisionInput::new(), ctx.branch(true),)).await?;
                // let output = track_await(CachedStageOutput::take_data(output, 0)).await?;
                info!("ConvertPrecision done, converted={}", output.converted().len());
                Ok(())
            }

            MotifCommands::PurgeModels(purge_models_args) => {
                debug!("PurgeModels: purge_models_args={purge_models_args:?}");
                let output = t!(Arc::new(PurgeModelsStage::<RunCtxImpl>::new(&ws, &store, purge_models_args.clone())).run(PurgeModelsInput::new(), ctx.branch(true),)).await?;
                // let output = track_await(CachedStageOutput::take_data(output, 0)).await?;
                info!("PurgeModels done, purged={}", output.purged().len());

                Ok(())
            }

            MotifCommands::PlotData(plot_slices_args) => {
                let loader = init_loader(ws);
                t!(run_plot(&ws, &store, &loader, &plot_slices_args, plot_slices_args.resolution(), &Cpu, ctx)).await?;

                Ok(())
            }
        }
    }
}

pub fn init_loader(ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>) -> Arc<RawTensorLoader<RunCtxImpl>> {
    let tensors_path = ws.read().tensors_path().clone();

    Arc::new(RawTensorLoader::<RunCtxImpl>::new(
        move |key, _| ThoTensor::new(SynKey::new([mono_key!(key.id(), format!("resource_id={}", key.id()))]), tch_result(Tensor::load(tensors_path.join(format!("{}_{}.pt", key.import_id(), key.file_stem()))))?),
        ThoDevice::from(0usize), //it's always cpu
    ))
}

/////////////////replaced by Hbac/Bird Args use for tests only
#[derive(clap::Subcommand, Clone, Debug)]
pub enum MotifAppCommands {
    #[command(flatten)]
    CommonCommands(CommonCommands),

    #[command(flatten)]
    Motifs(MotifCommands),
}

#[async_trait]
impl StageCommand for MotifAppCommands {
    type Store = Arc<dyn MotifsStore>;
    type Workspace = dyn crate::motifs_ws::MotifsWorkspace;
    type Ctx = RunCtxImpl;

    async fn run(self, service: &Arc<dyn MotifsStore>, ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, ctx: RunCtxImpl) -> anyhow::Result<()> {
        match self {
            MotifAppCommands::CommonCommands(c) => c.run(&(Arc::clone(&service) as Arc<dyn Store>), ws, ctx).await,
            MotifAppCommands::Motifs(c) => c.run(&(Arc::clone(&service) as Arc<dyn MotifsStore>), ws, ctx).await,
        }
    }
}

#[derive(clap::Subcommand, Clone, Debug)]
pub enum CommonCommands {
    SayHello(SayHelloArgs),
}

#[async_trait]
impl StageCommand for CommonCommands {
    type Store = Arc<dyn Store>;
    type Workspace = dyn crate::motifs_ws::MotifsWorkspace;
    type Ctx = RunCtxImpl;

    async fn run(self, _store: &Self::Store, _ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, _ctx: RunCtxImpl) -> anyhow::Result<()> {
        match self {
            CommonCommands::SayHello(args) => {
                debug!("CommonCommands::SayHello:run: args={args:?}");
            }
        }
        Ok(())
    }
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

    #[arg(long)]
    pub motifs_limit: Option<u32>,

    #[arg(long, default_value = "0.3")]
    pub max_motif_net_loss: f32, //filters motif nets for label training TODO move to train labels args

    #[arg(long)]
    pub selected_motif_nets_limit: Option<usize>, //filters motif nets for label training TODO move to train labels args

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
    pub fn find_resource_config(&self) -> anyhow::Result<Simple> {
        Ok(Simple::new(parse_resources(&self.find_resources).context("parse find_resources")?))
    }

    pub fn select_resource_config(&self) -> anyhow::Result<Simple> {
        Ok(Simple::new(parse_resources(&self.select_resources).context("parse select_resources")?))
    }

    pub fn train_resource_config(&self) -> NetResources<Labeled> {
        NetResources::new(Labeled::new(self.train_resource_group.clone()), Labeled::new(self.valid_resource_group.clone()))
    }

    pub fn train_final_resource_config(&self) -> NetResources<Logits> {
        NetResources::new(Logits::new(self.train_resource_group.clone()), Logits::new(self.valid_resource_group.clone()))
    }

    fn parse_labels_csv(raw: &str) -> Vec<String> {
        raw.split(',').map(|label| label.trim().to_string()).filter(|label| !label.is_empty()).collect()
    }

    pub fn labels_filter(&self) -> Option<Vec<String>> {
        self.labels.as_deref().map(Self::parse_labels_csv).filter(|labels| !labels.is_empty())
    }
}


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

pub fn sync_freq_filter_endpoint_init<'a>(_stage_id: Option<String>, init: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, _all_cuda_tensors: &'a AllTensors, _tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let init: Box<dyn Any> = init;
        let Ok(init) = init.downcast::<SyncFreqFilterIpcInit>() else {
            return Err(ThoError::new("SyncFreqFilter endpoint received an unexpected initializer type"));
        };
        let init = *init;
        let stage = SyncFreqFilter::<WokerRunCtx>::new_local(SyncFreqFilterIpcInit::new(init.filter_configs.clone(), init.num_channels, init.sample_rate, init.find_motifs_run_config_id, 1))?;

        // The common stateful wrapper prepares filter tensors on the worker device before each synchronous invocation.
        Ok(Box::new(WorkerWrapperWithState::new(Arc::new(stage), PhantomData::<JsonIpcCodec>)) as Box<dyn SyncStageIpcWrapper>)
    }
    .boxed()
}

pub fn cuda_worker_endpoint_inits() -> HashMap<String, MotifsInitFn> {
    let mut endpoints = train_motif_step_endpoint_inits();
    // Label and final training now execute through the same worker-owned proxy lifecycle as motif training.
    endpoints.extend(train_label_step_endpoint_inits());
    endpoints.extend(train_final_step_endpoint_inits());
    endpoints.extend([
        (PREPARE_CROSS_BATCH_ENDPOINT.to_string(), Box::new(prepare_cross_batch_endpoint_init) as MotifsInitFn),
        (CROSS_UNFOLD_CLC_ENDPOINT.to_string(), Box::new(sync_cross_unfold_clc_endpoint_init) as MotifsInitFn),
        (MOTIF_STATS_SYNC_ENDPOINT.to_string(), Box::new(motif_stats_sync_stage_endpoint_init) as MotifsInitFn),
        (MOTIF_STATS_CONSUME_ENDPOINT.to_string(), Box::new(motif_stats_consume_stage_endpoint_init) as MotifsInitFn),
        (SYNC_FREQ_FILTER_ENDPOINT.to_string(), Box::new(sync_freq_filter_endpoint_init) as MotifsInitFn),
        (APPLY_MOTIF_NET_ENDPOINT.to_string(), Box::new(apply_motif_net_endpoint_init) as MotifsInitFn),
        (APPLY_LABEL_NET_ENDPOINT.to_string(), Box::new(apply_label_net_endpoint_init) as MotifsInitFn),
        (STACK_APPLIED_MOTIF_NETS_ENDPOINT.to_string(), Box::new(stack_applied_motif_nets_init) as MotifsInitFn),
        (STACK_APPLIED_LABEL_NETS_ENDPOINT.to_string(), Box::new(stack_applied_label_nets_init) as MotifsInitFn),
        (SYNC_LABEL_BATCH_ENDPOINT.to_string(), Box::new(sync_label_batch_endpoint_init) as MotifsInitFn),
        (SYNC_LOGITS_BATCH_ENDPOINT.to_string(), Box::new(sync_logits_batch_init) as MotifsInitFn),
    ]);
    endpoints
}

fn resolve_median_window<P: FieldName + PartialEq>(named_vals: &[NamedVal<P>], window_path: &P) -> RangeOrValue<u32> {
    match named_vals.iter().find(|field| field.path() == window_path) {
        None => RangeOrValue::Value(100),
        Some(value) => {
            let window = value.val().to_i64(0) as u32;
            info!("Window: window={window}");
            // RangeOrValue::Value(window)
            RangeOrValue::Range {
                range: Range { start: window / 2, end: window * 2 },
                step: None,
            }
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
            RangeOrValue::Range {
                range: Range { start: window / 2, end: window * 2 },
                step: Some((window / 4, 0)),
            }
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
    resolve_median_window(named_vals, &FindAndSelectMotifsRunConfigFields::FindMotifsRunConfig(FindMotifsRunConfigFields::Window))
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
    dyn_step(named_vals, &FindAndSelectMotifsRunConfigFields::FindMotifsRunConfig(FindMotifsRunConfigFields::Window))
}

async fn save_run_configs<R: RunConfig + thoosa_properties_map::ToHashmap>(input_run_config_ids: Option<&[RunConfigId]>, store: &Arc<dyn MotifsStore>, run_configs: &mut Vec<R>, descr: &str) -> anyhow::Result<Vec<RunConfigId>> {
    let mut run_config_ids = Vec::with_capacity(run_configs.len());
    for run_config in run_configs.iter_mut() {
        let run_config_id = t!(store.save_run_config(input_run_config_ids, &<R>::type_name(), descr, run_config.to_properties_iter(None).collect(),)).await?;
        run_config.set_id(run_config_id);
        run_config_ids.push(run_config_id);
    }
    Ok(run_config_ids)
}
#[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "hbac/import_resources"))]
pub async fn import_resources<F, R, S>(
    ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
    store: &dyn thoosa_store::ExternalSqlWriter,
    labels: &[&str],
    labeled_resources: F,
    resource_to_tensor: R, //None for tests - do not import parquet files
    freq_filter: Option<Arc<SyncFreqFilter<RunCtxImpl>>>,
    insert_label: bool,
    init_data: S,
) -> anyhow::Result<Vec<ResourceId>>
where
    F: Fn() -> anyhow::Result<HashMap<String, Vec<(String, f32)>>>,
    R: Fn(String) -> anyhow::Result<Tensor> + Sync + Send,
    S: FnOnce(&Transaction) -> anyhow::Result<()>,
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
    let total_count = labeled_resources_map.iter().map(|e| e.1.len()).sum::<usize>();
    debug!("import_resources_with_pure_sets: labeled_resources_map={labeled_resources_map:?}");

    let ws_guard = ws.write();
    let import_descr = format!("Import on {}", cdt());
    let import_id = SqlStore::save_import(import_descr.as_str(), &t)?;

    let import_eta_guard = CollectEtaInvocationGuard::new("import", total_count, 1);
    let import_eta_guard_id = import_eta_guard.id();
    let (imported_tensors_stats, failed_tensors): (HashMap<String, (ResourceStats, Vec<(String, f32)>)>, Vec<String>) = labeled_resources_map
        .into_iter()
        .par_bridge()
        .fold(
            || (HashMap::new(), vec![]),
            |mut a, (resource_key, probs)| {
                trace!("import_resources: resource_key={resource_key} ok={} err={}", a.0.len(), a.1.len());

                match resource_to_tensor(resource_key.clone())
                    .and_then(|t| apply_import_freq_filter(freq_filter.as_deref(), resource_key.as_str(), t))
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

    let resource_ids = SqlStore::save_resources_labels(imported_tensors_stats.iter().map(|(resource_key, (stats, labels))| (resource_key.clone(), stats.length(), labels.clone())).collect(), import_id, &t)?;

    init_data(&t)?;

    t.commit()?;

    info!("Import id: {}, imported {} tensors to {:?}, failed {}", import_id, resource_ids.len(), ws_guard.tensors_path(), failed_tensors.len());

    Ok(resource_ids)
}

fn apply_import_freq_filter(freq_filter: Option<&SyncFreqFilter<RunCtxImpl>>, resource_key: &str, tensor: Tensor) -> anyhow::Result<Tensor> {
    match freq_filter {
        Some(freq_filter) => Ok(freq_filter
            .apply_tho_tensor(
                &ThoTensor::new(SynKey::new([mono_key!(resource_key.to_string(), format!("resource_key={resource_key}"))]), tensor)?, // .to_device(freq_filter.input_device())?,
            )?
            // .to_device(Cpu)?
            .into_local()?),
        None => Ok(tensor),
    }
}

pub async fn run_plot<C>(ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, store: &Arc<dyn MotifsStore>, loader: &Arc<RawTensorLoader<C>>, plot_params: &PlotDataArgs, resolution: Resolution, _device: &Device, mut ctx: C) -> anyhow::Result<()>
where
    C: RunCtx + MaxParallelCtx + Adjustable,
{
    let print_to_stdout = plot_params.print_to_stdout();

    let motif_loader_impl = MotifLoader::<RawTensorLoader<C>, C>::new(Arc::clone(store), Arc::clone(loader), ws).await?;

    let clc_dist = Arc::new(MotifStats::new(Arc::clone(store), Arc::clone(loader), Arc::clone(ws), Arc::clone(&motif_loader_impl)));

    match plot_params.data_by() {
        DataBy::Resource { resource_id, channels, freq_filters } => {
            debug!("run_plot: MotifsBy::Resource resource_id={resource_id} channels={channels:?} freq_filters={freq_filters:?}");
            t!(plot_resource(ws, store, loader, resource_id, channels, freq_filters, resolution, print_to_stdout, ctx, "")).await?;
            return Ok(());
        }
        DataBy::MotifNetApplied { resource_id, net_path } => {
            debug!("run_plot: MotifsBy::MotifNetApplied resource_id={resource_id} net_path={net_path:?}");
            t!(plot_motif_net_applied(ws, store, loader, resource_id, net_path, resolution, print_to_stdout, ctx)).await?;
            return Ok(());
        }
        DataBy::MotifId(motif_ids) => {
            debug!("run_plot: MotifsBy::MotifId motif_ids={motif_ids:?}");
            for motif_id in motif_ids {
                t!(plot_motif(&ws, ctx.branch(true), motif_id, &motif_loader_impl, Arc::clone(&clc_dist), None, resolution, print_to_stdout)).await?;
            }
        }
        DataBy::RunId {
            find_motifs_run_config_ids,
            select_motifs_run_config_ids,
        } => {
            debug!("run_plot: MotifsBy::RunId find_run_config_ids={find_motifs_run_config_ids:?} select_run_config_ids={select_motifs_run_config_ids:?}");
            for find_motifs_run_config_id in find_motifs_run_config_ids {
                if let Err(e) = t!(plot_found_motifs(ws, store, find_motifs_run_config_id, &motif_loader_impl, &clc_dist, select_motifs_run_config_ids.clone(), resolution, print_to_stdout, ctx.branch(true),)).await {
                    warn!("run_plot: failed with e={e:} on find_motifs_run_config_id={find_motifs_run_config_id}");
                }
            }
        }

        DataBy::SelectedMotifs { select_motifs_run_config_ids } => {
            for select_motifs_run_config_id in select_motifs_run_config_ids {
                if let Err(e) = t!(plot_selected_motifs(ws, store, select_motifs_run_config_id, &motif_loader_impl, &clc_dist, resolution, print_to_stdout, ctx.branch(true),)).await {
                    warn!("run_plot: failed with e={e:} on select_motifs_run_config_id={select_motifs_run_config_id}");
                }
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
        if value.is_finite() { Self::Finite(value) } else { Self::Special(value.to_string()) }
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

fn print_plot_data_to_stdout<S>(plot_data: Vec<(String, S, Vec<(f32, f32)>)>, caption: &str, size: (u32, u32), path: &impl AsRef<Path>, sidebar: Option<(&str, &[(String, f32)])>) -> anyhow::Result<()> {
    let dump = StdoutPlotDump {
        format: "thoosa_plot_data_v1",
        caption: caption.to_string(),
        target_path: path.as_ref().display().to_string(),
        size,
        series: plot_data
            .into_iter()
            .map(|(name, _, points)| StdoutPlotSeries {
                name,
                points: points.into_iter().map(|(x, y)| StdoutPlotPoint { x: x.into(), y: y.into() }).collect(),
            })
            .collect(),
        sidebar: sidebar.map(|(title, rows)| StdoutPlotSidebar {
            title: title.to_string(),
            rows: rows.iter().map(|(name, value)| StdoutPlotSidebarRow { name: name.clone(), value: (*value).into() }).collect(),
        }),
    };

    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &dump).with_context(|| "print plot data to stdout")?;
    writeln!(&mut stdout).with_context(|| "finish plot data stdout line")?;
    Ok(())
}

fn emit_or_plot_series<S>(print_to_stdout: bool, plot_data: Vec<(String, S, Vec<(f32, f32)>)>, caption: &str, size: (u32, u32), path: &impl AsRef<Path>) -> anyhow::Result<()>
where
    S: Into<ShapeStyle> + Clone,
{
    if print_to_stdout { print_plot_data_to_stdout(plot_data, caption, size, path, None) } else { plot_series(plot_data, caption, size, path) }
}

fn emit_or_plot_series_with_sidebar_values<S>(print_to_stdout: bool, plot_data: Vec<(String, S, Vec<(f32, f32)>)>, caption: &str, size: (u32, u32), path: &impl AsRef<Path>, sidebar_title: &str, sidebar_rows: &[(String, f32)]) -> anyhow::Result<()>
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
    ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
    store: &Arc<dyn MotifsStore>,
    find_motifs_run_config_id: RunConfigId,
    motif_loader_impl: &Arc<MotifLoader<RawTensorLoader<C>, C>>,
    clc_dist: &Arc<MotifStats<MotifLoader<RawTensorLoader<C>, C>, RawTensorLoader<C>, C>>,
    select_motifs_run_config_ids: Option<Vec<RunConfigId>>,
    resolution: Resolution,
    print_to_stdout: bool,
    mut ctx: C,
) -> anyhow::Result<()>
where
    C: RunCtx + MaxParallelCtx + Adjustable,
{
    let motifs = t!(store.list_motifs(find_motifs_run_config_id)).await?;
    let motifs_len = motifs.len();
    let done_cnt = Arc::new(AtomicU32::new(0));
    let ws = Arc::clone(ws);

    let futures = motifs.into_iter().map(|motif_id| {
        let done_cnt = Arc::clone(&done_cnt);
        let ws = Arc::clone(&ws);
        let clc_dist = Arc::clone(clc_dist);
        let motif_loader_impl = Arc::clone(motif_loader_impl);
        let ctx = ctx.branch(true);
        let select_motifs_run_config_ids = select_motifs_run_config_ids.clone();
        async move {
            debug!("run_plot: motif_id={motif_id}");

            let result = t!(plot_motif(&ws, ctx, motif_id as usize, &motif_loader_impl, clc_dist, select_motifs_run_config_ids, resolution, print_to_stdout)).await;
            t!(MotifLoader::<RawTensorLoader<RunCtxImpl>, RunCtxImpl>::cache()).await?.clear();

            let done = done_cnt.fetch_add(1, Relaxed);
            if done % 32 == 0 {
                info!("done {done} from  {motifs_len}");
            }

            result.map_err(|error| ThoError::from(error).context(format!("plot motif {motif_id}")))
        }
    });

    // Plot exports are independent; deliberately keep successful plots if another fails.
    let _ = t!(collect_futures_best_effort_with_progress(futures, motifs_len, FuturesOrdering::Unordered, &format!("plot_found_motifs_for_run_config_id_{find_motifs_run_config_id}"),)).await;

    Ok(())
}

async fn plot_selected_motifs<C>(
    ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
    store: &Arc<dyn MotifsStore>,
    select_motifs_run_config_id: RunConfigId,
    motif_loader_impl: &Arc<MotifLoader<RawTensorLoader<C>, C>>,
    clc_dist: &Arc<MotifStats<MotifLoader<RawTensorLoader<C>, C>, RawTensorLoader<C>, C>>,
    resolution: Resolution,
    print_to_stdout: bool,
    mut ctx: C,
) -> anyhow::Result<()>
where
    C: RunCtx + MaxParallelCtx + Adjustable,
{
    let channels = ws.read().channels();
    debug!("plot_selected_motifs: select_motifs_run_config_id={select_motifs_run_config_id} channels={channels}");

    let (_run_config_id, motifs) = t!(store.list_selected_motifs(&[select_motifs_run_config_id], channels, None)).await?.into_iter().next().context("no selected motif")?;

    debug!("plot_selected_motifs: motifs={motifs:?}");

    let motif_ids: Vec<usize> = motifs.into_iter().flat_map(|channel| channel.into_iter().map(|(motif_id, _)| motif_id as usize)).collect();

    if motif_ids.is_empty() {
        warn!("plot_selected_motifs: no motifs found for select_motifs_run_config_id={select_motifs_run_config_id}");
        return Ok(());
    }

    let motifs_len = motif_ids.len();
    let done_cnt = Arc::new(AtomicU32::new(0));
    let suffix = Some(vec![select_motifs_run_config_id]);
    let futures = motif_ids.into_iter().map(|motif_id| {
        let done_cnt = Arc::clone(&done_cnt);
        let clc_dist = Arc::clone(clc_dist);
        let motif_loader_impl = Arc::clone(motif_loader_impl);
        let ws = Arc::clone(ws);
        let suffix = suffix.clone();
        let ctx = ctx.branch(true);
        async move {
            debug!("plot_selected_motifs: motif_id={motif_id}");
            let result = t!(plot_motif(&ws, ctx, motif_id, &motif_loader_impl, clc_dist, suffix, resolution, print_to_stdout)).await;
            //MotifLoader::<RawTensorLoader<RunCtxImpl>, RunCtxImpl>::cache().await.clear();

            let done = done_cnt.fetch_add(1, Relaxed);
            if done % 32 == 0 {
                info!("done {done} from  {motifs_len}");
            }

            result.map_err(|error| ThoError::from(error).context(format!("plot motif {motif_id}")))
        }
    });

    // Plot exports are independent; deliberately keep successful plots if another fails.
    let _ = t!(collect_futures_best_effort_with_progress(futures, motifs_len, FuturesOrdering::Unordered, &format!("plot_selected_motifs_for_run_config_id_{select_motifs_run_config_id}"),)).await;

    Ok(())
}

async fn load_resource_tensor_for_plot<C>(store: &Arc<dyn MotifsStore>, loader: &Arc<RawTensorLoader<C>>, resource_id: ResourceId, mut ctx: C) -> anyhow::Result<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>
where
    C: RunCtx + MaxParallelCtx + Adjustable,
{
    let resource = t!(store.list_resources_by_ids(&vec![resource_id])).await?.into_iter().next().ok_or_else(|| anyhow::anyhow!("resource_id={resource_id} not found"))?;

    Ok(t!(Arc::clone(loader).run_cached(Arc::new(resource), ctx.branch(false),)).await?)
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

fn plot_resource_channels<C>(ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, resource_id: ResourceId, prefix: &str, suffix: &str, caption: &str, tensor: &ThoTensor<ChannelsData, SynKey>, channels: Option<&Vec<u32>>, resolution: Resolution, print_to_stdout: bool) -> anyhow::Result<()>
where
    C: RunCtx + MaxParallelCtx + Adjustable,
{
    let total_channels = tensor.size()[0] as u32;
    let channels_to_plot = channels.cloned().unwrap_or_else(|| (0..total_channels).collect());
    if channels_to_plot.is_empty() {
        return Err(anyhow::anyhow!("plot_resource_channels: no channels requested for resource_id={resource_id}"));
    }
    for channel in &channels_to_plot {
        if *channel >= total_channels {
            return Err(anyhow::anyhow!("plot_resource_channels: channel {channel} out of range [0,{total_channels}) for resource_id={resource_id}"));
        }
    }

    let mut color_iter = DistinctColorIter::new();
    let plot_data = channels_to_plot
        .into_iter()
        .map(|channel| {
            let channel_t = tensor.local()?.f_i((channel as i64, ..))?;
            Ok::<_, anyhow::Error>((format!("channel_{channel}"), color_iter.next().unwrap_or(BLACK), channel_t.iter::<f64>()?.enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>()))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    emit_or_plot_series(print_to_stdout, plot_data, caption, resolution.plot_size((2000, 1000)), &ws.read().plots_path().join("resource").join(format!("{prefix}resource_{resource_id}_{suffix}.png")))?;

    Ok(())
}

fn plot_single_line_tensor(ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, path_suffix: &str, caption: &str, line_t: &Tensor, resolution: Resolution, print_to_stdout: bool) -> anyhow::Result<()> {
    let line_t = line_t.f_view([-1])?;
    let points = line_t.iter::<f64>()?.enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<_>>();

    emit_or_plot_series(print_to_stdout, vec![("0".to_string(), BLACK, points)], caption, resolution.plot_size((2000, 1000)), &ws.read().plots_path().join("resource").join(format!("{path_suffix}.png")))?;

    Ok(())
}

pub async fn plot_resource<C>(
    ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
    store: &Arc<dyn MotifsStore>,
    loader: &Arc<RawTensorLoader<C>>,
    resource_id: ResourceId,
    channels: Option<Vec<u32>>,
    freq_filters: Option<Vec<InputFilter>>,
    resolution: Resolution,
    print_to_stdout: bool,
    mut ctx: C,
    prefix: &str,
) -> anyhow::Result<()>
where
    C: RunCtx + MaxParallelCtx + Adjustable,
{
    let raw = t!(load_resource_tensor_for_plot(store, loader, resource_id, ctx.branch(false))).await?;

    let raw_channels = channels.clone();
    let channels_frag = sanitize_plot_path_segment(&format!("{channels:?}",));

    let tensor = BorrowRef::borrow_ref(&raw, 0).await?;

    plot_resource_channels::<C>(
        ws,
        resource_id,
        prefix,
        &format!("raw_{channels_frag}",),
        &format!("Resource {resource_id} raw channels: {raw_channels:?}"),
        tensor.as_ref(),
        raw_channels.as_ref(),
        resolution,
        print_to_stdout,
    )?;

    if let Some(freq_filters) = freq_filters {
        let filter_frag = sanitize_plot_path_segment(&freq_filters.iter().map(|f| format!("{:?}", f)).collect::<String>()); //TODO remove "*Params"
        let configs = ws.read().ws_configs();
        let filtered = t!(FreqFilter::<C>::new(freq_filters.as_slice(), configs, resource_id, ctx.tho_device_idx()).await?.run(Arc::clone(&raw), ctx.branch(false),)).await?;

        let filtered_channels = channels.clone();
        let tensor = BorrowRef::borrow_ref(&filtered, 0).await?;

        plot_resource_channels::<C>(
            ws,
            resource_id,
            prefix,
            &format!("freq_filtered_{channels_frag}_{filter_frag}"),
            &format!("Resource {resource_id} channels: {filtered_channels:?} filters {freq_filters:?}"),
            tensor.as_ref(),
            filtered_channels.as_ref(),
            resolution,
            print_to_stdout,
        )?;
    }

    Ok(())
}

async fn plot_motif_net_applied<C>(ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, store: &Arc<dyn MotifsStore>, loader: &Arc<RawTensorLoader<C>>, resource_id: ResourceId, net_path: PathBuf, resolution: Resolution, print_to_stdout: bool, mut ctx: C) -> anyhow::Result<()>
where
    C: RunCtx + MaxParallelCtx + Adjustable,
{
    let raw = t!(load_resource_tensor_for_plot(store, loader, resource_id, ctx.branch(false))).await?;

    // raw.with_data_ref(0, ctx.branch(false), None, |tensor: &ThoTensor<ChannelsData2, SynKey>, _ctx| {
    //     plot_resource_channels::<C>(ws, resource_id, "raw_all_channels", &format!("Resource {resource_id} raw (all channels)"), tensor, None)
    // }).await?;

    let (path_channel, motif_id) = parse_motif_path(&net_path)?;
    let (store_channel, find_motifs_run_config_id) = t!(store.get_motif_meta(motif_id)).await?;

    let find_motifs_run_config: FindMotifsRunConfig = t!(load_run_config(store, find_motifs_run_config_id, FindMotifsRunConfig::type_name())).await?;

    if path_channel as usize != store_channel {
        warn!("plot_motif_net_applied_variant: channel mismatch path_channel={path_channel} store_channel={store_channel} motif_id={motif_id} net_path={net_path:?}");
    }
    let channel = store_channel as u32;
    let channel_only = vec![channel];
    let net_path_sfx = sanitize_plot_path_segment(&net_path.to_string_lossy());

    let tensor = BorrowRef::borrow_ref(&raw, 0).await?;

    plot_resource_channels::<C>(
        ws,
        resource_id,
        "",
        &format!("motif_net_input_raw_ch_{channel}"),
        &format!("Resource {resource_id} motif-net input raw channel={channel}"),
        tensor.as_ref(),
        Some(&channel_only),
        resolution,
        print_to_stdout,
    )?;

    let filtered_or_raw = if let Some(freq_filters) = find_motifs_run_config.freq_filter.clone() {
        let configs = ws.read().ws_configs();
        let filtered = t!(FreqFilter::<C>::new(freq_filters.as_slice(), configs, find_motifs_run_config_id, ctx.tho_device_idx()).await?.run(Arc::clone(&raw), ctx.branch(false),)).await?;

        let channel_only_filtered = vec![channel];

        {
            let tensor = BorrowRef::borrow_ref(&filtered, 0).await?;

            plot_resource_channels::<C>(
                ws,
                resource_id,
                "",
                &format!("motif_net_input_prefiltered_ch_{channel}_{net_path_sfx}"),
                &format!("Resource {resource_id} motif-net prefiltered channel={channel} find_run_config_id={find_motifs_run_config_id}"),
                tensor.as_ref(),
                Some(&channel_only_filtered),
                resolution,
                print_to_stdout,
            )?;
        }

        filtered
    } else {
        Arc::clone(&raw)
    };

    let (motif_net_params, motif_net) = {
        let ws_guard = ws.read();
        ws_guard.load_motif_net(&net_path, &Cpu, false, true)?
    };
    let motif_slices_step = 5u32; //motif_net_params.stride() as u32;
    let motif_net = Arc::new(DataWithExtra::new(motif_net.expect("motif_net"), MotifNetMeta::new(motif_id, channel, motif_net_params)));

    let tensor = BorrowRef::borrow_ref(&filtered_or_raw, 0).await?;

    let motif_net = Arc::clone(&motif_net);

    let _result = (|| {
        let out = apply_motif_net_forward(mono_key!(resource_id, format!("motif_{motif_id}_channel_{channel}")) + tensor.id(), tensor.as_ref(), motif_net.as_ref(), motif_slices_step, tensor.id())?;

        plot_single_line_tensor(
            ws,
            &format!("resource_{resource_id}_motif_net_applied_{motif_id}_{channel}_{net_path_sfx}"),
            &format!("Resource {resource_id} motif-net output motif_id={motif_id} channel={channel} find_run_config_id={find_motifs_run_config_id} net={net_path:?}"),
            &*out.local()?,
            resolution,
            print_to_stdout,
        )?;

        Ok::<_, anyhow::Error>(())
    })();

    Ok(())
}

pub async fn plot_motif<C>(
    ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
    mut ctx: C,
    motif_id: usize,
    motif_loader_impl: &Arc<MotifLoader<RawTensorLoader<C>, C>>,
    motif_stats: Arc<MotifStats<MotifLoader<RawTensorLoader<C>, C>, RawTensorLoader<C>, C>>,
    select_motif_run_config_ids: Option<Vec<RunConfigId>>,
    resolution: Resolution,
    print_to_stdout: bool,
) -> anyhow::Result<()>
where
    C: RunCtx + MaxParallelCtx + Adjustable,
{
    let select_motif_run_config_ids_sfx = select_motif_run_config_ids.as_ref().map(|v| v.iter().map(|v| format!("_{}", v)).collect::<String>()).unwrap_or("".to_string());
    let motif_loader_input = MotifLoaderInput::new(((motif_id as u32), "motif_id").into(), select_motif_run_config_ids);
    let motif_id_arc = Arc::new(motif_loader_input.clone());
    let metrics = Metrics::all();
    let clc_input = Arc::new(MotifStatsInput::from_motif_input(motif_loader_input, metrics.clone()));

    let motif: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>> = t!(Arc::clone(&motif_loader_impl).run_cached(Arc::clone(&motif_id_arc), ctx.branch(false),)).await?;

    let stats = t!(motif_stats.run(clc_input, ctx.branch(false))).await?;
    let metrics_for_plot: Vec<(String, f32)> = metrics.iter().map(|metric| Ok((format!("{metric:?}"), stats.channel_value(metric, 0)?))).collect::<anyhow::Result<Vec<_>>>().unwrap_or(vec![]);

    let metrics_for_plot = Arc::new(metrics_for_plot);

    let metrics_for_plot_ref = Arc::clone(&metrics_for_plot);
    let data_ref = BorrowRef::borrow_ref(&motif, ctx.tho_device_idx()).await?;
    let motif = data_ref.as_ref();
    let run_config_id = motif.extra().find_motifs_run_config_id;
    let channel = motif.extra().channel;

    let all_slices = thoosa_shared::CopyToDevice::copy_to_device(motif.data(), 0).await?.into_local()?;

    let _result = (move || {
        // trace!("plot_motif: motif.t={}", motif.data().t_ref());
        // let all_slices = motif.data().t_ref().unbind(0);
        let all_slices = all_slices.unbind(0);

        let slices_len = all_slices.len();
        let mut color_iter = DistinctColorIter::new();

        /*
        TODO param to enable stft
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
                true,
            );
            let spec_abs = spec.squeeze_dim(0).abs().transpose(0, 1);


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
        )?;*/

        let plot_data = all_slices
            .into_iter()
            .enumerate()
            .map(|(i, t)| Ok::<_, anyhow::Error>((format!("{i}"), color_iter.next().unwrap_or(BLACK), t.iter::<f64>()?.enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>())))
            .collect::<anyhow::Result<Vec<_>>>()?;

        emit_or_plot_series_with_sidebar_values(
            print_to_stdout,
            plot_data,
            format!("Motif: {motif_id} run id: {run_config_id} channel: {channel} number of slices: {slices_len}").as_str(),
            resolution.plot_size((2600, 1200)),
            &ws.read().plots_path().join(format!("{run_config_id}{select_motif_run_config_ids_sfx}")).join(format!("motif_{}.png", motif_id)),
            "Motif Metrics",
            metrics_for_plot_ref.as_slice(),
        )?;

        Ok::<_, anyhow::Error>(true)
    })();

    Ok(())
}

pub async fn run_grid<RCRV, F, RC, ST, CtxF>(input_run_config_ids: Option<Vec<RunConfigId>>, rcrv: RCRV, input: ST::In<'static>, ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, store: &Arc<dyn MotifsStore>, f: F, ctx_f: CtxF, descr: &str) -> anyhow::Result<Vec<RunConfigId>>
where
    F: AsyncFnOnce(Vec<RC>) -> anyhow::Result<Arc<ST>>,
    ST: Stage + AsyncStage + Send + Sync, // + From<(Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, Arc<dyn MotifsStore>, Arc<RwLock<LoadRunner<Tensor>>>, Vec<RC>, FC)>,
    RC: RunConfig + TryFrom<std::vec::IntoIter<NamedVal<<RCRV as GridRunConfigSource>::P>>, Error = anyhow::Error> + thoosa_properties_map::ToHashmap,
    RCRV: GridRunConfigSource + Debug,
    for<'c> <ST as Stage>::In<'c>: Send + Sync,
    for<'a> <ST as Stage>::Out: Send + Sync + SavableResult<Store = dyn MotifsStore, Workspace = dyn crate::motifs_ws::MotifsWorkspace, P<'a>: Default>,
    CtxF: FnOnce() -> <ST as Stage>::Ctx,
{
    debug!("run_all_grid_step: rcrv={rcrv:?}");

    let grid_run_configs = rcrv.grid_run_configs()?;
    debug!("run_grid: grid_run_configs={grid_run_configs:?}");

    let mut run_configs: Vec<RC> = grid_run_configs.to_run_configs()?;

    debug!("run_grid: run_configs={run_configs:?}");

    let run_config_ids = t!(save_run_configs(input_run_config_ids.as_ref().map(|v| v.as_slice()), &store, &mut run_configs, descr,)).await?;

    let stage = t!(f(run_configs)).await?;

    info!("run_grid: stage={stage:?}");

    // let mut savable = track_await(CachedStageOutput::take_data(track_await(stage.run(input, ctx_f())).await?, 0)).await?;
    let savable_r = t!(stage.run(input, ctx_f())).await;

    info!("run_grid: savable_result={savable_r:?}");

    let mut savable = savable_r?;

    info!("run_grid: savable={savable:?}");

    t!(savable.save(store, ws, <<ST as Stage>::Out as SavableResult>::P::default())).await?;

    Ok(run_config_ids)
}

pub async fn run_opt<RCRV, F, RC, ST, CtxF>(rcrv: RCRV, input: ST::In<'static>, ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, store: &Arc<dyn MotifsStore>, init_stage: F, ctx_f: CtxF) -> anyhow::Result<Vec<RunConfigId>>
where
    F: AsyncFnOnce(thoosa_shared::run_configs::TrainingRunConfigs<<RCRV as OptRunConfigSource>::P>) -> anyhow::Result<Arc<ST>>,
    ST: Stage + AsyncStage + Send + Sync, // + thoosa_common::runner::common::runnable::WrappedRunInputStage,
    RC: RunConfig + TryFrom<std::vec::IntoIter<NamedVal<<RCRV as OptRunConfigSource>::P>>, Error = anyhow::Error> + ToHashmap,
    RCRV: OptRunConfigSource,
    for<'c> <ST as Stage>::In<'c>: Send + Sync,
    for<'a> <ST as Stage>::Out: Send + Sync + SavableResult<Store = dyn MotifsStore, Workspace = dyn crate::motifs_ws::MotifsWorkspace, P<'a>: Default, R = Vec<RunConfigId>>,
    CtxF: FnOnce() -> <ST as Stage>::Ctx,
{
    let opt_run_configs = rcrv.opt_run_configs()?;
    debug!("run_all_grid_step: run_opt={opt_run_configs:?}");

    let stage = t!(init_stage(opt_run_configs)).await?;

    let mut savable = t!(stage.run(input, ctx_f())).await?;

    let run_config_id = panic_to_err_async!(savable.save(store, ws, <<ST as Stage>::Out as SavableResult>::P::default()))?;

    Ok(run_config_id)
}

//to check if there are no `#field_ident field value Real(87.0) is not valid: ThoosaError { msg: "can't convert Real(87.0)", frame_0

#[cfg(test)]
mod cli_tests {
    #[test]
    fn test_invalid_cli_configuration_preserves_parser_error() {
        let error = super::parse_cli_config::<std::collections::HashMap<String, u32>>("{count: 'bad'}", "test-command").unwrap_err();
        assert!(error.to_string().contains("test-command"));
        assert!(error.downcast_ref::<serde_json5::Error>().is_some());
    }

    use crate::cli::{MotifAppCommands, MotifCommands, import_resources, median_window_from_find_and_select};
    use crate::distance::{EucDistConfigRangeOrValue, SimilarityMetricRangeOrValue};
    use crate::fragments::DataBy;
    use crate::fragments::Resolution;
    use crate::motifs_store::MotifsStore;
    use crate::stages::find_and_select_motifs::FindAndSelectMotifsRunConfigFields;
    use crate::stages::find_motifs_run_config::{
        AccumulateStatsRangeOrValue, AggrRangeOrValue, CrossLoaderRunConfigRangeOrValue, FindMotifsRunConfigFields, FindMotifsRunConfigRangeOrValue, FinderKindRangeOrValue, MaxSlicesAbsRangeOrValue, MaxSlicesRangeOrValue, QuantileAggrRangeOrValue, SlicesFilterRangeOrValue,
    };
    use crate::stages::stft::SyncStftSelector;
    use crate::tch_test_utils::resource_tensor_two_easy_motifs;
    use clap::Parser;
    use log::debug;
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::ops::Range;
    use std::path::Path;
    use std::sync::Arc;
    use tch::Tensor;
    use thoosa_common::args::CommonArgs;
    use thoosa_common::debug_plot;
    use thoosa_common::runner::common::cachable::{DevicesWrap, SelectRef};
    use thoosa_common::runner::common::{MockRunCtx, mock_run_ctx};
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runner::run_ctx::RunCtxImpl;
    use thoosa_common::stages::freq_filter::{FreqFilter, SyncFreqFilter, SyncFreqFilterIpcInit};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::transforms::filters::InputFilterRangeOrValue::AverageFilter;
    use thoosa_common::transforms::filters::{AverageFilterParamsRangeOrValue, BandPassFilterParams, InputFilter, InputFilterRangeOrValue, MedianFilterParams, Padding};
    use thoosa_common::ws::WsConfigs;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::test_utils::TestSignalKind;
    use thoosa_shared::traits::AsyncStage;
    use thoosa_shared::traits::RunCtx;
    use thoosa_shared::val::{NamedVal, Val};
    use thoosa_shared::{RangeOrValue, ResourceType, cdt_for_dir};
    use thoosa_store::SqlStore;
    use thoosa_store::rusqlite::Transaction;

    #[derive(Parser, Debug)]
    #[command(version, about, long_about = None)]
    pub struct MotifAppArgs {
        #[command(flatten)]
        pub common_args: CommonArgs<MotifAppCommands>,
    }

    #[ctor::ctor]
    pub fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_write_perf_log_file_default_is_false() {
        let cli = MotifAppArgs::parse_from(["thoosa_motifs"]);
        assert!(!cli.common_args.write_perf_log_file);
    }

    #[test]
    fn test_write_perf_log_file_flag_enables_perf_file_logging() {
        let cli = MotifAppArgs::parse_from(["thoosa_motifs", "--write-perf-log-file"]);
        assert!(cli.common_args.write_perf_log_file);
    }

    #[test]
    fn test_plot_data_resolution_flag_parses_high() {
        let cli = MotifAppArgs::parse_from(["thoosa_motifs", "plot-data", "--data-by", "motif_id:7", "--resolution", "high"]);

        match cli.common_args.command {
            Some(MotifAppCommands::Motifs(MotifCommands::PlotData(args))) => assert_eq!(args.resolution(), Resolution::High),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn test_plot_data_stdout_flag_parses_true() {
        let cli = MotifAppArgs::parse_from(["thoosa_motifs", "plot-data", "--data-by", "motif_id:7", "--stdout"]);

        match cli.common_args.command {
            Some(MotifAppCommands::Motifs(MotifCommands::PlotData(args))) => assert!(args.print_to_stdout()),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn test_plot_data_motif_id_parses_multiple_ids() {
        let cli = MotifAppArgs::parse_from(["thoosa_motifs", "plot-data", "--data-by", "motif_id:1,2,3"]);

        match cli.common_args.command {
            Some(MotifAppCommands::Motifs(MotifCommands::PlotData(args))) => assert_eq!(args.data_by(), DataBy::MotifId(vec![1, 2, 3])),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn test_plot_data_motif_id_parses_spaced_quoted_ids() {
        let cli = MotifAppArgs::parse_from(["thoosa_motifs", "plot-data", "--data-by", "motif_id:'1, 2'"]);

        match cli.common_args.command {
            Some(MotifAppCommands::Motifs(MotifCommands::PlotData(args))) => assert_eq!(args.data_by(), DataBy::MotifId(vec![1, 2])),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn test_convert_precision_command_parses() {
        let cli = MotifAppArgs::parse_from(["thoosa_motifs", "convert-precision", "--nets-by", "run_config_id:1,2", "--target-precision", "bf16"]);

        match cli.common_args.command {
            Some(MotifAppCommands::Motifs(MotifCommands::ConvertPrecision(args))) => {
                assert_eq!(args.nets_by(), crate::stages::convert_precision::ConvertPrecisionBy::RunId(vec![1, 2]));
                assert_eq!(args.target_precision(), crate::stages::convert_precision::ConvertPrecisionTarget::Bf16);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn test_merge_selected_motifs_ids_parse_single_spaced_and_csv_values() {
        let single = MotifAppArgs::parse_from(["thoosa_motifs", "merge-selected-motifs", "--select-motifs-run-config-ids=1"]);
        match &single.common_args.command {
            Some(MotifAppCommands::Motifs(MotifCommands::MergeSelectedMotifs(args))) => assert_eq!(args.select_motifs_run_config_ids, vec![1]),
            other => panic!("unexpected command: {other:?}"),
        }

        let spaced = MotifAppArgs::parse_from(["thoosa_motifs", "merge-selected-motifs", "--select-motifs-run-config-ids='1, 2'"]);
        match &spaced.common_args.command {
            Some(MotifAppCommands::Motifs(MotifCommands::MergeSelectedMotifs(args))) => assert_eq!(args.select_motifs_run_config_ids, vec![1, 2]),
            other => panic!("unexpected command: {other:?}"),
        }

        let csv = MotifAppArgs::parse_from(["thoosa_motifs", "merge-selected-motifs", "--select-motifs-run-config-ids=1,2,3"]);
        match &csv.common_args.command {
            Some(MotifAppCommands::Motifs(MotifCommands::MergeSelectedMotifs(args))) => assert_eq!(args.select_motifs_run_config_ids, vec![1, 2, 3]),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn test_seq_scan_command_parses() {
        let cli = MotifAppArgs::parse_from(["thoosa_motifs", "seq-scan", "--resources", "group:find", "--run-config-cnt", "{step:10,window:100,metric:{MeanAbs:{center:false}},buckets_count:4}"]);

        match cli.common_args.command {
            Some(MotifAppCommands::Motifs(MotifCommands::SeqScan(args))) => {
                assert_eq!(args.resources, ResourceType::Group("find".to_owned()));
                assert_eq!(args.run_config_cnt, "{step:10,window:100,metric:{MeanAbs:{center:false}},buckets_count:4}");
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn test_run_all_config_0_preserves_dynamic_find_and_select_fields() {
        let config = FindMotifsRunConfigRangeOrValue {
            id: None,
            step: RangeOrValue::Value(10),

            window: RangeOrValue::Range {
                range: Range { start: 10, end: 501 },
                step: Some((50, 0)),
            },
            freq_filter: Some(vec![AverageFilter(AverageFilterParamsRangeOrValue {
                window: RangeOrValue::dynamic(median_window_from_find_and_select),
                step: RangeOrValue::Value(10),
            })]),
            finder_kind: FinderKindRangeOrValue::CrossLoader(CrossLoaderRunConfigRangeOrValue {
                similarity_metric: SimilarityMetricRangeOrValue::EucDist(EucDistConfigRangeOrValue { z_normalize: RangeOrValue::Value(false) }),
                slices_filter: SlicesFilterRangeOrValue::Accumulate(AccumulateStatsRangeOrValue {
                    aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue {
                        q: RangeOrValue::Range { range: 0.0..0.05, step: None },
                    }),
                }),
            }),
            max_slices: Some(MaxSlicesRangeOrValue::Abs(MaxSlicesAbsRangeOrValue { count: RangeOrValue::Value(1000) })),
            min_slices: Some(RangeOrValue::Value(2)),
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

        let resolved = resolve(&[NamedVal::new(FindAndSelectMotifsRunConfigFields::FindMotifsRunConfig(FindMotifsRunConfigFields::Window), Val::UInt(123))]);

        assert_eq!(resolved, RangeOrValue::Range { range: Range { start: 61, end: 246 }, step: None });
    }

    #[test]
    fn test_run_all_find_motifs_config_grid_0_uses_captured_window_bounds() {
        let median_window = move |named_vals: &[NamedVal<FindMotifsRunConfigFields>]| match named_vals.iter().find(|field| field.path().eq(&FindMotifsRunConfigFields::Window)) {
            None => RangeOrValue::Value(100),
            Some(value) => {
                let window = value.val().to_i64(0) as u32;
                let end = window.saturating_mul(3);
                let start = window / 4;
                RangeOrValue::Range {
                    range: Range { start, end },
                    step: Some(((end - start) / 7, 0)),
                }
            }
        };
        let config = FindMotifsRunConfigRangeOrValue {
            id: None,
            step: RangeOrValue::Value(10),
            // window: RangeOrValue::Value(200),
            window: RangeOrValue::Range {
                range: Range { start: 10, end: 501 },
                step: Some((50, 0)),
            },
            freq_filter: Some(vec![AverageFilter(AverageFilterParamsRangeOrValue {
                window: RangeOrValue::dynamic(median_window),
                step: RangeOrValue::Value(10),
            })]),
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
            min_slices: Some(RangeOrValue::Value(2)),
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

        assert_eq!(
            resolved,
            RangeOrValue::Range {
                range: Range { start: 30, end: 360 },
                step: Some((47, 0))
            }
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_import_resources() -> anyhow::Result<()> {
        let n = 1000i64;
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = Arc::new(RwLock::new(crate::motifs_ws::new_motifs_workspace(test_dir, WsConfigs::new(1, 2 * n as u32, 32000))?));

        // Import persists labels through MotifsStore, so the test database needs both schemas.
        let sql_store = Arc::new(SqlStore::in_mem_(vec![thoosa_store::create_tables, crate::motifs_store::create_motif_tables]).await?);
        let store: Arc<dyn MotifsStore> = sql_store.clone();
        // std::fs::remove_file("/tmp/test_main_db.db3").unwrap_or(());
        // let store = Arc::new(SqlStore::from_path("/tmp/test_main_db.db3")?);

        let imported_ids = import_resources(
            &ws,
            sql_store.as_ref(),
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
            |_t: &Transaction| Ok(()),
        )
        .await?;

        let imported_resources: Vec<Resource> = store.list_resources_by_ids(&imported_ids).await?;

        debug!("imported={imported_resources:?}");
        let mut imported_keys = imported_resources.into_iter().map(|r| r.file_stem().to_string()).collect::<Vec<_>>();
        imported_keys.sort();

        assert_eq!(vec!["t_0_0", "t_0_1", "t_1_0", "t_1_1",], imported_keys);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_import_resources_applies_freq_filter() -> anyhow::Result<()> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = Arc::new(RwLock::new(crate::motifs_ws::new_motifs_workspace(test_dir, WsConfigs::new(1, 100, 32000))?));
        // Import persists labels through MotifsStore, so the test database needs both schemas.
        let sql_store = Arc::new(SqlStore::in_mem_(vec![thoosa_store::create_tables, crate::motifs_store::create_motif_tables]).await?);
        let store: Arc<dyn MotifsStore> = sql_store.clone();

        let freq_filter = Arc::new(SyncFreqFilter::<RunCtxImpl>::new_local(SyncFreqFilterIpcInit::new(
            vec![InputFilter::MedianFilter(MedianFilterParams { window: 5 })],
            ws.read().channels() as i64,
            ws.read().sample_rate() as f64,
            0,
            1,
        ))?);

        let imported_ids = import_resources(
            &ws,
            sql_store.as_ref(),
            &["label_0"],
            || Ok(HashMap::from([("t_0_0".to_string(), vec![("label_0".to_string(), 1.0)])])),
            |_: String| Ok(Tensor::from_slice(&[0., 100., 0.]).view((1, -1))),
            Some(freq_filter),
            true,
            |_t: &Transaction| Ok(()),
        )
        .await?;

        let imported_resources: Vec<Resource> = store.list_resources_by_ids(&imported_ids).await?;
        let imported = &imported_resources[0];
        let imported_tensor = Tensor::load(ws.read().tensors_path().join(format!("{}_{}.pt", imported.import_id(), imported.file_stem())))?;
        let expected = Tensor::zeros([1, 3], (imported_tensor.kind(), imported_tensor.device()));

        assert!(imported_tensor.allclose(&expected, 1e-5, 1e-5, false));

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_resource_tensor_two_easy_motifs() -> anyhow::Result<()> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let (pulse_triangle_t, _intervals, _) = resource_tensor_two_easy_motifs(1000, 1, 100, 2, TestSignalKind::Pulse, 2, TestSignalKind::Triangle, 100, 0.01, 0.2, None)?;

        debug_plot!(&pulse_triangle_t, "pulse_triangle")?;

        let (pulse_sin_t, _intervals, _) = resource_tensor_two_easy_motifs(1000, 1, 100, 2, TestSignalKind::Pulse, 2, TestSignalKind::Sin, 100, 0.01, 0.2, None)?;
        debug_plot!(&pulse_sin_t, "pulse_sin")?;

        let ws = Arc::new(RwLock::new(crate::motifs_ws::mock_motifs_ws_simp(None, 1, 100, 10000)));

        for freq in 0..100 {
            let configs = ws.read().ws_configs();
            let freq_filter = FreqFilter::<MockRunCtx>::new(
                &[InputFilter::BandPassFilter(BandPassFilterParams {
                    low_hz: freq as f64,
                    band_hz: 1.0,
                    kernel_size: 100,
                    stride: 1,
                    padding: Padding::zero(),
                })],
                configs,
                0,
                0,
            )
            .await?;

            let mut ctx = mock_run_ctx::default(1);

            let pulse_triangle_filtered_t = Arc::clone(&freq_filter).run(Arc::new(DevicesWrap::wrap(ThoTensor::new("".into(), pulse_triangle_t.shallow_clone())?)), ctx.branch(false)).await?;
            let pulse_sin_filtered_t = freq_filter.run(Arc::new(DevicesWrap::wrap(ThoTensor::new("".into(), pulse_sin_t.shallow_clone())?)), ctx.branch(false)).await?;

            {
                let pulse_triangle_t = SelectRef::select_ref(&pulse_triangle_filtered_t, ctx.tho_device_idx(), None::<SyncStftSelector>).await?;
                let pulse_sin_t = SelectRef::select_ref(&pulse_sin_filtered_t, ctx.tho_device_idx(), None::<SyncStftSelector>).await?;

                debug_plot!(&*pulse_triangle_t.as_ref().local()?, "pulse_triangle_{freq}_filtered_t")?;
                debug_plot!(&*pulse_sin_t.as_ref().local()?, "pulse_sin_{freq}_filtered_t")?;

                Ok::<_, anyhow::Error>(())
            }?;
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
