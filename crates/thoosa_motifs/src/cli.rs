use std::io::Write;
// Commands call application operations defined independently of argument parsing.
pub use crate::resource_import::import_resources;
pub use crate::plotting::{plot_motif, plot_resource, run_plot};
pub use crate::worker_endpoints::{cuda_worker_endpoint_inits, sync_freq_filter_endpoint_init};
pub use crate::workflow::{dyn_step_find_and_select, median_window_from_find_and_select, median_window_from_find_motifs, median_window_from_find_motifs_grid, run_grid, run_opt};

use crate::motifs_store::MotifsStore;
use crate::stages::convert_precision::{ConvertPrecisionArgs, ConvertPrecisionInput, ConvertPrecisionStage};
use crate::stages::cross_loader::ResourcesConfigInput;
use crate::stages::export::{Export as ExportStage, ExportArgs, ExportInput};
use crate::stages::find_motifs_run_config::{FindMotifsRunConfigRangeOrValue, SelectMotifsRunConfigRangeOrValue};
use crate::stages::purge_models::{PurgeModelsArgs, PurgeModelsInput, PurgeModelsStage};
use crate::stages::tensor_loader::RawTensorLoader;
use anyhow::Context as _;
use async_trait::async_trait;
use clap::{Args, Subcommand};
use log::{debug, info};
use parking_lot::RwLock;
use std::fmt::Debug;
use std::fs::OpenOptions;
use std::process;
use std::sync::Arc;
use tch::Device::Cpu;
use tch::{Tensor};
use thoosa_common::resources::{Labeled, Logits, NetResources, Simple};
use thoosa_common::runner::run_ctx::RunCtxImpl;
use thoosa_common::tch_error::tch_result;
use thoosa_common::tho_device::ThoDevice;
use crate::find::{FindMotifsArgs, SelectMotifsArgs};
use crate::fragments::{PlotDataArgs, Resolution};
use crate::stages::apply_motif_nets::{ApplyMotifNets};
use crate::stages::merge_selected_motifs::{MergeSelectedMotifs, MergeSelectedMotifsArgs, MergeSelectedMotifsInput};
use crate::stages::motif_loader::{MotifLoader};
use crate::stages::motif_stats::{MotifStats};
use crate::stages::select_motifs::{SelectMotifsInput, SelectMotifsStage};
use crate::stages::seq_scan::SeqScanStage;
use crate::stages::train_final_net::{new_train_final_net};
use crate::stages::train_label_nets::{TrainLabelNetRunConfigRangeOrValue, TrainLabelNetsArgs, new_train_label_nets};
use crate::stages::train_motif_nets::{TrainMotifNetInput, TrainMotifNets};
use crate::train_nets::{TrainFinalNetArgs, TrainFinalNetRunConfigRangeOrValue, TrainMotifNetRunConfigRangeOrValue, TrainMotifsNetsArgs};
use thoosa_common::args::StageCommand;
use thoosa_common::stages::seq_scan::SeqScanRunConfigRangeOrValue;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::nets::common_nets::OptConfig;
use thoosa_common::transforms::filters::InputFilter;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::tracked_await::t;
use thoosa_shared::traits::AsyncStage;
use thoosa_shared::traits::{RunCtx};
use thoosa_shared::{mono_key, parse_resources};
use thoosa_shared::{ResourceType};
use thoosa_store::Store;

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

                let run_config_ids = t!(run_opt(
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

                let _run_config_ids = t!(run_opt(
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
                let _run_config_ids = t!(run_opt(
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

    // This is a step in filtered samples; 100 skips short EEG motifs after decimation.
    #[arg(long, default_value = "5")]
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
    use thoosa_common::runner::common::cachable::{BorrowRef, DevicesWrap};
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
                let pulse_triangle_t = BorrowRef::borrow_ref(&pulse_triangle_filtered_t, ctx.tho_device_idx()).await?;
                let pulse_sin_t = BorrowRef::borrow_ref(&pulse_sin_filtered_t, ctx.tho_device_idx()).await?;

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
