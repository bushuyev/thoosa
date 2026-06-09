use std::collections::HashMap;
use std::fmt::format;
use std::ops::Range;
use parking_lot::RwLock;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use futures::StreamExt;
use log::{debug, error, info};
use tch::Device::Cpu;
use tracing::Instrument;
use thoosa_common::my_tch::{InitDummyRangeOrValue, MotifInitModeRangeOrValue, TrimmedMeanRangeOrValue};
use thoosa_main::tasks::find_motifs_run_config::{AccumulateStatsRangeOrValue, AggrRangeOrValue, CrossLoaderRunConfigRangeOrValue, FinderKindRangeOrValue, FindMotifsRunConfigRangeOrValue, MaxSlicesAbsRangeOrValue, MaxSlicesRangeOrValue, MetricsRangeOrValue, MotifCriterionConstraintRangeOrValue, NormThicknessRunConfigRangeOrValue, QuantileAggrRangeOrValue, SelectMotifsRunConfigRangeOrValue, SlicesFilterRangeOrValue, Metrics, NormThicknessRunConfig};
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_common::runner::common::RunCtx;
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_common::runs::Simple;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::train_nets;
use thoosa_common::train_nets::{AdamWRangeOrValue, CommonTrainingRunConfigRangeOrValue, FinalNetArgsRangeOrValue, LabelNetTypeArgsRangeOrValue, LinNetArgsRangeOrValue, MotifCosine2DNetArgsRangeOrValue, MotifCosineNetArgsRangeOrValue, MotifsBy, OptConfig, OptimizersRangeOrValue, SelfAttentionNetArgsRangeOrValue, StftRunConfig, StftRunConfigRangeOrValue, TrainFinalNetRunConfigRangeOrValue, TrainLabelNetRunConfig, TrainLabelNetRunConfigRangeOrValue, TrainMotifNetRunConfig, TrainMotifNetRunConfigRangeOrValue};
use thoosa_common::transforms::filters::{BandPassFilterParamsRangeOrValue, PaddingRangeOrValue, ZNormFilterParamsRangeOrValue, ZeroPaddingRangeOrValue};
use thoosa_common::transforms::filters::InputFilterRangeOrValue::BandPassFilter;
use thoosa_common::transforms::filters::InputFilterRangeOrValue::ZNorm;
use thoosa_common::ws::Workspace;
use thoosa_main::cli::{run_grid, run_opt, run_plot, RunAllArgs};
use thoosa_main::{ApplyMotifNets, tasks, RawTensorLoader, TrainMotifNetInput, TrainMotifNets};
use thoosa_main::tasks::apply_cosine2d_motif_nets::ApplyCosine2DMotifNets;
use thoosa_main::tasks::bucketing::{BucketRunConfigRangeOrValue, BucketingConfigRangeOrValue, PlainBandAggregationMeanRangeOrValue, PlainConfigRangeOrValue};
use thoosa_main::tasks::bucketing::PlainBandAggregationRangeOrValue::Mean;
use thoosa_main::tasks::cross_loader::ResourcesConfigInput;
use thoosa_main::tasks::find_and_select_motifs::{FindAndSelectMotifsInput, FindAndSelectMotifsRunConfig, FindAndSelectMotifsRunConfigRangeOrValue};
use thoosa_main::tasks::motif_loader::MotifLoader;
use thoosa_main::tasks::motif_stats::{MotifStats, MotifStatsInput};
use thoosa_main::tasks::select_buckets::{BestSelectionModeRangeOrValue, FilterSelectionModeRangeOrValue, SelectBucketsRunConfigRangeOrValue, SelectBucketsStage, SelectionModeRangeOrValue};
use thoosa_main::tasks::select_motifs::{SelectMotifsInput};
use thoosa_main::tasks::stages_seq::ABStage;
use thoosa_main::tasks::stft::Stft;
use thoosa_main::tasks::train_label_nets::new_train_label_nets;
use thoosa_motifs::distance::{PearsonCorrConfigRangeOrValue, SimilarityMetricRangeOrValue};
use thoosa_motifs::find::FixedFindMotifsConfig;
use thoosa_motifs::fragments::{DataBy, PlotDataArgs};
use thoosa_shared::{collect_futures_with_progress, mono_key, panic_to_err_async, RangeOrValue, ResourceType, StreamBuffering, ThoError};
use thoosa_shared::collect_futures_eta::{collect_eta_record_completion, CollectEtaInvocationGuard};
use thoosa_shared::math::unordered_sets_of_size;
use thoosa_shared::ResourceType::NotInGroup;
use thoosa_shared::run_configs::{run_id, RunConfig, RunConfigId};
use thoosa_store::Store;
use thoosa_shared::tracked_await::track_await;

pub enum RunMode {
    RunAll {
        fixed_find_motifs_config:FixedFindMotifsConfig,
        find_motifs_run_config_range_or_value: FindMotifsRunConfigRangeOrValue,
        select_motifs_run_config:SelectBucketsRunConfigRangeOrValue,


        train_motif_nets_run_config_range_or_value:TrainMotifNetRunConfigRangeOrValue,
        train_motifs_opt_config:Arc<OptConfig>,

        train_label_nets_run_config_range_or_value:TrainLabelNetRunConfigRangeOrValue,
        train_labels_opt_config:Arc<OptConfig>,
    },
    #[allow(dead_code)]
    RunLabels {
        train_label_nets_run_config_range_or_value:TrainLabelNetRunConfigRangeOrValue,
        train_labels_opt_config:Arc<OptConfig>,
    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct ClefRunConfig {
    pub id: RunConfigId
}

impl RunConfig for ClefRunConfig {
    fn set_id(&mut self, id: RunConfigId) {
        self.id = id;
    }

    fn type_name() -> &'static str {
        "ClefRunConfig"
    }
}

pub(crate) async fn run_all(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    loader: &Arc<RawTensorLoader<RunCtxImpl>>,
    mut root_ctx: RunCtxImpl,
    run_all_args: &RunAllArgs,
    run_mode:RunMode
) -> Result<(), ThoError> {
    debug!("run_all");

    // Navigation note: this is the project-level run-all workflow invoked from bird_clef::main via run_cli.
    // It fans out by label, then chains find/select motifs -> train motif nets -> train label nets.
    let labels = track_await("run_all/1", store.list_labels(true)).instrument(tracing::debug_span!("run_all/1", file = file!(), line = line!())).await?;
    info!("run_all: labels={:?}", labels.len());
    debug!("run_all: labels={labels:?}");
    let labels_filter = run_all_args.labels_filter();
    let labels_filter_ref = labels_filter.as_ref();


    match run_mode {
        RunMode::RunAll { find_motifs_run_config_range_or_value, fixed_find_motifs_config, select_motifs_run_config, train_motif_nets_run_config_range_or_value, train_label_nets_run_config_range_or_value, train_motifs_opt_config, train_labels_opt_config } => {
            // Default BirdCLEF path: find motifs for each label/resource set, bypass SelectBucketsStage with select_all_motifs, then train motif and label nets.
            let find_motifs_run_config = Arc::new(find_motifs_run_config_range_or_value.clone());
            let select_motifs_run_config = Arc::new(select_motifs_run_config.clone());
            let train_motif_nets_run_config_range_or_value = Arc::new(train_motif_nets_run_config_range_or_value);
            let train_label_nets_run_config_range_or_value = Arc::new(train_label_nets_run_config_range_or_value);

            let find_resources = run_all_args.find_resources.clone();
            let _select_resources = run_all_args.select_resources.clone();
            let labels_eta_guard = CollectEtaInvocationGuard::new("labels", labels_filter_ref.map(|v|v.len()).unwrap_or(labels.len()), 1);
            let labels_eta_guard_id = labels_eta_guard.id();
            let futures = labels.into_iter().filter(|label|
                {
                    labels_filter_ref
                        .map(|labels| labels.iter().any(|filter_label| filter_label.eq_ignore_ascii_case(label)))
                        .unwrap_or(true)
                }
            ).map(|label| {
                info!("run_all: label={label}");

                let fixed_find_motif_config = fixed_find_motifs_config.clone();

                let mut label_ctx = root_ctx.branch(true);
                let init_ctx_find_motifs = label_ctx.branch(false);

                let find_motifs_run_config = Arc::clone(&find_motifs_run_config);

                let train_motif_nets_run_config_range_or_value = Arc::clone(&train_motif_nets_run_config_range_or_value);
                let train_label_nets_run_config_range_or_value = Arc::clone(&train_label_nets_run_config_range_or_value);
                let train_labels_opt_config = Arc::clone(&train_labels_opt_config);
                let train_motifs_opt_config = Arc::clone(&train_motifs_opt_config);


                let find_resources = find_resources.clone();
                let select_motifs_run_config = Arc::clone(&select_motifs_run_config);
                let label = label.clone();
                let label_for_span = label.clone();
                let ws = Arc::clone(&ws);
                let ws_spawn = Arc::clone(&ws);
                let store_spawn = Arc::clone(&store);
                let ws_plot = Arc::clone(&ws);

                let store = Arc::clone(&store);
                let loader = Arc::clone(&loader);
                let loader_spawn = Arc::clone(&loader);
                let run_all_args = run_all_args.clone();

                tokio::spawn(async move {
                    panic_to_err_async!(async move {

                        let find_motifs_input = Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(Simple::new(ResourceType::Set(format!("{find_resources}_{label}_train"))))));
                        let find_motifs_run_config_rov = Arc::clone(&find_motifs_run_config);

                        let fixed_find_motif_config = fixed_find_motif_config.clone();

                        let find_motifs_run_config_ids = run_grid::<_, _, _, _, _>(
                            None,
                            find_motifs_run_config_rov,
                            find_motifs_input,
                            &Arc::clone(&ws_spawn),
                            &Arc::clone(&store_spawn),
                            async move |run_configs| {
                                debug!("run_all: run_configs={run_configs:?}");

                                Ok(Arc::new(
                                    tasks::find_motifs::FindMotifsStage::new(
                                        &Arc::clone(&ws_spawn),
                                        &Arc::clone(&store_spawn),
                                        Arc::clone(&loader_spawn),
                                        run_configs,
                                        fixed_find_motif_config,
                                        ThoDevice::from(1),
                                        init_ctx_find_motifs
                                    )
                                    .instrument(tracing::debug_span!("FindMotifsStage::new",file = file!(),line = line!()))
                                    .await?,
                                ))
                            },
                            || label_ctx.branch(true),
                            &format!("find motifs for {label}")
                        ).instrument(tracing::debug_span!("run_all/2", file = file!(), line = line!())).await?;

                        info!("run_all: label={label} find_motifs_run_config_ids={find_motifs_run_config_ids:?}");

                        let select_motifs_input = Arc::new(DevicesWrap::wrap(SelectMotifsInput::standalone(ResourcesConfigInput::new(Simple::new(ResourceType::Set(format!("{find_resources}_{label}_train")))))));
                        let init_ctx_select_buckets = label_ctx.branch(true);
                        let ws_for_select_buckets = Arc::clone(&ws);
                        let store_for_select_buckets = Arc::clone(&store);
                        let loader_for_select_buckets = Arc::clone(&loader);
                        let select_motifs_input_for_select_buckets = Arc::clone(&select_motifs_input);
                        let find_motifs_run_config_ids_for_select_buckets = find_motifs_run_config_ids.clone();

                        let select_motifs_run_config_ids = run_grid::</*Labeled,*/ _, _, _, _, _>(
                            Some(find_motifs_run_config_ids.clone()),
                            Arc::clone(&select_motifs_run_config),
                            Arc::clone(&select_motifs_input),
                            &Arc::clone(&ws_for_select_buckets),
                            &store,
                            async move |run_configs| {
                                Ok(Arc::new(SelectBucketsStage::<RawTensorLoader<RunCtxImpl>, MotifLoader<RawTensorLoader<RunCtxImpl>, RunCtxImpl>, RunCtxImpl>::new(
                                    &ws_for_select_buckets,
                                    &store_for_select_buckets,
                                    Arc::clone(&loader_for_select_buckets),
                                    Arc::new(MotifLoader::new(
                                        Arc::clone(&store_for_select_buckets),
                                        Arc::clone(&loader_for_select_buckets),
                                        &ws_for_select_buckets
                                    )),
                                    run_configs,
                                    find_motifs_run_config_ids_for_select_buckets,
                                    select_motifs_input_for_select_buckets,
                                    init_ctx_select_buckets,
                                ).instrument(tracing::debug_span!("run_all/3", file = file!(), line = line!())).await?))
                            },
                            || label_ctx.branch(true),
                            &format!("select motifs for {label}")
                        ).instrument(tracing::debug_span!("run_all/4", file = file!(), line = line!())).await?;

                        // select_all_motifs is a deliberate bypass: it creates selection runs that include every found motif without invoking SelectBucketsStage criteria.
                        // let select_motifs_run_config_ids = store.select_all_motifs(&find_motifs_run_config_ids).await?;

                        info!("select motifs done: label={label} select_motifs_run_config_ids={select_motifs_run_config_ids:?}");

                        if run_all_args.with_plots {
                            for select_motifs_run_config_id in &select_motifs_run_config_ids {
                                run_plot(&ws_plot, &store, &loader, &PlotDataArgs::new(DataBy::SelectedMotifs { select_motifs_run_config_ids: vec![*select_motifs_run_config_id] }), run_all_args.resolution, &Cpu, label_ctx.branch(false)).instrument(tracing::debug_span!("run_all/5", file = file!(), line = line!())).await?;
                            }
                        }

                        let input_wrapper = Arc::new(DevicesWrap::wrap(TrainMotifNetInput::new(MotifsBy::SelectMotifsRunIds { select_motifs_run_config_ids: select_motifs_run_config_ids.clone() }, train_motifs_opt_config)));

                        // Run a grid over motif-net training configs; output is a set of run ids for trained motif nets.
                        let train_motif_net_run_config_ids = run_opt::</*Labeled,*/ _, _, TrainMotifNetRunConfig, _, _>(
                            train_motif_nets_run_config_range_or_value,
                            input_wrapper,
                            &ws,
                            &store,
                            async |run_configs| {
                                Ok(Arc::new(TrainMotifNets::<_, RunCtxImpl>::new(
                                    Arc::clone(&loader),
                                    &ws,
                                    &store,
                                    run_configs,
                                    run_all_args.train_motifs_epoch_size,
                                    Some(label.clone())
                                )?))
                            },
                            || label_ctx.branch(false),
                        ).instrument(tracing::debug_span!("run_all/6", file = file!(), line = line!())).await?;

                        info!("train motif nets done: label={label} train_motif_nets_run_config_ids={train_motif_net_run_config_ids:?}");

                        // Pre-create context and shared handles for label-net stage construction.
                        let mut init_ctx_train_labels = label_ctx.branch(false);
                        let store_in_train_labels = Arc::clone(&store);
                        let ws_in_train_labels = Arc::clone(&ws);
                        let store_in_train_labels = Arc::clone(&store);
                        let loader_in_train_labels = Arc::clone(&loader);


                        let train_labels_input = Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(
                            run_all_args.train_resource_config()
                        )/*.check(&store).await?*/));

                         let combo_sets = unordered_sets_of_size(train_motif_net_run_config_ids.as_slice(), run_all_args.selected_motif_nets_limit.unwrap_or(5));
                        let limit = combo_sets.len().min(10);//TODO ..100

                        let train_motif_net_run_config_ids_combos = combo_sets[..limit].to_vec();
                        let combos_eta = CollectEtaInvocationGuard::new(&format!("{label}_combos"), train_motif_net_run_config_ids_combos.len(), 1);
                        let best_min_loss = Arc::new(RwLock::new(f32::MAX));
                        for train_motif_net_run_config_ids_combo in train_motif_net_run_config_ids_combos {
                            debug!("run_all: train_motif_net_run_config_ids_combo={train_motif_net_run_config_ids_combo:?}");

                            let train_label_nets_run_config_range_or_value = Arc::clone(&train_label_nets_run_config_range_or_value);
                            let train_labels_input = Arc::clone(&train_labels_input);
                            let ws_in_train_labels = Arc::clone(&ws_in_train_labels);
                            let store_in_train_labels = Arc::clone(&store_in_train_labels);
                            let loader_in_train_labels = Arc::clone(&loader_in_train_labels);
                            let mut label_ctx = label_ctx.branch(false);
                            let train_labels_opt_config = Arc::clone(&train_labels_opt_config);
                            let mut init_ctx_train_labels =  init_ctx_train_labels.branch(false);
                            let best_min_loss= Arc::clone(&best_min_loss);
                            let label_for_opt = label.clone();
                            let ws_opt = Arc::clone(&ws_in_train_labels);
                            match panic_to_err_async!(
                                  run_opt::<_, _, TrainLabelNetRunConfig, _, _>(
                                    train_label_nets_run_config_range_or_value,
                                    train_labels_input,
                                    &ws_opt,
                                    &store,
                                    async move |run_configs| {
                                        let apply_motif_nets = Arc::new(ApplyCosine2DMotifNets::new(
                                            &ws_in_train_labels,
                                            &store_in_train_labels,
                                            train_motif_net_run_config_ids_combo.clone(),
                                            init_ctx_train_labels.branch(false),
                                            None, //Some(run_all_args.max_motif_net_loss), //to RunAll args,
                                            None, //run_all_args.selected_motif_nets_limit,
                                            run_all_args.motif_slices_step,
                                            None,
                                            false,
                                        ).instrument(tracing::debug_span!("run_all/7", file = file!(), line = line!())).await?);

                                        new_train_label_nets(
                                            train_motif_net_run_config_ids_combo.clone(),
                                            Arc::clone(&loader_in_train_labels),
                                            apply_motif_nets,
                                            run_configs,
                                            &ws_in_train_labels,
                                            &store_in_train_labels,
                                            train_labels_opt_config,
                                            Some(vec![label_for_opt.clone()]),
                                        ).instrument(tracing::debug_span!("run_all/8", file = file!(), line = line!())).await.map(|v| Arc::new(v))
                                    },
                                    || label_ctx.branch(false),
                                )
                            ) {
                                Ok(train_label_nets_run_config_ids) => {
                                    info!("train label nets done: label={label} run_config_ids={train_label_nets_run_config_ids:?}");
                                    let min_loss = store.list_trained_nets(train_label_nets_run_config_ids, None, None).instrument(tracing::debug_span!("run_all/9", file = file!(), line = line!())).await?
                                        .into_iter().map(|net| net.loss).reduce(f32::min).unwrap_or(f32::MAX);

                                    if min_loss < *best_min_loss.read() {
                                        *best_min_loss.write() = min_loss;
                                    }
                                    if min_loss < 0.3 {
                                        info!("stopping training label {label} got loss={min_loss}");

                                        break
                                    }
                                }
                                Err(e) => {
                                    error!("train label nets failed: label={label} e={e:?}");
                                }
                            };

                            collect_eta_record_completion(combos_eta.id());

                        }

                        info!("train label nets done: label={label} \x1b[38;50;208mbest_min_loss={best_min_loss:?}\x1b[0m");

                        collect_eta_record_completion(labels_eta_guard_id);
                        Ok::<_, ThoError>(())
                    })
                }.instrument(tracing::info_span!(
                    "bird_clef::run_label_workflow",
                    kind = "spawn",
                    file = file!(),
                    line = line!(),
                    task = format!("run workflow for label_{label_for_span}"),
                )))
            });

            track_await("run_all/10", futures::stream::iter(futures)
            .buffer_unordered(2)
            .collect::<Vec<_>>())
            .instrument(tracing::debug_span!("run_all/10", file = file!(), line = line!())).await;

            // let _ = collect_futures_with_progress(
            //     futures,
            //     labels.len(),
            //     4,
            //     StreamBuffering::Unordered,
            //     "bird_clef_labels",
            // ).await;

            Ok::<_, ThoError>(())
        }   
        RunMode::RunLabels { train_label_nets_run_config_range_or_value, train_labels_opt_config } => {
            let train_label_nets_run_config_range_or_value = Arc::new(train_label_nets_run_config_range_or_value);



            let futures = labels.iter().filter(|label|
                {
                    labels_filter_ref
                        .map(|labels| labels.iter().any(|filter_label| filter_label.eq_ignore_ascii_case(label)))
                        .unwrap_or(true)
                }
            ).map(|label| {
                info!("run_all: label={label}");
                let mut label_ctx = root_ctx.branch(true);
                let mut init_ctx_train_labels = label_ctx.branch(false);

                let train_labels_opt_config = Arc::clone(&train_labels_opt_config);
                let ws_in_train_labels = Arc::clone(&ws);
                let store_in_train_labels = Arc::clone(&store);
                let train_label_nets_run_config_range_or_value = Arc::clone(&train_label_nets_run_config_range_or_value);
                let train_labels_input = Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(
                    run_all_args.train_resource_config()
                )));

                async move {
                    let train_motif_net_run_config_ids = track_await("run_all/11", store.list_run_config_ids_by_descr(label.clone(), TrainMotifNetRunConfig::type_name())).instrument(tracing::debug_span!("run_all/11", file = file!(), line = line!())).await?;
                    if train_motif_net_run_config_ids.is_empty(){
                        return Ok(());
                    }


                    let combo_sets = unordered_sets_of_size(train_motif_net_run_config_ids.as_slice(), run_all_args.selected_motif_nets_limit.unwrap_or(3));
                    let limit = combo_sets.len().min(100);//TODO ..100

                    let train_motif_net_run_config_ids_combos = combo_sets[..limit].to_vec();
                    let combos_eta = CollectEtaInvocationGuard::new(&format!("{label}_combos"), train_motif_net_run_config_ids_combos.len(), 1);

                    for train_motif_net_run_config_ids_combo in train_motif_net_run_config_ids_combos {
                        let train_label_nets_run_config_range_or_value = Arc::clone(&train_label_nets_run_config_range_or_value);
                        let train_labels_input = Arc::clone(&train_labels_input);
                        let train_motif_net_run_config_ids = train_motif_net_run_config_ids.clone();
                        let ws_in_train_labels = Arc::clone(&ws_in_train_labels);
                        let store_in_train_labels = Arc::clone(&store_in_train_labels);
                        let mut init_ctx_train_labels = init_ctx_train_labels.branch(false);
                        let mut label_ctx = label_ctx.branch(false);
                        let train_labels_opt_config = Arc::clone(&train_labels_opt_config);
                        match panic_to_err_async!(
                              run_opt::<_, _, TrainLabelNetRunConfig, _, _>(
                                train_label_nets_run_config_range_or_value,
                                train_labels_input,
                                &ws,
                                &store,
                                async move |run_configs| {
                                    let apply_motif_nets = Arc::new(ApplyMotifNets::new(
                                        &ws_in_train_labels,
                                        &store_in_train_labels,
                                        train_motif_net_run_config_ids.clone(),
                                        init_ctx_train_labels.branch(false),
                                        Some(run_all_args.max_motif_net_loss),
                                        run_all_args.selected_motif_nets_limit,
                                        run_all_args.motif_slices_step,
                                        None,
                                        false,
                                    ).instrument(tracing::debug_span!("run_all/12", file = file!(), line = line!())).await?);

                                    new_train_label_nets(
                                        train_motif_net_run_config_ids.clone(),
                                        Arc::clone(&loader),
                                        apply_motif_nets,
                                        run_configs,
                                        &ws_in_train_labels,
                                        &store_in_train_labels,
                                        train_labels_opt_config,
                                        Some(vec![label.clone()]),
                                    ).instrument(tracing::debug_span!("run_all/13", file = file!(), line = line!())).await.map(|v| Arc::new(v))
                                },
                                || label_ctx.branch(false),
                            )
                        ) {
                            Ok(train_label_nets_run_config_ids) => {
                                info!("train label nets done: label={label} run_config_ids={train_label_nets_run_config_ids:?}");
                                let min_loss = track_await("run_all/14", store.list_trained_nets(train_label_nets_run_config_ids, None, None)).instrument(tracing::debug_span!("run_all/14", file = file!(), line = line!())).await?
                                    .into_iter().map(|net| net.loss).reduce(f32::min).unwrap_or(f32::MAX);

                                if min_loss < 0.1 {
                                    info!("stopping training label {label} got loss={min_loss}");

                                    break
                                }
                            }
                            Err(e) => {
                                error!("train label nets failed: label={label} e={e:?}");
                            }
                        };


                        collect_eta_record_completion(combos_eta.id());

                    }
                    info!("train label net done: label={label}");

                    Ok(())
                }
            });

            //
            let _ = track_await("run_all/15", collect_futures_with_progress(
                futures,
                labels.len(),
                16,
                StreamBuffering::Unordered,
                "bird_clef_labels",
            )).instrument(tracing::debug_span!("run_all/15", file = file!(), line = line!())).await;

            info!("Clef train labels done, run_id={:?}", run_id());

            Ok::<_, ThoError>(())
        }

    }?;



    Ok::<_, ThoError>(())

}


#[allow(dead_code)]
pub(crate) fn run_labels_config_0(run_all_params: &RunAllArgs) -> RunMode {
    //////////////////////train labels
    RunMode::RunLabels {
        train_label_nets_run_config_range_or_value: TrainLabelNetRunConfigRangeOrValue {
            id: None,
            common_training_args: CommonTrainingRunConfigRangeOrValue {
                epochs: RangeOrValue::Value(100),
                batch_size: RangeOrValue::Value(2),
                lr: RangeOrValue::Range{ range: Range { start: 0.00001, end: 0.001 }, step: None },
                patience: RangeOrValue::Value(10),
                min_delta: RangeOrValue::Value(1e-4_f32),
                optimizer: Some(OptimizersRangeOrValue::AdamW(AdamWRangeOrValue{
                    beta1: RangeOrValue::Value(0.9),
                    beta2: RangeOrValue::Value(0.999),
                    wd: RangeOrValue::Value(0.01),
                    amsgrad: RangeOrValue::Value(true),
                })),
            },

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
            // net: LabelNetTypeArgsRangeOrValue::SelfAttentionNet(SelfAttentionNetArgsRangeOrValue {
            //     // d_model: RangeOrValue::Value(64),
            //     head_size: RangeOrValue::Range { range: Range { start: 64, end: 256 }, step: None },
            //     n_heads: RangeOrValue::Range { range: Range { start: 4, end: 12 }, step: None },
            //     // ff_mult: RangeOrValue::Range { range: Range { start: 2, end: 4 }, step: None },
            //     ff_mult: RangeOrValue::Value(2),
            //     // dropout: RangeOrValue::Value(0.1),
            //     dropout: RangeOrValue::Range { range: Range { start: 0.01, end: 0.2 }, step: None },
            // })
            net: LabelNetTypeArgsRangeOrValue::LinNet(LinNetArgsRangeOrValue {
                // num_layers: RangeOrValue::Value(4),
                num_layers: RangeOrValue::Range { range: Range { start: 1, end: 4 }, step: None },
                features: RangeOrValue::Range { range: Range { start: 100, end: 1000 }, step: None },
            })

        },
        train_labels_opt_config: Arc::new(OptConfig::new_with_early_stopping(100, 10, 5, 10e-5)),
    }}



pub(crate) fn run_all_config_0(run_all_params: &RunAllArgs) -> RunMode {

    //gpt Navigation note: main.rs currently chooses this config. CLI values still enter through RunAllArgs;
    //gpt this function owns the default search spaces for motif discovery and training.

    RunMode::RunAll {
        fixed_find_motifs_config: FixedFindMotifsConfig::new(run_all_params.max_chunk_size, false),

        find_motifs_run_config_range_or_value: FindMotifsRunConfigRangeOrValue {
            id: None,
            step: RangeOrValue::Value(1000),
            window: RangeOrValue::Range{ range: Range {start: 16000, end: 72001}, step: Some((32000, 0)) },
            // window: RangeOrValue::Value(10000),
            // freq_filter: None,
            freq_filter: Some(vec![
               ZNorm(ZNormFilterParamsRangeOrValue{ _path_pd: Default::default() })
            ]),
            // freq_filter: Some(vec![
            //     AverageFilter(AverageFilterParamsRangeOrValue {
            //         window: RangeOrValue::dynamic(median_window_from_find_motifs_grid),
            //         step: RangeOrValue::Value(1),
            //     }),
            // ]),

            // freq_filter: Some(vec![
            // BandPassFilter(BandPassFilterParamsRangeOrValue {
            //     low_hz: RangeOrValue::Range { range: Range { start: 0., end: 16001. }, step: Some((4000., 1)) },
            //     band_hz: RangeOrValue::Value(1.),
            //     kernel_size: RangeOrValue::dynamic(median_window_from_find_motifs_grid),
            //     stride: RangeOrValue::Value(1),
            //     padding: PaddingRangeOrValue::Zero(ZeroPaddingRangeOrValue::default()),
            // }),
            // Abs(AbsFilterParamsRangeOrValue{ _path_pd: Default::default() }),
            // MedianFilter(MedianFilterParamsRangeOrValue{
            //     // window: RangeOrValue::Value(1000)
            //     window: RangeOrValue::dynamic(median_window_from_find_motifs_grid),
            // })
            // ]),
            // finder_kind: FinderKindRangeOrValue::CrossLoader(CrossLoaderRunConfigRangeOrValue {
            //     // similarity_metric: SimilarityMetricRangeOrValue::EucDist(EucDistConfigRangeOrValue { z_normalize: RangeOrValue::Value(false) }),
            //     similarity_metric: SimilarityMetricRangeOrValue::DotProduct(DotProductConfigRangeOrValue{ with_rfft: RangeOrValue::Value(true) }),
            //     // similarity_metric: SimilarityMetricRangeOrValue::PearsonCorr(PearsonCorrConfigRangeOrValue{ _path_pd: Default::default() }),
            //     // slices_filter: SlicesFilterRangeOrValue::Precalculate(PrecalculateStatsRangeOrValue {
            //     //     aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue{ k: RangeOrValue::Value(5.0) }),
            //     //     // aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue { q: RangeOrValue::Value(1.0) }),
            //     //     slices: RangeOrValue::Value(100),
            //     // }),
            //     slices_filter: SlicesFilterRangeOrValue::Accumulate(AccumulateStatsRangeOrValue {
            //         // aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue{ k: RangeOrValue::Value(1.0) }),
            //         aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue { q: RangeOrValue::Value(0.5) }),
            //     }),
            // }),
            finder_kind: FinderKindRangeOrValue::Bucket(BucketRunConfigRangeOrValue {
                id: None,
                bucket_count: RangeOrValue::Value(500),
                min_slices: RangeOrValue::Value(3),
                max_slices: RangeOrValue::Value(1000),
                bucket_config: BucketingConfigRangeOrValue::Plain(PlainConfigRangeOrValue {
                    band_hz: RangeOrValue::Value(64.0),
                    aggregation: Mean(PlainBandAggregationMeanRangeOrValue{ _path_pd: Default::default() }),
                    apply_hann_window: RangeOrValue::Value(true),
                    remove_dc_offset: RangeOrValue::Value(true),
                    normalize_lsh: RangeOrValue::Value(true),
                })
            }),
            max_slices: Some(MaxSlicesRangeOrValue::Abs(MaxSlicesAbsRangeOrValue { count: RangeOrValue::Value(100000) })),
        },


        select_motifs_run_config: SelectBucketsRunConfigRangeOrValue {
            id: None,
            // selection_mode: SelectionModeRangeOrValue::Filter(FilterSelectionModeRangeOrValue { threshold: RangeOrValue::Value(1.8) }),
            selection_mode: SelectionModeRangeOrValue::Best(BestSelectionModeRangeOrValue{ limit: RangeOrValue::Value(10) }),
        },

        train_motif_nets_run_config_range_or_value: TrainMotifNetRunConfigRangeOrValue {
            id: None,
            common_training_args: CommonTrainingRunConfigRangeOrValue {
                epochs: RangeOrValue::Value(10),
                batch_size: RangeOrValue::Value(10),
                // lr: RangeOrValue::Value(0.0001),
                lr: RangeOrValue::Range{ range: Range { start: 0.00001, end: 0.001 }, step: None },
                patience: RangeOrValue::Value(5),
                min_delta: RangeOrValue::Value(1e-4_f32),

                optimizer: Some(OptimizersRangeOrValue::AdamW(AdamWRangeOrValue{
                    beta1: RangeOrValue::Value(0.9),
                    beta2: RangeOrValue::Value(0.999),
                    wd: RangeOrValue::Value(0.01),
                    amsgrad: RangeOrValue::Value(true),
                })),
            },
            // net: train_nets::MotifNetTypeArgsRangeOrValue::LinNet(MotifLinNetArgsRangeOrValue {
            //     num_layers: RangeOrValue::Value(2),
            //     step: RangeOrValue::Value(1000),
            //     channels: RangeOrValue::Value(2),
            //     kernel: RangeOrValue::Value(1000),
            // }),

            // net: train_nets::MotifNetTypeArgsRangeOrValue::ConvNet(train_nets::MotifConvNetArgsRangeOrValue {
            //     conv_layers: RangeOrValue::Range { range: Range { start: 2, end: 4 }, step: None },
            //     // conv_layers: RangeOrValue::Value(1),
            //     // head_layers: RangeOrValue::Value(1),
            //     head_layers: RangeOrValue::Range { range: Range { start: 2, end: 4 }, step: None },
            //     step: RangeOrValue::Value(20),
            //     channels: RangeOrValue::Value(2), //RangeOrValue::Range{ range: Range {start: 2, end: 4}, step: None },
            //     // channels: RangeOrValue::Range{ range: Range {start: 2, end: 4}, step: None },
            //     // kernel: RangeOrValue::Value(10),
            //     kernel: RangeOrValue::Range{ range: Range {start: 100, end: 200}, step: None },
            //     head_pool: None,
            // })

            net: train_nets::MotifNetTypeArgsRangeOrValue::Cosine2DNet(MotifCosine2DNetArgsRangeOrValue{
                // step: RangeOrValue::Value(1),
                // channels: RangeOrValue::Range { range: Range { start: 4, end: 32 }, step: None },
                // kernel: RangeOrValue::Range { range: Range { start: 10, end: 100 }, step: None },
                a: RangeOrValue::Range { range: Range { start: 1.0, end: 10.0 }, step: None },
                b: RangeOrValue::Range { range: Range { start: 0.0, end: 1.0 }, step: None },
                init_mode: MotifInitModeRangeOrValue::Vec(vec![
                    MotifInitModeRangeOrValue::Mean(InitDummyRangeOrValue { _path_pd: Default::default() }),
                    // MotifInitModeRangeOrValue::Median(InitDummyRangeOrValue { _path_pd: Default::default() }),
                    // MotifInitModeRangeOrValue::TrimmedMean(TrimmedMeanRangeOrValue { trim_frac: RangeOrValue::Value(0.3) }),
                    // MotifInitModeRangeOrValue::CenteredMean(InitDummyRangeOrValue { _path_pd: Default::default() }),
                    MotifInitModeRangeOrValue::L2NormalizedMean(InitDummyRangeOrValue { _path_pd: Default::default() }),
                    // MotifInitModeRangeOrValue::ZScoreMean(InitDummyRangeOrValue { _path_pd: Default::default() }),
                    // MotifInitModeRangeOrValue::CenteredL2NormalizedMean(InitDummyRangeOrValue { _path_pd: Default::default() }),
                ]),
                stft_config: StftRunConfigRangeOrValue {
                    n_fft: RangeOrValue::Value(16000),
                    hop_length: Some(RangeOrValue::Value(4000)),
                    win_length: Some(RangeOrValue::Value(4000)),
                    normalized: RangeOrValue::Value(true),
                    onesided: RangeOrValue::Value(true),
                    align_to_window: RangeOrValue::Value(true),
                },
            })

        },
        train_motifs_opt_config: Arc::new(OptConfig::new_with_early_stopping(10, 3, 5, 10e-5)),

        train_label_nets_run_config_range_or_value: TrainLabelNetRunConfigRangeOrValue {
            id: None,
            common_training_args: CommonTrainingRunConfigRangeOrValue {
                epochs: RangeOrValue::Value(100),
                batch_size: RangeOrValue::Value(2),
                lr: RangeOrValue::Range{ range: Range { start: 0.00001, end: 0.001 }, step: None },
                patience: RangeOrValue::Value(5),
                min_delta: RangeOrValue::Value(1e-4_f32),
                optimizer: Some(OptimizersRangeOrValue::AdamW(AdamWRangeOrValue{
                    beta1: RangeOrValue::Value(0.9),
                    beta2: RangeOrValue::Value(0.999),
                    wd: RangeOrValue::Value(0.01),
                    amsgrad: RangeOrValue::Value(true),
                })),
            },

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
            // net: LabelNetTypeArgsRangeOrValue::SelfAttentionNet(SelfAttentionNetArgsRangeOrValue {
            //     // d_model: RangeOrValue::Value(64),
            //     head_size: RangeOrValue::Range { range: Range { start: 64, end: 256 }, step: None },
            //     n_heads: RangeOrValue::Range { range: Range { start: 4, end: 12 }, step: None },
            //     // ff_mult: RangeOrValue::Range { range: Range { start: 2, end: 4 }, step: None },
            //     ff_mult: RangeOrValue::Value(2),
            //     // dropout: RangeOrValue::Value(0.1),
            //     dropout: RangeOrValue::Range { range: Range { start: 0.01, end: 0.2 }, step: None },
            // })
            net: LabelNetTypeArgsRangeOrValue::LinNet(LinNetArgsRangeOrValue {
                // num_layers: RangeOrValue::Value(4),
                num_layers: RangeOrValue::Range { range: Range { start: 1, end: 4 }, step: None },
                features: RangeOrValue::Range { range: Range { start: 64, end: 512 }, step: None },
            })

        },
        train_labels_opt_config: Arc::new(OptConfig::new_with_early_stopping(30, 10, 5, 10e-5)),
    }
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use crate::run_all::{run_all, run_all_config_0};
    use log::debug;
    use parking_lot::RwLock;
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;
    use tch::{Device, Tensor};
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use thoosa_common::tho_device::ThoDevice;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::ws::FsWorkspace;
    use thoosa_main::cli::RunAllArgs;
    use thoosa_main::RawTensorLoader;
    use thoosa_motifs::fragments::Resolution;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::{mono_key, ThoError};
    use thoosa_shared::resource::ResourceStats;
    use thoosa_store::SqlStore;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_run_all() -> Result<(), ThoError>{
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

        let mut store = Arc::new(SqlStore::from_path(ws.read().store_path()).await?);
        let mut conn = store.open_external_writer_conn()?;


        let import_id = store.save_import("test".to_string()).await?;

        store.save_label("24279").await?;
        store.save_label("116570").await?;
        store.save_label("555123").await?;



        store.save_resources_labels(
            &HashMap::from([
                ("XC1039628.ogg".to_string(), (ResourceStats::new(vec![1, 100]), vec![("24279".into(), 1.0)]),)
            ]),
            1,
        ).await?;
        store.save_resources_labels(
            &HashMap::from([
                ("iNat1460166.ogg".to_string(), (ResourceStats::new(vec![1, 100]), vec![("116570".into(), 1.0)]),)
            ]),
            1,
        ).await?;

        store.save_resources_labels(
            &HashMap::from([
                ("iNat336366.ogg".to_string(), (ResourceStats::new(vec![1, 100]), vec![("555123".into(), 1.0)]),),
                ("iNat859875.ogg".to_string(), (ResourceStats::new(vec![1, 100]), vec![("555123".into(), 1.0)]),),
                ("iNat1370876.ogg".to_string(), (ResourceStats::new(vec![1, 100]), vec![("555123".into(), 1.0)]),),
            ]),
            import_id,
        ).await?;
        let t = conn.transaction()?;


        SqlStore::save_pure_sets(1, false, &t)?;
        SqlStore::save_pure_sets(3, false, &t)?;

        SqlStore::save_all_sets(&t)?;

        t.commit()?;


        let tensors_path = ws.read().tensors_path().clone();

        let loader = Arc::new(RawTensorLoader::<RunCtxImpl>::new(
            move |key, _| {
                let tensor = Tensor::load(tensors_path.join(format!("{}.pt", key.file_stem())))?;
                debug!("test_motifs: key={key:?} loaded tensor={tensor:?}");
                Ok(ThoTensor::new(SynKey::new([mono_key!(key.id(), format!("resource_id={}", key.id()))]), tensor)?)
            },
            ThoDevice::from(Device::Cpu),
        ));

        let mut ctx = RunCtxImpl::new(1, "test_run_all");

        let run_all_args = RunAllArgs{
            find_resources: "group:pure group 3".to_string(),
            select_resources: "group:pure group 1".to_string(),
            max_chunk_size: 100000,
            train_resource_group: "pure group 3".to_string(),
            valid_resource_group: "pure group 1".to_string(),
            train_motifs_epoch_size: 10,
            max_motif_net_loss: 0.4,
            selected_motif_nets_limit: None,
            motif_slices_step: 10,
            with_plots: false,
            resolution: Resolution::Normal,
            labels: None,
        };
        let all_configs = run_all_config_0(&run_all_args);

        run_all(
            &ws,
            &store,
            &loader,
            ctx,
            &run_all_args,
            all_configs,
        ).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_0() {}
}
