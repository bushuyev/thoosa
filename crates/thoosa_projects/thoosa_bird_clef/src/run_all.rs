use std::ops::Range;
use parking_lot::RwLock;
use std::sync::Arc;
use log::{debug, info};
use tch::Device::Cpu;
use thoosa_common::find_motifs_run_config::{AccumulateStatsRangeOrValue, AggrRangeOrValue, DotProductConfigRangeOrValue, EucDistConfigRangeOrValue, FindMotifsRunConfigRangeOrValue, MaxSlicesAbsRangeOrValue, MaxSlicesRangeOrValue, MeanAggrRangeOrValue, MetricsRangeOrValue, MotifCriterionConstraintRangeOrValue, NormThicknessRunConfigRangeOrValue, PearsonCorrConfigRangeOrValue, PrecalculateStatsRangeOrValue, QuantileAggrRangeOrValue, SelectMotifsRunConfigRangeOrValue, SimilarityMetricRangeOrValue, SlicesFilterRangeOrValue};
use thoosa_common::find_motifs_run_config::InputFilterRangeOrValue::AverageFilter;
use thoosa_common::find_motifs_run_config::InputFilterRangeOrValue::BandPassFilter;
use thoosa_common::find_motifs_run_config::InputFilterRangeOrValue::Abs;
use thoosa_common::find_motifs_run_config::InputFilterRangeOrValue::MedianFilter;
use thoosa_common::runner::common::cachable::DevicesWrap;
use thoosa_common::runner::common::RunCtx;
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_common::runs::Simple;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::train_nets;
use thoosa_common::train_nets::{AdamWRangeOrValue, CommonTrainingRunConfigRangeOrValue, FinalNetArgsRangeOrValue, LabelNetTypeArgsRangeOrValue, LinNetArgsRangeOrValue, MotifCosineNetArgsRangeOrValue, MotifsBy, OptConfig, OptimizersRangeOrValue, SelfAttentionNetArgsRangeOrValue, TrainFinalNetRunConfigRangeOrValue, TrainLabelNetsRunConfig, TrainLabelNetsRunConfigRangeOrValue, TrainMotifsNetsRunConfig, TrainMotifsNetsRunConfigRangeOrValue};
use thoosa_common::transforms::filters::{AbsFilterParams, AbsFilterParamsRangeOrValue, AverageFilterParamsRangeOrValue, BandPassFilterParamsRangeOrValue, MedianFilterParamsRangeOrValue, PaddingRangeOrValue, ZeroPaddingRangeOrValue};
use thoosa_common::ws::Workspace;
use thoosa_main::cli::{dyn_step_find_and_select, median_window_from_find_and_select, median_window_from_find_motifs, median_window_from_find_motifs_grid, run_grid, run_opt, run_plot, FindAndSelect, RunAllArgs};
use thoosa_main::{tasks, RawTensorLoader, TrainMotifNetInput, TrainMotifNets};
use thoosa_main::tasks::cross_loader::ResourcesConfigInput;
use thoosa_main::tasks::find_and_select_motifs::FindAndSelectMotifsRunConfigFields::SelectMotifsRunConfig;
use thoosa_main::tasks::find_and_select_motifs::{FindAndSelectMotifsInput, FindAndSelectMotifsRunConfig, FindAndSelectMotifsRunConfigFields, FindAndSelectMotifsRunConfigRangeOrValue, FindAndSelectMotifsStage};
use thoosa_main::tasks::motif_loader::MotifLoader;
use thoosa_main::tasks::select_motifs::{SelectMotifsInput, SelectMotifsStage};
use thoosa_main::tasks::train_label_nets::new_train_label_nets;
use thoosa_motifs::find::FixedFindMotifsConfig;
use thoosa_motifs::fragments::{DataBy, PlotDataArgs};
use thoosa_shared::{collect_futures_with_progress, panic_to_err_async, RangeOrValue, ResourceType, StreamBuffering, ThoError};
use thoosa_shared::ResourceType::NotInGroup;
use thoosa_shared::run_configs::RunId;
use thoosa_store::Store;

pub(crate) struct AllConfigs {
    pub find_and_select: FindAndSelect,


    pub train_motif_nets_run_config_range_or_value:TrainMotifsNetsRunConfigRangeOrValue,
    pub train_label_nets_run_config_range_or_value:TrainLabelNetsRunConfigRangeOrValue,
    pub train_final_net_run_config_range_or_value:TrainFinalNetRunConfigRangeOrValue,
    pub train_motifs_opt_config:Arc<OptConfig>,
    pub train_labels_opt_config:Arc<OptConfig>,
    pub train_final_opt_config:Arc<OptConfig>,
}


#[allow(unreachable_code)]
pub(crate) async fn run_all(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    loader: &Arc<RawTensorLoader<RunCtxImpl>>,
    mut root_ctx: RunCtxImpl,
    run_all_args: &RunAllArgs,
    all_config:AllConfigs
) -> Result<(), ThoError> {
    debug!("run_all");

    let labels = store.list_labels(true).await?;
    debug!("run_all: labels={labels:?}");
    let labels_filter = run_all_args.labels_filter();
    let labels_filter_ref = labels_filter.as_ref();


    match &all_config.find_and_select {
        FindAndSelect::GridOnly { find_motifs_run_config_range_or_value, fixed_find_motifs_config, select_motifs_run_config } => {
            let find_motifs_run_config = Arc::new(find_motifs_run_config_range_or_value.clone());
            let select_motifs_run_config = Arc::new(select_motifs_run_config.clone());
            let train_motif_nets_run_config_range_or_value = Arc::new(all_config.train_motif_nets_run_config_range_or_value);
            let train_label_nets_run_config_range_or_value = Arc::new(all_config.train_label_nets_run_config_range_or_value);

            let find_resources = run_all_args.find_resources.clone();
            let select_resources = run_all_args.select_resources.clone();

            let futures = labels.iter().filter(|label|
                {
                    labels_filter_ref
                        .map(|labels| labels.iter().any(|filter_label| filter_label.eq_ignore_ascii_case(label)))
                        .unwrap_or(true)
                }
            ).map(|label| {
                info!("run_all: label={label}");

                let fixed_find_motif_config = fixed_find_motifs_config.clone();

                let mut label_ctx = root_ctx.branch(Some(&format!("{label}_ctx")), true);
                let mut init_ctx_find_motifs = label_ctx.branch(Some("init_ctx_find_motifs"), false);

                let find_motifs_run_config = Arc::clone(&find_motifs_run_config);
                let select_motifs_run_config = Arc::clone(&select_motifs_run_config);
                let train_motif_nets_run_config_range_or_value = Arc::clone(&train_motif_nets_run_config_range_or_value);
                let train_label_nets_run_config_range_or_value = Arc::clone(&train_label_nets_run_config_range_or_value);
                let train_labels_opt_config = Arc::clone(&all_config.train_labels_opt_config);
                let train_motifs_opt_config = Arc::clone(&all_config.train_motifs_opt_config);

                let find_resources = find_resources.clone();
                let select_resources = select_resources.clone();
                async move {
                    panic_to_err_async!(async move {
                    let find_motifs_input = Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(Simple::new(ResourceType::Set(format!("{find_resources}_{label}_train"))))));
                    let find_motifs_run_config_rov = Arc::clone(&find_motifs_run_config);

                    let fixed_find_motif_config = fixed_find_motif_config.clone();

                    let find_motifs_run_ids = run_grid::<_, _, _, _, _>(
                        None,
                        find_motifs_run_config_rov,
                        find_motifs_input,
                        &ws,
                        &store,
                        async move |run_configs| {
                            debug!("run_all: run_configs={run_configs:?}");

                            Arc::new(
                                tasks::find_motifs::FindMotifsStage::new(
                                    &ws,
                                    &store,
                                    Arc::clone(loader),
                                    run_configs,
                                    fixed_find_motif_config,
                                    ThoDevice::from(1),
                                    init_ctx_find_motifs
                                ).await.expect("FindMotifsStage::new failed"),
                            )
                        },
                        || label_ctx.branch(Some("find_motifs"), true),
                    ).await?;

                    info!("run_all: label={label} find_motifs_run_ids={find_motifs_run_ids:?}");
                    let select_motifs_run_config = Arc::clone(&select_motifs_run_config);

                    let select_motifs_input = Arc::new(DevicesWrap::wrap(SelectMotifsInput::standalone(ResourcesConfigInput::new(run_all_args.select_resource_config()?).check(&store).await?)));

                    //gpt Run a grid over SelectMotifs configs; each run consumes FindMotifs outputs and produces selected motifs.
                    // let mut select_motifs_run_ids = vec![];
                    //
                    // for find_motifs_run_id in &find_motifs_run_ids {
                    //     select_motifs_run_ids.extend(run_grid::</*Labeled,*/ _, _, _, _, _>(
                    //         Some(vec![*find_motifs_run_id]),
                    //         Arc::clone(&select_motifs_run_config),
                    //         Arc::clone(&select_motifs_input),
                    //         &ws,
                    //         &store,
                    //         async |run_configs| {
                    //             Arc::new(SelectMotifsStage::<RawTensorLoader<RunCtxImpl>, MotifLoader<RawTensorLoader<RunCtxImpl>, RunCtxImpl>, RunCtxImpl>::new(
                    //                 &ws,
                    //                 &store,
                    //                 Arc::clone(&loader),
                    //                 Arc::new(MotifLoader::new(
                    //                     Arc::clone(&store),
                    //                     Arc::clone(&loader),
                    //                     &ws
                    //                 )),
                    //                 run_configs,
                    //                 vec![*find_motifs_run_id],
                    //                 run_all_args.max_chunk_size as i64,
                    //                 // #[cfg(test)]
                    //                 None
                    //             ).await.expect("SelectMotifsStage::new failed"))
                    //         },
                    //         || label_ctx.branch(Some("select_motifs"), true),
                    //     ).await?);
                    // }
                    let select_motifs_run_ids = store.select_all_motifs(&find_motifs_run_ids).await?;
                    info!("select motifs done: label={label} select_motifs_run_ids={select_motifs_run_ids:?}");

                    if run_all_args.with_plots {
                        for select_motifs_run_id in &select_motifs_run_ids {
                            run_plot(&ws, &store, &loader, &PlotDataArgs::new(DataBy::SelectedMotifs { select_motifs_run_ids: vec![*select_motifs_run_id] }), run_all_args.resolution, &Cpu, label_ctx.branch("plot".into(), false)).await?;
                        }
                    }

                    let input_wrapper = Arc::new(DevicesWrap::wrap(TrainMotifNetInput::new(MotifsBy::SelectMotifsRunIds { select_motifs_run_ids: select_motifs_run_ids.clone() }, train_motifs_opt_config)));

                    //gpt Run a grid over motif-net training configs; output is a set of run ids for trained motif nets.
                    let train_motif_nets_run_ids = run_opt::</*Labeled,*/ _, _, TrainMotifsNetsRunConfig, _, _>(
                        None,
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
                            )))
                        },
                        || label_ctx.branch(Some("train motif nets"), true),
                    ).await?;

                    info!("train motif nets done: label={label} train_motif_nets_run_ids={train_motif_nets_run_ids:?}");

                    //gpt Pre-create context and shared handles for label-net stage construction.
                    let init_ctx_train_labels = label_ctx.branch(Some("init label nets"), true);
                    let store_in_train_labels = Arc::clone(&store);
                    let ws_in_train_labels = Arc::clone(&ws);


                    let train_labels_input = Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(
                        run_all_args.train_resource_config()
                    )/*.check(&store).await?*/));

                    //gpt Run a grid over label-net training configs; consumes motif nets from previous stage.
                    match run_opt::<_, _, TrainLabelNetsRunConfig, _, _>(
                        None,
                        train_label_nets_run_config_range_or_value,
                        train_labels_input,
                        &ws,
                        &store,
                        async move |run_configs| {
                            new_train_label_nets(
                                train_motif_nets_run_ids,
                                Arc::clone(&loader),
                                run_configs,
                                &ws_in_train_labels,
                                &store_in_train_labels,
                                init_ctx_train_labels,
                                train_labels_opt_config,
                                Some(vec![label.clone()]),
                                Some(run_all_args.max_motif_net_loss), //to RunAll args,
                                run_all_args.selected_motif_nets_limit,
                                run_all_args.motif_slices_step
                            ).await.map(|v| Arc::new(v))
                        },
                        || label_ctx.branch(Some("train label nets"), true),
                    ).await {
                        Ok(train_label_nets_run_ids) => {
                            info!("train label nets done: label={label} run_ids={train_label_nets_run_ids:?}");
                        }
                        Err(e) => {
                            info!("train label nets failed: label={label} e={e:?}");
                        }
                    };

                    Ok::<_, ThoError>(())
                        })
                }
            });

            let _ = collect_futures_with_progress(
                futures,
                labels.len(),
                4,
                StreamBuffering::Unordered,
                "bird_clef_labels",
            ).await;

            Ok::<_, ThoError>(())
        }
        FindAndSelect::WithOpt { find_and_select_run_config_range_or_value, fixed_find_motifs_config, find_and_select_motifs_opt_config } => {
            let futures = labels.iter().filter(|label|
                {
                    labels_filter_ref
                        .map(|labels| labels.iter().any(|filter_label| filter_label.eq_ignore_ascii_case(label)))
                        .unwrap_or(true)
                }
            ).map(|label| {
                info!("run_all: label={label}");

                let fixed_find_motif_config = fixed_find_motifs_config.clone();

                let mut label_ctx = root_ctx.branch(Some(&format!("{label}_ctx")), true);
                let mut init_ctx_find_motifs = label_ctx.branch(Some("init_ctx_find_motifs"), false);

                let train_motif_nets_run_config_range_or_value = Arc::new(all_config.train_motif_nets_run_config_range_or_value.clone());
                let train_label_nets_run_config_range_or_value = Arc::new(all_config.train_label_nets_run_config_range_or_value.clone());

                let train_motif_nets_run_config_range_or_value = Arc::clone(&train_motif_nets_run_config_range_or_value);
                let train_label_nets_run_config_range_or_value = Arc::clone(&train_label_nets_run_config_range_or_value);
                let train_labels_opt_config = Arc::clone(&all_config.train_labels_opt_config);
                let train_motifs_opt_config = Arc::clone(&all_config.train_motifs_opt_config);
                //
                // let find_resources = find_resources.clone();
                // let select_resources = select_resources.clone();
                let find_and_select_run_config_range_or_value = find_and_select_run_config_range_or_value.clone();
                let find_and_select_motifs_opt_config = find_and_select_motifs_opt_config.clone();
                // let find_resources =
                let find_and_select_input = Arc::new(DevicesWrap::wrap(FindAndSelectMotifsInput::new(
                    format!("set:{}_{label}_train", run_all_args.find_resources), //run_all_args.find_resources.clone(),
                    run_all_args.select_resources.clone(),
                )));
                let mut ctx = root_ctx.branch(Some("opt find and select motifs"), true);
                async move {
                    let best_find_and_select_result_ids: Vec<RunId> = run_opt::</*Simple,*/ _, _, FindAndSelectMotifsRunConfig, _, _>(
                        None,
                        find_and_select_run_config_range_or_value,
                        find_and_select_input,
                        &ws,
                        &store,
                        async |run_configs| {
                            Ok(Arc::new(FindAndSelectMotifsStage::new(
                                Arc::clone(&loader),
                                &ws,
                                &store,
                                run_configs,
                                fixed_find_motif_config,
                                Arc::new(find_and_select_motifs_opt_config),
                                run_all_args.max_chunk_size as i64,
                            )))
                        },
                        || ctx.branch(Some("opt find and select motifs"), false),
                    ).await?;

                    info!("best_find_and_select_result_ids={best_find_and_select_result_ids:?}");

                    let select_motifs_run_ids = store
                        .list_input_run_ids(&best_find_and_select_result_ids)
                        .await?
                        .into_iter().map(
                        |opt_select_result_ids| opt_select_result_ids.1
                    ).collect::<Vec<_>>();

                    if run_all_args.with_plots {
                        for select_motifs_run_id in &select_motifs_run_ids {
                            run_plot(&ws, &store, &loader, &PlotDataArgs::new(DataBy::SelectedMotifs { select_motifs_run_ids: vec![*select_motifs_run_id] }), run_all_args.resolution, &Cpu, ctx.branch("plot".into(), false)).await?;
                        }
                    }


                    info!("opt FindAndSelectMotifsRunConfig done: select_motifs_run_ids={select_motifs_run_ids:?}");

                    let input_wrapper = Arc::new(DevicesWrap::wrap(TrainMotifNetInput::new(MotifsBy::SelectMotifsRunIds { select_motifs_run_ids: select_motifs_run_ids.clone() }, train_motifs_opt_config)));

                    let train_motif_nets_run_ids = run_opt::</*Labeled,*/ _, _, TrainMotifsNetsRunConfig, _, _>(
                        None,
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
                            )))
                        },
                        || label_ctx.branch(Some("train motif nets"), true),
                    ).await?;

                    info!("train motif nets done: train_motif_nets_run_ids={train_motif_nets_run_ids:?}");

                    //gpt Pre-create context and shared handles for label-net stage construction.
                    let init_ctx_train_labels = label_ctx.branch(Some("init label nets"), true);
                    let store_in_train_labels = Arc::clone(&store);
                    let ws_in_train_labels = Arc::clone(&ws);


                    let train_labels_input = Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(
                        run_all_args.train_resource_config()
                    )/*.check(&store).await?*/));

                    //gpt Run a grid over label-net training configs; consumes motif nets from previous stage.
                    let train_label_nets_run_ids = run_opt::<_, _, TrainLabelNetsRunConfig, _, _>(
                        None,
                        train_label_nets_run_config_range_or_value,
                        train_labels_input,
                        &ws,
                        &store,
                        async move |run_configs| {
                            new_train_label_nets(
                                train_motif_nets_run_ids,
                                Arc::clone(&loader),
                                run_configs,
                                &ws_in_train_labels,
                                &store_in_train_labels,
                                init_ctx_train_labels,
                                train_labels_opt_config,
                                Some(vec![label.clone()]),
                                Some(run_all_args.max_motif_net_loss), //to RunAll args,
                                run_all_args.selected_motif_nets_limit,
                                run_all_args.motif_slices_step
                            ).await.map(|v| Arc::new(v))
                        },
                        || label_ctx.branch(Some("train label nets"), true),
                    ).await?;

                    info!("train label nets done: run_ids={train_label_nets_run_ids:?}");
                    Ok(())
                }

            });

            let _ = collect_futures_with_progress(
                futures,
                labels.len(),
                4,
                StreamBuffering::Unordered,
                "bird_clef_labels",
            ).await;

            Ok(())
        }
    }

}

pub(crate) fn run_all_opt_config_0(run_all_params: &RunAllArgs) -> AllConfigs {

    let find_and_select =  FindAndSelect::WithOpt {
        find_and_select_run_config_range_or_value: FindAndSelectMotifsRunConfigRangeOrValue {
            id: None,
            find_motifs_run_config: FindMotifsRunConfigRangeOrValue {
                id: None,
                step: RangeOrValue::Value(1000),
                window: RangeOrValue::Range{ range: Range {start: 2000, end: 16000}, step: Some((2000, 0)) },
                // window: RangeOrValue::Value(1000),
                // freq_filter: None,
                // freq_filter: Some(vec![
                //     AverageFilter(AverageFilterParamsRangeOrValue {
                //         window: RangeOrValue::dynamic(median_window_from_find_motifs_grid),
                //         step: RangeOrValue::Value(1),
                //     }),
                // ]),

                freq_filter: Some(vec![BandPassFilter(BandPassFilterParamsRangeOrValue {
                    low_hz: RangeOrValue::Range { range: Range { start: 0., end: 16000. }, step: None },
                    band_hz: RangeOrValue::Range { range: Range { start: 2., end: 100. }, step: None },
                    // band_hz: RangeOrValue::Value(2.),
                    // kernel_size: RangeOrValue::Value(1000),
                    kernel_size: RangeOrValue::Range { range: Range { start: 1000, end: 8000 }, step: None },
                    stride: RangeOrValue::Value(1),
                    padding: PaddingRangeOrValue::Zero(ZeroPaddingRangeOrValue::default()),
                })]),
                // similarity_metric: SimilarityMetricRangeOrValue::EucDist(EucDistConfigRangeOrValue { z_normalize: RangeOrValue::Value(false) }),
                // similarity_metric: SimilarityMetricRangeOrValue::DotProduct(DotProductConfigRangeOrValue{ _path_pd: Default::default() }),
                similarity_metric: SimilarityMetricRangeOrValue::PearsonCorr(PearsonCorrConfigRangeOrValue{ _path_pd: Default::default() }),
                // slices_filter: SlicesFilterRangeOrValue::Precalculate(PrecalculateStatsRangeOrValue {
                //     // aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue{ k: RangeOrValue::Value(10.0) }),
                //     aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue { q: RangeOrValue::Value(1.0) }),
                //     slices: RangeOrValue::Value(100),
                // }),
                slices_filter: SlicesFilterRangeOrValue::Accumulate(AccumulateStatsRangeOrValue {
                    // aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue{ k: RangeOrValue::Value(1.0) }),
                    aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue { q: RangeOrValue::Value(0.05) }),
                }),
                max_slices: Some(MaxSlicesRangeOrValue::Abs(MaxSlicesAbsRangeOrValue { count: RangeOrValue::Value(1000) })),
            },
            select_motifs_run_config: SelectMotifsRunConfigRangeOrValue {
                id: None,
                extend_config: None,
                motif_criteria: vec![
                    MotifCriterionConstraintRangeOrValue {
                        metric: MetricsRangeOrValue::NormThickness(NormThicknessRunConfigRangeOrValue {
                            eps: RangeOrValue::Value(1e-8),
                        }),
                        min: RangeOrValue::Value(0.5),
                    }
                ],
                extend_max_slices: None,
            },
        },
        fixed_find_motifs_config: FixedFindMotifsConfig::new(run_all_params.max_chunk_size, false),
        find_and_select_motifs_opt_config: OptConfig { opt_max_evals: 100, opt_num_init: 10 },
    };


    /////////////////////train_motifs
    let train_motif_nets_run_config_range_or_value: TrainMotifsNetsRunConfigRangeOrValue = TrainMotifsNetsRunConfigRangeOrValue {
        id: None,
        common_training_args: CommonTrainingRunConfigRangeOrValue {
            epochs: RangeOrValue::Value(10),
            batch_size: RangeOrValue::Value(10),
            // lr: RangeOrValue::Value(0.0001),
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
        // net: train_nets::MotifNetTypeArgsRangeOrValue::ConvNet(train_nets::MotifConvNetArgsRangeOrValue {
        //     num_layers: RangeOrValue::Range { range: Range { start: 3, end: 5 }, step: None },
        //     step: RangeOrValue::Value(10),
        //     channels: RangeOrValue::Value(32),
        //     kernel: RangeOrValue::Value(10),
        // }),

        net: train_nets::MotifNetTypeArgsRangeOrValue::CosineNet(MotifCosineNetArgsRangeOrValue{
            // step: RangeOrValue::Value(1),
            // channels: RangeOrValue::Range { range: Range { start: 4, end: 32 }, step: None },
            // kernel: RangeOrValue::Range { range: Range { start: 10, end: 100 }, step: None },
            a: RangeOrValue::Range { range: Range { start: 1.0, end: 10.0 }, step: None },
            b: RangeOrValue::Range { range: Range { start: 0.0, end: 1.0 }, step: None },
        })

        // net: train_nets::MotifNetTypeArgsRangeOrValue::ConvNet(train_nets::MotifConvNetArgsRangeOrValue {
        //     conv_layers: RangeOrValue::Range { range: Range { start: 2, end: 24 }, step: None },
        //     head_layers: RangeOrValue::Range { range: Range { start: 2, end: 24 }, step: None },
        //     step: RangeOrValue::Value(10),
        //     channels: RangeOrValue::Range{ range: Range {start: 8, end: 256}, step: None },
        //     // kernel: RangeOrValue::Value(10),
        //     kernel: RangeOrValue::Range{ range: Range {start: 10, end: 100}, step: None },
        //     head_pool: None,
        // })

    };

    //////////////////////train labels
    let train_label_nets_run_config_range_or_value: TrainLabelNetsRunConfigRangeOrValue = TrainLabelNetsRunConfigRangeOrValue {
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
        net: LabelNetTypeArgsRangeOrValue::SelfAttentionNet(SelfAttentionNetArgsRangeOrValue {
            // d_model: RangeOrValue::Value(64),
            head_size: RangeOrValue::Range { range: Range { start: 64, end: 256 }, step: None },
            n_heads: RangeOrValue::Range { range: Range { start: 4, end: 12 }, step: None },
            // ff_mult: RangeOrValue::Range { range: Range { start: 2, end: 4 }, step: None },
            ff_mult: RangeOrValue::Value(2),
            // dropout: RangeOrValue::Value(0.1),
            dropout: RangeOrValue::Range { range: Range { start: 0.01, end: 0.2 }, step: None },
        })

    };
    //////////////////////train final
    let train_final_net_run_config_range_or_value: TrainFinalNetRunConfigRangeOrValue = TrainFinalNetRunConfigRangeOrValue {
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
        net: FinalNetArgsRangeOrValue {
            num_layers: RangeOrValue::Range { range: Range { start: 4, end: 32 }, step: None },
        },
    };
    AllConfigs{
        find_and_select,
        train_motif_nets_run_config_range_or_value,
        train_label_nets_run_config_range_or_value,
        train_final_net_run_config_range_or_value,
        train_motifs_opt_config: Arc::new(OptConfig::new(30, 10)),
        train_labels_opt_config: Arc::new(OptConfig::new(30, 10)),
        train_final_opt_config: Arc::new(OptConfig::new(30, 10))
    }
}


pub(crate) fn run_all_grid_config_0(run_all_params: &RunAllArgs) -> AllConfigs {

    let find_and_select =  FindAndSelect::GridOnly {
        find_motifs_run_config_range_or_value: FindMotifsRunConfigRangeOrValue {
            id: None,
            step: RangeOrValue::Value(1000),
            window: RangeOrValue::Range{ range: Range {start: 2000, end: 128201}, step: Some((16000, 0)) },
            // window: RangeOrValue::Value(10000),
            freq_filter: None,
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
            // similarity_metric: SimilarityMetricRangeOrValue::EucDist(EucDistConfigRangeOrValue { z_normalize: RangeOrValue::Value(false) }),
            similarity_metric: SimilarityMetricRangeOrValue::DotProduct(DotProductConfigRangeOrValue{ _path_pd: Default::default() }),
            // similarity_metric: SimilarityMetricRangeOrValue::PearsonCorr(PearsonCorrConfigRangeOrValue{ _path_pd: Default::default() }),
            // slices_filter: SlicesFilterRangeOrValue::Precalculate(PrecalculateStatsRangeOrValue {
            //     aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue{ k: RangeOrValue::Value(5.0) }),
            //     // aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue { q: RangeOrValue::Value(1.0) }),
            //     slices: RangeOrValue::Value(100),
            // }),
            slices_filter: SlicesFilterRangeOrValue::Accumulate(AccumulateStatsRangeOrValue {
                // aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue{ k: RangeOrValue::Value(1.0) }),
                aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue { q: RangeOrValue::Value(0.5) }),
            }),
            max_slices: Some(MaxSlicesRangeOrValue::Abs(MaxSlicesAbsRangeOrValue { count: RangeOrValue::Value(1000) })),
        },

        fixed_find_motifs_config: FixedFindMotifsConfig::new(run_all_params.max_chunk_size, false),
        select_motifs_run_config: SelectMotifsRunConfigRangeOrValue {
            id: None,
            extend_config: None,
            motif_criteria: vec![
                // MotifCriterionConstraintRangeOrValue {
                //     metric: MetricsRangeOrValue::NormThickness(NormThicknessRunConfigRangeOrValue {
                //         eps: RangeOrValue::Value(1e-8),
                //     }),
                //     min: RangeOrValue::Value(0.5),
                // }
            ],
            extend_max_slices: None,
        },
    };

    /////////////////////train_motifs
    let train_motif_nets_run_config_range_or_value: TrainMotifsNetsRunConfigRangeOrValue = TrainMotifsNetsRunConfigRangeOrValue {
        id: None,
        common_training_args: CommonTrainingRunConfigRangeOrValue {
            epochs: RangeOrValue::Value(10),
            batch_size: RangeOrValue::Value(10),
            // lr: RangeOrValue::Value(0.0001),
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
        // net: train_nets::MotifNetTypeArgsRangeOrValue::ConvNet(train_nets::MotifConvNetArgsRangeOrValue {
        //     num_layers: RangeOrValue::Range { range: Range { start: 3, end: 5 }, step: None },
        //     step: RangeOrValue::Value(10),
        //     channels: RangeOrValue::Value(32),
        //     kernel: RangeOrValue::Value(10),
        // }),

        net: train_nets::MotifNetTypeArgsRangeOrValue::CosineNet(MotifCosineNetArgsRangeOrValue{
            // step: RangeOrValue::Value(1),
            // channels: RangeOrValue::Range { range: Range { start: 4, end: 32 }, step: None },
            // kernel: RangeOrValue::Range { range: Range { start: 10, end: 100 }, step: None },
            a: RangeOrValue::Range { range: Range { start: 1.0, end: 10.0 }, step: None },
            b: RangeOrValue::Range { range: Range { start: 0.0, end: 1.0 }, step: None },
        })

        // net: train_nets::MotifNetTypeArgsRangeOrValue::ConvNet(train_nets::MotifConvNetArgsRangeOrValue {
        //     conv_layers: RangeOrValue::Range { range: Range { start: 2, end: 24 }, step: None },
        //     head_layers: RangeOrValue::Range { range: Range { start: 2, end: 24 }, step: None },
        //     step: RangeOrValue::Value(10),
        //     channels: RangeOrValue::Range{ range: Range {start: 8, end: 256}, step: None },
        //     // kernel: RangeOrValue::Value(10),
        //     kernel: RangeOrValue::Range{ range: Range {start: 10, end: 100}, step: None },
        //     head_pool: None,
        // })

    };

    //////////////////////train labels
    let train_label_nets_run_config_range_or_value: TrainLabelNetsRunConfigRangeOrValue = TrainLabelNetsRunConfigRangeOrValue {
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
            features: RangeOrValue::Range { range: Range { start: 100, end: 10000 }, step: None },
        })

    };
    //////////////////////train final
    let train_final_net_run_config_range_or_value: TrainFinalNetRunConfigRangeOrValue = TrainFinalNetRunConfigRangeOrValue {
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
        net: FinalNetArgsRangeOrValue {
            num_layers: RangeOrValue::Range { range: Range { start: 4, end: 32 }, step: None },
        },
    };
    AllConfigs{
        find_and_select,
        train_motif_nets_run_config_range_or_value,
        train_label_nets_run_config_range_or_value,
        train_final_net_run_config_range_or_value,
        train_motifs_opt_config: Arc::new(OptConfig::new(30, 10)),
        train_labels_opt_config: Arc::new(OptConfig::new(30, 10)),
        train_final_opt_config: Arc::new(OptConfig::new(30, 10))
    }
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use crate::run_all::{run_all, run_all_grid_config_0};
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
        let all_configs = run_all_grid_config_0(&run_all_args);

        run_all(
            &ws,
            &store,
            &loader,
            ctx,
            &run_all_args,
            all_configs,
        ).await
    }

}