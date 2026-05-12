use std::ops::Range;
use std::sync::Arc;
use log::info;
use parking_lot::RwLock;
use tch::Device::Cpu;
use thoosa_main::tasks::find_motifs_run_config::{AggrRangeOrValue, CrossLoaderRunConfigRangeOrValue, ExtendMotifsConfigRangeOrValue, FinderKindRangeOrValue, FindMotifsRunConfigRangeOrValue, MaxSlicesAbsRangeOrValue, MaxSlicesRangeOrValue, MetricsRangeOrValue, MotifCriterionConstraintRangeOrValue, NormThicknessRunConfigRangeOrValue, PeriodicityAcfRunConfigRangeOrValue, PrecalculateStatsRangeOrValue, QuantileAggrRangeOrValue, SelectMotifsRunConfigRangeOrValue, SlicesFilterRangeOrValue};
use thoosa_common::runner::common::cachable::DevicesWrap;
use thoosa_common::runner::common::RunCtx;
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::train_nets;
use thoosa_common::train_nets::{AdamWRangeOrValue, CommonTrainingRunConfigRangeOrValue, FinalNetArgsRangeOrValue, LabelNetTypeArgsRangeOrValue, MotifCosineNetArgsRangeOrValue, MotifsBy, OptConfig, OptimizersRangeOrValue, SelfAttentionNetArgsRangeOrValue, TrainFinalNetRunConfigRangeOrValue, TrainLabelNetRunConfig, TrainLabelNetRunConfigRangeOrValue, TrainMotifNetRunConfig, TrainMotifNetRunConfigRangeOrValue};
use thoosa_common::transforms::filters::AverageFilterParamsRangeOrValue;
use thoosa_common::transforms::filters::InputFilterRangeOrValue::AverageFilter;
use thoosa_common::ws::Workspace;
use thoosa_main::cli::{dyn_step_find_and_select, median_window_from_find_and_select, run_grid, run_opt, run_plot, RunAllArgs};
use thoosa_main::{tasks, RawTensorLoader, TrainMotifNetInput, TrainMotifNets};
use thoosa_main::tasks::cross_loader::ResourcesConfigInput;
use thoosa_main::tasks::find_and_select_motifs::{FindAndSelectMotifsInput, FindAndSelectMotifsRunConfig, FindAndSelectMotifsRunConfigFields, FindAndSelectMotifsRunConfigRangeOrValue, FindAndSelectMotifsStage};
use thoosa_main::tasks::motif_loader::MotifLoader;
use thoosa_main::tasks::select_motifs::{SelectMotifsInput, SelectMotifsStage};
use thoosa_main::tasks::train_label_nets::new_train_label_nets;
use thoosa_motifs::distance::{DotProductConfigRangeOrValue, EucDistConfigRangeOrValue, SimilarityMetricRangeOrValue};
use thoosa_motifs::find::FixedFindMotifsConfig;
use thoosa_motifs::fragments::{DataBy, PlotDataArgs};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::{RangeOrValue, ThoError};
use thoosa_shared::hyperparameter::FieldName;
use thoosa_shared::val::NamedVal;
use thoosa_store::Store;

pub enum FindAndSelect {
    GridOnly{
        find_motifs_run_config_range_or_value: FindMotifsRunConfigRangeOrValue,
        fixed_find_motifs_config:FixedFindMotifsConfig,
        select_motifs_run_config:SelectMotifsRunConfigRangeOrValue,
    },
    WithOpt{
        find_and_select_run_config_range_or_value: FindAndSelectMotifsRunConfigRangeOrValue,
        fixed_find_motifs_config:FixedFindMotifsConfig,
        find_and_select_motifs_opt_config:OptConfig,
    }
}


pub(crate) struct AllConfigs {
    pub find_and_select: FindAndSelect,


    pub train_motif_nets_run_config_range_or_value:TrainMotifNetRunConfigRangeOrValue,
    pub train_label_nets_run_config_range_or_value:TrainLabelNetRunConfigRangeOrValue,
    #[allow(dead_code)]
    pub train_final_net_run_config_range_or_value:TrainFinalNetRunConfigRangeOrValue,
    pub train_motifs_opt_config:Arc<OptConfig>,
    pub train_labels_opt_config:Arc<OptConfig>,
    #[allow(dead_code)]
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

    //gpt Build the input for FindMotifs: wrap resource config in DevicesWrap and allocate a fresh cache area.
    let find_motifs_input = Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(run_all_args.find_resource_config()?).check(&store).await?));

    //gpt Build the input for SelectMotifs: separate resource config so selection can use its own dataset.
    let select_motifs_input = Arc::new(DevicesWrap::wrap(SelectMotifsInput::standalone(ResourcesConfigInput::new(run_all_args.select_resource_config()?).check(&store).await?)));

    //gpt Build the input for label training: this is typically a labeled resource set.
    let train_labels_input = Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new( //TODO the group is needed to load resources not belonging to label being trained
       run_all_args.train_resource_config()
    ).check(&store).await?));

    //gpt Build the input for final net training: usually a higher-level resource config (e.g., logits inputs).
    let _train_final_input = Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(
        run_all_args.train_final_resource_config()).check(&store).await?));


    //gpt Pre-create context and shared handles for FindMotifs stage construction (kept outside the run_grid closure).
    let init_ctx_find_motifs = root_ctx.branch(Some("init find motifs"), true);
    let store_in_find_motifs = Arc::clone(&store);
    let ws_in_find_motifs = Arc::clone(&ws);
    let loader_in_find_motifs = Arc::clone(&loader);

    let find_and_select_input = Arc::new(DevicesWrap::wrap(FindAndSelectMotifsInput::new(
        run_all_args.find_resources.clone(),
        run_all_args.select_resources.clone(),
    )));


    let select_motifs_run_ids = match all_config.find_and_select {
        FindAndSelect::GridOnly { find_motifs_run_config_range_or_value, fixed_find_motifs_config, select_motifs_run_config } => {
            info!("find and select motifs opt skipped: no opt params in run config, falling back to grid search");

            //gpt Run a grid over FindMotifs configs, constructing a stage per grid point and executing it.
            let find_motifs_run_ids = run_grid::</*Simple,*/ _, _, _, _, _>(
                None,
                find_motifs_run_config_range_or_value,
                find_motifs_input,
                &ws,
                &store,
                async move |run_configs| {
                    Arc::new(
                        tasks::find_motifs::FindMotifsStage::new(
                            &ws_in_find_motifs,
                            &store_in_find_motifs,
                            loader_in_find_motifs,
                            run_configs,
                            fixed_find_motifs_config,
                            ThoDevice::from(1),
                            init_ctx_find_motifs
                        ).await.expect("FindMotifsStage::new failed"),
                    )
                },
                || root_ctx.branch(Some("find_motifs"), true),
            ).await?;

            info!("find motifs done, find_motif_run_ids={find_motifs_run_ids:?}");

            assert!(!find_motifs_run_ids.is_empty(), "no find_motifs_run_ids");

            //gpt Run a grid over SelectMotifs configs; each run consumes FindMotifs outputs and produces selected motifs.
            let mut select_motifs_run_ids = vec![];
            for find_motifs_run_id in &find_motifs_run_ids {
                select_motifs_run_ids.extend( run_grid::</*Labeled,*/ _, _, _, _, _>(
                    Some(vec![*find_motifs_run_id]),
                    select_motifs_run_config.clone(),
                    select_motifs_input.clone(),
                    &ws,
                    &store,
                    async |run_configs| {
                        Arc::new(SelectMotifsStage::<RawTensorLoader<RunCtxImpl>, MotifLoader<RawTensorLoader<RunCtxImpl>, RunCtxImpl>, RunCtxImpl>::new(
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
                            // #[cfg(test)]
                            None
                        ).await.expect("SelectMotifsStage::new failed"))
                    },
                    || root_ctx.branch(Some("select_motifs"), true),
                ).await?);

            }

            info!("select motifs done: find_motifs_run_ids={find_motifs_run_ids:?} select_motifs_run_ids={select_motifs_run_ids:?}");

            select_motifs_run_ids
        }
        FindAndSelect::WithOpt { find_and_select_run_config_range_or_value, fixed_find_motifs_config, find_and_select_motifs_opt_config } => {
            let best_find_and_select_result_ids:Vec<RunId>= run_opt::</*Simple,*/ _, _, FindAndSelectMotifsRunConfig, _, _>(
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
                        fixed_find_motifs_config,
                        Arc::new(find_and_select_motifs_opt_config),
                        run_all_args.max_chunk_size as i64,
                    )))
                },
                || root_ctx.branch(Some("opt find and select motifs"), true),
            ).await?;

            info!("opt FindAndSelectMotifsRunConfig done: find_and_select_run_ids={best_find_and_select_result_ids:?}");
            assert_eq!(best_find_and_select_result_ids.len(), 1); //should be one from run_opt

            let select_motifs_run_ids = store
                .list_input_run_ids(&best_find_and_select_result_ids)
                .await?
                .into_iter().map(
                |opt_select_result_ids| opt_select_result_ids.1
            ).collect::<Vec<_>>();

            info!("opt FindAndSelectMotifsRunConfig done: select_motifs_run_ids={select_motifs_run_ids:?}");

            select_motifs_run_ids
        }
    };


    if run_all_args.with_plots {
        for select_motifs_run_id in &select_motifs_run_ids {
            run_plot(&ws, &store, &loader, &PlotDataArgs::new(DataBy::SelectedMotifs { select_motifs_run_ids: vec![*select_motifs_run_id] }), run_all_args.resolution, &Cpu, root_ctx.branch("plot".into(), false)).await?;
        }
    }

    info!("find and select motifs  done: select_motifs_run_ids={select_motifs_run_ids:?}");

    //gpt Train motif nets needs a custom input wrapper (motif ids + opt config) rather than a resource config.
    let input_wrapper = Arc::new(DevicesWrap::wrap(TrainMotifNetInput::new(MotifsBy::SelectMotifsRunIds { select_motifs_run_ids: select_motifs_run_ids.clone() }, Arc::clone(&all_config.train_motifs_opt_config))));

    //gpt Run a grid over motif-net training configs; output is a set of run ids for trained motif nets.
    let train_motif_nets_run_ids = run_opt::</*Labeled,*/ _, _, TrainMotifNetRunConfig, _, _>(
        None,
        all_config.train_motif_nets_run_config_range_or_value,
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
        || root_ctx.branch(Some("train motif nets"), true),
    ).await?;

    info!("train motif nets done: train_motif_nets_run_ids={train_motif_nets_run_ids:?}");

    //gpt Pre-create context and shared handles for label-net stage construction.
    let init_ctx_train_labels = root_ctx.branch(Some("init label nets"), true);
    let store_in_train_labels = Arc::clone(&store);
    let ws_in_train_labels = Arc::clone(&ws);
    let labels_filter = run_all_args.labels_filter();

    //gpt Run a grid over label-net training configs; consumes motif nets from previous stage.
    let train_label_nets_run_ids = run_opt::< _, _, TrainLabelNetRunConfig, _, _>(
        None,
        all_config.train_label_nets_run_config_range_or_value,
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
                Arc::clone(&all_config.train_labels_opt_config),
                labels_filter.clone(),
                Some(run_all_args.max_motif_net_loss), //to RunAll args,
                run_all_args.selected_motif_nets_limit,
                run_all_args.motif_slices_step
            ).await.map(|v| Arc::new(v))
        },
        || root_ctx.branch(Some("train label nets"), true),
    ).await?;

    info!("train label nets done: run_ids={train_label_nets_run_ids:?}");

    /*
    ////////////////////////

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
        async move |run_configs| {
            new_train_final_net(
                train_label_nets_run_ids,
                loader_in_train_final,
                run_configs,
                &ws_in_train_final,
                &store_in_train_final,
                init_ctx_in_train_final,
                Arc::clone(&all_config.train_final_opt_config),
                10, //TODO
                run_all_args.selected_motif_nets_limit,
            ).await.map(|v| Arc::new(v))
        },
        || root_ctx.branch(Some("train final nets"), true, false),
    ).await?;
    */

    Ok(())
}

pub(crate) fn run_all_config_0(run_all_params: &RunAllArgs) -> AllConfigs {

    // let find_and_select =  FindAndSelect::GridOnly {
    //     find_motifs_run_config_range_or_value: run_all_find_motifs_config_grid_0(2, 2, 4),
    //     fixed_find_motifs_config: FixedFindMotifsConfig::new(run_all_params.max_chunk_size, false),
    //     select_motifs_run_config: run_all_select_motifs_config_grid_0::<SelectMotifsRunConfigFields>(),
    // };

    let find_and_select =  FindAndSelect::WithOpt {
        find_and_select_run_config_range_or_value: FindAndSelectMotifsRunConfigRangeOrValue {
            id: None,
            find_motifs_run_config: run_all_find_motifs_config_opt_0::<FindAndSelectMotifsRunConfigFields, _, _>(median_window_from_find_and_select, dyn_step_find_and_select),
            select_motifs_run_config: run_all_select_motifs_config_opt_0::<FindAndSelectMotifsRunConfigFields>(),
        },
        fixed_find_motifs_config: FixedFindMotifsConfig::new(run_all_params.max_chunk_size, false),
        find_and_select_motifs_opt_config: OptConfig::new(30, 10),
    };

    /////////////////////train_motifs
    let train_motif_nets_run_config_range_or_value: TrainMotifNetRunConfigRangeOrValue = TrainMotifNetRunConfigRangeOrValue {
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
    let train_label_nets_run_config_range_or_value: TrainLabelNetRunConfigRangeOrValue = TrainLabelNetRunConfigRangeOrValue {
        id: None,
        common_training_args: CommonTrainingRunConfigRangeOrValue {
            epochs: RangeOrValue::Value(100),
            batch_size: RangeOrValue::Value(10),
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
            batch_size: RangeOrValue::Value(10),
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
        train_motifs_opt_config: Arc::new(OptConfig::new(50, 10)),
        train_labels_opt_config: Arc::new(OptConfig::new(100, 10)),
        train_final_opt_config: Arc::new(OptConfig::new(30, 10))
    }
}


fn run_all_find_motifs_config_opt_0<P: FieldName + 'static, F, F1>(median_window: F, _dyn_step:F1) -> FindMotifsRunConfigRangeOrValue<P>
where
    F: Fn(&[NamedVal<P>]) -> RangeOrValue<u32> + Send + Sync + Clone + 'static,
    F1: Fn(&[NamedVal<P>]) -> RangeOrValue<u32> + Send + Sync + Clone + 'static,
{

    FindMotifsRunConfigRangeOrValue {
        id: None,
        step: RangeOrValue::Value(10),
        // step: RangeOrValue::dynamic(dyn_step),
        // window: RangeOrValue::Value(200),
        window: RangeOrValue::Range { range: Range { start: 10, end: 170 }, step: Some((20, 0)) },
        freq_filter: Some(vec![
            // MedianFilter(MedianFilterParamsRangeOrValue {
            //     window: RangeOrValue::dynamic(median_window.clone()),
            // }),
            AverageFilter(AverageFilterParamsRangeOrValue {
                window: RangeOrValue::dynamic(median_window),
                step: RangeOrValue::Value(10),
            }),
            // SavGolFilter(SavGolFilterParamsRangeOrValue {
            //     window_len: RangeOrValue::dynamic(median_window),//RangeOrValue::Range { range: Range { start: 100, end: 200 }, step: None },
            //     poly_order: RangeOrValue::Range { range: Range { start: 1, end: 10 }, step: None },
            //     stride: RangeOrValue::Value(10),
            //     padding: PaddingRangeOrValue::Zero(ZeroPaddingRangeOrValue::default()),
            // }),
        ]),
        finder_kind: FinderKindRangeOrValue::CrossLoader(CrossLoaderRunConfigRangeOrValue {
            // similarity_metric: SimilarityMetricRangeOrValue::EucDist(EucDistConfigRangeOrValue { z_normalize: RangeOrValue::Value(false) }),
            similarity_metric: SimilarityMetricRangeOrValue::DotProduct(DotProductConfigRangeOrValue { with_rfft: RangeOrValue::Value(false) }),
            // slices_filter: SlicesFilterRangeOrValue::Accumulate(AccumulateStatsRangeOrValue {
            //     aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue {
            //         // q: RangeOrValue::Range { range: 0.0..0.5, step: None },
            //         q: RangeOrValue::Value (2.0),
            //     }),
            // }),
            slices_filter: SlicesFilterRangeOrValue::Precalculate(PrecalculateStatsRangeOrValue {
                aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue {
                    q: RangeOrValue::Range { range: 0.0..0.5, step: None },
                    // q: RangeOrValue::Value (0.1),
                }),
                slices: RangeOrValue::Value(1000),
            }),
        }),
        max_slices: Some(MaxSlicesRangeOrValue::Abs(MaxSlicesAbsRangeOrValue { count: RangeOrValue::Value(1000) })),
    }
}


#[allow(dead_code)]
fn run_all_select_motifs_config_grid_0<P: FieldName + 'static>() -> SelectMotifsRunConfigRangeOrValue<P> {
    SelectMotifsRunConfigRangeOrValue {
        id: None,
        // extend_config: None,
        extend_config: Some(ExtendMotifsConfigRangeOrValue{
            slices_selector_metric: SimilarityMetricRangeOrValue::EucDist(EucDistConfigRangeOrValue { z_normalize: RangeOrValue::Value(false) }),
            slices_selector_aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue {
                // q: RangeOrValue::Value(0.05),
                q: RangeOrValue::Range { range: 0.0..0.05, step: None },
                // q: RangeOrValue::Range { range: 0.0..0.05, step: Some((0.05, 2)) },
            }),
        }),
        motif_criteria: vec![
            MotifCriterionConstraintRangeOrValue {
                metric: MetricsRangeOrValue::NormThickness(NormThicknessRunConfigRangeOrValue {
                    eps: RangeOrValue::Value(1e-8),
                }),
                min: RangeOrValue::Value(0.8),
            },
            MotifCriterionConstraintRangeOrValue {
                metric: MetricsRangeOrValue::PeriodicityAcf(PeriodicityAcfRunConfigRangeOrValue {
                    struct_lag_max: RangeOrValue::Value(1. / 64.),
                    lag_min: RangeOrValue::Value(1. / 32.),
                    lag_max: RangeOrValue::Value(0.5),
                    struct0: RangeOrValue::Value(0.2),
                    peak_good: RangeOrValue::Value(0.25),
                    peak_bad: RangeOrValue::Value(0.65),
                    eps: RangeOrValue::Value(1e-8),
                }),
                min: RangeOrValue::Value(0.80),
            },
        ],
        extend_max_slices: Some(MaxSlicesRangeOrValue::Abs(MaxSlicesAbsRangeOrValue { count: RangeOrValue::Value(1000) })),
    }
}

fn run_all_select_motifs_config_opt_0<P: FieldName + 'static>() -> SelectMotifsRunConfigRangeOrValue<P> {
    SelectMotifsRunConfigRangeOrValue {
        id: None,
        extend_config: None,
        // extend_config: Some(ExtendMotifsConfigRangeOrValue{
        //     slices_selector_metric: SimilarityMetricRangeOrValue::EucDist(EucDistConfigRangeOrValue { _path_pd: Default::default() }),
        //     slices_selector_aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue {
        //         // q: RangeOrValue::Value(0.05),
        //         q: RangeOrValue::Range { range: 0.0..0.05, step: None },
        //         // q: RangeOrValue::Range { range: 0.0..0.05, step: Some((0.05, 2)) },
        //     }),
        // }),
        motif_criteria: vec![
            MotifCriterionConstraintRangeOrValue {
                metric: MetricsRangeOrValue::NormThickness(NormThicknessRunConfigRangeOrValue {
                    eps: RangeOrValue::Value(1e-8),
                }),
                min: RangeOrValue::Value(0.9),
            },
            MotifCriterionConstraintRangeOrValue {
                metric: MetricsRangeOrValue::PeriodicityAcf(PeriodicityAcfRunConfigRangeOrValue {
                    struct_lag_max: RangeOrValue::Value(1. / 64.),
                    lag_min: RangeOrValue::Value(1. / 32.),
                    lag_max: RangeOrValue::Value(0.5),
                    struct0: RangeOrValue::Value(0.2),
                    peak_good: RangeOrValue::Value(0.25),
                    peak_bad: RangeOrValue::Value(0.65),
                    eps: RangeOrValue::Value(1e-8),
                }),
                min: RangeOrValue::Value(0.70),
            },
        ],
        extend_max_slices: Some(MaxSlicesRangeOrValue::Abs(MaxSlicesAbsRangeOrValue { count: RangeOrValue::Value(100) })),
    }
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::ops::Range;
    use std::path::Path;
    use std::sync::Arc;
    use log::debug;
    use parking_lot::RwLock;
    use rand::{rng, RngExt};
    use tch::{Device, Tensor};
    use thoosa_common::{debug_plot, train_nets};
    use thoosa_main::tasks::find_motifs_run_config::{AggrRangeOrValue, CoherenceRunConfigRangeOrValue, CrossLoaderRunConfigRangeOrValue, FinderKindRangeOrValue, FindMotifsRunConfigFields, FindMotifsRunConfigRangeOrValue, InlierRatioRunConfigRangeOrValue, MaxSlicesAbsRangeOrValue, MaxSlicesRangeOrValue, MetricsRangeOrValue, MotifCriterionConstraintRangeOrValue, NormThicknessRunConfigRangeOrValue, PrecalculateStatsRangeOrValue, QuantileAggrRangeOrValue, SelectMotifsRunConfigFields, SelectMotifsRunConfigRangeOrValue, SlicesFilterRangeOrValue};
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use thoosa_common::tch_test_utils::{resource_tensor_two_easy_motifs, Interval};
    use thoosa_common::tho_device::ThoDevice;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{AdamWRangeOrValue, CommonTrainingRunConfigRangeOrValue, FinalNetArgsRangeOrValue, LabelNetTypeArgsRangeOrValue, MotifCosineNetArgsRangeOrValue, OptConfig, OptimizersRangeOrValue, SelfAttentionNetArgsRangeOrValue, TrainFinalNetRunConfigRangeOrValue, TrainLabelNetRunConfigRangeOrValue, TrainMotifNetRunConfigRangeOrValue};
    use thoosa_common::transforms::filters::{BandPassFilterParamsRangeOrValue, PaddingRangeOrValue, ZeroPaddingRangeOrValue};
    use thoosa_common::transforms::filters::InputFilterRangeOrValue::BandPassFilter;
    use thoosa_common::ws::FsWorkspace;
    use thoosa_main::cli::{import_resources, RunAllArgs};
    use thoosa_main::RawTensorLoader;
    use thoosa_main::tasks::find_and_select_motifs::{FindAndSelectMotifsRunConfigFields, FindAndSelectMotifsRunConfigRangeOrValue};
    use thoosa_motifs::distance::{EucDistConfigRangeOrValue, SimilarityMetricRangeOrValue};
    use thoosa_motifs::find::FixedFindMotifsConfig;
    use thoosa_motifs::fragments::Resolution;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::{mono_key, panic_to_err_async, RangeOrValue, ThoError};
    use thoosa_shared::hyperparameter::FieldName;
    use thoosa_shared::perf_logging::{format_perf_map, PERF};
    use thoosa_shared::test_utils::MotifKind;
    use thoosa_store::rusqlite::Transaction;
    use thoosa_store::SqlStore;
    use crate::run_all::{run_all, AllConfigs, FindAndSelect};

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

        let imported_ids = import_resources(
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
            None,
            true,
            |t: &Transaction|{
                Ok(())
            },
            // Some(&[(20, true), (10, true)])
        ).await?;

        let resources = store.list_resources_by_ids(&imported_ids).await?;

        let loader = Arc::new(RawTensorLoader::<RunCtxImpl>::new(
            move |key, _| {
                let tensor = Tensor::load(tensors_path.join(format!("{}.pt", key.file_stem())))?;
                debug!("test_motifs: key={key:?} loaded tensor={tensor:?}");
                Ok(ThoTensor::new(SynKey::new([mono_key!(key.id(), format!("resource_id={}", key.id()))]), tensor)?)
            },
            ThoDevice::from(Device::Cpu),
        ));

        let mut ctx = RunCtxImpl::new(1, "test_run_all");
        // ctx.add_permit(global_ctx.acquire_many(0, 1, Arc::new(SynKey::new([mono_key!("test", "test")]))).await?);


        let run_all_args = RunAllArgs{
            find_resources: "group:pure group 10".to_string(),
            select_resources: "group:pure group 20".to_string(),
            max_chunk_size: 100000,
            train_resource_group: "pure group 20".to_string(),
            valid_resource_group: "pure group 10".to_string(),
            train_motifs_epoch_size: 10,
            max_motif_net_loss: 0.4,
            selected_motif_nets_limit: None,
            motif_slices_step: 10,
            with_plots: false,
            resolution: Resolution::Normal,
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
        )}){
            Ok(_) => {
                debug!("ok");
            }
            Err(e) => {
                debug!("e={e:?}");
                panic!("test failed")
            }
        };

        debug!("\n{}", format_perf_map(&PERF));

        global_ctx.shutdown().await;


        Ok(())
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

    fn run_all_config_test_1(run_all_args: &RunAllArgs) -> AllConfigs {
        let find_motifs_run_config_range_or_value = run_all_find_motifs_config_test_1::<FindMotifsRunConfigFields>();
        let fixed_find_motifs_config = FixedFindMotifsConfig::new(run_all_args.max_chunk_size, false);
        let select_motifs_run_config = run_all_select_motifs_config_test_1::<SelectMotifsRunConfigFields>();
        let find_and_select_run_config_range_or_value = FindAndSelectMotifsRunConfigRangeOrValue {
            id: None,
            find_motifs_run_config: run_all_find_motifs_config_test_1::<FindAndSelectMotifsRunConfigFields>(),
            select_motifs_run_config: run_all_select_motifs_config_test_1::<FindAndSelectMotifsRunConfigFields>(),
        };

        let find_and_select_grid =  FindAndSelect::GridOnly {
            find_motifs_run_config_range_or_value,
            fixed_find_motifs_config:fixed_find_motifs_config.clone(),
            select_motifs_run_config,
        };
        let find_and_select_opt =  FindAndSelect::WithOpt {
            find_and_select_run_config_range_or_value,
            fixed_find_motifs_config,
            find_and_select_motifs_opt_config: OptConfig::new(100, 10),
        };

        /////////////////////train_motifs
        let common_training_args_for_motifs = CommonTrainingRunConfigRangeOrValue {
            epochs: RangeOrValue::Value(150),
            batch_size: RangeOrValue::Value(10),
            // lr: RangeOrValue::Value(0.0001),
            lr: RangeOrValue::Range{ range: Range { start: 0.00001, end: 0.001 }, step: None },
            patience: RangeOrValue::Value(10),
            min_delta: RangeOrValue::Value(1e-4_f32),
            optimizer: Some(OptimizersRangeOrValue::AdamW(AdamWRangeOrValue{
                beta1: RangeOrValue::Value(0.9),
                beta2: RangeOrValue::Value(0.999),
                wd: RangeOrValue::Value(0.01),
                amsgrad: RangeOrValue::Value(false),
            })),
        };
        let common_training_args_for_labels = CommonTrainingRunConfigRangeOrValue {
            epochs: RangeOrValue::Value(150),
            batch_size: RangeOrValue::Value(10),
            lr: RangeOrValue::Range{ range: Range { start: 0.00001, end: 0.001 }, step: None },
            patience: RangeOrValue::Value(10),
            min_delta: RangeOrValue::Value(1e-4_f32),
            optimizer: Some(OptimizersRangeOrValue::AdamW(AdamWRangeOrValue{
                beta1: RangeOrValue::Value(0.9),
                beta2: RangeOrValue::Value(0.999),
                wd: RangeOrValue::Value(0.01),
                amsgrad: RangeOrValue::Value(false),
            })),
        };
        let common_training_args_for_final = CommonTrainingRunConfigRangeOrValue {
            epochs: RangeOrValue::Value(150),
            batch_size: RangeOrValue::Value(10),
            lr: RangeOrValue::Range{ range: Range { start: 0.00001, end: 0.001 }, step: None },
            patience: RangeOrValue::Value(10),
            min_delta: RangeOrValue::Value(1e-4_f32),
            optimizer: Some(OptimizersRangeOrValue::AdamW(AdamWRangeOrValue{
                beta1: RangeOrValue::Value(0.9),
                beta2: RangeOrValue::Value(0.999),
                wd: RangeOrValue::Value(0.01),
                amsgrad: RangeOrValue::Value(false),
            })),
        };



        let train_motif_nets_run_config_range_or_value: TrainMotifNetRunConfigRangeOrValue = TrainMotifNetRunConfigRangeOrValue {
            id: None,
            common_training_args: common_training_args_for_motifs,
            // net: train_nets::MotifNetTypeArgsRangeOrValue::ConvNet(train_nets::MotifConvNetArgsRangeOrValue { _dummy: 0 }),
            net: train_nets::MotifNetTypeArgsRangeOrValue::CosineNet(MotifCosineNetArgsRangeOrValue{
                a: RangeOrValue::Range { range: Range { start: 1.0, end: 10.0 }, step: None },
                b: RangeOrValue::Range { range: Range { start: 0.0, end: 1.0 }, step: None },
            }),
            // num_layers: RangeOrValue::Range { range: Range { start: 2, end: 8 }, step: None },

        };
        //////////////////////train labels
        let train_label_nets_run_config_range_or_value: TrainLabelNetRunConfigRangeOrValue = TrainLabelNetRunConfigRangeOrValue {
            id: None,
            common_training_args: common_training_args_for_labels,

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
                head_size: RangeOrValue::Value(2),
                n_heads: RangeOrValue::Value(4),
                ff_mult: RangeOrValue::Value(2),
                dropout: RangeOrValue::Value(0.1),
            })
        };
        //////////////////////train final
        let train_final_net_run_config_range_or_value: TrainFinalNetRunConfigRangeOrValue = TrainFinalNetRunConfigRangeOrValue {
            id: None,
            common_training_args: common_training_args_for_final,
            net: FinalNetArgsRangeOrValue {
                num_layers: RangeOrValue::Range { range: Range { start: 3, end: 5 }, step: None },
            },
        };

        AllConfigs{
            find_and_select: find_and_select_grid,
            train_motif_nets_run_config_range_or_value,
            train_label_nets_run_config_range_or_value,
            train_final_net_run_config_range_or_value,
            train_motifs_opt_config:Arc::new(OptConfig::new(60, 10)),
            train_labels_opt_config: Arc::new(OptConfig::new(60, 10)),
            train_final_opt_config: Arc::new(OptConfig::new(60, 10)),
        }
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

    fn run_all_find_motifs_config_test_1<P: FieldName + 'static>() -> FindMotifsRunConfigRangeOrValue<P> {
        FindMotifsRunConfigRangeOrValue {
            id: None,
            step: RangeOrValue::Value(1),
            window: RangeOrValue::Value(100),
            freq_filter: Some(vec![BandPassFilter(BandPassFilterParamsRangeOrValue {
                low_hz: RangeOrValue::Range { range: Range { start: 0., end: 100. }, step: Some((10., 1)) },
                band_hz: RangeOrValue::Value(2.),
                kernel_size: RangeOrValue::Value(10),
                stride: RangeOrValue::Value(1),
                padding: PaddingRangeOrValue::Zero(ZeroPaddingRangeOrValue::default()),
            })]),
            finder_kind: FinderKindRangeOrValue::CrossLoader(CrossLoaderRunConfigRangeOrValue {
                similarity_metric: SimilarityMetricRangeOrValue::EucDist(EucDistConfigRangeOrValue { z_normalize: RangeOrValue::Value(false) }),
                slices_filter: SlicesFilterRangeOrValue::Precalculate(PrecalculateStatsRangeOrValue {
                    aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue { q: RangeOrValue::Value(0.05) }),
                    slices: RangeOrValue::Value(100),
                }),
            }),
            max_slices: Some(MaxSlicesRangeOrValue::Abs(MaxSlicesAbsRangeOrValue { count: RangeOrValue::Value(100) })),
        }
    }

    fn run_all_select_motifs_config_test_1<P: FieldName + 'static>() -> SelectMotifsRunConfigRangeOrValue<P> {
        SelectMotifsRunConfigRangeOrValue {
            id: None,
            extend_config: None,
            motif_criteria: vec![
                MotifCriterionConstraintRangeOrValue {
                    metric: MetricsRangeOrValue::NormThickness(NormThicknessRunConfigRangeOrValue { eps: RangeOrValue::Value(1e-8) }),
                    min: RangeOrValue::Value(0.5),
                },
                MotifCriterionConstraintRangeOrValue {
                    metric: MetricsRangeOrValue::InlierRatio(InlierRatioRunConfigRangeOrValue {
                        k: RangeOrValue::Value(3.0),
                        eps: RangeOrValue::Value(1e-8),
                    }),
                    min: RangeOrValue::Value(0.5),
                },
                MotifCriterionConstraintRangeOrValue {
                    metric: MetricsRangeOrValue::Coherence(CoherenceRunConfigRangeOrValue {
                        k: RangeOrValue::Value(10.0),
                        eps: RangeOrValue::Value(1e-8),
                    }),
                    min: RangeOrValue::Value(0.5),
                },
            ],
            extend_max_slices: Some(MaxSlicesRangeOrValue::Abs(MaxSlicesAbsRangeOrValue { count: RangeOrValue::Value(100) })),
        }
    }

}
