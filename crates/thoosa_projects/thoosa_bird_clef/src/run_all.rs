use std::ops::Range;
use parking_lot::RwLock;
use std::sync::Arc;
use log::{debug, info};
use thoosa_common::find_motifs_run_config::{AggrRangeOrValue, DotProductConfigRangeOrValue, EucDistConfigRangeOrValue, FindMotifsRunConfigRangeOrValue, MaxSlicesAbsRangeOrValue, MaxSlicesRangeOrValue, MeanAggrRangeOrValue, PrecalculateStatsRangeOrValue, QuantileAggrRangeOrValue, SimilarityMetricRangeOrValue, SlicesFilterRangeOrValue};
use thoosa_common::find_motifs_run_config::InputFilterRangeOrValue::BandPassFilter;
use thoosa_common::runner::common::cachable::DevicesWrap;
use thoosa_common::runner::common::RunCtx;
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_common::runs::Simple;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::transforms::filters::{BandPassFilterParamsRangeOrValue, PaddingRangeOrValue, ZeroPaddingRangeOrValue};
use thoosa_common::ws::Workspace;
use thoosa_main::cli::{run_grid, RunAllArgs};
use thoosa_main::{tasks, RawTensorLoader};
use thoosa_main::tasks::cross_loader::ResourcesConfigInput;
use thoosa_main::tasks::select_motifs::SelectMotifsInput;
use thoosa_motifs::find::FixedFindMotifsConfig;
use thoosa_shared::{collect_futures_with_progress, RangeOrValue, ResourceType, StreamBuffering, ThoError};
use thoosa_shared::ResourceType::NotInGroup;
use thoosa_store::Store;


#[allow(unreachable_code)]
pub(crate) async fn run_all(
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    store: &Arc<Box<dyn Store>>,
    loader: &Arc<RawTensorLoader<RunCtxImpl>>,
    mut root_ctx: RunCtxImpl,
    run_all_args: &RunAllArgs,
    // all_config:AllConfigs
) -> Result<(), ThoError> {
    debug!("run_all");

    let labels = store.list_labels().await?;
    let fixed_find_motifs_config = FixedFindMotifsConfig::new(run_all_args.max_chunk_size, false);
    let mut init_ctx_find_motifs = root_ctx.branch(Some("init find motifs"), true);
    let find_motifs_run_config_rov = Arc::new(FindMotifsRunConfigRangeOrValue {
        id: None,
        step: RangeOrValue::Value(100),
        window: RangeOrValue::Range{ range: Range {start: 1000, end: 16000}, step: Some((1000, 0)) },
        // window: RangeOrValue::Value(1000),
        freq_filter: None,
        // freq_filter: Some(vec![BandPassFilter(BandPassFilterParamsRangeOrValue {
        //     low_hz: RangeOrValue::Range { range: Range { start: 0., end: 100. }, step: Some((10., 1)) },
        //     band_hz: RangeOrValue::Value(2.),
        //     kernel_size: RangeOrValue::Value(10),
        //     stride: RangeOrValue::Value(1),
        //     padding: PaddingRangeOrValue::Zero(ZeroPaddingRangeOrValue::default()),
        // })]),
        // similarity_metric: SimilarityMetricRangeOrValue::EucDist(EucDistConfigRangeOrValue { z_normalize: RangeOrValue::Value(false) }),
        similarity_metric: SimilarityMetricRangeOrValue::DotProduct(DotProductConfigRangeOrValue{ _path_pd: Default::default() }),
        slices_filter: SlicesFilterRangeOrValue::Precalculate(PrecalculateStatsRangeOrValue {
            aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue{ k: RangeOrValue::Value(0.1) }),
            // aggr: AggrRangeOrValue::Quantile(QuantileAggrRangeOrValue { q: RangeOrValue::Value(0.05) }),
            slices: RangeOrValue::Value(100),
        }),
        max_slices: Some(MaxSlicesRangeOrValue::Abs(MaxSlicesAbsRangeOrValue { count: RangeOrValue::Value(100) })),
    });


    let futures = labels.iter().map(|label|{
        let find_motifs_config = fixed_find_motifs_config.clone();
        let mut ctx = init_ctx_find_motifs.branch(Some("find_motifs"), false);
        let mut init_ctx_find_motifs = init_ctx_find_motifs.branch(Some("init_ctx_find_motifs"), false);

        let find_motifs_run_config_rov = Arc::clone(&find_motifs_run_config_rov);
        async move {
            let labels = store.list_labels().await?;

            debug!("run_all: labels={labels:?}");

            for label in labels {

                let find_motifs_input= Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(Simple::new(ResourceType::Set(format!("{label}/3"))))));
                let find_motifs_run_config_rov = Arc::clone(&find_motifs_run_config_rov);
                let ctx = ctx.branch(Some(format!("ctx_for_{label}").as_str()), true);
                let find_motifs_config = find_motifs_config.clone();

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
                                find_motifs_config,
                                ThoDevice::from(1),
                                ctx
                            ).await.expect("FindMotifsStage::new failed"),
                        )
                    },
                    || init_ctx_find_motifs.branch(Some("find_motifs"), true),
                ).await?;
                info!("find_motifs_run_ids: {find_motifs_run_ids:?}");
            }



            Ok(())
        }
    });

    let _ = collect_futures_with_progress(
        futures,
        labels.len(),
        16,
        StreamBuffering::Unordered,
        "bird_clef_labels",
    ).await;


    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use crate::run_all::run_all;
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


        run_all(
            &ws,
            &store,
            &loader,
            ctx,
            &run_all_args,
            // all_configs,
        ).await
    }
}