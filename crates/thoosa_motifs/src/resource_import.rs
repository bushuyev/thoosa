//! Resource import shared by commands and project data preparation.
use log::{debug, error, info, trace};
use parking_lot::RwLock;
use rayon::iter::{ParallelBridge, ParallelIterator};
use std::collections::HashMap;
use std::sync::Arc;
use tch::{Tensor};
use thoosa_common::runner::run_ctx::RunCtxImpl;
use thoosa_common::stages::freq_filter::{SyncFreqFilter};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::collect_futures_eta::{CollectEtaInvocationGuard, collect_eta_record_completion};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::resource::{ResourceId, ResourceStats};
use thoosa_shared::{mono_key};
use thoosa_shared::{cdt};
use thoosa_store::rusqlite::Transaction;
use thoosa_store::{SqlStore};

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
