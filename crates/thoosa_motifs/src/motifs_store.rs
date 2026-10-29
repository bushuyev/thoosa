use anyhow::Context as _;
use async_trait::async_trait;
use futures_util::future::BoxFuture;
use futures_util::{FutureExt, StreamExt};
use parking_lot::{Mutex, RwLock};
use std::array::IntoIter;
use std::backtrace::Backtrace;
use std::cell::RefCell;
use std::cmp::{max, min};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use log::{debug, error, info, trace, warn};
use mockall::predicate::eq;
use mockall::{PredicateBoxExt, mock};
use polars::polars_utils::index::Bounded;
use polars::polars_utils::itertools::Itertools;
use rand::seq::SliceRandom;
use rand::{Rng, RngExt};
use rusqlite::trace::TraceEvent;
use rusqlite::types::Value;
use rusqlite::{Connection, Error, OpenFlags, ToSql, Transaction, params};
use serde::{Deserialize, Serialize};
use thoosa_shared::resource::{Resource, ResourceId, ResourceStats, ResourceStore};
use thoosa_shared::run_configs::{RunConfigId, RunId, run_id};
use thoosa_shared::traits::DataId;
use thoosa_shared::val::Val;
use thoosa_shared::val::Val::UInt;
use thoosa_shared::*;
use thoosa_store::{LimitBy, Store, SetCond, RunConfigEntry, TrainedNetsBy, RunConfigRow, RunConfigFilter, RunMotifEntry, TrainedNetEntry, TrainedNet, SqlStore, conds_to_q};
use crate::motif_tracker_store::{CheckGroups, IdSlice};

#[async_trait]
/// Store boundary used by motif-search workflows while motif methods are migrated incrementally out of `Store`.
pub trait MotifsStore: Store {
    async fn list_label_nets_run_config_ids(
        &self,
        label: &str,
    ) -> anyhow::Result<Vec<RunConfigId>>;

    async fn save_resources_labels(&self, resource_stats_labels_probs: &HashMap<String, (ResourceStats, Vec<(String, f32)>)>, import_id: u32) -> anyhow::Result<Vec<ResourceId>>;
    async fn save_pure_sets(&self, max_len: usize, only_unused: bool) -> anyhow::Result<()>;
    async fn save_leaky_data_sets(&self, prefix: &str, limit: LimitBy, train_coeff: f32, valid_coeff: f32, min_p: f32, allow_leaky_split: bool) -> anyhow::Result<()>;
    async fn save_set_by_labels(&self, conds: Vec<(String, SetCond)>, only_unused: bool, limit: usize) -> anyhow::Result<()>;
    async fn save_groups_by_label_cnt(&self, max_resources_in_set: usize) -> anyhow::Result<()>;
    async fn export_apply_label_nets_records(&self, target_db_path: PathBuf, label_net_run_config_ids: Vec<RunConfigId>) -> anyhow::Result<()>;
    async fn resources_by_label(&self, min_p: f32) -> anyhow::Result<HashMap<String, Vec<Resource>>>;
    async fn list_resource_random_not_labeled(&self, label: &str, limit: u32) -> anyhow::Result<Vec<Resource>>;
    async fn save_label(&self, label: &str) -> anyhow::Result<u32>;
    async fn list_labels(&self, only_with_resources: bool) -> anyhow::Result<Vec<String>>;
    async fn save_params_value<'a>(&self, params: HashMap<String, Val>, run_config_id: RunConfigId, label: Option<&'a str>) -> anyhow::Result<()>;
    async fn save_slices(&self, run_config_id: RunConfigId, channel: i64, slices: Vec<(TensorIdSliceIdx, TensorIdSliceIdx, f32)>) -> anyhow::Result<()>;
    async fn save_groups_as_motifs_with_slices_matches(&self, run_config_id: RunConfigId, channel: i64, groups: HashMap<usize, HashSet<IdSlice>>, path: PathBuf) -> anyhow::Result<()>;
    async fn save_groups_as_motifs(&self, run_config_id: RunConfigId, channel: i64, groups: HashMap<usize, HashSet<IdSlice>>) -> anyhow::Result<Vec<u32>>;
    async fn add_slices_to_motif(&self, run_config_id: RunConfigId, motif_id: u32, channel: i64, slices: HashSet<IdSlice>) -> anyhow::Result<()>;
    async fn list_motifs(&self, run_config_id: RunConfigId) -> anyhow::Result<Vec<u32>>;
    async fn get_motif_meta(&self, motif_id: u32) -> anyhow::Result<(usize, RunConfigId)>;
    async fn list_motifs_slices(&self, run_config_id: RunConfigId, select_motifs_run_config_id: Option<Vec<RunConfigId>>) -> anyhow::Result<HashMap<Resource, Vec<MotifSlice>>>;
    async fn list_motif_slices(&self, motif_id: u32, select_motifs_run_config_id: Option<Vec<RunConfigId>>, limit: Option<u32>, all_but_in_run_config_id: Option<(RunConfigId, u32)>) -> anyhow::Result<(i64, RunConfigId, Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>)>;
    async fn list_random_slices_out_of_motif(&self, motif_id: u32, motif_width: u32, min_distance: Option<u32>, input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync>, desired_count: Option<usize>) -> anyhow::Result<(i64, RunConfigId, Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>)>;
    async fn list_random_slices_out_of_resources(&self, motif_id: u32, motif_width: u32, min_distance: Option<u32>, input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync>, desired_count: Option<usize>) -> anyhow::Result<(i64, RunConfigId, Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>)>;
    async fn list_run_motifs(&self, run_config_id: RunConfigId) -> anyhow::Result<Vec<RunMotifEntry>>;
    async fn save_selected_motifs(&self, run_config_id: RunConfigId, selected_motifs: Vec<(u32, f64)>) -> anyhow::Result<RunConfigId>;
    async fn select_all_motifs(&self, run_config_ids: &[RunConfigId]) -> anyhow::Result<Vec<RunConfigId>>;
    async fn list_selected_motifs(
        &self,
        motif_selection_ids: &[RunConfigId],
        channels: u32,
        limit: Option<u32>,
    ) -> anyhow::Result<HashMap<RunConfigId, Vec<Vec<(u32, f64)>>>>;
}

// These helpers write motif-owned tables, so they live with MotifsStore instead of the common SqlStore API.
fn insert_slices(
    t: &Transaction,
    run_config_id: RunConfigId,
    channel: i64,
    motif_id: i64,
    chunk: IntoIter<IdSlice, 500>,
) -> anyhow::Result<()> {
    let sql = "INSERT INTO slices (run_config_id, channel_id, motif_id, resource_id, slice_idx) VALUES";

    let (mut q, params) = chunk.into_iter().fold(
        (vec![sql], vec![]),
        |mut a: (Vec<&str>, Vec<Value>), v: IdSlice| {
            a.0.push("(?, ?, ?, ?, ?)");
            a.0.push(", ");
            a.1.push((run_config_id as i64).into());
            a.1.push(channel.into());
            a.1.push(motif_id.into());
            a.1.push(v.1.into());
            a.1.push(v.2.into());

            a
        },
    );

    q.pop();

    trace!("insert_slices: params={params:?}");

    match t.execute(
        &q.into_iter().collect::<String>(),
        rusqlite::params_from_iter(params.into_iter()),
    ) {
        Ok(slice_inserted) => {
            debug!("insert_slices: slice_inserted={slice_inserted}");
        }
        Err(e) => {
            debug!("insert_slices: e={e} sqlite_error={:?} run_config_id={run_config_id}, channel={channel}, motif_id={motif_id}",e.sqlite_error());
        }
    };

    Ok(())
}

fn insert_slices_matches_chunk_from_tmp(
    t: &Transaction,
    run_config_id: RunConfigId,
    channel: i64,
    motif_id: i64,
    chunk: IntoIter<IdSlice, 500>,
) -> anyhow::Result<()> {
    insert_slices(t, run_config_id, channel, motif_id, chunk.clone())?;

    // let mut stmt = t.prepare("SELECT * FROM slices")?;
    // let mut res = stmt.query([])?;
    // while let Some(r) =  res.next()?{
    //     debug!("slices: r={r:?}");
    // }

    let sql = format!("INSERT INTO slice_matches (motif_id, slice_id_0, slice_id_1, dist) \
                        SELECT s_0.motif_id, s_0.id, s_1.id, dist FROM slices_{run_config_id}_{channel}_tmp.slice_matches tsm \
                        JOIN slices s_0 on s_0.resource_id = tsm.tensor_id_0 AND s_0.slice_idx = tsm.slice_idx_0 \
                        JOIN slices s_1 on s_1.resource_id = tsm.tensor_id_1 AND s_1.slice_idx = tsm.slice_idx_1 \
                        WHERE s_0.motif_id=? AND s_1.motif_id=?
                        ");

    let slice_matches_updated = t.execute(sql.as_str(), [motif_id, motif_id])?;

    // let mut stmt = t.prepare("SELECT * FROM slice_matches")?;
    // let mut res = stmt.query([])?;
    // while let Some(r) =  res.next()?{
    //     debug!("slice_matches: r={r:?}");
    // }

    debug!("save_slices_groups_as_motifs_from_tmp_db: slice_matches_updated={slice_matches_updated}");

    Ok(())
}

#[async_trait]
impl MotifsStore for SqlStore {


    //TODO remove?
    async fn save_params_value<'a>(
        &self,
        params: HashMap<String, Val>,
        run_config_id: RunConfigId,
        label: Option<&'a str>,
    ) -> anyhow::Result<()> {
        let label = label.map(|v| v.to_string());
        self.with_writer_conn("save_params_value", move |conn| {
            let t = conn.transaction()?;
            let label_id = t.query_row("SELECT id FROM labels WHERE label=?", [label.as_deref()], |row| row.get(0)).ok();

            for property_val in params.into_iter() {
                Self::save_param_value(&t, run_config_id, &property_val.0, property_val.1, label_id)?;
            }

            t.commit()?;
            Ok(())
        })
        .await
    }

    async fn save_label(&self, label: &str) -> anyhow::Result<u32> {
        let label = label.to_string();
        self.with_writer_conn("save_label", move |conn| {
            let t = conn.transaction()?;
            let label_id = Self::save_label(label.as_str(), &t)?;
            t.commit()?;
            Ok(label_id)
        })
        .await
    }

    async fn list_labels(&self, only_with_resources: bool) -> anyhow::Result<Vec<String>> {
        self.with_reader_conn("list_labels", move |conn| {
            let labels = Self::list_labels(conn, only_with_resources).map(|v| v.into_iter().map(|id_label| id_label.1).collect())?;
            Ok(labels)
        })
        .await
    }

    //TODO folds to util fn
    async fn save_resources_labels(
        &self,
        resource_stats_labels_probs: &HashMap<String, (ResourceStats, Vec<(String, f32)>)>,
        import_id: u32,
    ) -> anyhow::Result<Vec<ResourceId>> {
        let resource_stats_labels_probs = resource_stats_labels_probs
            .iter()
            .map(|(resource_key, (stats, labels))| (resource_key.clone(), stats.length(), labels.clone()))
            .collect::<Vec<_>>();
        self.with_writer_conn("save_resources_labels", move |conn| {
            let t = conn.transaction()?;
            let resource_ids = Self::save_resources_labels(resource_stats_labels_probs, import_id, &t)?;
            t.commit()?;
            Ok(resource_ids)
        })
        .await
    }

    async fn save_groups_by_label_cnt(
        &self,
        max_resources_in_set: usize,
    ) -> anyhow::Result<()> {
        self.with_writer_conn("save_groups_by_label_cnt", move |conn| {
            let t = conn.transaction()?;
            Self::save_groups_by_label_cnt(max_resources_in_set, &t)?;
            t.commit()?;
            Ok(())
        })
        .await
    }

    async fn save_pure_sets(
        &self,
        max_len: usize,
        only_unused: bool,
    ) -> anyhow::Result<()> {
        self.with_writer_conn("save_pure_sets", move |conn| {
            let t = conn.transaction()?;
            Self::save_pure_sets(max_len, only_unused, &t)?;
            t.commit()?;
            Ok(())
        })
        .await
    }

    async fn save_leaky_data_sets(
        &self,
        prefix: &str,
        limit: LimitBy,
        train_coeff: f32,
        valid_coeff: f32,
        min_p: f32,
        allow_leaky_split: bool,
    ) -> anyhow::Result<()> {
        let prefix = prefix.to_string();
        self.with_writer_conn("save_data_sets", move |conn| {
            let t = conn.transaction()?;
            Self::save_leaky_data_sets(prefix.as_str(), limit, None, train_coeff, valid_coeff, min_p, &t)?;
            t.commit()?;
            Ok(())
        })
        .await
    }

    async fn list_resource_random_not_labeled(
        &self,
        label: &str,
        limit: u32,
    ) -> anyhow::Result<Vec<Resource>> {
        let label = label.to_string();
        self.with_reader_conn("list_resource_random_not_labeled", move |conn| {
            let r = conn.prepare(
                "SELECT r.id, r.import_id, r.key, r.length FROM resources AS r WHERE NOT EXISTS ( SELECT 1 FROM labeled_resources AS lr JOIN labels AS l ON l.id = lr.label_id WHERE lr.resource_id = r.id AND l.label = ? ) ORDER BY RANDOM() limit ?;"
            )?.query_map([Value::Text(label), Value::Integer(limit as i64)], |row| {
                Ok(Resource::new(
                    row.get::<usize, u32>(0)?,
                    row.get::<usize, u32>(1)?,
                    row.get::<usize, String>(2)?,
                    row.get::<usize, u32>(3)?,
                ))
            })?.collect::<rusqlite::Result<Vec<_>>>()?;

            Ok(r)
        })
        .await
    }

    async fn export_apply_label_nets_records(
        &self,
        target_db_path: PathBuf,
        label_net_run_config_ids: Vec<RunConfigId>,
    ) -> anyhow::Result<()> {
        self.with_writer_conn("export_apply_label_nets_records", move |conn| {
            Self::export_apply_label_nets_records(conn, target_db_path, label_net_run_config_ids)
        }).await
    }

    async fn resources_by_label(
        &self,
        min_p: f32,
    ) -> anyhow::Result<HashMap<String, Vec<Resource>>> {
        Self::validate_min_probability(min_p)?;
        self.with_reader_conn("resources_by_label", move |conn| {
            let labels = Self::list_labels(conn, true)?;
            let mut resources_by_label =
                labels.iter().map(|(_, label)| (label.clone(), Vec::new())).collect::<HashMap<String, Vec<Resource>>>();
            let mut stmt = conn.prepare(
                "SELECT l.label, r.id, r.import_id, r.key, r.length FROM labeled_resources lr \
                JOIN labels l ON l.id = lr.label_id \
                JOIN resources r ON r.id = lr.resource_id \
                WHERE lr.p >= ? \
                ORDER BY l.id, lr.resource_id",
            )?;
            let mut rows = stmt.query(params![min_p])?;

            while let Some(row) = rows.next()? {
                let label = row.get::<usize, String>(0)?;
                let resource = Resource::new(
                    row.get::<usize, u32>(1)?,
                    row.get::<usize, u32>(2)?,
                    row.get::<usize, String>(3)?,
                    row.get::<usize, u32>(4)?,
                );
                resources_by_label.entry(label).or_insert_with(Vec::new).push(resource);
            }

            Ok(resources_by_label)
        })
        .await
    }

    //TODO not used, remove?
    async fn save_set_by_labels(
        &self,
        conds: Vec<(String, SetCond)>,
        _only_unused: bool,
        limit: usize,
    ) -> anyhow::Result<()> {
        self.with_writer_conn("save_set_by_labels", move |conn| {
        let t = conn.transaction()?;

        {
            t.execute(
                "INSERT INTO resource_set (descr) VALUES (?)",
                params!["save_set_by_labels"],
            )?;

            let resource_set_id = t.query_row("SELECT last_insert_rowid()", [], |row| {
                row.get::<usize, i64>(0)
            })?;

            let q = [
                "INSERT INTO resource_resources_set (set_id, resource_id) \
                SELECT ?, id FROM (\
                SELECT r.id FROM resources r LEFT JOIN labeled_resources lr ON lr.resource_id=r.id LEFT JOIN labels l on l.id=lr.label_id where ",
                //(l.label='aaa' and p>0.9) or...
                &conds_to_q(&conds),
                " LIMIT ?\
                )"
            ].concat();
            debug!("q={:?}", q);
            let mut resource_resources_set_stmt = t.prepare(&q)?;

            let mut params = vec![Value::Integer(resource_set_id)];
            for cond in &conds {
                params.push(Value::Real(cond.1.v() as f64));
            }
            params.push(Value::Integer(limit as i64));

            if resource_resources_set_stmt
                .execute(rusqlite::params_from_iter(params.into_iter()))?
                == 0
            {
                // Empty sets are allowed; report the condition without discarding an error.
                warn!("no resources for set on {:?}", conds);
            }
        }
        Ok(t.commit()?)
        })
        .await
    }
    //TODO remove
    async fn save_slices(
        &self,
        run_config_id: RunConfigId,
        channel: i64,
        slices: Vec<(TensorIdSliceIdx, TensorIdSliceIdx, f32)>,
    ) -> anyhow::Result<()> {
        self.with_writer_conn("save_slices", move |conn| {
        let t = conn.transaction()?;

        assert!(slices.len() > 0);

        for slices_chunk in slices.chunks(500) {
            let (mut q, params) = slices_chunk.into_iter().fold(
                (vec!["INSERT INTO slice_matches (run_config_id, channel_id, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1, dist) VALUES"], vec![]), |mut a: (Vec<&str>, Vec<Value>), v| {
                    a.0.push("(?, ?, ?, ?, ?, ?, ?)");
                    a.0.push(", ");

                    a.1.push(Value::Integer(run_config_id as i64));
                    a.1.push(Value::Integer(channel));
                    a.1.push(Value::Integer(v.0.0 as i64));
                    a.1.push(Value::Integer(v.0.1 as i64));
                    a.1.push(Value::Integer(v.1.0 as i64));
                    a.1.push(Value::Integer(v.1.1 as i64));
                    a.1.push(Value::Real(v.2 as f64));
                    a
                });

            q.pop();

            t.execute(
                &q.into_iter().collect::<String>(),
                rusqlite::params_from_iter(params.into_iter()),
            )?;
        }
        t.commit()?;

        Ok(())
        })
        .await
    }

    async fn save_groups_as_motifs(
        &self,
        run_config_id: RunConfigId,
        channel: i64,
        groups: HashMap<usize, HashSet<IdSlice>>,
    ) -> anyhow::Result<Vec<u32>> {
        self.with_writer_conn("save_groups_as_motifs", move |conn| {
        let t = conn.transaction()?;
        let mut motif_ids = Vec::new();

        debug!("save_groups_as_motifs: run_config_id={run_config_id:?}, channel={channel:?} groups={}", groups.len());

        {
            for (group, slices) in groups {
                if slices.is_empty() {
                    continue
                }
                let motifs_inserted = match t.execute(
                        "INSERT INTO motifs (group_ord, run_config_id, channel_id) VALUES (?, ?, ?)",
                        [group as usize, run_config_id as usize, channel as usize],
                    ) {
                    Ok(inserted) => inserted,
                    Err(error) => return Err(Self::log_error((group, run_config_id, channel), &t, error).into()),
                };

                let motif_id: i64 = t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;

                let motif_id_u32 = u32::try_from(motif_id).with_context(|| format!("save_groups_as_motifs: motif_id={motif_id} does not fit into u32"))?;
                motif_ids.push(motif_id_u32);

                debug!("save_groups_as_motifs: group={group:?}, run_config_id={run_config_id:?} motif_id={motif_id} motifs_inserted={motifs_inserted} slices.len={}", slices.len());
                trace!("save_groups_as_motifs: slices={slices:?} ");

                let mut slices_chunks = slices.into_iter().array_chunks::<500>();
                while let Some(chunk) = slices_chunks.next() {
                    insert_slices(&t, run_config_id, channel, motif_id, chunk.into_iter())?
                }
                let remainder = slices_chunks.into_remainder();
                if remainder.len() != 0 {
                // if !remainder.is_empty() {
                    insert_slices(&t, run_config_id, channel, motif_id, remainder)?
                }
            }
        }

        t.commit()?;

        Ok(motif_ids)
        })
        .await
    }

    async fn add_slices_to_motif(
        &self,
        run_config_id: RunConfigId,
        motif_id: u32,
        channel: i64,
        slices: HashSet<IdSlice>,
    ) -> anyhow::Result<()> {
        self.with_writer_conn("add_slices_to_motif", move |conn| {
        let t = conn.transaction()?;

        debug!("save_slices_groups_as_motifs_from_file: run_config_id={run_config_id:?}, channel={channel:?} slices.len={}", slices.len());

        {
            let mut slices_chunks = slices.into_iter().array_chunks::<500>();
            while let Some(chunk) = slices_chunks.next() {
                insert_slices(&t, run_config_id, channel, motif_id as i64, chunk.into_iter())?
            }
            let remainder = slices_chunks.into_remainder();
            if remainder.len() != 0 {
            // if !remainder.is_empty() {
                insert_slices(&t, run_config_id, channel, motif_id as i64, remainder)?
            }
        }
        t.commit()?;
        Ok(())
        })
        .await
    }

    async fn save_groups_as_motifs_with_slices_matches(
        &self,
        run_config_id: RunConfigId,
        channel: i64,
        groups: HashMap<usize, HashSet<IdSlice>>,
        path: PathBuf,
    ) -> anyhow::Result<()> {
        self.with_writer_conn("save_groups_as_motifs_with_slices_matches", move |conn| {
        let t = conn.transaction()?;

        debug!("save_slices_groups_as_motifs_from_tmp_db: run_config_id={run_config_id:?}, channel={channel:?} path={}", path.to_string_lossy());

        {
            t.execute(
                format!(
                    "ATTACH DATABASE '{}' AS slices_{run_config_id}_{channel}_tmp",
                    path.to_string_lossy()
                )
                .as_str(),
                [],
            )?;

            //TODO split into insert motifs/insert slice_matches + move  group_slices from tracker store?
            for (group, slices) in groups {
                if slices.is_empty() {
                    continue
                }
                let motifs_inserted = t.execute(
                    "INSERT INTO motifs (group_ord, run_config_id, channel_id) VALUES (?, ?, ?)",
                    [group as usize, run_config_id as usize, channel as usize],
                )?; //TODO group not needed?

                let motif_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;

                debug!("save_slices_groups_as_motifs_from_tmp_db: group={group:?}, run_config_id={run_config_id:?} motif_id={motif_id} motifs_inserted={motifs_inserted} slices={slices:?} ");

                //each match appears twice in slices but will result in single raw returned from _tmp.slice_matches, should be faster than getting slices into set on slice_matches.id (IdSlice.0)
                let mut slices_chunks = slices.into_iter().array_chunks::<500>();

                while let Some(chunk) = slices_chunks.next() {
                    insert_slices_matches_chunk_from_tmp(
                        &t,
                        run_config_id,
                        channel,
                        motif_id,
                        chunk.into_iter(),
                    )?
                }

                let remainder = slices_chunks.into_remainder();
                if remainder.len() != 0 {
                // if !remainder.is_empty() {
                    insert_slices_matches_chunk_from_tmp(
                        &t, run_config_id, channel, motif_id, remainder,
                    )?
                }
            }
        }

        t.commit()?;

        let t = conn.transaction()?;
        t.execute(
            format!("DETACH DATABASE slices_{run_config_id}_{channel}_tmp").as_str(),
            [],
        )?;
        t.commit()?;

        Ok(())
        })
        .await
    }

    async fn get_motif_meta(&self, motif_id: u32) -> anyhow::Result<(usize, RunConfigId)> {
        self.with_reader_conn("get_motif_meta", move |conn| {
                let (channel, run_config_id) = Self::_motif_meta(motif_id, conn)?;
                debug!("get_motif_meta: motif_id={motif_id} channel={channel} run_config_id={run_config_id}");
                Ok::<_, anyhow::Error>((channel as usize, run_config_id))
            })
            .await
    }

    async fn list_motifs(&self, run_config_id: RunConfigId) -> anyhow::Result<Vec<u32>> {
        self.with_reader_conn("list_motifs", move |conn| {
            let motif_ids = {
                let mut stmt = conn.prepare("SELECT id FROM motifs WHERE run_config_id=?")?;
                stmt.query_map([run_config_id], |row| Ok(row.get::<usize, u32>(0)?))?
                    .collect::<Result<Vec<_>, Error>>()?
            };
            Ok(motif_ids)
        })
        .await
    }

    /**
     * returns map tensor_key - MotifSlice(motif_group, tensor slice idx, channel)
     */
    async fn list_motifs_slices(
        &self,
        run_config_id: RunConfigId,
        select_motifs_run_config_id: Option<Vec<RunConfigId>>,
    ) -> anyhow::Result<HashMap<Resource, Vec<MotifSlice>>> {
        self.with_reader_conn("list_motifs_slices", move |conn| {
            let slice_run_config_ids = select_motifs_run_config_id.map(|mut ids| {
                ids.push(run_config_id);
                ids.sort();
                ids.dedup();
                ids
            });

            let slice_filter_clause = slice_run_config_ids
                .as_ref()
                .map(|ids| format!(" AND s.run_config_id IN ({})", ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ")))
                .unwrap_or_default();

            let resources_motifs = {
                let mut stmt = conn.prepare(
                    &format!(
                        "SELECT m.id, m.channel_id, r.id, r.import_id, r.key, s.slice_idx, r.length \
                         FROM motifs m \
                         JOIN slices s ON s.motif_id=m.id \
                         JOIN resources r on r.id=s.resource_id \
                         WHERE m.run_config_id=?{}",
                        slice_filter_clause
                    ),
                )?;

                let mut params: Vec<Value> = vec![Value::Integer(run_config_id as i64)];
                if let Some(ids) = &slice_run_config_ids {
                    params.extend(ids.iter().map(|slice_run_config_id| Value::Integer(*slice_run_config_id as i64)));
                }

                let tensor_motifs: (HashMap<ResourceId, Vec<MotifSlice>>, HashMap<ResourceId, Resource>) = stmt
                    .query_map(rusqlite::params_from_iter(params.iter()), |row| {
                        Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?, row.get(5)?, row.get(6)?))
                    })?
                    .fold(
                        Ok((HashMap::new(), HashMap::new())),
                        |a: Result<(HashMap<ResourceId, Vec<MotifSlice>>, HashMap<ResourceId, Resource>), Error>,
                         v: Result<(u32, i64, u32, u32, String, i64, u32), Error>| {
                            let v = v?;
                            let mut a = a?;

                            a.0.entry(v.2).or_insert_with(|| vec![]).push(MotifSlice::new(v.0, v.5, v.1 as u16));

                            if !a.1.contains_key(&v.2) {
                                a.1.insert(v.2, Resource::new(v.2, v.3, v.4, v.6));
                            }

                            Ok(a)
                        },
                    )?;

                trace!("motifs_to_eeg_slices: eeg_motifs={:?}", tensor_motifs);

                tensor_motifs
                    .0
                    .into_iter()
                    .fold((HashMap::new(), tensor_motifs.1), |mut a, (resource_id, slices)| {
                        a.0.insert(a.1.remove(&resource_id).expect(format!("no resource for {}", resource_id).as_str()), slices);
                        a
                    })
                    .0
            };

            Ok(resources_motifs)
        })
        .await
    }


    ///returns channel, find_motifs_run_config_id, slices, resources map
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="list_motif_slices"))]
    async fn list_motif_slices(
        &self,
        motif_id: u32,
        select_motifs_run_config_ids: Option<Vec<RunConfigId>>, //None - all slices, empty - only slices from find_motifs, Some -  find_motifs + select_motif runs
        limit: Option<u32>,
        all_but_in_run_config_id: Option<(RunConfigId, u32/*channel*/)>, //TODO remove - load random slices from same channel/resources
    ) -> anyhow::Result<
        (
            i64,
            RunConfigId,
            Vec<TensorIdSliceIdx>,
            HashMap<ResourceId, Resource>,
        )> {
        debug!("list_motif_slices: motif_id={motif_id} select_motifs_run_config_id={select_motifs_run_config_ids:?} limit={limit:?} all_but_in_run_config_id={all_but_in_run_config_id:?}");
        self.with_reader_conn("list_motif_slices", move |conn| {

            // TODO(perf): this query is called for every MotifLoader run; add a prepared/cacheable path for the common motif_id + select_run filter and avoid ORDER BY RANDOM for large tables.
            let (slice_filter_clause, slice_filter_params) = if all_but_in_run_config_id.is_none() {
                match select_motifs_run_config_ids.as_ref() {
                    None => (String::new(), Vec::new()),
                    Some(ids_raw) => {
                        let mut ids = ids_raw.clone();
                        ids.sort();
                        ids.dedup();
                        if ids.is_empty() {
                            (" AND s.run_config_id IN (m.run_config_id)".to_string(), Vec::new())
                        } else {
                            let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
                            (format!(" AND s.run_config_id IN (m.run_config_id, {placeholders})"), ids)
                        }
                    }
                }
            } else {
                (String::new(), Vec::new())
            };

            let channel_slices = {
                let mut query = String::from(
                    "SELECT m.channel_id, m.run_config_id, s.run_config_id, r.id,  r.import_id, r.key, s.slice_idx \
                        FROM motifs m \
                        LEFT JOIN slices s on s.motif_id=m.id \
                        LEFT JOIN resources r on s.resource_id=r.id \
                        WHERE ",
                );

                let params: Vec<u32> = match all_but_in_run_config_id {
                    Some((_run_config_id, _channel_id)) => {
                        query.push_str("s.motif_id!=? AND m.channel_id=(SELECT channel_id FROM motifs WHERE id=?) ORDER BY RANDOM()");
                        vec![motif_id, motif_id]
                    }
                    None => {
                        query.push_str("s.motif_id=?");
                        query.push_str(&slice_filter_clause);
                        let mut values = vec![motif_id];
                        values.extend(slice_filter_params.iter().copied());
                        values
                    }
                };
                if let Some(limit) = limit {
                    query.push_str(&format!(" LIMIT {}", limit));
                }

                let mut stmt = conn.prepare(&query)?;
                debug!("list_motif_slices: params={params:?}");

                let (slices, channel_set, run_config_id, resources_map) = stmt
                    .query_map(rusqlite::params_from_iter(params.iter()), |row| {
                        Ok((
                            row.get::<usize, i64>(0)?,
                            row.get::<usize, RunConfigId>(1)?,
                            row.get::<usize, RunConfigId>(2)?,
                            row.get(3)?,
                            row.get(4)?,
                            row.get(5)?,
                            row.get(6)?,
                        ))
                    })?
                    .fold(
                        Ok((HashSet::new(), HashSet::new(), None, HashMap::new())),
                        |a: anyhow::Result<(HashSet<TensorIdSliceIdx>, HashSet<i64>, Option<RunConfigId>, HashMap<ResourceId, Resource>)>,
                         v: Result<(i64, RunConfigId, RunConfigId, u32, u32, String, u32), Error>| {
                            let (channel_id, motif_run_config_id, _slice_run_config_id, resource_id, import_id, key, slice_idx) = v?;
                            let (mut a, mut channel_set, mut run_config_id, mut resources_map) = a?;

                            if run_config_id.is_none() {
                                run_config_id.replace(motif_run_config_id);
                            } else {
                                assert_eq!(run_config_id, Some(motif_run_config_id));
                            }

                            a.insert(TensorIdSliceIdx(resource_id, slice_idx));
                            resources_map.insert(resource_id, Resource::new(resource_id, import_id, key, 0));
                            channel_set.insert(channel_id);

                            Ok((a, channel_set, run_config_id, resources_map))
                        },
                    )?;

                if all_but_in_run_config_id.is_none() {
                    assert_eq!(channel_set.len(), 1, "channel_set={channel_set:?} motif_id={motif_id}");
                    (channel_set.into_iter().next().expect("channel"), run_config_id.expect("no run id"), slices.into_iter().collect(), resources_map)
                } else {
                    let (run_config_id, channel_id) = all_but_in_run_config_id.expect("all_but_in_run_config_id");
                    (channel_id as i64, run_config_id, slices.into_iter().collect(), resources_map)
                }
            };

            Ok(channel_slices)
        })
        .await
    }

    ///gpt generated, returns slices from same resources and same channel as target motif but not belonging to the motif
    ///if not enough slices in same resources - opts to random resources
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SqlStore::list_random_slices_out_of_motif"))]
    async fn list_random_slices_out_of_motif(
        &self,
        motif_id: u32,
        motif_width: u32,
        min_distance: Option<u32>,
        input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync>,
        desired_count: Option<usize>,
    ) -> anyhow::Result<
        (
            i64,
            RunConfigId,
            Vec<TensorIdSliceIdx>,
            HashMap<ResourceId, Resource>,
        )> {
        debug!("list_random_slices_out_of_motif: motif_id={motif_id} motif_width={motif_width} min_distance={min_distance:?}");

        assert!(motif_width > 0, "motif_width must be greater than zero");
        let input_len_adjuster_in_motif = Arc::clone(&input_len_adjuster);
        // TODO(perf): candidate construction scans motif/resource context on every training batch; persist reusable candidate pools per motif/run/window/min_distance and sample from them cheaply.
        let (channel, run_config_id, mut all_candidates, mut resources_map, desired_count) = self.with_reader_conn("list_random_motif_slices", move |conn| {
            let (channel, run_config_id) = Self::_motif_meta(motif_id, conn)?;
            let (target_positions_by_res, num_motif_slices) = Self::load_motif_target_positions(motif_id, conn)?;
            let desired_count = desired_count.unwrap_or(num_motif_slices);
            if desired_count == 0 {
                return Ok((channel, run_config_id, vec![], HashMap::new(), desired_count));
            }

            let contexts =
                Self::load_motif_resource_contexts(motif_id, input_len_adjuster_in_motif.as_ref(), conn, target_positions_by_res)?;

            if contexts.is_empty() {
                return Err(anyhow::anyhow!("no resources found for channel {channel}, run_config_id={run_config_id}"));
            }

            let (all_candidates, resources_map) = Self::build_random_slice_candidates(&contexts, motif_width, min_distance);
            Ok((channel, run_config_id, all_candidates, resources_map, desired_count))
        }).await?;

        if all_candidates.len() < desired_count {
            let missing_count = desired_count - all_candidates.len();
            debug!("list_random_slices_out_of_motif: motif_id={motif_id} fell back to other resources for {missing_count} slices");

            let (fallback_channel, fallback_run_config_id, mut fallback_candidates, fallback_resources_map) = self
                .list_random_slices_out_of_resources(
                    motif_id,
                    motif_width,
                    min_distance,
                    input_len_adjuster,
                    Some(missing_count),
                )
                .await?;

            assert_eq!(channel, fallback_channel, "fallback channel differs for motif_id={motif_id}");
            assert_eq!(run_config_id, fallback_run_config_id, "fallback run_config_id differs for motif_id={motif_id}");

            all_candidates.append(&mut fallback_candidates);
            resources_map.extend(fallback_resources_map);
        }

        Self::shuffle_and_limit_candidates(&mut all_candidates, desired_count);
        Ok((channel, run_config_id, all_candidates, resources_map))
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SqlStore::list_random_slices_out_of_resources"))]
    async fn list_random_slices_out_of_resources(
        &self,
        motif_id: u32,
        motif_width: u32,
        min_distance: Option<u32>,
        input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync>,
        desired_count: Option<usize>,
    ) -> anyhow::Result<
        (
            i64,
            RunConfigId,
            Vec<TensorIdSliceIdx>,
            HashMap<ResourceId, Resource>,
        )> {
        debug!("list_random_slices_out_of_resources: motif_id={motif_id} motif_width={motif_width} min_distance={min_distance:?}");

        assert!(motif_width > 0, "motif_width must be greater than zero");

        self.with_reader_conn("list_random_slices_out_of_resources", move |conn| {
            let (channel, run_config_id) = Self::_motif_meta(motif_id, conn)?;
            // TODO(perf): fallback random-slice selection can be prefiltered by run/channel with indexed temp tables instead of rebuilding all non-motif contexts.
            let desired_count = match desired_count {
                Some(desired_count) => desired_count,
                None => Self::count_motif_slices(motif_id, conn)?,
            };

            if desired_count == 0 {
                return Ok((channel, run_config_id, vec![], HashMap::new()));
            }

            let contexts = Self::load_non_motif_resource_contexts(motif_id, run_config_id, channel, input_len_adjuster.as_ref(), conn)?;
            if contexts.is_empty() {
                return Ok((channel, run_config_id, vec![], HashMap::new()));
            }

            let (mut all_candidates, resources_map) = Self::build_random_slice_candidates(&contexts, motif_width, min_distance);
            Self::shuffle_and_limit_candidates(&mut all_candidates, desired_count);

            Ok((channel, run_config_id, all_candidates, resources_map))
        }).await
    }

    async fn list_run_motifs(&self, run_config_id: RunConfigId) -> anyhow::Result<Vec<RunMotifEntry>> {
        self.with_reader_conn("list_run_motifs", move |conn| {
            let selected_motifs = {
                let mut stmt = conn.prepare(
                    "SELECT sm.motif_id, m.channel_id, sm.rate, m.run_config_id,                       (SELECT COUNT(*) FROM slices s WHERE s.motif_id = sm.motif_id) AS slices_cnt,                       (SELECT COUNT(DISTINCT s.resource_id) FROM slices s WHERE s.motif_id = sm.motif_id) AS resources_cnt                      FROM selected_motifs sm                      JOIN motifs m ON m.id = sm.motif_id                      WHERE sm.run_config_id = ?                      ORDER BY m.channel_id ASC, sm.rate DESC, sm.motif_id ASC",
                )?;

                stmt.query_map([run_config_id], |row| {
                    Ok(RunMotifEntry {
                        motif_id: row.get::<usize, u32>(0)?,
                        channel_id: row.get::<usize, usize>(1)?,
                        score: row.get::<usize, f64>(2).ok(),
                        source_run_config_id: row.get::<usize, RunConfigId>(3)?,
                        slices_cnt: row.get::<usize, usize>(4)?,
                        resources_cnt: row.get::<usize, usize>(5)?,
                    })
                })?
                .collect::<Result<Vec<_>, Error>>()?
            };

            if !selected_motifs.is_empty() {
                debug!("list_run_motifs: run_config_id={run_config_id} selected_motifs_count={}", selected_motifs.len());
                return Ok(selected_motifs);
            }

            let motifs = {
                let mut stmt = conn.prepare(
                    "SELECT m.id, m.channel_id, m.run_config_id                      FROM motifs m                      WHERE m.run_config_id = ?                      ORDER BY m.channel_id ASC, m.id ASC",
                )?;

                stmt.query_map([run_config_id], |row| {
                    Ok(RunMotifEntry {
                        motif_id: row.get::<usize, u32>(0)?,
                        channel_id: row.get::<usize, usize>(1)?,
                        score: None,
                        source_run_config_id: row.get::<usize, RunConfigId>(2)?,
                        slices_cnt: 0,
                        resources_cnt: 0,
                    })
                })?
                .collect::<Result<Vec<_>, Error>>()?
            };

            debug!("list_run_motifs: run_config_id={run_config_id} motifs_count={}", motifs.len());
            Ok(motifs)
        })
        .await
    }

    async fn save_selected_motifs(
        &self,
        run_config_id: RunConfigId,
        selected_motifs: Vec<(u32, f64)>,
    ) -> anyhow::Result<RunConfigId> {
        debug!("save_selected_motifs: selected_motifs={:?}",selected_motifs);
        self.with_writer_conn("save_selected_motifs", move |conn| {
            let t = conn.transaction()?;

            for selected_motifs_chunk in selected_motifs.chunks(500) {
                let (mut q, params) = selected_motifs_chunk.into_iter().fold(
                    (vec!["INSERT INTO selected_motifs (run_config_id, motif_id, rate) VALUES "], vec![]),
                    |mut a: (Vec<&str>, Vec<Value>), (motif_id, rate)| {
                        a.0.push("(?, ?, ?)");
                        a.0.push(", ");
                        a.1.push(Value::Integer(run_config_id as i64));
                        a.1.push(Value::Integer(*motif_id as i64));
                        a.1.push(Value::Real(*rate));
                        a
                    },
                );

                q.pop();
                t.execute(&q.into_iter().collect::<String>(), rusqlite::params_from_iter(params.into_iter()))?;
            }

            t.commit()?;

            Ok(run_config_id)
        })
        .await
    }

    async fn select_all_motifs(&self, run_config_ids: &[RunConfigId]) -> anyhow::Result<Vec<RunConfigId>> {
        let run_config_ids = run_config_ids.to_vec();
        debug!("select_all_motifs: run_config_ids={run_config_ids:?}");
        self.with_writer_conn("select_all_motifs", move |conn| {
            let t = conn.transaction()?;
            let mut selected_run_config_ids = Vec::with_capacity(run_config_ids.len());


            for input_run_config_id in run_config_ids {
                t.execute("INSERT INTO run_configs (type, descr, run_id) VALUES (?, ?, ?)", params!["SelectMotifsRunConfig", "bypass", run_id()])?;

                let selected_run_config_id = t.query_row("SELECT last_insert_rowid()", [], |row| {
                    row.get::<usize, RunConfigId>(0)
                })?;

                t.execute("INSERT INTO run_configs_input (run_config_id, input_run_config_id) VALUES (?, ?)", params![selected_run_config_id, input_run_config_id])?;
                t.execute("INSERT INTO selected_motifs (run_config_id, motif_id, rate) SELECT ?, id, 1.0 FROM motifs WHERE run_config_id=?", params![selected_run_config_id, input_run_config_id])?;
                selected_run_config_ids.push(selected_run_config_id);
            }

            t.commit()?;

            Ok(selected_run_config_ids)
        })
        .await
    }

    /// Returns, for each run config, motifs grouped by channel as `(motif_id, rate)`.
    async fn list_selected_motifs(
        &self,
        motif_selection_ids: &[RunConfigId],
        channels: u32,
        limit: Option<u32>,
    ) -> anyhow::Result<HashMap<RunConfigId, Vec<Vec<(u32, f64)>>>> {
        assert!(!motif_selection_ids.is_empty(), "motif_selection_ids is empty");

        let placeholders = vec!["?"; motif_selection_ids.len()].join(", ");

        let mut sql = format!(
            "\
        SELECT DISTINCT
            sm.run_config_id,
            sm.motif_id,
            sm.rate,
            m.channel_id
        FROM selected_motifs AS sm
        LEFT JOIN motifs AS m ON m.id = sm.motif_id
        WHERE sm.run_config_id IN ({placeholders})
        ORDER BY sm.rate"
        );

        let mut params = motif_selection_ids
            .iter()
            .map(|&id| Value::Integer(id as i64))
            .collect::<Vec<_>>();

        if let Some(limit) = limit {
            sql.push_str(" DESC LIMIT ?");
            params.push(Value::Integer(limit as i64));
        }

        self.with_reader_conn("list_selected_motifs", move |conn| {
            debug!("list_selected_motifs: stmt={sql} channels={channels} params={params:?} limit={limit:?}");

            let mut stmt = conn.prepare(&sql)?;

            let rows = stmt.query_map(
                rusqlite::params_from_iter(&params),
                |row| {
                    Ok((
                        row.get::<_, RunConfigId>(0)?,
                        row.get::<_, u32>(1)?,
                        row.get::<_, f64>(2)?,
                        row.get::<_, usize>(3)?,
                    ))
                },
            )?;

            let mut result:HashMap<RunConfigId, Vec<Vec<(u32, f64)>>> = HashMap::new();

            for row in rows {
                let (run_config_id, motif_id, rate, channel_id) = row?;

                let channel_motifs = result.entry(run_config_id).or_insert_with(|| {
                    std::iter::repeat_with(|| Vec::with_capacity(10))
                        .take(channels as usize)
                        .collect()
                });

                channel_motifs[channel_id].push((motif_id, rate));
            }

            debug!("list_selected_motifs: result={result:?}");

            Ok(result)
        })
            .await
    }    async fn list_label_nets_run_config_ids(
        &self,
        label: &str,
    ) -> anyhow::Result<Vec<RunConfigId>> {
        let label = label.to_string();

        self.with_reader_conn("list_resource_random_not_labeled", move |conn| {
            let r = conn.prepare(
                "SELECT tn.run_config_id FROM trained_nets tn \
                LEFT JOIN run_configs rc ON tn.run_config_id=rc.id \
                WHERE \
                    rc.run_id=? \
                    AND rc.type='TrainLabelNetRunConfig' \
                    AND net_path LIKE ? \
                 ORDER BY date(run_on) DESC, tn.loss LIMIT 1"//TODO takes last one
            )?.query_map(params![
                run_id(),
                format!("{label}%")
            ], |row| {
                Ok(row.get::<usize, u32>(0)?)
            })?.collect::<rusqlite::Result<Vec<_>>>()?;

            Ok(r)
        })
            .await
    }

}

pub fn create_motif_tables( conn: &mut Connection) -> anyhow::Result<()> {
    let t = conn.transaction()?;

    t.execute(
        "CREATE TABLE IF NOT EXISTS labels (
                id          INTEGER PRIMARY KEY,
                label       VARCHAR(256),
                UNIQUE(label)
            )",
        (),
    )?;

    t.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_labels_label ON labels (label);",
        (),
    )?;

    t.execute(
        "CREATE TABLE IF NOT EXISTS labeled_resources (
                resource_id    NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
                label_id       NOT NULL REFERENCES labels(id) ON DELETE CASCADE,
                p              FLOAT NOT NULL,
                PRIMARY KEY (resource_id, label_id)
            )",
        (),
    )?;

    t.execute("CREATE INDEX IF NOT EXISTS idx_labeled_resources_label_id ON labeled_resources(label_id)", ())?;

    t.execute("commit", ())?;//TODO commit together with common tables

    t.execute(
        //group - ordinal as appeared when grouping, not needed
        "CREATE TABLE IF NOT EXISTS motifs (
                id              INTEGER PRIMARY KEY,
                group_ord       INTEGER NOT NULL,
                run_config_id          NOT NULL REFERENCES run_configs(id) ON DELETE CASCADE,
                channel_id      INTEGER NOT NULL
            )",
        (),
    )?;
    t.execute("CREATE INDEX IF NOT EXISTS idx_motifs_run_config_id ON motifs (run_config_id);", ())?;

    //motif_id in UNIQUE constraint is needed when new slices are added from different find_motifs in same select_motifs run.
    t.execute(
        "CREATE TABLE IF NOT EXISTS slices (
                id              INTEGER PRIMARY KEY,
                motif_id        NOT NULL REFERENCES motifs(id) ON DELETE CASCADE,
                run_config_id          NOT NULL REFERENCES run_configs(id) ON DELETE CASCADE,
                channel_id      INTEGER NOT NULL,
                resource_id     REFERENCES resources(id) ON DELETE CASCADE,
                slice_idx       INTEGER NOT NULL,
                UNIQUE(motif_id, run_config_id, resource_id, slice_idx)
            )",
        (),
    )?;
    // The UNIQUE index on (motif_id, run_config_id, resource_id, slice_idx) already covers the
    // motif_id-led access pattern used by list_run_motifs. Keep inserts fast by ensuring
    // we do not maintain extra slices indexes.
    t.execute("DROP INDEX IF EXISTS idx_slices_motif_id;", ())?;
    t.execute("DROP INDEX IF EXISTS idx_slices_motif_id_resource_id;", ())?;

    t.execute(
        "CREATE TABLE IF NOT EXISTS slice_matches (
                id              INTEGER PRIMARY KEY,
                motif_id        NOT NULL REFERENCES motifs(id) ON DELETE CASCADE,
                slice_id_0      NOT NULL REFERENCES slices(id) ON DELETE CASCADE,
                slice_id_1      NOT NULL REFERENCES slices(id) ON DELETE CASCADE,
                dist            FLOAT NOT NULL
            )",
        (),
    )?;

    //labels_motifs on resource_set
    t.execute(
        "CREATE TABLE IF NOT EXISTS selected_motifs (
                motif_id   REFERENCES motifs(id),
                run_config_id      NOT NULL REFERENCES run_configs(id) ON DELETE CASCADE,
                rate    FLOAT NOT NULL
            )",
        (),
    )?;
    t.execute("CREATE INDEX IF NOT EXISTS idx_selected_motifs_run_config_id_motif_id ON selected_motifs (run_config_id, motif_id);", ())?;

    Ok(())
}


mock! {
    pub MotifsStore {

    }

    #[async_trait]
    impl MotifsStore for MotifsStore {
        async fn list_label_nets_run_config_ids(
            &self,
            label: &str,
        ) -> anyhow::Result<Vec<RunConfigId>>;
        async fn save_resources_labels(&self, resource_stats_labels_probs: &HashMap<String, (ResourceStats, Vec<(String, f32)>)>, import_id: u32) -> anyhow::Result<Vec<ResourceId>>;
        async fn save_pure_sets(&self, max_len: usize, only_unused: bool) -> anyhow::Result<()>;
        async fn save_leaky_data_sets(&self, prefix: &str, limit: LimitBy, train_coeff: f32, valid_coeff: f32, min_p: f32, allow_leaky_split: bool) -> anyhow::Result<()>;
        async fn save_set_by_labels(&self, conds: Vec<(String, SetCond)>, only_unused: bool, limit: usize) -> anyhow::Result<()>;
        async fn save_groups_by_label_cnt(&self, max_resources_in_set: usize) -> anyhow::Result<()>;
        async fn export_apply_label_nets_records(&self, target_db_path: PathBuf, label_net_run_config_ids: Vec<RunConfigId>) -> anyhow::Result<()>;
        async fn resources_by_label(&self, min_p: f32) -> anyhow::Result<HashMap<String, Vec<Resource>>>;
        async fn list_resource_random_not_labeled(&self, label: &str, limit: u32) -> anyhow::Result<Vec<Resource>>;
        async fn save_label(&self, label: &str) -> anyhow::Result<u32>;
        async fn list_labels(&self, only_with_resources: bool) -> anyhow::Result<Vec<String>>;
        async fn save_params_value<'a>(&self, params: HashMap<String, Val>, run_config_id: RunConfigId, label: Option<&'a str>) -> anyhow::Result<()>;
        async fn save_slices(&self, run_config_id: RunConfigId, channel: i64, slices: Vec<(TensorIdSliceIdx, TensorIdSliceIdx, f32)>) -> anyhow::Result<()>;
        async fn save_groups_as_motifs_with_slices_matches(&self, run_config_id: RunConfigId, channel: i64, groups: HashMap<usize, HashSet<IdSlice>>, path: PathBuf) -> anyhow::Result<()>;
        async fn save_groups_as_motifs(&self, run_config_id: RunConfigId, channel: i64, groups: HashMap<usize, HashSet<IdSlice>>) -> anyhow::Result<Vec<u32>>;
        async fn add_slices_to_motif(&self, run_config_id: RunConfigId, motif_id: u32, channel: i64, slices: HashSet<IdSlice>) -> anyhow::Result<()>;
        async fn list_motifs(&self, run_config_id: RunConfigId) -> anyhow::Result<Vec<u32>>;
        async fn get_motif_meta(&self, motif_id: u32) -> anyhow::Result<(usize, RunConfigId)>;
        async fn list_motifs_slices(&self, run_config_id: RunConfigId, select_motifs_run_config_id: Option<Vec<RunConfigId>>) -> anyhow::Result<HashMap<Resource, Vec<MotifSlice>>>;
        async fn list_motif_slices(&self, motif_id: u32, select_motifs_run_config_id: Option<Vec<RunConfigId>>, limit: Option<u32>, all_but_in_run_config_id: Option<(RunConfigId, u32)>) -> anyhow::Result<(i64, RunConfigId, Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>)>;
        async fn list_random_slices_out_of_motif(&self, motif_id: u32, motif_width: u32, min_distance: Option<u32>, input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync>, desired_count: Option<usize>) -> anyhow::Result<(i64, RunConfigId, Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>)>;
        async fn list_random_slices_out_of_resources(&self, motif_id: u32, motif_width: u32, min_distance: Option<u32>, input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync>, desired_count: Option<usize>) -> anyhow::Result<(i64, RunConfigId, Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>)>;
        async fn list_run_motifs(&self, run_config_id: RunConfigId) -> anyhow::Result<Vec<RunMotifEntry>>;
        async fn save_selected_motifs(&self, run_config_id: RunConfigId, selected_motifs: Vec<(u32, f64)>) -> anyhow::Result<RunConfigId>;
        async fn select_all_motifs(&self, run_config_ids: &[RunConfigId]) -> anyhow::Result<Vec<RunConfigId>>;
        async fn list_selected_motifs(&self,motif_selection_ids: &[RunConfigId],channels: u32,limit: Option<u32>) -> anyhow::Result<HashMap<RunConfigId, Vec<Vec<(u32, f64)>>>>;
    }

    #[async_trait]
    impl thoosa_store::ExternalSqlWriter for MotifsStore {
        fn open_external_writer_conn(&self) -> anyhow::Result<Connection>;
    }

    #[async_trait]
    impl Store for MotifsStore{

        async fn save_import(&self, descr: String) -> anyhow::Result<u32>;
        async fn save_run(&self, descr: String) -> anyhow::Result<RunId>;
        async fn save_run_config<'a, 'b, 'c>(
            &self,
            input_run_config_id: Option<&'a [RunConfigId]>,
            run_config_type: &'b str,
            descr: &'c str,
            properties_vals: HashMap<String, Val>,
        ) -> anyhow::Result<RunConfigId>;

        async fn get_input_run_config_ids<'a>(&self, run_config_id: RunConfigId, input_type: Option<&'a str>) -> anyhow::Result<Vec<RunConfigId>>;

        async fn list_input_run_config_ids(&self, ids: &[RunConfigId]) -> anyhow::Result<Vec<(RunConfigId, RunConfigId)>>; //corr_run_config_id, run_config_id, label

        async fn list_run_config_ids_with_outputs(&self, input_run_config_ids: &[RunConfigId]) -> anyhow::Result<Vec<RunConfigId>>;

        async fn list_output_run_config_ids(&self, input_run_config_id: RunConfigId) -> anyhow::Result<Vec<RunConfigId>>; //corr_run_config_id, run_config_id, label

        async fn save_resources_set(
            &self,
            set: String,
            resource_keys: Vec<String>,
            group_id: Option<usize>,
        ) -> anyhow::Result<usize>;
        async fn save_resource_set_group(&self, descr: String) -> anyhow::Result<usize>;
        async fn list_resources_by_ids(
            &self,
            file_stem: &Vec<ResourceId>,
        ) -> anyhow::Result<Vec<Resource>>;



        async fn list_run_resources(&self, run_config_id: RunConfigId) -> anyhow::Result<Vec<String>>;

        async fn list_run_configs(&self) -> anyhow::Result<Vec<RunConfigEntry>>;

        async fn list_run_config_ids_by_descr(&self, descr:String, type_:&'static str) -> anyhow::Result<Vec<RunConfigId>>;
        async fn list_run_configs_page(&self, limit: usize, offset: usize) -> anyhow::Result<Vec<RunConfigEntry>>;
        async fn list_run_configs_by_ids(&self, ids: &[RunConfigId]) -> anyhow::Result<Vec<RunConfigEntry>>;

        async fn list_run_configs_filtered(&self, filter: RunConfigFilter) -> anyhow::Result<Vec<RunConfigEntry>>;
        async fn list_run_config_types(&self) -> anyhow::Result<Vec<String>>;

        async fn list_run_config_values(&self, run_config_id: RunConfigId) -> anyhow::Result<HashMap<String, Val>>;

        async fn save_trained_net(&self, run_config_id: RunConfigId, path: PathBuf, loss: f32, input_net_ids: Option<Vec<u32>>) -> anyhow::Result<u32>;

        async fn list_trained_nets(&self, run_config_ids: TrainedNetsBy, loss: Option<f32>, limit: Option<usize>) -> anyhow::Result<Vec<TrainedNet>>;
        async fn list_trained_net_entries(&self) -> anyhow::Result<Vec<TrainedNetEntry>>;
        async fn list_label_trained_nets_before(&self, run_on_before: String) -> anyhow::Result<Vec<TrainedNetEntry>>;
        async fn list_trained_nets_loss_gt(&self, loss: f32, net_type: &'static str) -> anyhow::Result<Vec<TrainedNetEntry>>;
        async fn list_unused_motif_trained_nets(&self) -> anyhow::Result<Vec<TrainedNetEntry>>;
        async fn delete_trained_nets(&self, net_ids: Vec<u32>) -> anyhow::Result<usize>;
        async fn update_run_config_run_on(&self, run_config_id: RunConfigId, run_on: String) -> anyhow::Result<()>;
        async fn list_input_nets(&self, net_id: u32) -> anyhow::Result<Vec<u32>>;

        async fn run_config_descr(&self, corr_run_config_id: RunConfigId) -> anyhow::Result<RunConfigRow>;
        // #[cfg(test)]
        // fn conn(&self) -> &parking_lot::Mutex<Connection>; //parking_lot:: for mockall
    }

    #[async_trait]
    impl ResourceStore for MotifsStore {
        async fn list_resources_by_file_stems<'a>(
            &self,
            file_stems: &'a [&'a str],
        ) -> anyhow::Result<Vec<Resource>>;

        async fn list_resource_set_resources_by_descr(
            &self,
            descr: &str,
        ) -> anyhow::Result<Vec<Resource>>;

        async fn list_resource_set_group_sets(
            &self,
            group_descr: &str,
        ) -> anyhow::Result<Vec<(usize, String)>>;

        async fn list_group_labeled_resources_sets(
            &self,
            group_decr: &str,
        ) -> anyhow::Result<Vec<(String, String)>>;

        async fn list_resource_set_resources(&self, set_id: usize) -> anyhow::Result<Vec<Resource>>;

        async fn list_resource_random_in_group(
            &self,
            group_descr: &str,
        ) -> anyhow::Result<Vec<(Resource, Vec<f32>)>>;

        async fn list_resources_not_in_group(
            &self,
            group_descr: &str,
            limit: u32,
        ) -> anyhow::Result<Vec<Resource>>;

        async fn list_random_resources(
            &self,
            limit: Option<u32>,
        ) -> anyhow::Result<Vec<Resource>>;
    }

}

impl fmt::Debug for MockMotifsStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MockStore").finish()
    }
}

/// Builds the motif-specific mock so motif tests do not depend on the general `Store` mock boundary.
pub struct MockMotifsStoreBuilder {
    store: MockMotifsStore,
}

//TODO all expectations are copy-pasted from MockStoreBuilder, replace with macros
impl MockMotifsStoreBuilder {
    pub fn new() -> Self {
        Self {
            store: MockMotifsStore::new(),
        }
    }

    //run_config_id, motif_id, channel, slices
    pub fn with_add_slices_to_motif(mut self, inputs: impl IntoIterator<Item = (RunConfigId, u32, i64, HashSet<IdSlice>)> + Debug) -> Self {
        debug!("with_add_slices_to_motif: inputs={inputs:?}");
        for inps in inputs {
            self.store.expect_add_slices_to_motif().with(eq(inps.0), eq(inps.1), eq(inps.2), eq(inps.3)).times(1).returning(|_, _, _, _| Ok(()));
        }

        self
    }

    pub fn with_add_slices_to_motif_at_least_once(mut self) -> Self {
        self.store.expect_add_slices_to_motif().times(1..).returning(|_, _, _, _| Ok(()));

        self
    }

    pub fn with_get_motif_meta<R: Into<HashMap<u32, Result<(usize, RunConfigId), ThoError>>>>(mut self, results: R) -> Self {
        let map = results.into();
        self.store.expect_get_motif_meta().returning(move |motif_id| {
            Ok(map.get(&motif_id).cloned().expect(&format!("mock is not configured to return motif meta for motif_id {motif_id}"))?)
        });
        self
    }

    pub fn with_get_input_run_config_ids<R: Into<HashMap<RunConfigId, Result<Vec<RunConfigId>, ThoError>>>>(mut self, results: R) -> Self {
        let map = results.into();
        self.store.expect_get_input_run_config_ids().returning(move |run_config_id, _input_type| {
            Ok(map.get(&run_config_id).cloned().expect(&format!("mock is not configured to return input_run_config_ids for run_config_id {run_config_id}"))?)
        });
        self
    }

    pub fn with_run_config_descr<R: Into<HashMap<RunConfigId, Result<RunConfigRow, ThoError>>>>(mut self, results: R) -> Self {
        let map = results.into();
        self.store.expect_run_config_descr().returning(move |run_config_id| {
            Ok(map.get(&run_config_id).cloned().expect(&format!("mock is not configured to return run_config_descr for run_config_id {run_config_id}"))?)
        });
        self
    }

    pub fn with_save_run_config(
        mut self,
        input_run_config_id: Option<Vec<RunConfigId>>,
        run_config_type: &'static str,
        descr: &'static str,
        _keys: &'static str,
        properties_vals: HashMap<String, Val>,
        result: Result<RunConfigId, ThoError>,
    ) -> Self {
        self.store.expect_save_run_config().returning(move |p_input_run_config_id, p_run_config_type, p_descr, p_properties_vals| {
            let p_input_run_config_id = p_input_run_config_id.map(|ids| ids.to_vec());
            if p_input_run_config_id == input_run_config_id && p_run_config_type == run_config_type && p_descr == descr && p_properties_vals == properties_vals {
                Ok(result.clone()?)
            } else {
                Err(anyhow::anyhow!(format!("mock is not configured to save_run_config for input_run_config_id={p_input_run_config_id:?} run_config_type={p_run_config_type} descr={p_descr} properties_vals={p_properties_vals:?}")))
            }
        });
        self
    }

    pub fn with_list_motifs<R: Into<HashMap<RunConfigId, Result<Vec<u32>, ThoError>>>>(mut self, results: R) -> Self {
        let map = results.into();
        self.store.expect_list_motifs().returning(move |run_config_id| {
            Ok(map.get(&run_config_id).cloned().expect(&format!("mock is not configured to return motifs for run id {run_config_id}"))?)
        });

        self
    }

    pub fn with_list_motif_slices(
        mut self,
        motif_id: u32,
        select_motifs_run_config_id: Option<Vec<RunConfigId>>,
        _limit: Option<u32>,
        _all_but_in_run_config_id: Option<(RunConfigId, u32)>,
        r: Result<(i64, RunConfigId, Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>), ThoError>,
    ) -> Self {
        self.store.expect_list_motif_slices().withf(move |p_motif_id, p_select_motifs_run_config_id, _limit, _all_but_in_run_config_id| *p_motif_id == motif_id && *p_select_motifs_run_config_id == select_motifs_run_config_id).times(1).returning(move |_, _, _, _| Ok(r.clone()?));

        self
    }

    pub fn with_list_random_slices_out_of_motif(
        mut self,
        motif_id: u32,
        motif_width: u32,
        min_distance: Option<u32>,
        r: Result<(i64, RunConfigId, Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>), ThoError>,
    ) -> Self {
        self.store.expect_list_random_slices_out_of_motif().returning(move |p_motif_id, p_width, p_min_distance, _input_len_adjuster, _desired_count| {
            if p_motif_id == motif_id && p_width == motif_width && p_min_distance == min_distance {
                Ok(r.clone()?)
            } else {
                Err(anyhow::anyhow!("Not found"))
            }
        });

        self
    }

    pub fn with_list_selected_motifs<R: Into<HashMap<Vec<RunConfigId>, Result<HashMap<RunConfigId, Vec<Vec<(u32, f64)>>>, ThoError>>>>(mut self, results: R) -> Self {
        let map = results.into();

        self.store.expect_list_selected_motifs().returning(move |run_config_id, _channels, _limit| {
            Ok(map.get(&run_config_id.to_vec()).cloned().expect(&format!("mock is not configured to return selected motifs for run id {run_config_id:?}"))?)
        });
        self
    }

    pub fn with_save_trained_net(mut self, inputs: impl IntoIterator<Item = (RunConfigId, PathBuf, f32)> + Debug) -> Self {
        debug!("with_save_trained_net: inputs={inputs:?}");
        for (run_config_id, path, loss) in inputs {
            self.store.expect_save_trained_net().with(eq(run_config_id), eq(path), eq(loss), eq(None)).times(1).returning(move |_, _, _, _| Ok(run_config_id));
        }
        self
    }

    pub fn with_list_trained_nets<I>(mut self, inputs: I) -> Self
    where
        I: IntoIterator<Item = (Vec<RunConfigId>, Option<f32>, Option<usize>, Result<Vec<TrainedNet>, ThoError>)> + Debug,
    {
        let data = inputs.into_iter().collect::<Vec<_>>();
        debug!("with_list_trained_nets: inputs={data:?}");
        let data = Arc::new(data);
        self.store.expect_list_trained_nets().returning(move |run_config_ids, loss, limit| {
            data.iter()
                .find(|(expected_run_config_ids, expected_loss, expected_limit, _)| expected_run_config_ids.as_slice() == run_config_ids.as_ref().as_slice() && *expected_loss == loss && *expected_limit == limit)
                .map(|(_, _, _, result)| Ok(result.clone()?))
                .unwrap_or_else(|| Err(anyhow::anyhow!(format!("mock is not configured to return trained nets for run ids {run_config_ids:?} loss={loss:?} limit={limit:?}"))))
        });
        self
    }

    pub fn with_list_input_nets<I>(mut self, inputs: I) -> Self
    where
        I: IntoIterator<Item = (RunConfigId, Result<Vec<u32>, ThoError>)> + Debug,
    {
        let data = inputs.into_iter().collect::<Vec<_>>();
        debug!("with_list_input_nets: inputs={data:?}");
        let data = Arc::new(data);
        self.store.expect_list_input_nets().returning(move |run_config_id| {
            data.iter()
                .find(|(expected_run_config_id, _)| expected_run_config_id == &run_config_id)
                .map(|(_, result)| Ok(result.clone()?))
                .unwrap_or_else(|| Err(anyhow::anyhow!(format!("with_list_input_nets is not configured to return nets for run_config_id={run_config_id:?}"))))
        });
        self
    }

    pub fn with_list_run_config_values<R: Into<HashMap<RunConfigId, Result<HashMap<String, Val>, ThoError>>>>(mut self, results: R) -> Self {
        let map = results.into();
        self.store.expect_list_run_config_values().returning(move |p_run_config_id| {
            Ok(map.get(&p_run_config_id).cloned().expect(&format!("mock is not configured to return run config values for run id {p_run_config_id}"))?)
        });

        self
    }

    pub fn with_list_output_run_config_ids<R: Into<HashMap<RunConfigId, Result<Vec<RunConfigId>, ThoError>>>>(mut self, results: R) -> Self {
        let map = results.into();
        self.store.expect_list_output_run_config_ids().returning(move |input_run_config_id| {
            Ok(map.get(&input_run_config_id).cloned().expect(&format!("mock is not configured to return output run ids for run id {input_run_config_id}"))?)
        });
        self
    }

    pub fn with_list_run_config_ids_with_outputs<R: Into<HashMap<Vec<RunConfigId>, Result<Vec<RunConfigId>, ThoError>>>>(mut self, results: R) -> Self {
        let map = results.into();
        self.store.expect_list_run_config_ids_with_outputs().returning(move |input_run_config_ids| {
            Ok(map.get(&input_run_config_ids.to_vec()).cloned().expect(&format!("mock is not configured to return run ids with outputs for input ids {input_run_config_ids:?}"))?)
        });
        self
    }

    pub fn with_list_resources_by_file_stems(mut self, expected_stems: Vec<String>, result: Result<Vec<Resource>, ThoError>) -> Self {
        self.store.expect_list_resources_by_file_stems().returning(move |requested| {
            let stems = requested.iter().map(|stem| stem.to_string()).collect::<Vec<_>>();
            if stems == expected_stems {
                Ok(result.clone()?)
            } else {
                Err(anyhow::anyhow!("Not found"))
            }
        });

        self
    }

    pub fn with_list_resource_set_group_sets<D: IntoIterator<Item = (&'static str, Result<Vec<(usize, String)>, ThoError>)>>(mut self, data: D) -> Self {
        let map: HashMap<&'static str, Result<Vec<(usize, String)>, ThoError>> = HashMap::from_iter(data);
        self.store.expect_list_resource_set_group_sets().returning(move |group| {
            Ok(map.get(group).expect(&format!("no mock data for {group} configured in list_group_labeled_resources_sets")).clone()?)
        });
        self
    }

    pub fn with_list_group_labeled_resources_sets<D: IntoIterator<Item = (&'static str, Result<Vec<(String, String)>, ThoError>)>>(mut self, data: D) -> Self {
        let map: HashMap<&'static str, Result<Vec<(String, String)>, ThoError>> = HashMap::from_iter(data);
        self.store.expect_list_group_labeled_resources_sets().returning(move |group| {
            Ok(map.get(group).expect(&format!("no mock data for {group} configured in list_group_labeled_resources_sets")).clone()?)
        });
        self
    }

    pub fn with_list_resources_not_in_group<D: IntoIterator<Item = (&'static str, Result<Vec<Resource>, ThoError>)>>(mut self, data: D) -> Self {
        let map: HashMap<&'static str, Result<Vec<Resource>, ThoError>> = HashMap::from_iter(data);
        self.store.expect_list_resources_not_in_group().returning(move |group, _limit| {
            Ok(map.get(group).expect(&format!("no mock data for {group} configured in list_resource_set_resources_by_descr")).clone()?)
        });
        self
    }

    pub fn with_list_random_resources<D: IntoIterator<Item = (Option<u32>, Result<Vec<Resource>, ThoError>)>>(mut self, data: D) -> Self {
        let map: HashMap<Option<u32>, Result<Vec<Resource>, ThoError>> = HashMap::from_iter(data);
        self.store.expect_list_random_resources().returning(move |limit| {
            Ok(map.get(&limit).expect(&format!("no mock data for limit={limit:?} configured in list_random_resources")).clone()?)
        });
        self
    }

    pub fn with_list_resource_random_not_labeled<D: IntoIterator<Item = (&'static str, Result<Vec<Resource>, ThoError>)>>(mut self, data: D) -> Self {
        let map: HashMap<&'static str, Result<Vec<Resource>, ThoError>> = HashMap::from_iter(data);
        self.store.expect_list_resource_random_not_labeled().returning(move |label, _limit| {
            Ok(map.get(label).expect(&format!("no mock data for {label} configured in list_resources_not_labled")).clone()?)
        });
        self
    }

    pub fn with_list_resource_random_in_group<D: IntoIterator<Item = (&'static str, Result<Vec<(Resource, Vec<f32>)>, ThoError>)>>(mut self, data: D) -> Self {
        let map: HashMap<&'static str, Result<Vec<(Resource, Vec<f32>)>, ThoError>> = HashMap::from_iter(data);
        self.store.expect_list_resource_random_in_group().returning(move |group| {
            Ok(map.get(group).expect(&format!("no mock data for {group} configured in list_resource_random_in_group")).clone()?)
        });
        self
    }

    pub fn with_list_resource_set_resources_by_descr<D: IntoIterator<Item = (&'static str, Result<Vec<Resource>, ThoError>)>>(mut self, data: D) -> Self {
        let map: HashMap<&'static str, Result<Vec<Resource>, ThoError>> = HashMap::from_iter(data);
        self.store.expect_list_resource_set_resources_by_descr().returning(move |group| {
            Ok(map.get(group).expect(&format!("no mock data for {group} configured in list_resource_set_resources_by_descr")).clone()?)
        });
        self
    }

    pub fn with_save_groups_as_motifs(mut self, mut check_groups: Vec<CheckGroups>) -> Self {
        let times = check_groups.len();
        check_groups.reverse();
        let i_atomic = AtomicUsize::new(0);
        self.store.expect_save_groups_as_motifs().withf(move |run_config_id, channel, groups| {
            let i = i_atomic.fetch_add(1, Relaxed);
            debug!("with_save_groups_as_motifs: run_config_id={run_config_id:?}, channel={channel:?}, groups={groups:?} i={i}");
            match &check_groups[i] {
                CheckGroups::Lengths(expected_groups_lengths) => {
                    let mut group_lens = groups.values().map(|group| group.len()).collect::<Vec<_>>();
                    group_lens.sort();
                    assert_eq!(&group_lens, expected_groups_lengths);
                }
                CheckGroups::MinGroups(num_groups) => assert!(num_groups <= &groups.len()),
            }
            true
        }).times(times).returning(|_, _, groups| Ok((0..groups.len()).map(|idx| idx as u32 + 1).collect()));
        self
    }

    pub fn with_save_groups_as_motifs_inputs(mut self, inputs: impl IntoIterator<Item = (RunConfigId, i64, HashMap<usize, HashSet<IdSlice>>, Vec<u32>)> + Debug) -> Self {
        for (run_config_id, channel, groups, motif_ids) in inputs {
            self.store.expect_save_groups_as_motifs().with(eq(run_config_id), eq(channel), eq(groups)).times(1).returning(move |_, _, _| Ok(motif_ids.clone()));
        }
        self
    }

    pub fn with_save_selected_motifs(mut self, inputs: impl IntoIterator<Item = (RunConfigId, Vec<(u32, f64)>, Result<RunConfigId, ThoError>)> + Debug) -> Self {
        for (run_config_id, selected_motifs, result) in inputs {
            self.store.expect_save_selected_motifs().with(eq(run_config_id), eq(selected_motifs)).times(1).returning(move |_, _| Ok(result.clone()?));
        }
        self
    }

    pub fn build(self) -> Arc<dyn MotifsStore> {
        Arc::new(self.store)
    }
}


#[cfg(test)]
mod store_tests {
    use thoosa_store::ExternalSqlWriter;
    use thoosa_shared::resource::ResourceStore;
    use std::collections::HashMap;
    use std::fmt::{Display, Formatter};
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Instant;

    use log::debug;
    use rusqlite::params;
    use parking_lot::{Mutex, RawMutex, RwLock};
    use thoosa_shared::resource::{Resource, ResourceId, ResourceStats};
    use thoosa_shared::run_configs::{RunConfigId, RUN_ID};
    use thoosa_shared::val::Val;
    use thoosa_shared::{TensorIdSliceIdx, ThoError, cdt, cdt_for_dir, MotifSlice};
    use super::{MotifsStore, create_motif_tables};
    use crate::motif_tracker_store::{MotifTrackerStore, SlicesMatch, SqlMotifTrackerStore};

    use thoosa_store::{LimitBy, RunConfigFilter, SetCond, SqlStore, Store, TrainedNetsBy};
    use std::path::PathBuf;

    // These legacy database tests now exercise the motif extension trait and schema explicitly.
    async fn test_store() -> anyhow::Result<Arc<SqlStore>> {
        Ok(Arc::new(SqlStore::in_mem_(vec![thoosa_store::create_tables, create_motif_tables]).await?))
    }

    pub(crate) async fn test_store_from_path<P: AsRef<Path>>(path: P) -> anyhow::Result<Arc<SqlStore>> {
        Ok(Arc::new(SqlStore::from_path_with_tables(path, vec![thoosa_store::create_tables, create_motif_tables]).await?))
    }

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_group_resources() -> anyhow::Result<()> {
        // fs::remove_file("/tmp/test_db.db3").unwrap_or(());
        // let mut store = test_store_from_path("/tmp/test_db.db3"))?;

        let mut store = test_store().await?;

        let import_id = store.save_import("test".to_string()).await?;

        let _aaa_id = store.save_label("aaa").await?;
        let _bbb_id = store.save_label("bbb").await?;
        let _ccc_id = store.save_label("ccc").await?;

        //pure resources
        let n = 100;
        store.save_resources_labels(
            &(0..n).fold(
                HashMap::new(),
                |mut a: HashMap<String, (ResourceStats, Vec<(String, f32)>)>, i| {
                    a.insert(
                        format!("{:3}_aaa", i),
                        (ResourceStats::new(vec![1, 100]), vec![("aaa".into(), 1.0)]),
                    );
                    a
                },
            ),
            import_id,
        ).await?;

        store.save_resources_labels(
            &(0..n).fold(
                HashMap::new(),
                |mut a: HashMap<String, (ResourceStats, Vec<(String, f32)>)>, i| {
                    a.insert(
                        format!("{:3}_bbb", i),
                        (ResourceStats::new(vec![1, 100]), vec![("bbb".into(), 1.0)]),
                    );
                    a
                },
            ),
            import_id,
        ).await?;

        store.save_resources_labels(
            &(0..n).fold(
                HashMap::new(),
                |mut a: HashMap<String, (ResourceStats, Vec<(String, f32)>)>, i| {
                    a.insert(
                        format!("{:3}_ccc", i),
                        (ResourceStats::new(vec![1, 100]), vec![("ccc".into(), 1.0)]),
                    );
                    a
                },
            ),
            import_id,
        ).await?;

        //two labels resources
        store.save_resources_labels(
            &(0..n).fold(
                HashMap::new(),
                |mut a: HashMap<String, (ResourceStats, Vec<(String, f32)>)>, i| {
                    a.insert(
                        format!("{:3}_ab", i),
                        (
                            ResourceStats::new(vec![1, 100]),
                            vec![("aaa".into(), 0.5), ("bbb".into(), 0.5)],
                        ),
                    );
                    a
                },
            ),
            import_id,
        ).await?;

        store.save_resources_labels(
            &(0..n).fold(
                HashMap::new(),
                |mut a: HashMap<String, (ResourceStats, Vec<(String, f32)>)>, i| {
                    a.insert(
                        format!("{:3}_bc", i),
                        (
                            ResourceStats::new(vec![1, 100]),
                            vec![("bbb".into(), 0.5), ("ccc".into(), 0.5)],
                        ),
                    );
                    a
                },
            ),
            import_id,
        ).await?;

        store.save_resources_labels(
            &(0..n).fold(
                HashMap::new(),
                |mut a: HashMap<String, (ResourceStats, Vec<(String, f32)>)>, i| {
                    a.insert(
                        format!("{:3}_ca", i),
                        (
                            ResourceStats::new(vec![1, 100]),
                            vec![("ccc".into(), 0.5), ("aaa".into(), 0.5)],
                        ),
                    );
                    a
                },
            ),
            import_id,
        ).await?;

        //three labels resources
        store.save_resources_labels(
            &(0..n).fold(
                HashMap::new(),
                |mut a: HashMap<String, (ResourceStats, Vec<(String, f32)>)>, i| {
                    a.insert(
                        format!("{:3}_abc", i),
                        (
                            ResourceStats::new(vec![1, 100]),
                            vec![
                                ("aaa".into(), 0.3),
                                ("bbb".into(), 0.3),
                                ("ccc".into(), 0.4),
                            ],
                        ),
                    );
                    a
                },
            ),
            import_id,
        ).await?;

        let t1 = Instant::now();
        store.save_groups_by_label_cnt(10).await?;

        assert_eq!(store.list_resource_set_group_sets("group 1 labels/all").await?.len(), 3);
        assert_eq!(store.list_resource_set_group_sets("group 2 labels/all").await?.len(), 3);
        assert_eq!(store.list_resource_set_group_sets("group 3 labels/all").await?.len(), 1);
        assert_eq!(store.list_resource_set_group_sets("group 1 labels/10/0-10").await?.len(), 3);

        debug!("done in {}", t1.elapsed().as_millis());

        // fs::remove_file(db_file)?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_list_resources_not_labled() -> anyhow::Result<()> {
        let mut store = test_store().await?;

        let import_id = store.save_import("test".to_string()).await?;
        store.save_label("aaa").await?;
        store.save_label("bbb").await?;

        store.save_resources_labels(
            &HashMap::from([
                (
                    "aaa_only".to_string(),
                    (ResourceStats::new(vec![1, 10]), vec![("aaa".into(), 1.0)]),
                ),
                (
                    "bbb_only".to_string(),
                    (ResourceStats::new(vec![1, 10]), vec![("bbb".into(), 1.0)]),
                ),
                (
                    "both".to_string(),
                    (
                        ResourceStats::new(vec![1, 10]),
                        vec![("aaa".into(), 0.5), ("bbb".into(), 0.5)],
                    ),
                ),
            ]),
            import_id,
        ).await?;

        let mut not_aaa = store.list_resource_random_not_labeled("aaa", 10).await?;
        not_aaa.sort_by(|a, b| a.file_stem().cmp(b.file_stem()));
        assert_eq!(vec!["bbb_only"], not_aaa.into_iter().map(|r| r.file_stem().to_string()).collect::<Vec<_>>());

        let mut not_bbb = store.list_resource_random_not_labeled("bbb", 10).await?;
        not_bbb.sort_by(|a, b| a.file_stem().cmp(b.file_stem()));
        assert_eq!(vec!["aaa_only"], not_bbb.into_iter().map(|r| r.file_stem().to_string()).collect::<Vec<_>>());

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_save_data_sets_group_created() -> anyhow::Result<()> {
        let mut store = test_store().await?;

        init_test_store(&store).await?;

        let labels_count = store.list_labels(true).await?.len();
        let train_sets = store.list_resource_set_group_sets("A_train").await?;
        let valid_sets = store.list_resource_set_group_sets("A_valid").await?;
        let test_sets = store.list_resource_set_group_sets("A_test").await?;

        assert_eq!(labels_count-1, train_sets.len());  //C label has no resources
        assert_eq!(labels_count-1, valid_sets.len());
        assert_eq!(labels_count-1, test_sets.len());
        assert!(train_sets.iter().all(|(_, descr)| descr.starts_with("A_")));
        assert!(valid_sets.iter().all(|(_, descr)| descr.starts_with("A_")));
        assert!(test_sets.iter().all(|(_, descr)| descr.starts_with("A_")));

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_save_data_sets_allow_leaky_split_reuses_resources() -> anyhow::Result<()> {
        let mut store = test_store().await?;
        let import_id = store.save_import("test".to_string()).await?;
        store.save_label("aaa").await?;
        store.save_label("bbb").await?;
        store.save_label("ccc").await?;
        store.save_label("ddd").await?;

        store.save_resources_labels(
            &HashMap::from([
                (
                    "aaa_only".to_string(),
                    (ResourceStats::new(vec![1, 100]), vec![("aaa".into(), 1.0)]),
                ),
                (
                    "bbb_only".to_string(),
                    (ResourceStats::new(vec![1, 100]), vec![("bbb".into(), 1.0)]),
                ),
                (
                    "ccc_only_0".to_string(),
                    (ResourceStats::new(vec![1, 100]), vec![("ccc".into(), 1.0)]),
                ),
                (
                    "ccc_only_1".to_string(),
                    (ResourceStats::new(vec![1, 100]), vec![("ccc".into(), 1.0)]),
                ),
                (
                    "ccc_only_2".to_string(),
                    (ResourceStats::new(vec![1, 100]), vec![("ccc".into(), 1.0)]),
                ),
                (
                    "ccc_only_3".to_string(),
                    (ResourceStats::new(vec![1, 100]), vec![("ccc".into(), 1.0)]),
                ),
            ]),
            import_id,
        ).await?;

        store.save_leaky_data_sets("L", LimitBy::Count(10), 0.6, 0.2, 1.0, true).await?;

        let mut train_sets_resources = HashMap::new();
        for (set_id, descr) in store.list_resource_set_group_sets("L10_train").await? {
            train_sets_resources.insert(
                descr,
                store
                    .list_resource_set_resources(set_id).await?
                    .into_iter()
                    .map(|resource| resource.file_stem().to_string())
                    .collect::<Vec<_>>(),
            );
        }

        let mut valid_sets_resources = HashMap::new();
        for (set_id, descr) in store.list_resource_set_group_sets("L10_valid").await? {
            valid_sets_resources.insert(
                descr,
                store
                    .list_resource_set_resources(set_id).await?
                    .into_iter()
                    .map(|resource| resource.file_stem().to_string())
                    .collect::<Vec<_>>(),
            );
        }

        let mut test_sets_resources = HashMap::new();
        for (set_id, descr) in store.list_resource_set_group_sets("L10_test").await? {
            test_sets_resources.insert(
                descr,
                store
                    .list_resource_set_resources(set_id).await?
                    .into_iter()
                    .map(|resource| resource.file_stem().to_string())
                    .collect::<Vec<_>>(),
            );
        }

        debug!("train_sets_resources={train_sets_resources:?}");
        debug!("valid_sets_resources={valid_sets_resources:?}");
        debug!("test_sets_resources={test_sets_resources:?}");

        assert_eq!(vec!["aaa_only".to_string()], train_sets_resources["L10_aaa_train"]);
        assert_eq!(vec!["aaa_only".to_string()], valid_sets_resources["L10_aaa_valid"]);
        assert_eq!(vec!["aaa_only".to_string()], test_sets_resources["L10_aaa_test"]);

        assert_eq!(vec!["bbb_only".to_string()], train_sets_resources["L10_bbb_train"]);
        assert_eq!(vec!["bbb_only".to_string()], valid_sets_resources["L10_bbb_valid"]);
        assert_eq!(vec!["bbb_only".to_string()], test_sets_resources["L10_bbb_test"]);

        // assert_eq!(vec!["ccc_only_0", "ccc_only_2"], train_sets_resources["L10_ccc_train"]);
        // assert_eq!(vec!["ccc_only_0"],               valid_sets_resources["L10_ccc_valid"]);
        // assert_eq!(vec!["ccc_only_1", "ccc_only_3"], test_sets_resources["L10_ccc_test"]);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_save_data_sets_limit_by_length_keeps_resources_within_total_length() -> anyhow::Result<()> {
        let mut store = test_store().await?;
        let import_id = store.save_import("test".to_string()).await?;
        store.save_label("aaa").await?;
        store.save_label("bbb").await?;

        store.save_resources_labels(
            &HashMap::from([
                (
                    "aaa_len_25".to_string(),
                    (ResourceStats::new(vec![1, 25]), vec![("aaa".into(), 1.0)]),
                ),
                (
                    "aaa_len_50".to_string(),
                    (ResourceStats::new(vec![1, 50]), vec![("aaa".into(), 1.0)]),
                ),
                (
                    "aaa_len_100".to_string(),
                    (ResourceStats::new(vec![1, 100]), vec![("aaa".into(), 1.0)]),
                ),
                (
                    "bbb_len_150".to_string(),
                    (ResourceStats::new(vec![1, 150]), vec![("bbb".into(), 1.0)]),
                ),
                (
                    "bbb_len_200".to_string(),
                    (ResourceStats::new(vec![1, 200]), vec![("bbb".into(), 1.0)]),
                ),
            ]),
            import_id,
        ).await?;

        store.save_leaky_data_sets("L", LimitBy::Length(100), 0.6, 0.2, 1.0, true).await?;

        let mut train_sets_resources = HashMap::new();
        for (set_id, descr) in store.list_resource_set_group_sets("L100_train").await? {
            train_sets_resources.insert(
                descr,
                store
                    .list_resource_set_resources(set_id).await?
                    .into_iter()
                    .map(|resource| resource.file_stem().to_string())
                    .collect::<Vec<_>>(),
            );
        }

        let mut valid_sets_resources = HashMap::new();
        for (set_id, descr) in store.list_resource_set_group_sets("L100_valid").await? {
            valid_sets_resources.insert(
                descr,
                store
                    .list_resource_set_resources(set_id).await?
                    .into_iter()
                    .map(|resource| resource.file_stem().to_string())
                    .collect::<Vec<_>>(),
            );
        }

        let mut test_sets_resources = HashMap::new();
        for (set_id, descr) in store.list_resource_set_group_sets("L100_test").await? {
            test_sets_resources.insert(
                descr,
                store
                    .list_resource_set_resources(set_id).await?
                    .into_iter()
                    .map(|resource| resource.file_stem().to_string())
                    .collect::<Vec<_>>(),
            );
        }

        assert_eq!(vec!["aaa_len_25".to_string()], train_sets_resources["L100_aaa_train"]);
        assert_eq!(vec!["aaa_len_50".to_string()], valid_sets_resources["L100_aaa_valid"]);
        assert_eq!(vec!["aaa_len_50".to_string()], test_sets_resources["L100_aaa_test"]);

        //if the smallest resources is bigger than limit it's still used
        assert_eq!(vec!["bbb_len_150".to_string()], train_sets_resources["L100_bbb_train"]);
        assert_eq!(vec!["bbb_len_150".to_string()], valid_sets_resources["L100_bbb_valid"]);
        assert_eq!(vec!["bbb_len_150".to_string()], test_sets_resources["L100_bbb_test"]);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_save_slices() -> anyhow::Result<()> {
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        fs::create_dir_all(&test_dir)?;
        let mut store = test_store_from_path(
            test_dir.join("test_db.db3"),
        ).await?;

        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        let resource_ids = init_test_store(&store).await?;

        let all_group_id = store.save_resource_set_group("group 1 labels/100/100-200".into()).await?;
        let _set_all_id = store.save_resources_set(
            "All".to_string(),
            vec!["1111".into(), "2222".into(), "3333".into()],
            Some(all_group_id),
        ).await?;

        let group_sets = store.list_resource_set_group_sets("pure group 10").await?;
        assert_eq!(3, group_sets.len());

        let mut keys_labels_p = store.list_resource_random_in_group("group 1 labels/100/100-200").await?;

        keys_labels_p.sort_by(|a, b| ((a.1[0] * 10.) as u32).cmp(&((b.1[0] * 10.) as u32)));

        debug!("keys_labels_p={:?}", keys_labels_p);

        assert_eq!(
            vec![
                vec![0.0, 0.0, 1.0],
                vec![0.5, 0.5, 0.0],
                vec![1.0, 0.0, 0.0],
            ],
            keys_labels_p
                .into_iter()
                .map(|v| v.1)
                .collect::<Vec<Vec<f32>>>()
        );

        //Vec<(TensorSliceIdx, TensorSliceIdx, f32)>
        let slices: Vec<SlicesMatch> = vec![
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[0], 100),
                TensorIdSliceIdx(resource_ids[0], 200),
                0.91,
            ),
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[0], 100),
                TensorIdSliceIdx(resource_ids[1], 200),
                0.92,
            ),
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[3], 100),
                TensorIdSliceIdx(resource_ids[3], 200),
                0.93,
            ),
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[3], 200),
                TensorIdSliceIdx(resource_ids[5], 200),
                0.94,
            ),
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[3], 200),
                TensorIdSliceIdx(resource_ids[5], 300),
                0.94,
            ),
        ];

        let find_motifs_run_config_id = store.save_run_config(
            None,
            &"todo",
            "find_motifs",
            HashMap::from([
                ("prop1".into(), Val::Int(1)),
                ("prop2".into(), Val::Real(2.0)),
                ("prop3".into(), Val::Str("333".to_string())),
            ]),
        ).await?;

        let slices_store_path = test_dir.join("test_slices_db.db3");
        let mut slices_store = SqlMotifTrackerStore::new(&slices_store_path, find_motifs_run_config_id, 7)
            .expect("can't create tmp sotre");

        slices_store.save_slices(slices.clone())?;

        let groups = slices_store.load_slices_into_groups(store.as_ref(), None, None).await?;

        {
            let store_guard = Arc::clone(&store);
            let mut guard = store_guard.open_external_writer_conn()?;
            let mut stmt = guard.prepare("SELECT sm.motif_id, s_0.resource_id, s_0.slice_idx, s_1.resource_id, s_1.slice_idx FROM slice_matches sm \
             JOIN slices s_0 ON s_0.id=sm.slice_id_0 \
             JOIN slices s_1 ON s_1.id=sm.slice_id_1 \
              ORDER BY s_0.resource_id, s_0.slice_idx, s_1.resource_id, s_1.slice_idx")?;

            let rows = stmt
                .query_map([], |row| {
                    Ok((
                        row.get(0)?, //motif_id
                        row.get(1)?, //tensor_id_0
                        row.get(2)?, //slice_idx_0
                        row.get(3)?, //tensor_id_1
                        row.get(4)?, //slice_idx_1
                    ))
                })?
                .map(|v| v.unwrap())
                .collect::<Vec<(i64, ResourceId, i64, ResourceId, i64)>>();

            debug!("rows={:?}", rows);
            debug!("slices={:?}", slices);

            for i in 0..5 {
                assert_eq!(&slices[i].0.0, &rows[i].1);
                assert_eq!(&slices[i].0.1, &(rows[i].2 as u32));
                assert_eq!(&slices[i].1.0, &rows[i].3);
                assert_eq!(&slices[i].1.1, &(rows[i].4 as u32));
            }
        };

        let motif_slices_map = store.list_motifs_slices(find_motifs_run_config_id, None).await?;
        debug!("motif_slices_map={:?}", motif_slices_map);

        // assert_eq!(4, map.len());
        let motif_slices_1111 = motif_slices_map
            .get(&Resource::new(resource_ids[0], 1, "", 0))
            .unwrap();
        assert_eq!(2, motif_slices_1111.len()); //100, 200

        let motif_slices_2222 = motif_slices_map
            .get(&Resource::new(resource_ids[1], 1, "", 0))
            .unwrap();
        assert_eq!(1, motif_slices_2222.len());

        let motif_slices_444 = motif_slices_map
            .get(&Resource::new(resource_ids[3], 1, "", 0))
            .unwrap();
        assert_eq!(2, motif_slices_444.len()); //100, 200
        assert_eq!(
            2,
            motif_slices_map
                .get(&Resource::new(resource_ids[5], 1, "", 0))
                .unwrap()
                .len()
        ); //200,300

        debug!("motif_slices_1111={motif_slices_1111:?}");
        debug!("motif_slices_444={motif_slices_444:?}");

        let select_motifs_run_config_id = store.save_run_config(
            Some(&[find_motifs_run_config_id]),
            &"todo",
            "find_motifs",
            HashMap::from([
                ("prop1".into(), Val::Int(1)),
                ("prop2".into(), Val::Real(2.0)),
                ("prop3".into(), Val::Str("333".to_string())),
            ]),
        ).await?;

        let _select_motifs_run_config_id = store.save_selected_motifs(
            select_motifs_run_config_id,
            vec![
                (motif_slices_444[0].motif_id, 0.1),  //motif_id corr
                (motif_slices_1111[0].motif_id, 0.2), //motif_id corr
            ],
        ).await?;

        let input_run_config_ids = store.get_input_run_config_ids(select_motifs_run_config_id, None).await?;

        assert_eq!(
            vec![find_motifs_run_config_id],
            input_run_config_ids
        );

        let (run_config, motifs) = store.list_selected_motifs(&[select_motifs_run_config_id], 20, None).await?.into_iter().next().expect("no selected motif");

        debug!("motifs: {:?}", motifs);

        //7th channel
        let mut motif_ids = motifs[7].iter().map(|(motif_id, _)| *motif_id).collect::<Vec<_>>();
        motif_ids.sort();
        let mut expected_ids = vec![motif_slices_444[0].motif_id, motif_slices_1111[0].motif_id];
        expected_ids.sort();
        assert_eq!(expected_ids, motif_ids);

        let selected_run_motifs = store.list_run_motifs(select_motifs_run_config_id).await?;
        assert_eq!(2, selected_run_motifs.len());
        assert!(selected_run_motifs.iter().all(|entry| entry.channel_id == 7));
        let selected_run_motifs_by_id =
            selected_run_motifs.iter().map(|entry| (entry.motif_id, entry)).collect::<HashMap<_, _>>();
        let motif_1111_entry = selected_run_motifs_by_id.get(&motif_slices_1111[0].motif_id).unwrap();
        let motif_444_entry = selected_run_motifs_by_id.get(&motif_slices_444[0].motif_id).unwrap();
        assert_eq!(Some(0.2), motif_1111_entry.score);
        assert_eq!(Some(0.1), motif_444_entry.score);
        assert_eq!(find_motifs_run_config_id, motif_1111_entry.source_run_config_id);
        assert_eq!(find_motifs_run_config_id, motif_444_entry.source_run_config_id);

        let (_channel, _run_config_id, motif_1111_slices_all, motif_1111_resources_map) =
            store.list_motif_slices(motif_slices_1111[0].motif_id, None, None, None).await?;
        let (_channel, _run_config_id, motif_444_slices_all, motif_444_resources_map) =
            store.list_motif_slices(motif_slices_444[0].motif_id, None, None, None).await?;
        assert_eq!(motif_1111_slices_all.len(), motif_1111_entry.slices_cnt);
        assert_eq!(motif_1111_resources_map.len(), motif_1111_entry.resources_cnt);
        assert_eq!(motif_444_slices_all.len(), motif_444_entry.slices_cnt);
        assert_eq!(motif_444_resources_map.len(), motif_444_entry.resources_cnt);

        let find_run_motifs = store.list_run_motifs(find_motifs_run_config_id).await?;
        assert_eq!(2, find_run_motifs.len());
        assert!(find_run_motifs.iter().all(|entry| entry.score.is_none()));
        assert!(find_run_motifs.iter().all(|entry| entry.source_run_config_id == find_motifs_run_config_id));
        let mut find_run_motif_ids = find_run_motifs.iter().map(|entry| entry.motif_id).collect::<Vec<_>>();
        find_run_motif_ids.sort();
        assert_eq!(expected_ids, find_run_motif_ids);
        //
        // //TODO
        let (channel, run_config_id_, mut slices, resources_map) = store.list_motif_slices(motif_slices_1111[0].motif_id, None, None, None).await?;
        slices.sort();

        debug!("slices={slices:?}");

        assert_eq!(
            vec![
                TensorIdSliceIdx(resource_ids[0], 100),
                TensorIdSliceIdx(resource_ids[0], 200),
                TensorIdSliceIdx(2, 200)
            ],
            slices
        );

        assert_eq!(7, channel);
        assert_eq!(find_motifs_run_config_id, run_config_id_);

        let (channel, run_config_id_, mut slices, resources_map) = store.list_motif_slices(motif_slices_1111[0].motif_id, None, Some(1), None).await?;
        slices.sort();


        let resources = store.list_resources_not_in_group("pure group 10", 10).await;
        debug!("resources={resources:?}");

        let (channel, run_config_id_, mut slices, resources_map) = store.list_motif_slices(motif_slices_1111[0].motif_id, None, Some(3), Some((0, 0))).await?;

        debug!("random slices={slices:?}");
        //
        // // let (_channel, _run_config_id_, mut slices) = store.list_motif_slices(1, Some(10), Some((run_config_id, channel as usize))).await?;
        // //
        // // slices.sort();
        // //
        // // if !m_1_first {
        // //     assert_eq!(vec![
        // //         TensorSliceIdx(1, 100), TensorSliceIdx(1, 200), TensorSliceIdx(2, 200)
        // //     ], slices);
        // // } else {
        // //     assert_eq!(vec![
        // //         TensorSliceIdx(3, 100), TensorSliceIdx(3, 200), TensorSliceIdx(6, 200), TensorSliceIdx(6, 300)
        // //     ], slices);
        // // }
        //
        // // debug!("corrs_descr={:?}", store.corrs_descr(corr_id)?);
        //
        // assert_eq!(
        //     vec![(1, "seizure_vote".into()), (0, "lpd_vote".into()), (0, "gpd_vote".into()), (0, "lrda_vote".into()), (0, "grda_vote".into()), (0, "other_vote".into())],
        //     store.run_configs_descr(select_motifs_run_config_id)?
        // );
        //
        // store.lock().delete_slices_without_matches(run_config_id)?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_select_all_motifs_creates_bypass_selection_runs() -> anyhow::Result<()> {
        let store = test_store().await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        let first_find_run_config_id = store.save_run_config(None, &"FindMotifsRunConfig", "first", HashMap::new()).await?;
        let second_find_run_config_id = store.save_run_config(None, &"BucketingRunConfig", "second", HashMap::new()).await?;

        let (first_motif_ids, second_motif_ids) = {
            let mut guard = store.open_external_writer_conn()?;
            let t = guard.transaction()?;
            let insert_motif = |run_config_id: RunConfigId, channel_id: i64, group_ord: i64| -> anyhow::Result<u32> {
                t.execute("INSERT INTO motifs (group_ord, run_config_id, channel_id) VALUES (?, ?, ?)", params![group_ord, run_config_id as i64, channel_id])?;
                let motif_id = t.query_row("SELECT last_insert_rowid()", [], |row| {
                    row.get::<usize, u32>(0)
                })?;
                Ok(motif_id)
            };

            let first_motif_ids = vec![insert_motif(first_find_run_config_id, 0, 0)?, insert_motif(first_find_run_config_id, 1, 1)?];
            let second_motif_ids = vec![insert_motif(second_find_run_config_id, 1, 0)?];
            t.commit()?;
            (first_motif_ids, second_motif_ids)
        };

        let selected_run_config_ids = store.select_all_motifs(&[first_find_run_config_id, second_find_run_config_id]).await?;

        assert_eq!(2, selected_run_config_ids.len());

        let configs = store.list_run_configs_by_ids(&selected_run_config_ids).await?;
        assert_eq!(2, configs.len());
        assert!(configs.iter().all(|run_config| run_config.run_config_type == "SelectMotifsRunConfig"));
        assert!(configs.iter().all(|run_config| run_config.descr == "bypass"));

        let input_run_config_ids = store.list_input_run_config_ids(&selected_run_config_ids).await?;
        assert!(input_run_config_ids.contains(&(selected_run_config_ids[0], first_find_run_config_id)));
        assert!(input_run_config_ids.contains(&(selected_run_config_ids[1], second_find_run_config_id)));

        let (_run_config_id, first_selected) = store.list_selected_motifs(&[selected_run_config_ids[0]], 2, None).await?.into_iter().next().expect("no selected motif");;
        assert_eq!(vec![(first_motif_ids[0], 1.0)], first_selected[0]);
        assert_eq!(vec![(first_motif_ids[1], 1.0)], first_selected[1]);

        let (_run_config_id, second_selected) = store.list_selected_motifs(&[selected_run_config_ids[1]], 2, None).await?.into_iter().next().expect("no selected motif");;
        assert_eq!(Vec::<(u32, f64)>::new(), second_selected[0]);
        assert_eq!(vec![(second_motif_ids[0], 1.0)], second_selected[1]);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_save_list_trained_nets() -> anyhow::Result<()> {
        let mut store = test_store().await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        let run_config_id = store.save_run_config(
            None,
            &"trained_nets",
            "trained_nets",
            HashMap::new(),
        ).await?;

        for i in 0..105 {
            store.save_trained_net(run_config_id, PathBuf::from(format!("nets/{i:03}")), i as f32, None).await?;
        }

        let selected = store.list_trained_nets(TrainedNetsBy::RunConfigIds(vec![run_config_id]), None, None).await?;
        assert_eq!(105, selected.len());
        assert_eq!(run_config_id, selected[0].run_config_id);
        assert_eq!(PathBuf::from("nets/000"), selected[0].net_path);
        assert_eq!(0.0, selected[0].loss);
        assert_eq!(run_config_id, selected[104].run_config_id);
        assert_eq!(PathBuf::from("nets/104"), selected[104].net_path);
        assert_eq!(104.0, selected[104].loss);

        let limited = store.list_trained_nets(TrainedNetsBy::RunConfigIds(vec![run_config_id]), None, Some(3)).await?;
        assert_eq!(3, limited.len());
        assert_eq!(PathBuf::from("nets/000"), limited[0].net_path);
        assert_eq!(PathBuf::from("nets/001"), limited[1].net_path);
        assert_eq!(PathBuf::from("nets/002"), limited[2].net_path);

        let filtered = store.list_trained_nets(TrainedNetsBy::RunConfigIds(vec![run_config_id]), Some(15.0), None).await?;
        assert_eq!(15, filtered.len());
        assert_eq!(PathBuf::from("nets/000"), filtered[0].net_path);
        assert_eq!(PathBuf::from("nets/014"), filtered[14].net_path);

        let filtered_limited = store.list_trained_nets(TrainedNetsBy::RunConfigIds(vec![run_config_id]), Some(15.0), Some(5)).await?;
        assert_eq!(5, filtered_limited.len());
        assert_eq!(PathBuf::from("nets/000"), filtered_limited[0].net_path);
        assert_eq!(PathBuf::from("nets/004"), filtered_limited[4].net_path);

        let second_run_config_id = store.save_run_config(
            None,
            &"trained_nets",
            "trained_nets_2",
            HashMap::new(),
        ).await?;
        store.save_trained_net(second_run_config_id, PathBuf::from("nets/second"), 0.5, None).await?;
        let multi_run = store.list_trained_nets(TrainedNetsBy::RunConfigIds(vec![run_config_id, second_run_config_id]), Some(1.0), None).await?;
        assert_eq!(2, multi_run.len());
        assert_eq!(run_config_id, multi_run[0].run_config_id);
        assert_eq!(PathBuf::from("nets/000"), multi_run[0].net_path);
        assert_eq!(second_run_config_id, multi_run[1].run_config_id);
        assert_eq!(PathBuf::from("nets/second"), multi_run[1].net_path);

        let empty_run_config_ids = store.list_trained_nets(TrainedNetsBy::RunConfigIds(vec![]), None, None).await?;
        assert!(empty_run_config_ids.is_empty());

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_list_motifs_slices_filters_slice_run_config_ids() -> anyhow::Result<()> {
        let mut store = test_store().await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        let resource_ids = {
            let import_id = store.save_import("test".to_string()).await?;
            store.save_label("aaa").await?;

            let ids = store.save_resources_labels(
                &HashMap::from([(
                    "res_a".to_string(),
                    (ResourceStats::new(vec![1, 10]), vec![("aaa".into(), 1.0)]),
                )]),
                import_id,
            ).await?;
            ids
        };

        let find_run_config_id =
            store.save_run_config(None, &"find", "find_motifs", HashMap::new()).await?;
        let select_run_config_id = store.save_run_config(
            Some(&[find_run_config_id]),
            &"select",
            "select_motifs",
            HashMap::new(),
        ).await?;

        {
            let mut guard = store.open_external_writer_conn()?;
            let t = guard.transaction()?;

            t.execute(
                "INSERT INTO motifs (id, group_ord, run_config_id, channel_id) VALUES (1, 0, ?, ?)",
                params![find_run_config_id, 0i64],
            )?;

            t.execute(
                "INSERT INTO slices (motif_id, run_config_id, channel_id, resource_id, slice_idx) VALUES (1, ?, ?, ?, ?)",
                params![find_run_config_id, 0i64, resource_ids[0], 10i64],
            )?;
            t.execute(
                "INSERT INTO slices (motif_id, run_config_id, channel_id, resource_id, slice_idx) VALUES (1, ?, ?, ?, ?)",
                params![select_run_config_id, 0i64, resource_ids[0], 30i64],
            )?;

            t.commit()?;
        }

        let resource = store.list_resources_by_ids(&resource_ids).await?.remove(0);

        let find_only = store.list_motifs_slices(find_run_config_id, Some(vec![])).await?;
        let all_runs = store.list_motifs_slices(find_run_config_id, None).await?;
        let both_runs =
            store.list_motifs_slices(find_run_config_id, Some(vec![select_run_config_id, select_run_config_id])).await?;

        let find_only_slices = find_only.get(&resource).expect("resource slices");
        assert_eq!(1, find_only_slices.len());
        assert_eq!(10, find_only_slices[0].slice_idx);

        let expect_two = |map: &HashMap<Resource, Vec<MotifSlice>>| {
            let slices = map.get(&resource).expect("resource slices");
            let mut slice_idxs = slices.iter().map(|s| s.slice_idx).collect::<Vec<_>>();
            slice_idxs.sort();
            assert_eq!(vec![10, 30], slice_idxs);
        };

        expect_two(&all_runs);
        expect_two(&both_runs);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_list_motif_slices_filters_slice_run_config_ids() -> anyhow::Result<()> {
        let mut store = test_store().await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        let resource_ids = {
            let import_id = store.save_import("test".to_string()).await?;
            store.save_label("aaa").await?;

            let ids = store.save_resources_labels(
                &HashMap::from([(
                    "res_a".to_string(),
                    (ResourceStats::new(vec![1, 10]), vec![("aaa".into(), 1.0)]),
                )]),
                import_id,
            ).await?;
            ids
        };

        let find_run_config_id =
            store.save_run_config(None, &"find", "find_motifs", HashMap::new()).await?;
        let select_run_config_id = store.save_run_config(
            Some(&[find_run_config_id]),
            &"select",
            "select_motifs",
            HashMap::new(),
        ).await?;

        {
            let mut guard = store.open_external_writer_conn()?;
            let t = guard.transaction()?;

            t.execute(
                "INSERT INTO motifs (id, group_ord, run_config_id, channel_id) VALUES (2, 0, ?, ?)",
                params![find_run_config_id, 0i64],
            )?;

            t.execute(
                "INSERT INTO slices (motif_id, run_config_id, channel_id, resource_id, slice_idx) VALUES (2, ?, ?, ?, ?)",
                params![find_run_config_id, 0i64, resource_ids[0], 12i64],
            )?;
            t.execute(
                "INSERT INTO slices (motif_id, run_config_id, channel_id, resource_id, slice_idx) VALUES (2, ?, ?, ?, ?)",
                params![select_run_config_id, 0i64, resource_ids[0], 32i64],
            )?;

            t.commit()?;
        }

        let (_, run_config_id, all_slices, _) = store.list_motif_slices(2, None, None, None).await?;
        assert_eq!(find_run_config_id, run_config_id);
        let mut idxs = all_slices.iter().map(|s| s.1).collect::<Vec<_>>();
        idxs.sort();
        assert_eq!(vec![12, 32], idxs);

        let (_, _, find_only, _) = store.list_motif_slices(2, Some(vec![]), None, None).await?;
        assert_eq!(vec![12], find_only.iter().map(|s| s.1).collect::<Vec<_>>());

        let (_, _, filtered, _) =
            store.list_motif_slices(2, Some(vec![select_run_config_id, select_run_config_id]), None, None).await?;
        let mut idxs = filtered.iter().map(|s| s.1).collect::<Vec<_>>();
        idxs.sort();
        assert_eq!(vec![12, 32], idxs);

        Ok(())
    }

    //TODO merge with init_test_store in cli::main_tests
    pub async fn init_test_store(
        store: &Arc<SqlStore>,
    ) -> anyhow::Result<Vec<ResourceId>> {
        let store_guard = store;
        let import_id = store_guard.save_import("test".to_string()).await?;

        let _aaa_id = store_guard.save_label("aaa").await?;
        let _bbb_id = store_guard.save_label("bbb").await?;
        let _ccc_id = store_guard.save_label("ccc").await?;//no resources

        let resource_ids = store_guard.save_resources_labels(
            &HashMap::from([
                (
                    "1111".into(),
                    (
                        ResourceStats::new(vec![1, 100]),
                        vec![("aaa".into(), 0.5), ("bbb".into(), 0.5)],
                    ),
                ),
                (
                    "2222".into(),
                    (ResourceStats::new(vec![1, 100]), vec![("aaa".into(), 1.0)]),
                ),
                (
                    "3333".into(),
                    (ResourceStats::new(vec![1, 100]), vec![("ccc".into(), 1.0)]),
                ),
                ("444".into(), (ResourceStats::new(vec![1, 100]), vec![])),
                ("555".into(), (ResourceStats::new(vec![1, 100]), vec![])),
                ("666".into(), (ResourceStats::new(vec![1, 100]), vec![])),
            ]),
            import_id,
        ).await?;

        debug!("resource_ids={resource_ids:?}");

        store_guard.save_pure_sets(10, false).await?;
        store_guard.save_leaky_data_sets("A", LimitBy::None, 0.6, 0.2, 1.0, false).await?;
        Ok(resource_ids)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_save_run_config() -> anyhow::Result<()> {
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        fs::create_dir_all(&test_dir)?;
        let mut store = test_store_from_path(
            test_dir.join("test_db.db3"),
        ).await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        let run_config_id = store.save_run_config(
            None,
            &"todo",
            "test",
            HashMap::from([
                ("prop1".into(), Val::Int(1)),
                ("prop2".into(), Val::Real(2.0)),
                ("prop3".into(), Val::Str("333".to_string())),
                ("prop4".into(), Val::Bool(false)),
                ("prop5".into(), Val::Bool(true)),
            ]),
        ).await?;

        let next_run_config_id = store.save_run_config(
            Some(&[run_config_id]),
            &"todo",
            "test next",
            HashMap::from([
                ("next_prop1".into(), Val::Int(1)),
                ("next_prop2".into(), Val::Real(2.0)),
                ("next_prop3".into(), Val::Str("zzz".to_string())),
            ]),
        ).await?;

        let loaded_params = store.list_run_config_values(run_config_id).await?;

        assert_eq!(loaded_params.get("prop1").expect("prop1"), &Val::Int(1));
        assert_eq!(loaded_params.get("prop2").expect("prop2"), &Val::Real(2.0));
        assert_eq!(
            loaded_params.get("prop3").expect("prop3"),
            &Val::Str("333".to_string())
        );
        assert_eq!(
            loaded_params.get("prop4").expect("prop4"),
            &Val::Bool(false)
        );
        assert_eq!(loaded_params.get("prop5").expect("prop5"), &Val::Bool(true));

        let netx_loaded_params = store.list_run_config_values(next_run_config_id).await?;

        debug!("netx_loaded_params={netx_loaded_params:?}");

        assert_eq!(
            netx_loaded_params.get("next_prop1").expect("next_prop1"),
            &Val::Int(1)
        );
        assert_eq!(
            netx_loaded_params.get("next_prop2").expect("next_prop2"),
            &Val::Real(2.0)
        );
        assert_eq!(
            netx_loaded_params.get("next_prop3").expect("next_prop3"),
            &Val::Str("zzz".to_string())
        );

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_list_run_configs_page_and_by_ids() -> anyhow::Result<()> {
        let mut store = test_store().await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        let run_config_id_1 = store.save_run_config(None, &"first", "test", HashMap::new()).await?;
        let run_config_id_2 = store.save_run_config(None, &"second", "test", HashMap::new()).await?;
        let run_config_id_3 = store.save_run_config(None, &"third", "test", HashMap::new()).await?;

        let first_page = store.list_run_configs_page(2, 0).await?;
        assert_eq!(vec![run_config_id_3, run_config_id_2], first_page.into_iter().map(|entry| entry.id).collect::<Vec<_>>());

        let second_page = store.list_run_configs_page(2, 2).await?;
        assert_eq!(vec![run_config_id_1], second_page.into_iter().map(|entry| entry.id).collect::<Vec<_>>());

        let loaded_by_ids = store.list_run_configs_by_ids(&[run_config_id_1, run_config_id_3]).await?;
        assert_eq!(vec![run_config_id_3, run_config_id_1], loaded_by_ids.into_iter().map(|entry| entry.id).collect::<Vec<_>>());

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_list_run_configs_filtered_returns_matches_and_ancestors() -> anyhow::Result<()> {
        let mut store = test_store().await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        let root_run_config_id = store.save_run_config(None, &"root", "root", HashMap::new()).await?;
        let select_run_config_id = store.save_run_config(Some(&[root_run_config_id]), &"SelectMotifsRunConfig", "alpha motif selection", HashMap::new()).await?;
        let child_run_config_id = store.save_run_config(Some(&[select_run_config_id]), &"TrainMotifNetRunConfig", "child", HashMap::new()).await?;
        let _other_run_config_id = store.save_run_config(None, &"OtherRunConfig", "beta", HashMap::new()).await?;

        let by_id = store.list_run_configs_filtered(RunConfigFilter {
            id: Some(select_run_config_id),
            ..RunConfigFilter::default()
        }).await?;
        assert_eq!(vec![select_run_config_id, root_run_config_id], by_id.into_iter().map(|entry| entry.id).collect::<Vec<_>>());

        let by_type = store.list_run_configs_filtered(RunConfigFilter {
            run_config_type: Some("TrainMotifNetRunConfig".to_string()),
            ..RunConfigFilter::default()
        }).await?;
        assert_eq!(vec![child_run_config_id, select_run_config_id, root_run_config_id], by_type.into_iter().map(|entry| entry.id).collect::<Vec<_>>());

        let by_descr = store.list_run_configs_filtered(RunConfigFilter {
            descr_substr: Some("alpha".to_string()),
            ..RunConfigFilter::default()
        }).await?;
        assert_eq!(vec![select_run_config_id, root_run_config_id], by_descr.into_iter().map(|entry| entry.id).collect::<Vec<_>>());

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_conv_cond() {
        let input = vec![
            ("aaa".into(), SetCond::Lt(0.4)),
            ("bbb".into(), SetCond::Gt(0.1)),
        ];

        let conds = super::conds_to_q(&input);
        assert_eq!("(l.label='aaa' and p<?) or (l.label='bbb' and p>?)", conds);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_get_prev_run_config_id() -> anyhow::Result<()> {
        let mut store = test_store().await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        let input_run_config_id = store.save_run_config(None, &"aaa", "test 0", HashMap::new()).await?;
        let run_config_id =
            store.save_run_config(Some(&[input_run_config_id]), &"aaa", "test 1", HashMap::new()).await?;

        assert_eq!(store.get_input_run_config_ids(run_config_id, None).await?[0], input_run_config_id);

        let input_run_config_id_0 = store.save_run_config(None, &"aaa", "test 0", HashMap::new()).await?;
        let input_run_config_id_1 = store.save_run_config(
            Some(&[input_run_config_id_0]),
            &"bbb",
            "test 1",
            HashMap::new(),
        ).await?;
        let run_config_id = store.save_run_config(
            Some(&[input_run_config_id_1]),
            &"ccc",
            "test 1",
            HashMap::new(),
        ).await?;

        assert_eq!(
            store.get_input_run_config_ids(run_config_id, Some(&"aaa")).await?[0],
            input_run_config_id_0
        );

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_list_input_run_config_ids_reads_large_sets_in_chunks() -> anyhow::Result<()> {
        let mut store = test_store().await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        let input_run_config_id = store.save_run_config(None, &"root", "root", HashMap::new()).await?;
        let mut run_config_ids = Vec::with_capacity(1001);
        for idx in 0..1001 {
            run_config_ids.push(
                store
                    .save_run_config(
                        Some(&[input_run_config_id]),
                        &format!("task-{idx}"),
                        "chunked",
                        HashMap::new(),
                    )
                    .await?,
            );
        }

        let mut input_pairs = store.list_input_run_config_ids(&run_config_ids).await?;
        input_pairs.sort();

        let mut expected_pairs = run_config_ids
            .into_iter()
            .map(|run_config_id| (run_config_id, input_run_config_id))
            .collect::<Vec<_>>();
        expected_pairs.sort();

        assert_eq!(expected_pairs, input_pairs);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_list_run_config_ids_with_outputs_reads_large_sets_in_chunks() -> anyhow::Result<()> {
        let mut store = test_store().await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        let mut parent_run_config_ids = Vec::with_capacity(1001);
        for idx in 0..1001 {
            let parent_run_config_id = store.save_run_config(None, &format!("parent-{idx}"), "parent", HashMap::new()).await?;
            if idx % 2 == 0 {
                store.save_run_config(Some(&[parent_run_config_id]), &format!("child-{idx}"), "child", HashMap::new()).await?;
            }
            parent_run_config_ids.push(parent_run_config_id);
        }

        let mut run_config_ids_with_outputs = store.list_run_config_ids_with_outputs(&parent_run_config_ids).await?;
        run_config_ids_with_outputs.sort();

        let mut expected_run_config_ids = parent_run_config_ids
            .into_iter()
            .enumerate()
            .filter_map(|(idx, run_config_id)| if idx % 2 == 0 { Some(run_config_id) } else { None })
            .collect::<Vec<_>>();
        expected_run_config_ids.sort();

        assert_eq!(expected_run_config_ids, run_config_ids_with_outputs);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_list_resources_by_file_stem() -> anyhow::Result<()> {
        let mut store = test_store().await?;
        {
            let import_id = store.save_import("test".to_string()).await?;

            store.save_resources_labels(
                &HashMap::from([
                    (
                        "resource_aaa_0".to_string(),
                        (ResourceStats::new(vec![1, 100]), vec![]),
                    ),
                    (
                        "resource_aaa_1".to_string(),
                        (ResourceStats::new(vec![1, 100]), vec![]),
                    ),
                    (
                        "resource_bbb_0".to_string(),
                        (ResourceStats::new(vec![1, 100]), vec![]),
                    ),
                    (
                        "resource_bbb_1".to_string(),
                        (ResourceStats::new(vec![1, 100]), vec![]),
                    ),
                ]),
                1,
            ).await?;
        }

        assert_eq!(
            vec!["resource_aaa_0".to_string(), "resource_aaa_1".to_string()],
            store
                .list_resources_by_file_stems(&vec!["%aaa%"]).await?
                .into_iter()
                .map(|r| r.file_stem().to_string())
                .collect::<Vec<String>>()
        );

        assert_eq!(
            vec!["resource_bbb_0".to_string(), "resource_bbb_1".to_string()],
            store
                .list_resources_by_file_stems(&vec!["%bbb%"]).await?
                .into_iter()
                .map(|r| r.file_stem().to_string())
                .collect::<Vec<String>>()
        );

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_resources_by_label_filters_by_min_probability() -> anyhow::Result<()> {
        let mut store = test_store().await?;
        let import_id = store.save_import("test".to_string()).await?;
        store.save_label("aaa").await?;
        store.save_label("bbb").await?;

        store.save_resources_labels(
            &HashMap::from([
                (
                    "aaa_only".to_string(),
                    (ResourceStats::new(vec![1, 100]), vec![("aaa".into(), 1.0)]),
                ),
                (
                    "mostly_aaa".to_string(),
                    (
                        ResourceStats::new(vec![1, 100]),
                        vec![("aaa".into(), 0.7), ("bbb".into(), 0.3)],
                    ),
                ),
                (
                    "mostly_bbb".to_string(),
                    (
                        ResourceStats::new(vec![1, 100]),
                        vec![("aaa".into(), 0.4), ("bbb".into(), 0.6)],
                    ),
                ),
            ]),
            import_id,
        ).await?;

        let mut resources_by_label = store.resources_by_label(0.6).await?;
        let mut aaa_resources = resources_by_label
            .remove("aaa")
            .unwrap()
            .into_iter()
            .map(|resource| resource.file_stem().to_string())
            .collect::<Vec<_>>();
        aaa_resources.sort();
        let mut bbb_resources = resources_by_label
            .remove("bbb")
            .unwrap()
            .into_iter()
            .map(|resource| resource.file_stem().to_string())
            .collect::<Vec<_>>();
        bbb_resources.sort();

        assert_eq!(vec!["aaa_only".to_string(), "mostly_aaa".to_string()], aaa_resources);
        assert_eq!(vec!["mostly_bbb".to_string()], bbb_resources);

        let resources_by_label = store.resources_by_label(1.0).await?;
        assert_eq!(
            vec!["aaa_only".to_string()],
            resources_by_label["aaa"]
                .iter()
                .map(|resource| resource.file_stem().to_string())
                .collect::<Vec<_>>()
        );
        assert!(resources_by_label["bbb"].is_empty());

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_list_random_slices_respects_constraints() -> anyhow::Result<()> {
        let mut store = test_store().await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        create_resources(&store, &[("res_a", 2048), ("res_b", 2048)]).await?;

        let run_config_id = store.save_run_config(None, &"find_motifs", "random test", HashMap::new()).await?;
        let channel_id = 3i64;

        let res_a = fetch_resource_id(&store, "res_a")?;
        let res_b = fetch_resource_id(&store, "res_b")?;

        let motif_id = insert_motif_with_slices(
            &store,
            run_config_id,
            channel_id,
            0,
            &[(res_a, 100), (res_a, 320)],
        )?;
        insert_motif_with_slices(
            &store,
            run_config_id,
            channel_id,
            1,
            &[(res_b, 50), (res_b, 400)],
        )?;

        const WIDTH: u32 = 40;
        let (channel, run_config_id, random_slices, resources_map) =
            store.list_random_slices_out_of_motif(motif_id, WIDTH, None, Arc::new(|length: u32| length), Some(2usize)).await?;

        assert_eq!(channel_id, channel);
        assert_eq!(random_slices.len(), 2);

        let mut per_resource: HashMap<ResourceId, Vec<u32>> = HashMap::new();
        for slice in &random_slices {
            let resource = resources_map
                .get(&slice.0)
                .expect("resource metadata missing");
            assert!(slice.1 > 0);
            assert!(slice.1 + WIDTH < resource.length());
            per_resource.entry(slice.0).or_default().push(slice.1);
        }

        let target_positions = HashMap::from([(res_a, vec![100u32, 320u32])]);
        for (resource_id, mut positions) in per_resource {
            positions.sort_unstable();
            for window in positions.windows(2) {
                assert!(window[0] + WIDTH < window[1]);
            }

            if let Some(targets) = target_positions.get(&resource_id) {
                for pos in &positions {
                    for target in targets {
                        assert!(*pos + WIDTH < *target || *target + WIDTH < *pos);
                    }
                }
            }
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_list_random_slices_respects_min_distance() -> anyhow::Result<()> {
        let mut store = test_store().await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        create_resources(&store, &[("res_main", 4096)]).await?;

        let run_config_id =
            store.save_run_config(None, &"find_motifs", "random spacing", HashMap::new()).await?;
        let channel_id = 5i64;
        let res_main = fetch_resource_id(&store, "res_main")?;

        let motif_id = insert_motif_with_slices(
            &store,
            run_config_id,
            channel_id,
            0,
            &[(res_main, 200), (res_main, 1200)],
        )?;

        const WIDTH: u32 = 50;
        const MIN_DISTANCE: u32 = 300;
        let (channel, run_config_id, random_slices, resources_map) =
            store.list_random_slices_out_of_motif(motif_id, WIDTH, Some(MIN_DISTANCE), Arc::new(|length: u32| length), Some(2usize)).await?;

        assert_eq!(channel_id, channel);
        assert_eq!(random_slices.len(), 2);

        let mut positions = random_slices
            .iter()
            .filter(|slice| slice.0 == res_main)
            .map(|slice| slice.1)
            .collect::<Vec<u32>>();
        positions.sort_unstable();

        assert_eq!(positions.len(), 2);
        assert!(positions[0] + MIN_DISTANCE < positions[1]);

        let resource = resources_map.get(&res_main).expect("resource missing");
        for pos in positions {
            assert!(pos > 0);
            assert!(pos + WIDTH < resource.length());
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_list_random_slices_uses_input_len_adjuster() -> anyhow::Result<()> {
        let mut store = test_store().await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        create_resources(&store, &[("res_main", 2000)]).await?;

        let run_config_id = store.save_run_config(None, &"find_motifs", "length adjust", HashMap::new()).await?;
        let channel_id = 6i64;
        let res_main = fetch_resource_id(&store, "res_main")?;

        let motif_id = insert_motif_with_slices(
            &store,
            run_config_id,
            channel_id,
            0,
            &[(res_main, 100), (res_main, 250)],
        )?;

        const WIDTH: u32 = 40;
        let input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync> = Arc::new(|length: u32| (length - 100) / 5);
        let (channel, _, random_slices, resources_map) =
            store.list_random_slices_out_of_motif(motif_id, WIDTH, None, Arc::clone(&input_len_adjuster), None).await?;

        assert_eq!(channel_id, channel);
        assert_eq!(random_slices.len(), 2);

        let resource = resources_map.get(&res_main).expect("resource missing");
        let adjusted_len = input_len_adjuster(resource.length());
        for slice in random_slices {
            assert!(slice.1 > 0);
            assert!(slice.1 + WIDTH < adjusted_len);
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 32)]
    async fn heavy_list_random_slices() -> anyhow::Result<()> {
        let mut store = test_store().await?;

        create_resources(&store, (0..100).map(|i| (&*format!("res_{i}").leak(), 2000u32)).collect::<Vec<_>>().as_slice()).await?;

        let run_config_id = store.save_run_config(None, &"find_motifs", "length adjust", HashMap::new()).await?;
        let channel_id = 6i64;
        let res_0 = fetch_resource_id(&store, "res_0")?;
        let res_10 = fetch_resource_id(&store, "res_1")?;

        let motif_id = insert_motif_with_slices(
            &store,
            run_config_id,
            channel_id,
            0,
            &[
                    (res_0, 100), (res_0, 500), (res_0, 1000),
                    (res_10, 100), (res_10, 500), (res_10, 1000),
            ],
        )?;

        const WIDTH: u32 = 200;
        let input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync> = Arc::new(|length: u32| length);
        let n = Instant::now();
        let (channel, _, random_slices, resources_map) = store.list_random_slices_out_of_motif(motif_id, WIDTH, None, Arc::clone(&input_len_adjuster), Some(100)).await?;

        debug!("random_slices.size={} elapsed={}", random_slices.len(), n.elapsed().as_millis());
        assert_eq!(100, random_slices.len());

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_list_random_slices_falls_back_to_other_resources() -> anyhow::Result<()> {
        let mut store = test_store().await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        create_resources(&store, &[("res_main", 220), ("res_other", 500)]).await?;

        let run_config_id = store.save_run_config(None, &"find_motifs", "fallback negatives", HashMap::new()).await?;
        let channel_id = 7i64;
        let res_main = fetch_resource_id(&store, "res_main")?;
        let res_other = fetch_resource_id(&store, "res_other")?;

        let motif_id = insert_motif_with_slices(
            &store,
            run_config_id,
            channel_id,
            0,
            &[(res_main, 40), (res_main, 100), (res_main, 160)],
        )?;
        insert_motif_with_slices(
            &store,
            run_config_id,
            channel_id,
            1,
            &[(res_other, 60), (res_other, 260)],
        )?;

        const WIDTH: u32 = 50;
        let (channel, returned_run_config_id, random_slices, resources_map) =
            store.list_random_slices_out_of_motif(motif_id, WIDTH, None, Arc::new(|length: u32| length), None).await?;

        assert_eq!(channel_id, channel);
        assert_eq!(run_config_id, returned_run_config_id);
        assert_eq!(random_slices.len(), 3);
        assert!(random_slices.iter().all(|slice| slice.0 == res_other), "random_slices={random_slices:?}");
        assert_eq!(resources_map.len(), 1);
        assert!(!resources_map.contains_key(&res_main));

        let resource = resources_map.get(&res_other).expect("resource missing");
        let mut positions = random_slices.iter().map(|slice| slice.1).collect::<Vec<u32>>();
        positions.sort_unstable();

        for pos in positions {
            assert!(pos > 0);
            assert!(pos + WIDTH < resource.length());
        }

        Ok(())
    }

    async fn create_resources(store: &Arc<SqlStore>, specs: &[(&str, u32)]) -> anyhow::Result<()> {
        let import_id = store.save_import("test resources".into()).await?;

        let resources = specs.iter().fold(
            HashMap::new(),
            |mut acc: HashMap<String, (ResourceStats, Vec<(String, f32)>)>, (name, length)| {
                acc.insert(
                    (*name).to_string(),
                    (ResourceStats::new(vec![1, *length]), vec![]),
                );
                acc
            },
        );

        store.save_resources_labels(&resources, import_id).await?;

        Ok(())
    }

    fn fetch_resource_id(store: &Arc<SqlStore>, key: &str) -> anyhow::Result<ResourceId> {
        let guard = store.open_external_writer_conn()?;
        let id = guard.query_row("SELECT id FROM resources WHERE key=?", [key], |row| {
            row.get::<usize, ResourceId>(0)
        })?;
        Ok(id)
    }

    fn insert_motif_with_slices(
        store: &Arc<SqlStore>,
        run_config_id: RunConfigId,
        channel_id: i64,
        group_ord: i64,
        slices: &[(ResourceId, i64)],
    ) -> anyhow::Result<u32> {
        assert!(slices.len() > 0, "insert_motif_with_slices: no slices");

        let mut guard = store.open_external_writer_conn()?;
        let t = guard.transaction()?;


        t.execute(
            "INSERT INTO motifs (group_ord, run_config_id, channel_id) VALUES (?, ?, ?)",
            params![group_ord, run_config_id as i64, channel_id],
        )?;

        let motif_id = t.query_row("SELECT last_insert_rowid()", [], |row| {
            row.get::<usize, u32>(0)
        })?;

        for (resource_id, slice_idx) in slices {
            t.execute(
                "INSERT INTO slices (run_config_id, channel_id, motif_id, resource_id, slice_idx) VALUES (?, ?, ?, ?, ?)",
                params![run_config_id as i64, channel_id, motif_id as i64, *resource_id as i64, *slice_idx],
            )?;
        }

        t.commit()?;

        Ok(motif_id)
    }
}


#[cfg(test)]
mod fs_store_tests {
    use crate::fs_store::{ChunkedSlicesReader, FileMotifTrackerStore};
    use crate::motif_tracker_store::{filter_groups_by_slices_count, IdSlice, SlicesMatch};
    use thoosa_store::{SqlStore, Store};
    use log::LevelFilter::Trace;
    use log::{debug, error};
    use parking_lot::{Mutex, RwLock};
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use thoosa_shared::test_utils::init_test_logging;
    use thoosa_shared::{TensorIdSliceIdx, ThoError, cdt_for_dir, group_slices};
    use thoosa_shared::run_configs::RUN_ID;
    use super::MotifsStore;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_single_part() -> anyhow::Result<()> {
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        debug!("test_single_part: test_dir={test_dir:?}");
        fs::create_dir_all(&test_dir)?;

        let mut store = super::store_tests::test_store_from_path(test_dir.join("main_store.db3")).await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        let resource_ids = super::store_tests::init_test_store(&store).await?;

        let run_config_id = store.save_run_config(None, &"todo", "test", HashMap::new()).await?;

        let mut file_store = FileMotifTrackerStore::new(test_dir, run_config_id, 3)?;

        file_store.save_slices(vec![
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 100), TensorIdSliceIdx(resource_ids[0], 200), 0.1),
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 200), TensorIdSliceIdx(resource_ids[0], 300), 0.2),
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 100), TensorIdSliceIdx(resource_ids[0], 700), 0.3),
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 700), TensorIdSliceIdx(resource_ids[0], 800), 0.4),
        ])?;

        file_store.load_slices_into_groups(store.as_ref(), None, None).await?;

        let resources = store.list_resources_by_ids(&resource_ids).await?;
        let map = store.list_motifs_slices(run_config_id, None).await?;

        let motifs_slices = map.get(&resources[0]).expect("resource 0");
        let mut slise_idxs = motifs_slices.into_iter().map(|s| s.slice_idx).collect::<Vec<_>>();
        slise_idxs.sort();

        assert_eq!(vec![100, 200, 300, 700, 800], slise_idxs);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_two_motifs() -> anyhow::Result<()> {
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        debug!("test_single_part: test_dir={test_dir:?}");
        fs::create_dir_all(&test_dir)?;

        let mut store = super::store_tests::test_store_from_path(test_dir.join("main_store.db3")).await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        let resource_ids = super::store_tests::init_test_store(&store).await?;

        let run_config_id = store.save_run_config(None, &"todo", "test", HashMap::new()).await?;

        let mut file_store = FileMotifTrackerStore::new(test_dir, run_config_id, 3)?;

        file_store.save_slices(vec![
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 100), TensorIdSliceIdx(resource_ids[1], 200), 0.1),
            SlicesMatch(TensorIdSliceIdx(resource_ids[1], 200), TensorIdSliceIdx(resource_ids[1], 700), 0.3),
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 300), TensorIdSliceIdx(resource_ids[1], 400), 0.2),
            SlicesMatch(TensorIdSliceIdx(resource_ids[1], 400), TensorIdSliceIdx(resource_ids[0], 800), 0.4),
        ])?;

        file_store.load_slices_into_groups(store.as_ref(), None, None).await?;

        let resources = store.list_resources_by_ids(&resource_ids).await?;
        let map = store.list_motifs_slices(run_config_id, None).await?;

        let motifs_slices = map.get(&resources[0]).expect("resource 0");
        let resoruce_0_motifs = motifs_slices.iter().map(|ms| ms.motif_id).collect::<HashSet<_>>();
        assert_eq!(resoruce_0_motifs.len(), 2);
        let mut slise_idxs = motifs_slices.into_iter().map(|s| s.slice_idx).collect::<Vec<_>>();
        slise_idxs.sort();

        assert_eq!(vec![100, 300, 800], slise_idxs);

        let motifs_slices = map.get(&resources[1]).expect("resource 0");
        let resoruce_1_motifs = motifs_slices.iter().map(|ms| ms.motif_id).collect::<HashSet<_>>();
        assert_eq!(resoruce_1_motifs.len(), 2);
        let mut slise_idxs = motifs_slices.into_iter().map(|s| s.slice_idx).collect::<Vec<_>>();
        slise_idxs.sort();

        assert_eq!(vec![200, 400, 700], slise_idxs);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_single_part_with_resources() -> anyhow::Result<()> {
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        debug!("test_single_part: test_dir={test_dir:?}");
        fs::create_dir_all(&test_dir)?;

        let mut store = super::store_tests::test_store_from_path(test_dir.join("main_store.db3")).await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        let resource_ids = super::store_tests::init_test_store(&store).await?;

        let run_config_id = store.save_run_config(None, &"todo", "test", HashMap::new()).await?;

        let mut file_store = FileMotifTrackerStore::new(test_dir, run_config_id, 3)?;

        file_store.save_slices(vec![
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 100), TensorIdSliceIdx(resource_ids[1], 200), 0.1),
            SlicesMatch(TensorIdSliceIdx(resource_ids[1], 200), TensorIdSliceIdx(resource_ids[0], 300), 0.2),
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 100), TensorIdSliceIdx(resource_ids[1], 700), 0.3),
            SlicesMatch(TensorIdSliceIdx(resource_ids[1], 700), TensorIdSliceIdx(resource_ids[0], 800), 0.4),
        ])?;

        file_store.load_slices_into_groups(store.as_ref(), None, None).await?;

        let resources = store.list_resources_by_ids(&resource_ids).await?;
        let map = store.list_motifs_slices(run_config_id, None).await?;

        let motifs_slices = map.get(&resources[0]).expect("resource 0");
        let mut slise_idxs_0 = motifs_slices.into_iter().map(|s| s.slice_idx).collect::<Vec<_>>();
        slise_idxs_0.sort();

        assert_eq!(vec![100, 300, 800], slise_idxs_0);

        let motifs_slices = map.get(&resources[1]).expect("resource 0");
        let mut slise_idxs_1 = motifs_slices.into_iter().map(|s| s.slice_idx).collect::<Vec<_>>();
        slise_idxs_1.sort();

        assert_eq!(vec![200, 700], slise_idxs_1);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_two_parts() -> anyhow::Result<()> {
        // log::set_max_level(Trace);
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        debug!("test_single_part: test_dir={test_dir:?}");
        fs::create_dir_all(&test_dir)?;

        let mut store = super::store_tests::test_store_from_path(test_dir.join("main_store.db3")).await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        let resource_ids = super::store_tests::init_test_store(&store).await?;

        let run_config_id = store.save_run_config(None, &"todo", "test", HashMap::new()).await?;

        let mut file_store = FileMotifTrackerStore::new(test_dir, run_config_id, 3)?;

        file_store.save_slices(vec![
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 100), TensorIdSliceIdx(resource_ids[0], 200), 0.1),
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 200), TensorIdSliceIdx(resource_ids[0], 300), 0.2),
        ])?;

        file_store.save_slices(vec![
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 100), TensorIdSliceIdx(resource_ids[0], 700), 0.3),
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 700), TensorIdSliceIdx(resource_ids[0], 800), 0.4),
        ])?;

        file_store.load_slices_into_groups(store.as_ref(), None, None).await?;

        let resources = store.list_resources_by_ids(&resource_ids).await?;
        let map = store.list_motifs_slices(run_config_id, None).await?;

        let motifs_slices = map.get(&resources[0]).expect("resource 0");
        let mut slise_idxs = motifs_slices.into_iter().map(|s| s.slice_idx).collect::<Vec<_>>();
        slise_idxs.sort();

        assert_eq!(vec![100, 200, 300, 700, 800], slise_idxs);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_no_matches() -> anyhow::Result<()> {
        // log::set_max_level(Trace);
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        debug!("test_single_part: test_dir={test_dir:?}");
        fs::create_dir_all(&test_dir)?;

        let mut store = super::store_tests::test_store_from_path(test_dir.join("main_store.db3")).await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        let resource_ids = super::store_tests::init_test_store(&store).await?;

        let run_config_id = store.save_run_config(None, &"todo", "test", HashMap::new()).await?;

        let mut file_store = FileMotifTrackerStore::new(test_dir, run_config_id, 3)?;

        file_store.load_slices_into_groups(store.as_ref(), None, None).await?;

        let map = store.list_motifs_slices(run_config_id, None).await?;

        assert_eq!(0, map.len());

        Ok(())
    }

    /*
     not a test - util to debug failing deserialization
    #[test]
     fn test_z() -> Result<(), ThoosaError>{

         let mut slice_id = 0;
         let run_config_id=28;
         let channel=19;

         log::set_max_level(Trace);
         let groups:HashMap<u16, HashSet<IdSlice>> = group_slices(

            // ChunkedSlicesReader::<SlicesMatch>::new(&PathBuf::from(format!("/tmp/ws_test/tmp/{run_config_id}_{channel}_slices.zstd").as_str()))?.flat_map(|slices|{
            ChunkedSlicesReader::<SlicesMatch>::new(&PathBuf::from("/tmp/2024-12-12T12_54_47.820_01_00_3468226743/tmp/1_0_slices.zstd"))?.flat_map(|slices|{
                 slices.unwrap().into_iter().map(move |slices:SlicesMatch|{
                     slice_id+=1;
                     (
                         IdSlice(slice_id, slices.0.0, slices.0.1),
                         IdSlice(slice_id, slices.1.0, slices.1.1)
                     )
                 })
             })
         );

         debug!("{run_config_id}_{channel}_slices slice_id={slice_id} groups.len={}", groups.len());
         let test_dir = Path::new("/tmp").join(cdt_for_dir());
         fs::create_dir_all(&test_dir)?;
         let sql_file = test_dir.join("test_store.db3");
         fs::copy("/tmp/ws_test/main_db.db3", &sql_file)?;

         debug!("sql_file={sql_file:?}");
         let store = Arc::new(SqlStore::from_path(sql_file)?);
         // let store = Arc::new(SqlStore::in_mem()?);

         match store.save_groups_as_motifs(
             28, 19,
             filter_groups_by_slices_count(None, groups)
         ) {
             Ok(v) => Ok(v),
             Err(e) => {
                 Err(e)
             }
         }?;

         Ok(())

     }*/
}
