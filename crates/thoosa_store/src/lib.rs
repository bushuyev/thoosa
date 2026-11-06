#![feature(iter_array_chunks)]
#![allow(deprecated)]
extern crate core;

mod tokio_rusqlite;

use anyhow::Context as _;
pub use rusqlite;

use std::array::IntoIter;
use std::backtrace::Backtrace;
use std::cmp::{max, min};
use std::collections::HashMap;
use std::fmt;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use async_trait::async_trait;

use log::{debug, error, trace, warn};
use mockall::mock;
use mockall::predicate::eq;
use rand::seq::SliceRandom;
use rusqlite::types::Value;
use rusqlite::{Connection, Error, OpenFlags, Transaction, params};
use rusqlite::trace::TraceEvent;
use serde::{Deserialize, Serialize};
use thoosa_shared::resource::{Resource, ResourceId, ResourceStore};
use thoosa_shared::run_configs::{RunConfigId, RunId};
use thoosa_shared::val::Val;
use thoosa_shared::val::Val::UInt;
use thoosa_shared::*;

#[derive(Debug)]
pub enum SetCond {
    Eq(f32),
    Gt(f32),
    Lt(f32),
}

impl SetCond {
    fn p(&self) -> &'static str {
        match self {
            SetCond::Eq(_v) => "p=",
            SetCond::Gt(_v) => "p>",
            SetCond::Lt(_v) => "p<",
        }
    }

    pub fn v(&self) -> f32 {
        match self {
            SetCond::Eq(v) => *v,
            SetCond::Gt(v) => *v,
            SetCond::Lt(v) => *v,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunConfigEntry {
    pub id: RunConfigId,
    pub run_config_type: String,
    pub descr: String,
    pub run_on: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunConfigRow {
    pub id: RunConfigId,
    pub r#type: String,
    pub descr: String,
    pub run_on: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainedNetEntry {
    pub id: u32,
    pub net_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainedNet {
    pub id: u32,
    pub run_config_id: RunConfigId,
    pub net_path: PathBuf,
    pub loss: f32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunConfigFilter {
    pub id: Option<RunConfigId>,
    pub run_config_type: Option<String>,
    pub descr_substr: Option<String>,
    pub run_on_from: Option<String>,
    pub run_on_to: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunMotifEntry {
    pub motif_id: u32,
    pub channel_id: usize,
    pub score: Option<f64>,
    pub source_run_config_id: RunConfigId,
    pub slices_cnt: usize,
    pub resources_cnt: usize,
}
#[derive(Debug)]
pub enum TrainedNetsBy {
    RunConfigIds(Vec<RunConfigId>),
    TrainNetId(Vec<u32>)
}

impl AsRef<Vec<u32>> for TrainedNetsBy {
    fn as_ref(&self) -> &Vec<u32> {
        match self{
            TrainedNetsBy::RunConfigIds(v) => v,
            TrainedNetsBy::TrainNetId(v) => v
        }
    }
}

/// Explicit SQLite import capability. Ordinary Store consumers use the actor-backed
/// operations; only schema-aware import code should open a separate writer.
pub trait ExternalSqlWriter: Debug + Send + Sync {
    fn open_external_writer_conn(&self) -> anyhow::Result<Connection>;
}

impl ExternalSqlWriter for SqlStore {
    fn open_external_writer_conn(&self) -> anyhow::Result<Connection> {
        let mut conn = Connection::open_with_flags(&self.conn_target, self.conn_flags)?;
        create_tables(&mut conn)?;
        Self::init_external_writer_conn(&mut conn)?;
        Ok(conn)
    }

}

#[async_trait]
pub trait Store: ResourceStore + Debug + Sync + Send {
    /// Bind this store session before saving configurations. Independent stores
    /// may execute different runs concurrently without process-global identity.
    fn bind_run(&self, _run_id: RunId) -> anyhow::Result<()> { anyhow::bail!("store does not support run sessions") }

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
}

mock! {
    pub Store {}

    #[async_trait]
    impl Store for Store {
        async fn save_import(&self, descr: String) -> anyhow::Result<u32>;
        async fn save_run(&self, descr: String) -> anyhow::Result<RunId>;
        async fn save_run_config<'a, 'b, 'c>(
            &self,
            input_run_config_id: Option<&'a [RunConfigId]>,
            run_config_type: &'b str,
            descr: &'c str,
            properties_vals: HashMap<String, Val>,
        ) -> anyhow::Result<RunConfigId>;
        async fn get_input_run_config_ids<'a>(&self,run_config_id: RunConfigId,input_type: Option<&'a str>,) -> anyhow::Result<Vec<RunConfigId>>;
        async fn list_input_run_config_ids(&self, ids: &[RunConfigId]) -> anyhow::Result<Vec<(RunConfigId, RunConfigId)>>;
        async fn list_run_config_ids_with_outputs(&self, input_run_config_ids: &[RunConfigId]) -> anyhow::Result<Vec<RunConfigId>>;
        async fn list_output_run_config_ids(&self, input_run_config_id: RunConfigId) -> anyhow::Result<Vec<RunConfigId>>;

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
        async fn save_trained_net(
            &self,
            run_config_id: RunConfigId,
            path: PathBuf,
            loss: f32,
            input_nets:Option<Vec<u32>>
        ) -> anyhow::Result<u32>;
        async fn list_trained_nets(
            &self,
            run_config_ids: TrainedNetsBy,
            loss: Option<f32>,
            limit: Option<usize>,
        ) -> anyhow::Result<Vec<TrainedNet>>;
        async fn list_trained_net_entries(&self) -> anyhow::Result<Vec<TrainedNetEntry>>;
        async fn list_label_trained_nets_before(&self, run_on_before: String) -> anyhow::Result<Vec<TrainedNetEntry>>;
        async fn list_trained_nets_loss_gt(&self, loss: f32, net_type:&'static str) -> anyhow::Result<Vec<TrainedNetEntry>>;
        async fn list_unused_motif_trained_nets(&self) -> anyhow::Result<Vec<TrainedNetEntry>>;
        async fn delete_trained_nets(&self, net_ids: Vec<u32>) -> anyhow::Result<usize>;
        async fn update_run_config_run_on(&self, run_config_id: RunConfigId, run_on: String) -> anyhow::Result<()>;
        async fn list_input_nets(&self, net_id: RunConfigId) -> anyhow::Result<Vec<u32>>;
        async fn run_config_descr(&self, corr_run_config_id: RunConfigId) -> anyhow::Result<RunConfigRow>;
    }

    #[async_trait]
    impl ResourceStore for Store {
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
            limit:u32
        ) -> anyhow::Result<Vec<Resource>>;

        async fn list_random_resources(
            &self,
            limit: Option<u32>
        ) -> anyhow::Result<Vec<Resource>>;

    }
}

impl fmt::Debug for MockStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MockStore").finish()
    }
}

pub enum LimitBy {
    Count(usize),
    Length(usize),
    None
}

//TODO move all prepared statements into init
#[derive(Debug)]
pub struct SqlStore {
    run_id: std::sync::OnceLock<RunId>,
    writer: tokio_rusqlite::Connection,
    readers: Arc<Vec<tokio_rusqlite::Connection>>,
    conn_target: String,
    conn_flags: OpenFlags,
    rr: AtomicUsize,
}

const SQL_STORE_READERS_CNT: usize = 8;
static IN_MEM_DB_COUNTER: AtomicUsize = AtomicUsize::new(0);
pub const TRAIN_LABEL_NETS_RUN_CONFIG_TYPE: &str = "TrainLabelNetRunConfig";
pub const TRAIN_FINAL_NETS_RUN_CONFIG_TYPE: &str = "TrainFinalNetRunConfig";
pub const TRAIN_MOTIFS_NETS_RUN_CONFIG_TYPE: &str = "TrainMotifNetRunConfig";

#[derive(Debug)]
pub struct RandomSliceResourceCtx {
    resource: Resource,
    adjusted_len: i64,
    target_positions: Vec<i64>,
}

#[async_trait]
impl ResourceStore for SqlStore {
    async fn list_resource_set_resources_by_descr(
        &self,
        descr: &str,
    ) -> anyhow::Result<Vec<Resource>> {
        let descr = descr.to_string();
        let descr_for_err = descr.clone();
        self.with_reader_conn("list_resource_set_resources_by_descr",  move|conn| {
            let r = Self::list_resource_set_resources(
                conn.query_row("SELECT id FROM resource_set WHERE descr=?", [descr.as_str()], |row| row.get(0))?,
                conn,
            )?;
            debug!("list_resource_set_resources_by_descr: descr={descr} r={r:?}");
            Ok(r)
        })
        .await.with_context(|| format!("list resource set with description {descr_for_err:?}"))
    }
    async fn list_resources_by_file_stems<'a>(
        &self,
        file_stems: &'a [&'a str],
    ) -> anyhow::Result<Vec<Resource>> {
        let file_stems = file_stems.iter().map(|v| (*v).to_string()).collect::<Vec<String>>();
        self.with_reader_conn("list_resources_by_file_stems", move |conn| {
            let (mut q, params) = file_stems.into_iter().fold(
                (vec!["SELECT id, import_id, key, length FROM resources WHERE "], vec![]),
                |mut a, v| {
                    a.0.push(" key LIKE ?");
                    a.0.push(" OR");
                    a.1.push(Value::Text(v));
                    a
                },
            );

            q.pop();
            q.push(" ORDER BY key");

            let resources = {
                let mut stmt = conn.prepare(q.into_iter().collect::<String>().as_str())?;
                debug!("list_resources_by_file_stems: stmt={stmt:?}");
                stmt.query_map(rusqlite::params_from_iter(params.into_iter()), |row| {
                    Ok(Resource::new(
                        row.get::<usize, u32>(0)?,
                        row.get::<usize, u32>(1)?,
                        row.get::<usize, String>(2)?,
                        row.get::<usize, u32>(3)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?
            };

            Ok(resources)
        })
        .await
    }

    async fn list_resource_set_group_sets(
        &self,
        group_descr: &str,
    ) -> anyhow::Result<Vec<(usize, String)>> {
        let group_descr = group_descr.to_string();
        self.with_reader_conn("list_resource_set_group_sets", move |conn| {
            let group_sets = {
                let mut stmt = conn.prepare("SELECT rs.id, rs.descr FROM resource_set rs LEFT JOIN resource_set_group rsg ON rsg.id=rs.group_id WHERE rsg.descr=?")?;

                stmt.query_map([group_descr.as_str()], |row| {
                    Ok((
                        row.get::<usize, usize>(0)?,
                        row.get::<usize, String>(1)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?
            };

            debug!("list_resource_set_group_sets: group_descr={group_descr:?} group_sets={group_sets:?}");

            Ok(group_sets)
        })
        .await
    }

    ///returns Vec<(label, resource_set_descr)>
    async fn list_group_labeled_resources_sets(
        &self,
        group_descr: &str,
    ) -> anyhow::Result<Vec<(String, String)>> {
        let group_descr = group_descr.to_string();
        debug!("list_group_labeled_resources_sets: group_descr={group_descr}");
        self.with_reader_conn("list_group_labeled_resources_sets", move |conn| {
            let labeled_resources_sets = {
                let mut stmt = conn.prepare("SELECT l.label, rs.descr FROM resource_set_group rsg JOIN resource_set rs ON rs.group_id=rsg.id JOIN labels l ON l.id=rs.label_id WHERE rsg.descr=? ORDER BY label")?;

                stmt.query_map([group_descr.as_str()], |row| {
                    Ok((
                        row.get::<usize, String>(0)?,
                        row.get::<usize, String>(1)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?
            };

            debug!("list_group_labeled_resources_sets: labeled_resources_sets={labeled_resources_sets:?}");
            Ok(labeled_resources_sets)
        })
        .await
    }

    async fn list_resource_set_resources(&self, set_id: usize) -> anyhow::Result<Vec<Resource>> {
        self.with_reader_conn("list_resource_set_resources", move |conn| {
            let set_resources = Self::list_resource_set_resources(set_id, conn)?;
            Ok(set_resources)
        })
        .await
    }

    async fn list_resource_random_in_group(
        &self,
        group_descr: &str,
    ) -> anyhow::Result<Vec<(Resource, Vec<f32>)>> {
        let group_descr = group_descr.to_string();
        self.with_reader_conn("list_resource_random_in_group", move |conn| {
            let labels = Self::list_labels(conn, true)?;
            debug!("list_resource_random_in_group: group_descr={:?} labels={:?}", group_descr, labels);

            let r: Vec<(Resource, Vec<f32>)> = conn.prepare(
                "SELECT r.id, r.import_id, r.key, GROUP_CONCAT(l.label || '/' || lr.p, ','), r.length FROM resource_set_group rsg \
                JOIN resource_set rs ON rs.group_id=rsg.id \
                JOIN resource_resources_set rrs ON rrs.set_id=rs.id \
                JOIN resources r ON r.id=rrs.resource_id \
                JOIN labeled_resources lr ON lr.resource_id=r.id \
                JOIN labels l ON l.id=lr.label_id \
                WHERE rsg.descr=? GROUP BY r.key ORDER BY RANDOM()"
            )?.query_map([Value::Text(group_descr.clone())], |row| {
                let key = row.get::<usize, u32>(0)?;
                let labels_p = row.get::<usize, String>(3)?;

                let labels_p_vec = labels_p.split(',').map(|entry| {
                    let (label, probability) = entry.rsplit_once('/').ok_or_else(|| rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid label probability"))))?;
                    let probability = probability.parse::<f32>().map_err(|error| rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, Box::new(error)))?;
                    Ok((label.to_string(), probability))
                }).collect::<rusqlite::Result<HashMap<String, f32>>>()?;

                trace!("list_resource_random_in_group: key={} labels_p={} labels_p_vec={:?}", key, labels_p, labels_p_vec);

                Ok((
                    Resource::new(
                        row.get::<usize, u32>(0)?,
                        row.get::<usize, u32>(1)?,
                        row.get::<usize, String>(2)?,
                        row.get::<usize, u32>(4)?
                    ),
                    labels.iter().map(|(_label_id, label)| *labels_p_vec.get(label).unwrap_or(&0.0)).collect::<Vec<f32>>()
                ))
            })?.collect::<rusqlite::Result<Vec<_>>>()?;

            debug!("list_resource_random_in_group: r.len={:?}", r.len());
            Ok(r)
        })
        .await
    }

    async fn list_resources_not_in_group(
        &self,
        group_descr: &str,
        limit:u32
    ) -> anyhow::Result<Vec<Resource>> {
        let group_descr = group_descr.to_string();
        self.with_reader_conn("list_resources_not_in_group", move |conn| {
            let group_sets = {
                let mut stmt = conn.prepare(
                    "SELECT r.id, r.import_id, r.key, r.length FROM resources AS r WHERE NOT EXISTS \
                    (SELECT 1 FROM resource_resources_set AS rrs JOIN resource_set AS rs  ON rs.id = rrs.set_id JOIN resource_set_group  AS g   ON g.id = rs.group_id WHERE rrs.resource_id = r.id AND g.descr = ? )\
                     ORDER BY RANDOM()  limit ?"
                )?;

                stmt.query_map([Value::Text(group_descr), Value::Integer(limit as i64)], |row| {
                    Ok(Resource::new(
                        row.get::<usize, u32>(0)?,
                        row.get::<usize, u32>(1)?,
                        row.get::<usize, String>(2)?,
                        row.get::<usize, u32>(3)?
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?
            };

            Ok(group_sets)
        })
        .await
    }

    async fn list_random_resources(
        &self,
        limit: Option<u32>
    ) -> anyhow::Result<Vec<Resource>> {
        self.with_reader_conn("list_random_resources", move |conn| {
            let resources = if let Some(limit) = limit {
                let mut stmt = conn.prepare("SELECT r.id, r.import_id, r.key, r.length FROM resources AS r ORDER BY RANDOM() LIMIT ?")?;
                stmt.query_map([Value::Integer(limit as i64)], |row| {
                    Ok(Resource::new(
                        row.get::<usize, u32>(0)?,
                        row.get::<usize, u32>(1)?,
                        row.get::<usize, String>(2)?,
                        row.get::<usize, u32>(3)?
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?
            } else {
                let mut stmt = conn.prepare("SELECT r.id, r.import_id, r.key, r.length FROM resources AS r ORDER BY RANDOM();")?;
                stmt.query_map([], |row| {
                    Ok(Resource::new(
                        row.get::<usize, u32>(0)?,
                        row.get::<usize, u32>(1)?,
                        row.get::<usize, String>(2)?,
                        row.get::<usize, u32>(3)?
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?
            };

            Ok(resources)
        })
        .await
    }


}
#[async_trait]
//TODO move out all prepare stmts
impl Store for SqlStore {
    fn bind_run(&self, run_id: RunId) -> anyhow::Result<()> {
        match self.run_id.set(run_id) {
            Ok(()) => Ok(()),
            Err(id) if self.run_id.get() == Some(&id) => Ok(()),
            Err(_) => anyhow::bail!("store is already bound to another run"),
        }
    }

    async fn save_import(&self, descr: String) -> anyhow::Result<u32> {
        self.with_writer_conn("save_import", move |conn| {
            let t = conn.transaction()?;
            let import_id = Self::save_import(descr.as_str(), &t)?;
            t.commit()?;
            Ok(import_id)
        })
        .await
    }

    async fn save_run(&self, descr: String) -> anyhow::Result<RunId> {
        debug!("SqlStore::save_run: descr={descr}");
        self.with_writer_conn("save_run", move |conn| {
            let t = conn.transaction().with_context(|| "start sqlite transaction")?;
            t.execute("INSERT INTO runs (descr) VALUES (?)", params![descr])?;

            let run_id = t.query_row("SELECT last_insert_rowid()", [], |row| {
                row.get::<usize, RunId>(0)
            })?;

            t.commit()?;
            Ok(run_id as RunId)
        })
        .await
    }

    async fn save_run_config<'a, 'b, 'c>(
        &self,
        input_run_config_id: Option<&'a [RunConfigId]>,
        run_config_type: &'b str,
        descr: &'c str,
        properties_vals: HashMap<String, Val>,
    ) -> anyhow::Result<RunConfigId> {
        let input_run_config_id = input_run_config_id.map(|v| v.to_vec());
        let run_config_type = run_config_type.to_string();
        let descr = descr.to_string();
        debug!("SqlStore::save_run_config: input_run_config_id={input_run_config_id:?} run_config_type={run_config_type} descr={descr}");
        let run_id = self.run_id();
        self.with_writer_conn("save_run_config", move |conn| {
            let t = conn.transaction().with_context(|| "start sqlite transaction")?;

            t.execute("INSERT INTO run_configs (type, descr, run_id) VALUES (?, ?, ?)", params![run_config_type, descr, run_id])?;

            let run_config_id = t.query_row("SELECT last_insert_rowid()", [], |row| {
                row.get::<usize, RunConfigId>(0)
            })?;

            Self::insert_input( input_run_config_id, &t, run_config_id, "INSERT INTO run_configs_input (run_config_id, input_run_config_id) VALUES")?;

            if properties_vals.len() > 0 {
                for property_val in properties_vals.into_iter() {
                    Self::save_param_value(&t, run_config_id, &property_val.0, property_val.1, None)?;
                }
            }

            t.commit()?;
            Ok(run_config_id as RunConfigId)
        })
        .await
    }

    async fn get_input_run_config_ids<'a>(
        &self,
        run_config_id: RunConfigId,
        input_type: Option<&'a str>,
    ) -> anyhow::Result<Vec<RunConfigId>> {
        let input_type = input_type.map(|v| v.to_string());
        debug!("SqlStore::get_input_run_config_ids: run_config_id={run_config_id} input_type={input_type:?}");
        self.with_reader_conn("get_input_run_config_ids", move |conn| {
            let input_run_config_ids = Self::get_input_run_config_ids_(conn, run_config_id, input_type.as_deref())?;
            debug!("get_input_run_config_ids: input_run_config_ids={input_run_config_ids:?}");
            Ok(input_run_config_ids)
        })
        .await
    }

    /// returns - Vec<run_config_id, input_run_config_id>
    async fn list_input_run_config_ids(&self, ids: &[RunConfigId]) -> anyhow::Result<Vec<(RunConfigId, RunConfigId)>> {
        assert!(!ids.is_empty());
        let ids = ids.to_vec();
        self.with_reader_conn("list_input_run_config_ids", move |conn| {
            let mut id_input_id = vec![];
            for ids_chunk in ids.chunks(500) {
                let (mut q, params) = ids_chunk.iter().fold(
                    (
                        vec!["SELECT run_config_id, input_run_config_id FROM run_configs_input WHERE run_config_id IN ("],
                        vec![],
                    ),
                    |mut a, v| {
                        a.0.push("?");
                        a.0.push(",");
                        a.1.push(*v);
                        a
                    },
                );

                q.pop();
                q.push(")");

                let mut stmt = conn.prepare(q.into_iter().collect::<String>().as_str())?;
                let chunk_rows = stmt
                    .query_map(rusqlite::params_from_iter(params.into_iter()), |row| {
                        Ok((row.get::<usize, RunConfigId>(0)?, row.get::<usize, RunConfigId>(1)?))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                id_input_id.extend(chunk_rows);
            }
            Ok(id_input_id)
        })
        .await
    }

    async fn list_run_config_ids_with_outputs(&self, input_run_config_ids: &[RunConfigId]) -> anyhow::Result<Vec<RunConfigId>> {
        if input_run_config_ids.is_empty() {
            return Ok(vec![]);
        }

        let input_run_config_ids = input_run_config_ids.to_vec();
        self.with_reader_conn("list_run_config_ids_with_outputs", move |conn| {
            let mut run_config_ids_with_outputs = vec![];

            for ids_chunk in input_run_config_ids.chunks(500) {
                let (mut q, params) = ids_chunk.iter().fold(
                    (
                        vec!["SELECT DISTINCT input_run_config_id FROM run_configs_input WHERE input_run_config_id IN ("],
                        vec![],
                    ),
                    |mut a, v| {
                        a.0.push("?");
                        a.0.push(",");
                        a.1.push(*v);
                        a
                    },
                );

                q.pop();
                q.push(")");

                let mut stmt = conn.prepare(q.into_iter().collect::<String>().as_str())?;
                let chunk_rows = stmt
                    .query_map(rusqlite::params_from_iter(params.into_iter()), |row| row.get::<usize, RunConfigId>(0))?
                    .collect::<Result<Vec<_>, _>>()?;
                run_config_ids_with_outputs.extend(chunk_rows);
            }

            Ok(run_config_ids_with_outputs)
        })
        .await
    }

    //TODO used in tests only, remove?
    async fn save_resources_set(
        &self,
        descr: String,
        resource_keys: Vec<String>,
        group_id: Option<usize>,
    ) -> anyhow::Result<usize> {
        self.with_writer_conn("save_resources_set", move |conn| {
            let t = conn.transaction()?;

            match group_id {
                Some(group_id) => {
                    t.execute("INSERT INTO resource_set (descr, group_id) VALUES (?, ?)", params![descr, group_id])?;
                }
                None => {
                    t.execute("INSERT INTO resource_set (descr) VALUES (?)", params![descr])?;
                }
            };

            let resource_set_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get::<usize, usize>(0))?;

            let q = vec!["INSERT INTO resource_resources_set (set_id, resource_id) SELECT ?, id FROM resources WHERE key IN ("];
            let params = vec![Value::Integer(resource_set_id as i64)];
            for resources_chunk in resource_keys.chunks(500) {
                let (mut q, params) = resources_chunk.iter().fold((q.clone(), params.clone()), |(mut q, mut params), key| {
                    params.push(Value::Text(key.clone()));
                    q.push("?");
                    q.push(",");
                    (q, params)
                });

                q.pop();
                q.push(")");
                t.execute(&q.into_iter().collect::<String>(), rusqlite::params_from_iter(params.into_iter()))?;
            }

            t.commit()?;
            Ok(resource_set_id)
        })
        .await
    }

    async fn save_resource_set_group(&self, descr: String) -> anyhow::Result<usize> {
        self.with_writer_conn("save_resource_set_group", move |conn| {
            let mut t = conn.transaction()?;
            let group_id = Self::save_resource_set_group(descr, &mut t)?;
            t.commit()?;
            Ok(group_id)
        })
        .await
    }

    async fn list_resources_by_ids(
        &self,
        resource_ids: &Vec<ResourceId>,
    ) -> anyhow::Result<Vec<Resource>> {
        assert!(!resource_ids.is_empty(), "resource_ids are empty");
        let resource_ids = resource_ids.clone();
        self.with_reader_conn("list_resources_by_ids", move |conn| {
            let (mut q, params) = resource_ids.into_iter().fold(
                (
                    vec!["SELECT id, import_id, key, length FROM resources WHERE id IN ("],
                    vec![],
                ),
                |mut a, v| {
                    a.0.push("?");
                    a.0.push(",");
                    a.1.push(v);
                    a
                },
            );

            q.pop();
            q.push(")");

            let resources = {
                let mut stmt = conn.prepare(q.into_iter().collect::<String>().as_str())?;

                stmt.query_map(rusqlite::params_from_iter(params.into_iter()), |row| {
                    Ok(Resource::new(
                        row.get::<usize, u32>(0)?,
                        row.get::<usize, u32>(1)?,
                        row.get::<usize, String>(2)?,
                        row.get::<usize, u32>(3)?,
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?
            };

            Ok(resources)
        })
        .await
    }

    async fn list_run_resources(&self, run_config_id: RunConfigId) -> anyhow::Result<Vec<String>> {
        self.with_reader_conn("list_run_resources", move |conn| {
            let result = {
                let count: i64 = conn.query_row("SELECT count(*) FROM run_resources WHERE run_config_id=?", [run_config_id], |row| row.get(0))?;
                debug!("list_run_resources: count={}", count);
                let mut stmt = conn.prepare("SELECT ord_id, key FROM run_resources WHERE run_config_id=? ORDER BY ord_id")?;

                stmt.query_map([run_config_id], |row| Ok((row.get::<usize, i64>(0)?, row.get::<usize, String>(1)?)))?
                    .enumerate()
                    .map(|(i, ord_key)| {
                        let (ord, key) = ord_key?;
                        assert!(ord == i as i64);
                        Ok(key)
                    })
                    .collect::<anyhow::Result<Vec<String>>>()
            };

            debug!("list_run_resources: result.is_ok={}", result.is_ok());
            result
        })
        .await
    }

    async fn list_run_configs(&self) -> anyhow::Result<Vec<RunConfigEntry>> {
        self.with_reader_conn("list_run_configs", move |conn| {
            let run_configs = {
                let mut stmt = conn.prepare("SELECT id, type, descr, run_on FROM run_configs ORDER BY id DESC")?;
                stmt.query_map([], |row| {
                    Ok(RunConfigEntry {
                        id: row.get::<usize, RunConfigId>(0)?,
                        run_config_type: row.get::<usize, String>(1)?,
                        descr: row.get::<usize, String>(2)?,
                        run_on: row.get::<usize, String>(3)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<RunConfigEntry>>>()?
            };

            Ok(run_configs)
        })
        .await
    }

    async fn list_run_config_ids_by_descr(&self, descr:String, type_:&'static str) -> anyhow::Result<Vec<RunConfigId>> {
        let run_id = self.run_id();
        self.with_reader_conn("list_run_configs", move |conn| {
            let run_configs = {
                let mut stmt = conn.prepare("SELECT id FROM run_configs WHERE run_id=? AND descr=? AND type=?")?;
                stmt.query_map(params![
                   run_id,
                   descr.as_str(),
                   type_
               ], |row| {
                    row.get::<usize, RunConfigId>(0)
                })?
                    .collect::<rusqlite::Result<Vec<RunConfigId>>>()?
            };

            Ok(run_configs)
        })
            .await
    }


    async fn list_run_configs_page(&self, limit: usize, offset: usize) -> anyhow::Result<Vec<RunConfigEntry>> {
        self.with_reader_conn("list_run_configs_page", move |conn| {
            let run_configs = {
                let mut stmt = conn.prepare("SELECT id, type, descr, run_on FROM run_configs ORDER BY id DESC LIMIT ? OFFSET ?")?;
                stmt.query_map([limit as i64, offset as i64], |row| {
                    Ok(RunConfigEntry {
                        id: row.get::<usize, RunConfigId>(0)?,
                        run_config_type: row.get::<usize, String>(1)?,
                        descr: row.get::<usize, String>(2)?,
                        run_on: row.get::<usize, String>(3)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<RunConfigEntry>>>()?
            };

            Ok(run_configs)
        })
        .await
    }

    async fn list_run_configs_by_ids(&self, ids: &[RunConfigId]) -> anyhow::Result<Vec<RunConfigEntry>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }

        let ids = ids.to_vec();
        self.with_reader_conn("list_run_configs_by_ids", move |conn| {
            let mut run_configs = vec![];

            for ids_chunk in ids.chunks(500) {
                let (mut q, params) = ids_chunk.iter().fold(
                    (vec!["SELECT id, type, descr, run_on FROM run_configs WHERE id IN ("], vec![]),
                    |mut a, v| {
                        a.0.push("?");
                        a.0.push(",");
                        a.1.push(*v);
                        a
                    },
                );
                q.pop();
                q.push(")");

                let mut stmt = conn.prepare(q.into_iter().collect::<String>().as_str())?;
                let chunk_rows = stmt
                    .query_map(rusqlite::params_from_iter(params.into_iter()), |row| {
                        Ok(RunConfigEntry {
                            id: row.get::<usize, RunConfigId>(0)?,
                            run_config_type: row.get::<usize, String>(1)?,
                            descr: row.get::<usize, String>(2)?,
                            run_on: row.get::<usize, String>(3)?,
                        })
                    })?
                    .collect::<rusqlite::Result<Vec<RunConfigEntry>>>()?;
                run_configs.extend(chunk_rows);
            }

            run_configs.sort_by(|a, b| b.id.cmp(&a.id));
            Ok(run_configs)
        })
        .await
    }

    async fn list_run_configs_filtered(&self, filter: RunConfigFilter) -> anyhow::Result<Vec<RunConfigEntry>> {
        self.with_reader_conn("list_run_configs_filtered", move |conn| {
            let mut where_clauses = vec![];
            let mut params = vec![];

            if let Some(id) = filter.id {
                where_clauses.push("rc.id = ?");
                params.push(Value::Integer(id as i64));
            }
            if let Some(run_config_type) = filter.run_config_type.as_ref() {
                where_clauses.push("rc.type = ?");
                params.push(Value::Text(run_config_type.clone()));
            }
            if let Some(descr_substr) = filter.descr_substr.as_ref() {
                where_clauses.push("rc.descr LIKE ? COLLATE NOCASE");
                params.push(Value::Text(format!("%{descr_substr}%")));
            }
            if let Some(run_on_from) = filter.run_on_from.as_ref() {
                where_clauses.push("rc.run_on >= ?");
                params.push(Value::Text(run_on_from.clone()));
            }
            if let Some(run_on_to) = filter.run_on_to.as_ref() {
                where_clauses.push("rc.run_on <= ?");
                params.push(Value::Text(run_on_to.clone()));
            }

            let where_sql = if where_clauses.is_empty() {
                "1 = 1".to_string()
            } else {
                where_clauses.join(" AND ")
            };
            let query = format!(
                "WITH RECURSIVE matched(id) AS (SELECT rc.id FROM run_configs rc WHERE {}), visible(id) AS (SELECT id FROM matched UNION SELECT rci.input_run_config_id FROM run_configs_input rci JOIN visible v ON rci.run_config_id = v.id) SELECT rc.id, rc.type, rc.descr, rc.run_on FROM run_configs rc JOIN visible v ON v.id = rc.id ORDER BY rc.id DESC",
                where_sql,
            );

            let mut stmt = conn.prepare(query.as_str())?;
            Ok(stmt.query_map(rusqlite::params_from_iter(params.into_iter()), |row| {
                Ok(RunConfigEntry {
                    id: row.get::<usize, RunConfigId>(0)?,
                    run_config_type: row.get::<usize, String>(1)?,
                    descr: row.get::<usize, String>(2)?,
                    run_on: row.get::<usize, String>(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<RunConfigEntry>>>()?)
        })
        .await
    }

    async fn list_run_config_types(&self) -> anyhow::Result<Vec<String>> {
        self.with_reader_conn("list_run_config_types", move |conn| {
            let run_config_types = {
                let mut stmt = conn.prepare("SELECT DISTINCT type FROM run_configs ORDER BY type ASC")?;
                stmt.query_map([], |row| row.get::<usize, String>(0))?
                    .collect::<rusqlite::Result<Vec<String>>>()?
            };

            Ok(run_config_types)
        })
        .await
    }

    async fn list_run_config_values(&self, run_config_id: RunConfigId) -> anyhow::Result<HashMap<String, Val>> {
        self.with_reader_conn("list_run_config_values", move |conn| {
            let result = {
                let mut stmt = conn.prepare("SELECT name, real_value, int_value, str_value, bool_value, unsigned FROM run_config_values WHERE run_config_id=?")?;

                stmt.query_map([run_config_id], |row| {
                    Ok((
                        row.get::<usize, String>(0)?,
                        row.get::<usize, Option<f64>>(1)?,
                        row.get::<usize, Option<i64>>(2)?,
                        row.get::<usize, Option<String>>(3)?,
                        row.get::<usize, Option<bool>>(4)?,
                        row.get::<usize, Option<bool>>(5)?.unwrap_or(false),
                    ))
                })?
                // NULL is optional data; a typed decoding failure must remain an error.
                .collect::<rusqlite::Result<Vec<_>>>()?
                .into_iter()
                .map(|(name, real_value, int_value, str_value, bool_value, unsigned)| {
                    let value = if let Some(value) = real_value { Val::Real(value) }
                    else if let Some(value) = int_value {
                        if unsigned { UInt(u32::try_from(value).with_context(|| format!("invalid unsigned value for {name}"))?) } else { Val::Int(value) }
                    } else if let Some(value) = str_value { Val::Str(value) }
                    else if let Some(value) = bool_value { Val::Bool(value) }
                    else { anyhow::bail!("no value for configuration property {name}"); };
                    Ok((name, value))
                })
                .collect::<anyhow::Result<HashMap<_, _>>>()?
            };

            debug!("list_run_config_values: run_config_id={run_config_id} result={result:?}");
            Ok(result)
        })
        .await
    }


    async fn list_output_run_config_ids(&self, input_run_config_id: RunConfigId) -> anyhow::Result<Vec<RunConfigId>> {
        self.with_reader_conn("list_output_run_config_ids", move |conn| {
            let q = "SELECT run_config_id FROM run_configs_input WHERE input_run_config_id=?";

            let id_input_id = {
                let mut stmt = conn.prepare(q)?;
                stmt.query_map([input_run_config_id], |row| row.get::<usize, usize>(0))?
                    .map(|v| Ok::<_, anyhow::Error>(v? as RunConfigId))
                    .collect::<anyhow::Result<Vec<_>>>()
            };
            id_input_id
        })
        .await
    }
    async fn save_trained_net( //TODO there are no 'untrained' nets - it's 'trained' just to make name longer and more distinctive
                               &self,
                               run_config_id: RunConfigId,
                               path: PathBuf,
                               loss: f32,
                               input_net_ids: Option<Vec<u32>>,
    ) -> anyhow::Result<u32> {
        self.with_writer_conn("save_trained_net", move |conn| {
            let t = conn.transaction()?;
            t.execute(
                "INSERT INTO trained_nets (net_path, run_config_id, loss) VALUES (?, ?, ?)",
                rusqlite::params![path.to_string_lossy().to_string(), run_config_id, loss],
            )?;

            let net_id = t.query_row("SELECT last_insert_rowid()", [], |row| {
                row.get::<usize, u32>(0)
            })?;


            Self::insert_input(input_net_ids, &t, net_id, "INSERT INTO trained_nets_input (net_id, input_id) VALUES")?;


            t.commit()?;
            Ok(net_id)
        })
        .await
    }

    async fn list_trained_nets(&self, by: TrainedNetsBy, loss: Option<f32>, limit: Option<usize>) -> anyhow::Result<Vec<TrainedNet>> {
        self.with_reader_conn("list_trained_nets", move |conn| {
            let (field, ids) = match by {
                TrainedNetsBy::RunConfigIds(ids) => {
                    ("run_config_id", ids)
                }
                TrainedNetsBy::TrainNetId(ids) => {
                    ("id", ids)
                }
            };
            if ids.is_empty() {
                return Ok(Vec::new());
            }

            let trained_nets = {
                let run_config_ids_clause = ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
                let query = [
                    "SELECT id, run_config_id, net_path, loss FROM trained_nets WHERE ",
                    field,
                    " IN (",
                    &run_config_ids_clause,
                    ")",
                    if loss.is_some() { " AND loss < ?" } else { "" },
                    " ORDER BY loss ASC",
                    if limit.is_some() { " LIMIT ?" } else { "" },
                ].concat();

                let mut params = ids
                    .iter()
                    .map(|run_config_id| rusqlite::types::Value::Integer(*run_config_id as i64))
                    .collect::<Vec<_>>();
                if let Some(loss) = loss {
                    params.push(rusqlite::types::Value::Real(loss as f64));
                }
                if let Some(limit) = limit {
                    params.push(rusqlite::types::Value::Integer(limit as i64));
                }

                let mut stmt = conn.prepare(&query)?;
                stmt.query_map(rusqlite::params_from_iter(params.into_iter()), |row| {
                    Ok(TrainedNet {
                        id: row.get::<usize, u32>(0)?,
                        run_config_id: row.get::<usize, RunConfigId>(1)?,
                        net_path: PathBuf::from(row.get::<usize, String>(2)?),
                        loss: row.get::<usize, f32>(3)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?
            };

            Ok(trained_nets)
        })
        .await
    }

    async fn list_trained_net_entries(&self) -> anyhow::Result<Vec<TrainedNetEntry>> {
        self.with_reader_conn("list_trained_net_entries", move |conn| {
            let mut stmt = conn.prepare("SELECT id, net_path FROM trained_nets")?;
            Ok(stmt.query_map([], Self::trained_net_entry)?
                .collect::<rusqlite::Result<Vec<_>>>()?)
        }).await
    }

    async fn list_label_trained_nets_before(&self, run_on_before: String) -> anyhow::Result<Vec<TrainedNetEntry>> {
        self.with_reader_conn("list_label_trained_nets_before", move |conn| {
            let mut stmt = conn.prepare("SELECT tn.id, tn.net_path FROM trained_nets tn LEFT JOIN run_configs rc ON tn.run_config_id=rc.id WHERE rc.type=? AND rc.run_on < ?")?;
            Ok(stmt.query_map(params![TRAIN_LABEL_NETS_RUN_CONFIG_TYPE, run_on_before], Self::trained_net_entry)?
                .collect::<rusqlite::Result<Vec<_>>>()?)
        }).await
    }

    async fn list_trained_nets_loss_gt(&self, loss: f32, net_type: &'static str) -> anyhow::Result<Vec<TrainedNetEntry>> {
        self.with_reader_conn("list_label_trained_nets_loss_gt", move |conn| {
            let mut stmt = conn.prepare("SELECT tn.id, tn.net_path FROM trained_nets tn LEFT JOIN run_configs rc ON tn.run_config_id=rc.id WHERE rc.type=? AND tn.loss > ?")?;
            Ok(stmt.query_map(params![net_type, loss], Self::trained_net_entry)?
                .collect::<rusqlite::Result<Vec<_>>>()?)
        }).await
    }

    async fn list_unused_motif_trained_nets(&self) -> anyhow::Result<Vec<TrainedNetEntry>> {
        self.with_reader_conn("list_unused_motif_trained_nets", move |conn| {
            let mut stmt = conn.prepare("SELECT tn.id, tn.net_path FROM trained_nets tn LEFT JOIN run_configs rc ON tn.run_config_id=rc.id WHERE rc.type=? AND NOT EXISTS (SELECT 1 FROM trained_nets_input nti WHERE nti.input_id=tn.id)")?;
            Ok(stmt.query_map(params![TRAIN_MOTIFS_NETS_RUN_CONFIG_TYPE], Self::trained_net_entry)?
                .collect::<rusqlite::Result<Vec<_>>>()?)
        }).await
    }

    async fn delete_trained_nets(&self, net_ids: Vec<u32>) -> anyhow::Result<usize> {
        self.with_writer_conn("delete_trained_nets", move |conn| {
            if net_ids.is_empty() {
                return Ok(0);
            }

            let placeholders = net_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
            let query = format!("DELETE FROM trained_nets WHERE id IN ({placeholders})");
            let params = net_ids.into_iter().map(|net_id| rusqlite::types::Value::Integer(net_id as i64)).collect::<Vec<_>>();
            let t = conn.transaction()?;
            let deleted = t.execute(&query, rusqlite::params_from_iter(params))?;
            t.commit()?;
            Ok(deleted)
        }).await
    }

    async fn update_run_config_run_on(&self, run_config_id: RunConfigId, run_on: String) -> anyhow::Result<()> {
        self.with_writer_conn("update_run_config_run_on", move |conn| {
            conn.execute("UPDATE run_configs SET run_on=? WHERE id=?", params![run_on, run_config_id])?;
            Ok(())
        }).await
    }

    //TODO rename list_input_nets_run_configs?
    async fn list_input_nets(&self, net_id: u32) -> anyhow::Result<Vec<u32>> {
        self.with_reader_conn("list_input_nets", move |conn| {
            let mut stmt = conn.prepare("SELECT tn.run_config_id FROM trained_nets_input tni LEFT JOIN trained_nets tn ON tn.id = tni.input_id WHERE tni.net_id=?")?;

            Ok(stmt.query_map(params![net_id], |row| {
                row.get::<usize, u32>(0)
            })?
                .collect::<rusqlite::Result<Vec<_>>>()?)
        }).await
    }

    async fn run_config_descr(&self, corr_run_config_id: RunConfigId) -> anyhow::Result<RunConfigRow> {
        self.with_reader_conn("run_config_descr", move |conn| {
            let run_config = conn.query_row("SELECT id, type, descr, run_on FROM run_configs WHERE id=?", [corr_run_config_id], |row| {
                Ok(RunConfigRow {
                    id: row.get::<usize, RunConfigId>(0)?,
                    r#type: row.get::<usize, String>(1)?,
                    descr: row.get::<usize, String>(2)?,
                    run_on: row.get::<usize, String>(3)?,
                })
            })?;
            debug!("run_config={run_config:?}");

            Ok(run_config)
        })
        .await
    }


}

fn init_sqlite_writer(conn: &mut rusqlite::Connection) -> rusqlite::Result<()> {

    conn.busy_timeout(Duration::from_secs(5))?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.pragma_update(None, "case_sensitive_like", "ON")?;

    // conn.trace_v2(
    //     TraceEventCodes::SQLITE_TRACE_STMT
    //         | TraceEventCodes::SQLITE_TRACE_PROFILE
    //         | TraceEventCodes::SQLITE_TRACE_CLOSE,
    //     Some(|ev: TraceEvent<'_>| {
    //         debug_trace_event(ev);
    //     }),
    // );

    Ok(())
}

fn init_sqlite_reader(conn: &mut rusqlite::Connection) -> rusqlite::Result<()> {

    conn.busy_timeout(Duration::from_secs(5))?;
    conn.pragma_update(None, "foreign_keys", "ON")?;

    // conn.trace_v2(
    //       TraceEventCodes::SQLITE_TRACE_STMT
    //     | TraceEventCodes::SQLITE_TRACE_PROFILE
    //     | TraceEventCodes::SQLITE_TRACE_CLOSE,
    //     Some(|ev: TraceEvent<'_>| {
    //         debug_trace_event(ev);
    //
    //     }),
    // );

    Ok(())
}

fn init_sqlite_reader_no_tmp_files(conn: &mut rusqlite::Connection) -> rusqlite::Result<()> {

    // conn.pragma_update(None, "wal_checkpoint", "TRUNCATE")?;
    // conn.pragma_update(None, "journal_mode", "DELETE")?;
    conn.pragma_update(None, "temp_store", "MEMORY")?;
    Ok(())
}

#[expect(dead_code, reason = "Optional callback for the commented SQLite trace hooks above.")]
fn debug_trace_event(ev: TraceEvent) {
    match ev {
        TraceEvent::Stmt(_a, b) => {
            debug!("[sqlite trace reader], TraceEvent::Stmt: {b:?}");
        }
        TraceEvent::Profile(_a, b) => {
            debug!("[sqlite trace reader], TraceEvent::Profile: {b:?}");
        }
        TraceEvent::Row(_a) => {
            debug!("[sqlite trace reader], TraceEvent::Row:");
        }
        TraceEvent::Close(_a) => {
            debug!("[sqlite trace reader], TraceEvent::Close:  backtrace={}", Backtrace::capture());
        }
        _x => {
            debug!("[sqlite trace reader] x");
        }
    }
}

type CreateTablesFn = fn(&mut Connection) -> anyhow::Result<()>;

//TODO move all prepare to init
impl SqlStore {
    pub fn run_id(&self) -> RunId { self.run_id.get().copied().unwrap_or(0) }

    fn trained_net_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<TrainedNetEntry> {
        Ok(TrainedNetEntry {
            id: row.get::<usize, u32>(0)?,
            net_path: PathBuf::from(row.get::<usize, String>(1)?),
        })
    }

    pub fn count_motif_slices(motif_id: u32, conn: &Connection) -> anyhow::Result<usize> {
        let count = conn.query_row("SELECT count(*) FROM slices WHERE motif_id = ?", [motif_id], |row| {
            row.get::<usize, i64>(0)
        })?;

        Ok(usize::try_from(count).with_context(|| format!("count_motif_slices: invalid count={count} for motif_id={motif_id}"))?)
    }

    pub fn load_motif_target_positions(
        motif_id: u32,
        conn: &Connection,
    ) -> anyhow::Result<(HashMap<ResourceId, Vec<i64>>, usize)> {
        let mut target_positions_by_res: HashMap<ResourceId, Vec<i64>> = HashMap::new();
        let mut num_motif_slices = 0usize;
        let mut stmt = conn.prepare(
            "SELECT s.resource_id, s.slice_idx
             FROM slices s
             WHERE s.motif_id = ?
             ORDER BY s.resource_id, s.slice_idx",
        )?;

        stmt.query_map([motif_id], |row| {
            Ok((
                row.get::<usize, ResourceId>(0)?,
                row.get::<usize, i64>(1)?,
            ))
        })?
        .try_for_each(|row| -> anyhow::Result<()> {
            let (resource_id, slice_idx) = row?;
            target_positions_by_res.entry(resource_id).or_default().push(slice_idx);
            num_motif_slices += 1;
            Ok(())
        })?;

        for positions in target_positions_by_res.values_mut() {
            positions.sort_unstable();
        }

        Ok((target_positions_by_res, num_motif_slices))
    }

    pub fn load_motif_resource_contexts(
        motif_id: u32,
        input_len_adjuster: &(dyn Fn(u32) -> u32 + Send + Sync),
        conn: &Connection,
        mut target_positions_by_res: HashMap<ResourceId, Vec<i64>>,
    ) -> anyhow::Result<Vec<RandomSliceResourceCtx>> {
        let mut stmt = conn.prepare(
            "SELECT DISTINCT r.id, r.import_id, r.key, r.length
             FROM slices s
             JOIN resources r ON r.id = s.resource_id
             WHERE s.motif_id = ?",
        )?;

        stmt.query_map(params![motif_id as i64], |row| {
            Ok((
                row.get::<usize, ResourceId>(0)?,
                row.get::<usize, u32>(1)?,
                row.get::<usize, String>(2)?,
                row.get::<usize, u32>(3)?,
            ))
        })?
        .map(|row| {
            let (resource_id, import_id, key, length) = row?;
            let resource = Resource::new(resource_id, import_id, key, length);

            Ok(RandomSliceResourceCtx {
                resource,
                adjusted_len: input_len_adjuster(length) as i64,
                target_positions: target_positions_by_res.remove(&resource_id).unwrap_or_default(),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()
    }

    pub fn load_non_motif_resource_contexts(
        motif_id: u32,
        _run_config_id: RunConfigId,
        _channel: i64,
        input_len_adjuster: &(dyn Fn(u32) -> u32 + Send + Sync),
        conn: &Connection,
    ) -> anyhow::Result<Vec<RandomSliceResourceCtx>> {
        let mut stmt = conn.prepare(
            "SELECT r.id, r.import_id, r.key, r.length FROM main.resources r WHERE r.id NOT IN (SELECT resource_id FROM slices s WHERE s.motif_id=?)",
        )?;

        Ok(stmt.query_map(params![motif_id as i64], |row| {
            Ok(RandomSliceResourceCtx {
                resource: Resource::new(
                    row.get::<usize, ResourceId>(0)?,
                    row.get::<usize, u32>(1)?,
                    row.get::<usize, String>(2)?,
                    row.get::<usize, u32>(3)?,
                ),
                adjusted_len: input_len_adjuster(row.get::<usize, u32>(3)?) as i64,
                target_positions: vec![],
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?)
    }

    fn merge_intervals(mut xs: Vec<(i64, i64)>) -> Vec<(i64, i64)> {
        if xs.is_empty() {
            return xs;
        }

        xs.sort_unstable_by_key(|&(l, _)| l);

        let mut out = Vec::with_capacity(xs.len());
        let mut cur = xs[0];

        for (l, r) in xs.into_iter().skip(1) {
            if l <= cur.1 + 1 {
                cur.1 = cur.1.max(r);
            } else {
                out.push(cur);
                cur = (l, r);
            }
        }
        out.push(cur);
        out
    }

    fn subtract_forbidden(allowed_l: i64, allowed_r: i64, forbidden: &[(i64, i64)]) -> Vec<(i64, i64)> {
        if allowed_l > allowed_r {
            return vec![];
        }

        let mut res = Vec::new();
        let mut cur = allowed_l;

        for &(l0, r0) in forbidden {
            let l = l0.max(allowed_l);
            let r = r0.min(allowed_r);

            if l > r {
                continue;
            }

            if cur < l {
                res.push((cur, l - 1));
            }

            cur = r + 1;
            if cur > allowed_r {
                break;
            }
        }

        if cur <= allowed_r {
            res.push((cur, allowed_r));
        }

        res
    }

    fn candidate_starts_from_free_intervals(free: &[(i64, i64)], step: i64) -> Vec<i64> {
        let mut out = Vec::new();

        for &(l, r) in free {
            let mut x = l;
            while x <= r {
                out.push(x);
                x += step;
            }
        }

        out
    }

    pub fn build_random_slice_candidates(
        contexts: &[RandomSliceResourceCtx],
        motif_width: u32,
        min_distance: Option<u32>,
    ) -> (Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>) {
        let width_i64 = motif_width as i64;
        let generated_guard = max(min_distance.unwrap_or(motif_width), motif_width) as i64;
        let generated_step = generated_guard + 1;
        let mut all_candidates = Vec::<TensorIdSliceIdx>::new();
        let mut resources_map = HashMap::<ResourceId, Resource>::new();

        for ctx in contexts {
            if ctx.adjusted_len <= width_i64 + 1 {
                continue;
            }

            let start_min = 1_i64;
            let start_max = ctx.adjusted_len - width_i64 - 1;

            if start_min > start_max {
                continue;
            }

            let forbidden = Self::merge_intervals(ctx.target_positions.iter().map(|&p| (p - width_i64, p + width_i64)).collect());
            let free = Self::subtract_forbidden(start_min, start_max, &forbidden);

            if free.is_empty() {
                continue;
            }

            let starts = Self::candidate_starts_from_free_intervals(&free, generated_step);
            if starts.is_empty() {
                continue;
            }

            resources_map.entry(ctx.resource.id()).or_insert_with(|| ctx.resource.clone());
            all_candidates.extend(starts.into_iter().map(|s| TensorIdSliceIdx(ctx.resource.id(), s as u32)));
        }

        (all_candidates, resources_map)
    }

    pub fn shuffle_and_limit_candidates(all_candidates: &mut Vec<TensorIdSliceIdx>, desired_count: usize) {
        let mut rng = rand::rng();
        all_candidates.shuffle(&mut rng);

        if all_candidates.len() > desired_count {
            all_candidates.truncate(desired_count);
        }
    }

    // Unwrap the worker envelope so anyhow retains the original application error.
    fn map_tokio_rusqlite_err(ctx: &'static str, error: tokio_rusqlite::Error<anyhow::Error>) -> anyhow::Error {
        let error = match error {
            tokio_rusqlite::Error::Error(inner) => inner,
            tokio_rusqlite::Error::Close((_, error)) => error.into(),
            tokio_rusqlite::Error::Panic(error) => error.into(),
            tokio_rusqlite::Error::ConnectionClosed => anyhow::anyhow!("sqlite connection closed"),
            tokio_rusqlite::Error::WorkerGone => anyhow::anyhow!("sqlite worker gone"),
        };
        error.context(ctx)
    }

    pub async fn with_writer_conn<T, F>(&self, ctx: &'static str, f: F) -> anyhow::Result<T>
    where
        T: Send + Debug + 'static,
        F: FnOnce(&mut Connection) -> anyhow::Result<T> + Send + 'static,
    {
        // Connection::call already catches worker panics; preserve its source without another wrapper.
        self.writer.call(f).await.map_err(|error| Self::map_tokio_rusqlite_err(ctx, error))
    }

    pub async fn with_reader_conn<T, F>(&self, ctx: &'static str, f: F) -> anyhow::Result<T>
    where
        T: Send + Debug + 'static,
        F: FnOnce(&mut Connection) -> anyhow::Result<T> + Send + 'static,
    {
        let reader_id = self.rr.fetch_add(1, Ordering::Relaxed) % self.readers.len();
        self.readers[reader_id].call(f).await.map_err(|error| Self::map_tokio_rusqlite_err(ctx, error))
    }

    fn save_resources_labels_do_insert(
        mut resource_keys_iter: IntoIter<(String, u32, Vec<(String, f32)>), 500>,
        import_id: u32,
        t: &Transaction,
    ) -> anyhow::Result<Vec<ResourceId>> {
        let (mut insert_resources_q, insert_resources_params, insert_labeled_resources_params) =
            resource_keys_iter.try_fold(
                (
                    vec!["INSERT INTO resources (key, length, import_id) VALUES"],
                    vec![],
                    vec![],
                ),
                |(
                    mut insert_resources_q,
                    mut insert_resources_params,
                    mut insert_labeled_resources_params,
                ),
                 (resource_key, len, labels)| {
                    insert_resources_params.push(Value::Text(resource_key));
                    insert_resources_params.push(Value::Integer(len as i64));
                    insert_resources_params.push(Value::Integer(import_id as i64));
                    insert_resources_q.push("(?, ?, ?)");
                    insert_resources_q.push(",");

                    let labels_ids = labels
                        .into_iter()
                        .map(|(label, p)| {
                            Ok((
                                t.query_row("SELECT id FROM labels WHERE label=?", [label.as_str()], |row| row.get(0))
                                    .with_context(|| format!("label {label} not found"))?,
                                p,
                            ))
                        })
                        .collect::<anyhow::Result<Vec<(u32, f32)>>>()?;
                    insert_labeled_resources_params.push(labels_ids);

                    Ok::<_, anyhow::Error>((
                        insert_resources_q,
                        insert_resources_params,
                        insert_labeled_resources_params,
                    ))
                },
            )?;

        insert_resources_q.pop();
        insert_resources_q.push(" RETURNING id");

        let q = insert_resources_q.into_iter().collect::<String>();
        let mut stmt = t.prepare(&q)?;

        let mut rows = stmt.query(rusqlite::params_from_iter(insert_resources_params.into_iter()))?;

        let mut resource_ids: Vec<u32> = vec![];
        while let Some(row) = rows.next()? {
            resource_ids.push(row.get(0)?);
        }

        let mut labels_iter = resource_ids
            .iter()
            .zip(insert_labeled_resources_params)
            .flat_map(move |(resource_id, labels)| labels.into_iter().map(move |(l, p)| (resource_id, l, p)))
            .array_chunks::<500>();

        while let Some(resource_labels) = labels_iter.next() {
            Self::save_resources_labels_insert_labels(resource_labels.into_iter(), t);
        }


        let remainder = labels_iter.into_remainder();
        if remainder.len() != 0 {
        // if !remainder.is_empty() {
            Self::save_resources_labels_insert_labels(remainder, t);
        }


        Ok(resource_ids.clone())//TODO 1.96.0-nightly: why ?
    }

    fn save_resources_labels_insert_labels(
        resource_labels: IntoIter<(&u32, u32, f32), 500>,
        t: &Transaction,
    ) {
        let (mut q, params) = resource_labels.fold(
            (
                vec!["INSERT INTO labeled_resources (resource_id, label_id, p) VALUES"],
                vec![],
            ),
            |(mut q, mut params): (Vec<&str>, Vec<Value>), v| {
                q.push("(?, ?, ?)");
                q.push(",");

                params.push(Value::Integer(*v.0 as i64));
                params.push(Value::Integer(v.1 as i64));
                params.push(Value::Real(v.2 as f64));
                (q, params)
            },
        );

        if params.len() > 0 {
            q.pop();
            let q_str = q.into_iter().collect::<String>();
            match t.execute(&q_str, rusqlite::params_from_iter(params.into_iter()))
            {
                Ok(_) => {}
                Err(e) => {
                    panic!("save_resources_labels_insert_labels: q_str={q_str:?} e={e}");
                }
            };
        }
    }

    pub async fn from_path<P: AsRef<Path>>(path: P) -> anyhow::Result<Arc<dyn Store>> {
       Ok(Arc::new(Self::from_path_(path).await?))
    }

    pub async fn from_path_<P: AsRef<Path>>(path: P) -> anyhow::Result<SqlStore> {

        Self::from_path_with_tables(path, vec![crate::create_tables]).await
    }

    pub async fn from_path_with_tables<P: AsRef<Path>>(path: P, create_tables: Vec<CreateTablesFn>) -> anyhow::Result<SqlStore> {
        let path_buf = path.as_ref().to_path_buf();
        let conn_target = path_buf.to_string_lossy().to_string();
        let conn_flags = OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE;

        let mut conn = Connection::open_with_flags(&path_buf, conn_flags)?;

        for f in create_tables {
            f(&mut conn)?;
        }

        let writer = tokio_rusqlite::Connection::open_with_flags(&path_buf, OpenFlags::SQLITE_OPEN_READ_WRITE).await?;
        writer.call(init_sqlite_writer).await?;

        let mut readers = Vec::with_capacity(SQL_STORE_READERS_CNT.max(1));
        for _ in 0..SQL_STORE_READERS_CNT.max(1) {
            let reader = tokio_rusqlite::Connection::open_with_flags(&path_buf, OpenFlags::SQLITE_OPEN_READ_ONLY).await?;
            reader.call(init_sqlite_reader).await?;
            readers.push(reader);
        }

        let readers = if readers.is_empty() {
            vec![writer.clone()]
        } else {
            readers
        };

        Ok(Self {
            writer,
            readers: Arc::new(readers),
            conn_target,
            conn_flags,
            rr: AtomicUsize::new(0),
            run_id: std::sync::OnceLock::new(),
        })
    }

    pub async fn from_path_readonly<P: AsRef<Path>>(path: P) -> anyhow::Result<Box<dyn Store>> {
        Ok(Box::new(Self::from_path_readonly_(path).await?))
    }

    /// Returns the concrete store so downstream crates can construct their specialized store trait objects.
    pub async fn from_path_readonly_<P: AsRef<Path>>(path: P) -> anyhow::Result<SqlStore> {
        let path_buf = path.as_ref().to_path_buf();
        let conn_target = path_buf.to_string_lossy().to_string();
        let conn_flags = OpenFlags::SQLITE_OPEN_READ_ONLY;

        let mut readers = Vec::with_capacity(SQL_STORE_READERS_CNT.max(1));
        for _ in 0..SQL_STORE_READERS_CNT.max(1) {
            let reader = tokio_rusqlite::Connection::open_with_flags(&path_buf, OpenFlags::SQLITE_OPEN_READ_ONLY).await?;
            reader.call(init_sqlite_reader_no_tmp_files).await?;
            readers.push(reader);
        }

        Ok(Self {
            writer: tokio_rusqlite::Connection::open_with_flags(&path_buf, OpenFlags::SQLITE_OPEN_READ_ONLY).await?,
            readers: Arc::new(readers),
            conn_target,
            conn_flags,
            rr: AtomicUsize::new(0),
            run_id: std::sync::OnceLock::new(),
        })
    }


    // #[cfg(test)]
    pub async fn in_mem() -> anyhow::Result<Arc<dyn Store>> {
        Ok(Arc::new(Self::in_mem_(vec![
            create_tables,
        ]).await?))
    }

    /// Returns the concrete in-memory store for specialized store trait objects in downstream tests.
    pub async fn in_mem_(create_tables: Vec<CreateTablesFn>) -> anyhow::Result<SqlStore> {
        let db_idx = IN_MEM_DB_COUNTER.fetch_add(1, Ordering::Relaxed);
        let uri = format!("file:thoosa_store_{db_idx}?mode=memory&cache=shared");
        let conn_flags = OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE | OpenFlags::SQLITE_OPEN_URI;

        let writer = tokio_rusqlite::Connection::open_with_flags(&uri, OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_URI).await?;
        writer.call(init_sqlite_writer).await?;

        let mut readers = Vec::with_capacity(SQL_STORE_READERS_CNT.max(1));
        for _ in 0..SQL_STORE_READERS_CNT.max(1) {
            let reader = tokio_rusqlite::Connection::open_with_flags(&uri, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI).await?;
            reader.call(init_sqlite_reader).await?;
            readers.push(reader);
        }

        let mut conn = Connection::open_with_flags(&uri, conn_flags)?;
        for f in create_tables {
            f(&mut conn)?;
        }


        let readers = if readers.is_empty() {
            vec![writer.clone()]
        } else {
            readers
        };

        Ok(Self {
            writer,
            readers: Arc::new(readers),
            conn_target: uri,
            conn_flags,
            rr: AtomicUsize::new(0),
            run_id: std::sync::OnceLock::new(),
        })
    }

    pub fn export_apply_label_nets_records(conn: &mut Connection, target_db_path: PathBuf, label_net_run_config_ids: Vec<RunConfigId>) -> anyhow::Result<()> {
        if label_net_run_config_ids.is_empty() {
            warn!("export_apply_label_nets_records: no label_net_run_config_ids");
            return Ok(());
        }

        let target_db_path = target_db_path.to_string_lossy().to_string();
        conn.execute("ATTACH DATABASE ?1 AS target", params![target_db_path])?;

        let result = (|| {
            let tx = conn.transaction()?;

            tx.execute_batch(
                r#"
                DROP TABLE IF EXISTS temp.export_label_run_config_ids;
                DROP TABLE IF EXISTS temp.export_trained_net_ids;
                DROP TABLE IF EXISTS temp.export_run_config_ids;
                DROP TABLE IF EXISTS temp.export_motif_ids;
                DROP TABLE IF EXISTS temp.export_resource_ids;

                CREATE TEMP TABLE export_label_run_config_ids (id INTEGER PRIMARY KEY);
                CREATE TEMP TABLE export_trained_net_ids (id INTEGER PRIMARY KEY);
                CREATE TEMP TABLE export_run_config_ids (id INTEGER PRIMARY KEY);
                CREATE TEMP TABLE export_motif_ids (id INTEGER PRIMARY KEY);
                CREATE TEMP TABLE export_resource_ids (id INTEGER PRIMARY KEY);
                "#,
            )?;

            for run_config_id in label_net_run_config_ids {
                tx.execute("INSERT OR IGNORE INTO export_label_run_config_ids (id) VALUES (?1)", params![run_config_id])?;
            }

            tx.execute_batch(
                r#"
                INSERT OR IGNORE INTO export_trained_net_ids (id)
                SELECT tn.id
                FROM main.trained_nets tn
                JOIN export_label_run_config_ids lrc ON lrc.id = tn.run_config_id;

                WITH RECURSIVE net_tree(id) AS (
                    SELECT id FROM export_trained_net_ids
                    UNION
                    SELECT tni.input_id
                    FROM main.trained_nets_input tni
                    JOIN net_tree nt ON nt.id = tni.net_id
                )
                INSERT OR IGNORE INTO export_trained_net_ids (id)
                SELECT id FROM net_tree;

                INSERT OR IGNORE INTO export_run_config_ids (id)
                SELECT tn.run_config_id
                FROM main.trained_nets tn
                JOIN export_trained_net_ids etn ON etn.id = tn.id;

                WITH RECURSIVE run_config_tree(id) AS (
                    SELECT id FROM export_run_config_ids
                    UNION
                    SELECT rci.input_run_config_id
                    FROM main.run_configs_input rci
                    JOIN run_config_tree rct ON rct.id = rci.run_config_id
                )
                INSERT OR IGNORE INTO export_run_config_ids (id)
                SELECT id FROM run_config_tree;

                INSERT OR IGNORE INTO export_motif_ids (id)
                SELECT DISTINCT sm.motif_id
                FROM main.selected_motifs sm
                JOIN export_run_config_ids erc ON erc.id = sm.run_config_id;

                INSERT OR IGNORE INTO export_motif_ids (id)
                SELECT DISTINCT m.id
                FROM main.motifs m
                JOIN export_run_config_ids erc ON erc.id = m.run_config_id;

                INSERT OR IGNORE INTO export_resource_ids (id)
                SELECT DISTINCT s.resource_id
                FROM main.slices s
                JOIN export_motif_ids emi ON emi.id = s.motif_id
                WHERE s.resource_id IS NOT NULL;

                INSERT OR IGNORE INTO target.imports (id, descr)
                SELECT DISTINCT i.id, i.descr
                FROM main.imports i
                JOIN main.resources r ON r.import_id = i.id
                JOIN export_resource_ids eri ON eri.id = r.id;

                INSERT OR IGNORE INTO target.resources (id, key, length, import_id)
                SELECT r.id, r.key, r.length, r.import_id
                FROM main.resources r
                JOIN export_resource_ids eri ON eri.id = r.id;

                INSERT OR IGNORE INTO target.labels (id, label)
                SELECT DISTINCT l.id, l.label
                FROM main.labels l
                WHERE EXISTS (SELECT 1 FROM main.labeled_resources lr JOIN export_resource_ids eri ON eri.id = lr.resource_id WHERE lr.label_id = l.id)
                   OR EXISTS (SELECT 1 FROM main.run_config_values rcv JOIN export_run_config_ids erc ON erc.id = rcv.run_config_id WHERE rcv.label_id = l.id);

                INSERT OR IGNORE INTO target.labeled_resources (resource_id, label_id, p)
                SELECT lr.resource_id, lr.label_id, lr.p
                FROM main.labeled_resources lr
                JOIN export_resource_ids eri ON eri.id = lr.resource_id;

                INSERT OR IGNORE INTO target.runs (id, descr, run_on)
                SELECT DISTINCT r.id, r.descr, r.run_on
                FROM main.runs r
                JOIN main.run_configs rc ON rc.run_id = r.id
                JOIN export_run_config_ids erc ON erc.id = rc.id;

                INSERT OR IGNORE INTO target.run_configs (id, type, descr, run_on, run_id)
                SELECT rc.id, rc.type, rc.descr, rc.run_on, rc.run_id
                FROM main.run_configs rc
                JOIN export_run_config_ids erc ON erc.id = rc.id;

                INSERT OR IGNORE INTO target.run_configs_input (input_run_config_id, run_config_id)
                SELECT rci.input_run_config_id, rci.run_config_id
                FROM main.run_configs_input rci
                JOIN export_run_config_ids input_erc ON input_erc.id = rci.input_run_config_id
                JOIN export_run_config_ids output_erc ON output_erc.id = rci.run_config_id;

                INSERT OR IGNORE INTO target.run_config_values (id, run_config_id, name, real_value, int_value, str_value, bool_value, unsigned, label_id)
                SELECT rcv.id, rcv.run_config_id, rcv.name, rcv.real_value, rcv.int_value, rcv.str_value, rcv.bool_value, rcv.unsigned, rcv.label_id
                FROM main.run_config_values rcv
                JOIN export_run_config_ids erc ON erc.id = rcv.run_config_id;

                INSERT OR IGNORE INTO target.motifs (id, group_ord, run_config_id, channel_id)
                SELECT m.id, m.group_ord, m.run_config_id, m.channel_id
                FROM main.motifs m
                JOIN export_motif_ids emi ON emi.id = m.id;

                INSERT OR IGNORE INTO target.slices (id, motif_id, run_config_id, channel_id, resource_id, slice_idx)
                SELECT s.id, s.motif_id, s.run_config_id, s.channel_id, s.resource_id, s.slice_idx
                FROM main.slices s
                JOIN export_motif_ids emi ON emi.id = s.motif_id
                JOIN export_run_config_ids erc ON erc.id = s.run_config_id;

                INSERT OR IGNORE INTO target.selected_motifs (motif_id, run_config_id, rate)
                SELECT sm.motif_id, sm.run_config_id, sm.rate
                FROM main.selected_motifs sm
                JOIN export_motif_ids emi ON emi.id = sm.motif_id
                JOIN export_run_config_ids erc ON erc.id = sm.run_config_id;

                INSERT OR IGNORE INTO target.trained_nets (id, net_path, run_config_id, loss)
                SELECT tn.id, tn.net_path, tn.run_config_id, tn.loss
                FROM main.trained_nets tn
                JOIN export_trained_net_ids etn ON etn.id = tn.id;

                INSERT OR IGNORE INTO target.trained_nets_input (input_id, net_id)
                SELECT tni.input_id, tni.net_id
                FROM main.trained_nets_input tni
                JOIN export_trained_net_ids input_etn ON input_etn.id = tni.input_id
                JOIN export_trained_net_ids output_etn ON output_etn.id = tni.net_id;
                "#,
            )?;

            tx.commit()?;
            Ok::<_, anyhow::Error>(())
        })();

        conn.execute_batch("DETACH DATABASE target;")?;
        result
    }


    fn list_resource_set_resources(set_id: usize, conn: &Connection) -> anyhow::Result<Vec<Resource>> {
        let set_resources = {
            // because the running-mean mask (LeRunningMean) is order-sensitive, any non-deterministic resource ordering (or random selection) changes which pairs pass the mask, ORDER BY id is to fix it.
            let mut stmt = conn.prepare("SELECT rs.id as id, rs.import_id, rs.key, rs.length FROM resource_resources_set rrs LEFT JOIN resources rs ON rs.id=rrs.resource_id WHERE rrs.set_id=? ORDER BY id")?;

            let vec = stmt
                .query_map([set_id], |row| {
                    Ok(Resource::new(
                        row.get::<usize, u32>(0)?,
                        row.get::<usize, u32>(1)?,
                        row.get::<usize, String>(2)?,
                        row.get::<usize, u32>(3)?,
                    ))
                    // row.get::<usize, u32>(0)
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;

            vec
        };
        debug!("list_resource_set_resources: set_id={set_id} set_resources={set_resources:?}");
        Ok(set_resources)
    }

    /**
    defines global label order
    */
    pub fn list_labels(conn: &Connection, only_with_resources: bool) -> anyhow::Result<Vec<(usize, String)>> {
        let set_resources = {
            let mut stmt = if only_with_resources {
                conn.prepare("SELECT id, label FROM labels  l where exists (select 1 from labeled_resources where label_id=l.id) ORDER BY id;")?
            } else {
                conn.prepare("SELECT id, label FROM labels ORDER BY id")?
            };

            let vec = stmt
                .query_map([], |row| {
                    Ok((row.get::<usize, usize>(0)?, row.get::<usize, String>(1)?))
                })?
                .collect::<Result<Vec<(usize, String)>, Error>>()?;

            vec
        };
        Ok(set_resources)
    }

    pub fn init_external_writer_conn(conn: &mut Connection) -> anyhow::Result<()> {
        init_sqlite_writer(conn)?;
        Ok(())
    }

    pub fn save_import(descr: &str, t: &Transaction) -> anyhow::Result<u32> {
        t.execute("INSERT INTO imports (descr) VALUES (?)", params![descr])?;
        let import_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;
        Ok(import_id)
    }

    pub fn save_label(label: &str, t: &Transaction) -> anyhow::Result<u32> {
        t.execute("INSERT INTO labels (label) VALUES (?)", params![label])?;
        let label_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;
        Ok(label_id)
    }

    pub fn save_resources_labels(
        resource_stats_labels_probs: Vec<(String, u32, Vec<(String, f32)>)>,
        import_id: u32,
        t: &Transaction,
    ) -> anyhow::Result<Vec<ResourceId>> {
        debug!("save_resources_labels: resource_labels_probs={resource_stats_labels_probs:?}");
        let mut counts = HashMap::new();
        for e in resource_stats_labels_probs.iter() {
            *counts.entry(e.0.clone()).or_insert(0) +=1;
        }
        for e in counts {
            if e.1 > 1 {
                warn!("key {} has {} entries", e.0, e.1);
            }
        }

        let resources_len = resource_stats_labels_probs.len();
        let mut resource_labels_iter = resource_stats_labels_probs.into_iter().array_chunks::<500>();
        let mut resource_ids = Vec::with_capacity(resources_len);

        while let Some(resource_keys) = resource_labels_iter.next() {
            resource_ids.extend(Self::save_resources_labels_do_insert(resource_keys.into_iter(), import_id, t)?);
        }

        let remainder = resource_labels_iter.into_remainder();
        if remainder.len() != 0 {
            resource_ids.extend(Self::save_resources_labels_do_insert(remainder, import_id, t)?);
        }

        Ok(resource_ids)
    }

    pub fn save_groups_by_label_cnt(
        max_resources_in_set: usize,
        t: &Transaction,
    ) -> anyhow::Result<()> {
        let labels_cnt_labels_resources: HashMap<usize, HashMap<String, Vec<usize>>> = {
            let mut stmt = t.prepare(
                "SELECT r.id, GROUP_CONCAT(l.label, ','), count(lr.resource_id) labels_cnt FROM resources r JOIN labeled_resources lr ON lr.resource_id=r.id JOIN labels l ON l.id=lr.label_id GROUP BY resource_id ORDER BY labels_cnt"
            )?;
            let mut rows = stmt.query(())?;
            let mut labels_cnt_labels_resources: HashMap<usize, HashMap<String, Vec<usize>>> = HashMap::new();

            while let Some(label_row) = rows.next()? {
                let resource_id = label_row.get::<usize, usize>(0)?;
                let labels = label_row.get::<usize, String>(1)?;
                let labels_cnt = label_row.get::<usize, usize>(2)?;

                let cnt_labels = labels_cnt_labels_resources.entry(labels_cnt).or_insert_with(|| HashMap::new());
                let labels_resources = cnt_labels.entry(labels).or_insert_with(|| vec![]);

                labels_resources.push(resource_id);
            }
            labels_cnt_labels_resources
        };

        debug!("save_groups_by_label_cnt: labels_cnt_labels_resources.len={:?}", labels_cnt_labels_resources.len());
        let mut rng = rand::rng();

        let labels_ids: HashMap<String, usize> = Self::list_labels(t, true)?.into_iter().fold(HashMap::new(), |mut a, v| {
            a.insert(v.1, v.0);
            a
        });

        for (labels_cnt, labels_resources) in labels_cnt_labels_resources {
            let max_labels_resources = labels_resources.values().map(|r| r.len()).sum::<usize>();

            debug!("save_groups_by_label_cnt: labels_cnt={} max_labels_resources={}, max_resources_in_set={}, max_labels_resources/max_resources_in_set={}",labels_cnt,max_labels_resources,max_resources_in_set,max_labels_resources / max_resources_in_set);

            let num_groups = max(1, max_labels_resources / max_resources_in_set).ilog2();
            let groups_max_resources = (0..num_groups)
                .map(|i| max_resources_in_set * 2usize.pow(i as u32) as usize)
                .collect::<Vec<usize>>();

            let groups_max_ids = groups_max_resources
                .iter()
                .map(|max_r| {
                    (0..max_labels_resources)
                        .step_by(*max_r)
                        .map(|i| {
                            Self::save_resource_set_group(
                                format!("group {} labels/{}/{}-{}", labels_cnt, max_r, i, i + *max_r),
                                t,
                            )
                            .map(|g| (i, i + *max_r, g))
                        })
                        .collect::<anyhow::Result<Vec<(usize, usize, usize)>>>()
                })
                .collect::<anyhow::Result<Vec<Vec<(usize, usize, usize)>>>>()?;

            let group_all_id = Self::save_resource_set_group(format!("group {} labels/all", labels_cnt), t)?;

            debug!("save_groups_by_label_cnt:  num_groups={:?}, groups_max_resources={:?}", num_groups, groups_max_resources);

            for (label, mut resources) in labels_resources {
                resources.shuffle(&mut rng);

                let label_id = labels_ids.get(&label);

                debug!("save_groups_by_label_cnt: labels={} resources.len()={} label_id={:?}", label, resources.len(), label_id);

                if resources.len() > 10 {
                    for (group_id, _max_resources) in groups_max_ids.iter().zip(groups_max_resources.iter()) {
                        for (resources_0, resources_1, group_id) in group_id {
                            if *resources_0 < resources.len() - 1 {
                                Self::insert_resource_set_resources(
                                    t,
                                    labels_cnt,
                                    *group_id,
                                    format!("{}/{}", labels_cnt, label),
                                    label_id,
                                    &resources[*resources_0..min(resources.len() - 1, *resources_1)],
                                )?;
                            }
                        }
                    }
                }
                Self::insert_resource_set_resources(
                    t,
                    labels_cnt,
                    group_all_id,
                    format!("{}/{}", labels_cnt, label),
                    label_id,
                    resources.as_slice(),
                )?;
            }
        }

        Ok(())
    }

    ///used in hbac
    pub fn save_pure_sets(
        max_len: usize,
        only_unused: bool,
        t: &Transaction,
    ) -> anyhow::Result<()> {
        debug!("save_pure_sets: max_len={max_len} only_unused={only_unused}");

        let group_id = Self::save_resource_set_group(format!("pure group {}", max_len), t)?;

        let mut resource_resources_set_stmt = t.prepare(
            if only_unused {
                "INSERT INTO resource_resources_set (set_id, resource_id) SELECT ?, resource_id FROM (\
                    SELECT lr.resource_id FROM labeled_resources lr LEFT JOIN resource_resources_set rrs ON rrs.resource_id=lr.resource_id \
                    GROUP BY lr.resource_id HAVING sum(p)=max(p) AND lr.label_id=? AND rrs.set_id IS NULL ORDER BY lr.resource_id\
                ) LIMIT ?"
            } else {
                "INSERT INTO resource_resources_set (set_id, resource_id) SELECT ?, resource_id FROM (\
                    SELECT lr.resource_id FROM labeled_resources lr GROUP BY lr.resource_id HAVING sum(p)=max(p) AND lr.label_id=? ORDER BY lr.resource_id\
                ) LIMIT ?"
            }
        )?;

        let mut stmt = t.prepare("SELECT id, label FROM labels ORDER BY id")?;
        let mut rows = stmt.query(())?;

        while let Some(label_row) = rows.next()? {
            debug!("save_pure_sets: row={:?}", label_row);
            let label_id = label_row.get::<usize, usize>(0)?;
            let label = label_row.get::<usize, String>(1)?;
            t.execute(
                "INSERT INTO resource_set (descr, group_id, label_id) VALUES (?, ?, ?)",
                params![format!("{label}/{max_len}"), group_id, label_id],
            )?;

            let resource_set_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get::<usize, usize>(0))?;

            resource_resources_set_stmt.execute(params![resource_set_id, label_id, max_len])?;
        }

        Ok(())
    }

    pub fn save_all_sets(t: &Transaction) -> anyhow::Result<()> {
        let all_group_id = Self::save_resource_set_group("all group".into(), t)?;

        let mut resource_resources_set_all_stmt = t.prepare("INSERT INTO resource_resources_set (resource_id, set_id) SELECT  lr.resource_id, ? FROM labeled_resources lr WHERE lr.label_id=?")?;

        let mut stmt = t.prepare("SELECT id, label FROM labels ORDER BY id")?;
        let mut rows = stmt.query(())?;
        while let Some(label_row) = rows.next()? {
            debug!("save_all_sets: row={:?}", label_row);
            let label_id = label_row.get::<usize, usize>(0)?;
            let label = label_row.get::<usize, String>(1)?;


            t.execute(
                "INSERT INTO resource_set (descr, group_id, label_id) VALUES (?, ?, ?)",
                params![format!("all_{label}"), all_group_id, label_id],
            )?;

            let resource_set_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get::<usize, usize>(0))?;
            resource_resources_set_all_stmt.execute(params![resource_set_id, label_id])?;
        }

        Ok(())
    }


    /// builds train/validation/tests resources sets copying same resource in all sets if not enough resources available
    pub fn save_leaky_data_sets(
        prefix: &str,
        limit: LimitBy,
        min_length: Option<u32>,
        train_coeff: f32,
        valid_coeff: f32,
        min_p: f32,
        t: &Transaction,
    ) -> anyhow::Result<()> {

        let num_prefix = match limit {
            LimitBy::Count(n) | LimitBy::Length(n)=> {
                format!("{prefix}{n}")
            }
            LimitBy::None => format!("{prefix}")
        };

        if train_coeff < 0.0 || valid_coeff < 0.0 {
            return Err(anyhow::anyhow!("train_coeff and valid_coeff must be >= 0.0 (train_coeff={train_coeff} valid_coeff={valid_coeff})"));
        }

        if train_coeff + valid_coeff > 1.0 {
            return Err(anyhow::anyhow!("train_coeff + valid_coeff must be <= 1.0 (train_coeff={train_coeff} valid_coeff={valid_coeff})"));
        }

        Self::validate_min_probability(min_p)?;

        let (labels, mut resources_by_label) = Self::resource_ids_by_label(min_p, t, match limit {
            LimitBy::Count(_) | LimitBy::Length(_) => {
                Some(num_prefix.as_str())
            }
            LimitBy::None => {
                None
            }
        }, min_length)?;

        let train_group_id = Self::save_resource_set_group(format!("{num_prefix}_train"), t)?;
        let valid_group_id = Self::save_resource_set_group(format!("{num_prefix}_valid"), t)?;
        let test_group_id = Self::save_resource_set_group(format!("{num_prefix}_test"), t)?;

        let labels_cnt = labels.len();
        let mut rng = rand::rng();

        for (label_id, label) in labels {
            let (mut resources, lengths) = resources_by_label.remove(&label_id).expect(&format!("no resources for {label}"));
            debug!("save_data_sets: label={label} train_resources={resources:?}");
            if resources.is_empty() {
                warn!("no resources for {label}");
                continue;
            }

            let available = match limit {
                LimitBy::Count(count) => {
                    resources.len().min(count)
                }
                LimitBy::Length(max_length) => {
                    Self::limit_by_length(lengths, max_length).max(1)
                }
                LimitBy::None => {
                    resources.shuffle(&mut rng); //TODO add param? The resources are ordered by length in resource_ids_by_label
                    resources.len()
                }
            };

            let selected = &resources[..available];

            let train_len = (available as f32 * train_coeff).floor() as usize;
            let valid_len = (available as f32 * valid_coeff).floor() as usize;
            let test_len = available.saturating_sub(train_len + valid_len);

            debug!("save_leaky_data_sets: label={label} train_len={train_len} valid_len={valid_len} test_len={test_len}");

            let (train_resources, mut valid_resources, mut test_resources) = match available {
                1=> {
                    (selected.to_vec(), selected.to_vec(), selected.to_vec())
                },
                2=> {
                    (selected[..1].to_vec(), selected[1..].to_vec(), selected[1..].to_vec())
                },
                3=> {
                    (selected[..1].to_vec(), selected[1..2].to_vec(), selected[2..3].to_vec())
                },
                _ => {
                    (selected[..train_len].to_vec(), selected[train_len..train_len+valid_len].to_vec(), selected[train_len+valid_len..train_len+valid_len+test_len].to_vec())
                }
            };
            debug_assert!(!train_resources.is_empty());
            if valid_resources.is_empty() {
                valid_resources.push(train_resources[0].clone());
            }

            if test_resources.is_empty() {
                test_resources.push(valid_resources[0].clone());
            }

            debug!("save_leaky_data_sets: label={label} train_resources={train_resources:?}");
            debug!("save_leaky_data_sets: label={label} valid_resources={valid_resources:?}");
            debug!("save_leaky_data_sets: label={label} test_resources={test_resources:?}");


            debug_assert!(!valid_resources.is_empty());
            debug_assert!(!test_resources.is_empty());

            Self::insert_resource_set_resources(t, labels_cnt, train_group_id, format!("{num_prefix}_{label}_train"), Some(&label_id), train_resources.as_slice())?;
            Self::insert_resource_set_resources(t, labels_cnt, valid_group_id, format!("{num_prefix}_{label}_valid"), Some(&label_id), valid_resources.as_slice())?;
            Self::insert_resource_set_resources(t, labels_cnt, test_group_id, format!("{num_prefix}_{label}_test"), Some(&label_id), test_resources.as_slice())?;
        }

        Ok(())
    }

    fn limit_by_length(lengths: Vec<usize>, max_length: usize) -> usize {
        let mut length_sum = 0;
        for (i, resource_length) in lengths.iter().enumerate() {
            length_sum += resource_length;
            if max_length <= length_sum {
                return i;
            }
        }
        0
    }

    fn resource_ids_by_label(
        min_p: f32,
        t: &Transaction,
        not_in_prefix: Option<&str>,
        min_length: Option<u32>,
    ) -> anyhow::Result<(Vec<(usize, String)>, HashMap<usize, (Vec<usize>, Vec<usize>)>)> {
        Self::validate_min_probability(min_p)?;
        let labels = Self::list_labels(t, true)?;
        let mut resources_by_label = labels.iter().map(|(label_id, _)| (*label_id, (Vec::new(), Vec::new()))).collect::<HashMap<usize, (Vec<usize>, Vec<usize>)>>();

        let mut stmt = t.prepare(
            &["SELECT lr.label_id, lr.resource_id, r.length FROM labeled_resources lr LEFT JOIN  resources r on r.id=lr.resource_id \
                WHERE lr.p >= ? ",
                min_length.as_ref().map(|_| "AND r.length > ? ").unwrap_or(""),
                not_in_prefix.as_ref().map(|_|"AND lr.resource_id NOT IN (select rrs.resource_id from resource_resources_set rrs left join resource_set rs on rs.id=rrs.set_id where rs.descr like ?) ").unwrap_or(""),
                "ORDER BY lr.label_id, r.length",//TODO order by r.length"
            ].join("")
        )?;
        let mut params:Vec<Value> = vec![min_p.into()];
        if let Some(min_length) = min_length {
            params.push(min_length.into());
        }
        if let Some(not_in_prefix) = not_in_prefix {
            params.push(format!("{not_in_prefix}\\_%").into());
        }

        let mut rows = stmt.query(rusqlite::params_from_iter(params.into_iter()))?;
        while let Some(row) = rows.next()? {
            let label_id = row.get::<usize, usize>(0)?;
            let resource_id = row.get::<usize, usize>(1)?;
            let length = row.get::<usize, usize>(2)?;

            let ids_length = resources_by_label.entry(label_id).or_insert((vec![], vec![]));
            ids_length.0.push(resource_id);
            ids_length.1.push(length);
        }

        Ok((labels, resources_by_label))
    }

    pub fn validate_min_probability(min_p: f32) -> anyhow::Result<()> {
        if !(0.0..=1.0).contains(&min_p) {
            return Err(anyhow::anyhow!("min_p must be in range [0.0, 1.0] (min_p={min_p})"));
        }
        Ok(())
    }

    pub fn save_param_value(
        t: &Transaction,
        run_config_id: RunConfigId,
        param_name: &str,
        value: Val,
        label_id: Option<usize>,
    ) -> anyhow::Result<()> {
        match value {
            Val::Real(v) => {
                t.execute("INSERT INTO run_config_values (run_config_id, label_id, name, real_value) VALUES (?, ?, ?, ?)", params![run_config_id, label_id, param_name, v])?;
            }
            Val::Int(v) => {
                t.execute("INSERT INTO run_config_values (run_config_id, label_id, name, int_value) VALUES (?, ?, ?, ?)", params![run_config_id, label_id, param_name, v])?;
            }
            Val::UInt(v) => {
                t.execute("INSERT INTO run_config_values (run_config_id, label_id, name, int_value, unsigned) VALUES (?, ?, ?, ?, 1)", params![run_config_id, label_id, param_name, v])?;
            }
            Val::Str(v) => {
                t.execute("INSERT INTO run_config_values (run_config_id, label_id, name, str_value) VALUES (?, ?, ?, ?)", params![run_config_id, label_id, param_name, v])?;
            }
            Val::Bool(v) => {
                t.execute("INSERT INTO run_config_values (run_config_id, label_id, name, bool_value) VALUES (?, ?, ?, ?)", params![run_config_id, label_id, param_name, v])?;
            }

        }

        Ok(())
    }

    fn save_resource_set_group(descr: String, t: &Transaction) -> anyhow::Result<usize> {
        t.execute(
            "INSERT INTO resource_set_group (descr) VALUES (?)",
            params![descr],
        )?;
        let group_id = t.query_row("SELECT last_insert_rowid()", [], |row| {
            row.get::<usize, usize>(0)
        })?;
        Ok(group_id)
    }

    /**
     * label_id - for pure sets
     */
    fn insert_resource_set_resources(
        t: &Transaction,
        _labels_cnt: usize,
        group_id: usize,
        descr: String,
        label_id: Option<&usize>,
        resources: &[usize],
    ) -> anyhow::Result<()> {

        debug!("insert_resource_set_resources: group_id={group_id} descr={descr} label_id={label_id:?} resources={resources:?}");

        t.execute(
            "INSERT INTO resource_set (descr, group_id, label_id) VALUES (?, ?, ?)",
            params![descr, group_id, label_id], //labels?
        )?;

        let resource_set_id = t.query_row("SELECT last_insert_rowid()", [], |row| {
            row.get::<usize, usize>(0)
        })?;

        for resources_chunk in resources.chunks(500) {
            let (mut q, params) = resources_chunk.into_iter().fold(
                (
                    vec!["INSERT INTO resource_resources_set (set_id, resource_id) VALUES"],
                    vec![],
                ),
                |mut a, resources_id| {
                    a.0.push("(?, ?)");
                    a.0.push(",");
                    a.1.push(resource_set_id);
                    a.1.push(*resources_id);
                    a
                },
            );

            q.pop();

            t.execute(
                &q.into_iter().collect::<String>(),
                rusqlite::params_from_iter(params.into_iter()),
            )?;
        }

        Ok(())
    }

    fn get_input_run_config_ids_(
        conn: &Connection,
        run_config_id: RunConfigId,
        input_type: Option<&str>,
    ) -> anyhow::Result<Vec<RunConfigId>> {
        let mut stmt = conn.prepare("SELECT rci.input_run_config_id, rc.type FROM run_configs_input rci JOIN run_configs rc ON rc.id=rci.input_run_config_id WHERE run_config_id=?")?;

        let rows = stmt.query_map([run_config_id], |row| {
            Ok((row.get::<usize, RunConfigId>(0)?, row.get::<usize, String>(1)?))
        })?.collect::<rusqlite::Result<Vec<_>>>()?;
        let mut vec = Vec::new();
        for (id, config_type) in rows {
            if input_type.is_some_and(|expected| expected != config_type) {
                vec.extend(Self::get_input_run_config_ids_(conn, id, input_type)?);
            } else {
                vec.push(id);
            }
        }

        Ok(vec)
    }

    pub fn log_error<C: Debug>(ctx: C, t: &Transaction, e: Error) -> Error {
        error!("log_error: ctx={ctx:?} sqlite_error={:?}", e.sqlite_error());

        if let Ok(mut stmt) = t.prepare("PRAGMA foreign_key_check;") {
            if let Ok(rows) = stmt.query_map([], |row| {
                let table: String = row.get(0)?;
                let rowid: i64 = row.get(1)?;
                let parent: String = row.get(2)?;
                Ok((table, rowid, parent))
            }) {
                let mut rows_cnt = 0;
                for row_res in rows {
                    rows_cnt += 1;
                    match row_res {
                        Ok((table, rowid, parent)) => {
                            error!(
                                "Foreign key constraint failed: table='{}', rowid={}, parent='{}'",
                                table, rowid, parent
                            );
                        }
                        Err(e) => {
                            error!("Error reading foreign_key_check row: {}", e);
                        }
                    }
                }
                debug!("PRAGMA foreign_key_check: got {rows_cnt} rows");
            } else {
                error!("Error executing PRAGMA foreign_key_check");
            }
        } else {
            error!("Failed to prepare PRAGMA foreign_key_check statement");
        }
        e
    }

    pub fn _motif_meta(motif_id: u32, conn: &Connection) -> anyhow::Result<(i64, RunConfigId)> {
        // Preload the motif metadata to identify its channel and run.
        let (channel, run_config_id): (i64, RunConfigId) = conn.query_row(
            "SELECT channel_id, run_config_id FROM motifs WHERE id=?",
            [motif_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        Ok((channel, run_config_id))
    }

    fn insert_input(input_run_config_id: Option<Vec<RunConfigId>>, t: &Transaction, run_config_id: RunConfigId, sql: &str) -> anyhow::Result<()> {
        if let Some(input_run_config_ids) = input_run_config_id {
            let (mut q, params) = input_run_config_ids.into_iter().fold(
                (
                    vec![sql],
                    vec![],
                ),
                |(mut q, mut params): (Vec<&str>, Vec<Value>), v| {
                    q.push("(?, ?)");
                    q.push(",");

                    params.push(Value::Integer(run_config_id as i64));
                    params.push(Value::Integer(v as i64)); //input_run_config_id
                    //
                    (q, params)
                },
            );

            if params.len() > 0 {
                q.pop();

                let q_str = q.into_iter().collect::<String>();

                debug!("SqlStore::insert_input: params={params:?}");

                t.execute(&q_str, rusqlite::params_from_iter(params.into_iter()))?;
            }
        }


        Ok(())
    }
}

pub fn create_tables( conn: &mut Connection) -> anyhow::Result<()> {

    init_sqlite_writer(conn)?;

    let t = conn.transaction()?;

    t.execute(
        "CREATE TABLE IF NOT EXISTS imports (
                id          INTEGER PRIMARY KEY,
                descr       VARCHAR(256)
            )",
        (),
    )?;

    t.execute(
        //TODO UNIQUE(key) doesn't work - no error
        "CREATE TABLE IF NOT EXISTS resources (
                id          INTEGER PRIMARY KEY,
                key         VARCHAR(256),
                length      INTEGER NOT NULL,
                import_id   NOT NULL REFERENCES imports(id) ON DELETE CASCADE,
                UNIQUE(key) )",
        (),
    )?;

    t.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_resources_key ON resources (key);",
        (),
    )?;

    t.execute(
        "CREATE TABLE IF NOT EXISTS resource_set_group (
                id          INTEGER PRIMARY KEY,
                descr       VARCHAR(256)
            )",
        (),
    )?;

    t.execute(
        "CREATE TABLE IF NOT EXISTS resource_set (
                id          INTEGER PRIMARY KEY,
                group_id    REFERENCES resource_set_group(id) ON DELETE CASCADE,
                label_id    REFERENCES labels(id) ON DELETE CASCADE,
                descr       VARCHAR(256))",
        (),
    )?;

    t.execute(
        "CREATE TABLE IF NOT EXISTS resource_resources_set (
                resource_id  NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
                set_id       NOT NULL REFERENCES resource_set(id) ON DELETE CASCADE,
                PRIMARY KEY (resource_id, set_id)
            )",
        (),
    )?;


    t.execute(
        "CREATE TABLE IF NOT EXISTS runs (
                    id              INTEGER PRIMARY KEY,
                    descr           VARCHAR(256),
                    run_on          DATETIME DEFAULT CURRENT_TIMESTAMP
                )",
        (),
    )?;


    //descr - dist, dist etc
    t.execute(
        "CREATE TABLE IF NOT EXISTS run_configs (
                id          INTEGER PRIMARY KEY,
                type        VARCHAR(256),
                descr       VARCHAR(256),
                run_on      DATETIME DEFAULT CURRENT_TIMESTAMP,
                run_id      NOT NULL REFERENCES runs(id) ON DELETE CASCADE
            )",
        (),
    )?;

    t.execute(
        "CREATE TABLE IF NOT EXISTS run_configs_input (
                input_run_config_id     NOT NULL REFERENCES run_configs(id) ON DELETE CASCADE,
                run_config_id     NOT NULL REFERENCES run_configs(id) ON DELETE CASCADE,

                PRIMARY KEY (input_run_config_id, run_config_id)
            )",
        (),
    )?;

    t.execute("CREATE INDEX IF NOT EXISTS idx_run_configs_type ON run_configs (type);", ())?;
    t.execute("CREATE INDEX IF NOT EXISTS idx_run_configs_run_on ON run_configs (run_on);", ())?;
    t.execute("CREATE INDEX IF NOT EXISTS idx_run_configs_input_run_config_id ON run_configs_input (run_config_id);", ())?;

    //key - resource id  //TODO remove ord_id?
    t.execute(
        "CREATE TABLE IF NOT EXISTS run_resources (
                run_config_id     NOT NULL REFERENCES run_configs(id) ON DELETE CASCADE,
                ord_id     INTEGER NOT NULL,
                key        VARCHAR(256),
                PRIMARY KEY (run_config_id, ord_id)
            )",
        (),
    )?;

    t.execute(
        "CREATE TABLE IF NOT EXISTS run_config_values (
                id          INTEGER PRIMARY KEY,
                run_config_id      NOT NULL REFERENCES run_configs(id) ON DELETE CASCADE,
                name        VARCHAR(256),
                real_value  REAL,
                int_value   INTEGER,
                str_value   VARCHAR(256),
                bool_value  INTEGER,
                unsigned    INTEGER DEFAULT 0,
                label_id    REFERENCES labels(id) ON DELETE CASCADE --TODO remove?
                )",
        (),
    )?;
    t.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_run_config_values_name_run_config_id ON run_config_values (run_config_id, name, label_id);", ())?;



    t.execute(
        "CREATE TABLE IF NOT EXISTS trained_nets (
                id    INTEGER PRIMARY KEY,
                net_path  VARCHAR(1024),
                run_config_id  NOT NULL REFERENCES run_configs(id) ON DELETE CASCADE,
                loss    FLOAT NOT NULL,
                UNIQUE(net_path)
            )",
        (),
    )?;

    t.execute(
        "CREATE TABLE IF NOT EXISTS trained_nets_input (
                input_id  NOT NULL REFERENCES trained_nets(id) ON DELETE CASCADE,
                net_id NOT NULL REFERENCES trained_nets(id) ON DELETE CASCADE,

                PRIMARY KEY (input_id, net_id)
            )",
        (),
    )?;

    t.execute("commit", ())?;

    Ok(())
}


pub fn conds_to_q(input: &Vec<(String, SetCond)>) -> String {
    assert!(input.len() > 0);
    let inp_len = input.len() * 6 - 1;

    input
        .into_iter()
        .flat_map(|v| ["(l.label='", v.0.as_str(), "' and ", v.1.p(), "?)", " or "])
        .take(inp_len)
        .collect::<String>()
}

pub  struct MockStoreBuilder {

    store:MockStore
}

impl MockStoreBuilder {
    pub fn new() -> Self {
        Self {
            store: MockStore::new()
        }
    }

    pub fn with_get_input_run_config_ids<R:Into<HashMap<RunConfigId, Result<Vec<RunConfigId>, ThoError>>>>(mut self, results: R) -> Self{
        let map = results.into();
        self.store.expect_get_input_run_config_ids().returning(
            move |run_config_id, _input_type|{
                map.get(&run_config_id).cloned().map(|result| Ok(result?)).expect(&format!("mock is not configured to return input_run_config_ids for run_config_id {run_config_id}"))
            }
        );
        self
    }

    pub fn with_run_config_descr<R: Into<HashMap<RunConfigId, Result<RunConfigRow, ThoError>>>>(mut self, results: R) -> Self {
        let map = results.into();
        self.store.expect_run_config_descr().returning(
            move |run_config_id| {
                map.get(&run_config_id).cloned().map(|result| Ok(result?)).expect(&format!("mock is not configured to return run_config_descr for run_config_id {run_config_id}"))
            }
        );
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
                Err(anyhow::anyhow!(format!(
                    "mock is not configured to save_run_config for input_run_config_id={p_input_run_config_id:?} run_config_type={p_run_config_type} descr={p_descr} properties_vals={p_properties_vals:?}"
                )))
            }
        });
        self
    }

    pub fn with_save_trained_net(
        mut self,
        inputs: impl IntoIterator<Item = (RunConfigId, PathBuf, f32)> + Debug,
    ) -> Self {
        debug!("with_save_trained_net: inputs={inputs:?}");
        for (run_config_id, path, loss) in inputs {
            self.store
                .expect_save_trained_net()
                .with(eq(run_config_id), eq(path), eq(loss), eq(None))
                .times(1)
                .returning(move |_, _, _, _| Ok(run_config_id));
        }
        self
    }

    pub fn with_list_trained_nets<I>(
        mut self,
        inputs: I,
    ) -> Self
    where
        I: IntoIterator<Item = (Vec<RunConfigId>, Option<f32>, Option<usize>, Result<Vec<TrainedNet>, ThoError>)> + Debug,
    {
        let data = inputs.into_iter().collect::<Vec<_>>();
        debug!("with_list_trained_nets: inputs={data:?}");
        let data = Arc::new(data);
        self.store.expect_list_trained_nets().returning(
            move |run_config_ids, loss, limit| {
                data.iter()
                    .find(|(expected_run_config_ids, expected_loss, expected_limit, _)| expected_run_config_ids.as_slice() == run_config_ids.as_ref().as_slice() && *expected_loss == loss && *expected_limit == limit)
                    .map(|(_, _, _, result)| Ok(result.clone()?))
                    .unwrap_or_else(|| Err(anyhow::anyhow!("mock is not configured to return trained nets for run ids {run_config_ids:?} loss={loss:?} limit={limit:?}")))
            }
        );
        self
    }

    pub fn with_list_input_nets<I>(
        mut self,
        inputs: I,
    ) -> Self
    where
        I: IntoIterator<Item = (RunConfigId, Result<Vec<u32>, ThoError>)> + Debug,
    {
        let data = inputs.into_iter().collect::<Vec<_>>();
        debug!("with_list_trained_nets: inputs={data:?}");
        let data = Arc::new(data);
        self.store.expect_list_input_nets().returning(
            move |run_config_id| {
                data.iter()
                    .find(|(expected_run_config_id, _)| expected_run_config_id == &run_config_id)
                    .map(|(_, result)| Ok(result.clone()?))
                    .unwrap_or_else(|| Err(anyhow::anyhow!("with_list_input_nets is not configured to return  nets for run_config_id={run_config_id:?}")))
            }
        );
        self
    }

    pub fn with_list_run_config_values<R: Into<HashMap<RunConfigId, Result<HashMap<String, Val>, ThoError>>>>(
        mut self,
        results: R,
    ) -> Self {
        let map = results.into();
        self.store.expect_list_run_config_values()
            .returning(move |p_run_config_id| {
                map.get(&p_run_config_id)
                    .cloned().map(|result| Ok(result?))
                    .expect(&format!("mock is not configured to return run config values for run id {p_run_config_id}"))
            });

        self
    }

    pub fn with_list_output_run_config_ids<R:Into<HashMap<RunConfigId, Result<Vec<RunConfigId>, ThoError>>>>(
        mut self,
        results: R,
    ) -> Self {
        let map = results.into();
        self.store.expect_list_output_run_config_ids().returning(
            move |input_run_config_id| {
                map.get(&input_run_config_id)
                    .cloned().map(|result| Ok(result?))
                    .expect(&format!("mock is not configured to return output run ids for run id {input_run_config_id}"))
            },
        );
        self
    }

    pub fn with_list_run_config_ids_with_outputs<R: Into<HashMap<Vec<RunConfigId>, Result<Vec<RunConfigId>, ThoError>>>>(
        mut self,
        results: R,
    ) -> Self {
        let map = results.into();
        self.store.expect_list_run_config_ids_with_outputs().returning(
            move |input_run_config_ids| {
                map.get(&input_run_config_ids.to_vec())
                    .cloned().map(|result| Ok(result?))
                    .expect(&format!("mock is not configured to return run ids with outputs for input ids {input_run_config_ids:?}"))
            },
        );
        self
    }

    pub fn with_list_resources_by_file_stems(
        mut self,
        expected_stems: Vec<String>,
        result: Result<Vec<Resource>, ThoError>,
    ) -> Self {
        self.store
            .expect_list_resources_by_file_stems()
            .returning(move |requested| {
                let stems = requested.iter().map(|stem| stem.to_string()).collect::<Vec<_>>();
                if stems == expected_stems {
                    Ok(result.clone()?)
                } else {
                    Err(anyhow::anyhow!("Not found"))
                }
            });

        self
    }

    pub fn with_list_resource_set_group_sets<D:IntoIterator<Item=(&'static str, Result<Vec<(usize, String)>, ThoError>)>>(mut self, data:D) ->Self {
        let map :HashMap<&'static str, Result<Vec<(usize, String)>, ThoError>> = HashMap::from_iter(data);

        self.store.expect_list_resource_set_group_sets()
            .returning(move |group| {
                Ok(map.get(group).expect(&format!("no mock data for {group} configured in list_group_labeled_resources_sets")).clone()?)
            });

        self
    }

    pub fn with_list_group_labeled_resources_sets<D:IntoIterator<Item=(&'static str, Result<Vec<(String, String)>, ThoError>)>>(mut self, data:D) -> Self {
        let map :HashMap<&'static str, Result<Vec<(String, String)>, ThoError>> = HashMap::from_iter(data.into_iter());

    self.store.expect_list_group_labeled_resources_sets()
        .returning(move |group| {
            Ok(map.get(group).expect(&format!("no mock data for {group} configured in list_group_labeled_resources_sets")).clone()?)
        });

        self
    }

    pub fn with_list_resources_not_in_group<D:IntoIterator<Item=(&'static str, Result<Vec<Resource>, ThoError>)>>(mut self, data:D) -> Self {
        let map :HashMap<&'static str, Result<Vec<Resource>, ThoError>> = HashMap::from_iter(data.into_iter());

        self.store.expect_list_resources_not_in_group()
            .returning(move |group, _limit| {
                Ok(map.get(group).expect(&format!("no mock data for {group} configured in list_resource_set_resources_by_descr")).clone()?)
            });


        self
    }

    pub fn with_list_random_resources<D:IntoIterator<Item=(Option<u32>, Result<Vec<Resource>, ThoError>)>>(mut self, data:D) -> Self {
        let map :HashMap<Option<u32>, Result<Vec<Resource>, ThoError>> = HashMap::from_iter(data.into_iter());

        self.store.expect_list_random_resources()
            .returning(move |limit| {
                Ok(map.get(&limit).expect(&format!("no mock data for limit={limit:?} configured in list_random_resources")).clone()?)
            });


        self
    }

    pub fn with_list_resource_random_in_group<D:IntoIterator<Item=(&'static str, Result<Vec<(Resource, Vec<f32>)>, ThoError>)>>(mut self, data:D) -> Self {
        let map: HashMap<&'static str, Result<Vec<(Resource, Vec<f32>)>, ThoError>> = HashMap::from_iter(data.into_iter());

        self.store.expect_list_resource_random_in_group()
            .returning(move |group| {
                Ok(map.get(group).expect(&format!("no mock data for {group} configured in list_resource_random_in_group")).clone()?)
            });

        self
    }

    pub fn with_list_resource_set_resources_by_descr<D:IntoIterator<Item=(&'static str, Result<Vec<Resource>, ThoError>)>>(mut self, data:D) -> Self {
        let map :HashMap<&'static str, Result<Vec<Resource>, ThoError>> = HashMap::from_iter(data.into_iter());

        self.store.expect_list_resource_set_resources_by_descr()
        .returning(move |group| {
            Ok(map.get(group).expect(&format!("no mock data for {group} configured in list_resource_set_resources_by_descr")).clone()?)
        });


        self
    }

    pub fn build(self)->Box<dyn Store> {
        Box::new(self.store)
    }

}

#[cfg(test)]
mod error_tests {
    #[tokio::test]
    async fn test_nullable_config_values_preserve_conversion_errors() -> anyhow::Result<()> {
        let store = super::SqlStore::in_mem_(vec![]).await?;
        store.with_writer_conn("configuration fixture", |conn| {
            conn.execute_batch("CREATE TABLE run_config_values (run_config_id, name, real_value, int_value, str_value, bool_value, unsigned); INSERT INTO run_config_values VALUES (1, 'valid', NULL, 7, NULL, NULL, NULL);")?;
            Ok(())
        }).await?;
        assert_eq!(super::Store::list_run_config_values(&store, 1).await?["valid"], thoosa_shared::val::Val::Int(7));
        store.with_writer_conn("corrupt configuration fixture", |conn| {
            conn.execute("UPDATE run_config_values SET real_value='not a number'", [])?;
            Ok(())
        }).await?;
        let error = super::Store::list_run_config_values(&store, 1).await.unwrap_err();
        assert!(error.downcast_ref::<rusqlite::Error>().is_some());
        Ok(())
    }

    #[tokio::test]
    async fn test_bad_resource_row_returns_database_error() -> anyhow::Result<()> {
        let store = super::SqlStore::in_mem_(vec![]).await?;
        store.with_writer_conn("resource fixture", |conn| {
            conn.execute_batch("CREATE TABLE resources (id, import_id, key, length); INSERT INTO resources VALUES ('bad id', 1, 'resource', 10);")?;
            Ok(())
        }).await?;
        let error = thoosa_shared::resource::ResourceStore::list_random_resources(&store, None).await.unwrap_err();
        assert!(error.downcast_ref::<rusqlite::Error>().is_some());
        Ok(())
    }

    use super::SqlStore;

    #[tokio::test]
    async fn test_database_workers_preserve_anyhow_sources() -> anyhow::Result<()> {
        let store = SqlStore::in_mem_(vec![]).await?;
        // Exercise both real worker routes, including anyhow downcasting after operation context.
        let writer_error = store.with_writer_conn("write fixture", |_| {
            Err::<(), _>(std::io::Error::other("fixture failure").into())
        }).await.unwrap_err();
        let reader_error = store.with_reader_conn("read fixture", |_| {
            Err::<(), _>(std::io::Error::other("fixture failure").into())
        }).await.unwrap_err();
        for (error, operation) in [(writer_error, "write fixture"), (reader_error, "read fixture")] {
            assert!(error.downcast_ref::<std::io::Error>().is_some());
            assert_eq!(format!("{error:#}"), format!("{operation}: fixture failure"));
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_database_worker_panic_preserves_source_and_worker() -> anyhow::Result<()> {
        let store = SqlStore::in_mem_(vec![]).await?;
        let error = store.with_writer_conn::<(), _>("panic fixture", |_| panic!("fixture panic")).await.unwrap_err();
        assert!(error.downcast_ref::<crate::tokio_rusqlite::PanicInfo>().is_some());
        assert!(format!("{error:#}").contains("fixture panic"));
        assert_eq!(store.with_writer_conn("after panic", |_| Ok(42)).await?, 42);
        Ok(())
    }
}

#[cfg(test)]
mod session_tests {
    use super::{SqlStore, Store};

    #[tokio::test]
    async fn test_store_run_sessions_are_independent() -> anyhow::Result<()> {
        let first = SqlStore::in_mem_(vec![]).await?;
        let second = SqlStore::in_mem_(vec![]).await?;
        first.bind_run(41)?;
        second.bind_run(42)?;
        assert_eq!(first.run_id(), 41);
        assert_eq!(second.run_id(), 42);
        assert!(first.bind_run(42).is_err());
        first.bind_run(41)?;
        Ok(())
    }
}
