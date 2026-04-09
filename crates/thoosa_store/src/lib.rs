#![feature(iter_array_chunks)]
#![allow(
    unused_imports,
    unused_variables,
    unused_mut,
    dead_code,
    non_snake_case,
    redundant_semicolons,
    noop_method_call
)]
extern crate core;

pub mod fs_store;
mod tokio_rusqlite;

pub use rusqlite;

use bytemuck::{Pod, Zeroable};
use parking_lot::{Mutex, RwLock};
use std::array::IntoIter;
use std::backtrace::Backtrace;
use std::cmp::{max, min};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use async_trait::async_trait;
use futures_util::future::BoxFuture;
use futures_util::{FutureExt, StreamExt};

use log::{debug, error, info, trace, warn};
use mockall::{mock, PredicateBoxExt};
use mockall::predicate::eq;
use rand::{Rng, RngExt};
use rand::seq::SliceRandom;
use rusqlite::types::Value;
use rusqlite::{Connection, Error, OpenFlags, Transaction, params};
use rusqlite::trace::TraceEvent;
use thoosa_shared::resource::{Resource, ResourceId, ResourceStats, ResourceStore};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::val::Val;
use thoosa_shared::val::Val::UInt;
use thoosa_shared::*;
use thoosa_shared::traits::DataId;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct IdSlice(pub u32, pub ResourceId, pub u32); //id, tensor id, slice idx

impl PartialEq<Self> for IdSlice {
    fn eq(&self, other: &Self) -> bool {
        self.1.eq(&other.1) && self.2.eq(&other.2)
    }
}

impl Eq for IdSlice {}

impl Hash for IdSlice {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.1.hash(state);
        self.2.hash(state);
    }
}

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

    fn v(&self) -> f32 {
        match self {
            SetCond::Eq(v) => *v,
            SetCond::Gt(v) => *v,
            SetCond::Lt(v) => *v,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunConfigEntry {
    pub id: RunId,
    pub run_config_type: String,
    pub descr: String,
    pub run_on: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunConfigFilter {
    pub id: Option<RunId>,
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
    pub source_run_id: RunId,
    pub slices_cnt: usize,
    pub resources_cnt: usize,
}

#[async_trait]
pub trait Store: ResourceStore + Debug + Sync + Send {
    async fn save_import(&self, descr: String) -> Result<u32, ThoError>;
    async fn save_run_config<'a, 'b, 'c>(
        &self,
        input_run_id: Option<&'a [RunId]>,
        run_config_type: &'b str,
        descr: &'c str,
        _keys: &'c str,
        properties_vals: HashMap<String, Val>,
    ) -> Result<RunId, ThoError>;

    async fn get_input_run_ids<'a>(&self, run_id: RunId, input_type: Option<&'a str>) -> Result<Vec<RunId>, ThoError>;

    async fn list_input_run_ids(&self, ids: &[RunId]) -> Result<Vec<(RunId, RunId)>, ThoError>; //corr_run_id, run_id, label

    async fn list_run_ids_with_outputs(&self, input_run_ids: &[RunId]) -> Result<Vec<RunId>, ThoError>;

    async fn list_output_run_ids(&self, input_run_id: RunId) -> Result<Vec<RunId>, ThoError>; //corr_run_id, run_id, label

    async fn save_resources_set(
        &self,
        set: String,
        resource_keys: Vec<String>,
        group_id: Option<usize>,
    ) -> Result<usize, ThoError>;
    async fn save_resource_set_group(&self, descr: String) -> Result<usize, ThoError>;
    async fn save_resources_labels(
        &self,
        resource_stats_labels_probs: &HashMap<String, (ResourceStats, Vec<(String, f32)>)>,
        import_id: u32,
    ) -> Result<Vec<ResourceId>, ThoError>;
    async fn save_pure_sets(
        &self,
        max_len: usize,
        only_unused: bool,
    ) -> Result<(), ThoError>;
    async fn save_pure_balanced_sets(
        &self,
        prefix: &str,
        limit: Option<usize>,
        train_coeff: f32,
        valid_coeff: f32,
    ) -> Result<(), ThoError>;
    async fn save_set_by_labels(
        &self,
        conds: Vec<(String, SetCond)>,
        only_unused: bool,
        limit: usize,
    ) -> Result<(), ThoError>;
    async fn save_groups_by_label_cnt(
        &self,
        max_resources_in_set: usize,
    ) -> Result<(), ThoError>;
    async fn list_resources_by_ids(
        &self,
        file_stem: &Vec<ResourceId>,
    ) -> Result<Vec<Resource>, ThoError>;
    async fn list_resource_random_not_labeled(
        &self,
        label: &str,
        limit: u32,
    ) -> Result<Vec<Resource>, ThoError>;

    async fn save_label(&self, label: &str) -> Result<u32, ThoError>;

    async fn list_labels(&self) -> Result<Vec<String>, ThoError>;

    async fn save_params_value<'a>(&self, params: HashMap<String, Val>, run_id: RunId, label: Option<&'a str>) -> Result<(), ThoError>;
    async fn save_slices(&self, run_id: RunId, channel: i64, slices: Vec<(TensorIdSliceIdx, TensorIdSliceIdx, f32)>) -> Result<(), ThoError>;

    async fn save_groups_as_motifs_with_slices_matches(&self, run_id: RunId, channel: i64, groups: HashMap<u16, HashSet<IdSlice>>, path: PathBuf) -> Result<(), ThoError>;

    async fn save_groups_as_motifs(&self, run_id: RunId, channel: i64, groups: HashMap<u16, HashSet<IdSlice>>) -> Result<(), ThoError>;

    async fn add_slices_to_motif(&self, run_id: RunId, motif_id: u32, channel: i64, slices: HashSet<IdSlice>) -> Result<(), ThoError>;

    async fn list_motifs(&self, run_id: RunId) -> Result<Vec<u32>, ThoError>;
    // fn get_motif_meta<'a>(&self, motif_id: u32) -> BoxFuture<'a, Result<(usize, RunId), ThoError>>;
    async fn get_motif_meta(&self, motif_id: u32) -> Result<(usize, RunId), ThoError>;
    async fn list_motifs_slices(
        &self,
        run_id: RunId,
        select_motifs_run_id: Option<Vec<RunId>>,
    ) -> Result<HashMap<Resource, Vec<MotifSlice>>, ThoError>;

    async fn list_motif_slices(
        &self,
        motif_id: u32,
        select_motifs_run_id: Option<Vec<RunId>>,
        limit: Option<u32>,
        all_but_in_run_id: Option<(RunId, u32)>,
    ) -> Result<
        (
            i64,
            RunId,
            Vec<TensorIdSliceIdx>,
            HashMap<ResourceId, Resource>,
        ),
        ThoError,
    >;

    async fn list_random_slices_out_of_motif(
        &self,
        motif_id: u32,
        motif_width: u32,
        min_distance: Option<u32>,
        input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync>, desired_count: Option<usize>,
    ) -> Result<
        (
            i64,
            RunId,
            Vec<TensorIdSliceIdx>,
            HashMap<ResourceId, Resource>,
        ),
        ThoError,
    >;

    async fn list_random_slices_out_of_resources(
        &self,
        motif_id: u32,
        motif_width: u32,
        min_distance: Option<u32>,
        input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync>,
        desired_count: Option<usize>,
    ) -> Result<
        (
            i64,
            RunId,
            Vec<TensorIdSliceIdx>,
            HashMap<ResourceId, Resource>,
        ),
        ThoError,
    >;

    async fn list_run_resources(&self, run_id: RunId) -> Result<Vec<String>, ThoError>;

    async fn list_run_configs(&self) -> Result<Vec<RunConfigEntry>, ThoError>;
    async fn list_run_configs_page(&self, limit: usize, offset: usize) -> Result<Vec<RunConfigEntry>, ThoError>;
    async fn list_run_configs_by_ids(&self, ids: &[RunId]) -> Result<Vec<RunConfigEntry>, ThoError>;

    async fn list_run_configs_filtered(&self, filter: RunConfigFilter) -> Result<Vec<RunConfigEntry>, ThoError>;
    async fn list_run_config_types(&self) -> Result<Vec<String>, ThoError>;

    async fn list_run_config_values(&self, run_id: RunId) -> Result<HashMap<String, Val>, ThoError>;

    async fn list_run_motifs(&self, run_id: RunId) -> Result<Vec<RunMotifEntry>, ThoError>;

    async fn save_selected_motifs(&self, run_id: RunId, selected_motifs: Vec<(u32, f64)>) -> Result<RunId, ThoError>;

    async fn list_selected_motifs(&self, motif_selection_id: RunId, channels: u32) -> Result<Vec<Vec<(u32, f64)>>, ThoError>;

    async fn save_selected_net(&self, run_id: RunId, path: PathBuf, loss: f32) -> Result<(), ThoError>;

    async fn list_selected_nets(&self, run_id: RunId, loss: Option<f32>, limit: Option<usize>) -> Result<Vec<(PathBuf, f32)>, ThoError>;

    async fn run_config_descr(&self, corr_run_id: RunId) -> Result<Vec<(i64, String)>, ThoError>;
    #[cfg(test)]
    fn conn(&self) -> &parking_lot::Mutex<Connection>; //parking_lot:: for mockall
}

mock! {
    pub Store {}

    #[async_trait]
    impl Store for Store {
        async fn save_import(&self, descr: String) -> Result<u32, ThoError>;
        async fn save_run_config<'a, 'b, 'c>(
            &self,
            input_run_id: Option<&'a [RunId]>,
            run_config_type: &'b str,
            descr: &'c str,
            _keys: &'c str,
            properties_vals: HashMap<String, Val>,
        ) -> Result<RunId, ThoError>;
        async fn get_input_run_ids<'a>(&self,run_id: RunId,input_type: Option<&'a str>,) -> Result<Vec<RunId>, ThoError>;
        async fn list_input_run_ids(&self, ids: &[RunId]) -> Result<Vec<(RunId, RunId)>, ThoError>;
        async fn list_run_ids_with_outputs(&self, input_run_ids: &[RunId]) -> Result<Vec<RunId>, ThoError>;
        async fn list_output_run_ids(&self, input_run_id: RunId) -> Result<Vec<RunId>, ThoError>;

        async fn save_resources_set(
            &self,
            set: String,
            resource_keys: Vec<String>,
            group_id: Option<usize>,
        ) -> Result<usize, ThoError>;
        async fn save_resource_set_group(&self, descr: String) -> Result<usize, ThoError>;
        async fn save_resources_labels(
            &self,
            resource_stats_labels_probs: &HashMap<String, (ResourceStats, Vec<(String, f32)>)>,
            import_id: u32,
        ) -> Result<Vec<ResourceId>, ThoError>;
        async fn save_pure_sets(
            &self,
            max_len: usize,
            only_unused: bool,
        ) -> Result<(), ThoError>;
        async fn save_pure_balanced_sets(
            &self,
            prefix: &str,
            limit: Option<usize>,
            train_coeff: f32,
            valid_coeff: f32,
        ) -> Result<(), ThoError>;
        async fn save_set_by_labels(
            &self,
            conds: Vec<(String, SetCond)>,
            only_unused: bool,
            limit: usize,
        ) -> Result<(), ThoError>;
        async fn save_groups_by_label_cnt(
            &self,
            max_resources_in_set: usize,
        ) -> Result<(), ThoError>;
        async fn list_resources_by_ids(
            &self,
            file_stem: &Vec<ResourceId>,
        ) -> Result<Vec<Resource>, ThoError>;
        async fn list_resource_random_not_labeled(
            &self,
            label: &str,
            limit: u32,
        ) -> Result<Vec<Resource>, ThoError>;
        async fn save_label(&self, label: &str) -> Result<u32, ThoError>;
        async fn list_labels(&self) -> Result<Vec<String>, ThoError>;
        async fn save_params_value<'a>(
            &self,
            params: HashMap<String, Val>,
            run_id: RunId,
            label: Option<&'a str>,
        ) -> Result<(), ThoError>;
        async fn save_slices(
            &self,
            run_id: RunId,
            channel: i64,
            slices: Vec<(TensorIdSliceIdx, TensorIdSliceIdx, f32)>,
        ) -> Result<(), ThoError>;
        async fn save_groups_as_motifs_with_slices_matches(
            &self,
            run_id: RunId,
            channel: i64,
            groups: HashMap<u16, HashSet<IdSlice>>,
            path: PathBuf,
        ) -> Result<(), ThoError>;
        async fn save_groups_as_motifs(
            &self,
            run_id: RunId,
            channel: i64,
            groups: HashMap<u16, HashSet<IdSlice>>,
        ) -> Result<(), ThoError>;

        async fn add_slices_to_motif(
            &self,
            run_id: RunId,
            motif_id: u32,
            channel: i64,
            groups: HashSet<IdSlice>,
        ) -> Result<(), ThoError>;

        async fn list_motifs(&self, run_id: RunId) -> Result<Vec<u32>, ThoError>;
        async fn list_motifs_slices(
            &self,
            run_id: RunId,
            select_motifs_run_id: Option<Vec<RunId>>,
        ) -> Result<HashMap<Resource, Vec<MotifSlice>>, ThoError>;

        // fn get_motif_meta<'a>(&self, motif_id: u32) -> BoxFuture<'a, Result<(usize, RunId), ThoError>>;
        async fn get_motif_meta(&self, motif_id: u32) -> Result<(usize, RunId), ThoError>;

        async fn list_motif_slices(
            &self,
            motif_id: u32,
            select_motifs_run_id: Option<Vec<RunId>>,
            limit: Option<u32>,
            all_but_in_run_id: Option<(RunId, u32)>,
        ) -> Result<
            (
                i64,
                RunId,
                Vec<TensorIdSliceIdx>,
                HashMap<ResourceId, Resource>,
            ),
            ThoError,
        >;
        async fn list_random_slices_out_of_motif(
            &self,
            motif_id: u32,
            motif_width: u32,
            min_distance: Option<u32>,
            input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync>,
            desired_count:Option<usize>
        ) -> Result<
            (
                i64,
                RunId,
                Vec<TensorIdSliceIdx>,
                HashMap<ResourceId, Resource>,
            ),
            ThoError,
        >;
        async fn list_random_slices_out_of_resources(
            &self,
            motif_id: u32,
            motif_width: u32,
            min_distance: Option<u32>,
            input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync>,
            desired_count: Option<usize>
        ) -> Result<
            (
                i64,
                RunId,
                Vec<TensorIdSliceIdx>,
                HashMap<ResourceId, Resource>,
            ),
            ThoError,
        >;
        async fn list_run_resources(&self, run_id: RunId) -> Result<Vec<String>, ThoError>;
        async fn list_run_configs(&self) -> Result<Vec<RunConfigEntry>, ThoError>;
        async fn list_run_configs_page(&self, limit: usize, offset: usize) -> Result<Vec<RunConfigEntry>, ThoError>;
        async fn list_run_configs_by_ids(&self, ids: &[RunId]) -> Result<Vec<RunConfigEntry>, ThoError>;
        async fn list_run_configs_filtered(&self, filter: RunConfigFilter) -> Result<Vec<RunConfigEntry>, ThoError>;
        async fn list_run_config_types(&self) -> Result<Vec<String>, ThoError>;
        async fn list_run_config_values(&self, run_id: RunId) -> Result<HashMap<String, Val>, ThoError>;
        async fn list_run_motifs(&self, run_id: RunId) -> Result<Vec<RunMotifEntry>, ThoError>;
        async fn save_selected_motifs(
            &self,
            run_id: RunId,
            selected_motifs: Vec<(u32, f64)>,
        ) -> Result<RunId, ThoError>;

        async fn list_selected_motifs(
            &self,
            motif_selection_id: RunId,
            channels: u32,
        ) -> Result<Vec<Vec<(u32, f64)>>, ThoError>;
        async fn save_selected_net(
            &self,
            run_id: RunId,
            path: PathBuf,
            loss: f32,
        ) -> Result<(), ThoError>;
        async fn list_selected_nets(
            &self,
            run_id: RunId,
            loss: Option<f32>,
            limit: Option<usize>,
        ) -> Result<Vec<(PathBuf, f32)>, ThoError>;
        async fn run_config_descr(&self, corr_run_id: RunId) -> Result<Vec<(i64, String)>, ThoError>;
        #[cfg(test)]
        fn conn(&self) -> &parking_lot::Mutex<Connection>;
    }

    #[async_trait]
    impl ResourceStore for Store {
        async fn list_resources_by_file_stems<'a>(
            &self,
            file_stems: &'a [&'a str],
        ) -> Result<Vec<Resource>, ThoError>;

        async fn list_resource_set_resources_by_descr(
            &self,
            descr: &str,
        ) -> Result<Vec<Resource>, ThoError>;

        async fn list_resource_set_group_sets(
            &self,
            group_descr: &str,
        ) -> Result<Vec<(usize, String)>, ThoError>;

        async fn list_group_labeled_resources_sets(
            &self,
            group_decr: &str,
        ) -> Result<Vec<(String, String)>, ThoError>;

        async fn list_resource_set_resources(&self, set_id: usize) -> Result<Vec<Resource>, ThoError>;

        async fn list_resource_random_in_group(
            &self,
            group_descr: &str,
        ) -> Result<Vec<(Resource, Vec<f32>)>, ThoError>;

        async fn list_resources_not_in_group(
            &self,
            group_descr: &str,
            limit:u32
        ) -> Result<Vec<Resource>, ThoError>;

        async fn list_random_resources(
            &self,
            limit: Option<u32>
        ) -> Result<Vec<Resource>, ThoError>;

    }
}

impl fmt::Debug for MockStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MockStore").finish()
    }
}

impl Debug for SqlMotifTrackerStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("").finish()
    }
}

#[derive(Debug)]
pub struct SqlStore {
    #[cfg(test)]
    conn: Mutex<Connection>,
    writer: tokio_rusqlite::Connection,
    readers: Arc<Vec<tokio_rusqlite::Connection>>,
    rr: AtomicUsize,
}

const SQL_STORE_READERS_CNT: usize = 8;
static IN_MEM_DB_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
struct RandomSliceResourceCtx {
    resource: Resource,
    adjusted_len: i64,
    target_positions: Vec<i64>,
}

#[async_trait]
impl ResourceStore for SqlStore {
    async fn list_resource_set_resources_by_descr(
        &self,
        descr: &str,
    ) -> Result<Vec<Resource>, ThoError> {
        let descr = descr.to_string();
        self.with_reader_conn("list_resource_set_resources_by_descr", move |conn| {
            let r = Self::list_resource_set_resources(
                conn.query_row("SELECT id FROM resource_set WHERE descr=?", [descr.as_str()], |row| row.get(0))?,
                conn,
            )?;
            debug!("list_resource_set_resources_by_descr: descr={descr} r={r:?}");
            Ok(r)
        })
        .await
    }
    async fn list_resources_by_file_stems<'a>(
        &self,
        file_stems: &'a [&'a str],
    ) -> Result<Vec<Resource>, ThoError> {
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
                .map(|v| v.unwrap())
                .collect()
            };

            Ok(resources)
        })
        .await
    }

    async fn list_resource_set_group_sets(
        &self,
        group_descr: &str,
    ) -> Result<Vec<(usize, String)>, ThoError> {
        let group_descr = group_descr.to_string();
        self.with_reader_conn("list_resource_set_group_sets", move |conn| {
            let group_sets = {
                let mut stmt = conn.prepare("SELECT rs.id, rs.descr FROM resource_set rs LEFT JOIN resource_set_group rsg ON rsg.id=rs.group_id WHERE rsg.descr=?")?;

                stmt.query_map([group_descr.as_str()], |row| {
                    Ok((
                        row.get::<usize, usize>(0).expect("row 0 in list_resource_set_group_sets"),
                        row.get::<usize, String>(1).expect("row 1 in list_resource_set_group_sets"),
                    ))
                })?
                .map(|v| v.unwrap())
                .collect()
            };

            Ok(group_sets)
        })
        .await
    }

    ///returns Vec<(label, resource_set_descr)>
    async fn list_group_labeled_resources_sets(
        &self,
        group_descr: &str,
    ) -> Result<Vec<(String, String)>, ThoError> {
        let group_descr = group_descr.to_string();
        debug!("list_group_labeled_resources_sets: group_descr={group_descr}");
        self.with_reader_conn("list_group_labeled_resources_sets", move |conn| {
            let labeled_resources_sets = {
                let mut stmt = conn.prepare("SELECT l.label, rs.descr FROM resource_set_group rsg JOIN resource_set rs ON rs.group_id=rsg.id JOIN labels l ON l.id=rs.label_id WHERE rsg.descr=? ORDER BY label")?;

                stmt.query_map([group_descr.as_str()], |row| {
                    Ok((
                        row.get::<usize, String>(0).expect("row 0 in list_group_labeled_resources_sets"),
                        row.get::<usize, String>(1).expect("row 1 in list_group_labeled_resources_sets"),
                    ))
                })?
                .map(|v| v.unwrap())
                .collect()
            };

            debug!("list_group_labeled_resources_sets: labeled_resources_sets={labeled_resources_sets:?}");
            Ok(labeled_resources_sets)
        })
        .await
    }

    async fn list_resource_set_resources(&self, set_id: usize) -> Result<Vec<Resource>, ThoError> {
        self.with_reader_conn("list_resource_set_resources", move |conn| {
            let set_resources = Self::list_resource_set_resources(set_id, conn)?;
            Ok(set_resources)
        })
        .await
    }

    async fn list_resource_random_in_group(
        &self,
        group_descr: &str,
    ) -> Result<Vec<(Resource, Vec<f32>)>, ThoError> {
        let group_descr = group_descr.to_string();
        self.with_reader_conn("list_resource_random_in_group", move |conn| {
            let labels = Self::list_labels(conn)?;
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

                let labels_p_vec = labels_p.split(",")
                    .map(|label_p| {
                        let mut label_p_v = label_p.split("/");
                        (label_p_v.next().expect("label").into(), label_p_v.next().expect("p").parse::<f32>().expect("parse"))
                    })
                    .fold(HashMap::new(), |mut a: HashMap<String, f32>, v| {
                        a.insert(v.0, v.1);
                        a
                    });

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
            })?.map(|v| v.unwrap()).collect();

            debug!("list_resource_random_in_group: r.len={:?}", r.len());
            Ok(r)
        })
        .await
    }

    async fn list_resources_not_in_group(
        &self,
        group_descr: &str,
        limit:u32
    ) -> Result<Vec<Resource>, ThoError> {
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
                .map(|v| v.unwrap())
                .collect()
            };

            Ok(group_sets)
        })
        .await
    }

    async fn list_random_resources(
        &self,
        limit: Option<u32>
    ) -> Result<Vec<Resource>, ThoError> {
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
                .map(|v| v.unwrap())
                .collect()
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
                .map(|v| v.unwrap())
                .collect()
            };

            Ok(resources)
        })
        .await
    }


}
#[async_trait]
//TODO move out all prepare stmts
impl Store for SqlStore {
    async fn save_import(&self, descr: String) -> Result<u32, ThoError> {
        self.with_writer_conn("save_import", move |conn| {
            let t = conn.transaction()?;
            t.execute("INSERT INTO imports (descr) VALUES (?)", params![descr])?;
            let import_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;
            t.commit()?;
            Ok(import_id)
        })
        .await
    }

    //TODO resources - save serialized resources config - run_resources is getting to big
    async fn save_run_config<'a, 'b, 'c>(
        &self,
        input_run_id: Option<&'a [RunId]>,
        run_config_type: &'b str,
        descr: &'c str,
        _keys: &'c str,
        properties_vals: HashMap<String, Val>,
    ) -> Result<RunId, ThoError> {
        let input_run_id = input_run_id.map(|v| v.to_vec());
        let run_config_type = run_config_type.to_string();
        let descr = descr.to_string();
        debug!("SqlStore::save_run_config: input_run_id={input_run_id:?} run_config_type={run_config_type} descr={descr}");
        self.with_writer_conn("save_run_config", move |conn| {
            let t = conn.transaction().map_err(|e| ThoError::from_error("start sqlite transaction", e))?;
            t.execute("INSERT INTO run_configs (type, descr) VALUES (?, ?)", params![run_config_type, descr])?;

            let run_id = t.query_row("SELECT last_insert_rowid()", [], |row| {
                row.get::<usize, RunId>(0)
            })?;

        /*if keys.len() > 0 {
            let (mut q, params) = keys.into_iter().enumerate().fold(
                //TODO  resources.id instead key  + group or set + create one-time set if resources passed as vec
                (vec!["INSERT INTO run_resources (run_id, ord_id, key) VALUES"], vec![]), |mut a: (Vec<&str>, Vec<Value>), (ord_id, v)| {
                    a.0.push("(?, ?, ?)");
                    a.0.push(", ");
                    a.1.push(Value::Integer(run_id as i64));
                    a.1.push(Value::Integer(ord_id as i64));
                    a.1.push(Value::Text(v.to_string()));
                    a
                });

            q.pop();

            t.execute(&q.into_iter().collect::<String>(), rusqlite::params_from_iter(params.into_iter()))?;
        }*/

        if let Some(input_run_ids) = input_run_id {
            let (mut q, params) = input_run_ids.into_iter().fold(
                (
                    vec!["INSERT INTO run_configs_input (run_id, input_run_id) VALUES"],
                    vec![],
                ),
                |(mut q, mut params): (Vec<&str>, Vec<Value>), v| {
                    q.push("(?, ?)");
                    q.push(",");

                    params.push(Value::Integer(run_id as i64));
                    params.push(Value::Integer(v as i64)); //input_run_id
                    //
                    (q, params)
                },
            );

            if params.len() > 0 {
                q.pop();

                let q_str = q.into_iter().collect::<String>();

                debug!("SqlStore::save_run_config: params={params:?}");

                t.execute(&q_str, rusqlite::params_from_iter(params.into_iter()))?;
            }
        }

        if properties_vals.len() > 0 {
            for property_val in properties_vals.into_iter() {
                Self::save_param_value(&t, run_id, &property_val.0, property_val.1, None)?;
            }
        }

        t.commit()?;
            Ok(run_id as RunId)
        })
        .await
    }

    async fn get_input_run_ids<'a>(
        &self,
        run_id: RunId,
        input_type: Option<&'a str>,
    ) -> Result<Vec<RunId>, ThoError> {
        let input_type = input_type.map(|v| v.to_string());
        debug!("SqlStore::get_input_run_ids: run_id={run_id} input_type={input_type:?}");
        self.with_reader_conn("get_input_run_ids", move |conn| {
            let input_run_ids = Self::get_input_run_ids_(conn, run_id, input_type.as_deref())?;
            debug!("get_input_run_ids: input_run_ids={input_run_ids:?}");
            Ok(input_run_ids)
        })
        .await
    }

    /// returns - Vec<run_id, input_run_id>
    async fn list_input_run_ids(&self, ids: &[RunId]) -> Result<Vec<(RunId, RunId)>, ThoError> {
        assert!(!ids.is_empty());
        let ids = ids.to_vec();
        self.with_reader_conn("list_input_run_ids", move |conn| {
            let mut id_input_id = vec![];
            for ids_chunk in ids.chunks(500) {
                let (mut q, params) = ids_chunk.iter().fold(
                    (
                        vec!["SELECT run_id, input_run_id FROM run_configs_input WHERE run_id IN ("],
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
                        Ok((row.get::<usize, RunId>(0)?, row.get::<usize, RunId>(1)?))
                    })?
                    .collect::<Result<Vec<_>, _>>()?;
                id_input_id.extend(chunk_rows);
            }
            Ok(id_input_id)
        })
        .await
    }

    async fn list_run_ids_with_outputs(&self, input_run_ids: &[RunId]) -> Result<Vec<RunId>, ThoError> {
        if input_run_ids.is_empty() {
            return Ok(vec![]);
        }

        let input_run_ids = input_run_ids.to_vec();
        self.with_reader_conn("list_run_ids_with_outputs", move |conn| {
            let mut run_ids_with_outputs = vec![];

            for ids_chunk in input_run_ids.chunks(500) {
                let (mut q, params) = ids_chunk.iter().fold(
                    (
                        vec!["SELECT DISTINCT input_run_id FROM run_configs_input WHERE input_run_id IN ("],
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
                    .query_map(rusqlite::params_from_iter(params.into_iter()), |row| row.get::<usize, RunId>(0))?
                    .collect::<Result<Vec<_>, _>>()?;
                run_ids_with_outputs.extend(chunk_rows);
            }

            Ok(run_ids_with_outputs)
        })
        .await
    }


    //TODO remove?
    async fn save_params_value<'a>(
        &self,
        params: HashMap<String, Val>,
        run_id: RunId,
        label: Option<&'a str>,
    ) -> Result<(), ThoError> {
        let label = label.map(|v| v.to_string());
        self.with_writer_conn("save_params_value", move |conn| {
            let t = conn.transaction()?;
            let label_id = t.query_row("SELECT id FROM labels WHERE label=?", [label.as_deref()], |row| row.get(0)).ok();

            for property_val in params.into_iter() {
                Self::save_param_value(&t, run_id, &property_val.0, property_val.1, label_id)?;
            }

            t.commit()?;
            Ok(())
        })
        .await
    }

    async fn save_label(&self, label: &str) -> Result<u32, ThoError> {
        let label = label.to_string();
        self.with_writer_conn("save_label", move |conn| {
            let t = conn.transaction()?;
            t.execute("INSERT INTO labels (label) VALUES (?)", params![label])?;
            let label_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;
            t.commit()?;
            Ok(label_id)
        })
        .await
    }

    async fn list_labels(&self) -> Result<Vec<String>, ThoError> {
        self.with_reader_conn("list_labels", move |conn| {
            let labels = Self::list_labels(conn).map(|v| v.into_iter().map(|id_label| id_label.1).collect())?;
            Ok(labels)
        })
        .await
    }

    //TODO folds to util fn
    async fn save_resources_labels(
        &self,
        resource_stats_labels_probs: &HashMap<String, (ResourceStats, Vec<(String, f32)>)>,
        import_id: u32,
    ) -> Result<Vec<ResourceId>, ThoError> {
        let resource_stats_labels_probs = resource_stats_labels_probs
            .iter()
            .map(|(resource_key, (stats, labels))| (resource_key.clone(), stats.length(), labels.clone()))
            .collect::<Vec<_>>();
        self.with_writer_conn("save_resources_labels", move |conn| {
            let t = conn.transaction()?;
            debug!("save_resources_labels: resource_labels_probs.len()={}", resource_stats_labels_probs.len());

            let resources_len = resource_stats_labels_probs.len();
            let mut resource_labels_iter = resource_stats_labels_probs.into_iter().array_chunks::<500>();
            let mut resource_ids = Vec::with_capacity(resources_len);

            while let Some(resource_keys) = resource_labels_iter.next() {
                resource_ids.extend(Self::save_resources_labels_do_insert(resource_keys.into_iter(), import_id, &t)?);
            }

            let remainder = resource_labels_iter.into_remainder();

            if remainder.len() != 0 {
            // if !remainder.is_empty() {
                resource_ids.extend(Self::save_resources_labels_do_insert(remainder, import_id, &t)?);
            }


            t.commit()?;
            Ok(resource_ids)
        })
        .await
    }

    //TODO used in tests only, remove?
    async fn save_resources_set(
        &self,
        descr: String,
        resource_keys: Vec<String>,
        group_id: Option<usize>,
    ) -> Result<usize, ThoError> {
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

    async fn save_resource_set_group(&self, descr: String) -> Result<usize, ThoError> {
        self.with_writer_conn("save_resource_set_group", move |conn| {
            let mut t = conn.transaction()?;
            let group_id = Self::save_resource_set_group(descr, &mut t)?;
            t.commit()?;
            Ok(group_id)
        })
        .await
    }

    async fn save_groups_by_label_cnt(
        &self,
        max_resources_in_set: usize,
    ) -> Result<(), ThoError> {
        self.with_writer_conn("save_groups_by_label_cnt", move |conn| {
            let t = conn.transaction()?;

        {
            let labels_cnt_labels_resources: HashMap<usize, HashMap<String, Vec<usize>>> = {
                let mut stmt = t.prepare(
                    "SELECT r.id, GROUP_CONCAT(l.label, ','), count(lr.resource_id) labels_cnt FROM resources r JOIN labeled_resources lr ON lr.resource_id=r.id JOIN labels l ON l.id=lr.label_id GROUP BY resource_id ORDER BY labels_cnt"
                )?;
                let mut rows = stmt.query(())?;
                let mut labels_cnt_labels_resources: HashMap<usize, HashMap<String, Vec<usize>>> =
                    HashMap::new(); //labels_cnt -> Labels->Resources. labels_cnt==1 - pure resources

                while let Some(label_row) = rows.next()? {
                    let resource_id = label_row.get::<usize, usize>(0)?;
                    let labels = label_row.get::<usize, String>(1)?;
                    let labels_cnt = label_row.get::<usize, usize>(2)?;

                    let cnt_labels = labels_cnt_labels_resources
                        .entry(labels_cnt)
                        .or_insert_with(|| HashMap::new());
                    let labels_resources = cnt_labels.entry(labels).or_insert_with(|| vec![]);

                    labels_resources.push(resource_id);
                }
                labels_cnt_labels_resources
            };

            debug!("save_groups_by_label_cnt: labels_cnt_labels_resources.len={:?}",labels_cnt_labels_resources.len());
            let mut rng = rand::rng();

            let labels_ids: HashMap<String, usize> =
                Self::list_labels(&t)?
                    .into_iter()
                    .fold(HashMap::new(), |mut a, v| {
                        a.insert(v.1, v.0);
                        a
                    });

            //labels_cnt number of labels a resource has - the fewer labels the more chances that resource is classified correctly
            for (labels_cnt, labels_resources) in labels_cnt_labels_resources {
                let max_labels_resources = labels_resources.values().map(|r| r.len()).sum::<usize>();

                debug!(
                    "save_groups_by_label_cnt: labels_cnt={} max_labels_resources={}, max_resources_in_set={}, max_labels_resources/max_resources_in_set={}",
                    labels_cnt,
                    max_labels_resources,
                    max_resources_in_set,
                    max_labels_resources / max_resources_in_set
                );

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
                                    format!(
                                        "group {} labels/{}/{}-{}",
                                        labels_cnt,
                                        max_r,
                                        i,
                                        i + *max_r
                                    ),
                                    &t,
                                )
                                .map(|g| (i, i + *max_r, g))
                            })
                            .collect::<Result<Vec<(usize, usize, usize)>, ThoError>>()
                    })
                    .collect::<Result<Vec<Vec<(usize, usize, usize)>>, ThoError>>()?;

                let group_all_id =
                    Self::save_resource_set_group(format!("group {} labels/all", labels_cnt), &t)?;

                debug!(
                    "save_groups_by_label_cnt:  num_groups={:?}, groups_max_resources={:?}",
                    num_groups, groups_max_resources
                );

                for (label, mut resources) in labels_resources {
                    resources.shuffle(&mut rng);

                    let label_id = labels_ids.get(&label); //if more than one label - label_id is None and Null in the db

                    debug!(
                        "save_groups_by_label_cnt: labels={} resources.len()={} label_id={:?}",
                        label,
                        resources.len(),
                        label_id
                    );

                    if resources.len() > 10 {
                        //TODO min resources

                        for (group_id, _max_resources) in
                            groups_max_ids.iter().zip(groups_max_resources.iter())
                        {
                            for (resources_0, resources_1, group_id) in group_id {
                                if *resources_0 < resources.len() - 1 {
                                    Self::insert_resource_set_resources(
                                        &t,
                                        labels_cnt,
                                        *group_id,
                                        format!("{}/{}", labels_cnt, label),
                                        label_id,
                                        &resources
                                            [*resources_0..min(resources.len() - 1, *resources_1)],
                                    )?;
                                }
                            }
                        }
                    }
                    Self::insert_resource_set_resources(
                        &t,
                        labels_cnt,
                        group_all_id,
                        format!("{}/{}", labels_cnt, label),
                        label_id,
                        resources.as_slice(),
                    )?;
                }
            }
        }
            t.commit()?;
            Ok(())
        })
        .await
    }

    async fn save_pure_sets(
        &self,
        max_len: usize,
        only_unused: bool,
    ) -> Result<(), ThoError> {
        self.with_writer_conn("save_pure_sets", move |conn| {
        let t = conn.transaction()?;

        let group_id = Self::save_resource_set_group(format!("pure group {}", max_len), &t)?;

        {
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

            //TODO replace with Self::list_labels
            let mut stmt = t.prepare("SELECT id, label FROM labels ORDER BY id")?;
            let mut rows = stmt.query(())?;
            while let Some(label_row) = rows.next()? {
                debug!("row={:?}", label_row);
                let label_id = label_row.get::<usize, usize>(0)?;
                t.execute(
                    "INSERT INTO resource_set (descr, group_id, label_id) VALUES (?, ?, ?)",
                    params![
                        format!("{}/{}", label_row.get::<usize, String>(1)?, max_len),
                        group_id,
                        label_id
                    ],
                )?;

                let resource_set_id = t.query_row("SELECT last_insert_rowid()", [], |row| {
                    row.get::<usize, usize>(0)
                })?;

                resource_resources_set_stmt.execute(params![resource_set_id, label_id, max_len])?;
            }
        }

        t.commit()?;
        Ok(())
        })
        .await
    }

    async fn save_pure_balanced_sets(
        &self,
        prefix: &str,
        limit: Option<usize>,
        train_coeff: f32,
        valid_coeff: f32,
    ) -> Result<(), ThoError> {
        let prefix = prefix.to_string();
        self.with_writer_conn("save_pure_balanced_sets", move |conn| {
        let t = conn.transaction()?;

        if train_coeff < 0.0 || valid_coeff < 0.0 {
            return Err(ThoError::new(format!("train_coeff and valid_coeff must be >= 0.0 (train_coeff={train_coeff} valid_coeff={valid_coeff})")));
        }

        if train_coeff + valid_coeff > 1.0 {
            return Err(ThoError::new(format!("train_coeff + valid_coeff must be <= 1.0 (train_coeff={train_coeff} valid_coeff={valid_coeff})")));
        }

        let labels = Self::list_labels(&t)?;
        let mut resources_by_label = labels
            .iter()
            .map(|(label_id, _)| (*label_id, Vec::new()))
            .collect::<HashMap<usize, Vec<usize>>>();

        {
            let mut stmt = t.prepare(
                "SELECT lr.label_id, lr.resource_id FROM labeled_resources lr GROUP BY lr.resource_id HAVING count(*)=1 ORDER BY lr.label_id, lr.resource_id",
            )?;
            let mut rows = stmt.query(())?;
            while let Some(row) = rows.next()? {
                let label_id = row.get::<usize, usize>(0)?;
                let resource_id = row.get::<usize, usize>(1)?;
                resources_by_label
                    .entry(label_id)
                    .or_insert_with(Vec::new)
                    .push(resource_id);
            }
        }

        let min_pure_resources = labels
            .iter()
            .map(|(label_id, _)| {
                resources_by_label
                    .get(label_id)
                    .map(|resources| resources.len())
                    .unwrap_or(0)
            })
            .min()
            .unwrap_or(0);

        if let Some(limit_val) = limit {
            if limit_val > min_pure_resources {
                return Err(ThoError::new(format!("limit {limit_val} exceeds smallest number of pure resources {min_pure_resources}")));
            }
        }

        let target_count = limit.unwrap_or(min_pure_resources);
        let train_count = ((target_count as f64) * (train_coeff as f64)).floor() as usize;
        let valid_count = ((target_count as f64) * (valid_coeff as f64)).floor() as usize;
        let test_count = target_count.saturating_sub(train_count + valid_count);

        debug!("save_pure_balanced_sets: prefix={prefix} target_count={target_count} train_count={train_count} valid_count={valid_count} test_count={test_count}");

        let train_group_id = Self::save_resource_set_group(format!("{prefix}_balanced_group_train"), &t)?;

        let valid_group_id = Self::save_resource_set_group(format!("{prefix}_balanced_group_validation"), &t)?;

        let test_group_id = Self::save_resource_set_group(format!("{prefix}_balanced_group_test"), &t)?;

        let labels_cnt = labels.len();
        let mut rng = rand::rng();

        for (label_id, label) in labels {
            let mut resources = resources_by_label.remove(&label_id).unwrap_or_default();
            resources.shuffle(&mut rng);
            let selected = resources
                .into_iter()
                .take(target_count)
                .collect::<Vec<usize>>();

            let (train_resources, rest) = selected.split_at(min(train_count, selected.len()));
            let (valid_resources, test_resources) = rest.split_at(min(valid_count, rest.len()));

            Self::insert_resource_set_resources(
                &t,
                labels_cnt,
                train_group_id,
                format!("{prefix}_balanced_set_{label}"),
                Some(&label_id),
                train_resources,
            )?;
            Self::insert_resource_set_resources(
                &t,
                labels_cnt,
                valid_group_id,
                format!("{prefix}_balanced_set_{label}"),
                Some(&label_id),
                valid_resources,
            )?;
            Self::insert_resource_set_resources(
                &t,
                labels_cnt,
                test_group_id,
                format!("{prefix}_balanced_set_{label}"),
                Some(&label_id),
                test_resources,
            )?;
        }

        t.commit()?;
        Ok(())
        })
        .await
    }

    async fn list_resource_random_not_labeled(
        &self,
        label: &str,
        limit: u32,
    ) -> Result<Vec<Resource>, ThoError> {
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
            })?.map(|v| v.unwrap()).collect();

            Ok(r)
        })
        .await
    }

    async fn list_resources_by_ids(
        &self,
        resource_ids: &Vec<ResourceId>,
    ) -> Result<Vec<Resource>, ThoError> {
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
                .map(|v| v.unwrap())
                .collect()
            };

            Ok(resources)
        })
        .await
    }

    //TODO not used, remove?
    async fn save_set_by_labels(
        &self,
        conds: Vec<(String, SetCond)>,
        _only_unused: bool,
        limit: usize,
    ) -> Result<(), ThoError> {
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
                ThoError::new(format!("no resources for set on {:?}", conds));
            }
        }
        Ok(t.commit()?)
        })
        .await
    }
    //TODO remove
    async fn save_slices(
        &self,
        run_id: RunId,
        channel: i64,
        slices: Vec<(TensorIdSliceIdx, TensorIdSliceIdx, f32)>,
    ) -> Result<(), ThoError> {
        self.with_writer_conn("save_slices", move |conn| {
        let t = conn.transaction()?;

        assert!(slices.len() > 0);

        for slices_chunk in slices.chunks(500) {
            let (mut q, params) = slices_chunk.into_iter().fold(
                (vec!["INSERT INTO slice_matches (run_id, channel_id, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1, dist) VALUES"], vec![]), |mut a: (Vec<&str>, Vec<Value>), v| {
                    a.0.push("(?, ?, ?, ?, ?, ?, ?)");
                    a.0.push(", ");

                    a.1.push(Value::Integer(run_id as i64));
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
        run_id: RunId,
        channel: i64,
        groups: HashMap<u16, HashSet<IdSlice>>,
    ) -> Result<(), ThoError> {
        self.with_writer_conn("save_groups_as_motifs", move |conn| {
        let t = conn.transaction()?;

        debug!("save_groups_as_motifs: run_id={run_id:?}, channel={channel:?}");

        {
            for (group, slices) in groups {
                if slices.is_empty() {
                    continue
                }
                let motifs_inserted = t
                    .execute(
                        "INSERT INTO motifs (group_ord, run_id, channel_id) VALUES (?, ?, ?)",
                        [group as usize, run_id as usize, channel as usize],
                    )
                    .map_err(|e| Self::log_error((run_id, channel, group), &t, e))?;

                let motif_id: i64 =
                    t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;

                debug!("save_groups_as_motifs: group={group:?}, run_id={run_id:?} motif_id={motif_id} motifs_inserted={motifs_inserted} slices.len={}", slices.len());
                trace!("save_groups_as_motifs: slices={slices:?} ");

                let mut slices_chunks = slices.into_iter().array_chunks::<500>();
                while let Some(chunk) = slices_chunks.next() {
                    Self::insert_slices(&t, run_id, channel, motif_id, chunk.into_iter())?
                }
                let remainder = slices_chunks.into_remainder();
                if remainder.len() != 0 {
                // if !remainder.is_empty() {
                    Self::insert_slices(&t, run_id, channel, motif_id, remainder)?
                }
            }
        }

        t.commit()?;

        Ok(())
        })
        .await
    }

    async fn add_slices_to_motif(
        &self,
        run_id: RunId,
        motif_id: u32,
        channel: i64,
        slices: HashSet<IdSlice>,
    ) -> Result<(), ThoError> {
        self.with_writer_conn("add_slices_to_motif", move |conn| {
        let t = conn.transaction()?;

        debug!("save_slices_groups_as_motifs_from_file: run_id={run_id:?}, channel={channel:?} slices.len={}", slices.len());

        {
            let mut slices_chunks = slices.into_iter().array_chunks::<500>();
            while let Some(chunk) = slices_chunks.next() {
                Self::insert_slices(&t, run_id, channel, motif_id as i64, chunk.into_iter())?
            }
            let remainder = slices_chunks.into_remainder();
            if remainder.len() != 0 {
            // if !remainder.is_empty() {
                Self::insert_slices(&t, run_id, channel, motif_id as i64, remainder)?
            }
        }
        t.commit()?;
        Ok(())
        })
        .await
    }

    async fn save_groups_as_motifs_with_slices_matches(
        &self,
        run_id: RunId,
        channel: i64,
        groups: HashMap<u16, HashSet<IdSlice>>,
        path: PathBuf,
    ) -> Result<(), ThoError> {
        self.with_writer_conn("save_groups_as_motifs_with_slices_matches", move |conn| {
        let t = conn.transaction()?;

        debug!("save_slices_groups_as_motifs_from_tmp_db: run_id={run_id:?}, channel={channel:?} path={}", path.to_string_lossy());

        {
            t.execute(
                format!(
                    "ATTACH DATABASE '{}' AS slices_{run_id}_{channel}_tmp",
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
                    "INSERT INTO motifs (group_ord, run_id, channel_id) VALUES (?, ?, ?)",
                    [group as usize, run_id as usize, channel as usize],
                )?; //TODO group not needed?

                let motif_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;

                debug!("save_slices_groups_as_motifs_from_tmp_db: group={group:?}, run_id={run_id:?} motif_id={motif_id} motifs_inserted={motifs_inserted} slices={slices:?} ");

                //each match appears twice in slices but will result in single raw returned from _tmp.slice_matches, should be faster than getting slices into set on slice_matches.id (IdSlice.0)
                let mut slices_chunks = slices.into_iter().array_chunks::<500>();

                while let Some(chunk) = slices_chunks.next() {
                    Self::insert_slices_matches_chunk_from_tmp(
                        &t,
                        run_id,
                        channel,
                        motif_id,
                        chunk.into_iter(),
                    )?
                }

                let remainder = slices_chunks.into_remainder();
                if remainder.len() != 0 {
                // if !remainder.is_empty() {
                    Self::insert_slices_matches_chunk_from_tmp(
                        &t, run_id, channel, motif_id, remainder,
                    )?
                }
            }
        }

        t.commit()?;

        let t = conn.transaction()?;
        t.execute(
            format!("DETACH DATABASE slices_{run_id}_{channel}_tmp").as_str(),
            [],
        )?;
        t.commit()?;

        Ok(())
        })
        .await
    }

    async fn get_motif_meta(&self, motif_id: u32) -> Result<(usize, RunId), ThoError> {
        self.with_reader_conn("get_motif_meta", move |conn| {
                let (channel, run_id) = Self::_motif_meta(motif_id, conn)?;
                debug!("get_motif_meta: motif_id={motif_id} channel={channel} run_id={run_id}");
                Ok::<_, ThoError>((channel as usize, run_id))
            })
            .await
    }

    async fn list_motifs(&self, run_id: RunId) -> Result<Vec<u32>, ThoError> {
        self.with_reader_conn("list_motifs", move |conn| {
            let motif_ids = {
                let mut stmt = conn.prepare("SELECT id FROM motifs WHERE run_id=?")?;
                stmt.query_map([run_id], |row| Ok(row.get::<usize, u32>(0)?))?
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
        run_id: RunId,
        select_motifs_run_id: Option<Vec<RunId>>,
    ) -> Result<HashMap<Resource, Vec<MotifSlice>>, ThoError> {
        self.with_reader_conn("list_motifs_slices", move |conn| {
            let slice_run_ids = select_motifs_run_id.map(|mut ids| {
                ids.push(run_id);
                ids.sort();
                ids.dedup();
                ids
            });

            let slice_filter_clause = slice_run_ids
                .as_ref()
                .map(|ids| format!(" AND s.run_id IN ({})", ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ")))
                .unwrap_or_default();

            let resources_motifs = {
                let mut stmt = conn.prepare(
                    &format!(
                        "SELECT m.id, m.channel_id, r.id, r.import_id, r.key, s.slice_idx, r.length \
                         FROM motifs m \
                         JOIN slices s ON s.motif_id=m.id \
                         JOIN resources r on r.id=s.resource_id \
                         WHERE m.run_id=?{}",
                        slice_filter_clause
                    ),
                )?;

                let mut params: Vec<Value> = vec![Value::Integer(run_id as i64)];
                if let Some(ids) = &slice_run_ids {
                    params.extend(ids.iter().map(|slice_run_id| Value::Integer(*slice_run_id as i64)));
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


    ///returns channel, find_motifs_run_id, slices, resources map
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="list_motif_slices"))]
    async fn list_motif_slices(
        &self,
        motif_id: u32,
        select_motifs_run_ids: Option<Vec<RunId>>, //None - all slices, empty - only slices from find_motifs, Some -  find_motifs + select_motif runs
        limit: Option<u32>,
        all_but_in_run_id: Option<(RunId, u32/*channel*/)>, //TODO remove - load random slices from same channel/resources
    ) -> Result<
        (
            i64,
            RunId,
            Vec<TensorIdSliceIdx>,
            HashMap<ResourceId, Resource>,
        ),
        ThoError,
    > {
        debug!("list_motif_slices: motif_id={motif_id} select_motifs_run_id={select_motifs_run_ids:?} limit={limit:?} all_but_in_run_id={all_but_in_run_id:?}");
        self.with_reader_conn("list_motif_slices", move |conn| {

            let (slice_filter_clause, slice_filter_params) = if all_but_in_run_id.is_none() {
                match select_motifs_run_ids.as_ref() {
                    None => (String::new(), Vec::new()),
                    Some(ids_raw) => {
                        let mut ids = ids_raw.clone();
                        ids.sort();
                        ids.dedup();
                        if ids.is_empty() {
                            (" AND s.run_id IN (m.run_id)".to_string(), Vec::new())
                        } else {
                            let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
                            (format!(" AND s.run_id IN (m.run_id, {placeholders})"), ids)
                        }
                    }
                }
            } else {
                (String::new(), Vec::new())
            };

            let channel_slices = {
                let mut query = String::from(
                    "SELECT m.channel_id, m.run_id, s.run_id, r.id,  r.import_id, r.key, s.slice_idx \
                        FROM motifs m \
                        LEFT JOIN slices s on s.motif_id=m.id \
                        LEFT JOIN resources r on s.resource_id=r.id \
                        WHERE ",
                );

                let params: Vec<u32> = match all_but_in_run_id {
                    Some((_run_id, _channel_id)) => {
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

                let (slices, channel_set, run_id, resources_map) = stmt
                    .query_map(rusqlite::params_from_iter(params.iter()), |row| {
                        Ok((
                            row.get::<usize, i64>(0)?,
                            row.get::<usize, RunId>(1)?,
                            row.get::<usize, RunId>(2)?,
                            row.get(3)?,
                            row.get(4)?,
                            row.get(5)?,
                            row.get(6)?,
                        ))
                    })?
                    .fold(
                        Ok((HashSet::new(), HashSet::new(), None, HashMap::new())),
                        |a: Result<(HashSet<TensorIdSliceIdx>, HashSet<i64>, Option<RunId>, HashMap<ResourceId, Resource>), ThoError>,
                         v: Result<(i64, RunId, RunId, u32, u32, String, u32), Error>| {
                            let (channel_id, motif_run_id, _slice_run_id, resource_id, import_id, key, slice_idx) = v?;
                            let (mut a, mut channel_set, mut run_id, mut resources_map) = a?;

                            if run_id.is_none() {
                                run_id.replace(motif_run_id);
                            } else {
                                assert_eq!(run_id, Some(motif_run_id));
                            }

                            a.insert(TensorIdSliceIdx(resource_id, slice_idx));
                            resources_map.insert(resource_id, Resource::new(resource_id, import_id, key, 0));
                            channel_set.insert(channel_id);

                            Ok((a, channel_set, run_id, resources_map))
                        },
                    )?;

                if all_but_in_run_id.is_none() {
                    assert_eq!(channel_set.len(), 1, "channel_set={channel_set:?} motif_id={motif_id}");
                    (channel_set.into_iter().next().expect("channel"), run_id.expect("no run id"), slices.into_iter().collect(), resources_map)
                } else {
                    let (run_id, channel_id) = all_but_in_run_id.expect("all_but_in_run_id");
                    (channel_id as i64, run_id, slices.into_iter().collect(), resources_map)
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
    ) -> Result<
        (
            i64,
            RunId,
            Vec<TensorIdSliceIdx>,
            HashMap<ResourceId, Resource>,
        ),
        ThoError,
    > {
        debug!("list_random_slices_out_of_motif: motif_id={motif_id} motif_width={motif_width} min_distance={min_distance:?}");

        assert!(motif_width > 0, "motif_width must be greater than zero");
        let input_len_adjuster_in_motif = Arc::clone(&input_len_adjuster);
        let (channel, run_id, mut all_candidates, mut resources_map, desired_count) = self.with_reader_conn("list_random_motif_slices", move |conn| {
            let (channel, run_id) = Self::_motif_meta(motif_id, conn)?;
            let (target_positions_by_res, num_motif_slices) = Self::load_motif_target_positions(motif_id, conn)?;
            let desired_count = desired_count.unwrap_or(num_motif_slices);
            if desired_count == 0 {
                return Ok((channel, run_id, vec![], HashMap::new(), desired_count));
            }

            let contexts =
                Self::load_motif_resource_contexts(motif_id, input_len_adjuster_in_motif.as_ref(), conn, target_positions_by_res)?;

            if contexts.is_empty() {
                return Err(ThoError::new(format!("no resources found for channel {channel}, run_id={run_id}")));
            }

            let (all_candidates, resources_map) = Self::build_random_slice_candidates(&contexts, motif_width, min_distance);
            Ok((channel, run_id, all_candidates, resources_map, desired_count))
        }).await?;

        if all_candidates.len() < desired_count {
            let missing_count = desired_count - all_candidates.len();
            debug!("list_random_slices_out_of_motif: motif_id={motif_id} fell back to other resources for {missing_count} slices");

            let (fallback_channel, fallback_run_id, mut fallback_candidates, fallback_resources_map) = self
                .list_random_slices_out_of_resources(
                    motif_id,
                    motif_width,
                    min_distance,
                    input_len_adjuster,
                    Some(missing_count),
                )
                .await?;

            assert_eq!(channel, fallback_channel, "fallback channel differs for motif_id={motif_id}");
            assert_eq!(run_id, fallback_run_id, "fallback run_id differs for motif_id={motif_id}");

            all_candidates.append(&mut fallback_candidates);
            resources_map.extend(fallback_resources_map);
        }

        Self::shuffle_and_limit_candidates(&mut all_candidates, desired_count);
        Ok((channel, run_id, all_candidates, resources_map))
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SqlStore::list_random_slices_out_of_resources"))]
    async fn list_random_slices_out_of_resources(
        &self,
        motif_id: u32,
        motif_width: u32,
        min_distance: Option<u32>,
        input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync>,
        desired_count: Option<usize>,
    ) -> Result<
        (
            i64,
            RunId,
            Vec<TensorIdSliceIdx>,
            HashMap<ResourceId, Resource>,
        ),
        ThoError,
    > {
        debug!("list_random_slices_out_of_resources: motif_id={motif_id} motif_width={motif_width} min_distance={min_distance:?}");

        assert!(motif_width > 0, "motif_width must be greater than zero");

        self.with_reader_conn("list_random_slices_out_of_resources", move |conn| {
            let (channel, run_id) = Self::_motif_meta(motif_id, conn)?;
            let desired_count = match desired_count {
                Some(desired_count) => desired_count,
                None => Self::count_motif_slices(motif_id, conn)?,
            };

            if desired_count == 0 {
                return Ok((channel, run_id, vec![], HashMap::new()));
            }

            let contexts = Self::load_non_motif_resource_contexts(motif_id, run_id, channel, input_len_adjuster.as_ref(), conn)?;
            if contexts.is_empty() {
                return Ok((channel, run_id, vec![], HashMap::new()));
            }

            let (mut all_candidates, resources_map) = Self::build_random_slice_candidates(&contexts, motif_width, min_distance);
            Self::shuffle_and_limit_candidates(&mut all_candidates, desired_count);

            Ok((channel, run_id, all_candidates, resources_map))
        }).await
    }

    async fn list_run_resources(&self, run_id: RunId) -> Result<Vec<String>, ThoError> {
        self.with_reader_conn("list_run_resources", move |conn| {
            let result = {
                let count: i64 = conn.query_row("SELECT count(*) FROM run_resources WHERE run_id=?", [run_id], |row| row.get(0))?;
                debug!("list_run_resources: count={}", count);
                let mut stmt = conn.prepare("SELECT ord_id, key FROM run_resources WHERE run_id=? ORDER BY ord_id")?;

                stmt.query_map([run_id], |row| Ok((row.get::<usize, i64>(0)?, row.get::<usize, String>(1)?)))?
                    .enumerate()
                    .map(|(i, ord_key)| {
                        ord_key
                            .map(|(ord, key)| {
                                assert!(ord == i as i64);
                                key
                            })
                            .map_err(ThoError::from)
                    })
                    .collect::<Result<Vec<String>, ThoError>>()
            };

            debug!("list_run_resources: result.is_ok={}", result.is_ok());
            result
        })
        .await
    }

    async fn list_run_configs(&self) -> Result<Vec<RunConfigEntry>, ThoError> {
        self.with_reader_conn("list_run_configs", move |conn| {
            let run_configs = {
                let mut stmt = conn.prepare("SELECT id, type, descr, run_on FROM run_configs ORDER BY id DESC")?;
                stmt.query_map([], |row| {
                    Ok(RunConfigEntry {
                        id: row.get::<usize, RunId>(0)?,
                        run_config_type: row.get::<usize, String>(1)?,
                        descr: row.get::<usize, String>(2)?,
                        run_on: row.get::<usize, String>(3)?,
                    })
                })?
                .map(|v| v.map_err(ThoError::from))
                .collect::<Result<Vec<RunConfigEntry>, ThoError>>()?
            };

            Ok(run_configs)
        })
        .await
    }

    async fn list_run_configs_page(&self, limit: usize, offset: usize) -> Result<Vec<RunConfigEntry>, ThoError> {
        self.with_reader_conn("list_run_configs_page", move |conn| {
            let run_configs = {
                let mut stmt = conn.prepare("SELECT id, type, descr, run_on FROM run_configs ORDER BY id DESC LIMIT ? OFFSET ?")?;
                stmt.query_map([limit as i64, offset as i64], |row| {
                    Ok(RunConfigEntry {
                        id: row.get::<usize, RunId>(0)?,
                        run_config_type: row.get::<usize, String>(1)?,
                        descr: row.get::<usize, String>(2)?,
                        run_on: row.get::<usize, String>(3)?,
                    })
                })?
                .map(|v| v.map_err(ThoError::from))
                .collect::<Result<Vec<RunConfigEntry>, ThoError>>()?
            };

            Ok(run_configs)
        })
        .await
    }

    async fn list_run_configs_by_ids(&self, ids: &[RunId]) -> Result<Vec<RunConfigEntry>, ThoError> {
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
                            id: row.get::<usize, RunId>(0)?,
                            run_config_type: row.get::<usize, String>(1)?,
                            descr: row.get::<usize, String>(2)?,
                            run_on: row.get::<usize, String>(3)?,
                        })
                    })?
                    .map(|v| v.map_err(ThoError::from))
                    .collect::<Result<Vec<RunConfigEntry>, ThoError>>()?;
                run_configs.extend(chunk_rows);
            }

            run_configs.sort_by(|a, b| b.id.cmp(&a.id));
            Ok(run_configs)
        })
        .await
    }

    async fn list_run_configs_filtered(&self, filter: RunConfigFilter) -> Result<Vec<RunConfigEntry>, ThoError> {
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
                "WITH RECURSIVE matched(id) AS (SELECT rc.id FROM run_configs rc WHERE {}), visible(id) AS (SELECT id FROM matched UNION SELECT rci.input_run_id FROM run_configs_input rci JOIN visible v ON rci.run_id = v.id) SELECT rc.id, rc.type, rc.descr, rc.run_on FROM run_configs rc JOIN visible v ON v.id = rc.id ORDER BY rc.id DESC",
                where_sql,
            );

            let mut stmt = conn.prepare(query.as_str())?;
            stmt.query_map(rusqlite::params_from_iter(params.into_iter()), |row| {
                Ok(RunConfigEntry {
                    id: row.get::<usize, RunId>(0)?,
                    run_config_type: row.get::<usize, String>(1)?,
                    descr: row.get::<usize, String>(2)?,
                    run_on: row.get::<usize, String>(3)?,
                })
            })?
            .map(|value| value.map_err(ThoError::from))
            .collect::<Result<Vec<RunConfigEntry>, ThoError>>()
        })
        .await
    }

    async fn list_run_config_types(&self) -> Result<Vec<String>, ThoError> {
        self.with_reader_conn("list_run_config_types", move |conn| {
            let run_config_types = {
                let mut stmt = conn.prepare("SELECT DISTINCT type FROM run_configs ORDER BY type ASC")?;
                stmt.query_map([], |row| row.get::<usize, String>(0))?
                    .map(|v| v.map_err(ThoError::from))
                    .collect::<Result<Vec<String>, ThoError>>()?
            };

            Ok(run_config_types)
        })
        .await
    }

    async fn list_run_config_values(&self, run_id: RunId) -> Result<HashMap<String, Val>, ThoError> {
        self.with_reader_conn("list_run_config_values", move |conn| {
            let result = {
                let mut stmt = conn.prepare("SELECT name, real_value, int_value, str_value, bool_value, unsigned FROM run_config_values WHERE run_id=?")?;

                stmt.query_map([run_id], |row| {
                    Ok((
                        row.get::<usize, String>(0)?,
                        row.get::<usize, f64>(1).ok(),
                        row.get::<usize, i64>(2).ok(),
                        row.get::<usize, String>(3).ok(),
                        row.get::<usize, u8>(4).ok().map(|v| if v == 1 { true } else { false }),
                        row.get::<usize, u8>(4).ok().map(|v| if v == 1 { true } else { false }).unwrap_or(false),
                    ))
                })?
                .map(|v| v.expect(format!("can't get run_config_values for {}", run_id).as_str()))
                .fold(HashMap::new(), |mut a, (name, real_value, int_value, str_value, bool_value, unsigned)| {
                    a.insert(
                        name.clone(),
                        real_value
                            .map(|v| v.into())
                            .or(int_value.map(|v| if unsigned { UInt(v as u32) } else { Val::Int(v) }))
                            .or(str_value.map(|v| v.into()))
                            .or(bool_value.map(|v| v.into()))
                            .expect(format!("no value for {}", name).as_str()),
                    );
                    a
                })
            };

            debug!("list_run_config_values: run_id={run_id} result={result:?}");
            Ok(result)
        })
        .await
    }

    async fn list_run_motifs(&self, run_id: RunId) -> Result<Vec<RunMotifEntry>, ThoError> {
        self.with_reader_conn("list_run_motifs", move |conn| {
            let selected_motifs = {
                let mut stmt = conn.prepare(
                    "SELECT sm.motif_id, m.channel_id, sm.rate, m.run_id,                       (SELECT COUNT(*) FROM slices s WHERE s.motif_id = sm.motif_id) AS slices_cnt,                       (SELECT COUNT(DISTINCT s.resource_id) FROM slices s WHERE s.motif_id = sm.motif_id) AS resources_cnt                      FROM selected_motifs sm                      JOIN motifs m ON m.id = sm.motif_id                      WHERE sm.run_id = ?                      ORDER BY m.channel_id ASC, sm.rate DESC, sm.motif_id ASC",
                )?;

                stmt.query_map([run_id], |row| {
                    Ok(RunMotifEntry {
                        motif_id: row.get::<usize, u32>(0)?,
                        channel_id: row.get::<usize, usize>(1)?,
                        score: row.get::<usize, f64>(2).ok(),
                        source_run_id: row.get::<usize, RunId>(3)?,
                        slices_cnt: row.get::<usize, usize>(4)?,
                        resources_cnt: row.get::<usize, usize>(5)?,
                    })
                })?
                .collect::<Result<Vec<_>, Error>>()?
            };

            if !selected_motifs.is_empty() {
                debug!("list_run_motifs: run_id={run_id} selected_motifs_count={}", selected_motifs.len());
                return Ok(selected_motifs);
            }

            let motifs = {
                let mut stmt = conn.prepare(
                    "SELECT m.id, m.channel_id, m.run_id                      FROM motifs m                      WHERE m.run_id = ?                      ORDER BY m.channel_id ASC, m.id ASC",
                )?;

                stmt.query_map([run_id], |row| {
                    Ok(RunMotifEntry {
                        motif_id: row.get::<usize, u32>(0)?,
                        channel_id: row.get::<usize, usize>(1)?,
                        score: None,
                        source_run_id: row.get::<usize, RunId>(2)?,
                        slices_cnt: 0,
                        resources_cnt: 0,
                    })
                })?
                .collect::<Result<Vec<_>, Error>>()?
            };

            debug!("list_run_motifs: run_id={run_id} motifs_count={}", motifs.len());
            Ok(motifs)
        })
        .await
    }

    async fn save_selected_motifs(
        &self,
        run_id: RunId,
        selected_motifs: Vec<(u32, f64)>,
    ) -> Result<RunId, ThoError> {
        debug!("save_selected_motifs: selected_motifs={:?}",selected_motifs);
        self.with_writer_conn("save_selected_motifs", move |conn| {
            let t = conn.transaction()?;

            for selected_motifs_chunk in selected_motifs.chunks(500) {
                let (mut q, params) = selected_motifs_chunk.into_iter().fold(
                    (vec!["INSERT INTO selected_motifs (run_id, motif_id, rate) VALUES "], vec![]),
                    |mut a: (Vec<&str>, Vec<Value>), (motif_id, rate)| {
                        a.0.push("(?, ?, ?)");
                        a.0.push(", ");
                        a.1.push(Value::Integer(run_id as i64));
                        a.1.push(Value::Integer(*motif_id as i64));
                        a.1.push(Value::Real(*rate));
                        a
                    },
                );

                q.pop();
                t.execute(&q.into_iter().collect::<String>(), rusqlite::params_from_iter(params.into_iter()))?;
            }

            t.commit()?;

            Ok(run_id)
        })
        .await
    }


    async fn list_output_run_ids(&self, input_run_id: RunId) -> Result<Vec<RunId>, ThoError> {
        self.with_reader_conn("list_output_run_ids", move |conn| {
            let q = "SELECT run_id FROM run_configs_input WHERE input_run_id=?";

            let id_input_id = {
                let mut stmt = conn.prepare(q)?;
                stmt.query_map([input_run_id], |row| row.get::<usize, usize>(0))?
                    .map(|v| v.map(|v| v as RunId).map_err(|e| e.into()))
                    .collect::<Result<Vec<_>, ThoError>>()
            };
            id_input_id
        })
        .await
    }

    /**k
     * returns vec of channel X (motif id, rate)
     */
    async fn list_selected_motifs(
        &self,
        motif_selection_id: RunId,
        channels: u32,
    ) -> Result<Vec<Vec<(u32, f64)>>, ThoError> {
        self.with_reader_conn("list_selected_motifs", move |conn| {
            let mut channels_motifs = (0..channels).into_iter().map(|_| Vec::with_capacity(10)).collect::<Vec<Vec<(u32, f64)>>>();
            {
                let mut stmt = conn.prepare(
                    "SELECT DISTINCT sm.motif_id, sm.rate, m.channel_id \
                    FROM selected_motifs sm LEFT JOIN motifs AS m ON m.id=sm.motif_id \
                    WHERE sm.run_id=? ORDER BY motif_id"
                )?;

                stmt.query_map([motif_selection_id], |row| {
                    Ok((row.get::<usize, u32>(0)?, row.get::<usize, f32>(1)?, row.get::<usize, usize>(2)?))
                })?
                .map(|v| v.unwrap())
                .for_each(|(motif_id, corr, channel_id)| {
                    channels_motifs[channel_id].push((motif_id, corr as f64))
                });
            }

            debug!("list_selected_motifs: channels_motifs={channels_motifs:?}");
            Ok(channels_motifs)
        })
        .await
    }

    async fn save_selected_net(
        &self,
        run_id: RunId,
        path: PathBuf,
        loss: f32,
    ) -> Result<(), ThoError> {
        self.with_writer_conn("save_selected_net", move |conn| {
            let t = conn.transaction()?;
            t.execute(
                "INSERT INTO selected_nets (net_path, run_id, loss) VALUES (?, ?, ?)",
                rusqlite::params![path.to_string_lossy().to_string(), run_id, loss],
            )?;

            t.commit()?;
            Ok(())
        })
        .await
    }

    async fn list_selected_nets(&self, run_id: RunId, loss: Option<f32>, limit: Option<usize>) -> Result<Vec<(PathBuf, f32)>, ThoError> {
        self.with_reader_conn("list_selected_nets", move |conn| {
            let selected_nets = {
                //TODO compose
                let (query, params): (&str, Vec<rusqlite::types::Value>) = match (loss, limit) {
                    (Some(loss), Some(limit)) => (
                        "SELECT net_path, loss FROM selected_nets WHERE run_id=? AND loss < ? ORDER BY loss ASC LIMIT ?",
                        vec![
                            rusqlite::types::Value::Integer(run_id as i64),
                            rusqlite::types::Value::Real(loss as f64),
                            rusqlite::types::Value::Integer(limit as i64),
                        ],
                    ),
                    (Some(loss), None) => (
                        "SELECT net_path, loss FROM selected_nets WHERE run_id=? AND loss < ? ORDER BY loss ASC",
                        vec![rusqlite::types::Value::Integer(run_id as i64), rusqlite::types::Value::Real(loss as f64)],
                    ),
                    (None, Some(limit)) => (
                        "SELECT net_path, loss FROM selected_nets WHERE run_id=? ORDER BY loss ASC LIMIT ?",
                        vec![
                            rusqlite::types::Value::Integer(run_id as i64),
                            rusqlite::types::Value::Integer(limit as i64),
                        ],
                    ),
                    (None, None) => (
                        "SELECT net_path, loss FROM selected_nets WHERE run_id=? ORDER BY loss ASC",
                        vec![rusqlite::types::Value::Integer(run_id as i64)],
                    ),
                };
                let mut stmt = conn.prepare(query)?;
                stmt.query_map(rusqlite::params_from_iter(params.into_iter()), |row| {
                    Ok((PathBuf::from(row.get::<usize, String>(0)?), row.get::<usize, f32>(1)?))
                })?
                .map(|v| v.unwrap())
                .collect::<Vec<_>>()
            };

            Ok(selected_nets)
        })
        .await
    }

    async fn run_config_descr(&self, corr_run_id: RunId) -> Result<Vec<(i64, String)>, ThoError> {
        self.with_reader_conn("run_config_descr", move |conn| {
            let descr: String = conn.query_row("SELECT descr FROM run_configs WHERE id=?", [corr_run_id], |row| row.get(0))?;
            debug!("descr={:?}", descr);

            parse_corr_labels(descr.as_str())
        })
        .await
    }

    #[cfg(test)]
    fn conn(&self) -> &Mutex<Connection> {
        &self.conn
    }
}

fn init_sqlite_writer(conn: &mut rusqlite::Connection) -> rusqlite::Result<()> {
    use rusqlite::trace::{TraceEvent, TraceEventCodes};

    conn.busy_timeout(Duration::from_secs(5))?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;

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
    use rusqlite::trace::{TraceEvent, TraceEventCodes};

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

fn debug_trace_event(ev: TraceEvent) {
    match ev {
        TraceEvent::Stmt(a, b) => {
            debug!("[sqlite trace reader], TraceEvent::Stmt: {b:?}");
        }
        TraceEvent::Profile(a, b) => {
            debug!("[sqlite trace reader], TraceEvent::Profile: {b:?}");
        }
        TraceEvent::Row(a) => {
            debug!("[sqlite trace reader], TraceEvent::Row:");
        }
        TraceEvent::Close(a) => {
            debug!("[sqlite trace reader], TraceEvent::Close:  backtrace={}", Backtrace::capture());
        }
        x => {
            debug!("[sqlite trace reader] x");
        }
    }
}

//TODO move all prepare to init
impl SqlStore {
    fn count_motif_slices(motif_id: u32, conn: &Connection) -> Result<usize, ThoError> {
        let count = conn.query_row("SELECT count(*) FROM slices WHERE motif_id = ?", [motif_id], |row| {
            row.get::<usize, i64>(0)
        })?;

        usize::try_from(count).map_err(|_| ThoError::new(format!("count_motif_slices: invalid count={count} for motif_id={motif_id}")))
    }

    fn load_motif_target_positions(
        motif_id: u32,
        conn: &Connection,
    ) -> Result<(HashMap<ResourceId, Vec<i64>>, usize), ThoError> {
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
        .try_for_each(|row| -> Result<(), ThoError> {
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

    fn load_motif_resource_contexts(
        motif_id: u32,
        input_len_adjuster: &(dyn Fn(u32) -> u32 + Send + Sync),
        conn: &Connection,
        mut target_positions_by_res: HashMap<ResourceId, Vec<i64>>,
    ) -> Result<Vec<RandomSliceResourceCtx>, ThoError> {
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
        .collect::<Result<Vec<_>, ThoError>>()
    }

    fn load_non_motif_resource_contexts(
        motif_id: u32,
        run_id: RunId,
        channel: i64,
        input_len_adjuster: &(dyn Fn(u32) -> u32 + Send + Sync),
        conn: &Connection,
    ) -> Result<Vec<RandomSliceResourceCtx>, ThoError> {
        let mut stmt = conn.prepare(
            "SELECT r.id, r.import_id, r.key, r.length FROM main.resources r WHERE r.id NOT IN (SELECT resource_id FROM slices s WHERE s.motif_id=?)",
        )?;

        stmt.query_map(params![motif_id as i64], |row| {
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
        .map(|row| row.map_err(ThoError::from))
        .collect::<Result<Vec<_>, ThoError>>()
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

    fn build_random_slice_candidates(
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

    fn shuffle_and_limit_candidates(all_candidates: &mut Vec<TensorIdSliceIdx>, desired_count: usize) {
        let mut rng = rand::rng();
        all_candidates.shuffle(&mut rng);

        if all_candidates.len() > desired_count {
            all_candidates.truncate(desired_count);
        }
    }

    fn map_tokio_rusqlite_err(
        ctx: &'static str,
        e: tokio_rusqlite::Error<ThoError>,
    ) -> ThoError {
        match e {
            tokio_rusqlite::Error::Error(inner) => inner,
            tokio_rusqlite::Error::ConnectionClosed => {
                ThoError::new(format!("{ctx}: connection closed"))
            }
            tokio_rusqlite::Error::Close((_, err)) => ThoError::from(err),
            _ => ThoError::new(format!("{ctx}: unexpected tokio_rusqlite error")),
        }
    }

    async fn with_writer_conn<T, F>(&self, ctx: &'static str, f: F) -> Result<T, ThoError>
    where
        T: Send + Debug + 'static,
        F: FnOnce(&mut Connection) -> Result<T, ThoError> + Send + 'static,
    {
        // self.writer.call(move |conn| {
        //     debug!("[db writer enter {ctx}");
        //     let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(conn)));
        //     match res {
        //         Ok(ok) => {
        //             debug!("[db writer exit {ctx}: {:?}", ok.as_ref().map(|_| "ok"));
        //             ok
        //         }
        //         Err(panic) => {
        //             error!("[db writer PANIC in {ctx}: {:?}", panic);
        //             Err(ThoError::new(format!("panic in sqlite closure: {ctx}")))
        //         }
        //     }
        // }).await.map_err(|e| Self::map_tokio_rusqlite_err(ctx, e))
        match panic_to_err_async!(
            async {
                self.writer
                .call(f).await.map_err(|e| Self::map_tokio_rusqlite_err(ctx, e))
            }
        ) {
            Ok(r) => {
                trace!("with_writer_conn: got result={r:?}");
                Ok(r)
            }
            Err(e) => {
                error!("with_writer_conn: got error={e:?}");
                Err(e)
            }
        }

    }

    async fn with_reader_conn<T, F>(&self, ctx: &'static str, f: F) -> Result<T, ThoError>
    where
        T: Send + Debug + 'static,
        F: FnOnce(&mut Connection) -> Result<T, ThoError> + Send + 'static,
    {
        let reader_id = self.rr.fetch_add(1, Ordering::Relaxed) % self.readers.len();


        match panic_to_err_async!(
            async {
                self.readers[reader_id]
                .call(f).await.map_err(|e| Self::map_tokio_rusqlite_err(ctx, e))
            }
        ) {
            Ok(r) => {
                trace!("with_reader_conn: got result={r:?}");
                Ok(r)
            }
            Err(e) => {
                error!("with_reader_conn: got error={e:?}");
                Err(e)
            }
        }


        // self.readers[reader_id].call(move |conn| {
        //     debug!("[db reader={reader_id}] enter {ctx}");
        //     let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(conn)));
        //     match res {
        //         Ok(ok) => {
        //             debug!("[db reader={reader_id}] exit {ctx}: {:?}", ok.as_ref().map(|_| "ok"));
        //             ok
        //         }
        //         Err(panic) => {
        //             error!("[db reader={reader_id}] PANIC in {ctx}: {:?}", panic);
        //             Err(ThoError::new(format!("panic in sqlite closure: {ctx}")))
        //         }
        //     }
        // }).await.map_err(|e| Self::map_tokio_rusqlite_err(ctx, e))
    }

    fn reader(&self) -> &tokio_rusqlite::Connection {
        let idx = self.rr.fetch_add(1, Ordering::Relaxed) % self.readers.len();
        &self.readers[idx]
    }

    fn save_resources_labels_do_insert(
        resource_keys_iter: IntoIter<(String, u32, Vec<(String, f32)>), 500>,
        import_id: u32,
        t: &Transaction,
    ) -> Result<Vec<ResourceId>, ThoError> {
        let (mut insert_resources_q, insert_resources_params, insert_labeled_resources_params) =
            resource_keys_iter.fold(
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
                            (
                                t.query_row("SELECT id FROM labels WHERE label=?", [label.as_str()], |row| row.get(0))
                                    .expect(format!("label {} not found", label).as_str()),
                                p,
                            )
                        })
                        .collect::<Vec<(u32, f32)>>();
                    insert_labeled_resources_params.push(labels_ids);

                    (
                        insert_resources_q,
                        insert_resources_params,
                        insert_labeled_resources_params,
                    )
                },
            );

        insert_resources_q.pop();
        insert_resources_q.push(" RETURNING id");

        let q = insert_resources_q.into_iter().collect::<String>();
        let mut stmt = t.prepare(&q)?;

        let mut rows = stmt.query(rusqlite::params_from_iter(insert_resources_params.into_iter()))?;

        let mut resource_ids: Vec<u32> = vec![];
        while let Some(row) = rows.next()? {
            resource_ids.push(row.get_unwrap(0));
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
            t.execute(&q_str, rusqlite::params_from_iter(params.into_iter()))
                .unwrap();
        }
    }

    pub async fn from_path<P: AsRef<Path>>(path: P) -> Result<Box<dyn Store>, ThoError> {
        let path_buf = path.as_ref().to_path_buf();

        let mut conn = Connection::open_with_flags(&path_buf, OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE)?;
        Self::create_tables(&mut conn)?;

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

        Ok(Box::new(Self {
            #[cfg(test)]
            conn: Mutex::new(conn),
            writer,
            readers: Arc::new(readers),
            rr: AtomicUsize::new(0),
        }))
    }
    // #[cfg(test)]
    pub async fn in_mem() -> Result<Box<dyn Store>, ThoError> {
        let db_idx = IN_MEM_DB_COUNTER.fetch_add(1, Ordering::Relaxed);
        let uri = format!("file:thoosa_store_{db_idx}?mode=memory&cache=shared");

        let writer = tokio_rusqlite::Connection::open_with_flags(&uri, OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_URI).await?;
        writer.call(init_sqlite_writer).await?;

        let mut readers = Vec::with_capacity(SQL_STORE_READERS_CNT.max(1));
        for _ in 0..SQL_STORE_READERS_CNT.max(1) {
            let reader = tokio_rusqlite::Connection::open_with_flags(&uri, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI).await?;
            reader.call(init_sqlite_reader).await?;
            readers.push(reader);
        }

        let mut conn = Connection::open_with_flags(&uri, OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE | OpenFlags::SQLITE_OPEN_URI)?;
        Self::create_tables(&mut conn)?;

        let readers = if readers.is_empty() {
            vec![writer.clone()]
        } else {
            readers
        };

        Ok(Box::new(Self {
            #[cfg(test)]
            conn: Mutex::new(conn),
            writer,
            readers: Arc::new(readers),
            rr: AtomicUsize::new(0),
        }))
    }

    fn create_tables( conn: &mut Connection) -> Result<(), ThoError> {

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

            //descr - dist, dist etc
            t.execute(
                "CREATE TABLE IF NOT EXISTS run_configs (
                id              INTEGER PRIMARY KEY,
                type            VARCHAR(256),
                descr           VARCHAR(256),
                run_on          DATETIME DEFAULT CURRENT_TIMESTAMP
            )",
                (),
            )?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS run_configs_input (
                input_run_id     NOT NULL REFERENCES run_configs(id) ON DELETE CASCADE,
                run_id     NOT NULL REFERENCES run_configs(id) ON DELETE CASCADE,

                PRIMARY KEY (input_run_id, run_id)
            )",
                (),
            )?;

            t.execute("CREATE INDEX IF NOT EXISTS idx_run_configs_type ON run_configs (type);", ())?;
            t.execute("CREATE INDEX IF NOT EXISTS idx_run_configs_run_on ON run_configs (run_on);", ())?;
            t.execute("CREATE INDEX IF NOT EXISTS idx_run_configs_input_run_id ON run_configs_input (run_id);", ())?;

            //key - resource id  //TODO remove ord_id?
            t.execute(
                "CREATE TABLE IF NOT EXISTS run_resources (
                run_id     NOT NULL REFERENCES run_configs(id) ON DELETE CASCADE,
                ord_id     INTEGER NOT NULL,
                key        VARCHAR(256),
                PRIMARY KEY (run_id, ord_id)
            )",
                (),
            )?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS run_config_values (
                id          INTEGER PRIMARY KEY,
                run_id      NOT NULL REFERENCES run_configs(id) ON DELETE CASCADE,
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
            t.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_run_config_values_name_run_id ON run_config_values (run_id, name, label_id);", ())?;

            t.execute(
                //group - ordinal as appeared when grouping, not needed
                "CREATE TABLE IF NOT EXISTS motifs (
                id              INTEGER PRIMARY KEY,
                group_ord       INTEGER NOT NULL,
                run_id          NOT NULL REFERENCES run_configs(id) ON DELETE CASCADE,
                channel_id      INTEGER NOT NULL
            )",
                (),
            )?;
            t.execute("CREATE INDEX IF NOT EXISTS idx_motifs_run_id ON motifs (run_id);", ())?;

            //motif_id in UNIQUE constraint is needed when new slices are added from different find_motifs in same select_motifs run.
            t.execute(
                "CREATE TABLE IF NOT EXISTS slices (
                id              INTEGER PRIMARY KEY,
                motif_id        NOT NULL REFERENCES motifs(id) ON DELETE CASCADE,
                run_id          NOT NULL REFERENCES run_configs(id) ON DELETE CASCADE,
                channel_id      INTEGER NOT NULL,
                resource_id     REFERENCES resources(id) ON DELETE CASCADE,
                slice_idx       INTEGER NOT NULL,
                UNIQUE(motif_id, run_id, resource_id, slice_idx)
            )",
                (),
            )?;
            // The UNIQUE index on (motif_id, run_id, resource_id, slice_idx) already covers the
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
                run_id      NOT NULL REFERENCES run_configs(id) ON DELETE CASCADE,
                rate    FLOAT NOT NULL
            )",
                (),
            )?;
            t.execute("CREATE INDEX IF NOT EXISTS idx_selected_motifs_run_id_motif_id ON selected_motifs (run_id, motif_id);", ())?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS selected_nets (
                net_path  VARCHAR(1024),
                run_id  NOT NULL REFERENCES run_configs(id) ON DELETE CASCADE,
                loss    FLOAT NOT NULL
            )",
                (),
            )?;

            t.execute("commit", ())?;

        Ok(())
    }

    fn list_resource_set_resources(set_id: usize, conn: &Connection) -> Result<Vec<Resource>, ThoError> {
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
                .map(|v| v.unwrap())
                .collect();

            vec
        };
        Ok(set_resources)
    }

    /**
    defines global label order
    */
    fn list_labels(conn: &Connection) -> Result<Vec<(usize, String)>, ThoError> {
        let set_resources = {
            let mut stmt = conn.prepare("SELECT id, label FROM labels ORDER BY id")?;

            let vec = stmt
                .query_map([], |row| {
                    Ok((row.get::<usize, usize>(0)?, row.get::<usize, String>(1)?))
                })?
                .collect::<Result<Vec<(usize, String)>, Error>>()?;

            vec
        };
        Ok(set_resources)
    }

    fn save_param_value(
        t: &Transaction,
        run_id: RunId,
        param_name: &str,
        value: Val,
        label_id: Option<usize>,
    ) -> Result<(), ThoError> {
        match value {
            Val::Real(v) => {
                t.execute("INSERT INTO run_config_values (run_id, label_id, name, real_value) VALUES (?, ?, ?, ?)", params![run_id, label_id, param_name, v])?;
            }
            Val::Int(v) => {
                t.execute("INSERT INTO run_config_values (run_id, label_id, name, int_value) VALUES (?, ?, ?, ?)", params![run_id, label_id, param_name, v])?;
            }
            Val::UInt(v) => {
                t.execute("INSERT INTO run_config_values (run_id, label_id, name, int_value, unsigned) VALUES (?, ?, ?, ?, 1)", params![run_id, label_id, param_name, v])?;
            }
            Val::Str(v) => {
                t.execute("INSERT INTO run_config_values (run_id, label_id, name, str_value) VALUES (?, ?, ?, ?)", params![run_id, label_id, param_name, v])?;
            }
            Val::Bool(v) => {
                t.execute("INSERT INTO run_config_values (run_id, label_id, name, bool_value) VALUES (?, ?, ?, ?)", params![run_id, label_id, param_name, v])?;
            }

        }

        Ok(())
    }

    fn save_resource_set_group(descr: String, t: &Transaction) -> Result<usize, ThoError> {
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
    ) -> Result<(), ThoError> {
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

    fn insert_slices(
        t: &Transaction,
        run_id: RunId,
        channel: i64,
        motif_id: i64,
        chunk: IntoIter<IdSlice, 500>,
    ) -> Result<(), ThoError> {
        let sql = "INSERT INTO slices (run_id, channel_id, motif_id, resource_id, slice_idx) VALUES";

        let (mut q, params) = chunk.into_iter().fold(
            (vec![sql], vec![]),
            |mut a: (Vec<&str>, Vec<Value>), v: IdSlice| {
                a.0.push("(?, ?, ?, ?, ?)");
                a.0.push(", ");
                a.1.push((run_id as i64).into());
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
                debug!("insert_slices: e={e} sqlite_error={:?} run_id={run_id}, channel={channel}, motif_id={motif_id}",e.sqlite_error());
            }
        };

        Ok(())
    }

    fn insert_slices_matches_chunk_from_tmp(
        t: &Transaction,
        run_id: RunId,
        channel: i64,
        motif_id: i64,
        chunk: IntoIter<IdSlice, 500>,
    ) -> Result<(), ThoError> {
        Self::insert_slices(t, run_id, channel, motif_id, chunk.clone())?;

        // let mut stmt = t.prepare("SELECT * FROM slices")?;
        // let mut res = stmt.query([])?;
        // while let Some(r) =  res.next()?{
        //     debug!("slices: r={r:?}");
        // }

        let sql = format!("INSERT INTO slice_matches (motif_id, slice_id_0, slice_id_1, dist) \
                            SELECT s_0.motif_id, s_0.id, s_1.id, dist FROM slices_{run_id}_{channel}_tmp.slice_matches tsm \
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

    fn get_input_run_ids_(
        conn: &Connection,
        run_id: RunId,
        input_type: Option<&str>,
    ) -> Result<Vec<RunId>, ThoError> {
        let mut stmt = conn.prepare("SELECT rci.input_run_id, rc.type FROM run_configs_input rci JOIN run_configs rc ON rc.id=rci.input_run_id WHERE run_id=?")?;

        let vec = stmt
            .query_map([run_id], |row| {
                Ok((row.get::<usize, RunId>(0)?, row.get::<usize, String>(1)?))
            })?
            .flat_map(|v| {
                let (run_id, config_type) = v.expect("no run_id, config_type");

                if input_type
                    .map(|input_type| input_type.to_string().ne(&config_type))
                    .unwrap_or(false)
                {
                    Box::new(
                        Self::get_input_run_ids_(conn, run_id, input_type)
                            .unwrap()
                            .into_iter(),
                    ) as Box<dyn Iterator<Item = RunId>>
                } else {
                    Box::new(std::iter::once(run_id)) as Box<dyn Iterator<Item = RunId>>
                }
            })
            .collect::<Vec<RunId>>();

        Ok(vec)
    }

    fn log_error<C: Debug>(ctx: C, t: &Transaction, e: Error) -> Error {
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

    fn _motif_meta(motif_id: u32, conn: &Connection) -> Result<(i64, RunId), ThoError> {
        // Pre-load the motif metadata to identify its channel and run.
        let (channel, run_id): (i64, RunId) = conn.query_row(
            "SELECT channel_id, run_id FROM motifs WHERE id=?",
            [motif_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        Ok((channel, run_id))
    }
}

fn conds_to_q(input: &Vec<(String, SetCond)>) -> String {
    assert!(input.len() > 0);
    let inp_len = input.len() * 6 - 1;

    input
        .into_iter()
        .flat_map(|v| ["(l.label='", v.0.as_str(), "' and ", v.1.p(), "?)", " or "])
        .take(inp_len)
        .collect::<String>()
}

pub struct SqlMotifTrackerStore {
    run_id: RunId,
    channel: u32,
    conn: Mutex<Connection>,
    path: PathBuf,
}

impl SqlMotifTrackerStore {
    pub fn new<P: AsRef<Path> + Sync + Send + Debug>(
        path: P,
        run_id: RunId,
        channel: u32,
    ) -> Result<Box<dyn MotifTrackerStore>, ThoError> {
        let mut conn = Connection::open(&path).expect("SqlMotifTrackerStore::new: connection");
        {
            let t = conn.transaction()?;

            debug!("SqlMotifTrackerStore::new: path={path:?}");

            t.execute(
                "CREATE TABLE IF NOT EXISTS slice_matches (
                id              INTEGER PRIMARY KEY,
                tensor_id_0     INTEGER NOT NULL,
                slice_idx_0     INTEGER NOT NULL,
                tensor_id_1     INTEGER NOT NULL,
                slice_idx_1     INTEGER NOT NULL,
                dist            FLOAT NOT NULL
            )",
                (),
            )?;

            t.commit()?;
        }
        Ok(Box::new(Self {
            run_id,
            channel,
            conn: Mutex::new(conn),
            path: path.as_ref().to_path_buf(),
        }))
    }
}

#[repr(C)]
#[derive(Debug, PartialEq, Copy, Clone, Pod, Zeroable)]
pub struct SlicesMatch(pub TensorIdSliceIdx, pub TensorIdSliceIdx, pub f32);

#[async_trait]
pub trait MotifTrackerStore: Debug + Send + Sync {
    fn save_slices(&mut self, slices: Vec<SlicesMatch>) -> Result<(), ThoError>;
    async fn load_slices_into_groups(&mut self, store: &Arc<Box<dyn Store>>, max_slices_counts: Option<u32>) -> Result<usize, ThoError>;

    fn run_id(&self) -> RunId;
    fn channel(&self) -> u32;
}

pub fn filter_groups_by_slices_count(max_slices_counts: Option<u32>, groups: HashMap<u16, HashSet<IdSlice>>) -> HashMap<u16, HashSet<IdSlice>> {
    match max_slices_counts {
        None => groups,
        Some(max_slices_counts) => groups.into_iter().fold(HashMap::new(), |mut a, v| {
            if v.1.len() < max_slices_counts as usize {
                a.insert(v.0, v.1);
            } else {
                warn!("skipping group with {} slices",v.1.len());
            }
            a
        }),
    }
}

#[async_trait]
impl MotifTrackerStore for SqlMotifTrackerStore {
    fn save_slices(&mut self, slices: Vec<SlicesMatch>) -> Result<(), ThoError> {
        debug!("MotifTrackerStore::save_slices: motifs.len()={:?}, self.run_id={:?}",slices.len(),self.run_id);

        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;

        assert!(slices.len() > 0);

        for slices_chunk in slices.chunks(500) {
            let (mut q, params) = slices_chunk.into_iter().fold(
                (vec!["INSERT INTO slice_matches (tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1, dist) VALUES"], vec![]), |mut a: (Vec<&str>, Vec<Value>), v| {
                    a.0.push("(?, ?, ?, ?, ?)");
                    a.0.push(", ");

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
    }

    async fn load_slices_into_groups(
        &mut self,
        store: &Arc<Box<dyn Store>>,
        max_slices_counts: Option<u32>,
    ) -> Result<usize, ThoError> {
        let groups = {
            let mut guard = self.conn.lock();
            debug!(
                "set_motifs: run_id={:?}, channel={:?}",
                self.run_id, self.channel
            );

            let mut stmt = guard.prepare(
                "SELECT id, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1 FROM slice_matches",
            )?;

            let groups = group_slices_try(stmt.query_map([], |row| {
                //TODO move group_slices out, not for store
                Ok((
                    IdSlice(row.get(0)?, row.get(1)?, row.get(2)?),
                    IdSlice(row.get(0)?, row.get(3)?, row.get(4)?),
                ))
            })?)?;

            debug!(
                "set_motifs: run_id={:?}, channel={:?} groups={:?}",
                self.run_id, self.channel, groups
            );

            drop(stmt);
            groups
        };

        let groups_by_slices_count = filter_groups_by_slices_count(max_slices_counts, groups);
        let groups_len = groups_by_slices_count.len();
        store.save_groups_as_motifs_with_slices_matches(
            self.run_id,
            self.channel as i64,
            groups_by_slices_count,
            self.path.clone(),
        ).await?;

        Ok(groups_len)
    }

    fn run_id(&self) -> RunId {
        self.run_id
    }

    fn channel(&self) -> u32 {
        self.channel
    }

}

#[cfg(test)]
mod store_tests {
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
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::val::Val;
    use thoosa_shared::{TensorIdSliceIdx, cdt, cdt_for_dir, MotifSlice};

    use crate::{
        MotifTrackerStore, RunConfigFilter, SetCond, SlicesMatch, SqlMotifTrackerStore, SqlStore, Store, ThoError,
    };
    use std::path::PathBuf;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_group_resources() -> Result<(), ThoError> {
        // fs::remove_file("/tmp/test_db.db3").unwrap_or(());
        // let mut store = SqlStore::from_path("/tmp/test_db.db3"))?;

        let mut store = SqlStore::in_mem().await?;

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

        //TODO asserts

        debug!("done in {}", t1.elapsed().as_millis());

        // fs::remove_file(db_file)?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_list_resources_not_labled() -> Result<(), ThoError> {
        let mut store = SqlStore::in_mem().await?;

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
    async fn test_save_pure_balanced_sets_group_created() -> Result<(), ThoError> {
        let mut store = Arc::new(SqlStore::in_mem().await?);

        init_test_store(&store).await?;

        let labels_count = store.list_labels().await?.len();
        let train_sets = store.list_resource_set_group_sets("A_balanced_group_train").await?;
        let valid_sets = store.list_resource_set_group_sets("A_balanced_group_validation").await?;
        let test_sets = store.list_resource_set_group_sets("A_balanced_group_test").await?;

        assert_eq!(labels_count, train_sets.len());
        assert_eq!(labels_count, valid_sets.len());
        assert_eq!(labels_count, test_sets.len());
        assert!(train_sets.iter().all(|(_, descr)| descr.starts_with("A_balanced_set_")));
        assert!(valid_sets.iter().all(|(_, descr)| descr.starts_with("A_balanced_set_")));
        assert!(test_sets.iter().all(|(_, descr)| descr.starts_with("A_balanced_set_")));

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_save_slices() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        fs::create_dir_all(&test_dir)?;
        let mut store = Arc::new(SqlStore::from_path(
            test_dir.join("test_db.db3"),
        ).await?);

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

        let find_motifs_run_id = store.save_run_config(
            None,
            &"todo",
            "find_motifs",
            "",
            HashMap::from([
                ("prop1".into(), Val::Int(1)),
                ("prop2".into(), Val::Real(2.0)),
                ("prop3".into(), Val::Str("333".to_string())),
            ]),
        ).await?;

        let slices_store_path = test_dir.join("test_slices_db.db3");
        let mut slices_store = SqlMotifTrackerStore::new(&slices_store_path, find_motifs_run_id, 7)
            .expect("can't create tmp sotre");

        slices_store.save_slices(slices.clone())?;

        let groups = slices_store.load_slices_into_groups(&store, None).await?;

        {
            let store_guard = Arc::clone(&store);
            let mut guard = store_guard.conn().lock();
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

        let motif_slices_map = store.list_motifs_slices(find_motifs_run_id, None).await?;
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

        let select_motifs_run_id = store.save_run_config(
            Some(&[find_motifs_run_id]),
            &"todo",
            "find_motifs",
            "",
            HashMap::from([
                ("prop1".into(), Val::Int(1)),
                ("prop2".into(), Val::Real(2.0)),
                ("prop3".into(), Val::Str("333".to_string())),
            ]),
        ).await?;

        let _select_motifs_run_id = store.save_selected_motifs(
            select_motifs_run_id,
            vec![
                (motif_slices_444[0].motif_id, 0.1),  //motif_id corr
                (motif_slices_1111[0].motif_id, 0.2), //motif_id corr
            ],
        ).await?;

        let input_run_ids = store.get_input_run_ids(select_motifs_run_id, None).await?;

        assert_eq!(
            vec![find_motifs_run_id],
            input_run_ids
        );

        let motifs = store.list_selected_motifs(select_motifs_run_id, 20).await?;

        debug!("motifs: {:?}", motifs);

        //7th channel
        let mut motif_ids = motifs[7].iter().map(|(motif_id, _)| *motif_id).collect::<Vec<_>>();
        motif_ids.sort();
        let mut expected_ids = vec![motif_slices_444[0].motif_id, motif_slices_1111[0].motif_id];
        expected_ids.sort();
        assert_eq!(expected_ids, motif_ids);

        let selected_run_motifs = store.list_run_motifs(select_motifs_run_id).await?;
        assert_eq!(2, selected_run_motifs.len());
        assert!(selected_run_motifs.iter().all(|entry| entry.channel_id == 7));
        let selected_run_motifs_by_id =
            selected_run_motifs.iter().map(|entry| (entry.motif_id, entry)).collect::<HashMap<_, _>>();
        let motif_1111_entry = selected_run_motifs_by_id.get(&motif_slices_1111[0].motif_id).unwrap();
        let motif_444_entry = selected_run_motifs_by_id.get(&motif_slices_444[0].motif_id).unwrap();
        assert_eq!(Some(0.2), motif_1111_entry.score);
        assert_eq!(Some(0.1), motif_444_entry.score);
        assert_eq!(find_motifs_run_id, motif_1111_entry.source_run_id);
        assert_eq!(find_motifs_run_id, motif_444_entry.source_run_id);

        let (_channel, _run_id, motif_1111_slices_all, motif_1111_resources_map) =
            store.list_motif_slices(motif_slices_1111[0].motif_id, None, None, None).await?;
        let (_channel, _run_id, motif_444_slices_all, motif_444_resources_map) =
            store.list_motif_slices(motif_slices_444[0].motif_id, None, None, None).await?;
        assert_eq!(motif_1111_slices_all.len(), motif_1111_entry.slices_cnt);
        assert_eq!(motif_1111_resources_map.len(), motif_1111_entry.resources_cnt);
        assert_eq!(motif_444_slices_all.len(), motif_444_entry.slices_cnt);
        assert_eq!(motif_444_resources_map.len(), motif_444_entry.resources_cnt);

        let find_run_motifs = store.list_run_motifs(find_motifs_run_id).await?;
        assert_eq!(2, find_run_motifs.len());
        assert!(find_run_motifs.iter().all(|entry| entry.score.is_none()));
        assert!(find_run_motifs.iter().all(|entry| entry.source_run_id == find_motifs_run_id));
        let mut find_run_motif_ids = find_run_motifs.iter().map(|entry| entry.motif_id).collect::<Vec<_>>();
        find_run_motif_ids.sort();
        assert_eq!(expected_ids, find_run_motif_ids);
        //
        // //TODO
        let (channel, run_id_, mut slices, resources_map) = store.list_motif_slices(motif_slices_1111[0].motif_id, None, None, None).await?;
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
        assert_eq!(find_motifs_run_id, run_id_);

        let (channel, run_id_, mut slices, resources_map) = store.list_motif_slices(motif_slices_1111[0].motif_id, None, Some(1), None).await?;
        slices.sort();


        let resources = store.list_resources_not_in_group("pure group 10", 10).await;
        debug!("resources={resources:?}");

        let (channel, run_id_, mut slices, resources_map) = store.list_motif_slices(motif_slices_1111[0].motif_id, None, Some(3), Some((0, 0))).await?;

        debug!("random slices={slices:?}");
        //
        // // let (_channel, _run_id_, mut slices) = store.list_motif_slices(1, Some(10), Some((run_id, channel as usize))).await?;
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
        //     store.run_configs_descr(select_motifs_run_id)?
        // );
        //
        // store.lock().delete_slices_without_matches(run_id)?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_save_list_selected_nets() -> Result<(), ThoError> {
        let mut store = SqlStore::in_mem().await?;

        let run_id = store.save_run_config(
            None,
            &"selected_nets",
            "selected_nets",
            "",
            HashMap::new(),
        ).await?;

        for i in 0..105 {
            store.save_selected_net(run_id, PathBuf::from(format!("nets/{i:03}")), i as f32).await?;
        }

        let selected = store.list_selected_nets(run_id, None, None).await?;
        assert_eq!(105, selected.len());
        assert_eq!(PathBuf::from("nets/000"), selected[0].0);
        assert_eq!(0.0, selected[0].1);
        assert_eq!(PathBuf::from("nets/104"), selected[104].0);
        assert_eq!(104.0, selected[104].1);

        let limited = store.list_selected_nets(run_id, None, Some(3)).await?;
        assert_eq!(3, limited.len());
        assert_eq!(PathBuf::from("nets/000"), limited[0].0);
        assert_eq!(PathBuf::from("nets/001"), limited[1].0);
        assert_eq!(PathBuf::from("nets/002"), limited[2].0);

        let filtered = store.list_selected_nets(run_id, Some(15.0), None).await?;
        assert_eq!(15, filtered.len());
        assert_eq!(PathBuf::from("nets/000"), filtered[0].0);
        assert_eq!(PathBuf::from("nets/014"), filtered[14].0);

        let filtered_limited = store.list_selected_nets(run_id, Some(15.0), Some(5)).await?;
        assert_eq!(5, filtered_limited.len());
        assert_eq!(PathBuf::from("nets/000"), filtered_limited[0].0);
        assert_eq!(PathBuf::from("nets/004"), filtered_limited[4].0);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_list_motifs_slices_filters_slice_run_ids() -> Result<(), ThoError> {
        let mut store = SqlStore::in_mem().await?;

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

        let find_run_id =
            store.save_run_config(None, &"find", "find_motifs", "", HashMap::new()).await?;
        let select_run_id = store.save_run_config(
            Some(&[find_run_id]),
            &"select",
            "select_motifs",
            "",
            HashMap::new(),
        ).await?;

        {
            let mut guard = store.conn().lock();
            let t = guard.transaction()?;

            t.execute(
                "INSERT INTO motifs (id, group_ord, run_id, channel_id) VALUES (1, 0, ?, ?)",
                params![find_run_id, 0i64],
            )?;

            t.execute(
                "INSERT INTO slices (motif_id, run_id, channel_id, resource_id, slice_idx) VALUES (1, ?, ?, ?, ?)",
                params![find_run_id, 0i64, resource_ids[0], 10i64],
            )?;
            t.execute(
                "INSERT INTO slices (motif_id, run_id, channel_id, resource_id, slice_idx) VALUES (1, ?, ?, ?, ?)",
                params![select_run_id, 0i64, resource_ids[0], 30i64],
            )?;

            t.commit()?;
        }

        let resource = store.list_resources_by_ids(&resource_ids).await?.remove(0);

        let find_only = store.list_motifs_slices(find_run_id, Some(vec![])).await?;
        let all_runs = store.list_motifs_slices(find_run_id, None).await?;
        let both_runs =
            store.list_motifs_slices(find_run_id, Some(vec![select_run_id, select_run_id])).await?;

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
    async fn test_list_motif_slices_filters_slice_run_ids() -> Result<(), ThoError> {
        let mut store = SqlStore::in_mem().await?;

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

        let find_run_id =
            store.save_run_config(None, &"find", "find_motifs", "", HashMap::new()).await?;
        let select_run_id = store.save_run_config(
            Some(&[find_run_id]),
            &"select",
            "select_motifs",
            "",
            HashMap::new(),
        ).await?;

        {
            let mut guard = store.conn().lock();
            let t = guard.transaction()?;

            t.execute(
                "INSERT INTO motifs (id, group_ord, run_id, channel_id) VALUES (2, 0, ?, ?)",
                params![find_run_id, 0i64],
            )?;

            t.execute(
                "INSERT INTO slices (motif_id, run_id, channel_id, resource_id, slice_idx) VALUES (2, ?, ?, ?, ?)",
                params![find_run_id, 0i64, resource_ids[0], 12i64],
            )?;
            t.execute(
                "INSERT INTO slices (motif_id, run_id, channel_id, resource_id, slice_idx) VALUES (2, ?, ?, ?, ?)",
                params![select_run_id, 0i64, resource_ids[0], 32i64],
            )?;

            t.commit()?;
        }

        let (_, run_id, all_slices, _) = store.list_motif_slices(2, None, None, None).await?;
        assert_eq!(find_run_id, run_id);
        let mut idxs = all_slices.iter().map(|s| s.1).collect::<Vec<_>>();
        idxs.sort();
        assert_eq!(vec![12, 32], idxs);

        let (_, _, find_only, _) = store.list_motif_slices(2, Some(vec![]), None, None).await?;
        assert_eq!(vec![12], find_only.iter().map(|s| s.1).collect::<Vec<_>>());

        let (_, _, filtered, _) =
            store.list_motif_slices(2, Some(vec![select_run_id, select_run_id]), None, None).await?;
        let mut idxs = filtered.iter().map(|s| s.1).collect::<Vec<_>>();
        idxs.sort();
        assert_eq!(vec![12, 32], idxs);

        Ok(())
    }

    //TODO merge with init_test_store in cli::main_tests
    pub async fn init_test_store(
        store: &Arc<Box<dyn Store>>,
    ) -> Result<Vec<ResourceId>, ThoError> {
        let store_guard = store;
        let import_id = store_guard.save_import("test".to_string()).await?;

        let _aaa_id = store_guard.save_label("aaa").await?;
        let _bbb_id = store_guard.save_label("bbb").await?;
        let _ccc_id = store_guard.save_label("ccc").await?;

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
        store_guard.save_pure_balanced_sets("A", None, 0.6, 0.2).await?;
        Ok(resource_ids)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_save_run_config() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        fs::create_dir_all(&test_dir)?;
        let mut store = Arc::new(SqlStore::from_path(
            test_dir.join("test_db.db3"),
        ).await?);

        let run_id = store.save_run_config(
            None,
            &"todo",
            "test",
            "",
            HashMap::from([
                ("prop1".into(), Val::Int(1)),
                ("prop2".into(), Val::Real(2.0)),
                ("prop3".into(), Val::Str("333".to_string())),
                ("prop4".into(), Val::Bool(false)),
                ("prop5".into(), Val::Bool(true)),
            ]),
        ).await?;

        let next_run_id = store.save_run_config(
            Some(&[run_id]),
            &"todo",
            "test next",
            "",
            HashMap::from([
                ("next_prop1".into(), Val::Int(1)),
                ("next_prop2".into(), Val::Real(2.0)),
                ("next_prop3".into(), Val::Str("zzz".to_string())),
            ]),
        ).await?;

        let loaded_params = store.list_run_config_values(run_id).await?;

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

        let netx_loaded_params = store.list_run_config_values(next_run_id).await?;

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
    async fn test_list_run_configs_page_and_by_ids() -> Result<(), ThoError> {
        let mut store = SqlStore::in_mem().await?;

        let run_id_1 = store.save_run_config(None, &"first", "test", "", HashMap::new()).await?;
        let run_id_2 = store.save_run_config(None, &"second", "test", "", HashMap::new()).await?;
        let run_id_3 = store.save_run_config(None, &"third", "test", "", HashMap::new()).await?;

        let first_page = store.list_run_configs_page(2, 0).await?;
        assert_eq!(vec![run_id_3, run_id_2], first_page.into_iter().map(|entry| entry.id).collect::<Vec<_>>());

        let second_page = store.list_run_configs_page(2, 2).await?;
        assert_eq!(vec![run_id_1], second_page.into_iter().map(|entry| entry.id).collect::<Vec<_>>());

        let loaded_by_ids = store.list_run_configs_by_ids(&[run_id_1, run_id_3]).await?;
        assert_eq!(vec![run_id_3, run_id_1], loaded_by_ids.into_iter().map(|entry| entry.id).collect::<Vec<_>>());

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_list_run_configs_filtered_returns_matches_and_ancestors() -> Result<(), ThoError> {
        let mut store = SqlStore::in_mem().await?;

        let root_run_id = store.save_run_config(None, &"root", "root", "", HashMap::new()).await?;
        let select_run_id = store.save_run_config(Some(&[root_run_id]), &"SelectMotifsRunConfig", "alpha motif selection", "", HashMap::new()).await?;
        let child_run_id = store.save_run_config(Some(&[select_run_id]), &"TrainMotifsNetsRunConfig", "child", "", HashMap::new()).await?;
        let _other_run_id = store.save_run_config(None, &"OtherRunConfig", "beta", "", HashMap::new()).await?;

        let by_id = store.list_run_configs_filtered(RunConfigFilter {
            id: Some(select_run_id),
            ..RunConfigFilter::default()
        }).await?;
        assert_eq!(vec![select_run_id, root_run_id], by_id.into_iter().map(|entry| entry.id).collect::<Vec<_>>());

        let by_type = store.list_run_configs_filtered(RunConfigFilter {
            run_config_type: Some("TrainMotifsNetsRunConfig".to_string()),
            ..RunConfigFilter::default()
        }).await?;
        assert_eq!(vec![child_run_id, select_run_id, root_run_id], by_type.into_iter().map(|entry| entry.id).collect::<Vec<_>>());

        let by_descr = store.list_run_configs_filtered(RunConfigFilter {
            descr_substr: Some("alpha".to_string()),
            ..RunConfigFilter::default()
        }).await?;
        assert_eq!(vec![select_run_id, root_run_id], by_descr.into_iter().map(|entry| entry.id).collect::<Vec<_>>());

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
    async fn test_get_prev_run_id() -> Result<(), ThoError> {
        let mut store = SqlStore::in_mem().await?;

        let input_run_id = store.save_run_config(None, &"aaa", "test 0", "", HashMap::new()).await?;
        let run_id =
            store.save_run_config(Some(&[input_run_id]), &"aaa", "test 1", "", HashMap::new()).await?;

        assert_eq!(store.get_input_run_ids(run_id, None).await?[0], input_run_id);

        let input_run_id_0 = store.save_run_config(None, &"aaa", "test 0", "", HashMap::new()).await?;
        let input_run_id_1 = store.save_run_config(
            Some(&[input_run_id_0]),
            &"bbb",
            "test 1",
            "",
            HashMap::new(),
        ).await?;
        let run_id = store.save_run_config(
            Some(&[input_run_id_1]),
            &"ccc",
            "test 1",
            "",
            HashMap::new(),
        ).await?;

        assert_eq!(
            store.get_input_run_ids(run_id, Some(&"aaa")).await?[0],
            input_run_id_0
        );

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_list_input_run_ids_reads_large_sets_in_chunks() -> Result<(), ThoError> {
        let mut store = SqlStore::in_mem().await?;

        let input_run_id = store.save_run_config(None, &"root", "root", "", HashMap::new()).await?;
        let mut run_ids = Vec::with_capacity(1001);
        for idx in 0..1001 {
            run_ids.push(
                store
                    .save_run_config(
                        Some(&[input_run_id]),
                        &format!("task-{idx}"),
                        "chunked",
                        "",
                        HashMap::new(),
                    )
                    .await?,
            );
        }

        let mut input_pairs = store.list_input_run_ids(&run_ids).await?;
        input_pairs.sort();

        let mut expected_pairs = run_ids
            .into_iter()
            .map(|run_id| (run_id, input_run_id))
            .collect::<Vec<_>>();
        expected_pairs.sort();

        assert_eq!(expected_pairs, input_pairs);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_list_run_ids_with_outputs_reads_large_sets_in_chunks() -> Result<(), ThoError> {
        let mut store = SqlStore::in_mem().await?;

        let mut parent_run_ids = Vec::with_capacity(1001);
        for idx in 0..1001 {
            let parent_run_id = store.save_run_config(None, &format!("parent-{idx}"), "parent", "", HashMap::new()).await?;
            if idx % 2 == 0 {
                store.save_run_config(Some(&[parent_run_id]), &format!("child-{idx}"), "child", "", HashMap::new()).await?;
            }
            parent_run_ids.push(parent_run_id);
        }

        let mut run_ids_with_outputs = store.list_run_ids_with_outputs(&parent_run_ids).await?;
        run_ids_with_outputs.sort();

        let mut expected_run_ids = parent_run_ids
            .into_iter()
            .enumerate()
            .filter_map(|(idx, run_id)| if idx % 2 == 0 { Some(run_id) } else { None })
            .collect::<Vec<_>>();
        expected_run_ids.sort();

        assert_eq!(expected_run_ids, run_ids_with_outputs);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_list_resources_by_file_stem() -> Result<(), ThoError> {
        let mut store = SqlStore::in_mem().await?;
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
    async fn test_list_random_slices_respects_constraints() -> Result<(), ThoError> {
        let mut store = SqlStore::in_mem().await?;

        create_resources(&store, &[("res_a", 2048), ("res_b", 2048)]).await?;

        let run_id = store.save_run_config(None, &"find_motifs", "random test", "", HashMap::new()).await?;
        let channel_id = 3i64;

        let res_a = fetch_resource_id(&store, "res_a")?;
        let res_b = fetch_resource_id(&store, "res_b")?;

        let motif_id = insert_motif_with_slices(
            &store,
            run_id,
            channel_id,
            0,
            &[(res_a, 100), (res_a, 320)],
        )?;
        insert_motif_with_slices(
            &store,
            run_id,
            channel_id,
            1,
            &[(res_b, 50), (res_b, 400)],
        )?;

        const WIDTH: u32 = 40;
        let (channel, run_id, random_slices, resources_map) =
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
    async fn test_list_random_slices_respects_min_distance() -> Result<(), ThoError> {
        let mut store = SqlStore::in_mem().await?;

        create_resources(&store, &[("res_main", 4096)]).await?;

        let run_id =
            store.save_run_config(None, &"find_motifs", "random spacing", "", HashMap::new()).await?;
        let channel_id = 5i64;
        let res_main = fetch_resource_id(&store, "res_main")?;

        let motif_id = insert_motif_with_slices(
            &store,
            run_id,
            channel_id,
            0,
            &[(res_main, 200), (res_main, 1200)],
        )?;

        const WIDTH: u32 = 50;
        const MIN_DISTANCE: u32 = 300;
        let (channel, run_id, random_slices, resources_map) =
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
    async fn test_list_random_slices_uses_input_len_adjuster() -> Result<(), ThoError> {
        let mut store = SqlStore::in_mem().await?;

        create_resources(&store, &[("res_main", 2000)]).await?;

        let run_id = store.save_run_config(None, &"find_motifs", "length adjust", "", HashMap::new()).await?;
        let channel_id = 6i64;
        let res_main = fetch_resource_id(&store, "res_main")?;

        let motif_id = insert_motif_with_slices(
            &store,
            run_id,
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
    async fn heavy_list_random_slices() -> Result<(), ThoError> {
        let mut store = SqlStore::in_mem().await?;

        create_resources(&store, (0..100).map(|i| (&*format!("res_{i}").leak(), 2000u32)).collect::<Vec<_>>().as_slice()).await?;

        let run_id = store.save_run_config(None, &"find_motifs", "length adjust", "", HashMap::new()).await?;
        let channel_id = 6i64;
        let res_0 = fetch_resource_id(&store, "res_0")?;
        let res_10 = fetch_resource_id(&store, "res_1")?;

        let motif_id = insert_motif_with_slices(
            &store,
            run_id,
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
    async fn test_list_random_slices_falls_back_to_other_resources() -> Result<(), ThoError> {
        let mut store = SqlStore::in_mem().await?;

        create_resources(&store, &[("res_main", 220), ("res_other", 500)]).await?;

        let run_id = store.save_run_config(None, &"find_motifs", "fallback negatives", "", HashMap::new()).await?;
        let channel_id = 7i64;
        let res_main = fetch_resource_id(&store, "res_main")?;
        let res_other = fetch_resource_id(&store, "res_other")?;

        let motif_id = insert_motif_with_slices(
            &store,
            run_id,
            channel_id,
            0,
            &[(res_main, 40), (res_main, 100), (res_main, 160)],
        )?;
        insert_motif_with_slices(
            &store,
            run_id,
            channel_id,
            1,
            &[(res_other, 60), (res_other, 260)],
        )?;

        const WIDTH: u32 = 50;
        let (channel, returned_run_id, random_slices, resources_map) =
            store.list_random_slices_out_of_motif(motif_id, WIDTH, None, Arc::new(|length: u32| length), None).await?;

        assert_eq!(channel_id, channel);
        assert_eq!(run_id, returned_run_id);
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

    async fn create_resources(store: &Box<dyn Store>, specs: &[(&str, u32)]) -> Result<(), ThoError> {
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

    fn fetch_resource_id(store: &Box<dyn Store>, key: &str) -> Result<ResourceId, ThoError> {
        let guard = store.conn().lock();
        let id = guard.query_row("SELECT id FROM resources WHERE key=?", [key], |row| {
            row.get::<usize, ResourceId>(0)
        })?;
        Ok(id)
    }

    fn insert_motif_with_slices(
        store: &Box<dyn Store>,
        run_id: RunId,
        channel_id: i64,
        group_ord: i64,
        slices: &[(ResourceId, i64)],
    ) -> Result<u32, ThoError> {
        assert!(slices.len() > 0, "insert_motif_with_slices: no slices");

        let mut guard = store.conn().lock();
        let t = guard.transaction()?;


        t.execute(
            "INSERT INTO motifs (group_ord, run_id, channel_id) VALUES (?, ?, ?)",
            params![group_ord, run_id as i64, channel_id],
        )?;

        let motif_id = t.query_row("SELECT last_insert_rowid()", [], |row| {
            row.get::<usize, u32>(0)
        })?;

        for (resource_id, slice_idx) in slices {
            t.execute(
                "INSERT INTO slices (run_id, channel_id, motif_id, resource_id, slice_idx) VALUES (?, ?, ?, ?, ?)",
                params![run_id as i64, channel_id, motif_id as i64, *resource_id as i64, *slice_idx],
            )?;
        }

        t.commit()?;

        Ok(motif_id)
    }
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

    //run_id, motif_id, channel, slices
    pub fn with_add_slices_to_motif(mut self, inputs:impl IntoIterator<Item=(RunId, u32, i64, HashSet<IdSlice>)> + Debug ) -> Self {
        debug!("with_add_slices_to_motif: inputs={inputs:?}");
        for inps in inputs {
            self.store.expect_add_slices_to_motif().with(
                eq(inps.0),
                eq(inps.1),
                eq(inps.2),
                eq(inps.3),
            ).times(1).return_const(Ok(()));
        }

        self
    }

    pub fn with_add_slices_to_motif_at_least_once(mut self)->Self {
        self.store.expect_add_slices_to_motif().times(1..).return_const(Ok(()));

        self
    }

    pub fn with_get_motif_meta<R:Into<HashMap<u32, Result<(usize, RunId), ThoError>>>>(mut self, results: R) -> Self {
        let map = results.into();
        self.store.expect_get_motif_meta().returning(
            move |motif_id|{
                let result = map.get(&motif_id).cloned().expect(&format!("mock is not configured to return motif meta for motif_id {motif_id}"));


                result
            }
        );
        self
    }

    pub fn with_get_input_run_ids<R:Into<HashMap<RunId, Result<Vec<RunId>, ThoError>>>>(mut self, results: R) -> Self{
        let map = results.into();
        self.store.expect_get_input_run_ids().returning(
            move |run_id, input_type|{
                map.get(&run_id).cloned().expect(&format!("mock is not configured to return input_run_ids for run_id {run_id}"))
            }
        );
        self
    }

    pub fn with_list_motifs<R:Into<HashMap<RunId, Result<Vec<u32>, ThoError>>>>(mut self, results: R) -> Self {
        let map = results.into();
        self.store.expect_list_motifs().returning(
            move |run_id|{
                map.get(&run_id).cloned().expect(&format!("mock is not configured to return motifs for run id {run_id}"))
            }
        );

        self
    }

    pub fn with_list_motif_slices(
        mut self,
        motif_id: u32,
        select_motifs_run_id: Option<Vec<RunId>>,
        limit: Option<u32>,
        all_but_in_run_id: Option<(RunId, u32)>, r:Result<(i64, RunId, Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>), ThoError>
    ) -> Self {

        let expected_select = select_motifs_run_id.clone();

        self.store.expect_list_motif_slices().returning(
            move |p_motif_id, p_select_motifs_run_id, limit, all_but_in_run_id| {
                debug!("with_list_motif_slices: p_motif_id={p_motif_id} motif_id={motif_id} p_select_motifs_run_id={p_select_motifs_run_id:?} expected_select={expected_select:?}");
                if p_motif_id == motif_id && p_select_motifs_run_id == expected_select {
                    r.clone()
                } else {
                    Err(ThoError::new("Not found"))
                }

            }
        );

        self
    }

    pub fn with_list_random_slices_out_of_motif(
        mut self,
        motif_id: u32,
        motif_width: u32,
        min_distance: Option<u32>,
        r: Result<(i64, RunId, Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>), ThoError>
    ) -> Self {
        self.store.expect_list_random_slices_out_of_motif().returning(
            move |p_motif_id, p_width, p_min_distance, _input_len_adjuster, desired_count| {
                if p_motif_id == motif_id && p_width == motif_width && p_min_distance == min_distance {
                    r.clone()
                } else {
                    Err(ThoError::new("Not found"))
                }
            }
        );

        self
    }

    pub fn with_list_selected_motifs<R:Into<HashMap<RunId, Result<Vec<Vec<(u32, f64)>>, ThoError>>>>(
        mut self,
        results: R
    ) -> Self {
        let map = results.into();
        self.store.expect_list_selected_motifs().returning(
            move |run_id, channels|{
                map.get(&run_id).cloned().expect(&format!("mock is not configured to return selected motifs for run id {run_id}"))
            }
        );
        self
    }

    pub fn with_save_selected_net(
        mut self,
        inputs: impl IntoIterator<Item = (RunId, PathBuf, f32)> + Debug,
    ) -> Self {
        debug!("with_save_selected_net: inputs={inputs:?}");
        for (run_id, path, loss) in inputs {
            self.store
                .expect_save_selected_net()
                .with(eq(run_id), eq(path), eq(loss))
                .times(1)
                .return_const(Ok(()));
        }
        self
    }

    pub fn with_list_selected_nets<I>(
        mut self,
        inputs: I,
    ) -> Self
    where
        I: IntoIterator<Item = (RunId, Option<f32>, Option<usize>, Result<Vec<(PathBuf, f32)>, ThoError>)> + Debug,
    {
        let data = inputs.into_iter().collect::<Vec<_>>();
        debug!("with_list_selected_nets: inputs={data:?}");
        let data = Arc::new(data);
        self.store.expect_list_selected_nets().returning(
            move |run_id, loss, limit| {
                data.iter()
                    .find(|(expected_run_id, expected_loss, expected_limit, _)| *expected_run_id == run_id && *expected_loss == loss && *expected_limit == limit)
                    .map(|(_, _, _, result)| result.clone())
                    .unwrap_or_else(|| Err(ThoError::new(format!("mock is not configured to return selected nets for run id {run_id} loss={loss:?} limit={limit:?}"))))
            }
        );
        self
    }

    pub fn with_list_run_config_values(mut self, run_id:RunId, r:Result<HashMap<String, Val>, ThoError>) -> Self {
        self.store.expect_list_run_config_values()
            .returning(move |p_run_id| {
                if p_run_id == run_id {
                    r.clone()
                } else {
                    Err(ThoError::new("Not found"))
                }
            });

        self
    }

    pub fn with_list_output_run_ids<R:Into<HashMap<RunId, Result<Vec<RunId>, ThoError>>>>(
        mut self,
        results: R,
    ) -> Self {
        let map = results.into();
        self.store.expect_list_output_run_ids().returning(
            move |input_run_id| {
                map.get(&input_run_id)
                    .cloned()
                    .expect(&format!("mock is not configured to return output run ids for run id {input_run_id}"))
            },
        );
        self
    }

    pub fn with_list_run_ids_with_outputs<R: Into<HashMap<Vec<RunId>, Result<Vec<RunId>, ThoError>>>>(
        mut self,
        results: R,
    ) -> Self {
        let map = results.into();
        self.store.expect_list_run_ids_with_outputs().returning(
            move |input_run_ids| {
                map.get(&input_run_ids.to_vec())
                    .cloned()
                    .expect(&format!("mock is not configured to return run ids with outputs for input ids {input_run_ids:?}"))
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
                    result.clone()
                } else {
                    Err(ThoError::new("Not found"))
                }
            });

        self
    }

    pub fn with_list_resource_set_group_sets<D:IntoIterator<Item=(&'static str, Result<Vec<(usize, String)>, ThoError>)>>(mut self, data:D) ->Self {
        let map :HashMap<&'static str, Result<Vec<(usize, String)>, ThoError>> = HashMap::from_iter(data);

        self.store.expect_list_resource_set_group_sets()
            .returning(move |group| {
                map.get(group).expect(&format!("no mock data for {group} configured in list_group_labeled_resources_sets")).clone()
            });

        self
    }

    pub fn with_list_group_labeled_resources_sets<D:IntoIterator<Item=(&'static str, Result<Vec<(String, String)>, ThoError>)>>(mut self, data:D) -> Self {
        let map :HashMap<&'static str, Result<Vec<(String, String)>, ThoError>> = HashMap::from_iter(data.into_iter());

    self.store.expect_list_group_labeled_resources_sets()
        .returning(move |group| {
            map.get(group).expect(&format!("no mock data for {group} configured in list_group_labeled_resources_sets")).clone()
        });

        self
    }

    pub fn with_list_resources_not_in_group<D:IntoIterator<Item=(&'static str, Result<Vec<Resource>, ThoError>)>>(mut self, data:D) -> Self {
        let map :HashMap<&'static str, Result<Vec<Resource>, ThoError>> = HashMap::from_iter(data.into_iter());

        self.store.expect_list_resources_not_in_group()
            .returning(move |group, _limit| {
                map.get(group).expect(&format!("no mock data for {group} configured in list_resource_set_resources_by_descr")).clone()
            });


        self
    }

    pub fn with_list_random_resources<D:IntoIterator<Item=(Option<u32>, Result<Vec<Resource>, ThoError>)>>(mut self, data:D) -> Self {
        let map :HashMap<Option<u32>, Result<Vec<Resource>, ThoError>> = HashMap::from_iter(data.into_iter());

        self.store.expect_list_random_resources()
            .returning(move |limit| {
                map.get(&limit).expect(&format!("no mock data for limit={limit:?} configured in list_random_resources")).clone()
            });


        self
    }

    pub fn with_list_resource_random_not_labeled<D:IntoIterator<Item=(&'static str, Result<Vec<Resource>, ThoError>)>>(mut self, data:D) -> Self {
        let map :HashMap<&'static str, Result<Vec<Resource>, ThoError>> = HashMap::from_iter(data.into_iter());

        self.store.expect_list_resource_random_not_labeled()
            .returning(move |label, _limit| {
                map.get(label).expect(&format!("no mock data for {label} configured in list_resources_not_labled")).clone()
            });

        self
    }

    pub fn with_list_resource_random_in_group<D:IntoIterator<Item=(&'static str, Result<Vec<(Resource, Vec<f32>)>, ThoError>)>>(mut self, data:D) -> Self {
        let map: HashMap<&'static str, Result<Vec<(Resource, Vec<f32>)>, ThoError>> = HashMap::from_iter(data.into_iter());

        self.store.expect_list_resource_random_in_group()
            .returning(move |group| {
                map.get(group).expect(&format!("no mock data for {group} configured in list_resource_random_in_group")).clone()
            });

        self
    }

    pub fn with_list_resource_set_resources_by_descr<D:IntoIterator<Item=(&'static str, Result<Vec<Resource>, ThoError>)>>(mut self, data:D) -> Self {
        let map :HashMap<&'static str, Result<Vec<Resource>, ThoError>> = HashMap::from_iter(data.into_iter());

        self.store.expect_list_resource_set_resources_by_descr()
        .returning(move |group| {
            map.get(group).expect(&format!("no mock data for {group} configured in list_resource_set_resources_by_descr")).clone()
        });


        self
    }

    pub fn with_save_groups_as_motifs(mut self)->Self{
        self.store.expect_save_groups_as_motifs()
            .withf(|run_id, channel, groups|{
                debug!("with_save_groups_as_motifs: run_id={run_id:?}, channel={channel:?}, groups={groups:?}");
                true
            })
            .return_const(Ok(()));
        self
    }

    pub fn build(self)->Box<dyn Store> {
        Box::new(self.store)
    }

}


pub fn mock_store_with_find_motifs_run_config(
    step: i32,
    window: i32,
    selected_motifs: Vec<Vec<(u32, f64)>>,
) -> Box<dyn Store> {
    let mut mock = MockStore::new();

    mock.expect_get_input_run_ids().return_const(Ok(vec![0]));
    mock.expect_list_run_config_values()
        .return_const(Ok(HashMap::from([
            ("FindMotifsRunConfig::id".to_string(), 1.into()),
            ("FindMotifsRunConfig::step".to_string(), step.into()),
            ("FindMotifsRunConfig::window".to_string(), window.into()),
            (
                "FindMotifsRunConfig::slices_filter.distance_running_mean_k".to_string(),
                true.into(),
            ),
            (
                "FindMotifsRunConfig::slices_filter.distance_running_mean_k.mean".to_string(),
                0.1.into(),
            ),
        ])));

    mock.expect_list_selected_motifs()
        .return_const(Ok(selected_motifs));

    Box::new(mock)
}
