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

use bytemuck::{Pod, Zeroable};
use parking_lot::{Mutex, RwLock};
use std::array::IntoIter;
use std::cmp::{max, min};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use log::{debug, error, trace, warn};
use mockall::mock;
use mockall::predicate::eq;
use rand::Rng;
use rand::seq::SliceRandom;
use rusqlite::types::Value;
use rusqlite::{Connection, Error, Transaction, params};

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

pub trait Store: ResourceStore + Debug + Sync + Send {
    fn save_import<'a>(&self, descr: String, t: &'a Transaction<'a>) -> Result<u32, ThoosaError>;
    fn save_run_config<'a>(
        &self,
        input_run_id: Option<&'a [RunId]>,
        run_config_type: &'a dyn Display,
        descr: &'a str,
        _keys: &'a str,
        properties_vals: HashMap<String, Val>,
    ) -> Result<RunId, ThoosaError>;

    fn get_input_run_ids<'a>(&self, run_id: RunId, input_type: Option<&'a dyn Display>) -> Result<Vec<RunId>, ThoosaError>;

    fn list_input_run_ids(&self, ids: &[RunId]) -> Result<Vec<(RunId, RunId)>, ThoosaError>; //corr_run_id, run_id, label

    fn list_output_run_ids(&mut self, input_run_id: RunId) -> Result<Vec<RunId>, ThoosaError>; //corr_run_id, run_id, label

    fn save_resources_set(
        &mut self,
        set: String,
        resource_keys: Vec<String>,
        group_id: Option<usize>,
    ) -> Result<usize, ThoosaError>;
    fn save_resource_set_group(&mut self, descr: String) -> Result<usize, ThoosaError>;
    fn save_resources_labels<'a>(
        &self,
        resource_stats_labels_probs: &'a HashMap<String, (ResourceStats, Vec<(String, f32)>)>,
        import_id: u32,
        t: &'a Transaction<'a>,
    ) -> Result<Vec<ResourceId>, ThoosaError>;
    fn save_pure_sets<'a>(
        &self,
        max_len: usize,
        only_unused: bool,
        t: &'a Transaction<'a>,
    ) -> Result<(), ThoosaError>;
    fn save_set_by_labels(
        &mut self,
        conds: Vec<(String, SetCond)>,
        only_unused: bool,
        limit: usize,
    ) -> Result<(), ThoosaError>;
    fn save_groups_by_label_cnt<'a>(
        &self,
        max_resources_in_set: usize,
        t: &'a Transaction<'a>,
    ) -> Result<(), ThoosaError>;
    fn list_resources_by_ids(
        &self,
        file_stem: &Vec<ResourceId>,
    ) -> Result<Vec<Resource>, ThoosaError>;
    fn list_resource_random_not_label(
        &self,
        label: &str,
        limit: i64,
    ) -> Result<Vec<Resource>, ThoosaError>;

    fn save_label<'a>(&self, label: &str, t: &'a Transaction<'a>) -> Result<u32, ThoosaError>;

    fn list_labels(&self) -> Result<Vec<String>, ThoosaError>;

    fn save_params_value<'a>(&mut self, params: HashMap<String, Val>, run_id: RunId, label: Option<&'a str>) -> Result<(), ThoosaError>;
    fn save_slices(&mut self, run_id: RunId, channel: i64, slices: Vec<(TensorIdSliceIdx, TensorIdSliceIdx, f32)>) -> Result<(), ThoosaError>;

    fn save_groups_as_motifs_with_slices_matches(&mut self, run_id: RunId, channel: i64, groups: HashMap<u16, HashSet<IdSlice>>, as_ref: &dyn AsRef<Path>) -> Result<(), ThoosaError>;

    fn save_groups_as_motifs(&mut self, run_id: RunId, channel: i64, groups: HashMap<u16, HashSet<IdSlice>>) -> Result<(), ThoosaError>;

    fn add_slices_to_motif(&mut self, run_id: RunId, motif_id: u32, channel: i64, slices: HashSet<IdSlice>) -> Result<(), ThoosaError>;

    fn list_motifs(&self, run_id: RunId) -> Result<Vec<usize>, ThoosaError>;
    fn get_motif_meta(&self, motif_id: u32) -> Result<(usize, RunId), ThoosaError>;
    fn list_motifs_slices(
        &self,
        run_id: RunId,
    ) -> Result<HashMap<Resource, Vec<MotifSlice>>, ThoosaError>;

    fn list_motif_slices(&self, motif_id: u32, limit: Option<u32>, all_but_in_run_id: Option<(RunId, u32)>) ->
    Result<(
        i64,
        RunId,
        Vec<TensorIdSliceIdx>,
        HashMap<ResourceId, Resource>,
    ), ThoosaError>;

    fn list_random_slices(
        &self,
        motif_id: u32,
        motif_width: u32,
        min_distance: Option<u32>,
    ) -> Result<
        (
            i64,
            RunId,
            Vec<TensorIdSliceIdx>,
            HashMap<ResourceId, Resource>,
        ),
        ThoosaError,
    >;

    fn list_run_resources(&self, run_id: RunId) -> Result<Vec<String>, ThoosaError>;

    fn list_run_config_values(&self, run_id: RunId) -> Result<HashMap<String, Val>, ThoosaError>;

    fn save_selected_motifs(&mut self, run_id: RunId, selected_motifs: Vec<(u32, f64)>) -> Result<RunId, ThoosaError>;

    fn list_selected_motifs(&self, motif_selection_id: RunId, channels: u32) -> Result<Vec<Vec<(u32, f64)>>, ThoosaError>;

    fn run_config_descr(&mut self, corr_run_id: RunId) -> Result<Vec<(i64, String)>, ThoosaError>;
    fn conn(&self) -> &parking_lot::Mutex<Connection>; //parking_lot:: for mockall
}

mock! {
    pub Store {}

    impl Store for Store {
        fn save_import<'a>(&self, descr: String, t: &'a Transaction<'a>) -> Result<u32, ThoosaError>;
        fn save_run_config<'a>(
            &self,
            input_run_id: Option<&'a [RunId]>,
            run_config_type: &'a dyn Display,
            descr: &'a str,
            _keys: &'a str,
            properties_vals: HashMap<String, Val>,
        ) -> Result<RunId, ThoosaError>;
        fn get_input_run_ids<'a>(&self,run_id: RunId,input_type: Option<&'a dyn Display>,) -> Result<Vec<RunId>, ThoosaError>;
        fn list_input_run_ids(&self, ids: &[RunId]) -> Result<Vec<(RunId, RunId)>, ThoosaError>;
        fn list_output_run_ids(&mut self, input_run_id: RunId) -> Result<Vec<RunId>, ThoosaError>;

        fn save_resources_set(
            &mut self,
            set: String,
            resource_keys: Vec<String>,
            group_id: Option<usize>,
        ) -> Result<usize, ThoosaError>;
        fn save_resource_set_group(&mut self, descr: String) -> Result<usize, ThoosaError>;
        fn save_resources_labels<'a>(
            &self,
            resource_stats_labels_probs: &'a HashMap<String, (ResourceStats, Vec<(String, f32)>)>,
            import_id: u32,
            t: &'a Transaction<'a>,
        ) -> Result<Vec<ResourceId>, ThoosaError>;
        fn save_pure_sets<'a>(
            &self,
            max_len: usize,
            only_unused: bool,
            t: &'a Transaction<'a>,
        ) -> Result<(), ThoosaError>;
        fn save_set_by_labels(
            &mut self,
            conds: Vec<(String, SetCond)>,
            only_unused: bool,
            limit: usize,
        ) -> Result<(), ThoosaError>;
        fn save_groups_by_label_cnt<'a>(
            &self,
            max_resources_in_set: usize,
            t: &'a Transaction<'a>,
        ) -> Result<(), ThoosaError>;
        fn list_resources_by_ids(
            &self,
            file_stem: &Vec<ResourceId>,
        ) -> Result<Vec<Resource>, ThoosaError>;
        fn list_resource_random_not_label(
            &self,
            label: &str,
            limit: i64,
        ) -> Result<Vec<Resource>, ThoosaError>;
        fn save_label<'a>(&self, label: &str, t: &'a Transaction<'a>) -> Result<u32, ThoosaError>;
        fn list_labels(&self) -> Result<Vec<String>, ThoosaError>;
        fn save_params_value<'a>(
            &mut self,
            params: HashMap<String, Val>,
            run_id: RunId,
            label: Option<&'a str>,
        ) -> Result<(), ThoosaError>;
        fn save_slices(
            &mut self,
            run_id: RunId,
            channel: i64,
            slices: Vec<(TensorIdSliceIdx, TensorIdSliceIdx, f32)>,
        ) -> Result<(), ThoosaError>;
        fn save_groups_as_motifs_with_slices_matches(
            &mut self,
            run_id: RunId,
            channel: i64,
            groups: HashMap<u16, HashSet<IdSlice>>,
            as_ref: &dyn AsRef<Path>,
        ) -> Result<(), ThoosaError>;
        fn save_groups_as_motifs(
            &mut self,
            run_id: RunId,
            channel: i64,
            groups: HashMap<u16, HashSet<IdSlice>>,
        ) -> Result<(), ThoosaError>;

        fn add_slices_to_motif(
            &mut self,
            run_id: RunId,
            motif_id: u32,
            channel: i64,
            groups: HashSet<IdSlice>,
        ) -> Result<(), ThoosaError>;

        fn list_motifs(&self, run_id: RunId) -> Result<Vec<usize>, ThoosaError>;
        fn list_motifs_slices(
            &self,
            run_id: RunId,
        ) -> Result<HashMap<Resource, Vec<MotifSlice>>, ThoosaError>;

        fn get_motif_meta(&self, motif_id: u32) -> Result<(usize, RunId), ThoosaError>;

        fn list_motif_slices(
            &self,
            motif_id: u32,
            limit: Option<u32>,
            all_but_in_run_id: Option<(RunId, u32)>,
        ) -> Result<
            (
                i64,
                RunId,
                Vec<TensorIdSliceIdx>,
                HashMap<ResourceId, Resource>,
            ),
            ThoosaError,
        >;
        fn list_random_slices(
            &self,
            motif_id: u32,
            motif_width: u32,
            min_distance: Option<u32>,
        ) -> Result<
            (
                i64,
                RunId,
                Vec<TensorIdSliceIdx>,
                HashMap<ResourceId, Resource>,
            ),
            ThoosaError,
        >;
        fn list_run_resources(&self, run_id: RunId) -> Result<Vec<String>, ThoosaError>;
        fn list_run_config_values(&self, run_id: RunId) -> Result<HashMap<String, Val>, ThoosaError>;
        fn save_selected_motifs(
            &mut self,
            run_id: RunId,
            selected_motifs: Vec<(u32, f64)>,
        ) -> Result<RunId, ThoosaError>;

        fn list_selected_motifs(
            &self,
            motif_selection_id: RunId,
            channels: u32,
        ) -> Result<Vec<Vec<(u32, f64)>>, ThoosaError>;
        fn run_config_descr(&mut self, corr_run_id: RunId) -> Result<Vec<(i64, String)>, ThoosaError>;
        fn conn(&self) -> &parking_lot::Mutex<Connection>;
    }

    impl ResourceStore for Store {
        fn list_resources_by_file_stems<'a>(
            &self,
            file_stems: &'a [&'a str],
        ) -> Result<Vec<Resource>, ThoosaError>;

        fn list_resource_set_resources_by_descr(
            &self,
            descr: &str,
        ) -> Result<Vec<Resource>, ThoosaError>;

        fn list_resource_set_group_sets(
            &self,
            group_descr: &str,
        ) -> Result<Vec<(usize, String)>, ThoosaError>;

        fn list_group_labeled_resources_sets(
            &self,
            group_decr: &str,
        ) -> Result<Vec<(String, String)>, ThoosaError>;

        fn list_resource_set_resources(&self, set_id: usize) -> Result<Vec<Resource>, ThoosaError>;

        fn list_resource_random_in_group(
            &self,
            group_descr: &str,
        ) -> Result<Vec<(Resource, Vec<f32>)>, ThoosaError>;

        fn list_resources_not_in_group(
            &self,
            group_descr: &str,
        ) -> Result<Vec<Resource>, ThoosaError>;

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
    conn: Mutex<Connection>,
}

impl ResourceStore for SqlStore {
    fn list_resource_set_resources_by_descr(
        &self,
        descr: &str,
    ) -> Result<Vec<Resource>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let r = Self::list_resource_set_resources(
            t.query_row(
                "SELECT id FROM resource_set WHERE descr=?",
                [descr],
                |row| row.get(0),
            )?,
            &t,
        )?;
        debug!("list_resource_set_resources_by_descr: descr={descr} r={r:?}");

        t.commit()?;

        Ok(r)
    }
    fn list_resources_by_file_stems<'a>(
        &self,
        file_stems: &'a [&'a str],
    ) -> Result<Vec<Resource>, ThoosaError> {
        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;

        let (mut q, params) = file_stems.into_iter().fold(
            (
                vec!["SELECT id, import_id, key, length FROM resources WHERE "],
                vec![],
            ),
            |mut a, v| {
                a.0.push(" key LIKE ?");
                a.0.push(" OR");
                a.1.push(*v);
                a
            },
        );

        q.pop();
        q.push(" ORDER BY key");

        let resources = {
            let mut stmt = t.prepare(q.into_iter().collect::<String>().as_str())?;

            debug!("list_resources_by_file_stems: stmt={stmt:?}");

            let r = stmt
                .query_map(rusqlite::params_from_iter(params.into_iter()), |row| {
                    Ok(Resource::new(
                        row.get::<usize, u32>(0)?,
                        row.get::<usize, u32>(1)?,
                        row.get::<usize, String>(2)?,
                        row.get::<usize, u32>(3)?,
                    ))
                })?
                .map(|v| v.unwrap())
                .collect();

            r
        };

        Ok(resources)
    }

    fn list_resource_set_group_sets(
        &self,
        group_descr: &str,
    ) -> Result<Vec<(usize, String)>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let group_sets = {
            let mut stmt = t.prepare("SELECT rs.id, rs.descr FROM resource_set rs LEFT JOIN resource_set_group rsg ON rsg.id=rs.group_id WHERE rsg.descr=?")?;

            let vec = stmt
                .query_map([group_descr], |row| {
                    //TODO move group_slices out, not for store
                    Ok((
                        row.get::<usize, usize>(0)
                            .expect("row 0 in list_resource_set_group_sets"),
                        row.get::<usize, String>(1)
                            .expect("row 1 in list_resource_set_group_sets"),
                    ))
                })?
                .map(|v| v.unwrap())
                .collect();

            vec
        };

        t.commit()?;

        Ok(group_sets)
    }

    ///returns Vec<(label, resource_set_descr)>
    fn list_group_labeled_resources_sets(
        &self,
        group_descr: &str,
    ) -> Result<Vec<(String, String)>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let labeled_resources_sets = {
            let mut stmt = t.prepare("SELECT l.label, rs.descr FROM resource_set_group rsg JOIN resource_set rs ON rs.group_id=rsg.id JOIN labels l ON l.id=rs.label_id WHERE rsg.descr=?")?;

            let vec = stmt
                .query_map([group_descr], |row| {
                    //TODO move group_slices out, not for store
                    Ok((
                        row.get::<usize, String>(0)
                            .expect("row 0 in list_group_labeled_resources_sets"),
                        row.get::<usize, String>(1)
                            .expect("row 1 in list_group_labeled_resources_sets"),
                    ))
                })?
                .map(|v| v.unwrap())
                .collect();

            vec
        };

        t.commit()?;

        debug!(
            "list_group_labeled_resources_sets: labeled_resources_sets={labeled_resources_sets:?}"
        );

        Ok(labeled_resources_sets)
    }

    fn list_resource_set_resources(&self, set_id: usize) -> Result<Vec<Resource>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let set_resources = Self::list_resource_set_resources(set_id, &t)?;

        t.commit()?;

        Ok(set_resources)
    }

    fn list_resource_random_in_group(
        &self,
        group_descr: &str,
    ) -> Result<Vec<(Resource, Vec<f32>)>, ThoosaError> {
        //vec<(key, vec<(lablel, p)>>
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let labels = Self::list_labels(&t)?;

        debug!(
            "list_resource_random_in_group: group_descr={:?} labels={:?}",
            group_descr, labels
        );

        let r: Vec<(Resource, Vec<f32>)> = t.prepare(
            "SELECT r.id, r.import_id, r.key, GROUP_CONCAT(l.label || '/' || lr.p, ','), r.length FROM resource_set_group rsg \
            JOIN resource_set rs ON rs.group_id=rsg.id \
            JOIN resource_resources_set rrs ON rrs.set_id=rs.id \
            JOIN resources r ON r.id=rrs.resource_id \
            JOIN labeled_resources lr ON lr.resource_id=r.id \
            JOIN labels l ON l.id=lr.label_id \
            WHERE rsg.descr=? GROUP BY r.key ORDER BY RANDOM()"
        )?.query_map([Value::Text(group_descr.to_string())], |row| {
            let key = row.get::<usize, u32>(0)?;
            let labels_p = row.get::<usize, String>(3)?;

            let labels_p_vec = labels_p.split(",")
                .map(|label_p| {
                    let mut label_p_v = label_p.split("/");
                    (
                        label_p_v.next().expect("label").into(),
                        label_p_v.next().expect("p").parse::<f32>().expect("parse")
                    )
                }
                ).fold(HashMap::new(), |mut a: HashMap<String, f32>, v| {
                a.insert(v.0, v.1);
                a
            });

            trace!("list_resource_random_in_group: key={} labels_p={} labels_p_vec={:?}", key, labels_p, labels_p_vec);

            Ok(
                (
                    Resource::new(
                        row.get::<usize, u32>(0)?,
                        row.get::<usize, u32>(1)?,
                        row.get::<usize, String>(2)?,
                        row.get::<usize, u32>(4)?
                    ),
                    labels.iter().map(|(_label_id, label)| *labels_p_vec.get(label).unwrap_or(&0.0)).collect::<Vec<f32>>()
                )
            )
        })?.map(|v| v.unwrap()).collect();

        t.commit()?;

        debug!("list_resource_random_in_group: r.len={:?}", r.len());

        Ok(r)
    }

    fn list_resources_not_in_group(
        &self,
        group_descr: &str,
    ) -> Result<Vec<Resource>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let group_sets = {
            let mut stmt = t.prepare("SELECT r.id, r.import_id, r.key, r.length FROM resources AS r WHERE NOT EXISTS ( SELECT 1 FROM resource_resources_set AS rrs JOIN resource_set        AS rs  ON rs.id = rrs.set_id JOIN resource_set_group  AS g   ON g.id = rs.group_id WHERE rrs.resource_id = r.id AND g.descr = ? )")?;

            let vec = stmt
                .query_map([group_descr], |row| {
                    //TODO move group_slices out, not for store
                    Ok(Resource::new(
                        row.get::<usize, u32>(0)?,
                        row.get::<usize, u32>(1)?,
                        row.get::<usize, String>(2)?,
                        row.get::<usize, u32>(3)?
                    ))
                })?
                .map(|v| v.unwrap())
                .collect();

            vec
        };

        t.commit()?;

        Ok(group_sets)

    }

}

//TODO move out all prepare stmts
impl Store for SqlStore {
    fn save_import(&self, descr: String, t: &Transaction) -> Result<u32, ThoosaError> {
        t.execute("INSERT INTO imports (descr) VALUES (?)", params![descr])?;

        let import_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;

        Ok(import_id)
    }

    //TODO resources - save serialized resources config - run_resources is getting to big
    fn save_run_config(
        &self,
        input_run_id: Option<&[RunId]>,
        run_config_type: &dyn Display,
        descr: &str,
        _keys: &str,
        properties_vals: HashMap<String, Val>,
    ) -> Result<RunId, ThoosaError> {
        debug!("SqlStore::save_run_config: input_run_id={input_run_id:?} run_config_type={run_config_type} descr={descr}");

        let mut conn_guard = self.conn.lock();
        let t = conn_guard
            .transaction()
            .map_err(|e| ThoosaError::new(e.to_string()))?; //TODO error

        t.execute(
            "INSERT INTO run_configs (type, descr) VALUES (?, ?)",
            params![run_config_type.to_string(), descr],
        )?;

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

                    params.push(Value::Integer(run_id as i64)); //resource id
                    params.push(Value::Integer(*v as i64)); //label id
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
    }

    fn get_input_run_ids(
        &self,
        run_id: RunId,
        input_type: Option<&dyn Display>,
    ) -> Result<Vec<RunId>, ThoosaError> {
        debug!("SqlStore::get_input_run_ids: run_id={run_id} input_type={:?}",input_type.map(|v| v.to_string()));

        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;

        let input_run_ids = { Self::get_input_run_ids_(&t, run_id, input_type)? };

        t.commit()?;

        debug!("get_input_run_ids: input_run_ids={input_run_ids:?}");

        Ok(input_run_ids)
    }

    /// returns - Vec<run_id, input_run_id>
    fn list_input_run_ids(&self, ids: &[RunId]) -> Result<Vec<(RunId, RunId)>, ThoosaError> {
        assert!(!ids.is_empty());
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let (mut q, params) = ids.into_iter().fold(
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

        let id_input_id = {
            let mut stmt = t.prepare(q.into_iter().collect::<String>().as_str())?;

            let r = stmt
                .query_map(rusqlite::params_from_iter(params.into_iter()), |row| {
                    Ok((row.get::<usize, RunId>(0)?, row.get::<usize, RunId>(1)?))
                })?
                .map(|v| v.unwrap())
                .collect();

            r
        };
        t.commit()?;

        Ok(id_input_id)
    }


    //TODO remove?
    fn save_params_value(
        &mut self,
        params: HashMap<String, Val>,
        run_id: RunId,
        label: Option<&str>,
    ) -> Result<(), ThoosaError> {
        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;

        let label_id = t
            .query_row("SELECT id FROM labels WHERE label=?", [label], |row| {
                row.get(0)
            })
            .ok();

        for property_val in params.into_iter() {
            Self::save_param_value(&t, run_id, &property_val.0, property_val.1, label_id)?;
        }

        t.commit()?;

        Ok(())
    }

    fn save_label(&self, label: &str, t: &Transaction) -> Result<u32, ThoosaError> {
        t.execute("INSERT INTO labels (label) VALUES (?)", params![label])?;

        let label_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;

        Ok(label_id)
    }

    fn list_labels(&self) -> Result<Vec<String>, ThoosaError> {
        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;

        let labels = Self::list_labels(&t)?
            .into_iter()
            .map(|id_label| id_label.1)
            .collect();

        t.commit()?;

        Ok(labels)
    }

    //TODO folds to util fn
    fn save_resources_labels(
        &self,
        resource_stats_labels_probs: &HashMap<String, (ResourceStats, Vec<(String, f32)>)>,
        import_id: u32,
        t: &Transaction,
    ) -> Result<Vec<ResourceId>, ThoosaError> {
        debug!(
            "save_resources_labels: resource_labels_probs.len()={}",
            resource_stats_labels_probs.len()
        );

        let resources_len = resource_stats_labels_probs.len();

        let mut resource_labels_iter = resource_stats_labels_probs
            .into_iter()
            .array_chunks::<500>();

        let mut resource_ids = Vec::with_capacity(resources_len);

        while let Some(resource_keys) = resource_labels_iter.next() {
            resource_ids.extend(do_insert(resource_keys.into_iter(), import_id, &t)?);
        }

        if let Some(reminder) = resource_labels_iter.into_remainder() {
            if reminder.len() > 0 {
                resource_ids.extend(do_insert(reminder, import_id, &t)?);
            }
        }

        fn do_insert(
            resource_keys_iter: IntoIter<(&String, &(ResourceStats, Vec<(String, f32)>)), 500>,
            import_id: u32,
            t: &Transaction,
        ) -> Result<Vec<ResourceId>, ThoosaError> {
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
                     (resource_key, (stats, labels))| {
                        insert_resources_params.push(Value::Text(resource_key.clone()));
                        insert_resources_params.push(Value::Integer(stats.length() as i64));
                        insert_resources_params.push(Value::Integer(import_id as i64));
                        insert_resources_q.push("(?, ?, ?)");
                        insert_resources_q.push(",");

                        let labels_ids = labels
                            .into_iter()
                            .map(|(label, p)| {
                                (
                                    t.query_row(
                                        "SELECT id FROM labels WHERE label=?",
                                        [&label],
                                        |row| row.get(0),
                                    )
                                    .expect(format!("label {} not found", label).as_str()),
                                    p,
                                )
                            })
                            .collect::<Vec<(u32, &f32)>>();
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

            let mut rows = stmt.query(rusqlite::params_from_iter(
                insert_resources_params.into_iter(),
            ))?;

            let mut resource_ids: Vec<u32> = vec![];
            while let Some(row) = rows.next()? {
                //TODO  query_map ? not needed?
                resource_ids.push(row.get_unwrap(0));
            }

            let mut labels_iter = resource_ids
                .iter()
                .zip(insert_labeled_resources_params)
                .flat_map(move |(resource_id, labels)| {
                    labels.into_iter().map(move |(l, p)| (resource_id, l, p))
                })
                .array_chunks::<500>();

            while let Some(resource_labels) = labels_iter.next() {
                insert_labels(resource_labels.into_iter(), t);
            }

            if let Some(reminder) = labels_iter.into_remainder() {
                insert_labels(reminder, t);
            }

            Ok(resource_ids)
        }

        fn insert_labels(resource_labels: IntoIter<(&u32, u32, &f32), 500>, t: &Transaction) {
            let (mut q, params) = resource_labels.fold(
                (
                    vec!["INSERT INTO labeled_resources (resource_id, label_id, p) VALUES"],
                    vec![],
                ),
                |(mut q, mut params): (Vec<&str>, Vec<Value>), v| {
                    q.push("(?, ?, ?)");
                    q.push(",");

                    params.push(Value::Integer(*v.0 as i64)); //resource id
                    params.push(Value::Integer(v.1 as i64)); //label id
                    params.push(Value::Real(*v.2 as f64)); //p
                    //
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

        Ok(resource_ids)
    }

    //TODO used in tests only, remove?
    fn save_resources_set(
        &mut self,
        descr: String,
        resource_keys: Vec<String>,
        group_id: Option<usize>,
    ) -> Result<usize, ThoosaError> {
        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;

        match group_id {
            Some(group_id) => {
                t.execute(
                    "INSERT INTO resource_set (descr, group_id) VALUES (?, ?)",
                    params![descr, group_id],
                )?;
            }
            None => {
                t.execute(
                    "INSERT INTO resource_set (descr) VALUES (?)",
                    params![descr],
                )?;
            }
        };

        let resource_set_id = t.query_row("SELECT last_insert_rowid()", [], |row| {
            row.get::<usize, usize>(0)
        })?;

        let q = vec![
            "INSERT INTO resource_resources_set (set_id, resource_id) SELECT ?, id FROM resources WHERE key IN (",
        ];
        let resoure_id_str = resource_set_id.to_string();
        let params = vec![resoure_id_str.as_str()];
        for resources_chunk in resource_keys.chunks(500) {
            let (mut q, params) = resources_chunk.into_iter().fold(
                (q.clone(), params.clone()),
                |(mut q, mut params), key| {
                    params.push(key.as_str());
                    q.push("?");
                    q.push(",");
                    (q, params)
                },
            );

            q.pop();
            q.push(")");
            t.execute(
                &q.into_iter().collect::<String>(),
                rusqlite::params_from_iter(params.into_iter()),
            )?;
        }

        t.commit()?;

        Ok(resource_set_id)
    }

    fn save_resource_set_group(&mut self, descr: String) -> Result<usize, ThoosaError> {
        let mut conn_guard = self.conn.lock();
        let mut t = conn_guard.transaction()?;

        let group_id = Self::save_resource_set_group(descr, &mut t)?;

        t.commit()?;

        Ok(group_id)
    }

    fn save_groups_by_label_cnt(
        &self,
        max_resources_in_set: usize,
        t: &Transaction,
    ) -> Result<(), ThoosaError> {
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

            debug!(
                "save_groups_by_label_cnt: labels_cnt_labels_resources.len={:?}",
                labels_cnt_labels_resources.len()
            );
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
                let max_labels_resources =
                    labels_resources.values().map(|r| r.len()).sum::<usize>();
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
                                    t,
                                )
                                .map(|g| (i, i + *max_r, g))
                            })
                            .collect::<Result<Vec<(usize, usize, usize)>, ThoosaError>>()
                    })
                    .collect::<Result<Vec<Vec<(usize, usize, usize)>>, ThoosaError>>()?;

                let group_all_id =
                    Self::save_resource_set_group(format!("group {} labels/all", labels_cnt), t)?;

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
                                        t,
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
                        t,
                        labels_cnt,
                        group_all_id,
                        format!("{}/{}", labels_cnt, label),
                        label_id,
                        resources.as_slice(),
                    )?;
                }
            }
        }
        Ok(())
    }

    fn save_pure_sets(
        &self,
        max_len: usize,
        only_unused: bool,
        t: &Transaction,
    ) -> Result<(), ThoosaError> {
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

        Ok(())
    }

    fn list_resource_random_not_label(
        &self,
        label: &str,
        limit: i64,
    ) -> Result<Vec<Resource>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let r = t.prepare(
            "SELECT r.id, r.import_id, r.key, r.length FROM resources r INNER JOIN labeled_resources lr ON r.id=lr.resource_id INNER JOIN labels l ON l.id=lr.label_id WHERE l.label != ? ORDER BY RANDOM() limit ?;"
        )?.query_map([Value::Text(label.to_string()), Value::Integer(limit)], |row| { //TODO move group_slices out, not for store
            Ok(Resource::new(
                row.get::<usize, u32>(0)?,
                row.get::<usize, u32>(1)?,
                row.get::<usize, String>(2)?,
                row.get::<usize, u32>(3)?,
            ))
        })?.map(|v| v.unwrap()).collect();

        t.commit()?;

        Ok(r)
    }

    fn list_resources_by_ids(
        &self,
        resource_ids: &Vec<ResourceId>,
    ) -> Result<Vec<Resource>, ThoosaError> {
        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;

        let (mut q, params) = resource_ids.into_iter().fold(
            (
                vec!["SELECT id, import_id, key, length FROM resources WHERE id IN ("],
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
        q.push(")"); // ORDER BY key

        let resources = {
            let mut stmt = t.prepare(q.into_iter().collect::<String>().as_str())?;

            let r = stmt
                .query_map(rusqlite::params_from_iter(params.into_iter()), |row| {
                    Ok(Resource::new(
                        row.get::<usize, u32>(0)?,
                        row.get::<usize, u32>(1)?,
                        row.get::<usize, String>(2)?,
                        row.get::<usize, u32>(3)?,
                    ))
                })?
                .map(|v| v.unwrap())
                .collect();

            r
        };

        Ok(resources)
    }

    //TODO not used, remove?
    fn save_set_by_labels(
        &mut self,
        conds: Vec<(String, SetCond)>,
        _only_unused: bool,
        limit: usize,
    ) -> Result<(), ThoosaError> {
        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;

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
                ThoosaError::new(format!("no resources for set on {:?}", conds));
            }
        }
        Ok(t.commit()?)
    }
    //TODO remove
    fn save_slices(
        &mut self,
        run_id: RunId,
        channel: i64,
        slices: Vec<(TensorIdSliceIdx, TensorIdSliceIdx, f32)>,
    ) -> Result<(), ThoosaError> {
        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;

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
    }

    fn save_groups_as_motifs(
        &mut self,
        run_id: RunId,
        channel: i64,
        groups: HashMap<u16, HashSet<IdSlice>>,
    ) -> Result<(), ThoosaError> {
        let mut guard = self.conn.lock();

        let t = guard.transaction()?;

        debug!("save_slices_groups_as_motifs_from_file: run_id={run_id:?}, channel={channel:?}");

        {
            for (group, slices) in groups {
                let motifs_inserted = t
                    .execute(
                        "INSERT INTO motifs (group_ord, run_id, channel_id) VALUES (?, ?, ?)",
                        [group as usize, run_id as usize, channel as usize],
                    )
                    .map_err(|e| Self::log_error((run_id, channel, group), &t, e))?;

                let motif_id: i64 =
                    t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;

                debug!("save_slices_groups_as_motifs_from_tmp_db: group={group:?}, run_id={run_id:?} motif_id={motif_id} motifs_inserted={motifs_inserted}");
                trace!("save_slices_groups_as_motifs_from_tmp_db: slices={slices:?} ");

                let mut slices_chunks = slices.into_iter().array_chunks::<500>();
                while let Some(chunk) = slices_chunks.next() {
                    Self::insert_slices(&t, run_id, channel, motif_id, chunk.into_iter())?
                }
                if let Some(reminder) = slices_chunks.into_remainder() {
                    if reminder.len() > 0 {
                        Self::insert_slices(&t, run_id, channel, motif_id, reminder)?
                    }
                }
            }
        }

        t.commit()?;

        Ok(())
    }

    fn add_slices_to_motif(
        &mut self,
        run_id: RunId,
        motif_id: u32,
        channel: i64,
        slices: HashSet<IdSlice>,
    ) -> Result<(), ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        debug!("save_slices_groups_as_motifs_from_file: run_id={run_id:?}, channel={channel:?}");

        {
            let mut slices_chunks = slices.into_iter().array_chunks::<500>();
            while let Some(chunk) = slices_chunks.next() {
                Self::insert_slices(&t, run_id, channel, motif_id as i64, chunk.into_iter())?
            }
            if let Some(reminder) = slices_chunks.into_remainder() {
                if reminder.len() > 0 {
                    Self::insert_slices(&t, run_id, channel, motif_id as i64, reminder)?
                }
            }
        }
        t.commit()?;
        Ok(())
    }

    fn save_groups_as_motifs_with_slices_matches(
        &mut self,
        run_id: RunId,
        channel: i64,
        groups: HashMap<u16, HashSet<IdSlice>>,
        as_ref: &dyn AsRef<Path>,
    ) -> Result<(), ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        debug!(
            "save_slices_groups_as_motifs_from_tmp_db: run_id={run_id:?}, channel={channel:?} path={}",
            as_ref.as_ref().to_string_lossy()
        );

        {
            t.execute(
                format!(
                    "ATTACH DATABASE '{}' AS slices_{run_id}_{channel}_tmp",
                    as_ref.as_ref().to_string_lossy()
                )
                .as_str(),
                [],
            )?;

            //TODO split into insert motifs/insert slice_matches + move  group_slices from tracker store?
            for (group, slices) in groups {
                let motifs_inserted = t.execute(
                    "INSERT INTO motifs (group_ord, run_id, channel_id) VALUES (?, ?, ?)",
                    [group as usize, run_id as usize, channel as usize],
                )?; //TODO group not needed?

                let motif_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;

                debug!(
                    "save_slices_groups_as_motifs_from_tmp_db: group={group:?}, run_id={run_id:?} motif_id={motif_id} motifs_inserted={motifs_inserted} slices={slices:?} "
                );

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

                if let Some(reminder) = slices_chunks.into_remainder() {
                    if reminder.len() > 0 {
                        Self::insert_slices_matches_chunk_from_tmp(
                            &t, run_id, channel, motif_id, reminder,
                        )?
                    }
                }
            }
        }

        t.commit()?;

        let t = guard.transaction()?;
        t.execute(
            format!("DETACH DATABASE slices_{run_id}_{channel}_tmp").as_str(),
            [],
        )?;

        Ok(())
    }

    fn get_motif_meta(&self, motif_id: u32) -> Result<(usize, RunId), ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let (channel, run_id) = Self::_motif_meta(motif_id, &t)?;

        t.commit()?;
        Ok((channel as usize, run_id))
    }

    fn list_motifs(&self, run_id: RunId) -> Result<Vec<usize>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;
        let motif_ids = {
            let mut stmt = t.prepare("SELECT id FROM motifs WHERE run_id=?")?;

            let vec = stmt
                .query_map([run_id], |row| {
                    Ok(row.get::<usize, usize>(0)?) // motif_id,
                })?
                .collect::<Result<Vec<_>, Error>>()?;

            vec
        };

        t.commit()?;

        Ok(motif_ids)
    }

    /**
     * returns map tensor_key - MotifSlice(motif_group, tensor slice idx, channel)
     */
    fn list_motifs_slices(
        &self,
        run_id: RunId,
    ) -> Result<HashMap<Resource, Vec<MotifSlice>>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let resources_motifs = {
            let mut stmt = t.prepare(
                "\
                    SELECT m.id, m.channel_id, \
                    r.id, r.import_id, r.key, s.slice_idx, r.length \
                    FROM motifs m \
                    JOIN slices s ON s.motif_id=m.id \
                    JOIN resources r on r.id=s.resource_id \
                    WHERE m.run_id=?\
            ",
            )?;

            let tensor_motifs: (
                HashMap<ResourceId, Vec<MotifSlice>>,
                HashMap<ResourceId, Resource>,
            ) = stmt
                .query_map([run_id], |row| {
                    Ok((
                        row.get(0)?, // motif_id,
                        row.get(1)?, // channel_id,
                        row.get(2)?, // resource_id,
                        row.get(3)?, // import_0,
                        row.get(4)?, // key_0,
                        row.get(5)?, // slice_idx_0,
                        row.get(6)?, // length,
                    ))
                })?
                .fold(
                    Ok((HashMap::new(), HashMap::new())),
                    |a: Result<
                        (
                            HashMap<ResourceId, Vec<MotifSlice>>,
                            HashMap<ResourceId, Resource>,
                        ),
                        Error,
                    >,
                     v: Result<
                        (
                            u32,
                            i64, //motif_id, channel_id
                            u32,
                            u32,
                            String,
                            i64, //resource_id, import_0, key_0, slice_idx_0
                            u32, //length
                        ),
                        Error,
                    >| {
                        let v = v?;

                        let mut a = a?;

                        a.0.entry(v.2)
                            .or_insert_with(|| vec![])
                            .push(MotifSlice::new(v.0, v.5, v.1 as u16));

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
                .fold(
                    (HashMap::new(), tensor_motifs.1),
                    |mut a, (resource_id, slices)| {
                        a.0.insert(
                            a.1.remove(&resource_id)
                                .expect(format!("no resource for {}", resource_id).as_str()),
                            slices,
                        );
                        a
                    },
                )
                .0
        };
        t.commit()?;

        Ok(resources_motifs)
    }

    ///returns channel, find_motifs_run_id, slices, resources map
    fn list_motif_slices(
        &self,
        motif_id: u32,
        limit: Option<u32>,
        all_but_in_run_id: Option<(RunId, u32/*channel*/)>,//TODO remove - load random slices from same channel/resources
    ) -> Result<
        (
            i64,
            RunId,
            Vec<TensorIdSliceIdx>,
            HashMap<ResourceId, Resource>,
        ),
        ThoosaError,
    > {
        debug!("list_motif_slices: motif_id={motif_id} limit={limit:?} all_but_in_run_id={all_but_in_run_id:?}");

        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let channel_slices = {
            let mut stmt = t.prepare(
                &[
                    "SELECT m.channel_id, m.run_id, r.id,  r.import_id, r.key, s.slice_idx \
                    FROM motifs m \
                    LEFT JOIN slices s on s.motif_id=m.id \
                    LEFT JOIN resources r on s.resource_id=r.id \
                    WHERE s.motif_id
                ",
                    // if all_but_in_run_id.is_none() {"=? AND ? IS NOT NULL AND ? IS NOT NULL "} else {"!=? AND run_id=? AND channel_id=?"}, run_id? //TODO
                    if all_but_in_run_id.is_none() {
                        "=? AND ? IS NOT NULL "
                    } else {
                        "!=? AND m.channel_id=(SELECT channel_id FROM motifs WHERE id=?) ORDER BY RANDOM()"
                    },
                    &limit
                        .map(|limit| format!("LIMIT {}", limit))
                        .unwrap_or("".to_owned()),
                ]
                .join(""),
            )?;

            let params = match all_but_in_run_id {
                Some((_run_id, channel_id)) => [motif_id, motif_id],
                None => [motif_id, 1],
            };
            debug!("list_motif_slices: params={params:?}");

            let (slices, channel_set, run_id, resources_map) = stmt
                .query_map(params, |row| {
                    Ok((
                        row.get::<usize, i64>(0)?,   //channel_id
                        row.get::<usize, RunId>(1)?, //run_id
                        row.get(2)?,                 // resource_id,
                        row.get(3)?,                 // import_id,
                        row.get(4)?,                 // key,
                        row.get(5)?,                 // slice_idx,
                    ))
                })?
                .fold(
                    Ok((HashSet::new(), HashSet::new(), None, HashMap::new())),
                    |a: Result<
                        (
                            HashSet<TensorIdSliceIdx>,
                            HashSet<i64>,
                            Option<RunId>,
                            HashMap<ResourceId, Resource>,
                        ),
                        ThoosaError,
                    >,
                     v: Result<
                        (
                            i64,//channel_id
                            RunId, //, run_id
                            u32,//resource_id
                            u32,//import_id
                            String,//key
                            u32, // slice_idx
                        ),
                        Error,
                    >| {
                        
                        let v = v?;
                        let (mut a, mut channel_set, mut run_id, mut resources_map) = a?;

                        // a.insert(TensorSliceIdx(Resource::new(v.2, v.3, v.4, 0), v.5));
                        a.insert(TensorIdSliceIdx(v.2, v.5));
                        resources_map.insert(v.2, Resource::new(v.2, v.3, v.4, 0));

                        channel_set.insert(v.0); //TODO replace with Option

                        if all_but_in_run_id.is_none() {
                            if let Some(run_id) = run_id {
                                assert!(run_id == v.1);
                            } else {
                                run_id.replace(v.1);
                            }
                        }

                        Ok((a, channel_set, run_id, resources_map))
                    },
                )?;

            if all_but_in_run_id.is_none() {
                assert_eq!(channel_set.len(), 1, "channel_set={channel_set:?}");
                (
                    channel_set.into_iter().nth(0).expect("channel"),
                    run_id.expect("no run id"),
                    slices.into_iter().collect(),
                    resources_map,
                )
            } else {
                let (run_id, channel_id) = all_but_in_run_id.unwrap();
                (
                    channel_id as i64,
                    run_id,
                    slices.into_iter().collect(),
                    resources_map,
                ) //TODO
            }
        };

        t.commit()?;

        Ok(channel_slices)
    }

    fn list_random_slices(
        &self,
        motif_id: u32,
        motif_width: u32,
        min_distance: Option<u32>,
    ) -> Result<
        (
            i64, RunId,
            Vec<TensorIdSliceIdx>,
            HashMap<ResourceId, Resource>,
        ),
        ThoosaError,
    > {
        debug!("list_random_slices: motif_id={motif_id} motif_width={motif_width} min_distance={min_distance:?}");

        if motif_width == 0 {
            return Err(ThoosaError::new("motif_width must be greater than zero"));
        }

        // Keep helpers local to minimize surface area.
        fn respects_spacing(candidate: i64, guard_width: i64, sorted: &[i64]) -> bool {
            if sorted.is_empty() {
                return true;
            }

            let idx = sorted.partition_point(|&pos| pos < candidate);

            if idx > 0 && sorted[idx - 1] + guard_width >= candidate {
                return false;
            }

            if idx < sorted.len() && candidate + guard_width >= sorted[idx] {
                return false;
            }

            true
        }

        fn insert_sorted(vec: &mut Vec<i64>, value: i64) {
            let idx = vec.partition_point(|&pos| pos < value);
            vec.insert(idx, value);
        }

        struct RandomResourceCtx {
            resource: Resource,
            target_positions: Vec<i64>,
            generated_positions: Vec<i64>,
            length: i64,
        }

        // Everything here runs inside a short-lived transaction.
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        // Pre-load the motif metadata to identify its channel and run.
        let (channel, run_id) = Self::_motif_meta(motif_id, &t)?;

        let mut target_positions: HashMap<ResourceId, Vec<i64>> = HashMap::new();
        let mut target_resources: HashMap<ResourceId, Resource> = HashMap::new();
        let mut desired_count = 0usize;

        {
            // Collect all slice positions for the motif so we can avoid them later.
            let mut stmt = t.prepare(
                "SELECT s.resource_id, s.slice_idx, r.import_id, r.key, r.length \
                 FROM slices s \
                 JOIN resources r ON r.id=s.resource_id \
                 WHERE s.motif_id=? \
                 ORDER BY s.resource_id, s.slice_idx",
            )?;

            stmt.query_map([motif_id], |row| {
                Ok((
                    row.get::<usize, ResourceId>(0)?,
                    row.get::<usize, i64>(1)?,
                    row.get::<usize, u32>(2)?,
                    row.get::<usize, String>(3)?,
                    row.get::<usize, u32>(4)?,
                ))
            })?
            .try_for_each(|row| -> Result<(), ThoosaError> {
                let (resource_id, slice_idx, import_id, key, length) = row?;
                target_positions
                    .entry(resource_id)
                    .or_default()
                    .push(slice_idx);
                target_resources
                    .entry(resource_id)
                    .or_insert_with(|| Resource::new(resource_id, import_id, key.clone(), length));
                desired_count += 1;
                Ok(())
            })?;
        }

        for positions in target_positions.values_mut() {
            positions.sort_unstable();
        }

        if desired_count == 0 {
            t.commit()?;
            return Ok((channel, run_id, vec![], HashMap::new()));
        }

        // Build per-resource contexts to control spacing and resource metadata.
        let mut contexts: Vec<RandomResourceCtx> = {
            let mut stmt = t.prepare(
                "SELECT DISTINCT r.id, r.import_id, r.key, r.length \
                 FROM slices s \
                 JOIN resources r ON r.id=s.resource_id \
                 WHERE s.channel_id=? AND s.run_id=?",
            )?;

            stmt.query_map(params![channel, run_id as i64], |row| {
                Ok((
                    row.get::<usize, ResourceId>(0)?,
                    row.get::<usize, u32>(1)?,
                    row.get::<usize, String>(2)?,
                    row.get::<usize, u32>(3)?,
                ))
            })?
            .map(|row| {
                let (resource_id, import_id, key, length) = row?;
                let positions = target_positions
                    .remove(&resource_id)
                    .unwrap_or_default();
                Ok(RandomResourceCtx {
                    resource: Resource::new(resource_id, import_id, key, length),
                    target_positions: positions,
                    generated_positions: Vec::new(),
                    length: length as i64,
                })
            })
            .collect::<Result<Vec<_>, ThoosaError>>()?
        };

        if !target_positions.is_empty() {
            for (resource_id, positions) in target_positions.into_iter() {
                let resource = target_resources
                    .remove(&resource_id)
                    .ok_or_else(|| {
                        ThoosaError::new(format!(
                            "no resource metadata for target motif resource {resource_id}"
                        ))
                    })?;
                contexts.push(RandomResourceCtx {
                    length: resource.length() as i64,
                    resource,
                    target_positions: positions,
                    generated_positions: Vec::new(),
                });
            }
        }

        if contexts.is_empty() {
            return Err(ThoosaError::new(format!(
                "no resources found for channel {channel}"
            )));
        }

        let width_i64 = motif_width as i64;
        let generated_spacing = max(min_distance.unwrap_or(motif_width), motif_width) as i64;

        let mut rng = rand::rng();
        let mut generated = Vec::with_capacity(desired_count);
        let mut resources_map = HashMap::new();

        let target_indices = contexts
            .iter()
            .enumerate()
            .filter(|(_, ctx)| !ctx.target_positions.is_empty())
            .map(|(idx, _)| idx)
            .collect::<Vec<usize>>();

        let max_attempts = desired_count * 500 + 1000;
        let mut attempts = 0usize;

        // Heuristically explore candidate windows until we gather enough slices.
        while generated.len() < desired_count && attempts < max_attempts {
            attempts += 1;

            let ctx_index = if !target_indices.is_empty() && rng.random_range(0..10) < 7 {
                let pick = rng.random_range(0..target_indices.len());
                target_indices[pick]
            } else {
                rng.random_range(0..contexts.len())
            };

            let ctx = &mut contexts[ctx_index];

            if ctx.length <= width_i64 + 1 {
                continue;
            }

            let start_min = 1;
            let start_max = ctx.length - width_i64 - 1;

            if start_max < start_min {
                continue;
            }

            let candidate = if !ctx.target_positions.is_empty() && rng.random_range(0..10) < 6 {
                let anchor = ctx.target_positions[rng.random_range(0..ctx.target_positions.len())];
                let band = generated_spacing.max(width_i64) * 2;
                let local_min = max(start_min, anchor - band);
                let local_max = min(start_max, anchor + band);

                if local_min <= local_max {
                    rng.random_range(local_min..=local_max)
                } else {
                    rng.random_range(start_min..=start_max)
                }
            } else {
                rng.random_range(start_min..=start_max)
            };

            if !respects_spacing(candidate, width_i64, &ctx.target_positions) {
                continue;
            }

            if !respects_spacing(candidate, generated_spacing, &ctx.generated_positions) {
                continue;
            }

            insert_sorted(&mut ctx.generated_positions, candidate);
            resources_map
                .entry(ctx.resource.id())
                .or_insert_with(|| ctx.resource.clone());
            generated.push(TensorIdSliceIdx(ctx.resource.id(), candidate as u32));
        }

        if generated.len() < desired_count {
            return Err(ThoosaError::new(format!(
                "list_random_slices: unable to build {desired_count} slices for motif {motif_id}, produced {} candidates",
                generated.len()
            )));
        }

        t.commit()?;

        Ok((channel, run_id, generated, resources_map))
    }

    fn list_run_resources(&self, run_id: RunId) -> Result<Vec<String>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let result = {
            let count: i64 = t.query_row(
                "SELECT count(*) FROM run_resources WHERE run_id=?",
                [run_id],
                |row| row.get(0),
            )?;

            debug!("list_run_resources: count={}", count);

            let mut stmt =
                t.prepare("SELECT ord_id, key FROM run_resources WHERE run_id=? ORDER BY ord_id")?;

            let result = stmt
                .query_map([run_id], |row| {
                    Ok((row.get::<usize, i64>(0)?, row.get::<usize, String>(1)?))
                })?
                .enumerate()
                .map(|(i, ord_key)| {
                    ord_key
                        .map(|(ord, key)| {
                            assert!(ord == i as i64);
                            key
                        })
                        .map_err(|e| ThoosaError::new(e.to_string()))
                })
                .collect::<Result<Vec<String>, ThoosaError>>();

            result
        };

        t.commit()?;
        debug!("list_run_resources: result.is_ok={}", result.is_ok());

        result
    }

    fn list_run_config_values(&self, run_id: RunId) -> Result<HashMap<String, Val>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let result = {
            let mut stmt = t.prepare("SELECT name, real_value, int_value, str_value, bool_value, unsigned FROM run_config_values WHERE run_id=?")?;

            let result = stmt
                .query_map([run_id], |row| {
                    Ok((
                        row.get::<usize, String>(0)?,
                        row.get::<usize, f64>(1).ok(),
                        row.get::<usize, i64>(2).ok(),
                        row.get::<usize, String>(3).ok(),
                        row.get::<usize, u8>(4)
                            .ok()
                            .map(|v| if v == 1 { true } else { false }),
                        row.get::<usize, u8>(4)
                            .ok()
                            .map(|v| if v == 1 { true } else { false })
                            .unwrap_or(false),
                    ))
                })?
                .map(|v| v.expect(format!("can't get run_config_values for {}", run_id).as_str()))
                .fold(
                    HashMap::new(),
                    |mut a, (name, real_value, int_value, str_value, bool_value, unsigned)| {
                        a.insert(
                            name.clone(),
                            real_value
                                .map(|v| v.into())
                                .or(int_value.map(|v| {
                                    if unsigned {
                                        UInt(v as u32)
                                    } else {
                                        Val::Int(v)
                                    }
                                }))
                                .or(str_value.map(|v| v.into()))
                                .or(bool_value.map(|v| v.into()))
                                .expect(format!("no value for {}", name).as_str()),
                        );
                        a
                    },
                );

            result
        };

        t.commit()?;
        debug!("list_run_config_values: run_id={run_id} result={result:?}");

        Ok(result)
    }

    fn save_selected_motifs(
        &mut self,
        run_id: RunId,
        selected_motifs: Vec<(u32, f64)>,
    ) -> Result<RunId, ThoosaError> {
        debug!(
            "save_selected_motifs: selected_motifs={:?}",
            selected_motifs
        );

        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        for selected_motifs_chunk in selected_motifs.chunks(500) {
            let (mut q, params) = selected_motifs_chunk.into_iter().fold(
                (
                    vec!["INSERT INTO selected_motifs (run_id, motif_id, rate) VALUES "],
                    vec![],
                ), //TODO check if channel matches one in motif
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

            t.execute(
                &q.into_iter().collect::<String>(),
                rusqlite::params_from_iter(params.into_iter()),
            )?;
        }

        t.commit()?;

        Ok(run_id) //TODO remove, used in tests only
    }


    fn list_output_run_ids(&mut self, input_run_id: RunId) -> Result<Vec<RunId>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let q = "SELECT run_id FROM run_configs_input WHERE input_run_id=?";

        let id_input_id = {
            let mut stmt = t.prepare(q)?;

            stmt
                .query_map([input_run_id], |row| row.get::<usize, usize>(0))?
                .map(|v| v.map(|v|v as RunId).map_err(|e|e.into()))
                .collect::<Result<Vec<_>, ThoosaError>>()
        };
        t.commit()?;

        id_input_id
    }

    /**k
     * returns vec of channel X (motif id, rate)
     */
    fn list_selected_motifs(
        &self,
        motif_selection_id: RunId,
        channels: u32,
    ) -> Result<Vec<Vec<(u32, f64)>>, ThoosaError> {
        let mut channels_motifs = (0..channels)
            .into_iter()
            .map(|_| Vec::with_capacity(10))
            .collect::<Vec<Vec<(u32, f64)>>>();

        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        {
            let mut stmt = t.prepare("SELECT DISTINCT sm.motif_id, sm.rate, m.channel_id FROM selected_motifs sm LEFT JOIN motifs AS m ON m.id=sm.motif_id WHERE sm.run_id=?")?;

            stmt.query_map([motif_selection_id], |row| {
                Ok((
                    row.get::<usize, u32>(0)?,
                    row.get::<usize, f32>(1)?,
                    row.get::<usize, usize>(2)?,
                ))
            })?
            .map(|v| v.unwrap())
            .for_each(|(motif_id, corr, channel_id)| {
                channels_motifs[channel_id].push((motif_id, corr as f64))
            });
        }
        t.commit()?;

        debug!("list_selected_motifs: channels_motifs={channels_motifs:?}");

        Ok(channels_motifs)
    }

    fn run_config_descr(&mut self, corr_run_id: RunId) -> Result<Vec<(i64, String)>, ThoosaError> {
        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;

        let descr: String = t.query_row(
            "SELECT descr FROM run_configs WHERE id=?",
            [corr_run_id],
            |row| row.get(0),
        )?;

        debug!("descr={:?}", descr);

        parse_corr_labels(descr.as_str())
    }

    fn conn(&self) -> &Mutex<Connection> {
        &self.conn
    }
}

//TODO move all prepare to init
impl SqlStore {
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Box<dyn Store>, ThoosaError> {
        Self::create_tables(Connection::open(path)?)
    }
    // #[cfg(test)]
    pub fn in_mem() -> Result<Box<dyn Store>, ThoosaError> {
        Self::create_tables(Connection::open_in_memory()?)
    }

    //TODO rename channel_id to channel - not a pk
    fn create_tables(mut conn: Connection) -> Result<Box<dyn Store>, ThoosaError> {
        {
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

            t.execute(
                "CREATE TABLE IF NOT EXISTS slices (
                id              INTEGER PRIMARY KEY,
                motif_id        NOT NULL REFERENCES motifs(id) ON DELETE CASCADE,
                run_id          NOT NULL REFERENCES run_configs(id) ON DELETE CASCADE,
                channel_id      INTEGER NOT NULL,
                resource_id     REFERENCES resources(id) ON DELETE CASCADE,
                slice_idx       INTEGER NOT NULL,
                UNIQUE(channel_id, run_id, resource_id, slice_idx)
            )",
                (),
            )?;

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
                //todo label
                "CREATE TABLE IF NOT EXISTS selected_motifs (
                motif_id   REFERENCES motifs(id),
                run_id      NOT NULL REFERENCES run_configs(id) ON DELETE CASCADE,
                rate    FLOAT NOT NULL
            )",
                (),
            )?;

            t.execute("commit", ())?;
        }

        Ok(Box::new(Self {
            //TODO https://sqlite.org/cgi/src/doc/begin-concurrent/doc/begin_concurrent.md + pool
            conn: Mutex::new(conn),
        }))
    }

    fn list_resource_set_resources(
        set_id: usize,
        t: &Transaction,
    ) -> Result<Vec<Resource>, ThoosaError> {
        let set_resources = {
            let mut stmt = t.prepare("SELECT rs.id, rs.import_id, rs.key, rs.length FROM resource_resources_set rrs LEFT JOIN resources rs ON rs.id=rrs.resource_id WHERE rrs.set_id=?")?;

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
    fn list_labels(t: &Transaction) -> Result<Vec<(usize, String)>, ThoosaError> {
        let set_resources = {
            let mut stmt = t.prepare("SELECT id, label FROM labels ORDER BY id")?;

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
    ) -> Result<(), ThoosaError> {
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

    fn save_resource_set_group(descr: String, t: &Transaction) -> Result<usize, ThoosaError> {
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
    ) -> Result<(), ThoosaError> {
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
    ) -> Result<(), ThoosaError> {
        let sql =
            "INSERT INTO slices (run_id, channel_id, motif_id, resource_id, slice_idx) VALUES";

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
                debug!(
                    "insert_slices: e={e} sqlite_error={:?} run_id={run_id}, channel={channel}, motif_id={motif_id}",
                    e.sqlite_error()
                );
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
    ) -> Result<(), ThoosaError> {
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

        debug!(
            "save_slices_groups_as_motifs_from_tmp_db: slice_matches_updated={slice_matches_updated}"
        );

        Ok(())
    }

    fn get_input_run_ids_(
        t: &Transaction,
        run_id: RunId,
        input_type: Option<&dyn Display>,
    ) -> Result<Vec<RunId>, ThoosaError> {
        let mut stmt = t.prepare("SELECT rci.input_run_id, rc.type FROM run_configs_input rci JOIN run_configs rc ON rc.id=rci.input_run_id WHERE run_id=?")?;

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
                        Self::get_input_run_ids_(t, run_id, input_type)
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

    fn _motif_meta(motif_id: u32, t: &Transaction) -> Result<(i64, RunId), ThoosaError> {
        // Pre-load the motif metadata to identify its channel and run.
        let (channel, run_id): (i64, RunId) = t.query_row(
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
    ) -> Result<Box<dyn MotifTrackerStore>, ThoosaError> {
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

pub trait MotifTrackerStore: Debug + Send + Sync {
    fn save_slices(&mut self, slices: Vec<SlicesMatch>) -> Result<(), ThoosaError>;
    fn load_slices_into_groups(&mut self, store: &Arc<RwLock<Box<dyn Store>>>, max_slices_counts: Option<u32>) -> Result<(), ThoosaError>;

    fn run_id(&self) -> RunId;
    fn channel(&self) -> u32;
    fn path(&self) -> &PathBuf;
}

pub fn filter_groups_by_slices_count(max_slices_counts: Option<u32>, groups: HashMap<u16, HashSet<IdSlice>>) -> HashMap<u16, HashSet<IdSlice>> {
    match max_slices_counts {
        None => groups,
        Some(max_slices_counts) => groups.into_iter().fold(HashMap::new(), |mut a, v| {
            if v.1.len() < max_slices_counts as usize {
                a.insert(v.0, v.1);
            } else {
                warn!("load_slices_into_groups: skipping group with {} slices",v.1.len());
            }
            a
        }),
    }
}

impl MotifTrackerStore for SqlMotifTrackerStore {
    fn save_slices(&mut self, slices: Vec<SlicesMatch>) -> Result<(), ThoosaError> {
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

    fn load_slices_into_groups(
        &mut self,
        store: &Arc<RwLock<Box<dyn Store>>>,
        max_slices_counts: Option<u32>,
    ) -> Result<(), ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        debug!(
            "set_motifs: run_id={:?}, channel={:?}",
            self.run_id, self.channel
        );

        let groups = {
            let mut stmt = t.prepare(
                "SELECT id, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1 FROM slice_matches",
            )?;

            let groups = group_slices(
                stmt.query_map([], |row| {
                    //TODO move group_slices out, not for store
                    Ok((
                        IdSlice(row.get(0)?, row.get(1)?, row.get(2)?),
                        IdSlice(row.get(0)?, row.get(3)?, row.get(4)?),
                    ))
                })?
                .map(|v| v.unwrap()),
            );

            debug!(
                "set_motifs: run_id={:?}, channel={:?} groups={:?}",
                self.run_id, self.channel, groups
            );

            groups
        };
        t.commit()?;

        store.write().save_groups_as_motifs_with_slices_matches(
            self.run_id,
            self.channel as i64,
            filter_groups_by_slices_count(max_slices_counts, groups),
            &self.path,
        )
    }

    fn run_id(&self) -> RunId {
        self.run_id
    }

    fn channel(&self) -> u32 {
        self.channel
    }

    fn path(&self) -> &PathBuf {
        &self.path
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
    use thoosa_shared::{TensorIdSliceIdx, cdt, cdt_for_dir};

    use crate::{
        MotifTrackerStore, SetCond, SlicesMatch, SqlMotifTrackerStore, SqlStore, Store, ThoosaError,
    };

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_group_resources() -> Result<(), ThoosaError> {
        // fs::remove_file("/tmp/test_db.db3").unwrap_or(());
        // let mut store = SqlStore::from_path("/tmp/test_db.db3")?;

        let mut store = SqlStore::in_mem()?;

        let mut mutex_guard = store.conn().lock();
        let t = mutex_guard.transaction()?;

        let import_id = store.save_import("test".to_string(), &t)?;

        let _aaa_id = store.save_label("aaa", &t);
        let _bbb_id = store.save_label("bbb", &t);
        let _ccc_id = store.save_label("ccc", &t);

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
            &t,
        )?;

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
            &t,
        )?;

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
            &t,
        )?;

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
            &t,
        )?;

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
            &t,
        )?;

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
            &t,
        )?;

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
            &t,
        )?;

        let t1 = Instant::now();
        store.save_groups_by_label_cnt(10, &t)?;

        t.commit()?;
        //TODO asserts

        debug!("done in {}", t1.elapsed().as_millis());

        // fs::remove_file(db_file)?;
        Ok(())
    }

    #[test]
    fn test_save_slices() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        fs::create_dir_all(&test_dir)?;
        let mut store = Arc::new(RwLock::new(SqlStore::from_path(
            test_dir.join("test_db.db3"),
        )?));

        let resource_ids = init_test_store(&mut store)?;

        let all_group_id = store
            .write()
            .save_resource_set_group("group 1 labels/100/100-200".into())?;
        let _set_all_id = store.write().save_resources_set(
            "All".to_string(),
            vec!["1111".into(), "2222".into(), "3333".into()],
            Some(all_group_id),
        )?;

        let group_sets = store.read().list_resource_set_group_sets("pure group 10")?;
        assert_eq!(3, group_sets.len());

        let mut keys_labels_p = store
            .read()
            .list_resource_random_in_group("group 1 labels/100/100-200")?;

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

        let find_motifs_run_id = store.write().save_run_config(
            None,
            &"todo",
            "find_motifs",
            "",
            HashMap::from([
                ("prop1".into(), Val::Int(1)),
                ("prop2".into(), Val::Real(2.0)),
                ("prop3".into(), Val::Str("333".to_string())),
            ]),
        )?;

        let slices_store_path = test_dir.join("test_slices_db.db3");
        let mut slices_store = SqlMotifTrackerStore::new(&slices_store_path, find_motifs_run_id, 7)
            .expect("can't create tmp sotre");

        slices_store.save_slices(slices.clone())?;

        let groups = slices_store.load_slices_into_groups(&store, None)?;

        {
            let store_guard = store.write();
            let mut guard = store_guard.conn().lock();
            let t2 = guard.transaction()?;
            let mut stmt = t2.prepare("SELECT sm.motif_id, s_0.resource_id, s_0.slice_idx, s_1.resource_id, s_1.slice_idx FROM slice_matches sm \
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

        let motif_slices_map = store.read().list_motifs_slices(find_motifs_run_id)?;
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

        let select_motifs_run_id = store.write().save_run_config(
            Some(&[find_motifs_run_id]),
            &"todo",
            "find_motifs",
            "",
            HashMap::from([
                ("prop1".into(), Val::Int(1)),
                ("prop2".into(), Val::Real(2.0)),
                ("prop3".into(), Val::Str("333".to_string())),
            ]),
        )?;

        let select_motifs_run_id = store.write().save_selected_motifs(
            find_motifs_run_id,
            vec![
                (motif_slices_444[0].motif_id, 0.1),  //motif_id corr
                (motif_slices_1111[0].motif_id, 0.2), //motif_id corr
            ],
        )?;

        let input_run_ids = store.read().get_input_run_ids(select_motifs_run_id, None)?;

        assert_eq!(
            vec![find_motifs_run_id],
            input_run_ids
        );

        let motifs = store
            .read()
            .list_selected_motifs(select_motifs_run_id, 20)?;

        debug!("motifs: {:?}", motifs);

        //7th channel
        assert_eq!(motif_slices_444[0].motif_id, motifs[7][0].0);
        assert_eq!(motif_slices_1111[0].motif_id, motifs[7][1].0);
        //
        // //TODO
        let (channel, run_id_, mut slices, resources_map) =
            store
                .read()
                .list_motif_slices(motif_slices_1111[0].motif_id, None, None)?;
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

        let (channel, run_id_, mut slices, resources_map) =
            store
                .read()
                .list_motif_slices(motif_slices_1111[0].motif_id, Some(1), None)?;
        slices.sort();


        let resources = store.read().list_resources_not_in_group("pure group 10");
        debug!("resources={resources:?}");

        let (channel, run_id_, mut slices, resources_map) =
            store
                .read()
                .list_motif_slices(motif_slices_1111[0].motif_id, Some(3), Some((0, 0)))?;

        debug!("random slices={slices:?}");
        //
        // // let (_channel, _run_id_, mut slices) = store.list_motif_slices(1, Some(10), Some((run_id, channel as usize)))?;
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

    //TODO merge with init_test_store in cli::main_tests
    pub fn init_test_store(
        store: &mut Arc<RwLock<Box<dyn Store>>>,
    ) -> Result<Vec<ResourceId>, ThoosaError> {
        let store_guard = store.write();
        let mut mutex_guard = store_guard.conn().lock();
        let t = mutex_guard.transaction()?;

        let import_id = store_guard.save_import("test".to_string(), &t)?;

        let _aaa_id = store_guard.save_label("aaa", &t)?;
        let _bbb_id = store_guard.save_label("bbb", &t)?;
        let _ccc_id = store_guard.save_label("ccc", &t)?;

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
            &t,
        )?;

        debug!("resource_ids={resource_ids:?}");

        store_guard.save_pure_sets(10, false, &t)?;
        t.commit()?;
        Ok(resource_ids)
    }

    #[test]
    fn test_save_run_config() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        fs::create_dir_all(&test_dir)?;
        let mut store = Arc::new(RwLock::new(SqlStore::from_path(
            test_dir.join("test_db.db3"),
        )?));

        let run_id = store.write().save_run_config(
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
        )?;

        let next_run_id = store.write().save_run_config(
            Some(&[run_id]),
            &"todo",
            "test next",
            "",
            HashMap::from([
                ("next_prop1".into(), Val::Int(1)),
                ("next_prop2".into(), Val::Real(2.0)),
                ("next_prop3".into(), Val::Str("zzz".to_string())),
            ]),
        )?;

        let loaded_params = store.read().list_run_config_values(run_id)?;

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

        let netx_loaded_params = store.read().list_run_config_values(next_run_id)?;

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

    #[test]
    fn test_conv_cond() {
        let input = vec![
            ("aaa".into(), SetCond::Lt(0.4)),
            ("bbb".into(), SetCond::Gt(0.1)),
        ];

        let conds = super::conds_to_q(&input);
        assert_eq!("(l.label='aaa' and p<?) or (l.label='bbb' and p>?)", conds);
    }

    #[test]
    fn test_get_prev_run_id() -> Result<(), ThoosaError> {
        let mut store = SqlStore::in_mem()?;

        let input_run_id = store.save_run_config(None, &"aaa", "test 0", "", HashMap::new())?;
        let run_id =
            store.save_run_config(Some(&[input_run_id]), &"aaa", "test 1", "", HashMap::new())?;

        assert_eq!(store.get_input_run_ids(run_id, None)?[0], input_run_id);

        let input_run_id_0 = store.save_run_config(None, &"aaa", "test 0", "", HashMap::new())?;
        let input_run_id_1 = store.save_run_config(
            Some(&[input_run_id_0]),
            &"bbb",
            "test 1",
            "",
            HashMap::new(),
        )?;
        let run_id = store.save_run_config(
            Some(&[input_run_id_1]),
            &"ccc",
            "test 1",
            "",
            HashMap::new(),
        )?;

        assert_eq!(
            store.get_input_run_ids(run_id, Some(&"aaa"))?[0],
            input_run_id_0
        );

        Ok(())
    }

    #[test]
    fn test_list_resources_by_file_stem() -> Result<(), ThoosaError> {
        let mut store = SqlStore::in_mem()?;
        {
            let mut mutex_guard = store.conn().lock();
            let t = mutex_guard.transaction()?;

            let import_id = store.save_import("test".to_string(), &t)?;

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
                &t,
            )?;

            t.commit()?;
        }

        assert_eq!(
            vec!["resource_aaa_0".to_string(), "resource_aaa_1".to_string()],
            store
                .list_resources_by_file_stems(&vec!["%aaa%"])?
                .into_iter()
                .map(|r| r.file_stem().to_string())
                .collect::<Vec<String>>()
        );

        assert_eq!(
            vec!["resource_bbb_0".to_string(), "resource_bbb_1".to_string()],
            store
                .list_resources_by_file_stems(&vec!["%bbb%"])?
                .into_iter()
                .map(|r| r.file_stem().to_string())
                .collect::<Vec<String>>()
        );

        Ok(())
    }

    #[test]
    fn test_list_random_slices_respects_constraints() -> Result<(), ThoosaError> {
        let mut store = SqlStore::in_mem()?;

        create_resources(&store, &[("res_a", 2048), ("res_b", 2048)])?;

        let run_id = store.save_run_config(None, &"find_motifs", "random test", "", HashMap::new())?;
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
            store.list_random_slices(motif_id, WIDTH, None)?;

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

    #[test]
    fn test_list_random_slices_respects_min_distance() -> Result<(), ThoosaError> {
        let mut store = SqlStore::in_mem()?;

        create_resources(&store, &[("res_main", 4096)])?;

        let run_id =
            store.save_run_config(None, &"find_motifs", "random spacing", "", HashMap::new())?;
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
            store.list_random_slices(motif_id, WIDTH, Some(MIN_DISTANCE))?;

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

    fn create_resources(store: &Box<dyn Store>, specs: &[(&str, u32)]) -> Result<(), ThoosaError> {
        let mut guard = store.conn().lock();
        let t = guard.transaction()?;

        let import_id = store.save_import("test resources".into(), &t)?;

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

        store.save_resources_labels(&resources, import_id, &t)?;
        t.commit()?;

        Ok(())
    }

    fn fetch_resource_id(store: &Box<dyn Store>, key: &str) -> Result<ResourceId, ThoosaError> {
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
    ) -> Result<u32, ThoosaError> {
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

    pub fn with_get_motif_meta<R:Into<HashMap<u32, Result<(usize, RunId), ThoosaError>>>>(mut self, results: R)-> Self {
        let map = results.into();
        self.store.expect_get_motif_meta().returning(
            move |motif_id|{
                map.get(&motif_id).cloned().expect(&format!("mock is not configured to return motif meta for motif_id {motif_id}"))
            }
        );
        self
    }

    pub fn with_get_input_run_ids<R:Into<HashMap<RunId, Result<Vec<RunId>, ThoosaError>>>>(mut self, results: R)-> Self{
        let map = results.into();
        self.store.expect_get_input_run_ids().returning(
            move |run_id, input_type|{
                map.get(&run_id).cloned().expect(&format!("mock is not configured to return input_run_ids for run_id {run_id}"))
            }
        );
        self
    }

    pub fn with_list_motifs<R:Into<HashMap<RunId, Result<Vec<usize>, ThoosaError>>>>(mut self, results: R) -> Self {
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
        limit: Option<u32>,
        all_but_in_run_id: Option<(RunId, u32)>, r:Result<(i64, RunId, Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>), ThoosaError>
    ) -> Self {

        self.store.expect_list_motif_slices().returning(
            move |p_motif_id, limit, all_but_in_run_id| {
                debug!("with_list_motif_slices: p_motif_id={p_motif_id} motif_id={motif_id} ");
                if p_motif_id == motif_id {
                    r.clone()
                } else {
                    Err(ThoosaError::new("Not found"))
                }

            }
        );

        self
    }

    pub fn with_list_random_slices(
        mut self,
        motif_id: u32,
        motif_width: u32,
        min_distance: Option<u32>,
        r: Result<(i64, RunId, Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>), ThoosaError>
    ) -> Self {
        self.store.expect_list_random_slices().returning(
            move |p_motif_id, p_width, p_min_distance| {
                if p_motif_id == motif_id && p_width == motif_width && p_min_distance == min_distance {
                    r.clone()
                } else {
                    Err(ThoosaError::new("Not found"))
                }
            }
        );

        self
    }

    pub fn with_list_selected_motifs<R:Into<HashMap<RunId, Result<Vec<Vec<(u32, f64)>>, ThoosaError>>>>(
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

    pub fn with_list_run_config_values(mut self, run_id:RunId, r:Result<HashMap<String, Val>, ThoosaError>) -> Self {
        self.store.expect_list_run_config_values()
            .returning(move |p_run_id| {
                if p_run_id == run_id {
                    r.clone()
                } else {
                    Err(ThoosaError::new("Not found"))
                }
            });

        self
    }

    pub fn with_list_resources_by_file_stems(
        mut self,
        expected_stems: Vec<String>,
        result: Result<Vec<Resource>, ThoosaError>,
    ) -> Self {
        self.store
            .expect_list_resources_by_file_stems()
            .returning(move |requested| {
                let stems = requested.iter().map(|stem| stem.to_string()).collect::<Vec<_>>();
                if stems == expected_stems {
                    result.clone()
                } else {
                    Err(ThoosaError::new("Not found"))
                }
            });

        self
    }

    pub fn with_list_group_labeled_resources_sets<D:IntoIterator<Item=(&'static str, Result<Vec<(String, String)>, ThoosaError>)>>(mut self, data:D) -> Self {
        let map :HashMap<&'static str, Result<Vec<(String, String)>, ThoosaError>> = HashMap::from_iter(data.into_iter());

    self.store.expect_list_group_labeled_resources_sets()
        .returning(move |group| {
            map.get(group).expect(&format!("no mock data for {group} configured in list_group_labeled_resources_sets")).clone()
        });

        self
    }

    pub fn with_list_resources_not_in_group<D:IntoIterator<Item=(&'static str, Result<Vec<Resource>, ThoosaError>)>>(mut self, data:D) -> Self {
        let map :HashMap<&'static str, Result<Vec<Resource>, ThoosaError>> = HashMap::from_iter(data.into_iter());

        self.store.expect_list_resources_not_in_group()
            .returning(move |group| {
                map.get(group).expect(&format!("no mock data for {group} configured in list_resource_set_resources_by_descr")).clone()
            });


        self
    }

    pub fn with_list_resource_set_resources_by_descr<D:IntoIterator<Item=(&'static str, Result<Vec<Resource>, ThoosaError>)>>(mut self, data:D) -> Self {
        let map :HashMap<&'static str, Result<Vec<Resource>, ThoosaError>> = HashMap::from_iter(data.into_iter());

        self.store.expect_list_resource_set_resources_by_descr()
        .returning(move |group| {
            map.get(group).expect(&format!("no mock data for {group} configured in list_resource_set_resources_by_descr")).clone()
        });


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
