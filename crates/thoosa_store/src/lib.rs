#![feature(iter_array_chunks)]

pub mod fs_store;

use std::array::IntoIter;
use std::cmp::{max, min};
use std::collections::{HashMap, HashSet};
use std::{fmt, io, mem};
use std::fmt::{Debug, Display};
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc};
use parking_lot::{Mutex, RwLock};

use log::{debug, trace};
use rand::prelude::SliceRandom;
use rand::{thread_rng, Rng};
use rusqlite::{Connection, Error, params, Transaction};
use rusqlite::types::Value;

use thoosa_shared::*;
use thoosa_shared::resource::{Resource, ResourceId, ResourceStore};
use thoosa_shared::val::Val;



#[derive(Debug, Clone)]
pub struct IdSlice(i64, ResourceId, i64); //id, tensor id, slice idx

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
            SetCond::Eq(_v) => {
                "p="
            }
            SetCond::Gt(_v) => {
                "p>"
            }
            SetCond::Lt(_v) => {
                "p<"
            }
        }
    }

    fn v(&self) -> f32 {
        match self {
            SetCond::Eq(v) => {
                *v
            }
            SetCond::Gt(v) => {
                *v
            }
            SetCond::Lt(v) => {
                *v
            }
        }
    }
}


pub trait Store:ResourceStore + Sync + Send {
    fn save_import(&self, descr: String, t: &Transaction) -> Result<u32, ThoosaError>;
    fn save_tree_run(&mut self, input_run_id: Option<&[usize]>, descr: &str, keys: &str, properties_vals: HashMap<String, Val>) -> Result<usize, ThoosaError>;
    fn get_input_run_ids(&mut self, run_id: usize) -> Result<Vec<usize>, ThoosaError>;
    fn save_resources(&mut self, resource_keys: Vec<String>) -> Result<(), ThoosaError>;
    fn save_resources_set(&mut self, set: String, resource_keys: Vec<String>, group_id: Option<usize>) -> Result<usize, ThoosaError>;
    fn save_resource_set_group(&mut self, descr: String) -> Result<usize, ThoosaError>;
    fn save_pure_sets(&self, max_len: usize, only_unused: bool, t: &Transaction) -> Result<(), ThoosaError>;
    fn save_set_by_labels(&mut self, conds: Vec<(String, SetCond)>, only_unused: bool, limit: usize) -> Result<(), ThoosaError>;
    fn save_groups_by_label_cnt(&self, max_resources_in_set: usize, t: &Transaction) -> Result<(), ThoosaError>;
    fn list_resources_by_ids(&mut self, file_stem: &Vec<ResourceId>) -> Result<Vec<Resource>, ThoosaError>;
    fn list_resource_random_not_label(&mut self, label: &str, limit: i64) -> Result<Vec<Resource>, ThoosaError>;
    fn list_resource_random_in_group(&mut self, group_descr: &str) -> Result<Vec<(Resource, Vec<f32>)>, ThoosaError>;
    fn save_label(&self, label: &str, t: &Transaction) -> Result<u32, ThoosaError>;
    fn list_labels(&mut self) -> Result<Vec<String>, ThoosaError>;
    fn save_params_value(&mut self, params: HashMap<String, Val>, run_id: usize, label: Option<&str>) -> Result<(), ThoosaError>;
    fn save_resources_labels(&self, resource_labels_probs: &HashMap<String, Vec<(String, f32)>>, import_id: u32, t: &Transaction) -> Result<Vec<ResourceId>, ThoosaError>;
    fn save_slices(&mut self, run_id: usize, channel: i64, slices: Vec<(TensorSliceIdx <ResourceId>, TensorSliceIdx <ResourceId>, f32)>) -> Result<(), ThoosaError>;
    fn set_motifs_from_tmp_db(&mut self, run_id: usize, channel: i64, groups:HashMap<u16, Vec<IdSlice>>, as_ref: &dyn AsRef<Path>) -> Result<(), ThoosaError>;
    fn list_motifs_distances(&mut self, run_id: usize) -> Result<HashMap<usize, f32>, ThoosaError>;
    fn list_motifs_slices(&mut self, run_id: usize) -> Result<HashMap<Resource, Vec<MotifSlice>>, ThoosaError>;
    fn list_run_resources(&mut self, run_id: usize) -> Result<Vec<String>, ThoosaError>;
    fn list_run_params(&mut self, run_id: usize) -> Result<HashMap<String, Val>, ThoosaError>;
    fn save_selected_motifs(&mut self, run_id: usize, descr: &str, selected_motifs: Vec<Vec<(usize, f64)>>) -> Result<usize, ThoosaError>;
    fn list_input_run_ids(&mut self, ids: &Vec<usize>) -> Result<Vec<(usize, usize)>, ThoosaError>; //corr_run_id, run_id, label
    fn list_output_run_ids(&mut self, input_run_id: usize) -> Result<Vec<usize>, ThoosaError>; //corr_run_id, run_id, label
    fn list_selected_motifs(&mut self, motif_selection_id: usize) -> Result<Vec<Vec<(usize, f64)>>, ThoosaError>;
    fn list_motif_slices(&mut self, motif_id: usize, limit: Option<usize>, all_but_in_run_id: Option<(usize, usize)>) -> Result<(i64, usize, Vec<TensorSliceIdx<Resource>>), ThoosaError>;
    fn tree_run_descr(&mut self, corr_run_id: usize) -> Result<Vec<(i64, String)>, ThoosaError>;
    fn delete_slices_without_matches(&mut self, run_id: usize) -> Result<(), ThoosaError>;

    fn conn(&self) -> &Mutex<Connection>;
}

impl Debug for SqlMotifTrackerStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("")
            .finish()
    }
}

pub struct SqlStore {
    conn: Mutex<Connection>,
}

impl ResourceStore for SqlStore {
    fn list_resource_set_resources_by_descr(&mut self, descr: &str) -> Result<Vec<Resource>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let r = Self::list_resource_set_resources(t.query_row("SELECT id FROM resource_set WHERE descr=?", [descr], |row| row.get(0))?, &t)?;
        debug!("list_resource_set_resources_by_descr: descr={descr} r={r:?}");

        t.commit()?;

        Ok(r)
    }
    fn list_resources_by_file_stems(&mut self, file_stems: &Vec<&str>) -> Result<Vec<Resource>, ThoosaError> {
        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;

        let (mut q, params) = file_stems.into_iter().fold((vec!["SELECT id, import_id, key FROM resources WHERE "], vec![]), |mut a, v| {
            a.0.push(" key LIKE ?");
            a.0.push(" OR");
            a.1.push(*v);
            a
        });

        q.pop();
        q.push(" ORDER BY key");

        let resources = {
            let mut stmt = t.prepare(q.into_iter().collect::<String>().as_str())?;

            debug!("list_resources_by_file_stems: stmt={stmt:?}");

            let r = stmt.query_map(rusqlite::params_from_iter(params.into_iter()), |row| {
                Ok(Resource::new(
                    row.get::<usize, u32>(0)?,
                    row.get::<usize, u32>(1)?,
                    row.get::<usize, String>(2)?,
                ))
            })?.map(|v| v.unwrap()).collect();

            r
        };

        Ok(resources)
    }

    fn list_resource_set_group_sets(&mut self, group_descr: &str) -> Result<Vec<(usize, String)>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let group_sets = {
            let mut stmt = t.prepare("SELECT rs.id, rs.descr FROM resource_set rs LEFT JOIN resource_set_group rsg ON rsg.id=rs.group_id WHERE rsg.descr=?")?;


            let vec = stmt.query_map([group_descr], |row| { //TODO move group_slices out, not for store
                Ok((
                    row.get::<usize, usize>(0).expect("row 0 in list_resource_set_group_sets"),
                    row.get::<usize, String>(1).expect("row 1 in list_resource_set_group_sets")
                ))
            })?.map(|v| v.unwrap()).collect();

            vec
        };

        t.commit()?;

        Ok(group_sets)
    }

    fn list_group_labeled_resources_sets(&mut self, group_descr: &str) -> Result<Vec<(String, String)>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let labeled_resources_sets = {
            let mut stmt = t.prepare("SELECT l.label, rs.descr FROM resource_set_group rsg JOIN resource_set rs ON rs.group_id=rsg.id JOIN labels l ON l.id=rs.label_id WHERE rsg.descr=?")?;

            let vec = stmt.query_map([group_descr], |row| { //TODO move group_slices out, not for store
                Ok((
                    row.get::<usize, String>(0).expect("row 0 in list_group_labeled_resources_sets"),
                    row.get::<usize, String>(1).expect("row 1 in list_group_labeled_resources_sets")
                ))
            })?.map(|v| v.unwrap()).collect();

            vec
        };

        t.commit()?;

        debug!("list_group_labeled_resources_sets: labeled_resources_sets={labeled_resources_sets:?}");

        Ok(labeled_resources_sets)
    }

    fn list_resource_set_resources(&mut self, set_id: usize) -> Result<Vec<Resource>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let set_resources = Self::list_resource_set_resources(set_id, &t)?;

        t.commit()?;

        Ok(set_resources)
    }

}

impl Store for SqlStore {
    fn save_import(&self, descr: String, t: &Transaction) -> Result<u32, ThoosaError> {
        t.execute("INSERT INTO imports (descr) VALUES (?)", params![descr])?;

        let import_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;

        Ok(import_id)
    }

    //TODO resources - save serialized resources config - run_resources is getting to big
    fn save_tree_run(&mut self, input_run_ids_op: Option<&[usize]>, descr: &str, resources: &str, properties_vals: HashMap<String, Val>) -> Result<usize, ThoosaError> {
        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction().map_err(|e| ThoosaError::new(e.to_string()))?; //TODO error

        t.execute("INSERT INTO tree_run (descr) VALUES (?)", params![ descr])?;

        let run_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;

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

        if let Some(input_run_ids) = input_run_ids_op {
            let (mut q, params) = input_run_ids.into_iter().fold((vec!["INSERT INTO tree_run_input (run_id, input_run_id) VALUES"], vec![]), |(mut q, mut params): (Vec<&str>, Vec<Value>), v| {
                q.push("(?, ?)");
                q.push(",");

                params.push(Value::Integer(run_id)); //resource id
                params.push(Value::Integer(*v as i64)); //label id
                //
                (q, params)
            });

            if params.len() > 0 {
                q.pop();

                let q_str = q.into_iter().collect::<String>();

                t.execute(&q_str, rusqlite::params_from_iter(params.into_iter())).unwrap();
            }
        }

        if properties_vals.len() > 0 {
            for property_val in properties_vals.into_iter() {
                Self::save_param_value(&t, run_id as usize, &property_val.0, property_val.1, None)?;
            }
        }

        t.commit()?;

        Ok(run_id as usize)
    }

    fn get_input_run_ids(&mut self, run_id: usize) -> Result<Vec<usize>, ThoosaError> {
        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;
        //
        //
        // t.query_row("SELECT input_run_id FROM tree_run WHERE id=?", [run_id], |row| row.get(0)).map_err(|e|ThoosaError::new(e.to_string()))

        let input_run_ids = {
            let mut stmt = t.prepare("SELECT input_run_id FROM tree_run_input WHERE run_id=?")?;

            let vec = stmt.query_map([run_id], |row| {
                row.get::<usize, usize>(0)
            })?.collect::<Result<Vec<usize>, Error>>()?;

            vec
        };
        Ok(input_run_ids)
    }

    fn save_params_value(&mut self, params: HashMap<String, Val>, run_id: usize, label: Option<&str>) -> Result<(), ThoosaError> {
        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;


        let label_id = t.query_row("SELECT id FROM labels WHERE label=?", [label], |row| row.get(0)).ok();

        for property_val in params.into_iter() {
            Self::save_param_value(&t, run_id, &property_val.0, property_val.1, label_id)?;
        }

        t.commit()?;

        Ok(())
    }

    fn save_resources(&mut self, resources: Vec<String>) -> Result<(), ThoosaError> {
        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;

        for resources_chunk in resources.chunks(500) {
            let (mut q, params) = resources_chunk.into_iter().fold(
                (vec!["INSERT INTO resources (key) VALUES"], vec![]), |mut a: (Vec<&str>, Vec<Value>), v| {
                    a.0.push("(?)");
                    a.0.push(", ");

                    a.1.push(Value::Text(v.clone()));
                    a
                });

            q.pop();

            t.execute(&q.into_iter().collect::<String>(), rusqlite::params_from_iter(params.into_iter()))?;
        }
        t.commit()?;

        Ok(())
    }


    fn save_label(&self, label: &str, t: &Transaction) -> Result<u32, ThoosaError> {
        t.execute("INSERT INTO labels (label) VALUES (?)", params![label])?;

        let label_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;

        Ok(label_id)
    }

    fn list_labels(&mut self) -> Result<Vec<String>, ThoosaError> {
        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;

        let labels = Self::list_labels(&t)?.into_iter().map(|id_label| id_label.1).collect();

        t.commit()?;

        Ok(labels)
    }

    //TODO folds to util fn
    fn save_resources_labels(&self, resource_labels_probs: &HashMap<String, Vec<(String, f32)>>, import_id: u32, t: &Transaction) -> Result<Vec<ResourceId>, ThoosaError> {
        debug!("save_resources_labels: resource_labels_probs.len()={}", resource_labels_probs.len());

        let resources_len = resource_labels_probs.len();

        let mut resource_labels_iter = resource_labels_probs.into_iter().array_chunks::<500>();


        let mut resource_ids = Vec::with_capacity(resources_len);

        while let Some(resource_keys) = resource_labels_iter.next() {
            resource_ids.extend(
                do_insert(resource_keys.into_iter(), import_id, &t)?
            );
        }

        if let Some(reminder) = resource_labels_iter.into_remainder() {
            if reminder.len() > 0 {
                resource_ids.extend(
                    do_insert(reminder, import_id, &t)?
                );
            }
        }

        fn do_insert(resource_keys_iter: IntoIter<(&String, &Vec<(String, f32)>), 500>, import_id: u32, t: &Transaction) -> Result<Vec<ResourceId>, ThoosaError> {
            let (mut insert_resources_q, insert_resources_params, insert_labeled_resources_params) = resource_keys_iter
                .fold(
                    (
                        vec!["INSERT INTO resources (key, import_id) VALUES"],
                        vec![],
                        vec![]
                    ), |(mut insert_resources_q, mut insert_resources_params, mut insert_labeled_resources_params),
                        (resource_key, labels)| {
                        insert_resources_params.push(Value::Text(resource_key.clone()));
                        insert_resources_params.push(Value::Integer(import_id as i64));
                        insert_resources_q.push("(?, ?)");
                        insert_resources_q.push(",");


                        let labels_ids = labels.into_iter().map(|(label, p)|
                            (
                                t.query_row("SELECT id FROM labels WHERE label=?", [&label], |row| row.get(0)).expect(format!("label {} not found", label).as_str()),
                                p
                            )
                        ).collect::<Vec<(u32, &f32)>>();
                        insert_labeled_resources_params.push(labels_ids);


                        (insert_resources_q, insert_resources_params, insert_labeled_resources_params)
                    });

            insert_resources_q.pop();
            insert_resources_q.push(" RETURNING id");

            let q = insert_resources_q.into_iter().collect::<String>();
            let mut stmt = t.prepare(&q)?;

            let mut rows = stmt.query(rusqlite::params_from_iter(insert_resources_params.into_iter()))?;

            let mut resource_ids: Vec<u32> = vec![];
            while let Some(row) = rows.next()? { //TODO  query_map ? not needed?
                resource_ids.push(row.get_unwrap(0));
            }

            let mut labels_iter = resource_ids
                .iter()
                .zip(insert_labeled_resources_params)
                .flat_map(move |(resource_id, labels)| labels.into_iter().map(move |(l, p)| (resource_id, l, p)))
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
            let (mut q, params) = resource_labels.fold((vec!["INSERT INTO labeled_resources (resource_id, label_id, p) VALUES"], vec![]), |(mut q, mut params): (Vec<&str>, Vec<Value>), v| {
                q.push("(?, ?, ?)");
                q.push(",");

                params.push(Value::Integer(*v.0 as i64)); //resource id
                params.push(Value::Integer(v.1 as i64)); //label id
                params.push(Value::Real(*v.2 as f64)); //p
                //
                (q, params)
            });

            if params.len() > 0 {
                q.pop();

                let q_str = q.into_iter().collect::<String>();

                t.execute(&q_str, rusqlite::params_from_iter(params.into_iter())).unwrap();
            }
        }

        Ok(resource_ids)
    }

    fn save_resources_set(&mut self, descr: String, resource_keys: Vec<String>, group_id: Option<usize>) -> Result<usize, ThoosaError> {
        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;

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
        let resoure_id_str = resource_set_id.to_string();
        let params = vec![resoure_id_str.as_str()];
        for resources_chunk in resource_keys.chunks(500) {
            let (mut q, params) = resources_chunk.into_iter().fold((q.clone(), params.clone()), |(mut q, mut params), key| {
                params.push(key.as_str());
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
    }

    fn save_resource_set_group(&mut self, descr: String) -> Result<usize, ThoosaError> {
        let mut conn_guard = self.conn.lock();
        let mut t = conn_guard.transaction()?;

        let group_id = Self::save_resource_set_group(descr, &mut t)?;

        t.commit()?;

        Ok(group_id)
    }

    fn save_groups_by_label_cnt(&self, max_resources_in_set: usize, t: &Transaction) -> Result<(), ThoosaError> {
        {
            let labels_cnt_labels_resources: HashMap<usize, HashMap<String, Vec<usize>>> = {
                let mut stmt = t.prepare(
                    "SELECT r.id, GROUP_CONCAT(l.label, ','), count(lr.resource_id) labels_cnt FROM resources r JOIN labeled_resources lr ON lr.resource_id=r.id JOIN labels l ON l.id=lr.label_id GROUP BY resource_id ORDER BY labels_cnt"
                )?;
                let mut rows = stmt.query(())?;
                let mut labels_cnt_labels_resources: HashMap<usize, HashMap<String, Vec<usize>>> = HashMap::new(); //labels_cnt -> Labels->Resources. labels_cnt==1 - pure resources

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

            debug!("save_groups_by_label_cnt: labels_cnt_labels_resources.len()={:?}", labels_cnt_labels_resources.len());
            let mut rng = thread_rng();

            let labels_ids: HashMap<String, usize> = Self::list_labels(&t)?.into_iter().fold(HashMap::new(), |mut a, v| {
                a.insert(v.1, v.0);
                a
            });

            //labels_cnt number of labels a resource has - the fewer labels the more chances that resource is classified correctly
            for (labels_cnt, labels_resources) in labels_cnt_labels_resources {
                let max_labels_resources = labels_resources.values().map(|r| r.len()).sum::<usize>();
                debug!(
                    "save_groups_by_label_cnt: labels_cnt={} max_labels_resources={}, max_resources_in_set={}, max_labels_resources/max_resources_in_set={}",
                    labels_cnt, max_labels_resources, max_resources_in_set, max_labels_resources/max_resources_in_set
                );

                let num_groups = max(1, max_labels_resources / max_resources_in_set).ilog2();
                let groups_max_resources = (0..num_groups).map(|i| max_resources_in_set * 2usize.pow(i as u32) as usize).collect::<Vec<usize>>();

                let groups_max_ids = groups_max_resources.iter().map(
                    |max_r| {
                        (0..max_labels_resources).step_by(*max_r).map(|i| {
                            Self::save_resource_set_group(format!("group {} labels/{}/{}-{}", labels_cnt, max_r, i, i + *max_r), t).map(|g| (i, i + *max_r, g))
                        }).collect::<Result<Vec<(usize, usize, usize)>, ThoosaError>>()
                    }
                ).collect::<Result<Vec<Vec<(usize, usize, usize)>>, ThoosaError>>()?;

                let group_all_id = Self::save_resource_set_group(format!("group {} labels/all", labels_cnt), t)?;

                debug!("save_groups_by_label_cnt:  num_groups={:?}, groups_max_resources={:?}", num_groups, groups_max_resources);

                for (label, mut resources) in labels_resources {
                    resources.shuffle(&mut rng);

                    let label_id = labels_ids.get(&label); //if more than one label - label_id is None and Null in the db

                    debug!("save_groups_by_label_cnt: labels={} resources.len()={} label_id={:?}",label, resources.len(), label_id);

                    if resources.len() > 10 { //TODO min resources

                        for (group_id, _max_resources) in groups_max_ids.iter().zip(groups_max_resources.iter()) {
                            for (resources_0, resources_1, group_id) in group_id {
                                if *resources_0 < resources.len() - 1 {
                                    Self::insert_resource_set_resources(t, labels_cnt, *group_id, format!("{}/{}", labels_cnt, label), label_id, &resources[*resources_0..min(resources.len() - 1, *resources_1)])?;
                                }
                            }
                        }
                    }
                    Self::insert_resource_set_resources(t, labels_cnt, group_all_id, format!("{}/{}", labels_cnt, label), label_id, resources.as_slice())?;
                }
            }
        }
        Ok(())
    }

    fn save_pure_sets(&self, max_len: usize, only_unused: bool, t: &Transaction) -> Result<(), ThoosaError> {
        {
            let group_id = Self::save_resource_set_group(format!("pure group {}", max_len), t)?;

            let mut stmt = t.prepare("SELECT id, label FROM labels ORDER BY id")?; //TODO replace with Self::list_labels
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

            let mut rows = stmt.query(())?;
            while let Some(label_row) = rows.next()? {
                debug!("row={:?}", label_row);
                let label_id = label_row.get::<usize, usize>(0)?;
                t.execute("INSERT INTO resource_set (descr, group_id, label_id) VALUES (?, ?, ?)", params![format!("{}/{}", label_row.get::<usize, String>(1)?, max_len), group_id, label_id])?;

                let resource_set_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get::<usize, usize>(0))?;

                resource_resources_set_stmt.execute(params![resource_set_id, label_id, max_len])?;
            }
        }

        Ok(())
    }

    fn list_resource_random_not_label(&mut self, label: &str, limit: i64) -> Result<Vec<Resource>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let r = t.prepare(
            "SELECT r.id, r.import_id, r.key FROM resources r INNER JOIN labeled_resources lr ON r.id=lr.resource_id INNER JOIN labels l ON l.id=lr.label_id WHERE l.label != ? ORDER BY RANDOM() limit ?;"
        )?.query_map([Value::Text(label.to_string()), Value::Integer(limit)], |row| { //TODO move group_slices out, not for store
            Ok(Resource::new(
                row.get::<usize, u32>(0)?,
                row.get::<usize, u32>(1)?,
                row.get::<usize, String>(2)?,
            ))
        })?.map(|v| v.unwrap()).collect();

        t.commit()?;

        Ok(r)
    }

    fn list_resource_random_in_group(&mut self, group_descr: &str) -> Result<Vec<(Resource, Vec<f32>)>, ThoosaError> { //vec<(key, vec<(lablel, p)>>
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let labels = Self::list_labels(&t)?;

        debug!("list_resource_random_in_group: group_descr={:?} labels={:?}", group_descr, labels);

        let r: Vec<(Resource, Vec<f32>)> = t.prepare(
            "SELECT r.id, r.import_id, r.key, GROUP_CONCAT(l.label || '/' || lr.p, ',') FROM resource_set_group rsg \
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
                    row.get::<usize, String>(2)?
                    ),
                    labels.iter().map(|(_label_id, label)| *labels_p_vec.get(label).unwrap_or(&0.0)).collect::<Vec<f32>>()
                )
            )
        })?.map(|v| v.unwrap()).collect();

        t.commit()?;

        debug!("list_resource_random_in_group: r.len={:?}", r.len());

        Ok(r)
    }

    fn list_resources_by_ids(&mut self, file_stem: &Vec<ResourceId>) -> Result<Vec<Resource>, ThoosaError>{
        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;

        let (mut q, params) = file_stem.into_iter().fold((vec!["SELECT id, import_id, key FROM resources WHERE id IN ("], vec![]), |mut a, v| {
            a.0.push("?");
            a.0.push(",");
            a.1.push(*v);
            a
        });

        q.pop();
        q.push(") ORDER BY key");

        let resources = {
            let mut stmt = t.prepare(q.into_iter().collect::<String>().as_str())?;

            let r = stmt.query_map(rusqlite::params_from_iter(params.into_iter()), |row| {
                Ok(Resource::new(
                    row.get::<usize, u32>(0)?,
                    row.get::<usize, u32>(1)?,
                    row.get::<usize, String>(2)?,
                ))
            })?.map(|v| v.unwrap()).collect();

            r
        };

        Ok(resources)
    }


    fn save_set_by_labels(&mut self, conds: Vec<(String, SetCond)>, _only_unused: bool, limit: usize) -> Result<(), ThoosaError> {
        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;

        {
            t.execute("INSERT INTO resource_set (descr) VALUES (?)", params!["save_set_by_labels"])?;

            let resource_set_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get::<usize, i64>(0))?;

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

            if resource_resources_set_stmt.execute(rusqlite::params_from_iter(params.into_iter()))? == 0 {
                ThoosaError::new(format!("no resources for set on {:?}", conds));
            }
        }
        Ok(t.commit()?)
    }
    //TODO remove
    fn save_slices(&mut self, run_id: usize, channel: i64, slices: Vec<(TensorSliceIdx <ResourceId>, TensorSliceIdx <ResourceId>, f32)>) -> Result<(), ThoosaError> {
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
                    a.1.push(Value::Integer(v.0.1));
                    a.1.push(Value::Integer(v.1.0 as i64));
                    a.1.push(Value::Integer(v.1.1));
                    a.1.push(Value::Real(v.2 as f64));
                    a
                });

            q.pop();

            t.execute(&q.into_iter().collect::<String>(), rusqlite::params_from_iter(params.into_iter()))?;
        }
        t.commit()?;

        Ok(())
    }

    //TODO remove
    fn set_motifs_from_tmp_db(&mut self, run_id: usize, channel: i64, groups:HashMap<u16, Vec<IdSlice>>, path: &dyn AsRef<Path>) -> Result<(), ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        debug!("set_motifs: run_id={run_id:?}, channel={channel:?} path={}", path.as_ref().to_string_lossy());

        {

            t.execute(format!("ATTACH DATABASE '{}' AS slices_{run_id}_{channel}_tmp", path.as_ref().to_string_lossy()).as_str(), [])?;

            //TODO sort?
            for (group, slices) in groups {

                let motifs_inserted = t.execute("INSERT INTO motifs (group_ord, run_id, channel_id) VALUES (?, ?, ?)", [group as usize, run_id, channel as usize])?; //TODO group not needed?

                let motif_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;

                debug!("set_motifs: group={group:?}, run_id={run_id:?} motif_id={motif_id} motifs_inserted={motifs_inserted} slices={slices:?} ");

                //each match appears twice in slices but will result in single raw returned from _tmp.slice_matches, should be faster than getting slices into set on slice_matches.id (IdSlice.0)
                for slices_chunk in slices.chunks(500) {

                    let sql = format!("INSERT INTO slice_matches (motif_id, run_id, channel_id, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1, dist) \
                            SELECT ?, ?, ?, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1, dist FROM slices_{run_id}_{channel}_tmp.slice_matches WHERE id IN (");

                    let (mut q, params) = slices_chunk.into_iter().fold((vec![sql.as_str()], vec![
                        Value::Integer(motif_id),
                        Value::Integer(run_id as i64),
                        Value::Integer(channel)
                    ]),
                        |mut a: (Vec<&str>, Vec<Value>), v: &IdSlice| {
                            a.0.push("?");
                            a.0.push(", ");
                            a.1.push(Value::Integer(v.0));

                            a
                        });

                    q.pop();
                    q.push(")");

                    debug!("set_motifs: params={params:?}");

                    let slice_matches_updated = t.execute(&q.into_iter().collect::<String>(), rusqlite::params_from_iter(params.into_iter()))?;

                    debug!("set_motifs: slice_matches_updated={slice_matches_updated}");
                }
            }
        }

        t.commit()?;

        let t = guard.transaction()?;
        t.execute(format!("DETACH DATABASE slices_{run_id}_{channel}_tmp").as_str(), [])?;

        Ok(())
    }

    fn list_motifs_distances(&mut self, run_id: usize) -> Result<HashMap<usize, f32>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;
        let tensor_motifs: HashMap<usize, f32> = {
            let mut stmt = t.prepare("SELECT motif_id, avg(dist) FROM slice_matches GROUP BY motif_id HAVING motif_id NOT NULL AND run_id=?")?;

            let tensor_motifs: HashMap<usize, f32> = stmt.query_map([run_id], |row| {
                Ok((
                    row.get::<usize, usize>(0)?, // motif_id,
                    row.get::<usize, f32>(1)?, // dist
                ))
            })?.fold(HashMap::new(), |mut a, v| {
                let v = v.unwrap();
                a.insert(v.0, v.1);
                a
            });

            tensor_motifs
        };

        t.commit()?;

        Ok(tensor_motifs)
    }

    fn delete_slices_without_matches(&mut self, run_id: usize) -> Result<(), ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;
        t.execute("DELETE FROM slice_matches WHERE run_id=? AND motif_id IS NULL", [run_id])?;
        t.commit()?;

        guard.execute("VACUUM", [])?;

        Ok(())
    }

    /**
    * returns map tensor_key - MotifSlice(motif_group, tensor slice idx, channel)
    */
    fn list_motifs_slices(&mut self, run_id: usize) -> Result<HashMap<Resource, Vec<MotifSlice>>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let resources_motifs = {
            let mut stmt = t.prepare("\
                    SELECT sm.id, sm.motif_id, sm.channel_id, \
                    sm.tensor_id_0, rs_0.import_id, rs_0.key, sm.slice_idx_0, \
                    sm.tensor_id_1, rs_1.import_id, rs_1.key, sm.slice_idx_1 \
                    FROM slice_matches sm \
                    LEFT JOIN resources rs_0 on rs_0.id=sm.tensor_id_0 \
                    LEFT JOIN resources rs_1 ON rs_1.id=sm.tensor_id_1 \
                    WHERE run_id=?\
            ")?;

            let tensor_motifs: (HashMap<ResourceId, Vec<MotifSlice>>,HashMap<ResourceId, Resource>) = stmt.query_map([run_id], |row| {
                Ok((
                    row.get(0)?, // id,
                    row.get::<usize, usize>(1).ok(), // motif_id,
                    row.get(2)?, // channel_id,

                    row.get(3)?, // tensor_id_0,
                    row.get(4)?, // import_0,
                    row.get(5)?, // key_0,
                    row.get(6)?, // slice_idx_0,

                    row.get(7)?, // tensor_id_1,
                    row.get(8)?, // import_1,
                    row.get(9)?, // key_1,
                    row.get(10)?,// slice_idx_1
                ))
            })?.fold(
                Ok((HashMap::new(),HashMap::new())),
                |a: Result<(HashMap<ResourceId, Vec<MotifSlice>>,HashMap<ResourceId, Resource>), Error>, v: Result<(
                i64, Option<usize>, i64,//id, motif_id, channel_id

                u32, u32, String, i64,//tensor_id_0, import_0, key_0, slice_idx_0

                u32, u32, String, i64//tensor_id_1, import_1, key_1, slice_idx_1
                ), Error>| {

                let v = v?;
                if let Some(motif_id) = v.1 {
                    let mut a = a?;

                    a.0.entry(v.3).or_insert_with(|| vec![]).push(MotifSlice::new(motif_id, v.6, v.2 as u16));
                    a.0.entry(v.7).or_insert_with(|| vec![]).push(MotifSlice::new(motif_id, v.10, v.2 as u16));

                    if !a.1.contains_key(&v.3){
                        a.1.insert(v.3, Resource::new( v.3, v.4, v.5));
                    }
                    if !a.1.contains_key(&v.7){
                        a.1.insert(v.7, Resource::new( v.7, v.8, v.9));
                    }

                    Ok(a)
                } else {
                    a
                }
            })?;

            if tensor_motifs.0.len() < 10 {
                debug!("motifs_to_eeg_slices: eeg_motifs={:?}", tensor_motifs);
            }

            tensor_motifs.0.into_iter().fold((HashMap::new(), tensor_motifs.1), |mut a, (resource_id, slices)|{
                a.0.insert(
                    a.1.remove(&resource_id).expect(format!("not resource for {}",resource_id).as_str()),
                    slices
                );
                a
            }).0
        };
        t.commit()?;

        Ok(resources_motifs)
    }

    /**
    returns channel, find_motifs_run_id, slices
    */
    fn list_motif_slices(&mut self, motif_id: usize, limit: Option<usize>, all_but_in_run_id: Option<(usize, usize)>) -> Result<(i64, usize, Vec<TensorSliceIdx<Resource>>), ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let channel_slices = {
            let mut stmt = t.prepare(&[
                "SELECT sm.channel_id, sm.run_id, \
                    sm.tensor_id_0,  rs_0.import_id, rs_0.key, sm.slice_idx_0, \
                    sm.tensor_id_1,  rs_1.import_id, rs_1.key, sm.slice_idx_1 \
                    FROM slice_matches sm \
                    LEFT JOIN resources rs_0 on rs_0.id=sm.tensor_id_0 \
                    LEFT JOIN resources rs_1 ON rs_1.id=sm.tensor_id_1 \
                    WHERE motif_id
                ",
                // if all_but_in_run_id.is_none() {"=? AND ? IS NOT NULL AND ? IS NOT NULL "} else {"!=? AND run_id=? AND channel_id=?"}, run_id? //TODO
                if all_but_in_run_id.is_none() { "=? AND ? IS NOT NULL " } else { "!=? AND channel_id=?" },
                &limit.map(|limit| format!("LIMIT {}", limit)).unwrap_or("".to_owned())
            ].join(""))?;

            let params = match all_but_in_run_id {
                Some((_run_id, channel_id)) => [motif_id, channel_id],
                None => [motif_id, 0],
            };

            let (slices, channel_set, run_id) = stmt.query_map(params, |row| {
                Ok((
                    row.get::<usize, i64>(0)?, //channel_id
                    row.get::<usize, i64>(1)?, //run_id

                    row.get(2)?, // tensor_id_0,
                    row.get(3)?, // import_0,
                    row.get(4)?, // key_0,
                    row.get(5)?, // slice_idx_0,

                    row.get(6)?, // tensor_id_1,
                    row.get(7)?, // import_1,
                    row.get(8)?, // key_1,
                    row.get(9)?,// slice_idx_1


                ))
            })?.fold(Ok((HashSet::new(), HashSet::new(), None)), |a:Result<(HashSet<TensorSliceIdx<Resource>>, HashSet<i64>, Option<i64>), ThoosaError>, v:Result<(
                i64, i64,//channel_id, run_id

                u32, u32, String, i64,//tensor_id_0, import_0, key_0, slice_idx_0

                u32, u32, String, i64//tensor_id_1, import_1, key_1, slice_idx_1
            ), Error>| {
                let v = v?;
                let (mut a, mut channel_set,mut run_id) = a?;

                a.insert(TensorSliceIdx(Resource::new(v.2, v.3, v.4), v.5));
                a.insert(TensorSliceIdx(Resource::new(v.6, v.7, v.8), v.9));

                channel_set.insert(v.0); //TODO replace with Option

                if all_but_in_run_id.is_none() {
                    if let Some(run_id) = run_id {
                        assert!(run_id == v.1);
                    } else {
                        run_id.replace(v.1);
                    }
                }

                Ok((a, channel_set, run_id))
            })?;

            if all_but_in_run_id.is_none() {
                assert!(channel_set.len() == 1);
                (channel_set.into_iter().nth(0).expect("channel"), run_id.expect("no run id") as usize, slices.into_iter().collect())
            } else {
                let (run_id, channel_id) = all_but_in_run_id.unwrap();
                (channel_id as i64, run_id, slices.into_iter().collect()) //TODO
            }
        };

        t.commit()?;

        Ok(channel_slices)
    }


    fn list_run_resources(&mut self, run_id: usize) -> Result<Vec<String>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let result = {
            let count: i64 = t.query_row("SELECT count(*) FROM run_resources WHERE run_id=?", [run_id], |row| row.get(0))?;

            debug!("list_run_resources: count={}", count);

            let mut stmt = t.prepare("SELECT ord_id, key FROM run_resources WHERE run_id=? ORDER BY ord_id")?;


            let result = stmt.query_map([run_id], |row| {
                Ok((row.get::<usize, i64>(0)?, row.get::<usize, String>(1)?))
            })?.enumerate().map(|(i, ord_key)| {
                ord_key.map(|(ord, key)| {
                    assert!(ord == i as i64);
                    key
                }).map_err(|e| ThoosaError::new(e.to_string()))
            }).collect::<Result<Vec<String>, ThoosaError>>();

            result
        };

        t.commit()?;
        debug!("list_run_resources: result.is_ok={}", result.is_ok());

        result
    }

    fn list_run_params(&mut self, run_id: usize) -> Result<HashMap<String, Val>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        let result = {
            let mut stmt = t.prepare("SELECT name, real_value, int_value, str_value FROM run_param_values WHERE run_id=?")?;


            let result = stmt.query_map([run_id], |row| {
                Ok(
                    (
                        row.get::<usize, String>(0)?,
                        row.get::<usize, f64>(1).ok(),
                        row.get::<usize, i64>(2).ok(),
                        row.get::<usize, String>(3).ok()
                    )
                )
            })?.map(|v| v.expect(format!("can't get run_param_values for {}", run_id).as_str())).fold(HashMap::new(), |mut a, (name, real_value, int_value, str_value)| {
                a.insert(
                    name.clone(),
                    real_value.map(|v| v.into())
                        .or(int_value.map(|v| v.into()))
                        .or(str_value.map(|v| v.into()))
                        .expect(format!("no value for {}", name).as_str()),
                );
                a
            });

            result
        };

        t.commit()?;
        debug!("list_run_params: run_id={run_id} result={result:?}");

        Ok(result)
    }


    fn save_selected_motifs(&mut self, find_motifs_run_id: usize, descr: &str, selected_motifs: Vec<Vec<(usize, f64)>>) -> Result<usize, ThoosaError> {
        debug!("save_selected_motifs: selected_motifs={:?}", selected_motifs);

        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        t.execute("INSERT INTO tree_run (descr) VALUES (?)", params![descr])?;
        let motif_selection_run_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get::<usize, usize>(0))?;


        t.execute("INSERT INTO tree_run_input (run_id, input_run_id) VALUES (?, ?)", params![motif_selection_run_id, find_motifs_run_id])?;

        let selected_motifs = selected_motifs.into_iter().filter(|selected_motif| selected_motif.len() > 0).collect::<Vec<Vec<(usize, f64)>>>();
        debug!("save_selected_motifs: filtered selected_motifs={:?}", selected_motifs);

        if selected_motifs.len() > 0 {
            for selected_motifs_chunk in selected_motifs.chunks(500) {
                let (mut q, params) = selected_motifs_chunk.into_iter().flat_map(|channel_motif_corrs| channel_motif_corrs.into_iter()).into_iter().fold(
                    (vec!["INSERT INTO selected_motifs (tree_run_id, motif_id, rate) VALUES "], vec![]), //TODO check if channel matches one in motif
                    |mut a: (Vec<&str>, Vec<Value>), (motif_id, rate)| {
                        a.0.push("(?, ?, ?)");
                        a.0.push(", ");
                        a.1.push(Value::Integer(motif_selection_run_id as i64));
                        a.1.push(Value::Integer(*motif_id as i64));
                        a.1.push(Value::Real(*rate));

                        a
                    });

                q.pop();


                t.execute(&q.into_iter().collect::<String>(), rusqlite::params_from_iter(params.into_iter()))?;
            }
        }
        t.commit()?;

        Ok(motif_selection_run_id)
    }

    /*
     returns - Vec<id, input_run_id>
     */
    fn list_input_run_ids(&mut self, run_ids: &Vec<usize>) -> Result<Vec<(usize, usize)>, ThoosaError> {
        assert!(!run_ids.is_empty());
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;


        let (mut q, params) = run_ids.into_iter().fold((vec!["SELECT run_id, input_run_id FROM tree_run_input WHERE run_id IN ("], vec![]), |mut a, v| {
            a.0.push("?");
            a.0.push(",");
            a.1.push(*v);
            a
        });

        q.pop();
        q.push(")");

        let id__input_id = {
            let mut stmt = t.prepare(q.into_iter().collect::<String>().as_str())?;

            let r = stmt.query_map(rusqlite::params_from_iter(params.into_iter()), |row| {
                Ok((
                    row.get::<usize, usize>(0)?,
                    row.get::<usize, usize>(1)?,
                ))
            })?.map(|v| v.unwrap()).collect();

            r
        };
        t.commit()?;


        Ok(id__input_id)
    }

    fn list_output_run_ids(&mut self, input_run_id: usize) -> Result<Vec<usize>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;


        let q = "SELECT run_id FROM tree_run_input WHERE input_run_id=?";

        let id__input_id = {
            let mut stmt = t.prepare(q)?;

            let r = stmt.query_map([input_run_id], |row| {
                row.get::<usize, usize>(0)
            })?.map(|v| v.unwrap()).collect();

            r
        };
        t.commit()?;


        Ok(id__input_id)
    }

    /**k
        * returns vec of channel X (motif id, corr)
    */
    fn list_selected_motifs(&mut self, selected_motif_run_ids: usize) -> Result<Vec<Vec<(usize, f64)>>, ThoosaError> {
        let mut channels_motifs = (0..20).into_iter().map(|_| Vec::with_capacity(10)).collect::<Vec<Vec<(usize, f64)>>>();

        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        {
            let mut stmt = t.prepare("SELECT DISTINCT sm.motif_id, sm.rate, m.channel_id FROM selected_motifs sm LEFT JOIN motifs AS m ON m.id=sm.motif_id WHERE sm.tree_run_id=?")?;

            stmt.query_map([selected_motif_run_ids], |row| {
                Ok((row.get::<usize, usize>(0)?, row.get::<usize, f32>(1)?, row.get::<usize, usize>(2)?))
            })?.map(|v| v.unwrap()).for_each(|(motif_id, corr, channel_id)| {
                channels_motifs[channel_id].push((motif_id, corr as f64))
            });
        }
        t.commit()?;

        Ok(channels_motifs)
    }


    fn tree_run_descr(&mut self, run_id: usize) -> Result<Vec<(i64, String)>, ThoosaError> {
        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;

        let descr: String = t.query_row("SELECT descr FROM tree_run WHERE id=?", [run_id], |row| row.get(0))?;

        debug!("descr={:?}", descr);

        parse_corr_labels(descr.as_str())
    }

    fn conn(&self) -> &Mutex<Connection> {
        &self.conn
    }
}


impl SqlStore {

    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Box<dyn Store>, ThoosaError> {
        Self::create_tables(Connection::open(path)?)
    }
    // #[cfg(test)]
    pub fn in_mem() -> Result<Box<dyn Store>, ThoosaError> {
        Self::create_tables(Connection::open_in_memory()?)
    }


    fn create_tables(mut conn: Connection) -> Result<Box<dyn Store>, ThoosaError> {
        {
            let t = conn.transaction()?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS imports (
                id          INTEGER PRIMARY KEY,
                descr       VARCHAR(256)
            )", ())?;

            t.execute( //TODO UNIQUE(key) doesn't work - no error
                       "CREATE TABLE IF NOT EXISTS resources (
                id          INTEGER PRIMARY KEY,
                key         VARCHAR(256),
                import_id   NOT NULL REFERENCES imports(id) ON DELETE CASCADE,
                UNIQUE(key) )", ())?;

            t.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_resources_key ON resources (key);", ())?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS resource_set_group (
                id          INTEGER PRIMARY KEY,
                descr       VARCHAR(256)
            )", ())?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS resource_set (
                id          INTEGER PRIMARY KEY,
                group_id    REFERENCES resource_set_group(id) ON DELETE CASCADE,
                label_id    REFERENCES labels(id) ON DELETE CASCADE,
                descr       VARCHAR(256))",
                ())?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS resource_resources_set (
                resource_id  NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
                set_id       NOT NULL REFERENCES resource_set(id) ON DELETE CASCADE,
                PRIMARY KEY (resource_id, set_id)
            )", ())?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS labels (
                id          INTEGER PRIMARY KEY,
                label       VARCHAR(256),
                UNIQUE(label)
            )", ())?;

            t.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_labels_label ON labels (label);", ())?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS labeled_resources (
                resource_id    NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
                label_id       NOT NULL REFERENCES labels(id) ON DELETE CASCADE,
                p              FLOAT NOT NULL,
                PRIMARY KEY (resource_id, label_id)
            )", ())?;

            //descr - dist, dist etc
            t.execute(
                "CREATE TABLE IF NOT EXISTS tree_run (
                id              INTEGER PRIMARY KEY,
                descr           VARCHAR(256),
                run_on          DATETIME DEFAULT CURRENT_TIMESTAMP
            )", ())?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS tree_run_input (
                input_run_id     NOT NULL REFERENCES tree_run(id) ON DELETE CASCADE,
                run_id     NOT NULL REFERENCES tree_run(id) ON DELETE CASCADE,

                PRIMARY KEY (input_run_id, run_id)
            )", ())?;

            //key - resource id  //TODO remove ord_id?
            t.execute(
                "CREATE TABLE IF NOT EXISTS run_resources (
                run_id     NOT NULL REFERENCES tree_run(id) ON DELETE CASCADE,
                ord_id     INTEGER NOT NULL,
                key        VARCHAR(256),
                PRIMARY KEY (run_id, ord_id)
            )", ())?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS run_param_values (
                id          INTEGER PRIMARY KEY,
                run_id      NOT NULL REFERENCES tree_run(id) ON DELETE CASCADE,
                name        VARCHAR(256),
                real_value  REAL,
                int_value   INTEGER,
                str_value   VARCHAR(256),
                label_id    REFERENCES labels(id) ON DELETE CASCADE
                )",
                (),
            )?;
            t.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_run_param_values_name_run_id ON run_param_values (run_id, name, label_id);", ())?;

            t.execute( //group - ordinal as appeared when grouping, not needed
                       "CREATE TABLE IF NOT EXISTS motifs (
                id              INTEGER PRIMARY KEY,
                group_ord       INTEGER NOT NULL,
                run_id          NOT NULL REFERENCES tree_run(id) ON DELETE CASCADE,
                channel_id      INTEGER NOT NULL
            )", ())?;

            //TODO remove
            t.execute(
                "CREATE TABLE IF NOT EXISTS slice_matches (
                id              INTEGER PRIMARY KEY,
                run_id          NOT NULL REFERENCES tree_run(id) ON DELETE CASCADE,
                channel_id      INTEGER NOT NULL,
                tensor_id_0     REFERENCES resources(id) ON DELETE CASCADE,
                slice_idx_0     INTEGER NOT NULL,
                tensor_id_1     REFERENCES resources(id) ON DELETE CASCADE,
                slice_idx_1     INTEGER NOT NULL,
                dist             FLOAT NOT NULL,
                motif_id        REFERENCES motifs(id) ON DELETE CASCADE DEFAULT NULL
            )", ())?;

            //labels_motifs on resource_set
            t.execute( //todo label
                       "CREATE TABLE IF NOT EXISTS selected_motifs (
                motif_id   REFERENCES motifs(id),
                tree_run_id      NOT NULL REFERENCES tree_run(id) ON DELETE CASCADE,
                rate    FLOAT NOT NULL
            )", ())?;

            t.execute("commit", ())?;
        }

        Ok(Box::new(Self {
            conn: Mutex::new(conn)
        }))
    }

    fn list_resource_set_resources(set_id: usize, t: &Transaction) -> Result<Vec<Resource>, ThoosaError> {
        let set_resources = {
            let mut stmt = t.prepare("SELECT rs.id, rs.import_id, rs.key FROM resource_resources_set rrs LEFT JOIN resources rs ON rs.id=rrs.resource_id WHERE rrs.set_id=?")?;


            let vec = stmt.query_map([set_id], |row| {
                Ok(Resource::new(
                        row.get::<usize, u32>(0)?,
                        row.get::<usize, u32>(1)?,
                        row.get::<usize, String>(2)?,
                 ))
                // row.get::<usize, u32>(0)
            })?.map(|v| v.unwrap()).collect();

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

            let vec = stmt.query_map([], |row| {
                Ok((
                    row.get::<usize, usize>(0)?,
                    row.get::<usize, String>(1)?
                ))
            })?.collect::<Result<Vec<(usize, String)>, Error>>()?;

            vec
        };
        Ok(set_resources)
    }

    fn save_param_value(t: &Transaction, run_id: usize, param_name: &str, value: Val, label_id: Option<usize>) -> Result<(), ThoosaError> {
        match value {
            Val::Real(v) => {
                t.execute("INSERT INTO run_param_values (run_id, label_id, name, real_value) VALUES (?, ?, ?, ?)", params![run_id, label_id, param_name, v])?;
            }
            Val::Int(v) => {
                t.execute("INSERT INTO run_param_values (run_id, label_id, name, int_value) VALUES (?, ?, ?, ?)", params![run_id, label_id, param_name, v])?;
            }
            Val::Str(v) => {
                t.execute("INSERT INTO run_param_values (run_id, label_id, name, str_value) VALUES (?, ?, ?, ?)", params![run_id, label_id, param_name, v])?;
            }
        }

        Ok(())
    }


    fn save_resource_set_group(descr: String, t: &Transaction) -> Result<usize, ThoosaError> {
        t.execute("INSERT INTO resource_set_group (descr) VALUES (?)", params![descr])?;
        let group_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get::<usize, usize>(0))?;
        Ok(group_id)
    }

    /**
    * label_id - for pure sets
    */
    fn insert_resource_set_resources(t: &Transaction, _labels_cnt: usize, group_id: usize, descr: String, label_id: Option<&usize>, resources: &[usize]) -> Result<(), ThoosaError> {
        t.execute(
            "INSERT INTO resource_set (descr, group_id, label_id) VALUES (?, ?, ?)",
            params![descr, group_id, label_id], //labels?
        )?;

        let resource_set_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get::<usize, usize>(0))?;

        for resources_chunk in resources.chunks(500) {
            let (mut q, params) = resources_chunk.into_iter().fold((vec!["INSERT INTO resource_resources_set (set_id, resource_id) VALUES"], vec![]), |mut a, resources_id| {
                a.0.push("(?, ?)");
                a.0.push(",");
                a.1.push(resource_set_id);
                a.1.push(*resources_id);
                a
            });

            q.pop();

            t.execute(&q.into_iter().collect::<String>(), rusqlite::params_from_iter(params.into_iter()))?;
        }


        Ok(())
    }
}

fn conds_to_q(input: &Vec<(String, SetCond)>) -> String {
    assert!(input.len() > 0);
    let inp_len = input.len() * 6 - 1;

    input.into_iter().flat_map(|v| {
        ["(l.label='", v.0.as_str(), "' and ", v.1.p(), "?)", " or "]
    }).take(inp_len).collect::<String>()
}


pub struct SqlMotifTrackerStore {
    run_id: usize,
    channel: usize,
    conn: Mutex<Connection>,
    path: PathBuf
}

impl SqlMotifTrackerStore {




    pub fn new<P: AsRef<Path> + Sync + Send + Debug>(path: P, run_id: usize, channel: usize) -> Result<Box<dyn MotifTrackerStore>, ThoosaError> {
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
            )", ())?;

            t.commit()?;

        }
        Ok(Box::new(Self {
            run_id,
            channel,
            conn: Mutex::new(conn),
            path: path.as_ref().to_path_buf()
        }))
    }


}

pub trait MotifTrackerStore:Debug + Send + Sync {
    fn save_slices(&mut self, slices: Vec<(TensorSliceIdx <ResourceId>, TensorSliceIdx <ResourceId>, f32)>) -> Result<(), ThoosaError>;
    fn build_groups(&mut self) -> Result<HashMap<u16, Vec<IdSlice>>, ThoosaError>;

    fn run_id(&self) -> usize;
    fn channel(&self) -> usize;
    fn path(&self) -> &PathBuf;
}

impl MotifTrackerStore for SqlMotifTrackerStore {

    fn save_slices(&mut self, slices: Vec<(TensorSliceIdx<ResourceId>, TensorSliceIdx<ResourceId>, f32)>) -> Result<(), ThoosaError> {
        debug!("MotifTrackerStore::save_slices: motifs.len()={:?}, self.run_id={:?}", slices.len(), self.run_id);

        // self.store.lock().save_slices(self.run_id, self.channel as i64, motifs)?; //TODO move TensorSliceIdx to shared

        let mut conn_guard = self.conn.lock();
        let t = conn_guard.transaction()?;

        assert!(slices.len() > 0);

        for slices_chunk in slices.chunks(500) {
            let (mut q, params) = slices_chunk.into_iter().fold(
                (vec!["INSERT INTO slice_matches (tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1, dist) VALUES"], vec![]), |mut a: (Vec<&str>, Vec<Value>), v| {
                    a.0.push("(?, ?, ?, ?, ?)");
                    a.0.push(", ");

                    a.1.push(Value::Integer(v.0.0 as i64));
                    a.1.push(Value::Integer(v.0.1));
                    a.1.push(Value::Integer(v.1.0 as i64));
                    a.1.push(Value::Integer(v.1.1));
                    a.1.push(Value::Real(v.2 as f64));
                    a
                });

            q.pop();

            t.execute(&q.into_iter().collect::<String>(), rusqlite::params_from_iter(params.into_iter()))?;
        }
        t.commit()?;

        Ok(())
    }

    fn build_groups(&mut self) -> Result<HashMap<u16, Vec<IdSlice>>, ThoosaError> {
        let mut guard = self.conn.lock();
        let t = guard.transaction()?;

        debug!("set_motifs: run_id={:?}, channel={:?}", self.run_id, self.channel);

        let groups = {
            let mut stmt = t.prepare("SELECT id, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1 FROM slice_matches")?;

            let groups = group_slices(
                stmt.query_map([], |row| { //TODO move group_slices out, not for store
                    Ok(
                        (
                            IdSlice(row.get(0)?, row.get(1)?, row.get(2)?),
                            IdSlice(row.get(0)?, row.get(3)?, row.get(4)?)
                        )
                    )
                })?.map(|v| v.unwrap())
            );

            debug!("set_motifs: run_id={:?}, channel={:?} groups={:?}", self.run_id, self.channel, groups);


            groups
        };
        t.commit()?;


        Ok(groups)

    }

    fn run_id(&self) -> usize {
        self.run_id
    }

    fn channel(&self) -> usize {
        self.channel
    }

    fn path(&self) -> &PathBuf {
        &self.path
    }
}

#[cfg(test)]
mod store_tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use std::time::Instant;

    use log::debug;
    use thoosa_shared::resource::{Resource, ResourceId};
    use thoosa_shared::{cdt, cdt_for_dir, TensorSliceIdx};
    use thoosa_shared::val::Val;

    use crate::{ThoosaError, SetCond, SqlStore, Store, SqlMotifTrackerStore, MotifTrackerStore};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::init_test_logging();
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
        store.save_resources_labels(&(0..n).fold(HashMap::new(), |mut a: HashMap<String, Vec<(String, f32)>>, i| {
            a.insert(format!("{:3}_aaa", i), vec![("aaa".into(), 1.0)]);
            a
        }),
                                    import_id, &t)?;

        store.save_resources_labels(&(0..n).fold(HashMap::new(), |mut a: HashMap<String, Vec<(String, f32)>>, i| {
            a.insert(format!("{:3}_bbb", i), vec![("bbb".into(), 1.0)]);
            a
        }),
                                    import_id, &t)?;

        store.save_resources_labels(&(0..n).fold(HashMap::new(), |mut a: HashMap<String, Vec<(String, f32)>>, i| {
            a.insert(format!("{:3}_ccc", i), vec![("ccc".into(), 1.0)]);
            a
        }),
                                    import_id, &t)?;


        //two labels resources
        store.save_resources_labels(&(0..n).fold(HashMap::new(), |mut a: HashMap<String, Vec<(String, f32)>>, i| {
            a.insert(format!("{:3}_ab", i), vec![("aaa".into(), 0.5), ("bbb".into(), 0.5)]);
            a
        }),
                                    import_id, &t)?;

        store.save_resources_labels(&(0..n).fold(HashMap::new(), |mut a: HashMap<String, Vec<(String, f32)>>, i| {
            a.insert(format!("{:3}_bc", i), vec![("bbb".into(), 0.5), ("ccc".into(), 0.5)]);
            a
        }),
                                    import_id, &t)?;

        store.save_resources_labels(&(0..n).fold(HashMap::new(), |mut a: HashMap<String, Vec<(String, f32)>>, i| {
            a.insert(format!("{:3}_ca", i), vec![("ccc".into(), 0.5), ("aaa".into(), 0.5)]);
            a
        }),
                                    import_id, &t)?;

        //three labels resources
        store.save_resources_labels(&(0..n).fold(HashMap::new(), |mut a: HashMap<String, Vec<(String, f32)>>, i| {
            a.insert(format!("{:3}_abc", i), vec![("aaa".into(), 0.3), ("bbb".into(), 0.3), ("ccc".into(), 0.4)]);
            a
        }),
                                    import_id, &t)?;


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
        // std::fs::remove_file("/tmp/test_db.db3").unwrap_or(());
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        fs::create_dir_all(&test_dir)?;
        let mut store = SqlStore::from_path(test_dir.join("test_db.db3"))?;
        // let mut store = SqlStore::in_mem()?;
        let resource_ids= {
            let mut mutex_guard = store.conn().lock();
            let t = mutex_guard.transaction()?;

            let import_id = store.save_import("test".to_string(), &t)?;

            let _aaa_id = store.save_label("aaa", &t)?;
            let _bbb_id = store.save_label("bbb", &t)?;
            let _ccc_id = store.save_label("ccc", &t)?;


            let resource_ids = store.save_resources_labels(&HashMap::from([
                ("1111".into(), vec![("aaa".into(), 0.5), ("bbb".into(), 0.5)]),
                ("2222".into(), vec![("aaa".into(), 1.0)]),
                ("3333".into(), vec![("ccc".into(), 1.0)]),
                ("444".into(), vec![]),
                ("555".into(), vec![]),
                ("666".into(), vec![]),
            ]), import_id, &t)?;

            debug!("resource_ids={resource_ids:?}");

            store.save_pure_sets(10, false, &t)?;
            t.commit()?;
            resource_ids
        };
        // let _set_1_id = store.save_resources_set("Set1".to_string(), vec!["1111".into(), "2222".into()], None)?;
        // let _set_2_id = store.save_resources_set("Set2".to_string(), vec!["2222".into(), "3333".into()], None)?;


        // store.save_set_by_labels(vec![("aaa".into(), SetCond::Gt(0.9))], true, 10)?;

        let all_group_id = store.save_resource_set_group("group 1 labels/100/100-200".into())?;
        let _set_all_id = store.save_resources_set("All".to_string(), vec!["1111".into(), "2222".into(), "3333".into()], Some(all_group_id))?;


        let group_sets = store.list_resource_set_group_sets("pure group 10")?;
        assert_eq!(3, group_sets.len());


        let mut keys_labels_p = store.list_resource_random_in_group("group 1 labels/100/100-200")?;

        keys_labels_p.sort_by(|a, b| ((a.1[0] * 10.) as u32).cmp(&((b.1[0] * 10.) as u32)));

        debug!("keys_labels_p={:?}", keys_labels_p);

        assert_eq!(
            vec![
                vec![0.0, 0.0, 1.0],
                vec![0.5, 0.5, 0.0],
                vec![1.0, 0.0, 0.0],
            ],
            keys_labels_p.into_iter().map(|v|v.1).collect::<Vec<Vec<f32>>>()
        );

        //Vec<(TensorSliceIdx <ResourceId>, TensorSliceIdx <ResourceId>, f32)>
        let slices: Vec<(TensorSliceIdx<ResourceId>, TensorSliceIdx<ResourceId>, f32)>     = vec![
            (TensorSliceIdx(resource_ids[0], 100i64), TensorSliceIdx(resource_ids[0], 200i64), 0.91),
            (TensorSliceIdx(resource_ids[0], 100i64), TensorSliceIdx(resource_ids[1], 200i64), 0.92),
            (TensorSliceIdx(resource_ids[3], 100i64), TensorSliceIdx(resource_ids[3], 200i64), 0.93),
            (TensorSliceIdx(resource_ids[3], 200i64), TensorSliceIdx(resource_ids[5], 200i64), 0.94),
            (TensorSliceIdx(resource_ids[3], 200i64), TensorSliceIdx(resource_ids[5], 300i64), 0.94),
        ];


        let run_id = store.save_tree_run(
            None,
            "test", "",
            HashMap::from([("prop1".into(), Val::Int(1)), ("prop2".into(), Val::Real(2.0)), ("prop3".into(), Val::Str("333".to_string()))]),
        )?;
        /*
                let next_run_id = store.save_tree_run(
                    Some(&[run_id]),
                    "test next", "",
                    HashMap::from([("next_prop1".into(), Val::Int(1)), ("next_prop2".into(), Val::Real(2.0)), ("next_prop3".into(), Val::Str("333".to_string()))]),
                )?;

                store.save_params_value(
                    HashMap::from([("label.prop1".into(), Val::Int(10)), ("label.prop2".into(), Val::Real(20.0)), ("label.prop3".into(), Val::Str("3330".to_string()))]),
                    run_id,
                    Some("aaa"),
                )?;

                store.save_params_value(
                    HashMap::from([("label.prop1".into(), Val::Int(1)), ("label.prop2".into(), Val::Real(2.0)), ("label.prop3".into(), Val::Str("333".to_string()))]),
                    run_id,
                    Some("bbb"),
                )?;

                store.save_params_value(
                    HashMap::from([("label.prop1".into(), Val::Int(1)), ("label.prop2".into(), Val::Real(2.0)), ("label.prop3".into(), Val::Str("333".to_string()))]),
                    run_id,
                    None,
                )?;

                let loaded_params = store.list_run_params(run_id)?;

                // debug!("loaded_params={:?}", loaded_params);


                assert_eq!(loaded_params.get("prop1").expect("prop1"), &Val::Int(1));
                assert_eq!(loaded_params.get("prop2").expect("prop2"), &Val::Real(2.0));
                assert_eq!(loaded_params.get("prop3").expect("prop3"), &Val::Str("333".to_string()));

                let loaded_params_aaa = store.list_run_params(run_id)?;

                debug!("loaded_params_aaa={:?}", loaded_params_aaa);

                 */

        let slices_store_path = test_dir.join("test_slices_db.db3");
        let mut slices_store = SqlMotifTrackerStore::new(
            &slices_store_path,
            run_id,
            7
        ).expect("can't create tmp sotre");

        slices_store.save_slices(slices.clone())?;

        let groups = slices_store.build_groups()?;

        store.set_motifs_from_tmp_db(run_id, 7, groups, &slices_store_path)?;

        {
            let mut guard = store.conn().lock();
            let t2 = guard.transaction()?;
            let mut stmt = t2.prepare("SELECT run_id, channel_id, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1, motif_id FROM slice_matches ORDER BY tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1")?;

            let rows = stmt.query_map([], |row| {
                Ok((
                    row.get(0)?, //run_id
                    row.get(1)?, //channel_id
                    row.get(2)?, //tensor_id_0
                    row.get(3)?, //slice_idx_0
                    row.get(4)?, //tensor_id_1
                    row.get(5)?, //slice_idx_1
                    row.get(6)? //motif_id
                ))
            })?.map(|v| v.unwrap()).collect::<Vec<(i64, i64, ResourceId, i64, ResourceId, i64, i64)>>();

            debug!("rows={:?}", rows);

            for i in 0..5 {
                assert_eq!(&slices[i].0.0, &rows[i].2);
                assert_eq!(&slices[i].0.1, &rows[i].3);
                assert_eq!(&slices[i].1.0, &rows[i].4);
                assert_eq!(&slices[i].1.1, &rows[i].5);
            }

        };

        let map = store.list_motifs_slices(run_id)?;
        debug!("map={:?}", map);

        // assert_eq!(4, map.len());
        let motif_slices_1111 = map.get(&Resource::new(resource_ids[0], 1, "".into())).unwrap();
        assert_eq!(3, motif_slices_1111.len());

        let motif_slices_2222 = map.get(&Resource::new(resource_ids[1], 1, "".into())).unwrap();
        assert_eq!(1, motif_slices_2222.len());

        let motif_slices_444 = map.get(&Resource::new(resource_ids[3], 1, "".into())).unwrap();
        assert_eq!(4, motif_slices_444.len());
        assert_eq!(2, map.get(&Resource::new(resource_ids[5], 1, "".into())).unwrap().len());

        debug!("motif_slices_1111={motif_slices_1111:?}");
        debug!("motif_slices_444={motif_slices_444:?}");

        let motif_selection_run_id = store.save_selected_motifs(
            run_id,
            "test motifs",
            vec![vec![
                (motif_slices_444[0].motif_id, 0.1),//motif_id corr
                (motif_slices_1111[0].motif_id, 0.2)//motif_id corr
            ]],
        )?;

        let input_run_ids = store.list_input_run_ids(&vec![motif_selection_run_id])?;

        assert_eq!([(motif_selection_run_id, run_id)], input_run_ids.as_slice());

        let motifs = store.list_selected_motifs(motif_selection_run_id)?;

        debug!("motifs: {:?}", motifs);

        //7th channel
        assert_eq!(motif_slices_444[0].motif_id, motifs[7][0].0);
        assert_eq!(motif_slices_1111[0].motif_id, motifs[7][1].0);
        //
        // //TODO
        let (channel, run_id_, mut slices) = store.list_motif_slices(motif_slices_1111[0].motif_id, None, None)?;
        slices.sort();

        debug!("slices={slices:?}");

        assert_eq!(vec![
            TensorSliceIdx(Resource::new(resource_ids[0], 1, "1111".into()), 100), TensorSliceIdx(Resource::new(resource_ids[0], 1, "1111".into()), 200), TensorSliceIdx(Resource::new(2, 1, "444".into()), 200)
        ], slices);



        assert_eq!(7, channel);
        assert_eq!(run_id, run_id_);
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
        //     store.tree_run_descr(motif_selection_run_id)?
        // );
        //
        store.delete_slices_without_matches(run_id)?;

        Ok(())
    }

    #[test]
    fn test_conv_cond() {
        let input = vec![("aaa".into(), SetCond::Lt(0.4)), ("bbb".into(), SetCond::Gt(0.1))];

        let conds = super::conds_to_q(&input);
        assert_eq!("(l.label='aaa' and p<?) or (l.label='bbb' and p>?)", conds);
    }

    #[test]
    fn test_get_prev_run_id() -> Result<(), ThoosaError> {

        // std::fs::remove_file("/tmp/test_db.db3").unwrap_or(());
        // let mut store = SqlStore::from_path("/tmp/test_db.db3")?;
        let mut store = SqlStore::in_mem()?;

        let input_run_id = store.save_tree_run(None, "test 0", "", HashMap::new())?;
        let run_id = store.save_tree_run(Some(&[input_run_id]), "test 1", "", HashMap::new())?;

        assert_eq!(store.get_input_run_ids(run_id)?[0], input_run_id);

        Ok(())
    }

    #[test]
    fn test_list_resources_by_file_stem() -> Result<(), ThoosaError> {
        let mut store = SqlStore::in_mem()?;
        {
            let mut mutex_guard = store.conn().lock();
            let t = mutex_guard.transaction()?;

            let import_id = store.save_import("test".to_string(), &t)?;

            store.save_resources_labels(&HashMap::from([
                ("resource_aaa_0".to_string(), vec![]),
                ("resource_aaa_1".to_string(), vec![]),
                ("resource_bbb_0".to_string(), vec![]),
                ("resource_bbb_1".to_string(), vec![]),
            ]),
                                        1, &t)?;

            t.commit()?;
        }

        assert_eq!(
            vec![
                "resource_aaa_0".to_string(),
                "resource_aaa_1".to_string()
            ],
            store.list_resources_by_file_stems(&vec!["%aaa%"])?.into_iter().map(|r|r.file_stem().to_string()).collect::<Vec<String>>()
        );

        assert_eq!(
            vec![
                "resource_bbb_0".to_string(),
                "resource_bbb_1".to_string()
            ],
            store.list_resources_by_file_stems(&vec!["%bbb%"])?.into_iter().map(|r|r.file_stem().to_string()).collect::<Vec<String>>()
        );

        Ok(())
    }




}
