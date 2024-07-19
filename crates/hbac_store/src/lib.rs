#![feature(iter_array_chunks)]

use std::array::IntoIter;
use std::cmp::{max, min};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::{Arc, Mutex};

use log::{debug, trace};
use rand::prelude::SliceRandom;
use rand::thread_rng;
use rusqlite::{Connection, Error, params, Transaction};
use rusqlite::types::Value;

use hbac_shared::*;
use hbac_shared::val::Val;

pub struct StoreDumper {
    pub store:Arc<Mutex<dyn Store>>, //TODO pub
    pub run_id:usize,
    pub channel:usize
}

impl StoreDumper {
    pub fn new(store: Arc<Mutex<impl Store + 'static>>, run_id:usize, channel:usize) -> Self {
        Self {
            run_id,
            store,
            channel
        }
    }
}

#[derive(Debug)]
struct IdSlice(i64, String, i64); //id, tensor id, slice idx

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
            SetCond::Eq(_v)=>{
                "p="
            }
            SetCond::Gt(_v)=>{
                "p>"
            }
            SetCond::Lt(_v)=>{
               "p<"
            }
        }
    }

    fn v(&self) -> f32 {
        match self {
            SetCond::Eq(v)=>{
                *v
            }
            SetCond::Gt(v)=>{
                *v
            }
            SetCond::Lt(v)=>{
                *v
            }
        }
    }
}


pub trait Store:Sync + Send{
    fn save_tree_run(&mut self, input_run_id: Option<usize>, descr:&str, keys:&Vec<String>, properties_vals:HashMap<String, Val>) -> Result<usize, HbacError>;
    fn save_resources(&mut self, resource_keys:Vec<String>) -> Result<(), HbacError>;
    fn save_resources_set(&mut self, set:String, resource_keys:Vec<String>, group_id: Option<usize>) -> Result<usize, HbacError>;
    fn save_resource_set_group(&mut self, descr:String) -> Result<usize, HbacError>;
    fn save_pure_sets(&mut self, max_len:usize, only_unused: bool) -> Result<(), HbacError>;
    fn save_set_by_labels(&mut self, conds:Vec<(String, SetCond)>, only_unused:bool, limit:usize) -> Result<(), HbacError>;
    fn save_groups_by_label_cnt(&mut self, max_resources_in_set: usize) -> Result<(), HbacError>;
    fn list_resource_set_group_sets(&mut self, group_descr:&str) -> Result<Vec<(usize, String)>, HbacError>;
    fn list_group_labeled_resources_sets(&mut self, group_decr:&str) -> Result<Vec<(String, String)>, HbacError>;
    fn list_resource_set_resources(&mut self, set_id:usize)  -> Result<Vec<String>, HbacError>;
    fn list_resource_set_resources_by_descr(&mut self, descr:&str)  -> Result<Vec<String>, HbacError>;
    fn list_resource_random_not_label(&mut self, label:&str, limit:i64)  -> Result<Vec<String>, HbacError>;
    fn list_resource_random_in_group(&mut self, group_descr:&str)  -> Result<Vec<(String, Vec<f32>)>, HbacError>;
    fn save_label(&mut self, label:&str) -> Result<usize, HbacError>;
    fn list_labels(&mut self) -> Result<Vec<String>, HbacError>;

    // fn save_param(&mut self, name:&str, data_type:&str) -> Result<usize, HbacError>;
    fn save_params_value(&mut self, params:HashMap<String, Val>, run_id:usize, label:Option<&str>) -> Result<(), HbacError>;

    fn save_resources_labels(&mut self, resource_labels_probs:HashMap<String, Vec<(String, f32)>>) -> Result<(), HbacError>;
    fn save_slices(&mut self, run_id:usize, channel:i64, slices:Vec<((String, i64), (String, i64), f32)>) -> Result<(), HbacError>;
    fn set_motifs(&mut self, run_id:usize, channel:i64) -> Result<(), HbacError>;
    fn list_motifs_distances(&mut self, run_id:usize) -> Result<HashMap<usize, f32>, HbacError>;
    fn list_motifs_slices(&mut self, run_id:usize) -> Result<HashMap<String, Vec<MotifSlice>>, HbacError>;
    fn list_run_resources(&mut self, run_id:usize) -> Result<Vec<String>, HbacError>;
    fn list_run_params(&mut self, run_id:usize, label: Option<&str>) -> Result<HashMap<String, Val>, HbacError>;
    fn save_motifs_corrs(&mut self, run_id: usize, descr:&str, motifs_corrs:Vec<Vec<(usize, f64)>>, label: String) -> Result<usize, HbacError>;
    fn list_corrs(&mut self, ids:&Vec<usize>) -> Result<Vec<(usize, usize, String)>, HbacError>;//corr_run_id, run_id, label
    fn list_corr_motifs(&mut self, corr_run_id:usize) -> Result<Vec<Vec<(usize, f64)>>, HbacError>;
    fn list_motif_slices(&mut self, motif_id:usize, limit: Option<usize>, all_but_in_run_id: Option<(usize, usize)>) -> Result<(i64, usize, Vec<TensorSliceIdx<String>>), HbacError>;
    fn corrs_descr(&mut self, corr_run_id:usize) -> Result<Vec<(i64, String)>, HbacError>;
    fn delete_slices_without_matches(&mut self, run_id:usize) -> Result<(), HbacError>;
}

impl Debug for StoreDumper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("")
            .finish()
    }
}

pub struct SqlStore {
    conn: Mutex<Connection>
}

impl Store for SqlStore {

    fn save_tree_run(&mut self, _input_run_id: Option<usize>, descr:&str, keys:&Vec<String>, properties_vals:HashMap<String, Val>) -> Result<usize, HbacError> {
        let mut conn_guard = self.conn.lock()?;
        let t = conn_guard.transaction()?;

        t.execute("INSERT INTO tree_run (descr) VALUES (?)", params![descr])?;

        let run_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;

        if keys.len() > 0 {
            let (mut q, params) = keys.into_iter().enumerate().fold(
                (vec!["INSERT INTO run_resources (run_id, ord_id, key) VALUES"], vec![]), |mut a: (Vec<&str>, Vec<Value>), (ord_id, v)| {
                    a.0.push("(?, ?, ?)");
                    a.0.push(", ");
                    a.1.push(Value::Integer(run_id as i64));
                    a.1.push(Value::Integer(ord_id as i64));
                    a.1.push(Value::Text(v.clone()));
                    a
                });

            q.pop();

            t.execute(&q.into_iter().collect::<String>(), rusqlite::params_from_iter(params.into_iter()))?;
        }

        if properties_vals.len() > 0 {
            for property_val in properties_vals.into_iter() {
                Self::save_param_value(&t, run_id, &property_val.0, property_val.1, None)?;
            }
        }

        t.commit()?;

        Ok(run_id)
    }

    fn save_params_value(&mut self, params:HashMap<String, Val>, run_id:usize, label:Option<&str>) -> Result<(), HbacError> {
        let mut conn_guard = self.conn.lock()?;
        let t = conn_guard.transaction()?;


        let label_id = t.query_row("SELECT id FROM labels WHERE label=?", [label], |row| row.get(0)).ok();

        for property_val in params.into_iter() {
            Self::save_param_value(&t, run_id, &property_val.0, property_val.1, label_id)?;
        }

        t.commit()?;

        Ok(())
    }

    fn save_resources(&mut self, resources:Vec<String>) -> Result<(), HbacError>{
        let mut conn_guard = self.conn.lock()?;
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


    fn save_label(&mut self, label:&str) -> Result<usize, HbacError>{
        let mut conn_guard = self.conn.lock()?;
        let  t = conn_guard.transaction()?;

        t.execute("INSERT INTO labels (label) VALUES (?)", params![label])?;

        let label_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;

        t.commit()?;

        Ok(label_id)
    }


    // fn save_param(&mut self, name:&str, data_type:&str) -> Result<usize, HbacError>{
    //     let mut conn_guard = self.conn.lock()?;
    //     let mut t = conn_guard.transaction()?;
    //
    //     t.execute("INSERT INTO params (name, type) VALUES (?, ?)", params![name, data_type])?;
    //
    //     let label_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;
    //
    //     t.commit()?;
    //
    //     Ok(label_id)
    // }



    fn list_labels(&mut self) -> Result<Vec<String>, HbacError> {
        let mut conn_guard = self.conn.lock()?;
        let t = conn_guard.transaction()?;

        let labels = Self::list_labels(&t)?.into_iter().map(|id_label|id_label.1).collect();

        t.commit()?;

        Ok(labels)
    }


    //TODO folds to util fn
    fn save_resources_labels(&mut self, resource_labels_probs:HashMap<String, Vec<(String, f32)>>) -> Result<(), HbacError>{
        debug!("save_resources_labels: resource_labels_probs.len()={}", resource_labels_probs.len());

        fn _do_insert(resource_keys_iter: IntoIter<(String, Vec<(String, f32)>), 500>, t:&Transaction) -> Result<(), HbacError>{

            let (mut insert_resources_q, insert_resources_params, insert_labeled_resources_params) = resource_keys_iter.fold(
                (
                    vec!["INSERT INTO resources (key) VALUES"],
                    vec![],
                    vec![]

                ), |
                    (mut insert_resources_q, mut insert_resources_params, mut insert_labeled_resources_params),
                    (resource_key, labels)
                |{

                    insert_resources_params.push(resource_key.clone());
                    insert_resources_q.push("(?)");
                    insert_resources_q.push(",");


                    let labels_ids = labels.into_iter().map(|(label, p)|
                        (
                            t.query_row("SELECT id FROM labels WHERE label=?", [&label], |row| row.get(0)).expect(format!("label {} not found", label).as_str()),
                            p
                        )
                    ).collect::<Vec<(i64, f32)>>();
                    insert_labeled_resources_params.push(labels_ids);


                    (insert_resources_q, insert_resources_params, insert_labeled_resources_params)
            });

            insert_resources_q.pop();
            insert_resources_q.push(" RETURNING id");

            let q = insert_resources_q.into_iter().collect::<String>();
            let mut stmt = t.prepare(&q)?;

            let mut rows = stmt.query(rusqlite::params_from_iter(insert_resources_params.into_iter()))?;

            let mut resource_ids:Vec<i64> = vec![];
            while let Some(row) = rows.next()? {
                resource_ids.push(row.get_unwrap(0));
            }
            let mut labels_iter = resource_ids.into_iter().zip(insert_labeled_resources_params).flat_map(move |(id, labels)|labels.into_iter().map(move |(l, p)|(id, l, p))).array_chunks::<500>();

            fn _insert_labels(resource_labels: IntoIter<(i64, i64, f32), 500>, t: &Transaction){

                let (mut q, params) = resource_labels.fold((vec!["INSERT INTO labeled_resources (resource_id, label_id, p) VALUES"], vec![]), |(mut q, mut params):(Vec<&str>, Vec<Value>), v|{
                    q.push("(?, ?, ?)");
                    q.push(",");

                    params.push(Value::Integer(v.0));//resource id
                    params.push(Value::Integer(v.1));//label id
                    params.push(Value::Real(v.2 as f64));//p
                    //
                    (q, params)
                });

                if params.len()>0 {
                    q.pop();

                    let q_str = q.into_iter().collect::<String>();

                    t.execute(&q_str, rusqlite::params_from_iter(params.into_iter())).unwrap();
                }
            }

            while let Some(resource_labels) =  labels_iter.next() {
                _insert_labels(resource_labels.into_iter(), t);
            }

            if let Some(reminder) = labels_iter.into_remainder() {
                _insert_labels(reminder, t);
            }


            Ok(())
        }


        let mut conn_guard = self.conn.lock()?;
        let t = conn_guard.transaction()?;



        let mut resource_labels_iter = resource_labels_probs.into_iter().array_chunks::<500>();


        while let Some(resource_keys) =  resource_labels_iter.next() {
            _do_insert(resource_keys.into_iter(), &t)?;
        }

        if let Some(reminder) = resource_labels_iter.into_remainder() {
            if reminder.len() > 0 {
                _do_insert(reminder, &t)?;
            }
        }

        t.commit()?;

        Ok(())
    }

    fn save_resources_set(&mut self, descr:String, resource_keys:Vec<String>, group_id: Option<usize>) -> Result<usize, HbacError>{

        let mut conn_guard = self.conn.lock()?;
        let t = conn_guard.transaction()?;

        match group_id{
            Some(group_id)=>{
                t.execute("INSERT INTO resource_set (descr, group_id) VALUES (?, ?)", params![descr, group_id])?;
            }
            None=>{
                t.execute("INSERT INTO resource_set (descr) VALUES (?)", params![descr])?;
            }
        };


        let resource_set_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get::<usize, usize>(0))?;

        let q = vec!["INSERT INTO resource_resources_set (set_id, resource_id) SELECT ?, id FROM resources WHERE key IN ("];
        let resoure_id_str = resource_set_id.to_string();
        let params = vec![resoure_id_str.as_str()];
        for resources_chunk in resource_keys.chunks(500) {
            let (mut q, params) = resources_chunk.into_iter().fold((q.clone(), params.clone()), |(mut q, mut params), key|{
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

    fn save_resource_set_group(&mut self, descr:String)-> Result<usize, HbacError> {
        let mut conn_guard = self.conn.lock()?;
        let mut t = conn_guard.transaction()?;

        let group_id = Self::save_resource_set_group(descr, &mut t)?;

        t.commit()?;

        Ok(group_id)
    }

    fn save_groups_by_label_cnt(&mut self, max_resources_in_set: usize) -> Result<(), HbacError>{
        let mut conn_guard = self.conn.lock()?;
        let mut t = conn_guard.transaction()?;

        {
            let labels_cnt_labels_resources:HashMap<usize, HashMap<String, Vec<usize>>> = {
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
                    let  labels_resources = cnt_labels.entry(labels).or_insert_with(|| vec![]);

                    labels_resources.push(resource_id);
                }
                labels_cnt_labels_resources
            };

            debug!("save_groups_by_label_cnt: labels_cnt_labels_resources.len()={:?}", labels_cnt_labels_resources.len());
            let mut rng = thread_rng();

            let labels_ids:HashMap<String, usize> = Self::list_labels(&t)?.into_iter().fold(HashMap::new(), |mut a, v|{
                a.insert(v.1, v.0);
                a
            });

            //labels_cnt number of labels a resource has
            for (labels_cnt, labels_resources) in labels_cnt_labels_resources {

                let max_labels_resources = labels_resources.values().map(|r|r.len()).sum::<usize>();
                debug!(
                    "save_groups_by_label_cnt: labels_cnt={} max_labels_resources={}, max_resources_in_set={}, max_labels_resources/max_resources_in_set={}",
                    labels_cnt, max_labels_resources, max_resources_in_set, max_labels_resources/max_resources_in_set
                );

                let num_groups = max(1, max_labels_resources/max_resources_in_set).ilog2();
                let groups_max_resources = (0..num_groups).map(|i|max_resources_in_set * 2usize.pow(i as u32) as usize).collect::<Vec<usize>>();

                let groups_max_ids = groups_max_resources.iter().map(
                    |max_r| {
                        (0..max_labels_resources).step_by(*max_r).map(|i| {
                            Self::save_resource_set_group(format!("group {} labels/{}/{}-{}", labels_cnt, max_r, i, i+*max_r), &mut t).map(|g| (i, i+*max_r, g))
                        }).collect::<Result<Vec<(usize, usize, usize)>, HbacError>>()
                    }
                ).collect::<Result<Vec<Vec<(usize, usize, usize)>>, HbacError>>()?;

                let group_all_id = Self::save_resource_set_group(format!("group {} labels/all", labels_cnt), &mut t)?;

                debug!("save_groups_by_label_cnt:  num_groups={:?}, groups_max_resources={:?}", num_groups, groups_max_resources);

                for (labels, mut resources) in labels_resources {


                    resources.shuffle(&mut rng);

                    let label_id = labels_ids.get(&labels);//if more than one label - label_id is None and Null in the db

                    debug!("save_groups_by_label_cnt: labels={} resources.len()={} label_id={:?}",labels, resources.len(), label_id);

                    if resources.len() > 10 {//TODO min resources

                        for (group_id, _max_resources) in groups_max_ids.iter().zip(groups_max_resources.iter()) {
                            for (resources_0, resources_1, group_id) in group_id {
                                if *resources_0 < resources.len() - 1 {
                                    Self::insert_resource_set_resources(&mut t, labels_cnt, *group_id, format!("{}/{}", labels_cnt, labels),  label_id, &resources[*resources_0..min(resources.len() - 1, *resources_1)])?;
                                }
                            }
                        }
                    }
                    Self::insert_resource_set_resources(&mut t, labels_cnt, group_all_id, format!("{}/{}", labels_cnt, labels), label_id, resources.as_slice())?;
                }
            }
        }

        t.commit()?;

        Ok(())
    }



    fn save_pure_sets(&mut self, max_len: usize, only_unused: bool) -> Result<(), HbacError> {
        let mut conn_guard = self.conn.lock()?;
        let mut t = conn_guard.transaction()?;

        {
            let group_id = Self::save_resource_set_group(format!("pure group {}", max_len), &mut t)?;

            let mut stmt = t.prepare("SELECT id, label FROM labels ORDER BY id")?;//TODO replace with Self::list_labels
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

        t.commit()?;

        Ok(())
    }

    fn list_resource_set_group_sets(&mut self, group_descr:&str)  -> Result<Vec<(usize, String)>, HbacError> {
        let mut guard = self.conn.lock()?;
        let t = guard.transaction()?;

        let group_sets = {
            let mut stmt = t.prepare("SELECT rs.id, rs.descr FROM resource_set rs LEFT JOIN resource_set_group rsg ON rsg.id=rs.group_id WHERE rsg.descr=?")?;


            let vec = stmt.query_map([group_descr], |row| {//TODO move group_slices out, not for store
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

    fn list_group_labeled_resources_sets(&mut self, group_descr:&str) -> Result<Vec<(String, String)>, HbacError> {
        let mut guard = self.conn.lock()?;
        let t = guard.transaction()?;

        let labeled_resources_sets = {
            let mut stmt = t.prepare("SELECT l.label, rs.descr FROM resource_set_group rsg JOIN resource_set rs ON rs.group_id=rsg.id JOIN labels l ON l.id=rs.label_id WHERE rsg.descr=?")?;


            let vec = stmt.query_map([group_descr], |row| {//TODO move group_slices out, not for store
                Ok((
                    row.get::<usize, String>(0).expect("row 0 in list_group_labeled_resources_sets"),
                    row.get::<usize, String>(1).expect("row 1 in list_group_labeled_resources_sets")
                ))
            })?.map(|v| v.unwrap()).collect();

            vec

        };

        t.commit()?;

        Ok(labeled_resources_sets)
    }

    fn list_resource_set_resources_by_descr(&mut self, descr:&str)  -> Result<Vec<String>, HbacError> {
        let mut guard = self.conn.lock()?;
        let t = guard.transaction()?;

        let r = Self::list_resource_set_resources(t.query_row("SELECT id FROM resource_set WHERE descr=?", [descr], |row| row.get(0))?, &t)?;

        t.commit()?;

        Ok(r)
    }

    fn list_resource_random_not_label(&mut self, label:&str, limit:i64)  -> Result<Vec<String>, HbacError> {
        let mut guard = self.conn.lock()?;
        let t = guard.transaction()?;

        let r = t.prepare(
            "SELECT r.key FROM resources r INNER JOIN labeled_resources lr ON r.id=lr.resource_id INNER JOIN labels l ON l.id=lr.label_id WHERE l.label != ? ORDER BY RANDOM() limit ?;"
        )?.query_map([Value::Text(label.to_string()), Value::Integer(limit)], |row| {//TODO move group_slices out, not for store
            Ok(
                row.get(0)?
            )
        })?.map(|v| v.unwrap()).collect();

        t.commit()?;

        Ok(r)
    }

    fn list_resource_random_in_group(&mut self, group_descr:&str)  -> Result<Vec<(String, Vec<f32>)>, HbacError> {//vec<(key, vec<(lablel, p)>>
        let mut guard = self.conn.lock()?;
        let t = guard.transaction()?;

        let labels = Self::list_labels(&t)?;

        debug!("list_resource_random_in_group: group_descr={:?} labels={:?}", group_descr, labels);

        let r:Vec<(String, Vec<f32>)> = t.prepare(
            "SELECT r.key, GROUP_CONCAT(l.label || '/' || lr.p, ',') FROM resource_set_group rsg \
            JOIN resource_set rs ON rs.group_id=rsg.id \
            JOIN resource_resources_set rrs ON rrs.set_id=rs.id \
            JOIN resources r ON r.id=rrs.resource_id \
            JOIN labeled_resources lr ON lr.resource_id=r.id \
            JOIN labels l ON l.id=lr.label_id \
            WHERE rsg.descr=? GROUP BY r.key ORDER BY RANDOM()"
        )?.query_map([Value::Text(group_descr.to_string())], |row| {
            let key = row.get::<usize, String>(0)?;
            let labels_p = row.get::<usize, String>(1)?;

            let labels_p_vec =labels_p.split(",")
                .map(|label_p| {
                        let mut label_p_v = label_p.split("/");
                        (
                            label_p_v.next().expect("label").into(),
                            label_p_v.next().expect("p").parse::<f32>().expect("parse")
                        )

                    }
                ).fold( HashMap::new(), |mut a:HashMap<String, f32>, v|{
                    a.insert(v.0, v.1);
                    a
                });

            trace!("list_resource_random_in_group: key={} labels_p={} labels_p_vec={:?}", key, labels_p, labels_p_vec);

            Ok(
                (key, labels.iter().map(|(_label_id, label)|*labels_p_vec.get(label).unwrap_or(&0.0)).collect::<Vec<f32>>())
            )
        })?.map(|v| v.unwrap()).collect();

        t.commit()?;

        debug!("list_resource_random_in_group: r.len={:?}", r.len());

        Ok(r)
    }

    fn list_resource_set_resources(&mut self, set_id:usize)  -> Result<Vec<String>, HbacError> {
        let mut guard = self.conn.lock()?;
        let t = guard.transaction()?;

        let set_resources = Self::list_resource_set_resources(set_id, &t)?;

        t.commit()?;

        Ok(set_resources)
    }

    fn save_set_by_labels(&mut self, conds:Vec<(String, SetCond)>, _only_unused:bool, limit:usize) -> Result<(), HbacError> {
        let mut conn_guard = self.conn.lock()?;
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
                HbacError::new(format!("no resources for set on {:?}", conds));
            }
        }
        Ok(t.commit()?)
    }


    fn save_slices(&mut self, run_id:usize, channel:i64, slices:Vec<((String, i64), (String, i64), f32)>) -> Result<(), HbacError>{
        let mut conn_guard = self.conn.lock()?;
        let  t = conn_guard.transaction()?;

        assert!(slices.len() > 0);

        for slices_chunk in slices.chunks(500) {
            let (mut q, params) = slices_chunk.into_iter().fold(
                (vec!["INSERT INTO slice_matches (run_id, channel_id, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1, dot) VALUES"], vec![]), |mut a: (Vec<&str>, Vec<Value>), v| {
                    a.0.push("(?, ?, ?, ?, ?, ?, ?)");
                    a.0.push(", ");

                    a.1.push(Value::Integer(run_id as i64));
                    a.1.push(Value::Integer(channel as i64));
                    a.1.push(Value::Text(v.0.0.clone()));
                    a.1.push(Value::Integer(v.0.1));
                    a.1.push(Value::Text(v.1.0.clone()));
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

    fn set_motifs(&mut self, run_id:usize, channel:i64) -> Result<(), HbacError> {
        let mut guard = self.conn.lock()?;
        let t = guard.transaction()?;

        debug!("set_motifs: run_id={:?}, channel={:?}", run_id, channel);

        {
            let mut stmt = t.prepare("SELECT id, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1 FROM slice_matches WHERE run_id=? AND channel_id=?")?;

            let groups = group_slices(
                stmt.query_map([run_id, channel as usize], |row| {//TODO move group_slices out, not for store
                    Ok(
                        (
                            IdSlice(row.get(0)?, row.get(1)?, row.get(2)?),
                            IdSlice(row.get(0)?, row.get(3)?, row.get(4)?)
                        )
                    )
                })?.map(|v| v.unwrap())
            );

            debug!("set_motifs: run_id={:?}, channel={:?} groups={:?}", run_id, channel, groups);

            //TODO sort?
            for (group, slices) in groups {
                debug!("set_motifs: group={:?}, slices={:?} run_id={:?}", group, slices, run_id);

                t.execute("INSERT INTO motifs (group_ord, run_id, channel_id) VALUES (?, ?, ?)", [group as usize, run_id, channel as usize])?;//TODO group not needed?

                let group_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get(0))?;

                let (mut q, params) = slices.into_iter().fold(
                    (vec!["UPDATE slice_matches SET motif_id=? WHERE id IN ("], vec![Value::Integer(group_id)]),
                    |mut a: (Vec<&str>, Vec<Value>), v:IdSlice| {
                        a.0.push("?");
                        a.0.push(", ");
                        a.1.push(Value::Integer(v.0 as i64));

                        a
                    });

                q.pop();
                q.push(")");

                t.execute(&q.into_iter().collect::<String>(), rusqlite::params_from_iter(params.into_iter()))?;
            }

        }

        t.commit()?;

        Ok(())
    }

    fn list_motifs_distances(&mut self, run_id:usize) -> Result<HashMap<usize, f32>, HbacError>{

        let mut guard = self.conn.lock()?;
        let t = guard.transaction()?;
        let tensor_motifs: HashMap<usize, f32> = {

            let mut stmt = t.prepare("SELECT motif_id, avg(dot) FROM slice_matches GROUP BY motif_id HAVING run_id=?")?;

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

    fn delete_slices_without_matches(&mut self, run_id:usize) -> Result<(), HbacError>{
        let mut guard = self.conn.lock()?;
        let t = guard.transaction()?;
        t.execute("DELETE FROM slice_matches WHERE run_id=? AND motif_id IS NULL", [run_id])?;
        t.commit()?;

        guard.execute("VACUUM", [])?;

        Ok(())
    }

    /**
    * returns map tensor_key - MotifSlice(motif_group, tensor slice idx, channel)
    */
    fn list_motifs_slices(&mut self, run_id:usize) -> Result<HashMap<String, Vec<MotifSlice>>, HbacError>{
        let mut guard = self.conn.lock()?;
        let t = guard.transaction()?;

        let tensor_motifs = {
            let mut stmt = t.prepare("SELECT id, motif_id, channel_id, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1 FROM slice_matches WHERE run_id=?")?;

            let tensor_motifs: HashMap<String, Vec<MotifSlice>> = stmt.query_map([run_id], |row| {
                Ok((
                    row.get(0)?, // id,
                    row.get::<usize, usize>(1).ok(), // motif_id,
                    row.get(2)?, // channel_id,
                    row.get(3)?, // tensor_id_0,
                    row.get(4)?, // slice_idx_0,
                    row.get(5)?, // tensor_id_1,
                    row.get(6)?, // slice_idx_1
                ))
            })?.fold(Ok(HashMap::new()), |a: Result<HashMap<String, Vec<hbac_shared::MotifSlice>>, rusqlite::Error>, v: Result<(i64, Option<usize>, i64, String, i64, String, i64), rusqlite::Error>| {
                let v = v?;
                if let Some(motif_id) = v.1 {
                    let mut a = a?;

                    a.entry(v.3).or_insert_with(|| vec![]).push(MotifSlice::new(motif_id, v.4, v.2 as u16));
                    a.entry(v.5).or_insert_with(|| vec![]).push(MotifSlice::new(motif_id, v.6, v.2 as u16));

                    Ok(a)
                } else {
                    a
                }
            })?;

            if tensor_motifs.len() < 10 {
                debug!("motifs_to_eeg_slices: eeg_motifs={:?}", tensor_motifs);
            }

            tensor_motifs
        };
        t.commit()?;

        Ok(tensor_motifs)
    }

    fn list_run_resources(&mut self, run_id:usize) -> Result<Vec<String>, HbacError> {
        let mut guard = self.conn.lock()?;
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
                }).map_err(|e| HbacError::new(e.to_string()))
            }).collect::<Result<Vec<String>, HbacError>>();

            result
        };

        t.commit()?;
        debug!("list_run_resources: result.is_ok={}", result.is_ok());

        result
    }

    fn list_run_params(&mut self, run_id:usize, _label: Option<&str>) -> Result<HashMap<String, Val>, HbacError> {

        let mut guard = self.conn.lock()?;
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
            })?.map(|v|v.expect(format!("can't get run_param_values for {}", run_id).as_str())).fold(HashMap::new(), |mut a, (name, real_value, int_value, str_value)|{
                a.insert(
                    name.clone(),
                    real_value.map(|v|v.into())
                    .or(int_value.map(|v|v.into()))
                    .or(str_value.map(|v|v.into()))
                    .expect(format!("no value for {}", name).as_str())
                );
                a
            });

            result
        };

        t.commit()?;
        debug!("list_run_resources: result={:?}", result);

        Ok(result)

    }


    fn save_motifs_corrs(&mut self, run_id: usize, descr:&str, motifs_corrs:Vec<Vec<(usize, f64)>>, label: String) -> Result<usize, HbacError>{
        debug!("save_motifs_corrs: motifs_corrs={:?}", motifs_corrs);

        let mut guard = self.conn.lock()?;
        let t = guard.transaction()?;

        t.execute("INSERT INTO corrs_run (run_id, label, descr) VALUES (?, ?, ?)", params![run_id, label, descr])?;
        let corr_run_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get::<usize, usize>(0))?;

        let motifs_corrs = motifs_corrs.into_iter().filter(|corrs|corrs.len() > 0).collect::<Vec<Vec<(usize, f64)>>>();
        debug!("save_motifs_corrs: filtered motifs_corrs={:?}", motifs_corrs);

        if motifs_corrs.len() > 0 {

            for motifs_corrs_chunk in motifs_corrs.chunks(500) {

                let (mut q, params) = motifs_corrs_chunk.into_iter().flat_map(|channel_motif_corrs| channel_motif_corrs.into_iter()).into_iter().fold(
                    (vec!["INSERT INTO motifs_corrs (corr_id, motif_id, corr) VALUES "], vec![]), //TODO check if channel matches one in motif
                    |mut a: (Vec<&str>, Vec<Value>), (motif_id, corr)| {
                        a.0.push("(?, ?, ?)");
                        a.0.push(", ");
                        a.1.push(Value::Integer(corr_run_id as i64));
                        a.1.push(Value::Integer(*motif_id as i64));
                        a.1.push(Value::Real(*corr));

                        a
                    });

                q.pop();


                t.execute(&q.into_iter().collect::<String>(), rusqlite::params_from_iter(params.into_iter()))?;
            }


        }
        t.commit()?;

        Ok(corr_run_id)
    }

    fn list_corrs(&mut self, ids:&Vec<usize>) -> Result<Vec<(usize, usize, String)>, HbacError> {
        let mut guard = self.conn.lock()?;
        let t = guard.transaction()?;


        let (mut q, params) = ids.into_iter().fold((vec!["SELECT id, run_id, label FROM corrs_run WHERE id IN ("], vec![]), |mut a, v|{
            a.0.push("?");
            a.0.push(",");
            a.1.push(*v);
            a
        });

        q.pop();
        q.push(")");

        let corrs = {
            let mut stmt = t.prepare(q.into_iter().collect::<String>().as_str())?;

            let r= stmt.query_map(rusqlite::params_from_iter(params.into_iter()), |row| {
                Ok((
                    row.get::<usize, usize>(0)?,
                    row.get::<usize, usize>(1)?,
                    row.get::<usize, String>(2)?
                ))
            })?.map(|v| v.unwrap()).collect();

            r
        };
        t.commit()?;


        Ok(corrs)
    }
    /**
    * returns vec of channel X (motif id, corr)
    */
    fn list_corr_motifs(&mut self, corr_run_id:usize) -> Result<Vec<Vec<(usize, f64)>>, HbacError> {
        let mut channels_corr_motifs = (0..20).into_iter().map(|_|Vec::with_capacity(10)).collect::<Vec<Vec<(usize, f64)>>>();

        let mut guard = self.conn.lock()?;
        let t = guard.transaction()?;

        {
            let mut stmt = t.prepare("SELECT mc.motif_id, mc.corr, m.channel_id FROM motifs_corrs mc LEFT JOIN motifs AS m ON m.id=mc.motif_id WHERE mc.corr_id=?")?;

            stmt.query_map([corr_run_id], |row| {
                Ok((row.get::<usize, usize>(0)?, row.get::<usize, f32>(1)?, row.get::<usize, usize>(2)?))

            })?.map(|v|v.unwrap()).for_each(|(motif_id, corr, channel_id)| {

                channels_corr_motifs[channel_id].push((motif_id, corr as f64))
            });

        }
        t.commit()?;

        Ok(channels_corr_motifs)
    }

    fn list_motif_slices(&mut self, motif_id:usize, limit: Option<usize>, all_but_in_run_id: Option<(usize, usize)>) -> Result<(i64, usize, Vec<TensorSliceIdx<String>>), HbacError> {

        let mut guard = self.conn.lock()?;
        let t = guard.transaction()?;

        let channel_slices = {

            let mut stmt = t.prepare(&[
                "SELECT channel_id, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1, run_id FROM slice_matches where motif_id",
                // if all_but_in_run_id.is_none() {"=? AND ? IS NOT NULL AND ? IS NOT NULL "} else {"!=? AND run_id=? AND channel_id=?"}, run_id? //TODO
                if all_but_in_run_id.is_none() {"=? AND ? IS NOT NULL "} else {"!=? AND channel_id=?"},
                &limit.map(|limit| format!("LIMIT {}", limit)).unwrap_or("".to_owned())
            ].join(""))?;

            let params = match all_but_in_run_id {
                Some((_run_id, channel_id))=>[motif_id, channel_id],
                None =>[motif_id, 0],
            };

            let (slices, channel_set, run_id) = stmt.query_map(params, |row| {
                Ok((
                    row.get::<usize, i64>(0)?,//channel_id
                    row.get::<usize, String>(1)?,//tensor_id_0
                    row.get::<usize, i64>(2)?,//slice_idx_0
                    row.get::<usize, String>(3)?,//tensor_id_1
                    row.get::<usize, i64>(4)?,//slice_idx_1
                    row.get::<usize, usize>(5)?,//run_id
                ))
            })?.map(|v| v.unwrap()).fold((HashSet::new(), HashSet::new(), None), |(mut a, mut channel_set, mut run_id), v| {
                a.insert(TensorSliceIdx(v.1, v.2 as i64));
                a.insert(TensorSliceIdx(v.3, v.4 as i64));

                channel_set.insert(v.0);//TODO replace with Option

                if all_but_in_run_id.is_none() {
                    if let Some(run_id) = run_id {
                        assert!(run_id == v.5);
                    } else {
                        run_id.replace(v.5);
                    }
                }

                (a, channel_set, run_id)
            });

            if all_but_in_run_id.is_none() {
                assert!(channel_set.len() == 1);
                (channel_set.into_iter().nth(0).expect("channel"), run_id.expect("no run id"), slices.into_iter().collect())
            } else {
                let (run_id, channel_id) = all_but_in_run_id.unwrap();
                (channel_id as i64, run_id, slices.into_iter().collect()) //TODO
            }
        };

        t.commit()?;

        Ok(channel_slices)
    }

    fn corrs_descr(&mut self, corr_run_id:usize) -> Result<Vec<(i64, String)>, HbacError> {
        let mut conn_guard = self.conn.lock()?;
        let t = conn_guard.transaction()?;

        let descr:String = t.query_row("SELECT descr FROM corrs_run WHERE id=?", [corr_run_id], |row| row.get(0))?;

        debug!("descr={:?}", descr);

        parse_corr_labels(descr.as_str())

    }
}


impl SqlStore {

    pub fn from_path<P:AsRef<Path>>(path:P) -> Result<Self, HbacError> {
        let mut conn = Connection::open(path)?;

        Self::new_(conn)?
    }

    // #[cfg(test)]
    pub fn in_mem() -> Result<Self, HbacError> {
        let mut conn = Connection::open_in_memory()?;

        Self::new_(conn)?
    }


    fn new_(mut conn: Connection) -> Result<Result<SqlStore, HbacError>, HbacError> {
        {
            let t = conn.transaction()?;

            t.execute( //TODO UNIQUE(key) doesn't work - no error
                       "CREATE TABLE IF NOT EXISTS resources (
                id          INTEGER PRIMARY KEY,
                key         VARCHAR(256),
                UNIQUE(key) )",
                       ())?;

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

            //descr - dot, dist etc
            t.execute(
                "CREATE TABLE IF NOT EXISTS tree_run (
                id              INTEGER PRIMARY KEY,
                input_run_id    REFERENCES tree_run(id) ON DELETE CASCADE,
                descr           VARCHAR(256),
                run_on          DATETIME DEFAULT CURRENT_TIMESTAMP
            )", ())?;


            //key - resource id  //TODO remove ord_id?
            t.execute(
                "CREATE TABLE IF NOT EXISTS run_resources (
                run_id     NOT NULL REFERENCES tree_run(id) ON DELETE CASCADE,
                ord_id     INTEGER NOT NULL,
                key        VARCHAR(256),
                PRIMARY KEY (run_id, ord_id)
            )", ())?;

            // t.execute(
            //     "CREATE TABLE IF NOT EXISTS params (
            //     id          INTEGER PRIMARY KEY,
            //     name        VARCHAR(256),
            //     type        VARCHAR(256),
            //     UNIQUE(name) )",
            //     (),
            // )?;

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

            // t.execute(
            //     "CREATE TABLE IF NOT EXISTS runs_run_param_values (
            //     id          INTEGER PRIMARY KEY,
            //     value_id    NOT NULL REFERENCES run_param_values(id) ON DELETE CASCADE,
            //     run_id      NOT NULL REFERENCES tree_run(id) ON DELETE CASCADE
            // )",
            // (),
            // )?;

            t.execute( //group - ordinal as appeared when grouping, not needed
                       "CREATE TABLE IF NOT EXISTS motifs (
                id              INTEGER PRIMARY KEY,
                group_ord       INTEGER NOT NULL,
                run_id          NOT NULL REFERENCES tree_run(id) ON DELETE CASCADE,
                channel_id      INTEGER NOT NULL
            )", ())?;

            //dot - similarity
            //TODO!! tensor_id_0 - REFERENCES resource_id
            t.execute(
                "CREATE TABLE IF NOT EXISTS slice_matches (
                id              INTEGER PRIMARY KEY,
                run_id          NOT NULL REFERENCES tree_run(id) ON DELETE CASCADE,
                channel_id      INTEGER NOT NULL,
                tensor_id_0     VARCHAR(256) NOT NULL,
                slice_idx_0     INTEGER NOT NULL,
                tensor_id_1     VARCHAR(256) NOT NULL,
                slice_idx_1     INTEGER NOT NULL,
                dot             FLOAT NOT NULL,
                motif_id        REFERENCES motifs(id) ON DELETE CASCADE DEFAULT NULL
            )", ())?;

            //resource_set_id-
            t.execute(
                "CREATE TABLE IF NOT EXISTS corrs_run (
                id          INTEGER PRIMARY KEY,
                run_id      NOT NULL REFERENCES tree_run(id) ON DELETE CASCADE,
                descr       VARCHAR(256),
                label       VARCHAR(256),
                run_on      DATETIME DEFAULT CURRENT_TIMESTAMP
            )", ())?;

            //labels_motifs on resource_set
            t.execute( //todo label
                       "CREATE TABLE IF NOT EXISTS motifs_corrs (
                motif_id   REFERENCES motifs(id),
                corr_id    NOT NULL REFERENCES corrs_run(id) ON DELETE CASCADE,
                corr       FLOAT NOT NULL
            )", ())?;

            t.execute("commit", ())?;
        }

        Ok(Ok(Self {
            conn: Mutex::new(conn)
        }))
    }

    fn list_resource_set_resources(set_id: usize, t: &Transaction) -> Result<Vec<String>, HbacError> {
        let set_resources = {
            let mut stmt = t.prepare("SELECT key FROM resource_resources_set rrs LEFT JOIN resources rs ON rs.id=rrs.resource_id WHERE rrs.set_id=?")?;


            let vec = stmt.query_map([set_id], |row| {
                Ok(
                    row.get::<usize, String>(0).expect("row 1 in list_resource_set_group_sets")
                )
            })?.map(|v| v.unwrap()).collect();

            vec
        };
        Ok(set_resources)
    }


    /**
    defines global label order
    */
    fn list_labels(t: &Transaction) -> Result<Vec<(usize, String)>, HbacError> {
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

    fn save_param_value(t: &Transaction, run_id:usize, param_name:&str, value:Val, label_id:Option<usize>) -> Result<(), HbacError>{

        match value {
            Val::Real(v)=>{
                t.execute("INSERT INTO run_param_values (run_id, label_id, name, real_value) VALUES (?, ?, ?, ?)", params![run_id, label_id, param_name, v])?;
            }
            Val::Int(v)=>{
                t.execute("INSERT INTO run_param_values (run_id, label_id, name, int_value) VALUES (?, ?, ?, ?)", params![run_id, label_id, param_name, v])?;

            }
            Val::Str(v)=>{
                t.execute("INSERT INTO run_param_values (run_id, label_id, name, str_value) VALUES (?, ?, ?, ?)", params![run_id, label_id, param_name, v])?;
            }
        }

        Ok(())
    }


    fn save_resource_set_group(descr: String, t: &mut Transaction) -> Result<usize, HbacError> {

        t.execute("INSERT INTO resource_set_group (descr) VALUES (?)", params![descr])?;
        let group_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get::<usize, usize>(0))?;
        Ok(group_id)
    }

    /**
    * label_id - for pure sets
    */
    fn insert_resource_set_resources(t: &mut Transaction, _labels_cnt: usize, group_id: usize, descr:String, label_id:Option<&usize>, resources: &[usize]) -> Result<(), HbacError> {

         t.execute(
            "INSERT INTO resource_set (descr, group_id, label_id) VALUES (?, ?, ?)",
            params![descr, group_id, label_id]//labels?
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

    fn shut_down(self)-> Result<(), HbacError> {
        // let guard = self.conn.lock()?;
        //
        // guard.cache_flush()?;
        //
        //TODO
        // guard.close().unwrap();
        //
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



#[cfg(test)]
mod store_tests{
    use std::collections::HashMap;
    use std::fs;
    use std::time::Instant;

    use log::debug;
    use log::LevelFilter;
    use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger};

    use hbac_shared::TensorSliceIdx;
    use hbac_shared::val::Val;

    use crate::{HbacError, SetCond, SqlStore, Store};
    use hbac_shared::{cdt_for_dir};

    #[ctor::ctor]
    fn init() {
        hbac_shared::init_test_logging();
    }

    #[test]
    fn test_group_slices() -> Result<(), HbacError> {
        // fs::remove_file("/tmp/test_db.db3").unwrap_or(());
        // let mut store = SqlStore::from_path("/tmp/test_db.db3")?;

        let mut store = SqlStore::in_mem()?;

        let _aaa_id = store.save_label("aaa");
        let _bbb_id = store.save_label("bbb");
        let _ccc_id = store.save_label("ccc");

        //pure resources
        let n = 100;
        store.save_resources_labels((0..n).fold(HashMap::new(), |mut a:HashMap<String, Vec<(String, f32)>>, i|{
            a.insert(format!("{:3}_aaa", i), vec![("aaa".into(), 1.0)]);
            a
        }))?;

        store.save_resources_labels((0..n).fold(HashMap::new(), |mut a:HashMap<String, Vec<(String, f32)>>, i|{
            a.insert(format!("{:3}_bbb", i), vec![("bbb".into(), 1.0)]);
            a
        }))?;

        store.save_resources_labels((0..n).fold(HashMap::new(), |mut a:HashMap<String, Vec<(String, f32)>>, i|{
            a.insert(format!("{:3}_ccc", i), vec![("ccc".into(), 1.0)]);
            a
        }))?;


        //two labels resources
        store.save_resources_labels((0..n).fold(HashMap::new(), |mut a:HashMap<String, Vec<(String, f32)>>, i|{
            a.insert(format!("{:3}_ab", i), vec![("aaa".into(), 0.5), ("bbb".into(), 0.5)]);
            a
        }))?;

        store.save_resources_labels((0..n).fold(HashMap::new(), |mut a:HashMap<String, Vec<(String, f32)>>, i|{
            a.insert(format!("{:3}_bc", i), vec![("bbb".into(), 0.5), ("ccc".into(), 0.5)]);
            a
        }))?;

        store.save_resources_labels((0..n).fold(HashMap::new(), |mut a:HashMap<String, Vec<(String, f32)>>, i|{
            a.insert(format!("{:3}_ca", i), vec![("ccc".into(), 0.5), ("aaa".into(), 0.5)]);
            a
        }))?;

        //three labels resources
        store.save_resources_labels((0..n).fold(HashMap::new(), |mut a:HashMap<String, Vec<(String, f32)>>, i|{
            a.insert(format!("{:3}_abc", i), vec![("aaa".into(), 0.3), ("bbb".into(), 0.3), ("ccc".into(), 0.4)]);
            a
        }))?;

        let t1 = Instant::now();
        store.save_groups_by_label_cnt(10)?;

        debug!("done in {}", t1.elapsed().as_millis());

        // fs::remove_file(db_file)?;
        Ok(())
    }

    #[test]
    fn test_save_slices() -> Result<(), HbacError> {

        // fs::remove_file("/tmp/test_db.db3").unwrap_or(());
        // let mut store = SqlStore::from_path("/tmp/test_db.db3")?;
        let mut store = SqlStore::in_mem()?;

        let _aaa_id = store.save_label("aaa")?;
        let _bbb_id = store.save_label("bbb")?;
        let _ccc_id = store.save_label("ccc")?;

        store.save_resources_labels(HashMap::from([
            ("1111".into(), vec![("aaa".into(), 0.5), ("bbb".into(), 0.5)]),
            ("2222".into(), vec![("aaa".into(), 1.0)]),
            ("3333".into(), vec![("ccc".into(), 1.0)]),
        ]))?;

        let _set_1_id = store.save_resources_set("Set1".to_string(), vec!["1111".into(), "2222".into()], None)?;
        let _set_2_id = store.save_resources_set("Set2".to_string(), vec!["2222".into(), "3333".into()], None)?;

        store.save_pure_sets(10, false)?;
        store.save_set_by_labels(vec![("aaa".into(), SetCond::Gt(0.9))], true, 10)?;

        let all_group_id = store.save_resource_set_group("group 1 labels/100/100-200".into())?;
        let _set_all_id = store.save_resources_set("All".to_string(), vec!["1111".into(), "2222".into(), "3333".into()], Some(all_group_id))?;




        let group_sets = store.list_resource_set_group_sets("pure group 10")?;
        assert_eq!(3, group_sets.len());


        let mut keys_labels_p = store.list_resource_random_in_group("group 1 labels/100/100-200")?;

        keys_labels_p.sort_by(|a, b|a.0.cmp(&b.0));

        debug!("keys_labels_p={:?}", keys_labels_p);

        assert_eq!(
            vec![
                ("1111".to_string(), vec![0.5, 0.5, 0.0]),
                ("2222".to_string(), vec![1.0, 0.0, 0.0]),
                ("3333".to_string(), vec![0.0, 0.0, 1.0]),
            ],
            keys_labels_p
        );


        let map :Vec<((String, i64),(String, i64),f32)> = vec![
            (("0".to_string(), 100i64), ("0".to_string(), 200i64), 0.91),
            (("0".to_string(), 100i64), ("1".to_string(), 200i64), 0.92),
            (("3".to_string(), 100i64), ("3".to_string(), 200i64), 0.93),
            (("3".to_string(), 200i64), ("5".to_string(), 200i64), 0.94),
            (("3".to_string(), 200i64), ("5".to_string(), 300i64), 0.94),
        ];

        let run_id= store.save_tree_run(
            None,
            "test", &vec!["aaa".into(), "bbb".into()],
            HashMap::from([("prop1".into(), Val::Int(1)), ("prop2".into(), Val::Real(2.0)), ("prop3".into(), Val::Str("333".to_string()))])
        )?;

        store.save_params_value(
            HashMap::from([("label.prop1".into(), Val::Int(10)), ("label.prop2".into(), Val::Real(20.0)), ("label.prop3".into(), Val::Str("3330".to_string()))]),
            run_id,
            Some("aaa")
        )?;

        store.save_params_value(
            HashMap::from([("label.prop1".into(), Val::Int(1)), ("label.prop2".into(), Val::Real(2.0)), ("label.prop3".into(), Val::Str("333".to_string()))]),
            run_id,
            Some("bbb")
        )?;

        store.save_params_value(
            HashMap::from([("label.prop1".into(), Val::Int(1)), ("label.prop2".into(), Val::Real(2.0)), ("label.prop3".into(), Val::Str("333".to_string()))]),
            run_id,
            None
        )?;

        let loaded_params = store.list_run_params(run_id, None)?;

        debug!("loaded_params={:?}", loaded_params);


        assert_eq!(loaded_params.get("prop1").expect("prop1"), &Val::Int(1));
        assert_eq!(loaded_params.get("prop2").expect("prop2"), &Val::Real(2.0));
        assert_eq!(loaded_params.get("prop3").expect("prop3"), &Val::Str("333".to_string()));

        let loaded_params_aaa = store.list_run_params(run_id, Some("aaa"))?;

        debug!("loaded_params_aaa={:?}", loaded_params_aaa);

        store.save_slices(run_id,  7, map)?;

        store.set_motifs(run_id, 7)?;

        let m_1_first = {
            let mut guard = store.conn.lock()?;
            let t2 = guard.transaction()?;
            let mut stmt = t2.prepare("SELECT run_id, channel_id, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1, motif_id FROM slice_matches")?;

            let rows = stmt.query_map([], |row| {
                Ok((
                    row.get(0)?,//run_id
                    row.get(1)?,//channel_id
                    row.get(2)?,//tensor_id_0
                    row.get(3)?,//slice_idx_0
                    row.get(4)?,//tensor_id_1
                    row.get(5)?,//slice_idx_1
                    row.get(6)? //motif_id
                ))
            })?.map(|v|v.unwrap()).collect::<Vec<(i64, i64, String, i64, String, i64, i64)>>();

            debug!("rows={:?}", rows);

            let m_1_first = vec![
                (1, 7, "0".to_string(), 100, "0".to_string(), 200, 1),
                (1, 7, "0".to_string(), 100, "1".to_string(), 200, 1),
                (1, 7, "3".to_string(), 100, "3".to_string(), 200, 2),
                (1, 7, "3".to_string(), 200, "5".to_string(), 200, 2),
                (1, 7, "3".to_string(), 200, "5".to_string(), 300, 2)
            ] == rows;

            let m_2_first = vec![
                (1, 7, "0".to_string(), 100, "0".to_string(), 200, 2),
                (1, 7, "0".to_string(), 100, "1".to_string(), 200, 2),
                (1, 7, "3".to_string(), 100, "3".to_string(), 200, 1),
                (1, 7, "3".to_string(), 200, "5".to_string(), 200, 1),
                (1, 7, "3".to_string(), 200, "5".to_string(), 300, 1)
            ] == rows;

            assert!(m_2_first  || m_1_first  );                        //TODO motif ids are random

            let mut stmt = t2.prepare("SELECT run_id, ord_id, key FROM run_resources")?;

            let rows = stmt.query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?,  row.get(2)?))
            })?.map(|v|v.unwrap()).collect::<Vec<(usize, usize, String)>>();


            assert_eq!(vec![(1, 0, "aaa".to_string()), (1, 1, "bbb".to_string())], rows);

            m_1_first
        };

        let map = store.list_motifs_slices(run_id)?;
        debug!("map={:?}", map);

        assert_eq!(4, map.len());
        assert_eq!(3, map.get(&"0".to_string()).unwrap().len());
        assert_eq!(1, map.get(&"1".to_string()).unwrap().len());
        assert_eq!(4, map.get(&"3".to_string()).unwrap().len());
        assert_eq!(2, map.get(&"5".to_string()).unwrap().len());

        assert_eq!(vec!["aaa".to_owned(), "bbb".to_owned()], store.list_run_resources(run_id)?);

        let corr_id = store.save_motifs_corrs(
            run_id,
            "[(1, \"seizure_vote\"), (0, \"lpd_vote\"), (0, \"gpd_vote\"), (0, \"lrda_vote\"), (0, \"grda_vote\"), (0, \"other_vote\")]",
            vec![vec![(1, 0.1), (2, 0.2)]],
            "seizure_vote".into()
        )?;

        let corrs = store.list_corrs(&vec![corr_id])?;
        assert_eq!([(1, 1, "seizure_vote".to_string())], corrs.as_slice());

        let motifs = store.list_corr_motifs(corr_id)?;

        debug!("motifs: {:?}", motifs);

        assert_eq!(1, motifs[7][0].0);
        assert_eq!(2, motifs[7][1].0);

        let (channel, run_id_, mut slices) = store.list_motif_slices(1, None, None)?;
        slices.sort();

        if m_1_first {
            assert_eq!(vec![
                TensorSliceIdx("0".to_string(), 100), TensorSliceIdx("0".to_string(), 200), TensorSliceIdx("1".to_string(), 200)
            ], slices);
        } else {
            assert_eq!(vec![
                TensorSliceIdx("3".to_string(), 100), TensorSliceIdx("3".to_string(), 200), TensorSliceIdx("5".to_string(), 200), TensorSliceIdx("5".to_string(), 300)
            ], slices);
        }

        assert_eq!(7, channel);
        assert_eq!(run_id, run_id_);

        let (_channel, _run_id_, mut slices) = store.list_motif_slices(1, Some(10), Some((run_id as usize, channel as usize)))?;

        slices.sort();

        if !m_1_first {
            assert_eq!(vec![
                TensorSliceIdx("0".to_string(), 100), TensorSliceIdx("0".to_string(), 200), TensorSliceIdx("1".to_string(), 200)
            ], slices);
        } else {
            assert_eq!(vec![
                TensorSliceIdx("3".to_string(), 100), TensorSliceIdx("3".to_string(), 200), TensorSliceIdx("5".to_string(), 200), TensorSliceIdx("5".to_string(), 300)
            ], slices);
        }

        // debug!("corrs_descr={:?}", store.corrs_descr(corr_id)?);

        assert_eq!(
            vec![(1, "seizure_vote".into()), (0, "lpd_vote".into()), (0, "gpd_vote".into()), (0, "lrda_vote".into()), (0, "grda_vote".into()), (0, "other_vote".into())],
            store.corrs_descr(corr_id)?
        );

        store.delete_slices_without_matches(run_id)?;
        store.shut_down();
        // fs::remove_file(db_file)?;
        Ok(())
    }

    #[test]
    fn test_conv_cond(){
        let input = vec![("aaa".into(), SetCond::Lt(0.4)), ("bbb".into(), SetCond::Gt(0.1))];

        let conds = super::conds_to_q(&input);
        assert_eq!("(l.label='aaa' and p<?) or (l.label='bbb' and p>?)", conds);

    }

}
