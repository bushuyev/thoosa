extern crate core;

use std::backtrace::Backtrace;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::path::Path;
use rusqlite::{Connection, params, ToSql};
use log::debug;
use rusqlite::types::{ToSqlOutput, Value};
use rusqlite::types::ValueRef::Real;
use std::fmt;
use std::sync::MutexGuard;
use std::sync::{Arc, Mutex, PoisonError};
use hbac_shared::*;
use std::hash::{DefaultHasher, Hash, Hasher};



pub struct StoreDumper {
    pub store:Arc<Mutex<dyn Store>>, //TODO pub
    pub run_id:usize,
    pub name:String,
    pub channel:usize
}

impl StoreDumper {
    pub fn new(store: Arc<Mutex<impl Store + 'static>>, run_id:usize, name:String, channel:usize) -> Self {
        Self {
            run_id,
            store,
            name,
            channel
        }
    }
}

#[derive(Debug)]
struct IdSlice(i64, i64, i64); //id, tensor id, slice idx

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

pub trait Store:Sync + Send{
    fn save_run(&mut self, descr:&str, keys:&Vec<String>) -> Result<usize, HbacError>;
    fn save_slices(&mut self, run_id:usize, channel:i64, slices:Vec<((usize, i64), (usize, i64), f32)>) -> Result<(), HbacError>;
    fn set_motifs(&mut self, run_id:usize, channel:i64) -> Result<(), HbacError>;
    fn motifs_to_eeg_slices(&mut self, run_id:usize) -> Result<HashMap<i64, Vec<MotifSlice>>, HbacError>;
    fn list_run_resources(&mut self, run_id:usize) -> Result<Vec<String>, HbacError>;
    fn save_motifs_corrs(&mut self, descr:&str, motifs_corrs:Vec<Vec<(usize, f64)>>) -> Result<usize, HbacError>;
    fn list_corr_motifs(&mut self, corr_run_id:usize) -> Result<Vec<Vec<(usize, f64)>>, HbacError>;
    fn list_motif_slices(&mut self, motif_id:usize) -> Result<Vec<TensorSliceIdx>, HbacError>;
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

    fn save_run(&mut self, descr:&str, keys:&Vec<String>) -> Result<usize, HbacError> {
        let mut conn_guard = self.conn.lock()?;
        let mut t = conn_guard.transaction()?;

        t.execute("INSERT INTO matches_run (descr) VALUES (?)", params![descr])?;

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

        t.commit();

        Ok(run_id)
    }


    fn save_slices(&mut self, run_id:usize, channel:i64, slices:Vec<((usize, i64), (usize, i64), f32)>) -> Result<(), HbacError>{
        let mut conn_guard = self.conn.lock()?;
        let mut t = conn_guard.transaction()?;

        let (mut q, params) = slices.into_iter().fold(
            (vec!["INSERT INTO slice_matches (run_id, channel_id, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1, dot) VALUES"], vec![]), |mut a: (Vec<&str>, Vec<Value>), v| {
            a.0.push("(?, ?, ?, ?, ?, ?, ?)");
            a.0.push(", ");
            a.1.push(Value::Integer(run_id as i64));
            a.1.push(Value::Integer(channel as i64));
            a.1.push(Value::Integer(v.0.0 as i64));
            a.1.push(Value::Integer(v.0.1));
            a.1.push(Value::Integer(v.1.0 as i64));
            a.1.push(Value::Integer(v.1.1));
            a.1.push(Value::Real(v.2 as f64));
            a
        });

        q.pop();

        t.execute(&q.into_iter().collect::<String>(), rusqlite::params_from_iter(params.into_iter()))?;

        t.commit();

        Ok(())
    }

    fn set_motifs(&mut self, run_id:usize, channel:i64) -> Result<(), HbacError> {
        let mut guard = self.conn.lock()?;
        let t = guard.transaction()?;

        {
            let mut stmt = t.prepare("SELECT id, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1 FROM slice_matches WHERE run_id=? AND channel_id=?")?;

            let mut groups = group_slices(stmt.query_map([run_id, channel as usize], |row| {
                Ok((IdSlice(row.get(0)?, row.get(1)?, row.get(2)?), IdSlice(row.get(0)?, row.get(3)?, row.get(4)?)))
            })?.map(|v| v.unwrap()));

            //TODO sort?
            for (group, slices) in groups {

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

        t.commit();

        Ok(())
    }

    /**
    * returns map tensor_id - MotifSlice(motif_group, tensor slice idx, channel)
    */
    fn motifs_to_eeg_slices(&mut self, run_id:usize) -> Result<HashMap<i64, Vec<MotifSlice>>, HbacError>{
        let mut guard = self.conn.lock()?;
        let t = guard.transaction()?;


        let mut stmt = t.prepare("SELECT id, motif_id, channel_id, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1 FROM slice_matches WHERE run_id=?")?;

        let tensor_motifs:HashMap<i64, Vec<MotifSlice>> = stmt.query_map([run_id], |row| {
            Ok((
                row.get(0)?, // id,
                row.get(1).unwrap_or(-1), // motif,
                row.get(2)?, // channel_id,
                row.get(3)?, // tensor_id_0,
                row.get(4)?, // slice_idx_0,
                row.get(5)?, // tensor_id_1,
                row.get(6)?, // slice_idx_1
            ))

        })?.fold(Ok(HashMap::new()), |a:Result<HashMap<i64, Vec<hbac_shared::MotifSlice>>, rusqlite::Error>, v:Result<(i64, i64, i64, i64, i64, i64, i64), rusqlite::Error>|{
            let v = v?;
            if v.1 >=0 {
                let mut a = a?;

                a.entry(v.3).or_insert_with(|| vec![]).push(MotifSlice::new(v.1 as u16, v.4, v.2 as u16));
                a.entry(v.5).or_insert_with(|| vec![]).push(MotifSlice::new(v.1 as u16, v.6, v.2 as u16));

                Ok(a)
            } else {
                a
            }
        })?;

        if tensor_motifs.len() < 10 {
            debug!("motifs_to_eeg_slices: eeg_motifs={:?}", tensor_motifs);
        }

        Ok(tensor_motifs)
    }

    fn list_run_resources(&mut self, run_id:usize) -> Result<Vec<String>, HbacError> {
        let mut guard = self.conn.lock()?;
        let t = guard.transaction()?;

        let count:i64 = t.query_row("SELECT count(*) FROM run_resources WHERE run_id=?", [run_id], |row| row.get(0))?;

        debug!("list_run_resources: count={}", count);

        let mut stmt = t.prepare("SELECT ord_id, key FROM run_resources WHERE run_id=? ORDER BY ord_id")?;


        let result = stmt.query_map([run_id], |row| {

            Ok((row.get::<usize, i64>(0)?, row.get::<usize, String>(1)?))

        })?.enumerate().map(|(i, ord_key)|{
            ord_key.map(|(ord, key)| {
                assert!(ord == i as i64);
                key
            }).map_err(|e|HbacError::new(e.to_string()))
        }).collect::<Result<Vec<String>, HbacError>>();

        result
    }

    fn save_motifs_corrs(&mut self, descr:&str, motifs_corrs:Vec<Vec<(usize, f64)>>) -> Result<usize, HbacError>{
        let mut guard = self.conn.lock()?;
        let t = guard.transaction()?;

        t.execute("INSERT INTO corrs_run (descr) VALUES (?)", params![descr])?;

        let corr_run_id = t.query_row("SELECT last_insert_rowid()", [], |row| row.get::<usize, usize>(0))?;


        let (mut q, params) = motifs_corrs.into_iter().flat_map(|channel_motif_corrs|channel_motif_corrs.into_iter()).into_iter().fold(
            (vec!["INSERT INTO motifs_corrs (corr_id, motif_id, corr) VALUES "], vec![]), //TODO check if channel matches one in motif
            |mut a: (Vec<&str>, Vec<Value>), (motif_id, corr)| {
                a.0.push("(?, ?, ?)");
                a.0.push(", ");
                a.1.push(Value::Integer(corr_run_id as i64));
                a.1.push(Value::Integer(motif_id as i64));
                a.1.push(Value::Real(corr));

                a
            });

        q.pop();

        t.execute(&q.into_iter().collect::<String>(), rusqlite::params_from_iter(params.into_iter()))?;
        t.commit();

        Ok(corr_run_id)
    }

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
        t.commit();

        Ok(channels_corr_motifs)
    }

    fn list_motif_slices(&mut self, motif_id:usize) -> Result<Vec<TensorSliceIdx>, HbacError> {

        let mut guard = self.conn.lock()?;
        let t = guard.transaction()?;

        let slices = {
            let mut stmt = t.prepare("SELECT  channel_id, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1 FROM slice_matches where motif_id=?")?;

            let (slices, channel_set) = stmt.query_map([motif_id], |row| {
                Ok((
                    row.get::<usize, i64>(0)?,//channel_id
                    row.get::<usize, i64>(1)?,//tensor_id_0
                    row.get::<usize, i64>(2)?,//slice_idx_0
                    row.get::<usize, i64>(3)?,//tensor_id_1
                    row.get::<usize, i64>(4)?,//slice_idx_1
                ))
            })?.map(|v| v.unwrap()).fold((HashSet::new(), HashSet::new()), |(mut a, mut channel_set), v| {
                a.insert(TensorSliceIdx(v.1 as usize, v.2 as i64));
                a.insert(TensorSliceIdx(v.3 as usize, v.4 as i64));

                channel_set.insert(v.0);
                (a, channel_set)
            });

            assert!(channel_set.len() == 1);

            slices.into_iter().collect()
        };

        t.commit();

        Ok(slices)
    }
}


impl SqlStore {

    pub fn new<P:AsRef<Path>>(path:P) -> Result<Self, HbacError> {
        let mut conn = Connection::open(path)?;

        {
            let mut t = conn.transaction()?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS matches_run (
                id          INTEGER PRIMARY KEY,
                descr       VARCHAR(256),
                run_on      DATETIME DEFAULT CURRENT_TIMESTAMP
            )", ())?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS run_resources (
                run_id     REFERENCES matches_run(id) ON DELETE CASCADE,
                ord_id     INTEGER NOT NULL,
                key        VARCHAR(256),
                PRIMARY KEY (run_id, ord_id)
            )", ())?;

            t.execute( //group - ordinal as appeared when grouping, not needed
                "CREATE TABLE IF NOT EXISTS motifs (
                id              INTEGER PRIMARY KEY,
                group_ord       INTEGER NOT NULL,
                run_id          REFERENCES matches_run(id) ON DELETE CASCADE,
                channel_id      INTEGER NOT NULL
            )", ())?;

            t.execute(
            "CREATE TABLE IF NOT EXISTS slice_matches (
                id              INTEGER PRIMARY KEY,
                run_id          REFERENCES matches_run(id) ON DELETE CASCADE,
                channel_id      INTEGER NOT NULL,
                tensor_id_0     INTEGER NOT NULL,
                slice_idx_0     INTEGER NOT NULL,
                tensor_id_1     INTEGER NOT NULL,
                slice_idx_1     INTEGER NOT NULL,
                dot             FLOAT NOT NULL,
                motif_id        REFERENCES motifs(id) ON DELETE CASCADE DEFAULT NULL
            )", ())?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS corrs_run (
                id          INTEGER PRIMARY KEY,
                descr       VARCHAR(256),
                run_on      DATETIME DEFAULT CURRENT_TIMESTAMP
            )", ())?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS motifs_corrs (
                motif_id   REFERENCES motifs(id),
                corr_id    REFERENCES corrs_run(id),
                corr       FLOAT NOT NULL
            )", ())?;

            t.execute("commit", ())?;
        }

        Ok(Self {
            conn: Mutex::new(conn)
        })
    }

}


#[cfg(test)]
mod store_tests{
    use std::cmp::Ordering;
    use std::collections::HashMap;
    use std::error::Error;
    use std::fs;
    use rusqlite::Connection;
    use rusqlite::types::Value;
    use crate::{SqlStore, Store, HbacError};
    use hbac_shared::TensorSliceIdx;


    #[test]
    fn test_save_slices() -> Result<(), HbacError> {
        fs::remove_file("/tmp/test_db.db3").unwrap_or(());
        let mut store = SqlStore::new("/tmp/test_db.db3")?;

        let map :Vec<((usize, i64),(usize, i64),f32)> = vec![
            ((0usize, 100i64), (0usize, 200i64), 0.91),
            ((0usize, 100i64), (1usize, 200i64), 0.92),
            ((3usize, 100i64), (3usize, 200i64), 0.93),
            ((3usize, 200i64), (5usize, 200i64), 0.94),
            ((3usize, 200i64), (5usize, 300i64), 0.94),
        ];

        let run_id= store.save_run("test", &vec!["aaa".to_string(), "bbb".to_string()])?;

        store.save_slices(run_id,  7, map)?;

        store.set_motifs(run_id, 7)?;

        let m_1_first = {
            let mut guard = store.conn.lock()?;
            let t2 = guard.transaction()?;
            let mut stmt = t2.prepare("SELECT run_id, channel_id, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1, motif_id FROM slice_matches")?;

            let mut rows = stmt.query_map([], |row| {
                Ok((
                    row.get(0)?,//run_id
                    row.get(1)?,//channel_id
                    row.get(2)?,//tensor_id_0
                    row.get(3)?,//slice_idx_0
                    row.get(4)?,//tensor_id_1
                    row.get(5)?,//slice_idx_1
                    row.get(6)? //motif_id
                ))
            })?.map(|v|v.unwrap()).collect::<Vec<(i64, i64, i64, i64, i64, i64, i64)>>();

            let m_1_first = vec![
                (1, 7, 0, 100, 0, 200, 1),
                (1, 7, 0, 100, 1, 200, 1),
                (1, 7, 3, 100, 3, 200, 2),
                (1, 7, 3, 200, 5, 200, 2),
                (1, 7, 3, 200, 5, 300, 2)
            ] == rows;

            let m_2_first = vec![
                (1, 7, 0, 100, 0, 200, 2),
                (1, 7, 0, 100, 1, 200, 2),
                (1, 7, 3, 100, 3, 200, 1),
                (1, 7, 3, 200, 5, 200, 1),
                (1, 7, 3, 200, 5, 300, 1)
            ] == rows;

            assert!(m_2_first  || m_1_first  );                        //TODO motif ids are random

            let mut stmt = t2.prepare("SELECT run_id, ord_id, key FROM run_resources")?;

            let mut rows = stmt.query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?,  row.get(2)?))
            })?.map(|v|v.unwrap()).collect::<Vec<(usize, usize, String)>>();


            assert_eq!(vec![(1, 0, "aaa".to_string()), (1, 1, "bbb".to_string())], rows);

            m_1_first
        };

        let map = store.motifs_to_eeg_slices(run_id)?;
        println!("map={:?}", map);

        assert_eq!(4, map.len());
        assert_eq!(3, map.get(&0).unwrap().len());
        assert_eq!(1, map.get(&1).unwrap().len());
        assert_eq!(4, map.get(&3).unwrap().len());
        assert_eq!(2, map.get(&5).unwrap().len());

        assert_eq!(vec!["aaa".to_owned(), "bbb".to_owned()], store.list_run_resources(run_id)?);

        let corr_id = store.save_motifs_corrs("test", vec![vec![(1, 0.1), (2, 0.2)]])?;

        let motifs = store.list_corr_motifs(corr_id)?;

        println!("motifs: {:?}", motifs);

        assert_eq!(1, motifs[7][0].0);
        assert_eq!(2, motifs[7][1].0);

        let mut slices = store.list_motif_slices(1)?;
        slices.sort_by(|a, b| if a.0 + (a.1 as usize) < b.0 + (b.1 as usize) { Ordering::Less } else {Ordering::Greater});

        if m_1_first {
            assert_eq!(vec![
                TensorSliceIdx(0, 100), TensorSliceIdx(0, 200), TensorSliceIdx(1, 200)
            ], slices);
        } else {
            assert_eq!(vec![
                TensorSliceIdx(3, 100), TensorSliceIdx(3, 200), TensorSliceIdx(5, 200), TensorSliceIdx(5, 300)
            ], slices);
        }

        Ok(())
    }
}
