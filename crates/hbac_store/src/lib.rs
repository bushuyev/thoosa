extern crate core;

use std::backtrace::Backtrace;
use std::cell::RefCell;
use std::collections::HashMap;
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

        let run_id = t.execute("INSERT INTO matches_run (descr) VALUES (?)", params![descr])?;


        let (mut q, params) = keys.into_iter().fold(
            (vec!["INSERT INTO run_resources (run_id, key) VALUES"], vec![]), |mut a: (Vec<&str>, Vec<Value>), v| {
                a.0.push("(?, ?)");
                a.0.push(", ");
                a.1.push(Value::Integer(run_id as i64));
                a.1.push(Value::Text(v.clone()));
                a
            });

        q.pop();

        t.execute(&q.into_iter().collect::<String>(), rusqlite::params_from_iter(params.into_iter()))?;


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
                let (mut q, params) = slices.into_iter().fold(
                    (vec!["UPDATE slice_matches SET motif=? WHERE id IN ("], vec![Value::Integer(group as i64)]),
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
                id         INTEGER PRIMARY KEY,
                run_id     REFERENCES matches_run(id) ON DELETE CASCADE,
                key        VARCHAR(256)
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
                motif           INTEGER DEFAULT NULL
            )", ())?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS motifs (
                id    INTEGER PRIMARY KEY,
                descr    VARCHAR(256),
                run_on    DATETIME DEFAULT CURRENT_TIMESTAMP,
                channel_id  INTEGER NOT NULL
            )", ())?;
            //
            // t.execute(
            //     "CREATE TABLE IF NOT EXISTS motif_slices (
            //     id          INTEGER PRIMARY KEY,
            //     motif_id    REFERENCES motifs(id) ON DELETE CASCADE,
            //     tensor_id   INTEGER NOT NULL,
            //     slice_idx   INTEGER NOT NULL
            // )", ())?;
            //
            t.execute("commit", ())?;
        }

        Ok(Self {
            conn: Mutex::new(conn)
        })
    }

}


#[cfg(test)]
mod store_tests{
    use std::collections::HashMap;
    use std::error::Error;
    use std::fs;
    use rusqlite::Connection;
    use rusqlite::types::Value;
    use crate::{SqlStore, Store, HbacError};


    #[derive(Debug, PartialEq, Hash, Eq, Copy, Clone)]
    pub struct TensorSliceIdx(pub usize, pub i64);



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

        {
            let mut guard = store.conn.lock()?;
            let t2 = guard.transaction()?;
            let mut stmt = t2.prepare("SELECT run_id, channel_id, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1, motif FROM slice_matches")?;

            let mut rows = stmt.query_map([], |row| {
                Ok((
                    row.get(0)?,//run_id
                    row.get(1)?,//channel_id
                    row.get(2)?,//tensor_id_0
                    row.get(3)?,//slice_idx_0
                    row.get(4)?,//tensor_id_1
                    row.get(5)?,//slice_idx_1
                    row.get(6)?//motif_id
                ))
            })?.map(|v|v.unwrap()).collect::<Vec<(i64, i64, i64, i64, i64, i64, i64)>>();

            assert_eq!(vec![
                (1, 7, 0, 100, 0, 200, 0),
                (1, 7, 0, 100, 1, 200, 0),
                (1, 7, 3, 100, 3, 200, 1),
                (1, 7, 3, 200, 5, 200, 1),
                (1, 7, 3, 200, 5, 300, 1)
            ],
           rows);

            let mut stmt = t2.prepare("SELECT * FROM run_resources")?;

            let mut rows = stmt.query_map([], |row| {
                Ok((row.get(1)?,  row.get(2)?))
            })?.map(|v|v.unwrap()).collect::<Vec<(usize, String)>>();


            assert_eq!(vec![(1usize, "aaa".to_string()), (1usize, "bbb".to_string())], rows)

        }

        Ok(())
    }
}
