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


pub trait Store:Sync + Send{
    fn save_run(&mut self, name:&str) -> Result<usize, crate::StoreError>;
    fn save_slices(&mut self, run_id:usize, name:&str, channel:i64, slices:Vec<((usize, i64), (usize, i64), f32)>) -> Result<(), crate::StoreError>;
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

    fn save_run(&mut self, name:&str) -> Result<usize, crate::StoreError> {
        let mut conn_guard = self.conn.lock()?;
        let mut t = conn_guard.transaction()?;

        let run_id = t.execute("INSERT INTO matches_run (name) VALUES (?)", params![name])?;

        t.commit();

        Ok(run_id)
    }


    fn save_slices(&mut self, run_id:usize, name:&str, channel:i64, slices:Vec<((usize, i64), (usize, i64), f32)>) -> Result<(), StoreError>{
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

}

#[derive(Debug)]
pub enum StoreError {
    Some(String)
}

impl Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl StoreError {
    pub fn new(msg:String) -> Self {
        debug!("StoreError::new: msg={}", msg);
        StoreError::Some(msg)

    }
}

impl From<rusqlite::Error> for StoreError {
    fn from(value: rusqlite::Error) -> Self {
        debug!("StoreError::from: value={}", value);
        StoreError::new(value.to_string())
    }
}


impl From<PoisonError<MutexGuard<'_, Connection>>> for StoreError {
    fn from(value: PoisonError<MutexGuard<'_, Connection>>) -> Self {
        debug!("StoreError::from: value={}", value);
        StoreError::new(value.to_string())
    }
}


impl SqlStore {

    pub fn new<P:AsRef<Path>>(path:P) -> Result<Self, StoreError> {
        let mut conn = Connection::open(path)?;

        {
            let mut t = conn.transaction()?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS matches_run (
                id          INTEGER PRIMARY KEY,
                name        VARCHAR(256),
                run_on      DATETIME DEFAULT CURRENT_TIMESTAMP
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
                dot             FLOAT NOT NULL
            )", ())?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS motifs (
                id    INTEGER PRIMARY KEY,
                name    VARCHAR(256),
                run_on    DATETIME DEFAULT CURRENT_TIMESTAMP,
                channel_id  INTEGER NOT NULL
            )", ())?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS motif_slices (
                id          INTEGER PRIMARY KEY,
                motif_id    REFERENCES motif(id) ON DELETE CASCADE,
                tensor_id   INTEGER NOT NULL,
                slice_idx   INTEGER NOT NULL
            )", ())?;

            t.execute("commit", ())?;
        }

        Ok(Self {
            conn: Mutex::new(conn)
        })
    }



    pub fn save_motifs(&mut self, name:&str, channel:i64, motifs_slices:HashMap<u16, Vec<(usize, i64)>>) -> Result<(), StoreError>{
        let mut conn_guard = self.conn.lock()?;
        let mut t = conn_guard.transaction()?;

        let motif_id = t.execute("INSERT INTO motifs (name, channel_id) VALUES (?, ?)", (name, channel))?;

        debug!("save_motifs: name={} channel={} motif_id={} motifs_slices.len()={}", name, channel, motif_id, motifs_slices.len());

        for slices in motifs_slices.into_values() {

            let (mut q, params) = slices.into_iter().fold((vec!["INSERT INTO motif_slices (motif_id, tensor_id, slice_idx) VALUES"], vec![]), |mut a: (Vec<&str>, Vec<usize>), v| {
                a.0.push("(?, ?, ?)");
                a.0.push(", ");
                a.1.push(motif_id);
                a.1.push(v.0);
                a.1.push(v.1 as usize);
                a
            });

            q.pop();

            t.execute(&q.into_iter().collect::<String>(), rusqlite::params_from_iter(params.into_iter()))?;
        }

        t.commit();

        Ok(())
    }

}


#[cfg(test)]
mod store_tests{
    use std::collections::HashMap;
    use std::error::Error;
    use std::fs;
    use rusqlite::Connection;
    use crate::{SqlStore, Store, StoreError};


    #[derive(Debug, PartialEq, Hash, Eq, Copy, Clone)]
    pub struct TensorSliceIdx(pub usize, pub i64);


    #[test]
    fn test_save_motifs() -> Result<(), StoreError> {
        fs::remove_file("/tmp/test_db.db3").unwrap_or(());
        let mut store = SqlStore::new("/tmp/test_db.db3")?;
        println!("Ok");

        let map :HashMap<u16, Vec<(usize, i64)>> = HashMap::from([
            (0u16, vec![(0, 100), (0, 200)]),
            (1u16, vec![(1, 100), (1, 200)]),
        ]);

        println!("map: {:?}",  map);
        store.save_motifs("seizure_400", 0, map)?;

         {
             let mut conn_guard = store.conn.lock()?;
             let t2 = conn_guard.transaction()?;
           let mut stmt = t2.prepare("SELECT * FROM motif_slices")?;

           let mut rows = stmt.query([])?;
           while let Some(row) = rows.next()? {
               println!("row={:?}", row);
           }

       }

        Ok(())
    }


    #[test]
    fn test_save_slices() -> Result<(), StoreError> {
        fs::remove_file("/tmp/test_db.db3").unwrap_or(());
        let mut store = SqlStore::new("/tmp/test_db.db3")?;
        println!("Ok");

        let map :Vec<((usize, i64),(usize, i64),f32)> = vec![
            ((0usize, 100i64), (0usize, 200i64), 0.91),
            ((1usize, 100i64), (1usize, 200i64), 0.92),
        ];

        println!("map: {:?}",  map);
        let run_id= store.save_run("test")?;
        store.save_slices(run_id, "seizure_400", 0, map)?;

        {
            let mut guard = store.conn.lock()?;
            let t2 = guard.transaction()?;
            let mut stmt = t2.prepare("SELECT * FROM slice_matches")?;

            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                println!("row={:?}", row);
            }

        }

        Ok(())
    }
}
