extern crate core;

use std::backtrace::Backtrace;
use std::collections::HashMap;
use std::fmt::Display;
use std::path::Path;
use rusqlite::{Connection, params, ToSql};
use log::debug;
use rusqlite::types::{ToSqlOutput, Value};
use rusqlite::types::ValueRef::Real;

struct Store {
    conn:Connection
}

#[derive(Debug)]
enum StoreError {
    Some(String)
}

impl StoreError {
    pub fn new(msg:String) -> Self {
        debug!("StoreError::new: msg={}", msg);
        StoreError::Some(msg)

    }
}


impl<D> From<D> for StoreError where D: Display {

    fn from(value: D) -> Self {
        debug!("HbacError::from: value={}", value);
        StoreError::new(value.to_string())
    }
}

#[derive(Debug)]
struct X (Vec<usize>);

unsafe impl Sync for X{}
unsafe impl Send for X{}

// impl ToSql for X {
//     fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
//         Ok(ToSqlOutput::from(Value::from(1)))
//     }
// }




impl Store {
    pub fn new<P:AsRef<Path>>(path:P) -> Result<Self, StoreError> {
        let mut conn = Connection::open(path)?;

        {
            let mut t = conn.transaction()?;

            t.execute(
                "CREATE TABLE IF NOT EXISTS motif (
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
            conn
        })
    }

    pub fn save_motifs(&mut self, name:&str, channel:i64, motifs_slices:HashMap<u16, Vec<(usize, i64)>>) -> Result<(), StoreError>{

        let mut t = self.conn.transaction()?;

        let motif_id = t.execute("INSERT INTO motif (name, channel_id) VALUES (?, ?)", (name, channel))?;

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
    use crate::{Store, StoreError};


    #[derive(Debug, PartialEq, Hash, Eq, Copy, Clone)]
    pub struct TensorSliceIdx(pub usize, pub i64);


    #[test]
    fn test_it() -> Result<(), StoreError> {
        fs::remove_file("/tmp/test_db.db3").unwrap_or(());
        let mut store = Store::new("/tmp/test_db.db3")?;
        println!("Ok");

        let map :HashMap<u16, Vec<(usize, i64)>> = HashMap::from([
            (0u16, vec![(0, 100), (0, 200)]),
            (1u16, vec![(1, 100), (1, 200)]),
        ]);

        println!("map: {:?}",  map);
        store.save_motifs("seizure_400", 0, map)?;

         {
           let t2 = store.conn.transaction()?;
           let mut stmt = t2.prepare("SELECT * FROM motif_slices")?;

           let mut rows = stmt.query([])?;
           while let Some(row) = rows.next()? {
               println!("row={:?}", row);
           }

       }

        Ok(())
    }
}
