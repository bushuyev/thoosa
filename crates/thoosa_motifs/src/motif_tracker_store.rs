use crate::motifs_store::MotifsStore;
use async_trait::async_trait;
use bytemuck::{Pod, Zeroable};
use log::{debug, trace, warn};
use parking_lot::Mutex;
use rusqlite::types::Value;
use rusqlite::Connection;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thoosa_shared::resource::ResourceId;
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::{group_slices_try, TensorIdSliceIdx, ThoError};

// Tracker persistence belongs to the motif crate so it can depend on MotifsStore directly
// without exposing motif-specific callbacks through the common Store crate.
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

#[repr(C)]
#[derive(Debug, PartialEq, Copy, Clone, Pod, Zeroable)]
pub struct SlicesMatch(pub TensorIdSliceIdx, pub TensorIdSliceIdx, pub f32);

#[async_trait]
#[cfg_attr(test, mockall::automock)]
pub trait MotifTrackerStore: Debug + Send + Sync {
    fn save_slices(&mut self, slices: Vec<SlicesMatch>) -> anyhow::Result<()>;
    async fn load_slices_into_groups(&mut self, store: &dyn MotifsStore, min_slices_counts: Option<u32>, max_slices_counts: Option<u32>) -> anyhow::Result<usize>;

    fn run_config_id(&self) -> RunConfigId;
    fn channel(&self) -> u32;
}

#[cfg(test)]
pub(crate) mod mock_motif_tracker_store {
    use super::*;

    // Inject persistence so tests can observe both dump boundaries and contents.
    pub struct MockMotifTrackerStoreBuilder(MockMotifTrackerStore);

    impl MockMotifTrackerStoreBuilder {
        pub fn new() -> Self { Self(MockMotifTrackerStore::new()) }

        pub fn with_save_slices(mut self, save: impl FnMut(Vec<SlicesMatch>) -> anyhow::Result<()> + Send + 'static) -> Self {
            self.0.expect_save_slices().returning(save);
            self
        }

        pub fn build(self) -> Box<dyn MotifTrackerStore> { Box::new(self.0) }
    }
}

pub fn filter_groups_by_slices_count(
    min_slices_counts: Option<u32>,
    max_slices_counts: Option<u32>,
    mut groups: HashMap<usize, HashSet<IdSlice>>,
) -> HashMap<usize, HashSet<IdSlice>> {
    if min_slices_counts.is_none() && max_slices_counts.is_none() {
        return groups;
    }

    groups.retain(|group_id, slices| {
        let count = slices.len();

        let keep = min_slices_counts.is_none_or(|min| count >= min as usize)
            && max_slices_counts.is_none_or(|max| count < max as usize);

        if !keep {
            warn!("skipping group {group_id} with {count} slices");
        }

        keep
    });

    groups
}
pub struct SqlMotifTrackerStore {
    run_config_id: RunConfigId,
    channel: u32,
    conn: Arc<Mutex<Connection>>,
    path: PathBuf,
}

impl Debug for SqlMotifTrackerStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("").finish()
    }
}

impl SqlMotifTrackerStore {
    pub fn new<P: AsRef<Path> + Sync + Send + Debug>(
        path: P,
        run_config_id: RunConfigId,
        channel: u32,
    ) -> anyhow::Result<Box<dyn MotifTrackerStore>> {
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
            run_config_id,
            channel,
            conn: Arc::new(Mutex::new(conn)),
            path: path.as_ref().to_path_buf(),
        }))
    }
}

#[async_trait]
impl MotifTrackerStore for SqlMotifTrackerStore {
    fn save_slices(&mut self, slices: Vec<SlicesMatch>) -> anyhow::Result<()> {
        debug!("MotifTrackerStore::save_slices: motifs.len()={:?}, self.run_config_id={:?}",slices.len(),self.run_config_id);

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
        store: &dyn MotifsStore,
        min_slices_counts: Option<u32>,
        max_slices_counts: Option<u32>,
    ) -> anyhow::Result<usize> {
        // SQLite locking, row iteration, and grouping must not occupy a Tokio worker.
        let conn = Arc::clone(&self.conn);
        let groups_by_slices_count = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
            let guard = conn.lock();
            let mut stmt = guard.prepare(
                "SELECT id, tensor_id_0, slice_idx_0, tensor_id_1, slice_idx_1 FROM slice_matches",
            )?;
            let groups = group_slices_try(stmt.query_map([], |row| {
                Ok((
                    IdSlice(row.get(0)?, row.get(1)?, row.get(2)?),
                    IdSlice(row.get(0)?, row.get(3)?, row.get(4)?),
                ))
            })?)?;
            Ok(filter_groups_by_slices_count(Some(2), max_slices_counts, groups))
        }).await??;
        let groups_len = groups_by_slices_count.len();
        store.save_groups_as_motifs_with_slices_matches(
            self.run_config_id,
            self.channel as i64,
            groups_by_slices_count,
            self.path.clone(),
        ).await?;

        Ok(groups_len)
    }

    fn run_config_id(&self) -> RunConfigId {
        self.run_config_id
    }

    fn channel(&self) -> u32 {
        self.channel
    }
}

pub enum CheckGroups {
    Lengths(Vec<usize>),
    MinGroups(usize),
}
