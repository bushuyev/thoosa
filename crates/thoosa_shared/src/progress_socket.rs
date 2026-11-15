use crate::collect_futures_eta::CollectEtaProgressState;
use serde::{Deserialize, Serialize};

// Actions share the progress connection, but their replies belong only to the requesting client.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ProgressRequest {
    ShowCacheStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheStatistic {
    pub type_name: String,
    pub bytes: u64,
}

// Keep ordinary snapshots in their existing JSON shape for passive progress clients.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ProgressResponse {
    Snapshot(CollectEtaProgressState),
    CacheStatistics { cache_statistics: Result<Vec<CacheStatistic>, String> },
}
