use async_trait::async_trait;
use futures_util::future::BoxFuture;
use futures_util::{FutureExt, StreamExt};
use parking_lot::{Mutex, RwLock};
use std::array::IntoIter;
use std::backtrace::Backtrace;
use std::cell::RefCell;
use std::cmp::{max, min};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use log::{debug, error, info, trace, warn};
use mockall::predicate::eq;
use mockall::{PredicateBoxExt, mock};
use rand::seq::SliceRandom;
use rand::{Rng, RngExt};
use rusqlite::trace::TraceEvent;
use rusqlite::types::Value;
use rusqlite::{Connection, Error, OpenFlags, ToSql, Transaction, params};
use serde::{Deserialize, Serialize};
use thoosa_shared::resource::{Resource, ResourceId, ResourceStats, ResourceStore};
use thoosa_shared::run_configs::{RunConfigId, RunId, run_id};
use thoosa_shared::traits::DataId;
use thoosa_shared::val::Val;
use thoosa_shared::val::Val::UInt;
use thoosa_shared::*;
use thoosa_store::{LimitBy, Store, SetCond, IdSlice, RunConfigEntry, TrainedNetsBy, RunConfigRow, RunConfigFilter, RunMotifEntry, TrainedNetEntry, TrainedNet, SqlStore};

#[async_trait]
pub trait MotifsStore: Store {}

impl MotifsStore for SqlStore {

}


mock! {
    pub MotifsStore {}

    #[async_trait]
    impl MotifsStore for MotifsStore {

    }

    #[async_trait]
    impl Store for MotifsStore{
      fn open_external_writer_conn(&self) -> Result<Connection, ThoError>;
        async fn save_import(&self, descr: String) -> Result<u32, ThoError>;
        async fn save_run(&self, descr: String) -> Result<RunId, ThoError>;
        async fn save_run_config<'a, 'b, 'c>(
            &self,
            input_run_config_id: Option<&'a [RunConfigId]>,
            run_config_type: &'b str,
            descr: &'c str,
            properties_vals: HashMap<String, Val>,
        ) -> Result<RunConfigId, ThoError>;

        async fn get_input_run_config_ids<'a>(&self, run_config_id: RunConfigId, input_type: Option<&'a str>) -> Result<Vec<RunConfigId>, ThoError>;

        async fn list_input_run_config_ids(&self, ids: &[RunConfigId]) -> Result<Vec<(RunConfigId, RunConfigId)>, ThoError>; //corr_run_config_id, run_config_id, label

        async fn list_run_config_ids_with_outputs(&self, input_run_config_ids: &[RunConfigId]) -> Result<Vec<RunConfigId>, ThoError>;

        async fn list_output_run_config_ids(&self, input_run_config_id: RunConfigId) -> Result<Vec<RunConfigId>, ThoError>; //corr_run_config_id, run_config_id, label

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
        async fn save_leaky_data_sets(
            &self,
            prefix: &str,
            limit: LimitBy,
            train_coeff: f32,
            valid_coeff: f32,
            min_p: f32,
            allow_leaky_split: bool,
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
        async fn list_label_nets_run_config_ids(
            &self,
            label: &str,
        ) -> Result<Vec<RunConfigId>, ThoError>;

        async fn export_apply_label_nets_records(
            &self,
            target_db_path: PathBuf,
            label_net_run_config_ids: Vec<RunConfigId>,
        ) -> Result<(), ThoError>;
        async fn resources_by_label(
            &self,
            min_p: f32,
        ) -> Result<HashMap<String, Vec<Resource>>, ThoError>;
        async fn list_resource_random_not_labeled(
            &self,
            label: &str,
            limit: u32,
        ) -> Result<Vec<Resource>, ThoError>;

        async fn save_label(&self, label: &str) -> Result<u32, ThoError>;

        async fn list_labels(&self, only_with_resources: bool) -> Result<Vec<String>, ThoError>;

        async fn save_params_value<'a>(&self, params: HashMap<String, Val>, run_config_id: RunConfigId, label: Option<&'a str>) -> Result<(), ThoError>;
        async fn save_slices(&self, run_config_id: RunConfigId, channel: i64, slices: Vec<(TensorIdSliceIdx, TensorIdSliceIdx, f32)>) -> Result<(), ThoError>;

        async fn save_groups_as_motifs_with_slices_matches(&self, run_config_id: RunConfigId, channel: i64, groups: HashMap<u16, HashSet<IdSlice>>, path: PathBuf) -> Result<(), ThoError>;

        async fn save_groups_as_motifs(&self, run_config_id: RunConfigId, channel: i64, groups: HashMap<u16, HashSet<IdSlice>>) -> Result<Vec<u32>, ThoError>;

        async fn add_slices_to_motif(&self, run_config_id: RunConfigId, motif_id: u32, channel: i64, slices: HashSet<IdSlice>) -> Result<(), ThoError>;

        async fn list_motifs(&self, run_config_id: RunConfigId) -> Result<Vec<u32>, ThoError>;
        // fn get_motif_meta<'a>(&self, motif_id: u32) -> BoxFuture<'a, Result<(usize, RunId), ThoError>>;
        async fn get_motif_meta(&self, motif_id: u32) -> Result<(usize, RunConfigId), ThoError>;
        async fn list_motifs_slices(
            &self,
            run_config_id: RunConfigId,
            select_motifs_run_config_id: Option<Vec<RunConfigId>>,
        ) -> Result<HashMap<Resource, Vec<MotifSlice>>, ThoError>;

        async fn list_motif_slices(
            &self,
            motif_id: u32,
            select_motifs_run_config_id: Option<Vec<RunConfigId>>,
            limit: Option<u32>,
            all_but_in_run_config_id: Option<(RunConfigId, u32)>,
        ) -> Result<
            (
                i64,
                RunConfigId,
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
                RunConfigId,
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
                RunConfigId,
                Vec<TensorIdSliceIdx>,
                HashMap<ResourceId, Resource>,
            ),
            ThoError,
        >;

        async fn list_run_resources(&self, run_config_id: RunConfigId) -> Result<Vec<String>, ThoError>;

        async fn list_run_configs(&self) -> Result<Vec<RunConfigEntry>, ThoError>;

        async fn list_run_config_ids_by_descr(&self, descr:String, type_:&'static str) -> Result<Vec<RunConfigId>, ThoError>;
        async fn list_run_configs_page(&self, limit: usize, offset: usize) -> Result<Vec<RunConfigEntry>, ThoError>;
        async fn list_run_configs_by_ids(&self, ids: &[RunConfigId]) -> Result<Vec<RunConfigEntry>, ThoError>;

        async fn list_run_configs_filtered(&self, filter: RunConfigFilter) -> Result<Vec<RunConfigEntry>, ThoError>;
        async fn list_run_config_types(&self) -> Result<Vec<String>, ThoError>;

        async fn list_run_config_values(&self, run_config_id: RunConfigId) -> Result<HashMap<String, Val>, ThoError>;

        async fn list_run_motifs(&self, run_config_id: RunConfigId) -> Result<Vec<RunMotifEntry>, ThoError>;

        async fn save_selected_motifs(&self, run_config_id: RunConfigId, selected_motifs: Vec<(u32, f64)>) -> Result<RunConfigId, ThoError>;

        async fn select_all_motifs(&self, run_config_ids: &[RunConfigId]) -> Result<Vec<RunConfigId>, ThoError>;

        async fn list_selected_motifs(&self, motif_selection_id: RunConfigId, channels: u32) -> Result<Vec<Vec<(u32, f64)>>, ThoError>;

        async fn save_trained_net(&self, run_config_id: RunConfigId, path: PathBuf, loss: f32, input_net_ids: Option<Vec<u32>>) -> Result<u32, ThoError>;

        async fn list_trained_nets(&self, run_config_ids: TrainedNetsBy, loss: Option<f32>, limit: Option<usize>) -> Result<Vec<TrainedNet>, ThoError>;
        async fn list_trained_net_entries(&self) -> Result<Vec<TrainedNetEntry>, ThoError>;
        async fn list_label_trained_nets_before(&self, run_on_before: String) -> Result<Vec<TrainedNetEntry>, ThoError>;
        async fn list_trained_nets_loss_gt(&self, loss: f32, net_type: &'static str) -> Result<Vec<TrainedNetEntry>, ThoError>;
        async fn list_unused_motif_trained_nets(&self) -> Result<Vec<TrainedNetEntry>, ThoError>;
        async fn delete_trained_nets(&self, net_ids: Vec<u32>) -> Result<usize, ThoError>;
        async fn update_run_config_run_on(&self, run_config_id: RunConfigId, run_on: String) -> Result<(), ThoError>;
        async fn list_input_nets(&self, net_id: u32) -> Result<Vec<u32>, ThoError>;

        async fn run_config_descr(&self, corr_run_config_id: RunConfigId) -> Result<RunConfigRow, ThoError>;
        // #[cfg(test)]
        // fn conn(&self) -> &parking_lot::Mutex<Connection>; //parking_lot:: for mockall
    }

    #[async_trait]
    impl ResourceStore for MotifsStore {
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
            limit: u32,
        ) -> Result<Vec<Resource>, ThoError>;

        async fn list_random_resources(
            &self,
            limit: Option<u32>,
        ) -> Result<Vec<Resource>, ThoError>;
    }

}

impl fmt::Debug for MockMotifsStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MockStore").finish()
    }
}
