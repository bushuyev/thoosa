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
use thoosa_store::{CheckGroups, LimitBy, Store, SetCond, IdSlice, RunConfigEntry, TrainedNetsBy, RunConfigRow, RunConfigFilter, RunMotifEntry, TrainedNetEntry, TrainedNet, SqlStore};

#[async_trait]
/// Store boundary used by motif-search workflows while motif methods are migrated incrementally out of `Store`.
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

/// Builds the motif-specific mock so motif tests do not depend on the general `Store` mock boundary.
pub struct MockMotifsStoreBuilder {
    store: MockMotifsStore,
}

impl MockMotifsStoreBuilder {
    pub fn new() -> Self {
        Self {
            store: MockMotifsStore::new(),
        }
    }

    //run_config_id, motif_id, channel, slices
    pub fn with_add_slices_to_motif(mut self, inputs: impl IntoIterator<Item = (RunConfigId, u32, i64, HashSet<IdSlice>)> + Debug) -> Self {
        debug!("with_add_slices_to_motif: inputs={inputs:?}");
        for inps in inputs {
            self.store.expect_add_slices_to_motif().with(eq(inps.0), eq(inps.1), eq(inps.2), eq(inps.3)).times(1).return_const(Ok(()));
        }

        self
    }

    pub fn with_add_slices_to_motif_at_least_once(mut self) -> Self {
        self.store.expect_add_slices_to_motif().times(1..).return_const(Ok(()));

        self
    }

    pub fn with_get_motif_meta<R: Into<HashMap<u32, Result<(usize, RunConfigId), ThoError>>>>(mut self, results: R) -> Self {
        let map = results.into();
        self.store.expect_get_motif_meta().returning(move |motif_id| {
            map.get(&motif_id).cloned().expect(&format!("mock is not configured to return motif meta for motif_id {motif_id}"))
        });
        self
    }

    pub fn with_get_input_run_config_ids<R: Into<HashMap<RunConfigId, Result<Vec<RunConfigId>, ThoError>>>>(mut self, results: R) -> Self {
        let map = results.into();
        self.store.expect_get_input_run_config_ids().returning(move |run_config_id, _input_type| {
            map.get(&run_config_id).cloned().expect(&format!("mock is not configured to return input_run_config_ids for run_config_id {run_config_id}"))
        });
        self
    }

    pub fn with_run_config_descr<R: Into<HashMap<RunConfigId, Result<RunConfigRow, ThoError>>>>(mut self, results: R) -> Self {
        let map = results.into();
        self.store.expect_run_config_descr().returning(move |run_config_id| {
            map.get(&run_config_id).cloned().expect(&format!("mock is not configured to return run_config_descr for run_config_id {run_config_id}"))
        });
        self
    }

    pub fn with_save_run_config(
        mut self,
        input_run_config_id: Option<Vec<RunConfigId>>,
        run_config_type: &'static str,
        descr: &'static str,
        _keys: &'static str,
        properties_vals: HashMap<String, Val>,
        result: Result<RunConfigId, ThoError>,
    ) -> Self {
        self.store.expect_save_run_config().returning(move |p_input_run_config_id, p_run_config_type, p_descr, p_properties_vals| {
            let p_input_run_config_id = p_input_run_config_id.map(|ids| ids.to_vec());
            if p_input_run_config_id == input_run_config_id && p_run_config_type == run_config_type && p_descr == descr && p_properties_vals == properties_vals {
                result.clone()
            } else {
                Err(ThoError::new(format!("mock is not configured to save_run_config for input_run_config_id={p_input_run_config_id:?} run_config_type={p_run_config_type} descr={p_descr} properties_vals={p_properties_vals:?}")))
            }
        });
        self
    }

    pub fn with_list_motifs<R: Into<HashMap<RunConfigId, Result<Vec<u32>, ThoError>>>>(mut self, results: R) -> Self {
        let map = results.into();
        self.store.expect_list_motifs().returning(move |run_config_id| {
            map.get(&run_config_id).cloned().expect(&format!("mock is not configured to return motifs for run id {run_config_id}"))
        });

        self
    }

    pub fn with_list_motif_slices(
        mut self,
        motif_id: u32,
        select_motifs_run_config_id: Option<Vec<RunConfigId>>,
        _limit: Option<u32>,
        _all_but_in_run_config_id: Option<(RunConfigId, u32)>,
        r: Result<(i64, RunConfigId, Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>), ThoError>,
    ) -> Self {
        self.store.expect_list_motif_slices().withf(move |p_motif_id, p_select_motifs_run_config_id, _limit, _all_but_in_run_config_id| *p_motif_id == motif_id && *p_select_motifs_run_config_id == select_motifs_run_config_id).times(1).return_const(r);

        self
    }

    pub fn with_list_random_slices_out_of_motif(
        mut self,
        motif_id: u32,
        motif_width: u32,
        min_distance: Option<u32>,
        r: Result<(i64, RunConfigId, Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>), ThoError>,
    ) -> Self {
        self.store.expect_list_random_slices_out_of_motif().returning(move |p_motif_id, p_width, p_min_distance, _input_len_adjuster, _desired_count| {
            if p_motif_id == motif_id && p_width == motif_width && p_min_distance == min_distance {
                r.clone()
            } else {
                Err(ThoError::new("Not found"))
            }
        });

        self
    }

    pub fn with_list_selected_motifs<R: Into<HashMap<RunConfigId, Result<Vec<Vec<(u32, f64)>>, ThoError>>>>(mut self, results: R) -> Self {
        let map = results.into();
        self.store.expect_list_selected_motifs().returning(move |run_config_id, _channels| {
            map.get(&run_config_id).cloned().expect(&format!("mock is not configured to return selected motifs for run id {run_config_id}"))
        });
        self
    }

    pub fn with_save_trained_net(mut self, inputs: impl IntoIterator<Item = (RunConfigId, PathBuf, f32)> + Debug) -> Self {
        debug!("with_save_trained_net: inputs={inputs:?}");
        for (run_config_id, path, loss) in inputs {
            self.store.expect_save_trained_net().with(eq(run_config_id), eq(path), eq(loss), eq(None)).times(1).return_const(Ok(run_config_id));
        }
        self
    }

    pub fn with_list_trained_nets<I>(mut self, inputs: I) -> Self
    where
        I: IntoIterator<Item = (Vec<RunConfigId>, Option<f32>, Option<usize>, Result<Vec<TrainedNet>, ThoError>)> + Debug,
    {
        let data = inputs.into_iter().collect::<Vec<_>>();
        debug!("with_list_trained_nets: inputs={data:?}");
        let data = Arc::new(data);
        self.store.expect_list_trained_nets().returning(move |run_config_ids, loss, limit| {
            data.iter()
                .find(|(expected_run_config_ids, expected_loss, expected_limit, _)| expected_run_config_ids.as_slice() == run_config_ids.as_ref().as_slice() && *expected_loss == loss && *expected_limit == limit)
                .map(|(_, _, _, result)| result.clone())
                .unwrap_or_else(|| Err(ThoError::new(format!("mock is not configured to return trained nets for run ids {run_config_ids:?} loss={loss:?} limit={limit:?}"))))
        });
        self
    }

    pub fn with_list_input_nets<I>(mut self, inputs: I) -> Self
    where
        I: IntoIterator<Item = (RunConfigId, Result<Vec<u32>, ThoError>)> + Debug,
    {
        let data = inputs.into_iter().collect::<Vec<_>>();
        debug!("with_list_input_nets: inputs={data:?}");
        let data = Arc::new(data);
        self.store.expect_list_input_nets().returning(move |run_config_id| {
            data.iter()
                .find(|(expected_run_config_id, _)| expected_run_config_id == &run_config_id)
                .map(|(_, result)| result.clone())
                .unwrap_or_else(|| Err(ThoError::new(format!("with_list_input_nets is not configured to return nets for run_config_id={run_config_id:?}"))))
        });
        self
    }

    pub fn with_list_run_config_values<R: Into<HashMap<RunConfigId, Result<HashMap<String, Val>, ThoError>>>>(mut self, results: R) -> Self {
        let map = results.into();
        self.store.expect_list_run_config_values().returning(move |p_run_config_id| {
            map.get(&p_run_config_id).cloned().expect(&format!("mock is not configured to return run config values for run id {p_run_config_id}"))
        });

        self
    }

    pub fn with_list_output_run_config_ids<R: Into<HashMap<RunConfigId, Result<Vec<RunConfigId>, ThoError>>>>(mut self, results: R) -> Self {
        let map = results.into();
        self.store.expect_list_output_run_config_ids().returning(move |input_run_config_id| {
            map.get(&input_run_config_id).cloned().expect(&format!("mock is not configured to return output run ids for run id {input_run_config_id}"))
        });
        self
    }

    pub fn with_list_run_config_ids_with_outputs<R: Into<HashMap<Vec<RunConfigId>, Result<Vec<RunConfigId>, ThoError>>>>(mut self, results: R) -> Self {
        let map = results.into();
        self.store.expect_list_run_config_ids_with_outputs().returning(move |input_run_config_ids| {
            map.get(&input_run_config_ids.to_vec()).cloned().expect(&format!("mock is not configured to return run ids with outputs for input ids {input_run_config_ids:?}"))
        });
        self
    }

    pub fn with_list_resources_by_file_stems(mut self, expected_stems: Vec<String>, result: Result<Vec<Resource>, ThoError>) -> Self {
        self.store.expect_list_resources_by_file_stems().returning(move |requested| {
            let stems = requested.iter().map(|stem| stem.to_string()).collect::<Vec<_>>();
            if stems == expected_stems {
                result.clone()
            } else {
                Err(ThoError::new("Not found"))
            }
        });

        self
    }

    pub fn with_list_resource_set_group_sets<D: IntoIterator<Item = (&'static str, Result<Vec<(usize, String)>, ThoError>)>>(mut self, data: D) -> Self {
        let map: HashMap<&'static str, Result<Vec<(usize, String)>, ThoError>> = HashMap::from_iter(data);
        self.store.expect_list_resource_set_group_sets().returning(move |group| {
            map.get(group).expect(&format!("no mock data for {group} configured in list_group_labeled_resources_sets")).clone()
        });
        self
    }

    pub fn with_list_group_labeled_resources_sets<D: IntoIterator<Item = (&'static str, Result<Vec<(String, String)>, ThoError>)>>(mut self, data: D) -> Self {
        let map: HashMap<&'static str, Result<Vec<(String, String)>, ThoError>> = HashMap::from_iter(data);
        self.store.expect_list_group_labeled_resources_sets().returning(move |group| {
            map.get(group).expect(&format!("no mock data for {group} configured in list_group_labeled_resources_sets")).clone()
        });
        self
    }

    pub fn with_list_resources_not_in_group<D: IntoIterator<Item = (&'static str, Result<Vec<Resource>, ThoError>)>>(mut self, data: D) -> Self {
        let map: HashMap<&'static str, Result<Vec<Resource>, ThoError>> = HashMap::from_iter(data);
        self.store.expect_list_resources_not_in_group().returning(move |group, _limit| {
            map.get(group).expect(&format!("no mock data for {group} configured in list_resource_set_resources_by_descr")).clone()
        });
        self
    }

    pub fn with_list_random_resources<D: IntoIterator<Item = (Option<u32>, Result<Vec<Resource>, ThoError>)>>(mut self, data: D) -> Self {
        let map: HashMap<Option<u32>, Result<Vec<Resource>, ThoError>> = HashMap::from_iter(data);
        self.store.expect_list_random_resources().returning(move |limit| {
            map.get(&limit).expect(&format!("no mock data for limit={limit:?} configured in list_random_resources")).clone()
        });
        self
    }

    pub fn with_list_resource_random_not_labeled<D: IntoIterator<Item = (&'static str, Result<Vec<Resource>, ThoError>)>>(mut self, data: D) -> Self {
        let map: HashMap<&'static str, Result<Vec<Resource>, ThoError>> = HashMap::from_iter(data);
        self.store.expect_list_resource_random_not_labeled().returning(move |label, _limit| {
            map.get(label).expect(&format!("no mock data for {label} configured in list_resources_not_labled")).clone()
        });
        self
    }

    pub fn with_list_resource_random_in_group<D: IntoIterator<Item = (&'static str, Result<Vec<(Resource, Vec<f32>)>, ThoError>)>>(mut self, data: D) -> Self {
        let map: HashMap<&'static str, Result<Vec<(Resource, Vec<f32>)>, ThoError>> = HashMap::from_iter(data);
        self.store.expect_list_resource_random_in_group().returning(move |group| {
            map.get(group).expect(&format!("no mock data for {group} configured in list_resource_random_in_group")).clone()
        });
        self
    }

    pub fn with_list_resource_set_resources_by_descr<D: IntoIterator<Item = (&'static str, Result<Vec<Resource>, ThoError>)>>(mut self, data: D) -> Self {
        let map: HashMap<&'static str, Result<Vec<Resource>, ThoError>> = HashMap::from_iter(data);
        self.store.expect_list_resource_set_resources_by_descr().returning(move |group| {
            map.get(group).expect(&format!("no mock data for {group} configured in list_resource_set_resources_by_descr")).clone()
        });
        self
    }

    pub fn with_save_groups_as_motifs(mut self, mut check_groups: Vec<CheckGroups>) -> Self {
        let times = check_groups.len();
        check_groups.reverse();
        let i_atomic = AtomicUsize::new(0);
        self.store.expect_save_groups_as_motifs().withf(move |run_config_id, channel, groups| {
            let i = i_atomic.fetch_add(1, Relaxed);
            debug!("with_save_groups_as_motifs: run_config_id={run_config_id:?}, channel={channel:?}, groups={groups:?} i={i}");
            match &check_groups[i] {
                CheckGroups::Lengths(expected_groups_lengths) => {
                    let mut group_lens = groups.values().map(|group| group.len()).collect::<Vec<_>>();
                    group_lens.sort();
                    assert_eq!(&group_lens, expected_groups_lengths);
                }
                CheckGroups::MinGroups(num_groups) => assert!(num_groups <= &groups.len()),
            }
            true
        }).times(times).returning(|_, _, groups| Ok((0..groups.len()).map(|idx| idx as u32 + 1).collect()));
        self
    }

    pub fn with_save_groups_as_motifs_inputs(mut self, inputs: impl IntoIterator<Item = (RunConfigId, i64, HashMap<u16, HashSet<IdSlice>>, Vec<u32>)> + Debug) -> Self {
        for (run_config_id, channel, groups, motif_ids) in inputs {
            self.store.expect_save_groups_as_motifs().with(eq(run_config_id), eq(channel), eq(groups)).times(1).return_const(Ok(motif_ids));
        }
        self
    }

    pub fn with_save_selected_motifs(mut self, inputs: impl IntoIterator<Item = (RunConfigId, Vec<(u32, f64)>, Result<RunConfigId, ThoError>)> + Debug) -> Self {
        for (run_config_id, selected_motifs, result) in inputs {
            self.store.expect_save_selected_motifs().with(eq(run_config_id), eq(selected_motifs)).times(1).return_const(result);
        }
        self
    }

    pub fn build(self) -> Box<dyn MotifsStore> {
        Box::new(self.store)
    }
}
