use async_trait::async_trait;
use std::path::{Path, PathBuf};
use thoosa_common::stages::export::{ExportStore, ExportWorkspace};
use thoosa_common::ws::WsConfigs;
use thoosa_shared::ThoError;
use thoosa_shared::run_configs::RunConfigId;

use crate::motifs_store::MotifsStore;
use crate::motifs_ws::MotifsWorkspace;
use crate::stages::convert_precision::ConvertPrecisionNetType;

impl ExportWorkspace for dyn MotifsWorkspace {
    type ModelType = ConvertPrecisionNetType;

    fn create_export_workspace(target_dir: &Path, configs: WsConfigs) -> anyhow::Result<Box<Self>> {
        crate::motifs_ws::new_motifs_workspace(target_dir, configs)
    }

    fn exported_model_type() -> Self::ModelType {
        ConvertPrecisionNetType::Labels
    }

    fn dependency_model_type() -> Self::ModelType {
        ConvertPrecisionNetType::Motifs
    }

    fn all_model_type() -> Self::ModelType {
        ConvertPrecisionNetType::All
    }
}

#[async_trait]
impl ExportStore for dyn MotifsStore {
    async fn list_export_groups(&self) -> anyhow::Result<Vec<String>> {
        // Labels define the independently exportable model groups in the motif workflow.
        MotifsStore::list_labels(self, true).await
    }

    async fn list_exported_model_run_config_ids(&self, group: &str) -> anyhow::Result<Vec<RunConfigId>> {
        MotifsStore::list_label_nets_run_config_ids(self, group).await
    }

    async fn export_model_records(&self, target_db_path: PathBuf, run_config_ids: Vec<RunConfigId>) -> anyhow::Result<()> {
        MotifsStore::export_apply_label_nets_records(self, target_db_path, run_config_ids).await
    }
}

pub use thoosa_common::stages::export::{ExportArgs, ExportInput};
pub type ExportOutput = thoosa_common::stages::export::ExportOutput<ConvertPrecisionNetType>;
pub type Export<C> = thoosa_common::stages::export::Export<C, dyn MotifsWorkspace, dyn MotifsStore>;
