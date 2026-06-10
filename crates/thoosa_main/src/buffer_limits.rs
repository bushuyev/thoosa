use std::sync::{Arc, OnceLock};
use tokio::sync::watch;
use thoosa_common::runner::global_ctx::GlobalCtx;

pub struct BufferLimits {
    find_motifs_stages_buffer_unordered_limit_rs:tokio::sync::watch::Sender<usize>,
    find_motifs_stages_buffer_unordered_limit_tx:Arc<tokio::sync::watch::Receiver<usize>>
}

impl BufferLimits {
    pub fn new() -> Self {
        let (tx, rx ) = watch::channel(FIND_MOTIFS_STAGES_BUFFER_UNORDERED_LIMIT);

        Self { find_motifs_stages_buffer_unordered_limit_rs: tx, find_motifs_stages_buffer_unordered_limit_tx: Arc::new(rx) }
    }

    pub fn find_motifs_stages_buffer_unordered_limit_tx(&self) -> &Arc<tokio::sync::watch::Receiver<usize>> {
        &self.find_motifs_stages_buffer_unordered_limit_tx
    }
}

pub static BUFFER_LIMITS: OnceLock<BufferLimits> = OnceLock::new();

pub(crate) const FIND_MOTIFS_STAGES_BUFFER_UNORDERED_LIMIT: usize = 8;
pub(crate) const CROSS_LOADERS_LIMIT: usize = 8;
pub(crate) const BUCKETING_LIMIT: usize = 8;
pub(crate) const CLI_RUN_PLOT_MOTIF_SLICES_BUFFER_UNORDERED_LIMIT: usize = 8;
pub(crate) const APPLY_MOTIF_NETS_NEW_BUFFERED_LIMIT: usize = 8;
pub(crate) const APPLY_MOTIF_NETS_RUN_BUFFERED_LIMIT: usize = 8;
pub(crate) const CLC_RESOURCES_MOTIFS_DIST_OUTPUT_TO_CPU_BUFFERED_LIMIT: usize = 8;
pub(crate) const TRAIN_LABEL_NETS_RUN_BUFFER_UNORDERED_LIMIT: usize = 6;
pub(crate) const TRAIN_LABEL_NETS_RESOURCES_BATCH_LOAD_BUFFERED_LIMIT: usize = 8;
pub(crate) const LOAD_STACKED_MOTIFS_LIMIT: usize = 8;
pub(crate) const LOGITS_RESOURCES_BATCH_LOAD_BUFFERED_LIMIT: usize = 8;
pub(crate) const APPLY_LABEL_NETS_RUN_BUFFERED_LIMIT: usize = 8;
pub(crate) const TRAIN_MOTIF_NETS_RUN_BUFFER_UNORDERED_LIMIT: usize = 8;
pub(crate) const SELECT_MOTIFS_STAGE_RUN_CONFIGS: usize = 8;
pub(crate) const SELECT_MOTIFS_STAGE_FIND_MOTIFS_BUFFER_UNORDERED_LIMIT: usize = 8;
pub(crate) const SELECT_MOTIFS_STAGE_GOOD_MOTIFS_BUFFER_UNORDERED_LIMIT: usize = 8;
pub(crate) const STACKED_MOTIFS_LOADER_RUN: usize = 8;
pub(crate) const FIND_AND_SELECT_MOTIFS_RUN_BUFFER_ORDERED_LIMIT: usize = 8;
pub(crate) const MEL_PATCH_OPT_STAGE_RUN_BUFFER_UNORDERED_LIMIT: usize = 8;
