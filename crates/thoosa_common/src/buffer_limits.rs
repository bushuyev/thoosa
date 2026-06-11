use std::sync::{Arc, LazyLock};

use tokio::sync::watch;

pub struct DynamicBufferLimit {
    sender: watch::Sender<usize>,
    receiver: Arc<watch::Receiver<usize>>,
}

impl DynamicBufferLimit {
    fn new(limit: usize) -> Self {
        assert!(limit > 0, "buffer limit must be greater than zero");
        let (sender, receiver) = watch::channel(limit);
        Self {
            sender,
            receiver: Arc::new(receiver),
        }
    }

    pub fn receiver(&self) -> Arc<watch::Receiver<usize>> {
        Arc::clone(&self.receiver)
    }

    pub fn set(&self, limit: usize) {
        assert!(limit > 0, "buffer limit must be greater than zero");
        self.sender.send_replace(limit);
    }

    pub fn current(&self) -> usize {
        *self.receiver.borrow()
    }
}

pub struct BufferLimits {
    pub find_motifs_stages_buffer_unordered_limit: DynamicBufferLimit,
    pub cross_loaders_limit: DynamicBufferLimit,
    pub bucketing_limit: DynamicBufferLimit,
    pub cli_run_plot_motif_slices_buffer_unordered_limit: DynamicBufferLimit,
    pub apply_motif_nets_new_buffered_limit: DynamicBufferLimit,
    pub apply_motif_nets_run_buffered_limit: DynamicBufferLimit,
    pub clc_resources_motifs_dist_output_to_cpu_buffered_limit: DynamicBufferLimit,
    pub train_label_nets_run_buffer_unordered_limit: DynamicBufferLimit,
    pub train_label_nets_resources_batch_load_buffered_limit: DynamicBufferLimit,
    pub load_stacked_motifs_limit: DynamicBufferLimit,
    pub logits_resources_batch_load_buffered_limit: DynamicBufferLimit,
    pub apply_label_nets_run_buffered_limit: DynamicBufferLimit,
    pub train_motif_nets_run_buffer_unordered_limit: DynamicBufferLimit,
    pub select_motifs_stage_run_configs: DynamicBufferLimit,
    pub select_motifs_stage_find_motifs_buffer_unordered_limit: DynamicBufferLimit,
    pub select_motifs_stage_good_motifs_buffer_unordered_limit: DynamicBufferLimit,
    pub stacked_motifs_loader_run: DynamicBufferLimit,
    pub find_and_select_motifs_run_buffer_ordered_limit: DynamicBufferLimit,
    pub mel_patch_opt_stage_run_buffer_unordered_limit: DynamicBufferLimit,
    pub convert_precision_buffer_limit: DynamicBufferLimit,
    pub purge_models_buffer_limit: DynamicBufferLimit,
    pub select_buckets_motifs_buffer_limit: DynamicBufferLimit,
    pub bird_clef_labels_buffer_limit: DynamicBufferLimit,
    pub copy_concurrency: DynamicBufferLimit,
}

impl BufferLimits {
    pub fn new() -> Self {
        Self {
            find_motifs_stages_buffer_unordered_limit: DynamicBufferLimit::new(8),
            cross_loaders_limit: DynamicBufferLimit::new(8),
            bucketing_limit: DynamicBufferLimit::new(8),
            cli_run_plot_motif_slices_buffer_unordered_limit: DynamicBufferLimit::new(8),
            apply_motif_nets_new_buffered_limit: DynamicBufferLimit::new(8),
            apply_motif_nets_run_buffered_limit: DynamicBufferLimit::new(8),
            clc_resources_motifs_dist_output_to_cpu_buffered_limit: DynamicBufferLimit::new(8),
            train_label_nets_run_buffer_unordered_limit: DynamicBufferLimit::new(6),
            train_label_nets_resources_batch_load_buffered_limit: DynamicBufferLimit::new(8),
            load_stacked_motifs_limit: DynamicBufferLimit::new(8),
            logits_resources_batch_load_buffered_limit: DynamicBufferLimit::new(8),
            apply_label_nets_run_buffered_limit: DynamicBufferLimit::new(8),
            train_motif_nets_run_buffer_unordered_limit: DynamicBufferLimit::new(8),
            select_motifs_stage_run_configs: DynamicBufferLimit::new(8),
            select_motifs_stage_find_motifs_buffer_unordered_limit: DynamicBufferLimit::new(8),
            select_motifs_stage_good_motifs_buffer_unordered_limit: DynamicBufferLimit::new(8),
            stacked_motifs_loader_run: DynamicBufferLimit::new(8),
            find_and_select_motifs_run_buffer_ordered_limit: DynamicBufferLimit::new(8),
            mel_patch_opt_stage_run_buffer_unordered_limit: DynamicBufferLimit::new(8),
            convert_precision_buffer_limit: DynamicBufferLimit::new(4),
            purge_models_buffer_limit: DynamicBufferLimit::new(16),
            select_buckets_motifs_buffer_limit: DynamicBufferLimit::new(4),
            bird_clef_labels_buffer_limit: DynamicBufferLimit::new(16),
            copy_concurrency: DynamicBufferLimit::new(8),
        }
    }
}

impl Default for BufferLimits {
    fn default() -> Self {
        Self::new()
    }
}

pub static BUFFER_LIMITS: LazyLock<BufferLimits> = LazyLock::new(BufferLimits::new);

pub fn buffer_limits() -> &'static BufferLimits {
    &BUFFER_LIMITS
}
