//! Worker endpoint composition belongs to the feature backend, independent of CLI commands.
use crate::stages::apply_label_nets::{APPLY_LABEL_NET_ENDPOINT, STACK_APPLIED_LABEL_NETS_ENDPOINT, apply_label_net_endpoint_init, stack_applied_label_nets_init};
use crate::stages::apply_motif_nets::{APPLY_MOTIF_NET_ENDPOINT, STACK_APPLIED_MOTIF_NETS_ENDPOINT, apply_motif_net_endpoint_init, stack_applied_motif_nets_init};
use crate::stages::apply_motif_nets::cosine_group::{APPLY_COSINE_GROUP_ENDPOINT, apply_cosine_group_init};
use crate::stages::cross_batch_clc::{CROSS_UNFOLD_CLC_ENDPOINT, sync_cross_unfold_clc_endpoint_init};
use crate::stages::motif_stats::{MOTIF_STATS_CONSUME_ENDPOINT, MOTIF_STATS_SYNC_ENDPOINT, motif_stats_consume_stage_endpoint_init, motif_stats_sync_stage_endpoint_init};
use crate::stages::prepare_cross_batch::{PREPARE_CROSS_BATCH_ENDPOINT, prepare_cross_batch_endpoint_init};
use crate::stages::train_final_net::{SYNC_LOGITS_BATCH_ENDPOINT, sync_logits_batch_init, train_final_step_endpoint_inits};
use crate::stages::train_label_nets::{SYNC_LABEL_BATCH_ENDPOINT, sync_label_batch_endpoint_init, train_label_step_endpoint_inits};
use crate::stages::train_motif_nets::train_motif_step_endpoint_inits;
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use parking_lot::RwLock;
use std::any::Any;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use thoosa_common::ipc::cuda_worker::{AllTensors, InitFn, WokerRunCtx};
use thoosa_common::ipc::serialization::json::JsonIpcCodec;
use thoosa_common::ipc::worker_msg::WorkerInput;
use thoosa_common::ipc::wraps::SyncStageIpcWrapper;
use thoosa_common::ipc::wraps::WorkerWrapperWithState;
use thoosa_common::stages::freq_filter::{SYNC_FREQ_FILTER_ENDPOINT, SyncFreqFilter, SyncFreqFilterIpcInit};
use thoosa_common::tho_device::ThoDevice;
use thoosa_shared::ThoError;

type MotifsInitFn = InitFn<dyn crate::motifs_ws::MotifsWorkspace>;

pub fn sync_freq_filter_endpoint_init<'a>(_stage_id: Option<String>, init: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, _all_cuda_tensors: &'a AllTensors, _tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let init: Box<dyn Any> = init;
        let Ok(init) = init.downcast::<SyncFreqFilterIpcInit>() else {
            return Err(ThoError::new("SyncFreqFilter endpoint received an unexpected initializer type"));
        };
        let init = *init;
        let stage = SyncFreqFilter::<WokerRunCtx>::new_local(SyncFreqFilterIpcInit::new(init.filter_configs.clone(), init.num_channels, init.sample_rate, init.find_motifs_run_config_id, 1))?;

        // The common stateful wrapper prepares filter tensors on the worker device before each synchronous invocation.
        Ok(Box::new(WorkerWrapperWithState::new(Arc::new(stage), PhantomData::<JsonIpcCodec>)) as Box<dyn SyncStageIpcWrapper>)
    }
    .boxed()
}

pub fn cuda_worker_endpoint_inits() -> HashMap<String, MotifsInitFn> {
    let mut endpoints = train_motif_step_endpoint_inits();
    // Label and final training now execute through the same worker-owned proxy lifecycle as motif training.
    endpoints.extend(train_label_step_endpoint_inits());
    endpoints.extend(train_final_step_endpoint_inits());
    endpoints.extend([
        (PREPARE_CROSS_BATCH_ENDPOINT.to_string(), Box::new(prepare_cross_batch_endpoint_init) as MotifsInitFn),
        (CROSS_UNFOLD_CLC_ENDPOINT.to_string(), Box::new(sync_cross_unfold_clc_endpoint_init) as MotifsInitFn),
        (MOTIF_STATS_SYNC_ENDPOINT.to_string(), Box::new(motif_stats_sync_stage_endpoint_init) as MotifsInitFn),
        (MOTIF_STATS_CONSUME_ENDPOINT.to_string(), Box::new(motif_stats_consume_stage_endpoint_init) as MotifsInitFn),
        (SYNC_FREQ_FILTER_ENDPOINT.to_string(), Box::new(sync_freq_filter_endpoint_init) as MotifsInitFn),
        (APPLY_MOTIF_NET_ENDPOINT.to_string(), Box::new(apply_motif_net_endpoint_init) as MotifsInitFn),
        (APPLY_COSINE_GROUP_ENDPOINT.to_string(), Box::new(apply_cosine_group_init) as MotifsInitFn),
        (APPLY_LABEL_NET_ENDPOINT.to_string(), Box::new(apply_label_net_endpoint_init) as MotifsInitFn),
        (STACK_APPLIED_MOTIF_NETS_ENDPOINT.to_string(), Box::new(stack_applied_motif_nets_init) as MotifsInitFn),
        (STACK_APPLIED_LABEL_NETS_ENDPOINT.to_string(), Box::new(stack_applied_label_nets_init) as MotifsInitFn),
        (SYNC_LABEL_BATCH_ENDPOINT.to_string(), Box::new(sync_label_batch_endpoint_init) as MotifsInitFn),
        (SYNC_LOGITS_BATCH_ENDPOINT.to_string(), Box::new(sync_logits_batch_init) as MotifsInitFn),
    ]);
    endpoints
}
