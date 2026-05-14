#![feature(backtrace_frames)]
#![feature(unboxed_closures)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![allow(confusable_idents)]
#![allow(deprecated)]

extern crate self as thoosa_shared;

use crate::hyperparameter::{FieldName, Hyperparameter};
use crate::collect_futures_eta::{collect_eta_maybe_log, collect_eta_record_completion, CollectEtaInvocationGuard, COLLECT_ETA_LOG_INTERVAL};
use crate::resource::ResourceId;
use crate::val::{GridValueSet, NamedRange, NamedVal, PendingOptField, Val};
use bytemuck::{Pod, Zeroable};
use chrono::SecondsFormat;
use lazy_static::lazy_static;
use log::{debug, error, trace, warn};
use num_traits::PrimInt;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use rand::{Rng, RngExt};
use regex::Regex;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::backtrace::Backtrace;
use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::f32::consts::PI;
use std::fmt::{Debug, Display, Formatter};
use std::fs::read_dir;
use std::hash::Hash;
use std::ops::{Mul, Range};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::{fmt, fs};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use futures::stream::StreamExt;
use async_trait::async_trait;
use ambassador::delegatable_trait;
use mockall::automock;
use rand::seq::{IndexedRandom, SliceRandom};
#[cfg(test)]
use rand::rngs::StdRng;
#[cfg(test)]
use rand::SeedableRng;


pub mod resource;
pub mod val;

pub mod applicables_runner;
pub mod colors_iter;
pub mod hyperparameter;
pub mod opt_rng;pub mod run_configs;
pub mod test_utils;
pub mod traits;
pub mod data_with_extra;
pub mod perf_logging;
pub mod copy_mode;
pub mod comp_id;
pub mod collect_futures_eta;
pub mod device_stats;
pub mod math;
pub mod audio_utils;
pub mod tmp_file;
pub mod macros;

struct GroupSlicesBuilder<T> {
    pairs_groups: HashMap<T, u16>,
    last_group: u16,
    separated_groups: Vec<(u16, u16)>,
}

impl<T> GroupSlicesBuilder<T>
where
    T: Hash + Eq + Debug,
{
    fn new() -> Self {
        Self {
            pairs_groups: HashMap::new(),
            last_group: 0u16,
            separated_groups: vec![],
        }
    }

    fn push(&mut self, ln: usize, pair: (T, T)) {
        let g_0 = self.pairs_groups.get(&pair.0);
        let g_1 = self.pairs_groups.get(&pair.1);
        if let Some(g_0_val) = g_0 {
            if g_1.is_some() {
                trace!("separated group: g_0={:?} g_1={:?} ln={} pair={:?}", g_0, g_1, ln, pair);
                if g_0.ne(&g_1) {
                    self.separated_groups.push((*g_0_val, *g_1.unwrap()));
                }
            } else {
                self.pairs_groups.insert(pair.1, *g_0_val);
            }
        } else {
            if let Some(g_1_val) = g_1 {
                self.pairs_groups.insert(pair.0, *g_1_val);
            } else {
                self.pairs_groups.insert(pair.0, self.last_group);
                self.pairs_groups.insert(pair.1, self.last_group);

                self.last_group += 1;
            }
        }
    }

    fn finish(self) -> HashMap<u16, HashSet<T>> {
        let GroupSlicesBuilder {
            pairs_groups,
            separated_groups,
            last_group: _,
        } = self;

        trace!("separated_groups={:?}", separated_groups);
        let separated_groups_map = if !separated_groups.is_empty() {
            let map = group_slices(separated_groups.into_iter()) //group separated groups same as slices
                .values()
                .fold(HashMap::new(), |mut h, v| {
                    let mut iter = v.iter();
                    if let Some(head) = iter.next().copied() {
                        for i in iter {
                            h.insert(*i, head);
                        }
                    }
                    h
                });
            trace!("separated_groups group_motifs={:?}", map);
            Some(map)
        } else {
            None
        };

        let mut groups: HashMap<u16, HashSet<T>> = HashMap::new();
        for (val, group) in pairs_groups {
            let group = separated_groups_map
                .as_ref()
                .and_then(|m| m.get(&group))
                .unwrap_or(&group); //replace separated group pairs with one in pair
            let gr_vec = groups.entry(*group).or_insert_with(|| HashSet::new());
            gr_vec.insert(val);
        }

        trace!("group_motifs groups={:?}", groups);
        groups
    }
}

//group of similar slices - motif
pub fn group_slices<T>(pairs: impl Iterator<Item = (T, T)>) -> HashMap<u16, HashSet<T>> //TODO replace with group_slices_try
where
    T: Hash + Eq + Debug,
{
    let mut builder = GroupSlicesBuilder::new();
    for (ln, pair) in pairs.enumerate() {
        trace!("group_slices: ln={ln} pairs={pair:?}");
        builder.push(ln, pair);
    }
    builder.finish()
}

pub fn group_slices_try<T, E>(
    pairs: impl Iterator<Item = Result<(T, T), E>>,
) -> Result<HashMap<u16, HashSet<T>>, E>
where
    T: Hash + Eq + Debug,
{
    let mut builder = GroupSlicesBuilder::new();
    for (ln, pair) in pairs.enumerate() {
        let pair = pair?;
        trace!("group_slices: ln={ln} pairs={pair:?}");
        builder.push(ln, pair);
    }
    Ok(builder.finish())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamBuffering {
    Unordered,
    Ordered,
}

#[derive(Debug, Clone, Copy)]
pub struct CudaActivityPauseTicket {
    generation: u64,
    owner: bool,
}

impl CudaActivityPauseTicket {
    pub fn is_owner(self) -> bool {
        self.owner
    }
}

#[derive(Debug, Default)]
struct CudaActivityGateState {
    paused: bool,
    generation: u64,
}

#[derive(Debug, Default)]
pub struct CudaActivityGate {
    state: Mutex<CudaActivityGateState>,
    notify: tokio::sync::Notify,
}

impl CudaActivityGate {
    fn new() -> Self {
        Self::default()
    }

    pub fn is_paused(&self) -> bool {
        self.state.lock().paused
    }

    pub fn request_pause(&self) -> CudaActivityPauseTicket {
        debug!("CudaActivityGate::request_pause: self={self:?}");
        let mut state = self.state.lock();
        let ticket = CudaActivityPauseTicket {
            generation: state.generation,
            owner: !state.paused,
        };

        if !state.paused {
            state.paused = true;
            self.notify.notify_waiters();
        }

        ticket
    }

    pub fn finish_pause(&self, ticket: CudaActivityPauseTicket) {
        debug!("CudaActivityGate::finish_pause: self={self:?}");
        if !ticket.owner {
            return;
        }

        let mut state = self.state.lock();
        if state.paused && state.generation == ticket.generation {
            state.paused = false;
            state.generation += 1;
            self.notify.notify_waiters();
        }
    }

    pub async fn wait_if_paused(&self) {
        debug!("CudaActivityGate::wait_if_paused: self={self:?}");
        loop {
            let notified = {
                let state = self.state.lock();
                if !state.paused {
                    return;
                }
                self.notify.notified()
            };
            notified.await;
        }
    }

    pub async fn wait_for_pause_to_finish(&self, ticket: CudaActivityPauseTicket) {
        debug!("CudaActivityGate::wait_for_pause_to_finish: self={self:?}");
        loop {
            let notified = {
                let state = self.state.lock();
                if !state.paused || state.generation != ticket.generation {
                    return;
                }
                self.notify.notified()
            };
            notified.await;
        }
    }

    pub async fn wait_until_paused(&self) {
        debug!("CudaActivityGate::wait_until_paused: self={self:?}");
        loop {
            let notified = {
                let state = self.state.lock();
                if state.paused {
                    return;
                }
                self.notify.notified()
            };
            notified.await;
        }
    }
}

static CUDA_ACTIVITY_GATE: Lazy<CudaActivityGate> = Lazy::new(CudaActivityGate::new);

pub fn cuda_activity_gate() -> &'static CudaActivityGate {
    &CUDA_ACTIVITY_GATE
}

async fn collect_stream_with_progress<T, S>(
    mut stream: S,
    results: &mut Vec<T>,
    ok: &mut usize,
    err: &mut usize,
    total: usize,
    descr: &str,
    invocation_id: Option<u64>,
) where
    T:Debug,
    S: futures::stream::Stream<Item = Result<T, ThoError>> + Unpin,
{

    let mut eta_tick = tokio::time::interval(COLLECT_ETA_LOG_INTERVAL);
    eta_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let _ = eta_tick.tick().await;


    loop {
        cuda_activity_gate().wait_if_paused().await;

        tokio::select! {
            biased;
            _ = cuda_activity_gate().wait_until_paused(), if !cuda_activity_gate().is_paused() => {
                cuda_activity_gate().wait_if_paused().await;
            }
            _ = eta_tick.tick() => {
                collect_eta_maybe_log(false);
            }
            next = stream.next() => {
                let Some(r) = next else {
                    trace!("collect_stream_with_progress: {descr}: next={next:?}");

                    break;
                };

                if let Some(invocation_id) = invocation_id {
                    collect_eta_record_completion(invocation_id);
                }

                match r {
                    Ok(item) => {
                        *ok += 1;
                        debug!("collect_stream_with_progress: {descr} done {} ({}/{}) out of {}", *ok + *err, *ok, *err, total);
                        results.push(item);
                    }
                    Err(e) => {
                        *err += 1;
                        if e.is_oom() {
                            warn!("collect_stream_with_progress: {descr}: got oom {e:?}");
                        } else {
                            error!("collect_stream_with_progress: {descr}: {e:?}");
                        }
                    }
                }
            }
        }
    }

    debug!("collect_stream_with_progress: {descr} all done ");
}

pub const LOCK_TIMEOUT: Duration = Duration::from_mins(1);
pub const FUTURES_TIMEOUT: Duration = Duration::from_hours(240);//TODO should depend on future



struct AggPerf<'a>(&'a str, &'a str);
impl std::fmt::Display for AggPerf<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.0, self.1)
    }
}


#[tracing::instrument(level="debug", skip_all, fields(agg_perf = %AggPerf("sync_lock", descr)))]
#[inline]
pub fn with_perf_log<F, R>(f:F, descr:&str) -> R
where
    F:FnOnce()->R {

    f()
}


#[tracing::instrument(level="debug", skip_all, fields(agg_perf = %AggPerf("async_lock", descr)))]
#[inline]
pub async fn with_async_perf_log<F, R>(f:F, descr:&str) -> R
where
    F:AsyncFnOnce()->R {
    trace!("with_async_perf_log: descr={descr}");
    let r = f().await;
    trace!("with_async_perf_log: descr={descr} done");

    r
}

pub async fn collect_futures_with_progress<T, F>(
    futures: impl Iterator<Item=F>,
    len:usize,
    buffer_limit: usize,
    buffering: StreamBuffering,
    descr: &str,
) -> Vec<T>
where
    T:Debug,
    F: Future<Output = Result<T, ThoError>>,
{
    debug!("collect_futures_with_progress: descr={descr} len={len}");

    let mut results = Vec::with_capacity(len);
    let (mut ok, mut err) = (0usize, 0usize);
    let total = len;
    let eta_concurrency = buffer_limit.max(1).min(total.max(1));
    let eta_guard = (total > 0).then(|| CollectEtaInvocationGuard::new(descr, total, eta_concurrency));
    let eta_invocation_id = eta_guard.as_ref().map(CollectEtaInvocationGuard::id);


    match buffering {
        StreamBuffering::Unordered => {
            let mut stream = futures::stream::iter(futures)
            .map(|fut| async move {
                tokio::time::timeout(FUTURES_TIMEOUT, fut).await? //TODO add param
            })
            .buffer_unordered(buffer_limit);

            collect_stream_with_progress(
                &mut stream,
                &mut results,
                &mut ok,
                &mut err,
                total,
                descr,
                eta_invocation_id,
            )
            .await;
        }
        StreamBuffering::Ordered => {
            let mut stream = futures::stream::iter(futures)
                .map(|fut| async move {
                    tokio::time::timeout(FUTURES_TIMEOUT, fut).await?
                })
                .buffered(buffer_limit);

            collect_stream_with_progress(
                &mut stream,
                &mut results,
                &mut ok,
                &mut err,
                total,
                descr,
                eta_invocation_id,
            )
            .await;
        }
    }

    drop(eta_guard);
    if eta_invocation_id.is_some() {
        collect_eta_maybe_log(true);
    }

    results
}

#[repr(C)]
#[derive(Debug, PartialEq, Eq, Clone, Copy, PartialOrd, Ord, Pod, Zeroable, Hash)]
pub struct TensorIdSliceIdx(pub ResourceId, pub u32); //TODO named fields

impl Display for TensorIdSliceIdx {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{},{}", self.0, self.1)
    }
}

#[derive(Debug, Clone)]
pub enum CudaAction {
    NeedsRestart,
    HandleOOM,
    Other //wrong index, etc
}

#[derive(Debug, Clone)]
pub enum Cause {
    Cuda { action: CudaAction, msg: String },
    AdjusterFailed,
    UnsupportedDegradeLevel,
    ExpectedValue,
    ExpectedRange,
    InvalidRange { range: String },
    RegexParse(regex::Error),
    CliParse(Arc<clap::Error>),
    External {
        type_name: &'static str,
        source: Arc<dyn Error + Send + Sync>,
    },
    SomethingElse(String),
    NotFound,
    SubmitErrorClosed,
    SubmitErrorCancelled,
    JobError{job_id:u64, inner:Box<ThoosaError>},
    ContextReset
}

pub struct ThoError {
    //TODO enum
    pub msg: String,
    #[allow(dead_code)]
    backtrace: Backtrace,
    pub what: Cause,
}

pub type ThoosaError = ThoError;//TODO used in macros, remove

impl Clone for ThoError {
    fn clone(&self) -> Self {
        ThoError {
            msg: self.msg.clone(),
            backtrace: Backtrace::capture(),
            what: self.what.clone(),
        }
    }
}

impl Display for ThoError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "ThoError: {}, cause={:?}", self.msg, self.what)
    }
}

unsafe impl Send for ThoError {}

impl ThoError {
    pub fn new(msg: impl Into<String> + Display) -> Self {
        let msg = msg.into();
        let msg_lc = msg.to_lowercase();

        if msg_lc.contains("cuda") {
            Self::from_cuda_message(msg_lc)
        } else {
            Self::new_with_cause("Unknown error", Cause::SomethingElse(msg))
        }
    }

    pub fn from_cuda_message(msg: impl Into<String> + Display) -> Self {
        Self::new_with_cause("Cuda error", cuda_cause_from_msg(msg))
    }

    pub fn from_cuda_error(_msg: impl Into<String> + Display, cuda_error:i32) -> Self {
        Self::new_with_cause("Cuda error", cuda_cause_from_error_code(cuda_error))
    }

    pub fn new_with_cause(msg: impl Into<String> + Display, what: Cause) -> Self {
        debug!("ThoosaError::new: msg={msg}");

        let backtrace = Backtrace::capture();
        trace!("msg={msg} {}", backtrace.to_string());

        Self {
            msg: msg.into(),
            backtrace,
            what,
        }
    }

    // #[cfg(test)] - used from other crates
    pub fn oom() -> Self {
        debug!("ThoosaError::new: oom");
        let backtrace = Backtrace::capture();

        Self {
            msg: "oom".into(),
            backtrace,
            what: Cause::Cuda {
                action: CudaAction::HandleOOM,
                msg: "test".to_string(),
            },
        }
    }

    pub fn is_oom(&self) -> bool {
        match &self.what{
            Cause::Cuda { action, msg: _ } => {
                match action {
                    CudaAction::HandleOOM => true,
                    _=> false
                }
            }
            Cause::JobError { job_id: _, inner } => {
                inner.is_oom()
            },
            _=> false
        }

    }


    pub fn is_retryable_oom(&self) -> bool {
        match &self.what{
            Cause::Cuda { action, msg: _ } => {
                match action {
                    CudaAction::HandleOOM => true,
                    _=> false
                }
            }
            Cause::AdjusterFailed => true,
            Cause::JobError { job_id: _, inner } => {
                inner.is_retryable_oom()
            },
            _=> false
        }
    }

    pub fn needs_cuda_ctx_reset(&self) -> bool {
        match &self.what {
            Cause::Cuda { action, .. } => {
                match action {
                    CudaAction::NeedsRestart => true,
                    _=> false
                }
            },

            _=> false
        }
    }

    pub fn what(&self) -> &Cause {
        &self.what
    }

    #[deprecated] //it's not checked for cuda/oom. Use into()
    pub fn from_error<E>(msg: impl Into<String> + Display, err: E) -> Self
    where
        E: Error + Send + Sync + 'static,
    {
        ThoError::new_with_cause(
            msg,
            Cause::External {
                type_name: std::any::type_name::<E>(),
                source: Arc::new(err),
            },
        )
    }


    pub fn from_msg(msg: impl Into<String> + Display) -> Self
    {
        ThoError::new(msg)
    }


    #[deprecated] //it's not checked for cuda/oom. Use into()
    pub fn from_boxed_error(
        msg: impl Into<String> + Display,
        err: Box<dyn Error + Send + Sync + 'static>,
    ) -> Self {
        ThoError::new_with_cause(
            msg,
            Cause::External {
                type_name: std::any::type_name_of_val(err.as_ref()),
                source: Arc::from(err),
            },
        )
    }
}

fn cuda_cause(action: CudaAction, msg: impl Into<String>) -> Cause {
    Cause::Cuda {
        action,
        msg: msg.into(),
    }
}

/// Classifies known CUDA Runtime API `cudaError_t` and CUDA Driver API `CUresult`
/// numeric codes.
///
/// Important:
/// - CUDA Runtime and Driver APIs share many numeric values, e.g. OOM = 2,
///   illegal address = 700.
/// - libtorch usually reports Runtime API text, but if you get raw codes from
///   `/usr/local/cuda/include/cuda.h`, those are Driver API `CUresult` codes.
pub fn cuda_cause_from_error_code(code: i32) -> Cause {
    match code {
        // --------------------------------------------------------------------
        // Recoverable / mitigation path.
        // Runtime: cudaErrorMemoryAllocation = 2
        // Driver:  CUDA_ERROR_OUT_OF_MEMORY = 2
        // --------------------------------------------------------------------
        2 => cuda_cause(
            CudaAction::HandleOOM,
            "CUDA error 2: cudaErrorMemoryAllocation / CUDA_ERROR_OUT_OF_MEMORY: \
             out of memory; try dropping tensors, emptying cache, reducing batch/concurrency",
        ),

        // --------------------------------------------------------------------
        // Strong restart candidates: sticky/context-poisoning/device exception.
        // These are the important ones for libtorch worker restart policy.
        // --------------------------------------------------------------------

        // Runtime: cudaErrorECCUncorrectable = 214
        // Driver:  CUDA_ERROR_ECC_UNCORRECTABLE = 214
        214 => cuda_cause(
            CudaAction::NeedsRestart,
            "CUDA error 214: cudaErrorECCUncorrectable / CUDA_ERROR_ECC_UNCORRECTABLE: \
             uncorrectable ECC error; treat as fatal for the worker/GPU path",
        ),

        // Runtime: cudaErrorNvlinkUncorrectable = 220
        // Driver:  CUDA_ERROR_NVLINK_UNCORRECTABLE = 220
        220 => cuda_cause(
            CudaAction::NeedsRestart,
            "CUDA error 220: cudaErrorNvlinkUncorrectable / CUDA_ERROR_NVLINK_UNCORRECTABLE: \
             uncorrectable NVLink error; treat as fatal for the worker/GPU path",
        ),

        // Runtime: cudaErrorContained = 226
        // Usually indicates GPU error containment; CUDA docs describe this as
        // leaving the process inconsistent.
        226 => cuda_cause(
            CudaAction::NeedsRestart,
            "CUDA error 226: cudaErrorContained: contained device exception; \
             process state is inconsistent, restart worker",
        ),

        // Runtime: cudaErrorIllegalAddress = 700
        // Driver:  CUDA_ERROR_ILLEGAL_ADDRESS = 700
        700 => cuda_cause(
            CudaAction::NeedsRestart,
            "CUDA error 700: cudaErrorIllegalAddress / CUDA_ERROR_ILLEGAL_ADDRESS: \
             illegal memory access; CUDA context/process is likely poisoned",
        ),

        // Runtime: cudaErrorLaunchTimeout = 702
        // Driver:  CUDA_ERROR_LAUNCH_TIMEOUT = 702
        702 => cuda_cause(
            CudaAction::NeedsRestart,
            "CUDA error 702: cudaErrorLaunchTimeout / CUDA_ERROR_LAUNCH_TIMEOUT: \
             kernel launch timed out and was terminated; restart worker",
        ),

        // Runtime: cudaErrorContextIsDestroyed = 709
        // Driver:  CUDA_ERROR_CONTEXT_IS_DESTROYED = 709
        709 => cuda_cause(
            CudaAction::NeedsRestart,
            "CUDA error 709: cudaErrorContextIsDestroyed / CUDA_ERROR_CONTEXT_IS_DESTROYED: \
             CUDA context was destroyed or invalid; libtorch state may hold stale CUDA objects",
        ),

        // Runtime: cudaErrorAssert = 710
        // Driver:  CUDA_ERROR_ASSERT = 710
        710 => cuda_cause(
            CudaAction::NeedsRestart,
            "CUDA error 710: cudaErrorAssert / CUDA_ERROR_ASSERT: \
             device-side assert triggered; restart worker",
        ),

        // Runtime: cudaErrorHardwareStackError = 714
        // Driver:  CUDA_ERROR_HARDWARE_STACK_ERROR = 714
        714 => cuda_cause(
            CudaAction::NeedsRestart,
            "CUDA error 714: cudaErrorHardwareStackError / CUDA_ERROR_HARDWARE_STACK_ERROR: \
             hardware stack error; restart worker",
        ),

        // Runtime: cudaErrorIllegalInstruction = 715
        // Driver:  CUDA_ERROR_ILLEGAL_INSTRUCTION = 715
        715 => cuda_cause(
            CudaAction::NeedsRestart,
            "CUDA error 715: cudaErrorIllegalInstruction / CUDA_ERROR_ILLEGAL_INSTRUCTION: \
             illegal GPU instruction; restart worker",
        ),

        // Runtime: cudaErrorMisalignedAddress = 716
        // Driver:  CUDA_ERROR_MISALIGNED_ADDRESS = 716
        716 => cuda_cause(
            CudaAction::NeedsRestart,
            "CUDA error 716: cudaErrorMisalignedAddress / CUDA_ERROR_MISALIGNED_ADDRESS: \
             misaligned GPU memory access; restart worker",
        ),

        // Runtime: cudaErrorInvalidAddressSpace = 717
        // Driver:  CUDA_ERROR_INVALID_ADDRESS_SPACE = 717
        717 => cuda_cause(
            CudaAction::NeedsRestart,
            "CUDA error 717: cudaErrorInvalidAddressSpace / CUDA_ERROR_INVALID_ADDRESS_SPACE: \
             invalid address space in kernel; restart worker",
        ),

        // Runtime: cudaErrorInvalidPc = 718
        // Driver:  CUDA_ERROR_INVALID_PC = 718
        718 => cuda_cause(
            CudaAction::NeedsRestart,
            "CUDA error 718: cudaErrorInvalidPc / CUDA_ERROR_INVALID_PC: \
             invalid program counter; restart worker",
        ),

        // Runtime: cudaErrorLaunchFailure = 719
        // Driver:  CUDA_ERROR_LAUNCH_FAILED = 719
        719 => cuda_cause(
            CudaAction::NeedsRestart,
            "CUDA error 719: cudaErrorLaunchFailure / CUDA_ERROR_LAUNCH_FAILED: \
             unspecified launch failure; restart worker",
        ),

        // Runtime: cudaErrorTensorMemoryLeak = 721
        721 => cuda_cause(
            CudaAction::NeedsRestart,
            "CUDA error 721: cudaErrorTensorMemoryLeak: \
             tensor memory leak detected by device runtime; restart worker",
        ),

        // Runtime: cudaErrorMpsClientTerminated = 810
        // Driver:  CUDA_ERROR_MPS_CLIENT_TERMINATED = 810
        810 => cuda_cause(
            CudaAction::NeedsRestart,
            "CUDA error 810: cudaErrorMpsClientTerminated / CUDA_ERROR_MPS_CLIENT_TERMINATED: \
             MPS client terminated; restart worker",
        ),

        // Runtime: cudaErrorExternalDevice = 911
        911 => cuda_cause(
            CudaAction::NeedsRestart,
            "CUDA error 911: cudaErrorExternalDevice: \
             external device error; async case should be treated as fatal for the worker",
        ),

        // Runtime: cudaErrorUnknown = 999
        // Driver:  CUDA_ERROR_UNKNOWN = 999
        999 => cuda_cause(
            CudaAction::NeedsRestart,
            "CUDA error 999: cudaErrorUnknown / CUDA_ERROR_UNKNOWN: \
             unknown CUDA error; safest policy is to restart worker",
        ),

        // --------------------------------------------------------------------
        // Non-OOM, usually code/config/input bugs or environment failures.
        // These should not be “fixed” by reducing concurrency.
        // They are classified as Other because they do not necessarily mean
        // CUDA context poisoning.
        // --------------------------------------------------------------------

        // Runtime: cudaErrorInvalidValue = 1
        // Driver:  CUDA_ERROR_INVALID_VALUE = 1
        1 => cuda_cause(
            CudaAction::Other,
            "CUDA error 1: invalid value; likely API misuse, bad argument, bad shape/index, or stale handle",
        ),

        // Runtime: cudaErrorInitializationError = 3
        // Driver:  CUDA_ERROR_NOT_INITIALIZED = 3
        3 => cuda_cause(
            CudaAction::Other,
            "CUDA error 3: initialization/not-initialized error; check driver/runtime/context initialization",
        ),

        // Runtime: cudaErrorCudartUnloading = 4
        // Driver:  CUDA_ERROR_DEINITIALIZED = 4
        4 => cuda_cause(
            CudaAction::Other,
            "CUDA error 4: CUDA runtime/driver is deinitializing; usually process shutdown or teardown ordering",
        ),

        // Runtime: cudaErrorInvalidDevice = 101
        // Driver:  CUDA_ERROR_INVALID_DEVICE = 101
        101 => cuda_cause(
            CudaAction::Other,
            "CUDA error 101: invalid device; check selected CUDA device index",
        ),

        // Runtime: cudaErrorInvalidResourceHandle = 400
        // Driver:  CUDA_ERROR_INVALID_HANDLE = 400
        400 => cuda_cause(
            CudaAction::Other,
            "CUDA error 400: invalid resource/handle; possible stale stream/event/library handle or API misuse",
        ),

        // Runtime: cudaErrorNotReady = 600
        // Driver:  CUDA_ERROR_NOT_READY = 600
        600 => cuda_cause(
            CudaAction::Other,
            "CUDA error 600: not ready; async operation has not completed yet",
        ),

        // Runtime: cudaErrorIllegalState = 401 in newer CUDA runtimes.
        401 => cuda_cause(
            CudaAction::Other,
            "CUDA error 401: illegal state; usually API sequencing/capture/state misuse",
        ),

        // Runtime: cudaErrorInvalidConfiguration = 9
        9 => cuda_cause(
            CudaAction::Other,
            "CUDA error 9: invalid kernel launch configuration; fix launch/grid/block/shared-memory parameters",
        ),

        // Runtime: cudaErrorLaunchOutOfResources = 701
        // Driver:  CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701
        701 => cuda_cause(
            CudaAction::Other,
            "CUDA error 701: launch out of resources; reduce block size/register/shared-memory usage or fix custom kernel",
        ),

        _ => cuda_cause(
            CudaAction::Other,
            format!("CUDA error {code}: unclassified CUDA error code"),
        ),
    }
}
pub fn cuda_cause_from_msg(p0: impl Into<String> + Display) -> Cause {
    // Use Display, not Into<String>, so this also works nicely with things like
    // &TchError, anyhow::Error, etc.
    let msg = p0.to_string();
    let lower = msg.to_ascii_lowercase();

    // ------------------------------------------------------------------------
    // OOM first.
    // PyTorch/libtorch OOM messages often contain a lot of allocator stats:
    // "CUDA out of memory. Tried to allocate..."
    // ------------------------------------------------------------------------
    if contains_any(
        &lower,
        &[
            "cuda out of memory",
            "torch.cuda.outofmemoryerror",
            "outofmemoryerror",
            "cudaerror memory allocation",
            "cudaerrormemoryallocation",
            "cuda error: out of memory",
            "cuda_error_out_of_memory",
            "cuda_error_memory_allocation",
            "out of memory",
        ],
    ) {
        return cuda_cause(CudaAction::HandleOOM, msg);
    }

    // ------------------------------------------------------------------------
    // Fatal/sticky CUDA errors.
    // These are the ones after which I would restart the libtorch GPU worker.
    // ------------------------------------------------------------------------

    if contains_any(
        &lower,
        &[
            "device-side assert triggered",
            "cudaerrorassert",
            "cuda_error_assert",
            "cuda error: device-side assert",
            "error 710",
            "code=710",
            "code 710",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    if contains_any(
        &lower,
        &[
            "illegal memory access",
            "cudaerrorillegaladdress",
            "cuda_error_illegal_address",
            "cuda error: an illegal memory access was encountered",
            "error 700",
            "code=700",
            "code 700",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    if contains_any(
        &lower,
        &[
            "unspecified launch failure",
            "cudaerrorlaunchfailure",
            "cuda_error_launch_failed",
            "cuda error: unspecified launch failure",
            "error 719",
            "code=719",
            "code 719",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    if contains_any(
        &lower,
        &[
            "launch timed out",
            "launch timeout",
            "cudaerrorlaunchtimeout",
            "cuda_error_launch_timeout",
            "error 702",
            "code=702",
            "code 702",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    if contains_any(
        &lower,
        &[
            "hardware stack error",
            "cudaerrorhardwarestackerror",
            "cuda_error_hardware_stack_error",
            "error 714",
            "code=714",
            "code 714",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    if contains_any(
        &lower,
        &[
            "illegal instruction",
            "cudaerrorillegalinstruction",
            "cuda_error_illegal_instruction",
            "error 715",
            "code=715",
            "code 715",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    if contains_any(
        &lower,
        &[
            "misaligned address",
            "cudaerrormisalignedaddress",
            "cuda_error_misaligned_address",
            "error 716",
            "code=716",
            "code 716",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    if contains_any(
        &lower,
        &[
            "invalid address space",
            "cudaerrorinvalidaddressspace",
            "cuda_error_invalid_address_space",
            "error 717",
            "code=717",
            "code 717",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    if contains_any(
        &lower,
        &[
            "invalid program counter",
            "invalid pc",
            "cudaerrorinvalidpc",
            "cuda_error_invalid_pc",
            "error 718",
            "code=718",
            "code 718",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    if contains_any(
        &lower,
        &[
            "context is destroyed",
            "context was destroyed",
            "cudaerrorcontextisdestroyed",
            "cuda_error_context_is_destroyed",
            "error 709",
            "code=709",
            "code 709",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    if contains_any(
        &lower,
        &[
            "uncorrectable ecc",
            "ecc uncorrectable",
            "cudaerroreccuncorrectable",
            "cuda_error_ecc_uncorrectable",
            "error 214",
            "code=214",
            "code 214",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    if contains_any(
        &lower,
        &[
            "nvlink uncorrectable",
            "cudaerrornvlinkuncorrectable",
            "cuda_error_nvlink_uncorrectable",
            "error 220",
            "code=220",
            "code 220",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    if contains_any(
        &lower,
        &[
            "contained device exception",
            "cudaerrorcontained",
            "error 226",
            "code=226",
            "code 226",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    if contains_any(
        &lower,
        &[
            "tensor memory leak",
            "cudaerrortensormemoryleak",
            "error 721",
            "code=721",
            "code 721",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    if contains_any(
        &lower,
        &[
            "mps client terminated",
            "cudaerrormpsclientterminated",
            "cuda_error_mps_client_terminated",
            "error 810",
            "code=810",
            "code 810",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    if contains_any(
        &lower,
        &[
            "external device",
            "cudaerrorexternaldevice",
            "error 911",
            "code=911",
            "code 911",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    if contains_any(
        &lower,
        &[
            "cuda error: unknown error",
            "cudaerrorunknown",
            "cuda_error_unknown",
            "error 999",
            "code=999",
            "code 999",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    // ------------------------------------------------------------------------
    // CUDA library errors.
    //
    // These are not raw CUDA enum values, but in libtorch they often show up
    // after a previous fatal CUDA error poisoned the context. I would classify
    // the more severe ones as NeedsRestart unless you know they come from a
    // deterministic API/config bug.
    // ------------------------------------------------------------------------

    if contains_any(
        &lower,
        &[
            "cudnn_status_internal_error",
            "cudnn_status_execution_failed",
            "cudnn_status_mapping_error",
            "cudnn error: cudnn_status_internal_error",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    if contains_any(
        &lower,
        &[
            "cublas_status_execution_failed",
            "cublas_status_internal_error",
            "cublas_status_mapping_error",
            "cublas error",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    if contains_any(
        &lower,
        &[
            "cusolver_status_internal_error",
            "cusolver_status_execution_failed",
            "cusparse_status_execution_failed",
            "cusparse_status_internal_error",
            "cufft_internal_error",
            "cufft_exec_failed",
        ],
    ) {
        return cuda_cause(CudaAction::NeedsRestart, msg);
    }

    // ------------------------------------------------------------------------
    // Non-OOM / usually not restart-worthy.
    // Keep these after fatal matches because some messages contain extra text.
    // ------------------------------------------------------------------------

    if contains_any(
        &lower,
        &[
            "invalid device ordinal",
            "invalid device",
            "cudaerrorinvaliddevice",
            "cuda_error_invalid_device",
            "invalid argument",
            "invalid value",
            "cudaerrorinvalidvalue",
            "cuda_error_invalid_value",
            "invalid configuration argument",
            "cudaerrorinvalidconfiguration",
            "launch out of resources",
            "cudaerrorlaunchoutofresources",
            "cuda_error_launch_out_of_resources",
            "invalid resource handle",
            "cudaerrorinvalidresourcehandle",
            "cuda_error_invalid_handle",
            "index out of bounds",
            "index is out of bounds",
            "index out of range",
            "dimension out of range",
            "shape",
            "sizes",
            "expected scalar type",
            "expected dtype",
        ],
    ) {
        return cuda_cause(CudaAction::Other, msg);
    }

    cuda_cause(CudaAction::Other, msg)
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}


//TODO https://github.com/rust-lang/rust/issues/65280
impl Debug for ThoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug_struct = f.debug_struct("ThoosaError");
        let mut x = debug_struct
            .field("msg", &self.msg)
            .field("cause", &self.what);

        // #![feature(backtrace_frames)]
        for (i, frame) in self.backtrace.frames().iter().enumerate() {
            x = x.field(format!("frame_{}\n", i).as_str(), frame);
        }

        x.finish()
    }
}

impl From<Cause> for ThoError {
    fn from(value: Cause) -> Self {
        match value {
            e => Self::new_with_cause(format!("e={e:?}"), e),
        }
    }
}

impl<E> From<E> for ThoError
where
    E: Error + Send + Sync + 'static,
{
    fn from(value: E) -> Self {
        debug!("ThoosaError::from: value={}", value);

        ThoError::new(value.to_string())
    }
}



#[derive(Debug, Eq, Hash, PartialEq, Clone)]
pub struct MotifSlice {
    pub motif_id: u32,
    pub slice_idx: i64,
    pub channel: u16,
}

impl MotifSlice {
    pub fn new(motif_id: u32, slise_idx: i64, channel: u16) -> Self {
        Self {
            motif_id,
            slice_idx: slise_idx,
            channel,
        }
    }

    #[inline]
    pub fn motif_id(&self) -> u32 {
        self.motif_id
    }

    #[inline]
    pub fn slise_idx(&self) -> i64 {
        self.slice_idx
    }

    #[inline]
    pub fn channel(&self) -> u16 {
        self.channel
    }
}

pub fn random_saw(n: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    let step = 25usize;

    (0..n / step / 2)
        .map(|_| rng.random_range(-10. ..10.))
        .flat_map(|v| {
            (0..step)
                .map(move |i| v * i as f32)
                .chain((0..step).map(move |i| step as f32 * v - v * i as f32))
        })
        .collect::<Vec<f32>>()
}

pub fn saw_with_steps(n: usize, steps: usize, max_y: f32) -> Vec<f32> {
    let mut rng = rand::rng();

    let mut y = vec![0.; n];
    let mut x = 0;

    let def_step = n / steps;

    loop {
        let mut dx = min(
            n - x,
            (def_step as f32 * rng.random_range(0.5..1.5)) as usize,
        );
        if dx % 2 == 0 {
            dx += 1;
        }

        let max_y = max_y * rng.random_range(-1.5..1.5);

        if dx > 2 {
            let dy = max_y / (dx - 1) as f32 * 2.;

            for i in 1..dx / 2 {
                y[x + i] = if x + i > 0 { y[x + i - 1] } else { 0. } + dy;
                y[x + dx - i - 1] = y[x + i];
            }
        }

        y[x + dx / 2] = max_y;

        x = x + dx - 1;
        if x >= n {
            break;
        }
    }

    y
}

pub fn sig_three_imp(
    n: u32,
    max_y: f32,
    σ_range: Option<Range<f32>>,
    f_range: Option<Range<f32>>,
) -> Vec<f32> {
    let (σ, f) = σ_f_params(σ_range, f_range);

    join_with_offset::<f32>(
        0.,
        1.0,
        &[
            (
                (0.1 * n as f32) as usize,
                pulse_v_fit(0., σ, (0.3 * n as f32) as i64, f, 100., max_y),
            ),
            (
                (0.0 * n as f32) as usize,
                pulse_v_fit(0., σ, (0.2 * n as f32) as i64, f, 100., max_y),
            ),
            (
                (0.0 * n as f32) as usize,
                pulse_v_fit(0., σ, (0.3 * n as f32) as i64, f, 100., max_y),
            ),
        ],
        saw_with_steps(n as usize, 20, 0.2 * max_y),
    )
}

pub fn sig_two_imp(
    n: u32,
    max_y: f32,
    σ_range: Option<Range<f32>>,
    f_range: Option<Range<f32>>,
    _offset: f32,
) -> Vec<f32> {
    let (σ, f) = σ_f_params(σ_range, f_range);

    join_with_offset::<f32>(
        0.,
        1.0,
        &[
            (
                (0.1 * n as f32) as usize,
                pulse_v_fit(0., σ, (0.4 * n as f32) as i64, f, 100., max_y),
            ),
            (
                (0.0 * n as f32) as usize,
                pulse_v_fit(0., σ, (0.4 * n as f32) as i64, f, 100., max_y),
            ),
        ],
        saw_with_steps(n as usize, 20, 0.2 * max_y),
    )
}

pub fn sig_one_imp(
    n: u32,
    max_y: f32,
    σ_range: Option<Range<f32>>,
    f_range: Option<Range<f32>>,
    offset: f32,
) -> Vec<f32> {
    let (σ, f) = σ_f_params(σ_range, f_range);
    join_with_offset::<f32>(
        0.,
        1.0,
        &[(
            (offset * n as f32) as usize,
            pulse_v_fit(
                0.,
                σ,
                ((1.0 - 2. * offset) * n as f32) as i64,
                f,
                100.,
                max_y,
            ),
        )],
        saw_with_steps(n as usize, 20, 0.2 * max_y),
    )
}

fn σ_f_params(σ_range: Option<Range<f32>>, f_range: Option<Range<f32>>) -> (f32, f32) {
    let mut rng = rand::rng();
    let σ = σ_range.map(|r| rng.random_range(r)).unwrap_or(20.);
    let f = f_range.map(|r| rng.random_range(r)).unwrap_or(20.);
    (σ, f)
}

pub fn saw(n: u32) -> Vec<f32> {
    saw_with_steps(n as usize, 30, 100.)
}

pub fn pulses_in_const(n: u32, pulses: &[(usize, Vec<f32>)]) -> Vec<f32> {
    let pulses_len: usize = pulses
        .iter()
        .map(|(offset, pulse)| *offset + pulse.len())
        .sum();

    std::iter::repeat(0.)
        .take(n as usize)
        .zip(
            pulses
                .into_iter()
                .flat_map(|(offset, pulse)| {
                    std::iter::repeat(0f32)
                        .take(*offset)
                        .chain(pulse.into_iter().map(|v| *v))
                })
                .chain(std::iter::repeat(0f32).take(n as usize - pulses_len)),
        )
        .map(|(v1, v2)| 0.001 * v1 + v2)
        .collect::<Vec<f32>>()
}

pub fn pulse_in_rand(
    n: u32,
    pulses: &[(usize, Vec<f32>)],
    random_amplitude: f32,
    pulse_scale: f32,
    max_slope: f32,
) -> Vec<f32> {
    let random = test_utils::random_v(n, random_amplitude, max_slope);
    let _pulses_len: usize = pulses.iter().map(|(offset, _pulse)| *offset).sum();

    join_with_offset::<f32>(0., pulse_scale, pulses, random)
}

pub fn join_with_offset<T: Debug>(
    def: T,
    scale: T,
    pulses: &[(usize, Vec<T>)],
    main: Vec<T>,
) -> Vec<T>
where
    T: Copy + Clone + Mul<Output = T> + std::ops::Add<Output = T> + PartialEq + Default,
{
    let mut x: T = T::default();
    let main_len = main.len();
    main.into_iter()
        .zip(
            pulses
                .into_iter()
                .flat_map(|(offset, pulse)| {
                    std::iter::repeat(def)
                        .take(*offset)
                        .chain(pulse.into_iter().map(|v| *v))
                })
                .chain(
                    std::iter::repeat(def)
                        .take(main_len - pulses.last().map(|p| p.0 + p.1.len()).unwrap_or(0)),
                ),
        )
        .map(|(main_v, pulse_v)| {
            if pulse_v != def {
                scale * pulse_v
            } else {
                x = main_v;
                x
            }
        })
        .collect::<Vec<T>>()
}

pub fn gauss_v(μ: f32, σ: f32, n: i64) -> Vec<f32> {
    gauss_i(μ, σ, n).collect()
}

pub fn gauss_i(μ: f32, σ: f32, n: i64) -> impl Iterator<Item = f32> {
    (0..n).map(move |x| (-0.5 * ((x as f32 - μ) / σ).powi(2)).exp())
}

pub fn gauss_scaled(μ: f32, σ: f32, n: i64, max_x: f32, max_y: f32) -> Vec<f32> {
    (0..n)
        .map(move |i| -max_x / 2. + i as f32 * max_x / n as f32)
        .map(move |x| (-0.5 * ((x - μ) / σ).powi(2)).exp())
        .map(|y| y * max_y)
        .collect()
}

pub fn pulse_v_fit(μ: f32, σ: f32, n: i64, f: f32, max_x: f32, max_y: f32) -> Vec<f32> {
    gauss_scaled(μ, σ, n, max_x, max_y)
        .into_iter()
        .enumerate()
        .map(|(i, x)| x * (f * 2. * PI * i as f32 / n as f32).sin())
        .collect::<Vec<_>>()
}

pub fn sin_v(n: u32, p: f32) -> Vec<f32> {
    (0..n)
        .map(|x| 4. * (PI * (x as f32) / p).sin())
        .collect::<Vec<f32>>()
}

pub fn comp_sin(n: i64) -> impl Iterator<Item = f32> {
    (0..n).map(|x| x as f32).map(move |x| {
        30. * (PI / 100. * x).sin()
            + 4. * (PI / 50. * x.powf(1.2)).sin()
            + 8. * (PI / 25. * x.powf(1.2)).sin()
            + 12. * (PI / 12. * x.powf(1.2)).sin()
            + 6. * (PI / 6. * x.powf(1.2)).sin()
    })
}

pub fn comp_sin_v(n: i64) -> Vec<f32> {
    comp_sin(n).collect::<Vec<f32>>()
}

pub fn comp_sin_k(n: i64, k_v: Vec<f32>, scale: f32) -> impl Iterator<Item = f32> {
    (0..n)
        .map(|x| x as f32)
        .map(move |x| k_v.iter().map(|k| k * (PI / k * x.powf(1.2)).sin()).sum())
        .map(move |x: f32| x * scale)
}

pub fn comp_sin_k_v(n: i64, k_v: Vec<f32>, scale: f32) -> Vec<f32> {
    comp_sin_k(n, k_v, scale).collect::<Vec<f32>>()
}

pub fn pulse_v(μ: f32, σ: f32, n: i64, f: f32) -> Vec<f32> {
    gauss_i(μ, σ, n)
        .enumerate()
        .map(|(i, x)| (f * 2. * PI * i as f32 / n as f32).sin() * x)
        .collect::<Vec<_>>()
}


pub fn parse_key_val<T, U>(s: &str) -> Result<(T, U), Box<dyn Error + Send + Sync + 'static>>
where
    T: FromStr,
    T::Err: Error + Send + Sync + 'static,
    U: FromStr,
    U::Err: Error + Send + Sync + 'static,
{
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid KEY=value: no `=` found in `{s}`"))?;
    Ok((s[..pos].parse()?, s[pos + 1..].parse()?))
}

#[derive(Clone)]
pub enum RangeOrValue<V, T: FieldName = String> {
    Range {
        range: Range<V>,
        step: Option<(V, u32)>,//(value, number of digits to round to)
    }, //TODO replace step in params with number of steps? how choose number of decimals in step=to-from/steps?
    Value(V),
    Dynamic(Arc<dyn Fn(&[NamedVal<T>]) -> RangeOrValue<V> + Send + Sync>),
}

impl<V: Debug, T: FieldName> Debug for RangeOrValue<V, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Range { range, step } => f.debug_struct("Range").field("range", range).field("step", step).finish(),
            Self::Value(value) => f.debug_tuple("Value").field(value).finish(),
            Self::Dynamic(_) => f.write_str("Dynamic"),
        }
    }
}

impl<V: PartialEq, T: FieldName> PartialEq for RangeOrValue<V, T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Range { range: lhs_range, step: lhs_step },
                Self::Range { range: rhs_range, step: rhs_step },
            ) => lhs_range == rhs_range && lhs_step == rhs_step,
            (Self::Value(lhs), Self::Value(rhs)) => lhs == rhs,
            (Self::Dynamic(lhs), Self::Dynamic(rhs)) => Arc::ptr_eq(lhs, rhs),
            _ => false,
        }
    }
}

impl<V: Display + Serialize, T: FieldName> Serialize for RangeOrValue<V, T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            RangeOrValue::Value(v) => serializer.serialize_str(&v.to_string()),
            RangeOrValue::Range { range, step } => {
                let mut s = format!("{}..{}", range.start, range.end);
                if let Some((step_value, step_count)) = step {
                    s.push_str(&format!(":{},{}", step_value, step_count));
                }
                serializer.serialize_str(&s)
            }
            RangeOrValue::Dynamic(_) => Err(serde::ser::Error::custom("dynamic RangeOrValue values cannot be serialized")),
        }
    }
}

impl<'de, V: FromStr + Deserialize<'de>, T: FieldName> Deserialize<'de> for RangeOrValue<V, T>
where
    <V as FromStr>::Err: Display,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s: String = Deserialize::deserialize(deserializer)?;

        if let Some((range_part, step_part)) = s.split_once(':') {
            let mut range_iter = range_part.split("..");

            if let (Some(start), Some(end)) = (range_iter.next(), range_iter.next()) {
                let start: V = start.parse().map_err(serde::de::Error::custom)?;
                let end: V = end.parse().map_err(serde::de::Error::custom)?;

                let mut step_iter = step_part.split(',');
                if let (Some(step_value), Some(step_count)) = (step_iter.next(), step_iter.next()) {
                    let step_value: V = step_value.parse().map_err(serde::de::Error::custom)?;
                    let step_count: u32 = step_count.parse().map_err(serde::de::Error::custom)?;
                    return Ok(RangeOrValue::Range {
                        range: start..end,
                        step: Some((step_value, step_count)),
                    });
                }
            }
        } else if let Some((start, end)) = s.split_once("..") {
            let start: V = start.parse().map_err(serde::de::Error::custom)?;
            let end: V = end.parse().map_err(serde::de::Error::custom)?;

            return Ok(RangeOrValue::Range {
                range: start..end,
                step: None,
            });
        } else {
            return Ok(RangeOrValue::Value(
                s.parse().map_err(serde::de::Error::custom)?,
            ));
        }

        Err(serde::de::Error::custom("Invalid format for RangeOrValue"))
    }
}

pub fn add_range_or_value<'a, V: 'a + 'static, T: FieldName + 'static>(
    name: T,
    v: &'a RangeOrValue<V, T>,
    const_params: &mut Vec<NamedVal<T>>,
    grid_params: &mut Vec<NamedRange<T>>,
    opt_params: &mut Vec<Hyperparameter<T>>,
) where
    for<'b> Val: From<&'b V>,
    for<'b> Hyperparameter<T>: From<(T, &'b Range<V>)>,
{
    match v {
        RangeOrValue::Range {
            range,
            step: step_op,
        } => match step_op {
            None => {
                opt_params.push((name, range).into());
            }
            Some((step, mul)) => {
                grid_params.push(NamedRange::new(
                    name,
                    (&range.start).into()..(&range.end).into(),
                    (step.into(), *mul),
                ));
            }
        },
        RangeOrValue::Value(v) => {
            const_params.push(NamedVal::from((name, v.into())));
        }
        RangeOrValue::Dynamic(resolve) => {
            grid_params.push(NamedRange::from_grid_value_sets(vec![GridValueSet::with_pending_fields(
                vec![],
                vec![],
                vec![PendingOptField::from_range_or_value(name, Arc::clone(resolve))],
            )]));
        }
    }
}

pub fn add_prefix<T: FieldName>(prefix: Option<&T>, field: T) -> T {
    // prefix.map(|p| format!("{}.{}", p, field))
    //     .unwrap_or(field)
    match prefix {
        None => field.into(),
        Some(_prefix) => {
            // prefix.extend(field)
            field.into()
        }
    }
}

impl<T: FieldName> IntoIterator for RangeOrValue<u32, T> {
    type Item = u32;
    type IntoIter = Box<dyn Iterator<Item = u32>>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            RangeOrValue::Range { range, step } => {
                if let Some((step, _mul)) = step {
                    Box::new((range.start..range.end).step_by(step as usize))
                } else {
                    Box::new([range.start, range.end].into_iter())
                }
            }
            RangeOrValue::Value(v) => Box::new(std::iter::once(v)),
            RangeOrValue::Dynamic(_) => panic!("can't iterate unresolved dynamic RangeOrValue"),
        }
    }
}

impl<T: FieldName> IntoIterator for RangeOrValue<i32, T> {
    type Item = i32;
    type IntoIter = Box<dyn Iterator<Item = i32>>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            RangeOrValue::Range { range, step } => {
                if let Some((step, _mul)) = step {
                    Box::new((range.start..range.end).step_by(step as usize))
                } else {
                    Box::new([range.start, range.end].into_iter())
                }
            }
            RangeOrValue::Value(v) => Box::new(std::iter::once(v)),
            RangeOrValue::Dynamic(_) => panic!("can't iterate unresolved dynamic RangeOrValue"),
        }
    }
}

impl<T: FieldName> IntoIterator for RangeOrValue<f64, T> {
    type Item = f64;
    type IntoIter = Box<dyn Iterator<Item = f64>>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            RangeOrValue::Range { range, step } => {
                if let Some((step, mul)) = step {
                    Box::new(
                        ((range.start * <u32 as Into<f64>>::into(mul)) as i32
                            ..(range.end * <u32 as Into<f64>>::into(mul)) as i32)
                            .step_by((step * <u32 as Into<f64>>::into(mul)) as usize)
                            .map(move |v| {
                                <i32 as Into<f64>>::into(v) / <u32 as Into<f64>>::into(mul)
                            }),
                    )
                } else {
                    Box::new([range.start, range.end].into_iter())
                }
            }
            RangeOrValue::Value(v) => Box::new(std::iter::once(v)),
            RangeOrValue::Dynamic(_) => panic!("can't iterate unresolved dynamic RangeOrValue"),
        }
    }
}

impl<V: Display + PrimInt, T: FieldName> Display for RangeOrValue<V, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            RangeOrValue::Range { range, step: _ } => {
                write!(f, "{}-{}", range.start, range.end)
            }
            RangeOrValue::Value(v) => {
                write!(f, "{}", v)
            }
            RangeOrValue::Dynamic(_) => {
                write!(f, "<dynamic>")
            }
        }
    }
}

impl<V, T: FieldName> RangeOrValue<V, T>
where
    V: Clone + 'static,
{
    pub fn dynamic<F>(resolve: F) -> Self
    where
        F: Fn(&[NamedVal<T>]) -> RangeOrValue<V> + Send + Sync + 'static,
    {
        Self::Dynamic(Arc::new(resolve))
    }

    pub fn expect_value(&self) -> Result<&V, ThoError> {
        match self {
            RangeOrValue::Range { .. } => Err(ThoError::new_with_cause(
                "expected RangeOrValue::Value, got Range",
                Cause::ExpectedValue,
            )),
            RangeOrValue::Value(v) => Ok(v),
            RangeOrValue::Dynamic(_resolve) => Err(ThoError::new_with_cause(
                "expected RangeOrValue::Value, got Dynamic",
                Cause::ExpectedValue,
            )),
        }
    }

    pub fn expect_range(&self) -> Result<&Range<V>, ThoError> {
        match self {
            RangeOrValue::Range { range, step: _ } => Ok(range),
            RangeOrValue::Value(_v) => Err(ThoError::new_with_cause(
                "expected RangeOrValue::Range, got Value",
                Cause::ExpectedRange,
            )),
            RangeOrValue::Dynamic(_resolve) => Err(ThoError::new_with_cause(
                "expected RangeOrValue::Range, got Dynamic",
                Cause::ExpectedRange,
            )),
        }
    }
}

impl<T: FieldName> From<f64> for RangeOrValue<f64, T> {
    fn from(value: f64) -> Self {
        RangeOrValue::Value(value)
    }
}

impl<T: FieldName> From<f32> for RangeOrValue<f32, T> {
    fn from(value: f32) -> Self {
        RangeOrValue::Value(value)
    }
}

impl<T: FieldName> From<u32> for RangeOrValue<u32, T> {
    fn from(value: u32) -> Self {
        RangeOrValue::Value(value)
    }
}

impl<T: FieldName> From<i32> for RangeOrValue<i32, T> {
    fn from(value: i32) -> Self {
        RangeOrValue::Value(value)
    }
}

impl<T: FieldName> From<i64> for RangeOrValue<i64, T> {
    fn from(value: i64) -> Self {
        RangeOrValue::Value(value)
    }
}

impl<V, T: FieldName> From<Range<V>> for RangeOrValue<V, T>
where
    V: PartialEq + Into<RangeOrValue<V, T>>,
{
    fn from(r: Range<V>) -> Self {
        if r.start == r.end {
            r.start.into()
        } else {
            RangeOrValue::Range {
                range: r,
                step: None,
            }
        }
    }
}

pub fn parse_range_or_value<'a, V>(
    s: &str,
) -> Result<RangeOrValue<V>, Box<dyn Error + Send + Sync + 'static>>
where
    V: FromStr,
    V::Err: Error + Send + Sync + 'static,
{
    if s.contains("..") {
        let ((from, from_d_len), (to, to_d_len), step) =
            parse_range_str(s).expect("parse_range_str"); //.map_err(|e|)?;

        Ok(RangeOrValue::Range {
            range: from.parse()?..to.parse()?,
            step: step.map(|(step, step_len)| {
                (
                    step.parse().unwrap(),
                    10u32.pow(
                        [from_d_len, to_d_len, step_len]
                            .into_iter()
                            .max()
                            .unwrap_or(0) as u32,
                    ),
                )
            }),
        })
    } else {
        Ok(RangeOrValue::Value(s.parse()?))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResourceType {
    List(Vec<String>),
    Set(String),
    Group(String),
    NotInGroup(String),
}

pub fn parse_resources(s: &str) -> Result<ResourceType, Box<dyn Error + Send + Sync + 'static>> {
    let value_pos = s.find(":").expect("resource specifier should contain ':'") + 1;

    if s.starts_with("list") {
        Ok(ResourceType::List(
            s[value_pos..].split(",").map(|v| v.to_string()).collect(),
        ))
    } else if s.starts_with("set") {
        Ok(ResourceType::Set(s[value_pos..].to_string()))
    } else if s.starts_with("group") {
        Ok(ResourceType::Group(s[value_pos..].to_string()))
    } else if s.starts_with("not_group") {
        Ok(ResourceType::NotInGroup(s[value_pos..].to_string()))
    } else {
        panic!("unknown resource type {s}, expecting list, set or group")
    }
}

/**
calculates indexes for sliding  windows of size {motif_width} and step {step} in input of length {input_n} and splits them in N batches of  {batch_size} and reminder M
*/
pub fn batch_windows(
    input_n: usize,
    motif_width: usize,
    step: usize,
    batch_size: usize,
) -> (Vec<Vec<usize>>, Vec<usize>) {
    // (batches, reminder)
    debug!(
        "slice_windows: input_n={input_n}, motif_width={motif_width}, step={step} batch_size={batch_size}"
    );

    (0..=input_n - motif_width)
        .step_by(step)
        .fold((vec![], vec![]), |mut a, v| {
            a.1.push(v);
            if a.1.len() == batch_size {
                a.0.push(a.1);
                a.1 = vec![];
            }
            a
        })
}

pub const fn clc_num_batches(
    input_n: usize,
    motif_width: usize,
    step: usize,
    batch_size: usize,
) -> usize {
    let positions = (input_n - motif_width) / step;
    (positions + batch_size - 1) / batch_size
}

pub const fn slices_len(input_n: u32, motif_width: u32, step: u32) -> u32 {
    (input_n - motif_width) / step + 1
}

pub fn clc_step(input_n: u32, motif_width: u32, slices_num: u32) -> (u32, u32) {
    let v = input_n - motif_width;
    let div = slices_num - 1;

    let r = v % div;

    let pad = if r == 0 { 0 } else { div - r };

    let step = (v + pad) / div;

    (step, pad)
}

pub fn parse_corr_labels(descr: &str) -> Result<Vec<(i64, String)>, ThoError> {
    let re: Regex = Regex::new(r#"\((?<id>\d+),\s"(?<kind>\w+)"\)"#)
        .map_err(|e| {
            ThoError::new_with_cause("compile corr labels regex", Cause::RegexParse(e))
        })?; //TODO const, error
    Ok(re
        .captures_iter(descr)
        .map(|caps| {
            (
                caps["id"].parse::<i64>().expect(&caps["id"]),
                caps["kind"].into(),
            )
        })
        .collect::<Vec<(i64, String)>>())
}

pub fn cdt() -> String {
    chrono::offset::Local::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

pub fn cdt_for_dir() -> String {

    let mut rng = rand::rng();

    let ts = cdt();
    let ts = ts.trim_matches('"'); // remove leading/trailing quotes if present
    let ts = ts
        .replace(':', "_")
        .replace('+', "_")
        .replace('"', ""); // remove any remaining quotes

    format!("{ts}_{}", rng.next_u32())
}

pub static LOG_INIT: AtomicBool = AtomicBool::new(false);

pub fn recurse_files<F>(path: impl AsRef<Path>, filter: &F) -> std::io::Result<Vec<PathBuf>>
where
    F: Fn(&PathBuf) -> bool,
{
    let mut buf = vec![];
    if fs::exists(&path)? {
        let entries = read_dir(path)?;

        for entry in entries {
            let entry = entry?;
            let meta = entry.metadata()?;

            if meta.is_dir() {
                let mut subdir = recurse_files(entry.path(), filter)?;
                buf.append(&mut subdir);
            }

            if meta.is_file() {
                let path = entry.path();
                if filter(&path) {
                    buf.push(path);
                }
            }
        }
    }

    Ok(buf)
}

fn parse_range_str(
    range: &str,
) -> Result<((String, u8), (String, u8), Option<(String, u8)>), ThoError> {
    lazy_static!{
        static ref RANGE_RE:Regex = Regex::new(r"^(?P<from>-?\d+(\.(?P<from_d>\d+))?)\.{2}(?P<to>-?\d+(\.(?P<to_d>\d+))?)(:(?P<step>-?\d+(\.(?P<step_d>\d+))?))?$").unwrap();
    }

    RANGE_RE
        .captures(range)
        .map(|caps| {
            let from_i = caps
                .name("from")
                .map(|v| v.as_str().to_string())
                .expect("from");
            let from_d = caps
                .name("from_d")
                .and_then(|v| v.as_str().len().try_into().ok())
                .unwrap_or(0);

            let to_i = caps.name("to").map(|v| v.as_str().to_string()).expect("to");
            let to_d = caps
                .name("to_d")
                .and_then(|v| v.as_str().len().try_into().ok())
                .unwrap_or(0);

            let step_i = caps.name("step").map(|v| v.as_str().to_string());
            let step_d = caps
                .name("step_d")
                .and_then(|v| v.as_str().len().try_into().ok())
                .unwrap_or(0);

            Ok((
                (from_i, from_d),
                (to_i, to_d),
                step_i.map(|step| (step, step_d)),
            ))
        })
        .unwrap_or_else(|| {
            Err(ThoError::new_with_cause(
                format!("can't parse range {range}"),
                Cause::InvalidRange {
                    range: range.to_string(),
                },
            ))
        })
}

pub fn make_combos<V: Clone>(iters: impl Iterator<Item = impl Iterator<Item = V>>) -> Vec<Vec<V>> {
    let mut combos: Vec<Vec<V>> = vec![vec![]];

    for iter in iters {
        let mut new_combos = vec![];
        for item in iter {
            for combo in &combos {
                let mut new_combo = combo.clone();
                new_combo.push(item.clone());
                new_combos.push(new_combo);
            }
        }
        combos = new_combos;
    }

    combos
}

pub fn round_to_digits(value: f64, digits: u32) -> f64 {
    if digits == 0 {
        return value;
    }
    let factor = 10_f64.powi(digits as i32);
    (value * factor).round() / factor
}

pub fn arg_prefix(prev_prefix: Option<&str>, prefix: &str) -> String {
    prev_prefix
        .map(|v| v.to_string() + "-")
        .unwrap_or("".to_string())
        + prefix
}

pub fn camel_to_snake(input: &str) -> String {
    let mut snake = String::new();

    let mut chars = input.chars();

    snake.push(
        chars
            .next()
            .expect("input should have at least one character")
            .to_lowercase()
            .next()
            .unwrap(),
    );

    for ch in chars {
        if ch.is_uppercase() {
            snake.push('_');
        }
        snake.push(ch.to_lowercase().next().unwrap());
    }

    snake
}

pub fn snake_to_camel(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<String>()
}

pub fn conc_str(prefix: &str, field: &str) -> String {
    let mut s = String::with_capacity(prefix.len() + field.len());
    s.push_str(prefix);
    s.push_str(field);
    s
}

pub fn avg_hist(values: &[f64], n_bins: usize) -> Vec<(f64, u32)> {
    // Edge cases: empty data or zero bins
    if values.is_empty() || n_bins == 0 {
        return Vec::new();
    }

    // Find the minimum and maximum in the data
    let min_value = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_value = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // If all values are the same, place them all in the first bin
    if min_value == max_value {
        return vec![(min_value, values.len() as u32)];
    }

    // Determine bin size based on overall range and number of bins
    let bin_size = (max_value - min_value) / n_bins as f64;

    // Create a vector of counts initialized to 0
    let mut count_bins = vec![0u32; n_bins];
    let mut value_bins = vec![vec![]; n_bins];

    // Increment the appropriate bin for each value
    for &value in values {
        // Identify the bin index by its relative distance from min_value
        let mut bin_index = ((value - min_value) / bin_size) as usize;

        // Handle edge case where `value` might be exactly max_value
        // (floating-point math can push it into the next bin)
        if bin_index == n_bins {
            bin_index = n_bins - 1;
        }
        count_bins[bin_index] += 1;
        value_bins[bin_index].push(value);
    }

    count_bins
        .into_iter()
        .zip(value_bins)
        .map(|(count, v)| {
            (
                v.into_iter().sum::<f64>() / (count as f64 + f64::EPSILON),
                count,
            )
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct F64BitsKey(pub u64);

impl From<f64> for F64BitsKey {
    #[inline]
    fn from(x: f64) -> Self {
        Self(x.to_bits())
    }
}

#[cfg(test)]
mod shared_tests {
    #![allow(unused_imports, redundant_semicolons, dead_code)]
    /*
       a - b
       a - c
       e - f
       b - d
       f - g
       x - y


    group 1: a, b, c, d
    group 2: e, f, g
    group 3: x, y
    */
    use crate::hyperparameter::FieldName;
    use crate::pulse_v;
    use crate::val::NamedVal;
    use crate::{
        add_prefix, arg_prefix, avg_hist, batch_windows, camel_to_snake, clc_num_batches, group_slices,
        make_combos, parse_range_str, snake_to_camel, test_utils, Cause,
        RangeOrValue, ResourceType, ThoError, StreamBuffering,
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use clap::{Args, Parser};
    use log::debug;
    use std::collections::HashSet;
    use std::fmt::{Display, Formatter};

    #[ctor::ctor]
    fn init() {
        test_utils::init_test_logging();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_collect_futures_with_progress_waits_for_cuda_activity_gate() {
        let started = Arc::new(AtomicUsize::new(0));
        let pause_ticket = super::cuda_activity_gate().request_pause();

        let collect = {
            let started = Arc::clone(&started);
            tokio::spawn(async move {
                super::collect_futures_with_progress(
                    (0..4).map(move |idx| {
                        let started = Arc::clone(&started);
                        async move {
                            started.fetch_add(1, Ordering::SeqCst);
                            Ok::<_, ThoError>(idx)
                        }
                    }),
                    4,
                    4,
                    StreamBuffering::Unordered,
                    "test_collect_futures_with_progress_waits_for_cuda_activity_gate",
                )
                    .await
            })
        };

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(started.load(Ordering::SeqCst), 0);

        super::cuda_activity_gate().finish_pause(pause_ticket);

        let results = tokio::time::timeout(Duration::from_secs(2), collect)
            .await
            .expect("collector should resume")
            .expect("collector task should not panic");
        assert_eq!(results.len(), 4);
        assert_eq!(started.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn test_join_with_offset() {
        let pulse = pulse_v(50., 10., 100, 20.);
        let v = super::join_with_offset::<f32>(
            0.,
            10.0,
            &[(100, pulse.clone())],
            super::random_saw(550),
        );
        assert_eq!(550, v.len());

        let v = super::join_with_offset::<f32>(
            0.,
            10.0,
            &[(100, pulse.clone())],
            super::random_saw(2200),
        );
        assert_eq!(2200, v.len());
    }

    #[test]
    fn test_group_slices() {
        let pairs: Vec<(String, String)> = vec![
            ("a".into(), "b".into()),
            ("a".into(), "c".into()),
            ("e".into(), "f".into()),
            ("b".into(), "d".into()),
            ("f".into(), "g".into()),
            ("x".into(), "y".into()),
        ];

        let mut groups = group_slices(pairs.into_iter());

        let mut group_0 = groups.get_mut(&0).unwrap().iter().collect::<Vec<_>>();
        group_0.sort();
        assert_eq!(vec!["a", "b", "c", "d"], group_0);

        let mut group_1 = groups.get_mut(&1).unwrap().iter().collect::<Vec<_>>();
        group_1.sort();
        assert_eq!(vec!["e", "f", "g"], group_1);

        let mut group_2 = groups.get_mut(&2).unwrap().iter().collect::<Vec<_>>();
        group_2.sort();
        assert_eq!(vec!["x", "y"], group_2);
    }

    #[test]
    fn test_group_motifs_separated() {
        let pairs: Vec<(String, String)> = vec![
            ("l".into(), "m".into()),
            ("a".into(), "x".into()),
            ("a".into(), "y".into()),
            ("b".into(), "w".into()),
            ("b".into(), "v".into()),
            ("a".into(), "b".into()),
        ];

        let groups = group_slices(pairs.into_iter());

        assert_eq!(groups.len(), 2);

        let mut values: Vec<HashSet<String>> =
            groups.values().into_iter().map(|v| v.clone()).collect();
        values.sort_by(|a, b| b.len().cmp(&a.len()));

        let mut group_0 = values.get_mut(0).unwrap().iter().collect::<Vec<_>>();
        group_0.sort();
        assert_eq!(vec!["a", "b", "v", "w", "x", "y"], group_0);

        let mut group_1 = values.get_mut(1).unwrap().iter().collect::<Vec<_>>();
        group_1.sort();
        assert_eq!(vec!["l", "m"], group_1);
    }

    #[test]
    fn test_batch_windows_1000() {
        let input_n = 1000;
        let motif_width = 100;
        let step = 100;

        let batches = batch_windows(input_n, motif_width, step, 10);

        assert_eq!(
            vec![0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
            batches.0[0]
        );

        assert_eq!(batches.0.len(), 1);
        assert!(batches.1.is_empty());
    }

    #[test]
    fn test_batch_windows_3500() {
        let input_n = 3500;
        let motif_width = 100;
        let step = 100;

        let batches = batch_windows(input_n, motif_width, step, 10);

        assert_eq!(
            vec![0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
            batches.0[0]
        );
        assert_eq!(
            vec![1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            batches.0[1]
        );
        assert_eq!(
            vec![2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900],
            batches.0[2]
        );
        assert_eq!(vec![3000, 3100, 3200, 3300, 3400], batches.1)
    }

    #[test]
    fn test_clc_num_batches() {
        assert_eq!(clc_num_batches(1000, 100, 10, 1000), 1);

        assert_eq!(clc_num_batches(1000, 100, 10, 90), 1);

        assert_eq!(clc_num_batches(1000, 100, 10, 50), 2);

        assert_eq!(clc_num_batches(1000, 100, 10, 30), 3);
    }

    #[test]
    fn test_parse_corr_labels() -> Result<(), ThoError> {
        assert_eq!(
            vec![
                (1, "seizure_vote".into()),
                (0, "lpd_vote".into()),
                (0, "gpd_vote".into()),
                (0, "lrda_vote".into()),
                (0, "grda_vote".into()),
                (0, "other_vote".into())
            ],
            super::parse_corr_labels(
                "[(1, \"seizure_vote\"), (0, \"lpd_vote\"), (0, \"gpd_vote\"), (0, \"lrda_vote\"), (0, \"grda_vote\"), (0, \"other_vote\")]"
            )?
        );

        Ok(())
    }

    #[test]
    fn test_parse_range_or_value_range() -> Result<(), ThoError> {
        use crate::parse_range_or_value;

        #[derive(Debug, Parser, Clone, PartialEq)]
        struct TestParams {
            #[arg(long, value_parser=parse_range_or_value::<i32>)]
            window: RangeOrValue<i32>,

            #[arg(long, value_parser=parse_range_or_value::<f32>)]
            step: RangeOrValue<f32>,
        }

        let params =
            TestParams::try_parse_from(vec!["", "--step=1.1..2.2", "--window=1..2"].into_iter())
                .map_err(|e| {
                    ThoError::new_with_cause(
                        "parse range-or-value args",
                        Cause::CliParse(Arc::new(e)),
                    )
                })?; //TODO error

        assert_eq!(
            TestParams {
                window: RangeOrValue::Range {
                    range: 1..2,
                    step: None
                },
                step: RangeOrValue::Range {
                    range: 1.1..2.2,
                    step: None
                }
            },
            params
        );

        Ok(())
    }

    #[test]
    fn test_resources_param() -> Result<(), ThoError> {
        use crate::parse_resources;

        #[derive(Debug, Parser, Clone, PartialEq)]
        struct TestParams {
            #[arg(long,value_parser=parse_resources)]
            resources: ResourceType,
        }

        let params =
            TestParams::try_parse_from(vec!["", "--resources=list:aaa,bbb,ccc"].into_iter())
                .map_err(|e| {
                    ThoError::new_with_cause(
                        "parse resources args (list)",
                        Cause::CliParse(Arc::new(e)),
                    )
                })?; //TODO error

        assert_eq!(
            ResourceType::List(vec!["aaa".into(), "bbb".into(), "ccc".into()]),
            params.resources
        );
        let params = TestParams::try_parse_from(vec!["", "--resources=set:aaa"].into_iter())
            .map_err(|e| {
                ThoError::new_with_cause(
                    "parse resources args (set)",
                    Cause::CliParse(Arc::new(e)),
                )
            })?; //TODO error
        assert_eq!(ResourceType::Set("aaa".into()), params.resources);

        let params = TestParams::try_parse_from(vec!["", "--resources=group:bbb"].into_iter())
            .map_err(|e| {
                ThoError::new_with_cause(
                    "parse resources args (group)",
                    Cause::CliParse(Arc::new(e)),
                )
            })?; //TODO error
        assert_eq!(ResourceType::Group("bbb".into()), params.resources);

        Ok(())
    }

    //TODO when deserialized by serde range with step requires number of decimals - 1.01..2.2:0.1*,1*
    #[test]
    fn test_parse_range_or_value_range_with_step() -> Result<(), ThoError> {
        use crate::parse_range_or_value;

        #[derive(Debug, Parser, Clone, PartialEq)]
        struct TestParams {
            #[arg(long, value_parser=parse_range_or_value::<i32>)]
            window: RangeOrValue<i32>,

            #[arg(long, value_parser=parse_range_or_value::<f64>)]
            step: RangeOrValue<f64>,
        }

        let params = TestParams::try_parse_from(
            vec!["", "--step=-1.01..2.2:0.1", "--window=-20..20:2"].into_iter(),
        )
        .map_err(|e| {
            ThoError::new_with_cause(
                "parse range-or-value args with step",
                Cause::CliParse(Arc::new(e)),
            )
        })?; //TODO error

        assert_eq!(
            TestParams {
                window: RangeOrValue::Range {
                    range: -20..20,
                    step: Some((2, 1))
                },
                step: RangeOrValue::Range {
                    range: -1.01..2.2,
                    step: Some((0.1, 100))
                }
            },
            params
        );

        assert_eq!(
            (-20..20).step_by(2).collect::<Vec<i32>>(),
            params.window.into_iter().collect::<Vec<i32>>()
        );

        assert_eq!(
            (-101..220)
                .step_by(10)
                .map(|i| i as f64 / 100.)
                .collect::<Vec<f64>>(),
            params.step.into_iter().collect::<Vec<f64>>()
        );

        Ok(())
    }

    #[test]
    fn test_parse_range_or_value_value() -> Result<(), ThoError> {
        use crate::parse_range_or_value;

        #[derive(Debug, Parser, Clone, PartialEq)]
        struct TestParams {
            #[arg(long, value_parser=parse_range_or_value::<i32>)]
            window: RangeOrValue<i32>,

            #[arg(long, value_parser=parse_range_or_value::<f32>)]
            step: RangeOrValue<f32>,
        }

        let params = TestParams::try_parse_from(vec!["", "--step=1.1", "--window=10"].into_iter())
            .map_err(|e| {
                ThoError::new_with_cause(
                    "parse range-or-value args (value)",
                    Cause::CliParse(Arc::new(e)),
                )
            })?; //TODO error

        assert_eq!(
            TestParams {
                window: RangeOrValue::Value(10),
                step: RangeOrValue::Value(1.1)
            },
            params
        );

        Ok(())
    }

    #[test]
    fn test_parser_regex_str() -> Result<(), ThoError> {
        let ((from, from_d_len), (to, to_d_len), step) = parse_range_str("1.22..333.4444")?;

        assert_eq!("1.22".to_string(), from);
        assert_eq!(2, from_d_len);

        assert_eq!("333.4444".to_string(), to);
        assert_eq!(4, to_d_len);

        assert!(step.is_none());

        let ((from, from_d_len), (to, to_d_len), step) = parse_range_str("-1.22..-333.4444")?;

        assert_eq!("-1.22".to_string(), from);
        assert_eq!(2, from_d_len);

        assert_eq!("-333.4444".to_string(), to);
        assert_eq!(4, to_d_len);

        assert!(step.is_none());

        let ((from, from_d_len), (to, to_d_len), Some((step, step_d_len))) =
            parse_range_str("-1.22..-333.4444:7.8")?
        else {
            panic!("no step");
        };

        assert_eq!("-1.22".to_string(), from);
        assert_eq!(2, from_d_len);

        assert_eq!("-333.4444".to_string(), to);
        assert_eq!(4, to_d_len);

        assert_eq!("7.8".to_string(), step);
        assert_eq!(1, step_d_len);

        let ((from, from_d_len), (to, to_d_len), Some((step, step_d_len))) =
            parse_range_str("-1.22..-333.4444:-7.8")?
        else {
            panic!("no step");
        };
        // println!("from_i={from} from_d={from_d_len:?} to={to} to_d_len={to_d_len} step={step:?}");

        assert_eq!("-1.22".to_string(), from);
        assert_eq!(2, from_d_len);

        assert_eq!("-333.4444".to_string(), to);
        assert_eq!(4, to_d_len);

        assert_eq!("-7.8".to_string(), step);
        assert_eq!(1, step_d_len);

        Ok(())
    }

    #[test]
    fn test_make_combos() {
        assert_eq!(
            [
                ["a1", "b1", "c1"],
                ["a2", "b1", "c1"],
                ["a3", "b1", "c1"],
                ["a1", "b1", "c2"],
                ["a2", "b1", "c2"],
                ["a3", "b1", "c2"]
            ]
            .into_iter()
            .map(|v| v.to_vec())
            .collect::<Vec<_>>(),
            make_combos(
                [
                    vec!["a1", "a2", "a3"].into_iter(),
                    vec!["b1"].into_iter(),
                    vec!["c1", "c2"].into_iter()
                ]
                .into_iter()
            )
        );

        assert_eq!(
            [["a1", "b1",], ["a2", "b1",]]
                .into_iter()
                .map(|v| v.to_vec())
                .collect::<Vec<_>>(),
            make_combos([vec!["a1", "a2"].into_iter(), vec!["b1"].into_iter(),].into_iter())
        );

        assert_eq!(
            [["a1"]].into_iter().map(|v| v.to_vec()).collect::<Vec<_>>(),
            make_combos([vec!["a1"].into_iter(),].into_iter())
        );
    }

    #[test]
    fn test_arg_prefix() {
        assert_eq!("aaa", arg_prefix(None, "aaa"));

        assert_eq!("bbb-aaa", arg_prefix(Some("bbb"), "aaa"));
    }

    #[test]
    fn test_camel_to_snake() {
        assert_eq!(camel_to_snake("Hello"), "hello");

        assert_eq!(camel_to_snake("HelloTest"), "hello_test");

        assert_eq!(camel_to_snake("HelloTestOne"), "hello_test_one");
    }

    #[test]
    fn test_snake_to_camel() {
        assert_eq!(snake_to_camel("hello"), "Hello");

        assert_eq!(snake_to_camel("hello_test"), "HelloTest");

        assert_eq!(snake_to_camel("hello_test_one"), "HelloTestOne");
    }

    //TODO replace ...Fields with Derive
    #[test]
    fn test_append_sub_field_enums() {
        #[derive(Debug, Clone, PartialEq)]
        enum SubStructFields {
            FieldA,
            FieldB,
        }
        impl Display for SubStructFields {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                write!(f, "")
            }
        }
        impl FieldName for SubStructFields {}

        #[derive(Debug, Clone, PartialEq)]
        enum ParentStructAFields {
            Field0,
            SubStructA(SubStructFields),
            SubStructB(SubStructFields),
        }

        impl Display for ParentStructAFields {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                write!(f, "")
            }
        }
        impl FieldName for ParentStructAFields {}

        #[derive(Debug, Clone, PartialEq)]
        enum ParentStructBFields {
            Field0,
            SubStructB(SubStructFields),
        }

        impl Display for ParentStructBFields {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                write!(f, "")
            }
        }
        impl FieldName for ParentStructBFields {}

        fn fill_parent_a(params: &mut Vec<NamedVal<ParentStructAFields>>) {
            params.push(NamedVal::new::<ParentStructAFields, i32>(
                ParentStructAFields::Field0.into(),
                0.into(),
            ));
            fill_sub_struct(params, |v| ParentStructAFields::SubStructA(v));
        }

        fn fill_parent_b(params: &mut Vec<NamedVal<ParentStructBFields>>) {
            params.push(NamedVal::new::<ParentStructBFields, i32>(
                ParentStructBFields::Field0.into(),
                1.into(),
            ));
            fill_sub_struct(params, |v| ParentStructBFields::SubStructB(v));
        }

        fn fill_sub_struct<T: FieldName, F>(params: &mut Vec<NamedVal<T>>, append: F)
        where
            F: Fn(SubStructFields) -> T,
        {
            params.push(NamedVal::new::<T, i32>(
                append(SubStructFields::FieldA).into(),
                0.into(),
            ));
            params.push(NamedVal::new::<T, i32>(
                append(SubStructFields::FieldB).into(),
                1.into(),
            ));
        }

        let mut params_a = vec![];
        fill_parent_a(&mut params_a);

        let mut params_b = vec![];
        fill_parent_b(&mut params_b);

        debug!("params_a: {params_a:?}");
        debug!("params_b: {params_b:?}");

        assert_eq!(
            params_a[1].path(),
            &ParentStructAFields::SubStructA(SubStructFields::FieldA),
        );
        assert_eq!(
            params_a[2].path(),
            &ParentStructAFields::SubStructA(SubStructFields::FieldB),
        );

        assert_eq!(
            params_b[1].path(),
            &ParentStructBFields::SubStructB(SubStructFields::FieldA),
        );

        assert_eq!(
            params_b[2].path(),
            &ParentStructBFields::SubStructB(SubStructFields::FieldB),
        );
    }

    #[test]
    fn test_append_sub_field_string() {
        fn fill_parent_a(params: &mut Vec<NamedVal<String>>) {
            params.push(NamedVal::new::<String, i32>("Field0".to_string(), 0.into()));
            fill_sub_struct(params, |v| "SubStructA.".to_string() + &v);
        }

        fn fill_parent_b(params: &mut Vec<NamedVal<String>>) {
            params.push(NamedVal::new::<String, i32>("Field0".to_string(), 1.into()));
            fill_sub_struct(params, |v| "SubStructB.".to_string() + &v);
        }

        fn fill_sub_struct<F>(params: &mut Vec<NamedVal<String>>, append: F)
        where
            F: Fn(String) -> String,
        {
            params.push(NamedVal::new::<String, i32>(
                append("FieldA".to_string()).into(),
                0.into(),
            ));
            params.push(NamedVal::new::<String, i32>(
                append("FieldB".to_string()).into(),
                1.into(),
            ));
        }

        let mut params_a = vec![];
        fill_parent_a(&mut params_a);

        let mut params_b = vec![];
        fill_parent_b(&mut params_b);

        debug!("params_a: {params_a:?}");
        debug!("params_b: {params_b:?}");

        assert_eq!(params_a[1].path(), &"SubStructA.FieldA");
        assert_eq!(params_a[2].path(), &"SubStructA.FieldB");

        assert_eq!(params_b[1].path(), &"SubStructB.FieldA");
        assert_eq!(params_b[2].path(), &"SubStructB.FieldB");
    }

    #[test]
    pub fn test_hist() {
        let data = vec![1.2, 2.4, 2.5, 2.8, 3.1, 4.7, 2.1, 2.4, 1.9, 3.0];
        let n_bins = 5;

        let hist = avg_hist(&data, n_bins);

        // Print out the histogram
        println!("Histogram counts in each bin: {:?}", hist);
    }
}

pub fn replace_extension<'a>(path: &'a PathBuf, new_extension: &'a str) -> (&'a str, PathBuf) {
    let parent = path.parent().expect("parent");
    let file_stem = path
        .file_stem()
        .expect("file_stem")
        .to_str()
        .expect("to_str");
    let json = format!("{}.{}", file_stem, new_extension);
    let json = parent.join(json);
    (file_stem, json)
}

#[async_trait]//TODO it doesn't need to be dyn compatible, remove async_trait and change dst_device form usize to D:Into<ThoDevice> ?
#[delegatable_trait]
#[automock]
pub trait CopyToDevice {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> where Self:Sized;
}



#[macro_export]
macro_rules! panic_to_err_async {
    ($expr:expr) => {{

        // let x = std::panic::catch_unwind(core::panic::AssertUnwindSafe(async || $expr));
        let x = futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe(async {$expr})).await;

        match x {
            //no panic
            Ok(inner_result) => {
                match inner_result.await {
                    Ok(val) => {
                        log::debug!("panic_to_err_async: val: {val:?}");
                        Ok(val)
                    },
                    Err(err) => {
                        log::debug!("panic_to_err_async: err: {err:?}");
                        Err(thoosa_shared::ThoError::from(err))
                    }
                }
            },
            //panic
            Err(e) => {
                if let Some(msg) = e.downcast_ref::<String>() {
                    log::debug!("panic_to_err_async: String: {msg}");
                    let msg1 = msg.as_str();Err(ThoError::new(msg1))

                } else if let Some(msg) = e.downcast_ref::<&str>() {
                    log::debug!("panic_to_err_async: str: {msg}");
                    let msg1 = *msg;Err(ThoError::new(msg1))
                } else {
                    log::debug!("panic_to_err_async: unknown: {e:?}");
                    let msg = format!("{e:?}");Err(ThoError::new(msg))
                }
            }
        }
    }};
}
#[macro_export]
macro_rules! panic_to_err {
    ($expr:expr) => {{
        match std::panic::catch_unwind(core::panic::AssertUnwindSafe(|| $expr)) {
            //no panic
            Ok(inner_result) => match inner_result {
                Ok(val) => Ok(val),
                Err(err) => Err(ThoError::from(err)),
            },
            //panic
            Err(e) => {
                if let Some(msg) = e.downcast_ref::<String>() {
                    log::debug!("thoosa_err String: {msg}");
                    Err(ThoError::new(msg))

                // } else if let Some(thoosa_error) = e.downcast_ref::<ThoosaError>() {
                //     Err(*thoosa_error)

                } else if let Some(msg) = e.downcast_ref::<&str>() {
                    log::debug!("thoosa_err str: {msg}");
                    let msg1 = *msg;Err(ThoError::new(msg1))
                } else {
                    log::debug!("thoosa_err unknown: {e:?}");
                    let msg = format!("{e:?}");Err(ThoError::new(msg))
                }
            }
        }
    }};
}


pub fn shuffle_and_chunk_with_rng<R: rand::Rng + ?Sized>(
    n: usize,
    batch_size:u32,
    rng: &mut R,
) -> Vec<Box<[usize]>> {

    let mut vals: Vec<usize> = (0..n).collect();
    vals.shuffle(rng);

    let usable_len = n / batch_size as usize * batch_size as usize;
    vals.truncate(usable_len);

    vals.chunks_exact(batch_size as usize)
        .map(|chunk| chunk.to_vec().into_boxed_slice())
        .collect()
}



/// Returns up to `n` *references* to randomly chosen distinct elements (no clones).
pub fn sample_n_refs<'a, T, R: Rng + ?Sized>(v: &'a [T], n: usize, rng: &mut R) -> Vec<&'a T> {
    v.sample(rng, n).collect() // returns min(n, v.len())
}

/// Returns up to `n` randomly chosen distinct elements by *cloning* them.
pub fn sample_n_cloned<T: Clone, R: Rng + ?Sized>(v: &[T], n: usize, rng: &mut R) -> Vec<T> {
    v.sample(rng, n).cloned().collect()
}

/// Consumes the vector, shuffles it, and truncates to up to `n` elements (no clones).
pub fn sample_n_owned<T, R: Rng + ?Sized>(mut v: Vec<T>, n: usize, rng: &mut R) -> Vec<T> {
    v.shuffle(rng);
    v.truncate(n.min(v.len()));
    v
}

pub fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before Unix epoch")
        .as_millis()
}

#[cfg(test)]
#[test]
pub fn test_shuffle_and_chunk_with_rng(){
    let mut rng = StdRng::seed_from_u64(42);

    let chunks = shuffle_and_chunk_with_rng(10,3, &mut rng);


    assert_eq!(
        vec![[8, 6, 3].to_vec().into_boxed_slice(),
             [2, 1, 5].to_vec().into_boxed_slice(),
             [9, 7, 4].to_vec().into_boxed_slice()],
        chunks
    )
}

#[inline]
pub fn os_thread_id() -> u64 {
    #[cfg(target_os = "linux")]
    {
        // Works even if glibc doesn’t export gettid: call the syscall directly.
        unsafe { libc::syscall(libc::SYS_gettid) as u64 }
    }
}

pub fn backtrace_head_lines(n: usize) -> String {
    let bt = Backtrace::capture();
    format!("{bt:?}")
        .lines()
        .take(n)
        .collect::<Vec<_>>()
        .join("\n")
}
