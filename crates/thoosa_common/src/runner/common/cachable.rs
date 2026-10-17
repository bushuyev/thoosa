#![allow(unused_parens)]
#![allow(async_fn_in_trait)]

use std::backtrace::Backtrace;
use crate as thoosa_common;
use std::borrow::Borrow;

use mockall_double::double;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;


use async_trait::async_trait;
use dashmap::{DashMap, Entry};
use futures_util::FutureExt;
use futures_util::future::{BoxFuture, Shared};
use log::{debug, trace, warn};

use crate::data_with_extra::DataWithExtra;
use parking_lot::Mutex;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::{Debug, Formatter};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use thoosa_shared::n_trace_line;
use thoosa_shared::ThoError;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::ConsumeResult;
use thoosa_shared::traits::{AsyncStage, Stage, WithId};
use thoosa_shared::traits::{CachableIdTr, Data, Named, OutId, RunCtx, Select, ThoSelector};
use thoosa_shared::{CopyToDevice, async_perf_log, async_read_lock, async_write_lock, perf_log, sync_mutex_lock, sync_read_lock, sync_write_lock};
use thoosa_shims::stream::ThoStreamGuard;
use tokio::sync::{Notify, RwLockReadGuard, futures::OwnedNotified};

trait DeviceCopier<O>: Debug + Send + Sync {
    fn copy<'a>(&self, src: &'a O, dst_device: usize) -> BoxFuture<'a, Result<O, ThoError>>;
}

#[derive(Debug)]
struct CopyToDeviceCopier;

impl<O> DeviceCopier<O> for CopyToDeviceCopier
where
    O: CopyToDevice + Send + Sync,
{
    fn copy<'a>(&self, src: &'a O, dst_device: usize) -> BoxFuture<'a, Result<O, ThoError>> {
        async move { src.copy_to_device(dst_device).await }.boxed()
    }
}

pub trait CachedPurgeOutput: Clone + Send + Sync + 'static {
    fn dump_to_cpu<'a>(&'a self, device_idx: usize) -> BoxFuture<'a, Result<(), ThoError>>;

    fn cleanup_device<'a>(&'a self, device_idx: usize) -> BoxFuture<'a, Result<(), ThoError>>;

    fn drop_device<'a>(&'a self, device_idx: usize) -> BoxFuture<'a, Result<(), ThoError>>;

    fn is_on_device(&self, device_idx: usize) -> bool;

    fn has_any_data(&self) -> bool;

    fn is_on_cpu_before(&self, purge_until: u64) -> bool;


    fn mem_size(&self) -> u64;
}


impl<O> CachedPurgeOutput for Arc<DevicesWrap<O>>
where
    O: Data + Send + Sync + 'static,
{
    fn dump_to_cpu<'a>(&'a self, device_idx: usize) -> BoxFuture<'a, Result<(), ThoError>> {
        DevicesWrap::dump_to_cpu(self.as_ref(), device_idx).boxed()
    }

    fn cleanup_device<'a>(&'a self, device_idx: usize) -> BoxFuture<'a, Result<(), ThoError>> {
        // The cache and cleanup job intentionally both own handles. Actual
        // borrowers are represented by residency read guards, not Arc counts.
        DevicesWrap::cleanup_device(self.as_ref(), device_idx).boxed()
    }

    fn drop_device<'a>(&'a self, device_idx: usize) -> BoxFuture<'a, Result<(), ThoError>> {
        DevicesWrap::drop_device(self.as_ref(), device_idx).boxed()
    }

    fn is_on_device(&self, device_idx: usize) -> bool {
        DevicesWrap::is_on_device(self.as_ref(), device_idx)
    }

    fn has_any_data(&self) -> bool {
        self.resident_devices.iter().any(|resident| resident.load(Ordering::Acquire))
    }

    fn is_on_cpu_before(&self, purge_until: u64) -> bool {
        DevicesWrap::is_on_cpu_before(self.as_ref(), purge_until)
    }

    fn mem_size(&self) -> u64 {
        DevicesWrap::mem_size(self.as_ref())
    }
}


impl<O> CachedPurgeOutput for Arc<O>
where
    O: CachedPurgeOutput
{
    fn dump_to_cpu<'a>(&'a self, _device_idx: usize) -> BoxFuture<'a, Result<(), ThoError>> {
        async { Ok(()) }.boxed()
    }

    fn cleanup_device<'a>(&'a self, _device_idx: usize) -> BoxFuture<'a, Result<(), ThoError>> {
        async { Ok(()) }.boxed()
    }

    fn drop_device<'a>(&'a self, _device_idx: usize) -> BoxFuture<'a, Result<(), ThoError>> {
        async { Ok(()) }.boxed()
    }

    fn is_on_device(&self, device_idx: usize) -> bool {
        device_idx == 0
    }

    fn has_any_data(&self) -> bool { true }

    fn is_on_cpu_before(&self, _purge_until: u64) -> bool {
        true
    }

    fn mem_size(&self) -> u64 {
        self.as_ref().mem_size()
    }
}

/// Explicit cache-area users, shared by clones of one cached computation.
/// Counts control retention, while device-slot guards protect actual borrowers.
/// Registration must not wait for cleanup: a caller can already hold a residency
/// read guard when it requests another cached handle.
#[derive(Default)]
struct CacheResidency {
    users: AtomicUsize,
}

struct CacheUse {
    residency: Arc<CacheResidency>,
    released: AtomicBool,
}

impl CacheUse {
    fn acquire(residency: Arc<CacheResidency>) -> Self {
        residency.users.fetch_add(1, Ordering::AcqRel);
        Self { residency, released: AtomicBool::new(false) }
    }

    fn release(&self) -> bool {
        !self.released.swap(true, Ordering::AcqRel) && self.residency.users.fetch_sub(1, Ordering::AcqRel) == 1
    }
}

impl Drop for CacheUse {
    fn drop(&mut self) { self.release(); }
}

/// Shared computations retain the device generation that started them. Recovery
/// can discard that device's pending work without cancelling healthy cache entries.
pub struct CachedFuture<O: Clone> {
    inner: Shared<BoxFuture<'static, Result<O, ThoError>>>,
    origin: Option<(Arc<thoosa_shared::CudaActivityGate>, thoosa_shared::CudaDeviceGeneration)>,
    invalidated: Arc<AtomicBool>,
    users: Arc<Mutex<HashMap<usize, Arc<CacheResidency>>>>,
}

impl<O: Clone> CachedFuture<O> {
    pub fn for_device(inner: Shared<BoxFuture<'static, Result<O, ThoError>>>, gate: Option<Arc<thoosa_shared::CudaActivityGate>>) -> Self
    where O: Send + Sync + 'static,
    {
        let activity = match &gate {
            Some(gate) => thoosa_shared::CudaActivityScope::for_operation(gate).map(Some),
            None => Ok(thoosa_shared::CudaActivityScope::current()),
        };
        let origin = gate.map(|gate| {
            let generation = activity.as_ref().ok().and_then(|scope| scope.as_ref()).and_then(|scope| scope.generation(&gate)).unwrap_or_else(|| gate.device_generation());
            (gate, generation)
        });
        if origin.is_none() && matches!(&activity, Ok(None)) {
            return Self::with_origin(inner, origin, Arc::new(AtomicBool::new(false)));
        }
        let computation_origin = origin.clone();
        let inner = async move {
            let activity = activity?;
            let dependencies = activity.clone();
            let run = async move {
                let interrupted = async {
                    let origin_ended = async {
                        match &computation_origin {
                            Some((gate, generation)) => gate.wait_for_generation_end(*generation).await,
                            None => std::future::pending().await,
                        }
                    };
                    let dependency_ended = async {
                        match &dependencies {
                            Some(scope) => scope.wait_for_interruption().await,
                            None => std::future::pending().await,
                        }
                    };
                    tokio::select! { error = origin_ended => error, error = dependency_ended => error }
                };
                tokio::select! {
                    biased;
                    error = interrupted => Err(error),
                    result = inner => {
                        if let Some(scope) = &dependencies { scope.check_dependencies()?; }
                        if let Some((gate, generation)) = &computation_origin { gate.check_generation(*generation)?; }
                        result
                    }
                }
            };
            match activity { Some(scope) => scope.run(run).await, None => run.await }
        }.boxed().shared();
        // A cache owns its pending future even when every caller cancels. Keep
        // polling it so recovery can drop suspended locks/leases before draining.
        tokio::spawn(inner.clone());
        Self::with_origin(inner, origin, Arc::new(AtomicBool::new(false)))
    }

    fn with_origin(inner: Shared<BoxFuture<'static, Result<O, ThoError>>>, origin: Option<(Arc<thoosa_shared::CudaActivityGate>, thoosa_shared::CudaDeviceGeneration)>, invalidated: Arc<AtomicBool>) -> Self {
        Self { inner, origin, invalidated, users: Arc::new(Mutex::new(HashMap::new())) }
    }

    pub(crate) fn has_cache_users(&self) -> bool {
        self.users.lock().values().any(|residency| residency.users.load(Ordering::Acquire) != 0)
    }

    fn cleanup_registration(&self, device_idx: usize) -> Box<dyn Fn() -> BoxFuture<'static, Result<(), ThoError>> + Send + Sync>
    where O: CachedPurgeOutput,
    {
        let residency = self.users.lock().entry(device_idx).or_default().clone();
        let lease = Arc::new(CacheUse::acquire(residency));
        // A strong Shared here would create a cycle through the computation's
        // RunCtx. Weak identity also prevents stale cleanup purging a replacement.
        let weak = self.inner.downgrade();
        Box::new(move || {
            let last = lease.release();
            let residency = Arc::clone(&lease.residency);
            let future = weak.as_ref().and_then(|weak| weak.upgrade());
            async move {
                if !last || device_idx == 0 || GlobalCtx::is_shutting_down() { return Ok(()); }
                if residency.users.load(Ordering::Acquire) != 0 { return Ok(()); }
                if let Some(future) = future {
                    // The last RunCtx may drop just before Shared publishes its
                    // output. Await completion instead of silently skipping it.
                    if let Ok(data) = future.await { data.cleanup_device(device_idx).await?; }
                }
                Ok(())
            }.boxed()
        })
    }

    pub fn peek(&self) -> Option<&Result<O, ThoError>> { self.inner.peek() }

    pub fn ptr_eq(&self, other: &Self) -> bool { self.inner.ptr_eq(&other.inner) }

    pub(crate) fn invalidate_pending_on(&self, device: usize) -> bool {
        if self.origin.as_ref().is_some_and(|(gate, _)| gate.device_idx() == device) && !matches!(self.peek(), Some(Ok(_))) {
            self.invalidated.store(true, Ordering::Release);
            true
        } else { false }
    }
}

impl<O: Clone> Clone for CachedFuture<O> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone(), origin: self.origin.clone(), invalidated: Arc::clone(&self.invalidated), users: Arc::clone(&self.users) }
    }
}

impl<O: Clone> std::fmt::Debug for CachedFuture<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedFuture").field("origin", &self.origin).finish_non_exhaustive()
    }
}

impl<O: Clone> From<Shared<BoxFuture<'static, Result<O, ThoError>>>> for CachedFuture<O> {
    fn from(inner: Shared<BoxFuture<'static, Result<O, ThoError>>>) -> Self { Self::with_origin(inner, None, Arc::new(AtomicBool::new(false))) }
}

impl<O: Clone> Future for CachedFuture<O> {
    type Output = Result<O, ThoError>;

    fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Self::Output> {
        if self.invalidated.load(Ordering::Acquire) {
            return std::task::Poll::Ready(Err(ThoError::ContextReset));
        }
        let result = std::pin::Pin::new(&mut self.inner).poll(cx);
        if self.invalidated.load(Ordering::Acquire) {
            std::task::Poll::Ready(Err(ThoError::ContextReset))
        } else { result }
    }
}

// Callers own a handle, so cache lifetimes follow the runtime and active users
pub type StageCache<S> = DashMap<<<S as Stage>::Out as WithId>::ID, CachedFuture<<S as Stage>::Out>>;

#[async_trait]
pub trait Cached: AsyncStage + Named
where
    for <'i> Self::In<'i>: WithId + Send + Sync,
    <Self as Stage>::Out: CachedPurgeOutput +  WithId + Send + Sync + 'static,//
{
    // Identifies the concrete cache returned by cache_for(). For an adapter
    // output, use output_cache_key() to follow its actual cache owner.
    fn cache_key() -> crate::runner::global_ctx::CacheKey {
        crate::runner::global_ctx::CacheKey::of::<Self>()
    }

    // Adapters can return another stage's cached output. Exclusions must follow
    // that owner, including adapters which select a stage at runtime.
    fn output_cache_key(&self) -> crate::runner::global_ctx::CacheKey {
        Self::cache_key()
    }

    fn run_input_id(input: &Self::In<'static>) -> &'static <Self::In<'static> as WithId>::ID {
        unimplemented!()
    }

    async fn cache() -> Result<Arc<StageCache<Self>>, ThoError>
    where
        Self: Sized,
        Self::Out:  Sync + Send;

    async fn run_cached(
        self: Arc<Self>,
        input: Self::In<'static>,
        mut ctx: Self::Ctx,
    ) -> Result<Self::Out, ThoError>
    where
        Self: Sized + OutId,
        for<'a> <Self as Stage>::In<'a>: WithId + Send + Sync,
        <Self as Stage>::Out: CachedPurgeOutput + Clone + Send + Sync + 'static,//CachedPurgeOutput +
        <<Self as Stage>::Out as WithId>::ID: 'static,
    {
        let key = self.out_id(input.id());
        let device_idx = ctx.tho_device_idx();

        debug!("Cached::run_cached: key={key:?} device_idx={device_idx} ctx.id={:?}", ctx.id());
        // #[cfg(any(test, feature = "mock_global_ctx"))]
        // let _mock_ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

        let cache = track_await!(Self::cache()).await?;
        GlobalCtx::global().do_age();

        let (shared, cleanup) = match cache.entry(key.clone()) {
            Entry::Occupied(slot) => {
                let future = slot.get().clone();
                let cleanup = future.cleanup_registration(device_idx);
                (future, cleanup)
            },
            Entry::Vacant(slot) => {
                let gate = GlobalCtx::global().cuda_activity_gate(device_idx);
                let future = CachedFuture::for_device(Arc::clone(&self).run(input, ctx.branch(false)).boxed().shared(), gate);
                let cleanup = future.cleanup_registration(device_idx);
                slot.insert(future.clone());
                (future, cleanup)
            }
        };
        // Register under the map's entry guard so CPU eviction cannot remove
        // an entry between cache lookup and acquiring its explicit user lease.
        // Keep this context through completion; computation uses the same area.
        ctx.add_purge_gpu_closure(cleanup);
        let result = track_await!(shared.clone()).await;
        if result.is_err() {
            // A failed computation has no reusable output. Evict it without
            // requiring a device-generation change to permit a later attempt.
            // Compare identity: another caller may already have installed a new
            // generation's computation under this same logical cache key.
            cache.remove_if(&key, |_, current| current.ptr_eq(&shared));
        }
        result
    }
}


//https://github.com/rust-lang/rust/issues/100013
#[derive(Debug)]
pub struct DevicesWrap<O>
where
    O: Data + Send + Sync,
{
    // The old broad WithId requirement was a relic of an earlier caching design. DevicesWrap
    // operations do not need it; this erased field only keeps the stable reference required
    // by the conditional WithId impl when the wrapper is a Cached AsyncStage output.
    id: Box<dyn CachableIdTr>,

    cached_data: Box<[tokio::sync::RwLock<Option<O>>]>,//cached data never changes
    inflights: Box<[Mutex<Option<Arc<Notify>>>]>,
    size: O::Metrics,
    mem_size: u64,
    copy_to_device: Option<Arc<dyn DeviceCopier<O>>>,
    resident_devices: Box<[AtomicBool]>,
    gpu_resident_count: AtomicUsize,
    //contains cache age ( GlobalCtx::cache_age - increated when something is being cached from Cachable::run_cached). The smaller the value the longer it was on the cpu
    on_cpu_since: parking_lot::RwLock<Option<u64>>,
    pd: PhantomData<O>,
}



#[derive(Debug)]
pub enum ThoDataRef<'a, D> {
    Borrowed(&'a D),
    ReadLocked(RwLockReadGuard<'a, D>),
    Owned(D),
}

impl<D> ThoDataRef<'_, D> {
    pub fn as_ref(&self) -> &D {
        match self {
            Self::Borrowed(value) => value,
            Self::ReadLocked(value) => value,
            Self::Owned(value) => value,
        }
    }
}

impl <D>  Serialize for ThoDataRef<'_, D>
where
    D:Serialize
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer
    {
        match self {
            ThoDataRef::Borrowed(d) => {
                d.serialize(serializer)
            }
            ThoDataRef::ReadLocked(d) => {
                d.serialize(serializer)
            }
            ThoDataRef::Owned(v) => {
                v.serialize(serializer)
            }
        }
    }
}

impl <'de, D> Deserialize<'de> for ThoDataRef<'_, D>
where
    D:Deserialize<'de>
{
    fn deserialize<Ds>(deserializer: Ds) -> Result<Self, Ds::Error>
    where
        Ds: Deserializer<'de>
    {
        Ok(ThoDataRef::Owned(D::deserialize(deserializer)?))
    }
}


impl <O> BorrowRef for DevicesWrap<O>
where
    O: Data + CopyToDevice + Send + Sync,
{
    type Out<'a> = ThoDataRef<'a, O> where Self: 'a, O:'a;

    fn borrow_ref<'o>(&'o self, to_tho_device_idx: usize) -> impl Future<Output=Result<Self::Out<'o>, ThoError>> + Send
    where
        Self: 'o
    {
        async move {
            let read_guard = self.load_from_any_device_(to_tho_device_idx).await?;
            Ok(ThoDataRef::ReadLocked(RwLockReadGuard::map(
                read_guard,
                |val: &O| val,
            )))
        }
    }
}

impl <S, O> SelectRef<S> for DevicesWrap<O>
where
    S:ThoSelector + Clone + Debug + Send,
    O: Select<S, Output = O> + Data + CopyToDevice + Send + Sync,
{
    fn select_ref<'o>(&'o self, to_tho_device_idx: usize, selector: Option<S>) -> impl std::future::Future<Output = Result<<Self as BorrowRef>::Out<'o>, ThoError>> + std::marker::Send
    where
        O:'o,
        S:'o,
        Self:'o,
    {
        async move {
            match selector {
                None => {
                    debug!("DevicesWrap::borrow_to: no selector self.id={:?}", self.id);

                    BorrowRef::borrow_ref(self, to_tho_device_idx).await
                }
                Some(selector) => {
                    debug!("DevicesWrap::borrow_to: selector={selector:?} self.id={:?}", self.id);

                    if self.is_resident_on_device(to_tho_device_idx) {
                        debug!("DevicesWrap::borrow_to: self.id={:?} is_resident_on_device", self.id);

                        let guard = track_await!(self.cached_data[to_tho_device_idx].read()).await;
                        match guard.as_ref() {
                            None => {
                                self.mark_not_resident_on_device(to_tho_device_idx);
                                warn!("BorrowRef<S> for DevicesWrap::borrow_to: resident_on_device  flag for {to_tho_device_idx} was stale")
                            }
                            Some(data) => {
                                let sub_t: O = data.subview(selector.clone())?;
                                debug!("DevicesWrap::with_selected_data_ref: sub_t.size={:?}", sub_t.size());

                                return Ok(ThoDataRef::Owned(sub_t));
                            }
                        }
                    }

                    for (source_device_idx, op) in self.cached_data.iter().enumerate() {
                        if !self.is_resident_on_device(source_device_idx) {
                            continue;
                        }

                        let source_read_guard: RwLockReadGuard<'_, Option<O>> = track_await!(op.read()).await;

                        if let Some(src) = source_read_guard.as_ref() {
                            let sub_t: O = src.subview(selector.clone())?;

                            debug!("DevicesWrap::with_selected_data_ref: sub_t.size={:?} copying to to_device_idx={to_tho_device_idx} self.copy_to_device={:?}", sub_t.size(), self.copy_to_device);

                            let data = match &self.copy_to_device {
                                Some(copy_to_device) => track_await!(copy_to_device.copy(&sub_t, to_tho_device_idx)).await?,
                                None => sub_t,
                            };

                            debug!("DevicesWrap::with_selected_data_ref: data={data:?}");

                            return Ok(ThoDataRef::Owned(data))

                        }
                        drop(source_read_guard);

                    }

                    Err(ThoError::NotFound(Some(format!("select_ref: id={:?} selector={selector:?}", self.id))))
                }
            }
        }
    }
}



impl<'a, D> Borrow<D> for ThoDataRef<'a, D> {
    fn borrow(&self) -> &D {
        self.as_ref()
    }
}




enum InflightRole {
    Leader(Arc<Notify>),
    Follower(OwnedNotified),
}

fn join_or_start_inflight(inflight: &Mutex<Option<Arc<Notify>>>) -> InflightRole {
    let mut guard = perf_log!(sync_mutex_lock!(inflight));
    if let Some(notify) = guard.as_ref() {
        // Create the future while holding the mutex. notify_waiters remembers notifications
        // sent after future creation, even if the future has not been polled yet.
        InflightRole::Follower(notify.clone().notified_owned())
    } else {
        let notify = Arc::new(Notify::new());
        *guard = Some(notify.clone());
        InflightRole::Leader(notify)
    }
}

fn initial_residency(num_devices: usize, resident_device_idx: usize) -> (Box<[AtomicBool]>, AtomicUsize) {
    assert!(resident_device_idx < num_devices, "resident device index {resident_device_idx} is out of range for {num_devices} devices");
    let resident_devices = (0..num_devices).map(|idx| AtomicBool::new(idx == resident_device_idx)).collect::<Vec<_>>().into_boxed_slice();
    let gpu_resident_count = AtomicUsize::new(usize::from(resident_device_idx > 0));
    (resident_devices, gpu_resident_count)
}

impl<O> DevicesWrap<O>
where
    O: Data + Send + Sync,
{
    fn is_resident_on_device(&self, device_idx: usize) -> bool {
        self.resident_devices[device_idx].load(Ordering::Acquire)
    }

    fn is_all_on_cpu_resident(&self) -> bool {
        self.gpu_resident_count.load(Ordering::Acquire) == 0
    }

    fn mark_resident_on_device(&self, device_idx: usize) {
        let was_resident = self.resident_devices[device_idx].swap(true, Ordering::AcqRel);
        if !was_resident && device_idx > 0 {
            let previous_gpu_count = self.gpu_resident_count.fetch_add(1, Ordering::AcqRel);
            if previous_gpu_count == 0 {
                let mut on_cpu_since = perf_log!(sync_write_lock!(&self.on_cpu_since), "DevicesWrap::mark_resident_on_device on_cpu_since.write");
                if on_cpu_since.is_some() {
                    *on_cpu_since = None;
                }
            }
        }
    }

    fn mark_not_resident_on_device(&self, device_idx: usize) {
        let was_resident = self.resident_devices[device_idx].swap(false, Ordering::AcqRel);
        if was_resident && device_idx > 0 {
            let previous_gpu_count = self.gpu_resident_count.fetch_sub(1, Ordering::AcqRel);
            assert!(previous_gpu_count > 0, "gpu resident count underflow for {:?}", self.id);
        }
    }

    fn mark_on_cpu_since_if_all_on_cpu(&self, age: u64) {
        if self.is_all_on_cpu_resident() {
            let mut on_cpu_since = perf_log!(sync_write_lock!(&self.on_cpu_since), "DevicesWrap::mark_on_cpu_since_if_all_on_cpu on_cpu_since.write");
            if on_cpu_since.is_none() {
                *on_cpu_since = Some(age);
            }
        }
    }
}

impl <O> WithId for DevicesWrap<O>
where
    O: WithId + Data + Send + Sync,

{
    type ID = O::ID;

    fn id(&self) -> &Self::ID {
        // wrap_inner stores O::ID, so this downcast is guaranteed by construction.
        self.id.as_ref().downcast_ref::<O::ID>().expect("DevicesWrap stored an unexpected ID type")
    }
}


impl  <O> Output for DevicesWrap<O>
where
    O: Data + Send + Sync + Output {

}

impl  <O> Input for DevicesWrap<O>
where
    O: Data + Send + Sync + Input {

}

impl<O> DevicesWrap<O>
where
    O: Data + Send + Sync,
{
    pub fn wrap(o: O) -> Self
    where
        O: CopyToDevice + WithId,
    {
        Self::wrap_inner(o, Some(Arc::new(CopyToDeviceCopier)))
    }

    #[cfg(test)]
    pub fn wrap_cpu(o: O) -> Self
    where
        O: WithId,
    {
        assert_eq!(o.tho_device_idx(), 0);
        Self::wrap_inner(o, None)
    }

    fn wrap_inner(o: O, copy_to_device: Option<Arc<dyn DeviceCopier<O>>>) -> Self
    where
        O: WithId,
    {
        let id = o.id().clone();
        let size = o.size();
        let device_ordinal = o.tho_device_idx();
        let mem_size = o.mem_size();
        let num_devices: usize = GlobalCtx::global().device_count();
        let (resident_devices, gpu_resident_count) = initial_residency(num_devices, device_ordinal);

        debug!("DevicesWrap::wrap: id={id:?} device_ordinal={device_ordinal}");

        let mut o_op = Some(o);

        let device_futures: Box<[tokio::sync::RwLock<Option<O>>]> = (0..num_devices)
            .map(|device| {
                if device == device_ordinal {
                    tokio::sync::RwLock::new(o_op.take())
                } else {
                    tokio::sync::RwLock::new(None)
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let inflights = (0..num_devices)
            .map(|_device| Mutex::new(None))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            id: Box::new(id),
            cached_data: device_futures,
            inflights,
            size,
            resident_devices,
            gpu_resident_count,
            on_cpu_since: Default::default(),
            mem_size,
            copy_to_device,
            pd: PhantomData,
        }
    }


    pub fn size(&self) -> &O::Metrics {
        &self.size
    }

    async fn cleanup_device(&self, device_idx: usize) -> Result<(), ThoError> {
        if device_idx == 0 || GlobalCtx::is_shutting_down() { return Ok(()); }
        // Wait for real borrowers, then hold exclusive residency through backup
        // and removal. Cache handles and incidental clones do not prevent cleanup.
        if let Some(slot) = self.cached_data.get(device_idx) {
            let mut source = slot.write().await;
            self.offload_locked(device_idx, &mut source).await?;
        }
        Ok(())
    }

    async fn offload_locked(&self, device_idx: usize, source: &mut Option<O>) -> Result<(), ThoError> {
        if let Some(data) = source.as_ref() {
            // Release the CPU read guard before ensure_copied takes its write lock.
            let backed_up = self.cached_data[0].read().await.is_some();
            if !backed_up { self.ensure_copied(0, data, false).await?; }
            source.take();
            self.mark_not_resident_on_device(device_idx);
        }
        self.mark_on_cpu_since_if_all_on_cpu(GlobalCtx::global().cache_age());
        Ok(())
    }

    pub async fn dump_to_cpu(&self, device_idx: usize) -> Result<(), ThoError>
    {
        if GlobalCtx::is_shutting_down(){
            return Ok(());
        }

        if self.is_all_on_cpu_resident() {
            debug!("DevicesWrap::dump_to_cpu: 0.  id={:?} device_idx={device_idx} is_all_on_cpu_resident", self.id);

            self.mark_on_cpu_since_if_all_on_cpu(GlobalCtx::global().cache_age());
            return Ok(());
        }

        assert_ne!(device_idx, 0);

        debug!("DevicesWrap::dump_to_cpu: 0.  id={:?} device_idx={device_idx}", self.id);

        if let Some(Ok(mut lock)) = self.cached_data.get(device_idx).map(|v| v.try_write()) {
            self.offload_locked(device_idx, &mut lock).await?;
        } else {
            // OOM eviction may be called while its own input is borrowed, so it
            // must skip busy slots. Cache-area cleanup instead waits in cleanup_device.
            warn!("DevicesWrap::dump_to_cpu: can't drop self.id={:?}, locked, data={:?}", self.id, self.cached_data);
            debug!("DevicesWrap::dump_to_cpu: id={:?} trace: {}", self.id, n_trace_line(&Backtrace::capture(), 50));
        }

        let is_all_on_cpu = self.is_all_on_cpu_resident();
        let age = GlobalCtx::global().cache_age();
        if is_all_on_cpu {
            self.mark_on_cpu_since_if_all_on_cpu(age);
        }

        debug!("DevicesWrap::dump_to_cpu: 2. id={:?} device_idx={device_idx} is_all_on_cpu={is_all_on_cpu} age={age}", self.id);

        Ok(())
    }

    pub async fn drop_device(&self, device_idx: usize) -> Result<(), ThoError> {
        assert!(device_idx > 0, "can't drop cpu");
        drop(track_await!(self.cached_data[device_idx].write()).await.take());
        self.mark_not_resident_on_device(device_idx);

        Ok(())
    }



    pub fn is_on_cpu_before(&self, threshold_age: u64) -> bool {
        let on_cpu_since = perf_log!(sync_read_lock!(&self.on_cpu_since), "DevicesWrap::is_on_cpu_before on_cpu_since.read");
        let r = on_cpu_since.as_ref().map(|on_cpu_since| *on_cpu_since <= threshold_age).unwrap_or(false);

        debug!("DevicesWrap::is_on_cpu_before: self.id={:?} self.on_cpu_since={:?} r={r} threshold_age={threshold_age}", self.id, self.on_cpu_since);

        r
    }

    pub fn take_data(
        self,
        to_device_idx: usize,
    ) -> Result<O, ThoError> {
        let id = format!("{:?}", self.id);
        let mut cached_data = self.cached_data.into_vec();

        if to_device_idx < cached_data.len() {
            if let Some(data) = cached_data.remove(to_device_idx).into_inner() {//instance is being consumed, cached_data state doesn't matter
                debug!("take_data_prefer_device_or_any: to_device_idx={to_device_idx} data={data:?}");
                return Ok(data);
            }
        }

        Err(ThoError::new_with_cause(format!("DevicesWrap::take_data_prefer_device_or_any: no data for {id}"), ThoError::NotFound(Some(id))))
    }

    //needed only for TmpCached to return not yet loaded data retrieving TmpCached after dump to cpu.
    pub fn take_data_prefer_device_or_any(self, preferred_device_idx: usize) -> Result<O, ThoError> {//_or_any does not make sense, remove
        let id = format!("{:?}", self.id);
        let mut cached_data = self.cached_data.into_vec();

        if preferred_device_idx < cached_data.len() {
            if let Some(data) = cached_data.remove(preferred_device_idx).into_inner() {
                debug!("take_data_prefer_device_or_any: preferred_device_idx={preferred_device_idx} data={data:?}");
                return Ok(data);
            }
        }

        for slot in cached_data {
            if let Some(data) = slot.into_inner() {
                return Ok(data);
            }
        }

        Err(ThoError::new_with_cause(format!("DevicesWrap::take_data_prefer_device_or_any: no data for {id}"), ThoError::NotFound(Some(id))))
    }

    pub async fn load_from_any_device_(
        &self,
        to_tho_device_idx: usize,
    ) -> Result<RwLockReadGuard<'_, O>, ThoError>
    {
        debug!("DevicesWrap::from_any_device id={:?} to_device_idx={to_tho_device_idx}", self.id);
        trace!("DevicesWrap::from_any_device id={:?} to_device_idx={to_tho_device_idx} self.cached_data={:?}", self.id, self.cached_data);
        assert!(self.cached_data.len() > to_tho_device_idx, "Device index {to_tho_device_idx}(cache len is {}) too big {:?}", self.cached_data.len(), self.id);

        // Cleanup can move a GPU source to CPU after a scan passed CPU. Retry
        // both destination and source lookup to observe the committed backup.
        for _attempt in 0..2 {
            // Residency guards serialize materialized borrows with cache-area cleanup.
            if self.is_resident_on_device(to_tho_device_idx) {
                let lock = &self.cached_data[to_tho_device_idx];
                let read_guard = track_await!(lock.read()).await;
                if read_guard.is_some() {
                    return Ok(tokio::sync::RwLockReadGuard::map(
                        read_guard,
                        |val: &Option<O>| val.as_ref().expect("no value behind guard"),
                    ));
                }
                self.mark_not_resident_on_device(to_tho_device_idx);
                drop(read_guard);
            }

            debug!("DevicesWrap::from_any_device id={:?} copying to_device_idx={to_tho_device_idx}", self.id);

            for (source_device_idx, op) in self.cached_data.iter().enumerate() {
                if source_device_idx == to_tho_device_idx || !self.is_resident_on_device(source_device_idx) {
                    continue;
                }

                let source_read_guard = track_await!(op.read()).await;

                if source_read_guard.is_some() && self.copy_to_device.is_none() {
                    return Ok(RwLockReadGuard::map(
                        source_read_guard,
                        |val: &Option<O>| val.as_ref().expect("no value behind guard"),
                    ));
                }

                if let Some(src) = source_read_guard.as_ref() {
                    return track_await!(self.ensure_copied(to_tho_device_idx, src, true)).await.map(|v| v.expect("no value returned"));
                }
                self.mark_not_resident_on_device(source_device_idx);
                drop(source_read_guard);
            }
        }

        Err(ThoError::NotFound(Some(format!("load_from_any_device_: id={:?}", self.id))))
    }

    pub fn is_all_on_cpu(&self) -> impl Future<Output=bool> + Send {
        async move {
            trace!("DevicesWrap::is_all_on_cpu: self.id={:?}", self.id);

            let is_all_on_cpu = self.is_all_on_cpu_resident();
            debug!("DevicesWrap::is_all_on_cpu: self.id={:?} is_all_on_cpu={is_all_on_cpu}", self.id);
            is_all_on_cpu
        }
    }

    pub fn is_on_device(&self, tho_device_idx: usize) -> bool {
        self.resident_devices.get(tho_device_idx).map(|resident| resident.load(Ordering::Acquire)).unwrap_or(false)
    }

    pub fn mem_size(&self) -> u64 {
        self.mem_size
    }
}


impl<O> DevicesWrap<O>
where
    O: Data + Send + Sync
{

    pub fn id(&self) -> &<O as WithId>::ID
    where
        O: WithId,
    {
        // This mirrors the conditional WithId impl without imposing WithId on other operations.
        self.id.as_ref().downcast_ref::<O::ID>().expect("DevicesWrap stored an unexpected ID type")
    }

    async fn ensure_copied(
        &self,
        dst_idx: usize,
        src: &O,
        do_return: bool,
    ) -> Result<Option<RwLockReadGuard<'_, O>>, ThoError>
    {
        // Ensures inflight slot is cleared and waiters are notified even if the leader is canceled or panics.
        struct InflightGuard<'a> {
            inflight: &'a Mutex<Option<Arc<Notify>>>,
            notify: Arc<Notify>,
            cleared: bool,
        }

        impl<'a> InflightGuard<'a> {
            fn new(inflight: &'a Mutex<Option<Arc<Notify>>>, notify: Arc<Notify>) -> Self {
                Self {
                    inflight,
                    notify,
                    cleared: false,
                }
            }

            fn clear(&mut self) {
                if self.cleared {
                    return;
                }
                self.cleared = true;
                let mut g = perf_log!(sync_mutex_lock!(self.inflight));
                if g
                    .as_ref()
                    .map(|n| Arc::ptr_eq(n, &self.notify))
                    .unwrap_or(false)
                {
                    *g = None;
                }
                drop(g);
                self.notify.notify_waiters();
            }
        }

        impl Drop for InflightGuard<'_> {
            fn drop(&mut self) {
                self.clear();
            }
        }

        debug!("DevicesWrap::ensure_copied: self.id={:?} dst_idx={dst_idx} do_return={do_return} src={src:?}", self.id);

        loop {
            if self.is_resident_on_device(dst_idx) {
                let lock = &self.cached_data[dst_idx];
                let dst_read = track_await!(lock.read()).await;
                // ---- Fast path
                if dst_read.is_some() {
                    debug!("DevicesWrap::ensure_copied: self.id={:?} fast path", self.id);

                    if do_return {
                        return Ok(Some(RwLockReadGuard::map(dst_read, |v| {
                            v.as_ref().unwrap()
                        })));
                    } else {
                        return Ok(None);
                    }
                }
                self.mark_not_resident_on_device(dst_idx);
                drop(dst_read);
            }

            // ---- Try to join or become the leader
            // (Do not hold this lock while awaiting other locks, except briefly to set or read the notify)
            let leader_notify = match join_or_start_inflight(&self.inflights[dst_idx]) {
                InflightRole::Follower(notified) => {
                    // follower: wait until the leader finishes, then try fast path again
                    track_await!(notified)
                    .await;
                    continue;
                }
                InflightRole::Leader(notify) => notify,
            };

            // ---- Leader path
            let mut inflight_guard = InflightGuard::new(
                &self.inflights[dst_idx],
                leader_notify,
            );

            // Double-check: maybe dst was filled after we dropped the read lock
            if self.is_resident_on_device(dst_idx) {
                let lock = &self.cached_data[dst_idx];
                let dst_read = track_await!(lock.read()).await;
                if dst_read.is_some() {
                    inflight_guard.clear();
                    continue;
                }
                self.mark_not_resident_on_device(dst_idx);
                drop(dst_read);
            }

            let Some(copy_to_device) = &self.copy_to_device else {
                inflight_guard.clear();
                return Err(ThoError::new(format!("DevicesWrap::ensure_copied: no copy_to_device for {:?}", self.id)));
            };

            debug!("DevicesWrap::ensure_copied: self.id={:?} copy_to_device from src", self.id);
            let produced = track_await!(copy_to_device.copy(src, dst_idx)).await;

            let produced = match produced {
                Ok(v) => v,
                Err(e) => {
                    inflight_guard.clear();//TODO not needed, called from Drop?
                    return Err(e);
                }
            };


            // Install into dst (short critical section)
            {
                debug!("DevicesWrap::ensure_copied: self.id={:?} install copied to dst_idx={dst_idx} produced.device={:?}", self.id, produced.tho_device_idx());

                let lock = &self.cached_data[dst_idx];
                let mut w = track_await!(lock.write()).await;
                if w.is_none() {
                    *w = Some(produced);
                    self.mark_resident_on_device(dst_idx);
                }
            }

            // Clear inflight and wake followers
            inflight_guard.clear();//TODO not needed, called from Drop?
        }
    }

    #[cfg(test)]
    pub fn cached_data(&self) -> &Box<[tokio::sync::RwLock<Option<O>>]> {
        &self.cached_data
    }
}

pub trait BorrowRef
{
    type Out<'a> where Self: 'a;

    fn borrow_ref<'o>(&'o self, to_tho_device_idx: usize) -> impl Future<Output = Result<Self::Out<'o>, ThoError>> + Send
    where
        Self:'o;

}


impl <BR> BorrowRef for Arc<BR>
where
    BR:BorrowRef + Sync + Send,
{
    type Out<'a> = BR::Out<'a> where Self: 'a,  BR:'a;

    fn borrow_ref<'o>(&'o self, to_tho_device_idx: usize) -> impl Future<Output=Result<Self::Out<'o>, ThoError>> + Send
    where
        Self: 'o
    {
        async move {
            self.as_ref().borrow_ref(to_tho_device_idx).await
        }

    }
}

pub trait SelectRef<S>: BorrowRef
where
    S: ThoSelector + Clone + Debug + Send
{
    fn select_ref<'o>(&'o self, to_tho_device_idx: usize, selector: Option<S>/*TODO remove Option*/) -> impl Future<Output = Result<<Self as BorrowRef>::Out<'o>, ThoError>> + Send
    where
        S:'o, Self:'o;

}

impl <S, BR> SelectRef<S> for Arc<BR>
where
    BR: SelectRef<S> + Sync + Send,
    S:ThoSelector + Clone + Debug + Send
{
    fn select_ref<'o>(&'o self, to_tho_device_idx: usize, selector: Option<S>) -> impl Future<Output = Result<<Self as BorrowRef>::Out<'o>, ThoError>> + Send
    where
        S: 'o,
        Self: 'o
    {
        async move {
            <BR as SelectRef<S>>::select_ref(self.as_ref(), to_tho_device_idx, selector).await
        }
    }
}


/// Owned-data analogue of WithDataRef for sync-stage consume paths that need executor-controlled materialization.
pub trait ConsumeData {
    type O: Data + WithId + Send + Sync;
    //it should be run from OOM OOMHandler::consume where tensors can't be dumped to cpu while being used
    fn consume<R: Send, F, P>(
        self,
        to_device_idx: usize,
        run_ctx: P,
        do_run: F,
    ) -> impl Future<Output = ConsumeResult<Self::O, R>> + Send
    where
        F: FnOnce(Self::O, P) -> ConsumeResult<Self::O, R> + Send,
        P: RunCtx;

    fn burp(o:Self::O) -> impl Future<Output = Self> + Send;
}



impl <D, E> DevicesWrap<DataWithExtra<D, E>>
where
    D: Data + CopyToDevice + WithId + Sync + Send,
    E: Clone + Debug + Send + Sync
{

    pub async fn extra(&self) -> Result<E, ThoError> {
        // A concurrent dump_to_cpu can move data from a device slot to CPU while this method is scanning slots; the second pass avoids a false NotFound after waiting on the old device slot.
        for _attempt in 0..2 {
            for op in self.cached_data.iter() {
                let source_read_guard = track_await!(op.read()).await;

                if let Some(data_with_extra) = source_read_guard.as_ref() {
                    return Ok(data_with_extra.extra().clone());
                }
            }
        }

        Err(ThoError::new(format!("Can't get extra from {:?}", self.id)))
    }
}

#[cfg(test)]
mod tests {
    use crate::runner::global_ctx::MockGlobalCtx;
    use async_trait::async_trait;
    use parking_lot::Mutex;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use thoosa_shared::CopyToDevice;
    use thoosa_shared::copy_mode::CpuOrGpu;
    use thoosa_shared::tracked_await::track_await;
    use thoosa_shared::traits::{DummyWithId as DummyCachableId, RangeSelector, RunCtx, WithId};
    use tokio::sync::{Notify, Semaphore};
    use tokio::time::timeout;
    extern crate self as caching_runner;
    pub use crate::runner::common::*;
    use crate::runner::common::cachable::{DevicesWrap, InflightRole, SelectRef, join_or_start_inflight};
    use crate::runner::common::mock_run_ctx;
    use crate::runner::global_ctx;
    use crate::runner::global_ctx::mock_global_ctx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::runner::test_utils::MockDataWithId;

    #[derive(Clone, Copy, Debug)]
    enum CopyBehavior {
        BlockFirst,
        FailFirst,
        FailAlways,
    }

    #[derive(Debug)]
    struct ControlledCopyState {
        behavior: CopyBehavior,
        calls: AtomicUsize,
        entered: Semaphore,
        release: Semaphore,
    }

    impl ControlledCopyState {
        fn new(behavior: CopyBehavior) -> Arc<Self> {
            Arc::new(Self {
                behavior,
                calls: AtomicUsize::new(0),
                entered: Semaphore::new(0),
                release: Semaphore::new(0),
            })
        }
    }

    #[derive(Debug)]
    struct ControlledCopyData {
        id: DummyCachableId,
        device: usize,
        state: Arc<ControlledCopyState>,
    }

    impl ControlledCopyData {
        fn new(device: usize, behavior: CopyBehavior) -> (Self, Arc<ControlledCopyState>) {
            let state = ControlledCopyState::new(behavior);
            (
                Self {
                    id: DummyCachableId("controlled-copy"),
                    device,
                    state: state.clone(),
                },
                state,
            )
        }

        fn tho_device_idx(&self) -> usize {
            self.device
        }
    }

    impl WithId for ControlledCopyData {
        type ID = DummyCachableId;

        fn id(&self) -> &Self::ID {
            &self.id
        }
    }

    impl thoosa_shared::traits::Data for ControlledCopyData {
        type Metrics = [usize; 1];


        fn size(&self) -> Self::Metrics {
            [0]
        }

        fn tho_device_idx(&self) -> usize {
            self.device
        }

        fn mem_size(&self) -> u64 {
            0
        }
    }

    #[async_trait]
    impl CopyToDevice for ControlledCopyData {
        async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
            let call = self.state.calls.fetch_add(1, Ordering::SeqCst);
            if call == 0 && matches!(self.state.behavior, CopyBehavior::BlockFirst | CopyBehavior::FailFirst) {
                self.state.entered.add_permits(1);
                self.state.release.acquire().await.expect("release semaphore closed").forget();
            }
            if matches!(self.state.behavior, CopyBehavior::FailAlways) || call == 0 && matches!(self.state.behavior, CopyBehavior::FailFirst) {
                return Err(ThoError::new("controlled copy failure"));
            }

            Ok(Self {
                id: self.id.clone(),
                device: dst_device,
                state: self.state.clone(),
            })
        }
    }

    async fn wait_for_inflight_participants(wrap: &DevicesWrap<ControlledCopyData>, device_idx: usize, expected: usize) {
        timeout(Duration::from_secs(5), async {
            loop {
                let participants = wrap.inflights[device_idx].lock().as_ref().map(Arc::strong_count).unwrap_or(0);
                if participants >= expected {
                    return;
                }
                tokio::task::yield_now().await;
            }
        }).await.expect("timed out waiting for inflight participants");
    }

    #[tokio::test]
    async fn inflight_follower_observes_notify_before_first_poll() {
        let notify = Arc::new(Notify::new());
        let inflight = Mutex::new(Some(notify.clone()));
        let follower = match join_or_start_inflight(&inflight) {
            InflightRole::Follower(follower) => follower,
            InflightRole::Leader(_) => panic!("expected follower"),
        };

        notify.notify_waiters();
        timeout(Duration::from_secs(5), follower).await.expect("follower missed notification");
    }

    #[tokio::test]
    async fn test_ensure_copied_error_wakes_follower() -> Result<(), ThoError> {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);
        let (data, state) = ControlledCopyData::new(0, CopyBehavior::FailFirst);
        let tested = Arc::new(DevicesWrap::wrap(data));

        let leader = tokio::spawn({
            let tested = tested.clone();
            async move {
                let data = tested.load_from_any_device_(1).await?;
                Ok::<usize, ThoError>(data.tho_device_idx())
            }
        });
        state.entered.acquire().await.expect("entered semaphore closed").forget();

        let follower = tokio::spawn({
            let tested = tested.clone();
            async move {
                let data = tested.load_from_any_device_(1).await?;
                Ok::<usize, ThoError>(data.tho_device_idx())
            }
        });
        wait_for_inflight_participants(&tested, 1, 3).await;
        state.release.add_permits(1);

        assert!(leader.await.expect("leader task panicked").is_err());
        assert_eq!(timeout(Duration::from_secs(5), follower).await.expect("follower timed out").expect("follower task panicked")?, 1);
        assert_eq!(state.calls.load(Ordering::SeqCst), 2);
        assert!(tested.inflights[1].lock().is_none());
        Ok(())
    }

    #[tokio::test]
    async fn test_ensure_copied_cancellation_wakes_follower() -> Result<(), ThoError> {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);
        let (data, state) = ControlledCopyData::new(0, CopyBehavior::BlockFirst);
        let tested = Arc::new(DevicesWrap::wrap(data));

        let leader = tokio::spawn({
            let tested = tested.clone();
            async move {
                let data = tested.load_from_any_device_(1).await?;
                Ok::<usize, ThoError>(data.tho_device_idx())
            }
        });
        state.entered.acquire().await.expect("entered semaphore closed").forget();

        let follower = tokio::spawn({
            let tested = tested.clone();
            async move {
                let data = tested.load_from_any_device_(1).await?;
                Ok::<usize, ThoError>(data.tho_device_idx())
            }
        });
        wait_for_inflight_participants(&tested, 1, 3).await;
        leader.abort();
        assert!(leader.await.expect_err("leader was not cancelled").is_cancelled());

        assert_eq!(timeout(Duration::from_secs(5), follower).await.expect("follower timed out").expect("follower task panicked")?, 1);
        assert_eq!(state.calls.load(Ordering::SeqCst), 2);
        assert!(tested.inflights[1].lock().is_none());
        Ok(())
    }

    fn cached_mock_output() -> (super::CachedFuture<Arc<DevicesWrap<MockDataWithId>>>, Arc<DevicesWrap<MockDataWithId>>) {
        use futures_util::FutureExt;
        let data = Arc::new(DevicesWrap::wrap(MockDataWithId::for_device(1)));
        let value = Arc::clone(&data);
        let cached = super::CachedFuture::from(async move { Ok(value) }.boxed().shared());
        (cached, data)
    }

    #[tokio::test]
    async fn test_cleanup_tracks_areas_not_output_clones() -> Result<(), ThoError> {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);
        let (cached, data) = cached_mock_output();
        let cleanup_first = cached.cleanup_registration(1);
        let cleanup_last = cached.clone().cleanup_registration(1);
        let borrowed_handle = cached.clone().await?;
        cleanup_first().await?;
        assert!(data.is_on_device(1), "another area is still active");
        assert!(cached.has_cache_users());
        cleanup_last().await?;
        assert!(!cached.has_cache_users());
        assert!(!data.is_on_device(1), "cache and external handles must not suppress cleanup");
        assert!(borrowed_handle.is_on_device(0));
        cleanup_last().await?; // callback replay must not decrement the users twice.
        assert!(!cached.has_cache_users());
        Ok(())
    }

    #[tokio::test]
    async fn test_cleanup_waits_for_pending_result_and_uses_original_identity() -> Result<(), ThoError> {
        use futures_util::FutureExt;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);
        let (send, receive) = tokio::sync::oneshot::channel();
        let old_data = Arc::new(DevicesWrap::wrap(MockDataWithId::for_device(1)));
        let cached = super::CachedFuture::from(async move { receive.await.unwrap() }.boxed().shared());
        let cleanup = cached.cleanup_registration(1);
        let map = dashmap::DashMap::new();
        map.insert(0, cached.clone());
        let task = tokio::spawn(cleanup());
        tokio::task::yield_now().await;
        assert!(!task.is_finished());
        let (replacement, new_data) = cached_mock_output();
        map.insert(0, replacement);
        send.send(Ok(Arc::clone(&old_data))).unwrap();
        timeout(Duration::from_secs(5), task).await.unwrap().unwrap()?;
        assert!(!old_data.is_on_device(1));
        assert!(new_data.is_on_device(1), "stale cleanup must not look up a replacement by key");
        Ok(())
    }

    #[tokio::test]
    async fn test_cleanup_waits_for_borrow_and_allows_new_cached_user() -> Result<(), ThoError> {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);
        let (cached, data) = cached_mock_output();
        let cleanup = cached.cleanup_registration(1);
        cached.clone().await?;
        let borrowed = data.load_from_any_device_(1).await?;
        let task = cleanup();
        futures_util::pin_mut!(task);
        // Poll to the residency lock so this race does not depend on scheduling.
        assert!(futures_util::poll!(&mut task).is_pending());
        let next_cleanup = cached.cleanup_registration(1);
        let new_handle = timeout(Duration::from_secs(5), cached.clone()).await.unwrap()?;
        assert_eq!(borrowed.device(), &1);
        drop(borrowed);
        timeout(Duration::from_secs(5), task).await.unwrap()?;
        let restored = new_handle.load_from_any_device_(1).await?;
        assert_eq!(restored.device(), &1);
        drop(restored);
        next_cleanup().await?;
        assert!(new_handle.is_on_device(0));
        assert!(!new_handle.is_on_device(1));
        Ok(())
    }

    #[tokio::test]
    async fn test_cleanup_restore_race_observes_cpu_backup() -> Result<(), ThoError> {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);
        let (data, state) = ControlledCopyData::new(1, CopyBehavior::BlockFirst);
        let data = Arc::new(DevicesWrap::wrap(data));
        let cleanup = tokio::spawn({
            let data = Arc::clone(&data);
            async move { data.cleanup_device(1).await }
        });
        state.entered.acquire().await.unwrap().forget();
        let cpu_reader = data.load_from_any_device_(0);
        let gpu_reader = data.load_from_any_device_(1);
        futures_util::pin_mut!(cpu_reader, gpu_reader);
        assert!(futures_util::poll!(&mut cpu_reader).is_pending());
        assert!(futures_util::poll!(&mut gpu_reader).is_pending());
        state.release.add_permits(1);
        timeout(Duration::from_secs(5), cleanup).await.unwrap().unwrap()?;
        assert_eq!(timeout(Duration::from_secs(5), cpu_reader).await.unwrap()?.tho_device_idx(), 0);
        assert_eq!(timeout(Duration::from_secs(5), gpu_reader).await.unwrap()?.tho_device_idx(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_dump_to_cpu_error_preserves_source_data() {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);
        let (data, state) = ControlledCopyData::new(1, CopyBehavior::FailAlways);
        let tested = DevicesWrap::wrap(data);

        assert!(tested.dump_to_cpu(1).await.is_err());
        assert!(tested.cached_data[0].read().await.is_none());
        assert!(tested.cached_data[1].read().await.is_some());
        assert!(tested.is_on_device(1));
        assert_eq!(state.calls.load(Ordering::SeqCst), 1);
        assert!(tested.inflights[0].lock().is_none());
    }

    #[tokio::test]
    async fn test_dump_to_cpu() -> Result<(), ThoError> {

        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);

        let data = MockDataWithId::whatever();
        let tested = DevicesWrap::wrap(data);
        assert!(tested.is_all_on_cpu().await);
        let lock = &tested.cached_data[1];
        assert!(track_await!(lock.read()).await.is_none());

        let ctx_for_data = mock_run_ctx::default(1);

        let data_on_1 = tested.select_ref(1, None).await?;

        assert_eq!(data_on_1.as_ref().device(), &1);
        drop(data_on_1);//dump_to_cpu: can't drop self.id=DummyCachableId("id"), locked


        let lock = &tested.cached_data[1];
        assert!(track_await!(lock.read()).await.is_some());
        assert!(!tested.is_all_on_cpu().await);

        tested.dump_to_cpu(1).await?;
        let lock = &tested.cached_data[1];
        assert!(track_await!(lock.read()).await.is_none());
        assert!(tested.is_all_on_cpu().await);
        assert!(tested.is_on_cpu_before(0));

        Ok(())
    }



    #[tokio::test]
    async fn test_with_data() -> Result<(), ThoError> {
        let ctx = MockGlobalCtx::global_context();
        ctx.expect().returning(|| {
            Box::leak(Box::new(
                mock_global_ctx::build_default(),
            )) as &'static MockGlobalCtx
        });

        let wrap_a = Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "a")));
        let wrap_b = Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "d")));
        let wrap_vec = vec![
            Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "b"))),
            Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "c"))),
        ];
        let wrap_vec_second = vec![
            Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "e"))),
            Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "f"))),
        ];

        //TODO check asserts were actually invoked
        let ctx_for_data = mock_run_ctx::default(1);
        let a = wrap_a.select_ref(ctx_for_data.tho_device_idx(), None::<RangeSelector>).await?;
        let first = futures_util::future::try_join_all(wrap_vec.iter().map(|item| item.select_ref(ctx_for_data.tho_device_idx(), None::<RangeSelector>))).await?;
        let b = wrap_b.select_ref(ctx_for_data.tho_device_idx(), None::<RangeSelector>).await?;
        let second = futures_util::future::try_join_all(wrap_vec_second.iter().map(|item| item.select_ref(ctx_for_data.tho_device_idx(), None::<RangeSelector>))).await?;
        {
                assert_eq!(a.as_ref().device(), &1);
                assert_eq!(b.as_ref().device(), &1);
                assert_eq!(first.len(), 2);
                assert!(first.iter().all(|item| *item.as_ref().device() == 1));
                assert_eq!(second.len(), 2);
                assert!(second.iter().all(|item| *item.as_ref().device() == 1));
                Ok::<(), ThoError>(())
        }?;
        Ok(())
    }

    #[tokio::test]
    async fn test_with_data_single_val() -> Result<(), ThoError> {
        let ctx = MockGlobalCtx::global_context();
        ctx.expect().returning(|| {
            Box::leak(Box::new(
                mock_global_ctx::build_default(),
            )) as &'static global_ctx::MockGlobalCtx
        });

        let wrap_a = Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "a")));

        let ctx_for_data = mock_run_ctx::default(1);
        let a = wrap_a.select_ref(ctx_for_data.tho_device_idx(), None::<RangeSelector>).await?;
        {
                assert_eq!(a.as_ref().device(), &1);
                Ok::<(), ThoError>(())
        }?;
        Ok(())
    }
}
