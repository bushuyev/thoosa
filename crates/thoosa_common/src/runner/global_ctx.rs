use thoosa_common::tch_error::tch_result;
use crate::runner::purge::FuturesRunnerMapPurger;
use log::{debug, error, info, warn};
#[allow(unused_imports)]
use mockall::automock;
use crate::copy_ctx::{CopyCtx, CudaActorHandle, CudaLocalActorHandle};
use crate::ipc::cuda_process::CudaIPCActorHandle;
use crate::runner::cache_areas::{CacheAreaId, CacheAreasKeeper};
use crate::runner::syn_stage_runner::SyncStageRunner;
use crate::runner::cuda_recovery::CudaRecoveryCoordinator;
use crate::runner::purger::Purger;
use crate::tho_device::ThoDevice;
use futures::FutureExt;
use futures_util::future::BoxFuture;
use std::collections::{HashMap, HashSet};
use std::any::{Any, TypeId};
use mockall::Any as _;
use crate::runner::common::cachable::{CachedPurgeOutput, StageCache};
use thoosa_shared::traits::{Named, Stage, WithId};
use std::fmt::Debug;
use std::future::ready;
use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use thoosa_shared::collect_futures_eta::{CollectEtaInvocationGuard, collect_eta_record_completion};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::CachableIdTr;
use thoosa_shared::{CudaActivityGate, ThoError, async_perf_log, async_read_owned_lock, async_write_owned_lock, perf_log, sync_write_lock};
use thoosa_shims::context::ThoCudaContext;
use tokio::sync::RwLock as AsyncRwLock;
use tracing::Instrument;
use thoosa_shims::arena::{shared_arena_dup_fd, shared_arena_init};

pub static GLB_SHUTTING_DOWN:AtomicBool = AtomicBool::new(false);

///supposed to be internal, don't use it in Stages except when instantiating to check if it should be ipc proxy
#[derive(Debug)]
pub struct GlobalCtx {

    devices: Vec<DeviceExecutionCtx>,

    caches: Arc<CacheRegistry>,
    is_all_on_cpu: Vec<AtomicBool>,
    cache_time: AtomicU64, //for tracking CPU-only lifetime
    purger: Purger,

    is_ipc: bool,
    arena_fd:Option<OwnedFd> //TODO needed in IPC mode only
}

// Keep actor and runner ownership together: direct actor copies and scheduled
// stages must use the very same device gate. The runner retains its coordinator.
#[derive(Debug)]
struct DeviceExecutionCtx {
    runner: SyncStageRunner,
    cuda_actor: Option<Box<dyn CudaActorHandle>>,
}

impl DeviceExecutionCtx {
    fn cpu() -> Self {
        Self { runner: SyncStageRunner::new_cpu(1024), cuda_actor: None }
    }
}

static GLOBAL_CTX: OnceLock<GlobalCtx> = OnceLock::new();
static PROCESS_FAILURE: std::sync::LazyLock<tokio::sync::watch::Sender<Option<ThoError>>> = std::sync::LazyLock::new(|| tokio::sync::watch::channel(None).0);

// Use the same identity for cache lookup and OOM exclusions. Display names do
// not distinguish generic stage types; owned instance names need no leaks.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CacheKey {
    Typed(TypeId),
    Named(Arc<str>),//for StageWithState
}

impl CacheKey {
    pub fn of<S: 'static + ?Sized>() -> Self {
        Self::Typed(TypeId::of::<S>())
    }
}

#[derive(Debug)]
struct CacheRegistryEntry {
    name: Arc<str>,
    map: Option<Arc<dyn Any + Send + Sync>>,
    purger: Arc<dyn FuturesRunnerMapPurger>,
    gate: Arc<AsyncRwLock<()>>,
}

type CacheSnapshot = Vec<(CacheKey, Arc<CacheRegistryEntry>)>;

/// Application-owned stage maps and instance purgers share a registry, while
/// typed and named keys remain separate namespaces.
#[derive(Debug, Default)]
pub struct CacheRegistry {
    entries: parking_lot::RwLock<HashMap<CacheKey, Arc<CacheRegistryEntry>>>,
}

impl CacheRegistry {
    pub fn cache_for<S>(&self) -> Result<Arc<StageCache<S>>, ThoError>
    where
        S: Stage + Named + 'static,
        S::Out: CachedPurgeOutput + WithId,
        <S::Out as WithId>::ID: Sized,
    {
        let key = CacheKey::of::<S>();
        let existing = self.entries.read().get(&key).cloned();
        let entry = if let Some(entry) = existing {
            entry
        } else {
            // Publish the map and its purger together. No registry lock is held
            // over an await or while executing tensor/purge operations.
            let mut caches = self.entries.write();
            Arc::clone(caches.entry(key).or_insert_with(|| {
                let map = Arc::new(StageCache::<S>::new());
                Arc::new(CacheRegistryEntry {
                    name: S::name().into(),
                    map: Some(Arc::clone(&map) as Arc<dyn Any + Send + Sync>),
                    purger: map,
                    gate: Arc::new(AsyncRwLock::new(())),
                })
            }))
        };
        let erased = entry.map.as_ref().ok_or_else(|| ThoError::new(format!("missing typed cache map for {}", std::any::type_name::<S>())))?;
        // An internal registry invariant violation must never become an
        // unchecked pointer reinterpretation with the caller's output type.
        match Arc::clone(erased).downcast::<StageCache<S>>() {
            Ok(map) => Ok(map),
            Err(e) => Err(ThoError::new(format!("cache type mismatch for {}: {e:?}", std::any::type_name::<S>()))),
        }
    }

    fn snapshot(&self) -> CacheSnapshot {
        self.entries.read().iter().map(|(key, entry)| (key.clone(), Arc::clone(entry))).collect()
    }
}

#[derive(Debug, Clone)]
pub struct RegisteredCacheHandle {
    key: CacheKey,
    gate: Arc<AsyncRwLock<()>>,
}

impl RegisteredCacheHandle {
    pub fn key(&self) -> &CacheKey {
        &self.key
    }

    pub async fn lock_exclusive(&self) -> tokio::sync::OwnedRwLockWriteGuard<()> {
        async_perf_log!(
            async_write_owned_lock!(Arc::clone(&self.gate))
        )
    }

    #[cfg(any(test, feature = "mock_global_ctx"))]
    pub fn for_tests(name: String) -> Self {
        Self {
            key: CacheKey::Named(name.into()),
            gate: Arc::new(AsyncRwLock::new(())),
        }
    }
}

#[cfg_attr(any(test, feature = "mock_global_ctx"), automock)]
impl GlobalCtx {

    // TODO num_devices is to be replaces with Cuda devices list kept in a static const.
    // Currently it's assumed that there is single Cuda(0) device and first device is always CPU. So if num_devices==2 then devices would be [Cpu, Cuda(0)]
    fn new(num_devices: usize) -> anyhow::Result<Self> {
        debug!("GlobalCtx::new: num_devices={num_devices}");

        assert!(num_devices > 0);//TODO test if it works without GPU

        let mut devices = Vec::with_capacity(num_devices);
        devices.push(DeviceExecutionCtx::cpu());
        for tho_device_idx in 1..num_devices {
            let gate = Arc::new(CudaActivityGate::new(tho_device_idx));

            let recovery = CudaRecoveryCoordinator::with_backend(
                Arc::clone(&gate),
                Arc::new(GlobalCtx::cuda_restart),
                None,
                Some(Arc::new(GlobalCtx::fail_process))
            );

            devices.push(DeviceExecutionCtx {
                runner: SyncStageRunner::new_with_recovery(1024, recovery),
                cuda_actor: Some(Box::new(CudaLocalActorHandle::new(Arc::clone(&gate))?)),
            });
        }

        Ok(Self {
            devices,

            caches: Arc::new(CacheRegistry::default()),

            is_all_on_cpu: (0..num_devices).map(|_| AtomicBool::new(false)).collect(),


            cache_time: Default::default(),
            purger: Purger::new(),

            is_ipc: false,
            arena_fd: None,
        })
    }

    async fn new_ipc(num_devices: usize, cuda_worker_args: Vec<String>) -> anyhow::Result<Self> {
        debug!("GlobalCtx::new: num_devices={num_devices}");

        assert!(num_devices > 0);

        const ARENA_SIZE: usize = 48 * 1024 * 1024 * 1024; //TODO config

        let rc = unsafe {
            shared_arena_init(
                ARENA_SIZE
            )
        };

        if rc != 0 {
            anyhow::bail!("shared_arena_init failed");
        }

        let raw_fd = unsafe {
            shared_arena_dup_fd()
        };

        if raw_fd < 0 {
            anyhow::bail!("shared_arena_dup_fd failed");
        }


        let arena_fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };


        let mut devices = Vec::with_capacity(num_devices);
        devices.push(DeviceExecutionCtx::cpu());
        for tho_device_idx in 1..num_devices {
            let gate = Arc::new(CudaActivityGate::new(tho_device_idx));

            let recovery = CudaRecoveryCoordinator::with_backend(
                Arc::clone(&gate),
                Arc::new(GlobalCtx::cuda_restart),
                Some(Arc::new(|device, authority| Box::pin(async move { GlobalCtx::global().cuda_actor(device).retire_backend(authority).await }))),
                None
            );


            devices.push(DeviceExecutionCtx {
                runner: SyncStageRunner::new_with_recovery(1024, recovery),
                // Each actor retains its own descriptor so replacement workers can
                // remap the parent-owned arena independently of other devices.
                cuda_actor: Some(Box::new(CudaIPCActorHandle::new(Arc::clone(&gate), cuda_worker_args.clone(), arena_fd.try_clone()?).await?)),
            });
        }

        Ok(Self {
            devices,

            caches: Arc::new(CacheRegistry::default()),

            is_all_on_cpu: (0..num_devices).map(|_| AtomicBool::new(false)).collect(),

            cache_time: Default::default(),
            purger: Purger::new(),

            is_ipc: true,
            arena_fd: Some(arena_fd),
        })
    }

    fn cuda_restart(tho_device_idx: usize) -> BoxFuture<'static, Result<(), ThoError>> {
        async move {
            // This device's coordinator invokes the hook after its activity drains.
            info!("GlobalCtx::cuda_restart: tho_device_idx={tho_device_idx}");

            //drop slots belonging to the device being reset and remove cache entries with no other devices' (CPU) data, i.e. if the entry was not copied from or dumped to CPU yet
            let r = track_await!(GlobalCtx::global()
                .drop_data_on_device(tho_device_idx, "drop"))
                .await;

            info!("GlobalCtx::cuda_restart: drop_device done with result={r:?}");

            let gate = GlobalCtx::global().cuda_activity_gate(tho_device_idx).expect("no CUDA gate");

            info!("GlobalCtx::cuda_restart: gate={gate:?}");

            let authority = thoosa_shared::CudaActivityScope::recovery_ticket(&gate)?;

            info!("GlobalCtx::cuda_restart: authority={authority:?}");

            GlobalCtx::global()
                .cuda_actor(tho_device_idx)
                .restart_backend(authority)
                .await?;


            info!("GlobalCtx::cuda_restart: all done");

            Ok(())
        }.boxed()
    }

    /// Instance caches use the same gate as dispatch when creating device state.
    pub fn cuda_activity_gate(&self, tho_device_idx: usize) -> Option<Arc<CudaActivityGate>> {
        self.devices[tho_device_idx].runner.cuda_gate().cloned()
    }

    #[cfg(any(test, feature = "mock_global_ctx"))]
    fn mock_runner(&self, device: usize) -> &SyncStageRunner {
        // Keep the generated mock's owned runner separate from real delegation.
        self.get_runner(device)
    }

    #[cfg(test)]
    pub async fn restart_cuda_device(&self, device: usize) -> Result<(), ThoError> {
        let ticket = self.get_runner(device).request_cuda_reset(ThoError::ContextReset)?;
        // Admitted callers must release their activity before the independent
        // coordinator can drain. Waiting here would wait on the caller itself.
        if thoosa_shared::CudaActivityScope::current().is_some() {
            return Err(ThoError::ContextReset);
        }
        if let Some(ticket) = ticket {
            self.get_runner(device).cuda_gate().expect("CUDA device").wait_for_pause_to_finish(ticket).await?;
        }
        Ok(())
    }

    pub fn run_purge_gpu(&self, closures: Vec<Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>> + Send + Sync>>) ->Result<(), ThoError>{
        self.purger.run_purge_gpu(closures)
    }

    ///num_devices: number of devices - CPU + GPU count
    pub fn init_devices(num_devices: usize) -> anyhow::Result<&'static GlobalCtx> {
        debug!("GlobalCtx::init_devices: num_devices={num_devices}");

        if GLOBAL_CTX.set(GlobalCtx::new(num_devices)?).is_err() {
            anyhow::bail!("`init_devices` must only be called once at program start");
        }

        Ok(Self::global())
    }

    ///num_devices: number of devices - CPU + GPU count
    pub async fn init_devices_ipc(num_devices: usize/*TODO replace with DeviceInitStruct vec*/, cuda_worker_args: Vec<String>/*TODO move to device init struct*/) -> anyhow::Result<&'static GlobalCtx> {
        debug!("GlobalCtx::init_devices: num_devices={num_devices}");

        if GLOBAL_CTX.set(GlobalCtx::new_ipc(num_devices, cuda_worker_args).await?).is_err() {
            anyhow::bail!("`init_devices` must only be called once at program start");
        }

        Ok(Self::global())
    }


    pub fn cache_registry(&self) -> Arc<CacheRegistry> {
        Arc::clone(&self.caches)
    }

    pub fn register_cache(&self, cache: (String, Arc<dyn FuturesRunnerMapPurger>)) -> RegisteredCacheHandle {//TODO rename DataPurger?
        debug!("GlobalCtx::register_cache: cache={}", cache.0);
        let cache_type = cache.1.type_name();
        let name: Arc<str> = cache.0.into();
        let key = CacheKey::Named(Arc::clone(&name));
        let gate = Arc::new(AsyncRwLock::new(()));
        let entry = Arc::new(CacheRegistryEntry {
            name,
            map: None,
            purger: cache.1,
            gate: Arc::clone(&gate),
        });

        // Check before inserting so a duplicate cannot replace a live cache.
        let mut caches = perf_log!(sync_write_lock!(self.caches.entries));
        assert!(!caches.contains_key(&key), "duplicate {key:?}/{cache_type}");
        caches.insert(key.clone(), entry);

        RegisteredCacheHandle { key, gate }
    }

    fn caches_snapshot(&self, descr: &'static str) -> CacheSnapshot {
        perf_log!(self.caches.snapshot(), "{descr}")
    }

    async fn lock_cache_shared(entry: &Arc<CacheRegistryEntry>, ctx: &'static str) -> tokio::sync::OwnedRwLockReadGuard<()> {
        async_perf_log!(async_read_owned_lock!(Arc::clone(&entry.gate), "{ctx}"), "{ctx}")
    }

    pub fn remove_cache(&self, key: &CacheKey) {
        debug!("GlobalCtx::remove_cache: key={:?}", key);

        assert!(
            perf_log!(sync_write_lock!(self.caches.entries, "key={key:?}"))
            .remove(key)
            .is_some()
        );

        debug!("GlobalCtx::remove_cache: {:?} lock acquired/released, removed ", key);
    }

    #[inline]
    pub fn global() -> &'static Self {
        GLOBAL_CTX.get().expect("call `init_devices` first")
    }

    pub fn is_installed() -> bool {//used to check if it's not worker in KeyHolder::drop, replace with KeyHolder::is_main bool field?
        GLOBAL_CTX.get().is_some()
    }


    #[cfg(any(test, feature = "mock_global_ctx"))]
    pub fn real_global() -> &'static GlobalCtx {
        GLOBAL_CTX.get().expect("call `init_devices` first")
    }

    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    #[cfg(any(test, feature = "mock_global_ctx"))]
    pub fn is_all_on_cpu(&self, device_idx: usize, force_check: bool) -> bool {
        if !force_check {
            return self.is_all_on_cpu[device_idx].load(Relaxed);
        }

        let is_all_on_cpu = match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(|| handle.block_on(self.is_all_on_cpu_force_check(device_idx)))
            }
            _ => futures::executor::block_on(self.is_all_on_cpu_force_check(device_idx)),
        };
        self.is_all_on_cpu[device_idx].store(is_all_on_cpu, Relaxed);
        is_all_on_cpu
    }

    async fn is_all_on_cpu_force_check(&self, device_idx: usize) -> bool {
        let maps = self.caches_snapshot("GlobalCtx::is_all_on_cpu_force_check: timed out waiting for caches read lock");

        for (_cache_key, entry) in maps.iter() {
            let type_name = entry.name.as_ref();
            let _cache_guard = track_await!(Self::lock_cache_shared(entry, "GlobalCtx::is_all_on_cpu_force_check: timed out waiting for cache gate read lock")).await;
            if track_await!(entry.purger.has_gpu_data(device_idx)).await {
                debug!("GlobalCtx::is_all_on_cpu_force_check: type_name={type_name} has data on device_idx={device_idx}");
                return false;
            }
        }

        true
    }

    pub async fn reset_all_on_cpu(&self, device_idx: usize) {
        self.is_all_on_cpu[device_idx].store(false, Relaxed);
    }


    async fn debug_caches_snapshot(&self, device_idx: usize, descr: &str, maps: &CacheSnapshot) {
        debug!("GlobalCtx::debug_caches: descr={descr:?}");

        let mut total_gpu_mem = 0;
        for (_cache_key, entry) in maps.iter() {
            let type_name = entry.name.as_ref();
            let _cache_guard = track_await!(Self::lock_cache_shared(entry, "GlobalCtx::debug_caches: timed out waiting for cache gate read lock")).await;
            info!("GlobalCtx::debug_gpu: descr={descr} type_name={type_name}");
            match track_await!(entry.purger.debug_gpu(device_idx)).await {
                Ok(v) => total_gpu_mem += v,
                Err(e) => log::error!("GlobalCtx::debug_gpu: failed with {e:?}"),
            }
        }

        if !self.is_ipc {
            info!(
                "GlobalCtx::debug_gpu: descr={descr} device_idx={device_idx} total_gpu_mem={total_gpu_mem} device_stats={:?}",
                thoosa_shims::device_stats::get_device_stats((device_idx - 1) as i32)
            );
        }
    }

    pub async fn debug_caches(&self, device_idx: usize, descr: &str) {
        let maps = self.caches_snapshot("GlobalCtx::debug_caches: timed out waiting for caches read lock");
        track_await!(self.debug_caches_snapshot(device_idx, descr, &maps)).await;
    }

    /// Keep only the listed output IDs in each identified cache on the device.
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="GlobalCtx::dump_all_to_cpu"))]
    pub async fn dump_all_to_cpu<'a>(
        &'a self,
        required_ids: &HashMap<CacheKey, &'a HashSet<&'a dyn CachableIdTr>>,
        tho_device_idx: usize,
        descr: &str,
    ) -> Result<(), ThoError> {
        assert_ne!(tho_device_idx, 0);

        debug!("GlobalCtx::dump_all_to_cpu: tho_device_idx={tho_device_idx} required_ids={required_ids:?} descr={descr}");

        let maps = self.caches_snapshot("GlobalCtx::dump_all_to_cpu: timed out waiting for caches read lock");
        let dump_eta_guard = CollectEtaInvocationGuard::new(&format!("dump_all_to_cpu_{descr}"), maps.len(), 1);

        for (cache_key, entry) in maps.iter() {
            let type_name = entry.name.as_ref();
            let _cache_guard = track_await!(Self::lock_cache_shared(entry, "GlobalCtx::dump_all_to_cpu: timed out waiting for cache gate read lock")).await;
            if !track_await!(entry.purger.has_gpu_data(tho_device_idx)).await {
                debug!("GlobalCtx::dump_all_to_cpu: type_name={type_name} has no data on device_idx={tho_device_idx}, skipping");
                collect_eta_record_completion(dump_eta_guard.id());
                continue;
            }

            debug!("GlobalCtx::dump_all_to_cpu: type_name={type_name}");
            track_await!(entry.purger.purge_gpu(required_ids.get(cache_key).map(|v| *v), tho_device_idx, type_name)).await?;
            debug!("GlobalCtx::dump_all_to_cpu: type_name={type_name} descr={descr} done");
            collect_eta_record_completion(dump_eta_guard.id());
        }

        self.is_all_on_cpu[tho_device_idx].store(true, Relaxed);
        debug!("GlobalCtx::dump_all_to_cpu: tho_device_idx={tho_device_idx} required_ids={required_ids:?} descr={descr} done");

        #[cfg(tho_debug)]
        {
            track_await!(self.debug_caches(tho_device_idx, descr)).await;

            self.cuda_actor(tho_device_idx).debug_worker().await?;
        }

        Ok(())
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="GlobalCtx::dump_all_to_cpu"))]
    pub async fn drop_data_on_device<'a>(
        &'a self,
        tho_device_idx: usize,
        descr: &str,
    ) -> Result<(), ThoError> {
        assert_ne!(tho_device_idx, 0);
        info!("GlobalCtx::drop_data_on_device: tho_device_idx={tho_device_idx} descr={descr}");
        let maps = self.caches_snapshot("GlobalCtx::drop_data_on_device: timed out waiting for caches read lock");

        let dump_eta_guard = CollectEtaInvocationGuard::new(&format!("drop_device{descr}"), maps.len(), 1);

        for (_cache_key, entry) in maps.iter() {
            let type_name = entry.name.as_ref();
            let _cache_guard = track_await!(Self::lock_cache_shared(entry, "GlobalCtx::drop_data_on_device: timed out waiting for cache gate read lock")).await;
            debug!("GlobalCtx::drop_data_on_device: type_name={type_name}");
            match track_await!(entry.purger.drop_gpu(tho_device_idx, type_name)).await {
                Ok(_) => {
                    debug!("GlobalCtx::drop_data_on_device: type_name={type_name} descr={descr} done");
                }
                Err(e) => {
                    warn!("GlobalCtx::drop_data_on_device: type_name={type_name} descr={descr} failed with e={e:?}");
                }
            }

            collect_eta_record_completion(dump_eta_guard.id());
        }

        self.is_all_on_cpu[tho_device_idx].store(true, Relaxed);
        info!("GlobalCtx::drop_data_on_device: tho_device_idx={tho_device_idx} descr={descr} done");

        Ok(())
    }


    pub async fn purge_cpu(&self, retain_last: u64) {//retain_last - caches younger than current cache time - retain_last should not be purged
        debug!("GlobalCtx::purge_cpu: retain_last={retain_last}");


        let purge_until = self.cache_time.load(Relaxed).saturating_sub(retain_last);


        if purge_until > 0 {
            let maps = self.caches_snapshot("GlobalCtx::purge_cpu: timed out waiting for caches read lock");

            #[cfg(tho_debug)]
            track_await!(self.debug_caches_snapshot(1, "before purge_cpu", &maps)).await;

            for (_cache_key, entry) in maps.iter() {
                let type_name = entry.name.as_ref();
                let _cache_guard = track_await!(Self::lock_cache_shared(entry, "GlobalCtx::purge_cpu: timed out waiting for cache gate read lock")).await;
                let purged = track_await!(entry.purger.purge_cpu(purge_until)).await;
                debug!("GlobalCtx::purge_cpu: purged {purged} from {type_name}");
            }

            #[cfg(tho_debug)]
            track_await!(self.debug_caches_snapshot(1, "after purge_cpu", &maps)).await;
        }

        debug!("GlobalCtx::purge_cpu: purge_until={purge_until} done");
    }

    pub fn do_age(&self) {
        self.cache_time.fetch_add(1, Relaxed);
    }

    pub fn cache_age(&self) -> u64 {
        self.cache_time.load(Relaxed)
    }


    pub async fn shutdown(&self) {
        GLB_SHUTTING_DOWN.store(true, Relaxed);

        for (i, actor) in self.devices.iter().enumerate().filter_map(|(i, device)| device.cuda_actor.as_ref().map(|actor| (i, actor))) {
            if let Err(e) = actor.shutdown().await {
                error!("Got {e} while shutting down actor {i}");
            }
        }
    }

    fn fail_process(error: ThoError) {
        // Only local process-invalidating failures broaden the shutdown scope.
        GLB_SHUTTING_DOWN.store(true, Relaxed);
        for device in &Self::global().devices {
            if let Some(gate) = device.runner.cuda_gate() {
                gate.fail_process(error.clone());
            }
        }
        PROCESS_FAILURE.send_if_modified(|failure| {
            if failure.is_some() {
                return false;
            }
            *failure = Some(error);
            true
        });
    }

    pub fn process_failure() -> Option<ThoError> {
        PROCESS_FAILURE.borrow().clone()
    }

    pub async fn wait_for_process_failure() -> ThoError {
        let mut failure = PROCESS_FAILURE.subscribe();
        loop {
            if let Some(error) = failure.borrow_and_update().clone() { return error; }
            if failure.changed().await.is_err() { return ThoError::new("process failure notifier closed"); }
        }
    }

    #[inline]
    pub fn is_shutting_down()->bool {
        GLB_SHUTTING_DOWN.load(Relaxed)
    }

    //for tests and temporary fixes - use CudaActors for ipc/no ipc implementations
    pub fn is_ipc(&self) -> bool {
        self.is_ipc
    }
}

impl GlobalCtx {

    pub fn get_runner(&self, tho_device: usize) -> &SyncStageRunner {
        debug!("GlobalCtx::get_executor: tho_device={tho_device} self.devices.len={}", self.devices.len());
        &self.devices[tho_device].runner
    }

    pub fn cuda_actor<D:Into<ThoDevice>+ 'static>(&self, device:D)-> &Box<dyn CudaActorHandle> {
        let device:ThoDevice = device.into();
        self.devices.get(device.tho_device_idx()).and_then(|device| device.cuda_actor.as_ref()).expect(&format!("No CudaActor for device={device:?}"))
    }
}

#[cfg(any(test, feature = "mock_global_ctx"))]
pub mod mock_global_ctx {
    use std::fmt::{Debug, Formatter};
    use crate::runner::syn_stage_runner::SyncStageRunner;
    use crate::runner::global_ctx::MockGlobalCtx;
    use crate::runner::global_ctx::{GlobalCtx, RegisteredCacheHandle};
    use log::debug;
    use tokio::task::block_in_place;

    #[derive(Clone, Copy, Debug)]
    pub enum AcquireBehavior {
        Grant,
        Deny,
        Boom(&'static str)
    }

    #[derive(Debug)]
    pub struct MockGlobalCtxBuildParams {
        pub age: u64,
        pub device_count: usize,
        pub next_cache_areas: Vec<u64>,
        pub acquire_many: AcquireBehavior,
        pub acquire_one: AcquireBehavior,
    }

    impl Default for MockGlobalCtxBuildParams {
        fn default() -> Self {
            Self {
                age: 0,
                device_count: 2,
                next_cache_areas: vec![1],
                acquire_many: AcquireBehavior::Grant,
                acquire_one: AcquireBehavior::Grant,
            }
        }
    }

    #[cfg(any(test, feature = "mock_global_ctx"))]
    pub trait GlobalCtxMockGuard:Debug {}

    #[cfg(any(test, feature = "mock_global_ctx"))]
    struct GlobalCtxMockContexts<G, S, I> {
        _global_context: G,
        _shutdown_context: S,
        _is_installed_ctx:I
    }

    #[cfg(any(test, feature = "mock_global_ctx"))]
    impl<G, S, I> GlobalCtxMockGuard for GlobalCtxMockContexts<G, S, I> {}

    impl<G, S, I> Debug for GlobalCtxMockContexts<G, S, I> {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("GlobalCtxMockContexts").finish()
        }
    }


    #[cfg(any(test, feature = "mock_global_ctx"))]
    pub struct MockGlobalCtxBuilder {
        mock: MockGlobalCtx,
        real: Option<&'static GlobalCtx>
    }

    #[cfg(any(test, feature = "mock_global_ctx"))]
    impl MockGlobalCtxBuilder {
        pub fn with_reset_all_on_cpu(mut self) -> Self {
            if let Some(real) = self.real {

                let _ = self.mock.expect_reset_all_on_cpu().returning({

                    move |device_idx| {
                        tokio::runtime::Handle::current().block_on(real.reset_all_on_cpu(device_idx));
                    }
                });
            } else {
                let _ = self.mock.expect_reset_all_on_cpu();
            }

            self
        }

        pub fn with_age(mut self, out: u64) -> Self {


            if let Some(real) = self.real {

                self.mock.expect_do_age().returning({

                    move || {
                        real.do_age();
                    }
                });

                self.mock.expect_cache_age().returning({

                    move || {
                        real.cache_age()
                    }
                });

            } else {
                self.mock.expect_do_age();
                self.mock.expect_cache_age().return_const(out);
            }


            self
        }

        pub fn with_device_count(mut self, out: usize) -> Self {
            if let Some(real) = self.real {
                self.mock.expect_device_count().returning(move || {
                    real.device_count()
                });
            } else {
                self.mock.expect_device_count().return_const(out);
            }

            self
        }
        pub fn with_register_cache(mut self) -> Self {
            // Each mock runtime owns its caches too; real-runtime mocks must
            // share the actual registry so GPU purging sees the same maps.
            let registry = self.real.map(|real| real.cache_registry()).unwrap_or_else(|| std::sync::Arc::new(super::CacheRegistry::default()));
            self.mock.expect_cache_registry().returning(move || std::sync::Arc::clone(&registry));
            if let Some(real) = self.real {
                self.mock.expect_register_cache().returning({
                    move |cache| real.register_cache(cache)
                });
            } else {
                self.mock.expect_register_cache().returning(|cache| RegisteredCacheHandle::for_tests(cache.0));
            }

            self
        }

        pub fn with_remove_cache(mut self) -> Self {
            if let Some(real) = self.real {
                self.mock.expect_remove_cache().returning({
                    move |key| real.remove_cache(key)
                });
            } else {
                self.mock.expect_remove_cache();
            }

            self
        }

        pub fn with_get_runner(mut self, device_count: usize) -> Self {
            if let Some(real) = self.real {
                self.mock.expect_cuda_activity_gate().returning(move |device| real.cuda_activity_gate(device));
            } else {
                for device in 0..device_count {
                    let executor = if device == 0 {
                        SyncStageRunner::new_cpu(16)
                    } else {
                        SyncStageRunner::new_with_reset_hook(16, device, std::sync::Arc::new(|_| Box::pin(async { Ok(()) })))
                    };
                    self = self.with_runner(executor);
                }
            }
            self
        }

        pub fn with_runner(mut self, executor: SyncStageRunner) -> Self {
            // Expectations must preserve the same per-device ownership as the
            // application; a single CPU runner would hide recovery regressions.
            let gate = executor.cuda_gate().cloned();
            let device = gate.as_ref().map_or(0, |gate| gate.device_idx());
            self.mock.expect_cuda_activity_gate().withf(move |requested| *requested == device).returning(move |_| gate.clone());
            self.mock.expect_mock_runner().withf(move |requested| *requested == device).return_const(executor);
            self
        }

        pub fn with_dump_all_to_cpu(mut self) -> Self {
            if let Some(real) = self.real {
                let _ = self.mock.expect_dump_all_to_cpu().returning({
                    move |required_ids, device_idx, descr| {
                        block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(real.dump_all_to_cpu(required_ids, device_idx, descr))
                        })
                    }
                });
            } else {
                let _ = self.mock.expect_dump_all_to_cpu();
            }

            self
        }

        #[cfg(test)]
        pub fn with_run_purge_gpu_handler<F>(mut self, handler: F) -> Self
        where
            F: Fn(Vec<Box<dyn Fn() -> futures_util::future::BoxFuture<'static, Result<(), thoosa_shared::ThoError>> + Send + Sync>>) -> Result<(), thoosa_shared::ThoError> + Send + Sync + 'static,
        {
            // Tests observe the real owner's submissions without a background worker.
            self.mock.expect_run_purge_gpu().returning(handler);
            self
        }

        pub fn with_run_purge_gpu(mut self) -> Self {
            if let Some(real) = self.real {
                self.mock.expect_run_purge_gpu().times(0..).returning({
                    move |closures| {
                        real.run_purge_gpu(closures)
                    }
                });
            } else {
                self.mock.expect_run_purge_gpu().times(0..).returning(|_closures| Ok(()));
            }

            self
        }

        pub fn with_is_ipc(mut self, is_ipc:bool) -> Self {
            if let Some(real) = self.real {
                self.mock.expect_is_ipc().times(0..).returning({
                    move || {
                        real.is_ipc
                    }
                });
            } else {
                let _ = self.mock.expect_is_ipc().times(0..).returning(move || is_ipc);
            }

            self
        }

        pub fn with_is_all_on_cpu(mut self, is_all_on_cpu:bool) -> Self {
            if let Some(real) = self.real {
                self.mock.expect_is_all_on_cpu().times(0..).returning({
                    move |device_idx, force| {
                        real.is_all_on_cpu(device_idx, force)
                    }
                });
            } else {
                let _ = self.mock.expect_is_all_on_cpu().times(0..)
                    .returning(move |device_idx, force| is_all_on_cpu);
            }

            self
        }

        // #[cfg(tho_debug)]
        pub fn with_debug_caches(mut self) -> Self {
            if let Some(real) = self.real {
                self.mock.expect_debug_caches().times(0..).returning({
                    move |device_idx, descr| {
                        block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(
                                real.debug_caches(device_idx, descr)
                        )});

                    }
                });
            } else {
                let _ = self.mock.expect_debug_caches().times(0..)
                    .returning(move |_, _| {
                        ()
                    });
            }

            self
        }



        pub fn finish(self) -> MockGlobalCtx {
            self.mock
        }

        pub fn build(self) -> MockGlobalCtx {
            build_with_params_inner(self, MockGlobalCtxBuildParams::default(), true)
        }



        #[cfg(any(test, feature = "mock_global_ctx"))]
        pub fn install_global_ctx_mock(
            use_real_ctx: bool,
            is_ipc: bool,
        ) -> impl GlobalCtxMockGuard {
            use crate::runner::global_ctx::{
                MockGlobalCtx,
                mock_global_ctx,
            };

            debug!("install_global_ctx_mock: use_real_ctx={use_real_ctx} is_ipc={is_ipc}");
            let is_installed_ctx = MockGlobalCtx::is_installed_context();
            is_installed_ctx.expect().return_const(true);

            let shutdown_ctx = MockGlobalCtx::is_shutting_down_context();
            shutdown_ctx.expect().return_const(false);

            let global_ctx = MockGlobalCtx::global_context();

            let global: &'static MockGlobalCtx = if use_real_ctx {
                use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;

                let real: &'static RealGlobalCtx = RealGlobalCtx::real_global();

                debug!("install_global_ctx_mock: real={real:?}");

                mock_global_ctx::set_manual_method_real_ctx(Some(real));

                Box::leak(Box::new(from_real(real).build()))
            } else {
                mock_global_ctx::set_manual_method_real_ctx(None);

                Box::leak(Box::new(build_with_params(
                    MockGlobalCtxBuildParams::default(),
                    is_ipc,
                )))
            };

            global_ctx.expect().return_const(global);

            GlobalCtxMockContexts {
                _global_context: global_ctx,
                _shutdown_context: shutdown_ctx,
                _is_installed_ctx: is_installed_ctx
            }
        }
    }

    fn build_with_params_inner(builder: MockGlobalCtxBuilder, params: MockGlobalCtxBuildParams, is_ipc: bool) -> MockGlobalCtx {
        let mut builder = builder
            .with_age(params.age)
            .with_dump_all_to_cpu()
            .with_reset_all_on_cpu()
            .with_device_count(params.device_count)
            .with_register_cache()
            .with_remove_cache()
            .with_get_runner(params.device_count)
            .with_run_purge_gpu()
            .with_is_ipc(is_ipc)
            .with_is_all_on_cpu(true);

        // #[cfg(tho_debug)]
        builder = builder.with_debug_caches();

        builder.finish()
    }

    pub fn build_with_params(params: MockGlobalCtxBuildParams, is_ipc: bool) -> MockGlobalCtx {
        build_with_params_inner(build(), params, is_ipc)
    }

    pub fn build() -> MockGlobalCtxBuilder {
        MockGlobalCtxBuilder {
            mock: MockGlobalCtx::default(),
            real: None
        }
    }

    pub fn from_real(global_ctx: &'static GlobalCtx) -> MockGlobalCtxBuilder {

        MockGlobalCtxBuilder {
            mock: MockGlobalCtx::default(),
            real: Some(global_ctx)
        }
    }


    pub fn build_default() -> MockGlobalCtx {
        build_with_params(MockGlobalCtxBuildParams::default(), false)
    }

    pub static mut USE_REAL_CTX:bool = false;
    static MANUAL_METHOD_REAL_CTX: std::sync::OnceLock<parking_lot::RwLock<Option<&'static GlobalCtx>>> = std::sync::OnceLock::new();

    pub fn set_manual_method_real_ctx(real: Option<&'static GlobalCtx>) {
        *MANUAL_METHOD_REAL_CTX.get_or_init(|| parking_lot::RwLock::new(None)).write() = real;
    }

    pub fn manual_method_real_ctx() -> Option<&'static GlobalCtx> {
        *MANUAL_METHOD_REAL_CTX.get_or_init(|| parking_lot::RwLock::new(None)).read()
    }
}

#[cfg(any(test, feature = "mock_global_ctx"))]
impl MockGlobalCtx {
    pub fn get_runner(&self, device: usize) -> &SyncStageRunner {
        // Mockall owns values returned by reference. Delegate explicitly when
        // using real actors so stages and copies share the real device gate.
        if let Some(real) = mock_global_ctx::manual_method_real_ctx() { return real.get_runner(device); }
        self.mock_runner(device)
    }

    pub async fn copy_ctx<D:Into<ThoDevice>+ 'static>(&self, _device:D)-> tokio::sync::RwLockReadGuard<'static, CopyCtx> {
        static COPY_CTX_ONCE: std::sync::OnceLock<tokio::sync::RwLock<CopyCtx>> = std::sync::OnceLock::new();
        // This mock-only accessor cannot return initialization errors without changing the generated mock API.
        track_await!(COPY_CTX_ONCE.get_or_init(|| tokio::sync::RwLock::new(match CopyCtx::make_instance_for_cuda_device(0, 0.2, 16) {
            Ok(copy_ctx) => copy_ctx,
            Err(error) => panic!("failed to initialize mock CopyCtx: {error:?}"),
        })).read()).await
    }

    pub fn cuda_actor<D:Into<ThoDevice>+ 'static>(&self, device:D)-> &Box<dyn CudaActorHandle> {
        let device = device.into();
        if let Some(real) = mock_global_ctx::manual_method_real_ctx() {
            return real.cuda_actor(device);
        }

        static COPY_CTX_ONCE: std::sync::OnceLock<Box<dyn CudaActorHandle>> = std::sync::OnceLock::new();
        // This mock-only accessor must preserve the infallible generated mock API.
        COPY_CTX_ONCE.get_or_init(|| match CudaLocalActorHandle::new(Arc::new(CudaActivityGate::new(device.tho_device_idx()))) {
            Ok(actor) => Box::new(actor) as Box<dyn CudaActorHandle>,
            Err(error) => panic!("failed to initialize mock CUDA actor: {error:?}"),
        })
    }


}

#[cfg(test)]
mod tests {
    use crate::tch_error::tch_result;
    use crate as thoosa_common;
    use crate::dims::AnyDims;
    use crate::runner::common::cachable::{DevicesWrap, SelectRef};
    use crate::runner::global_ctx::CopyCtx;
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::runner::run_ctx::RunCtxImpl;
    use crate::runner::tmp_cache::TmpCached;
    use crate::tho_tensor::ThoTensor;
    use futures_util::{FutureExt, StreamExt, TryStreamExt};
    use log::debug;
    use parking_lot::{Condvar, Mutex, RwLock};
    use rand::{RngExt, rng};
    use std::collections::HashMap;
    use std::ffi::OsString;
    use std::future;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::Float;
    use tch::{Device, Tensor};
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use thoosa_shared::ThoError;
    use thoosa_shared::traits::RangeSelector;
    use thoosa_shared::traits::RunCtx;
    use thoosa_shims::context::ThoCudaContext;
    use tokio::task::spawn_blocking;
    use tracing::Instrument;
    use crate::ipc::cuda_wokrer::{CudaWorker, InitFn};
    use crate::ipc::tests::run_ipc_test;
    use crate::tho_device::ThoDevice;
    use crate::ws::mock_ws_simp;

    use thoosa_shared::comp_id::SynKey;

    // Exercise the real derive with varying output types and a generic argument
    // that does not change the output. Neither may share a cache accidentally.
    #[derive(Debug, cached_derive::Cached, thoosa_named_derive::Named)]
    struct GenericCacheStage<D: crate::dims::Dims + 'static, const TAG: u8>(std::marker::PhantomData<D>);

    impl<D: crate::dims::Dims + 'static, const TAG: u8> thoosa_shared::traits::Stage for GenericCacheStage<D, TAG> {
        type Ctx = RunCtxImpl;
        type In<'a> = Arc<DevicesWrap<ThoTensor<D, SynKey>>>;
        type Out = Arc<DevicesWrap<ThoTensor<D, SynKey>>>;
    }

    impl<D: crate::dims::Dims + 'static, const TAG: u8> thoosa_shared::traits::AsyncStage for GenericCacheStage<D, TAG> {}

    impl<D: crate::dims::Dims + 'static, const TAG: u8> thoosa_shared::traits::OutId for GenericCacheStage<D, TAG> {
        fn out_id<'a>(&self, input_id: &SynKey) -> SynKey { input_id.clone() }
    }

    #[tokio::test]
    async fn test_derived_caches_isolate_complete_stage_types() -> Result<(), ThoError> {
        use crate::runner::common::cachable::Cached;
        let mock = super::mock_global_ctx::build().with_register_cache().finish();
        let registry = mock.cache_registry();
        let global = super::MockGlobalCtx::global_context();
        global.expect().return_const(Box::leak(Box::new(mock)) as &'static super::MockGlobalCtx);
        let first = GenericCacheStage::<AnyDims, 0>::cache().await?;
        let other_output = GenericCacheStage::<crate::dims::ChannelsData, 0>::cache().await?;
        let other_tag = GenericCacheStage::<AnyDims, 1>::cache().await?;
        assert!(Arc::ptr_eq(&first, &GenericCacheStage::<AnyDims, 0>::cache().await?));
        assert!(!Arc::ptr_eq(&first, &other_tag));
        assert_eq!(registry.snapshot().len(), 3);
        first.insert("same-output".into(), futures_util::future::pending().boxed().shared().into());
        assert!(!other_output.contains_key(&"same-output".into()));
        assert!(!other_tag.contains_key(&"same-output".into()));
        Ok(())
    }

    #[test]
    fn test_cache_registry_concurrent_initialization_and_lifetime() -> Result<(), ThoError> {
        let registry = super::CacheRegistry::default();
        let barrier = std::sync::Barrier::new(16);
        let maps = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..16).map(|_| scope.spawn(|| {
                barrier.wait();
                registry.cache_for::<GenericCacheStage<AnyDims, 0>>()
            })).collect();
            handles.into_iter().map(|handle| handle.join().unwrap()).collect::<Result<Vec<_>, _>>()
        })?;
        assert!(maps.iter().all(|map| Arc::ptr_eq(map, &maps[0])));
        assert_eq!(registry.snapshot().len(), 1);
        let other_runtime = super::CacheRegistry::default();
        assert!(!Arc::ptr_eq(&maps[0], &other_runtime.cache_for::<GenericCacheStage<AnyDims, 0>>()?));
        let weak = Arc::downgrade(&maps[0]);
        drop(maps);
        assert!(weak.upgrade().is_some(), "the runtime owns the cache even without callers");
        drop(registry);
        assert!(weak.upgrade().is_none(), "dropping the registry must release both map and purger ownership");
        Ok(())
    }

    #[tokio::test]
    async fn test_cache_registry_purgers_use_the_registered_maps() -> Result<(), ThoError> {
        use crate::runner::common::cachable::CachedFuture;
        let runtime = GlobalCtx::new(1)?;
        let registry = runtime.cache_registry();
        let first = registry.cache_for::<GenericCacheStage<AnyDims, 0>>()?;
        let second = registry.cache_for::<GenericCacheStage<crate::dims::ChannelsData, 0>>()?;
        // Pending computations need no tensors or GPU to verify the actual
        // reset purgers invalidate both generic instantiations independently.
        let gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
        first.insert("same-output".into(), CachedFuture::for_device(futures_util::future::pending().boxed().shared(), Some(Arc::clone(&gate))));
        second.insert("same-output".into(), CachedFuture::for_device(futures_util::future::pending().boxed().shared(), Some(gate)));
        // The runtime snapshot must include every typed cache even though their
        // human-readable names are identical, alongside named instance caches.
        runtime.register_cache((<GenericCacheStage<AnyDims, 0> as thoosa_shared::traits::Named>::name().to_owned(), Arc::new(crate::runner::common::cachable::StageCache::<GenericCacheStage<AnyDims, 0>>::new())));
        let snapshot = runtime.caches_snapshot("typed cache test");
        assert_eq!(snapshot.len(), 3);
        for (_, entry) in snapshot {
            entry.purger.drop_gpu(1, &entry.name).await?;
        }
        assert!(first.is_empty());
        assert!(second.is_empty());
        Ok(())
    }

    #[test]
    fn test_cache_registry_rejects_incompatible_erased_type() -> Result<(), ThoError> {
        let registry = super::CacheRegistry::default();
        registry.cache_for::<GenericCacheStage<AnyDims, 0>>()?;
        // Simulate an internal invariant violation: the public API cannot insert
        // an arbitrary erased map, and the checked downcast must still reject it.
        Arc::get_mut(registry.entries.write().get_mut(&super::CacheKey::of::<GenericCacheStage<AnyDims, 0>>()).unwrap()).unwrap().map = Some(Arc::new(()));
        let error = registry.cache_for::<GenericCacheStage<AnyDims, 0>>().unwrap_err();
        assert!(error.to_string().contains("cache type mismatch"));
        Ok(())
    }

    #[tokio::test]
    async fn test_cache_registry_named_key_lifetime() -> Result<(), ThoError> {
        let runtime = GlobalCtx::new(1)?;
        let map = Arc::new(crate::runner::common::cachable::StageCache::<GenericCacheStage<AnyDims, 0>>::new());
        let handle = runtime.register_cache((format!("temporary_{}", 42), map));
        let super::CacheKey::Named(name) = handle.key() else { panic!("expected a named cache"); };
        let weak_name = Arc::downgrade(name);
        let snapshot = runtime.caches_snapshot("owned name lifetime");
        assert!(Arc::ptr_eq(&snapshot[0].1.gate, &handle.gate));
        runtime.remove_cache(handle.key());
        assert!(runtime.caches_snapshot("unregistered cache").is_empty());
        drop(handle);
        // An in-flight purge snapshot retains its name and gate after removal,
        // but releasing that last snapshot must free the dynamically owned name.
        assert!(weak_name.upgrade().is_some());
        drop(snapshot);
        assert!(weak_name.upgrade().is_none());
        Ok(())
    }

    async fn check_cache_key_exclusions() -> Result<(), ThoError> {
        use crate::runner::common::cachable::{Cached, StageCache};
        use std::collections::HashSet;
        use thoosa_shared::traits::{CachableIdTr, Named};
        type First = GenericCacheStage<AnyDims, 0>;
        type Second = GenericCacheStage<AnyDims, 1>;

        let runtime = GlobalCtx::global();
        let first_cache = First::cache().await?;
        let second_cache = Second::cache().await?;
        let named_cache = Arc::new(StageCache::<First>::new());
        let named_handle = runtime.register_cache((First::name().to_owned(), named_cache.clone()));
        assert_eq!(First::name(), Second::name());
        assert_ne!(First::cache_key(), Second::cache_key());
        assert_ne!(&First::cache_key(), named_handle.key());

        // All three caches deliberately have the same display name and output
        // ID. The real dump path must exclude only the requested cache's entry.
        let mut outputs = Vec::new();
        for (index, cache) in [first_cache, second_cache, named_cache].into_iter().enumerate() {
            let output = Arc::new(DevicesWrap::wrap(ThoTensor::<AnyDims, SynKey>::new("same-output".into(), tch_result(Tensor::f_from_slice(&[index as f32, 4.0]))?)?));
            let ready = future::ready(Ok(output.clone())).boxed().shared();
            ready.clone().await?;
            cache.insert("same-output".into(), ready.into());
            drop(output.load_from_any_device_(1).await?);
            outputs.push(output);
        }
        let id = SynKey::from("same-output");
        let keep: HashSet<&dyn CachableIdTr> = HashSet::from([&id as &dyn CachableIdTr]);
        runtime.dump_all_to_cpu(&HashMap::from([(First::cache_key(), &keep)]), 1, "typed cache exclusion").await?;
        assert!(outputs[0].is_on_device(1));
        assert!(!outputs[1].is_on_device(1));
        assert!(!outputs[2].is_on_device(1));

        // Production loaders chain stages and optionally select a branch. Run
        // those adapters against the populated caches and exclude their actual
        // output owner, rather than the adapter type which has no cache.
        use crate::stages::stages_seq::{PairStage, SingleOrPair};
        let first = Arc::new(GenericCacheStage::<AnyDims, 0>(std::marker::PhantomData));
        let second = Arc::new(GenericCacheStage::<AnyDims, 1>(std::marker::PhantomData));
        let pair = Arc::new(PairStage::<First, Second, RunCtxImpl>::new(first.clone(), second));
        let single = Arc::new(SingleOrPair::<First, PairStage<First, Second, RunCtxImpl>>::Single(first));
        let chained = Arc::new(SingleOrPair::<First, PairStage<First, Second, RunCtxImpl>>::Pair(pair));
        // Keep the cache area alive so its independent cleanup cannot race the
        // explicit OOM dumps being checked below.
        let mut ctx = RunCtxImpl::new(1, "adapter cache exclusion");
        for (loader, expected) in [(single, 0), (chained, 1)] {
            let output = Arc::clone(&loader).run_cached(outputs[0].clone(), ctx.branch(false)).await?;
            assert!(Arc::ptr_eq(&output, &outputs[expected]));
            assert_eq!(loader.output_cache_key(), if expected == 0 { First::cache_key() } else { Second::cache_key() });
            for output in &outputs { drop(output.load_from_any_device_(1).await?); }
            // Downstream tensor stages only see this wrapper, not its producer.
            let mut required_inputs = HashMap::new();
            output.add_required_ids(&mut required_inputs);
            assert_eq!(required_inputs.len(), 1);
            assert_eq!(required_inputs[&loader.output_cache_key()], keep);
            let required_ids = required_inputs.iter().map(|(key, ids)| (key.clone(), ids)).collect();
            runtime.dump_all_to_cpu(&required_ids, 1, "adapter cache exclusion").await?;
            for (index, output) in outputs.iter().enumerate() { assert_eq!(output.is_on_device(1), index == expected); }
        }

        drop(outputs[1].load_from_any_device_(1).await?);
        drop(outputs[2].load_from_any_device_(1).await?);
        runtime.dump_all_to_cpu(&HashMap::from([(named_handle.key().clone(), &keep)]), 1, "named cache exclusion").await?;
        assert!(!outputs[0].is_on_device(1));
        assert!(!outputs[1].is_on_device(1));
        assert!(outputs[2].is_on_device(1));
        runtime.dump_all_to_cpu(&HashMap::new(), 1, "no cache exclusions").await?;
        for (index, output) in outputs.iter().enumerate() {
            assert!(!output.is_on_device(1));
            let cpu = output.load_from_any_device_(0).await?;
            assert_eq!(tch_result(cpu.t_ref().f_double_value(&[0]))?, index as f64);
        }
        runtime.remove_cache(named_handle.key());
        // A pass-through stage can cache the same wrapper under another owner.
        // Both entries must be excluded or either purger could offload it.
        let ready = future::ready(Ok(outputs[0].clone())).boxed().shared();
        ready.clone().await?;
        Second::cache().await?.insert(id.clone(), ready.into());
        let alias = Arc::new(GenericCacheStage::<AnyDims, 1>(std::marker::PhantomData)).run_cached(outputs[0].clone(), ctx.branch(false)).await?;
        let mut required_inputs = HashMap::new();
        alias.add_required_ids(&mut required_inputs);
        assert_eq!(required_inputs.len(), 2);
        drop(alias.load_from_any_device_(1).await?);
        let required_ids = required_inputs.iter().map(|(key, ids)| (key.clone(), ids)).collect();
        runtime.dump_all_to_cpu(&required_ids, 1, "aliased cache exclusions").await?;
        assert!(alias.is_on_device(1));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_cache_key_exclusions_local() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
        check_cache_key_exclusions().await
    }

    #[test]
    fn test_cache_key_exclusions_ipc() -> Result<(), ThoError> {
        run_ipc_test(module_path!(), check_cache_key_exclusions(), async |socket| {
            let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
            CudaWorker::run_worker(socket, &ws, HashMap::new(), ThoDevice::from(Cuda(0))).await
        })
    }

    #[tokio::test]
    async fn test_mock_execution_contexts_are_per_device() {
        use super::mock_global_ctx::{build_with_params, MockGlobalCtxBuildParams};
        let mock = build_with_params(MockGlobalCtxBuildParams { device_count: 3, ..Default::default() }, false);
        assert!(mock.cuda_activity_gate(0).is_none());
        assert!(mock.get_runner(0).cuda_gate().is_none());
        let first = mock.cuda_activity_gate(1).unwrap();
        let second = mock.cuda_activity_gate(2).unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
        for device in 1..3 {
            assert!(Arc::ptr_eq(&mock.cuda_activity_gate(device).unwrap(), mock.get_runner(device).cuda_gate().unwrap()));
        }
        let ticket = mock.get_runner(1).request_cuda_reset(ThoError::ContextReset).unwrap().unwrap();
        tokio::time::timeout(Duration::from_secs(2), first.wait_for_pause_to_finish(ticket)).await.unwrap().unwrap();
        assert_eq!(first.device_generation().0, 1);
        assert_eq!(second.device_generation().0, 0);
        assert_eq!(mock.get_runner(2).submit(0, async { Ok(2) }, None).await.unwrap(), 2);
        assert_eq!(mock.get_runner(0).submit(0, async { Ok(0) }, None).await.unwrap(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_device_reset_local() -> Result<(), ThoError>{
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }


        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);


        do_test_device_reset().await?;

        Ok(())
    }

    async fn do_test_device_reset() -> Result<(), ThoError> {
        let cuda_actor = GlobalCtx::global().cuda_actor(1);
        let gate = GlobalCtx::global().cuda_activity_gate(1).unwrap();
        let generation = gate.device_generation();
        // Real-context mocks must dispatch through the real device runner too.
        assert!(std::ptr::eq(super::MockGlobalCtx::global().get_runner(1), GlobalCtx::global().get_runner(1)));

        let t_cpu = Tensor::rand(&[10000], FLOAT_CPU);
        let t_cuda = cuda_actor.copy(t_cpu.into(), 1.into()).await?;
        debug!("test_device_reset: t_cuda={t_cuda:?}");

        let tmp_cache: TmpCached<ThoTensor<AnyDims, u32>> = TmpCached::new("tmp").await;

        tmp_cache.insert(t_cuda.to_types(777u32)?);

        let t_0 = tmp_cache.get(&777u32).unwrap();
        let mut root_ctx = RunCtxImpl::new(1, "root_ctx");

        let sum_0 = {
            let t_0 = t_0.select_ref(root_ctx.tho_device_idx(), None::<RangeSelector>).await?;
            let t_0_cpu = cuda_actor.copy(t_0.shallow_clone::<AnyDims>().strip_types(), 0.into()).await?;
            Ok::<_, ThoError>(t_0_cpu.as_ref().sum_dim_intlist(0, false, None).double_value(&[]))
        }?;
        debug!("test_device_reset: sum_0={sum_0}");

        GlobalCtx::global().dump_all_to_cpu(&HashMap::new(), 1, "test").await?;


        cuda_actor.restart().await?;
        assert_eq!(gate.device_generation().0, generation.0 + 1);
        assert!(GlobalCtx::process_failure().is_none(), "healthy maintenance must not signal terminal local failure");
        debug!("test_device_reset: reset done");

        let t_0 = tmp_cache.get(&777u32).unwrap();
        let sum_1 = {
            let t_0 = t_0.select_ref(root_ctx.tho_device_idx(), None::<RangeSelector>).await?;
            let t_0_cpu = cuda_actor.copy(t_0.shallow_clone::<AnyDims>().strip_types(), 0.into()).await?;
            Ok::<_, ThoError>(t_0_cpu.as_ref().sum_dim_intlist(0, false, None).double_value(&[]))
        }?;

        debug!("test_device_reset: sum_1={sum_1}");

        assert_eq!(sum_0, sum_1);
        Ok(())
    }

    #[test]
    fn test_device_reset_ipc() -> Result<(), ThoError> {
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));

        run_ipc_test(
            module_path!(),
            async {

                do_test_device_reset().await?;

                Ok(())
            },
            async |sock_path: OsString| {
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    HashMap::new(),
                    ThoDevice::from(Device::Cuda(0)),
                ).await
            },
        )
    }

}
