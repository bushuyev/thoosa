use crate::runner::purge::FuturesRunnerMapPurger;
use log::{debug, error, info};
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
use thoosa_shared::{CudaActivityGate, ThoError, async_perf_log, async_read_owned_lock, async_write_owned_lock, perf_log, sync_read_lock, sync_write_lock};
use thoosa_shims::context::ThoCudaContext;
use tokio::sync::RwLock as AsyncRwLock;
use tracing::Instrument;
use thoosa_shims::arena::{shared_arena_dup_fd, shared_arena_init};

pub static GLB_SHUTTING_DOWN:AtomicBool = AtomicBool::new(false);

///supposed to be internal, don't use it in Stages except when instantiating to check if it should be ipc proxy
#[derive(Debug)]
pub struct GlobalCtx {

    devices: Vec<DeviceExecutionCtx>,

    num_devices: usize,//TODO replace with Vec<ThoDevice>

    caches: parking_lot::RwLock<HashMap<&'static str, Arc<CacheRegistryEntry>>>,
    is_all_on_cpu: Vec<AtomicBool>,
    cache_time: AtomicU64, //for tracking CPU-only lifetime
    purger: Purger,
    shutting_down:AtomicBool,//TODO not needed if tokio's shutdown_background in project main,

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

#[derive(Debug)]
struct CacheRegistryEntry {
    purger: Arc<dyn FuturesRunnerMapPurger>,
    gate: Arc<AsyncRwLock<()>>,
}

type CacheSnapshot = Vec<(&'static str, Arc<CacheRegistryEntry>)>;

#[derive(Debug, Clone)]
pub struct RegisteredCacheHandle {
    name: &'static str,
    gate: Arc<AsyncRwLock<()>>,
}

impl RegisteredCacheHandle {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub async fn lock_exclusive(&self) -> tokio::sync::OwnedRwLockWriteGuard<()> {
        async_perf_log!(
            async_write_owned_lock!(Arc::clone(&self.gate))
        )
    }

    #[cfg(any(test, feature = "mock_global_ctx"))]
    pub fn for_tests(name: &'static str) -> Self {
        Self {
            name,
            gate: Arc::new(AsyncRwLock::new(())),
        }
    }
}

#[cfg_attr(any(test, feature = "mock_global_ctx"), automock)]
impl GlobalCtx {

    //TODO num_devices is to be replaces with Cuda devices list. Currently it's assumed that there is single Cuda(0) device and first device is always CPU. So if num_devices==2 then devices would be [Cpu, Cuda[0]]
    fn new(num_devices: usize) -> anyhow::Result<Self> {
        debug!("GlobalCtx::new: num_devices={num_devices}");

        assert!(num_devices > 0);//TODO test if it works without GPU

        let mut devices = Vec::with_capacity(num_devices);
        devices.push(DeviceExecutionCtx::cpu());
        for tho_device_idx in 1..num_devices {
            let gate = Arc::new(CudaActivityGate::new(tho_device_idx));
            let recovery = CudaRecoveryCoordinator::new(Arc::clone(&gate), Arc::new(GlobalCtx::reset_cuda_context));
            let actor = CudaLocalActorHandle::new(Arc::clone(&gate))?;
            devices.push(DeviceExecutionCtx {
                runner: SyncStageRunner::new_with_recovery(1024, recovery),
                cuda_actor: Some(Box::new(actor)),
            });
        }

        Ok(Self {
            num_devices,
            devices,

            caches: parking_lot::RwLock::new(HashMap::new()),

            is_all_on_cpu: (0..num_devices).map(|_| AtomicBool::new(false)).collect(),


            cache_time: Default::default(),
            purger: Purger::new(),
            shutting_down: false.into(),

            is_ipc: false,
            arena_fd: None,
        })
    }

    async fn new_ipc(num_devices: usize, cuda_worker_args: Vec<String>) -> anyhow::Result<Self> {
        debug!("GlobalCtx::new: num_devices={num_devices}");

        assert!(num_devices > 0);//TODO test if it works without GPU

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
            let recovery = CudaRecoveryCoordinator::new(Arc::clone(&gate), Arc::new(GlobalCtx::reset_cuda_context));
            // Each actor retains its own descriptor so replacement workers can
            // remap the parent-owned arena independently of other devices.
            let actor = CudaIPCActorHandle::new(Arc::clone(&gate), cuda_worker_args.clone(), arena_fd.try_clone()?).await?;
            devices.push(DeviceExecutionCtx {
                runner: SyncStageRunner::new_with_recovery(1024, recovery),
                cuda_actor: Some(Box::new(actor)),
            });
        }

        Ok(Self {
            num_devices,
            devices,

            caches: parking_lot::RwLock::new(HashMap::new()),

            is_all_on_cpu: (0..num_devices).map(|_| AtomicBool::new(false)).collect(),


            cache_time: Default::default(),
            purger: Purger::new(),
            shutting_down: false.into(),

            is_ipc: true,
            arena_fd: Some(arena_fd),
        })
    }

    fn reset_cuda_context(tho_device_idx: usize) -> BoxFuture<'static, Result<(), ThoError>> {
        async move {
            // This device's coordinator invokes the hook after its activity drains.
            info!("GlobalCtx::reset_cuda_context: tho_device_idx={tho_device_idx}");

            track_await(GlobalCtx::global()
                .drop_data_on_device(tho_device_idx, "drop"))
                .await?;
            info!("GlobalCtx::reset_cuda_context: drop_device done");

            GlobalCtx::global()
                .cuda_actor(tho_device_idx)
                .restart_backend()
                .await?;


            info!("GlobalCtx::reset_cuda_context: all done");

            Ok(())
        }.boxed()
    }

    pub fn get_executor(&self, tho_device:usize) -> &SyncStageRunner {
        debug!("GlobalCtx::get_executor: tho_device={tho_device} self.devices.len={}", self.devices.len());
        &self.devices[tho_device].runner//tho_device is 0 in tests
    }

    #[cfg(test)]
    pub async fn restart_cuda_device(&self, device: usize) -> Result<(), ThoError> {
        let ticket = self.get_executor(device).request_cuda_reset(ThoError::ContextReset)?;
        // Admitted callers must release their activity before the independent
        // coordinator can drain. Waiting here would wait on the caller itself.
        if thoosa_shared::CudaActivityScope::current().is_some() {
            return Err(ThoError::ContextReset);
        }
        if let Some(ticket) = ticket {
            self.get_executor(device).cuda_gate().expect("CUDA device").wait_for_pause_to_finish(ticket).await?;
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


    pub fn register_cache(&self, cache: (&'static str, Arc<dyn FuturesRunnerMapPurger>)) -> RegisteredCacheHandle {//TODO rename DataPurger?
        debug!("GlobalCtx::register_cache: cache={}", cache.0);

        let gate = Arc::new(AsyncRwLock::new(()));
        let entry = Arc::new(CacheRegistryEntry {
            purger: cache.1,
            gate: Arc::clone(&gate),
        });

        assert!(
            perf_log!(sync_write_lock!(self.caches))
            .insert(cache.0, entry)
            .is_none()
        );

        debug!("GlobalCtx::register_cache: cache={} lock acquired/released, done", cache.0);

        RegisteredCacheHandle { name: cache.0, gate }
    }

    fn caches_snapshot(&self, descr: &'static str) -> CacheSnapshot {
        perf_log!(
            sync_read_lock!(self.caches, "{descr}")
                .iter()
                .map(|(type_name, entry)| (*type_name, Arc::clone(entry)))
                .collect(),
            "{descr}"
        )
    }

    async fn lock_cache_shared(entry: &Arc<CacheRegistryEntry>, ctx: &'static str) -> tokio::sync::OwnedRwLockReadGuard<()> {
        async_perf_log!(async_read_owned_lock!(Arc::clone(&entry.gate), "{ctx}"), "{ctx}")
    }

    pub fn remove_cache(&self, key:&'static str) {
        debug!("GlobalCtx::remove_cache: key={}", key);

        assert!(
            perf_log!(sync_write_lock!(self.caches, "key={key:?}"))
            .remove(key)
            .is_some()
        );

        debug!("GlobalCtx::remove_cache: {} lock acquired/released, removed ", key);
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
        self.num_devices
    }

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

        for (type_name, entry) in maps.iter() {
            let _cache_guard = track_await(Self::lock_cache_shared(entry, "GlobalCtx::is_all_on_cpu_force_check: timed out waiting for cache gate read lock")).await;
            if track_await(entry.purger.has_gpu_data(device_idx)).await {
                debug!("GlobalCtx::is_all_on_cpu_force_check: type_name={type_name} has data on device_idx={device_idx}");
                return false;
            }
        }

        true
    }

    pub async fn reset_all_on_cpu(&self, device_idx: usize) {
        self.is_all_on_cpu[device_idx].store(false, Relaxed);
    }


    // #[cfg(tho_debug)]
    async fn debug_caches_snapshot(&self, device_idx: usize, descr: &str, maps: &CacheSnapshot) {
        debug!("GlobalCtx::debug_caches: descr={descr:?}");

        let mut total_gpu_mem = 0;
        for (type_name, entry) in maps.iter() {
            let _cache_guard = track_await(Self::lock_cache_shared(entry, "GlobalCtx::debug_caches: timed out waiting for cache gate read lock")).await;
            info!("GlobalCtx::debug_gpu: descr={descr} type_name={type_name}");
            match track_await(entry.purger.debug_gpu(device_idx)).await {
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

    // #[cfg(tho_debug)]
    pub async fn debug_caches(&self, device_idx: usize, descr: &str) {
        let maps = self.caches_snapshot("GlobalCtx::debug_caches: timed out waiting for caches read lock");
        track_await(self.debug_caches_snapshot(device_idx, descr, &maps)).await;
    }

    ///ids_to_keep_by_type - should be few keys long - only inputs involved in the current operation
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="GlobalCtx::dump_all_to_cpu"))]
    pub async fn dump_all_to_cpu<'a>(
        &'a self,
        required_ids: &HashMap<&'static str, &'a HashSet<&'a dyn CachableIdTr>>, //TODO key is used for tmp_cache
        tho_device_idx: usize,
        descr: &str,
    ) -> Result<(), ThoError> {
        assert_ne!(tho_device_idx, 0);
        debug!("GlobalCtx::dump_all_to_cpu: tho_device_idx={tho_device_idx} required_ids={required_ids:?} descr={descr}");
        let maps = self.caches_snapshot("GlobalCtx::dump_all_to_cpu: timed out waiting for caches read lock");
        let dump_eta_guard = CollectEtaInvocationGuard::new(&format!("dump_all_to_cpu_{descr}"), maps.len(), 1);

        for (type_name, entry) in maps.iter() {
            let _cache_guard = track_await(Self::lock_cache_shared(entry, "GlobalCtx::dump_all_to_cpu: timed out waiting for cache gate read lock")).await;
            if !track_await(entry.purger.has_gpu_data(tho_device_idx)).await {
                debug!("GlobalCtx::dump_all_to_cpu: type_name={type_name} has no data on device_idx={tho_device_idx}, skipping");
                collect_eta_record_completion(dump_eta_guard.id());
                continue;
            }

            debug!("GlobalCtx::dump_all_to_cpu: type_name={type_name}");
            track_await(entry.purger.purge_gpu(required_ids.get(type_name).map(|v| *v), tho_device_idx, type_name)).await?;
            debug!("GlobalCtx::dump_all_to_cpu: type_name={type_name} descr={descr} done");
            collect_eta_record_completion(dump_eta_guard.id());
        }

        self.is_all_on_cpu[tho_device_idx].store(true, Relaxed);
        debug!("GlobalCtx::dump_all_to_cpu: tho_device_idx={tho_device_idx} required_ids={required_ids:?} descr={descr} done");

        #[cfg(tho_debug)]
        track_await(self.debug_caches(tho_device_idx, descr)).await;

        #[cfg(tho_debug)]
        self.cuda_actor(tho_device_idx).debug_worker().await?;

        Ok(())
    }

    ///ids_to_keep_by_type - should be few keys long - only inputs involved in the current operation
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

        for (type_name, entry) in maps.iter() {
            let _cache_guard = track_await(Self::lock_cache_shared(entry, "GlobalCtx::drop_data_on_device: timed out waiting for cache gate read lock")).await;
            debug!("GlobalCtx::drop_data_on_device: type_name={type_name}");
            track_await(entry.purger.drop_gpu(tho_device_idx, type_name)).await?;
            debug!("GlobalCtx::drop_data_on_device: type_name={type_name} descr={descr} done");
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
            track_await(self.debug_caches_snapshot(1, "before purge_cpu", &maps)).await;

            for (type_name, entry) in maps.iter() {
                let _cache_guard = track_await(Self::lock_cache_shared(entry, "GlobalCtx::purge_cpu: timed out waiting for cache gate read lock")).await;
                let purged = track_await(entry.purger.purge_cpu(purge_until)).await;
                debug!("GlobalCtx::purge_cpu: purged {purged} from {type_name}");
            }

            #[cfg(tho_debug)]
            track_await(self.debug_caches_snapshot(1, "after purge_cpu", &maps)).await;
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

    #[inline]
    pub fn is_shutting_down()->bool {
        GLB_SHUTTING_DOWN.load(Relaxed)
    }

    //for tests and temporary fixes - use CudaActors for ipc/no ipc implemenations
    pub fn is_ipc(&self) -> bool {
        self.is_ipc
    }
}

impl GlobalCtx {

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

        pub fn with_get_executor(mut self) -> Self {
            self.mock.expect_get_executor().times(0..).return_const(SyncStageRunner::new(16));
            self
        }

        pub fn with_executor(mut self, executor: SyncStageRunner) -> Self {
            // Recovery tests need a device-bound runner instead of the default
            // CPU runner, while retaining the normal GlobalCtx injection path.
            self.mock.expect_get_executor().times(0..).return_const(executor);
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
            .with_get_executor()
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
    pub async fn copy_ctx<D:Into<ThoDevice>+ 'static>(&self, _device:D)-> tokio::sync::RwLockReadGuard<'static, CopyCtx> {
        static COPY_CTX_ONCE: std::sync::OnceLock<tokio::sync::RwLock<CopyCtx>> = std::sync::OnceLock::new();
        // This mock-only accessor cannot return initialization errors without changing the generated mock API.
        track_await(COPY_CTX_ONCE.get_or_init(|| tokio::sync::RwLock::new(match CopyCtx::make_instance_for_cuda_device(0, 0.2, 16) {
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
    use crate as thoosa_common;
    use crate::dims::AnyDims;
    use crate::runner::common::cachable::{DevicesWrap, SelectRef};
    use crate::runner::global_ctx::CopyCtx;
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::runner::run_ctx::RunCtxImpl;
    use crate::runner::tmp_cache::TmpCached;
    use crate::tho_tensor::ThoTensor;
    use futures_util::{StreamExt, TryStreamExt};
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
