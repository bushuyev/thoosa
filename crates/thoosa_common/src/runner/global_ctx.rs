use crate::runner::purge::FuturesRunnerMapPurger;
use log::{debug, info};
#[allow(unused_imports)]
use mockall::automock;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::future::ready;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use futures_util::future::BoxFuture;
use thoosa_shared::traits::CachableIdTr;
use thoosa_shared::{async_perf_log, async_read_owned_lock, async_write_owned_lock, perf_log, sync_read_lock, sync_write_lock, ThoError};
use tokio::sync::RwLock as AsyncRwLock;
use crate::runner::cache_areas::{CacheAreaId, CacheAreasKeeper};
use crate::runner::purger::Purger;
use thoosa_shared::collect_futures_eta::{collect_eta_record_completion, CollectEtaInvocationGuard};
use crate::copy_ctx::{CopyCtx, CudaActorHandle, CudaLocalActorHandle};
use crate::runner::dyn_mpsc::DynamicTaskExecutor;
use crate::tho_device::ThoDevice;
use futures::FutureExt;
use thoosa_shims::context::ThoCudaContext;
use tracing::Instrument;
use thoosa_shared::tracked_await::track_await;
use crate::ipc::cuda_process::CudaIPCActorHandle;

#[derive(Debug)]
pub struct GlobalCtx {

    executors: Vec<DynamicTaskExecutor>,

    num_devices: usize,//TODO replace with Vec<ThoDevice>

    caches: parking_lot::RwLock<HashMap<&'static str, Arc<CacheRegistryEntry>>>,
    is_all_on_cpu: Vec<AtomicBool>,
    cache_time: AtomicU64, //for tracking CPU-only lifetime
    purger: Purger,
    shutting_down:AtomicBool,//TODO not needed if tokio's shutdown_background in project main,

    cuda_actors:Vec<Box<dyn CudaActorHandle>>,

    reset_mutex:Vec<parking_lot::Mutex<bool>>,
    is_ipc: bool,
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

    fn new(num_devices: usize) -> Self {
        debug!("GlobalCtx::new: num_devices={num_devices}");

        assert!(num_devices > 0);//TODO test if it works without GPU

        Self {
            num_devices,
            executors: (0..num_devices).map(|tho_device_idx| {
                DynamicTaskExecutor::new_with_reset_hook(1024, tho_device_idx, Arc::new(GlobalCtx::reset_cuda_context))
            })
            .collect(),

            caches: parking_lot::RwLock::new(HashMap::new()),

            is_all_on_cpu: (0..num_devices).map(|_| AtomicBool::new(false)).collect(),


            cache_time: Default::default(),
            purger: Purger::new(),
            shutting_down: false.into(),
            reset_mutex: (1..num_devices).map(|device_idx|parking_lot::Mutex::new(false)).collect(),

            cuda_actors: (1..num_devices).map(|tho_device_idx|Box::new(CudaLocalActorHandle::new(tho_device_idx)) as Box<dyn CudaActorHandle>).collect(),
            is_ipc: false,
        }
    }

    //TODO this should replace GlobalCtx::new when ipc refactoring is completed
    async fn new_ipc(num_devices: usize, cuda_worker_args: Vec<&'static str>) -> Result<Self, ThoError> {
        debug!("GlobalCtx::new: num_devices={num_devices}");

        assert!(num_devices > 0);//TODO test if it works without GPU

        let mut cuda_actors = Vec::with_capacity(num_devices.saturating_sub(1));
        for tho_device_idx in 1..num_devices {
            cuda_actors.push(Box::new(CudaIPCActorHandle::new(tho_device_idx, cuda_worker_args.clone()).await?) as Box<dyn CudaActorHandle>);
        }

        Ok(Self {
            num_devices,
            executors: (0..num_devices).map(|tho_device_idx| {
                DynamicTaskExecutor::new_with_reset_hook(1024, tho_device_idx, Arc::new(GlobalCtx::reset_cuda_context))
            })
                .collect(),

            caches: parking_lot::RwLock::new(HashMap::new()),

            is_all_on_cpu: (0..num_devices).map(|_| AtomicBool::new(false)).collect(),


            cache_time: Default::default(),
            purger: Purger::new(),
            shutting_down: false.into(),
            reset_mutex: (1..num_devices).map(|device_idx|parking_lot::Mutex::new(false)).collect(),

            cuda_actors,
            is_ipc: true,
        })
    }


    fn reset_cuda_context(tho_device_idx: usize) -> BoxFuture<'static, Result<(), ThoError>> {
        let mut guard  = GlobalCtx::global().reset_mutex[tho_device_idx-1].lock();
        if *guard {
            info!("GlobalCtx::reset_cuda_context: has been reset");

            return ready(Ok(())).boxed()
        }
        *guard = true;
        async move {
            info!("GlobalCtx::reset_cuda_context: tho_device_idx={tho_device_idx}");

            track_await(GlobalCtx::global()
                .drop_device(tho_device_idx, "drop"))
                .await?;
            info!("GlobalCtx::reset_cuda_context: drop_device done");

            GlobalCtx::global()
                .cuda_actor(tho_device_idx)
                .reset()
                .await?;


            info!("GlobalCtx::reset_cuda_context: all done");
            Ok(())
        }.boxed()
    }

    pub fn get_executor(&self, tho_device:usize) -> &DynamicTaskExecutor {
        debug!("GlobalCtx::get_executor: tho_device={tho_device} self.executors.len={}", self.executors.len());
        &self.executors[tho_device]//tho_device is 0 in tests
    }

    pub fn run_purge_gpu(&self, closures: Vec<Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>> + Send + Sync>>) ->Result<(), ThoError>{
        self.purger.run_purge_gpu(closures)
    }

    ///num_devices: number of devices - CPU + GPU count
    pub fn init_devices(num_devices: usize) -> &'static GlobalCtx {
        debug!("GlobalCtx::init_devices: num_devices={num_devices}");

        GLOBAL_CTX
            .set(GlobalCtx::new(num_devices))
            .expect("`init_devices` must only be called once at program start");

        Self::global()
    }

    ///num_devices: number of devices - CPU + GPU count
    pub async fn init_devices_ipc(num_devices: usize/*TODO replace with DeviceInitStruct vec*/, cuda_worker_args: Vec<&'static str>/*TODO move to device init struct*/) -> Result<&'static GlobalCtx, ThoError> {
        debug!("GlobalCtx::init_devices: num_devices={num_devices}");

        GLOBAL_CTX
            .set(GlobalCtx::new_ipc(num_devices, cuda_worker_args).await?)
            .expect("`init_devices` must only be called once at program start");

        Ok(Self::global())
    }


    pub fn register_cache(&self, cache: (&'static str, Arc<dyn FuturesRunnerMapPurger>)) -> RegisteredCacheHandle {
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
            perf_log!(sync_write_lock!(self.caches, "key={key}"))
            .remove(key)
            .is_some()
        );

        debug!("GlobalCtx::remove_cache: {} lock acquired/released, removed ", key);
    }

    pub fn global() -> &'static Self {
        GLOBAL_CTX.get().expect("call `init_devices` first")
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


    #[cfg(tho_debug)]
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

        info!(
        "GlobalCtx::debug_gpu: descr={descr} device_idx={device_idx} total_gpu_mem={total_gpu_mem} device_stats={:?}",
        thoosa_shims::device_stats::get_device_stats((device_idx - 1) as i32)
    );
    }

    #[cfg(tho_debug)]
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

        Ok(())
    }

    ///ids_to_keep_by_type - should be few keys long - only inputs involved in the current operation
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="GlobalCtx::dump_all_to_cpu"))]
    pub async fn drop_device<'a>(
        &'a self,
        tho_device_idx: usize,
        descr: &str,
    ) -> Result<(), ThoError> {
        assert_ne!(tho_device_idx, 0);
        info!("GlobalCtx::drop_device: tho_device_idx={tho_device_idx} descr={descr}");
        let maps = self.caches_snapshot("GlobalCtx::drop_device: timed out waiting for caches read lock");
        // let caches_lock = perf_log!(sync_read_lock!(self.caches));
        let dump_eta_guard = CollectEtaInvocationGuard::new(&format!("drop_device{descr}"), maps.len(), 1);

        for (type_name, entry) in maps.iter() {
            let _cache_guard = track_await(Self::lock_cache_shared(entry, "GlobalCtx::drop_device: timed out waiting for cache gate read lock")).await;
            debug!("GlobalCtx::drop_device: type_name={type_name}");
            track_await(entry.purger.drop_gpu(tho_device_idx, type_name)).await?;
            debug!("GlobalCtx::drop_device: type_name={type_name} descr={descr} done");
            collect_eta_record_completion(dump_eta_guard.id());
        }

        self.is_all_on_cpu[tho_device_idx].store(true, Relaxed);
        info!("GlobalCtx::drop_device: tho_device_idx={tho_device_idx} descr={descr} done");

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
        self.shutting_down.store(true, Relaxed);
        //TODO


    }

    pub fn is_shutting_down(&self)->bool {
        self.shutting_down.load(Relaxed)
    }

    //for tests and temporary fixes - use CudaActors for ipc/no ipc implemenations
    pub fn is_ipc(&self) -> bool {
        self.is_ipc
    }
}

impl GlobalCtx {

    pub fn cuda_actor<D:Into<ThoDevice>+ 'static>(&self, device:D)-> &Box<dyn CudaActorHandle> {
        let device:ThoDevice = device.into();
        let gpu_device_ordinal = device.gpu_ordinal();
        self.cuda_actors.get(gpu_device_ordinal).expect(&format!("No CudaActor for device={gpu_device_ordinal}"))
    }
}

#[cfg(any(test, feature = "mock_global_ctx"))]
pub mod mock_global_ctx {
    use crate::runner::global_ctx::{GlobalCtx, RegisteredCacheHandle};
    use log::debug;
    use tokio::task::block_in_place;
    use crate::runner::global_ctx::MockGlobalCtx;

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


        pub fn finish(self) -> MockGlobalCtx {
            self.mock
        }

        pub fn build(self) -> MockGlobalCtx {
            build_with_params_inner(self, MockGlobalCtxBuildParams::default(), true)
        }

        #[cfg(any(test, feature = "mock_global_ctx"))]
        pub fn install_global_ctx_mock(use_real_ctx: bool, is_ipc: bool) -> impl Drop {
            debug!("install_global_ctx_mock");
            use crate::runner::global_ctx::{mock_global_ctx, MockGlobalCtx};
            let ctx = MockGlobalCtx::global_context();
            if use_real_ctx {
                use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;

                let real: &'static RealGlobalCtx = RealGlobalCtx::real_global();
                mock_global_ctx::set_manual_method_real_ctx(Some(real));


                ctx.expect().returning(move || {
                    // NOTE: this leaks once per call to GlobalCtx::global() in the test.
                    Box::leak(Box::new(from_real(real).build()))
                });

                return ctx;
            }

            mock_global_ctx::set_manual_method_real_ctx(None);
            ctx.expect().returning(move || {
                Box::leak(Box::new(
                    build_with_params(MockGlobalCtxBuildParams::default(), is_ipc),
                )) as &'static MockGlobalCtx
            });

            ctx
        }

    }

    fn build_with_params_inner(builder: MockGlobalCtxBuilder, params: MockGlobalCtxBuildParams, is_ipc: bool) -> MockGlobalCtx {
        builder
            .with_age(params.age)
            .with_dump_all_to_cpu()
            .with_reset_all_on_cpu()
            .with_device_count(params.device_count)
            .with_register_cache()
            .with_remove_cache()
            .with_run_purge_gpu()
            .with_is_ipc(is_ipc)
            .with_is_all_on_cpu(true)
            .finish()
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
        track_await(COPY_CTX_ONCE.get_or_init(|| tokio::sync::RwLock::new(CopyCtx::make_instance(0.2, 16))).read()).await
    }

    pub fn cuda_actor<D:Into<ThoDevice>+ 'static>(&self, device:D)-> &Box<dyn CudaActorHandle> {
        let device = device.into();
        if let Some(real) = mock_global_ctx::manual_method_real_ctx() {
            return real.cuda_actor(device);
        }

        static COPY_CTX_ONCE: std::sync::OnceLock<Box<dyn CudaActorHandle>> = std::sync::OnceLock::new();
        COPY_CTX_ONCE.get_or_init(|| Box::new(CudaLocalActorHandle::new(device.ordinal())) as Box<dyn CudaActorHandle>)
    }


}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::future;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use futures_util::{StreamExt, TryStreamExt};
    use log::debug;
    use parking_lot::{Condvar, Mutex};
    use rand::{rng, RngExt};
    use tch::Device::{Cpu, Cuda};
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::Kind::Float;
    use tch::Tensor;
    use tokio::task::spawn_blocking;
    use tracing::Instrument;
    use thoosa_shared::ThoError;
    use thoosa_shims::context::ThoCudaContext;
    use thoosa_with_data::with_data;
    use crate::runner::common::cachable::DevicesWrap;
    use crate::runner::common::RunCtx;
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::run_ctx_2::RunCtxImpl;
    use crate::runner::global_ctx::CopyCtx;
    use crate as thoosa_common;
    use crate::cuda_utils::with_pooled_cuda_stream;
    use crate::dims::AnyDims;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::runner::tmp_cache::TmpCached;
    use crate::tho_tensor::ThoTensor;


    // #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[test]
    fn test_0() -> Result<(), ThoError>{
         fn test_xxx(gate:Arc<(Condvar, Mutex<bool>)>){
            let (cvar, lock) = &*gate;
            let mut released = lock.lock();

            while !*released {
                cvar.wait(&mut released);
            }

            debug!("test_0");
        }

        let gate = Arc::new((Condvar::new(), Mutex::new(false)));
        let gate_0 = Arc::clone(&gate);
        let gate_1 = Arc::clone(&gate);


        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .max_blocking_threads(2)
            .enable_all()
            .build()?;


        // #[tracing::instrument(level = "debug")]
        async fn zzzz() -> Result<(), ThoError>{
            tokio::time::sleep(Duration::from_secs(600)).await;
            Ok(())
        }


        rt.block_on(async {

            tokio::task::spawn(async move {
                debug!("spawn 0");
                let x = zzzz()
                    // .instrument(tracing::debug_span!("zzzz", file = file!(), line = line!()))
                    .await?;

                Ok::<(), ThoError>(())
            });

            // spawn_blocking(move ||{
            //     let i = test_xxx(gate_0)
            //         .instrument(tracing::debug_span!("test_0/1", file = file!(), line = line!()));
            //
            //     debug!("spawn_blocking 0");
            //     Ok::<_, ThoError>(i.into_inner())
            // });
            //
            // spawn_blocking(move ||{
            //     let i = test_xxx(gate_1)
            //         .instrument(tracing::debug_span!("test_0/1", file = file!(), line = line!()));
            //
            //     debug!("spawn_blocking 1");
            //     Ok::<_, ThoError>(i.into_inner())
            // });
            //
            // spawn_blocking(move ||{
            //
            //     debug!("spawn_blocking 2");
            //     Ok::<_, ThoError>(())
            // })


        });

        let (cvar, lock) = &*gate;
        *lock.lock() = true;


        Ok(())
    }

    // TODO
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_device_reset() -> Result<(), ThoError>{
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        // let cu_ctx = ThoCudaContext::new(0 as i32)?;
        // cu_ctx.reset()?;

        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let tmp_cache:TmpCached<ThoTensor<AnyDims, u32>> = TmpCached::new("tmp").await;

        // let t_0 = Arc::new(DevicesWrap::wrap(ThoTensor::new((), Tensor::rand(&[100000], FLOAT_CUDA))?));

        tmp_cache.insert(ThoTensor::new(777u32, Tensor::rand(&[100000], FLOAT_CUDA))?);
        let t_0 = tmp_cache.get(&777u32).unwrap();
        let mut root_ctx = RunCtxImpl::new(1, "root_ctx");

        let sum_0 = with_data!(root_ctx.branch(false), t_0; |ctx, t_0:&ThoTensor<AnyDims, u32>|{
            Ok::<_, ThoError>(t_0.as_ref().sum_dim_intlist(0, false, None).double_value(&[]))
        })?;
        debug!("sum_0={sum_0}");

        GlobalCtx::global().dump_all_to_cpu(&HashMap::new(), 1, "test").await?;
        // GlobalCtx::global().drop_device(1, "drop").await?;
        // let cu_ctx = ThoCudaContext::new(0 as i32)?;
        // cu_ctx.reset()?;

        let cuda_actor = GlobalCtx::global().cuda_actor(1);
        cuda_actor.reset().await?;
        debug!("reset done");

        let t_cpu = Tensor::rand(&[10, 1000], FLOAT_CPU);
        let t_cuda = cuda_actor.copy(t_cpu.into(), 1.into()).await?;
        debug!("t_cuda={t_cuda:?}");

        let t_0 = tmp_cache.get(&777u32).unwrap();
        let sum_1 = with_data!(root_ctx.branch(false), t_0; |ctx, t_0:&ThoTensor<AnyDims, u32>|{
            Ok::<_, ThoError>(t_0.as_ref().sum_dim_intlist(0, false, None).double_value(&[]))
        })?;

        debug!("sum_1={sum_1}");

        assert_eq!(sum_0, sum_1);

        Ok(())
    }

    fn sync_probe(label: &str, inst: &Instant) {
        let t = Instant::now();

        // Adjust device id if needed.
        tch::Cuda::synchronize(0);

        debug!("SYNC {label}: waited={:?} total_elapsed={:?}",t.elapsed(),inst.elapsed());
    }


    //nsys profile   -t cuda,osrt,nvtx   -s cpu   --cuda-memory-usage=true   --cudabacktrace=sync:1000000,memory:1000000,kernel:1000000   --force-overwrite=true   -o /tmp/tch-heavy   cargo test test_parallel_load -- --nocapture
    //nsys profile   --trace=cuda,osrt,nvtx   --gpu-metrics-devices=all   --cuda-memory-usage=true   -f true   -o /tmp/tch-heavy   cargo test test_parallel_load -- --nocapture
    //nsys-ui /tmp/tch-heavy.nsys-rep
    // #[tokio::test(flavor = "multi_thread", worker_threads = 64)]
    async fn test_parallel_load() -> Result<(), ThoError> {
        // let cu_ctx = ThoCudaContext::new(0 as i32).expect("failed to create CUDA context");
        // cu_ctx.bind_to_thread()?;
        GlobalCtx::init_devices(2);


        fn run_heavy(cycles: usize, n: i64, i:usize, inst:Arc<Instant>, input: Tensor) -> Result<Tensor, ThoError> {
            debug!("run_heavy: starting {i} elapsed={:?}", inst.elapsed());
            // Power of two makes the division exact in Float32.
            let mut a = input;

            // sync_probe(&format!("{i}/after ones"), &inst);

            for _ in 0..cycles {
                a = a
                    .f_matmul(&a)?
                    .f_div_scalar(n as f64)?;
            }
            // tch::Cuda::synchronize(0);

            // sync_probe(&format!("{i}/before return"), &inst);

            debug!("run_heavy: {i} done  elapsed={:?}", inst.elapsed());
            Ok(a)
        }

        let cycles = 10;

        let inst = Arc::new(Instant::now());

        let results = futures::stream::iter(0..100).map(|i| {
            let inst = Arc::clone(&inst);

            let n = rng().random_range(1_000..10_000);

            tokio::task::spawn(async move {
                let kind = Float;
                let cpu_input = Tensor::ones(&[n, n], FLOAT_CPU);
                let cuda_input = GlobalCtx::global().cuda_actor(Cuda(0)).copy(cpu_input.into(), 1.into()).await?.t();
                // let cuda_input = cpu_input.to(Cuda(0));

                let cuda_t =
                with_pooled_cuda_stream(1, |_|{
                    run_heavy(cycles, n, i, inst, cuda_input)
                }).await
                ?;


                Ok(GlobalCtx::global().cuda_actor(Cuda(0)).copy(cuda_t.into(), 0.into()).await?.t())
                // Ok(cuda_t.to(Cpu))
            })
        })
            .buffer_unordered(3)
            .map_err(ThoError::from)
            .and_then(future::ready)
            .try_collect::<Vec<_>>()
            .await?;

        // tch::Cuda::synchronize(0);



        let max_err = results.into_iter().map(|r|r
            .subtract_scalar(1.0)
            .abs()
            .max()
            .double_value(&[])).sum::<f64>();
        // let r = Tensor::f_cat(results.as_slice(), 0)?;

        // Force CUDA work to finish before measuring / checking.
        // let max_err = 0f64;


        debug!("run heavy={cycles} max_err={max_err} elapsed={}ms",inst.elapsed().as_millis());

        // assert!(max_err < 1e-6, "expected tensor full of 1.0, max_err={max_err}");

        Ok(())
    }
}
