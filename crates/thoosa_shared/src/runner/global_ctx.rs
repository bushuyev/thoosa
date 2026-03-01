use std::backtrace::Backtrace;
use crate::runner::permits::{NumPermits, MAX_PERMITS};
use crate::runner::permits::runner_queue::Arbiter;
use crate::runner::permits::{PermitSource, RunnerPerm};
use thoosa_shared::runner::purge::FuturesRunnerMapPurger;
use log::{debug, error, info};
#[allow(unused_imports)]
use mockall::automock;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, OnceLock};
use futures_util::future::BoxFuture;
use thoosa_shared::traits::CachableIdTr;
use thoosa_shared::{backtrace_head_lines, Cause, ThoError};
use tokio::sync::RwLock as AsyncRwLock;
use crate::runner::cache_areas::{CacheAreaId, CacheAreasKeeper};
use crate::runner::purger::Purger;
use crate::{with_async_perf_log, with_perf_log, LOCK_TIMEOUT};



#[derive(Debug)]
pub struct GlobalCtx {
    semaphores: Vec<Box<dyn PermitSource>>,
    num_devices: usize,
    device_locks: Vec<Arc<AsyncRwLock<()>>>,
    caches: tokio::sync::RwLock<HashMap<&'static str, &'static dyn FuturesRunnerMapPurger>>,
    is_all_on_cpu: Vec<AtomicBool>,
    caches_areas: parking_lot::RwLock<CacheAreasKeeper>,
    cache_age: AtomicU64, //for tracking CPU-only lifetime
    purge_lock: tokio::sync::Mutex<()>,
    purger: Purger,
    shutting_down:AtomicBool//TODO not needed if tokios shutdown_background in project main
}

static GLOBAL_CTX: OnceLock<GlobalCtx> = OnceLock::new();

#[cfg_attr(any(test, feature = "mock_global_ctx"), automock)]
impl GlobalCtx {

    fn new(num_devices: usize) -> Self {
        debug!("GlobalCtx::new: num_devices={num_devices}");

        assert!(num_devices > 0);//TODO test if it works without GPU


        let semaphores = (0..num_devices)
            // .map(|_| Box::new(RunnerSemaphore::new(100)) as Box<dyn PermitSource>) //TODO config
            .map(|device_idx| Box::new(Arbiter::new(MAX_PERMITS, device_idx)) as Box<dyn PermitSource>)
            .collect();

        debug!("GlobalCtx::new: semaphores={semaphores:?}");


        Self {
            semaphores,

            num_devices,

            device_locks: (0..num_devices)
                .map(|_| Arc::new(AsyncRwLock::new(())))
                .collect(),

            caches: tokio::sync::RwLock::new(HashMap::new()),

            is_all_on_cpu: (0..num_devices).map(|_| AtomicBool::new(false)).collect(),

            caches_areas: parking_lot::RwLock::new(CacheAreasKeeper::new(1000)),//100 * 64 cache areas

            cache_age: Default::default(),
            purge_lock: tokio::sync::Mutex::new(()),
            purger: Purger::new(),
            shutting_down: false.into()
        }
    }

    pub fn run_purge_gpu(&self, closures: Vec<Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>> + Send + Sync>>, cache_area: u64) ->Result<(), ThoError>{
        self.purger.run_purge_gpu(closures, cache_area)
    }

    ///num_devices: number of devices - CPU + GPU count
    pub fn init_devices(num_devices: usize) -> &'static GlobalCtx {
        debug!("GlobalCtx::init_devices: num_devices={num_devices}");

        GLOBAL_CTX
            .set(GlobalCtx::new(num_devices))
            .expect("`init_devices` must only be called once at program start");

        Self::global()
    }

    pub async fn purge_lock<'a>(&'a self) -> tokio::sync::MutexGuard<'a, ()> {
        with_async_perf_log(
            async || tokio::time::timeout(LOCK_TIMEOUT, self.purge_lock.lock()).await.expect("GlobalCtx::purge_lock: timed out waiting for purge lock"),
            "GlobalCtx/purge_lock",
        )
        .await
    }

    pub async fn register_cache(&self, cache: (&'static str, &'static dyn FuturesRunnerMapPurger)) {
        assert!(
            with_async_perf_log(
                async || tokio::time::timeout(LOCK_TIMEOUT, self.caches.write()).await.expect("GlobalCtx::register_cache: timed out waiting for caches write lock"),
                "GlobalCtx/register_cache/write_caches",
            )
            .await
            .insert(cache.0, cache.1)
            .is_none()
        );
    }

    pub async fn remove_cache(&self, key:&'static str) {

        assert!(
            with_async_perf_log(
                async || tokio::time::timeout(LOCK_TIMEOUT, self.caches.write()).await.expect(&format!("GlobalCtx::remove_cache, key={key}: timed out waiting for caches write lock")),
                "GlobalCtx/remove_cache/write_caches",
            )
            .await
            .remove(key)
            .is_some()
        );
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

    pub fn device_lock(device_idx: usize) -> Arc<AsyncRwLock<()>> {
        Arc::clone(&GlobalCtx::global().device_locks[device_idx])
    }

    pub fn is_all_on_cpu(&self, device_idx: usize) -> bool {
        self.is_all_on_cpu[device_idx].load(Relaxed)
    }

    pub async fn reset_all_on_cpu(&self, device_idx: usize) {
        self.is_all_on_cpu[device_idx].store(false, Relaxed);
    }

    pub async fn acquire_many(
        &self,
        device_idx: usize,
        num_permits: NumPermits,
        requester_id: Arc<dyn CachableIdTr>,
    ) -> Result<RunnerPerm, ThoError> {
        debug!("GlobalCtx::acquire_many: num_permits={num_permits} requester_id={requester_id:?} asking for many permits, self.semaphores[{device_idx}]={:?} \nbacktrace={}", self.semaphores[device_idx], Backtrace::capture());
        assert!(!requester_id.as_any().is::<()>(), "Got () requester_id backtrace={}", Backtrace::capture());

        self.semaphores[device_idx].acquire(num_permits, requester_id).await
    }

    pub async fn wait_for_permits_hint(&self, device_idx:usize) {
        self.semaphores[device_idx].wait_for_permits_hint().await;
    }

    pub async fn acquire_one(
        &self,
        device_idx: usize,
        requester_id: Arc<dyn CachableIdTr>,
    ) -> Result<RunnerPerm, ThoError> {

        assert!(!requester_id.as_any().is::<()>(), "Got () requester_id backtrace={}", Backtrace::capture());

        let mut num_tries = 0;

        debug!("GlobalCtx::acquire_one: requester_id={requester_id:?} asking for one permit, self.semaphores[{device_idx}]={:?} \nbacktrace={}", self.semaphores[device_idx], backtrace_head_lines(8));

        //can't handle OutOfPermits in with_data! and similar calls - acquire_one is needed to ensure no other thread acquired 100 permits and can drop all to cpu.
        loop {
            debug!("GlobalCtx::acquire_one: requester_id={requester_id:?} asking for single permit, {num_tries} attempt");

            match self.semaphores[device_idx].acquire(1, Arc::clone(&requester_id)).await {
                Ok(permit) => {
                    debug!("GlobalCtx::acquire_one: requester_id={requester_id:?} got single permit on {num_tries} attempt");
                    return Ok(permit);
                }
                Err(e) => {
                    match e.what {
                        Cause::OutOfPermits => {
                            num_tries+=1;
                            if num_tries > 1000 {//TODO config
                                return Err(ThoError::new("Can't get single permit"));
                            }
                            self.semaphores[device_idx].wait_for_permits_hint().await;
                        }

                        e => {
                            return Err(ThoError::new(format!("Can't get single permit, got {e:?}")));
                        }
                    }
                }
            }
        }
    }

    // #[cfg(debug_assertions)] TODO
    pub fn debug_semaphore(&self, device_idx:usize) -> &Box<dyn PermitSource>{
        &self.semaphores[device_idx]
    }

    // #[cfg(debug_assertions)] TODO
    pub async fn debug_caches(&self, device_idx:usize) {
        let maps: Vec<(&'static str, &'static dyn FuturesRunnerMapPurger)> = {
            let maps = with_async_perf_log(
                async || tokio::time::timeout(LOCK_TIMEOUT, self.caches.read()).await.expect("GlobalCtx::debug_caches: timed out waiting for caches read lock"),
                "GlobalCtx/debug_caches/read_caches",
            ).await;
            maps.iter().map(|(type_name, map)| (*type_name, *map)).collect()
        };

        for (type_name, map) in maps {
            info!("GlobalCtx::debug_caches: type_name={type_name}");
            if let Err(e) = map.debug_gpu(device_idx).await {
                error!("GlobalCtx::debug_caches: failed with {e:?}");
            }
        }
    }

    ///ids_to_keep_by_type - should be few keys long - only inputs involved in the current operation
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="GlobalCtx::dump_all_to_cpu"))]
    pub async fn dump_all_to_cpu<'a>(
        &'a self,
        required_ids: &HashMap<&'static str, &'a HashSet<&'a dyn CachableIdTr>>,
        device_idx: usize,
    ) -> Result<(), ThoError> {
        assert_ne!(device_idx, 0);
        debug!("GlobalCtx::dump_all_to_cpu: device_idx={device_idx} required_ids={required_ids:?}");

        //should be ok to read lock -  caches are expected to be updated (GlobalCtx::register_cache) rarely, perhaps on app startup only or before running set of runs
        let maps: Vec<(&'static str, &'static dyn FuturesRunnerMapPurger)> = {
            let maps = with_async_perf_log(
                async || tokio::time::timeout(LOCK_TIMEOUT, self.caches.read()).await.expect("GlobalCtx::dump_all_to_cpu: timed out waiting for caches read lock"),
                "GlobalCtx/dump_all_to_cpu/read_caches",
            )
                .await;
            maps.iter().map(|(type_name, map)| (*type_name, *map)).collect()
        };

        for (type_name, map) in maps {
            debug!("GlobalCtx::dump_all_to_cpu: type_name={type_name}");

            map.purge_gpu(required_ids.get(type_name).map(|v| *v), device_idx).await?;

            debug!("GlobalCtx::dump_all_to_cpu: type_name={type_name} done");
        }

        self.is_all_on_cpu[device_idx].store(true, Relaxed);
        Ok(())
    }

    pub async fn purge_cpu(&self, purge_last: u64) {

        //should be ok to read lock - see dump_all_to_cpu
        let maps: Vec<(&'static str, &'static dyn FuturesRunnerMapPurger)> = {
            let maps = with_async_perf_log(
                async || tokio::time::timeout(LOCK_TIMEOUT, self.caches.read()).await.expect("GlobalCtx::purge_cpu: timed out waiting for caches read lock"),
                "GlobalCtx/purge_cpu/read_caches",
            )
                .await;
            maps.iter().map(|(type_name, map)| (*type_name, *map)).collect()
        };

        let purge_since = self.cache_age.load(Relaxed).saturating_sub(purge_last);

        debug!("GlobalCtx::purge_cpu: purge_since={purge_since} maps.len={}", maps.len());
        if purge_since > 0 {
            // GlobalCtx::global().debug_caches(1).await;

            for (type_name, map) in maps {
                info!("GlobalCtx::purge_cpu: purged {} from {}", map.purge_cpu(purge_since).await, type_name);
            }

            // GlobalCtx::global().debug_caches(1).await;
        }
    }


    pub fn next_cache_area(&self) -> Result<CacheAreaId, ThoError> {
        with_perf_log(
            || self.caches_areas.try_write_for(LOCK_TIMEOUT).expect("GlobalCtx::next_cache_area: timed out waiting for cache_areas write lock"),
            "GlobalCtx/next_cache_area/write_cache_areas",
        )
        .next_available_cache_area()
    }

    pub fn dispose_cache_area(&self, cache_area:CacheAreaId) {
        with_perf_log(
            || self.caches_areas.try_write_for(LOCK_TIMEOUT).expect("GlobalCtx::dispose_cache_area: timed out waiting for cache_areas write lock"),
            "GlobalCtx/dispose_cache_area/write_cache_areas",
        )
        .dispose_cache_area(cache_area)
    }

    pub fn do_age(&self) {
        self.cache_age.fetch_add(1, Relaxed);
    }

    pub fn cache_age(&self) -> u64 {
        self.cache_age.load(Relaxed)
    }


    pub async fn shutdown(&self) {
        self.shutting_down.store(true, Relaxed);
        //TODO

        for a in &self.semaphores {
            a.shutdown().await;
        }

    }

    pub fn is_shutting_down(&self)->bool {
        self.shutting_down.load(Relaxed)
    }
}

#[cfg(any(test, feature = "mock_global_ctx"))]
pub mod mock_global_ctx {
    use std::sync::Arc;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use crate::runner::permits::{NumPermits, PermitSource, RunnerPerm};
    use crate::Cause;
    use crate::ThoError;
    use async_trait::async_trait;
    use crate::traits::CachableIdTr;
    use mockall::predicate::eq;
    use tokio::sync::Semaphore;
    use tokio::task::block_in_place;
    use crate::comp_id::SynKey;
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

    fn make_permit(num_permits: NumPermits) -> RunnerPerm {
        let sem = Arc::new(Semaphore::new(num_permits as usize));
        RunnerPerm::new(Arc::clone(&sem).try_acquire_many_owned(u32::from(num_permits)).expect("permit"))
    }

    #[derive(Debug)]
    struct DummyPermitSource;

    #[async_trait]
    impl PermitSource for DummyPermitSource {
        async fn acquire(
            &self,
            num_permits: NumPermits,
            _out_id: Arc<dyn CachableIdTr>,
        ) -> Result<RunnerPerm, ThoError> {
            Ok(make_permit(num_permits))
        }

        async fn wait_for_permits_hint(&self) {}

        fn move_permits(&self, _from: &SynKey, _to: &SynKey) {}
    }

    #[derive(Debug)]
    struct DelegatingPermitSource {
        real: &'static GlobalCtx,
        device_idx: usize,
    }

    #[async_trait]
    impl PermitSource for DelegatingPermitSource {
        async fn acquire(
            &self,
            num_permits: NumPermits,
            out_id: Arc<dyn CachableIdTr>,
        ) -> Result<RunnerPerm, ThoError> {
            self.real.debug_semaphore(self.device_idx).acquire(num_permits, out_id).await
        }

        async fn wait_for_permits_hint(&self) {
            self.real.debug_semaphore(self.device_idx).wait_for_permits_hint().await;
        }

        fn move_permits(&self, from: &SynKey, to: &SynKey) {
            self.real.debug_semaphore(self.device_idx).move_permits(from, to);
        }
    }

    fn dummy_semaphore() -> Box<dyn PermitSource> {
        Box::new(DummyPermitSource)
    }

    fn delegating_semaphore(real: &'static GlobalCtx, device_idx: usize) -> Box<dyn PermitSource> {
        Box::new(DelegatingPermitSource { real, device_idx })
    }


    pub struct MockGlobalCtxBuilder {
        mock: MockGlobalCtx,
        real: Option<&'static GlobalCtx>
    }

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
                    move |cache| {
                        block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(real.register_cache(cache));
                        })
                    }
                });
            } else {
                self.mock.expect_register_cache();
            }

            self
        }

        pub fn with_remove_cache(mut self) -> Self {
            if let Some(real) = self.real {
                self.mock.expect_remove_cache().returning({
                    move |key| {
                        tokio::task::block_in_place(move || {
                            tokio::runtime::Handle::current().block_on(real.remove_cache(key));
                        });
                    }
                });
            } else {
                self.mock.expect_remove_cache();
            }

            self
        }

        pub fn with_dump_all_to_cpu(mut self) -> Self {
            if let Some(real) = self.real {
                let _ = self.mock.expect_dump_all_to_cpu().returning({
                    move |required_ids, device_idx| {
                        tokio::runtime::Handle::current().block_on(real.dump_all_to_cpu(required_ids, device_idx))
                    }
                });
            } else {
                let _ = self.mock.expect_dump_all_to_cpu();
            }

            self
        }

        pub fn with_acquire_many(mut self, behavior: AcquireBehavior) -> Self {
            if let Some(real) = self.real {
                self.mock.expect_acquire_many().times(0..).returning({
                    move |device_idx, num_permits, requester_id| {
                        tokio::runtime::Handle::current().block_on(real.acquire_many(device_idx, num_permits, requester_id))
                    }
                });
            } else {
                self.mock.expect_acquire_many().times(0..).returning({
                    move |_device_idx, num_permits, _requester_id| {
                        match behavior {
                            AcquireBehavior::Grant => Ok(make_permit(num_permits)),
                            AcquireBehavior::Deny => Err(Cause::OutOfPermits.into()),
                            AcquireBehavior::Boom(msg) => Err(Cause::SomethingElse(msg.to_string()).into())
                        }
                    }
                });
            }

            self
        }

        pub fn with_acquire_one(mut self, behavior: AcquireBehavior) -> Self {
            if let Some(real) = self.real {
                self.mock.expect_acquire_one().times(0..).returning({
                    move |device_idx, requester_id| {
                        tokio::runtime::Handle::current().block_on(real.acquire_one(device_idx, requester_id))
                    }
                });
            } else {
                self.mock.expect_acquire_one().times(0..).returning({
                    move |_device_idx, _requester_id| {
                        match behavior {
                            AcquireBehavior::Grant => Ok(make_permit(1)),
                            AcquireBehavior::Deny => Err(Cause::OutOfPermits.into()),
                            AcquireBehavior::Boom(msg) => Err(Cause::SomethingElse(msg.to_string()).into())
                        }
                    }
                });
            }

            self
        }


        pub fn with_wait_for_permits_hint(mut self) -> Self {
            if let Some(real) = self.real {
                let _ = self.mock.expect_wait_for_permits_hint().times(0..).returning({
                    move |device_idx| {
                        tokio::runtime::Handle::current().block_on(real.wait_for_permits_hint(device_idx));
                    }
                });
            } else {
                let _ = self.mock.expect_wait_for_permits_hint().times(0..).returning(|_device_idx| {});
            }

            self
        }

        pub fn with_run_purge_gpu(mut self) -> Self {
            if let Some(real) = self.real {
                self.mock.expect_run_purge_gpu().times(0..).returning({
                    move |closures, cache_area| {
                        real.run_purge_gpu(closures, cache_area)
                    }
                });
            } else {
                self.mock.expect_run_purge_gpu().times(0..).returning(|_closures, _cache_area| Ok(()));
            }

            self
        }

        pub fn with_debug_semaphore(mut self) -> Self {
            if let Some(real) = self.real {
                for device_idx in 0..real.device_count() {
                    let sem = delegating_semaphore(real, device_idx);
                    self.mock.expect_debug_semaphore()
                        .with(eq(device_idx))
                        .times(0..)
                        .return_const(sem);
                }
            } else {
                self.mock.expect_debug_semaphore().times(0..).return_const(dummy_semaphore());
            }

            self
        }

        pub fn with_next_cache_area(mut self, mut cache_areas:Vec<u64>) -> Self {
            if let Some(real) = self.real {
                let _ = self.mock.expect_next_cache_area().returning({
                    move || {
                        real.next_cache_area()
                    }
                });
            } else {
                let _ = self.mock.expect_next_cache_area().returning(move ||{
                    Ok(cache_areas.pop().unwrap())
                });
            }

            self
        }

        pub fn finish(self) -> MockGlobalCtx {
            self.mock
        }

        pub fn build(self) -> MockGlobalCtx {
            build_with_params_inner(self, MockGlobalCtxBuildParams::default())
        }

        #[cfg(any(test, feature = "mock_global_ctx"))]
        pub fn install_global_ctx_mock() -> impl Drop {
            use crate::runner::global_ctx::{mock_global_ctx, MockGlobalCtx};
            let ctx = MockGlobalCtx::global_context();
            if unsafe {USE_REAL_CTX} {
                use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;

                let real: &'static RealGlobalCtx = RealGlobalCtx::real_global();


                ctx.expect().returning(move || {
                    // NOTE: this leaks once per call to GlobalCtx::global() in the test.
                    Box::leak(Box::new(from_real(real).build()))
                });

                return ctx;
            }

            ctx.expect().returning(|| {
                Box::leak(Box::new(
                    mock_global_ctx::build_default(),
                )) as &'static MockGlobalCtx
            });

            ctx
        }

    }

    fn build_with_params_inner(builder: MockGlobalCtxBuilder, params: MockGlobalCtxBuildParams) -> MockGlobalCtx {
        builder
            .with_age(params.age)
            .with_dump_all_to_cpu()
            .with_reset_all_on_cpu()
            .with_device_count(params.device_count)
            .with_register_cache()
            .with_remove_cache()
            .with_next_cache_area(params.next_cache_areas)
            .with_acquire_many(params.acquire_many)
            .with_acquire_one(params.acquire_one)
            .with_wait_for_permits_hint()
            .with_run_purge_gpu()
            .with_debug_semaphore()
            .finish()
    }

    pub fn build_with_params(params: MockGlobalCtxBuildParams) -> MockGlobalCtx {
        build_with_params_inner(build(), params)
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
        build_with_params(MockGlobalCtxBuildParams::default())
    }

    pub static mut USE_REAL_CTX:bool = false;
}
