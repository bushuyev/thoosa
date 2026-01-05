use crate::runner::permits::{NumPermits, MAX_PERMITS};
use crate::runner::permits::runner_queue::Arbiter;
use crate::runner::permits::{PermitSource, RunnerPerm};
use thoosa_shared::runner::purge::FuturesRunnerMapPurger;
use log::debug;
#[allow(unused_imports)]
use mockall::automock;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, OnceLock};
use thoosa_shared::traits::CachableIdTr;
use thoosa_shared::{Cause, ThoError};
use tokio::sync::RwLock as AsyncRwLock;

#[derive(Debug)]
pub struct CacheAreasKeeper {
    cache_areas: u64,
}

impl CacheAreasKeeper {
    pub fn next_available_cache_area(&mut self) -> Result<u64, ThoError> {
        let next = first_free_bit(self.cache_areas);
        if next == 0 {
            return Err(Cause::SomethingElse("no cache areas available".into()).into());
        }
        self.cache_areas |= next;

        Ok(next)
    }

    pub fn dispose_cache_area(&mut self, cache_area: u64)  {
        remove_mask(self.cache_areas, cache_area);
    }

    pub fn new() -> Self {
        Self { cache_areas: 0 }
    }
}

#[allow(dead_code)]
pub fn remove_mask(from: u64, mask: u64) -> u64 {
    !mask & from
}

pub fn first_free_bit(c: u64) -> u64 {
    if c == u64::MAX {
        0
    } else {
        1u64 << c.trailing_ones()
    }
}

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
}

static GLOBAL_CTX: OnceLock<GlobalCtx> = OnceLock::new();

#[cfg_attr(test, automock)]
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

            caches_areas: parking_lot::RwLock::new(CacheAreasKeeper::new()),

            cache_age: Default::default(),
            purge_lock: tokio::sync::Mutex::new(()),
        }
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
        self.purge_lock.lock().await
    }

    pub async fn register_cache(&self, cache: (&'static str, &'static dyn FuturesRunnerMapPurger)) {
        assert!(self.caches.write().await.insert(cache.0, cache.1).is_none());
    }

    pub async fn remove_cache(&self, key:&'static str) {

        assert!(self.caches.write().await.remove(key).is_some());
    }

    pub fn global() -> &'static Self {
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
        ordinal: usize,
        num_permits: NumPermits,
        requester_id: Arc<dyn CachableIdTr>,
    ) -> Result<RunnerPerm, ThoError> {
        self.semaphores[ordinal].acquire(num_permits, requester_id).await
    }

    pub async fn acquire_one(
        &self,
        ordinal: usize,
        requester_id: Arc<dyn CachableIdTr>,
    ) -> Result<RunnerPerm, ThoError> {

        let mut num_tries = 0;
        loop {

            if let Ok(permit) =self.semaphores[ordinal].acquire(1, Arc::clone(&requester_id)).await {

                debug!("GlobalCtx::acquire_one: requester_id={requester_id:?} got single permit on {num_tries} attempt");


                return Ok(permit);
            }
            num_tries+=1;
            if num_tries > 1000 {//TODO config
                return Err(ThoError::new("Can't get single permit"));
            }
        }
    }

    // #[cfg(debug_assertions)] TODO
    pub fn debug_semaphore(&self, device_idx:usize) -> &Box<dyn PermitSource>{
        &self.semaphores[device_idx]
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
        let maps = self.caches.read().await;

        for (type_name, map) in maps.iter() {
            debug!("GlobalCtx::dump_all_to_cpu: type_name={type_name}");

            map.purge_gpu(required_ids.get(type_name).map(|v| *v), device_idx).await?;

            debug!("GlobalCtx::dump_all_to_cpu: type_name={type_name} done");
        }

        self.is_all_on_cpu[device_idx].store(true, Relaxed);
        Ok(())
    }

    pub async fn purge_cpu(&self, purge_since: u64) {
        //should be ok to read lock - see dump_all_to_cpu
        let maps = self.caches.read().await;

        debug!("GlobalCtx::purge_cpu: maps.len={}", maps.len());

        for e in maps.iter() {
            e.1.purge_cpu(purge_since).await;
        }
    }

    pub fn next_cache_area(&self) -> Result<u64, ThoError> {
        self.caches_areas.write().next_available_cache_area()
    }

    pub fn dispose_cache_area(&self, cache_area:u64) {
        self.caches_areas.write().dispose_cache_area(cache_area)
    }

    #[allow(dead_code)]
    pub fn do_age(&self) {
        self.cache_age.fetch_add(1, Relaxed);
    }

    pub fn cache_age(&self) -> u64 {
        self.cache_age.load(Relaxed)
    }
}

#[cfg(test)]
pub mod mock_global_ctx {
    use crate::runner::global_ctx::{ MockGlobalCtx};

    pub struct MockGlobalCtxBuilder {
        mock: MockGlobalCtx,
    }

    impl MockGlobalCtxBuilder {
        pub fn with_reset_all_on_cpu(mut self) -> Self {
            let _ = self.mock.expect_reset_all_on_cpu();
            self
        }

        pub fn with_age(mut self, out: u64) -> Self {
            self.mock.expect_do_age();
            self.mock.expect_cache_age().return_const(out);

            self
        }

        pub fn with_device_count(mut self, out: usize) -> Self {
            self.mock.expect_device_count().return_const(out);

            self
        }
        pub fn with_register_cache(mut self) -> Self {
            self.mock.expect_register_cache();

            self
        }

        pub fn with_remove_cache(mut self) -> Self {
            self.mock.expect_remove_cache();

            self
        }

        pub fn with_dump_all_to_cpu(mut self) -> Self {
            let _ = self.mock.expect_dump_all_to_cpu();

            self
        }

        pub fn finish(self) -> MockGlobalCtx {
            self.mock
        }
    }

    pub fn build() -> MockGlobalCtxBuilder {
        MockGlobalCtxBuilder {
            mock: super::MockGlobalCtx::default(),
        }
    }

    pub fn build_default() -> MockGlobalCtx {
        build()
            .with_age(0)
            .with_dump_all_to_cpu()
            .with_reset_all_on_cpu()
            .with_device_count(2)
            .with_register_cache()
            .with_remove_cache()
            .finish()
    }

}
