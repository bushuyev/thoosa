use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, OnceLock};
#[allow(unused_imports)]
use mockall::automock;
use tokio::sync::RwLock as AsyncRwLock;
use crate::common::MyError;
use crate::common::cachable::CachableIdTr;
use crate::permits::NumPermits;
use crate::permits::{PermitSource, RunnerPerm};
use crate::permits::runner_queue::Arbiter;
use crate::purge::FuturesRunnerMapPurger;

#[derive(Debug)]
pub struct CacheAreasKeeper {
    cache_areas: u64,
}

impl CacheAreasKeeper {
    pub fn next_available_cache_area(&mut self) -> Result<u64, MyError> {
        let next = first_free_bit(self.cache_areas);
        if next == 0 {
            return Err(MyError::SomethingElse("no cache areas available".into()));
        }
        self.cache_areas |= next;

        Ok(next)
    }

    pub fn new() -> Self {
        Self { cache_areas: 0 }
    }
}

#[allow(dead_code)]
pub(crate) fn remove_mask(from: u64, mask: u64) -> u64 {
    !mask & from
}


pub(crate) fn first_free_bit(c: u64) -> u64 {
    if c == u64::MAX {
        0
    } else {
        1u64 << c.trailing_ones()
    }
}

#[derive(Debug)]
pub struct GlobalCtx {
    semaphores: Vec<Box<dyn PermitSource>>,
    device_count: usize,
    device_locks: Vec<Arc<AsyncRwLock<()>>>,
    caches: tokio::sync::RwLock<Vec<(&'static str, &'static dyn FuturesRunnerMapPurger)>>,
    is_all_on_cpu:Vec<AtomicBool>,
    caches_areas: parking_lot::RwLock<CacheAreasKeeper>,
    cache_age: AtomicU64,//for tracking CPU-only lifetime
    purge_lock:tokio::sync::Mutex<()>
}


static GLOBAL_CTX: OnceLock<GlobalCtx> = OnceLock::new();

#[cfg_attr(test, automock)]
impl GlobalCtx {
    fn new(device_count: usize) -> Self {
        Self {
            semaphores: (0..device_count)
                // .map(|_| Box::new(RunnerSemaphore::new(100)) as Box<dyn PermitSource>) //TODO config
                .map(|_| Box::new(Arbiter::new(100)) as Box<dyn PermitSource>)
                .collect(),

            device_count,

            device_locks: (0..device_count)
                .map(|_| Arc::new(AsyncRwLock::new(())))
                .collect(),

            caches: tokio::sync::RwLock::new(vec![]),

            is_all_on_cpu: (0..device_count)
                .map(|_| AtomicBool::new(false))
                .collect(),

            caches_areas: parking_lot::RwLock::new(CacheAreasKeeper::new()),

            cache_age: Default::default(),
            purge_lock: tokio::sync::Mutex::new(())
        }
    }

    pub fn init_devices(count: usize) -> &'static GlobalCtx {
        GLOBAL_CTX
            .set(GlobalCtx::new(count))
            .expect("`init_devices` must only be called once at program start");

        Self::global()
    }

    pub async fn purge_lock<'a>(&'a self) -> tokio::sync::MutexGuard<'a, ()>{
        self.purge_lock.lock().await
    }

    pub async fn register_cache(&self, cache: (&'static str, &'static dyn FuturesRunnerMapPurger)) {
        self.caches.write().await.push(cache);
    }

    pub fn global() -> &'static Self {
        GLOBAL_CTX.get().expect("call `init_devices` first")
    }

    pub fn device_count(&self) -> usize {
        self.device_count
    }

    pub fn device_lock(device_idx: usize) -> Arc<AsyncRwLock<()>> {
        Arc::clone(&GlobalCtx::global().device_locks[device_idx])
    }

    pub fn is_all_on_cpu(&self, device_idx:usize) -> bool {
        self.is_all_on_cpu[device_idx].load(Relaxed)
    }

    pub async fn reset_all_on_cpu(&self, device_idx:usize) {
        self.is_all_on_cpu[device_idx].store(false, Relaxed);
    }

    pub async fn acquire_many(&self, ordinal: usize, n: NumPermits, out_id: std::sync::Arc<dyn CachableIdTr>) -> Result<RunnerPerm, MyError> {
        self.semaphores[ordinal].acquire(n, out_id).await
    }

    ///ids_to_keep_by_type - should be few keys long - only inputs involved in the current operation
    pub async fn dump_all_to_cpu<'a>(
        &'a self,
        required_ids: &HashMap<&'static str, &'a HashSet<&'a dyn CachableIdTr>>,
        device_idx: usize,
    ) -> Result<(), MyError> {
        assert_ne!(device_idx, 0);
        println!("GlobalCtx::dump_all_to_cpu: device_idx={device_idx} required_ids={required_ids:?}");

        //should be ok to read lock -  caches are expected to be updated (GlobalCtx::register_cache) rarely, perhaps on app startup only or before running set of runs
        let maps = self.caches.read().await;

        for (type_name,  map) in maps.iter() {
            println!("GlobalCtx::dump_all_to_cpu: type_name={type_name}");

            map.purge_gpu(
                required_ids.get(type_name).map(|v|*v),

                device_idx
            ).await?;

            println!("GlobalCtx::dump_all_to_cpu: type_name={type_name} done");
        }

        self.is_all_on_cpu[device_idx].store( true, Relaxed);
        Ok(())
    }

    pub async fn purge_cpu(&self, purge_since: u64) {

        //should be ok to read lock - see dump_all_to_cpu
        let maps = self.caches.read().await;

        println!("GlobalCtx::purge_cpu: maps.len={}", maps.len());

        for e in maps.iter() {
            e.1.purge_cpu(purge_since).await;
        }
    }

    pub fn next_cache_area(&self) -> Result<u64, MyError> {
        self.caches_areas.write().next_available_cache_area()
    }

    #[allow(dead_code)]
    pub(crate) fn do_age(&self) {
        self.cache_age.fetch_add(1, Relaxed);
    }

    pub fn cache_age(&self) -> u64{
        self.cache_age.load( Relaxed)
    }

}



#[cfg(test)]
pub(crate) mod mock_global_ctx {
    use crate::global_ctx::MockGlobalCtx;

    pub struct MockGlobalCtxBuilder {
        mock:MockGlobalCtx
    }

    impl MockGlobalCtxBuilder {

        pub fn with_reset_all_on_cpu(mut self)->Self {
            let _ = self.mock.expect_reset_all_on_cpu();
            self
        }

        pub fn with_age(mut self, out:u64) -> Self {
            self.mock.expect_cache_age().return_const(out);

            self
        }

        pub fn with_dump_all_to_cpu(mut self) -> Self {
            let _ = self.mock.expect_dump_all_to_cpu();

            self
        }

        pub fn finish(self)->MockGlobalCtx {
            self.mock
        }


    }

    pub fn build() ->MockGlobalCtxBuilder {

        MockGlobalCtxBuilder {
            mock: super::MockGlobalCtx::default()
        }
    }
}
