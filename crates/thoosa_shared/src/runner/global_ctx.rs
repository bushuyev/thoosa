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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use futures_util::future::BoxFuture;
use thoosa_shared::traits::CachableIdTr;
use thoosa_shared::{Cause, ThoError};
use tokio::sync::RwLock as AsyncRwLock;
use thoosa_shared::runner::purger::PurgeCmd;
use crate::runner::purger::Purger;
/// Public “area handle”: global bit index (0..n_areas).
pub type CacheArea = u64;

/// Convert CacheArea(bit_index) -> (word_index, mask_in_word)
#[inline]
fn area_to_word_mask(area: CacheArea) -> (usize, u64) {
    let bit = area as usize;
    let word = bit / 64;
    let shift = (bit % 64) as u32;
    (word, 1u64 << shift)
}

#[inline]
fn valid_bits_mask_for_last_word(n_areas: usize) -> u64 {
    let rem = n_areas % 64;
    if rem == 0 { u64::MAX } else { (1u64 << rem) - 1 }
}

#[derive(Debug)]
pub struct CacheAreasKeeper {
    n_areas: usize,
    words: Vec<u64>, // memory region of bits: 1 = used, 0 = free
}

impl CacheAreasKeeper {
    pub fn new(n_areas: usize) -> Self {
        let n_words = (n_areas + 63) / 64;
        Self { n_areas, words: vec![0; n_words] }
    }

    /// Returns a u64 bit index (CacheArea).
    pub fn next_available_cache_area(&mut self) -> Result<CacheArea, ThoError> {
        debug!("CacheAreasKeeper::next_available_cache_area: self.n_areas={}", self.n_areas );
        if self.n_areas == 0 {
            return Err(Cause::SomethingElse("no cache areas available".into()).into());
        }

        let last = self.words.len() - 1;

        for (wi, w) in self.words.iter_mut().enumerate() {
            let valid = if wi == last { valid_bits_mask_for_last_word(self.n_areas) } else { u64::MAX };
            let free = (!*w) & valid;
            if free == 0 { continue; }

            let bit_in_word = free.trailing_zeros();        // 0..63
            let mask = 1u64 << bit_in_word;

            *w |= mask;

            let bit_index = (wi * 64 + bit_in_word as usize) as u64;
            return Ok(bit_index);
        }

        Err(Cause::SomethingElse("no cache areas available".into()).into())
    }

    pub fn dispose_cache_area(&mut self, area: CacheArea) {

        let bit = area as usize;
        if bit >= self.n_areas {
            debug!("dispose_cache_area: out-of-range area={area} n_areas={}", self.n_areas);
            return;
        }

        let (word, mask) = area_to_word_mask(area);
        debug!("dispose_cache_area: area={area} word={word} mask={mask:064b}");

        self.words[word] &= !mask;
    }

    /// Only for tests / introspection.
    pub fn words(&self) -> &[u64] { &self.words }
}

/// ---------- Atomic bitset for “SomeStruct holds many areas” ----------

#[derive(Debug)]
pub struct AtomicCacheAreas {
    n_areas: usize,
    words: Box<[AtomicU64]>,
}

impl AtomicCacheAreas {
    pub fn new(n_areas: usize) -> Self {
        let n_words = (n_areas + 63) / 64;
        let mut v = Vec::with_capacity(n_words);
        v.resize_with(n_words, || AtomicU64::new(0));
        Self { n_areas, words: v.into_boxed_slice() }
    }

    pub fn add_cache_area(&self, area: CacheArea) {
        let bit = area as usize;
        debug_assert!(bit < self.n_areas);
        let (word, mask) = area_to_word_mask(area);
        self.words[word].fetch_or(mask, Ordering::AcqRel);
    }

    pub fn remove_cache_area(&self, area: CacheArea) {
        let bit = area as usize;
        debug_assert!(bit < self.n_areas);
        let (word, mask) = area_to_word_mask(area);
        self.words[word].fetch_and(!mask, Ordering::AcqRel);
    }

    pub fn snapshot_words(&self) -> Vec<u64> {
        self.words.iter().map(|w| w.load(Ordering::Acquire)).collect()
    }
}

#[cfg(test)]
mod test_areas_keeper_tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use log::debug;
    use crate::runner::global_ctx::{AtomicCacheAreas, CacheAreasKeeper};
    use crate::ThoError;


    #[test]
    fn keeper_full_64_and_reuse() -> Result<(), ThoError> {
        let mut k = CacheAreasKeeper::new(64);

        for expected in 0..64u64 {
            let a = k.next_available_cache_area()?;
            assert_eq!(a, expected);
        }
        assert!(k.next_available_cache_area().is_err());
        assert_eq!(k.words()[0], u64::MAX);

        k.dispose_cache_area(0);
        k.dispose_cache_area(31);
        assert_eq!(k.words()[0], u64::MAX & !(1u64 << 0) & !(1u64 << 31));

        assert_eq!(k.next_available_cache_area()?, 0);
        assert_eq!(k.next_available_cache_area()?, 31);

        Ok(())
    }

    #[test]
    fn keeper_130_crosses_words() -> Result<(), ThoError> {
        let mut k = CacheAreasKeeper::new(130);

        for _ in 0..130 {
            k.next_available_cache_area()?;
        }
        assert!(k.next_available_cache_area().is_err());

        k.dispose_cache_area(0);
        k.dispose_cache_area(64);

        assert_eq!(k.next_available_cache_area()?, 0);
        assert_eq!(k.next_available_cache_area()?, 64);

        Ok(())
    }

    #[test]
    fn atomic_add_remove() -> Result<(), ThoError> {
        let mut k = CacheAreasKeeper::new(128);
        let s = AtomicCacheAreas::new(128);

        let a0 = k.next_available_cache_area()?;
        let a1 = k.next_available_cache_area()?;
        s.add_cache_area(a0);
        s.add_cache_area(a1);
        s.remove_cache_area(a0);
        s.remove_cache_area(a1);

        assert!(s.snapshot_words().into_iter().all(|w| w == 0));
        Ok(())
    }
    struct SomeStruct {
        cache_area: AtomicU64,
    }

    impl SomeStruct {
        pub fn add_cache_area(&self, ca: u64) -> u64 {
            self.cache_area.fetch_or(ca, Ordering::SeqCst) | ca
        }

        pub fn remove_cache_area(&self, ca: u64) -> u64 {
            self.cache_area.fetch_and(!ca, Ordering::AcqRel) & !ca
        }

        pub fn cache_areas(&self) -> u64 {
            self.cache_area.load(Ordering::Acquire)
        }
    }

    #[test]
    fn test_full()->Result<(), ThoError>{
        let mut keeper = CacheAreasKeeper::new(1);

        for i in 0..64 {
            let area = keeper.next_available_cache_area()?;
            debug!("area={area:064b} cache_areas={:064b}", keeper.words[0]);
        }

        // assert_eq!(0b1111111111111111111111111111111111111111111111111111111111111111, keeper.cache_areas);

        // keeper.dispose_cache_area(0b0000000000000000000000000000000000000000000000000000000000000001);
        //
        // assert_eq!(0b1111111111111111111111111111111111111111111111111111111111111110, keeper.cache_areas);
        //
        // keeper.dispose_cache_area(0b0000000000000000000000000000000010000000000000000000000000000000);
        //
        // assert_eq!(0b1111111111111111111111111111111101111111111111111111111111111110, keeper.cache_areas);
        //
        // assert_eq!(0b0000000000000000000000000000000000000000000000000000000000000001, keeper.next_available_cache_area()?);
        //
        // assert_eq!(0b0000000000000000000000000000000010000000000000000000000000000000, keeper.next_available_cache_area()?);


        Ok(())
    }

    #[test]
    fn sample(){
        let mut keeper = CacheAreasKeeper::new(3);

        let mut ss = SomeStruct {
            cache_area: AtomicU64::new(0)
        };

        let area_0 = keeper.next_available_cache_area().unwrap();
        ss.add_cache_area(area_0);
        let area_1 = keeper.next_available_cache_area().unwrap();
        ss.add_cache_area(area_1);


        ss.remove_cache_area(area_0);
        ss.remove_cache_area(area_1);

        assert_eq!(ss.cache_areas(), 0);

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
    purger: Purger
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

            caches_areas: parking_lot::RwLock::new(CacheAreasKeeper::new(10_000)),

            cache_age: Default::default(),
            purge_lock: tokio::sync::Mutex::new(()),
            purger: Purger::new()
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
        debug!("GlobalCtx::dispose_cache_area: cache_area={cache_area}");
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
