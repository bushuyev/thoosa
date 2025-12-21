use crate::rayon_future::RayonFuture;
use crate::run_configs::{Cachable, Runnable};
use crate::traits::{Named, SimpleId, WithId};
use async_trait::async_trait;
use dashmap::{DashMap, DashSet, Entry};
use futures::future::{BoxFuture, Shared};
use futures::FutureExt;
use futures_util::future::join_all;
use log::debug;
use parking_lot::RwLock;
use rayon::ThreadPool;
use std::collections::HashMap;
use std::fmt::Debug;
use std::future::Future;
use std::marker::Tuple;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::thread;
use std::time::Instant;

pub type CachableId = SimpleId;

pub trait Input: Debug {
    fn id(&self) -> &CachableId {
        Box::leak(Box::new(SimpleId("todo")))//FIXME
    }
}

#[async_trait]
pub trait Output: WithId + Debug {
    async fn copy_to_device(self: Arc<Self>, device_ordinal: usize) -> Arc<Self> {
        todo!()
    }//TODO remove
}

impl <O> WithId for Arc<O>  where O:WithId + Debug {
    type ID = <O as WithId>::ID;

    fn id(&self) -> &Self::ID {
        self.as_ref().id()
    }
}
impl <O> Output for Arc<O> where O:Output + Debug {}

impl Named for (){

}
impl WithId for () {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl Output for () {

}

impl WithId for CachableId {
    type ID = CachableId;

    fn id(&self) -> &Self::ID {
        todo!()
    }
}

impl Input for CachableId {
    fn id(&self) -> &CachableId {
        //TODO remove, use WithId::id
        self
    }
}

pub trait ToOrdinal: Debug + Sync + Send {
    fn ordinal(&self) -> usize;
}

pub trait Runner {
    type Ctx: ApplicableRunCtx;
}

#[derive(Debug)]
pub struct Statistics {
    load_started_count: AtomicU64,
    load_ended_count: AtomicU64,
    pure_run_time: AtomicU64,
    load_time: AtomicU64,

    copy_started_count: AtomicU64,
    copy_ended_count: AtomicU64,
    copy_time: AtomicU64,

    loaded_keys: DashSet<CachableId>,
    copied_keys: DashSet<CachableId>,

    purge_gpu_count: AtomicU64,

    purge_cpu_count: AtomicU64,
    purge_cpu_time: AtomicU64,
}

//TODO feature to enable statistics
impl Statistics {
    pub fn new() -> Self {
        Self {
            load_started_count: 0.into(),
            load_ended_count: 0.into(),
            pure_run_time: 0.into(),
            load_time: 0.into(),
            copy_started_count: 0.into(),
            copy_ended_count: 0.into(),
            copy_time: 0.into(),
            loaded_keys: Default::default(),
            copied_keys: Default::default(),
            purge_gpu_count: 0.into(),
            purge_cpu_count: 0.into(),
            purge_cpu_time: 0.into(),
        }
    }

    pub fn load_time(&self) -> u64 {
        self.load_time.load(Ordering::Relaxed)
    }

    pub fn on_new_load(&self, input: &CachableId) {
        self.load_started_count.fetch_add(1, Ordering::Relaxed);
        self.loaded_keys.insert(input.clone());
    }

    pub fn on_new_load_done(&self, load_time: u64) {
        self.load_ended_count.fetch_add(1, Ordering::Relaxed);
        self.load_time.fetch_add(load_time, Ordering::Relaxed);
    }

    pub fn track_pure_run_time(&self, pure_run_time: u64) {
        self.pure_run_time
            .fetch_add(pure_run_time, Ordering::Relaxed);
    }

    pub fn copy_time(&self) -> u64 {
        self.copy_time.load(Ordering::Relaxed)
    }

    pub fn on_copy(&self, input: &CachableId) {
        self.copy_started_count.fetch_add(1, Ordering::Relaxed);
        self.copied_keys.insert(input.clone());
    }

    pub fn on_copy_done(&self, copy_time: u64) {
        self.copy_time.fetch_add(copy_time, Ordering::Relaxed);
        self.copy_ended_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn on_purge_gpu_caches(&self) {
        self.purge_gpu_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn on_purge_cpu_caches(&self, purge_cpu_caches_time: u64, purged_count: u64) {
        self.purge_cpu_count
            .fetch_add(purged_count, Ordering::Relaxed);
        self.purge_cpu_time
            .fetch_add(purge_cpu_caches_time, Ordering::Relaxed);
    }
}



#[async_trait]
pub trait FuturesRunnerMap: Sync + Send {
    async fn purge_gpu_cache_area(
        &self,
        key: CachableId,
        cache_area: u64,
        pool: &ThreadPool,
        device: &dyn ToOrdinal,
        statistics: Arc<Statistics>,
    );
    fn purge_cpu_caches(&self, statistics: &Arc<Statistics>, older_than: &Instant);
}



struct CacheAreasKeeper {
    cache_areas: u64,
}

impl CacheAreasKeeper {
    fn next_available_cache_area(&mut self) -> u64 {
        let next = first_free_bit(self.cache_areas);
        self.cache_areas |= next;

        next
    }

    pub fn new() -> Self {
        Self { cache_areas: 0 }
    }
}

pub trait ApplicableRunCtx: Debug + Send + Sync {
    fn cache_area(&self) -> u64;
    fn into_cached_iter_mut(self) -> impl Iterator<Item = Vec<CachableId>>;
}


pub(crate) fn first_free_bit(c: u64) -> u64 {
    if c == u64::MAX {
        0
    } else {
        1u64 << c.trailing_ones()
    }
}

pub(crate) fn remove_mask(from: u64, mask: u64) -> u64 {
    !mask & from
}

#[cfg(test)]
pub async fn dummy_load(statistics: &Statistics, _input: &Arc<CachableId>) {
    for _i in 0..DUMMY_LOAD_N {
        let _i = 0;
        async {
            let t_0 = Instant::now();

            let a = vec![0.1; 1_000_000];
            let b = vec![0.1; 1_000_000];
            let _c = a
                .into_iter()
                .zip(b.into_iter())
                .map(|(a, b)| a + b)
                .collect::<Vec<f32>>();

            statistics.track_pure_run_time(t_0.elapsed().as_millis() as u64);
        }
        .await;

        debug!(
            "dummy_load i={_i} thread={:?}",
            std::thread::current().name()
        );
    }
}

#[allow(dead_code)]
const DUMMY_LOAD_N: u32 = 3;

#[allow(dead_code)]
fn yield_n(n: usize) -> impl Future<Output = ()> {
    struct YieldN {
        left: usize,
    }
    impl Future for YieldN {
        type Output = ();
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            if self.left == 0 {
                Poll::Ready(())
            } else {
                self.left -= 1;
                // ask the executor (our dummy runner) to poll us again
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
    YieldN { left: n }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, dead_code, unused_variables)]
    use super::{
        CachableId,
        Input, Output, ToOrdinal,
    };
    use crate::traits::WithId;
    use async_trait::async_trait;
    use std::sync::Arc;

    #[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
    pub enum Devices {
        Cpu,
        Cuda(usize),
    }

    impl ToOrdinal for Devices {
        fn ordinal(&self) -> usize {
            match self {
                Devices::Cpu => 0,
                Devices::Cuda(id) => 1 + *id,
            }
        }
    }

    impl From<usize> for Devices {
        fn from(value: usize) -> Self {
            match value {
                0 => Devices::Cpu,
                id => Devices::Cuda(id - 1),
            }
        }
    }

    #[derive(Debug)]
    struct DataResource {
        id: CachableId,
        device: Devices,
        data: Vec<f32>,
    }

    impl DataResource {
        fn new(id: String, device: Devices, data: Vec<f32>) -> Self {
            Self {
                id: id.into(),
                device,
                data,
            }
        }

        fn copy_to_device(&self, device: Devices) -> Self {
            Self {
                id: self.id.clone(),
                device,
                data: self.data.clone(),
            }
        }
    }

    impl WithId for DataResource {
        type ID = CachableId;

        fn id(&self) -> &Self::ID {
            todo!()
        }
    }

    impl Input for DataResource {
        fn id(&self) -> &CachableId {
            &self.id
        }
    }

    #[derive(Debug)]
    pub struct MyRunCtx {
        cache_area: u64,

        loaded_cached: Vec<CachableId>,
        applicable_a_cached: Vec<CachableId>,
        applicable_b_cached: Vec<CachableId>,
    }

    impl MyRunCtx {
        pub fn new(cache_area: u64) -> Self {
            Self {
                cache_area,
                loaded_cached: vec![],
                applicable_a_cached: vec![],
                applicable_b_cached: vec![],
            }
        }
    }

    #[derive(Debug)]
    struct TestLoader {
        device: Devices,
        steps: u8,
    }

    impl TestLoader {
        fn new(device: Devices, steps: u8) -> TestLoader {
            Self { device, steps }
        }
    }

    #[derive(Debug, Hash, Eq, PartialEq, Clone)]
    struct LoaderInputId {
        id: String,
        import_id: String,
        applied_id: CachableId,
    }

    impl LoaderInputId {
        pub fn new(id: String, import_id: String) -> Self {
            let applied_id = format!("{}{}", id, import_id).into();
            Self {
                id,
                import_id,
                applied_id,
            }
        }
    }
    impl WithId for LoaderInputId {
        type ID = CachableId;

        fn id(&self) -> &Self::ID {
            todo!()
        }
    }

    impl Input for LoaderInputId {
        fn id(&self) -> &CachableId {
            &self.applied_id
        }
    }

    impl From<&str> for LoaderInputId {
        fn from(value: &str) -> Self {
            LoaderInputId::new(value.to_string(), "".to_string())
        }
    }

    #[derive(Debug)]
    struct LoaderOutput(DataResource);
    #[async_trait]
    impl Output for LoaderOutput {
        async fn copy_to_device(self: Arc<Self>, device_ordinal: usize) -> Arc<Self> {
            Arc::new(LoaderOutput(self.0.copy_to_device(device_ordinal.into())))
        }
    }
    unsafe impl Send for LoaderOutput {}



    #[derive(Debug)]
    struct ApplicableA {
        run_id: String,
        loader: Arc<TestLoader>,
        device: Devices,
    }

    impl ApplicableA {
        fn new(run_id: String, device: Devices) -> ApplicableA {
            Self {
                run_id,
                loader: Arc::new(TestLoader::new(device, 1)),
                device,
            }
        }
    }

    impl WithId for LoaderOutput {
        type ID = CachableId;

        fn id(&self) -> &Self::ID {
            todo!()
        }
    }

    impl Input for LoaderOutput {
        fn id(&self) -> &CachableId {
            WithId::id(&self.0)
        }
    }

    #[derive(Debug)]
    struct ApplicableAOutput(DataResource);
    #[async_trait]
    impl Output for ApplicableAOutput {
        async fn copy_to_device(self: Arc<Self>, device_ordinal: usize) -> Arc<Self> {
            Arc::new(ApplicableAOutput(
                self.0.copy_to_device(device_ordinal.into()),
            ))
        }
    }



    #[derive(Debug)]
    struct ApplicableB {
        run_id: String,
        applicable_a: Arc<ApplicableA>,
        device: Devices,
    }

    impl ApplicableB {
        fn new(run_id: String, input_run_id: String, device: Devices) -> ApplicableB {
            Self {
                applicable_a: Arc::new(ApplicableA::new(input_run_id, device)),
                run_id,
                device,
            }
        }
    }

    impl WithId for ApplicableAOutput {
        type ID = CachableId;

        fn id(&self) -> &Self::ID {
            todo!()
        }
    }

    impl Input for ApplicableAOutput {
        fn id(&self) -> &CachableId {
            Input::id(&self.0)
        }
    }

    #[derive(Debug)]
    struct ApplicableBOutput(DataResource);

    impl WithId for ApplicableBOutput {
        type ID = CachableId;

        fn id(&self) -> &Self::ID {
            todo!()
        }
    }

    #[async_trait]
    impl Output for ApplicableBOutput {
        async fn copy_to_device(self: Arc<Self>, device_ordinal: usize) -> Arc<Self> {
            Arc::new(ApplicableBOutput(
                self.0.copy_to_device(device_ordinal.into()),
            ))
        }
    }





    #[test]
    fn test_remove_mask_bug() {
        // keep lower bit, clear upper bit
        let v = 0b11u64;
        let sub = 0b10u64;
        assert_eq!(0b01, super::remove_mask(v, sub)); // ← fails: returns 0
    }


}
