use crate::rayon_future::RayonFuture;
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
}

#[async_trait]
pub trait Output: WithId + Debug {
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

