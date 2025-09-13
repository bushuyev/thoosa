use crate::rayon_future::RayonFuture;
use async_trait::async_trait;
use dashmap::{DashMap, DashSet, Entry};
use futures::future::{BoxFuture, Shared};
use futures::FutureExt;
use futures_util::future::join_all;
use parking_lot::RwLock;
use rayon::ThreadPool;
use std::collections::HashMap;
use std::fmt::Debug;
use std::future::Future;
use std::marker::Tuple;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::thread;
use std::time::Instant;
use log::debug;
use crate::run_configs::{Cachable, Runnable};

pub type CachableId = String;

pub trait Input:Send+Sync+Debug {
    fn id(&self) -> &CachableId;
}



#[async_trait]
pub trait Output:Send+Sync+Debug {
    async fn copy_to_device(self:Arc<Self>, device_ordinal:usize) -> Arc<Self>;
}


impl Input for CachableId {
    fn id(&self) -> &CachableId {
        self
    }
}


pub trait ToOrdinal:Debug + Sync + Send {
    fn ordinal(&self) -> usize;
}


pub trait Runner {
    type Ctx:RunCtx;
}


impl  Debug for FuturesRunner {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FuturesRunner")
            .field("num_devices", &self.num_devices)
            .field("caches_areas", &self.caches_areas.read().cache_areas)
            .finish_non_exhaustive()
    }
}

pub struct DeviceThreadPools {
    cpu_pool: ThreadPool,
    gpu_pools: HashMap<usize, ThreadPool>,
}

impl DeviceThreadPools {

    pub fn map_pool(pool_name:&str)->ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .spawn_handler(|thread_builder| {
                debug!("spawning {pool_name} thread index={}", thread_builder.index());

                let b = thread::Builder::new();

                b.name(format!("{}/{:?}", pool_name, thread_builder.index()))
                    .spawn(|| thread_builder.run())?;

                Ok(())
            })

            .build()
            .expect(pool_name)
    }

    fn new(cpu_threads: usize, gpu_threads: &[usize]) -> Self {


        let cpu_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cpu_threads)
            .spawn_handler(|thread_builder| {
                debug!("spawning cpu thread index={}", thread_builder.index());

                let b = thread::Builder::new();

                b.name(format!("{}/{:?}", "cpu_pool", thread_builder.index()))
                    .spawn(|| thread_builder.run())?;

                Ok(())
            })

            .build()
            .expect("CPU pool");

        let mut gpu_pools = HashMap::new();
        for (device_id, num_threads) in gpu_threads.iter().enumerate() {
            let pool = rayon::ThreadPoolBuilder::new()
                .spawn_handler(|thread_builder| {
                    debug!("spawning gpu {device_id} thread index={}", thread_builder.index());

                    let b = thread::Builder::new();

                    b.name(format!("{}_{}/{:?}", "gpu_pool", {device_id}, thread_builder.index()))
                        .spawn(|| thread_builder.run())?;

                    Ok(())
                })

                .num_threads(*num_threads)
                .build()
                .expect(&format!("GPU {} pool", device_id));
            gpu_pools.insert(device_id, pool);
        }

        Self { cpu_pool, gpu_pools }
    }

    fn get_pool(&self, device: &dyn ToOrdinal) -> &rayon::ThreadPool {
        match device.ordinal() {
            0 => &self.cpu_pool,
            id => self.gpu_pools.get(&(id-1)).expect("GPU pool not found"),
        }
    }
}

#[derive(Debug)]
pub struct DataInDevices<O:Output + Sync + Send + 'static> {
    device_futures:Box<[Option<Shared<BoxFuture<'static, Arc<O>>>>]>,//0 - CPU, 1 - Cuda(0), 2 - Cuda(1), ...
    device_cache_areas:Box<[u64]>,
    in_cpu_since:Option<Instant>,//last time it was copied to device_futures[0]
}

impl <O> DataInDevices<O>
where
    O: Output + Send + Sync + 'static,

{

    fn new_with_future<I, A: Cachable<I, O, ()> + 'static> (
        input: Arc<I>,
        device_ordinal: usize,
        num_devices: usize,
        applicable: Arc<A>,
        runner: Arc<A::R>,
        statistics: &Arc<Statistics>,
        ctx: Option<Arc<RwLock<<<A as Runnable<I, O, ()>>::R as Runner>::Ctx>>>
    )
        -> (Self, Shared<BoxFuture<'static, Arc<O>>>)
    where I: Input+Send+Sync+'static
    {
        debug!("DataInDevices: new input={input:?} device_ordinal={device_ordinal}  ");

        let statistics_clone = statistics.clone();
        /*
        let mut i =0;
        while !can_run(i, device) {
            rayon::yield_now();
            i+=1
        }
        */
        let t_0 = Instant::now();
        let ctx_for_future = ctx.as_ref().map(|c|Arc::clone(c));


        let fut = async move {
            let r = applicable.async_run(input.clone(), runner, ctx_for_future).await;
            debug!("run_cached_task: done input={input:?}  ");
            statistics_clone.on_new_load_done(t_0.elapsed().as_millis() as u64);
            r
        }.boxed().shared();

        let mut device_futures = vec![const { None }; num_devices].into_boxed_slice();
        let mut device_cache_areas = vec![0; num_devices].into_boxed_slice();

        device_futures[device_ordinal] = Some(fut.clone());
        device_cache_areas[device_ordinal] = ctx.map(|v|v.read().cache_area()).unwrap_or(0);

        (
            Self {
                device_futures,
                device_cache_areas,
                in_cpu_since: None
            },
            fut
        )
    }

    fn get_from_any_device<R:RunCtx>(&mut self, device_ordinal:usize, _pool:&ThreadPool, statistics: &Arc<Statistics>, ctx: Option<Arc<RwLock<R>>> ) -> Shared<BoxFuture<'static, Arc<O>>> {
        debug!("get_from_any_device: device_ordinal={device_ordinal} ctx={ctx:?} self.device_cache_areas[device_ordinal]={:?}", self.device_cache_areas[device_ordinal]);

        self.device_cache_areas[device_ordinal] |= ctx.map(|v|v.read().cache_area()).unwrap_or(0);

        debug!("get_from_any_device: self.device_cache_areas[device_ordinal]={:?}", self.device_cache_areas[device_ordinal]);

        self.in_cpu_since = None;

        match self.device_futures[device_ordinal].as_ref() {
            Some(f) => {
                f.clone()
            },
            None => {

                //TODO cache device with data
                let (from_device, _) = self.device_futures
                    .iter()
                    .enumerate()
                    .find(|v| v.1.is_some())
                    .map(|v|(v.0, v.1.as_ref().unwrap().clone()))
                    .expect("No data to copy");

                self.device_copy_future(from_device, device_ordinal, statistics)
            }
        }
    }

    fn device_copy_future(
        &mut self,
        from_device: usize,
        to_device: usize,
        statistics: &Arc<Statistics>,
    ) -> Shared<BoxFuture<'static, Arc<O>>> {

        let data_fut = self.device_futures[from_device]
            .as_ref()
            .expect("source device future missing")
            .clone();

        let statistics_arc = Arc::clone(&statistics);
        let t_0 = Instant::now();

        let fut = async move {

            let from_data = data_fut.await;

            let r = from_data.copy_to_device(to_device).await;

            statistics_arc.on_copy_done(t_0.elapsed().as_millis() as u64);

            r
        }.boxed().shared();

        self.device_futures[to_device] = Some(fut.clone());
        self.device_cache_areas[to_device] |= self.device_cache_areas[from_device];
        fut
    }

}

#[derive(Debug)]
pub struct Statistics {

    load_started_count:AtomicU64,
    load_ended_count:AtomicU64,
    pure_run_time: AtomicU64,
    load_time: AtomicU64,

    copy_started_count:AtomicU64,
    copy_ended_count:AtomicU64,
    copy_time: AtomicU64,

    loaded_keys:DashSet<CachableId>,
    copied_keys:DashSet<CachableId>,

    purge_gpu_count:AtomicU64,

    purge_cpu_count:AtomicU64,
    purge_cpu_time: AtomicU64,
}

//TODO feature to enable statistics
impl Statistics {
    pub fn new() -> Self {
        Self {
            load_started_count:0.into(),
            load_ended_count:0.into(),
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

    pub fn on_new_load_done(&self, load_time:u64) {
        self.load_ended_count.fetch_add(1, Ordering::Relaxed);
        self.load_time.fetch_add(load_time, Ordering::Relaxed);
    }

    pub fn track_pure_run_time(&self, pure_run_time:u64) {
        self.pure_run_time.fetch_add(pure_run_time, Ordering::Relaxed);

    }

    pub fn copy_time(&self) -> u64 {
        self.copy_time.load(Ordering::Relaxed)
    }

    pub fn on_copy(&self, input: &CachableId) {
        self.copy_started_count.fetch_add(1, Ordering::Relaxed);
        self.copied_keys.insert(input.clone());
    }

    pub fn on_copy_done(&self, copy_time:u64) {
        self.copy_time.fetch_add(copy_time, Ordering::Relaxed);
        self.copy_ended_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn on_purge_gpu_caches(&self) {
        self.purge_gpu_count.fetch_add(1, Ordering::Relaxed);
    }


    pub fn on_purge_cpu_caches(&self, purge_cpu_caches_time:u64, purged_count:u64) {
        self.purge_cpu_count.fetch_add(purged_count, Ordering::Relaxed);
        self.purge_cpu_time.fetch_add(purge_cpu_caches_time, Ordering::Relaxed);
    }
}

#[derive(Debug)]
pub struct MapWithThread<O>
where
    O: Output + Send + Sync + 'static {
    map:DashMap<CachableId, DataInDevices<O>>,
    pool:ThreadPool,
    name:String
}

impl <O> MapWithThread<O>
where
    O: Output + Send + Sync + 'static {
    pub fn new(name:&str) -> Self {
        Self {
            name:name.to_string(),
            map: Default::default(),
            pool: DeviceThreadPools::map_pool(name)
        }
    }

    pub fn name(&self)->&str {
        &self.name
    }

    pub fn keys(&self) -> Vec<String> {
        self.map.iter().map(|v|v.key().clone()).collect::<Vec<String>>()
    }

    #[cfg(test)]
    pub fn get_cache_areas(&self, key:&str) -> Box<[u64]> {
        self.map.get::<CachableId>(&key.to_string()).expect(&format!("no {:?}", key)).device_cache_areas.clone()
    }

}

#[async_trait]
pub trait FuturesRunnerMap:Sync+Send{
    async fn purge_gpu_cache_area(&self, key: CachableId, cache_area:u64, pool:&ThreadPool, device:&dyn ToOrdinal, statistics: Arc<Statistics>);
    fn purge_cpu_caches(&self, statistics: &Arc<Statistics>, older_than:&Instant);
}

#[async_trait]
impl<O> FuturesRunnerMap for MapWithThread<O>
where
    O: Output + Send + Sync + 'static,
{
    async fn purge_gpu_cache_area(
        &self,
        key: CachableId,
        cache_area: u64,
        _pool: &ThreadPool,
        device:&dyn ToOrdinal,
        statistics: Arc<Statistics>,
    ) {

        debug!("purge_gpu_cache_area: key={key:?} cache_area={cache_area:?} device={device:?}");
        if let Some(mut entry) = self.map.get_mut(&key) {
            let device_idx = device.ordinal();

            let new_mask = remove_mask(entry.device_cache_areas[device_idx], cache_area);
            if device_idx != 0 && new_mask == 0 {//device_idx == 0 it's on CPU and will be purged by purge_cpu_caches
                statistics.on_purge_gpu_caches();

                if entry.device_futures[0].is_none() {
                    entry
                        .device_copy_future(device_idx, 0, &statistics)
                        .await;
                }

                entry.device_futures[device_idx] = None;//TODO Cancel if pending (causing Canceled errors)?
                entry.in_cpu_since = Some(Instant::now());
            }
            entry.device_cache_areas[device_idx] = new_mask;
        }
    }

    fn purge_cpu_caches(&self, statistics: &Arc<Statistics>, older_than: &Instant) {
        let now = Instant::now();

        let mut purged_count = 0u64;

        self.map.retain(|key, v| {
            let to_retain = v.in_cpu_since.map(|v| &v > older_than).unwrap_or(true);

            debug!("purge_cpu_caches: key={key:?} to_retain={to_retain:?} v.in_cpu_since={:?}", v.in_cpu_since);

            if !to_retain {
                purged_count+=1;
            }
            to_retain
        });

        statistics.on_purge_cpu_caches(now.elapsed().as_millis() as u64, purged_count);
    }
}

pub struct FuturesRunner {
    num_devices: usize,
    device_pools: DeviceThreadPools,

    caches_areas: RwLock<CacheAreasKeeper>,
    statistics: Arc<Statistics>
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

pub trait RunCtx:Debug + Send + Sync {
    fn cache_area(&self) -> u64;
    fn into_cached_iter_mut(self) -> impl Iterator<Item=Vec<CachableId>>;
}

impl FuturesRunner {

    pub fn new(devices_threads:&[usize]) -> Self {

        Self {
            device_pools: DeviceThreadPools::new(devices_threads[0], &devices_threads[1..]),
            num_devices: devices_threads.len(),


            caches_areas: RwLock::new(CacheAreasKeeper::new()),
            statistics: Arc::new(Statistics::new())
        }
    }

    //TODO OfMemSize param
    pub fn run_async_fn<I, O, AF>(
        self: &Arc<Self>,
        input: I,
        mut async_fn:AF,
        device:&impl ToOrdinal
    ) -> BoxFuture<'static, O>
    where
        O: Send + Sync + 'static,
        I: Tuple + Debug + Send + Sync + 'static,
        AF: AsyncFnMut(I)->O + Send,
        for <'a> <AF as AsyncFnMut<(I,)>>::CallRefFuture<'a>: Future<Output = O>  + Send + 'static,
    {

        let pool = self.device_pools.get_pool(device);

        pool.install(move ||{//or pool.scope(move |scope| { scope.spawn_future(async move {
            RayonFuture::<()>::install_future(
                async_fn(input)
            )
        }).boxed()

    }

    //TODO OfMemSize param
    pub fn run_fn<I, O, AF>(
        self: &Arc<Self>,
        input: I,
        mut fun:AF,
        device:&impl ToOrdinal
    ) -> O
    where
        O: Send + Sync + 'static,
        I: Tuple + Debug + Send + Sync + 'static,
        AF: FnMut(I)->O + Send,
    {

        let pool = self.device_pools.get_pool(device);

        pool.install(move ||{//or pool.scope(move |scope| { scope.spawn_future(async move {

            fun(input)

        })
    }



    pub fn run<I, O, A: Runnable<I, O, ()> + 'static>(
        self: &Arc<Self>,
        input: Arc<I>,
        runnable:&Arc<A>,
        runner:&Arc<A::R>,
        ctx: Option<Arc<RwLock<<<A as Runnable<I, O, ()>>::R as Runner>::Ctx>>>,
    ) -> BoxFuture<'static, O>
    where
        O: Debug + Send + Sync + 'static,
        I: Debug + Send+Sync+'static,
    {
        debug!("run_task");


        let statistics_arc = Arc::clone(&self.statistics);

        let t_0 = Instant::now();
        let device = runnable.device();
        let pool = self.device_pools.get_pool(&device);
        let applicable= Arc::clone(runnable);
        let runner= Arc::clone(runner);

        pool.install(move ||{//or pool.scope(move |scope| { scope.spawn_future(async move {
            RayonFuture::<()>::install_future(async move {
                let r = applicable.async_run(input.clone(), runner, ctx).await;
                debug!("run_task: done input={input:?}");

                statistics_arc.on_new_load_done(t_0.elapsed().as_millis() as u64);

                Arc::into_inner(r).unwrap()//TODO add arc in Applicable, remove for Runnable
            })
        }).boxed()

    }


    pub fn run_cached_task<I, O, A: Cachable<I, O, ()> + 'static>(
        self: &Arc<Self>,
        map_with_thread: &MapWithThread<O>,
        input: Arc<I>,
        applicable:&Arc<A>,
        runner:&Arc<A::R>,
        ctx: Option<Arc<RwLock<<<A as Runnable<I, O, ()>>::R as Runner>::Ctx>>>
    ) -> Shared<BoxFuture<'static, Arc<O>>>
    where
        O: Output + Send + Sync + 'static,
        I: Input+Send+Sync+'static,
    {
        // let map_name = map_with_thread.name();
        map_with_thread.pool.install(move || {
            let device = applicable.device();
            let id = applicable.applied_id(input.id());

            debug!("run_cached_task: id={id:?} installed ");
            let debug_id = id.clone();

            /*let mut entry = None::<Entry<AppliedId, DataInDevices<O>>>;
            let mut count = 0;
            while entry.is_none() {
                entry = map_with_thread.map.try_entry(id.clone());
                if entry.is_none() {
                    count+=1;
                    if count > 100 {
                        error!("count={count}");
                        panic!("stuck");
                    }
                    debug!("run_cached_task: id={id:?} yielding, map_name={} thread name={:?}", map_name, thread::current().name());
                    rayon::yield_now();
                } else {
                    debug!("run_cached_task: id={id:?} got entry, map_name={} thread name={:?}", map_name, thread::current().name());
                }
            }
            match entry.unwrap() {*/


            // let entry_op = map_with_thread.map.try_entry(id.clone());
            // if entry_op.is_none() {
            //     error!("map {map_name} is locked for entry id={id} ");
            // } else {
            //     debug!("map {map_name} is available for entry id={id} ");
            // }

            let fut = match map_with_thread.map.entry(id.clone()) {
                Entry::Occupied(mut e) => {
                    self.statistics.on_copy(input.id());
                    debug!("run_cached_task: id={debug_id:?} Occupied ");

                    let f = e.get_mut().get_from_any_device(
                        device.ordinal(),
                        self.device_pools.get_pool(&device),
                        &self.statistics,
                        ctx
                    ).clone();

                    f
                },
                Entry::Vacant(v) => {
                    self.statistics.on_new_load(input.id());
                    debug!("run_cached_task: id={debug_id:?} Vacant ");

                    let (data_in_devices_future, fut) = DataInDevices::new_with_future(
                        input,
                        device.ordinal(),
                        self.num_devices,
                        Arc::clone(applicable),
                        Arc::clone(&runner),
                        &self.statistics,
                        ctx
                    );
                    v.insert(data_in_devices_future);
                    debug!("run_cached_task: id={id:?} data_in_devices_future inserted ");
                    fut
                }
            };

            self.device_pools.get_pool(&device).install(||{
                RayonFuture::<()>::install_future(fut.clone());
            });

            fut
        })

    }


    pub async fn purge_all_gpu_caches<C:RunCtx>(
        self: Arc<Self>,
        ctx: C,
        device:&dyn ToOrdinal,
        maps_iter: impl Iterator<Item=&dyn FuturesRunnerMap> + Sized
    ) {
        let pool  = self.device_pools.get_pool(device);
        let cache_area = ctx.cache_area();
        debug!("purge_all_gpu_caches: cache_area={cache_area:?} device={device:?}");

        join_all(ctx
            .into_cached_iter_mut()
            .zip(maps_iter)
            .flat_map(|(ctx_keys, futs_map)|{
                ctx_keys.into_iter().map( |key| {//TODO keys may repeat a lot, fold in a set first?
                    futs_map.purge_gpu_cache_area(
                        key, cache_area, pool, device, Arc::clone(&self.statistics)
                    )
                })
            })).await;
    }

    pub fn purge_all_cpu_caches<'a>(&self, older_than: &Instant, maps_iter: impl Iterator<Item=&'a dyn FuturesRunnerMap> + Sized) {
        let statistics = Arc::clone(&self.statistics);

        for futures in maps_iter {

            futures.purge_cpu_caches(&statistics, older_than);
        }
    }

    pub fn next_available_cache_area(&self)->u64 {
        self.caches_areas.write().next_available_cache_area()
    }

    pub fn statistics(self:&Arc<Self>) -> &Statistics {
        &self.statistics
    }
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
            let _c = a.into_iter().zip(b.into_iter()).map(|(a, b)| a + b).collect::<Vec<f32>>();

            statistics.track_pure_run_time(t_0.elapsed().as_millis() as u64);

        }.await;

        debug!("dummy_load i={_i} thread={:?}", std::thread::current().name());
    }

}

#[allow(dead_code)]
const DUMMY_LOAD_N:u32 = 3;

#[allow(dead_code)]
fn yield_n(n: usize) -> impl Future<Output = ()> {
    struct YieldN { left: usize }
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
    use std::borrow::Cow;
    use super::{dummy_load, Cachable, DataInDevices, FuturesRunner, FuturesRunnerMap, Input, Output, CachableId, RunCtx, Runner, ToOrdinal, DeviceThreadPools, MapWithThread, yield_n};
    use crate::rayon_future::ScopeFutureExt;
    use async_trait::async_trait;
    use dashmap::DashMap;
    use futures::executor::block_on;
    use futures_util::future::{join_all, BoxFuture, Shared};
    use parking_lot::RwLock;
    use rayon::iter::ParallelIterator;
    use rayon::prelude::{IntoParallelRefIterator, ParallelBridge};
    use rayon::ThreadPool;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};
    use crossbeam::sync::WaitGroup;
    use log::{debug, set_max_level, Level, LevelFilter};
    use crate::applicables_runner::tests::Devices::Cpu;
    use crate::run_configs::Runnable;
    use crate::test_utils::set_panic_hook;

    #[derive(Debug, Copy, Clone, Eq, Hash, PartialEq)]
    pub enum Devices {
        Cpu,
        Cuda(usize)
    }

    impl ToOrdinal for Devices {
        fn ordinal(&self) -> usize {
            match self {
                Devices::Cpu => 0,
                Devices::Cuda(id) => 1 + *id
            }
        }
    }

    impl From<usize> for Devices {
        fn from(value: usize) -> Self {
            match value {
                0 => Devices::Cpu,
                id => Devices::Cuda(id-1)
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
        fn new(id:String, device: Devices, data: Vec<f32>) -> Self {
            Self { id, device, data }
        }

        fn copy_to_device(&self, device:Devices) -> Self {
            Self {
                id: self.id.clone(),
                device,
                data: self.data.clone()
            }
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
            Self { cache_area, loaded_cached: vec![], applicable_a_cached: vec![], applicable_b_cached: vec![] }
        }

    }

    impl RunCtx for MyRunCtx {
        fn into_cached_iter_mut(self) -> impl Iterator<Item=Vec<CachableId>> {
            [
                self.loaded_cached,
                self.applicable_a_cached,
                self.applicable_b_cached
            ].into_iter()
        }

        fn cache_area(&self) -> u64 {
            self.cache_area
        }

    }


    #[derive(Debug)]
    struct TestLoader {
        device: Devices,
        steps: u8
    }

    impl TestLoader {

        fn new(device: Devices, steps: u8) -> TestLoader {
            Self { device, steps }
        }

    }

    #[derive(Debug, Hash, Eq, PartialEq, Clone)]
    struct LoaderInputId {
        id:String,
        import_id:String,
        applied_id: CachableId
    }

    impl LoaderInputId {
        pub fn new(id: String, import_id: String) -> Self {
            let applied_id = format!("{}{}", id, import_id);
            Self { id, import_id,  applied_id}
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
        async fn copy_to_device(self:Arc<Self>, device_ordinal:usize) -> Arc<Self> {
            Arc::new(LoaderOutput(self.0.copy_to_device(device_ordinal.into())))
        }
    }
    unsafe impl Send for LoaderOutput {}

    #[async_trait]
    impl Runnable<LoaderInputId, LoaderOutput, ()> for TestLoader {
        type R = MyRunner;

        async fn async_run(self: Arc<Self>, input: Arc<LoaderInputId>, runner: Arc<MyRunner>, ctx: Option<Arc<RwLock<MyRunCtx>>>) -> Arc<LoaderOutput> {
            debug!("Loader::async_apply: input={input:?} ");
            // thread::sleep(Duration::new(0, 1000));
            dummy_load(&runner.futures_runner.statistics, &Arc::new(input.id().clone())).await;

            for step in 0..self.steps {
                // yield_n(1).await;
                dummy_load(&runner.futures_runner.statistics, &Arc::new(input.id().clone())).await;
            }

            Arc::new(LoaderOutput(DataResource::new(input.id().clone(), self.device, vec![0.1, 0.2, 0.3, 0.4])))
        }

        fn device(&self) -> impl ToOrdinal {
            self.device.clone()
        }
    }

    #[async_trait]
    impl Cachable<LoaderInputId, LoaderOutput, ()> for TestLoader {


        fn applied_id(&self, prev_id: &CachableId) -> String {
            Self::format_0(&prev_id)
        }

        fn name(&self) -> String {
            "".to_string()
        }

        fn run_id(&self) -> String {
            "".to_string()
        }

    }


    #[derive(Debug)]
    struct ApplicableA {
        run_id:String,
        loader:Arc<TestLoader>,
        device: Devices,
    }

    impl ApplicableA {
        fn new(run_id:String, device: Devices) -> ApplicableA {
            Self {
                run_id,
                loader: Arc::new(TestLoader::new(device, 1)),
                device
            }
        }
    }


    impl Input for LoaderOutput {
        fn id(&self) -> &CachableId {
            self.0.id()
        }
    }

    #[derive(Debug)]
    struct ApplicableAOutput(DataResource);
    #[async_trait]
    impl Output for ApplicableAOutput {
        async fn copy_to_device(self:Arc<Self>, device_ordinal:usize) -> Arc<Self> {
            Arc::new(ApplicableAOutput(self.0.copy_to_device(device_ordinal.into())))
        }
    }


    #[async_trait]
    impl Runnable<LoaderInputId, ApplicableAOutput, ()> for ApplicableA {
        type R = MyRunner;

        async fn async_run(self: Arc<Self>, input: Arc<LoaderInputId>, runner: Arc<MyRunner>, ctx: Option<Arc<RwLock<MyRunCtx>>>) -> Arc<ApplicableAOutput> {
            let resource_0 = runner.clone().run_loader(&input, &self.loader, ctx).await;
            // debug!("ApplicableA::apply: input={input:?}");

            dummy_load(&runner.futures_runner.statistics, &Arc::new(input.id().clone())).await;


            Arc::new(ApplicableAOutput(DataResource::new(self.applied_id(input.id()), self.device, [vec![0.1, 0.2, 0.3, 0.4], resource_0.0.data.clone()].concat())))
        }

        fn device(&self) -> impl ToOrdinal {
            self.device.clone()
        }
    }

    #[async_trait]
    impl Cachable<LoaderInputId, ApplicableAOutput, ()> for ApplicableA {
        fn applied_id(&self, prev_id: &CachableId) -> String {
            Cachable::format_id(self, &self.loader.applied_id(prev_id))
        }


        fn name(&self) -> String {
            "ApplicableA".to_string()
        }

        fn run_id(&self) -> String {
            self.run_id.clone()
        }

    }

    #[derive(Debug)]
    struct ApplicableB {
        run_id:String,
        applicable_a: Arc<ApplicableA>,
        device: Devices,
    }

    impl ApplicableB {

        fn new(run_id:String, input_run_id:String, device:Devices) -> ApplicableB {
            Self {
                applicable_a: Arc::new(ApplicableA::new(input_run_id, device)),
                run_id,
                device,
            }
        }

    }


    impl Input for ApplicableAOutput {
        fn id(&self) -> &CachableId {
            self.0.id()
        }
    }

    #[derive(Debug)]
    struct ApplicableBOutput(DataResource);
    #[async_trait]
    impl Output for ApplicableBOutput {
        async fn copy_to_device(self:Arc<Self>, device_ordinal:usize) -> Arc<Self> {
            Arc::new(ApplicableBOutput(self.0.copy_to_device(device_ordinal.into())))
        }
    }

    #[async_trait]
    impl Runnable<LoaderInputId, ApplicableBOutput, ()> for ApplicableB {
        type R = MyRunner;

        async fn async_run(self: Arc<Self>, input: Arc<LoaderInputId>, runner: Arc<MyRunner>, ctx: Option<Arc<RwLock<MyRunCtx>>>) -> Arc<ApplicableBOutput> {
            let resource_a = runner.clone().run_applicable_a(&input, &self.applicable_a, ctx).await;
            debug!("ApplicableB::apply: input={input:?}");

            dummy_load(&runner.futures_runner.statistics, &Arc::new(input.id().clone())).await;


            Arc::new(ApplicableBOutput(DataResource::new(self.applied_id(input.id()), self.device, [vec![0.1, 0.2, 0.3, 0.4], resource_a.0.data.clone()].concat())))
        }

        fn device(&self) -> impl ToOrdinal {
            self.device.clone()
        }
    }

    #[async_trait]
    impl Cachable<LoaderInputId, ApplicableBOutput, ()> for ApplicableB {
        fn applied_id(&self, input_id: &CachableId) -> String {
            Cachable::format_id(self, &self.applicable_a.applied_id(input_id))
        }

        fn name(&self) -> String {
            "ApplicableB".to_string()
        }

        fn run_id(&self) -> String {
            self.run_id.clone()
        }


    }


    #[derive(Debug)]
    struct MyRunner {
        futures_runner:Arc<FuturesRunner>,

        loader_cache_fut: MapWithThread<LoaderOutput>,

        applicable_a_cache_fut: MapWithThread<ApplicableAOutput>,

        applicable_b_cache_fut: MapWithThread<ApplicableBOutput>
    }

    impl Runner for MyRunner {
        type Ctx = MyRunCtx;
    }

    impl MyRunner {
        pub fn new() -> Self {
            Self {
                futures_runner: Arc::new(FuturesRunner::new(&[10, 2, 2])),

                loader_cache_fut: MapWithThread::new("loader_cache_pool"),

                applicable_a_cache_fut: MapWithThread::new("applicable_a_cache_pool"),

                applicable_b_cache_fut: MapWithThread::new("applicable_b_cache_pool"),
            }
        }


        pub fn run_loader(
            self: &Arc<Self>,
            input: &LoaderInputId,
            applicable: &Arc<TestLoader>,
            ctx: Option<Arc<RwLock<MyRunCtx>>>
        ) -> Shared<BoxFuture<'static, Arc<LoaderOutput>>> {

            if let Some(ctx) = ctx.as_ref() {
                ctx.write().loaded_cached.push(applicable.applied_id(input.id()));
            }

            self.futures_runner.run_cached_task(
                &self.loader_cache_fut,
                Arc::new(input.clone()),
                applicable,
                self,
                ctx,
            )
        }

        pub fn run_applicable_a(
            self: &Arc<Self>,
            input: &LoaderInputId,
            applicable: &Arc<ApplicableA>,
            ctx: Option<Arc<RwLock<MyRunCtx>>>
        ) -> Shared<BoxFuture<'static, Arc<ApplicableAOutput>>> {

            if let Some(ctx) = ctx.as_ref() {
                ctx.write().applicable_a_cached.push(applicable.applied_id(input.id()));
            }

            self.futures_runner.run_cached_task(
                &self.applicable_a_cache_fut,
                Arc::new(input.clone()),
                applicable,
                self,
                ctx,
            )
        }

        pub fn run_applicable_b(
            self: &Arc<Self>,
            input: &LoaderInputId,
            applicable: &Arc<ApplicableB>,
            ctx: Option<Arc<RwLock<MyRunCtx>>>
        ) -> Shared<BoxFuture<'static, Arc<ApplicableBOutput>>> {

            if let Some(ctx) = ctx.as_ref() {
                ctx.write().applicable_b_cached.push(applicable.applied_id(input.id()));
            }

            self.futures_runner.run_cached_task(
                &self.applicable_b_cache_fut,
                Arc::new(input.clone()),
                applicable,
                self,
                ctx,
            )
        }

        pub(crate)  fn to_futs_iter(&self) -> impl Iterator<Item = &dyn FuturesRunnerMap> {
            [
                &self.loader_cache_fut as &dyn FuturesRunnerMap,
                &self.applicable_a_cache_fut as &dyn FuturesRunnerMap,
                &self.applicable_b_cache_fut as &dyn FuturesRunnerMap
            ].into_iter()
        }

        pub async fn purge_all_gpu_caches(self:Arc<Self>, ctx: MyRunCtx, device:Devices){
            Arc::clone(&self.futures_runner).purge_all_gpu_caches(ctx, &device, self.to_futs_iter()).await;
        }

        fn next_ctx(&self) -> Arc<RwLock<MyRunCtx>> {
            Arc::new(RwLock::new(MyRunCtx::new(self.futures_runner.caches_areas.write().next_available_cache_area())))
        }

    }

    #[test]
    fn test_loader() {
        let t_0 = Instant::now();

        let my_loader = Arc::new(MyRunner::new());

        let res = block_on(my_loader.clone().run_loader(&LoaderInputId::new("0".to_string(), "1".to_string()), &Arc::new(TestLoader::new(Devices::Cpu, 1)), None));

        debug!("loader_cache keys:{:?}", my_loader.loader_cache_fut.keys());
        debug!("got res={res:?} in {:?}", t_0.elapsed());
        //TODO asserts
    }

    #[test]
    fn test_applicable_a() {
        let t_0 = Instant::now();

        let my_loader = Arc::new(MyRunner::new());

        let res = block_on(
            my_loader.clone().run_applicable_a(&LoaderInputId::new("0".to_string(), "1".to_string()), &Arc::new(ApplicableA::new("0".to_string(), Devices::Cpu)), None)
        );

        debug!("loader_cache keys:{:?}", my_loader.loader_cache_fut.keys());
        debug!("applicable_a_cache keys:{:?}", my_loader.applicable_a_cache_fut.keys());
        debug!("got res={res:?} in {:?}", t_0.elapsed());
    }

    #[test]
    fn test_applicable_b() {
        let t_0 = Instant::now();

        let my_loader = Arc::new(MyRunner::new());

        let res = block_on(
            my_loader.clone().run_applicable_b(&LoaderInputId::new("0".to_string(), "1".to_string()), &Arc::new(ApplicableB::new("0".to_string(), "0".to_string(), Devices::Cuda(0))), None)
        );


        debug!("loader_cache keys:{:?}", my_loader.loader_cache_fut.keys());
        debug!("applicable_a_cache keys:{:?}", my_loader.applicable_a_cache_fut.keys());
        debug!("applicable_b_cache keys:{:?}", my_loader.applicable_b_cache_fut.keys());
        // debug!("got res={res:?} in {:?}", t_0.elapsed());


        let res = block_on(
            my_loader.clone().run_applicable_b(&LoaderInputId::new("0".to_string(), "1".to_string()), &Arc::new(ApplicableB::new("1".to_string(), "0".to_string(), Devices::Cuda(1))), None)
        );

        debug!("loader_cache keys:{:?}", my_loader.loader_cache_fut.keys());
        debug!("applicable_a_cache keys:{:?}", my_loader.applicable_a_cache_fut.keys());
        debug!("applicable_b_cache keys:{:?}", my_loader.applicable_b_cache_fut.keys());
        debug!("got res={res:?} in {:?}", t_0.elapsed());
    }

    #[test]
    fn test_load(){
        // set_max_level(LevelFilter::Info);
        set_panic_hook();

        let resources = ["aaa", "bbb", "ccc", "ddd", "eee"];
        let devices = [Devices::Cpu, Devices::Cuda(0), Devices::Cuda(1)];

        let my_loader = Arc::new(MyRunner::new());
        let count = AtomicU32::new(0);
        let n = 10u32;
        let t_0 = Instant::now();

        let main_pool = main_pool(10);

        main_pool.scope(|scope| {

            resources
                .into_iter()
                .cycle()
                .take(n as usize)
                .enumerate()
                .zip(devices.into_iter().cycle())
                .par_bridge()
                .for_each(|((i, resource), device)| {

                    debug!("resource={resource}_{i}, device={device:?} start");

                    let res = block_on(
                        Arc::clone(&my_loader).run_loader(&LoaderInputId::new(format!("{resource}_{i}"), "1".to_string()), &Arc::new(TestLoader::new(device, 1)), None)
                    );

                    // debug!("resource={resource:?}, device={device:?} res={res:?} done");

                    count.fetch_add(1, Ordering::SeqCst);

                    // debug!("resource={resource:?}, device={device:?} res={res:?}");
                });

        });

        debug!("futures_runner={my_loader:?}");
        debug!("count={count:?}");
        debug!("done in {:?} futures_runner.statistics={:?}", t_0.elapsed().as_millis(), my_loader.futures_runner.statistics);
        assert_eq!(count.load(Ordering::SeqCst), n);
        assert_eq!(
            my_loader.futures_runner.statistics.load_started_count.load(Ordering::Relaxed),
            my_loader.futures_runner.statistics.load_ended_count.load(Ordering::Relaxed)
        );

        assert_eq!(my_loader.futures_runner.statistics.load_started_count.load(Ordering::Relaxed), 10/*(n*3) as u128*/);//3: Loader -> ApplicableA -> ApplicableB

        assert_eq!(my_loader.futures_runner.statistics.copy_started_count.load(Ordering::Relaxed), 0);

    }


    #[test]
    fn test_run_applicable_b(){
        // set_max_level(LevelFilter::Info);
        set_panic_hook();

        let resources = ["aaa", "bbb", "ccc", "ddd", "eee"];
        let devices = [Cpu, Devices::Cuda(0), Devices::Cuda(1)];

        let my_loader = Arc::new(MyRunner::new());
        let count = AtomicU32::new(0);
        let n = 10u32;
        let t_0 = Instant::now();

        let main_pool = main_pool(10);

        main_pool.scope(|scope| {

            resources
                .into_iter()
                .cycle()
                .take(n as usize)
                .enumerate()
                .zip(devices.into_iter().cycle())
                .par_bridge()
                .for_each(|((i, resource), device)| {

                    debug!("resource={resource}_{i}, device={device:?} start");

                    let res = block_on(
                        Arc::clone(&my_loader).run_applicable_b(&LoaderInputId::new(format!("{resource}_{i}"), "1".to_string()), &Arc::new(ApplicableB::new("1".to_string(), "0".to_string(), device)), None)
                    );

                    // debug!("resource={resource:?}, device={device:?} res={res:?} done");

                    count.fetch_add(1, Ordering::SeqCst);

                    // debug!("resource={resource:?}, device={device:?} res={res:?}");
                });

        });

        debug!("futures_runner={my_loader:?}");
        debug!("count={count:?}");
        debug!("done in {:?} futures_runner.statistics={:?}", t_0.elapsed().as_millis(), my_loader.futures_runner.statistics);
        assert_eq!(count.load(Ordering::SeqCst), n);
        assert_eq!(
            my_loader.futures_runner.statistics.load_started_count.load(Ordering::Relaxed),
            my_loader.futures_runner.statistics.load_ended_count.load(Ordering::Relaxed)
        );

        assert_eq!(my_loader.futures_runner.statistics.load_started_count.load(Ordering::Relaxed), 30/*(n*3) as u128*/);//3: Loader -> ApplicableA -> ApplicableB

        assert_eq!(my_loader.futures_runner.statistics.copy_started_count.load(Ordering::Relaxed), 0);

    }

    #[test]
    fn test_load_with_copy(){

        let resources = ["aaa", "bbb", "ccc", "ddd", "eee"];
        let devices = [Devices::Cpu, Devices::Cuda(0), Devices::Cuda(1)];

        let my_loader = Arc::new(MyRunner::new());
        let count = AtomicU32::new(0);
        let n = 10u32;
        let t_0 = Instant::now();

        let main_pool = main_pool(10);

        main_pool.scope(|scope| {

            resources
                .into_iter()
                .cycle()
                .take(n as usize)
                .enumerate()
                .zip(devices.into_iter().cycle())
                .par_bridge()
                .for_each(|((i, resource), device)| {

                    debug!("resource={resource}, device={device:?} start");

                    let res = block_on(
                        Arc::clone(&my_loader).run_applicable_b(&format!("{resource}").as_str().into(), &Arc::new(ApplicableB::new("1".to_string(), "0".to_string(), device)), None)
                    );

                    count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                });

        });

        debug!("done in {:?} futures_runner.statistics={:?}", t_0.elapsed().as_millis(), my_loader.futures_runner.statistics);
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), n);
        assert_eq!(
            my_loader.futures_runner.statistics.load_started_count.load(Ordering::Relaxed),
            my_loader.futures_runner.statistics.load_ended_count.load(Ordering::Relaxed)
        );

        assert_eq!(my_loader.futures_runner.statistics.load_started_count.load(Ordering::Relaxed), 15/*( n*3 / (n/resources.len() as u32)) as u128*/);//3: Loader -> ApplicableA -> ApplicableB

        assert_eq!(my_loader.futures_runner.statistics.copy_started_count.load(Ordering::Relaxed), 5);

    }

    #[test]
    fn test_format_id(){

        let resource_id = "aaa".to_string();

        let loader = TestLoader::new(Devices::Cpu, 1);

        assert_eq!("aaa", loader.applied_id(&resource_id));

        let applicable_a = ApplicableA::new("0".to_string(), Devices::Cuda(0));
        assert_eq!("aaa:ApplicableA_0", applicable_a.applied_id(&resource_id));

        let applicable_a = ApplicableA::new("1".to_string(), Devices::Cuda(0));
        assert_eq!("aaa:ApplicableA_1", applicable_a.applied_id(&resource_id));

        let applicable_b = ApplicableB::new("0".to_string(), "0".to_string(), Devices::Cuda(0));

        assert_eq!("aaa:ApplicableA_0:ApplicableB_0", applicable_b.applied_id(&resource_id));

        let applicable_b = ApplicableB::new("0".to_string(), "1".to_string(), Devices::Cuda(0));

        assert_eq!("aaa:ApplicableA_1:ApplicableB_0", applicable_b.applied_id(&resource_id));

        debug!("applicable_b={:?}", applicable_b.applied_id(&resource_id));
    }

    #[test]
    fn test_purge_cache(){

        let t_0 = Instant::now();

        let my_runner = Arc::new(MyRunner::new());

        let main_pool = main_pool(10);

        // let wg = WaitGroup::new();
        // let wg_pool = wg.clone();

        main_pool.scope(|scope| {
            let ctx_0 = my_runner.next_ctx();
            debug!("ctx_1={ctx_0:?}");
            let ctx_1 = my_runner.next_ctx();
            debug!("ctx_2={ctx_1:?}");

            main_pool.scope(|scope| {
                let ctx_0 = Arc::clone(&ctx_0);
                let futures_runner_0 = Arc::clone(&my_runner);
                // scope.spawn(|_|{
                let resources = block_on(scope.spawn_future(async move {
                    join_all([
                        futures_runner_0.run_applicable_b(&"xxx".into(), &Arc::new(ApplicableB::new("1".to_string(), "0".to_string(), Devices::Cuda(0))), Some(ctx_0.clone())),
                        futures_runner_0.run_applicable_b(&"aaa".into(), &Arc::new(ApplicableB::new("1".to_string(), "0".to_string(), Devices::Cuda(0))), Some(ctx_0.clone())),
                        futures_runner_0.run_applicable_b(&"bbb".into(), &Arc::new(ApplicableB::new("1".to_string(), "0".to_string(), Devices::Cuda(0))), Some(ctx_0.clone()))
                    ]).await
                }));

                assert_eq!(resources[0].0.device, Devices::Cuda(0));
                assert_eq!(resources[1].0.device, Devices::Cuda(0));
                assert_eq!(resources[2].0.device, Devices::Cuda(0));

                debug!("0. resources={resources:?}");
                // });

                let ctx_1 = Arc::clone(&ctx_1);
                let futures_runner_1 = Arc::clone(&my_runner);
                // scope.spawn(|_|{
                let roesources = block_on(scope.spawn_future(async move {
                    join_all([
                        futures_runner_1.run_applicable_b(&"xxx".into(), &Arc::new(ApplicableB::new("1".to_string(), "0".to_string(), Devices::Cuda(0))), Some(ctx_1.clone())),
                        futures_runner_1.run_applicable_b(&"ccc".into(), &Arc::new(ApplicableB::new("1".to_string(), "0".to_string(), Devices::Cuda(0))), Some(ctx_1.clone())),
                        futures_runner_1.run_applicable_b(&"ddd".into(), &Arc::new(ApplicableB::new("1".to_string(), "0".to_string(), Devices::Cuda(0))), Some(ctx_1.clone()))
                    ]).await
                }));

                assert_eq!(resources[0].0.device, Devices::Cuda(0));
                assert_eq!(resources[1].0.device, Devices::Cuda(0));
                assert_eq!(resources[2].0.device, Devices::Cuda(0));

                debug!("1. resources={resources:?}");
                // });

            });

            debug!("loader_cache_fut:");
            debug!("aaa cache_areas={:?}", my_runner.loader_cache_fut.get_cache_areas(&"aaa".to_string()));
            debug!("ccc cache_areas={:?}", my_runner.loader_cache_fut.get_cache_areas(&"ccc".to_string()));
            debug!("xxx cache_areas={:?}", my_runner.loader_cache_fut.get_cache_areas(&"xxx".to_string()));
            assert_eq!(*my_runner.loader_cache_fut.get_cache_areas(&"aaa".to_string()), [0, 1, 0]);//loaded into area 0
            assert_eq!(*my_runner.loader_cache_fut.get_cache_areas(&"ccc".to_string()), [0, 2, 0]);//loaded into area 1
            assert_eq!(*my_runner.loader_cache_fut.get_cache_areas(&"xxx".to_string()), [0, 1, 0]);//loaded into area 0, not used for are 1, xxx:ApplicableA_0:ApplicableB_1 is taken form cache

            debug!("applicable_a_cache_fut:");//loaded into area 0
            debug!("aaa cache_areas={:?}", my_runner.applicable_a_cache_fut.get_cache_areas(&"aaa:ApplicableA_0".to_string()));
            debug!("ccc cache_areas={:?}", my_runner.applicable_a_cache_fut.get_cache_areas(&"ccc:ApplicableA_0".to_string()));
            debug!("xxx cache_areas={:?}", my_runner.applicable_a_cache_fut.get_cache_areas(&"xxx:ApplicableA_0".to_string()));
            assert_eq!(*my_runner.applicable_a_cache_fut.get_cache_areas(&"aaa:ApplicableA_0".to_string()), [0, 1, 0]);
            assert_eq!(*my_runner.applicable_a_cache_fut.get_cache_areas(&"ccc:ApplicableA_0".to_string()), [0, 2, 0]);
            assert_eq!(*my_runner.applicable_a_cache_fut.get_cache_areas(&"xxx:ApplicableA_0".to_string()), [0, 1, 0]);

            debug!("applicable_b_cache_fut:");
            debug!("aaa cache_areas={:?}", my_runner.applicable_b_cache_fut.get_cache_areas(&"aaa:ApplicableA_0:ApplicableB_1".to_string()));
            debug!("ccc cache_areas={:?}", my_runner.applicable_b_cache_fut.get_cache_areas(&"ccc:ApplicableA_0:ApplicableB_1".to_string()));
            debug!("xxx cache_areas={:?}", my_runner.applicable_b_cache_fut.get_cache_areas(&"xxx:ApplicableA_0:ApplicableB_1".to_string()));
            assert_eq!(*my_runner.applicable_b_cache_fut.get_cache_areas(&"aaa:ApplicableA_0:ApplicableB_1".to_string()), [0, 1, 0]);
            assert_eq!(*my_runner.applicable_b_cache_fut.get_cache_areas(&"ccc:ApplicableA_0:ApplicableB_1".to_string()), [0, 2, 0]);
            assert_eq!(*my_runner.applicable_b_cache_fut.get_cache_areas(&"xxx:ApplicableA_0:ApplicableB_1".to_string()), [0, 3, 0]);

            debug!("------------------");
            // debug!("my_loader: {:?}", my_loader);

            main_pool.scope( |scope| {
                block_on(scope.spawn_future(my_runner.clone().purge_all_gpu_caches(Arc::into_inner(ctx_0).unwrap().into_inner(), Devices::Cuda(0))));
            });

            debug!("loader_cache_fut:");
            debug!("aaa cache_areas={:?}", my_runner.loader_cache_fut.get_cache_areas(&"aaa".to_string()));
            debug!("ccc cache_areas={:?}", my_runner.loader_cache_fut.get_cache_areas(&"ccc".to_string()));
            debug!("xxx cache_areas={:?}", my_runner.loader_cache_fut.get_cache_areas(&"xxx".to_string()));

            debug!("applicable_a_cache_fut:");
            debug!("aaa cache_areas={:?}", my_runner.applicable_a_cache_fut.get_cache_areas(&"aaa:ApplicableA_0".to_string()));
            debug!("ccc cache_areas={:?}", my_runner.applicable_a_cache_fut.get_cache_areas(&"ccc:ApplicableA_0".to_string()));
            debug!("xxx cache_areas={:?}", my_runner.applicable_a_cache_fut.get_cache_areas(&"xxx:ApplicableA_0".to_string()));

            debug!("applicable_b_cache_fut:");
            debug!("aaa cache_areas={:?}", my_runner.applicable_b_cache_fut.get_cache_areas(&"aaa:ApplicableA_0:ApplicableB_1".to_string()));
            debug!("ccc cache_areas={:?}", my_runner.applicable_b_cache_fut.get_cache_areas(&"ccc:ApplicableA_0:ApplicableB_1".to_string()));
            debug!("xxx cache_areas={:?}", my_runner.applicable_b_cache_fut.get_cache_areas(&"xxx:ApplicableA_0:ApplicableB_1".to_string()));
            debug!("------------------");
            // debug!("futures_runner: {:?}", futures_runner);

            main_pool.scope(|scope| {
                block_on(scope.spawn_future(my_runner.clone().purge_all_gpu_caches(Arc::into_inner(ctx_1).unwrap().into_inner(), Devices::Cuda(0))));
            });

            // debug!("futures_runner: {:?}", futures_runner);

        });

        // debug!("futures_runner: {:?}", futures_runner);
        // debug!("statistics: {:?}", my_loader.statistics);
        debug!("loader_cache_fut:");
        debug!("aaa cache_areas={:?}", my_runner.loader_cache_fut.get_cache_areas("aaa"));
        debug!("ccc cache_areas={:?}", my_runner.loader_cache_fut.get_cache_areas("ccc"));
        debug!("xxx cache_areas={:?}", my_runner.loader_cache_fut.get_cache_areas("xxx"));
        assert_eq!(*my_runner.loader_cache_fut.get_cache_areas("aaa"), [1, 0, 0]);
        assert_eq!(*my_runner.loader_cache_fut.get_cache_areas("ccc"), [2, 0, 0]);
        assert_eq!(*my_runner.loader_cache_fut.get_cache_areas("xxx"), [1, 0, 0]);


        debug!("applicable_a_cache_fut:");
        debug!("aaa cache_areas={:?}", my_runner.applicable_a_cache_fut.get_cache_areas("aaa:ApplicableA_0"));
        debug!("ccc cache_areas={:?}", my_runner.applicable_a_cache_fut.get_cache_areas("ccc:ApplicableA_0"));
        debug!("xxx cache_areas={:?}", my_runner.applicable_a_cache_fut.get_cache_areas("xxx:ApplicableA_0"));
        assert_eq!(*my_runner.applicable_a_cache_fut.get_cache_areas("aaa:ApplicableA_0"), [1, 0, 0]);
        assert_eq!(*my_runner.applicable_a_cache_fut.get_cache_areas("ccc:ApplicableA_0"), [2, 0, 0]);
        assert_eq!(*my_runner.applicable_a_cache_fut.get_cache_areas("xxx:ApplicableA_0"), [1, 0, 0]);


        debug!("applicable_b_cache_fut:");
        debug!("aaa cache_areas={:?}", my_runner.applicable_b_cache_fut.get_cache_areas("aaa:ApplicableA_0:ApplicableB_1"));
        debug!("ccc cache_areas={:?}", my_runner.applicable_b_cache_fut.get_cache_areas("ccc:ApplicableA_0:ApplicableB_1"));
        debug!("xxx cache_areas={:?}", my_runner.applicable_b_cache_fut.get_cache_areas("xxx:ApplicableA_0:ApplicableB_1"));
        assert_eq!(*my_runner.applicable_b_cache_fut.get_cache_areas("aaa:ApplicableA_0:ApplicableB_1"), [1, 0, 0]);
        assert_eq!(*my_runner.applicable_b_cache_fut.get_cache_areas("ccc:ApplicableA_0:ApplicableB_1"), [2, 0, 0]);
        assert_eq!(*my_runner.applicable_b_cache_fut.get_cache_areas("xxx:ApplicableA_0:ApplicableB_1"), [2, 0, 0]);


        std::thread::sleep(Duration::from_millis(100));
        my_runner.futures_runner.purge_all_cpu_caches(&(Instant::now()-Duration::from_millis(100)), my_runner.to_futs_iter());

        debug!("statistics={:?}", my_runner.futures_runner.statistics);
        assert_eq!(my_runner.futures_runner.statistics.purge_cpu_count.load(Ordering::Relaxed), 15);//(aaa, bbb, ccc, ddd, xxx) x (loader, applicable_a, applicable_b)
    }

    fn main_pool(num_threads: usize) -> ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .spawn_handler(|thread_builder| {
                let b = std::thread::Builder::new();

                b.name(format!("{}/{:?}", "main_pool", thread_builder.index()))
                    .spawn(|| thread_builder.run())?;

                Ok(())
            })
            .build()
            .expect("main pool")
    }

    #[test]
    fn test_remove_mask_bug() {
        // keep lower bit, clear upper bit
        let v   = 0b11u64;
        let sub = 0b10u64;
        assert_eq!(0b01, super::remove_mask(v, sub));     // ← fails: returns 0
    }

    #[test]
    fn test_run_async_fn(){
        let my_runner = Arc::new(MyRunner::new());
        let input = (0..10).map(|i| i).collect::<Vec<_>>();

        let sum = block_on(my_runner.futures_runner.run_async_fn((input,),async move |(input,):(Vec<i32>,)|{
            debug!("input={input:?}");
            input.par_iter().map(|i| {
                debug!("input={input:?} i={i} th name={:?}", thread::current().name());
                i*10
            }).sum::<i32>()

        }, &Cpu));

        debug!("sum={sum:?}");
        assert_eq!(sum, 450);
    }

    #[test]
    fn test_run_fn(){
        let my_runner = Arc::new(MyRunner::new());
        let input = (0..10).map(|i| i).collect::<Vec<_>>();

        let sum = my_runner.futures_runner.run_fn((input,), move |(input,):(Vec<i32>,)|{
            debug!("input={input:?}");
            input.par_iter().map(|i| {
                debug!("input={input:?} i={i} th name={:?}", thread::current().name());
                i*10
            }).sum::<i32>()
        }, &Cpu);

        debug!("sum={sum:?}");
        assert_eq!(sum, 450);
    }
}
