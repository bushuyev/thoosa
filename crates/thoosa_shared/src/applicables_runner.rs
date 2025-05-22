use crate::rayon_future::{RayonFuture, ScopeFutureExt};
use async_trait::async_trait;
use dashmap::{DashMap, Entry, Map};
use futures::channel::oneshot;
use futures::future::{BoxFuture, Shared};
use futures::FutureExt;
use futures_util::future::join_all;
use parking_lot::RwLock;
use rayon::ThreadPool;
use std::borrow::{Borrow, BorrowMut, Cow};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::io::Write;
use std::ops::{Deref, DerefMut, Sub};
use std::sync::Arc;
use std::time::Instant;
use log::debug;
use crate::resource::Resource;
use crate::run_configs::Applicable;

pub type AppliedId = String;

pub trait Input:Send+Sync+Debug {
    fn id(&self) -> &AppliedId;
}



#[async_trait]
pub trait Output:Send+Sync+Debug {
    async fn copy_to_device(self:Arc<Self>, device_ordinal:usize) -> Arc<Self>;
}


impl Input for AppliedId {
    fn id(&self) -> &AppliedId {
        self
    }
}


pub trait ToOrdinal:Debug + Sync + Send {
    fn ordinal(&self) -> usize;
}


pub trait Runner {
    type Ctx:RunCtx;
}


// #[async_trait]
// pub trait Applicable<In, Out>: Sync + Send
//     where  In: Input + Send + Sync,
//             Out: Output + Send + Sync
// {
//     type R: Runner + Sync + Send;
// 
//     async fn apply(
//         self:Arc<Self>,
//         input: Arc<In>,
//         runner: Arc<Self::R>,
//         ctx: Option<Arc<RwLock<<<Self as Applicable<In, Out>>::R as Runner>::Ctx>>>
//     ) -> Arc<Out>;
// 
//     
// 
//     fn format_id(&self, input_id:&AppliedId) -> AppliedId {
//         format!("{}:{}_{}", input_id, self.name(), self.run_id())
//     }
// 
//     fn format_0(prev_id:&str) -> AppliedId {
//         format!("{}", prev_id)
//     }
// 
//     fn applied_id(&self, input_id:&AppliedId) -> AppliedId;
//     fn name(&self)->String;
//     fn run_id(&self)->String;
// 
//     fn device(&self) -> &impl ToOrdinal;
// }


impl  Debug for FuturesRunner {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FuturesRunner")
            // .field("loader_cache", &self.loader_cache_fut)
            // .field("applicable_a_cache", &self.applicable_a_cache_fut)
            // .field("applicable_b_cache", &self.applicable_b_cache_fut)
            .finish_non_exhaustive()
    }
}

struct DeviceThreadPools {
    cpu_pool: ThreadPool,
    gpu_pools: HashMap<usize, ThreadPool>,
}

impl DeviceThreadPools {

    fn new(cpu_threads: usize, gpu_threads: &[usize]) -> Self {

        let cpu_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cpu_threads)
            .spawn_handler(|thread_builder| {
                debug!("spawning cpu thread index={}", thread_builder.index());

                let b = std::thread::Builder::new();

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

                    let b = std::thread::Builder::new();

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

    fn new<I, A: Applicable<I, O, ()> + 'static> (
        input: Arc<I>,
        device_ordinal: usize,
        num_devices: usize,
        pool: &ThreadPool,
        applicable: Arc<A>,
        runner: Arc<A::R>,
        statistics: &Arc<RwLock<Statistics>>,
        ctx: Option<Arc<RwLock<<<A as Applicable<I, O, ()>>::R as Runner>::Ctx>>>
    )
        -> (Self, Shared<BoxFuture<'static, Arc<O>>>)
    where I: Input+Send+Sync+'static
    {
        let (tx, rx) = oneshot::channel();
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

        pool.install(move ||{//or pool.scope(move |scope| { scope.spawn_future(async move {
            RayonFuture::<()>::install_future(async move {
                let r = applicable.async_apply(input.clone(), runner, ctx_for_future).await;
                // debug!("run_cached_task: done input={input:?}");
                let _ = tx.send(r);
                // statistics_clone.write().on_new_load_done(t_0.elapsed().as_millis());
            });
        });

        let fut = async move { rx.await.expect("Task execution failed") }
            .boxed()
            .shared();


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

    fn get_from_any_device<R:RunCtx>(&mut self, device_ordinal:usize, pool:&ThreadPool, statistics: &Arc<RwLock<Statistics>>, ctx: Option<Arc<RwLock<R>>> ) -> Shared<BoxFuture<'static, Arc<O>>> {
        // debug!("get_from_any_device: device_ordinal={device_ordinal} ctx={ctx:?} self.device_cache_areas[device_ordinal]={:?}", self.device_cache_areas[device_ordinal]);

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

                self.make_device_copy_future(from_device, device_ordinal, pool, statistics)
            }
        }
    }

    fn make_device_copy_future(
        &mut self,
        from_device: usize,
        to_device: usize,
        pool: &ThreadPool,
        statistics: &Arc<RwLock<Statistics>>,
    ) -> Shared<BoxFuture<'static, Arc<O>>> {

        let data_fut = self.device_futures[from_device]
            .as_ref()
            .expect("source device future missing")
            .clone();

        let (tx, rx) = oneshot::channel();
        let statistics_clone = statistics.clone();
        let t_0 = Instant::now();


        pool.install(move || { //or pool.scope(move |scope| { scope.spawn_future(async move {
            RayonFuture::<()>::install_future(async move {

                let from_data = data_fut.await;

                let _ = tx.send(from_data.copy_to_device(to_device).await);

                // statistics_clone
                //     .write()
                    // .on_copy_done(t_0.elapsed().as_millis());
            });
        });

        let fut = async move { rx.await.expect("Task execution failed") }
            .boxed()
            .shared();

        self.device_futures[to_device] = Some(fut.clone());
        self.device_cache_areas[to_device] |= self.device_cache_areas[from_device];
        fut
    }

}

#[derive(Debug)]
pub struct Statistics {

    load_started_count:u64,
    load_ended_count:u64,
    pure_run_time: u128,
    load_time: u128,

    copy_started_count:u64,
    copy_ended_count:u64,
    copy_time: u128,

    loaded_keys:HashSet<AppliedId>,
    copied_keys:HashSet<AppliedId>,

    purge_cpu_count:u64,
    purge_cpu_time: u128,
}


impl Statistics {
    pub fn new() -> Self {
        Self {
            load_started_count:0,
            load_ended_count:0,
            pure_run_time: 0,
            load_time: 0,
            copy_started_count: 0,
            copy_ended_count: 0,
            copy_time: 0,
            loaded_keys: Default::default(),
            copied_keys: Default::default(),
            purge_cpu_count: 0,
            purge_cpu_time: 0,
        }
    }

    pub fn load_time(&self) -> u128 {
        self.load_time
    }

    pub fn on_new_load(&mut self, input: &AppliedId) {
        self.load_started_count +=1;
        self.loaded_keys.insert(input.clone());
    }

    pub fn on_new_load_done(&mut self, load_time:u128) {
        self.load_ended_count +=1;
        self.load_time += load_time;
    }

    pub fn track_pure_run_time(&mut self, pure_run_time:u128) {
        self.pure_run_time += pure_run_time;

    }

    pub fn copy_time(&self) -> u128 {
        self.copy_time
    }

    pub fn on_copy(&mut self, input: &AppliedId) {
        self.copy_started_count +=1;
        self.copied_keys.insert(input.clone());
    }

    pub fn on_copy_done(&mut self, copy_time:u128) {
        self.copy_time += copy_time;
        self.copy_ended_count +=1;
    }

    pub fn on_purge_cpu_caches(&mut self, purge_cpu_caches_time:u128, purged_count:u64) {
        self.purge_cpu_count += purged_count;
        self.purge_cpu_time += purge_cpu_caches_time;
    }
}

#[async_trait]
pub trait FuturesRunnerMap:Sync+Send{
    async fn purge_gpu_cache_area(&self, key: AppliedId, cache_area:u64, pool:&ThreadPool, device:&dyn ToOrdinal, statistics: Arc<RwLock<Statistics>>);
    fn purge_cpu_caches(&self, statistics: &Arc<RwLock<Statistics>>, older_than:&Instant);

}
#[async_trait]
impl<O> FuturesRunnerMap for DashMap<AppliedId, DataInDevices<O>>
where
    O: Output + Send + Sync + 'static,
{
    async fn purge_gpu_cache_area(
        &self,
        key: AppliedId,
        cache_area: u64,
        pool: &ThreadPool,
        device:&dyn ToOrdinal,
        statistics: Arc<RwLock<Statistics>>,
    ) {

        debug!("purge_gpu_cache_area: key={key:?} cache_area={cache_area:?} device={device:?}");
        if let Some(mut entry) = self.get_mut(&key) {
            let device_idx = device.ordinal();

            let new_mask = remove_mask(entry.device_cache_areas[device_idx], cache_area);
            if device_idx != 0 && new_mask == 0 {//device_idx == 0 it's on CPU and will be purged by purge_cpu_caches

                if entry.device_futures[0].is_none() {
                    entry
                        .make_device_copy_future(device_idx, 0, pool, &statistics)
                        .await;
                }

                entry.device_futures[device_idx] = None;//TODO Cancel if pending?
                entry.in_cpu_since = Some(Instant::now());
            }
            entry.device_cache_areas[device_idx] = new_mask;
        }
    }

    fn purge_cpu_caches(&self, statistics: &Arc<RwLock<Statistics>>, older_than: &Instant) {
        let now = Instant::now();

        let mut purged_count = 0u64;

        self.retain(|key, v| {
            let to_retain = v.in_cpu_since.map(|v| &v > older_than).unwrap_or(true);

            debug!("purge_cpu_caches: key={key:?} to_retain={to_retain:?} v.in_cpu_since={:?}", v.in_cpu_since);

            if !to_retain {
                purged_count+=1;
            }
            to_retain
        });

        // statistics.write().on_purge_cpu_caches(now.elapsed().as_millis(), purged_count);
    }
}

pub struct FuturesRunner {
    num_devices: usize,
    device_pools: DeviceThreadPools,

    caches_areas: RwLock<CacheAreasKeeper>,
    statistics: Arc<RwLock<Statistics>>
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
    fn into_cached_iter_mut(self) -> impl Iterator<Item=Vec<AppliedId>>;
}

impl FuturesRunner {

    pub fn new(devices_threads:&[usize]) -> Self {

        Self {
            device_pools: DeviceThreadPools::new(devices_threads[0], &devices_threads[1..]),
            num_devices: devices_threads.len(),


            caches_areas: RwLock::new(CacheAreasKeeper::new()),
            statistics: Arc::new(RwLock::new(Statistics::new()))
        }
    }

    pub fn run_task<I, O, A: Applicable<I, O, ()> + 'static>(
        self: &Arc<Self>,
        input: Arc<I>,
        applicable:&Arc<A>,
        runner:&Arc<A::R>,
    ) -> BoxFuture<'static, Arc<O>>
    where
        O: Output + Send + Sync + 'static,
        I: Input+Send+Sync+'static,
    {
        let device = applicable.device();
        // let id = applicable.applied_id(input.id());

        debug!("run_task");

        let (tx, rx) = oneshot::channel();
        // let statistics_clone = statistics.clone();

        let t_0 = Instant::now();
        let device = applicable.device();
        let pool = self.device_pools.get_pool(&device);
        let applicable= Arc::clone(applicable);
        let runner= Arc::clone(runner);

        pool.install(move ||{//or pool.scope(move |scope| { scope.spawn_future(async move {
            RayonFuture::<()>::install_future(async move {
                let r = applicable.async_apply(input.clone(), runner, None).await;
                // debug!("run_cached_task: done input={input:?}");
                let _ = tx.send(r);
                // statistics_clone.write().on_new_load_done(t_0.elapsed().as_millis());
            });
        });

        async move { rx.await.expect("Task execution failed") }
            .boxed()

    }


    pub fn run_cached_task<I, O, A: Applicable<I, O, ()> + 'static>(
        self: &Arc<Self>,
        cache: &DashMap<AppliedId, DataInDevices<O>>,
        input: Arc<I>,
        applicable:&Arc<A>,
        runner:&Arc<A::R>,
        ctx: Option<Arc<RwLock<<<A as Applicable<I, O, ()>>::R as Runner>::Ctx>>>
    ) -> Shared<BoxFuture<'static, Arc<O>>>
    where
        O: Output + Send + Sync + 'static,
        I: Input+Send+Sync+'static,
    {
        let device = applicable.device();
        let id = applicable.applied_id(input.id());

        debug!("run_cached_task: id={id:?}");

        match cache.entry(id) {
            Entry::Occupied(mut e) => {
                // self.statistics.write().on_copy(input.id());

                let f = e.get_mut().get_from_any_device(
                    device.ordinal(),
                    self.device_pools.get_pool(&device),
                    &self.statistics,
                    ctx
                ).clone();

                f
            },
            Entry::Vacant(v) => {
                // self.statistics.write().on_new_load(input.id());

                let (device_copy_future, fut) = DataInDevices::new(
                    input,
                    device.ordinal(),
                    self.num_devices,
                    self.device_pools.get_pool(&device),
                    Arc::clone(applicable),
                    Arc::clone(&runner),
                    &self.statistics,
                    ctx
                );
                v.insert(device_copy_future);
                fut
            }
        }
    }


    pub async fn purge_all_gpu_caches<C:RunCtx>(
        self: Arc<Self>,
        ctx: C,
        device:&dyn ToOrdinal,
        maps_iter: impl Iterator<Item=&dyn FuturesRunnerMap> + Sized
    ) {
        let pool  = self.device_pools.get_pool(device);
        let cache_area = ctx.cache_area();

        join_all(ctx
            .into_cached_iter_mut()
            .zip(maps_iter)
            .flat_map(|(ctx_keys, mut futs_map)|{
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

    pub fn statistics(self:&Arc<Self>) -> impl Deref<Target = Statistics> + '_ {
        self.statistics.read()
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
pub async fn dummy_load(statistics: &RwLock<Statistics>, input: &Arc<AppliedId>) {


    for i in 0..DUMMY_LOAD_N {
        let i = 0;
        async {

            let t_0 = Instant::now();

            let a = vec![0.1; 1_000_000];
            let b = vec![0.1; 1_000_000];
            let c = a.into_iter().zip(b.into_iter()).map(|(a, b)| a + b).collect::<Vec<f32>>();

            statistics.write().track_pure_run_time(t_0.elapsed().as_millis());

        }.await;

        debug!("dummy_load i={i} thread={:?}", std::thread::current().name());
    }

}

const DUMMY_LOAD_N:u32 = 3;

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use super::{dummy_load, Applicable, DataInDevices, FuturesRunner, FuturesRunnerMap, Input, Output, AppliedId, RunCtx, Runner, ToOrdinal};
    use crate::rayon_future::ScopeFutureExt;
    use async_trait::async_trait;
    use dashmap::DashMap;
    use futures::executor::block_on;
    use futures_util::future::{join_all, BoxFuture, Shared};
    use parking_lot::RwLock;
    use rayon::iter::ParallelIterator;
    use rayon::prelude::ParallelBridge;
    use rayon::ThreadPool;
    use std::sync::atomic::AtomicU32;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use log::debug;

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
        id: AppliedId,
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
        fn id(&self) -> &AppliedId {
            &self.id
        }

    }


    #[derive(Debug)]
    pub struct MyRunCtx {
        cache_area: u64,

        loaded_cached: Vec<AppliedId>,
        applicable_a_cached: Vec<AppliedId>,
        applicable_b_cached: Vec<AppliedId>,
    }

    impl MyRunCtx {
        pub fn new(cache_area: u64) -> Self {
            Self { cache_area, loaded_cached: vec![], applicable_a_cached: vec![], applicable_b_cached: vec![] }
        }

    }

    impl RunCtx for MyRunCtx {
        fn into_cached_iter_mut(self) -> impl Iterator<Item=Vec<AppliedId>> {
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
        device: Devices
    }

    impl TestLoader {

        fn new(device: Devices) -> TestLoader {
            Self { device }
        }

    }

    #[derive(Debug, Hash, Eq, PartialEq, Clone)]
    struct LoaderInputId {
        id:String,
        import_id:String,
        applied_id:AppliedId
    }

    impl LoaderInputId {
        pub fn new(id: String, import_id: String) -> Self {
            let applied_id = format!("{}{}", id, import_id);
            Self { id, import_id,  applied_id}
        }
    }

    impl Input for LoaderInputId {
        fn id(&self) -> &AppliedId {
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
    impl Applicable<LoaderInputId, LoaderOutput, ()> for TestLoader {
        type R = MyRunner;

        async fn async_apply(self: Arc<Self>, input: Arc<LoaderInputId>, runner: Arc<MyRunner>, ctx: Option<Arc<RwLock<MyRunCtx>>>) -> Arc<LoaderOutput> {
            // debug!("Loader::apply_2: input={input:?}");
            // thread::sleep(Duration::new(0, 1000));
            dummy_load(&runner.futures_runner.statistics, &Arc::new(input.id().clone())).await;

            Arc::new(LoaderOutput(DataResource::new(input.id().clone(), self.device, vec![0.1, 0.2, 0.3, 0.4])))
        }

        fn applied_id(&self, prev_id: &AppliedId) -> String {
            Self::format_0(&prev_id)
        }

        fn name(&self) -> String {
            "".to_string()
        }

        fn run_id(&self) -> String {
            "".to_string()
        }

        fn device(&self) -> impl ToOrdinal {
            self.device.clone()
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
                loader: Arc::new(TestLoader::new(device)),
                device
            }
        }
    }


    impl Input for LoaderOutput {
        fn id(&self) -> &AppliedId {
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
    impl Applicable<LoaderInputId, ApplicableAOutput, ()> for ApplicableA {
        type R = MyRunner;

        async fn async_apply(self: Arc<Self>, input: Arc<LoaderInputId>, runner: Arc<MyRunner>, ctx: Option<Arc<RwLock<MyRunCtx>>>) -> Arc<ApplicableAOutput> {
            let resource_0 = runner.clone().run_loader(&input, &self.loader, ctx).await;
            // debug!("ApplicableA::apply: input={input:?}");

            dummy_load(&runner.futures_runner.statistics, &Arc::new(input.id().clone())).await;


            Arc::new(ApplicableAOutput(DataResource::new(self.applied_id(input.id()), self.device, [vec![0.1, 0.2, 0.3, 0.4], resource_0.0.data.clone()].concat())))

        }

        fn applied_id(&self, prev_id: &AppliedId) -> String {
            Applicable::format_id(self, &self.loader.applied_id(prev_id))
        }


        fn name(&self) -> String {
            "ApplicableA".to_string()
        }

        fn run_id(&self) -> String {
            self.run_id.clone()
        }

        fn device(&self) -> impl ToOrdinal {
            self.device.clone()
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
        fn id(&self) -> &AppliedId {
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
    impl Applicable<LoaderInputId, ApplicableBOutput, ()> for ApplicableB {
        type R = MyRunner;

        async fn  async_apply(self: Arc<Self>, input: Arc<LoaderInputId>, runner: Arc<MyRunner>, ctx: Option<Arc<RwLock<MyRunCtx>>>) -> Arc<ApplicableBOutput> {
            let resource_a = runner.clone().run_applicable_a(&input, &self.applicable_a, ctx).await;
            debug!("ApplicableB::apply: input={input:?}");

            dummy_load(&runner.futures_runner.statistics, &Arc::new(input.id().clone())).await;


            Arc::new(ApplicableBOutput(DataResource::new(self.applied_id(input.id()), self.device, [vec![0.1, 0.2, 0.3, 0.4], resource_a.0.data.clone()].concat())))
        }

        fn applied_id(&self, input_id: &AppliedId) -> String {
            Applicable::format_id(self, &self.applicable_a.applied_id(input_id))
        }

        fn name(&self) -> String {
            "ApplicableB".to_string()
        }

        fn run_id(&self) -> String {
            self.run_id.clone()
        }

        fn device(&self) -> impl ToOrdinal {
            self.device.clone()
        }
    }


    #[derive(Debug)]
    struct MyRunner {
        futures_runner:Arc<FuturesRunner>,

        loader_cache_fut: DashMap<AppliedId, DataInDevices<LoaderOutput>>,
        applicable_a_cache_fut: DashMap<AppliedId, DataInDevices<ApplicableAOutput>>,
        applicable_b_cache_fut: DashMap<AppliedId, DataInDevices<ApplicableBOutput>>,
    }

    impl Runner for MyRunner {
        type Ctx = MyRunCtx;
    }

    impl MyRunner {
        pub fn new() -> Self {
            Self {
                futures_runner: Arc::new(FuturesRunner::new(&[10, 2, 2])),

                loader_cache_fut: Default::default(),
                applicable_a_cache_fut: Default::default(),
                applicable_b_cache_fut: Default::default(),
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
                ctx
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
                ctx
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

        let res = block_on(my_loader.clone().run_loader(&LoaderInputId::new("0".to_string(), "1".to_string()), &Arc::new(TestLoader::new(Devices::Cpu)), None));

        debug!("loader_cache keys:{:?}", my_loader.loader_cache_fut.iter().map(|v|v.key().clone()).collect::<Vec<String>>());
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

        debug!("loader_cache keys:{:?}", my_loader.loader_cache_fut.iter().map(|v|v.key().clone()).collect::<Vec<String>>());
        debug!("applicable_a_cache keys:{:?}", my_loader.applicable_a_cache_fut.iter().map(|v|v.key().clone()).collect::<Vec<String>>());
        debug!("got res={res:?} in {:?}", t_0.elapsed());
    }

    #[test]
    fn test_applicable_b() {
        let t_0 = Instant::now();

        let my_loader = Arc::new(MyRunner::new());

        let res = block_on(
            my_loader.clone().run_applicable_b(&LoaderInputId::new("0".to_string(), "1".to_string()), &Arc::new(ApplicableB::new("0".to_string(), "0".to_string(), Devices::Cuda(0))), None)
        );


        debug!("loader_cache keys:{:?}", my_loader.loader_cache_fut.iter().map(|v|v.key().clone()).collect::<Vec<String>>());
        debug!("applicable_a_cache keys:{:?}", my_loader.applicable_a_cache_fut.iter().map(|v|v.key().clone()).collect::<Vec<String>>());
        debug!("applicable_b_cache keys:{:?}", my_loader.applicable_b_cache_fut.iter().map(|v|v.key().clone()).collect::<Vec<String>>());
        // debug!("got res={res:?} in {:?}", t_0.elapsed());


        let res = block_on(
            my_loader.clone().run_applicable_b(&LoaderInputId::new("0".to_string(), "1".to_string()), &Arc::new(ApplicableB::new("1".to_string(), "0".to_string(), Devices::Cuda(1))), None)
        );

        debug!("loader_cache keys:{:?}", my_loader.loader_cache_fut.iter().map(|v|v.key().clone()).collect::<Vec<String>>());
        debug!("applicable_a_cache keys:{:?}", my_loader.applicable_a_cache_fut.iter().map(|v|v.key().clone()).collect::<Vec<String>>());
        debug!("applicable_b_cache keys:{:?}", my_loader.applicable_b_cache_fut.iter().map(|v|v.key().clone()).collect::<Vec<String>>());
        debug!("got res={res:?} in {:?}", t_0.elapsed());
    }

    #[test]
    fn test_load(){

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

                    let res = futures::executor::block_on(
                        Arc::clone(&my_loader).run_applicable_b(&LoaderInputId::new(format!("{resource}_{i}"), "1".to_string()), &Arc::new(ApplicableB::new("1".to_string(), "0".to_string(), device)), None)
                    );

                    // debug!("resource={resource:?}, device={device:?} res={res:?} done");

                    count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                    // debug!("resource={resource:?}, device={device:?} res={res:?}");
                });

        });

        debug!("futures_runner={my_loader:?}");
        debug!("done in {:?} futures_runner.statistics={:?}", t_0.elapsed().as_millis(), my_loader.futures_runner.statistics.read());
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), n);
        assert_eq!(
            my_loader.futures_runner.statistics.read().load_started_count,
            my_loader.futures_runner.statistics.read().load_ended_count
        );

        assert_eq!(my_loader.futures_runner.statistics.read().load_started_count, 30/*(n*3) as u128*/);//3: Loader -> ApplicableA -> ApplicableB

        assert_eq!(my_loader.futures_runner.statistics.read().copy_started_count, 0);

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

                    // debug!("resource={resource:?}, device={device:?} res={res:?} done");

                    count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

                    // debug!("resource={resource:?}, device={device:?} res={res:?}");
                });

        });

        debug!("done in {:?} futures_runner.statistics={:?}", t_0.elapsed().as_millis(), my_loader.futures_runner.statistics.read());
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), n);
        assert_eq!(
            my_loader.futures_runner.statistics.read().load_started_count,
            my_loader.futures_runner.statistics.read().load_ended_count
        );

        assert_eq!(my_loader.futures_runner.statistics.read().load_started_count, 15/*( n*3 / (n/resources.len() as u32)) as u128*/);//3: Loader -> ApplicableA -> ApplicableB

        assert_eq!(my_loader.futures_runner.statistics.read().copy_started_count, 5);

    }

    #[test]
    fn test_format_id(){

        let resource_id = "aaa".to_string();

        let loader = TestLoader::new(Devices::Cpu);

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

        let my_loader = Arc::new(MyRunner::new());

        let main_pool = main_pool(10);


        main_pool.scope(|scope| {
            let ctx_0 = my_loader.next_ctx();
            debug!("ctx_1={ctx_0:?}");
            let ctx_1 = my_loader.next_ctx();
            debug!("ctx_2={ctx_1:?}");

            main_pool.scope(|scope| {
                let ctx_0 = Arc::clone(&ctx_0);
                let futures_runner_0 = Arc::clone(&my_loader);
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
                let futures_runner_1 = Arc::clone(&my_loader);
                // scope.spawn(|_|{
                let resources = block_on(scope.spawn_future(async move {
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
            debug!("aaa cache_areas={:?}", my_loader.loader_cache_fut.get("aaa").expect("aaa").device_cache_areas);
            debug!("ccc cache_areas={:?}", my_loader.loader_cache_fut.get("ccc").expect("ccc").device_cache_areas);
            debug!("xxx cache_areas={:?}", my_loader.loader_cache_fut.get("xxx").expect("xxx").device_cache_areas);
            assert_eq!(*my_loader.loader_cache_fut.get("aaa").expect("aaa").device_cache_areas, [0, 1, 0]);//loaded into area 0
            assert_eq!(*my_loader.loader_cache_fut.get("ccc").expect("ccc").device_cache_areas, [0, 2, 0]);//loaded into area 1
            assert_eq!(*my_loader.loader_cache_fut.get("xxx").expect("xxx").device_cache_areas, [0, 1, 0]);//loaded into area 0, not used for are 1, xxx:ApplicableA_0:ApplicableB_1 is taken form cache

            debug!("applicable_a_cache_fut:");//loaded into area 0
            debug!("aaa cache_areas={:?}", my_loader.applicable_a_cache_fut.get("aaa:ApplicableA_0").expect("aaa").device_cache_areas);
            debug!("ccc cache_areas={:?}", my_loader.applicable_a_cache_fut.get("ccc:ApplicableA_0").expect("ccc").device_cache_areas);
            debug!("xxx cache_areas={:?}", my_loader.applicable_a_cache_fut.get("xxx:ApplicableA_0").expect("xxx").device_cache_areas);
            assert_eq!(*my_loader.applicable_a_cache_fut.get("aaa:ApplicableA_0").expect("aaa").device_cache_areas, [0, 1, 0]);
            assert_eq!(*my_loader.applicable_a_cache_fut.get("ccc:ApplicableA_0").expect("ccc").device_cache_areas, [0, 2, 0]);
            assert_eq!(*my_loader.applicable_a_cache_fut.get("xxx:ApplicableA_0").expect("xxx").device_cache_areas, [0, 1, 0]);

            debug!("applicable_b_cache_fut:");
            debug!("aaa cache_areas={:?}", my_loader.applicable_b_cache_fut.get("aaa:ApplicableA_0:ApplicableB_1").expect("aaa").device_cache_areas);
            debug!("ccc cache_areas={:?}", my_loader.applicable_b_cache_fut.get("ccc:ApplicableA_0:ApplicableB_1").expect("ccc").device_cache_areas);
            debug!("xxx cache_areas={:?}", my_loader.applicable_b_cache_fut.get("xxx:ApplicableA_0:ApplicableB_1").expect("xxx:ApplicableA_0").device_cache_areas);
            assert_eq!(*my_loader.applicable_b_cache_fut.get("aaa:ApplicableA_0:ApplicableB_1").expect("aaa").device_cache_areas, [0, 1, 0]);
            assert_eq!(*my_loader.applicable_b_cache_fut.get("ccc:ApplicableA_0:ApplicableB_1").expect("ccc").device_cache_areas, [0, 2, 0]);
            assert_eq!(*my_loader.applicable_b_cache_fut.get("xxx:ApplicableA_0:ApplicableB_1").expect("xxx").device_cache_areas, [0, 3, 0]);

            debug!("------------------");
            // debug!("my_loader: {:?}", my_loader);

            main_pool.scope( |scope| {
                block_on(scope.spawn_future(my_loader.clone().purge_all_gpu_caches(Arc::into_inner(ctx_0).unwrap().into_inner(), Devices::Cuda(0))));
            });

            debug!("loader_cache_fut:");
            debug!("aaa cache_areas={:?}", my_loader.loader_cache_fut.get("aaa").expect("aaa").device_cache_areas);
            debug!("ccc cache_areas={:?}", my_loader.loader_cache_fut.get("ccc").expect("ccc").device_cache_areas);
            debug!("xxx cache_areas={:?}", my_loader.loader_cache_fut.get("xxx").expect("xxx").device_cache_areas);

            debug!("applicable_a_cache_fut:");
            debug!("aaa cache_areas={:?}", my_loader.applicable_a_cache_fut.get("aaa:ApplicableA_0").expect("aaa").device_cache_areas);
            debug!("ccc cache_areas={:?}", my_loader.applicable_a_cache_fut.get("ccc:ApplicableA_0").expect("ccc").device_cache_areas);
            debug!("xxx cache_areas={:?}", my_loader.applicable_a_cache_fut.get("xxx:ApplicableA_0").expect("xxx").device_cache_areas);

            debug!("applicable_b_cache_fut:");
            debug!("aaa cache_areas={:?}", my_loader.applicable_b_cache_fut.get("aaa:ApplicableA_0:ApplicableB_1").expect("aaa").device_cache_areas);
            debug!("ccc cache_areas={:?}", my_loader.applicable_b_cache_fut.get("ccc:ApplicableA_0:ApplicableB_1").expect("ccc").device_cache_areas);
            debug!("xxx cache_areas={:?}", my_loader.applicable_b_cache_fut.get("xxx:ApplicableA_0:ApplicableB_1").expect("xxx:ApplicableA_0").device_cache_areas);
            debug!("------------------");
            // debug!("futures_runner: {:?}", futures_runner);

            main_pool.scope(|scope| {
                block_on(scope.spawn_future(my_loader.clone().purge_all_gpu_caches(Arc::into_inner(ctx_1).unwrap().into_inner(), Devices::Cuda(0))));
            });

            // debug!("futures_runner: {:?}", futures_runner);

        });

        // debug!("futures_runner: {:?}", futures_runner);
        // debug!("statistics: {:?}", my_loader.statistics);
        debug!("loader_cache_fut:");
        debug!("aaa cache_areas={:?}", my_loader.loader_cache_fut.get("aaa").expect("aaa").device_cache_areas);
        debug!("ccc cache_areas={:?}", my_loader.loader_cache_fut.get("ccc").expect("ccc").device_cache_areas);
        debug!("xxx cache_areas={:?}", my_loader.loader_cache_fut.get("xxx").expect("xxx").device_cache_areas);
        assert_eq!(*my_loader.loader_cache_fut.get("aaa").expect("aaa").device_cache_areas, [1, 0, 0]);
        assert_eq!(*my_loader.loader_cache_fut.get("ccc").expect("ccc").device_cache_areas, [2, 0, 0]);
        assert_eq!(*my_loader.loader_cache_fut.get("xxx").expect("xxx").device_cache_areas, [1, 0, 0]);


        debug!("applicable_a_cache_fut:");
        debug!("aaa cache_areas={:?}", my_loader.applicable_a_cache_fut.get("aaa:ApplicableA_0").expect("aaa").device_cache_areas);
        debug!("ccc cache_areas={:?}", my_loader.applicable_a_cache_fut.get("ccc:ApplicableA_0").expect("ccc").device_cache_areas);
        debug!("xxx cache_areas={:?}", my_loader.applicable_a_cache_fut.get("xxx:ApplicableA_0").expect("xxx").device_cache_areas);
        assert_eq!(*my_loader.applicable_a_cache_fut.get("aaa:ApplicableA_0").expect("aaa").device_cache_areas, [1, 0, 0]);
        assert_eq!(*my_loader.applicable_a_cache_fut.get("ccc:ApplicableA_0").expect("ccc").device_cache_areas, [2, 0, 0]);
        assert_eq!(*my_loader.applicable_a_cache_fut.get("xxx:ApplicableA_0").expect("xxx").device_cache_areas, [1, 0, 0]);


        debug!("applicable_b_cache_fut:");
        debug!("aaa cache_areas={:?}", my_loader.applicable_b_cache_fut.get("aaa:ApplicableA_0:ApplicableB_1").expect("aaa").device_cache_areas);
        debug!("ccc cache_areas={:?}", my_loader.applicable_b_cache_fut.get("ccc:ApplicableA_0:ApplicableB_1").expect("ccc").device_cache_areas);
        debug!("xxx cache_areas={:?}", my_loader.applicable_b_cache_fut.get("xxx:ApplicableA_0:ApplicableB_1").expect("xxx").device_cache_areas);
        assert_eq!(*my_loader.applicable_b_cache_fut.get("aaa:ApplicableA_0:ApplicableB_1").expect("aaa").device_cache_areas, [1, 0, 0]);
        assert_eq!(*my_loader.applicable_b_cache_fut.get("ccc:ApplicableA_0:ApplicableB_1").expect("aaa").device_cache_areas, [2, 0, 0]);
        assert_eq!(*my_loader.applicable_b_cache_fut.get("xxx:ApplicableA_0:ApplicableB_1").expect("xxx").device_cache_areas, [2, 0, 0]);


        std::thread::sleep(Duration::from_millis(100));
        my_loader.futures_runner.purge_all_cpu_caches(&(Instant::now()-Duration::from_millis(100)), my_loader.to_futs_iter());

        debug!("statistics={:?}", my_loader.futures_runner.statistics);
        assert_eq!(my_loader.futures_runner.statistics.read().purge_cpu_count, 15);//(aaa, bbb, ccc, ddd, xxx) x (loader, applicable_a, applicable_b)
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

}
