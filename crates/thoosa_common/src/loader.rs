use parking_lot::{Mutex, RwLock};
use parking_lot_mpsc::*;
use log::{debug, trace};
use std::sync::Arc;
use std::hash::Hash;
use std::fmt::{Debug, Display};
use rayon::ThreadPool;
use std::collections::HashMap;

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::atomic::Ordering::Relaxed;
use futures::task::Context;
use std::{fmt, thread};
use std::pin::Pin;
use futures::task::Poll;
use std::ops::{Add, AddAssign, Deref, DerefMut};
use futures::Future;
use futures::{
    future::{BoxFuture, FutureExt},
    task::{waker_ref, ArcWake},
};
use tch::{IndexOp, Tensor};
use thoosa_shared::resource::{Resource, ResourceId};

struct RayonNotify {}

impl RayonNotify {
    pub fn new() -> Self {
        Self {}
    }
}
impl ArcWake for RayonNotify {
    fn wake_by_ref(_arc_self: &Arc<Self>) {}
}

pub struct Loader<V>
where
    V: Send + Debug + 'static,
{
    loader_fn: Arc<Box<dyn Fn(&Resource) -> SyncWrapper<V> + Sync + Send>>,
    loader_pool: ThreadPool,

    resources: HashMap<Resource, SyncWrapper<V>>,
    resources_being_loaded: HashMap<Resource, u64>,
}

impl<V> Loader<V>
where
    V: Send + Debug + 'static,
{
    pub fn preload(self_arc: Arc<RwLock<Loader<V>>>, keys: &[Resource], cache_area: u64) {
        debug!("preload_vec: keys={keys:?} cache_area={cache_area:064b}");
        let mut self_write_guard = self_arc.write();

        for key in keys {
            if let Some(loaded) = self_write_guard.resources.get_mut(&key) {
                loaded.cache_areas |= cache_area;
                debug!("preload_vec: adding key={key:?} loaded.cache_areas={:064b}", loaded.cache_areas);
                continue;
            }

            if let Some(being_loaded) = self_write_guard.resources_being_loaded.get_mut(&key) {
                *being_loaded |= cache_area;
                debug!("preload_vec: adding key={key:?} resources_being_loaded_guard.cache_area={:064b}", *being_loaded);
                continue;
            }

            self_write_guard.resources_being_loaded.insert(key.clone(), cache_area);


            debug!("preload_vec: adding key={key:?}");


            let loader = Arc::clone(&self_write_guard.loader_fn);
            let self_arc_pool = Arc::clone(&self_arc);

            let key = key.clone();

            self_write_guard.loader_pool.spawn(move || Self::loader_task(cache_area, loader, self_arc_pool, &key));
        };
    }

    pub fn load_future(self_arc: Arc<RwLock<Loader<V>>>, key: Resource, cache_area: u64) -> BoxFuture<'static, V> {
        let mut self_lock = self_arc.write();


        if !self_lock.resources.contains_key(&key) && !self_lock.resources_being_loaded.contains_key(&key) {
            self_lock.resources_being_loaded.insert(key.clone(), cache_area);

            debug!("preload_vec: adding key={key:?}");

            let loader = Arc::clone(&self_lock.loader_fn);
            let self_arc_pool = Arc::clone(&self_arc);

            let key = key.clone();

            self_lock.loader_pool.spawn(move || Self::loader_task(cache_area, loader, self_arc_pool, &key));
        }
        ResourceFuture::new(Arc::clone(&self_arc), key).boxed()
    }

    fn loader_task<'a>(
        cache_area: u64,
        loader_fn: Arc<Box<dyn Fn(&'a Resource) -> SyncWrapper<V> + Sync + Send>>,
        loader_arc: Arc<RwLock<Loader<V>>>,
        key: &'a Resource,
    ) {
        let mut v = loader_fn(key);
        debug!("preload_vec: in loader_pool 0 current_thread={:?} key={:?}", thread::current().id(), key);

        let mut loader_guard = loader_arc.write();


        v.cache_areas = loader_guard.resources_being_loaded.remove(&key).unwrap_or(cache_area);

        loader_guard.resources.insert(key.clone(), v);


        debug!("preload_vec: in loader_pool 1 current_thread={:?}", thread::current().id());
    }

    pub fn purge_resources(loader_arc: Arc<RwLock<Loader<V>>>, keys: &Vec<Resource>, cache_area: u64) {
        let mut loader_guard = loader_arc.write();


        for key in keys {
            if let Some(v) = loader_guard.resources.get_mut(key) {
                debug!("keys_to_purge: cache_area={:064b}", cache_area);
                v.cache_areas = remove_mask(v.cache_areas, cache_area);
                if v.cache_areas == 0 {
                    loader_guard.resources.remove(key);
                    debug!("keys_to_purge: key={key:?} removed");
                } else {
                    debug!("keys_to_purge: key={key:?} kept, remaining v.cache_areas={:064b}", v.cache_areas);
                }
            }
        }
    }
}

pub struct LoadRunner<V>
where
    V: Send + Debug + 'static,
{
    futures_pool: ThreadPool,
    features: Arc<RwLock<Vec<Mutex<BoxFuture<'static, ()>>>>>,
    features_sender: SyncSender<Mutex<BoxFuture<'static, ()>>>,
    stop: Arc<AtomicBool>,
    cache_areas: u64,
    loader: Arc<RwLock<Loader<V>>>,
}

impl<V> LoadRunner<V>
where
    V: Send + Debug + 'static,
{
    pub fn new<F>(loader: F, loader_threads: usize, futures_channel_bound: usize) -> Self
    where
        F: Fn(&Resource) -> SyncWrapper<V> + Sync + Send + 'static,
    {
        let futures: Arc<RwLock<Vec<Mutex<BoxFuture<()>>>>> = Arc::new(RwLock::new(vec![]));
        let features_rc = Arc::clone(&futures);


        let futures_pool = rayon::ThreadPoolBuilder::new().num_threads(11).spawn_handler(|thread_builder| {
            let b = thread::Builder::new();
            b.name(format!("{}/{:?}", "futures_pool", thread_builder.index()))
                .spawn(|| thread_builder.run())?;

            Ok(())
        }).build().expect("futures_pool");

        let notify = Arc::new(RayonNotify::new());
        let p_notify = Arc::clone(&notify);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_arc = Arc::clone(&stop);

        let (tx, rx) = sync_channel::<Mutex<BoxFuture<()>>>(futures_channel_bound);


        futures_pool.spawn(move || {
            debug!("loader_pool spawned");
            let waker = waker_ref(&p_notify);
            let mut cx = Context::from_waker(&waker);


            // let mut t = Instant::now();
            loop {
                if stop_arc.load(Relaxed) {
                    break;
                }

                let futures_len = futures.read().len();
                if futures_len > 0 {
                    if let Some(mut guard) = futures.try_write() {

                        guard.retain(|fut_mx| {
                            if let Poll::Ready(_t) = fut_mx.lock().as_mut().poll(&mut cx) {
                                //TODO num tries log?
                                false
                            } else {
                                true
                            }
                        });
                    };
                }
                if let Ok(future) = rx.try_recv() {
                    futures.write().push(future);
                }
            }
        });
        Self {
            futures_pool,

            features: features_rc,
            features_sender: tx,

            stop: Arc::clone(&stop),
            cache_areas: 0,

            loader: Arc::new(RwLock::new(Loader {
                loader_pool: rayon::ThreadPoolBuilder::new().num_threads(loader_threads).spawn_handler(|thread_builder| {
                    let b = thread::Builder::new();

                    b.name(format!("{}/{:?}", "loader_pool", thread_builder.index()))
                        .spawn(|| thread_builder.run())?;

                    Ok(())
                }).build().expect("loader_pool"),

                resources: HashMap::new(),
                resources_being_loaded: HashMap::new(),
                loader_fn: Arc::new(Box::new(loader)),
            })),
        }
    }

    #[allow(dead_code)]
    fn stop(&mut self) {
        self.stop.store(true, Relaxed);
    }


    pub fn exec<F>(&mut self, make_future: F)
    where
        F: FnOnce(Arc<RwLock<Loader<V>>>, Box<dyn FnOnce(&Vec<Resource>, u64) + Send>) -> BoxFuture<'static, ()>,//TODO 'static

    {
        let loader_for_purge = Arc::clone(&self.loader);

        self.features_sender.send(Mutex::new(make_future(
            Arc::clone(&self.loader),
            Box::new(move |keys, cache_area| {
                Loader::purge_resources(loader_for_purge, keys, cache_area);
            }),
        ))).expect("features_sender.send");
    }


    pub fn load_future(loader: Arc<RwLock<Loader<V>>>, key: Resource, cache_area: u64) -> BoxFuture<'static, V> {
        Loader::load_future(loader, key, cache_area)
    }

    pub fn preload(&mut self, keys: &[Resource], cache_area: u64) {
        Loader::preload(Arc::clone(&self.loader), keys, cache_area);
    }

    pub fn resources_len(&self) -> usize {
        self.loader.read().resources.len()
    }

    pub fn get_available_cache_area(&mut self) -> u64 {
        let next = first_free_bit(self.cache_areas);
        self.cache_areas |= next;

        next
    }

    pub fn purge_cache_keys(&mut self, keys: Vec<Resource>, area: u64) {
        let mut loader_guard = self.loader.write();

        debug!("purge_cache_keys: 0 {area:064b} resources_guard.len()={}", loader_guard.resources.len());
        let mut keys_to_remove = vec![];
        for key in keys {
            if let Some(sw) = loader_guard.resources.get_mut(&key) {
                sw.cache_areas = remove_mask(sw.cache_areas, area);
                debug!("purge_cache_keys: key={key:?} {:064b}", sw.cache_areas);
                if sw.cache_areas == 0 {
                    debug!("purge_cache_keys: removing key={key:?}");
                    keys_to_remove.push(key);
                }
            }
        }
        for key in keys_to_remove {
            loader_guard.resources.remove(&key);
        }

        debug!("purge_cache: 1 {area:064b} resources_guard.len()={}", loader_guard.resources.len());
    }

    pub fn purge_cache(&mut self, area: u64) {
        let mut loader_guard = self.loader.write();
        debug!("purge_cache: 0 {area:064b} resources_guard.len()={}", loader_guard.resources.len());

        loader_guard.resources.retain(|key, sw| {
            debug!("purge_cache: key={key:?} {:064b}", sw.cache_areas);

            sw.cache_areas = remove_mask(sw.cache_areas, area);
            debug!("purge_cache: key={key:?} {:064b}", sw.cache_areas);
            if sw.cache_areas == 0 {
                debug!("purge_cache: removing key={key:?}");
            }

            sw.cache_areas != 0
        });

        debug!("purge_cache: 1 {area:064b} resources_guard.len()={}", loader_guard.resources.len());
    }
}

pub(crate) fn first_free_bit(c: u64) -> u64 {
    let mut x_next = 1u64;
    if c == 0 {
        return x_next;
    } else if c == u64::MAX {
        return 0;
    }

    while !(c | !x_next) == 0 {
        x_next = x_next << 1;
    }
    x_next
}

pub(crate) fn remove_mask(from: u64, mask: u64) -> u64 {
    !mask & from
}


struct ResourceFuture<V>
where
    V: Send + Debug + 'static,
{
    key: Resource,
    loader: Arc<RwLock<Loader<V>>>,
}

impl<V> ResourceFuture<V>
where
    V: Send + Debug + 'static,
{
    pub fn new(loader: Arc<RwLock<Loader<V>>>, key: Resource) -> Self {
        Self {
            key,
            loader,
        }
    }
}

impl<V> Future for ResourceFuture<V>
where
    V: Send + Debug + 'static,
{
    type Output = V;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        trace!("polling key={:?}", self.key);
        if let Some(resource) = self.loader.read().resources.get(&self.key) {
            trace!("polling key={:?} Ready", self.key);
            Poll::Ready(resource.shallow_clone())
        } else {
            trace!("polling key={:?} Pending ", self.key);
            Poll::Pending
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct SyncWrapper<V>
where
    V: Send + Debug,
{
    v: V,
    shallow_clone_fn: fn(&V) -> V,
    cache_areas: u64,
}

impl<V:Send+Display+Debug> Display for SyncWrapper<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.v, f)
    }
}

impl SyncWrapper<Tensor> {
    pub fn new_t(t:Tensor) -> Self {
        SyncWrapper::new(t, |b|b.shallow_clone())
    }
}

impl SyncWrapper<Box<Tensor>> {
    pub fn new_tb(t:Box<Tensor>) -> Self {
        SyncWrapper::new(t, |b|Box::new(b.shallow_clone()))
    }
}

impl<V> SyncWrapper<V>
where
    V: Send + Debug,
{
    pub fn new(v: V, shallow_clone_fn: fn(&V) -> V) -> Self {
        Self {
            v,
            shallow_clone_fn,
            cache_areas: 0,
        }
    }

    pub fn shallow_clone(&self) -> V {
        (self.shallow_clone_fn)(&self.v)
    }
}
unsafe impl<V> Sync for SyncWrapper<V> where V: Send + Debug {}

impl<V> Deref for SyncWrapper<V>
where
    V: Send + Debug,
{
    type Target = V;
    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<V> DerefMut for SyncWrapper<V>
where
    V: Send + Debug,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.v
    }
}

impl From<Tensor> for SyncWrapper<Box<Tensor>> {
    fn from(t: Tensor) -> Self {
        SyncWrapper::new_tb(Box::new(t))
    }
}

impl AddAssign for SyncWrapper<Box<Tensor>> {
    fn add_assign(&mut self, rhs: Self) {
       *self.v+=*rhs.v;
    }
}


#[cfg(test)]
mod tests {
    use std::collections::{HashSet};
    use std::fmt::Debug;
    use std::future::{Future};
    use std::hash::Hash;
    use std::pin::Pin;

    use std::sync::atomic::{AtomicU32, AtomicU64};
    use std::task::{Context, Poll};
    use std::{thread};

    use std::time::{Duration, Instant};
    use crossbeam_utils::sync::WaitGroup;
    use log::debug;
    use rand::Rng;
    use tch::Device::Cuda;
    use tch::Kind::Float;
    use tch::Tensor;
    use thoosa_shared::resource::{Resource, ResourceId};
    use thoosa_shared::ThoosaError;



    #[ctor::ctor]
    fn init() {
        thoosa_shared::init_test_logging();
    }

    #[derive(Debug, Hash, PartialEq, Eq)]
    struct TestEntity {
        key: Resource,
    }

    #[allow(dead_code)]
    fn load_test_entity(key: &Resource) -> Box<TestEntity> {
        thread::sleep(Duration::new(0, 1000));
        Box::new(TestEntity { key: key.clone() })
    }


    struct TestFuture {
        val: i32,
        loaded: bool,
    }

    impl TestFuture {
        pub fn new(val: i32) -> Self {
            Self {
                val,
                loaded: val != 7,
            }
        }
    }

    impl Future for TestFuture {
        type Output = i32;

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            if !self.loaded {
                let waker = cx.waker().clone();
                thread::sleep(Duration::new(10, 0));
                debug!("7 ready");

                self.loaded = true;

                waker.wake_by_ref();
                Poll::Pending
            } else {
                Poll::Ready(10 * self.val)
            }
        }
    }
    #[allow(dead_code)]
    async fn load_it(i: i32) -> i32 {
        TestFuture::new(i).await
    }


    fn load_test_entity_sv(key: &Resource) -> super::SyncWrapper<TestEntity> {
        thread::sleep(Duration::new(0, rand::thread_rng().gen_range(1e7 as u32..3e9 as u32)));
        // thread::sleep(Duration::new(0, 5e8 as u32));

        super::SyncWrapper::new(
            TestEntity {
                key: key.clone()
            },
            |e| TestEntity {
                key: e.key.clone()
            },
        )
    }


    #[test]
    fn test_loader() -> Result<(), ThoosaError> {
        use parking_lot::{RwLock};
        use rayon::iter::IntoParallelIterator;
        use rayon::prelude::*;
        use log::{info};
        use std::sync::Arc;
        use super::*;
        use futures::{
            future::{FutureExt},

        };

        let pool_0 = rayon::ThreadPoolBuilder::new().num_threads(4).spawn_handler(|thread_builder| {
            let b = std::thread::Builder::new();
            b.name(format!("{}/{:?}", "pool_0", thread_builder.index()))
                .spawn(|| thread_builder.run())?;

            Ok(())
        }).build()?;


        let num_items = 10usize;
        let num_chunks = 1usize;
        let executor = LoadRunner::new(load_test_entity_sv, 4, 100);
        let executor_rc = Arc::new(RwLock::new(executor));
        // let wg_1 = wg.clone();


        let exec_0 = Arc::clone(&executor_rc);
        let exec_1 = Arc::clone(&executor_rc);
        let exec_2 = Arc::clone(&executor_rc);

        let wg = WaitGroup::new();
        let wg_0 = wg.clone();
        let wg_1 = wg.clone();

        let data_done = Arc::new(RwLock::new(HashSet::<Resource>::new()));
        let data_done_rc = Arc::clone(&data_done);
        let data_done_rc_0 = Arc::clone(&data_done_rc);
        let data_done_rc_1 = Arc::clone(&data_done_rc);

        let time_exec = Arc::new(AtomicU64::new(0));

        let time_exec_rc_0 = Arc::clone(&time_exec);
        let time_exec_rc_1 = Arc::clone(&time_exec);


        let t_0 = Instant::now();

        let aaa_resources = (0..num_chunks).map(|c|(0..num_items).map(|i|Resource::new( (c * num_items + i) as u32, 1,format!("aaa_{}", i))).collect::<Vec<_>>()).collect::<Vec<_>>();
        let bbb_resources = (num_chunks..2*num_chunks).map(|c| (0..num_items).map(|i|Resource::new((c * num_items + i) as u32, 1,format!("bbb_{}", i))).collect::<Vec<_>>()).collect::<Vec<_>>();
        let ccc_resources = (2*num_chunks..3*num_chunks).map(|c| (0..num_items).map(|i|Resource::new((c * num_items + i) as u32, 1,format!("bbb_{}", i))).collect::<Vec<_>>()).collect::<Vec<_>>();

        println!("aaa_resources={aaa_resources:?}");
        println!("bbb_resources={bbb_resources:?}");
        println!("ccc_resources={ccc_resources:?}");
        let bbb_resources_1= bbb_resources.clone();
        pool_0.spawn(move || {
            let cache_area_0 = exec_0.write().get_available_cache_area();
            debug!("cache_area_0={cache_area_0:064b}");

            (0..num_chunks).into_par_iter().for_each(|b| {

                exec_0.write().preload(&[aaa_resources[b].clone(), bbb_resources[b].clone()].concat(), cache_area_0);

                (0..num_items).for_each(|j| {
                    let i = b * 10 + j;

                    let wg_0_i = wg_0.clone();


                    let data_done_rc = Arc::clone(&data_done_rc_0);
                    let time_exec_rc = Arc::clone(&time_exec_rc_0);

                    let aaa_resources= aaa_resources.clone();
                    let bbb_resources= bbb_resources.clone();

                    exec_0.write().exec(move |loader_arc, keys_purge| {
                        async move {
                            let t = Instant::now();
                            let _out_a = LoadRunner::<TestEntity>::load_future(Arc::clone(&loader_arc), aaa_resources[b][j].clone(), cache_area_0).await;

                            let _out_b = LoadRunner::<TestEntity>::load_future(loader_arc, bbb_resources[b][j].clone(), cache_area_0).await;

                            let time = t.elapsed().as_millis();
                            // *waiting_time_guard += time;
                            //
                            let mut rw_lock_write_guard = data_done_rc.write();
                            rw_lock_write_guard.insert(aaa_resources[b][j].clone());
                            rw_lock_write_guard.insert(bbb_resources[b][j].clone());
                            drop(rw_lock_write_guard);
                            // *waiting_time_avg_guard = (*waiting_time_avg_guard * (n - 1) + time) / n;

                            let exec_time = rand::thread_rng().gen_range(100..1e9 as u32);
                            time_exec_rc.fetch_add(exec_time.into(), Relaxed);
                            thread::sleep(Duration::new(0, exec_time));

                            drop(wg_0_i);

                            (keys_purge)(&vec![aaa_resources[b][j].clone(), bbb_resources[b][j].clone()], cache_area_0);
                        }.boxed()
                    });

                    debug!("item {i} executed");
                });
            });
            drop(wg_0);
        });

        let bbb_resources = bbb_resources_1.clone();
        pool_0.spawn(move || {
            let cache_area_1 = exec_1.write().get_available_cache_area();
            debug!("cache_area_1={cache_area_1:064b}");

            (0..num_items / 10).into_par_iter().for_each(|b| {
                exec_1.write().preload(&[bbb_resources[b].clone(), ccc_resources[b].clone()].concat(), cache_area_1);

                (0..10).for_each(|j| {
                    let i = b * 10 + j;

                    let wg_0_i = wg_1.clone();

                    let data_done_rc = Arc::clone(&data_done_rc_1);
                    let time_exec_rc = Arc::clone(&time_exec_rc_1);

                    let bbb_resources= bbb_resources.clone();
                    let ccc_resources= ccc_resources.clone();

                    exec_1.write().exec(move |loader, keys_purge| {
                        async move {
                            let t = Instant::now();
                            let _out_a = Loader::<TestEntity>::load_future(Arc::clone(&loader), bbb_resources[b][j].clone(), cache_area_1).await;

                            let _out_b = Loader::<TestEntity>::load_future(loader, ccc_resources[b][j].clone(), cache_area_1).await;


                            let time = t.elapsed().as_millis();


                            let mut rw_lock_write_guard = data_done_rc.write();
                            rw_lock_write_guard.insert(bbb_resources[b][j].clone());
                            rw_lock_write_guard.insert(ccc_resources[b][j].clone());



                            let exec_time = rand::thread_rng().gen_range(100..2e9 as u32);
                            time_exec_rc.fetch_add(exec_time.into(), Relaxed);
                            thread::sleep(Duration::new(0, exec_time));



                            drop(wg_0_i);

                            // info!("1: key_aaa={key_bbb:?} key_bbb={key_ccc:?} done, time={time}");
                            (keys_purge)(&vec![bbb_resources[b][j].clone(), ccc_resources[b][j].clone()], cache_area_1);
                        }.boxed()
                    });

                    debug!("item {i} executed");
                });
            });

            drop(wg_1);
        });


        wg.wait();
        println!("data_done_rc.read().len()={}", data_done_rc.read().len());
        // while data_done_rc.read().len() != num_items * 3 {
        //     // println!("data_done={data_done_rc:?}", data_done_rc);
        //     thread::sleep(Duration::new(1, 0));
        // }
        // debug!("data_done={data_done_rc_1:?}");
        // thread::sleep(Duration::new(120, 0));
        let exec_time_mls = time_exec.load(Relaxed) / 1e6 as u64;
        let elapsed = t_0.elapsed().as_millis();
        let r = exec_time_mls as f64 / elapsed as f64;
        info!("all done, {}/{} in elapsed={elapsed} mls exec time={exec_time_mls} r={r}", data_done_rc.read().len(), exec_2.read().resources_len());

        assert_eq!(data_done_rc.read().len(), num_items * 3);

        // exec_2.write().stop();

        // thread::sleep(Duration::new(3, 0));

        Ok(())
    }

    #[test]
    fn test_first_free_bit() {
        assert_eq!(
            0b0000000000000000000000000000000000000000000000000000000000000001,
            super::first_free_bit(0b0000000000000000000000000000000000000000000000000000000000000000)
        );

        assert_eq!(
            0b0000000000000000000000000000000000000000000000000000000000000010,
            super::first_free_bit(0b0000000000000000000000000000000000000000000000000000000000000001)
        );

        assert_eq!(
            0b0000000000000000000000000000000000000000000000000000000000000100,
            super::first_free_bit(0b0000000000000000000000000000000000000000000000000000000000000011)
        );

        assert_eq!(
            0b0000000000000000000000000000000000000000000000000000000000000100,
            super::first_free_bit(0b0000000000000000000000000000000000000000000000000000000000001011)
        );

        assert_eq!(
            0b1000000000000000000000000000000000000000000000000000000000000000,
            super::first_free_bit(0b0111111111111111111111111111111111111111111111111111111111111111)
        );

        assert_eq!(
            0b0000000000000000000000000000000000000000000000000000000000000000,
            super::first_free_bit(0b1111111111111111111111111111111111111111111111111111111111111111)
        );
    }

    #[test]
    fn test_remove_mask() {
        assert_eq!(
            0b00000000,
            super::remove_mask(0b00000001, 0b000000001)
        );
        assert_eq!(
            0b00000000,
            super::remove_mask(0b00000000, 0b000000001)
        );
        assert_eq!(
            0b00000010,
            super::remove_mask(0b00000010, 0b000000001)
        );
        assert_eq!(
            0b11111101,
            super::remove_mask(0b11111111, 0b000000010)
        );
    }


}