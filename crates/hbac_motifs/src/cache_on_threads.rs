use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::future::Future;
use std::hash::Hash;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock};
use std::task::{Context, Poll, Waker};
use std::thread;
use std::sync::RwLockWriteGuard;
use futures::executor::block_on;
use log::debug;

struct LoaderFuture<K, V> where K: Hash + Eq + Send + Sync + 'static, V: Send + 'static {
    waker: Arc<Mutex<Option<Waker>>>,
    data: Arc<Mutex<Option<V>>>,
    pd:PhantomData<K>
}

impl<K, V> LoaderFuture<K, V> where K: Hash + Eq + Send + Sync + 'static, V: Send + 'static {

    pub fn new<F>(loader: Arc<F>, key:K) -> Self where F: Fn(&K)->V + Send + Sync + 'static{

        let waker = Arc::new(Mutex::new(None::<Waker>));
        let waker_in_thread  = Arc::clone(&waker);

        let data = Arc::new(Mutex::new(None));
        let data_in_thread = Arc::clone(&data);

        thread::spawn(move || {

            data_in_thread.lock().expect("data_in_thread").replace(
                (loader)(&key)
            );

            if let Some(waker) = waker_in_thread.lock().expect("waker_in_thread").take() {
                waker.wake()
            }
        });


        Self {
            waker,
            data,
            pd: PhantomData
        }
    }
}

impl<K, V> Future for LoaderFuture<K, V> where K: Hash + Eq + Send + Sync + 'static, V: Send + 'static {
    type Output = V;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
         match self.data.lock().expect("self.data").take() {
             None => {
                 self.waker.lock().expect("self.waker").replace(cx.waker().clone());

                 Poll::Pending
             }
             Some(data) => {
                 Poll::Ready(data)
             }
         }
    }
}

pub struct CacheOnThreads<K, V> where K:Hash + Eq + Send + Sync+ 'static + Debug, V:Send+'static{
    data_loaded:HashMap<K, V>,

    data_being_loaded:Arc<RwLock<HashMap<K, Arc<Mutex<Option<V>>>>>>,

    loader:Arc<Box<dyn Fn(&K)->V + Sync + Send>>,
    keys_fifo: VecDeque<K>,
    max_keys_fifo: usize,
    on_blocking_listener:Option<Box<dyn FnMut() + Send>>
}

impl <K, V> CacheOnThreads<K, V> where K:Hash + Eq + Clone + Send + Sync + 'static + Debug, V:Send +'static{

    pub fn new<F>(loader: F, max_keys_fifo: usize, on_blocking_listener:Option<Box<dyn FnMut() + Send>>) -> Self
        where F:Fn(&K) -> V + Sync + Send + 'static {
        Self {
            data_loaded: HashMap::new(),
            data_being_loaded: Arc::new(RwLock::new(HashMap::new())),
            loader:Arc::new(Box::new(loader)),
            keys_fifo: VecDeque::new(),
            max_keys_fifo,
            on_blocking_listener
        }
    }

    fn load_blocking(&mut self, key:K) ->bool {
        debug!("load_blocking: blocking on {:?}", key);

        self.on_blocking_listener.as_mut().map(|c|(c)());

        self.data_loaded.insert(
            key.clone(),
            block_on(LoaderFuture::new(Arc::clone(&self.loader), key.clone()))
        );
        debug!("load_blocking: {:?} was loaded", key);

        return true;
    }

    pub fn get_loaded(&self, key:&K) -> Option<&V> {
        // let data_being_loaded_guard = self.data_being_loaded.read().expect("data_being_loaded.read");
        // data_being_loaded_guard.get(key).and_then(|entry_mutex| {
            // let _lock = entry_mutex.lock();
            self.data_loaded.get(key)
        // })
    }

    fn purge_fifo(keys_fifo:&mut VecDeque<K>, max_keys_fifo:usize, data_being_loaded_guard:&mut HashMap<K, Arc<Mutex<Option<V>>>>, data_loaded:&mut HashMap<K, V>){

        if keys_fifo.len() > max_keys_fifo {
            if let Some(key_to_remove) = keys_fifo.pop_front() {
                debug!("removing {:?} ", key_to_remove);
                {
                    let entry_mutex = data_being_loaded_guard.entry(key_to_remove.clone()).or_insert_with(|| Arc::new(Mutex::new(None)));
                    let _lock = entry_mutex.lock().expect("entry_mutex for key_to_remove");

                    data_loaded.remove(&key_to_remove);
                }
                data_being_loaded_guard.remove(&key_to_remove);
            }
        }
    }

    fn update_loaded(&mut self) {
        let mut data_being_loaded_guard = self.data_being_loaded.write().expect("data_being_loaded");

        Self::purge_fifo(&mut self.keys_fifo, self.max_keys_fifo, &mut data_being_loaded_guard, &mut self.data_loaded);

        for (key, entry_mutex) in data_being_loaded_guard.iter() {

            if let Ok(ref mut entry_mutex_guard) = entry_mutex.try_lock() {
                if let Some(data) = entry_mutex_guard.take(){
                    debug!("update_loaded: {:?} was taken from preloaded", key);
                    self.data_loaded.insert(key.clone(), data);

                } else {
                    debug!("update_loaded: {key:?} is locked");
                }
            }
        }
    }

    pub fn blocking_get_vec(&mut self, keys:&[K]) -> Vec<&V> {

        assert!(keys.len() <= self.max_keys_fifo);

        let data_being_loaded_clone = Arc::clone(&self.data_being_loaded);
        let mut data_being_loaded_guard = data_being_loaded_clone.write().expect("data_being_loaded");

        for k in keys.into_iter() {
            self.blocking_get_guard(k, &mut data_being_loaded_guard).expect(format!("no {:?}", k).as_str());
        }

        keys.into_iter().map(|k|{
            self.get_loaded(k).expect(format!("no {:?}", k).as_str())
        }).collect()
    }

    pub fn blocking_get(&mut self, k:&K) -> Option<&V> {
        let data_being_loaded_clone = Arc::clone(&self.data_being_loaded);
        let mut data_being_loaded_guard = data_being_loaded_clone.write().expect("data_being_loaded");
        self.blocking_get_guard(k, &mut data_being_loaded_guard)
    }

    fn blocking_get_guard(&mut self, k:&K, data_being_loaded_guard:&mut RwLockWriteGuard<HashMap<K, Arc<Mutex<Option<V>>>>>) -> Option<&V> {

        if !self.data_loaded.contains_key(k) {
            debug!("blocking_get: loading {:?} self.keys_fifo()={:?}", k, self.keys_fifo);

            Self::purge_fifo(&mut self.keys_fifo, self.max_keys_fifo, data_being_loaded_guard, &mut self.data_loaded);
           
            self.keys_fifo.push_back(k.clone());

            let entry_mutex = data_being_loaded_guard.entry(k.clone()).or_insert_with(|| Arc::new(Mutex::new(None)));
            let entry_mutex_clone = Arc::clone(entry_mutex);
            let mut entry_mutex_guard = entry_mutex_clone.lock().expect("");

            if let Some(data) = entry_mutex_guard.take(){
                debug!("blocking_get: {:?} was taken from preloaded", k);
                self.data_loaded.insert(k.clone(), data);

            } else {

                self.load_blocking(k.clone());
            };


        } else {
            debug!("blocking_get: taking from loaded {:?} self.keys_fifo()={:?}", k, self.keys_fifo);
            if let Some((i, _e)) = self.keys_fifo.iter().enumerate().find(|(_i, e)| e == &k) {
                self.keys_fifo.remove(i);
            }
            self.keys_fifo.push_back(k.clone());

        }

        debug!("blocking_get: self.data_loaded={:?}", self.data_loaded.keys());
        self.data_loaded.get(k)
    }

    pub fn preload_and_blocking_get(&mut self, k:&K, preload:&[K]) -> Option<&V> {

        self.update_loaded();

        self.preload(preload);

        self.blocking_get(k)
    }

    pub fn preload(&mut self, keys:&[K]) {
        debug!("load_no_blocking: keys={:?}", keys);

        for key in keys {
            if self.data_loaded.contains_key(key) {//TODO lock?
                continue
            }

            let data_being_loaded_in_thread = Arc::clone(&self.data_being_loaded);

            let key_in_thread = key.clone();
            let loader_in_thread = self.loader.clone();

            thread::spawn(move || {
                let mut data_being_loaded_guard = data_being_loaded_in_thread.write().expect("data_being_loaded");

                let entry_mutex = data_being_loaded_guard.entry(key_in_thread.clone()).or_insert_with(|| Arc::new(Mutex::new(None)));

                let entry_lock_clone = Arc::clone(entry_mutex);

                drop(data_being_loaded_guard);
                let mut guard = entry_lock_clone.lock().expect("wakers_mx");

                debug!("load_no_blocking: key_in_thread={:?} loaded", key_in_thread);

                guard.replace(loader_in_thread(&key_in_thread));

            });
        }
    }

    pub fn is_empty(&self) -> bool {
        self.data_loaded.is_empty()
    }

}



////////////////////////////////////////////////

pub mod exec {

    use parking_lot::{Mutex, RwLock, RwLockWriteGuard};
    use parking_lot::lock_api::{MutexGuard, RawMutex};
    use parking_lot_mpsc::*;
    use rayon::iter::ParallelBridge;
    use rayon::iter::IntoParallelIterator;
    use rayon::prelude::*;
    use log::{info, debug, trace};
    use std::sync::Arc;
    use std::hash::Hash;
    use std::fmt::Debug;
    use rayon::ThreadPool;
    use std::collections::HashMap;
    use std::collections::VecDeque;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;
    use std::sync::atomic::Ordering::Relaxed;
    use futures::task::Context;
    use std::thread;
    use std::pin::Pin;
    use futures::task::Poll;
    use std::ops::Deref;
    use futures::Future;
    use futures::{
        future::{BoxFuture, FutureExt},
        task::{waker_ref, ArcWake},
    };
    use tch::Tensor;
    use crate::cache_on_threads::LoaderFuture;

    struct RayonNotify {}

    impl RayonNotify {
        pub fn new() -> Self {
            Self {}
        }
    }
    impl ArcWake for RayonNotify {
        fn wake_by_ref(arc_self: &Arc<Self>) {}
    }

    pub struct Loader<K, V>
    where
        K: Clone + Hash + Eq + Send + Sync + 'static + Debug,
        V: Send + Debug + 'static
    {
        //TODO pub
        pub loader_fn: Arc<Box<dyn Fn(&K) -> SyncWrapper<V> + Sync + Send>>,
        pub loader_pool: ThreadPool,
        pub resources: Arc<RwLock<HashMap<K, SyncWrapper<V>>>>,
        pub resources_being_loaded: Arc<RwLock<HashMap<K, u64>>>,
    }

    impl <K, V> Loader <K, V>
    where
        K: Clone + Hash + Eq + Send + Sync + 'static + Debug,
        V: Send + Debug + 'static
    {
        pub fn preload(&mut self, keys: Vec<K>, cache_area: u64) {
            debug!("preload_vec: keys={keys:?} cache_area={cache_area:064b}");
            let mut resources_guard = self.resources.write();
            let mut resources_being_loaded_guard = self.resources_being_loaded.write();

            for key in keys {
                if let Some(loaded) = resources_guard.get_mut(&key) {
                    loaded.cache_areas |= cache_area;
                    debug!("preload_vec: adding key={key:?} loaded.cache_areas={:064b}", loaded.cache_areas);
                    continue;
                }

                if let Some(being_loaded) = resources_being_loaded_guard.get_mut(&key) {
                    *being_loaded |= cache_area;
                    debug!("preload_vec: adding key={key:?} resources_being_loaded_guard.cache_area={:064b}", *being_loaded);
                    continue;
                }

                resources_being_loaded_guard.insert(key.clone(), cache_area);


                debug!("preload_vec: adding key={key:?}");


                let loader = Arc::clone(&self.loader_fn);
                let resources_arc = Arc::clone(&self.resources);
                let resources_being_loaded_arc = Arc::clone(&self.resources_being_loaded);
                let key = key.clone();

                self.loader_pool.spawn(move || {
                    let mut v = loader(&key);
                    debug!("preload_vec: in loader_pool 0 current_thread={:?} key={:?}", thread::current().id(), key);

                    let mut resources_arc_guard = resources_arc.write();

                    v.cache_areas = resources_being_loaded_arc.write().remove(&key).unwrap_or(cache_area);

                    resources_arc_guard.insert(key.clone(), v);


                    debug!("preload_vec: in loader_pool 1 current_thread={:?} key={:?}", thread::current().id(), key);
                });
            };
        }

        pub fn load_future(self_arc: Arc<RwLock<Loader<K, V>>>, key: K, cache_area: u64) -> BoxFuture<'static, V> {

            let self_lock = self_arc.write();
            let mut resources_guard = self_lock.resources.write();
            let mut resources_being_loaded_guard = self_lock.resources_being_loaded.write();

            if !resources_guard.contains_key(&key) && !resources_being_loaded_guard.contains_key(&key) {
                resources_being_loaded_guard.insert(key.clone(), cache_area);


                debug!("preload_vec: adding key={key:?}");


                let loader = Arc::clone(&self_lock.loader_fn);
                let resources_arc = Arc::clone(&self_lock.resources);
                let resources_being_loaded_arc = Arc::clone(&self_lock.resources_being_loaded);
                let key = key.clone();

                self_lock.loader_pool.spawn(move || {
                    let mut v = loader(&key);
                    debug!("preload_vec: in loader_pool 0 current_thread={:?} key={:?}", thread::current().id(), key);

                    let mut resources_arc_guard = resources_arc.write();

                    v.cache_areas = resources_being_loaded_arc.write().remove(&key).unwrap_or(cache_area);

                    resources_arc_guard.insert(key.clone(), v);


                    debug!("preload_vec: in loader_pool 1 current_thread={:?} key={:?}", thread::current().id(), key);
                });
            }
            ResourceFuture::new(Arc::clone(&self_arc), key).boxed()
        }
    }

    pub struct Executor<K, V>
    where
        K: Clone + Hash + Eq + Send + Sync + 'static + Debug,
        V: Send + Debug + 'static
    {
        notify: Arc<RayonNotify>,
        futures_pool: ThreadPool,


        features: Arc<RwLock<Vec<Mutex<BoxFuture<'static, ()>>>>>,
        features_sender: SyncSender<Mutex<BoxFuture<'static, ()>>>,

        keys_to_purge: Arc<RwLock<HashMap<K, usize>>>,

        keys_fifo: VecDeque<K>,


        stop: Arc<AtomicBool>,

        cache_domains:u64,
        //TODO pub
        pub loader:Arc<RwLock<Loader<K,V>>>
    }

    impl<K, V> Executor<K, V>
    where
        K: Clone + Hash + Eq + Send + Sync + 'static + Debug,
        V: Send + Debug + 'static
    {
        pub fn new<F>(loader: F, loader_threads: usize, futures_channel_bound: usize) -> Self
        where
            F: Fn(&K) -> SyncWrapper<V> + Sync + Send + 'static
        {
            let mut futures: Arc<RwLock<Vec<Mutex<BoxFuture<()>>>>> = Arc::new(RwLock::new(vec![]));
            let mut features_rc = Arc::clone(&futures);


            let futures_pool = rayon::ThreadPoolBuilder::new().num_threads(1).spawn_handler(|thread_builder| {
                let mut b = std::thread::Builder::new();
                b.name(format!("{}/{:?}", "futures_pool", thread_builder.index()))
                    .spawn(|| thread_builder.run())?;

                Ok(())
            }).build().expect("pool_1");

            let notify = Arc::new(RayonNotify::new());
            let p_notify = Arc::clone(&notify);
            let stop = Arc::new(AtomicBool::new(false));
            let stop_arc = Arc::clone(&stop);

            let (tx, rx) = sync_channel::<Mutex<BoxFuture<()>>>(futures_channel_bound);


            futures_pool.spawn(move || {
                debug!("loader_pool spawned");
                let waker = waker_ref(&p_notify);
                let mut cx = Context::from_waker(&waker);
                let mut cycle = 0usize;

                // let mut t = Instant::now();
                loop {
                    if stop_arc.load(Ordering::Relaxed) {
                        break
                    }
                    if cycle == usize::MAX {
                        cycle = 0;
                    } else {
                        cycle += 1;
                    }

                    let futures_len = futures.read().len();
                    if futures_len > 0 {
                        if let Some(mut guard) = futures.try_write() {
                            // if cycle % 10000 == 0 {
                            //     debug!("futures loop: in guard len={} ", guard.len());
                            // }
                            guard.retain(|fut_mx| {
                                if let Poll::Ready(t) = fut_mx.lock().as_mut().poll(&mut cx) {
                                    //TODO
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
                notify: Arc::clone(&notify),
                futures_pool,

                features: features_rc,
                features_sender: tx,

                keys_to_purge: Arc::new(RwLock::new(HashMap::new())),

                keys_fifo: VecDeque::new(),



                stop: Arc::clone(&stop),
                cache_domains: 0,

                loader: Arc::new(RwLock::new(Loader {
                    loader_pool: rayon::ThreadPoolBuilder::new().num_threads(loader_threads).spawn_handler(|thread_builder| {
                        let mut b = std::thread::Builder::new();
                        b.name(format!("{}/{:?}", "loader_pool", thread_builder.index()))
                            .spawn(|| thread_builder.run())?;

                        Ok(())
                    }).build().expect("loader_pool"),

                    resources: Arc::new(RwLock::new(HashMap::new())),
                    resources_being_loaded: Arc::new(RwLock::new(HashMap::new())),
                    loader_fn: Arc::new(Box::new(loader)),
                }))
            }
        }

        fn stop(&mut self) {
            self.stop.store(true, Relaxed);
        }


        pub fn exec<F>(&mut self, f: F)
        where
            F: FnOnce(Arc<RwLock<Loader<K,V>>>, Box<dyn FnOnce(Vec<K>, u64) + Send>) -> BoxFuture<'static, ()>,

        {
            let resources_for_purge = Arc::clone(&self.loader.write().resources);

            self.features_sender.send(Mutex::new((f)(
                Arc::clone(&self.loader),

                Box::new(move |keys, cache_area| {
                    let mut resources_for_purge_guard = resources_for_purge.write();

                    for key in keys {
                        if let Some(v) = resources_for_purge_guard.get_mut(&key) {
                            debug!("keys_to_purge: cache_area={:064b}", cache_area);
                            v.cache_areas = remove_mask(v.cache_areas, cache_area);
                            if v.cache_areas == 0 {
                                resources_for_purge_guard.remove(&key);
                                debug!("keys_to_purge: key={key:?} removed");
                            } else {
                                debug!("keys_to_purge: key={key:?} kept, remaining v.cache_areas={:064b}", v.cache_areas);
                            }
                        }
                    }
                })
            )));
        }


        pub fn load_future(loader: Arc<RwLock<Loader<K, V>>>, key: K) -> BoxFuture<'static, V> {
            ResourceFuture::new(loader, key).boxed()
        }

        pub fn resources_len(&self) -> usize {
            self.loader.read().resources.read().len()
        }

        pub fn get_available_cache_domain(&mut self) -> u64 {
            let next = first_free_bit(self.cache_domains);
            self.cache_domains |= next;

            next
        }

        pub fn purge_cache_keys(&mut self, keys:Vec<K>, area:u64) {
            let loader_guard = self.loader.write();
            let mut resources_guard = loader_guard.resources.write();
            debug!("purge_cache_keys: 0 {area:064b} resources_guard.len()={}", resources_guard.len());
            let mut keys_to_remove = vec![];
            for key in keys {
                if let Some(sw) = resources_guard.get_mut(&key) {
                    sw.cache_areas = remove_mask(sw.cache_areas, area);
                    debug!("purge_cache_keys: key={key:?} {:064b}", sw.cache_areas);
                    if sw.cache_areas == 0 {
                        debug!("purge_cache_keys: removing key={key:?}");
                        keys_to_remove.push(key);
                    }
                }
            }
            for key in keys_to_remove {
                resources_guard.remove(&key);
            }

            // resources_guard.retain(|key, sw|{
            //     debug!("purge_cache: key={key:?} {:064b}", sw.cache_areas);
            //
            //     sw.cache_areas = remove_mask(sw.cache_areas, area);
            //     debug!("purge_cache: key={key:?} {:064b}", sw.cache_areas);
            //     if sw.cache_areas == 0 {
            //         debug!("purge_cache: removing key={key:?}");
            //     }
            //
            //     sw.cache_areas != 0
            // });

            debug!("purge_cache: 1 {area:064b} resources_guard.len()={}", resources_guard.len());
        }


        pub fn purge_cache(&mut self, area:u64) {
            let loader_guard = self.loader.write();
            let mut resources_guard = loader_guard.resources.write();
            debug!("purge_cache: 0 {area:064b} resources_guard.len()={}", resources_guard.len());

            resources_guard.retain(|key, sw|{
                debug!("purge_cache: key={key:?} {:064b}", sw.cache_areas);

                sw.cache_areas = remove_mask(sw.cache_areas, area);
                debug!("purge_cache: key={key:?} {:064b}", sw.cache_areas);
                if sw.cache_areas == 0 {
                    debug!("purge_cache: removing key={key:?}");
                }

                sw.cache_areas != 0
            });

            debug!("purge_cache: 1 {area:064b} resources_guard.len()={}", resources_guard.len());
        }

    }

    pub(crate) fn first_free_bit(c:u64) -> u64 {
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

    pub (crate) fn remove_mask(from:u64, mask:u64) -> u64 {
        !mask & from
    }


    struct ResourceFuture<K, V>
    where
        K: Clone + Hash + Eq + Send + Sync + 'static + Debug,
        V: Send + Debug + 'static
    {
        x: usize,
        key: K,
        loader: Arc<RwLock<Loader<K,V>>>
    }

    impl<K, V> ResourceFuture<K, V>
    where
        K: Clone + Hash + Eq + Send + Sync + 'static + Debug,
        V: Send + Debug + 'static
    {
        pub fn new(loader: Arc<RwLock<Loader<K,V>>>, key: K) -> Self {
            Self {
                x: 0,
                key,
                loader
            }
        }
    }

    impl<K, V> Future for ResourceFuture<K, V>
    where
        K: Clone + Hash + Eq + Send + Sync + 'static + Debug,
        V: Send + Debug + 'static
    {
        type Output = V;

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            trace!("polling key={:?}", self.key);
            if let Some(resource) = self.loader.read().resources.read().get(&self.key) {
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
    where  V: Send + Debug {
        v:V,
        shallow_clone_fn:fn(&V)->V,
        cache_areas:u64
    }

    impl <V> SyncWrapper<V>
    where  V: Send + Debug {
        pub fn new(v:V, shallow_clone_fn:fn(&V)->V) -> Self {
            Self {
                v,
                shallow_clone_fn,
                cache_areas: 0
            }
        }

        pub fn shallow_clone(&self) -> V {
            (self.shallow_clone_fn)(&self.v)
        }
    }
    unsafe impl <V> Sync for SyncWrapper<V> where  V: Send + Debug {}

    impl <V> Deref for SyncWrapper<V>  where  V: Send + Debug {
        type Target = V;
        fn deref(&self) -> &Self::Target {
            &self.v
        }
    }
}


#[cfg(test)]
mod tests {

    use std::collections::{HashMap, HashSet, VecDeque};
    use std::fmt::Debug;
    use std::future::{Future};
    use std::hash::Hash;
    use std::io::Write;
    use std::ops::Deref;
    use std::pin::Pin;
    use std::sync::{Arc, Barrier, RwLock};
    use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::mpsc::Receiver;
    use std::task::{Context, Poll, Waker};
    use std::{mem, thread};
    use std::thread::spawn;
    use std::time::{Duration, Instant};
    use crossbeam_utils::sync::WaitGroup;
    use futures::AsyncWriteExt;
    use futures::executor::block_on;
    use futures::sink::Feed;
    use futures::task::{ArcWake, AtomicWaker, SpawnExt};
    use log::debug;
    use parking_lot::lock_api::Mutex;
    use rand::distributions::uniform::SampleBorrow;
    use rand::Rng;
    use rayon::{ThreadPool, yield_now};
    use hbac_shared::HbacError;
    use serde::Deserialize;
    use crate::cache_on_threads::CacheOnThreads;
    use crate::cache_on_threads::exec::Executor;

    #[ctor::ctor]
    fn init() {
        hbac_shared::init_test_logging();
    }

    #[derive(Debug, Hash, PartialEq, Eq)]
    struct TestEntity {
        key:String
    }

    fn load_test_entity(key:&String)->Box<TestEntity>{

        thread::sleep(Duration::new(0, 1000));
        Box::new(TestEntity {key:key.clone()})
    }


    struct TestFuture {
        val:i32,
        loaded: bool
    }

    impl TestFuture {
        pub fn new(val: i32) -> Self {
            Self {
                val,
                loaded: val != 7
            }
        }
    }

    impl Future for TestFuture {
        type Output = i32;

        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {

            if !self.loaded{
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
    async fn load_it(i:i32) -> i32 {
        TestFuture::new(i).await
    }

    #[test]
    fn test_cache_on_threads(){
        let mut cache = CacheOnThreads::new(load_test_entity, 3, None);
        let e_1_0= cache.blocking_get(&"Key1".into()).unwrap().key.clone();
        cache.blocking_get(&"Key1".into());
        cache.blocking_get(&"Key2".into());
        cache.blocking_get(&"Key3".into());
        cache.blocking_get(&"Key4".into());
        cache.blocking_get(&"Key5".into());
        cache.blocking_get(&"Key2".into());
        cache.blocking_get(&"Key1".into());

        let e_1 = cache.get_loaded(&"Key1".into()); //.map(|e| e as *const _ as *const u32).unwrap();
        let e_2 = cache.get_loaded(&"Key2".into()); //.map(|e| e as *const _ as *const u32).unwrap();

        // let xx = e_1.and_then(|e_1| e_2.map(|e_2|(e_1 as *const _, e_2 as *const _))).unwrap();

        // let p_1 = e_1.unwrap() as *const _;
        // let p_2 = e_2.unwrap() as *const _;
        // debug!("ee={:?}", e_1.unwrap() as *const _ == e_1_0.unwrap() as *const _);
        // debug!("e_1={:?}", e_1.unwrap() as *const _);
        // debug!("e_1_0={:?}", e_1_0.unwrap() as *const _);

        assert_eq!(e_1_0, "Key1");
        assert_eq!(e_1.unwrap().key, "Key1");
        assert_eq!(e_2.unwrap().key, "Key2");

        // let i:&TestEntity = unsafe { &*(e_1 as *const _ as *const &TestEntity)};
        // debug!("e_1={:?}", i);

    }

    #[test]
    fn test_load_no_blocking(){
        let mut cache = CacheOnThreads::new(load_test_entity, 3, None);
        cache.preload(&["Key1".into(), "Key2".into()]);

        cache.blocking_get(&"Key1".into());
        cache.blocking_get(&"Key2".into());

        let e_1 = cache.get_loaded(&"Key1".into()); //.map(|e| e as *const _ as *const u32).unwrap();
        let e_2 = cache.get_loaded(&"Key2".into()); //.map(|e| e as *const _ as *const u32).unwrap();

        assert_eq!(e_1.unwrap().key, "Key1");
        assert_eq!(e_2.unwrap().key, "Key2");

    }

    fn load_test_entity_sv(key:&String)->super::exec::SyncWrapper<TestEntity>{
        thread::sleep(Duration::new(0, rand::thread_rng().gen_range(1e7 as u32..3e9 as u32)));
        // thread::sleep(Duration::new(0, 5e8 as u32));

        super::exec::SyncWrapper::new(
            TestEntity {
                key:key.clone()
            },
            |e|TestEntity{
                key:e.key.clone()
            }
        )
    }


    #[test]
    fn test_executor() -> Result<(), HbacError>{
        use parking_lot::{Mutex, RwLock, RwLockWriteGuard};
        use parking_lot::lock_api::{MutexGuard, RawMutex};
        use parking_lot_mpsc::*;
        use rayon::iter::ParallelBridge;
        use rayon::iter::IntoParallelIterator;
        use rayon::prelude::*;
        use log::{info, trace};
        use std::sync::Arc;
        use std::hash::Hash;
        use std::fmt::Debug;
        use rayon::ThreadPool;
        use std::collections::HashMap;
        use super::exec::*;
        use futures::{
            future::{BoxFuture, FutureExt},
            task::{waker_ref, ArcWake},
        };

        let pool_0 = rayon::ThreadPoolBuilder::new().num_threads(4).spawn_handler(|thread_builder| {
            let mut b = std::thread::Builder::new();
            b.name(format!("{}/{:?}", "pool_0", thread_builder.index()))
                .spawn(|| thread_builder.run())?;

            Ok(())
        }).build()?;


        let num_items = 1000usize;
        let mut executor = Executor::new(load_test_entity_sv, 4, 100);
        let mut executor_rc = Arc::new(RwLock::new(executor));
        // let wg_1 = wg.clone();



        let exec_0 = Arc::clone(&executor_rc);
        let exec_1 = Arc::clone(&executor_rc);
        let exec_2 = Arc::clone(&executor_rc);

        let wg = WaitGroup::new();
        let wg_0 = wg.clone();
        let wg_1 = wg.clone();

        let data_done = Arc::new(RwLock::new(HashSet::<String>::new()));
        let data_done_rc = Arc::clone(&data_done);
        let data_done_rc_0 = Arc::clone(&data_done_rc);
        let data_done_rc_1 = Arc::clone(&data_done_rc);

        let time_exec = Arc::new(AtomicU64::new(0));

        let time_exec_rc_0 = Arc::clone(&time_exec);
        let time_exec_rc_1 = Arc::clone(&time_exec);


        let waiting_time = Arc::new(Mutex::new(0u128));
        let waiting_time_rc = Arc::clone(&waiting_time);


        let waiting_time_avg = Arc::new(Mutex::new(0u128));
        let waiting_time_avg_rc = Arc::clone(&waiting_time_avg);

        let t_0 = Instant::now();
        pool_0.spawn(move || {
            let cache_area_0 = exec_0.write().get_available_cache_domain();
            debug!("cache_area_0={cache_area_0:064b}");

            (0..num_items / 10).into_par_iter().for_each(|b| {

                exec_0.write().loader.write().preload(
                    (0..10).flat_map(|j| [
                        format!("aaa_{}", b * 10 + j),
                        format!("bbb_{}", b * 10 + j)
                    ]).collect::<Vec<String>>(),
                   cache_area_0
                );

                (0..10).for_each(|j| {
                    let i = b * 10 + j;

                    let wg_0_i = wg_0.clone();

                    let waiting_time_rc = Arc::clone(&waiting_time_rc);
                    let waiting_time_avg_rc = Arc::clone(&waiting_time_avg_rc);

                    let data_done_rc = Arc::clone(&data_done_rc_0);
                    let time_exec_rc = Arc::clone(&time_exec_rc_0);

                    exec_0.write().exec(move |loader_arc, keys_purge| {
                        async move {
                            let t = Instant::now();
                            let key_aaa = format!("aaa_{i}");
                            let out_a = Loader::<String, TestEntity>::load_future(Arc::clone(&loader_arc), key_aaa.clone(), cache_area_0).await;

                            let key_bbb = format!("bbb_{i}");
                            let out_b = Loader::<String, TestEntity>::load_future(loader_arc, key_bbb.clone(), cache_area_0).await;

                            // let mut waiting_time_guard = waiting_time_rc.lock();
                            let time = t.elapsed().as_millis();
                            // *waiting_time_guard += time;
                            //
                            let mut rw_lock_write_guard = data_done_rc.write();
                            rw_lock_write_guard.insert(key_aaa.clone());
                            rw_lock_write_guard.insert(key_bbb.clone());
                            drop(rw_lock_write_guard);
                            // let n = rw_lock_write_guard.len() as u128;
                            //
                            // let mut waiting_time_avg_guard = waiting_time_avg_rc.lock();
                            //
                            // *waiting_time_avg_guard = (*waiting_time_avg_guard * (n - 1) + time) / n;

                            let exec_time = rand::thread_rng().gen_range(100..1e9 as u32);
                            time_exec_rc.fetch_add(exec_time.into(), Relaxed);
                            thread::sleep(Duration::new(0, exec_time));
                            // let x: usize = (0..10_000).flat_map(|v| (0..10_000)).sum();
                            // trace!("x={x}");

                            drop(wg_0_i);

                            // info!("key_aaa={key_aaa} key_bbb={key_bbb} done, time={time} waiting_time_rc={} waiting_time_avg_guard={}", waiting_time_guard, waiting_time_avg_guard);
                            info!("0: key_aaa={key_aaa} key_bbb={key_bbb} done, time={time}",);
                            (keys_purge)(vec![key_aaa, key_bbb], cache_area_0);
                        }.boxed()
                    });

                    debug!("item {i} executed");
                });

            });
            drop(wg_0);
        });

        pool_0.spawn(move || {
            let cache_area_1 = exec_1.write().get_available_cache_domain();
            debug!("cache_area_1={cache_area_1:064b}");

            (0..num_items/10).into_par_iter().for_each(|b| {

                exec_1.write().loader.write().preload(
                    (0..10).flat_map(|j|[
                        format!("bbb_{}", b*10+j),
                        format!("ccc_{}", b*10+j)
                    ]).collect::<Vec<String>>(),
                    cache_area_1
                );

                (0..10).for_each(|j|{
                    let i = b * 10 + j;

                    let wg_0_i = wg_1.clone();

                    let data_done_rc = Arc::clone(&data_done_rc_1);
                    let time_exec_rc = Arc::clone(&time_exec_rc_1);

                    exec_1.write().exec(move |loader, keys_purge| {
                        async move {
                            let t = Instant::now();
                            let key_aaa = format!("bbb_{i}");
                            let out_a = Loader::<String,TestEntity>::load_future(Arc::clone(&loader), key_aaa.clone(), cache_area_1).await;

                            let key_bbb = format!("ccc_{i}");
                            let out_b = Loader::<String,TestEntity>::load_future(loader, key_bbb.clone(), cache_area_1).await;


                            let time = t.elapsed().as_millis();


                            let mut rw_lock_write_guard = data_done_rc.write();
                            rw_lock_write_guard.insert(key_aaa.clone());
                            rw_lock_write_guard.insert(key_bbb.clone());
                            // let n = rw_lock_write_guard.len() as u128;


                            let exec_time = rand::thread_rng().gen_range(100..2e9 as u32);
                            time_exec_rc.fetch_add(exec_time.into(), Relaxed);
                            thread::sleep(Duration::new(0, exec_time));
                            // let x:usize = (0..10_000).flat_map(|v| (0..10_000)).sum();
                            // trace!("x={x}");

                            drop(wg_0_i);

                            info!("1: key_aaa={key_aaa} key_bbb={key_bbb} done, time={time}");
                            (keys_purge)(vec![key_aaa, key_bbb], cache_area_1);
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
        let exec_time_mls = time_exec.load(Relaxed)/1e6 as u64;
        let elapsed = t_0.elapsed().as_millis();
        let r = exec_time_mls as f64/elapsed as f64;
        info!("all done, {}/{} in elapsed={elapsed} mls exec time={exec_time_mls} r={r}", data_done_rc.read().len(), exec_2.read().resources_len());

        assert_eq!(data_done_rc.read().len(), num_items * 3);

        // exec_2.write().stop();

        // thread::sleep(Duration::new(3, 0));

        Ok(())
    }

    #[test]
    fn test_first_free_bit(){
        assert_eq!(
            0b0000000000000000000000000000000000000000000000000000000000000001,
            super::exec::first_free_bit(0b0000000000000000000000000000000000000000000000000000000000000000)
        );

        assert_eq!(
            0b0000000000000000000000000000000000000000000000000000000000000010,
            super::exec::first_free_bit(0b0000000000000000000000000000000000000000000000000000000000000001)
        );

        assert_eq!(
            0b0000000000000000000000000000000000000000000000000000000000000100,
            super::exec::first_free_bit(0b0000000000000000000000000000000000000000000000000000000000000011)
        );

        assert_eq!(
            0b0000000000000000000000000000000000000000000000000000000000000100,
            super::exec::first_free_bit(0b0000000000000000000000000000000000000000000000000000000000001011)
        );

        assert_eq!(
            0b1000000000000000000000000000000000000000000000000000000000000000,
            super::exec::first_free_bit(0b0111111111111111111111111111111111111111111111111111111111111111)
        );

        assert_eq!(
            0b0000000000000000000000000000000000000000000000000000000000000000,
            super::exec::first_free_bit(0b1111111111111111111111111111111111111111111111111111111111111111)
        );
    }

    #[test]
    fn test_remove_mask(){
        assert_eq!(
            0b00000000,
            super::exec::remove_mask(0b00000001, 0b000000001)
        );
        assert_eq!(
            0b00000000,
            super::exec::remove_mask(0b00000000, 0b000000001)
        );
        assert_eq!(
            0b00000010,
            super::exec::remove_mask(0b00000010, 0b000000001)
        );
        assert_eq!(
            0b11111101,
            super::exec::remove_mask(0b11111111, 0b000000010)
        );
    }

    #[test]
    fn test_xxx() {
        // let mut x_0 = 0u64;
        // let x_1 = 1u64;
        // let x_2 = x_1 << 1;
        // let mut x_next = x_1;
        //
        // println!("!x_1 {:064b}", !x_1);
        // println!("!x_1 {:064b}", !(x_1<<1));
        //
        //
        // println!("x_0={x_0:064b}");
        // println!("x_1={x_1:064b}");
        // println!("x_2={x_2:064b}");
        //
        // x_0 |= x_1;
        // println!("1. x_0={x_0:064b}");
        //
        // x_0 |= x_2;
        // println!("2. x_0={x_0:064b} next={:064b}", super::exec::first_free_bit(x_0));
        //
        // x_0 ^= x_1;
        // println!("3. x_0={x_0:064b} next={:064b}", super::exec::first_free_bit(x_0));
        //
        // x_0 ^= x_2;
        // println!("4. x_0={x_0:064b}");
        let x_0 = 0b00010;
        let x_1 = 0b00001;

        println!("r={:064b}", !x_0 & x_1);

        let x_0 = 0b00010;
        let x_1 = 0b00011;

        println!("r={:064b}", !x_0 & x_1);
    }


}