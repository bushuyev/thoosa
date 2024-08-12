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


#[cfg(test)]
mod tests {

    use std::collections::{HashMap, HashSet};
    use std::fmt::Debug;
    use std::future::{Future};
    use std::hash::Hash;
    use std::io::Write;
    use std::ops::Deref;
    use std::pin::Pin;
    use std::sync::{Arc, RwLock};
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::mpsc::Receiver;
    use std::task::{Context, Poll, Waker};
    use std::{mem, thread};
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
    use rayon::ThreadPool;
    use hbac_shared::HbacError;
    use serde::Deserialize;
    use crate::cache_on_threads::CacheOnThreads;

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

    fn load_test_entity_sv(key:&String)->SyncWrapper<TestEntity>{
        thread::sleep(Duration::new(0, rand::thread_rng().gen_range(100..1e9 as u64).try_into().unwrap()));

        SyncWrapper::new(
            TestEntity {
                key:key.clone()
            },
            |e|TestEntity{
                key:e.key.clone()
            }
        )
    }

    #[derive(Debug, Hash, PartialEq, Eq)]
    struct SyncWrapper<V>
    where  V: Send + Debug {
        v:V,
        shallow_clone_fn:fn(&V)->V
    }

    impl <V> SyncWrapper<V>
    where  V: Send + Debug {
        pub fn new(v:V, shallow_clone_fn:fn(&V)->V) -> Self {
            Self {
                v,
                shallow_clone_fn
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


    #[test]
    fn test_loader_0() -> Result<(), HbacError> {
        use rayon::iter::ParallelIterator;
        use rayon::iter::IntoParallelIterator;
        use std::rc::Rc;
        use parking_lot::{Mutex, RwLock, MutexGuard, MappedMutexGuard, RwLockReadGuard, MappedRwLockReadGuard};

        use rayon::iter::ParallelBridge;
        use rayon::prelude::*;
        use crossbeam_utils::sync::WaitGroup;
        use std::borrow::BorrowMut;


        let num_threads = std::thread::available_parallelism()?;
        println!("num_threads={num_threads:?}");


        struct Loader<K, V>
        where
            K: Clone + Hash + Eq + Send + Sync + 'static + Debug,
            V: Send + Debug + 'static,
        {
            loader_pool:ThreadPool,
            data_loaded:Arc<RwLock<HashMap<K, SyncWrapper<V>>>>,
            loader_fn:Arc<Box<dyn Fn(&K)->SyncWrapper<V> + Sync + Send>>,
            can_read:Arc<RwLock<bool>>
        }

        impl<K, V> Loader<K, V>
        where
            K: Clone + Hash + Eq + Send + Sync + 'static + Debug,
            V: Send + Debug + 'static,
        {

            pub fn new<F>(loader:F) -> Result<Self, HbacError>
                where F:Fn(&K) -> SyncWrapper<V> + Sync + Send + 'static {


                Ok(Self {
                    loader_fn:Arc::new(Box::new(loader)),
                    loader_pool:rayon::ThreadPoolBuilder::new().num_threads(2).spawn_handler(|thread_builder| {

                        let mut b = std::thread::Builder::new();
                        b.name(format!("{}/{:?}", "pool_1", thread_builder.index()))
                            .spawn(|| thread_builder.run())?;

                        Ok(())
                    }).build()?,
                    data_loaded:Arc::new(RwLock::new(HashMap::new())),
                    can_read: Arc::new(RwLock::new(false))
                })
            }


            fn preload(&mut self, keys: Vec<K>) -> Result<(), HbacError> {
                println!("preload keys={keys:?}");

                for key in keys {
                    if self.data_loaded.read().contains_key(&key){
                        println!("skipping key={key:?}");
                        continue;
                    }
                    println!("loading key={key:?}");

                    let loader = Arc::clone(&self.loader_fn);
                    let data_loaded = Arc::clone(&self.data_loaded);

                    self.loader_pool.spawn(move ||{

                        let v = loader(&key);
                        println!("pool 0: current_thread={:?}", thread::current().id());
                        // println!("v={:?}", *v);


                        data_loaded.write().insert(key, v);

                    });
                };


                Ok(())
            }

            pub fn try_get(&self, key: &K) -> Result<MappedRwLockReadGuard<SyncWrapper<V>>, HbacError>  {

                if let Some(guard) = self.data_loaded.try_read() {

                    if guard.contains_key(key) {
                        return Ok(RwLockReadGuard::map(guard, |d: &HashMap<K, SyncWrapper<V>>| d.get(key).unwrap()));
                    } else {
                        Err(HbacError::new("try_get: no data"))
                    }
                } else {
                    Err(HbacError::new("try_get: can't lock"))
                }

            }

            pub fn get(&self, key: &K) -> Result<MappedRwLockReadGuard<SyncWrapper<V>>, HbacError> {

                let mut i = 0;
                for a in 0..10_000 {
                    if let Some(guard) = self.data_loaded.try_read() {

                        if guard.contains_key(key) {
                            return Ok(RwLockReadGuard::map(guard, |d: &HashMap<K, SyncWrapper<V>>| d.get(key).unwrap()));

                        }
                    }
                    i+=1;

                    println!("get: waiting for {key:?}");
                    rayon::yield_local();
                }

                Err(HbacError::new("xxx"))
            }

            pub fn test(&mut self)-> usize {
                1
            }

        }

        let main_pool = rayon::ThreadPoolBuilder::new().num_threads(8).spawn_handler(|thread_builder| {

            let mut b = std::thread::Builder::new();
            b.name(format!("{}/{:?}", "main_pool", thread_builder.index()))
                .spawn(|| thread_builder.run())?;

            Ok(())
        }).build()?;

        let loader_pool = rayon::ThreadPoolBuilder::new().num_threads(8).spawn_handler(|thread_builder| {

            let mut b = std::thread::Builder::new();
            b.name(format!("{}/{:?}", "main_pool", thread_builder.index()))
                .spawn(|| thread_builder.run())?;

            Ok(())
        }).build()?;

        let wg = WaitGroup::new();
        let wg_aaa = wg.clone();
        let wg_bbb = wg.clone();

        let mut loader = Box::new(Loader::new(load_test_entity_sv).expect("loader"));
        let loader_rc = Arc::new(RwLock::new(loader));
        let main_loader_rc = Arc::clone(&loader_rc);
        let loader_rc_aaa_main = Arc::clone(&loader_rc);
        let loader_rc_aaa = Arc::clone(&loader_rc);
        let loader_rc_bbb_main = Arc::clone(&loader_rc);
        let loader_rc_bbb = Arc::clone(&loader_rc);

        let can_read = Arc::new(RwLock::new(false));
        let mut can_read_cl = Arc::clone(&can_read);
        let mut can_read_aaa_main = Arc::clone(&can_read);
        let mut can_read_aaa = Arc::clone(&can_read);
        let mut can_read_bbb_main = Arc::clone(&can_read);
        let mut can_read_bbb = Arc::clone(&can_read);

        let loaded = Arc::new(RwLock::new(HashSet::new()));
        let loaded_cl = Arc::clone(&loaded);
        let loaded_aaa = Arc::clone(&loaded);
        let loaded_bbb = Arc::clone(&loaded);

        #[derive(Debug)]
        enum TaskState<K:Send> {
            NEEDS_DATA{keys:Vec<K>},
            RUNNING,
            DONE
        }


        trait Task:Send {
            type K: Clone + Hash + Eq + Send + Sync + Debug;
            type R:Send + Debug;

            fn run(&mut self, loader:Arc<RwLock<Box<Loader<Self::K, Self::R>>>>) -> Box<dyn Future<Output=()> + '_>;
            fn state_trns(&mut self) -> TaskState<Self::K>;
        }

        //TODO
        struct TestTask_1 {
            x:usize,
            state:TaskState<String>
        }

        impl TestTask_1 {
            pub fn new() -> Self {
                Self {
                    x: 0,
                    state:TaskState::NEEDS_DATA{
                        keys:(0..10).map(|i|i).map(|i|format!("aaa_{i}")).collect()
                    }
                }
            }
        }

        impl Task for TestTask_1 {
            type K = String;
            type R = TestEntity;

            // fn resources(&self) -> Vec<String> {
            //     (0..10).map(|i|i+self.x).map(|i|format!("aaa_{i}")).collect()
            // }

            fn run(&mut self, loader:Arc<RwLock<Box<Loader<String, TestEntity>>>>) -> Box<dyn Future<Output=()> + '_>{
                self.x += 10;
                todo!()
            }

            fn state_trns(&mut self) -> TaskState<String> {
                mem::replace(&mut self.state, TaskState::RUNNING)
            }

        }

        struct TestTask_2 {
            x:usize,
            state:TaskState<String>
        }

        impl TestTask_2 {
            pub fn new() -> Self {
                Self {
                    x: 0,
                    state:TaskState::NEEDS_DATA{
                        keys:(0..10).map(|i|i).map(|i|format!("bbb_{i}")).collect()
                    }
                }
            }
        }

        impl Task for TestTask_2 {
            type K = String;
            type R = TestEntity;

            // fn resources(&self) -> Vec<String> {
            //     (0..10).map(|i|i+self.x).map(|i|format!("bbb_{i}")).collect()
            // }

            fn run(&mut self, loader:Arc<RwLock<Box<Loader<String, TestEntity>>>>) -> Box<dyn Future<Output=()> + '_> {
                Box::new(async move {
                    loader.write().test();
                    self.x += 10;
                })
            }

            fn state_trns(&mut self) -> TaskState<String> {

                mem::replace(&mut self.state, TaskState::RUNNING)
            }

        }

        let mut tasks:Vec<Box<dyn Task<K=String, R=TestEntity>>> = vec![
            Box::new(TestTask_1::new()),
            Box::new(TestTask_2::new())
        ];

        for mut task in tasks {

            let can_read_cl = Arc::clone(&can_read_cl);
            let loader_rc = Arc::clone(&loader_rc);
            // println!("resources: {:?}", task.state());
            match task.state_trns() {
                TaskState::NEEDS_DATA {keys}=> {
                    let keys = keys.clone();
                    loader_pool.install( || {
                        rayon::spawn(move || {
                            *can_read_cl.write() = false;

                            loader_rc.write().preload(keys);
                            *can_read_cl.write() = true;
                        });
                    });

                },
                TaskState::RUNNING=> {
                    main_pool.install( move || {
                        rayon::spawn(move || {
                            // let waker = AtomicWaker::new();
                            // let mut cx = Context::from_waker(&waker);
                            // let fut = task.run(loader_rc);
                            // loop {
                            //     if let Poll::Ready(t) = f(&mut cx) {
                            //         return t;
                            //     }
                            //
                            //     rayon::yield_now();
                            //     // // Wait for a wakeup.
                            //     // while !thread_notify.unparked.swap(false, Ordering::Acquire) {
                            //     //     // No wakeup occurred. It may occur now, right before parking,
                            //     //     // but in that case the token made available by `unpark()`
                            //     //     // is guaranteed to still be available and `park()` is a no-op.
                            //     //     thread::park();
                            //     // }
                            // }

                        })
                    });
                },
                TaskState::DONE => {

                }
            }
        }


        main_pool.install( move || {


            rayon::spawn( move || {
                println!("getting aaa");

                (0..10).into_par_iter().for_each(|i| {
                    let key = format!("aaa_{i}");

                    for a in (0..usize::MAX) {
                        if *can_read_aaa_main.read() {
                            if let Some(lock) = loader_rc_aaa_main.try_read() {
                                if let Ok(v) = lock.try_get(&key) {
                                    println!("got key={key} v.1={v:?} in thread={:?} on a={a}", thread::current().name());
                                    loaded_cl.write().insert(key.clone());
                                    break;
                                }
                            }
                        }

                        rayon::yield_now();
                    }
                });
                println!("got aaa");
                drop(wg_aaa);
            });

            rayon::spawn( move || {
                println!("getting bbb");

                (0..10).into_par_iter().for_each(|i| {
                    let key = format!("bbb_{i}");

                    let e = {
                        let mut a = 0;
                        loop {
                            if *can_read_bbb_main.read() {
                                if let Some(lock) = loader_rc_bbb_main.try_read() {
                                    if let Ok(v) = lock.try_get(&key) {
                                        println!("got key={key} v.1={v:?} in thread={:?} on a={a}", thread::current().name());
                                        loaded_bbb.write().insert(key.clone());
                                        break Some(v.shallow_clone());
                                    }
                                }
                            }
                            a+=1;
                            if a == usize::MAX {
                                break None;
                            }

                            rayon::yield_now();
                        }
                    };

                });
                println!("got bbb");
                drop(wg_bbb);
            });

        });

        println!("aaaaaaaaaaaaaaaaaaaa");

        // loader_pool.install( || {
        //     rayon::spawn( move || {
        //         println!("loading aaa");
        //
        //         *can_read_aaa.write() = false;
        //
        //         loader_rc_aaa.write().preload((0..10).map(|i| format!("aaa_{i}")).collect());
        //         *can_read_aaa.write() = true;
        //
        //     });
        //
        //     rayon::spawn( move || {
        //         println!("loading bbb");
        //
        //         *can_read_bbb.write() = false;
        //
        //         loader_rc_bbb.write().preload((0..10).map(|i| format!("bbb_{i}")).collect());
        //         *can_read_bbb.write() = true;
        //
        //     });
        //
        //     println!("xxxxxxxxxxxxxxx");
        // });

        wg.wait();


        assert!(loaded.read().contains("aaa_9"));
        assert!(loaded.read().contains("bbb_9"));



        Ok(())
    }

    #[test]
    fn test_0() -> Result<(), HbacError>{
        use parking_lot::{Mutex, RwLock, RwLockWriteGuard};
        use parking_lot::lock_api::{MutexGuard, RawMutex};
        use rayon::iter::ParallelBridge;
        use rayon::iter::IntoParallelIterator;
        use rayon::prelude::*;
        use log::info;

        use futures::{
            future::{BoxFuture, FutureExt},
            task::{waker_ref, ArcWake},
        };

        let pool_0 = rayon::ThreadPoolBuilder::new().num_threads(2).spawn_handler(|thread_builder| {
            let mut b = std::thread::Builder::new();
            b.name(format!("{}/{:?}", "pool_0", thread_builder.index()))
                .spawn(|| thread_builder.run())?;

            Ok(())
        }).build()?;

        let pool_1 = rayon::ThreadPoolBuilder::new().num_threads(4).spawn_handler(|thread_builder| {
            let mut b = std::thread::Builder::new();
            b.name(format!("{}/{:?}", "pool_1", thread_builder.index()))
                .spawn(|| thread_builder.run())?;

            Ok(())
        }).build()?;




        // let wg_1 = wg.clone();


        struct RayonNotify {}

        impl RayonNotify {
            pub fn new() -> Self {
                Self {}
            }
        }
        impl ArcWake for RayonNotify {
            fn wake_by_ref(arc_self: &Arc<Self>) {}
        }

        struct Executor<K, V>
        where
            K: Clone + Hash + Eq + Send + Sync + 'static + Debug,
            V: Send + Debug + 'static
        {
            notify: Arc<RayonNotify>,
            futures_pool:ThreadPool,
            loader_pool:ThreadPool,
            features:Arc<RwLock<Vec<Mutex<BoxFuture<'static, ()>>>>>,
            resources:Arc<RwLock<HashMap<K, SyncWrapper<V>>>>,
            can_read:Arc<RwLock<bool>>,
            loader_fn:Arc<Box<dyn Fn(&K)->SyncWrapper<V> + Sync + Send>>,
        }

        impl <K, V> Executor<K, V>
        where
            K: Clone + Hash + Eq + Send + Sync + 'static + Debug,
            V: Send + Debug + 'static {

            pub fn new<F>(loader:F) -> Self
                where F:Fn(&K) -> SyncWrapper<V> + Sync + Send + 'static {

                let mut futures:Arc<RwLock<Vec<Mutex<BoxFuture<()>>>>> = Arc::new(RwLock::new(vec![]));
                let mut features_rc = Arc::clone(&futures);
                let notify = Arc::new(RayonNotify::new());
                let can_read = Arc::new(RwLock::new(true));
                let can_read_rc = Arc::clone(&can_read);
                let futures_pool = rayon::ThreadPoolBuilder::new().num_threads(1).spawn_handler(|thread_builder| {
                    let mut b = std::thread::Builder::new();
                    b.name(format!("{}/{:?}", "futures_pool", thread_builder.index()))
                        .spawn(|| thread_builder.run())?;

                    Ok(())
                }).build().expect("pool_1");
                let p_notify = Arc::clone(&notify);

                futures_pool.spawn(move ||{
                    debug!("loader_pool spawned");
                    let waker = waker_ref(&p_notify);
                    let mut cx = Context::from_waker(&waker);

                    loop {
                        // if let Some(can_read) = can_read.try_read() {
                            let futures_arc = Arc::clone(&futures);
                            // println!("futures loop: 0");
                            if let Some(mut guard) = futures_arc.try_write() {
                                debug!("futures loop: in guard len={} ", guard.len());
                                guard.retain(|fut_mx| {
                                    if let Poll::Ready(t) = fut_mx.lock().as_mut().poll(&mut cx) {
                                        false
                                    } else {
                                        true
                                    }
                                });

                                drop(guard);
                            };
                        // }
                        rayon::yield_local();
                    }
                });
                Self {
                    notify: Arc::clone(&notify),
                    futures_pool,
                    loader_pool:rayon::ThreadPoolBuilder::new().num_threads(8).spawn_handler(|thread_builder| {

                        let mut b = std::thread::Builder::new();
                        b.name(format!("{}/{:?}", "loader_pool", thread_builder.index()))
                            .spawn(|| thread_builder.run())?;

                        Ok(())
                    }).build().expect("loader_pool"),
                    features: features_rc,
                    resources: Arc::new(RwLock::new(HashMap::new())),
                    loader_fn:Arc::new(Box::new(loader)),
                    can_read:can_read_rc
                }
            }

            fn preload(&mut self, keys: Vec<K>) -> Result<(), HbacError> {
                debug!("preload keys={keys:?}");

                for key in keys {
                    if self.resources.read().contains_key(&key){
                        // println!("skipping key={key:?}");
                        continue;
                    }
                    debug!("loading key={key:?}");

                    let loader = Arc::clone(&self.loader_fn);
                    let data_loaded = Arc::clone(&self.resources);

                    self.loader_pool.spawn(move ||{

                        let v = loader(&key);
                        debug!("preload 0: current_thread={:?} key={:?}", thread::current().id(), key);

                        data_loaded.write().insert(key.clone(), v);
                        debug!("preload 1: current_thread={:?} key={:?}", thread::current().id(), key);

                    });
                };


                Ok(())
            }



            pub fn exec<F>(& mut self, f:F )
                where F:FnOnce()->BoxFuture<'static, ()>
            {
                // println!("exec");

                // *self.can_read.write() = false;

                loop {
                    if let Some(mut guard) = self.features.try_write() {
                        guard.push(Mutex::new((f)()));
                        // println!("feature pushed");
                        break;
                    } else {
                        // println!("exec: can't lock features");
                    }
                }

                // *self.can_read.write() = true;
            }

            pub fn load_future(resources:Arc<RwLock<HashMap<K, SyncWrapper<V>>>>, key:K) -> BoxFuture<'static, V> {
                ResourceFuture::new(resources,key).boxed()
            }
        }

        let mut executor = Executor::new(load_test_entity_sv);
        let mut executor_rc = Arc::new(RwLock::new(executor));


        struct ResourceFuture<K, V>  where
            K: Clone + Hash + Eq + Send + Sync + 'static + Debug,
            V: Send + Debug + 'static {
            x:usize,
            key:K,
            resources:Arc<RwLock<HashMap<K, V>>>
        }

        impl <K, V> ResourceFuture<K, V>  where
            K: Clone + Hash + Eq + Send + Sync + 'static + Debug,
            V: Send + Debug + 'static {
            pub fn new(resources:Arc<RwLock<HashMap<K, V>>>, key: K) -> Self {
                Self {
                    x:0,
                    key,
                    resources
                }
            }
        }

        impl <K, V>Future for ResourceFuture<K, SyncWrapper<V>>  where
            K: Clone + Hash + Eq + Send + Sync + 'static + Debug,
            V: Send + Debug + 'static{
            type Output = V;

            fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
                debug!("polling key={:?}", self.key);
                if let Some(resource) = self.resources.read().get(&self.key) {
                    debug!("polling key={:?} Ready", self.key);
                    Poll::Ready(resource.shallow_clone())
                } else {
                    debug!("polling key={:?} Pending ", self.key);
                    Poll::Pending
                }
            }
        }

        let exec_0 = Arc::clone(&executor_rc);
        let exec_1 = Arc::clone(&executor_rc);

        let wg = WaitGroup::new();
        let wg_0 = wg.clone();

        let data_done = Arc::new(RwLock::new(HashSet::new()));
        let data_done_rc = Arc::clone(&data_done);
        let data_done_rc_1 = Arc::clone(&data_done_rc);
        let num_items = 1000usize;

        let waiting_time = Arc::new(Mutex::new(0u128));
        let waiting_time_rc = Arc::clone(&waiting_time);


        let waiting_time_avg = Arc::new(Mutex::new(0u128));
        let waiting_time_avg_rc = Arc::clone(&waiting_time_avg);

        (0..num_items).for_each(|i| {
            loop {
                if let Some(mut exec_guard) = exec_1.try_write() {
                    // let mut write_guard = exec_guard.resources.write();
                    // write_guard.insert(format!("aaa_{i}"), format!("aaa_{i} OK"));
                    // write_guard.insert(format!("bbb_{i}"), format!("bbb_{i} OK"));
                    exec_guard.preload(vec![
                        format!("aaa_{i}"),
                        format!("bbb_{i}")
                    ]);
                    break
                } else {
                    debug!("executor_rc.try_write for preload failed");
                }
            }

        });


        pool_0.install(move || {

            (0..num_items).into_par_iter().for_each(|i| {

                let wg_0_i = wg_0.clone();
                let exec_rc = Arc::clone(&exec_0);
                let waiting_time_rc = Arc::clone(&waiting_time_rc);
                let waiting_time_avg_rc = Arc::clone(&waiting_time_avg_rc);

                let data_done_rc = Arc::clone(&data_done_rc);


                let mut exec_gurad = executor_rc.write();
                let arc = Arc::clone(&exec_gurad.resources);

                exec_gurad.exec(move || {
                    async move {
                        let t = Instant::now();
                        let key_aaa = format!("aaa_{i}");
                        let out_a = Executor::<String,TestEntity>::load_future(Arc::clone(&arc), key_aaa.clone()).await;

                        let key_bbb = format!("bbb_{i}");
                        let out_b = Executor::<String,TestEntity>::load_future(Arc::clone(&arc), key_bbb.clone()).await;

                        let mut waiting_time_guard = waiting_time_rc.lock();
                        let time = t.elapsed().as_millis();
                        *waiting_time_guard += time;



                        let mut rw_lock_write_guard = data_done_rc.write();
                        rw_lock_write_guard.insert(out_a);
                        rw_lock_write_guard.insert(out_b);
                        let n = rw_lock_write_guard.len() as u128;

                        let mut waiting_time_avg_guard = waiting_time_avg_rc.lock();

                        *waiting_time_avg_guard = (*waiting_time_avg_guard * (n-1) + time) / n;

                        drop(wg_0_i);

                        info!("key_aaa={key_aaa} key_bbb={key_bbb} done, time={time} waiting_time_rc={} waiting_time_avg_guard={}", waiting_time_guard, waiting_time_avg_guard);
                    }.boxed()
                });

            });
            drop(wg_0);
        });

        /*
        pool_1.spawn(move ||{
            (0..num_items).for_each(|i| {
                loop {
                    if let Some(mut exec_guard) = exec_1.try_write() {
                        // let mut write_guard = exec_guard.resources.write();
                        // write_guard.insert(format!("aaa_{i}"), format!("aaa_{i} OK"));
                        // write_guard.insert(format!("bbb_{i}"), format!("bbb_{i} OK"));
                        exec_guard.preload(vec![
                            format!("aaa_{i}"),
                            format!("bbb_{i}")
                        ]);
                        break
                    } else {
                        debug!("executor_rc.try_write for preload failed");
                    }
                }

            });

         });
         */


        wg.wait();
        while data_done_rc_1.read().len() != num_items * 2 {
            // println!("data_done={data_done_rc_1:?}");
            thread::sleep(Duration::new(1, 0));
        }
        debug!("data_done={data_done_rc_1:?}");
        // thread::sleep(Duration::new(3, 0));

        Ok(())
    }

}