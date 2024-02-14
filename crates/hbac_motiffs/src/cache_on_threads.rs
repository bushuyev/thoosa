use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::future::Future;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};
use std::thread;
use std::time::Duration;
use futures::executor::block_on;
use log::debug;
use tch::Tensor;
use std::rc::Rc;

// struct LoaderFuture <K, V> where K:Hash + Eq{
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

// impl <K, V> Future for LoaderFuture <K, V> where K:Hash + Eq{
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

    data_being_loaded:Arc<Mutex<HashMap<K, Arc<Mutex<Option<V>>>>>>,

    loader:Arc<Box<dyn Fn(&K)->V + Sync + Send>>,
    read_mx:Arc<Mutex<()>>,
    load_mx:Arc<Mutex<()>>,
    keys_fifo: VecDeque<K>
}

impl <K, V> CacheOnThreads<K, V> where K:Hash + Eq + Clone + Send + Sync + 'static + Debug, V:Send +'static{

    pub fn new<F>(loader: F) -> Self where F:Fn(&K) -> V + Sync + Send + 'static{
        Self {
            data_loaded: HashMap::new(),
            data_being_loaded: Arc::new(Mutex::new(HashMap::new())),
            loader:Arc::new(Box::new(loader)),
            read_mx:Arc::new(Mutex::new(())),
            load_mx:Arc::new(Mutex::new(())),
            keys_fifo: VecDeque::new()
        }
    }

    fn load_blocking(&mut self, key:K) ->bool {

        self.data_loaded.insert(
            key.clone(),
            block_on(LoaderFuture::new(Arc::clone(&self.loader), key.clone()))
        );
        debug!("{:?} was loaded", key);

        return true;
    }

    pub fn get_loaded(&self, key:&K) -> Option<&V> {
        self.data_loaded.get(key)
    }

    fn update_loaded(&mut self) {
        let mut data_being_loaded_guard = self.data_being_loaded.lock().expect("data_being_loaded");
        if self.keys_fifo.len() > 3 { //TODO 3
            if let Some(key_to_remove) = self.keys_fifo.pop_front() {
                debug!("removing {:?} ", key_to_remove);
                let entry_mutex = data_being_loaded_guard.entry(key_to_remove.clone()).or_insert_with(|| Arc::new(Mutex::new(None)));
                entry_mutex.lock().expect("entry_mutex for key_to_remove");

                self.data_loaded.remove(&key_to_remove);
                data_being_loaded_guard.remove(&key_to_remove);
            }
        }

        for (key, entry_mutex) in data_being_loaded_guard.iter() {
            let mut entry_mutex_guard = entry_mutex.lock().expect("");

            if let Some(data) = entry_mutex_guard.take(){
                debug!("update_loaded: {:?} was taken from preloaded", key);
                self.data_loaded.insert(key.clone(), data);

            }
        }
    }

    pub fn load_blocking_and_get(&mut self, k:&K) -> Option<&V> {

        let mut data_being_loaded_guard = self.data_being_loaded.lock().expect("data_being_loaded");
        debug!("loading {:?} self.keys_fifo()={:?}", k, self.keys_fifo);
        if !self.data_loaded.contains_key(k) {

            if self.keys_fifo.len() > 3 { //TODO 3
                if let Some(key_to_remove) = self.keys_fifo.pop_front() {
                    debug!("removing {:?} ", key_to_remove);
                    let entry_mutex = data_being_loaded_guard.entry(key_to_remove.clone()).or_insert_with(|| Arc::new(Mutex::new(None)));
                    entry_mutex.lock().expect("entry_mutex for key_to_remove");

                    self.data_loaded.remove(&key_to_remove);
                    data_being_loaded_guard.remove(&key_to_remove);
                }
            }
            self.keys_fifo.push_back(k.clone());

            let entry_mutex = data_being_loaded_guard.entry(k.clone()).or_insert_with(|| Arc::new(Mutex::new(None)));
            let entry_mutex_clone = Arc::clone(entry_mutex);
            let mut entry_mutex_guard = entry_mutex_clone.lock().expect("");

            if let Some(data) = entry_mutex_guard.take(){
                debug!("{:?} was taken from preloaded", k);
                self.data_loaded.insert(k.clone(), data);

            } else {
                drop(data_being_loaded_guard);

                self.load_blocking(k.clone());
            };


        } else {
            if let Some((i, e)) = self.keys_fifo.iter().enumerate().find(|(i, e)| e == &k) {
                self.keys_fifo.remove(i);
            }
            self.keys_fifo.push_back(k.clone());
            debug!("{:?} is already loaded", k);
        }
        
        self.get_loaded(&k)
    }

    pub fn load_blocking_preload_and_get(&mut self, k:&K, preload:&[K]) -> Option<&V> {

        self.update_loaded();

        self.load_no_blocking(preload);

        self.load_blocking_and_get(k)
    }

    pub fn load_no_blocking(&mut self, keys:&[K]) {
        debug!("load_no_blocking: keys={:?}", keys);

        for key in keys {
            if self.data_loaded.contains_key(key) {//TODO lock?
                continue
            }

            let data_being_loaded_in_thread = Arc::clone(&self.data_being_loaded);

            let key_in_thread = key.clone();
            let loader_in_thread = self.loader.clone();
            thread::spawn(move || {
                let mut data_being_loaded_guard = data_being_loaded_in_thread.lock().expect("data_being_loaded");

                let entry_mutex = data_being_loaded_guard.entry(key_in_thread.clone()).or_insert_with(|| Arc::new(Mutex::new(None)));

                let entry_lock_clone = Arc::clone(entry_mutex);

                drop(data_being_loaded_guard);
                let mut guard  =entry_lock_clone.lock().expect("wakers_mx");

                debug!("load_no_blocking: key_in_thread={:?} loaded", key_in_thread);

                guard.replace(loader_in_thread(&key_in_thread))

            });
        }

    }

}


#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll};
    use std::thread;
    use std::time::Duration;
    use futures::executor::block_on;
    use log::debug;
    use rayon::prelude::*;
    use crate::cache_on_threads::{CacheOnThreads};


    #[derive(Debug)]
    struct TestEntity {
        key:String
    }

    fn load_test_entity(key:&String)->Box<TestEntity>{
        thread::sleep(Duration::new(3, 0));
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

    async fn load_it(i:i32) -> i32 {
        TestFuture::new(i).await
    }


    #[test]
    fn test_it(){

        rayon::ThreadPoolBuilder::new().num_threads(10).build_global().unwrap();

        let x:i32 = (0..10).into_par_iter().map(| v_0|  {

            debug!("1.  v_0={}", v_0);

            let v = block_on(load_it(v_0));


            debug!("2. v={}", v);

            v

        }).sum();

        debug!("x={:?}", x);

        // debug!("x={:?}",block_on(load_it(3)));
    }

    #[test]
    fn test_cache_on_threads(){
        let mut cache = CacheOnThreads::new(load_test_entity);
        let e_1_0= cache.load_blocking_and_get(&"Key1".into()).unwrap().key.clone();
        cache.load_blocking_and_get(&"Key1".into());
        cache.load_blocking_and_get(&"Key2".into());
        cache.load_blocking_and_get(&"Key3".into());
        cache.load_blocking_and_get(&"Key4".into());
        cache.load_blocking_and_get(&"Key5".into());
        cache.load_blocking_and_get(&"Key2".into());
        cache.load_blocking_and_get(&"Key1".into());

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
        let mut cache = CacheOnThreads::new(load_test_entity);
        cache.load_no_blocking(&["Key1".into(), "Key2".into()]);

        cache.load_blocking_and_get(&"Key1".into());
        cache.load_blocking_and_get(&"Key2".into());

        let e_1 = cache.get_loaded(&"Key1".into()); //.map(|e| e as *const _ as *const u32).unwrap();
        let e_2 = cache.get_loaded(&"Key2".into()); //.map(|e| e as *const _ as *const u32).unwrap();

        assert_eq!(e_1.unwrap().key, "Key1");
        assert_eq!(e_2.unwrap().key, "Key2");

    }
}