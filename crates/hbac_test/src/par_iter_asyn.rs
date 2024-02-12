use std::cell::RefCell;
use std::collections::HashMap;
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

// struct LoaderFuture <K, V> where K:Hash + Eq{
struct LoaderFuture<K, V> where K: Hash + Eq + Send + Sync + 'static, V: Send + 'static {
    waker: Arc<Mutex<Option<Waker>>>,
    data: Arc<Mutex<Option<V>>>,
    pd:PhantomData<K>
}

impl<K, V> LoaderFuture<K, V> where K: Hash + Eq + Send + Sync + 'static, V: Send + 'static {
    pub fn new(loader: fn(&K) -> V, key:K) -> Self {
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

struct CacheOnThreads<K, V> where K:Hash + Eq + Send + Sync+ 'static + Debug, V:Send+'static{
    data_loaded:HashMap<K, V>,
    data_being_loaded:Arc<Mutex<HashMap<K, Arc<Mutex<()>>>>>,
    loader:fn(&K)->V,
    read_mx:Arc<Mutex<()>>,
    load_mx:Arc<Mutex<()>>
}

impl <K, V> CacheOnThreads<K, V> where K:Hash + Eq + Clone + Send + Sync + 'static + Debug, V:Send +'static{
    pub fn new(loader: fn(&K) -> V) -> Self {
        Self {
            data_loaded: HashMap::new(),
            data_being_loaded: Arc::new(Mutex::new(HashMap::new())),
            loader,
            read_mx:Arc::new(Mutex::new(())),
            load_mx:Arc::new(Mutex::new(()))
        }
    }

    pub fn load_blocking(&mut self, key:K) {
        // self.data_loaded.entry(key.clone()).or_insert_with(||(self.loader)(&key))

        // let guard = self.read_mx.lock().expect("data_loaded.lock");
        // if let Some(v) = self.data_loaded.get(&key) {
        //     drop(guard);
        //     return v;
        // }

        // let guard = self.load_mx.lock().expect("data_loaded.lock");
        // if let Some(v) = self.data_loaded.get(&key) {
        //     drop(guard);
        //     return v;
        // }



        if self.data_loaded.contains_key(&key) {
            println!("{:?} is already loaded", key);
            return;
        }
        println!("loading {:?}", key);

        self.data_loaded.insert(
            key.clone(),
            block_on(LoaderFuture::new(self.loader.clone(), key.clone()))
        );


        // drop(guard);

        // self.read_mx.lock().expect("data_loaded.lock");
        // self.data_loaded.get(&key).expect("no data")
    }

    pub fn get_loaded(&self, key:&K) -> Option<&V> {
        self.data_loaded.get(key)
    }

    pub fn load_and_get(&mut self, k:K) -> Option<&V> {
        let data_being_loaded_lock = Arc::clone(&self.data_being_loaded);
        let mut data_being_loaded_guard = data_being_loaded_lock.lock().expect("data_being_loaded");
        let entry_mutex = data_being_loaded_guard.entry(k.clone()).or_insert_with(|| Arc::new(Mutex::new(())));
        let entry_lock_clone = Arc::clone(entry_mutex);

        entry_lock_clone.lock().expect("wakers_mx");

        drop(data_being_loaded_guard);

        self.load_blocking(k.clone());
        self.get_loaded(&k)
    }



    fn _load_in_future(&mut self, key: &K) -> V {

        if let Ok(_) = self.read_mx.lock() {

        }

        // thread::spawn(move || {
        //
        //     (self.loader)(&key);
        //
        // });

        (self.loader)(&key)
    }
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
            println!("7 ready");

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

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::Duration;
    use futures::executor::block_on;
    use rayon::prelude::*;
    use crate::par_iter_asyn::{CacheOnThreads, load_it};


    #[derive(Debug)]
    struct TestEntity {
        key:String
    }

    fn load_test_entity(key:&String)->Box<TestEntity>{
        thread::sleep(Duration::new(10, 0));
        Box::new(TestEntity {key:key.clone()})
    }


    #[test]
    fn test_it(){

        rayon::ThreadPoolBuilder::new().num_threads(10).build_global().unwrap();

        let x:i32 = (0..10).into_par_iter().map(| v_0|  {

            println!("1.  v_0={}", v_0);

            let v = block_on(load_it(v_0));


            println!("2. v={}", v);

            v

        }).sum();

        println!("x={:?}", x);

        // println!("x={:?}",block_on(load_it(3)));
    }

    #[test]
    fn test_cache_on_threads(){
        let mut cache = CacheOnThreads::new(load_test_entity);
        cache.load_and_get("Key1".into());
        cache.load_and_get("Key1".into());
        cache.load_and_get("Key1".into());
        cache.load_and_get("Key1".into());
        cache.load_and_get("Key1".into());
        // let e_2 = cache.load_and_get("Key1".into()).map(|e| e as *const _ as *const u32).unwrap();
        let e_1 = cache.load_and_get("Key1".into()); //.map(|e| e as *const _ as *const u32).unwrap();

        // let xx = e_1.and_then(|e_1| e_2.map(|e_2|(e_1 as *const _, e_2 as *const _))).unwrap();

        // let p_1 = e_1.unwrap() as *const _;
        // let p_2 = e_2.unwrap() as *const _;
        // println!("ee={:?}", e_2 == e_1);
        println!("e_1={:?}", e_1);
        // let i:&TestEntity = unsafe { &*(e_1 as *const _ as *const &TestEntity)};
        // println!("e_1={:?}", i);

    }
}