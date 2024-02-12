use std::collections::HashMap;
use std::future::Future;
use std::hash::Hash;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};
use std::thread;
use std::time::Duration;
use futures::executor::block_on;

// struct LoaderFuture <K, V> where K:Hash + Eq{
struct LoaderFuture {
    wakers: Arc<Mutex<Vec<Waker>>>
}

impl LoaderFuture {
    pub fn new(wakers: Arc<Mutex<Vec<Waker>>>) -> Self {
        Self {wakers}
    }
}

// impl <K, V> Future for LoaderFuture <K, V> where K:Hash + Eq{
impl Future for LoaderFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        todo!()
    }
}

struct CacheOnThreads<K, V> where K:Hash + Eq + Send + Sync+ 'static, V:Send+'static{
    data_loaded:HashMap<K, V>,
    data_being_loaded:Arc<Mutex<HashMap<K, Arc<Mutex<Vec<Waker>>>>>>,
    loader:fn(&K)->V,
    read_mx:Arc<Mutex<()>>,
    load_mx:Arc<Mutex<()>>
}

impl <K, V> CacheOnThreads<K, V> where K:Hash + Eq + Clone + Send + Sync + 'static, V:Send +'static{
    pub fn new(loader: fn(&K) -> V) -> Self {
        Self {
            data_loaded: HashMap::new(),
            data_being_loaded: Arc::new(Mutex::new(HashMap::new())),
            loader,
            read_mx:Arc::new(Mutex::new(())),
            load_mx:Arc::new(Mutex::new(()))
        }
    }

    pub fn load_blocking(&mut self, key:K) -> &V{
        // self.data_loaded.entry(key.clone()).or_insert_with(||(self.loader)(&key))

        let guard = self.read_mx.lock().expect("data_loaded.lock");
        if let Some(v) = self.data_loaded.get(&key) {
            drop(guard);
            return v;
        }

        let guard = self.load_mx.lock().expect("data_loaded.lock");
        if let Some(v) = self.data_loaded.get(&key) {
            drop(guard);
            return v;
        }

        let mut data_being_loaded_guard = self.data_being_loaded.lock().expect("data_being_loaded");
        let wakers_mx = data_being_loaded_guard.entry(key.clone()).or_insert_with(|| Arc::new(Mutex::new(vec![])));

        let wakers_lock = wakers_mx.lock().expect("wakers_mx");
        let is_wakers_empty = wakers_lock.is_empty();

        let feature = LoaderFuture::new(Arc::clone(wakers_mx));

        if is_wakers_empty {
            let thread_data_being_loaded = Arc::clone(&self.data_being_loaded);
            let loader = self.loader.clone();

            thread::spawn(move || {

                let v = (loader)(&key);

                // let guard = self.load_mx.lock().expect("data_loaded.lock");
                // if let Some(v) = self.data_loaded.get(&key) {
                //
                // }

                if let Some(mut wakers) = thread_data_being_loaded.lock().expect("thread_data_being_loaded thread").remove(&key) {
                    for waker in &*wakers.lock().expect("wakers in future") {
                        waker.wake_by_ref();
                    };
                }
            });
        }
        drop(wakers_lock);
        drop(guard);

        block_on(feature);



        todo!()
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
    use futures::executor::block_on;
    use rayon::prelude::*;
    use crate::par_iter_asyn::{CacheOnThreads, load_it};


    #[derive(Debug)]
    struct TestEntity {
        key:String
    }

    fn load_test_entity(key:&String)->TestEntity{
        TestEntity {key:key.clone()}
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

        let e_1 = cache.load_blocking("Key1".into());

        println!("e_1={:?}", e_1);

    }
}