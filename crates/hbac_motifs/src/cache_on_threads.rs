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

                guard.replace(loader_in_thread(&key_in_thread))

            });
        }

    }

    pub fn is_empty(&self) -> bool {
        self.data_loaded.is_empty()
    }

}


#[cfg(test)]
mod tests {
    use std::collections::{HashMap, VecDeque};
    use std::fmt::Debug;
    use std::future::Future;
    use std::hash::Hash;
    use std::io::Write;
    use std::pin::Pin;
    use std::sync::{Arc, RwLock};
    use std::sync::mpsc::Receiver;
    use std::task::{Context, Poll};
    use std::thread;
    use std::time::Duration;
    use futures::AsyncWriteExt;
    use futures::task::SpawnExt;
    use log::debug;
    use rayon::ThreadPool;
    use hbac_shared::HbacError;
    use crate::cache_on_threads::CacheOnThreads;

    #[ctor::ctor]
    fn init() {
        hbac_shared::init_test_logging();
    }

    #[derive(Debug)]
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

    #[test]
    fn test_0() -> Result<(), HbacError>{
        use rayon::iter::ParallelIterator;
        use rayon::iter::IntoParallelIterator;
        use std::sync::mpsc;
        use std::sync::mpsc::{Sender, Receiver};
        use std::rc::Rc;
        use std::sync::Mutex;

        let num_threads = std::thread::available_parallelism()?;
        println!("num_threads={num_threads:?}");


        struct Loader<K, V>
        where
            K: Hash + Eq + Send + Sync + 'static + Debug,
            V: Send + Debug + 'static,
        {
            loader_pool:ThreadPool,
            sender_receiver_up:(Sender<K>, Receiver<K>),
            sender:Arc<Sender<(K, V)>>,

            data_loaded:Arc<RwLock<HashMap<K, Mutex<V>>>>,
            loader:Arc<Box<dyn Fn(&K)->V + Sync + Send>>,
        }

        impl<K, V> Loader<K, V>
        where
            K: Hash + Eq + Send + Sync + 'static + Debug,
            V: Send + Debug + 'static,
        {
            fn preload(&mut self, keys: Vec<K>) -> Result<(), HbacError> {
                for key in keys {


                    let loader = Arc::clone(&self.loader);
                    let data_loaded = Arc::clone(&self.data_loaded);
                    let sender_down = Arc::clone(&self.sender);

                    self.loader_pool.spawn(move ||{
                        let v = loader(&key);
                        println!("pool 0: current_thread={:?}", thread::current().id());
                        println!("v={v:?}");
                        // sender_down.send((key, r));//Tensors can't sync
                        data_loaded.write().expect("data_loaded write").insert(key, Mutex::new(v));

                    });
                };

                Ok(())
            }

            pub fn new<F>(loader:F) -> Result<Self, HbacError>
                where F:Fn(&K) -> V + Sync + Send + 'static {
                let (sender, receiver) = mpsc::channel();

                // let mut data_loaded = Arc::new(RwLock::new(HashMap::new()));
                // let data_loaded_arc = Arc::clone(&data_loaded);
                // rayon::spawn(move ||{
                //     while let Ok((k, v)) = receiver.recv() {
                //         println!("got k={k:?} {v:?}");
                //         data_loaded_arc.write().expect("data_loaded write").insert(k, Mutex::new(v));
                //     }
                // });

                Ok(Self {
                    loader:Arc::new(Box::new(loader)),
                    sender_receiver_up: mpsc::channel(),
                    sender: Arc::new(sender),
                    loader_pool:rayon::ThreadPoolBuilder::new().num_threads(2).spawn_handler(|thread_builder| {

                        let mut b = std::thread::Builder::new();
                        b.name(format!("{}/{:?}", "pool_1", thread_builder.index()))
                        .spawn(|| thread_builder.run())?;

                        Ok(())
                    }).build()?,
                    data_loaded:Arc::new(RwLock::new(HashMap::new()))
                })
            }
        }

        let mut loader = Loader::new(load_test_entity)?;

        loader.preload((0..10).map(|i|format!("aaa_{i}")).collect());
        println!("xxxxxxxxxxxxxxx");
        loader.preload((0..10).map(|i|format!("bbb_{i}")).collect());
        println!("zzzzzzzzzzzzzzz");



        thread::sleep(Duration::new(3, 0));

       /*
        let pool_0 = rayon::ThreadPoolBuilder::new().num_threads(2).spawn_handler(|thread_builder| {
            let mut b = std::thread::Builder::new();
            b.name(format!("{}/{:?}", "pool_0", thread_builder.name())).spawn(|| thread_builder.run())?;

            Ok(())
        }).build()?;

        pool_0.spawn(||{
            println!("span_0");
            thread::sleep(Duration::new(1, 0));
        });
        println!("0000000000");
        pool_0.spawn(||{
            println!("span_1");
            thread::sleep(Duration::new(1, 0));
        });
        println!("1111111111");
        thread::sleep(Duration::new(3, 0));


        let pool_1 = rayon::ThreadPoolBuilder::new().num_threads(3).spawn_handler(|thread_builder| {
            let mut b = std::thread::Builder::new();
            b.name(format!("{}/{:?}", "pool_1", thread_builder.name())).spawn(|| thread_builder.run())?;

            Ok(())
        }).build()?;


        rayon::join(
            || pool_0.install(|| {
                (0..10).into_par_iter().for_each(|v| {
                    thread::sleep(Duration::new(0, 100));
                    println!("pool 0: current_thread={:?}", thread::current().name());
                    std::io::stdout().flush().unwrap();
                });
            }),
            || pool_1.install(|| {
                (0..10).into_par_iter().for_each(|v| {
                    thread::sleep(Duration::new(0, 100));
                    println!("pool 1: current_thread={:?}", thread::current().name());
                    std::io::stdout().flush().unwrap();
                });
            }),
        );*/


        Ok(())
    }

}