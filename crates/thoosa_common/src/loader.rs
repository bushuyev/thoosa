use log::{Level, debug, log_enabled, trace};
use parking_lot::{Mutex, RwLock};
use rayon::ThreadPool;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::sync::Arc;

use crate::utils::tensor_mem_size;
use crossbeam_channel::Sender;
use futures::Future;
use futures::task::Context;
use futures::task::Poll;
use futures::{
    future::{BoxFuture, FutureExt},
};
use std::io::Write;
use std::ops::{AddAssign, Deref, DerefMut};
use std::pin::Pin;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU16};
use std::task::Waker;
use std::{fmt, thread};
use tch::{Device, Tensor};
use thoosa_shared::resource::Resource;

struct ResourcesBeingLoadedData<V>
where
    V: OfMemSize + Send + Debug + 'static,
{
    area: u64,
    future_ids: Vec<<LoadRunner<V> as KeyMaker>::Key>,
}

impl<V: OfMemSize + Send + Debug + 'static> ResourcesBeingLoadedData<V> {
    pub fn new(area: u64, future_id: Option<<LoadRunner<V> as KeyMaker>::Key>) -> Self {
        let future_ids = future_id
            .map(|future_id| vec![future_id])
            .unwrap_or(Vec::new());
        Self { area, future_ids }
    }

    pub fn add(&mut self, area: u64, future_id: Option<<LoadRunner<V> as KeyMaker>::Key>) {
        self.area |= area;
        if let Some(future_id) = future_id {
            self.future_ids.push(future_id);
        }
    }
}

pub struct Loader<V>
where
    V: OfMemSize + Send + Debug + 'static,
{
    loader_fn: Arc<Box<dyn Fn(&Resource) -> SyncWrapper<V> + Sync + Send>>,
    loader_pool: ThreadPool,

    resources: HashMap<Resource, SyncWrapper<V>>, //TODO replace Resource with ResourceId
    resources_being_loaded: HashMap<Resource, ResourcesBeingLoadedData<V>>,
}

impl<V> Loader<V>
where
    V: OfMemSize + Send + Debug + 'static,
{
    pub fn preload(
        self_arc: Arc<RwLock<Loader<V>>>,
        keys: &[Resource],
        cache_area: u64,
        done_sender: Sender<<LoadRunner<V> as KeyMaker>::Key>,
    ) {
        debug!("preload_vec: keys={keys:?} cache_area={cache_area:064b}");
        let mut self_write_guard = self_arc.write();

        for key in keys {
            if let Some(loaded) = self_write_guard.resources.get_mut(&key) {
                loaded.cache_areas |= cache_area;
                debug!(
                    "preload_vec: adding key={key:?} loaded.cache_areas={:064b}",
                    loaded.cache_areas
                );
                continue;
            }

            if let Some(being_loaded) = self_write_guard.resources_being_loaded.get_mut(&key) {
                // *being_loaded |= cache_area;
                being_loaded.add(cache_area, None);
                debug!(
                    "preload_vec: adding key={key:?} resources_being_loaded_guard.cache_area={:064b}",
                    being_loaded.area
                );
                continue;
            }

            self_write_guard
                .resources_being_loaded
                .insert(key.clone(), ResourcesBeingLoadedData::new(cache_area, None));

            debug!("preload_vec: adding key={key:?}");

            let loader = Arc::clone(&self_write_guard.loader_fn);
            let self_arc_pool = Arc::clone(&self_arc);

            let key = key.clone();
            let done_sender = done_sender.clone();

            self_write_guard.loader_pool.spawn(move || {
                Self::loader_task(cache_area, loader, self_arc_pool, &key, done_sender);
            });
        }
    }

    pub fn load_future(
        self_arc: Arc<RwLock<Loader<V>>>,
        key: Resource,
        cache_area: u64,
        future_id: <LoadRunner<V> as KeyMaker>::Key,
        done_sender: Sender<<LoadRunner<V> as KeyMaker>::Key>,
    ) -> BoxFuture<'static, V> {
        let mut self_lock = self_arc.write();

        if !self_lock.resources.contains_key(&key)
            && !self_lock.resources_being_loaded.contains_key(&key)
        {
            self_lock.resources_being_loaded.insert(
                key.clone(),
                ResourcesBeingLoadedData::new(cache_area, Some(future_id)),
            );

            debug!("Loader::load_future: adding key={key:?}");

            let loader_fn = Arc::clone(&self_lock.loader_fn);
            let self_arc_pool = Arc::clone(&self_arc);

            let key = key.clone();

            self_lock.loader_pool.spawn(move || {
                Self::loader_task(cache_area, loader_fn, self_arc_pool, &key, done_sender);
                debug!("Loader::load_future: sending done for future_id={future_id} key={key:?}");
            });
        } else if let Some(resources_being_loaded) = self_lock.resources_being_loaded.get_mut(&key)
        {
            resources_being_loaded.add(cache_area, Some(future_id));
        }

        ResourceFuture::new(Arc::clone(&self_arc), key).boxed()
    }

    fn loader_task<'a>(
        _cache_area: u64, //TODO remove cache_area - it's kept in resources_being_loaded?
        loader_fn: Arc<Box<dyn Fn(&'a Resource) -> SyncWrapper<V> + Sync + Send>>,
        loader_arc: Arc<RwLock<Loader<V>>>,
        key: &'a Resource,
        done_sender: Sender<<LoadRunner<V> as KeyMaker>::Key>,
    ) {
        let mut v = loader_fn(key);
        debug!(
            "Loader::loader_task: in loader_pool 0 current_thread={:?} key={:?}",
            thread::current().id(),
            key
        );

        let mut loader_guard = loader_arc.write();

        let data = loader_guard
            .resources_being_loaded
            .remove(&key)
            .expect(&format!("no data for {key:?}"));

        v.cache_areas = data.area;
        for future_id in data.future_ids {
            //futures may have been created for resources which were already being loaded or being preloaded
            done_sender.send(future_id).unwrap();
        }

        loader_guard.resources.insert(key.clone(), v);

        debug!(
            "Loader::loader_task: in loader_pool 1 current_thread={:?}",
            thread::current().id()
        );
    }

    pub fn purge_resources(loader_arc: Arc<RwLock<Loader<V>>>, keys: &[Resource], cache_area: u64) {
        let mut loader_guard = loader_arc.write();

        for key in keys {
            if let Some(v) = loader_guard.resources.get_mut(key) {
                debug!("purge_resources: cache_area={:064b}", cache_area);
                v.cache_areas = remove_mask(v.cache_areas, cache_area);
                if v.cache_areas == 0 {
                    loader_guard.resources.remove(key);
                    debug!("purge_resources: key={key:?} removed");
                } else {
                    debug!(
                        "purge_resources: key={key:?} kept, remaining v.cache_areas={:064b}",
                        v.cache_areas
                    );
                }
            }
        }
    }

    pub fn debug(loader_arc: &Arc<RwLock<Loader<V>>>, prefix: &str) {
        let loader_guard = loader_arc.read();
        let mut total_mem: HashMap<Device, usize> = HashMap::new();

        for (key, resource) in loader_guard.resources.iter() {
            let mem_size = resource.mem_size();
            debug!(
                "{prefix} loader debug: key={key:?} resource={resource:?} mem_size={mem_size:?}"
            );
            *total_mem.entry(mem_size.0).or_insert(0) += mem_size.1;
        }

        debug!("{prefix} loader debug: total_mem={total_mem:?}");
    }
}

pub trait OfMemSize {
    fn mem_size(&self) -> (Device, usize);
}

impl OfMemSize for Tensor {
    fn mem_size(&self) -> (Device, usize) {
        (self.device(), tensor_mem_size(self))
    }
}

pub trait KeyMaker {
    type Key: Eq + Hash + Debug;
    fn make_key(&self) -> Self::Key;
}

impl<V: OfMemSize + Send + Debug + 'static> KeyMaker for LoadRunner<V> {
    type Key = u16;

    fn make_key(&self) -> Self::Key {
        self.futures_counter.fetch_add(1, Relaxed) as Self::Key
    }
}

use std::sync::OnceLock;

static RAYON: OnceLock<rayon::ThreadPool> = OnceLock::new();

pub fn rayon_pool() -> &'static rayon::ThreadPool {
    RAYON.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(20) // tune!
            .thread_name(|i| format!("rayon-cpu-{i}"))
            .build()
            .expect("no rayon_pool")
    })
}

pub struct LoadRunner<V>
where
    V: OfMemSize + Send + Debug + 'static,
{
    #[allow(dead_code)]
    futures_pool: ThreadPool,
    #[allow(dead_code)]
    futures: Arc<RwLock<HashMap<<Self as KeyMaker>::Key, Mutex<BoxFuture<'static, ()>>>>>,
    new_futures_sender: Sender<(<Self as KeyMaker>::Key, Mutex<BoxFuture<'static, ()>>)>,
    done_futures_sender: Sender<<Self as KeyMaker>::Key>,
    stop: Arc<AtomicBool>,
    cache_areas: u64,
    loader: Arc<RwLock<Loader<V>>>,
    futures_counter: AtomicU16,
}

impl<V> LoadRunner<V>
where
    V: OfMemSize + Send + Debug + 'static,
{
    pub fn new<F>(
        loader: F,
        loader_threads: usize,
        futures_channel_bound: usize,
        num_future_threads: usize,
    ) -> Self
    where
        F: Fn(&Resource) -> SyncWrapper<V> + Sync + Send + 'static,
    {
        let futures: Arc<RwLock<HashMap<<Self as KeyMaker>::Key, Mutex<BoxFuture<()>>>>> =
            Arc::new(RwLock::new(HashMap::new()));

        let futures_rc = Arc::clone(&futures);

        //one for new_futures_receiver and another for done_futures_receiver
        let futures_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2 * num_future_threads)
            .spawn_handler(|thread_builder| {
                let b = thread::Builder::new();
                b.name(format!("{}/{:?}", "futures_pool", thread_builder.index()))
                    .spawn(|| thread_builder.run())?;

                Ok(())
            })
            .build()
            .expect("futures_pool");

        let stop = Arc::new(AtomicBool::new(false));

        let (new_futures_sender, new_futures_receiver) = crossbeam_channel::bounded::<(
            <Self as KeyMaker>::Key,
            Mutex<BoxFuture<()>>,
        )>(futures_channel_bound);
        let (done_futures_sender, done_futures_receiver) =
            crossbeam_channel::bounded::<<Self as KeyMaker>::Key>(futures_channel_bound);

        let new_futures_receiver_lock = Arc::new(new_futures_receiver);
        let done_futures_receiver_lock = Arc::new(done_futures_receiver);
        for p in 0..num_future_threads {
            let futures_for_add = Arc::clone(&futures);
            let futures_for_done = Arc::clone(&futures);

            let new_futures_receiver_lock = Arc::clone(&new_futures_receiver_lock);
            let done_futures_receiver_lock = Arc::clone(&done_futures_receiver_lock);

            futures_pool.spawn(move || {
                let mut cx = Context::from_waker(Waker::noop());

                while let Ok((future_id, fut)) = new_futures_receiver_lock.recv() {
                    debug!(
                        "LoadRunner::new: new_futures_receiver: p={p} got {}",
                        future_id
                    );

                    futures_for_add.write().insert(future_id, fut);

                    /*
                       LoadRunner polls futures only when external loaders finish their work to avoid polling all futures in a cycle.
                       The future just checks if resource is ready in its poll method, no other work is done in the future itself.
                       If a resource gets downloaded/built after is_pending calculated and before futures_for_push.write().insert
                       it gets stuck in Pending status. To avoid this, the future is first always inserted, then retrieved with read_guard.get(&key),
                       locked and held until key is removed if future is ready
                    */
                    let mut remove_immediately = false;
                    {
                        let read_guard = futures_for_add.read();
                        let mut guard = read_guard.get(&future_id).expect("just inserted").lock();

                        if guard.as_mut().poll(&mut cx).is_ready() {
                            remove_immediately = true;
                        }
                    }

                    if remove_immediately {
                        debug!(
                            "LoadRunner::new: new_futures_receiver: remove_immediately {}",
                            future_id
                        );
                        futures_for_add.write().remove(&future_id);
                    }
                }
            });

            futures_pool.spawn(move || {
                let mut cx = Context::from_waker(Waker::noop());

                while let Ok(future_id) = done_futures_receiver_lock.recv() {
                    debug!(
                        "LoadRunner::new: done_futures_receiver:  p={p} got {}",
                        future_id
                    );
                    std::io::stdout().flush().unwrap();

                    let mut map = futures_for_done.write();

                    if let std::collections::hash_map::Entry::Occupied(entry) =
                        map.entry(future_id)
                    {
                        debug!(
                            "LoadRunner::new: done_futures_receiver: getting {}",
                            future_id
                        );
                        // poll while *still* owning the entry
                        let ready = {
                            let fut_mx = entry.get();
                            let mut guard = fut_mx.lock();
                            guard.as_mut().poll(&mut cx).is_ready()
                        };

                        if ready {
                            entry.remove();
                            debug!(
                                "LoadRunner::new: done_futures_receiver: {} ready",
                                future_id
                            );
                        } else {
                            debug!(
                                "LoadRunner::new: done_futures_receiver: {} not ready",
                                future_id
                            );
                        }
                    }
                    // write‑lock is released
                    else {
                        debug!(
                            "LoadRunner::new: done_futures_receiver: no entry for {}",
                            future_id
                        );
                    }
                }
            });

            debug!("LoadRunner::new: spawned {p} futures_pool threads");
        }

        Self {
            futures_pool,

            futures: futures_rc,
            new_futures_sender,
            done_futures_sender,
            stop: Arc::clone(&stop),
            cache_areas: 0,

            loader: Arc::new(RwLock::new(Loader {
                loader_pool: rayon::ThreadPoolBuilder::new()
                    .num_threads(loader_threads)
                    .spawn_handler(|thread_builder| {
                        let b = thread::Builder::new();

                        b.name(format!("{}/{:?}", "loader_pool", thread_builder.index()))
                            .spawn(|| thread_builder.run())?;

                        Ok(())
                    })
                    .build()
                    .expect("loader_pool"),

                resources: HashMap::new(),
                resources_being_loaded: HashMap::new(),
                loader_fn: Arc::new(Box::new(loader)),
            })),

            futures_counter: AtomicU16::new(0),
        }
    }

    #[allow(dead_code)]
    fn stop(&mut self) {
        self.stop.store(true, Relaxed);
    }

    pub fn exec<F>(&mut self, make_future: F)
    where
        F: FnOnce(
            <Self as KeyMaker>::Key, //future_id
            Arc<RwLock<Loader<V>>>,
            Box<dyn FnOnce(&[Resource], u64) + Send>,
            Sender<<Self as KeyMaker>::Key>,
        ) -> BoxFuture<'static, ()>,
    {
        let loader_for_purge = Arc::clone(&self.loader);

        let future_id = self.futures_counter.fetch_add(1, Relaxed);

        debug!("exec_2: future_id={future_id}");

        let future = make_future(
            future_id,
            Arc::clone(&self.loader),
            Box::new(move |keys, cache_area| {
                Loader::purge_resources(loader_for_purge, keys, cache_area);

                trace!("exec: purge_resources done");
            }),
            self.done_futures_sender.clone(),
        );

        self.new_futures_sender
            .send((future_id, Mutex::new(future)))
            .expect("future_sender.send"); //TODO change to await

        trace!("exec: future {future_id} sent");
    }

    pub fn exec_scoped<'s, F>(&mut self, make_future: F)
    where
        F: FnOnce(
            <Self as KeyMaker>::Key,
            Arc<RwLock<Loader<V>>>,
            Box<dyn FnOnce(&[Resource], u64) + Send>,
            Sender<<Self as KeyMaker>::Key>,
        ) -> BoxFuture<'s, ()>,
    {
        let loader_for_purge = Arc::clone(&self.loader);

        let future_id = self.futures_counter.fetch_add(1, Relaxed);

        let future = make_future(
            future_id,
            Arc::clone(&self.loader),
            Box::new(move |keys, cache_area| {
                Loader::purge_resources(loader_for_purge, keys, cache_area);
            }),
            self.done_futures_sender.clone(),
        );

        let mut cx = Context::from_waker(Waker::noop());

        let mut pinned = Box::pin(future);
        loop {
            match pinned.as_mut().poll(&mut cx) {
                Poll::Ready(()) => {
                    break;
                }
                Poll::Pending => {
                    thread::yield_now();

                    if self.stop.load(Relaxed) {
                        break;
                    }
                }
            }
        }
    }

    pub fn debug(&self, prefix: &str) {
        if log_enabled!(Level::Debug) {
            Loader::debug(&self.loader, prefix);
        }
    }

    pub fn preload(&mut self, keys: &[Resource], cache_area: u64) {
        Loader::preload(
            Arc::clone(&self.loader),
            keys,
            cache_area,
            self.done_futures_sender.clone(),
        );
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

        debug!(
            "purge_cache_keys: 0 {area:064b} resources_guard.len()={}",
            loader_guard.resources.len()
        );
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

        debug!(
            "purge_cache: 1 {area:064b} resources_guard.len()={}",
            loader_guard.resources.len()
        );
    }

    pub fn purge_cache(&mut self, area: u64) {
        let mut loader_guard = self.loader.write();
        debug!(
            "purge_cache: 0 {area:064b} resources_guard.len()={}",
            loader_guard.resources.len()
        );

        loader_guard.resources.retain(|key, sw| {
            debug!("purge_cache: key={key:?} {:064b}", sw.cache_areas);

            sw.cache_areas = remove_mask(sw.cache_areas, area);
            debug!("purge_cache: key={key:?} {:064b}", sw.cache_areas);
            if sw.cache_areas == 0 {
                debug!("purge_cache: removing key={key:?}");
            }

            sw.cache_areas != 0
        });

        debug!(
            "purge_cache: 1 {area:064b} resources_guard.len()={}",
            loader_guard.resources.len()
        );
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

///the future never calls waker, the poll is done when done_futures_receiver receives completion notification from one of the loaders and/or at the end of top future
struct ResourceFuture<V>
where
    V: OfMemSize + Send + Debug + 'static,
{
    key: Resource,
    loader: Arc<RwLock<Loader<V>>>,
}

impl<V> ResourceFuture<V>
where
    V: OfMemSize + Send + Debug + 'static,
{
    pub fn new(loader: Arc<RwLock<Loader<V>>>, key: Resource) -> Self {
        Self { key, loader }
    }
}

impl<V> Future for ResourceFuture<V>
where
    V: OfMemSize + Send + Debug + 'static,
{
    type Output = V;

    fn poll(self: Pin<&mut Self>, _ctx: &mut Context<'_>) -> Poll<Self::Output> {
        debug!("polling key={:?}", self.key);

        if let Some(resource) = self.loader.read().resources.get(&self.key) {
            debug!("polling key={:?} Ready", self.key);

            Poll::Ready(resource.shallow_clone())
        } else {
            debug!("polling key={:?} Pending ", self.key);
            Poll::Pending
        }
    }
}

#[allow(unpredictable_function_pointer_comparisons)]
#[derive(Debug, Hash, PartialEq, Eq)]
pub struct SyncWrapper<V>
where
    V: OfMemSize + Send + Debug,
{
    v: V,
    shallow_clone_fn: fn(&V) -> V,
    cache_areas: u64,
}

impl<V: OfMemSize + Send + Display + Debug> Display for SyncWrapper<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.v, f)
    }
}

impl SyncWrapper<Tensor> {
    pub fn wrap_tensor(t: Tensor) -> Self {
        Self {
            v: t,
            shallow_clone_fn: |b| b.shallow_clone(),
            cache_areas: 0,
        }
    }
}

fn assert_sync<T: Sync>() {}

impl<V> SyncWrapper<V>
where
    V: OfMemSize + Send + Debug,
{
    pub fn wrap(v: V, shallow_clone_fn: fn(&V) -> V) -> Self
    where
        V: Sync,
    {
        assert_sync::<V>();

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
unsafe impl<V> Sync for SyncWrapper<V> where V: OfMemSize + Send + Debug {}

impl<V> Deref for SyncWrapper<V>
where
    V: OfMemSize + Send + Debug,
{
    type Target = V;
    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl<V> DerefMut for SyncWrapper<V>
where
    V: OfMemSize + Send + Debug,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.v
    }
}

impl From<Tensor> for SyncWrapper<Tensor> {
    fn from(t: Tensor) -> Self {
        SyncWrapper::wrap_tensor(t)
    }
}

impl AddAssign for SyncWrapper<Tensor> {
    fn add_assign(&mut self, rhs: Self) {
        self.v += rhs.v;
    }
}

#[cfg(test)]
#[allow(unused_imports, unused_variables, dead_code, unused_mut)]
mod tests {
    use std::collections::HashSet;
    use std::fmt::Debug;
    use std::future::Future;
    use std::hash::Hash;
    use std::pin::Pin;

    use crate::loader::{KeyMaker, LoadRunner, Loader, OfMemSize};
    use crossbeam_channel::Sender;
    use crossbeam_utils::sync::WaitGroup;
    use futures::FutureExt;
    use log::LevelFilter::{Info, Trace};
    use log::{LevelFilter, debug, set_max_level};
    use parking_lot::RwLock;
    use rand::Rng;
    use std::marker::PhantomData;
    use std::sync::Arc;
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64};
    use std::task::{Context, Poll};
    use std::thread;
    use std::time::{Duration, Instant};
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::Float;
    use tch::{Device, Tensor};
    use thoosa_shared::ThoError;
    use thoosa_shared::resource::{Resource, ResourceId};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[derive(Debug, Hash, PartialEq, Eq)]
    struct TestEntity {
        key: Resource,
    }

    impl OfMemSize for TestEntity {
        fn mem_size(&self) -> (Device, usize) {
            (Cpu, 1)
        }
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
        thread::sleep(Duration::new(
            1,
            rand::rng().random_range(1000u32..3_000_000u32),
        ));
        // thread::sleep(Duration::new(0, 5e8 as u32));

        super::SyncWrapper::wrap(TestEntity { key: key.clone() }, |e| TestEntity {
            key: e.key.clone(),
        })
    }


    #[test]
    fn test_first_free_bit() {
        assert_eq!(
            0b0000000000000000000000000000000000000000000000000000000000000001,
            super::first_free_bit(
                0b0000000000000000000000000000000000000000000000000000000000000000
            )
        );

        assert_eq!(
            0b0000000000000000000000000000000000000000000000000000000000000010,
            super::first_free_bit(
                0b0000000000000000000000000000000000000000000000000000000000000001
            )
        );

        assert_eq!(
            0b0000000000000000000000000000000000000000000000000000000000000100,
            super::first_free_bit(
                0b0000000000000000000000000000000000000000000000000000000000000011
            )
        );

        assert_eq!(
            0b0000000000000000000000000000000000000000000000000000000000000100,
            super::first_free_bit(
                0b0000000000000000000000000000000000000000000000000000000000001011
            )
        );

        assert_eq!(
            0b1000000000000000000000000000000000000000000000000000000000000000,
            super::first_free_bit(
                0b0111111111111111111111111111111111111111111111111111111111111111
            )
        );

        assert_eq!(
            0b0000000000000000000000000000000000000000000000000000000000000000,
            super::first_free_bit(
                0b1111111111111111111111111111111111111111111111111111111111111111
            )
        );
    }

    #[test]
    fn test_remove_mask() {
        assert_eq!(0b00000000, super::remove_mask(0b00000001, 0b000000001));
        assert_eq!(0b00000000, super::remove_mask(0b00000000, 0b000000001));
        assert_eq!(0b00000010, super::remove_mask(0b00000010, 0b000000001));
        assert_eq!(0b11111101, super::remove_mask(0b11111111, 0b000000010));
    }
}
