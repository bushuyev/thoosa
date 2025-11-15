#![allow(unused_parens)]
#![allow(async_fn_in_trait)]
use crate::common::runnable::{AsyncStage, Stage};
use crate::common::*;
#[double]
use crate::global_ctx::GlobalCtx;
use async_trait::async_trait;
use dashmap::{DashMap, Entry};
use futures_util::{FutureExt, StreamExt, TryStreamExt};
use futures_util::future::{BoxFuture, Shared, join_all, try_join_all};
use log::{debug, trace};
use mockall_double::double;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::mem;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use thoosa_shared::MyError;
use tokio::sync::{Notify, RwLockReadGuard};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::traits::{Data, Select};
use thoosa_with_data::with_data;

#[async_trait]
pub trait Cached: AsyncStage
where
    for <'i> Self::In<'i>: Data + CopyToDevice + WithId + Send + Sync,
    <Self as Stage>::Out: Data + CopyToDevice + WithId + Send + Sync + 'static
{
    async fn cache() -> &'static DashMap<
        <<Self as Stage>::Out as WithId>::ID,
        Shared<BoxFuture<'static, Result<Arc<DevicesWrap<Self::Out>>, ThoosaError>>>,
    >
    where
        Self: Sized,
        Self::Out: Data + Sync + Send;

    async fn run_cached(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError>
    where
        Self: Sized + OutId,
        for<'a> <Self as Stage>::In<'a>: Send + Sync,
        <Self as Stage>::Out: Send + Sync,
        <<Self as Stage>::Out as WithId>::ID: 'static,
        <Self as Stage>::Out: 'static,
        Self::Out: Data // <Self as Runnable>::Out:Data<Sub= <Self as Runnable>::Out>+CopyToDevice,
    {
        let key = self.out_id(WithId::id(input.as_ref()));
        let (cache_area, device_idx) = {
            let guard = ctx.read();
            (guard.cache_area(), guard.device_idx())
        };

        debug!("run_cached: key={key:?} device_idx={device_idx}");

        let cache = Self::cache().await;

        self.on_cached(input.as_ref(), &ctx);

        GlobalCtx::global().do_age();

        match cache.entry(key) {
            Entry::Occupied(slot) => {
                let shared = slot.get().clone();
                drop(slot);
                shared
                    .map(|res| {
                        if let Ok(data) = &res {
                            data.add_cache_area(cache_area);
                        }
                        res
                    })
                    .await
            }
            Entry::Vacant(slot) => {
                let fut = Arc::clone(&self)
                    .run(input, ctx)
                    .map(move |r| {
                        r.map(|v| {
                            // assert_eq!(v.device(), device_idx); - loaded to CPU, copied to GPU when used
                            // Arc::new(DataCacheWrapper::new(
                            //     v,
                            //     device_idx,
                            //     GlobalCtx::global().device_count(),
                            //     cache_area,
                            // ))v
                            v
                        })
                    })
                    .boxed()
                    .shared();

                slot.insert(fut.clone()); //slot consumed, fut.await doesn't block it

                fut.await
            }
        }
    }

    async fn purge_gpu_cache_area(
        keys: Vec<<Self::Out as WithId>::ID>,
        cache_area: u64,
        device_idx: usize,
    ) -> Result<(), ThoosaError>
    where
        Self: Sized + OutId,
        Self::Out: Data + Sync + Send,
    {
        debug!("Cachable::purge_gpu_cache_area: 0  cache_area={cache_area} device_idx={device_idx} keys={keys:?}");

        assert_ne!(device_idx, 0); //don't purge cpu

        let map = Self::cache().await;

        debug!("Cachable::purge_gpu_cache_area: 0  cache_area={cache_area} device_idx={device_idx} got map");

        for key in keys {
            // let key_device_mutex = Self::key_in_device_lock(&key, device_idx);
            // let key_guard = key_device_mutex.lock().await;
            debug!("Cachable::purge_gpu_cache_area: 1 key={key:?} cache_area={cache_area} device_idx={device_idx}");

            if let Some(value) = map.get(&key) {
                let shared_fut = value.value().clone();
                if shared_fut.peek().is_some() {//to avoid locking on itself
                    let data: Arc<DevicesWrap<Self::Out>> = shared_fut.await?;
                    let new_mask = data.remove_cache_area(cache_area);

                    debug!("Cachable::purge_gpu_cache_area: 2 key={key:?} new_mask={new_mask} ");
                    if new_mask == 0 {
                        data.dump_to_cpu(device_idx).await?;
                        // drop(key_guard);
                        debug!("Cachable::purge_gpu_cache_area: 3  key={key:?} done");
                    }
                }
            } else {
                debug!("Cachable::purge_gpu_cache_area: 4 key={key:?} {cache_area} device_idx={device_idx} not found"); //warning
            }
        }

        debug!("Cachable::purge_gpu_cache_area: 5 cache_area={cache_area} device_idx={device_idx}");

        Ok(())
    }

}

//when hidden behind a trait, impls for plain Ids and Data cause https://github.com/rust-lang/rust/issues/100013 so opting to enum
#[derive(Debug)]
pub struct DevicesWrap<O>//TODO rename to DeviceDataWrapper
where
    O: WithId + Data + CopyToDevice + Send + Sync,
{
    id: <O as WithId>::ID,
    cached_data: Box<[tokio::sync::RwLock<Option<O>>]>,
    inflights: Box<[tokio::sync::Mutex<Option<Arc<tokio::sync::Notify>>>]>,
    cache_area: AtomicU64,
    size: O::Metrics,
    mem_size: u64,
    //contains cache age ( GlobalCtx::cache_age - increated when something is being cached from Cachable::run_cached). The smaller the value the longer it was in the cpu
    on_cpu_since: parking_lot::RwLock<Option<u64>>,
    pd: PhantomData<O>,
}


impl <O> Input for DevicesWrap<O>
where
    O: WithId + Data + CopyToDevice + Send + Sync,

{

}

impl <O> Output for DevicesWrap<O>
where
    O: WithId + Data + CopyToDevice + Send + Sync,
{

}

impl <O> WithId for DevicesWrap<O>
where
    O: WithId + Data + CopyToDevice + Send + Sync,

{
    type ID = O::ID;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}


struct TracingWrapper<O, D>
where
    O: Data + CopyToDevice + WithId + Send + Sync + CopyToDevice,
    D: Deref<Target = O>,
{
    o: D,
    ctx: Option<&'static str>,
}

impl<O, D> Debug for TracingWrapper<O, D>
where
    O: Data + CopyToDevice + WithId + Send + Sync + CopyToDevice,
    D: Deref<Target = O>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TracingWrapper").finish()
    }
}
impl<O, D> TracingWrapper<O, D>
where
    O: Data + CopyToDevice + WithId + Send + Sync + CopyToDevice,
    D: Deref<Target = O>,
{
    pub fn new(o: D, init_ctx: Option<&'static str>) -> Self {
        if let Some(ctx) = init_ctx {
            o.track_used_in(ctx);
        }

        Self { o, ctx: init_ctx }
    }
}

impl<O, D> Drop for TracingWrapper<O, D>
where
    O: Data + CopyToDevice + WithId + Send + Sync + CopyToDevice,
    D: Deref<Target = O>,
{
    fn drop(&mut self) {
        if let Some(ctx) = self.ctx {
            self.o.track_used_in(format!("{} removed", ctx).leak())
        }
    }
}

impl<O, D> Deref for TracingWrapper<O, D>
where
    O: Data + CopyToDevice + WithId + Send + Sync + CopyToDevice,
    D: Deref<Target = O>,
{
    type Target = O;

    fn deref(&self) -> &Self::Target {
        &self.o
    }
}

unsafe impl<O, D> Send for TracingWrapper<O, D>
where
    O: Data + CopyToDevice + WithId + Send + Sync + CopyToDevice,
    D: Deref<Target = O>,
{
}


impl<O: Data + CopyToDevice + WithId + Send + Sync + CopyToDevice> DevicesWrap<O> {
    pub fn wrap(o: O, cache_area: u64) -> Self {
        let id = o.id().clone();
        let size = o.size();
        let device_ordinal = o.device();
        let mem_size = o.mem_size();

        let num_devices: usize = GlobalCtx::global().device_count();
        let mut o_op = Some(o);

        let device_futures: Box<[tokio::sync::RwLock<Option<O>>]> = (0..num_devices)
            .map(|device| {
                if device == device_ordinal {
                    tokio::sync::RwLock::new(o_op.take())
                } else {
                    tokio::sync::RwLock::new(None)
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let inflights = (0..num_devices)
            .map(|_device| tokio::sync::Mutex::new(None))
            .collect::<Vec<_>>()
            .into_boxed_slice();


        Self {
            id,
            cached_data: device_futures,
            inflights,
            cache_area: AtomicU64::new(cache_area),
            size,
            on_cpu_since: Default::default(),
            mem_size,
            pd: PhantomData,
        }
    }

    pub fn wrap_cpu(o: O) -> Self {//TODO replace with union
        let mem_size = o.mem_size();
        Self {
            id: o.id().clone(),
            cached_data: Box::new([tokio::sync::RwLock::new(Some(o))]),
            inflights: Box::new([]),
            cache_area: AtomicU64::new(0),
            size: Default::default(),
            on_cpu_since: Default::default(),
            mem_size,
            pd: PhantomData,
        }
    }

    pub async fn with_another<A, B, WA, WB, R>
    (
     aw: &'static WA,
     bw: &'static WB,
     to_device_idx: usize,
     _trace_ctx: Option<&'static str>,
     do_run: impl FnOnce(&A, &B) -> Result<R, ThoosaError>,
    ) -> Result<R, ThoosaError>
    where
        A:Data + CopyToDevice + WithId + Send + Sync + CopyToDevice + 'static,
        B:Data + CopyToDevice + WithId + Send + Sync + CopyToDevice + 'static,
        WA:AsRef<DevicesWrap<A>> + Debug + 'static,
        WB:AsRef<DevicesWrap<B>> + Debug + 'static,
    {
        let (me, other) = tokio::try_join!(
            tokio::spawn(aw.as_ref().load_from_any_device_(to_device_idx)),
            tokio::spawn(bw.as_ref().load_from_any_device_(to_device_idx))
        )?;
        let me = me?;
        let other = other?;

        do_run(&me, &other)
    }

    pub async fn with_data_array<W: AsRef<DevicesWrap<O>> + Debug, R>(
        other: &[W],
        to_device_idx: usize,
        _trace_ctx: Option<&'static str>,
        do_run: impl FnOnce(&[&O]) -> Result<R, ThoosaError>,
    ) -> Result<R, ThoosaError> {

        let results = tokio::try_join!(try_join_all(
            other
                .into_iter()
                .map(|w| w.as_ref().load_from_any_device_(to_device_idx)),
        ))?;


        if results.0.is_empty() {
            return Err(MyError::SomethingElse(format!("no data in all devices for {other:?} ")).into());
        }

        let refs: Vec<&O> = results.0.iter().map(|o| &**o).collect();
        do_run(&refs)
    }

    //TODO use streams + bounds
    pub async fn with_two_arrays<B, WA, WB, R>
    (
        aw: &[WA],
        bw: &[WB],
        to_device_idx: usize,
        _trace_ctx: Option<&'static str>,
        do_run: impl FnOnce(&[&O], &[&B]) -> Result<R, ThoosaError>,
    ) -> Result<R, ThoosaError>
    where
        // A: Data + CopyToDevice + WithId + Send + Sync + CopyToDevice + 'static,
        B: Data + CopyToDevice + WithId + Send + Sync + CopyToDevice + 'static,
        WA: AsRef<DevicesWrap<O>> + Debug + 'static,
        WB: AsRef<DevicesWrap<B>> + Debug + 'static,
    {
        let (me, other) = tokio::try_join!(
            try_join_all(aw
                .into_iter()
                .map(|w| w.as_ref().load_from_any_device_(to_device_idx))
            ),
            try_join_all(bw
                .into_iter()
                .map(|w| w.as_ref().load_from_any_device_(to_device_idx)),
        ))?;

        let refs_a:Vec<&O> = me.iter().map(|o| &**o).collect();
        let refs_b:Vec<&B> = other.iter().map(|o| &**o).collect();
        do_run(&refs_a, &refs_b)
    }


    pub fn id(&self) -> &<O as WithId>::ID {
        &self.id
    }

    pub async fn take_data(
        self,
        to_device_idx: usize,
    ) -> Result<O, ThoosaError> {

        let _ = self.load_from_any_device_(to_device_idx).await?;

        let mut v = self.cached_data.into_vec();
        let lock = v.swap_remove(to_device_idx);
        let opt: Option<O> = lock.into_inner();
        Ok(opt.expect(&format!("no data in {to_device_idx} for {:?}", self.id)))
    }

    pub async fn clone_from_cpu(&self) -> Result<O, ThoosaError>
    where O: Clone {
        Ok(self.cached_data[0].read().await.clone().expect("no data in cpu"))
    }

    pub async fn with_selected_data<S, R>(
        &self,
        to_device_idx: usize,
        selector: S,
        do_run: impl FnOnce(<O as Select<S>>::Output) -> Result<R, ThoosaError>,
    ) -> Result<R, ThoosaError>
    where
        O: Select<S>,
        <O as Select<S>>::Output: Data + CopyToDevice + WithId + Send + Sync + CopyToDevice,
        S: Clone,
    {
        let read_guard: RwLockReadGuard<'_, Option<O>> = self.cached_data[to_device_idx].read().await;

        if read_guard.is_some() {
            let data = read_guard
                .as_ref()
                .expect("no value behind guard")
                .subview(selector.clone())?;

            return do_run(data);
        }

        drop(read_guard);

        for (source_device_idx, op) in self.cached_data.iter().enumerate() {
            if source_device_idx == to_device_idx {
                continue;
            }

            let source_read_guard: RwLockReadGuard<'_, Option<O>> = op.read().await;

            if let Some(src) = source_read_guard.as_ref() {
                let sub_t = src.subview(selector.clone())?;
                debug!("DataCacheWrapper::with_selected_data: sub_t.size={:?} copying to to_device_idx={to_device_idx}",sub_t.size());
                let data = sub_t.copy_to_device(to_device_idx);

                return do_run(data.await);
            }
        }

        Err(MyError::NotFound.into())
    }


    fn do_run_trace_ref<R>(
        trace_ctx: Option<&'static str>,
        do_run: impl FnOnce(&O) -> Result<R, ThoosaError>,
        data: &O,
    ) -> Result<R, ThoosaError> {
        if trace_ctx.is_none() {
            do_run(data)
        } else {
            do_run(&TracingWrapper::new(data, trace_ctx))
        }
    }

    fn do_run_trace_owned<T, R>(
        trace_ctx: Option<&'static str>,
        do_run: impl FnOnce(&T) -> Result<R, ThoosaError>,
        data: T,
    ) -> Result<R, ThoosaError>
    where
        T: Data + CopyToDevice + WithId + Send + Sync + CopyToDevice,
    {
        if trace_ctx.is_none() {
            do_run(&data)
        } else {
            do_run(&TracingWrapper::new(&data, trace_ctx))
        }
    }

    async fn load_from_any_device_(
        &self,
        to_device_idx: usize,
    ) -> Result<impl Deref<Target = O> + Debug + Send, ThoosaError> {
        //purge shouldn't happen here since it requires MAX permits and it this is being executed other runs can't have MAX permits,
        //TODO add GlobalCtx::purge_lock just in case?
        debug!("DataCacheWrapper::from_any_device id={:?} device_idx={to_device_idx}",self.id);
        trace!("DataCacheWrapper::from_any_device id={:?} device_idx={to_device_idx} self.cached_data={:?}",self.id, self.cached_data);

        if to_device_idx != 0 {
            *self.on_cpu_since.write() = None;
        }

        let read_guard = self.cached_data[to_device_idx].read().await;

        if read_guard.is_some() {
            Ok(tokio::sync::RwLockReadGuard::map(
                read_guard,
                |val: &Option<O>| val.as_ref().expect("no value behind guard"),
            ))
        } else {
            drop(read_guard);

            for (source_device_idx, op) in self.cached_data.iter().enumerate() {
                let source_read_guard = op.read().await;

                if source_device_idx != to_device_idx && source_read_guard.is_some() {
                    drop(source_read_guard);
                    return self.copy_and_get(source_device_idx, to_device_idx).await;
                }
            }
            Err(MyError::NotFound.into())
        }
    }

    pub async fn load_from_any_device(
        &self,
        to_device_idx: usize,
    ) -> Result<(), ThoosaError> {
        //purge shouldn't happen here since it requires MAX permits and it this is being executed other runs can't have MAX permits,
        //TODO add GlobalCtx::purge_lock just in case?
        debug!("DataCacheWrapper::from_any_device id={:?} device_idx={to_device_idx}",self.id);
        trace!("DataCacheWrapper::from_any_device id={:?} device_idx={to_device_idx}  self.cached_data={:?}",self.id, self.cached_data);

        if to_device_idx != 0 {
            *self.on_cpu_since.write() = None;
        }

        let read_guard = self.cached_data[to_device_idx].read().await;

        if read_guard.is_some() {
            Ok(())
        } else {
            drop(read_guard);

            for (source_device_idx, op) in self.cached_data.iter().enumerate() {
                let source_read_guard = op.read().await;

                if source_device_idx != to_device_idx && source_read_guard.is_some() {
                    drop(source_read_guard);
                    self.ensure_copied(source_device_idx, to_device_idx, None, false).await?;
                    return Ok(());
                }
            }
            Err(MyError::NotFound.into())
        }
    }

    pub async fn get_copied(&self, from_device_idx: usize) -> Result<impl Deref<Target = O> + Debug + Send, ThoosaError> {
        let read_guard = self.cached_data[from_device_idx].read().await;

        if read_guard.is_some() {
            Ok(tokio::sync::RwLockReadGuard::map(
                read_guard,
                |val: &Option<O>| val.as_ref().expect("no value behind guard"),
            ))
        } else {
            Err(MyError::NotReady.into())
        }
    }


    pub async fn dump_to_cpu(&self, device_idx: usize) -> Result<(), ThoosaError> {
        assert_ne!(device_idx, 0);

        debug!("DataCacheWrapper::dump_to_cpu: 0.  id={:?} device_idx={device_idx}",self.id);

        if let Some(Ok(mut lock)) = self.cached_data.get(device_idx).map(|v|v.try_write()) {
            if self.cached_data[0].read().await.is_none() {
                let data = lock.take();

                self.ensure_copied(device_idx, 0, data.as_ref(), false).await?;

                debug!("DataCacheWrapper::dump_to_cpu: 1. id={:?} copied to cpu done",self.id);
            }
            drop(lock.take());

        } else {
            //if it's locked then it should be one of the parent futures being built, waiting on it will deadlock
            //TODO log warning?
            debug!("DataCacheWrapper::dump_to_cpu: can't drop self.id={:?}, locked, data={:?}",self.id, self.cached_data)
        }

        let is_all_on_cpu = self.is_all_on_cpu().await;
        if is_all_on_cpu {
            //may rewrite previous Some, but shouldn't be a problem
            *self.on_cpu_since.write() = Some(GlobalCtx::global().cache_age());
        }

        debug!("DataCacheWrapper::dump_to_cpu: 2. id={:?} device_idx={device_idx} is_all_on_cpu={is_all_on_cpu}",self.id);

        Ok(())
    }

    //to avoid write lock on dst device while copy is in progress use inflights
    async fn ensure_copied<'a>(
        &self,
        src_idx: usize,
        dst_idx: usize,
        src: Option<&O>,
        do_return: bool,
    ) -> Result<Option<RwLockReadGuard<'_, O>>, ThoosaError> {
        loop {
            let dst_read = self.cached_data[dst_idx].read().await;
            // ---- Fast path
            if dst_read.is_some() {
                if do_return {
                    return Ok(Some(RwLockReadGuard::map(dst_read, |v| {
                        v.as_ref().unwrap()
                    })));
                } else {
                    return Ok(None);
                }
            }
            drop(dst_read);

            // ---- Try to join or become the leader
            // (Do not hold this lock while awaiting other locks, except briefly to set or read the notify)
            let maybe_notify = {
                let mut g = self.inflights[dst_idx].lock().await;
                if let Some(n) = &*g {
                    // someone is already copying → wait on their notify
                    Some(n.clone())
                } else {
                    // become the leader: install a notify and proceed
                    let n = Arc::new(Notify::new());
                    *g = Some(n.clone());
                    None
                }
            };

            if let Some(n) = maybe_notify {
                // follower: wait until the leader finishes, then try fast path again
                n.notified().await;
                continue;
            }

            // ---- Leader path
            // Double-check: maybe dst was filled after we dropped the read lock
            if self.cached_data[dst_idx].read().await.is_some() {
                // clear inflight and wake any waiters
                let notify = {
                    let mut g = self.inflights[dst_idx].lock().await;
                    g.take().expect("leader must have set inflight")
                };
                notify.notify_waiters();
                continue;
            }

            // Read source value without holding dst locks
            let produced = match src {
                None => {
                    self.cached_data[src_idx]
                        .read()
                        .await
                        .as_ref()
                        .expect("src missing unexpectedly")
                        .copy_to_device(dst_idx)
                        .await
                }
                Some(v) => v.copy_to_device(dst_idx).await,
            };

            // Install into dst (short critical section)
            {
                let mut w = self.cached_data[dst_idx].write().await;
                if w.is_none() {
                    *w = Some(produced);
                }
            }

            // Clear inflight and wake followers
            let notify = {
                let mut g = self.inflights[dst_idx].lock().await;
                g.take().expect("leader must have set inflight")
            };
            notify.notify_waiters();
        }
    }

    async fn copy_and_get(
        &self,
        src_idx: usize,
        dst_idx: usize,
    ) -> Result<RwLockReadGuard<'_, O>, ThoosaError> {
        self.ensure_copied(src_idx, dst_idx, None, true)
            .await
            .map(|v| v.expect("no value returned"))
    }

    pub fn add_cache_area(&self, ca: u64) -> u64 {
        self.cache_area.fetch_or(ca, Ordering::SeqCst) | ca
    }

    pub fn remove_cache_area(&self, ca: u64) -> u64 {
        self.cache_area.fetch_and(!ca, Ordering::AcqRel) & !ca
    }

    pub fn cache_areas(&self) -> u64 {
        self.cache_area.load(Ordering::Acquire)
    }

    pub fn size(&self) -> &O::Metrics {
        &self.size
    }

    pub async fn is_all_on_cpu(&self) -> bool {
        debug!("DataCacheWrapper::is_all_on_cpu: self.id={:?}", self.id);

        for (device_idx, data) in self.cached_data.iter().skip(1).enumerate() {
            if data.read().await.is_some() {
                debug!("DataCacheWrapper::is_all_on_cpu: device_idx={device_idx}");
                return false;
            }
        }

        true
    }

    pub fn is_on_cpu_since(&self, threshold_age: u64) -> bool {
        let r = self
            .on_cpu_since
            .read()
            .map(|v| threshold_age < v)
            .unwrap_or(false);

        debug!(
            "DataCacheWrapper::is_on_cpu_since: self.id={:?} self.on_cpu_since={:?} r={r} cache_age={threshold_age}",
            self.id, self.on_cpu_since
        );

        r
    }

    #[cfg(test)]
    pub fn cached_data(&self) -> &Box<[tokio::sync::RwLock<Option<O>>]> {
        &self.cached_data
    }
}

///To let OOMHandler accept both plain wrapped data as it's stored in caches and composite inputs like module + tensor or batch + motif
pub trait WithDataRef: WithId + Debug {
    type O<'o>: Data + /*CopyToDevice +*/ WithId + Send + Sync;

    async fn with_data_ref<R, F>(
        &self,
        to_device_idx: usize,
        trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> Result<R, ThoosaError>
    where
       for <'f> F: FnOnce(&Self::O<'f>) -> Result<R, ThoosaError>;


    async fn dummy<R, F>(&self, f:F) -> Result<R, ThoosaError>
    where
        for <'f> F:  FnOnce() -> Result<R, ThoosaError> {

        (f)()

    }

    //shouldn't be here, used only in ClcOOMRunner which is test-only util
    fn _size<'f>(&self) -> <Self::O<'f> as Data>::Metrics{
        Default::default()
    }

    fn mem_size(&self) ->u64{
        0
    }
}

pub trait WithSelectDataRef<S>: WithDataRef
where
    S: Clone + Debug + 'static,
{
    async fn with_selected_data_ref<'o, R, F>(
        &self,
        to_device_idx: usize,
        selector: S,
        trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> Result<R, ThoosaError> where
     // Self::O<'o>: Select<S, Output = Self::O<'o>>,
     for<'f> F: FnOnce(&Self::O<'f>) -> Result<R, ThoosaError>;
}


impl <O: Data + CopyToDevice + WithId + Send + Sync + CopyToDevice> WithDataRef for DevicesWrap<O> {
    type O<'o> = O;

    async fn with_data_ref<R, F>(&self, to_device_idx: usize, trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoosaError>
    where
        for <'f> F: FnOnce(&Self::O<'f>) -> Result<R, ThoosaError>
    {
        match self.load_from_any_device_(to_device_idx).await {
            Ok(v) => {
                let data: &Self::O<'_> = v.deref();
                Self::do_run_trace_ref(trace_ctx, do_run, data)
            }
            Err(e) => Err(e),
        }
    }

    fn mem_size(&self) -> u64 {
        self.mem_size
    }

    fn _size<'f>(&self) -> <Self::O<'f> as Data>::Metrics {
        self.size().clone()
    }
}



impl<O, S> WithSelectDataRef<S> for DevicesWrap<O>
where
    S:Debug + Clone + 'static,
    O: Select<S, Output=O> + Data + CopyToDevice + WithId + Send + Sync + CopyToDevice,
{
    async fn with_selected_data_ref<'o, R, F>(&self, to_device_idx: usize, selector: S, trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoosaError>
    where
        Self::O<'o>: Select<S, Output=Self::O<'o>>,
        for <'f> F: FnOnce(&Self::O<'f>) -> Result<R, ThoosaError>
    {
        let read_guard: RwLockReadGuard<'_, Option<Self::O<'_>>> = self.cached_data[to_device_idx].read().await;

        if read_guard.is_some() {
            let base: &Self::O<'_> = read_guard
                .as_ref()
                .expect("no value behind guard");

            let data: Self::O<'_> = base.subview(selector.clone())?;

            return Self::do_run_trace_owned(trace_ctx, do_run, data);
        }

        drop(read_guard);

        for (source_device_idx, op) in self.cached_data.iter().enumerate() {
            if source_device_idx == to_device_idx {
                continue;
            }

            let source_read_guard: RwLockReadGuard<'_, Option<Self::O<'_>>> = op.read().await;

            if let Some(src) = source_read_guard.as_ref() {
                let sub_t: Self::O<'_> = src.subview(selector.clone())?;

                debug!("DataCacheWrapper::with_selected_data: sub_t.size={:?} copying to to_device_idx={to_device_idx}",sub_t.size());

                let data = sub_t.copy_to_device(to_device_idx);

                return Self::do_run_trace_owned(trace_ctx, do_run, data.await);
            }
        }

        Err(MyError::NotFound.into())
    }

}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use crate::common::{DropIf, MockRunCtx, OutId, ThoosaError};
    use crate::global_ctx::{MockGlobalCtx, mock_global_ctx};
    use crate::*;
    use log::debug;
    use mockall::mock;
    use parking_lot::RwLock;
    use thoosa_shared::MyError;
    use thoosa_shared::traits::{Adjustable, WithId};
    use thoosa_with_data::with_data;
    use crate::common::cachable::{DevicesWrap, WithDataRef};
    use crate::common::runnable::{Stage, SyncStage};
    use crate::oom_handler::{mock_input_context, Adjusters, Contexts, MockRunCtxContext, MockRunnableContext, OOMHandler};
    use crate::oom_handler::test_types::{TestInput, TestOutput};
    use crate::test_utils::MockDataWithId;

    #[tokio::test]
    async fn test_cache_area() -> Result<(), ThoosaError> {
        let ctx = MockGlobalCtx::global_context();
        ctx.expect().returning(|| {
            Box::leak(Box::new(
                mock_global_ctx::build_default(),
            )) as &'static global_ctx::MockGlobalCtx
        });

        let tested = super::DevicesWrap::wrap(MockDataWithId::whatever(), 0);

        let ca = tested.add_cache_area(u64::from_str_radix(
            "0000000000000000000000000000000000000000000000000000000000000001",
            2,
        )?);
        assert_eq!(ca, tested.cache_areas());
        assert_eq!(
            "0000000000000000000000000000000000000000000000000000000000000001",
            format!("{:064b}", tested.cache_areas())
        );

        let ca = tested.add_cache_area(u64::from_str_radix(
            "0000001000000000000000000000000000000000000000000000000000000001",
            2,
        )?);
        assert_eq!(ca, tested.cache_areas());

        assert_eq!(
            "0000001000000000000000000000000000000000000000000000000000000001",
            format!("{:064b}", tested.cache_areas())
        );

        let ca = tested.remove_cache_area(u64::from_str_radix(
            "0000000000000000000000000000000000000000000000000000000000000001",
            2,
        )?);
        assert_eq!(ca, tested.cache_areas());

        assert_eq!(
            "0000001000000000000000000000000000000000000000000000000000000000",
            format!("{:064b}", tested.cache_areas())
        );

        let ca = tested.remove_cache_area(u64::from_str_radix(
            "0000001000000000000000000000000000000000000000000000000000000000",
            2,
        )?);
        assert_eq!(ca, tested.cache_areas());
        assert_eq!(
            "0000000000000000000000000000000000000000000000000000000000000000",
            format!("{:064b}", tested.cache_areas())
        );

        Ok(())
    }

    // #[tokio::test]
    // async fn test_with_data() -> Result<(), ThoosaError> {
    //     let ctx = MockGlobalCtx::global_context();
    //     ctx.expect().returning(|| {
    //         Box::leak(Box::new(
    //             mock_global_ctx::build_default(),
    //         )) as &'static global_ctx::MockGlobalCtx
    //     });
    //     let tested = super::DevicesWrap::wrap(MockDataWithId::whatever(), 0);
    //
    //     tested
    //         .with_data_ref(0, None, |t| {
    //             debug!("t={t:?}");
    //             Ok(())
    //         })
    //         .await?;
    //     //TODO test
    //     Ok(())
    // }


    #[tokio::test]
    async fn test_dump_to_cpu() -> Result<(), ThoosaError> {
        let ctx = MockGlobalCtx::global_context();
        ctx.expect().returning(|| {
            Box::leak(Box::new(
                mock_global_ctx::build_default(),
            )) as &'static global_ctx::MockGlobalCtx
        });

        let data = MockDataWithId::whatever();
        let tested = super::DevicesWrap::wrap(data, 0);
        assert!(tested.cached_data[1].read().await.is_none());

        tested
            .with_data_ref(1, None, |data_on_1: &MockDataWithId| {
                assert_eq!(data_on_1.device(), &1);

                Ok(())
            })
            .await?;

        assert!(tested.cached_data[1].read().await.is_some());

        tested.dump_to_cpu(1).await?;
        assert!(tested.cached_data[1].read().await.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_with_data() -> Result<(), ThoosaError> {
        let ctx = MockGlobalCtx::global_context();
        ctx.expect().returning(|| {
            Box::leak(Box::new(
                mock_global_ctx::build_default(),
            )) as &'static global_ctx::MockGlobalCtx
        });

        let wrap_a = Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "a"), 0));
        let wrap_b = Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "d"), 0));
        let wrap_vec = vec![
            Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "b"), 0)),
            Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "c"), 0)),
        ];
        let wrap_vec_second = vec![
            Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "e"), 0)),
            Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "f"), 0)),
        ];

        //TODO ensure that asserts were actually invoked
        with_data!(1, wrap_a, [wrap_vec], wrap_b, [wrap_vec_second];
            |a: &MockDataWithId, first: &[&MockDataWithId], b: &MockDataWithId, second: &[&MockDataWithId]|{
                assert_eq!(a.device(), &1);
                assert_eq!(b.device(), &1);
                assert_eq!(first.len(), 2);
                assert!(first.iter().all(|item| *item.device() == 1));
                assert_eq!(second.len(), 2);
                assert!(second.iter().all(|item| *item.device() == 1));
                Ok::<(), ThoosaError>(())
        })?;
        Ok(())
    }




    // #[tokio::test]
    // async fn test_with_data_and_oom()-> Result<(), ThoosaError> {
    //
    //     let ctx = MockGlobalCtx::global_context();
    //     ctx.expect().returning(|| {
    //         Box::leak(Box::new(
    //             mock_global_ctx::build_default(),
    //         )) as &'static global_ctx::MockGlobalCtx
    //     });
    //
    //
    //     let (task, task_calls) = crate::oom_handler::mock_runnable::build([
    //         Err(ThoosaError::from(MyError::OutOfMemory)),
    //         Ok(TestOutput::new(10.)),
    //     ]);
    //     let (input_adj, input_calls) = crate::oom_handler::mock_input_adjuster::build([[0]..[1]]);
    //     let oom_handler = OOMHandler::new(vec![Adjusters::Input(Box::new(input_adj))], None);
    //     let ctx_mock = crate::common::mock_run_ctx::build()
    //         .with_device_idx(1)
    //         .with_acquired_permits(1)
    //         .with_drop_permits(vec![DropIf::IfNotMax])
    //         .finish();
    //
    //     let test_input = TestInput::new(1);
    //     let input = Arc::new(DevicesWrap::wrap(test_input.clone(), 0));
    //
    //     let out = ;
    //     // let mocked_stat = Moc
    //
    //     let __with_data_device = 1;
    //     async move {
    //         let __device_idx = __with_data_device;
    //         let __with_data_var_0 = ::std::sync::Arc::clone(&input);
    //         let __with_data_handle_0 = ::tokio::spawn({
    //             let __wrap = ::std::sync::Arc::clone(&__with_data_var_0);
    //             let __device_idx = __device_idx;
    //             async move {
    //                 __wrap.load_from_any_device(__device_idx).await?;
    //                 Ok::<(), ::thoosa_shared::ThoosaError>(())
    //             }
    //         });
    //         let __with_data_handle_res_0 = __with_data_handle_0.await?;
    //         __with_data_handle_res_0?;
    //         let __with_data_guard_0 = __with_data_var_0.get_copied(__device_idx).await?;
    //         let __with_data_ref_0 = &*__with_data_guard_0;
    //
    //
    //         // (|a: &TestInput| {
    //         //     assert_eq!(&test_input.data.0, &a.data.0);
    //         //     Ok::<(), ThoosaError>(())
    //         // })(__with_data_ref_0)
    //
    //         oom_handler
    //             .run(
    //                 &Arc::new(task),
    //                 Arc::clone(&input),
    //                 Arc::new(RwLock::new(ctx_mock)),
    //                 Contexts::new(
    //                     mock_input_context::build()
    //                         .with_partial_output(TestOutput::new(10.))
    //                         .with_all_done(&[true])
    //                         .with_build_final_output(TestOutput::new(10.))
    //                         .finish(),
    //                     MockRunnableContext::new(),
    //                     MockRunCtxContext::new(),
    //                 ),
    //             )
    //             .await?
    //
    //
    //
    //     }.await;
    //
    //
    //     Ok(())
    // }

    #[tokio::test]
    async fn test_with_data_single_val() -> Result<(), ThoosaError> {
        let ctx = MockGlobalCtx::global_context();
        ctx.expect().returning(|| {
            Box::leak(Box::new(
                mock_global_ctx::build_default(),
            )) as &'static global_ctx::MockGlobalCtx
        });

        let wrap_a = Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "a"), 0));

        with_data!(1, wrap_a;
            |a: &MockDataWithId|{
                assert_eq!(a.device(), &1);
                Ok::<(), ThoosaError>(())
        })?;
        Ok(())
    }
}
