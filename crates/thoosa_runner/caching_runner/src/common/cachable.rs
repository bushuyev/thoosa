#![allow(unused_parens)]
#![allow(async_fn_in_trait)]
use crate::common::runnable::Stage;
use crate::common::*;
#[double]
use crate::global_ctx::GlobalCtx;
use async_trait::async_trait;
use dashmap::{DashMap, Entry};
use futures_util::future::{join_all, BoxFuture, Shared};
use futures_util::FutureExt;
use mockall_double::double;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{Notify, RwLockReadGuard};

#[async_trait]
pub trait Cached: Stage {
    async fn cache() -> &'static DashMap<
        <<Self as Stage>::Out as WithId>::ID,
        Shared<BoxFuture<'static, Result<Arc<DataCacheWrapper<Self::Out>>, MyError>>>,
    >
    where
        Self: Sized;


    async fn run_cached<I>(
        self: Arc<Self>,
        input: Arc<I>,
        ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Arc<DataCacheWrapper<Self::Out>>, MyError>
    where
        Self: Sized + OutId,
        <<Self as Stage>::Out as WithId>::ID: 'static,
        <Self as Stage>::Out: 'static,
        // <Self as Runnable>::Out:Data<Sub= <Self as Runnable>::Out>+CopyToDevice,
        I:WithId<ID=<<Self as Stage>::In as WithId>::ID> + Send + Sync + 'static
    {
        let key = self.out_id(input.id());
        let (cache_area, device_idx) = {
            let guard = ctx.read();
            (guard.cache_area(), guard.device_idx())
        };

        println!("run_cached: key={key:?} device_idx={device_idx}");

        let cache = Self::cache().await;

        self.on_cached(input.as_ref(), &ctx);

        GlobalCtx::global().do_age();

        match cache.entry(key) {
            Entry::Occupied(slot) => {
                let shared = slot.get().clone();
                drop(slot);
                shared.map(|res| {
                    if let Ok(data) = &res {
                        data.add_cache_area(cache_area);
                    }
                    res
                }).await
            },
            Entry::Vacant(slot) => {

                let fut = Arc::clone(&self)
                    .run(input, ctx)
                    .map(move |r| {

                        r.map(|v|{
                            // assert_eq!(v.device(), device_idx); - loaded to CPU, copied to GPU when used
                            Arc::new(DataCacheWrapper::new(v, device_idx, GlobalCtx::global().device_count(), cache_area))
                        })
                    })
                    .boxed()
                    .shared();

                slot.insert(fut.clone());//slot consumed, fut.await doesn't block it

                fut.await

            }
        }
    }

    async fn purge_gpu_cache_area(
        keys: Vec<<Self::Out as WithId>::ID>,
        cache_area: u64,
        device_idx: usize,
    ) -> Result<(), MyError>
    where
        Self: Sized + OutId,
    {
        assert_ne!(device_idx, 0); //don't purge cpu

        println!("Cachable::purge_gpu_cache_area: 0  cache_area={cache_area} device_idx={device_idx}");

        let map = Self::cache().await;

        for key in keys {
            // let key_device_mutex = Self::key_in_device_lock(&key, device_idx);
            // let key_guard = key_device_mutex.lock().await;
            println!("Cachable::purge_gpu_cache_area: 1 key={key:?} cache_area={cache_area} device_idx={device_idx}");

            if let Some(value) = map.get(&key) {
                let data:Arc<DataCacheWrapper<Self::Out>> = value.value().clone().await?;
                let new_mask = data.remove_cache_area(cache_area);

                println!("Cachable::purge_gpu_cache_area: 2 key={key:?} new_mask={new_mask} ");
                if new_mask == 0 {
                    data.dump_to_cpu(device_idx).await?;
                    // drop(key_guard);
                    println!("Cachable::purge_gpu_cache_area: 3  key={key:?} done");
                }

            } else {
                println!("Cachable::purge_gpu_cache_area: 4 key={key:?} {cache_area} device_idx={device_idx} not found"); //warning
            }
        }

        println!("Cachable::purge_gpu_cache_area: 5 cache_area={cache_area} device_idx={device_idx}");

        Ok(())
    }

}


//when hidden behind a trait, impls for plain Ids and Data cause https://github.com/rust-lang/rust/issues/100013 so opting to enum
#[derive(Debug)]
pub struct DataCacheWrapper<O>
where
O: WithId + Data<Sub=O> + CopyToDevice + Send + Sync,

{
    id: <O as WithId>::ID,
    cached_data: Box<[tokio::sync::RwLock<Option<O>>]>,
    inflights: Box<[tokio::sync::Mutex<Option<Arc<tokio::sync::Notify>>>]>,
    cache_area:AtomicU64,
    size: O::Metrics,

    //contains cache age ( GlobalCtx::cache_age - increated when something is being cached from Cachable::run_cached). The smaller the value the longer it was in the cpu
    on_cpu_since: parking_lot::RwLock<Option<u64>>,

}

struct TracingWrapper<O, D>
where
    O: Data + WithId + Send + Sync + CopyToDevice,
    D: Deref<Target = O>
{
    o:D,
    ctx: Option<&'static str>
}

impl <O, D> Debug for TracingWrapper<O, D>
where
    O: Data + WithId + Send + Sync + CopyToDevice,
    D: Deref<Target = O>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TracingWrapper").finish()
    }
}
impl <O, D> TracingWrapper<O, D>
where
    O: Data + WithId + Send + Sync + CopyToDevice,
    D: Deref<Target = O>

{
    pub fn new(o: D, init_ctx: Option<&'static str>) -> Self {
        if let Some(ctx) = init_ctx {
            o.track_used_in(ctx);
        }

        Self { o, ctx:init_ctx }
    }
}

impl <O,D>Drop for TracingWrapper<O,D>
where
    O: Data + WithId + Send + Sync + CopyToDevice,
    D: Deref<Target = O>

{
    fn drop(&mut self) {
        if let Some(ctx) = self.ctx {
            self.o.track_used_in(format!("{} removed", ctx).leak())
        }
    }
}

impl <O, D> Deref for TracingWrapper<O, D>
where
    O: Data + WithId + Send + Sync + CopyToDevice,
    D: Deref<Target = O>
{
    type Target = O;

    fn deref(&self) -> &Self::Target {
        &self.o
    }
}

unsafe impl <O, D> Send for TracingWrapper<O, D>
where
    O: Data + WithId + Send + Sync + CopyToDevice,
    D: Deref<Target = O>{}


#[derive(Debug)]
pub enum Select<O:Data> {
    Full,
    SubSelect { sub:<O as Data>::SubSelect},
}


impl <O:Data<Sub=O>+WithId + Send + Sync+CopyToDevice> DataCacheWrapper<O>{

    pub fn new(o:O, _device_ordinal: usize, num_devices: usize, cache_area:u64) -> Self {

        let id = o.id().clone();
        let size = o.size();
        let device_ordinal = o.device();

        let mut o_op = Some(o);


        let device_futures: Box<[tokio::sync::RwLock<Option<O>>]> = (0..num_devices).map(|device|{
            if device == device_ordinal {
                tokio::sync::RwLock::new(o_op.take())
            } else {

                tokio::sync::RwLock::new(None)
            }
        }).collect::<Vec<_>>().into_boxed_slice();

        let inflights = (0..num_devices).map(|_device|tokio::sync::Mutex::new(None)).collect::<Vec<_>>().into_boxed_slice();


        Self {
            id,
            cached_data: device_futures,
            inflights,
            cache_area: AtomicU64::new(cache_area),
            size,
            on_cpu_since: Default::default(),
        }
    }

    pub async fn with_data_array<R>(
        other: &[&DataCacheWrapper<O>],
        to_device_idx: usize,
        _trace_ctx: Option<&'static str>,
        do_run: impl FnOnce(&[&O]) -> Result<R, MyError>,
    ) -> Result<R, MyError> {

        let results = join_all(other.into_iter().map(|w| w.load_from_any_device_(to_device_idx))).await;

        let mut owners: Vec<_> = Vec::new();
        let mut errs: Vec<MyError> = Vec::new();

        for r in results {
            match r {
                Ok(o) => {
                    owners.push(o);
                }
                Err(e) => errs.push(e),//TODO log errors
            }
        }

        if owners.is_empty() {
            return Err(MyError::SomethingElse("all failed".into()));
        }

        let refs: Vec<&O> = owners.iter().map(|o| &**o).collect();
        do_run(&refs)

    }

    pub fn id(&self)-> &<O as WithId>::ID {
        &self.id
    }

    pub async fn with_data<R>(
        &self,
        to_device_idx:usize,
        select: Option<&<O as Data>::SubSelect>,
        trace_ctx: Option<&'static str>,
        do_run:impl FnOnce(&O)->Result<R, MyError>) -> Result<R, MyError>
    {

        match select {

            None => {
                if to_device_idx != 0 {
                    GlobalCtx::global().reset_all_on_cpu(to_device_idx).await;
                }

                match self.load_from_any_device_(to_device_idx).await {
                    Ok(v) => {
                        Self::do_run_trace(trace_ctx, do_run, v.deref())
                    }
                    Err(e) => Err(e)
                }

            }
            Some (sub_select )  => {

                let read_guard =  self.cached_data[to_device_idx].read().await;

                if read_guard.is_some() {

                    let data = read_guard.as_ref().expect("no value behind guard").subview(sub_select);

                    Self::do_run_trace(trace_ctx, do_run, &data)

                } else {
                    drop(read_guard);

                    for (source_device_idx, op)  in self.cached_data.iter().enumerate() {
                        let source_read_guard = op.read().await;

                        if source_device_idx != to_device_idx{
                            if let Some(src) = source_read_guard.as_ref() {
                                let sub_t = src.subview(sub_select);
                                println!("DataCacheWrapper::with_data: sub_t.size={:?} copying to to_device_idx={to_device_idx}", sub_t.size());
                                let data = sub_t.copy_to_device(to_device_idx);

                                return Self::do_run_trace(trace_ctx, do_run, &data.await);

                            }
                        }

                    }
                    Err(MyError::NotFound)
                }
            }
        }
    }

    fn do_run_trace<R>(trace_ctx: Option<&'static str>, do_run: impl FnOnce(&O)->Result<R, MyError>, data: &O) -> Result<R, MyError> {
        if trace_ctx.is_none() {
            do_run(data)
        } else {
            do_run(&TracingWrapper::new(data, trace_ctx))
        }
    }


    async fn load_from_any_device_(&self, to_device_idx: usize) -> Result<impl Deref<Target = O> +Debug+ Send, MyError> {
        //purge shouldn't happen here since it requires MAX permits and it this is being executed other runs can't have MAX permits,
        //TODO add GlobalCtx::purge_lock just in case?
        println!("DataCacheWrapper::from_any_device id={:?} device_idx={to_device_idx} self.cached_data={:?}", self.id, self.cached_data);
        if to_device_idx != 0 {
            *self.on_cpu_since.write() = None;
        }

        let read_guard =  self.cached_data[to_device_idx].read().await;

        if read_guard.is_some() {
            Ok(tokio::sync::RwLockReadGuard::map(read_guard,  |val:&Option<O>|{
                val.as_ref().expect("no value behind guard")
            }))

        } else {
            drop(read_guard);

            for (source_device_idx, op)  in self.cached_data.iter().enumerate() {
                let source_read_guard = op.read().await;

                if source_device_idx != to_device_idx && source_read_guard.is_some(){
                    drop(source_read_guard);
                    return self.copy_and_get(source_device_idx, to_device_idx).await;

                }

            }
            Err(MyError::NotFound)
        }

    }

    pub async fn dump_to_cpu(&self, device_idx:usize) -> Result<(), MyError>{

        assert_ne!(device_idx, 0);

        println!("DataCacheWrapper::dump_to_cpu: 0.  id={:?} device_idx={device_idx}", self.id);


        if let Ok(mut lock) = self.cached_data[device_idx].try_write() {

            if self.cached_data[0].read().await.is_none(){
                let data = lock.take();

                self.ensure_copied(device_idx, 0, data.as_ref(), false).await?;

                println!("DataCacheWrapper::dump_to_cpu: 1. id={:?} copied to cpu done", self.id);

            } else {
                drop(lock.take());
            }
            
        } else {
            //if it's locked then it should be one of the parent futures being built, waiting on it will deadlock
            //TODO log warning?
            println!("DataCacheWrapper::dump_to_cpu: can't drop self.id={:?}, locked, data={:?}", self.id, self.cached_data[device_idx])
        }

        let is_all_on_cpu = self.is_all_on_cpu().await;
        if is_all_on_cpu {
            //may rewrite previous Some, but shouldn't be a problem
            *self.on_cpu_since.write() = Some(GlobalCtx::global().cache_age());
        }

        println!("DataCacheWrapper::dump_to_cpu: 2. id={:?} device_idx={device_idx} is_all_on_cpu={is_all_on_cpu}", self.id);

        Ok(())
    }

    //to avoid write lock on dst device while copy is in progress use inflights
    async fn ensure_copied<'a>(&self, src_idx: usize, dst_idx: usize, src:Option<&O>, do_return: bool) -> Result<Option<RwLockReadGuard<'_, O>>, MyError> {
        loop {
            let dst_read = self.cached_data[dst_idx].read().await;
            // ---- Fast path
            if dst_read.is_some() {
                if do_return {
                    return Ok(Some(RwLockReadGuard::map(dst_read, |v| v.as_ref().unwrap())));
                } else {
                    return Ok(None)
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
                Some(v) => {
                    v.copy_to_device(dst_idx)
                        .await
                }
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
    ) -> Result<RwLockReadGuard<'_, O>, MyError> {

        self.ensure_copied(src_idx, dst_idx, None, true).await.map(|v|v.expect("no value returned"))

    }



    pub fn add_cache_area(&self, ca: u64) -> u64{
        self.cache_area.fetch_or(ca, Ordering::SeqCst) | ca
    }

    pub fn remove_cache_area(&self, ca: u64)-> u64 {
        self.cache_area.fetch_and(!ca, Ordering::AcqRel) & !ca
    }

    pub fn cache_areas(&self) ->u64{
        self.cache_area.load(Ordering::Acquire)
    }


    pub fn size(&self) -> &O::Metrics {
        &self.size
    }

    pub async fn is_all_on_cpu(&self) -> bool {
        println!("DataCacheWrapper::is_all_on_cpu: self.id={:?}", self.id);

        for (device_idx, data) in self.cached_data.iter().skip(1).enumerate() {
            if data.read().await.is_some() {
                println!("DataCacheWrapper::is_all_on_cpu: device_idx={device_idx}");
                return false
            }
        }

        true
    }

    pub fn is_on_cpu_since(&self, threshold_age:u64) -> bool {
        let r = self.on_cpu_since.read().map(|v| threshold_age < v ).unwrap_or(false);

        println!("DataCacheWrapper::is_on_cpu_since: self.id={:?} self.on_cpu_since={:?} r={r} cache_age={threshold_age}", self.id, self.on_cpu_since);

        r
    }

    #[cfg(test)]
    pub fn cached_data(&self) -> &Box<[tokio::sync::RwLock<Option<O>>]> {
        &self.cached_data
    }
}

pub trait Runner {
    type R: Stage<In:Data<Sub=<Self::R as Stage>::In>>;

    async fn run(
        &self,
        runnable: &Arc<Self::R>,
        input: Arc<DataCacheWrapper<<Self::R as Stage>::In>>,
        ctx: Arc<RwLock<<Self::R as Stage>::Ctx>>,
        required_ids: &HashMap<&'static str, &HashSet<&dyn CachableIdTr>>
    ) -> Result<<Self::R as Stage>::Out, MyError>;

}

#[cfg(test)]
mod tests {
    use crate::common::MyError;
    use crate::global_ctx::{mock_global_ctx, MockGlobalCtx};
    use crate::*;
    
    use crate::test_utils::MockDataWithId;

    #[tokio::test]
    async fn test_cache_area() -> Result<(), MyError>{

        let tested = super::DataCacheWrapper::new(
            MockDataWithId::whatever(), 1, 2, 0
        );


        let ca = tested.add_cache_area(u64::from_str_radix("0000000000000000000000000000000000000000000000000000000000000001", 2)?);
        assert_eq!(ca, tested.cache_areas());
        assert_eq!(
            "0000000000000000000000000000000000000000000000000000000000000001",
            format!("{:064b}", tested.cache_areas())
        );

        let ca = tested.add_cache_area(u64::from_str_radix("0000001000000000000000000000000000000000000000000000000000000001", 2)?);
        assert_eq!(ca, tested.cache_areas());

        assert_eq!(
            "0000001000000000000000000000000000000000000000000000000000000001",
            format!("{:064b}", tested.cache_areas())
        );

        let ca = tested.remove_cache_area(u64::from_str_radix("0000000000000000000000000000000000000000000000000000000000000001", 2)?);
        assert_eq!(ca, tested.cache_areas());

        assert_eq!(
            "0000001000000000000000000000000000000000000000000000000000000000",
            format!("{:064b}", tested.cache_areas())
        );

        let ca = tested.remove_cache_area(u64::from_str_radix("0000001000000000000000000000000000000000000000000000000000000000", 2)?);
        assert_eq!(ca, tested.cache_areas());
        assert_eq!(
            "0000000000000000000000000000000000000000000000000000000000000000",
            format!("{:064b}", tested.cache_areas())
        );



        Ok(())
    }

    #[tokio::test]
    async fn test_with_data() -> Result<(), MyError> {
        let tested = super::DataCacheWrapper::new(
            MockDataWithId::whatever(), 1, 2, 0
        );

        tested.with_data(0, None, None, |t|{
            println!("t={t:?}");
            Ok(())
        }).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_dump_to_cpu() -> Result<(), MyError> {
        let ctx = MockGlobalCtx::global_context();
        ctx.expect().returning(|| Box::leak(Box::new(
            mock_global_ctx::build()
                .with_age(0)
                .with_dump_all_to_cpu()
                .with_reset_all_on_cpu()
                .finish()
        )) as &'static global_ctx::MockGlobalCtx);


        let data = MockDataWithId::whatever();
        let tested = super::DataCacheWrapper::new(
            data, 0, 2, 0
        );
        assert!(tested.cached_data[1].read().await.is_none());

        tested.with_data(
            1,
            None,
            None,
             |data_on_1| {
                assert_eq!(data_on_1.device(), Some(&1));

                Ok(())
            },
        ).await?;

        assert!(tested.cached_data[1].read().await.is_some());


        tested.dump_to_cpu(1).await?;
        assert!(tested.cached_data[1].read().await.is_none());



        Ok(())
    }

}

