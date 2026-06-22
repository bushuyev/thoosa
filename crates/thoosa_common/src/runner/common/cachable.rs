#![allow(unused_parens)]
#![allow(async_fn_in_trait)]
use crate as thoosa_common;

use mockall_double::double;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;


use async_trait::async_trait;
use dashmap::{DashMap, Entry};
use futures_util::FutureExt;
use futures_util::future::{BoxFuture, Shared};
use log::{debug, trace, warn};

use parking_lot::Mutex;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use thoosa_shared::{async_perf_log, async_read_lock, async_write_lock, perf_log, sync_mutex_lock, sync_read_lock, sync_write_lock, CopyToDevice};
use thoosa_shared::Cause;
use tokio::sync::{Notify, RwLockReadGuard, futures::OwnedNotified};
use thoosa_shared::copy_mode::{CopyMode, CpuOnly, CpuOrGpu};
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::traits::{Data, Named, Select};
use crate::runner::common::{OutId, RunCtx};
use crate::runner::common::runnable::{AsyncStage, ConsumeResult, Stage};
use thoosa_shared::ThoError;
use thoosa_shared::traits::WithId;
use thoosa_shared::tracked_await::track_await;



#[async_trait]
pub trait Cached: AsyncStage + Named
where
    for <'i> Self::In<'i>: Data + WithId + Send + Sync,
    <Self as Stage>::Out: Data + WithId + Send + Sync + 'static,
    <<Self as Stage>::Out as Data>::M: DevicesWrapCopyMode<<Self as Stage>::Out>
{
    async fn cache() -> &'static DashMap<
        <<Self as Stage>::Out as WithId>::ID,
        Shared<BoxFuture<'static, Result<Arc<DevicesWrap<Self::Out>>, ThoError>>>,
    >
    where
        Self: Sized,
        Self::Out: Data + Sync + Send;

    async fn run_cached(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        mut ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError>
    where
        Self: Sized + OutId,
        for<'a> <Self as Stage>::In<'a>: Send + Sync,
        <Self as Stage>::Out: Send + Sync,
        <<Self as Stage>::Out as WithId>::ID: 'static,
        <Self as Stage>::Out: 'static,
        Self::Out: Data // <Self as Runnable>::Out:Data<Sub= <Self as Runnable>::Out>+CopyToDevice,
    {
        let key = self.out_id(WithId::id(input.as_ref()));
        let device_idx = ctx.tho_device_idx();

        debug!("Cached::run_cached: key={key:?} device_idx={device_idx} ctx.id={:?}", ctx.id());
        // #[cfg(any(test, feature = "mock_global_ctx"))]
        // let _mock_ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

        let cache = track_await(Self::cache()).await;
        let key_for_purger = key.clone();

        ctx.add_purge_gpu_closure(
            Box::new( move || {

                let key_for_purger = key_for_purger.clone();
                trace!("Cachable::purge_gpu_cache_area: 1  key={key_for_purger:?}");

                async move {
                    let Some(shared_fut) = track_await(Self::cache()).await.get(&key_for_purger) else {
                        debug!("Cachable::purge_gpu_cache_area: key_for_purger={key_for_purger:?} not found in {:?}", Self::name()); //must have been purged by dump_all_to_cpu
                        return Ok(());
                    };

                    trace!("Cachable::purge_gpu_cache_area: 2 key={key_for_purger:?} shared_fut={shared_fut:?} ");

                    if let Some(data) = shared_fut
                        .peek()  //to avoid locking on itself
                        .and_then(|peeked| peeked.as_ref().ok())
                        .map(Arc::clone) {
                        debug!("Cachable::purge_gpu_cache_area: 3 key={key_for_purger:?} Arc::strong_count={:?}", Arc::strong_count(&data));
                        if Arc::strong_count(&data) == 1 {
                            track_await(data.dump_to_cpu(device_idx)).await?;

                            debug!("Cachable::purge_gpu_cache_area: 4  key={key_for_purger:?} done");
                        }
                    }

                    Ok(())
                }.boxed()
            })

       );




        GlobalCtx::global().do_age();

        #[cfg(any(debug_assertions, tho_debug))]
        let debug_key = key.clone();
        match cache.entry(key) {
            Entry::Occupied(slot) => {
                let shared = slot.get().clone();
                drop(slot);
                track_await(shared
                    .map(|res| {
                        if let Ok(_data) = &res {
                            #[cfg(any(debug_assertions, tho_debug))]
                            debug!("Cachable::run_cached, Entry::Occupied: key={debug_key:?}");

                            // data.add_cache_area(cache_area);
                        }
                        res
                    })).await
            }
            Entry::Vacant(slot) => {
                let fut = Arc::clone(&self)
                    .run(input, ctx)
                    .boxed()
                    .shared();

                slot.insert(fut.clone()); //slot consumed, fut.await doesn't block it

                track_await(fut).await.map(|v| {
                    #[cfg(any(debug_assertions, tho_debug))]
                    debug!("Cachable::run_cached, Entry::Vacant: key={debug_key:?}");

                    v
                })
            }
        }
    }

}


pub trait DevicesWrapCopyMode<O>: CopyMode + Debug + Send + Sync
where
    O: Data + WithId + Send + Sync,
{
    fn wrap<'f>(o:O) -> DevicesWrap<O> where Self:Sized;

    async fn load_from_any_device_(
        wrap:&DevicesWrap<O>,
        to_tho_device_idx: usize,
    ) -> Result<impl Deref<Target = O> + Debug + Send, ThoError>
    where
        Self:Sized;

    fn with_data_ref<'a, R: Send, F, P>(wrap:&'a DevicesWrap<O>, to_device_idx: usize, permits_ctx: P, trace_ctx: Option<&'static str>, do_run: F) -> BoxFuture<'a, Result<R, ThoError>>
    where
        Self:Sized,
        F: FnOnce(&O, &P) -> Result<R, ThoError> + Send + 'a,
        P:RunCtx + 'a;

    fn with_selected_data_ref<'a, 'o, R: Send, F, P, S>(wrap:&'a DevicesWrap<O>, to_device_idx: usize, permits_ctx: P, selector: S, trace_ctx: Option<&'static str>, do_run: F) -> BoxFuture<'a, Result<R, ThoError>>
    where
        Self:Sized,
        O: Select<S, Output=O>,
        for <'f> F: FnOnce(&O, &P) -> Result<R, ThoError> + Send + 'a,
        S:Debug + Clone + Send + 'static,
        P:RunCtx + 'a;

    async fn with_selected_data<S, R, P>(
        wrap:&DevicesWrap<O>,
        to_device_idx: usize,
        permits_ctx: P,
        selector: S,
        do_run: impl FnOnce(<O as Select<S>>::Output, &P) -> Result<R, ThoError>,
    ) -> Result<R, ThoError>
    where
        O: Select<S>,
        <O as Select<S>>::Output: Data + CopyToDevice + WithId + Send + Sync + CopyToDevice,
        S:Clone,
        P:RunCtx,
        Self:Sized;

    fn dump_to_cpu(wrap:&DevicesWrap<O>, device_idx: usize) -> impl std::future::Future<Output = Result<(), ThoError>> + std::marker::Send
        where Self:Sized;

    fn drop_device(wrap:&DevicesWrap<O>, device_idx: usize) -> impl std::future::Future<Output = Result<(), ThoError>> + std::marker::Send
    where Self:Sized;

    fn size(wrap: &DevicesWrap<O>) -> &<O as Data>::Metrics
        where Self:Sized;
    fn is_on_cpu_before(wrap: &DevicesWrap<O>, threshold_age: u64) -> bool
        where Self:Sized;
    async fn take_data(wrap: DevicesWrap<O>, to_device_idx: usize) -> Result<O, ThoError>
        where Self:Sized;

    fn is_all_on_cpu(wrap: &DevicesWrap<O>) -> impl Future<Output=bool> + Send
        where Self:Sized;
}


impl<O> DevicesWrapCopyMode<O> for CpuOnly<O>
where
    O:Data<M=Self> + WithId + Send + Sync
{
 
    fn wrap(o:O) -> DevicesWrap<O> where Self:Sized {
        let size = o.size();
        let mem_size = o.mem_size();
        assert_eq!(o.device(), 0);

        DevicesWrap {
            id: o.id().clone(),
            cached_data: Box::new([tokio::sync::RwLock::new(Some(o))]),
            inflights: Box::new([]),
            size,
            resident_devices: Box::new([AtomicBool::new(true)]),
            gpu_resident_count: AtomicUsize::new(0),
            on_cpu_since: Default::default(),
            mem_size,
            pd: PhantomData,
        }
    }

    async fn load_from_any_device_(wrap: &DevicesWrap<O>, _to_device_idx: usize) -> Result<impl Deref<Target=O> + Debug + Send, ThoError>
    where
        Self: Sized
    {
        let lock = &wrap.cached_data[0];
        let read_guard = track_await(lock.read()).await;

        if read_guard.is_some() {
            Ok(tokio::sync::RwLockReadGuard::map(
                read_guard,
                |val: &Option<O>| val.as_ref().expect("no value behind guard"),
            ))
        } else {

            Err(Cause::NotFound.into())
        }
    }

    fn with_data_ref<'a, R: Send, F, P>(wrap:&'a DevicesWrap<O>, to_device_idx: usize, permits_ctx: P, _trace_ctx: Option<&'static str>, do_run: F) -> BoxFuture<'a, Result<R, ThoError>>
    where
            for<'f> F: FnOnce(&O, &P) -> Result<R, ThoError> + Send + 'a,
            P: RunCtx + 'a
    {
        Box::pin(async move {
            match track_await(wrap.load_from_any_device_(to_device_idx)).await {
                Ok(v) => {
                    let data = v.deref();
                    do_run(data, &permits_ctx)
                }
                Err(e) => Err(e),
            }
        })
    }

    fn with_selected_data_ref<'a, 'o, R: Send, F, P, S>(wrap:&'a DevicesWrap<O>, to_device_idx: usize, permits_ctx: P, selector: S, _trace_ctx: Option<&'static str>, do_run: F) -> BoxFuture<'a, Result<R, ThoError>>
    where
        O: Select<S, Output=O>,
        for<'f> F: FnOnce(&O, &P) -> Result<R, ThoError> + Send + 'a,
        S:Debug + Clone + Send + 'static,
        P: RunCtx + 'a
    {
        Box::pin(async move {
            match track_await(wrap.load_from_any_device_(to_device_idx)).await {
                Ok(v) => {
                    let data = v.deref();
                    let sub = data.subview(selector)?;
                    do_run(&sub, &permits_ctx)
                }
                Err(e) => Err(e),
            }
        })
    }

    async fn with_selected_data<S, R, P>(wrap: &DevicesWrap<O>, to_device_idx: usize, permits_ctx: P, selector: S, do_run: impl FnOnce(<O as Select<S>>::Output, &P) -> Result<R, ThoError>) -> Result<R, ThoError>
    where
        O: Select<S>,
        <O as Select<S>>::Output: Data + CopyToDevice + WithId + Send + Sync + CopyToDevice,
        S:Clone,
        P:RunCtx,
        Self: Sized
    {
        match track_await(wrap.load_from_any_device_(to_device_idx)).await {
            Ok(v) => {
                let data = v.deref();
                do_run(data.subview(selector)?, &permits_ctx)
            }
            Err(e) => Err(e),
        }
    }


    fn dump_to_cpu(_wrap: &DevicesWrap<O>, _device_idx: usize) -> impl std::future::Future<Output = Result<(), ThoError>> + std::marker::Send
    where
        Self: Sized
    {
        async {
            Ok(())
        }
    }

    fn drop_device(_wrap: &DevicesWrap<O>, _device_idx: usize) -> impl Future<Output=Result<(), ThoError>> + Send
    where
        Self: Sized
    {
        async {
            Ok(())
        }
    }


    fn size(wrap: &DevicesWrap<O>) -> &<O as Data>::Metrics
    where
        Self: Sized
    {
        &wrap.size //TODO move to DevicesWrap?
    }

    fn is_on_cpu_before(_wrap: &DevicesWrap<O>, _threshold_age: u64) -> bool
    where
        Self: Sized
    {
        true
    }

    async fn take_data(wrap: DevicesWrap<O>, _to_device_idx: usize) -> Result<O, ThoError>
    where
        Self: Sized
    {
        let lock = &wrap.cached_data[0];
        Ok(track_await(lock.write()).await.take().expect(&format!("no cpu data {:?}", wrap.id)))
    }

    async fn is_all_on_cpu(_wrap: &DevicesWrap<O>) -> bool {
        true
    }
}


impl<O> DevicesWrapCopyMode<O> for CpuOrGpu<O>
where
    O:Data<M=Self> + CopyToDevice + WithId + Send + Sync
{

    fn wrap(o:O) -> DevicesWrap<O> where Self:Sized {

        let id = o.id().clone();
        let size = o.size();
        let device_ordinal = o.device();
        let mem_size = o.mem_size();
        let num_devices: usize = GlobalCtx::global().device_count();
        let (resident_devices, gpu_resident_count) = initial_residency(num_devices, device_ordinal);

        debug!("DevicesWrapCopyMode<O> for CpuOrGpu<O>::wrap: id={id:?} device_ordinal={device_ordinal}");

        // #[cfg(any(test, feature = "mock_global_ctx"))]
        // let _mock_ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

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
            .map(|_device| Mutex::new(None))
            .collect::<Vec<_>>()
            .into_boxed_slice();


        DevicesWrap {
            id,
            cached_data: device_futures,
            inflights,
            size,
            resident_devices,
            gpu_resident_count,
            on_cpu_since: Default::default(),
            mem_size,
            pd: PhantomData,
        }
    }

    async fn load_from_any_device_(wrap: &DevicesWrap<O>, to_tho_device_idx: usize) -> Result<impl Deref<Target=O> + Debug + Send, ThoError>
    where
        Self: Sized
    {
        //purge shouldn't happen here since it requires MAX max_parallel and it this is being executed other runs can't have MAX max_parallel,
        debug!("DataCacheWrapper::from_any_device id={:?} to_device_idx={to_tho_device_idx}",wrap.id);
        trace!("DataCacheWrapper::from_any_device id={:?} to_device_idx={to_tho_device_idx} self.cached_data={:?}",wrap.id, wrap.cached_data);
        assert!(wrap.cached_data.len() > to_tho_device_idx, "Device index {to_tho_device_idx}(cache len is {}) too big {:?}", wrap.cached_data.len(), wrap.id);

        //dump_to_cpu can only be executed either from DumpToCpu::adjust, but it's running with set_max_other_parallel(0) so nobody should be loading anything at the momemnt, or from run_cached clean-up at thoosa_common/src/runner/common/cachable.rs:95 when the context being dropped is the last one holding
        //reference to the cached object. And in the latter case it can indeed happen so that read_guard at thoosa_common/src/runner/common/cachable.rs:416  is obtained when device data is dropped at thoosa_common/src/runner/common/cachable.rs:590 but then it will be copied back from CPU (0 device) in cycle at thoosa_common/src/runner/common/cachable.rs:429
        if wrap.is_resident_on_device(to_tho_device_idx) {
            let lock = &wrap.cached_data[to_tho_device_idx];
            let read_guard = track_await(lock.read()).await;
            if read_guard.is_some() {
                return Ok(tokio::sync::RwLockReadGuard::map(
                    read_guard,
                    |val: &Option<O>| val.as_ref().expect("no value behind guard"),
                ));
            }
            drop(read_guard);
            wrap.mark_not_resident_on_device(to_tho_device_idx);
        }

        debug!("DataCacheWrapper::from_any_device id={:?} copying to_device_idx={to_tho_device_idx}",wrap.id);

        for (source_device_idx, op) in wrap.cached_data.iter().enumerate() {
            if source_device_idx == to_tho_device_idx || !wrap.is_resident_on_device(source_device_idx) {
                continue;
            }

            let source_read_guard = track_await(op.read()).await;

            if let Some(src) = source_read_guard.as_ref() {
                return track_await(wrap.ensure_copied(to_tho_device_idx, src, true)).await.map(|v| v.expect("no value returned"));
            }
            drop(source_read_guard);
            wrap.mark_not_resident_on_device(source_device_idx);
        }


        Err(Cause::NotFound.into())
    }

    fn with_data_ref<'a, R: Send, F, P>(wrap:&'a DevicesWrap<O>, to_tho_device_idx: usize, permits_ctx: P, trace_ctx: Option<&'static str>, do_run: F) -> BoxFuture<'a, Result<R, ThoError>>
    where
        F: FnOnce(&O, &P) -> Result<R, ThoError> + Send + 'a,
        P: RunCtx + 'a
    {
        Box::pin(async move {

            match track_await(wrap.load_from_any_device_(to_tho_device_idx)).await {
                Ok(v) => {
                    let data = v.deref();
                    DevicesWrap::do_run_trace_ref(trace_ctx, |data| do_run(data, &permits_ctx), data)

                }
                Err(e) => Err(e),
            }
        })
    }

    fn with_selected_data_ref<'a, 'o, R: Send, F, P, S>(wrap:&'a DevicesWrap<O>, to_tho_device_idx: usize, permits_ctx: P, selector: S, trace_ctx: Option<&'static str>, do_run: F) -> BoxFuture<'a, Result<R, ThoError>>
    where
        O: Select<S, Output=O>,
        for<'f> F: FnOnce(&O, &P) -> Result<R, ThoError> + Send + 'a,
        S:Debug + Clone + Send + 'static,
        P: RunCtx + 'a
    {
        Box::pin(async move {
            let permits_ctx = permits_ctx;

            if to_tho_device_idx != 0 && wrap.is_resident_on_device(0) {
                let source_read_guard: RwLockReadGuard<'_, Option<O>> = track_await(wrap.cached_data[0].read()).await;
                if let Some(src) = source_read_guard.as_ref() {
                    let sub_t: O = src.subview(selector.clone())?;

                    debug!("CpuOrGpu::with_selected_data_ref: cpu sub_t.size={:?} copying to to_device_idx={to_tho_device_idx}",sub_t.size());

                    let data = track_await(sub_t.copy_to_device(to_tho_device_idx)).await?;
                    return DevicesWrap::<O>::do_run_trace_owned(trace_ctx, |data| do_run(data, &permits_ctx), data);
                }
                drop(source_read_guard);
                wrap.mark_not_resident_on_device(0);
            }

            if wrap.is_resident_on_device(to_tho_device_idx) {
                let lock = &wrap.cached_data[to_tho_device_idx];
                let read_guard: RwLockReadGuard<'_, Option<O>> = track_await(lock.read()).await;
                if read_guard.is_some() {
                    let base: &O = read_guard
                        .as_ref()
                        .expect("no value behind guard");
                    let data: O = base.subview(selector.clone())?;
                    return DevicesWrap::<O>::do_run_trace_owned(trace_ctx, |data| do_run(data, &permits_ctx), data);
                }

                drop(read_guard);
                wrap.mark_not_resident_on_device(to_tho_device_idx);
            }

            for (source_device_idx, op) in wrap.cached_data.iter().enumerate() {
                if source_device_idx == to_tho_device_idx || !wrap.is_resident_on_device(source_device_idx) {
                    continue;
                }

                let source_read_guard: RwLockReadGuard<'_, Option<O>> = track_await(op.read()).await;

                if let Some(src) = source_read_guard.as_ref() {
                    let sub_t: O = src.subview(selector.clone())?;

                    debug!("CpuOrGpu::with_selected_data_ref: sub_t.size={:?} copying to to_device_idx={to_tho_device_idx}",sub_t.size());

                    let data = track_await(sub_t.copy_to_device(to_tho_device_idx)).await?;
                    return DevicesWrap::<O>::do_run_trace_owned(trace_ctx, |data| do_run(data, &permits_ctx), data);
                }
                drop(source_read_guard);
                wrap.mark_not_resident_on_device(source_device_idx);
            }

            Err(Cause::NotFound.into())
        })
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CpuOrGpu::with_selected_data"))]
    async fn with_selected_data<S, R, P>(wrap: &DevicesWrap<O>, to_device_idx: usize, permits_ctx: P, selector: S, do_run: impl FnOnce(<O as Select<S>>::Output, &P) -> Result<R, ThoError>) -> Result<R, ThoError>
    where
        O: Select<S>,
        <O as Select<S>>::Output: Data + CopyToDevice + WithId + Send + Sync + CopyToDevice,
        S:Clone,
        P:RunCtx,
        Self: Sized
    {
        let permits_ctx = permits_ctx;

        if wrap.is_resident_on_device(to_device_idx) {
            let lock = &wrap.cached_data[to_device_idx];
            let read_guard: RwLockReadGuard<'_, Option<O>> = track_await(lock.read()).await;
            if read_guard.is_some() {
                let data = read_guard
                    .as_ref()
                    .expect("no value behind guard")
                    .subview(selector.clone())?;
                return do_run(data, &permits_ctx);
            }

            drop(read_guard);
            wrap.mark_not_resident_on_device(to_device_idx);
        }

        for (source_device_idx, op) in wrap.cached_data.iter().enumerate() {
            if source_device_idx == to_device_idx || !wrap.is_resident_on_device(source_device_idx) {
                continue;
            }

            let source_read_guard: RwLockReadGuard<'_, Option<O>> = track_await(op.read()).await;

            if let Some(src) = source_read_guard.as_ref() {
                let sub_t = src.subview(selector.clone())?;
                debug!("DataCacheWrapper::with_selected_data: sub_t.size={:?} copying to to_device_idx={to_device_idx}",sub_t.size());

                let data = track_await(sub_t.copy_to_device(to_device_idx)).await?;
                return do_run(data, &permits_ctx);
            }
            drop(source_read_guard);
            wrap.mark_not_resident_on_device(source_device_idx);
        }

        Err(Cause::NotFound.into())
    }



    fn dump_to_cpu(wrap: &DevicesWrap<O>, device_idx: usize) -> impl std::future::Future<Output = Result<(), ThoError>> + std::marker::Send
    where
        Self: Sized
    {

        async move {
            if wrap.is_all_on_cpu_resident() {
                // #[cfg(any(test, feature = "mock_global_ctx"))]
                // let _mock_ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

                wrap.mark_on_cpu_since_if_all_on_cpu(GlobalCtx::global().cache_age());
                return Ok(());
            }

            assert_ne!(device_idx, 0);


            debug!("DataCacheWrapper::dump_to_cpu: 0.  id={:?} device_idx={device_idx}",wrap.id);

            if let Some(Ok(mut lock)) = wrap.cached_data.get(device_idx).map(|v| v.try_write()) {

                if track_await(wrap.cached_data[0].read()).await.is_none() {
                    if let Some(data) = lock.as_ref() {
                        track_await(wrap.ensure_copied(0, data, false)).await?;
                    }
                    drop(lock.take());
                    wrap.mark_not_resident_on_device(device_idx);

                    debug!("DataCacheWrapper::dump_to_cpu: 1.1. id={:?} copied to cpu done device_idx={device_idx}", wrap.id);
                } else {
                    drop(lock.take());
                    wrap.mark_not_resident_on_device(device_idx);
                }
            } else {
                // If it's locked, assume it's the current task's input. DumpToCpu only runs with
                // MAX_PERMITS or when cache_area.assigned_count() == 0 (from purge_gpu_closure added in run_cached), so no other tasks should be running; skipping avoids deadlock.
                warn!("DataCacheWrapper::dump_to_cpu: can't drop self.id={:?}, locked, data={:?}",wrap.id, wrap.cached_data) //if it was dumped to cpu from DumpAllToCpu and then purged
            }

            let is_all_on_cpu = wrap.is_all_on_cpu_resident();
            let age = GlobalCtx::global().cache_age();
            if is_all_on_cpu {
                //may rewrite previous Some, but shouldn't be a problem

                // #[cfg(any(test, feature = "mock_global_ctx"))]
                // let _mock_ctx = MockGlobalCtxBuilder::install_global_ctx_mock();


                wrap.mark_on_cpu_since_if_all_on_cpu(age);
            }

            debug!("DataCacheWrapper::dump_to_cpu: 2. id={:?} device_idx={device_idx} is_all_on_cpu={is_all_on_cpu} age={age}",wrap.id);

            Ok(())
        }
    }

    fn drop_device(wrap: &DevicesWrap<O>, device_idx: usize) -> impl Future<Output=Result<(), ThoError>> + Send
    where
        Self: Sized
    {
        async move {
            assert!(device_idx > 0, "can't drop cpu");
            drop(track_await(wrap.cached_data[device_idx].write()).await.take());
            wrap.mark_not_resident_on_device(device_idx);

            Ok(())
        }
    }

    fn size(wrap: &DevicesWrap<O>) -> &<O as Data>::Metrics
    where
        Self: Sized
    {
        &wrap.size
    }

    fn is_on_cpu_before(wrap: &DevicesWrap<O>, threshold_age: u64) -> bool {
        let on_cpu_since = perf_log!(sync_read_lock!(&wrap.on_cpu_since), "CpuOrGpu::is_on_cpu_before on_cpu_since.read");
        let r = on_cpu_since.as_ref().map(|on_cpu_since| *on_cpu_since <= threshold_age ).unwrap_or(false);

        debug!("DataCacheWrapper::is_on_cpu_before: self.id={:?} self.on_cpu_since={:?} r={r} threshold_age={threshold_age}",wrap.id, wrap.on_cpu_since);

        r
    }

    async fn take_data(wrap: DevicesWrap<O>, to_device_idx: usize) -> Result<O, ThoError>
    where
        Self: Sized
    {
        let _ = track_await(wrap.load_from_any_device_(to_device_idx)).await?;

        let mut v = wrap.cached_data.into_vec();
        let lock = v.remove(to_device_idx);
        let opt: Option<O> = lock.into_inner();
        Ok(opt.expect(&format!("no data in {to_device_idx} for {:?}", wrap.id)))
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="DevicesWrapCopyMode<O> for CpuOrGpu<O>::is_all_on_cpu"))]
    async fn is_all_on_cpu(wrap: &DevicesWrap<O>) -> bool {
        trace!("DataCacheWrapper::is_all_on_cpu: self.id={:?}", wrap.id);

        let is_all_on_cpu = wrap.is_all_on_cpu_resident();
        debug!("DataCacheWrapper::is_all_on_cpu: self.id={:?} is_all_on_cpu={is_all_on_cpu}", wrap.id);
        is_all_on_cpu
    }
}

//https://github.com/rust-lang/rust/issues/100013
#[derive(Debug)]
pub struct DevicesWrap<O>
where
    O: WithId + Data + Send + Sync,
{
    id: <O as WithId>::ID,

    cached_data: Box<[tokio::sync::RwLock<Option<O>>]>,
    inflights: Box<[Mutex<Option<Arc<Notify>>>]>,
    size: O::Metrics,
    mem_size: u64,
    resident_devices: Box<[AtomicBool]>,
    gpu_resident_count: AtomicUsize,
    //contains cache age ( GlobalCtx::cache_age - increated when something is being cached from Cachable::run_cached). The smaller the value the longer it was on the cpu
    on_cpu_since: parking_lot::RwLock<Option<u64>>,
    pd: PhantomData<O>,
}

enum InflightRole {
    Leader(Arc<Notify>),
    Follower(OwnedNotified),
}

fn join_or_start_inflight(inflight: &Mutex<Option<Arc<Notify>>>) -> InflightRole {
    let mut guard = perf_log!(sync_mutex_lock!(inflight));
    if let Some(notify) = guard.as_ref() {
        // Create the future while holding the mutex. notify_waiters remembers notifications
        // sent after future creation, even if the future has not been polled yet.
        InflightRole::Follower(notify.clone().notified_owned())
    } else {
        let notify = Arc::new(Notify::new());
        *guard = Some(notify.clone());
        InflightRole::Leader(notify)
    }
}

fn initial_residency(num_devices: usize, resident_device_idx: usize) -> (Box<[AtomicBool]>, AtomicUsize) {
    assert!(resident_device_idx < num_devices, "resident device index {resident_device_idx} is out of range for {num_devices} devices");
    let resident_devices = (0..num_devices).map(|idx| AtomicBool::new(idx == resident_device_idx)).collect::<Vec<_>>().into_boxed_slice();
    let gpu_resident_count = AtomicUsize::new(usize::from(resident_device_idx > 0));
    (resident_devices, gpu_resident_count)
}

impl<O> DevicesWrap<O>
where
    O: WithId + Data + Send + Sync,
{
    fn is_resident_on_device(&self, device_idx: usize) -> bool {
        self.resident_devices[device_idx].load(Ordering::Acquire)
    }

    fn is_all_on_cpu_resident(&self) -> bool {
        self.gpu_resident_count.load(Ordering::Acquire) == 0
    }

    fn mark_resident_on_device(&self, device_idx: usize) {
        let was_resident = self.resident_devices[device_idx].swap(true, Ordering::AcqRel);
        if !was_resident && device_idx > 0 {
            let previous_gpu_count = self.gpu_resident_count.fetch_add(1, Ordering::AcqRel);
            if previous_gpu_count == 0 {
                let mut on_cpu_since = perf_log!(sync_write_lock!(&self.on_cpu_since), "DevicesWrap::mark_resident_on_device on_cpu_since.write");
                if on_cpu_since.is_some() {
                    *on_cpu_since = None;
                }
            }
        }
    }

    fn mark_not_resident_on_device(&self, device_idx: usize) {
        let was_resident = self.resident_devices[device_idx].swap(false, Ordering::AcqRel);
        if was_resident && device_idx > 0 {
            let previous_gpu_count = self.gpu_resident_count.fetch_sub(1, Ordering::AcqRel);
            assert!(previous_gpu_count > 0, "gpu resident count underflow for {:?}", self.id);
        }
    }

    fn mark_on_cpu_since_if_all_on_cpu(&self, age: u64) {
        if self.is_all_on_cpu_resident() {
            let mut on_cpu_since = perf_log!(sync_write_lock!(&self.on_cpu_since), "DevicesWrap::mark_on_cpu_since_if_all_on_cpu on_cpu_since.write");
            if on_cpu_since.is_none() {
                *on_cpu_since = Some(age);
            }
        }
    }
}

impl <O> WithId for DevicesWrap<O>
where
    O: WithId + Data + Send + Sync,

{
    type ID = O::ID;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}


struct TracingWrapper<O, D>
where
    O: Data + WithId + Send + Sync + CopyToDevice,
    D: Deref<Target = O>,
{
    o: D,
    ctx: Option<&'static str>,
}

impl<O, D> Debug for TracingWrapper<O, D>
where
    O: Data + WithId + Send + Sync + CopyToDevice,
    D: Deref<Target = O>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TracingWrapper").finish()
    }
}
impl<O, D> TracingWrapper<O, D>
where
    O: Data + WithId + Send + Sync + CopyToDevice,
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
    O: Data + WithId + Send + Sync + CopyToDevice,
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
    O: Data + WithId + Send + Sync + CopyToDevice,
    D: Deref<Target = O>,
{
    type Target = O;

    fn deref(&self) -> &Self::Target {
        &self.o
    }
}

unsafe impl<O, D> Send for TracingWrapper<O, D>
where
    O: Data + WithId + Send + Sync + CopyToDevice,
    D: Deref<Target = O>,
{
}

impl<O> DevicesWrap<O>
where
    O: Data + WithId + Send + Sync,
    O::M: DevicesWrapCopyMode<O>,
{
    pub fn wrap(o: O) -> Self {
        O::M::wrap(o)
    }


    pub fn size(&self) -> &O::Metrics {
        O::M::size(self)
    }

    pub async fn dump_to_cpu(&self, device_idx: usize) -> Result<(), ThoError> {
        O::M::dump_to_cpu(self, device_idx).await
    }

    pub async fn drop_device(&self, device_idx: usize) -> Result<(), ThoError> {
        O::M::drop_device(self, device_idx).await
    }

    pub async fn with_selected_data<S, R, P>(
        &self,
        to_device_idx: usize,
        permits_ctx: P,
        selector: S,
        do_run: impl FnOnce(<O as Select<S>>::Output, &P) -> Result<R, ThoError>,
    ) -> Result<R, ThoError>
    where
        O: Select<S>,
        <O as Select<S>>::Output: Data + CopyToDevice + WithId + Send + Sync + CopyToDevice,
        P:RunCtx,
        S: Clone,
    {
        track_await(O::M::with_selected_data(self, to_device_idx, permits_ctx, selector, do_run)).await
    }

    pub fn is_on_cpu_before(&self, threshold_age: u64) -> bool {
        O::M::is_on_cpu_before(self, threshold_age)
    }

    pub async fn take_data(
        self,
        to_device_idx: usize,
    ) -> Result<O, ThoError> {
        track_await(O::M::take_data(self, to_device_idx)).await
    }

    pub async fn load_from_any_device_(
        &self,
        to_tho_device_idx: usize,
    ) -> Result<impl Deref<Target = O> + Debug + Send, ThoError> {
        track_await(O::M::load_from_any_device_(self, to_tho_device_idx)).await
    }

    pub fn take_data_prefer_device_or_any(self, preferred_device_idx: usize) -> Result<O, ThoError> {
        let id = format!("{:?}", self.id);
        let mut cached_data = self.cached_data.into_vec();

        if preferred_device_idx < cached_data.len() {
            if let Some(data) = cached_data.swap_remove(preferred_device_idx).into_inner() {
                return Ok(data);
            }
        }

        for slot in cached_data {
            if let Some(data) = slot.into_inner() {
                return Ok(data);
            }
        }

        Err(ThoError::new_with_cause(format!("DevicesWrap::take_data_prefer_device_or_any: no data for {id}"), Cause::NotFound))
    }

    pub fn is_all_on_cpu(&self) -> impl Future<Output=bool> + Send {
        O::M::is_all_on_cpu(self)
    }

    pub fn is_on_device(&self, tho_device_idx: usize) -> bool {
        self.resident_devices.get(tho_device_idx).map(|resident| resident.load(Ordering::Acquire)).unwrap_or(false)
    }
}


impl<O> DevicesWrap<O>
where
    O: Data<M=CpuOnly<O>> + WithId + Send + Sync
{
}

impl<O> DevicesWrap<O>
where
    O: Data<M=CpuOrGpu<O>> + CopyToDevice + WithId + Send + Sync
{

    pub fn id(&self) -> &<O as WithId>::ID {
        &self.id
    }


    pub async fn clone_from_cpu(&self) -> Result<O, ThoError>
    where O: Clone {
        let lock = &self.cached_data[0];
        Ok(track_await(lock.read()).await.clone().expect("no data in cpu"))
    }



    fn do_run_trace_ref<R>(
        trace_ctx: Option<&'static str>,
        do_run: impl FnOnce(&O) -> Result<R, ThoError>,
        data: &O,
    ) -> Result<R, ThoError> {
        if trace_ctx.is_none() {
            do_run(data)
        } else {
            do_run(&TracingWrapper::new(data, trace_ctx))
        }
    }

    fn do_run_trace_owned<T, R>(
        trace_ctx: Option<&'static str>,
        do_run: impl FnOnce(&T) -> Result<R, ThoError>,
        data: T,
    ) -> Result<R, ThoError>
    where
        T: Data + CopyToDevice + WithId + Send + Sync + CopyToDevice,
    {
        if trace_ctx.is_none() {
            do_run(&data)
        } else {
            do_run(&TracingWrapper::new(&data, trace_ctx))
        }
    }


    async fn ensure_copied(
        &self,
        dst_idx: usize,
        src: &O,
        do_return: bool,
    ) -> Result<Option<RwLockReadGuard<'_, O>>, ThoError> {
        // Ensures inflight slot is cleared and waiters are notified even if the leader is canceled or panics.
        struct InflightGuard<'a> {
            inflight: &'a Mutex<Option<Arc<Notify>>>,
            notify: Arc<Notify>,
            cleared: bool,
        }

        impl<'a> InflightGuard<'a> {
            fn new(inflight: &'a Mutex<Option<Arc<Notify>>>, notify: Arc<Notify>) -> Self {
                Self {
                    inflight,
                    notify,
                    cleared: false,
                }
            }

            fn clear(&mut self) {
                if self.cleared {
                    return;
                }
                self.cleared = true;
                let mut g = perf_log!(sync_mutex_lock!(self.inflight));
                if g
                    .as_ref()
                    .map(|n| Arc::ptr_eq(n, &self.notify))
                    .unwrap_or(false)
                {
                    *g = None;
                }
                drop(g);
                self.notify.notify_waiters();
            }
        }

        impl Drop for InflightGuard<'_> {
            fn drop(&mut self) {
                self.clear();
            }
        }

        debug!("DevicesWrap::ensure_copied: self.id={:?} dst_idx={dst_idx} do_return={do_return} src={src:?}", self.id);

        loop {
            if self.is_resident_on_device(dst_idx) {
                let lock = &self.cached_data[dst_idx];
                let dst_read = track_await(lock.read()).await;
                // ---- Fast path
                if dst_read.is_some() {
                    debug!("DevicesWrap::ensure_copied: self.id={:?} fast path", self.id);

                    if do_return {
                        return Ok(Some(RwLockReadGuard::map(dst_read, |v| {
                            v.as_ref().unwrap()
                        })));
                    } else {
                        return Ok(None);
                    }
                }
                drop(dst_read);
                self.mark_not_resident_on_device(dst_idx);
            }

            // ---- Try to join or become the leader
            // (Do not hold this lock while awaiting other locks, except briefly to set or read the notify)
            let leader_notify = match join_or_start_inflight(&self.inflights[dst_idx]) {
                InflightRole::Follower(notified) => {
                    // follower: wait until the leader finishes, then try fast path again
                    track_await(notified)
                    .await;
                    continue;
                }
                InflightRole::Leader(notify) => notify,
            };

            // ---- Leader path
            let mut inflight_guard = InflightGuard::new(
                &self.inflights[dst_idx],
                leader_notify,
            );

            // Double-check: maybe dst was filled after we dropped the read lock
            if self.is_resident_on_device(dst_idx) {
                let lock = &self.cached_data[dst_idx];
                let dst_read = track_await(lock.read()).await;
                if dst_read.is_some() {
                    inflight_guard.clear();
                    continue;
                }
                drop(dst_read);
                self.mark_not_resident_on_device(dst_idx);
            }

            debug!("DevicesWrap::ensure_copied: self.id={:?} copy_to_device from src", self.id);
            let produced = track_await(src.copy_to_device(dst_idx)).await;

            let produced = match produced {
                Ok(v) => v,
                Err(e) => {
                    inflight_guard.clear();//TODO not needed, called from Drop?
                    return Err(e);
                }
            };


            // Install into dst (short critical section)
            {
                debug!("DevicesWrap::ensure_copied: self.id={:?} install copied to dst_idx={dst_idx} produced.device={:?}", self.id, produced.device());

                let lock = &self.cached_data[dst_idx];
                let mut w = track_await(lock.write()).await;
                if w.is_none() {
                    *w = Some(produced);
                    self.mark_resident_on_device(dst_idx);
                }
            }

            // Clear inflight and wake followers
            inflight_guard.clear();//TODO not needed, called from Drop?
        }
    }

    #[cfg(test)]
    pub fn cached_data(&self) -> &Box<[tokio::sync::RwLock<Option<O>>]> {
        &self.cached_data
    }
}

///To let OOMHandler accept both plain wrapped data as it's stored in caches and composite inputs like module + tensor or batch + motif
pub trait WithDataRef: WithId + Debug + Send + Sync {
    type O<'o>: Data + /*CopyToDevice +*/ WithId + Send + Sync;

    fn with_data_ref<'a, R: Send + 'a, F, P>(
        &'a self,
        to_device_idx: usize,
        permits_ctx: P,
        trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> impl Future<Output = Result<R, ThoError>> + Send + 'a
    where
       for <'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
        P:RunCtx + 'a;



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
    S: Clone + Debug + Send + 'static,
{
    fn with_selected_data_ref<'a, 'o, R: Send + 'a, F, P>(
        &'a self,
        to_device_idx: usize,
        _permits_ctx: P,
        selector: S,
        trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> impl Future<Output = Result<R, ThoError>> + Send + 'a where
     // Self::O<'o>: Select<S, Output = Self::O<'o>>,
     for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
     P:RunCtx + 'a;

}

/// Owned-data analogue of WithDataRef for sync-stage consume paths that need executor-controlled materialization.
pub trait ConsumeData {
    type O: Data + WithId + Send + Sync;
    //it should be run from OOM OOMHandler::consume where tensors can't be dumped to cpu while being used
    fn consume<R: Send, F, P>(
        self,
        to_device_idx: usize,
        permits_ctx: P,
        trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> impl Future<Output = ConsumeResult<Self::O, R>> + Send
    where
        F: FnOnce(Self::O, P) -> ConsumeResult<Self::O, R> + Send,
        P: RunCtx;

    fn burp(o:Self::O) -> impl Future<Output = Self> + Send;
}

impl <O> WithDataRef for DevicesWrap<O>
where
    O: Data + WithId + Send + Sync,
    O::M: DevicesWrapCopyMode<O>,
{
    type O<'o> = O;

    fn with_data_ref<'a, R: Send + 'a, F, P>(&'a self, to_device_idx: usize, permits_ctx: P, trace_ctx: Option<&'static str>, do_run: F) -> impl Future<Output = Result<R, ThoError>> + Send + 'a
    where
            for <'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
            P:RunCtx + 'a
    {
        O::M::with_data_ref(self, to_device_idx, permits_ctx, trace_ctx, do_run)
    }

    fn mem_size(&self) -> u64 {
        self.mem_size
    }

    fn _size<'f>(&self) -> <Self::O<'f> as Data>::Metrics {
        self.size.clone()
    }
}

impl<O, S> WithSelectDataRef<S> for DevicesWrap<O>
where
    S:Debug + Clone + Send + 'static,
    O: Select<S, Output=O> + Data + WithId + Send + Sync + CopyToDevice,
    O::M: DevicesWrapCopyMode<O>,
{
    fn with_selected_data_ref<'a, 'o, R: Send + 'a, F, P>(&'a self, to_device_idx: usize, permits_ctx: P, selector: S, trace_ctx: Option<&'static str>, do_run: F) -> impl Future<Output = Result<R, ThoError>> + Send + 'a
    where
        Self::O<'o>: Select<S, Output=Self::O<'o>>,
        for <'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
        P:RunCtx + 'a
    {
        O::M::with_selected_data_ref(self, to_device_idx, permits_ctx, selector, trace_ctx, do_run)
    }

}

impl <D, E> DevicesWrap<DataWithExtra<D, E>>
where
    D: Data + CopyToDevice + WithId + Sync + Send,
    E: Clone + Debug + Send + Sync
{

    pub async fn extra(&self) -> Result<E, ThoError> {
        // A concurrent dump_to_cpu can move data from a device slot to CPU while this method is scanning slots; the second pass avoids a false NotFound after waiting on the old device slot.
        for _attempt in 0..2 {
            for op in self.cached_data.iter() {
                let source_read_guard = track_await(op.read()).await;

                if let Some(data_with_extra) = source_read_guard.as_ref() {
                    return Ok(data_with_extra.extra().clone());
                }
            }
        }

        Err(ThoError::new(format!("Can't get extra from {:?}", self.id)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use async_trait::async_trait;
    use parking_lot::Mutex;
    use tokio::sync::{Notify, Semaphore};
    use tokio::time::timeout;
    use thoosa_shared::{CopyToDevice, async_perf_log, async_read_lock};
    use thoosa_shared::copy_mode::CpuOrGpu;
    use thoosa_shared::tracked_await::track_await;
    use thoosa_shared::traits::{DummyCachableId, WithId};
    use crate::runner::global_ctx::MockGlobalCtx;
    use thoosa_with_data::with_data;
    extern crate self as caching_runner;
    pub use crate::runner::common::*;
    use crate::runner::cache_areas::CacheArea;
    use crate::runner::global_ctx::mock_global_ctx;
    use crate::runner::global_ctx;
    use crate::runner::common::cachable::{DevicesWrap, InflightRole, WithDataRef, join_or_start_inflight};
    use crate::runner::common::mock_run_ctx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::runner::test_utils::MockDataWithId;
    use crate as thoosa_common;

    #[derive(Clone, Copy, Debug)]
    enum CopyBehavior {
        BlockFirst,
        FailFirst,
        FailAlways,
    }

    #[derive(Debug)]
    struct ControlledCopyState {
        behavior: CopyBehavior,
        calls: AtomicUsize,
        entered: Semaphore,
        release: Semaphore,
    }

    impl ControlledCopyState {
        fn new(behavior: CopyBehavior) -> Arc<Self> {
            Arc::new(Self {
                behavior,
                calls: AtomicUsize::new(0),
                entered: Semaphore::new(0),
                release: Semaphore::new(0),
            })
        }
    }

    #[derive(Debug)]
    struct ControlledCopyData {
        id: DummyCachableId,
        device: usize,
        state: Arc<ControlledCopyState>,
    }

    impl ControlledCopyData {
        fn new(device: usize, behavior: CopyBehavior) -> (Self, Arc<ControlledCopyState>) {
            let state = ControlledCopyState::new(behavior);
            (
                Self {
                    id: DummyCachableId("controlled-copy"),
                    device,
                    state: state.clone(),
                },
                state,
            )
        }

        fn device(&self) -> usize {
            self.device
        }
    }

    impl WithId for ControlledCopyData {
        type ID = DummyCachableId;

        fn id(&self) -> &Self::ID {
            &self.id
        }
    }

    impl thoosa_shared::traits::Data for ControlledCopyData {
        type Metrics = [usize; 1];
        type M = CpuOrGpu<Self>;

        fn size(&self) -> Self::Metrics {
            [0]
        }

        fn device(&self) -> usize {
            self.device
        }

        fn mem_size(&self) -> u64 {
            0
        }
    }

    #[async_trait]
    impl CopyToDevice for ControlledCopyData {
        async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
            let call = self.state.calls.fetch_add(1, Ordering::SeqCst);
            if call == 0 && matches!(self.state.behavior, CopyBehavior::BlockFirst | CopyBehavior::FailFirst) {
                self.state.entered.add_permits(1);
                self.state.release.acquire().await.expect("release semaphore closed").forget();
            }
            if matches!(self.state.behavior, CopyBehavior::FailAlways) || call == 0 && matches!(self.state.behavior, CopyBehavior::FailFirst) {
                return Err(ThoError::new("controlled copy failure"));
            }

            Ok(Self {
                id: self.id.clone(),
                device: dst_device,
                state: self.state.clone(),
            })
        }
    }

    async fn wait_for_inflight_participants(wrap: &DevicesWrap<ControlledCopyData>, device_idx: usize, expected: usize) {
        timeout(Duration::from_secs(5), async {
            loop {
                let participants = wrap.inflights[device_idx].lock().as_ref().map(Arc::strong_count).unwrap_or(0);
                if participants >= expected {
                    return;
                }
                tokio::task::yield_now().await;
            }
        }).await.expect("timed out waiting for inflight participants");
    }

    #[tokio::test]
    async fn inflight_follower_observes_notify_before_first_poll() {
        let notify = Arc::new(Notify::new());
        let inflight = Mutex::new(Some(notify.clone()));
        let follower = match join_or_start_inflight(&inflight) {
            InflightRole::Follower(follower) => follower,
            InflightRole::Leader(_) => panic!("expected follower"),
        };

        notify.notify_waiters();
        timeout(Duration::from_secs(5), follower).await.expect("follower missed notification");
    }

    #[tokio::test]
    async fn ensure_copied_error_wakes_follower() -> Result<(), ThoError> {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false);
        let (data, state) = ControlledCopyData::new(0, CopyBehavior::FailFirst);
        let tested = Arc::new(DevicesWrap::wrap(data));

        let leader = tokio::spawn({
            let tested = tested.clone();
            async move {
                let data = tested.load_from_any_device_(1).await?;
                Ok::<usize, ThoError>(data.device())
            }
        });
        state.entered.acquire().await.expect("entered semaphore closed").forget();

        let follower = tokio::spawn({
            let tested = tested.clone();
            async move {
                let data = tested.load_from_any_device_(1).await?;
                Ok::<usize, ThoError>(data.device())
            }
        });
        wait_for_inflight_participants(&tested, 1, 3).await;
        state.release.add_permits(1);

        assert!(leader.await.expect("leader task panicked").is_err());
        assert_eq!(timeout(Duration::from_secs(5), follower).await.expect("follower timed out").expect("follower task panicked")?, 1);
        assert_eq!(state.calls.load(Ordering::SeqCst), 2);
        assert!(tested.inflights[1].lock().is_none());
        Ok(())
    }

    #[tokio::test]
    async fn ensure_copied_cancellation_wakes_follower() -> Result<(), ThoError> {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false);
        let (data, state) = ControlledCopyData::new(0, CopyBehavior::BlockFirst);
        let tested = Arc::new(DevicesWrap::wrap(data));

        let leader = tokio::spawn({
            let tested = tested.clone();
            async move {
                let data = tested.load_from_any_device_(1).await?;
                Ok::<usize, ThoError>(data.device())
            }
        });
        state.entered.acquire().await.expect("entered semaphore closed").forget();

        let follower = tokio::spawn({
            let tested = tested.clone();
            async move {
                let data = tested.load_from_any_device_(1).await?;
                Ok::<usize, ThoError>(data.device())
            }
        });
        wait_for_inflight_participants(&tested, 1, 3).await;
        leader.abort();
        assert!(leader.await.expect_err("leader was not cancelled").is_cancelled());

        assert_eq!(timeout(Duration::from_secs(5), follower).await.expect("follower timed out").expect("follower task panicked")?, 1);
        assert_eq!(state.calls.load(Ordering::SeqCst), 2);
        assert!(tested.inflights[1].lock().is_none());
        Ok(())
    }

    #[tokio::test]
    async fn dump_to_cpu_error_preserves_source_data() {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false);
        let (data, state) = ControlledCopyData::new(1, CopyBehavior::FailAlways);
        let tested = DevicesWrap::wrap(data);

        assert!(tested.dump_to_cpu(1).await.is_err());
        assert!(tested.cached_data[0].read().await.is_none());
        assert!(tested.cached_data[1].read().await.is_some());
        assert!(tested.is_on_device(1));
        assert_eq!(state.calls.load(Ordering::SeqCst), 1);
        assert!(tested.inflights[0].lock().is_none());
    }


    #[tokio::test]
    async fn test_dump_to_cpu() -> Result<(), ThoError> {

        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false);

        let data = MockDataWithId::whatever();
        let tested = DevicesWrap::wrap(data);
        assert!(tested.is_all_on_cpu().await);
        let lock = &tested.cached_data[1];
        assert!(track_await(lock.read()).await.is_none());

        let ctx_for_data = mock_run_ctx::default(1);
        tested
            .with_data_ref(1,
               ctx_for_data, None, |data_on_1: &MockDataWithId, _ctx| {
                   assert_eq!(data_on_1.device(), &1);

                   Ok(())
               })
            .await?;

        let lock = &tested.cached_data[1];
        assert!(track_await(lock.read()).await.is_some());
        assert!(!tested.is_all_on_cpu().await);

        tested.dump_to_cpu(1).await?;
        let lock = &tested.cached_data[1];
        assert!(track_await(lock.read()).await.is_none());
        assert!(tested.is_all_on_cpu().await);
        assert!(tested.is_on_cpu_before(0));

        Ok(())
    }

    #[tokio::test]
    async fn test_with_data() -> Result<(), ThoError> {
        let ctx = MockGlobalCtx::global_context();
        ctx.expect().returning(|| {
            Box::leak(Box::new(
                mock_global_ctx::build_default(),
            )) as &'static MockGlobalCtx
        });

        let wrap_a = Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "a")));
        let wrap_b = Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "d")));
        let wrap_vec = vec![
            Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "b"))),
            Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "c"))),
        ];
        let wrap_vec_second = vec![
            Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "e"))),
            Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "f"))),
        ];



        //TODO ensure that asserts were actually invoked
        let ctx_for_data = mock_run_ctx::default(1);
        with_data!(ctx_for_data, wrap_a, [wrap_vec], wrap_b, [wrap_vec_second];
            |_ctx, a: &MockDataWithId, first: &[&MockDataWithId], b: &MockDataWithId, second: &[&MockDataWithId]|{
                assert_eq!(a.device(), &1);
                assert_eq!(b.device(), &1);
                assert_eq!(first.len(), 2);
                assert!(first.iter().all(|item| *item.device() == 1));
                assert_eq!(second.len(), 2);
                assert!(second.iter().all(|item| *item.device() == 1));
                Ok::<(), ThoError>(())
        })?;
        Ok(())
    }

    #[tokio::test]
    async fn test_with_data_single_val() -> Result<(), ThoError> {
        let ctx = MockGlobalCtx::global_context();
        ctx.expect().returning(|| {
            Box::leak(Box::new(
                mock_global_ctx::build_default(),
            )) as &'static global_ctx::MockGlobalCtx
        });

        let wrap_a = Arc::new(DevicesWrap::wrap(MockDataWithId::for_device_and_id(0, "a")));

        let ctx_for_data = mock_run_ctx::default(1);
        with_data!(ctx_for_data, wrap_a;
            |_ctx, a: &MockDataWithId|{
                assert_eq!(a.device(), &1);
                Ok::<(), ThoError>(())
        })?;
        Ok(())
    }
}
