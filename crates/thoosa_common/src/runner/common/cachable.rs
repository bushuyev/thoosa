#![allow(unused_parens)]
#![allow(async_fn_in_trait)]
use crate as thoosa_common;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;

#[cfg(any(test, feature = "mock_global_ctx"))]
use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;

use async_trait::async_trait;
use dashmap::{DashMap, Entry};
use futures_util::FutureExt;
use futures_util::future::{BoxFuture, Shared};
use log::{debug, error, trace, warn};
use mockall_double::double;
use parking_lot::Mutex;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;
use thoosa_shared::{with_async_perf_log, with_perf_log, CopyToDevice, LOCK_TIMEOUT};
use thoosa_shared::Cause;
use tokio::sync::{Notify, RwLockReadGuard};
use crate::runner::cache_areas::CacheArea;
use thoosa_shared::copy_mode::{CopyMode, CpuOnly, CpuOrGpu};
use crate::runner::cache_areas::CacheAreaId;
use thoosa_shared::traits::{Data, Named, Select};
use crate::runner::common::{OutId, RunCtx};
use crate::runner::common::runnable::{AsyncStage, ConsumeResult, Stage};
use thoosa_shared::ThoError;
use thoosa_shared::traits::WithId;


async fn timed_tokio_read<'a, T>(lock: &'a tokio::sync::RwLock<T>, ctx: &'static str) -> tokio::sync::RwLockReadGuard<'a, T> {
    with_async_perf_log(
        async || tokio::time::timeout(LOCK_TIMEOUT, lock.read()).await.unwrap_or_else(|_| panic!("{ctx}: timed out waiting for tokio::sync::RwLock read lock")),
        ctx,
    )
    .await
}

async fn timed_tokio_write<'a, T>(lock: &'a tokio::sync::RwLock<T>, ctx: &'static str) -> tokio::sync::RwLockWriteGuard<'a, T> {
    with_async_perf_log(
        async || tokio::time::timeout(LOCK_TIMEOUT, lock.write()).await.unwrap_or_else(|_| panic!("{ctx}: timed out waiting for tokio::sync::RwLock write lock")),
        ctx,
    )
    .await
}

fn timed_parking_mutex_lock<'a, T>(lock: &'a Mutex<T>, ctx: &'static str) -> parking_lot::MutexGuard<'a, T> {
    with_perf_log(
        || lock.try_lock_for(LOCK_TIMEOUT).unwrap_or_else(|| panic!("{ctx}: timed out waiting for parking_lot::Mutex lock")),
        ctx,
    )
}

fn timed_parking_rw_read<'a, T>(lock: &'a parking_lot::RwLock<T>, ctx: &'static str) -> parking_lot::RwLockReadGuard<'a, T> {
    with_perf_log(
        || lock.try_read_for(LOCK_TIMEOUT).unwrap_or_else(|| panic!("{ctx}: timed out waiting for parking_lot::RwLock read lock")),
        ctx,
    )
}

fn timed_parking_rw_write<'a, T>(lock: &'a parking_lot::RwLock<T>, ctx: &'static str) -> parking_lot::RwLockWriteGuard<'a, T> {
    with_perf_log(
        || lock.try_write_for(LOCK_TIMEOUT).unwrap_or_else(|| panic!("{ctx}: timed out waiting for parking_lot::RwLock write lock")),
        ctx,
    )
}

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
        let device_idx = ctx.device_idx();

        debug!("Cached::run_cached: key={key:?} device_idx={device_idx} ctx.id={:?}", ctx.id());
        // #[cfg(any(test, feature = "mock_global_ctx"))]
        // let _mock_ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

        let cache = Self::cache().await;
        let key_for_purger = key.clone();

        ctx.add_purge_gpu_closure(
            Box::new( move || {

                // #[cfg(any(test, feature = "mock_global_ctx"))]
                // let _mock_ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

                let key_for_purger = key_for_purger.clone();
                trace!("Cachable::purge_gpu_cache_area: 1  key={key_for_purger:?}");

                async move {

                    let maybe_data = if let Some(value) = Self::cache().await.get(&key_for_purger) {
                        let shared_fut = value.value().clone();
                        trace!("Cachable::purge_gpu_cache_area: 2 key={key_for_purger:?} shared_fut={shared_fut:?} ");
                        shared_fut
                            .peek()  //to avoid locking on itself
                            .and_then(|peeked| peeked.as_ref().ok())
                            .map(Arc::clone)
                    } else {
                        warn!("Cachable::purge_gpu_cache_area: key_for_purger={key_for_purger:?} not found in {:?}", Self::name()); //warning
                        None
                    };

                    if let Some(data) = maybe_data {

                        // data.remove_cache_area(cache_area);

                        debug!("Cachable::purge_gpu_cache_area: 3 key={key_for_purger:?} Arc::strong_count={:?}", Arc::strong_count(&data));
                        if Arc::strong_count(&data) == 1 {
                            data.dump_to_cpu(device_idx).await?;

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
                shared
                    .map(|res| {
                        if let Ok(data) = &res {
                            #[cfg(any(debug_assertions, tho_debug))]
                            debug!("Cachable::run_cached, Entry::Occupied: key={debug_key:?}");

                            // data.add_cache_area(cache_area);
                        }
                        res
                    })
                    .await
            }
            Entry::Vacant(slot) => {
                let fut = Arc::clone(&self)
                    .run(input, ctx)
                    .boxed()
                    .shared();

                slot.insert(fut.clone()); //slot consumed, fut.await doesn't block it

                fut.await.map(|v| {
                    #[cfg(any(debug_assertions, tho_debug))]
                    debug!("Cachable::run_cached, Entry::Vacant: key={debug_key:?}");
                    // v.add_cache_area(cache_area);
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
        to_device_idx: usize,
    ) -> Result<impl Deref<Target = O> + Debug + Send, ThoError>
    where
        Self:Sized;

    fn with_data_ref<'a, R, F, P>(wrap:&'a DevicesWrap<O>, to_device_idx: usize, permits_ctx: P, trace_ctx: Option<&'static str>, do_run: F) -> BoxFuture<'a, Result<R, ThoError>>
    where
        Self:Sized,
        F: FnOnce(&O, &P) -> Result<R, ThoError> + Send + 'a,
        P:RunCtx + 'a;

    async fn with_selected_data_ref<'o, R, F, P, S>(wrap:&DevicesWrap<O>, to_device_idx: usize, permits_ctx: P, selector: S, trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoError>
    where
        Self:Sized,
        O: Select<S, Output=O>,
        for <'f> F: FnOnce(&O, &P) -> Result<R, ThoError>,
        S:Debug + Clone + 'static,
        P:RunCtx;

    fn with_selected_data_ref_send<'a, 'o, R, F, P, S>(wrap:&'a DevicesWrap<O>, to_device_idx: usize, permits_ctx: P, selector: S, trace_ctx: Option<&'static str>, do_run: F) -> BoxFuture<'a, Result<R, ThoError>>
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
            on_cpu_since: Default::default(),
            mem_size,
            pd: PhantomData,
        }
    }

    async fn load_from_any_device_(wrap: &DevicesWrap<O>, _to_device_idx: usize) -> Result<impl Deref<Target=O> + Debug + Send, ThoError>
    where
        Self: Sized
    {
        let read_guard = timed_tokio_read(&wrap.cached_data[0], "CpuOnly::load_from_any_device_").await;

        if read_guard.is_some() {
            Ok(tokio::sync::RwLockReadGuard::map(
                read_guard,
                |val: &Option<O>| val.as_ref().expect("no value behind guard"),
            ))
        } else {

            Err(Cause::NotFound.into())
        }
    }

    fn with_data_ref<'a, R, F, P>(wrap:&'a DevicesWrap<O>, to_device_idx: usize, permits_ctx: P, _trace_ctx: Option<&'static str>, do_run: F) -> BoxFuture<'a, Result<R, ThoError>>
    where
            for<'f> F: FnOnce(&O, &P) -> Result<R, ThoError> + Send + 'a,
            P: RunCtx + 'a
    {
        Box::pin(async move {
            match wrap.load_from_any_device_(to_device_idx).await {
                Ok(v) => {
                    let data = v.deref();
                    do_run(data, &permits_ctx)
                }
                Err(e) => Err(e),
            }
        })
    }

    async fn with_selected_data_ref<'o, R, F, P, S>(wrap:&DevicesWrap<O>, to_device_idx: usize, permits_ctx: P, selector: S, _trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoError>
    where
        O: Select<S, Output=O>,
        for<'f> F: FnOnce(&O, &P) -> Result<R, ThoError>,
        S:Debug + Clone + 'static,
        P: RunCtx
    {
        match wrap.load_from_any_device_(to_device_idx).await {
            Ok(v) => {
                let data = v.deref();
                let sub = data.subview(selector)?;
                do_run(&sub, &permits_ctx)
            }
            Err(e) => Err(e),
        }
    }

    fn with_selected_data_ref_send<'a, 'o, R, F, P, S>(wrap:&'a DevicesWrap<O>, to_device_idx: usize, permits_ctx: P, selector: S, _trace_ctx: Option<&'static str>, do_run: F) -> BoxFuture<'a, Result<R, ThoError>>
    where
        O: Select<S, Output=O>,
        for<'f> F: FnOnce(&O, &P) -> Result<R, ThoError> + Send + 'a,
        S:Debug + Clone + Send + 'static,
        P: RunCtx + 'a
    {
        Box::pin(async move {
            match wrap.load_from_any_device_(to_device_idx).await {
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
        match wrap.load_from_any_device_(to_device_idx).await {
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
        Ok(timed_tokio_write(&wrap.cached_data[0], "CpuOnly::take_data").await.take().expect(&format!("no cpu data {:?}", wrap.id)))
    }

    async fn is_all_on_cpu(_wrap: &DevicesWrap<O>) -> bool {
        true
    }
}


// impl<O, S> SelectMode<S> for CpuOrGpu<O>
// where
//     O:Data + CopyToDevice+ WithId + Send + Sync + Select<S>,
//     S:Clone
// {
// }


impl<O> DevicesWrapCopyMode<O> for CpuOrGpu<O>
where
    O:Data<M=Self> + CopyToDevice + WithId + Send + Sync
{

    fn wrap(o:O) -> DevicesWrap<O> where Self:Sized {

        let id = o.id().clone();
        let size = o.size();
        let device_ordinal = o.device();
        let mem_size = o.mem_size();

        debug!("DevicesWrapCopyMode<O> for CpuOrGpu<O>::wrap: id={id:?} device_ordinal={device_ordinal}");

        // #[cfg(any(test, feature = "mock_global_ctx"))]
        // let _mock_ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

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
            .map(|_device| Mutex::new(None))
            .collect::<Vec<_>>()
            .into_boxed_slice();


        DevicesWrap {
            id,
            cached_data: device_futures,
            inflights,
            size,
            on_cpu_since: Default::default(),
            mem_size,
            pd: PhantomData,
        }
    }

    async fn load_from_any_device_(wrap: &DevicesWrap<O>, to_device_idx: usize) -> Result<impl Deref<Target=O> + Debug + Send, ThoError>
    where
        Self: Sized
    {
        //purge shouldn't happen here since it requires MAX max_parallel and it this is being executed other runs can't have MAX max_parallel,
        //TODO add GlobalCtx::purge_lock just in case?
        debug!("DataCacheWrapper::from_any_device id={:?} to_device_idx={to_device_idx}",wrap.id);
        trace!("DataCacheWrapper::from_any_device id={:?} to_device_idx={to_device_idx} self.cached_data={:?}",wrap.id, wrap.cached_data);
        assert!(wrap.cached_data.len() > to_device_idx, "Device index {to_device_idx}(cache len is {}) too big {:?}",wrap.cached_data.len(), wrap.id);

        if to_device_idx != 0 {
            *timed_parking_rw_write(&wrap.on_cpu_since, "CpuOrGpu::load_from_any_device_ on_cpu_since.write") = None;
        }

        let read_guard = timed_tokio_read(&wrap.cached_data[to_device_idx], "CpuOrGpu::load_from_any_device_ target read").await;

        if read_guard.is_some() {
            Ok(tokio::sync::RwLockReadGuard::map(
                read_guard,
                |val: &Option<O>| val.as_ref().expect("no value behind guard"),
            ))
        } else {
            drop(read_guard);

            debug!("DataCacheWrapper::from_any_device id={:?} copying to_device_idx={to_device_idx}",wrap.id);

            for (source_device_idx, op) in wrap.cached_data.iter().enumerate() {
                let source_read_guard = timed_tokio_read(op, "CpuOrGpu::load_from_any_device_ source read").await;

                if source_device_idx != to_device_idx && source_read_guard.is_some() {
                    drop(source_read_guard);
                    return wrap.copy_and_get(source_device_idx, to_device_idx).await;
                }
            }
            Err(Cause::NotFound.into())
        }
    }

    fn with_data_ref<'a, R, F, P>(wrap:&'a DevicesWrap<O>, to_device_idx: usize, permits_ctx: P, trace_ctx: Option<&'static str>, do_run: F) -> BoxFuture<'a, Result<R, ThoError>>
    where
        F: FnOnce(&O, &P) -> Result<R, ThoError> + Send + 'a,
        P: RunCtx + 'a
    {
        Box::pin(async move {
            // let mut permits_ctx = permits_ctx;
            // if permits_ctx.acquired_permits() == 0 {
            //     let permit = GlobalCtx::global().acquire_one(to_device_idx, Arc::new(permits_ctx.id().clone())).await?;
            //     permits_ctx.add_permit(permit);
            // }


            match wrap.load_from_any_device_(to_device_idx).await {
                Ok(v) => {
                    let data = v.deref();

                    let stream = crate::copy_ctx::CopyCtx::instance(1).acquire_stream().await;
                    let _guard = thoosa_shims::stream::ThoStreamGuard::new(&stream)?;
                    let r =DevicesWrap::do_run_trace_ref(trace_ctx, |data| do_run(data, &permits_ctx), data);
                    stream.synchronize()?;
                    r
                }
                Err(e) => Err(e),
            }
        })
    }

    async fn with_selected_data_ref<'o, R, F, P, S>(wrap:&DevicesWrap<O>, to_device_idx: usize, permits_ctx: P, selector: S, trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoError>
    where
        O: Select<S, Output=O>,
        for<'f> F: FnOnce(&O, &P) -> Result<R, ThoError>,
        S:Debug + Clone + 'static,
        P: RunCtx
    {
        let permits_ctx = permits_ctx;
        // if permits_ctx.acquired_permits() == 0 {
        //     let permit = GlobalCtx::global().acquire_one(to_device_idx, Arc::new(permits_ctx.id().clone())).await?;
        //     permits_ctx.add_permit(permit);
        // }

        let read_guard: RwLockReadGuard<'_, Option<O>> = timed_tokio_read(&wrap.cached_data[to_device_idx], "CpuOrGpu::with_selected_data_ref target read").await;

        if read_guard.is_some() {
            let base: &O = read_guard
                .as_ref()
                .expect("no value behind guard");


            let data: O = base.subview(selector.clone())?;

            return DevicesWrap::<O>::do_run_trace_owned(trace_ctx, |data| do_run(data, &permits_ctx), data);
        }

        drop(read_guard);

        for (source_device_idx, op) in wrap.cached_data.iter().enumerate() {
            if source_device_idx == to_device_idx {
                continue;
            }

            let source_read_guard: RwLockReadGuard<'_, Option<O>> = timed_tokio_read(op, "CpuOrGpu::with_selected_data_ref source read").await;

            if let Some(src) = source_read_guard.as_ref() {
                let sub_t: O = src.subview(selector.clone())?;

                debug!("CpuOrGpu::with_selected_data_ref: sub_t.size={:?} copying to to_device_idx={to_device_idx}",sub_t.size());

                let data = sub_t.copy_to_device(to_device_idx).await?;

                let stream = crate::copy_ctx::CopyCtx::instance(1).acquire_stream().await;
                let _guard = thoosa_shims::stream::ThoStreamGuard::new(&stream)?;
                let r =DevicesWrap::<O>::do_run_trace_owned(trace_ctx, |data| do_run(data, &permits_ctx), data);
                stream.synchronize()?;
                return r;
            }
        }

        Err(Cause::NotFound.into())
    }

    fn with_selected_data_ref_send<'a, 'o, R, F, P, S>(wrap:&'a DevicesWrap<O>, to_device_idx: usize, permits_ctx: P, selector: S, trace_ctx: Option<&'static str>, do_run: F) -> BoxFuture<'a, Result<R, ThoError>>
    where
        O: Select<S, Output=O>,
        for<'f> F: FnOnce(&O, &P) -> Result<R, ThoError> + Send + 'a,
        S:Debug + Clone + Send + 'static,
        P: RunCtx + 'a
    {
        Box::pin(async move {
            let permits_ctx = permits_ctx;
            // if permits_ctx.acquired_permits() == 0 {
            //     let permit = GlobalCtx::global().acquire_one(to_device_idx, Arc::new(permits_ctx.id().clone())).await?;
            //     permits_ctx.add_permit(permit);
            // }

            let read_guard: RwLockReadGuard<'_, Option<O>> = timed_tokio_read(&wrap.cached_data[to_device_idx], "CpuOrGpu::with_selected_data_ref target read").await;

            if read_guard.is_some() {
                let base: &O = read_guard
                    .as_ref()
                    .expect("no value behind guard");


                let data: O = base.subview(selector.clone())?;

                return DevicesWrap::<O>::do_run_trace_owned(trace_ctx, |data| do_run(data, &permits_ctx), data);
            }

            drop(read_guard);

            for (source_device_idx, op) in wrap.cached_data.iter().enumerate() {
                if source_device_idx == to_device_idx {
                    continue;
                }

                let source_read_guard: RwLockReadGuard<'_, Option<O>> = timed_tokio_read(op, "CpuOrGpu::with_selected_data_ref source read").await;

                if let Some(src) = source_read_guard.as_ref() {
                    let sub_t: O = src.subview(selector.clone())?;

                    debug!("CpuOrGpu::with_selected_data_ref: sub_t.size={:?} copying to to_device_idx={to_device_idx}",sub_t.size());

                    let data = sub_t.copy_to_device(to_device_idx).await?;

                    let stream = crate::copy_ctx::CopyCtx::instance(1).acquire_stream().await;
                    let _guard = thoosa_shims::stream::ThoStreamGuard::new(&stream)?;
                    let r = DevicesWrap::<O>::do_run_trace_owned(trace_ctx, |data| do_run(data, &permits_ctx), data);

                    stream.synchronize()?;

                    return r;
                }
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
        // if permits_ctx.acquired_permits() == 0 {
        //     let permit = GlobalCtx::global().acquire_one(to_device_idx, Arc::new(permits_ctx.id().clone())).await?;
        //     permits_ctx.add_permit(permit);
        // }
        let read_guard: RwLockReadGuard<'_, Option<O>> = timed_tokio_read(&wrap.cached_data[to_device_idx], "CpuOrGpu::with_selected_data target read").await;

        if read_guard.is_some() {
            let data = read_guard
                .as_ref()
                .expect("no value behind guard")
                .subview(selector.clone())?;

            return do_run(data, &permits_ctx);
        }

        drop(read_guard);

        for (source_device_idx, op) in wrap.cached_data.iter().enumerate() {
            if source_device_idx == to_device_idx {
                continue;
            }

            let source_read_guard: RwLockReadGuard<'_, Option<O>> = timed_tokio_read(op, "CpuOrGpu::with_selected_data source read").await;

            if let Some(src) = source_read_guard.as_ref() {
                let sub_t = src.subview(selector.clone())?;
                debug!("DataCacheWrapper::with_selected_data: sub_t.size={:?} copying to to_device_idx={to_device_idx}",sub_t.size());

                let data = sub_t.copy_to_device(to_device_idx).await?;

                return do_run(data, &permits_ctx);
            }
        }

        Err(Cause::NotFound.into())
    }



    fn dump_to_cpu(wrap: &DevicesWrap<O>, device_idx: usize) -> impl std::future::Future<Output = Result<(), ThoError>> + std::marker::Send
    where
        Self: Sized
    {

        async move {
            if wrap.is_all_on_cpu().await {
                // #[cfg(any(test, feature = "mock_global_ctx"))]
                // let _mock_ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

                *timed_parking_rw_write(&wrap.on_cpu_since, "CpuOrGpu::dump_to_cpu on_cpu_since.write") = Some(GlobalCtx::global().cache_age());
                return Ok(());
            }

            assert_ne!(device_idx, 0);


            debug!("DataCacheWrapper::dump_to_cpu: 0.  id={:?} device_idx={device_idx}",wrap.id);

            if let Some(Ok(mut lock)) = wrap.cached_data.get(device_idx).map(|v| v.try_write()) {
                if timed_tokio_read(&wrap.cached_data[0], "CpuOrGpu::dump_to_cpu cpu read check").await.is_none() {
                    let data = lock.take();
                    debug!("DataCacheWrapper::dump_to_cpu: 1.0. id={:?} copied to cpu done device_idx={device_idx}", wrap.id);

                    wrap.ensure_copied(device_idx, 0, data.as_ref(), false).await?;

                    debug!("DataCacheWrapper::dump_to_cpu: 1.1. id={:?} copied to cpu done device_idx={device_idx}", wrap.id);
                } else {
                    drop(lock.take());
                }
            } else {
                // If it's locked, assume it's the current task's input. DumpToCpu only runs with
                // MAX_PERMITS or when cache_area.assigned_count() == 0 (from purge_gpu_closure added in run_cached), so no other tasks should be running; skipping avoids deadlock.
                warn!("DataCacheWrapper::dump_to_cpu: can't drop self.id={:?}, locked, data={:?}",wrap.id, wrap.cached_data) //if it was dumped to cpu from DumpAllToCpu and then purged
            }

            let is_all_on_cpu = wrap.is_all_on_cpu().await;
            let age = GlobalCtx::global().cache_age();
            if is_all_on_cpu {
                //may rewrite previous Some, but shouldn't be a problem

                // #[cfg(any(test, feature = "mock_global_ctx"))]
                // let _mock_ctx = MockGlobalCtxBuilder::install_global_ctx_mock();


                *timed_parking_rw_write(&wrap.on_cpu_since, "CpuOrGpu::dump_to_cpu on_cpu_since.write") = Some(age);
            }

            debug!("DataCacheWrapper::dump_to_cpu: 2. id={:?} device_idx={device_idx} is_all_on_cpu={is_all_on_cpu} age={age}",wrap.id);

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
        let on_cpu_since = timed_parking_rw_read(&wrap.on_cpu_since, "CpuOrGpu::is_on_cpu_before on_cpu_since.read");
        let r = on_cpu_since.as_ref().map(|on_cpu_since| *on_cpu_since <= threshold_age ).unwrap_or(false);

        debug!("DataCacheWrapper::is_on_cpu_before: self.id={:?} self.on_cpu_since={:?} r={r} threshold_age={threshold_age}",wrap.id, wrap.on_cpu_since);

        r
    }

    async fn take_data(wrap: DevicesWrap<O>, to_device_idx: usize) -> Result<O, ThoError>
    where
        Self: Sized
    {
        let _ = wrap.load_from_any_device_(to_device_idx).await?;

        let mut v = wrap.cached_data.into_vec();
        let lock = v.remove(to_device_idx);
        let opt: Option<O> = lock.into_inner();
        Ok(opt.expect(&format!("no data in {to_device_idx} for {:?}", wrap.id)))
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="DevicesWrapCopyMode<O> for CpuOrGpu<O>::is_all_on_cpu"))]
    async fn is_all_on_cpu(wrap: &DevicesWrap<O>) -> bool {
        trace!("DataCacheWrapper::is_all_on_cpu: self.id={:?}", wrap.id);

        for (device_idx, data) in wrap.cached_data.iter().enumerate().skip(1) {
            if timed_tokio_read(data, "CpuOrGpu::is_all_on_cpu read").await.is_some() {
                debug!("DataCacheWrapper::is_all_on_cpu: {:?} is on device_idx={device_idx}", wrap.id);
                return false;
            }
        }

        true
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
    //contains cache age ( GlobalCtx::cache_age - increated when something is being cached from Cachable::run_cached). The smaller the value the longer it was on the cpu
    on_cpu_since: parking_lot::RwLock<Option<u64>>,
    pd: PhantomData<O>,
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
        O::M::with_selected_data(self, to_device_idx, permits_ctx, selector, do_run).await
    }

    pub fn is_on_cpu_before(&self, threshold_age: u64) -> bool {
        O::M::is_on_cpu_before(self, threshold_age)
    }

    pub async fn take_data(
        self,
        to_device_idx: usize,
    ) -> Result<O, ThoError> {
        O::M::take_data(self, to_device_idx).await
    }

    pub async fn load_from_any_device_(
        &self,
        to_device_idx: usize,
    ) -> Result<impl Deref<Target = O> + Debug + Send, ThoError> {
        O::M::load_from_any_device_(self, to_device_idx).await
    }

    pub fn is_all_on_cpu(&self) -> impl Future<Output=bool> + Send {
        O::M::is_all_on_cpu(self)
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
        Ok(timed_tokio_read(&self.cached_data[0], "DevicesWrap::clone_from_cpu read").await.clone().expect("no data in cpu"))
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

    //to avoid write lock on dst device while copy is in progress use inflights
    async fn ensure_copied<'a>(
        &self,
        src_idx: usize,
        dst_idx: usize,
        src: Option<&O>,
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
                let mut g = timed_parking_mutex_lock(self.inflight, "DevicesWrap::ensure_copied inflight lock");
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

        debug!("DevicesWrap::ensure_copied: self.id={:?} src_idx={src_idx} dst_idx={dst_idx} do_return={do_return} src={src:?}", self.id);

        loop {
            let dst_read = timed_tokio_read(&self.cached_data[dst_idx], "DevicesWrap::ensure_copied dst read").await;
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

            // ---- Try to join or become the leader
            // (Do not hold this lock while awaiting other locks, except briefly to set or read the notify)
            let (maybe_notify, leader_notify) = {
                let mut g = timed_parking_mutex_lock(&self.inflights[dst_idx], "DevicesWrap::ensure_copied inflights lock");
                if let Some(n) = &*g {
                    // someone is already copying → wait on their notify
                    (Some(n.clone()), None)
                } else {
                    // become the leader: install a notify and proceed
                    let n = Arc::new(Notify::new());
                    *g = Some(n.clone());
                    (None, Some(n))
                }
            };

            if let Some(n) = maybe_notify {
                // follower: wait until the leader finishes, then try fast path again
                n.notified().await;
                continue;
            }

            // ---- Leader path
            let mut inflight_guard = InflightGuard::new(
                &self.inflights[dst_idx],
                leader_notify.expect("leader notify"),
            );

            // Double-check: maybe dst was filled after we dropped the read lock
            if timed_tokio_read(&self.cached_data[dst_idx], "DevicesWrap::ensure_copied dst read double-check").await.is_some() {
                inflight_guard.clear();
                continue;
            }

            // Read source value without holding dst locks
            let produced = match src {
                None => {
                    debug!("DevicesWrap::ensure_copied: self.id={:?} copy_to_device from cached_data", self.id);
                    let src_read_guard = timed_tokio_read(&self.cached_data[src_idx], "DevicesWrap::ensure_copied src read").await;
                    src_read_guard.as_ref().expect("src missing unexpectedly").copy_to_device(dst_idx).await
                }
                Some(v) => {
                    debug!("DevicesWrap::ensure_copied: self.id={:?} copy_to_device from src", self.id);

                    v.copy_to_device(dst_idx).await
                }
            };

            let produced = match produced {
                Ok(v) => v,
                Err(e) => {
                    return Err(e);
                }
            };


            // Install into dst (short critical section)
            {
                debug!("DevicesWrap::ensure_copied: self.id={:?} install copied to dst_idx={dst_idx} produced.device={:?}", self.id, produced.device());

                let mut w = timed_tokio_write(&self.cached_data[dst_idx], "DevicesWrap::ensure_copied dst write").await;
                if w.is_none() {
                    *w = Some(produced);
                }
            }

            // Clear inflight and wake followers
            inflight_guard.clear();
        }
    }

    async fn copy_and_get(
        &self,
        src_idx: usize,
        dst_idx: usize,
    ) -> Result<RwLockReadGuard<'_, O>, ThoError> {
        self.ensure_copied(src_idx, dst_idx, None, true)
            .await
            .map(|v| v.expect("no value returned"))
    }

    #[cfg(test)]
    pub fn cached_data(&self) -> &Box<[tokio::sync::RwLock<Option<O>>]> {
        &self.cached_data
    }
}

///To let OOMHandler accept both plain wrapped data as it's stored in caches and composite inputs like module + tensor or batch + motif
pub trait WithDataRef: WithId + Debug {
    type O<'o>: Data + /*CopyToDevice +*/ WithId + Send + Sync;

    async fn with_data_ref<R, F, P>(
        &self,
        to_device_idx: usize,
        permits_ctx: P,
        trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> Result<R, ThoError>
    where
       for <'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send,
        P:RunCtx;


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
    async fn with_selected_data_ref<'o, R, F, P>(
        &self,
        to_device_idx: usize,
        _permits_ctx: P,
        selector: S,
        trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> Result<R, ThoError> where
     // Self::O<'o>: Select<S, Output = Self::O<'o>>,
     for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError>,
     P:RunCtx;
}

// OOMHandler submits sync-stage work to DynamicTaskExecutor, and submit requires the
// whole future to be Send. The original WithDataRef stays non-Send because existing
// call sites rely on closures/futures that do not satisfy Send. This companion trait
// gives the executor path a Send-only entry point without tightening the base trait.
pub trait WithDataRefSend: WithDataRef {
    fn with_data_ref_send<R, F, P>(
        &self,
        to_device_idx: usize,
        permits_ctx: P,
        trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> impl Future<Output = Result<R, ThoError>> + Send
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send,
        P: RunCtx;
}

// Same split for selected access: OOMHandler's executor path needs a Send future,
// while the general WithSelectDataRef API must remain available to non-Send callers.
pub trait WithSelectDataRefSend<S>: WithDataRefSend + WithSelectDataRef<S>
where
    S: Clone + Debug + Send + 'static,
{
    fn with_selected_data_ref_send<'o, R, F, P>(
        &self,
        to_device_idx: usize,
        permits_ctx: P,
        selector: S,
        trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> impl Future<Output = Result<R, ThoError>> + Send
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send,
        P: RunCtx;
}

/// Owned-data analogue of WithDataRef for sync-stage consume paths that need executor-controlled materialization.
pub trait ConsumeData {
    type O: Data + WithId + Send + Sync;

    fn consume<R, F, P>(
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

    async fn with_data_ref<R, F, P>(&self, to_device_idx: usize, permits_ctx: P, trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoError>
    where
            for <'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send,
            P:RunCtx
    {
        O::M::with_data_ref(self, to_device_idx, permits_ctx, trace_ctx, do_run).await
    }

    fn mem_size(&self) -> u64 {
        self.mem_size
    }

    fn _size<'f>(&self) -> <Self::O<'f> as Data>::Metrics {
        self.size.clone()
    }
}

impl<O> WithDataRefSend for DevicesWrap<O>
where
    O: Data + WithId + Send + Sync,
    O::M: DevicesWrapCopyMode<O>,
{
    fn with_data_ref_send<R, F, P>(
        &self,
        to_device_idx: usize,
        permits_ctx: P,
        trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> impl Future<Output = Result<R, ThoError>> + Send
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send,
        P: RunCtx,
    {
        async move { O::M::with_data_ref(self, to_device_idx, permits_ctx, trace_ctx, do_run).await }
    }
}

impl<O, S> WithSelectDataRef<S> for DevicesWrap<O>
where
    S:Debug + Clone + 'static,
    O: Select<S, Output=O> + Data + WithId + Send + Sync + CopyToDevice,
    O::M: DevicesWrapCopyMode<O>,
{
    async fn with_selected_data_ref<'o, R, F, P>(&self, to_device_idx: usize, permits_ctx: P, selector: S, trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoError>
    where
        Self::O<'o>: Select<S, Output=Self::O<'o>>,
        for <'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError>,
        P:RunCtx
    {
        O::M::with_selected_data_ref(self, to_device_idx, permits_ctx, selector, trace_ctx, do_run).await
    }

}

impl<O, S> WithSelectDataRefSend<S> for DevicesWrap<O>
where
    S:Debug + Clone + Send + 'static,
    O: Select<S, Output=O> + Data + WithId + Send + Sync + CopyToDevice,
    O::M: DevicesWrapCopyMode<O>,
{
    fn with_selected_data_ref_send<'o, R, F, P>(
        &self,
        to_device_idx: usize,
        permits_ctx: P,
        selector: S,
        trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> impl Future<Output = Result<R, ThoError>> + Send
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send,
        P: RunCtx,
    {
        async move { O::M::with_selected_data_ref_send(self, to_device_idx, permits_ctx, selector, trace_ctx, do_run).await }
    }
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use crate::runner::global_ctx::MockGlobalCtx;
    use thoosa_with_data::with_data;
    extern crate self as caching_runner;
    pub use crate::runner::common::*;
    use crate::runner::cache_areas::CacheArea;
    use crate::runner::global_ctx::mock_global_ctx;
    use crate::runner::global_ctx;
    use crate::runner::common::cachable::{DevicesWrap, WithDataRef};
    use crate::runner::common::mock_run_ctx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::runner::test_utils::MockDataWithId;
    use crate as thoosa_common;


    #[tokio::test]
    async fn test_dump_to_cpu() -> Result<(), ThoError> {

        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

        let data = MockDataWithId::whatever();
        let tested = super::DevicesWrap::wrap(data);
        assert!(super::timed_tokio_read(&tested.cached_data[1], "test_dump_to_cpu read 1").await.is_none());

        let ctx_for_data = mock_run_ctx::default(1);
        tested
            .with_data_ref(1,
               ctx_for_data, None, |data_on_1: &MockDataWithId, _ctx| {
                   assert_eq!(data_on_1.device(), &1);

                   Ok(())
               })
            .await?;

        assert!(super::timed_tokio_read(&tested.cached_data[1], "test_dump_to_cpu read 2").await.is_some());

        tested.dump_to_cpu(1).await?;
        assert!(super::timed_tokio_read(&tested.cached_data[1], "test_dump_to_cpu read 3").await.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_with_data() -> Result<(), ThoError> {
        let ctx = MockGlobalCtx::global_context();
        ctx.expect().returning(|| {
            Box::leak(Box::new(
                mock_global_ctx::build_default(),
            )) as &'static global_ctx::MockGlobalCtx
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
