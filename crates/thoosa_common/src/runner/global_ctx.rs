use std::backtrace::Backtrace;
use crate::runner::purge::FuturesRunnerMapPurger;
use log::{debug, error, info};
#[allow(unused_imports)]
use mockall::automock;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, OnceLock};
use futures_util::future::BoxFuture;
use futures_util::StreamExt;
use thoosa_shared::traits::CachableIdTr;
use thoosa_shared::{backtrace_head_lines, with_async_perf_log, with_perf_log, Cause, ThoError, LOCK_TIMEOUT};
use tokio::sync::RwLock as AsyncRwLock;
use crate::runner::cache_areas::{CacheAreaId, CacheAreasKeeper};
use crate::runner::purger::Purger;
use thoosa_shared::collect_futures_eta::{collect_eta_record_completion, CollectEtaInvocationGuard};
use thoosa_shims::context::ThoCudaContext;
use crate::copy_ctx::CopyCtx;
use crate::runner::adjusters::max_parallel::{MaxParallel, MAX_PARALLEL};
use crate::runner::dyn_mpsc::DynamicTaskExecutor;
use crate::tho_device::ThoDevice;
use futures::FutureExt;

#[derive(Debug)]
pub struct GlobalCtx {

    executors: Vec<DynamicTaskExecutor>,
    num_devices: usize,

    caches: parking_lot::RwLock<HashMap<&'static str, Arc<CacheRegistryEntry>>>,
    is_all_on_cpu: Vec<AtomicBool>,
    caches_areas: parking_lot::RwLock<CacheAreasKeeper>,
    cache_time: AtomicU64, //for tracking CPU-only lifetime
    purger: Purger,
    shutting_down:AtomicBool,//TODO not needed if tokio's shutdown_background in project main,

    copy_ctx:Vec<tokio::sync::RwLock<CopyCtx>>
}

static GLOBAL_CTX: OnceLock<GlobalCtx> = OnceLock::new();

#[derive(Debug)]
struct CacheRegistryEntry {
    purger: Arc<dyn FuturesRunnerMapPurger>,
    gate: Arc<AsyncRwLock<()>>,
}

type CacheSnapshot = Vec<(&'static str, Arc<CacheRegistryEntry>)>;

#[derive(Debug, Clone)]
pub struct RegisteredCacheHandle {
    name: &'static str,
    gate: Arc<AsyncRwLock<()>>,
}

impl RegisteredCacheHandle {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub async fn lock_exclusive(&self) -> tokio::sync::OwnedRwLockWriteGuard<()> {
        with_async_perf_log(
            async || {
                tokio::time::timeout(LOCK_TIMEOUT, self.gate.clone().write_owned())
                    .await
                    .expect("RegisteredCacheHandle::lock_exclusive: timed out waiting for cache gate write lock")
            },
            "RegisteredCacheHandle/lock_exclusive/write_cache_gate",
        )
        .await
    }

    #[cfg(any(test, feature = "mock_global_ctx"))]
    pub fn for_tests(name: &'static str) -> Self {
        Self {
            name,
            gate: Arc::new(AsyncRwLock::new(())),
        }
    }
}

#[cfg_attr(any(test, feature = "mock_global_ctx"), automock)]
impl GlobalCtx {

    fn new(num_devices: usize) -> Self {
        debug!("GlobalCtx::new: num_devices={num_devices}");

        assert!(num_devices > 0);//TODO test if it works without GPU

        Self {
            num_devices,
            executors: (1..num_devices).map(|tho_device_idx| {
                DynamicTaskExecutor::new_with_reset_hook(1024, NonZeroUsize::new(16).unwrap(), tho_device_idx, Arc::new(GlobalCtx::reset_cuda_context))
            })
            .collect(),

            caches: parking_lot::RwLock::new(HashMap::new()),

            is_all_on_cpu: (0..num_devices).map(|_| AtomicBool::new(false)).collect(),

            caches_areas: parking_lot::RwLock::new(CacheAreasKeeper::new(1000)), //100 * 64 cache areas

            cache_time: Default::default(),
            purger: Purger::new(),
            shutting_down: false.into(),
            copy_ctx: (1..num_devices).map(|device_idx|tokio::sync::RwLock::new(CopyCtx::make_instance(0.2, 16))).collect()
        }
    }

    fn reset_cuda_context(tho_device_idx: usize) -> BoxFuture<'static, Result<(), ThoError>> {
        async move {
            info!("GlobalCtx::reset_cuda_context: tho_device_idx={tho_device_idx}");

            GlobalCtx::global().drop_device(tho_device_idx, "drop").await?;

            let cu_ctx = ThoCudaContext::new((tho_device_idx - 1) as i32)?;
            cu_ctx.reset()?;

            *GLOBAL_CTX.get().expect("ups").copy_ctx[tho_device_idx - 1].write().await = CopyCtx::make_instance(0.2, 16);

            info!("GlobalCtx::reset_cuda_context: done");
            Ok(())
        }.boxed()
    }

    pub fn get_executor(&self, tho_device:usize) -> &DynamicTaskExecutor {
        debug!("GlobalCtx::get_executor: tho_device={tho_device} self.executors.len={}", self.executors.len());
        &self.executors[tho_device.saturating_sub(1)]//tho_device is 0 in tests
    }

    pub fn run_purge_gpu(&self, closures: Vec<Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>> + Send + Sync>>) ->Result<(), ThoError>{
        self.purger.run_purge_gpu(closures)
    }

    ///num_devices: number of devices - CPU + GPU count
    pub fn init_devices(num_devices: usize) -> &'static GlobalCtx {
        debug!("GlobalCtx::init_devices: num_devices={num_devices}");

        GLOBAL_CTX
            .set(GlobalCtx::new(num_devices))
            .expect("`init_devices` must only be called once at program start");

        Self::global()
    }

    pub fn register_cache(&self, cache: (&'static str, Arc<dyn FuturesRunnerMapPurger>)) -> RegisteredCacheHandle {
        debug!("GlobalCtx::register_cache: cache={}", cache.0);

        let gate = Arc::new(AsyncRwLock::new(()));
        let entry = Arc::new(CacheRegistryEntry {
            purger: cache.1,
            gate: Arc::clone(&gate),
        });

        assert!(
            with_perf_log(
                || self.caches.try_write_for(LOCK_TIMEOUT).expect("GlobalCtx::register_cache: timed out waiting for caches write lock"),
                "GlobalCtx/register_cache/write_caches",
            )
            .insert(cache.0, entry)
            .is_none()
        );

        debug!("GlobalCtx::register_cache: cache={} lock acquired/released, done", cache.0);

        RegisteredCacheHandle { name: cache.0, gate }
    }

    fn caches_snapshot(&self, ctx: &'static str) -> CacheSnapshot {
        with_perf_log(
            || {
                self.caches
                    .try_read_for(LOCK_TIMEOUT)
                    .unwrap_or_else(|| panic!("{ctx}: timed out waiting for caches read lock"))
                    .iter()
                    .map(|(type_name, entry)| (*type_name, Arc::clone(entry)))
                    .collect()
            },
            ctx,
        )
    }

    async fn lock_cache_shared(entry: &Arc<CacheRegistryEntry>, ctx: &'static str) -> tokio::sync::OwnedRwLockReadGuard<()> {
        with_async_perf_log(
            async || {
                tokio::time::timeout(LOCK_TIMEOUT, entry.gate.clone().read_owned())
                    .await
                    .unwrap_or_else(|_| panic!("{ctx}: timed out waiting for cache gate read lock"))
            },
            ctx,
        )
        .await
    }

    pub fn remove_cache(&self, key:&'static str) {
        debug!("GlobalCtx::remove_cache: key={}", key);

        assert!(
            with_perf_log(
                || self.caches.try_write_for(LOCK_TIMEOUT).expect(&format!("GlobalCtx::remove_cache, key={key}: timed out waiting for caches write lock")),
                "GlobalCtx/remove_cache/write_caches",
            )
            .remove(key)
            .is_some()
        );

        debug!("GlobalCtx::remove_cache: {} lock acquired/released, removed ", key);
    }

    pub fn global() -> &'static Self {
        GLOBAL_CTX.get().expect("call `init_devices` first")
    }


    #[cfg(any(test, feature = "mock_global_ctx"))]
    pub fn real_global() -> &'static GlobalCtx {
        GLOBAL_CTX.get().expect("call `init_devices` first")
    }

    pub fn device_count(&self) -> usize {
        self.num_devices
    }

    pub fn is_all_on_cpu(&self, device_idx: usize) -> bool {
        self.is_all_on_cpu[device_idx].load(Relaxed)
    }

    pub async fn reset_all_on_cpu(&self, device_idx: usize) {
        self.is_all_on_cpu[device_idx].store(false, Relaxed);
    }


    #[cfg(tho_debug)]
    async fn debug_caches_snapshot(&self, device_idx: usize, descr: &str, maps: &CacheSnapshot) {
        debug!("GlobalCtx::debug_caches: descr={descr:?}");

        let mut total_gpu_mem = 0;
        for (type_name, entry) in maps.iter() {
            let _cache_guard = Self::lock_cache_shared(entry, "GlobalCtx::debug_caches: timed out waiting for cache gate read lock").await;
            info!("GlobalCtx::debug_gpu: descr={descr} type_name={type_name}");
            match entry.purger.debug_gpu(device_idx).await {
                Ok(v) => total_gpu_mem += v,
                Err(e) => error!("GlobalCtx::debug_gpu: failed with {e:?}"),
            }
        }

        info!(
        "GlobalCtx::debug_gpu: descr={descr} device_idx={device_idx} total_gpu_mem={total_gpu_mem} device_stats={:?}",
        thoosa_shims::device_stats::get_device_stats((device_idx - 1) as i32)
    );
    }

    #[cfg(tho_debug)]
    pub async fn debug_caches(&self, device_idx: usize, descr: &str) {
        let maps = self.caches_snapshot("GlobalCtx::debug_caches: timed out waiting for caches read lock");
        self.debug_caches_snapshot(device_idx, descr, &maps).await;
    }

    ///ids_to_keep_by_type - should be few keys long - only inputs involved in the current operation
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="GlobalCtx::dump_all_to_cpu"))]
    pub async fn dump_all_to_cpu<'a>(
        &'a self,
        required_ids: &HashMap<&'static str, &'a HashSet<&'a dyn CachableIdTr>>, //TODO key is used for tmp_cache
        tho_device_idx: usize,
        descr: &str,
    ) -> Result<(), ThoError> {
        assert_ne!(tho_device_idx, 0);
        debug!("GlobalCtx::dump_all_to_cpu: tho_device_idx={tho_device_idx} required_ids={required_ids:?} descr={descr}");
        let maps = self.caches_snapshot("GlobalCtx::dump_all_to_cpu: timed out waiting for caches read lock");
        let dump_eta_guard = CollectEtaInvocationGuard::new(&format!("dump_all_to_cpu_{descr}"), maps.len(), 1);

        for (type_name, entry) in maps.iter() {
            let _cache_guard = Self::lock_cache_shared(entry, "GlobalCtx::dump_all_to_cpu: timed out waiting for cache gate read lock").await;
            debug!("GlobalCtx::dump_all_to_cpu: type_name={type_name}");
            entry.purger.purge_gpu(required_ids.get(type_name).map(|v| *v), tho_device_idx, type_name).await?;
            debug!("GlobalCtx::dump_all_to_cpu: type_name={type_name} descr={descr} done");
            collect_eta_record_completion(dump_eta_guard.id());
        }

        self.is_all_on_cpu[tho_device_idx].store(true, Relaxed);
        debug!("GlobalCtx::dump_all_to_cpu: tho_device_idx={tho_device_idx} required_ids={required_ids:?} descr={descr} done");

        Ok(())
    }

    ///ids_to_keep_by_type - should be few keys long - only inputs involved in the current operation
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="GlobalCtx::dump_all_to_cpu"))]
    pub async fn drop_device<'a>(
        &'a self,
        tho_device_idx: usize,
        descr: &str,
    ) -> Result<(), ThoError> {
        assert_ne!(tho_device_idx, 0);
        debug!("GlobalCtx::drop_device: tho_device_idx={tho_device_idx} descr={descr}");
        let maps = self.caches_snapshot("GlobalCtx::drop_device: timed out waiting for caches read lock");
        let dump_eta_guard = CollectEtaInvocationGuard::new(&format!("drop_device{descr}"), maps.len(), 1);

        for (type_name, entry) in maps.iter() {
            let _cache_guard = Self::lock_cache_shared(entry, "GlobalCtx::drop_device: timed out waiting for cache gate read lock").await;
            debug!("GlobalCtx::drop_device: type_name={type_name}");
            entry.purger.drop_gpu(tho_device_idx, type_name).await?;
            debug!("GlobalCtx::drop_device: type_name={type_name} descr={descr} done");
            collect_eta_record_completion(dump_eta_guard.id());
        }

        self.is_all_on_cpu[tho_device_idx].store(true, Relaxed);
        debug!("GlobalCtx::drop_device: tho_device_idx={tho_device_idx} descr={descr} done");

        Ok(())
    }


    pub async fn purge_cpu(&self, retain_last: u64) {//retain_last - caches younger than current cache time - retain_last should not be purged
        debug!("GlobalCtx::purge_cpu: retain_last={retain_last}");


        let purge_until = self.cache_time.load(Relaxed).saturating_sub(retain_last);


        if purge_until > 0 {
            let maps = self.caches_snapshot("GlobalCtx::purge_cpu: timed out waiting for caches read lock");

            #[cfg(tho_debug)]
            self.debug_caches_snapshot(1, "before purge_cpu", &maps).await;

            for (type_name, entry) in maps.iter() {
                let _cache_guard = Self::lock_cache_shared(entry, "GlobalCtx::purge_cpu: timed out waiting for cache gate read lock").await;
                let purged = entry.purger.purge_cpu(purge_until).await;
                debug!("GlobalCtx::purge_cpu: purged {purged} from {type_name}");
            }

            #[cfg(tho_debug)]
            self.debug_caches_snapshot(1, "after purge_cpu", &maps).await;
        }

        debug!("GlobalCtx::purge_cpu: purge_until={purge_until} done");
    }

    pub fn next_cache_area(&self) -> Result<CacheAreaId, ThoError> {//TODO remove
        with_perf_log(
            || self.caches_areas.try_write_for(LOCK_TIMEOUT).expect("GlobalCtx::next_cache_area: timed out waiting for cache_areas write lock"),
            "GlobalCtx/next_cache_area/write_cache_areas",
        )
        .next_available_cache_area()
    }

    pub fn dispose_cache_area(&self, cache_area:CacheAreaId) {
        with_perf_log(
            || self.caches_areas.try_write_for(LOCK_TIMEOUT).expect("GlobalCtx::dispose_cache_area: timed out waiting for cache_areas write lock"),
            "GlobalCtx/dispose_cache_area/write_cache_areas",
        )
        .dispose_cache_area(cache_area)
    }

    pub fn do_age(&self) {
        self.cache_time.fetch_add(1, Relaxed);
    }

    pub fn cache_age(&self) -> u64 {
        self.cache_time.load(Relaxed)
    }


    pub async fn shutdown(&self) {
        self.shutting_down.store(true, Relaxed);
        //TODO


    }

    pub fn is_shutting_down(&self)->bool {
        self.shutting_down.load(Relaxed)
    }
}

impl GlobalCtx {
    pub async fn copy_ctx<D:Into<ThoDevice>+ 'static>(&self, device:D)-> tokio::sync::RwLockReadGuard<'_, CopyCtx> {
        let device:ThoDevice = device.into();
        let gpu_device_ordinal = device.gpu_ordinal();
        self.copy_ctx.get(gpu_device_ordinal).expect(&format!("No copy ctx for device={gpu_device_ordinal}")).read().await
    }
}

#[cfg(any(test, feature = "mock_global_ctx"))]
pub mod mock_global_ctx {
    use std::sync::Arc;
    use crate::runner::global_ctx::{GlobalCtx, RegisteredCacheHandle};
    use thoosa_shared::Cause;
    use thoosa_shared::ThoError;
    use async_trait::async_trait;
    use log::debug;
    use thoosa_shared::traits::CachableIdTr;
    use mockall::predicate::eq;
    use tokio::sync::Semaphore;
    use tokio::task::block_in_place;
    use thoosa_shared::comp_id::SynKey;
    use crate::runner::adjusters::max_parallel::MaxParallel;
    use crate::runner::cache_areas::CacheAreaId;
    use crate::runner::global_ctx::MockGlobalCtx;

    #[derive(Clone, Copy, Debug)]
    pub enum AcquireBehavior {
        Grant,
        Deny,
        Boom(&'static str)
    }

    #[derive(Debug)]
    pub struct MockGlobalCtxBuildParams {
        pub age: u64,
        pub device_count: usize,
        pub next_cache_areas: Vec<u64>,
        pub acquire_many: AcquireBehavior,
        pub acquire_one: AcquireBehavior,
    }

    impl Default for MockGlobalCtxBuildParams {
        fn default() -> Self {
            Self {
                age: 0,
                device_count: 2,
                next_cache_areas: vec![1],
                acquire_many: AcquireBehavior::Grant,
                acquire_one: AcquireBehavior::Grant,
            }
        }
    }



    #[cfg(any(test, feature = "mock_global_ctx"))]
    pub struct MockGlobalCtxBuilder {
        mock: MockGlobalCtx,
        real: Option<&'static GlobalCtx>
    }

    #[cfg(any(test, feature = "mock_global_ctx"))]
    impl MockGlobalCtxBuilder {
        pub fn with_reset_all_on_cpu(mut self) -> Self {
            if let Some(real) = self.real {

                let _ = self.mock.expect_reset_all_on_cpu().returning({

                    move |device_idx| {
                        tokio::runtime::Handle::current().block_on(real.reset_all_on_cpu(device_idx));
                    }
                });
            } else {
                let _ = self.mock.expect_reset_all_on_cpu();
            }

            self
        }

        pub fn with_age(mut self, out: u64) -> Self {


            if let Some(real) = self.real {

                self.mock.expect_do_age().returning({

                    move || {
                        real.do_age();
                    }
                });

                self.mock.expect_cache_age().returning({

                    move || {
                        real.cache_age()
                    }
                });

            } else {
                self.mock.expect_do_age();
                self.mock.expect_cache_age().return_const(out);
            }


            self
        }

        pub fn with_device_count(mut self, out: usize) -> Self {
            if let Some(real) = self.real {
                self.mock.expect_device_count().returning(move || {
                    real.device_count()
                });
            } else {
                self.mock.expect_device_count().return_const(out);
            }

            self
        }
        pub fn with_register_cache(mut self) -> Self {
            if let Some(real) = self.real {
                self.mock.expect_register_cache().returning({
                    move |cache| real.register_cache(cache)
                });
            } else {
                self.mock.expect_register_cache().returning(|cache| RegisteredCacheHandle::for_tests(cache.0));
            }

            self
        }

        pub fn with_remove_cache(mut self) -> Self {
            if let Some(real) = self.real {
                self.mock.expect_remove_cache().returning({
                    move |key| real.remove_cache(key)
                });
            } else {
                self.mock.expect_remove_cache();
            }

            self
        }

        pub fn with_dump_all_to_cpu(mut self) -> Self {
            if let Some(real) = self.real {
                let _ = self.mock.expect_dump_all_to_cpu().returning({
                    move |required_ids, device_idx, descr| {
                        block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(real.dump_all_to_cpu(required_ids, device_idx, descr))
                        })
                    }
                });
            } else {
                let _ = self.mock.expect_dump_all_to_cpu();
            }

            self
        }

        pub fn with_run_purge_gpu(mut self) -> Self {
            if let Some(real) = self.real {
                self.mock.expect_run_purge_gpu().times(0..).returning({
                    move |closures| {
                        real.run_purge_gpu(closures)
                    }
                });
            } else {
                self.mock.expect_run_purge_gpu().times(0..).returning(|_closures| Ok(()));
            }

            self
        }


        pub fn with_dispose_cache_area(mut self) -> Self {
            if let Some(real) = self.real {
                self.mock.expect_dispose_cache_area().times(0..).returning({
                    move | cache_area| {
                        real.dispose_cache_area(cache_area)
                    }
                });
            } else {
                self.mock.expect_dispose_cache_area().times(0..);
            }
            self
        }

        pub fn with_next_cache_area(mut self, mut cache_areas:Vec<u64>) -> Self {
            if let Some(real) = self.real {
                let _ = self.mock.expect_next_cache_area().returning({
                    move || {
                        real.next_cache_area()
                    }
                });
            } else {
                let _ = self.mock.expect_next_cache_area().returning(move ||{
                    Ok(cache_areas.pop().unwrap())
                });
            }

            self
        }

        pub fn finish(self) -> MockGlobalCtx {
            self.mock
        }

        pub fn build(self) -> MockGlobalCtx {
            build_with_params_inner(self, MockGlobalCtxBuildParams::default())
        }

        #[cfg(any(test, feature = "mock_global_ctx"))]
        pub fn install_global_ctx_mock(use_real_ctx: bool) -> impl Drop {
            debug!("install_global_ctx_mock");
            use crate::runner::global_ctx::{mock_global_ctx, MockGlobalCtx};
            let ctx = MockGlobalCtx::global_context();
            if use_real_ctx {
                use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;

                let real: &'static RealGlobalCtx = RealGlobalCtx::real_global();


                ctx.expect().returning(move || {
                    // NOTE: this leaks once per call to GlobalCtx::global() in the test.
                    Box::leak(Box::new(from_real(real).build()))
                });

                return ctx;
            }

            ctx.expect().returning(|| {
                Box::leak(Box::new(
                    mock_global_ctx::build_default(),
                )) as &'static MockGlobalCtx
            });

            ctx
        }

    }

    fn build_with_params_inner(builder: MockGlobalCtxBuilder, params: MockGlobalCtxBuildParams) -> MockGlobalCtx {
        builder
            .with_age(params.age)
            .with_dump_all_to_cpu()
            .with_reset_all_on_cpu()
            .with_device_count(params.device_count)
            .with_register_cache()
            .with_remove_cache()
            .with_next_cache_area(params.next_cache_areas)
            .with_run_purge_gpu()
            .with_dispose_cache_area()
            .finish()
    }

    pub fn build_with_params(params: MockGlobalCtxBuildParams) -> MockGlobalCtx {
        build_with_params_inner(build(), params)
    }

    pub fn build() -> MockGlobalCtxBuilder {
        MockGlobalCtxBuilder {
            mock: MockGlobalCtx::default(),
            real: None
        }
    }

    pub fn from_real(global_ctx: &'static GlobalCtx) -> MockGlobalCtxBuilder {

        MockGlobalCtxBuilder {
            mock: MockGlobalCtx::default(),
            real: Some(global_ctx)
        }
    }


    pub fn build_default() -> MockGlobalCtx {
        build_with_params(MockGlobalCtxBuildParams::default())
    }

    pub static mut USE_REAL_CTX:bool = false;
}

#[cfg(any(test, feature = "mock_global_ctx"))]
impl MockGlobalCtx {
    pub async fn copy_ctx<D:Into<ThoDevice>+ 'static>(&self, device:D)-> tokio::sync::RwLockReadGuard<'static, CopyCtx> {
        static COPY_CTX_ONCE: std::sync::OnceLock<tokio::sync::RwLock<CopyCtx>> = std::sync::OnceLock::new();
        COPY_CTX_ONCE.get_or_init(|| tokio::sync::RwLock::new(CopyCtx::make_instance(0.2, 16))).read().await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use log::debug;
    use tch::kind::FLOAT_CUDA;
    use tch::Tensor;
    use thoosa_shared::ThoError;
    use thoosa_shims::context::ThoCudaContext;
    use thoosa_with_data::with_data;
    use crate::runner::common::cachable::DevicesWrap;
    use crate::runner::common::RunCtx;
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::run_ctx_2::RunCtxImpl;
    use crate::runner::global_ctx::CopyCtx;
    use crate as thoosa_common;
    use crate::dims::AnyDims;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::runner::tmp_cache::TmpCached;
    use crate::tho_tensor::ThoTensor;

    // TODO
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_device_reset() -> Result<(), ThoError>{
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        // let cu_ctx = ThoCudaContext::new(0 as i32)?;
        // cu_ctx.reset()?;

        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let tmp_cache:TmpCached<ThoTensor<AnyDims, u32>> = TmpCached::new("tmp").await;

        // let t_0 = Arc::new(DevicesWrap::wrap(ThoTensor::new((), Tensor::rand(&[100000], FLOAT_CUDA))?));

        tmp_cache.insert(ThoTensor::new(777u32, Tensor::rand(&[100000], FLOAT_CUDA))?);
        let t_0 = tmp_cache.get(&777u32).unwrap();
        let mut root_ctx = RunCtxImpl::new(1, "root_ctx");

        let sum_0 = with_data!(root_ctx.branch("aaa".into(), false), t_0; |ctx, t_0:&ThoTensor<AnyDims, u32>|{
            Ok::<_, ThoError>(t_0.as_ref().sum_dim_intlist(0, false, None).double_value(&[]))
        })?;
        debug!("sum_0={sum_0}");

        GlobalCtx::global().dump_all_to_cpu(&HashMap::new(), 1, "test").await?;
        // GlobalCtx::reset_cuda_context(1).await?;
        // let cu_ctx = ThoCudaContext::new(0 as i32)?;
        // cu_ctx.reset()?;

        let t_0 = tmp_cache.get(&777u32).unwrap();
        let sum_1 = with_data!(root_ctx.branch("bbb".into(), false), t_0; |ctx, t_0:&ThoTensor<AnyDims, u32>|{
            Ok::<_, ThoError>(t_0.as_ref().sum_dim_intlist(0, false, None).double_value(&[]))
        })?;

        debug!("sum_1={sum_1}");

        assert_eq!(sum_0, sum_1);

        Ok(())
    }

}
