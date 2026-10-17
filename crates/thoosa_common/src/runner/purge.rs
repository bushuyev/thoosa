use crate::runner::common::cachable::{CachedPurgeOutput, DevicesWrap};
use async_trait::async_trait;
use dashmap::DashMap;
use futures_util::{StreamExt, TryStreamExt};
use futures_util::future::{BoxFuture, Shared};
use futures_util::stream;
use log::{debug, info, trace, warn};
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use dashmap::mapref::multiple::RefMulti;
use mockall::Any;
use thoosa_shared::{tracked_await, CopyToDevice};
use thoosa_shared::ThoError;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::traits::{CachableIdTr, Data};
use thoosa_shared::collect_futures_eta::{collect_eta_record_completion, CollectEtaInvocationGuard};
use thoosa_shared::tracked_await::track_await;

pub const PURGE_GPU_CONCURRENCY_LIMIT: usize = 10;

#[derive(Debug)]
pub struct WeakMapPurger<M> {
    map: std::sync::Weak<M>,
}

impl<M> WeakMapPurger<M> {
    pub fn new(map: std::sync::Weak<M>) -> Self {
        Self { map }
    }
}

#[async_trait]
impl<M> FuturesRunnerMapPurger for WeakMapPurger<M>
where
    M: FuturesRunnerMapPurger + Debug + Sync + Send + 'static,
{
    async fn has_gpu_data(&self, device_idx: usize) -> bool {
        match self.map.upgrade() {
            Some(map) => track_await!(map.has_gpu_data(device_idx)).await,
            None => false,
        }
    }

    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self.map.upgrade() {
            Some(map) => track_await!(map.purge_gpu(ids_to_keep, device_idx, name)).await,
            None => Ok(()),
        }
    }

    async fn drop_gpu(&self, device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self.map.upgrade() {
            Some(map) => track_await!(map.drop_gpu(device_idx, name)).await,
            None => Ok(()),
        }
    }

    // #[cfg(any(debug_assertions, tho_debug))]
    async fn debug_gpu(&self, device_idx: usize) -> Result<u64, ThoError> {
        match self.map.upgrade() {
            Some(map) => track_await!(map.debug_gpu(device_idx)).await,
            None => Ok(0),
        }
    }

    async fn purge_cpu(&self, purge_until: u64) -> usize {
        match self.map.upgrade() {
            Some(map) => track_await!(map.purge_cpu(purge_until)).await,
            None => 0,
        }
    }
}

#[async_trait]
pub trait FuturesRunnerMapPurger: Debug  + Sync + Send {//TODO rename to GPUMemPurger
    async fn has_gpu_data(&self, device_idx: usize) -> bool {
        false
    }

    async fn purge_gpu(
        &self,
        ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>,
        device_idx: usize,
        name: &str,
    ) -> Result<(), ThoError> {
        Ok(())
    }

    // Recovery removes only the failed device's data; healthy replicas survive.
    async fn drop_gpu(
        &self,
        device_idx: usize,
        name: &str,
    ) -> Result<(), ThoError> {
        Ok(())
    }

    
    async fn debug_gpu(
        &self,
        _device_idx: usize,
    ) -> Result<u64, ThoError> {
        Ok(0)
    }

    async fn purge_cpu(&self, purge_until: u64) -> usize {
        0
    }
}

pub fn check_ids_to_keep<K, V>(ids_to_keep:Option<&HashSet<&dyn CachableIdTr>>, entry:&RefMulti<K, V>)->bool
where
    K:Eq + Hash + CachableIdTr
{
    ids_to_keep
        .map(|set| {
            if set.is_empty() {
                return true;
            }
            let kd: &dyn CachableIdTr = entry.key();
            let keep_this_key = set.contains(kd);
            if keep_this_key {
                info!("check_ids_to_keep: kd={kd:?}")//TODO remove info
            }
            !keep_this_key
        })
        .unwrap_or(true)
}

#[async_trait]
impl<K, CO> FuturesRunnerMapPurger
    for DashMap<K, thoosa_common::runner::common::cachable::CachedFuture<CO>>
where
    K: CachableIdTr + Clone + Send + Sync + 'static + Eq + Hash,
    CO: CachedPurgeOutput + Debug,
{
    async fn has_gpu_data(&self, device_idx: usize) -> bool {
        self.iter().any(|entry| entry.value().peek().and_then(|value| value.as_ref().ok()).is_some_and(|output| output.is_on_device(device_idx)))
    }

    async fn purge_cpu(&self, purge_until: u64) -> usize {
        let mut purged_count = 0usize;
        // Removing a cache entry only releases cache ownership. External handles
        // and residency guards retain their allocations without Arc-count guesses.
        self.retain(|k, v| {
            debug!("FuturesRunnerMapPurger::purge_cpu: key={k:?}");
            if !v.has_cache_users() && v.peek()
                .and_then(|v| v.as_ref().map(|r| {
                    r.is_on_cpu_before(purge_until)
                }).ok())
                .unwrap_or(false)
            {
                purged_count+=1;
                debug!("FuturesRunnerMapPurger::purge_cpu: key={k:?} removed");
                false
            } else {
                true
            }
        });

        purged_count
    }

    async fn purge_gpu(
        &self,
        ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>,
        device_idx: usize,
        name: &str,
    ) -> Result<(), ThoError> {
        debug!("FuturesRunnerMapPurger::purge_gpu: ids_to_keep={ids_to_keep:?}");

        let items: Vec<(
            K,
            thoosa_common::runner::common::cachable::CachedFuture<CO>,
        )> = self.iter()
            .filter(|entry| {
                check_ids_to_keep(ids_to_keep, entry)
                && entry.value().peek().is_some()
                && entry.value().peek().as_ref().map(|v|v.is_ok()).unwrap_or(false)
            })
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();

        let dump_eta_guard = CollectEtaInvocationGuard::new(&format!("purge_gpu_{:?}", name), self.len(), 1);
        let dump_eta_guard_id = dump_eta_guard.id();
        track_await!(stream::iter(items)

            .for_each_concurrent(PURGE_GPU_CONCURRENCY_LIMIT, move |(key, fut)| async move {
                match track_await!(fut).await {
                    Ok(v) => {
                        if let Err(e) = track_await!(v.dump_to_cpu(device_idx)).await {
                            warn!("purge_gpu failed: key={key:?} err={e:?}");
                        } else {
                            debug!("purge_gpu ok: key={key:?}");
                        }
                    }
                    Err(e) => {
                        if e.is_oom() {
                            debug!("purge_gpu: Err key={key:?} err={e:?}")
                        } else {
                            warn!("purge_gpu: Err key={key:?} err={e:?}")
                        }
                    },
                }
                collect_eta_record_completion(dump_eta_guard_id);
            })).await;

        Ok(())
    }

    // #[cfg(any(debug_assertions, tho_debug))]
    async fn debug_gpu(&self, device_idx: usize) -> Result<u64, ThoError> {
        let items: Vec<(K, CO)> = self
            .iter()
            .filter_map(|entry| {
                entry
                    .value()
                    .peek()
                    .and_then(|value| value.as_ref().ok())
                    .cloned()
                    .map(|output| (entry.key().clone(), output))
            })
            .collect();

        let mut gpu_size = 0;
        for (key, output) in items {
            if output.is_on_device(device_idx) {
                let size = output.mem_size();
                trace!("key={key:?} size={size:?}");
                gpu_size += size;
            }
        }

        info!("debug_gpu: device_idx={device_idx} type_name={:?} self.len={} on_gpu: {gpu_size} bytes", self.type_name(), self.len());

        Ok(gpu_size)
    }

    async fn drop_gpu(&self, device_idx: usize, name: &str) -> Result<(), ThoError> {
        info!("FuturesRunnerMapPurger::drop_gpu: running name={name:?}");

        //because of "implementation of `dashmap::Map` is not general enough"
        let items: Vec<(
            K,
            thoosa_common::runner::common::cachable::CachedFuture<CO>,
        )> = self.iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();

        let dump_eta_guard = CollectEtaInvocationGuard::new(&format!("drop_gpu_{:?}", name), self.len(), 1);
        let dump_eta_guard_id = dump_eta_guard.id();


        track_await!(stream::iter(items.into_iter().map(Ok::<_, ThoError>)).try_for_each_concurrent(PURGE_GPU_CONCURRENCY_LIMIT, |(key, fut)| async move {
            if fut.invalidate_pending_on(device_idx) {
                self.remove_if(&key, |_, current| current.ptr_eq(&fut));
            } else if let Some(output) = fut.peek().and_then(|result| result.as_ref().ok()) {
                if output.is_on_device(device_idx) {
                    output.drop_device(device_idx).await?;
                    if !output.has_any_data() {
                        // Other devices may replace this entry while cleanup awaits.
                        // Remove only the empty future from this snapshot.
                        self.remove_if(&key, |_, current| current.ptr_eq(&fut));
                    }
                }
            }
            // Unresolved futures can belong to healthy devices. Never await or
            // evict them merely because this device is restarting.
            collect_eta_record_completion(dump_eta_guard_id);
            Ok(())
        })).await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use dashmap::DashMap;
    use futures_util::future::{BoxFuture, FutureExt, ready};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use thoosa_shared::applicables_runner::Output;
    use thoosa_shared::traits::{CachableIdTr, DummyWithId as DummyCachableId, WithId};
    use thoosa_shared::copy_mode::CpuOrGpu;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;

    #[tokio::test]
    async fn test_recovery_invalidates_only_affected_pending_cache_work_without_waiters() {
        use crate::runner::common::cachable::CachedFuture;
        use crate::tho_tensor::ThoTensor;
        use thoosa_shared::{CudaActivityGate, CudaActivityScope};
        type Wrapped = Arc<DevicesWrap<ThoTensor<crate::dims::AnyDims, DummyCachableId>>>;
        let first = Arc::new(CudaActivityGate::new(1));
        let second = Arc::new(CudaActivityGate::new(2));
        let map: DashMap<DummyCachableId, CachedFuture<Wrapped>> = DashMap::new();
        let (started, started_rx) = tokio::sync::oneshot::channel();
        let failed_gate = Arc::clone(&first);
        let failed = CachedFuture::for_device(async move {
            let _activity = CudaActivityScope::for_operation(&failed_gate)?;
            started.send(()).unwrap();
            std::future::pending::<Result<Wrapped, ThoError>>().await
        }.boxed().shared(), Some(Arc::clone(&first)));
        map.insert(DummyCachableId("failed"), failed.clone());
        let healthy = CachedFuture::for_device(std::future::pending::<Result<Wrapped, ThoError>>().boxed().shared(), Some(Arc::clone(&second)));
        map.insert(DummyCachableId("healthy"), healthy.clone());
        tokio::time::timeout(Duration::from_secs(2), started_rx).await.unwrap().unwrap();
        // No caller polls the failed computation. The cache must still release
        // its suspended activity before recovery reaches the cleanup phase.
        let ticket = first.request_pause(ThoError::ContextReset).unwrap();
        tokio::time::timeout(Duration::from_secs(2), first.wait_until_idle()).await.expect("an abandoned cache future must not pin recovery");
        map.drop_gpu(1, "device recovery").await.unwrap();
        assert!(!map.contains_key(&DummyCachableId("failed")));
        assert!(map.get(&DummyCachableId("healthy")).unwrap().ptr_eq(&healthy));
        assert!(matches!(failed.await.unwrap_err().what(), ThoError::ContextReset));
        first.finish_pause(ticket);
        // Cleanly unwind the independent background computation too.
        let ticket = second.request_pause(ThoError::ContextReset).unwrap();
        tokio::time::timeout(Duration::from_secs(2), second.wait_until_idle()).await.unwrap();
        second.finish_pause(ticket);
    }

    #[tokio::test]
    async fn test_cache_result_cannot_publish_across_recovery_boundary() {
        use crate::runner::common::cachable::CachedFuture;
        let gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
        let failing_gate = Arc::clone(&gate);
        let future = CachedFuture::for_device(async move {
            failing_gate.request_pause(ThoError::ContextReset).unwrap();
            Ok(42)
        }.boxed().shared(), Some(Arc::clone(&gate)));
        assert!(matches!(future.await.unwrap_err().what(), ThoError::ContextReset));
        tokio::time::timeout(Duration::from_secs(2), gate.wait_until_idle()).await.unwrap();
    }

    #[tokio::test]
    async fn test_recovery_cache_cleanup_preserves_healthy_devices_and_pending_entries() {
        use crate::runner::global_ctx::{MockGlobalCtx, mock_global_ctx};
        use crate::tho_tensor::ThoTensor;
        use crate::ipc::worker_msg::CudaWorkerId;
        use thoosa_shared::CudaDeviceGeneration;
        let global = MockGlobalCtx::global_context();
        let installed = MockGlobalCtx::is_installed_context();
        installed.expect().return_const(false);
        let mock = Box::leak(Box::new(mock_global_ctx::build().with_device_count(3).finish()));
        global.expect().return_const(&*mock);
        type Wrapped = Arc<DevicesWrap<ThoTensor<crate::dims::AnyDims, DummyCachableId>>>;
        let tensor = |device, id| Arc::new(DevicesWrap::wrap(ThoTensor::new_ref(DummyCachableId(id), 42, vec![1], tch::Kind::Float, None, CudaWorkerId { device_idx: device, generation: CudaDeviceGeneration(0) })));
        let failed: Wrapped = tensor(1, "failed");
        let healthy: Wrapped = tensor(2, "healthy");
        let cpu: Wrapped = Arc::new(DevicesWrap::wrap(ThoTensor::new(DummyCachableId("cpu"), tch::Tensor::f_from_slice(&[1f32]).unwrap()).unwrap()));
        let map: DashMap<DummyCachableId, thoosa_common::runner::common::cachable::CachedFuture<Wrapped>> = DashMap::new();
        for (key, output) in [("failed", &failed), ("healthy", &healthy), ("cpu", &cpu)] {
            let future = ready(Ok(Arc::clone(output))).boxed().shared();
            future.clone().await.unwrap();
            map.insert(DummyCachableId(key), future.into());
        }
        let pending = futures_util::future::pending::<Result<Wrapped, ThoError>>().boxed().shared();
        map.insert(DummyCachableId("pending"), pending.clone().into());
        let gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
        let completed = crate::runner::common::cachable::CachedFuture::for_device(ready(Ok(Arc::clone(&cpu))).boxed().shared(), Some(Arc::clone(&gate)));
        completed.clone().await.unwrap();
        map.insert(DummyCachableId("completed_on_failed_device"), completed.clone());
        let ticket = gate.request_pause(ThoError::ContextReset).unwrap();
        gate.wait_until_idle().await;
        tokio::time::timeout(Duration::from_secs(2), map.drop_gpu(1, "device recovery")).await.unwrap().unwrap();
        assert!(!failed.is_on_device(1));
        assert!(!map.contains_key(&DummyCachableId("failed")));
        assert!(healthy.is_on_device(2) && map.contains_key(&DummyCachableId("healthy")));
        assert!(cpu.is_on_device(0) && map.contains_key(&DummyCachableId("cpu")));
        assert!(map.get(&DummyCachableId("pending")).unwrap().ptr_eq(&pending.into()));
        assert!(Arc::ptr_eq(&completed.await.unwrap(), &cpu));
        assert!(map.contains_key(&DummyCachableId("completed_on_failed_device")));
        gate.finish_pause(ticket);
    }

    #[derive(Debug, Clone)]
    struct DummyOutput {
        id: DummyCachableId,
        purge_active: Arc<AtomicUsize>,
        max_purge: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl CopyToDevice for DummyOutput {
        async fn copy_to_device(&self, _dst_device: usize) -> Result<Self, ThoError> {
            let cur = self.purge_active.fetch_add(1, Ordering::SeqCst) + 1;
            loop {
                let prev = self.max_purge.load(Ordering::SeqCst);
                if cur > prev {
                    if self
                        .max_purge
                        .compare_exchange(prev, cur, Ordering::SeqCst, Ordering::SeqCst)
                        .is_ok()
                    {
                        break;
                    }
                } else {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
            self.purge_active.fetch_sub(1, Ordering::SeqCst);
            Ok(self.clone())
        }
    }

    impl Data for DummyOutput {
        type Metrics = [usize; 1];


        fn size(&self) -> Self::Metrics {
            [1]
        }

        fn tho_device_idx(&self) -> usize {
            1
        }

        fn mem_size(&self) -> u64 {
            size_of::<DummyOutput>() as u64
        }
    }

    impl WithId for DummyOutput {
        type ID = DummyCachableId;
        fn id(&self) -> &Self::ID {
            &self.id
        }
    }

    impl Output for DummyOutput {  }

    #[tokio::test(flavor = "multi_thread")]
    async fn tasks_are_throttled_and_complete() -> Result<(), ThoError> {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);


        let map: DashMap<
            DummyCachableId,
            thoosa_common::runner::common::cachable::CachedFuture<Arc<DevicesWrap<DummyOutput>>>,
        > = DashMap::new();

        let futures_active = Arc::new(AtomicUsize::new(0));
        let max_futures = Arc::new(AtomicUsize::new(0));
        let purge_active = Arc::new(AtomicUsize::new(0));
        let max_purge = Arc::new(AtomicUsize::new(0));
        let processed = Arc::new(AtomicUsize::new(0));

        let total = PURGE_GPU_CONCURRENCY_LIMIT * 4;
        for i in 0..total {
            let key: DummyCachableId =
                DummyCachableId::from(format!("key_{i}").leak() as &'static str);

            let out = DummyOutput {
                id: DummyCachableId::from(format!("out_{i}").leak() as &'static str),
                purge_active: purge_active.clone(),
                max_purge: max_purge.clone(),
            };
            let wrapper = Arc::new(DevicesWrap::wrap(out));
            let active_c = futures_active.clone();
            let max_c = max_futures.clone();
            let processed_c = processed.clone();
            let fut = async move {
                let cur = active_c.fetch_add(1, Ordering::SeqCst) + 1;
                loop {
                    let prev = max_c.load(Ordering::SeqCst);
                    if cur > prev {
                        if max_c
                            .compare_exchange(prev, cur, Ordering::SeqCst, Ordering::SeqCst)
                            .is_ok()
                        {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
                active_c.fetch_sub(1, Ordering::SeqCst);
                processed_c.fetch_add(1, Ordering::SeqCst);
                Ok(wrapper)
            }
            .boxed()
            .shared();

            map.insert(key, fut.into());
        }

        for reff in &map {
            reff.value().clone().await?;
        }

        map.purge_gpu(None, 1, "test").await?;

        assert_eq!(processed.load(Ordering::SeqCst), total);
        // debug!("max_purge={} max_futures={}", max_purge.load(Ordering::SeqCst), max_futures.load(Ordering::SeqCst));
        assert!(max_purge.load(Ordering::SeqCst) > 0);
        assert!(max_futures.load(Ordering::SeqCst) > 0);
        assert!(max_purge.load(Ordering::SeqCst) <= PURGE_GPU_CONCURRENCY_LIMIT);
        assert!(max_futures.load(Ordering::SeqCst) <= PURGE_GPU_CONCURRENCY_LIMIT);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn purge_gpu_skips_required_ids() -> Result<(), ThoError> {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);

        let keep_key = DummyCachableId::from("keep");
        let drop_key = DummyCachableId::from("drop");
        let purge_active = Arc::new(AtomicUsize::new(0));
        let max_purge = Arc::new(AtomicUsize::new(0));
        let keep_wrapper = Arc::new(DevicesWrap::wrap(DummyOutput {
            id: DummyCachableId::from("keep_out"),
            purge_active: Arc::clone(&purge_active),
            max_purge: Arc::clone(&max_purge),
        }));
        let drop_wrapper = Arc::new(DevicesWrap::wrap(DummyOutput {
            id: DummyCachableId::from("drop_out"),
            purge_active,
            max_purge,
        }));

        let map: DashMap<
            DummyCachableId,
            thoosa_common::runner::common::cachable::CachedFuture<Arc<DevicesWrap<DummyOutput>>>,
        > = DashMap::new();
        map.insert(keep_key.clone(), ready(Ok(Arc::clone(&keep_wrapper))).boxed().shared().into());
        map.insert(drop_key, ready(Ok(Arc::clone(&drop_wrapper))).boxed().shared().into());

        for reff in &map {
            reff.value().clone().await?;
        }
        assert!(map.has_gpu_data(1).await);

        let required_ids = HashSet::from([&keep_key as &dyn CachableIdTr]);
        map.purge_gpu(Some(&required_ids), 1, "test").await?;

        assert!(!keep_wrapper.is_all_on_cpu().await);
        assert!(drop_wrapper.is_all_on_cpu().await);
        assert!(map.has_gpu_data(1).await);
        Ok(())
    }
}
