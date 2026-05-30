use crate::runner::common::cachable::{DevicesWrap, DevicesWrapCopyMode};
use async_trait::async_trait;
use dashmap::DashMap;
use futures_util::StreamExt;
use futures_util::future::{BoxFuture, Shared};
use futures_util::stream;
use log::{debug, trace, warn};
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use dashmap::mapref::multiple::RefMulti;
use mockall::Any;
use thoosa_shared::CopyToDevice;
use thoosa_shared::ThoError;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::traits::{CachableIdTr, Data};
use thoosa_shared::collect_futures_eta::{collect_eta_record_completion, CollectEtaInvocationGuard};
use crate::runner::common::cachable::WithDataRef;

pub const PURGE_GPU_CONCURRENCY_LIMIT: usize = 10;

#[derive(Debug)]
pub struct StaticMapPurger<M: 'static> {
    map: &'static M,
}

impl<M> StaticMapPurger<M> {
    pub const fn new(map: &'static M) -> Self {
        Self { map }
    }
}

#[async_trait]
impl<M> FuturesRunnerMapPurger for StaticMapPurger<M>
where
    M: FuturesRunnerMapPurger + Debug + Sync + Send + 'static,
{
    async fn has_gpu_data(&self, device_idx: usize) -> bool {
        self.map.has_gpu_data(device_idx).await
    }

    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, name: &str) -> Result<(), ThoError> {
        self.map.purge_gpu(ids_to_keep, device_idx, name).await
    }

    async fn drop_gpu(&self, device_idx: usize, name: &str) -> Result<(), ThoError> {
        self.map.drop_gpu(device_idx, name).await
    }

    #[cfg(any(debug_assertions, tho_debug))]
    async fn debug_gpu(&self, device_idx: usize) -> Result<u64, ThoError> {
        self.map.debug_gpu(device_idx).await
    }

    async fn purge_cpu(&self, purge_until: u64) -> usize {
        self.map.purge_cpu(purge_until).await
    }
}

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
            Some(map) => map.has_gpu_data(device_idx).await,
            None => false,
        }
    }

    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self.map.upgrade() {
            Some(map) => map.purge_gpu(ids_to_keep, device_idx, name).await,
            None => Ok(()),
        }
    }

    async fn drop_gpu(&self, device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self.map.upgrade() {
            Some(map) => map.drop_gpu(device_idx, name).await,
            None => Ok(()),
        }
    }

    #[cfg(any(debug_assertions, tho_debug))]
    async fn debug_gpu(&self, device_idx: usize) -> Result<u64, ThoError> {
        match self.map.upgrade() {
            Some(map) => map.debug_gpu(device_idx).await,
            None => Ok(0),
        }
    }

    async fn purge_cpu(&self, purge_until: u64) -> usize {
        match self.map.upgrade() {
            Some(map) => map.purge_cpu(purge_until).await,
            None => 0,
        }
    }
}

#[async_trait]
pub trait FuturesRunnerMapPurger: Debug + Sync + Send {
    async fn has_gpu_data(&self, device_idx: usize) -> bool;

    async fn purge_gpu(
        &self,
        ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>,
        device_idx: usize,
        name: &str,
    ) -> Result<(), ThoError>;

    async fn drop_gpu(
        &self,
        device_idx: usize,
        name: &str,
    ) -> Result<(), ThoError>;

    #[cfg(any(debug_assertions, tho_debug))]
    async fn debug_gpu(
        &self,
        _device_idx: usize,
    ) -> Result<u64, ThoError> {
        Ok(0)
    }

    async fn purge_cpu(&self, purge_until: u64) -> usize;
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
            !set.contains(kd)
        })
        .unwrap_or(true)
}

#[async_trait]
impl<K, O> FuturesRunnerMapPurger
    for DashMap<K, Shared<BoxFuture<'static, Result<Arc<DevicesWrap<O>>, ThoError>>>>
where
    K: CachableIdTr + Clone + Send + Sync + 'static + Eq + Hash,
    O: Data + CopyToDevice + Output + Send + Sync + 'static,
    <O as Data>::M: DevicesWrapCopyMode<O>,
{
    async fn has_gpu_data(&self, device_idx: usize) -> bool {
        self.iter().any(|entry| entry.value().peek().and_then(|value| value.as_ref().ok()).is_some_and(|wrapper| wrapper.is_on_device(device_idx)))
    }

    async fn purge_cpu(&self, purge_until: u64) -> usize {
        let mut purged_count = 0usize;
        self.retain(|k, v| {
            debug!("FuturesRunnerMapPurger::purge_cpu: key={k:?}");
            if v.peek()
                .and_then(|v| v.as_ref().map(|r| {
                    let strong_count = Arc::strong_count(r);
                    debug!("FuturesRunnerMapPurger::purge_cpu: key={k:?} strong_count={}", Arc::strong_count(r));
                    r.is_on_cpu_before(purge_until) && strong_count == 1
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
            Shared<BoxFuture<'static, Result<Arc<DevicesWrap<O>>, ThoError>>>,
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
        stream::iter(items)

            .for_each_concurrent(PURGE_GPU_CONCURRENCY_LIMIT, move |(key, fut)| async move {
                match fut.await {
                    Ok(v) => {
                        if let Err(e) = v.dump_to_cpu(device_idx).await {
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
            })
            .await;

        Ok(())
    }

    #[cfg(any(debug_assertions, tho_debug))]
    async fn debug_gpu(&self, device_idx: usize) -> Result<u64, ThoError> {
        let items: Vec<(K, Arc<DevicesWrap<O>>)> = self
            .iter()
            .filter_map(|entry| {
                entry
                    .value()
                    .peek()
                    .and_then(|value| value.as_ref().ok())
                    .map(|wrapper| (entry.key().clone(), Arc::clone(wrapper)))
            })
            .collect();

        let mut gpu_size = 0;
        for (key, wrapper) in items {
            if !wrapper.is_all_on_cpu().await {
                let size = wrapper.mem_size();
                trace!("key={key:?} size={size:?}");
                gpu_size += size;
            }
        }

        debug!("debug_gpu: device_idx={device_idx} type_name={:?} self.len={} on_gpu: {gpu_size} bytes", self.type_name(), self.len());

        Ok(gpu_size)
    }

    async fn drop_gpu(&self, device_idx: usize, name: &str) -> Result<(), ThoError> {
        let items: Vec<(
            K,
            Shared<BoxFuture<'static, Result<Arc<DevicesWrap<O>>, ThoError>>>,
        )> = self.iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();

        let dump_eta_guard = CollectEtaInvocationGuard::new(&format!("drop_gpu_{:?}", name), self.len(), 1);
        let dump_eta_guard_id = dump_eta_guard.id();
        stream::iter(items)

            .for_each_concurrent(PURGE_GPU_CONCURRENCY_LIMIT, move |(key, fut)| async move {
                match fut.await {
                    Ok(v) => {
                        if let Err(e) = v.drop_device(device_idx).await {
                            warn!("drop_gpu failed: key={key:?} err={e:?}");
                        } else {
                            debug!("drop_gpu ok: key={key:?}");
                        }
                    }
                    Err(e) => {
                        if e.is_oom() {
                            debug!("drop_gpu: Err key={key:?} err={e:?}")
                        } else {
                            warn!("drop_gpu: Err key={key:?} err={e:?}")
                        }
                    },
                }
                collect_eta_record_completion(dump_eta_guard_id);
            })
            .await;

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
    use thoosa_shared::traits::{CachableIdTr, DummyCachableId, WithId};
    use thoosa_shared::copy_mode::CpuOrGpu;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;

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
        type M = CpuOrGpu<Self>;

        fn size(&self) -> Self::Metrics {
            [1]
        }

        fn device(&self) -> usize {
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
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false);


        let map: DashMap<
            DummyCachableId,
            Shared<BoxFuture<'static, Result<Arc<DevicesWrap<DummyOutput>>, ThoError>>>,
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

            map.insert(key, fut);
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
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false);

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
            Shared<BoxFuture<'static, Result<Arc<DevicesWrap<DummyOutput>>, ThoError>>>,
        > = DashMap::new();
        map.insert(keep_key.clone(), ready(Ok(Arc::clone(&keep_wrapper))).boxed().shared());
        map.insert(drop_key, ready(Ok(Arc::clone(&drop_wrapper))).boxed().shared());

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
