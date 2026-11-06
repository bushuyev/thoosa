//! Residency belongs to an async handle, independently of memoization or cache areas.
use std::sync::{Arc, Weak, atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering}};
use dashmap::DashMap;
use futures_util::{StreamExt, TryStreamExt};
use thoosa_shared::ThoError;
use tokio::sync::RwLock;
use super::purge::{CachePurger, PURGE_GPU_CONCURRENCY_LIMIT};

#[derive(Debug)]
pub struct ManagedResidency {
    devices: usize,
    next: AtomicU64,
    entries: DashMap<u64, Arc<Entry>>,
}

#[derive(Debug)]
struct Entry {
    resource: Weak<dyn CachePurger>,
    active: AtomicBool,
    gate: RwLock<()>,
}

/// An owned exclusion, scoped to a device and independent of cache names and IDs.
/// It retains only counters, so consuming an input never waits for its own lease.
#[derive(Debug, Default)]
pub struct ResidencyLease(Vec<Arc<AtomicUsize>>);

impl ResidencyLease {
    pub fn combine(leases: impl IntoIterator<Item = Self>) -> Self {
        Self(leases.into_iter().flat_map(|mut lease| std::mem::take(&mut lease.0)).collect())
    }
    pub(crate) fn acquire(counter: &Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        Self(vec![Arc::clone(counter)])
    }
}

impl Drop for ResidencyLease {
    fn drop(&mut self) {
        for counter in &self.0 { counter.fetch_sub(1, Ordering::AcqRel); }
    }
}

#[derive(Debug)]
pub(crate) struct Registration {
    registry: Weak<ManagedResidency>,
    id: u64,
    entry: Arc<Entry>,
}

impl Registration {
    fn remove(&self) {
        self.entry.active.store(false, Ordering::Release);
        if let Some(registry) = self.registry.upgrade() { registry.entries.remove(&self.id); }
    }
    /// A snapshot may still own the entry. Close admission before waiting for
    /// existing offloads, so consuming data cannot race a late weak upgrade.
    pub async fn detach(self) {
        self.remove();
        let _guard = self.entry.gate.write().await;
    }
}

impl Drop for Registration {
    fn drop(&mut self) { self.remove(); }
}

impl ManagedResidency {
    pub fn new(devices: usize) -> Arc<Self> {
        Arc::new(Self { devices, next: AtomicU64::new(0), entries: DashMap::new() })
    }
    pub fn device_count(&self) -> usize { self.devices }
    pub(crate) fn register(self: &Arc<Self>, resource: Arc<dyn CachePurger>) -> Registration {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let entry = Arc::new(Entry { resource: Arc::downgrade(&resource), active: AtomicBool::new(true), gate: RwLock::new(()) });
        self.entries.insert(id, Arc::clone(&entry));
        Registration { registry: Arc::downgrade(self), id, entry }
    }
    fn snapshot(&self) -> Vec<Arc<Entry>> { self.entries.iter().map(|entry| Arc::clone(entry.value())).collect() }

    pub async fn offload(&self, device: usize) -> Result<(), ThoError> {
        // No registry guard crosses an await; reuse the existing transfer and
        // concurrency limit. Leases and active tensor borrows are checked there.
        futures_util::stream::iter(self.snapshot()).map(Ok::<_, ThoError>).try_for_each_concurrent(PURGE_GPU_CONCURRENCY_LIMIT, |entry| async move {
            let _guard = entry.gate.read().await;
            if entry.active.load(Ordering::Acquire) {
                if let Some(resource) = entry.resource.upgrade() { resource.purge_gpu(None, device, "managed handle").await?; }
            }
            Ok(())
        }).await
    }

    pub async fn drop_device(&self, device: usize) -> Result<(), ThoError> {
        // Recovery invalidates even protected inputs. Preserve healthy replicas
        // and attempt every entry before reporting a failure.
        let mut failure = None;
        for entry in self.snapshot() {
            let _guard = entry.gate.read().await;
            if entry.active.load(Ordering::Acquire) {
                if let Some(resource) = entry.resource.upgrade() {
                    if let Err(error) = resource.drop_gpu(device, "managed handle").await { failure.get_or_insert(error); }
                }
            }
        }
        failure.map_or(Ok(()), Err)
    }

    pub async fn has_gpu_data(&self, device: usize) -> bool {
        for entry in self.snapshot() {
            let _guard = entry.gate.read().await;
            if entry.active.load(Ordering::Acquire) {
                if let Some(resource) = entry.resource.upgrade() {
                    if resource.has_gpu_data(device).await { return true; }
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::{common::cachable::DevicesWrap, global_ctx::mock_global_ctx::MockGlobalCtxBuilder, test_utils::MockDataWithId};

    #[tokio::test]
    async fn test_managed_handles_survive_cache_removal_and_isolate_equal_ids() -> Result<(), ThoError> {
        let _global = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);
        let registry = ManagedResidency::new(2);
        let first = Arc::new(DevicesWrap::wrap_in(MockDataWithId::for_device_and_id(1, "same-id"), &registry));
        let second = Arc::new(DevicesWrap::wrap_in(MockDataWithId::for_device_and_id(1, "same-id"), &registry));
        let mut cache = std::collections::HashMap::from([(1, Arc::clone(&first)), (2, Arc::clone(&second))]);
        cache.clear();
        assert_eq!(registry.entries.len(), 2, "cache removal must not hide external handles");
        let lease = first.protect(1);
        registry.offload(1).await?;
        assert!(first.is_on_device(1));
        assert!(second.is_on_device(0) && !second.is_on_device(1));
        drop(lease);
        registry.offload(1).await?;
        assert!(first.is_on_device(0) && !first.is_on_device(1));
        drop(first);
        drop(second);
        assert!(registry.entries.is_empty(), "registration must not retain dead handles");
        Ok(())
    }

    #[tokio::test]
    async fn test_managed_lease_cancellation_releases_protection() -> Result<(), ThoError> {
        let _global = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);
        let registry = ManagedResidency::new(2);
        let input = DevicesWrap::wrap_in(MockDataWithId::for_device(1), &registry);
        let lease = input.protect(1);
        let mut suspended = Box::pin(async move { let _lease = lease; std::future::pending::<()>().await; });
        assert!(futures_util::poll!(suspended.as_mut()).is_pending());
        registry.offload(1).await?;
        assert!(input.is_on_device(1));
        drop(suspended);
        registry.offload(1).await?;
        assert!(!input.is_on_device(1));
        Ok(())
    }

    #[tokio::test]
    async fn test_managed_recovery_invalidates_protected_uncached_handles() -> Result<(), ThoError> {
        let _global = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);
        let registry = ManagedResidency::new(2);
        let backed_up = DevicesWrap::wrap_in(MockDataWithId::for_device(1), &registry);
        drop(backed_up.load_from_any_device_(0).await?);
        let gpu_only = DevicesWrap::wrap_in(MockDataWithId::for_device(1), &registry);
        let _lease = ResidencyLease::combine([backed_up.protect(1), gpu_only.protect(1)]);
        registry.drop_device(1).await?;
        assert!(backed_up.is_on_device(0) && !backed_up.is_on_device(1));
        assert!(!gpu_only.is_on_device(0) && !gpu_only.is_on_device(1));
        assert!(!registry.has_gpu_data(1).await);
        Ok(())
    }

    #[tokio::test]
    async fn test_managed_registrations_belong_to_their_runtime() -> Result<(), ThoError> {
        let _global = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);
        let first = ManagedResidency::new(2);
        let second = ManagedResidency::new(2);
        let a = DevicesWrap::wrap_in(MockDataWithId::for_device(1), &first);
        let b = DevicesWrap::wrap_in(MockDataWithId::for_device(1), &second);
        first.offload(1).await?;
        assert!(!a.is_on_device(1) && b.is_on_device(1));
        second.offload(1).await?;
        assert!(!b.is_on_device(1));
        Ok(())
    }
}
