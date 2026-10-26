use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::future::Future;
use std::hash::Hash;
use std::sync::{Arc, LazyLock};
use thoosa_shared::{CudaActivityGate, CudaActivityScope, CudaDeviceGeneration, ThoError};
#[mockall_double::double]
use thoosa_common::runner::global_ctx::GlobalCtx;

/// Stores stage instances in separate typed buckets for each run-context type.
///
/// Each device generation owns separate entries. Per-entry initialization keeps
/// duplicate installs serialized without blocking another device during recovery.
pub struct InstanceCache {
    instances: LazyLock<tokio::sync::Mutex<HashMap<TypeId, Box<dyn Any + Send + Sync>>>>,
}

impl InstanceCache {
    pub const fn new() -> Self {
        Self {
            instances: LazyLock::new(|| tokio::sync::Mutex::new(HashMap::new())),
        }
    }

    pub async fn get_or_try_insert_with<C, K, V, F, Fut>(&self, device: usize, key: K, create: F) -> Result<Arc<V>, ThoError>
    where
        C: 'static,
        K: Clone + Eq + Hash + Send + Sync + 'static,
        V: Send + Sync + 'static,
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = Result<Arc<V>, ThoError>> + Send + 'static,
    {
        let gate = GlobalCtx::global().cuda_activity_gate(device);
        self.get_or_create::<C, K, V, F, Fut>(device, gate.as_ref(), key, create).await
    }

    async fn get_or_create<C, K, V, F, Fut>(&self, device: usize, gate: Option<&Arc<CudaActivityGate>>, key: K, create: F) -> Result<Arc<V>, ThoError>
    where
        C: 'static,
        K: Clone + Eq + Hash + Send + Sync + 'static,
        V: Send + Sync + 'static,
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = Result<Arc<V>, ThoError>> + Send + 'static,
    {
        let activity = gate.map(CudaActivityScope::for_operation).transpose()?;
        let generation = gate.map(|gate| activity.as_ref().expect("device activity").generation(gate).expect("device generation")).unwrap_or_default();
        type Entries<K, V> = HashMap<(usize, CudaDeviceGeneration, K), Arc<tokio::sync::OnceCell<Arc<V>>>>;
        let entry = {
            let mut instances = self.instances.lock().await;
            let cache = instances.entry(TypeId::of::<C>()).or_insert_with(|| Box::new(Entries::<K, V>::new()));
            let cache = cache.downcast_mut::<Entries<K, V>>().ok_or_else(|| {
                ThoError::new(format!("Instance cache type mismatch for context={}, key={}, value={}", type_name::<C>(), type_name::<K>(), type_name::<V>()))
            })?;
            // Evict only this device's retired instances. Existing users keep
            // their Arc, whose proxy identity still rejects use after restart.
            cache.retain(|(cached_device, cached_generation, _), _| *cached_device != device || *cached_generation == generation);
            Arc::clone(cache.entry((device, generation, key)).or_insert_with(|| Arc::new(tokio::sync::OnceCell::new())))
        };
        // An IPC install cannot be undone by dropping its caller. Keep the
        // per-entry initializer alive until it finishes or its device recovers,
        // so a cancelled caller cannot cause a duplicate install in this worker.
        tokio::spawn(async move {
            let initialize = async { entry.get_or_try_init(create).await.cloned() };
            match activity {
                Some(activity) => {
                    let dependencies = activity.clone();
                    activity.run(async {
                        tokio::select! {
                            biased;
                            error = dependencies.wait_for_interruption() => Err(error),
                            instance = initialize => {
                                dependencies.check_dependencies()?;
                                instance
                            },
                        }
                    }).await
                }
                None => initialize.await,
            }
        }).await.map_err(|error| ThoError::new(format!("instance initialization task failed: {error}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_cancelled_caller_does_not_restart_pending_initialization() {
        tokio::time::timeout(Duration::from_secs(2), async {
            let cache = InstanceCache::new();
            let gate = Arc::new(CudaActivityGate::new(1));
            let (started, started_rx) = tokio::sync::oneshot::channel();
            let (complete, completed) = tokio::sync::oneshot::channel();
            let mut first = Box::pin(cache.get_or_create::<(), _, _, _, _>(1, Some(&gate), 10, move || async move {
                started.send(()).unwrap();
                completed.await.unwrap();
                Ok(Arc::new(7u32))
            }));
            assert!(futures::poll!(first.as_mut()).is_pending());
            started_rx.await.unwrap();
            drop(first);
            let mut second = Box::pin(cache.get_or_create::<(), _, _, _, _>(1, Some(&gate), 10, || async { panic!("cancelled caller caused a duplicate install"); #[allow(unreachable_code)] Ok(Arc::new(8u32)) }));
            assert!(futures::poll!(second.as_mut()).is_pending());
            complete.send(()).unwrap();
            assert_eq!(*second.await.unwrap(), 7);
        }).await.expect("the existing install must finish for the next caller");
    }

    #[tokio::test]
    async fn test_instances_are_local_to_device_and_generation() {
        let cache = InstanceCache::new();
        let first = Arc::new(CudaActivityGate::new(1));
        let second = Arc::new(CudaActivityGate::new(2));
        let old = cache.get_or_create::<(), _, _, _, _>(1, Some(&first), 10, || async { Ok(Arc::new(1)) }).await.unwrap();
        let other = cache.get_or_create::<(), _, _, _, _>(2, Some(&second), 10, || async { Ok(Arc::new(2)) }).await.unwrap();
        let cached = cache.get_or_create::<(), _, _, _, _>(1, Some(&first), 10, || async { panic!("duplicate initialization"); #[allow(unreachable_code)] Ok(Arc::new(3)) }).await.unwrap();
        assert!(Arc::ptr_eq(&old, &cached));
        let ticket = first.request_pause(ThoError::ContextReset).unwrap();
        first.wait_until_idle().await;
        first.finish_pause(ticket);
        let fresh = cache.get_or_create::<(), _, _, _, _>(1, Some(&first), 10, || async { Ok(Arc::new(3)) }).await.unwrap();
        assert!(!Arc::ptr_eq(&old, &fresh));
        assert_eq!(*fresh, 3);
        let cached_other = cache.get_or_create::<(), _, _, _, _>(2, Some(&second), 10, || async { panic!("healthy instance was evicted"); #[allow(unreachable_code)] Ok(Arc::new(4)) }).await.unwrap();
        assert!(Arc::ptr_eq(&other, &cached_other));
    }

    #[tokio::test]
    async fn test_initialization_is_per_entry_and_interrupted_by_recovery() {
        tokio::time::timeout(Duration::from_secs(2), async {
            let cache = InstanceCache::new();
            let first = Arc::new(CudaActivityGate::new(1));
            let second = Arc::new(CudaActivityGate::new(2));
            let mut creating = Box::pin(cache.get_or_create::<(), _, _, _, _>(1, Some(&first), 10, || std::future::pending::<Result<Arc<u32>, ThoError>>()));
            assert!(futures::poll!(creating.as_mut()).is_pending());
            let mut duplicate = Box::pin(cache.get_or_create::<(), _, _, _, _>(1, Some(&first), 10, || async { panic!("concurrent duplicate install"); #[allow(unreachable_code)] Ok(Arc::new(5u32)) }));
            assert!(futures::poll!(duplicate.as_mut()).is_pending());
            let other = cache.get_or_create::<(), _, _, _, _>(2, Some(&second), 10, || async { Ok(Arc::new(7u32)) }).await.unwrap();
            assert_eq!(*other, 7);
            let ticket = first.request_pause(ThoError::ContextReset).unwrap();
            assert!(matches!(creating.await.unwrap_err().what(), ThoError::ContextReset));
            assert!(matches!(duplicate.await.unwrap_err().what(), ThoError::ContextReset));
            first.wait_until_idle().await;
            first.finish_pause(ticket);
            let fresh = cache.get_or_create::<(), _, _, _, _>(1, Some(&first), 10, || async { Ok(Arc::new(9u32)) }).await.unwrap();
            assert_eq!(*fresh, 9);
        }).await.expect("initialization must not block another device or its own recovery");
    }
}
