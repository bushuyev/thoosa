use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::future::Future;
use std::hash::Hash;
use std::sync::{Arc, LazyLock};
use thoosa_shared::ThoError;

/// Stores stage instances in separate typed buckets for each run-context type.
///
/// The mutex intentionally remains locked while an instance is created so concurrent callers
/// cannot install duplicate IPC endpoints for the same cache key.
pub(crate) struct InstanceCache {
    instances: LazyLock<tokio::sync::Mutex<HashMap<TypeId, Box<dyn Any + Send + Sync>>>>,
}

impl InstanceCache {
    pub(crate) const fn new() -> Self {
        Self {
            instances: LazyLock::new(|| tokio::sync::Mutex::new(HashMap::new())),
        }
    }

    pub(crate) async fn get_or_try_insert_with<C, K, V, F, Fut>(&self, key: K, create: F) -> Result<Arc<V>, ThoError>
    where
        C: 'static,
        K: Clone + Eq + Hash + Send + Sync + 'static,
        V: Send + Sync + 'static,
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<Arc<V>, ThoError>> + Send,
    {
        let mut instances = self.instances.lock().await;
        let cache = instances.entry(TypeId::of::<C>()).or_insert_with(|| Box::new(HashMap::<K, Arc<V>>::new()));
        let cache = cache.downcast_mut::<HashMap<K, Arc<V>>>().ok_or_else(|| {
            ThoError::new(format!("Instance cache type mismatch for context={}, key={}, value={}", type_name::<C>(), type_name::<K>(), type_name::<V>()))
        })?;

        if let Some(instance) = cache.get(&key) {
            return Ok(Arc::clone(instance));
        }

        let instance = create().await?;
        cache.insert(key, Arc::clone(&instance));
        Ok(instance)
    }
}
