use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{Mutex, mpsc};
use thoosa_shared::ThoError;
use super::worker_msg::{CudaWorkerId, WorkerInput};
use super::serialization::{PostcardToFromBytes, ToFromBytes};

/// Reconstruction is opt-in: mutable training progress belongs to its async owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageRecovery {
    OwnerManaged,//Async stage which instantiated the sync stage is responsible for re-creating it after restart
    RecreateFromInit,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct EndpointKey {
    pub endpoint_name: String,
    pub stage_id: Option<String>,
}

// Reusing a key must serialize a pending old deletion with the new installation.
pub(crate) type EndpointSlot = Arc<Mutex<Option<(u64, CudaWorkerId)>>>;

#[derive(Debug)]
pub(crate) enum EndpointEvent {
    Release { key: EndpointKey, token: u64, slot: EndpointSlot },
    Forget { key: EndpointKey, token: u64 },
}

#[derive(Debug)]
pub(crate) struct EndpointRegistration {
    pub key: EndpointKey,
    pub token: u64,
    pub slot: EndpointSlot,
    device: usize,
    // Only initializer payloads are retained; requests and tensor identities are never replayed.
    init: Vec<u8>,
    installed_generation: AtomicU64,
    events: mpsc::UnboundedSender<EndpointEvent>,
}

impl EndpointRegistration {
    pub fn init_data(&self) -> Result<Box<dyn WorkerInput>, ThoError> {
        Ok(PostcardToFromBytes::from_bytes(&self.init)?)
    }

    pub fn installed_worker(&self) -> Option<CudaWorkerId> {
        let generation = self.installed_generation.load(Ordering::Acquire);
        (generation != u64::MAX).then_some(CudaWorkerId { device_idx: self.device, generation: thoosa_shared::CudaDeviceGeneration(generation) })
    }

    pub fn installed(&self, worker: CudaWorkerId) {
        self.installed_generation.store(worker.generation.0, Ordering::Release);
    }
}

impl Drop for EndpointRegistration {
    fn drop(&mut self) {
        // Logical removal must reach the actor even while CUDA admission is paused.
        let _ = self.events.send(EndpointEvent::Release { key: self.key.clone(), token: self.token, slot: Arc::clone(&self.slot) });
    }
}

#[derive(Debug)]
struct Entry {
    token: u64,
    owner: Weak<EndpointRegistration>,
    slot: EndpointSlot,
}

#[derive(Debug)]
pub(crate) struct EndpointRegistry {
    entries: HashMap<EndpointKey, Entry>,
    next_token: u64,
    events: mpsc::UnboundedSender<EndpointEvent>,
}

impl EndpointRegistry {
    pub fn new(events: mpsc::UnboundedSender<EndpointEvent>) -> Self {
        Self { entries: HashMap::new(), next_token: 0, events }
    }

    pub fn register(&mut self, device: usize, key: EndpointKey, init: Box<dyn WorkerInput>) -> Result<Arc<EndpointRegistration>, ThoError> {
        if self.entries.get(&key).is_some_and(|entry| entry.owner.upgrade().is_some()) {
            return Err(ThoError::new(format!("endpoint already has a live registration: {key:?}")));
        }
        let init = PostcardToFromBytes::to_bytes(&init)?;
        let slot = self.entries.get(&key).map(|entry| Arc::clone(&entry.slot)).unwrap_or_default();
        let token = self.next_token;
        self.next_token = token.checked_add(1).ok_or_else(|| ThoError::new("endpoint registration token overflow"))?;
        let owner = Arc::new(EndpointRegistration { key: key.clone(), token, slot: Arc::clone(&slot), device, init, installed_generation: AtomicU64::new(u64::MAX), events: self.events.clone() });
        self.entries.insert(key, Entry { token, owner: Arc::downgrade(&owner), slot });
        Ok(owner)
    }

    pub fn contains(&self, owner: &EndpointRegistration) -> bool {
        self.entries.get(&owner.key).is_some_and(|entry| entry.token == owner.token)
    }

    pub fn forget(&mut self, key: &EndpointKey, token: u64) {
        // A delayed destructor/acknowledgement cannot unregister a new owner of the same key.
        if self.entries.get(key).is_some_and(|entry| entry.token == token && entry.owner.upgrade().is_none()) {
            self.entries.remove(key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_release_does_not_forget_replacement_owner() -> Result<(), ThoError> {
        let (events, mut receiver) = mpsc::unbounded_channel();
        let mut registry = EndpointRegistry::new(events);
        let key = EndpointKey { endpoint_name: "stage".into(), stage_id: Some("1".into()) };
        let old = registry.register(1, key.clone(), Box::new(()))?;
        // Snapshot decoding must work repeatedly without consuming the recipe.
        old.init_data()?;
        old.init_data()?;
        assert!(registry.register(1, key.clone(), Box::new(())).is_err());
        let slot = old.slot.clone();
        let token = old.token;
        drop(old);
        let new = registry.register(1, key.clone(), Box::new(()))?;
        assert!(Arc::ptr_eq(&slot, &new.slot), "replacement must serialize with pending cleanup");
        assert_ne!(new.token, token);
        assert!(matches!(receiver.recv().await, Some(EndpointEvent::Release { token: released, .. }) if released == token));
        registry.forget(&key, token);
        assert!(registry.contains(&new), "old cleanup cannot unregister the replacement");
        let token = new.token;
        drop(new);
        registry.forget(&key, token);
        assert!(registry.entries.is_empty());
        Ok(())
    }

    #[test]
    fn test_endpoint_identity_preserves_both_components() {
        use crate::ipc::cuda_process::endpoint_stage_key;
        assert_ne!(endpoint_stage_key("ab".into(), Some(&"c")), endpoint_stage_key("a".into(), Some(&"bc")));
        assert_ne!(endpoint_stage_key("a".into(), None::<&&str>), endpoint_stage_key("a".into(), Some(&"")));
    }
}
