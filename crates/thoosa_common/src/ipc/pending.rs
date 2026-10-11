use std::collections::HashMap;
use parking_lot::Mutex;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use thoosa_shared::ThoError;
use super::worker_msg::{CudaWorkerId, CudaWorkerMsgIn, CudaWorkerMsgOut};

pub(super) type KeepAlive = Box<dyn Send>;
type Reply = oneshot::Sender<Result<CudaWorkerMsgOut, ThoError>>;

struct Request {
    reply: Reply,
    keepalive: Option<KeepAlive>,
}

#[derive(Default)]
struct State {
    closed: Option<ThoError>,
    exited: bool,
    requests: HashMap<u64, Request>,
    retained: Vec<KeepAlive>,
}

impl Drop for State {
    fn drop(&mut self) {
        // If the runtime itself is torn down before reaping, releasing arena
        // buffers could let a surviving worker write into reused allocations.
        // Quarantine them for process lifetime when exit cannot be established.
        if !self.exited {
            for (_, request) in self.requests.drain() {
                if let Some(keepalive) = request.keepalive { std::mem::forget(keepalive); }
            }
            for keepalive in self.retained.drain(..) { std::mem::forget(keepalive); }
        }
    }
}

/// Closing registration and enqueueing share one lock. Closing never resolves
/// in-flight requests: only a checked reply or confirmed child exit can do that.
pub(super) struct PendingRequests {
    worker: CudaWorkerId,
    state: Mutex<State>,
    pub failed: CancellationToken,
}

impl std::fmt::Debug for PendingRequests {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingRequests").field("worker", &self.worker).finish_non_exhaustive()
    }
}

impl PendingRequests {
    pub fn new(worker: CudaWorkerId) -> Self {
        Self { worker, state: Mutex::new(State::default()), failed: CancellationToken::new() }
    }

    pub fn enqueue(&self, permit: mpsc::Permit<'_, CudaWorkerMsgIn>, request: CudaWorkerMsgIn, reply: Reply, keepalive: Option<KeepAlive>, shutdown: bool) -> Result<(), ThoError> {
        request.worker_id.check(self.worker)?;
        let mut state = self.state.lock();
        if state.exited { return Err(ThoError::ContextReset); }
        if !shutdown && let Some(error) = &state.closed { return Err(error.clone()); }
        if state.requests.contains_key(&request.request_id) { return Err(ThoError::new("duplicate worker request identity")); }
        state.requests.insert(request.request_id, Request { reply, keepalive });
        permit.send(request);
        Ok(())
    }

    pub fn close(&self, error: ThoError) {
        self.state.lock().closed.get_or_insert(error);
    }

    pub fn retain_until_reply(&self, id: u64, owner: KeepAlive) -> Result<(), ThoError> {
        let mut state = self.state.lock();
        let request = state.requests.get_mut(&id).ok_or(ThoError::ContextReset)?;
        // Serialization must not drop invocation tensor references while the
        // worker is still reading them. Preserve any existing copy allocation too.
        request.keepalive = Some(Box::new((request.keepalive.take(), owner)));
        Ok(())
    }

    pub fn transport_failed(&self, error: ThoError) {
        self.close(ThoError::ContextReset.context(format!("worker transport failed: {error}")));
        self.failed.cancel();
    }

    pub fn failure(&self) -> ThoError {
        self.state.lock().closed.clone().unwrap_or(ThoError::ContextReset)
    }

    pub fn complete(&self, response: CudaWorkerMsgOut) -> Result<(), ThoError> {
        response.worker_id.check(self.worker)?;
        if let super::worker_msg::OpResult::Failed(error) = &response.status && error.requires_worker_exit() {
            // A failed CUDA synchronization cannot certify that device access
            // to host storage ended. Report promptly, retain storage until exit.
            let request = {
                let mut state = self.state.lock();
                state.closed.get_or_insert_with(|| error.clone().into());
                let mut request = state.requests.remove(&response.request_id);
                if let Some(keepalive) = request.as_mut().and_then(|request| request.keepalive.take()) { state.retained.push(keepalive); }
                request
            };
            self.failed.cancel();
            if let Some(request) = request { let _ = request.reply.send(Ok(response)); }
            return Ok(());
        }
        self.finish_one(response.request_id, Ok(response));
        Ok(())
    }

    pub fn finish_one(&self, id: u64, result: Result<CudaWorkerMsgOut, ThoError>) {
        let request = self.state.lock().requests.remove(&id);
        if let Some(Request { reply, keepalive }) = request {
            drop(keepalive);
            let _ = reply.send(result);
        }
    }

    pub fn exited(&self, error: ThoError) {
        let (requests, retained) = {
            let mut state = self.state.lock();
            state.closed.get_or_insert(error.clone());
            state.exited = true;
            (std::mem::take(&mut state.requests), std::mem::take(&mut state.retained))
        };
        drop(retained);
        for (_, Request { reply, keepalive }) in requests {
            drop(keepalive);
            let _ = reply.send(Err(error.clone()));
        }
    }

    pub fn quarantine(&self, error: ThoError) {
        let requests = {
            let mut state = self.state.lock();
            state.closed.get_or_insert(error.clone());
            let mut requests = std::mem::take(&mut state.requests);
            for request in requests.values_mut() {
                if let Some(keepalive) = request.keepalive.take() { state.retained.push(keepalive); }
            }
            requests
        };
        // Cancellation or an unreapable child must not strand request waiters.
        // Their allocations stay retained (and leak on Drop if exit is unknown).
        for (_, request) in requests { let _ = request.reply.send(Err(error.clone())); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use super::super::worker_msg::{Operation, OpResult};
    use thoosa_shared::{CudaAction, CudaDeviceGeneration};

    fn worker() -> CudaWorkerId { CudaWorkerId { device_idx: 1, generation: CudaDeviceGeneration(3) } }

    #[tokio::test]
    async fn test_retirement_closes_registration_and_retains_cancelled_request_storage_until_exit() {
        let pending = PendingRequests::new(worker());
        let (tx, mut rx) = mpsc::channel(2);
        let storage = Arc::new(vec![0u8; 1024]);
        let weak = Arc::downgrade(&storage);
        let (reply, response) = oneshot::channel();
        pending.enqueue(tx.reserve().await.unwrap(), CudaWorkerMsgIn::new(worker(), 1, Operation::Debug), reply, Some(Box::new(storage)), false).unwrap();
        rx.recv().await.unwrap();
        drop(response);
        let reserved = tx.reserve().await.unwrap();
        pending.close(ThoError::ContextReset);
        let (reply, _) = oneshot::channel();
        assert!(pending.enqueue(reserved, CudaWorkerMsgIn::new(worker(), 2, Operation::Debug), reply, None, false).is_err());
        assert!(rx.try_recv().is_err());
        assert!(weak.upgrade().is_some(), "cancelled callers cannot free worker storage");
        pending.exited(ThoError::ContextReset);
        assert!(weak.upgrade().is_none());
    }

    #[tokio::test]
    async fn test_cancelled_caller_retains_serialized_request_until_reply_or_exit() {
        for transport_failure in [false, true] {
            let pending = PendingRequests::new(worker());
            let (tx, _rx) = mpsc::channel(1);
            let storage = Arc::new(vec![0u8; 8]);
            let request_owner = Arc::new(vec![1u8; 8]);
            let storage_weak = Arc::downgrade(&storage);
            let request_weak = Arc::downgrade(&request_owner);
            let (reply, response) = oneshot::channel();
            pending.enqueue(tx.reserve().await.unwrap(), CudaWorkerMsgIn::new(worker(), 1, Operation::Debug), reply, Some(Box::new(storage)), false).unwrap();
            // The socket writer adds the serialized request's owners without
            // replacing the allocation already retained for a copy operation.
            pending.retain_until_reply(1, Box::new(request_owner)).unwrap();
            drop(response);
            assert!(storage_weak.upgrade().is_some() && request_weak.upgrade().is_some());
            if transport_failure {
                pending.transport_failed(ThoError::new("socket closed"));
                assert!(storage_weak.upgrade().is_some() && request_weak.upgrade().is_some());
                pending.exited(ThoError::ContextReset);
            } else {
                pending.complete(CudaWorkerMsgOut { worker_id: worker(), request_id: 1, status: OpResult::Ok(None) }).unwrap();
            }
            assert!(storage_weak.upgrade().is_none() && request_weak.upgrade().is_none());
        }
    }

    #[tokio::test]
    async fn test_fatal_reply_wakes_caller_but_keeps_storage_until_exit() {
        let pending = PendingRequests::new(worker());
        let (tx, _rx) = mpsc::channel(1);
        let storage = Arc::new(vec![0u8; 1024]);
        let weak = Arc::downgrade(&storage);
        let (reply, response) = oneshot::channel();
        pending.enqueue(tx.reserve().await.unwrap(), CudaWorkerMsgIn::new(worker(), 1, Operation::Debug), reply, Some(Box::new(storage)), false).unwrap();
        pending.complete(CudaWorkerMsgOut { worker_id: worker(), request_id: 1, status: ThoError::Cuda { action: CudaAction::NeedsRestart, msg: "failed synchronization".into() }.into() }).unwrap();
        assert!(matches!(response.await.unwrap().unwrap().status, OpResult::Failed(_)));
        assert!(pending.failed.is_cancelled());
        assert!(pending.failure().needs_cuda_ctx_reset());
        assert!(weak.upgrade().is_some());
        pending.exited(ThoError::ContextReset);
        assert!(weak.upgrade().is_none());
    }

    #[tokio::test]
    async fn test_failed_stream_drain_keeps_storage_until_exit_without_fabricating_fatal_code() {
        let pending = PendingRequests::new(worker());
        let (tx, _rx) = mpsc::channel(1);
        let storage = Arc::new(vec![0u8; 8]);
        let weak = Arc::downgrade(&storage);
        let (reply, response) = oneshot::channel();
        pending.enqueue(tx.reserve().await.unwrap(), CudaWorkerMsgIn::new(worker(), 1, Operation::Debug), reply, Some(Box::new(storage)), false).unwrap();
        // A failed drain is the last-resort transport failure, even though both
        // underlying errors are recoverable in isolation. DMA may still be live.
        let error = crate::copy_ctx::complete_cuda_stream(Err(ThoError::oom()), || Err(ThoError::oom())).result.unwrap_err();
        pending.complete(CudaWorkerMsgOut { worker_id: worker(), request_id: 1, status: error.into() }).unwrap();
        assert!(matches!(response.await.unwrap().unwrap().status, OpResult::Failed(_)));
        assert!(pending.failed.is_cancelled());
        assert!(pending.failure().is_cuda_recovery_failed() && !pending.failure().needs_cuda_ctx_reset());
        assert!(weak.upgrade().is_some());
        pending.exited(ThoError::ContextReset);
        assert!(weak.upgrade().is_none());
    }

    #[tokio::test]
    async fn test_unconfirmed_exit_wakes_callers_without_releasing_storage() {
        let pending = PendingRequests::new(worker());
        let (tx, _rx) = mpsc::channel(1);
        let storage = Arc::new(vec![0u8; 1024]);
        let weak = Arc::downgrade(&storage);
        let (reply, response) = oneshot::channel();
        pending.enqueue(tx.reserve().await.unwrap(), CudaWorkerMsgIn::new(worker(), 1, Operation::Debug), reply, Some(Box::new(storage)), false).unwrap();
        pending.quarantine(ThoError::ContextReset);
        assert!(response.await.unwrap().is_err());
        assert!(weak.upgrade().is_some());
        pending.exited(ThoError::ContextReset);
        assert!(weak.upgrade().is_none());
    }
}
