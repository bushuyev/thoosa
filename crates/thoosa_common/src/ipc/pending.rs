use super::worker_msg::{CudaWorkerId, CudaWorkerMsgIn, CudaWorkerMsgOut};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use thoosa_shared::ThoError;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

pub(super) type KeepAlive = Box<dyn Send>;
type Reply = oneshot::Sender<Result<CudaWorkerMsgOut, ThoError>>;

struct Request {
    reply: Reply,
    keepalive: Option<KeepAlive>,
    summary: String,
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
                if let Some(keepalive) = request.keepalive {
                    std::mem::forget(keepalive);
                }
            }
            for keepalive in self.retained.drain(..) {
                std::mem::forget(keepalive);
            }
        }
    }
}

/// Closing registration and enqueueing share one lock. Closing never resolves
/// in-flight requests: only a checked reply or confirmed child exit can do that.
pub(super) struct PendingRequests {
    worker: CudaWorkerId,
    pub peer_pid: Option<u32>,
    pub gate: Option<Arc<thoosa_shared::CudaActivityGate>>,
    pub runtime_failure: Option<Arc<crate::runner::global_ctx::RuntimeFailure>>,
    state: Mutex<State>,
    pub failed: CancellationToken,
}

impl std::fmt::Debug for PendingRequests {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingRequests").field("worker", &self.worker).field("peer_pid", &self.peer_pid).finish_non_exhaustive()
    }
}

impl PendingRequests {
    pub fn new(worker: CudaWorkerId) -> Self {
        Self {
            worker,
            peer_pid: None,
            gate: None,
            runtime_failure: None,
            state: Mutex::new(State::default()),
            failed: CancellationToken::new(),
        }
    }

    pub fn enqueue(&self, permit: mpsc::Permit<'_, CudaWorkerMsgIn>, request: CudaWorkerMsgIn, reply: Reply, keepalive: Option<KeepAlive>, shutdown: bool) -> Result<(), ThoError> {
        request.worker_id.check(self.worker)?;
        let mut state = self.state.lock();
        if state.exited {
            return Err(ThoError::CudaBeingRecovered);
        }
        if !shutdown && let Some(error) = &state.closed {
            return Err(error.clone());
        }
        if state.requests.contains_key(&request.request_id) {
            return Err(ThoError::new("duplicate worker request identity"));
        }
        state.requests.insert(request.request_id, Request { reply, keepalive, summary: request.operation.summary() });
        permit.send(request);
        Ok(())
    }

    pub fn close(&self, error: ThoError) {
        self.state.lock().closed.get_or_insert(error);
    }

    pub fn retain_until_reply(&self, id: u64, owner: KeepAlive) -> Result<(), ThoError> {
        let mut state = self.state.lock();
        let request = state.requests.get_mut(&id).ok_or(ThoError::CudaBeingRecovered)?;
        // Serialization must not drop invocation tensor references while the
        // worker is still reading them. Preserve any existing copy allocation too.
        request.keepalive = Some(Box::new((request.keepalive.take(), owner)));
        Ok(())
    }

    pub fn transport_failed(&self, error: ThoError) {
        // A protocol bug must remain terminal, even if a socket failure raced
        // it. Do not let the transport guard replace it with a reset request.
        if error.is_ipc_protocol_failure() {
            self.state.lock().closed = Some(error.clone());
            // Publish before actor scheduling/recovery can observe a concurrent
            // socket error and mistake this protocol bug for a restart request.
            if let Some(gate) = &self.gate {
                gate.fail_process(error.clone());
            }
            if let Some(failure) = &self.runtime_failure {
                failure.fail(error);
            }
        } else {
            self.close(ThoError::CudaBeingRecovered.context(format!("worker transport failed: {error}")));
        }
        self.failed.cancel();
    }

    pub fn describe(&self, request_id: u64) -> String {
        self.state.lock().requests.get(&request_id).map(|request| request.summary.clone()).unwrap_or_else(|| "unknown request".into())
    }

    pub fn failure(&self) -> ThoError {
        self.state.lock().closed.clone().unwrap_or(ThoError::CudaBeingRecovered)
    }

    pub fn complete(&self, response: CudaWorkerMsgOut) -> Result<(), ThoError> {
        if response.worker_id != self.worker {
            return Err(ThoError::IpcProtocol(format!(
                "response worker {:?} does not match {:?}; request_id={} {}",
                response.worker_id,
                self.worker,
                response.request_id,
                self.describe(response.request_id)
            )));
        }
        if let super::worker_msg::OpResult::Failed(error) = &response.status
            && error.requires_worker_exit()
        {
            // A failed CUDA synchronization cannot certify that device access
            // to host storage ended. Report promptly, retain storage until exit.
            let request = {
                let mut state = self.state.lock();
                let error: ThoError = error.clone().into();
                if error.is_ipc_protocol_failure() {
                    state.closed = Some(error);
                } else {
                    state.closed.get_or_insert(error);
                }
                let mut request = state.requests.remove(&response.request_id);
                if let Some(keepalive) = request.as_mut().and_then(|request| request.keepalive.take()) {
                    state.retained.push(keepalive);
                }
                request
            };
            let failure: ThoError = error.clone().into();
            if failure.is_ipc_protocol_failure() {
                self.transport_failed(failure);
            }
            self.failed.cancel();
            if let Some(request) = request {
                let _ = request.reply.send(Ok(response));
            }
            return Ok(());
        }
        self.finish_one(response.request_id, Ok(response));
        Ok(())
    }

    pub fn finish_one(&self, id: u64, result: Result<CudaWorkerMsgOut, ThoError>) {
        let request = self.state.lock().requests.remove(&id);
        if let Some(Request { reply, keepalive, .. }) = request {
            drop(keepalive);
            let _ = reply.send(result);
        }
    }

    pub fn exited(&self, error: ThoError) {
        let (requests, retained, error) = {
            let mut state = self.state.lock();
            state.closed.get_or_insert(error.clone());
            // An I/O/retirement race must not erase the terminal decoder cause
            // already published to the gate and application.
            let error = state.closed.as_ref().filter(|error| error.is_ipc_protocol_failure()).cloned().unwrap_or(error);
            state.exited = true;
            (std::mem::take(&mut state.requests), std::mem::take(&mut state.retained), error)
        };
        drop(retained);
        for (_, Request { reply, keepalive, .. }) in requests {
            drop(keepalive);
            let _ = reply.send(Err(error.clone()));
        }
    }

    pub fn quarantine(&self, error: ThoError) {
        let (requests, error) = {
            let mut state = self.state.lock();
            state.closed.get_or_insert(error.clone());
            let error = state.closed.as_ref().filter(|error| error.is_ipc_protocol_failure()).cloned().unwrap_or(error);
            let mut requests = std::mem::take(&mut state.requests);
            for request in requests.values_mut() {
                if let Some(keepalive) = request.keepalive.take() {
                    state.retained.push(keepalive);
                }
            }
            (requests, error)
        };
        // Cancellation or an unreapable child must not strand request waiters.
        // Their allocations stay retained (and leak on Drop if exit is unknown).
        for (_, request) in requests {
            let _ = request.reply.send(Err(error.clone()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::worker_msg::{OpResult, Operation};
    use super::*;
    use std::sync::Arc;
    use thoosa_shared::{CudaAction, CudaDeviceGeneration};

    fn worker() -> CudaWorkerId {
        CudaWorkerId { device_idx: 1, generation: CudaDeviceGeneration(3) }
    }

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
        pending.close(ThoError::CudaBeingRecovered);
        let (reply, _) = oneshot::channel();
        assert!(pending.enqueue(reserved, CudaWorkerMsgIn::new(worker(), 2, Operation::Debug), reply, None, false).is_err());
        assert!(rx.try_recv().is_err());
        assert!(weak.upgrade().is_some(), "cancelled callers cannot free worker storage");
        pending.exited(ThoError::CudaBeingRecovered);
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
                pending.exited(ThoError::CudaBeingRecovered);
            } else {
                pending
                    .complete(CudaWorkerMsgOut {
                        worker_id: worker(),
                        request_id: 1,
                        status: OpResult::Ok(None),
                    })
                    .unwrap();
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
        // Use the constructor so injected CUDA errors capture their backtrace.
        pending
            .complete(CudaWorkerMsgOut {
                worker_id: worker(),
                request_id: 1,
                status: thoosa_shared::tho_error::cuda_error(CudaAction::NeedsRestart, "failed synchronization".into()).into(),
            })
            .unwrap();
        assert!(matches!(response.await.unwrap().unwrap().status, OpResult::Failed(_)));
        assert!(pending.failed.is_cancelled());
        assert!(pending.failure().needs_cuda_recovery());
        assert!(weak.upgrade().is_some());
        pending.exited(ThoError::CudaBeingRecovered);
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
        let error = crate::cuda_copier::complete_cuda_stream(Err(ThoError::oom()), || Err(ThoError::oom())).result.unwrap_err();
        pending
            .complete(CudaWorkerMsgOut {
                worker_id: worker(),
                request_id: 1,
                status: error.into(),
            })
            .unwrap();
        assert!(matches!(response.await.unwrap().unwrap().status, OpResult::Failed(_)));
        assert!(pending.failed.is_cancelled());
        assert!(pending.failure().is_cuda_recovery_failed() && !pending.failure().needs_cuda_recovery());
        assert!(weak.upgrade().is_some());
        pending.exited(ThoError::CudaBeingRecovered);
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
        pending.quarantine(ThoError::CudaBeingRecovered);
        assert!(response.await.unwrap().is_err());
        assert!(weak.upgrade().is_some());
        pending.exited(ThoError::CudaBeingRecovered);
        assert!(weak.upgrade().is_none());
    }
}
