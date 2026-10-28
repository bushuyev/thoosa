use std::sync::Arc;
use anyhow::Context;
use futures_util::future::BoxFuture;
use log::{debug, error, trace};
use parking_lot::Mutex;
use tokio::sync::{mpsc, watch};
use tracing::Instrument;
use thoosa_shared::{panic_to_err_async, CudaActivityGate, ThoError};
use thoosa_shared::tracked_await::track_await;

pub enum PurgeCmd {
    Gpu {
        closures: Vec<Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>> + Send + Sync>>
    },
}

/// Submission acknowledges queue admission, not completion. Shutdown closes admission
/// and drains accepted cleanup; dropping the owner also closes admission, without waiting.
#[derive(Debug)]
pub struct Purger {
    // Serialize submission with shutdown, without retaining any receiver outside the worker.
    tx: Mutex<Option<mpsc::UnboundedSender<PurgeCmd>>>,
    completion: watch::Receiver<Option<Result<(), ThoError>>>,
}

impl Purger {
    pub fn new(gate: Option<Arc<CudaActivityGate>>) -> Self {
        debug!("Purger::new");

        let (tx, rx) = mpsc::unbounded_channel::<PurgeCmd>();
        let (completed, completion) = watch::channel(None);
        let worker = tokio::spawn(Self::run(rx, gate).instrument(tracing::info_span!(
            "purger::worker",
            kind = "spawn",
            file = file!(),
            line = line!(),
            task = "process GPU purge commands",
        )));

        // Observe even an unwinding/aborted worker. This observer owns no work-channel
        // handles and remains alive to report failures if the Purger owner is dropped.
        tokio::spawn(async move {
            let result = match worker.await {
                Ok(result) => result,
                Err(error) => Err(ThoError::from(error).context("purger worker stopped")),
            };
            if let Err(error) = &result {
                error!("Purger worker failed: {error:#}");
            }
            completed.send_replace(Some(result));
        });

        Self { tx: Mutex::new(Some(tx)), completion }
    }

    async fn run(mut rx: mpsc::UnboundedReceiver<PurgeCmd>, gate: Option<Arc<CudaActivityGate>>) -> Result<(), ThoError> {
        let mut first_error = None;
        while let Some(PurgeCmd::Gpu { closures }) = track_await!(rx.recv()).await {
            // A terminal gate failure cannot safely drain GPU work. Returning drops
            // the receiver and pending callbacks, closes admission, and preserves the cause.
            if let Some(gate) = &gate {
                gate.wait_if_paused().await?;
            }
            debug!("Purger::run: closures.len={}", closures.len());
            for purge_cl in closures {
                match panic_to_err_async!(purge_cl()) {
                    Ok(()) => trace!("Purger::run: done purging"),
                    Err(error) => {
                        // Independent cleanup still gets its chance; shutdown reports the first failure.
                        error!("Purger::run: purge failed: {error:#}");
                        first_error.get_or_insert(error);
                    }
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    pub fn run_purge_gpu(&self, closures: Vec<Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>> + Send + Sync>>) -> Result<(), ThoError> {
        debug!("Purger::run_purge_gpu: closures.len={}", closures.len());
        let command = PurgeCmd::Gpu { closures };
        let rejected = {
            let tx = self.tx.lock();
            match tx.as_ref() {
                Some(tx) => tx.send(command).err().map(|error| error.0),
                None => Some(command),
            }
        };
        if let Some(command) = rejected {
            // Captured owners can submit cleanup from Drop; release the admission lock first.
            drop(command);
            match self.completion.borrow().as_ref() {
                Some(Err(error)) => Err(error.clone()),
                _ => Err(ThoError::new("purger is closed")),
            }
        } else {
            Ok(())
        }
    }

    /// Wait for accepted cleanup before device teardown. Worker failure discards
    /// pending callbacks and returns its error. Concurrent/cancelled waiters cannot
    /// detach the join or consume the result needed by another shutdown caller.
    pub async fn shutdown(&self) -> Result<(), ThoError> {
        self.tx.lock().take();
        let mut completion = self.completion.clone();
        loop {
            if let Some(result) = completion.borrow_and_update().clone() {
                return result;
            }
            completion.changed().await.context("purger completion observer stopped")?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::FutureExt;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::sync::Notify;

    async fn wait_for_completion(mut completion: watch::Receiver<Option<Result<(), ThoError>>>) -> Result<(), ThoError> {
        // Bound every lifecycle assertion so a leaked receiver or lost wakeup fails the test.
        let result = tokio::time::timeout(Duration::from_secs(5), completion.wait_for(|result| result.is_some())).await?.context("purger completion observer stopped")?;
        result.as_ref().expect("completed worker result").clone()
    }

    #[tokio::test]
    async fn test_failed_gate_closes_queue_and_releases_pending_cleanup() -> Result<(), ThoError> {
        let gate = Arc::new(CudaActivityGate::new(1));
        let ticket = gate.request_pause(ThoError::ContextReset)?;
        let purger = Purger::new(Some(Arc::clone(&gate)));
        let retained = Arc::new(());
        let weak = Arc::downgrade(&retained);
        let executed = Arc::new(AtomicUsize::new(0));
        for _ in 0..2 {
            let retained = Arc::clone(&retained);
            let executed = Arc::clone(&executed);
            purger.run_purge_gpu(vec![Box::new(move || {
                let _keep = &retained;
                executed.fetch_add(1, Ordering::SeqCst);
                async { Ok(()) }.boxed()
            })])?;
        }
        drop(retained);
        gate.fail_pause(ticket, Some(ThoError::MissingEndpoint("purger gate recovery".into())));

        // Do not call shutdown first: the worker itself must close the queue on failure.
        let failure = wait_for_completion(purger.completion.clone()).await.unwrap_err();
        assert!(matches!(failure.what(), ThoError::CudaRecoveryFailed(_)));
        assert_eq!(executed.load(Ordering::SeqCst), 0);
        assert!(weak.upgrade().is_none(), "failed admission must release queued callbacks");
        let rejected = purger.run_purge_gpu(vec![]).unwrap_err();
        assert!(matches!(rejected.what(), ThoError::CudaRecoveryFailed(_)));
        assert!(matches!(purger.shutdown().await.unwrap_err().what(), ThoError::CudaRecoveryFailed(_)));
        Ok(())
    }

    #[tokio::test]
    async fn test_worker_panic_is_observed_and_closes_queue() -> Result<(), ThoError> {
        let purger = Purger::new(None);
        purger.run_purge_gpu(vec![Box::new(|| async {
            // Panic during polling, beyond the legacy callback-construction panic adapter.
            tokio::task::yield_now().await;
            panic!("injected purge worker panic");
        }.boxed())])?;
        let retained = Arc::new(());
        let weak = Arc::downgrade(&retained);
        purger.run_purge_gpu(vec![Box::new(move || {
            let _keep = &retained;
            panic!("queued callback must not run after worker panic");
        })])?;

        let failure = wait_for_completion(purger.completion.clone()).await.unwrap_err();
        let ThoError::Anyhow(source) = &failure else { panic!("expected the worker JoinError") };
        assert!(source.downcast_ref::<tokio::task::JoinError>().is_some_and(|error| error.is_panic()));
        assert!(weak.upgrade().is_none());
        assert!(purger.run_purge_gpu(vec![]).is_err());
        assert!(purger.shutdown().await.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_shutdown_closes_admission_drains_in_order_and_survives_cancelled_waiter() -> Result<(), ThoError> {
        let purger = Purger::new(None);
        let started = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let processed = Arc::new(Mutex::new(Vec::new()));
        for value in 0..3 {
            let started = Arc::clone(&started);
            let release = Arc::clone(&release);
            let processed = Arc::clone(&processed);
            purger.run_purge_gpu(vec![Box::new(move || {
                let started = Arc::clone(&started);
                let release = Arc::clone(&release);
                let processed = Arc::clone(&processed);
                async move {
                    if value == 0 {
                        started.notify_one();
                        release.notified().await;
                    }
                    processed.lock().push(value);
                    Ok(())
                }.boxed()
            })])?;
        }
        tokio::time::timeout(Duration::from_secs(5), started.notified()).await?;
        let mut shutdown = Box::pin(purger.shutdown());
        assert!(futures_util::poll!(shutdown.as_mut()).is_pending(), "shutdown must wait for accepted work");
        assert!(purger.run_purge_gpu(vec![]).is_err(), "shutdown closes admission before draining");
        drop(shutdown);

        // Cancelling one waiter must not lose the join/result or cancel accepted work.
        release.notify_one();
        let (first, second) = tokio::time::timeout(Duration::from_secs(5), async { tokio::join!(purger.shutdown(), purger.shutdown()) }).await?;
        first?;
        second?;
        assert_eq!(*processed.lock(), vec![0, 1, 2]);
        purger.shutdown().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_owner_drop_drains_accepted_commands() -> Result<(), ThoError> {
        let purger = Purger::new(None);
        let completion = purger.completion.clone();
        let executed = Arc::new(AtomicUsize::new(0));
        for _ in 0..3 {
            let executed = Arc::clone(&executed);
            purger.run_purge_gpu(vec![Box::new(move || {
                let executed = Arc::clone(&executed);
                async move {
                    tokio::task::yield_now().await;
                    executed.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }.boxed()
            })])?;
        }
        // Only the completion observer survives; it must not keep queue admission open.
        drop(purger);
        wait_for_completion(completion).await?;
        assert_eq!(executed.load(Ordering::SeqCst), 3);
        Ok(())
    }

    #[tokio::test]
    async fn test_cleanup_error_still_drains_remaining_callbacks_and_is_reported() -> Result<(), ThoError> {
        let purger = Purger::new(None);
        let executed = Arc::new(AtomicUsize::new(0));
        let completed = Arc::clone(&executed);
        purger.run_purge_gpu(vec![
            Box::new(|| async { Err(ThoError::MissingEndpoint("purge".into())) }.boxed()),
            Box::new(move || {
                completed.fetch_add(1, Ordering::SeqCst);
                async { Ok(()) }.boxed()
            }),
        ])?;
        let completed = Arc::clone(&executed);
        purger.run_purge_gpu(vec![Box::new(move || {
            completed.fetch_add(1, Ordering::SeqCst);
            async { Ok(()) }.boxed()
        })])?;
        let error = tokio::time::timeout(Duration::from_secs(5), purger.shutdown()).await?.unwrap_err();
        assert!(error.is_missing_endpoint());
        assert_eq!(executed.load(Ordering::SeqCst), 2);
        assert!(purger.run_purge_gpu(vec![]).unwrap_err().is_missing_endpoint());
        Ok(())
    }
}
