use async_trait::async_trait;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use log::{error, info, trace, debug};
use thoosa_shared::traits::CachableIdTr;
use thoosa_shared::{MyError, ThoosaError};
use tokio::sync::{Notify, Semaphore, TryAcquireError, mpsc, oneshot};
use tokio::task::JoinHandle;
use crate::runner::permits::{DropSignal, NumPermits, PermitSource, RunnerPerm};

/// ---------- FIFO Arbiter built on tokio::Semaphore ----------

#[derive(Debug)]
pub struct Arbiter {
    #[allow(dead_code)]
    sem: Arc<Semaphore>, //TODO for statistics
    notify: Arc<Notify>,
    tx: Option<mpsc::Sender<Req>>,
    capacity: NumPermits,
    worker: Option<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    granted_counts_len: Arc<AtomicUsize>,//TODO remove?
    device_idx: usize
}

#[derive(Debug)]
struct Req {
    needed: NumPermits,
    reply: oneshot::Sender<Result<RunnerPerm, ThoosaError>>,
    requester_id: Arc<dyn CachableIdTr>,
}

impl Arbiter {
    pub fn new(capacity: NumPermits, device_idx: usize) -> Self {
        debug!("Arbiter::new: capacity={capacity} device_idx={device_idx}");

        let (tx, rx) = mpsc::channel::<Req>(1024);
        let sem = Arc::new(Semaphore::new(capacity as usize));
        let notify = Arc::new(Notify::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        let granted_counts_len = Arc::new(AtomicUsize::new(0));

        let worker = Self::spawn(
            Arc::clone(&sem),
            Arc::clone(&notify),
            Arc::clone(&shutdown),
            Arc::clone(&granted_counts_len),
            rx,
            device_idx,
        );
        Self {
            sem,
            notify,
            tx: Some(tx),
            capacity,
            worker: Some(worker),
            shutdown,
            granted_counts_len,
            device_idx
        }
    }

    async fn send_acquire_rq(
        &self,
        mut needed: NumPermits,
        requester_id: Arc<dyn CachableIdTr>,
    ) -> Result<RunnerPerm, ThoosaError> {
        // Clamp to capacity to avoid unfulfillable requests.
        if needed as usize > self.capacity as usize {
            needed = self.capacity;
        }

        let (reply, recv) = oneshot::channel();
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| MyError::SomethingElse("acquire_rq 0".to_string()))?;

        tx.send(Req {
            needed,
            reply,
            requester_id: requester_id.clone(),
        })
        .await
        .map_err(|e| MyError::SomethingElse(format!("send_acquire_rq: e={e:?}")))?;

        recv.await.map_err(|e| {
            debug!("Arbiter::send_acquire_rq: e={e:?}, requester_id={requester_id:?}");
            MyError::SomethingElse(format!("e={e:?}"))
        })?
    }

    pub async fn shutdown(&mut self) {
        debug!("Arbiter::shutdown: self.granted_counts_len={:?}", self.granted_counts_len);

        self.shutdown.store(true, Ordering::Release);
        self.notify.notify_waiters();
        if let Some(tx) = self.tx.take() {
            drop(tx);
        }
        if let Some(handle) = self.worker.take() {
            let _ = handle.await;
        }
    }

    fn spawn(
        sem: Arc<Semaphore>,
        notify: Arc<Notify>,
        shutdown: Arc<AtomicBool>,
        granted_counts_len: Arc<AtomicUsize>,
        mut rx: mpsc::Receiver<Req>,
        device_idx: usize,
    ) -> JoinHandle<()> {
        tokio::spawn(async move {
            debug!("Arbiter::spawn: device_idx={device_idx}");
            let mut q: VecDeque<Req> = VecDeque::new();

            // Track repeated OutOfPermits per requester_id to trigger escalation.
            let mut fail_counts: HashMap<Arc<dyn CachableIdTr>, u32> = HashMap::new();
            // Track permits granted per requester until dropped.
            let mut granted_counts: HashMap<Arc<dyn CachableIdTr>, usize> = HashMap::new();
            // When set, only this requester_id may receive permits; others get OutOfPermits immediately.
            let mut super_id: Option<Arc<dyn CachableIdTr>> = None;
            const SUPERTASK_FAIL_THRESHOLD: u32 = 3;
            let capacity: u32 = sem.available_permits() as u32;
            let (drop_tx, mut drop_rx) = mpsc::unbounded_channel::<Arc<dyn CachableIdTr>>();
            loop {
                if shutdown.load(Ordering::Acquire) {
                    debug!("Arbiter::spawn: shutting down device_idx={device_idx}");

                    while let Some(Req { reply, .. }) = q.pop_front() {
                        let _ =
                            reply.send(Err(MyError::SomethingElse("arbiter closed".into()).into()));
                    }
                    break;
                }

                // Granting policy:
                // - If a supertask context is set, reject others and only try to grant requested capacity to it.
                // - Otherwise, strict HOL grant if fits; on sustained contention, return OutOfPermits
                //   (increment fail count) to trigger drops/degradation. After threshold, promote to supertask.

                match super_id.take() {
                    Some(sc) => {
                        debug!("Arbiter::spawn: device_idx={device_idx} sc={sc:?}");

                        while let Some(rq) = q.pop_front() {
                            if rq.requester_id.as_ref().eq(sc.as_ref()) {
                                match Arc::clone(&sem).try_acquire_many_owned(rq.needed as u32) {
                                    Ok(inner) => {

                                        granted_counts.insert(rq.requester_id.clone(), inner.num_permits());
                                        granted_counts_len.store(granted_counts.len(), Ordering::Release);
                                        let perm = RunnerPerm::with_drop_signal(
                                            inner,
                                            DropSignal::new(
                                                Arc::clone(&notify),
                                                drop_tx.clone(),
                                                Arc::clone(&rq.requester_id),
                                            ),
                                        );
                                        debug!("Arbiter::spawn: sending super Ok reply to {:?}", rq.requester_id);
                                        let _ = rq.reply.send(Ok(perm));
                                        fail_counts.remove(&rq.requester_id);
                                        super_id = None;
                                    }
                                    Err(e) => {
                                        super_id = Some(Arc::clone(&sc));
                                        debug!("Arbiter::spawn: e={e:?}");
                                        info!("Not enough permits yet → wait for notify. device_idx={device_idx} super_id={super_id:?} q={q:?} granted_counts={granted_counts:?} capacity={capacity} sem.available_permits={}", sem.available_permits());
                                        // Not enough permits yet → wait for notify
                                    }
                                }
                            } else {
                                debug!("Arbiter::spawn: sending super OutOfPermits reply to {:?}", rq.requester_id);
                                match rq.reply.send(Err(MyError::OutOfPermits.into())){
                                    Ok(_) => {
                                        trace!("");
                                    }
                                    Err(e) => {
                                        error!("Arbiter::spawn: e={e:?} when sending response")
                                    }
                                };
                            }
                        }
                    }
                    None => {
                        while super_id.is_none() {
                            let Some(front) = q.front() else {
                                break;
                            };
                            match Arc::clone(&sem).try_acquire_many_owned(front.needed as u32) {
                                Ok(inner) => {
                                    debug!("Arbiter::spawn: device_idx={device_idx} acquired {}", front.needed);

                                    let Req {
                                        needed: _,
                                        reply,
                                        requester_id,
                                    } = q.pop_front().unwrap();

                                    granted_counts.insert(requester_id.clone(), inner.num_permits());

                                    granted_counts_len.store(granted_counts.len(), Ordering::Release);

                                    debug!("Arbiter::spawn: sending Ok reply to {:?}", requester_id);
                                    let _ = reply.send(Ok(RunnerPerm::with_drop_signal(
                                        inner,
                                        DropSignal::new(
                                            Arc::clone(&notify),
                                            drop_tx.clone(),
                                            Arc::clone(&requester_id),
                                        ),
                                    )));
                                    // Success → reset failure count for this context
                                    fail_counts.remove(&requester_id);
                                    // loop to try the new head
                                }
                                Err(TryAcquireError::NoPermits) => {
                                    debug!("Arbiter::spawn: device_idx={device_idx} TryAcquireError::NoPermits: q.len={} front={front:?} granted_counts={granted_counts:?}", q.len());
                                    // if q.len() >= 2 && front.needed > 1 {
                                        // Pop and fail the head to trigger drop/degradation.
                                        let Req { reply, requester_id, .. } = q.pop_front().unwrap();
                                        debug!("Arbiter::spawn: sending OutOfPermits reply to {:?}", requester_id);

                                        let _ = reply.send(Err(MyError::OutOfPermits.into()));
                                        let c = fail_counts.entry(requester_id.clone()).or_insert(0);
                                        *c += 1;
                                        if *c >= SUPERTASK_FAIL_THRESHOLD {
                                            super_id = Some(requester_id);
                                        }
                                    // } else {
                                    //     break;
                                    // }
                                }
                                Err(e) => {
                                    debug!("Arbiter::spawn: device_idx={device_idx} got e={e:?}");

                                    let Req { reply, .. } = q.pop_front().unwrap();
                                    let _ = reply
                                        .send(
                                            Err(MyError::SomethingElse("semaphore closed".into()).into()),
                                        );
                                }
                            }
                        }

                    }

                }


                tokio::select! {
                    Some(req) = rx.recv() => {
                        'done: {
                            debug!("Arbiter::spawn loop: req={req:?}");
                            for mut rq in &mut q {
                                if rq.requester_id.eq(&req.requester_id) {
                                    debug!("Arbiter::spawn loop: rq.requester_id={:?} is in the loop ", rq.requester_id);
                                    rq.needed = req.needed;
                                    req.reply.send(Err(MyError::OutOfPermits.into())).expect("send back failed");
                                    break 'done;
                                }
                            }
                            q.push_back(req);
                        }
                    }
                    Some(dropped) = drop_rx.recv() => {
                        granted_counts.remove(&dropped);
                        granted_counts_len.store(granted_counts.len(), Ordering::Release);
                    }
                    _ = notify.notified() => {
                        // someone dropped a permit → try to serve head again
                    }
                    else => {
                        if q.is_empty() { break; }
                    }
                }

                // If there is a supertask pending, try to serve it with full capacity now.
                if let Some(sc) = super_id.clone() {
                    if let Some(pos) = q.iter().position(|r| r.requester_id.as_ref().eq(sc.as_ref())) {

                        match Arc::clone(&sem).try_acquire_many_owned(capacity) {
                            Ok(inner) => {
                                let Req { reply, requester_id, .. } = q.remove(pos).unwrap();
                                granted_counts.insert(requester_id.clone(), inner.num_permits());
                                granted_counts_len.store(granted_counts.len(), Ordering::Release);
                                let _ = reply.send(Ok(RunnerPerm::with_drop_signal(
                                    inner,
                                    DropSignal::new(
                                        Arc::clone(&notify),
                                        drop_tx.clone(),
                                        Arc::clone(&requester_id),
                                    ),
                                )));
                                fail_counts.remove(&requester_id);
                                super_id = None;
                            }
                            Err(e) => {
                                /* not enough yet → wait for next notify */
                                debug!("Arbiter::spawn: e={e:?}");
                            }
                        }
                    }
                }
            }
            debug!("Arbiter::spawn: after loop");
        })
    }
}

impl Drop for Arbiter {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.notify.notify_waiters();
        self.tx.take();
        if let Some(handle) = self.worker.take() {
            if let Ok(rt) = tokio::runtime::Handle::try_current() {
                rt.spawn(async move {
                    let _ = handle.await;
                });
            } else {
                std::thread::spawn(move || {
                    if let Ok(rt) = tokio::runtime::Runtime::new() {
                        rt.block_on(async move {
                            let _ = handle.await;
                        });
                    } else {
                        handle.abort();
                    }
                });
            }
        }
    }
}

#[async_trait]
impl PermitSource for Arbiter {
    async fn acquire(
        &self,
        n: NumPermits,
        requester_id: Arc<dyn CachableIdTr>,
    ) -> Result<RunnerPerm, ThoosaError> {
        self.send_acquire_rq(n, requester_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::random_range;
    use thoosa_shared::traits::SimpleId;
    use tokio::task::JoinSet;

    #[derive(Debug)]
    struct SomeTask {
        i: usize,
        min_perms: NumPermits,
        step: NumPermits,
        current_perms: NumPermits,
    }
    impl SomeTask {
        pub fn new(i: usize, min_perms: NumPermits, step: NumPermits) -> Self {
            Self {
                i,
                min_perms,
                step,
                current_perms: step.max(1), // start non-zero
            }
        }

        async fn run(&self) -> Result<(), ()> {
            tokio::time::sleep(tokio::time::Duration::from_millis(random_range(10..100))).await;
            if self.current_perms < self.min_perms {
                debug!("task {self:?} failed");
                Err(())
            } else {
                debug!("task {self:?} ok");
                Ok(())
            }
        }
    }

    use log::debug;
    use std::sync::Arc;
    use crate::runner::permits::MAX_PERMITS;

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_fifo_arbiter() -> Result<(), ()> {
        let arb = Arc::new(Arbiter::new(100, 0));

        let tasks = [
            SomeTask::new(0, 80, 20),
            SomeTask::new(1, 100, 20),
            SomeTask::new(2, 80, 10),
        ];

        let mut js: JoinSet<Result<(), ()>> = JoinSet::new();

        for mut task in tasks {
            let arb = Arc::clone(&arb);
            let id = Arc::new(SimpleId::from(format!("task-{}", task.i))) as Arc<dyn CachableIdTr>;
            js.spawn(async move {
                loop {
                    // Clamp to capacity to avoid waiting forever on >capacity.
                    let need = (task.current_perms).min(MAX_PERMITS);

                    // Acquire exactly what we plan to use for THIS attempt.
                    let permit = match arb.send_acquire_rq(need, id.clone()).await {
                        Ok(v) => v,
                        Err(e) => match e.what() {
                            MyError::OutOfPermits => {
                                task.current_perms = 0;
                                continue;
                            }
                            _ => {
                                panic!()
                            }
                        },
                    };
                    debug!("task {} acquired {} permits", task.i, permit.num_permits());

                    // Run while holding the permit.
                    let ok = task.run().await.is_ok();

                    // Release before re-queueing (drop wakes arbiter).
                    drop(permit);

                    if ok {
                        debug!("task {} done", task.i);
                        break Ok(());
                    } else {
                        task.current_perms = (task.current_perms + task.step).min(MAX_PERMITS);
                    }
                }
            });
        }

        let results = js.join_all().await;
        assert!(results.iter().all(|r| r.is_ok()));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn graceful_shutdown_closes_waiters() {
        use tokio::time::{Duration, sleep};

        let arb = Arbiter::new(1, 0);
        // Hold the only permit so next acquire waits
        let permit = arb
            .send_acquire_rq(1, Arc::new(SimpleId::from("a")) as Arc<dyn CachableIdTr>)
            .await
            .unwrap();

        // Enqueue a request that will wait
        let (reply, recv) = oneshot::channel();
        arb.tx
            .as_ref()
            .unwrap()
            .send(Req {
                needed: 1,
                reply,
                requester_id: Arc::new(SimpleId::from("b")) as Arc<dyn CachableIdTr>,
            })
            .await
            .unwrap();

        // Give time for request to reach the worker
        sleep(Duration::from_millis(50)).await;

        // Drop the arbiter while a waiter is pending. Release the permit after
        // a short delay so the worker can observe the shutdown flag first.
        let drop_permit = tokio::spawn(async move {
            sleep(Duration::from_millis(20)).await;
            drop(permit);
        });
        drop(arb);
        let _ = drop_permit.await;

        // The waiting receiver should see an error
        let res = recv.await;
        assert!(matches!(res, Ok(Err(_)) | Err(_)));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn granted_counts_are_cleared_on_drop() {
        use tokio::time::{Duration, sleep};

        let arb = Arbiter::new(2, 0);
        let requester_id = Arc::new(SimpleId::from("granted-tracking")) as Arc<dyn CachableIdTr>;

        let permit = arb
            .send_acquire_rq(1, Arc::clone(&requester_id))
            .await
            .unwrap();
        assert_eq!(arb.granted_counts_len.load(Ordering::SeqCst), 1);

        drop(permit);
        sleep(Duration::from_millis(30)).await;

        assert_eq!(arb.granted_counts_len.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn send_after_close_returns_error() {
        let mut arb = Arbiter::new(1, 0);
        drop(arb.tx.take());
        let res = arb
            .send_acquire_rq(1, Arc::new(SimpleId::from("c")) as Arc<dyn CachableIdTr>)
            .await;
        assert!(matches!(
            res,
            Err(ThoosaError {
                what: MyError::SomethingElse(_),
                ..
            })
        ));
    }
}
