use async_trait::async_trait;
use std::collections::{HashMap, VecDeque};
use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use dashmap::DashMap;
use log::{debug, warn};
use thoosa_shared::traits::CachableIdTr;
use thoosa_shared::{Cause, ThoError};
use tokio::sync::{Notify, Semaphore, TryAcquireError, mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::timeout;
use crate::comp_id::SynKey;
use crate::runner::permits::{DropSignal, NumPermits, PermitSource, RunnerPerm};

pub struct Arbiter {
    #[allow(dead_code)]
    sem: Arc<Semaphore>, //TODO for statistics
    notify: Arc<Notify>,
    tx: Option<mpsc::Sender<Req>>,
    capacity: NumPermits,
    worker: Option<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,

    // #[cfg(debug_assertions)] TODO
    granted_counts: Arc<DashMap<Arc<dyn CachableIdTr>, usize>>,
    #[allow(dead_code)]
    device_idx: usize
}


#[derive(Debug)]
struct Req {
    needed: NumPermits,
    reply: oneshot::Sender<Result<RunnerPerm, ThoError>>,
    requester_id: Arc<dyn CachableIdTr>,
}

impl Arbiter {
    pub fn new(capacity: NumPermits, device_idx: usize) -> Self {
        debug!("Arbiter::new: capacity={capacity} device_idx={device_idx}");

        let (tx, rx) = mpsc::channel::<Req>(1024);
        let sem = Arc::new(Semaphore::new(capacity as usize));
        let notify = Arc::new(Notify::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        let granted_counts: Arc<DashMap<Arc<dyn CachableIdTr>, usize>> = Arc::new(DashMap::new());

        let worker = Self::spawn(
            Arc::clone(&sem),
            Arc::clone(&notify),
            Arc::clone(&shutdown),
            Arc::clone(&granted_counts),
            rx,
            device_idx,
            capacity
        );
        Self {
            sem,
            notify,
            tx: Some(tx),
            capacity,
            worker: Some(worker),
            shutdown,
            granted_counts,
            device_idx,
        }
    }

    async fn send_acquire_rq(
        &self,
        needed: NumPermits,
        requester_id: Arc<dyn CachableIdTr>,
    ) -> Result<RunnerPerm, ThoError> {
        assert!(needed <= self.capacity);


        let (reply, recv) = oneshot::channel();
        let tx = self
            .tx
            .as_ref()
            .ok_or_else(|| Cause::SomethingElse("acquire_rq 0".to_string()))?;

        tx.send(Req {
            needed,
            reply,
            requester_id: requester_id.clone(),
        })
        .await
        .map_err(|e| Cause::External {
            type_name: std::any::type_name::<mpsc::error::SendError<Req>>(),
            source: Arc::new(e),
        })?;

        recv.await.map_err(|e| {
            warn!("Arbiter::send_acquire_rq: e={e:?}, requester_id={requester_id:?}");

            Cause::External {
                type_name: std::any::type_name::<oneshot::error::RecvError>(),
                source: Arc::new(e),
            }
        })?
    }

    pub async fn shutdown(&self) {
        debug!("Arbiter::shutdown: self.granted_counts={:?}", self.granted_counts);

        self.shutdown.store(true, Ordering::Release);
        self.notify.notify_waiters();
        // if let Some(tx) = self.tx.take() {
        //     drop(tx);
        // }
        // if let Some(handle) = self.worker.take() {
        //     let _ = handle.await;
        // }
    }

    fn spawn(
        sem: Arc<Semaphore>,
        notify: Arc<Notify>,
        shutdown: Arc<AtomicBool>,
        granted_counts: Arc<DashMap<Arc<dyn CachableIdTr>, usize>>,
        mut rx: mpsc::Receiver<Req>,
        device_idx: usize,
        _capacity: NumPermits,
    ) -> JoinHandle<()> {
        tokio::spawn(async move {
            debug!("Arbiter::spawn: device_idx={device_idx}");

            let mut q: VecDeque<Req> = VecDeque::new();

            let mut fail_counts: HashMap<Arc<dyn CachableIdTr>, u32> = HashMap::new();
            // Supertask mode assumes queued tasks are independent; supertask must not depend
            // on other tasks on the same device or it can deadlock by rejecting them.
            let mut super_id: Option<Arc<dyn CachableIdTr>> = None;
            const SUPERTASK_FAIL_THRESHOLD: u32 = 3;

            let (drop_tx, mut drop_rx) = mpsc::unbounded_channel::<(Arc<dyn CachableIdTr>, u8)>();

            let mut rx_closed = false;

            #[inline]
            fn same_id(a: &Arc<dyn CachableIdTr>, b: &Arc<dyn CachableIdTr>) -> bool {
                a.as_ref().eq(b.as_ref())
            }

            fn fail_req(rq: Req, err: ThoError) {
                debug!("Arbiter::fail_req: rq={rq:?}");
                let _ = rq.reply.send(Err(err));
            }

            fn grant_req(
                rq: Req,
                inner: tokio::sync::OwnedSemaphorePermit,
                granted_counts: &DashMap<Arc<dyn CachableIdTr>, usize>,
                notify: &Arc<Notify>,
                drop_tx: &mpsc::UnboundedSender<(Arc<dyn CachableIdTr>, u8)>,
            ) {
                let n = inner.num_permits();

                debug!("Arbiter::grant_req, rq.requester_id={:?} num_permits={n}", rq.requester_id);


                granted_counts
                    .entry(rq.requester_id.clone())
                    .and_modify(|v| *v += n)
                    .or_insert(n);

                let perm = RunnerPerm::with_drop_signal(
                    inner,
                    DropSignal::new(
                        Arc::clone(notify),
                        drop_tx.clone(),
                        Arc::clone(&rq.requester_id),
                    ),
                );

                let _ = rq.reply.send(Ok(perm));
            }

            // Replace queued request for same requester (latest wins). Old waiter gets an error.
            fn upsert_req(q: &mut VecDeque<Req>, req: Req) {
                if let Some(existing) = q.iter_mut().find(|r| same_id(&r.requester_id, &req.requester_id)) {
                    let old_reply = std::mem::replace(&mut existing.reply, req.reply);
                    existing.needed = req.needed;
                    warn!("Arbiter::upsert_req: got another request from same sender: {:?}", req.requester_id);

                    let _ = old_reply.send(Err(
                        Cause::SomethingElse("replaced by newer request".into()).into()//TODO separate cause of this
                    ));
                } else {
                    debug!("Arbiter::upsert_req: req={:?}", req);
                    q.push_back(req);
                }
            }

            fn purge_non_super(q: &mut VecDeque<Req>, sid: &Arc<dyn CachableIdTr>) {
                let mut kept: VecDeque<Req> = VecDeque::with_capacity(q.len());
                while let Some(rq) = q.pop_front() {
                    if same_id(&rq.requester_id, sid) {
                        kept.push_back(rq);
                    } else {
                        let _ = rq.reply.send(Err(Cause::OutOfPermits.into()));
                    }
                }
                *q = kept;
            }

            loop {
                if shutdown.load(Ordering::Acquire) {
                    debug!("Arbiter::spawn: shutting down device_idx={device_idx}");

                    while let Some(rq) = q.pop_front() {
                        let _ = rq.reply.send(Err(
                            Cause::SomethingElse("arbiter closed".into()).into()
                        ));
                    }

                    granted_counts.clear();
                    break;
                }


                // 1) Try to make progress without waiting.
                let mut progressed = false;

                if let Some(sid) = super_id.as_ref() {
                    debug!("Arbiter::spawn, loop: super_id={sid:?}");
                    // Only attempt to serve sid; do NOT pop&lose on failure.
                    if let Some(pos) = q.iter().position(|r| same_id(&r.requester_id, sid)) {
                        let needed = {
                            // borrow ends here
                            let rq = &q[pos];
                            rq.needed
                        };

                        debug!("Arbiter::spawn, loop: super_id={sid:?} needed={needed} sum={}", granted_counts.iter().map(|e|*e.value()).sum::<usize>());

                        match Arc::clone(&sem).try_acquire_many_owned(needed as u32) {
                            Ok(inner) => {
                                let rq = q.remove(pos).expect("pos valid");

                                debug!("Arbiter::spawn, loop: super_id={sid:?} removed");

                                grant_req(
                                    rq,
                                    inner,
                                    &granted_counts,
                                    &notify,
                                    &drop_tx,
                                );
                                fail_counts.remove(sid);
                                super_id = None;
                                progressed = true;
                            }
                            Err(TryAcquireError::NoPermits) => {
                                // wait for notify / updated request / drop
                            }
                            Err(e) => {//means Closed
                                let rq = q.remove(pos).expect("pos valid");
                                debug!("Arbiter::spawn(super): got e={e:?}, super_id={sid:?} removed");
                                fail_req(rq, Cause::SomethingElse("semaphore closed".into()).into());
                                super_id = None;
                                progressed = true;
                            }
                        }
                    } else {
                        warn!("Arbiter::spawn(super): request with super_id={super_id:?} somehow got lost");
                        // no pending request for sid anymore
                        super_id = None;
                        progressed = true;
                    }
                } else {
                    debug!("Arbiter::spawn, loop: no sid granted_counts sum={}", granted_counts.iter().map(|e|*e.value()).sum::<usize>());

                    // Normal mode: repeatedly try to serve from head.
                    while !q.is_empty() {
                        // IMPORTANT: clone what we need, then release immutable borrow before pop_front()
                        let (needed, requester_id) = {
                            let front = q.front().expect("not empty");
                            debug!("Arbiter::spawn, loop: no sid, front={front:?}");
                            (
                                front.needed,
                                front.requester_id.clone(),
                            )
                        };
                        debug!("Arbiter::spawn, loop: no sid, before try_acquire_many_owned");
                        match Arc::clone(&sem).try_acquire_many_owned(needed as u32) {
                            Ok(inner) => {
                                debug!("Arbiter::spawn, loop: no sid, got inner={inner:?}");

                                let rq = q.pop_front().expect("front existed");
                                grant_req(
                                    rq,
                                    inner,
                                    &granted_counts,
                                    &notify,
                                    &drop_tx,
                                );
                                fail_counts.remove(&requester_id);
                                progressed = true;
                                continue;
                            }
                            Err(TryAcquireError::NoPermits) => {


                                // Reject head once to encourage downgrade, but DO NOT go idle: continue loop.
                                let rq = q.pop_front().expect("pop_front failed on NoPermits");
                                debug!("Arbiter::spawn: device_idx={device_idx} got NoPermits");

                                progressed = true;

                                let c = fail_counts.entry(requester_id.clone()).or_insert(0);
                                *c += 1;
                                if *c >= SUPERTASK_FAIL_THRESHOLD {
                                    super_id = Some(requester_id.clone());
                                    purge_non_super(&mut q, &requester_id);
                                    q.push_back(rq);

                                    break;
                                } else {
                                    fail_req(rq, Cause::OutOfPermits.into());
                                }
                                continue;
                            }
                            Err(e) => {
                                let rq = q.pop_front().expect(&format!("pop_front failed on NoPermits e={e:?}"));
                                debug!("Arbiter::spawn: device_idx={device_idx} got e={e:?}");
                                fail_req(rq, Cause::SomethingElse("semaphore closed".into()).into());
                                progressed = true;
                                continue;
                            }
                        }
                    }
                }

                if progressed {
                    continue;
                }

                // 2) Wait for events.
                tokio::select! {
                    Some(dropped) = drop_rx.recv() => {
                        let id = &dropped.0;
                        let n = dropped.1 as usize;

                        debug!("Arbiter::spawn: {id:?} dropped {n} permits");


                        if let Some(mut vref) = granted_counts.get_mut(id) {
                            if *vref >= n { *vref -= n; } else { *vref = 0; }
                            let now = *vref;
                            drop(vref);
                            if now == 0 {
                                granted_counts.remove(id);
                            }
                        } else {
                            warn!("Arbiter::spawn: drop for unknown id={id:?}");
                        }

                    }

                    opt = rx.recv() => {
                        match opt {
                            Some(req) => {
                                debug!("Arbiter::spawn, select, recv: req.requester_id={:?} needed={} super_id={super_id:?}", req.requester_id, req.needed);

                                if let Some(sid) = super_id.as_ref() {

                                    if !same_id(&req.requester_id, sid) {
                                        debug!("Arbiter::spawn, select, recv: sid={sid:?} req.requester_id={:?} sending OutOfPermits error", req.requester_id);

                                        let _ = req.reply.send(Err(Cause::OutOfPermits.into()));
                                    } else {
                                        debug!("Arbiter::spawn, select, recv: sid={sid:?} req.requester_id={:?} sending double request error", req.requester_id);

                                        // let _ = req.reply.send(Err(Cause::SomethingElse(format!("Got double request for super_id={sid:?}")).into()));
                                        //TODO this shold never happen but somehow still does - same super_id is received for the second time although it doesn't appear in the send logs.
                                        upsert_req(&mut q, req);
                                    }

                                } else {
                                    upsert_req(&mut q, req);
                                }
                            }
                            None => {
                                debug!("Arbiter::spawn, select, recv: got None");
                                rx_closed = true
                            },
                        }
                    }

                    _ = notify.notified() => {
                        debug!("Arbiter::spawn, select, notified");
                        // permit dropped -> try again
                    }
                }

                if rx_closed && q.is_empty() && !shutdown.load(Ordering::Acquire) {
                    debug!("Arbiter::spawn: rx closed, no work left -> exiting device_idx={device_idx}");
                    granted_counts.clear();
                    break;
                }
            }

            debug!("Arbiter::spawn: after loop device_idx={device_idx}");
        })
    }


}

impl Debug for Arbiter {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {

        writeln!(f, "Arbiter{{")?;
        writeln!(f, "capacity={}", self.capacity)?;
        writeln!(f, "device_idx={}", self.device_idx)?;
        writeln!(f, "sem.available_permits={}", self.sem.available_permits())?;

        writeln!(f, "  granted_counts: {{")?;

        let mut entries: Vec<(String, usize)> = self.granted_counts
                .iter()
                .map(|e| (e.key().clone(), *e.value()))
                .collect::<Vec<(Arc<dyn CachableIdTr>, usize)>>()
                .drain(..)
                .map(|(k, v)| (format!("{k:?}"), v)).collect();

        entries.sort_by(|a, b| a.0.cmp(&b.0));

        let mut sum = 0;
        for entry in entries {
            sum+= entry.1;
            writeln!(f, "    {:?}: {:?},", entry.0, entry.1)?;
        }

        writeln!(f, "  }}")?;
        writeln!(f, "granted_counts.sum={sum}")?;
        write!(f, "}}")

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
    ) -> Result<RunnerPerm, ThoError> {
        debug!("PermitsDebug: PermitSourcacquire:  n={n} requester_id={requester_id:?} granted_counts sum={}", self.granted_counts.iter().map(|e|*e.value()).sum::<usize>());
        self.send_acquire_rq(n, requester_id).await
    }

    async fn wait_for_permits_hint(&self) {
        //TODO conf timeout
        debug!("PermitsDebug: wait_for_permits_hint");
        let _ = timeout(Duration::from_millis(10000), self.notify.notified()).await;
    }

    //TODO #[cfg(debug_assertions)]
    fn move_permits(&self, from:&SynKey, to:&SynKey) {
        if from.eq(to) {
            return;
        }
        if let Some(mut from_ref) = self.granted_counts.get_mut(&(Arc::new(from.clone()) as Arc<dyn CachableIdTr>)) {
            let from_val = *from_ref;
            *from_ref = 0;

            drop(from_ref);

            let mut to_ref = self.granted_counts.entry(Arc::new(to.clone()) as Arc<dyn CachableIdTr>).or_insert(0);
            *to_ref += from_val;
            drop(to_ref);

            debug!("PermitSource::move_permits:  from={from:?} to={to:?} val={from_val}");
        }
    }

    async fn shutdown(&self) {
        Arbiter::shutdown(self).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::random_range;
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
    use crate::mono_key;
    use crate::simple_id;
    use crate::runner::permits::MAX_PERMITS;


    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_one_by_one() -> Result<(), ThoError> {
        let arb = Arc::new(Arbiter::new(100, 0));
        let mut aaa_permits = vec![];
        aaa_permits.push(arb.send_acquire_rq(10, Arc::new(mono_key!(0, "aaa")) as Arc<dyn CachableIdTr>).await?);

        debug!("aaa_permits={aaa_permits:?}");
        assert_eq!(aaa_permits.len(), 1);

        assert_eq!(aaa_permits[0].num_permits(), 10);

        Ok(())
    }


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
            let id = Arc::new(simple_id!(format!("task-{}", task.i))) as Arc<dyn CachableIdTr>;
            js.spawn(async move {
                loop {
                    // Clamp to capacity to avoid waiting forever on >capacity.
                    let need = (task.current_perms).min(MAX_PERMITS);

                    // Acquire exactly what we plan to use for THIS attempt.
                    let permit = match arb.send_acquire_rq(need, id.clone()).await {
                        Ok(v) => v,
                        Err(e) => match e.what() {
                            Cause::OutOfPermits => {
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
            .send_acquire_rq(1, Arc::new(simple_id!("a")) as Arc<dyn CachableIdTr>)
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
                requester_id: Arc::new(simple_id!("b")) as Arc<dyn CachableIdTr>,
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
        let requester_id = Arc::new(simple_id!("granted-tracking")) as Arc<dyn CachableIdTr>;

        let permit_0 = arb
            .send_acquire_rq(1, Arc::clone(&requester_id))
            .await
            .unwrap();
        assert_eq!(arb.granted_counts.get(&requester_id).unwrap().value(), &1usize);

        let permit_1 = arb
            .send_acquire_rq(1, Arc::clone(&requester_id))
            .await
            .unwrap();
        assert_eq!(arb.granted_counts.get(&requester_id).unwrap().value(), &2usize);

        debug!("arb={arb:?}");

        drop(permit_1);
        sleep(Duration::from_millis(30)).await;

        assert_eq!(arb.granted_counts.get(&requester_id).unwrap().value(), &1);

        drop(permit_0);
        sleep(Duration::from_millis(30)).await;

        assert!(arb.granted_counts.get(&requester_id).is_none());


    }

    #[tokio::test(flavor = "multi_thread")]
    async fn send_after_close_returns_error() {
        let mut arb = Arbiter::new(1, 0);
        drop(arb.tx.take());
        let res = arb
            .send_acquire_rq(1, Arc::new(simple_id!("c")) as Arc<dyn CachableIdTr>)
            .await;
        assert!(matches!(
            res,
            Err(ThoError {
                what: Cause::SomethingElse(_),
                ..
            })
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_task_keeps_permits() -> Result<(), ThoError>{
        struct TaskWithPermits {
            need_permits:u8,//in the real app it's not known how many permits will be needed - depends on the current task and resources available
            acquired_permits:Vec<RunnerPerm>
        }

        impl TaskWithPermits {
            fn do_some_stuff(&self) -> Result<(), ()> {
                if self.need_permits > self.sum_permits() {
                    Err(())
                } else {
                    Ok(())
                }
            }

            fn sum_permits(&self) -> u8 {
                self.acquired_permits.iter().map(|p| p.num_permits()).sum()
            }

            pub fn new(min_permits:u8) -> Self {
                Self { need_permits: min_permits, acquired_permits: vec![]}
            }
        }
        let arb = Arbiter::new(100, 0);

        let mut task_0 = TaskWithPermits::new(80);
        let mut ask_for_permits = 0;
        loop {
            ask_for_permits+=10;
            match arb.send_acquire_rq(ask_for_permits, Arc::new(mono_key!(0, "task_0")) as Arc<dyn CachableIdTr>).await {
                Ok(perms) => {
                    task_0.acquired_permits.push(perms);//keep permits
                    match task_0.do_some_stuff() {
                        Ok(_) => {
                            //mission accomplished - drop task, permits, etc
                            return Ok(())
                        }
                        Err(_e) => {
                            //try to grab more permits in the next cycle
                        }
                    }
                }
                Err(_) => {//assume it's OutOfPermits
                    task_0.acquired_permits.clear(); //let other tasks to run when there's not enough permits for us
                    ask_for_permits = task_0.sum_permits();//but let's start asking in the next cycle from where we've left

                    //TODO how to avoid going into next cycle immediately and wait for some even on arb telling that there is a chance there will be permits for us
                }
            };

        }

    }
}
