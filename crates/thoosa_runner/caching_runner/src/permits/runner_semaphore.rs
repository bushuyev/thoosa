use crate::permits::{NumPermits, PermitSource, RunnerPerm};
use async_trait::async_trait;
use log::debug;
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;
use thoosa_shared::traits::CachableIdTr;
use thoosa_shared::{MyError, ThoosaError};
use tokio::sync::Semaphore;
use tokio::time;

impl RunnerSemaphore {
    pub fn new(permits: NumPermits) -> RunnerSemaphore {
        Self {
            sem: Arc::new(Semaphore::new(permits as usize)),
            permits_waited_for: Mutex::new(0),
            timeout: Duration::from_secs(5),
        }
    }

    #[cfg(test)]
    fn new_with_timeout(permits: NumPermits, timeout: Duration) -> RunnerSemaphore {
        Self {
            sem: Arc::new(Semaphore::new(permits as usize)),
            permits_waited_for: Mutex::new(0),
            timeout,
        }
    }

    async fn acquire_many_owned(&self, n: NumPermits) -> Result<RunnerPerm, ThoosaError> {
        {
            let mut waited = self.permits_waited_for.lock();
            *waited += n;
            debug!(
                "RunnerSemaphore::acquire_many_owned: n={n} permits_waited_for={} available_permits={}",
                *waited,
                self.sem.available_permits()
            );
        }

        let permit_fut = Arc::clone(&self.sem).acquire_many_owned(n as u32);
        match time::timeout(self.timeout, permit_fut).await {
            Ok(Ok(v)) => {
                let mut waited = self.permits_waited_for.lock();
                *waited -= n;
                debug!("RunnerSemaphore::acquire_many_owned returning {n} permits");
                Ok(RunnerPerm::new(v))
            }
            Ok(Err(_)) => {
                let mut waited = self.permits_waited_for.lock();
                *waited -= n;
                debug!("RunnerSemaphore::acquire_many_owned semaphore closed");
                Err(MyError::OutOfPermits.into())
            }
            Err(_) => {
                let mut waited = self.permits_waited_for.lock();
                *waited -= n;
                debug!("RunnerSemaphore::acquire_many_owned timeout");
                Err(MyError::OutOfPermits.into())
            }
        }
    }
}

#[async_trait]
impl PermitSource for RunnerSemaphore {
    async fn acquire(
        &self,
        n: NumPermits,
        _out_id: std::sync::Arc<dyn CachableIdTr>,
    ) -> Result<RunnerPerm, ThoosaError> {
        self.acquire_many_owned(n).await
    }
}

#[derive(Debug)]
pub struct RunnerSemaphore {
    permits_waited_for: Mutex<NumPermits>,
    sem: Arc<Semaphore>,
    timeout: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{Duration, sleep};

    #[tokio::test]
    async fn acquire_after_close_returns_error() {
        let sem = RunnerSemaphore::new_with_timeout(1, Duration::from_millis(100));
        sem.sem.close();
        let res = sem.acquire_many_owned(1).await;
        assert!(matches!(
            res,
            Err(ThoosaError {
                what: MyError::OutOfPermits,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn high_contention_only_one_succeeds() {
        use std::sync::Arc;

        let sem = Arc::new(RunnerSemaphore::new_with_timeout(
            1,
            Duration::from_millis(100),
        ));
        let mut handles = vec![];
        for _ in 0..3 {
            let s = Arc::clone(&sem);
            handles.push(tokio::spawn(async move {
                let res = s.acquire_many_owned(1).await;
                if let Ok(_perm) = &res {
                    sleep(Duration::from_millis(200)).await;
                }
                res
            }));
        }

        let mut ok = 0;
        let mut err = 0;
        for h in handles {
            match h.await.unwrap() {
                Ok(_) => ok += 1,
                Err(_) => err += 1,
            }
        }

        assert_eq!(ok, 1);
        assert_eq!(err, 2);
    }
}
