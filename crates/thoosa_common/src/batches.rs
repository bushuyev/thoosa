use crate::dims::{Dims, ThoBatch};
use futures_util::stream::BoxStream;
use futures_util::StreamExt;
use std::fmt::Debug;
use std::future::Future;
use std::sync::Arc;
use log::debug;
use crate::runner::common::RunCtx;
use thoosa_shared::ThoError;

#[derive(Clone, Debug, PartialEq, Eq)]
/// Carries the per-sample valid time lengths for a padded batch.
pub struct BatchLengths {
    lengths: Vec<i64>,
}

impl BatchLengths {
    pub fn new(lengths: Vec<i64>) -> Self {
        Self {
            lengths,
        }
    }

    pub fn as_slice(&self) -> &[i64] {
        self.lengths.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [i64] {
        self.lengths.as_mut_slice()
    }
}

impl From<Vec<i64>> for BatchLengths {
    fn from(lengths: Vec<i64>) -> Self {
        Self::new(lengths)
    }
}

pub trait AsyncDataBatch<DX: Dims, DY: Dims, C: RunCtx>: Debug + Send + Sync {
    fn train_batch_stream<'a>(&'a self, ctx: C) -> BoxStream<'a, Result<(ThoBatch<DX>, Option<BatchLengths>, ThoBatch<DY>), ThoError>>;

    fn valid_batch_stream<'a>(&'a self, ctx: C) -> BoxStream<'a, Result<(ThoBatch<DX>, Option<BatchLengths>, ThoBatch<DY>), ThoError>>;

    fn num_batches(&self) -> usize {
        0
    }
}

impl<DX: Dims, DY: Dims, C: RunCtx, B: AsyncDataBatch<DX, DY, C>> AsyncDataBatch<DX, DY, C> for Arc<B> {
    fn train_batch_stream<'a>(&'a self, ctx: C) -> BoxStream<'a, Result<(ThoBatch<DX>, Option<BatchLengths>, ThoBatch<DY>), ThoError>> {
        self.as_ref().train_batch_stream(ctx)
    }

    fn valid_batch_stream<'a>(&'a self, ctx: C) -> BoxStream<'a, Result<(ThoBatch<DX>, Option<BatchLengths>, ThoBatch<DY>), ThoError>> {
        self.as_ref().valid_batch_stream(ctx)
    }

}

pub fn preload_one<S>(source: S) -> BoxStream<'static, S::Item>
where
    S: futures::stream::Stream + Send + 'static,
    S::Item: Send + 'static,
{
    debug!("preload_one");

    let (tx, rx) = tokio::sync::mpsc::channel(1); // exactly one ready batch ahead

    tokio::spawn(async move {
        futures_util::pin_mut!(source);

        while let Some(item) = source.next().await {
            if tx.send(item).await.is_err() {
                break;
            }
        }
    });

    futures::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    })
        .boxed()
}

// Prefetch at most one batch ahead. Reserve channel capacity before starting the next batch
// so we never build more than one unconsumed batch in the background.
pub fn prefetch_next_batch_stream<T, F, I>(futures: I) -> BoxStream<'static, Result<T, ThoError>>
where
    T: Send + 'static,
    F: Future<Output = Result<T, ThoError>> + Send + 'static,
    I: IntoIterator<Item = F> + Send + 'static,
    I::IntoIter: Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel(1);

    tokio::spawn(async move {
        for fut in futures {
            let permit = match tx.reserve().await {
                Ok(permit) => permit,
                Err(_) => break,
            };
            permit.send(fut.await);
        }
    });

    futures_util::stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|item| (item, rx))
    }).boxed()
}

#[cfg(test)]
mod tests {
    use super::prefetch_next_batch_stream;
    use futures_util::StreamExt;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use thoosa_shared::ThoError;
    use tokio::sync::Semaphore;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_prefetch_next_batch_stream_prefetches_only_one_batch_ahead() -> Result<(), ThoError> {
        let started = Arc::new(AtomicUsize::new(0));
        let release_batches = Arc::new(Semaphore::new(0));
        let started_for_futures = Arc::clone(&started);
        let release_batches_for_futures = Arc::clone(&release_batches);

        let futures = (0..3).map(move |i| {
            let started = Arc::clone(&started_for_futures);
            let release_batches = Arc::clone(&release_batches_for_futures);

            async move {
                started.fetch_add(1, Ordering::SeqCst);
                if i > 0 {
                    let permit = release_batches.acquire().await.expect("semaphore closed");
                    permit.forget();
                }
                Ok::<_, ThoError>(i)
            }
        });

        let mut stream = prefetch_next_batch_stream(futures);
        let first = stream.next().await.expect("first batch should be present")?;
        assert_eq!(first, 0);

        tokio::time::timeout(Duration::from_millis(100), async {
            while started.load(Ordering::SeqCst) < 2 {
                tokio::task::yield_now().await;
            }
        }).await.expect("second batch should start in background");
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(started.load(Ordering::SeqCst), 2);

        release_batches.add_permits(2);
        let second = stream.next().await.expect("second batch should be present")?;
        let third = stream.next().await.expect("third batch should be present")?;
        assert_eq!(second, 1);
        assert_eq!(third, 2);

        Ok(())
    }
}

pub mod mock_async_data_batch {
    use crate::batches::AsyncDataBatch;
    use crate::batches::BatchLengths;
    use crate::dims::{Dims, ThoBatch};
    use futures::stream::BoxStream;
    use mockall::mock;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use crate::runner::common::RunCtx;
    use thoosa_shared::ThoError;

    mock! {
        #[derive(Debug)]
        pub TestBatch<DX: Dims, DY: Dims, C: RunCtx> {}

        impl<DX: Dims, DY: Dims, C: RunCtx> AsyncDataBatch<DX, DY, C> for TestBatch<DX, DY, C> {
            fn train_batch_stream<'a>(&'a self, ctx: C) -> BoxStream<'a, Result<(ThoBatch<DX>, Option<BatchLengths>, ThoBatch<DY>), ThoError>>;
            fn valid_batch_stream<'a>(&'a self, ctx: C) -> BoxStream<'a, Result<(ThoBatch<DX>, Option<BatchLengths>, ThoBatch<DY>), ThoError>>;
        }
    }

    pub struct MockAsyncDataBatchBuilder<DX: Dims, DY: Dims, C: RunCtx> {
        mock: MockTestBatch<DX, DY, C>,
    }

    impl<DX: Dims, DY: Dims, C: RunCtx + 'static> MockAsyncDataBatchBuilder<DX, DY, C> {
        pub fn with_train_stream_calls<F>(mut self, calls: Arc<AtomicUsize>, stream_fn: F) -> Self
        where
            F: Fn(C) -> BoxStream<'static, Result<(ThoBatch<DX>, Option<BatchLengths>, ThoBatch<DY>), ThoError>> + Send + Sync + 'static,
        {
            let stream_fn = Arc::new(stream_fn);
            self.mock.expect_train_batch_stream().returning(move |ctx| {
                let calls = Arc::clone(&calls);
                let stream_fn = Arc::clone(&stream_fn);
                calls.fetch_add(1, Ordering::SeqCst);
                (stream_fn)(ctx)
            });
            self
        }

        pub fn with_valid_stream_calls<F>(mut self, calls: Arc<AtomicUsize>, stream_fn: F) -> Self
        where
            F: Fn(C) -> BoxStream<'static, Result<(ThoBatch<DX>, Option<BatchLengths>, ThoBatch<DY>), ThoError>> + Send + Sync + 'static,
        {
            let stream_fn = Arc::new(stream_fn);
            self.mock.expect_valid_batch_stream().returning(move |ctx| {
                let calls = Arc::clone(&calls);
                let stream_fn = Arc::clone(&stream_fn);
                calls.fetch_add(1, Ordering::SeqCst);
                (stream_fn)(ctx)
            });
            self
        }

        pub fn finish(self) -> MockTestBatch<DX, DY, C> {
            self.mock
        }
    }

    pub fn build<DX: Dims, DY: Dims, C: RunCtx>() -> MockAsyncDataBatchBuilder<DX, DY, C> {
        MockAsyncDataBatchBuilder {
            mock: Default::default(),
        }
    }
}
