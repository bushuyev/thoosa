use crate::dims::{Dims, ThoBatch};
use futures_util::StreamExt;
use futures_util::stream::BoxStream;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::future::Future;
use std::sync::Arc;
use thoosa_shared::ThoError;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::RunCtx;
use tracing::Instrument;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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

// The stream owns the producer even before its first poll. Dropping a consumer
// cancels source work; channel EOF is not success until the producer has joined.
struct Prefetched<T> {
    rx: tokio::sync::mpsc::Receiver<Result<T, ThoError>>,
    producer: Option<tokio::task::JoinHandle<()>>,
}

impl<T> Drop for Prefetched<T> {
    fn drop(&mut self) {
        if let Some(producer) = &self.producer { producer.abort(); }
    }
}

impl<T> futures::Stream for Prefetched<T> {
    type Item = Result<T, ThoError>;

    fn poll_next(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match this.rx.poll_recv(cx) {
            std::task::Poll::Ready(None) => {}
            other => return other,
        }
        let Some(producer) = this.producer.as_mut() else { return std::task::Poll::Ready(None); };
        match std::pin::Pin::new(producer).poll(cx) {
            std::task::Poll::Pending => std::task::Poll::Pending,
            std::task::Poll::Ready(result) => {
                this.producer.take();
                std::task::Poll::Ready(result.err().map(|error| Err(ThoError::from(error).context("batch producer failed"))))
            }
        }
    }
}

pub fn preload_one<S, T>(source: S) -> BoxStream<'static, Result<T, ThoError>>
where
    S: futures::Stream<Item = Result<T, ThoError>> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let producer = tokio::spawn(async move {
        futures_util::pin_mut!(source);
        loop {
            // Reserve before polling the source: capacity includes the batch being produced.
            let Ok(permit) = track_await!(tx.reserve()).await else { break; };
            match track_await!(source.next()).await {
                Some(item) => { permit.send(item); }
                None => break,
            }
        }
    }.instrument(tracing::info_span!(
        "batches::preload_one",
        kind = "spawn",
        file = file!(),
        line = line!(),
        task = "preload one batch ahead",
    )));
    Prefetched { rx, producer: Some(producer) }.boxed()
}

pub fn prefetch_next_batch_stream<T, F, I>(futures: I) -> BoxStream<'static, Result<T, ThoError>>
where
    T: Send + 'static,
    F: Future<Output = Result<T, ThoError>> + Send + 'static,
    I: IntoIterator<Item = F> + Send + 'static,
    I::IntoIter: Send + 'static,
{
    // Both entry points share capacity, cancellation, and producer-error handling.
    preload_one(futures_util::stream::iter(futures).then(|future| future))
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

    #[tokio::test]
    async fn test_prefetch_reports_producer_panic_after_buffered_items() -> Result<(), ThoError> {
        use super::preload_one;
        for prefetch in [false, true] {
            let futures = (0..2).map(|index| async move {
                if index == 1 { panic!("injected batch failure"); }
                Ok::<_, ThoError>(index)
            });
            let mut stream = if prefetch { prefetch_next_batch_stream(futures) } else { preload_one(futures_util::stream::iter(futures).then(|future| future)) };
            assert_eq!(stream.next().await.expect("buffered item")?, 0);
            let error = stream.next().await.expect("producer failure must not look like EOF").unwrap_err();
            let ThoError::Anyhow(source) = error else { panic!("expected JoinError source") };
            assert!(source.downcast_ref::<tokio::task::JoinError>().is_some_and(|error| error.is_panic()));
            assert!(stream.next().await.is_none());
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_dropping_prefetch_cancels_pending_source_work() -> Result<(), ThoError> {
        use super::preload_one;
        for prefetch in [false, true] {
            let retained = Arc::new(());
            let weak = Arc::downgrade(&retained);
            let started = Arc::new(tokio::sync::Notify::new());
            let notify = Arc::clone(&started);
            let future = async move {
                let _retained = retained;
                notify.notify_one();
                futures_util::future::pending::<()>().await;
                Ok::<_, ThoError>(())
            };
            let stream = if prefetch { prefetch_next_batch_stream([future]) } else { preload_one(futures_util::stream::once(future)) };
            tokio::time::timeout(Duration::from_secs(5), started.notified()).await?;
            // No source item will ever arrive to wake a channel-only cancellation path.
            drop(stream);
            tokio::time::timeout(Duration::from_secs(5), async {
                while weak.upgrade().is_some() { tokio::task::yield_now().await; }
            }).await?;
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_preload_reserves_capacity_before_polling_source() -> Result<(), ThoError> {
        use super::preload_one;
        let started = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&started);
        let source = futures_util::stream::iter(0..3).map(move |item| {
            observed.fetch_add(1, Ordering::SeqCst);
            Ok(item)
        });
        let mut stream = preload_one(source);
        // Let the producer fill its slot without consuming the first item.
        tokio::time::timeout(Duration::from_secs(5), async {
            while started.load(Ordering::SeqCst) == 0 { tokio::task::yield_now().await; }
        }).await?;
        for _ in 0..10 { tokio::task::yield_now().await; }
        assert_eq!(started.load(Ordering::SeqCst), 1);
        for expected in 0..3 { assert_eq!(stream.next().await.expect("batch")?, expected); }
        assert!(stream.next().await.is_none());
        Ok(())
    }

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
    use crate::dims::{Dims, ThoBatch};
    use crate::nets::batches::AsyncDataBatch;
    use crate::nets::batches::BatchLengths;
    use futures::stream::BoxStream;
    use mockall::mock;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use thoosa_shared::ThoError;
    use thoosa_shared::traits::RunCtx;

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
