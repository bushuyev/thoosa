use crate::dims::{Dims, ThoBatch};
use futures_util::stream::{BoxStream, FuturesUnordered};
use futures_util::StreamExt;
use std::fmt::Debug;
use std::future::Future;
use std::sync::Arc;
use thoosa_shared::runner::common::RunCtx;
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
}

impl<DX: Dims, DY: Dims, C: RunCtx, B: AsyncDataBatch<DX, DY, C>> AsyncDataBatch<DX, DY, C> for Arc<B> {
    fn train_batch_stream<'a>(&'a self, ctx: C) -> BoxStream<'a, Result<(ThoBatch<DX>, Option<BatchLengths>, ThoBatch<DY>), ThoError>> {
        self.as_ref().train_batch_stream(ctx)
    }

    fn valid_batch_stream<'a>(&'a self, ctx: C) -> BoxStream<'a, Result<(ThoBatch<DX>, Option<BatchLengths>, ThoBatch<DY>), ThoError>> {
        self.as_ref().valid_batch_stream(ctx)
    }
}

// Spawn batch-build futures so they keep making progress even while the caller awaits train_step.
pub fn spawn_prefetched_batch_stream<T, F, I>(futures: I, buffer_limit: usize) -> BoxStream<'static, Result<T, ThoError>>
where
    T: Send + 'static,
    F: Future<Output = Result<T, ThoError>> + Send + 'static,
    I: IntoIterator<Item = F> + Send + 'static,
    I::IntoIter: Send + 'static,
{
    let limit = buffer_limit.max(1);
    let (tx, rx) = tokio::sync::mpsc::channel(limit);

    tokio::spawn(async move {
        let mut iter = futures.into_iter();
        let mut pending = FuturesUnordered::new();
        let mut exhausted = false;

        loop {
            while pending.len() < limit {
                match iter.next() {
                    Some(fut) => pending.push(tokio::spawn(fut)),
                    None => {
                        exhausted = true;
                        break;
                    }
                }
            }

            if pending.is_empty() {
                break;
            }

            let item = match pending.next().await.expect("pending checked above") {
                Ok(item) => item,
                Err(e) => Err(ThoError::new(format!("spawn_prefetched_batch_stream join error: {e}"))),
            };

            if tx.send(item).await.is_err() {
                break;
            }

            if exhausted && pending.is_empty() {
                break;
            }
        }
    });

    futures_util::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|item| (item, rx)) }).boxed()
}

pub mod mock_async_data_batch {
    use crate::batches::AsyncDataBatch;
    use crate::batches::BatchLengths;
    use crate::dims::{Dims, ThoBatch};
    use futures::stream::BoxStream;
    use mockall::mock;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use thoosa_shared::runner::common::RunCtx;
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
