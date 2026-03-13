use crate::dims::{Dims, ThoBatch};
use async_trait::async_trait;
use futures::stream::BoxStream;
use std::fmt::Debug;
use std::sync::Arc;
use thoosa_shared::ThoError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchLengths {
    time: Vec<i64>,
    motif_net_time: Option<Vec<Vec<i64>>>,
}

impl BatchLengths {
    pub fn new(time: Vec<i64>) -> Self {
        Self {
            time,
            motif_net_time: None,
        }
    }

    pub fn with_motif_net_time(time: Vec<i64>, motif_net_time: Vec<Vec<i64>>) -> Result<Self, ThoError> {
        if time.len() != motif_net_time.len() {
            return Err(ThoError::new(format!(
                "BatchLengths::with_motif_net_time batch mismatch: time={} motif_net_time={}",
                time.len(),
                motif_net_time.len()
            )));
        }
        Ok(Self {
            time,
            motif_net_time: Some(motif_net_time),
        })
    }

    pub fn as_slice(&self) -> &[i64] {
        self.time.as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [i64] {
        self.time.as_mut_slice()
    }

    pub fn motif_net_time(&self) -> Option<&[Vec<i64>]> {
        self.motif_net_time.as_deref()
    }
}

impl From<Vec<i64>> for BatchLengths {
    fn from(time: Vec<i64>) -> Self {
        Self::new(time)
    }
}

#[async_trait]
pub trait AsyncDataBatch<DX: Dims, DY: Dims>: Debug + Send + Sync {

    async fn train_batch_stream<'a>(&'a self) -> BoxStream<'a, Result<(ThoBatch<DX>, Option<BatchLengths>, ThoBatch<DY>), ThoError>>;

    async fn valid_batch_stream<'a>(&'a self) -> BoxStream<'a, Result<(ThoBatch<DX>, Option<BatchLengths>, ThoBatch<DY>), ThoError>>;
}

#[async_trait]
impl<DX: Dims, DY: Dims, B:AsyncDataBatch<DX, DY>>  AsyncDataBatch<DX, DY> for Arc<B> {
    async fn train_batch_stream<'a>(&'a self) -> BoxStream<'a, Result<(ThoBatch<DX>, Option<BatchLengths>, ThoBatch<DY>), ThoError>> {
        self.as_ref().train_batch_stream().await
    }

    async fn valid_batch_stream<'a>(&'a self) -> BoxStream<'a, Result<(ThoBatch<DX>, Option<BatchLengths>, ThoBatch<DY>), ThoError>> {
        self.as_ref().valid_batch_stream().await
    }
}

pub mod mock_async_data_batch {
    use crate::batches::AsyncDataBatch;
    use crate::batches::BatchLengths;
    use crate::dims::{Dims, ThoBatch};
    use async_trait::async_trait;
    use futures::stream::BoxStream;
    use mockall::mock;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use thoosa_shared::ThoError;

    mock!{

        #[derive(Debug)]
        pub TestBatch<DX: Dims, DY: Dims> {

        }

        #[async_trait]
        impl<DX: Dims, DY: Dims> AsyncDataBatch<DX, DY> for TestBatch<DX, DY>{
            async fn train_batch_stream<'a>(&'a self)-> BoxStream<'a, Result<(ThoBatch<DX>, Option<BatchLengths>,  ThoBatch<DY>), ThoError >>;
            async fn valid_batch_stream<'a>(&'a self)-> BoxStream<'a, Result<(ThoBatch<DX>, Option<BatchLengths>, ThoBatch<DY>), ThoError >>;
        }

    }

    pub struct MockAsyncDataBatchBuilder<DX: Dims, DY: Dims>  {
        mock: MockTestBatch<DX, DY>
    }

    impl<DX: Dims, DY: Dims> MockAsyncDataBatchBuilder<DX, DY> {
        pub fn with_train_stream_calls<F>(mut self, calls: Arc<AtomicUsize>, stream_fn: F) -> Self
        where
            F: Fn() -> BoxStream<'static, Result<(ThoBatch<DX>, Option<BatchLengths>, ThoBatch<DY>), ThoError>>
                + Send
                + Sync
                + 'static,
        {
            let stream_fn = Arc::new(stream_fn);
            self.mock.expect_train_batch_stream().returning(move || {
                let calls = Arc::clone(&calls);
                let stream_fn = Arc::clone(&stream_fn);
                calls.fetch_add(1, Ordering::SeqCst);
                (stream_fn)()
            });
            self
        }

        pub fn with_valid_stream_calls<F>(mut self, calls: Arc<AtomicUsize>, stream_fn: F) -> Self
        where
            F: Fn() -> BoxStream<'static, Result<(ThoBatch<DX>, Option<BatchLengths>, ThoBatch<DY>), ThoError>>
                + Send
                + Sync
                + 'static,
        {
            let stream_fn = Arc::new(stream_fn);
            self.mock.expect_valid_batch_stream().returning(move || {
                let calls = Arc::clone(&calls);
                let stream_fn = Arc::clone(&stream_fn);
                calls.fetch_add(1, Ordering::SeqCst);
                (stream_fn)()
            });
            self
        }

        pub fn finish(self) -> MockTestBatch<DX, DY>{
            self.mock
        }
    }

    pub fn build<DX: Dims, DY: Dims>()->MockAsyncDataBatchBuilder<DX, DY> {
        MockAsyncDataBatchBuilder {
            mock: Default::default(),
        }
    }
}
