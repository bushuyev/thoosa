use crate::dims::{Dims, ThoBatch};
use crate::utils::{DataBatch, Splittable, SplittableDataBatch};
use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use futures::Stream;
use futures_util::FutureExt;
use rand::seq::{IndexedRandom, SliceRandom};
use rand::Rng;
use std::async_iter::AsyncIterator;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use thoosa_shared::ThoosaError;

#[async_trait]
pub trait AsyncDataBatch<DX: Dims, DY: Dims>: Debug + Send + Sync {

    async fn train_batch_stream<'a>(&'a self) -> BoxStream<'a, Result<(ThoBatch<DX>, ThoBatch<DY>), ThoosaError>>;

    async fn valid_batch_stream<'a>(&'a self) -> BoxStream<'a, Result<(ThoBatch<DX>, ThoBatch<DY>), ThoosaError>>;
}

#[async_trait]
impl<DX: Dims, DY: Dims, B:AsyncDataBatch<DX, DY>>  AsyncDataBatch<DX, DY> for Arc<B> {
    async fn train_batch_stream<'a>(&'a self) -> BoxStream<'a, Result<(ThoBatch<DX>, ThoBatch<DY>), ThoosaError>> {
        self.as_ref().train_batch_stream().await
    }

    async fn valid_batch_stream<'a>(&'a self) -> BoxStream<'a, Result<(ThoBatch<DX>, ThoBatch<DY>), ThoosaError>> {
        self.as_ref().valid_batch_stream().await
    }
}

pub mod mock_async_data_batch {
    use crate::batches::AsyncDataBatch;
    use crate::dims::{Dims, ThoBatch};
    use async_trait::async_trait;
    use futures::stream::{BoxStream};
    use mockall::mock;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use thoosa_shared::ThoosaError;

    mock!{

        #[derive(Debug)]
        pub TestBatch<DX: Dims, DY: Dims> {

        }

        #[async_trait]
        impl<DX: Dims, DY: Dims> AsyncDataBatch<DX, DY> for TestBatch<DX, DY>{
            async fn train_batch_stream<'a>(&'a self)-> BoxStream<'a, Result<(ThoBatch<DX>, ThoBatch<DY>), ThoosaError>>;
            async fn valid_batch_stream<'a>(&'a self)-> BoxStream<'a, Result<(ThoBatch<DX>, ThoBatch<DY>), ThoosaError>>;
        }

    }

    pub struct MockAsyncDataBatchBuilder<DX: Dims, DY: Dims>  {
        mock: MockTestBatch<DX, DY>
    }

    impl<DX: Dims, DY: Dims> MockAsyncDataBatchBuilder<DX, DY> {
        pub fn with_train_stream_calls<F>(mut self, calls: Arc<AtomicUsize>, stream_fn: F) -> Self
        where
            F: Fn() -> BoxStream<'static, Result<(ThoBatch<DX>, ThoBatch<DY>), ThoosaError>>
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
            F: Fn() -> BoxStream<'static, Result<(ThoBatch<DX>, ThoBatch<DY>), ThoosaError>>
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
