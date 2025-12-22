use crate::dims::{AnyDims, Dims, ThoBatch};
use crate::utils::{DataBatch, Splittable, SplittableDataBatch};
use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use futures::Stream;
use futures_util::FutureExt;
use parking_lot::RwLock;
use rand::seq::{IndexedRandom, SliceRandom};
use rand::Rng;
use std::async_iter::AsyncIterator;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use std::fmt;
use tch::Tensor;
use thoosa_shared::ThoosaError;


pub struct ThoBatchIter<'a, DX:Dims, DY:Dims> {
    pd:PhantomData<&'a (DX, DY)>,
    next_fn:Box<dyn Fn()-> Box<dyn Stream<Item=(ThoBatch<DX>, ThoBatch<DY>)>>>
}

impl <'a, DX:Dims, DY:Dims> ThoBatchIter<'a, DX, DY> {
    pub fn new(next_fn:Box<dyn Fn()-> Box<dyn Stream<Item=(ThoBatch<DX>, ThoBatch<DY>)>>>) -> Self {
        Self {
            pd:PhantomData,
            next_fn
        }
    }

    fn train_batches(&self) -> Box<dyn Stream<Item=(ThoBatch<DX>, ThoBatch<DY>)>> {
        (self.next_fn)()
    }
}

#[async_trait]
pub trait AsyncDataBatch<DX: Dims, DY: Dims>: Debug + Send + Sync {
    /// returns (inputs, labels)
    async fn train_batch(&self) -> Result<(ThoBatch<DX>, Tensor), ThoosaError>{
        todo!()
    }
    async fn valid_batch(&self) -> Result<(ThoBatch<DX>, Tensor), ThoosaError> {
        todo!()
    }

    async fn train_batch_stream(&self) -> BoxStream<'_, Result<(ThoBatch<DX>, ThoBatch<DY>), ThoosaError>> {
        todo!()
    }

    async fn valid_batch_stream(&self) -> BoxStream<'_, Result<(ThoBatch<DX>, ThoBatch<DY>), ThoosaError>> {
        todo!()
    }
}

pub mod mock_async_data_batch {
    use crate::batches::AsyncDataBatch;
    use crate::dims::Dims;
    use mockall::mock;

    mock!{

        #[derive(Debug)]
        pub TestBatch<DX: Dims, DY: Dims> {

        }

        impl<DX: Dims, DY: Dims> AsyncDataBatch<DX, DY> for TestBatch<DX, DY>{

        }

    }

    pub struct MockAsyncDataBatchBuilder<DX: Dims, DY: Dims>  {
        mock: MockTestBatch<DX, DY>
    }

    impl<DX: Dims, DY: Dims> MockAsyncDataBatchBuilder<DX, DY> {
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

///////////////////////
//batch transform, for the single tensor transform pass transform_fn to the batch impl
pub struct AsyncBatchTransform<DI: Dims, DO: Dims, B, BF>
where
    B: AsyncDataBatch<DI, AnyDims> + Debug,
    BF: FnMut(ThoBatch<DI>) -> ThoBatch<DO> + Sync + Send,
{
    inner: Arc<Box<B>>,
    transform_fn: Arc<RwLock<BF>>,
    _pd_di: PhantomData<DI>,
    _pd_do: PhantomData<DO>,
}

impl<DI: Dims, DO: Dims, B, F> Debug for AsyncBatchTransform<DI, DO, B, F>
where
    B: AsyncDataBatch<DI, AnyDims> + Debug,
    F: FnMut(ThoBatch<DI>) -> ThoBatch<DO> + Sync + Send,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchTransform")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<DI: Dims, DO: Dims, B, F> AsyncBatchTransform<DI, DO, B, F>
where
    B: AsyncDataBatch<DI, AnyDims> + Debug,
    F: FnMut(ThoBatch<DI>) -> ThoBatch<DO> + Sync + Send,
{
    pub fn new(inner: Arc<Box<B>>, transform_fn: F) -> Self {
        Self {
            inner,
            transform_fn: Arc::new(RwLock::new(transform_fn)),
            _pd_di: PhantomData,
            _pd_do: PhantomData,
        }
    }
    pub fn from_rc(inner: Arc<Box<B>>, transform_fn: Arc<RwLock<F>>) -> Self {
        Self {
            inner,
            transform_fn,
            _pd_di: PhantomData,
            _pd_do: PhantomData,
        }
    }
}

#[async_trait]
impl<DI: Dims, DO: Dims, B, F> AsyncDataBatch<DO, AnyDims> for AsyncBatchTransform<DI, DO, B, F>
where
    B: AsyncDataBatch<DI,AnyDims> + Debug,
    F: FnMut(ThoBatch<DI>) -> ThoBatch<DO> + Sync + Send,
{
    async fn train_batch(&self) -> Result<(ThoBatch<DO>, Tensor), ThoosaError> {
        let (inputs, labels) = self.inner.train_batch().await?;
        Ok((self.transform_fn.write()(inputs), labels))
    }

    async fn valid_batch(&self) -> Result<(ThoBatch<DO>, Tensor), ThoosaError> {
        let (inputs, labels) = self.inner.valid_batch().await?;
        Ok((self.transform_fn.write()(inputs), labels))
    }
}

impl<DI: Dims, DO: Dims + Send, B, F> Splittable<1> for AsyncBatchTransform<DI, DO, B, F>
where
    B: SplittableDataBatch<DI> + Debug + AsyncDataBatch<DI, AnyDims>,
    F: FnMut(ThoBatch<DI>) -> ThoBatch<DO> + Sync + Send,
{
    fn split(&self) -> Option<[Self; 1]>
    where
        Self: Sized,
    {
        let Some([split]) = self.inner.split() else {
            panic!("BatchTransform::split() called on empty inner");
        };
        Some([AsyncBatchTransform::from_rc(
            Arc::new(Box::new(split)),
            Arc::clone(&self.transform_fn),
        )])
    }
}