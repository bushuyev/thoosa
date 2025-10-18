use std::collections::HashSet;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem;
use std::sync::Arc;
use dashmap::DashMap;
use futures_util::future::{BoxFuture, Shared};
use futures_util::{FutureExt, StreamExt};
use rand::distr::{Alphanumeric, SampleString};
use tokio::io::AsyncWriteExt;
use tokio::runtime::Handle;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{CachableIdTr, Data, SimpleId, WithId};
use crate::common::cachable::{Cached, DevicesWrap};
use crate::common::{OutId};
use crate::common::runnable::Stage;
use crate::global_ctx::GlobalCtx;
use crate::purge::FuturesRunnerMapPurger;


///To keep tensors in async functions in between async calls so that data can be dumped to CPU when competing task wins 100 Permits and requests others to be dumped.
#[derive(Debug)]
pub struct TmpCached<D:Data + WithId  + Output+ Debug + Send + Sync + 'static> {
    name: &'static str,
    cache_area:u64,
    map: Box<DashMap<
        <D as WithId>::ID,
        Shared<BoxFuture<'static, Result<Arc<DevicesWrap<D>>, ThoosaError>>>,
    >>

}

impl<D: Data + WithId + Output + Debug + Send + Sync + 'static> TmpCached<D> {
    pub fn insert(&self, value: D) {
        let cache_area = self.cache_area;
        self.map.insert(
            value.id().clone(),
            async move {
                Ok(Arc::new(DevicesWrap::wrap(value, cache_area)))
            }.boxed().shared()
        );

    }

    pub async fn take_all(mut self, device:usize) -> Vec<D> {
        let mut res = vec![];
        for (key, val) in mem::take(&mut self.map).into_iter() {
            res.push(

                Arc::into_inner(val.await.unwrap()).unwrap().take_data(device).await.unwrap()

            );
        }

        res
    }

}

impl <D:Data + WithId+ Output+ Debug+ Send + Sync + 'static> TmpCached<D> {

    pub async fn new(name: &'static str, cache_area: u64) -> Self {

        let name = format!("{}_{}", name, Alphanumeric.sample_string(&mut rand::rng(), 16)).leak();

        let map: Box<DashMap<<D as WithId>::ID, Shared<BoxFuture<'static, Result<Arc<DevicesWrap<D>>, ThoosaError>>>>> = Box::new(Default::default());

        let ptr_ref: *const DashMap<<D as WithId>::ID, Shared<BoxFuture<'static, Result<Arc<DevicesWrap<D>>, ThoosaError>>>> = std::ptr::from_ref(&*map);

        let map_ref: &'static DashMap<<D as WithId>::ID, Shared<BoxFuture<'static, Result<Arc<DevicesWrap<D>>, ThoosaError>>>, > = unsafe {
            &*(ptr_ref as *const DashMap<<D as WithId>::ID, Shared<BoxFuture<'static, Result<Arc<DevicesWrap<D>>, ThoosaError>>> >)
        };

        GlobalCtx::global().register_cache((name, map_ref as &'static dyn FuturesRunnerMapPurger,)).await;

        Self {
            name,
            map,
            cache_area
        }

    }
}

impl <D:Data + WithId+ Output+ Debug+ Send + Sync + 'static> Drop for TmpCached<D> {
    fn drop(&mut self) {
        tokio::task::block_in_place(move || {
            Handle::current().block_on(async move {
                GlobalCtx::global().remove_cache(self.name).await
            });
        });

    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::pin::Pin;
    use futures_util::FutureExt;
    use thoosa_shared::{CopyToDevice, ThoosaError};
    use thoosa_shared::applicables_runner::Output;
    use thoosa_shared::traits::{Data, SimpleId, WithId};
    use crate::global_ctx;
    // use crate::common::Data;
    use crate::global_ctx::{mock_global_ctx, GlobalCtx, MockGlobalCtx};
    use crate::tmp_cache::TmpCached;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_tmp_cache() -> Result<(), ThoosaError> {

        let ctx = MockGlobalCtx::global_context();
        ctx.expect().returning(|| {
            Box::leak(Box::new(
                mock_global_ctx::build_default(),
            )) as &'static global_ctx::MockGlobalCtx
        });

        GlobalCtx::init_devices(2);

        #[derive(Debug, Clone)]
        struct DummyData {
            id:SimpleId,
            device:usize
        }

        impl CopyToDevice for DummyData {
            fn copy_to_device<'life0, 'async_trait>(&'life0 self, dst_device: usize) -> Pin<Box<dyn Future<Output=Self> + Send + 'async_trait>>
            where
                'life0: 'async_trait,
                Self: 'async_trait
            {
                async move {
                    DummyData {
                        id: self.id.clone(),
                        device: dst_device,
                    }
                }.shared().boxed()
            }
        }

        impl Data for DummyData {
            type Metrics = usize;

            fn size(&self) -> Self::Metrics {
                0
            }

            fn device(&self) -> usize {
                0
            }
        }
        impl WithId for DummyData {
            type ID = SimpleId;

            fn id(&self) -> &Self::ID {
                &self.id
            }
        }
        impl Output for DummyData {}

        impl DummyData {
            pub fn new(id: SimpleId, device:usize) -> Self {
                Self { id, device}
            }
        }


        let tmp_cache:TmpCached<DummyData> = TmpCached::new("tmp", GlobalCtx::global().next_cache_area().expect("next_cache_area failed")).await;


        tmp_cache.insert(DummyData::new("foo".into(), 1));


        GlobalCtx::global().dump_all_to_cpu(&HashMap::new(), 1).await?;


        let all = tmp_cache.take_all(1).await;

        println!("{:?}", all);

        assert_eq!(all[0].device, 1);

        Ok(())

    }


}