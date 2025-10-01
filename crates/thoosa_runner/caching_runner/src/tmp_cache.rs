/*use std::collections::HashSet;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use dashmap::DashMap;
use futures_util::future::{BoxFuture, Shared};
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{CachableIdTr, SimpleId, WithId};
use crate::common::cachable::{Cached, DataCacheWrapper};
use crate::common::{Data, OutId};
use crate::common::runnable::Stage;
use crate::global_ctx::GlobalCtx;
use crate::purge::FuturesRunnerMapPurger;

#[derive(Debug)]
pub struct TmpCached<D:Data + WithId  + Output+ Debug + Send + Sync + 'static> {
    pd:PhantomData<D>,
}

static __cached_map_ptr_for_MotifLoader: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
#[allow(non_upper_case_globals)]
static __cached_register_for_MotifLoader: tokio::sync::OnceCell<()> = tokio::sync::OnceCell::const_new();

impl <D:Data + WithId+ Output+ Debug+ Send + Sync + 'static> TmpCached<D> {


    async fn cache() -> &'static DashMap<
        SimpleId,
        Shared<BoxFuture<'static, Result<Arc<DataCacheWrapper<D>>, ThoosaError>>>,
    >
    where
        Self: Sized {

        let ptr_ref = __cached_map_ptr_for_MotifLoader.get_or_init(|| {
            let map: &'static dashmap::DashMap<SimpleId, Shared<BoxFuture<'static, Result<Arc<D>, ThoosaError>>>> = Box::leak(Box::new(Default::default()));

            map as *const DashMap<SimpleId, Shared<BoxFuture<'static, Result<Arc<D>, ThoosaError>>>, > as usize
        });
        let map: &'static DashMap<SimpleId, Shared<BoxFuture<'static, Result<Arc<DataCacheWrapper<D>>, ThoosaError>>>, > = unsafe {
            &*((*ptr_ref) as *const DashMap<SimpleId, Shared<BoxFuture<'static, Result<Arc<DataCacheWrapper<D>>, ThoosaError>>> >)
        };

        let cache_name: &'static str = "tmp";
        __cached_register_for_MotifLoader.get_or_init(|| async move { GlobalCtx::global().register_cache((cache_name, map as &'static dyn FuturesRunnerMapPurger,)).await; }).await;
        map
    }

    pub fn new() -> Self {
        Self { pd: PhantomData }
    }
}
#[async_trait::async_trait]
impl <D:Data + WithId+ Output+ Debug+ Send + Sync + 'static> FuturesRunnerMapPurger for TmpCached<D> {
    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize) -> Result<(), ThoosaError> {
        todo!()
    }

    async fn purge_cpu(&self, i: u64) {
        todo!()
    }
}*/

#[cfg(test)]
mod tests {
    // use std::pin::Pin;
    // use futures_util::FutureExt;
    // use thoosa_common::tho_tensor::CopyToDevice;
    // use thoosa_shared::applicables_runner::Output;
    // use thoosa_shared::traits::{SimpleId, WithId};
    // use crate::common::Data;
    // use crate::global_ctx::GlobalCtx;
    // use crate::tmp_cache::TmpCached;

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_0(){
        // unsafe {
        //     let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        // }
/*

        GlobalCtx::init_devices(2);

        #[derive(Debug, Clone)]
        struct DummyData {
            id:SimpleId,
        }

        impl CopyToDevice for DummyData {
            fn copy_to_device<'life0, 'async_trait>(&'life0 self, dst_device: usize) -> Pin<Box<dyn Future<Output=Self> + Send + 'async_trait>>
            where
                'life0: 'async_trait,
                Self: 'async_trait
            {
                async {DummyData { id: self.id.clone() }}.shared().boxed()
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


        let tmp_cache:TmpCached<DummyData> = TmpCached::new();

*/

    }


}