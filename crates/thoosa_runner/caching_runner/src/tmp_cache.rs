use std::borrow::Borrow;
use std::collections::HashSet;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem;
use std::sync::Arc;
use async_trait::async_trait;
use dashmap::DashMap;
use futures_util::future::{BoxFuture, Shared};
use futures_util::FutureExt;
use rand::distr::{Alphanumeric, SampleString};
use log::debug;
use tokio::io::AsyncWriteExt;
use tokio::runtime::Handle;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{CachableIdTr, Data, SimpleId, WithId};
use thoosa_shared::CopyToDevice;
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
        DevicesWrap<D>,
    >>

}

impl<D: Data + WithId + Output + Debug + Send + Sync + 'static> TmpCached<D> {
    pub fn insert(&self, value: D) {
        let cache_area = self.cache_area;
        self.map.insert(
            value.id().clone(),
            // async move {
            //     Ok(Arc::new(DevicesWrap::wrap(value, cache_area)))
            // }.boxed().shared()
            DevicesWrap::wrap(value, cache_area)
        );

    }

    pub async fn take_all(mut self, device:usize) -> Vec<D> {
        let mut res = vec![];
        for (key, val) in mem::take(&mut self.map).into_iter() {
            res.push(

                val.take_data(device).await.unwrap()

            );
        }

        res
    }
}

// impl <D:Data + WithId+ Output+ Debug+ Send + Sync + 'static, I: for <'a> Iterator<Item=&'a D>> Borrow<I> for TmpCached<D> {
//     fn borrow(&self) -> &D {
//         // self.map.iter()
//         todo!()
//     }
// }

impl <D:Data + WithId+ Output+ Debug+ Send + Sync + 'static> TmpCached<D> {

    pub async fn new(name: &'static str, cache_area: u64) -> Self {

        let name = format!("{}_{}", name, Alphanumeric.sample_string(&mut rand::rng(), 16)).leak();

        let map: Box<DashMap<<D as WithId>::ID, DevicesWrap<D>>> = Box::new(Default::default());

        let ptr_ref: *const DashMap<<D as WithId>::ID, DevicesWrap<D>> = std::ptr::from_ref(&*map);

        let map_ref: &'static DashMap<<D as WithId>::ID, DevicesWrap<D>, > = unsafe {
            &*(ptr_ref as *const DashMap<<D as WithId>::ID, DevicesWrap<D> >)
        };

        GlobalCtx::global().register_cache((name, map_ref as &'static dyn FuturesRunnerMapPurger,)).await;

        Self {
            name,
            map,
            cache_area
        }

    }
}

#[async_trait]
impl <D> FuturesRunnerMapPurger for DashMap<<D as thoosa_shared::traits::WithId>::ID, DevicesWrap<D>>
where
    D: Data + WithId + Output + CopyToDevice + Debug + Send + Sync + 'static,
{
    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize) -> Result<(), ThoosaError> {
        debug!("TmpCached::purge_gpu: ids_to_keep={ids_to_keep:?} device_idx={device_idx}");

        let keys: Vec<<D as WithId>::ID> = self
            .iter()
            .filter(|entry| {
                ids_to_keep
                    .map(|set| {
                        let key: &dyn CachableIdTr = entry.key();
                        !set.contains(key)
                    })
                    .unwrap_or(true)
            })
            .map(|entry| entry.key().clone())
            .collect();

        for key in keys {
            debug!("TmpCached::purge_gpu: key={key:?}");
            if let Some(value) = self.get(&key) {
                if let Err(err) = value.dump_to_cpu(device_idx).await {
                    debug!("TmpCached::purge_gpu: key={key:?} dump_to_cpu err={err:?}");
                } else {
                    debug!("TmpCached::purge_gpu: key={key:?} dump_to_cpu ok");
                }
            }
        }
        Ok(())
    }

    async fn purge_cpu(&self, i: u64) {
        debug!("TmpCached::purge_cpu: purge_since={i}");
        self.retain(|key, value| {
            let keep = !value.is_on_cpu_since(i);
            if !keep {
                debug!("TmpCached::purge_cpu: key={key:?} removed");
            }
            keep
        });
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
    use thoosa_dummy_data_derive::DummyData;
    use thoosa_shared::{CopyToDevice, ThoosaError};
    use thoosa_shared::applicables_runner::Output;
    use thoosa_shared::traits::{Data, SimpleId, WithId};
    use crate::global_ctx;
    // use crate::common::Data;
    use crate::global_ctx::{mock_global_ctx, GlobalCtx, MockGlobalCtx};
    use crate::tmp_cache::TmpCached;

    #[tokio::test]
    async fn dummy_data_clone_impl_uses_clone() -> Result<(), ThoosaError> {
        #[derive(Debug, Clone, DummyData)]
        #[dummy_data(clone)]
        struct Foo(SimpleId);

        impl WithId for Foo {
            type ID = SimpleId;

            fn id(&self) -> &Self::ID {
                &self.0
            }
        }

        impl Output for Foo {}

        let foo = Foo(SimpleId::from("foo"));
        let cloned = foo.copy_to_device(0).await;

        assert_eq!(foo.0, cloned.0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_tmp_cache() -> Result<(), ThoosaError> {

        let ctx = MockGlobalCtx::global_context();
        ctx.expect().returning(|| {
            Box::leak(Box::new(
                mock_global_ctx::build_default(),
            )) as &'static global_ctx::MockGlobalCtx
        });

        GlobalCtx::init_devices(2);

        #[derive(Debug, Clone, DummyData)]
        #[dummy_data(clone)]
        struct TestData {
            id:SimpleId,
            device:usize
        }


        impl WithId for TestData {
            type ID = SimpleId;

            fn id(&self) -> &Self::ID {
                &self.id
            }
        }
        impl Output for TestData {}

        impl TestData {
            pub fn new(id: SimpleId, device:usize) -> Self {
                Self { id, device}
            }
        }


        let tmp_cache:TmpCached<TestData> = TmpCached::new("tmp", GlobalCtx::global().next_cache_area().expect("next_cache_area failed")).await;


        tmp_cache.insert(TestData::new("foo".into(), 1));


        GlobalCtx::global().dump_all_to_cpu(&HashMap::new(), 1).await?;


        let all = tmp_cache.take_all(1).await;

        println!("{:?}", all);

        assert_eq!(all[0].device, 1);

        Ok(())

    }


}
