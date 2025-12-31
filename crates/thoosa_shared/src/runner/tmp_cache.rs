use std::collections::HashSet;
use std::fmt::Debug;
use std::mem;
use std::sync::Arc;
use std::future::Future;
use async_trait::async_trait;
use dashmap::DashMap;
use rand::distr::{Alphanumeric, SampleString};
use log::{debug, error};
use tokio::runtime::Handle;
use futures_util::{stream, StreamExt, TryStreamExt};
use mockall_double::double;
use crate::applicables_runner::Output;
use crate::ThoosaError;
use crate::traits::{CachableIdTr, Data, WithId};
use crate::CopyToDevice;
use crate::runner::common::cachable::{DevicesWrap, DevicesWrapCopyMode};
#[double]
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::purge::FuturesRunnerMapPurger;



///To keep tensors in async functions in between async calls so that data can be dumped to CPU when competing task wins 100 Permits and requests others to be dumped.
#[derive(Debug)]
pub struct TmpCached<D: Data + CopyToDevice + WithId + Output + Debug + Send + Sync + 'static>
where
    <D as Data>::M: DevicesWrapCopyMode<D>,
{
    name: &'static str,
    cache_area:u64,
    map: Box<DashMap<<D as WithId>::ID, Arc<DevicesWrap<D>>>>

}

impl<D: Data + CopyToDevice + WithId + Output + Debug + Send + Sync + 'static> TmpCached<D>
where
    <D as Data>::M: DevicesWrapCopyMode<D>,
{
    pub fn insert(&self, value: D) {
        let cache_area = self.cache_area;
        self.map.insert(
            value.id().clone(),

            Arc::new(DevicesWrap::wrap(value, cache_area))
        );

    }

    pub fn get(&self, id: &<D as WithId>::ID) -> Option<Arc<DevicesWrap<D>>> {
        self.map.get(id).map(|reff| Arc::clone(reff.value()))
    }

    pub async fn take_all(mut self, device:usize) -> Result<Vec<D>, ThoosaError> {
        let mut res = vec![];
        for (key, val) in mem::take(&mut self.map).into_iter() {

            let count = Arc::strong_count(&val);

            match Arc::into_inner(val) {
                None => {
                    error!("can't take {key:?} from cache: count={count}");
                }
                Some(val) => {
                    //TODO check: load_from_any_device_ is called from take_data so load_from_any_device should not be needed here
                    // val.load_from_any_device(device).await?;

                    res.push(
                        val.take_data(device).await.expect(&format!("can't take from cache for {key:?}"))
                    );
                }
            }
        }

        Ok(res)
    }
    pub fn clone_to_arcs_vec(&self) -> Vec<Arc<DevicesWrap<D>>> {
        self
            .map
            .iter()
            .map(|entry| Arc::clone(entry.value()))
            .collect()
    }

    pub async fn for_each<F, R, FR>(self, mut f: F) -> Result<Vec<R>, ThoosaError>
    where
        F: FnMut(<D as WithId>::ID, Arc<DevicesWrap<D>>) -> FR,
        FR: Future<Output = Result<R, ThoosaError>> + Send + 'static,
    {
        let items: Vec<_> = self
            .map
            .iter()
            .map(|entry| (entry.key().clone(), Arc::clone(entry.value())))
            .collect();

        stream::iter(items.into_iter().map(|(key, value)| f(key, value)))
            .buffer_unordered(10)
            .try_collect::<Vec<R>>()
            .await
    }


    pub async fn new(name: &'static str, cache_area: u64) -> Self {

        let name = format!("{}_{}", name, Alphanumeric.sample_string(&mut rand::rng(), 16)).leak();

        let map: Box<DashMap<<D as WithId>::ID, Arc<DevicesWrap<D>>>> = Box::new(Default::default());

        let ptr_ref: *const DashMap<<D as WithId>::ID, Arc<DevicesWrap<D>>> = std::ptr::from_ref(&*map);

        let map_ref: &'static DashMap<<D as WithId>::ID, Arc<DevicesWrap<D>>> = unsafe {
            &*(ptr_ref)
        };

        GlobalCtx::global().register_cache((name, map_ref as &'static dyn FuturesRunnerMapPurger,)).await;

        Self {
            name,
            map,
            cache_area
        }

    }

    pub async fn purge_gpu(&self, device_idx:usize) -> Result<(), ThoosaError>{
        self.map.purge_gpu(None, device_idx).await
    }
}

#[async_trait]
impl <D> FuturesRunnerMapPurger for DashMap<<D as WithId>::ID, Arc<DevicesWrap<D>>>
where
    D: Data + CopyToDevice + WithId + Output + Debug + Send + Sync + 'static,
    <D as Data>::M: DevicesWrapCopyMode<D>,
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

impl<D> Drop for TmpCached<D>
where
    D: Data + CopyToDevice + WithId + Output + Debug + Send + Sync + 'static,
    <D as Data>::M: DevicesWrapCopyMode<D>,
{
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
    use thoosa_dummy_data_derive::DummyData;
    use crate::{CopyToDevice, ThoosaError};
    use crate::applicables_runner::Output;
    use crate::copy_mode::CpuOrGpu;
    use crate::traits::{Data, SimpleId, WithId};
    use crate::runner::global_ctx;
    // use crate::common::Data;
    use crate::runner::global_ctx::{mock_global_ctx, GlobalCtx, MockGlobalCtx};
    use crate::runner::tmp_cache::TmpCached;

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

        impl Output for Foo {  }

        let foo = Foo(SimpleId::from("foo"));
        let cloned = foo.clone();

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

        #[derive(Debug, Clone)]
        struct TestData {
            id:SimpleId,
            device:usize
        }

        impl CopyToDevice for TestData {
            fn copy_to_device<'life0, 'async_trait>(&'life0 self, dst_device: usize) -> Pin<Box<dyn Future<Output=Result<Self, ThoosaError>> + Send + 'async_trait>>
            where
                Self: Sized,
                'life0: 'async_trait,
                Self: 'async_trait
            {
                Box::pin(async move {
                    Ok(TestData {
                        id: self.id.clone(),
                        device: dst_device
                    })
                })
            }
        }

        impl Data for TestData {
            type Metrics = ();
            type M = CpuOrGpu<Self>;

            fn size(&self) -> Self::Metrics {
                ()
            }

            fn device(&self) -> usize {
                self.device
            }
        }

        impl WithId for TestData {
            type ID = SimpleId;

            fn id(&self) -> &Self::ID {
                &self.id
            }
        }
        impl Output for TestData {  }

        impl TestData {
            pub fn new(id: SimpleId, device:usize) -> Self {
                Self { id, device}
            }
        }


        let tmp_cache:TmpCached<TestData> = TmpCached::new("tmp", GlobalCtx::global().next_cache_area().expect("next_cache_area failed")).await;


        tmp_cache.insert(TestData::new("foo".into(), 1));


        GlobalCtx::global().dump_all_to_cpu(&HashMap::new(), 1).await?;


        let all = tmp_cache.take_all(1).await?;

        println!("{:?}", all);

        assert_eq!(all[0].device, 1);

        Ok(())

    }


}
