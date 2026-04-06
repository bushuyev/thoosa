use std::collections::HashSet;
use std::fmt::Debug;
use std::mem;
use std::sync::Arc;
use argmin::solver::quasinewton::SR1;
use async_trait::async_trait;
use dashmap::DashMap;
use rand::distr::{Alphanumeric, SampleString};
use log::{debug, error, info};
use mockall_double::double;
use parking_lot::RwLock;
use tokio::runtime::Handle;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::{ThoError, LOCK_TIMEOUT};
use thoosa_shared::traits::{CachableIdTr, Data, WithId};
use thoosa_shared::CopyToDevice;
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
    pub name: String,
    registered_name: Option<&'static str>,
    // map: RwLock<Box<DashMap<<D as WithId>::ID, Arc<DevicesWrap<D>>>>>
    map: Box<DashMap<<D as WithId>::ID, Arc<DevicesWrap<D>>>>

}

impl<D: Data + CopyToDevice + WithId + Output + Debug + Send + Sync + 'static> TmpCached<D>
where
    <D as Data>::M: DevicesWrapCopyMode<D>,
{
    pub fn registered_name(&self) -> &'static str {
        self.registered_name.expect("TmpCached::registered_name: cache already unregistered")
    }

    async fn unregister_cache(&mut self) {
        if let Some(name) = self.registered_name.take() {
            GlobalCtx::global().remove_cache(name).await;
        }
    }

    pub fn insert(&self, value: D) {

        self.map.insert(value.id().clone(), Arc::new(DevicesWrap::wrap(value)));

    }

    #[cfg(test)]
    pub fn get(&self, id: &<D as WithId>::ID) -> Option<Arc<DevicesWrap<D>>> {
        // self.map.try_read_for(LOCK_TIMEOUT).expect("TmpCached::get").get(id).map(|reff| Arc::clone(reff.value()))
        self.map.get(id).map(|reff| Arc::clone(reff.value()))
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TmpCached::take_all"))]
    pub async fn take_all(mut self, device:usize) -> Result<Vec<D>, ThoError> {
        debug!("TmpCached::take_all: name={}", self.name);
        let _purge_lock = GlobalCtx::global().purge_lock().await;
        self.unregister_cache().await;

        let mut res = vec![];
        let map = {
            // let mut guard = self.map.try_write_for(LOCK_TIMEOUT).expect("TmpCached::take_all");
            // mem::take(&mut *guard)
            mem::take(&mut self.map)
        };

        for (key, val) in map.into_iter() {

            let count = Arc::strong_count(&val);

            match Arc::into_inner(val) {
                None => {
                    error!("can't take {key:?} from cache: count={count}");
                }
                Some(val) => {

                    res.push(
                        val.take_data(device).await.expect(&format!("can't take from cache for {key:?}"))
                    );
                }
            }
        }

        Ok(res)
    }


    pub async fn new(name: &'static str) -> Self {

        let name = format!("{}_{}", name, Alphanumeric.sample_string(&mut rand::rng(), 16));
        let registered_name = name.clone().leak();

        let map: Box<DashMap<<D as WithId>::ID, Arc<DevicesWrap<D>>>> = Box::new(Default::default());

        let ptr_ref: *const DashMap<<D as WithId>::ID, Arc<DevicesWrap<D>>> = std::ptr::from_ref(&*map);

        let map_ref: &'static DashMap<<D as WithId>::ID, Arc<DevicesWrap<D>>> = unsafe {
            &*(ptr_ref)
        };


        GlobalCtx::global().register_cache((registered_name, map_ref as &'static dyn FuturesRunnerMapPurger,)).await;

        Self {
            name,
            registered_name: Some(registered_name),
            // map: RwLock::new(map),
            map: map,
        }

    }

    // pub async fn purge_gpu(&self, device_idx:usize) -> Result<(), ThoError>{
    //     // self.map.write().purge_gpu(None, device_idx).await
    //     self.map.purge_gpu(None, device_idx).await
    // }
}



///for TmpCache
#[async_trait]
impl <D> FuturesRunnerMapPurger for DashMap<<D as WithId>::ID, Arc<DevicesWrap<D>>>
where
    D: Data + CopyToDevice + WithId + Output + Debug + Send + Sync + 'static,
    <D as Data>::M: DevicesWrapCopyMode<D>,
{
    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, name: &str) -> Result<(), ThoError> {
        debug!("TmpCached::purge_gpu: ids_to_keep={ids_to_keep:?} device_idx={device_idx}");
        if ids_to_keep.is_some() {
            debug!("TmpCached::purge_gpu: skipping required tmp cache name={name}");
            return Ok(());
        }

        let keys_vals: Vec<(<D as WithId>::ID, Arc<DevicesWrap<D>>)> = self
            .iter()
            .filter(|entry| {
                crate::runner::purge::check_ids_to_keep(ids_to_keep, entry)
            })
            .map(|entry| (entry.key().clone(), Arc::clone(entry.value())))
            .collect();

        for (key, val) in keys_vals {
            debug!("TmpCached::purge_gpu: key={key:?}");
            // if let Some(value) = self.get(&key) {
                if let Err(err) = val.dump_to_cpu(device_idx).await {
                    debug!("TmpCached::purge_gpu: key={key:?} dump_to_cpu err={err:?}");
                } else {
                    debug!("TmpCached::purge_gpu: key={key:?} dump_to_cpu ok");
                }
            // }
        }
        Ok(())
    }

    async fn purge_cpu(&self, purge_until: u64) -> usize {
       /* let mut purged_count = 0usize;

        debug!("TmpCached::purge_cpu: purge_since={purge_until}");
        self.retain(|key, value| {
            let keep = !value.is_on_cpu_before(purge_until);
            if !keep {
                purged_count+=1;
                debug!("TmpCached::purge_cpu: key={key:?} removed");
            }
            keep
        });

        purged_count*/
        0
    }
}

impl<D> Drop for TmpCached<D>
where
    D: Data + CopyToDevice + WithId + Output + Debug + Send + Sync + 'static,
    <D as Data>::M: DevicesWrapCopyMode<D>,
{
    fn drop(&mut self) {
        debug!("TmpCached::drop: {}", self.name);
        if let Some(name) = self.registered_name.take() {
            tokio::task::block_in_place(move || {
                Handle::current().block_on(async move {
                    GlobalCtx::global().remove_cache(name).await
                });
            });
        }

    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::future::Future;
    use std::pin::Pin;
    use thoosa_dummy_data_derive::DummyData;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_shared::applicables_runner::Output;
    use thoosa_shared::copy_mode::CpuOrGpu;
    use thoosa_shared::{simple_id, CopyToDevice, ThoError};
    // use crate::common::Data;
    use crate::runner::global_ctx::mock_global_ctx;
    use thoosa_shared::traits::{CachableIdTr, Data, SimpleId, WithId};
    // use crate::common::Data;
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::tmp_cache::TmpCached;

    #[tokio::test]
    async fn dummy_data_clone_impl_uses_clone() -> Result<(), ThoError> {
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

        let foo = Foo(simple_id!("foo"));
        let cloned = foo.clone();

        assert_eq!(foo.0, cloned.0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_tmp_cache() -> Result<(), ThoError> {

        use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
        RealGlobalCtx::init_devices(2);

        unsafe { mock_global_ctx::USE_REAL_CTX = true; }
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

        #[derive(Debug, Clone)]
        struct TestData {
            id:SimpleId,
            device:usize
        }

        impl CopyToDevice for TestData {
            fn copy_to_device<'life0, 'async_trait>(&'life0 self, dst_device: usize) -> Pin<Box<dyn Future<Output=Result<Self, ThoError>> + Send + 'async_trait>>
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

            fn mem_size(&self) -> u64 {
                0
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


        let tmp_cache:TmpCached<TestData> = TmpCached::new("tmp").await;
        let foo_id = simple_id!("foo");

        tmp_cache.insert(TestData::new(foo_id.clone(), 1));

        let required_tmp_cache_ids = HashSet::<&dyn CachableIdTr>::new();
        let required_ids = HashMap::from([(tmp_cache.registered_name(), &required_tmp_cache_ids)]);
        GlobalCtx::global().dump_all_to_cpu(&required_ids, 1, "test").await?;

        let cached = tmp_cache.get(&foo_id).expect("tmp cache entry missing");
        assert!(!cached.is_all_on_cpu().await);
        drop(cached);

        let all = tmp_cache.take_all(1).await?;

        println!("{:?}", all);

        assert_eq!(all[0].device, 1);

        Ok(())

    }


}
