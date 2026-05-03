use std::collections::HashSet;
use std::fmt::Debug;
use std::mem;
use std::sync::Arc;
use argmin::solver::quasinewton::SR1;
use async_trait::async_trait;
use dashmap::DashMap;
use rand::distr::{Alphanumeric, SampleString};
use log::{debug, error, info, warn};
use mockall_double::double;
use parking_lot::RwLock;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::ThoError;
use thoosa_shared::traits::{CachableIdTr, Data, WithId};
use thoosa_shared::CopyToDevice;
use crate::runner::common::cachable::{DevicesWrap, DevicesWrapCopyMode};
#[double]
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::global_ctx::RegisteredCacheHandle;
use crate::runner::purge::{FuturesRunnerMapPurger, WeakMapPurger};



///To keep tensors in async functions in between async calls so that data can be dumped to CPU when competing task wins 100 Permits and requests others to be dumped.
#[derive(Debug)]
pub struct TmpCached<D: Data + CopyToDevice + WithId + Output + Debug + Send + Sync + 'static>
where
    <D as Data>::M: DevicesWrapCopyMode<D>,
{
    pub name: String,
    registered_cache: Option<RegisteredCacheHandle>,
    map: Arc<DashMap<<D as WithId>::ID, Arc<DevicesWrap<D>>>>,

}

#[derive(Debug)]
pub struct TmpCacheTakeAllError<D> {
    pub data: Vec<D>,
    pub error: ThoError,
}

impl<D: Data + CopyToDevice + WithId + Output + Debug + Send + Sync + 'static> TmpCached<D>
where
    <D as Data>::M: DevicesWrapCopyMode<D>,
{
    pub fn registered_name(&self) -> &'static str {
        self.registered_cache.as_ref().expect("TmpCached::registered_name: cache already unregistered").name()
    }

    async fn unregister_cache(&mut self) -> Option<tokio::sync::OwnedRwLockWriteGuard<()>> {
        if let Some(cache) = self.registered_cache.take() {
            GlobalCtx::global().remove_cache(cache.name());
            Some(cache.lock_exclusive().await)
        } else {
            None
        }
    }

    pub fn insert(&self, value: D) {

        self.map.insert(value.id().clone(), Arc::new(DevicesWrap::wrap(value)));

    }



    #[cfg(test)]
    pub fn get(&self, id: &<D as WithId>::ID) -> Option<Arc<DevicesWrap<D>>> {
        self.map.get(id).map(|reff| Arc::clone(reff.value()))
    }

    fn take_owned_wrap(key: &<D as WithId>::ID, val: Arc<DevicesWrap<D>>) -> Option<DevicesWrap<D>> {
        let count = Arc::strong_count(&val);

        match Arc::into_inner(val) {
            None => {
                error!("can't take {key:?} from cache: count={count}");
                None
            }
            Some(val) => Some(val),
        }
    }

    fn recover_owned_value(res: &mut Vec<D>, key: &<D as WithId>::ID, val: DevicesWrap<D>, device: usize) {
        match val.take_data_prefer_device_or_any(device) {
            Ok(value) => res.push(value),
            Err(err) => error!("TmpCached::recover_owned_value: can't recover {key:?}: err={err:?}"),
        }
    }

    fn recover_remaining_values(
        res: &mut Vec<D>,
        entries: impl Iterator<Item = (<D as WithId>::ID, Arc<DevicesWrap<D>>)>,
        device: usize,
    ) {
        for (key, val) in entries {
            if let Some(val) = Self::take_owned_wrap(&key, val) {
                Self::recover_owned_value(res, &key, val, device);
            }
        }
    }

    fn contextualize_take_error(key: &<D as WithId>::ID, err: ThoError) -> ThoError {
        ThoError::new_with_cause(format!("can't take from cache for {key:?}: {}", err.msg), err.what().clone())
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TmpCached::take_all"))]
    pub async fn take_all(self, device:usize) -> Result<Vec<D>, ThoError> {//TODO it can have only one device set when created
        self.take_all_recovering(device).await.map_err(|err| err.error)
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TmpCached::take_all_recovering"))]
    pub async fn take_all_recovering(mut self, device:usize) -> Result<Vec<D>, TmpCacheTakeAllError<D>> {//TODO it can have only one device set when created
        debug!("TmpCached::take_all: name={}", self.name);
        let _cache_guard = self.unregister_cache().await;

        let mut res = vec![];
        let map = Arc::into_inner(mem::take(&mut self.map)).expect("TmpCached::take_all: map still in use after unregister_cache");
        let mut entries = map.into_iter();

        while let Some((key, val)) = entries.next() {
            let Some(devices_wrap) = Self::take_owned_wrap(&key, val) else {
                continue;
            };

            match devices_wrap.load_from_any_device_(device).await.map(|guard| drop(guard)) {
                Ok(()) => match devices_wrap.take_data_prefer_device_or_any(device) {
                    Ok(value) => res.push(value),
                    Err(err) => {
                        let error = Self::contextualize_take_error(&key, err);
                        Self::recover_remaining_values(&mut res, entries, device);
                        return Err(TmpCacheTakeAllError { data: res, error });
                    }
                }
                Err(err) => {
                    let error = Self::contextualize_take_error(&key, err);
                    Self::recover_owned_value(&mut res, &key, devices_wrap, device);
                    Self::recover_remaining_values(&mut res, entries, device);
                    return Err(TmpCacheTakeAllError { data: res, error });
                }
            }
        }

        Ok(res)
    }


    pub async fn new(name: &str) -> Self {

        let name = format!("{}_{}", name, Alphanumeric.sample_string(&mut rand::rng(), 16));
        let registered_name = name.clone().leak();

        let map: Arc<DashMap<<D as WithId>::ID, Arc<DevicesWrap<D>>>> = Arc::new(Default::default());
        let registered_cache = GlobalCtx::global()
            .register_cache((registered_name, Arc::new(WeakMapPurger::new(Arc::downgrade(&map))) as Arc<dyn FuturesRunnerMapPurger>));

        Self {
            name,
            registered_cache: Some(registered_cache),

            map,
        }

    }

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

            if let Err(err) = val.dump_to_cpu(device_idx).await {
                debug!("TmpCached::purge_gpu: key={key:?} dump_to_cpu err={err:?}");
            } else {
                debug!("TmpCached::purge_gpu: key={key:?} dump_to_cpu ok");
            }

        }
        Ok(())
    }

    async fn drop_gpu(&self, device_idx: usize, name: &str) -> Result<(), ThoError> {
        debug!("TmpCached::drop_gpu: device_idx={device_idx}");
        let keys_vals: Vec<(<D as WithId>::ID, Arc<DevicesWrap<D>>)> = self
            .iter()
            .map(|entry| (entry.key().clone(), Arc::clone(entry.value())))
            .collect();

        for (key, val) in keys_vals {
            debug!("TmpCached::drop_gpu: key={key:?}");

            if let Err(err) = val.drop_device(device_idx).await {
                debug!("TmpCached::drop_gpu: key={key:?} dump_to_cpu err={err:?}");
            } else {
                debug!("TmpCached::drop_gpu: key={key:?} dump_to_cpu ok");
            }

        }
        Ok(())
    }

    //it's just dropped
    async fn purge_cpu(&self, purge_until: u64) -> usize {
        0
    }
}

impl<D> Drop for TmpCached<D>
where
    D: Data + CopyToDevice + WithId + Output + Debug + Send + Sync + 'static,
    <D as Data>::M: DevicesWrapCopyMode<D>,
{
    fn drop(&mut self) {

        if let Some(cache) = self.registered_cache.take() {
            warn!("TmpCached::drop: {}", self.name);//i.e. take_all was not called

            GlobalCtx::global().remove_cache(cache.name());
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
    use thoosa_shared::{Cause, simple_id, CopyToDevice, ThoError, CudaAction};
    // use crate::common::Data;
    use crate::runner::global_ctx::mock_global_ctx;
    use thoosa_shared::traits::{CachableIdTr, Data, SimpleId, WithId};
    // use crate::common::Data;
    use crate::runner::global_ctx::GlobalCtx;
use crate::runner::global_ctx::RegisteredCacheHandle;
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

        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

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

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_take_all_recovering_preserves_data_on_cuda_reset_error() -> Result<(), ThoError> {
        use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
        RealGlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        #[derive(Debug, Clone)]
        struct TestData {
            id: SimpleId,
            device: usize,
        }

        impl CopyToDevice for TestData {
            fn copy_to_device<'life0, 'async_trait>(&'life0 self, dst_device: usize) -> Pin<Box<dyn Future<Output = Result<Self, ThoError>> + Send + 'async_trait>>
            where
                Self: Sized,
                'life0: 'async_trait,
                Self: 'async_trait
            {
                Box::pin(async move {
                    if self.device == 0 && dst_device == 1 {
                        Err(ThoError::new_with_cause(
                            "copy failed",
                            Cause::Cuda {
                                action: CudaAction::Other,
                                msg: "copy failed".into(),
                            },
                        ))
                    } else {
                        Ok(TestData {
                            id: self.id.clone(),
                            device: dst_device,
                        })
                    }
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

        let tmp_cache: TmpCached<TestData> = TmpCached::new("tmp").await;
        let foo_id = simple_id!("foo");

        tmp_cache.insert(TestData {
            id: foo_id.clone(),
            device: 0,
        });

        let err = tmp_cache
            .take_all_recovering(1)
            .await
            .expect_err("expected CUDA reset error");

        assert_eq!(err.data.len(), 1);
        assert_eq!(err.data[0].id, foo_id);
        assert_eq!(err.data[0].device, 0);

        Ok(())
    }


}
