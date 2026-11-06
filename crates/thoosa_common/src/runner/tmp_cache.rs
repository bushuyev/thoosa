use std::fmt::Debug;
use std::mem;
use std::sync::Arc;
use dashmap::DashMap;
use rand::distr::{Alphanumeric, SampleString};
use log::{debug, error};
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::ThoError;
use thoosa_shared::traits::{Data, WithId};
use thoosa_shared::CopyToDevice;
use crate::runner::common::cachable::{BorrowRef, DevicesWrap, ThoDataRef};
use thoosa_shared::tracked_await::t;



/// A temporary collection of automatically registered async data handles.
#[derive(Debug)]
pub struct TmpCached<D: Data + CopyToDevice + WithId + Output + Debug + Send + Sync + 'static>
{
    pub name: String,
    map: Arc<DashMap<<D as WithId>::ID, Arc<DevicesWrap<D>>>>,

}

#[derive(Debug)]
pub struct TmpCacheTakeAllError<D> {
    pub data: Vec<D>,
    pub error: ThoError,
}

// Queued entries own managed handles through producer cancellation. Keeping the
// collection here also removes the consumed entry and bounds queue storage.
#[derive(Debug)]
pub struct TmpCachedEntry<D: Data + CopyToDevice + WithId + Output + Debug + Send + Sync + 'static> {
    value: Arc<DevicesWrap<D>>,
    cache: Arc<TmpCached<D>>,
}

impl<D: Data + CopyToDevice + WithId + Output + Debug + Send + Sync + 'static> BorrowRef for TmpCachedEntry<D> {
    fn protect(&self, device: usize) -> thoosa_common::runner::residency::ResidencyLease { self.value.protect(device) }

    type Out<'a> = ThoDataRef<'a, D> where Self: 'a;

    fn borrow_ref<'a>(&'a self, device: usize) -> impl std::future::Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        BorrowRef::borrow_ref(&self.value, device)
    }
}

impl<D: Data + CopyToDevice + WithId + Output + Debug + Send + Sync + 'static> Drop for TmpCachedEntry<D> {
    fn drop(&mut self) {
        self.cache.map.remove(self.value.id());
    }
}

impl<D: Data + CopyToDevice + WithId + Output + Debug + Send + Sync + 'static> TmpCached<D>
{
    pub fn ids(&self) -> Vec<D::ID> { self.map.iter().map(|entry| entry.key().clone()).collect() }

    pub fn protect(&self, device: usize) -> thoosa_common::runner::residency::ResidencyLease {
        thoosa_common::runner::residency::ResidencyLease::combine(self.map.iter().map(|entry| entry.value().protect(device)))
    }

    pub fn insert(&self, value: D) {
        let value = Arc::new(DevicesWrap::wrap(value));
        self.map.insert(value.id().clone(), value);
    }

    // Queued entries need distinct IDs so dropping an old entry cannot remove
    // a newer queue member. Their residency registrations are independent.
    pub fn insert_guarded(self: &Arc<Self>, value: D) -> TmpCachedEntry<D> {
        let value = Arc::new(DevicesWrap::wrap(value));
        assert!(self.map.insert(value.id().clone(), Arc::clone(&value)).is_none(), "duplicate guarded temporary cache entry");
        TmpCachedEntry { value, cache: Arc::clone(self) }
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

    async fn recover_owned_value(res: &mut Vec<D>, key: &<D as WithId>::ID, val: DevicesWrap<D>, device: usize) {
        match val.take_data_prefer_device_or_any(device).await {
            Ok(value) => res.push(value),
            Err(err) => error!("TmpCached::recover_owned_value: can't recover {key:?}: err={err:?}"),
        }
    }

    async fn recover_remaining_values(
        res: &mut Vec<D>,
        entries: impl Iterator<Item = (<D as WithId>::ID, Arc<DevicesWrap<D>>)>,
        device: usize,
    ) {
        for (key, val) in entries {
            if let Some(val) = Self::take_owned_wrap(&key, val) {
                Self::recover_owned_value(res, &key, val, device).await;
            }
        }
    }

    fn contextualize_take_error(key: &<D as WithId>::ID, err: ThoError) -> ThoError {
        err.context(format!("can't take from cache for {key:?}"))
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TmpCached::take_all"))]
    pub async fn take_all(self, device:usize) -> anyhow::Result<Vec<D>> {//TODO it can have only one device set when created
        match t!(self.take_all_recovering(device)).await {
            Ok(data) => Ok(data),
            Err(error) => Err(error.error.into()),
        }
    }

    //if data has been dumped to cpu and error happens when loading it back
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TmpCached::take_all_recovering"))]
    pub async fn take_all_recovering(mut self, tho_device_idx:usize) -> Result<Vec<D>, TmpCacheTakeAllError<D>> {//TODO it can have only one device set when created
        debug!("TmpCached::take_all_recovering: name={} map.len={}", self.name, self.map.len());

        let mut res = vec![];
        let map = Arc::into_inner(mem::take(&mut self.map)).expect("TmpCached::take_all: map still shared while consuming");
        let mut entries = map.into_iter();

        while let Some((key, val)) = entries.next() {
            let Some(devices_wrap) = Self::take_owned_wrap(&key, val) else {
                continue;
            };

            match t!(devices_wrap.load_from_any_device_(tho_device_idx)).await.map(|guard| drop(guard)) {
                Ok(()) => match devices_wrap.take_data(tho_device_idx).await {

                    Ok(value) => {
                        debug!("TmpCached::take_all: adding value={value:?}");

                        res.push(value)
                    },
                    Err(err) => {//TODO error is not not recoverable, doesn't make sense to return TmpCacheTakeAllError
                        debug!("TmpCached::take_all_recovering: take_data: err={err:?}");

                        let error = Self::contextualize_take_error(&key, err);
                        Self::recover_remaining_values(&mut res, entries, tho_device_idx).await;
                        return Err(TmpCacheTakeAllError { data: res, error });
                    }
                }
                Err(err) => {
                    debug!("TmpCached::take_all_recovering: load_from_any_device_: err={err:?}");

                    let error = Self::contextualize_take_error(&key, err);
                    Self::recover_owned_value(&mut res, &key, devices_wrap, tho_device_idx).await;
                    Self::recover_remaining_values(&mut res, entries, tho_device_idx).await;
                    debug!("TmpCached::take_all_recovering: res={res:?}");

                    return Err(TmpCacheTakeAllError { data: res, error });
                }
            }
        }

        debug!("TmpCached::take_all_recovering: res={res:?}");

        Ok(res)
    }


    pub async fn new(name: &str) -> Self {

        Self::register(format!("{}_{}", name, Alphanumeric.sample_string(&mut rand::rng(), 16)))
    }

    // Restored values construct new managed handles; the runner reacquires their
    // leases. The collection name is diagnostic and no longer controls eviction.
    pub fn restore(name: String, values: impl IntoIterator<Item = D>) -> Self {
        let cache = Self::register(name);
        for value in values { cache.insert(value); }
        cache
    }

    fn register(name: String) -> Self {
        let map: Arc<DashMap<<D as WithId>::ID, Arc<DevicesWrap<D>>>> = Arc::new(Default::default());

        Self {
            name,

            map,
        }

    }

}



#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use crate::runner::common::cachable::BorrowRef;
    use std::future::Future;
    use std::pin::Pin;
        use log::debug;
    use thoosa_cpu_data_derive::CpuData;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_shared::applicables_runner::Output;
    use thoosa_shared::{debug_id, CopyToDevice, ThoError, CudaAction};
    // use crate::common::Data;
    use thoosa_shared::traits::{Data, DebugId, WithId};
    // use crate::common::Data;
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::tmp_cache::TmpCached;

    #[tokio::test]
    async fn dummy_data_clone_impl_uses_clone() -> Result<(), ThoError> {
        #[derive(Debug, Clone, CpuData)]
        #[cpu_data(shallow)]
        struct Foo(DebugId);

        impl WithId for Foo {
            type ID = DebugId;

            fn id(&self) -> &Self::ID {
                &self.0
            }
        }

        impl Output for Foo {  }

        let foo = Foo(debug_id!("foo"));
        let cloned = foo.clone();

        assert_eq!(foo.0, cloned.0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_tmp_cache() -> Result<(), ThoError> {

        use std::sync::Arc;
        use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
        RealGlobalCtx::init_devices(2)?;

        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        #[derive(Debug, Clone)]
        struct TestData {
            id: DebugId,
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


            fn size(&self) -> Self::Metrics {
                ()
            }

            fn tho_device_idx(&self) -> usize {
                self.device
            }

            fn mem_size(&self) -> u64 {
                0
            }
        }

        impl WithId for TestData {
            type ID = DebugId;

            fn id(&self) -> &Self::ID {
                &self.id
            }
        }
        impl Output for TestData {  }

        impl TestData {
            pub fn new(id: DebugId, device:usize) -> Self {
                Self { id, device}
            }
        }


        let tmp_cache:TmpCached<TestData> = TmpCached::new("tmp").await;
        let foo_id = debug_id!("foo");

        tmp_cache.insert(TestData::new(foo_id.clone(), 1));

        let lease = tmp_cache.get(&foo_id).unwrap().protect(1);
        let other_id = debug_id!("other");
        tmp_cache.insert(TestData::new(other_id.clone(), 1));
        GlobalCtx::global().dump_all_to_cpu(1, "test").await?;

        let cached = tmp_cache.get(&foo_id).expect("tmp cache entry missing");
        assert!(!cached.is_all_on_cpu().await);
        assert!(tmp_cache.get(&other_id).unwrap().is_all_on_cpu().await);
        drop(cached);

        drop(lease);
        let cache_name = tmp_cache.name.clone();
        let all = tmp_cache.take_all(1).await?;
        assert!(all.iter().all(|data| data.device == 1));

        // Rebuilt inputs get a new lease, independently of their old registration.
        let restored = TmpCached::restore(cache_name, all);
        let _restored_lease = restored.get(&foo_id).unwrap().protect(1);
        GlobalCtx::global().dump_all_to_cpu(1, "restored inputs").await?;
        assert!(!restored.get(&foo_id).unwrap().is_all_on_cpu().await);
        assert!(restored.get(&other_id).unwrap().is_all_on_cpu().await);

        // A queued prepared batch protects only its own entry in its named cache.
        let queued = Arc::new(TmpCached::new("queued inputs").await);
        let entry = queued.insert_guarded(TestData::new(foo_id.clone(), 1));
        queued.insert(TestData::new(other_id.clone(), 1));
        let _entry_lease = entry.protect(1);
        GlobalCtx::global().dump_all_to_cpu(1, "queued input").await?;
        assert!(!entry.value.is_all_on_cpu().await);
        assert!(queued.get(&other_id).unwrap().is_all_on_cpu().await);

        Ok(())

    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_take_all_recovering_preserves_data_on_cuda_reset_error() -> Result<(), ThoError> {
        use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
        RealGlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        #[derive(Debug, Clone)]
        struct TestData {
            id: DebugId,
            device: usize,
        }

        #[async_trait]
        impl CopyToDevice for TestData {
            async fn copy_to_device(&self, tho_dst_device: usize) -> Result<Self, ThoError> where Self:Sized{

                if tho_dst_device == 0 {
                    Ok(TestData {
                        id: self.id.clone(),
                        device: tho_dst_device,
                    })
                } else {
                    Err(ThoError::new_with_cause(
                        "copy failed",
                        // Use the constructor so injected CUDA errors capture their backtrace.
                        thoosa_shared::tho_error::cuda_error(CudaAction::Other, "copy failed".into()),
                    ))
                }

            }
        }

        impl Data for TestData {
            type Metrics = ();


            fn size(&self) -> Self::Metrics {
                ()
            }

            fn tho_device_idx(&self) -> usize {
                self.device
            }

            fn mem_size(&self) -> u64 {
                0
            }
        }

        impl WithId for TestData {
            type ID = DebugId;

            fn id(&self) -> &Self::ID {
                &self.id
            }
        }

        impl Output for TestData {  }

        let tmp_cache: TmpCached<TestData> = TmpCached::new("tmp").await;
        let foo_id = debug_id!("foo");

        tmp_cache.insert(TestData {
            id: foo_id.clone(),
            device: 1,
        });

        GlobalCtx::global().dump_all_to_cpu(1, "test").await?;

        let result = tmp_cache
            .take_all_recovering(1)
            .await;


        debug!("test_take_all_recovering_preserves_data_on_cuda_reset_error: result={result:?}");

        let err = result
            .expect_err("expected CUDA reset error");

        assert_eq!(err.data.len(), 1);
        assert_eq!(err.data[0].id, foo_id);
        assert_eq!(err.data[0].device, 0);//0 - was dumped to CPU

        Ok(())
    }


}
