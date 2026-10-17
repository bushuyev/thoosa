use std::collections::HashSet;
use std::fmt::Debug;
use std::mem;
use std::sync::Arc;
use async_trait::async_trait;
use dashmap::DashMap;
use futures_util::{stream, TryStreamExt};
use rand::distr::{Alphanumeric, SampleString};
use log::{debug, error, warn};
use mockall_double::double;
use tokio::sync::RwLockReadGuard;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::collect_futures_eta::{collect_eta_record_completion, CollectEtaInvocationGuard};
use thoosa_shared::ThoError;
use thoosa_shared::traits::{CachableIdTr, Data, WithId};
use thoosa_shared::CopyToDevice;
use crate::runner::common::cachable::{BorrowRef, CachedPurgeOutput, DevicesWrap, ThoDataRef};
#[double]
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::global_ctx::RegisteredCacheHandle;
use crate::runner::purge::{FuturesRunnerMapPurger, WeakMapPurger, PURGE_GPU_CONCURRENCY_LIMIT};
use thoosa_shared::tracked_await::track_await;



///To keep tensors in async functions in between async calls so that data can be dumped to CPU when competing task wins 100 Permits and requests others to be dumped.
#[derive(Debug)]
pub struct TmpCached<D: Data + CopyToDevice + WithId + Output + Debug + Send + Sync + 'static>
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

// Keep an individual queued value registered until its consumer finishes, even
// if the producer is cancelled. Dropping the entry also bounds the cache size.
#[derive(Debug)]
pub struct TmpCachedEntry<D: Data + CopyToDevice + WithId + Output + Debug + Send + Sync + 'static> {
    value: Arc<DevicesWrap<D>>,
    cache: Arc<TmpCached<D>>,
}

impl<D: Data + CopyToDevice + WithId + Output + Debug + Send + Sync + 'static> BorrowRef for TmpCachedEntry<D> {
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
    pub fn registered_name(&self) -> &'static str {
        self.registered_cache.as_ref().expect("TmpCached::registered_name: cache already unregistered").name()
    }

    async fn unregister_cache(&mut self) -> Option<tokio::sync::OwnedRwLockWriteGuard<()>> {
        if let Some(cache) = self.registered_cache.take() {
            GlobalCtx::global().remove_cache(cache.name());
            Some(track_await!(cache.lock_exclusive()).await)
        } else {
            None
        }
    }

    pub fn insert(&self, value: D) {

        self.map.insert(value.id().clone(), Arc::new(DevicesWrap::wrap(value)));

    }

    // Entries need distinct IDs while alive; replacing a queued entry would
    // leave its old tensors invisible to OOM offload.
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
        err.context(format!("can't take from cache for {key:?}"))
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TmpCached::take_all"))]
    pub async fn take_all(self, device:usize) -> anyhow::Result<Vec<D>> {//TODO it can have only one device set when created
        match track_await!(self.take_all_recovering(device)).await {
            Ok(data) => Ok(data),
            Err(error) => Err(error.error.into()),
        }
    }

    //if data has been dumped to cpu and error happens when loading it back
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TmpCached::take_all_recovering"))]
    pub async fn take_all_recovering(mut self, tho_device_idx:usize) -> Result<Vec<D>, TmpCacheTakeAllError<D>> {//TODO it can have only one device set when created
        debug!("TmpCached::take_all_recovering: name={} map.len={}", self.name, self.map.len());
        let _cache_guard = track_await!(self.unregister_cache()).await;

        let mut res = vec![];
        let map = Arc::into_inner(mem::take(&mut self.map)).expect("TmpCached::take_all: map still in use after unregister_cache");
        let mut entries = map.into_iter();

        while let Some((key, val)) = entries.next() {
            let Some(devices_wrap) = Self::take_owned_wrap(&key, val) else {
                continue;
            };

            match track_await!(devices_wrap.load_from_any_device_(tho_device_idx)).await.map(|guard| drop(guard)) {
                Ok(()) => match devices_wrap.take_data(tho_device_idx) {

                    Ok(value) => {
                        debug!("TmpCached::take_all: adding value={value:?}");

                        res.push(value)
                    },
                    Err(err) => {//TODO error is not not recoverable, doesn't make sense to return TmpCacheTakeAllError
                        debug!("TmpCached::take_all_recovering: take_data: err={err:?}");

                        let error = Self::contextualize_take_error(&key, err);
                        Self::recover_remaining_values(&mut res, entries, tho_device_idx);
                        return Err(TmpCacheTakeAllError { data: res, error });
                    }
                }
                Err(err) => {
                    debug!("TmpCached::take_all_recovering: load_from_any_device_: err={err:?}");

                    let error = Self::contextualize_take_error(&key, err);
                    Self::recover_owned_value(&mut res, &key, devices_wrap, tho_device_idx);
                    Self::recover_remaining_values(&mut res, entries, tho_device_idx);
                    debug!("TmpCached::take_all_recovering: res={res:?}");

                    return Err(TmpCacheTakeAllError { data: res, error });
                }
            }
        }

        debug!("TmpCached::take_all_recovering: res={res:?}");

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
{
    async fn has_gpu_data(&self, device_idx: usize) -> bool {
        self.iter().any(|entry| entry.value().is_on_device(device_idx))
    }

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

            if let Err(err) = track_await!(val.dump_to_cpu(device_idx)).await {
                debug!("TmpCached::purge_gpu: key={key:?} dump_to_cpu err={err:?}");
            } else {
                debug!("TmpCached::purge_gpu: key={key:?} dump_to_cpu ok");
            }

        }
        Ok(())
    }

    async fn drop_gpu(&self, device_idx: usize, name: &str) -> Result<(), ThoError> {
        debug!("TmpCached::drop_gpu: device_idx={device_idx}");
        let items: Vec<(<D as WithId>::ID, Arc<DevicesWrap<D>>)> = self
            .iter()
            .map(|entry| (entry.key().clone(), Arc::clone(entry.value())))
            .collect();

        let dump_eta_guard = CollectEtaInvocationGuard::new(&format!("drop_gpu{:?}", name), self.len(), 1);
        let dump_eta_guard_id = dump_eta_guard.id();

        track_await!(stream::iter(items.into_iter().map(Ok::<_, ThoError>)).try_for_each_concurrent(PURGE_GPU_CONCURRENCY_LIMIT, |(key, data)| async move {
            if data.is_on_device(device_idx) {
                data.drop_device(device_idx).await?;
                if !data.has_any_data() {
                    self.remove(&key);
                }
            }
            collect_eta_record_completion(dump_eta_guard_id);
            Ok(())
        })).await?;


        Ok(())
    }

    //it's just dropped
    async fn purge_cpu(&self, _purge_until: u64) -> usize {
        0
    }
}

impl<D> Drop for TmpCached<D>
where
    D: Data + CopyToDevice + WithId + Output + Debug + Send + Sync + 'static,
{
    fn drop(&mut self) {

        if let Some(cache) = self.registered_cache.take() {
            // Guarded entries remove themselves as consumers finish, so an
            // empty cache is also a normal completion path without take_all.
            if !self.map.is_empty() {
                warn!("TmpCached::drop: {}", self.name);//i.e. take_all was not called
            }

            GlobalCtx::global().remove_cache(cache.name());
        }

    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::future::Future;
    use std::pin::Pin;
    use async_trait::async_trait;
    use log::debug;
    use thoosa_cpu_data_derive::CpuData;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_shared::applicables_runner::Output;
    use thoosa_shared::copy_mode::CpuOrGpu;
    use thoosa_shared::{debug_id, CopyToDevice, ThoError, CudaAction};
    // use crate::common::Data;
    use crate::runner::global_ctx::mock_global_ctx;
    use thoosa_shared::traits::{CachableIdTr, Data, DebugId, WithId};
    // use crate::common::Data;
    use crate::runner::global_ctx::GlobalCtx;
use crate::runner::global_ctx::RegisteredCacheHandle;
    use crate::runner::tmp_cache::TmpCached;

    #[tokio::test]
    async fn dummy_data_clone_impl_uses_clone() -> Result<(), ThoError> {
        #[derive(Debug, Clone, CpuData)]
        #[dummy_data(clone)]
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
                        ThoError::Cuda {
                            action: CudaAction::Other,
                            msg: "copy failed".into(),
                        },
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

        GlobalCtx::global().dump_all_to_cpu(&HashMap::new(), 1, "test").await?;

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
