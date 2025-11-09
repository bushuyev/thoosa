use crate::common::cachable::DevicesWrap;
use async_trait::async_trait;
use dashmap::DashMap;
use futures_util::StreamExt;
use futures_util::future::{BoxFuture, Shared};
use futures_util::stream;
use log::debug;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use dashmap::iter::Iter;
use dashmap::mapref::multiple::RefMulti;
use thoosa_shared::CopyToDevice;
use thoosa_shared::ThoosaError;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::traits::{CachableIdTr, Data};

pub(crate) const PURGE_GPU_CONCURRENCY_LIMIT: usize = 10;

#[async_trait]
pub trait FuturesRunnerMapPurger: Debug + Sync + Send {
    async fn purge_gpu(
        &self,
        ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>,
        device_idx: usize,
    ) -> Result<(), ThoosaError>;
    // async fn purge(&self, device_idx: usize);

    async fn purge_cpu(&self, i: u64);
}

#[async_trait]
impl<K, O> FuturesRunnerMapPurger
    for DashMap<K, Shared<BoxFuture<'static, Result<Arc<DevicesWrap<O>>, ThoosaError>>>>
where
    K: CachableIdTr + Clone + Send + Sync + 'static + Eq + Hash,
    O: Data + CopyToDevice + Output + Send + Sync + 'static,
{
    async fn purge_cpu(&self, purge_since: u64) {
        self.retain(|k, v| {
            debug!("FuturesRunnerMapPurger::purge_cpu: key={k:?}");
            if v.peek()
                .and_then(|v| v.as_ref().map(|r| r.is_on_cpu_since(purge_since)).ok())
                .unwrap_or(false)
            {
                debug!("FuturesRunnerMapPurger::purge_cpu: key={k:?} removed");
                false
            } else {
                true
            }
        });
    }

    async fn purge_gpu(
        &self,
        ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>,
        device_idx: usize,
    ) -> Result<(), ThoosaError> {
        debug!("FuturesRunnerMapPurger::purge_gpu: ids_to_keep={ids_to_keep:?}");

        let items: Vec<(
            K,
            Shared<BoxFuture<'static, Result<Arc<DevicesWrap<O>>, ThoosaError>>>,
        )> = self
            .iter()
            .filter(|entry| {
                let keep = ids_to_keep
                    .map(|set| {
                        let kd: &dyn CachableIdTr = entry.key();
                        !set.contains(kd)
                    })
                    .unwrap_or(true);
                keep && entry.value().peek().is_some()
            })
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();

        stream::iter(items)
            //TODO  limit config
            .for_each_concurrent(PURGE_GPU_CONCURRENCY_LIMIT, move |(key, fut)| async move {
                match fut.await {
                    Ok(v) => {
                        if let Err(e) = v.dump_to_cpu(device_idx).await {
                            debug!("purge_gpu failed: key={key:?} err={e:?}");
                        } else {
                            debug!("purge_gpu ok: key={key:?}");
                        }
                    }
                    Err(e) => debug!("purge_gpu: Err key={key:?} err={e:?}"),
                }
            })
            .await;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::global_ctx::{MockGlobalCtx, mock_global_ctx};
    use async_trait::async_trait;
    use dashmap::DashMap;
    use futures_util::future::{BoxFuture, FutureExt};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use thoosa_shared::applicables_runner::Output;
    use thoosa_shared::traits::{DummyCachableId, WithId};

    #[derive(Debug, Clone)]
    struct DummyOutput {
        id: DummyCachableId,
        purge_active: Arc<AtomicUsize>,
        max_purge: Arc<AtomicUsize>,
    }

    #[async_trait(?Send)]
    impl CopyToDevice for DummyOutput {
        async fn copy_to_device(&self, _dst_device: usize) -> Self {
            let cur = self.purge_active.fetch_add(1, Ordering::SeqCst) + 1;
            loop {
                let prev = self.max_purge.load(Ordering::SeqCst);
                if cur > prev {
                    if self
                        .max_purge
                        .compare_exchange(prev, cur, Ordering::SeqCst, Ordering::SeqCst)
                        .is_ok()
                    {
                        break;
                    }
                } else {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
            self.purge_active.fetch_sub(1, Ordering::SeqCst);
            self.clone()
        }
    }

    impl Data for DummyOutput {
        type Metrics = [usize; 1];

        fn size(&self) -> Self::Metrics {
            [1]
        }

        fn device(&self) -> usize {
            1
        }
    }

    impl WithId for DummyOutput {
        type ID = DummyCachableId;
        fn id(&self) -> &Self::ID {
            &self.id
        }
    }

    impl Output for DummyOutput {}

    #[tokio::test(flavor = "multi_thread")]
    async fn tasks_are_throttled_and_complete() -> Result<(), ThoosaError> {
        let ctx = MockGlobalCtx::global_context();
        ctx.expect().returning(|| {
            Box::leak(Box::new(mock_global_ctx::build_default()))
                as &'static MockGlobalCtx
        });

        let map: DashMap<
            DummyCachableId,
            Shared<BoxFuture<'static, Result<Arc<DevicesWrap<DummyOutput>>, ThoosaError>>>,
        > = DashMap::new();

        let futures_active = Arc::new(AtomicUsize::new(0));
        let max_futures = Arc::new(AtomicUsize::new(0));
        let purge_active = Arc::new(AtomicUsize::new(0));
        let max_purge = Arc::new(AtomicUsize::new(0));
        let processed = Arc::new(AtomicUsize::new(0));

        let total = PURGE_GPU_CONCURRENCY_LIMIT * 4;
        for i in 0..total {
            let key: DummyCachableId =
                DummyCachableId::from(format!("key_{i}").leak() as &'static str);

            let out = DummyOutput {
                id: DummyCachableId::from(format!("out_{i}").leak() as &'static str),
                purge_active: purge_active.clone(),
                max_purge: max_purge.clone(),
            };
            let wrapper = Arc::new(DevicesWrap::wrap(out, 0));
            let active_c = futures_active.clone();
            let max_c = max_futures.clone();
            let processed_c = processed.clone();
            let fut = async move {
                let cur = active_c.fetch_add(1, Ordering::SeqCst) + 1;
                loop {
                    let prev = max_c.load(Ordering::SeqCst);
                    if cur > prev {
                        if max_c
                            .compare_exchange(prev, cur, Ordering::SeqCst, Ordering::SeqCst)
                            .is_ok()
                        {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
                active_c.fetch_sub(1, Ordering::SeqCst);
                processed_c.fetch_add(1, Ordering::SeqCst);
                Ok(wrapper)
            }
            .boxed()
            .shared();

            map.insert(key, fut);
        }

        for reff in &map {
            reff.value().clone().await?;
        }

        map.purge_gpu(None, 1).await?;

        assert_eq!(processed.load(Ordering::SeqCst), total);
        // debug!("max_purge={} max_futures={}", max_purge.load(Ordering::SeqCst), max_futures.load(Ordering::SeqCst));
        assert!(max_purge.load(Ordering::SeqCst) > 0);
        assert!(max_futures.load(Ordering::SeqCst) > 0);
        assert!(max_purge.load(Ordering::SeqCst) <= PURGE_GPU_CONCURRENCY_LIMIT);
        assert!(max_futures.load(Ordering::SeqCst) <= PURGE_GPU_CONCURRENCY_LIMIT);
        Ok(())
    }
}
