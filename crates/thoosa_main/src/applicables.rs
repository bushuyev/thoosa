use async_trait::async_trait;
use dashmap::DashMap;
use futures_util::future::{BoxFuture, Shared};
use parking_lot::RwLock;
use std::ops::Deref;
use std::sync::Arc;
use tch::Device;
use thoosa_common::dims::TCD;
use thoosa_shared::applicables_runner::{Applicable2, AppliedId, DataInDevices, FuturesRunner, Input, Output, RunCtx, Runner, ToOrdinal};
use thoosa_shared::resource::Resource;

#[derive(Debug)]
pub struct ThoosaDevice(Device);

impl Deref for ThoosaDevice {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Device> for ThoosaDevice {
    fn from(value: Device) -> Self {
        ThoosaDevice(value)
    }
}

#[derive(Debug)]
pub struct ThoosaRunCtx {
    cache_area: u64,

    loaded: Vec<AppliedId>,
}

impl RunCtx for ThoosaRunCtx {
    fn into_cached_iter_mut(self) -> impl Iterator<Item=Vec<AppliedId>> {
        [
            self.loaded,
        ].into_iter()
    }

    fn cache_area(&self) -> u64 {
        self.cache_area
    }

}


struct Loader {
    device: ThoosaDevice,
    loader_fn: Box<dyn Fn(&Resource, &Device) -> TCD + Sync + Send>,
}

impl Loader {

    fn new<F>(loader_fn:F, device: ThoosaDevice)-> Loader
    where
        F: Fn(&Resource, &Device) -> TCD + Sync + Send + 'static,
    {
        Self {
            device,
            loader_fn:Box::new(loader_fn),
        }
    }

}

#[async_trait]
impl Applicable2<Resource, TCD> for Loader {
    type R = ThoosaRunner;

    async fn apply(self: Arc<Self>, input: Arc<Resource>, runner: Arc<ThoosaRunner>, ctx: Option<Arc<RwLock<ThoosaRunCtx>>>) -> Arc<TCD> {

        Arc::new((self.loader_fn)(&input, &*self.device))
    }

    fn applied_id(&self, prev_id: &AppliedId) -> AppliedId {
        Self::format_0(&prev_id)
    }

    fn name(&self) -> String {
        "".to_string()
    }

    fn run_id(&self) -> String {
        "".to_string()
    }

    fn device(&self) -> &impl ToOrdinal {
        &self.device
    }
}

impl ToOrdinal for ThoosaDevice {
    fn ordinal(&self) -> usize {
        match self.0 {
            Device::Cpu => 0,
            Device::Cuda(d) => d+1,
            _ => {
                unimplemented!()
            }
        }
    }
}


#[derive(Debug)]
struct ThoosaRunner {
    futures_runner:Arc<FuturesRunner>,

    loader_cache_fut: DashMap<AppliedId, DataInDevices<TCD>>,
}

impl Runner for ThoosaRunner {
    type Ctx = ThoosaRunCtx;
}

impl ThoosaRunner {
    pub fn new() -> Self {
        Self {
            futures_runner: Arc::new(FuturesRunner::new(&[10, 2, 2])),

            loader_cache_fut: Default::default(),
        }
    }

    pub fn run_loader(
        self: &Arc<Self>,
        input: &Resource,
        applicable: &Arc<Loader>,
        ctx: Option<Arc<RwLock<ThoosaRunCtx>>>
    ) -> Shared<BoxFuture<'static, Arc<TCD>>> {
        if let Some(ctx) = ctx.as_ref() {
            ctx.write().loaded.push(applicable.applied_id(Input::id(input)));
        }

        self.futures_runner.run_cached_task(
            &self.loader_cache_fut,
            Arc::new(input.clone()),
            applicable,
            self,
            ctx
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::applicables::{Loader, ThoosaRunner};
    use std::sync::Arc;
    use std::time::Instant;
    use futures::executor::block_on;
    use log::debug;
    use tch::Kind::Float;
    use tch::{Device, Tensor};
    use thoosa_common::dims::TCD;
    use thoosa_shared::resource::Resource;


    #[test]
    fn test_loader(){
        let t_0 = Instant::now();

        let runner = Arc::new(ThoosaRunner::new());

        let loader = Arc::new(Loader::new(
            |key, device| {
                debug!("loader_fn: thread={:?}", std::thread::current().name());
                Tensor::rand(&[1000], (Float, *device)).into()
            },
            Device::Cpu.into()
        ));

        let res:Arc<TCD> = block_on(runner.clone().run_loader(
            &Resource::new(0, 1, "aaa".to_string(), 1000),
            &loader,
            None
        ));

        debug!("loader_cache keys:{:?}", runner.loader_cache_fut.iter().map(|v|v.key().clone()).collect::<Vec<String>>());
        debug!("got res={res:?} in {:?}", t_0.elapsed());

        assert_eq!(res.size(), [1000]);
    }
}