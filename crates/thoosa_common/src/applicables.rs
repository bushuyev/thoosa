use std::fmt::{Debug, Formatter};
use async_trait::async_trait;
use dashmap::DashMap;
use futures_util::future::{join_all, BoxFuture, Shared};
use parking_lot::RwLock;
use std::ops::Deref;
use std::sync::Arc;
use log::debug;
use tch::{Device, Tensor};
use tch::Kind::Float;
use crate::dims::{TCD, TCDF, TCSD};
use thoosa_shared::applicables_runner::{AppliedId, DataInDevices, FuturesRunner, FuturesRunnerMap, Input, Output, RunCtx, Runner, Statistics, ToOrdinal};
use thoosa_shared::resource::Resource;
use thoosa_shared::run_configs::Applicable;
use crate::find_motifs_run_config::FindMotifsApplicable;

#[derive(Debug, Clone)]
pub struct ThoDevice(Device);

impl Deref for ThoDevice {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Device> for ThoDevice {
    fn from(value: Device) -> Self {
        ThoDevice(value)
    }
}

impl From<ThoDevice> for Device {
    fn from(value: ThoDevice) -> Self {
        value.0
    }
}

impl From<usize> for ThoDevice {
    fn from(value: usize) -> Self {
        match value {
            0 => ThoDevice(Device::Cpu),
            d => ThoDevice(Device::Cuda(d-1)),
        }
    }
}


#[derive(Debug)]
pub struct ThoRunCtx {
    cache_area: u64,

    loaded: Vec<AppliedId>,
    find_motifs: Vec<AppliedId>,
}

impl ThoRunCtx {
    pub fn new(cache_area:u64) -> Self {
        Self {
            cache_area,
            loaded: Vec::new(),
            find_motifs: Vec::new(),
        }
    }
}

impl RunCtx for ThoRunCtx {
    fn into_cached_iter_mut(self) -> impl Iterator<Item=Vec<AppliedId>> {
        [
            self.loaded,
            self.find_motifs,
        ].into_iter()
    }

    fn cache_area(&self) -> u64 {
        self.cache_area
    }

}

pub struct TCDLoader {
    device: ThoDevice,
    loader_fn: Box<dyn Fn(&Resource, &Device) -> TCD + Sync + Send>,
}

impl TCDLoader {

    pub fn new<F>(loader_fn:F, device: ThoDevice) -> TCDLoader
    where
        F: Fn(&Resource, &Device) -> TCD + Sync + Send + 'static,
    {
        Self {
            device,
            loader_fn:Box::new(loader_fn),
        }
    }
}

impl Debug for TCDLoader {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TCDLoader")
            .field("device", &self.device)
            .finish()
    }
}

#[async_trait]
impl Applicable<Resource, TCD, ()> for TCDLoader {
    type R = ThoRunner;

    async fn async_apply(self: Arc<Self>, input: Arc<Resource>, runner: Arc<ThoRunner>, ctx: Option<Arc<RwLock<ThoRunCtx>>>) -> Arc<TCD> {

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

    fn device(&self) -> impl ToOrdinal {
        self.device.clone()
    }
}

impl ToOrdinal for ThoDevice {
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
pub struct ThoRunner {
    futures_runner:Arc<FuturesRunner>,

    loader_cache_fut: DashMap<AppliedId, DataInDevices<TCD>>,
    find_motifs_cache_fut: DashMap<AppliedId, DataInDevices<TCDF>>,

    tcd_loaders:Vec<Arc<TCDLoader>>
}

impl Runner for ThoRunner {
    type Ctx = ThoRunCtx;
}

impl ThoRunner {

    pub fn new<F>(devices_threads: &[usize], loader_fn:F) -> Self where
        F: Fn(&Resource, &Device) -> TCD + Sync + Send + Clone + 'static {

        let mut tcd_loaders = Vec::with_capacity(devices_threads.len());
        tcd_loaders.push(
            Arc::new(TCDLoader::new(loader_fn.clone(), Device::Cpu.into()))
        );

        for i in 0..devices_threads.len()-1 {
            tcd_loaders.push(
                Arc::new(TCDLoader::new(loader_fn.clone(), Device::Cuda(i).into()))
            );
        }

        Self {
            futures_runner: Arc::new(FuturesRunner::new(&devices_threads)),//TODO config
            loader_cache_fut: Default::default(),
            find_motifs_cache_fut: Default::default(),
            tcd_loaders
        }
    }

    pub fn run_task<I, O, A: Applicable<I, O, (), R=Self> + 'static>(
        self: &Arc<Self>,
        input: Arc<I>,
        applicable:&Arc<A>,
    ) -> BoxFuture<'static, Arc<O>>
    where
        O: Output + Send + Sync + 'static,
        I: Input+Send+Sync+'static,
    {
        self.futures_runner.run_task(input, applicable, self)
    }

    pub fn run_loader(
        self: &Arc<Self>,
        input: &Resource,
        on_device: &Device,
        ctx: Option<Arc<RwLock<ThoRunCtx>>>
    ) -> Shared<BoxFuture<'static, Arc<TCD>>> {
        let applicable = &self.tcd_loaders[ThoDevice::from(*on_device).ordinal()];

        // if let Some(ctx) = ctx.as_ref() {
        //     ctx.write().loaded.push(applicable.applied_id(Input::id(input)));
        // }

        self.futures_runner.run_cached_task(
            &self.loader_cache_fut,
            Arc::new(input.clone()),
            applicable,
            self,
            ctx
        )
    }

    pub fn run_find_motifs_applicable(
        self: &Arc<Self>,
        input: &Arc<TCD>,
        applicable: &Arc<FindMotifsApplicable>,
        ctx: Option<Arc<RwLock<ThoRunCtx>>>
    ) -> Shared<BoxFuture<'static, Arc<TCDF>>> {

        // if let Some(ctx) = ctx.as_ref() {
        //     ctx.write().loaded.push(<FindMotifsApplicable as Applicable<TCD, TCDF, ()>>::applied_id(applicable, input.id()));
        // }

        self.futures_runner.run_cached_task(
            &self.find_motifs_cache_fut,
            Arc::clone(&input),
            applicable,
            self,
            ctx
        )
    }

    pub(crate) fn to_futs_iter(&self) -> impl Iterator<Item = &dyn FuturesRunnerMap> {
        [
            &self.loader_cache_fut as &dyn FuturesRunnerMap,
            &self.find_motifs_cache_fut as &dyn FuturesRunnerMap,
        ].into_iter()
    }

    pub async fn purge_all_gpu_caches(self:Arc<Self>, ctx: ThoRunCtx, device:Device){
        Arc::clone(&self.futures_runner).purge_all_gpu_caches(ctx, &ThoDevice::from(device), self.to_futs_iter()).await;
    }

    pub fn next_ctx(&self) -> Arc<RwLock<ThoRunCtx>> {
        Arc::new(RwLock::new(ThoRunCtx::new(self.futures_runner.next_available_cache_area())))
    }

    pub fn futures_runner(self:&Arc<Self>) -> &Arc<FuturesRunner> {
        &self.futures_runner
    }
    
    pub fn loader_keys(&self) -> Vec<String> {
        self.loader_cache_fut.iter().map(|v|v.key().clone()).collect::<Vec<String>>()
    }
    
    pub fn find_motif_keys(&self) -> Vec<String> {
        self.find_motifs_cache_fut.iter().map(|v|v.key().clone()).collect::<Vec<String>>()
    }
}

#[cfg(test)]
mod tests {
    use super::{TCDLoader, ThoRunner};
    use std::sync::Arc;
    use std::time::Instant;
    use futures::executor::block_on;
    use log::debug;
    use tch::Kind::Float;
    use tch::{Device, Tensor};
    use crate::dims::TCD;
    use thoosa_shared::resource::Resource;


    #[test]
    fn test_loader(){
        let t_0 = Instant::now();

        let runner = Arc::new(ThoRunner::new(
            &[10, 2, 2],
              |key, device| {
                  debug!("loader_fn: thread={:?}", std::thread::current().name());
                  Tensor::rand(&[1000], (Float, *device)).into()
              }
        ));

        let res:Arc<TCD> = block_on(runner.clone().run_loader(
            &Resource::new(0, 1, "aaa".to_string(), 1000),
            &Device::Cpu,
            None
        ));

        debug!("loader_cache keys:{:?}", runner.loader_cache_fut.iter().map(|v|v.key().clone()).collect::<Vec<String>>());
        debug!("got res={res:?} in {:?}", t_0.elapsed());

        assert_eq!(res.size(), [1000]);
    }
}