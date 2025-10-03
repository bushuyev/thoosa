use ambassador::Delegate;
use async_trait::async_trait;
use caching_runner::common::{ambassador_impl_PermitsCtx, DropIf, PermitsCtx, RunCtx};
use caching_runner::oom_handler::Adjustable;
use caching_runner::permits::{NumPermits, RunnerPerm};
use thoosa_shared::traits::{Named, SimpleId};
use torch_runner::TorchPermsCtx;

#[derive(Debug, Delegate)]
#[delegate(PermitsCtx, target = "perms")]
pub struct RunCtx2 {
    device_idx: usize,
    cache_area: u64,
    name: &'static str,
    perms: TorchPermsCtx,
}

impl Adjustable for RunCtx2 {}

impl Named for RunCtx2 {
    fn name(&self) -> &'static str {
        self.name
    }
}
impl RunCtx2 {
    pub fn new(device_idx: usize, cache_area: u64, name: &'static str) -> Self {
        Self {
            device_idx,
            cache_area,
            name,
            perms: TorchPermsCtx::new(name),
        }
    }
}

#[async_trait]
impl RunCtx for RunCtx2 {
    fn device_idx(&self) -> usize {
        self.device_idx
    }

    fn cache_area(&self) -> u64 {
        self.cache_area
    }
}
