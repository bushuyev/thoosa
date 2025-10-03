use ambassador::Delegate;
use async_trait::async_trait;
use caching_runner::common::{ambassador_impl_PermitsCtx, DropIf, PermitsCtx, RunCtx};
use caching_runner::oom_handler::Adjustable;
use caching_runner::permits::{NumPermits, RunnerPerm};
use thoosa_shared::traits::{Named, SimpleId};

#[derive(Debug, Delegate)]
#[delegate(PermitsCtx, target = "perms")]
pub struct RunCtx2 {
    device_idx: usize,
    cache_area: u64,
    name: &'static str,
    perms: RunCtx2Perms,
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
            perms: RunCtx2Perms::new(name),
        }
    }
}

#[derive(Debug)]
struct RunCtx2Perms {
    name: &'static str,
}

impl RunCtx2Perms {
    fn new(name: &'static str) -> Self {
        Self { name }
    }
}

impl Named for RunCtx2Perms {
    fn name(&self) -> &'static str {
        self.name
    }
}

impl PermitsCtx for RunCtx2Perms {
    fn add_permit(&mut self, _permit: RunnerPerm) {
        todo!()
    }

    fn drop_permits(&mut self, _drop_if: DropIf) {
        todo!()
    }

    fn acquired_permits(&self) -> NumPermits {
        todo!()
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
