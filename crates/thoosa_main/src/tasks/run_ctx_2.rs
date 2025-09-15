use async_trait::async_trait;
use caching_runner::common::{DropIf, RunCtx};
use caching_runner::oom_handler::Adjustable;
use caching_runner::permits::{NumPermits, RunnerPerm};
use thoosa_shared::traits::{Named, SimpleId};




#[derive(Debug)]
pub struct RunCtx2 {
    device_idx: usize,
    permits: Vec<RunnerPerm>,
    cache_area: u64,
    name: &'static str
}

impl Adjustable for RunCtx2 {

}

impl Named for RunCtx2 {
    fn name(&self) -> &'static str {
        self.name
    }
}
impl RunCtx2 {
    pub fn new(device_idx: usize, cache_area: u64, name: &'static str) -> Self {
        Self { device_idx, permits:vec![], cache_area, name }
    }
}


#[async_trait]
impl RunCtx for RunCtx2 {
    fn add_permit(&mut self, permit: RunnerPerm) {
        todo!()
    }

    fn drop_permits(&mut self, drop_if: DropIf) {
        todo!()
    }

    fn device_idx(&self) -> usize {
        todo!()
    }

    fn acquired_permits(&self) -> NumPermits {
        todo!()
    }

    fn cache_area(&self) -> u64 {
        todo!()
    }
}