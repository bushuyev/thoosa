// removed unused feature gate to reduce warnings

use crate::permits::{NumPermits, RunnerPerm};
use ambassador::delegatable_trait;
use futures_util::future::BoxFuture;
use async_trait::async_trait;
use mockall::automock;
use parking_lot::RwLock;
use runnable::Stage;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::ops::{Deref, Range};
use std::sync::Arc;
use thoosa_shared::CopyToDevice;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{CachableIdTr, Named, WithId};

pub mod cachable;
pub mod runnable;

#[async_trait]
#[delegatable_trait]
#[automock(type D=MockCopyToDevice; type Metrics=[usize;1]; type Sub=Self; type SubSelect=Range<<Self as Data>::Metrics>;)]
pub trait Data: Debug + CopyToDevice {
    type Metrics: Debug + Clone + Send + Sync + Default;

    fn size(&self) -> Self::Metrics;

    fn track_used_in(&self, _token: &'static str) {}

    fn device(&self) -> usize;
}

#[delegatable_trait]
pub trait Select<S> {
    type Output: CopyToDevice;

    fn subview(&self, selector: S) -> Self::Output;
}



#[async_trait]
impl CopyToDevice for MockData {
    async fn copy_to_device(&self, _dst_device: usize) -> Self {
        MockData::new()
    }
}

#[derive(Debug)]
pub enum DropIf {
    Always,
    IfNotMax,
}

#[delegatable_trait]
pub trait PermitsCtx: Named + Debug + Send + Sync {
    fn add_permit(&mut self, permit: RunnerPerm);

    fn drop_permits(&mut self, drop_if: DropIf);

    fn acquired_permits(&self) -> NumPermits;
}

pub trait RunCtx: PermitsCtx {
    fn device_idx(&self) -> usize;

    fn purge_gpu_caches(_ctx: Arc<RwLock<Self>>) -> BoxFuture<'static, Result<(), ThoosaError>>
    where
        Self: Sized,
    {
        Box::pin(async { Ok(()) })
    }

    fn cache_area(&self) -> u64;
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod mock_run_ctx {
    use super::{DropIf, PermitsCtx, RunCtx};
    use crate::permits::{NumPermits, RunnerPerm};
    use futures_util::future::BoxFuture;
    use mockall::mock;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use thoosa_shared::traits::Named;
    use thoosa_shared::ThoosaError;

    mock! {
        #[derive(Debug)]
        pub RunCtx {}

        impl Named for RunCtx {
            fn name(&self) -> &'static str {
                "MockRunCtx"
            }
        }

        impl PermitsCtx for RunCtx {
            fn add_permit(&mut self, permit: RunnerPerm);
            fn drop_permits(&mut self, drop_if: DropIf);
            fn acquired_permits(&self) -> NumPermits;
        }

        impl RunCtx for RunCtx {
            fn device_idx(&self) -> usize;
            fn purge_gpu_caches(ctx: Arc<RwLock<Self>>) -> BoxFuture<'static, Result<(), ThoosaError>>;
            fn cache_area(&self) -> u64;
        }
    }

    pub struct MockRunCtxBuilder {
        mock: MockRunCtx,
    }

    impl MockRunCtxBuilder {
        pub fn with_device_idx(mut self, out: usize) -> Self {
            self.mock.expect_device_idx().return_const(out);
            self
        }

        pub fn with_acquired_permits(mut self, out: NumPermits) -> Self {
            self.mock.expect_acquired_permits().return_const(out);
            self
        }

        pub fn with_drop_permits(mut self, inp: Vec<DropIf>) -> Self {
            for param in inp {
                self.mock.expect_drop_permits().call(param);
            }

            self
        }

        pub fn finish(self) -> MockRunCtx {
            self.mock
        }
    }

    pub fn build() -> MockRunCtxBuilder {
        let mut mock = super::MockRunCtx::new();
        mock.expect_name().return_const("MockRunCtx");
        MockRunCtxBuilder { mock }
    }
}

pub use mock_run_ctx::MockRunCtx;

// pub trait Input: WithId + Debug {}
// pub trait Output: WithId + Debug {}

// #[cfg(test)]
// pub mod mock_input_output {
//     use mockall::*;
//
//     mock! {
//         #[derive(Debug)]
//         pub Input{}
//
//         impl super::WithId for Input {
//             type ID = crate::common::cachable::mock_cachable_id::MockCachableId;
//
//             fn id(&self) -> &crate::common::cachable::mock_cachable_id::MockCachableId;
//         }
//         impl super::Input for Input {
//
//         }
//
//         impl super::CopyToDevice for Input {}
//
//         impl super::Data for Input {
//             type Metrics = [usize;1];
//             type D:CopyToDevice = Self;
//             type Sub:CopyToDevice = Self;
//             type SubSelect = std::ops::Range<Self::Metrics>;
//
//         }
//     }
//
//
// }

/////////////////////////////////

pub trait OutId: Stage
where
    Self: 'static,
    <Self as Stage>::In<'static>: WithId,
{
    fn out_id(
        &self,
        input_id: &<<Self as Stage>::In<'static> as WithId>::ID,
    ) -> <Self::Out as WithId>::ID;

    fn on_cached(
        &self,
        _input: &impl WithId<ID = <<Self as Stage>::In<'static> as WithId>::ID>,
        _ctx: &Arc<RwLock<Self::Ctx>>,
    ) {
    }
}
