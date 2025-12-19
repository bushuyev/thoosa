// removed unused feature gate to reduce warnings

use crate::permits::{NumPermits, RunnerPerm, MAX_PERMITS};
use ambassador::delegatable_trait;
use futures_util::future::BoxFuture;
use async_trait::async_trait;
use parking_lot::RwLock;
use runnable::Stage;
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::ops::Deref;
use std::sync::Arc;
use thoosa_shared::CopyToDevice;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{CachableIdTr, Metrics, Named, WithId};

pub mod cachable;
pub mod runnable;



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

pub trait RunCtx: PermitsCtx {//TODO remove PermitsCtx from RunCtx
    fn device_idx(&self) -> usize;

    fn purge_gpu_caches(_ctx: Arc<RwLock<Self>>) -> BoxFuture<'static, Result<(), ThoosaError>>
    where
        Self: Sized,
    {
        Box::pin(async { Ok(()) })
    }

    fn cache_area(&self) -> u64;

    fn branch(&mut self, new_name:&str)-> Arc<RwLock<Self>> {
        todo!()
    }
}

#[cfg_attr(test, allow(dead_code))]
pub mod mock_run_ctx {
    use super::{mock_run_ctx, DropIf, PermitsCtx, RunCtx};
    use crate::permits::{NumPermits, RunnerPerm};
    use futures_util::future::BoxFuture;
    use mockall::mock;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use mockall::predicate::always;
    use thoosa_shared::traits::{Adjustable, Named};
    use thoosa_shared::ThoosaError;
    use crate::common::DropIf::IfNotMax;

    mock! {
        #[derive(Debug)]
        pub RunCtx {}

        impl Named for RunCtx {
            fn name(&self) -> &'static str {
                "MockRunCtx"
            }
        }

        impl Adjustable for RunCtx{}

        impl PermitsCtx for RunCtx {
            fn add_permit(&mut self, permit: RunnerPerm);
            fn drop_permits(&mut self, drop_if: DropIf);
            fn acquired_permits(&self) -> NumPermits;
        }

        impl RunCtx for RunCtx {
            fn device_idx(&self) -> usize;
            fn purge_gpu_caches(ctx: Arc<RwLock<Self>>) -> BoxFuture<'static, Result<(), ThoosaError>>;
            fn cache_area(&self) -> u64;
            fn branch(&mut self, new_name:&str)-> Arc<RwLock<Self>>;
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

        pub fn with_cache_area(mut self, out: u64) -> Self {
            self.mock.expect_cache_area().return_const(out);
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

        pub fn with_add_permits(mut self) -> Self {
            self.mock.expect_add_permit()
                .with(always())
                .return_const(());

            self
        }

        pub fn with_purge_gpu_caches(mut self) -> Self {
            MockRunCtx::purge_gpu_caches_context().expect().with(always()).returning( |_| Box::pin(async { Ok(()) }));

            self
        }

        pub fn with_branch(mut self, depth: u8) -> Self {
            self.mock.expect_branch()
                .with(always())
                .returning(move |_| {
                    let builder = build()
                        .with_device_idx(1)
                        .with_acquired_permits(100)
                        .with_add_permits()
                        .with_drop_permits(vec![IfNotMax])
                        .with_cache_area(0);

                    Arc::new(RwLock::new(
                        if depth == 0 {
                            builder.finish()
                        } else {
                            builder
                            .with_branch(depth - 1)
                            .finish()
                        }
                    ))
                });

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

    pub fn default(device_idx: usize) -> MockRunCtx {
        build()
            .with_device_idx(device_idx)
            .with_acquired_permits(100)
            .with_add_permits()
            .with_drop_permits(vec![IfNotMax])
            .with_cache_area(0)
            .with_branch(4)
            .finish()
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
        for<'a> <Self as Stage>::In<'a>: WithId,
        <Self as Stage>::Out: WithId,
{
    fn out_id<'a>(
        &self,
        input_id: &<<Self as Stage>::In<'a> as WithId>::ID,
    ) -> <Self::Out as WithId>::ID;

}
