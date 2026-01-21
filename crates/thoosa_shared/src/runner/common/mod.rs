// removed unused feature gate to reduce warnings

use ambassador::delegatable_trait;
use futures_util::future::BoxFuture;
use parking_lot::RwLock;
use runnable::Stage;
use std::fmt::Debug;
use std::sync::Arc;
use crate::ThoError;
use crate::traits::{CachableIdTr, Named, WithId};
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

pub trait RunCtx: PermitsCtx + WithId {//TODO remove PermitsCtx from RunCtx
    fn device_idx(&self) -> usize;

    fn purge_gpu_caches(&mut self) -> BoxFuture<'static, Result<(), ThoError>>
    where
        Self: Sized,
    {
        //TODO remove - replaced by add_purge_gpu_closure + purge on drop
        Box::pin(async { Ok(()) })
    }


    fn cache_area(&self) -> u64;

    fn branch(&mut self, new_name: Option<&str>, with_new_cache_area: bool, give_permits_back_to_parent: bool) -> Self
    where
        Self: Sized;

    fn add_purge_gpu_closure(&mut self, cl: Box<dyn Fn() -> BoxFuture<'static, Result<(), ThoError>> + Send + Sync>);

}

#[cfg_attr(test, allow(dead_code))]
pub mod mock_run_ctx {
    use super::{DropIf, PermitsCtx, RunCtx};
    use futures_util::future::BoxFuture;
    use mockall::mock;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use mockall::predicate::always;
    use tracing::debug;
    use thoosa_shared::traits::{Adjustable, Named};
    use thoosa_shared::{mono_key, ThoError};
    use crate::comp_id::{MonoKey, SynKey};
    use crate::traits::WithId;
    use crate::runner::oom_handler::MockRunCtxContext;

    mock! {
        #[derive(Debug)]
        pub RunCtx {}

        impl Named for RunCtx {
            fn name() -> &'static str {
                "MockRunCtx"
            }
        }

        impl Adjustable for RunCtx{}

        impl PermitsCtx for RunCtx {
            fn add_permit(&mut self, permit: crate::runner::permits::RunnerPerm);
            fn drop_permits(&mut self, drop_if: DropIf);
            fn acquired_permits(&self) -> crate::runner::permits::NumPermits;
        }

        impl WithId for RunCtx {
            type ID = MonoKey;
            fn id(&self) -> &MonoKey;
        }

        impl RunCtx for RunCtx {
            fn device_idx(&self) -> usize;
            fn purge_gpu_caches(&mut self) -> BoxFuture<'static, Result<(), ThoError >>;
            fn cache_area(&self) -> u64;
            fn branch<'a>(&mut self, new_name:Option<&'a str>, with_new_cache_area:bool, give_permits_back_to_parent:bool)-> Self;
            fn add_purge_gpu_closure(&mut self, cl: Box<dyn Fn() -> BoxFuture<'static, Result<(), ThoError>> + Send + Sync>);
        }
    }

    pub struct MockRunCtxBuilder {
        mock: MockRunCtx,
        device_idx: Option<usize>
    }

    impl MockRunCtxBuilder {

        pub fn with_id(mut self, id: u32) -> Self {
            self.mock.expect_id().return_const(mono_key!(id, "mock_run_ctx"));
            self
        }

        pub fn with_device_idx(mut self, device_idx: usize) -> Self {
            self.device_idx = Some(device_idx);
            self.mock.expect_device_idx().return_const(device_idx);
            self
        }

        pub fn with_cache_area(mut self, out: u64) -> Self {
            self.mock.expect_cache_area().return_const(out);
            self
        }


        pub fn with_acquired_permits(mut self, out: crate::runner::permits::NumPermits) -> Self {
            self.mock.expect_acquired_permits().times(0..).return_const(out);
            self
        }

        pub fn with_add_purge_gpu_closure(mut self) -> Self{
            self.mock.expect_add_purge_gpu_closure().withf(|_cl| true).times(0..).returning(|_cl| ());

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
            self.mock.expect_purge_gpu_caches().times(0..).returning( || Box::pin(async { Ok(()) }));

            self
        }

        pub fn with_branch(mut self, depth: u8) -> Self {
            self.mock.expect_branch()
                // .with(always(), always())
                .returning(move |_name:Option<&str>, _with_new_cache_area:bool, give_permits_back_to_parent:bool| {

                    let builder = build()
                        .with_id(depth as u32)
                        .with_device_idx(self.device_idx.unwrap_or(0))
                        .with_acquired_permits(100)
                        .with_add_permits()
                        .with_drop_permits(vec![DropIf::IfNotMax])
                        .with_add_purge_gpu_closure()
                        .with_cache_area(0);

                    if depth == 0 {
                        builder.finish()
                    } else {
                        builder
                        .with_branch(depth - 1)
                        .finish()
                    }
                });

            self
        }

        pub fn finish(self) -> MockRunCtx {
            self.mock
        }
    }

    pub fn build() -> MockRunCtxBuilder {
        let mock = super::MockRunCtx::new();
        MockRunCtx::name_context().expect().return_const("MockRunCtx");//TODO return ctx
        MockRunCtxBuilder { mock, device_idx: None }
    }

    pub fn default(device_idx: usize) -> MockRunCtx {
        build()
            .with_id(0)
            .with_device_idx(device_idx)
            .with_acquired_permits(100)
            .with_acquired_permits(1)
            .with_add_permits()
            .with_drop_permits(vec![DropIf::IfNotMax])
            .with_cache_area(0)
            .with_branch(255)
            .with_acquired_permits(1)
            .with_add_purge_gpu_closure()
            .finish()
    }
}

pub use mock_run_ctx::MockRunCtx;
use crate::comp_id::SynKey;
use crate::runner::permits::{NumPermits, RunnerPerm};
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
