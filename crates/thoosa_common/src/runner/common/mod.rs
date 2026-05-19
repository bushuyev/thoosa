// removed unused feature gate to reduce warnings

use ambassador::delegatable_trait;
use futures_util::future::BoxFuture;
use runnable::Stage;
use std::fmt::Debug;
use thoosa_shared::ThoError;
use thoosa_shared::traits::{Named, WithId};
pub mod cachable;
pub mod runnable;



#[derive(Debug)]
pub enum DropIf {
    Always,
    IfNotMax,
}

#[delegatable_trait]
pub trait MaxParallelCtx: Named + Debug + Send + Sync {
    fn set_max_other_parallel(&mut self, permit: MaxParallel);

    fn max_other_parallel(&self) -> MaxParallel;
}

pub trait RunCtx: MaxParallelCtx + WithId {
    fn device_idx(&self) -> usize;

    fn branch(&mut self, new_name: Option<&str>, with_new_cache_area: bool) -> Self
    where
        Self: Sized;

    fn add_purge_gpu_closure(&mut self, cl: Box<dyn Fn() -> BoxFuture<'static, Result<(), ThoError>> + Send + Sync>);

}

#[cfg_attr(test, allow(dead_code))]
pub mod mock_run_ctx {
    use super::{MaxParallelCtx, RunCtx};
    use futures_util::future::BoxFuture;
    use log::debug;
    use mockall::mock;
    use mockall::predicate::always;
    use thoosa_shared::traits::{Adjustable, Named};
    use thoosa_shared::{mono_key, ThoError};
    use thoosa_shared::comp_id::MonoKey;
    use thoosa_shared::traits::WithId;
    use crate::runner::adjusters::max_parallel::{MaxParallel, MAX_PARALLEL};
    mock! {
        #[derive(Debug)]
        pub RunCtx {}

        impl Named for RunCtx {
            fn name() -> &'static str {
                "MockRunCtx"
            }
        }

        impl Adjustable for RunCtx{}

        impl MaxParallelCtx for RunCtx {
            fn set_max_other_parallel(&mut self, permit: MaxParallel);
            fn max_other_parallel(&self) -> MaxParallel;
        }

        impl WithId for RunCtx {
            type ID = MonoKey;
            fn id(&self) -> &MonoKey;
        }

        impl RunCtx for RunCtx {
            fn device_idx(&self) -> usize;
            fn branch<'a>(&mut self, new_name:Option<&'a str>, with_new_cache_area:bool)-> Self;
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

        pub fn with_max_other_parallel(mut self, out: crate::runner::adjusters::max_parallel::MaxParallel) -> Self {
            self.mock.expect_max_other_parallel().times(0..).return_const(out);
            self
        }

        pub fn with_add_purge_gpu_closure(mut self) -> Self{
            self.mock.expect_add_purge_gpu_closure().withf(|_cl| true).times(0..).returning(|_cl| ());

            self
        }

        pub fn with_set_max_other_parallel(mut self) -> Self {
            self.mock.expect_set_max_other_parallel()
                .with(always())
                .return_const(());

            self
        }


        pub fn with_branch(mut self, depth: u32) -> Self {
            self.mock.expect_branch()
                // .with(always(), always())
                .returning(move |_name:Option<&str>, _with_new_cache_area:bool| {

                    debug!("with_branch: depth={depth}");

                    let builder = build()
                        .with_id(depth)
                        .with_device_idx(self.device_idx.unwrap_or(0))
                        .with_max_other_parallel(MAX_PARALLEL)
                        .with_set_max_other_parallel()
                        .with_add_purge_gpu_closure();

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
            .with_max_other_parallel(100)
            .with_max_other_parallel(1)
            .with_set_max_other_parallel()
            .with_branch(1024)
            .with_max_other_parallel(1)
            .with_add_purge_gpu_closure()
            .finish()
    }
}

pub use mock_run_ctx::MockRunCtx;
use crate::runner::adjusters::max_parallel::MaxParallel;
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
