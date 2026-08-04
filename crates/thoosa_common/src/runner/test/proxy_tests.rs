use futures_util::future::{BoxFuture, FutureExt, ready};
use futures_util::{StreamExt, ready};
use std::sync::Arc;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use thoosa_proxy_derive::Proxy;
use thoosa_shared::ThoError;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::traits::{MaxParallelCtx, Named, RunCtx, Stage, SyncStage, WithId};


#[derive(Debug, Serialize, Deserialize)]
pub struct TestInput{}
impl Input for TestInput{}


#[derive(Debug, Serialize, Deserialize)]
pub struct TestOutput{}
impl Output for TestOutput{}


#[derive(Debug)]
pub struct TestRunCtx;

impl Named for TestRunCtx {}

impl MaxParallelCtx for TestRunCtx{
    fn set_max_other_parallel(&mut self, permit: u8) {}

    fn max_other_parallel(&self) -> u8 { 0 }
}

impl WithId for TestRunCtx{
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl RunCtx for TestRunCtx {
    fn tho_device_idx(&self) -> usize {
        1
    }

    fn branch(&mut self, with_new_cache_area: bool) -> Self
    where
        Self: Sized
    {
        TestRunCtx
    }

    fn add_purge_gpu_closure(&mut self, cl: Box<dyn Fn() -> BoxFuture<'static, Result<(), ThoError>> + Send + Sync>) {
    }
}

const TEST_ENDPOINT: &str = "TestStage";

#[derive(Debug, Proxy)]
#[proxy(
    name = "ProxyTestStage",
    endpoint = "TEST_ENDPOINT",
    new_args = "tho_device_idx: usize",

)]
pub struct TestStage {

}

impl Stage for TestStage {
    type Ctx = TestRunCtx;
    type In<'a> = TestInput;
    type Out = TestOutput;
}

impl SyncStage for TestStage {
    fn do_run<'f>(self: Arc<Self>, _input: &Self::In<'f>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        Ok(TestOutput{})
    }
}


mod tests {
    use log::debug;
    use std::sync::Arc;
    use thoosa_common::runner::test::proxy_tests::{TestInput, TestRunCtx, TestStage};
    use thoosa_shared::ThoError;
    use thoosa_shared::traits::SyncStage;

    #[test]
    fn test_0() -> Result<(), ThoError>{
        let test_stage = Arc::new(TestStage{});
        let input = TestInput{};
        let ctx = TestRunCtx;
        let result = test_stage.do_run(&input, &ctx)?;


        debug!("result={result:?}");

        Ok(())
    }
}