use std::ops::Deref;
use std::sync::Arc;
use crate::runner::common::OutId;
use crate::runner::common::runnable::{ConsumeResult, Stage, SyncStage};
use thoosa_shared::traits::{Adjustable, WithId};
use thoosa_shared::ThoError;



#[derive(Debug)]
pub struct StreamStage<S>
where
    S:SyncStage
{
    stage:Arc<S>
}

impl <S> StreamStage<S>
where
    S:SyncStage
{
    pub fn new(stage: Arc<S>) -> Self {
        Self { stage }
    }
}

impl <S> Stage for StreamStage<S>
where
    S:SyncStage
{
    type Ctx = S::Ctx;
    type In<'a> = S::In<'a>;
    type Out = S::Out;
}

impl <S> SyncStage for StreamStage<S>
where
    S:SyncStage
{
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {

        // let (out, stream) = with_pooled_cuda_stream(ctx.device_idx(), false,move |_|{
            Arc::clone(&self.stage).do_run(input, ctx)
        // })?;
        // if let Some(stream) = stream {
        //     stream.synchronize()?;
        // }

        // out
    }

    fn consume<'f>(self: Arc<Self>, input: Self::In<'f>, ctx: Self::Ctx) -> ConsumeResult<Self::In<'f>, Self::Out> {
        Arc::clone(&self.stage).consume(input, ctx)
    }
}

impl <S> AsRef<S> for StreamStage<S>
where
    S:SyncStage
{
    fn as_ref(&self) -> &S {
        self.stage.as_ref()
    }
}

impl<S> Deref for StreamStage<S>
where
    S: SyncStage,
{
    type Target = S;

    fn deref(&self) -> &Self::Target {
        self.stage.as_ref()
    }
}

impl<S> OutId for StreamStage<S>
where
    S: SyncStage + OutId,
    for<'a> <S as Stage>::In<'a>: WithId,
    <S as Stage>::Out: WithId,
{
    fn out_id<'a>(
        &self,
        input_id: &<<Self as Stage>::In<'a> as WithId>::ID,
    ) -> <Self::Out as WithId>::ID {
        self.stage.out_id(input_id)
    }
}

impl<S> Adjustable for StreamStage<S>
where
    S: SyncStage + Adjustable,
{
}


#[cfg(test)]
mod tests {
    use std::ops::Mul;
    use log::debug;
    use rayon::yield_now;
    use tch::kind::FLOAT_CUDA;
    use tch::Tensor;
    use tokio::time::Instant;
    use thoosa_shared::os_thread_id;
    use crate::cuda_utils::with_pooled_cuda_stream;
    use crate::tho_tensor::ThoTensor;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_0() {

        fn run_tensor_ops(label:&str) {
            let inst = Instant::now();
            let mut t = Tensor::rand([100, 100, 100_000], FLOAT_CUDA);
            for i in  0..1000 {
                t = if i % 2 == 0 {
                    t.pow_tensor_scalar(2.0)
                } else {
                    t.pow_tensor_scalar(0.5)
                };

                // debug!("{label}_{i}_{}", os_thread_id());
            }
            debug!("{label} done in {}", inst.elapsed().as_millis())
        }
        const WITH_STREAM:bool = true;
        let inst = Instant::now();
        tokio::join!(
            tokio::spawn(async move {
                if WITH_STREAM {
                    with_pooled_cuda_stream(1, false, |_|{
                        run_tensor_ops("aaa");
                    }).await.unwrap();
                } else {
                    run_tensor_ops("aaa");
                }
            }),
            tokio::spawn(async {
                if WITH_STREAM {
                    with_pooled_cuda_stream(1, false, |_|{
                        run_tensor_ops("bbb");
                    }).await.unwrap();
                } else {
                    run_tensor_ops("bbb");
                }

            })
        );

        debug!("all done in {} with_stream={WITH_STREAM}", inst.elapsed().as_millis())
    }

}