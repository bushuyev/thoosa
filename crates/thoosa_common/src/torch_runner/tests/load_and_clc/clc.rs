use std::collections::HashMap;
use std::sync::Arc;
use log::debug;
use tch::{Device, Kind, Scalar, Tensor};
use num_traits::ToPrimitive;
use tch::Kind::Float;
use tch::Device::Cuda;
use thoosa_named_derive::Named;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::{simple_id, ThoError};
use thoosa_shared::traits::{Adjustable, MonoMetrics, SimpleId};
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};
use crate::copy_ctx::CudaActorHandle;
use crate::dims::Dims;
use crate::ipc::cuda_process::encode_endpoint_payload;
use crate::ipc::json_ser::{encode_json_sync_stage_run_input, SyncStageJsonWrapper};
use crate::runner::common::cachable::DevicesWrap;
use crate::runner::common::{MaxParallelCtx, OutId, RunCtx};
use crate::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::oom_handler::{Contexts, NoopInputContext, NoopRunCtxContext, NoopRunnableContext, OOMHandler};
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::runner::test::FailUntil;
use crate::tho_tensor::ThoTensor;
use crate::torch_runner::tests::load_and_clc::{ClcDynSyncStage, SYNC_CLC_ENDPOINT};
use crate::torch_runner::tests::load_and_clc::loader::LoadedTensor;
use crate as thoosa_common;
use crate::tho_device::ThoDevice;

// //////////////////////////////
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct MulK;

impl Dims for MulK {
    type Metrics = MonoMetrics<1, 2>;
}

#[derive(Debug, cached_derive::Cached, Named)]
#[cache_name = "ClcTaskOutputId"]
pub(crate) struct ClcStage {
    pub(crate) sync_clc: Arc<ClcDynSyncStage>,
}

#[derive(Debug)]
pub(crate) struct SyncClcStage {
    id: SynKey,
    k: ThoTensor<MulK, SimpleId>,
    fail_until: Vec<FailUntil>,
    use_matmul: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SyncClcTaskInit {
    id: SynKey,k: f64, dim: i64, fail_until: Vec<FailUntil>, use_matmul: bool
}

impl SyncClcTaskInit {
    pub fn new(id: SynKey,k: f64, dim: i64, fail_until: Vec<FailUntil>, use_matmul: bool) ->Self {
        Self {
            id, k, dim, fail_until,  use_matmul
        }
    }
}

impl SyncClcStage {
    pub async fn new<K>(id: SynKey, k: K, fail_until: Vec<FailUntil>, dim: i64, kind: Kind, device: ThoDevice, use_matmul: bool) -> ClcDynSyncStage
    where
        K: Into<Scalar> + ToPrimitive + Copy,
    {
        if GlobalCtx::global().is_ipc() {
            let k = k.to_f64().expect("SyncClcTask::new: IPC task multiplier must convert to f64");
            let proxy = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(ProxySyncClcStage::new(id, k, dim, fail_until, kind, device, use_matmul))
            }).expect("SyncClcTask::new: install IPC proxy sync task");
            ClcDynSyncStage::new(Arc::new(proxy))
        } else {
            ClcDynSyncStage::new(Arc::new(Self::new_local(id, k, fail_until, dim, kind, device, use_matmul)))
        }
    }

    fn new_local<K>(id: SynKey, k: K, fail_until: Vec<FailUntil>, dim: i64, kind: Kind, device: ThoDevice, use_matmul: bool) -> Self
    where
        K: Into<Scalar> + Copy,
    {
        let k =
            if use_matmul {
                ThoTensor::new(
                    simple_id!("mul_k"),
                    Tensor::ones([dim, 1], (kind, device.into()))
                        .f_mul_scalar_(k)
                        .unwrap(),
                ).unwrap()
            } else {
                ThoTensor::new(
                    simple_id!("mul_k"),
                    Tensor::ones([dim], (kind, device.into()))
                        .f_mul_scalar_(k)
                        .unwrap(),
                ).unwrap()
            };

        debug!("ClcTask::new: use_matmul={use_matmul} i.size={:?}", k.size());

        Self { id, k, fail_until, use_matmul }
    }
    pub fn init(init: SyncClcTaskInit) -> SyncClcStage {
        let SyncClcTaskInit{id, k, dim, fail_until, use_matmul} = init;
        SyncClcStage::new_local(
            id, k, fail_until, dim, Float, Cuda(0).into(), use_matmul
        )
    }

    pub async fn new_proxy(
        id: SynKey,
        k: f64,
        fail_until: Vec<FailUntil>,
        dim: i64,
        kind: Kind,
        device: ThoDevice,
        use_matmul: bool,
    ) -> Result<ProxySyncClcStage, ThoError> {

        ProxySyncClcStage::new(id, k, dim, fail_until, kind, device, use_matmul).await
    }
}


impl ClcStage {
    pub async fn new<K>(
        id: SynKey,
        k: K,
        fail_until: Vec<FailUntil>,
        dim: i64,
        kind: Kind,
        device: ThoDevice,
        use_matmul: bool,
    ) -> Self
    where
        K: Into<Scalar> + ToPrimitive + Copy,
    {

        Self {
            sync_clc: Arc::new(SyncClcStage::new(id, k, fail_until, dim, kind, device, use_matmul).await),
        }
    }

    pub async fn new_proxy(
        id: SynKey,
        k: f64,
        fail_until: Vec<FailUntil>,
        dim: i64,
        kind: Kind,
        device: ThoDevice,
        use_matmul: bool,
    ) -> Result<Self, ThoError> {

        Ok(Self {
            sync_clc: Arc::new(ClcDynSyncStage::new(Arc::new(SyncClcStage::new_proxy(id, k, fail_until, dim, kind, device, use_matmul).await?))),
        })
    }

}

impl Adjustable for ClcStage {}

impl Adjustable for SyncClcStage {}

impl OutId for ClcStage {
    fn out_id(&self, input_id: &SynKey) -> SynKey {

        self.sync_clc.out_id(input_id)
    }
}

impl OutId for SyncClcStage {
    fn out_id(&self, input_id: &SynKey) -> SynKey {

        &self.id + input_id
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ClcStageOutput;

impl Dims for ClcStageOutput {
    type Metrics = MonoMetrics<1, 2>;
}

impl Stage for ClcStage {
    type In<'a> = ThoTensor<LoadedTensor, SynKey>;
    type Out = ThoTensor<ClcStageOutput, SynKey>;
    type Ctx = RunCtxImpl;
}

impl Stage for SyncClcStage {
    type In<'a> = ThoTensor<LoadedTensor, SynKey>;
    type Out = ThoTensor<ClcStageOutput, SynKey>;
    type Ctx = RunCtxImpl;
}

impl SyncStage for SyncClcStage {
    fn do_run<'a>(
        self: Arc<Self>,
        input: &Self::In<'a>,
        ctx: &Self::Ctx,
    ) -> Result<Self::Out, ThoError>
    where
        Self: Sized,
    {

        debug!("MulTask::do_run: loaded.id={:?} self.fail_until={:?}", input.id(), self.fail_until);

        let device_idx = ctx.tho_device_idx();

        for fail_until in &self.fail_until {
            match fail_until {
                FailUntil::MemLimit(mem_limit) => {
                    let _device_idx = device_idx;

                    if input.size()[0] > *mem_limit as i64 {
                        debug!("MulTask::do_run: MyError::OutOfMemory on MemLimit mem_limit={mem_limit}");
                        return Err(ThoError::oom());
                    }
                }
                FailUntil::NaxParallel(max_other_parallel_threshold) => {
                    let max_other_parallel = ctx.max_other_parallel();
                    debug!("MulTask::do_run: ctx={ctx:?} max_other_parallel={max_other_parallel} max_other_parallel_threshold={max_other_parallel_threshold}");

                    if max_other_parallel > *max_other_parallel_threshold {
                        return Err(ThoError::oom());
                    }
                }

                FailUntil::DumpedToCpu => {
                    let device_idx = ctx.tho_device_idx();
                    let is_all_on_cpu = GlobalCtx::global().is_all_on_cpu(device_idx, false);
                    debug!("MulTask::do_run: ctx.name={:?} device_idx={} is_all_on_cpu={is_all_on_cpu}","ctx.name()",device_idx);

                    if !is_all_on_cpu {
                        debug!("MulTask::do_run: MyError::OutOfMemory on DumpedToCpu is_all_on_cpu={is_all_on_cpu}");
                        return Err(ThoError::oom());
                    }
                }
                FailUntil::Never => {}
            }
        }

        if self.use_matmul {
            LoadedTensor::clc_sum(self.out_id(input.id()), input, &self.k)
        } else {
            LoadedTensor::clc_hadamard(self.out_id(input.id()), input, &self.k)
        }
    }
}

#[derive(Debug)]
pub struct ProxySyncClcStage {
    id: SynKey,
    fail_until_dump_to_cpu: bool,// DumpedToCpu check can't be done in worker because it needs GlobalCtx::global().is_all_on_cpu
}

impl Stage for ProxySyncClcStage {
    type In<'a> = ThoTensor<LoadedTensor, SynKey>;
    type Out = ThoTensor<ClcStageOutput, SynKey>;
    type Ctx = RunCtxImpl;
}

impl Adjustable for ProxySyncClcStage {}

impl OutId for ProxySyncClcStage {
    fn out_id(&self, input_id: &SynKey) -> SynKey {
        &self.id + input_id
    }
}

impl ProxySyncClcStage {
    pub async fn new(id: SynKey,k: f64, dim: i64, fail_until: Vec<FailUntil>,kind: Kind, tho_device: ThoDevice, use_matmul: bool) -> Result<Self, ThoError> {
        debug!("ProxySyncClcTask::new: id={id:?} k={k:?}");
        let (fail_until_dump_to_cpu, fail_until) = {
            let (dumped, rest): (Vec<_>, Vec<_>) =
                fail_until.into_iter().partition(|e| *e == FailUntil::DumpedToCpu);

            (!dumped.is_empty(), rest)
        };

        GlobalCtx::global()
            .cuda_actor(tho_device)
            .install_endpoint(encode_endpoint_payload(SYNC_CLC_ENDPOINT, &serde_json::to_string(&SyncClcTaskInit::new(id.clone(), k, dim, fail_until, use_matmul))?))
            .await?;

        Ok(Self {
            id,
            fail_until_dump_to_cpu,
        })
    }
}

impl SyncStage for ProxySyncClcStage {
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        debug!("ProxySyncClcTask::new: input={input:?}");


        if self.fail_until_dump_to_cpu {
            let device_idx = ctx.tho_device_idx();
            let is_all_on_cpu = GlobalCtx::global().is_all_on_cpu(device_idx, false);
            debug!("ProxySyncClcTask::do_run: device_idx={device_idx} is_all_on_cpu={is_all_on_cpu}");
            if !is_all_on_cpu {
                return Err(ThoError::oom());
            }
        }


        tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(async move {

            let response = GlobalCtx::global()
                .cuda_actor(ctx.tho_device_idx())
                .invoke_endpoint(encode_endpoint_payload(SYNC_CLC_ENDPOINT, &encode_json_sync_stage_run_input(serde_json::to_string(input)?, ctx)?))
                .await?;


            serde_json::from_str::<ThoTensor<ClcStageOutput, SynKey>>(&response).map_err(ThoError::from)
        }))

    }
}

// #[async_trait]
impl AsyncStage for ClcStage {
    type RunInput = Arc<DevicesWrap<Self::In<'static>>>;
    type RunOutput = Arc<DevicesWrap<Self::Out>>;

    fn run_input_id(input: &Self::RunInput) -> &<<Self as Stage>::In<'static> as thoosa_shared::traits::WithId>::ID {
        thoosa_shared::traits::WithId::id(input)
    }
    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let device_idx = ctx.tho_device_idx();

        debug!("AsyncStage for ClcTask::run: device_idx={device_idx} input={input:?}");

        let output = OOMHandler::new(vec![], None).run(
            &self.sync_clc,
            input,
            ctx,
            Contexts::new(
                NoopInputContext(None),
                NoopRunnableContext {},
                NoopRunCtxContext {},
            )
        ).await?;


        Ok(Arc::new(DevicesWrap::wrap(output)))
    }

}
