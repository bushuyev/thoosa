use crate as thoosa_common;
use crate::dims::Dims;
use crate::runner::common::cachable::DevicesWrap;
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::oom_handler::{NoContexts, NoInputAdjuster, NoRunCtxAdjuster, NoTaskAdjuster, OOMHandler};
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::runner::test::FailUntil;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use crate::torch_runner::tests::load_and_clc::loader::{LoadedRange, LoadedTensor};
use crate::torch_runner::tests::load_and_clc::SYNC_CLC_ENDPOINT;
use crate::ipc::codec::JsonIpcCodec;
use crate::ipc::json_ser::SyncStageIpcWrapper;
use crate::ipc::wraps::WorkerWrapperWithState;
use crate::ipc::wraps2::{DynSyncStageWithStateEnum2, ProxyStage2};
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use log::debug;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tch::Device::Cuda;
use tch::Kind::Float;
use tch::{Device, Kind, Scalar, Tensor};
use thoosa_named_derive::Named;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::traits::{Adjustable, AsyncStage, Data, ErasedStageWithState, MaxParallelCtx, MonoMetrics, OutId, RunCtx, SimpleId, Stage, StageWithState, SyncStage};
use thoosa_shared::{CopyToDevice, ThoError, simple_id};
use tokio::sync::{Mutex, MutexGuard};
use crate::ipc::worker_msg::{WorkerInput, WorkerOutput};

// //////////////////////////////
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct MulK;

impl Dims for MulK {
    type Metrics = MonoMetrics<1, 2>;
}

#[derive(Debug, cached_derive::Cached, Named)]
#[cache_name = "ClcTaskOutputId"]
pub(crate) struct ClcStage {
    id: SynKey,
    pub(crate) sync_clc: Arc<ClcDynSyncStage>,
}

#[derive(Debug)]
pub struct SyncClcStage {
    id: SynKey,
    // Keep the multiplier behind the state contract so worker execution never exports it.
    k: Mutex<ThoTensor<MulK, SimpleId>>,
    fail_until: Vec<FailUntil>,
    use_matmul: bool,
}

#[derive(Debug)]
pub(crate) struct ClcDynSyncStage {
    id: SynKey,
    fail_until_dump_to_cpu: bool,
    stage: Arc<DynSyncStageWithStateEnum2<SyncClcStage>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SyncClcTaskInit {
    id: SynKey,k: f64, dim: i64, fail_until: Vec<FailUntil>, use_matmul: bool
}

// Register the CLC configuration for postcard's typed install message.
#[typetag::serde]
impl WorkerInput for SyncClcTaskInit {}

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
            let (dumped, fail_until): (Vec<_>, Vec<_>) = fail_until.into_iter().partition(|value| *value == FailUntil::DumpedToCpu);
            let init = SyncClcTaskInit::new(id.clone(), k, dim, fail_until, use_matmul);
            ClcDynSyncStage {
                id,
                fail_until_dump_to_cpu: !dumped.is_empty(),
                // ProxyStage2 installs the typed initializer and retains the existing endpoint stage id.
                stage: Arc::new(DynSyncStageWithStateEnum2::Proxy {
                    stage: Arc::new(ProxyStage2::proxy_with_input_cloner(
                        SYNC_CLC_ENDPOINT,
                        0,
                        device.tho_device_idx(),
                        Box::new(init),
                        Arc::new(|input: &ThoTensor<LoadedTensor, SynKey>| Ok(Box::new(input.clone_ref()?) as Box<dyn WorkerInput>)),
                    ).await.expect("SyncClcStage::new: install endpoint")),
                    init: (),
                }),
            }
        } else {
            ClcDynSyncStage {
                id: id.clone(),
                fail_until_dump_to_cpu: false,
                stage: Arc::new(DynSyncStageWithStateEnum2::Local(Arc::new(Self::new_local(id, k, fail_until, dim, kind, device, use_matmul)))),
            }
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

        Self { id, k: Mutex::new(k), fail_until, use_matmul }
    }
    pub fn init(init: SyncClcTaskInit) -> SyncClcStage {
        let SyncClcTaskInit{id, k, dim, fail_until, use_matmul} = init;
        SyncClcStage::new_local(
            id, k, fail_until, dim, Float, Cuda(0).into(), use_matmul
        )
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

        Self { id: id.clone(), sync_clc: Arc::new(SyncClcStage::new(id, k, fail_until, dim, kind, device, use_matmul).await) }
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

        Ok(Self { id: id.clone(), sync_clc: Arc::new(SyncClcStage::new(id, k, fail_until, dim, kind, device, use_matmul).await) })
    }

}

impl Adjustable for ClcStage {}

impl Adjustable for SyncClcStage {}

impl OutId for ClcStage {
    fn out_id(&self, input_id: &SynKey) -> SynKey {

        &self.id + input_id
    }
}

impl OutId for SyncClcStage {
    fn out_id(&self, input_id: &SynKey) -> SynKey {

        &self.id + input_id
    }
}

impl OutId for ClcDynSyncStage {
    fn out_id(&self, input_id: &SynKey) -> SynKey {
        &self.id + input_id
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ClcStageOutput;

impl Dims for ClcStageOutput {
    type Metrics = MonoMetrics<1, 2>;
}

#[typetag::serde]
impl WorkerOutput for ClcStageOutput {

}

#[typetag::serde(name = "ThoTensor_ClcStageOutput")]
impl WorkerOutput for ThoTensor<ClcStageOutput, SynKey> {}

impl Stage for ClcStage {
    type In<'a> = Arc<DevicesWrap<ThoTensor<LoadedTensor, SynKey>>>;
    type Out = Arc<DevicesWrap<ThoTensor<ClcStageOutput, SynKey>>>;
    type Ctx = RunCtxImpl;
}

impl Stage for SyncClcStage {
    type In<'a> = ThoTensor<LoadedTensor, SynKey>;
    type Out = ThoTensor<ClcStageOutput, SynKey>;
    type Ctx = RunCtxImpl;
}

impl SyncStage for SyncClcStage {
    fn as_with_state(&self) -> Option<&dyn ErasedStageWithState<Self>> {
        Some(self)
    }
}

impl StageWithState for SyncClcStage {
    type S = ThoTensor<MulK, SimpleId>;

    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<MutexGuard<'_, Self::S>, ThoError>> {
        async move {
            let mut k = self.k.lock().await;
            if k.device() != to_device_idx {
                *k = k.copy_to_device(to_device_idx).await?;
            }
            Ok(k)
        }.boxed()
    }

    fn do_run_with_state<'a>(self: Arc<Self>, input: &Self::In<'a>, k: MutexGuard<'_, Self::S>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {

        debug!("MulTask::do_run: loaded.id={:?} input.size={:?} self.fail_until={:?}", input.id(), input.size(), self.fail_until);

        let device_idx = ctx.tho_device_idx();

        for fail_until in &self.fail_until {
            match fail_until {
                FailUntil::MemLimit(mem_limit) => {
                    let _device_idx = device_idx;


                    if input.size()[1] > *mem_limit as i64 {
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
            LoadedTensor::clc_sum(self.out_id(input.id()), input, &k)
        } else {
            LoadedTensor::clc_hadamard(self.out_id(input.id()), input, &k)
        }
    }
}

impl Adjustable for ClcDynSyncStage {}

impl Stage for ClcDynSyncStage {
    type In<'a> = ThoTensor<LoadedTensor, SynKey>;
    type Out = ThoTensor<ClcStageOutput, SynKey>;
    type Ctx = RunCtxImpl;
}

impl SyncStage for ClcDynSyncStage {
    fn as_with_state(&self) -> Option<&dyn ErasedStageWithState<Self>> {
        match self.stage.as_ref() {
            DynSyncStageWithStateEnum2::Local(_) => Some(self),
            DynSyncStageWithStateEnum2::Proxy { .. } => None,
        }
    }

    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        if self.fail_until_dump_to_cpu && !GlobalCtx::global().is_all_on_cpu(ctx.tho_device_idx(), false) {
            return Err(ThoError::oom());
        }
        Arc::clone(&self.stage).do_run(input, ctx)
    }
}

impl StageWithState for ClcDynSyncStage {
    type S = ThoTensor<MulK, SimpleId>;

    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<MutexGuard<'_, Self::S>, ThoError>> {
        StageWithState::pre_run(self.stage.as_ref(), to_device_idx)
    }

    fn do_run_with_state<'a>(self: Arc<Self>, input: &Self::In<'a>, state: MutexGuard<'_, Self::S>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        StageWithState::do_run_with_state(Arc::clone(&self.stage), input, state, ctx)
    }
}

pub(crate) fn sync_clc_endpoint_init(init: Box<dyn WorkerInput>, _ws: Arc<parking_lot::RwLock<Box<dyn crate::ws::Workspace>>>, _all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>) -> BoxFuture<'_, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    let init: Box<dyn std::any::Any> = init;
    let init = init.downcast::<SyncClcTaskInit>().expect("SyncClcStage endpoint received an unexpected initializer type");
    async move {
        // WorkerWrapperWithState keeps the multiplier on the worker across calls.
        Ok(Box::new(WorkerWrapperWithState::<JsonIpcCodec, _, _, _>::new(Arc::new(SyncClcStage::init(*init)), std::marker::PhantomData)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}

// #[async_trait]
impl AsyncStage for ClcStage {
    // type RunInput = Arc<DevicesWrap<Self::In<'static>>>;
    // type RunOutput = Arc<DevicesWrap<Self::Out>>;

    // fn run_input_id(input: &Self::In<'static>) -> &<<Self as Stage>::In<'static> as thoosa_shared::traits::WithId>::ID {
    //     thoosa_shared::traits::WithId::id(input)
    // }
    async fn run(
        self: Arc<Self>,
        input: Self::In<'static>,
        ctx: Self::Ctx,
    ) -> Result<Self::Out, ThoError> {
        let device_idx = ctx.tho_device_idx();

        debug!("AsyncStage for ClcTask::run: device_idx={device_idx} input={input:?}");

        let output = OOMHandler::<NoInputAdjuster, NoTaskAdjuster, NoRunCtxAdjuster, LoadedRange>::new(vec![], None).run(
            &self.sync_clc,
            input,
            ctx,
            NoContexts
        ).await?;


        Ok(Arc::new(DevicesWrap::wrap(output)))
    }

}
