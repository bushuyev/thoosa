use cached_derive::Cached;
use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use thoosa_named_derive::Named;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::traits::{Adjustable, Data, MonoMetrics, Select, SimpleId, WithId};
use tokio::sync::Mutex;
use tch::{IndexOp, Kind, Tensor};
use log::debug;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::{mono_key, CopyToDevice, ThoError};
use crate::copy_ctx::CudaActorHandle;
use crate::dims::Dims;
use crate::ipc::cuda_process::encode_endpoint_payload;
use crate::runner::common::cachable::{DevicesWrap, WithDataRef};
use crate::runner::common::{OutId, RunCtx};
use crate::runner::common::runnable::{AsyncStage, DynSyncStageBox, Stage, SyncStage};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::runner::test_utils::parse_numeric_part;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use crate::torch_runner::tests;
use crate::utils::lin_for_size;
use crate as thoosa_common;
use crate::torch_runner::tests::load_and_clc::{kind_from_worker_name, kind_to_worker_name, ClcTaskOutput, MulK, WorkerStageInput, SYNC_TEST_TENSOR_LOADER_ENDPOINT};

//////////////////LOADER
#[derive(Debug, Clone, Eq, PartialEq, Hash, Named)]
pub struct LoaderInputId(&'static str);

impl Serialize for LoaderInputId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.0)
    }
}

impl<'de> Deserialize<'de> for LoaderInputId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let id = String::deserialize(deserializer)?;
        Ok(Self(Box::leak(id.into_boxed_str())))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, DummyData)]
pub struct LoaderInput {
    id: LoaderInputId,
}

impl Input for LoaderInput {  }

impl WithId for LoaderInput {
    type ID = LoaderInputId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl From<&'static str> for LoaderInput {
    fn from(value: &'static str) -> Self {
        LoaderInput {
            id: LoaderInputId(value),
        }
    }
}


#[derive(Debug, Clone)]
pub struct LoadedTensor;

impl Dims for LoadedTensor {
    type Metrics = MonoMetrics<1, 2>;

}

impl LoadedTensor {}

impl Select<LoadedRange> for ThoTensor<LoadedTensor, SynKey> {
    type Output = ThoTensor<LoadedTensor, SynKey>;

    fn subview(&self, selector: LoadedRange) -> Result<Self::Output, ThoError> {
        let ranges = MonoMetrics::<1,2>::map_to_ranges(&selector.range.start, &selector.range.end)?;//TODO unwrap

        let sub_tensor = self.t_ref().i((ranges[0].clone(), ranges[1].clone()));

        let new_id = self.id().clone(); //SimpleId::from(format!("{}:{ranges:?}", self.id().0));

        debug!("Select for ThoTensor2<LoadedTensor, SimpleId>: ranges={ranges:?} sub_tensor={sub_tensor:?}");

        ThoTensor::new(new_id, sub_tensor)
    }
}


#[derive(Debug, Clone, Default)]
pub(crate) struct  LoadedRange {
    range:std::ops::Range<MonoMetrics<1, 2>>
}

impl Adjustable for LoadedRange {}

impl From<std::ops::Range<MonoMetrics<1, 2>>> for LoadedRange {
    fn from(value: std::ops::Range<MonoMetrics<1, 2>>) -> Self {
        Self { range: value }
    }
}

impl WorkerStageInput for LoaderInput {
    fn decode_for_worker(input: String, _all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError> {
        Ok(serde_json::from_str(&input)?)
    }
}

impl LoadedTensor {
    pub(crate) fn clc_hadamard(
        id: SynKey,
        loaded: &ThoTensor<LoadedTensor, SynKey>,
        k: &ThoTensor<MulK, SimpleId>,
    ) -> Result<ThoTensor<ClcTaskOutput, SynKey>, ThoError> {

        ThoTensor::new( id,loaded.t_ref().multiply(&k.as_ref()))

    }

    pub(crate) fn clc_sum(
        id: SynKey,
        loaded: &ThoTensor<LoadedTensor, SynKey>,
        k: &ThoTensor<MulK, SimpleId>,
    ) -> Result<ThoTensor<ClcTaskOutput, SynKey>, ThoError> {
        debug!("LoadedTensor::clc_sum: loaded={:?} k.t={:?}",loaded.t_ref(),k.t_ref());

        let multed = loaded.t_ref().f_multiply(&k.as_ref());

        debug!("LoadedTensor::clc_sum: multed={multed:?}");

        let sum = multed?
            .f_sum_dim_intlist(1, false, loaded.t_ref().kind());

        debug!("LoadedTensor::clc_sum: sum={sum:?}");

        let tensor = sum?
            .view((-1, 1));
        ThoTensor::new(
            id,
            tensor
        )
    }
}

#[derive(Debug, Cached, Named)]
#[cache_name = "LoadedTensorId"]
pub struct TestTensorLoader {
    sync_loader: Arc<LoaderDynSyncStage>
}

impl TestTensorLoader {
    pub fn new(size: MonoMetrics<1, 2>, kind: Kind) -> Self {
        Self {
            sync_loader: Arc::new(SyncTestTensorLoader::new(size, kind))
        }
    }
}

impl OutId for TestTensorLoader {
    fn out_id(
        &self,
        input_id: &<Self::In<'static> as WithId>::ID,
    ) -> <Self::Out as WithId>::ID {
        debug!("TestTensorLoader::out_id: input_id={input_id:?}");
        self.sync_loader.out_id(input_id)
    }
}

impl Stage for TestTensorLoader {
    type In<'a> = LoaderInput;
    type Out = ThoTensor<LoadedTensor, SynKey>;
    type Ctx = RunCtxImpl;
}

// #[async_trait]
impl AsyncStage for TestTensorLoader {
    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let sync_loader = Arc::clone(&self.sync_loader);

        let r=    input.with_data_ref(ctx.tho_device_idx(), ctx, None, |input, ctx|{
            Ok(sync_loader.do_run(input, ctx))
        }).await??;

        Ok(Arc::new(DevicesWrap::wrap(r)))

    }

}

impl OutId for SyncTestTensorLoader {
    fn out_id(
        &self,
        input_id: &<Self::In<'static> as WithId>::ID,
    ) -> <Self::Out as WithId>::ID {
        debug!("SyncTestTensorLoader::out_id: input_id={input_id:?}");
        SynKey::new([mono_key!(input_id.0, "resource key")])
    }
}

#[derive(Debug)]
pub struct SyncTestTensorLoader {
    size: MonoMetrics<1, 2>,
    kind: Kind,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SyncTestTensorLoaderInit {
    size: [i64; 2],
    kind: String,
}

impl SyncTestTensorLoaderInit {
    fn new(size: MonoMetrics<1, 2>, kind: Kind) -> Self {
        Self {
            size: (&size).into(),
            kind: kind_to_worker_name(kind).to_string(),
        }
    }
}

impl SyncTestTensorLoader {
    pub fn new(size: MonoMetrics<1, 2>, kind: Kind) -> LoaderDynSyncStage {
        if GlobalCtx::global().is_ipc() {
            let proxy = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(ProxySyncTestTensorLoader::new(size, kind))
            }).expect("SyncTestTensorLoader::new: install IPC proxy loader");
            LoaderDynSyncStage::new(Arc::new(proxy))
        } else {
            LoaderDynSyncStage::new(Arc::new(Self::new_local(size, kind)))
        }
    }

    fn new_local(size: MonoMetrics<1, 2>, kind: Kind) -> Self {
        Self { size, kind }
    }

    pub(crate) fn init(init: SyncTestTensorLoaderInit) -> Result<Self, ThoError> {
        Ok(Self::new_local(init.size.into(), kind_from_worker_name(&init.kind)?))
    }
}

#[derive(Debug)]
struct ProxySyncTestTensorLoader {
    size: MonoMetrics<1, 2>,
}

impl ProxySyncTestTensorLoader {
    async fn new(size: MonoMetrics<1, 2>, kind: Kind) -> Result<Self, ThoError> {
        debug!("ProxySyncTestTensorLoader::new: size={size:?} kind={kind:?}");
        GlobalCtx::global().cuda_actor(1)
            .install_endpoint(encode_endpoint_payload(SYNC_TEST_TENSOR_LOADER_ENDPOINT, &serde_json::to_string(&SyncTestTensorLoaderInit::new(size.clone(), kind))?)).await?;

        Ok(Self {
            size,
        })
    }
}

impl Stage for ProxySyncTestTensorLoader {
    type In<'a> = LoaderInput;
    type Out = ThoTensor<LoadedTensor, SynKey>;
    type Ctx = RunCtxImpl;
}

impl OutId for ProxySyncTestTensorLoader {
    fn out_id(
        &self,
        input_id: &<Self::In<'static> as WithId>::ID,
    ) -> <Self::Out as WithId>::ID {
        debug!("ProxySyncTestTensorLoader::out_id: input_id={input_id:?}");
        SynKey::new([mono_key!(input_id.0, "resource key")])
    }
}

impl SyncStage for ProxySyncTestTensorLoader {
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        debug!("ProxySyncTestTensorLoader::do_run: input={input:?} size={:?}", self.size);

        tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(async move {

            let response = GlobalCtx::global().cuda_actor(ctx.tho_device_idx()).invoke_endpoint(encode_endpoint_payload(SYNC_TEST_TENSOR_LOADER_ENDPOINT, &serde_json::to_string(input)?)).await?;

            serde_json::from_str::<ThoTensor<LoadedTensor, SynKey>>(&response).map_err(ThoError::from)
        }))
    }
}

impl Stage for SyncTestTensorLoader {
    type In<'a> = LoaderInput;
    type Out = ThoTensor<LoadedTensor, SynKey>;
    type Ctx = RunCtxImpl;
}

impl SyncStage for SyncTestTensorLoader {
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let k = parse_numeric_part(WithId::id(input).0).unwrap_or(1.0);

        let device_idx = ctx.tho_device_idx();


        let tensor = ThoTensor::new(
            self.out_id(WithId::id(input)),
            lin_for_size(
                k,
                (&self.size).into(),
                self.kind,
                &ThoDevice::from(device_idx),
            ),
        )?;

        Ok(tensor)
    }
}

type LoaderDynSyncStage = DynSyncStageBox<LoaderInput, ThoTensor<LoadedTensor, SynKey>, RunCtxImpl>;