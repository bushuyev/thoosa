use super::{ApplyMotifNetInit, GlobalCtx};
use crate::dims::MotifProbability;
use crate::motif_tho_tensor::MotifThoTensor;
use crate::motifs_ws::MotifsWorkspace;
use crate::train_nets::MotifNetTypeParams;
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use parking_lot::RwLock;
use std::marker::PhantomData;
use std::sync::{Arc, LazyLock};
use tch::{Device::Cpu, IndexOp, Kind::Float, Tensor};
use thoosa_common::dims::ChannelsData;
use thoosa_common::instance_cache::InstanceCache;
use thoosa_common::ipc::cuda_worker::{AllTensors, WokerRunCtx, WorkerStageInput};
use thoosa_common::ipc::proxy::{DynSyncStageEnum, ProxyStage};
use thoosa_common::ipc::serialization::json::JsonIpcCodec;
use thoosa_common::ipc::worker_msg::WorkerInput;
use thoosa_common::ipc::wraps::{SyncStageIpcWrapper, WorkerWrapper};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep};
use thoosa_common::runner::blocking::BlockingPool;
use thoosa_common::runner::common::cachable::{BorrowRef, DevicesWrap, SelectRef, ThoDataRef};
use thoosa_common::runner::oom_handler::{NoInputAdjuster, NoStageAdjuster, OOMHandler, WithRunCtxContext};
use thoosa_common::runner::residency::ResidencyLease;
use thoosa_common::runner::sync_stage::SyncStage;
use thoosa_common::tch_error::tch_result;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::utils::pad_to;
use thoosa_named_derive::Named;
use thoosa_shared::ThoError;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::traits::{Adjustable, AsyncStage, MaxParallelCtx, OutId, RunCtx, Stage, WithId};
use thoosa_shims::inference_mode::InferenceModeGuard;

pub(crate) const APPLY_COSINE_GROUP_ENDPOINT: &str = "ApplyCosineMotifGroup";
pub(super) const MAX_COSINE_GROUP: usize = 32;

pub(super) type Probabilities = Arc<DevicesWrap<ThoTensor<MotifProbability, SynKey>>>;
pub(super) type Samples = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>;

static SCAN_INSTANCES: InstanceCache = InstanceCache::new();
static BANK_LOAD_POOL: LazyLock<BlockingPool> = LazyLock::new(|| BlockingPool::new(4));

#[derive(Debug, cached_derive::Cached, Named)]
pub(super) struct ApplyCosineGroup<C: RunCtx + MaxParallelCtx + Adjustable + 'static> {
    id: SynKey,
    spec: ScanSpec,
    bank: Probabilities,
    scan: Arc<DynSyncStageEnum<SyncApplyCosineGroup<C>>>,
    pub(super) models: Vec<ApplyMotifNetInit>,
}

impl<C: RunCtx + MaxParallelCtx + Adjustable + 'static> ApplyCosineGroup<C> {
    pub(super) async fn try_new(models: &[ApplyMotifNetInit], width: u32, device_idx: usize, ws: Arc<RwLock<Box<dyn MotifsWorkspace>>>) -> Result<Option<Arc<Self>>, ThoError> {
        let first = &models[0];
        let spec = ScanSpec { channel: first.channel, width, step: first.slice_step };
        let mut id = SynKey::new([thoosa_shared::mono_key!(APPLY_COSINE_GROUP_ENDPOINT, "cosine group")]);
        for model in models {
            id = &id + &model.inference_id();
        }
        let bank = BANK_LOAD_POOL
            .run_owned({
                let id = id.clone();
                let models = models.to_vec();
                move || load_bank(id, &models, spec.width, &ws)
            })
            .await;
        let bank = match bank {
            Ok(Some(bank)) => bank,
            Ok(None) => return Ok(None),
            Err(error) if error.is_oom() => return Ok(None),
            Err(error) => return Err(error),
        };
        let is_ipc = device_idx != 0 && GlobalCtx::global().is_ipc();
        let scan = SCAN_INSTANCES
            .get_or_try_insert_recreatable_with::<C, _, _, _, _>(device_idx, GlobalCtx::global().cuda_activity_gate(device_idx), is_ipc, (), move || async move {
                let stage = if is_ipc {
                    DynSyncStageEnum::Proxy { stage: Arc::new(ProxyStage::proxy_recreatable(APPLY_COSINE_GROUP_ENDPOINT, Some("0".into()), device_idx, Box::new(())).await?), init: () }
                } else {
                    DynSyncStageEnum::Local(Arc::new(SyncApplyCosineGroup(PhantomData)))
                };
                Ok(Arc::new(stage))
            })
            .await?;
        Ok(Some(Arc::new(Self { id, spec, bank, scan, models: models.to_vec() })))
    }
}

fn load_bank(id: SynKey, models: &[ApplyMotifNetInit], width: u32, ws: &RwLock<Box<dyn MotifsWorkspace>>) -> Result<Option<Probabilities>, ThoError> {
    let _inference = InferenceModeGuard::enabled()?;
    let mut rows = Vec::with_capacity(models.len());
    for model in models {
        let (params, module) = ws.read().load_motif_net(&model.motif_net_path, &Cpu, model.quantize, true)?;
        let Some(module) = module else { return Ok(None) };
        if !matches!(params.net_type, MotifNetTypeParams::CosineNetParams(_)) || params.num_features() != width {
            return Ok(None);
        }
        let Some(row) = pack_detector(module.var_store(), i64::from(width))? else { return Ok(None) };
        rows.push(row);
    }
    // CPU residency is the authoritative copy after offload or worker replacement.
    // One arena copy supplies all normalized kernels and affine parameters.
    let bank = tch_result(pack_bank(&rows, i64::from(width)))?;
    Ok(Some(Arc::new(DevicesWrap::wrap(ThoTensor::new(id, bank)?))))
}

fn pack_bank(rows: &[Tensor], width: i64) -> Result<Tensor, tch::TchError> {
    // Store contiguous kernels first, then the two affine parameter arrays.
    // This keeps the upload as one tensor and avoids repacking each invocation.
    let rows = Tensor::f_stack(rows, 0)?;
    Tensor::f_cat(&[
        rows.f_narrow(1, 0, width)?.f_reshape([-1])?,
        rows.f_select(1, width)?,
        rows.f_select(1, width + 1)?,
    ], 0)
}

fn pack_detector(store: &tch::nn::VarStore, width: i64) -> Result<Option<Tensor>, ThoError> {
    let variables = store.variables();
    // These names are produced by the two-node cosine_motif_node sequence.
    // Check shapes, precision and freezing before bypassing the module forward.
    let (Some(kernel), Some(alpha), Some(beta)) = (
        variables.get("seq_1.cosine_bank.kernel"),
        variables.get("seq_1.cosine_bank.alpha"),
        variables.get("seq_1.cosine_bank.beta"),
    ) else {
        return Ok(None);
    };
    if variables.len() != 3 || kernel.size() != [1, 1, width] || alpha.dim() != 0 || beta.dim() != 0
        || [kernel, alpha, beta].iter().any(|tensor| tensor.kind() != Float || tensor.requires_grad())
    {
        return Ok(None);
    }
    tch_result((|| {
        let norm = kernel.f_mul(kernel)?.f_sum_dim_intlist(&[1i64, 2][..], true, Float)?.f_sqrt()?.f_clamp_min(1e-12)?;
        let unit = kernel.f_div(&norm)?.f_view([width])?;
        Ok(Some(Tensor::f_cat(&[unit, alpha.f_view([1])?, beta.f_view([1])?], 0)?))
    })())
}

impl<C: RunCtx + MaxParallelCtx + Adjustable + 'static> Stage for ApplyCosineGroup<C> {
    type Ctx = C;
    type In<'a> = Samples;
    type Out = Probabilities;
}

impl<C: RunCtx + MaxParallelCtx + Adjustable + 'static> OutId for ApplyCosineGroup<C> {
    fn out_id(&self, input: &SynKey) -> SynKey {
        &self.id + input
    }
}

impl<C: RunCtx + MaxParallelCtx + Adjustable + 'static> AsyncStage for ApplyCosineGroup<C> {
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "ApplyCosineGroup::run"))]
    async fn run(self: Arc<Self>, input: Samples, ctx: C) -> Result<Probabilities, ThoError> {
        let input = Arc::new(ScanInputWrapper { id: self.out_id(input.id()), spec: self.spec, samples: input, bank: Arc::clone(&self.bank) });
        let oom: OOMHandler<'_, '_, NoInputAdjuster, NoStageAdjuster, _, ()> = OOMHandler::throttle_and_offload(MaxParallelStep::FAST, "apply_cosine_group");
        let output = oom.run(&self.scan, input, ctx, WithRunCtxContext::new(MaxParallelAndDumpToCpuAdjustableContex::default())).await?;
        Ok(Arc::new(DevicesWrap::wrap(output.into_inner())))
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct ScanSpec {
    channel: u32,
    width: u32,
    step: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct ScanInput<'a> {
    id: SynKey,
    spec: ScanSpec,
    samples: ThoDataRef<'a, ThoTensor<ChannelsData, SynKey>>,
    bank: ThoDataRef<'a, ThoTensor<MotifProbability, SynKey>>,
}

#[typetag::serde(name = "CosineMotifGroupInput")]
impl WorkerInput for ScanInput<'static> {}

impl WorkerStageInput for ScanInput<'_> {
    fn clone_refs(&self) -> Result<Box<dyn WorkerInput>, ThoError> {
        Ok(Box::new(ScanInput {
            id: self.id.clone(),
            spec: self.spec,
            samples: ThoDataRef::Owned(self.samples.as_ref().clone_ref()?),
            bank: ThoDataRef::Owned(self.bank.as_ref().clone_ref()?),
        }))
    }

    fn ref_to_tensor(&self, tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {
        Ok(Self {
            id: self.id.clone(),
            spec: self.spec,
            samples: ThoDataRef::Owned(self.samples.as_ref().ref_to_tensor(tensors, remove)?),
            bank: ThoDataRef::Owned(self.bank.as_ref().ref_to_tensor(tensors, remove)?),
        })
    }
}

impl WithId for ScanInput<'_> {
    type ID = SynKey;

    fn id(&self) -> &SynKey {
        &self.id
    }
}

impl thoosa_shared::applicables_runner::Input for ScanInput<'_> {}

#[derive(Debug)]
struct ScanInputWrapper {
    id: SynKey,
    spec: ScanSpec,
    samples: Samples,
    bank: Probabilities,
}

impl BorrowRef for ScanInputWrapper {
    type Out<'a> = ScanInput<'a>;

    fn protect(&self, device: usize) -> ResidencyLease {
        ResidencyLease::combine([self.samples.protect(device), self.bank.protect(device)])
    }

    async fn borrow_ref<'a>(&'a self, device: usize) -> Result<Self::Out<'a>, ThoError>
    where
        Self: 'a,
    {
        Ok(ScanInput {
            id: self.id.clone(),
            spec: self.spec,
            samples: self.samples.borrow_ref(device).await?,
            bank: self.bank.borrow_ref(device).await?,
        })
    }
}

impl SelectRef<()> for ScanInputWrapper {
    async fn select_ref<'a>(&'a self, device: usize, _selector: ()) -> Result<Self::Out<'a>, ThoError>
    where
        (): 'a,
        Self: 'a,
    {
        self.borrow_ref(device).await
    }
}

#[derive(Debug)]
struct SyncApplyCosineGroup<C>(PhantomData<C>);

impl<C: RunCtx + 'static> Adjustable for SyncApplyCosineGroup<C> {}

impl<C: RunCtx + 'static> Stage for SyncApplyCosineGroup<C> {
    type Ctx = C;
    type In<'a> = ScanInput<'a>;
    type Out = MotifThoTensor<MotifProbability>;
}

impl<C: RunCtx + 'static> SyncStage for SyncApplyCosineGroup<C> {
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "SyncApplyCosineGroup::do_run"))]
    fn do_run<'a>(self: Arc<Self>, input: &ScanInput<'a>, _ctx: &C) -> Result<Self::Out, ThoError> {
        let _inference = InferenceModeGuard::enabled()?;
        let samples = input.samples.as_ref().local()?;
        let bank = input.bank.as_ref().local()?;
        let scores = scan(&samples, &bank, input.spec)?;
        Ok(ThoTensor::new(input.id.clone(), scores)?.into())
    }
}

fn scan(samples: &Tensor, bank: &Tensor, spec: ScanSpec) -> Result<Tensor, ThoError> {
    let width = i64::from(spec.width);
    let step = i64::from(spec.step);
    let signal = tch_result(samples.f_i((i64::from(spec.channel), ..)))?;
    let signal = if signal.size()[0] < width { pad_to(&signal, &[width], 0.0)? } else { signal };
    tch_result((|| {
        let signal = signal.f_to_kind(Float)?;
        // Square the signal before unfolding so overlapping windows share the
        // same squared samples as well as the same norms across detectors.
        let windows = signal.f_mul(&signal)?.f_unfold(0, width, step)?;
        let norms = windows.f_sum_dim_intlist(&[1i64][..], false, Float)?.f_sqrt()?.f_clamp_min(1e-12)?;
        // Strided convolution computes the dot products without unfolding the
        // signal for each kernel. Every row shares the same norms.
        let detectors = bank.size()[0] / (width + 2);
        let kernels_len = detectors * width;
        let kernels = bank.f_narrow(0, 0, kernels_len)?.f_view([detectors, 1, width])?;
        let alpha = bank.f_narrow(0, kernels_len, detectors)?.f_unsqueeze(1)?;
        let beta = bank.f_narrow(0, kernels_len + detectors, detectors)?.f_unsqueeze(1)?;
        let dots = signal.f_view([1, 1, -1])?.f_conv1d(&kernels, None::<&Tensor>, [step], [0], [1], 1)?.f_squeeze_dim(0)?;
        dots.f_div(&norms)?.f_mul(&alpha)?.f_add(&beta)
    })())
}

pub(crate) fn apply_cosine_group_init<'a>(
    _stage_id: Option<String>,
    _init: Box<dyn WorkerInput>,
    _ws: Arc<RwLock<Box<dyn MotifsWorkspace>>>,
    _tensors: &'a AllTensors,
    _device: &'a ThoDevice,
) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move { Ok(Box::new(WorkerWrapper::new(Arc::new(SyncApplyCosineGroup::<WokerRunCtx>(PhantomData)), PhantomData::<JsonIpcCodec>)) as Box<dyn SyncStageIpcWrapper>) }.boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::apply_motif_nets::apply_motif_net_forward;
    use crate::stages::load_motif_net::MotifNetMeta;
    use crate::train_nets::MotifNetParams;
    use thoosa_common::data_with_extra::DataWithExtra;
    use thoosa_common::nets::common_nets::ThoModuleWithStore;
    use thoosa_common::nets::nodes::{InitDummy, InitMode};
    use thoosa_shared::mono_key;

    #[test]
    fn test_group_matches_individual_cosine_detectors() -> anyhow::Result<()> {
        let _inference = InferenceModeGuard::enabled()?;
        let mut packed = Vec::new();
        let mut detectors = Vec::new();
        for (index, weights) in [[1.0f32, -0.5, 0.25, 2.0], [0.0; 4], [-1.0, 0.25, 0.5, -2.0]].into_iter().enumerate() {
            let id = mono_key!(index as u32, "detector");
            let params = MotifNetParams::new_cosine(4, 4.0 + index as f64, -0.5 * index as f64, InitMode::Mean(InitDummy {}));
            let mut store = tch::nn::VarStore::new(Cpu);
            let module = params.motif_net(None)?.build_tch_module(&store.root())?;
            store.variables().get_mut("seq_1.cosine_bank.kernel").expect("cosine kernel").f_copy_(&Tensor::f_from_slice(&weights)?.f_view([1, 1, 4])?)?;
            store.freeze();
            packed.push(pack_detector(&store, 4)?.expect("frozen cosine detector"));
            detectors.push(DataWithExtra::new(ThoModuleWithStore::with_factory(id, module, store, None), MotifNetMeta::new(index as u32, 1, params)));
        }
        let bank = pack_bank(&packed, 4)?;
        // These cases cover short/zero input, ordinary windows, and unused tail samples.
        for samples in [&[0.0f32, 0.0][..], &[1.0, -4.0, 2.0, 0.0, 5.0, 7.0, -1.0, 6.0, 9.0]] {
            let signal = Tensor::f_from_slice(samples)?;
            let batch = Tensor::f_stack(&[signal.f_zeros_like()?, signal], 0)?;
            let id = SynKey::new([mono_key!("sample", "sample")]);
            let input = ThoTensor::new(id.clone(), batch)?;
            for step in [1, 3, 5] {
                let grouped = scan(&*input.local()?, &bank, ScanSpec { channel: 1, width: 4, step })?;
                for (row, detector) in detectors.iter().enumerate() {
                    let expected = apply_motif_net_forward(id.clone(), &input, detector, step, &id)?;
                    assert!(grouped.f_get(row as i64)?.f_allclose(&*expected.local()?, 1e-5, 1e-6, false)?, "step={step} row={row} samples={samples:?}");
                }
            }
        }
        Ok(())
    }
}
