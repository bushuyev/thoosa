use thoosa_common::tch_error::tch_result;
use anyhow::Context as _;
use crate::dims::{Batch, Dims};
#[cfg(test)]
use crate::dims::AnyDims;
use crate::copy_ctx::{complete_cuda_stream, finish_cuda_run};
#[cfg(test)]
use crate::nets::builder::{Node, SeqBuilder};
use crate::nets::nodes::ThoModule;
use crate::nets::state_transfer::{StateTensors, VarStoreState};
use thoosa_common::runner::sync_stage::StageStateInfo;
use crate::runner::global_ctx::GlobalCtx;
use crate::tho_device::ThoDevice;
#[cfg(test)]
use crate::tho_tensor::ThoTensor;
use async_trait::async_trait;
use log::debug;
use std::fmt::{Debug, Formatter};
use std::pin::Pin;
use std::sync::Arc;
use tch::Device::Cpu;
use tch::nn::{Optimizer, OptimizerConfig, VarStore};
use tch::{Device, Tensor, nn, no_grad};
#[cfg(test)]
use tch::Kind;
use thoosa_shared::{panic_to_err, tho_error, MoveToDevice, ThoError};
use thoosa_shims::stream::ThoCudaStream;
use thoosa_shims::optimizers::{AdamGroupSnapshot, AdamOptimizerSnapshot, AdamParamStateSnapshot, AdamWOptimizerSnapshot, SgdGroupSnapshot, SgdOptimizerSnapshot, SgdParamStateSnapshot, adam_snapshot, adam_w_snapshot, restore_adam_snapshot, restore_adam_w_snapshot, restore_sgd_snapshot, sgd_snapshot};
use tokio::task::JoinSet;


#[path = "training_metadata.rs"]
mod training_metadata;
use training_metadata::{TrainingMetadata, OptimizerMetadata};

pub struct ModConfig<DI, DO>
where
    DI: Dims, DO: Dims
{
    optimizer_builder: Box<dyn Fn(&VarStore) -> anyhow::Result<Pin<Box<Optimizer>>> + Send + Sync>,
    net: Box<dyn Fn(&VarStore) -> anyhow::Result<Box<dyn ThoModule<Batch<DI>, Batch<DO>>>> + Send + Sync>,
    opt_config: ThoOptimizers
}

impl <DI, DO> Debug for ModConfig<DI, DO>
where
    DI: Dims, DO: Dims
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModConfig")
            .finish()
    }
}

impl<DI, DO> ModConfig<DI, DO>
where
    DI: Dims,
    DO: Dims,
{
    /*
     * Erase the concrete optimizer configuration so owners such as TrainStep2
     * do not need to add the optimizer type to their public generic contract.
     */
    pub fn new<NB>(opt_config: ThoOptimizers, net: NB, lr: f64) -> Self
    where
        NB: Fn(&VarStore) -> anyhow::Result<Box<dyn ThoModule<Batch<DI>, Batch<DO>>>> + Send + Sync + 'static,
    {
        let opt_config_for_builder = opt_config.clone();
        Self {
            optimizer_builder: Box::new( move |var_store| Ok(Box::pin(opt_config_for_builder.build(var_store, lr)?))),
            net: Box::new(net),
            opt_config
        }
    }
}

#[derive(Debug, Clone)]
pub enum ThoOptimizers {
    Adam {config:nn::Adam},
    AdamW{config: nn::AdamW},
    Sgd {config: nn::Sgd},
    //TODO other optimizers
}

impl ThoOptimizers {
    fn build(&self, vs:&VarStore, lr: f64) -> anyhow::Result<Optimizer>
    {
        Ok(match self {
            ThoOptimizers::Adam { config } => config.build(vs, lr)?,
            ThoOptimizers::AdamW { config } => config.build(vs, lr)?,
            ThoOptimizers::Sgd { config } => config.build(vs, lr)?,
        })
    }

    unsafe fn snapshot(&self, optimizer: *const torch_sys::C_optimizer, tho_device_idx: usize) -> anyhow::Result<OptimizerSnapshot> {
        // The configuration is the authoritative optimizer type because tch erases it behind Optimizer.
        Ok(match self {
            ThoOptimizers::Adam { .. } => OptimizerSnapshot::Adam(unsafe { adam_snapshot(optimizer, tho_device_idx)? }),
            ThoOptimizers::AdamW { .. } => OptimizerSnapshot::AdamW(unsafe { adam_w_snapshot(optimizer, tho_device_idx)? }),
            ThoOptimizers::Sgd { .. } => OptimizerSnapshot::Sgd(unsafe { sgd_snapshot(optimizer, tho_device_idx)? }),
        })
    }

    unsafe fn restore_snapshot(&self, snapshot: &OptimizerSnapshot, optimizer: *mut torch_sys::C_optimizer) -> anyhow::Result<()> {
        match (self, snapshot) {
            (ThoOptimizers::Adam { .. }, OptimizerSnapshot::Adam(snapshot)) => Ok(unsafe { restore_adam_snapshot(snapshot, optimizer) }?),
            (ThoOptimizers::AdamW { .. }, OptimizerSnapshot::AdamW(snapshot)) => Ok(unsafe { restore_adam_w_snapshot(snapshot, optimizer) }?),
            (ThoOptimizers::Sgd { .. }, OptimizerSnapshot::Sgd(snapshot)) => Ok(unsafe { restore_sgd_snapshot(snapshot, optimizer) }?),
            (config, snapshot) => Err(anyhow::anyhow!("optimizer configuration {config:?} does not match snapshot {snapshot:?}")),
        }
    }
}

impl From<nn::Adam> for ThoOptimizers {
    fn from(config: nn::Adam) -> Self {
        ThoOptimizers::Adam { config }
    }
}

impl From<nn::AdamW> for ThoOptimizers {
    fn from(config: nn::AdamW) -> Self {
        ThoOptimizers::AdamW { config }
    }
}

impl From<nn::Sgd> for ThoOptimizers {
    fn from(config: nn::Sgd) -> Self {
        ThoOptimizers::Sgd { config }
    }
}



#[derive(Debug)]
pub struct ThoTrainModule<DI, DO>
where
    DI: Dims, DO: Dims
{
    config:ModConfig<DI, DO>,
    opt_state:OptSnapshot<DI, DO>,
    ipc_export: Option<TrainingState>,
    transaction_failed: bool
}

#[derive(Debug)]
enum OptSnapshot<DI, DO>
where
    DI: Dims, DO: Dims
{
    Opt{opt:Pin<Box<Optimizer>>, vs:VarStore, module:Box<dyn ThoModule<Batch<DI>, Batch<DO>>>},
    Snapshot{ cpu_snap: OptimizerSnapshot, vs:VarStore},
    Offloaded(TrainingState)
}

impl <DI, DO> OptSnapshot<DI, DO>
where
    DI: Dims, DO: Dims
{
    fn device(&self) -> ThoDevice {
        match self {
            OptSnapshot::Opt { vs,.. }| OptSnapshot::Snapshot { vs,.. } => {
                ThoDevice::try_from(vs.device()).expect("CPU/CUDA model device")
            }
            OptSnapshot::Offloaded(_) => thoosa_common::tho_device::ThoDevice::from(0usize),
        }
    }
}

#[derive(Debug)]
enum OptimizerSnapshot {
    Adam(AdamOptimizerSnapshot),
    AdamW(AdamWOptimizerSnapshot),
    Sgd(SgdOptimizerSnapshot),
}

#[derive(Debug)]
struct TrainingState {
    variables: VarStoreState,
    tensors: StateTensors,
    optimizer: OptimizerSnapshot,
}

impl OptimizerSnapshot {
    fn map_tensors(&self, device: usize, mut map: impl FnMut(&str, &Tensor) -> Result<Tensor, ThoError>) -> Result<Self, ThoError> {
        fn adam(snapshot: &AdamOptimizerSnapshot, device: usize, map: &mut impl FnMut(&str, &Tensor) -> Result<Tensor, ThoError>) -> Result<AdamOptimizerSnapshot, ThoError> {
            let mut groups = Vec::new();
            for (g, group) in snapshot.groups.iter().enumerate() {
                let mut states = Vec::new();
                for (p, state) in group.states.iter().enumerate() {
                    states.push(state.as_ref().map(|state| Ok::<_, ThoError>(AdamParamStateSnapshot {
                        step: state.step,
                        exp_avg: map(&format!("opt/{g}/{p}/exp_avg"), &state.exp_avg)?,
                        exp_avg_sq: map(&format!("opt/{g}/{p}/exp_avg_sq"), &state.exp_avg_sq)?,
                        max_exp_avg_sq: state.max_exp_avg_sq.as_ref().map(|tensor| map(&format!("opt/{g}/{p}/max_exp_avg_sq"), tensor)).transpose()?,
                    })).transpose()?);
                }
                groups.push(AdamGroupSnapshot { options: group.options, states });
            }
            Ok(AdamOptimizerSnapshot { defaults: snapshot.defaults, groups, tho_device_idx: device })
        }
        Ok(match self {
            Self::Adam(snapshot) => Self::Adam(adam(snapshot, device, &mut map)?),
            Self::AdamW(snapshot) => Self::AdamW(adam(snapshot, device, &mut map)?),
            Self::Sgd(snapshot) => {
                let mut groups = Vec::new();
                for (g, group) in snapshot.groups.iter().enumerate() {
                    let mut states = Vec::new();
                    for (p, state) in group.states.iter().enumerate() {
                        states.push(state.as_ref().map(|state| map(&format!("opt/{g}/{p}/momentum"), &state.momentum_buffer).map(|momentum_buffer| SgdParamStateSnapshot { momentum_buffer })).transpose()?);
                    }
                    groups.push(SgdGroupSnapshot { options: group.options, states });
                }
                Self::Sgd(SgdOptimizerSnapshot { defaults: snapshot.defaults, groups, tho_device_idx: device })
            }
        })
    }
}

impl <DI, DO> ThoTrainModule<DI, DO>
where
    DI: Dims, DO: Dims
{
    pub fn new_on_device(config: ModConfig<DI, DO>, device: ThoDevice) -> anyhow::Result<Self> {
        let (module, var_store) = Self::init_opt(&config.net, device)?;
        let opt = (config.optimizer_builder)(&var_store)?;
        Ok(Self {
            config,
            opt_state: OptSnapshot::Opt { opt, vs: var_store, module },
            ipc_export: None,
            transaction_failed: false,
        })
    }

    fn init_opt(net: &dyn Fn(&VarStore) -> anyhow::Result<Box<dyn ThoModule<Batch<DI>, Batch<DO>>>>, device: ThoDevice) -> anyhow::Result<(Box<dyn ThoModule<Batch<DI>, Batch<DO>>>, VarStore)> {
        let var_store = nn::VarStore::new(device.into());
        let module = net(&var_store)?;
        Ok((module, var_store))
    }

    pub fn opt(&mut self) ->&mut Optimizer {
        match &mut self.opt_state {
            OptSnapshot::Opt { opt, .. } => {
                opt
            }
            OptSnapshot::Snapshot { .. } | OptSnapshot::Offloaded(_) => {
                panic!("wrong state")
            }
        }
    }

    pub fn module(&mut self) ->&mut Box<dyn ThoModule<Batch<DI>, Batch<DO>>> {
        match &mut self.opt_state {
            OptSnapshot::Opt { module, .. } => {
                module
            }
            OptSnapshot::Snapshot { .. } | OptSnapshot::Offloaded(_) => {
                panic!("wrong state")
            }
        }
    }

    pub fn into_var_store(self) -> anyhow::Result<VarStore> {
        match self.opt_state {
            OptSnapshot::Opt { opt, vs, module } => {
                /*
                 * Drop all other owners of the optimizer variables before
                 * returning the VarStore to the caller.
                 */
                drop(module);
                drop(opt);
                Ok(vs)
            }
            OptSnapshot::Snapshot { vs, .. } => Ok(vs),
            OptSnapshot::Offloaded(_) => Err(anyhow::anyhow!("training state is offloaded")),
        }
    }

    pub fn save_var_store_to_stream(&self, output: &mut dyn std::io::Write) -> anyhow::Result<()> {
        match &self.opt_state {
            OptSnapshot::Opt { vs, .. } | OptSnapshot::Snapshot { vs, .. } => Ok(vs.save_to_stream(output)?),
            OptSnapshot::Offloaded(_) => Err(anyhow::anyhow!("training state is offloaded")),
        }
    }

    pub fn save_var_store<T>(&self, path:T) -> anyhow::Result<()>
    where
        T:AsRef<std::path::Path> + Debug
    {
        debug!("ThoTrainModule::save_var_store: path={path:?} self.opt_state={:?}", self.opt_state);

        match &self.opt_state {
            OptSnapshot::Opt { vs, .. } | OptSnapshot::Snapshot { vs, .. } => {
                Ok(panic_to_err!(vs.save(&path))?)
            },
            OptSnapshot::Offloaded(_) => Err(anyhow::anyhow!("training state is offloaded")),
        }

    }

    pub fn has_gpu_data(&self, device: usize) -> bool {
        device != 0 && match &self.opt_state {
            OptSnapshot::Offloaded(state) => state.tensors.has_device(device),
            _ => self.opt_state.device().tho_device_idx() == device,
        }
    }

    fn capture_state(&self) -> Result<TrainingState, ThoError> {
        let (vs, snapshot) = match &self.opt_state {
            OptSnapshot::Opt { opt, vs, .. } => (vs, unsafe { self.config.opt_config.snapshot(opt.as_raw_c_optimizer(), ThoDevice::try_from(vs.device())?.tho_device_idx())? }),
            _ => return Err(ThoError::new("training state is not ready for IPC export")),
        };
        let mut tensors = StateTensors::default();
        let variables = VarStoreState::capture(vs, &mut tensors)?;
        // Separate scalar metadata from tensor storage so checkpoint restoration
        // recovers options, absent-state markers, and counters from the same epoch.
        let optimizer = snapshot.map_tensors(0, |key, tensor| {
            tensors.insert(key.to_owned(), tch_result(tensor.f_detach())?)?;
            Ok(Tensor::new())
        })?;
        Ok(TrainingState { variables, tensors, optimizer })
    }

    pub fn state_info(&mut self) -> Result<StageStateInfo, ThoError> {
        self.ipc_export = Some(self.capture_state()?);
        let state = self.ipc_export.as_ref().expect("captured state");
        let mut info = state.tensors.info();
        info.scalars = Some(Box::new(TrainingMetadata { variables: state.variables.clone(), optimizer: OptimizerMetadata::capture(&state.optimizer) }));
        Ok(info)
    }

    pub(crate) fn reset_state(&mut self, mut info: StageStateInfo) -> Result<(), ThoError> {
        let tensors = StateTensors::from_info(&info);
        let metadata = info.scalars.take().ok_or_else(|| ThoError::new("training checkpoint lacks metadata"))?;
        let metadata = (metadata as Box<dyn std::any::Any>).downcast::<TrainingMetadata>().map_err(|_| ThoError::new("invalid training checkpoint metadata"))?;
        // Replace the entire failed state; never export or reuse partial optimizer updates.
        self.ipc_export = None;
        self.transaction_failed = true;
        self.opt_state = OptSnapshot::Offloaded(TrainingState { variables: metadata.variables, tensors, optimizer: metadata.optimizer.into_snapshot() });
        Ok(())
    }

    pub fn state_tensor(&mut self, key: &str) -> Result<Tensor, ThoError> {
        if self.ipc_export.is_none() { self.ipc_export = Some(self.capture_state()?); }
        self.ipc_export.as_ref().expect("captured state").tensors.get(key)
    }

    pub fn drop_gpu(&mut self, device: usize) -> Result<(), ThoError> {
        if !self.has_gpu_data(device) { return Ok(()); }
        if let OptSnapshot::Offloaded(state) = &mut self.opt_state {
            state.tensors.clear();
            return Ok(());
        }
        let mut state = match self.ipc_export.take() {
            Some(state) => state,
            None => self.capture_state()?,
        };
        if state.tensors.info().tensors.is_empty() { return Ok(()); }
        state.tensors.clear();
        // Drop the optimizer, module and store together; keeping any one of
        // them alive would keep its parameter storage allocated on the GPU.
        self.opt_state = OptSnapshot::Offloaded(state);
        Ok(())
    }

    pub fn set_state_tensor(&mut self, key: &str, tensor: Tensor, device: usize) -> Result<(), ThoError> {
        let OptSnapshot::Offloaded(state) = &mut self.opt_state else {
            return Err(ThoError::new("training state is already loaded"));
        };
        state.tensors.set(key, tensor, device)?;
        if state.tensors.complete() {
            let vs = state.variables.restore(&state.tensors, device)?;
            let module = (self.config.net)(&vs)?;
            let opt = (self.config.optimizer_builder)(&vs)?;
            let snapshot = state.optimizer.map_tensors(device, |key, _| state.tensors.get(key))?;
            unsafe { self.config.opt_config.restore_snapshot(&snapshot, opt.as_raw_c_optimizer())?; }
            self.opt_state = OptSnapshot::Opt { opt, vs, module };
            self.transaction_failed = false;
        }
        Ok(())
    }

    /// A failed batch invalidates this epoch. Recovery restores a complete CPU
    /// checkpoint; no model or optimizer storage is duplicated on the hot path.
    pub(crate) fn training_transaction<T>(&mut self, run: impl FnOnce(&mut Self) -> Result<T, ThoError>) -> Result<T, ThoError> {
        self.ensure_training_usable()?;
        // The execution stream is set either in `impl dyn CudaActorHandle::run_stage/consume_stage` in Local Mode or in thoosa_common::ipc::wraps::run_worker_stage(_with_state) in IPC mode.
        // this does not fit the usual approach of setting/synching stream outside SyncStages
        let stream = match Device::from(self.opt_state.device()) {
            Device::Cuda(index) => Some(ThoCudaStream::current(index as i32)?),//steam from StreamGuard returned from thoosa_common::copy_ctx::HandleCtx::sync_ctx
            _ => None,
        };
        self.training_transaction_with(run, |module| {
            module.opt().step();
            Ok(())
        }, || stream.as_ref().map_or(Ok(()), |stream| complete_cuda_stream(stream.synchronize(), || stream.synchronize()).result))
    }

    // Keep phase ordering in one place; injected operations also allow failures at
    // the update/completion boundary to be tested without inducing CUDA corruption.
    fn training_transaction_with<T>(
        &mut self,
        run: impl FnOnce(&mut Self) -> Result<T, ThoError>,
        commit: impl FnOnce(&mut Self) -> Result<(), ThoError>,
        mut synchronize: impl FnMut() -> Result<(), ThoError>,
    ) -> Result<T, ThoError> {
        self.ensure_training_usable()?;
        self.ipc_export = None;
        let accumulated = finish_cuda_run(panic_to_err!({
            self.opt().zero_grad();
            run(self)
        }), synchronize());
        let output = match accumulated {
            Ok(output) => output,
            Err(error) => {
                self.transaction_failed = true;
                return Err(Self::failed_epoch(error));
            }
        };
        // No rollback snapshot of parameters or momentum is needed on the hot path.
        // From this point an error (including completion/gradient cleanup) is terminal.
        let committed = finish_cuda_run(panic_to_err!({
            commit(self)?;
            self.opt().zero_grad();
            Ok::<_, ThoError>(())
        }), synchronize());
        if let Err(cause) = committed {
            self.transaction_failed = true;
            self.ipc_export = None;
            if cause.needs_cuda_recovery() || cause.is_cuda_recovery_failed() { return Err(cause); }
            return Err(Self::failed_epoch(cause));
        }
        Ok(output)
    }

    pub(crate) fn ensure_training_usable(&self) -> Result<(), ThoError> {
        if self.transaction_failed { return Err(ThoError::new("training session invalidated; restore an epoch checkpoint before reuse")); }
        Ok(())
    }

    fn failed_epoch(cause: ThoError) -> ThoError {
        if cause.needs_cuda_recovery() || cause.is_cuda_recovery_failed() { return cause; }
        // Preserve allocation failures exposed as panics by tch without allowing
        // the generic executor to replay a batch against mutated training state.
        let cause = if matches!(cause.what(), ThoError::Anyhow(_)) && cause.is_retryable_oom() {
            ThoError::from(anyhow::Error::new(cause).context(tho_error::cuda_error(thoosa_shared::CudaAction::HandleOOM, "training allocation failed".into())))

        } else {
            cause
        };
        ThoError::TrainingSessionFailed(Arc::new(thoosa_shared::TrainingSessionFailure { original: cause, recovery: None }))
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.opt().set_lr(lr);
    }
}

// The stage mutex owns training state; Tensor and Optimizer are Send, but not Sync.
#[async_trait]
impl <DI, DO> MoveToDevice for ThoTrainModule<DI, DO>
where
    DI: Dims, DO: Dims
{

    async fn move_to_device(&mut self, tho_dst_device: usize) -> Result<(), ThoError>
    where
        Self:Sized
    {
        if matches!(self.opt_state, OptSnapshot::Offloaded(_)) {
            return Err(ThoError::new("training state is offloaded or incompletely restored"));
        }
        // An export snapshot must never survive another training step or move.
        self.ipc_export = None;
        debug!("ThoTrainModule::move_to_device: tho_dst_device={tho_dst_device} opt_state.device={:?}", self.opt_state.device());
        if tho_dst_device == self.opt_state.device().tho_device_idx() {
            return Ok(())
        }

        if tho_dst_device == 0 {
            /*
             * Finish reading the non-Sync Optimizer before awaiting tensor
             * copies so the async future never carries an Optimizer reference.
             */
            let (cuda_snapshot, vs) = match &self.opt_state {
                OptSnapshot::Opt { opt, vs, module: _module } => {
                    (unsafe { self.config.opt_config.snapshot(opt.as_raw_c_optimizer(), ThoDevice::try_from(vs.device())?.tho_device_idx())? }, vs)
                }
                _ => {
                    // var store should be already on cpu
                    return Ok(())
                }
            };

            self.opt_state = OptSnapshot::Snapshot {
                cpu_snap:clone_owned_optimizer_snapshot_to(cuda_snapshot, ThoDevice::from(0usize)).await?,
                vs:clone_var_store_to(&vs, tho_dst_device.into()).await?,
            };

            Ok(())
        } else {
            let (cpu_snap, cpu_var_store) = match &self.opt_state {
                /*
                 * Detach owned tensor handles while this is still synchronous;
                 * only those Send handles are carried into the async copies.
                 */
                OptSnapshot::Snapshot { cpu_snap, vs } => (detach_optimizer_snapshot(cpu_snap), vs),
                _ => {
                    // var store should be already on device
                    return Ok(())
                }
            };

            let (module, mut cuda_vs ) = Self::init_opt(&self.config.net, tho_dst_device.into())?;


            let cuda_snapshot = clone_owned_optimizer_snapshot_to(cpu_snap, tho_dst_device.into()).await?;
            let restored_vs = clone_var_store_to(&cpu_var_store, tho_dst_device.into()).await?;
            replace_data(&mut cuda_vs, restored_vs)?;

            /*
             * cuda_vs is the registered destination whose tensors received the
             * restored data, so both the optimizer and returned state must use it.
             */
            let opt = (self.config.optimizer_builder)(&cuda_vs)?;
            unsafe {self.config.opt_config.restore_snapshot(&cuda_snapshot, opt.as_raw_c_optimizer())}?;

            self.opt_state = OptSnapshot::Opt{opt, vs: cuda_vs, module };

            Ok(())
        }
    }

}

struct VarSpec {
    name: String,
    tensor: Tensor,

    // None means present only in named_variables.
    // Some((order, group)) means also present in trainable_variables.
    trainable: Option<(usize, usize)>,

    requires_grad: bool,
}

fn add_named_tensor(
    vs: &VarStore,
    full_name: &str,
    tensor: Tensor,
    trainable: bool,
    group: usize,
) -> Tensor {
    let (parent, leaf) = full_name
        .rsplit_once('.')
        .map_or(("", full_name), |(parent, leaf)| (parent, leaf));

    let mut path = vs.root().set_group(group);

    for component in parent.split('.').filter(|part| !part.is_empty()) {
        path = path.sub(component);
    }

    path.add(leaf, tensor, trainable)
}

pub async fn clone_adam_snapshot_to(
    src: &AdamOptimizerSnapshot,
    tho_device: ThoDevice,
) -> anyhow::Result<AdamOptimizerSnapshot> {
    /*
     * Own every tensor handle before the first await. Tensor is Send but not
     * Sync, so borrowing snapshot states across an await makes this future
     * non-Send even though the copy actor only needs owned tensor handles.
     */
    clone_owned_adam_snapshot_to(detach_adam_snapshot(src), tho_device).await
}

pub async fn clone_adam_w_snapshot_to(
    src: &AdamWOptimizerSnapshot,
    tho_device: ThoDevice,
) -> anyhow::Result<AdamWOptimizerSnapshot> {
    // AdamW shares the tensor-state representation but keeps a named copy entry point for optimizer-specific routing.
    clone_owned_adam_w_snapshot_to(detach_adam_w_snapshot(src), tho_device).await
}

pub async fn clone_sgd_snapshot_to(
    src: &SgdOptimizerSnapshot,
    tho_device: ThoDevice,
) -> anyhow::Result<SgdOptimizerSnapshot> {
    // Detaching before the await keeps the non-Sync borrowed snapshot out of the returned future.
    clone_owned_sgd_snapshot_to(detach_sgd_snapshot(src), tho_device).await
}

fn detach_adam_snapshot(src: &AdamOptimizerSnapshot) -> AdamOptimizerSnapshot {
    detach_adam_like_snapshot(src)
}

fn detach_adam_w_snapshot(src: &AdamWOptimizerSnapshot) -> AdamWOptimizerSnapshot {
    detach_adam_like_snapshot(src)
}

fn detach_sgd_snapshot(src: &SgdOptimizerSnapshot) -> SgdOptimizerSnapshot {
    SgdOptimizerSnapshot {
        defaults: src.defaults,
        groups: src.groups.iter().map(|src_group| {
            SgdGroupSnapshot {
                options: src_group.options,
                states: src_group.states.iter().map(|src_state| {
                    src_state.as_ref().map(|src_state| SgdParamStateSnapshot {
                        momentum_buffer: src_state.momentum_buffer.detach(),
                    })
                }).collect(),
            }
        }).collect(),
        tho_device_idx: src.tho_device_idx,
    }
}

fn detach_adam_like_snapshot(src: &AdamOptimizerSnapshot) -> AdamOptimizerSnapshot {
    AdamOptimizerSnapshot {
        defaults: src.defaults,
        groups: src.groups.iter().map(|src_group| {
            AdamGroupSnapshot {
                options: src_group.options,
                states: src_group.states.iter().map(|src_state| {
                    src_state.as_ref().map(|src_state| {
                        AdamParamStateSnapshot {
                            step: src_state.step,
                            exp_avg: src_state.exp_avg.detach(),
                            exp_avg_sq: src_state.exp_avg_sq.detach(),
                            max_exp_avg_sq: src_state.max_exp_avg_sq.as_ref().map(Tensor::detach),
                        }
                    })
                }).collect(),
            }
        }).collect(),
        tho_device_idx: src.tho_device_idx,
    }
}

async fn clone_owned_adam_snapshot_to(
    src: AdamOptimizerSnapshot,
    tho_device: ThoDevice,
) -> anyhow::Result<AdamOptimizerSnapshot> {
    clone_owned_adam_like_snapshot_to(src, tho_device, "Adam").await
}

async fn clone_owned_adam_w_snapshot_to(
    src: AdamWOptimizerSnapshot,
    tho_device: ThoDevice,
) -> anyhow::Result<AdamWOptimizerSnapshot> {
    clone_owned_adam_like_snapshot_to(src, tho_device, "AdamW").await
}

async fn clone_owned_adam_like_snapshot_to(src: AdamOptimizerSnapshot, tho_device: ThoDevice, optimizer_name: &str) -> anyhow::Result<AdamOptimizerSnapshot> {
    debug!("clone_owned_adam_like_snapshot_to: optimizer_name={optimizer_name} src={src:?} tho_device={tho_device:?}");
    let actor_device_idx = optimizer_copy_actor_device_idx(src.tho_device_idx, tho_device);

    let mut dst_groups = vec![];
    for src_group in src.groups {
        let AdamGroupSnapshot { options, states } = src_group;
        let mut dst_states = vec![];
        for state in states {
            match state {
                None => {
                    dst_states.push(None)
                }
                Some(src_state) => {
                    let AdamParamStateSnapshot { step, exp_avg, exp_avg_sq, max_exp_avg_sq } = src_state;
                    dst_states.push(Some(AdamParamStateSnapshot{
                        step,
                        exp_avg: GlobalCtx::global().cuda_actor(actor_device_idx).copy(exp_avg.into(), tho_device.into()).await?.into(),
                        exp_avg_sq: GlobalCtx::global().cuda_actor(actor_device_idx).copy(exp_avg_sq.into(), tho_device.into()).await?.into(),
                        max_exp_avg_sq: match max_exp_avg_sq {
                            None => None,
                            Some(v) => {
                                Some(GlobalCtx::global().cuda_actor(actor_device_idx).copy(v.into(), tho_device.into()).await?.into())
                            }
                        },
                    }))
                }
            }
        }

        dst_groups.push(AdamGroupSnapshot{
            options,
            states: dst_states,
        })
    }

    Ok(AdamOptimizerSnapshot{
        defaults: src.defaults,
        groups: dst_groups,
        tho_device_idx: tho_device.tho_device_idx(),
    })
}

async fn clone_owned_sgd_snapshot_to(src: SgdOptimizerSnapshot, tho_device: ThoDevice) -> anyhow::Result<SgdOptimizerSnapshot> {
    debug!("clone_owned_sgd_snapshot_to: src={src:?} tho_device={tho_device:?}");
    let actor_device_idx = optimizer_copy_actor_device_idx(src.tho_device_idx, tho_device);

    let mut dst_groups = vec![];
    for src_group in src.groups {
        let SgdGroupSnapshot { options, states } = src_group;
        let mut dst_states = vec![];
        for state in states {
            dst_states.push(match state {
                None => None,
                Some(src_state) => Some(SgdParamStateSnapshot {
                    momentum_buffer: GlobalCtx::global().cuda_actor(actor_device_idx).copy(src_state.momentum_buffer.into(), tho_device.into()).await?.into(),
                }),
            });
        }
        dst_groups.push(SgdGroupSnapshot { options, states: dst_states });
    }

    Ok(SgdOptimizerSnapshot {
        defaults: src.defaults,
        groups: dst_groups,
        tho_device_idx: tho_device.tho_device_idx(),
    })
}

fn optimizer_copy_actor_device_idx(src_tho_device_idx: usize, tho_device: ThoDevice) -> usize {
    // A CUDA actor is required even when the source is CPU, so CPU-to-GPU copies select the destination actor.
    if tho_device.is_cuda() { tho_device.tho_device_idx() } else { src_tho_device_idx }
}

fn detach_optimizer_snapshot(src: &OptimizerSnapshot) -> OptimizerSnapshot {
    match src {
        OptimizerSnapshot::Adam(snapshot) => OptimizerSnapshot::Adam(detach_adam_snapshot(snapshot)),
        OptimizerSnapshot::AdamW(snapshot) => OptimizerSnapshot::AdamW(detach_adam_w_snapshot(snapshot)),
        OptimizerSnapshot::Sgd(snapshot) => OptimizerSnapshot::Sgd(detach_sgd_snapshot(snapshot)),
    }
}

async fn clone_owned_optimizer_snapshot_to(src: OptimizerSnapshot, tho_device: ThoDevice) -> anyhow::Result<OptimizerSnapshot> {
    Ok(match src {
        OptimizerSnapshot::Adam(snapshot) => OptimizerSnapshot::Adam(clone_owned_adam_snapshot_to(snapshot, tho_device).await?),
        OptimizerSnapshot::AdamW(snapshot) => OptimizerSnapshot::AdamW(clone_owned_adam_w_snapshot_to(snapshot, tho_device).await?),
        OptimizerSnapshot::Sgd(snapshot) => OptimizerSnapshot::Sgd(clone_owned_sgd_snapshot_to(snapshot, tho_device).await?),
    })
}


pub async fn clone_var_store_to(
    src: &VarStore,
    tho_device: ThoDevice,
) -> anyhow::Result<VarStore> {

    let actor_device_idx = if tho_device.is_cuda() {
        assert_eq!(src.device(), Cpu);
        tho_device.tho_device_idx()
    } else  {
        assert_ne!(src.device(), Cpu);
        ThoDevice::try_from(src.device())?.tho_device_idx()
    };

    /*
     * Snapshot all Rust-side VarStore metadata before the first await.
     *
     * The returned specs own shallow tensor handles, so the spawned stages do
     * not borrow either the source VarStore or its mutex guard.
     */
    let (mut specs, src_kind) = {
        let variables = src.variables_.lock().unwrap();

        let specs = variables
            .named_variables
            .iter()
            .map(|(name, tensor)| {
                assert!(!tensor.grad().defined(), "can't copy store with defined grad");

                assert_ne!(tensor.device(), *tho_device, "can't copy variable {name:?}: source and destination devices are both {:?}", tho_device);

                let trainable = variables
                    .trainable_variables
                    .iter()
                    .enumerate()
                    .find_map(|(order, var)| {
                        var.tensor
                            .is_set_to(tensor)
                            .then_some((order, var.group))
                    });

                VarSpec {
                    name: name.clone(),
                    tensor: tensor.shallow_clone(),
                    trainable,
                    requires_grad: tensor.requires_grad(),
                }
            })
            .collect::<Vec<_>>();

        (specs, src.kind())
    };

    /*
     * Path::add appends trainable variables. Sort before copying so the
     * eventual registration order matches the source VarStore.
     */
    specs.sort_by_key(|spec| {
        spec.trainable
            .map_or((1, usize::MAX), |(order, _group)| (0, order))
    });

    let spec_count = specs.len();
    let mut copies = JoinSet::new();

    for (index, spec) in specs.into_iter().enumerate() {
        copies.spawn(async move {
            let detached = spec.tensor.detach();

            /*
             * The same-device case was already rejected before spawning.
             */
            debug_assert_ne!(detached.device(), *tho_device);

            let copied = GlobalCtx::global().cuda_actor(actor_device_idx).copy(detached.into(), tho_device.into()).await?;

            debug_assert_eq!(<ThoDevice as Into<ThoDevice>>::into(copied.tho_device()), tho_device);

            Ok::<_, ThoError>((index, spec, copied))
        });
    }

    /*
     * JoinSet returns results in completion order, not insertion order.
     */
    let mut copied_specs = Vec::with_capacity(spec_count);

    while let Some(joined) = copies.join_next().await {
        // Keep task failure separate from a CUDA failure returned by the copy itself.
        let copied = joined.context("join VarStore tensor copy task")??;

        copied_specs.push(copied);
    }

    copied_specs.sort_by_key(|(index, _, _)| *index);

    /*
     * Registration remains sequential because VarStore's trainable variable
     * ordering must be deterministic.
     */
    let mut dst = VarStore::new(tho_device.into());
    dst.set_kind(src_kind);

    for (_index, spec, copied) in copied_specs {
        let (trainable, group) = match spec.trainable {
            Some((_order, group)) => (true, group),
            None => (false, 0),
        };

        let dst_tensor = add_named_tensor(
            &dst,
            &spec.name,
            copied.into(),
            trainable,
            group,
        );

        /*
         * Path::add(..., true) enables requires_grad. Restore the actual
         * source state, including the state of a frozen VarStore.
         */
        let _ = dst_tensor.set_requires_grad(spec.requires_grad);

        assert!(
            !dst_tensor.grad().defined(),
            "copied variable {:?} unexpectedly has a defined grad",
            spec.name,
        );
    }

    Ok(dst)
}


pub fn replace_data(dst: &mut  VarStore, src: VarStore) -> anyhow::Result<()> {
    let mut dst_variables = dst.variables_.lock().unwrap();
    let src_variables = src.variables_.lock().unwrap();

    for name in dst_variables.named_variables.keys() {
        if !src_variables.named_variables.contains_key(name) {
            return Err(anyhow::anyhow!("no {name} in src"));
        }
    }

    for (name, dst_var) in dst_variables.named_variables.iter_mut() {
        let src_var = &src_variables.named_variables[name];

        if dst_var.device() != src_var.device() {
            return Err(anyhow::anyhow!("dst device {:?} != {:?} src device", dst_var.device(), src_var.device()));
        }

        if dst_var.kind() != src_var.kind() {
            return Err(anyhow::anyhow!("dst device {:?} != {:?} src device", dst_var.kind(),  src_var.kind()));
        }

        if dst_var.size() != src_var.size() {
            return Err(anyhow::anyhow!("dst size {:?} != {:?} src device", dst_var.size(),  src_var.size()));
        }

        if dst_var.grad().defined() {

            return Err(anyhow::anyhow!("destination parameter {name} has a defined gradient"));
        }

        no_grad(|| dst_var.f_set_data(src_var))?;
    }

    Ok(())
}

// Optimizer comparison is test scaffolding, not part of the runtime API.
#[cfg(test)]
async fn do_test_reset_optimizer(opt_config: ThoOptimizers, optimizer_name: &'static str) -> anyhow::Result<()>{
    // Every supported optimizer uses the same deterministic trajectory so device restoration is compared under identical pressure.
    const IMAGE_DIM: i64 = 128;
    const TEACHER_HIDDEN: i64 = 64;
    const HIDDEN_NODES: i64 = 128;
    const LABELS: i64 = 10;

    const TRAIN_SAMPLES: i64 = 2_048;
    const TEST_SAMPLES: i64 = 512;
    const BATCH_SIZE: i64 = 64;

    // Only ten epochs after resetting the optimizer.
    const EPOCHS: i64 = 60;
    const RESET_EPOCH: i64 = 50;
    const LEARNING_RATE: f64 = 1e-3;

    pub fn test_net() -> anyhow::Result<Node> {
        /*
         * Keep the same classifier topology that this optimizer-state test
         * used before ModConfig switched from Module to ThoModule.
         */
        SeqBuilder::new(IMAGE_DIM as u32, None)?
            .linear_s(
                HIDDEN_NODES as u32,
                "layer1",
            )?
            .func(|t|Ok(Tensor::f_relu(t)?))?
            .linear_s(
                LABELS as u32,
                "layer2",
            )?
            .end_seq()
    }

    fn make_split(
        samples: i64,
        teacher_w1: &Tensor,
        teacher_w2: &Tensor,
        feature_scales: &Tensor,
    ) -> (Tensor, Tensor) {
        // The latent features have a well-conditioned distribution.
        let latent = Tensor::randn(
            [samples, IMAGE_DIM],
            (Kind::Float, Device::Cuda(0)),
        );

        // The student sees an ill-conditioned version where feature scales
        // range approximately from exp(-4) to exp(4).
        let images = &latent * feature_scales;

        // Labels are generated by a fixed nonlinear teacher network, so there
        // is an actual relationship between images and labels.
        let labels = latent
            .matmul(teacher_w1)
            .relu()
            .matmul(teacher_w2)
            .argmax(-1, false);

        (images, labels)
    }

    async fn train_once(
        reset_optimizer: bool,
        opt_config: ThoOptimizers,
        train_images: &Tensor,
        train_labels: &Tensor,
        test_images: &Tensor,
        test_labels: &Tensor,
    ) -> anyhow::Result<Vec<f64>> {

        // Makes both runs start from identical model parameters.
        tch::manual_seed(1234);

        let mut losses = Vec::with_capacity(EPOCHS as usize);

        let mut tho_train_mod = ThoTrainModule::new_on_device(ModConfig::new(
            opt_config,
            /*
             * build_tch_module already returns the boxed ThoModule expected
             * by ModConfig, so wrapping it again would create Box<Box<_>>.
             */
            |vs| test_net()?.build_tch_module::<Batch<AnyDims>, Batch<AnyDims>>(&vs.root()),
            LEARNING_RATE,
        ), thoosa_common::tho_device::ThoDevice::from(1usize))?;

        for epoch in 1..=EPOCHS {

            if reset_optimizer && epoch == RESET_EPOCH {

                tho_train_mod.opt().zero_grad();

                 tho_train_mod.move_to_device(0).await?;
                // ///////////////
                tho_train_mod.move_to_device(1).await?;

            }

            // Module reconstruction consumes CUDA RNG, so reseed after the optional reset to preserve identical mini-batch ordering.
            tch::manual_seed(1234 + epoch);

            let permutation = Tensor::randperm(
                TRAIN_SAMPLES,
                (Kind::Int64, Device::Cuda(0)),
            );

            let mut start = 0;

            while start < TRAIN_SAMPLES {
                let len = BATCH_SIZE.min(TRAIN_SAMPLES - start);

                let indices = permutation.narrow(0, start, len);
                let batch_images = train_images.index_select(0, &indices);
                let batch_labels = train_labels.index_select(0, &indices);

                /*
                 * ThoModule receives typed ThoTensor inputs and returns typed
                 * logits; unwrap only to Tensor for the libtorch loss API.
                 */
                let batch_images: ThoTensor<Batch<AnyDims>, _> = batch_images.into();
                let logits = tho_train_mod.module().tho_forward(&batch_images, None, true)?;
                let loss = logits.as_ref().f_cross_entropy_loss::<&Tensor>(&batch_labels, None, tch::Reduction::Mean, -100, 0.0)?;

                tho_train_mod.opt().backward_step(&loss);

                start += len;
            }

            let train_images: ThoTensor<Batch<AnyDims>, _> = train_images.shallow_clone().into();
            let train_logits = tho_train_mod.module().tho_forward(&train_images, None, false)?;
            let train_loss = train_logits.as_ref().f_cross_entropy_loss::<&Tensor>(train_labels, None, tch::Reduction::Mean, -100, 0.0)?.double_value(&[]);

            let test_images: ThoTensor<Batch<AnyDims>, _> = test_images.shallow_clone().into();
            let test_logits = tho_train_mod.module().tho_forward(&test_images, None, false)?;
            let test_accuracy = test_logits.as_ref().f_argmax(-1, false)?.f_eq_tensor(test_labels)?.f_to_kind(Kind::Float)?.f_mean(Kind::Float)?.double_value(&[]);

            losses.push(train_loss);

            if matches!(epoch, 49 | 50 | 51 | 52 | 55 | 60) {
                tracing::debug!(
                    reset_optimizer,
                    epoch,
                    train_loss,
                    test_accuracy = 100.0 * test_accuracy,
                    "training progress",
                );
            }
        }

        Ok(losses)
    }

    tch::manual_seed(42);

    let teacher_w1 = Tensor::randn(
        [IMAGE_DIM, TEACHER_HIDDEN],
        (Kind::Float, Device::Cuda(0)),
    ) / (IMAGE_DIM as f64).sqrt();

    let teacher_w2 = Tensor::randn(
        [TEACHER_HIDDEN, LABELS],
        (Kind::Float, Device::Cuda(0)),
    ) / (TEACHER_HIDDEN as f64).sqrt();

    let feature_positions = Tensor::arange(
        IMAGE_DIM,
        (Kind::Float, Device::Cuda(0)),
    );

    let feature_scales = (
        feature_positions / ((IMAGE_DIM - 1) as f64) * 8.0 - 4.0
    )
        .exp();

    let (train_images, train_labels) = make_split(
        TRAIN_SAMPLES,
        &teacher_w1,
        &teacher_w2,
        &feature_scales,
    );

    let (test_images, test_labels) = make_split(
        TEST_SAMPLES,
        &teacher_w1,
        &teacher_w2,
        &feature_scales,
    );

    let preserved_losses = train_once(
        false,
        opt_config.clone(),
        &train_images,
        &train_labels,
        &test_images,
        &test_labels,
    ).await?;

    let reset_losses = train_once(
        true,
        opt_config,
        &train_images,
        &train_labels,
        &test_images,
        &test_labels,
    ).await?;

    for epoch in 49..=60 {
        let index = (epoch - 1) as usize;

        debug!("epoch={epoch} preserved={} reset={} difference={}", preserved_losses[index],reset_losses[index], reset_losses[index] - preserved_losses[index]);
    }

    // Compare the first epoch affected by the replacement.
    let first_affected_index = (RESET_EPOCH - 1) as usize;

    let immediate_difference = (
        reset_losses[first_affected_index]
            - preserved_losses[first_affected_index]
    ).abs();

    assert!(immediate_difference < 1e-3, "{optimizer_name} produced a diverging trajectory after reset: preserved={}, reset={}", preserved_losses[first_affected_index], reset_losses[first_affected_index]);

    Ok(())
}


#[cfg(test)]
mod tests {
    use crate::nets::tho_varstore::{clone_var_store_to, do_test_reset_optimizer};
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use log::debug;
    use tch::Device::Cpu;
    use tch::nn::{Init, VarStore};
    use tch::{Tensor, nn};

    fn create_variables(vs: &VarStore) {
        let p = vs.root().sub("my_module");

        let _weight = p.var(
            "weight",
            &[128, 64],
            Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        );

        let _t0 = p.add(
            "t_0",
            Tensor::zeros([10, 1000], (tch::Kind::Float, vs.device())),
            false,
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_clone_var_store_to() -> anyhow::Result<()> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let vs_cpu = VarStore::new(Cpu);
        create_variables(&vs_cpu);
        let cpu_vars = vs_cpu.variables();
        let cpu_t_0 = cpu_vars.get("my_module.t_0").expect("no cpu t_0");
        let cpu_weight = cpu_vars.get("my_module.weight").expect("no cpu weight");

        let mut bytes = Vec::new();
        vs_cpu.save_to_stream(&mut bytes)?;

        debug!("source variables={:?}",vs_cpu.variables().keys().collect::<Vec<_>>());


        let vs_cuda = clone_var_store_to(&vs_cpu, thoosa_common::tho_device::ThoDevice::from(1usize)).await?;

        let cuda_vars = vs_cuda.variables();
        for (name, tensor) in &cuda_vars {
            debug!("name={name}, device={:?}, shape={:?}, requires_grad={}",tensor.device(),tensor.size(),tensor.requires_grad(),);
        }

        let cpu_vars_1 = clone_var_store_to(&vs_cuda, thoosa_common::tho_device::ThoDevice::from(0usize)).await?;

        let cpu_vars_1 = cpu_vars_1.variables();
        assert!(cpu_t_0.allclose(cpu_vars_1.get("my_module.t_0").expect("no cpu t_0"), 0.01, 0.01, false));
        assert!(cpu_weight.allclose(cpu_vars_1.get("my_module.weight").expect("no cpu weight"), 0.01, 0.01, false));

        Ok(())
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_reset_adam_local() -> anyhow::Result<()> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let config = nn::Adam { beta1: 0.99, beta2: 0.9999, ..Default::default() }.amsgrad(true).into();
        do_test_reset_optimizer(config, "Adam").await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_reset_adam_w_local() -> anyhow::Result<()> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let config = nn::AdamW { beta1: 0.99, beta2: 0.9999, ..Default::default() }.amsgrad(true).into();
        do_test_reset_optimizer(config, "AdamW").await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_reset_sgd_local() -> anyhow::Result<()> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let config = nn::Sgd { momentum: 0.9, dampening: 0.0, wd: 0.01, nesterov: true }.into();
        do_test_reset_optimizer(config, "SGD").await
    }


}

#[cfg(test)]
#[path = "training_transaction_tests.rs"]
mod training_transaction_tests;
