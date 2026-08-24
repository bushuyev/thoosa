use crate::builder::{Node, SeqBuilder};
use crate::dims::{AnyDims, Batch, Dims};
use crate::my_tch::ThoModule;
use crate::runner::global_ctx::GlobalCtx;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::{ThoTensor, ThoTensorInner};
use async_trait::async_trait;
use log::debug;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;
use tch::Device::{Cpu, Cuda};
use tch::nn::{Module, Optimizer, OptimizerConfig, Path, VarStore};
use tch::{Device, Kind, Tensor, nn, no_grad};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{AsyncStage, RunCtx, Stage, SyncStage};
use thoosa_shared::traits::{Named, ThoSelector};
use thoosa_shared::{CopyToDevice, MoveToDevice, ThoError, panic_to_err};
use thoosa_shims::optimizers::{AdamGroupSnapshot, AdamOptimizerSnapshot, AdamParamStateSnapshot, AdamWOptimizerSnapshot, SgdGroupSnapshot, SgdOptimizerSnapshot, SgdParamStateSnapshot, adam_snapshot, adam_w_snapshot, restore_adam_snapshot, restore_adam_w_snapshot, restore_sgd_snapshot, sgd_snapshot};
use tokio::task::JoinSet;


pub struct ModConfig<DI, DO>
where
    DI: Dims, DO: Dims
{
    optimizer_builder: Box<dyn Fn(&VarStore) -> Result<Pin<Box<Optimizer>>, ThoError> + Send + Sync>,
    net: Box<dyn Fn(&VarStore) -> Result<Box<dyn ThoModule<Batch<DI>, Batch<DO>>>, ThoError> + Send + Sync>,
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
        NB: Fn(&VarStore) -> Result<Box<dyn ThoModule<Batch<DI>, Batch<DO>>>, ThoError> + Send + Sync + 'static,
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
    fn build(&self, vs:&VarStore, lr: f64) -> Result<Optimizer,ThoError>
    {
        Ok(match self {
            ThoOptimizers::Adam { config } => config.build(vs, lr)?,
            ThoOptimizers::AdamW { config } => config.build(vs, lr)?,
            ThoOptimizers::Sgd { config } => config.build(vs, lr)?,
        })
    }

    pub fn build_for(&self, vs: &VarStore, lr: f64) -> Result<Optimizer, ThoError> {
        // Keep legacy callers on the same optimizer construction path used by
        // stateful training modules.
        self.build(vs, lr)
    }

    unsafe fn snapshot(&self, optimizer: *const torch_sys::C_optimizer, tho_device_idx: usize) -> Result<OptimizerSnapshot, ThoError> {
        // The configuration is the authoritative optimizer type because tch erases it behind Optimizer.
        Ok(match self {
            ThoOptimizers::Adam { .. } => OptimizerSnapshot::Adam(unsafe { adam_snapshot(optimizer, tho_device_idx)? }),
            ThoOptimizers::AdamW { .. } => OptimizerSnapshot::AdamW(unsafe { adam_w_snapshot(optimizer, tho_device_idx)? }),
            ThoOptimizers::Sgd { .. } => OptimizerSnapshot::Sgd(unsafe { sgd_snapshot(optimizer, tho_device_idx)? }),
        })
    }

    unsafe fn restore_snapshot(&self, snapshot: &OptimizerSnapshot, optimizer: *mut torch_sys::C_optimizer) -> Result<(), ThoError> {
        match (self, snapshot) {
            (ThoOptimizers::Adam { .. }, OptimizerSnapshot::Adam(snapshot)) => unsafe { restore_adam_snapshot(snapshot, optimizer) },
            (ThoOptimizers::AdamW { .. }, OptimizerSnapshot::AdamW(snapshot)) => unsafe { restore_adam_w_snapshot(snapshot, optimizer) },
            (ThoOptimizers::Sgd { .. }, OptimizerSnapshot::Sgd(snapshot)) => unsafe { restore_sgd_snapshot(snapshot, optimizer) },
            (config, snapshot) => Err(ThoError::new(format!("optimizer configuration {config:?} does not match snapshot {snapshot:?}"))),
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
    opt_state:OptSnapshot<DI, DO>
}

#[derive(Debug)]
enum OptSnapshot<DI, DO>
where
    DI: Dims, DO: Dims
{
    Opt{opt:Pin<Box<Optimizer>>, vs:VarStore, module:Box<dyn ThoModule<Batch<DI>, Batch<DO>>>},
    Snapshot{ cpu_snap: OptimizerSnapshot, vs:VarStore}
}

impl <DI, DO> OptSnapshot<DI, DO>
where
    DI: Dims, DO: Dims
{
    fn device(&self) -> ThoDevice {
        match self {
            OptSnapshot::Opt { vs,.. }| OptSnapshot::Snapshot { vs,.. } => {
                vs.device().into()
            }
        }
    }
}

#[derive(Debug)]
enum OptimizerSnapshot {
    Adam(AdamOptimizerSnapshot),
    AdamW(AdamWOptimizerSnapshot),
    Sgd(SgdOptimizerSnapshot),
}

impl <DI, DO> ThoTrainModule<DI, DO>
where
    DI: Dims, DO: Dims
{
    pub fn new_on_device(config: ModConfig<DI, DO>, device: ThoDevice) -> Result<Self, ThoError> {
        let (module, var_store) = Self::init_opt(&config.net, device)?;
        let opt = (config.optimizer_builder)(&var_store)?;
        Ok(Self {
            config,
            opt_state: OptSnapshot::Opt { opt, vs: var_store, module },
        })
    }

    fn init_opt(net: &dyn Fn(&VarStore) -> Result<Box<dyn ThoModule<Batch<DI>, Batch<DO>>>, ThoError>, device: ThoDevice) -> Result<(Box<dyn ThoModule<Batch<DI>, Batch<DO>>>, VarStore), ThoError> {
        let var_store = nn::VarStore::new(device.into());
        let module = net(&var_store)?;
        Ok((module, var_store))
    }

    pub fn opt(&mut self) ->&mut Optimizer {
        match &mut self.opt_state {
            OptSnapshot::Opt { opt, .. } => {
                opt
            }
            OptSnapshot::Snapshot { .. } => {
                panic!("wrong state")
            }
        }
    }

    pub fn module(&mut self) ->&mut Box<dyn ThoModule<Batch<DI>, Batch<DO>>> {
        match &mut self.opt_state {
            OptSnapshot::Opt { module, .. } => {
                module
            }
            OptSnapshot::Snapshot { .. } => {
                panic!("wrong state")
            }
        }
    }

    pub fn into_var_store(self) -> Result<VarStore, ThoError> {
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
        }
    }

    pub fn save_var_store_to_stream(&self, output: &mut dyn std::io::Write) -> Result<(), ThoError> {
        match &self.opt_state {
            OptSnapshot::Opt { vs, .. } | OptSnapshot::Snapshot { vs, .. } => vs.save_to_stream(output).map_err(ThoError::from),
        }
    }

    pub fn save_var_store<T>(&self, path:T) -> Result<(), ThoError>
    where
        T:AsRef<std::path::Path> + Debug
    {
        debug!("ThoTrainModule::save_var_store: path={path:?} self.opt_state={:?}", self.opt_state);

        match &self.opt_state {
            OptSnapshot::Opt { vs, .. } | OptSnapshot::Snapshot { vs, .. } => {
                panic_to_err!(vs.save(&path))
            },
        }

    }

    pub fn set_lr(&mut self, lr: f64) {
        self.opt().set_lr(lr);
    }
}

unsafe impl <DI, DO> Send for OptSnapshot<DI, DO>
where
DI: Dims, DO: Dims
{}

unsafe impl <DI, DO> Send for ThoTrainModule<DI, DO> where
    DI: Dims, DO: Dims
{}
unsafe impl <DI, DO> Sync for ThoTrainModule<DI, DO>
where
    DI: Dims, DO: Dims
{}

#[async_trait]
impl <DI, DO> MoveToDevice for ThoTrainModule<DI, DO>
where
    DI: Dims, DO: Dims
{

    async fn move_to_device(&mut self, tho_dst_device: usize) -> Result<(), ThoError>
    where
        Self:Sized
    {
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
                OptSnapshot::Opt { opt, vs, module } => {
                    (unsafe { self.config.opt_config.snapshot(opt.as_raw_c_optimizer(), ThoDevice::from(vs.device()).tho_device_idx())? }, vs)
                }
                _ => {
                    // var store should be already on cpu
                    return Ok(())
                }
            };

            self.opt_state = OptSnapshot::Snapshot {
                cpu_snap:clone_owned_optimizer_snapshot_to(cuda_snapshot, Cpu.into()).await?,
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
) -> Result<AdamOptimizerSnapshot, ThoError> {
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
) -> Result<AdamWOptimizerSnapshot, ThoError> {
    // AdamW shares the tensor-state representation but keeps a named copy entry point for optimizer-specific routing.
    clone_owned_adam_w_snapshot_to(detach_adam_w_snapshot(src), tho_device).await
}

pub async fn clone_sgd_snapshot_to(
    src: &SgdOptimizerSnapshot,
    tho_device: ThoDevice,
) -> Result<SgdOptimizerSnapshot, ThoError> {
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
) -> Result<AdamOptimizerSnapshot, ThoError> {
    clone_owned_adam_like_snapshot_to(src, tho_device, "Adam").await
}

async fn clone_owned_adam_w_snapshot_to(
    src: AdamWOptimizerSnapshot,
    tho_device: ThoDevice,
) -> Result<AdamWOptimizerSnapshot, ThoError> {
    clone_owned_adam_like_snapshot_to(src, tho_device, "AdamW").await
}

async fn clone_owned_adam_like_snapshot_to(src: AdamOptimizerSnapshot, tho_device: ThoDevice, optimizer_name: &str) -> Result<AdamOptimizerSnapshot, ThoError> {
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

async fn clone_owned_sgd_snapshot_to(src: SgdOptimizerSnapshot, tho_device: ThoDevice) -> Result<SgdOptimizerSnapshot, ThoError> {
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

async fn clone_owned_optimizer_snapshot_to(src: OptimizerSnapshot, tho_device: ThoDevice) -> Result<OptimizerSnapshot, ThoError> {
    Ok(match src {
        OptimizerSnapshot::Adam(snapshot) => OptimizerSnapshot::Adam(clone_owned_adam_snapshot_to(snapshot, tho_device).await?),
        OptimizerSnapshot::AdamW(snapshot) => OptimizerSnapshot::AdamW(clone_owned_adam_w_snapshot_to(snapshot, tho_device).await?),
        OptimizerSnapshot::Sgd(snapshot) => OptimizerSnapshot::Sgd(clone_owned_sgd_snapshot_to(snapshot, tho_device).await?),
    })
}


pub async fn clone_var_store_to(
    src: &VarStore,
    tho_device: ThoDevice,
) -> Result<VarStore, ThoError> {

    let actor_device_idx = if tho_device.is_cuda() {
        assert_eq!(src.device(), Cpu);
        tho_device.tho_device_idx()
    } else  {
        assert_ne!(src.device(), Cpu);
        Into::<ThoDevice>::into(src.device()).tho_device_idx()
    };

    /*
     * Snapshot all Rust-side VarStore metadata before the first await.
     *
     * The returned specs own shallow tensor handles, so the spawned tasks do
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
        let copied = joined
            .map_err(|error| {
                tch::TchError::Torch(format!(
                    "VarStore tensor copy task failed: {error}"
                ))
            })??;

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


pub fn replace_data(dst: &mut  VarStore, src: VarStore) -> Result<(), ThoError> {
    let mut dst_variables = dst.variables_.lock().unwrap();
    let src_variables = src.variables_.lock().unwrap();

    for name in dst_variables.named_variables.keys() {
        if !src_variables.named_variables.contains_key(name) {
            return Err(ThoError::new(format!("no {name} in src")));
        }
    }

    for (name, dst_var) in dst_variables.named_variables.iter_mut() {
        let src_var = &src_variables.named_variables[name];

        if dst_var.device() != src_var.device() {
            return Err(ThoError::new(format!("dst device {:?} != {:?} src device", dst_var.device(), src_var.device())));
        }

        if dst_var.kind() != src_var.kind() {
            return Err(ThoError::new(format!("dst device {:?} != {:?} src device", dst_var.kind(),  src_var.kind())));
        }

        if dst_var.size() != src_var.size() {
            return Err(ThoError::new(format!("dst size {:?} != {:?} src device", dst_var.size(),  src_var.size())));
        }

        if dst_var.grad().defined() {

            return Err(ThoError::new(format!("destination parameter {name} has a defined gradient")));
        }

        no_grad(|| dst_var.f_set_data(src_var))?;
    }

    Ok(())
}

async fn do_test_reset_optimizer(opt_config: ThoOptimizers, optimizer_name: &'static str) -> Result<(), ThoError>{
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

    pub fn test_net() -> Result<Node, ThoError> {
        /*
         * Keep the same classifier topology that this optimizer-state test
         * used before ModConfig switched from Module to ThoModule.
         */
        SeqBuilder::new(IMAGE_DIM as u32, None)?
            .linear_s(
                HIDDEN_NODES as u32,
                "layer1",
            )?
            .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?
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
    ) -> Result<Vec<f64>, ThoError> {

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
        ), Cuda(0).into())?;

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

            let train_images: ThoTensor<Batch<AnyDims>, _> = train_images.into();
            let train_logits = tho_train_mod.module().tho_forward(&train_images, None, false)?;
            let train_loss = train_logits.as_ref().f_cross_entropy_loss::<&Tensor>(train_labels, None, tch::Reduction::Mean, -100, 0.0)?.double_value(&[]);

            let test_images: ThoTensor<Batch<AnyDims>, _> = test_images.into();
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
    use crate::builder::{Node, SeqBuilder};
    use crate::dims::{AnyDims, Batch};
    use crate::ipc::cuda_wokrer::CudaWorker;
    use crate::ipc::json_ser::SyncStageIpcWrapper;
    use crate::ipc::tests::run_ipc_test;
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::tho_device::ThoDevice;
    use crate::tho_tensor::ThoTensor;
    use crate::tho_varstore::{clone_adam_snapshot_to, clone_var_store_to, do_test_reset_optimizer, replace_data, ModConfig, OptSnapshot, ThoTrainModule};
    use log::debug;
    use std::collections::HashMap;
    use std::io::Cursor;
    use std::sync::{Arc, Mutex};
    use tch::Device::{Cpu, Cuda};
    use tch::kind::FLOAT_CPU;
    use tch::nn::{Init, Module, OptimizerConfig, VarStore, Variables};
    use tch::{Device, Kind, Tensor, nn};
    use thoosa_shared::{CopyToDevice, ThoError};
    use thoosa_shims::optimizers::*;

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
    async fn test_clone_var_store_to() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let vs_cpu = VarStore::new(Cpu);
        create_variables(&vs_cpu);
        let cpu_vars = vs_cpu.variables();
        let cpu_t_0 = cpu_vars.get("my_module.t_0").expect("no cpu t_0");
        let cpu_weight = cpu_vars.get("my_module.weight").expect("no cpu weight");

        let mut bytes = Vec::new();
        vs_cpu.save_to_stream(&mut bytes)?;

        debug!("source variables={:?}",vs_cpu.variables().keys().collect::<Vec<_>>());


        let vs_cuda = clone_var_store_to(&vs_cpu, Cuda(0).into()).await?;

        let cuda_vars = vs_cuda.variables();
        for (name, tensor) in &cuda_vars {
            debug!("name={name}, device={:?}, shape={:?}, requires_grad={}",tensor.device(),tensor.size(),tensor.requires_grad(),);
        }

        let cpu_vars_1 = clone_var_store_to(&vs_cuda, Cpu.into()).await?;

        let cpu_vars_1 = cpu_vars_1.variables();
        assert!(cpu_t_0.allclose(cpu_vars_1.get("my_module.t_0").expect("no cpu t_0"), 0.01, 0.01, false));
        assert!(cpu_weight.allclose(cpu_vars_1.get("my_module.weight").expect("no cpu weight"), 0.01, 0.01, false));

        Ok(())
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_reset_adam_local() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let config = nn::Adam { beta1: 0.99, beta2: 0.9999, ..Default::default() }.amsgrad(true).into();
        do_test_reset_optimizer(config, "Adam").await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_reset_adam_w_local() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let config = nn::AdamW { beta1: 0.99, beta2: 0.9999, ..Default::default() }.amsgrad(true).into();
        do_test_reset_optimizer(config, "AdamW").await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_reset_sgd_local() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let config = nn::Sgd { momentum: 0.9, dampening: 0.0, wd: 0.01, nesterov: true }.into();
        do_test_reset_optimizer(config, "SGD").await
    }


}
