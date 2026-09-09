use std::ffi::{c_int, c_ulong};
use std::mem::MaybeUninit;
use std::ptr;
use log::debug;
use tch::Tensor;
use torch_sys::{C_optimizer, C_tensor};
use thoosa_shared::ThoError;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct C_AdamOptions {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub amsgrad: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct C_AdamGroup {
    pub options: C_AdamOptions,
    pub parameter_count: c_ulong,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct C_AdamParamState {
    pub step: i64,
    pub has_state: c_int,
    pub exp_avg: *mut C_tensor,
    pub exp_avg_sq: *mut C_tensor,
    pub max_exp_avg_sq: *mut C_tensor,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct C_SgdOptions {
    pub lr: f64,
    pub momentum: f64,
    pub dampening: f64,
    pub weight_decay: f64,
    pub nesterov: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct C_SgdGroup {
    pub options: C_SgdOptions,
    pub parameter_count: c_ulong,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct C_SgdParamState {
    pub has_state: c_int,
    pub momentum_buffer: *mut C_tensor,
}

unsafe extern "C" {
    pub fn tho_adam_group_count(
        opt: *const C_optimizer,
    ) -> c_ulong;

    pub fn tho_adam_get_defaults(
        opt: *const C_optimizer,
        out: *const C_AdamOptions,
    );

    pub fn tho_adam_get_group(
        opt: *const C_optimizer,
        group_index: c_ulong,
        out: *const C_AdamGroup,
    );

    pub fn tho_adam_get_param_state(
        opt: *const C_optimizer,
        group_index: c_ulong,
        parameter_index: c_ulong,
        out: *const C_AdamParamState,
    );

    pub fn tho_adam_set_defaults(
        opt: *const C_optimizer,
        options: *const C_AdamOptions,
    );

    pub fn tho_adam_set_group_options(
        opt: *const C_optimizer,
        group_index: c_ulong,
        options: *const C_AdamOptions,
    );

    pub fn tho_adam_set_param_state(
        opt: *const C_optimizer,
        group_index: c_ulong,
        parameter_index: c_ulong,
        has_state: c_int,
        step: i64,
        exp_avg: *mut C_tensor,
        exp_avg_sq: *mut C_tensor,
        max_exp_avg_sq: *mut C_tensor,
    );

    pub fn tho_adam_w_group_count(
        opt: *const C_optimizer,
    ) -> c_ulong;

    pub fn tho_adam_w_get_defaults(
        opt: *const C_optimizer,
        out: *const C_AdamOptions,
    );

    pub fn tho_adam_w_get_group(
        opt: *const C_optimizer,
        group_index: c_ulong,
        out: *const C_AdamGroup,
    );

    pub fn tho_adam_w_get_param_state(
        opt: *const C_optimizer,
        group_index: c_ulong,
        parameter_index: c_ulong,
        out: *const C_AdamParamState,
    );

    pub fn tho_adam_w_set_defaults(
        opt: *const C_optimizer,
        options: *const C_AdamOptions,
    );

    pub fn tho_adam_w_set_group_options(
        opt: *const C_optimizer,
        group_index: c_ulong,
        options: *const C_AdamOptions,
    );

    pub fn tho_adam_w_set_param_state(
        opt: *const C_optimizer,
        group_index: c_ulong,
        parameter_index: c_ulong,
        has_state: c_int,
        step: i64,
        exp_avg: *mut C_tensor,
        exp_avg_sq: *mut C_tensor,
        max_exp_avg_sq: *mut C_tensor,
    );

    pub fn tho_sgd_group_count(
        opt: *const C_optimizer,
    ) -> c_ulong;

    pub fn tho_sgd_get_defaults(
        opt: *const C_optimizer,
        out: *const C_SgdOptions,
    );

    pub fn tho_sgd_get_group(
        opt: *const C_optimizer,
        group_index: c_ulong,
        out: *const C_SgdGroup,
    );

    pub fn tho_sgd_get_param_state(
        opt: *const C_optimizer,
        group_index: c_ulong,
        parameter_index: c_ulong,
        out: *const C_SgdParamState,
    );

    pub fn tho_sgd_set_defaults(
        opt: *const C_optimizer,
        options: *const C_SgdOptions,
    );

    pub fn tho_sgd_set_group_options(
        opt: *const C_optimizer,
        group_index: c_ulong,
        options: *const C_SgdOptions,
    );

    pub fn tho_sgd_set_param_state(
        opt: *const C_optimizer,
        group_index: c_ulong,
        parameter_index: c_ulong,
        has_state: c_int,
        momentum_buffer: *mut C_tensor,
    );
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamOptionsSnapshot {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub amsgrad: bool,
}

#[derive(Debug)]
pub struct AdamParamStateSnapshot {
    pub step: i64,
    pub exp_avg: Tensor,
    pub exp_avg_sq: Tensor,
    pub max_exp_avg_sq: Option<Tensor>,
}




impl PartialEq for AdamParamStateSnapshot {
    fn eq(&self, other: &Self) -> bool {
        self.step == other.step
        && self.exp_avg.allclose(&other.exp_avg, 0.01, 0.01, true)
        && self.exp_avg_sq.allclose(&other.exp_avg_sq, 0.01, 0.01, true)
        && (
            self.max_exp_avg_sq.is_none() && other.max_exp_avg_sq.is_none()
            ||
            self.max_exp_avg_sq.as_ref().map(|v|other.max_exp_avg_sq.as_ref().map(|other|other.allclose(&v, 0.01, 0.01, true)).unwrap_or(false)).unwrap_or(false)
        )
    }
}

#[derive(Debug, PartialEq)]
pub struct AdamGroupSnapshot {
    pub options: AdamOptionsSnapshot,

    // Parameter-index order within this group.
    //
    // None means Adam has not initialized state for this parameter yet.
    pub states: Vec<Option<AdamParamStateSnapshot>>,
}

#[derive(Debug, PartialEq)]
pub struct AdamOptimizerSnapshot {
    pub defaults: AdamOptionsSnapshot,
    pub groups: Vec<AdamGroupSnapshot>,
    pub tho_device_idx:usize
}

// LibTorch's AdamW snapshot fields match Adam exactly, so share the owned Rust representation while keeping optimizer-specific entry points.
pub type AdamWOptionsSnapshot = AdamOptionsSnapshot;
pub type AdamWParamStateSnapshot = AdamParamStateSnapshot;
pub type AdamWGroupSnapshot = AdamGroupSnapshot;
pub type AdamWOptimizerSnapshot = AdamOptimizerSnapshot;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SgdOptionsSnapshot {
    pub lr: f64,
    pub momentum: f64,
    pub dampening: f64,
    pub weight_decay: f64,
    pub nesterov: bool,
}

#[derive(Debug)]
pub struct SgdParamStateSnapshot {
    pub momentum_buffer: Tensor,
}

impl PartialEq for SgdParamStateSnapshot {
    fn eq(&self, other: &Self) -> bool {
        self.momentum_buffer.allclose(&other.momentum_buffer, 0.01, 0.01, true)
    }
}

#[derive(Debug, PartialEq)]
pub struct SgdGroupSnapshot {
    pub options: SgdOptionsSnapshot,

    // None represents parameters for which SGD has not created a momentum buffer.
    pub states: Vec<Option<SgdParamStateSnapshot>>,
}

#[derive(Debug, PartialEq)]
pub struct SgdOptimizerSnapshot {
    pub defaults: SgdOptionsSnapshot,
    pub groups: Vec<SgdGroupSnapshot>,
    pub tho_device_idx: usize,
}

impl AdamOptionsSnapshot {
    fn from_ffi(value: C_AdamOptions) -> Self {
        Self {
            lr: value.lr,
            beta1: value.beta1,
            beta2: value.beta2,
            eps: value.eps,
            weight_decay: value.weight_decay,
            amsgrad: value.amsgrad != 0,
        }
    }

    fn to_ffi(self) -> C_AdamOptions {
        C_AdamOptions {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
            amsgrad: i32::from(self.amsgrad),
        }
    }
}

impl SgdOptionsSnapshot {
    fn from_ffi(value: C_SgdOptions) -> Self {
        Self {
            lr: value.lr,
            momentum: value.momentum,
            dampening: value.dampening,
            weight_decay: value.weight_decay,
            nesterov: value.nesterov != 0,
        }
    }

    fn to_ffi(self) -> C_SgdOptions {
        C_SgdOptions {
            lr: self.lr,
            momentum: self.momentum,
            dampening: self.dampening,
            weight_decay: self.weight_decay,
            nesterov: i32::from(self.nesterov),
        }
    }
}

type GroupCountFn = unsafe extern "C" fn(*const C_optimizer) -> c_ulong;
type GetDefaultsFn = unsafe extern "C" fn(*const C_optimizer, *const C_AdamOptions);
type GetGroupFn = unsafe extern "C" fn(*const C_optimizer, c_ulong, *const C_AdamGroup);
type GetParamStateFn = unsafe extern "C" fn(*const C_optimizer, c_ulong, c_ulong, *const C_AdamParamState);
type SetDefaultsFn = unsafe extern "C" fn(*const C_optimizer, *const C_AdamOptions);
type SetGroupOptionsFn = unsafe extern "C" fn(*const C_optimizer, c_ulong, *const C_AdamOptions);
type SetParamStateFn = unsafe extern "C" fn(*const C_optimizer, c_ulong, c_ulong, c_int, i64, *mut C_tensor, *mut C_tensor, *mut C_tensor);

#[derive(Clone, Copy)]
struct AdamShimFns {
    optimizer_name: &'static str,
    group_count: GroupCountFn,
    get_defaults: GetDefaultsFn,
    get_group: GetGroupFn,
    get_param_state: GetParamStateFn,
    set_defaults: SetDefaultsFn,
    set_group_options: SetGroupOptionsFn,
    set_param_state: SetParamStateFn,
}

const ADAM_SHIM_FNS: AdamShimFns = AdamShimFns {
    optimizer_name: "Adam",
    group_count: tho_adam_group_count,
    get_defaults: tho_adam_get_defaults,
    get_group: tho_adam_get_group,
    get_param_state: tho_adam_get_param_state,
    set_defaults: tho_adam_set_defaults,
    set_group_options: tho_adam_set_group_options,
    set_param_state: tho_adam_set_param_state,
};

const ADAM_W_SHIM_FNS: AdamShimFns = AdamShimFns {
    optimizer_name: "AdamW",
    group_count: tho_adam_w_group_count,
    get_defaults: tho_adam_w_get_defaults,
    get_group: tho_adam_w_get_group,
    get_param_state: tho_adam_w_get_param_state,
    set_defaults: tho_adam_w_set_defaults,
    set_group_options: tho_adam_w_set_group_options,
    set_param_state: tho_adam_w_set_param_state,
};

fn owned_tensor_from_ptr(
    ptr: *mut C_tensor,
    optimizer_name: &str,
    description: &str,
) -> Result<Tensor, ThoError> {
    if ptr.is_null() {
        return Err(ThoError::new(format!(
            "C++ {optimizer_name} shim returned a null {description}"
        )));
    }

    // The C++ shim returned `new torch::Tensor(...)`.
    // Tensor now owns that pointer and will eventually call at_free().
    Ok(unsafe { Tensor::from_ptr(ptr) })
}

pub unsafe fn adam_snapshot(c_optimizer: *const C_optimizer, tho_device_idx:usize) -> Result<AdamOptimizerSnapshot, ThoError> {
    // The concrete shim table preserves LibTorch runtime-type checks while sharing the identical Adam-family snapshot traversal.
    unsafe { adam_like_snapshot(c_optimizer, tho_device_idx, ADAM_SHIM_FNS) }
}

pub unsafe fn adam_w_snapshot(c_optimizer: *const C_optimizer, tho_device_idx:usize) -> Result<AdamWOptimizerSnapshot, ThoError> {
    unsafe { adam_like_snapshot(c_optimizer, tho_device_idx, ADAM_W_SHIM_FNS) }
}

unsafe fn adam_like_snapshot(c_optimizer: *const C_optimizer, tho_device_idx:usize, shims: AdamShimFns) -> Result<AdamOptimizerSnapshot, ThoError> {
    let group_count = unsafe { (shims.group_count)(c_optimizer) };

    let mut raw_defaults = MaybeUninit::<C_AdamOptions>::zeroed();


    unsafe { (shims.get_defaults)(c_optimizer, raw_defaults.as_mut_ptr()) };


    debug!("raw_defaults={raw_defaults:?}");

    let defaults = AdamOptionsSnapshot::from_ffi(
        unsafe { raw_defaults.assume_init() },
    );

    debug!("defaults={defaults:?}");

    let mut groups = Vec::with_capacity(group_count as usize);

    for group_index in 0..group_count {
        let mut raw_group = MaybeUninit::<C_AdamGroup>::zeroed();


        unsafe { (shims.get_group)(c_optimizer, group_index, raw_group.as_mut_ptr()) };

        let raw_group = unsafe { raw_group.assume_init() };

        let mut states = Vec::with_capacity(raw_group.parameter_count as usize);

        for parameter_index in 0..raw_group.parameter_count {
            debug!("parameter_index={parameter_index}");
            let mut raw_state = MaybeUninit::<C_AdamParamState>::zeroed();

            unsafe { (shims.get_param_state)(
                c_optimizer,
                group_index,
                parameter_index,
                raw_state.as_mut_ptr(),
            ) };

            let raw_state = unsafe { raw_state.assume_init() };

            if raw_state.has_state == 0 {
                debug_assert!(raw_state.exp_avg.is_null());
                debug_assert!(raw_state.exp_avg_sq.is_null());
                debug_assert!(raw_state.max_exp_avg_sq.is_null());

                states.push(None);
                continue;
            }

            /*
             * Convert exp_avg first. If a subsequent conversion fails,
             * the already-created Tensor is dropped and its C++ pointer
             * is freed, so this does not leak partially returned state.
             */
            let exp_avg = owned_tensor_from_ptr(
                raw_state.exp_avg,
                shims.optimizer_name,
                "exp_avg",
            )?;

            let exp_avg_sq = owned_tensor_from_ptr(
                raw_state.exp_avg_sq,
                shims.optimizer_name,
                "exp_avg_sq",
            )?;

            let max_exp_avg_sq =
                if raw_state.max_exp_avg_sq.is_null() {
                    None
                } else {
                    Some(owned_tensor_from_ptr(
                        raw_state.max_exp_avg_sq,
                        shims.optimizer_name,
                        "max_exp_avg_sq",
                    )?)
                };

            states.push(Some(AdamParamStateSnapshot {
                step: raw_state.step,
                exp_avg,
                exp_avg_sq,
                max_exp_avg_sq,
            }));
        }

        groups.push(AdamGroupSnapshot {
            options: AdamOptionsSnapshot::from_ffi(
                raw_group.options,
            ),
            states,
        });
    }

    Ok(AdamOptimizerSnapshot {
        defaults,
        groups,
        tho_device_idx
    })
}


pub unsafe fn restore_adam_snapshot(
    snapshot: &AdamOptimizerSnapshot,
    c_optimizer: *mut C_optimizer
) -> Result<(), ThoError> {
    unsafe { restore_adam_like_snapshot(snapshot, c_optimizer, ADAM_SHIM_FNS) }
}

pub unsafe fn restore_adam_w_snapshot(
    snapshot: &AdamWOptimizerSnapshot,
    c_optimizer: *mut C_optimizer
) -> Result<(), ThoError> {
    unsafe { restore_adam_like_snapshot(snapshot, c_optimizer, ADAM_W_SHIM_FNS) }
}

unsafe fn restore_adam_like_snapshot(snapshot: &AdamOptimizerSnapshot, c_optimizer: *mut C_optimizer, shims: AdamShimFns) -> Result<(), ThoError> {
    let destination_group_count = unsafe { (shims.group_count)(c_optimizer) };

    if destination_group_count != snapshot.groups.len() as u64 {
        return Err(ThoError::new(format!(
            "{} group-count mismatch: destination has {}, \
                 snapshot has {}",
            shims.optimizer_name,
            destination_group_count,
            snapshot.groups.len(),
        )));
    }

    let raw_defaults = snapshot.defaults.to_ffi();

    unsafe { (shims.set_defaults)(
        c_optimizer,
        &raw_defaults,
    ) };

    for (group_index, snapshot_group) in snapshot.groups.iter().enumerate() {
        let mut destination_group =
            MaybeUninit::<C_AdamGroup>::zeroed();


        unsafe { (shims.get_group)(
            c_optimizer,
            group_index as c_ulong,
            destination_group.as_mut_ptr(),
        ) };

        let destination_group =
            unsafe { destination_group.assume_init() };

        if destination_group.parameter_count != snapshot_group.states.len() as u64 {
            return Err(ThoError::new(format!(
                "{} parameter-count mismatch in group \
                     {group_index}: destination has {}, snapshot has {}",
                shims.optimizer_name,
                destination_group.parameter_count,
                snapshot_group.states.len(),
            )));
        }

        let raw_options = snapshot_group.options.to_ffi();


        unsafe { (shims.set_group_options)(
            c_optimizer,
            group_index as c_ulong,
            &raw_options,
        ) };

        for (parameter_index, state) in
            snapshot_group.states.iter().enumerate()
        {
            match state {
                None => {

                    unsafe { (shims.set_param_state)(
                        c_optimizer,
                        group_index as c_ulong,
                        parameter_index as c_ulong,
                        0,
                        0,
                        ptr::null_mut(),
                        ptr::null_mut(),
                        ptr::null_mut(),
                    ) };
                }

                Some(state) => {
                    let max_exp_avg_sq = state
                        .max_exp_avg_sq
                        .as_ref()
                        .map_or(
                            ptr::null_mut(),
                            |tensor| {
                                tensor.as_ptr()
                                    as *mut C_tensor
                            },
                        );


                    unsafe { (shims.set_param_state)(
                        c_optimizer,
                        group_index as c_ulong,
                        parameter_index as c_ulong,
                        1,
                        state.step,
                        state.exp_avg.as_ptr()
                            as *mut C_tensor,
                        state.exp_avg_sq.as_ptr()
                            as *mut C_tensor,
                        max_exp_avg_sq,
                    ) };
                }
            }
        }
    }

    Ok(())
}

pub unsafe fn sgd_snapshot(c_optimizer: *const C_optimizer, tho_device_idx: usize) -> Result<SgdOptimizerSnapshot, ThoError> {
    let group_count = unsafe { tho_sgd_group_count(c_optimizer) };
    let mut raw_defaults = MaybeUninit::<C_SgdOptions>::zeroed();
    unsafe { tho_sgd_get_defaults(c_optimizer, raw_defaults.as_mut_ptr()) };
    let defaults = SgdOptionsSnapshot::from_ffi(unsafe { raw_defaults.assume_init() });

    let mut groups = Vec::with_capacity(group_count as usize);
    for group_index in 0..group_count {
        let mut raw_group = MaybeUninit::<C_SgdGroup>::zeroed();
        unsafe { tho_sgd_get_group(c_optimizer, group_index, raw_group.as_mut_ptr()) };
        let raw_group = unsafe { raw_group.assume_init() };

        let mut states = Vec::with_capacity(raw_group.parameter_count as usize);
        for parameter_index in 0..raw_group.parameter_count {
            let mut raw_state = MaybeUninit::<C_SgdParamState>::zeroed();
            unsafe { tho_sgd_get_param_state(c_optimizer, group_index, parameter_index, raw_state.as_mut_ptr()) };
            let raw_state = unsafe { raw_state.assume_init() };

            if raw_state.has_state == 0 {
                debug_assert!(raw_state.momentum_buffer.is_null());
                states.push(None);
            } else {
                // The C++ shim transfers ownership of the returned momentum-buffer handle to Rust.
                states.push(Some(SgdParamStateSnapshot {
                    momentum_buffer: owned_tensor_from_ptr(raw_state.momentum_buffer, "SGD", "momentum_buffer")?,
                }));
            }
        }

        groups.push(SgdGroupSnapshot {
            options: SgdOptionsSnapshot::from_ffi(raw_group.options),
            states,
        });
    }

    Ok(SgdOptimizerSnapshot {
        defaults,
        groups,
        tho_device_idx,
    })
}

pub unsafe fn restore_sgd_snapshot(snapshot: &SgdOptimizerSnapshot, c_optimizer: *mut C_optimizer) -> Result<(), ThoError> {
    let destination_group_count = unsafe { tho_sgd_group_count(c_optimizer) };
    if destination_group_count != snapshot.groups.len() as c_ulong {
        return Err(ThoError::new(format!("SGD group-count mismatch: destination has {}, snapshot has {}", destination_group_count, snapshot.groups.len())));
    }

    let raw_defaults = snapshot.defaults.to_ffi();
    unsafe { tho_sgd_set_defaults(c_optimizer, &raw_defaults) };

    for (group_index, snapshot_group) in snapshot.groups.iter().enumerate() {
        let mut destination_group = MaybeUninit::<C_SgdGroup>::zeroed();
        unsafe { tho_sgd_get_group(c_optimizer, group_index as c_ulong, destination_group.as_mut_ptr()) };
        let destination_group = unsafe { destination_group.assume_init() };

        if destination_group.parameter_count != snapshot_group.states.len() as c_ulong {
            return Err(ThoError::new(format!("SGD parameter-count mismatch in group {group_index}: destination has {}, snapshot has {}", destination_group.parameter_count, snapshot_group.states.len())));
        }

        let raw_options = snapshot_group.options.to_ffi();
        unsafe { tho_sgd_set_group_options(c_optimizer, group_index as c_ulong, &raw_options) };

        for (parameter_index, state) in snapshot_group.states.iter().enumerate() {
            match state {
                None => unsafe { tho_sgd_set_param_state(c_optimizer, group_index as c_ulong, parameter_index as c_ulong, 0, ptr::null_mut()) },
                Some(state) => unsafe { tho_sgd_set_param_state(c_optimizer, group_index as c_ulong, parameter_index as c_ulong, 1, state.momentum_buffer.as_ptr() as *mut C_tensor) },
            }
        }
    }

    Ok(())
}


#[cfg(test)]
mod tests {
    use log::debug;
    use tch::{Device, Kind, Tensor, nn};
    use tch::Device::Cuda;
    use torch_sys::{C_optimizer, C_tensor};
    use thoosa_shared::ThoError;
    use crate::optimizers::{AdamOptimizerSnapshot, SgdOptimizerSnapshot, adam_snapshot, adam_w_snapshot, restore_adam_snapshot, restore_adam_w_snapshot, restore_sgd_snapshot, sgd_snapshot};

    #[derive(Clone, Copy)]
    enum TestOptimizer {
        Adam,
        AdamW,
        Sgd,
    }

    #[derive(Debug, PartialEq)]
    enum TestOptimizerSnapshot {
        Adam(AdamOptimizerSnapshot),
        Sgd(SgdOptimizerSnapshot),
    }

    impl TestOptimizer {
        fn name(self) -> &'static str {
            match self {
                Self::Adam => "Adam",
                Self::AdamW => "AdamW",
                Self::Sgd => "SGD",
            }
        }

        fn initialized_optimizer(self) -> Result<RawOptimizer, ThoError> {
            // Non-default options verify that both defaults and parameter-group options survive restoration.
            let optimizer = unsafe {
                match self {
                    Self::Adam => torch_sys::ato_adam(0.1, 0.8, 0.95, 0.01, 1e-7, true),
                    Self::AdamW => torch_sys::ato_adamw(0.1, 0.8, 0.95, 0.01, 1e-7, true),
                    Self::Sgd => torch_sys::ato_sgd(0.1, 0.8, 0.0, 0.01, 1),
                }
            };
            RawOptimizer::new(optimizer, self.name())
        }

        unsafe fn snapshot(self, optimizer: *const C_optimizer) -> Result<TestOptimizerSnapshot, ThoError> {
            match self {
                Self::Adam => Ok(TestOptimizerSnapshot::Adam(unsafe { adam_snapshot(optimizer, 1)? })),
                Self::AdamW => Ok(TestOptimizerSnapshot::Adam(unsafe { adam_w_snapshot(optimizer, 1)? })),
                Self::Sgd => Ok(TestOptimizerSnapshot::Sgd(unsafe { sgd_snapshot(optimizer, 1)? })),
            }
        }

        unsafe fn restore(self, snapshot: &TestOptimizerSnapshot, optimizer: *mut C_optimizer) -> Result<(), ThoError> {
            match (self, snapshot) {
                (Self::Adam, TestOptimizerSnapshot::Adam(snapshot)) => unsafe { restore_adam_snapshot(snapshot, optimizer) },
                (Self::AdamW, TestOptimizerSnapshot::Adam(snapshot)) => unsafe { restore_adam_w_snapshot(snapshot, optimizer) },
                (Self::Sgd, TestOptimizerSnapshot::Sgd(snapshot)) => unsafe { restore_sgd_snapshot(snapshot, optimizer) },
                _ => Err(ThoError::new(format!("{} test snapshot type mismatch", self.name()))),
            }
        }

        fn default_optimizer(self) -> Result<RawOptimizer, ThoError> {
            let optimizer = unsafe { match self {
                Self::Adam => {
                    let config = nn::Adam::default();
                    torch_sys::ato_adam(0.1, config.beta1, config.beta2, config.wd, config.eps, config.amsgrad)
                }
                Self::AdamW => {
                    let config = nn::AdamW::default();
                    torch_sys::ato_adamw(0.1, config.beta1, config.beta2, config.wd, config.eps, config.amsgrad)
                }
                Self::Sgd => {
                    let config = nn::Sgd::default();
                    torch_sys::ato_sgd(0.1, config.momentum, config.dampening, config.wd, i32::from(config.nesterov))
                }
            } };
            RawOptimizer::new(optimizer, self.name())
        }
    }

    struct RawOptimizer(*mut C_optimizer);

    impl RawOptimizer {
        fn new(optimizer: *mut C_optimizer, optimizer_name: &str) -> Result<Self, ThoError> {
            if optimizer.is_null() {
                Err(ThoError::new(format!("failed to construct {optimizer_name} optimizer")))
            } else {
                Ok(Self(optimizer))
            }
        }
    }

    impl Drop for RawOptimizer {
        fn drop(&mut self) {
            // Raw torch-sys constructors transfer ownership to the caller, so release every test optimizer on all exit paths.
            unsafe { torch_sys::ato_free(self.0) };
        }
    }

    #[test]
    fn test_tho_snapshot_adam_states()-> Result<(), ThoError>{
        do_test_tho_snapshot_optimizer_states(TestOptimizer::Adam)
    }

    #[test]
    fn test_tho_snapshot_adam_w_states()-> Result<(), ThoError>{
        do_test_tho_snapshot_optimizer_states(TestOptimizer::AdamW)
    }

    #[test]
    fn test_tho_snapshot_sgd_states()-> Result<(), ThoError>{
        do_test_tho_snapshot_optimizer_states(TestOptimizer::Sgd)
    }

    fn do_test_tho_snapshot_optimizer_states(optimizer_kind: TestOptimizer) -> Result<(), ThoError> {
        let (vs_0, optimizer_0) = initialized_optimizer(Cuda(0), optimizer_kind)?;
        let snapshot_0 = unsafe { optimizer_kind.snapshot(optimizer_0.0)? };
        debug!("do_test_tho_snapshot_optimizer_states: optimizer={} snapshot_0={snapshot_0:?}", optimizer_kind.name());

        let optimizer_1 = optimizer_kind.default_optimizer()?;
        let variables = vs_0.variables_.lock().unwrap();
        for variable in &variables.trainable_variables {
            unsafe { torch_sys::ato_add_parameters(optimizer_1.0, variable.tensor.as_ptr() as *mut C_tensor, variable.group) };
        }
        drop(variables);

        unsafe { optimizer_kind.restore(&snapshot_0, optimizer_1.0) }?;
        let snapshot_1 = unsafe { optimizer_kind.snapshot(optimizer_1.0)? };
        debug!("do_test_tho_snapshot_optimizer_states: optimizer={} snapshot_1={snapshot_1:?}", optimizer_kind.name());
        assert_eq!(snapshot_0, snapshot_1);

        Ok(())
    }

    fn initialized_optimizer(device: Device, optimizer_kind: TestOptimizer) -> Result<(nn::VarStore, RawOptimizer), ThoError> {
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        // Simplest possible affine network:
        //
        //     y = x * weight + bias
        //
        // Non-zero initial values ensure weight decay participates in the first update.
        let weight = root.var("weight", &[1], nn::Init::Const(0.25));
        let bias = root.var("bias", &[1], nn::Init::Const(-0.5));

        // Adam variants use AMSGrad, while SGD uses momentum, so every optional tensor-state path is exercised.
        let optimizer = optimizer_kind.initialized_optimizer()?;
        let variables = vs.variables_.lock().unwrap();
        for variable in &variables.trainable_variables {
            unsafe { torch_sys::ato_add_parameters(optimizer.0, variable.tensor.as_ptr() as *mut C_tensor, variable.group) };
        }
        drop(variables);

        let input = Tensor::from_slice(&[2.0_f32]).to_device(device);
        let target = Tensor::from_slice(&[3.0_f32]).to_device(device);

        let prediction = &input * &weight + &bias;
        let loss = (&prediction - &target)
            .pow_tensor_scalar(2)
            .mean(Kind::Float);

        // One step lazily initializes every optimizer's per-parameter tensor state.
        loss.backward();
        unsafe { torch_sys::ato_step(optimizer.0) };
        unsafe { torch_sys::ato_zero_grad(optimizer.0) };

        Ok((vs, optimizer))
    }
}
