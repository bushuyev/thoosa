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


#[cfg(test)]
mod tests {
    use log::debug;
    use tch::{Device, Kind, Tensor, nn};
    use tch::Device::Cuda;
    use torch_sys::{C_optimizer, C_tensor};
    use thoosa_shared::ThoError;
    use crate::optimizers::{AdamOptimizerSnapshot, adam_snapshot, adam_w_snapshot, restore_adam_snapshot, restore_adam_w_snapshot};

    #[derive(Clone, Copy)]
    enum AdamFamily {
        Adam,
        AdamW,
    }

    impl AdamFamily {
        fn name(self) -> &'static str {
            match self {
                Self::Adam => "Adam",
                Self::AdamW => "AdamW",
            }
        }

        fn create(self, lr: f64, beta1: f64, beta2: f64, wd: f64, eps: f64, amsgrad: bool) -> Result<RawOptimizer, ThoError> {
            let optimizer = unsafe {
                match self {
                    Self::Adam => torch_sys::ato_adam(lr, beta1, beta2, wd, eps, amsgrad),
                    Self::AdamW => torch_sys::ato_adamw(lr, beta1, beta2, wd, eps, amsgrad),
                }
            };
            if optimizer.is_null() {
                Err(ThoError::new(format!("failed to construct {} optimizer", self.name())))
            } else {
                Ok(RawOptimizer(optimizer))
            }
        }

        unsafe fn snapshot(self, optimizer: *const C_optimizer) -> Result<AdamOptimizerSnapshot, ThoError> {
            match self {
                Self::Adam => unsafe { adam_snapshot(optimizer, 1) },
                Self::AdamW => unsafe { adam_w_snapshot(optimizer, 1) },
            }
        }

        unsafe fn restore(self, snapshot: &AdamOptimizerSnapshot, optimizer: *mut C_optimizer) -> Result<(), ThoError> {
            match self {
                Self::Adam => unsafe { restore_adam_snapshot(snapshot, optimizer) },
                Self::AdamW => unsafe { restore_adam_w_snapshot(snapshot, optimizer) },
            }
        }

        fn default_optimizer(self) -> Result<RawOptimizer, ThoError> {
            match self {
                Self::Adam => {
                    let config = nn::Adam::default();
                    self.create(0.1, config.beta1, config.beta2, config.wd, config.eps, config.amsgrad)
                }
                Self::AdamW => {
                    let config = nn::AdamW::default();
                    self.create(0.1, config.beta1, config.beta2, config.wd, config.eps, config.amsgrad)
                }
            }
        }
    }

    struct RawOptimizer(*mut C_optimizer);

    impl Drop for RawOptimizer {
        fn drop(&mut self) {
            // Raw torch-sys constructors transfer ownership to the caller, so release both test optimizers on every exit path.
            unsafe { torch_sys::ato_free(self.0) };
        }
    }

    #[test]
    fn test_tho_snapshot_adam_states()-> Result<(), ThoError>{
        do_test_tho_snapshot_adam_states(AdamFamily::Adam)
    }

    #[test]
    fn test_tho_snapshot_adam_w_states()-> Result<(), ThoError>{
        do_test_tho_snapshot_adam_states(AdamFamily::AdamW)
    }

    fn do_test_tho_snapshot_adam_states(family: AdamFamily) -> Result<(), ThoError> {
        let (vs_0, optimizer_0) = initialized_adam_family(Cuda(0), family)?;
        let snapshot_0 = unsafe { family.snapshot(optimizer_0.0)? };
        debug!("do_test_tho_snapshot_adam_states: optimizer={} snapshot_0={snapshot_0:?}", family.name());

        let optimizer_1 = family.default_optimizer()?;
        let variables = vs_0.variables_.lock().unwrap();
        for variable in &variables.trainable_variables {
            unsafe { torch_sys::ato_add_parameters(optimizer_1.0, variable.tensor.as_ptr() as *mut C_tensor, variable.group) };
        }
        drop(variables);

        unsafe { family.restore(&snapshot_0, optimizer_1.0) }?;
        let snapshot_1 = unsafe { family.snapshot(optimizer_1.0)? };
        debug!("do_test_tho_snapshot_adam_states: optimizer={} snapshot_1={snapshot_1:?}", family.name());
        assert_eq!(snapshot_0, snapshot_1);

        Ok(())
    }

    fn initialized_adam_family(device: Device, family: AdamFamily) -> Result<(nn::VarStore, RawOptimizer), ThoError> {
        let vs = nn::VarStore::new(device);
        let root = vs.root();

        // Simplest possible affine network:
        //
        //     y = x * weight + bias
        //
        // Non-zero initial values ensure weight decay also participates
        // in the first Adam update.
        let weight = root.var("weight", &[1], nn::Init::Const(0.25));
        let bias = root.var("bias", &[1], nn::Init::Const(-0.5));

        // AMSGrad ensures the optional max_exp_avg_sq tensor is included in both optimizer variants' snapshots.
        let optimizer = family.create(0.1, 0.8, 0.95, 0.01, 1e-7, true)?;
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

        // One step lazily initializes all Adam-family state tensors and advances their counters.
        loss.backward();
        unsafe { torch_sys::ato_step(optimizer.0) };
        unsafe { torch_sys::ato_zero_grad(optimizer.0) };

        Ok((vs, optimizer))
    }
}
