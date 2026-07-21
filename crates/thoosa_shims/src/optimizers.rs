use std::ffi::{c_int, c_ulong};
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
        opt: *mut C_optimizer,
    ) -> c_ulong;

    pub fn tho_adam_get_defaults(
        opt: *mut C_optimizer,
        out: *mut C_AdamOptions,
    );

    pub fn tho_adam_get_group(
        opt: *mut C_optimizer,
        group_index: c_ulong,
        out: *mut C_AdamGroup,
    );

    pub fn tho_adam_get_param_state(
        opt: *mut C_optimizer,
        group_index: c_ulong,
        parameter_index: c_ulong,
        out: *mut C_AdamParamState,
    );

    pub fn tho_adam_set_defaults(
        opt: *mut C_optimizer,
        options: *const C_AdamOptions,
    );

    pub fn tho_adam_set_group_options(
        opt: *mut C_optimizer,
        group_index: c_ulong,
        options: *const C_AdamOptions,
    );

    pub fn tho_adam_set_param_state(
        opt: *mut C_optimizer,
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

fn owned_tensor_from_ptr(
    ptr: *mut C_tensor,
    description: &str,
) -> Result<Tensor, ThoError> {
    if ptr.is_null() {
        return Err(ThoError::new(format!(
            "C++ Adam shim returned a null {description}"
        )));
    }

    // The C++ shim returned `new torch::Tensor(...)`.
    // Tensor now owns that pointer and will eventually call at_free().
    Ok(unsafe { Tensor::from_ptr(ptr) })
}

#[cfg(test)]
mod tests {
    use std::ffi::c_ulong;
    use std::mem::MaybeUninit;
    use std::ptr;
    use log::debug;
    use tch::{nn, Device, Tensor, Kind};
    use tch::Device::Cuda;
    use tch::nn::{Adam, Optimizer, OptimizerConfig};
    use torch_sys::{C_optimizer, C_tensor};
    use tracing::field::debug;
    use thoosa_shared::ThoError;
    use crate::optimizers::{tho_adam_group_count, C_AdamOptions, tho_adam_get_defaults, AdamOptionsSnapshot, tho_adam_get_group, C_AdamGroup, C_AdamParamState, tho_adam_get_param_state, AdamParamStateSnapshot, owned_tensor_from_ptr, AdamGroupSnapshot, AdamOptimizerSnapshot, tho_adam_set_defaults, tho_adam_set_group_options, tho_adam_set_param_state};

    #[test]
    fn test_tho_snapshot_adam_states()-> Result<(), ThoError>{
        let (vs_0, c_optimizer_0) = initialized_adam(Cuda(0))?;
        let snapshot_0 = unsafe {adam_snapshot(c_optimizer_0)?};

        debug!("snapshot_0={snapshot_0:?}");

        let c_optimizer_1 = default_adam()?;//initialized_adam(Cuda(0))?.1; //
        let v = vs_0.variables_.lock().unwrap();
        for var in &v.trainable_variables {
            unsafe {
                torch_sys::ato_add_parameters(c_optimizer_1, var.tensor.as_ptr() as *mut C_tensor, var.group)
            }
        }
        drop(v);
        unsafe {restore_adam_snapshot(&snapshot_0, c_optimizer_1)}?;

        let snapshot_1 = unsafe {adam_snapshot(c_optimizer_1)?};
        debug!("snapshot_1={snapshot_1:?}");

        assert_eq!(snapshot_0, snapshot_1);

        Ok(())
    }

    unsafe fn adam_snapshot(c_optimizer: *mut C_optimizer) -> Result<AdamOptimizerSnapshot, ThoError> {
        let group_count = tho_adam_group_count(c_optimizer);

        let mut raw_defaults = MaybeUninit::<C_AdamOptions>::zeroed();


        tho_adam_get_defaults(c_optimizer, raw_defaults.as_mut_ptr());


        debug!("raw_defaults={raw_defaults:?}");

        let defaults = AdamOptionsSnapshot::from_ffi(
            unsafe { raw_defaults.assume_init() },
        );

        debug!("defaults={defaults:?}");

        let mut groups = Vec::with_capacity(group_count as usize);

        for group_index in 0..group_count {
            let mut raw_group = MaybeUninit::<C_AdamGroup>::zeroed();


            tho_adam_get_group(c_optimizer, group_index, raw_group.as_mut_ptr());

            let raw_group = unsafe { raw_group.assume_init() };

            let mut states = Vec::with_capacity(raw_group.parameter_count as usize);

            for parameter_index in 0..raw_group.parameter_count {
                debug!("parameter_index={parameter_index}");
                let mut raw_state = MaybeUninit::<C_AdamParamState>::zeroed();

                tho_adam_get_param_state(
                    c_optimizer,
                    group_index,
                    parameter_index,
                    raw_state.as_mut_ptr(),
                );

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
                    "Adam exp_avg",
                )?;

                let exp_avg_sq = owned_tensor_from_ptr(
                    raw_state.exp_avg_sq,
                    "Adam exp_avg_sq",
                )?;

                let max_exp_avg_sq =
                    if raw_state.max_exp_avg_sq.is_null() {
                        None
                    } else {
                        Some(owned_tensor_from_ptr(
                            raw_state.max_exp_avg_sq,
                            "Adam max_exp_avg_sq",
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
        })
    }

    pub unsafe fn restore_adam_snapshot(
        snapshot: &AdamOptimizerSnapshot,
        c_optimizer: *mut C_optimizer
    ) -> Result<(), ThoError> {
        let destination_group_count = tho_adam_group_count(c_optimizer);

        if destination_group_count != snapshot.groups.len() as u64 {
            return Err(ThoError::new(format!(
                "Adam group-count mismatch: destination has {}, \
                 snapshot has {}",
                destination_group_count,
                snapshot.groups.len(),
            )));
        }

        let raw_defaults = snapshot.defaults.to_ffi();

        tho_adam_set_defaults(
            c_optimizer,
            &raw_defaults,
        );

        for (group_index, snapshot_group) in snapshot.groups.iter().enumerate() {
            let mut destination_group =
                MaybeUninit::<C_AdamGroup>::zeroed();


            tho_adam_get_group(
                c_optimizer,
                group_index as c_ulong,
                destination_group.as_mut_ptr(),
            );

            let destination_group =
                unsafe { destination_group.assume_init() };

            if destination_group.parameter_count != snapshot_group.states.len() as u64 {
                return Err(ThoError::new(format!(
                    "Adam parameter-count mismatch in group \
                     {group_index}: destination has {}, snapshot has {}",
                    destination_group.parameter_count,
                    snapshot_group.states.len(),
                )));
            }

            let raw_options = snapshot_group.options.to_ffi();


            tho_adam_set_group_options(
                c_optimizer,
                group_index as c_ulong,
                &raw_options,
            );

            for (parameter_index, state) in
                snapshot_group.states.iter().enumerate()
            {
                match state {
                    None => {

                            tho_adam_set_param_state(
                                c_optimizer,
                                group_index as c_ulong,
                                parameter_index as c_ulong,
                                0,
                                0,
                                ptr::null_mut(),
                                ptr::null_mut(),
                                ptr::null_mut(),
                            )
                        ;
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


                            tho_adam_set_param_state(
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
                            )
;
                    }
                }
            }
        }

        Ok(())
    }

    pub fn default_adam() -> Result<*mut C_optimizer, ThoError>{
        let adam_config = nn::Adam::default();

        Ok(unsafe { torch_sys::ato_adam(0.1, adam_config.beta1, adam_config.beta2, adam_config.wd, adam_config.eps, adam_config.amsgrad)})
    }

    pub fn initialized_adam(
        device: Device,
    ) -> Result<(nn::VarStore, *mut C_optimizer), ThoError> {
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

        let adam_config = nn::Adam {
            beta1: 0.8,
            beta2: 0.95,
            wd: 0.01,
            eps: 1e-7,

            // Important: causes max_exp_avg_sq to be created.
            amsgrad: true,
        };
        let c_optimizer =unsafe { torch_sys::ato_adam(0.1, adam_config.beta1, adam_config.beta2, adam_config.wd, adam_config.eps, adam_config.amsgrad)};
        let v = vs.variables_.lock().unwrap();
        for var in &v.trainable_variables {
            unsafe {
                torch_sys::ato_add_parameters(c_optimizer, var.tensor.as_ptr() as *mut C_tensor, var.group)
            }
        }
        drop(v);

        // let mut optimizer = adam_config.build(&vs, 1e-3).map_err(|e|ThoError::from_msg(format!("e={e:?}")))?;

        let input = Tensor::from_slice(&[2.0_f32]).to_device(device);
        let target = Tensor::from_slice(&[3.0_f32]).to_device(device);

        let prediction = &input * &weight + &bias;
        let loss = (&prediction - &target)
            .pow_tensor_scalar(2)
            .mean(Kind::Float);

        // Lazily initializes Adam state for both weight and bias,
        // then increments their step counters to 1.
        // optimizer.backward_step(&loss);

        loss.backward();
        unsafe {torch_sys::ato_step(c_optimizer)};

        // `backward_step` does not clear gradients after stepping.
        // Uncomment this if the returned VarStore must have undefined grads.
        //
        // optimizer.zero_grad();
        unsafe {torch_sys::ato_zero_grad(c_optimizer)};

        Ok((vs, c_optimizer))
    }
}