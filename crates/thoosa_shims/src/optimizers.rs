use std::ffi::{c_int, c_ulong};
use torch_sys::{C_optimizer, C_tensor};

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
    pub fn ato_adam_group_count(
        opt: *mut C_optimizer,
    ) -> c_ulong;

    pub fn ato_adam_get_defaults(
        opt: *mut C_optimizer,
        out: *mut C_AdamOptions,
    );

    pub fn ato_adam_get_group(
        opt: *mut C_optimizer,
        group_index: c_ulong,
        out: *mut C_AdamGroup,
    );

    pub fn ato_adam_get_param_state(
        opt: *mut C_optimizer,
        group_index: c_ulong,
        parameter_index: c_ulong,
        out: *mut C_AdamParamState,
    );

    pub fn ato_adam_set_defaults(
        opt: *mut C_optimizer,
        options: *const C_AdamOptions,
    );

    pub fn ato_adam_set_group_options(
        opt: *mut C_optimizer,
        group_index: c_ulong,
        options: *const C_AdamOptions,
    );

    pub fn ato_adam_set_param_state(
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
#[cfg(test)]
mod tests {
    use tch::{nn, Device, Tensor, Kind};
    use tch::Device::Cuda;
    use tch::nn::{Adam, Optimizer, OptimizerConfig};
    use torch_sys::{C_optimizer, C_tensor};
    use thoosa_shared::ThoError;
    use crate::optimizers::ato_adam_group_count;

    #[test]
    fn test_tho_snapshot_adam_states()-> Result<(), ThoError>{
        let adam = initialized_adam(Cuda(0))?;
        unsafe {
            ato_adam_group_count(adam.1);

        }

        Ok(())
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