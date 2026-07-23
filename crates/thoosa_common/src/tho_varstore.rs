use std::fmt::Debug;
use async_trait::async_trait;
use tch::{no_grad, Kind, Tensor, Device};
use tch::Device::Cpu;
use tch::nn::VarStore;
use tokio::task::JoinSet;
use thoosa_shared::{CopyToDevice, ThoError};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Named, ThoSelector};
use crate::dims::Dims;
use crate::runner::global_ctx::GlobalCtx;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::{ThoTensor, ThoTensorInner};



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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::Cursor;
    use std::sync::{Arc, Mutex};
    use log::debug;
    use tch::Device::{Cpu, Cuda};
    use tch::kind::FLOAT_CPU;
    use tch::nn::{Init, VarStore, Variables};
    use tch::Tensor;
    use thoosa_shared::ThoError;
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::tho_varstore::clone_var_store_to;

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
    async fn test_0() -> Result<(), ThoError> {
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
}
