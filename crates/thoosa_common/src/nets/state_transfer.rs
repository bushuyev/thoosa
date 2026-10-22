use thoosa_common::tch_error::tch_result;
use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};
use tch::{Device, Kind, Tensor, nn::VarStore};
use thoosa_shared::ThoError;
use thoosa_shared::ipc_types::{TensorKey};
use thoosa_common::ipc::tensor_info::{WorkerTensorInfoPayload};
use thoosa_common::runner::sync_stage::StageStateInfo;
use crate::tho_device::ThoDevice;
use crate::cuda_utils::tensor_mem_size;
use crate::tho_tensor::ThoTensor;

/// Tensor storage and its schema have separate lifetimes: the schema remains in
/// the worker while the parent owns the CPU copy of the tensors.
#[derive(Debug, Default)]
pub(crate) struct StateTensors(BTreeMap<String, StateTensor>);

#[derive(Debug)]
struct StateTensor {
    tensor: Option<ThoTensor>,
    kind: Kind,
    shape: Vec<i64>,
}

impl StateTensors {
    pub(crate) fn from_info(info: &StageStateInfo) -> Self {
        Self(info.tensors.iter().map(|tensor| (tensor.tensor_key.to_tensor_key(), StateTensor {
            tensor: None, kind: tensor.kind.into(), shape: tensor.shape.clone(),
        })).collect())
    }

    pub(crate) fn insert(&mut self, key: String, tensor: Tensor) -> Result<(), ThoError> {
        self.0.insert(key, StateTensor { kind: tensor.kind(), shape: tensor.size(), tensor: Some(ThoTensor::new((), tensor)?) });
        Ok(())
    }

    pub(crate) fn info(&self) -> StageStateInfo {
        // CudaWorker supplies the installed endpoint identity. Model state does
        // not need to know which crate or stage installed its owner.
        StageStateInfo::new(self.0.iter().map(|(key, slot)| WorkerTensorInfoPayload::new(TensorKey::stage_state("", "", key), slot.kind.into(), slot.shape.clone(), tensor_mem_size(slot.kind, &slot.shape))).collect(), None)
    }

    pub(crate) fn get(&self, key: &str) -> Result<Tensor, ThoError> {
        let slot = self.0.get(key).ok_or_else(|| ThoError::NotFound(Some(key.to_owned())))?;
        let tensor = slot.tensor.as_ref().ok_or_else(|| ThoError::new("model state is offloaded or incompletely restored"))?;
        // The transfer request owns this alias until the device copy completes.
        Ok(tch_result(tensor.as_ref().f_alias())?)
    }

    pub(crate) fn set(&mut self, key: &str, tensor: Tensor, device: usize) -> Result<(), ThoError> {
        let slot = self.0.get_mut(key).ok_or_else(|| ThoError::NotFound(Some(key.to_owned())))?;
        if slot.tensor.is_some() {
            return Err(ThoError::new(format!("model state tensor {key} is already loaded")));
        }
        if tensor.kind() != slot.kind || tensor.size() != slot.shape || tensor.device() != Device::from(ThoDevice::from(device)) {
            return Err(ThoError::new(format!("model state tensor {key} has incompatible kind, shape, or device")));
        }
        slot.tensor = Some(ThoTensor::new((), tensor)?);
        Ok(())
    }

    pub(crate) fn clear(&mut self) {
        for slot in self.0.values_mut() { slot.tensor = None; }
    }

    pub(crate) fn complete(&self) -> bool {
        self.0.values().all(|slot| slot.tensor.is_some())
    }

    pub(crate) fn has_device(&self, device: usize) -> bool {
        self.0.values().any(|slot| slot.tensor.as_ref().is_some_and(|t| t.tho_device_idx() == device))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct VarStoreState {
    kind: thoosa_common::ipc::tensor_info::IpcTensorKind,
    variables: Vec<VariableState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VariableState {
    name: String,
    trainable: Option<(usize, usize)>,
    requires_grad: bool,
}

impl VarStoreState {
    pub(crate) fn capture(vs: &VarStore, tensors: &mut StateTensors) -> Result<Self, ThoError> {
        let vars = vs.variables_.lock().map_err(|_| ThoError::new("VarStore lock poisoned"))?;
        let mut variables = Vec::with_capacity(vars.named_variables.len());
        for (name, tensor) in &vars.named_variables {
            let trainable = vars.trainable_variables.iter().enumerate().find_map(|(order, var)| var.tensor.is_set_to(tensor).then_some((order, var.group)));
            variables.push(VariableState { name: name.clone(), trainable, requires_grad: tensor.requires_grad() });
            tensors.insert(format!("var/{name}"), tch_result(tensor.f_detach())?)?;
        }
        // Optimizer snapshots address parameters by group and registration
        // order, not by the arbitrary iteration order of named_variables.
        variables.sort_by(|a, b| a.trainable.map(|(order, _)| order).unwrap_or(usize::MAX).cmp(&b.trainable.map(|(order, _)| order).unwrap_or(usize::MAX)).then(a.name.cmp(&b.name)));
        Ok(Self { kind: vs.kind().into(), variables })
    }

    pub(crate) fn restore(&self, tensors: &StateTensors, device: usize) -> Result<VarStore, ThoError> {
        let mut vs = VarStore::new(ThoDevice::from(device).into());
        vs.set_kind(self.kind.into());
        for var in &self.variables {
            let tensor = tensors.get(&format!("var/{}", var.name))?;
            let (trainable, group) = var.trainable.map_or((false, 0), |(_, group)| (true, group));
            let (parent, name) = var.name.rsplit_once('.').unwrap_or(("", &var.name));
            let mut path = vs.root().set_group(group);
            for component in parent.split('.').filter(|s| !s.is_empty()) { path = path.sub(component); }
            let registered = path.add(name, tensor, trainable);
            let _ = tch_result(registered.f_set_requires_grad(var.requires_grad))?;
        }
        Ok(vs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_store_state_preserves_buffers_groups_and_order() -> Result<(), ThoError> {
        let mut original = VarStore::new(Device::Cpu);
        original.set_kind(Kind::Double);
        // Registration order deliberately differs from name order, with a
        // frozen trainable parameter and a non-trainable integer buffer.
        let frozen = original.root().sub("net").set_group(2).add("z", tch_result(Tensor::f_ones([2], (Kind::Double, Device::Cpu)))?, true);
        let _ = tch_result(frozen.f_set_requires_grad(false))?;
        let _ = original.root().set_group(0).add("a", tch_result(Tensor::f_ones([1], (Kind::Float, Device::Cpu)))?, true);
        let _ = original.root().add("counter", tch_result(Tensor::f_from_slice(&[7i64]))?, false);
        let mut tensors = StateTensors::default();
        let metadata = VarStoreState::capture(&original, &mut tensors)?;
        let cpu_copy = tensors.0.keys().map(|key| Ok((key.clone(), tensors.get(key)?))).collect::<Result<Vec<_>, ThoError>>()?;
        tensors.clear();
        assert!(!tensors.complete());
        for (key, tensor) in cpu_copy.into_iter().rev() { tensors.set(&key, tensor, 0)?; }
        let restored = metadata.restore(&tensors, 0)?;
        assert_eq!(restored.kind(), Kind::Double);
        let vars = restored.variables_.lock().unwrap();
        assert_eq!(vars.trainable_variables.len(), 2);
        assert_eq!(vars.trainable_variables[0].group, 2);
        assert!(vars.trainable_variables[0].tensor.is_set_to(&vars.named_variables["net.z"]));
        assert!(!vars.named_variables["net.z"].requires_grad());
        assert_eq!(vars.trainable_variables[1].group, 0);
        assert!(vars.trainable_variables[1].tensor.is_set_to(&vars.named_variables["a"]));
        assert!(vars.named_variables["a"].requires_grad());
        assert_eq!(vars.named_variables["a"].kind(), Kind::Float);
        assert_eq!(vars.named_variables["counter"].kind(), Kind::Int64);
        assert_eq!(tch_result(vars.named_variables["counter"].f_int64_value(&[0]))?, 7);
        Ok(())
    }
}
