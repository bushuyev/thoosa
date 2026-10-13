// Only scalar metadata is serialized here; tensor storage uses the existing
// parent-owned state-copy protocol, including optimizer state tensors.
use super::*;
use serde::{Deserialize, Serialize};
use thoosa_shims::optimizers::{AdamOptionsSnapshot, SgdOptionsSnapshot};

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct TrainingMetadata {
    pub variables: VarStoreState,
    pub optimizer: OptimizerMetadata,
}

#[typetag::serde]
impl thoosa_shared::ipc_types::WorkerOutput for TrainingMetadata {}

#[derive(Debug, Serialize, Deserialize)]
pub(super) enum OptimizerMetadata {
    Adam(AdamMetadata),
    AdamW(AdamMetadata),
    Sgd(SgdMetadata),
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct AdamMetadata {
    #[serde(with = "AdamOptions")]
    defaults: AdamOptionsSnapshot,
    groups: Vec<AdamGroup>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AdamGroup {
    #[serde(with = "AdamOptions")]
    options: AdamOptionsSnapshot,
    // None means this parameter has never acquired optimizer state.
    states: Vec<Option<(i64, bool)>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct SgdMetadata {
    #[serde(with = "SgdOptions")]
    defaults: SgdOptionsSnapshot,
    groups: Vec<SgdGroup>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SgdGroup {
    #[serde(with = "SgdOptions")]
    options: SgdOptionsSnapshot,
    states: Vec<bool>,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "AdamOptionsSnapshot")]
struct AdamOptions {
    lr: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64, amsgrad: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "SgdOptionsSnapshot")]
struct SgdOptions {
    lr: f64, momentum: f64, dampening: f64, weight_decay: f64, nesterov: bool,
}

impl OptimizerMetadata {
    pub fn capture(snapshot: &OptimizerSnapshot) -> Self {
        fn adam(snapshot: &AdamOptimizerSnapshot) -> AdamMetadata {
            AdamMetadata { defaults: snapshot.defaults, groups: snapshot.groups.iter().map(|group| AdamGroup {
                options: group.options,
                states: group.states.iter().map(|state| state.as_ref().map(|state| (state.step, state.max_exp_avg_sq.is_some()))).collect(),
            }).collect() }
        }
        match snapshot {
            OptimizerSnapshot::Adam(snapshot) => Self::Adam(adam(snapshot)),
            OptimizerSnapshot::AdamW(snapshot) => Self::AdamW(adam(snapshot)),
            OptimizerSnapshot::Sgd(snapshot) => Self::Sgd(SgdMetadata { defaults: snapshot.defaults, groups: snapshot.groups.iter().map(|group| SgdGroup {
                options: group.options, states: group.states.iter().map(Option::is_some).collect(),
            }).collect() }),
        }
    }

    pub fn into_snapshot(self) -> OptimizerSnapshot {
        fn adam(metadata: AdamMetadata) -> AdamOptimizerSnapshot {
            AdamOptimizerSnapshot { defaults: metadata.defaults, tho_device_idx: 0, groups: metadata.groups.into_iter().map(|group| AdamGroupSnapshot {
                options: group.options,
                states: group.states.into_iter().map(|state| state.map(|(step, maximum)| AdamParamStateSnapshot {
                    step, exp_avg: Tensor::new(), exp_avg_sq: Tensor::new(), max_exp_avg_sq: maximum.then(Tensor::new),
                })).collect(),
            }).collect() }
        }
        match self {
            Self::Adam(metadata) => OptimizerSnapshot::Adam(adam(metadata)),
            Self::AdamW(metadata) => OptimizerSnapshot::AdamW(adam(metadata)),
            Self::Sgd(metadata) => OptimizerSnapshot::Sgd(SgdOptimizerSnapshot { defaults: metadata.defaults, tho_device_idx: 0, groups: metadata.groups.into_iter().map(|group| SgdGroupSnapshot {
                options: group.options,
                states: group.states.into_iter().map(|present| present.then(|| SgdParamStateSnapshot { momentum_buffer: Tensor::new() })).collect(),
            }).collect() }),
        }
    }
}
