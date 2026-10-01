use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::ops::{Deref, DerefMut};
use thoosa_common::data_with_extra::DataWithExtra;
use thoosa_common::dims::Batch;
use thoosa_common::ipc::cuda_wokrer::{AllTensors, WorkerStageOutput};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::ThoError;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::ipc_types::WorkerOutput;
use thoosa_shared::traits::WithId;

use crate::dims::{ChannelsMotifsSlices, LabelProbability, MotifProbability, MotifSlices};

/// Keeps motif-owned IPC registrations outside `thoosa_common` while preserving the tensor payload.
#[derive(Debug, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct MotifWorkerOutput<O>(O);

impl<O> MotifWorkerOutput<O> {
    pub fn into_inner(self) -> O {
        self.0
    }
}

impl<O> From<O> for MotifWorkerOutput<O> {
    fn from(value: O) -> Self {
        Self(value)
    }
}

impl<O> Deref for MotifWorkerOutput<O> {
    type Target = O;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<O> DerefMut for MotifWorkerOutput<O> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<O> Output for MotifWorkerOutput<O>
where
    O: Output,
{
}

impl<O> WithId for MotifWorkerOutput<O>
where
    O: WithId,
{
    type ID = O::ID;

    fn id(&self) -> &Self::ID {
        self.0.id()
    }
}

impl<O> WorkerStageOutput for MotifWorkerOutput<O>
where
    O: WorkerStageOutput,
{
    fn encode_from_worker(self, all_cuda_tensors: &AllTensors, tho_device: &ThoDevice) -> Result<Self, ThoError> {
        Ok(Self(self.0.encode_from_worker(all_cuda_tensors, tho_device)?))
    }
}

pub type MotifThoTensor<D, Id = SynKey> = MotifWorkerOutput<ThoTensor<D, Id>>;

#[typetag::serde(name = "ThoTensor_MotifSlices")]
impl WorkerOutput for MotifThoTensor<MotifSlices> {}

#[typetag::serde(name = "ThoTensor_ChannelsMotifsSlices")]
impl WorkerOutput for MotifThoTensor<ChannelsMotifsSlices> {}

#[typetag::serde(name = "ThoTensor_MotifProbability")]
impl WorkerOutput for MotifThoTensor<MotifProbability> {}

#[typetag::serde(name = "ThoTensor_LabelProbability")]
impl WorkerOutput for MotifThoTensor<LabelProbability> {}

#[typetag::serde(name = "ThoTensor_Batch_LabelProbability")]
impl WorkerOutput for MotifThoTensor<Batch<LabelProbability>> {}

pub type MotifProbabilitiesWorkerOutput = MotifWorkerOutput<DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>>;

#[typetag::serde(name = "DataWithExtra")]
impl WorkerOutput for MotifProbabilitiesWorkerOutput {}
