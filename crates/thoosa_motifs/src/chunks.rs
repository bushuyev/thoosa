use log::{debug, trace, warn};
use std::collections::VecDeque;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem;
use std::ops::Deref;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::Arc;
use tch::kind::FLOAT_CPU;
use tch::{Device, IndexOp, Kind, TchError, Tensor};
use thoosa_common::dims::{TCD, TCDF};
use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig};
use thoosa_common::loader::SyncWrapper;
use thoosa_common::utils::{SplitQueue, Splittable};
use thoosa_shared::TensorIdSliceIdx;
use thoosa_shared::resource::ResourceId;

#[derive(Debug)]
pub struct ChunkOfSlicedTensors {
    pub(crate) resource_ids: Vec<ResourceId>,
    pub(crate) sizes: Vec<i64>, //num splits in a ResourceId
    pub t: SyncWrapper<Tensor>,
    step: u32,
    split_at: Option<i64>, //split within tensor
}

pub struct ChunkOfSlicedTensorsIter {
    iter: Box<dyn Iterator<Item = TensorIdSliceIdx> + Send>,
}

impl ChunkOfSlicedTensors {
    pub fn new(keys: Vec<ResourceId>, sizes: Vec<i64>, step: u32, t: Tensor) -> Self {
        Self {
            resource_ids: keys,
            sizes, //number of slices produced from each tensor
            step,
            t: SyncWrapper::wrap_tensor(t),
            split_at: None,
        }
    }

    // //TODO clone
    pub fn to_vec(&self) -> Vec<TensorIdSliceIdx> {
        let step = self.step as i64;
        let split_at_idx = self
            .split_at
            .map(|split_at_idx| split_at_idx * step)
            .unwrap_or(0);

        self.sizes
            .clone()
            .into_iter()
            .zip(self.resource_ids.clone().into_iter())
            .enumerate()
            .flat_map(|(t_i, (size, id))| {
                (0..size).map(move |i| (id.clone(), i, if t_i == 0 { split_at_idx } else { 0 }))
            })
            .map(move |v| TensorIdSliceIdx(v.0, (v.2 + step * v.1) as u32))
            .collect()
    }

    pub fn keys_sizes(&self) -> Vec<(ResourceId, i64)> {
        self.resource_ids
            .iter()
            .zip(self.sizes.iter())
            .map(|v| (v.0.clone(), v.1.clone()))
            .collect()
    }

    pub fn is_split(&self) -> bool {
        self.split_at.is_some()
    }

    //TODO Option instead  of [0.0]
    pub fn drop_tensor(&mut self) {
        mem::swap(
            &mut self.t,
            &mut SyncWrapper::wrap_tensor(Tensor::from_slice(&[0.0])),
        );
    }



    pub fn start_split_queue(chunk: ChunkOfSlicedTensors) -> SplitQueue<2, ChunkOfSlicedTensors> {
        SplitQueue::new(chunk)
    }
}

impl Splittable<2> for ChunkOfSlicedTensors {
    fn split(&self) -> Option<[Self; 2]> {
        //only one tensor in the chunk, splitting tensor
        if self.resource_ids.len() == 1 {
            assert_eq!(self.sizes.len(), 1);

            let split_at_idx = self.sizes[0] / 2;
            debug!(
                "ChunkOfSlicedTensors::split: 0 self.sizes={:?} self.keys={:?} self.t.size()={:?} split_at_idx={split_at_idx}",
                self.sizes,
                self.resource_ids,
                self.t.size()
            );
            if split_at_idx > 0 {
                Some([
                    Self {
                        resource_ids: self.resource_ids.clone(),
                        sizes: vec![split_at_idx],
                        step: self.step,
                        t: SyncWrapper::wrap_tensor(self.t.i((.., ..split_at_idx, ..))),
                        split_at: self.split_at,
                    },
                    Self {
                        resource_ids: self.resource_ids.clone(),
                        sizes: vec![self.sizes[0] - split_at_idx],
                        step: self.step,
                        t: SyncWrapper::wrap_tensor(self.t.i((.., split_at_idx.., ..))),
                        split_at: self
                            .split_at
                            .map(|v| Some(v + split_at_idx))
                            .unwrap_or(Some(split_at_idx)),
                    },
                ])
            } else {
                None
            }
        //more than one tensor in the chunk, splitting chunk
        } else {
            let split_at = self.resource_ids.len() / 2;

            assert!(split_at > 0);

            let split_at_tensors_idx = self.sizes.iter().take(split_at).sum::<i64>();

            debug!(
                "ChunkOfSlicedTensors::split: 1 self.sizes={:?} self.keys={:?} self.t.size()={:?} split_at={split_at} split_at_idx={split_at_tensors_idx}",
                self.sizes,
                self.resource_ids,
                self.t.size()
            );

            Some([
                Self {
                    resource_ids: self.resource_ids[..split_at].to_vec(),
                    sizes: self.sizes[..split_at].to_vec(),
                    step: self.step,
                    t: SyncWrapper::wrap_tensor(self.t.i((.., ..split_at_tensors_idx, ..))),

                    split_at: self.split_at,
                },
                Self {
                    resource_ids: self.resource_ids[split_at..].to_vec(),
                    sizes: self.sizes[split_at..].to_vec(),
                    step: self.step,
                    t: SyncWrapper::wrap_tensor(self.t.i((.., split_at_tensors_idx.., ..))),
                    split_at: None, //self.split_at.map(|v|Some(v+ split_at_tensors_idx)).unwrap_or(Some(split_at_tensors_idx)),
                },
            ])
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::chunks::{ChunkOfSlicedTensors, ChunkOfSlicedTensorsIter};
    use log::debug;
    use std::slice::Iter;
    use tch::Device::Cpu;
    use tch::Kind::Float;
    use tch::Tensor;
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::loader::SyncWrapper;
    use thoosa_common::utils::Splitter;
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_shared::resource::ResourceId;
    use thoosa_shared::test_utils::init_test_logging;
    use thoosa_shared::{TensorIdSliceIdx, ThoosaError};

    #[ctor::ctor]
    fn init() {
        init_test_logging();
    }



}
