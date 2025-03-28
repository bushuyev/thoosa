use std::collections::VecDeque;
use tch::{Device, IndexOp, Kind, Tensor};
use std::fmt::Debug;
use std::marker::PhantomData;
use log::{debug, trace, warn};
use std::mem;
use std::ops::Deref;
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::Arc;
use thoosa_common::loader::SyncWrapper;
use thoosa_common::train_nets::{FindMotifsRunConfig, FindMotifsRunConfigApplicable};
use thoosa_common::utils::{SplitQueue, Splittable};
use thoosa_shared::resource::ResourceId;
use thoosa_shared::run_configs::Applicable;
use thoosa_shared::TensorIdSliceIdx;

#[derive(Debug)]
pub struct ChunkOfSlicedTensors where {
    pub(crate) resource_ids:Vec <ResourceId>,
    pub(crate)  sizes: Vec<i64>,//num splits in a ResourceId
    pub   t: SyncWrapper<Tensor>,
    step:u32,
    split_at:Option<i64>//split within tensor
}

pub struct ChunkOfSlicedTensorsIter {
    iter:Box<dyn Iterator<Item=TensorIdSliceIdx> + Send>
}

impl  ChunkOfSlicedTensors {
    pub fn new(keys:  Vec <ResourceId>, sizes: Vec<i64>, step:u32, t: Tensor) -> Self {
        Self {
            resource_ids: keys,
            sizes,//number of slices produced from each tensor
            step,
            t:SyncWrapper::new_t(t),
            split_at: None,
        }
    }

    // //TODO clone
    pub fn to_vec(&self) -> Vec<TensorIdSliceIdx>{
        let step = self.step as i64;
        let split_at_idx = self.split_at.map(|split_at_idx| split_at_idx * step).unwrap_or(0);

        self.sizes.clone().into_iter()
            .zip(self.resource_ids.clone().into_iter())
            .enumerate()
            .flat_map(
                |(t_i, (size, id))|(0..size).map(move |i|(id.clone(), i, if t_i == 0 {split_at_idx} else {0}))
            )
            .map(move |v| TensorIdSliceIdx(v.0, (v.2 + step * v.1) as u32))
            .collect()
    }

    pub fn keys_sizes(&self)->Vec<(ResourceId, i64)>{
        self.resource_ids.iter().zip(self.sizes.iter()).map(|v|(v.0.clone(), v.1.clone())).collect()
    }

    pub fn is_split(&self) -> bool {
        self.split_at.is_some()
    }


    //TODO Option instead  of [0.0]
    pub fn drop_tensor(&mut self) {
        mem::swap(&mut self.t, &mut SyncWrapper::new_t(Tensor::from_slice(&[0.0])));
    }

    /// returns:  channels x t.size = num slices = 1+(len-window)/step x window
    pub fn slice_tensors_chunk(run_config:&FindMotifsRunConfigApplicable, tensors_chunk: &[SyncWrapper<Box<Tensor>>], keys: Vec <ResourceId>, device: Device) -> ChunkOfSlicedTensors {

        debug!("ChunkOfSlicedTensors: applicable={run_config:?}");

        let  (tensor, sizes) = run_config.apply(tensors_chunk).expect("ChunkOfSlicedTensors apply failed");


        debug!("ChunkOfSlicedTensors::slice_tensors_chunk: final tensor.size={:?} device={:?}", tensor.size(), tensor.device());

        ChunkOfSlicedTensors::new(
            keys,
            sizes,
            run_config.step(),
            tensor,
        )
    }

    pub fn start_split_queue(chunk:ChunkOfSlicedTensors) -> SplitQueue<2, ChunkOfSlicedTensors>{
        SplitQueue::new(chunk)
    }
}

impl Splittable<2> for ChunkOfSlicedTensors {
    fn split(&self) -> Option<[Self; 2]> {
        //only one tensor in the chunk, splitting tensor
        if self.resource_ids.len() == 1 {
            assert_eq!(self.sizes.len(), 1);

            let split_at_idx = self.sizes[0]/2;
            debug!("ChunkOfSlicedTensors::split: 0 self.sizes={:?} self.keys={:?} self.t.size()={:?} split_at_idx={split_at_idx}", self.sizes, self.resource_ids, self.t.size());
            if split_at_idx > 0 {
                Some([
                    Self {
                        resource_ids: self.resource_ids.clone(),
                        sizes: vec![split_at_idx],
                        step: self.step,
                        t: SyncWrapper::new_t(self.t.i((.., ..split_at_idx, ..))),
                        split_at: self.split_at,
                    },
                    Self {
                        resource_ids: self.resource_ids.clone(),
                        sizes: vec![self.sizes[0] - split_at_idx],
                        step: self.step,
                        t: SyncWrapper::new_t(self.t.i((.., split_at_idx.., ..))),
                        split_at: self.split_at.map(|v| Some(v + split_at_idx)).unwrap_or(Some(split_at_idx)),
                    }
                ])
            } else {
                None
            }
        //more than one tensor in the chunk, splitting chunk
        } else {
            let split_at = self.resource_ids.len() / 2;

            assert!(split_at > 0);

            let split_at_tensors_idx = self.sizes.iter().take(split_at).sum::<i64>();

            debug!("ChunkOfSlicedTensors::split: 1 self.sizes={:?} self.keys={:?} self.t.size()={:?} split_at={split_at} split_at_idx={split_at_tensors_idx}", self.sizes, self.resource_ids, self.t.size());

            Some([
                Self {
                    resource_ids: self.resource_ids[..split_at].to_vec(),
                    sizes: self.sizes[..split_at].to_vec(),
                    step: self.step,
                    t: SyncWrapper::new_t(self.t.i((.., ..split_at_tensors_idx, ..))),

                    split_at: self.split_at,
                },
                Self {
                    resource_ids: self.resource_ids[split_at..].to_vec(),
                    sizes: self.sizes[split_at..].to_vec(),
                    step: self.step,
                    t: SyncWrapper::new_t(self.t.i((.., split_at_tensors_idx.., ..))),
                    split_at: None, //self.split_at.map(|v|Some(v+ split_at_tensors_idx)).unwrap_or(Some(split_at_tensors_idx)),
                }
            ])
        }
    }

}


#[cfg(test)]
mod tests {
    use std::slice::Iter;
    use log::debug;
    use tch::Device::Cpu;
    use tch::Kind::Float;
    use tch::Tensor;
    use thoosa_common::loader::SyncWrapper;
    use thoosa_common::train_nets::{DummyFindMotifsRunConfigCtx, FindMotifsRunConfig, FindMotifsRunConfigCtx};
    use thoosa_common::utils::Splitter;
    use thoosa_shared::{TensorIdSliceIdx, ThoosaError};
    use thoosa_shared::resource::ResourceId;
    use thoosa_shared::test_utils::init_test_logging;
    use crate::chunks::{ChunkOfSlicedTensors, ChunkOfSlicedTensorsIter};

    #[ctor::ctor]
    fn init() {
        init_test_logging();
    }

    #[test]
    fn test_slice_tensors_chunk(){
        //2 channels
        let tensor_a = Tensor::from_slice(&[
            0.00, 0.01, 0.02, 0.03,
            0.10, 0.11, 0.12, 0.13,
        ]).view([2, -1]).totype(Float); // Shape: [2, 4]

        let tensor_b = Tensor::from_slice(&[
            1.00, 1.01, 1.02,
            1.10, 1.11, 1.12,
        ]).view([2, -1]).totype(Float); // Shape: [2, 3]


        let chunk = ChunkOfSlicedTensors::slice_tensors_chunk(
            &FindMotifsRunConfig::new(None, 1, 2, None, None).to_applicable(DummyFindMotifsRunConfigCtx),
            &vec![tensor_a.into(), tensor_b.into()], vec![0, 1],
            Cpu
        );

        debug!("result={}", chunk.t);

        let expected = Tensor::stack(&[
            Tensor::from_slice(&[ //channel 0
                0.00, 0.01,
                0.01, 0.02,
                0.02, 0.03,

                1.00, 1.01,
                1.01, 1.02,
            ]).view((5,2)),
            Tensor::from_slice(&[ //channel 1
                0.10, 0.11,
                0.11, 0.12,
                0.12, 0.13,

                1.10, 1.11,
                1.11, 1.12,
            ]).view((5,2)),
        ], 0).totype(Float);

        debug!("expected={}", expected);

        assert_eq!(*chunk.t, expected);

    }

    #[test]
    fn test_slice_tensors_chunk_non_multiple_size(){

        let tensor = Tensor::from_slice(&[
            0.0, 0.1, 0.2, // Channel 0
            1.0, 1.1, 1.2, // Channel 1
        ]).view([2, 3]).totype(Float);

        let chunk = ChunkOfSlicedTensors::slice_tensors_chunk(
            &FindMotifsRunConfig::new(None, 1, 2, None, None).to_applicable(DummyFindMotifsRunConfigCtx),
            &vec![tensor.into()], vec![0],
            Cpu
        );

        let expected = Tensor::stack(&[
            Tensor::from_slice(&[ //channel 0
                0.0, 0.1,
                0.1, 0.2,
            ]).view((2,2)),
            Tensor::from_slice(&[ //channel 1
                1.0, 1.1,
                1.1, 1.2,

            ]).view((2,2)),
        ], 0).totype(Float);

        assert_eq!(*chunk.t, expected);
    }

    #[test]
    fn test_iters(){
        let tensor = Tensor::stack(&[
            Tensor::from_slice((0..100).map(|i| 0. + i as f32/100.).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((0..100).map(|i| 1. + i as f32/100.).collect::<Vec<_>>().as_slice()),
        ], 0).view([2, 100]).totype(Float);


        let chunk = ChunkOfSlicedTensors::slice_tensors_chunk(
            &FindMotifsRunConfig::new(None, 10, 20, None, None).to_applicable(DummyFindMotifsRunConfigCtx),
            &vec![tensor.into()], vec![0],
            Cpu
        );

        let slices = chunk.to_vec().into_iter().collect::<Vec<_>>();

        assert_eq!(
            vec![TensorIdSliceIdx(0, 0), TensorIdSliceIdx(0, 10), TensorIdSliceIdx(0, 20), TensorIdSliceIdx(0, 30), TensorIdSliceIdx(0, 40), TensorIdSliceIdx(0, 50), TensorIdSliceIdx(0, 60), TensorIdSliceIdx(0, 70), TensorIdSliceIdx(0, 80)],
            slices
        );
    }


    #[test]
    fn test_split_single_tensor(){

        let tensor = Tensor::stack(&[
            Tensor::from_slice((0..100).map(|i| 0. + i as f32/100.).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((0..100).map(|i| 1. + i as f32/100.).collect::<Vec<_>>().as_slice()),
        ], 0).view([2, 100]).totype(Float);

        debug!("tensor={tensor}");

        let chunk = ChunkOfSlicedTensors::slice_tensors_chunk(
            &FindMotifsRunConfig::new(None, 10, 20, None, None).to_applicable(DummyFindMotifsRunConfigCtx),
            &vec![tensor.into()], vec![0],
            Cpu
        );

        let unsplit_slices = chunk.t.shallow_clone();

        debug!("chunk.t={}", chunk.t);

        let test_fn = |chunk:&ChunkOfSlicedTensors|->Result<Tensor, ThoosaError> {
            debug!("chunk.t={:?}", chunk.t);
            if chunk.t.size()[1] > 2 {
                Err(ThoosaError::oom())
            } else {

                Ok(chunk.t.shallow_clone())
            }
        };

        let mut iters = vec![];
        let mut split_chunks = vec![];

        let mut split_queue = ChunkOfSlicedTensors::start_split_queue(chunk);
        while let Some(chunk) = split_queue.next() {

            match test_fn(&chunk) {
                Ok(t) => {
                    split_chunks.push(t);
                    iters.push(chunk.to_vec().into_iter());
                }
                Err(_) => {
                    split_queue.split(chunk).expect("split");
                }
            }
            debug!("split_queue={split_queue:?}");
        }

        assert_eq!(iters.len(), 5);
        // for iter in iters {
        //     println!("iter={:?}", iter.collect::<Vec<_>>());
        // }

        let slices = iters.into_iter().flat_map(|i|i).collect::<Vec<_>>();

        let glued_chunks = Tensor::cat(split_chunks.as_slice(), 1);
        debug!("split_chunks={split_chunks:?}");
        debug!("glued_chunks={glued_chunks}");

        assert_eq!(
            vec![TensorIdSliceIdx(0, 0), TensorIdSliceIdx(0, 10), TensorIdSliceIdx(0, 20), TensorIdSliceIdx(0, 30), TensorIdSliceIdx(0, 40), TensorIdSliceIdx(0, 50), TensorIdSliceIdx(0, 60), TensorIdSliceIdx(0, 70), TensorIdSliceIdx(0, 80)],
            slices
        );

        assert_eq!(unsplit_slices, glued_chunks);

    }

    #[test]
    fn test_split_two_tensors(){

        let tensor_aaa = Tensor::stack(&[
            Tensor::from_slice((0..200).skip(1).step_by(2).map(|i| 0. + i as f32/100.).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((0..200).skip(1).step_by(2).map(|i| 1. + i as f32/100.).collect::<Vec<_>>().as_slice()),
        ], 0).view([2, 100]).totype(Float);

        let tensor_bbb = Tensor::stack(&[
            Tensor::from_slice((0..200).step_by(2).map(|i| 0. + i as f32/100.).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((0..200).step_by(2).map(|i| 1. + i as f32/100.).collect::<Vec<_>>().as_slice()),
        ], 0).view([2, 100]).totype(Float);

        debug!("tensor_aaa={tensor_aaa}");
        debug!("tensor_bbb={tensor_bbb}");

        let chunk = ChunkOfSlicedTensors::slice_tensors_chunk(
            &FindMotifsRunConfig::new(None, 10, 20, None, None).to_applicable(DummyFindMotifsRunConfigCtx),
            &vec![tensor_aaa.into(), tensor_bbb.into()], vec![0, 1],
            Cpu
        );

        let unsplit_slices = chunk.t.shallow_clone();

        debug!("chunk.t={}", chunk.t);

        let test_fn = |chunk:&ChunkOfSlicedTensors|->Result<Tensor, ThoosaError> {
            debug!("chunk.t={:?}", chunk.t);
            if chunk.t.size()[1] > 2 {
                Err(ThoosaError::oom())
            } else {

                Ok(chunk.t.shallow_clone())
            }
        };

        let mut iters = vec![];
        let mut split_chunks = vec![];

        let mut split_queue = ChunkOfSlicedTensors::start_split_queue(chunk);
        while let Some(chunk) = split_queue.next() {

            match test_fn(&chunk) {
                Ok(t) => {
                    split_chunks.push(t);
                    iters.push(chunk.to_vec().into_iter());
                }
                Err(_) => {
                    split_queue.split(chunk).expect("split");
                }
            }
            debug!("split_queue={split_queue:?}");
        }

        assert_eq!(iters.len(), 10);
        // for iter in iters {
        //     println!("iter={:?}", iter.collect::<Vec<_>>());
        // }

        let slices = iters.into_iter().flat_map(|i|i).collect::<Vec<_>>();

        let glued_chunks = Tensor::cat(split_chunks.as_slice(), 1);
        debug!("split_chunks={split_chunks:?}");
        debug!("glued_chunks={glued_chunks}");

        assert_eq!(
            vec![
                TensorIdSliceIdx(0, 0), TensorIdSliceIdx(0, 10), TensorIdSliceIdx(0, 20), TensorIdSliceIdx(0, 30), TensorIdSliceIdx(0, 40), TensorIdSliceIdx(0, 50), TensorIdSliceIdx(0, 60), TensorIdSliceIdx(0, 70), TensorIdSliceIdx(0, 80),
                TensorIdSliceIdx(1, 0), TensorIdSliceIdx(1, 10), TensorIdSliceIdx(1, 20), TensorIdSliceIdx(1, 30), TensorIdSliceIdx(1, 40), TensorIdSliceIdx(1, 50), TensorIdSliceIdx(1, 60), TensorIdSliceIdx(1, 70), TensorIdSliceIdx(1, 80)
            ],
            slices
        );

        assert_eq!(unsplit_slices, glued_chunks);

    }


}