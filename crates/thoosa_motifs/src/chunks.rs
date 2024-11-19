use std::collections::VecDeque;
use tch::{Device, IndexOp, Kind, Tensor};
use std::fmt::Debug;
use std::marker::PhantomData;
use log::{debug, warn};
use std::mem;
use std::ops::Deref;
use std::sync::Arc;
use thoosa_common::loader::SyncWrapper;
use thoosa_shared::resource::ResourceId;
use thoosa_shared::TensorSliceIdx;

#[derive(Debug)]
pub struct ChunkOfSlicedTensors <ResourceId> where {
    pub(crate)  keys:Vec <ResourceId>,
    pub(crate)  sizes: Vec<i64>,//num splits in a ResourceId
    pub(crate)  t: SyncWrapper<Tensor>,
    step:u32,
    split_at:Option<i64>
}

pub struct ChunkOfSlicedTensorsIter<ResourceId:Clone+Debug+Eq> {
    iter:Box<dyn Iterator<Item=TensorSliceIdx<ResourceId>> + Send>
}


impl <ResourceId:Clone+Debug+Eq+Sync+Send+'static> ChunkOfSlicedTensors <ResourceId> {
    pub fn new(keys:  Vec <ResourceId>, sizes: Vec<i64>, step:u32, t: Tensor) -> Self {
        Self {
            keys,
            sizes,//number of slices produced from each tensor
            step,
            t:SyncWrapper::new_t(t),
            split_at: None,
        }
    }

    // //TODO clone
    pub fn to_vec(&self) -> Vec<TensorSliceIdx<ResourceId>>{
        let step = self.step as i64;
        let split_at_idx = self.split_at.map(|split_at_idx| split_at_idx * step).unwrap_or(0);

        self.sizes.clone().into_iter()
            .zip(self.keys.clone().into_iter())
            .flat_map(
                |(size, id)|(0..size).map(move |i|(id.clone(), i))
            )
            .map(move |v|TensorSliceIdx(v.0, split_at_idx + step  * v.1))
            .collect()


    }

    pub fn keys_sizes(&self)->Vec<(ResourceId, i64)>{
        self.keys.iter().zip(self.sizes.iter()).map(|v|(v.0.clone(), v.1.clone())).collect()
    }

    pub fn split(&self) -> [Self;2] {
        //only one tensor in the chunk, splitting tensor
        if self.keys.len() == 1 {
            assert_eq!(self.sizes.len(), 1);

            let split_at_idx = self.sizes[0]/2;
            debug!("ChunkOfSlicedTensors::split: 0 self.sizes={:?} self.keys={:?} self.t.size()={:?} split_at_idx={split_at_idx}", self.sizes, self.keys, self.t.size());
            [
                Self {
                    keys: self.keys.clone(),
                    sizes: vec![split_at_idx],
                    step: self.step,
                    t: SyncWrapper::new_t(self.t.i((.., ..split_at_idx, ..))),
                    split_at: self.split_at,
                },
                Self {
                    keys: self.keys.clone(),
                    sizes: vec![self.sizes[0]-split_at_idx],
                    step: self.step,
                    t: SyncWrapper::new_t(self.t.i((.., split_at_idx.., ..))),
                    split_at: self.split_at.map(|v|Some(v+split_at_idx)).unwrap_or(Some(split_at_idx)),
                }
            ]
        //more than one tensor in the chunk, splitting chunk
        } else {
            let split_at = self.keys.len() / 2;

            assert!(split_at > 0);

            let split_at_idx = self.sizes.iter().take(split_at).sum::<i64>();

            debug!("ChunkOfSlicedTensors::split: 1 self.sizes={:?} self.keys={:?} self.t.size()={:?} split_at={split_at} split_at_idx={split_at_idx}", self.sizes, self.keys, self.t.size());

            [
                Self {
                    keys: self.keys[..split_at].to_vec(),
                    sizes: self.sizes[..split_at].to_vec(),
                    step: self.step,
                    t: SyncWrapper::new_t(self.t.i((.., ..split_at_idx, ..))),

                    split_at: self.split_at,
                },
                Self {
                    keys: self.keys[split_at..].to_vec(),
                    sizes: self.sizes[split_at..].to_vec(),
                    step: self.step,
                    t: SyncWrapper::new_t(self.t.i((.., split_at_idx.., ..))),
                    split_at: self.split_at.map(|v|Some(v+split_at_idx)).unwrap_or(Some(split_at_idx)),
                }
            ]
        }
    }

    //TODO Option instead  of [0.0]
    pub fn drop_tensor(&mut self) {
        mem::swap(&mut self.t, &mut SyncWrapper::new_t(Tensor::from_slice(&[0.0])));
    }

    /// returns:  channels x t.size = num slices = 1+(len-window)/step x window
    pub fn slice_tensors_chunk(step: u32, window: u32, tensors_chunk: &[SyncWrapper<Box<Tensor>>], keys: Vec <ResourceId>, device: Device) -> ChunkOfSlicedTensors <ResourceId> {

        debug!("slice_tensors_chunk: step={step} window={window} tensors_chunk.len()={}", tensors_chunk.len());

        let mut slices = Vec::with_capacity(tensors_chunk.len());

        for tensor in tensors_chunk {
            let input_length = tensor.size()[1];

            if input_length >= window as i64 {

                let unfolded = tensor.unfold(1, window as i64, step as i64);
                // channels x num_slices x window, transpose  to num_slices x channels x window
                let unfolded = unfolded.transpose(0, 1);
                slices.push(unfolded);
            } else {
                warn!("skipping tensor")
            }
        }

        let tensor= if slices.is_empty() {
                                    //channels
            Tensor::zeros(&[0, tensors_chunk[0].size()[0], window as i64], (Kind::Float, device))
        } else {
            //slices x channels x window -> channels x slices x window
            Tensor::cat(&slices, 0).transpose(0, 1)
        };
        debug!("slice_tensors_chunk: final tensor.size={:?} device={:?}", tensor.size(), tensor.device());

        ChunkOfSlicedTensors::new(
            keys,
            slices.iter().map(|t| t.size()[0]).collect::<Vec<_>>(), //num slices in each tensor,
            step,
            tensor,
        )
    }

    pub fn start_split_queue(chunk:ChunkOfSlicedTensors<ResourceId>) -> SplitQueue<ResourceId>{
        let queue = VecDeque::from([chunk]);

        SplitQueue {
            queue
        }
    }

}

#[derive(Debug)]
pub struct SplitQueue<R> {
    queue:VecDeque<ChunkOfSlicedTensors<R>>
}

impl <R:Clone+Debug+Eq+Sync+Send+'static> SplitQueue<R> {
    pub fn next(&mut self) -> Option<ChunkOfSlicedTensors<R>> {
        self.queue.pop_front()
    }

    pub fn split(&mut self, chunk:ChunkOfSlicedTensors<R>)  {
        debug!("SplitQueue::split: chunk={chunk:?}");
        let [chunk_0, chunk_1] = chunk.split();
        debug!("SplitQueue::split: chunk_0={chunk_0:?}");
        debug!("SplitQueue::split: chunk_1={chunk_1:?}");
        debug!("-----------------------");
        self.queue.push_front(chunk_1);
        self.queue.push_front(chunk_0);

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
    use thoosa_shared::{init_test_logging, TensorSliceIdx, ThoosaError};
    use thoosa_shared::resource::ResourceId;
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
            1, 2,
            &vec![tensor_a.into(), tensor_b.into()], vec!["aaa", "bbb"],
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
            1, 2,
            &vec![tensor.into()], vec!["aaa"],
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
            10, 20,
            &vec![tensor.into()], vec!["aaa"],
            Cpu
        );

        let slices = chunk.to_vec().into_iter().collect::<Vec<_>>();

        assert_eq!(
            vec![TensorSliceIdx("aaa", 0), TensorSliceIdx("aaa", 10), TensorSliceIdx("aaa", 20), TensorSliceIdx("aaa", 30), TensorSliceIdx("aaa", 40), TensorSliceIdx("aaa", 50), TensorSliceIdx("aaa", 60), TensorSliceIdx("aaa", 70), TensorSliceIdx("aaa", 80)],
            slices
        );
    }


    #[test]
    fn test_split(){

        let tensor = Tensor::stack(&[
            Tensor::from_slice((0..100).map(|i| 0. + i as f32/100.).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((0..100).map(|i| 1. + i as f32/100.).collect::<Vec<_>>().as_slice()),
        ], 0).view([2, 100]).totype(Float);

        let chunk = ChunkOfSlicedTensors::slice_tensors_chunk(
            10, 20,
            &vec![tensor.into()], vec!["aaa"],
            Cpu
        );

        let test_fn = |chunk:&ChunkOfSlicedTensors<&str>|->Result<(), ThoosaError> {
            debug!("chunk.t={:?}", chunk.t);
            if chunk.t.size()[1] > 2 {
                Err(ThoosaError::oom())
            } else {

                Ok(())
            }
        };

        let mut iters = vec![];

        let mut split_queue = ChunkOfSlicedTensors::start_split_queue(chunk);
        while let Some(chunk) = split_queue.next() {

            match test_fn(&chunk) {
                Ok(_) => {
                    iters.push(chunk.to_vec().into_iter());
                }
                Err(_) => {
                    split_queue.split(chunk);
                }
            }
            debug!("split_queue={split_queue:?}");
        }

        assert_eq!(iters.len(), 5);
        // for iter in iters {
        //     println!("iter={:?}", iter.collect::<Vec<_>>());
        // }

        let slices = iters.into_iter().flat_map(|i|i).collect::<Vec<_>>();

        // println!("slices={slices:?}");

        assert_eq!(
            vec![TensorSliceIdx("aaa", 0), TensorSliceIdx("aaa", 10), TensorSliceIdx("aaa", 20), TensorSliceIdx("aaa", 30), TensorSliceIdx("aaa", 40), TensorSliceIdx("aaa", 50), TensorSliceIdx("aaa", 60), TensorSliceIdx("aaa", 70), TensorSliceIdx("aaa", 80)],
            slices
        );

    }

}