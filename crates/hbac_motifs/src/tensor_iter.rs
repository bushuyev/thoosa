

use std::sync::atomic::{AtomicUsize, Ordering};

use log::{debug};
use tch::Tensor;

use hbac_shared::HbacError;
use hbac_shared::TensorSliceIdx;


pub trait TensorsStore {
    fn get_tensor(&mut self, index: usize) -> Result<&Box<Tensor>, HbacError>;

    fn get_tensors(&mut self, indexes: &[usize]) -> Result<Vec<Tensor>, HbacError> {
        indexes.into_iter().map(|i|*i).map( |i| self.get_tensor(i).map(|t|t.shallow_clone())).collect()
    }

    fn tensors_len(&self) -> usize;
}


pub struct TensorSliceIter {//TODO SliceIndex?
    tensors_store: Box<dyn TensorsStore + Send>,
    tensor_index_0:usize,
    tensor_index_1:usize,

    slice_index_0:i64,
    slice_index_1:i64,

    dim:usize,
    step:i64,
    m:i64,
    iteration_n:AtomicUsize,
}

impl TensorSliceIter {

    pub fn new(tensors_store: Box<impl TensorsStore + Send +'static>, dim:usize, m:usize, step:usize) -> Self {

        assert!(step <= m);

        debug!("tensors_store.tensors_len()={}", tensors_store.tensors_len());

        Self {
            tensors_store,
            tensor_index_0: 0,
            tensor_index_1: 0,

            slice_index_0:0,
            slice_index_1:step as i64,

            dim,
            step: step as i64,
            m: m as i64,
            iteration_n:  AtomicUsize::new(0)
        }
    }
}
pub type TensorSlice = (TensorSliceIdx<String>, Tensor); //tensor idx, slice, tensor

impl Iterator for TensorSliceIter {
    type Item = Result<(TensorSlice, TensorSlice, usize), HbacError>;

    fn next(&mut self) -> Option<Self::Item> {
        debug!("self.tensor_index_0={:?}, self.slice_index_0={:?}, self.tensor_index_1={:?}, self.slice_index_1={:?}", self.tensor_index_0, self.slice_index_0, self.tensor_index_1, self.slice_index_1);
        let tensors_len = self.tensors_store.tensors_len();

        if self.tensor_index_0 < self.tensors_store.tensors_len() {
            // let t_0 = self.tensors_store.get_tensor(self.tensor_index_0).unwrap();
            // let t_1 = self.tensors_store.get_tensor(self.tensor_index_1).unwrap();

            let mut tensors = self.tensors_store.get_tensors(&[self.tensor_index_0, self.tensor_index_1]).unwrap();
            let t_1 = tensors.pop().unwrap();
            let t_0 = tensors.pop().unwrap();

            let v = Some(Ok((
                //TODO self.tensor_index_0.to_string() -> store.index_to_key
                (TensorSliceIdx(self.tensor_index_0.to_string(), self.slice_index_0), t_0.f_slice(self.dim as i64, Some(self.slice_index_0), Some(self.slice_index_0 + self.m), 1).unwrap()),//TODO unwrap
                (TensorSliceIdx(self.tensor_index_1.to_string(), self.slice_index_1), t_1.f_slice(self.dim as i64, Some(self.slice_index_1), Some(self.slice_index_1 + self.m), 1).unwrap()),
                self.iteration_n.fetch_add(1, Ordering::Relaxed)
            )));


            //TODO m < tensor
            if self.slice_index_1 + self.step + self.m <= t_1.size()[self.dim] {
                self.slice_index_1 += self.step;


            } else {

                if self.tensor_index_1 < tensors_len -1 {
                    self.tensor_index_1 += 1;
                    self.slice_index_1 = 0;

                } else {
                    if self.slice_index_0 + self.step + self.m <= t_0.size()[self.dim] {
                        self.slice_index_0 += self.step;

                    } else {
                        self.slice_index_0 = 0;
                        self.tensor_index_0 += 1;
                    }

                    self.slice_index_1 = self.slice_index_0 + self.m;
                    self.tensor_index_1 = self.tensor_index_0;

                    if self.slice_index_1 + self.m > self.tensors_store.get_tensor(self.tensor_index_1).unwrap().size()[self.dim] {//to avoid "i=10 0: [8.0, 9.0] 1: []" in test_one_inner
                        self.slice_index_0 = 0;
                        self.tensor_index_0 += 1;

                        self.slice_index_1 = self.slice_index_0 + self.m;
                        self.tensor_index_1 = self.tensor_index_0;
                    }

                }
            }
            return v;

        }

        None
    }
}

#[cfg(test)]
mod tests {

    use tch::{Device, Kind, Tensor};

    use hbac_common::utils::*;
    use hbac_shared::HbacError;
    use crate::tensor_iter::{TensorSliceIter, TensorsStore};


    struct VecTensorsStore {
        tensors:Vec<Box<Tensor>>
    }

    impl VecTensorsStore {
        pub fn new(tensors: Vec<Tensor>) -> Box<Self> {
            Box::new(Self { tensors:tensors.into_iter().map(|t|Box::new(t)).collect() })
        }
    }
    impl TensorsStore for VecTensorsStore {
        fn get_tensor(&mut self, index: usize) -> Result<&Box<Tensor>, HbacError> {
            Ok(&self.tensors[index])
        }

        fn tensors_len(&self) -> usize {
            self.tensors.len()
        }
    }

    #[test]
    fn test_one_inner(){

        let t_0= Tensor::vstack(&[
            Tensor::from_slice((0..10).collect::<Vec<i64>>().as_slice()),
            Tensor::from_slice((0..10).collect::<Vec<i64>>().as_slice())
        ]);


        let ts = TensorSliceIter::new(VecTensorsStore::new(vec![t_0]), 1, 2, 2);
        let r = ts_to_vec(ts);
        // x x x x x
        //   x x x x
        //     x x x
        //       x x
        //         x

        assert_eq!(
            vec![
                "i=0 0: [0.0, 1.0] 1: [2.0, 3.0]",
                "i=1 0: [0.0, 1.0] 1: [4.0, 5.0]",
                "i=2 0: [0.0, 1.0] 1: [6.0, 7.0]",
                "i=3 0: [0.0, 1.0] 1: [8.0, 9.0]",
                "i=4 0: [2.0, 3.0] 1: [4.0, 5.0]",
                "i=5 0: [2.0, 3.0] 1: [6.0, 7.0]",
                "i=6 0: [2.0, 3.0] 1: [8.0, 9.0]",
                "i=7 0: [4.0, 5.0] 1: [6.0, 7.0]",
                "i=8 0: [4.0, 5.0] 1: [8.0, 9.0]",
                "i=9 0: [6.0, 7.0] 1: [8.0, 9.0]",
            ],
            r
        );

    }

    #[test]
    fn test_one_inner_m_3(){

        let t_0= Tensor::vstack(&[
            Tensor::from_slice((0..10).collect::<Vec<i64>>().as_slice()),
            Tensor::from_slice((0..10).collect::<Vec<i64>>().as_slice())
        ]);


        let ts = TensorSliceIter::new(VecTensorsStore::new(vec![t_0]), 1, 3, 2);
        let r = ts_to_vec(ts);

        assert_eq!(
            vec![
                "i=0 0: [0.0, 1.0, 2.0] 1: [2.0, 3.0, 4.0]",
                "i=1 0: [0.0, 1.0, 2.0] 1: [4.0, 5.0, 6.0]",
                "i=2 0: [0.0, 1.0, 2.0] 1: [6.0, 7.0, 8.0]",
                "i=3 0: [2.0, 3.0, 4.0] 1: [5.0, 6.0, 7.0]", //+m instead of +step
                "i=4 0: [2.0, 3.0, 4.0] 1: [7.0, 8.0, 9.0]",
                "i=5 0: [4.0, 5.0, 6.0] 1: [7.0, 8.0, 9.0]",
            ],
            r
        );

    }

    #[test]
    fn test_one_inner_m_3_unfit(){

        let t_0= Tensor::vstack(&[
            Tensor::from_slice((0..11).collect::<Vec<i64>>().as_slice()),
            Tensor::from_slice((0..11).collect::<Vec<i64>>().as_slice())
        ]);


        let ts = TensorSliceIter::new(VecTensorsStore::new(vec![t_0]), 1, 3, 2);
        let r = ts_to_vec(ts);


        assert_eq!(
            vec![
                "i=0 0: [0.0, 1.0, 2.0] 1: [2.0, 3.0, 4.0]",
                "i=1 0: [0.0, 1.0, 2.0] 1: [4.0, 5.0, 6.0]",
                "i=2 0: [0.0, 1.0, 2.0] 1: [6.0, 7.0, 8.0]",
                "i=3 0: [0.0, 1.0, 2.0] 1: [8.0, 9.0, 10.0]",
                "i=4 0: [2.0, 3.0, 4.0] 1: [5.0, 6.0, 7.0]",
                "i=5 0: [2.0, 3.0, 4.0] 1: [7.0, 8.0, 9.0]",
                "i=6 0: [4.0, 5.0, 6.0] 1: [7.0, 8.0, 9.0]",
            ],
            r
        );

    }

    fn ts_to_vec(ts: TensorSliceIter) -> Vec<String> {
        ts.enumerate().map(|(i, v)| {
            match v {
                Ok(ref x) => {
                    format!("i={} 0: {:?} 1: {:?}", i, from::<Vec<Vec<f32>>>(&x.0.1)[0], from::<Vec<Vec<f32>>>(&x.1.1)[0])
                }
                Err(e) => {
                    format!("e={:?}", e)
                }
            }
        }).collect::<Vec<String>>()
    }

    #[test]
    fn test_one_inner_unfit() {
        let t_0 = Tensor::vstack(&[
            Tensor::from_slice((0..11).collect::<Vec<i64>>().as_slice()),
            Tensor::from_slice((0..11).collect::<Vec<i64>>().as_slice())
        ]);


        let ts = TensorSliceIter::new(VecTensorsStore::new(vec![t_0]), 1, 2, 2);
        let r = ts_to_vec(ts);

        assert_eq!(
            vec![
                "i=0 0: [0.0, 1.0] 1: [2.0, 3.0]",
                "i=1 0: [0.0, 1.0] 1: [4.0, 5.0]",
                "i=2 0: [0.0, 1.0] 1: [6.0, 7.0]",
                "i=3 0: [0.0, 1.0] 1: [8.0, 9.0]",
                "i=4 0: [2.0, 3.0] 1: [4.0, 5.0]",
                "i=5 0: [2.0, 3.0] 1: [6.0, 7.0]",
                "i=6 0: [2.0, 3.0] 1: [8.0, 9.0]",
                "i=7 0: [4.0, 5.0] 1: [6.0, 7.0]",
                "i=8 0: [4.0, 5.0] 1: [8.0, 9.0]",
                "i=9 0: [6.0, 7.0] 1: [8.0, 9.0]",
            ],
            r
        )
    }

    #[test]
    fn test_two_inner(){

        let t_0= Tensor::vstack(&[
            Tensor::from_slice((0..10).collect::<Vec<i64>>().as_slice()),
            Tensor::from_slice((0..10).collect::<Vec<i64>>().as_slice())
        ]);

        let t_1= Tensor::vstack(&[
            Tensor::from_slice((0..100).step_by(10).collect::<Vec<i64>>().as_slice()),
            Tensor::from_slice((0..100).step_by(10).collect::<Vec<i64>>().as_slice())
        ]);

        let ts = TensorSliceIter::new(VecTensorsStore::new(vec![t_0, t_1]), 1, 2, 2);
        let r = ts_to_vec(ts);

        assert_eq!(
            vec![
                "i=0 0: [0.0, 1.0] 1: [2.0, 3.0]",
                "i=1 0: [0.0, 1.0] 1: [4.0, 5.0]",
                "i=2 0: [0.0, 1.0] 1: [6.0, 7.0]",
                "i=3 0: [0.0, 1.0] 1: [8.0, 9.0]",
                "i=4 0: [0.0, 1.0] 1: [0.0, 10.0]",
                "i=5 0: [0.0, 1.0] 1: [20.0, 30.0]",
                "i=6 0: [0.0, 1.0] 1: [40.0, 50.0]",
                "i=7 0: [0.0, 1.0] 1: [60.0, 70.0]",
                "i=8 0: [0.0, 1.0] 1: [80.0, 90.0]",
                "i=9 0: [2.0, 3.0] 1: [4.0, 5.0]",
                "i=10 0: [2.0, 3.0] 1: [6.0, 7.0]",
                "i=11 0: [2.0, 3.0] 1: [8.0, 9.0]",
                "i=12 0: [2.0, 3.0] 1: [0.0, 10.0]",
                "i=13 0: [2.0, 3.0] 1: [20.0, 30.0]",
                "i=14 0: [2.0, 3.0] 1: [40.0, 50.0]",
                "i=15 0: [2.0, 3.0] 1: [60.0, 70.0]",
                "i=16 0: [2.0, 3.0] 1: [80.0, 90.0]",
                "i=17 0: [4.0, 5.0] 1: [6.0, 7.0]",
                "i=18 0: [4.0, 5.0] 1: [8.0, 9.0]",
                "i=19 0: [4.0, 5.0] 1: [0.0, 10.0]",
                "i=20 0: [4.0, 5.0] 1: [20.0, 30.0]",
                "i=21 0: [4.0, 5.0] 1: [40.0, 50.0]",
                "i=22 0: [4.0, 5.0] 1: [60.0, 70.0]",
                "i=23 0: [4.0, 5.0] 1: [80.0, 90.0]",
                "i=24 0: [6.0, 7.0] 1: [8.0, 9.0]",
                "i=25 0: [6.0, 7.0] 1: [0.0, 10.0]",
                "i=26 0: [6.0, 7.0] 1: [20.0, 30.0]",
                "i=27 0: [6.0, 7.0] 1: [40.0, 50.0]",
                "i=28 0: [6.0, 7.0] 1: [60.0, 70.0]",
                "i=29 0: [6.0, 7.0] 1: [80.0, 90.0]",
                "i=30 0: [0.0, 10.0] 1: [20.0, 30.0]",
                "i=31 0: [0.0, 10.0] 1: [40.0, 50.0]",
                "i=32 0: [0.0, 10.0] 1: [60.0, 70.0]",
                "i=33 0: [0.0, 10.0] 1: [80.0, 90.0]",
                "i=34 0: [20.0, 30.0] 1: [40.0, 50.0]",
                "i=35 0: [20.0, 30.0] 1: [60.0, 70.0]",
                "i=36 0: [20.0, 30.0] 1: [80.0, 90.0]",
                "i=37 0: [40.0, 50.0] 1: [60.0, 70.0]",
                "i=38 0: [40.0, 50.0] 1: [80.0, 90.0]",
                "i=39 0: [60.0, 70.0] 1: [80.0, 90.0]",
            ],
            r
        );
    }

    #[allow(dead_code)]
    fn load_rand_tensor(_key:&String) ->Box<Tensor>{
        Box::new(Tensor::rand(&[2, 10], (Kind::Float, Device::Cpu)))
    }
}
