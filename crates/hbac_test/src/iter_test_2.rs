use std::marker::PhantomData;
use std::ops::Index;
use std::slice::SliceIndex;
use tch::{TchError, Tensor};
use crate::GenError;

struct TensorsSource<'a> {//TODO SliceIndex?
    tensors:Vec<Tensor>,
    tensor_index_0:usize,
    tensor_index_1:usize,

    slice_index_0:i64,
    slice_index_1:i64,

    tensor_0:Option<Tensor>,

    dim:i64,
    m:i64,
    pd:PhantomData<&'a ()>
}

impl <'a> TensorsSource<'a> {
    pub fn new(inner: Vec<Tensor>, dim:i64, m:i64) -> Self {
        Self {
            tensors: inner,
            tensor_index_0: 0,
            tensor_index_1: 0,

            slice_index_0:0,
            slice_index_1:0,
            tensor_0: None,

            dim,
            m,
            pd: PhantomData
        }
    }

    fn next(&'a mut self) -> Option<Result<(&'a Tensor, &'a Tensor), GenError>> {
        if self.tensor_index_0 < self.tensors.len() {
            // return Some(Ok((
            //     // &self.tensors[self.tensor_index_0].f_slice(self.dim, Some(self.slice_index_0), Some(self.slice_index_0 + self.m), 1).unwrap(),
            //     // &self.tensors[self.tensor_index_1].f_slice(self.dim, Some(self.slice_index_1), Some(self.slice_index_1 + self.m), 1).unwrap()
            // )))
            return Some(Ok((self.tensor_0.as_ref().unwrap(), self.tensor_0.as_ref().unwrap())))

        }

        None
    }
}

// impl <'a> Iterator for TensorsSource<'a> {
//     type Item = Result<(&'a Tensor, &'a Tensor), GenError>;
//
//     fn next(&'a mut self) -> Option<Self::Item> {
//         if self.tensor_index_0 < self.tensors.len() {
//             // return Some(Ok((
//             //     // &self.tensors[self.tensor_index_0].f_slice(self.dim, Some(self.slice_index_0), Some(self.slice_index_0 + self.m), 1).unwrap(),
//             //     // &self.tensors[self.tensor_index_1].f_slice(self.dim, Some(self.slice_index_1), Some(self.slice_index_1 + self.m), 1).unwrap()
//             // )))
//             return Some(Ok((self.tensor_0.as_ref().unwrap(), self.tensor_0.as_ref().unwrap())))
//
//         }
//
//         None
//     }
// }


#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};
    use crate::iter_test_2::TensorsSource;

    #[test]
    fn test_one_inner(){

        // let t_0 = Tensor::randn(&[2, 100], (Kind::Float, Device::Cpu));
        let t_0= Tensor::vstack(&[
            Tensor::from_slice((0..10).collect::<Vec<i64>>().as_slice()),
            Tensor::from_slice((0..10).collect::<Vec<i64>>().as_slice())
        ]);

        let t_1= Tensor::vstack(&[
            Tensor::from_slice((0..100).step_by(10).collect::<Vec<i64>>().as_slice()),
            Tensor::from_slice((0..100).step_by(10).collect::<Vec<i64>>().as_slice())
        ]);


        let ts = TensorsSource::new(vec![t_0, t_1], 1, 10);
    }
}
