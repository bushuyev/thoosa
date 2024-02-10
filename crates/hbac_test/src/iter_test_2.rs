use std::marker::PhantomData;
use std::ops::Index;
use std::slice::SliceIndex;
use tch::{TchError, Tensor};
use crate::GenError;

struct TensorsSource {//TODO SliceIndex?
    tensors:Vec<Tensor>,
    tensor_index_0:usize,
    tensor_index_1:usize,

    slice_index_0:i64,
    slice_index_1:i64,

    dim:usize,
    step:i64,
    m:i64,
}

impl TensorsSource {
    pub fn new(tensors: Vec<Tensor>, dim:usize, m:i64, step:i64) -> Self {

        assert!(step <= m);

        Self {
            tensors,
            tensor_index_0: 0,
            tensor_index_1: 0,

            slice_index_0:0,
            slice_index_1:0,

            dim,
            step,
            m
        }
    }

}

impl Iterator for TensorsSource {
    type Item = Result<(Tensor, Tensor), GenError>;

    fn next(&mut self) -> Option<Self::Item> {
        println!("self.tensor_index_0={:?}, self.slice_index_0={:?}, self.tensor_index_1={:?}, self.slice_index_1={:?}", self.tensor_index_0, self.slice_index_0, self.tensor_index_1, self.slice_index_1);


        if self.tensor_index_0 < self.tensors.len() {

            let v = Some(Ok((
                self.tensors[self.tensor_index_0].f_slice(self.dim as i64, Some(self.slice_index_0), Some(self.slice_index_0 + self.m), 1).unwrap(),
                self.tensors[self.tensor_index_1].f_slice(self.dim as i64, Some(self.slice_index_1), Some(self.slice_index_1 + self.m), 1).unwrap()
            )));


            //TODO m < tensor
            if self.slice_index_1 + self.m < self.tensors[self.tensor_index_1].size()[self.dim] {
                self.slice_index_1 += self.step;

                println!("1");

            } else {

                if self.tensor_index_1 < self.tensors.len() -1 {
                    self.tensor_index_1 += 1;
                    self.slice_index_1 = self.slice_index_0;

                    println!("2.1");

                } else {

                    if self.slice_index_0 + self.m < self.tensors[self.tensor_index_0].size()[self.dim] {
                        self.slice_index_0 += self.step;

                        println!("2.2.1");
                    } else {
                        println!("2.2.2");

                        self.slice_index_0 = 0;
                        self.tensor_index_0 += 1;
                    }

                    self.tensor_index_1 = self.tensor_index_0;
                    self.slice_index_1 = self.slice_index_0;
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
    use crate::GenError;
    use crate::iter_test_2::TensorsSource;

    #[test]
    fn test_one_inner(){

        let t_0= Tensor::vstack(&[
            Tensor::from_slice((0..10).collect::<Vec<i64>>().as_slice()),
            Tensor::from_slice((0..10).collect::<Vec<i64>>().as_slice())
        ]);


        let ts = TensorsSource::new(vec![t_0], 1, 2, 2);
        let r = ts.collect::<Vec<Result<(Tensor, Tensor), GenError>>>();
        // x x x x x
        //   x x x x
        //     x x x
        //       x x
        //         x
        assert_eq!(r.len(), 15);
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


        let ts = TensorsSource::new(vec![t_0, t_1], 1, 2, 2);
        let r = ts.collect::<Vec<Result<(Tensor, Tensor), GenError>>>();
        // x x x x x x x x x x
        //   x x x x x x x x x
        //     x x x x x x x x
        //       x x x x x x x
        //         x x x x x x
        //           x x x x x
        //             x x x x
        //               x x x
        //                 x x
        //                   x
        //

        println!("r.len()={:?}", r.len());
        // assert_eq!(r.len(), 15);
    }
}
