use std::marker::PhantomData;
use std::ops::Index;
use std::slice::SliceIndex;
use log::debug;
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
            slice_index_1:step,

            dim,
            step,
            m,
        }
    }

}

impl Iterator for TensorsSource {
    type Item = Result<(Tensor, Tensor), GenError>;

    fn next(&mut self) -> Option<Self::Item> {
        debug!("self.tensor_index_0={:?}, self.slice_index_0={:?}, self.tensor_index_1={:?}, self.slice_index_1={:?}", self.tensor_index_0, self.slice_index_0, self.tensor_index_1, self.slice_index_1);


        if self.tensor_index_0 < self.tensors.len() {

            let v = Some(Ok((
                self.tensors[self.tensor_index_0].f_slice(self.dim as i64, Some(self.slice_index_0), Some(self.slice_index_0 + self.m), 1).unwrap(),
                self.tensors[self.tensor_index_1].f_slice(self.dim as i64, Some(self.slice_index_1), Some(self.slice_index_1 + self.m), 1).unwrap()
            )));


            //TODO m < tensor
            if self.slice_index_1 + self.step + self.m <= self.tensors[self.tensor_index_1].size()[self.dim] {
                self.slice_index_1 += self.step;

                debug!("1");

            } else {

                if self.tensor_index_1 < self.tensors.len() -1 {
                    self.tensor_index_1 += 1;
                    self.slice_index_1 = 0;

                    debug!("2.1");

                } else {

                    if self.slice_index_0 + self.step + self.m <= self.tensors[self.tensor_index_0].size()[self.dim] {
                        self.slice_index_0 += self.step;

                        debug!("2.2.1");
                    } else {
                        debug!("2.2.2");

                        self.slice_index_0 = 0;
                        self.tensor_index_0 += 1;
                    }

                    self.slice_index_1 = self.slice_index_0 + self.step;
                    self.tensor_index_1 = self.tensor_index_0;

                    if self.slice_index_1 + self.m > self.tensors[self.tensor_index_1].size()[self.dim] {//to avoid "i=10 0: [8.0, 9.0] 1: []" in test_one_inner
                        self.slice_index_0 = 0;
                        self.tensor_index_0 += 1;

                        self.slice_index_1 = self.slice_index_0 + self.step;
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
    use std::io::Write;
    use std::time::Instant;
    use log::{debug, LevelFilter};
    use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger};
    use tch::{Device, Kind, Tensor};
    use crate::GenError;
    use crate::iter_test_2::TensorsSource;
    use crate::tests::from;



    #[test]
    fn test_one_inner(){

        let t_0= Tensor::vstack(&[
            Tensor::from_slice((0..10).collect::<Vec<i64>>().as_slice()),
            Tensor::from_slice((0..10).collect::<Vec<i64>>().as_slice())
        ]);


        let ts = TensorsSource::new(vec![t_0], 1, 2, 2);
        let r = ts_to_vec(ts);
        // x x x x x
        //   x x x x
        //     x x x
        //       x x
        //         x
        for r in r.clone() {
            println!("r={:?}", r);
        }
        assert_eq!(vec![
            "i=0 0: [0.0, 1.0] 1: [2.0, 3.0]",
            "i=1 0: [0.0, 1.0] 1: [4.0, 5.0]",
            "i=2 0: [0.0, 1.0] 1: [6.0, 7.0]",
            "i=3 0: [0.0, 1.0] 1: [8.0, 9.0]",
            "i=4 0: [2.0, 3.0] 1: [4.0, 5.0]",
            "i=5 0: [2.0, 3.0] 1: [6.0, 7.0]",
            "i=6 0: [2.0, 3.0] 1: [8.0, 9.0]",
            "i=7 0: [4.0, 5.0] 1: [6.0, 7.0]",
            "i=8 0: [4.0, 5.0] 1: [8.0, 9.0]",
            "i=9 0: [6.0, 7.0] 1: [8.0, 9.0]"
             // "i=10 0: [8.0, 9.0] 1: []"
            ], r
        );

    }

    #[test]
    fn test_one_inner_m_3(){

        let t_0= Tensor::vstack(&[
            Tensor::from_slice((0..10).collect::<Vec<i64>>().as_slice()),
            Tensor::from_slice((0..10).collect::<Vec<i64>>().as_slice())
        ]);


        let ts = TensorsSource::new(vec![t_0], 1, 3, 2);
        let r = ts_to_vec(ts);
        //
        // for r in r.clone() {
        //     println!("r={:?}", r);
        // }
        assert_eq!(vec![
            "i=0 0: [0.0, 1.0, 2.0] 1: [2.0, 3.0, 4.0]",
            "i=1 0: [0.0, 1.0, 2.0] 1: [4.0, 5.0, 6.0]",
            "i=2 0: [0.0, 1.0, 2.0] 1: [6.0, 7.0, 8.0]",
            "i=3 0: [2.0, 3.0, 4.0] 1: [4.0, 5.0, 6.0]",
            "i=4 0: [2.0, 3.0, 4.0] 1: [6.0, 7.0, 8.0]",
            "i=5 0: [4.0, 5.0, 6.0] 1: [6.0, 7.0, 8.0]",
            // "i=10 0: [8.0, 9.0] 1: []"
        ], r
        );

    }

    #[test]
    fn test_one_inner_m_3_unfit(){

        let t_0= Tensor::vstack(&[
            Tensor::from_slice((0..11).collect::<Vec<i64>>().as_slice()),
            Tensor::from_slice((0..11).collect::<Vec<i64>>().as_slice())
        ]);


        let ts = TensorsSource::new(vec![t_0], 1, 3, 2);
        let r = ts_to_vec(ts);

        // for r in r.clone() {
        //     println!("r={:?}", r);
        // }
        assert_eq!(vec![
            "i=0 0: [0.0, 1.0, 2.0] 1: [2.0, 3.0, 4.0]",
            "i=1 0: [0.0, 1.0, 2.0] 1: [4.0, 5.0, 6.0]",
            "i=2 0: [0.0, 1.0, 2.0] 1: [6.0, 7.0, 8.0]",
            "i=3 0: [0.0, 1.0, 2.0] 1: [8.0, 9.0, 10.0]",
            "i=4 0: [2.0, 3.0, 4.0] 1: [4.0, 5.0, 6.0]",
            "i=5 0: [2.0, 3.0, 4.0] 1: [6.0, 7.0, 8.0]",
            "i=6 0: [2.0, 3.0, 4.0] 1: [8.0, 9.0, 10.0]",
            "i=7 0: [4.0, 5.0, 6.0] 1: [6.0, 7.0, 8.0]",
            "i=8 0: [4.0, 5.0, 6.0] 1: [8.0, 9.0, 10.0]",
            "i=9 0: [6.0, 7.0, 8.0] 1: [8.0, 9.0, 10.0]",
        ], r
        );

    }

    fn ts_to_vec(ts: TensorsSource) -> Vec<String> {
        let r = ts.enumerate().map(|(i, v)| {
            match v {
                Ok(ref x) => {
                    format!("i={} 0: {:?} 1: {:?}", i, from::<Vec<Vec<f32>>>(&x.0)[0], from::<Vec<Vec<f32>>>(&x.1)[0])
                    // x.0.print();
                    // x.1.print();
                }
                Err(e) => {
                    format!("e={:?}", e)
                }
            }
        }).collect::<Vec<String>>();
        r
    }

    #[test]
    fn test_one_inner_unfit(){

        let t_0= Tensor::vstack(&[
            Tensor::from_slice((0..11).collect::<Vec<i64>>().as_slice()),
            Tensor::from_slice((0..11).collect::<Vec<i64>>().as_slice())
        ]);


        let ts = TensorsSource::new(vec![t_0], 1, 2, 2);
        let r = ts_to_vec(ts);
        // for r in r.clone() {
        //     println!("r={:?}", r);
        // }
        // assert_eq!(r.len(), 15);
        assert_eq!(vec![
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

        ], r)
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
        let r = ts_to_vec(ts);
        // for r in r.clone() {
        //     println!("r={:?}", r);
        // }
        assert_eq!(vec![
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

        ], r);
        // assert_eq!(r.len(), 15);
    }

    // #[test]
    fn test_time(){

        let all_t = (0..4_000).map(|_|Tensor::rand(&[20, 100_000], (Kind::Float, Device::Cpu))).collect::<Vec<Tensor>>();
        let ts = TensorsSource::new(all_t, 1, 1_000, 100);

        let t1 = Instant::now();
        let mut s = 0;
        for (i, r) in ts.enumerate() {
            if i % 1000 == 0 {
                print!(".");
                std::io::stdout().flush().unwrap();
            }
            match r {
                Ok(_) => {
                    s+=1;
                }
                Err(_) => {}
            }
        }
        debug!("done in {} s={}", t1.elapsed().as_millis(), s);
    }
}
