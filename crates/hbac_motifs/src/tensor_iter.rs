use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Index;
use std::slice::SliceIndex;
use std::sync::atomic::{AtomicI64, AtomicU8, Ordering};
use std::thread;
use std::time::Duration;
use log::debug;
use tch::{Device, Kind, TchError, Tensor};
use crate::HbacError;
use rayon::prelude::*;
use crate::cache_on_threads::CacheOnThreads;
use crate::TensorSliceIdx;

trait TensorsStore {
    fn get_tensor(&mut self, index: usize) -> Result<&Box<Tensor>, HbacError>;

    fn tensors_len(&self) -> usize;
}
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

//TODO remove
struct LazyTensorsStore {
    tensor_refs:Vec<String>,
    loaded:HashMap<String, Box<Tensor>>,
    tensor_loader: Box<dyn TensorLoader + Send>
}

trait TensorLoader {
    fn load(&self, tensor_ref:&String)->Tensor;
}

struct HashTensorLoader {
    tensors:HashMap<String, Tensor>
}

impl HashTensorLoader {
    pub fn new(tensors: HashMap<String, Tensor>) -> Box<Self> {
        Box::new(Self { tensors })
    }
}

impl TensorLoader for HashTensorLoader {
    fn load(&self, tensor_ref: &String) -> Tensor {
        debug!("tensor_ref={:?} self.tensors={:?}", tensor_ref, self.tensors);
        self.tensors.get(tensor_ref).unwrap().copy()
    }
}

impl LazyTensorsStore {

    pub fn new(tensor_refs: Vec<String>, tensor_loader: Box<dyn TensorLoader + Send>) -> Box<Self> {
        let mut loaded = HashMap::new();

        preload(&mut loaded, &tensor_loader, &tensor_refs[0]);

        Box::new(Self {
            tensor_refs,
            loaded: loaded,
            tensor_loader
        })
    }
}

impl TensorsStore for LazyTensorsStore {
    fn get_tensor(&mut self, index: usize) -> Result<&Box<Tensor>, HbacError> {

        if index < self.tensor_refs.len() -1 {
            preload(&mut self.loaded, &self.tensor_loader, &self.tensor_refs[index + 1]);
        }

        let loaded_tensor = self.loaded.get(&self.tensor_refs[index]);

        match loaded_tensor {
            None => {
                Err(HbacError::new("not tensor".into()))
            }
            Some(tensor) => {
                Ok(tensor)
            }
        }
    }

    fn tensors_len(&self) -> usize {
        self.tensor_refs.len()
    }
}

fn preload(loaded:&mut HashMap<String, Box<Tensor>>, tensor_loader: &Box<dyn TensorLoader + Send>, tensor_ref: &String) {
    loaded.insert(tensor_ref.clone(), Box::new(tensor_loader.load(tensor_ref)));
}


pub(crate) struct TensorsIter {//TODO SliceIndex?
    tensors_store: Box<dyn TensorsStore + Send>,
    tensor_index_0:usize,
    tensor_index_1:usize,

    slice_index_0:i64,
    slice_index_1:i64,

    dim:usize,
    step:i64,
    m:i64,
    iteration_n:AtomicI64,
}

impl TensorsIter {
    pub fn new(tensors_store: Box<impl TensorsStore + Send +'static>, dim:usize, m:i64, step:i64) -> Self {

        assert!(step <= m);

        debug!("tensors_store.tensors_len()={}", tensors_store.tensors_len());

        Self {
            tensors_store,
            tensor_index_0: 0,
            tensor_index_1: 0,

            slice_index_0:0,
            slice_index_1:step,

            dim,
            step,
            m,
            iteration_n:  AtomicI64::new(0)
        }
    }
}
pub type TensorSlice = ((TensorSliceIdx), Tensor); //tensor idx, slice, tensor

impl Iterator for TensorsIter {
    type Item = Result<(TensorSlice, TensorSlice, i64), HbacError>;

    fn next(&mut self) -> Option<Self::Item> {
        debug!("self.tensor_index_0={:?}, self.slice_index_0={:?}, self.tensor_index_1={:?}, self.slice_index_1={:?}", self.tensor_index_0, self.slice_index_0, self.tensor_index_1, self.slice_index_1);


        if self.tensor_index_0 < self.tensors_store.tensors_len() {
            let v = Some(Ok((
                (TensorSliceIdx(self.tensor_index_0, self.slice_index_0), self.tensors_store.get_tensor(self.tensor_index_0).unwrap().f_slice(self.dim as i64, Some(self.slice_index_0), Some(self.slice_index_0 + self.m), 1).unwrap()),//TODO unwrap
                (TensorSliceIdx(self.tensor_index_1, self.slice_index_1), self.tensors_store.get_tensor(self.tensor_index_1).unwrap().f_slice(self.dim as i64, Some(self.slice_index_1), Some(self.slice_index_1 + self.m), 1).unwrap()),
                self.iteration_n.fetch_add(1, Ordering::Relaxed)
            )));


            //TODO m < tensor
            if self.slice_index_1 + self.step + self.m <= self.tensors_store.get_tensor(self.tensor_index_1).unwrap().size()[self.dim] {
                self.slice_index_1 += self.step;

                debug!("1");

            } else {

                if self.tensor_index_1 < self.tensors_store.tensors_len() -1 {
                    self.tensor_index_1 += 1;
                    self.slice_index_1 = 0;

                    debug!("2.1");

                } else {
                    if self.slice_index_0 + self.step + self.m <= self.tensors_store.get_tensor(self.tensor_index_0).unwrap().size()[self.dim] {
                        self.slice_index_0 += self.step;

                        debug!("2.2.1");
                    } else {
                        debug!("2.2.2");

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



pub(crate) struct CacheOnThreadsStore {
    cache:CacheOnThreads<String, Box<Tensor>>,
    keys:Vec<String>,
    preload_d:usize
}

impl CacheOnThreadsStore {
    pub fn new<F>(loader: F, keys: Vec<String>, preload_d:usize) -> Self where F:Fn(&String) -> Box<Tensor> + Sync + Send + 'static{
        Self {
            cache: CacheOnThreads::new(loader),
            keys,
            preload_d

        }
    }

    fn index_to_key(&self, i:usize)->&String {
        &self.keys[i]
    }
}

impl TensorsStore for CacheOnThreadsStore {
    fn get_tensor(&mut self, index: usize) -> Result<&Box<Tensor>, HbacError> {

        if index+self.preload_d < self.keys.len() && index % self.preload_d == 0 {
            debug!("1. get_tensor: index={}", index);

            self.cache.load_blocking_preload_and_get(
                &self.index_to_key(index).clone(),
                (index..index + self.preload_d).map(|i| self.index_to_key(i).clone()).collect::<Vec<String>>().as_slice()
            ).ok_or(HbacError::new("no tensor".into()))

        } else {
            debug!("2. get_tensor: index={}", index);

            self.cache.load_blocking_and_get(
                &self.index_to_key(index).clone()
            ).ok_or(HbacError::new("no tensor".into()))
        }
    }

    fn tensors_len(&self) -> usize {
        self.keys.len()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::Write;
    use std::time::Instant;
    use log::{debug, LevelFilter, SetLoggerError};
    use rayon::prelude::ParallelBridge;
    use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger};
    use tch::{Device, Kind, Tensor};
    use crate::HbacError;
    use crate::tensor_iter::{LazyTensorsStore, TensorsIter, HashTensorLoader, VecTensorsStore, CacheOnThreadsStore};
    use hbac_common::utils::*;
    use rayon::iter::ParallelIterator;

    #[ctor::ctor]
    fn init() {
        match CombinedLogger::init(
            vec![
                TermLogger::new(LevelFilter::Debug, Config::default(), TerminalMode::Mixed, ColorChoice::Auto)
            ]
        ){
            Ok(_) => {}
            Err(_) => {}
        };
    }


    #[test]
    fn test_one_inner(){

        let t_0= Tensor::vstack(&[
            Tensor::from_slice((0..10).collect::<Vec<i64>>().as_slice()),
            Tensor::from_slice((0..10).collect::<Vec<i64>>().as_slice())
        ]);


        let ts = TensorsIter::new(VecTensorsStore::new(vec![t_0]), 1, 2, 2);
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


        let ts = TensorsIter::new(VecTensorsStore::new(vec![t_0]), 1, 3, 2);
        let r = ts_to_vec(ts);
        //
        // for r in r.clone() {
        //     println!("r={:?}", r);
        // }
        assert_eq!(vec![
            "i=0 0: [0.0, 1.0, 2.0] 1: [2.0, 3.0, 4.0]",
            "i=1 0: [0.0, 1.0, 2.0] 1: [4.0, 5.0, 6.0]",
            "i=2 0: [0.0, 1.0, 2.0] 1: [6.0, 7.0, 8.0]",
            "i=3 0: [2.0, 3.0, 4.0] 1: [5.0, 6.0, 7.0]", //+m instead of +step
            "i=4 0: [2.0, 3.0, 4.0] 1: [7.0, 8.0, 9.0]",
            "i=5 0: [4.0, 5.0, 6.0] 1: [7.0, 8.0, 9.0]",
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


        let ts = TensorsIter::new(VecTensorsStore::new(vec![t_0]), 1, 3, 2);
        let r = ts_to_vec(ts);

        // for r in r.clone() {
        //     println!("r={:?}", r);
        // }
        assert_eq!(vec![
            "i=0 0: [0.0, 1.0, 2.0] 1: [2.0, 3.0, 4.0]",
            "i=1 0: [0.0, 1.0, 2.0] 1: [4.0, 5.0, 6.0]",
            "i=2 0: [0.0, 1.0, 2.0] 1: [6.0, 7.0, 8.0]",
            "i=3 0: [0.0, 1.0, 2.0] 1: [8.0, 9.0, 10.0]",
            "i=4 0: [2.0, 3.0, 4.0] 1: [5.0, 6.0, 7.0]",
            "i=5 0: [2.0, 3.0, 4.0] 1: [7.0, 8.0, 9.0]",
            "i=6 0: [4.0, 5.0, 6.0] 1: [7.0, 8.0, 9.0]"
        ], r
        );

    }

    fn ts_to_vec(ts: TensorsIter) -> Vec<String> {
        let r = ts.enumerate().map(|(i, v)| {
            match v {
                Ok(ref x) => {
                    format!("i={} 0: {:?} 1: {:?}", i, from::<Vec<Vec<f32>>>(&x.0.1)[0], from::<Vec<Vec<f32>>>(&x.1.1)[0])
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


        let ts = TensorsIter::new(VecTensorsStore::new(vec![t_0]), 1, 2, 2);
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


        let ts = TensorsIter::new(VecTensorsStore::new(vec![t_0, t_1]), 1, 2, 2);
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

    #[test]
    fn test_two_inner_lazy() {
        let t_0 = Tensor::vstack(&[
            Tensor::from_slice((0..10).collect::<Vec<i64>>().as_slice()),
            Tensor::from_slice((0..10).collect::<Vec<i64>>().as_slice())
        ]);

        let t_1 = Tensor::vstack(&[
            Tensor::from_slice((0..100).step_by(10).collect::<Vec<i64>>().as_slice()),
            Tensor::from_slice((0..100).step_by(10).collect::<Vec<i64>>().as_slice())
        ]);

        let mut tensors_map = HashMap::new();
        tensors_map.insert("t_0".into(), t_0);
        tensors_map.insert("t_1".into(), t_1);


        let ts = TensorsIter::new(
            LazyTensorsStore::new(vec!["t_0".into(), "t_1".into()], HashTensorLoader::new(tensors_map)),
            1, 2, 2
        );

        // let r = ts_to_vec(ts);
        ts.par_bridge().for_each(|v|{
            println!("v={:?}", v);
        });
    }

    fn load_rand_tensor(key:&String) ->Box<Tensor>{
        Box::new(Tensor::rand(&[2, 10], (Kind::Float, Device::Cpu)))
    }

    #[test]
    fn test_cache_on_threads() {
        let vec_8:Vec<String> = vec![
            "key1".into(),
            "key2".into(),
            "key3".into(),
            "key4".into(),
            "key5".into(),
            "key6".into(),
            "key7".into(),
            "key8".into()
        ];

        let vec_100 = (0..20).map(|i|format!("key_{}", i)).collect::<Vec<String>>();

        let ts = TensorsIter::new(
            Box::new(CacheOnThreadsStore::new(load_rand_tensor, vec_100, 2)),
            1, 2, 2
        );

        // let r = ts_to_vec(ts);
        let dots = ts.par_bridge().map(|r|{
            match r {
                Ok(v) => {
                    v.0.1.tensordot(&v.1.1, 1, 1)
                }
                Err(e) => {
                    panic!("v={:?}", e);
                }
            }
        }).collect::<Vec<Tensor>>();

        println!("dots {}", dots.len());
    }

    // #[test]
    fn test_time(){

        let all_t = (0..4_000).map(|_|Tensor::rand(&[20, 100_000], (Kind::Float, Device::Cpu))).collect::<Vec<Tensor>>();
        let ts = TensorsIter::new(VecTensorsStore::new(all_t), 1, 1_000, 100);

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
