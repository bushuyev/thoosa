extern crate core;

use std::fmt::Display;
use std::sync::{Arc, Mutex};

use log::debug;
use tch::{Device, Kind, Tensor};
use rand::Rng;
use rayon::iter::{ParallelBridge, ParallelIterator};

pub mod utils;
mod tensor_iter;
mod cache_on_threads;
pub mod fragments;
pub mod find;

use std::io::Write;
use std::time::Instant;
use hbac_common::HbacError;



#[derive(Debug, PartialEq, Hash, Eq, Copy, Clone)]
pub struct TensorSliceIdx(pub usize, pub i64);

impl From<(usize, i64)> for TensorSliceIdx {
    fn from(value: (usize, i64)) -> Self {
        TensorSliceIdx(value.0, value.1)
    }
}

impl Display for TensorSliceIdx {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{},{}", self.0, self.1)
    }
}