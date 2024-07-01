#![feature(iter_array_chunks)]
extern crate core;

use std::fmt::Display;
use tch::Device;

pub mod utils;

pub mod builder;
pub mod my_tch;
pub mod simp;
pub mod train_nets;
mod opt_steps;
mod opt;
mod bfgs;
mod argmin;
mod gaussian;
mod acquisitions;
mod parzen;

pub trait ResourceLoader:Send+Sync {
    type K;
    type V;

    fn load(&self, key:&Self::K, device:Device)->Box<Self::V>;
}