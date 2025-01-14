#![allow(uncommon_codepoints)]
#![feature(trait_upcasting)]
// #![feature(const_trait_impl)]
// #![feature(effects)]
// #![feature(iter_array_chunks)]
extern crate core;

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
pub mod parzen;
pub mod loader;
pub mod ws;
pub mod runs;
pub mod bool_iter;
pub mod input_probs;

pub trait ResourceLoader:Send+Sync {
    type K;
    type V;

    fn load(&self, key:&Self::K, device:Device)->Box<Self::V>;
}