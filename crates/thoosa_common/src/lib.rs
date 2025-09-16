#![feature(async_fn_traits, fn_traits, unboxed_closures)]
extern crate core;

use tch::Device;

pub mod utils;

pub mod builder;
pub mod my_tch;
pub mod simp;
pub mod train_nets;
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
pub mod transforms;
pub mod find_motifs_run_config;
pub mod dims;
pub mod applicables;
pub mod batches;
pub mod tho_tensor;

pub trait ResourceLoader:Send+Sync {
    type K;
    type V;

    fn load(&self, key:&Self::K, device:Device)->Self::V;
}
