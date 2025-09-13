// #![allow(uncommon_codepoints)]
// #![feature(trait_upcasting)]
// #![feature(const_trait_impl)]
// #![feature(effects)]
// #![feature(iter_array_chunks)]
#![feature(async_fn_traits, fn_traits, unboxed_closures)]
#![allow(
    unused_imports,
    unused_variables,
    unused_mut,
    dead_code,
    non_snake_case,
    redundant_semicolons,
    noop_method_call,
    mismatched_lifetime_syntaxes,
    suspicious_double_ref_op,
    unpredictable_function_pointer_comparisons,
    unused_parens,
    unused_attributes
)]
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

pub trait ResourceLoader:Send+Sync {
    type K;
    type V;

    fn load(&self, key:&Self::K, device:Device)->Self::V;
}
