#![feature(async_fn_traits, unboxed_closures)]
#![feature(generic_const_exprs)]
#![feature(associated_type_defaults)]
#![allow(async_fn_in_trait)]
#![allow(incomplete_features)]
#![allow(confusable_idents)]
#![allow(deprecated)]
extern crate core;
extern crate self as thoosa_common;

pub mod run_config_loader;
pub mod tho_device;
pub mod utils;

pub mod args;
pub mod bool_iter;
pub mod commands;
pub mod cuda_copier;
pub mod cuda_utils;
pub mod data_with_extra;
pub mod dims;
pub mod ipc;
pub mod nets;
pub mod parzen;
pub mod progress_socket;
pub mod resources;
pub mod runner;
pub mod stages;
pub mod tho_tensor;
pub mod torch_runner;
pub mod transforms;
pub mod ws;

mod version {
    include!(concat!(env!("OUT_DIR"), "/version.rs"));
}

pub mod instance_cache;
pub mod tch_error;
