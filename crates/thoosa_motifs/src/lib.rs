#![feature(unboxed_closures)]
#![feature(iter_array_chunks)]
#![allow(
    unused_imports,
    unused_variables,
    unused_mut,
    dead_code,
    non_snake_case,
    non_camel_case_types,
    redundant_semicolons,
    noop_method_call,
    unused_parens,
    deprecated
)]
extern crate core;

pub mod distance;
pub mod find;
pub mod fragments;
pub mod pearson_corr;
pub mod criteria;
pub mod common_stats;
pub mod fs_store;
pub mod motif_tracker_store;
pub mod motifs_store;
