#![feature(iter_array_chunks)]
#![feature(async_fn_traits, fn_traits, unboxed_closures)]
#![allow(
    unused_imports,
    unused_variables,
    unused_mut,
    dead_code,
    non_snake_case,
    redundant_semicolons,
    noop_method_call,
    unused_parens
)]
extern crate core;

pub mod distance;
pub mod find;
pub mod fragments;
pub mod pearson_corr;
pub mod criteria;
