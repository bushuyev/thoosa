#![feature(async_fn_traits, fn_traits, unboxed_closures)]
#![allow(
    unused_imports,
    unused_variables,
    unused_mut,
    dead_code,
    non_snake_case,
    redundant_semicolons,
    noop_method_call
)]
pub mod cli;
pub mod data;
mod tasks;
pub mod version;