#![feature(iter_array_chunks)]
#![feature(async_fn_traits, fn_traits, unboxed_closures)]
#![allow(unused_imports, unused_variables, unused_mut, dead_code, non_snake_case, redundant_semicolons, noop_method_call, unused_parens)]
extern crate core;

mod tensor_iter;
pub mod fragments;
pub mod find;
pub mod distance;
pub mod chunks;
mod tasks;
