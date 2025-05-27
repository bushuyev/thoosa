#![feature(iter_array_chunks)]
#![feature(async_fn_traits, fn_traits, unboxed_closures)]
extern crate core;

mod tensor_iter;
pub mod fragments;
pub mod find;
pub mod distance;
pub mod chunks;
