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
mod buffer_limits;
mod tasks;
pub mod version;

pub use tasks::cross_dims::ChannelsData2;
pub use tasks::apply_motif_nets::ApplyMotifNets;
pub use tasks::random_slices_stats::random_non_overlapping_slices;
pub use tasks::tensor_loader::RawTensorLoader;
pub use tasks::train_motif_nets::{TrainMotifNetInput, TrainMotifNets};
pub use tasks::transforms::freq_filter::FreqFilter;
