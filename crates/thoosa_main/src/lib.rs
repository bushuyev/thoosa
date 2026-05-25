#![feature(unboxed_closures)]
#![recursion_limit = "256"]
#![allow(
    unused_imports,
    unused_variables,
    unused_mut,
    dead_code,
    non_snake_case,
    private_interfaces,
    redundant_semicolons,
    noop_method_call,
    deprecated
)]
pub mod cli;
pub mod data;
mod buffer_limits;
mod progress_socket;
pub mod tasks;
pub mod version;

//TODO use directly?
pub use tasks::cross_dims::ChannelsData;
pub use tasks::apply_motif_nets::ApplyMotifNets;
pub use tasks::motif_stats::random_non_overlapping_slices;
pub use tasks::tensor_loader::RawTensorLoader;
pub use tasks::train_motif_nets::{TrainMotifNetInput, TrainMotifNets};
pub use tasks::transforms::freq_filter::FreqFilter;
