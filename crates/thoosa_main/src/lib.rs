#![feature(unboxed_closures)]
#![feature(trivial_bounds)]
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
pub mod stages;

//TODO use directly?
pub use thoosa_common::dims::ChannelsData;
pub use stages::apply_motif_nets::ApplyMotifNets;
pub use stages::motif_stats::random_non_overlapping_slices;
pub use stages::tensor_loader::RawTensorLoader;
pub use stages::train_motif_nets::{TrainMotifNetInput, TrainMotifNets};
pub use stages::transforms::freq_filter::FreqFilter;
