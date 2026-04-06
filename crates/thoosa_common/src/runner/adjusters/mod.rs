use mockall_double::double;
use thoosa_shared::traits::CachableIdTr;
#[double]
use crate::runner::global_ctx::GlobalCtx;

pub mod dump_to_cpu;
pub mod max_parallel;
pub mod splits;