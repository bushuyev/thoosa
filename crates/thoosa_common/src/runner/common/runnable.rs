use crate::runner::common::cachable::DevicesWrap;
use std::sync::Arc;
use thoosa_shared::ThoError;
use thoosa_shared::traits::{ErasedStageState, Stage, StageWithState};
use tokio::sync::MutexGuard;
//some operation on data - pattern search, dnn train, whatever what takes input and produces output

pub type WrappedStageRunInput<S> = Arc<DevicesWrap<<S as Stage>::In<'static>>>;
pub type WrappedStageRunOutput<S> = Arc<DevicesWrap<<S as Stage>::Out>>;
pub type PlainStageRunOutput<S> = Arc<<S as Stage>::Out>;



