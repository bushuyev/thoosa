pub mod tho_tensor;
pub mod json;

use crate::ipc::cuda_wokrer::{WorkerStageInput, WorkerStageOutput};
use crate::ipc::worker_msg::{WorkerInput};
use json::IpcCodec;
use futures_util::FutureExt;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::io::Write;
use thoosa_shared::ThoError;
use thoosa_shared::traits::MaxParallelCtx;
use thoosa_shared::traits::{Stage};
use thoosa_common::runner::sync_stage::{SyncStage};


pub trait ToFromBytes { //TODO rename
    fn to_bytes<T>(value: &T) -> anyhow::Result<Vec<u8>>
    where
        T: Serialize + ?Sized;
    fn from_bytes<'a, T>(s: &'a [u8]) -> anyhow::Result<T>
    where
        T: Deserialize<'a>;
}

pub struct PostcardToFromBytes;

impl ToFromBytes for PostcardToFromBytes {
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="PostcardToFromBytes::to_bytes"))]
    fn to_bytes<T>(value: &T) -> anyhow::Result<Vec<u8>>
    where
        T: Serialize + ?Sized{
        Ok(postcard::to_allocvec(&value)?)
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="PostcardToFromBytes::from_bytes"))]
    fn from_bytes<'a, T>(s: &'a [u8]) -> anyhow::Result<T>
    where
        T: Deserialize<'a>
    {
        Ok(postcard::from_bytes::<T>(s)?)
    }
}
