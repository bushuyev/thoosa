pub mod tho_tensor;

use crate::ipc::codec::IpcCodec;
use crate::ipc::cuda_wokrer::{WorkerStageInput, WorkerStageOutput};
use crate::ipc::worker_msg::{WorkerInput, WorkerOutput};
use futures_util::FutureExt;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::io::Write;
use thoosa_shared::ThoError;
use thoosa_shared::traits::MaxParallelCtx;
use thoosa_shared::traits::{Stage, SyncStage};


pub trait ToFromBytes { //TODO rename
    fn to_bytes<T>(value: &T) -> Result<Vec<u8>, ThoError>
    where
        T: Serialize + ?Sized;
    fn from_bytes<'a, T>(s: &'a [u8]) -> Result<T, ThoError>
    where
        T: Deserialize<'a>;
}

pub struct PostcardToFromBytes;

impl ToFromBytes for PostcardToFromBytes {
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="PostcardToFromBytes::to_bytes"))]
    fn to_bytes<T>(value: &T) -> Result<Vec<u8>, ThoError>
    where
        T: Serialize + ?Sized{
        postcard::to_allocvec(&value).map_err(|e|ThoError::new(e.to_string()))
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="PostcardToFromBytes::from_bytes"))]
    fn from_bytes<'a, T>(s: &'a [u8]) -> Result<T, ThoError>
    where
        T: Deserialize<'a>
    {
        postcard::from_bytes::<T>(s).map_err(|e|ThoError::new(e.to_string()))
    }
}
