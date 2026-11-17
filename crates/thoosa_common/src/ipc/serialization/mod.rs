pub mod json;
pub mod tho_tensor;

use serde::{Deserialize, Serialize};
use crate::ipc::worker_msg::CudaWorkerId;
use thoosa_shared::ThoError;

// Both messages begin with these positional fields. Decode only this prefix on
// failure, so an invalid dynamic payload cannot hide its sender or request ID.
#[derive(Debug, Deserialize)]
pub(crate) struct MessageHeader {
    pub worker_id: CudaWorkerId,
    pub request_id: u64,
}

pub(crate) fn message_header(frame: &[u8]) -> postcard::Result<(MessageHeader, &[u8])> {
    postcard::take_from_bytes(frame)
}

pub(crate) fn decode_failure(direction: &str, peer: &str, frame: &[u8], error: &anyhow::Error) -> ThoError {
    // typetag's output map starts with its type tag. Recover it without invoking
    // any tensor deserializers; this diagnostic path never acquires remote refs.
    let output_tag = || -> Option<&str> {
        let (_, payload) = message_header(frame).ok()?;
        let ((status, present), payload) = postcard::take_from_bytes::<(u32, u8)>(payload).ok()?;
        if status != 0 || present != 1 { return None; }
        let ((fields, key, tag), _) = postcard::take_from_bytes::<(usize, &str, &str)>(payload).ok()?;
        (fields > 0 && key == "type").then_some(tag)
    };
    let header = message_header(frame).map(|(header, _)| format!("worker_id={:?} request_id={}", header.worker_id, header.request_id)).unwrap_or_else(|error| format!("unreadable header: {error}"));
    let excerpt: String = frame.iter().take(256).map(|byte| format!("{byte:02x}")).collect();
    ThoError::IpcProtocol(format!("decoding {direction} from {peer}: {header}; output_tag={:?}; frame_len={}; first_256_bytes={excerpt}; error={error:#}", output_tag(), frame.len()))
}

pub trait ToFromBytes {
    fn to_bytes<T>(value: &T) -> anyhow::Result<Vec<u8>>
    where
        T: Serialize + ?Sized;
    fn from_bytes<'a, T>(s: &'a [u8]) -> anyhow::Result<T>
    where
        T: Deserialize<'a>;
}

pub struct PostcardToFromBytes;

impl ToFromBytes for PostcardToFromBytes {
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "PostcardToFromBytes::to_bytes"))]
    fn to_bytes<T>(value: &T) -> anyhow::Result<Vec<u8>>
    where
        T: Serialize + ?Sized,
    {
        Ok(postcard::to_allocvec(&value)?)
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "PostcardToFromBytes::from_bytes"))]
    fn from_bytes<'a, T>(s: &'a [u8]) -> anyhow::Result<T>
    where
        T: Deserialize<'a>,
    {
        let (value, remaining) = postcard::take_from_bytes::<T>(s)?;
        // A frame contains exactly one message. Ignored trailing bytes can hide
        // mismatched schemas even when the first fields happened to deserialize.
        anyhow::ensure!(remaining.is_empty(), "{} trailing bytes after {}", remaining.len(), std::any::type_name::<T>());
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_postcard_frame_rejects_trailing_bytes() -> anyhow::Result<()> {
        let mut bytes = PostcardToFromBytes::to_bytes(&42u32)?;
        bytes.push(0);
        assert!(PostcardToFromBytes::from_bytes::<u32>(&bytes).unwrap_err().to_string().contains("trailing bytes"));
        Ok(())
    }
}
