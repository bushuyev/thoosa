pub mod tho_tensor;
pub mod json;

use serde::{Deserialize, Serialize};


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
