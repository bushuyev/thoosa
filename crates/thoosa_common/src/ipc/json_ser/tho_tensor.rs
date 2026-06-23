use std::fmt::Debug;
use serde::de::Error as DeError;
use serde::ser::{Error as SerError, SerializeStruct};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tch::Kind;
use thoosa_shared::ThoError;
use thoosa_shared::traits::Named;
use crate::dims::Dims;
use crate::tho_tensor::{ThoTensor, ThoTensorInner};

#[derive(Debug, Clone)]
struct ThoTensorRef<Id> {
    id: Id,
    ref_key: u64,
    size: Vec<i64>,
    kind: Kind,
}

impl<Id> Serialize for ThoTensorRef<Id>
where
    Id: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ThoTensorRef", 4)?;

        state.serialize_field("id", &self.id)?;
        state.serialize_field("ref_key", &self.ref_key)?;
        state.serialize_field("size", &self.size)?;
        state.serialize_field("kind", &SerdeKind(self.kind))?;

        state.end()
    }
}

impl<'de, Id> Deserialize<'de> for ThoTensorRef<Id>
where
    Id: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper<Id> {
            id: Id,
            ref_key: u64,
            size: Vec<i64>,
            kind: SerdeKind,
        }

        let helper = Helper::deserialize(deserializer)?;

        Ok(Self {
            id: helper.id,
            ref_key: helper.ref_key,
            size: helper.size,
            kind: helper.kind.0,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct SerdeKind(Kind);

impl Serialize for SerdeKind {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = kind_to_str(&self.0).map_err(S::Error::custom)?;
        serializer.serialize_str(s)
    }
}

impl<'de> Deserialize<'de> for SerdeKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;

        let kind = str_to_kind(&s).ok_or_else(|| {
            D::Error::custom(format!("unknown tch::Kind string: {s:?}"))
        })?;

        Ok(Self(kind))
    }
}
pub fn kind_to_str(kind: &Kind) -> Result<&'static str, String> {
    match kind {
        Kind::Uint8 => Ok("uint8"),
        Kind::Int8 => Ok("int8"),
        Kind::Int16 => Ok("int16"),
        Kind::Int => Ok("int32"),
        Kind::Int64 => Ok("int64"),
        Kind::Half => Ok("float16"),
        Kind::Float => Ok("float32"),
        Kind::Double => Ok("float64"),
        Kind::BFloat16 => Ok("bfloat16"),
        Kind::Bool => Ok("bool"),

        Kind::ComplexHalf => Ok("complex32"),
        Kind::ComplexFloat => Ok("complex64"),
        Kind::ComplexDouble => Ok("complex128"),

        Kind::QInt8 => Ok("qint8"),
        Kind::QUInt8 => Ok("quint8"),
        Kind::QInt32 => Ok("qint32"),
        Kind::Float8e5m2 => Ok("Float8e5m2"),
        Kind::Float8e4m3fn => Ok("Float8e4m3fn"),
        Kind::Float8e5m2fnuz => Ok("Float8e5m2fnuz"),
        Kind::Float8e4m3fnuz => Ok("Float8e4m3fnuz"),

        other => Err(format!("unsupported tch::Kind variant: {other:?}")),
    }
}

pub fn str_to_kind(s: &str) -> Option<Kind> {
    Some(match s {
        "uint8" => Kind::Uint8,
        "int8" => Kind::Int8,
        "int16" => Kind::Int16,
        "int32" | "int" => Kind::Int,
        "int64" => Kind::Int64,

        "float16" | "half" => Kind::Half,
        "float32" | "float" => Kind::Float,
        "float64" | "double" => Kind::Double,
        "bfloat16" => Kind::BFloat16,

        "bool" => Kind::Bool,

        "complex32" | "complex_half" => Kind::ComplexHalf,
        "complex64" | "complex_float" => Kind::ComplexFloat,
        "complex128" | "complex_double" => Kind::ComplexDouble,

        "qint8" => Kind::QInt8,
        "quint8" => Kind::QUInt8,
        "qint32" => Kind::QInt32,
        "Float8e5m2" => Kind::Float8e5m2,
        "Float8e4m3fn" => Kind::Float8e4m3fn,
        "Float8e5m2fnuz" => Kind::Float8e5m2fnuz,
        "Float8e4m3fnuz" => Kind::Float8e4m3fnuz,
        _ => return None,
    })
}


impl<D, Id> Serialize for ThoTensor<D, Id>
where
    D: Dims,
    Id: Named + Debug + Clone + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer
    {
        match &self.inner {
            ThoTensorInner::Ref { key, size, kind } => {

                let mut state = serializer.serialize_struct("ThoTensor", 4)?;

                state.serialize_field("id", &self.id)?;
                state.serialize_field("ref_key", key)?;
                state.serialize_field("size", size)?;
                state.serialize_field("kind", kind_to_str(kind).map_err(S::Error::custom)?)?;
                state.end()
            }
            ThoTensorInner::Val { .. } => Err(serde::ser::Error::custom("only out-of-process ThoTensor refs can be serialized")),
        }
    }
}

impl<'de, D, Id> Deserialize<'de> for ThoTensor<D, Id>
where
    D: Dims,
    Id: Named + Debug + Clone + Deserialize<'de>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'de>,
    {
        let tensor_ref = ThoTensorRef::deserialize(deserializer)?;
        Ok(ThoTensor::new_ref(tensor_ref.id, tensor_ref.ref_key, tensor_ref.size, tensor_ref.kind))
    }
}