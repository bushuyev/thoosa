use std::fmt::Debug;
use log::{debug, warn};
use serde::de::Error as DeError;
use serde::ser::{Error as SerError, SerializeStruct};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tch::Kind;
use thoosa_shared::ThoError;
use thoosa_shared::traits::{Named, ThoSelector};
use crate::dims::Dims;
use crate::tho_tensor::{ThoTensor, ThoTensorInner};



impl<'de, Id, DM> Deserialize<'de> for ThoTensor<DM,Id>
where
    DM: Dims,
    Id: Debug + Named + Clone + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        debug!("Deserialize for ThoTensor::deserialize:");

        #[derive(Debug, Deserialize)]
        struct Helper<Id> {
            id: Id,
            ref_key: u64,
            size: Vec<i64>,
            kind: SerdeKind,
            selector:Option<Box<dyn ThoSelector>>,
            worker_id: crate::ipc::worker_msg::CudaWorkerId
        }

        let helper = Helper::deserialize(deserializer)?;

        debug!("Deserialize for ThoTensor::deserialize: helper={helper:?}");

        Ok(ThoTensor::new_ref(
            helper.id,
            helper.ref_key,
            helper.size,
            helper.kind.0,
            helper.selector,
            helper.worker_id
        ))

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
        unexpected => {
            warn!("unexpected kind: {unexpected}");
            return None
        },
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
        debug!("Serialize for ThoTensor::serialize: self={self:?}");
        match &self.inner {
            ThoTensorInner::Ref { key, size, kind, selector } => {

                // Postcard encodes struct fields positionally, so every field, including a None selector, must keep its slot.
                let mut state = serializer.serialize_struct("ThoTensor", 6)?;

                state.serialize_field("id", &self.id)?;
                state.serialize_field("ref_key", &key.key)?;
                state.serialize_field("size", size)?;

                state.serialize_field("kind", kind_to_str(kind).map_err(S::Error::custom)?)?;
                state.serialize_field("selector", selector)?;
                state.serialize_field("worker_id", &key.worker_id)?;
                state.end()
            }
            ThoTensorInner::Val { .. } => Err(serde::ser::Error::custom("only out-of-process ThoTensor refs can be serialized")),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::collections::HashMap;
    use std::fmt::Debug;
    use log::debug;
    use serde::{Deserialize, Serialize};
    use tch::Kind;
    use tracing::field::debug;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use crate::dims::AnyDims;
    use crate::ipc::worker_msg::{CudaWorkerMsgIn, Operation, WorkerInput};
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::runner::global_ctx::MockGlobalCtx;
    use crate::tho_tensor::{ThoTensor, ThoTensorInner};

    #[test]
    fn test_postcard_tho_tensor_ref_test(){
        let is_installed_ctx = MockGlobalCtx::is_installed_context();
        is_installed_ctx.expect().return_const(false);

        let t_ref:ThoTensor<AnyDims, SynKey> = ThoTensor::new_ref(mono_key!(0u32, "test").into(), 10, vec![3,10], Kind::Float,  None, crate::ipc::worker_msg::CudaWorkerId { device_idx: 2, generation: thoosa_shared::CudaDeviceGeneration(7) });
        // let t_ref:ThoTensor<AnyDims, ()> = ThoTensor::new_ref((), 10, vec![3,10], Kind::Float,  None, crate::ipc::worker_msg::CudaWorkerId { device_idx: 2, generation: thoosa_shared::CudaDeviceGeneration(7) });

        let cuda_tensor_bytes = serde_json::to_vec(&t_ref).unwrap();
        let cuda_tensor:ThoTensor<AnyDims, SynKey> = serde_json::from_slice(&cuda_tensor_bytes).unwrap();
        // let cuda_tensor:ThoTensor<AnyDims, ()> = serde_json::from_slice(&cuda_tensor_bytes).unwrap();

        debug!("cuda_tensor={cuda_tensor:?}");

        let ThoTensorInner::Ref { key, size, kind, selector } = cuda_tensor.inner else {
            panic!("should be ref");
        };

        assert_eq!(key.key, 10);
        assert_eq!(kind, Kind::Float);
        assert_eq!(size, vec![3,10]);
        assert_eq!(key.worker_id.device_idx, 2);
        assert_eq!(key.worker_id.generation, thoosa_shared::CudaDeviceGeneration(7));

        let cuda_tensor_bytes = postcard::to_allocvec(&t_ref).unwrap();
        let cuda_tensor:ThoTensor<AnyDims, SynKey> =  postcard::from_bytes::<ThoTensor<AnyDims, SynKey>>(&cuda_tensor_bytes).unwrap();
        // let cuda_tensor:ThoTensor<AnyDims, ()> =  postcard::from_bytes::<ThoTensor<AnyDims, ()>>(&cuda_tensor_bytes).unwrap();


        let ThoTensorInner::Ref { key, size, kind, selector } = cuda_tensor.inner else {
            panic!("should be ref");
        };

        assert_eq!(key.key, 10);
        assert_eq!(kind, Kind::Float);
        assert_eq!(size, vec![3,10]);
        assert_eq!(key.worker_id.device_idx, 2);
        assert_eq!(key.worker_id.generation, thoosa_shared::CudaDeviceGeneration(7));

    }

    #[test]
    fn test_postcard_with_enum(){
        trait StageWrapper:Debug {

        }


        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct StageAInit {
            value:u32
        }
        impl StageAInit {
            pub fn new(value: u32) -> Self {
                Self { value }
            }
        }


        #[derive(Debug)]
        struct StageA {}

        impl StageWrapper for StageA{}
        impl StageA {
            pub fn new(init:StageAInit) -> Self {
                Self {}
            }
        }



        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct StageBInit {
            value:u32
        }

        impl StageBInit {
            pub fn new(value: u32) -> Self {
                Self { value }
            }
        }

        #[derive(Debug)]
        struct StageB {}
        impl StageWrapper for StageB{}
        impl StageB {
            pub fn new(init:StageBInit) -> Self {
                Self {}
            }
        }

        #[typetag::serde]
        impl WorkerInput for StageAInit {

        }



        fn test_a_init(init:Box<dyn WorkerInput>) -> Box<dyn StageWrapper> {
            let init: Box<dyn Any> = init;

            let init_a = init.downcast::<StageAInit>().unwrap();
            Box::new(StageA::new(*init_a)) as Box<dyn StageWrapper>
        }

        fn test_b_init(init:Box<dyn WorkerInput>) -> Box<dyn StageWrapper> {
            let init: Box<dyn Any> = init;

            let init_b = init.downcast::<StageBInit>().unwrap();
            Box::new(StageB::new(*init_b)) as Box<dyn StageWrapper>
        }

        let mut endpoint_inits = HashMap::from([
            (
                "test_a".to_string(),
                Box::new(test_a_init) as Box<dyn Fn(Box<dyn WorkerInput>)->Box<dyn StageWrapper>>
            ),
            (
                "test_b".to_string(),
                Box::new(test_b_init) as Box<dyn Fn(Box<dyn WorkerInput>)->Box<dyn StageWrapper>>
            )
        ]);

        let msg = CudaWorkerMsgIn::new(crate::ipc::worker_msg::CudaWorkerId { device_idx: 2, generation: thoosa_shared::CudaDeviceGeneration(7) }, 0, Operation::InstallStage{
            endpoint_name:"test_a".to_string(),
            stage_id: Some(0.to_string()),
            init_data: Box::new(StageAInit::new(1))
        });

        let msg_bytes = postcard::to_allocvec(&msg).unwrap();
        let msg_de: CudaWorkerMsgIn =  postcard::from_bytes::<CudaWorkerMsgIn>(&msg_bytes).unwrap();
        debug!("msg_de={msg_de:?}");

        assert_eq!(msg.request_id, msg_de.request_id);

        let Operation::InstallStage{ endpoint_name: key, stage_id, init_data } = msg.operation else {
            panic!("wrong operation");
        };

        assert_eq!("test_a".to_string(), key);

        let mut stage_a_init = endpoint_inits.get(&key).unwrap();

        let stage_a = stage_a_init(init_data);


        debug!("stage_a={stage_a:?}");

    }
}
