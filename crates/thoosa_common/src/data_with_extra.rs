use async_trait::async_trait;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::Debug;
use std::sync::Arc;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::traits::{Adjustable, ThoSelector};
use thoosa_shared::traits::{Data, Select, WithId};
use thoosa_shared::{CopyToDevice, ThoError};

#[derive(Debug)]
pub struct DataWithExtra<D, E>
where
    D: Data + CopyToDevice + WithId,
    E: Debug + Send + Sync,
{
    pub data: D,       //needs to be copied to GPU
    pub extra: Arc<E>, // CPU-only, small, or at least not huge, usually cloneable struct which is logically related to data
}

impl<D, E> DataWithExtra<D, E>
where
    D: Data + CopyToDevice + WithId,
    E: Debug + Send + Sync,
{
    pub fn new(data: D, extra: E) -> Self {
        Self { data, extra: Arc::new(extra) }
    }

    pub fn data(&self) -> &D {
        &self.data
    }

    pub fn extra(&self) -> &E {
        &self.extra
    }
}

impl<D, E> Serialize for DataWithExtra<D, E>
where
    D: Serialize + Data + CopyToDevice + WithId + Sync,
    E: Serialize + Debug + Send + Sync,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("DataWithExtra", 2)?;

        state.serialize_field("data", &self.data)?;
        state.serialize_field("extra", self.extra.as_ref())?;

        state.end()
    }
}

impl<'de, Da, E> Deserialize<'de> for DataWithExtra<Da, E>
where
    Da: Deserialize<'de> + Data + CopyToDevice + WithId + Sync,
    E: Deserialize<'de> + Debug + Send + Sync,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct DataWithExtra_<Da, E> {
            data: Da,
            extra: E,
        }

        let helper = DataWithExtra_::<Da, E>::deserialize(deserializer)?;

        Ok(DataWithExtra { data: helper.data, extra: Arc::new(helper.extra) })
    }
}

#[async_trait]
impl<D, E> CopyToDevice for DataWithExtra<D, E>
where
    D: Data + CopyToDevice + WithId + Sync,
    E: Debug + Send + Sync,
{
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        Ok(DataWithExtra {
            data: self.data.copy_to_device(dst_device).await?,
            extra: Arc::clone(&self.extra),
        })
    }
}

impl<D, E> Data for DataWithExtra<D, E>
where
    D: Data + CopyToDevice + WithId + Sync,
    E: Debug + Send + Sync,
{
    type Metrics = D::Metrics;

    fn size(&self) -> Self::Metrics {
        self.data.size()
    }

    fn tho_device_idx(&self) -> usize {
        self.data.tho_device_idx()
    }

    fn mem_size(&self) -> u64 {
        self.data.mem_size()
    }
}

impl<D, E> Input for DataWithExtra<D, E>
where
    D: Data + CopyToDevice + WithId + Sync,
    E: Debug + Send + Sync,
{
}
impl<D, E> WithId for DataWithExtra<D, E>
where
    D: Data + CopyToDevice + WithId,
    E: Debug + Send + Sync,
{
    type ID = D::ID;

    fn id(&self) -> &Self::ID {
        self.data.id()
    }
}
impl<D, E> Output for DataWithExtra<D, E>
where
    D: Data + CopyToDevice + WithId + Sync,
    E: Debug + Send + Sync,
{
}

impl<S, D, E> Select<S> for DataWithExtra<D, E>
where
    S: ThoSelector,
    D: Data + CopyToDevice + WithId + Select<S, Output = D> + Sync,
    E: Debug + Send + Sync,
{
    type Output = Self;

    fn subview(&self, selector: S) -> Result<Self::Output, ThoError> {
        Ok(DataWithExtra {
            data: self.data.subview(selector)?,
            extra: self.extra.clone(),
        })
    }
}

impl<D, E> Adjustable for DataWithExtra<D, E>
where
    D: Data + CopyToDevice + WithId + Send + Sync,
    E: Debug + Send + Sync,
{
}
