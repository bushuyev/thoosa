use std::fmt::Debug;
use std::sync::Arc;
use async_trait::async_trait;
use thoosa_shared::traits::Adjustable;
use crate::applicables_runner::{Input, Output};
use crate::{CopyToDevice, ThoError};
// use crate::copy_mode::CpuOrGpu;
use crate::traits::{Data, Select, WithId};

#[derive(Debug)]
pub struct DataWithExtra<D, E>
where
    D:Data+CopyToDevice+WithId,
    E:Debug + Send + Sync,
{
    data: D,//needs to be copied to GPU
    extra: Arc<E>,// CPU-only, small, or at least not huge, usually cloneable struct which is logically related to data
}

impl<D, E> DataWithExtra<D, E>
where
    D: Data + CopyToDevice + WithId,
    E: Debug + Send + Sync,
{
    pub fn new(data: D, extra: E) -> Self {
        Self { data, extra:Arc::new(extra) }
    }

    pub fn data(&self) -> &D {
        &self.data
    }

    pub fn extra(&self) -> &E {
        &self.extra
    }
}

#[async_trait]
impl <D, E> CopyToDevice for DataWithExtra<D, E>
where
    D: Data + CopyToDevice + WithId + Sync,
    E: Debug + Send + Sync,
{
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError>{
        Ok(DataWithExtra {
            data: self.data.copy_to_device(dst_device).await?,
            extra: Arc::clone(&self.extra),
        })
    }
}

impl <D, E> Data for DataWithExtra<D, E>
where
    D: Data + CopyToDevice + WithId + Sync,
    E:Debug + Send + Sync,
{
    type Metrics = D::Metrics;
    // type M = CpuOrGpu<Self>;

    fn size(&self) -> Self::Metrics {
        self.data.size()
    }

    fn device(&self) -> usize {
        self.data.device()
    }

    fn mem_size(&self) -> u64 {
        self.data.mem_size()
    }
}


impl <D, E> Input for DataWithExtra<D, E>
where
    D: Data + CopyToDevice + WithId + Sync,
    E:Debug + Send + Sync,
{

}
impl <D, E> WithId for DataWithExtra<D, E>
where
    D: Data + CopyToDevice + WithId,
    E:Debug + Send + Sync,
{
    type ID = D::ID;

    fn id(&self) -> &Self::ID {
        self.data.id()
    }
}
impl <D, E> Output for DataWithExtra<D, E>
where
    D: Data + CopyToDevice + WithId + Sync,
    E:Debug + Send + Sync,
{
}


impl <Selector, D, E> Select<Selector> for DataWithExtra<D, E>
where
    D: Data + CopyToDevice + WithId + Select<Selector, Output=D> + Sync,
    E:Debug + Send + Sync,
{
    type Output = Self;

    fn subview(&self, selector: Selector) -> Result<Self::Output, ThoError> {
        Ok(DataWithExtra {
            data: self.data.subview(selector)?,
            extra: self.extra.clone()
        })
    }
}

impl <D, E> Adjustable for DataWithExtra<D, E>
where
    D: Data + CopyToDevice + WithId + Send + Sync,
    E:Debug + Send + Sync,
{}

#[cfg(test)]
pub mod tests {

    #[test]
    fn test_0(){

    }

}
