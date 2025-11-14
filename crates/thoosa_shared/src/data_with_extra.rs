use std::fmt::Debug;
use async_trait::async_trait;
use thoosa_shared::traits::Adjustable;
use crate::applicables_runner::{Input, Output};
use crate::{CopyToDevice, ThoosaError};
use crate::traits::{Data, DataId, Select, WithId};

#[derive(Debug)]
pub struct DataWithExtra<D:Data+CopyToDevice+WithId, E:Clone+Debug> {
    data: D,
    extra: E,
}

impl<D:Data+CopyToDevice+WithId, E:Clone+Debug> DataWithExtra<D, E> {
    pub fn new(data: D, extra: E) -> Self {
        Self { data, extra }
    }

    pub fn data(&self) -> &D {
        &self.data
    }

    pub fn extra(&self) -> &E {
        &self.extra
    }
}

#[async_trait]
impl <D:Data+CopyToDevice+WithId + Sync, E:Clone+Debug+Sync> CopyToDevice for DataWithExtra<D, E> {
    async fn copy_to_device(&self, dst_device: usize) -> Self{
        DataWithExtra {
            data: self.data.copy_to_device(dst_device).await,
            extra: self.extra.clone(),
        }
    }
}

impl <D:Data+CopyToDevice+WithId+Sync, E:Clone+Debug+Sync> Data for DataWithExtra<D, E> {
    type Metrics = D::Metrics;

    fn size(&self) -> Self::Metrics {
        self.data.size()
    }

    fn device(&self) -> usize {
        self.data.device()
    }
}


impl <D:Data+CopyToDevice+WithId, E:Clone+Debug> Input for DataWithExtra<D, E> {}
impl <D:Data+CopyToDevice+WithId, E:Clone+Debug> WithId for DataWithExtra<D, E> {
    type ID = D::ID;

    fn id(&self) -> &Self::ID {
        self.data.id()
    }
}
impl <D:Data+CopyToDevice+WithId, E:Clone+Debug> Output for DataWithExtra<D, E> {}


impl <'s, S, D, E> Select<'s, S> for DataWithExtra<D, E>
where
    D:Select<'s, S, Output=D>,
    D:Data+CopyToDevice+WithId+Sync + 's,
    E:Clone+Debug+Sync + 's

{
    type Output= Self;

    fn subview(&self, selector: S) -> Result<Self::Output, ThoosaError> {
        Ok(DataWithExtra {
            data: self.data.subview(selector)?,
            extra: self.extra.clone()
        })
    }
}


impl <D:Data+CopyToDevice+WithId+Send+Sync, E:Clone+Debug+Send+Sync> Adjustable for DataWithExtra<D, E> {}