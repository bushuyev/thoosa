use std::ops::Range;
#[cfg(test)]
use crate::common::*;
#[cfg(test)]
use ambassador::Delegate;
use async_trait::async_trait;
use log::debug;
use thoosa_shared::{CopyToDevice, ThoosaError};
use thoosa_shared::traits::*;
use thoosa_shared::traits::{DummyCachableId, MockWithId};

#[derive(Debug, Clone)]
pub struct DummyTensor(pub Vec<f32>);

impl From<Vec<f32>> for DummyTensor {
    fn from(value: Vec<f32>) -> Self {
        DummyTensor(value)
    }
}

#[async_trait]
impl CopyToDevice for DummyTensor {
    async fn copy_to_device(&self, _dst_device: usize) -> Self {
        self.clone()
    }
}

pub fn parse_numeric_part(name: &str) -> Option<f64> {
    debug!("parse_numeric_part={name:?}");
    let digits = name.rsplit_once('_')?.1;
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }

    let int_val: u64 = digits.parse().ok()?;

    if digits.starts_with('0') && digits.len() > 1 {
        let scale = 10_u64.pow(digits.len() as u32 - 1);
        Some(int_val as f64 / scale as f64)
    } else {
        Some(int_val as f64)
    }
}

#[derive(Debug, Delegate)]
// #[delegate(Data, target = "data")]
#[delegate(WithId, target = "with_id")]
#[cfg(test)]
pub struct MockDataWithId {
    data: MockData,
    with_id: MockWithId,
    device: usize,
}

#[cfg(test)]
impl MockDataWithId {
    pub fn whatever() -> Self {
        Self::for_device_and_id(0, "id")
    }

    pub fn for_device(device: usize) -> Self {
        let mut me = Self::whatever();
        me.device = device;

        me
    }

    pub fn for_device_and_id(device: usize, id:&'static str) -> Self {
        let mut with_id = MockWithId::new();
        with_id
            .expect_id()
            .return_const(DummyCachableId::from(id));

        let mut mock_data = MockData::new();
        mock_data.expect_size().return_const([0]);

        Self {
            data: mock_data,
            with_id,
            device,
        }
    }

    pub fn device(&self) -> &usize {
        &self.device
    }
}

#[cfg(test)]
impl Select<(), std::ops::Range<[usize; 1]>> for MockDataWithId {
    type Output = MockDataWithId;

    fn subview(&self, selector: Range<[usize; 1]>) -> Result<Self::Output, ThoosaError> {
        let mut data = MockData::new();
        let len = selector.end[0].saturating_sub(selector.start[0]);
        data.expect_size().return_const([len]);

        let mut with_id = MockWithId::new();
        with_id.expect_id().return_const(self.id().clone());

        Ok(MockDataWithId {
            data,
            with_id,
            device: self.device,
        })
    }
}

#[cfg(test)]
impl Data for MockDataWithId {
    type Metrics = [usize; 1];


    fn size(&self) -> Self::Metrics {
        self.data.size()
    }

    fn device(&self) -> usize {
        self.device
    }
}

#[cfg(test)]
#[async_trait]
impl CopyToDevice for MockDataWithId {
    async fn copy_to_device(&self, dst_device: usize) -> Self {
        MockDataWithId::for_device_and_id(dst_device, self.id().0)
    }
}
