use std::ops::Range;

use async_trait::async_trait;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tch::IndexOp;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::traits::{Adjustable, Data, Metrics, MonoMetrics, Select, WithId};
use thoosa_shared::{CopyToDevice, ThoError};

use crate::dims::Dims;
use crate::tho_tensor::ThoTensor;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Named)]
pub(crate) struct FileInputId(pub(crate) &'static str);

impl Serialize for FileInputId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.0)
    }
}

impl<'de> Deserialize<'de> for FileInputId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(FileInputId(value.leak()))
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, Named)]
pub(crate) struct ModelTestId(pub(crate) &'static str);

impl Serialize for ModelTestId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.0)
    }
}

impl<'de> Deserialize<'de> for ModelTestId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Ok(ModelTestId(value.leak()))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FileInput {
    pub(crate) id: FileInputId,
}

impl Input for FileInput {}

impl WithId for FileInput {
    type ID = FileInputId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl From<&'static str> for FileInput {
    fn from(s: &'static str) -> Self {
        Self { id: FileInputId(s) }
    }
}

#[async_trait]
impl Data for FileInput {
    type Metrics = ();
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {}

    fn device(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        0
    }
}

#[derive(Debug)]
pub(crate) struct FileListInput {
    pub(crate) files: Vec<(&'static str, f64)>,
    pub(crate) id: ModelTestId,
}

impl FileListInput {
    pub(crate) fn new(files: Vec<(&'static str, f64)>, id: ModelTestId) -> Self {
        Self { files, id }
    }
}

impl Input for FileListInput {}

impl WithId for FileListInput {
    type ID = ModelTestId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[async_trait]
impl Data for FileListInput {
    type Metrics = ();
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {}

    fn device(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        0
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub(crate) struct ABData;

impl Dims for ABData {
    type Metrics = ABMetrics;
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ABRange {
    pub(crate) range: Range<ABMetrics>,
}

impl Adjustable for ABRange {}

impl Select<ABRange> for ThoTensor<ABData, ModelTestId> {
    type Output = ThoTensor<ABData, ModelTestId>;

    fn subview(&self, selector: ABRange) -> Result<Self::Output, ThoError> {
        use std::cmp::min;
        let size = ABMetrics::from(self.size());
        let d0 = selector.range.start.dim_0..min(size.dim_0, selector.range.end.dim_0);

        ThoTensor::new(self.id().clone(), self.t_ref().i((d0, ..)))
    }
}

impl From<Range<ABMetrics>> for ABRange {
    fn from(value: Range<ABMetrics>) -> Self {
        ABRange { range: value }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ABMetrics {
    pub(crate) dim_0: i64,
    pub(crate) dim_1: i64,
}

impl Metrics for ABMetrics {
    fn cap(&self, _other: &mut Self) {}
}

impl From<Vec<i64>> for ABMetrics {
    fn from(v: Vec<i64>) -> Self {
        ABMetrics {
            dim_0: *v.first().unwrap_or(&0),
            dim_1: *v.get(1).unwrap_or(&0),
        }
    }
}

impl std::ops::AddAssign<&ABMetrics> for ABMetrics {
    fn add_assign(&mut self, rhs: &ABMetrics) {
        self.dim_0 += rhs.dim_0;
    }
}

impl std::ops::AddAssign<ABMetrics> for ABMetrics {
    fn add_assign(&mut self, rhs: ABMetrics) {
        self.dim_0 += rhs.dim_0;
    }
}

impl std::ops::DivAssign<i64> for ABMetrics {
    fn div_assign(&mut self, rhs: i64) {
        if rhs != 0 {
            self.dim_0 /= rhs;
        }
    }
}

impl PartialOrd for ABMetrics {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dim_0.partial_cmp(&other.dim_0)
    }
}

impl std::ops::Add<&ABMetrics> for &ABMetrics {
    type Output = ABMetrics;

    fn add(self, rhs: &ABMetrics) -> Self::Output {
        ABMetrics {
            dim_0: self.dim_0 + rhs.dim_0,
            dim_1: self.dim_1,
        }
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub(crate) struct ClsOut;

impl Dims for ClsOut {
    type Metrics = MonoMetrics<1, 2>;
}

impl ClsOut {
    pub(crate) fn accuracy(t: &ThoTensor<ClsOut, ModelTestId>) -> f64 {
        t.double_value(&[0])
    }
}
