use std::cmp::Ordering;
use std::ops::{Add, AddAssign, DivAssign};
use log::debug;
use thoosa_shared::{MyError, ThoosaError};

#[derive(Debug, Clone, Default)]
pub struct TwoDimsMetrics {
    pub(crate) dim_0:i64,
    pub(crate) dim_1:i64
}

impl TwoDimsMetrics {
    pub fn channels(&self) -> i64 {
        self.dim_0
    }

    pub fn seq_len(&self) -> i64 {
        self.dim_1
    }
}

impl Add<&TwoDimsMetrics> for &TwoDimsMetrics {
    type Output = TwoDimsMetrics;

    fn add(self, rhs: &TwoDimsMetrics) -> Self::Output {
        TwoDimsMetrics {
            dim_0: self.dim_0 + rhs.dim_0,
            dim_1: self.dim_1,
        }
    }
}

impl DivAssign<i64> for TwoDimsMetrics {
    fn div_assign(&mut self, rhs: i64) {
        self.dim_0 /= rhs;
    }
}

impl PartialEq<i64> for TwoDimsMetrics {
    fn eq(&self, other: &i64) -> bool {
        self.dim_0.eq(other)
    }
}

impl PartialOrd<i64> for TwoDimsMetrics {
    fn partial_cmp(&self, other: &i64) -> Option<Ordering> {
        self.dim_0.partial_cmp( other)
    }
}

impl PartialEq<TwoDimsMetrics> for TwoDimsMetrics {
    fn eq(&self, other: &TwoDimsMetrics) -> bool {
        self.dim_0.eq(&other.dim_0)
    }
}

impl PartialOrd<TwoDimsMetrics> for TwoDimsMetrics {
    fn partial_cmp(&self, other: &TwoDimsMetrics) -> Option<Ordering> {
        self.dim_0.partial_cmp(&other.dim_0)
    }
}

impl <'a> AddAssign<&'a TwoDimsMetrics> for TwoDimsMetrics {
    fn add_assign(&mut self, rhs: &'a TwoDimsMetrics) {
        self.dim_0+=rhs.dim_0;
        debug!("AddAssign for LoadedTensorMetrics::add_assign rhs.dim_0={} self.dim_0={}", rhs.dim_0, self.dim_0);
        // self.dim_1+=rhs.dim_1;TODO dimension
    }
}

impl TryFrom<Vec<i64>> for TwoDimsMetrics {
    type Error = ThoosaError;

    fn try_from(value: Vec<i64>) -> Result<Self, Self::Error> {
        let &[dim_0, dim_1, ..] = value.as_slice() else {
            return Err(MyError::SomethingElse(format!("can't make LoadedTensorMetrics from {value:?}")).into());
        };

        Ok(Self {
            dim_0,
            dim_1
        })
    }
}

impl From<&TwoDimsMetrics> for [i64;2] {
    fn from(value: &TwoDimsMetrics) -> Self {
        [value.dim_0, value.dim_1]
    }
}


impl From<[i64;2]> for TwoDimsMetrics {
    fn from([dim_0, dim_1]: [i64; 2]) -> Self {
        Self {
            dim_0,
            dim_1
        }
    }
}

impl From<&[i64;2]> for TwoDimsMetrics {
    fn from([dim_0, dim_1]: &[i64; 2]) -> Self {
        Self {
            dim_0: *dim_0,
            dim_1: *dim_1
        }
    }
}

