use log::debug;
use std::cmp::Ordering;
use std::ops::{Add, AddAssign, DivAssign};
use thoosa_shared::{MyError, ThoosaError};

#[derive(Debug, Clone, Default)]
pub struct TwoDimsMetrics<const Idx:usize, const Size:usize> {//TODO rename to SingleDimMetrics and use I to detect which dim to act on
     dim_0: i64,
     dim_1: i64,
}

impl <const Idx:usize, const Size:usize> TwoDimsMetrics<Idx, Size> {
    pub fn dim_0(&self) -> i64 {
        self.dim_0
    }

    pub fn dim_1(&self) -> i64 {
        self.dim_1
    }
}

impl <const Idx:usize, const Size:usize> Add<&TwoDimsMetrics<Idx, Size>> for &TwoDimsMetrics<Idx, Size> {
    type Output = TwoDimsMetrics<Idx, Size>;

    fn add(self, rhs: &TwoDimsMetrics<Idx, Size>) -> Self::Output {
        TwoDimsMetrics {
            dim_0: self.dim_0 + rhs.dim_0,
            dim_1: self.dim_1,
        }
    }
}

impl <const Idx:usize, const Size:usize> DivAssign<i64> for TwoDimsMetrics<Idx, Size> {
    fn div_assign(&mut self, rhs: i64) {
        self.dim_0 /= rhs;
    }
}

impl <const Idx:usize, const Size:usize> PartialEq<i64> for TwoDimsMetrics<Idx, Size> {
    fn eq(&self, other: &i64) -> bool {
        self.dim_0.eq(other)
    }
}

impl <const Idx:usize, const Size:usize> PartialOrd<i64> for TwoDimsMetrics<Idx, Size> {
    fn partial_cmp(&self, other: &i64) -> Option<Ordering> {
        self.dim_0.partial_cmp(other)
    }
}

impl <const Idx:usize, const Size:usize> PartialEq<TwoDimsMetrics<Idx, Size>> for TwoDimsMetrics<Idx, Size> {
    fn eq(&self, other: &TwoDimsMetrics<Idx, Size>) -> bool {
        self.dim_0.eq(&other.dim_0)
    }
}

impl <const Idx:usize, const Size:usize> PartialOrd<TwoDimsMetrics<Idx, Size>> for TwoDimsMetrics<Idx, Size> {
    fn partial_cmp(&self, other: &TwoDimsMetrics<Idx, Size>) -> Option<Ordering> {
        self.dim_0.partial_cmp(&other.dim_0)
    }
}

impl<'a, const Idx:usize, const Size:usize> AddAssign<&'a TwoDimsMetrics<Idx, Size>> for TwoDimsMetrics<Idx, Size> {
    fn add_assign(&mut self, rhs: &'a TwoDimsMetrics<Idx, Size>) {
        self.dim_0 += rhs.dim_0;
        debug!(
            "AddAssign for LoadedTensorMetrics::add_assign rhs.dim_0={} self.dim_0={}",
            rhs.dim_0, self.dim_0
        );
        // self.dim_1+=rhs.dim_1;TODO dimension
    }
}

impl <const Idx:usize, const Size:usize> TryFrom<Vec<i64>> for TwoDimsMetrics<Idx, Size> {
    type Error = ThoosaError;

    fn try_from(value: Vec<i64>) -> Result<Self, Self::Error> {
        let &[dim_0, dim_1, ..] = value.as_slice() else {
            return Err(MyError::SomethingElse(format!(
                "can't make LoadedTensorMetrics from {value:?}"
            ))
            .into());
        };

        Ok(Self { dim_0, dim_1 })
    }
}

impl <const Idx:usize, const Size:usize> From<&TwoDimsMetrics<Idx, Size>> for [i64; 2] {
    fn from(value: &TwoDimsMetrics<Idx, Size>) -> Self {
        [value.dim_0, value.dim_1]
    }
}

impl <const Idx:usize, const Size:usize> From<[i64; 2]> for TwoDimsMetrics<Idx, Size> {
    fn from([dim_0, dim_1]: [i64; 2]) -> Self {
        Self { dim_0, dim_1 }
    }
}

impl <const Idx:usize, const Size:usize> From<&[i64; 2]> for TwoDimsMetrics<Idx, Size> {
    fn from([dim_0, dim_1]: &[i64; 2]) -> Self {
        Self {
            dim_0: *dim_0,
            dim_1: *dim_1,
        }
    }
}
