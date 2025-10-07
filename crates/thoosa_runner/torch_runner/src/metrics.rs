use log::debug;
use std::cmp::Ordering;
use std::ops::{Add, AddAssign, DivAssign};
use thoosa_shared::{MyError, ThoosaError};

#[derive(Debug, Clone, Default)]
pub struct TwoDimsMetrics<const I:usize> {//TODO rename to SingleDimMetrics and use I to detect which dim to act on
     dim_0: i64,
     dim_1: i64,
}

impl <const I:usize> TwoDimsMetrics<I> {
    pub fn dim_0(&self) -> i64 {
        self.dim_0
    }

    pub fn dim_1(&self) -> i64 {
        self.dim_1
    }
}

impl <const I:usize> Add<&TwoDimsMetrics<I>> for &TwoDimsMetrics<I> {
    type Output = TwoDimsMetrics<I>;

    fn add(self, rhs: &TwoDimsMetrics<I>) -> Self::Output {
        TwoDimsMetrics {
            dim_0: self.dim_0 + rhs.dim_0,
            dim_1: self.dim_1,
        }
    }
}

impl <const I:usize> DivAssign<i64> for TwoDimsMetrics<I> {
    fn div_assign(&mut self, rhs: i64) {
        self.dim_0 /= rhs;
    }
}

impl <const I:usize> PartialEq<i64> for TwoDimsMetrics<I> {
    fn eq(&self, other: &i64) -> bool {
        self.dim_0.eq(other)
    }
}

impl <const I:usize> PartialOrd<i64> for TwoDimsMetrics<I> {
    fn partial_cmp(&self, other: &i64) -> Option<Ordering> {
        self.dim_0.partial_cmp(other)
    }
}

impl <const I:usize> PartialEq<TwoDimsMetrics<I>> for TwoDimsMetrics<I> {
    fn eq(&self, other: &TwoDimsMetrics<I>) -> bool {
        self.dim_0.eq(&other.dim_0)
    }
}

impl <const I:usize> PartialOrd<TwoDimsMetrics<I>> for TwoDimsMetrics<I> {
    fn partial_cmp(&self, other: &TwoDimsMetrics<I>) -> Option<Ordering> {
        self.dim_0.partial_cmp(&other.dim_0)
    }
}

impl<'a, const I:usize> AddAssign<&'a TwoDimsMetrics<I>> for TwoDimsMetrics<I> {
    fn add_assign(&mut self, rhs: &'a TwoDimsMetrics<I>) {
        self.dim_0 += rhs.dim_0;
        debug!(
            "AddAssign for LoadedTensorMetrics::add_assign rhs.dim_0={} self.dim_0={}",
            rhs.dim_0, self.dim_0
        );
        // self.dim_1+=rhs.dim_1;TODO dimension
    }
}

impl <const I:usize> TryFrom<Vec<i64>> for TwoDimsMetrics<I> {
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

impl <const I:usize> From<&TwoDimsMetrics<I>> for [i64; 2] {
    fn from(value: &TwoDimsMetrics<I>) -> Self {
        [value.dim_0, value.dim_1]
    }
}

impl <const I:usize> From<[i64; 2]> for TwoDimsMetrics<I> {
    fn from([dim_0, dim_1]: [i64; 2]) -> Self {
        Self { dim_0, dim_1 }
    }
}

impl <const I:usize> From<&[i64; 2]> for TwoDimsMetrics<I> {
    fn from([dim_0, dim_1]: &[i64; 2]) -> Self {
        Self {
            dim_0: *dim_0,
            dim_1: *dim_1,
        }
    }
}
