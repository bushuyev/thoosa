use log::debug;
use std::cmp::Ordering;
use std::ops::{Add, AddAssign, DivAssign};
use thoosa_shared::{MyError, ThoosaError};

#[derive(Debug, Clone)]
pub struct TwoDimsMetrics<const Idx:usize, const Size:usize> {//TODO rename to SingleDimMetrics and use I to detect which dim to act on
     dims: [i64; Size]
}

//TODO remove it?
impl <const Idx:usize, const Size:usize> Default for TwoDimsMetrics<Idx, Size> {
    fn default() -> Self {
        Self {
            dims: [0; Size]
        }
    }
}

impl <const Idx:usize, const Size:usize> TwoDimsMetrics<Idx, Size> {

    //TODO remove
    pub fn dim_0(&self) -> i64 {
        self.dims[0]
    }

    //TODO remove
    pub fn dim_1(&self) -> i64 {
        self.dims[1]
    }
}

impl <const Idx:usize, const Size:usize> Add<&TwoDimsMetrics<Idx, Size>> for &TwoDimsMetrics<Idx, Size> {
    type Output = TwoDimsMetrics<Idx, Size>;

    fn add(self, rhs: &TwoDimsMetrics<Idx, Size>) -> Self::Output {
        let mut new_dims = self.dims;
        new_dims[Idx] = new_dims[Idx] + rhs.dims[Idx];

        TwoDimsMetrics {
            dims: new_dims,
        }
    }
}

impl <const Idx:usize, const Size:usize> DivAssign<i64> for TwoDimsMetrics<Idx, Size> {
    fn div_assign(&mut self, rhs: i64) {
        self.dims[Idx] /= rhs;
    }
}

impl <const Idx:usize, const Size:usize> PartialEq<i64> for TwoDimsMetrics<Idx, Size> {
    fn eq(&self, other: &i64) -> bool {
        self.dims[Idx].eq(other)
    }
}

impl <const Idx:usize, const Size:usize> PartialOrd<i64> for TwoDimsMetrics<Idx, Size> {
    fn partial_cmp(&self, other: &i64) -> Option<Ordering> {
        self.dims[Idx].partial_cmp(other)
    }
}

impl <const Idx:usize, const Size:usize> PartialEq<TwoDimsMetrics<Idx, Size>> for TwoDimsMetrics<Idx, Size> {
    fn eq(&self, other: &TwoDimsMetrics<Idx, Size>) -> bool {
        self.dims[Idx].eq(&other.dims[Idx])
    }
}

impl <const Idx:usize, const Size:usize> PartialOrd<TwoDimsMetrics<Idx, Size>> for TwoDimsMetrics<Idx, Size> {
    fn partial_cmp(&self, other: &TwoDimsMetrics<Idx, Size>) -> Option<Ordering> {
        self.dims[Idx].partial_cmp(&other.dims[Idx])
    }
}

impl<'a, const Idx:usize, const Size:usize> AddAssign<&'a TwoDimsMetrics<Idx, Size>> for TwoDimsMetrics<Idx, Size> {
    fn add_assign(&mut self, rhs: &'a TwoDimsMetrics<Idx, Size>) {
        self.dims[Idx] += rhs.dims[Idx];

        debug!("AddAssign for LoadedTensorMetrics::add_assign rhs.dim_0={} self.dim_0={}",rhs.dims[Idx], self.dims[Idx]);
    }
}

impl <const Idx:usize, const Size:usize> TryFrom<Vec<i64>> for TwoDimsMetrics<Idx, Size> {
    type Error = ThoosaError;

    fn try_from(value: Vec<i64>) -> Result<Self, Self::Error> {
        assert!(Size > 1);
        assert_eq!(value.len(), Size);

        Ok(Self { dims: value.clone().try_into().expect(&format!("can't convert {value:?} to dims")) })//TODO avoid value.clone
    }
}

impl <const Idx:usize, const Size:usize> From<&TwoDimsMetrics<Idx, Size>> for [i64; 2] {
    fn from(value: &TwoDimsMetrics<Idx, Size>) -> Self {
        value.dims.as_slice().try_into().expect(&format!("can't convert {value:?} to [i64; 2]"))
    }
}

impl <const Idx:usize, const Size:usize> From<[i64; 2]> for TwoDimsMetrics<Idx, Size> {
    fn from(value: [i64; 2]) -> Self {
        assert_eq!(2, Size);

        let mut me = Self::default();
        me.dims[0] = value[0];
        me.dims[1] = value[1];
        me
    }
}

impl <const Idx:usize, const Size:usize> From<&[i64; 2]> for TwoDimsMetrics<Idx, Size> {
    fn from(value: &[i64; 2]) -> Self {
        assert_eq!(2, Size);

        let mut me = Self::default();
        me.dims[0] = value[0];
        me.dims[1] = value[1];
        me
    }
}
