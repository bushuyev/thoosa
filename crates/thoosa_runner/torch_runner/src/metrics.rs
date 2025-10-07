use log::debug;
use std::cmp::Ordering;
use std::ops::{Add, AddAssign, DivAssign, Range};
use thoosa_shared::{MyError, ThoosaError};
use thoosa_shared::traits::Metrics;


///For adjusting one of tensor's dimensions
#[derive(Debug, Clone)]
pub struct MonoMetrics<const Idx:usize, const Size:usize> {
     dims: [i64; Size]
}

impl <const Idx:usize, const Size:usize> Metrics for MonoMetrics<Idx, Size> {
    fn default_end(&self) -> Self {
        MonoMetrics {
            dims: self.dims.iter().enumerate().map(|(i, v)|{
                if i == Idx {
                    0
                } else {
                    *v
                }
            }).collect::<Vec<_>>().try_into().expect("could not convert array"),
        }
    }
}

//TODO remove it?
impl <const Idx:usize, const Size:usize> Default for MonoMetrics<Idx, Size> {
    fn default() -> Self {
        Self {
            dims: [0; Size]
        }
    }
}

impl <const Idx:usize, const Size:usize> MonoMetrics<Idx, Size> {

    //TODO remove
    pub fn dim_0(&self) -> i64 {
        self.dims[0]
    }

    //TODO remove
    pub fn dim_1(&self) -> i64 {
        self.dims[1]
    }

    pub fn map_to_ranges(start:&Self, end:&Self) ->Result<[Range<i64>; Size], ThoosaError> {

        start.dims
            .iter()
            .zip(end.dims.iter())
            .enumerate()
            .map(|(i, (start, end))| {
                if i == Idx {
                    *start..*end
                } else {
                    0..*end
                }

            })
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|e|ThoosaError::new(format!("can't map metrics to ranges: e={e:?}")))
    }

}

impl <const Idx:usize, const Size:usize> Add<&MonoMetrics<Idx, Size>> for &MonoMetrics<Idx, Size> {
    type Output = MonoMetrics<Idx, Size>;

    fn add(self, rhs: &MonoMetrics<Idx, Size>) -> Self::Output {
        let mut new_dims = self.dims;
        new_dims[Idx] = new_dims[Idx] + rhs.dims[Idx];

        MonoMetrics {
            dims: new_dims,
        }
    }
}

impl <const Idx:usize, const Size:usize> DivAssign<i64> for MonoMetrics<Idx, Size> {
    fn div_assign(&mut self, rhs: i64) {
        self.dims[Idx] /= rhs;
    }
}

impl <const Idx:usize, const Size:usize> PartialEq<i64> for MonoMetrics<Idx, Size> {
    fn eq(&self, other: &i64) -> bool {
        self.dims[Idx].eq(other)
    }
}

impl <const Idx:usize, const Size:usize> PartialOrd<i64> for MonoMetrics<Idx, Size> {
    fn partial_cmp(&self, other: &i64) -> Option<Ordering> {
        self.dims[Idx].partial_cmp(other)
    }
}

impl <const Idx:usize, const Size:usize> PartialEq<MonoMetrics<Idx, Size>> for MonoMetrics<Idx, Size> {
    fn eq(&self, other: &MonoMetrics<Idx, Size>) -> bool {
        self.dims[Idx].eq(&other.dims[Idx])
    }
}

impl <const Idx:usize, const Size:usize> PartialOrd<MonoMetrics<Idx, Size>> for MonoMetrics<Idx, Size> {
    fn partial_cmp(&self, other: &MonoMetrics<Idx, Size>) -> Option<Ordering> {
        self.dims[Idx].partial_cmp(&other.dims[Idx])
    }
}

impl<'a, const Idx:usize, const Size:usize> AddAssign<&'a MonoMetrics<Idx, Size>> for MonoMetrics<Idx, Size> {
    fn add_assign(&mut self, rhs: &'a MonoMetrics<Idx, Size>) {
        self.dims[Idx] += rhs.dims[Idx];

        debug!("AddAssign for LoadedTensorMetrics::add_assign rhs.dim_0={} self.dim_0={}",rhs.dims[Idx], self.dims[Idx]);
    }
}

impl <const Idx:usize, const Size:usize> TryFrom<Vec<i64>> for MonoMetrics<Idx, Size> {
    type Error = ThoosaError;

    fn try_from(value: Vec<i64>) -> Result<Self, Self::Error> {
        assert!(Size > 1);
        assert_eq!(value.len(), Size);

        Ok(Self { dims: value.clone().try_into().expect(&format!("can't convert {value:?} to dims")) })//TODO avoid value.clone
    }
}

impl <const Idx:usize, const Size:usize> From<&MonoMetrics<Idx, Size>> for [i64; 2] {
    fn from(value: &MonoMetrics<Idx, Size>) -> Self {
        value.dims.as_slice().try_into().expect(&format!("can't convert {value:?} to [i64; 2]"))
    }
}

impl <const Idx:usize, const Size:usize> From<[i64; 2]> for MonoMetrics<Idx, Size> {
    fn from(value: [i64; 2]) -> Self {
        assert_eq!(2, Size);

        let mut me = Self::default();
        me.dims[0] = value[0];
        me.dims[1] = value[1];
        me
    }
}

impl <const Idx:usize, const Size:usize> From<&[i64; 2]> for MonoMetrics<Idx, Size> {
    fn from(value: &[i64; 2]) -> Self {
        assert_eq!(2, Size);

        let mut me = Self::default();
        me.dims[0] = value[0];
        me.dims[1] = value[1];
        me
    }
}
