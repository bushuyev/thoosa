use ambassador::delegatable_trait;
use mockall::automock;
use std::any::Any;
use std::cmp::Ordering;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Add, AddAssign, Deref, DivAssign, Range};
use log::debug;
use uuid::uuid;
use crate::applicables_runner::{CachableId, Input};
use crate::ThoosaError;
use uuid::Uuid;

#[delegatable_trait]
pub trait CachableIdTr: Any + Debug + Send + Sync {
    fn eqq(&self, _other: &dyn CachableIdTr) -> bool {
        false
    }

    fn as_any<'a>(&'a self) -> &'a (dyn Any + Send + Sync);
    fn eq_dyn(&self, other: &dyn CachableIdTr) -> bool;
    fn hash_dyn(&self, state: &mut dyn Hasher);
}

pub mod mock_cachable_id {
    use crate::traits::CachableIdTr;
    use mockall::mock;
    use std::any::Any;
    use std::hash::{Hash, Hasher};

    mock! {
        #[derive(Debug, Clone, Eq, PartialEq, Hash)]
        pub CachableId {}

        impl CachableIdTr for CachableId {
            fn eqq(&self, other:&dyn CachableIdTr) -> bool;
            #[allow(unused_parens)]
            fn as_any<'a>(&'a self) ->  (&'a (dyn Any + Send + Sync));
            fn eq_dyn(&self, other: &dyn CachableIdTr) -> bool;
            fn hash_dyn(&self, state: &mut dyn Hasher);
        }

        impl Clone for CachableId {
            fn clone(&self) -> Self;
            fn clone_from(&mut self, source: &Self);
        }


    }

    impl Hash for MockCachableId {
        fn hash<H: Hasher>(&self, _state: &mut H) {}
    }

    impl Eq for MockCachableId {}

    impl PartialEq for MockCachableId {
        fn eq(&self, _other: &Self) -> bool {
            false
        }
    }
}

// dyn Key itself: Eq/Hash via the object-safe methods
impl PartialEq for dyn CachableIdTr {
    fn eq(&self, other: &Self) -> bool {
        self.eq_dyn(other)
    }
}

impl Eq for dyn CachableIdTr {}

impl Hash for dyn CachableIdTr {
    fn hash<H: Hasher>(&self, h: &mut H) {
        struct Erased<'h, H2: Hasher>(&'h mut H2);
        impl<'h, H2: Hasher> Hasher for Erased<'h, H2> {
            fn finish(&self) -> u64 {
                self.0.finish()
            }
            fn write(&mut self, b: &[u8]) {
                self.0.write(b)
            }
        }
        let mut e = Erased(h);
        self.hash_dyn(&mut e);
    }
}

impl<T: Send + Sync + Debug + Eq + Hash + Named + 'static> CachableIdTr for T {
    #[allow(unused_parens)]
    fn as_any<'a>(&'a self) -> (&'a (dyn Any + Send + Sync)) {
        self
    }

    fn eq_dyn(&self, other: &dyn CachableIdTr) -> bool {
        other
            .as_any()
            .downcast_ref::<T>()
            .map_or(false, |v| self == v)
    }

    fn hash_dyn(&self, state: &mut dyn Hasher) {
        struct Erased<'h>(&'h mut dyn Hasher);
        impl<'h> Hasher for Erased<'h> {
            fn finish(&self) -> u64 {
                self.0.finish()
            }
            fn write(&mut self, b: &[u8]) {
                self.0.write(b)
            }
        }
        let mut e = Erased(state);
        self.hash(&mut e);
    }
}

impl dyn CachableIdTr {
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }
}

//triggers blanked impl<T: ... Named + 'static> CachableId for T
#[automock]
pub trait Named {
    fn name(&self) -> &'static str {
        "n/a"
    }
}
///////////////////////////////////
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct SimpleId(pub &'static str);

unsafe impl Send for SimpleId {}
unsafe impl Sync for SimpleId {}



impl Named for SimpleId {
    fn name(&self) -> &'static str {
        "SimpleId"
    }
}

impl From<&'static str> for SimpleId {
    fn from(value: &'static str) -> Self {
        SimpleId(value)
    }
}

impl From<String> for SimpleId {
    fn from(value: String) -> Self {
        SimpleId(value.leak())
    }
}

impl Display for SimpleId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
///////////////////////////////////
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct TmpId(Uuid);


impl TmpId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}
unsafe impl Send for TmpId {}
unsafe impl Sync for TmpId {}



impl Named for TmpId {
    fn name(&self) -> &'static str {
        "TmpId"
    }
}



impl Display for TmpId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
///--------------------
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct DataId(pub u32);//for sql-store based stuff

unsafe impl Send for DataId {}
unsafe impl Sync for DataId {}


impl Named for DataId {
    fn name(&self) -> &'static str {
        "DataId"
    }
}

impl WithId for DataId {
    type ID = Self;

    fn id(&self) -> &Self::ID {
        self
    }
}

impl Input for DataId {}

impl From<&u32> for DataId {
    fn from(value: &u32) -> Self {
        DataId(*value)
    }
}

impl From<u32> for DataId {
    fn from(value: u32) -> Self {
        DataId(value)
    }
}

impl Deref for DataId {
    type Target = u32;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Display for DataId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
///////////////////////////////////




#[delegatable_trait]
#[automock(type ID=DummyCachableId;)]
pub trait WithId {
    type ID: CachableIdTr + Send + Sync + Eq + Hash + Clone + Debug;

    fn id(&self) -> &Self::ID;
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct DummyCachableId(&'static str);

impl Named for DummyCachableId {}

impl From<&'static str> for DummyCachableId {
    fn from(value: &'static str) -> Self {
        DummyCachableId(value)
    }
}


pub trait Metrics:Default {//TODO remove Default
    fn default_end(&self) -> Self {
        Default::default()
    }

    fn cap(&self, other:&mut  Self)  {
        todo!()
    }
}

impl Metrics for () {} //for stages without output

impl Metrics for [usize; 1] {}

//TODO  for tests, replace with MonoMetrics
impl Metrics for i64 {
    fn cap(&self, other: &mut Self) {
        if *other > *self {

            *other = self.clone();
        }
    }
}
impl Metrics for usize {
    fn cap(&self, other: &mut Self) {
        if *other > *self {

            *other = self.clone();
        }
    }

}

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

    fn cap(&self, other:&mut  Self)  {
        for (my_dim, other_dim) in self.dims.iter().zip(other.dims.iter_mut()) {
            if my_dim < other_dim {
                *other_dim = *my_dim;
            }
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