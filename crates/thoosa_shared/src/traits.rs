use ambassador::delegatable_trait;
use mockall::automock;
use std::any::Any;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Add, AddAssign, Deref, DivAssign, Range};
use std::sync::Arc;
use log::debug;
use crate::applicables_runner::Input;
use crate::{CopyToDevice, ThoError};
use uuid::Uuid;
use async_trait::async_trait;
use parking_lot::RwLock;
use thoosa_dummy_data_derive::DummyData;
use crate::copy_mode::{CopyMode, CpuOnly};

#[delegatable_trait]
pub trait CachableIdTr: Any + Debug + Send + Sync {

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
pub trait Named {// like Type name
    fn name(&self) -> &'static str {
        "n/a"
    }
}
///////////////////////////////////
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct SimpleId(
    #[cfg(debug_assertions)]
    pub Cow<'static, str>,

);

#[macro_export]
macro_rules! simple_id {
    ($name:literal $(,)?) => {{
        #[cfg(debug_assertions)]
        {
            thoosa_shared::traits::SimpleId(
                std::borrow::Cow::Borrowed($name)
            )
        }
        #[cfg(not(debug_assertions))]
        thoosa_shared::traits::SimpleId{}
    }};
    ($name:expr $(,)?) => {{
         #[cfg(debug_assertions)]
        {
            thoosa_shared::traits::SimpleId(
                ::std::borrow::Cow::Owned(::std::string::String::from($name))
            )
        }
        #[cfg(not(debug_assertions))]
        thoosa_shared::traits::SimpleId{}
    }}
}

unsafe impl Send for SimpleId {}
unsafe impl Sync for SimpleId {}

impl Named for SimpleId {
    fn name(&self) -> &'static str {
        "SimpleId"
    }
}

#[cfg(debug_assertions)]
impl From<&'static str> for SimpleId {
    fn from(value: &'static str) -> Self {
        SimpleId(value.into())
    }
}

#[cfg(debug_assertions)]
impl From<String> for SimpleId {
    fn from(value: String) -> Self {
        simple_id!(&value)
    }
}
#[cfg(debug_assertions)]
impl Display for SimpleId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {

        write!(f, "{}", self.0)

    }
}
#[cfg(not(debug_assertions))]
impl Display for SimpleId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "n/a")
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
#[derive(Debug, Eq, PartialEq, Hash, Clone, DummyData)]
#[dummy_data(clone)]
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



//TODO should be needed only for cachable, remove from Output super trait
#[delegatable_trait]
#[automock(type ID=DummyCachableId;)]
pub trait WithId {
    type ID: CachableIdTr + Send + Sync + Eq + Hash + Clone + Debug;

    fn id(&self) -> &Self::ID;
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct DummyCachableId(pub &'static str);

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

    fn cap(&self, _other:&mut  Self)  {
        todo!()
    }

    fn num_elts(&self)->usize{
        0
    }
}

impl Metrics for () {} //for stages without output

impl Metrics for [usize; 1] {
    fn num_elts(&self) -> usize {
        self[0]
    }
}

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

pub struct Assert<const COND: bool>;
pub trait IsTrue {}
impl IsTrue for Assert<true> {}

//for resources batch vec in ClcResourcesMotifsDistInputDeviceWrapper
impl Metrics for Range<i64> {}

///For adjusting one of tensor's dimensions
///Idx - dimension to split, Size - number of dimensions
#[derive(Debug, Clone)]
pub struct MonoMetrics<const IDX:usize, const SIZE:usize> {
     dims: [i64; SIZE]
}

impl <const IDX:usize, const SIZE:usize> Metrics for MonoMetrics<IDX, SIZE> {
    fn default_end(&self) -> Self {
        MonoMetrics {
            dims: self.dims.iter().enumerate().map(|(i, v)|{
                if i == IDX {
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

    fn num_elts(&self) -> usize {
        self.dims.iter().fold(1, |a, v| a * (*v as usize))
    }
}

//TODO remove it?
impl <const IDX:usize, const SIZE:usize> Default for MonoMetrics<IDX, SIZE> {
    fn default() -> Self {
        Self {
            dims: [0; SIZE]
        }
    }
}

impl <const IDX:usize, const SIZE:usize> MonoMetrics<IDX, SIZE> {

    //TODO remove - replace with 'effective dim Idx and 'others'
    pub fn dim_0(&self) -> i64 {
        self.dims[0]
    }

    //TODO remove
    pub fn dim_1(&self) -> i64 {
        self.dims[1]
    }

    pub fn dim<const DIM: usize>(&self) -> i64
    where
        Assert<{ DIM < SIZE }>: IsTrue,
    {
        self.dims[DIM]
    }
    pub fn map_to_ranges(start:&Self, end:&Self) ->Result<[Range<i64>; SIZE], ThoError> {

        start.dims
            .iter()
            .zip(end.dims.iter())
            .enumerate()
            .map(|(i, (start, end))| {
                if i == IDX {
                    *start..*end
                } else {
                    0..*end
                }

            })
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| ThoError::new("can't map metrics to ranges: length mismatch"))
    }

}

impl <const IDX:usize, const SIZE:usize> Add<&MonoMetrics<IDX, SIZE>> for &MonoMetrics<IDX, SIZE> {
    type Output = MonoMetrics<IDX, SIZE>;

    fn add(self, rhs: &MonoMetrics<IDX, SIZE>) -> Self::Output {
        let mut new_dims = self.dims;
        new_dims[IDX] = new_dims[IDX] + rhs.dims[IDX];

        MonoMetrics {
            dims: new_dims,
        }
    }
}

impl <const IDX:usize, const SIZE:usize> DivAssign<i64> for MonoMetrics<IDX, SIZE> {
    fn div_assign(&mut self, rhs: i64) {
        self.dims[IDX] /= rhs;
    }
}

impl <const IDX:usize, const SIZE:usize> PartialEq<i64> for MonoMetrics<IDX, SIZE> {
    fn eq(&self, other: &i64) -> bool {
        self.dims[IDX].eq(other)
    }
}

impl <const IDX:usize, const SIZE:usize> PartialOrd<i64> for MonoMetrics<IDX, SIZE> {
    fn partial_cmp(&self, other: &i64) -> Option<Ordering> {
        self.dims[IDX].partial_cmp(other)
    }
}

impl <const IDX:usize, const SIZE:usize> PartialEq<MonoMetrics<IDX, SIZE>> for MonoMetrics<IDX, SIZE> {
    fn eq(&self, other: &MonoMetrics<IDX, SIZE>) -> bool {
        self.dims[IDX].eq(&other.dims[IDX])
    }
}

impl <const IDX:usize, const SIZE:usize> PartialOrd<MonoMetrics<IDX, SIZE>> for MonoMetrics<IDX, SIZE> {
    fn partial_cmp(&self, other: &MonoMetrics<IDX, SIZE>) -> Option<Ordering> {
        self.dims[IDX].partial_cmp(&other.dims[IDX])
    }
}

impl<'a, const IDX:usize, const SIZE:usize> AddAssign<&'a MonoMetrics<IDX, SIZE>> for MonoMetrics<IDX, SIZE> {
    fn add_assign(&mut self, rhs: &'a MonoMetrics<IDX, SIZE>) {
        self.dims[IDX] += rhs.dims[IDX];

        debug!("AddAssign for LoadedTensorMetrics::add_assign rhs.dim_0={} self.dim_0={}",rhs.dims[IDX], self.dims[IDX]);
    }
}

impl<const IDX:usize, const SIZE:usize> AddAssign<MonoMetrics<IDX, SIZE>> for MonoMetrics<IDX, SIZE> {
    fn add_assign(&mut self, rhs: MonoMetrics<IDX, SIZE>) {
        self.dims[IDX] += rhs.dims[IDX];
    }
}

impl <const IDX:usize, const SIZE:usize> TryFrom<Vec<i64>> for MonoMetrics<IDX, SIZE> {
    type Error = ThoError;

    fn try_from(value: Vec<i64>) -> Result<Self, Self::Error> {
        assert!(SIZE > 0);
        assert_eq!(value.len(), SIZE);

        Ok(Self { dims: value.clone().try_into().expect(&format!("can't convert {value:?} to dims")) })//TODO avoid value.clone
    }
}

impl <const IDX:usize, const SIZE:usize> From<&MonoMetrics<IDX, SIZE>> for [i64; 2] {
    fn from(value: &MonoMetrics<IDX, SIZE>) -> Self {
        value.dims.as_slice().try_into().expect(&format!("can't convert {value:?} to [i64; 2]"))
    }
}

impl <const IDX:usize, const SIZE:usize> From<[i64; 2]> for MonoMetrics<IDX, SIZE> {
    fn from(value: [i64; 2]) -> Self {
        assert_eq!(2, SIZE);

        let mut me = Self::default();
        me.dims[0] = value[0];
        me.dims[1] = value[1];
        me
    }
}

impl <const IDX:usize, const SIZE:usize> From<&[i64; 2]> for MonoMetrics<IDX, SIZE> {
    fn from(value: &[i64; 2]) -> Self {
        assert_eq!(2, SIZE);

        let mut me = Self::default();
        me.dims[0] = value[0];
        me.dims[1] = value[1];
        me
    }
}

impl <const IDX:usize, const SIZE:usize> From<[i64; 3]> for MonoMetrics<IDX, SIZE> {
    fn from(value: [i64; 3]) -> Self {
        assert_eq!(3, SIZE);

        let mut me = Self::default();
        me.dims[0] = value[0];
        me.dims[1] = value[1];
        me.dims[2] = value[2];
        me
    }
}


impl From<usize> for MonoMetrics<0, 1> {
    fn from(value: usize) -> Self {
        MonoMetrics {
            dims: [value as i64],
        }
    }
}

#[async_trait]
#[delegatable_trait]
#[automock(type D=MockCopyToDevice; type Metrics=[usize;1]; type M=CpuOnly<Self>; type Sub=Self; type SubSelect=Range<<Self as Data>::Metrics>;)]
pub trait Data: Debug {
    type Metrics: Metrics + Debug + Clone + Send + Sync + Default;

    type M:CopyMode;

    fn size(&self) -> Self::Metrics;

    fn track_used_in(&self, _token: &'static str) {}

    fn device(&self) -> usize;

    fn mem_size(&self) -> u64 {
        // let x:CpuOnly<Self> = CpuOnly::new();
        0
    }
}

impl WithId for MockData {
    type ID = ();

    fn id(&self) -> &Self::ID {
        todo!()
    }
}

impl <D:Data + ?Sized> Data for &D {
    type Metrics = D::Metrics;
    type M = D::M;

    fn size(&self) -> Self::Metrics {
        <D as Data>::size(*self)
    }

    fn track_used_in(&self, token: &'static str) {
        <D as Data>::track_used_in(*self, token)
    }

    fn device(&self) -> usize {
        <D as Data>::device(*self)
    }

    fn mem_size(&self) -> u64 {
        <D as Data>::mem_size(*self)
    }


}

impl <D> Data for Arc<D>
where
    D:Data
{
    type Metrics = D::Metrics;
    type M = D::M;

    fn size(&self) -> Self::Metrics {
        self.as_ref().size()
    }

    fn device(&self) -> usize {
        self.as_ref().device()
    }
}

impl Data for (){
    type Metrics = ();
    type M = CpuOnly<()>;

    fn size(&self) -> Self::Metrics {
        ()
    }

    fn device(&self) -> usize {
        0
    }
}


#[async_trait]
impl CopyToDevice for MockData {
    async fn copy_to_device(&self, _dst_device: usize) -> Result<Self, ThoError> {
        Ok(MockData::new())
    }
}

#[delegatable_trait]
pub trait Select<S> {
    type Output; //: CopyToDevice;

    fn subview(&self, selector: S) -> Result<Self::Output, ThoError>;
}

pub trait Adjustable: Send + Sync {}

impl Adjustable for (){}

impl<A> Adjustable for Arc<RwLock<A>> where A: Adjustable {}

impl<T: Send + Sync> Adjustable for std::ops::Range<T> {}
