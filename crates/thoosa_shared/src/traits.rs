use crate::ThoError;
use crate::applicables_runner::{Input, Output};
use ambassador::delegatable_trait;
use async_trait::async_trait;
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use log::debug;
use mockall::automock;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::any::Any;
#[cfg(any(debug_assertions, tho_debug))]
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::{Add, AddAssign, Deref, DivAssign, Range};
use std::path::PathBuf;
use std::sync::Arc;
use thoosa_cpu_data_derive::CpuData;
use thoosa_named_derive::Named;
use tokio::sync::MutexGuard;
use uuid::Uuid;
use crate::ipc_types::{WorkerOutput, WorkerTensorInfoPayload};

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

impl Named for String {

}


//triggers blanked impl<T: ... Named + 'static> CachableId for T
#[automock]
pub trait Named {// like Type name
    fn name() -> &'static str {
        "n/a"
    }
}
///////////////////////////////////
///this should be used when id is only needed for debug. In release build is empty struct. TODO - make it static counter in release?
#[derive(Debug, Eq, PartialEq, Hash, Clone, Named, Serialize, Deserialize)]
pub struct DebugId(
    #[cfg(any(debug_assertions, tho_debug))]
    pub Cow<'static, str>,

);

#[macro_export]
macro_rules! debug_id {
    ($name:literal $(,)?) => {{
        #[cfg(any(debug_assertions, tho_debug))]
        {
            thoosa_shared::traits::DebugId(
                std::borrow::Cow::Borrowed($name)
            )
        }
        #[cfg(not(any(debug_assertions, tho_debug)))]
        thoosa_shared::traits::DebugId{}
    }};
    ($name:expr $(,)?) => {{
         #[cfg(any(debug_assertions, tho_debug))]
        {
            thoosa_shared::traits::DebugId(
                ::std::borrow::Cow::Owned(::std::string::String::from($name))
            )
        }
        #[cfg(not(any(debug_assertions, tho_debug)))]
        thoosa_shared::traits::DebugId{}
    }}
}

unsafe impl Send for DebugId {}
unsafe impl Sync for DebugId {}

#[cfg(any(debug_assertions, tho_debug))]
impl From<&'static str> for DebugId {
    fn from(value: &'static str) -> Self {
        DebugId(value.into())
    }
}

#[cfg(any(debug_assertions, tho_debug))]
impl From<String> for DebugId {
    fn from(value: String) -> Self {
        debug_id!(&value)
    }
}
#[cfg(any(debug_assertions, tho_debug))]
impl Display for DebugId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {

        write!(f, "{}", self.0)

    }
}
#[cfg(not(any(debug_assertions, tho_debug)))]
impl Display for DebugId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "n/a")
    }
}


///--------------------
#[derive(Debug, Eq, PartialEq, Hash, Clone, CpuData, Named)]
#[dummy_data(clone)]
pub struct DataId(pub u32);//for sql-store based stuff

unsafe impl Send for DataId {}
unsafe impl Sync for DataId {}

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

#[delegatable_trait]
#[automock(type ID=DummyCachableId;)]
pub trait WithId {
    type ID: CachableIdTr + Send + Sync + Eq + Hash + Clone + Debug + ?Sized;

    fn id(&self) -> &Self::ID;
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Named)]
#[name = "n/a"]
pub struct DummyCachableId(pub &'static str);

impl From<&'static str> for DummyCachableId {
    fn from(value: &'static str) -> Self {
        DummyCachableId(value)
    }
}


pub trait Metrics:Default {
    fn default_end(&self) -> Self {
        Default::default()
    }

    fn cap(&self, _other:&mut  Self)  {
        
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

impl<const IDX: usize, const SIZE: usize> Serialize for MonoMetrics<IDX, SIZE> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.dims.as_slice().serialize(serializer)
    }
}

impl<'de, const IDX: usize, const SIZE: usize> Deserialize<'de> for MonoMetrics<IDX, SIZE> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let dims = Vec::<i64>::deserialize(deserializer)?;
        let dims = dims.try_into().map_err(|dims: Vec<i64>| serde::de::Error::invalid_length(dims.len(), &"an array matching the metric size"))?;
        Ok(Self { dims })
    }
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

    //TODO remove
    pub fn dim_0(&self) -> i64 {
        self.dims[0]
    }

    //TODO remove
    pub fn dim_1(&self) -> i64 {
        self.dims[1]
    }
    //FIXME
    pub fn dim_2(&self) -> i64 {
        self.dims[2]
    }

    pub fn dim<const DIM: usize>(&self) -> i64
    where
        Assert<{ DIM < SIZE }>: IsTrue,
    {
        self.dims[DIM]
    }
    pub fn map_to_ranges(start:&Self, end:&Self) -> [Range<i64>; SIZE] {
        // Both metrics have exactly SIZE entries, so constructing the array directly avoids an impossible length error.
        std::array::from_fn(|i| {
            if i == IDX {
                start.dims[i]..end.dims[i]
            } else {
                0..end.dims[i]
            }
        })
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
        // assert_eq!(value.len(), SIZE);

        Ok(Self { dims: value.clone().try_into().unwrap_or( {

            (0..SIZE).map(|i|value.get(i).map(|v|*v).unwrap_or(0)).collect::<Vec<_>>().try_into().unwrap()
        }) })//TODO avoid value.clone
    }
}

impl <const IDX:usize, const SIZE:usize> From<&MonoMetrics<IDX, SIZE>> for [i64; 2] {
    fn from(value: &MonoMetrics<IDX, SIZE>) -> Self {
        value.dims.as_slice().try_into().expect(&format!("can't convert {value:?} to [i64; 2]"))
    }
}

impl <const IDX:usize, const SIZE:usize> From<[i64; 2]> for MonoMetrics<IDX, SIZE> {
    fn from(value: [i64; 2]) -> Self {
        // assert_eq!(2, SIZE);

        let mut me = Self::default();
        me.dims[0] = value[0];
        me.dims[1] = value[1];
        me
    }
}

impl <const IDX:usize, const SIZE:usize> From<&[i64; 2]> for MonoMetrics<IDX, SIZE> {
    fn from(value: &[i64; 2]) -> Self {
        // assert_eq!(2, SIZE);

        let mut me = Self::default();
        me.dims[0] = value[0];
        me.dims[1] = value[1];
        me
    }
}

impl <const IDX:usize, const SIZE:usize> From<[i64; 3]> for MonoMetrics<IDX, SIZE> {
    fn from(value: [i64; 3]) -> Self {
        // assert_eq!(3, SIZE);

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

    fn size(&self) -> Self::Metrics;

    fn tho_device_idx(&self) -> usize;

    fn mem_size(&self) -> u64;

}

impl <D:Data + ?Sized> Data for &D {
    type Metrics = D::Metrics;

    fn size(&self) -> Self::Metrics {
        <D as Data>::size(*self)
    }

    fn tho_device_idx(&self) -> usize {
        <D as Data>::tho_device_idx(*self)
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

    fn size(&self) -> Self::Metrics {
        self.as_ref().size()
    }

    fn tho_device_idx(&self) -> usize {
        self.as_ref().tho_device_idx()
    }

    fn mem_size(&self) -> u64 {
        Arc::as_ref(self).mem_size()
    }
}

impl Data for (){
    type Metrics = ();

    fn size(&self) -> Self::Metrics {
        ()
    }

    fn tho_device_idx(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        0
    }
}

#[typetag::serde(tag = "type")]
pub trait ThoSelector:Debug + Send + Sync {
    fn clone_box(&self)->Box<dyn ThoSelector>;

    fn tensor_ranges(&self) -> Result<Vec<Range<i64>>, ThoError> {
        Err(ThoError::new(format!("selector {self:?} cannot be applied directly to a tensor")))
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct RangeSelector(pub Range<[i64; 1]>);

impl Adjustable for RangeSelector {}

impl Deref for RangeSelector {
    type Target = Range<[i64; 1]>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[typetag::serde]
impl ThoSelector for RangeSelector {
    fn clone_box(&self) -> Box<dyn ThoSelector> {
        Box::new(RangeSelector(self.0.clone()))
    }

    fn tensor_ranges(&self) -> Result<Vec<Range<i64>>, ThoError> {
        Ok(vec![self.0.start[0]..self.0.end[0]])
    }
}

#[typetag::serde(name = "UnitSelector")]
impl ThoSelector for () {
    fn clone_box(&self) -> Box<dyn ThoSelector> {
        Box::new(())
    }
}

#[typetag::serde(name = "I64RangeSelector")]
impl ThoSelector for Range<i64> {
    fn clone_box(&self) -> Box<dyn ThoSelector> {
        Box::new(self.clone())
    }

    fn tensor_ranges(&self) -> Result<Vec<Range<i64>>, ThoError> {
        Ok(vec![self.clone()])
    }
}


#[delegatable_trait]
pub trait Select<S>
where S:ThoSelector
{
    type Output;

    fn subview(&self, selector: S) -> Result<Self::Output, ThoError>;
}

pub trait Adjustable: Send + Sync {}

impl Adjustable for (){}

impl<A> Adjustable for Arc<RwLock<A>> where A: Adjustable {}

impl<T: Send + Sync> Adjustable for std::ops::Range<T> {}


impl <A:Adjustable> Adjustable for Arc<A> {}

/// steepness: larger values mean a steeper slope.
/// flatness: larger values mean a flatter slope.
///https://www.wolframalpha.com/input?i=plot+1%2F%281%2Babs%280.2*%28x-10%29%29%5E5%29
pub fn hump(x: f64, center: f64, steepness: f64, flatness: f64) -> f64 {
    assert!(steepness > 0.0);
    assert!(flatness > 0.0);
    1.0 / (1.0 + (steepness * (x - center)).abs().powf(flatness))
}

pub trait FromJsonFile {
    fn from_file(path: &PathBuf) -> anyhow::Result<Self>
        where Self:Sized;
}

pub trait ToJsonFile:Debug {
    fn to_file(&self, path: &PathBuf) -> anyhow::Result<()>;
}

#[delegatable_trait]
pub trait MaxParallelCtx: Named + Debug + Send + Sync {
    fn set_max_other_parallel(&mut self, permit: u8);//u8 - MaxParallel

    fn max_other_parallel(&self) -> u8;
}

pub trait RunCtx: WithId + Debug + Send + Sync {
    fn tho_device_idx(&self) -> usize;

    fn branch(&mut self, with_new_cache_area: bool) -> Self
    where
        Self: Sized;

    fn add_purge_gpu_closure(&mut self, cl: Box<dyn Fn() -> BoxFuture<'static, Result<(), ThoError>> + Send + Sync>);

}

//it's
pub trait OutId: Stage
where
        for<'a> <Self as Stage>::In<'a>: WithId,
        <Self as Stage>::Out: WithId,
{
    fn out_id<'a>(
        &self,
        input_id: &<<Self as Stage>::In<'a> as WithId>::ID,
    ) -> <Self::Out as WithId>::ID;

}

pub type StageHash = [u8; 32];
/// a Stage is just a function with explicitly defined state, execution context and input so that all three can be adjusted to handle OOM
/// see ['thoosa_common::runner::oom_handler::Adjusters'] enum
pub trait Stage: Debug {
    type Ctx: RunCtx;

    type In<'a>: Input;

    type Out: Output;

}

impl <S> Stage for Arc<S>
where
S:Stage{
    type Ctx = S::Ctx;
    type In<'a> = S::In<'a>;
    type Out = S::Out;
}

pub trait AsyncStage:Stage + Debug + Send + Sync + 'static
where
        for <'a> Self::In<'a>:  Send + Sync,
        Self::Out: Send + Sync,
{


    //the DevicesWrap is in input and output so that data could be dumped to CPU at any time (if other stage wins 100 GPU)
    //plain inputs can override RunInput to avoid DevicesWrap. Tensor-holding inputs should keep the default so they can be dumped/loaded by device.
    // Returning DevicesWrap also forces callers to specify the device to load data from, keeping the logic clear and transparent (as opposed to implicitly taking the device from the stage context).
    fn run(self: Arc<Self>, _input: Self::In<'static>, _ctx: Self::Ctx) -> impl Future<Output = Result<Self::Out, ThoError>> + Send

    {
        async {
            unimplemented!()
        }
    }
}


pub trait SyncStage: Stage + Debug {


    fn as_with_state(&self) -> Option<&dyn ErasedStageWithState<Self>>
    where
        Self: Sized,
    {
        None
    }

    fn do_run<'f>(self: Arc<Self>, _input: &Self::In<'f>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {//TODO remove _ctx - it's used for logging and in tests
        unimplemented!()
    }

    fn consume<'f>(
        self: Arc<Self>,
        _input: Self::In<'f>,
        _ctx: Self::Ctx,
    ) -> ConsumeResult<Self::In<'f>, Self::Out> {
        unimplemented!()
    }
}

pub trait StageWithState: SyncStage + Sync {
    type S: Debug + Send + Sync;

    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<MutexGuard<'_, Self::S>, ThoError>>;
    fn get_state_info(&self, tho_device: usize) -> BoxFuture<'_, Result<StageStateInfo, ThoError>> {
        async {
            Ok(StageStateInfo::empty())
        }.boxed()
    }

    fn do_run_with_state<'f>(self: Arc<Self>, _input: &Self::In<'f>, _state: MutexGuard<'_, Self::S>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        unimplemented!()
    }
}

pub trait ErasedStageState<S>: Send
where
    S: SyncStage,
{
    fn do_run_with_state<'f>(self: Box<Self>, stage: Arc<S>, input: &S::In<'f>, ctx: &S::Ctx) -> Result<S::Out, ThoError>;
}

//for switching between ipc proxy and local
pub trait DynSyncStage<I, O, C>: Debug + Send + Sync + 'static
where
    I: Input /* + WithId*/ + 'static,
    O: Output/* + WithId*/ + 'static,
    C: RunCtx + 'static,
{
    fn do_run_dyn(self: Arc<Self>, input: &I, ctx: &C) -> Result<O, ThoError>;


}

pub trait DynOutId<I, O>: Debug + Send + Sync + 'static
where
    I: Input + WithId + 'static,
    O: Output + WithId + 'static
{
    fn out_id_dyn(&self, input_id: &I::ID) -> O::ID;
}

pub trait DynSyncStageOutId<I, O, C>:DynSyncStage<I, O, C> + DynOutId<I, O>
where
    I: Input + WithId + 'static,
    O: Output + WithId + 'static,
    C: RunCtx + 'static
{

}

impl <I, O, C, T> DynSyncStageOutId<I,O,C> for T
where
    I: Input + WithId + 'static,
    O: Output + WithId + 'static,
    C: RunCtx + 'static,
    T: DynSyncStage<I,O,C> + DynOutId<I, O>
{

}

impl<T, I, O, C> DynSyncStage<I, O, C> for T
where
    T: SyncStage + Send + Sync + 'static,
    for<'a> T: Stage<In<'a> = I, Out = O, Ctx = C>,
    I: Input + WithId + 'static,
    O: Output + WithId + 'static,
    C: RunCtx + 'static,
{
    fn do_run_dyn(self: Arc<Self>, input: &I, ctx: &C) -> Result<O, ThoError> {
        self.do_run(input, ctx)
    }


}

impl<T, I, O> DynOutId<I, O> for T
where
    for<'a> T: Stage<In<'a> = I, Out = O>,
    T: OutId + Send + Sync + 'static,
    I: Input + WithId + 'static,
    O: Output + WithId + 'static,
{

    fn out_id_dyn(&self, input_id: &I::ID) -> O::ID {
        self.out_id(input_id)
    }
}


#[derive(Debug)]
pub struct DynSyncStageBox<I, O, C>
where
    I: Input + WithId + 'static,
    O: Output + WithId + 'static,
    C: RunCtx + 'static,
{
    inner: Arc<dyn DynSyncStageOutId<I, O, C>>,
}

impl<I, O, C> DynSyncStageBox<I, O, C>
where
    I: Input + WithId + 'static,
    O: Output + WithId + 'static,
    C: RunCtx + 'static,
{
    pub fn new(inner: Arc<dyn DynSyncStageOutId<I, O, C>>) -> Self {
        Self { inner }
    }
}

impl<I, O, C> Stage for DynSyncStageBox<I, O, C>
where
    I: Input + WithId + 'static,
    O: Output + WithId + 'static,
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = I;
    type Out = O;
}

impl<I, O, C> SyncStage for DynSyncStageBox<I, O, C>
where
    I: Input + WithId + 'static,
    O: Output + WithId + 'static,
    C: RunCtx + 'static,
{
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        Arc::clone(&self.inner).do_run_dyn(input, ctx)
    }
}

impl<I, O, C> OutId for DynSyncStageBox<I, O, C>
where
    I: Input + WithId + 'static,
    O: Output + WithId + 'static,
    C: RunCtx + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        self.inner.out_id_dyn(input_id)
    }
}

impl<I, O, C> Adjustable for DynSyncStageBox<I, O, C>
where
    I: Input + WithId + 'static,
    O: Output + WithId + 'static,
    C: RunCtx + 'static,
{
}

#[derive(Debug)]
pub enum ConsumeResult<I, O> {
    Failed{input:I, error:ThoError},
    Ok(O)
}

///for keeping stages with different state types in the CudaWorker's installed stages map
pub trait ErasedStageWithState<S>
where
    S: SyncStage,
{
    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<Box<dyn ErasedStageState<S> + '_>, ThoError>>;
}

impl<S> ErasedStageWithState<S> for S
where
    S: StageWithState,
{
    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<Box<dyn ErasedStageState<S> + '_>, ThoError>> {
        async move {
            let state = StageWithState::pre_run(self, to_device_idx).await?;
            // Keep the concrete guard type inside the erased value while it spans later async preparation.
            Ok(Box::new(StageStateGuard::<S> { state }) as Box<dyn ErasedStageState<S>>)
        }.boxed()
    }
}


struct StageStateGuard<'a, S>
where
    S: StageWithState,
{
    state: MutexGuard<'a, S::S>,
}

impl<S> ErasedStageState<S> for StageStateGuard<'_, S>
where
    S: StageWithState,
{
    fn do_run_with_state<'f>(self: Box<Self>, stage: Arc<S>, input: &S::In<'f>, ctx: &S::Ctx) -> Result<S::Out, ThoError> {
        let Self { state } = *self;
        StageWithState::do_run_with_state(stage, input, state, ctx)
    }
}



#[derive(Debug, Serialize, Deserialize)]
pub struct StageStateInfo {
    tensors:HashMap<String, WorkerTensorInfoPayload>
}

impl StageStateInfo {
    pub fn new(tensors: HashMap<String, WorkerTensorInfoPayload>) -> Self {
        Self { tensors }
    }

    pub fn empty() -> Self {
        Self { tensors:HashMap::new() }
    }
}

#[typetag::serde]
impl WorkerOutput for StageStateInfo {

}
