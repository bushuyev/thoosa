// removed unused feature gate to reduce warnings

use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::sync::Arc;
use ambassador::delegatable_trait;
use async_trait::async_trait;
use mockall::automock;
use parking_lot::RwLock;
use runnable::Runnable;
use crate::common::cachable::CachableId;
use crate::permits::{NumPermits, RunnerPerm};

pub mod runnable;
pub mod cachable;



#[async_trait]
#[delegatable_trait]
#[automock(type D=MockCopyToDevice; type Metrics=[usize;1]; type Sub=Self; type SubSelect=Range<<Self as Data>::Metrics>;)]
pub trait Data: Debug + CopyToDevice {
    type Metrics: Debug + Clone + Send + Sync + Default;


    type Sub:CopyToDevice;
    type SubSelect:Debug + Send + Sync;

    fn size(&self)-> Self::Metrics;

    fn subview(&self, _select: &Self::SubSelect) -> Self::Sub {
        unimplemented!()
    }

    fn track_used_in(&self, _token:&'static str){}

    fn device(&self)->usize;
}

#[async_trait]
impl CopyToDevice for MockData{
    async fn copy_to_device(&self, _dst_device: usize) -> Self {
        MockData::new()
    }
}


#[derive(Debug)]
pub enum DropIf {
    Always,
    IfNotMax
}


#[automock]
#[async_trait]
pub trait RunCtx: Named + Debug + Send + Sync {
    fn add_permit(&mut self, permit: RunnerPerm);

    fn drop_permits(&mut self, drop_if: DropIf);

    fn device_idx(&self) -> usize;

    fn acquired_permits(&self) -> NumPermits;

    async fn purge_gpu_caches(_ctx: Arc<RwLock<Self>>) -> Result<(), MyError>
    where
        Self: Sized,
    {
        Ok(())
    }

    fn cache_area(&self) -> u64;

}
impl Named for MockRunCtx {}

#[cfg(test)]
pub(crate) mod mock_run_ctx {
    use crate::permits::NumPermits;
    use super::{DropIf, MockRunCtx};

    pub struct MockRunCtxBuilder {
        mock:MockRunCtx
    }

    impl MockRunCtxBuilder {
        pub fn with_device_idx(mut self, out:usize) ->Self {
            self.mock.expect_device_idx().return_const(out);
            self
        }

        pub fn with_acquired_permits(mut self, out:NumPermits) ->Self {
            self.mock.expect_acquired_permits().return_const(out);
            self
        }

        pub fn with_drop_permits(mut self, inp:Vec<DropIf>) ->Self {
            for param in inp {
                self.mock.expect_drop_permits().call(param);
            }

            self
        }

        pub fn finish(self)->MockRunCtx {
            self.mock
        }
    }

    pub fn build() ->MockRunCtxBuilder {

        MockRunCtxBuilder {
            mock: super::MockRunCtx::new()
        }
    }
}


pub trait Input: WithId + Debug {}
pub trait Output: WithId + Debug {}



// #[cfg(test)]
// pub mod mock_input_output {
//     use mockall::*;
//
//     mock! {
//         #[derive(Debug)]
//         pub Input{}
//
//         impl super::WithId for Input {
//             type ID = crate::common::cachable::mock_cachable_id::MockCachableId;
//
//             fn id(&self) -> &crate::common::cachable::mock_cachable_id::MockCachableId;
//         }
//         impl super::Input for Input {
//
//         }
//
//         impl super::CopyToDevice for Input {}
//
//         impl super::Data for Input {
//             type Metrics = [usize;1];
//             type D:CopyToDevice = Self;
//             type Sub:CopyToDevice = Self;
//             type SubSelect = std::ops::Range<Self::Metrics>;
//
//         }
//     }
//
//
// }

#[derive(Debug, Clone)]
pub enum MyError {
    OutOfMemory,
    OutOfPermits,
    UnsupportedDegadeLevel,
    SomethingElse(String),
    NotFound
}

impl From<&MyError> for MyError {
    fn from(e: &MyError) -> Self {
        e.clone()
    }
}

impl <T:ToString> From<T> for MyError {
    fn from(msg: T) -> Self {
        let msg = msg.to_string();
        if msg.contains("out of memory") || msg.contains("can't allocate memory") {
            println!("MyError: out of memory: msg={msg}");
            MyError::OutOfMemory
        } else {
            MyError::SomethingElse(msg)
        }

    }
}

#[delegatable_trait]
#[automock(type ID=DummyCachableId;)]
pub trait WithId {
    type ID: CachableId + Send + Sync + Eq + Hash + Clone + Debug;

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


/////////////////////////////////

pub trait OutId:Runnable {
    fn out_id(&self, input_id: &<Self::In as WithId>::ID) -> <Self::Out as WithId>::ID;

    fn on_cached(&self, _input: &impl WithId<ID=<<Self as Runnable>::In as WithId>::ID>, _ctx: &Arc<RwLock<Self::Ctx>>){

    }
}

#[async_trait]
#[delegatable_trait]
#[automock]
pub trait CopyToDevice {
    async fn copy_to_device(&self, dst_device: usize) ->Self;
}


//triggers blanked impl<T: ... Named + 'static> CachableId for T
#[automock]
pub trait Named {
    fn name(&self) -> &'static str {
        "n/a"
    }
}


pub trait Dims:Debug + Send + Sync {
    fn use_data<D:Data>(_t:&D) {}
}

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct SimpleId(pub &'static str);

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

#[macro_export]
macro_rules! panic_to_err {
    ($expr:expr) => {{
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            println!("in panic_to_err");
            $expr
        })) {
            //no panic
            Ok(inner_result) => match inner_result {
                Ok(val) => Ok(val),
                Err(err) => Err(MyError::from(err)),
            },
            //panic
            Err(e) => {
                let msg = if let Some(msg) = e.downcast_ref::<String>() {
                    println!("panic_to_err: String msg={msg}");
                    msg.clone()
                //TODO
                // } else if let Some(thoosa_error) = e.downcast_ref::<ThoosaError>() {
                //     Err(*thoosa_error)
                } else if let Some(msg) = e.downcast_ref::<&str>() {
                    println!("panic_to_err: &str msg={msg}");
                    msg.to_string()
                } else {
                    println!("panic_to_err: unknown msg={e:?}");
                    format!("{e:?}")
                };

                if msg.contains("out of memory") {
                    Err(MyError::OutOfMemory)
                } else {
                    Err(MyError::SomethingElse(msg))
                }
            }
        }
    }};
}


#[macro_export]
macro_rules! panic_to_err_async {

    ($expr:expr) => {{
         let x = std::panic::catch_unwind(std::panic::AssertUnwindSafe(async || {
            $expr
        }));

         match x {
            //no panic
            Ok(inner_result) => match inner_result.await {
                Ok(val) => Ok(val),
                Err(err) => Err(MyError::from(err))
            },
            //panic
            Err(e) => {
                if let Some(msg) = e.downcast_ref::<String>() {
                    println!("thoosa_err String: {msg}");
                    Err(MyError::from(msg))
                //TODO
                // } else if let Some(thoosa_error) = e.downcast_ref::<ThoosaError>() {
                //     Err(*thoosa_error)

                } else if let Some(msg) = e.downcast_ref::<&str>() {
                    println!("thoosa_err str: {msg}");
                    Err(MyError::from(*msg))
                } else {
                    println!("thoosa_err unknown: {e:?}");
                    Err(MyError::from(format!("{e:?}")))
                }
            }
        }
    }};
}
