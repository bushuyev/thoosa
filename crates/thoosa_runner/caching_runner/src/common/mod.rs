// removed unused feature gate to reduce warnings

use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;
use std::ops::{Deref, Range};
use std::sync::Arc;
use ambassador::delegatable_trait;
use async_trait::async_trait;
use mockall::automock;
use parking_lot::RwLock;
use runnable::Stage;
use thoosa_common::tho_tensor::CopyToDevice;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{CachableIdTr, Named, WithId};
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

    async fn purge_gpu_caches(_ctx: Arc<RwLock<Self>>) -> Result<(), ThoosaError>
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


// pub trait Input: WithId + Debug {}
// pub trait Output: WithId + Debug {}



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


/////////////////////////////////

pub trait OutId: Stage where Self::In:WithId {
    fn out_id(&self, input_id: &<Self::In as WithId>::ID) -> <Self::Out as WithId>::ID;

    fn on_cached(&self, _input: &impl WithId<ID=<<Self as Stage>::In as WithId>::ID>, _ctx: &Arc<RwLock<Self::Ctx>>){

    }
}

