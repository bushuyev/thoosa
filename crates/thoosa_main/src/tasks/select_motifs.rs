use std::marker::PhantomData;
use std::sync::Arc;
use parking_lot::RwLock;
use caching_runner::common::cachable::Cached;
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::find_motifs_run_config::SelectMotifsRunConfig;
use thoosa_common::runs::Labeled;
use thoosa_common::ws::Workspace;
use thoosa_motifs::find::{FixedFindMotifsConfig, SelectMotifsSavableResult};
use thoosa_shared::resource::Resource;
use thoosa_shared::traits::SimpleId;
use thoosa_store::Store;
use torch_runner::data::ThoTensor2;
use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::run_ctx_2::RunCtx2;

#[derive(Debug)]
pub struct SelectMotifsStage<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=RunCtx2> + Cached + OutId + 'static,
    C: RunCtx + 'static
{
    // store: Arc<RwLock<Box<dyn Store>>>,
    loader:Arc<L>,
    pd:PhantomData<C>
}

impl <L, C> SelectMotifsStage<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=RunCtx2> + Cached + OutId + 'static,
    C: RunCtx + 'static
{

    pub fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        loader:Arc<L>,
        run_configs:Vec<SelectMotifsRunConfig>,
    ) -> Self {
        Self {
            // store,
            loader,
            pd: Default::default(),

        }
    }

}

impl <L, C> Stage  for  SelectMotifsStage<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=RunCtx2> + Cached + OutId + 'static,
    C: RunCtx + 'static

{
    type Ctx = RunCtx2;
    type In<'a> where Self: 'a = ResourcesConfigInput<Labeled>;
    type Out = SelectMotifsSavableResult;
}
impl <L, C> AsyncStage  for  SelectMotifsStage<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=RunCtx2> + Cached + OutId + 'static,
    C: RunCtx + 'static
{

}