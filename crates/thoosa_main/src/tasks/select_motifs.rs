use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use async_trait::async_trait;
use futures_util::{StreamExt, TryStreamExt};
use log::{debug, error, info};
use parking_lot::RwLock;
use caching_runner::common::cachable::{Cached, DevicesWrap};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::find_motifs_run_config::SelectMotifsRunConfig;
use thoosa_common::runs::Labeled;
use thoosa_common::ws::Workspace;
use thoosa_motifs::find::{FixedFindMotifsConfig, SelectMotifsSavableResult};
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{DataId, SimpleId};
use thoosa_store::Store;
use torch_runner::data::ThoTensor2;
use crate::cli::Commands::SelectMotifs;
use crate::tasks::cross_dims::{ChannelsData2, MotifSlices};
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::motif_loader::{MotifLoader, MotifMeta};
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::RawTensorLoader;

#[derive(Debug)]
pub struct SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + 'static
{

    store: Arc<RwLock<Box<dyn Store>>>,
    loader:Arc<L>,
    motif_loader:Arc<ML>,
    pd:PhantomData<C>,
    run_configs:Vec<SelectMotifsRunConfig>,
    find_motifs_run_ids:Vec<RunId>,
}

impl <L, ML, C> SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + 'static
{

    pub fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        loader:Arc<L>,
        motif_loader:Arc<ML>,
        run_configs:Vec<SelectMotifsRunConfig>,
        find_motifs_run_ids:Vec<RunId>,
    ) -> Self {
        Self {
            store: Arc::clone(store),
            loader,
            motif_loader,
            pd: Default::default(),
            run_configs,
            find_motifs_run_ids
        }
    }

}


impl <L,ML, C> Stage  for  SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + 'static

{
    type Ctx = C;
    type In<'a> = ResourcesConfigInput<Labeled>;
    type Out = SelectMotifsSavableResult;
}

#[async_trait]
impl <L, ML, C> AsyncStage  for  SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + 'static
{
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
        let resources = input.with_data_ref(0, None, |input|{
            Ok(input
                .resources_config()
                .to_resources(&self.store)?
                .into_iter()
                .map(|r| Arc::new(r))
                .collect::<Vec<_>>())
        }).await?;


        for find_motifs_run_id in &self.find_motifs_run_ids {

            for run_config in self.run_configs.iter() {
                let motifs = self.store.read().list_motifs(*find_motifs_run_id)?;
                let motifs_len = motifs.len();
                let done_cnt = Arc::new(AtomicU32::new(0));

                futures::stream::iter(motifs)
                    .map(Ok::<usize, ThoosaError>)
                    .try_for_each_concurrent(10, |motif_id| {
                        let done_cnt = Arc::clone(&done_cnt);
                        let motif_loader_impl = Arc::clone(&self.motif_loader);
                        let ctx = Arc::clone(&ctx);
                        async move {
                            let motif = motif_loader_impl.run_cached(Arc::new(DevicesWrap::wrap_cpu((motif_id as u32).into())), ctx).await?;
                            debug!("SelectMotifsStage::run: find_motifs_run_id={find_motifs_run_id} run_config.id={:?} motif={motif:?}", run_config.id);

                            Ok(())
                        }
                    })
                    .await?;
            }
        }


        Ok(Arc::new(DevicesWrap::wrap_cpu(SelectMotifsSavableResult::new(0, self.run_configs[0].clone(), vec![]))))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use parking_lot::RwLock;
    use tch::Tensor;
    use caching_runner::common::cachable::DevicesWrap;
    use caching_runner::common::MockRunCtx;
    use caching_runner::common::runnable::AsyncStage;
    use caching_runner::global_ctx;
    use caching_runner::global_ctx::{GlobalCtx};
    use thoosa_common::find_motifs_run_config::{ByLabelVarianceConfig, FindMotifsRunConfig, MotifSelector, SelectMotifsRunConfig};
    use thoosa_common::runs::Labeled;
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::{TensorIdSliceIdx, ThoosaError};
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::traits::{DataId, SimpleId};
    use thoosa_store::MockStoreBuilder;
    use torch_runner::data::ThoTensor2;
    use crate::tasks::cross_dims::ChannelsData2;
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::motif_loader::{mock_motif_loader, MotifMeta};
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::select_motifs::SelectMotifsStage;
    use crate::tasks::tensor_loader::{mock_tensor_loader, mock_tensor_loader_2};
    use crate::tasks::tensor_loader::mock_tensor_loader::make_wrapper;

    #[ctor::ctor]
    fn test_init() {
        thoosa_shared::test_utils::init_test_logging();
    }


    #[tokio::test]
    async fn test_two_labels() -> Result<(), ThoosaError>{
       GlobalCtx::init_devices(2);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 3, 100, 1000)));
        let run_id = 1;

        let resource_a: Resource = (1, 1, "resource_a", 1000).into();
        let resource_b: Resource = (2, 1, "resource_b", 1000).into();

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 1;

        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
            .with_list_motifs([
                (find_motifs_run_id as RunId, Ok(vec![1usize]))
            ])
                .with_list_group_labeled_resources_sets()
            .build()));

       

        let loader = Arc::new(
            mock_tensor_loader_2::build()
                .with_data([
                    (resource_a.clone(), Tensor::from_slice(&[1.0, 2.0]).view((1,2))),
                    (resource_b.clone(), Tensor::from_slice(&[3.0, 4.0]).view((1,2))),
                ])
                .finish(),
        );

        let motif_tensor = ThoTensor2::new(DataId::from(999u32), Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]).view((6, 3)));

        let motif_wrapper = Arc::new(DevicesWrap::wrap(DataWithExtra::new(motif_tensor, MotifMeta { channel: 0, run_id: 1 }), 0));


        let motif_loader = mock_motif_loader::build().with_result(Arc::clone(&motif_wrapper)).finish();


        let select_motifs_stage:Arc<SelectMotifsStage<_, _,MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new_(Some(select_motifs_run_id), MotifSelector::ByLabelVariance(ByLabelVarianceConfig{ _dummy: 0 }))],
            vec![find_motifs_run_id]
        ));


        let result = select_motifs_stage.run(
            Arc::new(DevicesWrap::wrap_cpu(ResourcesConfigInput::new("test".into(), Labeled::new("group".into())))), Arc::new(RwLock::new(MockRunCtx::new()))
        ).await?;

        Ok(())
    }

}