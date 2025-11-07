use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use futures_util::{StreamExt, TryStreamExt};
use log::debug;
use parking_lot::RwLock;
use tch::Device::{Cpu, Cuda};
use tch::nn::VarStore;
use tch::Tensor;
use caching_runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_common::dims::{Dims, ThoBatch};
use thoosa_common::runs::Empty;
use thoosa_common::train_nets::{opt_train_2, run_train_motif_net, MotifNetParams, OptConfig, TensorsDataBatch, TrainMotifsNetsRunConfig, TrainMotifsNetsRunConfigFields, TrainMotifsNetsRunConfigResult};
use thoosa_common::utils::{DataBatch, DummyListener};
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_run_config_args::OptRunConfigSource;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::resource::Resource;
use thoosa_shared::run_configs::{OptRunConfigs, RunId};
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Adjustable, DataId, SimpleId, WithId};
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_store::Store;
use thoosa_with_data::with_data;
use torch_runner::data::{DataBatch2, ThoTensor2};
use crate::tasks::cross_dims::{ChannelsData2, MotifSlices};
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::motif_loader::{MotifLoader, MotifMeta};
use crate::tasks::run_ctx_2::MotifsCtx;

#[derive(Debug)]
pub struct TrainMotifNet<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    run_configs: Arc<OptRunConfigs<
        Vec<NamedVal<TrainMotifsNetsRunConfigFields>>,
        Vec<NamedRange<TrainMotifsNetsRunConfigFields>>,
        Vec<Hyperparameter<TrainMotifsNetsRunConfigFields>>,
        TrainMotifsNetsRunConfigFields,
    >>,
    motif_loader:Arc<MotifLoader<L,C>>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<RwLock<Box<dyn Store>>>,
    pd:PhantomData<C>

}

#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct TrainMotifNetInput{
    id:SimpleId,
    motif_selection_run_id: Vec<RunId>,
    opt_config:Arc<OptConfig>
}

impl Input for TrainMotifNetInput {}
impl TrainMotifNetInput {
    pub fn new(motif_selection_run_id: Vec<RunId>, opt_config:Arc<OptConfig>) -> Self {
        Self { id: SimpleId("TrainMotifNetInput"), motif_selection_run_id, opt_config }
    }
}

impl WithId for TrainMotifNetInput {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}
impl <L, C> Stage for TrainMotifNet<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    type Ctx = C;
    type In<'a> = TrainMotifNetInput;//MotifID
    type Out = TrainMotifsNetsRunConfigResult;
}

impl <L, C> TrainMotifNet<L, C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
{
    pub fn new(
        loader:Arc<L>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        run_configs:  OptRunConfigs<
            Vec<NamedVal<TrainMotifsNetsRunConfigFields>>,
            Vec<NamedRange<TrainMotifsNetsRunConfigFields>>,
            Vec<Hyperparameter<TrainMotifsNetsRunConfigFields>>,
            TrainMotifsNetsRunConfigFields,
        >    ) -> Self {
        let motif_loader = Arc::new(MotifLoader::new(
            Arc::clone(&store),
            Arc::clone(&loader)
        ));

        Self {
            motif_loader,
            run_configs: Arc::new(run_configs),
            ws: Arc::clone(&ws),
            store: Arc::clone(&store),
            pd: PhantomData
        }
    }

}
#[async_trait]
impl <L, C> AsyncStage for TrainMotifNet<L, C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
{
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
        let (device_idx, cache_area, ctx_name) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area(), guard.name())
        };

        let (motif_selection_run_ids, opt_config) = input.with_data_ref(0, None, |input|{
            Ok((input.motif_selection_run_id.clone(), input.opt_config.clone()))
        }).await?;

        let channels = self.ws.read().channels();



        for motif_selection_run_id in motif_selection_run_ids {

            let x = self.store.read().list_selected_motifs(motif_selection_run_id, channels)?;

            let result = futures::stream::iter(
                x.into_iter()
                    .flat_map(|channels_motifs|
                        channels_motifs.into_iter())
            )
                .map(|motif_id_rate| {
                    let motif_loader = Arc::clone(&self.motif_loader);
                    let ctx = Arc::clone(&ctx);
                    let opt_config = Arc::clone(&opt_config);
                    let train_motifs_nodes_run_configs = Arc::clone(&self.run_configs);
                    async move {
                        let motif = motif_loader.run(Arc::new(DevicesWrap::wrap_cpu(motif_id_rate.0.into())), ctx).await?;

                        opt_train_2(
                            opt_config,
                            async move |vs: &VarStore, run_config: TrainMotifsNetsRunConfig, _listener| {
                                //TODO batch size
                                // let all_slices_0_cl = all_slices_0_cl.iter().map(|v| v.shallow_clone()).collect::<Vec<_>>();
                                // let all_slices_1_cl = all_slices_1_cl.iter().map(|v| v.shallow_clone()).collect::<Vec<_>>();
                                let mut batch = with_data!(device_idx, motif; |motif:&DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>|{

                                    // let motif_data = motif.data();
                                    // let motif_meta = motif.extra();



                                    Ok::<_, ThoosaError>(AugmentationsBatch::new(motif))
                                })?;

                                 // AugmentationsBatch::new();

                                run_train_motif_net(1000, &run_config, &Cuda(0), &mut batch, vs, &mut DummyListener/*, listener*/)
                            },
                            train_motifs_nodes_run_configs,
                            Cuda(0),
                            "motifs"
                        ).await.map(|(var_store, motif_node_params, train_motifs_nets_run_config, min_loss)| {
                            // Arc::new(Box::new(TrainMotifsNetsRunConfigResult::new(min_loss, input.channel, input.motif_id, var_store, input.find_motifs_run_config.id.unwrap()/* TODO train_motifs_run_config.id.expect("id")*/, motif_node_params, train_motifs_nets_run_config)))
                            ()
                        }).expect("TODO opt_train failed");


                        Ok::<_, ThoosaError>(())
                    }
                })
                .buffer_unordered(10)
                .try_collect::<Vec<_>>()
                .await?;

        }

        Ok(Arc::new(DevicesWrap::wrap_cpu(TrainMotifsNetsRunConfigResult::new(0.1, 0, 0, VarStore::new(Cpu), 0, MotifNetParams::new(1, 2), TrainMotifsNetsRunConfig::new(Some(0))))))
    }

}

#[derive(Debug)]
pub struct AugmentationsBatch {

}

impl AugmentationsBatch {
    pub fn new(motif:&DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>) -> Self {
        debug!("TrainMotifNet::run: motif={motif:?}");
        Self {}
    }
}

impl<D: Dims> DataBatch<D> for AugmentationsBatch {
    fn train_batch(&self) -> (ThoBatch<D>, Tensor) {
        todo!()
    }

    fn valid_batch(&self) -> (ThoBatch<D>, Tensor) {
        todo!()
    }
}

/*
impl DataBatch2<MotifSlices, DataId> for AugmentationsBatch {
    fn train_batch(&self) -> (ThoTensor2<MotifSlices, DataId>, ThoTensor2<MotifSlices, ()>) {
        todo!()
    }

    fn valid_batch(&self) -> (ThoTensor2<MotifSlices, DataId>, ThoTensor2<MotifSlices, ()>) {
        todo!()
    }
}

 */


// impl SyncStage

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Arc;
    use log::LevelFilter::Trace;
    use parking_lot::RwLock;
    use tokio::fs;
    use caching_runner::common::cachable::DevicesWrap;
    use caching_runner::common::mock_run_ctx;
    use caching_runner::common::runnable::AsyncStage;
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::train_nets::{CommonTrainingRunConfig, CommonTrainingRunConfigFields, OptConfig, TrainMotifsNetsRunConfig, TrainMotifsNetsRunConfigFields};
    use thoosa_common::utils::reset_print_options;
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::{MotifKind, TensorIdSliceIdx, ThoosaError};
    use thoosa_shared::val::{NamedRange, NamedVal, Val};
    use thoosa_store::MockStoreBuilder;
    use crate::tasks::select_motifs::tests::resource_tensor;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use crate::tasks::train_motif_net::{TrainMotifNet, TrainMotifNetInput};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_0() -> Result<(), ThoosaError>{
        let test_dir = Path::new("/tmp/test_single_label_one_motif");//.join(cdt_for_dir());
        fs::create_dir_all(test_dir).await?;

        log::set_max_level(Trace);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 10000)));
        let find_motifs_run_id:RunId = 1;
        let select_motifs_run_id:RunId = 2;
        let resource_a: Resource = Resource::new(1, 1, "resource_a", 900);
        let resource_b: Resource = Resource::new(1, 1, "resource_b", 900);
        let (resource_a_t, resource_a_intervals) = resource_tensor(resource_a.length() as usize, &[
            vec![(3, 100, MotifKind::Pulse)],
            vec![(3, 100, MotifKind::Sin)],
            vec![]
        ], 100)?;
        let (resource_b_t, resource_a_intervals) = resource_tensor(resource_b.length() as usize, &[
            vec![(3, 100, MotifKind::Pulse)],
            vec![(3, 100, MotifKind::Sin)],
            vec![]
        ], 100)?;

        let motif_id = 10u32;
        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
                .with_list_run_config_values(
                    find_motifs_run_id,
                    Ok(FindMotifsRunConfig::new(Some(find_motifs_run_id), 50, 100, None, None).to_properties_iter(None).collect()),
                )
                .with_list_motif_slices(motif_id, None, None, Ok((0i64, find_motifs_run_id, vec![
                    TensorIdSliceIdx(1, 100),
                    TensorIdSliceIdx(1, 200),
                    TensorIdSliceIdx(2, 100)
                ], HashMap::from([
                    (1, resource_a.clone()),
                    (2, resource_b.clone())
                ]))))
                .with_list_selected_motifs([
                    (select_motifs_run_id, Ok(vec![vec![(motif_id, 1.0f64)]]))
                ])
                // .with_list_motifs([
                //     (find_motifs_run_id as RunId, Ok(channel_0_motifs_ids.clone()))
                // ])
                // .with_list_group_labeled_resources_sets([
                //     (resources_group, Ok(vec![
                //         ("Lable_0".to_string(), "resource_set_for_label_0".to_string()),
                //     ]))
                // ])
                // .with_list_resource_set_resources_by_descr([
                //     ("resource_set_for_label_0", Ok(vec![resource_a.clone()])),
                // ])
                // .with_list_resources_not_in_group([
                //     (resources_group, Ok(vec![resource_a.clone()])),
                // ])
                // .with_add_slices_to_motif(crate::tasks::select_motifs::tests::resources_add_slices_to_motif_expectations_2(run_id, &channel_0_motifs_ids, &resources_channels_intervals))
                .build()));

        let run_configs =         thoosa_shared::run_configs::OptRunConfigs::new(
            thoosa_shared::run_configs::GridRunConfigs::new(
                thoosa_shared::run_configs::ConstConfigs::new(vec![
                    NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Cycles), Val::Int(3)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::BatchSize), Val::Int(10)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), Val::Real(0.1)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::MaxLoss), Val::Real(0.01)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::OptMaxEvals), Val::Int(10)),
                    NamedVal::new(TrainMotifsNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::OptNInit), Val::Int(10)),
                ]),
                vec![
                    NamedRange::new(TrainMotifsNetsRunConfigFields::NumLayers, Val::Int(0)..Val::Int(10), (Val::Int(10), 0))
                ],
            ),
            vec![],
        );



        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached([
                    (resource_a.clone(), resource_a_t),
                ])
                .finish(),
        );


        let train_motif_net = Arc::new(TrainMotifNet::new(
            loader,
            &ws,
            &store,
            run_configs
        ));

        let input = Arc::new(DevicesWrap::wrap_cpu(TrainMotifNetInput::new(vec![select_motifs_run_id], Arc::new(OptConfig::new(100, 3)))));

        let r = train_motif_net.run(input, Arc::new(RwLock::new(mock_run_ctx::default(0)))).await?;

        Ok(())
    }

}