use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use futures_util::{StreamExt, TryStreamExt};
use log::debug;
use parking_lot::RwLock;
use tch::nn::{AdamW, OptimizerConfig, VarStore};
use tch::{Device, Reduction, Tensor};
use cached_derive::Cached;
use caching_runner::common::cachable::{Cached, DevicesWrap};
use caching_runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::tho_tensor::{Labels, ThoTensor};
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, SimpleId, WithId};
use crate::tasks::cross_dims::ChannelsData2;
use caching_runner::common::{OutId, RunCtx};
use caching_runner::global_ctx::GlobalCtx;
use thoosa_common::applicables::ThoDevice;
use thoosa_common::batches::AsyncDataBatch;
use thoosa_common::cuda_utils::cuda_mem;
use thoosa_common::dims::{Batch, LabelProbability, ThoBatch};
use thoosa_common::runs::{Labeled, NetResources, SavableResult};
use thoosa_common::train_nets::{flat_label_node, opt_train_2, FixedNumSlicesNetParams, LabelNetParams, LabelNetTypeParams, OptConfig, TrainLabelNetsRunConfig, TrainLabelNetsRunConfigFields, TrainLabelNetsRunConfigResult};
use thoosa_common::utils::train_2;
use thoosa_common::ws::Workspace;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::{CopyToDevice, ThoosaError};
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::run_configs::{OptRunConfigs, RunId};
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_store::Store;
use thoosa_with_data::with_data;
use crate::tasks::apply_motif_nets::ApplyMotifNets;
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::run_ctx_2::MotifsCtx;
use rand::{seq::SliceRandom, rng};

#[derive(Debug)]
pub struct TrainLabelNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    run_configs: Arc<OptRunConfigs<
        Vec<NamedVal<TrainLabelNetsRunConfigFields>>,
        Vec<NamedRange<TrainLabelNetsRunConfigFields>>,
        Vec<Hyperparameter<TrainLabelNetsRunConfigFields>>,
        TrainLabelNetsRunConfigFields,
    >>,
    loader: Arc<L>,
    apply_motif_nets:Arc<ApplyMotifNets<C>>,
    store: Arc<RwLock<Box<dyn Store>>>,
    opt_config:Arc<OptConfig>
}

impl <L, C> TrainLabelNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    pub async fn new<R>(
        select_motifs_run_id: Vec<RunId>,
        loader: Arc<L>,
        run_configs: OptRunConfigs<Vec<NamedVal<TrainLabelNetsRunConfigFields>>, Vec<NamedRange<TrainLabelNetsRunConfigFields>>, Vec<Hyperparameter<TrainLabelNetsRunConfigFields>>, TrainLabelNetsRunConfigFields>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        stage_ctx:Arc<parking_lot::RwLock<R>>,
        opt_config:Arc<OptConfig>
    ) -> Result<Self, ThoosaError>
where
    R:RunCtx + MotifsCtx + Adjustable + 'static
    {

        Ok(Self {
            run_configs: Arc::new(run_configs),
            loader,
            apply_motif_nets: Arc::new(ApplyMotifNets::new("".into(), ws, store, select_motifs_run_id, stage_ctx).await?),
            store: Arc::clone(store),
            opt_config
        })
    }
}

#[derive(Debug)]
pub struct TrainLabelNetsInput {
    id:SimpleId
}

impl TrainLabelNetsInput {
    pub fn new(id: SimpleId) -> Self {
        Self { id }
    }
}

#[async_trait]
impl CopyToDevice for TrainLabelNetsInput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoosaError> where Self:Sized {
        todo!()
    }
}

impl Data for TrainLabelNetsInput {
    type Metrics = ();

    fn size(&self) -> Self::Metrics {
        ()
    }

    fn device(&self) -> usize {
        0
    }
}


impl Input for TrainLabelNetsInput {

}

impl WithId for TrainLabelNetsInput {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}


#[derive(Debug)]
pub struct TrainLabelNetsOutput {
    id:SimpleId
}
impl WithId for TrainLabelNetsOutput {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}
impl Output for TrainLabelNetsOutput {

}

impl SavableResult for TrainLabelNetsOutput {
    fn value(&self) -> f32 {
        todo!()
    }

    fn save(&mut self, store: &Arc<RwLock<Box<dyn Store>>>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<u32, ThoosaError> {
        todo!()
    }
}

#[async_trait]
impl CopyToDevice for TrainLabelNetsOutput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoosaError> where Self:Sized {
        todo!()
    }
}

impl Data for TrainLabelNetsOutput {
    type Metrics = ();

    fn size(&self) -> Self::Metrics {
        ()
    }

    fn device(&self) -> usize {
        0
    }
}

impl TrainLabelNetsOutput {
    pub fn new(id: SimpleId) -> Self {
        Self { id }
    }
}

impl <L, C> Stage for TrainLabelNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    type Ctx = C;
    type In<'a> = ResourcesConfigInput<NetResources<Labeled>>;
    type Out = TrainLabelNetsOutput;
}

impl<L, C> AsyncStage for TrainLabelNets<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> impl Future<Output=Result<Arc<DevicesWrap<Self::Out>>, ThoosaError>> + Send {
        let (device_idx, cache_area, ctx_name) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area(), guard.name())
        };
        let train_motifs_nodes_run_configs = Arc::clone(&self.run_configs);
        let ctx = Arc::clone(&ctx);
        let store = Arc::clone(&self.store);

        async move {
            let (mut train_labels_resource_keys, mut valid_label_resource_keys) = with_data!(0, input; |input:&ResourcesConfigInput<NetResources<Labeled>>|{
                Ok::<_, ThoosaError>(input.resources_config().to_resources(&self.store)?)
            })?;
            debug!("TrainLabelNets::run: train_labels_resource_keys={train_labels_resource_keys:?}");
            debug!("TrainLabelNets::run: valid_label_resource_keys={train_labels_resource_keys:?}");

            assert_eq!(train_labels_resource_keys.len(), valid_label_resource_keys.len());

            //should not be needed
            // train_labels_resource_keys.sort_by(|a, b|a.0.cmp(&b.0));
            // valid_label_resource_keys.sort_by(|a, b|a.0.cmp(&b.0));

            let mut labels_others_train_valid_resources:HashMap<String, (Vec<&Resource>, Vec<&Resource>)> = train_labels_resource_keys.iter().fold(HashMap::new(), |mut a, v|{
                assert!(a.insert(v.0.clone(), (vec![], vec![])).is_none());
                a
            });

            let labels_tain_and_valid_sets = train_labels_resource_keys.into_iter().zip(valid_label_resource_keys.into_iter()).fold(HashMap::new(), |mut a,(train, valid)|{
                assert_eq!(train.0, valid.0);//train and valid labels should come in same order from db

                assert!(a.insert(train.0, (train.1, valid.1)).is_none());
                a
            });


            for (other_label, train_valid_resources) in labels_others_train_valid_resources.iter_mut(){
                for (label, resources) in &labels_tain_and_valid_sets {
                    if label.ne(other_label) {
                        train_valid_resources.0.extend(resources.0.iter());
                        train_valid_resources.1.extend(resources.1.iter());
                    }
                }
            }

            //collecting resources not belonging to the label, shuffling and taking as many as in the label.
            let labels_others_train_valid_resources = Arc::new(labels_others_train_valid_resources.into_iter().map(|(label, resources)|{
                let mut train_resources = resources.0;
                train_resources.shuffle(&mut rng());
                train_resources.truncate(labels_tain_and_valid_sets.get(&label).expect(&format!("no {label} for train resources")).0.len());


                let mut valid_resources = resources.1;
                valid_resources.shuffle(&mut rng());
                valid_resources.truncate(labels_tain_and_valid_sets.get(&label).expect(&format!("no {label} for valid resources")).1.len());

                (
                    label,
                    (
                    train_resources.into_iter().map(|r|r.clone()).collect::<Vec<_>>(),
                    valid_resources.into_iter().map(|r|r.clone()).collect::<Vec<_>>()
                    )
                )
            }).collect::<HashMap<String, (Vec<Resource>, Vec<Resource>)>>());
            debug!("TrainLabelNets::run: labels_others_train_valid_resources={labels_others_train_valid_resources:?}");

            let out_key = {
                let out_id = SimpleId("todo");  //self.out_id(input.id());
                Arc::new(out_id) as Arc<dyn CachableIdTr>
            };


            futures::stream::iter(labels_tain_and_valid_sets.clone())
            .map(|label_tain_and_valid_sets|{


                let opt_config = Arc::clone(&self.opt_config);
                let ctx = Arc::clone(&ctx);
                let out_key = Arc::clone(&out_key);
                let train_motifs_nodes_run_configs = Arc::clone(&train_motifs_nodes_run_configs);
                let store = Arc::clone(&self.store);

                let labels_others_train_valid_resources = Arc::clone(&labels_others_train_valid_resources);

                tokio::spawn(async move {
                    let (label, resources) = label_tain_and_valid_sets;
                    debug!("TrainLabelNets::run: label={label} resources={resources:?}");

                    let resources_x = labels_others_train_valid_resources.get(&label).expect(&format!("no resources not belonging to {label}"));
                    debug!("TrainLabelNets::run: label={label} resources_x={resources_x:?}");

                    opt_train_2(
                        opt_config,

                        async move  |vs: &VarStore, run_config: TrainLabelNetsRunConfig, _listener| {
                            if ctx.read().acquired_permits() == 0 { //TODO move to with_data!
                                let result = GlobalCtx::global()
                                    .acquire_many(device_idx, 1, out_key.clone())
                                    .await?;
                                ctx.write().add_permit(result); //permits can be dropped only from OOM handler or when context is dropped
                            }
                            let resources_batch = ResourcesBatch::new(resources.0.clone());

                            // let batch = MotifsBatch::new(Arc::clone(&motif), Arc::clone(&random_slices), run_config.common_training_args.batch_size as i64, device_idx, cache_area).await?;
                            //
                            train_label_net_2(&run_config, &resources_batch,  &ThoDevice::from(device_idx), vs).await?;
                            Ok((0.0, LabelNetParams::new(20, 2, 10, 100, LabelNetTypeParams::FixedNumSlices(FixedNumSlicesNetParams::new()))))
                        },
                        train_motifs_nodes_run_configs,
                        ThoDevice::from(device_idx).into(),
                        "motifs",
                    ).await.map(|(var_store, label_node_params, train_motifs_nets_run_config, min_loss)| {

                        debug!("TrainLabelNet::run: label_node_params={label_node_params:?} train_motifs_nets_run_config={train_motifs_nets_run_config:?} min_loss={min_loss}");
                        TrainLabelNetsRunConfigResult::new(
                            0,
                            label,
                            min_loss,
                            label_node_params,
                            train_motifs_nets_run_config,
                            var_store,

                        )
                    })
                })

            })
                .buffer_unordered(10)
                .try_collect::<Vec<_>>().await?
                .into_iter()
                .collect::<Result<Vec<_>, ThoosaError>>()?;




            Ok(Arc::new(DevicesWrap::wrap(TrainLabelNetsOutput::new("".into()), cache_area)))
        }
    }
}


#[derive(Debug)]
struct ResourcesBatch {

}


impl ResourcesBatch {
    pub fn new(resources:Vec<Resource>) -> Self {
        Self {}
    }
}

#[async_trait]
impl AsyncDataBatch<ChannelsData2> for ResourcesBatch {
    async fn train_batch(&self) -> Result<(ThoBatch<ChannelsData2>, Tensor), ThoosaError> {
        todo!()
    }

    async fn valid_batch(&self) -> Result<(ThoBatch<ChannelsData2>, Tensor), ThoosaError> {
        todo!()
    }
}

pub async fn train_label_net_2(
    run_config: &TrainLabelNetsRunConfig,
    batch: &dyn AsyncDataBatch<ChannelsData2>,
    device: &Device,
    vs: &VarStore,
) -> Result<f32, ThoosaError> {
    debug!("train_label_net:  ");

    let device = device.clone();

    let optimizer = AdamW::default().amsgrad(true);

    //TODO
    // if let Some(adam_w) = run_config.common_training_args.adam_w {
    //     debug!("train_label_nodes: setting adam_w={adam_w:?}");
    //     optimizer = optimizer.beta1(adam_w.beta1).beta2(adam_w.beta2);
    // }

    debug!("train_label_net: cuda_mem={:?}", cuda_mem());

    let params = LabelNetParams::new(0, 0, 3, 10, LabelNetTypeParams::FixedNumSlices(FixedNumSlicesNetParams::new()));
    let loss = train_2(
        &device,
        batch,
        flat_label_node(&params, 0).build_tch_module::<Batch<ChannelsData2>, Batch<LabelProbability>>(&vs.root())?,
        |pred: &Tensor, target: &Tensor| {
            debug!("train_label_net: pred.size={:?}", pred.size());
            let pred_y = pred.squeeze();
            debug!("train_label_net: pred_y={:?} target={:?} kind={:?}",pred_y.size(),target.size(),target.kind());
            pred_y.binary_cross_entropy::<Tensor>(&target, None, Reduction::Mean)
        },
        run_config.common_training_args.cycles as usize,
        optimizer.build(&vs, run_config.common_training_args.lr)?,
        |t| {
            debug!(
                "train_label_net: t.size={:?} cuda_mem={:?}",
                t.size(),
                cuda_mem()
            );
            t
        },
        None,
    )
        .await?;

    Ok(loss)
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use tokio::fs;
    use std::path::Path;
    use std::sync::{Arc};
    use log::LevelFilter::Debug;
    use parking_lot::RwLock;
    use tch::Device::Cpu;
    use caching_runner::common::cachable::DevicesWrap;
    use caching_runner::common::mock_run_ctx;
    use caching_runner::common::runnable::AsyncStage;
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_common::runs::{Labeled, NetResources};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{CommonTrainingRunConfigFields, FixedNumSlicesNetArgsFields, LabelNetTypeArgsFields, OptConfig, TrainLabelNetsRunConfigFields};
    use thoosa_common::utils::reset_print_options;
    use thoosa_common::ws::{mock_ws_simp, WsMockBuilder};
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::{MotifKind, TensorIdSliceIdx, ThoosaError};
    use thoosa_shared::hyperparameter::Hyperparameter;
    use thoosa_shared::val::{NamedVal, Val};
    use thoosa_store::MockStoreBuilder;
    use crate::tasks::apply_motif_nets::tests::motif_test_net_;
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::select_motifs::tests::{motifs_tensor, resource_tensor};
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use crate::tasks::train_label_nets::{TrainLabelNets, TrainLabelNetsInput};

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_0() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp/test_train_motif_nets");//.join(cdt_for_dir());
        fs::create_dir_all(test_dir).await?;

        log::set_max_level(Debug);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let input_n_0 = 2000;
        let find_motifs_run_id:RunId = 1;
        let select_motifs_run_id:RunId = 2;
        let motif_width = 100;

        let motif_0_0_id = 0; //channel 0 motif 0
        let motif_0_1_id = 1; //channel 0 motif 1
        let motif_1_0_id = 0; //channel 1 motif 0

        let pulse_motif = Arc::new(ThoTensor::new("motif_0_0".into(), motifs_tensor(100, 3, MotifKind::Pulse)));
        let triangle_motif = Arc::new(ThoTensor::new("motif_0_1".into(), motifs_tensor(100, 3, MotifKind::Triangle)));
        let sin_motif = Arc::new(ThoTensor::new("motif_1_0".into(), motifs_tensor(100, 3, MotifKind::Sin)));
        let train_group = "train_group";
        let valid_group = "valid_group";

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 3, 100, input_n_0 as u32)
                .with_load_motif_nets(vec![
                    vec![
                        //channel 0
                        motif_test_net_(Cpu, Arc::clone(&pulse_motif)),
                        motif_test_net_(Cpu, Arc::clone(&triangle_motif)),
                        motif_test_net_(Cpu, Arc::clone(&sin_motif)),
                    ],
                    vec![
                        //channel 1
                        motif_test_net_(Cpu, Arc::clone(&sin_motif)),
                    ],
                    vec![],
                ])
                .build(),
        ));

        let resources_a = (0..10).map(|i|Resource::new(i, 1, format!("resource_a_{i}"), 900)).collect::<Vec<_>>();
        let resources_b = (10..20).map(|i|Resource::new(i, 1, format!("resource_b_{i}"), 900)).collect::<Vec<_>>();
        let resources_c = (20..30).map(|i|Resource::new(i, 1, format!("resource_c_{i}"), 900)).collect::<Vec<_>>();
        let resources_x = (30..40).map(|i|Resource::new(i, 1, format!("resource_x_{i}"), 900)).collect::<Vec<_>>();

        let (mut resources_map, motif_intervals) = resources_a.iter().zip(resources_b.iter().zip(resources_c.iter().zip(resources_x.iter()))).try_fold((HashMap::new(), vec![]), |mut a, (r_a, (r_b, (r_c, r_x)))| {

            let (t, intervals) = resource_tensor(r_a.length() as usize, &[
                vec![(3, motif_width, MotifKind::Pulse)],
                vec![(1, motif_width, MotifKind::Sin)],
                vec![]
            ], 100)?;

            a.0.insert(r_a.clone(), t);

            a.1.extend(intervals[0][0].iter().map(|interval|TensorIdSliceIdx(r_a.id(), interval.start as u32)));


            let (t, intervals) = resource_tensor(r_b.length() as usize, &[
                vec![(3, motif_width, MotifKind::Triangle)],
                vec![(1, motif_width, MotifKind::Sin)],
                vec![]
            ], 100)?;

            a.0.insert(r_b.clone(), t);

            a.1.extend(intervals[0][0].iter().map(|interval|TensorIdSliceIdx(r_b.id(), interval.start as u32)));


            let (t, intervals) = resource_tensor(r_c.length() as usize, &[
                vec![(3, motif_width, MotifKind::Sin)],
                vec![(1, motif_width, MotifKind::Sin)],
                vec![]
            ], 100)?;

            a.0.insert(r_b.clone(), t);

            a.1.extend(intervals[0][0].iter().map(|interval|TensorIdSliceIdx(r_c.id(), interval.start as u32)));


            let (t, intervals) = resource_tensor(r_x.length() as usize, &[
                vec![],
                vec![],
                vec![]
            ], 100)?;

            a.0.insert(r_b.clone(), t);


            Ok::<_, ThoosaError>(a)
        })?;


        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached(resources_map)
                .finish(),
        );

        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
            .with_list_selected_motifs([
                (select_motifs_run_id, Ok(
                    vec![
                        vec![(motif_0_0_id, 1.0f64), (motif_0_1_id, 1.0f64)], //channel 0
                        vec![(motif_1_0_id, 1.0f64)], //channel 1
                        vec![]//channel 2
                    ]
                ))
            ])
            .with_list_group_labeled_resources_sets([
                (train_group, Ok(vec![
                    ("Label_A".to_string(), "train_resource_set_for_label_a".to_string()),
                    ("Label_B".to_string(), "train_resource_set_for_label_b".to_string()),
                    ("Label_C".to_string(), "train_resource_set_for_label_c".to_string())
                ])),
                (valid_group, Ok(vec![
                    ("Label_A".to_string(), "valid_resource_set_for_label_a".to_string()),
                    ("Label_B".to_string(), "valid_resource_set_for_label_b".to_string()),
                    ("Label_C".to_string(), "valid_resource_set_for_label_c".to_string())
                ]))

            ])
            .with_list_resource_set_resources_by_descr([
                ("train_resource_set_for_label_a", Ok(resources_a.clone())),
                ("train_resource_set_for_label_b", Ok(resources_b.clone())),
                ("train_resource_set_for_label_c", Ok(resources_c.clone())),

                //same resources, should be ok for test
                ("valid_resource_set_for_label_a", Ok(resources_a.clone())),
                ("valid_resource_set_for_label_b", Ok(resources_b.clone())),
                ("valid_resource_set_for_label_c", Ok(resources_c.clone())),

            ])
            .with_list_resource_random_not_labeled([
                (train_group, Ok(resources_x.clone())),
                (valid_group, Ok(resources_x.clone())),
            ])
            .build()));


        let run_configs = thoosa_shared::run_configs::OptRunConfigs::new(
            thoosa_shared::run_configs::GridRunConfigs::new(
                thoosa_shared::run_configs::ConstConfigs::new(vec![
                    NamedVal::new(TrainLabelNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Cycles), Val::Int(20)),
                    NamedVal::new(TrainLabelNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::BatchSize), Val::Int(11)),
                    NamedVal::new(TrainLabelNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), Val::Real(0.001)),
                    NamedVal::new(TrainLabelNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::MaxLoss), Val::Real(0.01)),
                    NamedVal::new(TrainLabelNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::OptMaxEvals), Val::Int(10)),
                    NamedVal::new(TrainLabelNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::OptNInit), Val::Int(10)),
                    NamedVal::new(TrainLabelNetsRunConfigFields::Net(LabelNetTypeArgsFields::FixedNumSlices(FixedNumSlicesNetArgsFields::NumLayers)), Val::Int(3)),
                    NamedVal::new(TrainLabelNetsRunConfigFields::Net(LabelNetTypeArgsFields::FixedNumSlices(FixedNumSlicesNetArgsFields::SlicesLen)), Val::Int(10)),
                ]),
                vec![
                    // NamedRange::new(TrainLabelNetsRunConfigFields::NumLayers, Val::Int(0)..Val::Int(10), (Val::Int(10), 0))
                ],
            ),
            vec![
                Hyperparameter::UniformFloat{name: TrainLabelNetsRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), lower: 3., upper: 4.}
            ],
        );

        let train_label_nets = Arc::new(TrainLabelNets::new(
            vec![select_motifs_run_id],
            loader,
            run_configs,
            &ws,
            &store,
            Arc::new(RwLock::new(mock_run_ctx::default(1))),
            Arc::new(OptConfig::new(10, 3))
        ).await?);


        let input = Arc::new(DevicesWrap::wrap(
            ResourcesConfigInput::new("test".into(), NetResources::new(Labeled::new(train_group.to_string()), Labeled::new("valid_group".to_string()))),
            0
        ));

        train_label_nets.run(input, Arc::new(RwLock::new(mock_run_ctx::default(1)))).await?;

        Ok(())
    }
}