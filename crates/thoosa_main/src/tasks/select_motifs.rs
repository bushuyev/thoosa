use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use async_trait::async_trait;
use futures_util::{StreamExt, TryStreamExt};
use log::{debug, error, info, trace, warn};
use parking_lot::RwLock;
use tch::Device::Cpu;
use tch::Kind::{Float, Uint8};
use tch::{IndexOp, Kind, Tensor};
use tch::kind::FLOAT_CUDA;
use tokio::io::AsyncWriteExt;
use tokio::task::{JoinHandle, JoinSet};
use caching_runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use caching_runner::adjusters::permits::{PermitsAdjustContextImpl, PermitsAdjuster, PermitsAndDumpToCpuAdjustableContex};
use caching_runner::adjusters::splits::{OutputBuilder, SplitsAdjuster, SplitsAdjusterContextImpl};
use caching_runner::common::cachable::{Cached, DevicesWrap, WithDataRef, WithSelectDataRef};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage, SyncStage};
use caching_runner::global_ctx::GlobalCtx;
use caching_runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunCtxContext, NoopRunnableContext, OOMHandler};
use caching_runner::permits::{PermitStep, MAX_PERMITS};
use caching_runner::tmp_cache::TmpCached;
use thoosa_common::bool_iter::BoolTensorIterator;
use thoosa_common::copy_ctx::CopyCtx;
use thoosa_common::cuda_utils::pack_bits_bool;
use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig, SelectMotifsRunConfig};
use thoosa_common::runs::{Labeled, Simple};
use thoosa_common::utils::pad_to;
use thoosa_common::ws::Workspace;
use thoosa_motifs::find::{FixedFindMotifsConfig, SelectMotifsSavableResult};
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Adjustable, DataId, MonoMetrics, SimpleId, WithId};
use thoosa_store::{IdSlice, Store};
use thoosa_with_data::with_data;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::applicables_runner::Output;
use crate::cli::Commands::SelectMotifs;
use crate::tasks::average_euclidean_distance::{AverageDistanceValue, AverageEuclideanDistanceStage};
use crate::tasks::stacked_moitfs_loader::{StackedMotifsInput, StackedMotifsLoader, StackedMotifsOutput, StackedMotifsOutputId};
use crate::tasks::cross_batch_clc::{unfold_and_cat, CrossInput, UnfoldParams};
use crate::tasks::cross_dims::{ChannelsData2, ChannelsMotifsDists, ChannelsMotifsSlices, MotifSlices};
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::dists_tracker::DistsMask;
use crate::tasks::load_motif_and_clc_dist::{DistOutputBuilder, LoadMotifAndClcDistanceStage, MotifSlicesSelector};
use crate::tasks::motif_loader::{MotifLoader, MotifMeta};
use crate::tasks::run_ctx_2::{MotifsCtx, RunCtx2};
use crate::tasks::tensor_loader::RawTensorLoader;
use crate::tasks::resources_motifs_clc_dist::{ClcResourcesMotifsDist, ClcResourcesMotifsDistInput, ClcResourcesMotifsDistOutput, ClcResourcesMotifsDistInputDeviceWrapper, ClcResourcesMotifsDistInputDeviceWrapperSelector};

pub use crate::tasks::resources_motifs_clc_dist::{fold_dists, fold_mask};
use crate::tasks::slices_iter::SlicesIter;

pub struct SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{

    id:SimpleId,
    store: Arc<RwLock<Box<dyn Store>>>,
    ws:Arc<RwLock<Box<dyn Workspace>>>,
    loader:Arc<L>,
    motif_loader:Arc<ML>,
    stacked_motifs_loader:Arc<StackedMotifsLoader<ML, C>>,
    pd:PhantomData<C>,
    run_configs:Vec<SelectMotifsRunConfig>,
    find_motifs_run_ids:Vec<RunId>,
    load_and_clc_avg_dist:Arc<LoadMotifAndClcDistanceStage<ML, AverageEuclideanDistanceStage<C>, C>>,
    clc_resources_motifs_dist:Arc<ClcResourcesMotifsDist<C>>,
    // output_builder: fn(<Self as WithId>::ID) -> Box<dyn OutputBuilder<O = ClcRresourcesMotifsDistOutput>> + Send + Sync>,
    max_batch:i64//TODO to SelectMotifsSFixedConfig
}

impl <L, ML, C> Debug for SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectMotifsStage")
            .field("id", self.id())
            .field("run_configs", &self.run_configs)
            .field("find_motifs_run_ids", &self.find_motifs_run_ids)
            .finish()
    }
}

impl  <L, ML, C> WithId for SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug)]
struct SelectMotifsSavableResultBuilder {
    id:StackedMotifsOutputId,
    output:Vec<JoinHandle<Result<ClcResourcesMotifsDistOutput, ThoosaError>>>,//TODO TempCache
    channels:u32,
    step:usize
}

impl SelectMotifsSavableResultBuilder {
    pub async fn new(id:StackedMotifsOutputId, channels:u32, step:usize, cache_area: u64) -> Self {

        Self {
            id,
            output: vec![],
            channels,
            step
        }
    }
}

impl OutputBuilder for SelectMotifsSavableResultBuilder {
    type O = ClcResourcesMotifsDistOutput;

    fn add_partial_output(&mut self, o: Self::O) {
        debug!("SelectMotifsSavableResultBuilder::add_partial_output: o={o:?}");

       self.output.push (tokio::spawn(async {
           Ok(o.to_cpu().await) //TODO copy_d2h_from_ptr should return Result
       }));
    }

    async fn build(self) -> Result<Self::O, ThoosaError> {

        let mut resource_ids_sizes = vec![];


        let output:Vec<ClcResourcesMotifsDistOutput> = futures::future::try_join_all(self.output).await?
            .into_iter()
            .collect::<Result<Vec<_>, ThoosaError>>()?;

        // partial output contains masks for
        // motif_c0_0, motif_c0_1... motif_c0_m.
        // motif_c1_0, motif_c1_1... motif_c1_n
        // x
        // unfolded resource_partN_0, unfolded resource_partN_1, ...
        //
        //if a chanel has no motifs then all partial outputs will contain None and corresponding vec in channel_masks will be empty.

        let channel_masks_motif_ids = output.into_iter().fold(
            (
                (0..self.channels).map(|c|vec![]).collect::<Vec<Vec<Tensor>>>(),
                (0..self.channels).map(|c|vec![]).collect::<Vec<Vec<DataId>>>()
            ),

            |mut channel_masks, partial_output|{

                debug!("SelectMotifsSavableResultBuilder::build: partial_output={partial_output:?}");

                assert_eq!(partial_output.channel_masks_dists_ids.len() as u32, self.channels);

                for (c, mask_op) in partial_output.channel_masks_dists_ids.into_iter().enumerate(){
                    debug!("SelectMotifsSavableResultBuilder::build: c={c} mask_op={mask_op:?}");

                    if let Some(mask) = mask_op {
                        let mask_t = mask.0.unpack().0;

                        debug!("SelectMotifsSavableResultBuilder::build: c={c} mask_t=\n{mask_t}");

                        channel_masks.0[c].push(mask_t);
                        channel_masks.1[c] = mask.1;//TODO collect dists, needed for dumping plots only
                    }
                };

                resource_ids_sizes.extend(partial_output.resources_iter.resource_ids_sizes);

                channel_masks
            });

        let channel_masks_dists_ids = channel_masks_motif_ids.0.into_iter().zip(channel_masks_motif_ids.1).map(|channel_masks_motif_ids|{
            debug!("SelectMotifsSavableResultBuilder::build: channel_masks_motif_ids={channel_masks_motif_ids:?}");

            if channel_masks_motif_ids.0.is_empty() {
                None
            } else {
                Some(
                    (
                        DistsMask::new(
                            Tensor::concat(&channel_masks_motif_ids.0, 1), //1 dim - resources
                            None,
                            false,
                            None
                        ),
                        channel_masks_motif_ids.1
                    ),
                )
            }

        })
        .collect::<Vec<_>>();

        Ok(ClcResourcesMotifsDistOutput::new(
            self.id.find_motifs_run_id,
            channel_masks_dists_ids,
            SlicesIter::new(self.step, resource_ids_sizes)
        ))
    }
}


impl <L, ML, C> SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{

    pub fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        loader:Arc<L>,
        motif_loader:Arc<ML>,
        run_configs:Vec<SelectMotifsRunConfig>,
        find_motifs_run_ids:Vec<RunId>,
        max_batch: i64,
        #[cfg(test)]
        fail_if_size_bigger_than:Option<usize>
    ) -> Self {
        let load_and_clc_avg_dist = Arc::new(LoadMotifAndClcDistanceStage::new(
            Arc::clone(&motif_loader), // as Arc<dyn MotifLoaderStage<Output = <AverageEuclideanDistanceStage as Stage>::In<'static>>>,
            Arc::new(AverageEuclideanDistanceStage::<C>::new()),
        ));

        let batched_motifs_loader = Arc::new(StackedMotifsLoader::new(Arc::clone(ws), Arc::clone(&motif_loader), Arc::clone(&load_and_clc_avg_dist), Arc::clone(store)));
        let clc_resources_motifs_dist = Arc::new(ClcResourcesMotifsDist::<C>::new(
            ws.read().channels() as usize,
             #[cfg(test)]
             fail_if_size_bigger_than
        ));
        let channels = ws.read().channels();
        Self {
            id: "SelectMotifsStage".into(),
            store: Arc::clone(store),
            ws: Arc::clone(ws),
            loader,
            motif_loader,
            stacked_motifs_loader: batched_motifs_loader,
            run_configs,
            find_motifs_run_ids,
            load_and_clc_avg_dist,
            clc_resources_motifs_dist,
            pd: Default::default(),
            // output_builder: Box::new(move |id| Box::new(SelectMotifsSavableResultBuilder::new(id, channels))),
            max_batch
        }
    }

}


impl <L,ML, C> Stage  for  SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static

{
    type Ctx = C;
    type In<'a> = ResourcesConfigInput<Simple>;
    type Out = SelectMotifsSavableResult;
}

// #[async_trait]
impl <L, ML, C> AsyncStage  for  SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
        let (device_idx, cache_area, ctx_name) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area(), guard.name())
        };

        let resources = input.with_data_ref(0, Arc::clone(&ctx), None,
    |input: &ResourcesConfigInput<Simple>|{//TODO replace with with_data!(device_idx,
                Ok(input
                    .resources_config()
                    .to_resources(&self.store)?
                    .into_iter()
                    .collect::<Vec<_>>())
            }).await?;



        debug!("SelectMotifsStage::run: resources={resources:?}");
        if resources.is_empty() {
            warn!("SelectMotifsStage::run: no labels resources");
            return Err("Nothing to do ".into())
        }


        let num_channels = self.ws.read().channels() as usize;


        let mut stacked_motifs = vec![];

        // Loading motifs into GPU hoping everything fits. If not, data will be dumped to CPU either right here or when clc_stage fails with OOM.
        for find_motifs_run_id in &self.find_motifs_run_ids {

            let find_motifs_run_config = Arc::new(FindMotifsRunConfig::from_properties_map__(
                *find_motifs_run_id,
                &self.store.read().list_run_config_values(*find_motifs_run_id)?,
            )?);

            for select_motifs_run_config in self.run_configs.iter() {


                match Arc::clone(&self.stacked_motifs_loader).run_cached(Arc::new(DevicesWrap::wrap_cpu(
                    StackedMotifsInput::new(
                        Arc::clone(&find_motifs_run_config),
                        Arc::new(select_motifs_run_config.clone())
                    ))),
                    Arc::clone(&ctx)
                ).await {
                    Ok(motif_data) => {
                        stacked_motifs.push((Arc::clone(&find_motifs_run_config), select_motifs_run_config, motif_data));
                    }
                    Err(e) => {
                        //TODO dump to CPU if OOM
                        error!("Can't load select_motifs_run_config: e={e:?} select_motifs_run_config={select_motifs_run_config:?}")
                    }
                }
            }
        }

        let mut t_1_batch: Vec<_> = vec![];
        let mut t_1_batch_resources: Vec<_> = vec![];
        let mut t_1_batch_dim_0_size = 0;

        let mut skip_0_i = 0;

        let num_resources = resources.len();
        let total_size  = resources.iter().map(|resource|resource.length() as usize).sum::<usize>();
        let mut total_processed_size = 0;


        let mut good_motif_ids = vec![];
        for (i_0, resource_0) in resources.iter()/*.flat_map(|label_resources|label_resources.1.iter())*/.enumerate() {
            debug!("SelectMotifsStage::run: on i_0={i_0} resource_0={resource_0:?} num_resources={num_resources}");

            let t_0 = Arc::clone(&self.loader)
                .run_cached(Arc::new(DevicesWrap::wrap_cpu(resource_0.clone())), Arc::clone(&ctx))
                .await?;

            t_1_batch_dim_0_size += t_0.size().dim_1();
            total_processed_size += t_0.size().dim_1();

            t_1_batch.push(t_0);
            t_1_batch_resources.push(resource_0);

            if t_1_batch_dim_0_size < self.max_batch && i_0 < num_resources -1 {
                continue;
            }
            info!("running batch {} length done {}%", t_1_batch.len(), ((total_processed_size as f32/total_size as f32) * 100.0) as u32 );

            let t_1_batch_len = t_1_batch.len();

            let expanded_motif_ids = futures::stream::iter(stacked_motifs.clone())
                .map(|(find_motifs_run_config, select_motifs_run_config, motif_data)| {
                    let t_1_batch = t_1_batch.clone(); //clones Arcs

                    let store = Arc::clone(&self.store);
                    let clc_stage = Arc::clone(&self.clc_resources_motifs_dist);
                    let ctx_clone = Arc::clone(&ctx);
                    debug!("SelectMotifsStage::run: t_0_batch={t_1_batch:?} \nbatched_motifs_output_wrap={motif_data:?}");

                    let t_1_batch_resources = t_1_batch_resources.clone();
                    let resource_ids = t_1_batch_resources
                        .iter()
                        .map(|resource| resource.id())
                        .collect::<Vec<_>>();


                    let clc_resources_input = Arc::new(ClcResourcesMotifsDistInputDeviceWrapper::new(
                        format!("ClcResourcesMotifsDistInputDeviceWrapper: resource_ids={resource_ids:?}").into(),
                        t_1_batch,
                        resource_ids,
                        motif_data
                    ));

                    debug!("SelectMotifsStage::run: clc_resources_input={clc_resources_input:?}");
                    // let output_builder = (self.output_builder)(SimpleId("zzz"));
                    async move {
                        let out_id = Arc::new(SimpleId("zzz")); //self.clc_resources_motifs_dist.out_id(input.id()); //.clone();

                        let required_ids = HashMap::new();

                        let ClcResourcesMotifsDistOutput {
                            select_motifs_run_id: find_motifs_run_config_id,
                            channel_masks_dists_ids,
                            resources_iter,
                            ..
                        } = OOMHandler::new(
                            vec![
                                Adjusters::RunContext(Box::new(PermitsAdjuster::new(MAX_PERMITS, *PermitStep::TWENTY, move |n| GlobalCtx::global().acquire_many(device_idx, n, out_id.clone())))),
                                Adjusters::RunContext(Box::new(DumpToCpu::new())),
                                Adjusters::Input(Box::new(SplitsAdjuster::<ClcResourcesMotifsDistInputDeviceWrapperSelector, i64>::new(1))), //TODO
                            ], None)
                            .run(
                                &clc_stage,
                                Arc::clone(&clc_resources_input),
                                ctx_clone,
                                Contexts::new(
                                    SplitsAdjusterContextImpl::new(t_1_batch_len as i64, SelectMotifsSavableResultBuilder::new(StackedMotifsOutputId::new(
                                        find_motifs_run_config.id.expect("find_motifs_run_config"), select_motifs_run_config.id.expect("select_motifs_run_config")
                                    ), num_channels as u32, find_motifs_run_config.step as usize, cache_area).await),
                                    NoopRunnableContext {},
                                    PermitsAndDumpToCpuAdjustableContex::new(
                                        PermitsAdjustContextImpl::new(ctx_name),
                                        DumpToCpuContextImpl::new(&required_ids),
                                    ),
                                ),
                            )
                            .await?;

                        // let ClcRresourcesMotifsDistOutput {
                        //     select_motifs_run_id: find_motifs_run_config_id,
                        //     channel_masks_dists_ids,
                        //     resources_iter,
                        //     ..
                        // } = clc_resources_input.with_data_ref(
                        //     1,
                        //       Arc::clone(&ctx_clone), None, |v|{
                        //           clc_stage.do_run(v, ctx_clone)
                        //       }
                        // ).await?;


                        debug!("SelectMotifsStage::run: channel_masks_dists_ids={channel_masks_dists_ids:?}");
                        let mut motif_ids = vec![];
                        for channel_mask_dists_ids in channel_masks_dists_ids.into_iter().enumerate() {
                            if let (channel, Some(mask_dists_ids)) = channel_mask_dists_ids {

                                debug!("SelectMotifsStage::run: channel={channel} mask_dists_ids={:?}", mask_dists_ids.1);

                                let (cpu_mask, _) = mask_dists_ids.0.to_cpu().await?.unpack();

                                trace!("SelectMotifsStage::run: channel={channel} cpu_mask=\n{cpu_mask}");

                                trace!("SelectMotifsStage::run: channel={channel} cpu_mask=\n{cpu_mask}");

                                let mut mask_in_channel = BoolTensorIterator::from(&cpu_mask);
                                for motif_id in mask_dists_ids.1 {
                                    let resources_iter= resources_iter.clone();

                                    let mut slices = HashSet::new();
                                    for slice_1 in resources_iter {

                                        if let Some(mask) = mask_in_channel.next() {
                                            trace!("SelectMotifsStage::run: motif_id={motif_id:?} channel={channel} slice_1={slice_1:?} mask={mask}");
                                            if mask {
                                                slices.insert(IdSlice(0, slice_1.0, slice_1.1));
                                            }
                                        } else {
                                            warn!("SelectMotifsStage::run: no mask for motif_id={motif_id:?} channel={channel} slice_1={slice_1:?} ");
                                        }
                                    }

                                    if !slices.is_empty(){
                                        motif_ids.push(motif_id.clone());
                                        store.write().add_slices_to_motif(find_motifs_run_config_id, *motif_id, channel as i64, slices)?;
                                    }
                                }
                            }
                        }

                        Ok::<Vec<DataId>, ThoosaError>(motif_ids)

                    }
                })
                .buffer_unordered(10)
                .filter_map(|v| async {match v{
                    Ok(v) => {
                        Some(v)
                    }
                    Err(e) => {
                        error!("SelectMotifsStage: failed to select slices for {stacked_motifs:?} and {t_1_batch:?}");
                        None
                    }
                }})
                .collect::<Vec<_>>()
                .await;

            good_motif_ids.extend(expanded_motif_ids.into_iter().flat_map(|v|v.into_iter()).map(|motif_id|(*motif_id, 1.0)));//TODO rate


            t_1_batch_dim_0_size = 0;;
            t_1_batch.clear();
            t_1_batch_resources.clear();
        }

        debug!("SelectMotifsStage::run: good_motif_ids={good_motif_ids:?}");

        Ok(Arc::new(DevicesWrap::wrap_cpu(SelectMotifsSavableResult::new(&self.find_motifs_run_ids, self.run_configs[0].clone(), good_motif_ids))))
    }
}


#[cfg(test)]
pub mod tests {
    use std::collections::{HashMap, HashSet};
    use std::path::Path;
    use std::sync::Arc;
    use log::{debug, LevelFilter};
    use log::Level::Debug;
    use log::LevelFilter::{Info, Trace};
    use parking_lot::RwLock;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::{IndexOp, Kind, Tensor};
    use tch::Device::Cuda;
    use tch::Kind::Bool;
    use tokio::fs;
    use caching_runner::adjusters::splits::OutputBuilder;
    use caching_runner::common::cachable::DevicesWrap;
    use caching_runner::common::{mock_run_ctx, MockRunCtx};
    use caching_runner::common::DropIf::IfNotMax;
    use caching_runner::common::mock_run_ctx::build;
    use caching_runner::common::runnable::AsyncStage;
    use caching_runner::global_ctx;
    use caching_runner::global_ctx::{GlobalCtx};
    use caching_runner::permits::MAX_PERMITS;
    use thoosa_common::find_motifs_run_config::{ByLabelVarianceConfig, FindMotifsRunConfig, MotifSelector, SelectMotifsRunConfig};
    use thoosa_common::runs::{Labeled, Simple};
    use thoosa_common::utils::{draw_function_plot, reset_print_options, vec_f32_from, z_norm, z_norm_2};
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_motifs::distance::clc_dist_d2_and_mask;
    use thoosa_motifs::generators::noise::generate_motifs_with_avg_distance;
    use thoosa_motifs::generators::non_overlapping::Interval;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::{cdt_for_dir, gen_motif, MotifKind, TensorIdSliceIdx, ThoosaError};
    use thoosa_shared::ResourceType::NotInGroup;
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::traits::{DataId, SimpleId};
    use thoosa_store::{IdSlice, MockStoreBuilder};
    use thoosa_common::tho_tensor::ThoTensor;
    use crate::tasks::cross_dims::ChannelsData2;
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::dists_tracker::DistsMask;
    use crate::tasks::motif_loader::{mock_motif_loader, MotifMeta};
    use crate::tasks::resources_motifs_clc_dist::ClcResourcesMotifsDistOutput;
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::select_motifs::{fold_dists, fold_mask, SelectMotifsSavableResultBuilder, SelectMotifsStage};
    use crate::tasks::slices_iter::SlicesIter;
    use crate::tasks::stacked_moitfs_loader::StackedMotifsOutputId;
    use crate::tasks::tensor_loader::{mock_tensor_loader};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_1_resource_1_label_1_channel_1_motif() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp/test_single_label_one_motif");//.join(cdt_for_dir());
        fs::create_dir_all(test_dir).await?;

        log::set_max_level(Trace);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 10000)));
        let run_id = 1;

        let resource_a: Resource = Resource::new(1, 1, "resource_a", 10000);

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 1;

        let channel_0_motifs_ids = vec![1];

        let resources_group = "group";

        let (resource_a_t, resource_a_intervals) = resource_tensor(resource_a.length() as usize, &[vec![(30, 100, MotifKind::Pulse)]], 100)?;
        let resources_channels_intervals = [(&resource_a, &[channel_0_motifs_ids.as_slice()][..], &resource_a_intervals)];

        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
                .with_list_run_config_values(
                    run_id,
                    Ok(FindMotifsRunConfig::new(Some(run_id), 1, 100, None, None).to_properties_iter(None).collect()),
                )
                .with_list_motifs([
                    (find_motifs_run_id as RunId, Ok(channel_0_motifs_ids.clone()))
                ])
                .with_list_resources_not_in_group([
                    (resources_group, Ok(vec![resource_a.clone()])),
                ])
                .with_add_slices_to_motif(resources_add_slices_to_motif_expectations(run_id, &channel_0_motifs_ids, &resources_channels_intervals))//<-- assert
                .build()));

        debug!("test_single_label_one_motif: resource_a_t={resource_a_t:?} resource_a_intervals={resource_a_intervals:?}");

        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached([
                    (resource_a.clone(), resource_a_t.shallow_clone()),
                ])
                .finish(),
        );

        let motif_t = motifs_tensor(100, 3, MotifKind::Pulse);


        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (channel_0_motifs_ids[0] as u32, (MotifMeta { channel: 0, find_motifs_run_id: find_motifs_run_id, window: 100 }, motif_t)),
        ]).finish());

        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new_(Some(select_motifs_run_id), MotifSelector::ByLabelVariance(ByLabelVarianceConfig { _dummy: 0 }))],
            vec![find_motifs_run_id],
            1000,
            None,
        ));

        let result = select_motifs_stage.run(
            Arc::new(DevicesWrap::wrap_cpu(ResourcesConfigInput::new("test".into(), Simple::new(NotInGroup(resources_group.into()))))), Arc::new(RwLock::new(mock_run_ctx::default(1))),
        ).await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_2_resources_1_label_1_channel_1_motif() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp/test_single_label_one_motif");//.join(cdt_for_dir());
        fs::create_dir_all(test_dir).await?;

        log::set_max_level(Info);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));
        let run_id = 1;

        let resource_a: Resource = Resource::new(1, 1, "resource_a", 800);
        let resource_b: Resource = Resource::new(2, 1, "resource_b", 600);

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 1;

        let channel_0_motifs_ids = vec![1];

        let resources_group = "group";

        let (resource_a_t, resource_a_intervals) = resource_tensor(resource_a.length() as usize, &[vec![(3, 100, MotifKind::Pulse)]], 100)?;
        let (resource_b_t, resource_b_intervals) = resource_tensor(resource_b.length() as usize, &[vec![(2, 100, MotifKind::Pulse)]], 200)?;
        let resources_channels_intervals = [(&resource_a, &[channel_0_motifs_ids.as_slice()][..], &resource_a_intervals), (&resource_b, &[channel_0_motifs_ids.as_slice()][..],  &resource_b_intervals)];

        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
                .with_list_run_config_values(
                    run_id,
                    Ok(FindMotifsRunConfig::new(Some(run_id), 1, 100, None, None).to_properties_iter(None).collect()),
                )
                .with_list_motifs([
                    (find_motifs_run_id as RunId, Ok(channel_0_motifs_ids.clone()))
                ])
                .with_list_resources_not_in_group([
                    (resources_group, Ok(vec![resource_a.clone(), resource_b.clone()])),
                ])
                .with_add_slices_to_motif(resources_add_slices_to_motif_expectations(run_id, &channel_0_motifs_ids, &resources_channels_intervals))//<-- assert
                .build()));

        // debug!("test_single_label_one_motif: resource_a_t={resource_a_t:?} resource_a_intervals={resource_a_intervals:?}");
        // debug!("test_single_label_one_motif: resource_b_t={resource_b_t:?} resource_b_intervals={resource_b_intervals:?}");

        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached([
                    (resource_a.clone(), resource_a_t.shallow_clone()),
                    (resource_b.clone(), resource_b_t.shallow_clone()),
                ])
                .finish(),
        );

        let motif_t = motifs_tensor(100, 3, MotifKind::Pulse);


        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (channel_0_motifs_ids[0] as u32, (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_t)),
        ]).finish());

        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new_(Some(select_motifs_run_id), MotifSelector::ByLabelVariance(ByLabelVarianceConfig { _dummy: 0 }))],
            vec![find_motifs_run_id],
            1000,
            None
        ));

        let result = select_motifs_stage.run(
            Arc::new(DevicesWrap::wrap_cpu(ResourcesConfigInput::new("test".into(), Simple::new(NotInGroup(resources_group.into()))))), Arc::new(RwLock::new(mock_run_ctx::default(0))),
        ).await?;

        Ok(())
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_1_resource_1_label_1_channel_2_motifs() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp/test_single_label_one_motif");//.join(cdt_for_dir());
        fs::create_dir_all(test_dir).await?;

        log::set_max_level(Info);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));
        let run_id = 1;

        let resource_a: Resource = Resource::new(1, 1, "resource_a", 1000);

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 1;

        let motifs_ids = vec![1, 2];

        let resources_group = "group";

        let (resource_a_t, resource_a_intervals) = resource_tensor(resource_a.length() as usize, &[vec![(1, 100, MotifKind::Pulse), (1, 100, MotifKind::Triangle)]], 300)?;

        let resources_channels_intervals = [(&resource_a, &[motifs_ids.as_slice()][..], &resource_a_intervals)];

        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
                .with_list_run_config_values(
                    run_id,
                    Ok(FindMotifsRunConfig::new(Some(run_id), 1, 100, None, None).to_properties_iter(None).collect()),
                )
                .with_list_motifs([
                    (find_motifs_run_id as RunId, Ok(motifs_ids.clone()))
                ])
                .with_list_resources_not_in_group([
                    (resources_group, Ok(vec![resource_a.clone()])),
                ])
                .with_add_slices_to_motif(resources_add_slices_to_motif_expectations(run_id, &motifs_ids, &resources_channels_intervals))//<-- assert
                .build()));


        debug!("test_single_label_one_motif: resource_a_t={resource_a_t:?} resource_a_intervals={resource_a_intervals:?}");

        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached([
                    (resource_a.clone(), resource_a_t.shallow_clone()),
                ])
                .finish(),
        );

        let motif_0_t = motifs_tensor(100, 3, MotifKind::Pulse);
        let motif_1_t = motifs_tensor(100, 2, MotifKind::Triangle);


        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (motifs_ids[0] as u32, (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_0_t)),
            (motifs_ids[1] as u32, (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_1_t)),
        ]).finish());


        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new_(Some(select_motifs_run_id), MotifSelector::ByLabelVariance(ByLabelVarianceConfig { _dummy: 0 }))],
            vec![find_motifs_run_id],
            1000,
            None
        ));


        let result = select_motifs_stage.run(
            Arc::new(DevicesWrap::wrap_cpu(ResourcesConfigInput::new("test".into(), Simple::new(NotInGroup(resources_group.into()))))), Arc::new(RwLock::new(mock_run_ctx::default(0))),
        ).await?;

        Ok(())
    }

    fn resources_add_slices_to_motif_expectations(run_id: RunId, channel_0_motifs_ids: &[usize], resources_channel_intervals: &[(&Resource, &[&[usize]], &Vec<Vec<Vec<Interval>>>)]) -> Vec<(RunId, u32, i64, HashSet<IdSlice>)> {

        let mut motif_hash = HashMap::new();
        for (resource, channel_0_motifs_ids, channel_intervals) in resources_channel_intervals {

            for (c, motifs_intervals) in channel_intervals.iter().enumerate() {

                for (m_idx, intervals) in motifs_intervals.iter().enumerate() {

                    let motif_id = channel_0_motifs_ids[c][m_idx]; //.get(m_idx).expect(&format!("no motif for index {m_idx}"));
                    let mut motif_intervals = motif_hash.entry((motif_id.clone(), c)).or_insert_with(HashSet::new);

                    for interval in intervals {
                        motif_intervals.insert(IdSlice(0, resource.id(), interval.start as u32));
                    }
                }
            }
        }

        motif_hash.into_iter().map(|(motif_id_channel, v)| (run_id, motif_id_channel.0 as u32, motif_id_channel.1 as i64, v)).collect::<Vec<_>>()

    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_3_resources_2_labels_3_channels_3_motifs_cuda() -> Result<(), ThoosaError> {
        log::set_max_level(LevelFilter::Debug);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 3, 100, 10000)));

        let resource_a: Resource = Resource::new(1, 1, "resource_a", 900);
        let resource_b: Resource = Resource::new(2, 1, "resource_b", 800);
        let resource_c: Resource = Resource::new(3, 1, "resource_c", 600);

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 2;

        let channel_0_motif_0 = 1;
        let channel_0_motif_1 = 2;
        let channel_1_motif_0 = 3;

        let channel_0_motifs_ids = vec![channel_0_motif_0, channel_0_motif_1];
        let channel_1_motifs_ids = vec![channel_1_motif_0];
        let channel_2_motifs_ids:Vec<usize> = vec![];

        let resources_group = "group";

        let (resource_a_t, resource_a_intervals) = resource_tensor(resource_a.length() as usize, &[
            vec![(3, 100, MotifKind::Pulse)],
            vec![(2, 100, MotifKind::Sin)],
            vec![]
        ], 100)?;
        let (resource_b_t, resource_b_intervals) = resource_tensor(resource_b.length() as usize, &[
            vec![(2, 100, MotifKind::Pulse)],
            vec![(1, 100, MotifKind::Sin)],
            vec![]
        ], 100)?;
        let (resource_c_t, resource_c_intervals) = resource_tensor(resource_c.length() as usize, &[
            vec![(2, 100, MotifKind::Triangle)],
            vec![(1, 100, MotifKind::Sin)],
            vec![]
        ], 100)?;

        let resource_ab_channel_0_motifs = [channel_0_motif_0];
        let resource_c_channel_0_motifs = [channel_0_motif_1];
        let resource_abc_channel_1_motifs = [channel_1_motif_0];
        let resources_channels_intervals = [
            (&resource_a, &[&resource_ab_channel_0_motifs[..], &resource_abc_channel_1_motifs[..], &[][..]][..], &resource_a_intervals),
            (&resource_b, &[&resource_ab_channel_0_motifs[..], &resource_abc_channel_1_motifs[..], &[][..]][..], &resource_b_intervals),
            (&resource_c, &[&resource_c_channel_0_motifs[..], &resource_abc_channel_1_motifs[..], &[][..]][..], &resource_c_intervals),
        ];

        debug!("test_3_resources_2_labels_3_channels_3_motifs: resources_channels_intervals={resources_channels_intervals:?}");

        let expected_add_slices_to_motif_calls = resources_add_slices_to_motif_expectations(find_motifs_run_id, &channel_0_motifs_ids, &resources_channels_intervals);
        debug!("test_3_resources_2_labels_3_channels_3_motifs: expected_add_slices_to_motif_calls={expected_add_slices_to_motif_calls:?}");

        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
                .with_list_run_config_values(
                    find_motifs_run_id,
                    Ok(FindMotifsRunConfig::new(Some(find_motifs_run_id), 1, 100, None, None).to_properties_iter(None).collect()),
                )
                .with_list_motifs([
                    (find_motifs_run_id as RunId, Ok(vec![
                        channel_0_motif_0,
                        channel_0_motif_1,
                        channel_1_motif_0
                    ]))
                ])
                .with_list_resources_not_in_group([
                    (resources_group, Ok(vec![resource_a.clone(), resource_b.clone(), resource_c.clone()])),
                ])
                .with_add_slices_to_motif(expected_add_slices_to_motif_calls)//<-- assert
                .build()));


        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached([
                    (resource_a.clone(), resource_a_t),
                    (resource_b.clone(), resource_b_t),
                    (resource_c.clone(), resource_c_t),
                ])
                .finish(),
        );

        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (channel_0_motif_0 as u32, (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motifs_tensor(100, 3, MotifKind::Pulse))),
            (channel_0_motif_1 as u32, (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motifs_tensor(100, 2, MotifKind::Triangle))),
            (channel_1_motif_0 as u32, (MotifMeta { channel: 1, find_motifs_run_id, window: 100 }, motifs_tensor(100, 4, MotifKind::Sin))),
        ]).finish());


        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new_(Some(select_motifs_run_id), MotifSelector::ByLabelVariance(ByLabelVarianceConfig { _dummy: 0 }))],
            vec![find_motifs_run_id],
            10000,//TODO fails with max_batch 1000
            Some(801)//TODO check if ran through OOMHandler
        ));


        let result = select_motifs_stage.run(
            Arc::new(DevicesWrap::wrap_cpu(ResourcesConfigInput::new("test".into(), Simple::new(NotInGroup(resources_group.into()))))), Arc::new(RwLock::new(mock_run_ctx::default(1))),
        ).await?;

        Ok(())
    }

    ///found motif's slices of same kind
    //TODO make slices a bit different
    pub fn motifs_tensor(motif_len:usize, num_slices: usize, motif_kind: MotifKind) -> Tensor {


        let t = Tensor::stack(
            generate_motifs_with_avg_distance(
                gen_motif(motif_len, &motif_kind).as_slice(), num_slices, 0.3, 0.4, None
            ).0.into_iter().map(|motif| Tensor::from_slice(motif.as_slice())).collect::<Vec<_>>().as_slice(), 0
        );

        debug!("motifs_tensor: t.size={:?}", t.size());

        t
    }

    ///channels_motif_kinds -  channels x [(how many times motif should appear in the sequence, motif length , MotifKind)]
    ///returns Tensor [channels x resource_len] + channels[ motifs[ intervals[Interval] ] ]
    pub fn resource_tensor(resource_len:usize, channels_motif_kinds:&[Vec<(usize, usize, MotifKind)>], gap: usize) -> Result<(Tensor, Vec<Vec<Vec<Interval>>>), ThoosaError> {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        use rand::seq::SliceRandom;
        const STEP_SIGMA: f32 = 0.4;

        fn normal_step(rng: &mut StdRng, sigma: f32) -> f32 {
            let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
            let u2: f32 = rng.random::<f32>();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f32::consts::TAU * u2;
            sigma * r * theta.cos()
        }

        fn distribute_slack(rng: &mut StdRng, segments: usize, slack: usize) -> Vec<usize> {
            if segments == 0 {
                return Vec::new();
            }

            if slack == 0 {
                return vec![0; segments];
            }

            let mut offsets = vec![0usize; segments];
            let mut weights: Vec<f32> = (0..segments).map(|_| rng.random::<f32>().max(f32::MIN_POSITIVE)).collect();
            let weight_sum: f32 = weights.iter().sum();

            if !weight_sum.is_finite() || weight_sum <= f32::MIN_POSITIVE {
                offsets[0] = slack;
                return offsets;
            }

            let mut allocated = 0usize;
            for (offset, weight) in offsets.iter_mut().zip(weights.iter()) {
                let share = ((*weight / weight_sum) * slack as f32).floor() as usize;
                *offset = share;
                allocated = allocated.saturating_add(share);
            }

            let allocated = allocated.min(slack);
            let remainder = slack.saturating_sub(allocated);

            if remainder > 0 {
                let mut indices: Vec<usize> = (0..segments).collect();
                indices.shuffle(rng);
                for idx in indices.into_iter().cycle().take(remainder) {
                    offsets[idx] += 1;
                }
            }

            offsets
        }

        let mut channels_signals = Vec::with_capacity(channels_motif_kinds.len());
        let mut channels_intervals = Vec::with_capacity(channels_motif_kinds.len());

        let channels_count = channels_motif_kinds.len().max(1);

        for (channel_idx, channel_kinds) in channels_motif_kinds.iter().enumerate() {
            let total_repeats: usize = channel_kinds.iter().map(|(repeats, _, _)| *repeats).sum();

            let mut rng = StdRng::from_os_rng();
            let mut signal = vec![0f32; resource_len];
            for idx in 1..resource_len {
                signal[idx] = signal[idx - 1] + normal_step(&mut rng, STEP_SIGMA);
            }

            if total_repeats == 0 {
                channels_signals.push(signal);
                channels_intervals.push(channel_kinds.iter().map(|_| Vec::new()).collect());
                continue;
            }

            let total_motif_space = channel_kinds.iter().try_fold(0usize, |acc, (repeats, motif_len, _)| {
                let motif_space = motif_len.checked_mul(*repeats).ok_or_else(|| ThoosaError::new("motif length overflow"))?;
                acc.checked_add(motif_space).ok_or_else(|| ThoosaError::new("motif placement overflows usize"))
            })?;
            let total_gap_space = gap.checked_mul(total_repeats.saturating_sub(1)).ok_or_else(|| ThoosaError::new("motif gap sizing overflows usize"))?;
            let required_space = total_motif_space.checked_add(total_gap_space).ok_or_else(|| ThoosaError::new("motif layout exceeds addressable range"))?;

            if required_space > resource_len {
                return Err(ThoosaError::new(format!(
                    "cannot place {total_repeats} motifs with min_gap={gap} into series len={resource_len}"
                )));
            }

            let extra_space = resource_len - required_space;
            let channel_start = resource_len * channel_idx / channels_count;
            let channel_end = resource_len * (channel_idx + 1) / channels_count;
            let segment_len = channel_end.saturating_sub(channel_start);
            let use_segment = segment_len >= required_space;
            let slack = if use_segment {
                segment_len - required_space
            } else {
                extra_space
            };
            let start_bound = if use_segment { channel_start } else { 0 };
            let end_bound = if use_segment { channel_end } else { resource_len };

            let segment_offsets = distribute_slack(&mut rng, total_repeats + 1, slack);
            let mut current = start_bound
                .checked_add(segment_offsets.get(0).copied().unwrap_or(0))
                .ok_or_else(|| ThoosaError::new(format!(
                    "motif placement exceeds resource length for channel {channel_idx}"
                )))?;
            let mut channel_motif_intervals = Vec::with_capacity(channel_kinds.len());
            let mut placed = 0usize;

            for (repeats, motif_len, kind) in channel_kinds.iter() {
                if *repeats == 0 || *motif_len == 0 {
                    channel_motif_intervals.push(Vec::new());
                    continue;
                }

                let motifs = generate_motifs_with_avg_distance(
                    gen_motif(*motif_len, kind).as_slice(),
                    (*repeats).max(2),
                    0.1,
                    0.2,//should be lower than in motif where it's 0.2-0.3, TODO move to params
                    None,
                ).0;

                let mut motif_intervals = Vec::with_capacity(*repeats);

                for motif in motifs.into_iter().take(*repeats) {
                    let start = current;
                    let end = start.checked_add(*motif_len).ok_or_else(|| ThoosaError::new(format!(
                        "motif placement exceeds resource length for channel {channel_idx}"
                    )))?;
                    if end > end_bound {
                        return Err(ThoosaError::new(format!(
                            "motif placement exceeds resource length for channel {channel_idx}"
                        )));
                    }

                    for (offset, value) in motif.iter().enumerate() {
                        signal[start + offset] = *value;
                    }

                    motif_intervals.push(Interval { start, end });
                    placed += 1;
                    current = end;
                    if placed < total_repeats {
                        let extra_gap = segment_offsets.get(placed).copied().unwrap_or(0);
                        let spacing = gap.checked_add(extra_gap).ok_or_else(|| ThoosaError::new("motif gap sizing overflows usize"))?;
                        current = current.checked_add(spacing).ok_or_else(|| ThoosaError::new(format!(
                            "motif placement exceeds resource length for channel {channel_idx}"
                        )))?;
                    }
                }

                channel_motif_intervals.push(motif_intervals);
            }

            // ensure we push placeholders for motifs that had zero repeats but were skipped
            while channel_motif_intervals.len() < channel_kinds.len() {
                channel_motif_intervals.push(Vec::new());
            }

            channels_signals.push(signal);
            channels_intervals.push(channel_motif_intervals);
        }

        let t = Tensor::stack(channels_signals.into_iter().map(|data|Tensor::from_slice(data.as_slice())).collect::<Vec<Tensor>>().as_slice(), 0);

        debug!("resource_tensor: t.size={:?} intervals={channels_intervals:?}", t.size());

        Ok((t, channels_intervals))
    }

    #[test]
    fn resource_tensor_supports_multiple_kinds_per_channel() -> Result<(), ThoosaError> {
        let (tensor, intervals) = resource_tensor(
            1000,
            &[vec![(1, 50, MotifKind::Pulse), (1, 60, MotifKind::Sin)], vec![(2, 80, MotifKind::Triangle)]], 300,
        )?;

        assert_eq!(tensor.size()[0], 2i64);
        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0].len(), 2);
        assert_eq!(intervals[1].len(), 1);
        assert_eq!(intervals[0][0].len(), 1);
        assert_eq!(intervals[0][1].len(), 1);
        assert_eq!(intervals[1][0].len(), 2);
        let first_pulse = intervals[0][0].last().unwrap();
        let first_sin = intervals[0][1].first().unwrap();
        assert!(first_pulse.end <= first_sin.start);

        Ok(())
    }

    #[test]
    fn resource_tensor_offsets_channels_when_space_allows() -> Result<(), ThoosaError> {
        let (_, intervals) = resource_tensor(
            1000,
            &[vec![(1, 100, MotifKind::Pulse)], vec![(1, 120, MotifKind::Triangle)]], 300,
        )?;

        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0].len(), 1);
        assert_eq!(intervals[1].len(), 1);
        assert_eq!(intervals[0][0].len(), 1);
        assert_eq!(intervals[1][0].len(), 1);
        let first_channel_interval = &intervals[0][0][0];
        let second_channel_interval = &intervals[1][0][0];
        assert!(first_channel_interval.end <= second_channel_interval.start);

        Ok(())
    }

    #[test]
    fn test_fold_mask(){
        let mask = Tensor::from_slice(&[
            false, true,   false, false, false, //1
            false, true,   false, false, false, //1
            false, false,  false, false, false, //1

            false, false,  true,  false, false, //2
            false, false,  false, false, false, //2
        ] ).view((1, 5, -1));
        debug!("mask: \n{mask}");


        let ids = vec![1, 1, 1, 2, 2];

        let folded_mask = fold_mask(mask, &ids, 1);

        assert_eq!(
            Tensor::from_slice(&[
                false, true,   false, false, false, //1
                false, false,  true,  false, false, //2
            ] ).view((1, 2, -1)),
            folded_mask,
        );

    }

    #[test]
    fn test_fold_mask_two_dims(){
        let mask_0 = Tensor::from_slice(&[
            false, true,   false, false, false, //1
            false, true,   false, false, false, //1
            false, false,  false, false, false, //1

            false, false,  true,  false, false, //2
            false, false,  false, false, false, //2
        ] ).view((1, 5, -1));
        let mask_1 = Tensor::from_slice(&[
            false, false,  true,  false, false, //1
            false, false,  true,  false, false, //1
            false, false,  false, false, false, //1

            false, false,  false,  false, false, //2
            false, true,  false, false, false, //2
        ] ).view((1, 5, -1));
        let mask = Tensor::cat(&[mask_0, mask_1], 0);
        debug!("mask: \n{mask}");


        let ids = vec![1, 1, 1, 2, 2];

        let folded_mask = fold_mask(mask, &ids, 1);

        assert_eq!(
            Tensor::from_slice(&[
                false, true,   false, false, false, //1
                false, false,  true,  false, false, //2
            ] ).view((1, 2, -1)),
            folded_mask.narrow(0, 0, 1),
        );

        assert_eq!(
            Tensor::from_slice(&[
                false, false,  true, false, false, //1
                false, true,   false,  false, false, //2
            ] ).view((1, 2, -1)),
            folded_mask.narrow(0, 1, 1),
        );

    }



    #[test]
    fn test_fold_mask_all_false() {
        let mask = Tensor::zeros([1, 4, 3], (Kind::Bool, tch::Device::Cpu));
        let ids = vec![7, 7, 8, 8];
        let folded = fold_mask(mask, &ids, 1);
        let expected = Tensor::zeros([1, 2, 3], (Kind::Bool, tch::Device::Cpu));
        assert_eq!(folded, expected);
    }

    #[test]
    fn test_fold_dists_basic_avg_rows() {
        // shape: [1, 5, 3], fold along dim=1 with ids = [1,1,1,2,2]
        // Group 1 (rows 0..2) avg -> [3,3,3]
        // Group 2 (rows 3..4) avg -> [5,5,10]
        let x = Tensor::from_slice(&[
            // row 0
            1.0f32, 2.0, 3.0,
            // row 1
            3.0, 2.0, 1.0,
            // row 2
            5.0, 5.0, 5.0,
            // row 3
            10.0, 0.0, 10.0,
            // row 4
            0.0, 10.0, 10.0,
        ])
            .view([1, 5, 3]);

        let ids = vec![1, 1, 1, 2, 2];

        let folded = fold_dists(x, &ids, 1);
        let expected = Tensor::from_slice(&[
            // group 1 avg:
            3.0f32, 3.0, 3.0,
            // group 2 avg:
            5.0, 5.0, 10.0,
        ])
            .view([1, 2, 3]);

        assert_eq!(folded, expected);
    }

    #[test]
    fn test_fold_dists_last_dim_avg_cols() {
        // Fold along the last dimension (dim=2). shape: [2, 2, 4]
        // ids = [10,10,20,20] -> two groups of contiguous columns
        let x = Tensor::from_slice(&[
            // batch 0
            // row 0
            1.0f32, 3.0, 5.0, 7.0,
            // row 1
            2.0, 4.0, 6.0, 8.0,
            // batch 1
            // row 0
            10.0, 0.0, 10.0, 0.0,
            // row 1
            0.0, 10.0, 20.0, 30.0,
        ]).view([2, 2, 4]);

        let ids = vec![10, 10, 20, 20];

        let folded = fold_dists(x, &ids, 2);

        // For batch 0:
        // row0: [1,3,5,7] -> A=(1+3)/2=2, B=(5+7)/2=6
        // row1: [2,4,6,8] -> A=3, B=7
        //
        // For batch 1:
        // row0: [10,0,10,0] -> A=5, B=5
        // row1: [0,10,20,30] -> A=5, B=25
        let expected = Tensor::from_slice(&[
            // batch 0
            2.0f32, 6.0, // row 0 -> [A,B]
            3.0, 7.0,    // row 1 -> [A,B]
            // batch 1
            5.0, 5.0,    // row 0
            5.0, 25.0,   // row 1
        ])
            .view([2, 2, 2]);

        assert_eq!(folded, expected);
    }

    #[test]
    fn test_fold_mask_bool_or_contiguous() {
        // sanity check for boolean folding using same machinery
        let mask = Tensor::from_slice(&[
            false, true,   false, false, false, //1
            false, true,   false, false, false, //1
            false, false,  false, false, false, //1
            false, false,  true,  false, false, //2
            false, false,  false, false, false, //2
        ])
            .view([1, 5, 5]);

        let ids = vec![1, 1, 1, 2, 2];

        let folded_mask = fold_mask(mask, &ids, 1);

        let expected = Tensor::from_slice(&[
            false, true,  false, false, false, // group 1 (OR of first 3)
            false, false, true,  false, false, // group 2 (OR of last 2)
        ])
            .view([1, 2, 5]);

        assert_eq!(folded_mask, expected);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn heavy_test_oom_cuda() -> Result<(), ThoosaError> {
        log::set_max_level(LevelFilter::Debug);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 3, 100, 10000)));
        let run_id = 1;


        let resource_len = 100_000;
        let resources_label_0 = (0..100).map(|i|Resource::new(i, 1, format!("resource_l0_{i}"), resource_len)).collect::<Vec<_>>();
        let resources_label_1 = (0..100).map(|i|Resource::new(i, 1, format!("resource_l1_{i}"), resource_len)).collect::<Vec<_>>();
        let resources_x = (0..100).map(|i|Resource::new(i, 1, format!("resource_x_{i}"), resource_len)).collect::<Vec<_>>();

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 1;

        let motif_ids = (0..10000usize).collect::<Vec<_>>();


        let resources_group = "group";


        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
                .with_list_run_config_values(
                    run_id,
                    Ok(FindMotifsRunConfig::new(Some(run_id), 10, 100, None, None).to_properties_iter(None).collect()),
                )
                .with_list_motifs([
                    (find_motifs_run_id as RunId, Ok(motif_ids.clone()))
                ])
                .with_list_group_labeled_resources_sets([
                    (resources_group, Ok(vec![
                        ("Lable_0".to_string(), "resource_set_for_label_0".to_string()),
                        ("Lable_1".to_string(), "resource_set_for_label_1".to_string())
                    ]))
                ])
                .with_list_resource_set_resources_by_descr([
                    ("resource_set_for_label_0", Ok(resources_label_0.clone())),
                    ("resource_set_for_label_1", Ok(resources_label_1.clone()))
                ])
                .with_list_resources_not_in_group([
                    (resources_group, Ok(resources_x)),
                ])

                .with_add_slices_to_motif_at_least_once()//fails because builder is not implemented
                .build()));


        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached(
                    resources_label_0.iter().chain(resources_label_1.iter()).map(|resouce|{
                        (resouce.clone(),
                            resource_tensor(resource_len as usize, &[
                                vec![(3, 100, MotifKind::Pulse)],
                                vec![(3, 100, MotifKind::Sin)],
                                vec![]
                            ], 100).unwrap().0
                        )
                    })
                )
                .finish(),
        );

        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached(
            motif_ids.iter().map(|m|(*m as u32, (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motifs_tensor(100, 3, MotifKind::Pulse))))
        ).finish());


        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new_(Some(select_motifs_run_id), MotifSelector::ByLabelVariance(ByLabelVarianceConfig { _dummy: 0 }))],
            vec![find_motifs_run_id],
            30_000_000,//slices
            None
        ));

        let ctx =         build()
            .with_device_idx(1)
            .with_acquired_permits(MAX_PERMITS)
            .with_add_permits()
            .with_drop_permits(vec![IfNotMax])
            .with_cache_area(0)
            .finish();

        let result = select_motifs_stage.run(
            Arc::new(DevicesWrap::wrap_cpu(ResourcesConfigInput::new("test".into(), Simple::new(NotInGroup(resources_group.into()))))), Arc::new(RwLock::new(ctx)),
        ).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_output_builder() -> Result<(), ThoosaError>  {
        let num_channels = 3;
        let mut output_builder = Box::new(SelectMotifsSavableResultBuilder::new(StackedMotifsOutputId::new(0,0), num_channels as u32, 50, 0).await);

        let mask_0 = Tensor::rand(&[3, 5, 7], FLOAT_CUDA).lt(0.5);
        output_builder.add_partial_output(
            ClcResourcesMotifsDistOutput::new(
                0,
                vec![
                    Some(
                        (
                            DistsMask::new(
                                //3 channels, 5 motifs, 7 resource slices
                                mask_0,
                                None,
                                false,
                                None,
                            ),
                            vec![
                                0.into()
                            ]
                        )
                    ),
                    None,
                    None
                ],
                SlicesIter::new(100, vec![
                    (0u32, 100),
                    (1u32, 100)
                ])
            )
        );

        let mask_1 = Tensor::rand(&[3, 5, 7], FLOAT_CUDA).lt(0.5);
        output_builder.add_partial_output(
            ClcResourcesMotifsDistOutput::new(
                0,
                vec![
                    Some(
                        (
                            DistsMask::new(
                                //3 channels, 5 motifs, 7 resource slices
                                mask_1,
                                None,
                                false,
                                None,
                            ),
                            vec![
                                0.into()
                            ]
                        )
                    ),
                    None,
                    None
                ],
                SlicesIter::new(100, vec![
                    (0u32, 100),
                    (1u32, 100)
                ])
            )
        );

        let ClcResourcesMotifsDistOutput {
            channel_masks_dists_ids,
            ..
        } = output_builder.build().await?;

        for masks_dists_ids in channel_masks_dists_ids {
            match masks_dists_ids {
                None => {}
                Some(v) => {
                    let (mask, _) = v.0.unpack();
                    debug!("mask=\t{mask}");//TODO check mask - should be concat mask_0  mask_1
                }
            }
        }


        Ok(())
    }

}

