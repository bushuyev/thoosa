use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use async_trait::async_trait;
use futures_util::{StreamExt, TryStreamExt};
use log::{debug, error, info, trace, warn};
use parking_lot::RwLock;
use tch::Device::Cpu;
use tch::Kind::Float;
use tch::{IndexOp, Kind, Tensor};
use tokio::io::AsyncWriteExt;
use caching_runner::common::cachable::{with_two_arrays_ext, with_two_vars_ext, Cached, DevicesWrap};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage};
use caching_runner::tmp_cache::TmpCached;
use thoosa_common::applicables::ThoDevice;
use thoosa_common::bool_iter::BoolTensorIterator;
use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig, SelectMotifsRunConfig};
use thoosa_common::runs::Labeled;
use thoosa_common::utils::pad_to;
use thoosa_common::ws::Workspace;
use thoosa_motifs::distance::MaskMaker::{Le, LocalMin};
use thoosa_motifs::find::{FixedFindMotifsConfig, SelectMotifsSavableResult};
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Adjustable, DataId, SimpleId, WithId};
use thoosa_shims::stream::{StreamGuard, ThoCudaStream};
use thoosa_store::{IdSlice, Store};
use thoosa_with_data::with_data;
use torch_runner::data::ThoTensor2;
use crate::cli::Commands::SelectMotifs;
use crate::tasks::average_euclidean_distance::{AverageDistanceValue, AverageEuclideanDistanceStage};
use crate::tasks::cross_batch_clc::{unfold_and_cat, CrossInput, UnfoldParams};
use crate::tasks::cross_dims::{ChannelsData2, ChannelsMotifsDists, ChannelsMotifsSlices, MotifSlices};
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::dists_tracker::dist_iter;
use crate::tasks::load_motif_and_clc_dist::LoadMotifAndClcDistanceStage;
use crate::tasks::motif_loader::{MotifLoader, MotifMeta};
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::slices_iter::SlicesIter;
use crate::tasks::tensor_loader::RawTensorLoader;

#[derive(Debug)]
pub struct SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{

    store: Arc<RwLock<Box<dyn Store>>>,
    ws:Arc<RwLock<Box<dyn Workspace>>>,
    loader:Arc<L>,
    motif_loader:Arc<ML>,
    pd:PhantomData<C>,
    run_configs:Vec<SelectMotifsRunConfig>,
    find_motifs_run_ids:Vec<RunId>,
    load_and_clc_avg_dist:Arc<LoadMotifAndClcDistanceStage<ML, AverageEuclideanDistanceStage<C>, C>>
}

impl <L, ML, C> SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{

    pub fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        loader:Arc<L>,
        motif_loader:Arc<ML>,
        run_configs:Vec<SelectMotifsRunConfig>,
        find_motifs_run_ids:Vec<RunId>,
    ) -> Self {
        let load_and_clc_avg_dist = Arc::new(LoadMotifAndClcDistanceStage::new(
            Arc::clone(&motif_loader), // as Arc<dyn MotifLoaderStage<Output = <AverageEuclideanDistanceStage as Stage>::In<'static>>>,
            Arc::new(AverageEuclideanDistanceStage::<C>::new()),
        ));

        Self {
            store: Arc::clone(store),
            ws: Arc::clone(ws),
            loader,
            motif_loader,
            pd: Default::default(),
            run_configs,
            find_motifs_run_ids,
            load_and_clc_avg_dist
        }
    }

}


impl <L,ML, C> Stage  for  SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static

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
    C: RunCtx + Adjustable + 'static
{
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
        let (device_idx, cache_area) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area())
        };

        let labels_resources = input.with_data_ref(0, None, |input|{
            Ok(input
                .resources_config()
                .to_resources(&self.store)?
                .into_iter()
                .map(|r| Arc::new(r))
                .collect::<Vec<_>>())
        }).await?;
        let max_batch = 1000;//TODO

        debug!("SelectMotifsStage::run: resources={labels_resources:?}");



        let tmp_moitfs_cache:TmpCached<DataWithExtra<ThoTensor2<ChannelsMotifsSlices, SimpleId>, (Vec<Vec<DataId>>, FindMotifsRunConfig, SelectMotifsRunConfig)>> = TmpCached::new("motif slices", cache_area).await;
        let tmp_motifs_avg_dists_cache:TmpCached<ThoTensor2<ChannelsMotifsDists, SimpleId>> = TmpCached::new("motif dists", cache_area).await;

        let num_channels = self.ws.read().channels();
        let mut motif_avg_dists = Arc::new(parking_lot::RwLock::new(HashMap::new()));

        for find_motifs_run_id in &self.find_motifs_run_ids {

            let find_motifs_run_config = FindMotifsRunConfig::from_properties_map__(
                *find_motifs_run_id,
                &self.store.read().list_run_config_values(*find_motifs_run_id)?,
            )?;

            for select_motifs_run_config in self.run_configs.iter() {
                let motif_ids = self.store.read().list_motifs(*find_motifs_run_id)?;
                let motifs_len = motif_ids.len();
                let done_cnt = Arc::new(AtomicU32::new(0));

                let loaded_motifs = futures::stream::iter(motif_ids)
                    .map( |motif_id| {
                        let done_cnt = Arc::clone(&done_cnt);
                        let motif_loader_clone = Arc::clone(&self.motif_loader);
                        let load_and_clc_avg_dist_clone = Arc::clone(&self.load_and_clc_avg_dist);
                        let ctx = Arc::clone(&ctx);
                        let ctx_for_clc = Arc::clone(&ctx);
                        let motif_avg_dists = Arc::clone(&motif_avg_dists);

                        async move {
                            let motif_id_wrap = Arc::new(DevicesWrap::wrap_cpu((motif_id as u32).into()));
                            let mut motif = motif_loader_clone.run_cached(Arc::clone(&motif_id_wrap), ctx).await?;



                            let avg_motif_dist = load_and_clc_avg_dist_clone
                                .run_cached(Arc::clone(&motif_id_wrap), ctx_for_clc)
                                .await?
                                .with_data_ref(0, None, |dist_val: &AverageDistanceValue| Ok(dist_val.value()))
                                .await?;

                            debug!("SelectMotifsStage::run: find_motifs_run_id={find_motifs_run_id} run_config.id={:?} motif={motif:?} avg_motif_dist={avg_motif_dist}", select_motifs_run_config.id);

                            motif_avg_dists.write().insert(DataId(motif_id as u32), avg_motif_dist);


                            Ok::<_, ThoosaError>(motif)
                        }
                    })
                    .buffer_unordered(10)
                    .try_collect::<Vec<_>>()
                    .await?;

                debug!("SelectMotifsStage::run: loaded_motifs={loaded_motifs:?}");

                //building tensor with all motifs' slices (ChannelsMotifSlices) found by current find_motifs_run_id, storing them in tmp_cache. Channels where number of motifs is less than max slices are padded with zeros.
                // channels x motif_0_slice_0...motif_0_slice_last + motif_1_slice_0...motif_1_slice_last + ...
                with_data!(device_idx, [loaded_motifs];|data:&[&DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>]|{


                    let channels_motifs = data.into_iter().fold(

                        (0..num_channels).map(|_|vec![]).collect::<Vec<_>>(),//vec for each channel

                        |mut acc, item| {

                            assert_eq!(
                                find_motifs_run_config.window, item.extra().window,
                                "window is not same: find_motifs_run_config.window={} item.extra().window={}item.extra().window, motif_id={:?}", find_motifs_run_config.window, item.extra().window, item.data().id()
                            );

                            acc[item.extra().channel as usize].push(item.data());
                        acc
                    });

                    let max_motif_slices = channels_motifs
                        .iter()
                        .map(|v|
                            v.iter()
                            .map(|t|t.size()[0] as usize)//num motif slices known for each motif
                            .sum::<usize>())
                        .max().unwrap_or(0);

                    // let empty_slice = Tensor::zeros(&[find_motifs_run_config.window as i64], (Float, ThoDevice::from(device_idx).into()));

                    // let dummy_slice
                    let mut channels_motif_ids = vec![Vec::<DataId>::new(); num_channels as usize];
                    let mut channels_motifs_tensors_vec = vec![];
                    let mut channels_motif_avg_dists_tensors_vec = vec![];
                    for c in 0..num_channels {

                        let channel_motifs = &channels_motifs.get(c as usize).expect(&format!("not enough channels: len={} c={}", channels_motifs.len(), c));

                        debug!("SelectMotifsStage::run: channel {c} max_motif_slices={max_motif_slices} channel_motifs={channel_motifs:?}");

                        let channel_motifs_tensors = channel_motifs.iter().map(|motif|  {
                            debug!("SelectMotifsStage::run: motif={motif:?} id={}", motif.id());
                            for _ in 0..motif.size()[0] {
                                channels_motif_ids[c as usize].push(motif.id().clone());
                            }

                            motif.t_ref().copy()
                        }).collect::<Vec<_>>();

                        //building dists tensors for each channel to compare matches with
                        let channel_avg_dists_tensors = channel_motifs.iter().map(|motif|  {
                            let read_guard = motif_avg_dists.read();
                            let avg_dist = read_guard.get(&motif.id()).unwrap();

                            debug!("SelectMotifsStage::run: motif.id={} dist={avg_dist}", motif.id());

                            Tensor::from_slice(vec![*avg_dist; motif.size()[0] as usize].as_slice())
                        }).collect::<Vec<_>>();

                        debug!("SelectMotifsStage::run: c={c}, channel_motifs_tensors={:?}", channel_motifs_tensors);

                        let (motifs_t, moitf_avg_dists_t) = if channel_motifs_tensors.is_empty() {
                            (
                                Tensor::zeros(&[max_motif_slices as i64, find_motifs_run_config.window  as i64], (Float, ThoDevice::from(device_idx).into())),
                                Tensor::zeros(&[max_motif_slices as i64], (Float, ThoDevice::from(device_idx).into()))
                            )

                        } else {
                            let channel_t = Tensor::cat(&channel_motifs_tensors, 0);
                            let motif_avg_dists_t = Tensor::cat(&channel_avg_dists_tensors, 0);

                            if channel_t.size()[0] < max_motif_slices as i64 {
                                debug!("SelectMotifsStage::run: c={c}, channel_t.size={:?} padding to {:?}", channel_t.size(), [max_motif_slices as i64, find_motifs_run_config.window  as i64]);
                                (
                                    pad_to(&channel_t, &[max_motif_slices as i64, find_motifs_run_config.window  as i64]).expect("pad to failed"),
                                    pad_to(&motif_avg_dists_t, &[max_motif_slices as i64]).expect("pad to failed")
                                )

                            } else {
                                debug!("SelectMotifsStage::run: c={c}, channel_t.size={:?}", channel_t.size());

                                (channel_t, motif_avg_dists_t)
                            }
                        };

                        channels_motif_avg_dists_tensors_vec.push(moitf_avg_dists_t);
                        channels_motifs_tensors_vec.push(motifs_t);

                    };

                    debug!("SelectMotifsStage::run: channels_dists_tensors_vec={channels_motif_avg_dists_tensors_vec:?}");
                    debug!("SelectMotifsStage::run: channels_motifs_tensors_vec={channels_motifs_tensors_vec:?}");

                    let channels_motifs_t = Tensor::stack(&channels_motifs_tensors_vec, 0);
                    let channels_motif_avg_dists_t = Tensor::stack(&channels_motif_avg_dists_tensors_vec, 0);

                    debug!("SelectMotifsStage::run: num_channels={num_channels}, channels_motif_ids={channels_motif_ids:?} channels_motifs_t.size={:?}", channels_motifs_t.size());
                    trace!("SelectMotifsStage::run: channels_motif_avg_dists_t=\n{channels_motif_avg_dists_t}");
                    trace!("SelectMotifsStage::run: channels_motifs_t=\n{channels_motifs_t}");

                    let find_and_select_motifs_id = SimpleId(format!("{}_{}", find_motifs_run_config.id.expect("no id in find_motifs_run_config"), select_motifs_run_config.id.expect("no id in select_motifs_run_config")).leak());
                    tmp_moitfs_cache.insert(
                        DataWithExtra::new(
                            ThoTensor2::new(
                                find_and_select_motifs_id.clone(),
                                channels_motifs_t
                            ),
                            (channels_motif_ids, find_motifs_run_config.clone(), select_motifs_run_config.clone())
                        )//TODO leak
                    );

                    tmp_motifs_avg_dists_cache.insert(
                        ThoTensor2::new(
                            find_and_select_motifs_id,
                            channels_motif_avg_dists_t
                        )
                    );

                    Ok::<(), ThoosaError>(())
                })?;
            }
        }


        let find_motifs_runs_motifs = tmp_moitfs_cache.clone_to_arcs_vec();


        debug!("SelectMotifsStage::run: motif_dists={motif_avg_dists:?}");

        for label_resources in labels_resources {
            let resources = &label_resources.1;
            //t_0_batch - moitfs
            let mut t_1_batch: Vec<_> = vec![];
            let mut t_1_batch_resources: Vec<_> = vec![];
            let mut t_1_batch_len = 0;

            let mut skip_0_i = 0;

            for (i_0, resource_0) in resources.iter().enumerate() {
                info!("on i_0={i_0} resource_0={resource_0:?}");

                let t_0 = Arc::clone(&self.loader)
                    .run_cached(Arc::new(DevicesWrap::wrap_cpu(resource_0.clone())), Arc::clone(&ctx))
                    .await?;

                t_1_batch_len += t_0.size().dim_1();
                t_1_batch.push(t_0);
                t_1_batch_resources.push(resource_0);

                if t_1_batch_len < max_batch && i_0 < resources.len() - 1 {
                    continue;
                }

                let r = futures::stream::iter(find_motifs_runs_motifs.clone())
                    .map(|motifs_arc_wrap| {
                        let t_1_batch = t_1_batch.clone(); //cones Arcs
                        // let value_vec = vec![Arc::clone(&value)];

                        let motifs_avg_dists_arc_wrap = tmp_motifs_avg_dists_cache.get(motifs_arc_wrap.id()).expect(&format!("no dists for {}", motifs_arc_wrap.id()));

                        let store = Arc::clone(&self.store);
                        debug!("SelectMotifsStage::run: t_0_batch={t_1_batch:?} \nmotifs_arc_wrap={motifs_arc_wrap:?} \ndists_arc_wrap={motifs_avg_dists_arc_wrap:?}");
                        let t_1_batch_resources = t_1_batch_resources.clone();
                        async move {

                            with_data!(device_idx, [t_1_batch], motifs_arc_wrap, motifs_avg_dists_arc_wrap;
                                |
                                    batch_v:&[&ThoTensor2<ChannelsData2, SimpleId>],
                                    motifs_v:&DataWithExtra<ThoTensor2<ChannelsMotifsSlices, SimpleId>, (Vec<Vec<DataId>>, FindMotifsRunConfig, SelectMotifsRunConfig)>,
                                    motifs_avg_dists:&ThoTensor2<ChannelsMotifsDists, SimpleId>
                                |{

                                let unfold_params = UnfoldParams::new(1, motifs_v.extra().1.window as i64, 50/*motifs_v.extra().1.step as i64*/);
                                let (batch_1_t, batch_1_sizes) = unfold_and_cat(&batch_v, &unfold_params)?;
                                assert_eq!(batch_1_sizes.len(), batch_v.len());

                                let resource_ids_sizes_0: Vec<_> = t_1_batch_resources
                                    .into_iter()
                                    .zip(batch_1_sizes.into_iter())
                                    .map(|v| (v.0.id(), v.1 as usize))
                                    .collect();

                                let resources_iter = SlicesIter::new(unfold_params.step() as usize, resource_ids_sizes_0);

                                debug!("SelectMotifsStage::run: batch_0_t.size()={:?} motifs_v.data().t_ref().size={:?} motifs_avg_dists.t_ref().size()={:?}", batch_1_t.size(), motifs_v.data().t_ref().size(), motifs_avg_dists.t_ref().size());

                                let motifs_stream = ThoCudaStream::new(0); //TODO
                                let guard = StreamGuard::set(&motifs_stream); //TODO Device

                                let mut mask_makers = vec![
                                    LocalMin { dim: 2 },
                                    Le {
                                       // we'll have mask/dists from clc_dist_d2_and_mask in shape[channels x num_motifs x num_blocks_in_batch] so motif's average dists which are  [channels x num_motifs] should be expanded to match calculated dists
                                        v:motifs_avg_dists.t_ref().unsqueeze(-1).expand([-1, -1, batch_1_t.data().size()[1]], false)
                                    }
                                ];

                                let motifs_stream = ThoCudaStream::new(0); //TODO
                                let channel_masks_dists_ids  = {
                                    let guard = StreamGuard::set(&motifs_stream); //TODO Device


                                    let (mask, dists) = thoosa_motifs::distance::clc_dist_d2_and_mask(
                                        motifs_v.data().t_ref().shallow_clone(),
                                        batch_1_t,
                                        false, //if true dists are returned
                                        &mut mask_makers,
                                    )?;
                                    let keep_dists = true;

                                    let channel_masks_dists_ids = (0..num_channels)
                                        .map(|c| {
                                            (
                                                fold_mask(mask.i((0,.., ..)), &motifs_v.extra().0[c as usize], 0).to_device(Cpu),
                                                dists.as_ref().map(|d|fold_dists(d.i((0,.., ..)), &motifs_v.extra().0[c as usize], 0).to_device(Cpu)),
                                                motifs_v.extra().0[c as usize].iter().fold((vec![], HashSet::new()), |mut acc, d| {
                                                    if !acc.1.contains(d) {
                                                        acc.1.insert(d);
                                                        acc.0.push(d.clone());
                                                    }
                                                    acc
                                                }).0
                                            )
                                        })
                                        .collect::<Vec<(Tensor, Option<Tensor>, Vec<DataId>)>>();

                                    channel_masks_dists_ids
                                };

                                debug!("SelectMotifsStage::run: channel_masks_dists_ids={channel_masks_dists_ids:?}");
                                for (channel, mask_idsts_ids) in channel_masks_dists_ids.into_iter().enumerate() {

                                    debug!("SelectMotifsStage::run: channel={channel} mask_idsts_ids={:?}", mask_idsts_ids.2);

                                    let mut mask_in_channel = BoolTensorIterator::from(&mask_idsts_ids.0);
                                    for motif_id in mask_idsts_ids.2 {
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
                                           store.write().add_slices_to_motif(motifs_v.extra().2.id.expect("no select_motifs_run_config_id"), *motif_id, channel as i64, slices)?;
                                        }
                                    }
                                }

                                Ok::<(), ThoosaError>(())
                            })?;

                            Ok::<(), ThoosaError>(())
                        }
                    })
                    .buffer_unordered(10)
                    .try_collect::<Vec<_>>()
                    .await?;

            }
        }

        Ok(Arc::new(DevicesWrap::wrap_cpu(SelectMotifsSavableResult::new(0, self.run_configs[0].clone(), vec![]))))
    }
}


fn fold_mask<I>(mask: Tensor, ids: &Vec<I>, dim: usize) -> Tensor
where
    I:PartialEq + Debug
{

    reduce_by_groups(mask, |chunk|{
        chunk.any_dim(dim as i64, true)
    }, ids, dim)
}

fn fold_dists<I>(mask: Tensor, ids: &Vec<I>, dim: usize) -> Tensor
where
    I:PartialEq + Debug
{

    reduce_by_groups(mask, |chunk|{
        chunk.mean_dim(&[dim as i64][..], true, chunk.kind())
    }, ids, dim)
}


fn reduce_by_groups<I, F>(full: Tensor, reduce:F, ids: &Vec<I>, dim: usize) -> Tensor
    where
        I:PartialEq + Debug,
        F: Fn(Tensor) -> Tensor,
{
    debug!("fold_mask: mask={full:?} ids={ids:?}");
    trace!("fold_mask: mask=\n{full}");

    assert!(!ids.is_empty(), "ids must not be empty");

    let sizes = full.size();
    let nd = sizes.len();
    assert!(0 <= dim && dim < nd, "dim out of range");
    assert_eq!(sizes[dim] as usize, ids.len(), "ids length mismatch");


    let mut out: Vec<Tensor> = Vec::new();

    // reduce a contiguous chunk [start, start+len) along `dim`
    let mut push_group = |start: usize, len: usize| {
        let s = start as i64;
        let l = len as i64;

        // slice the chunk and OR-reduce along the folded dim, keeping dim for concat
        let chunk = full.narrow(dim as i64, s, l);
        let reduced = reduce(chunk);
        out.push(reduced);
    };

    // one-pass scan over contiguous ids
    let mut start = 0usize;
    let mut prev = &ids[0];
    for (i, id) in ids.iter().enumerate().skip(1) {
        if id != prev {
            push_group(start, i - start);
            start = i;
            prev = id;
        }
    }
    // flush the last run
    push_group(start, ids.len() - start);

    Tensor::cat(&out, dim as i64)
}


#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::path::Path;
    use std::sync::Arc;
    use log::debug;
    use log::LevelFilter::Trace;
    use parking_lot::RwLock;
    use tch::kind::FLOAT_CPU;
    use tch::{IndexOp, Kind, Tensor};
    use tokio::fs;
    use caching_runner::common::cachable::DevicesWrap;
    use caching_runner::common::{mock_run_ctx, MockRunCtx};
    use caching_runner::common::runnable::AsyncStage;
    use caching_runner::global_ctx;
    use caching_runner::global_ctx::{GlobalCtx};
    use thoosa_common::find_motifs_run_config::{ByLabelVarianceConfig, FindMotifsRunConfig, MotifSelector, SelectMotifsRunConfig};
    use thoosa_common::runs::Labeled;
    use thoosa_common::utils::{draw_function_plot, reset_print_options, vec_f32_from, z_norm, z_norm_2};
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_motifs::distance::clc_dist_d2_and_mask;
    use thoosa_motifs::generators::noise::generate_motifs_with_avg_distance;
    use thoosa_motifs::generators::non_overlapping::Interval;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::{cdt_for_dir, gen_motif, MotifKind, TensorIdSliceIdx, ThoosaError};
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::traits::{DataId, SimpleId};
    use thoosa_store::{IdSlice, MockStoreBuilder};
    use torch_runner::data::ThoTensor2;
    use crate::tasks::cross_dims::ChannelsData2;
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::motif_loader::{mock_motif_loader, MotifMeta};
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::select_motifs::{fold_dists, fold_mask, SelectMotifsStage};
    use crate::tasks::tensor_loader::{mock_tensor_loader};

    #[ctor::ctor]
    fn test_init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_single_label_one_motif() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp/test_single_label_one_motif");//.join(cdt_for_dir());
        fs::create_dir_all(test_dir).await?;

        log::set_max_level(Trace);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));
        let run_id = 1;

        let resource_a: Resource = Resource::new(1, 1, "resource_a", 1000);

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 1;

        let motif_0_in_channel_0 = 1u32;

        let resources_group = "group";

        let (resource_a_t, resource_a_intervals) = resource_tensor(resource_a.length() as usize, 100, &[vec![(2, MotifKind::Pulse)]])?;

        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
                .with_list_run_config_values(
                    run_id,
                    Ok(FindMotifsRunConfig::new(Some(run_id), 100, 100, None, None).to_properties_iter(None).collect()),
                )
                .with_list_motifs([
                    (find_motifs_run_id as RunId, Ok(vec![
                        motif_0_in_channel_0 as usize,
                    ]))
                ])
                .with_list_group_labeled_resources_sets([
                    (resources_group, Ok(vec![
                        ("Lable_0".to_string(), "resource_set_for_label_0".to_string()),
                    ]))
                ])
                .with_list_resource_set_resources_by_descr([
                    ("resource_set_for_label_0", Ok(vec![resource_a.clone()])),
                ])
                .with_add_slices_to_motif(resource_a_intervals.iter().enumerate().map(|(c, channel)|{
                    let slices = channel.get(0)
                        .map(|motif_intervals| motif_intervals.iter().map(|int| IdSlice(0, resource_a.id(), int.start as u32)).collect::<HashSet<_>>())
                        .unwrap_or_default();
                    (1, motif_0_in_channel_0, c as i64, slices)
                }))
                .build()));

        // draw_function_plot(
        //     &test_dir.join(&format!("resource_a{}.png", cdt_for_dir())),
        //     (2000, 1000),
        //     vec_f32_from(&resource_a_t.i((0, ..))).into_iter().enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>().as_slice(),
        //     &resource_a_intervals[0].iter().map(|v|(*v).into()).collect::<Vec<(f32, f32)>>(),
        //     "resource_a"
        // )?;

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

        // draw_function_plot(
        //     &test_dir.join(&format!("motif_0_{}.png", cdt_for_dir())),
        //     (2000, 1000),
        //     vec_f32_from(&motif_t.i((0, ..))).into_iter().enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>().as_slice(),
        //     &resource_a_intervals[0].iter().map(|v|(*v).into()).collect::<Vec<(f32, f32)>>(),
        //     "motif_0"
        // )?;


        // let (mask, dists) = clc_dist_d2_and_mask(z_norm(&resource_a_t.unfold(1, 100, 50), 1), z_norm(&motif_t, 1), true, &mut vec![])?;
        let (mask, dists) = clc_dist_d2_and_mask(resource_a_t.unfold(1, 100, 50), motif_t.shallow_clone(), true, &mut vec![])?;
        debug!("test_single_label_one_motif: mask=\n{mask} dists=\n{}", dists.unwrap());

        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (motif_0_in_channel_0, (MotifMeta { channel: 0, run_id: find_motifs_run_id, window: 100 }, motif_t)),
        ]).finish());


        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new_(Some(select_motifs_run_id), MotifSelector::ByLabelVariance(ByLabelVarianceConfig { _dummy: 0 }))],
            vec![find_motifs_run_id],
        ));


        let result = select_motifs_stage.run(
            Arc::new(DevicesWrap::wrap_cpu(ResourcesConfigInput::new("test".into(), Labeled::new(resources_group.into())))), Arc::new(RwLock::new(mock_run_ctx::default())),
        ).await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_1_label_1_channel_2_motifs() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp/test_single_label_one_motif");//.join(cdt_for_dir());
        fs::create_dir_all(test_dir).await?;

        log::set_max_level(Trace);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));
        let run_id = 1;

        let resource_a: Resource = Resource::new(1, 1, "resource_a", 1000);

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 1;

        let channel_0_motifs_ids = vec![1u32, 2u32];

        let resources_group = "group";

        let (resource_a_t, resource_a_intervals) = resource_tensor(resource_a.length() as usize, 100, &[vec![(1, MotifKind::Pulse), (1, MotifKind::Triangle)]])?;


        let add_slices_entries_a = resources_add_slices_to_motif_expectations(run_id, &resource_a, &channel_0_motifs_ids, &resource_a_intervals);

        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
                .with_list_run_config_values(
                    run_id,
                    Ok(FindMotifsRunConfig::new(Some(run_id), 100, 100, None, None).to_properties_iter(None).collect()),
                )
                .with_list_motifs([
                    (find_motifs_run_id as RunId, Ok(vec![
                        channel_0_motifs_ids[0] as usize,
                        channel_0_motifs_ids[1] as usize,
                    ]))
                ])
                .with_list_group_labeled_resources_sets([
                    (resources_group, Ok(vec![
                        ("Lable_0".to_string(), "resource_set_for_label_0".to_string()),
                    ]))
                ])
                .with_list_resource_set_resources_by_descr([
                    ("resource_set_for_label_0", Ok(vec![resource_a.clone()])),
                ])
                .with_add_slices_to_motif(add_slices_entries_a)
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
            (channel_0_motifs_ids[0], (MotifMeta { channel: 0, run_id: find_motifs_run_id, window: 100 }, motif_0_t)),
            (channel_0_motifs_ids[1], (MotifMeta { channel: 0, run_id: find_motifs_run_id, window: 100 }, motif_1_t)),
        ]).finish());


        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new_(Some(select_motifs_run_id), MotifSelector::ByLabelVariance(ByLabelVarianceConfig { _dummy: 0 }))],
            vec![find_motifs_run_id],
        ));


        let result = select_motifs_stage.run(
            Arc::new(DevicesWrap::wrap_cpu(ResourcesConfigInput::new("test".into(), Labeled::new(resources_group.into())))), Arc::new(RwLock::new(mock_run_ctx::default())),
        ).await?;

        Ok(())
    }

    fn resources_add_slices_to_motif_expectations(run_id: RunId, resource_a: &Resource, channel_0_motifs_ids: &Vec<u32>, resource_a_intervals: &Vec<Vec<Vec<Interval>>>) -> Vec<(RunId, u32, i64, HashSet<IdSlice>)> {
        resource_a_intervals.iter().enumerate().flat_map(|(c, channel_intervals)| {
            let channel_0_motifs_ids_ref = &channel_0_motifs_ids;
            let resource_a_ref = &resource_a;

            channel_intervals.iter().enumerate().map(move |(m_idx, motif_intervals)| {
                let motif_id = channel_0_motifs_ids_ref.get(m_idx).expect(&format!("no motif for index {m_idx}"));

                (run_id, *motif_id, c as i64, motif_intervals.iter().map(|int| IdSlice(0, resource_a_ref.id(), int.start as u32)).collect::<HashSet<_>>())
            })
        }).collect::<Vec<_>>()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_two_labels() -> Result<(), ThoosaError> {
        log::set_max_level(Trace);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 3, 100, 1000)));
        let run_id = 1;

        let resource_a: Resource = Resource::new(1, 1, "resource_a", 900);
        let resource_b: Resource = Resource::new(2, 1, "resource_b", 800);
        let resource_c: Resource = Resource::new(3, 1, "resource_c", 600);

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 1;

        let motif_0_in_channel_0 = 1u32;
        let motif_1_in_channel_0 = 2u32;
        let motif_0_in_channel_1 = 3u32;

        let resources_group = "group";

        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
                .with_list_run_config_values(
                    run_id,
                    Ok(FindMotifsRunConfig::new(Some(run_id), 100, 100, None, None).to_properties_iter(None).collect()),
                )
                .with_list_motifs([
                    (find_motifs_run_id as RunId, Ok(vec![
                        motif_0_in_channel_0 as usize,
                        motif_1_in_channel_0 as usize,
                        motif_0_in_channel_1 as usize
                    ]))
                ])
                .with_list_group_labeled_resources_sets([
                    (resources_group, Ok(vec![
                        ("Lable_0".to_string(), "resource_set_for_label_0".to_string()),
                        ("Lable_1".to_string(), "resource_set_for_label_1".to_string())
                    ]))
                ])
                .with_list_resource_set_resources_by_descr([
                    ("resource_set_for_label_0", Ok(vec![resource_a.clone(), resource_b.clone()])),
                    ("resource_set_for_label_1", Ok(vec![resource_c.clone()]))
                ])

                .build()));

        let (resource_a_t, resource_a_intervals) = resource_tensor(resource_a.length() as usize, 100, &[vec![(3, MotifKind::Pulse)], vec![(3, MotifKind::Triangle)], vec![(3, MotifKind::Sin)]])?;
        let (resource_b_t, resource_b_intervals) = resource_tensor(resource_b.length() as usize, 100, &[vec![(3, MotifKind::Pulse)], vec![(3, MotifKind::Triangle)], vec![(3, MotifKind::Sin)]])?;
        let (resource_c_t, resource_c_intervals) = resource_tensor(resource_c.length() as usize, 100, &[vec![(2, MotifKind::Pulse)], vec![(2, MotifKind::Triangle)], vec![(2, MotifKind::Sin)]])?;


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
            (motif_0_in_channel_0, (MotifMeta { channel: 0, run_id: find_motifs_run_id, window: 100 }, motifs_tensor(100, 3, MotifKind::Pulse))),
            (motif_1_in_channel_0, (MotifMeta { channel: 0, run_id: find_motifs_run_id, window: 100 }, motifs_tensor(100, 2, MotifKind::Triangle))),
            (motif_0_in_channel_1, (MotifMeta { channel: 1, run_id: find_motifs_run_id, window: 100 }, motifs_tensor(100, 4, MotifKind::Sin))),
        ]).finish());


        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new_(Some(select_motifs_run_id), MotifSelector::ByLabelVariance(ByLabelVarianceConfig { _dummy: 0 }))],
            vec![find_motifs_run_id],
        ));


        let result = select_motifs_stage.run(
            Arc::new(DevicesWrap::wrap_cpu(ResourcesConfigInput::new("test".into(), Labeled::new(resources_group.into())))), Arc::new(RwLock::new(mock_run_ctx::default())),
        ).await?;

        Ok(())
    }

    ///found motif's slices of same kind
    //TODO make slices a bit different
    pub fn motifs_tensor(motif_len:usize, num_slices: usize, motif_kind: MotifKind) -> Tensor {


        let t = Tensor::stack(
            generate_motifs_with_avg_distance(
                gen_motif(motif_len, &motif_kind).as_slice(), num_slices, 0.2, 0.3, None
            ).0.into_iter().map(|motif| Tensor::from_slice(motif.as_slice())).collect::<Vec<_>>().as_slice(), 0
        );

        debug!("motifs_tensor: t.size={:?}", t.size());

        t
    }

    ///channels_motif_kinds - channels x (how many times motif should appear in the sequence x MotifKind)
    ///returns Tensor [channels x resource_len] + channels[ motifs[ intervals[Interval] ] ]
    pub fn resource_tensor(resource_len:usize, motif_len:usize, channels_motif_kinds:&[Vec<(usize, MotifKind)>]) -> Result<(Tensor, Vec<Vec<Vec<Interval>>>), ThoosaError> {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        const MIN_GAP: usize = 300;
        const STEP_SIGMA: f32 = 0.4;

        fn normal_step(rng: &mut StdRng, sigma: f32) -> f32 {
            let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
            let u2: f32 = rng.random::<f32>();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f32::consts::TAU * u2;
            sigma * r * theta.cos()
        }

        let mut channels_signals = Vec::with_capacity(channels_motif_kinds.len());
        let mut channels_intervals = Vec::with_capacity(channels_motif_kinds.len());

        let channels_count = channels_motif_kinds.len().max(1);

        for (channel_idx, channel_kinds) in channels_motif_kinds.iter().enumerate() {
            let total_repeats: usize = channel_kinds.iter().map(|(repeats, _)| *repeats).sum();

            let mut rng = StdRng::from_os_rng();
            let mut signal = vec![0f32; resource_len];
            for idx in 1..resource_len {
                signal[idx] = signal[idx - 1] + normal_step(&mut rng, STEP_SIGMA);
            }

            if motif_len == 0 || total_repeats == 0 {
                channels_signals.push(signal);
                channels_intervals.push(channel_kinds.iter().map(|_| Vec::new()).collect());
                continue;
            }

            let total_motif_space = motif_len.checked_mul(total_repeats).ok_or_else(|| ThoosaError::new("motif placement overflows usize"))?;
            let total_gap_space = MIN_GAP.checked_mul(total_repeats.saturating_sub(1)).ok_or_else(|| ThoosaError::new("motif gap sizing overflows usize"))?;
            let required_space = total_motif_space.checked_add(total_gap_space).ok_or_else(|| ThoosaError::new("motif layout exceeds addressable range"))?;

            if required_space > resource_len {
                return Err(ThoosaError::new(format!(
                    "cannot place {total_repeats} motifs (len={motif_len}) with min_gap={MIN_GAP} into series len={resource_len}"
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

            let mut current = start_bound + slack / 2;
            let mut channel_motif_intervals = Vec::with_capacity(channel_kinds.len());
            let mut placed = 0usize;

            for (repeats, kind) in channel_kinds.iter() {
                if *repeats == 0 {
                    channel_motif_intervals.push(Vec::new());
                    continue;
                }

                let motifs = generate_motifs_with_avg_distance(
                    gen_motif(motif_len, kind).as_slice(),
                    (*repeats).max(2),
                    0.1,
                    0.2,
                    None,
                ).0;

                let mut motif_intervals = Vec::with_capacity(*repeats);

                for motif in motifs.into_iter().take(*repeats) {
                    let start = current;
                    let end = start + motif_len;
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
                        current = current.checked_add(MIN_GAP).ok_or_else(|| ThoosaError::new("motif gap sizing overflows usize"))?;
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
            50,
            &[vec![(1, MotifKind::Pulse), (1, MotifKind::Sin)], vec![(2, MotifKind::Triangle)]],
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
            100,
            &[vec![(1, MotifKind::Pulse)], vec![(1, MotifKind::Triangle)]],
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
}
