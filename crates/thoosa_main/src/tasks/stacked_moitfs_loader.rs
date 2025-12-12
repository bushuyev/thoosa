use std::any::Any;
use std::collections::HashMap;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt, TryStreamExt};
use log::{debug, trace, warn};
use parking_lot::RwLock;
use tch::Kind::Float;
use tch::{IndexOp, Tensor};
use cached_derive::Cached;
use caching_runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage};
use caching_runner::tmp_cache::TmpCached;
use thoosa_common::applicables::ThoDevice;
use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig, SelectMotifsRunConfig};
use thoosa_common::utils::pad_to;
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::{CachableId, Input, Output};
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::{CopyToDevice, ThoosaError};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, DataId, MonoMetrics, Named, SimpleId, WithId};
use thoosa_store::Store;
use thoosa_with_data::with_data;
use thoosa_common::tho_tensor::ThoTensor;
use crate::tasks::average_euclidean_distance::{AverageDistanceValue, AverageEuclideanDistanceStage};
use crate::tasks::cross_dims::{ChannelsMotifsDists, ChannelsMotifsSlices, MotifSlices};
use crate::tasks::load_motif_and_clc_dist::LoadMotifAndClcDistanceStage;
use crate::tasks::motif_loader::{MotifLoaderInput, MotifMeta};
use crate::tasks::run_ctx_2::MotifsCtx;

#[derive(Debug, Clone, DummyData)]
pub struct StackedMotifsInput {

    id:SimpleId,
    find_motifs_run_config: Arc<FindMotifsRunConfig>,
    // select_motifs_run_config: Arc<SelectMotifsRunConfig>,
}

impl StackedMotifsInput {
    pub fn new( find_config: Arc<FindMotifsRunConfig>) -> Self {
        debug!("StackedMotifsInput: find_config={find_config:?}");
        Self {
            // id: format!("{}_{}", find_config.id.unwrap(), select_config.id.unwrap()).into(),//TODO in select motifs SelectMotifsRunConfig is saved in run_grid right before running stage so no sense to cache by select_motifs_run_config. No other cases yet.
            id: format!("{}", find_config.id.unwrap()/*, select_config.id.unwrap()*/).into(),
            find_motifs_run_config: find_config,
            // select_motifs_run_config: select_config
        }
    }
}
impl Input for StackedMotifsInput {}

impl WithId for StackedMotifsInput {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct StackedMotifsOutputId {
    pub(crate)  find_motifs_run_id:RunId,
    pub(crate)  select_motifs_run_id:Vec<RunId>//not used, can be used to return previously ran select_motifs ids
}

impl StackedMotifsOutputId {
    pub fn new(find_motifs_run_id: RunId, select_motifs_run_id: Vec<RunId>) -> Self {
        Self { find_motifs_run_id, select_motifs_run_id }
    }
}

impl Named for StackedMotifsOutputId {}


#[derive(Debug)]
pub struct StackedMotifsOutput {
    id:StackedMotifsOutputId,
    pub motifs: DataWithExtra<ThoTensor<ChannelsMotifsSlices, SimpleId>, (Vec<Vec<DataId>>, Arc<FindMotifsRunConfig>, Vec<Arc<SelectMotifsRunConfig>>)>,
    pub motifs_avg_dists: ThoTensor<ChannelsMotifsDists, SimpleId>
}

impl StackedMotifsOutput {
    pub fn new(id: StackedMotifsOutputId, motifs: DataWithExtra<ThoTensor<ChannelsMotifsSlices, SimpleId>, (Vec<Vec<DataId>>, Arc<FindMotifsRunConfig>, Vec<Arc<SelectMotifsRunConfig>>)>, motifs_avg_dists: ThoTensor<ChannelsMotifsDists, SimpleId>) -> Self {
        Self { id, motifs, motifs_avg_dists }
    }


}

#[async_trait]
impl CopyToDevice for StackedMotifsOutput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoosaError> {
        Ok(StackedMotifsOutput {
            id:self.id.clone(),
            motifs: self.motifs.copy_to_device(dst_device).await?,
            motifs_avg_dists: self.motifs_avg_dists.copy_to_device(dst_device).await?,
        })

    }
}

impl Data for StackedMotifsOutput {
    type Metrics = MonoMetrics<1,3>;

    fn size(&self) -> Self::Metrics {
        self.motifs.size()
    }

    fn device(&self) -> usize {
        self.motifs.device()
    }
}

impl WithId for StackedMotifsOutput {
    type ID = StackedMotifsOutputId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for StackedMotifsOutput {}

#[derive(Debug, Cached)]
pub struct StackedMotifsLoader<ML, C>
where
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    motif_loader:Arc<ML>,
    load_and_clc_avg_dist:Arc<LoadMotifAndClcDistanceStage<ML, AverageEuclideanDistanceStage<C>, C>>,
    ws:Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<RwLock<Box<dyn Store>>>,
    pd:PhantomData<C>,
}


impl <ML, C> OutId for StackedMotifsLoader<ML, C>
where
ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
C: RunCtx + MotifsCtx + Adjustable + 'static
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        // format!("BatchedMotifsLoader_{}",  input_id).into()
        StackedMotifsOutputId::new(0, vec![])
    }
}

impl <ML, C> StackedMotifsLoader<ML, C>
where
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    pub fn new(
        ws: Arc<RwLock<Box<dyn Workspace>>>,
        motif_loader: Arc<ML>,
        load_and_clc_avg_dist: Arc<LoadMotifAndClcDistanceStage<ML, AverageEuclideanDistanceStage<C>, C>>,
        store: Arc<RwLock<Box<dyn Store>>>) -> Self
    {
        Self {
            ws,
            motif_loader,
            load_and_clc_avg_dist,
            store,
            pd: Default::default(),
        }
    }

}

impl <ML, C> Stage for StackedMotifsLoader<ML, C>
where
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static

{
    type Ctx = C;
    type In<'a> = StackedMotifsInput;
    type Out = StackedMotifsOutput;
}

// #[async_trait]
impl <ML, C> AsyncStage for StackedMotifsLoader<ML, C>
where
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static

{
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
        let (device_idx, cache_area) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area())
        };
        let num_channels = self.ws.read().channels();


        let done_cnt = Arc::new(AtomicU32::new(0));
        let mut motif_avg_dists = Arc::new(parking_lot::RwLock::new(HashMap::new()));

        let input = input.clone_from_cpu().await?;

        let motif_ids = self.store.read().list_motifs(input.find_motifs_run_config.id.expect("no id in find_motifs_run_config"))?;

        let loaded_motifs = futures::stream::iter(motif_ids)
            .map( |motif_id| {
                let done_cnt = Arc::clone(&done_cnt);

                let motif_loader_clone = Arc::clone(&self.motif_loader);
                let load_and_clc_avg_dist_clone = Arc::clone(&self.load_and_clc_avg_dist);

                let ctx = Arc::clone(&ctx);
                let ctx_for_clc = Arc::clone(&ctx);
                let motif_avg_dists = Arc::clone(&motif_avg_dists);

                let find_motifs_run_config_id = input.find_motifs_run_config.id;
                // let select_motifs_run_config_id = input.select_motifs_run_config.id;

                async move {
                    let motif_id_wrap = Arc::new(DevicesWrap::wrap_cpu(MotifLoaderInput::find_motif_slices_only((motif_id as u32).into())));//TODO used only from select motifs, when needed from somewhere where selected motifs needed too add select_motif_run_ids StackedMotifsInput
                    let motif = motif_loader_clone.run_cached(Arc::clone(&motif_id_wrap), ctx).await?;


                    let avg_motif_dist = load_and_clc_avg_dist_clone
                        .run_cached(Arc::clone(&motif_id_wrap), Arc::clone(&ctx_for_clc))
                        .await?
                        .with_data_ref(0, Arc::clone(&ctx_for_clc), None, |dist_val: &AverageDistanceValue| Ok(dist_val.value()))
                        .await?;

                    debug!("SelectMotifsStage::run: find_motifs_run_id={:?}  motif={motif:?} avg_motif_dist={avg_motif_dist}", find_motifs_run_config_id);

                    motif_avg_dists.write().insert(DataId(motif_id as u32), avg_motif_dist);


                    Ok::<_, ThoosaError>(motif)
                }
            })
            .buffered/*_unordered*/(10)//TODO buffered- to keep motif ids order consistent, try buffer_unordered and then sort?
            .filter_map(|r| async move {
                match r {
                    Ok(v) => {
                        Some(v)
                    }
                    Err(e) => {
                        warn!("Can't load motif: {e:?}");
                        None
                    }
                }
            })
            .collect::<Vec<_>>()
            .await;

        with_data!(device_idx, [loaded_motifs];|data:&[&DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>]|{

            let channels_motifs = data.into_iter().fold(

                (0..num_channels).map(|_|vec![]).collect::<Vec<_>>(),//vec for each channel

                |mut acc, item| {

                    assert_eq!(
                        input.find_motifs_run_config.window, item.extra().window,
                        "window is not same: find_motifs_run_config.window={} item.extra().window={}item.extra().window, motif_id={:?}", input.find_motifs_run_config.window, item.extra().window, item.data().id()
                    );

                    if item.size().dim_0() < 1000 {//TODO config + sample random 1000

                        acc[item.extra().channel as usize].push(item.data());
                    }
                acc
            });

            let max_motif_slices = channels_motifs
                .iter()
                .map(|v|
                    v.iter()
                    .map(|t|t.size()[0] as usize)//num motif slices known for each motif
                    .sum::<usize>())
                .max().unwrap_or(0);


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



                debug!("SelectMotifsStage::run: c={c}, channel_motifs_tensors={:?}", channel_motifs_tensors);

                let (motifs_t, moitf_avg_dists_t) = if channel_motifs_tensors.is_empty() {
                    (
                        Tensor::zeros(&[max_motif_slices as i64, input.find_motifs_run_config.window  as i64], (Float, ThoDevice::from(device_idx).into())),
                        Tensor::zeros(&[max_motif_slices as i64], (Float, ThoDevice::from(device_idx).into()))
                    )

                } else {
                    //building dists tensors for each channel to compare matches with
                    let channel_avg_dists_tensors = channel_motifs.iter()
                    .filter_map(|motif|  {
                        let read_guard = motif_avg_dists.read();
                        read_guard.get(&motif.id()).map(|avg_dist|{
                             debug!("SelectMotifsStage::run: motif.id={} dist={avg_dist}", motif.id());

                             Some(Tensor::from_slice(vec![*avg_dist; motif.size()[0] as usize].as_slice()))
                        })

                    })
                    .map(|v|v.unwrap())
                    .collect::<Vec<_>>();

                    let channel_t = Tensor::cat(&channel_motifs_tensors, 0);
                    let motif_avg_dists_t = Tensor::cat(&channel_avg_dists_tensors, 0).to_device(ThoDevice::from(device_idx).into());

                    if channel_t.size()[0] < max_motif_slices as i64 {
                        debug!("SelectMotifsStage::run: c={c}, channel_t.size={:?} padding to {:?}", channel_t.size(), [max_motif_slices as i64, input.find_motifs_run_config.window  as i64]);
                        (
                            pad_to(&channel_t, &[max_motif_slices as i64, input.find_motifs_run_config.window  as i64]).expect("pad to failed"),
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

            debug!("SelectMotifsStage::run: channels_motif_avg_dists_tensors_vec={channels_motif_avg_dists_tensors_vec:?}");
            debug!("SelectMotifsStage::run: channels_motifs_tensors_vec={channels_motifs_tensors_vec:?}");

            let channels_motifs_t = Tensor::stack(&channels_motifs_tensors_vec, 0);
            let channels_motif_avg_dists_t = Tensor::stack(&channels_motif_avg_dists_tensors_vec, 0);

            debug!("SelectMotifsStage::run: num_channels={num_channels}, channels_motif_ids={channels_motif_ids:?} channels_motifs_t.size={:?}", channels_motifs_t.size());
            trace!("SelectMotifsStage::run: channels_motif_avg_dists_t=\n{channels_motif_avg_dists_t}");
            trace!("SelectMotifsStage::run: channels_motifs_t=\n{channels_motifs_t}");

            let find_motifs_run_id = input.find_motifs_run_config.id.expect("no id in find_motifs_run_config");
            // let select_motifs_run_id = input.select_motifs_run_config.id.expect("no id in select_motifs_run_config");
            //select_motifs_run_id  -
            let select_motifs_run_id = vec![];//can be used to return select_motifs_run_i ran earlier on find_motifs_run_id
            let find_and_select_motifs_id = SimpleId(format!("{find_motifs_run_id}_{select_motifs_run_id:?}").leak());

            Ok(Arc::new(DevicesWrap::wrap(StackedMotifsOutput::new(
                StackedMotifsOutputId::new(find_motifs_run_id, select_motifs_run_id),
                DataWithExtra::new(
                    ThoTensor::new(
                        find_and_select_motifs_id.clone(),
                        channels_motifs_t
                    ),
                    (channels_motif_ids, Arc::clone(&input.find_motifs_run_config), vec![])
                ),
                 ThoTensor::new(
                    find_and_select_motifs_id,
                    channels_motif_avg_dists_t
                )
            ), cache_area)))

        })

    }

    fn on_cached<'a>(&self, input: &impl WithId<ID=<<Self as Stage>::In<'a> as WithId>::ID>, ctx: &Arc<RwLock<Self::Ctx>>)
    where
        Self: 'a
    {
        ctx.write().add_stacked_motifs_id(self.out_id(input.id()));
    }
}
