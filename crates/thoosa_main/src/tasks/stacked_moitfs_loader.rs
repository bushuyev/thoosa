use std::collections::HashMap;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use async_trait::async_trait;
use futures_util::{StreamExt, TryStreamExt};
use log::{debug, trace};
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
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::{CopyToDevice, ThoosaError};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::traits::{Adjustable, Data, DataId, MonoMetrics, SimpleId, WithId};
use thoosa_store::Store;
use thoosa_with_data::with_data;
use thoosa_common::tho_tensor::ThoTensor2;
use crate::tasks::average_euclidean_distance::{AverageDistanceValue, AverageEuclideanDistanceStage};
use crate::tasks::cross_dims::{ChannelsMotifsDists, ChannelsMotifsSlices, MotifSlices};
use crate::tasks::load_motif_and_clc_dist::LoadMotifAndClcDistanceStage;
use crate::tasks::motif_loader::MotifMeta;
use crate::tasks::run_ctx_2::MotifsCtx;

#[derive(Debug, Clone, DummyData)]
pub struct BatchedMotifsInput {

    id:SimpleId,
    find_motifs_run_config: Arc<FindMotifsRunConfig>,
    select_motifs_run_config: Arc<SelectMotifsRunConfig>,
}

impl BatchedMotifsInput {
    pub fn new( find_config: Arc<FindMotifsRunConfig>, select_config: Arc<SelectMotifsRunConfig>) -> Self {
        Self {
            id: format!("{}_{}", find_config.id.unwrap(), select_config.id.unwrap()).into(),
            find_motifs_run_config: find_config,
            select_motifs_run_config: select_config
        }
    }
}
impl Input for BatchedMotifsInput {}

impl WithId for BatchedMotifsInput {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug)]
pub struct StackedMotifsOutput {
    id:SimpleId,
    pub motifs: DataWithExtra<ThoTensor2<ChannelsMotifsSlices, SimpleId>, (Vec<Vec<DataId>>, Arc<FindMotifsRunConfig>, Arc<SelectMotifsRunConfig>)>,
    pub motifs_avg_dists:ThoTensor2<ChannelsMotifsDists, SimpleId>
}

impl StackedMotifsOutput {
    pub fn new(id: SimpleId, motifs: DataWithExtra<ThoTensor2<ChannelsMotifsSlices, SimpleId>, (Vec<Vec<DataId>>, Arc<FindMotifsRunConfig>, Arc<SelectMotifsRunConfig>)>, motifs_avg_dists: ThoTensor2<ChannelsMotifsDists, SimpleId>) -> Self {
        Self { id, motifs, motifs_avg_dists }
    }


}

#[async_trait]
impl CopyToDevice for StackedMotifsOutput {
    async fn copy_to_device(&self, dst_device: usize) -> Self {
        StackedMotifsOutput {
            id:self.id.clone(),
            motifs: self.motifs.copy_to_device(dst_device).await,
            motifs_avg_dists: self.motifs_avg_dists.copy_to_device(dst_device).await,
        }

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
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for StackedMotifsOutput {}

#[derive(Debug, Cached)]
pub struct BatchedMotifsLoader<ML, C>
where
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    motif_loader:Arc<ML>,
    load_and_clc_avg_dist:Arc<LoadMotifAndClcDistanceStage<ML, AverageEuclideanDistanceStage<C>, C>>,
    ws:Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<RwLock<Box<dyn Store>>>,
    pd:PhantomData<C>,
}


impl <ML, C> OutId for BatchedMotifsLoader<ML, C>
where
ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
C: RunCtx + MotifsCtx + Adjustable + 'static
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        format!("BatchedMotifsLoader_{}",  input_id).into()
    }
}

impl <ML, C> BatchedMotifsLoader<ML, C>
where
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
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

impl <ML, C> Stage for BatchedMotifsLoader<ML, C>
where
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static

{
    type Ctx = C;
    type In<'a> = BatchedMotifsInput;
    type Out = StackedMotifsOutput;
}

#[async_trait]
impl <ML, C> AsyncStage for BatchedMotifsLoader<ML, C>
where
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
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
            let select_motifs_run_config_id = input.select_motifs_run_config.id;

            async move {
                let motif_id_wrap = Arc::new(DevicesWrap::wrap_cpu((motif_id as u32).into()));
                let mut motif = motif_loader_clone.run_cached(Arc::clone(&motif_id_wrap), ctx).await?;


                let avg_motif_dist = load_and_clc_avg_dist_clone
                    .run_cached(Arc::clone(&motif_id_wrap), ctx_for_clc)
                    .await?
                    .with_data_ref(0, None, |dist_val: &AverageDistanceValue| Ok(dist_val.value()))
                    .await?;

                debug!("SelectMotifsStage::run: find_motifs_run_id={:?} run_config.id={:?} motif={motif:?} avg_motif_dist={avg_motif_dist}", find_motifs_run_config_id, select_motifs_run_config_id);

                motif_avg_dists.write().insert(DataId(motif_id as u32), avg_motif_dist);


                Ok::<_, ThoosaError>(motif)
            }
        })
        .buffer_unordered(10)//TODO buffer_unordered
        .try_collect::<Vec<_>>()
        .await?;

        with_data!(device_idx, [loaded_motifs];|data:&[&DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>]|{

            let channels_motifs = data.into_iter().fold(

                (0..num_channels).map(|_|vec![]).collect::<Vec<_>>(),//vec for each channel

                |mut acc, item| {

                    assert_eq!(
                        input.find_motifs_run_config.window, item.extra().window,
                        "window is not same: find_motifs_run_config.window={} item.extra().window={}item.extra().window, motif_id={:?}", input.find_motifs_run_config.window, item.extra().window, item.data().id()
                    );

                    if item.size().dim_0() < 1000 {

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
                    let channel_avg_dists_tensors = channel_motifs.iter().map(|motif|  {
                        let read_guard = motif_avg_dists.read();
                        let avg_dist = read_guard.get(&motif.id()).unwrap();

                        debug!("SelectMotifsStage::run: motif.id={} dist={avg_dist}", motif.id());

                        Tensor::from_slice(vec![*avg_dist; motif.size()[0] as usize].as_slice())
                    }).collect::<Vec<_>>();

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

            let find_and_select_motifs_id = SimpleId(format!("{}_{}", input.find_motifs_run_config.id.expect("no id in find_motifs_run_config"), input.select_motifs_run_config.id.expect("no id in select_motifs_run_config")).leak());
            Ok(Arc::new(DevicesWrap::wrap(StackedMotifsOutput::new(
                "".into(),
                DataWithExtra::new(
                    ThoTensor2::new(
                        find_and_select_motifs_id.clone(),
                        channels_motifs_t
                    ),
                    (channels_motif_ids, Arc::clone(&input.find_motifs_run_config), Arc::clone(&input.select_motifs_run_config))
                ),
                 ThoTensor2::new(
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
        ctx.write().add_batched_motifs_id(self.out_id(input.id()));
    }
}
