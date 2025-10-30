use std::collections::HashMap;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use async_trait::async_trait;
use futures_util::{StreamExt, TryStreamExt};
use log::debug;
use parking_lot::RwLock;
use cached_derive::Cached;
use caching_runner::common::cachable::{Cached, DevicesWrap};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage};
use caching_runner::tmp_cache::TmpCached;
use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig, SelectMotifsRunConfig};
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::{CopyToDevice, ThoosaError};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::traits::{Adjustable, Data, DataId, MonoMetrics, SimpleId, WithId};
use thoosa_with_data::with_data;
use torch_runner::data::ThoTensor2;
use crate::tasks::average_euclidean_distance::{AverageDistanceValue, AverageEuclideanDistanceStage};
use crate::tasks::cross_dims::{ChannelsMotifsDists, ChannelsMotifsSlices, MotifSlices};
use crate::tasks::load_motif_and_clc_dist::LoadMotifAndClcDistanceStage;
use crate::tasks::motif_loader::MotifMeta;

#[derive(Debug, Clone, DummyData)]
pub struct BatchedMotifsInput {
    id:RunId, //TODO Enum with options - ByFindMotifsRunId, etc
    motif_ids: Vec<usize>,
}

impl BatchedMotifsInput {
    pub fn new(id:RunId, motif_ids: Vec<usize>) -> Self {
        Self { id, motif_ids }
    }
}
impl Input for BatchedMotifsInput {}

impl WithId for BatchedMotifsInput {
    type ID = RunId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}


#[derive(Debug)]
pub struct BatchedMotifsOutput {
    id:SimpleId,
    motifs: DataWithExtra<ThoTensor2<ChannelsMotifsSlices, SimpleId>, (Vec<Vec<DataId>>, FindMotifsRunConfig, SelectMotifsRunConfig)>,
    motifs_avg_dists:ThoTensor2<ChannelsMotifsDists, SimpleId>
}

#[async_trait]
impl CopyToDevice for BatchedMotifsOutput {
    async fn copy_to_device(&self, dst_device: usize) -> Self {
        BatchedMotifsOutput {
            id:self.id.clone(),
            motifs: self.motifs.copy_to_device(dst_device).await,
            motifs_avg_dists: self.motifs_avg_dists.copy_to_device(dst_device).await,
        }

    }
}

impl Data for BatchedMotifsOutput {
    type Metrics = MonoMetrics<1,3>;

    fn size(&self) -> Self::Metrics {
        self.motifs.size()
    }

    fn device(&self) -> usize {
        self.motifs.device()
    }
}

impl WithId for BatchedMotifsOutput {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for BatchedMotifsOutput {}

#[derive(Debug, Cached)]
pub struct BatchedMotifsLoader<ML, C>
where
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    motif_loader:Arc<ML>,
    load_and_clc_avg_dist:Arc<LoadMotifAndClcDistanceStage<ML, AverageEuclideanDistanceStage<C>, C>>,
    pd:PhantomData<C>,
}


impl <ML, C> OutId for BatchedMotifsLoader<ML, C>
where
ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
C: RunCtx + Adjustable + 'static
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        format!("BatchedMotifsLoader_{}",  input_id).into()
    }
}

impl <ML, C> BatchedMotifsLoader<ML, C>
where
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    pub fn new(motif_loader:Arc<ML>, load_and_clc_avg_dist:Arc<LoadMotifAndClcDistanceStage<ML, AverageEuclideanDistanceStage<C>, C>>) -> Self {
        Self {
            motif_loader,
            load_and_clc_avg_dist,
            pd: Default::default(),
        }
    }

}

impl <ML, C> Stage for BatchedMotifsLoader<ML, C>
where
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static

{
    type Ctx = C;
    type In<'a> = BatchedMotifsInput;
    type Out = BatchedMotifsOutput;
}

#[async_trait]
impl <ML, C> AsyncStage for BatchedMotifsLoader<ML, C>
where
    ML: for <'i> AsyncStage<In<'i>=DataId, Out=DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static

{
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
        let done_cnt = Arc::new(AtomicU32::new(0));
        let mut motif_avg_dists = Arc::new(parking_lot::RwLock::new(HashMap::new()));

        let input = input.clone_from_cpu().await?;

        let loaded_motifs = futures::stream::iter(input.motif_ids)
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

                // debug!("SelectMotifsStage::run: find_motifs_run_id={find_motifs_run_id} run_config.id={:?} motif={motif:?} avg_motif_dist={avg_motif_dist}", select_motifs_run_config.id);

                motif_avg_dists.write().insert(DataId(motif_id as u32), avg_motif_dist);


                Ok::<_, ThoosaError>(motif)
            }
        })
        .buffer_unordered(10)
        .try_collect::<Vec<_>>()
        .await?;



        Err(ThoosaError::new("all failed"))
    }
}