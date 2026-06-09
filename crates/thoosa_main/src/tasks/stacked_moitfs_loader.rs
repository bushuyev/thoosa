use std::any::Any;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use async_trait::async_trait;
use log::{debug, trace, warn};
use parking_lot::RwLock;
use tch::Kind::Float;
use tch::{IndexOp, Tensor};
use cached_derive::Cached;
use thoosa_common::dims::MotifSlices;
use thoosa_named_derive::Named;
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use thoosa_common::runner::common::{OutId, RunCtx};
use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::tho_device::ThoDevice;
use crate::tasks::find_motifs_run_config::{FindMotifsRunConfig, SelectMotifsRunConfig};
use thoosa_common::utils::pad_to;
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::{collect_futures_with_progress, mono_key, sample_n_owned, CopyToDevice, StreamBuffering, ThoError};
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, DataId, MonoMetrics, WithId};
use thoosa_store::Store;
use thoosa_with_data::with_data;
use thoosa_common::tho_tensor::ThoTensor;
use crate::tasks::cross_dims::ChannelsMotifsSlices;
use crate::tasks::motif_loader::{MotifLoaderInput, MotifMeta};
use thoosa_shared::comp_id::{MonoKey, SynKey};
use crate::buffer_limits::STACKED_MOTIFS_LOADER_RUN;
use tracing::Instrument;
use thoosa_shared::tracked_await::track_await;

#[derive(Debug, Clone, DummyData)]
pub struct StackedMotifsInput {

    id: SynKey,
    find_motifs_run_config: Arc<FindMotifsRunConfig>,
}

impl StackedMotifsInput {
    pub fn new(find_config: Arc<FindMotifsRunConfig>) -> Result<Self, ThoError> {
        debug!("StackedMotifsInput: find_config={find_config:?}");
        let find_motifs_run_config_id = find_config.id.ok_or_else(|| ThoError::new("StackedMotifsInput::new: no id in find_config"))?;
        Ok(Self {
            id: SynKey::new([
                mono_key!(find_motifs_run_config_id, format!("find_motifs_run_config_id={find_motifs_run_config_id}")),
            ]),
            find_motifs_run_config: find_config,
        })
    }
}
impl Input for StackedMotifsInput {}

impl WithId for StackedMotifsInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, Named)]
#[name = "n/a"]
pub struct StackedMotifsOutputId {
    pub(crate)  find_motifs_run_config_id: RunConfigId,
    pub(crate)  select_motifs_run_config_id:Vec<RunConfigId>//not used, can be used to return previously ran select_motifs ids
}

impl StackedMotifsOutputId {
    pub fn new(find_motifs_run_config_id: RunConfigId, select_motifs_run_config_id: Vec<RunConfigId>) -> Self {
        Self { find_motifs_run_config_id, select_motifs_run_config_id }
    }
}

#[derive(Debug)]
pub struct StackedMotifsOutput {
    id:SynKey,
    pub motifs: DataWithExtra<ThoTensor<ChannelsMotifsSlices, SynKey>, (Vec<Vec<SynKey>>, Arc<FindMotifsRunConfig>, Vec<Arc<SelectMotifsRunConfig>>)>,
}

impl StackedMotifsOutput {
    pub fn new(id: SynKey, motifs: DataWithExtra<ThoTensor<ChannelsMotifsSlices, SynKey>, (Vec<Vec<SynKey>>, Arc<FindMotifsRunConfig>, Vec<Arc<SelectMotifsRunConfig>>)>) -> Self {
        Self { id, motifs }
    }


}

#[async_trait]
impl CopyToDevice for StackedMotifsOutput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        Ok(StackedMotifsOutput {
            id:self.id.clone(),
            motifs: track_await("copy_to_device/1", self.motifs.copy_to_device(dst_device)).instrument(tracing::debug_span!("copy_to_device/1", file = file!(), line = line!())).await?,
        })

    }
}

impl Data for StackedMotifsOutput {
    type Metrics = MonoMetrics<1,3>;
    type M = CpuOrGpu<Self>;

    fn size(&self) -> Self::Metrics {
        self.motifs.size()
    }

    fn device(&self) -> usize {
        self.motifs.device()
    }

    fn mem_size(&self) -> u64 {
        self.motifs.mem_size()
    }
}

impl WithId for StackedMotifsOutput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for StackedMotifsOutput {}

///loads motif slices found during initial find motifs stage. Used from SelectMotifs - slices added there are not loaded
#[derive(Debug, Cached, Named)]
pub struct StackedMotifsLoader<ML, C>
where
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    id:MonoKey,
    motif_loader:Arc<ML>,
    ws:Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<Box<dyn Store>>,
    pd:PhantomData<C>,
}


impl <ML, C> OutId for StackedMotifsLoader<ML, C>
where
ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
C: RunCtx + Adjustable + 'static
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        // StackedMotifsOutputId::new(0, vec![])
        self.id.clone() + input_id
    }
}

impl <ML, C> StackedMotifsLoader<ML, C>
where
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    pub fn new(
        ws: Arc<RwLock<Box<dyn Workspace>>>,
        motif_loader: Arc<ML>,
        store: Arc<Box<dyn Store>>) -> Self
    {
        Self {
            id: "StackedMotifsLoader".into(),
            ws,
            motif_loader,
            store,
            pd: Default::default(),
        }
    }

}

impl <ML, C> Stage for StackedMotifsLoader<ML, C>
where
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static

{
    type Ctx = C;
    type In<'a> = StackedMotifsInput;
    type Out = StackedMotifsOutput;
}

// #[async_trait]
impl <ML, C> AsyncStage for StackedMotifsLoader<ML, C>
where
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static

{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="StackedMotifsLoader::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let device_idx = ctx.tho_device_idx();

        let num_channels = thoosa_shared::perf_log!(thoosa_shared::sync_read_lock!(self.ws))
        .channels();
        let find_motifs_run_config = with_data!(ctx.branch(false), input; |_ctx, input: &StackedMotifsInput|{
            Ok::<Arc<FindMotifsRunConfig>, ThoError>(Arc::clone(&input.find_motifs_run_config))
        })?;

        let find_motifs_run_config_id = find_motifs_run_config.id;
        let find_motifs_run_config_id = find_motifs_run_config_id.ok_or_else(|| ThoError::new("StackedMotifsLoader::run: no id in find_motifs_run_config"))?;
        let motif_ids = track_await("run/1", self
            .store
            .list_motifs(find_motifs_run_config_id))
            .instrument(tracing::debug_span!("run/1", file = file!(), line = line!())).await?;
        debug!("StackedMotifsLoader::run: motif_ids={motif_ids:?}");

        if motif_ids.is_empty() {
            warn!("no motif_ids found in find_motifs_run_config_id={find_motifs_run_config_id:?}");
            return Err(ThoError::new(format!("no motif_ids found in find_motifs_run_config_id={find_motifs_run_config_id:?}")))
        }

        let motif_ids_len = motif_ids.len();
        let futures = motif_ids
            .into_iter()
            .map( |motif_id| {
                let motif_loader_clone = Arc::clone(&self.motif_loader);

                let mut ctx = ctx.branch(false);

                async move {
                    let motif_id_wrap = Arc::new(DevicesWrap::wrap(MotifLoaderInput::find_motif_slices_only((motif_id, "motif_id").into()).clone()));//TODO used only from select motifs, when needed from somewhere where selected motifs needed too add select_motif_run_config_ids StackedMotifsInput

                    match track_await("run/2", motif_loader_clone.run_cached(
                        motif_id_wrap, ctx.branch(false)
                    )).instrument(tracing::debug_span!("run/2", file = file!(), line = line!())).await {
                        Ok(motif) => {
                            debug!("StackedMotifsLoader::run: find_motifs_run_config_id={:?} motif={motif:?}", find_motifs_run_config_id);
                            Ok::<_, ThoError>(Some(motif))
                        }
                        Err(e) => {
                            warn!("Can't load motif: {e:?}");
                            Ok::<_, ThoError>(None)
                        }
                    }
                }
            });

        let loaded_motifs = track_await("run/3", collect_futures_with_progress(
            futures,
            motif_ids_len,
            STACKED_MOTIFS_LOADER_RUN,
            StreamBuffering::Ordered,
            "StackedMotifsLoader::run",
        ))
        .instrument(tracing::debug_span!("run/3", file = file!(), line = line!())).await
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

        with_data!(ctx.branch(false), [loaded_motifs];|_ctx, data:&[&DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>]|{

            let channels_motifs = data.iter().fold(

                (0..num_channels).map(|_|vec![]).collect::<Vec<_>>(),//vec for each channel

                |mut acc, item| {

                    assert_eq!(
                        find_motifs_run_config.window, item.extra().window,
                        "window is not same: find_motifs_run_config.window={} item.extra().window={}item.extra().window, motif_id={:?}", find_motifs_run_config.window, item.extra().window, item.data().id()
                    );

                    // if item.size().dim_0() < 1000 {//TODO config + sample random 1000

                        acc[item.extra().channel as usize].push(item.data());
                    // }
                acc
            });

            let max_motif_slices = channels_motifs
                .iter()
                .map(|v|
                    v.iter()
                    .map(|t|t.size().dim_0() as usize)//num motif slices known for each motif
                    .sum::<usize>())
                .max().unwrap_or(0);

            assert!(max_motif_slices > 0, "no motif slices for {data:?}");
            //no more awaits - DeviceWrapper is not needed

            let mut channels_motif_ids = vec![Vec::<SynKey>::new(); num_channels as usize];//contains channels x motif ids * number of slices in the motif
            let mut channels_motifs_tensors_vec = vec![];
            for c in 0..num_channels {

                let channel_motifs = channels_motifs
                    .get(c as usize)
                    .ok_or_else(|| ThoError::new(format!("StackedMotifsLoader::run: not enough channels: len={} c={}", channels_motifs.len(), c)))?;

                debug!("StackedMotifsLoader::run: channel {c} max_motif_slices={max_motif_slices} channel_motifs={channel_motifs:?}");

                let channel_motifs_tensors = channel_motifs.iter().map(|motif|  {
                    debug!("SelectMotifsStage::run: motif={motif:?} id={}", motif.id());
                    for _ in 0..motif.size().dim_0() {
                        channels_motif_ids[c as usize].push(motif.id().clone());
                    }

                    motif.t_ref().copy()
                }).collect::<Vec<_>>();

                debug!("StackedMotifsLoader::run: c={c}, channel_motifs_tensors={:?}", channel_motifs_tensors);

                let motifs_t = if channel_motifs_tensors.is_empty() {
                    Tensor::f_zeros([max_motif_slices as i64, find_motifs_run_config.window as i64], (Float, ThoDevice::from(device_idx).into()))?
                } else {
                    let channel_t = Tensor::f_cat(&channel_motifs_tensors, 0)?;

                    if channel_t.size()[0] < max_motif_slices as i64 {
                        debug!("StackedMotifsLoader::run: c={c}, channel_t.size={:?} padding to {:?}", channel_t.size(), [max_motif_slices as i64, find_motifs_run_config.window  as i64]);
                        pad_to(&channel_t, &[max_motif_slices as i64, find_motifs_run_config.window  as i64], 0.0)?
                    } else {
                        debug!("StackedMotifsLoader::run: c={c}, channel_t.size={:?}", channel_t.size());
                        channel_t
                    }
                };

                channels_motifs_tensors_vec.push(motifs_t);

            };

            debug!("StackedMotifsLoader::run: channels_motifs_tensors_vec={channels_motifs_tensors_vec:?}");

            let channels_motifs_t = Tensor::f_stack(&channels_motifs_tensors_vec, 0)?;

            debug!("StackedMotifsLoader::run: num_channels={num_channels}, channels_motif_ids={channels_motif_ids:?} channels_motifs_t.size={:?}", channels_motifs_t.size());
            trace!("StackedMotifsLoader::run: channels_motifs_t=\n{channels_motifs_t}");

            let find_motifs_run_config_id = find_motifs_run_config.id.ok_or_else(|| ThoError::new("StackedMotifsLoader::run: no id in find_motifs_run_config"))?;


            let motifs = ThoTensor::new(
                mono_key!("motifs", "motifs") + &mono_key!(find_motifs_run_config_id, format!("find_motifs_run_config_id_{find_motifs_run_config_id}")),
                channels_motifs_t,
            )?;

            Ok(Arc::new(DevicesWrap::wrap(StackedMotifsOutput::new(
                mono_key!("StackedMotifsOutput", "StackedMotifsOutput") + &mono_key!(find_motifs_run_config_id, format!("find_motifs_run_config_id_{find_motifs_run_config_id}")),
                DataWithExtra::new(
                    motifs,
                    (channels_motif_ids, Arc::clone(&find_motifs_run_config), vec![])
                ),
            ))))

        })

    }

}
