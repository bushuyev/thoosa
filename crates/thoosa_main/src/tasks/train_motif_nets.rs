#![allow(unexpected_cfgs)]

use std::collections::HashSet;
use std::future::ready;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;
use async_trait::async_trait;
use futures_util::{FutureExt, StreamExt};
use futures_util::stream::BoxStream;
use log::{debug, error, info, trace, warn};

use parking_lot::{Mutex, RwLock};
use plotters::prelude::BLACK;
use plotters::style::RED;
use tch::Device::{Cpu, Cuda};
use tch::kind::FLOAT_CUDA;
use tch::nn::{AdamW, OptimizerConfig, VarStore};
use tch::{IndexOp, Reduction, Tensor};
use tch::Kind::Float;
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::common::{OutId, RunCtx};
use thoosa_common::runner::common::runnable::{AsyncStage, Stage, SyncStage};

use mockall_double::double;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::debug_plot;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::batches::{AsyncDataBatch, BatchLengths};
use thoosa_common::dims::{MotifProbability, MotifSlices, Slice, ThoBatch};
use thoosa_common::my_tch::MotifInitMode;
use thoosa_common::parzen::optimizer::CompletedTrial;
use thoosa_common::runs::Empty;
use thoosa_common::tho_tensor::{ThoTensor};
use thoosa_common::train_nets::{lin_motif_node, opt_train, MotifNetParams, MotifNetTypeArgs, MotifNetTypeArgsFields, MotifsBy, OptConfig, TrainMotifNetResult, TrainMotifsNetsResult, TrainMotifNetRunConfig, TrainMotifNetRunConfigFields};
use thoosa_common::utils::{mix_x_binary_with_labels, plot_spectrogram, plot_series_with_lanes, to_vec_f32, DummyListener};
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_run_config_args::OptRunConfigSource;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::resource::Resource;
use thoosa_shared::run_configs::{OptRunConfigs, RunConfig, RunConfigId};
use thoosa_shared::{collect_futures_with_progress, mono_key, perf_log, sync_read_lock, CopyToDevice, FuturesOrdering, ThoError};
use thoosa_shared::traits::{Adjustable, Data, DataId, WithId};
use thoosa_shared::val::{NamedRange, NamedVal, Val};
use thoosa_store::Store;
use thoosa_with_data::{with_data, with_data_on_str};
use thoosa_shared::comp_id::SynKey;
use thoosa_common::dims::ChannelsData;
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta, RandomSlicesLoader, SlicesLoaderInput};
use crate::tasks::motifs_batch_builder::{make_batch_with, MixupJitterCfg};
use crate::tasks::stage_ids::TRAIN_MOTIF_NETS;
use crate::tasks::stages_seq::ABStage;
use crate::tasks::stft::Stft;
use crate::tasks::tensor_loader::RawTensorLoader;
use crate::tasks::train_net::{TrainNet, TrainNetInput};
use thoosa_shared::tracked_await::track_await;

type RandomSlicesStft<C> = Stft<C, MotifSlices, MotifSlices>;
type RandomSlicesLoaderWithStft<L, C> = ABStage<RandomSlicesLoader<L, C>, RandomSlicesStft<C>, C>;

#[derive(Debug)]
enum RandomSlicesStage<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData, SynKey>, Ctx = C, RunInput = Arc<Resource>, RunOutput = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    Raw(Arc<RandomSlicesLoader<L, C>>),
    WithStft(Arc<RandomSlicesLoaderWithStft<L, C>>),
}

impl<L, C> Stage for RandomSlicesStage<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData, SynKey>, Ctx = C, RunInput = Arc<Resource>, RunOutput = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = SlicesLoaderInput;
    type Out = ThoTensor<MotifSlices, SynKey>;
}

impl<L, C> OutId for RandomSlicesStage<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData, SynKey>, Ctx = C, RunInput = Arc<Resource>, RunOutput = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        match self {
            RandomSlicesStage::Raw(stage) => stage.out_id(input_id),
            RandomSlicesStage::WithStft(stage) => stage.out_id(input_id),
        }
    }
}

impl<L, C> AsyncStage for RandomSlicesStage<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData, SynKey>, Ctx = C, RunInput = Arc<Resource>, RunOutput = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type RunInput = Arc<Self::In<'static>>;
    type RunOutput = Arc<DevicesWrap<Self::Out>>;

    fn run_input_id(input: &Self::RunInput) -> &<<Self as Stage>::In<'static> as thoosa_shared::traits::WithId>::ID {
        thoosa_shared::traits::WithId::id(input)
    }
    async fn run(self: Arc<Self>, input: Arc<Self::In<'static>>, ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        match &*self {
            RandomSlicesStage::Raw(stage) => track_await(Arc::clone(stage).run_cached(input, ctx)).await,
            RandomSlicesStage::WithStft(stage) => track_await(Arc::clone(stage).run_cached(input, ctx)).await,
        }
    }
}

#[derive(Debug, Default)]
struct CurrentRandomSlices {
    slices: Option<Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>>,
    uses: u32,
}

#[derive(Debug)]
pub struct TrainMotifNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C, RunInput=Arc<Resource>, RunOutput=Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    id: SynKey,
    run_configs: Arc<OptRunConfigs<
        Vec<NamedVal<TrainMotifNetRunConfigFields>>,
        Vec<NamedRange<TrainMotifNetRunConfigFields>>,
        Vec<Hyperparameter<TrainMotifNetRunConfigFields>>,
        TrainMotifNetRunConfigFields,
    >>,
    motif_loader:Arc<MotifLoader<L,C>>,
    random_slices_loader:Arc<RandomSlicesLoader<L,C>>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<Box<dyn Store>>,
    epoch_size: u32,
    descr: Option<String>,
    pd:PhantomData<C>,

}


impl <L, C> OutId for TrainMotifNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C, RunInput=Arc<Resource>, RunOutput=Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{

    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        ()
    }
}


#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct TrainMotifNetInput{
    id: SynKey,
    motifs_by: MotifsBy,
    opt_config:Arc<OptConfig>
}

impl Input for TrainMotifNetInput {}
impl TrainMotifNetInput {
    pub fn new(motifs_by: MotifsBy, opt_config:Arc<OptConfig>) -> Self {
        Self { id: SynKey::new([mono_key!("TrainMotifNetInput", "stage")]), motifs_by, opt_config }
    }
}

impl WithId for TrainMotifNetInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

fn copy_motif(input: &ThoTensor<MotifSlices, SynKey>) -> Result<ThoTensor<MotifSlices, SynKey>, ThoError> {
    let mut copied = input.t_ref().f_zeros_like()?;
    copied.f_copy_(input.t_ref())?;
    ThoTensor::new(input.id().clone(), copied)
}

async fn run_stft_as_motif_slices<C>(
    stft: &Arc<RandomSlicesStft<C>>,
    input: ThoTensor<MotifSlices, SynKey>,
    mut ctx: C,
    _branch_name: &'static str,
) -> Result<ThoTensor<MotifSlices, SynKey>, ThoError>
where
    C: RunCtx + Adjustable + 'static,
{
    let device_idx = ctx.tho_device_idx();
    let out_id = input.id().clone();
    let specs = track_await(Arc::clone(&stft)
        .run(
            Arc::new(DevicesWrap::wrap(input)),
            ctx.branch(false),
        )).await?;
    let specs = track_await(Arc::into_inner(specs)
        .ok_or_else(|| ThoError::new("run_stft_as_motif_slices: stft output should not be shared"))?
        .take_data(device_idx)).await?;
    ThoTensor::new(out_id, specs.t())
}

impl <L, C> Stage for TrainMotifNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C, RunInput=Arc<Resource>, RunOutput=Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    type Ctx = C;
    type In<'a> = TrainMotifNetInput;//MotifID
    type Out = TrainMotifsNetsResult;
}

impl <L, C> TrainMotifNets<L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C, RunInput=Arc<Resource>, RunOutput=Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> + Cached + OutId + 'static,
{
    pub fn new(
        loader:Arc<L>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        run_configs:  OptRunConfigs<
            Vec<NamedVal<TrainMotifNetRunConfigFields>>,
            Vec<NamedRange<TrainMotifNetRunConfigFields>>,
            Vec<Hyperparameter<TrainMotifNetRunConfigFields>>,
            TrainMotifNetRunConfigFields,
        >,
        epoch_size: u32,
        descr: Option<String>
    ) -> Result<Self, ThoError> {
        let motif_loader = Arc::new(MotifLoader::new(
            Arc::clone(&store),
            Arc::clone(&loader),
            ws
        ));

        let random_slices_loader = Arc::new(RandomSlicesLoader::new(
            Arc::clone(&store),
            Arc::clone(&loader),
            ws
        ));

        Ok(Self {
            id: mono_key!(TRAIN_MOTIF_NETS, "TrainMotifNets").into(),
            motif_loader,
            random_slices_loader,
            run_configs: Arc::new(run_configs),
            ws: Arc::clone(&ws),
            store: Arc::clone(&store),
            epoch_size,
            descr,
            pd: PhantomData
        })
    }
}


async fn completed_train_motif_trials(store: &Arc<Box<dyn Store>>, select_motifs_run_config_ids: &[RunConfigId]) -> Result<Vec<CompletedTrial>, ThoError> {
    let mut train_parent_ids = Vec::new();
    let mut seen_parent_candidates = HashSet::new();
    for select_motifs_run_config_id in select_motifs_run_config_ids {
        for output_run_config_id in track_await(store.list_output_run_config_ids(*select_motifs_run_config_id)).await? {
            if seen_parent_candidates.insert(output_run_config_id) {
                train_parent_ids.push(output_run_config_id);
            }
        }
    }

    if train_parent_ids.is_empty() {
        return Ok(vec![]);
    }

    let train_parent_ids = track_await(store
        .list_run_configs_by_ids(&train_parent_ids)).await?
        .into_iter()
        .filter(|entry| entry.run_config_type == TrainMotifNetRunConfig::type_name() && entry.descr == "TrainMotifsNetsResult")
        .map(|entry| entry.id)
        .collect::<Vec<_>>();

    let mut child_ids = Vec::new();
    let mut seen_child_candidates = HashSet::new();
    for train_parent_id in train_parent_ids {
        for output_run_config_id in track_await(store.list_output_run_config_ids(train_parent_id)).await? {
            if seen_child_candidates.insert(output_run_config_id) {
                child_ids.push(output_run_config_id);
            }
        }
    }

    if child_ids.is_empty() {
        return Ok(vec![]);
    }

    let child_entries = track_await(store
        .list_run_configs_by_ids(&child_ids)).await?
        .into_iter()
        .filter(|entry| entry.run_config_type == TrainMotifNetRunConfig::type_name() && entry.descr == "TrainMotifNetResult")
        .collect::<Vec<_>>();

    let mut trials = Vec::new();
    for child_entry in child_entries {
        let values = track_await(store.list_run_config_values(child_entry.id)).await?;
        let Some(loss) = values.get("loss").and_then(val_to_f32) else {
            continue;
        };
        if loss.is_finite() {
            trials.push(CompletedTrial::new(loss, values));
        }
    }
    Ok(trials)
}

fn val_to_f32(val: &Val) -> Option<f32> {
    match val {
        Val::Real(v) => Some(*v as f32),
        Val::Int(v) => Some(*v as f32),
        Val::UInt(v) => Some(*v as f32),
        Val::Str(v) => v.parse().ok(),
        Val::Bool(_) => None,
    }
}

impl <L, C> AsyncStage for TrainMotifNets<L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C, RunInput=Arc<Resource>, RunOutput=Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> + Cached + OutId + 'static,
{
    type RunInput = TrainMotifNetInput;
    type RunOutput = Arc<DevicesWrap<Self::Out>>;

    fn run_input_id(input: &Self::RunInput) -> &<<Self as Stage>::In<'static> as thoosa_shared::traits::WithId>::ID {
        thoosa_shared::traits::WithId::id(input)
    }
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainMotifNets::run"))]
    async fn run(self: Arc<Self>, input: TrainMotifNetInput, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {

        let device_idx = ctx.tho_device_idx();


        let channels = perf_log!(sync_read_lock!(self.ws)).channels();

        debug!("TrainMotifNets::run: input={input:?}");

        let (select_motifs_run_config_ids, motifs_rates_selected_motifs_run_config_ids) = match input.motifs_by {

            MotifsBy::SelectMotifsRunIds { select_motifs_run_config_ids } => {

                let mut motifs_rates_selected_motifs_run_config_ids = Vec::new();
                for select_motifs_run_config_id in &select_motifs_run_config_ids {
                    let selected_motifs = track_await(self.store.list_selected_motifs(*select_motifs_run_config_id, channels)).await?;
                    debug!("TrainMotifNets::run: select_motifs_run_config_id={select_motifs_run_config_id} selected_motifs={selected_motifs:?}");
                    motifs_rates_selected_motifs_run_config_ids.extend(
                        selected_motifs
                            .into_iter()
                            .flat_map(|channels_motifs| channels_motifs.into_iter().map(move |motif_id_rate| (motif_id_rate, *select_motifs_run_config_id))),
                    );
                }
                (select_motifs_run_config_ids, motifs_rates_selected_motifs_run_config_ids)
            }
            MotifsBy::MotifId { motif_id, select_motifs_run_config_ids } => {

                let (channel, find_motifs_run_config_id) = track_await(self.store.get_motif_meta(motif_id)).await?;
                let select_motifs_run_config_ids = match select_motifs_run_config_ids {
                    Some(select_motifs_run_config_ids) => select_motifs_run_config_ids,
                    None => track_await(self.store.list_output_run_config_ids(find_motifs_run_config_id)).await?,
                };
                let mut motifs_rates_selected_motifs_run_config_ids = Vec::new();
                for select_motifs_run_config_id in &select_motifs_run_config_ids {
                    let selected_motifs = track_await(self.store.list_selected_motifs(*select_motifs_run_config_id, channels)).await?;
                    let motif_is_selected = selected_motifs
                        .get(channel)
                        .map(|channel_motifs| channel_motifs.iter().any(|(selected_motif_id, _)| *selected_motif_id == motif_id))
                        .unwrap_or(false);
                    if motif_is_selected {
                        motifs_rates_selected_motifs_run_config_ids.push(((motif_id, 1.0f64), *select_motifs_run_config_id));
                    }
                }
                (select_motifs_run_config_ids, motifs_rates_selected_motifs_run_config_ids)
            }
        };
        let motifs_rates_selected_motifs_run_config_ids_len = motifs_rates_selected_motifs_run_config_ids.len();
        debug!("TrainMotifNets::run: motifs_rates_selected_motifs_run_config_ids_len={motifs_rates_selected_motifs_run_config_ids_len} motifs_rates_selected_motifs_run_config_ids={motifs_rates_selected_motifs_run_config_ids:?}");

        let reusable_completed_trials = track_await(completed_train_motif_trials(&self.store, &select_motifs_run_config_ids)).await?;
        debug!("TrainMotifNets::run: reusable_completed_trials.len={}", reusable_completed_trials.len());

        let futures = motifs_rates_selected_motifs_run_config_ids.into_iter()
        .map(|(motif_id_rate, select_motifs_run_config_id)|{
            let mut ctx = ctx.branch(false);
            let mut ctx_run = ctx.branch(false);

            let motif_loader = Arc::clone(&self.motif_loader);
            let random_slices_loader = Arc::clone(&self.random_slices_loader);
            let opt_config = Arc::clone(&input.opt_config);
            let train_motifs_nodes_run_configs = Arc::clone(&self.run_configs);
            let store = Arc::clone(&self.store);
            let epoch_size = self.epoch_size;
            let reusable_completed_trials = reusable_completed_trials.clone();
            let descr = self.descr.clone();

            async move {

                let motif: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>> = track_await(motif_loader.run_cached(
                    Arc::new(MotifLoaderInput::new((motif_id_rate.0, "motif_id").into(), Some(vec![select_motifs_run_config_id]))),
                    ctx.branch(false),
                )).await?;

                let motif_net = Arc::clone(&motif);

                let motif_meta = track_await(motif.extra()).await?;
                let motif_width = motif_meta.window;




                let random_slices_loader_for_batch = Arc::clone(&random_slices_loader);

                debug!("TrainMotifNets::run: motif={motif:?}");

                let mut ctx_net = ctx.branch(false);
                let motif_batch = Arc::clone(&motif);


                let  ctx_net_0 = ctx_net.branch(false);
                let  ctx_net_1 = ctx_net.branch(false);


                let train_net = Arc::new(TrainNet::new(
                    move |mut ctx:C, rc:&TrainMotifNetRunConfig| {
                        match &rc.net {
                            MotifNetTypeArgs::Cosine2DNet(args) => {
                                debug!("TrainMotifNets::run: batch, rc={rc:?} motif={motif_batch:?}");

                                let motif = Arc::clone(&motif_batch);
                                let raw_random_slices_loader = Arc::clone(&random_slices_loader_for_batch);
                                let stft = Arc::new(RandomSlicesStft::new_for_dims(args.stft_config.clone()).unwrap());//TODO unwrap
                                let random_slices_loader: Arc<RandomSlicesStage<L, C>> = Arc::new(RandomSlicesStage::WithStft(Arc::new(ABStage::new(
                                    SynKey::new([mono_key!("RandomSlicesLoaderWithStft", "stage")]),
                                    raw_random_slices_loader,
                                    Arc::clone(&stft),
                                ))));
                                let motif_meta = motif_meta.clone();
                                async move {
                                    let stft_input = with_data_on_str!(ctx.branch(false), Arc::clone(&motif); |_ctx, motif:&DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>|{
                                        copy_motif(motif.data())
                                    })?;

                                    let motif_specs = track_await(run_stft_as_motif_slices(
                                        &stft,
                                        stft_input,
                                        ctx.branch(false),
                                        "train_motif_nets_motif_stft",
                                    )).await?;

                                    let motif_specs: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>> = Arc::new(DevicesWrap::wrap(DataWithExtra::new(motif_specs, motif_meta)));

                                    debug!("TrainMotifNets::run: batch, rc.id={:?} motif_specs={motif_specs:?}", rc.id);

                                    track_await(MotifsBatch::new(motif_specs, Arc::clone(&random_slices_loader), rc.common_training_args.batch_size as i64, device_idx, ctx, epoch_size)).await
                                }.boxed()
                            }
                            MotifNetTypeArgs::CosineNet(args) => {
                                debug!("TrainMotifNets::run: batch, rc={rc:?}");

                                let motif = Arc::clone(&motif_batch);
                                let random_slices_loader: Arc<RandomSlicesStage<L, C>> = Arc::new(RandomSlicesStage::Raw(Arc::clone(&random_slices_loader_for_batch)));
                                async move {
                                    track_await(MotifsBatch::new(motif, Arc::clone(&random_slices_loader), rc.common_training_args.batch_size as i64, device_idx, ctx, epoch_size)).await
                                }.boxed()
                            }
                            _ => {
                                debug!("TrainMotifNets::run: batch, rc={rc:?}");

                                let motif = Arc::clone(&motif_batch);
                                let random_slices_loader: Arc<RandomSlicesStage<L, C>> = Arc::new(RandomSlicesStage::Raw(Arc::clone(&random_slices_loader_for_batch)));
                                async move {
                                    track_await(MotifsBatch::new(motif, Arc::clone(&random_slices_loader), rc.common_training_args.batch_size as i64, device_idx, ctx, epoch_size)).await
                                }.boxed()
                            }
                        }

                    },
                    move |mut ctx:C, rc:&TrainMotifNetRunConfig, vs: Arc<Mutex<VarStore>>| {
                        // let mut ctx = ctx_net.branch(Some(&format!("train_motif_nets_task_{}_3", motif_id_rate.0)), false);
                        let motif = Arc::clone(&motif);


                        async move {
                            let motif = Arc::clone(&motif);
                            // let mut ctx = ctx_net.branch(Some(&format!("train_motif_nets_task_{}_3", motif_id_rate.0)), false);

                            match &rc.net {
                                MotifNetTypeArgs::ConvNet(net) => {
                                    let params = MotifNetParams::new_conv(motif_width, net.conv_layers, net.head_layers, net.step, net.channels, net.kernel);

                                    Ok((params.motif_net(None)?.build_tch_module(&vs.lock().root())?, params))
                                },
                                MotifNetTypeArgs::LinNet(net) => {
                                    let params = MotifNetParams::new_lin(motif_width, net.num_layers, net.step);
                                    Ok((params.motif_net(None)?.build_tch_module(&vs.lock().root())?, params))
                                },
                                MotifNetTypeArgs::CosineNet(net) => {
                                    let params = MotifNetParams::new_cosine(motif_width, net.a, net.b, net.init_mode.clone());


                                    let cosine_motif = Some(with_data_on_str!(ctx.branch(false), Arc::clone(&motif); |_ctx, motif:&DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>|{

                                        let src = motif.data().as_ref();
                                        let mut dst = src.f_zeros_like()?;
                                        dst.f_copy_(&src)?;
                                        Ok::<_, ThoError>(Arc::new(ThoTensor::<MotifSlices, SynKey>::new(motif.id().clone(), dst)?))
                                    })?);


                                    Ok((params.motif_net(cosine_motif)?.build_tch_module(&vs.lock().root())?, params))
                                },
                                MotifNetTypeArgs::Cosine2DNet(args) => {
                                    let stft = Arc::new(RandomSlicesStft::new_for_dims(args.stft_config.clone())?);
                                    debug!("TrainMotifNets::run: motif_id={} motif_width={motif_width}", motif_id_rate.0);


                                    let stft_input = with_data_on_str!(ctx.branch(false), Arc::clone(&motif); |_ctx, motif:&DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>|{
                                        copy_motif(motif.data())
                                    })?;

                                    let motif_specs = track_await(run_stft_as_motif_slices(
                                        &stft,
                                        stft_input,
                                        ctx.branch(false),
                                        "train_motif_nets_motif_stft",
                                    )).await?;
                                    let size = motif_specs.size();
                                    let params = MotifNetParams::new_cosine_2d(size[2] as u32, size[1] as u32, args.a, args.b, args.init_mode.clone(), motif_width);

                                    let cosine_motif = motif_specs.as_ref();
                                    let mut dst = cosine_motif.as_ref().as_ref().f_zeros_like()?;
                                    dst.f_copy_(&cosine_motif)?;//kernel will be changed during training
                                    debug!("TrainMotifNets::run: dst={dst:?}");

                                    Ok((params.motif_net(Some(Arc::new(ThoTensor::new(motif.id().clone(), dst)?)))?.build_tch_module(&vs.lock().root())?, params))
                                }
                            }
                        }.boxed()
                    },
                    opt_config.clone(), //TODO clone
                    train_motifs_nodes_run_configs.clone(), //TODO clone
                    |pred_y: &Tensor, target: &Tensor| {

                        // let loss = pred_y.binary_cross_entropy::<Tensor>(&target.totype(Float), None, Reduction::Mean).set_requires_grad(true);
                        let loss = pred_y.f_binary_cross_entropy_with_logits::<Tensor>(&target.f_totype(Float)?, None, None, Reduction::Mean)?;
                        debug!("TrainMotifNets::run, loss_fn: pred_y ={pred_y} target={target} loss={loss}");

                        Ok(loss)
                    },
                    move |rc:&TrainMotifNetRunConfig, vs: &VarStore|{
                        AdamW::default().amsgrad(true).build(&vs, rc.common_training_args.lr).map_err(|e|e.into())
                    },
                    |t:ThoBatch<Slice>| {
                        t
                    },
                    |rc:&TrainMotifNetRunConfig|{
                        rc.common_training_args.epochs
                    },
                    1,
                    format!("motif_{}", motif_id_rate.0)
                ).with_reusable_completed_trials(reusable_completed_trials));

                debug!("TrainMotifNets::run: train_net={train_net:?}");

                let train_net_output = track_await(Arc::into_inner(track_await(train_net.run(
                    Arc::new(TrainNetInput::new(SynKey::new([mono_key!("", "empty")]))),

                    ctx_run
                )).await?).unwrap().take_data(0)).await?;

                debug!("TrainMotifNets::run, done: motif_id_rate={motif_id_rate:?} min_loss={}", train_net_output.min_loss);

                let (channel, _find_motifs_run_config_id) = track_await(store.get_motif_meta(motif_id_rate.0)).await?;
                Ok(TrainMotifNetResult::new(
                    train_net_output.min_loss,
                    channel as u32,
                    motif_id_rate.0,
                    train_net_output.var_store,
                    select_motifs_run_config_id,
                    train_net_output.node_params,
                    train_net_output.run_config,
                    descr
                ))
            }

        });

        debug!("TrainMotifNets::run: futures.len={}", futures.len());


        let results = track_await(collect_futures_with_progress(
            futures,
            motifs_rates_selected_motifs_run_config_ids_len,
            FuturesOrdering::Unordered,
            "train motif",
        )).await;

        Ok(Arc::new(DevicesWrap::wrap_plain(TrainMotifsNetsResult::new(select_motifs_run_config_ids, results))))
    }

}


#[derive(Debug)]
pub struct MotifsBatch<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData, SynKey>, Ctx = C, RunInput = Arc<Resource>, RunOutput = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    motif_id: SynKey,
    batch_size: u32,
    epoch_size: u32,
    train_motifs: Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>,
    valid_motifs: Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>,
    random_slices_loader: Arc<RandomSlicesStage<L, C>>,
    random_slices: Mutex<CurrentRandomSlices>,
    random_slices_limit: u32,
    motif_meta: MotifMeta,
}

impl<L, C> MotifsBatch<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData, SynKey>, Ctx = C, RunInput = Arc<Resource>, RunOutput = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> + Cached + OutId + 'static,
    C: RunCtx + Adjustable,
{
    pub async fn new(
        motif: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>,
        random_slices_loader: Arc<RandomSlicesStage<L, C>>,
        batch_size: i64,
        _device_idx: usize,
        mut ctx: C,
        epoch_size: u32,
    ) -> Result<Self, ThoError> {
        debug!("MotifsBatch::new: motif={motif:?}");

        let ctx_for_split = ctx.branch(false);
        with_data!(ctx_for_split, motif; |_ctx, motif:&DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>|{
            let motif_meta = motif.extra().clone();

            let num_motif_slices = motif.data().t_ref().size()[0];
            debug!("MotifsBatch::new: motif.data={:?}", motif.data());
            assert!(num_motif_slices > 1);
            let num_validation_slices = (num_motif_slices as f32 * 0.2) as i64;
            let num_validation_slices = if num_validation_slices <= 0 { 1 } else { num_validation_slices };
            let split = num_motif_slices - num_validation_slices;

            debug!("MotifsBatch::new: motif={:?} num_motif_slices={num_motif_slices} num_validation_slices={num_validation_slices} split={split}", motif.extra());

            let motifs_t = motif.data().t_ref();
            debug_plot!(&motifs_t, "motifs_t{}", motif.id())?;

            // let train_motifs = motifs_t.i((0..split, .., ..));
            // let valid_motifs = motifs_t.i((split.., .., ..));

            let (train_motifs, valid_motifs) = match motifs_t.size().len() {
                2 => (motifs_t.i((0..split, ..)), motifs_t.i((split.., ..))),
                3 => (motifs_t.i((0..split, .., ..)), motifs_t.i((split.., .., ..))),
                len => return Err(ThoError::new(format!("MotifsBatch::new: unsupported motif tensor rank={len} size={:?}", motifs_t.size()))),
            };

            // debug_plot!(&train_motifs, "train_motif_net_new_train_motif_{}", motif.id())?;
            // debug_plot!(&train_motifs, "train_motif_net_new_valid_motifs_{}", motif.id())?;

            let train_motifs = ThoTensor::new(SynKey::new([mono_key!("train_motifs", "split")]), train_motifs)?;
            let valid_motifs = ThoTensor::new(SynKey::new([mono_key!("valid_motifs", "split")]), valid_motifs)?;

            Ok::<_, ThoError>(Self {
                motif_id: motif.id().clone(),
                train_motifs: Arc::new(DevicesWrap::wrap(train_motifs)),
                valid_motifs: Arc::new(DevicesWrap::wrap(valid_motifs)),
                random_slices_loader,
                random_slices: Mutex::new(CurrentRandomSlices::default()),
                random_slices_limit: num_motif_slices as u32,
                motif_meta,
                batch_size: batch_size as u32,
                epoch_size,
            })
        })
    }

    async fn make_batch(&self, motifs: Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>, motif_id: &SynKey, descr: &str, mut ctx: C) -> Result<(ThoBatch<Slice>, Tensor), ThoError> {
        let random_slices = track_await(self.current_random_slices(ctx.branch(false))).await?;
        let num_slices = (self.batch_size / 2) as i64;

        debug!("MotifsBatch::make_batch: descr={descr} random_slices={random_slices:?} num_slices={num_slices}");

        with_data!(ctx.branch(false), motifs, random_slices; |_ctx, motifs:&ThoTensor<MotifSlices, SynKey>, random_slices:&ThoTensor<MotifSlices, SynKey>|{
            debug!("MotifsBatch::make_batch: descr={descr} num_slices={num_slices}");

            // TODO(perf): make_batch_with is called twice for every train batch and does heavy random/mixup tensor setup; prebuild reusable mixed batches per epoch when memory allows.
            let motifs = make_batch_with(motifs.t_ref(), num_slices, None,/*Some(MixupJitterCfg { do_mixup: true, do_jitter: true, alpha: 0.4, sigma: 0.0 })*/)?;//TODO MixupJitterCfg
            let random = make_batch_with(random_slices.t_ref(), num_slices, None,/*Some(MixupJitterCfg { do_mixup: true, do_jitter: true, alpha: 0.4, sigma: 0.0 })*/)?;
            // let motifs = &motifs.t;
            // let random = &random_slices.t;

            #[cfg(tho_debug_plots)]
            plot_spectrogram(
                &motifs.i((0, .., ..)),
                format!("/tmp/plots/spectrogram_motif_{}.png", motif_id),
                1200,
                600,
            )?;

            #[cfg(tho_debug_plots)]
            plot_spectrogram(
                &random.i((0, .., ..)),
                format!("/tmp/plots/spectrogram_random_{}.png", motif_id),
                1200,
                600,
            )?;

            debug_plot!(&motifs.as_ref(), "motifs_batch_make_batch_{motif_id}_{descr}", plot_repeats=11)?;
            debug_plot!(&random.as_ref(), "motifs_batch_make_batch_random_{motif_id}_{descr}", plot_repeats=11)?;

            let (x, _lengths, y) = mix_x_binary_with_labels(&motifs, None, &random, None)?;

            debug!("MotifsBatch::train_batch: motifs={motifs:?} random={random:?}");
            debug!("MotifsBatch::train_batch: x={x:?} y={y:?}");

            Ok::<_, ThoError>((x.into(), y))
        })
    }

    async fn current_random_slices(&self, mut ctx: C) -> Result<Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>>, ThoError> {
        let epoch_size = self.epoch_size.max(1);
        {
            let mut current_random_slices = self.random_slices.lock();
            if let Some(random_slices) = current_random_slices.slices.clone() {
                if current_random_slices.uses < epoch_size {
                    current_random_slices.uses += 1;
                    return Ok(random_slices);
                }
                current_random_slices.slices = None;
                current_random_slices.uses = 0;
            }
        }

        // TODO(perf): asynchronously prefetch the next random-slice epoch before this cache misses so RandomSlicesLoader latency is hidden behind TrainStep work.
        let random_slices: Arc<DevicesWrap<ThoTensor<MotifSlices, SynKey>>> = track_await(Arc::clone(&self.random_slices_loader)
            .run(
                Arc::new(SlicesLoaderInput::new(self.motif_id.vals[0].to_data_id(), self.random_slices_limit)),
                ctx.branch(false),
            )).await?;

        let mut current_random_slices = self.random_slices.lock();
        current_random_slices.slices = Some(Arc::clone(&random_slices));
        current_random_slices.uses = 1;
        Ok(random_slices)
    }
}

impl<L, C> AsyncDataBatch<Slice, MotifProbability, C> for MotifsBatch<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData, SynKey>, Ctx = C, RunInput = Arc<Resource>, RunOutput = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> + Cached + OutId + 'static,
    C: RunCtx + Adjustable,
{
    fn train_batch_stream<'a>(&'a self, mut ctx: C) -> BoxStream<'a, Result<(ThoBatch<Slice>, Option<BatchLengths>, ThoBatch<MotifProbability>), ThoError>> {
        debug!("MotifsBatch::train_batch_stream: 0. self.epoch_size={}", self.epoch_size);


        let train_motifs = Arc::clone(&self.train_motifs);
        let motif_id = self.motif_id.clone();
        let epoch_size = self.epoch_size as usize;

        futures_util::stream::iter(0..epoch_size)
            .then(move |epoch| {
                let train_motifs = Arc::clone(&train_motifs);
                let motif_id = motif_id.clone();
                let ctx = ctx.branch(false);
                async move {

                    track_await(self.make_batch(
                        train_motifs,
                        &motif_id,
                        "train",
                        ctx,
                    )).await
                }
            })
            .map(|fut_res| fut_res.map(|v| (v.0, None, v.1.into())))
            .boxed()
    }

    fn valid_batch_stream<'a>(&'a self, mut ctx: C) -> BoxStream<'a, Result<(ThoBatch<Slice>, Option<BatchLengths>, ThoBatch<MotifProbability>), ThoError>> {
        debug!("MotifsBatch::valid_batch_stream: 0. self.epoch_size={}", self.epoch_size);


        let valid_motifs = Arc::clone(&self.valid_motifs);
        let motif_id = self.motif_id.clone();
        let epoch_size = self.epoch_size as usize;

        futures_util::stream::iter(0..epoch_size)
            .then(move |_| {
                let valid_motifs = Arc::clone(&valid_motifs);
                let motif_id = motif_id.clone();
                let ctx = ctx.branch(false);
                async move {

                    track_await(self.make_batch(
                        valid_motifs,
                        &motif_id,
                        "valid",
                        ctx,
                    )).await
                }
            })
            .map(|fut_res| fut_res.map(|v| (v.0, None, v.1.into())))
            .boxed()
    }

    fn num_batches(&self) -> usize {
        self.epoch_size as usize
    }
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Arc;
    use log::debug;
    use log::LevelFilter::{Debug, Trace};
    use parking_lot::RwLock;
    use tch::{Device, IndexOp};
    use tch::Device::Cuda;
    use tch::nn::VarStore;
    use tokio::fs;
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::common::mock_run_ctx;
    use thoosa_common::runner::common::{MaxParallelCtx, RunCtx};
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::dims::{Batch, MotifProbability, Slice};
    use crate::tasks::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::tch_test_utils;
    use thoosa_common::train_nets::{CommonTrainingRunConfigFields, MotifConvNetArgsFields, MotifCosineNetArgsFields, MotifNetParams, MotifNetTypeArgsFields, MotifsBy, OptConfig, ThoModuleWithStore, TrainMotifNetRunConfigFields, TrainMotifsNetsResult};
    use thoosa_common::debug_plot;
    use thoosa_common::utils::{reset_print_options, test_net};
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::{OptRunConfigs, RunConfigId};
    use thoosa_shared::{TensorIdSliceIdx, ThoError};
    use thoosa_shared::traits::Named;
    use thoosa_shared::hyperparameter::Hyperparameter;
    use thoosa_shared::val::{NamedRange, NamedVal, Val};
    use thoosa_store::{MockStoreBuilder, RunConfigRow};
    use thoosa_with_data::with_data;
    use thoosa_common::tch_test_utils::resource_tensor;
    use thoosa_shared::test_utils::MotifKind;
    use thoosa_common::dims::MotifSlices;
    use thoosa_common::my_tch::MotifInitModeFields;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use crate::tasks::train_motif_nets::{TrainMotifNetInput, TrainMotifNets};

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_train_motif_nets() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp/test_train_motif_nets");//.join(cdt_for_dir());
        fs::create_dir_all(test_dir).await?;

        log::set_max_level(Trace);
        reset_print_options();
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 10000)));
        let find_motifs_run_config_id: RunConfigId = 1;
        let select_motifs_run_config_id: RunConfigId = 2;
        let motif_width = 100;

        let resources = (0..10).map(|i|Resource::new(i, 1, format!("resource_{i}"), 900)).collect::<Vec<_>>();

        let (resources_map, motif_intervals) = resources.iter().try_fold((HashMap::new(), vec![]), |mut a, r| {

            let (t, intervals) = resource_tensor(
                r.length() as usize,
                 &[
                    vec![(3, motif_width, MotifKind::Pulse)],
                    vec![(1, motif_width, MotifKind::Sin)],
                    vec![]
                ],
                 100, 10.0, true, true
            )?;

            a.0.insert(r.clone(), t);

            //0 channel - the only motif
            a.1.extend(intervals[0][0].iter().map(|interval|TensorIdSliceIdx(r.id(), interval.start as u32)));

            Ok::<_, ThoError>(a)
        })?;


       let resources_id_map = resources.iter().fold(HashMap::new(), |mut a, v|{
           a.insert(v.id().clone(), v.clone());
           a
       });

        let motif_id = 10u32;
        let mut store = Arc::new(
            MockStoreBuilder::new()
                .with_list_run_config_values([(
                    find_motifs_run_config_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_config_id), 50, 100, None, None).to_properties_iter(None).collect()),
                )])
                .with_run_config_descr([(
                    find_motifs_run_config_id,
                    Ok(RunConfigRow {
                        id: find_motifs_run_config_id,
                        r#type: "FindMotifsRunConfig".to_string(),
                        descr: "test find".to_string(),
                        run_on: "2026-01-01 00:00:00".to_string(),
                    }),
                )])
                .with_list_motif_slices(motif_id, Some(vec![select_motifs_run_config_id]), None, None, Ok((0i64, find_motifs_run_config_id, motif_intervals, resources_id_map.clone()

                )))
                .with_list_random_slices_out_of_motif(motif_id, motif_width as u32, None, Ok((0i64, find_motifs_run_config_id, (0..10).flat_map(|i|vec![
                    TensorIdSliceIdx(i, 100),
                    TensorIdSliceIdx(i, 300),
                    TensorIdSliceIdx(i, 500)
                ]).collect(), resources_id_map)))

                .with_list_selected_motifs([
                    (select_motifs_run_config_id, Ok(vec![vec![(motif_id, 1.0f64)]]))
                ])
                .with_list_output_run_config_ids([(select_motifs_run_config_id, Ok(vec![]))])
                .with_get_motif_meta( [(motif_id, Ok((0usize, find_motifs_run_config_id)))])

                .build());

        // let run_configs = cos_net_run_configs();
        let run_configs = cosine_net_run_configs();

        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached(resources_map)
                .finish(),
        );


        let train_motif_net = Arc::new(TrainMotifNets::new(
            loader,
            &ws,
            &store,
            run_configs,
            10,
            None
        )?);

        let input = TrainMotifNetInput::new(MotifsBy::SelectMotifsRunIds { select_motifs_run_config_ids: vec![select_motifs_run_config_id] }, Arc::new(OptConfig::new(3, 2)));

        let mut run_ctx = mock_run_ctx::default(1);
        let run_ctx_branch = run_ctx.branch(false);
        let r:Arc<DevicesWrap<TrainMotifsNetsResult>> = train_motif_net.run(input, run_ctx_branch).await?;

        fn run_test_net(params: &MotifNetParams, vs: &VarStore) -> Result<(), ThoError> {
            let net = params.motif_net(None)?.build_tch_module::<MotifSlices, MotifProbability>(&vs.root())?;

            let (t, intervals) = resource_tensor(
                1000usize,
                &[
                    vec![(2, 100, MotifKind::Pulse)],
                ],
                100,
                10.0,
                true,
                true,
            )?;

            debug!("intervals={intervals:?} t.size={:?}", t.size());
            debug_plot!(&t, "channel_0")?;

            let output = net.tho_forward(&t.to_device(Cuda(0)).i((0, ..)).unfold(0, 100, 1).into(), None, false)?;

            debug!("output.t=\n{}", output.t_ref().view((1,-1)));

            debug_plot!(&output.t_ref().view((1,-1)), "applied 0")?;

            Ok(())
        }

        with_data!(run_ctx.branch(false), r; |_ctx, r:&TrainMotifsNetsResult|{

            debug!("test_num_layers_opt: result={r:?}");

            assert!(r.train_motif_nets_results[0].loss < 0.69);

            run_test_net(&r.train_motif_nets_results[0].params, &r.train_motif_nets_results[0].vs)?;

            Ok::<_, ThoError>(())
        })?;

        Ok(())
    }

    fn cos_net_run_configs() -> OptRunConfigs<Vec<NamedVal<TrainMotifNetRunConfigFields>>, Vec<NamedRange<TrainMotifNetRunConfigFields>>, Vec<Hyperparameter<TrainMotifNetRunConfigFields>>, TrainMotifNetRunConfigFields> {
        thoosa_shared::run_configs::OptRunConfigs::new(
            thoosa_shared::run_configs::GridRunConfigs::new(
                thoosa_shared::run_configs::ConstConfigs::new(vec![
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Epochs), Val::Int(300)),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::BatchSize), Val::Int(10)),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Patience), Val::Int(10)),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::MinDelta), Val::Real(1e-4)),
                    // NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), Val::Real(0.0001)),
                    NamedVal::new(TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::ConvNet(MotifConvNetArgsFields::Step)), Val::Int(1)),
                    NamedVal::new(TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::ConvNet(MotifConvNetArgsFields::Kernel)), Val::Int(100)),
                    NamedVal::new(TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::ConvNet(MotifConvNetArgsFields::HeadLayers)), Val::Int(8)),
                ]),
                vec![
                    // NamedRange::new(TrainMotifNetRunConfigFields::NumLayers, Val::Int(0)..Val::Int(10), (Val::Int(10), 0))
                ],
            ),
            vec![
                // Hyperparameter::UniformInt{name: TrainMotifNetRunConfigFields::NumLayers, lower: 6, upper: 8},
                Hyperparameter::UniformFloat { name: TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), lower: 0.00001, upper: 0.0001 },
                Hyperparameter::UniformInt { name: TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::ConvNet(MotifConvNetArgsFields::Channels)), lower: 8, upper: 32 }
            ],
        )
    }

    fn cosine_net_run_configs() -> OptRunConfigs<Vec<NamedVal<TrainMotifNetRunConfigFields>>, Vec<NamedRange<TrainMotifNetRunConfigFields>>, Vec<Hyperparameter<TrainMotifNetRunConfigFields>>, TrainMotifNetRunConfigFields> {
        thoosa_shared::run_configs::OptRunConfigs::new(
            thoosa_shared::run_configs::GridRunConfigs::new(
                thoosa_shared::run_configs::ConstConfigs::new(vec![
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Epochs), Val::Int(300)),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::BatchSize), Val::Int(10)),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Patience), Val::Int(10)),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::MinDelta), Val::Real(1e-4)),
                    // NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), Val::Real(0.0001)),
                    // NamedVal::new(TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::CosineNet(MotifCosineNetArgsFields::A)), Val::Real(5.0)),
                    NamedVal::new(TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::CosineNet(MotifCosineNetArgsFields::B)), Val::Real(0.0)),
                    NamedVal::new(TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::CosineNet(MotifCosineNetArgsFields::InitMode(MotifInitModeFields::Mean__Selected))), Val::UInt(0)),
                ]),
                vec![
                    // NamedRange::new(TrainMotifNetRunConfigFields::NumLayers, Val::Int(0)..Val::Int(10), (Val::Int(10), 0))
                ],
            ),
            vec![
                // Hyperparameter::UniformInt{name: TrainMotifNetRunConfigFields::NumLayers, lower: 6, upper: 8},
                Hyperparameter::UniformFloat { name: TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), lower: 0.00001, upper: 0.0001 },
                Hyperparameter::UniformFloat { name: TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::CosineNet(MotifCosineNetArgsFields::A)), lower: 1.0, upper: 10.0 }
            ],
        )
    }

}
