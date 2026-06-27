use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;
use async_trait::async_trait;
use log::{error, warn};
use mockall_double::double;
use parking_lot::{Mutex, RwLock};
use tch::{IndexOp, Kind, Tensor};
use tokio::io::AsyncWriteExt;
use tokio::task::spawn_blocking;
use tracing::debug;
use thoosa_common::dims::{AnyDims, AnyMetrics, Batch, ChannelsData, MotifProbability, Slice};
use thoosa_common::copy_ctx::ExecSyncStageActionResult;
use thoosa_common::my_tch::{MySequential, ThoModule, ThoModuleDowncast};
use thoosa_common::nodes::cosine_2d::{cosine_conv2d_bank_logits, CosineConv2dBankConfig, MyCosineConv2d};
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep, MAX_PARALLEL};
use thoosa_common::runner::common::cachable::{Cached, ConsumeData, DevicesWrap, WithDataRef};
use thoosa_common::runner::common::{OutId, RunCtx};
use thoosa_common::runner::common::runnable::{AsyncStage, ConsumeResult, Stage, SyncStage};
use thoosa_common::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
use thoosa_common::runner::tmp_cache::TmpCached;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::{MotifCosine2DNetArgs, StftRunConfig, ThoModuleWithStore, TrainMotifNetRunConfig};
use thoosa_common::train_nets::MotifNetTypeArgs::Cosine2DNet;
use thoosa_common::utils::{load_run_config, pad_to};
use thoosa_common::ws::Workspace;
use thoosa_named_derive::Named;
use thoosa_properties_map::FromHashmap;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::{CpuOnly, CpuOrGpu};
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::{collect_futures_with_progress, mono_key, CopyToDevice, FuturesOrdering, ThoError};
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, MonoMetrics, WithId};
use thoosa_store::{Store, TrainedNet};
use thoosa_with_data::with_data;
use crate::ApplyMotifNets;
use crate::tasks::apply_motif_nets::{BatchAppliedMotifNets, BatchAppliedMotifNetsInput};
use crate::tasks::find_motifs_run_config::FindMotifsRunConfig;
use crate::tasks::load_motif_net::{LoadMotifNet, LoadMotifNetInput, MotifNetMeta};
use crate::tasks::stage_ids::APPLY_MOTIF_NETS;
use crate::tasks::stft::SyncStft;

#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_shims::stream::ThoStreamGuard;
use thoosa_shared::tracked_await::track_await;

#[derive(Debug, cached_derive::Cached, Named)]
pub struct ApplyCosine2DMotifNets<C>
where
    C: RunCtx + Adjustable + 'static
{
    id: SynKey,
    pd:PhantomData<C>,
    conv2d_tensors:Vec<MyCosineConv2dTensors>,
    stft_run_configs: HashMap<u64, StftRunConfig>,
    slice_step: u32,
    pub trained_net_ids: Vec<u32>,
    run_stft: bool
}


impl <C>OutId for ApplyCosine2DMotifNets<C>
where
    C: RunCtx + Adjustable + 'static
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}


impl<C> ApplyCosine2DMotifNets<C>
where
    C: RunCtx + Adjustable + 'static
{
    pub async fn new<R>(
        ws: &Arc<parking_lot::RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        train_motif_net_run_config_ids: Vec<RunConfigId>,
        mut stage_ctx:R,
        max_loss: Option<f32>,
        trained_nets_limit: Option<usize>,
        slice_step: u32,
        // #[cfg(test)]
        fail_if_size_bigger_than:Option<usize>,
        quantize: bool
    ) -> Result<Self, ThoError>
    where
        R:RunCtx + Adjustable + 'static
    {
        debug!("ApplyMotifNets::new: train_motif_net_run_config_ids={train_motif_net_run_config_ids:?} trained_nets_limit={trained_nets_limit:?} quantize={quantize} slice_step={slice_step}");


        // Load all motif nets matching max_loss and trained_nets_limit for the provided train_motif_nets_run_config_ids.
        let motif_nets = track_await(store.list_trained_nets(train_motif_net_run_config_ids.clone(), max_loss, trained_nets_limit)).await?;

        if motif_nets.is_empty() {
            return  Err(ThoError::new(format!("No selected nets for train_motif_nets_run_config_ids={train_motif_net_run_config_ids:?}")));
        };

        track_await(Self::from_trained_nets(&ws, &store, stage_ctx, slice_step, fail_if_size_bigger_than, motif_nets, quantize, true)).await
    }

    pub async fn from_trained_nets<R>(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>, //for logging only
        mut stage_ctx: R,
        slice_step: u32,
        fail_if_size_bigger_than: Option<usize>,
        trained_nets: Vec<TrainedNet>,
        quantize: bool,
        run_stft: bool
    ) -> Result<ApplyCosine2DMotifNets<C>, ThoError>
    where
        R: RunCtx + Adjustable + 'static
    {
        let load_motif_net = Arc::new(LoadMotifNet::<R>::new(&ws, quantize));//TODO move to new
        let trained_net_ids = trained_nets.iter().map(|v|v.id).collect::<Vec<_>>();
        let trained_nets_len = trained_nets.len();
        let id = mono_key!(APPLY_MOTIF_NETS, "ApplyMotifNets") + &SynKey::new(trained_nets.iter().map(|net| mono_key!(net.id, format!("trained_net_id_{}", net.id))).collect::<Vec<_>>());
        let store = Arc::clone(store);


        let tmp_cache = track_await(TmpCached::new("tmp_apply_cosine2d_motif_nets")).await;
        let tmp_cache_name = tmp_cache.registered_name();
        let tmp_cache: Arc<Mutex<TmpCached<CosineConv2dTensorPart>>> = Arc::new(Mutex::new(tmp_cache));

        //this defines order in which motif nets should be applied to a resource before running label net

        let futures = trained_nets
            .into_iter()
            .enumerate()
            .map(|(trained_net_idx, trained_net)| {
                debug!("ApplyMotifNets::new: trained_net={trained_net:?}");

                let trained_net_id = trained_net.id;
                let train_motif_net_run_config_id = trained_net.run_config_id;
                let path = trained_net.net_path;
                let ws = Arc::clone(&ws);
                let store = Arc::clone(&store);
                let load_motif_net = Arc::clone(&load_motif_net);
                let load_ctx = stage_ctx.branch(false);
                let load_ctx_2 = stage_ctx.branch(false);

                let tmp_cache = Arc::clone(&tmp_cache);

                async move {
                    // let train_motif_net_run_config: TrainMotifNetRunConfig = load_run_config(&store, train_motif_net_run_config_id).await?;
                    let train_motif_net_run_config: TrainMotifNetRunConfig = TrainMotifNetRunConfig::from_properties_map(None, &track_await(store.list_run_config_values(train_motif_net_run_config_id)).await?)?;

                    debug!("ApplyMotifNets::new: train_motif_net_run_config={train_motif_net_run_config:?}");

                    let Cosine2DNet(MotifCosine2DNetArgs{stft_config, ..}) = train_motif_net_run_config.net else {
                        warn!("ApplyMotifNets::new: unexpected={train_motif_net_run_config:?}");

                        return Ok(None);
                    };

                    match track_await(load_motif_net.run_cached(LoadMotifNetInput::new(path, false), load_ctx)).await {
                        Ok(net) => {


                            let (hash_value, config) = track_await(net.with_data_ref(0, load_ctx_2, None, |module_with_store:&DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>, _ctx|{
                                let channel = module_with_store.extra().channel();
                                let motif_width = module_with_store.extra().motif_net_params.num_features();

                                let hash_value = stft_config.hash_value();

                                debug!("ApplyMotifNets::new: loaded module_with_store={module_with_store:?} hash_value={hash_value:?}");

                                for tensor_part in extract_cosine_conv_2d_parts(channel, motif_width, hash_value, trained_net_id, trained_net_idx, module_with_store.data().module())? {
                                    tmp_cache.lock().insert(tensor_part);
                                }

                                Ok((hash_value, stft_config))
                            })).await?;

                            Ok::<_, ThoError>(Some((hash_value, config)))
                        }
                        Err(e) => {
                            error!("ApplyMotifNets::new: can't load net for motif; there must have been an error while training the net: e={e:?}");
                            Ok::<_, ThoError>(None)
                        }
                    }
                }
            });

        let stft_run_configs: HashMap<u64, StftRunConfig> = track_await(collect_futures_with_progress(
            futures,
            trained_nets_len,
            FuturesOrdering::Ordered, //to keep same order as in trained_nets_ids
            "ApplyMotifNets::new/load motif nets",
        )).await
            .into_iter()
            .flatten()
            .collect::<HashMap<_, _>>();

        if stft_run_configs.is_empty() {
            return Err(ThoError::new(format!("No motif nets for trained_net_ids={trained_net_ids:?}")));
        };

        let batch_cosine_conv2d_tensors = Arc::new(BatchCosineConv2dTensors::<R>::new(mono_key!(0, "BatchCosineConv2dTensors").into()));
        let batch_input = BatchCosineConv2dTensorsInputWrapper::new(id.clone(), Arc::into_inner(tmp_cache).expect("Arc::into_inner(tmp_cache) failed").into_inner());
        let required_tmp_cache_ids = std::collections::HashSet::<&dyn CachableIdTr>::new();
        let required_ids = HashMap::from([(tmp_cache_name, &required_tmp_cache_ids)]);
        let oom_handler: OOMHandler<'_, '_, _, _, _, _, ()> = OOMHandler::new(
            vec![
                Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::FIFTY))),
                Adjusters::RunContext(Box::new(DumpToCpu::new("batch_apply_cosine2d_motif_nets"))),
            ],
            None,
        );
        let raw_conv2d_tensors = track_await(oom_handler
            .consume(
                &batch_cosine_conv2d_tensors,
                batch_input,
                stage_ctx.branch(true),
                Contexts::new(
                    NoopInputContext::<BatchCosineConv2dTensors<R>>(None),
                    NoopRunnableContext {},
                    MaxParallelAndDumpToCpuAdjustableContex::new(
                        MaxParallelAdjustContextImpl::new(),
                        DumpToCpuContextImpl::new(&required_ids),
                    ),
                ),
            )).await?;
        let conv2d_tensors = track_await(move_cosine_conv2d_tensors_to_cpu(raw_conv2d_tensors.conv2d_tensors)).await?;

        debug!("ApplyMotifNets::new: stft_run_configs={stft_run_configs:?} conv2d_tensors={conv2d_tensors:?} trained_net_ids={trained_net_ids:?}");

        Ok(Self {
            id,
            pd: Default::default(),
            stft_run_configs,
            conv2d_tensors,
            slice_step,
            trained_net_ids,
            run_stft
        })
    }

    pub fn num_nets(&self) -> usize {
        self.conv2d_tensors.iter().map(|v|v.num_nets).sum()
    }


    pub fn motif_nets_ids(&self) -> Vec<u32> {
        self.trained_net_ids.clone()
    }
}

impl<C> Stage for ApplyCosine2DMotifNets<C>
where
    C: RunCtx + Adjustable + 'static
{
    type Ctx = C;
    type In<'a> = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>;

    type Out = Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>>>;// same as in ApplyMotifNets
}


impl<C> AsyncStage for ApplyCosine2DMotifNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn run(self: Arc<Self>, input: Self::In<'static>, mut ctx: Self::Ctx) -> impl Future<Output=Result<Self::Out, ThoError>> + Send {
        let input_id = input.id().clone();

        let self_clone = Arc::clone(&self);
        async move {
            let kernels = self_clone.conv2d_tensors.iter().map(|tensors| Arc::clone(&tensors.kernels)).collect::<Vec<_>>();
            let alphas = self_clone.conv2d_tensors.iter().map(|tensors| Arc::clone(&tensors.a)).collect::<Vec<_>>();
            let betas = self_clone.conv2d_tensors.iter().map(|tensors| Arc::clone(&tensors.b)).collect::<Vec<_>>();
            let cuda_actor = GlobalCtx::global().cuda_actor(ctx.tho_device_idx());

            let ctx_for_data = ctx.branch(false);
            let res = track_await(cuda_actor.exec_sync_stage(move |stream| async move {
                let res = with_data!(ctx_for_data, input, [kernels], [alphas], [betas]; |_ctx, input:&ThoTensor<ChannelsData, SynKey>, kernels: &[&ThoTensor<AnyDims, SynKey>], alphas: &[&ThoTensor<AnyDims, SynKey>], betas: &[&ThoTensor<AnyDims, SynKey>]|{
                    debug!("ApplyCosine2DMotifNets::run: input={input:?}");
                    let stream_ref = stream.as_ref().ok_or_else(|| ThoError::new("ApplyCosine2DMotifNets::run requires a local CUDA stream"))?;
                    let _guard = ThoStreamGuard::new(stream_ref)?;
                    let motifs_applied = self_clone.conv2d_tensors.iter().enumerate().map(|(tensors_idx, tensors)|{
                        let stft_config = self_clone.stft_run_configs.get(&tensors.stft_hash_value).unwrap();
                        debug!("ApplyCosine2DMotifNets::run: tensors={tensors:?} slice_step={} stft_config={stft_config:?}", self_clone.slice_step);


                        let stft = SyncStft::<C>::new("".into(), stft_config.clone())?;

                        let t = input.t_ref().f_i((tensors.channel as i64, ..))?;
                        let size = t.size();
                        let pad_to_width = tensors.motif_width.max(stft_config.n_fft as u32);
                         let stft_t = if self_clone.run_stft {
                            let t = if size.len() == 1 {
                                if size[0] >= pad_to_width as i64 {
                                    t
                                } else {
                                    debug!("ApplyCosine2DMotifNets::run: padding t={t:?} to pad_to={}", pad_to_width);

                                    pad_to(&t, &[pad_to_width as i64], 0.0)?
                                }
                            } else if size.len()== 2 {//batch of motif slices
                                if size[1] >= pad_to_width as i64 {
                                    t
                                } else {
                                    debug!("ApplyCosine2DMotifNets::run: padding t={t:?} to pad_to={}", pad_to_width);

                                    pad_to(&t, &[size[0], pad_to_width as i64], 0.0)?
                                }
                            } else {
                                panic!("ApplyCosine2DMotifNets::run: unexpected input={input:?} shape: {:?}", size);
                            };


                            stft.build_spectrogram(
                                &t
                            )?
                        } else {
                            t
                        };

                        debug!("ApplyCosine2DMotifNets::run: stft_t={stft_t:?}");
                        let window = tensors.motif_width as i64/stft_config.win_length.as_ref().map(|v|*v).unwrap_or(1);
                        let step = self_clone.slice_step as i64 /stft_config.hop_length.as_ref().map(|v|*v).unwrap_or(1);

                        debug!("ApplyCosine2DMotifNets::run: step={step} window={window}");
                        //unfold time frames
                        let stft_t = stft_t.unfold(1,window,step);

                        debug!("ApplyCosine2DMotifNets::run: unfolded stft_t={stft_t:?}");
                        let stft_t = stft_t.permute([1, 0, 2]);

                        debug!("ApplyCosine2DMotifNets::run: permuted stft_t={stft_t:?}");

                         apply_cosine_kernel_bank_to_batched_spectrogram(
                           &stft_t.unsqueeze(1),
                           kernels[tensors_idx].as_ref(),
                           alphas[tensors_idx].as_ref(),
                           betas[tensors_idx].as_ref(),
                           CosineConv2dBankConfig::default()
                        ).and_then(|v|{
                            let unbound = v.f_squeeze_dim(2)?.f_unbind(1)?;
                            debug!("ApplyCosine2DMotifNets::run: unbound={unbound:?}");
                            unbound.into_iter().map(|v| ThoTensor::<MotifProbability, SynKey>::new("".into(), v)).collect::<Result<Vec<_>, ThoError>>()
                        })
                    }).collect::<Result<Vec<_>, ThoError>>()?
                        .into_iter()
                        .flatten()
                        .collect::<Vec<ThoTensor<MotifProbability, SynKey>>>();

                    debug!("ApplyCosine2DMotifNets::run: motifs_applied={motifs_applied:?}");

                    Ok(motifs_applied)
                })?;

                Ok(ExecSyncStageActionResult::Sync { stream, out: res })
            })).await?;

            //TODO stretch motifs in BatchAppliedMotifNets
            let batch = Arc::new(BatchAppliedMotifNets::<C>::new(mono_key!(0, "BatchAppliedMotifNets").into()));

            match batch.consume(BatchAppliedMotifNetsInput::new(self.out_id(&input_id), res, ctx.tho_device_idx()), ctx.branch(false)) {
                ConsumeResult::Failed { input, error } => {
                    Err(ThoError::from(error))
                }
                ConsumeResult::Ok(v) => {
                    Ok(Arc::new(DevicesWrap::wrap(v)))
                }
            }

            // Err(ThoError::new("zzz"))
            // Ok(Arc::new(DevicesWrap::wrap(DataWithExtra::new(ThoTensor::new(self.out_id(&input_id), Tensor::concat(res.as_slice(), 1).squeeze_dim(2))?, vec![]))))
        }
    }
}



pub fn apply_cosine_kernel_bank_to_batched_spectrogram(
    spec: &Tensor,     // [B,C,KH,W]
    kernels: &Tensor,  // [M,C,KH,KW]
    alpha: &Tensor,    // [M]
    beta: &Tensor,     // [M]
    cfg: CosineConv2dBankConfig,
) -> Result<Tensor, ThoError> {
    let spec_sz = spec.size();
    if spec_sz.len() != 4 {
        return Err(ThoError::new(format!("spec must have shape [B,C,KH,W], got {:?}", spec_sz)));
    }

    let spec_c = spec_sz[1];
    let spec_kh = spec_sz[2];

    let k_sz = kernels.size();
    if k_sz.len() != 4 {
        return Err(ThoError::new(format!("kernels must have shape [M,C,KH,KW], got {:?}", k_sz)));
    }

    let kernel_c = k_sz[1];
    let kernel_kh = k_sz[2];

    if kernel_c != spec_c {
        return Err(ThoError::new(format!("kernel C must match spec C: spec C={}, kernel C={}", spec_c, kernel_c)));
    }

    if kernel_kh != spec_kh {
        return Err(ThoError::new(format!("for this wrapper kernels must cover full spectrogram height: spec KH={}, kernel KH={}", spec_kh, kernel_kh)));
    }

    let logits = cosine_conv2d_bank_logits(
        spec,
        kernels,
        alpha,
        beta,
        cfg,
    )?; // [B,M,1,W_out]

    let out_sz = logits.size();
    if out_sz[2] != 1 {
        return Err(ThoError::new(format!("expected H_out=1 because kernel KH should cover full input KH, got output {:?}", out_sz)));
    }

    Ok(logits.f_squeeze_dim(2)?) // [B,M,W_out]
}

#[derive(Debug)]
struct MyCosineConv2dTensors {
    channel:u32,
    motif_width:u32,
    stft_hash_value: u64,
    kernels: Arc<DevicesWrap<CosineConv2dTensor>>,
    a: Arc<DevicesWrap<CosineConv2dTensor>>,
    b: Arc<DevicesWrap<CosineConv2dTensor>>,
    num_nets: usize,
}

type CosineConv2dTensor = ThoTensor<AnyDims, SynKey>;

#[derive(Debug)]
struct RawMyCosineConv2dTensors {
    channel:u32,
    motif_width:u32,
    stft_hash_value: u64,
    kernels: CosineConv2dTensor,
    a: CosineConv2dTensor,
    b: CosineConv2dTensor,
    num_nets: usize,
}

impl RawMyCosineConv2dTensors {
    fn mem_size(&self) -> u64 {
        self.kernels.mem_size() + self.a.mem_size() + self.b.mem_size()
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
enum CosineConv2dTensorKind {
    Kernel,
    Alpha,
    Beta,
}

impl CosineConv2dTensorKind {
    fn id_value(self) -> u32 {
        match self {
            CosineConv2dTensorKind::Kernel => 0,
            CosineConv2dTensorKind::Alpha => 1,
            CosineConv2dTensorKind::Beta => 2,
        }
    }
}

#[derive(Debug)]
struct CosineConv2dTensorPart {
    id: SynKey,
    channel: u32,
    motif_width: u32,
    stft_hash_value: u64,
    trained_net_order: usize,
    kind: CosineConv2dTensorKind,
    tensor: CosineConv2dTensor,
}

impl CosineConv2dTensorPart {
    fn new(channel: u32, motif_width: u32, stft_hash_value: u64, trained_net_id: u32, trained_net_order: usize, kind: CosineConv2dTensorKind, tensor: Tensor) -> Result<Self, ThoError> {
        let id = cosine_conv2d_tensor_part_id(channel, motif_width, stft_hash_value, trained_net_id, trained_net_order, kind);
        Ok(Self {
            tensor: ThoTensor::new(id.clone(), tensor)?,
            id,
            channel,
            motif_width,
            stft_hash_value,
            trained_net_order,
            kind,
        })
    }
}

impl WithId for CosineConv2dTensorPart {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Data for CosineConv2dTensorPart {
    type Metrics = AnyMetrics;
    type M = CpuOrGpu<Self>;

    fn size(&self) -> Self::Metrics {
        self.tensor.size().into()
    }

    fn device(&self) -> usize {
        self.tensor.device()
    }

    fn mem_size(&self) -> u64 {
        self.tensor.mem_size()
    }
}

#[async_trait]
impl CopyToDevice for CosineConv2dTensorPart {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        Ok(Self {
            id: self.id.clone(),
            channel: self.channel,
            motif_width: self.motif_width,
            stft_hash_value: self.stft_hash_value,
            trained_net_order: self.trained_net_order,
            kind: self.kind,
            tensor: track_await(self.tensor.copy_to_device(dst_device)).await?,
        })
    }
}

impl Output for CosineConv2dTensorPart {}

#[derive(Debug)]
struct BatchCosineConv2dTensorsInput {
    id: SynKey,
    device_idx: usize,
    parts: Vec<CosineConv2dTensorPart>,
}

impl BatchCosineConv2dTensorsInput {
    fn new(id: SynKey, parts: Vec<CosineConv2dTensorPart>, device_idx: usize) -> Self {
        Self { id, device_idx, parts }
    }
}

impl WithId for BatchCosineConv2dTensorsInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Data for BatchCosineConv2dTensorsInput {
    type Metrics = MonoMetrics<0, 1>;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        self.parts.len().into()
    }

    fn device(&self) -> usize {
        self.device_idx
    }

    fn mem_size(&self) -> u64 {
        self.parts.iter().map(|part| part.mem_size()).sum()
    }
}

impl Input for BatchCosineConv2dTensorsInput {}
impl Adjustable for BatchCosineConv2dTensorsInput {}

#[derive(Debug)]
struct BatchCosineConv2dTensorsInputWrapper {
    id: SynKey,
    tmp_cache: TmpCached<CosineConv2dTensorPart>,
}

impl BatchCosineConv2dTensorsInputWrapper {
    fn new(id: SynKey, tmp_cache: TmpCached<CosineConv2dTensorPart>) -> Self {
        Self { id, tmp_cache }
    }
}

impl WithId for BatchCosineConv2dTensorsInputWrapper {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl ConsumeData for BatchCosineConv2dTensorsInputWrapper {
    type O = BatchCosineConv2dTensorsInput;

    fn consume<R: Send, F, P>(
        self,
        to_device_idx: usize,
        ctx: P,
        _trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> impl Future<Output = ConsumeResult<Self::O, R>> + Send
    where
        F: FnOnce(Self::O, P) -> ConsumeResult<Self::O, R> + Send,
        P: RunCtx,
    {
        async move {
            let BatchCosineConv2dTensorsInputWrapper { id, tmp_cache } = self;
            match track_await(tmp_cache.take_all_recovering(to_device_idx)).await {
                Ok(parts) => do_run(BatchCosineConv2dTensorsInput::new(id, parts, to_device_idx), ctx),
                Err(err) => ConsumeResult::Failed { input: BatchCosineConv2dTensorsInput::new(id, err.data, to_device_idx), error: err.error },
            }
        }
    }

    fn burp(o: Self::O) -> impl Future<Output = Self> + Send {
        async move {
            let cache = track_await(TmpCached::new("tmp_apply_cosine2d_motif_nets_burp")).await;
            for part in o.parts {
                cache.insert(part);
            }
            BatchCosineConv2dTensorsInputWrapper::new(o.id, cache)
        }
    }
}

#[derive(Debug)]
struct BatchedCosineConv2dTensors {
    id: SynKey,
    device_idx: usize,
    conv2d_tensors: Vec<RawMyCosineConv2dTensors>,
}

impl WithId for BatchedCosineConv2dTensors {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Data for BatchedCosineConv2dTensors {
    type Metrics = MonoMetrics<0, 1>;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        self.conv2d_tensors.len().into()
    }

    fn device(&self) -> usize {
        self.device_idx
    }

    fn mem_size(&self) -> u64 {
        self.conv2d_tensors.iter().map(|tensors| tensors.mem_size()).sum()
    }
}

impl Output for BatchedCosineConv2dTensors {}

#[derive(Debug, Named)]
struct BatchCosineConv2dTensors<C>
where
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    pd: PhantomData<C>,
}

impl<C> BatchCosineConv2dTensors<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn new(id: SynKey) -> Self {
        Self { id, pd: PhantomData }
    }
}

impl<C> Adjustable for BatchCosineConv2dTensors<C>
where
    C: RunCtx + Adjustable + 'static,
{
}

impl<C> OutId for BatchCosineConv2dTensors<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<C> Stage for BatchCosineConv2dTensors<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = BatchCosineConv2dTensorsInput;
    type Out = BatchedCosineConv2dTensors;
}

#[derive(Debug, Default)]
struct CosineConv2dTensorPartGroup {
    kernels: Vec<usize>,
    alphas: Vec<usize>,
    betas: Vec<usize>,
}

impl<C> SyncStage for BatchCosineConv2dTensors<C>
where
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="BatchCosineConv2dTensors::consume"))]
    fn consume<'f>(self: Arc<Self>, mut input: Self::In<'f>, _ctx: Self::Ctx) -> ConsumeResult<Self::In<'f>, Self::Out> {
        let input_id = input.id().clone();
        let parts = std::mem::take(&mut input.parts);
        let mut groups: HashMap<(u32, u32, u64), CosineConv2dTensorPartGroup> = HashMap::new();

        for (part_idx, part) in parts.iter().enumerate() {
            let group = groups.entry((part.channel, part.motif_width, part.stft_hash_value)).or_default();
            match part.kind {
                CosineConv2dTensorKind::Kernel => group.kernels.push(part_idx),
                CosineConv2dTensorKind::Alpha => group.alphas.push(part_idx),
                CosineConv2dTensorKind::Beta => group.betas.push(part_idx),
            }
        }

        let mut groups = groups.into_iter().collect::<Vec<_>>();
        groups.sort_by_key(|((channel, motif_width, stft_hash_value), _)| (*channel, *motif_width, *stft_hash_value));

        let mut conv2d_tensors = Vec::with_capacity(groups.len());
        for ((channel, motif_width, stft_hash_value), mut group) in groups {
            group.kernels.sort_by_key(|part_idx| parts[*part_idx].trained_net_order);
            group.alphas.sort_by_key(|part_idx| parts[*part_idx].trained_net_order);
            group.betas.sort_by_key(|part_idx| parts[*part_idx].trained_net_order);

            if group.kernels.is_empty() || group.kernels.len() != group.alphas.len() || group.kernels.len() != group.betas.len() {
                input.parts = parts;
                return ConsumeResult::Failed { input, error: ThoError::new(format!("BatchCosineConv2dTensors::consume: mismatched cosine tensor parts for channel={channel} motif_width={motif_width} stft_hash_value={stft_hash_value}: kernels={} alphas={} betas={}", group.kernels.len(), group.alphas.len(), group.betas.len())) };
            }

            let group_id = cosine_conv2d_tensor_group_id(channel, motif_width, stft_hash_value);
            let kernels = match cat_cosine_conv2d_parts(&parts, &group.kernels, &group_id + mono_key!(0, "kernels")) {
                Ok(kernels) => kernels,
                Err(error) => {
                    input.parts = parts;
                    return ConsumeResult::Failed { input, error };
                }
            };
            let a = match cat_cosine_conv2d_parts(&parts, &group.alphas, &group_id + mono_key!(1, "alphas")) {
                Ok(a) => a,
                Err(error) => {
                    input.parts = parts;
                    return ConsumeResult::Failed { input, error };
                }
            };
            let b = match cat_cosine_conv2d_parts(&parts, &group.betas, &group_id + mono_key!(2, "betas")) {
                Ok(b) => b,
                Err(error) => {
                    input.parts = parts;
                    return ConsumeResult::Failed { input, error };
                }
            };

            conv2d_tensors.push(RawMyCosineConv2dTensors { channel, motif_width, stft_hash_value, kernels, a, b, num_nets: group.kernels.len() });
        }

        ConsumeResult::Ok(BatchedCosineConv2dTensors { id: self.out_id(&input_id), device_idx: input.device_idx, conv2d_tensors })
    }
}

fn cat_cosine_conv2d_parts(parts: &[CosineConv2dTensorPart], indexes: &[usize], id: SynKey) -> Result<CosineConv2dTensor, ThoError> {
    let tensor_refs = indexes.iter().map(|part_idx| parts[*part_idx].tensor.as_ref()).collect::<Vec<_>>();
    ThoTensor::new(id, Tensor::f_cat(tensor_refs.as_slice(), 0)?)
}

async fn move_cosine_conv2d_tensors_to_cpu(raw_tensors: Vec<RawMyCosineConv2dTensors>) -> Result<Vec<MyCosineConv2dTensors>, ThoError> {
    let mut tensors = Vec::with_capacity(raw_tensors.len());
    for raw in raw_tensors {
        tensors.push(MyCosineConv2dTensors {
            channel: raw.channel,
            motif_width: raw.motif_width,
            stft_hash_value: raw.stft_hash_value,
            kernels: track_await(move_cosine_tensor_to_cpu(raw.kernels)).await?,
            a: track_await(move_cosine_tensor_to_cpu(raw.a)).await?,
            b: track_await(move_cosine_tensor_to_cpu(raw.b)).await?,
            num_nets: raw.num_nets,
        });
    }
    Ok(tensors)
}

async fn move_cosine_tensor_to_cpu(tensor: CosineConv2dTensor) -> Result<Arc<DevicesWrap<CosineConv2dTensor>>, ThoError> {
    let id = tensor.id().clone();
    let mask = tensor.t();
    let mask = if mask.device().eq(&tch::Device::Cpu) {
        mask
    } else {
        let mask = if mask.is_contiguous() { mask } else { mask.f_contiguous()? };
        let device = ThoDevice::from(mask.device());
        let kind = mask.kind();

        track_await(GlobalCtx::global().cuda_actor(device).copy(mask.into(), 0.into())).await?.t()
    };
    Ok(Arc::new(DevicesWrap::wrap(ThoTensor::new(id, mask)?)))
}

fn extract_cosine_conv_2d(mut nets: Vec<Box<dyn ThoModule<Batch<Slice>, MotifProbability>>>) -> Result<RawMyCosineConv2dTensors, ThoError> {

    let mut kernels = Vec::with_capacity(nets.len());
    let mut alphas = Vec::with_capacity(nets.len());
    let mut betas = Vec::with_capacity(nets.len());
    for net in &mut nets {
        extract_cosine_conv_2d_once(&mut kernels, &mut alphas, &mut betas, net);
    }



    Ok(RawMyCosineConv2dTensors {
        channel: 0,
        motif_width: 0,
        stft_hash_value: 0,
        kernels: ThoTensor::new(mono_key!(0, "kernels").into(), Tensor::f_cat(kernels.as_slice(), 0)?)?,
        a: ThoTensor::new(mono_key!(1, "alphas").into(), Tensor::f_cat(alphas.as_slice(), 0)?)?,
        b: ThoTensor::new(mono_key!(2, "betas").into(), Tensor::f_cat(betas.as_slice(), 0)?)?,
        num_nets: kernels.len(),
    })
}

fn extract_cosine_conv_2d_once(kernels: &mut Vec<Tensor>, alphas: &mut Vec<Tensor>, betas: &mut Vec<Tensor>, net: &Box<dyn ThoModule<Batch<Slice>, MotifProbability>>) {
    let net_1 = cosine_conv_2d_layer(net);
    kernels.push(net_1.kernel.copy());
    alphas.push(net_1.alpha.unsqueeze(0).copy());
    betas.push(net_1.beta.unsqueeze(0).copy());
}

fn extract_cosine_conv_2d_parts(channel: u32, motif_width: u32, stft_hash_value: u64, trained_net_id: u32, trained_net_order: usize, net: &Box<dyn ThoModule<Batch<Slice>, MotifProbability>>) -> Result<Vec<CosineConv2dTensorPart>, ThoError> {
    let net_1 = cosine_conv_2d_layer(net);
    Ok(vec![
        CosineConv2dTensorPart::new(channel, motif_width, stft_hash_value, trained_net_id, trained_net_order, CosineConv2dTensorKind::Kernel, net_1.kernel.copy())?,
        CosineConv2dTensorPart::new(channel, motif_width, stft_hash_value, trained_net_id, trained_net_order, CosineConv2dTensorKind::Alpha, net_1.alpha.unsqueeze(0).copy())?,
        CosineConv2dTensorPart::new(channel, motif_width, stft_hash_value, trained_net_id, trained_net_order, CosineConv2dTensorKind::Beta, net_1.beta.unsqueeze(0).copy())?,
    ])
}

fn cosine_conv_2d_layer(net: &Box<dyn ThoModule<Batch<Slice>, MotifProbability>>) -> &MyCosineConv2d {
    let net_1_ref: &dyn ThoModule<Batch<Slice>, MotifProbability> = net.as_ref();
    debug!("cosine_conv_2d_layer: net_1_ref={net_1_ref:?}");

    let net_1_cosine_2_d = net_1_ref.as_any().downcast_ref::<MySequential<Batch<Slice>, MotifProbability>>().expect("downcast_mut fails");

    let layer = net_1_cosine_2_d.layers.get(1).expect("no MyCosineConv2d layer"); //TODO
    debug!("cosine_conv_2d_layer: layer={layer:?}");

    layer.as_ref().as_any().downcast_ref::<MyCosineConv2d>().expect("net_1_cosine_2_d")
}

fn cosine_conv2d_tensor_group_id(channel: u32, motif_width: u32, stft_hash_value: u64) -> SynKey {
    [
        mono_key!(channel, "channel"),
        mono_key!(motif_width, "motif_width"),
        mono_key!(stft_hash_value as usize, "stft_hash_value"),
    ].into()
}

fn cosine_conv2d_tensor_part_id(channel: u32, motif_width: u32, stft_hash_value: u64, trained_net_id: u32, trained_net_order: usize, kind: CosineConv2dTensorKind) -> SynKey {
    let suffix = SynKey::from([
        mono_key!(trained_net_id, "trained_net_id"),
        mono_key!(trained_net_order, "trained_net_order"),
        mono_key!(kind.id_value(), "tensor_kind"),
    ]);
    &cosine_conv2d_tensor_group_id(channel, motif_width, stft_hash_value) + &suffix
}


#[cfg(test)]
mod tests {
    use std::mem;
    use std::sync::Arc;
    use log::LevelFilter::Trace;
    use log::{set_max_level, trace};
    use parking_lot::RwLock;
    use tch::Device::Cpu;
    use tch::kind::FLOAT_CPU;
    use tch::nn::VarStore;
    use tch::{Device, IndexOp, Tensor};
    use tracing::debug;
    use thoosa_common::dims::{AnyDims, Batch, MotifProbability, Slice, Spectorgram};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{cosine_2d_motif_node, MotifCosine2dNetParams, MotifNetParams, MotifNetTypeParams, StftRunConfig, ThoModuleWithStore, TrainMotifNetRunConfig};
    use thoosa_common::my_tch::{InitDummy, MotifInitMode, MySequential, ThoModule};
    use thoosa_common::nodes::cosine_2d::{CosineConv2dBankConfig, MyCosineConv2d};
    use thoosa_common::runner::common::{mock_run_ctx, MockRunCtx};
    use thoosa_common::ws::Workspace;
    use thoosa_properties_map::FromHashmap;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::run_configs::RunConfigId;
    use thoosa_shared::ThoError;
    use thoosa_store::{SqlStore, Store};
    use crate::ApplyMotifNets;
    use crate::tasks::apply_motif_nets::dummy_motif_net;
    use crate::tasks::apply_motif_nets::tests::apply_motif_nets_3_channels_2_motifs;
    use crate::tasks::apply_cosine2d_motif_nets::{apply_cosine_kernel_bank_to_batched_spectrogram, extract_cosine_conv_2d, ApplyCosine2DMotifNets};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_apply_cosine_kernel_bank_to_batched_spectrogram() -> Result<(), ThoError>{
        set_max_level(Trace);
        let mut vs_0 = VarStore::new(Cpu);
        let motif_width = 100;
        //stft - n_fft - 50, hop - 10, win - 20 , (100-20/10) + 1 = 9

        //batch, freqs, timeslots - stft run on a motif [1 x motif_width]
        let kernel_1 = rand_kernel()?;
        debug!("test_apply_cosine_kernel_bank_to_batched_spectrogram: kernel_1={kernel_1:?}");
        trace!("test_apply_cosine_kernel_bank_to_batched_spectrogram: kernel_1=\n{kernel_1}");
        let mut net_1:Box<dyn ThoModule<Batch<Slice>, MotifProbability>> = cosine_2d_motif_node(&MotifCosine2dNetParams::new(0, 0, 1.0, 0.0, MotifInitMode::Mean(InitDummy{}), motif_width), Some(Arc::new(ThoTensor::new("motifs".into(), kernel_1.shallow_clone())?)))?.build_tch_module(&vs_0.root())?;


        let kernel_2 = rand_kernel()?;
        debug!("test_apply_cosine_kernel_bank_to_batched_spectrogram: kernel_2={kernel_2:?}");
        trace!("test_apply_cosine_kernel_bank_to_batched_spectrogram: kernel_2=\n{kernel_2}");

        let mut net_2:Box<dyn ThoModule<Batch<Slice>, MotifProbability>> = cosine_2d_motif_node(&MotifCosine2dNetParams::new(0, 0, 1.0, 0.0, MotifInitMode::Mean(InitDummy{}), motif_width), Some(Arc::new(ThoTensor::new("motifs".into(), kernel_2.shallow_clone())?)))?.build_tch_module(&vs_0.root())?;

        // let raw_tensor = Tensor::arange(1000.0, FLOAT_CPU);
        // debug!("raw_tensor=\n{raw_tensor}");
        // let stft_t = raw_tensor.unfold(0, motif_width, 100).stft(50, 10, 20, None::<Tensor>, true, true, true, true);
        // debug!("stft_t={stft_t:?}"); - 10, 26, 9

        let stft_t = Tensor::ones([3, 26, 9], FLOAT_CPU).into();

        let r_1 = net_1.tho_forward(&stft_t, None, false)?;
        let r_2 = net_2.tho_forward(&stft_t, None, false)?;


        // debug!("r_2=\n{r_2}");

        debug!("test_apply_cosine_kernel_bank_to_batched_spectrogram: net_1={net_1:?} type_name={:?}", net_1.as_ref().type_name());

        let tensors = extract_cosine_conv_2d(vec![net_1, net_2])?;

        let r = apply_cosine_kernel_bank_to_batched_spectrogram(
            &stft_t.as_ref().unsqueeze(1),
            tensors.kernels.as_ref(),
            tensors.a.as_ref(),
            tensors.b.as_ref(),
            CosineConv2dBankConfig::default()
        )?;

        trace!("test_apply_cosine_kernel_bank_to_batched_spectrogram: r_1=\n{r_1} r_d=\n{r_2}");
        let r_12 = Tensor::stack(&[r_1, r_2], 1);
        trace!("test_apply_cosine_kernel_bank_to_batched_spectrogram: r_12=\n{r_12}");

        trace!("test_apply_cosine_kernel_bank_to_batched_spectrogram: r=\n{r}");

        assert!(r_12.allclose(&r, 0.001, 0.001, false));


        Ok(())
    }

    fn rand_kernel() -> Result<Tensor, ThoError> {
        // Ok(Some(Arc::new(ThoTensor::new("motifs".into(), Tensor::stack((0..26 /* freqs (Kh) - 50/2+1 */).map(|i| Tensor::arange(9, FLOAT_CPU).sin).collect::<Vec<_>>().as_slice(), 0).unsqueeze(0).shallow_clone())?)))
        Ok(Tensor::rand(&[1, 26, 9], FLOAT_CPU))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_with_oom_uneven() -> Result<(), ThoError> {
        apply_motif_nets_3_channels_2_motifs(None, 5 * 32000, 32000, new_apply_motif_nets, cosine_2d_motif_node_, StftRunConfig::new(
            16000,
            4000.into(),
            4000.into(),
            true, true, true,
        ), 40000, 32000).await
    }

    async fn new_apply_motif_nets(trigger_oom_len: Option<usize>, slice_step: u32, select_motifs_run_config_id: RunConfigId, ws: &Arc<RwLock<Box<dyn Workspace>>>, store: &mut Arc<Box<dyn Store>>) -> Result<Arc<ApplyCosine2DMotifNets<MockRunCtx>>, ThoError> {
        let apply_motif_nets = Arc::new(ApplyCosine2DMotifNets::new(
            &ws,
            &store,
            vec![select_motifs_run_config_id],
            mock_run_ctx::default(1),
            Some(0.3),
            None,
            slice_step,
            trigger_oom_len,
            false,
        ).await?);
        Ok(apply_motif_nets)
    }

    pub fn cosine_2d_motif_node_(
        device: Device,
        motif_t: Arc<ThoTensor<Batch<Slice>, SynKey>>,
    ) -> (
        Box<dyn Fn() -> MotifNetParams + Sync + Send>,
        Box<dyn Fn() -> ThoModuleWithStore<Batch<Slice>, MotifProbability> + Sync + Send>,
    ) {
        let id = motif_t.id().clone();
        let motif_width = motif_t.as_ref().t_ref().size()[1] as u32;
        let cosine2d_net_params = MotifCosine2dNetParams {
            kw_time: 0,
            kh_freq: 0,
            a: 10.0,
            b: 0.0,
            init_mode: Default::default(),
            motif_width,
        };
        let net_type = MotifNetTypeParams::CosineNet2DParams(cosine2d_net_params.clone());
        let motif_init = Arc::new(ThoTensor::new("".into(), motif_t.as_ref().as_ref().stft(
            16000,
            4000,
            4000,
            None::<Tensor>,
            true,
            true,
            true,
            true
        ).abs()
        ).unwrap());

        (
            Box::new( move || {

                MotifNetParams {
                    net_type: net_type.clone()
                }
            }),
            Box::new( move || {
                let vs = VarStore::new(Cpu);
                // *dummy_motif_net::<Batch<Slice>, MotifProbability, SynKey>(id.clone(), motif_width, 1., device, "test", &motif_t.as_ref().t_ref().i((0, ..)), step as i64).unwrap()//first slice
                // cosine_2d_motif_node()
                ThoModuleWithStore::new("".into(), cosine_2d_motif_node(&cosine2d_net_params, Some(Arc::clone(&motif_init))).expect("cosine_2d_motif_node").build_tch_module(&vs.root()).expect("build_tch_module"))
            }),
        )
    }

    #[test]
    fn test_stft_unfold(){
        let t = Tensor::rand(&[10000], FLOAT_CPU);
        let t_stft = t.stft(
            100,
            10,
            10,
            None::<Tensor>,
            true,
            true,
            true,
            true,
        );
        debug!("test_stft_unfold: t_stft={t_stft:?}");

        let t_stft_unfold = t_stft.unfold(1, 200, 200);

        debug!("test_stft_unfold: t_stft_unfold={t_stft_unfold:?}");

        let t_unfold = t.unfold(0, 200, 200);

        debug!("test_stft_unfold: t_unfold={t_unfold:?}");

        let t_unfold_stft = t_unfold.stft(
            100,
            10,
            10,
            None::<Tensor>,
            true,
            true,
            true,
            true,
        );

        debug!("test_stft_unfold: t_unfold_stft={t_unfold_stft:?}");

        assert!(
            t_stft_unfold.flatten(1,2).allclose(
                &t_unfold_stft.transpose(0,1).flatten(1,2),
                0.01,0.01, false
            )
        )
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_0() -> Result<(), ThoError> {
        let store = SqlStore::from_path(&"/home/bu/tmp/ws_clef/main_db.db3").await?;

        let train_motif_net_run_config: TrainMotifNetRunConfig = TrainMotifNetRunConfig::from_properties_map(None, &store.list_run_config_values(39713).await?)?;

        debug!("train_motif_net_run_config={train_motif_net_run_config:?}");

        Ok(())
    }
}
