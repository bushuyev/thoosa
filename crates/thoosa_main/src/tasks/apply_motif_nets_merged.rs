use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;
use log::{error, warn};
use parking_lot::{RawRwLock, RwLock};
use tch::{IndexOp, Kind, Tensor};
use tch::Device::Cuda;
use tokio::io::AsyncWriteExt;
use tracing::debug;
use thoosa_common::dims::{AnyDims, Batch, ChannelsData, MotifProbability, Slice};
use thoosa_common::my_tch::{MySequential, ThoModule, ThoModuleDowncast};
use thoosa_common::nodes::cosine_2d::{cosine_conv2d_bank_logits, CosineConv2dBankConfig, MyCosineConv2d};
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use thoosa_common::runner::common::{OutId, RunCtx};
use thoosa_common::runner::common::runnable::{AsyncStage, ConsumeResult, Stage, SyncStage};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::{MotifCosine2DNetArgs, StftRunConfig, ThoModuleWithStore, TrainMotifNetRunConfig};
use thoosa_common::train_nets::MotifNetTypeArgs::Cosine2DNet;
use thoosa_common::utils::{load_run_config, pad_to};
use thoosa_common::ws::Workspace;
use thoosa_named_derive::Named;
use thoosa_properties_map::FromHashmap;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::{collect_futures_with_progress, mono_key, StreamBuffering, ThoError};
use thoosa_shared::traits::{Adjustable, WithId};
use thoosa_store::{Store, TrainedNet};
use thoosa_with_data::with_data;
use crate::ApplyMotifNets;
use crate::tasks::apply_motif_nets::{BatchAppliedMotifNets, BatchAppliedMotifNetsInput};
use crate::tasks::find_motifs_run_config::FindMotifsRunConfig;
use crate::tasks::load_motif_net::{LoadMotifNet, LoadMotifNetInput, MotifNetMeta};
use crate::tasks::stage_ids::APPLY_MOTIF_NETS;
use crate::tasks::stft::SyncStft;

#[derive(Debug, cached_derive::Cached, Named)]
pub struct ApplyMotifNetsMerged<C>
where
    C: RunCtx + Adjustable + 'static
{
    id: SynKey,
    pd:PhantomData<C>,
    conv2d_tensors:Vec<MyCosineConv2dTensors>,
    stft_run_configs: HashMap<u64, StftRunConfig>,
    slice_step: u32,
    pub trained_net_ids: Vec<u32>,
}

impl<C> Stage for ApplyMotifNetsMerged<C>
where
    C: RunCtx + Adjustable + 'static
{
    type Ctx = C;
    type In<'a> = ThoTensor<ChannelsData, SynKey>;

    type Out = DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>;// same as in ApplyMotifNets
}

impl <C>OutId for ApplyMotifNetsMerged<C>
where
    C: RunCtx + Adjustable + 'static
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}


impl<C> ApplyMotifNetsMerged<C>
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
        let motif_nets = store.list_trained_nets(train_motif_net_run_config_ids.clone(), max_loss, trained_nets_limit).await?;

        if motif_nets.is_empty() {
            return  Err(ThoError::new(format!("No selected nets for train_motif_nets_run_config_ids={train_motif_net_run_config_ids:?}")));
        };

        Self::from_trained_nets(&ws, &store, stage_ctx, slice_step, fail_if_size_bigger_than, motif_nets, quantize).await
    }

    pub async fn from_trained_nets<R>(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>, //for logging only
        mut stage_ctx: R,
        slice_step: u32,
        fail_if_size_bigger_than: Option<usize>,
        trained_nets: Vec<TrainedNet>,
        quantize: bool
    ) -> Result<ApplyMotifNetsMerged<C>, ThoError>
    where
        R: RunCtx + Adjustable + 'static
    {
        let load_motif_net = Arc::new(LoadMotifNet::<R>::new(&ws, quantize));//TODO move to new
        let trained_net_ids = trained_nets.iter().map(|v|v.id).collect::<Vec<_>>();
        let trained_nets_len = trained_nets.len();
        let id = mono_key!(APPLY_MOTIF_NETS, "ApplyMotifNets") + &SynKey::new(trained_nets.iter().map(|net| mono_key!(net.id, format!("trained_net_id_{}", net.id))).collect::<Vec<_>>());


        let channels_widths_net_tensors:Arc<RwLock<HashMap<(u32, u32, u64), (Vec<ThoTensor>, Vec<ThoTensor>, Vec<ThoTensor>)>>> = Arc::new(RwLock::new(HashMap::new()));

        //this defines order in which motif nets should be applied to a resource before running label net

        let futures = trained_nets
            .into_iter()
            .map(|trained_net| {
                debug!("ApplyMotifNets::new: trained_net={trained_net:?}");

                let trained_net_id = trained_net.id;
                let train_motif_net_run_config_id = trained_net.run_config_id;
                let path = trained_net.net_path;
                let ws = Arc::clone(&ws);
                let load_motif_net = Arc::clone(&load_motif_net);
                let load_ctx = stage_ctx.branch(Some("apply_motif_nets_load_motif_net"), false);
                let load_ctx_2 = stage_ctx.branch(Some("apply_motif_nets_load_motif_net"), false);

                let channels_widths_net_tensors = Arc::clone(&channels_widths_net_tensors);

                async move {
                    // let train_motif_net_run_config: TrainMotifNetRunConfig = load_run_config(&store, train_motif_net_run_config_id).await?;
                    let train_motif_net_run_config: TrainMotifNetRunConfig = TrainMotifNetRunConfig::from_properties_map(None, &store.list_run_config_values(train_motif_net_run_config_id).await?)?;

                    debug!("ApplyMotifNets::new: train_motif_net_run_config={train_motif_net_run_config:?}");

                    let Cosine2DNet(MotifCosine2DNetArgs{stft_config, ..}) = train_motif_net_run_config.net else {
                        warn!("ApplyMotifNets::new: unexpected={train_motif_net_run_config:?}");

                        return Ok(None);
                    };

                    match load_motif_net.run_cached(Arc::new(DevicesWrap::wrap(LoadMotifNetInput::new(path, false))), load_ctx).await {
                        Ok(net) => {


                            let (hash_value, config) = net.with_data_ref(0, load_ctx_2, None, |module_with_store:&DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>, ctx|{
                                let channel = module_with_store.extra().channel();
                                let motif_width = module_with_store.extra().motif_net_params.num_features();

                                let mut write_guard = channels_widths_net_tensors.write();
                                let hash_value = stft_config.hash_value();

                                debug!("ApplyMotifNets::new: loaded module_with_store={module_with_store:?} hash_value={hash_value:?}");

                                let vecs = write_guard.entry((channel, motif_width, hash_value)).or_insert((vec![], vec![], vec![]));

                                extract_cosine_conv_2d_once_(&mut vecs.0, &mut vecs.1, &mut vecs.2, module_with_store.data().module());

                                Ok((hash_value, stft_config))
                            }).await?;

                            Ok::<_, ThoError>(Some((hash_value, config)))
                        }
                        Err(e) => {
                            error!("ApplyMotifNets::new: can't load net for motif; there must have been an error while training the net: e={e:?}");
                            Ok::<_, ThoError>(None)
                        }
                    }
                }
            });

        let stft_run_configs: HashMap<u64, StftRunConfig> = collect_futures_with_progress(
            futures,
            trained_nets_len,
            crate::buffer_limits::APPLY_MOTIF_NETS_NEW_BUFFERED_LIMIT,
            StreamBuffering::Ordered, //to keep same order as in trained_nets_ids
            "ApplyMotifNets::new/load motif nets",
        )
            .await
            .into_iter()
            .flatten()
            .collect::<HashMap<_, _>>();

        let conv2d_tensors = Arc::into_inner(channels_widths_net_tensors)
            .expect("Arc::into_inner(channels_widths_net_tensors) failed")
            .into_inner()
            .into_iter()
            .map(|((channel, motif_width, stft_hash_value),(kernels, alphas, betas))|{
            MyCosineConv2dTensors {
                channel,
                motif_width,
                stft_hash_value,
                kernels:Tensor::cat(kernels.into_iter().map(|v|v.t).collect::<Vec<_>>().as_slice(), 0).to_device(ThoDevice::from(stage_ctx.device_idx()).into()).into(),//TODO to_device(Cuda(0))
                a: Tensor::cat(alphas.into_iter().map(|v|v.t).collect::<Vec<_>>().as_slice(), 0).to_device(ThoDevice::from(stage_ctx.device_idx()).into()).into(),
                b: Tensor::cat(betas.into_iter().map(|v|v.t).collect::<Vec<_>>().as_slice(), 0).to_device(ThoDevice::from(stage_ctx.device_idx()).into()).into(),
            }
        }).collect();


        if stft_run_configs.is_empty() {
            return Err(ThoError::new(format!("No motif nets for trained_net_ids={trained_net_ids:?}")));
        };
        debug!("ApplyMotifNets::new: stft_run_configs={stft_run_configs:?} conv2d_tensors={conv2d_tensors:?} trained_net_ids={trained_net_ids:?}");

        Ok(Self {
            id,
            pd: Default::default(),
            stft_run_configs,
            conv2d_tensors,
            slice_step,
            trained_net_ids
        })
    }

    pub fn num_nets(&self) -> usize {
        self.conv2d_tensors.iter().map(|v|v.kernels.size()[0] as usize).sum()
    }


    pub fn motif_nets_ids(&self) -> Vec<u32> {
        self.trained_net_ids.clone()
    }
}

impl<C> AsyncStage for ApplyMotifNetsMerged<C>
where
    C: RunCtx + Adjustable + 'static
{
    fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> impl Future<Output=Result<Arc<DevicesWrap<Self::Out>>, ThoError>> + Send {
        let input_id = input.id().clone();

        let self_clone = Arc::clone(&self);
        async move {

            let res = with_data!(ctx.branch("ApplyMotifNetsMerged::run".into(), false), input; |ctx, input:&ThoTensor<ChannelsData, SynKey>|{
                debug!("ApplyMotifNetsMerged::run: input={input:?}");
                let motifs_applied = self_clone.conv2d_tensors.iter().map(|tensors|{
                    let stft_config = self_clone.stft_run_configs.get(&tensors.stft_hash_value).unwrap();
                    debug!("ApplyMotifNetsMerged::run: tensors={tensors:?} slice_step={} stft_config={stft_config:?}", self_clone.slice_step);


                    let stft = SyncStft::<C>::new("".into(), stft_config.clone());

                    let t = input.t.f_i((tensors.channel as i64, ..))?;
                    let t = if t.size()[0] >= tensors.motif_width as i64 {
                        t
                    } else {
                        pad_to(&t, &[tensors.motif_width as i64], 0.0)?
                    };

                    let stft_t = stft.build_spectrogram(
                        &t
                    )?;

                    debug!("ApplyMotifNetsMerged::run: stft_t={stft_t:?}");
                    let window = tensors.motif_width as i64/stft_config.win_length.as_ref().map(|v|*v).unwrap_or(1);
                    let step = self_clone.slice_step as i64 /stft_config.hop_length.as_ref().map(|v|*v).unwrap_or(1);

                    debug!("ApplyMotifNetsMerged::run: step={step} window={window}");
                    //unfold time frames
                    let stft_t = stft_t.unfold(1,window,step);

                    debug!("ApplyMotifNetsMerged::run: unfolded stft_t={stft_t:?}");
                    let stft_t = stft_t.permute([1, 0, 2]);

                    debug!("ApplyMotifNetsMerged::run: permuted stft_t={stft_t:?}");

                     apply_cosine_kernel_bank_to_batched_spectrogram(
                       &stft_t.unsqueeze(1),
                       tensors.kernels.as_ref(),
                       tensors.a.as_ref(),
                       tensors.b.as_ref(),
                       CosineConv2dBankConfig::default()
                    ).and_then(|v|{
                        let unbound = v.f_squeeze_dim(2)?.f_unbind(1)?;
                        debug!("ApplyMotifNetsMerged::run: unbound={unbound:?}");
                        unbound.into_iter().map(|v| ThoTensor::<MotifProbability, SynKey>::new("".into(), v)).collect::<Result<Vec<_>, ThoError>>()
                    })
                }).collect::<Result<Vec<_>, ThoError>>()?
                    .into_iter()
                    .flatten()
                    .collect::<Vec<ThoTensor<MotifProbability, SynKey>>>();

                debug!("ApplyMotifNetsMerged::run: motifs_applied={motifs_applied:?}");

                Ok(motifs_applied)
            })?;

            //TODO stretch motifs in BatchAppliedMotifNets
            let batch = Arc::new(BatchAppliedMotifNets::<C>::new(mono_key!(0, "BatchAppliedMotifNets").into()));

            match batch.consume(BatchAppliedMotifNetsInput::new(self.out_id(input.id()), res, ctx.device_idx()), ctx.branch("BatchAppliedMotifNetsInput".into(), false)) {
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
    kernels: ThoTensor,
    a: ThoTensor,
    b: ThoTensor,
}

fn extract_cosine_conv_2d(mut nets: Vec<Box<dyn ThoModule<Batch<Slice>, MotifProbability>>>) -> MyCosineConv2dTensors {

    let mut kernels = Vec::with_capacity(nets.len());
    let mut alphas = Vec::with_capacity(nets.len());
    let mut betas = Vec::with_capacity(nets.len());
    for net in &mut nets {
        extract_cosine_conv_2d_once(&mut kernels, &mut alphas, &mut betas, net);
    }



    MyCosineConv2dTensors {
        channel: 0,
        motif_width: 0,
        stft_hash_value: 0,
        kernels: Tensor::cat(kernels.as_slice(), 0).copy().into(),
        a: Tensor::cat(alphas.as_slice(), 0).copy().into(),
        b: Tensor::cat(betas.as_slice(), 0).copy().into(),
    }
}

fn extract_cosine_conv_2d_once(kernels: &mut Vec<Tensor>, alphas: &mut Vec<Tensor>, betas: &mut Vec<Tensor>, net: &Box<dyn ThoModule<Batch<Slice>, MotifProbability>>) {
    let net_1_ref: &dyn ThoModule<Batch<Slice>, MotifProbability> = net.as_ref();
    debug!("extract_cosine_conv_2d_once: net_1_ref={net_1_ref:?}");

    let net_1_cosine_2_d = net_1_ref.as_any().downcast_ref::<MySequential<Batch<Slice>, MotifProbability>>().expect("downcast_mut fails");

    let layer = net_1_cosine_2_d.layers.get(1).expect("no MyCosineConv2d layer"); //TODO
    debug!("extract_cosine_conv_2d_once: layer={layer:?}");

    let net_1: &MyCosineConv2d = layer.as_ref().as_any().downcast_ref::<MyCosineConv2d>().expect("net_1_cosine_2_d");

    kernels.push(net_1.kernel.shallow_clone());
    alphas.push(net_1.alpha.unsqueeze(0).shallow_clone());
    betas.push(net_1.beta.unsqueeze(0).shallow_clone());
}

fn extract_cosine_conv_2d_once_(kernels: &mut Vec<ThoTensor>, alphas: &mut Vec<ThoTensor>, betas: &mut Vec<ThoTensor>, net: &Box<dyn ThoModule<Batch<Slice>, MotifProbability>>) {
    let net_1_ref: &dyn ThoModule<Batch<Slice>, MotifProbability> = net.as_ref();
    debug!("extract_cosine_conv_2d_once_: net_1_ref={net_1_ref:?}");

    let net_1_cosine_2_d = net_1_ref.as_any().downcast_ref::<MySequential<Batch<Slice>, MotifProbability>>().expect("downcast_mut fails");

    let layer = net_1_cosine_2_d.layers.get(1).expect("no MyCosineConv2d layer"); //TODO
    debug!("extract_cosine_conv_2d_once_: layer={layer:?}");

    let net_1: &MyCosineConv2d = layer.as_ref().as_any().downcast_ref::<MyCosineConv2d>().expect("net_1_cosine_2_d");

    kernels.push(ThoTensor::new((), net_1.kernel.shallow_clone()).unwrap());//TODO
    alphas.push(ThoTensor::new((),net_1.alpha.unsqueeze(0)).unwrap());
    betas.push(ThoTensor::new((),net_1.beta.unsqueeze(0)).unwrap());
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
    use crate::tasks::apply_motif_nets_merged::{apply_cosine_kernel_bank_to_batched_spectrogram, extract_cosine_conv_2d, ApplyMotifNetsMerged};

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

        let tensors = extract_cosine_conv_2d(vec![net_1, net_2]);

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

    async fn new_apply_motif_nets(trigger_oom_len: Option<usize>, slice_step: u32, select_motifs_run_config_id: RunConfigId, ws: &Arc<RwLock<Box<dyn Workspace>>>, store: &mut Arc<Box<dyn Store>>) -> Result<Arc<ApplyMotifNetsMerged<MockRunCtx>>, ThoError> {
        let apply_motif_nets = Arc::new(ApplyMotifNetsMerged::new(
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
