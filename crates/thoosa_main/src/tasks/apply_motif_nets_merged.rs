use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use log::error;
use parking_lot::RwLock;
use tch::{Kind, Tensor};
use tracing::debug;
use thoosa_common::dims::{AnyDims, Batch, ChannelsData, MotifProbability, Slice};
use thoosa_common::my_tch::{MySequential, ThoModule};
use thoosa_common::nodes::cosine_2d::{cosine_conv2d_bank_logits, CosineConv2dBankConfig, MyCosineConv2d};
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use thoosa_common::runner::common::RunCtx;
use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::ThoModuleWithStore;
use thoosa_common::ws::Workspace;
use thoosa_named_derive::Named;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::{collect_futures_with_progress, mono_key, StreamBuffering, ThoError};
use thoosa_shared::traits::Adjustable;
use thoosa_store::Store;
use crate::ApplyMotifNets;
use crate::tasks::load_motif_net::{LoadMotifNet, LoadMotifNetInput, MotifNetMeta};
use crate::tasks::stage_ids::APPLY_MOTIF_NETS;

#[derive(Debug, cached_derive::Cached, Named)]
pub struct ApplyMotifNetsMerged<C>
where
    C: RunCtx + Adjustable + 'static
{
    id: SynKey,
    pd:PhantomData<C>
}

impl<C> Stage for ApplyMotifNetsMerged<C>
where
    C: RunCtx + Adjustable + 'static
{
    type Ctx = C;
    type In<'a> = ThoTensor<ChannelsData, SynKey>;

    type Out = DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>;// same as in ApplyMotifNets
}

impl<C> AsyncStage for ApplyMotifNetsMerged<C>
where
    C: RunCtx + Adjustable + 'static
{


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
        trained_nets: Vec<(u32, PathBuf, f32)>,
        quantize: bool
    ) -> Result<ApplyMotifNetsMerged<C>, ThoError>
    where
        R: RunCtx + Adjustable + 'static
    {
        let load_motif_net = Arc::new(LoadMotifNet::<R>::new(&ws, quantize));//TODO move to new
        let trained_net_ids = trained_nets.iter().map(|v|v.0).collect::<Vec<_>>();
        let trained_nets_len = trained_nets.len();
        let id = mono_key!(APPLY_MOTIF_NETS, "ApplyMotifNets") + &SynKey::new(trained_nets.iter().map(|net| mono_key!(net.0, format!("trained_net_id_{}", net.0))).collect::<Vec<_>>());

        let mut kernels = Vec::with_capacity(trained_nets.len());
        let mut alphas = Vec::with_capacity(trained_nets.len());
        let mut betas = Vec::with_capacity(trained_nets.len());

        //this defines order in which motif nets should be applied to a resource before running label net
        let futures = trained_nets
            .into_iter()
            .map(|(id, path, _loss)| {
                let ws = Arc::clone(&ws);
                let load_motif_net = Arc::clone(&load_motif_net);
                let load_ctx = stage_ctx.branch(Some("apply_motif_nets_load_motif_net"), false);
                let load_ctx_2 = stage_ctx.branch(Some("apply_motif_nets_load_motif_net"), false);
                async move {
                    match load_motif_net.run_cached(Arc::new(DevicesWrap::wrap(LoadMotifNetInput::new(path, false))), load_ctx).await {
                        Ok(net) => {
                            log::debug!("ApplyMotifNets::new: loaded net={net:?}");

                            let tensors = net.with_data_ref(0, load_ctx_2,None, |v:&DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>, ctx|{

                                extract_cosine_conv_2d_once(&mut kernels, &mut alphas, &mut betas, v.data().module());

                                Ok(())
                            }).await?;

                            Ok::<_, ThoError>(Some((id, net)))
                        }
                        Err(e) => {
                            error!("ApplyMotifNets::new: can't load net for motif; there must have been an error while training the net: e={e:?}");
                            Ok::<_, ThoError>(None)
                        }
                    }
                }
            });

        let loaded_motif_nets = collect_futures_with_progress(
            futures,
            trained_nets_len,
            crate::buffer_limits::APPLY_MOTIF_NETS_NEW_BUFFERED_LIMIT,
            StreamBuffering::Ordered, //to keep same order as in trained_nets_ids
            "ApplyMotifNets::new/load motif nets",
        )
            .await
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        if loaded_motif_nets.is_empty() {
            return Err(ThoError::new(format!("No motif nets for trained_net_ids={trained_net_ids:?}")));
        };

        Ok(Self {
            id,
            pd: Default::default(),
        })
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

struct MyCosineConv2dTensors {
    kernels: Tensor,
    a: Tensor,
    b: Tensor,
}

fn extract_cosine_conv_2d(mut nets: Vec<Box<dyn ThoModule<AnyDims, AnyDims>>>) -> MyCosineConv2dTensors {

    let mut kernels = Vec::with_capacity(nets.len());
    let mut alphas = Vec::with_capacity(nets.len());
    let mut betas = Vec::with_capacity(nets.len());
    for net in &mut nets {
        extract_cosine_conv_2d_once(&mut kernels, &mut alphas, &mut betas, net);
    }



    MyCosineConv2dTensors {
        kernels: Tensor::cat(kernels.as_slice(), 0).copy(),
        a: Tensor::cat(alphas.as_slice(), 0).copy(),
        b: Tensor::cat(betas.as_slice(), 0).copy(),
    }
}

fn extract_cosine_conv_2d_once(kernels: &mut Vec<Tensor>, alphas: &mut Vec<Tensor>, betas: &mut Vec<Tensor>, net: &mut Box<dyn ThoModule<AnyDims, AnyDims>>) {
    let net_1_ref: &mut dyn ThoModule<AnyDims, AnyDims> = net.as_mut();
    let net_1_cosine_2_d = net_1_ref.as_any_mut().downcast_mut::<MySequential<AnyDims, AnyDims>>().expect("downcast_mut fails");

    let mut layer = net_1_cosine_2_d.layers.get_mut(1).expect("no MyCosineConv2d layer"); //TODO
    let mut net_1: &mut MyCosineConv2d = layer.as_mut().as_any_mut().downcast_mut::<MyCosineConv2d>().expect("net_1_cosine_2_d");

    kernels.push(net_1.kernel.shallow_clone());
    alphas.push(net_1.alpha.unsqueeze(0).shallow_clone());
    betas.push(net_1.beta.unsqueeze(0).shallow_clone());
}

#[cfg(test)]
mod tests {
    use std::mem;
    use std::sync::Arc;
    use log::LevelFilter::Trace;
    use log::{set_max_level, trace};
    use tch::Device::Cpu;
    use tch::kind::FLOAT_CPU;
    use tch::nn::VarStore;
    use tch::Tensor;
    use tracing::debug;
    use thoosa_common::dims::{AnyDims, Spectorgram};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{cosine_2d_motif_node, MotifCosine2dNetParams};
    use thoosa_common::my_tch::{InitDummy, MotifInitMode, MySequential, ThoModule};
    use thoosa_common::nodes::cosine_2d::{CosineConv2dBankConfig, MyCosineConv2d};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::ThoError;
    use crate::tasks::apply_motif_nets_merged::{apply_cosine_kernel_bank_to_batched_spectrogram, extract_cosine_conv_2d};

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
        let mut net_1:Box<dyn ThoModule<AnyDims, AnyDims>> = cosine_2d_motif_node(&MotifCosine2dNetParams::new(0, 0, 1.0, 0.0, MotifInitMode::Mean(InitDummy{})), Some(Arc::new(ThoTensor::new("motifs".into(),kernel_1.shallow_clone())?)))?.build_tch_module(&vs_0.root())?;


        let kernel_2 = rand_kernel()?;
        debug!("test_apply_cosine_kernel_bank_to_batched_spectrogram: kernel_2={kernel_2:?}");
        trace!("test_apply_cosine_kernel_bank_to_batched_spectrogram: kernel_2=\n{kernel_2}");

        let mut net_2:Box<dyn ThoModule<AnyDims, AnyDims>> = cosine_2d_motif_node(&MotifCosine2dNetParams::new(0, 0, 1.0, 0.0, MotifInitMode::Mean(InitDummy{})), Some(Arc::new(ThoTensor::new("motifs".into(),kernel_2.shallow_clone())?)))?.build_tch_module(&vs_0.root())?;

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
            &tensors.kernels,
            &tensors.a,
            &tensors.b,
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
}
