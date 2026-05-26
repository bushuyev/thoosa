use std::marker::PhantomData;
use tch::{Kind, Tensor};
use tracing::debug;
use thoosa_common::dims::{ChannelsData, MotifProbability};
use thoosa_common::nodes::cosine_2d::{cosine_conv2d_bank_logits, CosineConv2dBankConfig};
use thoosa_common::runner::common::RunCtx;
use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_named_derive::Named;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::ThoError;
use thoosa_shared::traits::Adjustable;

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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use log::LevelFilter::Trace;
    use log::{set_max_level, trace};
    use mockall::Any;
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
    use crate::tasks::apply_motif_nets_merged::apply_cosine_kernel_bank_to_batched_spectrogram;

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

        let net_2:Box<dyn ThoModule<AnyDims, AnyDims>> = cosine_2d_motif_node(&MotifCosine2dNetParams::new(0, 0, 1.0, 0.0, MotifInitMode::Mean(InitDummy{})), Some(Arc::new(ThoTensor::new("motifs".into(),kernel_2.shallow_clone())?)))?.build_tch_module(&vs_0.root())?;

        // let raw_tensor = Tensor::arange(1000.0, FLOAT_CPU);
        // debug!("raw_tensor=\n{raw_tensor}");
        // let stft_t = raw_tensor.unfold(0, motif_width, 100).stft(50, 10, 20, None::<Tensor>, true, true, true, true);
        // debug!("stft_t={stft_t:?}"); - 10, 26, 9

        let stft_t = Tensor::ones([3, 26, 9], FLOAT_CPU).into();

        let r_1 = net_1.tho_forward(&stft_t, None, false)?;
        // let r_2 = net_2.tho_forward(&stft_t, None, false)?;


        // debug!("r_2=\n{r_2}");

        debug!("test_apply_cosine_kernel_bank_to_batched_spectrogram: net_1={net_1:?} type_name={:?}", net_1.type_name());
        let mut leak1 = Box::leak(net_1);
        let net_1_cosine_2_d = leak1.as_any_mut().downcast_mut::<MySequential<AnyDims, AnyDims>>().expect("net_1");

        let leak2 = Box::leak(net_1_cosine_2_d.layers.remove(1));
        let net_1_ = leak2.as_any().downcast_ref::<MyCosineConv2d>().expect("net_1_cosine_2_d");;

        let r = apply_cosine_kernel_bank_to_batched_spectrogram(
            &stft_t.as_ref().unsqueeze(1),
            &Tensor::stack(&[net_1_.kernel.shallow_clone()/*, kernel_2*/], 0),
            &Tensor::ones(&[1], FLOAT_CPU),
            &Tensor::zeros(&[1], FLOAT_CPU),
            CosineConv2dBankConfig::default()
        )?;

        debug!("r_1=\n{r_1}");
        debug!("r=\n{r}");

        Ok(())
    }

    fn rand_kernel() -> Result<Tensor, ThoError> {
        // Ok(Some(Arc::new(ThoTensor::new("motifs".into(), Tensor::stack((0..26 /* freqs (Kh) - 50/2+1 */).map(|i| Tensor::arange(9, FLOAT_CPU).sin).collect::<Vec<_>>().as_slice(), 0).unsqueeze(0).shallow_clone())?)))
        Ok(Tensor::rand(&[1, 26, 9], FLOAT_CPU))
    }
}