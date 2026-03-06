use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;

use cached_derive::Cached;
use log::{debug, info, warn};
use parking_lot::RwLock;
use tch::{Device, IndexOp, Kind, Tensor};
use thoosa_common::dims::Dims;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::runner::adjusters::splits::{SplitsAdjuster, SplitsAdjusterContextImpl};
use thoosa_shared::runner::cache_areas::CacheArea;
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::runner::oom_handler::{Adjusters, Contexts, NoopRunCtxContext, NoopRunnableContext, OOMHandler};
use thoosa_shared::traits::{Adjustable, MonoMetrics, Select, WithId};
use thoosa_shared::{mono_key, ThoError};
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_store::Store;
use thoosa_with_data::with_data;

use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::random_slices_stats::RANDOM_STATS_GROUP;

#[derive(Clone, Debug)]
pub struct WelchInputChannels;

impl Dims for WelchInputChannels {
    type Metrics = MonoMetrics<0, 2>; // split by channels, preserve full time axis
}

#[derive(Clone, Debug)]
pub struct ChannelsWelchPsd;

impl Dims for ChannelsWelchPsd {
    type Metrics = MonoMetrics<0, 2>; // channels x frequency_bins
}

#[derive(Debug, Clone)]
pub struct WelchChannelsSelector(pub Range<MonoMetrics<0, 2>>);

impl Default for WelchChannelsSelector {
    fn default() -> Self {
        Self(Default::default()..Default::default())
    }
}

impl From<Range<MonoMetrics<0, 2>>> for WelchChannelsSelector {
    fn from(value: Range<MonoMetrics<0, 2>>) -> Self {
        Self(value)
    }
}

impl Select<WelchChannelsSelector> for ThoTensor<WelchInputChannels, SynKey> {
    type Output = ThoTensor<WelchInputChannels, SynKey>;

    fn subview(&self, selector: WelchChannelsSelector) -> Result<Self::Output, ThoError> {
        let size: MonoMetrics<0, 2> = self.size().try_into()?;
        let ranges = MonoMetrics::<0, 2>::map_to_ranges(&selector.0.start, &selector.0.end)?;
        debug!("WelchChannelsSelector::subview: ranges={ranges:?} size={size:?}");
        let sub_t = self.t_ref().f_i((ranges[0].clone(), ranges[1].clone()))?;
        ThoTensor::new(self.id().clone(), sub_t)
    }
}

impl Adjustable for WelchChannelsSelector {}

#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct FrequenciesStatsInput<RC>
where
    RC: ResourcesConfig + Debug + Send + Sync + Clone,
{
    id: SynKey,
    resources: Option<ResourcesConfigInput<RC>>,
    nperseg: u32,
    noverlap: u32,
}

impl<RC> FrequenciesStatsInput<RC>
where
    RC: ResourcesConfig + Debug + Send + Sync + Clone,
{
    pub fn new(resources: Option<ResourcesConfigInput<RC>>, nperseg: u32, noverlap: u32) -> Self {
        let resources_key = resources
            .as_ref()
            .map(|v| format!("{:?}", v.resources_config()))
            .unwrap_or_else(|| format!("not_in_group:{RANDOM_STATS_GROUP}"));

        Self {
            id: SynKey::new([
                mono_key!("FrequenciesStatsInput", "stage"),
                mono_key!(resources_key.clone(), format!("resources={resources_key}")),
                mono_key!(nperseg, format!("nperseg={nperseg}")),
                mono_key!(noverlap, format!("noverlap={noverlap}")),
            ]),
            resources,
            nperseg,
            noverlap,
        }
    }

    pub fn resources(&self) -> Option<&ResourcesConfigInput<RC>> {
        self.resources.as_ref()
    }

    pub fn nperseg(&self) -> u32 {
        self.nperseg
    }

    pub fn noverlap(&self) -> u32 {
        self.noverlap
    }
}

impl<RC> Input for FrequenciesStatsInput<RC>
where
    RC: ResourcesConfig + Debug + Send + Sync + Clone,
{
}

impl<RC> WithId for FrequenciesStatsInput<RC>
where
    RC: ResourcesConfig + Debug + Send + Sync + Clone,
{
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug, Named)]
struct WelchPsdPerResource<C>
where
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    sample_rate: f64,
    nperseg: i64,
    noverlap: i64,
    pd: PhantomData<C>,
}

impl<C> WelchPsdPerResource<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn new(sample_rate: f64, nperseg: i64, noverlap: i64) -> Self {
        Self {
            id: SynKey::new([mono_key!("WelchPsdPerResource", "stage")]),
            sample_rate,
            nperseg,
            noverlap,
            pd: PhantomData,
        }
    }
}

impl<C> Adjustable for WelchPsdPerResource<C> where C: RunCtx + Adjustable + 'static {}

impl<C> OutId for WelchPsdPerResource<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        mono_key!("welch_psd", "welch_psd") + input_id
    }
}

impl<C> Stage for WelchPsdPerResource<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = ThoTensor<WelchInputChannels, SynKey>;
    type Out = ThoTensor<ChannelsWelchPsd, SynKey>;
}

impl<C> SyncStage for WelchPsdPerResource<C>
where
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="WelchPsdPerResource::do_run"))]
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let t = input.t_ref();
        let size = t.size();
        if size.len() != 2 {
            return Err(ThoError::new(format!("WelchPsdPerResource expected [C, T], got {:?}", size)));
        }
        let channels = size[0];
        let out_id = self.out_id(input.id());

        let mut psd_by_channel = Vec::with_capacity(channels as usize);
        for channel in 0..channels {
            let x = t.f_i((channel, ..))?;
            let psd = welch_psd_1d(&x, self.sample_rate, self.nperseg, self.noverlap)?;
            psd_by_channel.push(psd);
        }

        let out = if psd_by_channel.is_empty() {
            Tensor::f_empty([0, welch_freq_bins(self.nperseg)], (Kind::Float, t.device()))?
        } else {
            Tensor::f_stack(psd_by_channel.as_slice(), 0)?
        };

        ThoTensor::new(out_id, out)
    }
}

#[derive(Debug, Named)]
pub struct WelchPSD<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug + Send + Sync + Clone + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + Send + Sync + 'static,
    C: RunCtx + Adjustable + 'static,
{
    store: Arc<Box<dyn Store>>,
    loader: Arc<L>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    pd: PhantomData<(RC, C)>,
}

impl<RC, L, C> WelchPSD<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug + Send + Sync + Clone + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + Send + Sync + 'static,
    C: RunCtx + Adjustable + 'static,
{
    pub fn new(store: Arc<Box<dyn Store>>, loader: Arc<L>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Self {
        Self {
            store,
            loader,
            ws: Arc::clone(ws),
            pd: PhantomData,
        }
    }

    async fn load_resources_for_input(&self, resources_config: Option<RC>) -> Result<Vec<Resource>, ThoError> {
        match resources_config {
            Some(resources_config) => resources_config.to_resources(&self.store).await,
            None => {
                let resources = self.store.list_resources_not_in_group(RANDOM_STATS_GROUP, u32::MAX).await?;
                if resources.is_empty() {
                    warn!("WelchPSD::load_resources_for_input: no resources found outside group={RANDOM_STATS_GROUP}");
                }
                Ok(resources)
            }
        }
    }
}

impl<RC, L, C> OutId for WelchPSD<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug + Send + Sync + Clone + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + Send + Sync + 'static,
    C: RunCtx + Adjustable + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        mono_key!("welch_psd", "welch_psd") + input_id
    }
}

impl<RC, L, C> Stage for WelchPSD<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug + Send + Sync + Clone + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + Send + Sync + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = FrequenciesStatsInput<RC>;
    type Out = ThoTensor<ChannelsWelchPsd, SynKey>;
}

impl<RC, L, C> AsyncStage for WelchPSD<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug + Send + Sync + Clone + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + Send + Sync + 'static,
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="WelchPSD::run"))]
    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<FrequenciesStatsInput<RC>>>,
        mut ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError>
    where
        for<'a> Self::In<'a>: Send + Sync,
    {
        let cache_area = ctx.cache_area();
        let input_id = input.id().clone();

        let (resources_config, nperseg, noverlap) = with_data!(
            ctx.branch(Some("welch_psd_input"), false, false),
            input;
            |_ctx, input: &FrequenciesStatsInput<RC>| {
                Ok::<_, ThoError>((
                    input.resources().map(|v| v.resources_config().clone()),
                    input.nperseg() as i64,
                    input.noverlap() as i64,
                ))
            }
        )?;

        validate_welch_params(nperseg, noverlap)?;

        let sample_rate = thoosa_shared::with_perf_log(
            || self.ws.try_read_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
            "WelchPSD/run/read_ws",
        )
        .sample_rate() as f64;

        let resources = self.load_resources_for_input(resources_config).await?;
        if resources.is_empty() {
            return Err(ThoError::new("WelchPSD: no resources to process"));
        }

        let welch_stage = Arc::new(WelchPsdPerResource::<C>::new(sample_rate, nperseg, noverlap));
        let mut psd_sum: Option<Tensor> = None;
        let mut processed_resources = 0usize;

        for resource in resources.into_iter() {
            if resource.length() < nperseg as u32 {
                return Err(ThoError::new(format!(
                    "WelchPSD: resource_id={} length={} is shorter than nperseg={nperseg}",
                    resource.id(), resource.length()
                )));
            }

            let loaded = Arc::clone(&self.loader)
                .run_cached(
                    Arc::new(DevicesWrap::wrap(resource.clone(), CacheArea::default())),
                    ctx.branch(Some(&format!("welch_psd_load_resource_{}", resource.id())), false, false),
                )
                .await?;

            let welch_input = loaded
                .with_data_ref(0, ctx.branch(Some("welch_psd_wrap_input"), false, false), None, |loaded: &ThoTensor<ChannelsData2, SynKey>, _ctx| {
                    ThoTensor::<WelchInputChannels, SynKey>::new(loaded.id().clone(), loaded.t_ref().shallow_clone())
                })
                .await?;
            let welch_input = Arc::new(DevicesWrap::wrap(welch_input, cache_area.into()));

            let welch_input_size = welch_input.size().clone();
            let out_id = welch_stage.out_id(welch_input.id());
            let channels = welch_input_size.dim_0();

            debug!("WelchPSD::run: resource_id={} welch_input_size={welch_input_size:?} nperseg={nperseg} noverlap={noverlap}", resource.id());

            let resource_psd = OOMHandler::new(
                vec![Adjusters::Input(Box::new(SplitsAdjuster::<WelchChannelsSelector, MonoMetrics<0, 2>>::new([1, 0].into())))],
                None,
            )
            .run(
                &welch_stage,
                Arc::clone(&welch_input),
                ctx.branch(Some(&format!("welch_psd_resource_{}", resource.id())), false, false),
                Contexts::new(
                    SplitsAdjusterContextImpl::new(
                        welch_input_size,
                        SingleDimTensorOutputBuilder::new(
                            out_id,
                            0,
                            &[channels, welch_freq_bins(nperseg)],
                            (Kind::Float, Device::Cpu),
                            BuildAs::Append,
                        )?,
                    ),
                    NoopRunnableContext {},
                    NoopRunCtxContext {},
                ),
            )
            .await?;

            let resource_psd_t = resource_psd.t_ref().shallow_clone();

            psd_sum = Some(match psd_sum {
                None => resource_psd_t,
                Some(acc) => acc.f_add(&resource_psd_t)?,
            });
            processed_resources += 1;

            if processed_resources % 32 == 0 {
                info!("WelchPSD::run: processed {processed_resources} resources");
            }
        }

        let psd_sum = psd_sum.ok_or_else(|| ThoError::new("WelchPSD: no PSD values computed"))?;
        let psd_mean = psd_sum.f_div_scalar(processed_resources as f64)?;
        let out = ThoTensor::new(self.out_id(&input_id), psd_mean)?;

        Ok(Arc::new(DevicesWrap::wrap(out, cache_area.into())))
    }
}

fn validate_welch_params(nperseg: i64, noverlap: i64) -> Result<(), ThoError> {
    if nperseg <= 1 {
        return Err(ThoError::new(format!("WelchPSD: nperseg must be > 1, got {nperseg}")));
    }
    if noverlap < 0 {
        return Err(ThoError::new(format!("WelchPSD: noverlap must be >= 0, got {noverlap}")));
    }
    if noverlap >= nperseg {
        return Err(ThoError::new(format!("WelchPSD: noverlap must be < nperseg, got noverlap={noverlap} nperseg={nperseg}")));
    }
    Ok(())
}

fn welch_freq_bins(nperseg: i64) -> i64 {
    (nperseg / 2) + 1
}

fn hann_window_periodic(length: i64, device: Device) -> Result<Tensor, ThoError> {
    if length <= 1 {
        return Err(ThoError::new(format!("hann_window_periodic length must be > 1, got {length}")));
    }

    let n = Tensor::f_arange(length, (Kind::Float, device))?;
    let factor = 2.0 * std::f64::consts::PI / length as f64;
    let phase = n.f_mul_scalar(factor)?;
    let cos = phase.f_cos()?;
    Ok(cos.f_neg()?.f_add_scalar(1.0)?.f_mul_scalar(0.5)?)
}

pub fn welch_psd_1d(x: &Tensor, fs: f64, nperseg: i64, noverlap: i64) -> Result<Tensor, ThoError> {
    validate_welch_params(nperseg, noverlap)?;
    if fs <= 0.0 {
        return Err(ThoError::new(format!("WelchPSD: fs must be positive, got {fs}")));
    }
    if x.dim() != 1 {
        return Err(ThoError::new(format!("WelchPSD: expected 1D tensor, got shape {:?}", x.size())));
    }

    let x = x.f_to_kind(Kind::Float)?;
    let x = x.f_subtract(&x.f_mean(Kind::Float)?)?;
    let t = x.size()[0];
    if t < nperseg {
        return Err(ThoError::new(format!("WelchPSD: signal shorter than nperseg; len={t} nperseg={nperseg}")));
    }

    let step = nperseg - noverlap;
    let win = hann_window_periodic(nperseg, x.device())?;
    let win_power = win.f_pow_tensor_scalar(2)?.f_sum(Kind::Float)?;

    let segs = x.f_unfold(0, nperseg, step)?;
    if segs.size().first().copied().unwrap_or(0) == 0 {
        return Err(ThoError::new("WelchPSD: no segments produced"));
    }
    let segs = segs.f_mul(&win)?;

    let x_fft = segs.f_fft_rfft(Some(nperseg), 1, "forward")?;
    let periodograms = x_fft
        .f_abs()?
        .f_pow_tensor_scalar(2)?
        .f_div(&win_power.f_mul_scalar(fs)?)?;

    Ok(periodograms.f_mean_dim(0, false, None)?)
}

#[cfg(test)]
mod tests {
    use super::welch_psd_1d;
    use tch::Kind::Float;
    use tch::Tensor;
    use thoosa_shared::ThoError;

    #[test]
    fn welch_psd_detects_tone_peak() -> Result<(), ThoError> {
        let fs = 1000.0f64;
        let freq = 50.0f64;
        let n = 2000i64;
        let idx = Tensor::f_arange(n, (Float, tch::Device::Cpu))?;
        let phase = idx.f_mul_scalar(2.0 * std::f64::consts::PI * freq / fs)?;
        let x = phase.f_sin()?;

        let nperseg = 512i64;
        let psd = welch_psd_1d(&x, fs, nperseg, 256)?;
        let peak_idx = psd.f_argmax(0, false)?.int64_value(&[]);
        let peak_hz = peak_idx as f64 * fs / nperseg as f64;

        assert!((peak_hz - freq).abs() <= fs / nperseg as f64 * 2.0, "peak_hz={peak_hz} freq={freq}");
        Ok(())
    }
}
