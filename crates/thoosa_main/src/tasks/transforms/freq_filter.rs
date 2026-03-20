use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use log::debug;
use parking_lot::RwLock;
use tch::{IndexOp, Tensor};
use tch::Device::Cuda;
use cached_derive::Cached;
use thoosa_named_derive::Named;
use thoosa_common::debug_plot;
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use thoosa_shared::runner::common::OutId;
use thoosa_shared::runner::common::RunCtx;
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage};
use thoosa_shared::runner::global_ctx::GlobalCtx;
use thoosa_common::dims::{AnyMetrics, Dims};
use thoosa_common::find_motifs_run_config::InputFilter;
use thoosa_common::transforms::filters::{band_pass_filter, notch_band_rfft, Padding};
use thoosa_shared::{mono_key, ThoError};
use thoosa_shared::traits::{CachableIdTr, WithId};
use thoosa_shared::comp_id::{MonoKey, SynKey};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::ws::Workspace;
use thoosa_shared::run_configs::RunId;
use thoosa_shared::math::sg_smoothing_coeffs;
use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::transforms::smoothing::{median, moving_average};
use thoosa_shared::runner::run_ctx_2::RunCtxWithPermits;

#[derive(Debug, Clone)]
struct BandPassDims;

impl Dims for BandPassDims {
    type Metrics = AnyMetrics;//TODO
}

#[derive(Debug)]
enum PreparedFilter {
    NotchBand { low_hz: f64, high_hz: f64 },
    BandPass {
        kernel: ThoTensor<BandPassDims, SynKey>,
        stride: u32,
        padding: Padding,
    },
    SavGol {
        weights: ThoTensor<BandPassDims, SynKey>,
        window_len: i64,
        stride: u32,
        padding: Padding,
    },
    Median {
        window: u32,
    },
    MovingAverage {
        window: u32,
    },
}

#[derive(Debug, Cached, Named)]
pub struct FreqFilter<C>
where
    C: RunCtx + 'static,
{
    id:MonoKey,
    filters: Vec<PreparedFilter>,
    _ctx: PhantomData<C>,
    num_channels:i64,
    sample_rate:f64
}
//repalce with two config - band pass and band notch
impl<C> FreqFilter<C>
where
    C: RunCtx + 'static,
{
    pub fn new(filter_configs: &[InputFilter], ws: &Arc<RwLock<Box<dyn Workspace>>>, find_motifs_run_config_id: RunId) -> Result<Self, ThoError> {
        let (num_channels, sample_rate)  = {
            let guard = thoosa_shared::with_perf_log(
                || ws.try_read_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
                "FreqFilter::new/ws_read",
            );
            (guard.channels() as i64, guard.sample_rate() as f64)
        };

        let mut filters = Vec::with_capacity(filter_configs.len());
        for filter_config in filter_configs {
            match filter_config {
                InputFilter::NotchBandFilter(notch_config) => {
                    filters.push(PreparedFilter::NotchBand {
                        low_hz: notch_config.low_hz,
                        high_hz: notch_config.high_hz,
                    });
                }
                InputFilter::BandPassFilter(band_pass_config) => {
                    let high_hz = band_pass_config.low_hz + band_pass_config.band_hz;

                    let kernel = band_pass_filter(
                        band_pass_config.low_hz,
                        high_hz,
                        band_pass_config.kernel_size as i64,
                        sample_rate,
                        1e-6
                    )?.f_repeat([num_channels, 1, 1])?.f_to_device(Cuda(0))?;//TODO device

                    debug!("FreqFilter::new: band_pass_config={band_pass_config:?} kernel.size={:?}", kernel.size());

                    filters.push(PreparedFilter::BandPass {
                        kernel: ThoTensor::new(
                            SynKey::new([
                                mono_key!(format!("{}", band_pass_config.low_hz), format!("{} low_hz", band_pass_config.low_hz)),
                                mono_key!(format!("{}", high_hz), format!("{} high_hz", high_hz) ),
                            ]),
                            kernel
                        )?,
                        stride: band_pass_config.stride,
                        padding: band_pass_config.padding,
                    });
                }
                InputFilter::SavGolFilter(sav_gol_config) => {
                    let window_len = if sav_gol_config.window_len % 2 == 0 {
                        sav_gol_config.window_len + 1 //TODO add constraint in TPE
                    } else {
                        sav_gol_config.window_len
                    };

                    if window_len < 3  {
                        return Err(ThoError::new(format!("window_len must be odd and >= 3, got {}", window_len)));
                    }

                    if sav_gol_config.poly_order < 0 || sav_gol_config.poly_order >= window_len as i64 {
                        return Err(ThoError::new(format!("poly_order must be >= 0 and < window_len, got poly_order={} window_len={}", sav_gol_config.poly_order, window_len)));
                    }

                    let coeffs = sg_smoothing_coeffs(window_len as usize, sav_gol_config.poly_order as usize)?;
                    let weights = Tensor::from_slice(&coeffs)
                        .f_to_kind(tch::Kind::Float)?
                        .f_to_device(Cuda(0))?
                        .f_view([1, 1, window_len as i64])?
                        .f_repeat([num_channels, 1, 1])?;

                    filters.push(PreparedFilter::SavGol {
                        weights: ThoTensor::new(
                            SynKey::new([
                                mono_key!(format!("{}", sav_gol_config.window_len), "window_len"),
                                mono_key!(format!("{}", sav_gol_config.poly_order), "poly_order"),
                                mono_key!(format!("{:?}", sav_gol_config.padding), "padding"),
                            ]),
                            weights
                        )?,
                        window_len: window_len as i64,
                        stride: sav_gol_config.stride,
                        padding: sav_gol_config.padding,
                    });
                }
                InputFilter::MedianFilter(median_config) => {
                    let effective_window = if median_config.window % 2 == 0 {
                        median_config.window + 1
                    } else {
                        median_config.window
                    };
                    if effective_window <= 3 {
                        return Err(ThoError::new(format!("median window must be > 3 after odd normalization, got {}", median_config.window)));
                    }
                    filters.push(PreparedFilter::Median {
                        window: median_config.window,
                    });
                }
                InputFilter::AverageFilter(moving_average_config) => {
                    let effective_window = if moving_average_config.window % 2 == 0 {
                        moving_average_config.window + 1
                    } else {
                        moving_average_config.window
                    };
                    if effective_window <= 3 {
                        return Err(ThoError::new(format!("moving average window must be > 3 after odd normalization, got {}", moving_average_config.window)));
                    }
                    filters.push(PreparedFilter::MovingAverage {
                        window: moving_average_config.window,
                    });
                }
            }
        }

        Ok(Self {
            id: mono_key!(find_motifs_run_config_id, "FreqFilter"),
            filters,
            _ctx: PhantomData,
            num_channels,
            sample_rate
        })
    }

    pub fn apply_tho_tensor(&self, input_tensor: &ThoTensor<ChannelsData2, SynKey>) -> Result<ThoTensor<ChannelsData2, SynKey>, ThoError> {
        let mut filtered_t = input_tensor.t_ref().shallow_clone();
        for filter in self.filters.iter() {
            match filter {
                PreparedFilter::NotchBand { low_hz, high_hz } => {
                    let filtered_t_out = notch_band_rfft(&filtered_t, self.sample_rate, *low_hz, *high_hz)?;

                    debug!("FreqFilter::apply_tho_tensor, NotchBand: id={} filtered_t.size={:?} filtered_t_out.size={:?}", input_tensor.id(), filtered_t.size(), filtered_t_out.size());

                    filtered_t = filtered_t_out;
                    // debug_plot!(&filtered_t, "{}_NotchBand", input_tensor.id())?;
                }
                PreparedFilter::BandPass { kernel, stride, padding } => {

                    let input_t = filtered_t.as_ref().f_unsqueeze(0)?;

                    let output = self.run_conv_with_padding(*stride, padding, &kernel.as_ref(), &input_t);//TODO conf stride?
                    debug!("FreqFilter::apply_tho_tensor, BandPass: id={} kernel.size={:?} input.size={:?} output.size={:?}", input_tensor.id(), kernel.size(), input_t.size(), output.as_ref().map(|v|v.size()));
                    filtered_t = output?


                    // debug_plot!(&filtered_t, "{}_BandPass", input_tensor.id())?;
                }
                PreparedFilter::SavGol { weights, window_len, stride, padding } => {
                    let t = filtered_t.size()[1];
                    if t < *window_len {
                        return Err(ThoError::new(format!("time length T={} must be >= window_len={}", t, window_len)));
                    }

                    let input_t = filtered_t.as_ref().f_unsqueeze(0)?;


                    let output = self.run_conv_with_padding(*stride, padding, &weights.as_ref(), &input_t);
                    debug!("FreqFilter::apply_tho_tensor, SavGol: id={} weights.size={:?} input.size={:?} output.size={:?}", input_tensor.id(), weights.size(), input_t.size(), output.as_ref().map(|v|v.size()));
                    filtered_t = output?

                }
                PreparedFilter::Median { window } => {
                    let filtered_t_out = median(&filtered_t, *window)?;
                    debug!("FreqFilter::apply_tho_tensor, Median: id={} window={} filtered_t.size={:?} filtered_t_out.size={:?}", input_tensor.id(), window, filtered_t.size(), filtered_t_out.size());
                    filtered_t = filtered_t_out;
                }
                PreparedFilter::MovingAverage { window } => {
                    let filtered_t_out = moving_average(&filtered_t, *window)?;
                    debug!("FreqFilter::apply_tho_tensor, MovingAverage: id={} window={} filtered_t.size={:?} filtered_t_out.size={:?}", input_tensor.id(), window, filtered_t.size(), filtered_t_out.size());
                    filtered_t = filtered_t_out;
                }
            }
        }

        Ok(ThoTensor::new(self.out_id(input_tensor.id()), filtered_t)?)
    }

    fn run_conv_with_padding(&self, stride: u32, padding: &Padding, kernel_t: &Tensor, input_t: &Tensor) -> Result<Tensor, ThoError> {
        let t = input_t.size()[1];
        let kernel_size = kernel_t.size()[2];
        let pad = kernel_size / 2;
        let stride_i64 = stride as i64;


        let output_3d = match padding {
            Padding::Zero(_) => input_t.f_conv1d(&kernel_t, None::<Tensor>, stride_i64, pad, 1, self.num_channels)?,
            Padding::Replicate(_) => {
                let left = input_t.f_narrow(2, 0, 1)?.f_repeat([1, 1, pad])?;
                let right = input_t.f_narrow(2, t - 1, 1)?.f_repeat([1, 1, pad])?;
                let xpad = Tensor::f_cat(&[&left, input_t, &right], 2)?;
                xpad.f_conv1d(&kernel_t, None::<Tensor>, stride_i64, 0, 1, self.num_channels)?
            }
            Padding::Reflect(_) => {
                if pad >= t {
                    return Err(ThoError::new(format!("reflect padding requires pad < T, got pad={} T={}", pad, t)));
                }
                let left = input_t.f_narrow(2, 1, pad)?.f_flip([2])?;
                let right = input_t.f_narrow(2, t - pad - 1, pad)?.f_flip([2])?;
                let xpad = Tensor::f_cat(&[&left, input_t, &right], 2)?;
                xpad.f_conv1d(&kernel_t, None::<Tensor>, stride_i64, 0, 1, self.num_channels)?
            }
        };

        Ok(output_3d.f_view((self.num_channels, -1))?)
    }
}

impl<C> OutId for FreqFilter<C>
where
    C: RunCtx + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<C> Stage for FreqFilter<C>
where
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = ThoTensor<ChannelsData2, SynKey>;
    type Out = ThoTensor<ChannelsData2, SynKey>;
}

// #[async_trait]
impl<C> AsyncStage for FreqFilter<C>
where
    C: RunCtx + 'static,
{

    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        mut ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {

        let device_idx = ctx.device_idx();
        let cache_area = ctx.cache_area();
        let out_key = {
            let out_id = self.out_id(input.id());
            Arc::new(out_id) as Arc<dyn CachableIdTr>
        };

        let num_devices = GlobalCtx::global().device_count();
        let mut ctx_for_data = ctx.branch(Some("freq_filter_data"), false, false);

        let filtered = input
            .with_data_ref(
                device_idx,
               ctx_for_data, None, |input_tensor: &ThoTensor<ChannelsData2, SynKey>, _ctx| {

               self.apply_tho_tensor(input_tensor)
           })
            .await?;

        Ok(Arc::new(DevicesWrap::wrap(filtered, cache_area.into())))
    }
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use log::debug;
    use parking_lot::RwLock;
    use tch::kind::FLOAT_CPU;
    use tch::Tensor;
    use thoosa_shared::runner::common::cachable::DevicesWrap;
    use thoosa_shared::runner::cache_areas::CacheArea;
    use thoosa_shared::runner::common::runnable::AsyncStage;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig, InputFilter};
    use thoosa_common::transforms::filters::{BandPassFilterParams, MedianFilterParams, AverageFilterParams, Padding, SavGolFilterParams};
    use thoosa_shared::ThoError;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::utils::from;
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_shared::runner::run_ctx_2::RunCtxWithPermits;
    use crate::tasks::transforms::smoothing::{median, moving_average};
    use crate::tasks::transforms::freq_filter::FreqFilter;
    use thoosa_shared::runner::common::RunCtx;
    use thoosa_with_data::with_data;

    #[tokio::test]
    async fn test_band_pass() ->Result<(), ThoError> {

        let filter_configs = vec![
            // InputFilter::NotchBandFilter(thoosa_common::transforms::filters::NotchBandFilterParams {
            //     low_hz: 58.0,
            //     high_hz: 62.0,
            // }),
            // InputFilter::SavGolFilter(SavGolFilterParams {
            //     window_len: 11,
            //     poly_order: 2,
            //     padding: SgPadding::reflect(),
            // }),
            InputFilter::BandPassFilter(BandPassFilterParams{
                low_hz: 0.0,
                band_hz: 100.0,
                kernel_size: 859,
                stride: 10,
                padding: Padding::zero(),
            }),
        ];


        run_freq_filter(filter_configs).await
    }

    #[tokio::test]
    async fn test_savitzky_golay() ->Result<(), ThoError> {

        let filter_configs = vec![

            InputFilter::SavGolFilter(SavGolFilterParams {
                window_len: 11,
                poly_order: 2,
                stride: 1,
                padding: Padding::reflect(),
            }),
        ];


        run_freq_filter(filter_configs).await
    }

    #[test]
    fn test_apply_tho_tensor_with_median_filter() -> Result<(), ThoError> {
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
        let freq_filter = FreqFilter::<RunCtxWithPermits>::new(&[InputFilter::MedianFilter(MedianFilterParams { window: 5 })], &ws, 0)?;
        let input_t = Tensor::from_slice(&[1.0f32, 10.0, 2.0, 9.0, 3.0, 5.0, 4.0, 3.0, 2.0, 1.0]).f_view([2, 5])?;
        let input = ThoTensor::new(SynKey::new([mono_key!("median", "median")]), input_t)?;

        let output = freq_filter.apply_tho_tensor(&input)?;
        let expected = median(input.as_ref(), 5)?;
        let output_flat = output.as_ref().f_view([-1])?;
        let expected_flat = expected.f_view([-1])?;

        assert_eq!(from::<Vec<f32>>(&output_flat), from::<Vec<f32>>(&expected_flat));

        Ok(())
    }

    #[test]
    fn test_apply_tho_tensor_with_moving_average_filter() -> Result<(), ThoError> {
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
        let freq_filter = FreqFilter::<RunCtxWithPermits>::new(&[InputFilter::AverageFilter(AverageFilterParams { window: 4 })], &ws, 0)?;
        let input_t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0]).f_view([2, 5])?;
        let input = ThoTensor::new(SynKey::new([mono_key!("moving_average", "moving_average")]), input_t)?;

        let output = freq_filter.apply_tho_tensor(&input)?;
        let expected = moving_average(input.as_ref(), 4)?;

        assert!(output.as_ref().allclose(&expected, 1e-5, 1e-5, false));

        Ok(())
    }


    async fn run_freq_filter(filter_configs: Vec<InputFilter>) -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 20, 100, 10000)));

        let bf = FreqFilter::new(&filter_configs, &ws, 0)?;

        let mut ctx = RunCtxWithPermits::new(1, "test");

        let size = [20, 12800];
        let r = Arc::new(bf).run(
            Arc::new(DevicesWrap::wrap(ThoTensor::new(SynKey::new([mono_key!("test", "test")]), Tensor::rand(&size, FLOAT_CPU))?, CacheArea::default())),
            ctx.branch(Some("freq_filter_test"), false, false)
        ).await?;

        let find_motifs_run_config = FindMotifsRunConfig::new_with_running_mean(None, 10, 100, Some(filter_configs), None);

        let adjuster = find_motifs_run_config.input_len_adjuster();


        with_data!(ctx.branch(Some("freq_filter_test"), false, false), r; |ctx, r|{
            debug!("filter r={r:?}");
            Ok::<_, ThoError>(())
        })?;


        // assert_eq!(adjuster(size[1] as u32), r.size().dim_1() as u32);
        //TODO asserts

        Ok(())
    }
}
