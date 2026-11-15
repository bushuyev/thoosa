//! Plot loading and output independent of CLI argument dispatch.
use thoosa_common::runner::common::cachable::Cached;
use thoosa_shared::traits::AsyncStage;
use std::io::Write;
use tch::IndexOp;
use thoosa_shared::run_configs::RunConfig;
use crate::dims::MotifSlices;
use crate::motifs_store::MotifsStore;
use crate::stages::find_motifs_run_config::{FindMotifsRunConfig};
use crate::stages::tensor_loader::RawTensorLoader;
use anyhow::Context as _;
use log::{debug, info, warn};
use parking_lot::RwLock;
use plotters::prelude::BLACK;
use plotters::style::ShapeStyle;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use tch::Device::Cpu;
use tch::{Device, Tensor};
use thoosa_common::dims::ChannelsData;
use thoosa_common::runner::run_ctx::RunCtxImpl;
use crate::fragments::{DataBy, PlotDataArgs, Resolution};
use crate::motifs_ws::parse_motif_path;
use crate::stages::apply_motif_nets::apply_motif_net_forward;
use crate::stages::load_motif_net::MotifNetMeta;
use crate::stages::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta};
use crate::stages::motif_stats::{MotifStats, MotifStatsInput};
use crate::stages::slices_stats::Metrics;
use thoosa_common::data_with_extra::DataWithExtra;
use thoosa_common::runner::common::cachable::{BorrowRef, DevicesWrap};
use thoosa_common::stages::freq_filter::{FreqFilter};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::utils::{load_run_config, plot_series, plot_series_with_sidebar_values};
use thoosa_common::transforms::filters::InputFilter;
use thoosa_shared::colors_iter::DistinctColorIter;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::resource::{ResourceId};
use thoosa_shared::run_configs::{RunConfigId};
use thoosa_shared::tracked_await::t;
use thoosa_shared::traits::{Adjustable, MaxParallelCtx, RunCtx};
use thoosa_shared::{FuturesOrdering, collect_futures_best_effort_with_progress, mono_key};
use thoosa_shared::{ThoError};

pub async fn run_plot<C>(ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, store: &Arc<dyn MotifsStore>, loader: &Arc<RawTensorLoader<C>>, plot_params: &PlotDataArgs, resolution: Resolution, _device: &Device, mut ctx: C) -> anyhow::Result<()>
where
    C: RunCtx + MaxParallelCtx + Adjustable,
{
    let print_to_stdout = plot_params.print_to_stdout();

    let motif_loader_impl = MotifLoader::<RawTensorLoader<C>, C>::new(Arc::clone(store), Arc::clone(loader), ws).await?;

    let clc_dist = Arc::new(MotifStats::new(Arc::clone(store), Arc::clone(loader), Arc::clone(ws), Arc::clone(&motif_loader_impl)));

    match plot_params.data_by() {
        DataBy::Resource { resource_id, channels, freq_filters } => {
            debug!("run_plot: MotifsBy::Resource resource_id={resource_id} channels={channels:?} freq_filters={freq_filters:?}");
            t!(plot_resource(ws, store, loader, resource_id, channels, freq_filters, resolution, print_to_stdout, ctx, "")).await?;
            return Ok(());
        }
        DataBy::MotifNetApplied { resource_id, net_path } => {
            debug!("run_plot: MotifsBy::MotifNetApplied resource_id={resource_id} net_path={net_path:?}");
            t!(plot_motif_net_applied(ws, store, loader, resource_id, net_path, resolution, print_to_stdout, ctx)).await?;
            return Ok(());
        }
        DataBy::MotifId(motif_ids) => {
            debug!("run_plot: MotifsBy::MotifId motif_ids={motif_ids:?}");
            for motif_id in motif_ids {
                t!(plot_motif(&ws, ctx.branch(true), motif_id, &motif_loader_impl, Arc::clone(&clc_dist), None, resolution, print_to_stdout)).await?;
            }
        }
        DataBy::RunId {
            find_motifs_run_config_ids,
            select_motifs_run_config_ids,
        } => {
            debug!("run_plot: MotifsBy::RunId find_run_config_ids={find_motifs_run_config_ids:?} select_run_config_ids={select_motifs_run_config_ids:?}");
            for find_motifs_run_config_id in find_motifs_run_config_ids {
                if let Err(e) = t!(plot_found_motifs(ws, store, find_motifs_run_config_id, &motif_loader_impl, &clc_dist, select_motifs_run_config_ids.clone(), resolution, print_to_stdout, ctx.branch(true),)).await {
                    warn!("run_plot: failed with e={e:} on find_motifs_run_config_id={find_motifs_run_config_id}");
                }
            }
        }

        DataBy::SelectedMotifs { select_motifs_run_config_ids } => {
            for select_motifs_run_config_id in select_motifs_run_config_ids {
                if let Err(e) = t!(plot_selected_motifs(ws, store, select_motifs_run_config_id, &motif_loader_impl, &clc_dist, resolution, print_to_stdout, ctx.branch(true),)).await {
                    warn!("run_plot: failed with e={e:} on select_motifs_run_config_id={select_motifs_run_config_id}");
                }
            }
        }
    }

    Ok(())
}

#[derive(Serialize)]
#[serde(untagged)]
enum StdoutPlotNumber {
    Finite(f32),
    Special(String),
}

impl From<f32> for StdoutPlotNumber {
    fn from(value: f32) -> Self {
        if value.is_finite() { Self::Finite(value) } else { Self::Special(value.to_string()) }
    }
}

#[derive(Serialize)]
struct StdoutPlotPoint {
    x: StdoutPlotNumber,
    y: StdoutPlotNumber,
}

#[derive(Serialize)]
struct StdoutPlotSeries {
    name: String,
    points: Vec<StdoutPlotPoint>,
}

#[derive(Serialize)]
struct StdoutPlotSidebarRow {
    name: String,
    value: StdoutPlotNumber,
}

#[derive(Serialize)]
struct StdoutPlotSidebar {
    title: String,
    rows: Vec<StdoutPlotSidebarRow>,
}

#[derive(Serialize)]
struct StdoutPlotDump {
    format: &'static str,
    caption: String,
    target_path: String,
    size: (u32, u32),
    series: Vec<StdoutPlotSeries>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sidebar: Option<StdoutPlotSidebar>,
}

fn print_plot_data_to_stdout<S>(plot_data: Vec<(String, S, Vec<(f32, f32)>)>, caption: &str, size: (u32, u32), path: &impl AsRef<Path>, sidebar: Option<(&str, &[(String, f32)])>) -> anyhow::Result<()> {
    let dump = StdoutPlotDump {
        format: "thoosa_plot_data_v1",
        caption: caption.to_string(),
        target_path: path.as_ref().display().to_string(),
        size,
        series: plot_data
            .into_iter()
            .map(|(name, _, points)| StdoutPlotSeries {
                name,
                points: points.into_iter().map(|(x, y)| StdoutPlotPoint { x: x.into(), y: y.into() }).collect(),
            })
            .collect(),
        sidebar: sidebar.map(|(title, rows)| StdoutPlotSidebar {
            title: title.to_string(),
            rows: rows.iter().map(|(name, value)| StdoutPlotSidebarRow { name: name.clone(), value: (*value).into() }).collect(),
        }),
    };

    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &dump).with_context(|| "print plot data to stdout")?;
    writeln!(&mut stdout).with_context(|| "finish plot data stdout line")?;
    Ok(())
}

fn emit_or_plot_series<S>(print_to_stdout: bool, plot_data: Vec<(String, S, Vec<(f32, f32)>)>, caption: &str, size: (u32, u32), path: &impl AsRef<Path>) -> anyhow::Result<()>
where
    S: Into<ShapeStyle> + Clone,
{
    if print_to_stdout { print_plot_data_to_stdout(plot_data, caption, size, path, None) } else { plot_series(plot_data, caption, size, path) }
}

fn emit_or_plot_series_with_sidebar_values<S>(print_to_stdout: bool, plot_data: Vec<(String, S, Vec<(f32, f32)>)>, caption: &str, size: (u32, u32), path: &impl AsRef<Path>, sidebar_title: &str, sidebar_rows: &[(String, f32)]) -> anyhow::Result<()>
where
    S: Into<ShapeStyle> + Clone,
{
    if print_to_stdout {
        print_plot_data_to_stdout(plot_data, caption, size, path, Some((sidebar_title, sidebar_rows)))
    } else {
        plot_series_with_sidebar_values(plot_data, caption, size, path, sidebar_title, sidebar_rows)
    }
}

async fn plot_found_motifs<C>(
    ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
    store: &Arc<dyn MotifsStore>,
    find_motifs_run_config_id: RunConfigId,
    motif_loader_impl: &Arc<MotifLoader<RawTensorLoader<C>, C>>,
    clc_dist: &Arc<MotifStats<MotifLoader<RawTensorLoader<C>, C>, RawTensorLoader<C>, C>>,
    select_motifs_run_config_ids: Option<Vec<RunConfigId>>,
    resolution: Resolution,
    print_to_stdout: bool,
    mut ctx: C,
) -> anyhow::Result<()>
where
    C: RunCtx + MaxParallelCtx + Adjustable,
{
    let motifs = t!(store.list_motifs(find_motifs_run_config_id)).await?;
    let motifs_len = motifs.len();
    let done_cnt = Arc::new(AtomicU32::new(0));
    let ws = Arc::clone(ws);

    let futures = motifs.into_iter().map(|motif_id| {
        let done_cnt = Arc::clone(&done_cnt);
        let ws = Arc::clone(&ws);
        let clc_dist = Arc::clone(clc_dist);
        let motif_loader_impl = Arc::clone(motif_loader_impl);
        let ctx = ctx.branch(true);
        let select_motifs_run_config_ids = select_motifs_run_config_ids.clone();
        async move {
            debug!("run_plot: motif_id={motif_id}");

            let result = t!(plot_motif(&ws, ctx, motif_id as usize, &motif_loader_impl, clc_dist, select_motifs_run_config_ids, resolution, print_to_stdout)).await;
            t!(MotifLoader::<RawTensorLoader<RunCtxImpl>, RunCtxImpl>::cache()).await?.clear();

            let done = done_cnt.fetch_add(1, Relaxed);
            if done % 32 == 0 {
                info!("done {done} from  {motifs_len}");
            }

            result.map_err(|error| ThoError::from(error).context(format!("plot motif {motif_id}")))
        }
    });

    // Plot exports are independent; deliberately keep successful plots if another fails.
    let _ = t!(collect_futures_best_effort_with_progress(futures, motifs_len, FuturesOrdering::Unordered, &format!("plot_found_motifs_for_run_config_id_{find_motifs_run_config_id}"),)).await;

    Ok(())
}

async fn plot_selected_motifs<C>(
    ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
    store: &Arc<dyn MotifsStore>,
    select_motifs_run_config_id: RunConfigId,
    motif_loader_impl: &Arc<MotifLoader<RawTensorLoader<C>, C>>,
    clc_dist: &Arc<MotifStats<MotifLoader<RawTensorLoader<C>, C>, RawTensorLoader<C>, C>>,
    resolution: Resolution,
    print_to_stdout: bool,
    mut ctx: C,
) -> anyhow::Result<()>
where
    C: RunCtx + MaxParallelCtx + Adjustable,
{
    let channels = ws.read().channels();
    debug!("plot_selected_motifs: select_motifs_run_config_id={select_motifs_run_config_id} channels={channels}");

    let (_run_config_id, motifs) = t!(store.list_selected_motifs(&[select_motifs_run_config_id], channels, None)).await?.into_iter().next().context("no selected motif")?;

    debug!("plot_selected_motifs: motifs={motifs:?}");

    let motif_ids: Vec<usize> = motifs.into_iter().flat_map(|channel| channel.into_iter().map(|(motif_id, _)| motif_id as usize)).collect();

    if motif_ids.is_empty() {
        warn!("plot_selected_motifs: no motifs found for select_motifs_run_config_id={select_motifs_run_config_id}");
        return Ok(());
    }

    let motifs_len = motif_ids.len();
    let done_cnt = Arc::new(AtomicU32::new(0));
    let suffix = Some(vec![select_motifs_run_config_id]);
    let futures = motif_ids.into_iter().map(|motif_id| {
        let done_cnt = Arc::clone(&done_cnt);
        let clc_dist = Arc::clone(clc_dist);
        let motif_loader_impl = Arc::clone(motif_loader_impl);
        let ws = Arc::clone(ws);
        let suffix = suffix.clone();
        let ctx = ctx.branch(true);
        async move {
            debug!("plot_selected_motifs: motif_id={motif_id}");
            let result = t!(plot_motif(&ws, ctx, motif_id, &motif_loader_impl, clc_dist, suffix, resolution, print_to_stdout)).await;
            //MotifLoader::<RawTensorLoader<RunCtxImpl>, RunCtxImpl>::cache().await.clear();

            let done = done_cnt.fetch_add(1, Relaxed);
            if done % 32 == 0 {
                info!("done {done} from  {motifs_len}");
            }

            result.map_err(|error| ThoError::from(error).context(format!("plot motif {motif_id}")))
        }
    });

    // Plot exports are independent; deliberately keep successful plots if another fails.
    let _ = t!(collect_futures_best_effort_with_progress(futures, motifs_len, FuturesOrdering::Unordered, &format!("plot_selected_motifs_for_run_config_id_{select_motifs_run_config_id}"),)).await;

    Ok(())
}

async fn load_resource_tensor_for_plot<C>(store: &Arc<dyn MotifsStore>, loader: &Arc<RawTensorLoader<C>>, resource_id: ResourceId, mut ctx: C) -> anyhow::Result<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>
where
    C: RunCtx + MaxParallelCtx + Adjustable,
{
    let resource = t!(store.list_resources_by_ids(&vec![resource_id])).await?.into_iter().next().ok_or_else(|| anyhow::anyhow!("resource_id={resource_id} not found"))?;

    Ok(t!(Arc::clone(loader).run_cached(Arc::new(resource), ctx.branch(false),)).await?)
}

fn sanitize_plot_path_segment(v: &str) -> String {
    let s = v
        .chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' | '.' => c,
            _ => '_',
        })
        .collect::<String>();
    if s.is_empty() { "plot".to_string() } else { s }
}

fn plot_resource_channels<C>(ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, resource_id: ResourceId, prefix: &str, suffix: &str, caption: &str, tensor: &ThoTensor<ChannelsData, SynKey>, channels: Option<&Vec<u32>>, resolution: Resolution, print_to_stdout: bool) -> anyhow::Result<()>
where
    C: RunCtx + MaxParallelCtx + Adjustable,
{
    let total_channels = tensor.size()[0] as u32;
    let channels_to_plot = channels.cloned().unwrap_or_else(|| (0..total_channels).collect());
    if channels_to_plot.is_empty() {
        return Err(anyhow::anyhow!("plot_resource_channels: no channels requested for resource_id={resource_id}"));
    }
    for channel in &channels_to_plot {
        if *channel >= total_channels {
            return Err(anyhow::anyhow!("plot_resource_channels: channel {channel} out of range [0,{total_channels}) for resource_id={resource_id}"));
        }
    }

    let mut color_iter = DistinctColorIter::new();
    let plot_data = channels_to_plot
        .into_iter()
        .map(|channel| {
            let channel_t = tensor.local()?.f_i((channel as i64, ..))?;
            Ok::<_, anyhow::Error>((format!("channel_{channel}"), color_iter.next().unwrap_or(BLACK), channel_t.iter::<f64>()?.enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>()))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    emit_or_plot_series(print_to_stdout, plot_data, caption, resolution.plot_size((2000, 1000)), &ws.read().plots_path().join("resource").join(format!("{prefix}resource_{resource_id}_{suffix}.png")))?;

    Ok(())
}

fn plot_single_line_tensor(ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, path_suffix: &str, caption: &str, line_t: &Tensor, resolution: Resolution, print_to_stdout: bool) -> anyhow::Result<()> {
    let line_t = line_t.f_view([-1])?;
    let points = line_t.iter::<f64>()?.enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<_>>();

    emit_or_plot_series(print_to_stdout, vec![("0".to_string(), BLACK, points)], caption, resolution.plot_size((2000, 1000)), &ws.read().plots_path().join("resource").join(format!("{path_suffix}.png")))?;

    Ok(())
}

pub async fn plot_resource<C>(
    ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
    store: &Arc<dyn MotifsStore>,
    loader: &Arc<RawTensorLoader<C>>,
    resource_id: ResourceId,
    channels: Option<Vec<u32>>,
    freq_filters: Option<Vec<InputFilter>>,
    resolution: Resolution,
    print_to_stdout: bool,
    mut ctx: C,
    prefix: &str,
) -> anyhow::Result<()>
where
    C: RunCtx + MaxParallelCtx + Adjustable,
{
    let raw = t!(load_resource_tensor_for_plot(store, loader, resource_id, ctx.branch(false))).await?;

    let raw_channels = channels.clone();
    let channels_frag = sanitize_plot_path_segment(&format!("{channels:?}",));

    let tensor = BorrowRef::borrow_ref(&raw, 0).await?;

    plot_resource_channels::<C>(
        ws,
        resource_id,
        prefix,
        &format!("raw_{channels_frag}",),
        &format!("Resource {resource_id} raw channels: {raw_channels:?}"),
        tensor.as_ref(),
        raw_channels.as_ref(),
        resolution,
        print_to_stdout,
    )?;

    if let Some(freq_filters) = freq_filters {
        let filter_frag = sanitize_plot_path_segment(&freq_filters.iter().map(|f| format!("{:?}", f)).collect::<String>()); //TODO remove "*Params"
        let configs = ws.read().ws_configs();
        let filtered = t!(FreqFilter::<C>::new(freq_filters.as_slice(), configs, resource_id, ctx.tho_device_idx()).await?.run(Arc::clone(&raw), ctx.branch(false),)).await?;

        let filtered_channels = channels.clone();
        let tensor = BorrowRef::borrow_ref(&filtered, 0).await?;

        plot_resource_channels::<C>(
            ws,
            resource_id,
            prefix,
            &format!("freq_filtered_{channels_frag}_{filter_frag}"),
            &format!("Resource {resource_id} channels: {filtered_channels:?} filters {freq_filters:?}"),
            tensor.as_ref(),
            filtered_channels.as_ref(),
            resolution,
            print_to_stdout,
        )?;
    }

    Ok(())
}

async fn plot_motif_net_applied<C>(ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, store: &Arc<dyn MotifsStore>, loader: &Arc<RawTensorLoader<C>>, resource_id: ResourceId, net_path: PathBuf, resolution: Resolution, print_to_stdout: bool, mut ctx: C) -> anyhow::Result<()>
where
    C: RunCtx + MaxParallelCtx + Adjustable,
{
    let raw = t!(load_resource_tensor_for_plot(store, loader, resource_id, ctx.branch(false))).await?;

    // raw.with_data_ref(0, ctx.branch(false), None, |tensor: &ThoTensor<ChannelsData2, SynKey>, _ctx| {
    //     plot_resource_channels::<C>(ws, resource_id, "raw_all_channels", &format!("Resource {resource_id} raw (all channels)"), tensor, None)
    // }).await?;

    let (path_channel, motif_id) = parse_motif_path(&net_path)?;
    let (store_channel, find_motifs_run_config_id) = t!(store.get_motif_meta(motif_id)).await?;

    let find_motifs_run_config: FindMotifsRunConfig = t!(load_run_config(store, find_motifs_run_config_id, FindMotifsRunConfig::type_name())).await?;

    if path_channel as usize != store_channel {
        warn!("plot_motif_net_applied_variant: channel mismatch path_channel={path_channel} store_channel={store_channel} motif_id={motif_id} net_path={net_path:?}");
    }
    let channel = store_channel as u32;
    let channel_only = vec![channel];
    let net_path_sfx = sanitize_plot_path_segment(&net_path.to_string_lossy());

    let tensor = BorrowRef::borrow_ref(&raw, 0).await?;

    plot_resource_channels::<C>(
        ws,
        resource_id,
        "",
        &format!("motif_net_input_raw_ch_{channel}"),
        &format!("Resource {resource_id} motif-net input raw channel={channel}"),
        tensor.as_ref(),
        Some(&channel_only),
        resolution,
        print_to_stdout,
    )?;

    let filtered_or_raw = if let Some(freq_filters) = find_motifs_run_config.freq_filter.clone() {
        let configs = ws.read().ws_configs();
        let filtered = t!(FreqFilter::<C>::new(freq_filters.as_slice(), configs, find_motifs_run_config_id, ctx.tho_device_idx()).await?.run(Arc::clone(&raw), ctx.branch(false),)).await?;

        let channel_only_filtered = vec![channel];

        {
            let tensor = BorrowRef::borrow_ref(&filtered, 0).await?;

            plot_resource_channels::<C>(
                ws,
                resource_id,
                "",
                &format!("motif_net_input_prefiltered_ch_{channel}_{net_path_sfx}"),
                &format!("Resource {resource_id} motif-net prefiltered channel={channel} find_run_config_id={find_motifs_run_config_id}"),
                tensor.as_ref(),
                Some(&channel_only_filtered),
                resolution,
                print_to_stdout,
            )?;
        }

        filtered
    } else {
        Arc::clone(&raw)
    };

    let (motif_net_params, motif_net) = {
        let ws_guard = ws.read();
        ws_guard.load_motif_net(&net_path, &Cpu, false, true)?
    };
    let motif_slices_step = 5u32; //motif_net_params.stride() as u32;
    let motif_net = Arc::new(DataWithExtra::new(motif_net.expect("motif_net"), MotifNetMeta::new(motif_id, channel, motif_net_params)));

    let tensor = BorrowRef::borrow_ref(&filtered_or_raw, 0).await?;

    let motif_net = Arc::clone(&motif_net);

    let _result = (|| {
        let out = apply_motif_net_forward(mono_key!(resource_id, format!("motif_{motif_id}_channel_{channel}")) + tensor.id(), tensor.as_ref(), motif_net.as_ref(), motif_slices_step, tensor.id())?;

        plot_single_line_tensor(
            ws,
            &format!("resource_{resource_id}_motif_net_applied_{motif_id}_{channel}_{net_path_sfx}"),
            &format!("Resource {resource_id} motif-net output motif_id={motif_id} channel={channel} find_run_config_id={find_motifs_run_config_id} net={net_path:?}"),
            &*out.local()?,
            resolution,
            print_to_stdout,
        )?;

        Ok::<_, anyhow::Error>(())
    })();

    Ok(())
}

pub async fn plot_motif<C>(
    ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
    mut ctx: C,
    motif_id: usize,
    motif_loader_impl: &Arc<MotifLoader<RawTensorLoader<C>, C>>,
    motif_stats: Arc<MotifStats<MotifLoader<RawTensorLoader<C>, C>, RawTensorLoader<C>, C>>,
    select_motif_run_config_ids: Option<Vec<RunConfigId>>,
    resolution: Resolution,
    print_to_stdout: bool,
) -> anyhow::Result<()>
where
    C: RunCtx + MaxParallelCtx + Adjustable,
{
    let select_motif_run_config_ids_sfx = select_motif_run_config_ids.as_ref().map(|v| v.iter().map(|v| format!("_{}", v)).collect::<String>()).unwrap_or("".to_string());
    let motif_loader_input = MotifLoaderInput::new(((motif_id as u32), "motif_id").into(), select_motif_run_config_ids);
    let motif_id_arc = Arc::new(motif_loader_input.clone());
    let metrics = Metrics::all();
    let clc_input = Arc::new(MotifStatsInput::from_motif_input(motif_loader_input, metrics.clone()));

    let motif: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>> = t!(Arc::clone(&motif_loader_impl).run_cached(Arc::clone(&motif_id_arc), ctx.branch(false),)).await?;

    let stats = t!(motif_stats.run(clc_input, ctx.branch(false))).await?;
    let metrics_for_plot: Vec<(String, f32)> = metrics.iter().map(|metric| Ok((format!("{metric:?}"), stats.channel_value(metric, 0)?))).collect::<anyhow::Result<Vec<_>>>().unwrap_or(vec![]);

    let metrics_for_plot = Arc::new(metrics_for_plot);

    let metrics_for_plot_ref = Arc::clone(&metrics_for_plot);
    let data_ref = BorrowRef::borrow_ref(&motif, ctx.tho_device_idx()).await?;
    let motif = data_ref.as_ref();
    let run_config_id = motif.extra().find_motifs_run_config_id;
    let channel = motif.extra().channel;

    let all_slices = thoosa_shared::CopyToDevice::copy_to_device(motif.data(), 0).await?.into_local()?;

    let _result = (move || {
        // trace!("plot_motif: motif.t={}", motif.data().t_ref());
        // let all_slices = motif.data().t_ref().unbind(0);
        let all_slices = all_slices.unbind(0);

        let slices_len = all_slices.len();
        let mut color_iter = DistinctColorIter::new();

        /*
        TODO param to enable stft
        let target_dit = ws.read().plots_path().join(format!("{run_config_id}{select_motif_run_config_ids_sfx}"));

        fs::create_dir_all(&target_dit)?;
        let mut specs = vec![];
        for (i, slice) in all_slices.iter().enumerate() {
            let spec = slice.stft(
                16000,
                4000,
                16000,
                None::<Tensor>,
                false,
                true,
                true,
                true,
            );
            let spec_abs = spec.squeeze_dim(0).abs().transpose(0, 1);


            plot_spectrogram(
                &spec_abs,
                &target_dit.join(format!("spec_{i}_{}.png", motif_id)),
                1200,
                600,
            )?;
            specs.push(spec_abs);
        }
        let all_spects_t = Tensor::stack(&specs, 0).sum_dim_intlist(0, false, None);

        plot_spectrogram(
            &all_spects_t,
            &target_dit.join(format!("spec_all_{}.png", motif_id)),
            1200,
            600,
        )?;*/

        let plot_data = all_slices
            .into_iter()
            .enumerate()
            .map(|(i, t)| Ok::<_, anyhow::Error>((format!("{i}"), color_iter.next().unwrap_or(BLACK), t.iter::<f64>()?.enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>())))
            .collect::<anyhow::Result<Vec<_>>>()?;

        emit_or_plot_series_with_sidebar_values(
            print_to_stdout,
            plot_data,
            format!("Motif: {motif_id} run id: {run_config_id} channel: {channel} number of slices: {slices_len}").as_str(),
            resolution.plot_size((2600, 1200)),
            &ws.read().plots_path().join(format!("{run_config_id}{select_motif_run_config_ids_sfx}")).join(format!("motif_{}.png", motif_id)),
            "Motif Metrics",
            metrics_for_plot_ref.as_slice(),
        )?;

        Ok::<_, anyhow::Error>(true)
    })();

    Ok(())
}
