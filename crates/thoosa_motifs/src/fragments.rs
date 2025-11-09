use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::fs;
use std::hash::Hash;
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::distance::{average_euclidean_distance, clc_dist_d2};
use clap::{Args, ValueEnum};
use crossbeam_utils::sync::WaitGroup;
use futures::FutureExt;
use futures_util::future::join_all;
use log::{debug, info, trace, warn};
use parking_lot::RwLock;
use regex::Regex;
use std::panic::AssertUnwindSafe;
use tch::Device;
use tch::Device::Cpu;
use tch::IndexOp;
use tch::Kind::Float;
use tch::Tensor;
use thoosa_common::applicables::{ThoRunCtx, ThoRunner};
use thoosa_common::find_motifs_run_config::{
    FindMotifsApplicable, FindMotifsRunConfig, FindMotifsRunConfigOptions,
};
use thoosa_common::loader::{LoadRunner, Loader, SyncWrapper};
use thoosa_shared::panic_to_err;
use thoosa_common::utils::*;
use thoosa_common::ws::Workspace;
use thoosa_shared::TensorIdSliceIdx;
use thoosa_shared::ThoosaError;
use thoosa_shared::resource::{Resource, ResourceId};
use thoosa_shared::run_configs::Runnable;
use thoosa_shared::run_configs::{Cachable, RunId};
use thoosa_shared::*;

#[derive(Debug, Args)]
pub struct MotifsToFragmentsParams {
    #[arg(long, default_value = "../../data_local/motifs")]
    motifs_dir: String,

    #[arg(long, default_value = "../../data_local/fragments")]
    fragments_dir: String,

    #[arg(long, default_value_t = 1000)]
    window: i64,
}

impl MotifsToFragmentsParams {
    pub fn motifs_dir(&self) -> &str {
        &self.motifs_dir
    }

    pub fn fragments_dir(&self) -> &str {
        &self.fragments_dir
    }

    pub fn window(&self) -> i64 {
        self.window
    }
}

#[derive(Debug)]
pub struct MotifsTensors {
    pub channels_motifs_slices: Tensor,
    pub channels_motifs_avg_dists: Tensor,
    pub channels_motifs_mask: Tensor,
    pub channels_motifs_ids: Tensor,
}
//should be read-only
unsafe impl Sync for MotifsTensors {}

impl MotifsTensors {
    pub fn new(
        channels_motifs_slices: Tensor,
        channels_motifs_avg_dists: Tensor,
        channels_motifs_ids: Tensor,
    ) -> Self {
        Self {
            channels_motifs_slices,
            channels_motifs_mask: channels_motifs_avg_dists.ge(0.), //avg dists padded with f32::MIN
            channels_motifs_avg_dists,
            channels_motifs_ids,
        }
    }
}

/**
* returns tensor channels x max motifs in channels x first slice in motifs group. + channels x motif ids + dists
* motifs order is random because of iterating over hashmap entries but always consistent between tensor and vec
*/
pub async fn motifs_to_tensor_2(
    resource_id_motif_slices: HashMap<Resource, Vec<MotifSlice>>,
    runner: &Arc<ThoRunner>,
    find_motifs_run_config: &FindMotifsRunConfig,
    channels: i64,
    device: Device,
    area: &Arc<RwLock<ThoRunCtx>>,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
) -> Result<MotifsTensors, ThoosaError> {
    // debug!(
    //     "motifs_to_tensor: resource_id_motif_slices.len={} device={device:?} find_motifs_run_config={find_motifs_run_config:?}",
    //     resource_id_motif_slices.len()
    // );
    // trace!("motifs_to_tensor: resource_id_motif_slices={resource_id_motif_slices:?}");
    //
    // assert!(
    //     resource_id_motif_slices.len() > 0,
    //     "resource_id_motif_slices is empty in motifs_to_tensor"
    // );
    //
    // let channels_motifs_slices: Arc<
    //     RwLock<HashMap<u16, HashMap<usize, Vec<SyncWrapper<Tensor>>>>>,
    // > = Arc::new(RwLock::new(HashMap::new()));
    //
    // let channels_motifs_slices_arc = Arc::clone(&channels_motifs_slices);
    //
    // // let wg = WaitGroup::new();
    // // let wg_1 = wg.clone();
    //
    // let find_motifs_run_config_applicable =
    //     Arc::new(find_motifs_run_config.to_applicable(ws.read().deref(), Cpu));
    // debug!(
    //     "motifs_to_tensor: find_motifs_run_config_applicable={find_motifs_run_config_applicable:?}"
    // );
    // // runner.write().exec(move |future_id, loader, _purge_resources, done_sender| {
    // //     async move {
    // //
    // for (resource, motif_slices) in resource_id_motif_slices {
    //     debug!("motifs_to_tensor: resource={resource:?}");
    //     let t = runner
    //         .run_loader(&resource, &device, Some(area.clone()))
    //         .await;
    //     //             let t =   Loader::<Tensor>::load_future(Arc::clone(&loader), resource, area, future_id, done_sender.clone()).await;
    //     debug!("motifs_to_tensor: t.size={:?}", t.size());
    //     let applied = runner
    //         .run_find_motifs_applicable(&t, &find_motifs_run_config_applicable, Some(area.clone()))
    //         .await;
    //     //             let applied = find_motifs_run_config_applicable.apply(t.into(), FindMotifsRunConfigOptions::FilterUnfold).expect("find_motifs_run_config_for_loader.apply");
    //     debug!("motifs_to_tensor: applied t.size={:?}", applied.size());
    //     //
    //     for motif_slice in motif_slices {
    //         debug!(
    //             "motifs_to_tensor: motif_slice={motif_slice:?} applied.as_ref().tensor.size={:?}",
    //             applied.as_ref().t.size()
    //         );
    //
    //         // let motif_slice_t_x = find_motifs_run_config_applicable.unfold(&applied).tensor.i((
    //         //     motif_slice.channel as i64,
    //         //     motif_slice.slice_idx/ find_motifs_run_config_applicable.step() as i64,
    //         //     ..
    //         // ));
    //
    //         let motif_slice_t = applied.t.i((
    //             motif_slice.channel as i64,
    //             motif_slice.slice_idx
    //                 ..motif_slice.slice_idx
    //                     + find_motifs_run_config_applicable.run_config().window() as i64,
    //         ));
    //         //channel_t.narrow(1, motif_slice.slice_idx, find_motifs_run_config_for_loader.window as i64);
    //
    //         // debug!("motifs_to_tensor: motif_slice_t_x=\n{motif_slice_t_x} motif_slice_t=\n{motif_slice_t}");
    //         debug!(
    //             "motifs_to_tensor: motif_slice_t.size={:?}",
    //             motif_slice_t.size()
    //         );
    //
    //         let mut write_guard = channels_motifs_slices_arc.write();
    //         let channels_motifs = write_guard
    //             .entry(motif_slice.channel)
    //             .or_insert_with(|| HashMap::new());
    //         let motif_slices = channels_motifs
    //             .entry(motif_slice.motif_id as usize)
    //             .or_insert_with(|| Vec::new());
    //
    //         motif_slices.push(SyncWrapper::wrap_tensor(motif_slice_t));
    //     }
    // }
    // //
    // //         drop(wg_1);
    // //
    // //     }.boxed()
    // // });
    //
    // // wg.wait();
    //
    // debug!(
    //     "motifs_to_tensor: channels_motifs_slices.len={:?}",
    //     channels_motifs_slices.read().len()
    // );
    // trace!(
    //     "motifs_to_tensor: channels_motifs_slices={:?}",
    //     channels_motifs_slices
    // );
    //
    // let max_motifs_num = channels_motifs_slices
    //     .read()
    //     .values()
    //     .map(|motif_slices| motif_slices.len())
    //     .max()
    //     .expect("No motifs") as i64;
    //
    // let mut channel_motif_ids: Vec<Vec<usize>> = (0..channels)
    //     .into_iter()
    //     .map(|_| Vec::with_capacity(max_motifs_num as usize))
    //     .collect();
    // let zeros_t = Tensor::zeros(find_motifs_run_config.window as i64, (Float, device));
    //
    // let mut channels_motifs_slices_t = Vec::with_capacity(channels as usize);
    // let mut channels_motifs_avg_dists_t = Vec::with_capacity(channels as usize);
    //
    // for channel in 0..channels {
    //     let motif_slices = channels_motifs_slices
    //         .write()
    //         .remove(&(channel as u16))
    //         .unwrap_or(HashMap::new());
    //     let mut motifs_avg_dists = Vec::with_capacity(motif_slices.len());
    //
    //     let mut motifs = motif_slices.into_iter().map(|(motif_id, slices)| {
    //         let slices_len = slices.len();
    //
    //         if slices.len() > 10_000 {//TODO config
    //             Err(ThoosaError::new(format!("skipped motif {motif_id} with too many slices ({})", slices.len())))
    //         } else {
    //             let slices_t = Tensor::vstack(slices.into_iter().map(|s| s.shallow_clone()).collect::<Vec<Tensor>>().as_slice());
    //             debug!("motifs_to_tensor: motif_id={motif_id} slices_t={:?}", slices_t.size());
    //             trace!("motifs_to_tensor: slices_t=\n{slices_t}");
    //
    //             match panic_to_err!(average_euclidean_distance(&slices_t)) {
    //                 Ok(motif_avg_dist) => {
    //                     trace!("motifs_to_tensor: motif_avg_dist={motif_avg_dist}");
    //                     motifs_avg_dists.push(motif_avg_dist);
    //                     channel_motif_ids[channel as usize].push(motif_id);
    //
    //                     Ok(slices_t.mean_dim(0, false, Float).copy())
    //                 },
    //                 Err(e) => {
    //                     Err(ThoosaError::new(format!("can't load motif {motif_id} because of oom error, num slices={slices_len}")))
    //                 }
    //             }
    //         }
    //
    //     }).filter(|v|v.is_ok()).map(|v|v.unwrap()).collect::<Vec<Tensor>>();
    //
    //     //make all motifs same length
    //     if (motifs.len() as i64) < max_motifs_num {
    //         debug!(
    //             "motifs_to_tensor: padding channel={channel} {} to {max_motifs_num} ",
    //             motifs.len()
    //         );
    //
    //         motifs
    //             .extend((0..max_motifs_num - motifs.len() as i64).map(|_| zeros_t.shallow_clone()));
    //         motifs_avg_dists.extend(vec![
    //             f32::MIN;
    //             max_motifs_num as usize - motifs_avg_dists.len()
    //         ]);
    //
    //         let channel_motif_ids_len = channel_motif_ids[channel as usize].len();
    //         channel_motif_ids[channel as usize].extend(vec![
    //             0;
    //             max_motifs_num as usize
    //                 - channel_motif_ids_len
    //         ]);
    //     }
    //     channels_motifs_avg_dists_t.push(Tensor::from_slice(motifs_avg_dists.as_slice()));
    //     channels_motifs_slices_t.push(
    //         Tensor::vstack(motifs.as_slice())
    //             .view((max_motifs_num, find_motifs_run_config.window as i64)),
    //     )
    // }
    //
    // debug!("motifs_to_tensor: channels_motifs_avg_dists_t={channels_motifs_avg_dists_t:?}");
    // debug!("motifs_to_tensor: channel_motif_ids={channel_motif_ids:?}");
    //
    // Ok(MotifsTensors::new(
    //     Tensor::vstack(channels_motifs_slices_t.as_slice()).view((channels, max_motifs_num, -1)),
    //     Tensor::vstack(channels_motifs_avg_dists_t.as_slice()).view((channels, max_motifs_num))
    //         + 1e-2, //+ 1e-2 if motif avg is 0 (in tests) there will be no matches with tensors where dists close to 0 but not equal
    //     Tensor::from_slice(
    //         channel_motif_ids
    //             .into_iter()
    //             .flat_map(|channel_motif_ids| channel_motif_ids.into_iter())
    //             .map(|v| v as i32)
    //             .collect::<Vec<_>>()
    //             .as_slice(),
    //     )
    //     .view((channels, -1)),
    // ))
    todo!()
}

/**
* returns tensor channels x max motifs in channels x first slice in motifs group. + channels x motif ids + dists
* motifs order is random because of iterating over hashmap entries but always consistent between tensor and vec
*/
pub fn motifs_to_tensor(
    resource_id_motif_slices: HashMap<Resource, Vec<MotifSlice>>,
    loader: &Arc<RwLock<LoadRunner<Tensor>>>,
    find_motifs_run_config: &FindMotifsRunConfig,
    channels: i64,
    device: Device,
    area: u64,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
) -> Result<MotifsTensors, ThoosaError> {
    // debug!(
    //     "motifs_to_tensor: resource_id_motif_slices.len={} device={device:?} find_motifs_run_config={find_motifs_run_config:?}",
    //     resource_id_motif_slices.len()
    // );
    // trace!("motifs_to_tensor: resource_id_motif_slices={resource_id_motif_slices:?}");
    //
    // assert!(
    //     resource_id_motif_slices.len() > 0,
    //     "resource_id_motif_slices is empty in motifs_to_tensor"
    // );
    //
    // let channels_motifs_slices: Arc<
    //     RwLock<HashMap<u16, HashMap<usize, Vec<SyncWrapper<Tensor>>>>>,
    // > = Arc::new(RwLock::new(HashMap::new()));
    //
    // let channels_motifs_slices_arc = Arc::clone(&channels_motifs_slices);
    //
    // let wg = WaitGroup::new();
    // let wg_1 = wg.clone();
    //
    // let find_motifs_run_config_applicable =
    //     find_motifs_run_config.to_applicable(ws.read().deref(), Cpu);
    //
    // loader
    //     .write()
    //     .exec(move |future_id, loader, _purge_resources, done_sender| {
    //         async move {
    //             for (resource, motif_slices) in resource_id_motif_slices {
    //                 debug!("motifs_to_tensor: resource={resource:?}");
    //                 let t = Loader::<Tensor>::load_future(
    //                     Arc::clone(&loader),
    //                     resource,
    //                     area,
    //                     future_id,
    //                     done_sender.clone(),
    //                 )
    //                 .await;
    //                 debug!("motifs_to_tensor: t.size={:?}", t.size());
    //                 let applied = find_motifs_run_config_applicable
    //                     .run(t.into(), FindMotifsRunConfigOptions::FilterUnfold)
    //                     .expect("find_motifs_run_config_for_loader.apply");
    //                 debug!("motifs_to_tensor: applied t.size={:?}", applied.size());
    //
    //                 for motif_slice in motif_slices {
    //                     debug!("motifs_to_tensor: motif_slice={motif_slice:?}");
    //
    //                     let motif_slice_t = applied.as_ref().i((
    //                         motif_slice.channel as i64,
    //                         motif_slice.slice_idx / find_motifs_run_config_applicable.step() as i64,
    //                         ..,
    //                     )); //channel_t.narrow(1, motif_slice.slice_idx, find_motifs_run_config_for_loader.window as i64);
    //
    //                     debug!(
    //                         "motifs_to_tensor: motif_slice_t.size={:?}",
    //                         motif_slice_t.size()
    //                     );
    //
    //                     let mut write_guard = channels_motifs_slices_arc.write();
    //                     let channels_motifs = write_guard
    //                         .entry(motif_slice.channel)
    //                         .or_insert_with(|| HashMap::new());
    //                     let motif_slices = channels_motifs
    //                         .entry(motif_slice.motif_id as usize)
    //                         .or_insert_with(|| Vec::new());
    //
    //                     motif_slices.push(SyncWrapper::wrap_tensor(motif_slice_t));
    //                 }
    //             }
    //
    //             drop(wg_1);
    //         }
    //         .boxed()
    //     });
    //
    // wg.wait();
    //
    // debug!(
    //     "motifs_to_tensor: channels_motifs_slices.len={:?}",
    //     channels_motifs_slices.read().len()
    // );
    // trace!(
    //     "motifs_to_tensor: channels_motifs_slices={:?}",
    //     channels_motifs_slices
    // );
    //
    // let max_motifs_num = channels_motifs_slices
    //     .read()
    //     .values()
    //     .map(|motif_slices| motif_slices.len())
    //     .max()
    //     .expect("No motifs") as i64;
    //
    // let mut channel_motif_ids: Vec<Vec<usize>> = (0..channels)
    //     .into_iter()
    //     .map(|_| Vec::with_capacity(max_motifs_num as usize))
    //     .collect();
    // let zeros_t = Tensor::zeros(find_motifs_run_config.window as i64, (Float, device));
    //
    // let mut channels_motifs_slices_t = Vec::with_capacity(channels as usize);
    // let mut channels_motifs_avg_dists_t = Vec::with_capacity(channels as usize);
    //
    // for channel in 0..channels {
    //     let motif_slices = channels_motifs_slices
    //         .write()
    //         .remove(&(channel as u16))
    //         .unwrap_or(HashMap::new());
    //     let mut motifs_avg_dists = Vec::with_capacity(motif_slices.len());
    //
    //     let mut motifs = motif_slices.into_iter().map(|(motif_id, slices)| {
    //         let slices_len = slices.len();
    //
    //         if slices.len() > 10_000 {//TODO config
    //             Err(ThoosaError::new(format!("skipped motif {motif_id} with too many slices ({})", slices.len())))
    //         } else {
    //             let slices_t = Tensor::vstack(slices.into_iter().map(|s| s.shallow_clone()).collect::<Vec<Tensor>>().as_slice());
    //             debug!("motifs_to_tensor: motif_id={motif_id} slices_t={:?}", slices_t.size());
    //             trace!("motifs_to_tensor: slices_t=\n{slices_t}");
    //
    //             match panic_to_err!(average_euclidean_distance(&slices_t)) {
    //                 Ok(motif_avg_dist) => {
    //                     trace!("motifs_to_tensor: motif_avg_dist={motif_avg_dist}");
    //                     motifs_avg_dists.push(motif_avg_dist);
    //                     channel_motif_ids[channel as usize].push(motif_id);
    //
    //                     Ok(slices_t.mean_dim(0, false, Float).copy())
    //                 },
    //                 Err(e) => {
    //                     Err(ThoosaError::new(format!("can't load motif {motif_id} because of oom error, num slices={slices_len}")))
    //                 }
    //             }
    //         }
    //
    //     }).filter(|v|v.is_ok()).map(|v|v.unwrap()).collect::<Vec<Tensor>>();
    //
    //     //make all motifs same length
    //     if (motifs.len() as i64) < max_motifs_num {
    //         debug!(
    //             "motifs_to_tensor: padding channel={channel} {} to {max_motifs_num} ",
    //             motifs.len()
    //         );
    //
    //         motifs
    //             .extend((0..max_motifs_num - motifs.len() as i64).map(|_| zeros_t.shallow_clone()));
    //         motifs_avg_dists.extend(vec![
    //             f32::MIN;
    //             max_motifs_num as usize - motifs_avg_dists.len()
    //         ]);
    //
    //         let channel_motif_ids_len = channel_motif_ids[channel as usize].len();
    //         channel_motif_ids[channel as usize].extend(vec![
    //             0;
    //             max_motifs_num as usize
    //                 - channel_motif_ids_len
    //         ]);
    //     }
    //     channels_motifs_avg_dists_t.push(Tensor::from_slice(motifs_avg_dists.as_slice()));
    //     channels_motifs_slices_t.push(
    //         Tensor::vstack(motifs.as_slice())
    //             .view((max_motifs_num, find_motifs_run_config.window as i64)),
    //     )
    // }
    //
    // debug!("motifs_to_tensor: channels_motifs_avg_dists_t={channels_motifs_avg_dists_t:?}");
    // debug!("motifs_to_tensor: channel_motif_ids={channel_motif_ids:?}");
    //
    // Ok(MotifsTensors::new(
    //     Tensor::vstack(channels_motifs_slices_t.as_slice()).view((channels, max_motifs_num, -1)),
    //     Tensor::vstack(channels_motifs_avg_dists_t.as_slice()).view((channels, max_motifs_num))
    //         + 1e-2, //+ 1e-2 if motif avg is 0 (in tests) there will be no matches with tensors where dists close to 0 but not equal
    //     Tensor::from_slice(
    //         channel_motif_ids
    //             .into_iter()
    //             .flat_map(|channel_motif_ids| channel_motif_ids.into_iter())
    //             .map(|v| v as i32)
    //             .collect::<Vec<_>>()
    //             .as_slice(),
    //     )
    //     .view((channels, -1)),
    // ))
    todo!()
}

pub fn write_motifs_slices<F>(
    base_path: &str,
    output_dir: &str,
    eeg_id_fragments_map: HashMap<usize, Vec<MotifSlice>>,
    eegs: Vec<String>,
    read_eeg_to_tensor: F,
    window: i64,
) -> Result<(), ThoosaError>
where
    F: Fn(&str, &str, Device) -> Result<Tensor, ThoosaError> + Sync,
{
    debug!("write_motifs_slices");
    let output_dir = Path::new(output_dir);
    fs::create_dir_all(output_dir)?;

    let total: usize = eeg_id_fragments_map
        .values()
        .map(|fragments| fragments.len())
        .sum();
    debug!("write_motifs_slices: total={}", total);

    let progress = AtomicU64::new(0);

    for (eeg_idx, channels_fragments) in eeg_id_fragments_map {
        let eeg_id = eegs[eeg_idx].clone();
        let eeg_t = (read_eeg_to_tensor)(base_path, eeg_id.as_str(), Device::Cpu)?;

        for fragment in channels_fragments {
            let channel_path = output_dir.join(fragment.channel.to_string());
            let motif_path = channel_path.join(fragment.motif_id.to_string());
            fs::create_dir_all(motif_path.clone())?;

            let channel_t = eeg_t.get(fragment.channel as i64);
            let channel_t = channel_t.view([1, -1]);

            let fragment_t = channel_t.f_slice(
                1,
                Some(fragment.slice_idx),
                Some(fragment.slice_idx + window),
                1,
            )?;

            fragment_t.save(motif_path.join(format!("{}_{}.pt", eeg_id, fragment.slice_idx)))?;

            progress.fetch_add(1, Ordering::Relaxed);
            info!(
                "progress: {} out of {} total",
                progress.load(Ordering::Relaxed),
                total
            );
        }
    }

    Ok(())
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum MotifsBy {
    MotifId(usize), //TODO u32
    RunId(RunId),
}

impl From<&str> for MotifsBy {
    fn from(v: &str) -> MotifsBy {
        if v.starts_with("motif_id:") {
            MotifsBy::MotifId(
                v[v.find(":").expect(": not found") + 1..]
                    .parse()
                    .expect("can't parse {v} to motif_id"),
            )
        } else if v.starts_with("run_id:") {
            MotifsBy::RunId(
                v[v.find(":").expect(": not found") + 1..]
                    .parse()
                    .expect("can't parse {v} to run_id"),
            )
        } else {
            panic!("unexpected parameter {v}, should be motif:id:nnn or run_id:nnn")
        }
    }
}

#[derive(Debug, Args, Clone)]
pub struct PlotMotifSlicesArgs {
    #[arg(long, value_enum)]
    motifs_by: MotifsBy,

    #[arg(long, value_enum)]
    pub limit: Option<u32>,
}

impl PlotMotifSlicesArgs {
    pub fn motifs_by(&self) -> MotifsBy {
        self.motifs_by
    }

    pub fn new(motifs_by: MotifsBy) -> Self {
        Self {
            motifs_by,
            limit: None,
        }
    }
}

pub async fn load_motif_slices_2(
    channel: i64,
    find_motifs_run_config_applicable: &Arc<FindMotifsApplicable>,
    slices: Vec<TensorIdSliceIdx>,
    resources_map: HashMap<ResourceId, Resource>,
    runner: &Arc<ThoRunner>,
    device: &Device,
    ctx: Arc<RwLock<ThoRunCtx>>,
) -> Vec<(ResourceId, Vec<Tensor>)> {
    // debug!(
    //     "load_motif_slices: channel={}, find_motifs_run_config={:?}, slices={}",
    //     channel,
    //     find_motifs_run_config_applicable,
    //     slices.len()
    // );
    //
    // let tensors_ids_slices = slices.iter().fold(HashMap::new(), |mut a, slice| {
    //     a.entry(slice.0.clone()) //TODO clone
    //         .or_insert_with(|| Vec::with_capacity(10))
    //         .push(slice.1);
    //     a
    // });
    //
    // debug!(
    //     "load_motif_slices: tensors_ids_slices.len={}",
    //     tensors_ids_slices.len()
    // );
    //
    // // let mut all_slices:Arc<RwLock<Vec<(ResourceId, Vec<SyncWrapper<Tensor>>)>>> = Arc::new(RwLock::new(vec![]));//TODO remove Arc or add par_iter?
    // // let wg = WaitGroup::new();
    // let window = find_motifs_run_config_applicable.run_config().window;
    //
    // let all_slices = join_all(tensors_ids_slices.keys().collect::<Vec<_>>().chunks(100).flat_map(|tensor_keys| { //TODO remove chunks?
    //
    //     debug!("load_motif_slices: tensor_keys={:?}", tensor_keys);
    //
    //     tensor_keys
    //         .iter()
    //         .map(|v| resources_map.get(*v).expect(format!("no resource {v}").as_str()).clone()) //TODO clone + &Resource to load_future
    //         .map(|resource| {
    //             let ctx = Arc::clone(&ctx);
    //             let tensors_ids_slices = tensors_ids_slices.clone();
    //             async move {
    //                 let slices = tensors_ids_slices.get(&resource.id()).expect(format!("no {resource:?}").as_str()).clone();
    //
    //
    //                 let tensor_idx_id = resource.id();
    //                 // let t = Loader::<Tensor>::load_future(loader, resource, 0, future_id, done_sender).await;
    //                 let t = runner.run_find_motifs_applicable(&runner.run_loader(&resource, device, Some(Arc::clone(&ctx))).await, find_motifs_run_config_applicable, Some(Arc::clone(&ctx))).await;
    //                 // let t= find_motifs_run_config_applicable.apply(t.into(), FindMotifsRunConfigOptions::Filter).unwrap();
    //
    //                 debug!("load_motif_slices: tensor_idx={tensor_idx_id} slices.len={} t.size={:?}", slices.len(), t.size());
    //
    //                 // all_slices.write().push(
    //                 (
    //                     tensor_idx_id,
    //                     slices.into_iter().filter(|slice_idx| { //TODO filter_map
    //                         let slice_ok = t.size()[1] >= (*slice_idx + window) as i64;
    //
    //                         if !slice_ok {
    //                             warn!("slice doesn't fit into tensor: tensor_idx_id={tensor_idx_id}, t.size()={:?}, *slice_idx={slice_idx:?}, window={window:?}", t.size())
    //                         }
    //
    //                         slice_ok
    //                     }).map(|slice_idx|
    //                         t.t.i((channel, slice_idx as i64..(slice_idx + window) as i64))
    //                     ).collect::<Vec<Tensor>>()
    //                 )
    //             }
    //         })
    //         .collect::<Vec<_>>()
    // })).await;
    // // wg.wait();
    //
    // debug!("load_motif_slices: loaded slices={}", slices.len());
    //
    // all_slices
    todo!()
}

pub fn load_motif_slices(
    channel: i64,
    find_motifs_run_config_applicable: &FindMotifsApplicable,
    slices: Vec<TensorIdSliceIdx>,
    resources_map: HashMap<ResourceId, Resource>,
    loader: &Arc<RwLock<LoadRunner<Tensor>>>,
) -> Vec<(ResourceId, Vec<Tensor>)> {
    debug!(
        "load_motif_slices: channel={}, find_motifs_run_config={:?}, slices={}",
        channel,
        find_motifs_run_config_applicable,
        slices.len()
    );

    let tensors_ids_slices = slices.iter().fold(HashMap::new(), |mut a, slice| {
        a.entry(slice.0.clone()) //TODO clone
            .or_insert_with(|| Vec::with_capacity(10))
            .push(slice.1);
        a
    });

    debug!(
        "load_motif_slices: tensors_ids_slices.len={}",
        tensors_ids_slices.len()
    );

    let mut all_slices: Arc<RwLock<Vec<(ResourceId, Vec<SyncWrapper<Tensor>>)>>> =
        Arc::new(RwLock::new(vec![])); //TODO remove Arc or add par_iter?
    let wg = WaitGroup::new();
    let window = find_motifs_run_config_applicable.run_config().window;

    tensors_ids_slices.keys().collect::<Vec<_>>().chunks(100).for_each(|tensor_keys|{//TODO remove chunks?

        debug!("load_motif_slices: tensor_keys={:?}", tensor_keys);

        tensor_keys
            .iter()
            .map(|v|resources_map.get(*v).expect(format!("no resource {v}").as_str()).clone())//TODO clone + &Resource to load_future
            .for_each(|resource| {

            let slices = tensors_ids_slices.get(&resource.id()).expect(format!("no {resource:?}").as_str()).clone();

            let all_slices = Arc::clone(&all_slices);
            let wg = wg.clone();
            loader.write().exec_scoped(move |future_id, loader, _purge_resources, done_sender| {//TODO purge
                async move {
                    let tensor_idx_id = resource.id();
                    let t = Loader::<Tensor>::load_future(loader, resource, 0, future_id, done_sender).await;
                    let t= find_motifs_run_config_applicable.run(t.into(), FindMotifsRunConfigOptions::Filter).unwrap();

                    debug!("load_motif_slices: tensor_idx={tensor_idx_id} slices.len={} t.size={:?}", slices.len(), t.size());

                    all_slices.write().push(
                        (tensor_idx_id,
                         slices.into_iter().filter(|slice_idx| {//TODO filter_map
                             let slice_ok = t.size()[1] >= (*slice_idx  + window)as i64;

                             if !slice_ok {
                                 warn!("slice doesn't fit into tensor: tensor_idx_id={tensor_idx_id}, t.size()={:?}, *slice_idx={slice_idx:?}, window={window:?}", t.size())
                             }

                             slice_ok
                         }).map(|slice_idx|
                             SyncWrapper::wrap_tensor(t.as_ref().i((channel, slice_idx as i64..(slice_idx + window )as i64)))
                         ).collect::<Vec<SyncWrapper<Tensor>>>()
                        )
                    );
                    drop(wg);
                }.boxed()
            });
        });

    });
    wg.wait();

    debug!("load_motif_slices: loaded slices={}", slices.len());

    let t = all_slices
        .write()
        .drain(..)
        .map(|v| (v.0, v.1.into_iter().map(|b| b.shallow_clone()).collect()))
        .collect();
    t
}

#[derive(Debug, Args, Clone)]
pub struct MotifsToTensorArgs {
    #[arg(long)]
    run_id: RunId,

    #[arg(long, default_value_t = 1000)]
    window: i64,
}

impl MotifsToTensorArgs {
    pub fn window(&self) -> i64 {
        self.window
    }
    pub fn run_id(&self) -> RunId {
        self.run_id
    }
}

#[cfg(test)]
mod motifs_tests {
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::ops::Index;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::UNIX_EPOCH;

    use tch::{Device, IndexOp, Tensor};

    use log::debug;
    use parking_lot::RwLock;
    use tch::Device::Cpu;
    use tch::Kind::Float;
    use tch::kind::FLOAT_CPU;
    use thoosa_common::applicables::ThoRunner;
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::utils::*;
    use thoosa_common::ws::{Workspace, mock_ws_simp};
    use thoosa_shared::MotifSlice;
    use thoosa_shared::TensorIdSliceIdx;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::{Resource, ResourceId};
    use thoosa_shared::{ThoosaError, cdt_for_dir};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    // #[test]
    // fn test_motifs_to_tensor_2() -> Result<(), ThoosaError> {
    //     let dir = format!(
    //         "/tmp/test_{}",
    //         std::time::SystemTime::now()
    //             .duration_since(UNIX_EPOCH)?
    //             .as_secs()
    //     );
    //     debug!("dir={}", dir);
    //     let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 1000, 1000)));
    //
    //     let mut slices_map: HashMap<Resource, Vec<MotifSlice>> = HashMap::new();
    //
    //     slices_map.insert(
    //         Resource::new(1, 1, "111", 0),
    //         vec![
    //             MotifSlice::new(10, 100, 0),
    //             MotifSlice::new(11, 301, 1),
    //             MotifSlice::new(20, 500, 0),
    //             MotifSlice::new(21, 701, 1),
    //         ],
    //     );
    //
    //     slices_map.insert(
    //         Resource::new(2, 2, "222", 0),
    //         vec![
    //             MotifSlice::new(10, 200, 0),
    //             MotifSlice::new(11, 401, 1),
    //             MotifSlice::new(20, 600, 0),
    //             MotifSlice::new(21, 801, 1),
    //         ],
    //     );
    //
    //     fn test_tensor(eeg_id: &Resource) -> Tensor {
    //         let data = (0..1000).map(|i| i as f64).collect::<Vec<f64>>();
    //         match eeg_id.file_stem() {
    //             "111" => Tensor::vstack(&[
    //                 Tensor::from_slice(data.as_slice()), //ch 0
    //                 Tensor::from_slice(data.as_slice()), //ch 1
    //             ]),
    //
    //             "222" => Tensor::vstack(&[
    //                 Tensor::from_slice(data.as_slice()), //ch 0
    //                 Tensor::from_slice(data.as_slice()), //ch 1
    //             ]),
    //
    //             _ => panic!("eeg_id={:?}", eeg_id),
    //         }
    //     }
    //
    //     let runner = Arc::new(ThoRunner::new(&[10, 2], move |key, device| {
    //         debug!("loader_fn: thread={:?}", std::thread::current().name());
    //
    //         ThoTensor::new(Input::id(key).clone(), test_tensor(key))
    //     }));
    //
    //     let motifs_tensors = futures::executor::block_on(super::motifs_to_tensor_2(
    //         slices_map,
    //         &runner,
    //         &FindMotifsRunConfig::new(Some(1), 10, 10, None, None),
    //         2,
    //         Device::Cpu,
    //         &runner.next_ctx(),
    //         &ws,
    //     ))?;
    //
    //     debug!("slices_t={}", motifs_tensors.channels_motifs_slices);
    //     debug!("avg_dists_t={}", motifs_tensors.channels_motifs_avg_dists);
    //     debug!("motif_ids={:?}", motifs_tensors.channels_motifs_ids);
    //
    //     //16 = 7+9
    //     let motif_ids = from::<Vec<Vec<i64>>>(&motifs_tensors.channels_motifs_ids)
    //         .into_iter()
    //         .flat_map(|v| v.into_iter().map(|v| v as usize))
    //         .collect::<Vec<usize>>();
    //
    //     let mut motif_ids_sorted = motif_ids.clone();
    //     motif_ids_sorted.sort();
    //
    //     assert_eq!(vec![10, 11, 20, 21], motif_ids_sorted);
    //     //10 motif slices: 100-110, 200-210
    //     let motif_10_test = Tensor::stack(
    //         &[
    //             Tensor::from_slice((100..110).map(|v| v as f32).collect::<Vec<_>>().as_slice()),
    //             Tensor::from_slice((200..210).map(|v| v as f32).collect::<Vec<_>>().as_slice()),
    //         ],
    //         0,
    //     )
    //     .mean_dim(0, false, Float);
    //
    //     debug!("motif_10_test={motif_10_test}");
    //     let motif_10_idx = motif_ids
    //         .iter()
    //         .enumerate()
    //         .find(|(i, v)| v == &&10)
    //         .map(|v| v.0)
    //         .expect("motif 10");
    //
    //     let motif_10 = motifs_tensors
    //         .channels_motifs_slices
    //         .i((0, motif_10_idx as i64));
    //
    //     debug!("motif_10={motif_10}");
    //
    //     assert_eq!(motif_10_test, motif_10);
    //
    //     //TODO other motifs
    //
    //     Ok(())
    // }

    #[test]
    fn test_motifs_to_tensor() -> Result<(), ThoosaError> {
        let dir = format!(
            "/tmp/test_{}",
            std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)?
                .as_secs()
        );
        debug!("dir={}", dir);
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 1000, 1000)));

        let mut slices_map: HashMap<Resource, Vec<MotifSlice>> = HashMap::new();

        slices_map.insert(
            Resource::new(1, 1, "111", 0),
            vec![
                MotifSlice::new(10, 100, 0),
                MotifSlice::new(11, 301, 1),
                MotifSlice::new(20, 500, 0),
                MotifSlice::new(21, 701, 1),
            ],
        );

        slices_map.insert(
            Resource::new(2, 2, "222", 0),
            vec![
                MotifSlice::new(10, 200, 0),
                MotifSlice::new(11, 401, 1),
                MotifSlice::new(20, 600, 0),
                MotifSlice::new(21, 801, 1),
            ],
        );

        fn test_tensor(eeg_id: &Resource) -> Tensor {
            let data = (0..1000).map(|i| i as f64).collect::<Vec<f64>>();
            match eeg_id.file_stem() {
                "111" => Tensor::vstack(&[
                    Tensor::from_slice(data.as_slice()), //ch 0
                    Tensor::from_slice(data.as_slice()), //ch 1
                ]),

                "222" => Tensor::vstack(&[
                    Tensor::from_slice(data.as_slice()), //ch 0
                    Tensor::from_slice(data.as_slice()), //ch 1
                ]),

                _ => panic!("eeg_id={:?}", eeg_id),
            }
        }

        let loader = Arc::new(RwLock::new(LoadRunner::new(
            move |key: &Resource| SyncWrapper::wrap_tensor(test_tensor(key)),
            4,
            100,
            2,
        )));

        let motifs_tensors = super::motifs_to_tensor(
            slices_map,
            &loader,
            &FindMotifsRunConfig::new(None, 10, 10, None, None),
            2,
            Device::Cpu,
            1,
            &ws,
        )?;

        debug!("slices_t={}", motifs_tensors.channels_motifs_slices);
        debug!("avg_dists_t={}", motifs_tensors.channels_motifs_avg_dists);
        debug!("motif_ids={:?}", motifs_tensors.channels_motifs_ids);

        //16 = 7+9
        let motif_ids = from::<Vec<Vec<i64>>>(&motifs_tensors.channels_motifs_ids)
            .into_iter()
            .flat_map(|v| v.into_iter().map(|v| v as usize))
            .collect::<Vec<usize>>();

        let mut motif_ids_sorted = motif_ids.clone();
        motif_ids_sorted.sort();

        assert_eq!(vec![10, 11, 20, 21], motif_ids_sorted);
        //10 motif slices: 100-110, 200-210
        let motif_10_test = Tensor::stack(
            &[
                Tensor::from_slice((100..110).map(|v| v as f32).collect::<Vec<_>>().as_slice()),
                Tensor::from_slice((200..210).map(|v| v as f32).collect::<Vec<_>>().as_slice()),
            ],
            0,
        )
        .mean_dim(0, false, Float);

        debug!("motif_10_test={motif_10_test}");
        let motif_10_idx = motif_ids
            .iter()
            .enumerate()
            .find(|(i, v)| v == &&10)
            .map(|v| v.0)
            .expect("motif 10");

        let motif_10 = motifs_tensors
            .channels_motifs_slices
            .i((0, motif_10_idx as i64));

        debug!("motif_10={motif_10}");

        assert_eq!(motif_10_test, motif_10);

        //TODO other motifs

        Ok(())
    }

    #[test]
    fn test_write_motifs_slices() -> Result<(), ThoosaError> {
        let dir = format!("/tmp/test_write_motifs_slices/{}", cdt_for_dir());
        fs::create_dir_all(&dir)?;

        let mut map: HashMap<usize, Vec<MotifSlice>> = HashMap::new();

        let channel_idx = 1;
        let motif_group = 0;
        let slise_idx = 1;
        map.insert(
            0usize,
            vec![MotifSlice::new(motif_group, slise_idx, channel_idx)],
        );

        fn test_eeg_to_t(_base_dir: &str, _eeg_id: &str, _: Device) -> Result<Tensor, ThoosaError> {
            let t = Tensor::vstack(&[
                Tensor::from_slice(&[0., 1., 2., 3.]),
                Tensor::from_slice(&[0.1, 10.1, 20.1, 30.1]),
            ]);
            Ok(t)
        }

        super::write_motifs_slices(
            "../../data",
            dir.as_str(),
            map,
            vec!["568657".into()],
            test_eeg_to_t,
            2,
        )?;

        let loaded_slice = Tensor::load(
            Path::new(&dir)
                .join(channel_idx.to_string())
                .join(motif_group.to_string())
                .join(format!("568657_{}.pt", slise_idx)),
        )?;

        assert!(
            Tensor::from_slice(&[10.1, 20.1])
                .view([1, -1])
                .equal(&loaded_slice)
        );

        Ok(())
    }

    #[test]
    fn test_load_motif_slices() -> Result<(), ThoosaError> {
        fn test_load(_eeg_id: &Resource) -> SyncWrapper<Tensor> {
            let t = Tensor::vstack(&[
                Tensor::from_slice(&[0., 1., 2., 3.]),        //ch 0
                Tensor::from_slice(&[0.1, 10.1, 20.1, 30.1]), //ch 1
            ]);

            SyncWrapper::wrap_tensor(t)
        }

        let executor = LoadRunner::new(test_load, 4, 100, 2);
        let executor_rc = Arc::new(RwLock::new(executor));

        let motif_slices = super::load_motif_slices(
            0,
            &FindMotifsRunConfig::new(None, 1, 2, None, None)
                .to_applicable(&mock_ws_simp(None, 1, 1000, 100), Cpu),
            vec![TensorIdSliceIdx(1, 2)],
            HashMap::from([(1, Resource::new(1, 1, "aaa", 100))]),
            &executor_rc,
        );

        assert_eq!(
            vec![(1, vec![Tensor::from_slice(&[2.0, 3.0])])],
            motif_slices
        );

        debug!("motif_slices={:?}", motif_slices);

        Ok(())
    }

    #[test]
    fn test_load_motif_slices_5000() -> Result<(), ThoosaError> {
        fn test_eeg_to_t(_eeg_id: &Resource) -> SyncWrapper<Tensor> {
            let t = Tensor::vstack(&[
                Tensor::from_slice(
                    (0..16781)
                        .map(|v| v as f32)
                        .collect::<Vec<f32>>()
                        .as_slice(),
                ), //ch 0
            ]);

            SyncWrapper::wrap_tensor(t)
        }

        let executor = LoadRunner::new(test_eeg_to_t, 4, 100, 2);
        let executor_rc = Arc::new(parking_lot::RwLock::new(executor));

        let motif_slices = super::load_motif_slices(
            0,
            &FindMotifsRunConfig::new(None, 1, 5000, None, None)
                .to_applicable(&mock_ws_simp(None, 1, 1000, 100), Cpu),
            vec![TensorIdSliceIdx(123, 2)],
            HashMap::from([(123, Resource::new(123, 1, "aaa", 100))]),
            &executor_rc,
        );

        assert_eq!(123, motif_slices[0].0);
        assert_eq!(5000, motif_slices[0].1[0].size()[0]);

        Ok(())
    }

    #[test]
    fn test_0() {
        let t = Tensor::from_slice(&[2., 4., 1., 1., 3., 2., 3., 2., 3.])
            .view((3, 3))
            .mean_dim(0, true, Float);

        println!("t={t}");
    }
}
