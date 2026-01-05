use std::collections::HashMap;
use crate::tasks::cross_batch_clc::{CrossClcResultMerge, CrossUnfoldClcOutput};
use crate::tasks::run_ctx_2::RunCtx2;
use thoosa_shared::runner::common::cachable::DevicesWrap;
use thoosa_shared::runner::common::runnable::{Stage, SyncStage};
use thoosa_shared::runner::common::RunCtx;
use log::{debug, error, info, trace, warn};
use parking_lot::{Mutex, RwLock};
use std::fmt::{Debug, Formatter};
use std::iter::{Repeat, Zip};
use std::mem;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
use tch::{IndexOp, Kind, Tensor};
use futures::{future::BoxFuture, stream, FutureExt, StreamExt, TryStreamExt};
use tch::Device::{Cpu, Cuda};
use tch::Kind::{Bool, Float, Uint8};
use thoosa_common::bool_iter::BoolTensorIterator;
use thoosa_common::copy_ctx::CopyCtx;
use thoosa_common::cuda_utils::{pack_bits_bool, unpack_bits_u8};
use thoosa_common::dims::AnyDims;
use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
use thoosa_common::loader::rayon_pool;
use thoosa_common::tho_device::ThoDevice;
use thoosa_shared::traits::MonoMetrics;
use thoosa_shared::{CopyToDevice, TensorIdSliceIdx};
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::traits::Data;
use thoosa_common::ws::Workspace;
use thoosa_motifs::find::{FixedFindMotifsConfig, SimpleMotifTracker};
use thoosa_shared::ThoError;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::traits::WithId;
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use thoosa_store::fs_store::FileMotifTrackerStore;
use thoosa_store::{SqlMotifTrackerStore, Store};
use thoosa_common::tho_tensor::ThoTensor;

// Data impl is minimal; DistsMask is still meant for one-way GPU->CPU copies.
#[derive(Debug)]
pub struct DistsMask {
    mask: Tensor,
    dists: Option<Tensor>,
    sizes: Option<(Vec<i64>, Option<Vec<i64>>)>,//if packed
    stream_to_sync: Option<ThoCudaStream>,
}


unsafe impl Sync for DistsMask {}

impl Output for DistsMask {}
impl Input for DistsMask {}

impl WithId for DistsMask {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        todo!()
    }
}

impl Data for DistsMask {
    type Metrics = ();
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        ()
    }

    fn device(&self) -> usize {
        match self.mask.device() {
            Cpu => 0,
            Cuda(idx) => idx as usize,
            _ => 0,
        }
    }
}

impl DistsMask {

    pub fn new(mask: Tensor, dists: Option<Tensor>, pack: bool, stream_to_sync: Option<ThoCudaStream>) -> Result<Self, ThoError> {
        debug!("DistsMask::new: mask={mask:?} dists={dists:?}");

        assert_eq!(mask.kind(), Bool);
        let sizes = if pack {
            Some((mask.size(), dists.as_ref().map(|t|t.size())))
        } else {
            None
        };

        let mask = if pack {
            pack_bits_bool(&mask)?
        } else {
            mask
        };

        Ok(Self {
            sizes,
            mask,
            dists,
            stream_to_sync
        })
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="DistsMask::to_cpu"))]
    pub async fn to_cpu(self) -> Result<Self, ThoError>  {
        debug!("DistsMask::to_cpu: self.mask.device={:?}", self.mask.device());

        if self.mask.device().eq(&Cpu) {
            Ok(self)
        } else {
            if let Some(stream) = self.stream_to_sync {
                stream.synchronize();
            }

            Ok(Self {

                mask: if self.sizes.is_some() {//was packed
                    CopyCtx::instance(ThoDevice::from(self.mask.device())).move_t(self.mask, 0, Uint8).await?
                } else {
                    CopyCtx::instance(ThoDevice::from(self.mask.device())).move_t(self.mask, 0, Bool).await?
                },

                dists: None,
                sizes: self.sizes,
                stream_to_sync:None
            })
        }
    }

    pub async fn to_cpu_with_dists(self) -> Result<Self, ThoError>  {
        debug!("DistsMask::to_cpu_with_dists: self.mask.device={:?}", self.mask.device());

        if self.mask.device().eq(&Cpu) {
            let dists = match self.dists {
                Some(dists) if !dists.device().eq(&Cpu) => {
                    let device = ThoDevice::from(dists.device());
                    let kind = dists.kind();
                    Some(CopyCtx::instance(device).move_t(dists, 0, kind).await?)
                }
                other => other,
            };
            Ok(Self {
                mask: self.mask,
                dists,
                sizes: self.sizes,
                stream_to_sync: None,
            })
        } else {
            if let Some(stream) = self.stream_to_sync {
                stream.synchronize();
            }

            let device = ThoDevice::from(self.mask.device());
            let mask = if self.sizes.is_some() {//was packed
                CopyCtx::instance(device).move_t(self.mask, 0, Uint8).await?
            } else {
                CopyCtx::instance(device).move_t(self.mask, 0, Bool).await?
            };
            let dists = match self.dists {
                Some(dists) if dists.device().eq(&Cpu) => Some(dists),
                Some(dists) => {
                    let kind = dists.kind();
                    let dists_device = ThoDevice::from(dists.device());
                    Some(CopyCtx::instance(dists_device).move_t(dists, 0, kind).await?)
                }
                None => None,
            };

            Ok(Self {
                mask,
                dists,
                sizes: self.sizes,
                stream_to_sync: None,
            })
        }
    }

    pub fn unpack(self) -> Result<(Tensor, Option<Tensor>), ThoError> {
        debug!("DistsMask::unpack: self.sizes={:?}", self.sizes);
        match &self.sizes {//was packed
            Some(size) => {
                Ok((
                    unpack_bits_u8(
                        &self.mask,
                        Some(size.0.iter().fold(1, |mut a, v| {a*=v; a}))
                    )
                    .f_to_kind(Bool)?
                    .f_view(size.0.as_slice())?,
                    self.dists
                ))
            },
            None => {
                Ok((self.mask, self.dists))
            }
        }
    }

}

impl CrossClcResultMerge for DistsMask {
    //gpt generated
    fn to_cpu(self) -> BoxFuture<'static, Result<Self, ThoError>> {
        async move { DistsMask::to_cpu(self).await }.boxed()
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CrossClcResultMerge for DistsMask::merge_batch_1"))]
    fn merge_batch_1(self, other: Self) -> Result<Self, ThoError> {
        let left = self;
        let right = other;

        // Ensure any pending GPU work that produced the masks/dists is complete
        // before we read/concatenate them on the current stream.
        if let Some(stream) = left.stream_to_sync.as_ref() {
            stream.synchronize();
        }
        if let Some(stream) = right.stream_to_sync.as_ref() {
            stream.synchronize();
        }

        // Packed and unpacked representations cannot be mixed.
        let pack = left.sizes.is_some();
        if pack != right.sizes.is_some() {
            return Err(ThoError::new("DistsMask merge: pack mismatch"));
        }
        // Unpacked path is simple: concatenate along batch-1 (dim 2) and rewrap.
        if !pack {
            let mask = Tensor::f_concat(&[left.mask, right.mask], 2)?;
            let dists = match (left.dists, right.dists) {
                (Some(left), Some(right)) => Some(Tensor::f_concat(&[left, right], 2)?),
                (None, None) => None,
                _ => return Err(ThoError::new("DistsMask merge: dists mismatch")),
            };

            return Ok(DistsMask::new(mask, dists, false, None)?);
        }

        // Packed path: sizes keep the original logical (bool) shape before packing.
        let left_sizes = left
            .sizes
            .as_ref()
            .map(|v| v.0.clone())
            .ok_or_else(|| ThoError::new("DistsMask merge: missing left sizes"))?;
        let right_sizes = right
            .sizes
            .as_ref()
            .map(|v| v.0.clone())
            .ok_or_else(|| ThoError::new("DistsMask merge: missing right sizes"))?;

        // All dimensions except the last must match; last dim is concatenated.
        if left_sizes.len() != right_sizes.len() || left_sizes.is_empty() {
            return Err(ThoError::new("DistsMask merge: size mismatch"));
        }

        let last_dim = left_sizes.len() - 1;
        if left_sizes[..last_dim] != right_sizes[..last_dim] {
            return Err(ThoError::new("DistsMask merge: size mismatch"));
        }

        // "rows" is the number of logical rows after flattening all but the last dim.
        let left_last = left_sizes[last_dim];
        let right_last = right_sizes[last_dim];
        let rows: i64 = left_sizes[..last_dim]
            .iter()
            .fold(1i64, |a, v| a.saturating_mul(*v));

        // Total bit counts are needed for unpacking, including any per-tensor padding.
        let total_left_bits: i64 = left_sizes.iter().fold(1i64, |a, v| a.saturating_mul(*v));
        let total_right_bits: i64 = right_sizes.iter().fold(1i64, |a, v| a.saturating_mul(*v));

        // If last dim is byte-aligned for both inputs, concatenate the packed bytes
        // directly (fast path). Otherwise unpack to bits, concat, then repack.
        let packed_mask = if left_last % 8 == 0 && right_last % 8 == 0 {
            let left_bytes = left_last / 8;
            let right_bytes = right_last / 8;
            let left_view = left.mask.f_view([rows, left_bytes])?;
            let right_view = right.mask.f_view([rows, right_bytes])?;
            Tensor::f_concat(&[left_view, right_view], 1)?
                .f_reshape([-1])?
                .f_contiguous()?
        } else {
            let left_bits = unpack_bits_u8(&left.mask, Some(total_left_bits));
            let right_bits = unpack_bits_u8(&right.mask, Some(total_right_bits));
            let left_bits = left_bits.f_view([rows, left_last])?;
            let right_bits = right_bits.f_view([rows, right_last])?;
            let merged_bits = Tensor::f_concat(&[left_bits, right_bits], 1)?;
            pack_bits_bool(&merged_bits.f_reshape([-1])?)?
        };

        // Dists are optional, but must be present (or absent) on both sides.
        let dists = match (left.dists, right.dists) {
            (Some(left), Some(right)) => Some(Tensor::f_concat(&[left, right], 2)?),
            (None, None) => None,
            _ => return Err(ThoError::new("DistsMask merge: dists mismatch")),
        };

        // Update stored sizes to reflect the concatenated last dimension.
        let mut merged_sizes = left_sizes;
        merged_sizes[last_dim] += right_last;
        let merged_dists_sizes = left
            .sizes
            .and_then(|v| v.1)
            .zip(right.sizes.and_then(|v| v.1))
            .map(|(mut left_dists, right_dists)| {
                // Dists sizes have the same shape rules as the mask.
                if left_dists.len() != right_dists.len() {
                    return Err(ThoError::new("DistsMask merge: dists size mismatch"));
                }
                let last = left_dists.len() - 1;
                if left_dists[..last] != right_dists[..last] {
                    return Err(ThoError::new("DistsMask merge: dists size mismatch"));
                }
                left_dists[last] += right_dists[last];
                Ok(left_dists)
            })
            .transpose()?;

        // Build a packed DistsMask without re-packing, preserving packed mask bytes.
        Ok(DistsMask {
            mask: packed_mask,
            dists,
            sizes: Some((merged_sizes, merged_dists_sizes)),
            stream_to_sync: None,
        })
    }

    //gpt generated
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CrossClcResultMerge for DistsMask::merge_batch_1_many"))]
    fn merge_batch_1_many(self, others: Vec<Self>) -> Result<Self, ThoError> {
        if others.is_empty() {
            return Ok(self);
        }

        // Collect all parts so we can compute sizes and preallocate once.
        let mut parts = Vec::with_capacity(others.len() + 1);
        parts.push(self);
        parts.extend(others);

        // Ensure any pending GPU work is complete before we read from tensors.
        for part in &parts {
            if let Some(stream) = part.stream_to_sync.as_ref() {
                stream.synchronize();
            }
        }

        // Packed and unpacked representations cannot be mixed.
        let pack = parts[0].sizes.is_some();
        if parts.iter().any(|p| p.sizes.is_some() != pack) {
            return Err(ThoError::new("DistsMask merge: pack mismatch"));
        }

        // Unpacked path: preallocate output and copy slices along last dim.
        if !pack {
            let first_size = parts[0].mask.size();
            if first_size.is_empty() {
                return Err(ThoError::new("DistsMask merge: size mismatch"));
            }
            let last_dim = first_size.len() - 1;
            let device = parts[0].mask.device();
            let mut total_last = first_size[last_dim];

            // Validate shapes/devices and compute total concatenated length.
            for part in parts.iter().skip(1) {
                let size = part.mask.size();
                if size.len() != first_size.len() || size[..last_dim] != first_size[..last_dim] {
                    return Err(ThoError::new("DistsMask merge: size mismatch"));
                }
                if part.mask.device() != device {
                    return Err(ThoError::new("DistsMask merge: device mismatch"));
                }
                total_last += size[last_dim];
            }

            let mut merged_mask_sizes = first_size;
            merged_mask_sizes[last_dim] = total_last;
            let mut mask_out = Tensor::zeros(merged_mask_sizes.as_slice(), (Bool, device));

            // Copy each partial mask into the preallocated output.
            let mut offset = 0;
            for part in parts.iter() {
                let len = part.mask.size()[last_dim];
                mask_out
                    .narrow(last_dim as i64, offset, len)
                    .copy_(&part.mask);
                offset += len;
            }

            // Dists are optional, but must be present (or absent) on every part.
            let has_dists = parts[0].dists.is_some();
            if parts.iter().any(|p| p.dists.is_some() != has_dists) {
                return Err(ThoError::new("DistsMask merge: dists mismatch"));
            }

            let dists = if has_dists {
                let first_dists = parts[0].dists.as_ref().unwrap();
                let dists_device = first_dists.device();
                let dists_kind = first_dists.kind();
                let dists_size = first_dists.size();
                if dists_size.is_empty() {
                    return Err(ThoError::new("DistsMask merge: dists size mismatch"));
                }
                let dists_last_dim = dists_size.len() - 1;
                let mut total_dists_last = dists_size[dists_last_dim];

                // Validate dists shapes/devices and compute total concatenated length.
                for part in parts.iter().skip(1) {
                    let dists = part.dists.as_ref().unwrap();
                    let size = dists.size();
                    if size.len() != dists_size.len()
                        || size[..dists_last_dim] != dists_size[..dists_last_dim]
                    {
                        return Err(ThoError::new("DistsMask merge: dists size mismatch"));
                    }
                    if dists.device() != dists_device {
                        return Err(ThoError::new("DistsMask merge: dists device mismatch"));
                    }
                    total_dists_last += size[dists_last_dim];
                }

                let mut merged_dists_sizes = dists_size;
                merged_dists_sizes[dists_last_dim] = total_dists_last;
                let mut dists_out =
                    Tensor::zeros(merged_dists_sizes.as_slice(), (dists_kind, dists_device));

                // Copy each partial dists tensor into the preallocated output.
                let mut offset = 0;
                for part in parts.iter() {
                    let dists = part.dists.as_ref().unwrap();
                    let len = dists.size()[dists_last_dim];
                    dists_out
                        .narrow(dists_last_dim as i64, offset, len)
                        .copy_(dists);
                    offset += len;
                }

                Some(dists_out)
            } else {
                None
            };

            // Rewrap the merged tensors as an unpacked DistsMask.
            return Ok(DistsMask::new(mask_out, dists, false, None)?);
        }

        // Packed path: use stored logical sizes to avoid unpacking unless required.
        let first_sizes = parts[0]
            .sizes
            .as_ref()
            .ok_or_else(|| ThoError::new("DistsMask merge: missing left sizes"))?;
        let base_sizes = first_sizes.0.clone();
        if base_sizes.is_empty() {
            return Err(ThoError::new("DistsMask merge: size mismatch"));
        }
        let last_dim = base_sizes.len() - 1;
        let device = parts[0].mask.device();

        // Validate shapes/devices and compute total concatenated length.
        let mut total_last = base_sizes[last_dim];
        let mut all_aligned = base_sizes[last_dim] % 8 == 0;

        for part in parts.iter().skip(1) {
            let sizes = part
                .sizes
                .as_ref()
                .ok_or_else(|| ThoError::new("DistsMask merge: missing right sizes"))?;
            if sizes.0.len() != base_sizes.len() || sizes.0[..last_dim] != base_sizes[..last_dim] {
                return Err(ThoError::new("DistsMask merge: size mismatch"));
            }
            if part.mask.device() != device {
                return Err(ThoError::new("DistsMask merge: device mismatch"));
            }
            total_last += sizes.0[last_dim];
            all_aligned &= sizes.0[last_dim] % 8 == 0;
        }

        // "rows" is the number of logical rows after flattening all but the last dim.
        let rows: i64 = base_sizes[..last_dim]
            .iter()
            .fold(1i64, |a, v| a.saturating_mul(*v));

        // If every chunk is byte-aligned, stitch packed bytes directly.
        // Otherwise, unpack to bits, copy, then repack.
        let packed_mask = if all_aligned {
            let total_bytes = total_last / 8;
            let mut out = Tensor::zeros([rows, total_bytes], (Uint8, device));
            let mut byte_offset = 0;
            for part in parts.iter() {
                let sizes = part.sizes.as_ref().unwrap();
                let bytes = sizes.0[last_dim] / 8;
                let part_view = part.mask.f_view([rows, bytes])?;
                out.narrow(1, byte_offset, bytes).copy_(&part_view);
                byte_offset += bytes;
            }
            out.f_reshape([-1])?
        } else {
            let mut out_bits = Tensor::zeros([rows, total_last], (Bool, device));
            let mut offset = 0;
            for part in parts.iter() {
                let sizes = part.sizes.as_ref().unwrap();
                let part_last = sizes.0[last_dim];
                let total_bits = sizes.0.iter().fold(1i64, |a, v| a.saturating_mul(*v));
                let bits = unpack_bits_u8(&part.mask, Some(total_bits));
                let bits = bits.f_view([rows, part_last])?;
                out_bits.narrow(1, offset, part_last).copy_(&bits);
                offset += part_last;
            }
            pack_bits_bool(&out_bits.f_reshape([-1])?)?
        };

        // Dists are optional, but must be present (or absent) on every part.
        let has_dists = parts[0].dists.is_some();
        if parts.iter().any(|p| p.dists.is_some() != has_dists) {
            return Err(ThoError::new("DistsMask merge: dists mismatch"));
        }

        let (dists, merged_dists_sizes) = if has_dists {
            let first_dists = parts[0].dists.as_ref().unwrap();
            let dists_device = first_dists.device();
            let dists_kind = first_dists.kind();
            let dists_size = first_sizes
                .1
                .as_ref()
                .ok_or_else(|| ThoError::new("DistsMask merge: dists size mismatch"))?
                .clone();
            if dists_size.is_empty() {
                return Err(ThoError::new("DistsMask merge: dists size mismatch"));
            }
            let dists_last_dim = dists_size.len() - 1;
            let mut total_dists_last = dists_size[dists_last_dim];

            // Validate dists shapes/devices and compute total concatenated length.
            for part in parts.iter().skip(1) {
                let sizes = part
                    .sizes
                    .as_ref()
                    .ok_or_else(|| ThoError::new("DistsMask merge: dists size mismatch"))?;
                let size = sizes
                    .1
                    .as_ref()
                    .ok_or_else(|| ThoError::new("DistsMask merge: dists size mismatch"))?;
                if size.len() != dists_size.len()
                    || size[..dists_last_dim] != dists_size[..dists_last_dim]
                {
                    return Err(ThoError::new("DistsMask merge: dists size mismatch"));
                }
                if part.dists.as_ref().unwrap().device() != dists_device {
                    return Err(ThoError::new("DistsMask merge: dists device mismatch"));
                }
                total_dists_last += size[dists_last_dim];
            }

            let mut merged_dists_sizes = dists_size;
            merged_dists_sizes[dists_last_dim] = total_dists_last;
            let mut dists_out =
                Tensor::zeros(merged_dists_sizes.as_slice(), (dists_kind, dists_device));

            // Copy each partial dists tensor into the preallocated output.
            let mut offset = 0;
            for part in parts.iter() {
                let dists = part.dists.as_ref().unwrap();
                let len = dists.size()[dists_last_dim];
                dists_out
                    .narrow(dists_last_dim as i64, offset, len)
                    .copy_(dists);
                offset += len;
            }

            (Some(dists_out), Some(merged_dists_sizes))
        } else {
            (None, None)
        };

        // Update stored sizes to reflect the concatenated last dimension.
        let mut merged_sizes = base_sizes;
        merged_sizes[last_dim] = total_last;
        // Build a packed DistsMask without re-packing, preserving packed mask bytes.
        Ok(DistsMask {
            mask: packed_mask,
            dists,
            sizes: Some((merged_sizes, merged_dists_sizes)),
            stream_to_sync: None,
        })
    }
}

pub struct DistsTracker<C>
where
    C: RunCtx + 'static,
{
    channel_trackers: Vec<Arc<RwLock<SimpleMotifTracker<'static>>>>,
    store: Arc<RwLock<Box<dyn Store>>>,
    _ctx: PhantomData<C>,
}

impl<C> Debug for DistsTracker<C>
where
    C: RunCtx + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistsTracker")
            .field("channel_trackers", &self.channel_trackers)
            .finish()
    }
}

impl<C> DistsTracker<C>
where
    C: RunCtx + 'static,
{
    pub fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        run_config: &FindMotifsRunConfig,
        fixed_config: &FixedFindMotifsConfig,
        max_slices_counts: Option<u32>,
    ) -> Self {
        let channel_trackers = (0..ws.read().channels())
            .map(|channel| {
                debug!("DistsTracker:new: channel={channel}");

                Arc::new(RwLock::new(SimpleMotifTracker::new(
                    if fixed_config.keep_slice_matches {
                        SqlMotifTrackerStore::new(
                            ws.read().tmp_run_store_path(
                                run_config.id.expect("id") as u32,
                                channel as u32,
                            ),
                            run_config.id.expect("id"),
                            channel,
                        )
                        .expect("can't create tmp store")
                    } else {
                        FileMotifTrackerStore::new(
                            ws.read().tmp_path(),
                            run_config.id.expect("id"),
                            channel,
                        )
                        .expect("can't create tmp store")
                    },
                    max_slices_counts,
                )))
            })
            .collect::<Vec<Arc<RwLock<SimpleMotifTracker>>>>();

        debug!("DistsTracker: channel_trackers={:?}", channel_trackers);
        Self {
            channel_trackers,
            store: Arc::clone(store),
            _ctx: PhantomData,
        }
    }


    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="DistsTracker::build_groups"))]
    pub async fn build_groups(&self) -> Result<(), ThoError>{
        let mut trackers_done = AtomicU32::new(0);
        let store = Arc::clone(&self.store);
        let channel_trackers = self.channel_trackers.clone();//TODO clone
        let channels_groups = Arc::new(RwLock::new(HashMap::new()));
        let num_channels = channel_trackers.len();
        let channels_groups_arc = Arc::clone(&channels_groups);
        tokio::task::spawn_blocking(move || {
            use rayon::prelude::*;
            rayon_pool().install(|| {

                channel_trackers.par_iter().enumerate().try_for_each(|(channel, mt)| {
                    let groups_inserted = mt.write().build_groups(&store)?;
                    channels_groups_arc.write().insert(channel, groups_inserted);
                    Ok::<_, ThoError>(())
                })
            })
        }).await??;

        let g = channels_groups.read();
        info!("groups inserted: {:?}", (0..self.channel_trackers.len()).map(|c|g.get(&c).map(|v|*v).ok_or(0)).collect::<Vec<_>>());
        Ok(())
    }
}

pub fn dist_iter(dists_flattened: Option<Tensor>) -> Box<dyn Iterator<Item = f64>> {
    //TODO use Either::
    match dists_flattened.as_ref().map(|v| v.iter::<f64>().unwrap()) {
        None => Box::new(std::iter::repeat(0f64)) as Box<dyn Iterator<Item = f64>>,
        Some(v) => Box::new(v) as Box<dyn Iterator<Item = f64>>,
    }
}


impl<C> Stage for DistsTracker<C>
where
    C: RunCtx + 'static,
{
    type Ctx = C;

    type In<'a> = CrossUnfoldClcOutput<DistsMask>;
    type Out = ();

}

impl<C> SyncStage for DistsTracker<C>
where
    C: RunCtx + 'static,
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="DistsTracker::consume"))]
    async fn consume<'a>(
        self: Arc<Self>,
        input: Self::In<'a>,
        ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Self::Out, ThoError> {


        //TODO clone to resources ref
        let iter_0 = input.iter_0();
        let iter_1 = input.iter_1();

        let (mask_cpu, dists_cpu)= input.clc_result().unpack()?;
        let mask_cpu_flat:ThoTensor<AnyDims> = ThoTensor::from(mask_cpu.f_flatten(1, 2)?);//it's going to be used read only so ok to wrap it in Send + Sync  ThoTensor
        let dists_cpu_flat = match dists_cpu.as_ref() {
            Some(d) => Some(ThoTensor::from(d.f_flatten(1, 2)?)),
            None => None,
        };

        tokio::task::spawn_blocking(move || {
            use rayon::prelude::*;
            rayon_pool().install(|| {
                self.channel_trackers.par_iter().enumerate().try_for_each(|(channel, mt)| {
                    let iter_0 = iter_0.clone();
                    let iter_1 = iter_1.clone();

                    let mask_cpu_flattened = mask_cpu_flat.f_i((channel as i64, ..))?;
                    let dists_flattened = match dists_cpu_flat.as_ref() {
                        Some(d) => Some(d.f_i((channel as i64, ..))?),
                        None => None,
                    };


                    debug!("find_motifs_by_euc_dist: dists_flattened.size={:?}",dists_flattened.as_ref().map(|v| v.size()));
                    trace!("find_motifs_by_euc_dist: mask_cpu_flattened=\n{mask_cpu_flattened}");
                    trace!("find_motifs_by_euc_dist: dists_flattened=\n{:?}",dists_flattened);

                    let mut channel_mask: BoolTensorIterator = (&mask_cpu_flattened).into();
                    let mut dists_and_mask_iter = channel_mask.zip(dist_iter(dists_flattened));//

                    debug!("find_motifs_by_euc_dist: channel={channel} mask_cpu_flattened.size={:?}",mask_cpu_flattened.size());
                    trace!("find_motifs_by_euc_dist: mask_cpu_flattened={mask_cpu_flattened}");


                    for slice_0 in iter_0 {
                        let iter_1 = iter_1.clone();

                        for slice_1 in iter_1 {
                            if let Some((mask, dist)) = dists_and_mask_iter.next() {
                                trace!("find_motifs_by_euc_dist: slice_0={slice_0:?}  slice_1={slice_1:?} mask={mask:?} dist={dist}");

                                if mask
                                    && (slice_0.0 != slice_1.0 || slice_0.1 != slice_0.1
                                ) {
                                    let slice_0_clone = slice_0.clone();
                                    let slice_1_clone = slice_1.clone();


                                    mt.write().track(
                                        slice_0_clone,
                                        slice_1_clone,
                                        dist as f32, //TODO actual distances not needed?
                                    );
                                }
                            } else {
                                error!("find_motifs_by_euc_dist: not enough dists");
                                panic!("find_motifs_by_euc_dist: not enough dists");
                            }
                        }
                    }


                    mt.write().dump();
                    Ok::<(), ThoError>(())
                })
            })
        })
        .await??;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use log::debug;
    use crate::tasks::cross_batch_clc::CrossClcResultMerge;
    use crate::tasks::dists_tracker::DistsMask;
    use tch::{Kind, Tensor};
    use thoosa_common::utils::reset_print_options;
    use thoosa_shared::{Cause, ThoError};
    #[ctor::ctor]
    pub fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test]
    async fn test_0() -> Result<(), ThoError> {
        Ok(())
    }

    fn mask_values(mask: &Tensor) -> Vec<i64> {
        mask.reshape([-1])
            .to_kind(Kind::Int64)
            .contiguous()
            .iter::<i64>()
            .unwrap()
            .collect::<Vec<_>>()
    }

    fn dists_values(dists: &Tensor) -> Vec<f64> {
        dists.reshape([-1])
            .to_kind(Kind::Double)
            .contiguous()
            .iter::<f64>()
            .unwrap()
            .collect::<Vec<_>>()
    }

    #[test]
    fn test_merge_batch_1_packed_with_dists() -> Result<(), ThoError> {
        let left_mask = Tensor::from_slice(&[0.0f32, 1.0]).view([1, 1, 2]).lt(0.5);
        let right_mask = Tensor::from_slice(&[1.0f32, 0.0]).view([1, 1, 2]).lt(0.5);
        let left_dists = Tensor::from_slice(&[1.0f32, 2.0]).view([1, 1, 2]);
        let right_dists = Tensor::from_slice(&[3.0f32, 4.0]).view([1, 1, 2]);

        let left = DistsMask::new(left_mask, Some(left_dists), true, None)?;
        let right = DistsMask::new(right_mask, Some(right_dists), true, None)?;

        let merged = left.merge_batch_1(right)?;
        let (mask, dists) = merged.unpack()?;
        let dists = dists.expect("dists");

        assert_eq!(mask.size(), vec![1, 1, 4]);
        assert_eq!(dists.size(), vec![1, 1, 4]);
        assert_eq!(mask_values(&mask), vec![1, 0, 0, 1]);
        assert_eq!(dists_values(&dists), vec![1.0, 2.0, 3.0, 4.0]);

        Ok(())
    }

    #[test]
    fn test_merge_batch_1_without_dists() -> Result<(), ThoError> {
        let left_mask = Tensor::from_slice(&[0.0f32]).view([1, 1, 1]).lt(0.5);
        let right_mask = Tensor::from_slice(&[1.0f32, 0.0]).view([1, 1, 2]).lt(0.5);

        let left = DistsMask::new(left_mask, None, false, None)?;
        let right = DistsMask::new(right_mask, None, false, None)?;

        let merged = left.merge_batch_1(right)?;
        let (mask, dists) = merged.unpack()?;

        assert!(dists.is_none());
        assert_eq!(mask.size(), vec![1, 1, 3]);
        assert_eq!(mask_values(&mask), vec![1, 0, 1]);

        Ok(())
    }

    #[test]
    fn test_merge_batch_1_dists_mismatch() -> Result<(), ThoError> {
        let left_mask = Tensor::from_slice(&[0.0f32]).view([1, 1, 1]).lt(0.5);
        let right_mask = Tensor::from_slice(&[1.0f32]).view([1, 1, 1]).lt(0.5);
        let left_dists = Tensor::from_slice(&[1.0f32]).view([1, 1, 1]);

        let left = DistsMask::new(left_mask, Some(left_dists), false, None)?;
        let right = DistsMask::new(right_mask, None, false, None)?;

        let err = left.merge_batch_1(right).expect_err("expected dists mismatch");
        match err.what() {
            Cause::SomethingElse(msg) => {
                assert_eq!(msg, "DistsMask merge: dists mismatch");
            }
            other => panic!("unexpected error: {other:?}"),
        }

        Ok(())
    }

    #[test]
    fn test_merge_batch_1_pack_mismatch() -> Result<(), ThoError> {
        let left_mask = Tensor::from_slice(&[0.0f32]).view([1, 1, 1]).lt(0.5);
        let right_mask = Tensor::from_slice(&[1.0f32]).view([1, 1, 1]).lt(0.5);

        let left = DistsMask::new(left_mask, None, true, None)?;
        let right = DistsMask::new(right_mask, None, false, None)?;

        let err = left.merge_batch_1(right).expect_err("expected pack mismatch");
        match err.what() {
            Cause::SomethingElse(msg) => {
                assert_eq!(msg, "DistsMask merge: pack mismatch");
            }
            other => panic!("unexpected error: {other:?}"),
        }

        Ok(())
    }

    #[test]
    fn test_merge_batch_1_many_packed_with_dists() -> Result<(), ThoError> {
        let left_mask = Tensor::from_slice(&[0.0f32, 1.0]).view([1, 1, 2]).lt(0.5);
        let mid_mask = Tensor::from_slice(&[1.0f32, 0.0]).view([1, 1, 2]).lt(0.5);
        let right_mask = Tensor::from_slice(&[0.0f32, 0.0]).view([1, 1, 2]).lt(0.5);

        let left_dists = Tensor::from_slice(&[1.0f32, 2.0]).view([1, 1, 2]);
        let mid_dists = Tensor::from_slice(&[3.0f32, 4.0]).view([1, 1, 2]);
        let right_dists = Tensor::from_slice(&[5.0f32, 6.0]).view([1, 1, 2]);

        let left = DistsMask::new(left_mask, Some(left_dists), true, None)?;
        let mid = DistsMask::new(mid_mask, Some(mid_dists), true, None)?;
        let right = DistsMask::new(right_mask, Some(right_dists), true, None)?;

        let merged = left.merge_batch_1_many(vec![mid, right])?;
        let (mask, dists) = merged.unpack()?;
        let dists = dists.expect("dists");

        assert_eq!(mask.size(), vec![1, 1, 6]);
        assert_eq!(dists.size(), vec![1, 1, 6]);
        assert_eq!(mask_values(&mask), vec![1, 0, 0, 1, 1, 1]);
        assert_eq!(dists_values(&dists), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        Ok(())
    }

    #[test]
    fn test_merge_batch_1_many_without_dists() -> Result<(), ThoError> {
        reset_print_options();
        let left_mask = Tensor::from_slice(&[0.0f32]).view([1, 1, 1]).lt(0.5);
        let mid_mask = Tensor::from_slice(&[1.0f32, 0.0]).view([1, 1, 2]).lt(0.5);
        let right_mask = Tensor::from_slice(&[1.0f32]).view([1, 1, 1]).lt(0.5);

        let left = DistsMask::new(left_mask, None, false, None)?;
        let mid = DistsMask::new(mid_mask, None, false, None)?;
        let right = DistsMask::new(right_mask, None, false, None)?;

        let merged = left.merge_batch_1_many(vec![mid, right])?;
        let (mask, dists) = merged.unpack()?;

        debug!("mask=\n{mask}");

        assert!(dists.is_none());
        assert_eq!(mask.size(), vec![1, 1, 4]);
        assert_eq!(mask_values(&mask), vec![1, 0, 1, 0]);

        Ok(())
    }

    #[test]
    fn test_merge_batch_1_many_dists_mismatch() -> Result<(), ThoError> {
        let left_mask = Tensor::from_slice(&[0.0f32]).view([1, 1, 1]).lt(0.5);
        let mid_mask = Tensor::from_slice(&[1.0f32]).view([1, 1, 1]).lt(0.5);
        let left_dists = Tensor::from_slice(&[1.0f32]).view([1, 1, 1]);

        let left = DistsMask::new(left_mask, Some(left_dists), false, None)?;
        let mid = DistsMask::new(mid_mask, None, false, None)?;

        let err = left
            .merge_batch_1_many(vec![mid])
            .expect_err("expected dists mismatch");
        match err.what() {
            Cause::SomethingElse(msg) => {
                assert_eq!(msg, "DistsMask merge: dists mismatch");
            }
            other => panic!("unexpected error: {other:?}"),
        }

        Ok(())
    }

    #[test]
    fn test_merge_batch_1_many_pack_mismatch() -> Result<(), ThoError> {
        let left_mask = Tensor::from_slice(&[0.0f32]).view([1, 1, 1]).lt(0.5);
        let mid_mask = Tensor::from_slice(&[1.0f32]).view([1, 1, 1]).lt(0.5);

        let left = DistsMask::new(left_mask, None, true, None)?;
        let mid = DistsMask::new(mid_mask, None, false, None)?;

        let err = left
            .merge_batch_1_many(vec![mid])
            .expect_err("expected pack mismatch");
        match err.what() {
            Cause::SomethingElse(msg) => {
                assert_eq!(msg, "DistsMask merge: pack mismatch");
            }
            other => panic!("unexpected error: {other:?}"),
        }

        Ok(())
    }
}
