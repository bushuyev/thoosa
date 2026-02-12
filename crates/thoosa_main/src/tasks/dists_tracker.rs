use std::collections::HashMap;
use crate::tasks::cross_batch_clc::{CrossClcResultMerge, CrossUnfoldClcOutput};
use log::{debug, info, trace};
use parking_lot::RwLock;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use tch::{Kind, Tensor};
use futures::{future::BoxFuture, FutureExt};
use tch::Device;
use tch::Device::{Cpu, Cuda};
use tch::Kind::{Bool, Uint8};
use thoosa_common::copy_ctx::CopyCtx;
use thoosa_common::cuda_utils::{pack_bits_bool, unpack_bits_u8};
use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
use thoosa_common::loader::rayon_pool;
use thoosa_common::tho_device::ThoDevice;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::traits::{Data, WithId};
use thoosa_common::ws::Workspace;
use thoosa_motifs::find::{FixedFindMotifsConfig, SimpleMotifTracker};
use thoosa_shared::ThoError;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::runner::common::runnable::{Stage, SyncStage};
use thoosa_shared::runner::common::RunCtx;
use thoosa_shims::stream::ThoCudaStream;
use thoosa_shims::stream::ThoStreamGuard;
use thoosa_store::fs_store::FileMotifTrackerStore;
use thoosa_store::{SlicesMatch, SqlMotifTrackerStore, Store};

// DistsMask is meant for one-way GPU->CPU copies.
#[derive(Debug)]
pub struct DistsMask {
    mask: Tensor,
    dists: Option<Tensor>,
    mask_type: DistsMaskType,
    stream_to_sync: Option<ThoCudaStream>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistsMaskType {
    AsIs,
    MaskPacked(Vec<i64>, Option<Vec<i64>>),
    MaskIndexes(Vec<i64>),
}

#[derive(Debug)]
enum DistsPtr {
    F32(*const f32),
    F64(*const f64),
}

impl DistsPtr {
    fn value(&self, idx: usize) -> f64 {
        match self {
            // Safety: ptr comes from a contiguous CPU Tensor that is owned by the iterator
            // (kept alive for the entire iteration), and idx is bounds-checked by the caller.
            // Safe alternatives like Tensor::iter or per-element indexing are much slower and/or
            // allocate; we intentionally use raw pointers for a zero-copy, tight loop.
            DistsPtr::F32(ptr) => unsafe { *ptr.add(idx) as f64 },
            DistsPtr::F64(ptr) => unsafe { *ptr.add(idx) },
        }
    }
}

#[derive(Debug)]
pub(crate) struct DenseMaskMatchIter {
    mask: Tensor,
    dists: Option<Tensor>,
    mask_ptr: *const u8,
    dists_ptr: Option<DistsPtr>,
    index: u64,
    total: u64,
    stride_0: u64,
    dim_2: u64,
}

impl DenseMaskMatchIter {
    fn new(mask: Tensor, dists: Option<Tensor>, mask_size: &[i64]) -> Self {
        let dim_1 = mask_size[1] as u64;
        let dim_2 = mask_size[2] as u64;
        let stride_0 = dim_1 * dim_2;
        let total = (mask_size[0] as u64) * stride_0;
        // We keep `mask` owned by the iterator, so the raw pointer remains valid
        // for the whole iteration. Using raw pointers avoids per-element Tensor
        // indexing, which is significantly slower for large masks.
        let mask_ptr = mask.data_ptr() as *const u8;
        let dists_ptr = dists.as_ref().map(dists_ptr);
        Self {
            mask,
            dists,
            mask_ptr,
            dists_ptr,
            index: 0,
            total,
            stride_0,
            dim_2,
        }
    }
}

impl Iterator for DenseMaskMatchIter {
    type Item = (u32, u32, u32, f64);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.total {
            let idx = self.index as usize;
            // Safety: idx < total (checked above), mask_ptr points to contiguous CPU
            // data owned by `self.mask`, and we only read from it.
            // Safe alternatives would either allocate or call into libtorch per element.
            let mask_val = unsafe { *self.mask_ptr.add(idx) != 0 };
            let flat = self.index;
            self.index += 1;
            if !mask_val {
                continue;
            }

            let i_0 = flat / self.stride_0;
            let rem = flat - i_0 * self.stride_0;
            let i_1 = rem / self.dim_2;
            let i_2 = rem - i_1 * self.dim_2;

            let dist = self
                .dists_ptr
                .as_ref()
                .map(|ptr| ptr.value(idx))
                .unwrap_or(0.0);

            return Some((
                u32::try_from(i_0).expect("DenseMaskMatchIter: channel overflow u32"),
                u32::try_from(i_1).expect("DenseMaskMatchIter: i_0 overflow u32"),
                u32::try_from(i_2).expect("DenseMaskMatchIter: i_1 overflow u32"),
                dist,
            ));
        }
        None
    }
}

#[derive(Debug)]
pub(crate) struct PackedMaskMatchIter {
    mask: Tensor,
    dists: Option<Tensor>,
    bytes_ptr: *const u8,
    dists_ptr: Option<DistsPtr>,
    bit_index: u64,
    total_bits: u64,
    stride_0: u64,
    dim_2: u64,
}

impl PackedMaskMatchIter {
    fn new(mask: Tensor, dists: Option<Tensor>, mask_size: &[i64]) -> Self {
        let dim_1 = mask_size[1] as u64;
        let dim_2 = mask_size[2] as u64;
        let stride_0 = dim_1 * dim_2;
        let total_bits = (mask_size[0] as u64) * stride_0;
        // Packed masks are stored as Uint8 bytes. We keep the Tensor alive and
        // read raw bytes to avoid unpacking into a full Bool tensor on CPU.
        let bytes_ptr = mask.data_ptr() as *const u8;
        let dists_ptr = dists.as_ref().map(dists_ptr);
        Self {
            mask,
            dists,
            bytes_ptr,
            dists_ptr,
            bit_index: 0,
            total_bits,
            stride_0,
            dim_2,
        }
    }
}

impl Iterator for PackedMaskMatchIter {
    type Item = (u32, u32, u32, f64);

    fn next(&mut self) -> Option<Self::Item> {
        while self.bit_index < self.total_bits {
            let idx = self.bit_index as usize;
            let byte_idx = (self.bit_index / 8) as usize;
            let bit_in_byte = (self.bit_index % 8) as u8;
            // Safety: byte_idx is derived from bit_index < total_bits, and the
            // packed mask Tensor is contiguous and owned by the iterator.
            // This avoids the cost of unpacking all bits up front.
            let byte = unsafe { *self.bytes_ptr.add(byte_idx) };
            let is_set = (byte & (0x80 >> bit_in_byte)) != 0;
            let flat = self.bit_index;
            self.bit_index += 1;
            if !is_set {
                continue;
            }

            let i_0 = flat / self.stride_0;
            let rem = flat - i_0 * self.stride_0;
            let i_1 = rem / self.dim_2;
            let i_2 = rem - i_1 * self.dim_2;

            let dist = self
                .dists_ptr
                .as_ref()
                .map(|ptr| ptr.value(idx))
                .unwrap_or(0.0);

            return Some((
                u32::try_from(i_0).expect("PackedMaskMatchIter: channel overflow u32"),
                u32::try_from(i_1).expect("PackedMaskMatchIter: i_0 overflow u32"),
                u32::try_from(i_2).expect("PackedMaskMatchIter: i_1 overflow u32"),
                dist,
            ));
        }
        None
    }
}

#[derive(Debug)]
struct TrueCoords3dIter {
    flat_ptr: *const i64,
    len: usize,
    pos: usize,
    d1d2: i64,
    d2: i64,
}

impl Iterator for TrueCoords3dIter {
    type Item = (u32, u32, u32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len {
            return None;
        }
        // Safety: flat_ptr comes from a contiguous Int64 Tensor held by the
        // iterator; pos < len is enforced above. Using raw pointers avoids
        // converting the tensor to a Vec or using per-element accessors.
        let v = unsafe { *self.flat_ptr.add(self.pos) };
        self.pos += 1;
        assert!(v >= 0, "TrueCoords3dIter: negative index {v}");

        let i_0 = v / self.d1d2;
        let rem = v - i_0 * self.d1d2;
        let i_1 = rem / self.d2;
        let i_2 = rem - i_1 * self.d2;

        Some((
            u32::try_from(i_0).expect("TrueCoords3dIter: i_0 overflow u32"),
            u32::try_from(i_1).expect("TrueCoords3dIter: i_1 overflow u32"),
            u32::try_from(i_2).expect("TrueCoords3dIter: i_2 overflow u32"),
        ))
    }
}

impl FlatIndicesMatchIter {
    fn new(indices: Tensor, dists: Option<Tensor>, mask_size: &[i64]) -> Self {
        let d1d2 = mask_size[1].saturating_mul(mask_size[2]);
        let d2 = mask_size[2];
        assert!(d1d2 > 0 && d2 > 0, "FlatIndicesMatchIter: invalid mask size");
        let coords = TrueCoords3dIter {
            // Indices are stored as a flat Int64 tensor (computed on GPU, moved
            // to CPU). We keep the tensor alive and iterate by raw pointer for
            // speed and to avoid extra copies.
            flat_ptr: indices.data_ptr() as *const i64,
            len: indices.numel() as usize,
            pos: 0,
            d1d2,
            d2,
        };
        let dists_ptr = dists.as_ref().map(dists_ptr);
        Self {
            indices,
            dists,
            coords,
            dists_ptr,
            dist_pos: 0,
        }
    }
}

#[derive(Debug)]
pub(crate) struct FlatIndicesMatchIter {
    indices: Tensor,
    dists: Option<Tensor>,
    coords: TrueCoords3dIter,
    dists_ptr: Option<DistsPtr>,
    dist_pos: usize,
}

impl Iterator for FlatIndicesMatchIter {
    type Item = (u32, u32, u32, f64);

    fn next(&mut self) -> Option<Self::Item> {
        let (c, i_0, i_1) = self.coords.next()?;
        let dist = self
            .dists_ptr
            .as_ref()
            .map(|ptr| {
                // Safety: dist_pos is incremented in lockstep with coords iteration,
                // and dists_ptr points to a contiguous CPU tensor owned by self.
                let v = ptr.value(self.dist_pos);
                self.dist_pos += 1;
                v
            })
            .unwrap_or(0.0);
        Some((c, i_0, i_1, dist))
    }
}

#[derive(Debug)]
pub(crate) struct TripletIndicesMatchIter {
    indices: Tensor,
    dists: Option<Tensor>,
    indices_ptr: *const i64,
    dists_ptr: Option<DistsPtr>,
    pos: usize,
    total: usize,
}

impl Iterator for TripletIndicesMatchIter {
    type Item = (u32, u32, u32, f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.total {
            return None;
        }

        let base = self.pos * 3;
        // Safety: indices_ptr is derived from a contiguous Int64 Tensor that we
        // own. base and base+2 are in-bounds because total == indices.size()[0].
        // Raw pointer access avoids per-element Tensor calls inside tight loops.
        let c = unsafe { *self.indices_ptr.add(base) };
        let i_0 = unsafe { *self.indices_ptr.add(base + 1) };
        let i_1 = unsafe { *self.indices_ptr.add(base + 2) };
        assert!(c >= 0 && i_0 >= 0 && i_1 >= 0, "TripletIndicesMatchIter: negative index");

        let dist = self
            .dists_ptr
            .as_ref()
            .map(|ptr| ptr.value(self.pos))
            .unwrap_or(0.0);

        self.pos += 1;

        Some((
            u32::try_from(c).expect("TripletIndicesMatchIter: channel overflow u32"),
            u32::try_from(i_0).expect("TripletIndicesMatchIter: i_0 overflow u32"),
            u32::try_from(i_1).expect("TripletIndicesMatchIter: i_1 overflow u32"),
            dist,
        ))
    }
}

impl TripletIndicesMatchIter {
    fn new(indices: Tensor, dists: Option<Tensor>) -> Self {
        let total = indices.size()[0] as usize;
        // We keep `indices` alive and read via raw pointer for efficiency. The
        // tensor is required to be contiguous and on CPU by construction.
        let indices_ptr = indices.data_ptr() as *const i64;
        let dists_ptr = dists.as_ref().map(dists_ptr);
        Self {
            indices,
            dists,
            indices_ptr,
            dists_ptr,
            pos: 0,
            total,
        }
    }
}

#[derive(Debug)]
pub(crate) enum IndicesMatchIter {
    Flat(FlatIndicesMatchIter),
    Triplet(TripletIndicesMatchIter),
}

impl Iterator for IndicesMatchIter {
    type Item = (u32, u32, u32, f64);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IndicesMatchIter::Flat(iter) => iter.next(),
            IndicesMatchIter::Triplet(iter) => iter.next(),
        }
    }
}

impl IndicesMatchIter {
    fn new(mask: Tensor, dists: Option<Tensor>, mask_size: &[i64]) -> Self {
        let size = mask.size();
        if size.len() == 1 {
            IndicesMatchIter::Flat(FlatIndicesMatchIter::new(mask, dists, mask_size))
        } else {
            assert!(
                size.len() == 2 && size[1] == 3,
                "IndicesMatchIter: expected Nx3 indices, got {size:?}"
            );
            IndicesMatchIter::Triplet(TripletIndicesMatchIter::new(mask, dists))
        }
    }
}

#[derive(Debug)]
pub(crate) enum DistsMaskMatchIter {
    Dense(DenseMaskMatchIter),
    Packed(PackedMaskMatchIter),
    Indices(IndicesMatchIter),
}

// Safety: the iterator only reads from CPU tensors that it owns. We never
// mutate the underlying memory, and the raw pointers stay valid for the life
// of the iterator because the owning Tensors are stored inside the enum.
// We still avoid sharing the same iterator across threads concurrently.
unsafe impl Send for DistsMaskMatchIter {}

impl Iterator for DistsMaskMatchIter {
    type Item = (u32, u32, u32, f64);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DistsMaskMatchIter::Dense(iter) => iter.next(),
            DistsMaskMatchIter::Packed(iter) => iter.next(),
            DistsMaskMatchIter::Indices(iter) => iter.next(),
        }
    }
}

fn dists_ptr(dists: &Tensor) -> DistsPtr {
    match dists.kind() {
        // These raw pointers are used only for read-only iteration on CPU.
        // Safe alternatives require per-element Tensor APIs or copying to Vec,
        // which are too costly in hot paths.
        Kind::Float => DistsPtr::F32(dists.data_ptr() as *const f32),
        Kind::Double => DistsPtr::F64(dists.data_ptr() as *const f64),
        other => panic!("DistsMask: unsupported dists kind {other:?}"),
    }
}


// Safety: DistsMask is treated as immutable data once constructed; it owns
// its Tensors and does not expose interior mutability. We only move or read
// it across threads, which is safe for the underlying CPU buffers.
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

    pub fn new(
        mask: Tensor,
        dists: Option<Tensor>,
        mask_type: DistsMaskType,
        stream_to_sync: Option<ThoCudaStream>,
    ) -> Result<Self, ThoError> {
        debug!("DistsMask::new: mask={mask:?} dists={dists:?} mask_type={mask_type:?}");

        if mask.kind() != Bool {
            return Err(ThoError::new("DistsMask::new: mask must be Bool"));
        }

        let mask_size = mask.size();
        assert_eq!(mask_size.len(), 3, "DistsMask::new: expected 3D mask, got {mask_size:?}");

        if let Some(dists) = dists.as_ref() {
            assert_eq!(dists.device(), mask.device(), "DistsMask::new: mask/dists device mismatch");
            assert_eq!(dists.size(), mask_size, "DistsMask::new: mask/dists size mismatch");
        }

        let _guard = if stream_to_sync.is_some() && mask.device().ne(&Cpu) {
            Some(ThoStreamGuard::set(
                stream_to_sync.as_ref().expect("guard stream missing"),
            ))
        } else {
            None
        };

        match mask_type {
            DistsMaskType::AsIs => {
                assert!(mask.device().eq(&Cpu), "DistsMask::new: AsIs expects CPU mask");
                if let Some(dists) = dists.as_ref() {
                    assert!(dists.device().eq(&Cpu), "DistsMask::new: AsIs expects CPU dists");
                }
                Ok(Self {
                    mask,
                    dists,
                    mask_type: DistsMaskType::AsIs,
                    stream_to_sync,
                })
            }
            DistsMaskType::MaskPacked(_, _) => {
                assert!(mask.device().ne(&Cpu), "DistsMask::new: MaskPacked expects non-CPU mask");
                if let Some(dists) = dists.as_ref() {
                    assert!(dists.device().ne(&Cpu), "DistsMask::new: MaskPacked expects non-CPU dists");
                }
                let dists_size = dists.as_ref().map(|t| t.size());
                let packed = pack_bits_bool(&mask)?;
                Ok(Self {
                    mask: packed,
                    dists,
                    mask_type: DistsMaskType::MaskPacked(
                        mask_size.clone(),
                        dists_size,
                    ),
                    stream_to_sync,
                })
            }
            DistsMaskType::MaskIndexes(_) => {
                assert!(mask.device().ne(&Cpu), "DistsMask::new: MaskIndexes expects non-CPU mask");
                if let Some(dists) = dists.as_ref() {
                    assert!(dists.device().ne(&Cpu), "DistsMask::new: MaskIndexes expects non-CPU dists");
                }
                let indices = mask.f_nonzero()?;
                debug!("DistsMask::new, MaskIndexes: indices={indices}");

                let indices = indices.f_to_kind(Kind::Int64)?;
                let weights = Tensor::from_slice(&[
                    mask_size[1].saturating_mul(mask_size[2]),
                    mask_size[2],
                    1i64,
                ])
                .f_to_device(mask.device())?
                .f_to_kind(Kind::Int64)?;

                let flat = (&indices * &weights).f_sum_dim_intlist(&[1i64][..], false, Kind::Int64)?;
                // CopyCtx assumes contiguous tensors for raw pointer copies.
                let flat = if flat.is_contiguous() {
                    flat
                } else {
                    flat.f_contiguous()?
                };
                let dists = match dists {
                    Some(dists) => {
                        let gathered = gather_dists_for_indices(&dists, &indices, &mask_size)?;
                        Some(if gathered.is_contiguous() {
                            gathered
                        } else {
                            gathered.f_contiguous()?
                        })
                    }
                    None => None,
                };
                Ok(Self {
                    mask: flat,
                    dists,
                    mask_type: DistsMaskType::MaskIndexes(mask_size),
                    stream_to_sync,
                })
            }
        }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="DistsMask::to_cpu"))]
    pub async fn to_cpu(self) -> Result<Self, ThoError>  {
        debug!("DistsMask::to_cpu: self.mask.device={:?}", self.mask.device());

        if let Some(stream) = self.stream_to_sync.as_ref() {
            stream.synchronize();
        }

        match self.mask_type {
            DistsMaskType::AsIs => {
                assert!(self.mask.device().eq(&Cpu), "DistsMask::to_cpu: AsIs expects CPU mask");

                if let Some(dists) = self.dists.as_ref() {
                    assert!(dists.device().eq(&Cpu), "DistsMask::to_cpu: AsIs expects CPU dists");
                }
                Ok(Self {
                    mask: self.mask,
                    dists: self.dists,
                    mask_type: DistsMaskType::AsIs,
                    stream_to_sync: None,
                })
            }
            mask_type @ (DistsMaskType::MaskPacked(_, _) | DistsMaskType::MaskIndexes(_)) => {

                let mask = if self.mask.device().eq(&Cpu) {
                    self.mask
                } else {
                    let mask = if self.mask.is_contiguous() {
                        self.mask
                    } else {
                        self.mask.f_contiguous()?
                    };
                    let device = ThoDevice::from(mask.device());
                    let kind = mask.kind();
                    CopyCtx::instance(device).move_t(mask, 0, kind).await?
                };
                let dists = match self.dists {
                    Some(dists) if dists.device().eq(&Cpu) => Some(dists),
                    Some(dists) => {
                        let dists = if dists.is_contiguous() {
                            dists
                        } else {
                            dists.f_contiguous()?
                        };
                        let kind = dists.kind();
                        let dists_device = ThoDevice::from(dists.device());
                        Some(CopyCtx::instance(dists_device).move_t(dists, 0, kind).await?)
                    }
                    None => None,
                };
                Ok(Self {
                    mask,
                    dists,
                    mask_type,
                    stream_to_sync: None,
                })
            }
        }
    }


    pub fn unpack(self) -> Result<(Tensor, Option<Tensor>), ThoError> {
        match self.mask_type {
            DistsMaskType::AsIs => Ok((self.mask, self.dists)),
            DistsMaskType::MaskPacked(mask_size, _) => {
                Ok((unpack_packed_mask(&self.mask, &mask_size)?, self.dists))
            }
            DistsMaskType::MaskIndexes(_) => Err(ThoError::new(
                "DistsMask::unpack: mask is stored as indices",
            )),
        }
    }

    //this is expected to be called after mask is copied to CPU
    pub fn into_indices_iter(self) -> Result<DistsMaskMatchIter, ThoError> {
        let DistsMask {
            mask,
            dists,
            mask_type,
            ..
        } = self;



        match mask_type {
            DistsMaskType::AsIs => {
                assert!(mask.device().eq(&Cpu), "DistsMask::into_indices_iter: expected CPU mask");
                let mask = if mask.is_contiguous() {
                    mask
                } else {
                    mask.f_contiguous()?
                };
                let mask_size = mask.size();
                assert_eq!(mask_size.len(), 3, "DistsMask::into_indices_iter: expected 3D mask, got {mask_size:?}");
                let dists = match dists {
                    Some(dists) => {
                        assert!(
                            dists.device().eq(&Cpu),
                            "DistsMask::into_indices_iter: dists must be on CPU"
                        );
                        let dists = if dists.is_contiguous() {
                            dists
                        } else {
                            dists.f_contiguous()?
                        };
                        Some(dists)
                    }
                    None => None,
                };
                Ok(DistsMaskMatchIter::Dense(DenseMaskMatchIter::new(mask, dists, &mask_size)))
            }
            DistsMaskType::MaskPacked(mask_size, _) => {
                assert!(mask.device().eq(&Cpu), "DistsMask::into_indices_iter: expected CPU packed mask");
                assert_eq!(mask.kind(), Uint8, "DistsMask::into_indices_iter: expected packed mask to be Uint8");
                assert_eq!(mask_size.len(), 3, "DistsMask::into_indices_iter: expected 3D mask, got {mask_size:?}");
                let mask = if mask.is_contiguous() {
                    mask
                } else {
                    mask.f_contiguous()?
                };
                let dists = match dists {
                    Some(dists) => {
                        assert!(dists.device().eq(&Cpu), "DistsMask::into_indices_iter: dists must be on CPU");
                        let dists = if dists.is_contiguous() {
                            dists
                        } else {
                            dists.f_contiguous()?
                        };
                        Some(dists)
                    }
                    None => None,
                };
                Ok(DistsMaskMatchIter::Packed(PackedMaskMatchIter::new(mask, dists, &mask_size)))
            }
            DistsMaskType::MaskIndexes(mask_size) => {
                assert!(mask.device().eq(&Cpu), "DistsMask::into_indices_iter: indices must be on CPU");
                assert_eq!(mask.kind(), Kind::Int64, "DistsMask::into_indices_iter: expected Int64 indices");
                let mask = if mask.is_contiguous() {
                    mask
                } else {
                    mask.f_contiguous()?
                };
                let indices_size = mask.size();
                assert_eq!(mask_size.len(), 3, "DistsMask::into_indices_iter: expected 3D mask, got {mask_size:?}");
                let indices_count = if indices_size.len() == 1 {
                    indices_size[0]
                } else {
                    assert!(
                        indices_size.len() == 2 && indices_size[1] == 3,
                        "DistsMask::into_indices_iter: expected Nx3 indices, got {indices_size:?}"
                    );
                    indices_size[0]
                };
                let dists = match dists {
                    Some(dists) => {
                        assert!(dists.device().eq(&Cpu), "DistsMask::into_indices_iter: dists must be on CPU");
                        let dists = if dists.is_contiguous() {
                            dists
                        } else {
                            dists.f_contiguous()?
                        };
                        let dists = dists.f_reshape([-1])?;
                        assert_eq!(
                            dists.numel() as i64,
                            indices_count,
                            "DistsMask::into_indices_iter: dists size mismatch"
                        );
                        Some(dists)
                    }
                    None => None,
                };
                Ok(DistsMaskMatchIter::Indices(IndicesMatchIter::new(mask, dists, &mask_size)))
            }
        }
    }

    pub fn mask_size_hint(&self) -> Result<Vec<i64>, ThoError> {
        match &self.mask_type {
            DistsMaskType::AsIs => Ok(self.mask.size()),
            DistsMaskType::MaskPacked(mask_size, _) => Ok(mask_size.clone()),
            DistsMaskType::MaskIndexes(mask_size) => Ok(mask_size.clone()),
        }
    }

    fn unpack_mask_cpu(&self) -> Result<Tensor, ThoError> {
        let mask = self.unpack_mask_device(&Cpu)?;
        assert!(mask.device().eq(&Cpu), "DistsMask::unpack_mask_cpu: expected CPU");
        Ok(mask)
    }

    fn unpack_mask_device(&self, device: &Device) -> Result<Tensor, ThoError> {
        match &self.mask_type {
            DistsMaskType::MaskIndexes(_) => Err(ThoError::new(
                "DistsMask::unpack_mask_device: mask is indices",
            )),
            DistsMaskType::MaskPacked(mask_size, _) => {
                let bits = unpack_packed_mask(&self.mask, mask_size)?;
                if &bits.device() == device {
                    Ok(bits)
                } else {
                    Ok(bits.f_to_device(*device)?)
                }
            }
            DistsMaskType::AsIs => {
                if &self.mask.device() == device {
                    Ok(self.mask.shallow_clone())
                } else {
                    Ok(self.mask.f_to_device(*device)?)
                }
            }
        }
    }

}

fn rebase_flat_indices_last_dim(
    flat: Tensor,
    size: &[i64],
    last_offset: i64,
    new_last: i64,
) -> Result<Tensor, ThoError> {
    if flat.numel() == 0 {
        return Ok(flat);
    }
    assert_eq!(flat.kind(), Kind::Int64, "DistsMask: expected Int64 flat indices");
    assert_eq!(size.len(), 3, "DistsMask: expected 3D size, got {size:?}");

    let device = flat.device();
    let dim_1 = size[1];
    let dim_2 = size[2];
    let d1d2 = dim_1.saturating_mul(dim_2);

    let d1d2_t = Tensor::from(d1d2).f_to_device(device)?;
    let dim_2_t = Tensor::from(dim_2).f_to_device(device)?;
    let dim_1_t = Tensor::from(dim_1).f_to_device(device)?;
    let new_last_t = Tensor::from(new_last).f_to_device(device)?;
    let offset_t = Tensor::from(last_offset).f_to_device(device)?;

    let i_0 = flat.f_floor_divide(&d1d2_t)?;
    let rem = &flat - &i_0 * &d1d2_t;
    let i_1 = rem.f_floor_divide(&dim_2_t)?;
    let i_2 = rem - &i_1 * &dim_2_t;
    let i_2 = i_2 + offset_t;

    let row = &i_0 * &dim_1_t + &i_1;
    Ok((&row * new_last_t + i_2).f_to_kind(Kind::Int64)?)
}

fn unpack_packed_mask(mask: &Tensor, mask_size: &[i64]) -> Result<Tensor, ThoError> {
    assert!(!mask_size.is_empty(), "DistsMask::unpack_packed_mask: empty mask size");

    let total_bits = mask_size
        .iter()
        .fold(1i64, |a, v| a.saturating_mul(*v));
    Ok(unpack_bits_u8(mask, Some(total_bits))
        .f_to_kind(Bool)?
        .f_view(mask_size)?)
}

fn gather_dists_for_indices(
    dists: &Tensor,
    indices: &Tensor,
    mask_size: &[i64],
) -> Result<Tensor, ThoError> {
    assert_eq!(mask_size.len(), 3, "DistsMask::gather_dists_for_indices: expected 3D mask, got {mask_size:?}");
    assert_eq!(dists.device(), indices.device(), "DistsMask::gather_dists_for_indices: dists/indices device mismatch");

    let indices_size = indices.size();

    assert!(indices_size.len() == 2 && indices_size[1] == 3, "DistsMask::gather_dists_for_indices: expected Nx3 indices, got {indices_size:?}");

    if indices_size[0] == 0 {
        return Ok(Tensor::zeros([0], (dists.kind(), dists.device())));
    }

    let indices = indices.f_contiguous()?.f_to_kind(Kind::Int64)?;
    let dists_flat = dists.f_contiguous()?.f_reshape([-1])?;

    let stride_0 = mask_size[1] * mask_size[2];
    let stride_1 = mask_size[2];

    let c = indices.f_select(1, 0)?;
    let i_0 = indices.f_select(1, 1)?;
    let i_1 = indices.f_select(1, 2)?;

    let stride_0_t = Tensor::from(stride_0).f_to_device(dists.device())?;
    let stride_1_t = Tensor::from(stride_1).f_to_device(dists.device())?;

    let lin = (&c * stride_0_t + &i_0 * stride_1_t + &i_1).f_to_kind(Kind::Int64)?;
    Ok(dists_flat.f_gather(0, &lin, false)?)
}

impl CrossClcResultMerge for DistsMask {

    fn to_cpu(self) -> BoxFuture<'static, Result<Self, ThoError>> {
        async move { DistsMask::to_cpu(self).await }.boxed()
    }

    //gpt generated
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CrossClcResultMerge for DistsMask::merge_batch_1"))]
    fn merge_batch_1(self, other: Self) -> Result<Self, ThoError> {
        let left = self;
        let right = other;

        if let Some(stream) = left.stream_to_sync.as_ref() {
            stream.synchronize();
        }
        if let Some(stream) = right.stream_to_sync.as_ref() {
            stream.synchronize();
        }

        let DistsMask {
            mask: left_mask,
            dists: left_dists,
            mask_type: left_type,
            ..
        } = left;
        let DistsMask {
            mask: right_mask,
            dists: right_dists,
            mask_type: right_type,
            ..
        } = right;

        match (left_type, right_type) {
            (DistsMaskType::AsIs, DistsMaskType::AsIs) => {
                let mask = Tensor::f_concat(&[left_mask, right_mask], 2)?;
                let dists = match (left_dists, right_dists) {
                    (Some(left), Some(right)) => Some(Tensor::f_concat(&[left, right], 2)?),
                    (None, None) => None,
                    _ => return Err(ThoError::new("DistsMask merge: dists mismatch")),
                };

                Ok(DistsMask {
                    mask,
                    dists,
                    mask_type: DistsMaskType::AsIs,
                    stream_to_sync: None,
                })
            }
            (
                DistsMaskType::MaskPacked(left_sizes, left_dists_size),
                DistsMaskType::MaskPacked(right_sizes, right_dists_size),
            ) => {
                assert!(left_sizes.len() == right_sizes.len() && !left_sizes.is_empty(), "DistsMask merge: size mismatch");

                let last_dim = left_sizes.len() - 1;
                assert_eq!(left_sizes[..last_dim], right_sizes[..last_dim], "DistsMask merge: size mismatch");

                let left_last = left_sizes[last_dim];
                let right_last = right_sizes[last_dim];
                let rows: i64 = left_sizes[..last_dim]
                    .iter()
                    .fold(1i64, |a, v| a.saturating_mul(*v));

                let total_left_bits: i64 =
                    left_sizes.iter().fold(1i64, |a, v| a.saturating_mul(*v));
                let total_right_bits: i64 =
                    right_sizes.iter().fold(1i64, |a, v| a.saturating_mul(*v));

                let packed_mask = if left_last % 8 == 0 && right_last % 8 == 0 {
                    let left_bytes = left_last / 8;
                    let right_bytes = right_last / 8;
                    let left_view = left_mask.f_view([rows, left_bytes])?;
                    let right_view = right_mask.f_view([rows, right_bytes])?;
                    Tensor::f_concat(&[left_view, right_view], 1)?
                        .f_reshape([-1])?
                        .f_contiguous()?
                } else {
                    let left_bits = unpack_bits_u8(&left_mask, Some(total_left_bits))
                        .f_to_kind(Bool)?
                        .f_view([rows, left_last])?;
                    let right_bits = unpack_bits_u8(&right_mask, Some(total_right_bits))
                        .f_to_kind(Bool)?
                        .f_view([rows, right_last])?;
                    let merged_bits = Tensor::f_concat(&[left_bits, right_bits], 1)?;
                    pack_bits_bool(&merged_bits.f_reshape([-1])?)?
                };

                let dists = match (left_dists, right_dists) {
                    (Some(left), Some(right)) => {
                        assert!(left_dists_size.is_some(), "DistsMask merge: dists size mismatch");
                        assert!(right_dists_size.is_some(), "DistsMask merge: dists size mismatch");

                        let left_size = left_dists_size.expect("DistsMask merge: dists size mismatch");
                        let right_size = right_dists_size.expect("DistsMask merge: dists size mismatch");

                        assert!(left_size.len() == right_size.len() && !left_size.is_empty(), "DistsMask merge: dists size mismatch");
                        let dists_last = left_size.len() - 1;

                        assert_eq!(left_size[..dists_last], right_size[..dists_last], "DistsMask merge: dists size mismatch");
                        Some(Tensor::f_concat(&[left, right], dists_last as i64)?)
                    }
                    (None, None) => None,
                    _ => return Err(ThoError::new("DistsMask merge: dists mismatch")),
                };

                let mut merged_sizes = left_sizes;
                merged_sizes[last_dim] = left_last + right_last;
                let merged_dists_sizes = dists.as_ref().map(|t| t.size());

                Ok(DistsMask {
                    mask: packed_mask,
                    dists,
                    mask_type: DistsMaskType::MaskPacked(merged_sizes, merged_dists_sizes),
                    stream_to_sync: None,
                })
            }
            (DistsMaskType::MaskIndexes(left_size), DistsMaskType::MaskIndexes(right_size)) => {
                assert!(left_size.len() == right_size.len() && !left_size.is_empty(), "DistsMask merge: size mismatch");

                let last_dim = left_size.len() - 1;

                assert_eq!(left_size[..last_dim], right_size[..last_dim], "DistsMask merge: size mismatch");
                assert_eq!(left_mask.device(), right_mask.device(), "DistsMask merge: device mismatch");

                let left_indices_size = left_mask.size();
                let right_indices_size = right_mask.size();

                assert!(
                    (left_indices_size.len() == 1 || (left_indices_size.len() == 2 && left_indices_size[1] == 3))
                        && left_indices_size.len() == right_indices_size.len(),
                    "DistsMask merge: indices shape mismatch"
                );
                assert!(
                    right_indices_size.len() == 1 || (right_indices_size.len() == 2 && right_indices_size[1] == 3),
                    "DistsMask merge: indices shape mismatch"
                );
                let left_count = left_indices_size[0];
                let right_count = right_indices_size[0];

                let new_last = left_size[last_dim] + right_size[last_dim];
                let (left_indices, right_indices) = if left_indices_size.len() == 1 {
                    let left_indices = rebase_flat_indices_last_dim(
                        left_mask,
                        &left_size,
                        0,
                        new_last,
                    )?;
                    let right_indices = rebase_flat_indices_last_dim(
                        right_mask,
                        &right_size,
                        left_size[last_dim],
                        new_last,
                    )?;
                    (left_indices, right_indices)
                } else {
                    let mut right_indices = right_mask;
                    if right_count > 0 {
                        let offset = Tensor::from(left_size[last_dim])
                            .f_to_device(right_indices.device())?;
                        let _ = right_indices.narrow(1, 2, 1).f_add_(&offset)?;
                    }
                    (left_mask, right_indices)
                };

                let indices = if left_count == 0 {
                    right_indices
                } else if right_count == 0 {
                    left_indices
                } else {
                    Tensor::f_concat(&[left_indices, right_indices], 0)?
                };

                let dists = match (left_dists, right_dists) {
                    (Some(left), Some(right)) => {
                        assert_eq!(left.device(), right.device(), "DistsMask merge: dists device mismatch");
                        assert!(left.numel() as i64 == left_count && right.numel() as i64 == right_count, "DistsMask merge: dists size mismatch");
                        Some(Tensor::f_concat(&[left, right], 0)?)
                    }
                    (None, None) => None,
                    _ => return Err(ThoError::new("DistsMask merge: dists mismatch")),
                };

                let mut merged_size = left_size;
                merged_size[last_dim] = new_last;

                Ok(DistsMask {
                    mask: indices,
                    dists,
                    mask_type: DistsMaskType::MaskIndexes(merged_size),
                    stream_to_sync: None,
                })
            }
            _ => Err(ThoError::new("DistsMask merge: type mismatch")),
        }
    }

    //gpt generated
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CrossClcResultMerge for DistsMask::merge_batch_1_many"))]
    fn merge_batch_1_many(self, others: Vec<Self>) -> Result<Self, ThoError> {
        if others.is_empty() {
            return Ok(self);
        }

        let mut parts = Vec::with_capacity(others.len() + 1);
        parts.push(self);
        parts.extend(others);

        for part in &parts {
            if let Some(stream) = part.stream_to_sync.as_ref() {
                stream.synchronize();
            }
        }

        let first_type = parts[0].mask_type.clone();
        match first_type {
            DistsMaskType::AsIs => {
                if parts
                    .iter()
                    .any(|p| !matches!(p.mask_type, DistsMaskType::AsIs))
                {
                    return Err(ThoError::new("DistsMask merge: type mismatch"));
                }

                let first_size = parts[0].mask.size();
                assert!(!first_size.is_empty(), "DistsMask merge: size mismatch");

                let last_dim = first_size.len() - 1;
                let device = parts[0].mask.device();
                let mut total_last = first_size[last_dim];

                for part in parts.iter().skip(1) {
                    let size = part.mask.size();
                    assert!(
                        size.len() == first_size.len() && size[..last_dim] == first_size[..last_dim],
                        "DistsMask merge: size mismatch"
                    );
                    assert_eq!(part.mask.device(), device, "DistsMask merge: device mismatch");
                    total_last += size[last_dim];
                }

                let mut merged_mask_sizes = first_size;
                merged_mask_sizes[last_dim] = total_last;
                let mut mask_out = Tensor::zeros(merged_mask_sizes.as_slice(), (Bool, device));

                let mut offset = 0;
                for part in parts.iter() {
                    let len = part.mask.size()[last_dim];
                    mask_out
                        .narrow(last_dim as i64, offset, len)
                        .copy_(&part.mask);
                    offset += len;
                }

                let has_dists = parts[0].dists.is_some();
                if parts.iter().any(|p| p.dists.is_some() != has_dists) {
                    return Err(ThoError::new("DistsMask merge: dists mismatch"));
                }

                let dists = if has_dists {
                    let first_dists = parts[0].dists.as_ref().unwrap();
                    let dists_device = first_dists.device();
                    let dists_kind = first_dists.kind();
                    let dists_size = first_dists.size();
                    assert!(!dists_size.is_empty(), "DistsMask merge: dists size mismatch");

                    let dists_last_dim = dists_size.len() - 1;
                    let mut total_dists_last = dists_size[dists_last_dim];

                    for part in parts.iter().skip(1) {
                        let dists = part.dists.as_ref().unwrap();
                        let size = dists.size();

                        assert!(size.len() == dists_size.len() && size[..dists_last_dim] == dists_size[..dists_last_dim], "DistsMask merge: dists size mismatch");
                        assert_eq!(dists.device(), dists_device, "DistsMask merge: dists device mismatch");

                        total_dists_last += size[dists_last_dim];
                    }

                    let mut merged_dists_sizes = dists_size;
                    merged_dists_sizes[dists_last_dim] = total_dists_last;
                    let mut dists_out =
                        Tensor::zeros(merged_dists_sizes.as_slice(), (dists_kind, dists_device));

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

                Ok(DistsMask {
                    mask: mask_out,
                    dists,
                    mask_type: DistsMaskType::AsIs,
                    stream_to_sync: None,
                })
            }
            DistsMaskType::MaskPacked(base_sizes, base_dists_size) => {
                if parts.iter().any(|p| {
                    !matches!(p.mask_type, DistsMaskType::MaskPacked(_, _))
                }) {
                    return Err(ThoError::new("DistsMask merge: type mismatch"));
                }

                assert!(
                    !base_sizes.is_empty(),
                    "DistsMask merge: size mismatch"
                );
                let last_dim = base_sizes.len() - 1;
                let device = parts[0].mask.device();

                let mut total_last = base_sizes[last_dim];
                let mut all_aligned = base_sizes[last_dim] % 8 == 0;

                for part in parts.iter().skip(1) {
                    let sizes = match &part.mask_type {
                        DistsMaskType::MaskPacked(sizes, _) => sizes,
                        _ => return Err(ThoError::new("DistsMask merge: type mismatch")),
                    };

                    assert!(sizes.len() == base_sizes.len() && sizes[..last_dim] == base_sizes[..last_dim], "DistsMask merge: size mismatch");
                    assert_eq!(part.mask.device(), device, "DistsMask merge: device mismatch");

                    total_last += sizes[last_dim];
                    all_aligned &= sizes[last_dim] % 8 == 0;
                }

                let rows: i64 = base_sizes[..last_dim]
                    .iter()
                    .fold(1i64, |a, v| a.saturating_mul(*v));

                let packed_mask = if all_aligned {
                    let total_bytes = total_last / 8;
                    let mut out = Tensor::zeros([rows, total_bytes], (Uint8, device));
                    let mut byte_offset = 0;
                    for part in parts.iter() {
                        let sizes = match &part.mask_type {
                            DistsMaskType::MaskPacked(sizes, _) => sizes,
                            _ => return Err(ThoError::new("DistsMask merge: type mismatch")),
                        };
                        let bytes = sizes[last_dim] / 8;
                        let part_view = part.mask.f_view([rows, bytes])?;
                        out.narrow(1, byte_offset, bytes).copy_(&part_view);
                        byte_offset += bytes;
                    }
                    out.f_reshape([-1])?
                } else {
                    let mut out_bits = Tensor::zeros([rows, total_last], (Bool, device));
                    let mut offset = 0;
                    for part in parts.iter() {
                        let sizes = match &part.mask_type {
                            DistsMaskType::MaskPacked(sizes, _) => sizes,
                            _ => return Err(ThoError::new("DistsMask merge: type mismatch")),
                        };
                        let part_last = sizes[last_dim];
                        let total_bits =
                            sizes.iter().fold(1i64, |a, v| a.saturating_mul(*v));
                        let bits = unpack_bits_u8(&part.mask, Some(total_bits))
                            .f_to_kind(Bool)?
                            .f_view([rows, part_last])?;
                        out_bits.narrow(1, offset, part_last).copy_(&bits);
                        offset += part_last;
                    }
                    pack_bits_bool(&out_bits.f_reshape([-1])?)?
                };

                let has_dists = parts[0].dists.is_some();
                if parts.iter().any(|p| p.dists.is_some() != has_dists) {
                    return Err(ThoError::new("DistsMask merge: dists mismatch"));
                }

                let (dists, merged_dists_sizes) = if has_dists {
                    assert!(base_dists_size.is_some(), "DistsMask merge: dists size mismatch");

                    let dists_size = base_dists_size.expect("DistsMask merge: dists size mismatch");
                    assert!(!dists_size.is_empty(), "DistsMask merge: dists size mismatch");

                    let dists_last_dim = dists_size.len() - 1;
                    let mut total_dists_last = 0;
                    let mut dists_device = None;
                    let mut dists_kind = None;

                    for part in parts.iter() {
                        let sizes = match &part.mask_type {
                            DistsMaskType::MaskPacked(_, dists_size) => dists_size,
                            _ => return Err(ThoError::new("DistsMask merge: type mismatch")),
                        };
                        assert!(sizes.is_some(), "DistsMask merge: dists size mismatch");
                        let size = sizes
                            .as_ref()
                            .expect("DistsMask merge: dists size mismatch");
                        assert!(size.len() == dists_size.len() && size[..dists_last_dim] == dists_size[..dists_last_dim], "DistsMask merge: dists size mismatch");
                        let dists = part.dists.as_ref().unwrap();

                        if let Some(dev) = dists_device {
                            assert_eq!(dists.device(), dev, "DistsMask merge: dists device mismatch");
                        } else {
                            dists_device = Some(dists.device());
                        }
                        if let Some(kind) = dists_kind {
                            if dists.kind() != kind {
                                return Err(ThoError::new("DistsMask merge: dists size mismatch"));
                            }
                        } else {
                            dists_kind = Some(dists.kind());
                        }
                        total_dists_last += size[dists_last_dim];
                    }

                    let mut merged_dists_sizes = dists_size;
                    merged_dists_sizes[dists_last_dim] = total_dists_last;
                    let device = dists_device.unwrap();
                    let kind = dists_kind.unwrap();
                    let mut dists_out =
                        Tensor::zeros(merged_dists_sizes.as_slice(), (kind, device));

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

                let mut merged_sizes = base_sizes;
                merged_sizes[last_dim] = total_last;

                Ok(DistsMask {
                    mask: packed_mask,
                    dists,
                    mask_type: DistsMaskType::MaskPacked(merged_sizes, merged_dists_sizes),
                    stream_to_sync: None,
                })
            }
            DistsMaskType::MaskIndexes(_) => {
                let base_size = match &parts[0].mask_type {
                    DistsMaskType::MaskIndexes(size) => size.clone(),
                    _ => return Err(ThoError::new("DistsMask merge: type mismatch")),
                };

                let last_dim = base_size.len() - 1;
                let mut total_last = 0i64;
                for part in parts.iter() {
                    let size = match &part.mask_type {
                        DistsMaskType::MaskIndexes(size) => size,
                        _ => return Err(ThoError::new("DistsMask merge: type mismatch")),
                    };
                    assert!(
                        size.len() == base_size.len() && size[..last_dim] == base_size[..last_dim],
                        "DistsMask merge: size mismatch"
                    );
                    total_last += size[last_dim];
                }

                let mut parts_iter = parts.into_iter();
                let first = parts_iter
                    .next()
                    .ok_or_else(|| ThoError::new("DistsMask merge: missing left part"))?;
                let DistsMask {
                    mask: first_mask,
                    dists: first_dists,
                    mask_type: first_type,
                    ..
                } = first;

                assert!(!base_size.is_empty(), "DistsMask merge: size mismatch");
                let device = first_mask.device();
                let first_indices_size = first_mask.size();

                let indices_rank = first_indices_size.len();
                assert!(
                    indices_rank == 1 || (indices_rank == 2 && first_indices_size[1] == 3),
                    "DistsMask merge: indices shape mismatch"
                );
                let first_count = first_indices_size[0];

                let has_dists = first_dists.is_some();
                if let Some(dists) = first_dists.as_ref() {
                    assert_eq!(dists.device(), device, "DistsMask merge: dists device mismatch");
                    assert_eq!(dists.numel() as i64, first_count, "DistsMask merge: dists size mismatch");
                }

                let mut indices_parts = Vec::new();
                let mut offset = base_size[last_dim];
                let mut indices = if indices_rank == 1 {
                    rebase_flat_indices_last_dim(first_mask, &base_size, 0, total_last)?
                } else {
                    first_mask
                };
                indices_parts.push(indices);
                let mut dists_parts = Vec::new();
                if let Some(dists) = first_dists {
                    dists_parts.push(dists);
                }

                for part in parts_iter {
                    let DistsMask {
                        mask,
                        dists,
                        mask_type,
                        ..
                    } = part;

                    let size = match mask_type {
                        DistsMaskType::MaskIndexes(size) => size,
                        _ => return Err(ThoError::new("DistsMask merge: type mismatch")),
                    };
                    assert!(size.len() == base_size.len() && size[..last_dim] == base_size[..last_dim], "DistsMask merge: size mismatch");
                    assert_eq!(mask.device(), device, "DistsMask merge: device mismatch");

                    let indices_size = mask.size();
                    assert!(
                        indices_size.len() == indices_rank
                            && (indices_rank == 1 || indices_size[1] == 3),
                        "DistsMask merge: indices shape mismatch"
                    );
                    let indices_count = indices_size[0];

                    let mut indices = if indices_rank == 1 {
                        rebase_flat_indices_last_dim(mask, &size, offset, total_last)?
                    } else {
                        mask
                    };
                    if indices_count > 0 && indices_rank == 2 {
                        let offset_t = Tensor::from(offset).f_to_device(device)?;
                        let _ = indices.narrow(1, 2, 1).f_add_(&offset_t)?;
                    }

                    if has_dists {
                        let dists = dists.ok_or_else(|| {
                            ThoError::new("DistsMask merge: dists mismatch")
                        })?;
                        assert_eq!(dists.device(), device, "DistsMask merge: dists device mismatch");
                        assert_eq!(dists.numel() as i64, indices_count, "DistsMask merge: dists size mismatch");
                        dists_parts.push(dists);
                    } else if dists.is_some() {
                        return Err(ThoError::new("DistsMask merge: dists mismatch"));
                    }

                    indices_parts.push(indices);
                    offset += size[last_dim];
                }

                let indices = if indices_parts.len() == 1 {
                    indices_parts.remove(0)
                } else {
                    Tensor::f_concat(&indices_parts, 0)?
                };
                let dists = if has_dists {
                    if dists_parts.len() == 1 {
                        Some(dists_parts.remove(0))
                    } else {
                        Some(Tensor::f_concat(&dists_parts, 0)?)
                    }
                } else {
                    None
                };

                let mut merged_size = base_size;
                merged_size[last_dim] = total_last;

                Ok(DistsMask {
                    mask: indices,
                    dists,
                    mask_type: DistsMaskType::MaskIndexes(merged_size),
                    stream_to_sync: None,
                })
            }
        }
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
        debug!("DistsTracker:new: fixed_config={fixed_config:?} channels={}", ws.read().channels());

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

        debug!("DistsTracker::build_groups: channel_trackers={channel_trackers:?}");

        let channels_groups_for_rayon = Arc::clone(&channels_groups);

        tokio::task::spawn_blocking(move || {
            use rayon::prelude::*;
            rayon_pool().install(|| {

                channel_trackers.par_iter().enumerate().try_for_each(|(channel, mt)| {
                    let groups_inserted = mt.write().build_groups(&store)?;
                    channels_groups_for_rayon.write().insert(channel, groups_inserted);
                    Ok::<_, ThoError>(())
                })
            })
        }).await??;

        let channels_groups_guard = channels_groups.read();
        info!("groups inserted: {:?}", (0..self.channel_trackers.len()).map(|c|channels_groups_guard.get(&c).map(|v|*v).ok_or(0)).collect::<Vec<_>>());
        Ok(())
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
        _ctx: Self::Ctx,
    ) -> Result<Self::Out, ThoError> {
        //TODO clone to resources ref
        let iter_0 = input.iter_0();
        let iter_1 = input.iter_1();

        let slices_0: Vec<_> = iter_0.collect();
        let slices_1: Vec<_> = iter_1.collect();

        debug!("DistsTracker::consume: slices_0={slices_0:?} slices_1={slices_1:?}");

        let clc_result = input.clc_result();
        let mask_size = clc_result.mask_size_hint()?;

        assert_eq!(mask_size.len(), 3, "DistsTracker::consume: expected 3D mask, got {mask_size:?}");
        assert!(mask_size[1] as usize == slices_0.len() && mask_size[2] as usize == slices_1.len(), "DistsTracker::consume: mask size mismatch mask={mask_size:?} slices_0={} slices_1={}", slices_0.len(), slices_1.len());
        assert_eq!(mask_size[0] as usize, self.channel_trackers.len(), "DistsTracker::consume: channel mismatch mask_channels={} trackers={}", mask_size[0], self.channel_trackers.len());

        let channel_trackers = self.channel_trackers.clone();

        debug!("DistsTracker::consume: channel_trackers.len={}", channel_trackers.len());
        tokio::task::spawn_blocking(move || {
            use rayon::prelude::*;
            rayon_pool().install(|| {
                debug!("DistsTracker::consume: in rayon pool");

                let mut match_indices = clc_result.into_indices_iter()?;
                let mut matches_by_channel = vec![Vec::<SlicesMatch>::new(); self.channel_trackers.len()];

                debug!("DistsTracker::consume: match_indices={match_indices:?}");

                match_indices.par_bridge().for_each(|(channel, i_0, i_1, dist) | {



                    let channel = channel as usize;
                    let i_0 = i_0 as usize;
                    let i_1 = i_1 as usize;

                    assert!(channel < matches_by_channel.len(), "DistsTracker::consume: channel index out of bounds {channel}");
                    assert!(i_0 < slices_0.len(), "DistsTracker::consume: mask index out of bounds (batch 0)");
                    assert!(i_1 < slices_1.len(), "DistsTracker::consume: mask index out of bounds (batch 1)");

                    let slice_0 = slices_0[i_0];
                    let slice_1 = slices_1[i_1];

                    trace!("DistsTracker::consume: channel={channel:?} i_0={i_0} i_1={i_1} slice_0={slice_0:?}  slice_1={slice_1:?} dist={dist}");

                    if slice_0.0 != slice_1.0 || slice_0.1 != slice_0.1 {
                        trace!("DistsTracker::consume, track: slice_0={slice_0:?} slice_1={slice_1:?}");

                        channel_trackers[channel].write().track(slice_0, slice_1, dist as f32);
                    }
                });

                channel_trackers
                    .into_par_iter()
                    .zip(matches_by_channel.into_par_iter())
                    .try_for_each(|(mt, matches)| {
                        let mut guard = mt.write();
                        guard.dump();
                        Ok::<(), ThoError>(())
                    })?;

                Ok::<_, ThoError>(())
            })
        })
        .await??;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use futures_util::StreamExt;
    use log::debug;
    use crate::tasks::cross_batch_clc::CrossClcResultMerge;
    use crate::tasks::dists_tracker::{DistsMask, DistsMaskType};
    use tch::{Device, Kind, Tensor};
    use tch::Device::Cuda;
    use tch::kind::FLOAT_CPU;
    use thoosa_common::utils::reset_print_options;
    use thoosa_shared::{Cause, ThoError};
    use thoosa_shared::runner::global_ctx::GlobalCtx;

    #[ctor::ctor]
    pub fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test]
    async fn test_0() -> Result<(), ThoError> {
        let v:Vec<usize>= Vec::default();

        debug!("v={}", v.capacity());
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
        if !tch::Cuda::is_available() {
            return Ok(());
        }
        let device = Device::Cuda(0);

        let left_mask = Tensor::from_slice(&[0.0f32, 1.0]).view([1, 1, 2]).lt(0.5);
        let right_mask = Tensor::from_slice(&[1.0f32, 0.0]).view([1, 1, 2]).lt(0.5);
        let left_dists = Tensor::from_slice(&[1.0f32, 2.0]).view([1, 1, 2]);
        let right_dists = Tensor::from_slice(&[3.0f32, 4.0]).view([1, 1, 2]);

        let left_mask = left_mask.f_to_device(device)?;
        let right_mask = right_mask.f_to_device(device)?;
        let left_dists = left_dists.f_to_device(device)?;
        let right_dists = right_dists.f_to_device(device)?;

        let left = DistsMask::new(
            left_mask,
            Some(left_dists),
            DistsMaskType::MaskPacked(vec![1, 1, 2], Some(vec![1, 1, 2])),
            None,
        )?;
        let right = DistsMask::new(
            right_mask,
            Some(right_dists),
            DistsMaskType::MaskPacked(vec![1, 1, 2], Some(vec![1, 1, 2])),
            None,
        )?;

        let merged = left.merge_batch_1(right)?;
        let (mask, dists) = merged.unpack()?;
        let mask = mask.f_to_device(Device::Cpu)?;
        let dists = dists.expect("dists").f_to_device(Device::Cpu)?;

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

        let left = DistsMask::new(left_mask, None, DistsMaskType::AsIs, None)?;
        let right = DistsMask::new(right_mask, None, DistsMaskType::AsIs, None)?;

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

        let left = DistsMask::new(left_mask, Some(left_dists), DistsMaskType::AsIs, None)?;
        let right = DistsMask::new(right_mask, None, DistsMaskType::AsIs, None)?;

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
        if !tch::Cuda::is_available() {
            return Ok(());
        }
        let device = Device::Cuda(0);

        let left_mask = Tensor::from_slice(&[0.0f32]).view([1, 1, 1]).lt(0.5);
        let right_mask = Tensor::from_slice(&[1.0f32]).view([1, 1, 1]).lt(0.5);

        let left_mask = left_mask.f_to_device(device)?;
        let left = DistsMask::new(
            left_mask,
            None,
            DistsMaskType::MaskPacked(vec![1, 1, 1], None),
            None,
        )?;
        let right = DistsMask::new(right_mask, None, DistsMaskType::AsIs, None)?;

        let err = left.merge_batch_1(right).expect_err("expected pack mismatch");
        match err.what() {
            Cause::SomethingElse(msg) => {
                assert_eq!(msg, "DistsMask merge: type mismatch");
            }
            other => panic!("unexpected error: {other:?}"),
        }

        Ok(())
    }

    #[test]
    fn test_merge_batch_1_many_packed_with_dists() -> Result<(), ThoError> {
        if !tch::Cuda::is_available() {
            return Ok(());
        }
        let device = Device::Cuda(0);

        let left_mask = Tensor::from_slice(&[0.0f32, 1.0]).view([1, 1, 2]).lt(0.5);
        let mid_mask = Tensor::from_slice(&[1.0f32, 0.0]).view([1, 1, 2]).lt(0.5);
        let right_mask = Tensor::from_slice(&[0.0f32, 0.0]).view([1, 1, 2]).lt(0.5);

        let left_dists = Tensor::from_slice(&[1.0f32, 2.0]).view([1, 1, 2]);
        let mid_dists = Tensor::from_slice(&[3.0f32, 4.0]).view([1, 1, 2]);
        let right_dists = Tensor::from_slice(&[5.0f32, 6.0]).view([1, 1, 2]);

        let left_mask = left_mask.f_to_device(device)?;
        let mid_mask = mid_mask.f_to_device(device)?;
        let right_mask = right_mask.f_to_device(device)?;
        let left_dists = left_dists.f_to_device(device)?;
        let mid_dists = mid_dists.f_to_device(device)?;
        let right_dists = right_dists.f_to_device(device)?;

        let left = DistsMask::new(
            left_mask,
            Some(left_dists),
            DistsMaskType::MaskPacked(vec![1, 1, 2], Some(vec![1, 1, 2])),
            None,
        )?;
        let mid = DistsMask::new(
            mid_mask,
            Some(mid_dists),
            DistsMaskType::MaskPacked(vec![1, 1, 2], Some(vec![1, 1, 2])),
            None,
        )?;
        let right = DistsMask::new(
            right_mask,
            Some(right_dists),
            DistsMaskType::MaskPacked(vec![1, 1, 2], Some(vec![1, 1, 2])),
            None,
        )?;

        let merged = left.merge_batch_1_many(vec![mid, right])?;
        let (mask, dists) = merged.unpack()?;
        let mask = mask.f_to_device(Device::Cpu)?;
        let dists = dists.expect("dists").f_to_device(Device::Cpu)?;

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

        let left = DistsMask::new(left_mask, None, DistsMaskType::AsIs, None)?;
        let mid = DistsMask::new(mid_mask, None, DistsMaskType::AsIs, None)?;
        let right = DistsMask::new(right_mask, None, DistsMaskType::AsIs, None)?;

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

        let left = DistsMask::new(left_mask, Some(left_dists), DistsMaskType::AsIs, None)?;
        let mid = DistsMask::new(mid_mask, None, DistsMaskType::AsIs, None)?;

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
        if !tch::Cuda::is_available() {
            return Ok(());
        }
        let device = Device::Cuda(0);

        let left_mask = Tensor::from_slice(&[0.0f32]).view([1, 1, 1]).lt(0.5);
        let mid_mask = Tensor::from_slice(&[1.0f32]).view([1, 1, 1]).lt(0.5);

        let left_mask = left_mask.f_to_device(device)?;
        let left = DistsMask::new(
            left_mask,
            None,
            DistsMaskType::MaskPacked(vec![1, 1, 1], None),
            None,
        )?;
        let mid = DistsMask::new(mid_mask, None, DistsMaskType::AsIs, None)?;

        let err = left
            .merge_batch_1_many(vec![mid])
            .expect_err("expected pack mismatch");
        match err.what() {
            Cause::SomethingElse(msg) => {
                assert_eq!(msg, "DistsMask merge: type mismatch");
            }
            other => panic!("unexpected error: {other:?}"),
        }

        Ok(())
    }

    #[test]
    fn test_into_match_indices_with_dists_dense() -> Result<(), ThoError> {
        let mask = Tensor::from_slice(&[0.0f32, 1.0, 0.0])
            .view([1, 1, 3])
            .lt(0.5);
        let dists = Tensor::from_slice(&[1.0f32, 2.0, 3.0]).view([1, 1, 3]);

        let mask = DistsMask::new(mask, Some(dists), DistsMaskType::AsIs, None)?;
        let mut matches = mask.into_indices_iter()?;
        let mut idx_values = Vec::new();
        let mut dists_values = Vec::new();

        while let Some((channel, i_0, i_1, dist)) = matches.next() {
            idx_values.push(channel);
            idx_values.push(i_0);
            idx_values.push(i_1);
            dists_values.push(dist);
        }

        assert_eq!(idx_values, vec![0, 0, 0, 0, 0, 2]);
        assert_eq!(dists_values, vec![1.0, 3.0]);

        Ok(())
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_mask_indexes() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);

        let dists = Tensor::from_slice(&[
            0.1, 0.0, 0.0,
            0.0, 0.0, 0.1,
            0.1, 0.0, 0.0,

            0.0, 0.0, 0.1,
            0.1, 0.0, 0.0,
            0.0, 0.1, 0.0,

            0.0, 0.0, 0.0,
            0.0, 0.1, 0.0,
            0.0, 0.0, 0.0,
        ]).to_device(Cuda(0)).view((3, 3, 3));

        let mask = dists.gt(0.05);

        debug!("dists=\n{dists}");
        debug!("mask=\n{mask}");


        let dists_masks = DistsMask::new(mask, Some(dists), DistsMaskType::MaskIndexes(vec![3, 3, 3]), None)?.to_cpu().await?;


        let mut iter = dists_masks.into_indices_iter()?;
    

        assert_eq!(
            vec![
                (0, 0, 0, 0.1),
                (0, 1, 2, 0.1),
                (0, 2, 0, 0.1),

                (1, 0, 2, 0.1),
                (1, 1, 0, 0.1),
                (1, 2, 1, 0.1),

                (2, 1, 1, 0.1)
            ],
            iter.into_iter().collect::<Vec<_>>()
        );

        Ok(())
    }
}
