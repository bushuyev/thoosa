use crate::cuda_utils::{copy_d2h, copy_d2h_from_ptr, copy_h2d, copy_h2d_from_ptr, cuda_mem, move_d2h, move_h2d, PinnedHostBuffer};
use thoosa_shims::context::ThoCudaContext;
use thoosa_shims::stream::ThoCudaStream;
use log::debug;
use std::fmt::{Debug, Formatter};
use std::sync::{Arc, OnceLock};
use mockall_double::double;
use tch::Kind::Int64;
use tch::{Kind, Tensor};
use tch::Device::Cpu;
use tub::{Guard, Pool};
#[double]
use crate::runner::global_ctx::GlobalCtx;
use thoosa_shared::ThoError;

use crate::tho_device::ThoDevice;

pub struct CopyCtx {
    cu_ctx: Arc<ThoCudaContext>,
    pub(crate) buffs: usize,
    pub(crate) buf_size: usize,
    stream_pool: Arc<Pool<ThoCudaStream>>,
    bufs_pool: Arc<Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>>,
}

impl Debug for CopyCtx {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CopyCtx")
            .field("cu_ctx", &self.cu_ctx)
            .field("buffs", &self.buffs)
            .field("buf_size", &self.buf_size)
            .field("bufs_pool.remaining_capacity", &self.bufs_pool.remaining_capacity())
            .finish()
    }
}

impl CopyCtx {
    pub fn new(buffs: usize, buf_size: usize) -> Self {
        let cu_ctx = ThoCudaContext::new(0).expect("failed to create CUDA context");
        let bufs_pool = Self::build_pool(buffs, buf_size, &cu_ctx);

        Self {
            cu_ctx,
            buffs,
            buf_size,
            stream_pool: Arc::new(Pool::from_iter((0..32).map(|i|ThoCudaStream::from_pool(0/*TODO device*/, false).unwrap()))),//TODO unwrap)),
            bufs_pool,
        }
    }

    fn build_pool(
        num_buffs: usize,
        buf_size: usize,
        cu_ctx: &Arc<ThoCudaContext>,
    ) -> Arc<Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>> {
        cu_ctx
            .bind_to_thread()
            .expect("failed to bind CUDA context to thread");
        let device = cu_ctx.ordinal() as i32;
        Arc::new(Pool::from_iter((0..num_buffs).map(|_| {
            (
                PinnedHostBuffer::new(buf_size).unwrap(),
                Arc::new(ThoCudaStream::from_pool(device, true).unwrap()),//TODO unwrap
            )
        })))
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub fn rebuild(&mut self, mem_k: f64, buffs: usize) {
        assert_eq!(self.bufs_pool.remaining_capacity(), self.buffs);
        self
            .cu_ctx
            .bind_to_thread()
            .expect("failed to bind CUDA context to thread");

        drop(Arc::into_inner(std::mem::replace(
            &mut self.bufs_pool,
            Arc::new(Pool::from_vec(vec![(
                PinnedHostBuffer::new(1).unwrap(),
                Arc::new(ThoCudaStream::from_pool(self.cu_ctx.ordinal() as i32, true).unwrap()),
            )])),
        )));

        self.buffs = buffs;
        self.buf_size = Self::clc_buff_size(mem_k, buffs);
        self.bufs_pool = Self::build_pool(buffs, self.buf_size, &self.cu_ctx)
    }

    //TODO these are 'exec' streams, not for copy
    pub async fn acquire_stream(&self) -> Guard<ThoCudaStream> {
        self.stream_pool.acquire().await
    }

    pub fn make_instance(mem_k: f64, buffs: usize) -> CopyCtx {
        let buf_size = Self::clc_buff_size(mem_k, buffs);

        CopyCtx::new(buffs, buf_size)
    }

    fn clc_buff_size(mem_k: f64, buffs: usize) -> usize {
        let (_free_mem, total_mem) = cuda_mem();

        let init_type = Int64;

        let max_elt_size = init_type.elt_size_in_bytes();
        let elts_in_mem = (total_mem as f64 * mem_k) as usize / max_elt_size; //123456; //25000; //123456; //

        let elts_in_buff = elts_in_mem / buffs;

        debug!("CopyCtx::clc_buff_size: buf_size={elts_in_buff} buffs={buffs}");

        let buf_size = elts_in_buff * max_elt_size;
        buf_size
    }

    pub async fn move_t(&self, src_t: Tensor, dst_device: usize, kind: Kind) -> Result<Tensor, ThoError> {
        if dst_device == 0 {
            move_d2h(kind, src_t, self.buf_size, &self.bufs_pool).await
        } else {
            move_h2d(kind, src_t, self.buf_size, &self.bufs_pool, dst_device).await
        }
    }

    pub async fn copy<R: AsRef<Tensor>>(&self, src_t: &R, dst_device: usize, kind: Kind) -> Result<Tensor, ThoError> {
        if dst_device == 0 {
            copy_d2h(kind, src_t, self.buf_size, &self.bufs_pool).await
        } else {
            copy_h2d(kind, src_t, self.buf_size, &self.bufs_pool, dst_device).await
        }
    }
    
    pub async fn copy_from_ptr(
        &self,
        src_ptr: usize,
        src_size: &[i64],
        dst_device: usize,
        kind: Kind,
    ) -> Result<Tensor, ThoError> {
        let t = if dst_device == 0 {
            copy_d2h_from_ptr(kind, self.buf_size, &self.bufs_pool, src_ptr, src_size).await
        } else {
            copy_h2d_from_ptr(
                kind,
                self.buf_size,
                &self.bufs_pool,
                dst_device,
                src_ptr,
                src_size,
            )
            .await
        };

        // println!("copy_from_ptr: copied t.size={:?} dst_device={dst_device} t=\n{t}", t.size());

        t
    }

    pub fn buf_size(&self) -> usize {
        self.buf_size
    }
}
