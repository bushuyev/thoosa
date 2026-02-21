use futures_util::future::BoxFuture;
use log::{debug, error, trace, warn};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use thoosa_shared::runner::global_ctx::GlobalCtx;
use crate::ThoError;

pub enum PurgeCmd {
    Gpu {
        cache_area: u64,
        closures: Vec<Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>> + Send + Sync>>
    },
}

#[derive(Debug)]
pub struct Purger {
    tx: mpsc::UnboundedSender<PurgeCmd>,
    _worker: Option<JoinHandle<()>>,
}

impl Purger {
    pub fn new() -> Self {

        let (tx, mut rx) = mpsc::unbounded_channel::<PurgeCmd>();
        let worker = tokio::spawn(async move {
            // cache is owned here
            loop {
                if GlobalCtx::global().is_shutting_down() {
                    return
                }

                if rx.is_closed() {
                    warn!("Purger::new, loop: channel is closed");
                    return ;
                }

                match rx.recv().await {
                    Some(PurgeCmd::Gpu { closures, cache_area }) => {
                        trace!("Purger::new, loop, got cmd: closures.len={}, cache_area={cache_area:064b} ", closures.len());


                        for purge_cl in &closures {
                            if GlobalCtx::global().is_shutting_down() {
                                break;
                            }
                            // A shutdown can still occur while inside the purge_cl call spitting CUDA_ERROR_DEINITIALIZED, so it would need to be checked somewhere in cuda_utils::copy_h2d_from_ptr.
                            // However, that would be rather wasteful since the app is not going to shut down very often.
                            match purge_cl().await {
                                Ok(()) => {
                                    trace!("Purger::new, loop: done purging ")
                                }
                                Err(e) => {
                                    error!("Purger::new, loop:  purge failed: {e:?}")
                                }
                            }
                        }

                        GlobalCtx::global().dispose_cache_area(cache_area);

                    }
                    None => break,
                }
            }
        });

        Self { tx, _worker: Some(worker),  }
    }

    pub fn run_purge_gpu(&self, closures: Vec<Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>> + Send + Sync>>, cache_area: u64) -> Result<(), ThoError>{
        debug!("Purger::run_purge_gpu: closures.len={}", closures.len());
        if GlobalCtx::global().is_shutting_down() {
            debug!("Purger::run_purge_gpu: shutdown set, skipping purge");
            return Ok(());
        }
        self.tx.send(PurgeCmd::Gpu {cache_area, closures}).map_err(|e|e.into())
    }


}
