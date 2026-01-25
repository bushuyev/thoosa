use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::atomic::Ordering::Relaxed;
use futures_util::future::BoxFuture;
use log::{debug, error};
use tokio::sync::{mpsc, oneshot};
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
    worker: Option<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
}

impl Purger {
    pub fn new() -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_in_loop = Arc::clone(&shutdown);
        let (tx, mut rx) = mpsc::unbounded_channel::<PurgeCmd>();
        let worker = tokio::spawn(async move {
            // cache is owned here
            loop {
                if shutdown_in_loop.load(Relaxed) {
                    return
                }

                match rx.recv().await {
                    Some(PurgeCmd::Gpu { closures, cache_area }) => {
                        debug!("Purger::new, loop, got cmd: closures.len={}, cache_area={cache_area:064b} ", closures.len());


                        for cl in &closures {
                            match (cl)().await {
                                Ok(()) => {
                                    debug!("Purger::new, loop: done purging ")
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

        Self { tx, worker: Some(worker), shutdown:  Arc::clone(&shutdown) }
    }

    //TODO rename to send_purge_gpu_rq
    pub fn run_purge_gpu(&self, closures: Vec<Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>> + Send + Sync>>, cache_area: u64) -> Result<(), ThoError>{
        debug!("Purger::run_purge_gpu: closures.len={}", closures.len());
        self.tx.send(PurgeCmd::Gpu {cache_area, closures}).map_err(|e|e.into())
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true,Ordering::Relaxed);
    }
}