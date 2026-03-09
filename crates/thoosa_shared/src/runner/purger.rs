use std::sync::Arc;
use futures_util::future::BoxFuture;
use log::{debug, error, trace, warn};
use tokio::sync::{mpsc, RwLock};
use tokio::sync::mpsc::error::SendError;
use tokio::task::JoinHandle;
use thoosa_shared::runner::global_ctx::GlobalCtx;
use crate::{panic_to_err, panic_to_err_async, ThoError};

pub enum PurgeCmd {
    Gpu {
        cache_area: u64,
        closures: Vec<Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>> + Send + Sync>>
    },
}

#[derive(Debug)]
pub struct Purger {
    tx: mpsc::UnboundedSender<PurgeCmd>,
    //TODO Arc<RwLock seems to prevent self.tx.send failing "channel closed"
    rx_arc: Arc<RwLock<mpsc::UnboundedReceiver<PurgeCmd>>>,
    worker: JoinHandle<()>,
}

impl Purger {
    pub fn new() -> Self {
        debug!("Purger::new");

        let (tx, mut rx) = mpsc::unbounded_channel::<PurgeCmd>();
        let rx_arc = Arc::new(RwLock::new(rx));
        let rx_arc_self = Arc::clone(&rx_arc);
        let worker = tokio::spawn(async move {
            // cache is owned here
            loop {

                match rx_arc.write().await.recv().await {
                    Some(PurgeCmd::Gpu { closures, cache_area }) => {
                        debug!("Purger::new, loop, got cmd: closures.len={}, cache_area={cache_area:064b} ", closures.len());

                        // match panic_to_err_async!({
                            for purge_cl in &closures {

                                match panic_to_err_async!(
                                    purge_cl()
                                ) {
                                    Ok(_) => {
                                        trace!("Purger::new, loop: done purging ")
                                    }
                                    Err(e) => {
                                        error!("Purger::new, loop:  purge failed: {e:?}")
                                    }
                                }
                            }

                            match panic_to_err!(Ok::<_, ThoError>(GlobalCtx::global().dispose_cache_area(cache_area))){
                                Ok(_) => {
                                    trace!("Purger::new, loop: done dispose_cache_area ")
                                }
                                Err(e) => {
                                    error!("Purger::new, loop: dispose_cache_area failed: {e:?}")
                                }
                            };

                            // Ok::<_, ThoError>(())
                        // }){
                        //     Ok(_) => {
                        //         trace!("Purger::new, loop, panic_to_err_async: done purging ")
                        //     }
                        //     Err(e) => {
                        //         error!("Purger::new, loop, panic_to_err_async:  purge failed: {e:?}")
                        //     }
                        // };
                    }
                    None => {
                        debug!("Purger::new, loop, break");
                        break;
                    },
                }
            }
            warn!("Purger::new: loop exited");
        });

        Self { tx, rx_arc:rx_arc_self, worker,  }
    }

    pub fn run_purge_gpu(&self, closures: Vec<Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>> + Send + Sync>>, cache_area: u64) -> Result<(), ThoError>{
        debug!("Purger::run_purge_gpu: closures.len={}", closures.len());

        match self.tx.send(PurgeCmd::Gpu {cache_area, closures}) {
            Ok(_) => {
                Ok(())
            }
            Err(e) => {
                warn!("Purger::run_purge_gpu: got e {e:?}");

                Err(e.into())
            }
        }
    }


}
