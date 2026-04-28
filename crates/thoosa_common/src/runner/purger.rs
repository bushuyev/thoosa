use std::sync::Arc;
use futures_util::future::BoxFuture;
use log::{debug, error, trace, warn};
use mockall_double::double;
use tokio::sync::{mpsc, RwLock};
use tokio::task::JoinHandle;
use thoosa_shared::{panic_to_err, panic_to_err_async, ThoError};

#[double]
use crate::runner::global_ctx::GlobalCtx;



pub enum PurgeCmd {
    Gpu {
        closures: Vec<Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>> + Send + Sync>>
    },
}

#[derive(Debug)]
pub struct Purger {
    tx: mpsc::UnboundedSender<PurgeCmd>,
    //TODO Arc<RwLock seems to prevent self.tx.send failing "channel closed"
    _rx_arc: Arc<RwLock<mpsc::UnboundedReceiver<PurgeCmd>>>,
    _worker: JoinHandle<()>,
}

impl Purger {
    pub fn new() -> Self {
        debug!("Purger::new");

        let (tx, rx) = mpsc::unbounded_channel::<PurgeCmd>();
        let rx_arc = Arc::new(RwLock::new(rx));
        let rx_arc_self = Arc::clone(&rx_arc);
        let worker = tokio::spawn(async move {
            loop {

                match rx_arc.write().await.recv().await {
                    Some(PurgeCmd::Gpu { closures }) => {
                        debug!("Purger::new, loop, got cmd: closures.len={}", closures.len());

                            for purge_cl in &closures {

                                match panic_to_err_async!(
                                    purge_cl()
                                ) {
                                    Ok(_) => {
                                        trace!("Purger::new, loop: done purging")
                                    }
                                    Err(e) => {
                                        error!("Purger::new, loop:  purge failed: {e:?}")
                                    }
                                }
                            }
                    }
                    None => {
                        debug!("Purger::new, loop, break");
                        break;
                    },
                }
            }
            warn!("Purger::new: loop exited");
        });

        Self {
            tx,
            _rx_arc: rx_arc_self,
            _worker: worker,
        }
    }

    pub fn run_purge_gpu(&self, closures: Vec<Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>> + Send + Sync>>) -> Result<(), ThoError>{
        debug!("Purger::run_purge_gpu: closures.len={}", closures.len());

        match self.tx.send(PurgeCmd::Gpu { closures}) {
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
