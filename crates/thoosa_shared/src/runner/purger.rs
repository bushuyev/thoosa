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
}

impl Purger {
    pub fn new() -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel::<PurgeCmd>();
        let worker = tokio::spawn(async move {
            // cache is owned here
            loop {
                match rx.recv().await {
                    Some(PurgeCmd::Gpu { closures, cache_area }) => {

                        for cl in &closures {
                            match (cl)().await {
                                Ok(()) => {
                                    debug!("done purging ")
                                }
                                Err(e) => {
                                    error!("purge failed: {e:?}")
                                }
                            }
                        }

                        GlobalCtx::global().dispose_cache_area(cache_area);

                    }
                    None => break,
                }
            }
        });

        Self { tx, worker: Some(worker) }
    }

    pub fn run_purge_gpu(&self, closures: Vec<Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>> + Send + Sync>>, cache_area: u64) -> Result<(), ThoError>{
        self.tx.send(PurgeCmd::Gpu {cache_area, closures}).map_err(|e|e.into())
    }
}