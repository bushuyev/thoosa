use std::sync::Arc;
use async_trait::async_trait;
use clap::{Args, Subcommand};
use log::LevelFilter;
use parking_lot::RwLock;
use thoosa_shared::ThoError;
use thoosa_shared::traits::RunCtx;
use crate::ws::Workspace;

#[async_trait]
pub trait StageCommand {
    type Store;
    type Ctx;
    async fn run(self, store:&Self::Store, ws: &Arc<RwLock<Box<dyn Workspace>>>, ctx:Self::Ctx) -> Result<(), ThoError>;
}


#[derive(Args, Clone, Debug)]
pub struct CommonArgs<C>
where
    C: Subcommand  + StageCommand+ Clone + std::fmt::Debug,
{
    #[arg(long, default_value = ".")]
    pub ws_path: String,

    #[arg(long, default_value = "INFO")]
    pub term_logger_level: LevelFilter,

    #[arg(long, default_value = "INFO")]
    pub write_logger_level: LevelFilter,

    #[arg(long, default_value_t = false)]
    pub term_logger_extra_fields: bool,

    #[arg(long, default_value_t = false)]
    pub write_logger_extra_fields: bool,

    #[arg(long, default_value_t = false)]
    pub write_perf_log_file: bool,

    #[arg(long)]
    pub loader_threads: Option<usize>,

    #[arg(long, default_value_t = 100)]
    pub futures_channel_bound: usize,

    #[arg(long)]
    pub run_id:Option<u32>,

    #[arg(long)]
    pub ipc:bool,


    #[command(subcommand)]
    pub command: Option<C>,
}
