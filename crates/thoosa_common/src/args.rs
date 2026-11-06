use crate::ws::Workspace;
use async_trait::async_trait;
use clap::{Args, Subcommand};
use log::LevelFilter;
use parking_lot::RwLock;
use std::sync::Arc;

#[async_trait]
pub trait StageCommand {
    type Store;
    /// Lets feature commands retain their extended workspace trait object at the common CLI boundary.
    type Workspace: Workspace + ?Sized = dyn Workspace;
    type Ctx;
    async fn run(self, store: &Self::Store, ws: &Arc<RwLock<Box<Self::Workspace>>>, ctx: Self::Ctx) -> anyhow::Result<()>;
}

/// The runtime currently supports CPU-only execution or CPU plus the first visible GPU.
#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum ExecutionDevice {
    Auto,
    Cpu,
    Cuda0,
}

impl ExecutionDevice {
    pub fn resolve(self, ipc: bool, cuda_available: bool) -> anyhow::Result<usize> {
        let device = match self {
            Self::Auto => usize::from(ipc || cuda_available),
            Self::Cpu => 0,
            Self::Cuda0 => {
                anyhow::ensure!(ipc || cuda_available, "CUDA device 0 is not available");
                1
            }
        };
        // The IPC parent hides CUDA; only its worker can validate GPU availability.
        anyhow::ensure!(!ipc || device == 1, "IPC execution requires --execution-device cuda0 or auto");
        Ok(device)
    }
}

#[derive(Args, Clone, Debug)]
pub struct CommonArgs<C>
where
    C: Subcommand + StageCommand + Clone + std::fmt::Debug,
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
    pub run_id: Option<u32>,

    #[arg(long)]
    pub ipc: bool,

    #[arg(long, value_enum, default_value = "auto")]
    pub execution_device: ExecutionDevice,

    #[command(subcommand)]
    pub command: Option<C>,
}

#[cfg(test)]
mod tests {
    use super::ExecutionDevice::*;

    #[test]
    fn test_execution_device_selection_is_explicit_and_cpu_safe() {
        assert_eq!(Auto.resolve(false, false).unwrap(), 0);
        assert_eq!(Auto.resolve(false, true).unwrap(), 1);
        assert_eq!(Cpu.resolve(false, true).unwrap(), 0);
        assert_eq!(Auto.resolve(true, false).unwrap(), 1);
        assert!(Cuda0.resolve(false, false).is_err());
        assert!(Cpu.resolve(true, false).is_err());
    }
}
