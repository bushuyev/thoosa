use std::env;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::str::FromStr;
use log::{debug, info, LevelFilter};
use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger, WriteLogger};
use tch::{Device, Kind, Tensor};
use hbac_motifs::find::{find_motifs_iter, FindMotifsParams};
use crate::data::{clc_records, read_motif_to_iter};
use crate::version::VERSION;
use clap::{Args, Parser, Subcommand};
use polars::export::rayon;
use polars::prelude::Series;
use hbac_motifs::utils::cdt;
use hbac_common::HbacError;
use hbac_motifs::{find, fragments, TensorSliceIdx};
use hbac_motifs::fragments::{FragmentMotifsParams, group_motifs, list_motifs};
use std::collections::HashMap;

mod my_tch;
mod data;
mod builder;
mod utils;
mod simp;
mod version;



#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(long, default_value="../../data")]
    base_path: String,

    #[arg(long, default_value = "INFO")]
    term_logger_level: String,

    #[arg(long, default_value = "INFO")]
    write_logger_level: String,

    #[arg(long, default_value = "hbac.log")]
    logfile: String,

    #[arg(long, default_value = None )]
    limit_eeg: Option<usize>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    #[command()]
    FindMotifs (FindMotifsParams),
    MotifsToFragments(FragmentMotifsParams),
}


fn main() -> Result<(), HbacError> {

    let cli = Cli::parse();

    CombinedLogger::init(
        vec![
            TermLogger::new(LevelFilter::from_str(cli.term_logger_level.as_str())?, Config::default(), TerminalMode::Mixed, ColorChoice::Auto),
            WriteLogger::new(LevelFilter::from_str(cli.write_logger_level.as_str())?, Config::default(), File::create(cli.logfile)?),
        ]
    )?;

    let device = if tch::utils::has_cuda() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    info!("VERSION: {} device={:?}, base_path={:?} cpus={}", VERSION, device, cli.base_path, std::thread::available_parallelism()?);
    info!("limit_eeg: {:?}", cli.limit_eeg);
    std::io::stdout().flush()?;

    let base_path = cli.base_path.clone();

    match &cli.command {

        Some(Commands::FindMotifs(find_motifs_params)) => {
            info!("find_motifs_params={:?}", find_motifs_params);

            let eeg_id_series = data::EEGs::good_eegs_of(base_path.as_str(), find_motifs_params.kind())?;

            let eeg_id_series = match cli.limit_eeg {
                None => {
                    eeg_id_series
                }
                Some(v) => {
                    eeg_id_series.limit(v)
                }
            };

            let records_num = clc_records(&eeg_id_series, base_path.as_str())? as i64;
            debug!("records_num={}", records_num);

            find::find_motifs(
                move |key: &String| {
                    Box::new(data::eeg_to_tensor(base_path.as_str(), key.as_ref(), device).unwrap())
                },
                find_motifs_params,
                records_num,
                eeg_id_series.i64()?.into_iter()
            )?;
        }

        Some(Commands::MotifsToFragments(fragment_motifs_params)) => {
            fragments::motifs_to_eeg_slices(
                list_motifs(fragment_motifs_params.motifs_dir())?,
                read_motif_to_iter
            )?;
        }
        None => {}
    }

    info!("All done");
    Ok(())
}

#[cfg(test)]
mod main_tests {
    use crate::main;

    #[test]
    fn test_it(){
        main();
    }
}