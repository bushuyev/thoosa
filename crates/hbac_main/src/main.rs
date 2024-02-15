use std::env;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::str::FromStr;
use log::{debug, info, LevelFilter};
use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger, WriteLogger};
use tch::{Device, Kind, Tensor};
use hbac_motifs::find::find_motifs_iter;
use crate::data::{clc_records};
use crate::version::VERSION;
use clap::{Args, Parser, Subcommand};
use polars::export::rayon;
use polars::prelude::Series;
use hbac_motifs::utils::cdt;
use hbac_common::HbacError;

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

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    #[command()]
    FindMotifs (FindMotifsParams),
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
    std::io::stdout().flush()?;

    let base_path = cli.base_path.clone();

    match &cli.command {
        Some(Commands::FindMotifs(find_motifs_params)) => {
            debug!("find_motifs_params={:?}", find_motifs_params);

            let eeg_id_series = data::EEGs::good_eegs_of(base_path.as_str(), find_motifs_params.kind.as_str())?;

            let records_num = clc_records(&eeg_id_series, base_path.as_str())? as i64;

            find_motifs(
                move |key: &String| {
                    Box::new(data::eeg_to_tensor(base_path.as_str(), key.as_ref(), device).unwrap())
                },
                find_motifs_params,
                device,
                eeg_id_series,
                records_num
            )?;
        }
        None => {}
    }

    info!("All done");
    Ok(())
}

#[derive(Debug, Args)]
struct FindMotifsParams {
    #[arg(long, default_value_t = 1000)]
    step: i64,

    #[arg(long, default_value_t = 1000)]
    window: i64,

    #[arg(long, default_value_t = 0.9)]
    min_dot: f32,

    #[arg(long, default_value = "seizure_vote")]
    kind: String,

    #[arg(long, default_value = "../../data_local/motifs")]
    motifs_dir: String,

    #[arg(long, default_value_t = 10)]
    preload_d: usize,

    #[arg(long, default_value = None )]
    limit_eeg: Option<usize>,
}

fn find_motifs<F>(loader:F, cli: &FindMotifsParams, device: Device, eeg_id_series: Series, records_num:i64) -> Result<(), HbacError> where F:Fn(&String) -> Box<Tensor> + Sync + Send + 'static {
    let eeg_id_series = match cli.limit_eeg {
        None => {
            eeg_id_series
        }
        Some(v) => {
            eeg_id_series.limit(v)
        }
    };



    let slots_to_compare = (records_num - cli.window) / cli.step;
    let slots_to_compare = slots_to_compare.pow(2) / 2 - slots_to_compare;

    let motifs_dir = format!("{}/{}", cli.motifs_dir, cdt());

    if let Err(e) = std::fs::create_dir_all(motifs_dir.clone()) {
        panic!("can't create {}: {:?}", motifs_dir, e);
    }


    let eeg_ids_str = eeg_id_series.i64()?.into_iter().map(|v| v.unwrap().to_string()).collect::<Vec<String>>();
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(false)
        .open(format!("{}/{}_eegs.txt", motifs_dir, cli.kind))
        .unwrap();

    write!(file, "{:?}", eeg_ids_str);

    find_motifs_iter(
        loader,
        eeg_ids_str,
        cli.window,
        cli.step,
        (0..20).map(|_| cli.min_dot).collect(),
        slots_to_compare as usize,
        motifs_dir.as_str(),
        cli.preload_d
    );
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