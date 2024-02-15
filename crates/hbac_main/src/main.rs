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
use clap::{Parser, Subcommand};
use polars::export::rayon;
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

    #[arg(long, default_value = "INFO")]
    term_logger_level: String,

    #[arg(long, default_value = "INFO")]
    write_logger_level: String,

    #[arg(long, default_value = "hbac.log")]
    logfile: String,
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

    let eeg_id_series = data::EEGs::good_eegs_of(cli.base_path.as_str(), cli.kind.as_str())?;

    // let eeg_id_series = cli.limit_eeg.map_or(eeg_id_series, |v|eeg_id_series.limit(v));
    let eeg_id_series = match cli.limit_eeg {
        None => {
            eeg_id_series
        }
        Some(v) => {
            eeg_id_series.limit(v)
        }
    };

    let records_num = clc_records(&eeg_id_series, cli.base_path.as_str())? as i64;

    let slots_to_compare = (records_num - cli.window) / cli.step;
    let slots_to_compare = slots_to_compare.pow(2) / 2 - slots_to_compare;

    let motifs_dir = format!("{}/{}", cli.motifs_dir, cdt());

    if let Err(e) = std::fs::create_dir_all(motifs_dir.clone()){
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
        move |key:&String|{
            Box::new(data::eeg_to_tensor(cli.base_path.as_str(), key.as_ref(), device).unwrap())
        },
        eeg_ids_str,
        cli.window,
        cli.step,
        (0..20).map(|_|cli.min_dot).collect(),
        slots_to_compare as usize,
        motifs_dir.as_str(),
        cli.preload_d
    );

    info!("All done");
    Ok(())
}