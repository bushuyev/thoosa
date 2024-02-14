use std::env;
use std::fs::File;
use std::io::Write;
use log::{debug, info, LevelFilter};
use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger, WriteLogger};
use tch::{Device, Kind, Tensor};
use hbac_motiffs::find_motiffs_iter;
use crate::data::{clc_records, GenError};
use crate::version::VERSION;
use clap::{Parser, Subcommand};

mod my_tch;
mod data;
mod builder;
mod utils;
mod simp;
mod version;



#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(short, long, default_value=Some("../../data".into()))]
    base_path: Option<String>,

    #[arg(short, long, default_value_t = 1000)]
    step: u32,


    #[arg(short, long, default_value_t = 1000)]
    window: u32,
}



fn main() -> Result<(), GenError>{

    let cli = Cli::parse();


    println!("{:?}", cli.base_path);

    fn load_sig_tensor(key:&String) ->Box<Tensor>{
        // Box::new(Tensor::rand(&[2, 1000], (Kind::Float, Device::Cpu)))
        Box::new(data::eeg_to_tensor("../../data", key.as_ref()).unwrap())
    }

    CombinedLogger::init(
        vec![
            TermLogger::new(LevelFilter::Debug, Config::default(), TerminalMode::Mixed, ColorChoice::Auto),
            WriteLogger::new(LevelFilter::Debug, Config::default(), File::create("hbac.log")?),
        ]
    )?;

    info!("VERSION: {}", VERSION);
    std::io::stdout().flush()?;

    let args: Vec<String> = env::args().collect();
    let base_path = args.get(1).map(|s|s.as_str()).unwrap_or("../../data");

    let device = if tch::utils::has_cuda() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    debug!("device={:?}, base_path={:?}", device, base_path);


    let eeg_id_series = data::EEGs::good_eegs_of("../../data", "seizure_vote")?;

    let eeg_id_series = eeg_id_series.limit(3);


    let records_num:usize = clc_records(&eeg_id_series)?;


    find_motiffs_iter(
        load_sig_tensor,
        // vec!["1628180742".into(), "1019408895".into(), "3523834378".into()],
        eeg_id_series.str()?.into_iter().map(|v|v.unwrap().into()).collect::<Vec<String>>(),
        1000,
        1000,
        (0..20).map(|_|0.9).collect(),
        "../../data_local/motiffs"
    );

    Ok(())
}