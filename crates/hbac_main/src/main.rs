use std::backtrace::Backtrace;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::str::FromStr;
use log::{debug, error, info, LevelFilter};
use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger, WriteLogger};
use tch::{Device, Kind, Tensor};
use hbac_motifs::find::{select_motifs, find_motifs_iter, SelectMotifsParams, FindMotifsParams, iter_all_in_dir};
use crate::data::{clc_records, read_eeg_index, read_motif_to_iter};
use crate::version::VERSION;
use clap::{Args, Parser, Subcommand};
use polars::export::rayon;
use polars::prelude::Series;
use hbac_common::utils::cdt;
use hbac_common::HbacError;
use hbac_motifs::{find, fragments, TensorSliceIdx};
use hbac_motifs::fragments::{MotifsToTensorParams, group_motifs, list_motifs, motifs_to_tensor, plot_fragments, PlotFragmentsParams, write_motifs_slices, MotifsToFragmentsParams};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

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
    limit: Option<usize>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Debug, Subcommand)]
enum Commands {
    #[command()]
    FindMotifs (FindMotifsParams),
    MotifsToFragments(MotifsToFragmentsParams),
    PlotFragments(PlotFragmentsParams),
    MotifsToTensor(MotifsToTensorParams),
    SelectMotifs(SelectMotifsParams),
}


fn main()  {
    let t1 = Instant::now();
    match _main() {

        Ok(_) => {
            info!("All done in {} mls", t1.elapsed().as_millis());
        }

        Err(e) => {
            error!("failed with {:?} in {} mls", e, t1.elapsed().as_millis());
        }
    }
}
fn _main() -> Result<(), HbacError> {

    let cli = Cli::parse();

    rayon::ThreadPoolBuilder::new().num_threads(4).build_global()?;

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
    info!("limit_eeg: {:?}", cli.limit);
    std::io::stdout().flush()?;

    let base_path = cli.base_path.clone();

    match &cli.command {

        Some(Commands::FindMotifs(find_motifs_params)) => {
            info!("find_motifs_params={:?}", find_motifs_params);

            let eeg_id_series = data::EEGs::good_eegs_of(base_path.as_str(), find_motifs_params.kind())?;

            let eeg_id_series = match cli.limit {
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
            write_motifs_slices(
                cli.base_path.as_str(),
                fragment_motifs_params.fragments_dir(),
                fragments::motifs_to_eeg_slices(
                    list_motifs(fragment_motifs_params.motifs_dir())?,
                    read_motif_to_iter
                )?,
                read_eeg_index(fragment_motifs_params.motifs_dir())?.1,
                data::eeg_to_tensor,
                fragment_motifs_params.window()
            )?;
        }

        Some(Commands::PlotFragments(plot_fragments_params)) => {
            plot_fragments(
                plot_fragments_params
            )?;
        }

        Some(Commands::MotifsToTensor(choose_fragments_params)) => {
            let motifs_eeg_index = read_eeg_index(choose_fragments_params.motifs_dir())?;
            let t = motifs_to_tensor(
                cli.base_path.as_str(),
                fragments::motifs_to_eeg_slices(
                    list_motifs(choose_fragments_params.motifs_dir())?,
                    read_motif_to_iter
                )?,
                motifs_eeg_index.1,
                data::eeg_to_tensor,
                choose_fragments_params.window(),
                20
            )?;

            if let Some(out) =  choose_fragments_params.output_dir() {
                let save_to = motifs_eeg_index.0.map(|v|v.replace(".txt", ".pt")).unwrap_or("unknown.pt".into());
                t.save(Path::new(out).join(save_to));

            }

        }

        Some(Commands::SelectMotifs(find_good_motifs_params)) => {

            debug!("select_motifs: label_tensors()={:?}", find_good_motifs_params.label_tensors());

            let t = select_motifs(

                Tensor::load(find_good_motifs_params.motifs_tensor_path())?,

                find_good_motifs_params.label_tensors().into_iter().map(|label_tensors_path| {
                    Ok((label_tensors_path.0, iter_all_in_dir(label_tensors_path.1.as_str())?)) //(label, path)
                }).collect::<Result<Vec<_>, HbacError>>()?,

                cli.limit
            )?;

            t.save(find_good_motifs_params.output_motifs_tensor_path())?;
        }

        None => {}
    }

    Ok(())
}

#[cfg(test)]
mod main_tests {
    use crate::main;

    #[test]
    fn test_it(){
        // main();
    }
}