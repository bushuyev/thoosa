use std::backtrace::Backtrace;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::str::FromStr;
use log::{debug, error, info, LevelFilter};
use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger, WriteLogger};
use tch::{Device, Kind, Tensor};
use hbac_motifs::find::{find_motifs_iter, FindMotifsParams, iter_all_in_dir, select_motifs, SelectMotifsParams};
use crate::data::{clc_records, EEGs, read_eeg_index, read_motif_to_iter};
use crate::version::VERSION;
use clap::{Args, Parser, Subcommand};
use polars::export::rayon;
use polars::prelude::Series;
use hbac_common::utils::cdt;
use hbac_shared::HbacError;
use hbac_motifs::{find, fragments};
use hbac_motifs::fragments::{list_motifs, motifs_to_tensor, MotifsToFragmentsParams, MotifsToTensorParams, plot_fragments, PlotFragmentsParams, write_motifs_slices};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use hbac_store::{SqlStore, Store};
use hbac_shared::*;
use hbac_shared::TensorSliceIdx;


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


#[derive(Debug, Args)]
pub struct FillClasseResources {

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

    let store = Arc::new(Mutex::new(SqlStore::new("hbac.db3")?));

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
                eeg_id_series.i64()?.into_iter(),
                Arc::clone(&store)
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
            // let motifs_eeg_index = read_eeg_index(choose_fragments_params.motifs_dir())?;
            //
            // let t = motifs_to_tensor(
            //     cli.base_path.as_str(),
            //     fragments::motifs_to_eeg_slices(
            //         list_motifs(choose_fragments_params.motifs_dir())?,
            //         read_motif_to_iter
            //     )?,
            //     motifs_eeg_index.1,
            //     data::eeg_to_tensor,
            //     choose_fragments_params.window(),
            //     20
            // )?;
            //
            // if let Some(out) =  choose_fragments_params.output_dir() {
            //     let save_to = motifs_eeg_index.0.map(|v|v.replace(".txt", ".pt")).unwrap_or("unknown.pt".into());
            //     t.save(Path::new(out).join(save_to));
            // }
        }

        Some(Commands::SelectMotifs(find_good_motifs_params)) => {

            debug!("select_motifs: label_tensors()={:?}", find_good_motifs_params.label_tensors());

            let mut store_guard = store.lock()?;
            let motifs = motifs_to_tensor(
                cli.base_path.as_str(),
                store_guard.motifs_to_eeg_slices(1)?,
                vec![],
                data::eeg_to_tensor,
                1000,
                20
            )?;

            let t = select_motifs(

                // Tensor::load(find_good_motifs_params.motifs_tensor_path())?,
                motifs,

                find_good_motifs_params.label_tensors().into_iter().map(|label_tensors_path| {
                    Ok((label_tensors_path.0, iter_all_in_dir(label_tensors_path.1.as_str())?)) //(label, path)
                }).collect::<Result<Vec<_>, HbacError>>()?,

                cli.limit,
                1000
            )?;

            t.save(find_good_motifs_params.output_motifs_tensor_path())?;
        }

        None => {}
    }

    Ok(())
}

#[cfg(test)]
mod main_tests {
    use hbac_shared::HbacError;
    use hbac_store::{SqlStore, Store};
    use crate::data::{read_eeg_index, read_motif_to_iter, read_motif_to_iter_imp};

    #[test]
    fn test_it(){
        // main();
    }

    #[test]
    fn motifs_to_store() -> Result<(), HbacError>{
        let runs = vec![
            ("2024-02-14T17:36:36Z", "seizure"),
            ("2024-02-15T17:14:09Z", "lpd"),
            ("2024-02-16T06:34:04Z", "gpd"),
            ("2024-02-16T11:31:30Z", "lrda"),
            ("2024-02-16T16:29:11Z", "grda"),
            ("2024-02-19T07:12:01Z", "other"),
        ];

        let mut store = SqlStore::new("hbac_imp.db3")?;

        for (dir, desc) in runs {
            let (_, eeg_ids) = read_eeg_index(format!("../../data_down/14_02_results/{}", dir).as_str())?;

            let run_id = store.save_run(format!("{}/csv motifs import", desc).as_str(), &eeg_ids)?;

            for channel in 0..20 {
                let r = read_motif_to_iter_imp(format!("../../data_down/14_02_results/{}/motifs_{}.csv", dir, channel).as_str())?;

                for chunk in r.chunks(500) {
                    store.save_slices(run_id, channel, chunk.to_vec());
                }

                store.set_motifs(run_id, channel);

                println!("desc={}, channel={}, run_id={}, r.len()={}", desc, channel, run_id, r.len());
            }
        }
        Ok(())
    }
}