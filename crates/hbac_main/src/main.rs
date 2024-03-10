use std::backtrace::Backtrace;
use std::{env, fs};
use std::cmp::max;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::str::FromStr;
use log::{debug, error, info, LevelFilter};
use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger, WriteLogger};
use tch::{Device, IndexOp, Kind, nn, Reduction, Tensor};
use hbac_motifs::find::{FindMotifsParams, iter_all_in_dir, select_motifs, SelectMotifsParams};
use crate::data::{clc_records, EEGs, eegs_to_labels_tensors, read_eeg_index, read_motif_to_iter};
use crate::version::VERSION;
use clap::{Args, Parser, Subcommand};
use polars::export::rayon;
use polars::prelude::Series;
use hbac_common::utils::{cdt, DataBatch, dot_norm, from, norm_2, plot_series};
use hbac_shared::HbacError;
use hbac_motifs::{find, fragments};
use hbac_motifs::fragments::{check_slices, list_motifs, load_motif_slices, motifs_to_tensor, MotifsToFragmentsParams, MotifsToTensorParams, PlotMotifSlicesParams, write_motifs_slices};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use plotters::prelude::BLACK;
use tch::nn::{ConvConfig, OptimizerConfig, VarStore};
use hbac_common::builder::{Node, SeqBuilder};
use hbac_common::train_nets::{ClassTensorsDataBatch, train_on_eeg, TrainOnEEGParams, TrainOnMotifsParams};
use hbac_common::simp::DummyDs;
use hbac_common::train_nets;
use hbac_store::{SqlStore, Store};
use hbac_shared::*;
use hbac_shared::TensorSliceIdx;
use crate::Commands::TrainOnEEG;


mod data;
mod utils;

mod version;


#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(long, default_value="../../data")]
    base_path: String,

    #[arg(long, default_value="hbac.db3")]
    store_path: String,

    #[arg(long, default_value = "INFO")]
    term_logger_level: String,

    #[arg(long, default_value = "INFO")]
    write_logger_level: String,

    #[arg(long, default_value = "hbac.log")]
    logfile: String,

    #[arg(long)]
    limit: Option<usize>,

    #[arg(long)]
    num_threads: Option<usize>,

    #[command(subcommand)]
    command: Option<Commands>,
}


#[derive(Debug, Subcommand)]
enum Commands {
    #[command()]
    FindMotifs (FindMotifsParams),
    PlotMotifSlices(PlotMotifSlicesParams),
    MotifsToTensor(MotifsToTensorParams),
    SelectMotifs(SelectMotifsParams),
    TrainOnMotifs(TrainOnMotifsParams),
    TrainOnEEG(TrainOnEEGParams),
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

    let store = Arc::new(Mutex::new(SqlStore::new(cli.store_path)?));

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

    let cpu_num = std::thread::available_parallelism()?;
    let num_threads = cli.num_threads.unwrap_or_else(||max(2, cpu_num.get()/2));

    info!("VERSION: {} device={:?}, base_path={:?} cpus={} num_threads={}", VERSION, device, cli.base_path, cpu_num, num_threads);
    info!("limit_eeg: {:?}", cli.limit);
    std::io::stdout().flush()?;


    rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global()?;

    let base_path = cli.base_path.clone();

    match &cli.command {

        Some(Commands::FindMotifs(find_motifs_params)) => {
            info!("find_motifs_params={:?}", find_motifs_params);

            let eeg_id_series = EEGs::good_eegs_of(base_path.as_str(), find_motifs_params.kind(), cli.limit)?;


            let records_num = clc_records(&eeg_id_series, base_path.as_str())?;
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

        Some(Commands::PlotMotifSlices(plot_motif_slices_params)) => {
            let mut store_guard = store.lock()?;

            let (channel, run_id, slices) = store_guard.list_motif_slices(plot_motif_slices_params.motif_id(), None, None)?;

            let slices = load_motif_slices(cli.base_path.as_str(), channel, 1000, slices, store_guard.list_run_resources(run_id)?, data::eeg_to_tensor, device);

            let all_sices = slices.iter().flat_map(|(eeg_id, slices)| slices.iter()).collect::<Vec<&Box<Tensor>>>();

            check_slices(&all_sices, 0.9);

            plot_series(
                all_sices.into_iter().map(|t|{
                    (format!("0"), BLACK, t.iter::<f64>().unwrap().enumerate().map(|(i, v)|(i as f32, v as f32)).collect::<Vec<(f32, f32)>>())
                }).collect(),
                format!("Motif {}", plot_motif_slices_params.motif_id()).as_str(),
                (10000, 10000),
                &Path::new(plot_motif_slices_params.output_dir()).join(format!("motif_{}.png", plot_motif_slices_params.motif_id()))
            ).unwrap();
        }

        Some(Commands::MotifsToTensor(choose_fragments_params)) => {

            let mut store_guard = store.lock()?;

            let t = motifs_to_tensor(
                cli.base_path.as_str(),
                store_guard.motifs_to_eeg_slices(choose_fragments_params.run_id())?,
                store_guard.list_run_resources(choose_fragments_params.run_id())?,
                data::eeg_to_tensor,
                choose_fragments_params.window(),
                20,
                device
            )?;

            if let Some(out) =  choose_fragments_params.output_dir() {
                t.0.save(Path::new(out).join(format!("motifs_in_{}", choose_fragments_params.run_id())));
            }
        }

        Some(Commands::SelectMotifs(select_motifs_params)) => {

            info!("select_motifs: select_motifs_params={:?}", select_motifs_params);

            let mut store_guard = store.lock()?;
            let (motifs, channel_motif_ids) = motifs_to_tensor(
                cli.base_path.as_str(),
                store_guard.motifs_to_eeg_slices(select_motifs_params.run_id())?,
                store_guard.list_run_resources(select_motifs_params.run_id())?,
                data::eeg_to_tensor,
                select_motifs_params.window() as i64,
                20,
                device
            )?;

            info!("select_motifs: motifs tensor built, size={:?}", motifs.size());

            let channels_good_motif_idxs = select_motifs(
                motifs,
                eegs_to_labels_tensors(cli.base_path.as_str(), select_motifs_params.label_tensors(), &device, cli.limit.as_ref()),
                cli.limit,
                select_motifs_params.step(),
                device
            )?;

            let motifs_corrs = channel_motif_ids.into_iter().zip(channels_good_motif_idxs.into_iter()).map(|(motif_ids, motif_idxs_corss)|{
                motif_idxs_corss.into_iter().map(|(motif_id, motif_idx_corr)|{
                    (motif_ids[motif_id as usize], motif_idx_corr)
                }).collect::<Vec<(usize, f64)>>()
            }).collect::<Vec<Vec<(usize, f64)>>>();

            debug!("select_motifs: motifs_corrs={:?}", motifs_corrs);
            info!("select_motifs: motifs_corrs.len()={:?}", motifs_corrs.len());

            store_guard.save_motifs_corrs(format!("{:?}", select_motifs_params.label_tensors()).as_str(), motifs_corrs)?;//TODO label_tensors

            // t.save(find_good_motifs_params.output_motifs_tensor_path())?;
        }

        Some(Commands::TrainOnMotifs(train_on_motif_params)) => {
            debug!("train_on_motif: train_on_motif_params={:?}", train_on_motif_params);
            let motifs_iter = {
                let mut store_guard = store.lock()?;

                fs::create_dir_all(train_on_motif_params.output_dir());

                // for motif_id in train_on_motif_params.motif_ids() {
                store_guard.list_corr_motifs(1)?//TODO param
                    .into_iter()
                    .flat_map(|channel_motifs|channel_motifs.into_iter())
                    .filter(|(motif_id, corr)| *corr > 0.05)
                    .map(|(motif_id, corr)|motif_id)
            };

            let node = train_nets::motif_node(train_on_motif_params.window() as i64);

            for motif_id in motifs_iter {

                let mut store_guard = store.lock()?;
                let window = train_on_motif_params.window() as i64;

                let (channel, run_id, slices) = store_guard.list_motif_slices(motif_id, None, None)?;

                debug!("train_on_motif: 1. channel={}, run_id={}, slices={}", channel, run_id, slices.len());


                let all_sices_1 = train_nets::flat_map(load_motif_slices(cli.base_path.as_str(), channel, window, slices, store_guard.list_run_resources(run_id)?, data::eeg_to_tensor, device));

                let (channel, run_id, slices) = store_guard.list_motif_slices(motif_id, Some(all_sices_1.len()), Some((run_id, channel as usize)))?;

                debug!("train_on_motif: 0. channel={}, run_id={}, slices={}", channel, run_id, slices.len());

                let all_sices_0 = train_nets::flat_map(load_motif_slices(cli.base_path.as_str(), channel, window, slices, store_guard.list_run_resources(run_id)?, data::eeg_to_tensor, device));

                debug!("train_on_motif: all_sices_1.len()={}, all_sices_0.len()={} all_sices_1[0].size()={:?}", all_sices_1.len(), all_sices_0.len(), all_sices_1[0].size());


                let (loss, mut vs) = train_nets::train_on_motif(all_sices_0, all_sices_1, train_on_motif_params, device, &node)?;

                debug!("train_on_motif: motif_id={} loss={}", motif_id, loss);

                if loss < 0.2 {
                    info!("train_on_motif: savgin motif_id={} loss={}", motif_id, loss);

                    let channel_models = Path::new(train_on_motif_params.output_dir()).join(channel.to_string());
                    fs::create_dir_all(&channel_models);

                    vs.save(channel_models.join(format!("{}_{:3}.safetensors", motif_id, loss)))?;
                }
            }
        }

        Some(TrainOnEEG(train_on_eeg_params)) => {

            train_on_eeg(
                train_on_eeg_params,
                eegs_to_labels_tensors(
                    cli.base_path.as_str(),
                    &vec![(1, "seizure_vote".into()), (0, "lpd_vote".into()), (0, "gpd_vote".into()), (0, "lrda_vote".into()), (0, "grda_vote".into()), (0, "other_vote".into())],//TODO load from corrs_run
                    &device,
                    cli.limit.as_ref()
                    //684400 - TODO cut at max_input
                ),
                10,
                1000,
                100,
                100,
                20,
                684400
            )?;
        }

        None => {}
    }

    Ok(())
}


#[cfg(test)]
mod main_tests {
    use hbac_shared::HbacError;
    use hbac_store::{SqlStore, Store};
    use crate::data::{read_eeg_index, read_motif_to_iter_imp};

    #[test]
    fn test_it() -> Result<(), HbacError>{
        // debug!("XXXXXXXXXXXXXXXXX");
        // main();
        // EEGs::good_eegs_of("../../data/", "seizure_vote", Some(10))?.i64()?.into_iter().enumerate().map(|(i, v):(usize, Option<i64>)|{
        //     // String::from(v.expect(format!("no value in {}", i))).as_str();
        //
        //     data::eeg_to_tensor("../../data/", v.expect(format!("no value in {}", i).as_str()).to_string().as_str(), Device::Cpu)
        // }).try_for_each(|t|{
        //     println!("t={:?}", t?.size());
        //
        //     Ok::<(), HbacError>(())
        // })?;

        // let ce = Tensor::from_slice(&[1., 1., 0., 0.]).binary_cross_entropy::<Tensor>(&Tensor::from_slice(&[1., 1., 0., 0.1]), None, Reduction::Mean);
        // ce.print();



        Ok(())
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