use std::backtrace::Backtrace;
use std::{env, fs};
use std::cmp::max;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::str::FromStr;
use log::{debug, error, info, LevelFilter};
use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger, WriteLogger};
use tch::{Device, Kind, nn, Reduction, Tensor};
use hbac_motifs::find::{find_motifs_iter, FindMotifsParams, iter_all_in_dir, select_motifs, SelectMotifsParams};
use crate::data::{clc_records, EEGs, read_eeg_index, read_motif_to_iter};
use crate::version::VERSION;
use clap::{Args, Parser, Subcommand};
use polars::export::rayon;
use polars::prelude::Series;
use hbac_common::utils::{cdt, DataBatch, dot_norm, norm_2, plot_series};
use hbac_shared::HbacError;
use hbac_motifs::{find, fragments};
use hbac_motifs::fragments::{check_slices, ClassTensorsDataBatch, list_motifs, load_motif_slices, motifs_to_tensor, MotifsToFragmentsParams, MotifsToTensorParams, PlotMotifSlicesParams, write_motifs_slices};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use plotters::prelude::BLACK;
use tch::nn::{ConvConfig, OptimizerConfig};
use hbac_common::builder::SeqBuilder;
use hbac_common::simp::DummyDs;
use hbac_store::{SqlStore, Store};
use hbac_shared::*;
use hbac_shared::TensorSliceIdx;



mod data;
mod utils;

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

    #[arg(long)]
    limit: Option<usize>,

    #[arg(long)]
    num_threads: Option<usize>,

    #[command(subcommand)]
    command: Option<Commands>,
}


#[derive(Debug, Args)]
pub struct TrainOnMotifParams {
    #[arg(long)]
    motif_id: usize,

    #[arg(long, default_value = "/tmp/motif_models")]
    output_dir: String,

    #[arg(long, default_value_t = 1000)]
    cycles: usize,

    #[arg(long, default_value_t = 1.0e-5)]
    lr: f64,
}

impl TrainOnMotifParams {
    pub fn motif_id(&self) -> usize {
        self.motif_id
    }
    pub fn output_dir(&self) -> &str {
        &self.output_dir
    }

    pub fn cycles(&self) -> usize {
        self.cycles
    }
    pub fn lr(&self) -> f64 {
        self.lr
    }
}

#[derive(Debug, Subcommand)]
enum Commands {
    #[command()]
    FindMotifs (FindMotifsParams),
    PlotMotifSlices(PlotMotifSlicesParams),
    MotifsToTensor(MotifsToTensorParams),
    SelectMotifs(SelectMotifsParams),
    TrainOnMotif(TrainOnMotifParams),
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
                select_motifs_params.label_tensors().into_iter().map(|label_tensors_path| {
                    let eeg_ids = EEGs::good_eegs_of(cli.base_path.as_str(), label_tensors_path.1.as_str(), cli.limit)?.i64()?.into_iter()
                        .enumerate()
                        .map(|(i, v)| v.expect(format!("no value in {}", i).as_str()).to_string())
                        .collect::<Vec<String>>();

                    Ok((label_tensors_path.0, eeg_ids.into_iter().map(|eeg_id| {//TODO CacheOnThreads
                        data::eeg_to_tensor(cli.base_path.as_str(), eeg_id.as_str(), device)
                    }))) //(label, path)
                }).collect::<Result<Vec<_>, HbacError>>()?,
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

        Some(Commands::TrainOnMotif(train_on_motif_params)) => {
            debug!("train_on_motif: train_on_motif_params={:?}", train_on_motif_params);

            let mut store_guard = store.lock()?;

            let (channel, run_id, slices) = store_guard.list_motif_slices(train_on_motif_params.motif_id(), None, None)?;

            debug!("train_on_motif: 1. channel={}, run_id={}, slices={}", channel, run_id, slices.len());

            let all_sices_1 = flat_map(load_motif_slices(cli.base_path.as_str(), channel, 1000, slices, store_guard.list_run_resources(run_id)?, data::eeg_to_tensor, device));


            let (channel, run_id, slices) = store_guard.list_motif_slices(train_on_motif_params.motif_id(), Some(all_sices_1.len()), Some((run_id, channel as usize)))?;

            debug!("train_on_motif: 0. channel={}, run_id={}, slices={}", channel, run_id, slices.len());

            let all_sices_0 = flat_map(load_motif_slices(cli.base_path.as_str(), channel, 1000, slices, store_guard.list_run_resources(run_id)?, data::eeg_to_tensor, device));

            debug!("train_on_motif: all_sices_1.len()={}, all_sices_0.len()={} all_sices_1[0].size()={:?}", all_sices_1.len(), all_sices_0.len(), all_sices_1[0].size());


            let vs = nn::VarStore::new(device);

            let net = SeqBuilder::new(1000)
                .conv(100, 1, 1, ConvConfig::default())
                .linear()
                .end_seq().build_tch_module(&vs.root())?;

            info!("train_on_motif: training on {} slices", all_sices_0.len());

            hbac_common::utils::train(
                &Device::Cpu,
                &ClassTensorsDataBatch::new(
                    HashMap::from([
                        (
                            0,
                            all_sices_0
                        ),
                        (
                            1,
                            all_sices_1
                        )
                    ]),
                    10,
                    0.8
                ) as &dyn DataBatch,
                net,
                &vs,
                |pred: &Tensor, target: &Tensor| {
                    pred.cross_entropy_loss::<Tensor>(&target.totype(Kind::Int64), None, Reduction::Mean, -100, 0.0)
                },
                train_on_motif_params.cycles,
                nn::AdamW::default().build(&vs, train_on_motif_params.lr)?
            )?;
        }
        None => {}
    }

    Ok(())
}

fn flat_map(slices: Vec<(String, Vec<Box<Tensor>>)>) -> Vec<Box<Tensor>> {
    slices.into_iter().flat_map(|(eeg_id, slices)| slices.into_iter()).collect::<Vec<Box<Tensor>>>()
}


#[cfg(test)]
mod main_tests {
    use log::debug;
    use tch::Device;
    use hbac_shared::HbacError;
    use hbac_store::{SqlStore, Store};
    use crate::data;
    use crate::data::{EEGs, read_eeg_index, read_motif_to_iter, read_motif_to_iter_imp};

    #[test]
    fn test_it() -> Result<(), HbacError>{
        debug!("XXXXXXXXXXXXXXXXX");
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