extern crate core;

use std::backtrace::Backtrace;
use std::{env, fs};
use std::cmp::max;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::str::FromStr;
use log::{debug, error, info, LevelFilter, warn};
use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger, WriteLogger};
use tch::{Device, IndexOp, Kind, nn, Reduction, Tensor};
use hbac_motifs::find::{FindMotifsParams, iter_all_in_dir, select_motifs, SelectMotifsParams};
use crate::data::{clc_records, EEGs, eegs_to_labels_tensors, eegs_to_logits_tensors, parquet_or_load_tensor, read_eeg_index, read_motif_to_iter};
use crate::version::VERSION;
use clap::{Args, Parser, Subcommand};
use polars::export::rayon;
use polars::prelude::{LazyCsvReader, Series};
use hbac_common::utils::{cdt, DataBatch, dot_norm, from, list_devices, load_tensor, norm_2, plot_series};
use hbac_shared::HbacError;
use hbac_motifs::{find, fragments};
use hbac_motifs::fragments::{check_slices, list_motifs, load_motif_slices, motifs_to_tensor, MotifsToFragmentsParams, MotifsToTensorParams, PlotMotifSlicesParams, write_motifs_slices};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use plotters::prelude::BLACK;
use polars::export::rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelBridge};
use tch::Device::Cpu;
use tch::nn::{ConvConfig, OptimizerConfig, VarStore};

use hbac_common::builder::{Node, SeqBuilder};
use hbac_common::train_nets::{ClassTensorsDataBatch, train_on_eeg, TrainOnEEGParams, TrainOnMotifsParams, TrainFinalParams, train_final, InferParams, infer};
use hbac_common::simp::DummyDs;
use hbac_common::train_nets;
use hbac_store::{SqlStore, Store};
use hbac_shared::*;
use hbac_shared::TensorSliceIdx;
use crate::Commands::TrainOnEEG;
use polars::export::rayon::iter::ParallelIterator;


mod data;
mod utils;

mod version;


#[derive(Parser, Clone)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(long, default_value="../../data")]
    base_path: String,

    #[arg(long)]
    tensors_path: Option<String>,

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


#[derive(Debug, Args, Clone)]
pub struct RunAllParams {

}

#[derive(Debug, Args, Clone)]
pub struct InitResourcesParams {

}

#[derive(Debug, Subcommand, Clone)]
enum Commands {
    #[command()]
    InitResources (InitResourcesParams),
    FindMotifs (FindMotifsParams),
    PlotMotifSlices(PlotMotifSlicesParams),
    MotifsToTensor(MotifsToTensorParams),
    SelectMotifs(SelectMotifsParams),
    TrainOnMotifs(TrainOnMotifsParams),
    TrainOnEEG(TrainOnEEGParams),
    TrainFinal(TrainFinalParams),
    RunAll(RunAllParams),
    Infer(InferParams),
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

    CombinedLogger::init(
        vec![
            TermLogger::new(LevelFilter::from_str(cli.term_logger_level.as_str())?, Config::default(), TerminalMode::Mixed, ColorChoice::Auto),
            WriteLogger::new(LevelFilter::from_str(cli.write_logger_level.as_str())?, Config::default(), File::create(&cli.logfile)?),
        ]
    ).expect("failed to init logging");

    let store = Arc::new(Mutex::new(SqlStore::new(&cli.store_path)?));


    let devices = list_devices();
    let device = if tch::utils::has_cuda() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    let cpu_num = std::thread::available_parallelism()?;
    let num_threads = cli.num_threads.unwrap_or_else(||max(2, cpu_num.get()/2));

    info!("VERSION: {} devices={:?}, base_path={:?} cpus={} num_threads={}", VERSION, devices, cli.base_path, cpu_num, num_threads);
    info!("limit_eeg: {:?}", cli.limit);
    std::io::stdout().flush()?;


    rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global()?;
    let classes = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"];//TODO conf

    match &cli.command {

        Some(Commands::InitResources(init_resources_params)) => {
            debug!("InitResources: init_resources_params={:?}", init_resources_params);
            init_resources(
                &cli,
                init_resources_params,
                Arc::clone(&store)
            )?;
        }
        /**
        good_eegs_of
        class=seizure_vote len=2227
        class=lpd_vote len=1181
        class=gpd_vote len=806
        class=lrda_vote len=384
        class=grda_vote len=1154
        class=other_vote len=4208
        */
        Some(Commands::FindMotifs(find_motifs_params)) => {
            info!("find_motifs_params={:?} devices={:?}", find_motifs_params, devices);

            run_find_motifs(&cli, &store, devices, classes, find_motifs_params)?;

        }

        Some(Commands::SelectMotifs(select_motifs_params)) => {
            info!("select_motifs_params={:?} devices={:?}", select_motifs_params, devices);

            run_select_motifs(&cli, &store, devices, classes, &select_motifs_params);

        }

        Some(Commands::TrainOnMotifs(train_on_motif_params)) => {
            debug!("train_on_motif: train_on_motif_params={:?}", train_on_motif_params);
            let _ = train_on_motif_params.corr_run();

            let devices_len = devices.len();
            let run_ids = vec![1usize, 2, 3, 4, 5, 6];
            let devices_corr_run = devices.into_iter().cycle().zip(run_ids.chunks(devices_len).into_iter()).collect::<Vec<(Device, &[usize])>>();
            let cli_clone = cli.clone();
            devices_corr_run.into_par_iter().for_each( move |(device_i, corrs_run)| {
                info!("TrainOnMotifs: corrs_run: {:?} device_i={:?}", corrs_run, device_i);

                for corr_run in corrs_run {
                    train_on_motifs(&cli_clone, Arc::clone(&store), device_i, train_on_motif_params, *corr_run).unwrap();
                }

                info!("TrainOnMotifs: corrs_run: {:?} device_i={:?} done ", corrs_run, device_i);
            });
        }

        Some(TrainOnEEG(train_on_eeg_params)) => {

            let mut store_guard = store.lock()?;

            let labels_kinds = store_guard.corrs_descr(train_on_eeg_params.corr_run())?;

            let vs = train_on_eeg(
                train_on_eeg_params,
                eegs_to_labels_tensors(
                    cli.base_path.as_str(),
                    cli.tensors_path,
                    &labels_kinds,
                    &device,
                    cli.limit.as_ref()
                    //684400 - TODO cut at max_input
                ),
                5,
                1000,
                100,
                100,
                20,
                684400,
                &device
            )?;

            vs.save(format!("{}/corr_model_{}.safetensors", train_on_eeg_params.eeg_models_dir(), train_on_eeg_params.corr_run()))?;
        }

        Some(Commands::TrainFinal(train_final_params)) => {

            let labels_kinds = vec![
                (0, "seizure_vote".to_string()),
                (1, "lpd_vote".to_string()),
                (2, "gpd_vote".to_string()),
                (3, "lrda_vote".to_string()),
                (4, "grda_vote".to_string()),
                (5, "other_vote".to_string())
            ];

            let vs = train_final(
                train_final_params,
                eegs_to_logits_tensors(
                    cli.base_path.as_str(),
                    &labels_kinds,//TODO load from corrs_run
                    &device,
                    cli.limit.as_ref()
                    //684400 - TODO cut at max_input
                ),
                5,
                1000,
                100,
                100,
                20,
                684400,

                &device
            )?;

            vs.save(format!("{}/final.safetensors", train_final_params.models_dir()))?;
        }

        Some(Commands::RunAll(run_all_params)) => {

            let run_ids = run_find_motifs(&cli, &store, devices.clone(), classes, &FindMotifsParams::dummy(Some("pure group 200".into())))?;

            for run_id in run_ids {
                run_select_motifs(&cli, &store, devices.clone(), classes, &SelectMotifsParams::dummy(run_id, vec![
                    ("seizure".into(), "seizure/10".into()),
                    ("lpd".into(), "lpd/10".into()),
                    ("gpd".into(), "gpd/10".into()),
                    ("lrda".into(), "lrda/10".into()),
                    ("grda".into(), "grda/10".into()),
                    ("other".into(), "other/10".into()),
                ]));
            }
        }

        Some(Commands::Infer(infer_params)) => {
            let t = infer(
                &infer_params,
                |path:&Path|{
                    data::parquet_to_tensor_by_path(<Path as AsRef<Path>>::as_ref(path), *devices.first().expect("first device"))
                },
                1000,
                100,
                100,
                20,
                684400,
                &device,
            )?;

            data::save_tensor(
                &["eeg_id","seizure_vote","lpd_vote","gpd_vote","lrda_vote","grda_vote","other_vote"],
                t.0,
                &t.1,
                infer_params.output()
            )

        }
        Some(Commands::PlotMotifSlices(plot_motif_slices_params)) => {
            plot_motif_slices(&cli, &store, *devices.first().expect("first device"), &plot_motif_slices_params)?;
        }

        Some(Commands::MotifsToTensor(choose_fragments_params)) => {
            motifs_to_tensor_cmd(&cli, &store, *devices.first().expect("first device"), choose_fragments_params)?;
        }
        None => {
            warn!("No command");
        }
    }

    Ok(())
}

pub fn run_select_motifs(cli: &Cli, store: &Arc<Mutex<SqlStore>>, devices: Vec<Device>, classes: [&str; 6], select_motifs_params: &SelectMotifsParams) -> Result<(), HbacError>{
    // let class_run_id: HashMap<&str, usize> = HashMap::from([
    //     ("other_vote", 1usize),
    //     ("gpd_vote", 2usize),
    //     ("grda_vote", 3usize),
    //     ("lpd_vote", 4usize),
    //     ("lrda_vote", 5usize),
    //     ("seizure_vote", 6usize),
    // ]);
    // let devices_len = devices.len();
    // let devices_classes = devices.into_iter().cycle().zip(classes.chunks(devices_len).into_iter()).collect::<Vec<(Device, &[&str])>>();

    let chunk_len = select_motifs_params.classes_tensors().len()/devices.len();

    let cli_clone = cli.clone();
    let zip = devices.into_iter().cycle().zip(select_motifs_params.labels_datasets().chunks(chunk_len).into_iter()).collect::<Vec<(Device, &[(String, String)])>>();
    zip.into_par_iter().for_each(move |(device_i, labels_sets)| {
        for labels_set in labels_sets {
            // let classes_tensors = classes.iter().map(|cl| (if class.eq(cl) { 1 } else { 0 }, cl.to_string())).collect();
            // debug!("SelectMotifs: classes_tensors: {:?} device_i={:?}", classes_tensors, device_i);

            select_motifs_in_class(
                &cli_clone, Arc::clone(&store),
                device_i,
                &select_motifs_params
                // *class_run_id.get(class).expect("run_id")/*select_motifs_params.run_id()*/
            ).expect("select_motifs_in_class");
        }
    });

    Ok(())
}

fn run_find_motifs(cli: &Cli, store: &Arc<Mutex<SqlStore>>, devices: Vec<Device>, classes: [&str; 6], find_motifs_params: &FindMotifsParams) -> Result<Vec<usize>, HbacError>{
    // let devices_classes = devices.into_iter().cycle().zip(classes.into_iter()).collect::<Vec<(Device, &str)>>();

    let resource_sets = { store.lock().unwrap().list_resource_set_group_sets(find_motifs_params.resource_group()).expect("list_resource_set_group_sets")};

    let chunk_len = resource_sets.len()/devices.len();
    let devices_resource_sets = devices.into_iter().cycle().zip(resource_sets.chunks(chunk_len).into_iter()).collect::<Vec<(Device, &[(usize, String)])>>();

    let mut run_ids = Arc::new(Mutex::new(vec![]));
    devices_resource_sets.into_par_iter().for_each(move |(device_i, chunk_resource_sets)| {
        for resource_set in chunk_resource_sets {
            // let eeg_id_series = EEGs::good_eegs_of(cli.base_path.as_str(), class, cli.limit).unwrap();
            let eeg_id_series = { store.lock().unwrap().list_resource_set_resources(resource_set.0).expect("list_resource_set_resources") };

            let records_num = clc_records(&eeg_id_series, cli.base_path.as_str()).unwrap();
            debug!("records_num={}", records_num);

            let tensors_path = cli.tensors_path.clone();
            let base_path = cli.base_path.clone();
            let run_id = find::find_motifs_euc(
                move |key: &String| {
                    if let Some(tensors_path) = tensors_path.as_ref() {
                        Box::new(load_tensor(tensors_path.as_str(), key.as_ref(), device_i).unwrap())
                    } else {
                        Box::new(data::parquet_to_tensor(base_path.as_str(), key.as_ref(), device_i).unwrap())
                    }
                },
                &FindMotifsParams::for_label(resource_set.1.as_str(), find_motifs_params.clone()),
                records_num,
                eeg_id_series,
                Arc::clone(&store), 20,
            ).unwrap();

            run_ids.lock().expect("run_ids lock").push(run_id);
        }
    });

    Ok(run_ids.lock().expect("run_ids lock on return").clone())
}

fn motifs_to_tensor_cmd(cli: &Cli, store: &Arc<Mutex<SqlStore>>, device: Device, choose_fragments_params: &MotifsToTensorParams) -> Result<(), HbacError> {
    let mut store_guard = store.lock()?;
    let base_path_1 = cli.base_path.clone();
    let tensors_path = cli.tensors_path.clone();
    let t = motifs_to_tensor(
        store_guard.motifs_to_eeg_slices(choose_fragments_params.run_id())?,
        move |key: &String| {

            if let Some(tensors_path) = tensors_path.as_ref() {
                Box::new(load_tensor(tensors_path.as_str(), key.as_ref(), device).unwrap())
            } else {
                Box::new(data::parquet_to_tensor(base_path_1.as_str(), key.as_ref(), device).unwrap())
            }
        },
        choose_fragments_params.window(),
        20,
        device
    )?;

    if let Some(out) = choose_fragments_params.output_dir() {
        t.0.save(Path::new(out).join(format!("motifs_in_{}", choose_fragments_params.run_id())));
    }
    Ok(())
}

fn init_resources(cli:&Cli, params:&InitResourcesParams, store:Arc<Mutex<impl Store + 'static>>) -> Result<(), HbacError> {
    let mut store_guard = store.lock()?;

    store_guard.save_label("seizure");
    store_guard.save_label("lpd");
    store_guard.save_label("gpd");
    store_guard.save_label("lrda");
    store_guard.save_label("grda");
    store_guard.save_label("other");


    store_guard.save_resources_labels(data::labeled_resourses(Path::new(&cli.base_path).join("train.csv"))?)?;

    store_guard.save_pure_sets(1000);
    store_guard.save_pure_sets(200);
    store_guard.save_pure_sets(10);
    /*
    sqlite> select count(*), lb.label from (select count(*), label_id from labeled_resources group by resource_id having sum(p)=max(p)) as cnt left join labels lb on lb.id=cnt.label_id group by label_id;
    2227|seizure
    1181|lpd
    806|gpd
    384|lrda
    1154|grda
    4208|other
    */

    Ok(())
}


fn plot_motif_slices(cli: &Cli, store: &Arc<Mutex<SqlStore>>, device: Device, plot_motif_slices_params: &&PlotMotifSlicesParams) -> Result<(), HbacError> {
    let mut store_guard = store.lock()?;

    let (channel, run_id, slices) = store_guard.list_motif_slices(plot_motif_slices_params.motif_id(), None, None)?;
    let base_path = cli.base_path.as_str();

    let slices = load_motif_slices(
        cli.base_path.as_str(), channel, 1000, slices, store_guard.list_run_resources(run_id)?,
        move |eeg_id| {
            parquet_or_load_tensor(base_path, cli.tensors_path.clone(), eeg_id, device)
        },
        device
    );

    let all_sices = slices.iter().flat_map(|(eeg_id, slices)| slices.iter()).collect::<Vec<&Box<Tensor>>>();

    check_slices(&all_sices, 0.9);

    plot_series(
        all_sices.into_iter().map(|t| {
            (format!("0"), BLACK, t.iter::<f64>().unwrap().enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>())
        }).collect(),
        format!("Motif {}", plot_motif_slices_params.motif_id()).as_str(),
        (10000, 10000),
        &Path::new(plot_motif_slices_params.output_dir()).join(format!("motif_{}.png", plot_motif_slices_params.motif_id()))
    ).unwrap();
    Ok(())
}

fn train_on_motifs(cli: &Cli, store:Arc<Mutex<impl Store + 'static>>, device: Device, train_on_motif_params: &TrainOnMotifsParams, corr_run: usize) -> Result<(), HbacError> {
    debug!("train_on_motifs: device={:?}, corr_run={:?}", device, corr_run);

    let motifs_iter = {
        let mut store_guard = store.lock()?;

        // for motif_id in train_on_motif_params.motif_ids() {

        store_guard.list_corr_motifs(corr_run)?//TODO param
            .into_iter()
            .flat_map(|channel_motifs| channel_motifs.into_iter())
            .filter(|(motif_id, corr)| *corr > train_on_motif_params.min_corr())
            .map(|(motif_id, corr)| motif_id)
    };

    let node = train_nets::motif_node(train_on_motif_params.window() as i64);

    let output_dir = Path::new(train_on_motif_params.output_dir()).join(format!("corr_run_{}", corr_run));//TODO corr_run

    fs::create_dir_all(&output_dir);

    for motif_id in motifs_iter {
        let mut store_guard = store.lock()?;
        let window = train_on_motif_params.window() as i64;

        let (channel, run_id, slices) = store_guard.list_motif_slices(motif_id, None, None)?;

        debug!("train_on_motif: 1. channel={}, run_id={}, slices={}", channel, run_id, slices.len());

        let all_sices_1 = train_nets::flat_map(
            load_motif_slices(
                cli.base_path.as_str(), channel, window, slices, store_guard.list_run_resources(run_id)?,
                move |eeg_id| {
                    parquet_or_load_tensor(cli.base_path.as_str(), cli.tensors_path.clone(), eeg_id, device)
                },
                device
            )
        );


        let (channel, run_id, slices) = store_guard.list_motif_slices(motif_id, Some(all_sices_1.len()), Some((run_id, channel as usize)))?;

        debug!("train_on_motif: 0. channel={}, run_id={}, slices={}", channel, run_id, slices.len());

        let all_sices_0 = train_nets::flat_map(
            load_motif_slices(
                cli.base_path.as_str(), channel, window, slices, store_guard.list_run_resources(run_id)?,
                move |eeg_id| {
                    parquet_or_load_tensor(cli.base_path.as_str(), cli.tensors_path.clone(), eeg_id, device)
                },
                device
            )
        );


        debug!("train_on_motif: all_sices_1.len()={}, all_sices_0.len()={} all_sices_1[0].size()={:?}", all_sices_1.len(), all_sices_0.len(), all_sices_1[0].size());

        let (loss, mut vs) = train_nets::train_on_motif(all_sices_0, all_sices_1, train_on_motif_params, device, &node)?;

        debug!("train_on_motif: motif_id={} loss={}", motif_id, loss);

        if loss < train_on_motif_params.max_loss() {
            info!("train_on_motif: saving motif_id={} loss={}", motif_id, loss);

            let channel_models = output_dir.join(channel.to_string());
            fs::create_dir_all(&channel_models);

            vs.save(channel_models.join(format!("{}_{:3}.safetensors", motif_id, loss)))?;
        }
    }
    Ok(())
}

fn select_motifs_in_class(cli: &Cli, store:Arc<Mutex<impl Store + 'static>>, device: Device, select_motifs_params: &SelectMotifsParams) -> Result<(), HbacError> {
    debug!("select_motifs_in_class: device={:?} cli.tensors_path={:?}", device, cli.tensors_path);

    let tensors_path = cli.tensors_path.clone();
    let base_path = cli.base_path.clone();

    let (fragments_map, _) = {
        let mut store_guard = store.lock()?;

        (
            store_guard.motifs_to_eeg_slices(select_motifs_params.run_id())?,
            () //store_guard.list_run_resources(select_motifs_params.run_id())?
        )
    };

    let (motifs, channel_motif_ids) = motifs_to_tensor(
        fragments_map,
        // data::parquet_to_tensor,
        move |key: &String| {
            if let Some(tensors_path) = tensors_path.as_ref() {
                Box::new(load_tensor(tensors_path.as_str(), key.as_ref(), device).unwrap())
            } else {
                Box::new(data::parquet_to_tensor(base_path.as_str(), key.as_ref(), device).unwrap())
            }
        },
        select_motifs_params.window() as i64,
        20,
        device
    )?;

    info!("select_motifs: motifs tensor built, size={:?}", motifs.size());

    let tensors_path = cli.tensors_path.clone();
    let channels_good_motif_idxs = select_motifs(
        motifs,
        eegs_to_labels_tensors(cli.base_path.as_str(), tensors_path, select_motifs_params.labels_datasets(), &device, cli.limit.as_ref()),
        cli.limit,
        select_motifs_params.step(),
        device
    )?;

    let motifs_corrs = channel_motif_ids.into_iter().zip(channels_good_motif_idxs.into_iter()).map(|(motif_ids, motif_idxs_corss)| {
        motif_idxs_corss.into_iter().map(|(motif_id, motif_idx_corr)| {
            (motif_ids[motif_id as usize], motif_idx_corr)
        }).collect::<Vec<(usize, f64)>>()
    }).collect::<Vec<Vec<(usize, f64)>>>();

    debug!("select_motifs: motifs_corrs={:?}", motifs_corrs);
    info!("select_motifs: motifs_corrs.len()={:?}", motifs_corrs.len());

    store.lock()?.save_motifs_corrs(format!("{:?}", select_motifs_params.classes_tensors()).as_str(), motifs_corrs)?;//TODO label_tensors
    Ok(())
}


#[cfg(test)]
mod main_tests {
    use std::fs;
    use std::sync::{Arc, Mutex};
    use tch::Device::Cpu;
    use hbac_motifs::find::{FindMotifsParams, SelectMotifsParams};
    use hbac_shared::HbacError;
    use hbac_store::{SqlStore, Store};
    use crate::{Cli, Commands, run_find_motifs, run_select_motifs};
    use crate::Commands::{FindMotifs, SelectMotifs};
    // use crate::data::{clc_records, read_eeg_index, read_motif_to_iter_imp};

    #[test]
    fn test_it() -> Result<(), HbacError>{

        
        
        // clc_records()
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
    fn test_run_find_motifs() -> Result<(), HbacError>{
        let cli = Cli{
            base_path: "../../data/".to_string(),
            tensors_path: None,
            store_path: "".to_string(),
            term_logger_level: "".to_string(),
            write_logger_level: "".to_string(),
            logfile: "".to_string(),
            limit: None,
            num_threads: None,
            command: Some(FindMotifs(FindMotifsParams::dummy(None))),
        };

        fs::remove_file("/tmp/test_main_db.db3").unwrap_or(());
        let store = Arc::new(Mutex::new(SqlStore::new("/tmp/test_main_db.db3")?));

        let classes = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"];//TODO conf


        run_find_motifs(
            &cli,
            &store,
            vec![Cpu],
            classes,
            &FindMotifsParams::dummy(Some("pure group 200".into()))
        );

        Ok(())
    }

    #[test]
    fn test_run_select_motifs() -> Result<(), HbacError>{
        let cli = Cli{
            base_path: "../../data/".to_string(),
            tensors_path: None,
            store_path: "".to_string(),
            term_logger_level: "".to_string(),
            write_logger_level: "".to_string(),
            logfile: "".to_string(),
            limit: None,
            num_threads: None,
            command: Some(SelectMotifs(SelectMotifsParams::dummy(vec![]))),
        };

        let store = Arc::new(Mutex::new(SqlStore::new("/tmp/test_main_db.db3")?));
        let classes = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"];//TODO conf

        run_select_motifs(
            &cli,
            &store,
            vec![Cpu],
            classes,
            &SelectMotifsParams::dummy(vec![("seizure".into(), "seizure/10".into())])
        )?;

        Ok(())
    }

    // #[test]
    // fn motifs_to_store() -> Result<(), HbacError>{
    //     let runs = vec![
    //         ("2024-02-14T17:36:36Z", "seizure"),
    //         ("2024-02-15T17:14:09Z", "lpd"),
    //         ("2024-02-16T06:34:04Z", "gpd"),
    //         ("2024-02-16T11:31:30Z", "lrda"),
    //         ("2024-02-16T16:29:11Z", "grda"),
    //         ("2024-02-19T07:12:01Z", "other"),
    //     ];
    //
    //     let mut store = SqlStore::new("hbac_imp.db3")?;
    //
    //     for (dir, desc) in runs {
    //         let (_, eeg_ids) = read_eeg_index(format!("../../data_down/14_02_results/{}", dir).as_str())?;
    //
    //         let run_id = store.save_run(format!("{}/csv motifs import", desc).as_str(), &eeg_ids)?;
    //
    //         for channel in 0..20 {
    //             let r = read_motif_to_iter_imp(format!("../../data_down/14_02_results/{}/motifs_{}.csv", dir, channel).as_str())?;
    //
    //             for chunk in r.chunks(500) {
    //                 store.save_slices(run_id, channel, chunk.to_vec());
    //             }
    //
    //             store.set_motifs(run_id, channel);
    //
    //             println!("desc={}, channel={}, run_id={}, r.len()={}", desc, channel, run_id, r.len());
    //         }
    //     }
    //     Ok(())
    // }
}