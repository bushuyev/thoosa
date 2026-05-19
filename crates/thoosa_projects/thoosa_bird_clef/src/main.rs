#![recursion_limit = "1024"] //for deep async future type - to avoid "error: queries overflow the depth limit"
#![allow(deprecated)]
#![allow(unused_imports, unused_variables, unused_mut, noop_method_call)]

use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use crate::data::{ogg_to_tensor, resource_to_tensor};
use crate::labels::LABELS;
use log::{debug, error, info, warn};
use std::time::Instant;
use futures::{StreamExt, TryStreamExt};
use parking_lot::RwLock;
use tch::{Cuda, Device, IndexOp, Kind, Tensor};
use tch::Kind::{BFloat16, Float};
use tch::kind::FLOAT_CPU;
use tokio::task::JoinHandle;
use thoosa_common::dims::LabelProbability;
use thoosa_common::runner::common::{mock_run_ctx, RunCtx};
use thoosa_common::runner::common::cachable::DevicesWrap;
use thoosa_common::runner::common::runnable::AsyncStage;
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::InferArgs;
use thoosa_common::utils::{kick_register_cuda_hooks, pad_to_2, to_vec_f32};
use thoosa_common::ws::Workspace;
use thoosa_main::cli::{run_cli, RunAllArgs};
use thoosa_main::{ChannelsData2, RawTensorLoader};
use thoosa_main::tasks::apply_label_nets::ApplyLabelNets;
use thoosa_shared::{collect_futures_with_progress, mono_key, StreamBuffering, ThoError};
use thoosa_shared::comp_id::SynKey;
use thoosa_shims::context::ThoCudaContext;
use thoosa_store::rusqlite::Transaction;
use thoosa_store::{LimitBy, SqlStore, Store};
use thoosa_with_data::with_data;
use crate::run_all::{run_all, run_all_config_0, run_labels_config_0};

mod data;
mod labels;
mod run_all;

const SLOTS_PER_FILE: usize = 12;
const SLOT_SECONDS: usize = 5;
const EXPECTED_LABELS: usize = 234;

fn main() -> Result<(), ThoError>{
    info!("bird_clef main");
    if let Ok(thoosa_num_interop_threads)= std::env::var("THOOSA_NUM_INTEROP_THREADS").map_err(|e|ThoError::from(e)).and_then(|v|v.parse::<u32>().map_err(|e|ThoError::from(e))){
        info!("thoosa_num_interop_threads={thoosa_num_interop_threads}");
        tch::set_num_interop_threads(thoosa_num_interop_threads as i32);
    };

    let worker_threads = std::env::var("THOOSA_WORKER_THREADS").map_err(|e|ThoError::from(e)).and_then(|v|v.parse::<usize>().map_err(|e|ThoError::from(e))).ok();
    let blocking_threads = std::env::var("THOOSA_BLOCKING_THREADS").map_err(|e|ThoError::from(e)).and_then(|v|v.parse::<usize>().map_err(|e|ThoError::from(e))).ok();

    let t1 = Instant::now();

    let cpu_num = std::thread::available_parallelism().map(|nz| nz.get()).unwrap_or(2);

    // Navigation note: each Tokio worker pins CUDA device 0 before any stage runs.
    // CUDA init failures from BirdCLEF usually happen here or later in GlobalCtx::init_devices inside run_cli.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .on_thread_start(|| {
            if tch::Cuda::is_available() {
                //runs cuCtxSetCurrent
                if let Err(err) = ThoCudaContext::new(0) {
                    panic!("Failed to create context: {err:?}"); //TODO device
                }
            }
        })
        .worker_threads(worker_threads.unwrap_or(cpu_num/2))
        .max_blocking_threads(blocking_threads.unwrap_or(cpu_num/2))
        .enable_all()
        .build()
        .map_err(|err| {
            error!("main: {err:?}");
            ThoError::from_error("tokio runtime build failed", err)
        })?;

    let run_result = rt.block_on(async move {
        //gpt Navigation note: run_cli is the shared thoosa entry point. The arguments below are the
        //gpt BirdCLEF adapters: label import, OGG->tensor loading, dataset-set seeding, run-all orchestration, and optional inference.
        run_cli(
            |source_path| data::labeled_resourses(source_path),
            resource_to_tensor,
            |t: &Transaction|{
                SqlStore::save_all_sets(&t)?;

                //gpt This import-resources hook seeds named resource sets. Later CLI args like --find-resources=H1000000 resolve through these generated sets.
                //resources used in previous N-set not included, sorted by size
                SqlStore::save_leaky_data_sets("A", LimitBy::Count(1), None, 0.6, 0.2, 1.0, &t)?;
                SqlStore::save_leaky_data_sets("A", LimitBy::Count(3), None, 0.6, 0.2, 1.0, &t)?;
                SqlStore::save_leaky_data_sets("A", LimitBy::Count(6), None, 0.6, 0.2, 1.0, &t)?;

                SqlStore::save_leaky_data_sets("B", LimitBy::Count(3), Some(32000), 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("B", LimitBy::Count(6), Some(32000), 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("B", LimitBy::Count(10), Some(32000), 0.6, 0.2, 0.9, &t)?;

                SqlStore::save_leaky_data_sets("C", LimitBy::Count(10), None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("C", LimitBy::Count(20), None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("C", LimitBy::Count(30), None, 0.6, 0.2, 0.9, &t)?;

                SqlStore::save_leaky_data_sets("D", LimitBy::Count(100), None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("E", LimitBy::Count(200), None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("F", LimitBy::Count(300), None, 0.6, 0.2, 0.9, &t)?;

                SqlStore::save_leaky_data_sets("G", LimitBy::Length(500_000), None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("H", LimitBy::Length(1_000_000), None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("I", LimitBy::Length(3_000_000), None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("J", LimitBy::Length(10_000_000), None, 0.6, 0.2, 0.9, &t)?;


                //shuffles randomly
                SqlStore::save_leaky_data_sets("X", LimitBy::None, None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("Y", LimitBy::None, None, 0.6, 0.2, 0.9, &t)?;
                SqlStore::save_leaky_data_sets("Z", LimitBy::None, None, 0.6, 0.2, 0.9, &t)?;

                Ok(())
            },
            async |
                ws: &Arc<RwLock<Box<dyn Workspace>>>,
                store: &Arc<Box<dyn Store>>,
                loader: &Arc<RawTensorLoader<RunCtxImpl>>,
                root_ctx: RunCtxImpl,
                run_all_args: &RunAllArgs
            |{
                debug!("bird_clef run all");
                let run_mode = run_all_config_0(run_all_args);
                // let run_mode = run_labels_config_0(run_all_args);

                //gpt run-all starts here after run_cli has opened the workspace/store and built the RawTensorLoader over saved *.pt tensors.
                run_all(
                    ws,
                    store,
                    loader,
                    root_ctx,
                    run_all_args,
                    run_mode
                ).await?;

                Ok(())
            },
            &LABELS,
            1,
            u32::MAX,
            32000,
            Some(async |params:&InferArgs, ws:&Arc<RwLock<Box<dyn Workspace>>>, store:&Arc<Box<dyn Store>>,|{
                debug!("bird_clef: infer");
                infer_labels(params, ws, store).await

            }),
        ).await
    });


    match run_result {
        Ok(_) => {
            info!("All done in {} mls", t1.elapsed().as_millis());
        }

        Err(e) => {
            error!("failed with {:?} in {} mls", e, t1.elapsed().as_millis());
        }
    }
    std::process::exit(0x0000); //for jupyter
}


async fn infer_labels(args:&InferArgs, ws:&Arc<RwLock<Box<dyn Workspace>>>, store:&Arc<Box<dyn Store>>) -> Result<(), ThoError>{
    let mut ctx = RunCtxImpl::new(0, "root");


    let mut pathnames = Vec::with_capacity(600);

    for (entry_i, entry) in fs::read_dir(&args.input_dir)?.take(args.limit.unwrap_or(usize::MAX)).enumerate() {
        let entry = entry?;
        let path = entry.path();
        let is_ogg = path.extension().map(|e| e.eq("ogg")).unwrap_or(false);
        debug!("path: {path:?} is_file={} is_ogg={}", path.is_file(), is_ogg);

        if path.is_file() && is_ogg {
            debug!("path: {path:?}");

            pathnames.push(path.to_string_lossy().to_string());
        } else {
            warn!("path {path:?} is not an ogg");
        }
    }

    let num_files = pathnames.len();

    let mut oggs_t = Tensor::empty([num_files as i64, 1920000], (Kind::BFloat16, Device::Cpu));//1920000 = 32000 × 60 one minute
    for (entry_i, ogg_filename) in pathnames.iter().enumerate() {
        let ogg_t = ogg_to_tensor(ogg_filename, BFloat16)?;
        debug!("ogg_filename: {ogg_filename:?} loaded={ogg_t:?} requires_grad={}", ogg_t.requires_grad());

        oggs_t.get(entry_i as i64).f_copy_(&pad_to_2(&ogg_t.view(-1), &[1920000], true, false)?.0)?;
    }

    debug!("loaded {:?} ", oggs_t);
    let oggs_tt = Arc::new(DevicesWrap::wrap(ThoTensor::<ChannelsData2, SynKey>::new(mono_key!(0, "all".to_string()).into(), oggs_t.view((1, -1)))?));//
    let label_models = Arc::new(ws.read().label_models()?);


    let mut label_results: HashMap<String, Vec<f32>> = HashMap::new();
    for (label_i, label) in LABELS.iter().enumerate()/*.skip(0).take(20)*/ {

        let apply_label_nets = load_apply_label_nets_model(Arc::clone(ws), Arc::clone(store), ctx.branch("load_apply_label_nets_model".into(), false), Arc::clone(&label_models), &LABELS[label_i], num_files * SLOTS_PER_FILE, args.motif_slices_step as u32).await;


        let apply_label_nets = match apply_label_nets {
            Ok(v) => {
                if v.num_nets() == 0 {
                    warn!("no nets for {label}");
                    label_results.insert(label.to_string(), vec![]);
                    continue
                }
                debug!("got v={v:?} fro {label}");
                v
            }
            Err(e) => {
                warn!("failed to load net for {label}, got e={e}");
                label_results.insert(label.to_string(), vec![]);
                continue
            }
        };

        debug!("label={label:?}  apply_label_nets loaded");//path={path:?}

        match Arc::clone(&apply_label_nets).run(Arc::clone(&oggs_tt), ctx.branch(Some(&format!("ctx for {label}")), false)).await{
            Ok(results) => {
                with_data!(ctx.branch(Some(&format!("ctx for {label}")), false), results; |ctx, results:&ThoTensor<LabelProbability, SynKey>| {

                    let probs = to_vec_f32(results.as_ref());
                    info!("{label_i} label={label:?}");// slice={slice} path={path:?}{entry_i}
                    debug!("{label_i} label={label:?}  probs={probs:?}");// slice={slice} path={path:?}{entry_i}
                    assert_eq!(probs.len(), num_files * 12, "label={label} probs.len={}", probs.len()); //600 files * minute

                    label_results.insert(label.to_string(), probs);


                    Ok(())
                })?;

            }
            Err(e) => {
                warn!("got e={e} for {label}");
                label_results.insert(label.to_string(), vec![]);
            }
        };

        debug!("label={label:?}  inferred");

    }
    debug!("label_results={}", label_results.len());

    debug!("pathnames={pathnames:?}");
    debug!("label_results={label_results:?}");

    write_birdclef_submission_csv(
        &pathnames,
        &label_results,
        &LABELS,
        "submission.csv"
    )?;
    Ok(())
}

async fn load_apply_label_nets_model(ws: Arc<RwLock<Box<dyn Workspace>>>, store: Arc<Box<dyn Store>>, ctx:  RunCtxImpl, label_models: Arc<HashMap<String, Vec<(PathBuf, PathBuf)>>>, label: &str, batch_in_chunks: usize, motif_slices_step: u32) -> Result<Arc<ApplyLabelNets<RunCtxImpl>>, ThoError> {
    let label_model = label_models.get(&label.to_string());
    let run_ids = store.list_label_nets_run_ids(label).await?;
    debug!("load_apply_label_nets_model: label={label:?} label_model={label_model:?} run_ids={run_ids:?}");

    let apply_label_nets = Arc::new(ApplyLabelNets::<RunCtxImpl>::new(
        &ws,
        &store,
        run_ids,
        ctx,
        motif_slices_step,
        batch_in_chunks,
    ).await?);
    Ok(apply_label_nets)
}


pub fn write_birdclef_submission_csv(
    pathnames: &[String],
    label_results: &HashMap<String, Vec<f32>>,
    labels: &[&str],
    output_csv: impl AsRef<Path>,
) -> Result<(), ThoError> {
    debug!("write_birdclef_submission_csv: labels.len={}", labels.len());

    if labels.len() != EXPECTED_LABELS {
        return Err(ThoError::new(format!("expected {EXPECTED_LABELS} labels, got {}", labels.len())));
    }

    let expected_rows = pathnames.len() * SLOTS_PER_FILE;

    // Validate once and keep refs in fixed LABELS order.
    let mut label_probs: Vec<(&str, &Vec<f32>)> = Vec::with_capacity(labels.len());
    let zero_vec = vec![0.0; expected_rows];

    for &label in labels {
        let probs = label_results.get(label).expect(&format!("missing predictions for label {label:?}"));

        if probs.is_empty() {

            label_probs.push((label, &zero_vec));
            continue;
        }

        if probs.len() != expected_rows {
            let msg = format!("label {label:?} has {} predictions, expected {expected_rows} = {} files * {SLOTS_PER_FILE} slots", probs.len(), pathnames.len() );
            error!("{}", msg);
            return Err(ThoError::new(msg));
        }

        label_probs.push((label, probs));
    }

    debug!("write_birdclef_submission_csv: output_csv={:?}", output_csv.as_ref());
    let file = File::create(output_csv)?;

    let mut w = BufWriter::new(file);

    // Header: row_id,<label1>,<label2>,...
    write_csv_field(&mut w, "row_id")?;
    for &(label, _) in &label_probs {
        write!(w, ",")?;
        write_csv_field(&mut w, label)?;
    }
    writeln!(w)?;

    debug!("write_birdclef_submission_csv: pathnames.len={}", pathnames.len());
    // Rows:
    // file_i slot 0 => stem_5
    // file_i slot 1 => stem_10
    // ...
    // file_i slot 11 => stem_60
    for (file_i, pathname) in pathnames.iter().enumerate() {
        let stem = Path::new(pathname)
            .file_stem()
            .and_then(|s| s.to_str())
            .expect(&format!("could not extract UTF-8 file stem from path {pathname:?}"));

        for slot_i in 0..SLOTS_PER_FILE {
            let row_i = file_i * SLOTS_PER_FILE + slot_i;
            let end_time = (slot_i + 1) * SLOT_SECONDS;

            let row_id = format!("{stem}_{end_time}");
            write_csv_field(&mut w, &row_id)?;

            for &(_, probs) in &label_probs {
                let p = probs[row_i];

                if !p.is_finite() {
                    return Err(ThoError::new(&format!("non-finite probability for row_id={row_id:?}, row_i={row_i}: {p}")));
                }

                write!(w, ",{p}")?;
            }

            writeln!(w)?;
        }
    }

    w.flush()?;

    debug!("write_birdclef_submission_csv done");//path={path:?}

    Ok(())
}

fn write_csv_field<W: Write>(w: &mut W, s: &str) -> std::io::Result<()> {
    let needs_quotes = s.bytes().any(|b| matches!(b, b',' | b'"' | b'\n' | b'\r'));

    if !needs_quotes {
        write!(w, "{s}")?;
        return Ok(());
    }

    write!(w, "\"")?;
    for ch in s.chars() {
        if ch == '"' {
            write!(w, "\"\"")?;
        } else {
            write!(w, "{ch}")?;
        }
    }
    write!(w, "\"")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::Arc;
    use log::{debug, info, set_max_level};
    use log::LevelFilter::Trace;
    use parking_lot::RwLock;
    use polars::prelude::{AnyValue, CsvReadOptions, PolarsError, PolarsResult, SerReader};
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_main::tasks::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::train_nets::InferArgs;
    use thoosa_common::ws::{FsWorkspace, WsMockBuilder};
    use thoosa_shared::perf_logging::{format_perf_map, PERF};
    use thoosa_shared::ThoError;
    use thoosa_shared::val::Val;
    use thoosa_store::{MockStoreBuilder, SqlStore};
    use thoosa_store::rusqlite::fallible_iterator::FallibleIterator;
    use crate::{infer_labels, write_birdclef_submission_csv};
    use crate::labels::LABELS;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn heavy_test_infer_labels() -> Result<(), ThoError> {
        GlobalCtx::init_devices(1);
        //TODO
        // let ws = Arc::new(RwLock::new(
        //     WsMockBuilder::new(None, 1, 100, u32::MAX)
        //
        //     .build(),
        // ));
        //
        // let store = Arc::new(
        //     MockStoreBuilder::new()
        //
        //     .build(),
        // );

        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            "/home/bu/tmp/ws_clef",
            1,
            0,//not used
            32000,
        )?));
        let store = Arc::new(SqlStore::from_path(ws.read().store_path()).await?);

        let params = InferArgs::new(0, "test_data/train_soundscapes".to_string(), "".to_string(), 1000);

        // let params = InferArgs::new(0, "/home/bu/dvl/data/kaggle/birdclef-2026/train_soundscapes".to_string(), "".to_string(), 1000);

        infer_labels(&params, &ws, &store).await?;


        info!("\n{}", format_perf_map(&PERF));

        Ok(())
    }

    #[test]
    fn test_write_birdclef_submission_csv() -> Result<(), ThoError>{
        set_max_level(Trace);
        let inputs = ["input_a".into(), "input_b".into()];
        write_birdclef_submission_csv(
            &inputs,
            &HashMap::from_iter(
                LABELS
                    .iter()
                    .enumerate()
                    .map(|(label_i, label)|(
                            label.to_string(),
                            inputs.iter().enumerate().flat_map(|(input_i, _)| (0..12).map(move |chunk| {
                                let v= (
                                    1_000_000 +
                                        label_i * 1_000 +
                                        input_i * 1_00 +
                                        chunk
                                    );
                                debug!("test_write_birdclef_submission_csv: label_i={label_i} input_i={input_i} chunk={chunk}, v={v}");
                                v as f32
                            })).collect()
                        )
                    )
            ),
            &LABELS,
            "/tmp/sub.csv"
        )?;


        let results = read_csv_with_polars("/tmp/sub.csv")?;

        for (raw_i, raw) in results.iter().skip(1).enumerate() {
            for (column_i, val) in raw.iter().skip(1).enumerate() {
                debug!("test_write_birdclef_submission_csv: raw_i={raw_i} column_i={column_i} val={val:?}");

                let v = match val {
                    Val::Real(v) => *v as u64,
                    Val::Int(v) => *v as u64,
                    Val::UInt(v) => *v as u64,
                    Val::Str(v) => v.parse::<u64>()?,
                    Val::Bool(_) => {
                        panic!("no bools expected")
                    }
                };
                let label_i = column_i as u64;
                let input_i =  raw_i as u64 / 12;
                let chunk = raw_i as u64 % 12;
                debug!("test_write_birdclef_submission_csv: raw_i={raw_i} column_i={column_i} v={v:?} label_i={label_i} input_i={input_i} chunk={chunk}");

                assert_eq!(
                  1_000_000u64 +
                    label_i * 1_000 +
                    input_i * 1_00 +
                    chunk,
                    v
                );
            }
        }

        Ok(())
    }

    pub fn read_csv_with_polars<P: AsRef<Path>>(path: P) -> PolarsResult<Vec<Vec<Val>>> {
        let df = CsvReadOptions::default()
            .with_has_header(false)
            .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))?
            .finish()?;

        (0..df.height())
            .map(|row_i| {
                df.get(row_i)
                    .expect("row index is in bounds")
                    .into_iter()
                    .map(any_value_to_val)
                    .collect()
            })
            .collect()
    }

    fn any_value_to_val(v: AnyValue<'_>) -> PolarsResult<Val> {
        Ok(match v {
            AnyValue::Boolean(v) => Val::Bool(v),

            AnyValue::Int8(v) => Val::Int(v as i64),
            AnyValue::Int16(v) => Val::Int(v as i64),
            AnyValue::Int32(v) => Val::Int(v as i64),
            AnyValue::Int64(v) => Val::Int(v),

            AnyValue::UInt8(v) => Val::UInt(v as u32),
            AnyValue::UInt16(v) => Val::UInt(v as u32),
            AnyValue::UInt32(v) => Val::UInt(v),
            AnyValue::UInt64(v) => {
                if let Ok(v) = u32::try_from(v) {
                    Val::UInt(v)
                } else {
                    return Err(PolarsError::ComputeError(
                        "u64 value does not fit into Val::UInt(u32)".into(),
                    ));
                }
            }

            AnyValue::Float32(v) => Val::Real(v as f64),
            AnyValue::Float64(v) => Val::Real(v),

            AnyValue::String(v) => {
                // Val::Int(v.parse::<i64>().expect(&format!("can't parse '{v}'")))
                Val::Str(v.to_owned())
            },

            other => {
                return Err(PolarsError::ComputeError(
                    format!("unsupported CSV value for Val: {other:?}").into(),
                ));
            }
        })
    }

}
