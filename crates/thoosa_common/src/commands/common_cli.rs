use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::OpenOptions;
use std::io::Write;
use std::ops::ControlFlow;
use std::ops::ControlFlow::{Break, Continue};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use clap::Subcommand;
use log::{debug, info, warn};
use parking_lot::RwLock;
use tch::Tensor;
use tokio::runtime::Handle;
use tokio::sync::watch;
use tokio::{select, time};
use tokio::time::MissedTickBehavior;
use torch_sys::at_grad_set_enabled;
use tracing::Instrument;
use thoosa_shared::perf_logging::{install_panic_hook, SyncMakeWriter, init_tracing, format_perf_map, PERF};
use thoosa_shared::run_configs::{run_id, RUN_ID};
use thoosa_shared::ThoError;
use thoosa_shared::tracked_await::{snapshot_not_ready, track_await};
use thoosa_shims::cpp_log::init_cpp_logging;
use thoosa_store::rusqlite::Transaction;
use thoosa_store::Store;
use crate::args::{CommonArgs, StageCommand};
use crate::ipc::cuda_process::CUDA_VISIBLE_DEVICES;
use crate::progress_socket::CollectEtaProgressSocketServer;
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::run_ctx::RunCtxImpl;

use crate::utils::list_devices;
use crate::ws::{Workspace, WsConfigs};
use crate::version::VERSION;

///main process function: sets up logging, spawns IPC Worker if needed, strats CPU purger
pub async fn run_common_cli<C, CS, R, W>(
    args: CommonArgs<C>,
    store:Arc<CS>,
    ws: Arc<RwLock<Box<W>>>,
    resource_to_tensor: R,
    cdt_for_dir: String,
) -> anyhow::Result<()>
where
    C: Subcommand + StageCommand<Store=Arc<CS>, Workspace=W, Ctx=RunCtxImpl>+ Clone + std::fmt::Debug + Debug,
    CS:?Sized + Store,
    W: Workspace + ?Sized,
    R: Fn(&dyn AsRef<Path>, &str) -> anyhow::Result<Tensor> + Sync + Send,
{
    unsafe {
        at_grad_set_enabled(0);
    };

    let log_filename = ws.read()
        .log_path()
        .join(format!("thoosa_{}.log", cdt_for_dir).to_owned());


    let perf_path = if args.write_perf_log_file {
        Some(
            ws.read()
                .log_path()
                .join(format!("thoosa_perf_{}.log", cdt_for_dir).to_owned()),
        )
    } else {
        None
    };

    install_panic_hook(SyncMakeWriter::new(OpenOptions::new()
        .create(true)
        .append(true)
        .open( ws.read()
            .log_path()
            .join(format!("thoosa_panic_{}.log", cdt_for_dir)))?));


    let _logging_guards = init_tracing(
        args.term_logger_level,
        args.term_logger_extra_fields,
        args.write_logger_level,
        args.write_logger_extra_fields,
        log_filename,
        "info",
        perf_path,
        None,
    )?;

    init_cpp_logging();

    let _collect_eta_progress_server = match CollectEtaProgressSocketServer::start_default() {
        Ok(server) => Some(server),
        Err(err) => {
            warn!("run_cli: failed to start collect ETA progress socket server: {err}");
            None
        }
    };

    RUN_ID.set(match args.run_id{
        None => {
            // TODO throw error if command doesn't support new run_id - inference, export etc
            track_await!(store.save_run(format!("{args:?}"))).await?
        }
        Some(run_id) => run_id
    }).expect("can't set run_id");

    let devices = list_devices();

    let cpu_num = std::thread::available_parallelism()?;

    info!(
        "VERSION: {VERSION} devices={devices:?}, base_path={:?} cpus={cpu_num} tch::get_num_interop_threads={}, tch::get_num_threads={} run_id={}",
        args.ws_path,
        tch::get_num_interop_threads(),
        tch::get_num_threads(),
        run_id()
    );
    std::io::stdout().flush()?;

    debug!("run_common_cli: args={args:?}");

    if args.ipc {
        info!("Running in IPC mode");

        if !std::env::var(CUDA_VISIBLE_DEVICES).map(|v|v.eq("-1")).unwrap_or(false) {
            // Startup failures must reach the CLI result so automation sees failure
            // and stack-owned resources are cleaned up normally.
            anyhow::bail!("IPC mode requires CUDA_VISIBLE_DEVICES=-1");
        }

        GlobalCtx::init_devices_ipc(2, vec![
            format!("--ws-path={}", args.ws_path),
            format!("--term-logger-level={}", args.term_logger_level),
            format!("--write-logger-level={}", args.write_logger_level)
        ]).await?;
    } else {
        // kick_register_cuda_hooks();
        GlobalCtx::init_devices(2)?;
    }


    let (stop_cpu_purge_tx, mut rx) = watch::channel(false);
    let metrics = Handle::current().metrics();
    let workers = metrics.num_workers();
    let mut prev_poll = vec![0u64; workers];
    let mut prev_park = vec![0u64; workers];
    let mut prev_busy = vec![std::time::Duration::ZERO; workers];

    let mut purge_cpu_worker = tokio::spawn(async move {
        let mut ticker = time::interval(Duration::from_secs(60));//TODO config
        track_await!(ticker.tick()).await;
        ticker.set_missed_tick_behavior(MissedTickBehavior::Skip);
        loop {
            select! {
                _ = ticker.tick() => {

                    #[cfg(tho_debug)]
                    {
                        info!(
                            "[tokio] alive_tasks={} global_q={} blocking_q={}",
                            metrics.num_alive_tasks(),
                            metrics.global_queue_depth(),
                            metrics.blocking_queue_depth(),
                        );
                    }
                    if log::log_enabled!(log::Level::Debug) {
                        for w in 0..workers {
                               let poll = metrics.worker_poll_count(w);
                                let park = metrics.worker_park_count(w);
                                let busy = metrics.worker_total_busy_duration(w);

                                debug!(
                                    "  w{w}: poll+{} park+{} busy+{:?}",
                                    poll - prev_poll[w],
                                    park - prev_park[w],
                                    busy - prev_busy[w],
                                );

                                prev_poll[w] = poll;
                                prev_park[w] = park;
                                prev_busy[w] = busy;
                        }

                        let tasks = snapshot_not_ready();
                        debug!("TASKS pending={}", tasks.len());
                        for task in tasks {
                            debug!("  {task}");
                        }
                    }

                    // CPU cache maintenance uses the cache registry's own locks;
                    // a recovering GPU must not stop unrelated CPU work.
                    debug!("run_cli: purge_cpu tick");
                    GlobalCtx::global().purge_cpu(10).instrument(tracing::debug_span!("run_cli/4", file = file!(), line = line!())).await;
                    debug!("run_cli: purge_cpu done");
                }
                _ = rx.changed() => {
                    if *rx.borrow() {
                        debug!("stopping cpu purger");
                        break;
                    }
                }
            }
        }
    }.instrument(tracing::info_span!(
        "cli::purge_cpu_worker",
        kind = "spawn",
        file = file!(),
        line = line!(),
        task = "periodically purge CPU caches and log Tokio metrics",
    )));

    let command_result = if let Some(command ) = args.command{
        let mut root_ctx = RunCtxImpl::new(1, "root_ctx");//TODO config
        debug!("run_common_cli: command={command:?}");
        // A local CUDA failure can originate in detached copy work even if the
        // command no longer awaits that request. Still terminate the application.
        tokio::select! {
            biased;
            failure = GlobalCtx::wait_for_process_failure() => Err(failure.into()),
            result = command.run(&store, &ws, root_ctx) => result,
        }
    } else { Ok(()) };



    info!("perf: \n{}", format_perf_map(&PERF));

    track_await!(GlobalCtx::global().shutdown()).await;

    let _ = stop_cpu_purge_tx.send(true);
    // A purge may be waiting for work in the poisoned local process. It must
    // not delay the application shutdown requested by the recovery coordinator.
    tokio::select! {
        _ = GlobalCtx::wait_for_process_failure() => {
            purge_cpu_worker.abort();
            let _ = track_await!(purge_cpu_worker).await;
        }
        _ = track_await!(&mut purge_cpu_worker) => {}
    }


    info!("run_id {} done", run_id());

    if let Some(failure) = GlobalCtx::process_failure() { return Err(failure.into()); }
    command_result
}
