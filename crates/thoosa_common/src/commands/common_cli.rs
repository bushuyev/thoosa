use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::OpenOptions;
use std::io::Write;
use std::ops::ControlFlow;
use std::ops::ControlFlow::{Break, Continue};
use std::path::Path;
use std::sync::Arc;
use clap::Subcommand;
use log::{debug, info, warn};
use parking_lot::RwLock;
use tch::Tensor;
use torch_sys::at_grad_set_enabled;
use thoosa_shared::perf_logging::{install_panic_hook, SyncMakeWriter, init_tracing};
use thoosa_shared::run_configs::{run_id, RUN_ID};
use thoosa_shared::ThoError;
use thoosa_shared::tracked_await::track_await;
use thoosa_shims::cpp_log::init_cpp_logging;
use thoosa_store::rusqlite::Transaction;
use thoosa_store::Store;
use crate::args::{CommonArgs, StageCommand};
use crate::ipc::cuda_process::CUDA_VISIBLE_DEVICES;
use crate::progress_socket::CollectEtaProgressSocketServer;
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::train_nets::InferArgs;
use crate::utils::list_devices;
use crate::ws::{Workspace, WsConfigs};
use crate::version::VERSION;
pub async fn run_common_cli<C, CS, R>(
    args: CommonArgs<C>,
    store:Arc<CS>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    resource_to_tensor: R,
    cdt_for_dir: String,
) -> Result<ControlFlow<(), ()>, ThoError>
where
    C: Subcommand + StageCommand<Store=Arc<CS>, Ctx=RunCtxImpl>+ Clone + std::fmt::Debug + Debug,
    CS:?Sized + Store,
    R: Fn(&dyn AsRef<Path>, &str) -> Result<Tensor, ThoError> + Sync + Send,
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
            // TODO
            // if let Some(Infer(_) | Export(_)) = &old_commands {
            //     panic!("No run_id")
            // }
            track_await(store.save_run(format!("{args:?}"))).await?
        }
        Some(run_id) => run_id
    }).expect("can't set run_id");

    let devices = list_devices();

    let cpu_num = std::thread::available_parallelism()?;

    //TODO torch.backends.cuda.matmul.allow_tf32 = True
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
            println!("Run with CUDA_VISIBLE_DEVICES=-1 in IPC mode");
            std::process::exit(0);
        }

        thoosa_common::runner::global_ctx::GlobalCtx::init_devices_ipc(2, vec![
            format!("--ws-path={}", args.ws_path),
            format!("--term-logger-level={}", args.term_logger_level),
            format!("--write-logger-level={}", args.write_logger_level)
        ]).await?;
    } else {
        // kick_register_cuda_hooks();
        thoosa_common::runner::global_ctx::GlobalCtx::init_devices(2)?;
    }


    if let Some(command ) = args.command{
        let mut root_ctx = RunCtxImpl::new(1, "root_ctx");//TODO config
        debug!("run_common_cli: command={command:?}");
        command.run(&store, &ws, root_ctx).await?;

        return Ok(Break(()))
    }

    Ok(Continue(()))
}