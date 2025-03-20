use std::sync::atomic::Ordering;
use simplelog::{ColorChoice, CombinedLogger, Config, TermLogger, TerminalMode};
use log::{debug, error, LevelFilter};
use std::env;
use std::env::VarError;
use std::str::FromStr;
use crate::LOG_INIT;



pub fn init_test_logging() {

    if !LOG_INIT.load(Ordering::Relaxed) {

        LOG_INIT.store(true, Ordering::Relaxed);
        CombinedLogger::init(

            vec![
                TermLogger::new(
                    LevelFilter::Trace,//to let set trace level from tests with set_max_level
                    Config::default(),
                    TerminalMode::Mixed,
                    ColorChoice::Auto
                )
            ]
        ).unwrap_or_else(|e| {
            debug!("CombinedLogger::init failed: {:?}", e);
        });

        log::set_max_level(
            env::var("RUST_LOG").and_then(|v| LevelFilter::from_str(v.as_str()).map_err(|e| {
                println!("Can't find loglevel: {}", e);
                VarError::NotPresent
            })).unwrap_or(LevelFilter::Debug)
        );

        env::set_var("POLARS_FMT_MAX_COLS", "100");
        env::set_var("POLARS_FMT_MAX_ROWS", "100");

        std::panic::set_hook(Box::new(|panic_info| {
            let backtrace = std::backtrace::Backtrace::capture();
            error!("Panic info: {:?}", panic_info);
            error!("Backtrace: {:?}", backtrace);
        }));
    }
}
