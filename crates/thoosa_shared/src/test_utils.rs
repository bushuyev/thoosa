use std::sync::atomic::Ordering;
use simplelog::{format_description, ColorChoice, CombinedLogger, Config, ConfigBuilder, TermLogger, TerminalMode, ThreadLogMode, ThreadPadding};
use log::{debug, error, trace, LevelFilter};
use std::env;
use std::env::VarError;
use std::fmt::Debug;
use std::ops::{Add, Div, Sub};
use std::str::FromStr;
use num_traits::{Num, Signed, ToPrimitive};
use crate::LOG_INIT;



pub fn init_test_logging() {

    if !LOG_INIT.load(Ordering::Relaxed) {

        LOG_INIT.store(true, Ordering::Relaxed);
        CombinedLogger::init(

            vec![
                TermLogger::new(
                    LevelFilter::Trace,//to let set trace level from tests with set_max_level
                    ConfigBuilder::new()
                        // Optional: pick the exact date/time layout you want
                        .set_time_format_custom(format_description!("[year]-[month]-[day] [hour]:[minute]:[second]"))
                        // Print the *name* (or the ID if unnamed); use `Both` to fall back automatically
                        .set_thread_mode(ThreadLogMode::Names)
                        // Show it for every record (Error is the highest-priority level, so this means “always”)
                        .set_thread_level(LevelFilter::Error)
                        // Optional: pad so columns line up nicely
                        .set_thread_padding(ThreadPadding::Right(12))
                        .build(),
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


    }
}

pub fn set_panic_hook() {
    // std::panic::set_hook(Box::new(|panic_info| {
    //     let backtrace = std::backtrace::Backtrace::capture();
    //     error!("Panic info: {:?}", panic_info);
    //     for frame in backtrace.frames() {
    //         error!("{:?}", frame);
    //     }
    //
    // }));
    let default_hook = std::panic::take_hook();

    std::panic::set_hook(Box::new(move |info| {
        debug!("Panic info: {:?}", info);
        // 1. dump the payload (works for both &str and String)
        if let Some(s) = info.payload().downcast_ref::<&str>() {
            log::error!("panic payload: {s}");
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            log::error!("panic payload: {s}");
        }

        // 2. optional extra back-trace
        // log::error!("{:?}", std::backtrace::Backtrace::force_capture());
        let backtrace = std::backtrace::Backtrace::force_capture();
        for frame in backtrace.frames() {
            debug!("{:?}\n", frame);
        }
        log::logger().flush();   // make sure async loggers flush

        // 3. hand control back to the standard reporter so you
        //    still get the nicely formatted panic + back-trace
        default_hook(info);
    }));
}

pub fn are_close<T>(a: &[T], b: &[T], p: T) -> bool
where
    T: Num + Signed + PartialOrd + Copy + ToPrimitive + Debug,
{
    assert_eq!(a.len(), b.len());

    !a.iter()
        .zip(b.iter())
        .any(|(&x, &y)| {
            if x == y {
                trace!("0. x={x:?} y={y:?}");

                false
            } else if let (Some(xf), Some(yf), Some(pf)) = (ToPrimitive::to_f64(&x), ToPrimitive::to_f64(&y), ToPrimitive::to_f64(&p)) {
                let diff = (xf - yf).abs() / ( (xf.abs() + yf.abs())/2.);
                trace!("1. x={x:?} y={y:?} diff={diff:?}");
                diff  > pf
            } else {
                let diff = (x - y).abs();
                trace!("1. x={x:?} y={y:?} diff={diff:?}");

                diff > p
            }
        })
}

#[cfg(test)]
mod tests {
    use crate::test_utils::are_close;

    #[test]
    fn test_are_close(){
        assert!(are_close(
            &vec![1, 2, 3, 4, 5],
            &vec![1, 2, 3, 4, 5],
            0
        ));

        assert!(are_close(
            &vec![1, 2, 3, 4, 5],
            &vec![2, 3, 4, 5, 6],
            1
        ));

        assert!(are_close(
            &vec![1., 2., 3., 4., 5.],
            &vec![2., 3., 4., 5., 6.],
            0.7
        ));

        assert!(!are_close(
            &vec![1., 2., 3., 4., 5.],
            &vec![2., 3., 4., 5., 6.],
            0.1
        ));


        assert!(!are_close(
            &vec![1, 2, 3, 4, 5],
            &vec![4, 4, 4, 4, 4],
            1
        ));


    }
}
