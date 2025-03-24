use std::cmp::max;
use std::sync::atomic::Ordering;
use simplelog::{ColorChoice, CombinedLogger, Config, TermLogger, TerminalMode};
use log::{debug, error, LevelFilter};
use std::env;
use std::env::VarError;
use std::fmt::Debug;
use std::ops::{Add, Div, Sub};
use std::str::FromStr;
use num_traits::{Num, Signed, ToPrimitive};
use num_traits::real::Real;
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


    }
}

pub fn set_panic_hook() {
    std::panic::set_hook(Box::new(|panic_info| {
        let backtrace = std::backtrace::Backtrace::capture();
        error!("Panic info: {:?}", panic_info);
        for frame in backtrace.frames() {
            error!("{:?}", frame);
        }

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
                // debug!("0. x={x:?} y={y:?}");

                false
            } else if let (Some(xf), Some(yf), Some(pf)) = (ToPrimitive::to_f64(&x), ToPrimitive::to_f64(&y), ToPrimitive::to_f64(&p)) {
                let diff = (xf - yf).abs() / ( (xf.abs() + yf.abs())/2.);
                // debug!("1. x={x:?} y={y:?} diff={diff:?}");
                diff  > pf
            } else {
                let diff = (x - y).abs();
                // debug!("1. x={x:?} y={y:?} diff={diff:?}");

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
            0.5
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