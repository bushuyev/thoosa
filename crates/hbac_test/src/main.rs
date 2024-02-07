extern crate core;

use std::fmt::Display;

use log::debug;
use tch::{Device, Kind, kind, Tensor};
use rand::Rng;

mod utils;

#[derive(Debug)]
pub struct GenError {
    msg:String
}

impl GenError {
    pub fn new(msg:String) -> Self {
        Self {
            msg
        }
    }
}


impl<D> From<D> for GenError where D: Display {
    fn from(value: D) -> Self {
        debug!("value={}", value);
        GenError::new(value.to_string())
    }
}

fn main() {

    let t = Tensor::from_slice(&[1., 2., 3.]);
    let t_0 = Tensor::zeros([3], (Kind::Float, Device::Cpu));
    // let t = Tensor::pad_sequence()

    println!("xxx h={:?}", Tensor::hstack(&[t.view([-1]), t_0.view([-1])]));
    println!("xxx t={:?} t_0={:?}", t.flip([0]), t_0);
}

fn z_norm(t:&Tensor, dim: i64) -> Tensor {

    let mean_t = t.mean_dim(dim, false, Kind::Float);
    let std_t = t.std(false);
    // println!("mean={:?}", vec_f32_from(&mean_t));
    // println!("std_t={:?}", vec_f32_from(&std_t));
    let z_t = (t - mean_t) / std_t;
    // println!("z_t={:?}", vec_f32_from(&z_t));
    z_t
}


mod tests {
    use std::f64::consts::PI;
    use std::iter::Map;
    use std::ops::Range;
    use std::time::Instant;

    use log::{debug, LevelFilter};
    use plotters::prelude::{BLACK, BLUE, MAGENTA};
    use plotters::style::{GREEN, RED};
    use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger};
    use tch::{Device, Kind, kind, Tensor};
    use chrono::Utc;
    use rand::Rng;
    use crate::{GenError, z_norm};

    #[ctor::ctor]
    fn init() {
        CombinedLogger::init(
            vec![
                TermLogger::new(LevelFilter::Debug, Config::default(), TerminalMode::Mixed, ColorChoice::Auto)
            ]
        ).unwrap();
    }

    #[test]
    pub fn test_z_norm() {
        assert_eq!(vec![-1.2247449, 0.0, 1.2247449], vec_f32_from(&z_norm(&Tensor::from_slice(&[1., 2., 3.]), 0i64)));
    }

    #[test]
    pub fn test_norm() -> Result<(), GenError>{
        let t = Tensor::from_slice(&[1., 2., 3.]);
        let norm = norm(&t);
        assert_eq!(1.0, vec_f32_from(&norm.dot(&norm))[0]);
        Ok(())
    }

    fn norm(t: &Tensor) -> Tensor {
        t.view([-1]) / t.dot(&t).sqrt()
    }

    #[test]
    pub fn test_dot() -> Result<(), GenError>{
        let n = 1000i64;
        let x = lin_t(n);
        // let gauss_1 = gauss_t(400., 100., n);
        let pulse_1 = pulse_v(80., 20., n, 20.);
        let random = random_v(1000, 10., 1.);
        let const_v = (0..n).map(|v|100f64).collect::<Vec<_>>();

        let sig = Tensor::from_slice(
            random.into_iter().zip(std::iter::repeat(0f64).take(100).chain(pulse_1.into_iter())).map(|(v1, v2)| 0.001 * v1+v2).collect::<Vec<f64>>().as_slice()
        );

        // // let t_10 = Tensor::from_slice((0..10).map(|i| i as f64).collect::<Vec<_>>().as_slice());
        // let t_1_snippet = gauss_1.f_slice(0, Some(0i64), Some(100i64), 1)?;
        // let t_2_snippet = gauss_1.f_slice(0, Some(0i64), Some(100i64), 1)?;
        // let t_2_1_snippet = gauss_1.f_slice(0, Some(10i64), Some(110i64), 1)?;
        //
        // let dot_0 = norm(&t_1_snippet).dot(&norm(&t_2_snippet));
        // let dot_1 = norm(&t_1_snippet).dot(&norm(&t_2_1_snippet));
        // //
        // println!("dot_0={:?}", dot_0);
        // println!("dot_1={:?}", dot_1);

        crate::utils::plot(
            vec![
                (format!("G1"), BLACK, vec_f32_from(&x).into_iter().zip(vec_f32_from(&sig)).collect::<Vec<(f32, f32)>>()),

            ], "EEG", (1000, 480), format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        ).unwrap();

        Ok(())
    }

    #[test]
    pub fn test_find_motifs_fft() -> Result<(), GenError>{
        let n = 1000i64;
        let x = lin_t(n);
        let gauss_1 = gauss_t(400., 100., n); //f.flip([0]);
        let gauss_2 = gauss_t(600., 100., n); //.flip([0]);

        let m = 200;

        let zeros = Tensor::zeros([m], (Kind::Float, Device::Cpu));

        let t1 = Instant::now();
        let mut corr:Vec<f32> = vec![];
        let mut dots:Vec<f32> = vec![];
        let mut dots_ff:Vec<f32> = vec![];
        // for d_1 in (0..n-m) {
        let d_1 = 0; {
            let t_1_snippet = gauss_1.f_slice(0, Some(d_1), Some((d_1+m) as i64), 1)?;
            let t_1_snippet_zn = z_norm(&t_1_snippet, 0);
            let t_1_snippet_ff = t_1_snippet.fft_fft(None, 0, "backward");

            for d_2 in (0..n-m) {
                let t_2_snippet = gauss_2.f_slice(0, Some(d_2 as i64), Some((d_2+m) as i64), 1)?;
                let t_2_snippet_zn = z_norm(&t_2_snippet, 0);
                let t_2_snippet_ff = t_2_snippet.fft_fft(None, 0, "backward");

                let dp = t_1_snippet_zn.dot(&t_2_snippet_zn);
                let dp_ff = t_1_snippet_ff.dot(&t_2_snippet_ff);
                // let t_2_snippet_zp = Tensor::hstack(&[t_2_snippet.view([-1]), zeros.view([-1])]);

                // corr.push(vec_f32_from(&d.abs()));
                dots.push(vec_f32_from(&dp)[0]/100.0);
                dots_ff.push(vec_f32_from(&dp_ff)[0]/200.0);
                // debug!("d.size()={:?}", dp.size());
            }
        }
        debug!("done in {}", t1.elapsed().as_millis());
        debug!("1. dots.len= {}", dots.len());
        dots.extend((0..n as usize -dots.len()).map(|_|0.).collect::<Vec<f32>>());
        debug!("2. dots.len= {}", dots.len());


        // let gauss_1_0 = gauss_1.f_slice(0, Some(300), Some(500), 1)?;

        crate::utils::plot(
            vec![
                (format!("G1"), BLACK, vec_f32_from(&x).into_iter().zip(vec_f32_from(&gauss_1)).collect::<Vec<(f32, f32)>>()),
                // (format!("G1"), BLACK, vec_f32_from(&lin_t(200)).into_iter().zip(vec_f32_from(&gauss_1_0)).collect::<Vec<(f32, f32)>>()),
                (format!("G2"), BLUE, vec_f32_from(&x).into_iter().zip(vec_f32_from(&gauss_2)).collect::<Vec<(f32, f32)>>()),
                (format!("dots"), GREEN, dots.into_iter().enumerate().map(|(i, v)|(i as f32, v)).collect::<Vec<(f32, f32)>>()),
                (format!("dots_ff"), RED, dots_ff.into_iter().enumerate().map(|(i, v)|(i as f32, v)).collect::<Vec<(f32, f32)>>())

            ], "EEG", (1000, 480), format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        ).unwrap();
        // find_motifs();
        println!("ok");

        Ok(())
    }


    #[test]
    pub fn test_find_motifs() -> Result<(), GenError>{
        let n = 1000i64;
        let x = lin_t(n);
        let gauss_1 = gauss_t(400., 100., n); //f.flip([0]);
        let gauss_2 = gauss_t(600., 100., n); //.flip([0]);

        let m = 100;

        let zeros = Tensor::zeros([m], (Kind::Float, Device::Cpu));

        let t1 = Instant::now();
        let mut corr:Vec<f32> = vec![];
        let mut dots:Vec<f32> = vec![];
        // for d_1 in (0..n-m) {
        let d_1 = 200; {
            let t_1_snippet = gauss_1.f_slice(0, Some(d_1), Some((d_1+m) as i64), 1)?;

            for d_2 in (0..n-m) {
                let t_2_snippet = norm(&gauss_2.f_slice(0, Some(d_2 as i64), Some((d_2+m) as i64), 1)?);

                let dp = norm(&t_1_snippet).dot(&t_2_snippet);

                dots.push(vec_f32_from(&dp)[0]);
            }
        }
        debug!("done in {}", t1.elapsed().as_millis());
        debug!("1. dots.len= {}", dots.len());
        dots.extend((0..n as usize -dots.len()).map(|_|0.).collect::<Vec<f32>>());
        debug!("2. dots.len= {}", dots.len());


        // let gauss_1_0 = gauss_1.f_slice(0, Some(300), Some(500), 1)?;

        crate::utils::plot(
            vec![
                (format!("G1"), BLACK, vec_f32_from(&x).into_iter().zip(vec_f32_from(&gauss_1)).collect::<Vec<(f32, f32)>>()),
                // (format!("G1"), BLACK, vec_f32_from(&lin_t(200)).into_iter().zip(vec_f32_from(&gauss_1_0)).collect::<Vec<(f32, f32)>>()),
                (format!("G2"), BLUE, vec_f32_from(&x).into_iter().zip(vec_f32_from(&gauss_2)).collect::<Vec<(f32, f32)>>()),
                (format!("dots"), MAGENTA, dots.into_iter().enumerate().map(|(i, v)|(i as f32, v)).collect::<Vec<(f32, f32)>>())

            ], "EEG", (1000, 480), format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        ).unwrap();
        // find_motifs();
        println!("ok");

        Ok(())
    }

    fn gauss_t(μ: f64, σ: f64, n: i64) -> Tensor {
        Tensor::from(gauss_i(μ, σ, n).collect::<Vec<_>>().as_slice())
    }

    fn gauss_i(μ: f64, σ: f64, n: i64) -> impl Iterator<Item=f64> {
        (0..n).map(move |x| (-0.5 * ((x as f64 - μ) / σ).powi(2)).exp())
    }

    fn pulse_t(μ: f64, σ: f64, n: i64, f: f64) -> Tensor {
        Tensor::from(pulse_v(μ, σ, n, f).as_slice())
    }

    fn pulse_v(μ: f64, σ: f64, n: i64, f: f64) -> Vec<f64> {
        gauss_i(μ, σ, n).enumerate().map(|(i, x)| (f * 2. * PI *  i as f64 / n as f64).sin() * x).collect::<Vec<_>>()
    }


    fn random_v(N: usize, amplitude: f64, max_slope: f64) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut signal = Vec::with_capacity(N);

        let mut current_value = 0.0;
        let mut prev_value = 0.0;

        for _ in 0..N {
            let slope = rng.gen_range(-max_slope..max_slope);
            let next_value = current_value + slope;

            // Check if the slope exceeds the maximum allowed slope
            if (next_value - prev_value).abs() > max_slope {
                let direction = if next_value > prev_value { -1.0 } else { 1.0 };
                current_value = prev_value + direction * max_slope;
            } else {
                current_value = next_value;
            }

            signal.push(current_value * amplitude);
            prev_value = current_value;
        }

        signal
    }

    fn lin_t(n: i64) -> Tensor {
        Tensor::from((0..n).map(|x| x as f32).collect::<Vec<f32>>().as_slice())
    }


    //from tch-0.14.0/tests/test_utils.rs
    pub fn from<'a, T>(t: &'a Tensor) -> T
        where
            <T as TryFrom<&'a tch::Tensor>>::Error: std::fmt::Debug,
            T: TryFrom<&'a Tensor>,
    {
        T::try_from(t).unwrap()
    }


    pub fn vec_f32_from(t: &Tensor) -> Vec<f32> {
        from::<Vec<f32>>(&t.reshape(-1))
    }
}