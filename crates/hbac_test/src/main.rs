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
    use std::collections::VecDeque;
    use std::f64::consts::PI;
    use std::iter::Map;
    use std::mem::size_of_val;
    use std::ops::Range;
    use std::time::Instant;
    use std::io;
    use std::io::Write;

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
    pub fn test_filt(){
        let sig = [1, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1];
        let m = 4;
        let n = sig.len();
        let mut motif_start_op:Option<usize> = None;
        let min_dot = 1;
        let mut motifs:Vec<(usize, usize)> = vec![];

        for (i, dot) in sig.into_iter().enumerate(){
            match motif_start_op {
                None => {
                    if dot > min_dot {
                        motif_start_op = Some(i);
                    }
                }
                Some(motif_start) => {
                    if dot <= min_dot {
                        motifs.push((motif_start, i));
                        motif_start_op = None;
                    }
                }
            }
        }

        assert_eq!([(1, 3), (5, 8)], motifs.as_slice());

        println!("motifs={:?}", motifs);
    }

    #[test]
    pub fn   test_z_norm() {
        assert_eq!(vec![-1.2247449, 0.0, 1.2247449], vec_f32_from(&z_norm(&Tensor::from_slice(&[1., 2., 3.]), 0i64)));
    }

    #[test]
    pub fn test_norm() -> Result<(), GenError>{
        let t = Tensor::from_slice(&[1., 2., 3.]);
        let norm = norm(&t, &[-1]);
        // norm.print();
        assert_eq!(1.0, vec_f32_from(&norm.dot(&norm))[0]);


        let t = Tensor::vstack(&[
            Tensor::from_slice(&[1., 2., 3.]),
            Tensor::from_slice(&[1., 2., 3.])
        ]);
        t.print();
        let norm = crate::tests::norm(&t, &[1, -1]);
        norm.print();
        // assert_eq!(1.0, vec_f32_from(&norm.dot(&norm))[0]);

        Ok(())
    }

    fn norm(t: &Tensor, dim: &[i64]) -> Tensor {
        t.view(dim) / t.dot(&t).sqrt()
    }

    #[test]
    pub fn test_pulses_in_rand() -> Result<(), GenError>{
        let n = 1000i64;
        let x = lin_t(n);
        // let gauss_1 = gauss_t(400., 100., n);

        let to_pulses_in_random = pulses_in_rand(
            n,
            &[
                (100, pulse_v(80., 20., 200, 20.)),
                (100, pulse_v(80., 20., 200, 20.))
            ]
        );

        let sig = Tensor::from_slice(
            to_pulses_in_random.as_slice()
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
        println!("sig={:?}", sig.size());

        crate::utils::plot_series(
            vec![
                (format!("G1"), BLACK, vec_f32_from(&x).into_iter().zip(vec_f32_from(&sig)).collect::<Vec<(f32, f32)>>()),

            ], "EEG", (1000, 500), format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        ).unwrap();

        Ok(())
    }

    #[test]
    fn test_motifs_match(){
        let t1 = Instant::now();

        let n = 10_000;

        let mut t = Box::new(Tensor::zeros(&[n, n], (Kind::Float, Device::Cpu)));

        for i in (0..n) {
            if i % 100 == 0{
                debug!("i={}", i);
            }

            // std::io::stdout().flush().expect("flush");


            for j in (0..n) {

                t.index_put_(
                    &[
                        Some(Tensor::from_slice(&[i])),//row
                        Some(Tensor::from_slice(&[j])) //col
                    ],
                    &Tensor::from_slice(&[ (i * j) as f32  ]),
                    false
                );
            }
        }


        // debug!("mem size {}", size_of_val(&t.as_ptr()));
        debug!("done {} in {}", n, t1.elapsed().as_millis());
    }



    fn self_dot(t1: &Tensor, i_0: i64, i_1: i64, dim:i64, step:i64, d_i: i64) -> Result<Tensor, GenError> {
        let slice_1 = t1.f_slice(dim, Some(i_0), Some(i_0+d_i), step)?;
        let slice_2 = t1.f_slice(dim, Some(i_1), Some(i_1+d_i), step)?;

        // slice_t.print();

        Ok(slice_1.tensordot(&slice_2, [dim], [dim]).diag(0))
    }

    fn self_dot_norm(t1: &Tensor, i_0: i64, i_1: i64, dim:i64, step:i64, d_i: i64) -> Result<Tensor, GenError> {
        let slice_1 = norm(&t1.f_slice(dim, Some(i_0), Some(i_0+d_i), step)?, &[1, -1]);
        let slice_2 = norm(&t1.f_slice(dim, Some(i_1), Some(i_1+d_i), step)?, &[1, -1]);

        // slice_t.print();

        Ok(slice_1.tensordot(&slice_2, [dim], [dim]).diag(0))
    }


    #[test]
    fn test_self_dot() -> Result<(), GenError>{
        let t1 = Tensor::from_slice2(&[
            &[1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
            &[10., 20., 30., 40., 50., 60., 70., 80., 90.],
            &[100., 200., 300., 400., 500., 600., 700., 800., 900.]
        ]);

        //(1,2,3) x
        //(1,2,3)
        let d_0 = self_dot(&t1, 0, 0, 1, 1, 3)?;
        assert_eq!(Tensor::from_slice(&[14., 1400., 140000.]), d_0);

        //(1,2,3) x
        //(2,3,4)
        let d_1 = self_dot(&t1, 0, 1, 1, 1, 3)?;
        assert_eq!(Tensor::from_slice(&[20., 2000., 200000.]), d_1);

        //(1,2,3) x
        //(3,4,5)
        let d_2 = self_dot(&t1, 0, 2, 1, 1, 3)?;
        assert_eq!(Tensor::from_slice(&[26., 2600., 260000.]), d_2);

        //(1,2,3) x
        //(4,5,6) = 26-(1*3+2*3+3*3)+(1*4+2*4+3*4)
        let d_3 = self_dot(&t1, 0, 3, 1, 1, 3)?;
        assert_eq!(Tensor::from_slice(&[32., 3200., 320000.]), d_3);


        Ok(())
    }

    #[test]
    fn test_motifs_match_vec(){
        let t1 = Instant::now();

        let n = 1_0;

        // let mut rows_vec:Box<Vec<Vec<f32>>> = Box::new(Vec::with_capacity(n));
        let mut t = Box::new(Tensor::zeros(&[n, n], (Kind::Uint8, Device::Cpu)));

        for i in (0..n) {
            if i % 1000 == 0 {
                debug!("i={}", i);
            }

            // std::io::stdout().flush().expect("flush");
            let mut row_vec:Vec<u8> = Vec::with_capacity(n as usize);

            for j in (0..n) {
                row_vec.push((i * j) as u8)

            }
            // rows_vec.push(row_vec);
            t.index_put_(
                &[
                    Some(Tensor::from_slice(&[i])),//row
                ],
                &Tensor::from_slice(row_vec.as_slice()),
                false
            );
        }

        t.print();

        // debug!("mem size {}", size_of_val(&t.as_ptr()));
        debug!("done {} in {}", n, t1.elapsed().as_millis());

        // let t1_vec = from::<Vec<Vec<f32>>>(&t);
        // debug!("t1_vec={:?}", t1_vec.len());
        for i in (0..n){
            for j in (0..n) {
                print!("{:?} ", t.int64_value(&[i, j]));
            }
            println!();
        }


        // let mut t = Box::new(Tensor::from_slice(rows_vec.as_slice()));
        // debug!("done t {:?} in {}", t.size(), t1.elapsed().as_millis());
    }


    fn pulses_in_rand(n: i64, pulses: &[(usize, Vec<f64>)]) -> Vec<f64> {
        let random = random_v(n, 10., 1.);
        let pulses_len:usize = pulses.iter().map(|(offset, pulse)| *offset + pulse.len()).sum();

        random.into_iter().zip(
            pulses
                .into_iter()
                .flat_map(
                    |(offset, pulse)|
                        std::iter::repeat(0f64)
                        .take(*offset)
                        .chain(pulse.into_iter().map(|v|*v))
                )
                .chain(std::iter::repeat(0f64).take(n as usize-pulses_len))
        ).map(|(v1, v2)| 0.001 * v1 + v2).collect::<Vec<f64>>()
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

        crate::utils::plot_series(
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
        // let sig_1 = gauss_t(400., 100., n); //f.flip([0]);
        // let sig_2 = gauss_t(600., 100., n); //.flip([0]);

        let sig_1 = Tensor::from_slice(
            pulses_in_rand(n, &[]).as_slice()
        );

        let sig_2 = Tensor::from_slice(
            pulses_in_rand(n, &[]).as_slice()
        );
        println!("sig_1.size()={:?}, sig_2.size()={:?}", sig_1.size(), sig_2.size());
        let sig_1_c = sig_1.copy();
        let sig_2_c = sig_2.copy();
        let t = Tensor::vstack(&[sig_1, sig_2]);
        println!("t.size={:?}", t.size());



        let m = 10;

        let zeros = Tensor::zeros([m], (Kind::Float, Device::Cpu));

        let t1 = Instant::now();
        let mut dots:Vec<f32> = vec![];


        for i_0 in (0..n-m) {
        // let d_1 = 200; {
            // let t_1_snippet = sig_1.f_slice(0, Some(d_1), Some((d_1+m) as i64), 1)?;

            for i_i in (i_0..n-m) {
                // println!("d_1={:?}, d_2={:?}", d_1, d_2);

                // let t_2_snippet = norm(&sig_2.f_slice(0, Some(d_2 as i64), Some((d_2+m) as i64), 1)?);

                // let dp = norm(&t_1_snippet).dot(&t_2_snippet);
                let dp = self_dot_norm(&t, i_0, i_i, 1, 1, m)?;

                dots.push(vec_f32_from(&dp)[0]);
            }
        }
        debug!("done in {}", t1.elapsed().as_millis());
        debug!("1. dots.len= {}", dots.len());
        // dots.extend((0..n as usize -dots.len()).map(|_|0.).collect::<Vec<f32>>());
        // debug!("2. dots.len= {}", dots.len());


        // let gauss_1_0 = gauss_1.f_slice(0, Some(300), Some(500), 1)?;

        crate::utils::plot_series(
            vec![
                (format!("G1"), BLACK, vec_f32_from(&x).into_iter().zip(vec_f32_from(&sig_1_c)).collect::<Vec<(f32, f32)>>()),
                // (format!("G1"), BLACK, vec_f32_from(&lin_t(200)).into_iter().zip(vec_f32_from(&gauss_1_0)).collect::<Vec<(f32, f32)>>()),
                (format!("G2"), BLUE, vec_f32_from(&x).into_iter().zip(vec_f32_from(&sig_2_c)).collect::<Vec<(f32, f32)>>()),
                // (format!("dots"), MAGENTA, dots.into_iter().enumerate().map(|(i, v)|(i as f32, v)).collect::<Vec<(f32, f32)>>())

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


    fn random_v(n: i64, amplitude: f64, max_slope: f64) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut signal = Vec::with_capacity(n as usize);

        let mut current_value = 0.0;
        let mut prev_value = 0.0;

        for _ in 0..n {
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