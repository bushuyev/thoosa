use tch::{kind, Tensor};

pub fn find_motifs(){
    println!("find_motifs");
}

fn precompute_stats(x: &Tensor, m: usize) -> (Tensor, Tensor) {


    // let n = x.len();
    // let mut cumsum = x.clone();
    // cumsum.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);
    // let mut cumsum2 = x.map(|o| o.powi(2));
    // cumsum2.accumulate_axis_inplace(Axis(0), |&prev, curr| *curr += prev);
    // let sum_t = cumsum.slice(s![(m - 1)..]).to_owned()
    //     - concatenate![Axis(0), Array1::zeros(1), cumsum.slice(s![..(n - m)])];
    // let sum_t2 = cumsum2.slice(s![(m - 1)..]).to_owned()
    //     - concatenate![Axis(0), Array1::zeros(1), cumsum2.slice(s![..(n - m)])];
    // let mean_t = sum_t / (m as f32);
    // let mean_t2 = sum_t2 / (m as f32);
    // let mean_tp2 = mean_t.map(|o| o.powi(2));
    // let sigma_t2 = mean_t2 - mean_tp2;
    // let sigma_t = sigma_t2.mapv(f32::sqrt);
    // (mean_t, sigma_t)
    (
        x.mean_dim(0, true,  kind::Kind::Float),
        x.std_dim(0, true, true)
    )
}

#[cfg(test)]
mod tests {
    use std::time::Instant;
    use log::{debug, LevelFilter};
    use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger};
    use crate::motifs::{find_motifs, precompute_stats};
    use rand::{distributions::Uniform, Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use approx::assert_relative_eq;
    use plotters::prelude::{BLACK, BLUE};
    use tch::Tensor;
    use crate::data::GenError;

    pub fn random_data(n: usize, seed: u64) -> Vec<f32> {
        let rng = ChaCha8Rng::seed_from_u64(seed);
        rng.sample_iter(Uniform::new_inclusive(0.0, 100.0))
            .take(n)
            .collect::<Vec<_>>()
    }

    #[ctor::ctor]
    fn init() {
        CombinedLogger::init(
            vec![
                TermLogger::new(LevelFilter::Debug, Config::default(), TerminalMode::Mixed, ColorChoice::Auto)
            ]
        ).unwrap();
    }

    #[test]
    pub fn test_find_motifs() -> Result<(), GenError>{

        println!("ok");

        Ok(())
    }

    fn gauss_t(μ: f32, σ: f32, n: i32) -> Tensor {
        Tensor::from((0..n).map(|x| (-0.5 * ((x as f32 - μ) / σ).powi(2)).exp()).collect::<Vec<_>>().as_slice())
    }

    fn lin_t(n: i32) -> Tensor {
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