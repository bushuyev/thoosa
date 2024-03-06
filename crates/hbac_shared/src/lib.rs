use std::collections::HashMap;
use std::hash::Hash;
use log::debug;
use std::fmt::Display;
use std::backtrace::Backtrace;
use std::error::Error;
use std::f32::consts::PI;
use std::ops::Mul;
use rand::Rng;

//group of similar slices - motif
pub fn group_slices<T>(pairs: impl Iterator<Item = (T, T)>) -> HashMap<u16, Vec<T>> where T:Hash + Eq + core::fmt::Debug {
    let mut pairs_groups: HashMap<T, u16> = HashMap::new();
    let mut last_group = 0u16;

    let mut separated_groups:  Vec<(u16, u16)> = vec![];

    for (ln, pair) in pairs.enumerate() {
        let g_0 = pairs_groups.get(&pair.0);
        let g_1 = pairs_groups.get(&pair.1);
        if let Some(g_0_val) = g_0 {
            if g_1.is_some() {
                debug!("separated group: g_0={:?} g_1={:?} ln={} pair={:?}", g_0, g_1, ln, pair);
                if g_0.ne(&g_1) {
                    separated_groups.push((*g_0_val, *g_1.unwrap()));
                }
            } else {
                pairs_groups.insert(pair.1, *g_0_val);
            }
        } else {
            if let Some(g_1_val) = g_1 {
                pairs_groups.insert(pair.0, *g_1_val);
            } else {
                pairs_groups.insert(pair.0, last_group);
                pairs_groups.insert(pair.1, last_group);

                last_group += 1;
            }
        }
    }

    debug!("separated_groups={:?}", separated_groups);
    let separated_groups_map = if !separated_groups.is_empty() {
        let map = group_slices(separated_groups.into_iter()).values().fold(HashMap::new(), |mut h, v| {
            for i in v.iter().skip(1) {
                h.insert(v[0], *i);
            }
            h
        });
        debug!("separated_groups group_motifs={:?}", map);
        Some(map)
    } else {
        None
    };

    let mut groups: HashMap<u16, Vec<T>> = HashMap::new();
    for (val, group) in pairs_groups {
        let group= separated_groups_map.as_ref().and_then(|m|m.get(&group)).unwrap_or(&group);
        let mut gr_vec = groups.entry(*group).or_insert_with(|| vec![]);
        gr_vec.push(val);
    }

    debug!("group_motifs groups={:?}", groups);
    groups
}



#[derive(Debug, PartialEq, Hash, Eq, Copy, Clone)]
pub struct TensorSliceIdx(pub usize, pub i64);

impl From<(usize, i64)> for TensorSliceIdx {
    fn from(value: (usize, i64)) -> Self {
        TensorSliceIdx(value.0, value.1)
    }
}

impl Display for TensorSliceIdx {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{},{}", self.0, self.1)
    }
}

#[derive(Debug)]
pub struct HbacError {
    msg:String,
    backtrace:Backtrace
}

impl HbacError {

    pub fn new(msg:String) -> Self {
        debug!("HbacError::new: msg={}", msg);
        Self {
            msg,
            backtrace:Backtrace::capture()
        }
    }
}


impl<D> From<D> for HbacError where D: Display {

    fn from(value: D) -> Self {
        debug!("HbacError::from: value={}", value);
        HbacError::new(value.to_string())
    }
}

#[derive(Debug, Eq, Hash, PartialEq)]
pub struct MotifSlice {
    pub motif: u16, //TODO pub
    pub slise_idx: i64,
    pub channel_idx: u16,
}

impl MotifSlice {

    pub fn new(motif: u16, slise_idx: i64, channel_idx: u16) -> Self {
        Self { motif, slise_idx, channel_idx }
    }

    #[inline]
    pub fn motif(&self) -> u16 {
        self.motif
    }

    #[inline]
    pub fn slise_idx(&self) -> i64 {
        self.slise_idx
    }

    #[inline]
    pub fn channel_idx(&self) -> u16 {
        self.channel_idx
    }
}

#[cfg(test)]
pub mod tests {
    use log::LevelFilter;
    use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger};

    #[ctor::ctor]
    pub fn init() {
        CombinedLogger::init(
            vec![
                TermLogger::new(LevelFilter::Debug, Config::default(), TerminalMode::Mixed, ColorChoice::Auto)
            ]
        ).unwrap_or_else(|e|{
            println!("CombinedLogger::init failed: {:?}", e);
        });


        std::env::set_var("POLARS_FMT_MAX_COLS", "100");
        std::env::set_var("POLARS_FMT_MAX_ROWS", "100");
    }
}

pub fn pulses_in_const(n: i64, pulses: &[(usize, Vec<f32>)]) -> Vec<f32> {
    let pulses_len:usize = pulses.iter().map(|(offset, pulse)| *offset + pulse.len()).sum();

    std::iter::repeat(0.).take(n as usize).zip(
        pulses
            .into_iter()
            .flat_map(
                |(offset, pulse)|
                    std::iter::repeat(0f32)
                        .take(*offset)
                        .chain(pulse.into_iter().map(|v|*v))
            )
            .chain(std::iter::repeat(0f32).take(n as usize-pulses_len))
    ).map(|(v1, v2)| 0.001 * v1 + v2).collect::<Vec<f32>>()
}


pub fn sig_in_rand(n: i64, pulses: &[(usize, Vec<f32>)], amplitude: f32, scale: f32) -> Vec<f32> {
    let random = random_v(n, amplitude, 0.1);
    let pulses_len:usize = pulses.iter().map(|(offset, pulse)| *offset ).sum();

    join_with_offset::<f32>(n, 0., scale, pulses, random)
}

pub fn join_with_offset<T>(n: i64, def:T, scale:T, pulses: &[(usize, Vec<T>)], main: Vec<T>) -> Vec<T> where T:Copy + Clone + Mul<Output = T> + std::ops::Add<Output = T>{
    main.into_iter().zip(
        pulses
            .into_iter()
            .flat_map(
                |(offset, pulse)|
                    std::iter::repeat(def)
                        .take(*offset)
                        .chain(pulse.into_iter().map(|v| *v))
            )
            .chain(std::iter::repeat(def).take(n as usize - pulses.len()))
    ).map(|(v1, v2)| (scale * v1) + v2).collect::<Vec<T>>()
}



pub fn gauss_i(μ: f32, σ: f32, n: i64) -> impl Iterator<Item=f32> {
    (0..n).map(move |x| (-0.5 * ((x as f32 - μ) / σ).powi(2)).exp())
}

pub fn sin_v(n: i64, p: f32) -> Vec<f32> {
    (0..n).map(|x| 4. * (PI * (x as f32)/ p).sin()).collect::<Vec<f32>>()
}
pub fn comp_sin(n: i64) -> impl Iterator<Item=f32> {
    (0..n).map(|x|x as f32).map(move |x|
        30. * (PI / 100. * x).sin() +
            4. * (PI / 50. * x.powf(1.2)).sin() +
            8. * (PI / 25. * x.powf(1.2)).sin() +
            12. * (PI / 12. * x.powf(1.2)).sin() +
            6. * (PI / 6. * x.powf(1.2)).sin()
    )
}

pub fn comp_sin_v(n: i64) -> Vec<f32> {
    comp_sin(n).collect::<Vec<f32>>()
}

pub fn comp_sin_k(n: i64, k_v:Vec<f32>, scale:f32) -> impl Iterator<Item=f32> {
    (0..n).map(|x|x as f32).map(move |x|
        k_v.iter().map(|k| k * (PI / k * x.powf(1.2)).sin()).sum()
    ).map(move |x:f32|x * scale)
}

pub fn comp_sin_k_v(n: i64, k_v:Vec<f32>, scale:f32) -> Vec<f32> {
    comp_sin_k(n, k_v, scale).collect::<Vec<f32>>()
}



pub fn pulse_v(μ: f32, σ: f32, n: i64, f: f32) -> Vec<f32> {
    gauss_i(μ, σ, n).enumerate().map(|(i, x)| (f * 2. * PI *  i as f32 / n as f32).sin() * x).collect::<Vec<_>>()
}


pub fn random_v(n: i64, amplitude: f32, max_slope: f32) -> Vec<f32> {
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


pub fn parse_key_val<T, U>(s: &str) -> Result<(T, U), Box<dyn Error + Send + Sync + 'static>>
    where
        T: std::str::FromStr,
        T::Err: Error + Send + Sync + 'static,
        U: std::str::FromStr,
        U::Err: Error + Send + Sync + 'static,
{
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid KEY=value: no `=` found in `{s}`"))?;
    Ok((s[..pos].parse()?, s[pos + 1..].parse()?))
}


#[cfg(test)]
mod shared_tests {
    /*
      a - b
      a - c
      e - f
      b - d
      f - g
      x - y


   group 1: a, b, c, d
   group 2: e, f, g
   group 3: x, y
   */

    use crate::group_slices;

    #[test]
    fn test_group_motifs(){
        let pairs:Vec<(String, String)> = vec![
            ("a".into(), "b".into()),
            ("a".into(), "c".into()),
            ("e".into(), "f".into()),
            ("b".into(), "d".into()),
            ("f".into(), "g".into()),
            ("x".into(), "y".into()),
        ];


        let mut groups = group_slices(pairs.into_iter());

        let group_0 = groups.get_mut(&0).unwrap();
        group_0.sort();
        assert_eq!(&vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()], group_0);

        let group_1 = groups.get_mut(&1).unwrap();
        group_1.sort();
        assert_eq!(&vec!["e".to_string(), "f".to_string(), "g".to_string()], group_1);

        let group_2 = groups.get_mut(&2).unwrap();
        group_2.sort();
        assert_eq!(&vec!["x".to_string(), "y".to_string()], group_2);
    }

    #[test]
    fn test_group_motifs_separated(){
        let pairs:Vec<(String, String)> = vec![
            ("l".into(), "m".into()),
            ("a".into(), "x".into()),
            ("a".into(), "y".into()),
            ("b".into(), "w".into()),
            ("b".into(), "v".into()),
            ("a".into(), "b".into()),
        ];


        let mut groups = group_slices(pairs.into_iter());

        assert_eq!(groups.len(), 2);

        let mut values:Vec<Vec<String>> = groups.values().into_iter().map(|v|v.clone()).collect();
        values.sort_by(|a, b|b.len().cmp(&a.len()));


        let group_0 = values.get_mut(0).unwrap();
        group_0.sort();
        assert_eq!(&vec!["a".to_string(), "b".to_string(), "v".to_string(), "w".to_string(), "x".to_string(), "y".to_string()], group_0);

        let group_1 = values.get_mut(1).unwrap();
        group_1.sort();
        assert_eq!(&vec!["l".to_string(), "m".to_string()], group_1);

    }
}