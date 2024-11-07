// #![feature(backtrace_frames)]

use std::backtrace::Backtrace;
use std::cmp::{max, min};
use std::collections::HashMap;
use std::error::Error;
use std::f32::consts::PI;
use std::fmt::{Debug, Display, Formatter};
use std::fmt;
use std::hash::Hash;
use std::ops::{Div, Mul, Range};
use std::sync::atomic::{AtomicBool, Ordering};
use chrono::SecondsFormat;
use log::{debug, error, trace, LevelFilter};
use rand::{Rng, RngCore};
use regex::Regex;
use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger};
use std::env;
use std::env::VarError;
use std::ffi::OsStr;
use std::fs::read_dir;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use lazy_static::lazy_static;
use num_traits::{Num, PrimInt};


pub mod repeat_iter;
pub mod val;

//group of similar slices - motif
pub fn group_slices<T>(pairs: impl Iterator<Item = (T, T)>) -> HashMap<u16, Vec<T>>
where T:Hash + Eq + Debug {

    let mut pairs_groups: HashMap<T, u16> = HashMap::new();
    let mut last_group = 0u16;

    let mut separated_groups:  Vec<(u16, u16)> = vec![];

    for (ln, pair) in pairs.enumerate() {
        let g_0 = pairs_groups.get(&pair.0);
        let g_1 = pairs_groups.get(&pair.1);
        if let Some(g_0_val) = g_0 {
            if g_1.is_some() {
                trace!("separated group: g_0={:?} g_1={:?} ln={} pair={:?}", g_0, g_1, ln, pair);
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
        let map = group_slices(separated_groups.into_iter()) //group separated groups same as slices
            .values()
            .fold(HashMap::new(), |mut h, v| {

                for i in v.iter().skip(1) {
                    h.insert(*i, v[0]);
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
        let group= separated_groups_map.as_ref().and_then(|m|m.get(&group)).unwrap_or(&group); //replace separated group pairs with one in pair
        let gr_vec = groups.entry(*group).or_insert_with(|| vec![]);
        gr_vec.push(val);
    }

    debug!("group_motifs groups={:?}", groups);
    groups
}



#[derive(Debug, PartialEq, Hash, Eq, Copy, Clone, PartialOrd, Ord)]
pub struct TensorSliceIdx<T>(pub T, pub i64) where T:Eq+Display+Debug;

impl From<(String, i64)> for TensorSliceIdx<String> {
    fn from(value: (String, i64)) -> Self {
        TensorSliceIdx(value.0, value.1)
    }
}


impl <T> Display for TensorSliceIdx<T> where T:Eq+Display+Debug{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{},{}", self.0, self.1)
    }
}
// #[derive(Debug)]
pub struct ThoosaError {//TODO enum
    msg:String,
    #[allow(dead_code)]
    backtrace:Backtrace
}

impl ThoosaError {

    pub fn new(msg:impl Into<String> + Display) -> Self {
        error!("ThoosaError::new: msg={}", msg);
        let backtrace = Backtrace::capture();

        debug!("{}", backtrace.to_string());
        Self {
            msg: msg.into(),
            backtrace
        }
    }
}

// impl Display for ThoosaError {
//     fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
//         write!(f, "{}", self.msg)
//     }
// }
//TODO remove impl<D> From<D> for ThoosaError
// impl std::error::Error for ThoosaError {
//
// }

//TODO https://github.com/rust-lang/rust/issues/65280
impl Debug for ThoosaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug_struct = f.debug_struct("ThoosaError");
        let x=  debug_struct
            .field("msg", &self.msg);

        // #![feature(backtrace_frames)]
        // for (i, frame) in self.backtrace.frames().iter().enumerate() {
        //     x = x.field(format!("frame_{}\n", i).as_str(), frame);
        // }

        x.finish()

    }
}


impl<D> From<D> for ThoosaError
where D: Display {

    fn from(value: D) -> Self {
        error!("ThoosaError::from: value={}", value);
        ThoosaError::new(value.to_string())
    }
}

#[derive(Debug, Eq, Hash, PartialEq)]
pub struct MotifSlice {
    pub motif_id: usize, //TODO pub
    pub slise_idx: i64,
    pub channel_idx: u16,
}

impl MotifSlice {

    pub fn new(motif_id: usize, slise_idx: i64, channel_idx: u16) -> Self {
        Self { motif_id, slise_idx, channel_idx }
    }

    #[inline]
    pub fn motif_id(&self) -> usize {
        self.motif_id
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

pub fn random_saw(n: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let step = 25usize;

    (0..n/step/2)
        .map(|_| rng.gen_range(-10. .. 10.))
        .flat_map(
            |v| {
                (0..step).map(move |i| v * i as f32).chain((0..step).map(move |i| step as f32 * v - v * i as f32))
            }
        ).collect::<Vec<f32>>()
}

pub fn random_saw_with_steps(n: usize, steps: usize, max_y: f32) -> Vec<f32> {
    let mut rng = rand::thread_rng();

    let mut y = vec![0.;n];
    let mut x = 0;

    let def_step = n/steps;

    loop {
        let mut dx = min(n-x, (def_step as f32 * rng.gen_range(0.5 .. 1.5)) as usize);
        if dx % 2 == 0 {
            dx+=1;
        }

        let max_y = max_y  * rng.gen_range(-1.5 .. 1.5);

        if dx > 2 {
            let dy = max_y / (dx-1) as f32 * 2.;

            for i in 1..dx / 2  {
                y[x + i] = if x + i > 0 { y[x + i - 1] } else { 0. } + dy;
                y[x + dx - i - 1] = y[x + i];
            }

        }

        y[x + dx / 2 ] = max_y;

        x = x+dx-1;
        if x>=n{
            break
        }

    }

    y
}

pub fn sig_three_imp(n: i64, max_y: f32, σ_range: Option<Range<f32>>, f_range: Option<Range<f32>>) -> Vec<f32> {
    let (σ, f) = σ_f_params(σ_range, f_range);

    join_with_offset::<f32>(0., 1.0, &[
        ((0.1*n as f32) as usize, pulse_v_fit(0., σ, (0.3 * n as f32) as i64, f, 100., max_y)),
        ((0.0*n as f32) as usize, pulse_v_fit(0., σ, (0.2 * n as f32) as i64, f, 100., max_y)),
        ((0.0*n as f32) as usize, pulse_v_fit(0., σ, (0.3 * n as f32) as i64, f, 100., max_y)),
    ], random_saw_with_steps(n as usize, 20, 100.))
}

pub fn sig_two_imp(n: i64, max_y: f32, σ_range: Option<Range<f32>>, f_range: Option<Range<f32>>) -> Vec<f32> {
    let (σ, f) = σ_f_params(σ_range, f_range);

    join_with_offset::<f32>(0., 1.0, &[
        ((0.1*n as f32) as usize, pulse_v_fit(0., σ, (0.4*n as f32) as i64, f, 100., max_y)),
        ((0.0*n as f32) as usize, pulse_v_fit(0., σ, (0.4*n as f32) as i64, f, 100., max_y)),
    ], random_saw_with_steps(n as usize, 20, 100.))
}

pub fn sig_one_imp(n: i64, max_y: f32, σ_range: Option<Range<f32>>, f_range: Option<Range<f32>>) -> Vec<f32> {
    let (σ, f) = σ_f_params(σ_range, f_range);
    join_with_offset::<f32>(0., 1.0, &[
        ((0.1*n as f32) as usize, pulse_v_fit(0., σ, (0.8*n as f32) as i64, f, 100., max_y)),
    ], random_saw_with_steps(n as usize, 20, 100.))
}

fn σ_f_params(σ_range: Option<Range<f32>>, f_range: Option<Range<f32>>) -> (f32, f32) {
    let mut rng = rand::thread_rng();
    let σ = σ_range.map(|r| rng.gen_range(r)).unwrap_or(20.);
    let f = f_range.map(|r| rng.gen_range(r)).unwrap_or(20.);
    (σ, f)
}

pub fn saw(n: i64) -> Vec<f32> {
    random_saw_with_steps(n as usize, 30, 100.)
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


pub fn pulse_in_rand(n: i64, pulses: &[(usize, Vec<f32>)], random_amplitude: f32, pulse_scale: f32, max_slope: f32) -> Vec<f32> {
    let random = random_v(n, random_amplitude, max_slope);
    let _pulses_len:usize = pulses.iter().map(|(offset, _pulse)| *offset ).sum();

    join_with_offset::<f32>(0., pulse_scale, pulses, random)
}

pub fn join_with_offset<T:Debug>(def:T, scale:T, pulses: &[(usize, Vec<T>)], main: Vec<T>) -> Vec<T> where T:Copy + Clone + Mul<Output = T> + std::ops::Add<Output = T> + PartialEq + Default{
    let mut x:T = T::default();
    let main_len = main.len();
    main.into_iter().zip(
        pulses
            .into_iter()
            .flat_map(
                |(offset, pulse)| {
                    std::iter::repeat(def)
                        .take(*offset)
                        .chain(pulse.into_iter().map(|v| *v))
                }
            )
            .chain(std::iter::repeat(def).take(main_len - pulses.last().map(|p| p.0 + p.1.len()).unwrap_or(0)))

    ).map(|(main_v, pulse_v)| {
        if pulse_v !=def {
            scale * pulse_v
        } else {
            x = main_v;
            x
        }
    }).collect::<Vec<T>>()
}

pub fn gauss_v(μ: f32, σ: f32, n: i64) -> Vec<f32> {
    gauss_i(μ, σ, n).collect()
}

pub fn gauss_i(μ: f32, σ: f32, n: i64) -> impl Iterator<Item=f32> {
    (0..n).map(move |x| (-0.5 * ((x as f32 - μ) / σ).powi(2)).exp())
}

pub fn gauss_scaled(μ: f32, σ: f32, n: i64, max_x:f32, max_y:f32) -> Vec<f32> {
    (0..n).map(move |i| -max_x /2. + i as f32 * max_x /n as f32).map(move |x| (-0.5 * ((x - μ) / σ).powi(2)).exp()).map(|y|y*max_y).collect()
}

pub fn pulse_v_fit(μ: f32, σ: f32, n: i64, f: f32, max_x:f32, max_y:f32) -> Vec<f32> {
    gauss_scaled(μ, σ, n, max_x, max_y).into_iter().enumerate().map(|(i, x)|  x * ( f * 2. * PI *  i as f32 / n as f32).sin() ).collect::<Vec<_>>()
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
    let mut signal = Vec::with_capacity(n as usize);

    let mut current_value = 0.0;
    let mut prev_value = 0.0;

    let rnd_vals_0 = (0..n).flat_map(move |_i| std::iter::repeat(      rand::thread_rng().gen_range(-max_slope..max_slope)).take(1)).collect::<Vec<f32>>();
    let rnd_vals_1 = (0..n / 10).flat_map(move |_i|std::iter::repeat(   rand::thread_rng().gen_range(-max_slope..max_slope)).take(10)).collect::<Vec<f32>>();
    let rnd_vals_2 = (0..n / 60).flat_map(move |_i|std::iter::repeat(  rand::thread_rng().gen_range(-max_slope..max_slope)).take(100)).collect::<Vec<f32>>();
    let rnd_vals_3 = (0..n / 120).flat_map(move |_i|std::iter::repeat(  rand::thread_rng().gen_range(-max_slope..max_slope)).take(300)).collect::<Vec<f32>>();

    rnd_vals_0.into_iter().zip(
        rnd_vals_1.into_iter().zip(
            rnd_vals_2.into_iter().zip(
                rnd_vals_3.into_iter()
            )
        )
    ).for_each(|(r_0, (r_1, (r_2, r_3)))| {
        // debug!("random_v: r_0={:?}, r_1={:?}, r_2={:?}, r_3={:?}", r_0, r_1, r_2, r_3);

        current_value = prev_value + 10.* r_0 + r_1 +  r_2 + r_3;
        signal.push(current_value * amplitude);
        prev_value = current_value;
    });

    signal
}


pub fn parse_key_val<T, U>(s: &str) -> Result<(T, U), Box<dyn Error + Send + Sync + 'static>>
    where
        T: FromStr,
        T::Err: Error + Send + Sync + 'static,
        U: FromStr,
        U::Err: Error + Send + Sync + 'static,
{
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid KEY=value: no `=` found in `{s}`"))?;
    Ok((s[..pos].parse()?, s[pos + 1..].parse()?))
}

#[derive(Debug, Clone, PartialEq)]
pub enum RangeOrValue<V> {
    Range{range:Range<V>, step:Option<(V, u32)>},//TODO replace step in params with number of steps? how choose number of decimals in step=to-from/steps?
    Value(V)
}

impl IntoIterator for RangeOrValue<u32> {
    type Item = u32;
    type IntoIter = Box<dyn Iterator<Item = u32>>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            RangeOrValue::Range { range, step } => {
                if let Some((step, mul)) = step {
                    Box::new(
                        (range.start .. range.end).step_by(step as usize)
                    )
                } else {
                    Box::new([range.start, range.end].into_iter())
                }
            }
            RangeOrValue::Value(v) => Box::new(std::iter::once(v)),
        }
    }
}

impl IntoIterator for RangeOrValue<i32> {
    type Item = i32;
    type IntoIter = Box<dyn Iterator<Item = i32>>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            RangeOrValue::Range { range, step } => {
                if let Some((step, mul)) = step {
                    Box::new(
                        (range.start .. range.end)
                            .step_by(step as usize)
                    )
                } else {
                    Box::new(std::iter::empty())
                }
            }
            RangeOrValue::Value(v) => Box::new(std::iter::once(v)),
        }
    }
}


impl IntoIterator for RangeOrValue<f64> {
    type Item = f64;
    type IntoIter = Box<dyn Iterator<Item = f64>>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            RangeOrValue::Range { range, step } => {
                if let Some((step, mul)) = step {
                    Box::new(
                        ((range.start * <u32 as Into<f64>>::into(mul)) as i32 .. (range.end * <u32 as Into<f64>>::into(mul)) as i32)
                            .step_by((step * <u32 as Into<f64>>::into(mul)) as usize)
                            .map(move |v|<i32 as Into<f64>>::into(v)/<u32 as Into<f64>>::into(mul))
                    )
                } else {
                    Box::new(std::iter::empty())
                }
            }
            RangeOrValue::Value(v) => Box::new(std::iter::once(v)),
        }
    }
}



/*impl<V: 'static + Mul<V, Output = usize> + From<u32> + Div<Output=V>> IntoIterator for RangeOrValue<V>

{
    type Item = V;
    type IntoIter = Box<dyn Iterator<Item = V>>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            RangeOrValue::Range { range, step } => {
                if let Some((step, mul)) = step {
                    Box::new(
                        ((range.start * mul.into()) as u32 .. (range.end * mul.into()) as u32)
                            .step_by(step * mul.into())
                            .map(move |v|<u32 as Into<V>>::into(v)/mul.into())
                    )
                } else {
                    Box::new(std::iter::empty())
                }
            }
            RangeOrValue::Value(v) => Box::new(std::iter::once(v)),
        }
    }
}
*/
impl <V:Display+PrimInt> Display for RangeOrValue<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            RangeOrValue::Range{range, step} => {
                write!(f, "{}-{}", range.start, range.end)
            }
            RangeOrValue::Value(v) => {
                write!(f, "{}", v)
            }
        }
    }
}

impl<V> RangeOrValue<V> where V: Clone + 'static  {
    pub fn expect_value(&self) -> Result<&V, ThoosaError> {
        match self {
            RangeOrValue::Range{..} => {
                Err(ThoosaError::new("expected value"))
            }
            RangeOrValue::Value(v) => {
                Ok(v)
            }
        }
    }

    pub fn expect_range(&self) -> Result<&Range<V>, ThoosaError> {
        match self {
            RangeOrValue::Range{range, step} => {
                Ok(range)
            }
            RangeOrValue::Value(v) => {
                Err(ThoosaError::new("expected range"))
            }
        }
    }

}

impl  From<f64> for RangeOrValue<f64>  {
    fn from(value: f64) -> Self {
        RangeOrValue::Value(value)
    }
}

impl  From<u32> for RangeOrValue<u32>  {
    fn from(value: u32) -> Self {
        RangeOrValue::Value(value)
    }
}

impl  From<i32> for RangeOrValue<i32>  {
    fn from(value: i32) -> Self {
        RangeOrValue::Value(value)
    }
}


impl <V>  From<Range<V>> for RangeOrValue<V> where V:PartialEq + Into<RangeOrValue<V>> {
    fn from(r: Range<V>) -> Self {
        if r.start == r.end {
            r.start.into()
        } else {
            RangeOrValue::Range {range: r, step: None}
        }
    }
}



pub fn parse_range_or_value<'a, V>(s: &str) -> Result<RangeOrValue<V>, Box<dyn Error + Send + Sync + 'static>>
where
    V: FromStr,
    V::Err: Error + Send + Sync + 'static {

    if s.contains("..") {
        let ((from, from_d_len), (to, to_d_len), step) =  parse_range_str(s).expect("parse_range_str"); //.map_err(|e|)?;

        Ok(RangeOrValue::Range{
            range: from.parse()?..to.parse()?,
            step:step.map(|(step, step_len)|(step.parse().unwrap(), 10u32.pow([from_d_len,to_d_len, step_len].into_iter().max().unwrap_or(0) as u32)))
        })

    } else {
        Ok(RangeOrValue::Value(s.parse()?))
    }

}

#[derive(Debug, Clone, PartialEq)]
pub enum Resources {
    List(Vec<String>),
    Set(String),
    Group(String)
}

pub fn parse_resources<'a, V>(s: &str) -> Result<Resources, Box<dyn Error + Send + Sync + 'static>> {
    let value_pos = s.find(":").expect("resource specifier should contain ':'") + 1;

    if s.starts_with("list") {
        Ok(Resources::List(s[value_pos..].split(",").map(|v|v.to_string()).collect()))

    } else if s.starts_with("set") {
        Ok(Resources::Set(s[value_pos..].to_string()))

    } else if s.starts_with("group") {
        Ok(Resources::Group(s[value_pos..].to_string()))

    } else {
        panic!("unexpected param {s}")
    }
}

/**
calculates indexes for sliding  windows of size {motif_width} and step {step} in input of length {input_n} and splits them in N batches of  {batch_size} and reminder M
*/
pub fn batch_windows(input_n: usize, motif_width: usize, step: usize, batch_size: usize) -> (Vec<Vec<usize>>, Vec<usize>) { // (batches, reminder)
    debug!("slice_windows: input_n={input_n}, motif_width={motif_width}, step={step} batch_size={batch_size}");

    (0..=input_n - motif_width).step_by(step).fold((vec![], vec![]), |mut a, v| {
        a.1.push(v);
        if a.1.len() == batch_size {
            a.0.push(a.1);
            a.1 = vec![];
        }
        a
    })
}

pub fn clc_num_batches(input_n: usize, motif_width: usize, step: usize, batch_size: usize) -> usize {
    (((input_n-motif_width)/step) as f32 / batch_size as f32).ceil() as usize
}

pub fn parse_corr_labels(descr: &str) -> Result<Vec<(i64, String)>, ThoosaError> {
    let re: Regex = Regex::new(r#"\((?<id>\d+),\s"(?<kind>\w+)"\)"#).map_err(|e|ThoosaError::new(e.to_string()))?;//TODO const, error
    Ok(re.captures_iter(descr).map(|caps| (caps["id"].parse::<i64>().expect(&caps["id"]), caps["kind"].into())).collect::<Vec<(i64, String)>>())
}



pub fn cdt() -> String {
    chrono::offset::Local::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

pub fn cdt_for_dir() ->String {//TODO move to shared
    cdt().replace(':', "_").replace('+', "_")
}

pub static LOG_INIT: AtomicBool = AtomicBool::new(false);

// #[cfg(test)]
pub fn init_test_logging() {

    if !LOG_INIT.load(Ordering::Relaxed) {

        LOG_INIT.store(true, Ordering::Relaxed);

        CombinedLogger::init(
            vec![
                TermLogger::new(
                    env::var("RUST_LOG").and_then(|v| LevelFilter::from_str(v.as_str()).map_err(|e| {
                        println!("Can't find loglevel: {}", e);
                        VarError::NotPresent
                    })).unwrap_or(LevelFilter::Debug),
                    Config::default(),
                    TerminalMode::Mixed,
                    ColorChoice::Auto
                )
            ]
        ).unwrap_or_else(|e| {
            debug!("CombinedLogger::init failed: {:?}", e);
        });


        std::env::set_var("POLARS_FMT_MAX_COLS", "100");
        std::env::set_var("POLARS_FMT_MAX_ROWS", "100");
    }
}

pub fn recurse_files<F>(path: impl AsRef<Path>, filter:&F) -> std::io::Result<Vec<PathBuf>>
where F:Fn(&PathBuf)->bool{
    let mut buf = vec![];
    let entries = read_dir(path)?;

    for entry in entries {
        let entry = entry?;
        let meta = entry.metadata()?;

        if meta.is_dir() {
            let mut subdir = recurse_files(entry.path(), filter)?;
            buf.append(&mut subdir);
        }

        if meta.is_file() {
            let path = entry.path();
            debug!("path={path:?} path.ends_with(safetensors)={} ", path.ends_with("safetensors"));
            // if path.extension().map(|e|e.eq("safetensors")).unwrap_or(false) { //TODO param
            if filter(&path) { //TODO param
                buf.push(path);
            }
        }
    }

    Ok(buf)
}

fn parse_range_str(range: &str) -> Result<((String, u8), (String, u8), Option<(String, u8)>), ThoosaError> {

    lazy_static!{
        static ref RANGE_RE:Regex = Regex::new(r"^(?P<from>-?\d+(\.(?P<from_d>\d+))?)\.{2}(?P<to>-?\d+(\.(?P<to_d>\d+))?)(:(?P<step>-?\d+(\.(?P<step_d>\d+))?))?$").unwrap();
    }

    RANGE_RE.captures(range).map(|caps|{

        let from_i = caps.name("from").map(|v|v.as_str().to_string()).expect("from");
        let from_d = caps.name("from_d").and_then(|v| v.as_str().len().try_into().ok()).unwrap_or(0);

        let to_i = caps.name("to").map(|v|v.as_str().to_string()).expect("to");
        let to_d = caps.name("to_d").and_then(|v| v.as_str().len().try_into().ok()).unwrap_or(0);

        let step_i = caps.name("step").map(|v|v.as_str().to_string());
        let step_d = caps.name("step_d").and_then(|v| v.as_str().len().try_into().ok()).unwrap_or(0);


        Ok(((from_i, from_d), (to_i, to_d), step_i.map(|step| (step, step_d))))
    }).unwrap_or_else(||Err(ThoosaError::new(format!("can't parse range {range}"))))

}


pub fn make_combos<V:Clone>(iters: Vec<impl Iterator<Item=V>>) -> Vec<Vec<V>> {
    let mut combos: Vec<Vec<V>> = vec![vec![]];

    for iter in iters {
        let mut new_combos = vec![];
        for item in iter {
            for combo in &combos {
                let mut new_combo = combo.clone();
                new_combo.push(item.clone());
                new_combos.push(new_combo);
            }
        }
        combos = new_combos;
    }

    combos
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
    use std::fmt::Display;
    use std::sync::Arc;
    use std::vec::IntoIter;
    use clap::{Args, Parser};


    use crate::{batch_windows, clc_num_batches, make_combos, group_slices, parse_range_str, RangeOrValue, ThoosaError, Resources};
    use crate::pulse_v;

    #[ctor::ctor]
    fn init() {
        super::init_test_logging();
    }

    #[test]
    fn test_join_with_offset(){
        let pulse = pulse_v(50., 10., 100, 20.);
        let v = super::join_with_offset::<f32>(0., 10.0, &[(100, pulse.clone())], super::random_saw(550));
        assert_eq!(550, v.len());

        let v = super::join_with_offset::<f32>(0., 10.0, &[(100, pulse.clone())], super::random_saw(2200));
        assert_eq!(2200, v.len());

    }

    #[test]
    fn test_group_slices(){
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


        let groups = group_slices(pairs.into_iter());

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

    #[test]
    fn test_batch_windows_1000(){
        let input_n = 1000;
        let motif_width = 100;
        let step = 100;

        let batches = batch_windows(input_n, motif_width, step, 10);

        assert_eq!(
            vec![0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
            batches.0[0]
        );

        assert_eq!(batches.0.len(), 1);
        assert!(batches.1.is_empty());

    }

    #[test]
    fn test_batch_windows_3500(){
        let input_n = 3500;
        let motif_width = 100;
        let step = 100;

        let batches = batch_windows(input_n, motif_width, step, 10);

        assert_eq!(vec![0,    100,  200,  300,  400,  500,  600,  700,  800,  900], batches.0[0]);
        assert_eq!(vec![1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900], batches.0[1]);
        assert_eq!(vec![2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900], batches.0[2]);
        assert_eq!(vec![3000, 3100, 3200, 3300, 3400], batches.1)
    }

    #[test]
    fn test_clc_num_batches(){
        assert_eq!(
            clc_num_batches(1000, 100, 10, 1000),
            1
        );

        assert_eq!(
            clc_num_batches(1000, 100, 10, 90),
            1
        );

        assert_eq!(
            clc_num_batches(1000, 100, 10, 50),
            2
        );

        assert_eq!(
            clc_num_batches(1000, 100, 10, 30),
            3
        );
    }

    #[test]
    fn test_parse_corr_labels()  -> Result<(), ThoosaError>{
        assert_eq!(
            vec![(1, "seizure_vote".into()), (0, "lpd_vote".into()), (0, "gpd_vote".into()), (0, "lrda_vote".into()), (0, "grda_vote".into()), (0, "other_vote".into())],
            super::parse_corr_labels("[(1, \"seizure_vote\"), (0, \"lpd_vote\"), (0, \"gpd_vote\"), (0, \"lrda_vote\"), (0, \"grda_vote\"), (0, \"other_vote\")]")?
        );

        Ok(())
    }

    #[test]
    fn test_parse_range_or_value_range() -> Result<(), ThoosaError>{
        use crate::parse_range_or_value;

        #[derive(Debug, Parser, Clone, PartialEq)]
        struct TestParams {

            #[arg(long, value_parser=parse_range_or_value::<i32>)]
            window: RangeOrValue<i32>,


            #[arg(long, value_parser=parse_range_or_value::<f32>)]
            step: RangeOrValue<f32>
        }

        let params = TestParams::try_parse_from(vec!["", "--step=1.1..2.2", "--window=1..2"].into_iter()).map_err(|e|ThoosaError::new(e.to_string()))?;//TODO error


        assert_eq!(TestParams { window: RangeOrValue::Range{range: 1..2, step:None}, step: RangeOrValue::Range{range: 1.1..2.2, step:None} }, params);

        Ok(())
    }

    #[test]
    fn test_resources_param()-> Result<(), ThoosaError>{
        use crate::parse_resources;

        #[derive(Debug, Parser, Clone, PartialEq)]
        struct TestParams {

            #[arg(long,value_parser=parse_resources::<Resources>)]
            resources: Resources,
        }

        let params = TestParams::try_parse_from(vec!["", "--resources=list:aaa,bbb,ccc"].into_iter()).map_err(|e|ThoosaError::new(e.to_string()))?;//TODO error

        assert_eq!(
            Resources::List(vec!["aaa".into(), "bbb".into(), "ccc".into()]),
            params.resources
        );
        let params = TestParams::try_parse_from(vec!["", "--resources=set:aaa"].into_iter()).map_err(|e|ThoosaError::new(e.to_string()))?;//TODO error
        assert_eq!(
            Resources::Set("aaa".into()),
            params.resources
        );

        let params = TestParams::try_parse_from(vec!["", "--resources=group:bbb"].into_iter()).map_err(|e|ThoosaError::new(e.to_string()))?;//TODO error
        assert_eq!(
            Resources::Group("bbb".into()),
            params.resources
        );

        Ok(())
    }

    #[test]
    fn test_parse_range_or_value_range_with_step() -> Result<(), ThoosaError>{
        use crate::parse_range_or_value;

        #[derive(Debug, Parser, Clone, PartialEq)]
        struct TestParams {

            #[arg(long, value_parser=parse_range_or_value::<i32>)]
            window: RangeOrValue<i32>,


            #[arg(long, value_parser=parse_range_or_value::<f64>)]
            step: RangeOrValue<f64>
        }

        let params = TestParams::try_parse_from(vec!["", "--step=-1.01..2.2:0.1", "--window=-20..20:2"].into_iter()).map_err(|e|ThoosaError::new(e.to_string()))?;//TODO error


        assert_eq!(TestParams { window: RangeOrValue::Range{range: -20..20, step:Some((2, 1))}, step: RangeOrValue::Range{range: -1.01..2.2, step:Some((0.1, 100)) } }, params);

        assert_eq!(
            (-20..20).step_by(2).collect::<Vec<i32>>(),
            params.window.into_iter().collect::<Vec<i32>>()
        );


        assert_eq!(
            (-101..220).step_by(10).map(|i|i as f64/100.).collect::<Vec<f64>>(),
            params.step.into_iter().collect::<Vec<f64>>()
        );

        Ok(())
    }


    #[test]
    fn test_parse_range_or_value_value() -> Result<(), ThoosaError>{
        use crate::parse_range_or_value;

        #[derive(Debug, Parser, Clone, PartialEq)]
        struct TestParams {

            #[arg(long, value_parser=parse_range_or_value::<i32>)]
            window: RangeOrValue<i32>,


            #[arg(long, value_parser=parse_range_or_value::<f32>)]
            step: RangeOrValue<f32>
        }

        let params = TestParams::try_parse_from(vec!["", "--step=1.1", "--window=10"].into_iter()).map_err(|e|ThoosaError::new(e.to_string()))?;//TODO error

        assert_eq!(TestParams { window: RangeOrValue::Value(10), step: RangeOrValue::Value(1.1) }, params);


        Ok(())
    }


    #[test]
    fn test_parser_regex_str() -> Result<(), ThoosaError> {
        let ((from, from_d_len), (to, to_d_len), step) = parse_range_str("1.22..333.4444")?;

        assert_eq!("1.22".to_string(), from);
        assert_eq!(2, from_d_len);

        assert_eq!("333.4444".to_string(), to);
        assert_eq!(4, to_d_len);

        assert!(step.is_none());

        let ((from, from_d_len), (to, to_d_len), step) = parse_range_str("-1.22..-333.4444")?;

        assert_eq!("-1.22".to_string(), from);
        assert_eq!(2, from_d_len);

        assert_eq!("-333.4444".to_string(), to);
        assert_eq!(4, to_d_len);

        assert!(step.is_none());



        let ((from, from_d_len), (to, to_d_len), Some((step, step_d_len))) = parse_range_str("-1.22..-333.4444:7.8")? else {
            panic!("no step");
        };

        assert_eq!("-1.22".to_string(), from);
        assert_eq!(2, from_d_len);

        assert_eq!("-333.4444".to_string(), to);
        assert_eq!(4, to_d_len);

        assert_eq!("7.8".to_string(), step);
        assert_eq!(1, step_d_len);


        let ((from, from_d_len), (to, to_d_len), Some((step, step_d_len))) = parse_range_str("-1.22..-333.4444:-7.8")? else {
            panic!("no step");
        };
        // println!("from_i={from} from_d={from_d_len:?} to={to} to_d_len={to_d_len} step={step:?}");

        assert_eq!("-1.22".to_string(), from);
        assert_eq!(2, from_d_len);

        assert_eq!("-333.4444".to_string(), to);
        assert_eq!(4, to_d_len);

        assert_eq!("-7.8".to_string(), step);
        assert_eq!(1, step_d_len);



        Ok(())
    }

    #[test]
    fn test_make_combos() {
        assert_eq!(
            [
               ["a1", "b1", "c1"],
               ["a2", "b1", "c1"],
               ["a3", "b1", "c1"],
               ["a1", "b1", "c2"],
               ["a2", "b1", "c2"],
               ["a3", "b1", "c2"]
           ].into_iter().map(|v|v.to_vec()).collect::<Vec<_>>(),
            make_combos(
               vec![
                   vec!["a1", "a2", "a3"].into_iter(),
                   vec!["b1"].into_iter(),
                   vec!["c1", "c2"].into_iter()
               ]
           )
        );

        assert_eq!(
            [
                ["a1", "b1", ],
                ["a2", "b1", ]
            ].into_iter().map(|v|v.to_vec()).collect::<Vec<_>>(),
            make_combos(
                vec![
                    vec!["a1", "a2"].into_iter(),
                    vec!["b1"].into_iter(),
                ]
            )
        );

        assert_eq!(
            [
                ["a1"]
            ].into_iter().map(|v|v.to_vec()).collect::<Vec<_>>(),
            make_combos(
                vec![
                    vec!["a1"].into_iter(),
                ]
            )
        );

    }

}