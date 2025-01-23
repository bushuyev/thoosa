use std::collections::HashMap;
use std::ops::Range;
use crate::{RangeOrValue, ThoosaError};


#[derive(Debug, PartialEq, Clone)]
pub enum Val {
    Real(f64),
    Int(i64),
    UInt(u32),
    Str(String),
    Bool(bool)
}

pub struct NamedVal{name: String, val:Val}
pub struct NamedRange{name: String, range:Range<Val>, step:(Val, u32)}


impl From<(&str, u32)> for NamedVal {
    fn from((name, value): (&str, u32)) -> Self {
        NamedVal{name: name.to_string(), val:value.into()}
    }
}

impl TryFrom<(&str, &RangeOrValue<f64>)> for NamedRange {
    type Error = ();

    fn try_from((name, value): (&str, &RangeOrValue<f64>)) -> Result<Self, Self::Error> {
        if let RangeOrValue::Range{range,step} = value {
            if let Some(step) = step {
                return Ok(NamedRange {
                    name: name.into(),
                    range: Val::Real(range.start)..Val::Real(range.end),
                    step: (Val::Real(step.0), step.1)
                })
            }
        }
        Err(())
    }
}
//
// impl TryFrom<&RangeOrValue<i32>> for Hyperparameter {
//     type Error = ();
//
//     fn try_from(value: &RangeOrValue<i32>) -> Result<Self, Self::Error> {
//         if let RangeOrValue::Range{range, ..} = value {
//             Ok(Hyperparameter::UniformInt {
//                 name: "param_0".into(),
//                 lower: range.start as i64, //TODO as i64
//                 upper: range.end as i64
//             })
//         } else {
//             Err(())
//         }
//     }
// }


impl TryFrom<(&str, &RangeOrValue<u32>)> for NamedRange {
    type Error = ();

    fn try_from((name, value): (&str, &RangeOrValue<u32>)) -> Result<Self, Self::Error> {
        if let RangeOrValue::Range{range,step} = value {
            if let Some(step) = step {
                return Ok(NamedRange {
                    name: name.into(),
                    range: Val::UInt(range.start)..Val::UInt(range.end),
                    step: (Val::UInt(step.0), step.1)
                })
            }
        }
        Err(())
    }
}

//TODO remove
impl Val {

    pub fn as_usize(&self) -> Result<usize, ThoosaError> {
        match self {
            Self::Int(v)=> Ok(*v as usize),
            _=> Err(ThoosaError::new("can't be usize"))
        }
    }

    pub fn as_u32(&self) -> Result<u32, ThoosaError> {
        match self {
            Self::UInt(v)=> Ok(*v),
            _=> Err(ThoosaError::new("can't be u32"))
        }
    }


    pub fn as_i64(&self) -> Result<i64, ThoosaError> {
        match self {
            Self::Int(v)=> Ok(*v as i64),
            _=> Err(ThoosaError::new("can't be i64"))
        }
    }

    pub fn as_f32(&self) -> Result<f32, ThoosaError> {
        match self {
            Self::Real(v)=> Ok(*v as f32),
            _=> Err(ThoosaError::new("can't be f32"))
        }
    }

    pub fn as_str(&self) -> Result<String, ThoosaError> {
        match self {
            Self::Str(v)=> Ok(v.clone()),
            Self::Int(v)  => Ok(format!("{}", v)),
            Self::UInt(v)=> Ok(format!("{}", v)),
            Self::Real(v)=> Ok(format!("{}", v)),
            Val::Bool(v) =>Ok(format!("{}", v)),
        }
    }
}

//TODO
impl TryFrom<Val> for f64 {
    type Error = ThoosaError;

    fn try_from(value: Val) -> Result<f64, ThoosaError> {
        match value {
            Val::UInt(v)=> v.try_into().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            Val::Real(v)=> v.try_into().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            Val::Str(v)=> v.parse().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            _ => Err(ThoosaError::new(format!("can't convert {value:?}")))
        }
    }
}

//TODO
impl TryFrom<Val> for u32 {
    type Error = ThoosaError;

    fn try_from(value: Val) -> Result<u32, ThoosaError> {
        match value {
            Val::UInt(v)=> v.try_into().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            Val::Str(v)=> v.parse().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            _ => Err(ThoosaError::new(format!("can't convert {value:?}")))
        }
    }
}

impl From<f32> for Val {
    fn from(v: f32) -> Self {
        Val::Real(v as f64)
    }
}

impl From<usize> for Val {
    fn from(v: usize) -> Self {
        Val::Int(v as i64)
    }
}

impl From<i64> for Val {
    fn from(v: i64) -> Self {
        Val::Int(v)
    }
}

impl From<i32> for Val {
    fn from(v: i32) -> Self {
        Val::Int(v as i64)
    }
}

impl From<u32> for Val {
    fn from(v: u32) -> Self {
        Val::UInt(v)
    }
}


impl From<&u32> for Val {
    fn from(v: &u32) -> Self {
        Val::Int(*v as i64)
    }
}


impl From<f64> for Val {
    fn from(v: f64) -> Self {
        Val::Real(v)
    }
}


impl From<&str> for Val {
    fn from(v: &str) -> Self {
        Val::Str(v.to_string())
    }
}

impl From<String> for Val {
    fn from(v: String) -> Self {
        Val::Str(v)
    }
}

impl From<&f32> for Val {
    fn from(v: &f32) -> Self {
        Val::Real(*v as f64)
    }
}

impl From<&usize> for Val {
    fn from(v: &usize) -> Self {
        Val::Int(*v as i64)
    }
}

impl From<&i64> for Val {
    fn from(v: &i64) -> Self {
        Val::Int(*v)
    }
}

impl From<&i32> for Val {
    fn from(v: &i32) -> Self {
        Val::Int(*v as i64)
    }
}
impl From<bool> for Val {
    fn from(v: bool) -> Self {
        Val::Bool(v)
    }
}

pub fn prefix_properties(prefix:&str, map: HashMap<String, Val>) -> HashMap<String, Val>{
    map.into_iter().fold(HashMap::new(), |mut a, v| {
        a.insert(format!("{}.{}", prefix, v.0), v.1);
        a
    })
}

pub fn unprefix_properties(prefix:&str, map: &HashMap<String, Val>) -> HashMap<String, Val> {
    map.into_iter().fold(HashMap::new(),  |mut a, v| {
        if let Some(p) = v.0.strip_prefix(prefix) {
            a.insert(p.strip_prefix(".").expect("no . after prefix").to_string(), v.1.clone());
        }
        a
    })
}
