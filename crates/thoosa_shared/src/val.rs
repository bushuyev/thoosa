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

impl Val {
    pub fn to_i64(&self, p:u32) -> i64 {

        match self {
            Val::Real(v) => {
                (v * 10_f64.powi(p as i32)) as i64
            }
            Val::Int(v) => {
                *v
            }
            Val::UInt(v) => {
                *v as i64
            }
            Val::Str(v) => {
                v.parse().expect(format!("Could not parse {v} string").as_str())
            }
            Val::Bool(v) => {
                if *v { 1 } else { 0 }
            }
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct NamedVal{
    name: String,
    val:Val
}

impl NamedVal {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn val(&self) -> &Val {
        &self.val
    }
}

#[derive(Debug)]
pub struct NamedRange{
    name: String,
    range:Range<Val>,
    step:(Val, u32)
}


impl NamedRange {
    pub fn new(name: &str, range: Range<Val>, step: (Val, u32)) -> Self {
        Self { name:name.into(), range, step }
    }
}
//
// impl From<(&str, u32)> for NamedVal {
//     fn from((name, value): (&str, u32)) -> Self {
//         NamedVal{name: name.to_string(), val:value.into()}
//     }
// }

impl From<(&str, Val)> for NamedVal {
    fn from((name, val): (&str, Val)) -> Self {
        NamedVal{name: name.to_string(), val}
    }
}


// impl From<(&str, &Val)> for NamedVal {
//     fn from((name, val): (&str, &Val)) -> Self {
//         NamedVal{name: name.to_string(), val:val.clone()}
//     }
// }

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


impl From<&NamedRange> for Vec<NamedVal> {
    fn from(value: &NamedRange) -> Self {

        let start:i64 = value.range.start.to_i64(value.step.1);
        let end:i64 = value.range.end.to_i64(value.step.1);
        let step:usize = value.step.0.to_i64(value.step.1) as usize;

        (start..end).step_by(step).map(|i| (
            value.name.as_str(),
            <Val as Into<Val>>::into(i.into())
        ).into()).collect::<Vec<NamedVal>>()
    }
}

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

impl TryFrom<(&str, &RangeOrValue<u32>)> for NamedVal {
    type Error = ();

    fn try_from((name, value): (&str, &RangeOrValue<u32>)) -> Result<Self, Self::Error> {
        if let RangeOrValue::Value(v) = value {
            return Ok(NamedVal {
                name: name.into(),
                val: v.into()
            });
        }
        Err(())
    }
}


impl TryFrom<Val> for f64 {
    type Error = ThoosaError;

    fn try_from(value: Val) -> Result<f64, ThoosaError> {
        match value {
            Val::Int(v)=> Ok(v as f64), //TODO
            Val::UInt(v)=> v.try_into().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            Val::Real(v)=> v.try_into().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            Val::Str(v)=> v.parse().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            _ => Err(ThoosaError::new(format!("can't convert {value:?}")))
        }
    }
}


impl TryFrom<Val> for i64 {
    type Error = ThoosaError;

    fn try_from(value: Val) -> Result<i64, ThoosaError> {
        match value {
            Val::Int(v)=> v.try_into().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            Val::UInt(v)=> v.try_into().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            Val::Str(v)=> v.parse().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            _ => Err(ThoosaError::new(format!("can't convert {value:?}")))
        }
    }
}


impl TryFrom<&Val> for i64 {
    type Error = ThoosaError;

    fn try_from(value: &Val) -> Result<i64, ThoosaError> {
        match value {
            Val::Int(v)=> (*v).try_into().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            Val::UInt(v)=> (*v).try_into().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            Val::Str(v)=> v.parse().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            _ => Err(ThoosaError::new(format!("can't convert {value:?}")))
        }
    }
}

impl TryFrom<Val> for u32 {
    type Error = ThoosaError;

    fn try_from(value: Val) -> Result<u32, ThoosaError> {
        match value {
            Val::Int(v)=> v.try_into().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            Val::UInt(v)=> v.try_into().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            Val::Str(v)=> v.parse().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            _ => Err(ThoosaError::new(format!("can't convert {value:?}")))
        }
    }
}


impl TryFrom<Val> for f32 {
    type Error = ThoosaError;

    fn try_from(value: Val) -> Result<f32, ThoosaError> {
        match value {
            Val::Real(v)=> Ok(v as f32),//TODO safe
            Val::Int(v)=> Ok(v as f32),
            Val::UInt(v)=> Ok(v as f32),
            Val::Str(v)=> v.parse().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            _ => Err(ThoosaError::new(format!("can't convert {value:?}")))
        }
    }
}


impl TryFrom<&Val> for f64 {
    type Error = ThoosaError;

    fn try_from(value: &Val) -> Result<f64, ThoosaError> {
        match value {
            Val::UInt(v)=> (*v).try_into().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            Val::Real(v)=> (*v).try_into().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            Val::Str(v)=> v.parse().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            _ => Err(ThoosaError::new(format!("can't convert {value:?}")))
        }
    }
}


impl TryFrom<&Val> for u32 {
    type Error = ThoosaError;

    fn try_from(value: &Val) -> Result<u32, ThoosaError> {
        match value {
            Val::UInt(v)=> Ok(*v),
            Val::Str(v)=> v.parse().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            _ => Err(ThoosaError::new(format!("can't convert {value:?}")))
        }
    }
}


impl TryFrom<&Val> for f32 {
    type Error = ThoosaError;

    fn try_from(value: &Val) -> Result<f32, ThoosaError> {
        match value {
            Val::Real(v)=> Ok(*v as f32),
            Val::UInt(v)=> Ok(*v as f32),
            Val::Str(v)=> v.parse().map_err(|e|ThoosaError::new(format!("e={e:?}"))),
            _ => Err(ThoosaError::new(format!("can't convert {value:?}")))
        }
    }
}

impl TryFrom<&Val> for &str {
    type Error = ThoosaError;

    fn try_from(value: &Val) -> Result<&'static str, ThoosaError> {
        match value {
            v => v.try_into().map_err(|e|ThoosaError::new(format!("e={e:?}")))
        }
    }
}

impl TryFrom<Val> for usize {
    type Error = ThoosaError;

    fn try_from(value: Val) -> Result<usize, ThoosaError> {
        match value {
            Val::Int(v)=> Ok(v as usize),
            Val::UInt(v)=> Ok(v as usize),
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


impl TryFrom<&NamedVal> for f64 {
    type Error = ThoosaError;

    fn try_from(value: &NamedVal) -> Result<Self, Self::Error> {
        value.val.clone().try_into()
    }
}


impl TryFrom<&NamedVal> for f32 {
    type Error = ThoosaError;

    fn try_from(value: &NamedVal) -> Result<Self, Self::Error> {
        value.val.clone().try_into()
    }
}

impl TryFrom<&NamedVal> for u32 {
    type Error = ThoosaError;

    fn try_from(value: &NamedVal) -> Result<Self, Self::Error> {
        value.val.clone().try_into()
    }
}