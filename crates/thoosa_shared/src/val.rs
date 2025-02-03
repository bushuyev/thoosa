use std::collections::HashMap;
use std::ops::Range;
use num_traits::ToPrimitive;
use crate::{round_to_digits, RangeOrValue, ThoosaError};


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

#[derive(Clone, Debug)]
pub struct NamedRange{
    name: String,
    range:Range<Val>,
    step:(Val, u32)
}


impl NamedRange {
    pub fn new(name: &str, range: Range<Val>, step: (Val, u32)) -> Self {
        Self { name:name.into(), range, step }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn range(&self) -> &Range<Val> {
        &self.range
    }

    pub fn step(&self) -> &(Val, u32) {
        &self.step
    }

}

#[derive(PartialEq)]
enum IterationKind {
    Int,
    UInt,
    Real,
    Bool,
    Unsupported,
}



pub struct NamedRangeIter {
    name: String,
    // We'll store everything as floats internally:
    current: f64,
    end: f64,
    step: f64,
    digits: u32,
    // Whether we should produce Int/UInt/Real/Bool on output:
    iteration_kind: IterationKind,
    // If done == true, we've finished iteration
    done: bool,
    // track whether step is forward (current < end) or backward (current > end)
    forward: bool,
}


impl Iterator for NamedRangeIter {
    type Item = NamedVal;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Check if we've passed beyond the half-open boundary
        //   * If forward and current >= end => done
        //   * If backward and current <= end => done
        let out_of_bounds = if self.forward {
            self.current >= self.end
        } else {
            self.current <= self.end
        };

        if out_of_bounds {
            self.done = true;
            return None;
        }

        // Produce the correct variant according to iteration_kind
        let val = match self.iteration_kind {
            IterationKind::Int => Val::Int(self.current as i64),
            IterationKind::UInt => {
                // guard against negative or large float => might be out-of-range
                // but for simplicity, just cast:
                Val::UInt(self.current as u32)
            }
            IterationKind::Real => {
                let rounded = round_to_digits(self.current, self.digits);
                Val::Real(rounded)
            }
            IterationKind::Bool => {
                Val::Bool(self.current >= 0.5)
            }
            IterationKind::Unsupported => {
                return None;
            }
        };

        let item = NamedVal {
            name: self.name.clone(),
            val,
        };

        let next_val = self.current + self.step;

        // Round if we are working in Real mode, so error doesn’t accumulate
        let next_val = if matches!(self.iteration_kind, IterationKind::Real) {
            round_to_digits(next_val, self.digits)
        } else {
            next_val
        };
        self.current = next_val;

        Some(item)
    }
}

impl IntoIterator for NamedRange {
    type Item = NamedVal;
    type IntoIter = NamedRangeIter;

    fn into_iter(self) -> Self::IntoIter {
        let start_val = &self.range.start;
        let end_val = &self.range.end;
        let (step_val, digits) = &self.step;

        // 1) Convert start/end/step into f64 if possible
        // 2) Decide iteration_kind from the start type
        //    (or unify to Real if there's a mismatch).
        let (start_f, kind_start) = val_to_f64_and_kind(start_val);
        let (end_f, kind_end) = val_to_f64_and_kind(end_val);
        let (step_f, _) = val_to_f64_and_kind(step_val);

        // If either is unsupported or there's a mismatch you can't handle:
        let iteration_kind = if kind_start == IterationKind::Unsupported {
            IterationKind::Unsupported
        } else if kind_end == IterationKind::Unsupported {
            // Alternatively, unify to Real or panic!
            IterationKind::Unsupported
        } else {
            // If the start was Real, we use Real iteration
            // Otherwise we keep the start's iteration kind
            match kind_start {
                IterationKind::Real => IterationKind::Real,
                _ => {
                    // If end or step is Real, unify all to Real
                    // else keep start kind
                    if kind_end == IterationKind::Real || matches!(step_val, Val::Real(_)) {
                        IterationKind::Real
                    } else {
                        kind_start
                    }
                }
            }
        };

        // If step_f == 0.0, you might handle that specially (avoid infinite loop).
        // For demonstration, skip that check.

        NamedRangeIter {
            name: self.name,
            current: start_f,
            end: end_f,
            step: step_f,
            digits: *digits,
            done: false,
            forward: start_f < end_f,
            iteration_kind,
        }
    }
}

fn val_to_f64_and_kind(v: &Val) -> (f64, IterationKind) {
    match v {
        Val::Int(i)  => (*i as f64, IterationKind::Int),
        Val::UInt(u) => (*u as f64, IterationKind::UInt),
        Val::Real(r) => (*r, IterationKind::Real),
        Val::Bool(b) => (if *b { 1.0 } else { 0.0 }, IterationKind::Bool),
        Val::Str(_)  => (0.0, IterationKind::Unsupported),
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
            Val::Real(v)=> Ok(v as u32), //v.to_u32().ok_or_else(|| ThoosaError::new(format!("can't convert {value:?}"))),
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
            Val::Int(v)=> (*v).to_f64().ok_or_else(|| ThoosaError::new(format!("can't convert {value:?}"))),
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
            Val::Real(v)=> Ok(*v as u32), //TODO v.to_u32().ok_or_else(|| ThoosaError::new(format!("can't convert {value:?}"))),
            Val::Int(v)=> (*v).to_u32().ok_or_else(|| ThoosaError::new(format!("can't convert {value:?}"))),
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

impl TryFrom<&Val> for String {
    type Error = ThoosaError;

    fn try_from(value: &Val) -> Result<String, ThoosaError> {
        match value {
            Val::Str(v) => Ok(v.clone()), // Return a string slice with the correct lifetime
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

impl From<&f64> for Val {
    fn from(v: &f64) -> Self {
        Val::Real(*v)
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

#[cfg(test)]
mod tests {
    use log::debug;
    use crate::val::{NamedRange, NamedVal, Val};


    #[test]
    fn test_named_range_int_iteration() {
        let named_range = NamedRange {
            name: "SomeName".to_string(),
            range: Val::Int(-1)..Val::Int(3),
            step: (Val::Int(1), 2),
        };

        let result: Vec<NamedVal> = named_range.into_iter().collect();

        let expected = vec![
            NamedVal { name: "SomeName".to_string(), val: Val::Int(-1) },
            NamedVal { name: "SomeName".to_string(), val: Val::Int(0)  },
            NamedVal { name: "SomeName".to_string(), val: Val::Int(1)  },
            NamedVal { name: "SomeName".to_string(), val: Val::Int(2)  },
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_named_range_float_iteration() {
        let named_range = NamedRange {
            name: "FloatRange".to_string(),
            range: Val::Real(0.0)..Val::Real(1.0),
            step: (Val::Real(0.3), 2),
        };

        let result: Vec<NamedVal> = named_range.into_iter().collect();

        let expected = vec![
            NamedVal { name: "FloatRange".to_string(), val: Val::Real(0.0)  },
            NamedVal { name: "FloatRange".to_string(), val: Val::Real(0.3)  },
            NamedVal { name: "FloatRange".to_string(), val: Val::Real(0.6)  },
            NamedVal { name: "FloatRange".to_string(), val: Val::Real(0.9)  },
        ];

        assert_eq!(result, expected);
    }

}