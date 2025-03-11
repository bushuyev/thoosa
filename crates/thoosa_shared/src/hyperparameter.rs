use std::fmt::{Debug, Display, Formatter};
use std::mem;
use std::ops::{Add, Range};
use rand::distributions::Uniform;
use rand::Rng;
use log::trace;
use crate::opt_rng::opt_rnd;
use crate::RangeOrValue;
use crate::val::Val;
use once_cell::sync::Lazy;

pub trait FieldName: Display + Debug + Clone  {//+ for<'a>  Add<&'a dyn AsRef<Self>>

}


impl FieldName for String {}

static TRUE_CAT: Lazy<String> = Lazy::new(|| String::from("TRUE_CAT"));
static FALSE_CAT: Lazy<String> = Lazy::new(|| String::from("FALSE_CAT"));


#[derive(Debug, Clone, PartialEq)]
pub enum Hyperparameter<T: FieldName> {
    UniformFloat{name: T, lower:f64, upper:f64},
    UniformInt{name: T, lower:i64, upper:i64},
    Categorical{name: T, values:Vec<&'static str>}
}

impl <T: FieldName> Hyperparameter<T> {

    pub fn random_sample(&self) -> Val {
        match self {
            Hyperparameter::UniformFloat { name, lower, upper } => {
                let uniform = opt_rnd().sample(Uniform::new(0., 1.));
                let sample = lower + uniform * (upper - lower) ;
                trace!("UniformFloat={uniform:?} lb={lower} ub={upper} sample={sample} ");
                Val::Real(sample)
            }
            Hyperparameter::Categorical { name, values } => {
                let uniform = opt_rnd().sample(Uniform::new(0., 1.));
                let sample = uniform * values.len() as f32;
                trace!("Categorical={sample:?} values={values:?} ");
                Val::Int(sample as i64)
            }
            Hyperparameter::UniformInt { name, lower, upper } => {
                let uniform = opt_rnd().sample(Uniform::new(0., 1.));
                let sample = *lower as f64+ uniform * ((upper - lower) as f64) ;
                trace!("UniformInt={uniform:?} lb={lower} ub={upper} sample={sample} ");
                Val::Int(sample as i64)
            }
        }
    }

    pub fn name(&self) -> &T {
        match self {
            Hyperparameter::UniformFloat { name, .. } => name,
            Hyperparameter::UniformInt { name, .. } => name,
            Hyperparameter::Categorical { name, .. } => name,
        }
    }

}

impl <T: FieldName> From<(T, &Range<f64>)> for Hyperparameter<T> {
    fn from(value: (T, &Range<f64>)) -> Self {
        Hyperparameter::UniformFloat {
            name: value.0,
            lower: value.1.start.into(),
            upper: value.1.end.into()
        }
    }
}

impl <T: FieldName> From<(T, &Range<u32>)> for Hyperparameter<T> {
    fn from(value: (T, &Range<u32>)) -> Self {
        Hyperparameter::UniformInt {
            name: value.0,
            lower: value.1.start.into(),
            upper: value.1.end.into()
        }
    }
}

impl <T: FieldName> From<(T, &Range<i32>)> for Hyperparameter<T> {
    fn from(value: (T, &Range<i32>)) -> Self {
        Hyperparameter::UniformInt {
            name: value.0,
            lower: value.1.start.into(),
            upper: value.1.end.into()
        }
    }
}

impl <T: FieldName> From<(T, &Range<i64>)> for Hyperparameter<T> {
    fn from(value: (T, &Range<i64>)) -> Self {
        Hyperparameter::UniformInt {
            name: value.0,
            lower: value.1.start.into(),
            upper: value.1.end.into()
        }
    }
}


impl <T: FieldName> From<(T, &Range<f32>)> for Hyperparameter<T> {
    fn from(value: (T, &Range<f32>)) -> Self {
        Hyperparameter::UniformFloat {
            name: value.0,
            lower: value.1.start.into(),
            upper: value.1.end.into()
        }
    }
}

impl <T: FieldName> From<(T, &Range<bool>)> for Hyperparameter<T> {
    fn from(value: (T, &Range<bool>)) -> Self {
        Hyperparameter::Categorical {
            name: value.0,

            values: vec![&TRUE_CAT, &FALSE_CAT],
        }
    }
}


impl <T: FieldName> TryFrom<(T, &RangeOrValue<f64>)> for Hyperparameter<T> {
    type Error = ();

    fn try_from((name, value): (T, &RangeOrValue<f64>)) -> Result<Self, Self::Error> {
        if let RangeOrValue::Range{range,step} = value {
            if step.is_none() {
                return Ok(Hyperparameter::UniformFloat {
                    name,
                    lower: range.start,
                    upper: range.end
                });
            }
        }
        Err(())
    }
}

impl <T: FieldName> TryFrom<(T, &RangeOrValue<i32>)> for Hyperparameter<T> {
    type Error = ();

    fn try_from((name, value): (T, &RangeOrValue<i32>)) -> Result<Self, Self::Error> {
        if let RangeOrValue::Range{range, step} = value {
            if step.is_none() {
                return Ok(Hyperparameter::UniformInt {
                    name,
                    lower: range.start as i64, //TODO as i64
                    upper: range.end as i64
                });
            }
        }
        Err(())
    }
}

impl <T: FieldName> TryFrom<(T, &RangeOrValue<u32>)> for Hyperparameter<T> {
    type Error = ();

    fn try_from((name, value): (T, &RangeOrValue<u32>)) -> Result<Self, Self::Error> {
        if let RangeOrValue::Range{range, step} = value {
            if step.is_none() {
                return Ok(Hyperparameter::UniformInt {
                    name,
                    lower: range.start as i64, //TODO as i64
                    upper: range.end as i64
                });
            }
        }
        Err(())
    }
}