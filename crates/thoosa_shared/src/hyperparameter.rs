use std::fmt::{Debug, Display, Formatter};
use std::mem;
use std::ops::{Add, Range};
use rand::distributions::Uniform;
use rand::Rng;
use log::trace;
use crate::opt_rng::opt_rnd;
use crate::RangeOrValue;
use crate::val::Val;

pub trait FieldName: Display + Debug + Clone  {//+ for<'a>  Add<&'a dyn AsRef<Self>>

}


impl FieldName for String {}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldsPath<T: FieldName> {
    path:Vec<T>
}

impl <T: FieldName> FieldsPath<T> {
    pub fn new(first:T) -> Self {
        FieldsPath { path:vec![first] }
    }

    pub fn root()->Self {
        FieldsPath { path:Vec::new() }
    }

    pub fn path(&self) -> &T {//TODO remove
        &self.path[0]
    }
}

impl From<&str> for FieldsPath<String> {
    fn from(path: &str) -> Self {
        FieldsPath {
            path:path.split(".").map(|p| p.to_string()).collect::<Vec<String>>()
        }
    }
}

impl <T: FieldName> From<T> for FieldsPath<T> {
    fn from(field: T) -> Self {
        FieldsPath {
            path:vec![field]
        }
    }
}

impl <T: FieldName> Add<FieldsPath<T>> for FieldsPath<T> {
    type Output = Self;

    fn add(self, rhs: FieldsPath<T>) -> Self::Output {
        let mut params = self.path.clone();
        params.extend(rhs.path);

        FieldsPath {
            path: params,
        }
    }
}

impl <T: FieldName> Display for FieldsPath<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.path.fmt(f)
    }
}

// impl<T: ParamName> Extend<Self> for ParamsPath<T> {
//     fn extend<I: IntoIterator<Item=Self>>(&mut self, iter: I) {
//         todo!()
//     }
// }

impl <T: FieldName> FieldName for FieldsPath<T> {}

// impl <T:ParamName> Extend<T> for ParamsPath<T> {
//
//     fn extend<I: IntoIterator<Item=T>>(&mut self, iter: I) {
//         for item in iter {
//             self.params.push(item);
//         }
//     }
// }



// impl <T:ParamName> MyAdd for T {
//     type Output = T;
//
//     fn add(self, rhs: &T) -> Self::Output {
//         todo!()
//     }
// }


// impl <'a> ParamName for &'a str {}

#[derive(Debug, Clone, PartialEq)]
pub enum Hyperparameter<T: FieldName> {
    UniformFloat{name: FieldsPath<T>, lower:f64, upper:f64},
    UniformInt{name: FieldsPath<T>, lower:i64, upper:i64},
    Categorical{name: FieldsPath<T>, values:Vec<String>}
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

    pub fn name(&self) -> &FieldsPath<T> {
        match self {
            Hyperparameter::UniformFloat { name, .. } => name,
            Hyperparameter::UniformInt { name, .. } => name,
            Hyperparameter::Categorical { name, .. } => name,
        }
    }

}

impl <T: FieldName> From<(FieldsPath<T>, &Range<f64>)> for Hyperparameter<T> {
    fn from(value: (FieldsPath<T>, &Range<f64>)) -> Self {
        Hyperparameter::UniformFloat {
            name: value.0,
            lower: value.1.start.into(),
            upper: value.1.end.into()
        }
    }
}

impl <T: FieldName> From<(FieldsPath<T>, &Range<u32>)> for Hyperparameter<T> {
    fn from(value: (FieldsPath<T>, &Range<u32>)) -> Self {
        Hyperparameter::UniformInt {
            name: value.0,
            lower: value.1.start.into(),
            upper: value.1.end.into()
        }
    }
}

impl <T: FieldName> From<(FieldsPath<T>, &Range<f32>)> for Hyperparameter<T> {
    fn from(value: (FieldsPath<T>, &Range<f32>)) -> Self {
        Hyperparameter::UniformFloat {
            name: value.0,
            lower: value.1.start.into(),
            upper: value.1.end.into()
        }
    }
}



impl <T: FieldName> TryFrom<(FieldsPath<T>, &RangeOrValue<f64>)> for Hyperparameter<T> {
    type Error = ();

    fn try_from((name, value): (FieldsPath<T>, &RangeOrValue<f64>)) -> Result<Self, Self::Error> {
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

impl <T: FieldName> TryFrom<(FieldsPath<T>, &RangeOrValue<i32>)> for Hyperparameter<T> {
    type Error = ();

    fn try_from((name, value): (FieldsPath<T>, &RangeOrValue<i32>)) -> Result<Self, Self::Error> {
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

impl <T: FieldName> TryFrom<(FieldsPath<T>, &RangeOrValue<u32>)> for Hyperparameter<T> {
    type Error = ();

    fn try_from((name, value): (FieldsPath<T>, &RangeOrValue<u32>)) -> Result<Self, Self::Error> {
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