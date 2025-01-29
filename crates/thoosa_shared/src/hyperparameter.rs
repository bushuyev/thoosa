use std::ops::Range;
use rand::distributions::Uniform;
use rand::Rng;
use log::trace;
use crate::opt_rng::opt_rnd;
use crate::RangeOrValue;
use crate::val::Val;


#[derive(Debug, Clone, PartialEq)]
pub enum Hyperparameter {
    UniformFloat{name:String, lower:f64, upper:f64},
    UniformInt{name:String, lower:i64, upper:i64},
    Categorical{name:String, values:Vec<String>}
}

impl Hyperparameter {

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

    pub fn name(&self) -> &str {
        match self {
            Hyperparameter::UniformFloat { name, .. } => name,
            Hyperparameter::UniformInt { name, .. } => name,
            Hyperparameter::Categorical { name, .. } => name,
        }
    }

}

impl From<(&str, Range<f64>)> for Hyperparameter {
    fn from(value: (&str, Range<f64>)) -> Self {
        Hyperparameter::UniformFloat {
            name: value.0.into(),
            lower: value.1.start.into(),
            upper: value.1.end.into()
        }
    }
}

impl From<(&str, Range<u32>)> for Hyperparameter {
    fn from(value: (&str, Range<u32>)) -> Self {
        Hyperparameter::UniformInt {
            name: value.0.into(),
            lower: value.1.start.into(),
            upper: value.1.end.into()
        }
    }
}


impl TryFrom<(&str, &RangeOrValue<f64>)> for Hyperparameter {
    type Error = ();

    fn try_from((name, value): (&str, &RangeOrValue<f64>)) -> Result<Self, Self::Error> {
        if let RangeOrValue::Range{range,step} = value {
            if step.is_none() {
                return Ok(Hyperparameter::UniformFloat {
                    name: name.into(),
                    lower: range.start,
                    upper: range.end
                });
            }
        }
        Err(())
    }
}

impl TryFrom<(&str, &RangeOrValue<i32>)> for Hyperparameter {
    type Error = ();

    fn try_from((name, value): (&str, &RangeOrValue<i32>)) -> Result<Self, Self::Error> {
        if let RangeOrValue::Range{range, step} = value {
            if step.is_none() {
                return Ok(Hyperparameter::UniformInt {
                    name: name.into(),
                    lower: range.start as i64, //TODO as i64
                    upper: range.end as i64
                });
            }
        }
        Err(())
    }
}

impl TryFrom<(&str, &RangeOrValue<u32>)> for Hyperparameter {
    type Error = ();

    fn try_from((name, value): (&str, &RangeOrValue<u32>)) -> Result<Self, Self::Error> {
        if let RangeOrValue::Range{range, step} = value {
            if step.is_none() {
                return Ok(Hyperparameter::UniformInt {
                    name: name.into(),
                    lower: range.start as i64, //TODO as i64
                    upper: range.end as i64
                });
            }
        }
        Err(())
    }
}