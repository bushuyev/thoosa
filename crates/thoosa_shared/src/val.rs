use crate::hyperparameter::{FieldName, Hyperparameter};
use crate::{RangeOrValue, ThoError, round_to_digits};
use num_traits::ToPrimitive;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;
use log::warn;

#[derive(Debug, PartialEq, Clone)]
pub enum Val {
    Real(f64),
    Int(i64),
    UInt(u32),
    Str(String),
    Bool(bool),
}

impl Val {
    pub fn to_i64(&self, p: u32) -> i64 {
        match self {
            Val::Real(v) => (v * 10_f64.powi(p as i32)) as i64,
            Val::Int(v) => *v,
            Val::UInt(v) => *v as i64,
            Val::Str(v) => v
                .parse()
                .expect(format!("Could not parse {v} string").as_str()),
            Val::Bool(v) => {
                if *v {
                    1
                } else {
                    0
                }
            }

        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct NamedVal<T: FieldName> {
    path: T,
    val: Val,
}

impl<T: FieldName> NamedVal<T> {
    pub fn new<P: Into<T>, V: Into<Val>>(into_path: P, val: V) -> Self {
        Self {
            path: into_path.into(),
            val: val.into(),
        }
    }

    pub fn val(&self) -> &Val {
        &self.val
    }

    pub fn path(&self) -> &T {
        &self.path
    }
}

#[derive(Debug, Clone)]
pub enum ResolvedOptField<T: FieldName> {
    Const(NamedVal<T>),
    Grid(NamedRange<T>),
    Opt(Hyperparameter<T>),
}

impl<T: FieldName + 'static> ResolvedOptField<T> {
    pub fn from_range_or_value<'a, V>(name: T, value: &'a RangeOrValue<V>) -> Self
    where
        Val: From<&'a V>,
        Hyperparameter<T>: From<(T, &'a Range<V>)>,
    {
        match value {
            RangeOrValue::Range { range, step } => match step {
                None => Self::Opt((name, range).into()),
                Some((step, mul)) => Self::Grid(NamedRange::new(
                    name,
                    (&range.start).into()..(&range.end).into(),
                    (step.into(), *mul),
                )),
            },
            RangeOrValue::Value(v) => Self::Const(NamedVal::from((name, v.into()))),
            RangeOrValue::Dynamic(_) => panic!("dynamic RangeOrValue must resolve to a static value"),
        }
    }
}

#[derive(Clone)]
pub enum PendingOptField<T: FieldName> {
    Dynamic(Arc<dyn Fn(&[NamedVal<T>]) -> ResolvedOptField<T> + Send + Sync>),
}

impl<T: FieldName + 'static> PendingOptField<T> {
    pub fn from_range_or_value<V>(name: T, resolve: fn(&[NamedVal<T>]) -> RangeOrValue<V>) -> Self
    where
        for<'a> Val: From<&'a V>,
        for<'a> Hyperparameter<T>: From<(T, &'a Range<V>)>,
        V: 'static,
    {
        Self::Dynamic(Arc::new(move |named_vals| {
            let resolved = resolve(named_vals);
            ResolvedOptField::from_range_or_value(name.clone(), &resolved)
        }))
    }

    pub fn resolve(&self, named_vals: &[NamedVal<T>]) -> ResolvedOptField<T> {
        match self {
            Self::Dynamic(resolve) => resolve(named_vals),
        }
    }
}

impl<T: FieldName> Debug for PendingOptField<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dynamic(_) => f.write_str("Dynamic"),
        }
    }
}

impl<T: FieldName> PartialEq for PendingOptField<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Dynamic(lhs), Self::Dynamic(rhs)) => Arc::ptr_eq(lhs, rhs),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
// to run opt on search space from common hyperparams + specific to each grid values combo (filters with filter-specific params in find motifs)
// gpt: Represents one concrete grid-search branch: a bundle of fixed values to apply
// together plus opt hyperparameters that are only valid for that branch.
pub struct GridValueSet<T: FieldName> {
    values: Vec<NamedVal<T>>,
    opt_params: Vec<Hyperparameter<T>>,
    pending_fields: Vec<PendingOptField<T>>,
}

impl<T: FieldName + 'static> GridValueSet<T> {
    pub fn new(values: Vec<NamedVal<T>>, opt_params: Vec<Hyperparameter<T>>) -> Self {
        Self {
            values,
            opt_params,
            pending_fields: vec![],
        }
    }

    pub fn with_pending_fields(values: Vec<NamedVal<T>>, opt_params: Vec<Hyperparameter<T>>, pending_fields: Vec<PendingOptField<T>>) -> Self {
        Self {
            values,
            opt_params,
            pending_fields,
        }
    }

    pub fn values(&self) -> &Vec<NamedVal<T>> {
        &self.values
    }

    pub fn opt_params(&self) -> &Vec<Hyperparameter<T>> {
        &self.opt_params
    }

    pub fn pending_fields(&self) -> &Vec<PendingOptField<T>> {
        &self.pending_fields
    }

    pub fn into_parts(self) -> (Vec<NamedVal<T>>, Vec<Hyperparameter<T>>) {
        assert!(self.pending_fields.is_empty(), "GridValueSet::into_parts called with unresolved pending fields");
        (self.values, self.opt_params)
    }

    pub fn into_parts_with_pending(self) -> (Vec<NamedVal<T>>, Vec<Hyperparameter<T>>, Vec<PendingOptField<T>>) {
        (self.values, self.opt_params, self.pending_fields)
    }

    pub fn resolve_pending_fields(self) -> Vec<Self> {
        let mut pending = vec![self];
        let mut resolved = Vec::new();

        while let Some(mut value_set) = pending.pop() {
            if value_set.pending_fields.is_empty() {
                resolved.push(value_set);
                continue;
            }

            let pending_field = value_set.pending_fields.remove(0);
            match pending_field.resolve(value_set.values.as_slice()) {
                ResolvedOptField::Const(value) => {
                    value_set.values.push(value);
                    pending.push(value_set);
                }
                ResolvedOptField::Opt(opt_param) => {
                    value_set.opt_params.push(opt_param);
                    pending.push(value_set);
                }
                ResolvedOptField::Grid(grid_range) => {
                    for branch in grid_range.grid_value_sets().into_iter().rev() {
                        let (branch_values, branch_opt_params, branch_pending_fields) = branch.into_parts_with_pending();
                        let mut branched_value_set = value_set.clone();
                        branched_value_set.values.extend(branch_values);
                        branched_value_set.opt_params.extend(branch_opt_params);
                        branched_value_set.pending_fields.extend(branch_pending_fields);
                        pending.push(branched_value_set);
                    }
                }
            }
        }

        resolved
    }
}

#[derive(Clone, Debug)]
pub enum NamedRange<T: FieldName> {
    Range {
        name: T,
        range: Range<Val>,
        step: (Val, u32), //for NamedRangeIter step/digits -> round_to_digits(value: f64, digits: u32)
    },
    // Stores pre-expanded grid branches when a single "range" choice needs to
    // contribute multiple fixed values and/or branch-specific opt params.
    ValueSets(Vec<GridValueSet<T>>),
}

impl<T: FieldName + 'static> NamedRange<T> {
    pub fn new(name: T, range: Range<Val>, step: (Val, u32)) -> Self {
        Self::Range { name, range, step }
    }

    pub fn from_value_sets(value_sets: Vec<Vec<NamedVal<T>>>) -> Self {
        Self::ValueSets(
            value_sets
                .into_iter()
                .map(|values| GridValueSet::new(values, vec![]))
                .collect(),
        )
    }

    pub fn from_grid_value_sets(value_sets: Vec<GridValueSet<T>>) -> Self {
        Self::ValueSets(value_sets)
    }

    pub fn range(&self) -> &Range<Val> {
        match self {
            Self::Range { range, .. } => range,
            Self::ValueSets(_) => panic!("range() is not available for NamedRange::ValueSets"),
        }
    }

    pub fn step(&self) -> &(Val, u32) {
        match self {
            Self::Range { step, .. } => step,
            Self::ValueSets(_) => panic!("step() is not available for NamedRange::ValueSets"),
        }
    }

    pub fn value_sets(&self) -> Vec<Vec<NamedVal<T>>> {
        self.grid_value_sets()
            .into_iter()
            .map(|value_set| value_set.into_parts().0)
            .collect()
    }

    pub fn grid_value_sets(&self) -> Vec<GridValueSet<T>> {
        match self {
            Self::Range { .. } => self
                .clone()
                .into_iter()
                .map(|v| GridValueSet::new(vec![v], vec![]))
                .collect(),
            Self::ValueSets(value_sets) => value_sets.clone(),
        }
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

pub struct NamedRangeIter<T: FieldName> {
    name: T,
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

impl<T: FieldName> Iterator for NamedRangeIter<T> {
    type Item = NamedVal<T>;

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
            IterationKind::Bool => Val::Bool(self.current >= 0.5),
            IterationKind::Unsupported => {
                return None;
            }
        };

        let item = NamedVal {
            path: self.name.clone(),
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

impl<T: FieldName + 'static> IntoIterator for NamedRange<T> {
    type Item = NamedVal<T>;
    type IntoIter = std::vec::IntoIter<NamedVal<T>>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            NamedRange::Range { name, range, step } => {
                let start_val = &range.start;
                let end_val = &range.end;
                let (step_val, digits) = &step;

                let (start_f, kind_start) = val_to_f64_and_kind(start_val);
                let (end_f, kind_end) = val_to_f64_and_kind(end_val);
                let (step_f, _) = val_to_f64_and_kind(step_val);

                let iteration_kind = if kind_start == IterationKind::Unsupported {
                    IterationKind::Unsupported
                } else if kind_end == IterationKind::Unsupported {
                    IterationKind::Unsupported
                } else {
                    match kind_start {
                        IterationKind::Real => IterationKind::Real,
                        _ => {
                            if kind_end == IterationKind::Real || matches!(step_val, Val::Real(_)) {
                                IterationKind::Real
                            } else {
                                kind_start
                            }
                        }
                    }
                };

                NamedRangeIter {
                    name,
                    current: start_f,
                    end: end_f,
                    step: step_f,
                    digits: *digits,
                    done: false,
                    forward: start_f < end_f,
                    iteration_kind,
                }
                .collect::<Vec<_>>()
                .into_iter()
            }
            NamedRange::ValueSets(value_sets) => value_sets
                .into_iter()
                .flat_map(|value_set| value_set.into_parts().0)
                .collect::<Vec<_>>()
                .into_iter(),
        }
    }
}

fn val_to_f64_and_kind(v: &Val) -> (f64, IterationKind) {
    match v {
        Val::Int(i) => (*i as f64, IterationKind::Int),
        Val::UInt(u) => (*u as f64, IterationKind::UInt),
        Val::Real(r) => (*r, IterationKind::Real),
        Val::Bool(b) => (if *b { 1.0 } else { 0.0 }, IterationKind::Bool),
        Val::Str(_) => (0.0, IterationKind::Unsupported),
    }
}

impl<T: FieldName> From<(T, Val)> for NamedVal<T> {
    fn from((name, val): (T, Val)) -> Self {
        NamedVal { path: name, val }
    }
}

impl<T: FieldName + 'static, P: FieldName> TryFrom<(T, &RangeOrValue<f64, P>)> for NamedRange<T> {
    type Error = ();

    fn try_from((name, value): (T, &RangeOrValue<f64, P>)) -> Result<Self, Self::Error> {
        if let RangeOrValue::Range { range, step } = value {
            if let Some(step) = step {
                return Ok(NamedRange::new(name, Val::Real(range.start)..Val::Real(range.end), (Val::Real(step.0), step.1)));
            }
        }
        Err(())
    }
}

impl From<&NamedRange<String>> for Vec<NamedVal<String>> {
    fn from(value: &NamedRange<String>) -> Self {
        match value {
            NamedRange::Range { name, range, step } => {
                let start: i64 = range.start.to_i64(step.1);
                let end: i64 = range.end.to_i64(step.1);
                let step: usize = step.0.to_i64(step.1) as usize;

                (start..end)
                    .step_by(step)
                    .map(|i| (name.clone(), <Val as Into<Val>>::into(i.into())).into())
                    .collect::<Vec<NamedVal<String>>>()
            }
            NamedRange::ValueSets(value_sets) => value_sets.iter().flat_map(|value_set| value_set.values().clone()).collect(),
        }
    }
}

impl<T: FieldName + 'static, P: FieldName> TryFrom<(T, &RangeOrValue<u32, P>)> for NamedRange<T> {
    type Error = ();

    fn try_from((name, value): (T, &RangeOrValue<u32, P>)) -> Result<Self, Self::Error> {
        if let RangeOrValue::Range { range, step } = value {
            if let Some(step) = step {
                return Ok(NamedRange::new(name, Val::UInt(range.start)..Val::UInt(range.end), (Val::UInt(step.0), step.1)));
            }
        }
        Err(())
    }
}

impl<T: FieldName, P: FieldName> TryFrom<(T, &RangeOrValue<u32, P>)> for NamedVal<T> {
    type Error = ();

    fn try_from((name, value): (T, &RangeOrValue<u32, P>)) -> Result<Self, Self::Error> {
        if let RangeOrValue::Value(v) = value {
            return Ok(NamedVal {
                path: name,
                val: v.into(),
            });
        }
        Err(())
    }
}

impl<T: FieldName> TryFrom<&Hyperparameter<T>> for NamedVal<T> {
    type Error = ();

    fn try_from(value: &Hyperparameter<T>) -> Result<Self, Self::Error> {
        Ok(NamedVal::new(value.name().clone(), value.random_sample()))
    }
}

impl TryFrom<Val> for f64 {
    type Error = ThoError;

    fn try_from(value: Val) -> Result<f64, ThoError> {
        match value {
            Val::Int(v) => Ok(v as f64), //TODO
            Val::UInt(v) => v
                .try_into()
                .map_err(ThoError::from),
            Val::Real(v) => v
                .try_into()
                .map_err(ThoError::from),
            Val::Str(v) => v.parse().map_err(ThoError::from),
            _ => Err(ThoError::new(format!("TryFrom<Val> for f64: can't convert {value:?}"))),
        }
    }
}

impl TryFrom<Val> for i64 {
    type Error = ThoError;

    fn try_from(value: Val) -> Result<i64, ThoError> {
        match value {
            Val::Real(v) =>  {
                if  v - v.trunc() > f64::EPSILON {
                    warn!("Truncating {v:?} to i64");
                }
                Ok(v as i64)
            },
            Val::Int(v) => v
                .try_into()
                .map_err(ThoError::from),
            Val::UInt(v) => v
                .try_into()
                .map_err(ThoError::from),
            Val::Str(v) => v.parse().map_err(ThoError::from),
            _ => Err(ThoError::new(format!("TryFrom<Val> for i64: can't convert {value:?}"))),
        }
    }
}

impl TryFrom<&Val> for i64 {
    type Error = ThoError;

    fn try_from(value: &Val) -> Result<i64, ThoError> {
        match value {
            Val::Real(v) =>  {
                if  *v - v.trunc() > f64::EPSILON {
                    warn!("Truncating {v:?} to i64");
                }
                Ok(*v as i64)
            },
            Val::Int(v) => (*v)
                .try_into()
                .map_err(ThoError::from),
            Val::UInt(v) => (*v)
                .try_into()
                .map_err(ThoError::from),
            Val::Str(v) => v.parse().map_err(ThoError::from),
            _ => Err(ThoError::new(format!("TryFrom<&Val> for i64: can't convert {value:?}"))),
        }
    }
}

impl TryFrom<Val> for u32 {
    type Error = ThoError;

    fn try_from(value: Val) -> Result<u32, ThoError> {
        match value {
            Val::Real(v) => Ok(v as u32), //v.to_u32().ok_or_else(|| ThoosaError::new(format!("can't convert {value:?}"))),
            Val::Int(v) => v
                .try_into()
                .map_err(ThoError::from),
            Val::UInt(v) => v
                .try_into()
                .map_err(ThoError::from),
            Val::Str(v) => v.parse().map_err(ThoError::from),
            _ => Err(ThoError::new(format!("TryFrom<Val> for u32: can't convert {value:?}"))),
        }
    }
}

impl TryFrom<Val> for f32 {
    type Error = ThoError;

    fn try_from(value: Val) -> Result<f32, ThoError> {
        match value {
            Val::Real(v) => Ok(v as f32), //TODO safe
            Val::Int(v) => Ok(v as f32),
            Val::UInt(v) => Ok(v as f32),
            Val::Str(v) => v.parse().map_err(ThoError::from),
            _ => Err(ThoError::new(format!("can't convert {value:?}"))),
        }
    }
}

impl TryFrom<&Val> for f64 {
    type Error = ThoError;

    fn try_from(value: &Val) -> Result<f64, ThoError> {
        match value {
            Val::Int(v) => (*v)
                .to_f64()
                .ok_or_else(|| ThoError::new(format!("can't convert {value:?}"))),
            Val::UInt(v) => (*v)
                .try_into()
                .map_err(ThoError::from),
            Val::Real(v) => (*v)
                .try_into()
                .map_err(ThoError::from),
            Val::Str(v) => v.parse().map_err(ThoError::from),
            _ => Err(ThoError::new(format!("can't convert {value:?}"))),
        }
    }
}

impl TryFrom<&Val> for u32 {
    type Error = ThoError;

    fn try_from(value: &Val) -> Result<u32, ThoError> {
        match value {
            Val::Real(v) => Ok(*v as u32), //TODO v.to_u32().ok_or_else(|| ThoosaError::new(format!("can't convert {value:?}"))),
            Val::Int(v) => (*v)
                .to_u32()
                .ok_or_else(|| ThoError::new(format!("can't convert {value:?}"))),
            Val::UInt(v) => Ok(*v),
            Val::Str(v) => v.parse().map_err(ThoError::from),
            _ => Err(ThoError::new(format!("can't convert {value:?}"))),
        }
    }
}

impl TryFrom<&Val> for i32 {
    type Error = ThoError;

    fn try_from(value: &Val) -> Result<i32, ThoError> {
        match value {
            Val::Real(v) => Ok(*v as i32), //TODO v.to_u32().ok_or_else(|| ThoosaError::new(format!("can't convert {value:?}"))),
            Val::Int(v) => (*v)
                .to_i32()
                .ok_or_else(|| ThoError::new(format!("can't convert {value:?}"))),
            Val::UInt(v) => Ok(*v as i32),
            Val::Str(v) => v.parse().map_err(ThoError::from),
            _ => Err(ThoError::new(format!("can't convert {value:?}"))),
        }
    }
}

impl TryFrom<&Val> for f32 {
    type Error = ThoError;

    fn try_from(value: &Val) -> Result<f32, ThoError> {
        match value {
            Val::Real(v) => Ok(*v as f32),
            Val::UInt(v) => Ok(*v as f32),
            Val::Str(v) => v.parse().map_err(ThoError::from),
            _ => Err(ThoError::new(format!("can't convert {value:?}"))),
        }
    }
}

impl TryFrom<&Val> for String {
    type Error = ThoError;

    fn try_from(value: &Val) -> Result<String, ThoError> {
        match value {
            Val::Str(v) => Ok(v.to_string()), // Return a string slice with the correct lifetime
            v => v
                .try_into()
                .map_err(ThoError::from),
        }
    }
}

impl TryFrom<Val> for usize {
    type Error = ThoError;

    fn try_from(value: Val) -> Result<usize, ThoError> {
        match value {
            Val::Int(v) => Ok(v as usize),
            Val::UInt(v) => Ok(v as usize),
            Val::Str(v) => v.parse().map_err(ThoError::from),
            _ => Err(ThoError::new(format!("can't convert {value:?}"))),
        }
    }
}

impl TryFrom<&Val> for bool {
    type Error = ThoError;

    fn try_from(value: &Val) -> Result<Self, Self::Error> {
        match value {
            Val::Bool(val) => Ok(*val),
            _ => Err(ThoError::new(format!("can't convert {value:?}"))),
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
        Val::UInt(*v)
    }
}

impl From<&bool> for Val {
    fn from(v: &bool) -> Self {
        Val::Bool(*v)
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

impl<T: FieldName> TryFrom<&NamedVal<T>> for f64 {
    type Error = ThoError;

    fn try_from(value: &NamedVal<T>) -> Result<Self, Self::Error> {
        value.val.clone().try_into()
    }
}

impl<T: FieldName> TryFrom<&NamedVal<T>> for f32 {
    type Error = ThoError;

    fn try_from(value: &NamedVal<T>) -> Result<Self, Self::Error> {
        value.val.clone().try_into()
    }
}

impl<T: FieldName> TryFrom<&NamedVal<T>> for u32 {
    type Error = ThoError;

    fn try_from(value: &NamedVal<T>) -> Result<Self, Self::Error> {
        value.val.clone().try_into()
    }
}

#[cfg(test)]
mod tests {

    use crate::hyperparameter::Hyperparameter;
    use crate::val::{GridValueSet, NamedRange, NamedVal, Val};

    #[test]
    fn test_named_range_int_iteration() {
        let named_range = NamedRange::new("SomeName".into(), Val::Int(-1)..Val::Int(3), (Val::Int(1), 0));

        let result: Vec<NamedVal<String>> = named_range.into_iter().collect();

        let expected = vec![
            NamedVal {
                path: "SomeName".into(),
                val: Val::Int(-1),
            },
            NamedVal {
                path: "SomeName".into(),
                val: Val::Int(0),
            },
            NamedVal {
                path: "SomeName".into(),
                val: Val::Int(1),
            },
            NamedVal {
                path: "SomeName".into(),
                val: Val::Int(2),
            },
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_named_range_float_iteration() {
        let named_range = NamedRange::new("FloatRange".into(), Val::Real(0.0)..Val::Real(1.0), (Val::Real(0.3), 1));

        let result: Vec<NamedVal<String>> = named_range.into_iter().collect();

        let expected = vec![
            NamedVal {
                path: "FloatRange".into(),
                val: Val::Real(0.0),
            },
            NamedVal {
                path: "FloatRange".into(),
                val: Val::Real(0.3),
            },
            NamedVal {
                path: "FloatRange".into(),
                val: Val::Real(0.6),
            },
            NamedVal {
                path: "FloatRange".into(),
                val: Val::Real(0.9),
            },
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_named_range_value_sets() {
        assert_eq!(
            NamedRange::<String>::from_grid_value_sets(vec![
                GridValueSet::new(vec![NamedVal::new("ChoiceA".to_string(), Val::Int(1))], vec![]),
                GridValueSet::new(
                    vec![
                        NamedVal::new("ChoiceB".to_string(), Val::Int(2)),
                        NamedVal::new("ChoiceBExtra".to_string(), Val::Bool(true)),
                    ],
                    vec![Hyperparameter::UniformInt {
                        name: "ChoiceBOpt".to_string(),
                        lower: 2,
                        upper: 5,
                    }],
                ),
            ])
            .grid_value_sets(),
            vec![
                GridValueSet::new(vec![NamedVal::new("ChoiceA".to_string(), Val::Int(1))], vec![]),
                GridValueSet::new(
                    vec![
                        NamedVal::new("ChoiceB".to_string(), Val::Int(2)),
                        NamedVal::new("ChoiceBExtra".to_string(), Val::Bool(true)),
                    ],
                    vec![Hyperparameter::UniformInt {
                        name: "ChoiceBOpt".to_string(),
                        lower: 2,
                        upper: 5,
                    }],
                ),
            ],
        );
    }
}
