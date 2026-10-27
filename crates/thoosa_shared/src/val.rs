use crate::hyperparameter::{FieldName, Hyperparameter};
use crate::{RangeOrValue, round_to_digits};
use num_traits::ToPrimitive;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

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

    pub fn to_f64(&self) -> f64 {
        match self {
            Val::Real(v) => *v,
            Val::Int(v) => *v as f64,
            Val::UInt(v) => *v as f64,
            Val::Str(v) => v
                .parse()
                .expect(format!("Could not parse {v} string").as_str()),
            Val::Bool(v) => {
                if *v {
                    1.0
                } else {
                    0.0
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
    pub fn from_range_or_value<V>(name: T, resolve: Arc<dyn Fn(&[NamedVal<T>]) -> RangeOrValue<V> + Send + Sync>) -> Self
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

    pub fn resolve_pending_fields(self) -> anyhow::Result<Vec<Self>> {
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
                    for branch in grid_range.grid_value_sets()?.into_iter().rev() {
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

        Ok(resolved)
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

    pub fn value_sets(&self) -> anyhow::Result<Vec<Vec<NamedVal<T>>>> {
        Ok(self.grid_value_sets()?.into_iter().map(|value_set| value_set.into_parts().0).collect())
    }

    pub fn grid_value_sets(&self) -> anyhow::Result<Vec<GridValueSet<T>>> {
        match self {
            Self::Range { .. } => self.clone().into_iter().map(|value| value.map(|v| GridValueSet::new(vec![v], vec![]))).collect(),
            Self::ValueSets(value_sets) => Ok(value_sets.clone()),
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

impl<T: FieldName> NamedRangeIter<T> {
    fn new(name: T, range: Range<Val>, step: (Val, u32)) -> anyhow::Result<Self> {
        let (current, start_kind) = val_to_f64_and_kind(&range.start);
        let (end, end_kind) = val_to_f64_and_kind(&range.end);
        let (step_value, step_kind) = val_to_f64_and_kind(&step.0);
        // Validate even directly constructed enum variants, before producing any grid values.
        anyhow::ensure!(start_kind != IterationKind::Unsupported && end_kind != IterationKind::Unsupported && step_kind != IterationKind::Unsupported, "range {name} requires numeric bounds and step");
        anyhow::ensure!(current.is_finite() && end.is_finite() && step_value.is_finite(), "range {name} requires finite bounds and step");
        anyhow::ensure!(step_value != 0.0, "range {name} step must not be zero");
        anyhow::ensure!(current == end || (end > current) == (step_value > 0.0), "range {name} step must move toward the end");
        let iteration_kind = if start_kind == IterationKind::Real || end_kind == IterationKind::Real || step_kind == IterationKind::Real { IterationKind::Real } else { start_kind };
        anyhow::ensure!(iteration_kind != IterationKind::Real || step.1 <= 308, "range {name} precision exceeds finite f64 decimal scaling");
        Ok(Self { name, current, end, step: step_value, digits: step.1, iteration_kind, done: current == end, forward: current < end })
    }
}

impl<T: FieldName> Iterator for NamedRangeIter<T> {
    type Item = anyhow::Result<NamedVal<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let out_of_bounds = if self.forward { self.current >= self.end } else { self.current <= self.end };
        if out_of_bounds {
            self.done = true;
            return None;
        }

        let current = if self.iteration_kind == IterationKind::Real { round_to_digits(self.current, self.digits) } else { self.current };
        let next = self.current + self.step;
        let next = if self.iteration_kind == IterationKind::Real { round_to_digits(next, self.digits) } else { next };
        // Recheck every step: rounding or increasing float magnitude can stop progress after valid earlier steps.
        let advances = if self.forward { next > self.current } else { next < self.current };
        if !current.is_finite() || !next.is_finite() || !advances {
            self.done = true;
            return Some(Err(anyhow::anyhow!("range {} cannot make finite progress from {} with step {} and precision {}", self.name, self.current, self.step, self.digits)));
        }
        let val = match self.iteration_kind {
            IterationKind::Int => Val::Int(current as i64),
            IterationKind::UInt => Val::UInt(current as u32),
            IterationKind::Real => Val::Real(current),
            IterationKind::Bool => Val::Bool(current >= 0.5),
            IterationKind::Unsupported => unreachable!("range types were validated at construction"),
        };
        self.current = next;
        Some(Ok(NamedVal { path: self.name.clone(), val }))
    }
}

impl<T: FieldName + 'static> IntoIterator for NamedRange<T> {
    type Item = anyhow::Result<NamedVal<T>>;
    type IntoIter = Box<dyn Iterator<Item = Self::Item> + Send>;

    fn into_iter(self) -> Self::IntoIter {
        // Keep both branches lazy so callers can take a prefix without allocating the entire range.
        match self {
            NamedRange::Range { name, range, step } => match NamedRangeIter::new(name, range, step) {
                Ok(iter) => Box::new(iter),
                Err(error) => Box::new(std::iter::once(Err(error))),
            },
            NamedRange::ValueSets(value_sets) => Box::new(value_sets.into_iter().flat_map(|value_set| value_set.into_parts().0).map(Ok)),
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

impl TryFrom<&NamedRange<String>> for Vec<NamedVal<String>> {
    type Error = anyhow::Error;

    fn try_from(value: &NamedRange<String>) -> anyhow::Result<Self> {
        // Use the same validated iteration as grid expansion, including descending and real ranges.
        value.clone().into_iter().collect()
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
    type Error = anyhow::Error;

    fn try_from(value: Val) -> anyhow::Result<f64> {
        match value {
            Val::Int(v) => Ok(v as f64), //TODO
            Val::UInt(v) => Ok(v.try_into()?),
            Val::Real(v) => Ok(v.try_into()?),
            Val::Str(v) => Ok(v.parse()?),
            _ => Err(anyhow::anyhow!("TryFrom<Val> for f64: can't convert {value:?}")),
        }
    }
}

// Reject truncation and saturation before casting. The exclusive upper bound matters:
// i64::MAX rounds up to 2^63 in f64, whereas i64::MIN is exactly representable.
fn real_to_i64(value: f64) -> anyhow::Result<i64> {
    anyhow::ensure!(value.is_finite() && value.fract() == 0.0, "can't convert {value:?} to an integer: expected a finite whole number");
    anyhow::ensure!(value >= i64::MIN as f64 && value < -(i64::MIN as f64), "can't convert {value:?} to i64: out of range");
    Ok(value as i64)
}

impl TryFrom<Val> for i64 {
    type Error = anyhow::Error;

    fn try_from(value: Val) -> anyhow::Result<i64> {
        // Ownership must not change validation of numeric configuration values.
        Self::try_from(&value)
    }
}

impl TryFrom<&Val> for i64 {
    type Error = anyhow::Error;

    fn try_from(value: &Val) -> anyhow::Result<i64> {
        match value {
            Val::Real(v) => real_to_i64(*v),
            Val::Int(v) => Ok((*v).try_into()?),
            Val::UInt(v) => Ok((*v).try_into()?),
            Val::Str(v) => Ok(v.parse()?),
            _ => Err(anyhow::anyhow!("TryFrom<&Val> for i64: can't convert {value:?}")),
        }
    }
}

impl TryFrom<Val> for u32 {
    type Error = anyhow::Error;

    fn try_from(value: Val) -> anyhow::Result<u32> {
        Self::try_from(&value)
    }
}

impl TryFrom<Val> for f32 {
    type Error = anyhow::Error;

    fn try_from(value: Val) -> anyhow::Result<f32> {
        match value {
            Val::Real(v) => Ok(v as f32), //TODO safe
            Val::Int(v) => Ok(v as f32),
            Val::UInt(v) => Ok(v as f32),
            Val::Str(v) => Ok(v.parse()?),
            _ => Err(anyhow::anyhow!("can't convert {value:?}")),
        }
    }
}

impl TryFrom<&Val> for f64 {
    type Error = anyhow::Error;

    fn try_from(value: &Val) -> anyhow::Result<f64> {
        match value {
            Val::Int(v) => (*v)
                .to_f64()
                .ok_or_else(|| anyhow::anyhow!("can't convert {value:?}")),
            Val::UInt(v) => Ok((*v).try_into()?),
            Val::Real(v) => Ok((*v).try_into()?),
            Val::Str(v) => Ok(v.parse()?),
            _ => Err(anyhow::anyhow!("can't convert {value:?}")),
        }
    }
}

impl TryFrom<&Val> for u32 {
    type Error = anyhow::Error;

    fn try_from(value: &Val) -> anyhow::Result<u32> {
        match value {
            Val::Real(v) => Ok(real_to_i64(*v)?.try_into()?),
            Val::Int(v) => (*v)
                .to_u32()
                .ok_or_else(|| anyhow::anyhow!("can't convert {value:?}")),
            Val::UInt(v) => Ok(*v),
            Val::Str(v) => Ok(v.parse()?),
            _ => Err(anyhow::anyhow!("can't convert {value:?}")),
        }
    }
}

impl TryFrom<&Val> for i32 {
    type Error = anyhow::Error;

    fn try_from(value: &Val) -> anyhow::Result<i32> {
        match value {
            Val::Real(v) => Ok(real_to_i64(*v)?.try_into()?),
            Val::Int(v) => (*v)
                .to_i32()
                .ok_or_else(|| anyhow::anyhow!("can't convert {value:?}")),
            Val::UInt(v) => Ok((*v).try_into()?),
            Val::Str(v) => Ok(v.parse()?),
            _ => Err(anyhow::anyhow!("can't convert {value:?}")),
        }
    }
}

impl TryFrom<&Val> for f32 {
    type Error = anyhow::Error;

    fn try_from(value: &Val) -> anyhow::Result<f32> {
        match value {
            Val::Real(v) => Ok(*v as f32),
            Val::UInt(v) => Ok(*v as f32),
            Val::Str(v) => Ok(v.parse()?),
            _ => Err(anyhow::anyhow!("can't convert {value:?}")),
        }
    }
}

impl TryFrom<&Val> for String {
    type Error = anyhow::Error;

    fn try_from(value: &Val) -> anyhow::Result<String> {
        match value {
            Val::Str(v) => Ok(v.to_string()), // Return a string slice with the correct lifetime
            // Non-string values must fail instead of recursively invoking this conversion.
            v => Err(anyhow::anyhow!("can't convert {v:?} to String")),
        }
    }
}

impl TryFrom<&Val> for usize {
    type Error = anyhow::Error;

    fn try_from(value: &Val) -> anyhow::Result<usize> {
        match value {
            Val::Int(v) => Ok((*v).try_into()?),
            Val::UInt(v) => Ok((*v).try_into()?),
            Val::Str(v) => Ok(v.parse()?),
            _ => Err(anyhow::anyhow!("can't convert {value:?}")),
        }
    }
}

impl TryFrom<Val> for usize {
    type Error = anyhow::Error;

    fn try_from(value: Val) -> anyhow::Result<usize> {
        Self::try_from(&value)
    }
}

impl TryFrom<&Val> for bool {
    type Error = anyhow::Error;

    fn try_from(value: &Val) -> Result<Self, Self::Error> {
        match value {
            Val::Bool(val) => Ok(*val),
            _ => Err(anyhow::anyhow!("can't convert {value:?}")),
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
    type Error = anyhow::Error;

    fn try_from(value: &NamedVal<T>) -> Result<Self, Self::Error> {
        value.val.clone().try_into()
    }
}

impl<T: FieldName> TryFrom<&NamedVal<T>> for f32 {
    type Error = anyhow::Error;

    fn try_from(value: &NamedVal<T>) -> Result<Self, Self::Error> {
        value.val.clone().try_into()
    }
}

impl<T: FieldName> TryFrom<&NamedVal<T>> for u32 {
    type Error = anyhow::Error;

    fn try_from(value: &NamedVal<T>) -> Result<Self, Self::Error> {
        value.val.clone().try_into()
    }
}

#[cfg(test)]
mod tests {
    use crate::hyperparameter::Hyperparameter;
    use crate::val::{GridValueSet, NamedRange, NamedVal, Val};

    fn assert_owned_and_borrowed<T>(value: Val, expected: Option<T>)
    where
        T: TryFrom<Val> + for<'a> TryFrom<&'a Val> + PartialEq + std::fmt::Debug,
    {
        assert_eq!(T::try_from(&value).ok(), expected, "borrowed {value:?}");
        assert_eq!(T::try_from(value.clone()).ok(), expected, "owned {value:?}");
    }

    #[test]
    fn test_integer_conversion_rejects_fractional_and_nonfinite_reals() {
        // Tiny fractions and negative fractions must fail just like visibly fractional positives.
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.5, -1.5, f64::EPSILON, -f64::EPSILON, f64::from_bits(1), -f64::from_bits(1)] {
            assert_owned_and_borrowed::<i64>(Val::Real(value), None);
            assert_owned_and_borrowed::<u32>(Val::Real(value), None);
            assert!(i32::try_from(&Val::Real(value)).is_err(), "accepted {value:?}");
        }
        for value in [-0.0, 0.0, 42.0] {
            assert_owned_and_borrowed(Val::Real(value), Some(value as i64));
            assert_owned_and_borrowed(Val::Real(value), Some(value as u32));
            assert_eq!(i32::try_from(&Val::Real(value)).unwrap(), value as i32);
        }
    }

    #[test]
    fn test_i64_conversion_preserves_representable_boundaries() {
        let lower = i64::MIN as f64;
        let upper = -(i64::MIN as f64);
        assert_owned_and_borrowed(Val::Real(lower), Some(i64::MIN));
        assert_owned_and_borrowed(Val::Real(lower.next_up()), Some(i64::MIN + 1024));
        assert_owned_and_borrowed(Val::Real(upper.next_down()), Some(i64::MAX - 1023));
        for value in [lower.next_down(), upper, upper.next_up(), f64::MAX, -f64::MAX] {
            assert_owned_and_borrowed::<i64>(Val::Real(value), None);
        }
        assert_owned_and_borrowed(Val::Int(i64::MAX), Some(i64::MAX));
        assert_owned_and_borrowed(Val::UInt(u32::MAX), Some(i64::from(u32::MAX)));
        assert_owned_and_borrowed(Val::Str(i64::MAX.to_string()), Some(i64::MAX));
    }

    #[test]
    fn test_narrow_integer_conversion_checks_bounds() -> anyhow::Result<()> {
        for value in [Val::Real(f64::from(u32::MAX)), Val::Int(i64::from(u32::MAX)), Val::UInt(u32::MAX), Val::Str(u32::MAX.to_string())] {
            assert_owned_and_borrowed(value, Some(u32::MAX));
        }
        for value in [Val::Real(-1.0), Val::Int(-1), Val::Real(f64::from(u32::MAX) + 1.0), Val::Int(i64::from(u32::MAX) + 1), Val::Str("4294967296".into())] {
            assert_owned_and_borrowed::<u32>(value, None);
        }
        for expected in [i32::MIN, 0, i32::MAX] {
            assert_eq!(i32::try_from(&Val::Real(f64::from(expected)))?, expected);
            assert_eq!(i32::try_from(&Val::Int(i64::from(expected)))?, expected);
        }
        assert_eq!(i32::try_from(&Val::UInt(i32::MAX as u32))?, i32::MAX);
        for value in [Val::UInt(i32::MAX as u32 + 1), Val::UInt(u32::MAX), Val::Int(i64::from(i32::MIN) - 1), Val::Int(i64::from(i32::MAX) + 1), Val::Real(f64::from(i32::MIN) - 1.0), Val::Real(f64::from(i32::MAX) + 1.0)] {
            assert!(i32::try_from(&value).is_err(), "accepted {value:?}");
        }
        Ok(())
    }

    #[test]
    fn test_usize_conversion_checks_sign_and_platform_width() {
        for value in [Val::Int(-1), Val::Int(i64::MIN), Val::Str("-1".into())] {
            assert_owned_and_borrowed::<usize>(value, None);
        }
        assert_owned_and_borrowed(Val::Int(0), Some(0usize));
        assert_owned_and_borrowed(Val::UInt(42), Some(42usize));
        assert_owned_and_borrowed(Val::Str(usize::MAX.to_string()), Some(usize::MAX));
        assert_owned_and_borrowed::<usize>(Val::Str((usize::MAX as u128 + 1).to_string()), None);
        // The same signed input fits a 64-bit pointer but must be rejected on narrower targets.
        if usize::BITS >= i64::BITS {
            assert_owned_and_borrowed(Val::Int(i64::MAX), Some(i64::MAX as usize));
        } else {
            assert_owned_and_borrowed(Val::Int(usize::MAX as i64), Some(usize::MAX));
            assert_owned_and_borrowed::<usize>(Val::Int(usize::MAX as i64 + 1), None);
            assert_owned_and_borrowed::<usize>(Val::Int(i64::MAX), None);
        }
    }

    #[test]
    fn test_string_conversion_rejects_non_strings() -> anyhow::Result<()> {
        assert_eq!(String::try_from(&Val::Str("value".into()))?, "value");
        for value in [Val::Int(1), Val::UInt(1), Val::Real(1.0), Val::Bool(true)] {
            assert!(String::try_from(&value).is_err(), "unexpected String conversion for {value:?}");
        }
        Ok(())
    }

    #[test]
    fn test_named_range_rejects_invalid_parameters() {
        // These previously looped forever, or overflowed decimal scaling before making progress.
        for (start, end, step, digits) in [
            (0.0, 1.0, 0.0, 0),
            (0.0, 1.0, -1.0, 0),
            (1.0, 0.0, 1.0, 0),
            (f64::NAN, 1.0, 1.0, 0),
            (0.0, f64::NAN, 1.0, 0),
            (0.0, 1.0, f64::NAN, 0),
            (0.0, f64::INFINITY, 1.0, 0),
            (f64::NEG_INFINITY, 0.0, 1.0, 0),
            (0.0, 1.0, f64::INFINITY, 0),
            (0.0, 1.0, 0.01, 1),
            (0.0, 1.0, 0.1, u32::MAX),
            (10.0, 20.0, 1.0, 308),
        ] {
            let mut iter = NamedRange::new("invalid".to_string(), Val::Real(start)..Val::Real(end), (Val::Real(step), digits)).into_iter();
            assert!(iter.next().unwrap().is_err(), "accepted start={start}, end={end}, step={step}, digits={digits}");
            assert!(iter.next().is_none());
        }
        let invalid = NamedRange::new("string".to_string(), Val::Str("0".into())..Val::Int(2), (Val::Int(1), 0));
        assert!(invalid.grid_value_sets().is_err());
    }

    #[test]
    fn test_named_range_detects_later_precision_stall() {
        let start = 9_007_199_254_740_991.0;
        let mut iter = NamedRange::new("precision".to_string(), Val::Real(start)..Val::Real(start + 5.0), (Val::Real(1.0), 0)).into_iter();
        assert!(iter.next().unwrap().is_ok());
        assert!(iter.next().unwrap().is_err());
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_named_range_descending_empty_and_borrowed_conversion() -> anyhow::Result<()> {
        let range = NamedRange::new("descending".to_string(), Val::Int(3)..Val::Int(-2), (Val::Int(-2), 0));
        let expected = vec![NamedVal::new("descending".to_string(), Val::Int(3)), NamedVal::new("descending".to_string(), Val::Int(1)), NamedVal::new("descending".to_string(), Val::Int(-1))];
        assert_eq!(Vec::<NamedVal<String>>::try_from(&range)?, expected);
        assert_eq!(range.into_iter().collect::<anyhow::Result<Vec<_>>>()?, expected);
        let empty = NamedRange::new("empty".to_string(), Val::Int(2)..Val::Int(2), (Val::Int(1), 0));
        assert!(empty.into_iter().next().is_none());
        Ok(())
    }

    #[test]
    fn test_named_range_large_prefix_is_lazy() -> anyhow::Result<()> {
        // Collecting the entire range here would allocate billions of NamedVals before take() runs.
        let range = NamedRange::new("large".to_string(), Val::UInt(0)..Val::UInt(u32::MAX), (Val::UInt(1), 0));
        let prefix = range.into_iter().take(3).collect::<anyhow::Result<Vec<_>>>()?;
        assert_eq!(prefix, (0..3).map(|i| NamedVal::new("large".to_string(), Val::UInt(i))).collect::<Vec<_>>());
        let sets = NamedRange::from_value_sets(vec![prefix[..2].to_vec(), prefix[2..].to_vec()]);
        assert_eq!(sets.into_iter().collect::<anyhow::Result<Vec<_>>>()?, prefix);
        Ok(())
    }

    #[test]
    fn test_dynamic_range_error_reaches_grid_expansion() {
        use crate::val::PendingOptField;
        use std::sync::Arc;
        let pending = PendingOptField::from_range_or_value("dynamic".to_string(), Arc::new(|_| crate::RangeOrValue::Range { range: 0u32..3, step: Some((0, 0)) }));
        let grid = crate::run_configs::GridRunConfigs::new(crate::run_configs::ConstConfigs::new(vec![]), vec![NamedRange::from_grid_value_sets(vec![GridValueSet::with_pending_fields(vec![], vec![], vec![pending])])]);
        assert!(grid.grid_to_vals().is_err());
    }

    #[test]
    fn test_named_range_int_iteration() {
        let named_range = NamedRange::new("SomeName".into(), Val::Int(-1)..Val::Int(3), (Val::Int(1), 0));

        let result: Vec<NamedVal<String>> = named_range.into_iter().collect::<anyhow::Result<_>>().unwrap();

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

        let result: Vec<NamedVal<String>> = named_range.into_iter().collect::<anyhow::Result<_>>().unwrap();

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
            .grid_value_sets().unwrap(),
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

pub fn val_to_f32(val: &Val) -> Option<f32> {
    match val {
        Val::Real(v) => Some(*v as f32),
        Val::Int(v) => Some(*v as f32),
        Val::UInt(v) => Some(*v as f32),
        Val::Str(v) => v.parse().ok(),
        Val::Bool(_) => None,
    }
}
