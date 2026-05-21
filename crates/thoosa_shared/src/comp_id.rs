#[cfg(any(debug_assertions, tho_debug))]
use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::Add;
use thoosa_named_derive::Named;
use thoosa_shared::traits::DataId;

#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum ElvVal {
    Str(String),
    Num(usize),
    Bool(bool),
}

impl Display for ElvVal {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            //TODO use same ?
            ElvVal::Str(v) => {
                write!(f, "{}", v)
            }
            ElvVal::Num(v) => {
                write!(f, "{}", v)
            }
            ElvVal::Bool(v) => {
                write!(f, "{}", v)
            }
        }
    }
}
impl From<u32> for ElvVal {
    fn from(value: u32) -> Self {
        ElvVal::Num(value as usize)
    }
}

impl From<usize> for ElvVal {
    fn from(value: usize) -> Self {
        ElvVal::Num(value)
    }
}

impl From<i32> for ElvVal {
    fn from(value: i32) -> Self {
        ElvVal::Num(value as usize)
    }
}


impl From<String> for ElvVal {
    fn from(value: String) -> Self {
        ElvVal::Str(value)
    }
}

impl From<&str> for ElvVal {
    fn from(value: &str) -> Self {
        ElvVal::Str(value.to_string())
    }
}

impl From<bool> for ElvVal {
    fn from(v: bool) -> Self {
        ElvVal::Bool(v)
    }
}

#[derive(Debug, Clone, Ord, Named)]
pub struct MonoKey {
    val: ElvVal,

    #[cfg(any(debug_assertions, tho_debug))]
    name: Cow<'static, str>,
}

impl Display for MonoKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        #[cfg(any(debug_assertions, tho_debug))]
        write!(f, "{}_{}", self.val, self.name)?;

        #[cfg(not(any(debug_assertions, tho_debug)))]
        write!(f, "{}", self.val)?;

        Ok(())
    }
}

impl From<&str> for MonoKey {
    fn from(value: &str) -> Self {
        MonoKey {
            val: value.into(),
            #[cfg(any(debug_assertions, tho_debug))]
            name: Cow::Owned(value.to_owned()),
        }
    }
}

impl PartialOrd for MonoKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

impl MonoKey {
    pub fn new(
        val: ElvVal,
        #[cfg(any(debug_assertions, tho_debug))] name: Cow<'static, str>,
    ) -> Self {
        Self {
            val,
            #[cfg(any(debug_assertions, tho_debug))]
            name,
        }
    }

    #[cfg(any(debug_assertions, tho_debug))]
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    //TODO add a safer way to get to data_id
    pub fn to_data_id(&self) -> DataId{
        match &self.val {

            ElvVal::Num(val) => {
                DataId(*val as u32)//TODO as
            }
            val => {
                panic!("can't convert {val} to data_id")
            }
        }
    }
}

#[macro_export]
macro_rules! mono_key {
    ($val:expr, $name:literal $(,)?) => {{
        #[cfg(any(debug_assertions, tho_debug))]
        {
            thoosa_shared::comp_id::MonoKey::new(
                ($val).into(),
                ::std::borrow::Cow::Borrowed($name),
            )
        }
        #[cfg(not(any(debug_assertions, tho_debug)))]
        {
            thoosa_shared::comp_id::MonoKey::new(($val).into())
        }
    }};

    ($val:expr, $name:expr $(,)?) => {{
        #[cfg(any(debug_assertions, tho_debug))]
        {
            thoosa_shared::comp_id::MonoKey::new(
                ($val).into(),
                ::std::borrow::Cow::Owned(::std::string::String::from($name)),
            )
        }
        #[cfg(not(any(debug_assertions, tho_debug)))]
        {
            thoosa_shared::comp_id::MonoKey::new(($val).into())
        }
    }};
}

impl Eq for MonoKey {}
impl PartialEq for MonoKey {
    fn eq(&self, other: &Self) -> bool {
        self.val.eq(&other.val)
    }
}

impl Hash for MonoKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.val.hash(state)
    }
}


/// ✅ Dynamic-length key, stored as a boxed slice (one allocation, no capacity slack).
#[derive(Eq, PartialEq, Hash, Clone, Ord, PartialOrd, Named)]
pub struct SynKey {
    pub vals: Box<[MonoKey]>,
}

impl Display for SynKey{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for val in &self.vals {
            write!(f, "_{}_", val)?;
        }

        Ok(())
    }
}

impl fmt::Debug for SynKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SynKey[")?;
        for (idx, mono) in self.vals.iter().enumerate() {
            if idx > 0 {
                f.write_str("/")?;
            }
            fmt_mono_key(mono, f)?;
        }
        f.write_str("]")
    }
}

fn fmt_elv_val(val: &ElvVal, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match val {
        ElvVal::Num(value) => write!(f, "{value}"),
        ElvVal::Str(value) => write!(f, "{value}"),
        ElvVal::Bool(value) => write!(f, "{value}")
    }
}

#[cfg(any(debug_assertions, tho_debug))]
fn fmt_mono_key(mono: &MonoKey, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("(")?;
    if let ElvVal::Str(v) = &mono.val {
        if v.eq(mono.name.as_ref()) {
            fmt_elv_val(&mono.val, f)?;
        } else {
            fmt_elv_name_val(&mono, f)?;
        }
    } else {
        fmt_elv_name_val(&mono, f)?;
    }

    f.write_str(")")
}

#[allow(dead_code)]
fn fmt_elv_name_val(mono: &&MonoKey, f: &mut Formatter) -> fmt::Result {
    fmt_elv_val(&mono.val, f)?;
    #[cfg(any(debug_assertions, tho_debug))]
    f.write_str(",")?;

    #[cfg(any(debug_assertions, tho_debug))]
    write!(f, "{:?}", mono.name.as_ref())?;

    Ok(())
}

#[cfg(not(any(debug_assertions, tho_debug)))]
fn fmt_mono_key(mono: &MonoKey, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("(")?;
    fmt_elv_val(&mono.val, f)?;
    f.write_str(")")
}

impl SynKey {
    pub fn new<I:Into<Box<[MonoKey]>>>(vals: I) -> Self {
        Self { vals: vals.into() }
    }

    pub fn from_vec(vals: Vec<MonoKey>) -> Self {
        Self { vals: vals.into_boxed_slice() }
    }

    pub fn as_slice(&self) -> &[MonoKey] {
        &self.vals
    }

    pub fn len(&self) -> usize {
        self.vals.len()
    }

    pub fn parent(&self) -> SynKey {
        SynKey::new(&self.vals[..self.vals.len()])
    }
}

/// Nice ergonomics: `CompKey::from([..])` with only one allocation (boxing the array).
impl<const N: usize> From<[MonoKey; N]> for SynKey {
    fn from(vals: [MonoKey; N]) -> Self {
        Self { vals: Box::new(vals) }
    }
}

impl From<MonoKey> for SynKey {
    fn from(vals: MonoKey) -> Self {
        Self { vals: Box::new([vals]) }
    }
}

impl From<&str> for SynKey {
    fn from(value: &str) -> Self {
        Self {
            vals: Box::new([mono_key!(value, value)])
        }
    }
}

impl From<String> for SynKey {
    fn from(value: String) -> Self {
        Self {
            vals: Box::new([mono_key!(value.clone(), value)])
        }
    }
}

/// Add now produces a dynamic-length CompKey (LEN = runtime sum).
impl Add<&SynKey> for &SynKey {
    type Output = SynKey;

    fn add(self, rhs: &SynKey) -> Self::Output {
        let mut v = Vec::with_capacity(self.vals.len() + rhs.vals.len());
        v.extend(self.vals.iter().cloned());
        v.extend(rhs.vals.iter().cloned());
        SynKey::from_vec(v)
    }
}

impl Add<SynKey> for &SynKey {
    type Output = SynKey;

    fn add(self, rhs: SynKey) -> Self::Output {
        let mut v = Vec::with_capacity(self.vals.len() + rhs.vals.len());
        v.extend(self.vals.iter().cloned());
        v.extend(rhs.vals.into_iter());
        SynKey::from_vec(v)
    }
}

impl Add<&MonoKey> for &SynKey {
    type Output = SynKey;

    fn add(self, rhs: &MonoKey) -> Self::Output {
        let mut v = Vec::with_capacity(self.vals.len() + 1);
        v.extend(self.vals.iter().cloned());
        v.push(rhs.clone());
        SynKey::from_vec(v)
    }
}

impl Add<MonoKey> for &SynKey {
    type Output = SynKey;

    fn add(self, rhs: MonoKey) -> Self::Output {
        let mut v = Vec::with_capacity(self.vals.len() + 1);
        v.extend(self.vals.iter().cloned());
        v.push(rhs);
        SynKey::from_vec(v)
    }
}


impl Add<&MonoKey> for MonoKey {
    type Output = SynKey;

    fn add(self, rhs: &MonoKey) -> Self::Output {
        SynKey {
            vals: Box::new([
                self,
                rhs.clone()
            ]),
        }
    }
}

impl Add<&SynKey> for &MonoKey {
    type Output = SynKey;

    fn add(self, rhs: &SynKey) -> Self::Output {
        let mut v = Vec::with_capacity(rhs.vals.len() + 1);
        v.push(self.clone());
        v.extend(rhs.vals.iter().cloned());
        SynKey::from_vec(v)
    }
}

impl Add<&SynKey> for MonoKey {
    type Output = SynKey;

    fn add(self, rhs: &SynKey) -> Self::Output {
        let mut v = Vec::with_capacity(rhs.vals.len() + 1);
        v.push(self);
        v.extend(rhs.vals.iter().cloned());
        SynKey::from_vec(v)
    }
}



impl From<(u32, &str)> for SynKey {
    fn from(value: (u32, &str)) -> Self {
        SynKey::from([mono_key!(value.0, value.1)])
    }
}

#[cfg(test)]
mod tests {
    use log::debug;
    use crate::comp_id::SynKey;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_add_1_2() {
        let key_a: SynKey = [mono_key!(0, "key1")].into();
        let key_b: SynKey = [mono_key!(1, "key2"), mono_key!(2, "key3")].into();

        let key_c = &key_a + &key_b;
        debug!("key_c={key_c:?}");

        assert_eq!(
            key_c.as_slice(),
            &[
                mono_key!(0, "key1"),
                mono_key!(1, "key2"),
                mono_key!(2, "key3"),
            ]
        );
    }

    #[test]
    fn test_partial_ord() {
        let key_0: SynKey = [mono_key!(0, "a"), mono_key!(10, "b")].into();
        let key_1: SynKey = [mono_key!(1, "a"), mono_key!(1, "b")].into();

        assert!(key_1.gt(&key_0));
    }
}
