use std::borrow::Cow;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::Add;
use crate::traits::Named;

#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum ElvVal {
    Str(String),
    U32(u32),
}

impl From<u32> for ElvVal {
    fn from(value: u32) -> Self {
        ElvVal::U32(value)
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

#[derive(Debug, Clone, Ord)]
pub struct MonoKey {
    val: ElvVal,

    #[cfg(debug_assertions)]
    name: Cow<'static, str>,
}

impl From<&str> for MonoKey {
    fn from(value: &str) -> Self {
        MonoKey {
            val: value.into(),
            #[cfg(debug_assertions)]
            name: Cow::Owned(value.to_owned()),
        }
    }
}

impl PartialOrd for MonoKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

impl Named for MonoKey {
    fn name(&self) -> &'static str {
        "MonoKey"
    }
}

impl MonoKey {
    pub fn new(
        val: ElvVal,
        #[cfg(debug_assertions)] name: Cow<'static, str>,
    ) -> Self {
        Self {
            val,
            #[cfg(debug_assertions)]
            name,
        }
    }

    #[cfg(debug_assertions)]
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }
}

#[macro_export]
macro_rules! mono_key {
    ($val:expr, $name:literal $(,)?) => {{
        #[cfg(debug_assertions)]
        {
            thoosa_shared::comp_id::MonoKey::new(
                ($val).into(),
                ::std::borrow::Cow::Borrowed($name),
            )
        }
        #[cfg(not(debug_assertions))]
        {
            thoosa_shared::comp_id::MonoKey::new(($val).into())
        }
    }};

    ($val:expr, $name:expr $(,)?) => {{
        #[cfg(debug_assertions)]
        {
            thoosa_shared::comp_id::MonoKey::new(
                ($val).into(),
                ::std::borrow::Cow::Owned(::std::string::String::from($name)),
            )
        }
        #[cfg(not(debug_assertions))]
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
#[derive(Debug, Eq, PartialEq, Hash, Clone, Ord, PartialOrd)]
pub struct SynKey {
    vals: Box<[MonoKey]>,
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

impl Add<&MonoKey> for &SynKey {
    type Output = SynKey;

    fn add(self, rhs: &MonoKey) -> Self::Output {
        let mut v = Vec::with_capacity(self.vals.len() + 1);
        v.extend(self.vals.iter().cloned());
        v.push(rhs.clone());
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



impl Named for SynKey {
    fn name(&self) -> &'static str {
        "CompKey"
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
    use crate::comp_id::{SynKey, MonoKey};

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
                MonoKey { val: 0.into(), name: "key1".into() },
                MonoKey { val: 1.into(), name: "key2".into() },
                MonoKey { val: 2.into(), name: "key3".into() },
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
