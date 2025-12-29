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
pub struct KeyElt {
    val: ElvVal,

    #[cfg(debug_assertions)]
    name: Cow<'static, str>,
}

impl PartialOrd for KeyElt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

impl KeyElt {
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
macro_rules! key_elt {
    ($val:expr, $name:literal $(,)?) => {{
        #[cfg(debug_assertions)]
        {
            thoosa_shared::comp_id::KeyElt::new(
                ($val).into(),
                ::std::borrow::Cow::Borrowed($name),
            )
        }
        #[cfg(not(debug_assertions))]
        {
            thoosa_shared::comp_id::KeyElt::new(($val).into())
        }
    }};

    ($val:expr, $name:expr $(,)?) => {{
        #[cfg(debug_assertions)]
        {
            thoosa_shared::comp_id::KeyElt::new(
                ($val).into(),
                ::std::borrow::Cow::Owned(::std::string::String::from($name)),
            )
        }
        #[cfg(not(debug_assertions))]
        {
            thoosa_shared::comp_id::KeyElt::new(($val).into())
        }
    }};
}

impl Eq for KeyElt {}
impl PartialEq for KeyElt {
    fn eq(&self, other: &Self) -> bool {
        self.val.eq(&other.val)
    }
}

impl Hash for KeyElt {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.val.hash(state)
    }
}

/// ✅ Dynamic-length key, stored as a boxed slice (one allocation, no capacity slack).
#[derive(Debug, Eq, PartialEq, Hash, Clone, Ord, PartialOrd)]
pub struct CompKey {
    vals: Box<[KeyElt]>,
}

impl CompKey {
    pub fn new<I:Into<Box<[KeyElt]>>>(vals: I) -> Self {
        Self { vals: vals.into() }
    }

    pub fn from_vec(vals: Vec<KeyElt>) -> Self {
        Self { vals: vals.into_boxed_slice() }
    }

    pub fn as_slice(&self) -> &[KeyElt] {
        &self.vals
    }

    pub fn len(&self) -> usize {
        self.vals.len()
    }
}

/// Nice ergonomics: `CompKey::from([..])` with only one allocation (boxing the array).
impl<const N: usize> From<[KeyElt; N]> for CompKey {
    fn from(vals: [KeyElt; N]) -> Self {
        Self { vals: Box::new(vals) }
    }
}

impl From<&str> for CompKey {
    fn from(value: &str) -> Self {
        Self {
            vals: Box::new([key_elt!(value, value)])
        }
    }
}

/// Add now produces a dynamic-length CompKey (LEN = runtime sum).
impl Add<&CompKey> for &CompKey {
    type Output = CompKey;

    fn add(self, rhs: &CompKey) -> Self::Output {
        let mut v = Vec::with_capacity(self.vals.len() + rhs.vals.len());
        v.extend(self.vals.iter().cloned());
        v.extend(rhs.vals.iter().cloned());
        CompKey::from_vec(v)
    }
}

impl Named for CompKey {
    fn name(&self) -> &'static str {
        "CompKey"
    }
}

impl From<(u32, &str)> for CompKey {
    fn from(value: (u32, &str)) -> Self {
        CompKey::from([key_elt!(value.0, value.1)])
    }
}

#[cfg(test)]
mod tests {
    use log::debug;
    use crate::comp_id::{CompKey, KeyElt};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_add_1_2() {
        let key_a: CompKey = [key_elt!(0, "key1")].into();
        let key_b: CompKey = [key_elt!(1, "key2"), key_elt!(2, "key3")].into();

        let key_c = &key_a + &key_b;
        debug!("key_c={key_c:?}");

        assert_eq!(
            key_c.as_slice(),
            &[
                KeyElt { val: 0.into(), name: "key1".into() },
                KeyElt { val: 1.into(), name: "key2".into() },
                KeyElt { val: 2.into(), name: "key3".into() },
            ]
        );
    }

    #[test]
    fn test_partial_ord() {
        let key_0: CompKey = [key_elt!(0, "a"), key_elt!(10, "b")].into();
        let key_1: CompKey = [key_elt!(1, "a"), key_elt!(1, "b")].into();

        assert!(key_1.gt(&key_0));
    }
}
