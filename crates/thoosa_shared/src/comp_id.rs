use std::any::TypeId;
#[cfg(any(debug_assertions, tho_debug))]
use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::ops::Add;
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use thoosa_digest_id_derive::DigestHash;
use thoosa_named_derive::Named;
use thoosa_shared::traits::DataId;
use crate::traits::StageHash;

#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum ElvVal {
    Path(PathBuf),//for LoadNet - so that it can be loaded by path in CudaWorker avoiding SqlStore there for getting path by net_id
    Str(String),
    Num(usize),
    Bool(bool),
    DigestHash(blake3::Hash)
}

impl ::thoosa_digest_id::DigestHash for ElvVal
{
    fn update_digest(&self, __digest_hasher: &mut ::thoosa_digest_id::blake3::Hasher ) {
        ::thoosa_digest_id::__private::begin_enum(__digest_hasher, "thoosa.ElvVal:v1" );
        match self {
            Self::Path(__digest_field_0) => {
                ::thoosa_digest_id::__private::variant(__digest_hasher, "Path", 1usize );
                ::thoosa_digest_id::__private::field(__digest_hasher, "0",
                );
                // <&str as ::thoosa_digest_id::DigestHash>::update_digest(__digest_field_0.as_os_str().as_bytes(), __digest_hasher );
                __digest_hasher.update(&__digest_field_0.as_os_str().as_bytes());
            }
            Self::Str(__digest_field_0) => {
                ::thoosa_digest_id::__private::variant(__digest_hasher, "Str", 1usize );
                ::thoosa_digest_id::__private::field(__digest_hasher, "0",
                );
                <String as ::thoosa_digest_id::DigestHash>::update_digest(__digest_field_0, __digest_hasher );
            }
            Self::Num(__digest_field_0) => {
                ::thoosa_digest_id::__private::variant(__digest_hasher, "Num", 1usize );
                ::thoosa_digest_id::__private::field(__digest_hasher, "0",
                );
                <usize as ::thoosa_digest_id::DigestHash>::update_digest(__digest_field_0, __digest_hasher );
            }
            Self::Bool(__digest_field_0) => {
                ::thoosa_digest_id::__private::variant(__digest_hasher, "Bool", 1usize );
                ::thoosa_digest_id::__private::field(__digest_hasher, "0",
                );
                <bool as ::thoosa_digest_id::DigestHash>::update_digest(__digest_field_0, __digest_hasher );
            }
            Self::DigestHash(__digest_field_0)
            => {
                ::thoosa_digest_id::__private::variant(__digest_hasher, "DigestHash", 1usize );
                ::thoosa_digest_id::__private::field(__digest_hasher, "0" );
                __digest_hasher.update(&__digest_field_0.as_slice());
            }
        }
    }
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
            ElvVal::Path(path) => {
                write!(f, "{}", path.to_string_lossy())
            }
            ElvVal::DigestHash(hash) => {
                write!(f, "{}", hash)
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

impl TryFrom<i32> for ElvVal {
    type Error = std::num::TryFromIntError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        // Signed input is accepted only if it preserves the unsigned key's identity.
        Ok(ElvVal::Num(usize::try_from(value)?))
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

impl From<PathBuf> for ElvVal {
    fn from(v: PathBuf) -> Self {
        ElvVal::Path(v)
    }
}

impl From<blake3::Hash> for ElvVal {
    fn from(v: blake3::Hash) -> Self {
        ElvVal::DigestHash(v)
    }
}


#[derive(Debug, Clone, DigestHash, Named, Serialize, Deserialize)]
#[digest(tag = "thoosa.MonoKey:v1")]
pub struct MonoKey {
    val: ElvVal,

    #[cfg(any(debug_assertions, tho_debug))]
    name: Cow<'static, str>,
}

impl Display for MonoKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
       
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

    pub fn to_data_id(&self) -> anyhow::Result<DataId> {
        // Database IDs are narrower than cache keys; never alias a large key to a different row.
        match &self.val {
            ElvVal::Num(val) => Ok(DataId(u32::try_from(*val)?)),
            val => anyhow::bail!("can't convert {val} to data_id: expected a numeric key"),
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
#[derive(Eq, PartialEq, Hash, Clone, DigestHash, Named, Serialize, Deserialize)]
#[digest(tag = "thoosa.SynKey:v1")]
pub struct SynKey {//σύνθετος
    pub vals: Box<[MonoKey]>,
}

impl Display for SynKey{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for val in self.vals.iter() {
            write!(f, "/{}", val)?;
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
        ElvVal::Bool(value) => write!(f, "{value}"),
        ElvVal::Path(value) => write!(f, "{}", value.to_string_lossy()),
        ElvVal::DigestHash(value) => write!(f, "{value}")
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

impl<const N: usize> From<[MonoKey; N]> for SynKey {
    fn from(vals: [MonoKey; N]) -> Self {
        Self { vals: Box::new(vals) }
    }
}

impl From<Vec<MonoKey>> for SynKey {
    fn from(vals: Vec<MonoKey>) -> Self {
        Self { vals: vals.into_boxed_slice() }
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

impl Add<MonoKey> for MonoKey {
    type Output = SynKey;

    fn add(self, rhs: MonoKey) -> Self::Output {
        SynKey {
            vals: Box::new([
                self.clone(),
                rhs
            ]),
        }
    }
}


impl Add<MonoKey> for &MonoKey {
    type Output = SynKey;

    fn add(self, rhs: MonoKey) -> Self::Output {
        SynKey {
            vals: Box::new([
                self.clone(),
                rhs
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

pub mod hashmap_as_vec {
    use std::collections::HashMap;
    use serde::{Deserializer, Serializer};
    use super::*;

    pub fn serialize<S, K, V>(
        value: &HashMap<K, V>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        K: Serialize,
        V: Serialize,
    {
        value.iter().collect::<Vec<_>>().serialize(serializer)
    }

    pub fn deserialize<'de, D, K, V>(
        deserializer: D,
    ) -> Result<HashMap<K, V>, D::Error>
    where
        D: Deserializer<'de>,
        K: Deserialize<'de> + Eq + Hash,
        V: Deserialize<'de>,
    {
        let entries = Vec::<(K, V)>::deserialize(deserializer)?;
        Ok(entries.into_iter().collect())
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
    fn test_identity_conversions_reject_negative_and_narrowing_keys() -> anyhow::Result<()> {
        use super::ElvVal;
        assert!(ElvVal::try_from(-1i32).is_err());
        assert!(ElvVal::try_from(i32::MIN).is_err());
        assert_eq!(ElvVal::try_from(0i32)?, ElvVal::from(0u32));
        assert_eq!(ElvVal::try_from(i32::MAX)?, ElvVal::from(i32::MAX as u32));
        assert_eq!(mono_key!(u32::MAX, "max").to_data_id()?.0, u32::MAX);
        assert_eq!(mono_key!(0usize, "zero").to_data_id()?.0, 0);
        assert!(mono_key!("0", "string").to_data_id().is_err());
        // These two keys previously narrowed to the same database ID on 64-bit platforms.
        if let Some(large) = (u32::MAX as usize).checked_add(1) {
            assert!(mono_key!(large, "large").to_data_id().is_err());
        }
        Ok(())
    }

    #[test]
    fn test_add_1_2() {
        let key_a: SynKey = [mono_key!(0u32, "key1")].into();
        let key_b: SynKey = [mono_key!(1u32, "key2"), mono_key!(2u32, "key3")].into();

        let key_c = &key_a + &key_b;
        debug!("key_c={key_c:?}");

        assert_eq!(
            key_c.as_slice(),
            &[
                mono_key!(0u32, "key1"),
                mono_key!(1u32, "key2"),
                mono_key!(2u32, "key3"),
            ]
        );
    }
}
