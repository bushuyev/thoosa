use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::ops::Add;
use crate::traits::Named;

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum ElvVal {
    Str(String),
    U32(u32)
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



#[derive(Debug, Clone)]
pub struct KeyElt {
    val: ElvVal,

    #[cfg(debug_assertions)]
    name: Cow<'static, str>,
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

    // Optional: easy access in debug builds
    #[cfg(debug_assertions)]
    pub fn name(&self) -> &str {
        self.name.as_ref()
    }
}

#[macro_export]
macro_rules! key_elt {
    // string literal -> borrowed
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

    // any expression -> owned (works for String, &str, &String, etc.)
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
            // $name intentionally not referenced here
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


#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct CompKey<const LEN:usize> {
    vals:[KeyElt;LEN]
}

impl <const LEN:usize> CompKey<LEN> {
    pub fn new(vals: [KeyElt; LEN]) -> Self {
        Self { vals }
    }
}

unsafe impl<const LEN:usize> Send for CompKey<LEN> {}
unsafe impl<const LEN:usize> Sync for CompKey<LEN> {}

impl<const LEN:usize> Named for CompKey<LEN> {
    fn name(&self) -> &'static str {
        "CompKey"
    }
}

impl<const LEN_A: usize, const LEN_B: usize> Add<&CompKey<LEN_B>> for &CompKey<LEN_A>
where
    [(); LEN_A + LEN_B]:, // “const-evaluatable” witness
{
    type Output = CompKey<{ LEN_A + LEN_B }>;

    fn add(self, rhs: &CompKey<LEN_B>) -> Self::Output {
        let mut it = self.vals.iter().chain(rhs.vals.iter());
        let vals: [KeyElt; LEN_A + LEN_B] = std::array::from_fn(|_| {
            it.next()
                .map(|v|v.clone())
                .expect(&format!("adding composite keys failed: self={:?} rhs={:?}", self, rhs))
        });
        CompKey::new(vals)
    }
}

impl From<(u32, &'static str)> for CompKey<1> {
    fn from(value: (u32, &'static str)) -> Self {
        CompKey::new([key_elt!(value.0, value.1)])
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
    fn test_add_1_2(){
        
        let key_a = CompKey::new([key_elt!(0, "key1")]);
        let key_b = CompKey::new([key_elt!(1, "key2"), key_elt!(2, "key3")]);

        let key_c:CompKey<3> = &key_a + &key_b;
        debug!("key_c={key_c:?}");

        assert_eq!(
             [KeyElt { val: 0.into(), name: "key1".into() }, KeyElt { val: 1.into(), name: "key2".into() }, KeyElt { val: 2.into(), name: "key3".into() }],
            key_c.vals
        );
    }

}