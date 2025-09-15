use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use ambassador::delegatable_trait;
use mockall::automock;

#[delegatable_trait]
pub trait CachableIdTr: Any + Debug + Send + Sync  {

    fn eqq(&self, _other:&dyn CachableIdTr) -> bool{
        false
    }

    fn as_any<'a>(&'a self) ->  &'a (dyn Any + Send + Sync);
    fn eq_dyn(&self, other: &dyn CachableIdTr) -> bool;
    fn hash_dyn(&self, state: &mut dyn Hasher);
}



pub mod mock_cachable_id{
    use mockall::mock;
    use std::any::Any;
    use std::hash::{Hash, Hasher};
    use crate::traits::CachableIdTr;

    mock! {
        #[derive(Debug, Clone, Eq, PartialEq, Hash)]
        pub CachableId {}

        impl CachableIdTr for CachableId {
            fn eqq(&self, other:&dyn CachableIdTr) -> bool;
            #[allow(unused_parens)]
            fn as_any<'a>(&'a self) ->  (&'a (dyn Any + Send + Sync));
            fn eq_dyn(&self, other: &dyn CachableIdTr) -> bool;
            fn hash_dyn(&self, state: &mut dyn Hasher);
        }

        impl Clone for CachableId {
            fn clone(&self) -> Self;
            fn clone_from(&mut self, source: &Self);
        }


    }

    impl Hash for MockCachableId {
        fn hash<H: Hasher>(&self, _state: &mut H) {

        }
    }

    impl Eq for MockCachableId {

    }

    impl PartialEq for MockCachableId {
        fn eq(&self, _other: &Self) -> bool {
            false
        }
    }

}


// dyn Key itself: Eq/Hash via the object-safe methods
impl PartialEq for dyn CachableIdTr {
    fn eq(&self, other: &Self) -> bool { self.eq_dyn(other) }
}

impl Eq for dyn CachableIdTr {}

impl Hash for dyn CachableIdTr {
    fn hash<H: Hasher>(&self, h: &mut H) {
        struct Erased<'h, H2: Hasher>(&'h mut H2);
        impl<'h, H2: Hasher> Hasher for Erased<'h, H2> {
            fn finish(&self) -> u64 { self.0.finish() }
            fn write(&mut self, b: &[u8]) { self.0.write(b) }
        }
        let mut e = Erased(h);
        self.hash_dyn(&mut e);
    }
}

impl<T: Send + Sync + Debug + Eq + Hash + Named + 'static> CachableIdTr for T {

    #[allow(unused_parens)]
    fn as_any<'a>(&'a self) ->  (&'a (dyn Any + Send + Sync)) { self }

    fn eq_dyn(&self, other: &dyn CachableIdTr) -> bool {
        other.as_any().downcast_ref::<T>().map_or(false, |v| self == v)
    }

    fn hash_dyn(&self, state: &mut dyn Hasher) {
        struct Erased<'h>(&'h mut dyn Hasher);
        impl<'h> Hasher for Erased<'h> {
            fn finish(&self) -> u64 { self.0.finish() }
            fn write(&mut self, b: &[u8]) { self.0.write(b) }
        }
        let mut e = Erased(state);
        self.hash(&mut e);
    }
}

impl dyn CachableIdTr {
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }
}

//triggers blanked impl<T: ... Named + 'static> CachableId for T
#[automock]
pub trait Named {
    fn name(&self) -> &'static str {
        "n/a"
    }
}

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct SimpleId(pub &'static str);

impl Named for SimpleId {
    fn name(&self) -> &'static str {
        "SimpleId"
    }
}

impl From<&'static str> for SimpleId {
    fn from(value: &'static str) -> Self {
        SimpleId(value)
    }
}

impl From<String> for SimpleId {
    fn from(value: String) -> Self {
        SimpleId(value.leak())
    }
}

impl Display for SimpleId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[delegatable_trait]
#[automock(type ID=DummyCachableId;)]
pub trait WithId {
    type ID: CachableIdTr + Send + Sync + Eq + Hash + Clone + Debug;

    fn id(&self) -> &Self::ID;
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct DummyCachableId(&'static str);

impl Named for DummyCachableId {}

impl From<&'static str> for DummyCachableId {
    fn from(value: &'static str) -> Self {
        DummyCachableId(value)
    }
}