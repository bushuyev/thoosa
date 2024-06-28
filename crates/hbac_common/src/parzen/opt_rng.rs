use std::cell::{Ref, RefMut, UnsafeCell};
use std::rc::Rc;
use rand::rngs::adapter::ReseedingRng;
use rand::rngs::OsRng;
use rand::{Rng, RngCore, SeedableRng, thread_rng};
use rand_mt::Mt19937GenRand32;

use std::cell::RefCell;
use rand::prelude::ThreadRng;

thread_local! {
    //https://stackoverflow.com/questions/75840391/why-is-the-default-seed-of-mt19937-5489 - to make it produce same values as rng = np.random.RandomState(5489)
    static FOO: Rc<UnsafeCell<Mt19937GenRand32>> = Rc::new(UnsafeCell::new(Mt19937GenRand32::from(5489)));
}


#[cfg(not(test))]
pub fn opt_rnd<'a>() -> ThreadRng {
    thread_rng()
}


#[cfg(test)]
pub fn opt_rnd<'a>() -> &'a mut Mt19937GenRand32 {
    unsafe { &mut *FOO.with(|foo| foo.clone()).get() }
}

