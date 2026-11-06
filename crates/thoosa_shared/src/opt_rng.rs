#[cfg(not(test))]
use rand::rngs::ThreadRng;
#[cfg(test)]
use rand_mt::Mt19937GenRand32;

#[cfg(test)]
thread_local! {
    // Preserve the deterministic test sequence without exposing an unchecked mutable reference.
    static RNG: std::cell::RefCell<Mt19937GenRand32> = std::cell::RefCell::new(Mt19937GenRand32::from(5489));
}

// The callback keeps the mutable borrow inside this call; no RNG reference can escape.
#[cfg(not(test))]
pub fn opt_rnd<R>(f: impl FnOnce(&mut ThreadRng) -> R) -> R {
    f(&mut rand::rng())
}

#[cfg(test)]
pub fn opt_rnd<R>(f: impl FnOnce(&mut Mt19937GenRand32) -> R) -> R {
    RNG.with(|rng| f(&mut rng.borrow_mut()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_preserves_thread_local_sequence() {
        // A fresh thread starts with the documented seed, independent of other tests.
        std::thread::spawn(|| {
            let mut expected = Mt19937GenRand32::from(5489);
            for _ in 0..10 {
                assert_eq!(opt_rnd(|rng| rng.next_u32()), expected.next_u32());
            }
        })
        .join()
        .unwrap();
    }

    #[test]
    fn test_rng_rejects_overlapping_borrows_and_recovers() {
        let nested = std::panic::catch_unwind(|| opt_rnd(|_| opt_rnd(|rng| rng.next_u32())));
        assert!(nested.is_err());
        // Unwinding releases the checked borrow, so later calls remain usable.
        opt_rnd(|rng| rng.next_u32());
    }
}
