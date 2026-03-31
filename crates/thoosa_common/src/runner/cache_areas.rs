use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use log::debug;
use thoosa_shared::{Cause, ThoError};

/// Public “area handle”: global bit index (0..n_areas).
pub type CacheAreaId = u64;

#[inline]
fn area_to_word_mask(area: CacheAreaId) -> Option<(usize, u64)> {
    if area == 0 {
        return None;
    }
    let bit = (area - 1) as usize; // <-- 1-based -> 0-based
    let word = bit / 64;
    let shift = (bit % 64) as u32;
    Some((word, 1u64 << shift))
}

static CACHE_AREAS_WORDS:usize = 10;

#[derive(Debug)]
pub struct CacheAreasKeeper {
    cursor: usize,
    words: Vec<u64>, // memory region of bits: 1 = used, 0 = free
}

impl CacheAreasKeeper {
    /// Allocate `n_words * 64` areas.
    pub fn new(n_words: usize) -> Self {
        Self { words: vec![0; n_words], cursor: 0 }
    }

    pub fn capacity(&self) -> u64 {
        (self.words.len() as u64) * 64
    }

    /// Returns a non-zero CacheArea. First is 1.
    pub fn next_available_cache_area(&mut self) -> Result<CacheAreaId, ThoError> {
        if self.words.is_empty() {
            return Err(Cause::SomethingElse("no cache areas available".into()).into());
        }

        let n = self.words.len();

        for step in 0..n {
            let wi = (self.cursor + step) % n;
            let w = self.words[wi];

            if w == u64::MAX {
                continue; // this word is full
            }

            let free = !w;
            let bit_in_word = free.trailing_zeros(); // 0..63 (because w != MAX)
            let mask = 1u64 << bit_in_word;

            self.words[wi] |= mask;
            self.cursor = wi;

            let bit_index_0_based = wi * 64 + bit_in_word as usize;
            let area_1_based = (bit_index_0_based as u64) + 1; // <-- key change

            debug!("next_available_cache_area: returning area_1_based={area_1_based:?}");

            return Ok(area_1_based);
        }

        Err(Cause::SomethingElse("no cache areas available".into()).into())
    }

    pub fn dispose_cache_area(&mut self, area: CacheAreaId) {
        let Some((word, mask)) = area_to_word_mask(area) else {
            debug!("dispose_cache_area: got area=0 (ignored)");
            return;
        };

        if let Some(w) = self.words.get_mut(word) {
            debug!("dispose_cache_area: area={area} word={word} mask={mask:064b}");
            *w &= !mask;
            self.cursor = word; // optional hint for faster reuse
        } else {
            debug!("dispose_cache_area: out-of-range area={area} (word={word})");
        }
    }

    pub fn words(&self) -> &[u64] { &self.words }
}

impl Default for CacheAreasKeeper {
    fn default() -> Self {
        CacheAreasKeeper::new(CACHE_AREAS_WORDS)
    }
}

#[cfg(test)]
mod cache_area_keeper_tests {
    use thoosa_shared::ThoError;
    use crate::runner::cache_areas::{CacheAreaId, CacheAreasKeeper};

    fn alloc_n(k: &mut CacheAreasKeeper, n: usize) -> Result<Vec<CacheAreaId>, ThoError> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(k.next_available_cache_area()?);
        }
        Ok(out)
    }

    #[test]
    fn test_first_area_is_one_and_never_zero() -> Result<(), ThoError> {
        let mut k = CacheAreasKeeper::new(1);
        let a = k.next_available_cache_area()?;
        assert_eq!(a, 1);
        assert_ne!(a, 0);
        Ok(())
    }

    #[test]
    fn test_allocates_sequentially_within_one_word_and_then_full() -> Result<(), ThoError> {
        let mut k = CacheAreasKeeper::new(1);

        let areas = alloc_n(&mut k, 64)?;
        assert_eq!(areas[0], 1);
        assert_eq!(areas[63], 64);

        // exactly full now
        assert!(k.next_available_cache_area().is_err());
        assert_eq!(k.words()[0], u64::MAX);

        Ok(())
    }

    #[test]
    fn test_dispose_and_reuse_within_one_word_lowest_free_first() -> Result<(), ThoError> {
        let mut k = CacheAreasKeeper::new(1);
        let _ = alloc_n(&mut k, 64)?;

        // Free bits 0, 31, 63 (areas 1, 32, 64)
        k.dispose_cache_area(1);
        k.dispose_cache_area(32);
        k.dispose_cache_area(64);

        // Re-allocate: should pick lowest free bit in that word each time
        assert_eq!(k.next_available_cache_area()?, 1);
        assert_eq!(k.next_available_cache_area()?, 32);
        assert_eq!(k.next_available_cache_area()?, 64);

        Ok(())
    }

    #[test]
    fn crosses_word_boundary() -> Result<(), ThoError> {
        let mut k = CacheAreasKeeper::new(2);

        // Capacity = 128 areas
        assert_eq!(k.capacity(), 128);

        // The 65th allocation should be area 65 (first bit of word 2)
        let areas = alloc_n(&mut k, 65)?;
        assert_eq!(areas[0], 1);
        assert_eq!(areas[63], 64);
        assert_eq!(areas[64], 65);

        Ok(())
    }

    #[test]
    fn test_full_two_words_then_error() -> Result<(), ThoError> {
        let mut k = CacheAreasKeeper::new(2);
        let _ = alloc_n(&mut k, 128)?;
        assert!(k.next_available_cache_area().is_err());

        assert_eq!(k.words()[0], u64::MAX);
        assert_eq!(k.words()[1], u64::MAX);
        Ok(())
    }

    #[test]
    fn test_dispose_zero_is_ignored_and_does_not_change_state() -> Result<(), ThoError> {
        let mut k = CacheAreasKeeper::new(1);

        let a1 = k.next_available_cache_area()?;
        assert_eq!(a1, 1);

        let before = k.words().to_vec();
        k.dispose_cache_area(0); // should be ignored
        let after = k.words().to_vec();

        assert_eq!(before, after);
        Ok(())
    }

    #[test]
    fn test_dispose_out_of_range_is_ignored_and_does_not_change_state() -> Result<(), ThoError> {
        let mut k = CacheAreasKeeper::new(1);
        let _ = alloc_n(&mut k, 10)?;

        let before = k.words().to_vec();
        let out_of_range = k.capacity() + 123; // > 64 for 1 word
        k.dispose_cache_area(out_of_range);
        let after = k.words().to_vec();

        assert_eq!(before, after);
        Ok(())
    }

    #[test]
    fn test_reuse_after_dispose_may_depend_on_cursor_across_words() -> Result<(), ThoError> {
        // This test documents current behavior: allocation starts scanning from `cursor`.
        let mut k = CacheAreasKeeper::new(2);
        let _ = alloc_n(&mut k, 128)?;
        assert!(k.next_available_cache_area().is_err());

        // Free one in word0 and one in word1; last dispose sets cursor to that word.
        k.dispose_cache_area(1);   // word0 bit0
        k.dispose_cache_area(65);  // word1 bit0 (cursor ends at word1)

        // Because cursor points at word1, it reuses 65 first, then 1.
        assert_eq!(k.next_available_cache_area()?, 65);
        assert_eq!(k.next_available_cache_area()?, 1);

        Ok(())
    }
}


/// A thing to keep track of cache-areas assigned.
#[derive(Debug)]
pub struct CacheArea {
    cache_words: Box<[AtomicU64]>,
    assigned_count: AtomicU32, // number of set bits across all words
}


impl PartialEq for CacheArea {
    fn eq(&self, other: &Self) -> bool {
        if self.cache_words.len() != other.cache_words.len() {
            return false;
        }

        self.cache_words
            .iter()
            .zip(other.cache_words.iter())
            .all(|(a, b)| a.load(Ordering::Acquire) == b.load(Ordering::Acquire))
    }
}

impl Eq for CacheArea {}


impl From<u64> for CacheArea {
    fn from(value: u64) -> Self {
        let cache_area = CacheArea::new(CACHE_AREAS_WORDS);

        cache_area.add_cache_area(value);

        cache_area
    }
}

impl Clone for CacheArea {
    fn clone(&self) -> Self {

        let mut v = Vec::with_capacity(self.cache_words.len());
        for w in self.cache_words.iter() {
            v.push(AtomicU64::new(w.load(Ordering::Acquire)));
        }

        Self {
            cache_words: v.into_boxed_slice(),
            assigned_count: AtomicU32::new(self.assigned_count.load(Ordering::Acquire)),
        }
    }
}
impl CacheArea {
    /// Allocates storage for `n_words * 64` possible areas (handles 1..=n_words*64).
    pub fn new(n_words: usize) -> Self {
        let mut v = Vec::with_capacity(n_words);
        v.resize_with(n_words, || AtomicU64::new(0));
        Self {
            cache_words: v.into_boxed_slice(),
            assigned_count: AtomicU32::new(0),
        }
    }

    pub fn capacity(&self) -> CacheAreaId {
        (self.cache_words.len() as CacheAreaId) * 64
    }

    #[inline]
    pub fn is_cached(&self) -> bool {
        self.assigned_count.load(Ordering::Acquire) != 0
    }

    /// Assign an area to this entity. Returns true if it was newly added.
    pub fn add_cache_area(&self, ca: CacheAreaId) -> bool {
        let Some((word, mask)) = self.area_to_word_mask_checked(ca) else {
            return false; // ca==0 or out-of-range
        };

        let prev = self.cache_words[word].fetch_or(mask, Ordering::AcqRel);

        // If bit was previously 0, we truly added it.
        if (prev & mask) == 0 {
            self.assigned_count.fetch_add(1, Ordering::AcqRel);
            true
        } else {
            false
        }
    }

    pub fn contains(&self, ca: CacheAreaId) -> bool {
        let Some((word, mask)) = self.area_to_word_mask_checked(ca) else {
            return false;
        };

        (self.cache_words[word].load(Ordering::Acquire) & mask) != 0
    }

    /// Remove an assigned area. Returns true if it was actually removed.
    pub fn remove_cache_area(&self, ca: CacheAreaId) -> bool {
        let Some((word, mask)) = self.area_to_word_mask_checked(ca) else {
            return false; // ca==0 or out-of-range
        };

        let prev = self.cache_words[word].fetch_and(!mask, Ordering::AcqRel);

        // If bit was previously 1, we truly removed it.
        if (prev & mask) != 0 {
            self.assigned_count.fetch_sub(1, Ordering::AcqRel);
            true
        } else {
            false
        }
    }

    /// Snapshot for debugging/tests.
    pub fn snapshot_words(&self) -> Vec<u64> {
        self.cache_words
            .iter()
            .map(|w| w.load(Ordering::Acquire))
            .collect()
    }

    #[inline]
    fn area_to_word_mask_checked(&self, ca: CacheAreaId) -> Option<(usize, u64)> {
        if ca == 0 {
            return None;
        }
        let bit0 = (ca - 1) as usize; // 1-based -> 0-based
        let word = bit0 / 64;
        if word >= self.cache_words.len() {
            return None;
        }
        let shift = (bit0 % 64) as u32;
        Some((word, 1u64 << shift))
    }

    pub fn assigned_count(&self) -> u32 {
        self.assigned_count.load(Ordering::Acquire)
    }
}

impl Default for CacheArea {
    fn default() -> Self {
        CacheArea::new(CACHE_AREAS_WORDS)
    }
}

#[cfg(test)]
mod cache_area_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn basic_add_remove_single_word() {
        let c = CacheArea::new(1);
        assert_eq!(c.capacity(), 64);
        assert!(!c.is_cached());
        assert_eq!(c.assigned_count(), 0);

        // First add sets cached=true
        assert!(c.add_cache_area(1));
        assert!(c.is_cached());
        assert_eq!(c.assigned_count(), 1);

        // Adding same area again is a no-op
        assert!(!c.add_cache_area(1));
        assert_eq!(c.assigned_count(), 1);

        // Remove clears cached
        assert!(c.remove_cache_area(1));
        assert!(!c.is_cached());
        assert_eq!(c.assigned_count(), 0);

        // Removing again is a no-op (no underflow)
        assert!(!c.remove_cache_area(1));
        assert_eq!(c.assigned_count(), 0);
    }

    #[test]
    fn multi_word_mapping_and_snapshot() {
        let c = CacheArea::new(2);
        assert_eq!(c.capacity(), 128);

        // ca=1 -> word0 bit0
        assert!(c.add_cache_area(1));
        // ca=64 -> word0 bit63 (since 1-based)
        assert!(c.add_cache_area(64));
        // ca=65 -> word1 bit0
        assert!(c.add_cache_area(65));

        let snap = c.snapshot_words();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0], (1u64 << 0) | (1u64 << 63));
        assert_eq!(snap[1], 1u64 << 0);

        assert!(c.is_cached());
        assert_eq!(c.assigned_count(), 3);

        // Remove ca=64 (word0 bit63)
        assert!(c.remove_cache_area(64));
        let snap2 = c.snapshot_words();
        assert_eq!(snap2[0], 1u64 << 0);
        assert_eq!(snap2[1], 1u64 << 0);
        assert_eq!(c.assigned_count(), 2);
    }

    #[test]
    fn zero_and_out_of_range_are_ignored() {
        let c = CacheArea::new(1);

        assert!(!c.add_cache_area(0));
        assert!(!c.remove_cache_area(0));
        assert_eq!(c.assigned_count(), 0);
        assert_eq!(c.snapshot_words()[0], 0);

        // capacity=64 => valid IDs are 1..=64
        assert!(!c.add_cache_area(65));
        assert!(!c.remove_cache_area(65));
        assert_eq!(c.assigned_count(), 0);
        assert_eq!(c.snapshot_words()[0], 0);
    }

    #[test]
    fn concurrent_unique_adds_and_removes() {
        let c = Arc::new(CacheArea::new(2)); // capacity=128

        // Add 16 distinct areas concurrently
        let mut ths = Vec::new();
        for i in 0..16u64 {
            let c2 = Arc::clone(&c);
            ths.push(thread::spawn(move || {
                let area = i + 1; // 1..=16
                assert!(c2.add_cache_area(area));
            }));
        }
        for t in ths {
            t.join().unwrap();
        }

        assert!(c.is_cached());
        assert_eq!(c.assigned_count(), 16);

        // Remove them concurrently
        let mut ths2 = Vec::new();
        for i in 0..16u64 {
            let c2 = Arc::clone(&c);
            ths2.push(thread::spawn(move || {
                let area = i + 1;
                assert!(c2.remove_cache_area(area));
            }));
        }
        for t in ths2 {
            t.join().unwrap();
        }

        assert!(!c.is_cached());
        assert_eq!(c.assigned_count(), 0);
        assert!(c.snapshot_words().into_iter().all(|w| w == 0));
    }

    #[test]
    fn eq_compares_words_snapshot() {
        let a = CacheArea::new(2);
        let b = CacheArea::new(2);

        assert_eq!(a, b);

        a.add_cache_area(1);
        assert_ne!(a, b);

        b.add_cache_area(1);
        assert_eq!(a, b);

        a.add_cache_area(65); // word 1, bit 0
        assert_ne!(a, b);

        b.add_cache_area(65);
        assert_eq!(a, b);
    }

    #[test]
    fn eq_compares_default() {
        let a = CacheArea::default();
        let b = CacheArea::default();

        assert_eq!(a, b);
    }

    #[test]
    fn contains_basic() {
        let c = CacheArea::new(2); // 128 capacity

        assert!(!c.contains(0));
        assert!(!c.contains(1));
        assert!(!c.contains(65));

        c.add_cache_area(1);
        c.add_cache_area(65);

        assert!(c.contains(1));
        assert!(c.contains(65));
        assert!(!c.contains(2));
        assert!(!c.contains(64));

        c.remove_cache_area(1);
        assert!(!c.contains(1));
        assert!(c.contains(65));
    }

    #[test]
    fn contains_out_of_range_is_false() {
        let c = CacheArea::new(1); // capacity 64
        assert!(!c.contains(65));
        assert!(!c.contains(u64::MAX));
    }
}

