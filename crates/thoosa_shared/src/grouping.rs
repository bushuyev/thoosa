//! Connected components of similar motif slices, including fallible file input.
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use log::trace;

struct GroupSlicesBuilder<T> {
    pairs_groups: HashMap<T, usize>,
    parents: Vec<usize>,
}

impl<T> GroupSlicesBuilder<T>
where
    T: Hash + Eq + Debug,
{
    fn new() -> Self {
        Self { pairs_groups: HashMap::new(), parents: Vec::new() }
    }

    fn root(&mut self, mut group: usize) -> usize {
        // Path halving avoids recursive merging and keeps repeated bridges inexpensive.
        while self.parents[group] != group {
            self.parents[group] = self.parents[self.parents[group]];
            group = self.parents[group];
        }
        group
    }

    fn push(&mut self, _ln: usize, pair: (T, T)) {
        match (self.pairs_groups.get(&pair.0).copied(), self.pairs_groups.get(&pair.1).copied()) {
            (Some(a), Some(b)) => {
                let a = self.root(a);
                let b = self.root(b);
                self.parents[a.max(b)] = a.min(b);
            }
            (Some(group), None) => { self.pairs_groups.insert(pair.1, group); }
            (None, Some(group)) => { self.pairs_groups.insert(pair.0, group); }
            (None, None) => {
                // IDs index allocated storage, so they cannot wrap independently of capacity.
                let group = self.parents.len();
                self.parents.push(group);
                self.pairs_groups.insert(pair.0, group);
                self.pairs_groups.insert(pair.1, group);
            }
        }
    }

    fn finish(mut self) -> HashMap<usize, HashSet<T>> {
        let mut groups: HashMap<usize, HashSet<T>> = HashMap::new();
        for (value, group) in std::mem::take(&mut self.pairs_groups) {
            groups.entry(self.root(group)).or_default().insert(value);
        }
        groups
    }
}

//group of similar slices - motif
pub fn group_slices<T>(pairs: impl Iterator<Item = (T, T)>) -> HashMap<usize, HashSet<T>> //TODO replace with group_slices_try
where
    T: Hash + Eq + Debug,
{
    let mut builder = GroupSlicesBuilder::new();
    for (ln, pair) in pairs.enumerate() {
        trace!("group_slices: ln={ln} pairs={pair:?}");
        builder.push(ln, pair);
    }
    builder.finish()
}

pub fn group_slices_try<T, E>(
    pairs: impl Iterator<Item = Result<(T, T), E>>,
) -> Result<HashMap<usize, HashSet<T>>, E>
where
    T: Hash + Eq + Debug,
{
    let mut builder = GroupSlicesBuilder::new();
    for (ln, pair) in pairs.enumerate() {
        let pair = pair?;
        trace!("group_slices: ln={ln} pairs={pair:?}");
        builder.push(ln, pair);
    }
    Ok(builder.finish())
}

