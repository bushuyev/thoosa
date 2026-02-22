use thoosa_shared::TensorIdSliceIdx;
use thoosa_shared::resource::ResourceId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceSlices {
    pub id: ResourceId,
    pub size: usize,
}

impl ResourceSlices {
    pub fn new(id: ResourceId, size: usize) -> Self {
        Self { id, size }
    }
}

impl From<(ResourceId, usize)> for ResourceSlices {
    fn from(value: (ResourceId, usize)) -> Self {
        Self {
            id: value.0,
            size: value.1,
        }
    }
}

#[derive(Debug, Clone)] //TODO clone? reset idx
pub struct SlicesIter {
    pub step: usize,//TODO pub
    pub resource_ids_sizes: Vec<ResourceSlices>,
    outer_idx: usize, // which (ResourceId, len) we're on
    inner_idx: usize, // position within the current len
}

impl SlicesIter {
    pub fn new(step: usize, resource_ids_sizes: Vec<(ResourceId, usize)>) -> Self {
        let resource_ids_sizes = resource_ids_sizes
            .into_iter()
            .map(ResourceSlices::from)
            .collect();
        Self::new_with_offsets(step, resource_ids_sizes)
    }

    pub fn new_with_offsets(step: usize, resource_ids_sizes: Vec<ResourceSlices>) -> Self {
        Self {
            step,
            resource_ids_sizes,
            outer_idx: 0,
            inner_idx: 0,
        }
    }
}

impl Iterator for SlicesIter {
    type Item = TensorIdSliceIdx;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Are we out of (id, size) pairs?
            let (rid, size) = match self.resource_ids_sizes.get(self.outer_idx) {
                Some(entry) => (entry.id, entry.size),
                None => return None,
            };

            // If there is still something to yield for the current (rid, size)…
            if self.inner_idx < size {
                let out = TensorIdSliceIdx(
                    rid,
                    (self.inner_idx * self.step) as u32,
                );
                self.inner_idx += 1;
                return Some(out);
            }

            // Otherwise advance to the next resource entry and reset inner index.
            self.outer_idx += 1;
            self.inner_idx = 0;
            // loop to skip possible zero-sized entries
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // exact total remaining = (remaining in current) + sum of rest
        let mut total = 0usize;
        if let Some(entry) = self.resource_ids_sizes.get(self.outer_idx) {
            total += entry.size.saturating_sub(self.inner_idx);
        }
        if self.outer_idx + 1 < self.resource_ids_sizes.len() {
            total += self.resource_ids_sizes[(self.outer_idx + 1)..]
                .iter()
                .map(|entry| entry.size)
                .sum::<usize>();
        }
        (total, Some(total))
    }
}

#[cfg(test)]
mod tests {
    use thoosa_shared::TensorIdSliceIdx;
    // If this file lives at `crate::tasks::slices_iter`, keep the path below.
    // Otherwise, you can also use `use super::*;`
    use super::{ResourceSlices, SlicesIter};

    #[test]
    fn test_basic() {
        let it = SlicesIter::new(3, vec![(1, 4), (2, 3)]);
        let got: Vec<_> = it.collect();

        // Expected:
        // (1, 0), (1, 3), (1, 6), (1, 9),
        // (2, 0), (2, 3), (2, 6)
        let expected = vec![
            TensorIdSliceIdx(1, 0),
            TensorIdSliceIdx(1, 3),
            TensorIdSliceIdx(1, 6),
            TensorIdSliceIdx(1, 9),
            TensorIdSliceIdx(2, 0),
            TensorIdSliceIdx(2, 3),
            TensorIdSliceIdx(2, 6),
        ];
        assert_eq!(got, expected);
    }

    #[test]
    fn test_zero_lengths_and_step_1() {
        let it = SlicesIter::new(1, vec![(10, 0), (11, 1), (12, 2), (13, 0)]);
        let got: Vec<_> = it.collect();
        let expected = vec![
            TensorIdSliceIdx(11, 0),
            TensorIdSliceIdx(12, 0),
            TensorIdSliceIdx(12, 1),
        ];
        assert_eq!(got, expected);
    }

    #[test]
    fn test_offsets() {
        let it = SlicesIter::new_with_offsets(
            2,
            vec![
                ResourceSlices::new(1, 2),
                ResourceSlices::new(2, 1),
            ],
        );
        let got: Vec<_> = it.collect();
        let expected = vec![
            TensorIdSliceIdx(1, 5),
            TensorIdSliceIdx(1, 7),
            TensorIdSliceIdx(2, 3),
        ];
        assert_eq!(got, expected);
    }
}
