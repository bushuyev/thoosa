use rayon::prelude::*;

struct CompositeVecsIterator<'a> {
    data: &'a [Vec<i32>],
    chunk_size: usize,
    outer_index: usize,
    inner_index1: usize,
    inner_index2: usize,
}

impl<'a> CompositeVecsIterator<'a> {
    pub fn new(data: &'a [Vec<i32>], chunk_size: usize) -> Self {
        CompositeVecsIterator {
            data,
            chunk_size,
            outer_index: 0,
            inner_index1: 0,
            inner_index2: 0,
        }
    }
}

impl<'a> Iterator for CompositeVecsIterator<'a> {
    type Item = (Vec<i32>, Vec<i32>);

    fn next(&mut self) -> Option<Self::Item> {
        while self.outer_index < self.data.len() {
            let vec1 = &self.data[self.outer_index];

            while self.inner_index1 < vec1.len() {
                let start1 = self.inner_index1;
                let end1 = (start1 + self.chunk_size).min(vec1.len());
                let chunk1 = vec1[start1..end1].to_vec();

                self.inner_index1 = end1;

                // Start iterating over the second vector from the beginning
                self.inner_index2 = 0;

                // Iterate over the remaining vectors
                for j in (self.outer_index + 1)..self.data.len() {
                    let vec2 = &self.data[j];

                    while self.inner_index2 < vec2.len() {
                        let start2 = self.inner_index2;
                        let end2 = (start2 + self.chunk_size).min(vec2.len());
                        let chunk2 = vec2[start2..end2].to_vec();

                        self.inner_index2 = end2;

                        return Some((chunk1.clone(), chunk2));
                    }

                    // Reset inner_index2 for the next iteration over vec2
                    self.inner_index2 = 0;
                }
            }

            // Reset inner_index1 and move to the next outer vector
            self.inner_index1 = 0;
            self.outer_index += 1;
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use crate::iter_test::CompositeVecsIterator;

    #[test]
    pub fn test_it(){
        let data = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]];
        let chunk_size = 2;
        let mut iter = CompositeVecsIterator::new(&data, chunk_size);

        while let Some((chunk1, chunk2)) = iter.next() {
            println!("{:?}", chunk1);
            println!("{:?}", chunk2);
            println!("-----");
        }
    }
}