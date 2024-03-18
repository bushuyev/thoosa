use std::iter::Fuse;

pub enum Repeat {
    First,
    Last,
    Both
}

//https://stackoverflow.com/questions/76218621/rust-iterator-that-repeats-the-first-and-last-element-of-another-iterator
pub struct FirstLastRepeat<I, T> {
    iter: Fuse<I>,
    head: Option<T>,
    tail: Option<T>,
    stash: Option<T>,
}



impl<I, T> FirstLastRepeat<I, T>
    where
        I: Iterator<Item = T>,
{
    pub fn new(iter: I, repeat:Repeat) -> Self {
        let mut iter = iter.fuse();
        let head = iter.next();

        match repeat {
            Repeat::First => todo!(),
            Repeat::Last => Self {
                iter,
                head: None,
                tail: None,
                stash: head
            },
            Repeat::Both => Self {
                iter,
                head,
                tail: None,
                stash: None
            }
        }

    }
}

impl<I, T> Iterator for FirstLastRepeat<I, T>
    where
        I: Iterator<Item = T>,
        T: Clone,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if let v @ Some(_) = self.head.take() {
            self.stash = v.clone();
            return v;
        }
        if let v @ Some(_) = self.tail.take() {
            return v;
        }
        let next = self.stash.take();
        match self.iter.next() {
            v @ Some(_) => self.stash = v,
            None => self.tail = next.clone(),
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use crate::repeat_iter::{FirstLastRepeat, Repeat};

    #[test]
    fn test_both() {
        let iter = [1, 2, 3].into_iter();
        let flr = FirstLastRepeat::new(iter, Repeat::Both).collect::<Vec<i32>>();

        assert_eq!(vec![1, 1, 2, 3, 3], flr);
    }

    #[test]
    fn test_last() {
        let iter = [1, 2, 3].into_iter();
        let flr = FirstLastRepeat::new(iter, Repeat::Last).collect::<Vec<i32>>();

        assert_eq!(vec![1, 2, 3, 3], flr);
    }

}