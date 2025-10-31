use log::trace;
use tch::{Kind, Tensor};

pub struct BoolTensorIterator<'a> {
    data_ptr: *const u8,
    index: usize,
    numel: usize,
    _marker: std::marker::PhantomData<&'a Tensor>,
}

impl<'a> BoolTensorIterator<'a> {
    fn new(tensor: &'a Tensor) -> Self {
        assert_eq!(Kind::Bool, tensor.kind());
        let numel = tensor.numel() as usize;
        let data_ptr = tensor.data_ptr() as *const u8;
        BoolTensorIterator {
            data_ptr,
            index: 0,
            numel,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a> Iterator for BoolTensorIterator<'a> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.numel {
            None
        } else {
            unsafe {
                //works both with Bool and Uint8
                let bits = *self.data_ptr.add(self.index);
                trace!("BoolTensorIterator::next: bits={bits:#08b}");
                let value = bits != 0;
                self.index += 1;
                Some(value)
            }
        }
    }
}

impl<'a> From<&'a Tensor> for BoolTensorIterator<'a> {
    fn from(t: &'a Tensor) -> Self {
        BoolTensorIterator::new(t)
    }
}
