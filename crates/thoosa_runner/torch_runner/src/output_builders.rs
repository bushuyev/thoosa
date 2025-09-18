use caching_runner::adjusters::splits::OutputBuilder;
use tch::Device;
use tch::Kind;
use thoosa_common::dims::Dims;
use thoosa_shared::traits::WithId;
use crate::data::ThoTensor2;


#[allow(dead_code)]
#[derive(Debug)]
pub enum BuildAs {
    Append,
    Sum
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct SingleDimTensorOutputBuilder<D>
where
    D:Dims
{
    output: ThoTensor2<D, thoosa_shared::traits::SimpleId>,
    added_partial_output_len:i64,
    add: BuildAs
}

impl <D:Dims> SingleDimTensorOutputBuilder<D> {
    #[allow(dead_code)]
    pub fn new(id:<ThoTensor2<D, thoosa_shared::traits::SimpleId> as WithId>::ID, size:&[i64], options: (Kind, Device), add: BuildAs) -> Self {
        Self {
            output: ThoTensor2::zeros(id, size, options),
            added_partial_output_len:0,
            add
        }
    }
}

impl <D> OutputBuilder for SingleDimTensorOutputBuilder<D>
where
    D:Dims
{
    type O = ThoTensor2<D, thoosa_shared::traits::SimpleId>;

    fn add_partial_output(&mut self, partial: Self::O) {
        let partial_size = partial.size();

        match self.add {
            BuildAs::Append => {
                let mut t = self.output.to_mut();

                t.copy_partial_from_(0, self.added_partial_output_len, partial_size[0], &partial)

            }
            BuildAs::Sum => {
                let mut t = self.output.to_mut();

                t.add_(&partial).unwrap();

            }
        }

        self.added_partial_output_len+= partial_size[0];

    }

    fn build(self: Box<Self>) -> Self::O {
        self.output
    }
}

#[cfg(test)]
mod tests {
    use log::debug;
    use tch::Kind::Float;
    use tch::kind::FLOAT_CPU;
    use tch::Tensor;
    use caching_runner::adjusters::splits::OutputBuilder;
    use thoosa_common::dims::Dims;
    use crate::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
    use crate::data::ThoTensor2;

    #[derive(Debug, Clone)]
    struct TestDims;
    impl Dims for TestDims {}


    #[test]
    fn test_append(){

        let mut builder:Box<SingleDimTensorOutputBuilder<TestDims>> = Box::new(SingleDimTensorOutputBuilder::new("test".into(), &[10], FLOAT_CPU, BuildAs::Append));

        builder.add_partial_output(ThoTensor2::new("1".into(), Tensor::from_slice(&[0,1,2])));
        builder.add_partial_output(ThoTensor2::new("1".into(), Tensor::from_slice(&[3,4,5])));
        builder.add_partial_output(ThoTensor2::new("1".into(), Tensor::from_slice(&[6,7,8])));
        builder.add_partial_output(ThoTensor2::new("1".into(), Tensor::from_slice(&[9])));

        let out = builder.build();

        debug!("out=\n{}", out.t());

        assert!(out.t().allclose(&Tensor::from_slice(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]).totype(Float), 0.01, 0.01, false));
    }

    #[test]
    fn test_sum(){

        let mut builder:Box<SingleDimTensorOutputBuilder<TestDims>> = Box::new(SingleDimTensorOutputBuilder::new("test".into(), &[3], FLOAT_CPU, BuildAs::Sum));

        builder.add_partial_output(ThoTensor2::new("1".into(), Tensor::from_slice(&[0,1,2])));
        builder.add_partial_output(ThoTensor2::new("1".into(), Tensor::from_slice(&[3,4,5])));
        builder.add_partial_output(ThoTensor2::new("1".into(), Tensor::from_slice(&[6,7,8])));

        let out = builder.build();

        debug!("out=\n{}", out.t());

        assert!(out.t().allclose(&Tensor::from_slice(&[9., 12., 15.]).totype(Float), 0.01, 0.01, false));
    }

}
