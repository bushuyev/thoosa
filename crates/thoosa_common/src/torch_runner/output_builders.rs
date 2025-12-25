use crate::tho_tensor::ThoTensor;
use std::fmt::Debug;
use log::debug;
use tch::Device;
use tch::Kind;
use thoosa_shared::runner::adjusters::splits::OutputBuilder;
use crate::dims::Dims;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{SimpleId, WithId};

#[allow(dead_code)]
#[derive(Debug)]
pub enum BuildAs {
    Append,
    Sum,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct SingleDimTensorOutputBuilder<D>
where
    D: Dims,
{
    output: ThoTensor<D, SimpleId>,
    added_partial_output_len: i64,
    add: BuildAs,
    dim: i64
}

impl<D: Dims> SingleDimTensorOutputBuilder<D> {
    #[allow(dead_code)]
    pub fn new(
        id: <ThoTensor<D, SimpleId> as WithId>::ID,
        dim: i64,
        size: &[i64],
        options: (Kind, Device),
        add: BuildAs,
    ) -> Self {
        Self {
            output: ThoTensor::zeros(id, size, options),
            added_partial_output_len: 0,
            add,
            dim
        }
    }
}

impl<D> OutputBuilder for SingleDimTensorOutputBuilder<D>
where
    D: Dims,
{
    type O = ThoTensor<D, SimpleId>;

    fn add_partial_output(&mut self, partial: Self::O) {
        let partial_size = partial.size();


        match self.add {
            BuildAs::Append => {
                debug!("SingleDimTensorOutputBuilder::add_partial_output, BuildAs::Append: self.dim={} self.output.size={:?} self.added_partial_output_len={:?} self.output.size={:?} partial_size={partial_size:?}",self.dim,  self.output.size(),  self.added_partial_output_len, self.output.size());

                let mut t = self.output.to_mut();

                t.copy_partial_from_(self.dim, self.added_partial_output_len, partial_size[self.dim as usize], &partial)
            }
            BuildAs::Sum => {
                debug!("SingleDimTensorOutputBuilder::add_partial_output, BuildAs::Sum: self.output.size={:?} partial_size={partial_size:?}", self.output.size());

                let mut t = self.output.to_mut();

                t.add_(&partial).unwrap();
            }
        }

        self.added_partial_output_len += partial_size[self.dim as usize];
    }

    async fn build(self: Self) -> Result<Self::O, ThoosaError> {
        Ok(self.output)
    }
}

#[cfg(test)]
mod tests {
    use crate::tho_tensor::ThoTensor;
    use log::debug;
    use tch::Kind::Float;
    use tch::Tensor;
    use tch::kind::FLOAT_CPU;
    use thoosa_shared::runner::adjusters::splits::OutputBuilder;
    use crate::dims::{AnyMetrics, AnySub, Dims};
    use thoosa_shared::ThoosaError;
    use thoosa_shared::traits::SimpleId;
    use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};

    #[derive(Debug, Clone)]
    struct TestDims;
    impl Dims for TestDims {
        type Metrics = AnyMetrics;
    }

    #[tokio::test]
    async fn test_append() -> Result<(), ThoosaError>{
        let mut builder: Box<SingleDimTensorOutputBuilder<TestDims>> = Box::new(
            SingleDimTensorOutputBuilder::new("test".into(), 0, &[10], FLOAT_CPU, BuildAs::Append),
        );

        builder.add_partial_output(ThoTensor::new("1".into(), Tensor::from_slice(&[0, 1, 2]))?);
        builder.add_partial_output(ThoTensor::new("1".into(), Tensor::from_slice(&[3, 4, 5]))?);
        builder.add_partial_output(ThoTensor::new("1".into(), Tensor::from_slice(&[6, 7, 8]))?);
        builder.add_partial_output(ThoTensor::new("1".into(), Tensor::from_slice(&[9]))?);

        let out = builder.build().await?;

        debug!("out=\n{}", out.t_ref());

        assert!(out.t_ref().allclose(
            &Tensor::from_slice(&[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]).totype(Float),
            0.01,
            0.01,
            false
        ));

        Ok(())
    }

    #[tokio::test]
    async fn test_sum() -> Result<(), ThoosaError>{
        let mut builder: Box<SingleDimTensorOutputBuilder<TestDims>> = Box::new(
            SingleDimTensorOutputBuilder::new("test".into(), 0, &[3], FLOAT_CPU, BuildAs::Sum),
        );

        builder.add_partial_output(ThoTensor::new("1".into(), Tensor::from_slice(&[0, 1, 2]))?);
        builder.add_partial_output(ThoTensor::new("1".into(), Tensor::from_slice(&[3, 4, 5]))?);
        builder.add_partial_output(ThoTensor::new("1".into(), Tensor::from_slice(&[6, 7, 8]))?);

        let out = builder.build().await?;

        debug!("out=\n{}", out.t_ref());

        assert!(out.t_ref().allclose(
            &Tensor::from_slice(&[9., 12., 15.]).totype(Float),
            0.01,
            0.01,
            false
        ));

        Ok(())
    }
}
