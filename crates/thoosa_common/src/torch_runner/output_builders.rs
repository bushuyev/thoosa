use crate::tho_tensor::ThoTensor;
use std::fmt::Debug;
use std::hash::Hash;
use log::debug;
use tch::Device;
use tch::Kind;
use thoosa_shared::runner::adjusters::splits::OutputBuilder;
use crate::dims::Dims;
use thoosa_shared::ThoError;
use thoosa_shared::traits::{Named, WithId};

#[allow(dead_code)]
#[derive(Debug)]
pub enum BuildAs {
    Append,
    Sum,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct SingleDimTensorOutputBuilder<Id, D>
where
    Id:Named + Eq + Hash + Clone +Debug + Send + Sync + 'static,
    D: Dims,
{
    output: ThoTensor<D, Id>,
    added_partial_output_len: i64,
    add: BuildAs,
    dim: i64
}

impl<Id, D: Dims> SingleDimTensorOutputBuilder<Id, D>
where
    Id:Named + Eq + Hash + Clone +Debug  + Send  + Sync + 'static,
    D: Dims,

{
    #[allow(dead_code)]
    pub fn new(
        id: <ThoTensor<D, Id> as WithId>::ID,
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

impl<Id, D> OutputBuilder for SingleDimTensorOutputBuilder<Id, D>
where
    D: Dims,
    Id:Named + Eq + Hash + Clone +Debug  + Send  + Sync + 'static,
{
    type O = ThoTensor<D, Id>;

    async fn add_partial_output(&mut self, partial: Self::O) {
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

    async fn build(self: Self) -> Result<Self::O, ThoError> {
        Ok(self.output)
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use crate::tho_tensor::ThoTensor;
    use log::debug;
    use tch::Kind::Float;
    use tch::Tensor;
    use tch::kind::FLOAT_CPU;
    use thoosa_shared::runner::adjusters::splits::OutputBuilder;
    use crate::dims::{AnyMetrics, AnySub, Dims};
    use thoosa_shared::simple_id;
    use thoosa_shared::ThoError;
    use thoosa_shared::traits::SimpleId;
    use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};

    #[derive(Debug, Clone)]
    struct TestDims;
    impl Dims for TestDims {
        type Metrics = AnyMetrics;
    }

    #[tokio::test]
    async fn test_append() -> Result<(), ThoError>{
        let mut builder: Box<SingleDimTensorOutputBuilder<SimpleId, TestDims>> = Box::new(
            SingleDimTensorOutputBuilder::new(simple_id!("test"), 0, &[10], FLOAT_CPU, BuildAs::Append),
        );

        builder.add_partial_output(ThoTensor::new(simple_id!("1"), Tensor::from_slice(&[0, 1, 2]))?).await;
        builder.add_partial_output(ThoTensor::new(simple_id!("1"), Tensor::from_slice(&[3, 4, 5]))?).await;
        builder.add_partial_output(ThoTensor::new(simple_id!("1"), Tensor::from_slice(&[6, 7, 8]))?).await;
        builder.add_partial_output(ThoTensor::new(simple_id!("1"), Tensor::from_slice(&[9]))?).await;

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
    async fn test_sum() -> Result<(), ThoError>{
        let mut builder: Box<SingleDimTensorOutputBuilder<SimpleId, TestDims>> = Box::new(
            SingleDimTensorOutputBuilder::new(simple_id!("test"), 0, &[3], FLOAT_CPU, BuildAs::Sum),
        );

        builder.add_partial_output(ThoTensor::new(simple_id!("1"), Tensor::from_slice(&[0, 1, 2]))?).await;
        builder.add_partial_output(ThoTensor::new(simple_id!("1"), Tensor::from_slice(&[3, 4, 5]))?).await;
        builder.add_partial_output(ThoTensor::new(simple_id!("1"), Tensor::from_slice(&[6, 7, 8]))?).await;

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
