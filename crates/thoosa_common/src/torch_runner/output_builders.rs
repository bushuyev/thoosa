use crate::tho_tensor::ThoTensor;
use std::fmt::Debug;
use std::hash::Hash;
use log::debug;
use tch::Device;
use tch::Kind;
use crate::runner::adjusters::splits::OutputBuilder;
use crate::dims::Dims;
use thoosa_shared::ThoError;
use thoosa_shared::CopyToDevice;
use thoosa_shared::traits::{Data, Named, WithId};

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
    dim: i64,
    partial_output_device_idx: Option<usize>,
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
    ) -> Result<Self, ThoError> {
        Ok(Self {
            output: ThoTensor::zeros(id, size, options)?,
            added_partial_output_len: 0,
            add,
            dim,
            partial_output_device_idx: None,
        })
    }

    pub fn copy_partial_outputs_to_device(mut self, device_idx: usize) -> Self {
        self.partial_output_device_idx = Some(device_idx);
        self
    }
}

impl<Id, D> OutputBuilder for SingleDimTensorOutputBuilder<Id, D>
where
    D: Dims,
    Id:Named + Eq + Hash + Clone +Debug  + Send  + Sync + 'static,
{
    type O = ThoTensor<D, Id>;

    async fn add_partial_output(&mut self, partial: Self::O) -> Result<(), ThoError> {
        // Report transfer failures before the caller commits split progress.
        let partial = if let Some(device_idx) = self.partial_output_device_idx {
            if partial.tho_device_idx() == device_idx {
                partial
            } else {
                partial.copy_to_device(device_idx).await?
            }
        } else {
            partial
        };

        let partial_size = partial.size();
        let partial_len = *partial_size.get(self.dim as usize).ok_or_else(|| ThoError::new("partial output dimension is out of bounds"))?;


        match self.add {
            BuildAs::Append => {
                debug!("SingleDimTensorOutputBuilder::add_partial_output, BuildAs::Append: self.dim={} self.output.size={:?} self.added_partial_output_len={:?} self.output.size={:?} partial_size={partial_size:?}",self.dim,  self.output.size(),  self.added_partial_output_len, self.output.size());

                let mut t = self.output.to_mut();

                t.copy_partial_from_(self.dim, self.added_partial_output_len, partial_len, &partial)?;
            }
            BuildAs::Sum => {
                debug!("SingleDimTensorOutputBuilder::add_partial_output, BuildAs::Sum: self.output.size={:?} partial_size={partial_size:?}", self.output.size());
                if self.output.tho_device_idx() != partial.tho_device_idx() {
                    // Addition requires both operands on the destination device, including GPU outputs.
                    let partial = partial.copy_to_device(self.output.tho_device_idx()).await?;
                    self.output.to_mut().add_(&partial)?;
                } else {
                    let mut t = self.output.to_mut();
                    t.add_(&partial)?;
                }
    
            }
        }

        self.added_partial_output_len += partial_len;
        Ok(())
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
    use crate::runner::adjusters::splits::OutputBuilder;
    use crate::dims::{AnyMetrics, Dims};
    use thoosa_shared::debug_id;
    use thoosa_shared::ThoError;
    use thoosa_shared::traits::DebugId;
    use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};

    #[derive(Debug, Clone)]
    struct TestDims;
    impl Dims for TestDims {
        type Metrics = AnyMetrics;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_sum_copies_to_destination_device() -> Result<(), ThoError> {
        use crate::tch_error::tch_result;
        use crate::runner::global_ctx::GlobalCtx;
        use tch::Device::{Cpu, Cuda};
        GlobalCtx::init_devices(2)?;
        let _ctx = crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
        // Exercise both directions; CPU-to-GPU previously attempted a CPU-to-CPU copy.
        for (source, destination) in [(Cpu, Cuda(0)), (Cuda(0), Cpu)] {
            let mut builder: SingleDimTensorOutputBuilder<DebugId, TestDims> = SingleDimTensorOutputBuilder::new(debug_id!("sum"), 0, &[3], (Float, destination), BuildAs::Sum)?;
            for _ in 0..2 {
                let tensor = tch_result(Tensor::f_ones([3], (Float, source)))?;
                builder.add_partial_output(ThoTensor::new(debug_id!("partial"), tensor)?).await?;
            }
            let output = builder.build().await?;
            assert_eq!(output.t_ref().device(), destination);
            let cpu = tch_result(output.t_ref().f_to_device(Cpu))?;
            assert!(tch_result(cpu.f_equal(&tch_result(Tensor::f_from_slice(&[2f32, 2., 2.]))?))?);
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_failed_append_and_sum_report_errors_before_advancing() -> Result<(), ThoError> {
        use crate::tch_error::tch_result;

        // Both narrow/copy and addition failures must be observed by append, not build.
        for mode in [BuildAs::Append, BuildAs::Sum] {
            let mut builder: SingleDimTensorOutputBuilder<DebugId, TestDims> = SingleDimTensorOutputBuilder::new(debug_id!("test"), 0, &[2, 2], FLOAT_CPU, mode)?;
            let invalid = ThoTensor::new(debug_id!("invalid"), tch_result(Tensor::f_ones([2, 3], FLOAT_CPU))?)?;
            assert!(builder.add_partial_output(invalid).await.is_err());
            assert_eq!(builder.added_partial_output_len, 0);

            let valid = ThoTensor::new(debug_id!("valid"), tch_result(Tensor::f_ones([2, 2], FLOAT_CPU))?)?;
            builder.add_partial_output(valid).await?;
            assert_eq!(builder.added_partial_output_len, 2);
            let output = builder.build().await?;
            assert!(tch_result(output.t_ref().f_equal(&tch_result(Tensor::f_ones([2, 2], FLOAT_CPU))?))?);
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_failed_append_range_preserves_existing_output() -> Result<(), ThoError> {
        use crate::tch_error::tch_result;

        let mut builder: SingleDimTensorOutputBuilder<DebugId, TestDims> = SingleDimTensorOutputBuilder::new(debug_id!("test"), 0, &[3], FLOAT_CPU, BuildAs::Append)?;
        builder.add_partial_output(ThoTensor::new(debug_id!("first"), tch_result(Tensor::f_from_slice(&[1., 2.]))?)?).await?;
        // A failed append must not skip the remaining slot or corrupt accepted data.
        let invalid = ThoTensor::new(debug_id!("too long"), tch_result(Tensor::f_from_slice(&[8., 9.]))?)?;
        assert!(builder.add_partial_output(invalid).await.is_err());
        assert_eq!(builder.added_partial_output_len, 2);
        builder.add_partial_output(ThoTensor::new(debug_id!("last"), tch_result(Tensor::f_from_slice(&[3.]))?)?).await?;
        let output = builder.build().await?;
        assert!(tch_result(output.t_ref().f_equal(&tch_result(Tensor::f_from_slice(&[1., 2., 3.]))?))?);
        Ok(())
    }

    #[tokio::test]
    async fn test_append() -> Result<(), ThoError>{
        let mut builder: Box<SingleDimTensorOutputBuilder<DebugId, TestDims>> = Box::new(
            SingleDimTensorOutputBuilder::new(debug_id!("test"), 0, &[10], FLOAT_CPU, BuildAs::Append)?,
        );

        builder.add_partial_output(ThoTensor::new(debug_id!("1"), Tensor::from_slice(&[0, 1, 2]))?).await?;
        builder.add_partial_output(ThoTensor::new(debug_id!("1"), Tensor::from_slice(&[3, 4, 5]))?).await?;
        builder.add_partial_output(ThoTensor::new(debug_id!("1"), Tensor::from_slice(&[6, 7, 8]))?).await?;
        builder.add_partial_output(ThoTensor::new(debug_id!("1"), Tensor::from_slice(&[9]))?).await?;

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
        let mut builder: Box<SingleDimTensorOutputBuilder<DebugId, TestDims>> = Box::new(
            SingleDimTensorOutputBuilder::new(debug_id!("test"), 0, &[3], FLOAT_CPU, BuildAs::Sum)?,
        );

        builder.add_partial_output(ThoTensor::new(debug_id!("1"), Tensor::from_slice(&[0, 1, 2]))?).await?;
        builder.add_partial_output(ThoTensor::new(debug_id!("1"), Tensor::from_slice(&[3, 4, 5]))?).await?;
        builder.add_partial_output(ThoTensor::new(debug_id!("1"), Tensor::from_slice(&[6, 7, 8]))?).await?;

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
