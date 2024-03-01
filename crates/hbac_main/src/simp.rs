use std::f32::consts::PI;
use tch::Tensor;
use crate::utils::DataBatch;

pub(crate) struct TestDs {
    sin_y:Tensor,
    x2_y:Tensor,
}

impl TestDs {
    pub fn new() -> Self {

        Self {
            sin_y: Tensor::from_slice((0..1000).map(|x| 4. * (PI * (x as f32)/ 20.).sin()).collect::<Vec<f32>>().as_slice()),
            x2_y: Tensor::from_slice((0..1000).map(|x| 2. * (x as f32/ 50.).powi(2) - 4.  ).collect::<Vec<f32>>().as_slice())
        }
    }
}

impl DataBatch for TestDs {
    fn train_batch(&self, batch_size: usize) -> (Tensor, Tensor) {
        let (x, y):(Vec<Tensor>, Vec<Tensor>) = (0..batch_size).fold((vec![], vec![]), |mut x_y, i| {

            if i % 2 == 0 {
                x_y.0.push(self.sin_y.copy());
                x_y.1.push(
                    Tensor::from_slice(&[0f32])
                )

            } else {
                x_y.0.push(self.x2_y.copy());
                x_y.1.push(
                    Tensor::from_slice(&[1f32])
                )
            }

            x_y
        });

        (
            Tensor::stack(
                x.as_slice(),
                0,
            ).view([batch_size as i64, 1i64, 100i64]),
            Tensor::stack(
                y.as_slice(),
                0,
            )
        )
    }

    fn valid(&self, batch_size:Option<usize>) -> (Tensor, Tensor) {
        todo!()
    }
}

