use tch::Tensor;

use crate::utils::DataBatch;

#[derive(Debug)]
pub struct DummyDs {
    y_0:Tensor,
    y_1:Tensor,
    batch_size: usize
}

impl DummyDs {
    pub fn new(y_0:Tensor, y_1:Tensor) -> Self {

        Self {
            y_0,
            y_1,
            // y_1: Tensor::from_slice((0..1000).map(|x| 2. * (x as f32/ 50.).powi(2) - 4.  ).collect::<Vec<f32>>().as_slice()),
            batch_size: 10
        }
    }
}

impl DataBatch for DummyDs {
    fn train_batch(&mut self) -> (Tensor, Tensor) {
        let (x, y):(Vec<Tensor>, Vec<Tensor>) = (0..self.batch_size).fold((vec![], vec![]), |mut x_y, i| {

            if i % 2 == 0 {
                x_y.0.push(self.y_0.copy());
                x_y.1.push(
                    Tensor::from_slice(&[0f32])
                )

            } else {
                x_y.0.push(self.y_1.copy());
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
            ).view([self.batch_size as i64, -1]),

            Tensor::stack(
                y.as_slice(),
                0,
            ).view([self.batch_size as i64])
        )
    }

    fn valid_batch(&mut self) -> (Tensor, Tensor) {
        self.train_batch()
    }
}

