use std::f32::consts::PI;
use tch::Tensor;
use crate::utils::DataBatch;

struct TestDs {
    sin_y:Tensor,
    x2_y:Tensor,
}

impl TestDs {
    pub fn new() -> Self {

        Self {
            sin_y: Tensor::from_slice((0..100).map(|x| 4. * (PI * (x as f32)/ 20.).sin()).collect::<Vec<f32>>().as_slice()),
            x2_y: Tensor::from_slice((0..100).map(|x| 2. * (x as f32/ 50.).powi(2) - 4.  ).collect::<Vec<f32>>().as_slice())
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

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;
    use log::debug;
    use plotters::prelude::{GREEN, RED};
    use plotters::style::BLUE;
    use sci_rs::signal::filter::design::{DigitalFilter, FilterBandType, FilterOutputType, FilterType, iirfilter_dyn};
    use sci_rs::signal::filter::sosfiltfilt_dyn;
    use tch::{Device, nn, Reduction, Tensor};
    use tch::nn::ConvConfig;
    use crate::{configure_dense, utils};
    use crate::simp::TestDs;
    use crate::utils::DataBatch;

    #[test]
    pub fn plot_ds(){
        let tensors = TestDs::new().train_batch(10);

        debug!("tensors.0.size()={:?}", tensors.0.size());
        debug!("tensors.1.size()={:?}", tensors.1.size());

        let t_x_0 = tensors.0.get(0).get(0); //batch, channels (1), data
        let t_x_1 = tensors.0.get(1).get(0); //batch, channels (1), data

        let t_y_0 = tensors.1.get(0); // batch, data
        let t_y_1 = tensors.1.get(1); // batch, data

        let x_0 =   t_x_0.iter::<f64>().unwrap().enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>();
        let x_1 =   t_x_1.iter::<f64>().unwrap().enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>();

        // debug!("x={:?}", x_1);
        // debug!("y={:?}", t_y);

        //
        utils::plot(
            vec![
                (format!("{:?}", t_y_0.iter::<f64>().unwrap().collect::<Vec<f64>>()), GREEN, x_0),
                (format!("{:?}", t_y_1.iter::<f64>().unwrap().collect::<Vec<f64>>()), RED, x_1),
            ], "data", (640, 480)
        ).unwrap();
    }

    #[test]
    pub fn test_filter() {
        let n = 1000;
        let x = (0..n).map(|x| x as f32).collect::<Vec<_>>();
        let sin_y_5 = (0..n).map(|x|(x as f32, (5.0 * 2.0 * PI * x as f32 / n as f32).sin())).collect::<Vec<(f32, f32)>>();
        let sin_y_20 = (0..n).map(|x|(x as f32, (20.0 * 2.0 * PI * x as f32 / n as f32).sin())).collect::<Vec<(f32, f32)>>();
        let sin_y_5_20 = sin_y_5.iter().zip(sin_y_20.iter()).map(|(y_1, y_2)| (y_1.0, y_1.1 + y_2.1)).collect::<Vec<(f32, f32)>>();

        let filter = iirfilter_dyn::<f32>(
            8,
            vec![1., 10.],
            None,
            None,
            Some(FilterBandType::Bandpass),
            Some(FilterType::Butterworth),
            Some(false),
            Some(FilterOutputType::Sos),
            Some(1000.),
        );
        let DigitalFilter::Sos(sos) = filter else { panic!("Not SOS filter") };
        let filtered: Vec<(f32, f32)> = sosfiltfilt_dyn(
            sin_y_5_20.iter().map(|v|v.1),
            &sos.sos
        )
            .iter()
            .enumerate()
            .map(|(i, v)| (i as f32, *v))
            .collect::<Vec<(f32, f32)>>();

        utils::plot(
            vec![
                // (format!("sin 5"), GREEN, sin_y_5),
                // (format!("sin 20"), RED, sin_y_20),
                (format!("sin 5+20"), RED, sin_y_5_20),
                (format!("filtered"), BLUE, filtered),
            ], "data", (640, 480)
        ).unwrap();
    }

    #[test]
    pub fn main(){

        let device = if tch::utils::has_cuda() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };


        let seq = crate::SeqBuilder::new(100)
            .conv(10, 1, 1, ConvConfig::default())
            .max_pool(10, 1, 1)
            .linear_s(1)
            .func(Tensor::sigmoid)
            .end_seq();

        let vs = nn::VarStore::new(device.clone());

        utils::train(
            &device,
            &TestDs::new(),
            seq.build_tch_module(&vs.root()).unwrap(),
            &vs,
            |pred:&Tensor, target:&Tensor|pred.binary_cross_entropy::<Tensor>(target, None, Reduction::Mean)
        );

        debug!("OK")
    }

    pub fn my_loss(pred:&Tensor, target:&Tensor) -> Tensor {
        pred.binary_cross_entropy::<Tensor>(target, None, Reduction::Mean)
    }
}
