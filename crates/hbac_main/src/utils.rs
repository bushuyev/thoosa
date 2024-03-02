use tch::Tensor;
use hbac_shared::HbacError;

pub fn collect_to_vec(t:&Tensor)-> Result<Vec<f64>, HbacError> {
    Ok(t.iter::<f64>()?.collect::<Vec<f64>>())
}