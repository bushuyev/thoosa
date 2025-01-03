use std::collections::HashMap;
use log::{debug, info};
use tch::{Kind, Tensor};

pub struct InputProbs {
    labels:Vec<String>,
    inputs_probs:HashMap<String, Vec<f32>>
}

impl InputProbs {
    pub fn new(labels: Vec<String>) -> Self {
        Self { labels, inputs_probs: HashMap::new()}
    }

    pub fn add(&mut self, input:String, probs:Vec<f32>) {
        assert_eq!(self.labels.len(), probs.len());
        assert!(self.inputs_probs.insert(input, probs).is_none());
    }

    pub fn labels(&self) -> &Vec<String> {
        &self.labels
    }

    pub fn inputs_probs(&self) -> &HashMap<String, Vec<f32>> {
        &self.inputs_probs
    }
}


// ----------------------------------------
// Metric utilities
// ----------------------------------------

/// Compute Mean Squared Error between two [n, c] probability tensors.
fn mse_loss(inferred: &Tensor, expected: &Tensor) -> Tensor {
    (inferred - expected).pow_tensor_scalar(2).mean(Kind::Float)
}

/// Compute cross-entropy for "soft" targets.
///    CE = - mean_over_n( sum_over_c( expected[i,c] * log(inferred[i,c]) ) )
/// Make sure `inferred` has no zero entries if your data can be zero.
fn soft_cross_entropy(inferred: &Tensor, expected: &Tensor) -> Tensor {
    let log_inferred = inferred.log();
    // Summation over c => shape [n], then average over n
    - (expected * log_inferred).sum(Kind::Float) / inferred.size()[0] as f64
}

/// Compute argmax-based accuracy:
///   1) argmax along last dimension for both inferred and expected
///   2) compare how many match
fn argmax_accuracy(inferred: &Tensor, expected: &Tensor) -> f64 {
    let pred = inferred.argmax(-1, false);  // shape [n]
    let gold = expected.argmax(-1, false);  // shape [n]
    let correct = pred.eq_tensor(&gold).to_kind(Kind::Float).sum(Kind::Float);
    correct.double_value(&[]) / inferred.size()[0] as f64
}

// ----------------------------------------
// Comparison function
// ----------------------------------------

/// Compare two `InputProbs` (assumed same labels, same keys) via MSE, cross-entropy, accuracy.
///
/// 1) For each key (input) in `expected`, retrieve the corresponding probability vectors
///    in both `expected` and `inferred`.
/// 2) Convert them to two [n, c] Tensors, where n = number of keys and c = number of labels.
/// 3) Compute:
///     - Mean Squared Error (MSE)
///     - Soft Cross-Entropy
///     - Argmax accuracy
pub fn compare_input_probs(expected: &InputProbs, inferred: &InputProbs) -> (f64, f64) {
    // 1) Check labels are identical (in the same order).
    assert!(
        expected.labels == inferred.labels,
        "Labels differ between expected and inferred!"
    );
    let c = expected.labels.len();

    // 2) Collect keys in a consistent order.
    //    (We assume they have exactly the same keys.)
    let mut keys: Vec<_> = expected.inputs_probs.keys().cloned().collect();
    keys.sort(); // Ensure a deterministic order, if desired
    let n = keys.len();
    if n == 0 {
        panic!("No keys to compare!");
    }

    // 3) Build data vectors for each matching key in order.
    let mut expected_data = Vec::with_capacity(n * c);
    let mut inferred_data = Vec::with_capacity(n * c);

    for key in &keys {
        let exp_probs = &expected.inputs_probs[key];
        let inf_probs = &inferred.inputs_probs[key];
        expected_data.extend_from_slice(exp_probs);
        inferred_data.extend_from_slice(inf_probs);
    }

    // 4) Convert to [n, c] Tensors
    let t_expected = Tensor::from_slice(&expected_data).reshape(&[n as i64, c as i64]);
    let t_inferred = Tensor::from_slice(&inferred_data).reshape(&[n as i64, c as i64]);

    // (Optional) clamp small values if zero-prob is not valid for cross entropy:
    // let eps = 1e-9;
    // let t_inferred = t_inferred.clamp_min(eps);

    // 5) Compute metrics
    let mse_val = mse_loss(&t_inferred, &t_expected);
    let mse = mse_val.double_value(&[]);

    let ce_val = soft_cross_entropy(&t_inferred, &t_expected);
    let soft_ce = ce_val.double_value(&[]);

    let acc_val_f = argmax_accuracy(&t_inferred, &t_expected);

    // 6) Print metrics
    info!("Number of compared inputs = {}", n);
    info!("MSE Loss         = {:.6}", mse);
    info!("Soft CrossEntropy= {:.6}", soft_ce);
    info!("Argmax Accuracy  = {:.2}%", acc_val_f * 100.0);

    (mse, soft_ce)
}

#[cfg(test)]
pub mod tests {
    use crate::input_probs::{compare_input_probs, InputProbs};

    #[test]
    fn test_0(){
        // Example: 3 labels
        let labels = vec!["cat".into(), "dog".into(), "mouse".into()];

        let mut exp = InputProbs::new(labels.clone());
        let mut inf = InputProbs::new(labels.clone());

        // Suppose we have 2 inputs: "img1" and "img2"
        // with 3-class probabilities each (cat, dog, mouse).

        // "img1": expected => cat=0, dog=1, mouse=0
        exp.add("img1".into(), vec![0.0, 1.0, 0.0]);
        // "img2": expected => cat=1, dog=0, mouse=0
        exp.add("img2".into(), vec![1.0, 0.0, 0.0]);

        // Inferred probabilities
        inf.add("img1".into(), vec![0.2, 0.7, 0.1]);
        inf.add("img2".into(), vec![0.6, 0.3, 0.1]);

        // Compare
        compare_input_probs(&exp, &inf);
    }

}