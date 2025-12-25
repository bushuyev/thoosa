use std::f64::consts::PI;
use tch::{Device, Kind, Tensor};

#[cfg(test)]
pub fn generate_signal(t: &Tensor, modulator_fr: f64, carrier_fr: f64) -> Tensor {
    let modulator = (t * modulator_fr * std::f64::consts::PI).sin();
    let carrier = (t * 2.0 * std::f64::consts::PI * carrier_fr).sin();
    (1.0 + 0.5 * modulator) * carrier
}

#[cfg(test)]
pub fn complex_conv(signal: &Tensor, wavelet_real: &Tensor, wavelet_imag: &Tensor) -> Tensor {
    assert_eq!(signal.size().len(), 3); //for conv B x C x L

    let w_real = wavelet_real.flip([-1]).unsqueeze(0).unsqueeze(0); // [1, 1, K]
    let w_imag = wavelet_imag.flip([-1]).unsqueeze(0).unsqueeze(0);

    let conv_real = signal.conv1d(&w_real, None::<Tensor>, 1, 1, 1, 1);
    let conv_imag = signal.conv1d(&w_imag, None::<Tensor>, 1, 1, 1, 1);

    (conv_real.pow_tensor_scalar(2.0) + conv_imag.pow_tensor_scalar(2.0))
        .sqrt()
        .squeeze()
}

#[cfg(test)]
pub fn sin_t(t: &Tensor, fr: f64) -> Tensor {
    (t * 2.0 * PI * fr).sin()
}
