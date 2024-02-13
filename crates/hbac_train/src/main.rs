use tch::{Device, Kind, Tensor};
use hbac_motiffs::find_motiffs_iter;

mod my_tch;
mod data;
mod builder;
mod utils;
mod simp;
mod version;

fn load_sig_tensor(key:&String) ->Box<Tensor>{
    // Box::new(Tensor::rand(&[2, 1000], (Kind::Float, Device::Cpu)))
    Box::new(data::eeg_to_tensor("../../data", key.as_ref()).unwrap())
}

fn main() {

    find_motiffs_iter(
        load_sig_tensor,
        vec!["1628180742".into(), "1019408895".into()],
        600,
        600,
        0.9
    );
}