use std::sync::Arc;
use chrono::Utc;
use futures_util::StreamExt;
use log::debug;
use plotters::prelude::BLACK;
use tch::Device::Cpu;
use tch::nn::{ConvConfig, VarStore};
use tch::{IndexOp, Tensor};
use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
use thoosa_common::builder::{Node, SeqBuilder};
use thoosa_common::dims::AnyDims;
use thoosa_common::my_tch::ThoModule;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::{any_len_label_node, flat_label_node, motif_node_2, FixedNumSlicesNetParams, LabelNetParams, LabelNetTypeParams, MotifNetParams};
use thoosa_common::train_nets::LabelNetTypeParams::FixedNumSlices;
use thoosa_common::utils::{pad_to, plot_series};
use thoosa_shared::{gen_motif, MotifKind, ThoosaError};
use thoosa_shared::colors_iter::DistinctColorIter;
use crate::find::plot_motif_slices;
use crate::generators::noise::{generate_motifs_with_avg_distance, resource_tensor};

#[test]
fn test_motif_node() -> Result<(), ThoosaError>{

    let vs = VarStore::new(Cpu);
    let motif_width = 100;

    let node_0_channel_0 = motif_node_2(&MotifNetParams::new(motif_width, 3, 1)).build_tch_module::<AnyDims, AnyDims>(&vs.root())?;
    // let node_1_channel_0 = motif_node_2(&MotifNetParams::new(motif_width, 3, 1)).build_tch_module::<AnyDims, AnyDims>(&vs.root())?;
    // let node_0_channel_1 = motif_node_2(&MotifNetParams::new(motif_width, 3, 1)).build_tch_module::<AnyDims, AnyDims>(&vs.root())?;

    let node_0_channel_0_output = apply_motif_node(motif_width, &node_0_channel_0, &make_batch(777, 4)?, 0);
    let node_0_channel_0_output = apply_motif_node(motif_width, &node_0_channel_0, &make_batch(1111, 4)?, 0);
    let node_0_channel_0_output = apply_motif_node(motif_width, &node_0_channel_0, &make_batch(1000, 4)?, 0);
    // let node_1_channel_0_output = apply_motif_node(motif_width, batch_len, node_1_channel_0, &batch, 0);
    // let node_0_channel_1_output = apply_motif_node(motif_width, batch_len, node_0_channel_1, &batch, 0);


    Ok(())
}

fn make_batch(max_len: usize, batch_len: i64) -> Result<Tensor, ThoosaError> {

    let data_width = max_len;

    let batch_len = batch_len;


    let (resource_a_t, resource_a_intervals) = resource_tensor(data_width, &[vec![(2, 100, MotifKind::Pulse)]], 100)?;
    let (resource_b_t, resource_b_intervals) = resource_tensor(data_width, &[vec![(2, 100, MotifKind::Pulse)]], 100)?;
    let (resource_c_t, resource_c_intervals) = resource_tensor(data_width, &[vec![(2, 100, MotifKind::Pulse)]], 100)?;
    let (resource_d_t, resource_d_intervals) = resource_tensor(data_width, &[vec![(2, 100, MotifKind::Pulse)]], 100)?;

    let batch = Tensor::stack(&[resource_a_t, resource_b_t, resource_c_t, resource_d_t], 0);
    Ok(batch)
}

#[test]
fn test_label_node() -> Result<(), ThoosaError>{

    //[8, 3, 9901]

    let vs = VarStore::new(Cpu);

    let node = any_len_label_node(&LabelNetParams::new(0, 3, 0, 0, FixedNumSlices(FixedNumSlicesNetParams{}))).build_tch_module::<AnyDims, AnyDims>(&vs.root())?;

    let motif_node_output = node.tho_forward(&ThoTensor::new("".into(), Tensor::rand(&[8, 3, 9901], FLOAT_CPU)), true);
    // let motif_node_output = node.tho_forward(&ThoTensor::new("".into(), Tensor::rand(&[10, 3, 123], FLOAT_CPU)), true);

    debug!("motif_node_output={motif_node_output:?}");

    Ok(())
}



fn apply_motif_node(motif_width: u32,  node_0_channel_0: &Box<dyn ThoModule<AnyDims, AnyDims>>, batch: &Tensor, channel: i64) -> Tensor {
    debug!("batch={batch:?}");
    let tensor_single_channel = batch.i((.., channel, ..));

    debug!("tensor_single_channel={tensor_single_channel:?}");

    let motif_node_input = tensor_single_channel.unfold(1, motif_width as i64, 1).reshape([-1, motif_width as i64]);

    debug!("motif_node_input={motif_node_input:?}");

    let motif_node_output = node_0_channel_0.tho_forward(&ThoTensor::new("".into(), motif_node_input), true);

    debug!("motif_node_output={motif_node_output:?}");

    let final_output = motif_node_output.t.view((batch.size()[0], -1));

    debug!("final_output={final_output:?}");

    final_output
}
