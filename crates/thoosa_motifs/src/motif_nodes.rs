use chrono::Utc;
use log::debug;
use plotters::prelude::BLACK;
use tch::Device::Cpu;
use tch::nn::{ConvConfig, VarStore};
use tch::Tensor;
use thoosa_common::builder::{Node, SeqBuilder};
use thoosa_common::dims::AnyDims;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::{motif_node_2, MotifNetParams};
use thoosa_common::utils::plot_series;
use thoosa_shared::{gen_motif, MotifKind, ThoosaError};
use thoosa_shared::colors_iter::DistinctColorIter;
use crate::find::plot_motif_slices;
use crate::generators::noise::generate_motifs_with_avg_distance;

#[test]
fn test_motif_node() -> Result<(), ThoosaError>{


    let motif_width = 100;
    let vs = VarStore::new(Cpu);
    let node = motif_node_2(&MotifNetParams::new(motif_width, 3)).build_tch_module::<AnyDims, AnyDims>(&vs.root())?;

    let motifs_vec = generate_motifs_with_avg_distance(
        gen_motif(motif_width as usize, &MotifKind::Pulse).as_slice(),
        8,
        0.1,
        0.2, //should be lower than in motif where it's 0.2-0.3, TODO move to params
        None,
    ).0;

    let mut color_iter = DistinctColorIter::new();

    // plot_series(
    //     motifs_vec
    //         .iter()
    //         .enumerate()
    //         .map(|(i, t)| {
    //             (
    //                 format!("{i}"),
    //                 color_iter.next().unwrap_or(BLACK),
    //                 t.iter()
    //                     .enumerate()
    //                     .map(|(i, v)| (i as f32, *v))
    //                     .collect::<Vec<(f32, f32)>>(),
    //             )
    //         })
    //         .collect(),
    //     format!("test").as_str(),
    //     (2000, 1000),
    //     &format!("/tmp/plots/test_motif_node_{:?}.png", Utc::now().to_rfc3339()).as_str(),
    // )?;

    let motifs_batch = Tensor::stack(
        motifs_vec
            .into_iter()
            .map(|v| Tensor::from_slice(v.as_slice()))
            .collect::<Vec<_>>().as_slice(),
        0,
    );

    debug!("motifs_batch={motifs_batch:?}");


    let res = node.tho_forward(&ThoTensor::new("".into(), motifs_batch), true);

    debug!("res=\n{}", res.t);


    Ok(())
}
