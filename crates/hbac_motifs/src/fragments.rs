use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::fs::DirEntry;
use std::hash::Hash;
use std::ops::Deref;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use chrono::Utc;
use clap::Args;
use clap::builder::TypedValueParser;
use log::{debug, error, info, warn};
use plotters::prelude::{BLACK, RGBColor};
use rand::thread_rng;
use hbac_shared::HbacError;
use crate::tensor_iter::TensorSlice;
use hbac_shared::TensorSliceIdx;
use regex::Regex;
use tch::{Kind, Tensor};
use tch::Device;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use hbac_common::utils::*;
use hbac_shared::*;
use rayon::prelude::*;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelBridge;
use tch::IndexOp;
use rand::prelude::*;

#[derive(Debug, Args)]
pub struct MotifsToFragmentsParams {
    #[arg(long, default_value = "../../data_local/motifs")]
    motifs_dir: String,

    #[arg(long, default_value = "../../data_local/fragments")]
    fragments_dir: String,

    #[arg(long, default_value_t = 1000)]
    window: i64,
}

impl MotifsToFragmentsParams {
    pub fn motifs_dir(&self) -> &str {
        &self.motifs_dir
    }

    pub fn fragments_dir(&self) -> &str {
        &self.fragments_dir
    }


    pub fn window(&self) -> i64 {
        self.window
    }
}

// pub fn group_motifs<T>(pairs: impl Iterator<Item = (T, T)>) -> HashMap<u16, Vec<T>> where T:Hash + Eq + core::fmt::Debug {
//     let mut pairs_groups: HashMap<T, u16> = HashMap::new();
//     let mut last_group = 0u16;
//
//     let mut separated_groups:  Vec<(u16, u16)> = vec![];
//
//     for (ln, pair) in pairs.enumerate() {
//         let g_0 = pairs_groups.get(&pair.0);
//         let g_1 = pairs_groups.get(&pair.1);
//         if let Some(g_0_val) = g_0 {
//             if g_1.is_some() {
//                 debug!("separated group: g_0={:?} g_1={:?} ln={} pair={:?}", g_0, g_1, ln, pair);
//                 if g_0.ne(&g_1) {
//                     separated_groups.push((*g_0_val, *g_1.unwrap()));
//                 }
//             } else {
//                 pairs_groups.insert(pair.1, *g_0_val);
//             }
//         } else {
//             if let Some(g_1_val) = g_1 {
//                 pairs_groups.insert(pair.0, *g_1_val);
//             } else {
//                 pairs_groups.insert(pair.0, last_group);
//                 pairs_groups.insert(pair.1, last_group);
//
//                 last_group += 1;
//             }
//         }
//     }
//
//     debug!("separated_groups={:?}", separated_groups);
//     let separated_groups_map = if !separated_groups.is_empty() {
//         let map = group_motifs(separated_groups.into_iter()).values().fold(HashMap::new(), |mut h, v| {
//             for i in v.iter().skip(1) {
//                 h.insert(v[0], *i);
//             }
//             h
//         });
//         debug!("separated_groups group_motifs={:?}", map);
//         Some(map)
//     } else {
//         None
//     };
//
//     let mut groups: HashMap<u16, Vec<T>> = HashMap::new();
//     for (val, group) in pairs_groups {
//         let group= separated_groups_map.as_ref().and_then(|m|m.get(&group)).unwrap_or(&group);
//         let mut gr_vec = groups.entry(*group).or_insert_with(|| vec![]);
//         gr_vec.push(val);
//     }
//
//     debug!("group_motifs groups={:?}", groups);
//     groups
// }

/**
* returns Vec<(channel_id, chanel's motifs file)>
*/
pub fn list_motifs(motifs_dir:&str) -> Result<Vec<(u16, String)>, HbacError>{
    let re = Regex::new(r".+_(?<id>[0-9]+).csv$").unwrap();//TODO lazy_static

    let mut motiffs:Vec<(u16, String)> = vec![];
    for p in fs::read_dir(motifs_dir)?.into_iter() {
        let path = p?.path();
        let file_name = path.to_str().expect("file_name");

        if let Some(caps) = re.captures(file_name) {
            debug!("file_name={} id={}", file_name, &caps["id"]);

            motiffs.push((caps["id"].parse::<u16>()?, file_name.to_string()))
        };
    }

    motiffs.sort_by(|a, b| a.0.cmp(&b.0));

    Ok(motiffs)
}

/**
* returns map tensor_id - MotifSlice(motif_group, tensor slice idx, channel)
*/
pub fn motifs_to_eeg_slices(all_channels_motifs: Vec<(u16, String)>, motif_loader:fn(&str) -> Result<Vec<(TensorSliceIdx<String>, TensorSliceIdx<String>)>, HbacError>) -> Result<HashMap<String, Vec<MotifSlice>>, HbacError>{

    //motif files (per channel) -> motif groups -> eeg slices
    let mut tensorid_motifs_sets: HashMap<String, HashSet<MotifSlice>> = HashMap::new();

    for channel_motifs in all_channels_motifs {

        debug!("motifs_to_eeg_slices: channel={:?}", channel_motifs.0);
        let motifs = group_slices(motif_loader(channel_motifs.1.as_str())?.into_iter());

        for group_slices in motifs {

            debug!("motifs_to_eeg_slices: group={:?}", group_slices.0);
            //group by tensor(eeg) id to read from eeg files
            for tensor_slice in group_slices.1 {
                let mut eegs_fragments = tensorid_motifs_sets.entry(tensor_slice.0).or_insert_with(|| HashSet::new());

                eegs_fragments.insert(MotifSlice::new(group_slices.0, tensor_slice.1, channel_motifs.0)); //motifs group, slice, channel_idx (column in eeg file)
            }
        }
    }

    debug!("motifs_to_eeg_slices: eeg_motifs_sets={:?}", tensorid_motifs_sets);

    let mut tensor_motifs:HashMap<String, Vec<MotifSlice>> = HashMap::new();
    for (tensor_id, slices_set) in tensorid_motifs_sets.into_iter(){
        let mut slices_vec:Vec<MotifSlice> = slices_set.into_iter().collect();
        slices_vec.sort_by(|a, b|a.slise_idx().cmp(&b.slise_idx()));
        tensor_motifs.insert(tensor_id, slices_vec);
    }

    debug!("motifs_to_eeg_slices: eeg_motifs={:?}", tensor_motifs);

    Ok(tensor_motifs)
}

/**
* returns tensor cahnnels x max motifs in channels x first slice in motifs group. Fills with Tensor::zeros when motifs in channel < max
*/
pub fn motifs_to_tensor<F>(base_path:&str, eeg_id_fragments_map: HashMap<String, Vec<MotifSlice>>, eegs:Vec<String>, read_eeg_to_tensor:F, window: i64, channels: i64, device: Device) -> Result<(Tensor, Vec<Vec<usize>>), HbacError>
    where F:Fn(&String) -> Box<Tensor> + Sync + Send + 'static {

    let mut channels_motifs_slices:HashMap<u16, HashMap<u16, Vec<Tensor>>> = HashMap::new();

    for (eeg_idx, channels_motifslices) in eeg_id_fragments_map {
        // eeg_id__fragments_map.par_iter().try_for_each(|(eeg_idx, channels_fragments)|{  //thread '<unknown>' has overflowed its stack

        // let eeg_id = eeg_idx; //eegs[eeg_idx].clone();
        let eeg_t = (read_eeg_to_tensor)(&eeg_idx);//TODO use always CPU? no clc

        for fragment in channels_motifslices {

            let channel_t = eeg_t.get(fragment.channel_idx as i64);
            let channel_t = channel_t.view([1, -1]);

            let motifslice_t = channel_t.f_slice(1, Some(fragment.slise_idx), Some(fragment.slise_idx + window), 1)?;

            let  mut channels_motifs = channels_motifs_slices.entry(fragment.channel_idx).or_insert_with(||HashMap::new());
            let  mut groups_slices = channels_motifs.entry(fragment.motif).or_insert_with(||Vec::new());

            groups_slices.push(motifslice_t);
        }
    }

    fn choose_slice(slices:&mut Vec<Tensor>) ->Tensor {

         slices.pop().unwrap()//TODO choose best or blend slices
    }

    let groups_num = channels_motifs_slices.values().map(|groups_slices|groups_slices.len()).max().expect("groups_num") as i64;

    let mut channel_motif_ids:Vec<Vec<usize>> = (0..channels).into_iter().map(|_|Vec::new()).collect();

    let tensor = Tensor::vstack(
         (0..channels).map(|channel| (channel, channels_motifs_slices.remove(&(channel as u16)).expect(format!("no {}", channel).as_str())))
            .map(|(channel, mut motif_slices)| motif_slices.into_iter().map(|(motif_id, mut slices)| {
                channel_motif_ids[channel as usize].push(motif_id as usize);
                choose_slice(&mut slices)
            }).collect::<Vec<Tensor>>()).map(|mut group| {

            if (group.len() as i64) < groups_num {
                group.extend((0..groups_num-group.len() as i64).map(|_|Tensor::zeros((window), (tch::Kind::Float, device))))
            }

            let t = Tensor::vstack(group.as_slice());

            t.view((groups_num, window))
        }).collect::<Vec<Tensor>>().as_slice()
    );

    let tensor = tensor.view((channels, groups_num, -1));
    Ok((tensor, channel_motif_ids))

}

pub fn write_motifs_slices<F>(base_path:&str, output_dir:&str, eeg_id_fragments_map: HashMap<usize, Vec<MotifSlice>>, eegs:Vec<String>, read_eeg_to_tensor:F, window: i64) -> Result<(), HbacError>
    where F:Fn(&str, &str, Device) -> Result<Tensor, HbacError> + Sync {

    debug!("write_motifs_slices");
    let output_dir = Path::new(output_dir);
    fs::create_dir_all(output_dir)?;

    let total:usize = eeg_id_fragments_map.values().map(|fragments|fragments.len()).sum();
    debug!("write_motifs_slices: total={}", total);

    let mut progress = AtomicU64::new(0);

    for (eeg_idx, channels_fragments) in eeg_id_fragments_map {
    // eeg_id__fragments_map.par_iter().try_for_each(|(eeg_idx, channels_fragments)|{  //thread '<unknown>' has overflowed its stack

        let eeg_id = eegs[eeg_idx].clone();
        let eeg_t = (read_eeg_to_tensor)(base_path, eeg_id.as_str(), Device::Cpu)?;

        for fragment in channels_fragments {
            let channel_path = output_dir.join(fragment.channel_idx.to_string());
            let motif_path = channel_path.join(fragment.motif.to_string());
            fs::create_dir_all(motif_path.clone())?;


            // eeg_t.print();

            let channel_t = eeg_t.get(fragment.channel_idx as i64);
            let channel_t = channel_t.view([1, -1]);

            // println!("channel_t: {}", fragment.channel_idx);
            // channel_t.print();

            let fragment_t = channel_t.f_slice(1, Some(fragment.slise_idx), Some(fragment.slise_idx+ window), 1)?;


            // println!("fragment_t:");
            // fragment_t.print();

            fragment_t.save(motif_path.join(format!("{}_{}.pt", eeg_id, fragment.slise_idx)));

            progress.fetch_add(1, Ordering::Relaxed);
            info!("progress: {} out of {} total", progress.load(Ordering::Relaxed), total);

        }

        // Ok::<(), HbacError>(())
    }

    Ok(())
}

#[derive(Debug, Args)]
pub struct PlotMotifSlicesParams {

    #[arg(long)]
    motif_id: usize,

    #[arg(long, default_value = "/tmp/motif_plots")]
    output_dir: String,

}

impl PlotMotifSlicesParams {
    pub fn motif_id(&self) -> usize {
        self.motif_id
    }
    pub fn output_dir(&self) -> &str {
        &self.output_dir
    }
}

pub fn check_slices(slices: &Vec<&Box<Tensor>>, min_dot:f32) {

    slices.iter().enumerate().for_each(|(i, s_0)| {
        slices.iter().skip(i).for_each(|s_1| {
            let dp = dot_norm(1, &s_0.view((1, -1)), &s_1.view((1, -1))).unwrap().double_value(&[0]) as f32;

            if dp < min_dot {
                warn!("dot: {}", dp);
            } else {
                debug!("dot: {}", dp);
            }

        });
    });
}


pub fn load_motif_slices<F>(base_path: &str, channel: i64, widow: i64, slices: Vec<TensorSliceIdx<String>>, eegs: Vec<String>, load_tensor: F, device: Device) -> Vec<(String, Vec<Box<Tensor>>)>
    where F: Fn(&str, &str, Device) -> Result<Tensor, HbacError> + Sync {

    debug!("load_motif_slices: channel={}, widow={}, slices={}", channel, widow, slices.len());

    let tensors_ids_slices = slices.iter().fold(HashMap::new(), |mut a, slice| {
        a
            .entry(slice.0.clone())//TODO clone
            .or_insert_with(|| Vec::with_capacity(10))
            .push(slice.1);
        a
    });

    let mut all_slices:Vec<(String, Vec<Box<Tensor>>)> = vec![];

    tensors_ids_slices.keys().collect::<Vec<_>>().chunks(100).for_each(|tensor_idxs|{

        let slices = tensor_idxs.par_iter().map(|tensor_idx|{
            // let tensor_idx:String = tensor_idx.clone();
            let slices = tensors_ids_slices.get(*tensor_idx).expect(format!("no {}", tensor_idx).as_str());

            // let resource = eegs[tensor_idx].clone();
            let t = load_tensor(base_path, tensor_idx.as_str(), device).unwrap();

            let take = t.size()[1]/widow;
            debug!("load_motif_slices: tensor_idx={} slices.len={} t.size={:?} take={}", tensor_idx, slices.len(), t.size(),  take  );

            //take - to skip last slice which might be not full
            (tensor_idx.to_string(), slices.into_iter().take(take as usize).map(|slice_idx| Box::new(t.i((channel, *slice_idx..*slice_idx + widow)))).collect::<Vec<Box<Tensor>>>())
        }).collect::<Vec<(String, Vec<Box<Tensor>>)>>();


        all_slices.extend(slices);
    });


    debug!("load_motif_slices: loaded slices={}",  slices.len());

    all_slices
}


#[derive(Debug, Args)]
pub struct MotifsToTensorParams {

    #[arg(long)]
    run_id: usize,

    #[arg(long)]
    output_dir: Option<String>,

    #[arg(long, default_value_t = 1000)]
    window: i64,
}

impl MotifsToTensorParams {

    pub fn output_dir(&self) -> Option<&str> {
        self.output_dir.as_ref().map(|x|x.as_str())
    }



    pub fn window(&self) -> i64 {
        self.window
    }
    pub fn run_id(&self) -> usize {
        self.run_id
    }
}
pub fn choose_fragments(params:&MotifsToTensorParams) ->Result<(), HbacError> {

    Ok(())
}

pub fn check_fragments(from: &impl AsRef<std::path::Path>,)->Result<(), HbacError> {
    let tensors:Vec<Tensor> = fs::read_dir(from)?
        .into_iter()
        .filter(|e|e.as_ref().is_ok_and(|e|e.file_type().is_ok_and(|f|f.is_file())))
        .map(|e|Tensor::load(e.unwrap().path()).unwrap()).collect();

    println!("tensors={:?}", tensors.len());

    tensors.iter().enumerate().for_each(|(i_0, t_0)|{
        tensors.iter().enumerate().skip(i_0).for_each(|(i_1, t_1)|{


            let dp = dot_norm(1, &t_0, &t_1).unwrap();

            println!("i_0={:?} i_1={:?} dp={:?}", i_0, i_1, dp.double_value(&[0]));

        });
    });

    Ok(())
}



#[cfg(test)]
mod motifs_tests {
    use tch::IndexOp;
    use std::collections::{HashMap, HashSet};
    use std::path::Path;
    use std::time::UNIX_EPOCH;
    use tch::{Device, Tensor};
    use hbac_shared::HbacError;
    use hbac_shared::TensorSliceIdx;
    use hbac_shared::MotifSlice;
    use rayon::iter::ParallelIterator;
    use rayon::iter::ParallelBridge;
    use regex::Regex;
    use hbac_common::train_nets::ClassTensorsDataBatch;
    use hbac_common::utils::*;



    #[test]
    fn test_motifs_to_eeg_slices_1() -> Result<(), HbacError>{
        fn test_motif_loader(file_name:&str) -> Result<Vec<(TensorSliceIdx<String>, TensorSliceIdx<String>)>, HbacError> {
            assert_eq!(file_name, "aaa");

            Ok(vec![
                (TensorSliceIdx("0".to_string(), 1000), TensorSliceIdx("1".to_string(), 2000))
            ])
        }

        let map = super::motifs_to_eeg_slices(vec![(0, "aaa".into())], test_motif_loader)?;

        println!("map={:?}", map);
        //tensor id (eeg) - group, slice, channel
        assert_eq!(map.get(&"0".to_string()).unwrap(), &[MotifSlice::new(0, 1000, 0)]);
        assert_eq!(map.get(&"1".to_string()).unwrap(), &[MotifSlice::new(0, 2000, 0)]);

        Ok(())
    }


    #[test]
    fn test_motifs_to_eeg_slices_1_same() -> Result<(), HbacError>{
        fn test_motif_loader(file_name:&str) -> Result<Vec<(TensorSliceIdx<String>, TensorSliceIdx<String>)>, HbacError> {
            assert_eq!(file_name, "aaa");

            Ok(vec![
                (TensorSliceIdx(0.to_string(), 1000), TensorSliceIdx(1.to_string(), 2000)),
                (TensorSliceIdx(0.to_string(), 1000), TensorSliceIdx(1.to_string(), 4000))
            ])
        }

        let map = super::motifs_to_eeg_slices(vec![(0, "aaa".into())], test_motif_loader)?;

        println!("map={:?}", map);
        //tensor id (eeg) - group, slice, channel1
        assert_eq!(map.get(&0.to_string()).unwrap(), &[MotifSlice::new(0, 1000, 0)]);//once
        assert_eq!(map.get(&1.to_string()).unwrap(), &[MotifSlice::new(0, 2000, 0), MotifSlice::new(0, 4000, 0)]);

        Ok(())
    }


    #[test]
    fn test_motifs_to_eeg_slices_2() -> Result<(), HbacError>{
        fn test_motif_loader(file_name:&str) -> Result<Vec<(TensorSliceIdx<String>, TensorSliceIdx<String>)>, HbacError> {
            assert_eq!(file_name, "aaa");

            Ok(vec![
                (TensorSliceIdx(0.to_string(), 1000), TensorSliceIdx(1.to_string(), 2000)),
                (TensorSliceIdx(0.to_string(), 3000), TensorSliceIdx(1.to_string(), 2000))
            ])
        }

        let map = super::motifs_to_eeg_slices(vec![(0, "aaa".into())], test_motif_loader)?;

        println!("map={:?}", map);
        //tensor id (eeg) - group, slice, channel
        assert_eq!(map.get(&0.to_string()).unwrap(), &[MotifSlice::new(0, 1000, 0), MotifSlice::new(0, 3000, 0)]);
        assert_eq!(map.get(&1.to_string()).unwrap(), &[MotifSlice::new(0, 2000, 0)]);

        Ok(())
    }


    #[test]
    fn test_motifs_to_eeg_slices_two_groups() -> Result<(), HbacError>{
        fn test_motif_loader(file_name:&str) -> Result<Vec<(TensorSliceIdx<String>, TensorSliceIdx<String>)>, HbacError> {
            assert_eq!(file_name, "aaa");

            Ok(vec![
                (TensorSliceIdx(0.to_string(), 1000), TensorSliceIdx(1.to_string(), 2000)),
                (TensorSliceIdx(0.to_string(), 3000), TensorSliceIdx(1.to_string(), 2000)),
                (TensorSliceIdx(0.to_string(), 5000), TensorSliceIdx(1.to_string(), 7000)),
                (TensorSliceIdx(0.to_string(), 9000), TensorSliceIdx(1.to_string(), 7000))
            ])
        }

        let map = super::motifs_to_eeg_slices(vec![(0, "aaa".into())], test_motif_loader)?;

        println!("map={:?}", map);
        //tensor id (eeg) - group, slice, channel
        assert_eq!(map.get(&0.to_string()).unwrap(), &[MotifSlice::new(0, 1000, 0), MotifSlice::new(0, 3000, 0), MotifSlice::new(1, 5000, 0), MotifSlice::new(1, 9000, 0)]);
        assert_eq!(map.get(&1.to_string()).unwrap(), &[MotifSlice::new(0, 2000, 0), MotifSlice::new(1, 7000, 0)]);


        Ok(())
    }


    #[test]
    fn test_motifs_to_eeg_slices_one_groups_sep() -> Result<(), HbacError>{
        fn test_motif_loader(file_name:&str) -> Result<Vec<(TensorSliceIdx<String>, TensorSliceIdx<String>)>, HbacError> {
            assert_eq!(file_name, "aaa");

            Ok(vec![
                (TensorSliceIdx(0.to_string(), 1000), TensorSliceIdx(1.to_string(), 2000)),
                (TensorSliceIdx(0.to_string(), 3000), TensorSliceIdx(1.to_string(), 2000)),
                (TensorSliceIdx(0.to_string(), 5000), TensorSliceIdx(1.to_string(), 7000)),
                (TensorSliceIdx(0.to_string(), 9000), TensorSliceIdx(1.to_string(), 7000)),
                (TensorSliceIdx(1.to_string(), 2000), TensorSliceIdx(1.to_string(), 7000)),
            ])
        }

        let map = super::motifs_to_eeg_slices(vec![(0, "aaa".into())], test_motif_loader)?;


        assert_eq!(map.len(), 2);

        let mut values:Vec<&Vec<MotifSlice>> = map.values().into_iter().collect();
        values.sort_by(|a, b|b.len().cmp(&a.len()));

        let groups  = values.iter().flat_map(|v|v.iter()).fold(HashSet::new(), |mut a, motif_slice|{
            a.insert(motif_slice.motif);
            a
        });

        assert_eq!(groups.len(), 1);
        let group = groups.into_iter().next().unwrap();

        //tensor id (eeg) - group, slice, channel
        assert_eq!(values.get(0).unwrap().as_slice(), &[
            MotifSlice::new(group, 1000, 0),
            MotifSlice::new(group, 3000, 0),
            MotifSlice::new(group, 5000, 0),
            MotifSlice::new(group, 9000, 0),
        ]);

        assert_eq!(values.get(1).unwrap().as_slice(), &[
            MotifSlice::new(group, 2000, 0),
            MotifSlice::new(group, 7000, 0)
        ]);


        Ok(())
    }

    #[test]
    fn test_motif_to_iter() -> Result<(), HbacError>{
        let motifs = &super::list_motifs("../../data_local/motifs/2024-02-18T21:08:49+01:00")?;

        assert_eq!(motifs.len(), 20);
        assert_eq!((0..20).collect::<Vec<u16>>(), motifs.into_iter().map(|v|v.0).collect::<Vec<u16>>());

        Ok(())
    }

    #[test]
    fn test_motifs_to_tensor() -> Result<(), HbacError>{
        let dir = format!("/tmp/test_{}", std::time::SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs());
        println!("dir={}", dir);

        let mut eeg_id_fragments_map: HashMap<String, Vec<MotifSlice>> = HashMap::new();

        eeg_id_fragments_map.insert("0".to_string(), vec![
            MotifSlice::new(7, 100, 0),
            MotifSlice::new(7, 301, 1),
            MotifSlice::new(9, 500, 0),
            MotifSlice::new(9, 701, 1)
        ]);

        eeg_id_fragments_map.insert("i".to_string(), vec![
            MotifSlice::new(7, 200, 0),
            MotifSlice::new(7, 401, 1),
            MotifSlice::new(9, 600, 0),
            MotifSlice::new(9, 801, 1)
        ]);

        fn test_eeg_to_t(eeg_id:&String) -> Box<Tensor> {
            let data = (0..1000).map(|i| i as f64).collect::<Vec<f64>>();
            match eeg_id.as_str() {
                "111"=>Box::new(Tensor::vstack(&[
                    Tensor::from_slice(data.as_slice()),//ch 0
                    Tensor::from_slice(data.as_slice()) //ch 1
                ])),

                "222"=>Box::new(Tensor::vstack(&[
                    Tensor::from_slice(data.as_slice()),//ch 0
                    Tensor::from_slice(data.as_slice()) //ch 1
                ])),

                _=>panic!("eeg_id={}", eeg_id)

            }
        }

        let (t, motif_ids) = super::motifs_to_tensor(
            "zzz",
            eeg_id_fragments_map,
            vec!["111".into(), "222".into()],
            test_eeg_to_t,
            10, 2,
            Device::Cpu
        )?;

        println!("motif_ids={:?}", motif_ids);

        //16 = 7+9
        assert_eq!(vec![16, 16], motif_ids.into_iter().map(|v|v.into_iter().sum::<usize>()).collect::<Vec<usize>>());

        assert_eq!(from::<Vec<f32>>(&t.get(0).get(0))[0] as i64 % 100, 0);//ch0, group/slice random
        assert_eq!(from::<Vec<f32>>(&t.get(1).get(0))[0] as i64 % 100, 1);//ch1, group/slice random


        Ok(())
    }

    #[test]
    fn test_write_motifs_slices() -> Result<(), HbacError>{
        let dir = format!("/tmp/test_{}", std::time::SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs());
        println!("dir={}", dir);

        let mut map: HashMap<usize, Vec<MotifSlice>> = HashMap::new();

        let channel_idx = 1;
        let motif_group = 0;
        let slise_idx = 1;
        map.insert(0usize, vec![MotifSlice::new(motif_group, slise_idx, channel_idx)]);

        fn test_eeg_to_t(base_dir:&str, eeg_id:&str, _:Device) -> Result<Tensor, HbacError> {
            let t = Tensor::vstack(&[
                Tensor::from_slice(&[0., 1.,  2.,  3.]),
                Tensor::from_slice(&[0.1, 10.1, 20.1, 30.1])
            ]);
            // println!("test_eeg_to_t:");
            // t.print();
            Ok(t)
        }

        super::write_motifs_slices(
            "../../data",
            dir.as_str(),
            map,
            vec!["568657".into()],
            test_eeg_to_t,
            2
        );

        let loaded_slice = Tensor::load(Path::new(&dir).join(channel_idx.to_string()).join(motif_group.to_string()).join(format!("{}.pt", slise_idx)))?;

        // println!("loaded_slice:");
        // loaded_slice.print();

        assert!(Tensor::from_slice(&[10.1, 20.1]).view([1, -1]).equal(&loaded_slice));

        Ok(())
    }


    #[test]
    fn test_load_motif_slices()-> Result<(), HbacError>{

        fn test_eeg_to_t(base_dir:&str, eeg_id:&str, _:Device) -> Result<Tensor, HbacError> {
            let t = Tensor::vstack(&[
                Tensor::from_slice(&[0., 1.,  2.,  3.]), //ch 0
                Tensor::from_slice(&[0.1, 10.1, 20.1, 30.1])  //ch 1
            ]);

            Ok(t)
        }

        let motif_slices = super::load_motif_slices(
            "base",
            0,
            2,
            vec![TensorSliceIdx(0.to_string(), 2)],
            vec!["123".into()],
            test_eeg_to_t,
            Device::Cpu
        );

        assert_eq!(vec![("123".to_string(), vec![Box::new(Tensor::from_slice(&[2.0, 3.0]))])], motif_slices);

        println!("motif_slices={:?}", motif_slices);

        Ok(())
    }

    #[test]
    fn test_check_slices()-> Result<(), HbacError> {

        super::check_slices(
            &vec![
                &Box::new(Tensor::from_slice(&[1., 1., 1.])),
                &Box::new(Tensor::from_slice(&[1., 1., 1.])),
                &Box::new(Tensor::from_slice(&[0., 1., 1.])),
            ],
            0.99
        );

        Ok(())
    }

    #[test]
    fn test_test() -> Result<(), HbacError>{

        let map = HashMap::from([
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (3, "d"),
        ]);

        println!("map={:?}", map);

        map.keys().collect::<Vec<_>>().chunks(10).par_bridge().for_each(|c|{
            println!("c={:?} v={:?}", c, map.get(c[0]));
        });

        // map.into_iter().chunks(5).into_iter().for_each(|v| {
        //     println!("{:?}", v);
        // });

        // let i = map.into_iter().array_chunks::<3>();
        // i.for_each(|c|{
        //     println!("{:?}", c);
        // });
        //
        // if let Some(reminder) = i.into_remainder() {
        //     println!("reminder={:?}", reminder);
        // }

        // let map = (0..100).fold(HashMap::new(), |mut h, v|{
        //     h.insert(v, (0..100).collect::<Vec<usize>>());
        //     h
        // });
        //
        // map.par_iter().try_for_each(|v|{
        //
        //     if *v.0 == 50 {
        //         Err::<(), HbacError>(HbacError::new("XXX".to_string()))
        //     } else {
        //         Ok::<(), HbacError>(())
        //     }
        //
        //
        //
        // })?;

        Ok(())
    }



}

