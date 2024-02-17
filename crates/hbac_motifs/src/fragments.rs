use std::collections::{HashMap, HashSet};
use std::fs;
use std::fs::DirEntry;
use std::hash::Hash;
use std::path::Path;
use clap::Args;
use log::debug;
use hbac_common::HbacError;
use crate::tensor_iter::TensorSlice;
use crate::TensorSliceIdx;
use regex::Regex;
use tch::Tensor;
use tch::Device;

#[derive(Debug, Args)]
pub struct FragmentMotifsParams {
    #[arg(long, default_value = "../../data_local/motifs")]
    motifs_dir: String,

    #[arg(long, default_value = "../../data_local/fragments")]
    fragments_dir: String,

    #[arg(long, default_value_t = 1000)]
    window: i64,
}

impl FragmentMotifsParams {
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

pub fn group_motifs<T>(pairs: impl Iterator<Item = (T, T)>) -> HashMap<u16, Vec<T>> where T:Hash + Eq + core::fmt::Debug {
    let mut pairs_groups: HashMap<T, u16> = HashMap::new();
    let mut last_group = 0u16;

    let mut separated_groups:  Vec<(u16, u16)> = vec![];

    for (ln, pair) in pairs.enumerate() {
        let g_0 = pairs_groups.get(&pair.0);
        let g_1 = pairs_groups.get(&pair.1);
        if let Some(g_0_val) = g_0 {
            if g_1.is_some() {
                debug!("should never happen");
                if g_0.ne(&g_1) {
                    separated_groups.push((*g_0_val, *g_1.unwrap()));
                    // panic!("g_0={:?} g_1={:?} ln={} pair={:?}", g_0, g_1, ln, pair);
                }
            } else {
                pairs_groups.insert(pair.1, *g_0_val);
            }
        } else {
            if let Some(g_1_val) = g_1 {
                pairs_groups.insert(pair.0, *g_1_val);
            } else {
                pairs_groups.insert(pair.0, last_group);
                pairs_groups.insert(pair.1, last_group);

                last_group += 1;
            }
        }
    }

    println!("separated_groups={:?}", separated_groups);
    let separated_groups_map = if !separated_groups.is_empty() {
        let map = group_motifs(separated_groups.into_iter()).values().fold(HashMap::new(), |mut h, v| {
            for i in v.iter().skip(1) {
                h.insert(v[0], *i);
            }
            h
        });
        println!("separated_groups group_motifs={:?}", map);
        Some(map)
    } else {
        None
    };

    let mut groups: HashMap<u16, Vec<T>> = HashMap::new();
    for (val, group) in pairs_groups {
        let group= separated_groups_map.as_ref().and_then(|m|m.get(&group)).unwrap_or(&group);
        let mut gr_vec = groups.entry(*group).or_insert_with(|| vec![]);
        gr_vec.push(val);
    }
    groups
}


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

#[derive(Debug, Eq, Hash, PartialEq)]
pub struct MotifSlice {
    motif_group: u16,
    slise_idx: i64,
    channel_idx: u16,
}

impl MotifSlice {

    pub fn new(motif_group: u16, slise_idx: i64, channel_idx: u16) -> Self {
        Self { motif_group, slise_idx, channel_idx }
    }

    #[inline]
    pub fn motif_group(&self) -> u16 {
        self.motif_group
    }

    #[inline]
    pub fn slise_idx(&self) -> i64 {
        self.slise_idx
    }

    #[inline]
    pub fn channel_idx(&self) -> u16 {
        self.channel_idx
    }
}

/**
* returns map tensor_id - (group, tensor slice idx, channel)
*/
pub fn motifs_to_eeg_slices(all_channels_motifs: Vec<(u16, String)>, motif_loader:fn(&str) -> Result<Vec<(TensorSliceIdx, TensorSliceIdx)>, HbacError>) -> Result<HashMap<usize, Vec<MotifSlice>>, HbacError>{

    //motif files (per channel) -> motif groups -> eeg slices
    let mut eeg_motifs_sets: HashMap<usize, HashSet<MotifSlice>> = HashMap::new();

    for channel_motifs in all_channels_motifs {
        debug!("channel: {:?}", channel_motifs.0);
        for group in group_motifs(motif_loader(channel_motifs.1.as_str())?.into_iter()) {
            debug!("group: {:?}", group.0);
            for tensor_slice in group.1 {
                let mut eegs_fragments = eeg_motifs_sets.entry(tensor_slice.0).or_insert_with(|| HashSet::new());

                eegs_fragments.insert(MotifSlice::new(group.0, tensor_slice.1, channel_motifs.0)); //motifs group, slice, channel_idx (column in eeg file)
            }
        }
    }


    let mut eeg_motifs:HashMap<usize, Vec<MotifSlice>> = HashMap::new();
    for (key, slices_set) in eeg_motifs_sets.into_iter(){
        let mut slices_vec:Vec<MotifSlice> = slices_set.into_iter().collect();
        slices_vec.sort_by(|a, b|a.slise_idx().cmp(&b.slise_idx()));
        eeg_motifs.insert(key, slices_vec);
    }
    Ok(eeg_motifs)
}

pub fn write_motifs_slices<F>(base_path:&str, output_dir:&str, eeg_id__fragments_map: HashMap<usize, Vec<MotifSlice>>, eegs:Vec<String>, read_eeg_to_tensor:F, window: i64) -> Result<(), HbacError>
    where F:Fn(&str, &str, Device) -> Result<Tensor, HbacError>{
    let output_dir = Path::new(output_dir);
    fs::create_dir_all(output_dir)?;


    for (eeg_idx, channels_fragments) in eeg_id__fragments_map {
        let eeg_id = eegs[eeg_idx].clone();
        let eeg_t = (read_eeg_to_tensor)(base_path, eeg_id.as_str(), Device::Cpu)?;

        for fragment in channels_fragments {
            let channel_path = output_dir.join(fragment.channel_idx.to_string());
            let motif_path = channel_path.join(fragment.motif_group.to_string());
            fs::create_dir_all(motif_path.clone())?;

            // println!("eeg_t:");
            // eeg_t.print();

            let channel_t = eeg_t.get(fragment.channel_idx as i64);
            let channel_t = channel_t.view([1, -1]);

            // println!("channel_t: {}", fragment.channel_idx);
            // channel_t.print();

            let fragment_t = channel_t.f_slice(1, Some(fragment.slise_idx), Some(fragment.slise_idx+ window), 1)?;


            // println!("fragment_t:");
            // fragment_t.print();

            fragment_t.save(motif_path.join(format!("{}.pt", fragment.slise_idx)));

        }
    }

    Ok(())
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::Path;
    use std::time::UNIX_EPOCH;
    use tch::Kind::Double;
    use tch::{Device, Tensor};
    use hbac_common::HbacError;
    use crate::TensorSliceIdx;
    use crate::fragments::group_motifs;
    use crate::fragments::MotifSlice;
    /*
        a - b
        a - c
        e - f
        b - d
        f - g
        x - y


     group 1: a, b, c, d
     group 2: e, f, g
     group 3: x, y
     */

    #[test]
    fn test_main(){
        let pairs:Vec<(String, String)> = vec![
            ("a".into(), "b".into()),
            ("a".into(), "c".into()),
            ("e".into(), "f".into()),
            ("b".into(), "d".into()),
            ("f".into(), "g".into()),
            ("x".into(), "y".into()),
        ];


        let mut groups = group_motifs(pairs.into_iter());

        println!("groups 0 = ={:?}", groups);
        let group_0 = groups.get_mut(&0).unwrap();
        group_0.sort();
        assert_eq!(&vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()], group_0);

        let group_1 = groups.get_mut(&1).unwrap();
        group_1.sort();
        assert_eq!(&vec!["e".to_string(), "f".to_string(), "g".to_string()], group_1);

        let group_2 = groups.get_mut(&2).unwrap();
        group_2.sort();
        assert_eq!(&vec!["x".to_string(), "y".to_string()], group_2);
    }

    #[test]
    fn test_main_separated(){
        let pairs:Vec<(String, String)> = vec![
            ("l".into(), "m".into()),
            ("a".into(), "x".into()),
            ("a".into(), "y".into()),
            ("b".into(), "w".into()),
            ("b".into(), "v".into()),
            ("a".into(), "b".into()),
        ];


        let mut groups = group_motifs(pairs.into_iter());

        println!("groups 0 = ={:?}", groups);
        assert_eq!(groups.len(), 2);

        let group_0 = groups.get_mut(&1).unwrap();
        group_0.sort();
        assert_eq!(&vec!["a".to_string(), "b".to_string(), "v".to_string(), "w".to_string(), "x".to_string(), "y".to_string()], group_0);

        let group_1 = groups.get_mut(&0).unwrap();
        group_1.sort();
        assert_eq!(&vec!["l".to_string(), "m".to_string()], group_1);

    }


    // #[test]
    fn test_it(){
        let mut pairs :Vec<(TensorSliceIdx, TensorSliceIdx)> = vec![];

        for j in (0..100) {
            for i in (0..10_000) {
                pairs.push((TensorSliceIdx((i + j) as usize, i as i64), TensorSliceIdx(i as usize, i as i64)));
            }
        }

        let mut groups = group_motifs(pairs.into_iter());

        println!("len: groups.len={}", groups.len());
    }

    #[test]
    fn test_motifs_to_eeg_slices_1() -> Result<(), HbacError>{
        fn test_motif_loader(file_name:&str) -> Result<Vec<(TensorSliceIdx, TensorSliceIdx)>, HbacError> {
            assert_eq!(file_name, "aaa");

            Ok(vec![
                (TensorSliceIdx(0, 1000), TensorSliceIdx(1, 2000))
            ])
        }

        let map = super::motifs_to_eeg_slices(vec![(0, "aaa".into())], test_motif_loader)?;

        println!("map={:?}", map);
        //tensor id (eeg) - group, slice, channel
        assert_eq!(map.get(&0).unwrap(), &[MotifSlice::new(0, 1000, 0)]);
        assert_eq!(map.get(&1).unwrap(), &[MotifSlice::new(0, 2000, 0)]);

        Ok(())
    }


    #[test]
    fn test_motifs_to_eeg_slices_1_same() -> Result<(), HbacError>{
        fn test_motif_loader(file_name:&str) -> Result<Vec<(TensorSliceIdx, TensorSliceIdx)>, HbacError> {
            assert_eq!(file_name, "aaa");

            Ok(vec![
                (TensorSliceIdx(0, 1000), TensorSliceIdx(1, 2000)),
                (TensorSliceIdx(0, 1000), TensorSliceIdx(1, 4000))
            ])
        }

        let map = super::motifs_to_eeg_slices(vec![(0, "aaa".into())], test_motif_loader)?;

        println!("map={:?}", map);
        //tensor id (eeg) - group, slice, channel1
        assert_eq!(map.get(&0).unwrap(), &[MotifSlice::new(0, 1000, 0)]);//once
        assert_eq!(map.get(&1).unwrap(), &[MotifSlice::new(0, 2000, 0), MotifSlice::new(0, 4000, 0)]);

        Ok(())
    }


    #[test]
    fn test_motifs_to_eeg_slices_2() -> Result<(), HbacError>{
        fn test_motif_loader(file_name:&str) -> Result<Vec<(TensorSliceIdx, TensorSliceIdx)>, HbacError> {
            assert_eq!(file_name, "aaa");

            Ok(vec![
                (TensorSliceIdx(0, 1000), TensorSliceIdx(1, 2000)),
                (TensorSliceIdx(0, 3000), TensorSliceIdx(1, 2000))
            ])
        }

        let map = super::motifs_to_eeg_slices(vec![(0, "aaa".into())], test_motif_loader)?;

        println!("map={:?}", map);
        //tensor id (eeg) - group, slice, channel
        assert_eq!(map.get(&0).unwrap(), &[MotifSlice::new(0, 1000, 0), MotifSlice::new(0, 3000, 0)]);
        assert_eq!(map.get(&1).unwrap(), &[MotifSlice::new(0, 2000, 0)]);

        Ok(())
    }


    #[test]
    fn test_motifs_to_eeg_slices_two_groups() -> Result<(), HbacError>{
        fn test_motif_loader(file_name:&str) -> Result<Vec<(TensorSliceIdx, TensorSliceIdx)>, HbacError> {
            assert_eq!(file_name, "aaa");

            Ok(vec![
                (TensorSliceIdx(0, 1000), TensorSliceIdx(1, 2000)),
                (TensorSliceIdx(0, 3000), TensorSliceIdx(1, 2000)),
                (TensorSliceIdx(0, 5000), TensorSliceIdx(1, 7000)),
                (TensorSliceIdx(0, 9000), TensorSliceIdx(1, 7000))
            ])
        }

        let map = super::motifs_to_eeg_slices(vec![(0, "aaa".into())], test_motif_loader)?;

        println!("map={:?}", map);
        //tensor id (eeg) - group, slice, channel
        assert_eq!(map.get(&0).unwrap(), &[MotifSlice::new(0, 1000, 0), MotifSlice::new(0, 3000, 0), MotifSlice::new(1, 5000, 0), MotifSlice::new(1, 9000, 0)]);
        assert_eq!(map.get(&1).unwrap(), &[MotifSlice::new(0, 2000, 0), MotifSlice::new(1, 7000, 0)]);


        Ok(())
    }


    #[test]
    fn test_motifs_to_eeg_slices_one_groups_sep() -> Result<(), HbacError>{
        fn test_motif_loader(file_name:&str) -> Result<Vec<(TensorSliceIdx, TensorSliceIdx)>, HbacError> {
            assert_eq!(file_name, "aaa");

            Ok(vec![
                (TensorSliceIdx(0, 1000), TensorSliceIdx(1, 2000)),
                (TensorSliceIdx(0, 3000), TensorSliceIdx(1, 2000)),
                (TensorSliceIdx(0, 5000), TensorSliceIdx(1, 7000)),
                (TensorSliceIdx(0, 9000), TensorSliceIdx(1, 7000)),
                (TensorSliceIdx(1, 2000), TensorSliceIdx(1, 7000)),
            ])
        }

        let map = super::motifs_to_eeg_slices(vec![(0, "aaa".into())], test_motif_loader)?;

        println!("map={:?}", map);
        //tensor id (eeg) - group, slice, channel
        assert_eq!(map.get(&0).unwrap(), &[MotifSlice::new(0, 1000, 0), MotifSlice::new(0, 3000, 0), MotifSlice::new(1, 5000, 0), MotifSlice::new(1, 9000, 0)]);
        assert_eq!(map.get(&1).unwrap(), &[MotifSlice::new(0, 2000, 0), MotifSlice::new(1, 7000, 0)]);


        Ok(())
    }

    #[test]
    fn test_motif_to_iter() -> Result<(), HbacError>{
        let motifs = &super::list_motifs("../../data_local/motifs/2024-02-15T18:04:50+01:00")?;

        assert_eq!(motifs.len(), 20);
        assert_eq!((0..20).collect::<Vec<u16>>(), motifs.into_iter().map(|v|v.0).collect::<Vec<u16>>());


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
}

