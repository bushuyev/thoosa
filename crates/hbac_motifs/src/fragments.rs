use std::collections::{HashMap, HashSet};
use std::fs;
use std::fs::DirEntry;
use std::hash::Hash;
use clap::Args;
use log::debug;
use hbac_common::HbacError;
use crate::tensor_iter::TensorSlice;
use crate::TensorSliceIdx;
use regex::Regex;


#[derive(Debug, Args)]
pub struct FragmentMotifsParams {
    #[arg(long, default_value = "../../data_local/motifs")]
    motifs_dir: String,

    #[arg(long, default_value = "../../data_local/fragments")]
    fragments_dir: String,
}

impl FragmentMotifsParams {
    pub fn motifs_dir(&self) -> &str {
        &self.motifs_dir
    }
}

pub fn group_motifs<T>(pairs: impl Iterator<Item = (T, T)>) -> HashMap<u16, Vec<T>> where T:Hash + Eq {
    let mut pairs_groups: HashMap<T, u16> = HashMap::new();
    let mut last_group = 0u16;
    for pair in pairs {
        let g_0 = pairs_groups.get(&pair.0);
        let g_1 = pairs_groups.get(&pair.1);
        if let Some(g_0_val) = g_0 {
            if g_1.is_some() {
                debug!("should never happen");
                if g_0.ne(&g_1) {
                    panic!("g_0={:?} g_1={:?}", g_0, g_1);
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

    let mut groups: HashMap<u16, Vec<T>> = HashMap::new();
    for (val, group) in pairs_groups {
        let mut gr_vec = groups.entry(group).or_insert_with(|| vec![]);
        gr_vec.push(val);
    }
    groups
}


pub fn list_motifs(motifs_dir:&str) -> Result<Vec<(u16, String)>, HbacError>{
    let re = Regex::new(r".+_(?<id>[0-9]+).csv$").unwrap();

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
* returns map tensor_id - (group, tensor slice idx, channel)
*/
pub fn motifs_to_eeg_slices(all_channels_motifs: Vec<(u16, String)>, motif_loader:fn(&str) -> Result<Vec<(TensorSliceIdx, TensorSliceIdx)>, HbacError>) -> Result<HashMap<usize, Vec<(u16, i64, u16)>>, HbacError>{

    //motif files (per channel) -> motif groups -> eeg slices
    let mut eeg_motifs_sets: HashMap<usize, HashSet<(u16, i64, u16)>> = HashMap::new();

    for channel_motifs in all_channels_motifs {

        for group in group_motifs(motif_loader(channel_motifs.1.as_str())?.into_iter()) {
            debug!("group: {:?}", group.0);
            for tensor_slice in group.1 {
                let mut eegs_fragments = eeg_motifs_sets.entry(tensor_slice.0).or_insert_with(|| HashSet::new());

                eegs_fragments.insert((group.0, tensor_slice.1, channel_motifs.0)); //motifs group, slice, channel_idx (column in eeg file)
            }
        }
    }

    let mut eeg_motifs:HashMap<usize, Vec<(u16, i64, u16)>> = HashMap::new();
    for (key, slices_set) in eeg_motifs_sets.into_iter(){
        let mut slices_vec:Vec<(u16, i64, u16)> = slices_set.into_iter().collect();
        slices_vec.sort_by(|a, b|a.1.cmp(&b.1));
        eeg_motifs.insert(key, slices_vec);
    }
    Ok(eeg_motifs)
}



#[cfg(test)]
mod tests {
    use hbac_common::HbacError;
    use crate::TensorSliceIdx;
    use crate::fragments::group_motifs;

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
        assert_eq!(map.get(&0).unwrap(), &[(0, 1000, 0)]);
        assert_eq!(map.get(&1).unwrap(), &[(0, 2000, 0)]);

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
        //tensor id (eeg) - group, slice, channel
        assert_eq!(map.get(&0).unwrap(), &[(0, 1000, 0)]);//once
        assert_eq!(map.get(&1).unwrap(), &[(0, 2000, 0), (0, 4000, 0)]);

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
        assert_eq!(map.get(&0).unwrap(), &[(0, 1000, 0), (0, 3000, 0)]);
        assert_eq!(map.get(&1).unwrap(), &[(0, 2000, 0)]);

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
        assert_eq!(map.get(&0).unwrap(), &[(0, 1000, 0), (0, 3000, 0), (1, 5000, 0), (1, 9000, 0)]);
        assert_eq!(map.get(&1).unwrap(), &[(0, 2000, 0), (1, 7000, 0)]);


        Ok(())
    }

    #[test]
    fn test_motif_to_iter() -> Result<(), HbacError>{
        let motifs = &super::list_motifs("../../data_local/motifs/2024-02-15T18:04:50+01:00")?;

        assert_eq!(motifs.len(), 20);
        assert_eq!((0..20).collect::<Vec<u16>>(), motifs.into_iter().map(|v|v.0).collect::<Vec<u16>>());


        Ok(())
    }
}

