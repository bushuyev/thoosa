use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use log::debug;
use crate::tensor_iter::TensorSlice;
use crate::TensorSliceIdx;




fn group_pairs<T>(pairs: impl Iterator<Item = (T, T)>) -> HashMap<u16, Vec<T>> where T:Hash + Eq {
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



#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use log::debug;
    use crate::TensorSliceIdx;
    use crate::group::group_pairs;

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


        let mut groups = group_pairs(pairs.into_iter());

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
    fn test_it(){
        let mut pairs :Vec<(TensorSliceIdx, TensorSliceIdx)> = vec![];

        for j in (0..100) {
            for i in (0..10_000) {
                pairs.push((TensorSliceIdx((i + j) as usize, i as i64), TensorSliceIdx(i as usize, i as i64)));
            }
        }

        let mut groups = group_pairs(pairs.into_iter());

        println!("len: groups.len={}", groups.len());
    }
}