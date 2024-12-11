use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::fs;
use std::hash::Hash;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::distance::{average_euclidean_distance, clc_dist_d2};
use clap::{Args, ValueEnum};
use crossbeam_utils::sync::WaitGroup;
use futures::FutureExt;
use log::{debug, info, trace, warn};
use parking_lot::RwLock;
use regex::Regex;
use tch::Device;
use tch::IndexOp;
use tch::Kind::Float;
use tch::Tensor;
use thoosa_common::loader::{LoadRunner, Loader, SyncWrapper};
use thoosa_common::utils::*;
use thoosa_shared::resource::{Resource, ResourceId};
use thoosa_shared::TensorIdSliceIdx;
use thoosa_shared::ThoosaError;
use thoosa_shared::*;

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


#[derive(Debug)]
pub struct MotifsTensors {
    pub channels_motifs_slices:Box<Tensor>,
    pub channels_motifs_avg_dists:Box<Tensor>,
    pub channels_motifs_mask:Box<Tensor>,
    pub channels_motifs_ids:Box<Tensor>,
}
//should be read-only
unsafe  impl Sync for MotifsTensors {}

impl MotifsTensors {
    pub fn new(channels_motifs_slices: Box<Tensor>, channels_motifs_avg_dists: Box<Tensor>, channels_motifs_ids: Box<Tensor>) -> Self {
        Self {
            channels_motifs_slices,
            channels_motifs_mask: Box::new(channels_motifs_avg_dists.ge(0.)),//avg dists padded with f32::MIN
            channels_motifs_avg_dists,
            channels_motifs_ids
        }
    }
}

/**
* returns tensor channels x max motifs in channels x first slice in motifs group. + channels x motif ids + dists
* motifs order is random because of iterating over hashmap entries but always consistent between tensor and vec
*/
pub fn motifs_to_tensor(
    resource_id_motif_slices: HashMap<Resource, Vec<MotifSlice>>, loader: &Arc<RwLock< LoadRunner<Box<Tensor>>>>, window: i64, channels: i64, device: Device,
    area: u64
) -> Result<MotifsTensors, ThoosaError> {

    debug!("motifs_to_tensor: resource_id_motif_slices.len={} device={device:?} window={window}", resource_id_motif_slices.len());
    trace!("motifs_to_tensor: resource_id_motif_slices={resource_id_motif_slices:?}");

    assert!(resource_id_motif_slices.len() > 0, "resource_id_motif_slices is empty in motifs_to_tensor");

    let channels_motifs_slices:Arc<RwLock<HashMap<u16, HashMap<usize, Vec<SyncWrapper<Box<Tensor>>>>>>> = Arc::new(RwLock::new(HashMap::new()));

    let channels_motifs_slices_arc = Arc::clone(&channels_motifs_slices);

    let wg = WaitGroup::new();
    let wg_1 = wg.clone();


    loader.write().exec(move |loader, _| {
        async move {

            for (resource, motif_slices) in resource_id_motif_slices {

                let t =   Loader::<Box<Tensor>>::load_future(Arc::clone(&loader), resource, area).await;
                debug!("motifs_to_tensor: eeg_t.device={:?}", t.device());

                for motif_slice in motif_slices {

                    let channel_t = t.get(motif_slice.channel as i64);
                    let channel_t = channel_t.view([1, -1]);

                    let motif_slice_t = channel_t.narrow(1, motif_slice.slice_idx, window);
                    // let motif_slice_t = t.i((motif_slice.channel, motif_slice.slice_idx..motif_slice.slice_idx+window));
                    // let motif_slice_t = t.copy((channel_t, motif_slice.slice_idx..motif_slice.slice_idx+window));

                    let mut write_guard = channels_motifs_slices_arc.write();
                    let  channels_motifs = write_guard.entry(motif_slice.channel).or_insert_with(||HashMap::new());
                    let  motif_slices = channels_motifs.entry(motif_slice.motif_id).or_insert_with(||Vec::new());


                    motif_slices.push(SyncWrapper::new_tb(Box::new(motif_slice_t)));
                    // motifs_slices_map_arc.write().insert(motif_slice.motif_id(), motif_slice);
                }
            }

            drop(wg_1);

        }.boxed()
    });

    wg.wait();


    debug!("motifs_to_tensor: channels_motifs_slices.len={:?}", channels_motifs_slices.read().len());
    trace!("motifs_to_tensor: channels_motifs_slices={:?}", channels_motifs_slices);

    let max_motifs_num = channels_motifs_slices.read().values().map(|motif_slices| motif_slices.len()).max().expect("No motifs") as i64;

    let mut channel_motif_ids:Vec<Vec<usize>> = (0..channels).into_iter().map(|_|Vec::with_capacity(max_motifs_num as usize)).collect();
    let zeros_t = Tensor::zeros(window, (Float, device));

    let mut channels_motifs_slices_t = Vec::with_capacity(channels as usize);
    let mut channels_motifs_avg_dists_t = Vec::with_capacity(channels as usize);



    for channel in 0..channels {
        let motif_slices = channels_motifs_slices.write().remove(&(channel as u16)).unwrap_or(HashMap::new());
        let mut motifs_avg_dists = Vec::with_capacity(motif_slices.len());


        let mut motifs = motif_slices.into_iter().map(|(motif_id, slices)| {
            let slices_t = Tensor::vstack(slices.into_iter().map(|s| *s.shallow_clone()).collect::<Vec<Tensor>>().as_slice());
            debug!("motifs_to_tensor: motif_id={motif_id} slices_t={:?}", slices_t.size());
            trace!("motifs_to_tensor: slices_t=\n{slices_t}");

            let motif_avg_dist = average_euclidean_distance(&slices_t);

            trace!("motifs_to_tensor: motif_avg_dist={motif_avg_dist}");
            motifs_avg_dists.push(motif_avg_dist);
            channel_motif_ids[channel as usize].push(motif_id);

            slices_t.mean_dim(0, false, Float).copy()
        }).collect::<Vec<Tensor>>();


        //make all motifs same length
        if (motifs.len() as i64) < max_motifs_num {
            debug!("motifs_to_tensor: padding channel={channel} {} to {max_motifs_num} ", motifs.len());

            motifs.extend((0..max_motifs_num - motifs.len() as i64).map(|_| zeros_t.shallow_clone()));
            motifs_avg_dists.extend(vec![f32::MIN; max_motifs_num as usize - motifs_avg_dists.len()]);

            let channel_motif_ids_len = channel_motif_ids[channel as usize].len();
            channel_motif_ids[channel as usize].extend(vec![0; max_motifs_num as usize - channel_motif_ids_len]);
        }
        channels_motifs_avg_dists_t.push(Tensor::from_slice(motifs_avg_dists.as_slice()));
        channels_motifs_slices_t.push(Tensor::vstack(motifs.as_slice()).view((max_motifs_num, window)))
    }



    debug!("motifs_to_tensor: channels_motifs_avg_dists_t={channels_motifs_avg_dists_t:?}");
    debug!("motifs_to_tensor: channel_motif_ids={channel_motif_ids:?}");

    Ok(MotifsTensors::new(
        Box::new(Tensor::vstack(
            channels_motifs_slices_t.as_slice()
        ).view((channels, max_motifs_num, -1))),

        Box::new(Tensor::vstack(
             channels_motifs_avg_dists_t.as_slice()
         ).view((channels, max_motifs_num)) + 1e-2), //+ 1e-2 if motif avg is 0 (in tests) there will be no matches with tensors where dists close to 0 but not equal

        Box::new(Tensor::from_slice(
            channel_motif_ids
                .into_iter()
                .flat_map(|channel_motif_ids|channel_motif_ids.into_iter())
                .map(|v|v as i32).collect::<Vec<_>>()
                .as_slice()
        ).view((channels, -1)))
    ))
}

pub fn write_motifs_slices<F>(base_path:&str, output_dir:&str, eeg_id_fragments_map: HashMap<usize, Vec<MotifSlice>>, eegs:Vec<String>, read_eeg_to_tensor:F, window: i64) -> Result<(), ThoosaError>
    where F:Fn(&str, &str, Device) -> Result<Tensor, ThoosaError> + Sync {

    debug!("write_motifs_slices");
    let output_dir = Path::new(output_dir);
    fs::create_dir_all(output_dir)?;

    let total:usize = eeg_id_fragments_map.values().map(|fragments|fragments.len()).sum();
    debug!("write_motifs_slices: total={}", total);

    let progress = AtomicU64::new(0);

    for (eeg_idx, channels_fragments) in eeg_id_fragments_map {

        let eeg_id = eegs[eeg_idx].clone();
        let eeg_t = (read_eeg_to_tensor)(base_path, eeg_id.as_str(), Device::Cpu)?;

        for fragment in channels_fragments {
            let channel_path = output_dir.join(fragment.channel.to_string());
            let motif_path = channel_path.join(fragment.motif_id.to_string());
            fs::create_dir_all(motif_path.clone())?;

            let channel_t = eeg_t.get(fragment.channel as i64);
            let channel_t = channel_t.view([1, -1]);

            let fragment_t = channel_t.f_slice(1, Some(fragment.slice_idx), Some(fragment.slice_idx + window), 1)?;

            fragment_t.save(motif_path.join(format!("{}_{}.pt", eeg_id, fragment.slice_idx)))?;

            progress.fetch_add(1, Ordering::Relaxed);
            info!("progress: {} out of {} total", progress.load(Ordering::Relaxed), total);

        }

    }

    Ok(())
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum MotifsBy {
    MotifId(usize),
    RunId(usize)
}

impl From<&str> for MotifsBy {
    fn from(v: &str) -> MotifsBy {
        if v.starts_with("motif_id:"){
            MotifsBy::MotifId(v[v.find(":").expect(": not found")+1..].parse().expect("can't parse {v} to motif_id"))
        } else if v.starts_with("run_id:") {
            MotifsBy::RunId(v[v.find(":").expect(": not found")+1..].parse().expect("can't parse {v} to run_id"))
        } else {
            panic!("unexpected parameter {v}, should be motif:id:nnn or run_id:nnn")
        }
    }
}

#[derive(Debug, Args, Clone)]
pub struct PlotMotifSlicesParams {

    #[arg(long, value_enum)]
    motifs_by: MotifsBy,

}

impl PlotMotifSlicesParams {

    pub fn motifs_by(&self) -> MotifsBy {
        self.motifs_by
    }

    pub fn new(motifs_by: MotifsBy) -> Self {
        Self { motifs_by }
    }
}



pub fn load_motif_slices (channel: i64, widow: i64, slices: Vec<TensorIdSliceIdx>, resources_map: HashMap<ResourceId, Resource>, loader: &Arc<RwLock< LoadRunner<Box<Tensor>>>>) -> Vec<(ResourceId, Vec<Box<Tensor>>)> {

    debug!("load_motif_slices: channel={}, widow={}, slices={}", channel, widow, slices.len());

    let tensors_ids_slices = slices.iter().fold(HashMap::new(), |mut a, slice| {
        a
            .entry(slice.0.clone())//TODO clone
            .or_insert_with(|| Vec::with_capacity(10))
            .push(slice.1);
        a
    });

    debug!("load_motif_slices: tensors_ids_slices.len={}", tensors_ids_slices.len());

    let mut all_slices:Arc<RwLock<Vec<(ResourceId, Vec<Box<SyncWrapper<Tensor>>>)>>> = Arc::new(RwLock::new(vec![]));
    let wg = WaitGroup::new();


    tensors_ids_slices.keys().collect::<Vec<_>>().chunks(100).for_each(|tensor_keys|{//TODO remove chunks?

        debug!("load_motif_slices: tensor_keys={:?}", tensor_keys);

        tensor_keys
            .iter()
            .map(|v|resources_map.get(*v).expect(format!("no resource {v}").as_str()).clone())//TODO clone + &Resource to load_future
            .for_each(|resource| {

            let slices = tensors_ids_slices.get(&resource.id()).expect(format!("no {resource:?}").as_str()).clone();

            let all_slices = Arc::clone(&all_slices);
            let wg = wg.clone();
            loader.write().exec(move |loader, _keys_purge| {//TODO purge
                async move {
                    let tensor_idx_id = resource.id();
                    let t = LoadRunner::<Box<Tensor>>::load_future(loader, resource, 0).await;

                    let take = t.size()[1] / widow;
                    debug!("load_motif_slices: tensor_idx={tensor_idx_id} slices.len={} t.size={:?} take={}", slices.len(), t.size(),  take  );

                    all_slices.write().push(
                        (tensor_idx_id,
                         slices.into_iter().filter(|slice_idx| {
                             let slice_ok = t.size()[1] >= (*slice_idx) as i64 + widow;

                             if !slice_ok {
                                 warn!("slice doesn't fit into tensor: tensor_idx_id={tensor_idx_id}, t.size()={:?}, *slice_idx={:?}, widow={:?}", t.size(), slice_idx, widow)
                             }

                             slice_ok
                         }).map(|slice_idx| Box::new(
                             SyncWrapper::new_t(t.i((channel, slice_idx as i64..(slice_idx as i64) + widow )))
                         )).collect::<Vec<Box<SyncWrapper<Tensor>>>>()
                        )
                    );
                    drop(wg);
                }.boxed()
            });
        });

    });
    wg.wait();

    debug!("load_motif_slices: loaded slices={}",  slices.len());

    let t= all_slices
        .write()
        .drain(..)
        .map(
            |v|
                (
                    v.0,
                    v.1.into_iter().map(|b|Box::new(b.shallow_clone())).collect()
                )
        )
        .collect();
    t
}


#[derive(Debug, Args, Clone)]
pub struct MotifsToTensorParams {

    #[arg(long)]
    run_id: usize,

    #[arg(long, default_value_t = 1000)]
    window: i64,
}

impl MotifsToTensorParams {

    pub fn window(&self) -> i64 {
        self.window
    }
    pub fn run_id(&self) -> usize {
        self.run_id
    }
}



#[cfg(test)]
mod motifs_tests {
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::ops::Index;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::UNIX_EPOCH;


    use tch::{Device, IndexOp, Tensor};

    use log::debug;
    use parking_lot::RwLock;
    use tch::kind::FLOAT_CPU;
    use tch::Kind::Float;
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::utils::*;
    use thoosa_shared::resource::{Resource, ResourceId};
    use thoosa_shared::MotifSlice;
    use thoosa_shared::TensorIdSliceIdx;
    use thoosa_shared::{cdt_for_dir, ThoosaError};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::init_test_logging();
    }


    #[test]
    fn test_motifs_to_tensor() -> Result<(), ThoosaError>{
        let dir = format!("/tmp/test_{}", std::time::SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs());
        debug!("dir={}", dir);

        let mut eeg_id_fragments_map: HashMap<Resource, Vec<MotifSlice>> = HashMap::new();

        eeg_id_fragments_map.insert(Resource::new(1, 1, "111".into(), 0), vec![
            MotifSlice::new(10, 100, 0),
            MotifSlice::new(11, 301, 1),
            MotifSlice::new(20, 500, 0),
            MotifSlice::new(21, 701, 1)
        ]);

        eeg_id_fragments_map.insert(Resource::new(2, 2, "222".into(), 0), vec![
            MotifSlice::new(10, 200, 0),
            MotifSlice::new(11, 401, 1),
            MotifSlice::new(20, 600, 0),
            MotifSlice::new(21, 801, 1)
        ]);

        fn test_eeg_to_t(eeg_id:&Resource) -> Box<Tensor> {
            let data = (0..1000).map(|i| i as f64).collect::<Vec<f64>>();
            match eeg_id.file_stem() {
                "111"=>Box::new(Tensor::vstack(&[
                    Tensor::from_slice(data.as_slice()),//ch 0
                    Tensor::from_slice(data.as_slice()) //ch 1
                ])),

                "222"=>Box::new(Tensor::vstack(&[
                    Tensor::from_slice(data.as_slice()),//ch 0
                    Tensor::from_slice(data.as_slice()) //ch 1
                ])),

                _=>panic!("eeg_id={:?}", eeg_id)

            }
        }

        let executor_rc = Arc::new(parking_lot::RwLock::new(LoadRunner::new(
            move |key: &Resource| {
                SyncWrapper::new_tb(
                    test_eeg_to_t(key)
                )
            }, 4, 100,
        )));


        let motifs_tensors = super::motifs_to_tensor(
            eeg_id_fragments_map,
            &executor_rc,
            10, 2,
            Device::Cpu,
            1
        )?;

        debug!("slices_t={}", motifs_tensors.channels_motifs_slices);
        debug!("avg_dists_t={}", motifs_tensors.channels_motifs_avg_dists);
        debug!("motif_ids={:?}", motifs_tensors.channels_motifs_ids);

        //16 = 7+9
        let motif_ids = from::<Vec<Vec<i64>>>(&motifs_tensors.channels_motifs_ids).into_iter()
            .flat_map(|v| v.into_iter().map(|v|v as usize))
            .collect::<Vec<usize>>();

        let mut motif_ids_sorted = motif_ids.clone();
        motif_ids_sorted.sort();

        assert_eq!(
            vec![10, 11, 20, 21],
            motif_ids_sorted
        );
        //10 motif slices: 100-110, 200-210
        let motif_10_test = Tensor::stack(&[
            Tensor::from_slice((100..110).map(|v|v as f32).collect::<Vec<_>>().as_slice()),
            Tensor::from_slice((200..210).map(|v|v as f32).collect::<Vec<_>>().as_slice())
            ], 0
        ).mean_dim(0, false, Float);

        debug!("motif_10_test={motif_10_test}");
        let motif_10_idx = motif_ids.iter().enumerate().find(|(i, v)|v == &&10).map(|v|v.0).expect("motif 10");

        let motif_10 = motifs_tensors.channels_motifs_slices.i((0, motif_10_idx as i64));

        debug!("motif_10={motif_10}");

        assert_eq!(motif_10_test, motif_10);

        //TODO other motifs

        Ok(())
    }

    #[test]
    fn test_write_motifs_slices() -> Result<(), ThoosaError>{

        let dir = format!("/tmp/test_write_motifs_slices/{}", cdt_for_dir());
        fs::create_dir_all(&dir)?;


        let mut map: HashMap<usize, Vec<MotifSlice>> = HashMap::new();

        let channel_idx = 1;
        let motif_group = 0;
        let slise_idx = 1;
        map.insert(0usize, vec![MotifSlice::new(motif_group, slise_idx, channel_idx)]);

        fn test_eeg_to_t(_base_dir:&str, _eeg_id:&str, _:Device) -> Result<Tensor, ThoosaError> {
            let t = Tensor::vstack(&[
                Tensor::from_slice(&[0., 1.,  2.,  3.]),
                Tensor::from_slice(&[0.1, 10.1, 20.1, 30.1])
            ]);
            Ok(t)
        }

        super::write_motifs_slices(
            "../../data",
            dir.as_str(),
            map,
            vec!["568657".into()],
            test_eeg_to_t,
            2
        )?;

        let loaded_slice = Tensor::load(Path::new(&dir).join(channel_idx.to_string()).join(motif_group.to_string()).join(format!("568657_{}.pt", slise_idx)))?;

        assert!(Tensor::from_slice(&[10.1, 20.1]).view([1, -1]).equal(&loaded_slice));

        Ok(())
    }


    #[test]
    fn test_load_motif_slices()-> Result<(), ThoosaError>{

        fn test_eeg_to_t(_eeg_id:&Resource) -> SyncWrapper<Box<Tensor>> {
            let t = Tensor::vstack(&[
                Tensor::from_slice(&[0., 1.,  2.,  3.]), //ch 0
                Tensor::from_slice(&[0.1, 10.1, 20.1, 30.1])  //ch 1
            ]);

            SyncWrapper::new(Box::new(t),|t|Box::new(t.shallow_clone()))
        }

        let executor = LoadRunner::new(test_eeg_to_t, 4, 100);
        let executor_rc = Arc::new(parking_lot::RwLock::new(executor));

        let motif_slices = super::load_motif_slices(
            0,
            2,
            vec![TensorIdSliceIdx(1, 2)],
            HashMap::from([(1, Resource::new(1, 1, "aaa".to_string(), 100))]),
            &executor_rc
        );

        assert_eq!(vec![(1, vec![Box::new(Tensor::from_slice(&[2.0, 3.0]))])], motif_slices);

        debug!("motif_slices={:?}", motif_slices);

        Ok(())
    }

    #[test]
    fn test_load_motif_slices_5000()-> Result<(), ThoosaError>{
        fn test_eeg_to_t(_eeg_id:&Resource) -> SyncWrapper<Box<Tensor>> {

            let t = Tensor::vstack(&[
                Tensor::from_slice((0..16781).map(|v| v as f32).collect::<Vec<f32>>().as_slice()) //ch 0
            ]);

            SyncWrapper::new(Box::new(t),|t|Box::new(t.shallow_clone()))
        }

        let executor = LoadRunner::new(test_eeg_to_t, 4, 100);
        let executor_rc = Arc::new(parking_lot::RwLock::new(executor));

        let motif_slices = super::load_motif_slices(
            0,
            5000,
            vec![TensorIdSliceIdx(123, 2)],
            HashMap::from([(123, Resource::new(123, 1, "aaa".to_string(), 100))]),
            &executor_rc
        );


        assert_eq!(123, motif_slices[0].0);
        assert_eq!(5000, motif_slices[0].1[0].size()[0]);

        Ok(())
    }


    #[test]
    fn test_0(){
        let t = Tensor::from_slice(&[
            2., 4., 1.,
            1., 3., 2.,
            3., 2., 3.,
        ]).view((3,3)).mean_dim(0, true, Float);


        println!("t={t}");
    }

}

