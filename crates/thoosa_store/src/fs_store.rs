use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::fs::File;
use std::io::{Read, Write};
use std::mem;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use log::{debug, info};
use parking_lot::Mutex;
use rand::{thread_rng, Rng};
use thoosa_shared::resource::ResourceId;
use thoosa_shared::{group_slices, TensorSliceIdx, ThoosaError};
use crate::{IdSlice, MotifTrackerStore, Store};

type SlicesMatch = (TensorSliceIdx<ResourceId>, TensorSliceIdx<ResourceId>, f32);

#[derive(Debug)]
pub struct FileMotifTrackerStore {
    run_id: usize,
    channel: usize,
    path: PathBuf,
    last_file:u32 //writes to run_id/chunks should be in sequence, no need for atomic
}

impl FileMotifTrackerStore {
    pub fn new<P: AsRef<Path> + Sync + Send + Debug>(path: P, run_id: usize, channel: usize) -> Result<Box<dyn MotifTrackerStore>, ThoosaError> {

        Ok(Box::new(Self {
            run_id,
            channel,
            path: path.as_ref().to_path_buf(),
            last_file: 0
        }))
    }

    fn make_file_path(&mut self, file_id: u32) -> PathBuf {
        self.path.join(format!("{}_{}_{}_slices.dat", self.run_id, self.channel, file_id).as_str())
    }
}

impl MotifTrackerStore for FileMotifTrackerStore {

    fn save_slices(&mut self, slices_pairs: Vec<SlicesMatch>) -> Result<(), ThoosaError> {
        debug!("MotifTrackerStore::save_slices: slices_pairs.len()={}", slices_pairs.len());

        //TODO append, new if size > threshold
        let mut file = File::create(self.make_file_path(self.last_file))?;

        self.last_file+=1;

        let num_bytes = slices_pairs.len() * size_of::<SlicesMatch>();

        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                slices_pairs.as_ptr() as *const u8,
                num_bytes,
            )
        };

        file.write_all(bytes).map_err(|e|ThoosaError::new(e.to_string()))
    }

    fn load_slices_into_groups(&mut self, store: &Arc<Mutex<Box<dyn Store>>>) -> Result<(), ThoosaError> {
        let elem_size = size_of::<SlicesMatch>();
        let mut slice_id = 0;
        let groups:HashMap<u16, HashSet<IdSlice>> = group_slices(

            (0..self.last_file).flat_map(|file_id|{

                let mut file = File::open(self.make_file_path(file_id)).unwrap();
                let metadata = file.metadata().unwrap();
                let file_size = metadata.len() as usize;

                if file_size % elem_size != 0 {
                    // return Err(ThoosaError::new("can't read"))//TODO
                }

                let num_elems = file_size / elem_size;

                let mut slices_matches: Vec<SlicesMatch> = Vec::with_capacity(num_elems);

                unsafe {
                    slices_matches.set_len(num_elems);

                    let data_ptr = slices_matches.as_mut_ptr() as *mut u8;
                    let data_slice = std::slice::from_raw_parts_mut(data_ptr, file_size);

                    file.read_exact(data_slice).unwrap();
                }

                slices_matches.into_iter().map(move |slices|{
                    slice_id+=1;
                    (IdSlice(slice_id, slices.0.0, slices.0.1), IdSlice(slice_id, slices.1.0, slices.1.1))
                })
            })
        );
        info!("inserting {} groups {} slices", groups.len(), groups.values().map(|s|s.len()).sum::<usize>());

        store.lock().save_groups_as_motifs(self.run_id, self.channel as i64, groups)
    }


    fn run_id(&self) -> usize {
        self.run_id
    }

    fn channel(&self) -> usize {
        self.channel
    }

    fn path(&self) -> &PathBuf {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;
    use log::debug;
    use parking_lot::{Mutex, RwLock};
    use thoosa_shared::{cdt_for_dir, TensorSliceIdx, ThoosaError};
    use crate::fs_store::FileMotifTrackerStore;
    use crate::SqlStore;

    #[test]
    fn test_single_part() -> Result<(), ThoosaError>{
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        debug!("test_single_part: test_dir={test_dir:?}");
        fs::create_dir_all(&test_dir)?;


        let mut store = Arc::new(Mutex::new(SqlStore::from_path(test_dir.join("main_store.db3"))?));

        let resource_ids= crate::store_tests::init_test_store(&mut store)?;

        let run_id = store.lock().save_tree_run(None, "test", "keys", HashMap::new())?;

        let mut file_store = FileMotifTrackerStore::new(
            test_dir,
            run_id,
            3
        )?;

        file_store.save_slices(vec![
            (TensorSliceIdx(resource_ids[0], 100), TensorSliceIdx(resource_ids[0], 200), 0.1),
            (TensorSliceIdx(resource_ids[0], 200), TensorSliceIdx(resource_ids[0], 300), 0.2),
            (TensorSliceIdx(resource_ids[0], 100), TensorSliceIdx(resource_ids[0], 700), 0.3),
            (TensorSliceIdx(resource_ids[0], 700), TensorSliceIdx(resource_ids[0], 800), 0.4),
        ])?;

        file_store.load_slices_into_groups(&store)?;

        let resources = store.lock().list_resources_by_ids(&resource_ids)?;
        let map = store.lock().list_motifs_slices(run_id)?;


        let motifs_slices = map.get(&resources[0]).expect("resource 0");
        let mut slise_idxs = motifs_slices.into_iter().map(|s|s.slice_idx).collect::<Vec<_>>();
        slise_idxs.sort();

        assert_eq!(
            vec![100, 200, 300, 700, 800],
            slise_idxs
        );

        Ok(())
    }

    #[test]
    fn test_two_motifs() -> Result<(), ThoosaError>{
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        debug!("test_single_part: test_dir={test_dir:?}");
        fs::create_dir_all(&test_dir)?;


        let mut store = Arc::new(Mutex::new(SqlStore::from_path(test_dir.join("main_store.db3"))?));

        let resource_ids= crate::store_tests::init_test_store(&mut store)?;

        let run_id = store.lock().save_tree_run(None, "test", "keys", HashMap::new())?;

        let mut file_store = FileMotifTrackerStore::new(
            test_dir,
            run_id,
            3
        )?;

        file_store.save_slices(vec![
            (TensorSliceIdx(resource_ids[0], 100), TensorSliceIdx(resource_ids[1], 200), 0.1),
            (TensorSliceIdx(resource_ids[1], 200), TensorSliceIdx(resource_ids[1], 700), 0.3),
            (TensorSliceIdx(resource_ids[0], 300), TensorSliceIdx(resource_ids[1], 400), 0.2),
            (TensorSliceIdx(resource_ids[1], 400), TensorSliceIdx(resource_ids[0], 800), 0.4),
        ])?;

        file_store.load_slices_into_groups(&store)?;

        let resources = store.lock().list_resources_by_ids(&resource_ids)?;
        let map = store.lock().list_motifs_slices(run_id)?;

        let motifs_slices = map.get(&resources[0]).expect("resource 0");
        let resoruce_0_motifs = motifs_slices.iter().map(|ms|ms.motif_id).collect::<HashSet<_>>();
        assert_eq!(resoruce_0_motifs.len(), 2);
        let mut slise_idxs = motifs_slices.into_iter().map(|s|s.slice_idx).collect::<Vec<_>>();
        slise_idxs.sort();

        assert_eq!(
            vec![100, 300, 800],
            slise_idxs
        );

        let motifs_slices = map.get(&resources[1]).expect("resource 0");
        let resoruce_1_motifs = motifs_slices.iter().map(|ms|ms.motif_id).collect::<HashSet<_>>();
        assert_eq!(resoruce_1_motifs.len(), 2);
        let mut slise_idxs = motifs_slices.into_iter().map(|s|s.slice_idx).collect::<Vec<_>>();
        slise_idxs.sort();

        assert_eq!(
            vec![200, 400, 700],
            slise_idxs
        );


        Ok(())
    }

    #[test]
    fn test_single_part_with_resources() -> Result<(), ThoosaError>{
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        debug!("test_single_part: test_dir={test_dir:?}");
        fs::create_dir_all(&test_dir)?;


        let mut store = Arc::new(Mutex::new(SqlStore::from_path(test_dir.join("main_store.db3"))?));

        let resource_ids= crate::store_tests::init_test_store(&mut store)?;

        let run_id = store.lock().save_tree_run(None, "test", "keys", HashMap::new())?;

        let mut file_store = FileMotifTrackerStore::new(
            test_dir,
            run_id,
            3
        )?;

        file_store.save_slices(vec![
            (TensorSliceIdx(resource_ids[0], 100), TensorSliceIdx(resource_ids[1], 200), 0.1),
            (TensorSliceIdx(resource_ids[1], 200), TensorSliceIdx(resource_ids[0], 300), 0.2),
            (TensorSliceIdx(resource_ids[0], 100), TensorSliceIdx(resource_ids[1], 700), 0.3),
            (TensorSliceIdx(resource_ids[1], 700), TensorSliceIdx(resource_ids[0], 800), 0.4),
        ])?;

        file_store.load_slices_into_groups(&store)?;

        let resources = store.lock().list_resources_by_ids(&resource_ids)?;
        let map = store.lock().list_motifs_slices(run_id)?;


        let motifs_slices = map.get(&resources[0]).expect("resource 0");
        let mut slise_idxs_0 = motifs_slices.into_iter().map(|s|s.slice_idx).collect::<Vec<_>>();
        slise_idxs_0.sort();

        assert_eq!(
            vec![100, 300, 800],
            slise_idxs_0
        );

        let motifs_slices = map.get(&resources[1]).expect("resource 0");
        let mut slise_idxs_1 = motifs_slices.into_iter().map(|s|s.slice_idx).collect::<Vec<_>>();
        slise_idxs_1.sort();

        assert_eq!(
            vec![200,700],
            slise_idxs_1
        );

        Ok(())
    }



    #[test]
    fn test_two_parts() -> Result<(), ThoosaError>{
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        debug!("test_single_part: test_dir={test_dir:?}");
        fs::create_dir_all(&test_dir)?;


        let mut store = Arc::new(Mutex::new(SqlStore::from_path(test_dir.join("main_store.db3"))?));

        let resource_ids= crate::store_tests::init_test_store(&mut store)?;

        let run_id = store.lock().save_tree_run(None, "test", "keys", HashMap::new())?;

        let mut file_store = FileMotifTrackerStore::new(
            test_dir,
            run_id,
            3
        )?;

        file_store.save_slices(vec![
            (TensorSliceIdx(resource_ids[0], 100), TensorSliceIdx(resource_ids[0], 200), 0.1),
            (TensorSliceIdx(resource_ids[0], 200), TensorSliceIdx(resource_ids[0], 300), 0.2),
        ])?;

        file_store.save_slices(vec![
            (TensorSliceIdx(resource_ids[0], 100), TensorSliceIdx(resource_ids[0], 700), 0.3),
            (TensorSliceIdx(resource_ids[0], 700), TensorSliceIdx(resource_ids[0], 800), 0.4)
        ])?;

        file_store.load_slices_into_groups(&store)?;

        let resources = store.lock().list_resources_by_ids(&resource_ids)?;
        let map = store.lock().list_motifs_slices(run_id)?;


        let motifs_slices = map.get(&resources[0]).expect("resource 0");
        let mut slise_idxs = motifs_slices.into_iter().map(|s|s.slice_idx).collect::<Vec<_>>();
        slise_idxs.sort();

        assert_eq!(
            vec![100, 200, 300, 700, 800],
            slise_idxs
        );

        Ok(())
    }
}