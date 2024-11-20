use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::io::Write;
use std::mem;
use std::path::{Path, PathBuf};
use rand::{thread_rng, Rng};
use thoosa_shared::resource::ResourceId;
use thoosa_shared::{TensorSliceIdx, ThoosaError};
use crate::{IdSlice, MotifTrackerStore};

pub struct FileMotifTrackerStore {
    pub run_id: usize,
    pub channel: usize,
    // conn: Mutex<Connection>,
    pub path: PathBuf
}

impl FileMotifTrackerStore {
    pub fn new<P: AsRef<Path> + Sync + Send + Debug>(path: P, run_id: usize, channel: usize) -> Result<FileMotifTrackerStore, ThoosaError> {

        Ok(Self {
            run_id,
            channel,
            // conn: Mutex::new(conn),
            path: path.as_ref().to_path_buf()
        })
    }

}

impl MotifTrackerStore for FileMotifTrackerStore {
    fn save_slices(&mut self, slices: Vec<(TensorSliceIdx<ResourceId>, TensorSliceIdx<ResourceId>, f32)>) -> Result<(), ThoosaError> {
        let rnd:(f32, f32, f32, f32) = thread_rng().gen();

        let rnd:u32 = (rnd.0 * 10000. + rnd.1 * 1000. + rnd.2 * 100. + rnd.3 * 10.) as u32;
        let mut file = File::create(&self.path).expect("can't write");


        let num_bytes = slices.len() * mem::size_of::<(u32, i64)>();

        // Convert the data to a byte slice
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                slices.as_ptr() as *const u8,
                num_bytes,
            )
        };

        // Write the bytes to the file
        file.write_all(bytes).map_err(|e|ThoosaError::new(e.to_string()))
    }

    fn set_motifs(&mut self) -> Result<HashMap<u16, Vec<IdSlice>>, ThoosaError> {
        todo!()
    }
}