use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::io::{Read, Write};
use std::mem;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::Mutex;
use rand::{thread_rng, Rng};
use thoosa_shared::resource::ResourceId;
use thoosa_shared::{group_slices, TensorSliceIdx, ThoosaError};
use crate::{IdSlice, MotifTrackerStore, Store};

#[derive(Debug)]
pub struct FileMotifTrackerStore {
    pub run_id: usize,
    pub channel: usize,
    pub path: PathBuf
}

impl FileMotifTrackerStore {
    pub fn new<P: AsRef<Path> + Sync + Send + Debug>(path: P, run_id: usize, channel: usize) -> Result<Box<dyn MotifTrackerStore>, ThoosaError> {

        Ok(Box::new(Self {
            run_id,
            channel,
            path: path.as_ref().to_path_buf()
        }))
    }

}

impl MotifTrackerStore for FileMotifTrackerStore {
    fn save_slices(&mut self, slices: Vec<(TensorSliceIdx<ResourceId>, TensorSliceIdx<ResourceId>, f32)>) -> Result<(), ThoosaError> {
        let rnd:(f32, f32, f32, f32) = thread_rng().gen();

        let rnd:u32 = (rnd.0 * 10000. + rnd.1 * 1000. + rnd.2 * 100. + rnd.3 * 10.) as u32;
        let mut file = File::create(&self.path).expect("can't write");


        let num_bytes = slices.len() * mem::size_of::<(u32, i64)>();

        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                slices.as_ptr() as *const u8,
                num_bytes,
            )
        };

        file.write_all(bytes).map_err(|e|ThoosaError::new(e.to_string()))
    }

    fn build_groups(&mut self, store: &Arc<Mutex<Box<dyn Store>>>) -> Result<(), ThoosaError> {
        let mut file = File::open("output.dat")?;
        let metadata = file.metadata()?;
        let file_size = metadata.len() as usize;
        let elem_size = mem::size_of::<(u32, i64)>();

        if file_size % elem_size != 0 {
            // return Err(io::Error::new(
            //     io::ErrorKind::InvalidData,
            //     "File size is not a multiple of element size",
            // ));
            return Err(ThoosaError::new("can't read"))
        }

        let num_elems = file_size / elem_size;

        let mut data_vec: Vec<(TensorSliceIdx<ResourceId>, TensorSliceIdx<ResourceId>, f32)> = Vec::with_capacity(num_elems);

        unsafe {
            data_vec.set_len(num_elems);

            let data_ptr = data_vec.as_mut_ptr() as *mut u8;
            let data_slice = std::slice::from_raw_parts_mut(data_ptr, file_size);

            file.read_exact(data_slice)?;
        }

        // Ok(group_slices(
        //     data_vec.into_iter().map(|v| { //TODO move group_slices out, not for store
        //             (
        //                 IdSlice(0, v.0.0, v.0.1),
        //                 IdSlice(0, v.1.0, v.1.1)
        //             )
        //     })
        // ));

        Ok(())
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