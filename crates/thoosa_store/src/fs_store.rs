use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::mem;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use log::{debug, info};
use parking_lot::Mutex;
use rand::{thread_rng, Rng};
use zstd::Encoder;
use thoosa_shared::resource::ResourceId;
use thoosa_shared::{group_slices, TensorSliceIdx, ThoosaError};
use crate::{IdSlice, MotifTrackerStore, Store};

type SlicesMatch = (TensorSliceIdx<ResourceId>, TensorSliceIdx<ResourceId>, f32);

#[derive(Debug)]
pub struct FileMotifTrackerStore {
    run_id: usize,
    channel: usize,
    file: File,
    path: PathBuf,
}

impl FileMotifTrackerStore {

    pub fn new<P: AsRef<Path> + Sync + Send + Debug>(path: P, run_id: usize, channel: usize) -> Result<Box<dyn MotifTrackerStore>, ThoosaError> {

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref().join(format!("{}_{}_slices.zstd", run_id, channel).as_str()))?;

        Ok(Box::new(Self {
            run_id,
            channel,
            file,
            path: path.as_ref().to_path_buf(),

        }))
    }

}

impl MotifTrackerStore for FileMotifTrackerStore {

    fn save_slices(&mut self, slices_pairs: Vec<SlicesMatch>) -> Result<(), ThoosaError> {
        debug!("MotifTrackerStore::save_slices: slices_pairs.len()={}", slices_pairs.len());

        //TODO append, new if size > threshold

        let mut encoder = Encoder::new(std::io::BufWriter::new(&self.file), 3)?; // Compression level 3

        // Convert the entire Vec<T> into a byte slice
        let data_bytes = unsafe {
            std::slice::from_raw_parts(
                slices_pairs.as_ptr() as *const u8,
                slices_pairs.len() * mem::size_of::<SlicesMatch>(),
            )
        };

        encoder.write_all(data_bytes)?;
        encoder.finish()?;

        Ok(())
    }

    fn load_slices_into_groups(&mut self, store: &Arc<Mutex<Box<dyn Store>>>) -> Result<(), ThoosaError> {
        let elem_size = size_of::<SlicesMatch>();
        let mut slice_id = 0;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(format!("{}_{}_slices.zstd", self.run_id, self.channel).as_str())?;

        let chunk_reader:ChunkReader< crate::fs_store::SlicesMatch > = ChunkReader::new(&self.path.join(format!("{}_{}_slices.zstd", self.run_id, self.channel).as_str()))?;
        let groups:HashMap<u16, HashSet<IdSlice>> = group_slices(

            chunk_reader.flat_map(|slices|{
                slices.unwrap().into_iter().map(move |slices|{
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
        todo!()
    }
}

pub struct ChunkReader<T> {
    reader: BufReader<File>,
    _phantom: PhantomData<T>,
}

impl<T> ChunkReader<T> {
    pub fn new(file_path: &Path) -> std::io::Result<Self> {
        let file = File::open(file_path)?;
        Ok(Self {
            reader: BufReader::new(file),
            _phantom: PhantomData,
        })
    }
}

impl<T> Iterator for ChunkReader<T> {
    type Item = std::io::Result<Vec<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        let current_position = self.reader.stream_position().ok()?;
        let mut buffer = Vec::new();

        match zstd::stream::read::Decoder::new(&mut self.reader) {
            Ok(mut decoder) => {
                match decoder.read_to_end(&mut buffer) {
                    Ok(_) => {
                        // Ensure the buffer size matches a multiple of the size of T
                        let elem_size = std::mem::size_of::<T>();
                        if buffer.len() % elem_size != 0 {
                            return Some(Err(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!(
                                    "Buffer size ({}) is not a multiple of type size ({})",
                                    buffer.len(),
                                    elem_size
                                ),
                            )));
                        }

                        // Convert the buffer into a Vec<T>
                        let num_elements = buffer.len() / elem_size;
                        let mut result = Vec::with_capacity(num_elements);

                        unsafe {
                            let ptr = buffer.as_ptr() as *const T;
                            for i in 0..num_elements {
                                result.push(ptr.add(i).read_unaligned());
                            }
                        }

                        Some(Ok(result))
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                        None // End of file
                    }
                    Err(e) => Some(Err(e)), // Propagate other errors
                }
            }
            Err(e) => {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    None // End of file
                } else {
                    self.reader.seek(SeekFrom::Start(current_position)).ok(); // Rewind on error
                    Some(Err(e)) // Propagate other errors
                }
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::io::{Read, Seek, SeekFrom, Write};
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

    #[test]
    fn test_z() {
        use zstd::stream::write::Encoder;

        fs::remove_file("/tmp/test.zstd").unwrap();

        fn write_chunks(file_path: &str, data_chunks: &[&[u8]]) -> std::io::Result<()> {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(file_path)?;

            for chunk in data_chunks {
                // Create a new encoder for each chunk (independent frame)
                let mut encoder = Encoder::new(std::io::BufWriter::new(&file), 3)?; // Compression level 3
                encoder.write_all(chunk)?;
                encoder.finish()?; // Finalize the frame
            }

            Ok(())
        }

        fn read_chunks(file_path: &str) -> std::io::Result<()> {
            let file = std::fs::File::open(file_path)?;
            let mut reader = std::io::BufReader::new(file);

            let mut buffer = Vec::new();

            loop {
                // Remember the current position before attempting to read
                let current_position = reader.stream_position()?;

                match zstd::stream::read::Decoder::new(&mut reader){
                    Ok(mut decoder) => {
                        buffer.clear();

                        match decoder.read_to_end(&mut buffer) {
                            Ok(_) => {
                                println!("Read chunk: {}", String::from_utf8_lossy(&buffer));
                            }
                            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                                // We've reached the end of the file while reading this frame
                                break;
                            }
                            Err(e) => return Err(e), // Propagate other errors
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                        // We've reached the end of the file while attempting to initialize a new frame
                        break;
                    }
                    Err(_) => {
                        // Seek back to the position before the failed attempt
                        reader.seek(SeekFrom::Start(current_position))?;
                        break;
                    }
                }
            }

            Ok(())
        }


        let data_chunks = vec![
            "Chunk 1: This is the first chunk of data.".as_bytes(),
            "Chunk 2: This is the second chunk of data.".as_bytes(),
            "Chunk 3: This is the third chunk of data.".as_bytes(),
        ];

        write_chunks("/tmp/test.zstd", data_chunks.as_slice()).unwrap();
        println!("Data chunks written to 'data.zst'!");

        read_chunks("/tmp/test.zstd").unwrap();

    }

}