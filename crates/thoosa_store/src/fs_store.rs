use crate::{IdSlice, MotifTrackerStore, SlicesMatch, Store, filter_groups_by_slices_count};
use bytemuck::{AnyBitPattern, Pod};
use log::{debug, error, info, trace, warn};
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::{fs, io};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thoosa_shared::run_configs::RunId;
use thoosa_shared::{ThoError, group_slices};
use zstd::stream::write::Encoder;

const FILE_MAGIC: [u8; 8] = *b"THOSCHNK";
const FILE_VERSION: u32 = 1;
const CHUNK_HEADER_SIZE: usize = 24;

struct ChunkedWriter {
    file: File,
    compression_level: i32,
    scratch: Vec<u8>,
}

impl ChunkedWriter {
    fn new(mut file: File, compression_level: i32) -> io::Result<Self> {
        let len = file.metadata()?.len();

        if len == 0 {
            file.write_all(&FILE_MAGIC)?;
            file.write_all(&FILE_VERSION.to_le_bytes())?;
            file.write_all(&[0u8; 4])?;
            file.flush()?;
        } else {
            file.seek(SeekFrom::Start(0))?;

            let mut magic = [0u8; FILE_MAGIC.len()];
            file.read_exact(&mut magic)?;
            if magic != FILE_MAGIC {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "unexpected slice store header"));
            }

            let mut version_bytes = [0u8; 4];
            file.read_exact(&mut version_bytes)?;
            let version = u32::from_le_bytes(version_bytes);
            if version != FILE_VERSION {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!("unsupported slice store version: {version}")));
            }

            let mut reserved = [0u8; 4];
            file.read_exact(&mut reserved)?;
        }

        file.seek(SeekFrom::End(0))?;

        Ok(Self {
            file,
            compression_level,
            scratch: Vec::with_capacity(8 * 1024),
        })
    }

    fn write_items<T: Pod>(&mut self, items: &[T]) -> io::Result<()> {
        if items.is_empty() {
            return Ok(());
        }

        let bytes = bytemuck::cast_slice(items);

        self.scratch.clear();
        {
            let mut encoder = Encoder::new(&mut self.scratch, self.compression_level)?;
            encoder.write_all(bytes)?;
            encoder.finish()?;
        }

        let decompressed_len = bytes.len() as u64;
        let item_count = items.len() as u64;
        let compressed_len = self.scratch.len() as u64;

        let mut header = [0u8; CHUNK_HEADER_SIZE];
        header[0..8].copy_from_slice(&decompressed_len.to_le_bytes());
        header[8..16].copy_from_slice(&item_count.to_le_bytes());
        header[16..24].copy_from_slice(&compressed_len.to_le_bytes());

        self.file.write_all(&header)?;
        self.file.write_all(&self.scratch)?;

        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }

    fn shutdown(&mut self) -> io::Result<()> {
        self.file.sync_all()
    }
}

pub struct FileMotifTrackerStore {
    run_id: RunId,
    channel: u32,
    writer: ChunkedWriter,
    path: PathBuf,
}

impl Debug for FileMotifTrackerStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileMotifTrackerStore")
            .field("run_id", &self.run_id)
            .field("channel", &self.channel)
            .field("path", &self.path)
            .finish()
    }
}

impl FileMotifTrackerStore {
    pub fn new<P: AsRef<Path> + Sync + Send + Debug>(path: P, run_id: RunId, channel: u32) -> Result<Box<dyn MotifTrackerStore>, ThoError> {
        let file_path = path.as_ref().join(format!("{run_id}_{channel}_slices.zstd").as_str());

        let file = std::fs::OpenOptions::new().create(true).read(true).write(true).open(&file_path)?;

        debug!("FileMotifTrackerStore::new: path={path:?} run_id={run_id} channel={channel} file_path={file_path:?}");

        let writer = ChunkedWriter::new(file, 3)?;

        Ok(Box::new(Self {
            run_id,
            channel,
            writer,
            path: file_path,
        }))
    }
}

impl MotifTrackerStore for FileMotifTrackerStore {
    fn save_slices(&mut self, slices_pairs: Vec<SlicesMatch>) -> Result<(), ThoError> {
        trace!("MotifTrackerStore::save_slices: slices_pairs.len()={}", slices_pairs.len());

        self.writer.write_items(&slices_pairs)?;
        self.writer.flush()?;

        Ok(())
    }

    fn load_slices_into_groups(&mut self, store: &Arc<RwLock<Box<dyn Store>>>, max_slices_counts: Option<u32>) -> Result<usize, ThoError> {
        debug!("MotifTrackerStore::load_slices_into_groups: max_slices_counts={max_slices_counts:?}");

        self.writer.shutdown()?;
        let mut slice_id = 0;

        let groups: HashMap<u16, HashSet<IdSlice>> = group_slices(ChunkedSlicesReader::new(&self.path)?.flat_map(|slices| {
            slices.unwrap().into_iter().map(move |slices: SlicesMatch| {
                slice_id += 1;
                (IdSlice(slice_id, slices.0.0, slices.0.1), IdSlice(slice_id, slices.1.0, slices.1.1))
            })
        }));
        debug!("MotifTrackerStore::load_slices_into_groups: inserting {} groups {} slices", groups.len(), groups.values().map(|s| s.len()).sum::<usize>());
        let groups_len = groups.len();
        store.write().save_groups_as_motifs(self.run_id, self.channel as i64, filter_groups_by_slices_count(max_slices_counts, groups))?;

        fs::remove_file(&self.path)?;

        Ok(groups_len)
    }

    fn run_id(&self) -> RunId {
        self.run_id
    }

    fn channel(&self) -> u32 {
        self.channel
    }

}

pub struct ChunkedSlicesReader<T: AnyBitPattern> {
    reader: BufReader<File>,
    _phantom: PhantomData<T>,
    file_path: PathBuf,
}

impl<T: AnyBitPattern> ChunkedSlicesReader<T> {
    pub fn new(file_path: &Path) -> std::io::Result<Self> {
        let mut file = File::open(&file_path)?;

        let mut magic = [0u8; FILE_MAGIC.len()];
        file.read_exact(&mut magic)?;
        if magic != FILE_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "unexpected slice store header"));
        }

        let mut version_bytes = [0u8; 4];
        file.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != FILE_VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("unsupported slice store version: {version}")));
        }

        let mut reserved = [0u8; 4];
        file.read_exact(&mut reserved)?;

        debug!("ChunkedSlicesReader::new: file_path={file_path:?}");

        Ok(Self {
            reader: BufReader::new(file),
            _phantom: PhantomData,
            file_path: file_path.to_path_buf(),
        })
    }
}

impl<T: AnyBitPattern> Iterator for ChunkedSlicesReader<T> {
    type Item = io::Result<Vec<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        debug!("ChunkedSlicesReader::next: next");

        let mut header = [0u8; CHUNK_HEADER_SIZE];
        let mut read = 0;

        while read < CHUNK_HEADER_SIZE {
            match self.reader.read(&mut header[read..]) {
                Ok(0) if read == 0 => {
                    warn!("ChunkedSlicesReader::next: empty file self.file_path={:?}", self.file_path);

                    return None
                },
                Ok(0) => {

                    return Some(Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated chunk header")));
                }
                Ok(n) => {
                    trace!("ChunkedSlicesReader::next: read {n} bytes from {:?}", self.file_path);

                    read += n
                },
                Err(e) => return Some(Err(e)),
            }
        }

        let decompressed_len = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
        let item_count = u64::from_le_bytes(header[8..16].try_into().unwrap());
        let compressed_len = u64::from_le_bytes(header[16..24].try_into().unwrap()) as usize;

        let mut compressed = vec![0u8; compressed_len];
        if let Err(e) = self.reader.read_exact(&mut compressed) {
            return Some(Err(io::Error::new(io::ErrorKind::UnexpectedEof, format!("truncated chunk payload: {e}"))));
        }

        let decompressed = match zstd::decode_all(&compressed[..]) {
            Ok(data) => data,
            Err(e) => return Some(Err(e)),
        };

        if decompressed.len() != decompressed_len {
            return Some(Err(io::Error::new(io::ErrorKind::InvalidData, format!("unexpected decompressed length: expected {decompressed_len}, got {}", decompressed.len()))));
        }

        let element_size = size_of::<T>();
        if decompressed.len() % element_size != 0 {
            return Some(Err(io::Error::new(io::ErrorKind::InvalidData, format!("decompressed buffer size {} not aligned to element size {}", decompressed.len(), element_size))));
        }

        let actual_item_count = (decompressed.len() / element_size) as u64;
        if actual_item_count != item_count {
            warn!("ChunkedSlicesReader: item_count mismatch: header={item_count}, calculated={actual_item_count} file_path={:?}", self.file_path);
        }

        let result = match bytemuck::try_cast_slice(&decompressed) {
            Ok(slice) => slice.to_vec(),
            Err(e) => {
                return Some(Err(io::Error::new(io::ErrorKind::InvalidData, format!("ChunkedSlicesReader: file_path={:?} cast error: {e:?}", self.file_path))));
            }
        };

        debug!("ChunkedSlicesReader::next: result.len={}", result.len());

        Some(Ok(result))
    }
}

#[cfg(test)]
mod tests {
    use crate::fs_store::{ChunkedSlicesReader, FileMotifTrackerStore, SlicesMatch};
    use crate::{IdSlice, SqlStore, filter_groups_by_slices_count};
    use log::LevelFilter::Trace;
    use log::{debug, error};
    use parking_lot::{Mutex, RwLock};
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use thoosa_shared::test_utils::init_test_logging;
    use thoosa_shared::{TensorIdSliceIdx, ThoError, cdt_for_dir, group_slices};

    #[test]
    fn test_single_part() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        debug!("test_single_part: test_dir={test_dir:?}");
        fs::create_dir_all(&test_dir)?;

        let mut store = Arc::new(RwLock::new(SqlStore::from_path(test_dir.join("main_store.db3"))?));

        let resource_ids = crate::store_tests::init_test_store(&mut store)?;

        let run_id = store.write().save_run_config(None, &"todo", "test", "keys", HashMap::new())?;

        let mut file_store = FileMotifTrackerStore::new(test_dir, run_id, 3)?;

        file_store.save_slices(vec![
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 100), TensorIdSliceIdx(resource_ids[0], 200), 0.1),
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 200), TensorIdSliceIdx(resource_ids[0], 300), 0.2),
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 100), TensorIdSliceIdx(resource_ids[0], 700), 0.3),
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 700), TensorIdSliceIdx(resource_ids[0], 800), 0.4),
        ])?;

        file_store.load_slices_into_groups(&store, None)?;

        let resources = store.read().list_resources_by_ids(&resource_ids)?;
        let map = store.read().list_motifs_slices(run_id, None)?;

        let motifs_slices = map.get(&resources[0]).expect("resource 0");
        let mut slise_idxs = motifs_slices.into_iter().map(|s| s.slice_idx).collect::<Vec<_>>();
        slise_idxs.sort();

        assert_eq!(vec![100, 200, 300, 700, 800], slise_idxs);

        Ok(())
    }

    #[test]
    fn test_two_motifs() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        debug!("test_single_part: test_dir={test_dir:?}");
        fs::create_dir_all(&test_dir)?;

        let mut store = Arc::new(RwLock::new(SqlStore::from_path(test_dir.join("main_store.db3"))?));

        let resource_ids = crate::store_tests::init_test_store(&mut store)?;

        let run_id = store.write().save_run_config(None, &"todo", "test", "keys", HashMap::new())?;

        let mut file_store = FileMotifTrackerStore::new(test_dir, run_id, 3)?;

        file_store.save_slices(vec![
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 100), TensorIdSliceIdx(resource_ids[1], 200), 0.1),
            SlicesMatch(TensorIdSliceIdx(resource_ids[1], 200), TensorIdSliceIdx(resource_ids[1], 700), 0.3),
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 300), TensorIdSliceIdx(resource_ids[1], 400), 0.2),
            SlicesMatch(TensorIdSliceIdx(resource_ids[1], 400), TensorIdSliceIdx(resource_ids[0], 800), 0.4),
        ])?;

        file_store.load_slices_into_groups(&store, None)?;

        let resources = store.read().list_resources_by_ids(&resource_ids)?;
        let map = store.read().list_motifs_slices(run_id, None)?;

        let motifs_slices = map.get(&resources[0]).expect("resource 0");
        let resoruce_0_motifs = motifs_slices.iter().map(|ms| ms.motif_id).collect::<HashSet<_>>();
        assert_eq!(resoruce_0_motifs.len(), 2);
        let mut slise_idxs = motifs_slices.into_iter().map(|s| s.slice_idx).collect::<Vec<_>>();
        slise_idxs.sort();

        assert_eq!(vec![100, 300, 800], slise_idxs);

        let motifs_slices = map.get(&resources[1]).expect("resource 0");
        let resoruce_1_motifs = motifs_slices.iter().map(|ms| ms.motif_id).collect::<HashSet<_>>();
        assert_eq!(resoruce_1_motifs.len(), 2);
        let mut slise_idxs = motifs_slices.into_iter().map(|s| s.slice_idx).collect::<Vec<_>>();
        slise_idxs.sort();

        assert_eq!(vec![200, 400, 700], slise_idxs);

        Ok(())
    }

    #[test]
    fn test_single_part_with_resources() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        debug!("test_single_part: test_dir={test_dir:?}");
        fs::create_dir_all(&test_dir)?;

        let mut store = Arc::new(RwLock::new(SqlStore::from_path(test_dir.join("main_store.db3"))?));

        let resource_ids = crate::store_tests::init_test_store(&mut store)?;

        let run_id = store.write().save_run_config(None, &"todo", "test", "keys", HashMap::new())?;

        let mut file_store = FileMotifTrackerStore::new(test_dir, run_id, 3)?;

        file_store.save_slices(vec![
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 100), TensorIdSliceIdx(resource_ids[1], 200), 0.1),
            SlicesMatch(TensorIdSliceIdx(resource_ids[1], 200), TensorIdSliceIdx(resource_ids[0], 300), 0.2),
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 100), TensorIdSliceIdx(resource_ids[1], 700), 0.3),
            SlicesMatch(TensorIdSliceIdx(resource_ids[1], 700), TensorIdSliceIdx(resource_ids[0], 800), 0.4),
        ])?;

        file_store.load_slices_into_groups(&store, None)?;

        let resources = store.read().list_resources_by_ids(&resource_ids)?;
        let map = store.read().list_motifs_slices(run_id, None)?;

        let motifs_slices = map.get(&resources[0]).expect("resource 0");
        let mut slise_idxs_0 = motifs_slices.into_iter().map(|s| s.slice_idx).collect::<Vec<_>>();
        slise_idxs_0.sort();

        assert_eq!(vec![100, 300, 800], slise_idxs_0);

        let motifs_slices = map.get(&resources[1]).expect("resource 0");
        let mut slise_idxs_1 = motifs_slices.into_iter().map(|s| s.slice_idx).collect::<Vec<_>>();
        slise_idxs_1.sort();

        assert_eq!(vec![200, 700], slise_idxs_1);

        Ok(())
    }

    #[test]
    fn test_two_parts() -> Result<(), ThoError> {
        // log::set_max_level(Trace);
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        debug!("test_single_part: test_dir={test_dir:?}");
        fs::create_dir_all(&test_dir)?;

        let mut store = Arc::new(RwLock::new(SqlStore::from_path(test_dir.join("main_store.db3"))?));

        let resource_ids = crate::store_tests::init_test_store(&mut store)?;

        let run_id = store.write().save_run_config(None, &"todo", "test", "keys", HashMap::new())?;

        let mut file_store = FileMotifTrackerStore::new(test_dir, run_id, 3)?;

        file_store.save_slices(vec![
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 100), TensorIdSliceIdx(resource_ids[0], 200), 0.1),
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 200), TensorIdSliceIdx(resource_ids[0], 300), 0.2),
        ])?;

        file_store.save_slices(vec![
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 100), TensorIdSliceIdx(resource_ids[0], 700), 0.3),
            SlicesMatch(TensorIdSliceIdx(resource_ids[0], 700), TensorIdSliceIdx(resource_ids[0], 800), 0.4),
        ])?;

        file_store.load_slices_into_groups(&store, None)?;

        let resources = store.read().list_resources_by_ids(&resource_ids)?;
        let map = store.read().list_motifs_slices(run_id, None)?;

        let motifs_slices = map.get(&resources[0]).expect("resource 0");
        let mut slise_idxs = motifs_slices.into_iter().map(|s| s.slice_idx).collect::<Vec<_>>();
        slise_idxs.sort();

        assert_eq!(vec![100, 200, 300, 700, 800], slise_idxs);

        Ok(())
    }

    #[test]
    fn test_no_matches() -> Result<(), ThoError> {
        // log::set_max_level(Trace);
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        debug!("test_single_part: test_dir={test_dir:?}");
        fs::create_dir_all(&test_dir)?;

        let mut store = Arc::new(RwLock::new(SqlStore::from_path(test_dir.join("main_store.db3"))?));

        let resource_ids = crate::store_tests::init_test_store(&mut store)?;

        let run_id = store.write().save_run_config(None, &"todo", "test", "keys", HashMap::new())?;

        let mut file_store = FileMotifTrackerStore::new(test_dir, run_id, 3)?;

        file_store.load_slices_into_groups(&store, None)?;

        let map = store.read().list_motifs_slices(run_id, None)?;

        assert_eq!(0, map.len());

        Ok(())
    }

    /*
     not a test - util to debug failing deserialization
    #[test]
     fn test_z() -> Result<(), ThoosaError>{

         let mut slice_id = 0;
         let run_id=28;
         let channel=19;

         log::set_max_level(Trace);
         let groups:HashMap<u16, HashSet<IdSlice>> = group_slices(

            // ChunkedSlicesReader::<SlicesMatch>::new(&PathBuf::from(format!("/tmp/ws_test/tmp/{run_id}_{channel}_slices.zstd").as_str()))?.flat_map(|slices|{
            ChunkedSlicesReader::<SlicesMatch>::new(&PathBuf::from("/tmp/2024-12-12T12_54_47.820_01_00_3468226743/tmp/1_0_slices.zstd"))?.flat_map(|slices|{
                 slices.unwrap().into_iter().map(move |slices:SlicesMatch|{
                     slice_id+=1;
                     (
                         IdSlice(slice_id, slices.0.0, slices.0.1),
                         IdSlice(slice_id, slices.1.0, slices.1.1)
                     )
                 })
             })
         );

         debug!("{run_id}_{channel}_slices slice_id={slice_id} groups.len={}", groups.len());
         let test_dir = Path::new("/tmp").join(cdt_for_dir());
         fs::create_dir_all(&test_dir)?;
         let sql_file = test_dir.join("test_store.db3");
         fs::copy("/tmp/ws_test/main_db.db3", &sql_file)?;

         debug!("sql_file={sql_file:?}");
         let store = Arc::new(RwLock::new(SqlStore::from_path(sql_file)?));
         // let store = Arc::new(RwLock::new(SqlStore::in_mem()?));

         match store.write().save_groups_as_motifs(
             28, 19,
             filter_groups_by_slices_count(None, groups)
         ) {
             Ok(v) => Ok(v),
             Err(e) => {
                 Err(e)
             }
         }?;

         Ok(())

     }*/
}
