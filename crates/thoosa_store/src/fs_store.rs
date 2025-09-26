use crate::{IdSlice, MotifTrackerStore, SlicesMatch, Store, filter_groups_by_slices_count};
use bytemuck::AnyBitPattern;
use log::{debug, error, info, trace, warn};
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::io;
use std::io::{BufReader, Read, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thoosa_shared::run_configs::RunId;
use thoosa_shared::{ThoosaError, group_slices};
use zstd_framed::table::ZstdFrame;
use zstd_framed::{ZstdReader, ZstdWriter};

pub struct FileMotifTrackerStore {
    run_id: RunId,
    channel: u32,
    writer: ZstdWriter<'static, File>,
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
    pub fn new<P: AsRef<Path> + Sync + Send + Debug>(
        path: P,
        run_id: RunId,
        channel: u32,
    ) -> Result<Box<dyn MotifTrackerStore>, ThoosaError> {


        let file = std::fs::OpenOptions::new().create(true).append(true).open(
            path.as_ref()
                .join(format!("{run_id}_{channel}_slices.zstd").as_str()),
        )?;

        debug!("FileMotifTrackerStore::new: path={path:?} run_id={run_id} channel={channel} file={file:?}");

        let writer = ZstdWriter::builder(file)
            .with_compression_level(3) // Set compression level
            .with_seek_table(u32::MAX)
            .build()?;

        Ok(Box::new(Self {
            run_id,
            channel,
            writer,
            path: path.as_ref().to_path_buf(),
        }))
    }
}

impl MotifTrackerStore for FileMotifTrackerStore {
    fn save_slices(&mut self, slices_pairs: Vec<SlicesMatch>) -> Result<(), ThoosaError> {

        let data_bytes = unsafe {
            std::slice::from_raw_parts(
                slices_pairs.as_ptr() as *const u8,
                slices_pairs.len() * size_of::<SlicesMatch>(),
            )
        };

        trace!(
            "MotifTrackerStore::save_slices: slices_pairs.len()={} data_bytes={data_bytes:?}",
            slices_pairs.len()
        );


        // self.writer.write_all(data_bytes)?;
        if let Err(e) = self.writer.write_all(data_bytes) {
            error!("write_all failed: kind={:?} os={:?} len={}", e.kind(), e.raw_os_error(), data_bytes.len());
            match e.kind() {
                io::ErrorKind::WriteZero => error!("Likely out-of-space on a fixed-capacity writer (Cursor<&mut [u8]>/Take) or a buggy Write impl."),
                io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset => error!("Peer closed; recreate the connection."),
                _ => {}
            }
            std::process::abort();
        }

        self.writer.finish_frame()?;
        self.writer.flush()?;

        Ok(())
    }

    fn load_slices_into_groups(
        &mut self,
        store: &Arc<RwLock<Box<dyn Store>>>,
        max_slices_counts: Option<u32>,
    ) -> Result<(), ThoosaError> {
        debug!(
            "MotifTrackerStore::load_slices_into_groups: max_slices_counts={max_slices_counts:?}"
        );

        self.writer.shutdown()?;
        let mut slice_id = 0;

        let groups: HashMap<u16, HashSet<IdSlice>> = group_slices(
            ZstdFramedReader::new(
                &self
                    .path
                    .join(format!("{}_{}_slices.zstd", self.run_id, self.channel).as_str()),
            )?
            .flat_map(|slices| {
                slices.unwrap().into_iter().map(move |slices: SlicesMatch| {
                    slice_id += 1;
                    (
                        IdSlice(slice_id, slices.0.0, slices.0.1),
                        IdSlice(slice_id, slices.1.0, slices.1.1),
                    )
                })
            }),
        );
        info!(
            "inserting {} groups {} slices",
            groups.len(),
            groups.values().map(|s| s.len()).sum::<usize>()
        );
        store.write().save_groups_as_motifs(
            self.run_id,
            self.channel as i64,
            filter_groups_by_slices_count(max_slices_counts, groups),
        )
    }

    fn run_id(&self) -> RunId {
        self.run_id
    }

    fn channel(&self) -> u32 {
        self.channel
    }

    fn path(&self) -> &PathBuf {
        todo!()
    }
}

pub struct ZstdFramedReader<T: AnyBitPattern> {
    reader: ZstdReader<'static, BufReader<File>>,
    frames: Vec<ZstdFrame>,
    _phantom: PhantomData<T>,
    frame_i: usize,
    file_path: PathBuf,
}

impl<T: AnyBitPattern> ZstdFramedReader<T> {
    pub fn new(file_path: &Path) -> std::io::Result<Self> {
        let mut reader = File::open(&file_path)?;
        let table = zstd_framed::table::read_seek_table(&mut reader)?.unwrap();

        trace!("ChunkReader: table={table:?}");

        Ok(Self {
            reader: zstd_framed::ZstdReader::builder(reader).build()?,
            frames: table.frames().collect(),
            frame_i: 0,
            _phantom: PhantomData,
            file_path: file_path.to_path_buf(),
        })
    }
}

impl<T: AnyBitPattern> Iterator for ZstdFramedReader<T> {
    type Item = io::Result<Vec<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.frame_i < self.frames.len() && self.frames[self.frame_i].decompressed_size() > 0 {
            let mut buff = vec![0; self.frames[self.frame_i].decompressed_size() as usize];
            self.reader.read_exact(&mut buff).unwrap();

            self.frame_i += 1;

            Some(Ok(bytemuck::try_cast_slice(&mut buff)
                .map_err(|e| {
                    error!(
                        "ZstdFramedReader:next: file_path={:?} self.frame_i={} e={e:?}",
                        self.file_path, self.frame_i
                    );
                    e
                })
                .expect("bytemuck failed")
                .to_vec()))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::fs_store::{FileMotifTrackerStore, SlicesMatch, ZstdFramedReader};
    use crate::{IdSlice, SqlStore, filter_groups_by_slices_count};
    use log::LevelFilter::Trace;
    use log::{debug, error};
    use parking_lot::{Mutex, RwLock};
    use std::collections::{HashMap, HashSet};
    use std::{fs, thread};
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::thread::Thread;
    use zstd_framed::ZstdWriter;
    use thoosa_shared::{TensorIdSliceIdx, ThoosaError, cdt_for_dir, group_slices};
    use thoosa_shared::test_utils::init_test_logging;

    #[test]
    fn test_single_part() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        debug!("test_single_part: test_dir={test_dir:?}");
        fs::create_dir_all(&test_dir)?;

        let mut store = Arc::new(RwLock::new(SqlStore::from_path(
            test_dir.join("main_store.db3"),
        )?));

        let resource_ids = crate::store_tests::init_test_store(&mut store)?;

        let run_id =
            store
                .write()
                .save_run_config(None, &"todo", "test", "keys", HashMap::new())?;

        let mut file_store = FileMotifTrackerStore::new(test_dir, run_id, 3)?;

        file_store.save_slices(vec![
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[0], 100),
                TensorIdSliceIdx(resource_ids[0], 200),
                0.1,
            ),
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[0], 200),
                TensorIdSliceIdx(resource_ids[0], 300),
                0.2,
            ),
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[0], 100),
                TensorIdSliceIdx(resource_ids[0], 700),
                0.3,
            ),
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[0], 700),
                TensorIdSliceIdx(resource_ids[0], 800),
                0.4,
            ),
        ])?;

        file_store.load_slices_into_groups(&store, None)?;

        let resources = store.read().list_resources_by_ids(&resource_ids)?;
        let map = store.read().list_motifs_slices(run_id)?;

        let motifs_slices = map.get(&resources[0]).expect("resource 0");
        let mut slise_idxs = motifs_slices
            .into_iter()
            .map(|s| s.slice_idx)
            .collect::<Vec<_>>();
        slise_idxs.sort();

        assert_eq!(vec![100, 200, 300, 700, 800], slise_idxs);

        Ok(())
    }

    #[test]
    fn test_two_motifs() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        debug!("test_single_part: test_dir={test_dir:?}");
        fs::create_dir_all(&test_dir)?;

        let mut store = Arc::new(RwLock::new(SqlStore::from_path(
            test_dir.join("main_store.db3"),
        )?));

        let resource_ids = crate::store_tests::init_test_store(&mut store)?;

        let run_id =
            store
                .write()
                .save_run_config(None, &"todo", "test", "keys", HashMap::new())?;

        let mut file_store = FileMotifTrackerStore::new(test_dir, run_id, 3)?;

        file_store.save_slices(vec![
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[0], 100),
                TensorIdSliceIdx(resource_ids[1], 200),
                0.1,
            ),
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[1], 200),
                TensorIdSliceIdx(resource_ids[1], 700),
                0.3,
            ),
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[0], 300),
                TensorIdSliceIdx(resource_ids[1], 400),
                0.2,
            ),
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[1], 400),
                TensorIdSliceIdx(resource_ids[0], 800),
                0.4,
            ),
        ])?;

        file_store.load_slices_into_groups(&store, None)?;

        let resources = store.read().list_resources_by_ids(&resource_ids)?;
        let map = store.read().list_motifs_slices(run_id)?;

        let motifs_slices = map.get(&resources[0]).expect("resource 0");
        let resoruce_0_motifs = motifs_slices
            .iter()
            .map(|ms| ms.motif_id)
            .collect::<HashSet<_>>();
        assert_eq!(resoruce_0_motifs.len(), 2);
        let mut slise_idxs = motifs_slices
            .into_iter()
            .map(|s| s.slice_idx)
            .collect::<Vec<_>>();
        slise_idxs.sort();

        assert_eq!(vec![100, 300, 800], slise_idxs);

        let motifs_slices = map.get(&resources[1]).expect("resource 0");
        let resoruce_1_motifs = motifs_slices
            .iter()
            .map(|ms| ms.motif_id)
            .collect::<HashSet<_>>();
        assert_eq!(resoruce_1_motifs.len(), 2);
        let mut slise_idxs = motifs_slices
            .into_iter()
            .map(|s| s.slice_idx)
            .collect::<Vec<_>>();
        slise_idxs.sort();

        assert_eq!(vec![200, 400, 700], slise_idxs);

        Ok(())
    }

    #[test]
    fn test_single_part_with_resources() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        debug!("test_single_part: test_dir={test_dir:?}");
        fs::create_dir_all(&test_dir)?;

        let mut store = Arc::new(RwLock::new(SqlStore::from_path(
            test_dir.join("main_store.db3"),
        )?));

        let resource_ids = crate::store_tests::init_test_store(&mut store)?;

        let run_id =
            store
                .write()
                .save_run_config(None, &"todo", "test", "keys", HashMap::new())?;

        let mut file_store = FileMotifTrackerStore::new(test_dir, run_id, 3)?;

        file_store.save_slices(vec![
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[0], 100),
                TensorIdSliceIdx(resource_ids[1], 200),
                0.1,
            ),
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[1], 200),
                TensorIdSliceIdx(resource_ids[0], 300),
                0.2,
            ),
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[0], 100),
                TensorIdSliceIdx(resource_ids[1], 700),
                0.3,
            ),
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[1], 700),
                TensorIdSliceIdx(resource_ids[0], 800),
                0.4,
            ),
        ])?;

        file_store.load_slices_into_groups(&store, None)?;

        let resources = store.read().list_resources_by_ids(&resource_ids)?;
        let map = store.read().list_motifs_slices(run_id)?;

        let motifs_slices = map.get(&resources[0]).expect("resource 0");
        let mut slise_idxs_0 = motifs_slices
            .into_iter()
            .map(|s| s.slice_idx)
            .collect::<Vec<_>>();
        slise_idxs_0.sort();

        assert_eq!(vec![100, 300, 800], slise_idxs_0);

        let motifs_slices = map.get(&resources[1]).expect("resource 0");
        let mut slise_idxs_1 = motifs_slices
            .into_iter()
            .map(|s| s.slice_idx)
            .collect::<Vec<_>>();
        slise_idxs_1.sort();

        assert_eq!(vec![200, 700], slise_idxs_1);

        Ok(())
    }

    #[test]
    fn test_two_parts() -> Result<(), ThoosaError> {
        // log::set_max_level(Trace);
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        debug!("test_single_part: test_dir={test_dir:?}");
        fs::create_dir_all(&test_dir)?;

        let mut store = Arc::new(RwLock::new(SqlStore::from_path(
            test_dir.join("main_store.db3"),
        )?));

        let resource_ids = crate::store_tests::init_test_store(&mut store)?;

        let run_id =
            store
                .write()
                .save_run_config(None, &"todo", "test", "keys", HashMap::new())?;

        let mut file_store = FileMotifTrackerStore::new(test_dir, run_id, 3)?;

        file_store.save_slices(vec![
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[0], 100),
                TensorIdSliceIdx(resource_ids[0], 200),
                0.1,
            ),
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[0], 200),
                TensorIdSliceIdx(resource_ids[0], 300),
                0.2,
            ),
        ])?;

        file_store.save_slices(vec![
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[0], 100),
                TensorIdSliceIdx(resource_ids[0], 700),
                0.3,
            ),
            SlicesMatch(
                TensorIdSliceIdx(resource_ids[0], 700),
                TensorIdSliceIdx(resource_ids[0], 800),
                0.4,
            ),
        ])?;

        file_store.load_slices_into_groups(&store, None)?;

        let resources = store.read().list_resources_by_ids(&resource_ids)?;
        let map = store.read().list_motifs_slices(run_id)?;

        let motifs_slices = map.get(&resources[0]).expect("resource 0");
        let mut slise_idxs = motifs_slices
            .into_iter()
            .map(|s| s.slice_idx)
            .collect::<Vec<_>>();
        slise_idxs.sort();

        assert_eq!(vec![100, 200, 300, 700, 800], slise_idxs);

        Ok(())
    }

    #[test]
    fn test_no_matches() -> Result<(), ThoosaError> {
        // log::set_max_level(Trace);
        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        debug!("test_single_part: test_dir={test_dir:?}");
        fs::create_dir_all(&test_dir)?;

        let mut store = Arc::new(RwLock::new(SqlStore::from_path(
            test_dir.join("main_store.db3"),
        )?));

        let resource_ids = crate::store_tests::init_test_store(&mut store)?;

        let run_id =
            store
                .write()
                .save_run_config(None, &"todo", "test", "keys", HashMap::new())?;

        let mut file_store = FileMotifTrackerStore::new(test_dir, run_id, 3)?;

        file_store.load_slices_into_groups(&store, None)?;

        let map = store.read().list_motifs_slices(run_id)?;

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

             // ZstdFramedReader::<SlicesMatch>::new(&PathBuf::from(format!("/tmp/ws_test/tmp/{run_id}_{channel}_slices.zstd").as_str()))?.flat_map(|slices|{
             ZstdFramedReader::<SlicesMatch>::new(&PathBuf::from("/tmp/2024-12-12T12_54_47.820_01_00_3468226743/tmp/1_0_slices.zstd"))?.flat_map(|slices|{
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

    #[test]
    fn test_write_all(){
        init_test_logging();

        debug!("test_write_all");

        let nums = [1, 2, 3, 4, 5];

        let mut file = File::create("/tmp/foo.txt").expect("create failed");
        let mut writer = ZstdWriter::builder(file)
            .with_compression_level(3) // Set compression level
            .with_seek_table(u32::MAX)
            .build().unwrap();

        let data_bytes = &[3, 0, 0, 0, 232, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 232, 3, 0, 0, 0, 0, 0, 0];

        let results: Vec<()> = thread::scope(|s| {
            let mut handles = Vec::new();

            handles.push(s.spawn(move || {
                for i in  0..1000 {

                    match writer.write_all(data_bytes) {
                        Ok(_) => {}
                        Err(e) => {
                            panic!("{:?}", e);
                        },
                    }

                    writer.finish_frame().unwrap();
                    writer.flush().unwrap();
                }
            }));

            for t in (0..32){
                handles.push(s.spawn(move || {
                    for i in 0..10000 {
                        debug!("worker {t} on {i} aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
                    }

                }));
            }

            // Collect results (join happens here)
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
    }
}
