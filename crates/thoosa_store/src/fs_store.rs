use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::{io, mem};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use log::{debug, info};
use parking_lot::Mutex;
use rand::{thread_rng, Rng};
use zstd_framed::{ZstdReader, ZstdWriter};
use zstd_framed::table::{ZstdFrame, ZstdSeekTable};
use thoosa_shared::resource::ResourceId;
use thoosa_shared::{group_slices, TensorSliceIdx, ThoosaError};
use crate::{filter_groups_by_slices_count, IdSlice, MotifTrackerStore, Store};


type SlicesMatch = (TensorSliceIdx<ResourceId>, TensorSliceIdx<ResourceId>, f32);

pub struct FileMotifTrackerStore {
    run_id: usize,
    channel: usize,
    writer: ZstdWriter<'static, File>,
    path: PathBuf,
}

impl Debug for FileMotifTrackerStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl FileMotifTrackerStore {

    pub fn new<P: AsRef<Path> + Sync + Send + Debug>(path: P, run_id: usize, channel: usize) -> Result<Box<dyn MotifTrackerStore>, ThoosaError> {

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref().join(format!("{}_{}_slices.zstd", run_id, channel).as_str()))?;

        let mut writer = zstd_framed::ZstdWriter::builder(file)
            .with_compression_level(3) // Set compression level
            .with_seek_table(100_000)
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
        debug!("MotifTrackerStore::save_slices: slices_pairs.len()={}", slices_pairs.len());


        let data_bytes = unsafe {
            std::slice::from_raw_parts(
                slices_pairs.as_ptr() as *const u8,
                slices_pairs.len() * mem::size_of::<SlicesMatch>(),
            )
        };

        self.writer.write_all(data_bytes)?;
        self.writer.finish_frame()?;
        self.writer.flush()?;

        Ok(())
    }

    fn load_slices_into_groups(&mut self, store: &Arc<Mutex<Box<dyn Store>>>, max_slices_counts: Option<u32>) -> Result<(), ThoosaError> {
        self.writer.shutdown()?;
        let mut slice_id = 0;

        let groups:HashMap<u16, HashSet<IdSlice>> = group_slices(

            ChunkReader::new(&self.path.join(format!("{}_{}_slices.zstd", self.run_id, self.channel).as_str()))?.flat_map(|slices|{
                slices.unwrap().into_iter().map(move |slices:SlicesMatch|{
                    slice_id+=1;
                    (
                        IdSlice(slice_id, slices.0.0, slices.0.1),
                        IdSlice(slice_id, slices.1.0, slices.1.1)
                    )
                })
            })
        );
        info!("inserting {} groups {} slices", groups.len(), groups.values().map(|s|s.len()).sum::<usize>());

        store.lock().save_groups_as_motifs(
            self.run_id, self.channel as i64,
            filter_groups_by_slices_count(max_slices_counts, groups)
        )
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
    reader: ZstdReader<'static, BufReader<File>>,
    frames:Vec<ZstdFrame>,
    _phantom: PhantomData<T>,
    frame_i:usize
}

impl<T> ChunkReader<T> {

    pub fn new(file_path: &Path) -> std::io::Result<Self> {
        let file = File::open(file_path)?;

        let mut reader = File::open(file_path)?;
        let table = zstd_framed::table::read_seek_table(&mut reader)?.unwrap();


        Ok(Self {
            reader: zstd_framed::ZstdReader::builder(reader).build()?,
            frames:table.frames().collect(),
            frame_i: 0,
            _phantom: PhantomData,
        })
    }
}

impl<T> Iterator for ChunkReader<T> {
    type Item = io::Result<Vec<T>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.frame_i < self.frames.len() {
            let mut buff = vec![0; self.frames[self.frame_i].decompressed_size() as usize];
            self.reader.read_exact(&mut buff).unwrap();

            self.frame_i+=1;

            Some(Ok(deserialize_buff(size_of::<T>(), &mut buff)))

        } else {
            None
        }
    }
}

fn deserialize_buff<T>(elem_size: usize, buff: &mut Vec<u8>) -> Vec<T> {
    let num_elements = buff.len() / elem_size;
    let mut result = Vec::with_capacity(num_elements);

    unsafe {
        let ptr = buff.as_ptr() as *const T;
        for i in 0..num_elements {
            result.push(ptr.add(i).read_unaligned());
        }
    }
    result
}



#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use log::debug;
    use parking_lot::{Mutex, RwLock};
    use thoosa_shared::{cdt_for_dir, TensorSliceIdx, ThoosaError};
    use crate::fs_store::{deserialize_buff, ChunkReader, FileMotifTrackerStore, SlicesMatch};
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

        file_store.load_slices_into_groups(&store, None)?;

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

        file_store.load_slices_into_groups(&store, None)?;

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

        file_store.load_slices_into_groups(&store, None)?;

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

        file_store.load_slices_into_groups(&store, None)?;

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
    fn test_z() -> Result<(), ThoosaError>{
        // #[repr(C)]
        // #[derive(Debug)]
        // struct MyStruct {
        //     x: i32,
        //     y: f64,
        //     z: u8,
        // }
        //
        //
        // pub fn write_structs<T>(file_path: &str, structs: &[T], writer: &mut zstd_framed::ZstdWriter<File>) -> std::io::Result<()> {
        //     use std::mem;
        //
        //     let data_bytes = unsafe {
        //         std::slice::from_raw_parts(
        //             structs.as_ptr() as *const u8,
        //             structs.len() * mem::size_of::<T>(),
        //         )
        //     };
        //
        //     writer.write_all(data_bytes)?;
        //     writer.finish_frame()?;
        //     writer.flush()?; // Ensure all data is flushed and the frame is finalized
        //     Ok(())
        // }
        // let file_path = "/tmp/test_structs.zstd";
        // fs::remove_file(file_path)?;
        //
        // let structs_0 = vec![
        //     MyStruct { x: 1, y: 1.1, z: 10 },
        //     MyStruct { x: 2, y: 2.2, z: 20 },
        //     MyStruct { x: 3, y: 3.3, z: 30 },
        //     MyStruct { x: 4, y: 4.3, z: 40 },
        // ];
        //
        // let file = std::fs::OpenOptions::new()
        //     .create(true)
        //     .append(true)
        //     .open(file_path)?;
        //
        // let mut writer = zstd_framed::ZstdWriter::builder(file)
        //     .with_compression_level(3) // Set compression level
        //     .with_seek_table(100_000)
        //     .build()?;
        //
        // write_structs(file_path, &structs_0, &mut writer)?;
        //
        // let structs_1 = vec![
        //     MyStruct { x: 10, y: 10.1, z: 100 },
        //     MyStruct { x: 20, y: 20.2, z: 200 },
        //     MyStruct { x: 30, y: 30.3, z: 255 },
        // ];
        //
        // write_structs(file_path, &structs_1, &mut writer)?;
        // writer.shutdown()?;
        //
        // println!("Data written to '{}'.", file_path);
        //
        // let file_path = "/tmp/test_structs.zstd";
        //
        // let mut reader = File::open(file_path)?;
        // let table = zstd_framed::table::read_seek_table(&mut reader)?.unwrap();
        // println!("table={table:?}");
        //
        // // let mut structs_num = 0;let result: Vec<T> =
        // // let mut chunks = 0;
        //
        //
        // let mut reader = zstd_framed::ZstdReader::builder(reader).build().unwrap();
        // let elem_size= size_of::<MyStruct>();
        //
        // for frame in table.frames() {
        //     let mut buff = vec![0; frame.decompressed_size() as usize];
        //     reader.read_exact(&mut buff)?;
        //
        //     let result: Vec<MyStruct> = deserialize_buff(elem_size, &mut buff);
        //
        //     println!("buff.len={} result={result:?}", buff.len());
        // }

        // let mut buff = vec![0;72];
        // reader.read_exact(&mut buff)?;
        // let result:Vec<MyStruct> = deserialize_buff(elem_size, &mut buff);
        // println!("buff.len={} result={result:?}", buff.len());

        // let chunk_reader: ChunkReader<MyStruct> = ChunkReader::new(PathBuf::from(file_path).as_path())?;
        // for chunk_result in chunk_reader {
        //     match chunk_result {
        //         Ok(vec) => {
        //             for struct_data in vec {
        //                 structs_num += 1;
        //                 println!("Read struct: {:?}", struct_data);
        //             }
        //         }
        //         Err(e) => eprintln!("Error reading chunk: {}", e),
        //     }
        //     chunks += 1;
        // }

        // println!("chunks = {}, structs_num = {}", chunks, structs_num);



        // println!("Data written to '{}'.", file_path);

        // println!("MyStruct size:{}", size_of::<MyStruct>());

        // let mut structs_num = 0;
        // let mut chunks = 0;
        // // Read data back
        // let chunk_reader: ChunkReader<MyStruct> = ChunkReader::new(PathBuf::from(file_path).as_path())?;
        // for chunk_result in chunk_reader {
        //     match chunk_result {
        //         Ok(vec) => {
        //             for struct_data in vec {
        //                 structs_num +=1;
        //                 println!("Read struct: {:?}", struct_data);
        //             }
        //         }
        //         Err(e) => eprintln!("Error reading chunk: {}", e),
        //     }
        //     chunks+=1;
        // }
        //
        // println!("chunks={chunks} structs_num={structs_num}");

        Ok(())

    }
}