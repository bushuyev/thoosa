use crate::motif_tracker_store::{filter_groups_by_slices_count, IdSlice, MotifTrackerStore, SlicesMatch};
use crate::motifs_store::MotifsStore;
use async_trait::async_trait;
use bytemuck::{AnyBitPattern, Pod};
use log::{debug, trace, warn};
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::{fs, io};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::{group_slices, ThoError};
use zstd::stream::write::Encoder;

const FILE_MAGIC: [u8; 8] = *b"THOSCHNK";
const FILE_VERSION: u32 = 1;
const CHUNK_HEADER_SIZE: usize = 24;

// The file-backed tracker remains byte-compatible while living beside the motif APIs it persists.
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
    run_config_id: RunConfigId,
    channel: u32,
    writer: ChunkedWriter,
    path: PathBuf,
}

impl Debug for FileMotifTrackerStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileMotifTrackerStore")
            .field("run_config_id", &self.run_config_id)
            .field("channel", &self.channel)
            .field("path", &self.path)
            .finish()
    }
}

impl FileMotifTrackerStore {
    pub fn new<P: AsRef<Path> + Sync + Send + Debug>(path: P, run_config_id: RunConfigId, channel: u32) -> anyhow::Result<Box<dyn MotifTrackerStore>> {
        let file_path = path.as_ref().join(format!("{run_config_id}_{channel}_slices.zstd").as_str());

        let file = std::fs::OpenOptions::new().create(true).read(true).write(true).open(&file_path)?;

        debug!("FileMotifTrackerStore::new: path={path:?} run_config_id={run_config_id} channel={channel} file_path={file_path:?}");

        let writer = ChunkedWriter::new(file, 3)?;

        Ok(Box::new(Self {
            run_config_id,
            channel,
            writer,
            path: file_path,
        }))
    }
}

#[async_trait]
impl MotifTrackerStore for FileMotifTrackerStore {
    fn save_slices(&mut self, slices_pairs: Vec<SlicesMatch>) -> anyhow::Result<()> {
        trace!("MotifTrackerStore::save_slices: slices_pairs.len()={}", slices_pairs.len());

        self.writer.write_items(&slices_pairs)?;
        self.writer.flush()?;

        Ok(())
    }

    async fn load_slices_into_groups(&mut self, store: &dyn MotifsStore, min_slices_counts: Option<u32>, max_slices_counts: Option<u32>) -> anyhow::Result<usize> {
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
        let motif_ids = store.save_groups_as_motifs(self.run_config_id, self.channel as i64, filter_groups_by_slices_count(min_slices_counts, max_slices_counts, groups)).await?;

        debug!("MotifTrackerStore::load_slices_into_groups: motif_ids={motif_ids:?}");

        fs::remove_file(&self.path)?;

        Ok(groups_len)
    }

    fn run_config_id(&self) -> RunConfigId {
        self.run_config_id
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
                    debug!("ChunkedSlicesReader::next: end of chunk stream self.file_path={:?}", self.file_path);

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
