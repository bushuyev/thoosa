use crate::motif_tracker_store::{filter_groups_by_slices_count, IdSlice, MotifTrackerStore, SlicesMatch};
use crate::motifs_store::MotifsStore;
use async_trait::async_trait;
use anyhow::Context;
use bytemuck::{AnyBitPattern, Pod};
use log::{debug, trace};
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fmt::{Debug, Formatter};
use std::fs::File;
use std::io;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::mem::size_of;
use std::path::{Path, PathBuf};
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::{group_slices_try, ThoError};
use std::sync::Arc;
use parking_lot::Mutex;
use zstd::stream::write::Encoder;

const FILE_MAGIC: [u8; 8] = *b"THOSCHNK";
const FILE_VERSION: u32 = 1;
const CHUNK_HEADER_SIZE: usize = 24;
// Bound each chunk, including decoder window memory. Writers split larger batches
// so the format remains readable without trusting allocation sizes from headers.
const MAX_CHUNK_BYTES: usize = 64 * 1024 * 1024;
const MAX_COMPRESSED_BYTES: usize = MAX_CHUNK_BYTES + MAX_CHUNK_BYTES / 128 + 1024;

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
        let element_size = size_of::<T>();
        if element_size == 0 || element_size > MAX_CHUNK_BYTES {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "unsupported slice item size"));
        }
        for chunk in items.chunks(MAX_CHUNK_BYTES / element_size) {
            self.write_chunk(chunk)?;
        }
        Ok(())
    }

    fn write_chunk<T: Pod>(&mut self, items: &[T]) -> io::Result<()> {
        let bytes = bytemuck::cast_slice(items);

        self.scratch.clear();
        {
            let mut encoder = Encoder::new(&mut self.scratch, self.compression_level)?;
            encoder.write_all(bytes)?;
            encoder.finish()?;
        }

        if self.scratch.len() > MAX_COMPRESSED_BYTES {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "compressed chunk exceeds allocation limit"));
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

    fn shutdown(&self) -> io::Result<()> {
        self.file.sync_all()
    }
}

pub struct FileMotifTrackerStore {
    run_config_id: RunConfigId,
    channel: u32,
    writer: Arc<Mutex<ChunkedWriter>>,
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
            writer: Arc::new(Mutex::new(writer)),
            path: file_path,
        }))
    }
}

#[async_trait]
impl MotifTrackerStore for FileMotifTrackerStore {
    fn save_slices(&mut self, slices_pairs: Vec<SlicesMatch>) -> anyhow::Result<()> {
        trace!("MotifTrackerStore::save_slices: slices_pairs.len()={}", slices_pairs.len());

        let mut writer = self.writer.lock();
        writer.write_items(&slices_pairs)?;
        writer.flush()?;

        Ok(())
    }

    async fn load_slices_into_groups(&mut self, store: &dyn MotifsStore, min_slices_counts: Option<u32>, max_slices_counts: Option<u32>) -> anyhow::Result<usize> {
        debug!("MotifTrackerStore::load_slices_into_groups: max_slices_counts={max_slices_counts:?}");

        // Finalization, decompression, grouping, and filtering share one blocking
        // phase. The writer lock also protects against reuse after caller cancellation.
        let writer = Arc::clone(&self.writer);
        let path = self.path.clone();
        let (groups_len, groups) = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
            let writer = writer.lock();
            writer.shutdown()?;
            let mut slice_id = 0u32;
            let pairs = ChunkedSlicesReader::<SlicesMatch>::new(&path)?.flat_map(|chunk| -> Box<dyn Iterator<Item = io::Result<SlicesMatch>>> {
                match chunk {
                    Ok(chunk) => Box::new(chunk.into_iter().map(Ok)),
                    Err(error) => Box::new(std::iter::once(Err(error))),
                }
            }).map(|pair| {
                let pair = pair?;
                slice_id = slice_id.checked_add(1).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "too many slice pairs"))?;
                Ok::<_, io::Error>((IdSlice(slice_id, pair.0.0, pair.0.1), IdSlice(slice_id, pair.1.0, pair.1.1)))
            });
            let groups = group_slices_try(pairs)?;
            Ok((groups.len(), filter_groups_by_slices_count(min_slices_counts, max_slices_counts, groups)))
        }).await?
            // Keep the original I/O error available for downcasting while identifying its file.
            .with_context(|| format!("loading motif slice groups from {}", self.path.display()))?;
        let motif_ids = store.save_groups_as_motifs(self.run_config_id, self.channel as i64, groups).await?;

        debug!("MotifTrackerStore::load_slices_into_groups: motif_ids={motif_ids:?}");

        tokio::fs::remove_file(&self.path).await?;

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

        Some(self.read_chunk(header))
    }
}


impl<T: AnyBitPattern> ChunkedSlicesReader<T> {
    fn read_chunk(&mut self, header: [u8; CHUNK_HEADER_SIZE]) -> io::Result<Vec<T>> {
        let decompressed_len = u64::from_le_bytes(header[0..8].try_into().unwrap());
        let item_count = u64::from_le_bytes(header[8..16].try_into().unwrap());
        let compressed_len = u64::from_le_bytes(header[16..24].try_into().unwrap());
        let element_size = size_of::<T>() as u64;
        // Validate all header relationships before allocating or entering zstd.
        if element_size == 0 || item_count == 0 || decompressed_len > MAX_CHUNK_BYTES as u64
            || item_count.checked_mul(element_size) != Some(decompressed_len)
            || compressed_len == 0 || compressed_len > MAX_COMPRESSED_BYTES as u64 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid or oversized slice chunk"));
        }
        let remaining = self.reader.get_ref().metadata()?.len().saturating_sub(self.reader.stream_position()?);
        if compressed_len > remaining {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated chunk payload"));
        }
        let compressed_len = usize::try_from(compressed_len).map_err(io::Error::other)?;
        let decompressed_len = usize::try_from(decompressed_len).map_err(io::Error::other)?;
        let mut compressed = Vec::new();
        compressed.try_reserve_exact(compressed_len).map_err(io::Error::other)?;
        compressed.resize(compressed_len, 0);
        self.reader.read_exact(&mut compressed)?;
        let mut decoder = zstd::stream::read::Decoder::new(compressed.as_slice())?;
        decoder.window_log_max(26)?;
        let mut decompressed = Vec::new();
        decompressed.try_reserve_exact(decompressed_len + 1).map_err(io::Error::other)?;
        // Read one extra byte to detect a lying header without decoding the full frame.
        decoder.take(decompressed_len as u64 + 1).read_to_end(&mut decompressed)?;
        if decompressed.len() != decompressed_len {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "unexpected decompressed chunk length"));
        }
        bytemuck::try_cast_slice(&decompressed).map(|slice: &[T]| slice.to_vec())
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, format!("invalid slice payload: {error}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TestFile(PathBuf);
    impl TestFile {
        fn new() -> io::Result<(Self, File)> {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!("thoosa-chunks-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
            let file = File::options().create_new(true).read(true).write(true).open(&path)?;
            Ok((Self(path), file))
        }
        fn payload(&self, decoded: u64, count: u64, compressed_len: u64, payload: &[u8]) -> io::Result<()> {
            let mut file = File::create(&self.0)?;
            file.write_all(&FILE_MAGIC)?;
            file.write_all(&FILE_VERSION.to_le_bytes())?;
            file.write_all(&[0; 4])?;
            for value in [decoded, count, compressed_len] { file.write_all(&value.to_le_bytes())?; }
            file.write_all(payload)
        }
    }
    impl Drop for TestFile {
        fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
    }

    #[tokio::test]
    async fn test_grouping_reports_corrupt_and_truncated_chunks_with_path() -> anyhow::Result<()> {
        for case in ["corrupt payload", "truncated payload", "truncated header"] {
            let (path, file) = TestFile::new()?;
            let compressed_len = if case == "truncated payload" { 5 } else { 4 };
            path.payload(size_of::<SlicesMatch>() as u64, 1, compressed_len, &[1, 2, 3, 4])?;
            let mut tracker = FileMotifTrackerStore {
                run_config_id: 1,
                channel: 0,
                writer: Arc::new(Mutex::new(ChunkedWriter::new(file, 1)?)),
                path: path.0.clone(),
            };
            if case == "truncated header" { tracker.writer.lock().file.set_len(16 + 8)?; }
            // No save expectation: malformed input must return its path and original
            // I/O cause, without persisting partial groups or removing the source file.
            let store = crate::motifs_store::MockMotifsStoreBuilder::new().build();
            let error = tracker.load_slices_into_groups(store.as_ref(), None, None).await.unwrap_err();
            let source = error.downcast_ref::<io::Error>().expect("original I/O error");
            if case != "corrupt payload" { assert_eq!(source.kind(), io::ErrorKind::UnexpectedEof, "{case}"); }
            assert!(error.to_string().contains(&path.0.display().to_string()), "missing path for {case}: {error:#}");
            assert!(path.0.exists(), "source file must be retained for {case}");
        }
        Ok(())
    }

    #[test]
    fn test_chunk_roundtrip_and_large_write_splitting() -> io::Result<()> {
        let (path, file) = TestFile::new()?;
        let mut writer = ChunkedWriter::new(file, 1)?;
        // Large writes remain readable under the same per-chunk allocation limit.
        let input = vec![7u8; MAX_CHUNK_BYTES + 1];
        writer.write_items(&input)?;
        writer.shutdown()?;
        let mut reader = ChunkedSlicesReader::<u8>::new(&path.0)?;
        assert_eq!(reader.next().expect("first chunk")?, input[..MAX_CHUNK_BYTES]);
        assert_eq!(reader.next().expect("second chunk")?, vec![7]);
        assert!(reader.next().is_none());
        Ok(())
    }

    #[test]
    fn test_chunk_rejects_invalid_lengths_before_allocation() -> io::Result<()> {
        let (path, _file) = TestFile::new()?;
        let compressed = zstd::encode_all(&[0u8; 4][..], 1)?;
        for (decoded, count, size) in [
            (u64::MAX, 1, compressed.len() as u64),
            (4, 1, u64::MAX),
            (4, u64::MAX, compressed.len() as u64),
            (4, 2, compressed.len() as u64),
            (MAX_CHUNK_BYTES as u64 + 4, MAX_CHUNK_BYTES as u64 / 4 + 1, compressed.len() as u64),
            (0, 0, compressed.len() as u64),
        ] {
            path.payload(decoded, count, size, &compressed)?;
            let error = ChunkedSlicesReader::<u32>::new(&path.0)?.next().expect("chunk").unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        }
        path.payload(4, 1, compressed.len() as u64 + 1, &compressed)?;
        assert_eq!(ChunkedSlicesReader::<u32>::new(&path.0)?.next().expect("chunk").unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
        path.payload(4, 1, compressed.len() as u64, &compressed)?;
        assert!(ChunkedSlicesReader::<[u8; 0]>::new(&path.0)?.next().expect("chunk").is_err());
        Ok(())
    }

    #[test]
    fn test_decompression_stops_at_declared_length_and_rejects_corruption() -> io::Result<()> {
        let (path, _file) = TestFile::new()?;
        let compressed = zstd::encode_all(&vec![0u8; 1024 * 1024][..], 1)?;
        // A tiny declared output cannot make the decoder allocate the expanded frame.
        path.payload(4, 1, compressed.len() as u64, &compressed)?;
        assert_eq!(ChunkedSlicesReader::<u32>::new(&path.0)?.next().expect("chunk").unwrap_err().kind(), io::ErrorKind::InvalidData);
        path.payload(4, 1, 4, &[1, 2, 3, 4])?;
        assert!(ChunkedSlicesReader::<u32>::new(&path.0)?.next().expect("chunk").is_err());
        let compressed = zstd::encode_all(&[0u8; 4][..], 1)?;
        let truncated = &compressed[..compressed.len() - 1];
        path.payload(4, 1, truncated.len() as u64, truncated)?;
        assert!(ChunkedSlicesReader::<u32>::new(&path.0)?.next().expect("chunk").is_err());
        Ok(())
    }
}
