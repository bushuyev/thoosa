use lewton::inside_ogg::OggStreamReader;
use log::debug;
use ogg::reading::PacketReader;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::fs::{File, read_dir};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use tch::Tensor;
use thoosa_shared::ThoosaError;

pub fn ogg_to_tensor<P: AsRef<Path> + Debug>(
    file: P,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    debug!("ogg_to_tensor: file={file:?}");

    let file = File::open(file)?;
    let reader = BufReader::new(file);
    let mut ogg_reader = OggStreamReader::new(reader)?;

    let num_channels = ogg_reader.ident_hdr.audio_channels as usize;
    let mut channels: Vec<Vec<_>> = vec![Vec::new(); num_channels];

    while let Some(pck_samples) = ogg_reader.read_dec_packet_itl()? {
        for (i, sample) in pck_samples.into_iter().enumerate() {
            let channel = i % num_channels;

            channels[channel].push(sample);
            //TODO convert i16 to f32 in range [-1.0, 1.0]?
            //channels[channel].push(sample as f32 / i16::MAX as f32);
        }
    }

    let tensor = Tensor::stack(
        &channels
            .into_iter()
            .map(|channel| Tensor::from_slice(channel.as_slice()))
            .collect::<Vec<Tensor>>(),
        0,
    );

    debug!("ogg_to_tensor: tensor.size={:?}", tensor.size());

    Ok(tensor)
}

#[allow(dead_code)]
pub fn clc_tracks<P: AsRef<Path>>(file: P) -> Result<usize, Box<dyn std::error::Error>> {
    let file = File::open(file)?;
    let reader = BufReader::new(file);
    let mut pr = PacketReader::new(reader);

    let mut serial_numbers = HashSet::new();

    while let Some(packet) = pr.read_packet()? {
        serial_numbers.insert(packet.stream_serial());
    }

    // println!("Found {} logical stream(s).", serial_numbers.len());
    // for (i, serial) in serial_numbers.iter().enumerate() {
    //     println!("  Stream #{}: serial={}", i + 1, serial);
    // }

    Ok(serial_numbers.len())
}

pub fn labeled_resourses<P: Into<PathBuf>>(
    _path: P,
) -> Result<HashMap<String, Vec<(String, f32)>>, ThoosaError> {
    /*let df_csv = CsvReadOptions::default()
        .with_ignore_errors(true)
        .with_has_header(true)
        .with_parse_options(CsvParseOptions::default().with_try_parse_dates(true))
        .try_into_reader_with_file_path(Some(<P>::into(path).join("train.csv")))?
        .finish()?;
    debug!("{}", df_csv);

    // debug!("df={:#?}", df.collect());*/

    let mut map = HashMap::new();
    let train_audio_dir =
        PathBuf::from("/home/bu/dvl/data/kaggle/BirdCLEF_2025").join("train_audio"); //<P>::into(/*path*/
    debug!("train_audio_dir={train_audio_dir:?}");

    for label_dir in read_dir(train_audio_dir)? {
        debug!("label_dir={label_dir:?}");
        let label_dir = label_dir?;
        let _meta = label_dir.metadata()?;
        let label = label_dir
            .path()
            .file_stem()
            .expect("file_stem")
            .to_str()
            .expect("label")
            .to_string();

        for file in read_dir(label_dir.path())? {
            let file = file?;
            debug!("    file={file:?}");

            map.insert(
                format!(
                    "{}_{}", //TODO train_audio?
                    label.clone(),
                    file.path()
                        .file_name()
                        .expect("zzz")
                        .to_str()
                        .expect("path")
                        .to_string(),
                ),
                vec![(label.clone(), 1.0)],
            );
        }
    }

    debug!("map={map:?}");

    Ok(map)
}

pub fn resource_to_tensor(source_path: &dyn AsRef<Path>, key: &str) -> Result<Tensor, ThoosaError> {
    Ok(ogg_to_tensor(
        source_path
            .as_ref()
            .join("train_audio")
            .join(key.replace("_", "/")),
    )?)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, dead_code, unused_variables)]
    use crate::data::{clc_tracks, labeled_resourses, ogg_to_tensor};
    use crate::labels::LABELS;
    use log::LevelFilter::Trace;
    use log::{debug, warn};
    use parking_lot::{Mutex, RawRwLock, RwLock};
    use plotters::prelude::{BLACK, RED};
    use std::fs;
    use std::fs::OpenOptions;
    use std::io::{BufWriter, Write};
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::Arc;
    use std::time::UNIX_EPOCH;
    use tch::Device::Cpu;
    use tch::Kind::Float;
    use tch::{Device, IndexOp, TchError, Tensor};
    use tempfile::NamedTempFile;
    use thoosa_common::ResourceLoader;
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::runs::Simple;
    use thoosa_common::utils::{
        from, hist_1_dim, lin_t, plot_series, skewness_kurtosis_1_dim, vec_f32_from,
    };
    use thoosa_common::ws::FsWorkspace;
    use thoosa_common::ws::Workspace;
    use thoosa_main::cli::Commands::{FindMotifs, SelectMotifs, TrainFinalNet, TrainLabelNets};
    use thoosa_main::cli::{Cli, import_resources};
    use thoosa_main::data::ParquetLoader;
    use thoosa_motifs::chunks::ChunkOfSlicedTensors;
    use thoosa_motifs::distance::MaskMaker::LeQuantile;
    use thoosa_motifs::distance::clc_dist_d2_and_mask;
    use thoosa_motifs::find::{SelectMotifsParams, find_motifs_by_euc_dist, plot_resource};
    use thoosa_shared::resource::{Resource, ResourceStore};
    use thoosa_shared::test_utils::{are_close, init_test_logging, set_panic_hook};
    use thoosa_shared::{ResourceType, ThoosaError, cdt_for_dir, recurse_files};
    use thoosa_store::SqlStore;

    #[ctor::ctor]
    fn init() {
        init_test_logging();
    }

    #[test]
    pub fn test_read_oog() -> Result<(), ThoosaError> {
        use lewton::inside_ogg::OggStreamReader;
        use std::fs::File;
        use std::io::BufReader;

        let file =
            File::open("/home/bu/dvl/data/kaggle/BirdCLEF_2025/train_audio/65349/iNat109039.ogg")?;
        let reader = BufReader::new(file);

        let mut ogg_reader = OggStreamReader::new(reader)?;

        let ident_hdr = &ogg_reader.ident_hdr;

        println!("Sample rate: {} Hz", ident_hdr.audio_sample_rate);
        println!("Channels: {}", ident_hdr.audio_channels);
        println!("Bitrate (nominal): {} bps", ident_hdr.bitrate_nominal);
        println!("Bitrate (max): {} bps", ident_hdr.bitrate_maximum);
        println!("Bitrate (min): {} bps", ident_hdr.bitrate_minimum);

        // Get comment header (Vorbis comment metadata)
        let comment_hdr = &ogg_reader.comment_hdr;
        println!("Vendor: {}", comment_hdr.vendor);

        println!("User Comments:");
        for comment in &comment_hdr.comment_list {
            println!("  {:?}", comment);
        }

        while let Some(pck_samples) = ogg_reader.read_dec_packet_itl()? {
            // pck_samples is Vec<i16> in interleaved format (e.g., stereo = L, R, L, R, ...)
            // Convert to float [-1.0, 1.0]
            let float_samples: Vec<f32> = pck_samples
                .into_iter()
                .map(|s| s as f32 / i16::MAX as f32)
                .collect();

            // Process your float samples here (e.g., stream to DSP, etc.)
            println!("Got {} float samples", float_samples.len());
        }

        Ok(())
    }

    // #[test]
    pub fn test_all() -> Result<(), ThoosaError> {
        let files = recurse_files(&"/home/bu/dvl/data/kaggle/BirdCLEF_2025/", &|path| {
            path.extension().map(|v| v.eq("ogg")).unwrap_or(false)
        })?;
        let files_cnt = files.len();
        debug!("files_cnt={files_cnt}");

        for (i, file) in files.iter().enumerate() {
            if i % 100 == 0 {
                debug!("{i} files done");
            }

            let tracks = clc_tracks(&file)?;
            // let tracks = ogg_to_tensor(&file)?;
            if tracks != 1 {
                debug!("file: {file:?} tracks={tracks}");
            }
        }

        debug!("{files_cnt} done");

        Ok(())
    }

    #[test]
    fn test_ogg_to_tensor() -> Result<(), ThoosaError> {
        let base = Path::new("/tmp").join(cdt_for_dir());
        fs::create_dir_all(&base)?; //TODO util

        // set_panic_hook();
        let sample_rate = 50; //44100;
        let duration_secs = 1;
        let freq = 5.; //440.0;
        let total_samples = sample_rate * duration_secs;

        let left: Vec<i16> = (0..total_samples)
            .map(|n| {
                let t = n as f32 / sample_rate as f32;
                (t * freq * std::f32::consts::TAU).sin() * i16::MAX as f32
            })
            .map(|s| s as i16)
            .collect();

        let right = left.clone(); // simple mono -> stereo

        let ogg_file = base.join("test_out.ogg");
        let sample = vec![left, right];
        write_ogg_from_pcm_i16(&sample, sample_rate, ogg_file.to_str().unwrap())?;

        debug!("ogg_file={ogg_file:?}");
        debug!("sample={sample:?}");

        let t = ogg_to_tensor(&ogg_file)?;

        debug!("t={t}");

        let data: Vec<Vec<i16>> = from(&t);

        debug!("data={t}");

        assert!(are_close(
            &sample[0],
            &data[0],
            (i16::MAX as f64 * 1e-3) as i16
        ));
        assert!(are_close(
            &sample[1],
            &data[1],
            (i16::MAX as f64 * 1e-3) as i16
        ));

        Ok(())
    }

    pub fn write_ogg_from_pcm_i16(
        channels: &Vec<Vec<i16>>,
        sample_rate: u32,
        output_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let num_channels = channels.len();

        // Validate that all channels have the same length
        let num_samples = channels.first().ok_or("No channels provided")?.len();

        if !channels.iter().all(|ch| ch.len() == num_samples) {
            return Err("All channels must have the same number of samples".into());
        }

        // Create temp WAV file
        let mut tmp_wav = NamedTempFile::new()?;
        let wav_path = tmp_wav.path().to_owned();

        let spec = hound::WavSpec {
            channels: num_channels as u16,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::new(BufWriter::new(&mut tmp_wav), spec)?;

        // Interleave and write samples
        for i in 0..num_samples {
            for ch in 0..num_channels {
                writer.write_sample(channels[ch][i])?;
            }
        }

        writer.finalize()?;

        // Convert to Ogg using ffmpeg
        let status = Command::new("ffmpeg")
            .args(&["-y", "-i"])
            .arg(wav_path)
            .arg(output_path)
            .status()?;

        if !status.success() {
            return Err("ffmpeg failed to encode to ogg".into());
        }

        Ok(())
    }

    #[test]
    fn test_labeled_resourses() -> Result<(), ThoosaError> {
        let resourses = labeled_resourses("test_data/")?;

        assert_eq!(28564, resourses.len());

        Ok(())
    }

    // #[test]
    fn test_plot_resource() -> Result<(), ThoosaError> {
        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            "/tmp/ws_test_clef",
            1,
            999,
            32000,
        )?));

        let store = Arc::new(RwLock::new(SqlStore::from_path(ws.read().store_path())?));

        plot_resource(
            Simple::new(ResourceType::List(vec![
                // "piepuf1_XC769654.ogg".into(),
                // "crbtan1_XC137679.ogg".into(),
                // "whfant1_XC669571.ogg".into(),
                // "blhpar1_XC116438.ogg".into(),
                "bugtan_XC946265.ogg".into(),
                "bugtan_XC946266.ogg".into(),
                "bugtan_XC946267.ogg".into(),
                "bugtan_XC953345.ogg".into(),
                "bugtan_XC959312.ogg".into(),
                "bugtan_XC959322.ogg".into(),
                "bugtan_XC962679.ogg".into(),
            ])),
            &ws,
            &store,
        )?;

        Ok(())
    }
}
