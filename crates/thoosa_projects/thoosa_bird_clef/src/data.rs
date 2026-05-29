use lewton::inside_ogg::OggStreamReader;
use log::{debug, warn};
use ogg::reading::PacketReader;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::fs::{read_dir, File};
use std::io::BufReader;
use std::path::Path;
use polars::prelude::{col, lit, CsvParseOptions, CsvReadOptions, IntoLazy, SerReader};
use tch::Kind::{BFloat16, Half};
use tch::{Kind, Tensor};
use thoosa_shared::ThoError;

pub fn ogg_to_tensor<P: AsRef<Path> + Debug>(
    file: P, kind: Kind,
) -> Result<Tensor, ThoError> {
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

    //gpt Navigation note: import-resources persists this channel-major Half tensor to ws/tensors; later RawTensorLoader reads that file instead of decoding OGG again.
    let tensor = Tensor::stack(
        &channels
            .into_iter()
            .map(|channel| Ok(
                Tensor::f_from_slice(channel.as_slice())?
                       .f_totype(kind)?
            ))
            .collect::<Result<Vec<Tensor>, ThoError>>()?,
        0,
    );

    debug!("ogg_to_tensor: tensor={:?}", tensor);

    Ok(tensor)
}

#[allow(dead_code)]
pub fn clc_tracks<P: AsRef<Path>>(file: P) -> Result<usize, ThoError> {
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

pub fn labeled_resourses(
    path: impl AsRef<Path>,
) -> Result<HashMap<String, Vec<(String, f32)>>, ThoError> {
    // Navigation note: keys are "{label}_{filename}" so resource_to_tensor can map them back to train_audio/{label}/{filename}. Keep this mapping in sync with Resource::file_stem consumers.
    let df_csv = CsvReadOptions::default()
        .with_ignore_errors(true)
        .with_has_header(true)
        .with_parse_options(CsvParseOptions::default().with_try_parse_dates(true))
        .try_into_reader_with_file_path(Some(path.as_ref().join("train.csv")))?
        .finish()?
        .lazy();


    debug!("df={:?}", df_csv.clone().limit(10).collect()?);

    let mut map = HashMap::new();

    let train_audio_dir = path.as_ref().join("train_audio");

    debug!("clef/labeled_resourses: train_audio_dir={train_audio_dir:?}");
    let mut counts = HashMap::new();
    let df_csv_ref = &df_csv;
    for label_dir in read_dir(train_audio_dir)? {

        let label_dir = label_dir?;
        let label = label_dir
            .path()
            .file_stem()
            .expect("file_stem")
            .to_str()
            .expect("label")
            .to_string();

        debug!("label_dir={label_dir:?} label={:?}", label);


        for file in read_dir(label_dir.path())? {
            let path_buf = file?.path();
            let file = path_buf.file_stem().expect("file_name").to_str().expect("file");
            let file_name = path_buf.file_name().expect("file_name").to_str().expect("file_name").to_string();
            let csv_filename = format!("{label}/{file_name}");

            let key = format!("{label}_{file_name}");

            let secondary_labels = df_csv_ref
                .clone()
                .filter(col("filename").eq(lit(csv_filename.as_str())))
                .select([col("secondary_labels")])
                .collect()?;

            let secondary_labels: Vec<String> = match secondary_labels
                .column("secondary_labels")?
                .str()?
                .into_iter()
                .next()
                .flatten() {
                Some(s) => parse_secondary_labels(s),
                None => Vec::new(),
            };
            debug!("clef/labeled_resourses: file={file:?} csv_filename={csv_filename} key={key} secondary_labels={secondary_labels:?}");
            let secondary_labels = secondary_labels.into_iter().filter(|secondary_label|{
                let r = !secondary_label.eq(&label);
                if !r {
                    warn!("clef/labeled_resourses: secondary_label={secondary_label} eq {label}");
                }
                r
            }).collect::<Vec<_>>();

            let labels = if secondary_labels.is_empty() {
                vec![(label.clone(), 1.0)]
            } else {
                let secondary_labels = secondary_labels.into_iter().filter(|secondary_label|!secondary_label.eq(&label)).collect::<Vec<_>>();
                let secondary_p = 0.1/secondary_labels.len() as f32;
                //train.cvs doesn't contain probabilities of the secondary labels
                std::iter::once((label.clone(), 0.9)).chain(secondary_labels.into_iter().map(|secondary_label|{
                    (secondary_label, secondary_p)
                })).collect()
            };

            debug!("clef/labeled_resourses: file={file:?} key={key} labels={labels:?}");

            *counts.entry(key.clone()).or_insert(0) +=1;

            map.insert(
                key,
                labels,
            );
        }
    }
    for e in counts {
        if e.1 > 1 {
            warn!("key {} has {} entries", e.0, e.1);
        }
    }

    debug!("map={map:?}");

    Ok(map)
}

fn parse_secondary_labels(s: &str) -> Vec<String> {
    let inner = s
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']');

    if inner.is_empty() {
        return Vec::new();
    }

    inner
        .split(',')
        .map(|x| x.trim().trim_matches('\'').trim_matches('"').to_string())
        .filter(|x| !x.is_empty())
        .collect()
}

pub fn resource_to_tensor(source_path: &dyn AsRef<Path>, key: &str) -> Result<Tensor, ThoError> {
    //gpt The key is intentionally reversible: labeled_resourses stores "class_file.ogg", and this restores "class/file.ogg".
    Ok(ogg_to_tensor(
        source_path
            .as_ref()
            .join("train_audio")
            .join(key.replace("_", "/")), BFloat16,
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
    use tch::Kind::{BFloat16, Float};
    use tch::{Device, IndexOp, TchError, Tensor};
    use tempfile::NamedTempFile;
    use thoosa_common::dims::Spectorgram;
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::runner::common::RunCtx;
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use thoosa_common::runs::Simple;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::StftRunConfig;
    use thoosa_common::utils::{from, hist_1_dim, lin_t, plot_series, plot_spectrogram, skewness_kurtosis_1_dim, to_vec_f32, z_norm};
    use thoosa_common::ws::FsWorkspace;
    use thoosa_common::ws::Workspace;
    use thoosa_main::cli::Commands::{FindMotifs, SelectMotifs, TrainFinalNet, TrainLabelNets};
    use thoosa_main::cli::{import_resources, Cli};
    use thoosa_main::tasks::stft::Stft;
    use thoosa_motifs::distance::MaskMaker::LeQuantile;
    use thoosa_motifs::distance::clc_d2_similarity_and_mask;
    use thoosa_motifs::find::{plot_resource, SelectMotifsParams};
    use thoosa_shared::resource::{Resource, ResourceStore};
    use thoosa_shared::test_utils::{are_close, init_test_logging};
    use thoosa_shared::{cdt_for_dir, recurse_files, sin_v_f, ResourceType, ThoError};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_store::SqlStore;
    use thoosa_with_data::with_data;

    #[ctor::ctor]
    fn init() {
        init_test_logging();
    }

    #[test]
    pub fn test_read_oog() -> Result<(), ThoError> {
        use lewton::inside_ogg::OggStreamReader;
        use std::fs::File;
        use std::io::BufReader;

        let file =
            File::open("/home/bu/dvl/data/kaggle/birdclef-2026/train_audio/555123/iNat336366.ogg")?;
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

    #[test]
    pub fn test_all() -> Result<(), ThoError> {
        let files = recurse_files(&"/home/bu/dvl/data/kaggle/birdclef-2026", &|path| {
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
    fn test_ogg_to_tensor() -> Result<(), ThoError> {
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

        let t = ogg_to_tensor(&ogg_file, BFloat16)?;

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
    ) -> Result<(), ThoError> {
        let num_channels = channels.len();

        // Validate that all channels have the same length
        let num_samples = channels
            .first()
            .ok_or_else(|| ThoError::new("no channels provided"))?
            .len();

        if !channels.iter().all(|ch| ch.len() == num_samples) {
            return Err(ThoError::new(
                "all channels must have the same number of samples",
            ));
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
            return Err(ThoError::new("ffmpeg failed to encode to ogg"));
        }

        Ok(())
    }

    #[test]
    fn test_labeled_resourses() -> Result<(), ThoError> {
        unsafe {
            std::env::set_var("POLARS_FMT_STR_LEN", "1000");
            std::env::set_var("POLARS_FMT_TABLE_WIDTH", "-1");
            std::env::set_var("POLARS_FMT_MAX_COLS", "-1");
            std::env::set_var("POLARS_FMT_MAX_ROWS", "-1");
        }

        let resourses = labeled_resourses("test_data/")?;

        debug!("resourses={resourses:?}");

        assert_eq!(5, resourses.len());
        assert_eq!(Some(&vec![("555123".to_string(), 1.0)]), resourses.get("555123_iNat336366.ogg"));
        assert_eq!(
            Some(&vec![
                ("24279".to_string(), 0.9),
                ("326272".to_string(), 0.05),
                ("67107".to_string(), 0.05),
            ]),
            resourses.get("24279_XC1039628.ogg"),
        );

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_plot_resource() -> Result<(), ThoError> {
        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            "/tmp/ws_clef",
            1,
            999,
            32000,
        )?));

        let store = Arc::new(SqlStore::from_path(ws.read().store_path()).await?);

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
        )
        .await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn plot_spec() -> Result<(), ThoError>{
        GlobalCtx::init_devices(2);
        let audio_path = PathBuf::from("test_data/train_audio/");

        let file = "555123/iNat336366.ogg";
        resource_spec(file, ogg_to_tensor(audio_path.join(&file), Float)?.view((1, -1))).await?;

        let file = "555123/iNat859875.ogg";
        resource_spec(file, ogg_to_tensor(audio_path.join(&file), Float)?.view((1, -1))).await?;

        let file = "555123/iNat1370876.ogg";
        resource_spec(file, ogg_to_tensor(audio_path.join(&file), Float)?.view((1, -1))).await?;

        let file = "116570/iNat1460166.ogg";
        resource_spec(file, ogg_to_tensor(audio_path.join(&file), Float)?.view((1, -1))).await?;

        let sig_t = Tensor::from_slice(sin_v_f(160_000, 8000.0, 32_000.0).as_slice());
        resource_spec("test", sig_t.view((1,-1))).await?;

        Ok(())
    }

    async fn resource_spec(file: &str, tensor: Tensor) -> Result<(), ThoError> {

        let t = z_norm(&tensor, 1)?;

        let stft = Arc::new(Stft::new(StftRunConfig::new(16000, 4000.into(), 4000.into(), true, true, true))?);
        let mut ctx = RunCtxImpl::new(1, "plot_spec");
        let spec = stft.run(Arc::new(DevicesWrap::wrap(ThoTensor::new("".into(), t)?)), ctx.branch(false)).await?;

        with_data!(ctx.branch(false), spec; |_, spec:&ThoTensor<Spectorgram, SynKey>|{
            let spec = spec.as_ref();
            debug!("spec = {spec:?}");
            plot_spectrogram(
                &spec.squeeze_dim(0),
                format!("/tmp/plots/spectrogram_znorm_{}.png", file.replace("/", "_")),
                1200,
                600,
            )?;

            Ok(())
        })?;
        Ok(())
    }
}
