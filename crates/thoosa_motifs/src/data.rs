use anyhow::Context as _;
use thoosa_common::tch_error::tch_result;
use crate::motifs_store::MotifsStore;
use std::marker::PhantomData;
use std::path::Path;

use log::debug;
use polars::prelude::*;
use polars::prelude::{LazyFrame, NamedFrom, ScanArgsParquet, Series};

use tch::{Device, Tensor};

use thoosa_shared::resource::Resource;
use thoosa_shared::tracked_await::track_await;
use tracing::Instrument;

#[derive(Default)]
pub enum Set {
    #[default]
    Train,
    Valid,
}

pub trait Splitter {
    type Type;

    fn split_to(&self, v: &Self::Type) -> Set;

    fn init(&self, df: LazyFrame) -> anyhow::Result<LazyFrame> {
        Ok(df.clone())
    }
}

pub struct AllTrainSplitter<E> {
    _pd: PhantomData<E>,
}

impl<E> Splitter for AllTrainSplitter<E> {
    type Type = E;

    fn split_to(&self, _v: &Self::Type) -> Set {
        Set::Train
    }
}

pub fn series_to_tensor(s: &Series) -> anyhow::Result<Tensor> {
    Ok(match s.dtype() {
        // DataType::UInt8 => {}
        // DataType::UInt16 => {}
        DataType::UInt32 => Tensor::f_from_slice(
            s.u32()?
                .to_vec()
                .iter()
                .map(|v| v.unwrap_or_default() as i32)
                .collect::<Vec<i32>>()
                .as_slice(),
        )?,
        DataType::UInt64 => Tensor::f_from_slice(
            s.i64()?
                .to_vec()
                .iter()
                .map(|v| v.unwrap_or_default() as i32)
                .collect::<Vec<i32>>()
                .as_slice(),
        )?,
        // DataType::Int8 => {}
        // DataType::Int16 => {}
        DataType::Int32 => Tensor::f_from_slice(
            s.u32()?
                .to_vec()
                .iter()
                .map(|v| v.unwrap_or_default() as i32)
                .collect::<Vec<i32>>()
                .as_slice(),
        )?,
        // DataType::Int64 => {
        //     // tensors.push(Tensor::from_slice(s.i64()?.to_vec().iter().map(|v| v.unwrap_or_default()).collect::<Vec<i64>>().as_slice()));
        // }
        DataType::Float32 => Tensor::f_from_slice(
            s.f32()?
                .to_vec()
                .iter()
                .map(|v| v.unwrap_or_default())
                .collect::<Vec<f32>>()
                .as_slice(),
        )?,
        DataType::Float64 => Tensor::f_from_slice(
            s.f64()?
                .to_vec()
                .iter()
                .map(|v| v.unwrap_or_default() as f32)
                .collect::<Vec<f32>>()
                .as_slice(),
        )?,
        _ => {
            todo!()
        }
    })
}


#[allow(dead_code)]
fn split_frame_by_offsets<I>(mut df: DataFrame, offsets_sec: I) -> PolarsResult<Vec<DataFrame>>
where
    I: IntoIterator<Item = i64> + Clone,
{
    let shape = df.shape();
    let it_1 = offsets_sec.clone().into_iter();
    let mut it_2 = offsets_sec
        .into_iter()
        .chain(std::iter::once(shape.0 as i64));
    it_2.next();

    let ids = it_2
        .zip(it_1)
        .map(|(v2, v1)| v2 - v1)
        .enumerate()
        .flat_map(|(i, frag)| std::iter::repeat(i as i64).take(frag as usize))
        .collect::<Vec<i64>>();

    assert!(ids.len() > 1);

    df.with_column(Series::from_vec("frag_id".into(), ids).into())?;

    df.partition_by::<_, &str>(["frag_id"], false)
}

pub async fn load_to_logits_tensors(
    group: &str,
    store: Arc<dyn MotifsStore>,
) -> anyhow::Result<Vec<(Resource, Vec<f32>)>> {
    Ok(track_await!(store
        .list_resource_random_in_group(group)).await.with_context(|| format!("loading resources in group {group}"))?
        .into_iter()
        .map(move |(resource_id, labels_p)| {
            debug!(
                "load_to_logits_tensors: resourceId={resource_id:?} labels_p={:?}",
                labels_p
            );
            (
                resource_id,
                labels_p, //Box::new(Tensor::from_slice(labels_p.as_slice()).to_device(*device))
            )
        })
        .collect())
}

pub fn parquet_to_tensor_by_path(
    path: impl AsRef<Path>,
    device: Device,
) -> anyhow::Result<Tensor> {
    let df = read_parquet_to_df(path)?;

    let tensor = Tensor::f_vstack(
        &df.columns()
            .into_iter()
            .map(|s| series_to_tensor(s.as_series().context("Parquet column is not a materialized series")?))
            .collect::<anyhow::Result<Vec<Tensor>>>()?,
    )?;
    Ok(tensor.f_set_requires_grad(false)?.f_to_device(device)?)
}

pub fn read_parquet_to_df(path: impl AsRef<Path>) -> anyhow::Result<DataFrame> {
    let scan_conf = ScanArgsParquet::default();

    let df = LazyFrame::scan_parquet(
        //TODO read_parallel?
        PlRefPath::try_from_path(path.as_ref())?, scan_conf,
    )?;

    df.collect().with_context(|| format!("reading Parquet file {}", path.as_ref().display()))
}

pub fn clc_records(resource_ids: &Vec<String>, base_dir: &str) -> anyhow::Result<usize> {
    let mut scan_conf = ScanArgsParquet::default();
    scan_conf.cache = false;

    resource_ids.iter().try_fold(0usize, |count, resource_id| {
        let path = format!("{base_dir}/train_eegs/{resource_id}.parquet");
        let df = LazyFrame::scan_parquet(PlRefPath::new(&path), scan_conf.clone())?.collect().with_context(|| format!("reading resource {path}"))?;
        count.checked_add(df.height()).context("resource record count overflow")
    })
}

pub fn save_tensor(columns: &[&str], ids: Vec<String>, t: &Tensor, path: &str) -> anyhow::Result<()> {
    // Validate the export shape before indexing names or creating a partial file.
    anyhow::ensure!(!columns.is_empty() && t.size().len() == 2, "CSV export requires column names and a matrix");
    anyhow::ensure!(t.size()[0] as usize == ids.len() && t.size()[1] as usize == columns.len() - 1, "CSV export dimensions do not match IDs and column names");
    let mut data = vec![Series::new(columns[0].into(), ids).into()];
    for (index, name) in columns.iter().skip(1).enumerate() {
        let tensor = tch_result(t.f_select(1, index as i64))?;
        let values = tch_result(Vec::<f32>::try_from(&tensor))?;
        data.push(Series::new((*name).into(), values).into());
    }
    let mut df = DataFrame::new(t.size()[0] as usize, data)?;
    let mut file = std::fs::File::create(path).with_context(|| format!("creating CSV export {path}"))?;
    CsvWriter::new(&mut file).finish(&mut df).with_context(|| format!("writing CSV export {path}"))?;
    Ok(())
}

pub async fn labeled_tensors_by_descr<'a>(
    label: &String,
    resource_set_descr: &String,
    store: Arc<dyn MotifsStore>,
) -> anyhow::Result<Vec<(i64, Vec<Resource>)>> {
    debug!(
        "labeled_tensors_by_descr: label={:?} resource_set_descr={:?}",
        label, resource_set_descr
    );

    let store_lock = store;

    Ok(vec![
        (
            1,
            store_lock
                .list_resource_set_resources_by_descr(resource_set_descr)
                .instrument(tracing::debug_span!("labeled_tensors_by_descr/1", file = file!(), line = line!())).await
                .with_context(|| format!("loading resource set {resource_set_descr}"))?,
        ),
        (
            0,
            store_lock
                .list_resource_random_not_labeled(label.as_str(), 100)
                .instrument(tracing::debug_span!("labeled_tensors_by_descr/2", file = file!(), line = line!())).await
                .with_context(|| format!("loading resources without label {label}"))?,
        ),
    ])
}

#[cfg(test)]
mod data_tests {
    use polars::prelude::*;
    use tch::Device::Cpu;
    use tch::Kind::Float;
    use tch::{IndexOp, Tensor};
    use crate::data::split_frame_by_offsets;
    use thoosa_shared::test_utils::init_test_logging;

    #[ctor::ctor]
    fn init() {
        init_test_logging();
    }


    #[test]
    pub fn test_split_frame_by_offsets() -> anyhow::Result<()> {
        let df = df!(
            "col1"  => (0..10).map(|i| i as f64).collect::<Vec<f64>>(),
            "col2"  => (0..10).map(|i| i as f64 * 10.).collect::<Vec<f64>>(),
        )?;

        let mut frames = split_frame_by_offsets(df, [0i64, 3i64, 8i64])?;

        frames.sort_by(|f1, f2| f1.shape().0.cmp(&f2.shape().0));

        assert_eq!(
            &df!(
                "col1"   => [8.0, 9.0],
                "col2"   => [80.0, 90.0],
            )?,
            &frames[0],
        );

        assert_eq!(
            &df!(
                "col1"   => [0.0, 1.0, 2.0],
                "col2"   => [0.0, 10.0, 20.0],
            )?,
            &frames[1],
        );

        assert_eq!(
            &df!(
                "col1"   => [3.0, 4.0, 5.0, 6.0, 7.0],
                "col2"   => [30.0, 40.0, 50.0, 60.0, 70.0],
            )?,
            &frames[2],
        );

        Ok(())
    }

    #[test]
    fn test_data_io_failures_are_returned() -> anyhow::Result<()> {
        // A file used as a parent guarantees failure without relying on permissions.
        let dir = std::path::Path::new("/tmp").join(thoosa_shared::cdt_for_dir());
        std::fs::create_dir_all(&dir)?;
        let blocker = dir.join("file");
        std::fs::write(&blocker, b"not a directory")?;
        let path = blocker.join("data.csv");
        let tensor = Tensor::f_zeros([1, 1], (Float, Cpu))?;
        let error = super::save_tensor(&["id", "value"], vec!["a".into()], &tensor, path.to_str().unwrap()).unwrap_err();
        assert!(error.downcast_ref::<std::io::Error>().is_some());
        assert!(super::read_parquet_to_df(&blocker).is_err());
        assert!(super::clc_records(&vec!["missing".into()], dir.to_str().unwrap()).is_err());
        std::fs::remove_dir_all(dir)?;
        Ok(())
    }

    #[test]
    fn test_save_tensor() -> anyhow::Result<()> {
        let t = Tensor::randn([3, 6], (Float, Cpu));

        let path = "/tmp/sub.csv";
        let ids = vec!["a111".to_string(), "b222".to_string(), "c333".to_string()];
        super::save_tensor(
            &[
                "eeg_id",
                "seizure_vote",
                "lpd_vote",
                "gpd_vote",
                "lrda_vote",
                "grda_vote",
                "other_vote",
            ],
            ids.clone(),
            &t,
            path,
        )?;

        let lf = LazyCsvReader::new(path.into()).finish().unwrap().collect()?;
        let mut checked = 0;
        for (column_i, column) in lf.columns().iter().enumerate() {
            if column_i == 0 {
                for (row_i, v) in column.str()?.iter().enumerate() {
                    assert_eq!(ids[row_i], v.unwrap());
                    checked += 1;
                }
            } else {
                for (row_i, v) in column.f64()?.iter().enumerate() {
                    assert_eq!(
                        format!(
                            "{:.3}",
                            t.i((row_i as i64, column_i as i64 - 1)).double_value(&[])
                        ),
                        format!("{:.3}", v.unwrap())
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 21);

        Ok(())
    }
}
