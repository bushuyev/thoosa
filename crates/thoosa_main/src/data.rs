use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::Add;
use std::path::Path;
use futures_util::StreamExt;
use parking_lot::Mutex;

use log::debug;
use polars::io::SerReader;
use polars::prelude::*;
use polars::prelude::{
    LazyFrame, NamedFrom,
    ScanArgsParquet, Series,
};

use tch::{Device, IndexOp, Tensor};

use thoosa_common::ResourceLoader;
use thoosa_common::utils::{from, load_tensor};
use thoosa_shared::resource::{Resource, ResourceId};
use thoosa_shared::ThoosaError;
use thoosa_shared::TensorSliceIdx;
use thoosa_store::{SqlStore, Store};

#[derive(Default)]
pub enum Set {
    #[default]
    Train,
    Valid
}

pub trait Splitter {
    type Type;

    fn split_to(&self, v:&Self::Type) ->Set;

    fn init(&self, df:LazyFrame) -> Result<LazyFrame, ThoosaError> {
        Ok(df.clone())
    }
}

pub struct AllTrainSplitter<E> {
    _pd:PhantomData<E>
}

impl <E> Splitter for AllTrainSplitter<E> {
    type Type = E;

    fn split_to(&self, _v: &Self::Type) -> Set {
        Set::Train
    }
}


pub fn series_to_tensor(s: &Series) ->  Result<Tensor, ThoosaError> {

    Ok(match s.dtype(){
        // DataType::UInt8 => {}
        // DataType::UInt16 => {}
        DataType::UInt32 => {
            Tensor::from_slice(s.u32()?.to_vec().iter().map(|v| v.unwrap_or_default() as i32).collect::<Vec<i32>>().as_slice())
        }
        DataType::UInt64 => {
            Tensor::from_slice(s.i64()?.to_vec().iter().map(|v| v.unwrap_or_default() as i32).collect::<Vec<i32>>().as_slice())
        }
        // DataType::Int8 => {}
        // DataType::Int16 => {}
        DataType::Int32 => {
            Tensor::from_slice(s.u32()?.to_vec().iter().map(|v| v.unwrap_or_default() as i32).collect::<Vec<i32>>().as_slice())
        }
        // DataType::Int64 => {
        //     // tensors.push(Tensor::from_slice(s.i64()?.to_vec().iter().map(|v| v.unwrap_or_default()).collect::<Vec<i64>>().as_slice()));
        // }
        DataType::Float32 => {
            Tensor::from_slice(s.f32()?.to_vec().iter().map(|v| v.unwrap_or_default()).collect::<Vec<f32>>().as_slice())
        }
        DataType::Float64 => {
            Tensor::from_slice(s.f64()?.to_vec().iter().map(|v| v.unwrap_or_default() as f32).collect::<Vec<f32>>().as_slice())
        }
       _=>{
           todo!()
       }
    })

}

#[allow(dead_code)]
fn arr_to_expr<T, X>(id_expr: Expr, iter: T, step: X) -> Expr
    where
        T: IntoIterator<Item = X>,
        X: Add + Into<Expr> + Copy,
        Expr: From<<X as Add>::Output>,
{
    iter.into_iter()
        .fold(None, |expr_op: Option<Expr>, v| {
            let new_expr = id_expr
                .clone()
                .gt_eq(v)
                .and(id_expr.clone().lt_eq(v + step));

            if let Some(exp) = expr_op {
                Some(exp.or(new_expr))
            } else {
                Some(new_expr)
            }
        })
        .expect("exp")
}

#[allow(dead_code)]
fn split_frame_by_offsets<I>(mut df: DataFrame, offsets_sec:I) -> PolarsResult<Vec<DataFrame>> where I: IntoIterator<Item = i64> + Clone {

    let shape = df.shape();
    let it_1 = offsets_sec.clone().into_iter();
    let mut it_2 = offsets_sec.into_iter().chain(std::iter::once(shape.0 as i64));
    it_2.next();


    let ids = it_2.zip(it_1)
        .map(|(v2, v1)| v2-v1)
        .enumerate()
        .flat_map(|(i, frag)| std::iter::repeat(i as i64).take(frag as usize))
        .collect::<Vec<i64>>();

    assert!(ids.len() > 1);

    df.with_column(Series::from_vec("frag_id", ids))?;

    df.partition_by(&["frag_id"], false)
}


pub fn load_to_logits_tensors (group:&str,  device: &Device, store: Arc<Mutex<Box<dyn Store>>>) -> Vec<(Resource, Vec<f32>)> {

    let mut store_lock = store.lock();

    store_lock.list_resource_random_in_group(group).expect("list_resource_random_in_group")
        .into_iter()
        .map(move |(resourceId, labels_p)|{

                debug!("labels_p={:?}", labels_p);
                (
                    resourceId,
                    labels_p, //Box::new(Tensor::from_slice(labels_p.as_slice()).to_device(*device))
                )
        })
        .collect()
}

pub fn parquet_to_tensor_by_path(path: impl AsRef<Path>, device: Device) -> Result<Tensor, ThoosaError> {
    let df = read_parquet_to_df(path)?;

    let tensor = Tensor::vstack(&df.get_columns().into_iter().map(|s| series_to_tensor(s).unwrap()).collect::<Vec<Tensor>>());
    Ok(tensor.set_requires_grad(false).to_device(device))
}

pub fn read_parquet_to_df(path: impl AsRef<Path>) -> Result<DataFrame, ThoosaError> {
    let scan_conf = ScanArgsParquet::default();

    let df = LazyFrame::scan_parquet( //TODO read_parallel?
      path,
      scan_conf,
    )?;

    Ok(df.collect()?)
}

pub fn clc_records(resource_ids: &Vec<String>, base_dir: &str) -> Result<usize, ThoosaError> {
    let mut scan_conf = ScanArgsParquet::default();
    scan_conf.cache = false;

    Ok(resource_ids.iter().map(|resource_id| {
        //TODO config path
        let df = LazyFrame::scan_parquet(
            format!("{}/train_eegs/{}.parquet", base_dir, resource_id),
            scan_conf.clone(),
        ).unwrap().collect().unwrap();

        debug!("eeg_id={:?} eeg_df.shape()={:?}", resource_id, df.shape());

        df.shape().0
    }).sum())
}


pub fn save_tensor(columns:&[&str], ids:Vec<String>, t:&Tensor, path: &str) {
    assert_eq!(t.size().len(), 2);
    assert_eq!(t.size()[1] as usize, columns.len() - 1);

    let mut df = DataFrame::new(
        [
            vec![Series::new(columns[0], ids)],

            columns.into_iter().skip(1).enumerate().map(move |(i, c)| {
                Series::new(c, from::<Vec<f32>>(&t.i((.., i as i64))).as_slice())
            }).collect()
        ].concat()
    ).unwrap();


    let mut file = std::fs::File::create(path).unwrap();
    CsvWriter::new(&mut file).finish(&mut df).unwrap();

}

pub fn labeled_tensors_by_descr<'a,  >(label:&String, resource_set_descr:&String, store: Arc<Mutex<Box<dyn Store>>>) -> Vec<(i64, Vec <Resource>)>  {
    debug!("labeled_tensors_by_descr: label={:?} resource_set_descr={:?}", label, resource_set_descr);

    let mut store_lock = store.lock();

    vec![
        (
            1,
            store_lock.list_resource_set_resources_by_descr(resource_set_descr).expect("list_resource_set_resources_by_descr")

        ),
        (
            0,
            store_lock.list_resource_random_not_label(label.as_str(), 100).expect("list_resource_random_not_label")

        )
    ]
}


#[cfg(test)]
mod data_tests {
    use log::debug;
    use polars::prelude::*;
    use tch::Device::Cpu;
    use tch::Kind::Float;
    use tch::{IndexOp, Tensor};
    use thoosa_shared::{TensorSliceIdx, ThoosaError};
    use thoosa_shared::init_test_logging;
    use crate::data::{arr_to_expr, split_frame_by_offsets};

    #[ctor::ctor]
    fn init() {
        init_test_logging();
    }

    #[test]
    pub fn test_arr_to_expr() -> Result<(), ThoosaError> {
        let df: DataFrame = df!(
            "val"       => [1, 2, 3, 4, 5, 6, 7, 8, 9],
        )?;

        let id_expr = col("id");

        let filter_exps = arr_to_expr(id_expr, [2, 6], 2);

        let df = df
            .lazy()
            .with_row_count("id", None)
            .filter(filter_exps)
            .collect()?;

        debug!("df={:?}", df);

        assert_eq!(
            df!(
                "id"   => [2, 3, 4, 6, 7, 8],
                "val"  => [3, 4, 5, 7, 8, 9],
            )?,
            df,
        );

        Ok(())
    }

    #[test]
    pub fn test_split_frame_by_offsets() -> Result<(), ThoosaError> {

        let df = df!(
            "col1"  => (0..10).map(|i| i as f64).collect::<Vec<f64>>(),
            "col2"  => (0..10).map(|i| i as f64 * 10.).collect::<Vec<f64>>(),
        )?;


        let mut frames = split_frame_by_offsets(df, [0i64, 3i64, 8i64])?;

        frames.sort_by(|f1, f2|f1.shape().0.cmp(&f2.shape().0));

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
    fn test_save_tensor() -> Result<(), ThoosaError>{
        let t = Tensor::randn([3, 6], (Float, Cpu));

        let path = "/tmp/sub.csv";
        let ids = vec!["a111".to_string(), "b222".to_string(), "c333".to_string()];
        super::save_tensor(
            &["eeg_id","seizure_vote","lpd_vote","gpd_vote","lrda_vote","grda_vote","other_vote"],
            ids.clone(),
            &t,
            path
        );

        let lf = LazyCsvReader::new(path).finish().unwrap().collect()?;
        let mut checked = 0;
        for (column_i, column) in lf.get_columns().iter().enumerate() {
            if column_i == 0 {
             for (row_i, v) in column.str()?.into_iter().enumerate() {
                 assert_eq!(ids[row_i], v.unwrap());
                 checked += 1;
             }

            } else {
                for (row_i, v) in column.f64()?.into_iter().enumerate() {
                    assert_eq!(
                        format!("{:.3}", t.i((row_i as i64, column_i as i64-1)).double_value(&[])),
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

pub struct ParquetLoader {
    base_path:String,
    tensors_path:Option<String>,
    dsp_filters: Option<Box<dyn Fn(Tensor) -> Tensor + Send + Sync>>,
    resource_to_path:Box<dyn Fn(&str, &str)->String  + Send + Sync>
}

impl ParquetLoader {

    pub fn new(
        base_path: String,
        tensors_path: Option<String>,
        dsp_filters: Option<Box<dyn Fn(Tensor)->Tensor +Send + Sync>>,
        resource_to_path:Box<dyn Fn(&str, &str)->String  + Send + Sync>
    ) -> Self {
        Self {
            base_path, tensors_path, dsp_filters, resource_to_path
        }
    }

}

impl ResourceLoader for ParquetLoader {
    type K = String;
    type V = Tensor;

    fn load(&self, key: &String, device:Device) -> Box<Tensor> {

        let t = if let Some(tensors_path) = self.tensors_path.as_ref() {//TODO move into another loader
            load_tensor(tensors_path.as_str(), key.as_str(), device)

        } else {
            parquet_to_tensor_by_path(
                (self.resource_to_path)(self.base_path.as_str(), key.as_str()),
                device
            )

        }.map(|t|{
            match self.dsp_filters.as_ref() {
                None => t,
                Some(f) => f(t)
            }

        }).expect(format!("can't load {}", key).as_str());

        Box::new(t)
    }
}