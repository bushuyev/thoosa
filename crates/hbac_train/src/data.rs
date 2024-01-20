#![feature(vec_into_raw_parts)]

use std::fmt::{Debug, Formatter};
use std::ops::Add;
use std::time::Instant;

use polars::error::PolarsError;
use polars::io::SerReader;
use polars::prelude::*;
use polars::prelude::{
    col, LazyCsvReader, LazyFileListReader, LazyFrame, NamedFrom,
    ScanArgsParquet, Series,
};
use tch::Tensor;

#[derive(Debug)]
pub struct GenError {}

impl GenError {
    pub fn new() -> Self {
        Self {}
    }
}

impl From<PolarsError> for GenError {
    fn from(value: PolarsError) -> Self {
        print!("value={:?}", value);
        GenError::new()
    }
}

pub struct EEGs {
    df: LazyFrame,
    base_path: String,
}

impl Debug for EEGs {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EEGs")
            .field("logical_plan", &self.df.logical_plan)
            .finish()
    }
}

impl EEGs {

    pub fn new(base_path: &str) -> Result<Self, GenError> {
        Ok(Self {
            df: LazyCsvReader::new(format!("{}/train.csv", base_path))
                .finish()?
                .with_streaming(true),
            base_path: base_path.into(),
        })
    }

    pub fn load_eegs(&self) -> Result<Box<Vec<Tensor>>, GenError> {
        let gr_by_eeg_id_df = self
            .df
            .clone()
            .collect()?
            // .sample_n(&Series::new("n", [10]), false, true, None)?
            .lazy()
            .top_k(10, [col("eeg_id")], [false], false, false)
            .group_by([col("eeg_id")])
            .agg([col("eeg_label_offset_seconds").sort(false).alias("eeg_label_offset_seconds")])
            .collect()?;

        println!("gr_by_eeg_id_df.shape()[0]={}", gr_by_eeg_id_df.shape().0);

        let mut tensors = Box::new(Vec::with_capacity(gr_by_eeg_id_df.shape().0));

        for (eeg_id, eeg_label_offset_seconds) in gr_by_eeg_id_df
            .column("eeg_id")?
            .i64()?
            .into_iter()
            .zip(gr_by_eeg_id_df.column("eeg_label_offset_seconds")?.list()?)
            .map(|(eeg_id, eeg_label_offset_seconds)| {
                (
                    eeg_id.unwrap_or_default(),
                    eeg_label_offset_seconds.unwrap_or_default(),
                )
            })
        {


            let id_expr = col("id");

            let fragments_in_mls = eeg_label_offset_seconds
                .f64()?
                .into_iter()
                .map(|v| v.expect("no id :(") as u32 * 200)
                .collect::<Vec<u32>>();


            // let filter_exps = arr_to_expr(
            //     id_expr,
            //     fragments_in_mls,
            //     50 * 200,
            // );

            println!("eeg_id ={:?}, fragments_in_mls={:?}", eeg_id, fragments_in_mls);

            let mut scan_conf = ScanArgsParquet::default();
            // scan_conf.cache = false;

            let eeg_df = LazyFrame::scan_parquet(
                format!("{}/train_eegs/{}.parquet", self.base_path, eeg_id),
                scan_conf,
            )
                .unwrap();
            let df = eeg_df
                // .with_row_count("id", None)
                // .filter(filter_exps)
                .collect()?;

            let mut column_tensors = Vec::with_capacity(df.shape().0);

            for s in df.get_columns() {
                column_tensors.push(series_to_tensor(s, )?);

            }

            tensors.push(Tensor::stack(column_tensors.as_slice(), 0));

            print!(".");
        }

        println!("tensors.len={}  size of tensors {}", tensors.len(),  std::mem::size_of_val(&*tensors));

        Ok(tensors)
    }
}

fn series_to_tensor(s: &Series) ->  Result<Tensor, GenError> {
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

fn arr_to_expr<T, X>(id_expr: Expr, iter: T, step: X) -> Expr
    where
        T: IntoIterator<Item = X>,
        X: Add + Into<Expr> + Copy,
        Expr: From<<X as Add>::Output>,
{
    iter.into_iter()
        .fold(None, |mut expr_op: Option<Expr>, v| {
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

fn main() -> Result<(), GenError> {
    let t1 = Instant::now();
    let eegs = EEGs::new("../../data")?;

    let result = eegs.load_eegs()?;

    println!("eegs={:?}", result.len());

    println!("done in {}", t1.elapsed().as_millis());

    Ok(())
}

#[cfg(test)]
mod tests {
    use polars::prelude::*;

    use crate::{EEGs, GenError};
    use crate::data::arr_to_expr;

    #[ctor::ctor]
    fn init() {
        std::env::set_var("POLARS_FMT_MAX_COLS", "100");
        std::env::set_var("POLARS_FMT_MAX_ROWS", "100");
    }

    #[test]
    pub fn test_load_eegs() -> Result<(), GenError> {
        let eegs = EEGs::new("../../data")?;

        println!("eegs={:?}", eegs.load_eegs());

        Ok(())
    }

    #[test]
    pub fn test_load_by_std() -> Result<(), GenError> {
        //TODO https://github.com/pola-rs/polars/issues/12324
        let df: DataFrame = df!(
            "x" => 0..8,
            "y" => (0..8).rev().collect::<Vec<i32>>(),
            "z" => (0..8).map(|x|x^2).collect::<Vec<i32>>(),
        )?;

        println!("1. df={:?}", df);

        // let df = df.lazy().with_columns([sum_horizontal(&[cols(["x", "y"])])]).collect()?;
        // let df = df.sum_horizontal(NullStrategy::Ignore)?;

        println!("2. df={:?}", df);

        Ok(())
    }

    #[test]
    pub fn test_arr_to_expr() -> Result<(), GenError> {
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

        println!("df={:?}", df);

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
    pub fn test_test() -> Result<(), GenError> {

        let mut df = df!(
            "col1"  => (0..10).map(|i| i as f64).collect::<Vec<f64>>(),
            "col2"  => (0..10).map(|i| i as f64 * 10.).collect::<Vec<f64>>(),
        )?;

        let rows_in_sec = 1;
        let mut offsets_sec = vec![0i64, 3i64, 8i64];


        let frames = split_frame_by_offsets(df, offsets_sec.into_iter());

        for frame in frames {
            println!("frame={:?}", frame);
        }


        // let fragment_ids =

        // df.with_row_count("id", None).partition_by("id");

        // for field in df.clone().lazy().schema()?.iter_fields() {
        //     println!("{:?}", field.name);
        //     let column_df = df.clone().lazy().select(&[col(field.name.as_str())]).collect()?;
        //     // column_df.column(field.name.as_str())?.f64().iter().collect();
        // }

        // for s in df.get_columns() {
        //     let t = Tensor::from_slice(s.f64()?.to_vec().iter().map(|v| v.unwrap_or_default()).collect::<Vec<f64>>().as_slice());
        //     t.print()
        // }


        Ok(())
    }

    fn split_frame_by_offsets<I>(mut df: DataFrame, offsets_sec:I) -> PolarsResult<Vec<DataFrame>> where I: IntoIterator<Item = i64> + Clone {
        // offsets_sec.push(df.shape().0);

        // let offsets_diff = offsets_sec.windows(2).map(|w|w[1]-w[0]).collect::<Vec<usize>>();

        let shape = df.shape();
        let mut it_1 = offsets_sec.clone().into_iter();
        let mut it_2 = offsets_sec.into_iter().chain(std::iter::once(shape.0 as i64));
        it_2.next();

        // let xxxxxx = it_2.zip(it_1).map(|(v2, v1)| v2-v1).collect::<Vec<i64>>();
        //
        // println!("xxxxxx={:?}", xxxxxx);


        // let ids = offsets_sec.windows(2).map(|w| w[1] - w[0]).into_iter().enumerate().flat_map(|(i, frag)| std::iter::repeat(i as i32).take(frag)).collect::<Vec<i32>>();
        let ids = it_2.zip(it_1).map(|(v2, v1)| v2-v1).enumerate().flat_map(|(i, frag)| std::iter::repeat(i as i64).take(frag as usize)).collect::<Vec<i64>>();

        println!("ids={:?}", ids);

        df.with_column(Series::from_vec("frag_id", ids));
        // df.with_column(Series::from_vec("xxx", (0..10).map(|i| 2. * i as f64).collect::<Vec<f64>>()));
        println!("df={:?}", df);

         df.partition_by(&["frag_id"], false)
    }
}
