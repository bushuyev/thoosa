#![feature(vec_into_raw_parts)]

use polars::error::PolarsError;
use polars::io::SerReader;
use polars::prelude::*;
use polars::prelude::{
    col, count, CsvReader, LazyCsvReader, LazyFileListReader, LazyFrame, NamedFrom,
    ScanArgsParquet, Series,
};
use polars::series::ops::*;
use std::fmt::{Debug, Formatter};
use std::ops::Add;
use std::ops::ControlFlow;
use std::time::Instant;
use tch::Tensor;

#[derive(Debug)]
struct GenError {}

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

    pub fn load_eegs(&self) -> Result<(), GenError> {
        let gr_by_eeg_id_df = self
            .df
            .clone()
            .collect()?
            .sample_n(&Series::new("n", [100_000]), false, true, None)?
            .lazy()
            .group_by([col("eeg_id")])
            .agg([col("eeg_label_offset_seconds").alias("eeg_label_offset_seconds")])
            .collect()?;

        let mut tensors = Box::new(vec![]);
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
            // println!("eeg_label_offset_seconds ={:?}", eeg_label_offset_seconds);

            let id_expr = col("id");

            let filter_exps = arr_to_expr(
                id_expr,
                eeg_label_offset_seconds
                    .f64()?
                    .into_iter()
                    .map(|v| v.expect("no id :(") as u32)
                    .collect::<Vec<u32>>(),
                50 * 200,
            );

            let eeg_df = LazyFrame::scan_parquet(
                format!("{}/train_eegs/{}.parquet", self.base_path, eeg_id),
                ScanArgsParquet::default(),
            )
            .unwrap();
            let df = eeg_df
                .with_row_count("id", None)
                .filter(filter_exps)
                .collect()?;

            // println!("df={:?}", df.shape());

            for s in df.get_columns() {
                match s.dtype(){
                    DataType::Boolean => {}
                    DataType::UInt8 => {}
                    DataType::UInt16 => {}
                    DataType::UInt32 => {
                        tensors.push(Tensor::from_slice(s.u32()?.to_vec().iter().map(|v| v.unwrap_or_default() as i32).collect::<Vec<i32>>().as_slice()));
                    }
                    DataType::UInt64 => {
                        tensors.push(Tensor::from_slice(s.i64()?.to_vec().iter().map(|v| v.unwrap_or_default()).collect::<Vec<i64>>().as_slice()));
                    }
                    DataType::Int8 => {}
                    DataType::Int16 => {}
                    DataType::Int32 => {
                        tensors.push(Tensor::from_slice(s.u32()?.to_vec().iter().map(|v| v.unwrap_or_default() as i32).collect::<Vec<i32>>().as_slice()));
                    }
                    DataType::Int64 => {
                        // tensors.push(Tensor::from_slice(s.i64()?.to_vec().iter().map(|v| v.unwrap_or_default()).collect::<Vec<i64>>().as_slice()));
                    }
                    DataType::Float32 => {
                        tensors.push(Tensor::from_slice(s.f32()?.to_vec().iter().map(|v| v.unwrap_or_default()).collect::<Vec<f32>>().as_slice()));
                    }
                    DataType::Float64 => {
                        tensors.push(Tensor::from_slice(s.f64()?.to_vec().iter().map(|v| v.unwrap_or_default()).collect::<Vec<f64>>().as_slice()));
                    }
                    // DataType::Decimal(_, _) => {}
                    DataType::String => {}
                    DataType::Binary => {}
                    DataType::Date => {}
                    DataType::Datetime(_, _) => {}
                    DataType::Duration(_) => {}
                    DataType::Time => {}
                    // DataType::Array(_, _) => {}
                    DataType::List(_) => {}
                    // DataType::Object(_, _) => {}
                    DataType::Null => {}
                    // DataType::Categorical(_, _) => {}
                    // DataType::Struct(_) => {}
                    DataType::Unknown => {}
                }
                // t.print()

            }
            print!(".");
        }

        println!("tensors.len={}", tensors.len());

        Ok(())
    }
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

    println!("eegs={:?}", eegs.load_eegs());

    println!("done in {}", t1.elapsed().as_millis());

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{arr_to_expr, EEGs, GenError};
    use ctor::*;
    use polars::prelude::*;
    use polars::series::ops::*;
    use tch::Tensor;
    use torch_sys::C_tensor;

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

        for s in df.get_columns() {
            let t = Tensor::from_slice(s.f64()?.to_vec().iter().map(|v| v.unwrap_or_default()).collect::<Vec<f64>>().as_slice());
            t.print()
        }



        Ok(())
    }
}
