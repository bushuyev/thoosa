use std::ops::ControlFlow;
use polars::error::PolarsError;
use polars::io::SerReader;
use polars::prelude::{col, count, CsvReader, LazyCsvReader, LazyFileListReader, LazyFrame, ScanArgsParquet};

#[derive(Debug)]
struct GenError {

}

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

fn main() -> Result<(), GenError>{
    std::env::set_var("POLARS_FMT_MAX_COLS", "100"); //FMT_MAX_COLS is pub(crate) in polars_core :(
    std::env::set_var("POLARS_FMT_MAX_ROWS", "100");


    // let df = LazyFrame::scan_parquet("../data/train.csv", ScanArgsParquet::default())?
    let df = LazyCsvReader::new("../../data/train.csv").finish()?
        .with_streaming(true)
        // .group_by([col("eeg_id")])
        // .agg([col("patient_id").count().alias("patient_id counts"), col("patient_id").unique().count().alias("unique patient_id count")])
        // .select([col("eeg_id").unique()])
        .group_by([col("eeg_id")])
        .agg([col("eeg_label_offset_seconds").alias("eeg_label_offset_seconds")])
        .collect()?;

    // println!("df: {:?}", df.head(Some(1000)));
    println!("df: {:?}", df.column("eeg_label_offset_seconds")?.list()?);


    for (eeg_id, eeg_label_offset_seconds) in df.column("eeg_id")?.i64()?.into_iter().zip(df.column("eeg_label_offset_seconds")?
        .list()?
        .into_iter())
        .take(3)
        .map(|(eeg_id, max_eeg_label_offset_seconds)| (eeg_id.unwrap_or_default(), max_eeg_label_offset_seconds.unwrap_or_default())) {

            for offset in eeg_label_offset_seconds.f64()? {
                println!("eeg_id={:?}, offset={:?}", eeg_id, offset)
            }

            // let eeg_df = LazyFrame::scan_parquet(format!("../data/train_eegs/{}.parquet", eeg_id), ScanArgsParquet::default()).unwrap();
            // let ekg_rows_count =  eeg_df.select([col("EKG").count().alias("EKG_count")]).collect().unwrap().column("EKG_count").unwrap().u32().unwrap().get(0).unwrap() as f64;
            // // let ekg_count = 0;
            //
            // let min_rows_eeg = (max_eeg_label_offset_seconds + 50.0) * 200.0;
            //
            // let ok = ekg_rows_count <= min_rows_eeg;
            // if !ok {
            //     println!("eeg_id={}, max_eeg_label_offset_seconds={}, ekg_rows_count={}, min_rows_eeg={} not ok={}", eeg_id, max_eeg_label_offset_seconds, ekg_rows_count, min_rows_eeg, ok);
            // }

    };

    // println!("df_csv: {:?}", df.head(Some(1000)));

    Ok(())
}
