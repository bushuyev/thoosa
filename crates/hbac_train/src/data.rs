#![feature(vec_into_raw_parts)]

use std::any::Any;
use std::fmt::{Debug, Display, Formatter};
use std::io::Write;
use std::marker::PhantomData;
use std::ops::Add;
use std::time::Instant;

use log::{debug, SetLoggerError};
use polars::error::PolarsError;
use polars::export::ahash::{HashMap, HashMapExt};
use polars::export::arrow::compute::filter::filter;
use polars::export::arrow::io::iterator::StreamingIterator;
use polars::io::SerReader;
use polars::prelude::*;
use polars::prelude::{
    col, LazyCsvReader, LazyFileListReader, LazyFrame, NamedFrom,
    ScanArgsParquet, Series,
};
use rand::{Rng, thread_rng};
use rand::distributions::WeightedError;
use rand::prelude::SliceRandom;
use tch::{Kind, Tensor};

use crate::utils::DataBatch;

#[derive(Debug)]
pub struct GenError {
    msg:String
}

impl GenError {
    pub fn new(msg:String) -> Self {
        Self {
            msg
        }
    }
}


impl<D> From<D> for GenError where D: Display {
    fn from(value: D) -> Self {
        debug!("value={}", value);
        GenError::new(value.to_string())
    }
}

#[derive(Debug)]
pub struct EEG{
    egg_map:Vec<Vec<f32>>,
    eeg_id: i64,
    set: i32,
    fragments_rows: Vec<u32>,
    seizure_vote: u8,
    lpd_vote: u8,
    gpd_vote: u8,
    lrda_vote: u8,
    grda_vote: u8,
    other_vote: u8
}

impl EEG {
    pub fn new(egg_map:Vec<Vec<f32>>, eeg_id:i64, set: i32, fragments_rows:Vec<u32>, seizure_vote:u8, lpd_vote:u8, gpd_vote:u8, lrda_vote:u8, grda_vote:u8, other_vote:u8) -> Self {
        Self {
            egg_map,
            eeg_id,
            set,
            fragments_rows,     seizure_vote,      lpd_vote,      gpd_vote,       lrda_vote,      grda_vote,      other_vote
        }
    }

    pub fn random_fragemnt_tensor<R>(&self, rnd:&mut R) -> Tensor where R: Rng + ?Sized {

        let fragment_idx = rnd.gen_range(0..self.fragments_rows.len());

        let i_1 = self.fragments_rows[fragment_idx] as usize;
        let i_2 = i_1 + 50 * 200;

        self.fragment_to_tensor(i_1, i_2)
    }

    pub fn all_fragemnts_tensor(&self) -> Vec<Tensor> {
        self.fragments_rows.iter().map(|i| self.fragment_to_tensor(*i as usize, *i as usize + 50 * 200)).collect()
    }

    pub fn votes_tensor(&self) -> Tensor {
        Tensor::from_slice(&[self.seizure_vote,  self.lpd_vote,      self.gpd_vote,       self.lrda_vote,      self.grda_vote,      self.other_vote]).totype(Kind::Float)
    }

    pub fn fragment_to_tensor(&self, i_1:usize, i_2:usize) -> Tensor {
        assert!(i_1 < i_2, "fragment_to_tensor: i_1 < i_2");

        Tensor::stack(self.egg_map.iter().map(|values| {
            Tensor::from_slice(&values[i_1..i_2])
        }).collect::<Vec<Tensor>>().as_slice(), 0)

    }
}

#[derive(Debug)]
pub struct EEGs {
    train_set: Vec<EEG>,
    valid_set: Vec<EEG>,
}

#[derive(Default)]
pub enum Set {
    #[default]
    Train,
    Valid
}

pub trait Splitter {
    type Type;

    fn split_to(&self, v:&Self::Type) ->Set;

    fn init(&self, df:LazyFrame) -> Result<LazyFrame, GenError> {
        Ok(df.clone())
    }
}

struct AllTrainSplitter<E> {
    _pd:PhantomData<E>
}

impl <E> Splitter for AllTrainSplitter<E> {
    type Type = E;

    fn split_to(&self, v: &Self::Type) -> Set {
        Set::Train
    }
}

pub(crate) struct ByPatientIdSplitter {
    train_frac:f32,
    train_patients:Vec<i64>,
    test_patients:Vec<i64>
}

impl ByPatientIdSplitter {
    pub fn new(train_frac:f32) -> Self {
        Self {
            train_frac: train_frac,
            train_patients: vec![],
            test_patients: vec![]
        }
    }
}

impl Splitter for ByPatientIdSplitter {
    type Type = EEG;

    fn split_to(&self, eeg: &Self::Type) -> Set {
        if eeg.set == 1 {
            Set::Valid
        } else {
            Set::Train
        }

    }

    fn init(&self, df: LazyFrame) -> Result<LazyFrame, GenError>{
        let mut df_sorted = df.clone().group_by(["patient_id"]).agg([
            col("seizure_vote").sum(),
            col("lpd_vote").sum(),
            col("gpd_vote").sum(),
            col("lrda_vote").sum(),
            col("grda_vote").sum(),
            col("other_vote").sum(),
        ]).sort_by_exprs([
             col("seizure_vote")
                 + col("lpd_vote")
                 + col("gpd_vote")
                 + col("lrda_vote")
                 + col("grda_vote")
                 + col("other_vote")
         ],
         vec![false], true, true
        )
        .select([col("patient_id")])
        .collect()?;

        let df_set = df_sorted.with_column(Series::new("set", (0..df_sorted.shape().0).map(|i| if rand::random::<f32>() > self.train_frac { 1i32 } else { 0i32 }).collect::<Vec<_>>()))?;

        Ok(df.clone().join(df_set.clone().lazy(), [col("patient_id")], [col("patient_id")], JoinArgs::new(JoinType::Left)))
        // Ok(df.clone().with_column(lit(1).alias("set"))) //.collect()?.lazy()
    }
}

impl EEGs {

    pub fn good_eegs(base_path: &str)  -> Result<Series, GenError>{
        let good_eegs_filter = Self::good_eegs_filter(base_path)?;

        Ok(good_eegs_filter
            .collect()?
            .column("eeg_id")?.clone())

    }

    pub fn good_eegs_of(base_path: &str, kind:&str)  -> Result<Series, GenError>{
        let good_eegs_filter = Self::good_eegs_filter(base_path)?;

        Ok(
            good_eegs_filter
            .filter(col(kind).gt(lit(0)))
                // .sort(kind, SortOptions{ descending: true, ..Default::default()})
                .sort("eeg_id", Default::default())
            .collect()?
            .column("eeg_id")?.clone()
        )

    }

    fn good_eegs_filter(base_path: &str) -> Result<LazyFrame, GenError> {
        Ok(LazyCsvReader::new(format!("{}/train.csv", base_path))
            .finish()?
            .with_streaming(true)
            .group_by([col("eeg_id")])
            .agg([
                col("seizure_vote").first(),
                col("lpd_vote").first(),
                col("gpd_vote").first(),
                col("lrda_vote").first(),
                col("grda_vote").first(),
                col("other_vote").first(),
            ])
            .with_column(concat_list(["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"])?.alias("all_votes").alias("all_votes"))
            .filter(
                col("all_votes").list().max().alias("max").eq(col("all_votes").list().sum().alias("sum"))
            ))
    }

    pub fn load_all_eegs(base_path: &str) -> Result<Box<EEGs>, GenError> {
        EEGs::load_eegs::<_>(base_path, None, None::<AllTrainSplitter<EEG>>)
    }
    pub fn load_eegs<S>(base_path: &str, filter:Option<Expr>, splitter:Option<S>) -> Result<Box<EEGs>, GenError> where S:Splitter<Type=EEG>{
        let df = LazyCsvReader::new(format!("{}/train.csv", base_path))
            .finish()?
            .with_streaming(true);

        let df = match splitter.as_ref() {
            None => {
                df
            }
            Some(splitter) => {
                splitter.init(df)?
            }
        };

        debug!("df={:?}", df.clone().collect()?.head(Some(10)));


        let gr_by_eeg_id_df = df
            .clone()
            // .collect()?
            // .sample_n(&Series::new("n", [3]), false, true, None)?
            // .lazy()
            // .with_column(concat_list([
            //     col("seizure_vote"),
            //     col("lpd_vote"),
            //     col("gpd_vote"),
            //     col("lrda_vote"),
            //     col("grda_vote"),
            //     col("other_vote")
            // ])?.alias("all_votes"))
            .filter(filter.unwrap_or(true.into()))
            // .filter(col("eeg_id").is_in(list([lit(1345588872)])))
            // .filter(col("eeg_id").is_in(list([lit(1345588872)])))
            // .filter(col("eeg_id").eq(lit(1345588872)))
            // .top_k(100, [col("eeg_id")], [false], false, false)
            .group_by([col("eeg_id")])
            .agg([
                col("eeg_label_offset_seconds").sort(false).alias("eeg_label_offset_seconds"),
                col("set").first(),
                col("seizure_vote").first(),
                col("lpd_vote").first(),
                col("gpd_vote").first(),
                col("lrda_vote").first(),
                col("grda_vote").first(),
                col("other_vote").first(),
            ])
            .collect()?;

        debug!("gr_by_eeg_id_df.shape()[0]={}", gr_by_eeg_id_df.shape().0);
        std::io::stdout().flush()?;

        let mut train_eegs = Vec::with_capacity(gr_by_eeg_id_df.shape().0);//TODO capacity
        let mut valid_eegs = Vec::with_capacity(gr_by_eeg_id_df.shape().0);

        for (eeg_id, set, eeg_label_offset_seconds, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote) in gr_by_eeg_id_df
            .column("eeg_id")?.i64()?.into_iter()
            .zip(gr_by_eeg_id_df.column("set")?.i32()?)
            .zip(gr_by_eeg_id_df.column("eeg_label_offset_seconds")?.list()?)
            .zip(gr_by_eeg_id_df.column("seizure_vote")?.i64()?)
            .zip(gr_by_eeg_id_df.column("lpd_vote")?.i64()?)
            .zip(gr_by_eeg_id_df.column("gpd_vote")?.i64()?)
            .zip(gr_by_eeg_id_df.column("lrda_vote")?.i64()?)
            .zip(gr_by_eeg_id_df.column("grda_vote")?.i64()?)
            .zip(gr_by_eeg_id_df.column("other_vote")?.i64()?)
            .map(|
                ((((((((eeg_id, set),
                    eeg_label_offset_seconds),
                    seizure_vote),
                    lpd_vote),
                    gpd_vote),
                    lrda_vote),
                    grda_vote),
                    other_vote)
            | {
                (
                    eeg_id.unwrap_or_default(),
                    set.unwrap_or_default(),
                    eeg_label_offset_seconds.unwrap_or_default(),
                    seizure_vote.unwrap_or_default(),
                    lpd_vote.unwrap_or_default(),
                    gpd_vote.unwrap_or_default(),
                    lrda_vote.unwrap_or_default(),
                    grda_vote.unwrap_or_default(),
                    other_vote.unwrap_or_default(),
                )
            })
        {


            let fragments_rows = eeg_label_offset_seconds
                .f64()?
                .into_iter()
                .map(|v| v.expect("no id :(") as u32 * 200)
                .collect::<Vec<u32>>();


            // debug!("eeg_id ={:?}, fragments_in_mls={:?} seizure_vote={:?}  lpd_vote={:?}, gpd_vote={:?}, lrda_vote={:?}, grda_vote={:?}, other_vote={:?}",
            //                 eeg_id, fragments_in_mls,     seizure_vote,      lpd_vote,      gpd_vote,       lrda_vote,      grda_vote,      other_vote);

            let mut scan_conf = ScanArgsParquet::default();
            // scan_conf.cache = false;

            let eeg_df = LazyFrame::scan_parquet(
                format!("{}/train_eegs/{}.parquet", base_path, eeg_id),
                scan_conf,
            )?;

            let df = eeg_df.collect()?;

            let mut eegs_vec = vec![];
            let mut names_vec = vec![];

            for s in df.get_columns() {
                names_vec.push(s.name().to_owned());

                match s.dtype(){
                    DataType::Float32 => {
                        eegs_vec.push(s.f32()?.downcast_iter().flat_map(|arr|arr.into_iter().map(|v| v.map(|v|*v).unwrap_or(0.0))).collect::<Vec<f32>>());
                    }
                    DataType::Float64 => {
                        eegs_vec.push(s.f64()?.downcast_iter().flat_map(|arr|arr.into_iter().map(|v| v.map(|v|*v as f32).unwrap_or(0.0))).collect::<Vec<f32>>());
                    }
                    _=>{
                        todo!()
                    }
                }

            }

            let eeg =   EEG::new(
                eegs_vec,
                eeg_id,
                set,
                fragments_rows,
                seizure_vote.try_into().expect("seizure_vote"),
                lpd_vote.try_into().expect("lpd_vote"),
                gpd_vote.try_into().expect("gpd_vote"),
                lrda_vote.try_into().expect("lrda_vote"),
                grda_vote.try_into().expect("grda_vote"),
                other_vote.try_into().expect("other_vote")
            );


            match splitter.as_ref().map(|s|s.split_to(&eeg)).unwrap_or_default() {
                Set::Train => {
                    train_eegs.push(eeg);
                }
                Set::Valid => {
                    valid_eegs.push(eeg);
                }
            }


            print!(".");
            std::io::stdout().flush()?;
        }

        debug!("tensors.len={}  valid_eegs.len={}", train_eegs.len(), valid_eegs.len());
        std::io::stdout().flush()?;

        Ok(Box::new(EEGs{
            train_set: train_eegs,
            valid_set: valid_eegs
        }))
    }

    pub fn len(&self) -> usize {
        self.train_set.len()
    }


    fn all_set(data: &Vec<EEG>) -> (Tensor, Tensor) {
        let (eegs, votes) = data.iter()
            .fold((vec![], vec![]), |mut a, eeg| {
                for fragemnt_tensor in eeg.all_fragemnts_tensor() {
                    a.0.push(fragemnt_tensor);
                    a.1.push(eeg.votes_tensor());
                }

                a
            });

        (
            Tensor::stack(
                eegs.as_slice(),
                0,
            ),
            Tensor::stack(
                votes.as_slice(),
                0,
            )
        )
    }

    fn random_subset(batch_size: usize, data: &Vec<EEG>) -> (Tensor, Tensor) {
        let mut rng = thread_rng();

        let (eegs, votes) = data
            .choose_multiple(&mut rng, batch_size)
            .fold((vec![], vec![]), |mut a, eeg| {
                let fragemnt_tensor = eeg.random_fragemnt_tensor(&mut rng);
                // debug!("random_fragemnt_tensor size: {:?}", tensor.size());

                a.0.push(fragemnt_tensor);
                a.1.push(eeg.votes_tensor());

                a
            });

        (
            Tensor::stack(
                eegs.as_slice(),
                0,
            ),
            Tensor::stack(
                votes.as_slice(),
                0,
            )
        )
    }
}

impl DataBatch for EEGs {
    fn train_batch(&self, batch_size: usize) -> (Tensor, Tensor) {
        Self::random_subset(batch_size, &self.train_set)
    }

    fn valid(&self, batch_size: Option<usize>) -> (Tensor, Tensor) {
        match batch_size {
            None => {
                Self::all_set(&self.valid_set)
            }
            Some(batch_size) => {
                Self::random_subset(batch_size, &self.valid_set)
            }
        }

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

pub fn eeg_to_tensor(base_path: &str, eeg_id: &str) -> Result<Tensor, GenError> {
    let scan_conf = ScanArgsParquet::default();

    let eeg_df = LazyFrame::scan_parquet(
        format!("{}/train_eegs/{}.parquet", base_path, eeg_id),
        scan_conf,
    )?;

    let df = eeg_df.collect()?;

    Ok(Tensor::vstack(&df.get_columns().into_iter().map(|s| series_to_tensor(s).unwrap()).collect::<Vec<Tensor>>()))
}

pub fn clc_records(eeg_id_series: &Series) -> Result<usize, GenError> {
    let mut scan_conf = ScanArgsParquet::default();
    scan_conf.cache = false;

    Ok(eeg_id_series.i64()?.into_iter().map(|eeg_id| {
        // println!("eeg_id={:?}", eeg_id.unwrap_or_default());
        let eeg_id = eeg_id.unwrap();
        let eeg_df = LazyFrame::scan_parquet(
            format!("{}/train_eegs/{}.parquet", "../../data", eeg_id),
            scan_conf.clone(),
        ).unwrap().collect().unwrap();

        // println!("eeg_id={:?} eeg_df.shape()={:?}", eeg_id, eeg_df.shape());

        eeg_df.shape().0
    }).sum())
}


#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use std::ops::Div;
    use std::time::Instant;

    use log::{debug, LevelFilter};
    use plotters::prelude::{GREEN, RED};
    use plotters::style::BLACK;
    use polars::export::ahash::{HashMap, HashMapExt};
    use polars::prelude::*;
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger, WriteLogger};
    use tch::Tensor;
    use hbac_motiffs::utils::*;

    use crate::*;
    use crate::data::{arr_to_expr, ByPatientIdSplitter, clc_records, EEG, EEGs, GenError, series_to_tensor, Set, split_frame_by_offsets, Splitter};
    use crate::utils::DataBatch;

    #[ctor::ctor]
    fn init() {
        CombinedLogger::init(
            vec![
                TermLogger::new(LevelFilter::Debug, Config::default(), TerminalMode::Mixed, ColorChoice::Auto)
            ]
        ).unwrap_or_else(|e|{
            println!("CombinedLogger::init failed: {:?}", e);
        });


        std::env::set_var("POLARS_FMT_MAX_COLS", "100");
        std::env::set_var("POLARS_FMT_MAX_ROWS", "100");
    }

    #[test]
    pub fn test_load_eegs() -> Result<(), GenError> {
        debug!("test_load_eegs");
        std::io::stdout().flush()?;

        // let good_eegs = EEGs::good_eegs("../../data")?;

        let eegs = EEGs::load_eegs (
            "../../data",
            // Some(col("eeg_id").eq(lit(1345588872))),
            Some(col("patient_id").is_in(lit(Series::new("eegs_", &[44615, 8033, 2641, 10111, 30539, 57503, 13252, 17637, 38015, 58327])))),
            // Some(col("eeg_id").is_in(lit(good_eegs))),
            // None,
            // Some(col("all_votes").list().max().eq(col("all_votes").list().sum())),
            Some(ByPatientIdSplitter::new(0.7)),
        )?;

        // assert_eq!(eegs.len(), 1);
        debug!("train eegs={:?}", eegs.train_set.len());
        debug!("test eegs={:?}", eegs.valid_set.len());


        // debug!("batch_x={:?}", eegs.TRAIN[0].egg_map[0].len());


        plot_series(
            vec![
                (format!("0"), BLACK, eegs.train_set[0].egg_map[0].iter().enumerate().map(|(i, v)|(i as f32, *v)).collect::<Vec<(f32, f32)>>()),

            ], "EEG", (10_000, 480), "0.png"
        ).unwrap();

        Ok(())
    }

    #[test]
    pub fn test_plot_eeg_tensor() -> Result<(), GenError> {
        // println!("tensor={:?}", tensor.size());

        let t_0 = data::eeg_to_tensor("../../data", "1628180742")?.f_slice(0, Some(0), Some(1), 1)?;
        let t_1 = data::eeg_to_tensor("../../data", "1019408895")?.f_slice(0, Some(0), Some(1), 1)?;

        // println!("tensor={:?}", tensor.size());

        //"1628180742".into(), "1019408895"
        plot_series(
            vec![
                (format!("1628180742"), BLACK, from::<Vec<f32>>(&t_0.reshape(-1)).iter().enumerate().map(|(i, v)|(i as f32, *v)).collect::<Vec<(f32, f32)>>()),
                (format!("1019408895"), RED, from::<Vec<f32>>(&t_1.reshape(-1)).iter().enumerate().map(|(i, v)|(i as f32, *v)).collect::<Vec<(f32, f32)>>()),

            ], "EEG", (10_000, 480), "/tmp/plots/1628180742_1019408895.png"
        ).unwrap();

        Ok(())
    }

    #[test]
    pub fn test_fetch_similarities() -> Result<(), GenError> {


        // let good_eegs = EEGs::good_eegs("../../data")?;
        let df = LazyCsvReader::new("../../data/train.csv")
            .finish()?
            .with_streaming(true)
            .group_by([col("eeg_id")])
            .agg([
                col("seizure_vote").first(),
                col("lpd_vote").first(),
                col("gpd_vote").first(),
                col("lrda_vote").first(),
                col("grda_vote").first(),
                col("other_vote").first(),
            ])
            .with_column(concat_list(["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"])?.alias("all_votes").alias("all_votes"))
            .filter(
                col("all_votes").list().max().alias("max").eq(col("all_votes").list().sum().alias("sum"))
            )
            .with_column(
                when(col("seizure_vote").gt(lit(0))).then(lit("seizure_vote"))
                    .otherwise(
                        when(col("lpd_vote").gt(lit(0))).then(lit("lpd_vote"))
                            .otherwise(
                                when(col("gpd_vote").gt(lit(0))).then(lit("gpd_vote"))
                                    .otherwise(
                                        when(col("lrda_vote").gt(lit(0))).then(lit("lrda_vote"))
                                            .otherwise(
                                                when(col("grda_vote").gt(lit(0))).then(lit("grda_vote"))
                                                    .otherwise(
                                                        lit("other_vote")
                                                    )
                                            )
                                    )
                            )
                    )
                    .alias("win_vote")
            )
            .group_by([col("win_vote")])
            .agg([count().alias("votes_count")])
            .collect()?;

        println!("df={:?}", df);

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
    pub fn test_split_frame_by_offsets() -> Result<(), GenError> {

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
    pub fn test_nth_fragment_to_tensor(){
        let mut eeg_map = vec![];

        eeg_map.push( (0..99).map(|v|v as f32).collect::<Vec<f32>>());
        eeg_map.push( (0..99).map(|v|(v * 10) as f32).collect::<Vec<f32>>());

        let eeg = EEG::new(
            eeg_map,
            1i64,
            0, vec![0, 3],
            0, 0, 0, 0, 0, 0
        );

        let t_1 = eeg.fragment_to_tensor(5, 10);

        t_1.print();

        assert_eq!(
            Tensor::stack(
                &[
                Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0]),
                Tensor::from_slice(&[50.0, 60.0, 70.0, 80.0, 90.0])
            ],
            0
            ),
            t_1);


    }

    #[test]
    pub fn test_eeg_to_tensor() -> Result<(), GenError> {
        // series_to_tensor()

        let base_path = "../../data";
        let eeg_id = "4454049";

        let t1 = Instant::now();
        let t = data::eeg_to_tensor(base_path, eeg_id)?;


        debug!("done in {}", t1.elapsed().as_millis());

        println!("t={:?}", t.size());

        assert_eq!([20, 17600], t.size().as_slice());

        Ok(())

    }



    // #[test]
    // pub fn test_load_by_std() -> Result<(), GenError> {
    //     //TODO https://github.com/pola-rs/polars/issues/12324
    //     let df: DataFrame = df!(
    //         "x" => 0..8,
    //         "y" => (0..8).rev().collect::<Vec<i32>>(),
    //         "z" => (0..8).map(|x|x^2).collect::<Vec<i32>>(),
    //     )?;
    //
    //     debug!("1. df={:?}", df);
    //
    //     // let df = df.lazy().with_columns([sum_horizontal(&[cols(["x", "y"])])]).collect()?;
    //     // let df = df.sum_horizontal(NullStrategy::Ignore)?;
    //
    //     debug!("2. df={:?}", df);
    //
    //     Ok(())
    // }
    #[test]
    pub fn test_test()-> Result<(), GenError> {
        // let good_eegs = crate::data::EEGs::good_eegs("../../data")?;

        // let df = LazyCsvReader::new(format!("{}/train.csv", "../../data"))
        //     .finish()?
        //     .with_streaming(true)
        //     .group_by([col("eeg_id")])
        //     .agg([
        //         col("seizure_vote").first(),
        //         col("lpd_vote").first(),
        //         col("gpd_vote").first(),
        //         col("lrda_vote").first(),
        //         col("grda_vote").first(),
        //         col("other_vote").first(),
        //     ])
        //     .with_column(concat_list(["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"])?.alias("all_votes").alias("all_votes"))
        //     .filter(
        //         col("all_votes").list().max().alias("max").eq(col("all_votes").list().sum().alias("sum"))
        //             .and(col("seizure_vote").gt(lit(0)))
        //     )
        //     .collect()?;

        let eeg_id_series = EEGs::good_eegs_of("../../data", "seizure_vote")?;

        let eeg_id_series = eeg_id_series.limit(10);


        let records_num:usize = clc_records(&eeg_id_series)?;

        println!("eeg_id_series={} records_num={:?}", eeg_id_series.len(), records_num);

        // let df = LazyCsvReader::new("../../data/train.csv")
        //     .finish()?
        //     .with_streaming(true);
        //
        // let mut df_sorted = df.clone().group_by(["patient_id"]).agg([
        //     col("seizure_vote").sum(),
        //     col("lpd_vote").sum(),
        //     col("gpd_vote").sum(),
        //     col("lrda_vote").sum(),
        //     col("grda_vote").sum(),
        //     col("other_vote").sum(),
        // ]).sort_by_exprs([
        //   col("seizure_vote")
        //   +col("lpd_vote")
        //   +col("gpd_vote")
        //   +col("lrda_vote")
        //   +col("grda_vote")
        //   +col("other_vote")
        // ], vec![false], true, true)
        //     .select([col("patient_id")])
        //     .collect()?;
        //
        // let df_set = df_sorted.with_column(Series::new("set", (0..df_sorted.shape().0).map(|i| if rand::random::<f32>() > 0.7 {1i32} else {0i32}).collect::<Vec<_>>()))?;
        //
        // let df_set = df.clone().join(df_set.clone().lazy(), [col("patient_id")], [col("patient_id")], JoinArgs::new(JoinType::Left)).collect()?;
        //
        // debug!("{:?}", df_set.head(Some(100)));

        /*let df = df!(
                        // 30   20    40    10    50
            "field_1" => [ 3,    4,   2,    5,    1],
            "field_2" => [ 27,  16,   38,   5,    49],
            "field_3" => ["c",  "b",  "d",  "a", "e"]
        )?;

        let df_sorted = df.lazy()
            .sort_by_exprs([
            col("field_1") + col("field_2")
            ], vec![false], true, true)
            .collect()?;

        debug!("df_sorted={:?}", df_sorted);*/


        // let choices = [1, 2, 4, 8, 16, 32, 3, 7];
        // let mut rng = thread_rng();
        //
        // let we = 1./5.;
        // let wo = 1./3.;
        //
        // let wen = we / (we + wo);
        // let won = wo / (we + wo);
        //
        // debug!("{:?}", choices.choose_multiple_weighted(&mut rng, 4, |x| {
        //     debug!("x={}, x%2={}  wen={}, won={}", x, x%2, wen, won);
        //     if x % 2 == 0 {wen} else {won}
        // } )?.map(|i|*i).collect::<Vec<i32>>());

       /* let df = DataFrame::new(vec![
            Series::new("groups", &(0..100).map(|i| if i % 10 == 0 {"a"} else { if i % 3 == 0 {"c"} else {"b"}}).collect::<Vec<&str>>().as_slice()),
            Series::new("values", &(0..100).collect::<Vec<i32>>().as_slice())
        ])?;

        let mut df = df.clone();
        let df = df.with_column(Series::new("set", (0..df.shape().0).map(|i| if rand::random::<f32>() > 0.7 {1i32} else {0i32}).collect::<Vec<_>>()))?;

        let dfs = df.partition_by(["set"], false)?;

        // debug!("df={:?}", df.head(Some(10)));
        debug!("dfs={:?}", dfs);*/

        // let df_filt = df
        //     .lazy()
        //     .filter(col("groups").is_in(lit(Series::new("OOOO", &["a", "b"]))))
        //     .select([all()])
        //     .collect()?;
        //
        // debug!("df={:?}", df_filt.head(Some(10)));

        // let grouped = df
        //     // .sample_frac(&Series::new("frac", &[0.5]), true, true, None)?
        //     .lazy()
        //         .group_by([col("field_1")])
        //         .agg([col("groups").first(), col("values").count().alias("count")])
        //         .collect()?;

        // debug!("grouped={:?}", grouped);

        // let over_gr = df.lazy().with_columns(col("field_1").count().over("field_2")).collect()?;
        // let mprob = |s:&mut [Series]| {
        //     debug!("s1={:?}", s);
        //     Ok(Some(
        //             (0..10).map(|v|Some(v as f32)).collect::<Float32Chunked>().into_series()
        //     ))
        // };

        // let over_gr = df.lazy()
        //     .with_columns([
        //         col("values").count().over([col("groups")]).alias("values_in_group_count"),
        //         col("values").count().alias("total_count")
        //     ])
        //     .select(&[
        //         col("groups"),
        //         col("total_count"),
        //         col("values_in_group_count").cast(DataType::Float32)/col("total_count").cast(DataType::Float32) //.div(col("values").count()) //.map_many(mprob, &[col("group_count")], GetOutput::from_type(DataType::Float32)).alias("xxx")
        //     ])
        //     .collect()?;

        // debug!("over_gr={:?}", over_gr);
        // df.with_row_count("id", None).partition_by("id");

        // for field in df.clone().lazy().schema()?.iter_fields() {
        //     debug!("{:?}", field.name);
        //     let column_df = df.clone().lazy().select(&[col(field.name.as_str())]).collect()?;
        //     // column_df.column(field.name.as_str())?.f64().iter().collect();
        // }

        // for s in df.get_columns() {
        //     let t = Tensor::from_slice(s.f64()?.to_vec().iter().map(|v| v.unwrap_or_default()).collect::<Vec<f64>>().as_slice());
        //     t.print()
        // }


        Ok(())
    }

}
