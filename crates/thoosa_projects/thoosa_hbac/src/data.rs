use log::debug;
use polars::datatypes::DataType;
use polars::frame::DataFrame;
use polars::prelude::{
    col, concat_list, lit, BooleanChunked, Expr, IntoLazy, JoinArgs,
    JoinType, LazyCsvReader, LazyFileListReader, LazyFrame, NamedFrom, ScanArgsParquet, Series, SortMultipleOptions,
};
use rand::{Rng, RngExt};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tch::{Device, Kind, Tensor};

use thoosa_common::dims::{Batch, ChannelsData};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_main::data::{parquet_to_tensor_by_path, AllTrainSplitter, Set, Splitter};
use thoosa_shared::ThoError;
use thoosa_shared::traits::SimpleId;

#[allow(dead_code)]
pub struct ByPatientIdSplitter {
    train_frac: f32,
    train_patients: Vec<i64>,
    test_patients: Vec<i64>,
}

impl ByPatientIdSplitter {
    #[allow(dead_code)]
    pub fn new(train_frac: f32) -> Self {
        Self {
            train_frac: train_frac,
            train_patients: vec![],
            test_patients: vec![],
        }
    }
}

impl Splitter for ByPatientIdSplitter {
    type Type = EEG;

    fn split_to(&self, eeg: &Self::Type) -> Set {
        if eeg.set == 1 { Set::Valid } else { Set::Train }
    }

    fn init(&self, df: LazyFrame) -> Result<LazyFrame, ThoError> {
        let mut df_sorted = df
            .clone()
            .group_by(["patient_id"])
            .agg([
                col("seizure_vote").sum(),
                col("lpd_vote").sum(),
                col("gpd_vote").sum(),
                col("lrda_vote").sum(),
                col("grda_vote").sum(),
                col("other_vote").sum(),
            ])
            .sort_by_exprs(
                [col("seizure_vote")
                    + col("lpd_vote")
                    + col("gpd_vote")
                    + col("lrda_vote")
                    + col("grda_vote")
                    + col("other_vote")],
                SortMultipleOptions {
                    descending: vec![false],
                    maintain_order: true,
                    nulls_last: vec![true],
                    multithreaded: false,
                    limit: None,
                },
            )
            .select([col("patient_id")])
            .collect()?;

        let df_set = df_sorted.with_column(Series::new(
            "set".into(),
            (0..df_sorted.shape().0)
                .map(|_i| {
                    if rand::random::<f32>() > self.train_frac {
                        1i32
                    } else {
                        0i32
                    }
                })
                .collect::<Vec<_>>(),
        ))?;

        Ok(df.clone().join(
            df_set.clone().lazy(),
            [col("patient_id")],
            [col("patient_id")],
            JoinArgs::new(JoinType::Left),
        ))
        // Ok(df.clone().with_column(lit(1).alias("set"))) //.collect()?.lazy()
    }
}

pub fn parquet_to_tensor(
    base_path: &str,
    resource_id: &str,
    device: Device,
) -> Result<Tensor, ThoError> {
    let path = format!("{}/train_eegs/{}.parquet", base_path, resource_id);

    parquet_to_tensor_by_path(path, device)
}

pub fn labeled_resourses(
    base: impl AsRef<Path>,
    file: impl AsRef<Path>,
) -> Result<HashMap<String, Vec<(String, f32)>>, ThoError> {
    let path = base.as_ref().join(file);
    let eeg_path = path.clone().parent().unwrap().to_path_buf();
    let df = LazyCsvReader::new(path).finish()?.with_streaming(true);

    let gr_by_eeg_id_df = EEGs::group_by_eeg_id(None, df)?;

    let mask: BooleanChunked = gr_by_eeg_id_df
        .column("eeg_id")?
        .i64()?
        .into_iter()
        .map(|opt| {
            opt.map(|id| {
                let p = PathBuf::from(format!("{}/train_eegs/{id}.parquet", eeg_path.display()));
                let b = p.exists();
                debug!("id={id:?} p={p:?} b={b:?}");
                b
            })
            .unwrap_or(false)
        })
        .collect();

    Ok(gr_by_eeg_id_df
        .filter(&mask)?
        .column("eeg_id")?
        .i64()?
        .into_iter()
        // .zip(gr_by_eeg_id_df.column("set")?.i32()?)
        // .zip(gr_by_eeg_id_df.column("eeg_label_offset_seconds")?.list()?)
        .zip(gr_by_eeg_id_df.column("seizure_vote")?.i64()?)
        .zip(gr_by_eeg_id_df.column("lpd_vote")?.i64()?)
        .zip(gr_by_eeg_id_df.column("gpd_vote")?.i64()?)
        .zip(gr_by_eeg_id_df.column("lrda_vote")?.i64()?)
        .zip(gr_by_eeg_id_df.column("grda_vote")?.i64()?)
        .zip(gr_by_eeg_id_df.column("other_vote")?.i64()?)
        .map(
            |(
                (
                    (
                        (
                            (
                                (
                                    eeg_id,
                                    // eeg_label_offset_seconds),
                                    seizure_vote,
                                ),
                                lpd_vote,
                            ),
                            gpd_vote,
                        ),
                        lrda_vote,
                    ),
                    grda_vote,
                ),
                other_vote,
            )| {
                (
                    eeg_id.unwrap_or_default(),
                    // eeg_label_offset_seconds.unwrap_or_default(),
                    seizure_vote.unwrap_or_default(),
                    lpd_vote.unwrap_or_default(),
                    gpd_vote.unwrap_or_default(),
                    lrda_vote.unwrap_or_default(),
                    grda_vote.unwrap_or_default(),
                    other_vote.unwrap_or_default(),
                )
            },
        )
        .map(
            |(eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote)| {
                let sum = (seizure_vote + lpd_vote + gpd_vote + lrda_vote + grda_vote + other_vote)
                    as f32;

                (
                    eeg_id.to_string(),
                    [
                        ("seizure".to_string(), seizure_vote as f32 / sum),
                        ("lpd".to_string(), lpd_vote as f32 / sum),
                        ("gpd".to_string(), gpd_vote as f32 / sum),
                        ("lrda".to_string(), lrda_vote as f32 / sum),
                        ("grda".to_string(), grda_vote as f32 / sum),
                        ("other".to_string(), other_vote as f32 / sum),
                    ]
                    .into_iter()
                    .filter(|v| v.1 > 0.)
                    .collect(),
                )
            },
        )
        .fold(HashMap::new(), |mut h, v| {
            h.insert(v.0, v.1);
            h
        }))
}

// #[test] TODO
#[allow(dead_code)]
pub fn eegs_to_tensors() -> Result<(), ThoError> {
    // for f in fs::read_dir("/media/bu/KINGSTON/gpd_fragments/3/1")? {
    //     let entry = f?;
    //
    //     let filename:String = entry.file_name().to_str().unwrap().to_string();
    //
    //     let filname = filename.replace(".pt", "");
    //     let eeg_split = filname.split("_").collect::<Vec<&str>>();
    //     debug!("{:?}", eeg_split);
    //
    //     let t = data::eeg_to_tensor("../../data", eeg_split[0], Device::Cpu)?;
    //
    //     fs::create_dir_all("/tmp/eegs_for_gpd_fragments_3_1");
    //     t.save(format!("/tmp/eegs_for_gpd_fragments_3_1/{}.pt", eeg_split[0]));
    //
    // }
    // let (Some(name), eegs) = read_eeg_index("../../data_down/14_02_results/2024-02-19T07:12:01Z")? else {
    //     panic!("no eeg index")
    // };

    /*let save_to = format!("/media/bu/KINGSTON/all_eegs.pt");
    fs::create_dir_all(save_to.clone());
    // for eeg in eegs {
    for (i, f) in fs::read_dir("../../data/train_eegs")?.enumerate() {
        if i % 1000 == 0 {
            debug!("{} done", i);
            std::io::stdout().flush()?;
        }
        let entry = f?;

        let t = data::eeg_to_tensor_by_path(entry.path(), Device::Cpu)?;

        t.save(format!("{}/{}.pt", save_to, entry.file_name().to_str().map(|v|v.replace(".parquet", ".pt")).unwrap_or(i.to_string())));
    }*/

    // debug!("saved to {}", name);

    let df = LazyCsvReader::new("../../data/train.csv")
        .finish()?
        .with_streaming(true);

    debug!("df={:?}", df.clone().collect()?.head(Some(10)));

    let gr_by_eeg_id_df = EEGs::group_by_eeg_id(None, df)?;

    debug!("gr_by_eeg_id_df.shape()[0]={}", gr_by_eeg_id_df.shape().0);
    std::io::stdout().flush()?;

    let mut count = 0;

    fs::create_dir_all("/media/bu/KINGSTON/all_eegs.pt")?;

    debug!("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");

    for (eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote) in
        gr_by_eeg_id_df
            .column("eeg_id")?
            .i64()?
            .into_iter()
            // .zip(gr_by_eeg_id_df.column("set")?.i32()?)
            // .zip(gr_by_eeg_id_df.column("eeg_label_offset_seconds")?.list()?)
            .zip(gr_by_eeg_id_df.column("seizure_vote")?.i64()?)
            .zip(gr_by_eeg_id_df.column("lpd_vote")?.i64()?)
            .zip(gr_by_eeg_id_df.column("gpd_vote")?.i64()?)
            .zip(gr_by_eeg_id_df.column("lrda_vote")?.i64()?)
            .zip(gr_by_eeg_id_df.column("grda_vote")?.i64()?)
            .zip(gr_by_eeg_id_df.column("other_vote")?.i64()?)
            .map(
                |(
                    (
                        (
                            (
                                (
                                    (
                                        eeg_id,
                                        // eeg_label_offset_seconds),
                                        seizure_vote,
                                    ),
                                    lpd_vote,
                                ),
                                gpd_vote,
                            ),
                            lrda_vote,
                        ),
                        grda_vote,
                    ),
                    other_vote,
                )| {
                    (
                        eeg_id.unwrap_or_default(),
                        // eeg_label_offset_seconds.unwrap_or_default(),
                        seizure_vote.unwrap_or_default(),
                        lpd_vote.unwrap_or_default(),
                        gpd_vote.unwrap_or_default(),
                        lrda_vote.unwrap_or_default(),
                        grda_vote.unwrap_or_default(),
                        other_vote.unwrap_or_default(),
                    )
                },
            )
    {
        if count % 1000 == 0 {
            debug!("{} done", count);
            std::io::stdout().flush()?;
        }

        // debug!("eeg_id={} {} {} {} {} {} {}",
        //        eeg_id,
        //        TryInto::<i64>::try_into(seizure_vote).expect("seizure_vote"),
        //        TryInto::<i64>::try_into(lpd_vote).expect("lpd_vote"),
        //        TryInto::<i64>::try_into(gpd_vote).expect("gpd_vote"),
        //        TryInto::<i64>::try_into(lrda_vote).expect("lrda_vote"),
        //        TryInto::<i64>::try_into(grda_vote).expect("grda_vote"),
        //        TryInto::<i64>::try_into(other_vote).expect("other_vote")
        // );
        // std::io::stdout().flush()?;

        let t = parquet_to_tensor("../../data", eeg_id.to_string().as_str(), Device::Cpu)?;

        let path = format!(
            "{}/{}_{}_{}_{}_{}_{}_{}.pt",
            "/media/bu/KINGSTON/all_eegs.pt",
            eeg_id,
            TryInto::<i64>::try_into(seizure_vote).expect("seizure_vote"),
            TryInto::<i64>::try_into(lpd_vote).expect("lpd_vote"),
            TryInto::<i64>::try_into(gpd_vote).expect("gpd_vote"),
            TryInto::<i64>::try_into(lrda_vote).expect("lrda_vote"),
            TryInto::<i64>::try_into(grda_vote).expect("grda_vote"),
            TryInto::<i64>::try_into(other_vote).expect("other_vote")
        );
        debug!("{}", path);
        t.save(path)?;

        count += 1;
    }

    debug!("count={}", count);

    Ok(())
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct EEG {
    egg_map: Vec<Vec<f32>>,
    eeg_id: i64,
    set: i32,
    fragments_rows: Vec<u32>,
    seizure_vote: u8,
    lpd_vote: u8,
    gpd_vote: u8,
    lrda_vote: u8,
    grda_vote: u8,
    other_vote: u8,
}

impl EEG {
    pub fn new(
        egg_map: Vec<Vec<f32>>,
        eeg_id: i64,
        set: i32,
        fragments_rows: Vec<u32>,
        seizure_vote: u8,
        lpd_vote: u8,
        gpd_vote: u8,
        lrda_vote: u8,
        grda_vote: u8,
        other_vote: u8,
    ) -> Self {
        Self {
            egg_map,
            eeg_id,
            set,
            fragments_rows,
            seizure_vote,
            lpd_vote,
            gpd_vote,
            lrda_vote,
            grda_vote,
            other_vote,
        }
    }

    pub fn random_fragemnt_tensor<R>(&self, rnd: &mut R) -> Tensor
    where
        R: Rng + ?Sized,
    {
        let fragment_idx = rnd.random_range(0..self.fragments_rows.len());

        let i_1 = self.fragments_rows[fragment_idx] as usize;
        let i_2 = i_1 + 50 * 200;

        self.fragment_to_tensor(i_1, i_2)
    }

    pub fn all_fragemnts_tensor(&self) -> Vec<Tensor> {
        self.fragments_rows
            .iter()
            .map(|i| self.fragment_to_tensor(*i as usize, *i as usize + 50 * 200))
            .collect()
    }

    pub fn votes_tensor(&self) -> Tensor {
        Tensor::from_slice(&[
            self.seizure_vote,
            self.lpd_vote,
            self.gpd_vote,
            self.lrda_vote,
            self.grda_vote,
            self.other_vote,
        ])
        .totype(Kind::Float)
    }

    pub fn fragment_to_tensor(&self, i_1: usize, i_2: usize) -> Tensor {
        assert!(i_1 < i_2, "fragment_to_tensor: i_1 < i_2");

        Tensor::stack(
            self.egg_map
                .iter()
                .map(|values| Tensor::from_slice(&values[i_1..i_2]))
                .collect::<Vec<Tensor>>()
                .as_slice(),
            0,
        )
    }
}

#[derive(Debug)]
pub struct EEGs {
    train_set: Vec<EEG>,
    _valid_set: Vec<EEG>,
    _batch_size: usize,
}

#[allow(dead_code)]
impl EEGs {
    pub fn good_eegs(base_path: &str) -> Result<Series, ThoError> {
        let good_eegs_filter = Self::good_eegs_filter(base_path)?;

        Ok(good_eegs_filter
            .collect()?
            .column("eeg_id")?
            .as_series()
            .map(|v| v.clone())
            .unwrap())
    }

    pub fn good_eegs_of(
        base_path: &str,
        kind: &str,
        limit: Option<usize>,
    ) -> Result<Series, ThoError> {
        let good_eegs_filter = Self::good_eegs_filter(base_path)?;

        let eeg_id_series = good_eegs_filter
            .filter(col(kind).gt(lit(0)))
            // .sort(kind, SortOptions{ descending: true, ..Default::default()})
            // .sort("eeg_id", Default::default())
            // .sort_by_exprs(vec![col(kind), col("eeg_id")], vec![true, false], true, true)
            .sort_by_exprs(
                vec![col(kind), col("eeg_id")],
                SortMultipleOptions {
                    descending: vec![true, false],
                    maintain_order: true,
                    nulls_last: vec![true],
                    multithreaded: false,
                    limit: None,
                },
            )
            .collect()?
            .column("eeg_id")?
            .clone();

        let s = eeg_id_series.as_series().unwrap();
        Ok(match limit {
            None => s.clone(),
            Some(v) => s.limit(v),
        })
    }

    fn good_eegs_filter(base_path: &str) -> Result<LazyFrame, ThoError> {
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
            .with_column(
                concat_list([
                    "seizure_vote",
                    "lpd_vote",
                    "gpd_vote",
                    "lrda_vote",
                    "grda_vote",
                    "other_vote",
                ])?
                .alias("all_votes")
                .alias("all_votes"),
            )
            .filter(
                col("all_votes")
                    .list()
                    .max()
                    .alias("max")
                    .eq(col("all_votes").list().sum().alias("sum")),
            ))
    }

    pub fn load_all_eegs(base_path: &str) -> Result<Box<EEGs>, ThoError> {
        EEGs::load_eegs::<_>(base_path, None, None::<AllTrainSplitter<EEG>>)
    }
    pub fn load_eegs<S>(
        base_path: &str,
        filter: Option<Expr>,
        splitter: Option<S>,
    ) -> Result<Box<EEGs>, ThoError>
    where
        S: Splitter<Type = EEG>,
    {
        let df = LazyCsvReader::new(format!("{}/train.csv", base_path))
            .finish()?
            .with_streaming(true);

        let df = match splitter.as_ref() {
            None => df,
            Some(splitter) => splitter.init(df)?,
        };

        debug!("df={:?}", df.clone().collect()?.head(Some(10)));

        let gr_by_eeg_id_df = Self::group_by_eeg_id(filter, df)?;

        debug!("gr_by_eeg_id_df.shape()[0]={}", gr_by_eeg_id_df.shape().0);
        std::io::stdout().flush()?;

        let mut train_eegs = Vec::with_capacity(gr_by_eeg_id_df.shape().0); //TODO capacity
        let mut valid_eegs = Vec::with_capacity(gr_by_eeg_id_df.shape().0);

        for (
            eeg_id,
            set,
            eeg_label_offset_seconds,
            seizure_vote,
            lpd_vote,
            gpd_vote,
            lrda_vote,
            grda_vote,
            other_vote,
        ) in gr_by_eeg_id_df
            .column("eeg_id")?
            .i64()?
            .into_iter()
            .zip(gr_by_eeg_id_df.column("set")?.i32()?)
            .zip(gr_by_eeg_id_df.column("eeg_label_offset_seconds")?.list()?)
            .zip(gr_by_eeg_id_df.column("seizure_vote")?.i64()?)
            .zip(gr_by_eeg_id_df.column("lpd_vote")?.i64()?)
            .zip(gr_by_eeg_id_df.column("gpd_vote")?.i64()?)
            .zip(gr_by_eeg_id_df.column("lrda_vote")?.i64()?)
            .zip(gr_by_eeg_id_df.column("grda_vote")?.i64()?)
            .zip(gr_by_eeg_id_df.column("other_vote")?.i64()?)
            .map(
                |(
                    (
                        (
                            (
                                (
                                    (((eeg_id, set), eeg_label_offset_seconds), seizure_vote),
                                    lpd_vote,
                                ),
                                gpd_vote,
                            ),
                            lrda_vote,
                        ),
                        grda_vote,
                    ),
                    other_vote,
                )| {
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
                },
            )
        {
            let fragments_rows = eeg_label_offset_seconds
                .f64()?
                .into_iter()
                .map(|v| v.expect("no id :(") as u32 * 200)
                .collect::<Vec<u32>>();

            // debug!("eeg_id ={:?}, fragments_in_mls={:?} seizure_vote={:?}  lpd_vote={:?}, gpd_vote={:?}, lrda_vote={:?}, grda_vote={:?}, other_vote={:?}",
            //                 eeg_id, fragments_in_mls,     seizure_vote,      lpd_vote,      gpd_vote,       lrda_vote,      grda_vote,      other_vote);

            let scan_conf = ScanArgsParquet::default();
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

                match s.dtype() {
                    DataType::Float32 => {
                        eegs_vec.push(
                            s.f32()?
                                .downcast_iter()
                                .flat_map(|arr| {
                                    arr.into_iter().map(|v| v.map(|v| *v).unwrap_or(0.0))
                                })
                                .collect::<Vec<f32>>(),
                        );
                    }
                    DataType::Float64 => {
                        eegs_vec.push(
                            s.f64()?
                                .downcast_iter()
                                .flat_map(|arr| {
                                    arr.into_iter().map(|v| v.map(|v| *v as f32).unwrap_or(0.0))
                                })
                                .collect::<Vec<f32>>(),
                        );
                    }
                    _ => {
                        todo!()
                    }
                }
            }

            let eeg = EEG::new(
                eegs_vec,
                eeg_id,
                set,
                fragments_rows,
                seizure_vote.try_into().expect("seizure_vote"),
                lpd_vote.try_into().expect("lpd_vote"),
                gpd_vote.try_into().expect("gpd_vote"),
                lrda_vote.try_into().expect("lrda_vote"),
                grda_vote.try_into().expect("grda_vote"),
                other_vote.try_into().expect("other_vote"),
            );

            match splitter
                .as_ref()
                .map(|s| s.split_to(&eeg))
                .unwrap_or_default()
            {
                Set::Train => {
                    train_eegs.push(eeg);
                }
                Set::Valid => {
                    valid_eegs.push(eeg);
                }
            }


        }

        debug!(
            "tensors.len={}  valid_eegs.len={}",
            train_eegs.len(),
            valid_eegs.len()
        );
        std::io::stdout().flush()?;

        Ok(Box::new(EEGs {
            train_set: train_eegs,
            _valid_set: valid_eegs,
            _batch_size: 10, //TODO param
        }))
    }

    pub fn group_by_eeg_id(filter: Option<Expr>, df: LazyFrame) -> Result<DataFrame, ThoError> {
        Ok(df
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
                col("eeg_label_offset_seconds").alias("eeg_label_offset_seconds"),
                // col("set").first(),
                col("seizure_vote").first(),
                col("lpd_vote").first(),
                col("gpd_vote").first(),
                col("lrda_vote").first(),
                col("grda_vote").first(),
                col("other_vote").first(),
            ])
            .collect()?)
    }

    pub fn len(&self) -> usize {
        self.train_set.len()
    }

    #[allow(dead_code)]
    fn all_set(data: &Vec<EEG>) -> (Tensor, Tensor) {
        let (eegs, votes) = data.iter().fold((vec![], vec![]), |mut a, eeg| {
            for fragemnt_tensor in eeg.all_fragemnts_tensor() {
                a.0.push(fragemnt_tensor);
                a.1.push(eeg.votes_tensor());
            }

            a
        });

        (
            Tensor::stack(eegs.as_slice(), 0),
            Tensor::stack(votes.as_slice(), 0),
        )
    }

    fn random_subset(
        batch_size: usize,
        data: &Vec<EEG>,
    ) -> (
        ThoTensor<Batch<ChannelsData>, SimpleId>,
        Tensor,
    ) {
        let mut rng = rand::rng();

        use std::collections::HashSet;
        let mut picked = HashSet::with_capacity(batch_size);
        let mut eegs = Vec::with_capacity(batch_size);
        let mut votes = Vec::with_capacity(batch_size);

        while eegs.len() < batch_size {
            let idx = rng.random_range(0..data.len());
            if picked.insert(idx) {
                let eeg = &data[idx];
                let fragemnt_tensor = eeg.random_fragemnt_tensor(&mut rng);
                eegs.push(fragemnt_tensor);
                votes.push(eeg.votes_tensor());
            }
        }

        (
            Tensor::stack(eegs.as_slice(), 0).into(),
            Tensor::stack(votes.as_slice(), 0),
        )
    }
}
#[cfg(test)]
mod tests {
    #![allow(unused_imports, dead_code, unused_variables)]
    use super::{parquet_to_tensor, ByPatientIdSplitter, EEG};
    use crate::data::EEGs;
    use std::collections::HashMap;
    use log::LevelFilter::Trace;
    use log::debug;
    use parking_lot::{Mutex, RwLock};
    use plotters::count;
    use plotters::prelude::{BLACK, RED};
    use polars::prelude::{
        col, concat_list, lit, when, LazyCsvReader, LazyFileListReader, NamedFrom, Series,
    };
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::UNIX_EPOCH;
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::Float;
    use tch::{Device, IndexOp, TchError, Tensor};
    use thoosa_main::tasks::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::tho_device::ThoDevice;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{CommonTrainingRunConfigFields, MotifConvNetArgsFields, MotifCosineNetArgsFields, MotifCosineNetParams, MotifLinNetArgsFields, MotifNetParams, MotifNetTypeArgsFields, MotifNetTypeParams, OptConfig, TrainMotifNetRunConfigFields};
    use thoosa_common::transforms::filters::{InputFilter, NotchBandFilterParams, Padding, SavGolFilterParams};
    use thoosa_common::utils::{assert_close, from, hist_1_dim, lin_t, plot_series, skewness_kurtosis_1_dim, to_vec_f32};
    use thoosa_common::ws::Workspace;
    use thoosa_common::ws::{mock_ws_simp, FsWorkspace};
    use thoosa_main::cli::Commands::{FindMotifs, SelectMotifs, TrainFinalNet, TrainLabelNets};
    use thoosa_main::cli::{import_resources, Cli};
    use thoosa_main::{ApplyMotifNets, ChannelsData2, RawTensorLoader, TrainMotifNetInput, TrainMotifNets};
    use thoosa_motifs::distance::MaskMaker::LeQuantile;
    use thoosa_motifs::distance::clc_d2_similarity_and_mask;
    use thoosa_motifs::find::SelectMotifsParams;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::hyperparameter::Hyperparameter;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::{ConstConfigs, GridRunConfigs, OptRunConfigs, RunId};
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::runner::common::mock_run_ctx;
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::common::{MockRunCtx, RunCtx};
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_main::data::parquet_to_tensor_by_path;
    use thoosa_main::random_non_overlapping_slices;
    use thoosa_shared::test_utils::init_test_logging;
    use thoosa_shared::val::{NamedRange, NamedVal, Val};
    use thoosa_shared::TensorIdSliceIdx;
    use thoosa_shared::{cdt_for_dir, ThoError};
    use thoosa_store::{MockStoreBuilder, SqlStore};
    use thoosa_shared::mono_key;
    use thoosa_store::rusqlite::Transaction;

    #[ctor::ctor]
    fn init() {
        init_test_logging();
    }


    // #[test]
    pub fn test_fetch_similarities() -> Result<(), ThoError> {
        // let good_eegs = EEGs::good_eegs("../../data")?;
        let df = LazyCsvReader::new("../../../data/train.csv")
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
            .with_column(
                concat_list([
                    "seizure_vote",
                    "lpd_vote",
                    "gpd_vote",
                    "lrda_vote",
                    "grda_vote",
                    "other_vote",
                ])?
                .alias("all_votes")
                .alias("all_votes"),
            )
            .filter(
                col("all_votes")
                    .list()
                    .max()
                    .alias("max")
                    .eq(col("all_votes").list().sum().alias("sum")),
            )
            .with_column(
                when(col("seizure_vote").gt(lit(0)))
                    .then(lit("seizure_vote"))
                    .otherwise(
                        when(col("lpd_vote").gt(lit(0)))
                            .then(lit("lpd_vote"))
                            .otherwise(
                                when(col("gpd_vote").gt(lit(0)))
                                    .then(lit("gpd_vote"))
                                    .otherwise(
                                        when(col("lrda_vote").gt(lit(0)))
                                            .then(lit("lrda_vote"))
                                            .otherwise(
                                                when(col("grda_vote").gt(lit(0)))
                                                    .then(lit("grda_vote"))
                                                    .otherwise(lit("other_vote")),
                                            ),
                                    ),
                            ),
                    )
                    .alias("win_vote"),
            )
            .group_by([col("win_vote")])
            .agg([col("*").count().alias("votes_count")])
            .collect()?;

        debug!("df={:?}", df);

        //TODO hash + assert
        df.column("win_vote")?
            .str()?
            .into_iter()
            .zip(df.column("votes_count")?.u32()?.into_iter())
            .for_each(|v| {
                debug!("v={:?}", v);
            });

        Ok(())
    }

    #[allow(dead_code)]
    // #[test] TODO to utils
    pub fn test_load_eegs() -> Result<(), ThoError> {
        debug!("test_load_eegs");
        std::io::stdout().flush()?;

        // let good_eegs = EEGs::good_eegs("../../data")?;

        let eegs = EEGs::load_eegs(
            "../../data",
            // Some(col("eeg_id").eq(lit(1345588872))),
            Some(col("patient_id").is_in(
                lit(Series::new(
                    "eegs_".into(),
                    &[
                        44615, 8033, 2641, 10111, 30539, 57503, 13252, 17637, 38015, 58327,
                    ],
                )),
                true,
            )),
            // Some(col("eeg_id").is_in(lit(good_eegs))),
            // None,
            // Some(col("all_votes").list().max().eq(col("all_votes").list().sum())),
            Some(ByPatientIdSplitter::new(0.7)),
        )?;

        // assert_eq!(eegs.len(), 1);
        debug!("train eegs={:?}", eegs.train_set.len());
        debug!("test eegs={:?}", eegs._valid_set.len());

        // debug!("batch_x={:?}", eegs.TRAIN[0].egg_map[0].len());

        plot_series(
            vec![(
                format!("0"),
                BLACK,
                eegs.train_set[0].egg_map[0]
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i as f32, *v))
                    .collect::<Vec<(f32, f32)>>(),
            )],
            "EEG",
            (10_000, 480),
            &"0.png",
        )
        .unwrap();

        Ok(())
    }

    #[test]
    pub fn test_nth_fragment_to_tensor() {
        let mut eeg_map = vec![];

        eeg_map.push((0..99).map(|v| v as f32).collect::<Vec<f32>>());
        eeg_map.push((0..99).map(|v| (v * 10) as f32).collect::<Vec<f32>>());

        let eeg = EEG::new(eeg_map, 1i64, 0, vec![0, 3], 0, 0, 0, 0, 0, 0);

        let t_1 = eeg.fragment_to_tensor(5, 10);

        assert_eq!(
            Tensor::stack(
                &[
                    Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0, 9.0]),
                    Tensor::from_slice(&[50.0, 60.0, 70.0, 80.0, 90.0])
                ],
                0
            ),
            t_1
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_import_resources() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 3, 1000, 1000)));

        // fs::remove_file("/tmp/test_main_db.db3").unwrap_or(());
        // let store = Arc::new(SqlStore::from_path("/tmp/test_main_db.db3")?);
        let store = Arc::new(SqlStore::in_mem().await?);

        let _resources = import_resources(
            &ws,
            &store,
            &["seizure", "lpd", "gpd", "lrda", "grda", "other"],
            || super::labeled_resourses(Path::new("test_data/eeg"), "train_10.csv"),
            // None::<Box<dyn Fn(String) -> Result<Tensor, ThoosaError> + Sync + Send>>,
            |key| {

                parquet_to_tensor_by_path(format!("test_data/eeg/train_eegs/{}.parquet", key).as_str(), Cpu)
                    .map_err(|e| e.into())
            },
            None,
            true,
            |t: &Transaction|{
                Ok(())
            },
        ).await?;

        let labels = store.list_labels(false).await?;

        assert_eq!(
            vec!["seizure", "lpd", "gpd", "lrda", "grda", "other"],
            labels
        );

        Ok(())
    }

    fn _test_cosine_net_run_configs(kernel: Val, cycles: Val, batch_size: Val) -> OptRunConfigs<Vec<NamedVal<TrainMotifNetRunConfigFields>>, Vec<NamedRange<TrainMotifNetRunConfigFields>>, Vec<Hyperparameter<TrainMotifNetRunConfigFields>>, TrainMotifNetRunConfigFields> {
        OptRunConfigs::new(
            GridRunConfigs::new(
                ConstConfigs::new(vec![
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Epochs), cycles),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::BatchSize), batch_size),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Patience), Val::Int(20)),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::MinDelta), Val::Real(1e-4)),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), Val::Real(0.00001)),
                ]),
                vec![],
            ),
            vec![
                // Hyperparameter::UniformFloat { name: TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), lower: 0.00001, upper: 0.0001 },
                // Hyperparameter::UniformInt { name: TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::CosineNet(MotifCosineNetArgsFields::Kernel)), lower: 10, upper: 100 },
                Hyperparameter::UniformFloat { name: TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::CosineNet(MotifCosineNetArgsFields::A)), lower: 1.0, upper: 10.0 },
                Hyperparameter::UniformFloat { name: TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::CosineNet(MotifCosineNetArgsFields::B)), lower: 0.0, upper: 0.1 },
            ],
        )
    }

    fn _test_lin_net_run_configs(kernel: Val, cycles: Val, batch_size: Val) -> OptRunConfigs<Vec<NamedVal<TrainMotifNetRunConfigFields>>, Vec<NamedRange<TrainMotifNetRunConfigFields>>, Vec<Hyperparameter<TrainMotifNetRunConfigFields>>, TrainMotifNetRunConfigFields> {
        OptRunConfigs::new(
            GridRunConfigs::new(
                ConstConfigs::new(vec![
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Epochs), cycles),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::BatchSize), batch_size),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Patience), Val::Int(10)),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::MinDelta), Val::Real(1e-4)),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), Val::Real(0.00001)),
                    NamedVal::new(TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::LinNet(MotifLinNetArgsFields::NumLayers)), Val::Int(8)),
                    NamedVal::new(TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::LinNet(MotifLinNetArgsFields::Step)), Val::Int(1)),
                    NamedVal::new(TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::LinNet(MotifLinNetArgsFields::Kernel)), kernel),
                ]),
                vec![],
            ),
            vec![
                // Hyperparameter::UniformFloat { name: TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), lower: 0.00001, upper: 0.0001 },
                Hyperparameter::UniformInt { name: TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::LinNet(MotifLinNetArgsFields::Channels)), lower: 8, upper: 10 },
            ],
        )
    }

    fn _test_conv_net_run_configs(kernel: Val, cycles: Val, batch_size: Val) -> OptRunConfigs<Vec<NamedVal<TrainMotifNetRunConfigFields>>, Vec<NamedRange<TrainMotifNetRunConfigFields>>, Vec<Hyperparameter<TrainMotifNetRunConfigFields>>, TrainMotifNetRunConfigFields> {
        OptRunConfigs::new(
            GridRunConfigs::new(
                ConstConfigs::new(vec![
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Epochs), cycles),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::BatchSize), batch_size),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Patience), Val::Int(10)),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::MinDelta), Val::Real(1e-4)),
                    NamedVal::new(TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), Val::Real(0.00001)),
                    // NamedVal::new(TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::ConvNet(MotifConvNetArgsFields::HeadLayers)), Val::Int(4)),
                    NamedVal::new(TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::ConvNet(MotifConvNetArgsFields::ConvLayers)), Val::Int(4)),
                    NamedVal::new(TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::ConvNet(MotifConvNetArgsFields::Step)), Val::Int(1)),
                    NamedVal::new(TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::ConvNet(MotifConvNetArgsFields::Kernel)), kernel),
                ]),
                vec![],
            ),
            vec![
                // Hyperparameter::UniformFloat { name: TrainMotifNetRunConfigFields::CommonTrainingArgs(CommonTrainingRunConfigFields::Lr), lower: 0.00001, upper: 0.0001 },
                Hyperparameter::UniformInt { name: TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::ConvNet(MotifConvNetArgsFields::ConvLayers)), lower: 2, upper: 24 },
                Hyperparameter::UniformInt { name: TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::ConvNet(MotifConvNetArgsFields::HeadLayers)), lower: 2, upper: 24 },
                Hyperparameter::UniformInt { name: TrainMotifNetRunConfigFields::Net(MotifNetTypeArgsFields::ConvNet(MotifConvNetArgsFields::Channels)), lower: 8, upper: 256 },
            ],
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn heavy_test_train_motif_nets_cosine_with_filtered_real_eeg() -> Result<(), ThoError> {
        if !tch::Cuda::is_available() {
            return Ok(());
        }
        GlobalCtx::init_devices(2);

        let find_motifs_run_id: RunId = 1;
        let select_motifs_run_id: RunId = 2;
        let train_motif_nets_run_id: RunId = 3;
        let motif_id = 10u32;
        let motif_width = 200u32;
        let max_slices = 500u32;
        let filter_stride = 5;
        let target_eeg_file = "1014985354.pt";
        let target_resource_id = target_eeg_file
            .strip_suffix(".pt")
            .ok_or_else(|| ThoError::new(format!("invalid target file name: {target_eeg_file}")))?
            .parse::<u32>()
            .map_err(|e| ThoError::new(format!("invalid target resource id in {target_eeg_file}: {e}")))?;

        let freq_filters = vec![
            InputFilter::SavGolFilter(SavGolFilterParams {
                window_len: 111,
                poly_order: 3,
                stride: filter_stride,
                padding: Padding::zero(),
            }),
            InputFilter::NotchBandFilter(NotchBandFilterParams {
                low_hz: 62.0,
                high_hz: 59.0,
            }),
        ];

        let load_eeg_tensor = |path: &Path| -> Result<Tensor, ThoError> {
            let mut raw_loaded = Tensor::load(path)?;
            if matches!(raw_loaded.size().as_slice(), [1, _, _]) {
                raw_loaded = raw_loaded.f_squeeze_dim(0)?;
            }
            let raw_loaded = match raw_loaded.size().as_slice() {
                [c, t] if c <= t => raw_loaded,
                [t, c] if t > c => raw_loaded.f_transpose(0, 1)?,
                dims => return Err(ThoError::new(format!("unexpected eeg shape in {}: {dims:?}", path.display()))),
            };
            raw_loaded.f_to_kind(Float).map_err(Into::into)
        };
        let raw_loaded = load_eeg_tensor(Path::new("test_data/eeg").join(target_eeg_file).as_path())?.f_to_device(Cuda(0))?;
        debug!("test_train_motif_nets_cosine_with_filtered_real_eeg: raw_loaded.size={:?}", raw_loaded.size());

        let raw_loaded = Tensor::cat(&[raw_loaded.copy(), raw_loaded], 1);//concatenating to have same slices
        let raw_loaded_for_apply = raw_loaded.shallow_clone();

        let channels = raw_loaded.size()[0] as u32;
        let ws = Arc::new(RwLock::new(FsWorkspace::new(Path::new("/tmp").join(cdt_for_dir()), channels, raw_loaded.size()[1] as u32, 200)?));

        let raw_tensor = ThoTensor::<ChannelsData2, SynKey>::new(
            SynKey::new([mono_key!("test_train_motif_nets_cosine_with_filtered_real_eeg", "raw")]),
            raw_loaded,
        )?;

        /*
        it's applied  when being loaded
        let filtered: Arc<DevicesWrap<ThoTensor<ChannelsData2, SynKey>>> = Arc::new(FreqFilter::<MockRunCtx>::new(freq_filters.as_slice(), &ws, find_motifs_run_id)?)
            .run(Arc::new(DevicesWrap::wrap(raw_tensor, CacheArea::default())), mock_run_ctx::default(1))
            .await?;

        let filtered_cpu = {
            let filtered_guard = filtered.load_from_any_device_(1).await?;
            filtered_guard.t_ref().f_to_device(Device::Cpu)?
        };*/

        let input_len = raw_tensor.size()[1];
        if input_len < motif_width as i64 {
            return Err(ThoError::new(format!("filtered tensor is too short: filtered_len={input_len} motif_width={motif_width}")));
        }
        let find_motifs_config = FindMotifsRunConfig::new_with_running_mean(
            Some(find_motifs_run_id),
            1,
            motif_width,
            Some(freq_filters.clone()),
            None,
        );
        let input_len_adjuster = find_motifs_config.input_len_adjuster();
        let filtered_len = input_len_adjuster(input_len as u32);

        let motif_slices = vec![
            TensorIdSliceIdx(target_resource_id, 100),
            TensorIdSliceIdx(target_resource_id, filtered_len / 2 + 100), //same slice from concatenated part
        ];
        let motif_slices_expected = motif_slices.clone();

        let target_resource = Resource::new(target_resource_id, 1, target_eeg_file.to_string(), input_len as u32);
        let motif_resources_by_id = HashMap::from([(target_resource_id, target_resource.clone())]);
        let mut loader_tensors_map = HashMap::from([(target_resource.clone(), raw_tensor)]);

        let random_resources = random_resources(target_eeg_file, load_eeg_tensor, channels, &mut loader_tensors_map)?;
        let (random_slices, random_resources_by_id) = random_non_overlapping_slices(random_resources.as_slice(), max_slices, motif_width, input_len_adjuster)?;
        assert!(random_slices.iter().all(|slice| slice.0 != target_resource_id));

        debug!("test_train_motif_nets_cosine_with_filtered_real_eeg: filtered_len={input_len} motif_slices={motif_slices:?} random_slices={random_slices:?}");
        let loader_tensors = Arc::new(Mutex::new(loader_tensors_map));

        let mut selected_motifs = vec![Vec::<(u32, f64)>::new(); channels as usize];
        selected_motifs[0].push((motif_id, 1.0));
        let trained_net_path = PathBuf::from(format!("0/{}_{}", motif_id, train_motif_nets_run_id));

        let find_motifs_props = find_motifs_config.to_properties_iter(None).collect();

        let store = Arc::new(
            MockStoreBuilder::new()
                .with_list_run_config_values(find_motifs_run_id, Ok(find_motifs_props))
                .with_get_motif_meta([(motif_id, Ok((0usize, find_motifs_run_id)))])
                .with_list_selected_motifs([(select_motifs_run_id, Ok(selected_motifs))])
                .with_list_motif_slices(
                    motif_id,
                    Some(vec![select_motifs_run_id]),
                    None,
                    None,
                    Ok((0i64, find_motifs_run_id, motif_slices, motif_resources_by_id.clone())),
                )
                .with_list_random_slices_out_of_motif(
                    motif_id,
                    motif_width,
                    None,
                    Ok((0i64, find_motifs_run_id, random_slices, random_resources_by_id)),
                )
                .with_list_trained_nets([
                    (vec![train_motif_nets_run_id], None, None, Ok(vec![(0, trained_net_path.clone(), 0.01)])),
                ])
                .build(),
        );

        let loader = Arc::new(RawTensorLoader::<MockRunCtx>::new(
            move |input_resource: &Resource, _device| {
                let t = thoosa_shared::perf_log!(thoosa_shared::sync_mutex_lock!(loader_tensors))
                    .get(input_resource)
                    .map(|v| v.as_ref().shallow_clone())
                    .ok_or_else(|| ThoError::new(format!("missing resource tensor for resource_id={}", input_resource.id())))?;

                debug!("input_resource={input_resource:?} t.device={:?}", t.device());
                ThoTensor::new(
                    SynKey::new([mono_key!(input_resource.id(), format!("filtered_resource_{}", input_resource.id()))]),
                    t,
                )
            },
            ThoDevice::from(0usize),
        ));

        //conv_layers: 4, head_layers: 3, stride: 1, channels: 147,
        //conv_layers: 11, head_layers: 3, stride: 1, channels: 218,
        //conv_layers: 21, head_layers: 2, stride: 1, channels: 215
        // let train_motif_nets = Arc::new(TrainMotifNets::new(loader, &ws, &store, _test_cosine_net_run_configs(Val::Int(10), Val::Int(300), Val::Int(10))));
        let train_motif_nets = Arc::new(TrainMotifNets::new(loader, &ws, &store, _test_conv_net_run_configs(Val::Int(10), Val::Int(300), Val::Int(100)), 10));
        // let train_motif_nets = Arc::new(TrainMotifNets::new(loader, &ws, &store, _test_lin_net_run_configs(Val::Int(100), Val::Int(300), Val::Int(10))));

        let input = Arc::new(DevicesWrap::wrap(TrainMotifNetInput::new(thoosa_common::train_nets::MotifsBy::SelectMotifsRunIds { select_motifs_run_ids: vec![select_motifs_run_id] }, Arc::new(OptConfig::new(100, 10)))));

        let mut run_ctx = mock_run_ctx::default(1);
        let result: Arc<DevicesWrap<thoosa_common::train_nets::TrainMotifsNetsResult>> = train_motif_nets
            .run(input, run_ctx.branch(Some("test_train_motif_nets_cosine_with_filtered_real_eeg"), false))
            .await?;
        let result = result.load_from_any_device_(1).await?;


        debug!("train_motif_nets: result={result:?}");
        assert!(result.train_motif_nets_results[0].loss < 0.2);

        assert!(!result.train_motif_nets_results.is_empty());

        let train_result = result.train_motif_nets_results.first().expect("empty train_motif_nets_results");
        let saved_path = ws.read().save_motif_net(&0u32, &train_motif_nets_run_id, &motif_id, &train_result.vs, &train_result.params)?;
        assert_eq!(saved_path, trained_net_path);

        let apply_motif_nets = Arc::new(ApplyMotifNets::new(
            &ws,
            &store,
            vec![train_motif_nets_run_id],
            mock_run_ctx::default(1),
            None,
            None,
            1,
            None
        ).await?);

        let out = Arc::clone(&apply_motif_nets).run(
            Arc::new(DevicesWrap::wrap(ThoTensor::new(
                SynKey::new([mono_key!("raw_loaded_0", "raw_loaded_0")]),
                raw_loaded_for_apply.shallow_clone(),
            )?)),
            mock_run_ctx::default(1),
        ).await?;

        let out = apply_motif_nets.run(
            Arc::new(DevicesWrap::wrap(ThoTensor::new(
                SynKey::new([mono_key!("raw_loaded_1", "raw_loaded_1")]),
                raw_loaded_for_apply,
            )?)),
            mock_run_ctx::default(1),
        ).await?;
        let out = out.load_from_any_device_(1).await?;
        let (top_vals, top_idx) = out.data().t_ref().f_i((0, ..))?.f_topk(2, 0, true, true)?;

        debug!("test_train_motif_nets_cosine_with_filtered_real_eeg: apply top_vals={} top_idx={}", top_vals, top_idx);

        let predicted_a = top_idx.double_value(&[0]);
        let predicted_b = top_idx.double_value(&[1]);
        let expected_a = motif_slices_expected[0].1 as f64;
        let expected_b = motif_slices_expected[1].1 as f64;
        let max_diff = 20.0;

        let d_direct = (predicted_a - expected_a).abs() + (predicted_b - expected_b).abs();
        let d_swapped = (predicted_a - expected_b).abs() + (predicted_b - expected_a).abs();
        if d_direct <= d_swapped {
            assert_close(expected_a, predicted_a, max_diff);
            assert_close(expected_b, predicted_b, max_diff);
        } else {
            assert_close(expected_a, predicted_b, max_diff);
            assert_close(expected_b, predicted_a, max_diff);
        }

        Ok(())
    }

    fn random_resources(target_eeg_file: &str, load_eeg_tensor: fn(&Path) -> Result<Tensor, ThoError>, channels: u32, loader_tensors_map: &mut HashMap<Resource, ThoTensor<ChannelsData2, SynKey>>) -> Result<Vec<Resource>, ThoError> {
        let eeg_dir = Path::new("test_data/eeg");
        let mut random_resources = Vec::new();
        let mut eeg_files = std::fs::read_dir(eeg_dir)
            .map_err(|e| ThoError::new(format!("failed to read {}: {e}", eeg_dir.display())))?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ThoError::new(format!("failed to read entry in {}: {e}", eeg_dir.display())))?;
        eeg_files.sort();

        for path in eeg_files {
            if path.extension().and_then(|ext| ext.to_str()) != Some("pt") {
                continue;
            }
            let file_name = path
                .file_name()
                .and_then(|v| v.to_str())
                .ok_or_else(|| ThoError::new(format!("invalid file name: {}", path.display())))?
                .to_string();
            if file_name == target_eeg_file {
                continue;
            }

            let resource_id = path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .ok_or_else(|| ThoError::new(format!("invalid file stem for {}", path.display())))?
                .parse::<u32>()
                .map_err(|e| ThoError::new(format!("invalid resource id in {}: {e}", path.display())))?;

            let tensor = load_eeg_tensor(path.as_path())?;
            if tensor.size()[0] as u32 != channels {
                debug!("skipping {file_name}: expected channels={channels}, got={:?}", tensor.size());
                continue;
            }
            let resource = Resource::new(resource_id, 1, file_name, tensor.size()[1] as u32);
            random_resources.push(resource.clone());
            loader_tensors_map.insert(
                resource.clone(),
                ThoTensor::<ChannelsData2, SynKey>::new(
                    SynKey::new([mono_key!(resource.id(), format!("filtered_resource_{}", resource.id()))]),
                    tensor,
                )?,
            );
        }
        Ok(random_resources)
    }
}
