use log::debug;
use polars::datatypes::DataType;
use polars::frame::DataFrame;
use polars::prelude::{Expr, IntoLazy, JoinArgs, JoinType, LazyCsvReader, LazyFileListReader, LazyFrame, NamedFrom, ScanArgsParquet, Series, SortMultipleOptions, col, concat_list, lit, BooleanChunked};
use rand::Rng;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use tch::{Device, Kind, Tensor};

use thoosa_common::dims::{Batch, ChannelsData};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::utils::DataBatch;
use thoosa_main::data::{AllTrainSplitter, Set, Splitter, parquet_to_tensor_by_path};
use thoosa_shared::ThoosaError;

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

    fn init(&self, df: LazyFrame) -> Result<LazyFrame, ThoosaError> {
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
) -> Result<Tensor, ThoosaError> {
    let path = format!("{}/train_eegs/{}.parquet", base_path, resource_id);

    parquet_to_tensor_by_path(path, device)
}

pub fn labeled_resourses(
    base: impl AsRef<Path>,
    file: impl AsRef<Path>,
) -> Result<HashMap<String, Vec<(String, f32)>>, ThoosaError> {
    let path = base.as_ref().join(file);
    let eeg_path = path.clone().parent().unwrap().to_path_buf();
    let df = LazyCsvReader::new(path).finish()?.with_streaming(true);


    let gr_by_eeg_id_df = EEGs::group_by_eeg_id(None, df)?;


    let mask: BooleanChunked = gr_by_eeg_id_df
        .column("eeg_id")?
        .i64()?
        .into_iter()
        .map(|opt| opt.map(|id|{

            let p = PathBuf::from(format!("{}/train_eegs/{id}.parquet", eeg_path.display()));
            let b = p.exists();
            debug!("id={id:?} p={p:?} b={b:?}");
            b

        }).unwrap_or(false))
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
pub fn eegs_to_tensors() -> Result<(), ThoosaError> {
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
    valid_set: Vec<EEG>,
    batch_size: usize,
}

#[allow(dead_code)]
impl EEGs {
    pub fn good_eegs(base_path: &str) -> Result<Series, ThoosaError> {
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
    ) -> Result<Series, ThoosaError> {
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

    fn good_eegs_filter(base_path: &str) -> Result<LazyFrame, ThoosaError> {
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

    pub fn load_all_eegs(base_path: &str) -> Result<Box<EEGs>, ThoosaError> {
        EEGs::load_eegs::<_>(base_path, None, None::<AllTrainSplitter<EEG>>)
    }
    pub fn load_eegs<S>(
        base_path: &str,
        filter: Option<Expr>,
        splitter: Option<S>,
    ) -> Result<Box<EEGs>, ThoosaError>
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

            print!(".");
            std::io::stdout().flush()?;
        }

        debug!(
            "tensors.len={}  valid_eegs.len={}",
            train_eegs.len(),
            valid_eegs.len()
        );
        std::io::stdout().flush()?;

        Ok(Box::new(EEGs {
            train_set: train_eegs,
            valid_set: valid_eegs,
            batch_size: 10, //TODO param
        }))
    }

    pub fn group_by_eeg_id(filter: Option<Expr>, df: LazyFrame) -> Result<DataFrame, ThoosaError> {
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
        ThoTensor<Batch<ChannelsData>, thoosa_shared::applicables_runner::CachableId>,
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

impl DataBatch<ChannelsData> for EEGs {
    fn train_batch(
        &self,
    ) -> (
        ThoTensor<Batch<ChannelsData>, thoosa_shared::applicables_runner::CachableId>,
        Tensor,
    ) {
        Self::random_subset(self.batch_size, &self.train_set)
    }

    fn valid_batch(
        &self,
    ) -> (
        ThoTensor<Batch<ChannelsData>, thoosa_shared::applicables_runner::CachableId>,
        Tensor,
    ) {
        Self::random_subset(self.batch_size, &self.valid_set)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, dead_code, unused_variables)]
    use super::{ByPatientIdSplitter, EEG, parquet_to_tensor};
    use crate::data::EEGs;
    use log::LevelFilter::Trace;
    use log::debug;
    use parking_lot::{Mutex, RwLock};
    use plotters::count;
    use plotters::prelude::{BLACK, RED};
    use polars::prelude::{
        LazyCsvReader, LazyFileListReader, NamedFrom, Series, col, concat_list, lit, when,
    };
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::UNIX_EPOCH;
    use tch::Device::Cpu;
    use tch::Kind::Float;
    use tch::{Device, IndexOp, TchError, Tensor};
    use thoosa_common::ResourceLoader;
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::utils::{
        from, hist_1_dim, lin_t, plot_series, skewness_kurtosis_1_dim, vec_f32_from,
    };
    use thoosa_common::ws::Workspace;
    use thoosa_common::ws::{FsWorkspace, mock_ws_simp};
    use thoosa_main::cli::Commands::{FindMotifs, SelectMotifs, TrainFinalNet, TrainLabelNets};
    use thoosa_main::cli::{Cli, import_resources};
    use thoosa_main::data::ParquetLoader;
    use thoosa_motifs::chunks::ChunkOfSlicedTensors;
    use thoosa_motifs::distance::MaskMaker::LeQuantile;
    use thoosa_motifs::distance::clc_dist_d2_and_mask;
    use thoosa_motifs::find::{SelectMotifsParams, find_motifs_by_euc_dist};
    use thoosa_shared::resource::Resource;
    use thoosa_shared::test_utils::init_test_logging;
    use thoosa_shared::{ThoosaError, cdt_for_dir};
    use thoosa_store::SqlStore;

    #[ctor::ctor]
    fn init() {
        init_test_logging();
    }

    // #[test]
    pub fn plot_eeg_tensor() -> Result<(), ThoosaError> {
        // log::set_max_level(Trace);
        // debug!("tensor={:?}", tensor.size());

        // let t_0 = parquet_to_tensor("../../data", "1628180742", Device::Cpu)?.i((0, ..));
        // let t_1 = parquet_to_tensor("../../data", "1019408895", Device::Cpu)?.i((0, ..));
        let t_0_0 = Tensor::load("/tmp/ws_import_test/tensors/1_1628180742.pt")?;
        let t_1_0 = Tensor::load("test_data/eeg/1513440817.pt")?;

        debug!("t_0.size={:?} t_1.size={:?}", t_0_0.size(), t_1_0.size());

        // debug!("tensor={:?}", tensor.size());

        //"1628180742".into(), "1019408895"
        plot_series(
            vec![
                (
                    format!("3780250851"),
                    BLACK,
                    from::<Vec<f32>>(&t_0_0.i((0, ..)).view(-1))
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (i as f32, *v))
                        .collect::<Vec<(f32, f32)>>(),
                ),
                (
                    format!("270230280"),
                    RED,
                    from::<Vec<f32>>(&t_1_0.i((0, ..)).view(-1))
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (i as f32, *v))
                        .collect::<Vec<(f32, f32)>>(),
                ),
            ],
            "EEG",
            (10_000, 480),
            &"/tmp/plots/ttt.png",
        )
        .unwrap();

        let window = 1000;
        let ws = mock_ws_simp(None, 1, 1000, 100);
        let t_0_0_0_sliced = ChunkOfSlicedTensors::slice_tensors_chunk(
            &FindMotifsRunConfig::new(None, 100, window, None, None).to_applicable(&ws, Cpu),
            &vec![t_0_0.into()], //sig_0 at idx 0
            vec![0],
            None,
        )
        .t;
        let t_0_1_0_sliced = ChunkOfSlicedTensors::slice_tensors_chunk(
            &FindMotifsRunConfig::new(None, 100, window, None, None).to_applicable(&ws, Cpu),
            &vec![t_1_0.into()], //sig_0 at idx 2000
            vec![0],
            None,
        )
        .t;

        let (le, Some(distances)) = clc_dist_d2_and_mask(
            t_0_0_0_sliced.shallow_clone(),
            t_0_1_0_sliced.shallow_clone(),
            true,
            &mut [LeQuantile {
                q: 0.1,
                with_mask: false,
            }],
        )?
        else {
            panic!("no distances")
        };
        let x = lin_t(distances.size()[1] as u32);
        plot_series(
            vec![
                (
                    format!("0ch"),
                    plotters::prelude::GREEN,
                    vec_f32_from(&x)
                        .into_iter()
                        .zip(from::<Vec<f32>>(&distances.i((0, 0, ..))))
                        .collect::<Vec<(f32, f32)>>(),
                ), //sig_0_v match
                (
                    format!("1ch"),
                    plotters::prelude::RED,
                    vec_f32_from(&x)
                        .into_iter()
                        .zip(from::<Vec<f32>>(&distances.i((1, 0, ..))))
                        .collect::<Vec<(f32, f32)>>(),
                ), //no matches
                (
                    format!("2ch"),
                    plotters::prelude::BLUE,
                    vec_f32_from(&x)
                        .into_iter()
                        .zip(from::<Vec<f32>>(&distances.i((2, 0, ..))))
                        .collect::<Vec<(f32, f32)>>(),
                ), //sig_2_v match
            ],
            "distance",
            (2000, 1000),
            &format!(
                "/tmp/plots/distance_{:?}.png",
                chrono::Utc::now().to_rfc3339()
            )
            .as_str(),
        )?;

        debug!("distances={distances:?}");

        let channels_dists_flat = distances.flatten(1, 2);

        // let channels_hists = channels_dists_flat.hist
        debug!("channels_dists_flat={:?}", channels_dists_flat.size());
        let hists = hist_1_dim(&channels_dists_flat, window);
        let x = lin_t(window);

        // let normal = Tensor::normal_(hists.i(0).mean(Float), hists.std(true))

        let (skewness, kurtosis) = skewness_kurtosis_1_dim(&hists.totype(Float));
        debug!("skewness={skewness}");
        debug!("kurtosis={kurtosis}");

        plot_series(
            vec![
                (
                    format!("0ch"),
                    plotters::prelude::GREEN,
                    vec_f32_from(&x)
                        .into_iter()
                        .zip(from::<Vec<f32>>(&hists.i(8)))
                        .collect::<Vec<(f32, f32)>>(),
                ), //sig_0_v match
                // (format!("1ch"), plotters::prelude::RED.into(),     vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&hists.i(1))).collect::<Vec<(f32, f32)>>()),//no matches
                (
                    format!("2ch"),
                    plotters::prelude::BLUE.into(),
                    vec_f32_from(&x)
                        .into_iter()
                        .zip(from::<Vec<f32>>(&hists.i(5)))
                        .collect::<Vec<(f32, f32)>>(),
                ), //sig_2_v match
            ],
            "hists",
            (2000, 1000),
            &format!("/tmp/plots/hists_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str(),
        )?;

        Ok(())
    }

    // #[test]
    pub fn test_fetch_similarities() -> Result<(), ThoosaError> {
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
    pub fn test_load_eegs() -> Result<(), ThoosaError> {
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
        debug!("test eegs={:?}", eegs.valid_set.len());

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

    #[test]
    fn test_import_resources() -> Result<(), ThoosaError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 3, 1000, 1000)));

        // fs::remove_file("/tmp/test_main_db.db3").unwrap_or(());
        // let store = Arc::new(RwLock::new(SqlStore::from_path("/tmp/test_main_db.db3")?));
        let store = Arc::new(RwLock::new(SqlStore::in_mem()?));

        let _resources = import_resources(
            &ws,
            &store,
            &["seizure", "lpd", "gpd", "lrda", "grda", "other"],
            || super::labeled_resourses(Path::new("test_data/eeg"), "train_10.csv"),
            // None::<Box<dyn Fn(String) -> Result<Tensor, ThoosaError> + Sync + Send>>,
            |key| {
                Tensor::load(format!("test_data/eeg/train_eegs/{}.parquet", key).as_str())
                    .map_err(|e| e.into())
            },
            true,
        )?;

        let labels = store.read().list_labels()?;

        assert_eq!(
            vec!["seizure", "lpd", "gpd", "lrda", "grda", "other"],
            labels
        );

        Ok(())
    }
}
