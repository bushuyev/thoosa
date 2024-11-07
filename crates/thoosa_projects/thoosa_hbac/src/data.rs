use std::path::Path;
use std::collections::HashMap;
use std::fs;
use std::fs::DirEntry;
use polars::prelude::{col, concat_list, lit, Expr, IntoLazy, JoinArgs, JoinType, LazyCsvReader, LazyFileListReader, LazyFrame, NamedFrom, ScanArgsParquet, Series};
use log::debug;
use polars::datatypes::DataType;
use polars::frame::DataFrame;
use tch::{Device, Kind, Tensor};
use rand::{thread_rng, Rng};
use std::io::Write;

use rand::prelude::SliceRandom;
use regex::Regex;

use thoosa_common::utils::DataBatch;
use thoosa_main::data::{parquet_to_tensor_by_path, AllTrainSplitter, Set, Splitter};
use thoosa_shared::ThoosaError;


#[allow(dead_code)]
pub struct ByPatientIdSplitter {
    train_frac:f32,
    train_patients:Vec<i64>,
    test_patients:Vec<i64>
}

impl ByPatientIdSplitter {

    #[allow(dead_code)]
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

    fn init(&self, df: LazyFrame) -> Result<LazyFrame, ThoosaError>{
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

        let df_set = df_sorted.with_column(Series::new("set", (0..df_sorted.shape().0).map(|_i| if rand::random::<f32>() > self.train_frac { 1i32 } else { 0i32 }).collect::<Vec<_>>()))?;

        Ok(df.clone().join(df_set.clone().lazy(), [col("patient_id")], [col("patient_id")], JoinArgs::new(JoinType::Left)))
        // Ok(df.clone().with_column(lit(1).alias("set"))) //.collect()?.lazy()
    }
}

pub fn parquet_to_tensor(base_path: &str, resource_id: &str, device: Device) -> Result<Tensor, ThoosaError> {
    let path = format!("{}/train_eegs/{}.parquet", base_path, resource_id);

    parquet_to_tensor_by_path(path, device)
}


pub fn labeled_resourses(path: impl AsRef<Path>) -> Result<HashMap<String, Vec<(String, f32)>>, ThoosaError>{
    let df = LazyCsvReader::new(path)
        .finish()?
        .with_streaming(true);

    let gr_by_eeg_id_df = EEGs::group_by_eeg_id(None, df)?;

    Ok(gr_by_eeg_id_df
        .column("eeg_id")?.i64()?.into_iter()
        // .zip(gr_by_eeg_id_df.column("set")?.i32()?)
        // .zip(gr_by_eeg_id_df.column("eeg_label_offset_seconds")?.list()?)
        .zip(gr_by_eeg_id_df.column("seizure_vote")?.i64()?)
        .zip(gr_by_eeg_id_df.column("lpd_vote")?.i64()?)
        .zip(gr_by_eeg_id_df.column("gpd_vote")?.i64()?)
        .zip(gr_by_eeg_id_df.column("lrda_vote")?.i64()?)
        .zip(gr_by_eeg_id_df.column("grda_vote")?.i64()?)
        .zip(gr_by_eeg_id_df.column("other_vote")?.i64()?)
        .map(|
            ((((((eeg_id,
                // eeg_label_offset_seconds),
                seizure_vote),
                lpd_vote),
                gpd_vote),
                lrda_vote),
                grda_vote),
                other_vote)
        | {
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
        }).map(|(eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote)|{
        let sum = (seizure_vote + lpd_vote + gpd_vote + lrda_vote + grda_vote + other_vote) as f32;

        (
            eeg_id.to_string(),
            [
                ("seizure".to_string(), seizure_vote as f32 / sum),
                ("lpd".to_string(), lpd_vote as f32 / sum),
                ("gpd".to_string(), gpd_vote as f32 / sum),
                ("lrda".to_string(), lrda_vote as f32 / sum),
                ("grda".to_string(), grda_vote as f32 / sum),
                ("other".to_string(), other_vote as f32 / sum)

            ].into_iter().filter(|v| v.1 > 0.).collect()
        )

    }).fold(HashMap::new(), |mut h, v| {
        h.insert(v.0, v.1);
        h
    })
    )

}

#[allow(dead_code)]
pub fn read_eeg_index(motif_dir:&str)->Result<(Option<String>, Vec<String>), ThoosaError> {
    // motif_dir
    let files:Vec<DirEntry> = fs::read_dir(motif_dir)?.into_iter().map(|p|p.unwrap()).filter(|p|p.file_name().to_str().unwrap().ends_with("txt")).collect();
    assert_eq!(files.len(), 1);


    debug!("files={:?}", files);
    let cnt = fs::read_to_string(files[0].path())?;
    debug!("{:?}", cnt);

    let re: Regex = Regex::new(r#""(?<id>\w+)""#)?;

    Ok(
        (
            files[0].path().file_name().and_then(|f|f.to_str()).map(|f|f.to_string()),
            re.captures_iter(cnt.as_str()).map(|caps| caps["id"].into()).collect::<Vec<String>>()
        )
    )

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

    for (eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote) in gr_by_eeg_id_df
        .column("eeg_id")?.i64()?.into_iter()
        // .zip(gr_by_eeg_id_df.column("set")?.i32()?)
        // .zip(gr_by_eeg_id_df.column("eeg_label_offset_seconds")?.list()?)
        .zip(gr_by_eeg_id_df.column("seizure_vote")?.i64()?)
        .zip(gr_by_eeg_id_df.column("lpd_vote")?.i64()?)
        .zip(gr_by_eeg_id_df.column("gpd_vote")?.i64()?)
        .zip(gr_by_eeg_id_df.column("lrda_vote")?.i64()?)
        .zip(gr_by_eeg_id_df.column("grda_vote")?.i64()?)
        .zip(gr_by_eeg_id_df.column("other_vote")?.i64()?)
        .map(|
            ((((((eeg_id,
                // eeg_label_offset_seconds),
                seizure_vote),
                lpd_vote),
                gpd_vote),
                lrda_vote),
                grda_vote),
                other_vote)
        | {
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
        })
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

        let path = format!("{}/{}_{}_{}_{}_{}_{}_{}.pt", "/media/bu/KINGSTON/all_eegs.pt",
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

        count+=1;
    }

    debug!("count={}", count);


    Ok(())
}


#[allow(dead_code)]
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
    batch_size: usize
}

#[allow(dead_code)]
impl EEGs {

    pub fn good_eegs(base_path: &str)  -> Result<Series, ThoosaError>{
        let good_eegs_filter = Self::good_eegs_filter(base_path)?;

        Ok(good_eegs_filter
            .collect()?
            .column("eeg_id")?.clone())

    }

    pub fn good_eegs_of(base_path: &str, kind:&str, limit:Option<usize>)  -> Result<Series, ThoosaError>{
        let good_eegs_filter = Self::good_eegs_filter(base_path)?;

        let eeg_id_series =
            good_eegs_filter
            .filter(col(kind).gt(lit(0)))
                // .sort(kind, SortOptions{ descending: true, ..Default::default()})
                // .sort("eeg_id", Default::default())
                .sort_by_exprs(vec![col(kind), col("eeg_id")], vec![true, false], true, true)
            .collect()?
            .column("eeg_id")?.clone();


        match limit {
            None => {
                Ok(eeg_id_series)
            }
            Some(v) => {
                Ok(eeg_id_series.limit(v))
            }
        }
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
            .with_column(concat_list(["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"])?.alias("all_votes").alias("all_votes"))
            .filter(
                col("all_votes").list().max().alias("max").eq(col("all_votes").list().sum().alias("sum"))
            ))
    }

    pub fn load_all_eegs(base_path: &str) -> Result<Box<EEGs>, ThoosaError> {
        EEGs::load_eegs::<_>(base_path, None, None::<AllTrainSplitter<EEG>>)
    }
    pub fn load_eegs<S>(base_path: &str, filter:Option<Expr>, splitter:Option<S>) -> Result<Box<EEGs>, ThoosaError> where S:Splitter<Type=EEG>{
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


        let gr_by_eeg_id_df = Self::group_by_eeg_id(filter, df)?;

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
            valid_set: valid_eegs,
            batch_size: 10//TODO param
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
                col("eeg_label_offset_seconds").sort(false).alias("eeg_label_offset_seconds"),
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
    fn train_batch(&mut self) -> (Tensor, Tensor) {
        Self::random_subset(self.batch_size, &self.train_set)
    }

    fn valid_batch(&mut self) -> (Tensor, Tensor) {
        Self::random_subset(self.batch_size, &self.valid_set)
    }
}

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::UNIX_EPOCH;
    use log::debug;
    use parking_lot::{Mutex, RwLock};
    use plotters::prelude::{BLACK, RED};
    use polars::prelude::{col, concat_list, count, lit, when, LazyCsvReader, LazyFileListReader, NamedFrom, Series};
    use tch::{Device, Tensor};
    use tch::Device::Cpu;
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::ResourceLoader;
    use thoosa_common::train_nets::{TrainFinalParams, TrainLabelNodeParams};
    use thoosa_common::utils::{cdt_for_dir, from, plot_series};
    use thoosa_common::ws::{DummyWorkspace, FsWorkspace};
    use thoosa_main::cli::Commands::{FindMotifs, SelectMotifs, TrainFinal, TrainOnLabel};
    use thoosa_main::data::ParquetLoader;
    use thoosa_main::cli::{init_resources, run_select_motifs, run_step, run_train_final, run_train_label_nodes, Cli, RunAllParams};
    use thoosa_motifs::find::{find_motifs_by_euc_dist, FindMotifsParams, SelectMotifsParams};
    use super::{parquet_to_tensor, ByPatientIdSplitter, EEG};
    use thoosa_common::ws::Workspace;
    use thoosa_shared::{init_test_logging, ThoosaError};
    use thoosa_store::SqlStore;
    use crate::data::EEGs;

    #[ctor::ctor]
    fn init() {
        init_test_logging();
    }


    #[allow(dead_code)]
    pub fn plot_eeg_tensor() -> Result<(), ThoosaError> {
        // debug!("tensor={:?}", tensor.size());

        let t_0 = parquet_to_tensor("../../data", "1628180742", Device::Cpu)?.f_slice(0, Some(0), Some(1), 1)?;
        let t_1 = parquet_to_tensor("../../data", "1019408895", Device::Cpu)?.f_slice(0, Some(0), Some(1), 1)?;

        // debug!("tensor={:?}", tensor.size());

        //"1628180742".into(), "1019408895"
        plot_series(
            vec![
                (format!("1628180742"), BLACK, from::<Vec<f32>>(&t_0.reshape(-1)).iter().enumerate().map(|(i, v)|(i as f32, *v)).collect::<Vec<(f32, f32)>>()),
                (format!("1019408895"), RED, from::<Vec<f32>>(&t_1.reshape(-1)).iter().enumerate().map(|(i, v)|(i as f32, *v)).collect::<Vec<(f32, f32)>>()),

            ], "EEG", (10_000, 480), &"/tmp/plots/1628180742_1019408895.png"
        ).unwrap();

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

        debug!("df={:?}", df);

        //TODO hash + assert
        df.column("win_vote")?.str()?.into_iter().zip(df.column("votes_count")?.u32()?.into_iter()).for_each(|v|{
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

            ], "EEG", (10_000, 480), &"0.png"
        ).unwrap();

        Ok(())
    }


    #[test]
    fn test_read_eeg_index() -> Result<(), ThoosaError>  {
        let dir = format!("/tmp/test_{}", std::time::SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs());
        std::fs::create_dir_all(dir.clone())?;

        let filename = format!("{}/test_eegs.txt", dir);
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(false)
            .open(filename.clone())
            .unwrap();

        let vec_1:Vec<String> = vec!["1123".into(), "34534".into(), "99877".into()];
        write!(file, "{:?}", vec_1)?;

        assert_eq!(vec_1, super::read_eeg_index(dir.as_str())?.1);
        Ok(())

    }

    // #[test] TODO to util
    #[allow(dead_code)]
    fn index_eegs_to_pt() -> Result<(), ThoosaError> {

        for eeg_file in super::read_eeg_index("../../data_down/14_02_results/2024-02-14T17:36:36Z/")?.1{
            debug!("{:?}", eeg_file);

            let t = super::parquet_to_tensor("../../data/", eeg_file.as_str(), Device::Cpu)?;

            t.save(format!("../../data_local/seizure_vote_eegs_pts/{}.pt", eeg_file))?;
        }

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
    fn test_run_find_motifs() -> Result<(), ThoosaError>{
        let path = std::env::current_dir()?;
        println!("The current directory is {}", path.display());
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        // let ws = Arc::new(RwLock::new(FsWorkspace::new(&test_dir, 20, 684400, Some(test_dir.join("tensors")))?));
        let ws = Arc::new(RwLock::new(DummyWorkspace::new(3)?));

        let cli = Cli{
            base_path: "test_data/eeg".to_string(),
            resource_pattern: "{}/train_eegs/{}.parquet".to_string(),
            ws_path: "/tmp".to_string(),
            tensors_path: None,
            store_path: "".to_string(),
            term_logger_level: "".to_string(),
            write_logger_level: "".to_string(),
            logfile: "".to_string(),
            limit: None,
            num_threads: None,
            command: Some(FindMotifs(FindMotifsParams::dummy(10, 10..10, "xxx".into(), 10, 0.3, None))),
            input_filters_kernel: None,
        };


        // fs::remove_file("/tmp/test_main_db.db3").unwrap_or(());
        // let store = Arc::new(Mutex::new(SqlStore::from_path("/tmp/test_main_db.db3")?));
        let store = Arc::new(Mutex::new(SqlStore::in_mem()?));


        let loader = Arc::new(Box::new(ParquetLoader::new(
            "test_data/eeg".into(), Some("test_data/eeg".into()), None,
            Box::new(|base_path,resource_id|format!("{}/train_eegs/{}.parquet", base_path, resource_id))
        )));
        let executor_rc = Arc::new(RwLock::new(LoadRunner::new(
            move |key: &Arc<String>| {

                SyncWrapper::new_tb({
                    debug!("test_run_find_motifs: key={key}");
                    loader.load(key, Device::Cpu)
                })
            }, 4, 100,
        )));

        init_resources(
            Arc::clone(&store),
            vec![
                "seizure",
                "lpd",
                "gpd",
                "lrda",
                "grda",
                "other"
            ],
            ||{
                super::labeled_resourses(Path::new(&cli.base_path).join("train_10.csv"))
            }
        )?;

        let motifs_run_ids = run_step(
            None,
            &RunAllParams::dummy("group 1 labels/all", "group 1 labels/all"),
            find_motifs_by_euc_dist,
            &ws,
            &store,
            &executor_rc,
            vec![Cpu]
        )?;


        debug!("motifs_run_ids={motifs_run_ids:?}");
        assert_eq!(vec![1], motifs_run_ids);

        Ok(())
    }


}
