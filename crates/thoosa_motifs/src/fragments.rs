use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::fs;
use std::hash::Hash;
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::str::FromStr;

use crate::distance::{average_euclidean_distance, clc_dist_d2};
use clap::{Args, ValueEnum};
use crossbeam_utils::sync::WaitGroup;
use futures::FutureExt;
use futures_util::future::join_all;
use log::{debug, info, trace, warn};
use parking_lot::RwLock;
use regex::Regex;
use std::panic::AssertUnwindSafe;
use tch::Device;
use tch::Device::Cpu;
use tch::IndexOp;
use tch::Kind::Float;
use tch::Tensor;
use thoosa_common::applicables::{ThoRunCtx};
use thoosa_common::find_motifs_run_config::{
    FindMotifsRunConfig, FindMotifsRunConfigOptions,
};
use thoosa_common::loader::{LoadRunner, Loader, SyncWrapper};
use thoosa_shared::panic_to_err;
use thoosa_common::utils::*;
use thoosa_common::ws::Workspace;
use thoosa_shared::TensorIdSliceIdx;
use thoosa_shared::ThoosaError;
use thoosa_shared::resource::{Resource, ResourceId};
use thoosa_shared::run_configs::{RunId};
use thoosa_shared::*;

#[derive(Debug, Args)]
pub struct MotifsToFragmentsParams {
    #[arg(long, default_value = "../../data_local/motifs")]
    motifs_dir: String,

    #[arg(long, default_value = "../../data_local/fragments")]
    fragments_dir: String,

    #[arg(long, default_value_t = 1000)]
    window: i64,
}

impl MotifsToFragmentsParams {
    pub fn motifs_dir(&self) -> &str {
        &self.motifs_dir
    }

    pub fn fragments_dir(&self) -> &str {
        &self.fragments_dir
    }

    pub fn window(&self) -> i64 {
        self.window
    }
}

#[derive(Debug)]
pub struct MotifsTensors {
    pub channels_motifs_slices: Tensor,
    pub channels_motifs_avg_dists: Tensor,
    pub channels_motifs_mask: Tensor,
    pub channels_motifs_ids: Tensor,
}
//should be read-only
unsafe impl Sync for MotifsTensors {}

impl MotifsTensors {
    pub fn new(
        channels_motifs_slices: Tensor,
        channels_motifs_avg_dists: Tensor,
        channels_motifs_ids: Tensor,
    ) -> Self {
        Self {
            channels_motifs_slices,
            channels_motifs_mask: channels_motifs_avg_dists.ge(0.), //avg dists padded with f32::MIN
            channels_motifs_avg_dists,
            channels_motifs_ids,
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum MotifsBy {
    MotifId(usize), //TODO u32
    RunId {
        find_motifs_run_id: RunId,
        select_motifs_run_ids: Option<Vec<RunId>>,
    },
    GoodMotifs(RunId),
}

impl MotifsBy {
    fn parse_run_id_payload(payload: &str) -> Result<Self, String> {
        let (find_part, select_part) = if let Some((find, rest)) = payload.split_once('_') {
            (find, Some(rest))
        } else {
            (payload, None)
        };

        let find_motifs_run_id = find_part
            .parse::<RunId>()
            .map_err(|e| format!("can't parse {payload:?} to find_motifs_run_id: {e}"))?;

        let select_motifs_run_ids = match select_part {
            None => None,
            Some("") => Some(vec![]),
            Some(values) => {
                let ids = values
                    .split(',')
                    .filter(|segment| !segment.is_empty())
                    .map(|segment| {
                        segment
                            .parse::<RunId>()
                            .map_err(|e| format!("can't parse {segment:?} to select_motifs_run_id: {e}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Some(ids)
            }
        };

        Ok(MotifsBy::RunId {
            find_motifs_run_id,
            select_motifs_run_ids,
        })
    }
}

impl FromStr for MotifsBy {
    type Err = String;

    fn from_str(v: &str) -> Result<Self, Self::Err> {
        if let Some(rest) = v.strip_prefix("motif_id:") {
            Ok(MotifsBy::MotifId(
                rest.parse()
                    .map_err(|e| format!("can't parse {rest:?} to motif_id: {e}"))?,
            ))
        } else if let Some(rest) = v.strip_prefix("run_id:") {
            MotifsBy::parse_run_id_payload(rest)
        } else if let Some(rest) = v.strip_prefix("good_motifs_run_id:") {
            Ok(MotifsBy::GoodMotifs(
                rest.parse()
                    .map_err(|e| format!("can't parse {rest:?} to good motifs run_id: {e}"))?,
            ))
        } else {
            Err(format!(
                "unexpected parameter {v}, should be motif_id:nnn or run_id:nnn"
            ))
        }
    }
}

impl From<&str> for MotifsBy {
    fn from(v: &str) -> MotifsBy {
        MotifsBy::from_str(v).unwrap_or_else(|err| panic!("{err}"))
    }
}

#[derive(Debug, Args, Clone)]
pub struct PlotMotifSlicesArgs {
    #[arg(long, value_enum)]
    motifs_by: MotifsBy,

    #[arg(long, value_enum)]
    pub limit: Option<u32>,
}

impl PlotMotifSlicesArgs {
    pub fn motifs_by(&self) -> MotifsBy {
        self.motifs_by.clone()
    }

    pub fn new(motifs_by: MotifsBy) -> Self {
        Self {
            motifs_by,
            limit: None,
        }
    }
}

#[derive(Debug, Args, Clone)]
pub struct MotifsToTensorArgs {
    #[arg(long)]
    run_id: RunId,

    #[arg(long, default_value_t = 1000)]
    window: i64,
}

impl MotifsToTensorArgs {
    pub fn window(&self) -> i64 {
        self.window
    }
    pub fn run_id(&self) -> RunId {
        self.run_id
    }
}

#[cfg(test)]
mod motifs_tests {
    use super::MotifsBy;
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::ops::Index;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::UNIX_EPOCH;

    use tch::{Device, IndexOp, Tensor};

    use log::debug;
    use parking_lot::RwLock;
    use tch::Device::Cpu;
    use tch::Kind::Float;
    use tch::kind::FLOAT_CPU;
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::utils::*;
    use thoosa_common::ws::{Workspace, mock_ws_simp};
    use thoosa_shared::MotifSlice;
    use thoosa_shared::TensorIdSliceIdx;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::{Resource, ResourceId};
    use thoosa_shared::{ThoosaError, cdt_for_dir};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn parses_motifs_by_run_id_filters() {
        use std::str::FromStr;

        let simple = MotifsBy::from_str("run_id:11").unwrap();
        assert_eq!(
            simple,
            MotifsBy::RunId {
                find_motifs_run_id: 11,
                select_motifs_run_ids: None
            }
        );

        let empty = MotifsBy::from_str("run_id:42_").unwrap();
        assert_eq!(
            empty,
            MotifsBy::RunId {
                find_motifs_run_id: 42,
                select_motifs_run_ids: Some(vec![])
            }
        );

        let filtered = MotifsBy::from_str("run_id:7_10,20,30").unwrap();
        assert_eq!(
            filtered,
            MotifsBy::RunId {
                find_motifs_run_id: 7,
                select_motifs_run_ids: Some(vec![10, 20, 30])
            }
        );

        let good = MotifsBy::from_str("good_motifs_run_id:99").unwrap();
        assert_eq!(good, MotifsBy::GoodMotifs(99));
    }

    #[test]
    fn test_0() {
        let t = Tensor::from_slice(&[2., 4., 1., 1., 3., 2., 3., 2., 3.])
            .view((3, 3))
            .mean_dim(0, true, Float);

        println!("t={t}");
    }
}
