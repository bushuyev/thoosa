use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::fs;
use std::hash::Hash;
use std::ops::Deref;
use std::path::Path;
use std::path::PathBuf;
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
use thoosa_common::find_motifs_run_config::{
    FindMotifsRunConfig,
    InputFilter,
};
use thoosa_common::loader::{LoadRunner, Loader, SyncWrapper};
use thoosa_shared::panic_to_err;
use thoosa_common::utils::*;
use thoosa_common::ws::Workspace;
use thoosa_shared::TensorIdSliceIdx;
use thoosa_shared::ThoError;
use thoosa_shared::resource::{Resource, ResourceId};
use thoosa_shared::run_configs::{RunId};
use thoosa_shared::*;



#[derive(Clone, PartialEq, Debug)]
pub enum DataBy {

    //--data-by=motif_id:1234
    MotifId(usize), //TODO u32
    //--data-by=find_motifs_run_id:1234[,5678][;select_motifs_run_ids:12,34,56]
    RunId {
        find_motifs_run_ids: Vec<RunId>,
        select_motifs_run_ids: Option<Vec<RunId>>,//for list_motif_slices's select_motifs_run_ids param
    },
    //--data-by=select_motifs_run_ids:12,34,56
    SelectedMotifs { select_motifs_run_ids: Vec<RunId> },
    //--data-by=resource:<resource_id>[;channels=1,2][;freq_filters=[...]]
    Resource {
        resource_id: ResourceId,
        channels: Option<Vec<u32>>,
        freq_filters: Option<Vec<InputFilter>>,
    },
    //--data-by=motif_net_applied:resource_id=<id>;net_path=<path>
    MotifNetApplied {
        resource_id: ResourceId,
        net_path: PathBuf,
    },
}

impl DataBy {
    fn parse_run_ids(payload: &str, run_id_name: &str) -> Result<Vec<RunId>, String> {
        payload
            .trim()
            .trim_matches(|ch| ch == '\'' || ch == '"')
            .split(',')
            .filter(|part| !part.is_empty())
            .map(|part| {
                part
                    .trim()
                    .trim_matches(|ch| ch == '\'' || ch == '"')
                    .replace(" ", "")
                    .parse::<RunId>()
                    .map_err(|e| format!("can't parse {part:?} to {run_id_name}: {e}"))
            })
            .collect()
    }

    fn parse_select_motifs_run_ids(payload: &str) -> Result<Vec<RunId>, String> {
        Self::parse_run_ids(payload, "select_motifs_run_id")
    }

    fn parse_find_motifs_run_ids(payload: &str) -> Result<Vec<RunId>, String> {
        let find_motifs_run_ids = Self::parse_run_ids(payload, "find_motifs_run_id")?;
        if find_motifs_run_ids.is_empty() {
            return Err("find_motifs_run_id payload is empty".to_string());
        }
        Ok(find_motifs_run_ids)
    }

    fn parse_run_id_payload(payload: &str) -> Result<Self, String> {
        let payload = payload
            .trim()
            .trim_matches(|ch| ch == '\'' || ch == '"');

        if payload.contains(';') {
            let mut segments = payload.split(';').filter(|segment| !segment.is_empty());
            let find_part = segments
                .next()
                .ok_or_else(|| "find_motifs_run_id payload is empty".to_string())?;

            let find_motifs_run_ids = Self::parse_find_motifs_run_ids(find_part)?;

            let mut select_motifs_run_ids = None;
            for segment in segments {
                let (raw_key, raw_value) = segment
                    .split_once('=')
                    .or_else(|| segment.split_once(':'))
                    .ok_or_else(|| format!("run_id payload segment {segment:?} should be key=value or key:value"))?;
                let key = raw_key
                    .trim()
                    .trim_matches(|ch| ch == '\'' || ch == '"');
                let value = raw_value
                    .trim()
                    .trim_matches(|ch| ch == '\'' || ch == '"');
                match key {
                    "select_motifs_run_ids" | "select_motifs_run_id" => {
                        select_motifs_run_ids = Some(Self::parse_select_motifs_run_ids(value)?);
                    }
                    _ => return Err(format!("unexpected run_id payload key {key:?}; expected select_motifs_run_ids")),
                }
            }

            return Ok(DataBy::RunId {
                find_motifs_run_ids,
                select_motifs_run_ids,
            });
        }

        let (find_part, select_part) = if let Some((find, rest)) = payload.split_once('_') {
            (find, Some(rest))
        } else {
            (payload, None)
        };

        let find_motifs_run_ids = Self::parse_find_motifs_run_ids(find_part)?;

        let select_motifs_run_ids = match select_part {
            None => None,
            Some("") => Some(vec![]),
            Some(values) => Some(Self::parse_select_motifs_run_ids(values)?),
        };

        Ok(DataBy::RunId {
            find_motifs_run_ids,
            select_motifs_run_ids,
        })
    }

    fn parse_channels(payload: &str) -> Result<Vec<u32>, String> {
        payload
            .split(',')
            .filter(|segment| !segment.is_empty())
            .map(|segment| {

                #[cfg(any(debug_assertions, tho_debug))] //logging is set up after cli args are parsed
                println!("parse_channels: segment={segment:?}");

                segment
                    .replace(" ", "")
                    .parse::<u32>()
                    .map_err(|e| format!("can't parse {segment:?} to channel: {e}"))
            })
            .collect()
    }

    fn parse_resource_payload(payload: &str) -> Result<Self, String> {
        #[cfg(any(debug_assertions, tho_debug))]
        println!("parse_resource_payload: payload={payload}");

        let mut segments = payload.split(';').filter(|segment| !segment.is_empty());
        let resource_id = segments
            .next()
            .ok_or_else(|| "resource payload is empty".to_string())?
            .parse::<ResourceId>()
            .map_err(|e| format!("can't parse resource_id in {payload:?}: {e}"))?;

        let mut channels = None;
        let mut freq_filters = None;

        #[cfg(any(debug_assertions, tho_debug))]
        println!("parse_resource_payload: segments={segments:?}");

        for segment in segments {
            let (key, value) = segment
                .split_once('=')
                .ok_or_else(|| format!("resource payload segment {segment:?} should be key=value"))?;

            #[cfg(any(debug_assertions, tho_debug))]
            println!("parse_resource_payload: key={key:?} value={value:?}");

            match key {
                "channels" => {
                    channels = Some(Self::parse_channels(value)?);
                }
                "freq_filters" => {
                    let parsed = serde_json5::from_str::<Vec<InputFilter>>(value)
                        .map_err(|e| format!("can't parse freq_filters JSON5 {value:?}: {e}"))?;
                    freq_filters = Some(parsed);
                }
                _ => {
                    return Err(format!("unexpected resource payload key {key:?}; expected channels or freq_filters"));
                }
            }
        }

        Ok(DataBy::Resource {
            resource_id,
            channels,
            freq_filters,
        })
    }

    fn parse_motif_net_applied_payload(payload: &str) -> Result<Self, String> {
        let segments = payload
            .split(';')
            .filter(|segment| !segment.is_empty())
            .collect::<Vec<_>>();

        if segments.len() == 2 && !segments[0].contains('=') && !segments[1].contains('=') {
            return Ok(DataBy::MotifNetApplied {
                resource_id: segments[0]
                    .parse::<ResourceId>()
                    .map_err(|e| format!("can't parse resource_id in {payload:?}: {e}"))?,
                net_path: PathBuf::from(segments[1]),
            });
        }

        let mut resource_id = None;
        let mut net_path = None;
        for segment in segments {
            let (key, value) = segment
                .split_once('=')
                .ok_or_else(|| format!("motif_net_applied payload segment {segment:?} should be key=value"))?;
            match key {
                "resource_id" => {
                    resource_id = Some(
                        value
                            .parse::<ResourceId>()
                            .map_err(|e| format!("can't parse resource_id {value:?}: {e}"))?,
                    );
                }
                "net_path" => {
                    net_path = Some(PathBuf::from(value));
                }
                _ => {
                    return Err(format!("unexpected motif_net_applied payload key {key:?}; expected resource_id or net_path"));
                }
            }
        }

        Ok(DataBy::MotifNetApplied {
            resource_id: resource_id.ok_or_else(|| "motif_net_applied payload missing resource_id".to_string())?,
            net_path: net_path.ok_or_else(|| "motif_net_applied payload missing net_path".to_string())?,
        })
    }
}

impl FromStr for DataBy {
    type Err = String;

    fn from_str(v: &str) -> Result<Self, Self::Err> {
        if let Some(rest) = v.strip_prefix("motif_id:") {
            Ok(DataBy::MotifId(
                rest.parse()
                    .map_err(|e| format!("can't parse {rest:?} to motif_id: {e}"))?,
            ))
        } else if let Some(rest) = v
            .strip_prefix("find_motifs_run_ids:")
            .or_else(|| v.strip_prefix("find_motifs_run_id:"))
            .or_else(|| v.strip_prefix("run_id:")) {
            DataBy::parse_run_id_payload(rest)
        } else if let Some(rest) = v
            .strip_prefix("select_motifs_run_ids:")
            .or_else(|| v.strip_prefix("select_motifs_run_id:"))
            .or_else(|| v.strip_prefix("good_motifs_run_id:")) {//renamed to select_motifs_run_id
            Ok(DataBy::SelectedMotifs {
                select_motifs_run_ids: Self::parse_select_motifs_run_ids(rest)?,
            })
        } else if let Some(rest) = v.strip_prefix("resource:") {
            DataBy::parse_resource_payload(rest)
        } else if let Some(rest) = v.strip_prefix("motif_net_applied:") {
            DataBy::parse_motif_net_applied_payload(rest)
        } else {
            Err(format!(
                "unexpected parameter {v}, should be motif_id:nnn, find_motifs_run_ids:nnn[,mmm][;select_motifs_run_ids[:=]...], select_motifs_run_ids:nnn[,mmm], resource:<resource_id>[;channels=1,2][;freq_filters=[...]], or motif_net_applied:resource_id=<id>;net_path=<path>"
            ))
        }
    }
}

impl From<&str> for DataBy {
    fn from(v: &str) -> DataBy {
        DataBy::from_str(v).unwrap_or_else(|err| panic!("{err}"))
    }
}

#[derive(Debug, Args, Clone)]
pub struct PlotDataArgs {
    #[arg(long, value_enum)]
    data_by: DataBy,

    #[arg(long, value_enum)]
    pub limit: Option<u32>,
}

impl PlotDataArgs {
    pub fn motifs_by(&self) -> DataBy {
        self.data_by.clone()
    }

    pub fn new(motifs_by: DataBy) -> Self {
        Self {
            data_by: motifs_by,
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
    use super::DataBy;
    use std::path::PathBuf;
    use thoosa_common::find_motifs_run_config::InputFilter;
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
    use thoosa_shared::{ThoError, cdt_for_dir};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn parses_data_by_motif_id_comment_format() {
        use std::str::FromStr;

        let parsed = DataBy::from_str("motif_id:1234").unwrap();
        assert_eq!(parsed, DataBy::MotifId(1234));
    }

    #[test]
    fn parses_data_by_find_motifs_run_id_comment_format() {
        use std::str::FromStr;

        let parsed = DataBy::from_str("find_motifs_run_id:11").unwrap();
        assert_eq!(parsed, DataBy::RunId { find_motifs_run_ids: vec![11], select_motifs_run_ids: None });
    }

    #[test]
    fn parses_data_by_find_motifs_run_id_with_select_filter_comment_format() {
        use std::str::FromStr;

        let parsed = DataBy::from_str("find_motifs_run_id:7;select_motifs_run_ids=10,20,30").unwrap();
        assert_eq!(parsed, DataBy::RunId { find_motifs_run_ids: vec![7], select_motifs_run_ids: Some(vec![10, 20, 30]) });
    }

    #[test]
    fn parses_data_by_run_id_multiple_find_and_select_run_ids() {
        use std::str::FromStr;

        let parsed = DataBy::from_str("run_id:1,2,3;select_motifs_run_ids:3,4,5").unwrap();
        assert_eq!(parsed, DataBy::RunId { find_motifs_run_ids: vec![1, 2, 3], select_motifs_run_ids: Some(vec![3, 4, 5]) });

        let parsed_quoted = DataBy::from_str("run_id:'1,2,3;select_motifs_run_ids:'3,4,5'").unwrap();
        assert_eq!(parsed_quoted, DataBy::RunId { find_motifs_run_ids: vec![1, 2, 3], select_motifs_run_ids: Some(vec![3, 4, 5]) });
    }

    #[test]
    fn parses_data_by_run_id_multiple_find_run_ids_without_select_filter() {
        use std::str::FromStr;

        let parsed = DataBy::from_str("run_id:1,2,3").unwrap();
        assert_eq!(parsed, DataBy::RunId { find_motifs_run_ids: vec![1, 2, 3], select_motifs_run_ids: None });
    }

    #[test]
    fn keeps_legacy_run_id_format_compatible() {
        use std::str::FromStr;

        let empty = DataBy::from_str("run_id:42_").unwrap();
        assert_eq!(empty, DataBy::RunId { find_motifs_run_ids: vec![42], select_motifs_run_ids: Some(vec![]) });
    }

    #[test]
    fn parses_data_by_selected_motifs_comment_prefix() {
        use std::str::FromStr;

        let parsed = DataBy::from_str("select_motifs_run_ids:99,101").unwrap();
        assert_eq!(parsed, DataBy::SelectedMotifs { select_motifs_run_ids: vec![99, 101] });
    }

    #[test]
    fn parses_data_by_resource_comment_format() {
        use std::str::FromStr;

        let resource = DataBy::from_str("resource:55;channels=1,3").unwrap();
        assert_eq!(
            resource,
            DataBy::Resource {
                resource_id: 55,
                channels: Some(vec![1, 3]),
                freq_filters: None,
            }
        );

        let resource = DataBy::from_str("resource:7;channels=0").unwrap();
        assert_eq!(
            resource,
            DataBy::Resource {
                resource_id: 7,
                channels: Some(vec![0]),
                freq_filters: None,
            }
        );

        let resource_with_filter = DataBy::from_str("resource:12;freq_filters=[{NotchBandFilter:{low_hz:58.0, high_hz:62.0}}]").unwrap();
        assert_eq!(
            resource_with_filter,
            DataBy::Resource {
                resource_id: 12,
                channels: None,
                freq_filters: Some(vec![InputFilter::NotchBandFilter(thoosa_common::transforms::filters::NotchBandFilterParams {
                    low_hz: 58.0,
                    high_hz: 62.0,
                })]),
            }
        );
    }

    #[test]
    fn parses_data_by_motif_net_applied_comment_format() {
        use std::str::FromStr;

        let parsed = DataBy::from_str("motif_net_applied:resource_id=101;net_path=13/22127_7777772").unwrap();
        assert_eq!(
            parsed,
            DataBy::MotifNetApplied {
                resource_id: 101,
                net_path: PathBuf::from("13/22127_7777772"),
            }
        );
    }
}
