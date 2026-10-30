use std::fmt::Debug;
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::Context as _;
use clap::{Args, ValueEnum};
use thoosa_common::transforms::filters::InputFilter;
use thoosa_common::utils::*;
use thoosa_shared::resource::ResourceId;
use thoosa_shared::run_configs::{RunConfigId};
use thoosa_shared::*;



#[derive(Clone, PartialEq, Debug)]
pub enum DataBy {

    //--data-by=motif_id:1234[,5678]
    MotifId(Vec<usize>), //TODO u32
    //--data-by=find_motifs_run_config_id:1234[,5678][;select_motifs_run_config_ids:12,34,56]
    RunId {
        find_motifs_run_config_ids: Vec<RunConfigId>,
        select_motifs_run_config_ids: Option<Vec<RunConfigId>>,//for list_motif_slices's select_motifs_run_config_ids param
    },
    //--data-by=select_motifs_run_config_ids:12,34,56
    SelectedMotifs { select_motifs_run_config_ids: Vec<RunConfigId> },
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

    fn parse_motif_ids(payload: &str) -> anyhow::Result<Vec<usize>> {
        let motif_ids = payload
            .trim()
            .trim_matches(|ch| ch == '\'' || ch == '"')
            .split(',')
            .filter(|part| !part.is_empty())
            .map(|part| {
                Ok(part
                    .trim()
                    .trim_matches(|ch| ch == '\'' || ch == '"')
                    .replace(" ", "")
                    .parse::<usize>()
                    .with_context(|| format!("can't parse {part:?} to motif_id"))?)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        if motif_ids.is_empty() {
            anyhow::bail!("motif_id payload is empty");
        }
        Ok(motif_ids)
    }

    fn parse_select_motifs_run_config_ids(payload: &str) -> anyhow::Result<Vec<RunConfigId>> {
        parse_run_config_ids_vec(payload)
    }

    fn parse_find_motifs_run_config_ids(payload: &str) -> anyhow::Result<Vec<RunConfigId>> {
        let find_motifs_run_config_ids = parse_run_config_ids_vec(payload)?;
        if find_motifs_run_config_ids.is_empty() {
            anyhow::bail!("find_motifs_run_config_id payload is empty");
        }
        Ok(find_motifs_run_config_ids)
    }

    fn parse_run_config_id_payload(payload: &str) -> anyhow::Result<Self> {
        let payload = payload
            .trim()
            .trim_matches(|ch| ch == '\'' || ch == '"');

        if payload.contains(';') {
            let mut segments = payload.split(';').filter(|segment| !segment.is_empty());
            let find_part = segments
                .next()
                .context("find_motifs_run_config_id payload is empty")?;

            let find_motifs_run_config_ids = Self::parse_find_motifs_run_config_ids(find_part)?;

            let mut select_motifs_run_config_ids = None;
            for segment in segments {
                let (raw_key, raw_value) = segment
                    .split_once('=')
                    .or_else(|| segment.split_once(':'))
                    .with_context(|| format!("run_config_id payload segment {segment:?} should be key=value or key:value"))?;
                let key = raw_key
                    .trim()
                    .trim_matches(|ch| ch == '\'' || ch == '"');
                let value = raw_value
                    .trim()
                    .trim_matches(|ch| ch == '\'' || ch == '"');
                match key {
                    "select_motifs_run_config_ids" | "select_motifs_run_config_id" => {
                        select_motifs_run_config_ids = Some(Self::parse_select_motifs_run_config_ids(value)?);
                    }
                    _ => anyhow::bail!("unexpected run_config_id payload key {key:?}; expected select_motifs_run_config_ids"),
                }
            }

            return Ok(DataBy::RunId {
                find_motifs_run_config_ids,
                select_motifs_run_config_ids,
            });
        }

        let (find_part, select_part) = if let Some((find, rest)) = payload.split_once('_') {
            (find, Some(rest))
        } else {
            (payload, None)
        };

        let find_motifs_run_config_ids = Self::parse_find_motifs_run_config_ids(find_part)?;

        let select_motifs_run_config_ids = match select_part {
            None => None,
            Some("") => Some(vec![]),
            Some(values) => Some(Self::parse_select_motifs_run_config_ids(values)?),
        };

        Ok(DataBy::RunId {
            find_motifs_run_config_ids,
            select_motifs_run_config_ids,
        })
    }

    fn parse_channels(payload: &str) -> anyhow::Result<Vec<u32>> {
        payload
            .split(',')
            .filter(|segment| !segment.is_empty())
            .map(|segment| {

                #[cfg(any(debug_assertions, tho_debug))] //logging is set up after cli args are parsed
                println!("parse_channels: segment={segment:?}");

                Ok(segment
                    .replace(" ", "")
                    .parse::<u32>()
                    .with_context(|| format!("can't parse {segment:?} to channel"))?)
            })
            .collect()
    }

    fn parse_resource_payload(payload: &str) -> anyhow::Result<Self> {
        #[cfg(any(debug_assertions, tho_debug))]
        println!("parse_resource_payload: payload={payload}");

        let mut segments = payload.split(';').filter(|segment| !segment.is_empty());
        let resource_id = segments
            .next()
            .context("resource payload is empty")?
            .parse::<ResourceId>()
            .with_context(|| format!("can't parse resource_id in {payload:?}"))?;

        let mut channels = None;
        let mut freq_filters = None;

        #[cfg(any(debug_assertions, tho_debug))]
        println!("parse_resource_payload: segments={segments:?}");

        for segment in segments {
            let (key, value) = segment
                .split_once('=')
                .with_context(|| format!("resource payload segment {segment:?} should be key=value"))?;

            #[cfg(any(debug_assertions, tho_debug))]
            println!("parse_resource_payload: key={key:?} value={value:?}");

            match key {
                "channels" => {
                    channels = Some(Self::parse_channels(value)?);
                }
                "freq_filters" => {
                    let parsed = serde_json5::from_str::<Vec<InputFilter>>(value)
                        .with_context(|| format!("can't parse freq_filters JSON5 {value:?}"))?;
                    freq_filters = Some(parsed);
                }
                _ => {
                    anyhow::bail!("unexpected resource payload key {key:?}; expected channels or freq_filters");
                }
            }
        }

        Ok(DataBy::Resource {
            resource_id,
            channels,
            freq_filters,
        })
    }

    fn parse_motif_net_applied_payload(payload: &str) -> anyhow::Result<Self> {
        let segments = payload
            .split(';')
            .filter(|segment| !segment.is_empty())
            .collect::<Vec<_>>();

        if segments.len() == 2 && !segments[0].contains('=') && !segments[1].contains('=') {
            return Ok(DataBy::MotifNetApplied {
                resource_id: segments[0]
                    .parse::<ResourceId>()
                    .with_context(|| format!("can't parse resource_id in {payload:?}"))?,
                net_path: PathBuf::from(segments[1]),
            });
        }

        let mut resource_id = None;
        let mut net_path = None;
        for segment in segments {
            let (key, value) = segment
                .split_once('=')
                .with_context(|| format!("motif_net_applied payload segment {segment:?} should be key=value"))?;
            match key {
                "resource_id" => {
                    resource_id = Some(
                        value
                            .parse::<ResourceId>()
                            .with_context(|| format!("can't parse resource_id {value:?}"))?,
                    );
                }
                "net_path" => {
                    net_path = Some(PathBuf::from(value));
                }
                _ => {
                    anyhow::bail!("unexpected motif_net_applied payload key {key:?}; expected resource_id or net_path");
                }
            }
        }

        Ok(DataBy::MotifNetApplied {
            resource_id: resource_id.context("motif_net_applied payload missing resource_id")?,
            net_path: net_path.context("motif_net_applied payload missing net_path")?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum Resolution {
    #[default]
    Normal,
    High,
}

impl Resolution {
    pub fn plot_size(self, normal_size: (u32, u32)) -> (u32, u32) {
        match self {
            Self::Normal => normal_size,
            // Bias the extra pixels toward the X axis so dense time-series details remain visible.
            Self::High => (normal_size.0.saturating_mul(5), normal_size.1.saturating_mul(2)),
        }
    }
}

impl FromStr for DataBy {
    type Err = anyhow::Error;

    fn from_str(v: &str) -> Result<Self, Self::Err> {
        if let Some(rest) = v.strip_prefix("motif_id:") {
            Ok(DataBy::MotifId(Self::parse_motif_ids(rest)?))
        } else if let Some(rest) = v
            .strip_prefix("find_motifs_run_config_ids:")
            .or_else(|| v.strip_prefix("find_motifs_run_config_id:"))
            .or_else(|| v.strip_prefix("run_config_id:")) {
            DataBy::parse_run_config_id_payload(rest)
        } else if let Some(rest) = v
            .strip_prefix("select_motifs_run_config_ids:")
            .or_else(|| v.strip_prefix("select_motifs_run_config_id:"))
            .or_else(|| v.strip_prefix("good_motifs_run_config_id:")) {//renamed to select_motifs_run_config_id
            Ok(DataBy::SelectedMotifs {
                select_motifs_run_config_ids: Self::parse_select_motifs_run_config_ids(rest)?,
            })
        } else if let Some(rest) = v.strip_prefix("resource:") {
            DataBy::parse_resource_payload(rest)
        } else if let Some(rest) = v.strip_prefix("motif_net_applied:") {
            DataBy::parse_motif_net_applied_payload(rest)
        } else {
            anyhow::bail!(
                "unexpected parameter {v}, should be motif_id:nnn[,mmm], find_motifs_run_config_ids:nnn[,mmm][;select_motifs_run_config_ids[:=]...], select_motifs_run_config_ids:nnn[,mmm], resource:<resource_id>[;channels=1,2][;freq_filters=[...]], or motif_net_applied:resource_id=<id>;net_path=<path>"
            )
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

    #[arg(long, value_enum, default_value_t = Resolution::Normal)]
    pub resolution: Resolution,

    #[arg(long, default_value_t = false)]
    pub stdout: bool,
}

impl PlotDataArgs {
    pub fn data_by(&self) -> DataBy {
        self.data_by.clone()
    }

    pub fn resolution(&self) -> Resolution {
        self.resolution
    }

    pub fn print_to_stdout(&self) -> bool {
        self.stdout
    }

    pub fn new(motifs_by: DataBy) -> Self {
        Self {
            data_by: motifs_by,
            limit: None,
            resolution: Resolution::Normal,
            stdout: false,
        }
    }
}

#[derive(Debug, Args, Clone)]
pub struct MotifsToTensorArgs {
    #[arg(long)]
    run_config_id: RunConfigId,

    #[arg(long, default_value_t = 1000)]
    window: i64,
}

impl MotifsToTensorArgs {
    pub fn window(&self) -> i64 {
        self.window
    }
    pub fn run_config_id(&self) -> RunConfigId {
        self.run_config_id
    }
}

#[cfg(test)]
mod motifs_tests {
    use super::{DataBy, PlotDataArgs, Resolution};
    use std::path::PathBuf;


    use thoosa_common::transforms::filters::InputFilter;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn parses_data_by_motif_id_comment_format() {
        use std::str::FromStr;

        let parsed = DataBy::from_str("motif_id:1234").unwrap();
        assert_eq!(parsed, DataBy::MotifId(vec![1234]));
    }

    #[test]
    fn parses_data_by_multiple_motif_ids() {
        use std::str::FromStr;

        let parsed = DataBy::from_str("motif_id:1,2,3").unwrap();
        assert_eq!(parsed, DataBy::MotifId(vec![1, 2, 3]));

        let parsed_spaced = DataBy::from_str("motif_id:'1, 2'").unwrap();
        assert_eq!(parsed_spaced, DataBy::MotifId(vec![1, 2]));
    }

    #[test]
    fn parses_data_by_find_motifs_run_config_id_comment_format() {
        use std::str::FromStr;

        let parsed = DataBy::from_str("find_motifs_run_config_id:11").unwrap();
        assert_eq!(parsed, DataBy::RunId { find_motifs_run_config_ids: vec![11], select_motifs_run_config_ids: None });
    }

    #[test]
    fn parses_data_by_find_motifs_run_config_id_with_select_filter_comment_format() {
        use std::str::FromStr;

        let parsed = DataBy::from_str("find_motifs_run_config_id:7;select_motifs_run_config_ids=10,20,30").unwrap();
        assert_eq!(parsed, DataBy::RunId { find_motifs_run_config_ids: vec![7], select_motifs_run_config_ids: Some(vec![10, 20, 30]) });
    }

    #[test]
    fn parses_data_by_run_config_id_multiple_find_and_select_run_config_ids() {
        use std::str::FromStr;

        let parsed = DataBy::from_str("run_config_id:1,2,3;select_motifs_run_config_ids:3,4,5").unwrap();
        assert_eq!(parsed, DataBy::RunId { find_motifs_run_config_ids: vec![1, 2, 3], select_motifs_run_config_ids: Some(vec![3, 4, 5]) });

        let parsed_quoted = DataBy::from_str("run_config_id:'1,2,3;select_motifs_run_config_ids:'3,4,5'").unwrap();
        assert_eq!(parsed_quoted, DataBy::RunId { find_motifs_run_config_ids: vec![1, 2, 3], select_motifs_run_config_ids: Some(vec![3, 4, 5]) });
    }

    #[test]
    fn parses_data_by_run_config_id_multiple_find_run_config_ids_without_select_filter() {
        use std::str::FromStr;

        let parsed = DataBy::from_str("run_config_id:1,2,3").unwrap();
        assert_eq!(parsed, DataBy::RunId { find_motifs_run_config_ids: vec![1, 2, 3], select_motifs_run_config_ids: None });
    }

    #[test]
    fn high_resolution_scales_plot_size_for_finer_detail() {
        assert_eq!(Resolution::Normal.plot_size((2000, 1000)), (2000, 1000));
        assert_eq!(Resolution::High.plot_size((2000, 1000)), (10_000, 2_000));
    }

    #[test]
    fn plot_data_args_new_defaults_to_normal_resolution() {
        let args = PlotDataArgs::new(DataBy::MotifId(vec![17]));

        assert_eq!(args.data_by(), DataBy::MotifId(vec![17]));
        assert_eq!(args.resolution(), Resolution::Normal);
        assert_eq!(args.limit, None);
        assert!(!args.print_to_stdout());
    }

    #[test]
    fn keeps_legacy_run_config_id_format_compatible() {
        use std::str::FromStr;

        let empty = DataBy::from_str("run_config_id:42_").unwrap();
        assert_eq!(empty, DataBy::RunId { find_motifs_run_config_ids: vec![42], select_motifs_run_config_ids: Some(vec![]) });
    }

    #[test]
    fn parses_data_by_selected_motifs_comment_prefix() {
        use std::str::FromStr;

        let parsed = DataBy::from_str("select_motifs_run_config_ids:99,101").unwrap();
        assert_eq!(parsed, DataBy::SelectedMotifs { select_motifs_run_config_ids: vec![99, 101] });
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
