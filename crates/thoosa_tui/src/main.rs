use interprocess::local_socket::{
    tokio::{prelude::*, Stream},
    GenericFilePath, GenericNamespaced, Name, ToFsName, ToNsName,
};
use chrono::{NaiveDate, NaiveDateTime};
use std::collections::{HashMap, HashSet};
use std::io;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use clap::Parser;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode};
use log::{debug, info, warn};
use ratatui::Frame;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use thoosa_shared::collect_futures_eta::{CollectEtaProgressInvocation, CollectEtaProgressState, COLLECT_ETA_PROGRESS_SOCKET_ID};
use thoosa_shared::device_stats::ThoDeviceStats;
use thoosa_shared::val::Val;
use thoosa_shared::ThoError;
use thoosa_store::{RunConfigEntry, RunConfigFilter, RunMotifEntry, SqlStore, Store};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tracing::Instrument;

const SEARCH_TYPE_NONE_LABEL: &str = "no value";
const RUN_ON_DATE_FORMAT: &str = "%Y-%m-%d";
const RUN_ON_DATETIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S";
const DEFAULT_PLOT_PROJECT: &str = "thoosa_hbac";

#[derive(Clone, Debug)]
struct RunTreeNode {
    entry: RunConfigEntry,
    input_run_config_ids: Vec<u32>,
    output_run_config_ids: Vec<u32>,
    children_loaded: bool,
    expanded: bool,
}

#[derive(Clone, Debug, PartialEq)]
enum VisibleRunConfigRowKind {
    RunOnDate {
        run_on_date: String,
    },
    RunConfig {
        run_config_id: u32,
    },
    Motif {
        run_config_id: u32,
        motif: RunMotifEntry,
    },
}

#[derive(Clone, Debug, PartialEq)]
struct VisibleRunConfigRow {
    kind: VisibleRunConfigRowKind,
    depth: usize,
    has_children: bool,
    expanded: bool,
}

impl VisibleRunConfigRow {
    fn run_on_date_group(run_on_date: String, expanded: bool) -> Self {
        Self {
            kind: VisibleRunConfigRowKind::RunOnDate { run_on_date },
            depth: 0,
            has_children: true,
            expanded,
        }
    }

    fn run_config(run_config_id: u32, depth: usize, has_children: bool, expanded: bool) -> Self {
        Self {
            kind: VisibleRunConfigRowKind::RunConfig { run_config_id },
            depth,
            has_children,
            expanded,
        }
    }

    fn motif(run_config_id: u32, motif: RunMotifEntry, depth: usize) -> Self {
        Self {
            kind: VisibleRunConfigRowKind::Motif { run_config_id, motif },
            depth,
            has_children: false,
            expanded: false,
        }
    }

    fn run_config_id(&self) -> Option<u32> {
        match self.kind {
            VisibleRunConfigRowKind::RunOnDate { .. } => None,
            VisibleRunConfigRowKind::RunConfig { run_config_id } => Some(run_config_id),
            VisibleRunConfigRowKind::Motif { .. } => None,
        }
    }

    fn run_on_date(&self) -> Option<&str> {
        match &self.kind {
            VisibleRunConfigRowKind::RunOnDate { run_on_date } => Some(run_on_date.as_str()),
            VisibleRunConfigRowKind::RunConfig { .. } | VisibleRunConfigRowKind::Motif { .. } => None,
        }
    }

    fn tree_motif(&self) -> Option<&RunMotifEntry> {
        match &self.kind {
            VisibleRunConfigRowKind::RunOnDate { .. } | VisibleRunConfigRowKind::RunConfig { .. } => None,
            VisibleRunConfigRowKind::Motif { motif, .. } => Some(motif),
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "thoosa_tui")]
#[command(about = "Inspect run configs and values from workspace database")]
struct Args {
    ws_path: PathBuf,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RunConfigSearch {
    id: String,
    run_config_type: Option<String>,
    descr: String,
    run_on_from: String,
    run_on_to: String,
}

impl RunConfigSearch {
    fn is_empty(&self) -> bool {
        self.id.trim().is_empty()
            && self.run_config_type.is_none()
            && self.descr.trim().is_empty()
            && self.run_on_from.trim().is_empty()
            && self.run_on_to.trim().is_empty()
    }

    fn normalized(&self) -> Result<NormalizedRunConfigSearch, String> {
        let id = parse_run_config_id_filter(&self.id)?;
        let descr = self.descr.trim().to_string();
        let run_on_from = normalize_run_on_filter(&self.run_on_from, false)?;
        let run_on_to = normalize_run_on_filter(&self.run_on_to, true)?;

        if let (Some(from), Some(to)) = (&run_on_from, &run_on_to) {
            if from > to {
                return Err("run_on from must be <= run_on to".to_string());
            }
        }

        Ok(NormalizedRunConfigSearch {
            id,
            run_config_type: self.run_config_type.clone(),
            descr_lower: descr.to_lowercase(),
            run_on_from,
            run_on_to,
        })
    }

    fn summary(&self) -> String {
        if self.is_empty() {
            return "none".to_string();
        }

        [
            if self.id.trim().is_empty() {
                None
            } else {
                Some(format!("id={}", self.id.trim()))
            },
            self.run_config_type.as_ref().map(|value| format!("type={value}")),
            if self.descr.trim().is_empty() {
                None
            } else {
                Some(format!("descr~{}", self.descr.trim()))
            },
            if self.run_on_from.trim().is_empty() {
                None
            } else {
                Some(format!("from={}", self.run_on_from.trim()))
            },
            if self.run_on_to.trim().is_empty() {
                None
            } else {
                Some(format!("to={}", self.run_on_to.trim()))
            },
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<String>>()
        .join(", ")
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct SearchPopupState {
    draft: RunConfigSearch,
    selected_field_idx: usize,
    type_selected_idx: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SearchPopupRow {
    Field(SearchField),
    Ok,
    Cancel,
}

impl SearchPopupRow {
    fn from_idx(idx: usize) -> Self {
        if let Some(field) = SearchField::ALL.get(idx).copied() {
            Self::Field(field)
        } else if idx == SearchField::ALL.len() {
            Self::Ok
        } else {
            Self::Cancel
        }
    }

    fn count() -> usize {
        SearchField::ALL.len() + 2
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PanelSearchTarget {
    RunConfigs,
    Motifs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PanelSearchMatch {
    line_idx: usize,
    start: usize,
    len: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct PanelSearchState {
    draft: String,
    pattern: String,
    target: Option<PanelSearchTarget>,
    current_match_idx: Option<usize>,
    input_open: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NormalizedRunConfigSearch {
    id: Option<u32>,
    run_config_type: Option<String>,
    descr_lower: String,
    run_on_from: Option<String>,
    run_on_to: Option<String>,
}

impl NormalizedRunConfigSearch {
    fn matches(&self, entry: &RunConfigEntry) -> bool {
        if let Some(id) = self.id {
            if entry.id != id {
                return false;
            }
        }

        if let Some(run_config_type) = self.run_config_type.as_ref() {
            if &entry.run_config_type != run_config_type {
                return false;
            }
        }

        if !self.descr_lower.is_empty() && !entry.descr.to_lowercase().contains(self.descr_lower.as_str()) {
            return false;
        }

        if let Some(run_on_from) = self.run_on_from.as_ref() {
            if entry.run_on < *run_on_from {
                return false;
            }
        }

        if let Some(run_on_to) = self.run_on_to.as_ref() {
            if entry.run_on > *run_on_to {
                return false;
            }
        }

        true
    }
}

struct App {
    store: Box<dyn Store>,
    ws_path: PathBuf,
    db_path: PathBuf,
    all_run_configs: Vec<RunConfigEntry>,
    loaded_input_pairs: Vec<(u32, u32)>,
    known_output_run_config_ids: HashMap<u32, Vec<u32>>,
    known_run_config_ids_with_outputs: HashSet<u32>,
    paged_run_config_offset: usize,
    has_more_paged_run_configs: bool,
    run_config_tree: HashMap<u32, RunTreeNode>,
    run_configs: Vec<VisibleRunConfigRow>,
    main_list_offset: usize,
    selected_idx: usize,
    selected_values: Vec<(String, String)>,
    selected_run_motifs: Vec<RunMotifEntry>,
    show_selected_run_motifs: bool,
    run_motifs_cache: HashMap<u32, Vec<RunMotifEntry>>,
    collapsed_run_on_dates: HashSet<String>,
    selected_motif_idx: usize,
    motif_list_offset: usize,
    prefer_right_motifs: bool,
    status: Option<String>,
    selected_area: SelectedArea,
    run_on_sort_order: SortOrder,
    search_filters: RunConfigSearch,
    run_config_types: Vec<String>,
    type_popup_selected_idx: usize,
    search_popup: SearchPopupState,
    view_mode: ViewMode,
    view_popup_selected_idx: usize,
    pending_gg: Option<(SelectedArea, Instant)>,
    pending_plot_open: Option<PendingMotifPlotOpen>,
    panel_search: PanelSearchState,
    progress_state: CollectEtaProgressState,
    progress_status: String,
    progress_last_update: Option<Instant>,
}

struct PendingMotifPlotOpen {
    motif_id: u32,
    started_at: Instant,
    task: JoinHandle<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SelectedArea {
    LeftRunConfigs,
    RightMotifs,
    ProgressView,
    TypeFilterPopup,
    SearchPopup,
    SearchTypePopup,
    ViewPopup,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ViewMode {
    RunConfigs,
    Progress,
}

impl ViewMode {
    const ALL: [Self; 2] = [Self::RunConfigs, Self::Progress];

    fn label(self) -> &'static str {
        match self {
            Self::RunConfigs => "Run configs",
            Self::Progress => "Progress",
        }
    }
}

#[derive(Clone, Copy)]
enum SortOrder {
    Asc,
    Desc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SearchField {
    Id,
    Type,
    Descr,
    RunOnFrom,
    RunOnTo,
}

impl SearchField {
    const ALL: [Self; 5] = [Self::Id, Self::Type, Self::Descr, Self::RunOnFrom, Self::RunOnTo];

    fn label(self) -> &'static str {
        match self {
            Self::Id => "id",
            Self::Type => "type",
            Self::Descr => "descr",
            Self::RunOnFrom => "run_on from",
            Self::RunOnTo => "run_on to",
        }
    }
}

impl App {
    fn new(store: Box<dyn Store>, ws_path: PathBuf, db_path: PathBuf) -> Self {
        Self {
            store,
            ws_path,
            db_path,
            all_run_configs: vec![],
            loaded_input_pairs: vec![],
            known_output_run_config_ids: HashMap::new(),
            known_run_config_ids_with_outputs: HashSet::new(),
            paged_run_config_offset: 0,
            has_more_paged_run_configs: true,
            run_config_tree: HashMap::new(),
            run_configs: vec![],
            main_list_offset: 0,
            selected_idx: 0,
            selected_values: vec![],
            selected_run_motifs: vec![],
            show_selected_run_motifs: false,
            run_motifs_cache: HashMap::new(),
            collapsed_run_on_dates: HashSet::new(),
            selected_motif_idx: 0,
            motif_list_offset: 0,
            prefer_right_motifs: false,
            status: None,
            selected_area: SelectedArea::LeftRunConfigs,
            run_on_sort_order: SortOrder::Desc,
            search_filters: RunConfigSearch::default(),
            run_config_types: vec![],
            type_popup_selected_idx: 0,
            search_popup: SearchPopupState::default(),
            view_mode: ViewMode::RunConfigs,
            view_popup_selected_idx: 0,
            pending_gg: None,
            pending_plot_open: None,
            panel_search: PanelSearchState::default(),
            progress_state: CollectEtaProgressState::default(),
            progress_status: "Waiting for progress socket.".to_string(),
            progress_last_update: None,
        }
    }

    fn selected_run_config_id(&self) -> Option<u32> {
        self.run_configs.get(self.selected_idx).and_then(VisibleRunConfigRow::run_config_id)
    }

    fn selected_run_entry(&self) -> Option<&RunConfigEntry> {
        self.selected_run_config_id()
            .and_then(|run_config_id| self.run_config_tree.get(&run_config_id).map(|node| &node.entry))
    }

    fn selected_tree_row(&self) -> Option<&VisibleRunConfigRow> {
        self.run_configs.get(self.selected_idx)
    }

    fn selected_run_on_date(&self) -> Option<&str> {
        self.selected_tree_row().and_then(VisibleRunConfigRow::run_on_date)
    }

    fn selected_tree_motif(&self) -> Option<&RunMotifEntry> {
        self.selected_tree_row().and_then(VisibleRunConfigRow::tree_motif)
    }

    fn selected_motif(&self) -> Option<&RunMotifEntry> {
        self.selected_run_motifs.get(self.selected_motif_idx)
    }

    fn motif_panel_visible(&self) -> bool {
        self.show_selected_run_motifs && !self.selected_run_motifs.is_empty()
    }

    fn has_selected_run_motifs(&self) -> bool {
        self.motif_panel_visible()
    }

    fn selected_run_supports_motifs(&self) -> bool {
        self.selected_run_entry()
            .map(|entry| run_config_type_supports_motifs(entry.run_config_type.as_str()))
            .unwrap_or(false)
    }

    fn selected_run_supports_inline_motif_rows(&self) -> bool {
        self.selected_run_entry()
            .map(|entry| run_config_type_supports_inline_motif_rows(entry.run_config_type.as_str()))
            .unwrap_or(false)
    }

    fn selected_popup_type(&self) -> Option<String> {
        self.run_config_types.get(self.type_popup_selected_idx).cloned()
    }

    fn selected_search_popup_row(&self) -> SearchPopupRow {
        SearchPopupRow::from_idx(self.search_popup.selected_field_idx)
    }

    fn selected_search_field(&self) -> Option<SearchField> {
        match self.selected_search_popup_row() {
            SearchPopupRow::Field(field) => Some(field),
            SearchPopupRow::Ok | SearchPopupRow::Cancel => None,
        }
    }

    fn selected_search_popup_type(&self) -> Option<String> {
        if self.search_popup.type_selected_idx == 0 {
            None
        } else {
            self.run_config_types.get(self.search_popup.type_selected_idx - 1).cloned()
        }
    }

    fn selected_view_mode(&self) -> ViewMode {
        ViewMode::ALL
            .get(self.view_popup_selected_idx)
            .copied()
            .unwrap_or(ViewMode::RunConfigs)
    }

    fn sort_order_label(&self) -> &'static str {
        match self.run_on_sort_order {
            SortOrder::Asc => "asc",
            SortOrder::Desc => "desc",
        }
    }

    fn default_run_configs_area(&self) -> SelectedArea {
        if self.prefer_right_motifs && self.has_selected_run_motifs() {
            SelectedArea::RightMotifs
        } else {
            SelectedArea::LeftRunConfigs
        }
    }

    fn run_on_date_expanded(&self, run_on_date: &str) -> bool {
        !self.collapsed_run_on_dates.contains(run_on_date)
    }

    fn active_panel_search_target(&self) -> Option<PanelSearchTarget> {
        if self.view_mode != ViewMode::RunConfigs {
            return None;
        }

        match self.selected_area {
            SelectedArea::LeftRunConfigs => Some(PanelSearchTarget::RunConfigs),
            SelectedArea::RightMotifs => Some(PanelSearchTarget::Motifs),
            _ => None,
        }
    }

    fn open_panel_search_input(&mut self) {
        let Some(target) = self.active_panel_search_target() else {
            return;
        };
        if self.panel_search.target != Some(target) {
            self.panel_search.pattern.clear();
            self.panel_search.current_match_idx = None;
        }
        self.panel_search.target = Some(target);
        self.panel_search.draft = self.panel_search.pattern.clone();
        self.panel_search.input_open = true;
    }

    fn apply_panel_search_input(&mut self) {
        let Some(target) = self.panel_search.target else {
            return;
        };
        self.panel_search.pattern = self.panel_search.draft.trim().to_string();
        self.panel_search.current_match_idx = None;
        self.panel_search.input_open = false;
        if self.panel_search.pattern.is_empty() {
            self.status = Some("Panel search cleared.".to_string());
            return;
        }

        let matches = self.panel_search_matches(target);
        self.status = Some(if matches.is_empty() {
            format!("No matches for '{}'.", self.panel_search.pattern)
        } else {
            format!("{} matches for '{}'.", matches.len(), self.panel_search.pattern)
        });
    }

    fn close_panel_search_input(&mut self) {
        self.panel_search.input_open = false;
    }

    fn panel_search_pattern(&self, target: PanelSearchTarget) -> Option<&str> {
        if self.panel_search.target == Some(target) && !self.panel_search.pattern.is_empty() {
            Some(self.panel_search.pattern.as_str())
        } else {
            None
        }
    }

    fn panel_search_lines(&self, target: PanelSearchTarget) -> Vec<String> {
        match target {
            PanelSearchTarget::RunConfigs => self.run_configs.iter().map(|row| format_run_tree_row(self, row)).collect::<Vec<String>>(),
            PanelSearchTarget::Motifs => self.selected_run_motifs.iter().map(format_run_motif_row).collect::<Vec<String>>(),
        }
    }

    fn panel_search_matches(&self, target: PanelSearchTarget) -> Vec<PanelSearchMatch> {
        let Some(pattern) = self.panel_search_pattern(target) else {
            return vec![];
        };
        if pattern.is_empty() {
            return vec![];
        }

        self.panel_search_lines(target)
            .into_iter()
            .enumerate()
            .flat_map(|(line_idx, line)| {
                find_case_insensitive_matches(line.as_str(), pattern)
                    .into_iter()
                    .map(move |(start, len)| PanelSearchMatch { line_idx, start, len })
            })
            .collect::<Vec<PanelSearchMatch>>()
    }

    fn navigate_panel_search(&mut self, forward: bool) -> bool {
        let Some(target) = self.active_panel_search_target() else {
            return false;
        };
        if self.panel_search.target != Some(target) || self.panel_search.pattern.is_empty() {
            return false;
        }

        let matches = self.panel_search_matches(target);
        if matches.is_empty() {
            self.panel_search.current_match_idx = None;
            self.status = Some(format!("No matches for '{}'.", self.panel_search.pattern));
            return false;
        }

        let current_match_idx = self.panel_search.current_match_idx.filter(|idx| *idx < matches.len());
        if current_match_idx != self.panel_search.current_match_idx {
            self.panel_search.current_match_idx = current_match_idx;
        }

        let next_idx = match (current_match_idx, forward) {
            (Some(current_idx), true) if current_idx + 1 < matches.len() => current_idx + 1,
            (Some(current_idx), false) if current_idx > 0 => current_idx - 1,
            (None, true) => 0,
            (None, false) => matches.len().saturating_sub(1),
            (Some(_), true) => {
                self.status = Some("Reached last match.".to_string());
                return false;
            }
            (Some(_), false) => {
                self.status = Some("Reached first match.".to_string());
                return false;
            }
        };

        self.panel_search.current_match_idx = Some(next_idx);
        self.reveal_panel_search_match(target, matches[next_idx]);
        self.status = Some(format!("Match {}/{} for '{}'.", next_idx + 1, matches.len(), self.panel_search.pattern));
        true
    }

    fn reveal_panel_search_match(&mut self, target: PanelSearchTarget, search_match: PanelSearchMatch) {
        match target {
            PanelSearchTarget::RunConfigs => {
                if self.run_configs.is_empty() {
                    self.selected_idx = 0;
                    self.main_list_offset = 0;
                    return;
                }

                self.selected_idx = search_match.line_idx.min(self.run_configs.len().saturating_sub(1));
                let page_size = self.main_page_size();
                let max_offset = self.max_main_list_offset();
                scroll_offset_by_pages_until_visible(&mut self.main_list_offset, page_size, max_offset, self.selected_idx);
            }
            PanelSearchTarget::Motifs => {
                if self.selected_run_motifs.is_empty() {
                    self.selected_motif_idx = 0;
                    self.motif_list_offset = 0;
                    return;
                }

                self.selected_motif_idx = search_match.line_idx.min(self.selected_run_motifs.len().saturating_sub(1));
                let page_size = self.motif_page_size();
                let max_offset = self.max_motif_list_offset();
                scroll_offset_by_pages_until_visible(&mut self.motif_list_offset, page_size, max_offset, self.selected_motif_idx);
            }
        }
    }

    fn apply_selected_run_motifs(&mut self, motifs: Vec<RunMotifEntry>, keep_motif_id: Option<u32>) {
        self.selected_run_motifs = motifs;
        if self.panel_search.target == Some(PanelSearchTarget::Motifs) {
            self.panel_search.current_match_idx = None;
        }
        if self.selected_run_motifs.is_empty() {
            self.show_selected_run_motifs = false;
            self.selected_motif_idx = 0;
            self.motif_list_offset = 0;
            if self.selected_area == SelectedArea::RightMotifs {
                self.selected_area = SelectedArea::LeftRunConfigs;
                self.prefer_right_motifs = false;
            }
            return;
        }

        self.selected_motif_idx = keep_motif_id
            .and_then(|motif_id| self.selected_run_motifs.iter().position(|motif| motif.motif_id == motif_id))
            .unwrap_or(0);
        self.ensure_motif_selection_visible();
    }

    async fn load_run_motifs(&mut self, run_config_id: u32) -> Result<Vec<RunMotifEntry>, ThoError> {
        if let Some(cached) = self.run_motifs_cache.get(&run_config_id).cloned() {
            return Ok(cached);
        }

        let motifs = self.store.list_run_motifs(run_config_id).instrument(tracing::debug_span!("load_run_motifs/1", file = file!(), line = line!())).await?;
        self.run_motifs_cache.insert(run_config_id, motifs.clone());
        Ok(motifs)
    }

    async fn load_selected_run_motifs(&mut self) -> bool {
        let keep_motif_id = self.selected_motif().map(|motif| motif.motif_id);
        let Some(run_config_id) = self.selected_run_config_id() else {
            self.selected_run_motifs.clear();
            self.selected_motif_idx = 0;
            self.motif_list_offset = 0;
            return false;
        };

        if !self.selected_run_supports_motifs() {
            self.selected_run_motifs.clear();
            self.selected_motif_idx = 0;
            self.motif_list_offset = 0;
            return false;
        }

        match self.load_run_motifs(run_config_id).instrument(tracing::debug_span!("load_selected_run_motifs/1", file = file!(), line = line!())).await {
            Ok(motifs) => {
                self.apply_selected_run_motifs(motifs, keep_motif_id);
                if self.selected_run_motifs.is_empty() {
                    self.status = Some(format!("No motifs found for run_config_id={run_config_id}."));
                    false
                } else {
                    self.status = None;
                    true
                }
            }
            Err(e) => {
                warn!("load_selected_run_motifs failed: run_config_id={} err={e:?}", run_config_id);
                self.status = Some(format!("Failed to list motifs for run_config_id={run_config_id}: {e:?}"));
                false
            }
        }
    }

    async fn toggle_selected_run_inline_motifs(&mut self) -> bool {
        let Some(run_config_id) = self.selected_run_config_id() else {
            return false;
        };
        if !self.selected_run_supports_inline_motif_rows() {
            return false;
        }

        if self.run_config_tree.get(&run_config_id).map(|node| node.expanded).unwrap_or(false) {
            if let Some(node) = self.run_config_tree.get_mut(&run_config_id) {
                node.expanded = false;
            }
            self.rebuild_visible_run_configs(Some(run_config_id));
            return true;
        }

        match self.load_run_motifs(run_config_id).instrument(tracing::debug_span!("toggle_selected_run_inline_motifs/1", file = file!(), line = line!())).await {
            Ok(motifs) if motifs.is_empty() => {
                self.status = Some(format!("No motifs found for run_config_id={run_config_id}."));
                false
            }
            Ok(_) => {
                if let Some(node) = self.run_config_tree.get_mut(&run_config_id) {
                    node.expanded = true;
                }
                self.rebuild_visible_run_configs(Some(run_config_id));
                self.status = None;
                true
            }
            Err(e) => {
                warn!("toggle_selected_run_inline_motifs failed: run_config_id={} err={e:?}", run_config_id);
                self.status = Some(format!("Failed to list motifs for run_config_id={run_config_id}: {e:?}"));
                false
            }
        }
    }

    async fn expand_selected_run_inline_motifs(&mut self) -> bool {
        let Some(run_config_id) = self.selected_run_config_id() else {
            return false;
        };
        if !self.selected_run_supports_inline_motif_rows() || self.run_config_tree.get(&run_config_id).map(|node| node.expanded).unwrap_or(false) {
            return false;
        }

        match self.load_run_motifs(run_config_id).instrument(tracing::debug_span!("expand_selected_run_inline_motifs/1", file = file!(), line = line!())).await {
            Ok(motifs) if motifs.is_empty() => {
                self.status = Some(format!("No motifs found for run_config_id={run_config_id}."));
                false
            }
            Ok(_) => {
                if let Some(node) = self.run_config_tree.get_mut(&run_config_id) {
                    node.expanded = true;
                }
                self.rebuild_visible_run_configs(Some(run_config_id));
                self.status = None;
                true
            }
            Err(e) => {
                warn!("expand_selected_run_inline_motifs failed: run_config_id={} err={e:?}", run_config_id);
                self.status = Some(format!("Failed to list motifs for run_config_id={run_config_id}: {e:?}"));
                false
            }
        }
    }

    fn main_page_size(&self) -> usize {
        let rows = terminal::size().map(|(_, r)| r).unwrap_or(24);
        let input_rows = if self.panel_search.input_open { 3 } else { 0 };
        rows.saturating_sub(3 + input_rows).max(1) as usize
    }

    fn run_config_batch_size(&self) -> usize {
        (self.main_page_size().max(25) * 4).clamp(100, 500)
    }

    fn merge_run_configs(&mut self, entries: Vec<RunConfigEntry>) {
        let mut existing_ids = self.all_run_configs.iter().map(|entry| entry.id).collect::<HashSet<u32>>();
        for entry in entries {
            if existing_ids.insert(entry.id) {
                self.all_run_configs.push(entry);
            }
        }
    }

    async fn load_next_run_config_page(&mut self, keep_run_config_id: Option<u32>) -> Result<bool, ThoError> {
        if !self.has_more_paged_run_configs {
            return Ok(false);
        }

        let limit = self.run_config_batch_size();
        let page = self.store.list_run_configs_page(limit, self.paged_run_config_offset).instrument(tracing::debug_span!("load_next_run_config_page/1", file = file!(), line = line!())).await?;
        if page.is_empty() {
            self.has_more_paged_run_configs = false;
            let loaded_input_pairs = self.loaded_input_pairs.clone();
            let known_output_run_config_ids = self.known_output_run_config_ids.clone();
            self.rebuild_run_config_tree(&loaded_input_pairs, &known_output_run_config_ids);
            self.rebuild_visible_run_configs(keep_run_config_id);
            return Ok(false);
        }

        let page_ids = page.iter().map(|entry| entry.id).collect::<Vec<u32>>();
        let run_config_ids_with_outputs = if page_ids.is_empty() {
            vec![]
        } else {
            self.store.list_run_config_ids_with_outputs(&page_ids).instrument(tracing::debug_span!("load_next_run_config_page/2", file = file!(), line = line!())).await?
        };
        let input_pairs = if page_ids.is_empty() {
            vec![]
        } else {
            self.store.list_input_run_config_ids(&page_ids).instrument(tracing::debug_span!("load_next_run_config_page/3", file = file!(), line = line!())).await?
        };

        self.paged_run_config_offset += page.len();
        self.has_more_paged_run_configs = page.len() == limit;
        self.merge_run_configs(page);
        self.known_run_config_ids_with_outputs.extend(run_config_ids_with_outputs);
        self.loaded_input_pairs.extend(input_pairs);
        let loaded_input_pairs = self.loaded_input_pairs.clone();
        let known_output_run_config_ids = self.known_output_run_config_ids.clone();
        self.rebuild_run_config_tree(&loaded_input_pairs, &known_output_run_config_ids);
        self.rebuild_visible_run_configs(keep_run_config_id);
        Ok(true)
    }

    async fn ensure_all_run_configs_loaded(&mut self, keep_run_config_id: Option<u32>) -> Result<(), ThoError> {
        while self.has_more_paged_run_configs {
            if !self.load_next_run_config_page(keep_run_config_id).instrument(tracing::debug_span!("ensure_all_run_configs_loaded/1", file = file!(), line = line!())).await? {
                break;
            }
        }
        Ok(())
    }

    async fn ensure_run_config_ids_loaded(&mut self, run_config_ids: &[u32], keep_run_config_id: Option<u32>) -> Result<bool, ThoError> {
        let existing_ids = self.all_run_configs.iter().map(|entry| entry.id).collect::<HashSet<u32>>();
        let missing_ids = run_config_ids
            .iter()
            .copied()
            .filter(|run_config_id| !existing_ids.contains(run_config_id))
            .collect::<Vec<u32>>();
        if missing_ids.is_empty() {
            return Ok(false);
        }

        let run_configs = self.store.list_run_configs_by_ids(&missing_ids).instrument(tracing::debug_span!("ensure_run_config_ids_loaded/1", file = file!(), line = line!())).await?;
        let input_pairs = self.store.list_input_run_config_ids(&missing_ids).instrument(tracing::debug_span!("ensure_run_config_ids_loaded/2", file = file!(), line = line!())).await?;
        self.merge_run_configs(run_configs);
        self.loaded_input_pairs.extend(input_pairs);
        let loaded_input_pairs = self.loaded_input_pairs.clone();
        let known_output_run_config_ids = self.known_output_run_config_ids.clone();
        self.rebuild_run_config_tree(&loaded_input_pairs, &known_output_run_config_ids);
        self.rebuild_visible_run_configs(keep_run_config_id);
        Ok(true)
    }

    async fn ensure_run_children_loaded(&mut self, run_config_id: u32, keep_run_config_id: Option<u32>) -> Result<bool, ThoError> {
        let output_run_config_ids = self.store.list_output_run_config_ids(run_config_id).instrument(tracing::debug_span!("ensure_run_children_loaded/1", file = file!(), line = line!())).await?;
        if output_run_config_ids.is_empty() {
            self.known_run_config_ids_with_outputs.remove(&run_config_id);
        } else {
            self.known_run_config_ids_with_outputs.insert(run_config_id);
        }
        self.known_output_run_config_ids.insert(run_config_id, output_run_config_ids.clone());
        let loaded_any = self.ensure_run_config_ids_loaded(&output_run_config_ids, keep_run_config_id).instrument(tracing::debug_span!("ensure_run_children_loaded/2", file = file!(), line = line!())).await?;
        if !loaded_any {
            let loaded_input_pairs = self.loaded_input_pairs.clone();
            let known_output_run_config_ids = self.known_output_run_config_ids.clone();
            self.rebuild_run_config_tree(&loaded_input_pairs, &known_output_run_config_ids);
            self.rebuild_visible_run_configs(keep_run_config_id);
        }
        Ok(!output_run_config_ids.is_empty())
    }

    async fn maybe_load_more_run_configs(&mut self) -> Result<bool, ThoError> {
        if self.search_filters.is_empty()
            && self.has_more_paged_run_configs
            && self.selected_idx + self.main_page_size() >= self.run_configs.len()
        {
            return self.load_next_run_config_page(self.selected_run_config_id()).instrument(tracing::debug_span!("maybe_load_more_run_configs/1", file = file!(), line = line!())).await;
        }

        Ok(false)
    }

    fn max_main_list_offset(&self) -> usize {
        self.run_configs.len().saturating_sub(self.main_page_size())
    }

    fn ensure_main_selection_visible(&mut self) {
        let page_size = self.main_page_size();
        if self.run_configs.is_empty() || page_size == 0 {
            self.main_list_offset = 0;
            return;
        }

        if self.selected_idx < self.main_list_offset {
            self.main_list_offset = self.selected_idx;
        } else if self.selected_idx >= self.main_list_offset + page_size {
            self.main_list_offset = self.selected_idx + 1 - page_size;
        }

        self.main_list_offset = self.main_list_offset.min(self.max_main_list_offset());
    }

    fn ensure_main_selection_visible_after_down(&mut self) {
        let page_size = self.main_page_size();
        if self.run_configs.is_empty() || page_size == 0 {
            self.main_list_offset = 0;
            return;
        }

        if self.selected_idx >= self.main_list_offset + page_size {
            self.main_list_offset = self.selected_idx + 1 - page_size;
        }

        self.main_list_offset = self.main_list_offset.min(self.max_main_list_offset());
    }

    fn motif_page_size(&self) -> usize {
        let rows = terminal::size().map(|(_, r)| r).unwrap_or(24) as usize;
        rows.saturating_sub(10).max(4) / 2
    }

    fn max_motif_list_offset(&self) -> usize {
        self.selected_run_motifs.len().saturating_sub(self.motif_page_size())
    }

    fn ensure_motif_selection_visible(&mut self) {
        let page_size = self.motif_page_size();
        if self.selected_run_motifs.is_empty() || page_size == 0 {
            self.motif_list_offset = 0;
            return;
        }

        if self.selected_motif_idx < self.motif_list_offset {
            self.motif_list_offset = self.selected_motif_idx;
        } else if self.selected_motif_idx >= self.motif_list_offset + page_size {
            self.motif_list_offset = self.selected_motif_idx + 1 - page_size;
        }

        self.motif_list_offset = self.motif_list_offset.min(self.max_motif_list_offset());
    }

    fn ensure_motif_selection_visible_after_down(&mut self) {
        let page_size = self.motif_page_size();
        if self.selected_run_motifs.is_empty() || page_size == 0 {
            self.motif_list_offset = 0;
            return;
        }

        if self.selected_motif_idx >= self.motif_list_offset + page_size {
            self.motif_list_offset = self.selected_motif_idx + 1 - page_size;
        }

        self.motif_list_offset = self.motif_list_offset.min(self.max_motif_list_offset());
    }

    fn popup_page_size(&self) -> usize {
        let rows = terminal::size().map(|(_, r)| r).unwrap_or(24);
        popup_height(rows, self.run_config_types.len()).saturating_sub(2).max(1) as usize
    }

    fn search_type_popup_page_size(&self) -> usize {
        let rows = terminal::size().map(|(_, r)| r).unwrap_or(24);
        popup_height(rows, self.run_config_types.len() + 1).saturating_sub(2).max(1) as usize
    }

    fn view_popup_page_size(&self) -> usize {
        let rows = terminal::size().map(|(_, r)| r).unwrap_or(24);
        popup_height(rows, ViewMode::ALL.len()).saturating_sub(2).max(1) as usize
    }

    fn move_main_down(&mut self, step: usize) -> bool {
        let moved = move_down(&mut self.selected_idx, self.run_configs.len(), step);
        if moved {
            self.ensure_main_selection_visible_after_down();
        }
        moved
    }

    fn move_main_up(&mut self, step: usize) -> bool {
        if self.run_configs.is_empty() {
            return false;
        }

        let old_selected = self.selected_idx;
        let old_offset = self.main_list_offset;
        let moved = move_up(&mut self.selected_idx, self.run_configs.len(), step);
        if !moved {
            return false;
        }

        if step == 1 {
            if old_selected == old_offset {
                self.main_list_offset = self.main_list_offset.saturating_sub(1);
            }
        } else {
            self.ensure_main_selection_visible();
        }

        true
    }

    fn move_main_to_top(&mut self) -> bool {
        let moved = move_to_top(&mut self.selected_idx, self.run_configs.len());
        if moved {
            self.main_list_offset = 0;
        }
        moved
    }

    fn move_main_to_bottom(&mut self) -> bool {
        let moved = move_to_bottom(&mut self.selected_idx, self.run_configs.len());
        if moved {
            self.main_list_offset = self.max_main_list_offset();
        }
        moved
    }

    async fn move_main_down_lazy(&mut self, step: usize) -> bool {
        if self.move_main_down(step) {
            if let Err(err) = self.maybe_load_more_run_configs().instrument(tracing::debug_span!("move_main_down_lazy/1", file = file!(), line = line!())).await {
                warn!("move_main_down_lazy failed to load more run configs: {err:?}");
                self.status = Some(format!("Failed to load more run_configs: {err:?}"));
            }
            return true;
        }

        match self.maybe_load_more_run_configs().instrument(tracing::debug_span!("move_main_down_lazy/2", file = file!(), line = line!())).await {
            Ok(true) => self.move_main_down(step),
            Ok(false) => false,
            Err(err) => {
                warn!("move_main_down_lazy failed to load more run configs: {err:?}");
                self.status = Some(format!("Failed to load more run_configs: {err:?}"));
                false
            }
        }
    }

    async fn move_main_to_bottom_lazy(&mut self) -> bool {
        if self.search_filters.is_empty() && self.has_more_paged_run_configs {
            if let Err(err) = self.ensure_all_run_configs_loaded(self.selected_run_config_id()).instrument(tracing::debug_span!("move_main_to_bottom_lazy/1", file = file!(), line = line!())).await {
                warn!("move_main_to_bottom_lazy failed to load all run configs: {err:?}");
                self.status = Some(format!("Failed to load all run_configs: {err:?}"));
                return false;
            }
        }

        self.move_main_to_bottom()
    }

    fn move_motif_down(&mut self, step: usize) -> bool {
        let moved = move_down(&mut self.selected_motif_idx, self.selected_run_motifs.len(), step);
        if moved {
            self.ensure_motif_selection_visible_after_down();
        }
        moved
    }

    fn move_motif_up(&mut self, step: usize) -> bool {
        if self.selected_run_motifs.is_empty() {
            return false;
        }

        let old_selected = self.selected_motif_idx;
        let old_offset = self.motif_list_offset;
        let moved = move_up(&mut self.selected_motif_idx, self.selected_run_motifs.len(), step);
        if !moved {
            return false;
        }

        if step == 1 {
            if old_selected == old_offset {
                self.motif_list_offset = self.motif_list_offset.saturating_sub(1);
            }
        } else {
            self.ensure_motif_selection_visible();
        }

        true
    }

    fn move_motif_to_top(&mut self) -> bool {
        let moved = move_to_top(&mut self.selected_motif_idx, self.selected_run_motifs.len());
        if moved {
            self.motif_list_offset = 0;
        }
        moved
    }

    fn move_motif_to_bottom(&mut self) -> bool {
        let moved = move_to_bottom(&mut self.selected_motif_idx, self.selected_run_motifs.len());
        if moved {
            self.motif_list_offset = self.max_motif_list_offset();
        }
        moved
    }

    fn move_popup_down(&mut self, step: usize) -> bool {
        move_down(&mut self.type_popup_selected_idx, self.run_config_types.len(), step)
    }

    fn move_popup_up(&mut self, step: usize) -> bool {
        move_up(&mut self.type_popup_selected_idx, self.run_config_types.len(), step)
    }

    fn move_popup_to_top(&mut self) -> bool {
        move_to_top(&mut self.type_popup_selected_idx, self.run_config_types.len())
    }

    fn move_popup_to_bottom(&mut self) -> bool {
        move_to_bottom(&mut self.type_popup_selected_idx, self.run_config_types.len())
    }

    fn move_search_popup_down(&mut self, step: usize) -> bool {
        move_down(&mut self.search_popup.selected_field_idx, SearchPopupRow::count(), step)
    }

    fn move_search_popup_up(&mut self, step: usize) -> bool {
        move_up(&mut self.search_popup.selected_field_idx, SearchPopupRow::count(), step)
    }

    fn move_search_popup_to_top(&mut self) -> bool {
        move_to_top(&mut self.search_popup.selected_field_idx, SearchPopupRow::count())
    }

    fn move_search_popup_to_bottom(&mut self) -> bool {
        move_to_bottom(&mut self.search_popup.selected_field_idx, SearchPopupRow::count())
    }

    fn move_search_type_popup_down(&mut self, step: usize) -> bool {
        move_down(&mut self.search_popup.type_selected_idx, self.run_config_types.len() + 1, step)
    }

    fn move_search_type_popup_up(&mut self, step: usize) -> bool {
        move_up(&mut self.search_popup.type_selected_idx, self.run_config_types.len() + 1, step)
    }

    fn move_search_type_popup_to_top(&mut self) -> bool {
        move_to_top(&mut self.search_popup.type_selected_idx, self.run_config_types.len() + 1)
    }

    fn move_search_type_popup_to_bottom(&mut self) -> bool {
        move_to_bottom(&mut self.search_popup.type_selected_idx, self.run_config_types.len() + 1)
    }

    fn move_view_popup_down(&mut self, step: usize) -> bool {
        move_down(&mut self.view_popup_selected_idx, ViewMode::ALL.len(), step)
    }

    fn move_view_popup_up(&mut self, step: usize) -> bool {
        move_up(&mut self.view_popup_selected_idx, ViewMode::ALL.len(), step)
    }

    fn move_view_popup_to_top(&mut self) -> bool {
        move_to_top(&mut self.view_popup_selected_idx, ViewMode::ALL.len())
    }

    fn move_view_popup_to_bottom(&mut self) -> bool {
        move_to_bottom(&mut self.view_popup_selected_idx, ViewMode::ALL.len())
    }

    fn clear_pending_gg(&mut self) {
        self.pending_gg = None;
    }

    fn expire_pending_gg_if_needed(&mut self) {
        if let Some((_, when)) = self.pending_gg {
            if when.elapsed() > Duration::from_millis(750) {
                self.pending_gg = None;
            }
        }
    }

    fn handle_g_press(&mut self, area: SelectedArea) -> bool {
        if let Some((pending_area, when)) = self.pending_gg {
            if pending_area == area && when.elapsed() <= Duration::from_millis(750) {
                self.pending_gg = None;
                return true;
            }
        }
        self.pending_gg = Some((area, Instant::now()));
        false
    }

    fn sync_search_popup_type_selection(&mut self) {
        self.search_popup.type_selected_idx = self
            .search_popup
            .draft
            .run_config_type
            .as_ref()
            .and_then(|selected| self.run_config_types.iter().position(|value| value == selected))
            .map(|idx| idx + 1)
            .unwrap_or(0);
    }

    async fn refresh_run_config_types(&mut self) -> Result<(), ThoError> {
        self.run_config_types = self.store.list_run_config_types().instrument(tracing::debug_span!("refresh_run_config_types/1", file = file!(), line = line!())).await?;
        Ok(())
    }

    async fn open_filter_popup(&mut self) {
        if let Err(err) = self.refresh_run_config_types().instrument(tracing::debug_span!("open_filter_popup/1", file = file!(), line = line!())).await {
            warn!("open_filter_popup failed to refresh run_config types: {err:?}");
            self.status = Some(format!("Failed to list run_config types: {err:?}"));
        }

        self.search_popup = SearchPopupState {
            draft: self.search_filters.clone(),
            selected_field_idx: 0,
            type_selected_idx: 0,
        };
        self.sync_search_popup_type_selection();
        self.selected_area = SelectedArea::SearchPopup;
    }

    fn apply_search_type_from_popup(&mut self) {
        self.search_popup.draft.run_config_type = self.selected_search_popup_type();
        self.selected_area = SelectedArea::SearchPopup;
    }

    fn open_search_type_popup(&mut self) {
        self.sync_search_popup_type_selection();
        self.selected_area = SelectedArea::SearchTypePopup;
    }

    fn reset_search_popup_draft(&mut self) {
        self.search_popup.draft = RunConfigSearch::default();
        self.search_popup.type_selected_idx = 0;
        self.status = Some("Filter draft reset.".to_string());
    }

    async fn apply_search_from_popup(&mut self) -> bool {
        match self.search_popup.draft.normalized() {
            Ok(_) => {
                let keep_run_config_id = self.selected_run_config_id();
                self.search_filters = self.search_popup.draft.clone();
                if let Err(err) = self.try_refresh_run_configs(keep_run_config_id).instrument(tracing::debug_span!("apply_search_from_popup/1", file = file!(), line = line!())).await {
                    warn!("apply_search_from_popup failed to refresh run configs: {err:?}");
                    self.status = Some(format!("Failed to list run_configs for filter: {err:?}"));
                    return false;
                }
                self.selected_area = SelectedArea::LeftRunConfigs;
                self.prefer_right_motifs = false;
                self.status = Some(format!("Run config filters: {}", self.search_filters.summary()));
                true
            }
            Err(err) => {
                self.status = Some(format!("Invalid filter: {err}"));
                false
            }
        }
    }

    fn append_to_search_popup(&mut self, ch: char) -> bool {
        match self.selected_search_field() {
            Some(SearchField::Id) => {
                if ch.is_ascii_digit() {
                    self.search_popup.draft.id.push(ch);
                    true
                } else {
                    false
                }
            }
            Some(SearchField::Type) | None => false,
            Some(SearchField::Descr) => {
                self.search_popup.draft.descr.push(ch);
                true
            }
            Some(SearchField::RunOnFrom) => append_run_on_char(&mut self.search_popup.draft.run_on_from, ch),
            Some(SearchField::RunOnTo) => append_run_on_char(&mut self.search_popup.draft.run_on_to, ch),
        }
    }

    fn pop_from_search_popup(&mut self) -> bool {
        match self.selected_search_field() {
            Some(SearchField::Id) => self.search_popup.draft.id.pop().is_some(),
            Some(SearchField::Type) => {
                let had_value = self.search_popup.draft.run_config_type.is_some();
                self.search_popup.draft.run_config_type = None;
                self.search_popup.type_selected_idx = 0;
                had_value
            }
            Some(SearchField::Descr) => self.search_popup.draft.descr.pop().is_some(),
            Some(SearchField::RunOnFrom) => self.search_popup.draft.run_on_from.pop().is_some(),
            Some(SearchField::RunOnTo) => self.search_popup.draft.run_on_to.pop().is_some(),
            None => false,
        }
    }

    fn open_view_popup(&mut self) {
        self.view_popup_selected_idx = match self.view_mode {
            ViewMode::RunConfigs => 0,
            ViewMode::Progress => 1,
        };
        self.selected_area = SelectedArea::ViewPopup;
    }

    fn apply_view_mode_from_popup(&mut self) {
        self.view_mode = self.selected_view_mode();
        self.selected_area = match self.view_mode {
            ViewMode::RunConfigs => self.default_run_configs_area(),
            ViewMode::Progress => SelectedArea::ProgressView,
        };
        debug!("apply_view_mode_from_popup: mode={}", self.view_mode.label());
    }

    fn apply_progress_event(&mut self, event: ProgressClientEvent) {
        match event {
            ProgressClientEvent::Snapshot(snapshot) => {
                self.progress_state = snapshot;
                self.progress_status = "Connected to progress socket.".to_string();
                self.progress_last_update = Some(Instant::now());
            }
            ProgressClientEvent::Status(status) => {
                self.progress_status = status;
            }
        }
    }

    fn progress_lines(&self) -> Vec<String> {
        let mut lines = vec![format!("socket: {}", self.progress_status)];
        lines.push(format!("uptime: {}", format_progress_uptime(self.progress_state.started_at_unix_ms)));
        lines.push(format!("success OOM runs: {}", self.progress_state.oom_success));
        lines.push(format!("failed OOM runs: {}", self.progress_state.oom_all_failed));
        match self.progress_last_update {
            Some(last_update) => {
                lines.push(format!("last update: {}ms ago", last_update.elapsed().as_millis()));
            }
            None => {
                lines.push("last update: <none>".to_string());
            }
        }
        match self.progress_state.device_stats {
            Some(device_stats) => {
                lines.extend(format_progress_device_stats(device_stats));
            }
            None => {
                lines.push("gpu stats: <unavailable>".to_string());
            }
        }
        lines.push(String::new());

        if self.progress_state.invocations.is_empty() {
            lines.push("No active progress invocations.".to_string());
        } else {
            for invocation in &self.progress_state.invocations {
                lines.push(format_progress_invocation(invocation));
            }
        }

        lines
    }

    fn rebuild_run_config_tree(&mut self, input_pairs: &[(u32, u32)], outputs_by_run_config_id: &HashMap<u32, Vec<u32>>) {
        let old_expanded = self
            .run_config_tree
            .iter()
            .map(|(run_config_id, node)| (*run_config_id, node.expanded))
            .collect::<HashMap<u32, bool>>();
        let run_on_by_id = self
            .all_run_configs
            .iter()
            .map(|entry| (entry.id, entry.run_on.as_str()))
            .collect::<HashMap<u32, &str>>();
        let entries_by_id = self
            .all_run_configs
            .iter()
            .cloned()
            .map(|entry| (entry.id, entry))
            .collect::<HashMap<u32, RunConfigEntry>>();
        let mut input_map = HashMap::<u32, HashSet<u32>>::new();
        let mut output_map = HashMap::<u32, HashSet<u32>>::new();

        for (run_config_id, input_run_config_id) in input_pairs {
            if entries_by_id.contains_key(run_config_id) && entries_by_id.contains_key(input_run_config_id) {
                input_map.entry(*run_config_id).or_default().insert(*input_run_config_id);
                output_map.entry(*input_run_config_id).or_default().insert(*run_config_id);
            }
        }

        for (run_config_id, output_run_config_ids) in outputs_by_run_config_id {
            if !entries_by_id.contains_key(run_config_id) {
                continue;
            }

            let outputs = output_map.entry(*run_config_id).or_default();
            for output_run_config_id in output_run_config_ids {
                if entries_by_id.contains_key(output_run_config_id) {
                    outputs.insert(*output_run_config_id);
                }
            }
        }

        self.run_config_tree = entries_by_id
            .into_iter()
            .map(|(run_config_id, entry)| {
                let mut input_run_config_ids = input_map.remove(&run_config_id).unwrap_or_default().into_iter().collect::<Vec<u32>>();
                let mut output_run_config_ids = output_map.remove(&run_config_id).unwrap_or_default().into_iter().collect::<Vec<u32>>();
                let children_loaded = outputs_by_run_config_id.contains_key(&run_config_id)
                    || !self.has_more_paged_run_configs
                    || (output_run_config_ids.is_empty() && !self.known_run_config_ids_with_outputs.contains(&run_config_id));
                let default_expanded = !output_run_config_ids.is_empty();
                sort_run_config_ids(&mut input_run_config_ids, &run_on_by_id, self.run_on_sort_order);
                sort_run_config_ids(&mut output_run_config_ids, &run_on_by_id, self.run_on_sort_order);
                (
                    run_config_id,
                    RunTreeNode {
                        entry,
                        input_run_config_ids,
                        output_run_config_ids,
                        children_loaded,
                        expanded: old_expanded.get(&run_config_id).copied().unwrap_or(default_expanded),
                    },
                )
            })
            .collect();
    }

    fn run_visible_with_descendants(
        &self,
        run_config_id: u32,
        filter_active: bool,
        matching_run_config_ids: &HashSet<u32>,
        memo: &mut HashMap<u32, bool>,
        path: &mut HashSet<u32>,
    ) -> bool {
        if let Some(visible) = memo.get(&run_config_id) {
            return *visible;
        }

        if !path.insert(run_config_id) {
            return false;
        }

        let visible = !filter_active
            || matching_run_config_ids.contains(&run_config_id)
            || self
                .run_config_tree
                .get(&run_config_id)
                .map(|node| {
                    node.output_run_config_ids.iter().any(|child_run_config_id| {
                        self.run_visible_with_descendants(*child_run_config_id, filter_active, matching_run_config_ids, memo, path)
                    })
                })
                .unwrap_or(false);
        path.remove(&run_config_id);
        memo.insert(run_config_id, visible);
        visible
    }

    fn append_visible_run_rows(
        &self,
        run_config_id: u32,
        depth: usize,
        filter_active: bool,
        matching_run_config_ids: &HashSet<u32>,
        memo: &mut HashMap<u32, bool>,
        path: &mut HashSet<u32>,
        mut appended_run_config_ids: Option<&mut HashSet<u32>>,
        rows: &mut Vec<VisibleRunConfigRow>,
    ) {
        let already_appended = appended_run_config_ids
            .as_ref()
            .map(|appended_run_config_ids| appended_run_config_ids.contains(&run_config_id))
            .unwrap_or(false);
        if already_appended || !path.insert(run_config_id) {
            return;
        }

        let Some(node) = self.run_config_tree.get(&run_config_id) else {
            path.remove(&run_config_id);
            return;
        };
        let visible_children = node
            .output_run_config_ids
            .iter()
            .filter(|child_run_config_id| self.run_visible_with_descendants(**child_run_config_id, filter_active, matching_run_config_ids, memo, path))
            .copied()
            .collect::<Vec<u32>>();
        let inline_motifs = if node.expanded && run_config_type_supports_inline_motif_rows(node.entry.run_config_type.as_str()) {
            self.run_motifs_cache.get(&run_config_id).cloned().unwrap_or_default()
        } else {
            vec![]
        };
        let has_inline_motif_rows = run_config_type_supports_inline_motif_rows(node.entry.run_config_type.as_str())
            && self.run_motifs_cache.get(&run_config_id).map(|motifs| !motifs.is_empty()).unwrap_or(true);

        rows.push(VisibleRunConfigRow::run_config(
            run_config_id,
            depth,
            !visible_children.is_empty() || !node.children_loaded || has_inline_motif_rows,
            node.expanded,
        ));
        if let Some(appended_run_config_ids) = appended_run_config_ids.as_mut() {
            appended_run_config_ids.insert(run_config_id);
        }

        if node.expanded {
            for motif in inline_motifs {
                rows.push(VisibleRunConfigRow::motif(run_config_id, motif, depth + 1));
            }
            for child_run_config_id in visible_children {
                self.append_visible_run_rows(
                    child_run_config_id,
                    depth + 1,
                    filter_active,
                    matching_run_config_ids,
                    memo,
                    path,
                    appended_run_config_ids.as_deref_mut(),
                    rows,
                );
            }
        }

        path.remove(&run_config_id);
    }

    fn group_visible_run_rows_by_date(&self, rows: Vec<VisibleRunConfigRow>) -> Vec<VisibleRunConfigRow> {
        let mut date_order = vec![];
        let mut rows_by_date = HashMap::<String, Vec<VisibleRunConfigRow>>::new();
        let mut current_run_on_date = None::<String>;
        let mut current_block = vec![];

        for row in rows {
            if row.depth == 0 {
                if let Some(run_on_date) = current_run_on_date.take() {
                    rows_by_date.entry(run_on_date).or_default().append(&mut current_block);
                }

                let run_on_date = row
                    .run_config_id()
                    .and_then(|run_config_id| self.run_config_tree.get(&run_config_id))
                    .map(|node| run_on_date_label(node.entry.run_on.as_str()))
                    .unwrap_or_else(|| "<unknown>".to_string());
                if !rows_by_date.contains_key(&run_on_date) {
                    date_order.push(run_on_date.clone());
                }
                current_run_on_date = Some(run_on_date);
            }
            current_block.push(row);
        }

        if let Some(run_on_date) = current_run_on_date {
            rows_by_date.entry(run_on_date).or_default().append(&mut current_block);
        }

        let mut grouped_rows = vec![];
        for run_on_date in date_order {
            let expanded = self.run_on_date_expanded(run_on_date.as_str());
            grouped_rows.push(VisibleRunConfigRow::run_on_date_group(run_on_date.clone(), expanded));
            if expanded {
                let mut child_rows = rows_by_date.remove(&run_on_date).unwrap_or_default();
                for row in &mut child_rows {
                    row.depth += 1;
                }
                grouped_rows.extend(child_rows);
            }
        }

        grouped_rows
    }

    fn rebuild_visible_run_configs(&mut self, keep_run_config_id: Option<u32>) {
        let keep_run_on_date = if keep_run_config_id.is_none() {
            self.selected_run_on_date().map(|run_on_date| run_on_date.to_string())
        } else {
            None
        };
        let filter_active = !self.search_filters.is_empty();
        let run_on_by_id = self
            .all_run_configs
            .iter()
            .map(|entry| (entry.id, entry.run_on.as_str()))
            .collect::<HashMap<u32, &str>>();
        let matching_run_config_ids = if filter_active {
            self.search_filters
                .normalized()
                .ok()
                .map(|active_search| {
                    self.run_config_tree
                        .values()
                        .filter(|node| active_search.matches(&node.entry))
                        .map(|node| node.entry.id)
                        .collect::<HashSet<u32>>()
                })
                .unwrap_or_default()
        } else {
            HashSet::new()
        };

        let mut root_run_config_ids = self
            .run_config_tree
            .iter()
            .filter_map(|(run_config_id, node)| {
                if node
                    .input_run_config_ids
                    .iter()
                    .all(|input_run_config_id| !self.run_config_tree.contains_key(input_run_config_id))
                {
                    Some(*run_config_id)
                } else {
                    None
                }
            })
            .collect::<Vec<u32>>();
        sort_run_config_ids(&mut root_run_config_ids, &run_on_by_id, self.run_on_sort_order);

        let mut memo = HashMap::new();
        let mut rows = vec![];
        let mut dedup_run_config_ids = filter_active.then(HashSet::new);
        for run_config_id in root_run_config_ids {
            if self.run_visible_with_descendants(run_config_id, filter_active, &matching_run_config_ids, &mut memo, &mut HashSet::new()) {
                self.append_visible_run_rows(
                    run_config_id,
                    0,
                    filter_active,
                    &matching_run_config_ids,
                    &mut memo,
                    &mut HashSet::new(),
                    dedup_run_config_ids.as_mut(),
                    &mut rows,
                );
            }
        }

        let appended_run_config_ids = dedup_run_config_ids.clone().unwrap_or_else(|| {
            rows.iter()
                .filter_map(VisibleRunConfigRow::run_config_id)
                .collect::<HashSet<u32>>()
        });
        let mut remaining_run_config_ids = self
            .run_config_tree
            .keys()
            .filter(|run_config_id| !appended_run_config_ids.contains(run_config_id))
            .copied()
            .collect::<Vec<u32>>();
        sort_run_config_ids(&mut remaining_run_config_ids, &run_on_by_id, self.run_on_sort_order);
        for run_config_id in remaining_run_config_ids {
            if self.run_visible_with_descendants(run_config_id, filter_active, &matching_run_config_ids, &mut memo, &mut HashSet::new()) {
                self.append_visible_run_rows(
                    run_config_id,
                    0,
                    filter_active,
                    &matching_run_config_ids,
                    &mut memo,
                    &mut HashSet::new(),
                    dedup_run_config_ids.as_mut(),
                    &mut rows,
                );
            }
        }

        self.run_configs = self.group_visible_run_rows_by_date(rows);
        if self.panel_search.target == Some(PanelSearchTarget::RunConfigs) {
            self.panel_search.current_match_idx = None;
        }

        if let Some(run_config_id) = keep_run_config_id {
            if let Some(idx) = self.run_configs.iter().position(|entry| entry.run_config_id() == Some(run_config_id)) {
                self.selected_idx = idx;
                self.ensure_main_selection_visible();
                return;
            }
        }

        if let Some(run_on_date) = keep_run_on_date {
            if let Some(idx) = self.run_configs.iter().position(|entry| entry.run_on_date() == Some(run_on_date.as_str())) {
                self.selected_idx = idx;
                self.ensure_main_selection_visible();
                return;
            }
        }

        if self.run_configs.is_empty() {
            self.selected_idx = 0;
            self.main_list_offset = 0;
        } else if self.selected_idx >= self.run_configs.len() {
            self.selected_idx = self.run_configs.len() - 1;
        }

        self.ensure_main_selection_visible();
    }

    async fn toggle_selected_run_expanded(&mut self) -> bool {
        if let Some(run_on_date) = self.selected_run_on_date().map(|run_on_date| run_on_date.to_string()) {
            if self.run_on_date_expanded(run_on_date.as_str()) {
                self.collapsed_run_on_dates.insert(run_on_date);
            } else {
                self.collapsed_run_on_dates.remove(run_on_date.as_str());
            }
            self.rebuild_visible_run_configs(None);
            return true;
        }

        let Some(run_config_id) = self.selected_run_config_id() else {
            return false;
        };
        let needs_children_load = self
            .run_config_tree
            .get(&run_config_id)
            .map(|node| !node.children_loaded)
            .unwrap_or(false);
        if needs_children_load {
            match self.ensure_run_children_loaded(run_config_id, Some(run_config_id)).instrument(tracing::debug_span!("toggle_selected_run_expanded/1", file = file!(), line = line!())).await {
                Ok(false) => {
                    self.status = Some(format!("No child runs found for run_config_id={run_config_id}."));
                    return false;
                }
                Ok(true) => {}
                Err(e) => {
                    warn!("toggle_selected_run_expanded failed to lazy-load children: run_config_id={run_config_id} err={e:?}");
                    self.status = Some(format!("Failed to list child runs for run_config_id={run_config_id}: {e:?}"));
                    return false;
                }
            }
        }

        let Some(node) = self.run_config_tree.get_mut(&run_config_id) else {
            return false;
        };
        if node.output_run_config_ids.is_empty() {
            return false;
        }
        node.expanded = !node.expanded;
        self.rebuild_visible_run_configs(Some(run_config_id));
        true
    }

    async fn expand_selected_run(&mut self) -> bool {
        if let Some(run_on_date) = self.selected_run_on_date().map(|run_on_date| run_on_date.to_string()) {
            if self.run_on_date_expanded(run_on_date.as_str()) {
                return false;
            }
            self.collapsed_run_on_dates.remove(run_on_date.as_str());
            self.rebuild_visible_run_configs(None);
            return true;
        }

        let Some(run_config_id) = self.selected_run_config_id() else {
            return false;
        };
        let needs_children_load = self
            .run_config_tree
            .get(&run_config_id)
            .map(|node| !node.children_loaded)
            .unwrap_or(false);
        if needs_children_load {
            match self.ensure_run_children_loaded(run_config_id, Some(run_config_id)).instrument(tracing::debug_span!("expand_selected_run/1", file = file!(), line = line!())).await {
                Ok(false) => {
                    self.status = Some(format!("No child runs found for run_config_id={run_config_id}."));
                    return false;
                }
                Ok(true) => {}
                Err(e) => {
                    warn!("expand_selected_run failed to lazy-load children: run_config_id={run_config_id} err={e:?}");
                    self.status = Some(format!("Failed to list child runs for run_config_id={run_config_id}: {e:?}"));
                    return false;
                }
            }
        }

        let Some(node) = self.run_config_tree.get_mut(&run_config_id) else {
            return false;
        };
        if node.output_run_config_ids.is_empty() || node.expanded {
            return false;
        }
        node.expanded = true;
        self.rebuild_visible_run_configs(Some(run_config_id));
        true
    }

    async fn open_selected_run_from_left(&mut self) -> bool {
        if self.toggle_selected_run_expanded().instrument(tracing::debug_span!("open_selected_run_from_left/1", file = file!(), line = line!())).await {
            return true;
        }

        self.toggle_selected_run_inline_motifs().instrument(tracing::debug_span!("open_selected_run_from_left/2", file = file!(), line = line!())).await
    }

    async fn open_selected_run_right(&mut self) -> bool {
        if self.expand_selected_run().instrument(tracing::debug_span!("open_selected_run_right/1", file = file!(), line = line!())).await {
            return true;
        }

        self.expand_selected_run_inline_motifs().instrument(tracing::debug_span!("open_selected_run_right/2", file = file!(), line = line!())).await
    }

    async fn open_selected_run_motifs(&mut self) -> bool {
        if self.load_selected_run_motifs().instrument(tracing::debug_span!("open_selected_run_motifs/1", file = file!(), line = line!())).await {
            self.show_selected_run_motifs = true;
            self.selected_area = SelectedArea::RightMotifs;
            self.prefer_right_motifs = true;
            return true;
        }

        false
    }

    async fn toggle_selected_run_motifs_panel(&mut self) -> bool {
        if self.motif_panel_visible() {
            self.show_selected_run_motifs = false;
            if self.selected_area == SelectedArea::RightMotifs {
                self.selected_area = SelectedArea::LeftRunConfigs;
            }
            self.prefer_right_motifs = false;
            self.status = Some("Motif list hidden.".to_string());
            return true;
        }

        self.open_selected_run_motifs().instrument(tracing::debug_span!("toggle_selected_run_motifs_panel/1", file = file!(), line = line!())).await
    }

    fn collapse_selected_run(&mut self) -> bool {
        if let Some(run_on_date) = self.selected_run_on_date().map(|run_on_date| run_on_date.to_string()) {
            if !self.run_on_date_expanded(run_on_date.as_str()) {
                return false;
            }
            self.collapsed_run_on_dates.insert(run_on_date);
            self.rebuild_visible_run_configs(None);
            return true;
        }

        let Some(run_config_id) = self.selected_run_config_id() else {
            return false;
        };
        let Some(node) = self.run_config_tree.get_mut(&run_config_id) else {
            return false;
        };
        if node.output_run_config_ids.is_empty() || !node.expanded {
            return false;
        }
        node.expanded = false;
        self.rebuild_visible_run_configs(Some(run_config_id));
        true
    }

    fn toggle_run_on_sort_order(&mut self) {
        self.run_on_sort_order = match self.run_on_sort_order {
            SortOrder::Asc => SortOrder::Desc,
            SortOrder::Desc => SortOrder::Asc,
        };

        let keep_run_config_id = self.selected_run_config_id();
        self.rebuild_visible_run_configs(keep_run_config_id);
        debug!("toggle_run_on_sort_order: order={}", self.sort_order_label());
    }

    async fn reset_search_filters(&mut self) {
        if !self.search_filters.is_empty() {
            let keep_run_config_id = self.selected_run_config_id();
            self.search_filters = RunConfigSearch::default();
            self.search_popup = SearchPopupState::default();
            match self.try_refresh_run_configs(keep_run_config_id).instrument(tracing::debug_span!("reset_search_filters/1", file = file!(), line = line!())).await {
                Ok(()) => {
                    self.status = Some("Run config filters reset.".to_string());
                    debug!("reset_search_filters");
                }
                Err(err) => {
                    warn!("reset_search_filters failed: {err:?}");
                    self.status = Some(format!("Failed to reset run_config filters: {err:?}"));
                }
            }
        }
    }

    async fn apply_type_filter_from_popup(&mut self) {
        if let Some(selected_type) = self.selected_popup_type() {
            let keep_run_config_id = self.selected_run_config_id();
            self.search_filters.run_config_type = Some(selected_type.clone());
            if let Err(err) = self.try_refresh_run_configs(keep_run_config_id).instrument(tracing::debug_span!("apply_type_filter_from_popup/1", file = file!(), line = line!())).await {
                warn!("apply_type_filter_from_popup failed to refresh run configs: {err:?}");
                self.status = Some(format!("Failed to list run_configs for type filter: {err:?}"));
                return;
            }
            self.status = Some(format!("Type filter applied: {selected_type}"));
            debug!("apply_type_filter_from_popup: type={selected_type}");
        }
    }

    async fn open_type_filter_popup(&mut self) {
        match self.refresh_run_config_types().instrument(tracing::debug_span!("open_type_filter_popup/1", file = file!(), line = line!())).await {
            Ok(()) => {
                if self.run_config_types.is_empty() {
                    self.status = Some("No run_config types found.".to_string());
                    return;
                }

                self.type_popup_selected_idx = self
                    .search_filters
                    .run_config_type
                    .as_ref()
                    .and_then(|active| self.run_config_types.iter().position(|t| t == active))
                    .unwrap_or(0);

                self.selected_area = SelectedArea::TypeFilterPopup;
                self.status = None;
                debug!("open_type_filter_popup: count={}", self.run_config_types.len());
            }
            Err(err) => {
                warn!("open_type_filter_popup failed: {err:?}");
                self.status = Some(format!("Failed to list run_config types: {err:?}"));
            }
        }
    }

    async fn load_filtered_run_configs(&mut self, keep_run_config_id: Option<u32>) -> Result<(), ThoError> {
        let normalized = self.search_filters.normalized().map_err(ThoError::new)?;
        let run_configs = self
            .store
            .list_run_configs_filtered(RunConfigFilter {
                id: normalized.id,
                run_config_type: normalized.run_config_type,
                descr_substr: if normalized.descr_lower.is_empty() { None } else { Some(normalized.descr_lower) },
                run_on_from: normalized.run_on_from,
                run_on_to: normalized.run_on_to,
            })
            .instrument(tracing::debug_span!("load_filtered_run_configs/1", file = file!(), line = line!())).await?;
        let run_config_ids = run_configs.iter().map(|entry| entry.id).collect::<Vec<u32>>();
        let input_pairs = if run_config_ids.is_empty() {
            vec![]
        } else {
            self.store.list_input_run_config_ids(run_config_ids.as_slice()).instrument(tracing::debug_span!("load_filtered_run_configs/2", file = file!(), line = line!())).await?
        };

        self.all_run_configs = run_configs;
        self.loaded_input_pairs = input_pairs;
        self.known_output_run_config_ids.clear();
        self.known_run_config_ids_with_outputs.clear();
        self.paged_run_config_offset = self.all_run_configs.len();
        self.has_more_paged_run_configs = false;
        let loaded_input_pairs = self.loaded_input_pairs.clone();
        let known_output_run_config_ids = self.known_output_run_config_ids.clone();
        self.rebuild_run_config_tree(loaded_input_pairs.as_slice(), &known_output_run_config_ids);
        self.rebuild_visible_run_configs(keep_run_config_id);
        Ok(())
    }

    async fn try_refresh_run_configs(&mut self, keep_run_config_id: Option<u32>) -> Result<(), ThoError> {
        self.all_run_configs.clear();
        self.loaded_input_pairs.clear();
        self.known_output_run_config_ids.clear();
        self.known_run_config_ids_with_outputs.clear();
        self.paged_run_config_offset = 0;
        self.has_more_paged_run_configs = true;
        self.run_config_tree.clear();
        self.run_configs.clear();

        if self.search_filters.is_empty() {
            self.load_next_run_config_page(keep_run_config_id).instrument(tracing::debug_span!("try_refresh_run_configs/1", file = file!(), line = line!())).await.map(|_| ())
        } else {
            self.load_filtered_run_configs(keep_run_config_id).instrument(tracing::debug_span!("try_refresh_run_configs/2", file = file!(), line = line!())).await
        }
    }

    async fn refresh_run_configs(&mut self) {
        let keep_run_config_id = self.selected_run_config_id();
        match self.try_refresh_run_configs(keep_run_config_id).instrument(tracing::debug_span!("refresh_run_configs/1", file = file!(), line = line!())).await {
            Ok(()) => {
                self.status = None;
                debug!(
                    "refresh_run_configs: loaded total={} visible={} paged_offset={} more_pages={} filters={}",
                    self.all_run_configs.len(),
                    self.run_configs.len(),
                    self.paged_run_config_offset,
                    self.has_more_paged_run_configs,
                    self.search_filters.summary()
                );
            }
            Err(e) => {
                warn!("refresh_run_configs failed: {e:?}");
                self.all_run_configs.clear();
                self.loaded_input_pairs.clear();
                self.known_output_run_config_ids.clear();
                self.known_run_config_ids_with_outputs.clear();
                self.paged_run_config_offset = 0;
                self.has_more_paged_run_configs = false;
                self.run_config_tree.clear();
                self.run_configs.clear();
                self.selected_idx = 0;
                self.selected_values.clear();
                self.selected_run_motifs.clear();
                self.status = Some(format!("Failed to list run_configs: {e:?}"));
            }
        }
    }

    async fn refresh_selected_details(&mut self) {
        self.selected_values.clear();
        self.selected_run_motifs.clear();
        self.selected_motif_idx = 0;
        self.motif_list_offset = 0;

        let Some(selected_row) = self.selected_tree_row() else {
            if self.selected_area == SelectedArea::RightMotifs {
                self.selected_area = SelectedArea::LeftRunConfigs;
                self.prefer_right_motifs = false;
            }
            if self.status.is_none() {
                self.status = Some("No run_configs found.".to_string());
            }
            return;
        };

        if selected_row.tree_motif().is_some() {
            if self.selected_area == SelectedArea::RightMotifs {
                self.selected_area = SelectedArea::LeftRunConfigs;
                self.prefer_right_motifs = false;
            }
            self.status = None;
            return;
        }

        let Some(run_config_id) = self.selected_run_config_id() else {
            self.status = None;
            return;
        };

        match self.store.list_run_config_values(run_config_id).instrument(tracing::debug_span!("refresh_selected_details/1", file = file!(), line = line!())).await {
            Ok(values) => {
                self.status = None;
                let mut rows = values
                    .into_iter()
                    .map(|(name, value)| (name, value_to_string(&value)))
                    .collect::<Vec<(String, String)>>();
                rows.sort_by(|a, b| a.0.cmp(&b.0));
                debug!("refresh_selected_values: run_config_id={} rows={}", run_config_id, rows.len());
                self.selected_values = rows;
            }
            Err(e) => {
                warn!("refresh_selected_details failed to list values: run_config_id={} err={e:?}", run_config_id);
                self.status = Some(format!("Failed to list run_config_values for run_config_id={run_config_id}: {e:?}"));
            }
        }

        if let Some(cached_motifs) = self.run_motifs_cache.get(&run_config_id).cloned() {
            self.apply_selected_run_motifs(cached_motifs, None);
        }
    }

    fn open_selected_motif_plot(&mut self) {
        if let Some(pending_plot_open) = &self.pending_plot_open {
            self.status = Some(format!("Already opening plot for motif_id={}.", pending_plot_open.motif_id));
            return;
        }

        let Some(selected_motif) = self.selected_motif().cloned() else {
            self.status = Some("No motif selected.".to_string());
            return;
        };

        let motif_id = selected_motif.motif_id;
        let ws_path = self.ws_path.clone();
        self.status = Some(format!("Opening plot for motif_id={motif_id}..."));
        self.pending_plot_open = Some(PendingMotifPlotOpen {
            motif_id,
            started_at: Instant::now(),
            task: tokio::spawn(async move { open_motif_plot_task(ws_path, motif_id).instrument(tracing::debug_span!("open_selected_motif_plot/1", file = file!(), line = line!())).await }.instrument(tracing::info_span!(
                "tui::open_motif_plot",
                kind = "spawn",
                file = file!(),
                line = line!(),
                task = format!("open motif plot for motif_id={motif_id}"),
            ))),
        });
    }

    async fn poll_pending_plot_open(&mut self) {
        let Some(is_finished) = self.pending_plot_open.as_ref().map(|pending_plot_open| pending_plot_open.task.is_finished()) else {
            return;
        };
        if !is_finished {
            return;
        }

        let pending_plot_open = self.pending_plot_open.take().expect("pending plot open exists");
        match pending_plot_open.task.instrument(tracing::debug_span!("poll_pending_plot_open/1", file = file!(), line = line!())).await {
            Ok(status) => {
                self.status = Some(status);
            }
            Err(err) if err.is_cancelled() => {
                self.status = Some(format!("Opening plot for motif_id={} was cancelled.", pending_plot_open.motif_id));
            }
            Err(err) => {
                self.status = Some(format!("Plot task failed for motif_id={}: {err}", pending_plot_open.motif_id));
            }
        }
    }
}

struct TerminalGuard;

impl TerminalGuard {
    fn enter() -> Result<Self, Box<dyn std::error::Error>> {
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen)?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
    }
}

#[derive(Debug)]
enum ProgressClientEvent {
    Snapshot(CollectEtaProgressState),
    Status(String),
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logger();
    let args = Args::parse();
    let db_path = args.ws_path.join("main_db.db3");
    info!("starting thoosa_tui: ws_path={} db_path={}", args.ws_path.display(), db_path.display());
    let store = SqlStore::from_path(&db_path).instrument(tracing::debug_span!("main/1", file = file!(), line = line!())).await.map_err(|e| io::Error::other(format!("{e:?}")))?;

    let mut app = App::new(store, args.ws_path.clone(), db_path);
    app.refresh_run_configs().instrument(tracing::debug_span!("main/2", file = file!(), line = line!())).await;
    app.refresh_selected_details().instrument(tracing::debug_span!("main/3", file = file!(), line = line!())).await;

    if !io::stdout().is_terminal() || !io::stdin().is_terminal() {
        warn!("no TTY detected, switching to non-interactive mode");
        return run_non_interactive(&app);
    }

    run_ui(&mut app).instrument(tracing::debug_span!("main/4", file = file!(), line = line!())).await
}

async fn run_ui(app: &mut App) -> Result<(), Box<dyn std::error::Error>> {
    let _terminal_guard = TerminalGuard::enter()?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<ProgressClientEvent>();
    let progress_task = spawn_progress_client(COLLECT_ETA_PROGRESS_SOCKET_ID.to_string(), progress_tx);

    loop {
        drain_progress_events(app, &mut progress_rx);
        app.poll_pending_plot_open().instrument(tracing::debug_span!("run_ui/1", file = file!(), line = line!())).await;
        terminal.draw(|frame| render(frame, app))?;

        if event::poll(Duration::from_millis(200))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                app.expire_pending_gg_if_needed();
                if key.code != KeyCode::Char('g') {
                    app.clear_pending_gg();
                }

                if app.panel_search.input_open {
                    match key.code {
                        KeyCode::Esc => {
                            app.close_panel_search_input();
                        }
                        KeyCode::Enter => {
                            app.apply_panel_search_input();
                        }
                        KeyCode::Backspace => {
                            app.panel_search.draft.pop();
                        }
                        KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) && !key.modifiers.contains(KeyModifiers::ALT) => {
                            app.panel_search.draft.push(ch);
                        }
                        _ => {}
                    }
                    continue;
                }

                if key.code == KeyCode::Char('q') {
                    break;
                }

                if key.code == KeyCode::Char(' ') && key.modifiers.is_empty() && matches!(app.selected_area, SelectedArea::LeftRunConfigs | SelectedArea::RightMotifs | SelectedArea::ProgressView) {
                    app.open_view_popup();
                    continue;
                }

                if matches!(key.code, KeyCode::Char('f') | KeyCode::Char('F')) && is_unmodified_or_shift(&key) && app.selected_area == SelectedArea::LeftRunConfigs {
                    app.open_filter_popup().instrument(tracing::debug_span!("run_ui/2", file = file!(), line = line!())).await;
                    continue;
                }

                if key.code == KeyCode::Char('/') && key.modifiers.is_empty() && matches!(app.selected_area, SelectedArea::LeftRunConfigs | SelectedArea::RightMotifs) {
                    app.open_panel_search_input();
                    continue;
                }

                match app.selected_area {
                    SelectedArea::LeftRunConfigs => {
                        let mut selection_changed = false;

                        match key.code {
                            KeyCode::Char('r') if key.modifiers.is_empty() => {
                                app.refresh_run_configs().instrument(tracing::debug_span!("run_ui/3", file = file!(), line = line!())).await;
                                app.refresh_selected_details().instrument(tracing::debug_span!("run_ui/4", file = file!(), line = line!())).await;
                            }
                            KeyCode::Char('s') if key.modifiers.is_empty() => {
                                app.toggle_run_on_sort_order();
                                app.refresh_selected_details().instrument(tracing::debug_span!("run_ui/5", file = file!(), line = line!())).await;
                            }
                            KeyCode::Char('t') if key.modifiers.is_empty() => {
                                app.open_type_filter_popup().instrument(tracing::debug_span!("run_ui/6", file = file!(), line = line!())).await;
                            }
                            KeyCode::Char('T') if is_unmodified_or_shift(&key) => {
                                app.reset_search_filters().instrument(tracing::debug_span!("run_ui/7", file = file!(), line = line!())).await;
                                app.refresh_selected_details().instrument(tracing::debug_span!("run_ui/8", file = file!(), line = line!())).await;
                            }
                            KeyCode::Tab if key.modifiers.is_empty() => {
                                app.open_selected_run_motifs().instrument(tracing::debug_span!("run_ui/9", file = file!(), line = line!())).await;
                            }
                            KeyCode::Char('m') if key.modifiers.is_empty() => {
                                app.toggle_selected_run_motifs_panel().instrument(tracing::debug_span!("run_ui/10", file = file!(), line = line!())).await;
                            }
                            KeyCode::Enter if key.modifiers.is_empty() => {
                                app.open_selected_run_from_left().instrument(tracing::debug_span!("run_ui/11", file = file!(), line = line!())).await;
                            }
                            KeyCode::Right if key.modifiers.is_empty() => {
                                app.open_selected_run_right().instrument(tracing::debug_span!("run_ui/12", file = file!(), line = line!())).await;
                            }
                            KeyCode::Left if key.modifiers.is_empty() => {
                                app.collapse_selected_run();
                            }
                            KeyCode::Down | KeyCode::Char('j') if key.modifiers.is_empty() => {
                                selection_changed = app.move_main_down_lazy(1).instrument(tracing::debug_span!("run_ui/13", file = file!(), line = line!())).await;
                            }
                            KeyCode::Up | KeyCode::Char('k') if key.modifiers.is_empty() => {
                                selection_changed = app.move_main_up(1);
                            }
                            KeyCode::Char('d') if is_ctrl_char(&key, 'd') => {
                                selection_changed = app.move_main_down_lazy(app.main_page_size()).instrument(tracing::debug_span!("run_ui/14", file = file!(), line = line!())).await;
                            }
                            KeyCode::Char('u') if is_ctrl_char(&key, 'u') => {
                                selection_changed = app.move_main_up(app.main_page_size());
                            }
                            KeyCode::Char('G') if is_unmodified_or_shift(&key) => {
                                selection_changed = app.move_main_to_bottom_lazy().instrument(tracing::debug_span!("run_ui/15", file = file!(), line = line!())).await;
                            }
                            KeyCode::Char('g') if key.modifiers.is_empty() => {
                                if app.handle_g_press(SelectedArea::LeftRunConfigs) {
                                    selection_changed = app.move_main_to_top();
                                }
                            }
                            KeyCode::Char('n') if key.modifiers.is_empty() => {
                                app.navigate_panel_search(true);
                            }
                            KeyCode::Char('p') if key.modifiers.is_empty() => {
                                app.navigate_panel_search(false);
                            }
                            _ => {}
                        }

                        if selection_changed {
                            app.refresh_selected_details().instrument(tracing::debug_span!("run_ui/16", file = file!(), line = line!())).await;
                        }
                    }
                    SelectedArea::RightMotifs => {
                        match key.code {
                            KeyCode::Tab | KeyCode::Left if key.modifiers.is_empty() => {
                                app.selected_area = SelectedArea::LeftRunConfigs;
                                app.prefer_right_motifs = false;
                            }
                            KeyCode::Enter if key.modifiers.is_empty() => {
                                app.open_selected_motif_plot();
                            }
                            KeyCode::Char('m') if key.modifiers.is_empty() => {
                                app.toggle_selected_run_motifs_panel().instrument(tracing::debug_span!("run_ui/17", file = file!(), line = line!())).await;
                            }
                            KeyCode::Down | KeyCode::Char('j') if key.modifiers.is_empty() => {
                                app.move_motif_down(1);
                            }
                            KeyCode::Up | KeyCode::Char('k') if key.modifiers.is_empty() => {
                                app.move_motif_up(1);
                            }
                            KeyCode::Char('d') if is_ctrl_char(&key, 'd') => {
                                app.move_motif_down(app.motif_page_size());
                            }
                            KeyCode::Char('u') if is_ctrl_char(&key, 'u') => {
                                app.move_motif_up(app.motif_page_size());
                            }
                            KeyCode::Char('G') if is_unmodified_or_shift(&key) => {
                                app.move_motif_to_bottom();
                            }
                            KeyCode::Char('g') if key.modifiers.is_empty() => {
                                if app.handle_g_press(SelectedArea::RightMotifs) {
                                    app.move_motif_to_top();
                                }
                            }
                            KeyCode::Char('n') if key.modifiers.is_empty() => {
                                app.navigate_panel_search(true);
                            }
                            KeyCode::Char('p') if key.modifiers.is_empty() => {
                                app.navigate_panel_search(false);
                            }
                            _ => {}
                        }
                    }
                    SelectedArea::ProgressView => {}
                    SelectedArea::TypeFilterPopup => {
                        match key.code {
                            KeyCode::Esc => {
                                app.selected_area = app.default_run_configs_area();
                            }
                            KeyCode::Enter => {
                                app.apply_type_filter_from_popup().instrument(tracing::debug_span!("run_ui/18", file = file!(), line = line!())).await;
                                app.selected_area = app.default_run_configs_area();
                                app.refresh_selected_details().instrument(tracing::debug_span!("run_ui/19", file = file!(), line = line!())).await;
                            }
                            KeyCode::Char('T') if is_unmodified_or_shift(&key) => {
                                app.reset_search_filters().instrument(tracing::debug_span!("run_ui/20", file = file!(), line = line!())).await;
                                app.selected_area = app.default_run_configs_area();
                                app.refresh_selected_details().instrument(tracing::debug_span!("run_ui/21", file = file!(), line = line!())).await;
                            }
                            KeyCode::Down | KeyCode::Char('j') if key.modifiers.is_empty() => {
                                app.move_popup_down(1);
                            }
                            KeyCode::Up | KeyCode::Char('k') if key.modifiers.is_empty() => {
                                app.move_popup_up(1);
                            }
                            KeyCode::Char('d') if is_ctrl_char(&key, 'd') => {
                                app.move_popup_down(app.popup_page_size());
                            }
                            KeyCode::Char('u') if is_ctrl_char(&key, 'u') => {
                                app.move_popup_up(app.popup_page_size());
                            }
                            KeyCode::Char('G') if is_unmodified_or_shift(&key) => {
                                app.move_popup_to_bottom();
                            }
                            KeyCode::Char('g') if key.modifiers.is_empty() => {
                                if app.handle_g_press(SelectedArea::TypeFilterPopup) {
                                    app.move_popup_to_top();
                                }
                            }
                            _ => {}
                        }
                    }
                    SelectedArea::SearchPopup => {
                        match key.code {
                            KeyCode::Esc => {
                                app.selected_area = app.default_run_configs_area();
                            }
                            KeyCode::Enter => {
                                match app.selected_search_popup_row() {
                                    SearchPopupRow::Field(SearchField::Type) => {
                                        app.open_search_type_popup();
                                    }
                                    SearchPopupRow::Field(_) => {}
                                    SearchPopupRow::Ok => {
                                        if app.apply_search_from_popup().instrument(tracing::debug_span!("run_ui/22", file = file!(), line = line!())).await {
                                            app.refresh_selected_details().instrument(tracing::debug_span!("run_ui/23", file = file!(), line = line!())).await;
                                        }
                                    }
                                    SearchPopupRow::Cancel => {
                                        app.selected_area = app.default_run_configs_area();
                                    }
                                }
                            }
                            KeyCode::Right | KeyCode::Left if app.selected_search_field() == Some(SearchField::Type) => {
                                app.open_search_type_popup();
                            }
                            KeyCode::Backspace => {
                                app.pop_from_search_popup();
                            }
                            KeyCode::Char('r') if is_ctrl_char(&key, 'r') => {
                                app.reset_search_popup_draft();
                            }
                            KeyCode::Down | KeyCode::Char('j') if key.modifiers.is_empty() => {
                                app.move_search_popup_down(1);
                            }
                            KeyCode::Up | KeyCode::Char('k') if key.modifiers.is_empty() => {
                                app.move_search_popup_up(1);
                            }
                            KeyCode::Char('G') if is_unmodified_or_shift(&key) => {
                                app.move_search_popup_to_bottom();
                            }
                            KeyCode::Char('g') if key.modifiers.is_empty() => {
                                if app.handle_g_press(SelectedArea::SearchPopup) {
                                    app.move_search_popup_to_top();
                                }
                            }
                            KeyCode::Char(ch) if !key.modifiers.contains(KeyModifiers::CONTROL) && !key.modifiers.contains(KeyModifiers::ALT) => {
                                app.append_to_search_popup(ch);
                            }
                            _ => {}
                        }
                    }
                    SelectedArea::SearchTypePopup => {
                        match key.code {
                            KeyCode::Esc => {
                                app.selected_area = SelectedArea::SearchPopup;
                            }
                            KeyCode::Enter => {
                                app.apply_search_type_from_popup();
                            }
                            KeyCode::Down | KeyCode::Char('j') if key.modifiers.is_empty() => {
                                app.move_search_type_popup_down(1);
                            }
                            KeyCode::Up | KeyCode::Char('k') if key.modifiers.is_empty() => {
                                app.move_search_type_popup_up(1);
                            }
                            KeyCode::Char('d') if is_ctrl_char(&key, 'd') => {
                                app.move_search_type_popup_down(app.search_type_popup_page_size());
                            }
                            KeyCode::Char('u') if is_ctrl_char(&key, 'u') => {
                                app.move_search_type_popup_up(app.search_type_popup_page_size());
                            }
                            KeyCode::Char('G') if is_unmodified_or_shift(&key) => {
                                app.move_search_type_popup_to_bottom();
                            }
                            KeyCode::Char('g') if key.modifiers.is_empty() => {
                                if app.handle_g_press(SelectedArea::SearchTypePopup) {
                                    app.move_search_type_popup_to_top();
                                }
                            }
                            _ => {}
                        }
                    }
                    SelectedArea::ViewPopup => {
                        match key.code {
                            KeyCode::Esc => {
                                app.selected_area = match app.view_mode {
                                    ViewMode::RunConfigs => app.default_run_configs_area(),
                                    ViewMode::Progress => SelectedArea::ProgressView,
                                };
                            }
                            KeyCode::Enter => {
                                app.apply_view_mode_from_popup();
                            }
                            KeyCode::Down | KeyCode::Char('j') if key.modifiers.is_empty() => {
                                app.move_view_popup_down(1);
                            }
                            KeyCode::Up | KeyCode::Char('k') if key.modifiers.is_empty() => {
                                app.move_view_popup_up(1);
                            }
                            KeyCode::Char('d') if is_ctrl_char(&key, 'd') => {
                                app.move_view_popup_down(app.view_popup_page_size());
                            }
                            KeyCode::Char('u') if is_ctrl_char(&key, 'u') => {
                                app.move_view_popup_up(app.view_popup_page_size());
                            }
                            KeyCode::Char('G') if is_unmodified_or_shift(&key) => {
                                app.move_view_popup_to_bottom();
                            }
                            KeyCode::Char('g') if key.modifiers.is_empty() => {
                                if app.handle_g_press(SelectedArea::ViewPopup) {
                                    app.move_view_popup_to_top();
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    if let Some(pending_plot_open) = app.pending_plot_open.take() {
        pending_plot_open.task.abort();
        let _ = pending_plot_open.task.instrument(tracing::debug_span!("run_ui/24", file = file!(), line = line!())).await;
    }
    progress_task.abort();
    let _ = progress_task.instrument(tracing::debug_span!("run_ui/25", file = file!(), line = line!())).await;
    terminal.show_cursor()?;
    Ok(())
}

fn render(frame: &mut Frame<'_>, app: &App) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(frame.area());

    match app.view_mode {
        ViewMode::RunConfigs => render_run_configs_view(frame, app, outer[0]),
        ViewMode::Progress => render_progress_view(frame, app, outer[0]),
    }

    if app.selected_area == SelectedArea::TypeFilterPopup {
        render_type_filter_popup(frame, app);
    } else if matches!(app.selected_area, SelectedArea::SearchPopup | SelectedArea::SearchTypePopup) {
        render_search_popup(frame, app);
        if app.selected_area == SelectedArea::SearchTypePopup {
            render_search_type_popup(frame, app);
        }
    } else if app.selected_area == SelectedArea::ViewPopup {
        render_view_popup(frame, app);
    }

    render_status_line(frame, app, outer[1]);
}

fn render_run_configs_view(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(48), Constraint::Percentage(52)])
        .split(area);

    let (left_list_area, left_input_area) = if app.panel_search.input_open {
        let left_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(3)])
            .split(layout[0]);
        (left_layout[0], Some(left_layout[1]))
    } else {
        (layout[0], None)
    };

    let list_window_start = app.main_list_offset.min(app.run_configs.len().saturating_sub(1));
    let list_window_end = (list_window_start + app.main_page_size()).min(app.run_configs.len());
    let left_items = if app.run_configs.is_empty() {
        vec![ListItem::new("No entries in run_configs.")]
    } else {
        build_searchable_list_items(
            app.run_configs[list_window_start..list_window_end].iter().map(|entry| format_run_tree_row(app, entry)).collect::<Vec<String>>(),
            list_window_start,
            app,
            PanelSearchTarget::RunConfigs,
        )
    };

    let left_title = if app.search_filters.is_empty() {
        format!("run_configs [{} / {}]", app.run_configs.len(), app.all_run_configs.len())
    } else {
        format!("run_configs [filtered {} / {}]", app.run_configs.len(), app.all_run_configs.len())
    };

    let left_block = Block::default().title(left_title).borders(Borders::ALL);
    let left_list = List::new(left_items)
        .block(left_block)
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        .highlight_symbol(">> ");

    let mut left_state = ListState::default();
    if !app.run_configs.is_empty() {
        left_state.select(Some(app.selected_idx.min(app.run_configs.len().saturating_sub(1)).saturating_sub(list_window_start)));
    }
    frame.render_stateful_widget(left_list, left_list_area, &mut left_state);

    if let Some(input_area) = left_input_area {
        render_panel_search_input(frame, app, input_area);
    }

    if !app.motif_panel_visible() {
        let right = Paragraph::new(render_run_value_lines(app).join("\n"))
            .block(Block::default().title("run_config_values").borders(Borders::ALL))
            .wrap(Wrap { trim: false });
        frame.render_widget(right, layout[1]);
        return;
    }

    let right_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
        .split(layout[1]);
    let motif_window_start = app.motif_list_offset.min(app.selected_run_motifs.len().saturating_sub(1));
    let motif_window_end = (motif_window_start + app.motif_page_size()).min(app.selected_run_motifs.len());
    let motif_items = build_searchable_list_items(
        app.selected_run_motifs[motif_window_start..motif_window_end].iter().map(format_run_motif_row).collect::<Vec<String>>(),
        motif_window_start,
        app,
        PanelSearchTarget::Motifs,
    );
    let motif_list = List::new(motif_items)
        .block(Block::default().title(format!("motifs [{}]", app.selected_run_motifs.len())).borders(Borders::ALL))
        .highlight_style(if app.selected_area == SelectedArea::RightMotifs {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().add_modifier(Modifier::BOLD)
        })
        .highlight_symbol(">> ");
    let mut motif_state = ListState::default();
    motif_state.select(Some(app.selected_motif_idx.min(app.selected_run_motifs.len().saturating_sub(1)).saturating_sub(motif_window_start)));
    frame.render_stateful_widget(motif_list, right_layout[0], &mut motif_state);

    let motif_details = Paragraph::new(render_motif_detail_lines(app).join("\n"))
        .block(Block::default().title("motif details").borders(Borders::ALL))
        .wrap(Wrap { trim: false });
    frame.render_widget(motif_details, right_layout[1]);
}

fn format_run_tree_row(app: &App, row: &VisibleRunConfigRow) -> String {
    let indent = "  ".repeat(row.depth);

    match &row.kind {
        VisibleRunConfigRowKind::RunOnDate { run_on_date } => {
            let tree_marker = if row.expanded { "▼" } else { "▶" };
            format!("{}{} {}", indent, tree_marker, run_on_date)
        }
        VisibleRunConfigRowKind::RunConfig { run_config_id } => {
            let Some(node) = app.run_config_tree.get(run_config_id) else {
                return format!("<missing run {}>", run_config_id);
            };
            let tree_marker = if row.has_children {
                if row.expanded {
                    "▼"
                } else if node.children_loaded || !node.output_run_config_ids.is_empty() {
                    "▶"
                } else {
                    "▷"
                }
            } else {
                "•"
            };
            format!(
                "{}{} {:>6}  {:<19}  {:<24}  {}",
                indent, tree_marker, node.entry.id, node.entry.run_on, node.entry.run_config_type, node.entry.descr
            )
        }
        VisibleRunConfigRowKind::Motif { motif, .. } => {
            let score = motif.score.map(|value| format!("{value:.4}")).unwrap_or_else(|| "-".to_string());
            format!(
                "{}• motif {:>8}  ch {:<3}  slices {:<6}  resources {:<6}  score {}",
                indent, motif.motif_id, motif.channel_id, motif.slices_cnt, motif.resources_cnt, score
            )
        }
    }
}

fn render_run_value_lines(app: &App) -> Vec<String> {
    let mut right_lines = vec![];

    if let Some(run_entry) = app.selected_run_entry() {
        right_lines.push(format!("run_config_id: {}", run_entry.id));
        right_lines.push(format!("type: {}", run_entry.run_config_type));
        right_lines.push(format!("run_on: {}", run_entry.run_on));
        right_lines.push(format!("descr: {}", run_entry.descr));
    } else if let Some(run_on_date) = app.selected_run_on_date() {
        right_lines.push(format!("run_on date: {}", run_on_date));
    } else if let Some(motif) = app.selected_tree_motif() {
        right_lines.push(format!("motif_id: {}", motif.motif_id));
        right_lines.push(format!("source_run_config_id: {}", motif.source_run_config_id));
        right_lines.push(format!("channel: {}", motif.channel_id));
        if motif_counts_loaded(motif) {
            right_lines.push(format!("slices: {}", motif.slices_cnt));
            right_lines.push(format!("resources: {}", motif.resources_cnt));
        } else {
            right_lines.push("slices: <not loaded>".to_string());
            right_lines.push("resources: <not loaded>".to_string());
        }
    } else {
        right_lines.push("run_config_id: <none>".to_string());
    }

    right_lines.push(format!("filters: {}", app.search_filters.summary()));
    right_lines.push(format!("id filter: {}", display_search_text(&app.search_filters.id, "<none>")));
    right_lines.push(format!("type filter: {}", app.search_filters.run_config_type.as_deref().unwrap_or(SEARCH_TYPE_NONE_LABEL)));
    right_lines.push(format!("descr filter: {}", display_search_text(&app.search_filters.descr, "<none>")));
    right_lines.push(format!("run_on from: {}", display_search_text(&app.search_filters.run_on_from, "<none>")));
    right_lines.push(format!("run_on to: {}", display_search_text(&app.search_filters.run_on_to, "<none>")));
    right_lines.push(String::new());

    if app.selected_run_supports_motifs() && !app.motif_panel_visible() {
        let motifs_line = app
            .selected_run_config_id()
            .and_then(|run_config_id| app.run_motifs_cache.get(&run_config_id))
            .map(|motifs| {
                if motifs.is_empty() {
                    "motifs: none".to_string()
                } else {
                    "motifs: press m to view".to_string()
                }
            })
            .unwrap_or_else(|| "motifs: press m to load".to_string());
        right_lines.push(motifs_line);
        right_lines.push(String::new());
    }

    for (name, value) in &app.selected_values {
        right_lines.push(format!("{name} = {value}"));
    }

    if let Some(status) = &app.status {
        right_lines.push(String::new());
        right_lines.push(format!("status: {status}"));
    }

    right_lines
}

fn motif_counts_loaded(motif: &RunMotifEntry) -> bool {
    motif.score.is_some() || motif.slices_cnt != 0 || motif.resources_cnt != 0
}

fn format_run_motif_row(motif: &RunMotifEntry) -> String {
    match motif.score {
        Some(score) => format!(
            "{:>8}  ch {:<2}  score {:>8.5}  slices {:>5}  resources {:>4}",
            motif.motif_id, motif.channel_id, score, motif.slices_cnt, motif.resources_cnt
        ),
        None if motif_counts_loaded(motif) => format!(
            "{:>8}  ch {:<2}  slices {:>5}  resources {:>4}",
            motif.motif_id, motif.channel_id, motif.slices_cnt, motif.resources_cnt
        ),
        None => format!("{:>8}  ch {:<2}", motif.motif_id, motif.channel_id),
    }
}

fn render_motif_detail_lines(app: &App) -> Vec<String> {
    let mut lines = vec![];

    if let Some(run_entry) = app.selected_run_entry() {
        lines.push(format!("run_config_id: {}", run_entry.id));
        lines.push(format!("type: {}", run_entry.run_config_type));
        lines.push(format!("descr: {}", run_entry.descr));
    }

    lines.push(String::new());

    if let Some(motif) = app.selected_motif() {
        lines.push(format!("motif_id: {}", motif.motif_id));
        lines.push(format!("channel: {}", motif.channel_id));
        lines.push(format!("source find_motifs run_config_id: {}", motif.source_run_config_id));
        lines.push(format!("score: {}", motif.score.map(|value| format!("{value:.6}")).unwrap_or_else(|| "<n/a>".to_string())));
        if motif_counts_loaded(motif) {
            lines.push(format!("slices: {}", motif.slices_cnt));
            lines.push(format!("resources: {}", motif.resources_cnt));
        } else {
            lines.push("slices: <not loaded>".to_string());
            lines.push("resources: <not loaded>".to_string());
        }
        lines.push(String::new());
        lines.push("[Enter] open plot".to_string());
        lines.push("[m] switch back to run_config_values".to_string());
        lines.push("The plot is generated on demand if it does not exist yet.".to_string());
    } else {
        lines.push("No motif selected.".to_string());
    }

    if !app.selected_values.is_empty() {
        lines.push(String::new());
        lines.push("run_config_values:".to_string());
        for (name, value) in &app.selected_values {
            lines.push(format!("{name} = {value}"));
        }
    }

    if let Some(status) = &app.status {
        lines.push(String::new());
        lines.push(format!("status: {status}"));
    }

    lines
}

fn render_progress_view(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let progress = Paragraph::new(app.progress_lines().join("\n"))
        .block(Block::default().title("progress").borders(Borders::ALL))
        .wrap(Wrap { trim: false });
    frame.render_widget(progress, area);
}

fn render_status_line(frame: &mut Frame<'_>, app: &App, area: Rect) {
    if area.width == 0 {
        return;
    }

    let left = status_hint_text(app);
    let max_right_width = (area.width as usize).saturating_sub(1).min(((area.width as usize) * 2 / 5).max(24));
    let right = status_right_text(app, max_right_width);
    let right_width = right.chars().count().min(area.width as usize) as u16;
    if right_width >= area.width {
        frame.render_widget(Paragraph::new(right), area);
        return;
    }

    let layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(1), Constraint::Length(right_width)])
        .split(area);
    frame.render_widget(Paragraph::new(left), layout[0]);
    frame.render_widget(Paragraph::new(right), layout[1]);
}

fn render_panel_search_input(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let title = match app.panel_search.target.unwrap_or(PanelSearchTarget::RunConfigs) {
        PanelSearchTarget::RunConfigs => "Search run_configs",
        PanelSearchTarget::Motifs => "Search motifs",
    };
    let value = if app.panel_search.draft.is_empty() {
        "<pattern>".to_string()
    } else {
        app.panel_search.draft.clone()
    };
    let input = Paragraph::new(format!("/{value}_"))
        .block(Block::default().title(title).borders(Borders::ALL));
    frame.render_widget(input, area);
}

fn build_searchable_list_items(lines: Vec<String>, line_offset: usize, app: &App, target: PanelSearchTarget) -> Vec<ListItem<'static>> {
    let matches = app.panel_search_matches(target);
    let active_match = app
        .panel_search
        .current_match_idx
        .filter(|idx| *idx < matches.len())
        .and_then(|idx| matches.get(idx).copied());
    let mut matches_by_line = HashMap::<usize, Vec<PanelSearchMatch>>::new();
    for search_match in matches {
        matches_by_line.entry(search_match.line_idx).or_default().push(search_match);
    }

    lines
        .into_iter()
        .enumerate()
        .map(|(line_idx, line)| {
            let absolute_line_idx = line_offset + line_idx;
            let bright_match = active_match.filter(|search_match| search_match.line_idx == absolute_line_idx);
            if let Some(line_matches) = matches_by_line.get(&absolute_line_idx) {
                ListItem::new(highlighted_search_line(line, line_matches.as_slice(), bright_match))
            } else {
                ListItem::new(Line::from(line))
            }
        })
        .collect::<Vec<ListItem<'static>>>()
}

fn highlighted_search_line(line: String, matches: &[PanelSearchMatch], bright_match: Option<PanelSearchMatch>) -> Line<'static> {
    if matches.is_empty() {
        return Line::from(line);
    }

    let mut ordered_matches = matches.to_vec();
    ordered_matches.sort_by_key(|search_match| (search_match.start, search_match.len));
    let char_starts = line.char_indices().map(|(byte_idx, _)| byte_idx).chain(std::iter::once(line.len())).collect::<Vec<usize>>();
    let total_chars = char_starts.len().saturating_sub(1);
    let mut spans = vec![];
    let mut cursor_char = 0usize;

    for search_match in ordered_matches {
        let start_char = search_match.start.min(total_chars);
        let end_char = (search_match.start + search_match.len).min(total_chars);
        if start_char >= end_char || end_char <= cursor_char {
            continue;
        }

        if start_char > cursor_char {
            spans.push(Span::raw(line[char_starts[cursor_char]..char_starts[start_char]].to_string()));
        }

        let style = if bright_match.map(|active| active.start == search_match.start && active.len == search_match.len).unwrap_or(false) {
            Style::default().bg(Color::Cyan).add_modifier(Modifier::UNDERLINED)
        } else {
            Style::default().bg(Color::DarkGray)
        };
        spans.push(Span::styled(line[char_starts[start_char]..char_starts[end_char]].to_string(), style));
        cursor_char = end_char;
    }

    if cursor_char < total_chars {
        spans.push(Span::raw(line[char_starts[cursor_char]..].to_string()));
    }

    Line::from(spans)
}

fn status_hint_text(app: &App) -> String {
    if app.panel_search.input_open {
        let target = match app.panel_search.target.unwrap_or(PanelSearchTarget::RunConfigs) {
            PanelSearchTarget::RunConfigs => "run_configs",
            PanelSearchTarget::Motifs => "motifs",
        };
        return format!("search {target} | type pattern | Enter apply | Backspace delete | Esc cancel");
    }

    match app.selected_area {
        SelectedArea::LeftRunConfigs => format!(
            "run tree | sort:{} (s) | j/k move | Ctrl-d/u page | gg/G | Enter toggle | Left/Right tree | m toggle motifs | Tab open motifs | / search | n/p next-prev | F filter | t type | T clear | r reload | Space view | q quit",
            app.sort_order_label()
        ),
        SelectedArea::RightMotifs => "motifs | j/k move | Ctrl-d/u page | gg/G | Enter plot | m values | / search | n/p next-prev | Tab/Left back | Space view | q quit".to_string(),
        SelectedArea::ProgressView => "progress | Space view | q quit".to_string(),
        SelectedArea::TypeFilterPopup => "type-filter | j/k move | Ctrl-d/u page | gg/G | Enter apply | Esc close | T clear".to_string(),
        SelectedArea::SearchPopup => "filter | j/k move | type to edit | Backspace delete | Enter type/open | Enter on Ok applies | Ctrl-r reset | Esc close".to_string(),
        SelectedArea::SearchTypePopup => "filter type | j/k move | Ctrl-d/u page | gg/G | Enter choose | Esc close".to_string(),
        SelectedArea::ViewPopup => "view | j/k move | Ctrl-d/u page | gg/G | Enter apply | Esc close".to_string(),
    }
}

fn status_right_text(app: &App, max_width: usize) -> String {
    if let Some(pending_plot_open) = &app.pending_plot_open {
        return truncate_status_right(
            format!("opening motif {} {}", pending_plot_open.motif_id, endless_progress_bar(pending_plot_open.started_at, 12)).as_str(),
            max_width,
        );
    }

    truncate_status_left(format!("db: {}", app.db_path.display()).as_str(), max_width)
}

fn truncate_status_left(text: &str, max_width: usize) -> String {
    if text.chars().count() <= max_width {
        return text.to_string();
    }
    if max_width <= 3 {
        return ".".repeat(max_width);
    }

    let tail = text.chars().rev().take(max_width - 3).collect::<Vec<char>>().into_iter().rev().collect::<String>();
    format!("...{tail}")
}

fn truncate_status_right(text: &str, max_width: usize) -> String {
    if text.chars().count() <= max_width {
        return text.to_string();
    }
    if max_width <= 3 {
        return ".".repeat(max_width);
    }

    let head = text.chars().take(max_width - 3).collect::<String>();
    format!("{head}...")
}

fn endless_progress_bar(started_at: Instant, width: usize) -> String {
    if width == 0 {
        return "[]".to_string();
    }

    let segment_len = width.clamp(1, 4);
    let travel = width.saturating_sub(segment_len);
    let tick = (started_at.elapsed().as_millis() / 120) as usize;
    let offset = if travel == 0 {
        0
    } else {
        let cycle = travel * 2;
        let phase = tick % cycle.max(1);
        if phase <= travel { phase } else { cycle - phase }
    };
    let mut chars = vec![' ' ; width];
    for idx in 0..segment_len {
        chars[offset + idx] = '=';
    }
    format!("[{}]", chars.into_iter().collect::<String>())
}

fn render_type_filter_popup(frame: &mut Frame<'_>, app: &App) {
    let area = frame.area();
    let popup_h = popup_height(area.height, app.run_config_types.len());
    let popup_w = (area.width.saturating_mul(70) / 100).max(30).min(area.width.saturating_sub(2).max(1));
    let popup_x = area.x + area.width.saturating_sub(popup_w) / 2;
    let popup_y = area.y + area.height.saturating_sub(popup_h) / 2;
    let popup_rect = Rect::new(popup_x, popup_y, popup_w, popup_h);

    let popup_items = if app.run_config_types.is_empty() {
        vec![ListItem::new("No run_config types found")]
    } else {
        app.run_config_types
            .iter()
            .map(|t| {
                let marker = if app.search_filters.run_config_type.as_deref() == Some(t.as_str()) { "*" } else { " " };
                ListItem::new(format!("{marker} {t}"))
            })
            .collect::<Vec<ListItem>>()
    };

    let popup = List::new(popup_items)
        .block(Block::default().title("Select run_configs.type (Enter apply, Esc cancel)").borders(Borders::ALL))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        .highlight_symbol(">> ");

    let mut state = ListState::default();
    if !app.run_config_types.is_empty() {
        state.select(Some(app.type_popup_selected_idx));
    }

    frame.render_widget(Clear, popup_rect);
    frame.render_stateful_widget(popup, popup_rect, &mut state);
}

fn render_search_popup(frame: &mut Frame<'_>, app: &App) {
    let area = frame.area();
    let popup_h = popup_height(area.height, SearchPopupRow::count());
    let popup_w = (area.width.saturating_mul(78) / 100).max(54).min(area.width.saturating_sub(2).max(1));
    let popup_x = area.x + area.width.saturating_sub(popup_w) / 2;
    let popup_y = area.y + area.height.saturating_sub(popup_h) / 2;
    let popup_rect = Rect::new(popup_x, popup_y, popup_w, popup_h);

    let mut popup_items = SearchField::ALL
        .iter()
        .map(|field| ListItem::new(format_search_field_row(*field, &app.search_popup)))
        .collect::<Vec<ListItem>>();
    popup_items.push(ListItem::new("[ Ok ]"));
    popup_items.push(ListItem::new("[ Cancel ]"));

    let popup = List::new(popup_items)
        .block(Block::default().title("Filter run_configs (Enter type/open, Ok applies, Ctrl-r reset)").borders(Borders::ALL))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        .highlight_symbol(">> ");

    let mut state = ListState::default();
    state.select(Some(app.search_popup.selected_field_idx));

    frame.render_widget(Clear, popup_rect);
    frame.render_stateful_widget(popup, popup_rect, &mut state);
}

fn render_search_type_popup(frame: &mut Frame<'_>, app: &App) {
    let area = frame.area();
    let popup_h = popup_height(area.height, app.run_config_types.len() + 1);
    let popup_w = (area.width.saturating_mul(42) / 100).max(24).min(area.width.saturating_sub(2).max(1));
    let popup_x = area.x + area.width.saturating_sub(popup_w) / 2;
    let popup_y = area.y + area.height.saturating_sub(popup_h) / 2;
    let popup_rect = Rect::new(popup_x, popup_y, popup_w, popup_h);

    let popup_items = std::iter::once(ListItem::new(format!(
        "{} {}",
        if app.search_popup.draft.run_config_type.is_none() { "*" } else { " " },
        SEARCH_TYPE_NONE_LABEL
    )))
    .chain(app.run_config_types.iter().map(|value| {
        let marker = if app.search_popup.draft.run_config_type.as_deref() == Some(value.as_str()) { "*" } else { " " };
        ListItem::new(format!("{marker} {value}"))
    }))
    .collect::<Vec<ListItem>>();

    let popup = List::new(popup_items)
        .block(Block::default().title("Select filter type (Enter choose, Esc cancel)").borders(Borders::ALL))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        .highlight_symbol(">> ");

    let mut state = ListState::default();
    state.select(Some(app.search_popup.type_selected_idx));

    frame.render_widget(Clear, popup_rect);
    frame.render_stateful_widget(popup, popup_rect, &mut state);
}

fn render_view_popup(frame: &mut Frame<'_>, app: &App) {
    let area = frame.area();
    let popup_h = popup_height(area.height, ViewMode::ALL.len());
    let popup_w = (area.width.saturating_mul(45) / 100).max(22).min(area.width.saturating_sub(2).max(1));
    let popup_x = area.x + area.width.saturating_sub(popup_w) / 2;
    let popup_y = area.y + area.height.saturating_sub(popup_h) / 2;
    let popup_rect = Rect::new(popup_x, popup_y, popup_w, popup_h);

    let popup_items = ViewMode::ALL
        .iter()
        .map(|mode| {
            let marker = if *mode == app.view_mode { "*" } else { " " };
            ListItem::new(format!("{marker} {}", mode.label()))
        })
        .collect::<Vec<ListItem>>();

    let popup = List::new(popup_items)
        .block(Block::default().title("Select panel (Enter apply, Esc cancel)").borders(Borders::ALL))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        .highlight_symbol(">> ");

    let mut state = ListState::default();
    state.select(Some(app.view_popup_selected_idx));

    frame.render_widget(Clear, popup_rect);
    frame.render_stateful_widget(popup, popup_rect, &mut state);
}

fn popup_height(term_rows: u16, items_count: usize) -> u16 {
    let max_h = term_rows.saturating_sub(4).max(3);
    let desired = items_count.saturating_add(2) as u16;
    desired.clamp(3, max_h)
}

fn sort_run_config_ids(run_config_ids: &mut Vec<u32>, run_on_by_id: &HashMap<u32, &str>, sort_order: SortOrder) {
    match sort_order {
        SortOrder::Asc => run_config_ids.sort_by(|a, b| {
            run_on_by_id
                .get(a)
                .cmp(&run_on_by_id.get(b))
                .then_with(|| a.cmp(b))
        }),
        SortOrder::Desc => run_config_ids.sort_by(|a, b| {
            run_on_by_id
                .get(b)
                .cmp(&run_on_by_id.get(a))
                .then_with(|| a.cmp(b))
        }),
    }
}

fn run_config_type_supports_motifs(run_config_type: &str) -> bool {
    matches!(
        run_config_type,
        "find_motifs" | "select_motifs" | "FindMotifsRunConfig" | "SelectMotifsRunConfig"
    )
}

fn run_config_type_supports_inline_motif_rows(run_config_type: &str) -> bool {
    matches!(run_config_type, "find_motifs" | "FindMotifsRunConfig")
}

fn format_progress_device_stats(device_stats: ThoDeviceStats) -> Vec<String> {
    vec![
        format!("gpu allocated: {}", format_progress_bytes(device_stats.allocated)),
        format!("gpu active: {}", format_progress_bytes(device_stats.active)),
        format!("gpu reserved: {}", format_progress_bytes(device_stats.reserved)),
        format!("gpu requested: {}", format_progress_bytes(device_stats.requested)),
        format!("gpu non-releasable: {}", format_progress_bytes(device_stats.non_releasable)),
        format!("gpu reusable cached: {}", format_progress_bytes(device_stats.reusable_cached())),
        format!("gpu pending free: {}", format_progress_bytes(device_stats.pending_free())),
        format!("gpu rounding overhead: {}", format_progress_bytes(device_stats.rounding_overhead())),
    ]
}

fn format_progress_bytes(bytes: i64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];

    let sign = if bytes < 0 { "-" } else { "" };
    let mut value = bytes.unsigned_abs() as f64;
    let mut unit_idx = 0usize;
    while value >= 1024.0 && unit_idx + 1 < UNITS.len() {
        value /= 1024.0;
        unit_idx += 1;
    }

    if unit_idx == 0 {
        format!("{sign}{} {}", value as u64, UNITS[unit_idx])
    } else {
        format!("{sign}{value:.1} {}", UNITS[unit_idx])
    }
}

fn format_progress_uptime(started_at_unix_ms: u64) -> String {
    if started_at_unix_ms == 0 {
        return "<unknown>".to_string();
    }

    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let elapsed = now_ms.saturating_sub(started_at_unix_ms);
    format_duration_short(Duration::from_millis(elapsed))
}

fn format_duration_short(duration: Duration) -> String {
    let secs = duration.as_secs();
    let days = secs / 86_400;
    let hours = (secs % 86_400) / 3_600;
    let mins = (secs % 3_600) / 60;
    let rem_secs = secs % 60;

    if days > 0 {
        format!("{days}d {hours:02}h {mins:02}m")
    } else if hours > 0 {
        format!("{hours}h {mins:02}m {rem_secs:02}s")
    } else if mins > 0 {
        format!("{mins}m {rem_secs:02}s")
    } else {
        format!("{rem_secs}s")
    }
}

fn format_search_field_row(field: SearchField, popup: &SearchPopupState) -> String {
    let is_selected = SearchField::ALL
        .get(popup.selected_field_idx)
        .copied()
        .map(|selected| selected == field)
        .unwrap_or(false);

    let raw_value = match field {
        SearchField::Id => search_text_value(&popup.draft.id, "<digits only>", is_selected),
        SearchField::Type => popup.draft.run_config_type.clone().unwrap_or_else(|| SEARCH_TYPE_NONE_LABEL.to_string()),
        SearchField::Descr => search_text_value(&popup.draft.descr, "<like %descr%>", is_selected),
        SearchField::RunOnFrom => search_text_value(&popup.draft.run_on_from, "<YYYY-MM-DD or YYYY-MM-DD HH:MM:SS>", is_selected),
        SearchField::RunOnTo => search_text_value(&popup.draft.run_on_to, "<YYYY-MM-DD or YYYY-MM-DD HH:MM:SS>", is_selected),
    };

    format!("{:<12} {}", field.label(), raw_value)
}

fn search_text_value(value: &str, placeholder: &str, is_selected: bool) -> String {
    let trimmed = value.trim();
    let base = if trimmed.is_empty() { placeholder } else { trimmed };
    if is_selected {
        format!("{base}_")
    } else {
        base.to_string()
    }
}

fn display_search_text(value: &str, default: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        default.to_string()
    } else {
        trimmed.to_string()
    }
}

fn normalize_for_case_insensitive_search(text: &str) -> (Vec<char>, Vec<usize>) {
    let mut normalized = vec![];
    let mut mapping = vec![];
    for (char_idx, ch) in text.chars().enumerate() {
        for lower_char in ch.to_lowercase() {
            normalized.push(lower_char);
            mapping.push(char_idx);
        }
    }
    (normalized, mapping)
}

fn find_case_insensitive_matches(line: &str, pattern: &str) -> Vec<(usize, usize)> {
    let (normalized_line, line_mapping) = normalize_for_case_insensitive_search(line);
    let (normalized_pattern, _) = normalize_for_case_insensitive_search(pattern);
    if normalized_pattern.is_empty() || normalized_pattern.len() > normalized_line.len() {
        return vec![];
    }

    let mut matches = vec![];
    let mut idx = 0usize;
    while idx + normalized_pattern.len() <= normalized_line.len() {
        if normalized_line[idx..idx + normalized_pattern.len()] == normalized_pattern[..] {
            let start = line_mapping[idx];
            let end = line_mapping[idx + normalized_pattern.len() - 1] + 1;
            matches.push((start, end.saturating_sub(start)));
            idx += normalized_pattern.len();
        } else {
            idx += 1;
        }
    }

    matches
}

fn append_run_on_char(target: &mut String, ch: char) -> bool {
    if ch.is_ascii_digit() || matches!(ch, '-' | ':' | ' ') {
        target.push(ch);
        true
    } else {
        false
    }
}

fn parse_run_config_id_filter(value: &str) -> Result<Option<u32>, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    if !trimmed.chars().all(|ch| ch.is_ascii_digit()) {
        return Err("id must contain digits only".to_string());
    }

    trimmed
        .parse::<u32>()
        .map(Some)
        .map_err(|err| format!("invalid id: {err}"))
}

fn run_on_date_label(run_on: &str) -> String {
    run_on.split_once(' ').map(|(date, _)| date).unwrap_or(run_on).to_string()
}

fn normalize_run_on_filter(value: &str, end_of_day: bool) -> Result<Option<String>, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    if let Ok(date_time) = NaiveDateTime::parse_from_str(trimmed, RUN_ON_DATETIME_FORMAT) {
        return Ok(Some(date_time.format(RUN_ON_DATETIME_FORMAT).to_string()));
    }

    if let Ok(date) = NaiveDate::parse_from_str(trimmed, RUN_ON_DATE_FORMAT) {
        let Some(date_time) = date.and_hms_opt(
            if end_of_day { 23 } else { 0 },
            if end_of_day { 59 } else { 0 },
            if end_of_day { 59 } else { 0 },
        ) else {
            return Err("invalid run_on date".to_string());
        };
        return Ok(Some(date_time.format(RUN_ON_DATETIME_FORMAT).to_string()));
    }

    Err("run_on must be YYYY-MM-DD or YYYY-MM-DD HH:MM:SS".to_string())
}

fn move_down(selected: &mut usize, len: usize, step: usize) -> bool {
    if len == 0 {
        return false;
    }

    let old = *selected;
    *selected = selected.saturating_add(step).min(len - 1);
    *selected != old
}

fn move_up(selected: &mut usize, len: usize, step: usize) -> bool {
    if len == 0 {
        return false;
    }

    let old = *selected;
    *selected = selected.saturating_sub(step);
    *selected != old
}

fn move_to_top(selected: &mut usize, len: usize) -> bool {
    if len == 0 {
        return false;
    }

    let old = *selected;
    *selected = 0;
    *selected != old
}

fn move_to_bottom(selected: &mut usize, len: usize) -> bool {
    if len == 0 {
        return false;
    }

    let old = *selected;
    *selected = len - 1;
    *selected != old
}

fn scroll_offset_by_pages_until_visible(offset: &mut usize, page_size: usize, max_offset: usize, selected_idx: usize) {
    if page_size == 0 {
        *offset = 0;
        return;
    }

    while selected_idx < *offset {
        let next_offset = offset.saturating_sub(page_size);
        if next_offset == *offset {
            break;
        }
        *offset = next_offset;
    }
    while selected_idx >= offset.saturating_add(page_size) {
        let next_offset = offset.saturating_add(page_size).min(max_offset);
        if next_offset == *offset {
            break;
        }
        *offset = next_offset;
    }

    *offset = (*offset).min(max_offset);
}

fn is_ctrl_char(key: &KeyEvent, c: char) -> bool {
    key.code == KeyCode::Char(c) && key.modifiers.contains(KeyModifiers::CONTROL)
}

fn is_unmodified_or_shift(key: &KeyEvent) -> bool {
    !key.modifiers.contains(KeyModifiers::CONTROL) && !key.modifiers.contains(KeyModifiers::ALT)
}

fn format_progress_invocation(invocation: &CollectEtaProgressInvocation) -> String {
    format!("{:>6} / {:<6} {}", invocation.completed, invocation.total, invocation.descr)
}

fn drain_progress_events(app: &mut App, rx: &mut mpsc::UnboundedReceiver<ProgressClientEvent>) {
    while let Ok(event) = rx.try_recv() {
        app.apply_progress_event(event);
    }
}

fn spawn_progress_client(socket_id: String, tx: mpsc::UnboundedSender<ProgressClientEvent>) -> JoinHandle<()> {
    tokio::spawn(async move {
        run_progress_client(socket_id, tx).instrument(tracing::debug_span!("spawn_progress_client/1", file = file!(), line = line!())).await;
    }.instrument(tracing::info_span!(
        "tui::progress_client",
        kind = "spawn",
        file = file!(),
        line = line!(),
        task = "receive progress socket events",
    )))
}

async fn run_progress_client(socket_id: String, tx: mpsc::UnboundedSender<ProgressClientEvent>) {
    let mut last_status = None;
    loop {
        if !send_progress_status(&tx, &mut last_status, "Connecting to progress socket.".to_string()) {
            break;
        }

        match connect_progress_stream(&socket_id).instrument(tracing::debug_span!("run_progress_client/1", file = file!(), line = line!())).await {
            Ok(stream) => {
                if !send_progress_status(&tx, &mut last_status, "Connected to progress socket.".to_string()) {
                    break;
                }
                if let Err(err) = read_progress_stream(stream, &tx, &mut last_status).instrument(tracing::debug_span!("run_progress_client/2", file = file!(), line = line!())).await {
                    if !send_progress_status(&tx, &mut last_status, format!("Progress socket error: {err}")) {
                        break;
                    }
                } else if !send_progress_status(&tx, &mut last_status, "Progress socket disconnected.".to_string()) {
                    break;
                }
            }
            Err(err) => {
                if !send_progress_status(&tx, &mut last_status, format!("Waiting for progress socket: {err}")) {
                    break;
                }
            }
        }

        tokio::time::sleep(Duration::from_secs(1)).instrument(tracing::debug_span!("run_progress_client/3", file = file!(), line = line!())).await;
    }
}

async fn read_progress_stream(
    stream: Stream,
    tx: &mpsc::UnboundedSender<ProgressClientEvent>,
    last_status: &mut Option<String>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut lines = BufReader::new(stream).lines();
    while let Some(line) = lines.next_line().instrument(tracing::debug_span!("read_progress_stream/1", file = file!(), line = line!())).await? {
        let snapshot: CollectEtaProgressState = serde_json::from_str(&line)?;
        if tx.send(ProgressClientEvent::Snapshot(snapshot)).is_err() {
            return Ok(());
        }
        *last_status = Some("Connected to progress socket.".to_string());
    }
    Ok(())
}

fn send_progress_status(
    tx: &mpsc::UnboundedSender<ProgressClientEvent>,
    last_status: &mut Option<String>,
    status: String,
) -> bool {
    if last_status.as_deref() == Some(status.as_str()) {
        return true;
    }
    *last_status = Some(status.clone());
    tx.send(ProgressClientEvent::Status(status)).is_ok()
}

async fn connect_progress_stream(socket_id: &str) -> io::Result<Stream> {
    let mut last_err = None;
    for name in progress_socket_names(socket_id)? {
        match Stream::connect(name.borrow()).instrument(tracing::debug_span!("connect_progress_stream/1", file = file!(), line = line!())).await {
            Ok(stream) => return Ok(stream),
            Err(err) => {
                last_err = Some(err);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| io::Error::other("no local socket names available")))
}

fn progress_socket_names(socket_id: &str) -> io::Result<Vec<Name<'static>>> {
    let mut names = Vec::with_capacity(2);
    if GenericNamespaced::is_supported() {
        names.push(format!("{socket_id}.sock").to_ns_name::<GenericNamespaced>()?);
    }
    if cfg!(windows) {
        names.push(format!(r"\\.\pipe\{socket_id}").to_fs_name::<GenericFilePath>()?);
    } else {
        names.push(format!("/tmp/{socket_id}.sock").to_fs_name::<GenericFilePath>()?);
    }
    Ok(names)
}

fn cargo_workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")))
        .to_path_buf()
}

fn release_plot_binary_candidates() -> Vec<PathBuf> {
    let workspace_root = cargo_workspace_root();
    vec![
        workspace_root.join(format!("target/x86_64-unknown-linux-gnu/release/{DEFAULT_PLOT_PROJECT}")),
        workspace_root.join(format!("target/release/{DEFAULT_PLOT_PROJECT}")),
    ]
}

fn plot_command(ws_path: &Path, motif_id: u32, print_to_stdout: bool) -> Command {
    let mut command = release_plot_binary_candidates()
        .into_iter()
        .find(|candidate| candidate.exists())
        .map(|candidate| Command::new(candidate))
        .unwrap_or_else(|| {
            let mut cargo_command = Command::new("cargo");
            cargo_command.current_dir(cargo_workspace_root());
            cargo_command.args(["run", "--release", "-p", DEFAULT_PLOT_PROJECT, "--"]);
            cargo_command
        });

    command.args([
        "--term-logger-level",
        "ERROR",
        "--write-logger-level",
        "ERROR",
        "--ws-path",
    ]);
    command.arg(ws_path);
    command.args(["plot-data", "--data-by"]);
    command.arg(format!("motif_id:{motif_id}"));
    if print_to_stdout {
        command.arg("--stdout");
    }
    command
}

async fn resolve_motif_plot_target(ws_path: &Path, motif_id: u32) -> Result<PathBuf, String> {
    let output = plot_command(ws_path, motif_id, true)
        .output()
        .instrument(tracing::debug_span!("resolve_motif_plot_target/1", file = file!(), line = line!())).await
        .map_err(|err| format!("spawn failed: {err}"))?;
    if !output.status.success() {
        return Err(command_output_error(&output.stdout, &output.stderr));
    }
    parse_plot_target_path(&output.stdout)
}

async fn generate_motif_plot(ws_path: &Path, motif_id: u32) -> Result<(), String> {
    let output = plot_command(ws_path, motif_id, false)
        .output()
        .instrument(tracing::debug_span!("generate_motif_plot/1", file = file!(), line = line!())).await
        .map_err(|err| format!("spawn failed: {err}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(command_output_error(&output.stdout, &output.stderr))
    }
}

async fn open_motif_plot_task(ws_path: PathBuf, motif_id: u32) -> String {
    match resolve_motif_plot_target(&ws_path, motif_id).instrument(tracing::debug_span!("open_motif_plot_task/1", file = file!(), line = line!())).await {
        Ok(target_path) => {
            let target_path = if target_path.is_absolute() { target_path } else { ws_path.join(target_path) };
            if !target_path.exists() {
                if let Err(err) = generate_motif_plot(&ws_path, motif_id).instrument(tracing::debug_span!("open_motif_plot_task/2", file = file!(), line = line!())).await {
                    return format!("Failed to generate plot for motif_id={motif_id}: {err}");
                }
            }

            match open_path_in_system_viewer(&target_path) {
                Ok(()) => format!("Opened plot: {}", target_path.display()),
                Err(err) => format!("Failed to open plot {}: {err}", target_path.display()),
            }
        }
        Err(err) => format!("Failed to resolve plot for motif_id={motif_id}: {err}"),
    }
}

fn parse_plot_target_path(stdout: &[u8]) -> Result<PathBuf, String> {
    let stdout = String::from_utf8_lossy(stdout);
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if let Some(target_path) = parsed.get("target_path").and_then(|value| value.as_str()) {
                return Ok(PathBuf::from(target_path));
            }
        }
    }

    Err("plot command did not emit a target_path".to_string())
}

fn command_output_error(stdout: &[u8], stderr: &[u8]) -> String {
    let stderr = String::from_utf8_lossy(stderr);
    if !stderr.trim().is_empty() {
        return stderr.trim().to_string();
    }

    let stdout = String::from_utf8_lossy(stdout);
    if !stdout.trim().is_empty() {
        return stdout.trim().to_string();
    }

    "command failed without output".to_string()
}

fn open_path_in_system_viewer(path: &Path) -> Result<(), String> {
    let mut command = if cfg!(target_os = "macos") {
        let mut command = std::process::Command::new("open");
        command.arg(path);
        command
    } else if cfg!(target_os = "windows") {
        let mut command = std::process::Command::new("cmd");
        command.args(["/C", "start", ""]);
        command.arg(path);
        command
    } else {
        let mut command = std::process::Command::new("xdg-open");
        command.arg(path);
        command
    };

    command.spawn().map(|_| ()).map_err(|err| err.to_string())
}

fn value_to_string(value: &Val) -> String {
    match value {
        Val::Real(v) => v.to_string(),
        Val::Int(v) => v.to_string(),
        Val::UInt(v) => v.to_string(),
        Val::Str(v) => v.clone(),
        Val::Bool(v) => v.to_string(),
    }
}

fn init_logger() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("thoosa_tui=debug")).try_init();
}

fn run_non_interactive(app: &App) -> Result<(), Box<dyn std::error::Error>> {
    println!("thoosa_tui non-interactive mode");
    println!("db: {}", app.db_path.display());
    println!("run_configs: {}", app.run_configs.len());
    println!();

    for row in app.run_configs.iter().take(20) {
        println!("{}", format_run_tree_row(app, row));
    }

    if app.run_configs.len() > 20 {
        println!("... {} more rows", app.run_configs.len() - 20);
    }

    println!();
    if let Some(run_config_id) = app.selected_run_config_id() {
        println!("selected run_config_id: {run_config_id}");
    } else {
        println!("selected run_config_id: <none>");
    }

    for (name, value) in &app.selected_values {
        println!("{name} = {value}");
    }

    if let Some(status) = &app.status {
        println!("status: {status}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process;
    use std::sync::atomic::{AtomicU64, Ordering};
    use interprocess::local_socket::ListenerOptions;
    use thoosa_store::MockStore;
    use tokio::io::AsyncWriteExt;
    use thoosa_shared::run_configs::RUN_ID;

    #[test]
    fn test_value_to_string_all_variants() {
        assert_eq!("1.5", value_to_string(&Val::Real(1.5)));
        assert_eq!("-2", value_to_string(&Val::Int(-2)));
        assert_eq!("7", value_to_string(&Val::UInt(7)));
        assert_eq!("abc", value_to_string(&Val::Str("abc".to_string())));
        assert_eq!("true", value_to_string(&Val::Bool(true)));
    }


    #[test]
    fn test_render_run_value_lines_excludes_db_path() {
        let store = Box::new(MockStore::new()) as Box<dyn Store>;
        let app = App::new(store, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));

        assert!(!render_run_value_lines(&app).iter().any(|line| line.starts_with("db: ")));
    }

    #[test]
    fn test_apply_view_mode_from_popup_switches_focus() {
        let store = Box::new(MockStore::new()) as Box<dyn Store>;
        let mut app = App::new(store, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));

        app.open_view_popup();
        app.view_popup_selected_idx = 1;
        app.apply_view_mode_from_popup();

        assert_eq!(app.view_mode, ViewMode::Progress);
        assert_eq!(app.selected_area, SelectedArea::ProgressView);
    }

    #[test]
    fn test_normalize_run_on_filter_supports_date_and_datetime() {
        assert_eq!(normalize_run_on_filter("2026-03-13", false).unwrap(), Some("2026-03-13 00:00:00".to_string()));
        assert_eq!(normalize_run_on_filter("2026-03-13", true).unwrap(), Some("2026-03-13 23:59:59".to_string()));
        assert_eq!(normalize_run_on_filter("2026-03-13 14:15:16", false).unwrap(), Some("2026-03-13 14:15:16".to_string()));
    }

    #[test]
    fn test_run_config_search_matches_all_columns() {
        let search = RunConfigSearch {
            id: "42".to_string(),
            run_config_type: Some("select_motifs".to_string()),
            descr: "Alpha".to_string(),
            run_on_from: "2026-03-12".to_string(),
            run_on_to: "2026-03-13".to_string(),
        };
        let normalized = search.normalized().unwrap();

        assert!(normalized.matches(&RunConfigEntry {
            id: 42,
            run_config_type: "select_motifs".to_string(),
            descr: "alpha beta".to_string(),
            run_on: "2026-03-13 10:30:00".to_string(),
        }));
        assert!(!normalized.matches(&RunConfigEntry {
            id: 41,
            run_config_type: "select_motifs".to_string(),
            descr: "alpha beta".to_string(),
            run_on: "2026-03-13 10:30:00".to_string(),
        }));
        assert!(!normalized.matches(&RunConfigEntry {
            id: 42,
            run_config_type: "train_motif_nets".to_string(),
            descr: "alpha beta".to_string(),
            run_on: "2026-03-13 10:30:00".to_string(),
        }));
        assert!(!normalized.matches(&RunConfigEntry {
            id: 42,
            run_config_type: "select_motifs".to_string(),
            descr: "gamma".to_string(),
            run_on: "2026-03-13 10:30:00".to_string(),
        }));
        assert!(!normalized.matches(&RunConfigEntry {
            id: 42,
            run_config_type: "select_motifs".to_string(),
            descr: "alpha beta".to_string(),
            run_on: "2026-03-14 00:00:00".to_string(),
        }));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_apply_search_from_popup_filters_visible_rows() ->Result<(), ThoError>{
        let store = SqlStore::in_mem().await?;
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        let alpha_run_config_id = store.save_run_config(None, "select_motifs", "first alpha", HashMap::new()).await.unwrap();
        let _beta_run_config_id = store.save_run_config(None, "train_motif_nets", "second beta", HashMap::new()).await.unwrap();

        let mut app = App::new(store, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));
        app.selected_area = SelectedArea::SearchPopup;
        app.search_popup.draft = RunConfigSearch {
            id: alpha_run_config_id.to_string(),
            run_config_type: Some("select_motifs".to_string()),
            descr: "alpha".to_string(),
            run_on_from: String::new(),
            run_on_to: String::new(),
        };

        assert!(app.apply_search_from_popup().await);
        assert_eq!(app.selected_area, SelectedArea::LeftRunConfigs);
        assert_eq!(app.search_filters.id, alpha_run_config_id.to_string());
        assert_eq!(app.run_configs.len(), 2);
        assert!(app.run_configs[0].run_on_date().is_some());
        assert_eq!(app.run_configs[1].run_config_id(), Some(alpha_run_config_id));

        Ok(())
    }

    #[test]
    fn test_rebuild_visible_run_configs_groups_runs_under_run_on_date() {
        let store = Box::new(MockStore::new()) as Box<dyn Store>;
        let mut app = App::new(store, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));
        app.has_more_paged_run_configs = false;
        app.all_run_configs = vec![
            RunConfigEntry {
                id: 1,
                run_config_type: "FindMotifsRunConfig".to_string(),
                descr: "older root".to_string(),
                run_on: "2026-03-14 08:00:00".to_string(),
            },
            RunConfigEntry {
                id: 2,
                run_config_type: "FindMotifsRunConfig".to_string(),
                descr: "newer root".to_string(),
                run_on: "2026-03-15 09:00:00".to_string(),
            },
            RunConfigEntry {
                id: 3,
                run_config_type: "TrainMotifNetRunConfig".to_string(),
                descr: "child".to_string(),
                run_on: "2026-03-15 10:00:00".to_string(),
            },
        ];
        app.rebuild_run_config_tree(&[(3, 2)], &HashMap::new());
        app.rebuild_visible_run_configs(None);

        assert_eq!(Some("2026-03-15"), app.run_configs.get(0).and_then(VisibleRunConfigRow::run_on_date));
        assert_eq!(Some(2), app.run_configs.get(1).and_then(VisibleRunConfigRow::run_config_id));
        assert_eq!(1, app.run_configs.get(1).map(|row| row.depth).unwrap_or_default());
        assert_eq!(Some(3), app.run_configs.get(2).and_then(VisibleRunConfigRow::run_config_id));
        assert_eq!(2, app.run_configs.get(2).map(|row| row.depth).unwrap_or_default());
        assert_eq!(Some("2026-03-14"), app.run_configs.get(3).and_then(VisibleRunConfigRow::run_on_date));
        assert_eq!(Some(1), app.run_configs.get(4).and_then(VisibleRunConfigRow::run_config_id));
        assert_eq!(1, app.run_configs.get(4).map(|row| row.depth).unwrap_or_default());
    }

    #[test]
    fn test_collapse_selected_run_collapses_run_on_date_group() {
        let store = Box::new(MockStore::new()) as Box<dyn Store>;
        let mut app = App::new(store, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));
        app.has_more_paged_run_configs = false;
        app.all_run_configs = vec![RunConfigEntry {
            id: 1,
            run_config_type: "FindMotifsRunConfig".to_string(),
            descr: "root".to_string(),
            run_on: "2026-03-14 08:00:00".to_string(),
        }];
        app.rebuild_run_config_tree(&[], &HashMap::new());
        app.rebuild_visible_run_configs(None);

        assert_eq!(Some("2026-03-14"), app.run_configs.get(0).and_then(VisibleRunConfigRow::run_on_date));
        assert!(app.collapse_selected_run());
        assert_eq!(1, app.run_configs.len());
        assert_eq!(Some("2026-03-14"), app.run_configs.get(0).and_then(VisibleRunConfigRow::run_on_date));
        assert!(!app.run_configs[0].expanded);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_filtered_view_deduplicates_shared_descendants() {
        let store = MockStore::new();
        let mut app = App::new(Box::new(store) as Box<dyn Store>, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));
        app.has_more_paged_run_configs = false;
        app.all_run_configs = vec![
            RunConfigEntry {
                id: 1,
                run_config_type: "root_a".to_string(),
                descr: "first root".to_string(),
                run_on: "2026-03-14 08:00:00".to_string(),
            },
            RunConfigEntry {
                id: 2,
                run_config_type: "root_b".to_string(),
                descr: "second root".to_string(),
                run_on: "2026-03-14 09:00:00".to_string(),
            },
            RunConfigEntry {
                id: 3,
                run_config_type: "shared".to_string(),
                descr: "shared child".to_string(),
                run_on: "2026-03-14 10:00:00".to_string(),
            },
        ];
        app.search_filters = RunConfigSearch {
            descr: "shared".to_string(),
            ..RunConfigSearch::default()
        };
        app.rebuild_run_config_tree(&[(3, 1), (3, 2)], &HashMap::new());
        app.rebuild_visible_run_configs(Some(3));

        assert_eq!(1, app.run_configs.iter().filter(|row| row.run_config_id() == Some(1)).count());
        assert_eq!(1, app.run_configs.iter().filter(|row| row.run_config_id() == Some(2)).count());
        assert_eq!(1, app.run_configs.iter().filter(|row| row.run_config_id() == Some(3)).count());
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_refresh_run_configs_uses_paged_startup_loading() {
        let mut store = MockStore::new();
        store.expect_list_run_configs_page().times(1).returning(|limit, offset| {
            assert!(limit >= 1);
            assert_eq!(0, offset);
            Ok(vec![
                RunConfigEntry {
                    id: 2,
                    run_config_type: "child".to_string(),
                    descr: "child run".to_string(),
                    run_on: "2026-03-14 09:00:00".to_string(),
                },
                RunConfigEntry {
                    id: 1,
                    run_config_type: "root".to_string(),
                    descr: "root run".to_string(),
                    run_on: "2026-03-14 08:00:00".to_string(),
                },
            ])
        });
        store.expect_list_run_config_ids_with_outputs().times(1).returning(|ids| {
            assert_eq!(vec![2, 1], ids.to_vec());
            Ok(vec![1])
        });
        store.expect_list_input_run_config_ids().times(1).returning(|ids| {
            assert_eq!(vec![2, 1], ids.to_vec());
            Ok(vec![(2, 1)])
        });
        let mut app = App::new(Box::new(store) as Box<dyn Store>, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));

        app.refresh_run_configs().await;

        assert_eq!(2, app.all_run_configs.len());
        assert_eq!(3, app.run_configs.len());
        assert!(!app.has_more_paged_run_configs);
        assert_eq!(Some("2026-03-14"), app.run_configs.first().and_then(VisibleRunConfigRow::run_on_date));
        assert_eq!(Some(1), app.run_configs.get(1).and_then(VisibleRunConfigRow::run_config_id));
        assert_eq!(Some(2), app.run_configs.get(2).and_then(VisibleRunConfigRow::run_config_id));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_refresh_run_configs_marks_visible_leaf_without_lazy_triangle() {
        let mut store = MockStore::new();
        store.expect_list_run_configs_page().times(1).returning(|limit, offset| {
            assert!(limit >= 1);
            assert_eq!(0, offset);
            Ok(vec![RunConfigEntry {
                id: 7,
                run_config_type: "SelectMotifsRunConfig".to_string(),
                descr: "leaf".to_string(),
                run_on: "2026-03-14 09:00:00".to_string(),
            }])
        });
        store.expect_list_run_config_ids_with_outputs().times(1).returning(|ids| {
            assert_eq!(vec![7], ids.to_vec());
            Ok(vec![])
        });
        store.expect_list_input_run_config_ids().times(1).returning(|ids| {
            assert_eq!(vec![7], ids.to_vec());
            Ok(vec![])
        });
        let mut app = App::new(Box::new(store) as Box<dyn Store>, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));

        app.refresh_run_configs().await;

        assert_eq!(2, app.run_configs.len());
        assert_eq!(Some("2026-03-14"), app.run_configs[0].run_on_date());
        let row = app.run_configs.get(1).expect("row");
        assert!(!row.has_children);
        assert!(app.run_config_tree.get(&7).map(|node| node.children_loaded).unwrap_or(false));
        assert!(format_run_tree_row(&app, row).trim_start().starts_with("•"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_expand_selected_run_lazy_loads_children() {
        let mut store = MockStore::new();
        store.expect_list_output_run_config_ids().times(1).returning(|run_config_id| {
            assert_eq!(1, run_config_id);
            Ok(vec![2])
        });
        store.expect_list_run_configs_by_ids().times(1).returning(|ids| {
            assert_eq!(vec![2], ids.to_vec());
            Ok(vec![RunConfigEntry {
                id: 2,
                run_config_type: "child".to_string(),
                descr: "child run".to_string(),
                run_on: "2026-03-14 09:00:00".to_string(),
            }])
        });
        store.expect_list_input_run_config_ids().times(1).returning(|ids| {
            assert_eq!(vec![2], ids.to_vec());
            Ok(vec![(2, 1)])
        });
        let mut app = App::new(Box::new(store) as Box<dyn Store>, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));
        app.has_more_paged_run_configs = true;
        app.all_run_configs = vec![RunConfigEntry {
            id: 1,
            run_config_type: "root".to_string(),
            descr: "root run".to_string(),
            run_on: "2026-03-14 08:00:00".to_string(),
        }];
        app.known_run_config_ids_with_outputs.insert(1);
        app.rebuild_run_config_tree(&[], &HashMap::new());
        app.rebuild_visible_run_configs(Some(1));

        assert!(app.expand_selected_run().await);
        assert_eq!(3, app.run_configs.len());
        assert_eq!(Some(2), app.run_configs.get(2).and_then(VisibleRunConfigRow::run_config_id));
        assert!(app.run_config_tree.get(&1).map(|node| node.children_loaded).unwrap_or(false));
        assert!(app.run_config_tree.get(&1).map(|node| node.expanded).unwrap_or(false));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_refresh_selected_details_skips_motif_loading_until_requested() {
        let mut store = MockStore::new();
        store.expect_list_run_config_values().times(1).returning(|run_config_id| {
            assert_eq!(7, run_config_id);
            Ok(HashMap::new())
        });
        let mut app = App::new(Box::new(store) as Box<dyn Store>, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));
        app.has_more_paged_run_configs = false;
        app.all_run_configs = vec![RunConfigEntry {
            id: 7,
            run_config_type: "SelectMotifsRunConfig".to_string(),
            descr: "motif selection".to_string(),
            run_on: "2026-03-14 08:00:00".to_string(),
        }];
        app.rebuild_run_config_tree(&[], &HashMap::new());
        app.rebuild_visible_run_configs(Some(7));

        app.refresh_selected_details().await;

        assert!(app.selected_run_motifs.is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_load_selected_run_motifs_fetches_on_demand() {
        let mut store = MockStore::new();
        store.expect_list_run_motifs().times(1).returning(|run_config_id| {
            assert_eq!(7, run_config_id);
            Ok(vec![RunMotifEntry {
                motif_id: 11,
                channel_id: 2,
                score: Some(0.4),
                source_run_config_id: 6,
                slices_cnt: 5,
                resources_cnt: 3,
            }])
        });
        let mut app = App::new(Box::new(store) as Box<dyn Store>, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));
        app.has_more_paged_run_configs = false;
        app.all_run_configs = vec![RunConfigEntry {
            id: 7,
            run_config_type: "SelectMotifsRunConfig".to_string(),
            descr: "motif selection".to_string(),
            run_on: "2026-03-14 08:00:00".to_string(),
        }];
        app.rebuild_run_config_tree(&[], &HashMap::new());
        app.rebuild_visible_run_configs(Some(7));

        assert!(app.load_selected_run_motifs().await);
        assert_eq!(1, app.selected_run_motifs.len());
        assert_eq!(11, app.selected_run_motifs[0].motif_id);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_open_selected_run_motifs_opens_motifs_for_leaf() {
        let mut store = MockStore::new();
        store.expect_list_run_motifs().times(1).returning(|run_config_id| {
            assert_eq!(86439, run_config_id);
            Ok(vec![RunMotifEntry {
                motif_id: 11,
                channel_id: 2,
                score: Some(0.4),
                source_run_config_id: 86434,
                slices_cnt: 5,
                resources_cnt: 3,
            }])
        });
        let mut app = App::new(Box::new(store) as Box<dyn Store>, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));
        app.has_more_paged_run_configs = false;
        app.all_run_configs = vec![RunConfigEntry {
            id: 86439,
            run_config_type: "SelectMotifsRunConfig".to_string(),
            descr: "motif selection".to_string(),
            run_on: "2026-03-18 18:48:40".to_string(),
        }];
        app.rebuild_run_config_tree(&[], &HashMap::new());
        app.rebuild_visible_run_configs(Some(86439));

        assert!(app.open_selected_run_motifs().await);
        assert_eq!(app.selected_area, SelectedArea::RightMotifs);
        assert!(app.prefer_right_motifs);
        assert_eq!(1, app.selected_run_motifs.len());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_toggle_selected_run_motifs_panel_hides_loaded_motifs() {
        let mut store = MockStore::new();
        store.expect_list_run_motifs().times(1).returning(|run_config_id| {
            assert_eq!(86439, run_config_id);
            Ok(vec![RunMotifEntry {
                motif_id: 11,
                channel_id: 2,
                score: Some(0.4),
                source_run_config_id: 86434,
                slices_cnt: 5,
                resources_cnt: 3,
            }])
        });
        let mut app = App::new(Box::new(store) as Box<dyn Store>, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));
        app.has_more_paged_run_configs = false;
        app.all_run_configs = vec![RunConfigEntry {
            id: 86439,
            run_config_type: "SelectMotifsRunConfig".to_string(),
            descr: "motif selection".to_string(),
            run_on: "2026-03-18 18:48:40".to_string(),
        }];
        app.rebuild_run_config_tree(&[], &HashMap::new());
        app.rebuild_visible_run_configs(Some(86439));

        assert!(app.toggle_selected_run_motifs_panel().await);
        assert!(app.motif_panel_visible());
        assert_eq!(app.selected_area, SelectedArea::RightMotifs);
        assert_eq!(1, app.selected_run_motifs.len());

        assert!(app.toggle_selected_run_motifs_panel().await);
        assert!(!app.motif_panel_visible());
        assert_eq!(app.selected_area, SelectedArea::LeftRunConfigs);
        assert_eq!(1, app.selected_run_motifs.len());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_open_selected_run_motifs_opens_motifs_even_with_children() {
        let mut store = MockStore::new();
        store.expect_list_run_motifs().times(1).returning(|run_config_id| {
            assert_eq!(86439, run_config_id);
            Ok(vec![RunMotifEntry {
                motif_id: 11,
                channel_id: 2,
                score: Some(0.4),
                source_run_config_id: 86434,
                slices_cnt: 5,
                resources_cnt: 3,
            }])
        });
        let mut app = App::new(Box::new(store) as Box<dyn Store>, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));
        app.has_more_paged_run_configs = false;
        app.all_run_configs = vec![
            RunConfigEntry {
                id: 86439,
                run_config_type: "SelectMotifsRunConfig".to_string(),
                descr: "motif selection".to_string(),
                run_on: "2026-03-18 18:48:40".to_string(),
            },
            RunConfigEntry {
                id: 86440,
                run_config_type: "TrainMotifNetRunConfig".to_string(),
                descr: "train motif nets".to_string(),
                run_on: "2026-03-18 18:49:40".to_string(),
            },
        ];
        app.rebuild_run_config_tree(&[(86440, 86439)], &HashMap::new());
        if let Some(node) = app.run_config_tree.get_mut(&86439) {
            node.expanded = false;
        }
        app.rebuild_visible_run_configs(Some(86439));

        assert!(app.open_selected_run_motifs().await);
        assert_eq!(app.selected_area, SelectedArea::RightMotifs);
        assert!(app.prefer_right_motifs);
        assert_eq!(1, app.selected_run_motifs.len());
        assert!(!app.run_config_tree.get(&86439).map(|node| node.expanded).unwrap_or(true));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_open_selected_run_from_left_expands_find_motifs_inline() {
        let mut store = MockStore::new();
        store.expect_list_run_motifs().times(1).returning(|run_config_id| {
            assert_eq!(86434, run_config_id);
            Ok(vec![RunMotifEntry {
                motif_id: 123,
                channel_id: 1,
                score: None,
                source_run_config_id: 86434,
                slices_cnt: 20,
                resources_cnt: 7,
            }])
        });
        let mut app = App::new(Box::new(store) as Box<dyn Store>, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));
        app.has_more_paged_run_configs = false;
        app.all_run_configs = vec![RunConfigEntry {
            id: 86434,
            run_config_type: "FindMotifsRunConfig".to_string(),
            descr: "find motifs".to_string(),
            run_on: "2026-03-18 18:29:59".to_string(),
        }];
        app.rebuild_run_config_tree(&[], &HashMap::new());
        app.rebuild_visible_run_configs(Some(86434));

        assert!(app.open_selected_run_from_left().await);
        assert_eq!(app.selected_area, SelectedArea::LeftRunConfigs);
        assert_eq!(3, app.run_configs.len());
        assert!(app.run_config_tree.get(&86434).map(|node| node.expanded).unwrap_or(false));
        assert!(app.run_configs.get(2).and_then(VisibleRunConfigRow::tree_motif).is_some());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_open_selected_run_right_expands_find_motifs_inline() {
        let mut store = MockStore::new();
        store.expect_list_run_motifs().times(1).returning(|run_config_id| {
            assert_eq!(86434, run_config_id);
            Ok(vec![RunMotifEntry {
                motif_id: 123,
                channel_id: 1,
                score: None,
                source_run_config_id: 86434,
                slices_cnt: 20,
                resources_cnt: 7,
            }])
        });
        let mut app = App::new(Box::new(store) as Box<dyn Store>, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));
        app.has_more_paged_run_configs = false;
        app.all_run_configs = vec![RunConfigEntry {
            id: 86434,
            run_config_type: "FindMotifsRunConfig".to_string(),
            descr: "find motifs".to_string(),
            run_on: "2026-03-18 18:29:59".to_string(),
        }];
        app.rebuild_run_config_tree(&[], &HashMap::new());
        app.rebuild_visible_run_configs(Some(86434));

        assert!(app.open_selected_run_right().await);
        assert_eq!(app.selected_area, SelectedArea::LeftRunConfigs);
        assert_eq!(3, app.run_configs.len());
        assert!(app.run_config_tree.get(&86434).map(|node| node.expanded).unwrap_or(false));
        assert!(app.run_configs.get(2).and_then(VisibleRunConfigRow::tree_motif).is_some());
    }

    #[test]
    fn test_move_main_to_bottom_updates_offset() {
        let store = Box::new(MockStore::new()) as Box<dyn Store>;
        let mut app = App::new(store, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));
        app.run_configs = (0..30)
            .map(|idx| VisibleRunConfigRow::run_config(idx as u32, 0, false, true))
            .collect();

        assert!(app.move_main_to_bottom());
        assert_eq!(app.selected_idx, 29);
        assert_eq!(app.main_list_offset, 9);
    }

    #[test]
    fn test_move_main_up_keeps_offset_until_top_visible_row() {
        let store = Box::new(MockStore::new()) as Box<dyn Store>;
        let mut app = App::new(store, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));
        app.run_configs = (0..30)
            .map(|idx| VisibleRunConfigRow::run_config(idx as u32, 0, false, true))
            .collect();
        app.selected_idx = 15;
        app.main_list_offset = 9;

        assert!(app.move_main_up(1));
        assert_eq!(app.selected_idx, 14);
        assert_eq!(app.main_list_offset, 9);

        app.selected_idx = 9;
        app.main_list_offset = 9;

        assert!(app.move_main_up(1));
        assert_eq!(app.selected_idx, 8);
        assert_eq!(app.main_list_offset, 8);
    }

    #[test]
    fn test_progress_lines_include_device_stats() {
        let store = Box::new(MockStore::new()) as Box<dyn Store>;
        let mut app = App::new(store, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));
        app.progress_state = CollectEtaProgressState {
            invocations: vec![],
            started_at_unix_ms: 123,
            oom_all_failed: 2,
            oom_success: 1,
            device_stats: Some(ThoDeviceStats {
                allocated: 2048,
                active: 3072,
                reserved: 4096,
                requested: 1536,
                non_releasable: 512,
            }),
        };

        let lines = app.progress_lines();

        assert!(lines.iter().any(|line| line == "gpu allocated: 2.0 KiB"));
        assert!(lines.iter().any(|line| line == "gpu reusable cached: 1.0 KiB"));
        assert!(lines.iter().any(|line| line == "gpu pending free: 1.0 KiB"));
        assert!(lines.iter().any(|line| line == "gpu rounding overhead: 512 B"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_progress_client_receives_snapshot() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        static NEXT_TEST_SOCKET_ID: AtomicU64 = AtomicU64::new(1);

        let socket_id = format!("thoosa-progress-tui-test-{}-{}", process::id(), NEXT_TEST_SOCKET_ID.fetch_add(1, Ordering::Relaxed));
        let listener = create_test_listener(&socket_id)?;
        let (tx, mut rx) = mpsc::unbounded_channel();
        let client_task = spawn_progress_client(socket_id.clone(), tx);
        let server_task = tokio::spawn(async move {
            let stream = listener.accept().await?;
            let snapshot = CollectEtaProgressState {
                invocations: vec![CollectEtaProgressInvocation {
                    descr: "apply_label_nets".to_string(),
                    total: 9,
                    completed: 4,
                }],
                started_at_unix_ms: 123,
                oom_all_failed: 2,
                oom_success: 0,
                device_stats: Some(ThoDeviceStats {
                    allocated: 1024,
                    active: 2048,
                    reserved: 4096,
                    requested: 512,
                    non_releasable: 256,
                }),
            };
            let mut writer = &stream;
            let mut bytes = serde_json::to_vec(&snapshot)?;
            bytes.push(b'\n');
            writer.write_all(&bytes).await?;
            writer.flush().await?;
            Ok::<_, Box<dyn std::error::Error + Send + Sync>>(())
        });

        let mut received = None;
        for _ in 0..6 {
            let Some(event) = tokio::time::timeout(Duration::from_secs(2), rx.recv()).await? else {
                break;
            };
            if let ProgressClientEvent::Snapshot(snapshot) = event {
                received = Some(snapshot);
                break;
            }
        }

        client_task.abort();
        let _ = client_task.await;
        server_task.await??;

        assert_eq!(
            received,
            Some(CollectEtaProgressState {
                invocations: vec![CollectEtaProgressInvocation {
                    descr: "apply_label_nets".to_string(),
                    total: 9,
                    completed: 4,
                }],
                started_at_unix_ms: 123,
                oom_all_failed: 2,
                oom_success: 0,
                device_stats: Some(ThoDeviceStats {
                    allocated: 1024,
                    active: 2048,
                    reserved: 4096,
                    requested: 512,
                    non_releasable: 256,
                }),
            })
        );

        Ok(())
    }

    fn create_test_listener(socket_id: &str) -> io::Result<interprocess::local_socket::tokio::Listener> {
        let mut last_err = None;
        for name in progress_socket_names(socket_id)? {
            match ListenerOptions::new().name(name).try_overwrite(true).create_tokio() {
                Ok(listener) => return Ok(listener),
                Err(err) => {
                    last_err = Some(err);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| io::Error::other("no local socket names available")))
    }

}
