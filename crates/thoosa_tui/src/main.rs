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
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use thoosa_shared::collect_futures_eta::{CollectEtaProgressInvocation, CollectEtaProgressState, COLLECT_ETA_PROGRESS_SOCKET_ID};
use thoosa_shared::val::Val;
use thoosa_shared::ThoError;
use thoosa_store::{RunConfigEntry, RunMotifEntry, SqlStore, Store};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

const SEARCH_TYPE_NONE_LABEL: &str = "no value";
const RUN_ON_DATE_FORMAT: &str = "%Y-%m-%d";
const RUN_ON_DATETIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S";
const DEFAULT_PLOT_PROJECT: &str = "thoosa_hbac";

#[derive(Clone, Debug)]
struct RunTreeNode {
    entry: RunConfigEntry,
    input_run_ids: Vec<u32>,
    output_run_ids: Vec<u32>,
    children_loaded: bool,
    expanded: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VisibleRunConfigRow {
    run_id: u32,
    depth: usize,
    has_children: bool,
    expanded: bool,
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
        let id = parse_run_id_filter(&self.id)?;
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
    known_output_run_ids: HashMap<u32, Vec<u32>>,
    paged_run_config_offset: usize,
    has_more_paged_run_configs: bool,
    run_config_tree: HashMap<u32, RunTreeNode>,
    run_configs: Vec<VisibleRunConfigRow>,
    main_list_offset: usize,
    selected_idx: usize,
    selected_values: Vec<(String, String)>,
    selected_run_motifs: Vec<RunMotifEntry>,
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
    progress_state: CollectEtaProgressState,
    progress_status: String,
    progress_last_update: Option<Instant>,
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
            known_output_run_ids: HashMap::new(),
            paged_run_config_offset: 0,
            has_more_paged_run_configs: true,
            run_config_tree: HashMap::new(),
            run_configs: vec![],
            main_list_offset: 0,
            selected_idx: 0,
            selected_values: vec![],
            selected_run_motifs: vec![],
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
            progress_state: CollectEtaProgressState::default(),
            progress_status: "Waiting for progress socket.".to_string(),
            progress_last_update: None,
        }
    }

    fn selected_run_id(&self) -> Option<u32> {
        self.run_configs.get(self.selected_idx).map(|entry| entry.run_id)
    }

    fn selected_run_entry(&self) -> Option<&RunConfigEntry> {
        self.selected_run_id()
            .and_then(|run_id| self.run_config_tree.get(&run_id).map(|node| &node.entry))
    }

    fn selected_motif(&self) -> Option<&RunMotifEntry> {
        self.selected_run_motifs.get(self.selected_motif_idx)
    }

    fn has_selected_run_motifs(&self) -> bool {
        !self.selected_run_motifs.is_empty()
    }

    fn selected_popup_type(&self) -> Option<String> {
        self.run_config_types.get(self.type_popup_selected_idx).cloned()
    }

    fn selected_search_field(&self) -> SearchField {
        SearchField::ALL
            .get(self.search_popup.selected_field_idx)
            .copied()
            .unwrap_or(SearchField::Id)
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

    fn main_page_size(&self) -> usize {
        let rows = terminal::size().map(|(_, r)| r).unwrap_or(24);
        rows.saturating_sub(3).max(1) as usize
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

    async fn load_next_run_config_page(&mut self, keep_run_id: Option<u32>) -> Result<bool, ThoError> {
        if !self.has_more_paged_run_configs {
            return Ok(false);
        }

        let limit = self.run_config_batch_size();
        let page = self.store.list_run_configs_page(limit, self.paged_run_config_offset).await?;
        if page.is_empty() {
            self.has_more_paged_run_configs = false;
            let loaded_input_pairs = self.loaded_input_pairs.clone();
            let known_output_run_ids = self.known_output_run_ids.clone();
            self.rebuild_run_config_tree(&loaded_input_pairs, &known_output_run_ids);
            self.rebuild_visible_run_configs(keep_run_id);
            return Ok(false);
        }

        let page_ids = page.iter().map(|entry| entry.id).collect::<Vec<u32>>();
        let input_pairs = if page_ids.is_empty() {
            vec![]
        } else {
            self.store.list_input_run_ids(&page_ids).await?
        };

        self.paged_run_config_offset += page.len();
        self.has_more_paged_run_configs = page.len() == limit;
        self.merge_run_configs(page);
        self.loaded_input_pairs.extend(input_pairs);
        let loaded_input_pairs = self.loaded_input_pairs.clone();
        let known_output_run_ids = self.known_output_run_ids.clone();
        self.rebuild_run_config_tree(&loaded_input_pairs, &known_output_run_ids);
        self.rebuild_visible_run_configs(keep_run_id);
        Ok(true)
    }

    async fn ensure_all_run_configs_loaded(&mut self, keep_run_id: Option<u32>) -> Result<(), ThoError> {
        while self.has_more_paged_run_configs {
            if !self.load_next_run_config_page(keep_run_id).await? {
                break;
            }
        }
        Ok(())
    }

    async fn ensure_run_ids_loaded(&mut self, run_ids: &[u32], keep_run_id: Option<u32>) -> Result<bool, ThoError> {
        let existing_ids = self.all_run_configs.iter().map(|entry| entry.id).collect::<HashSet<u32>>();
        let missing_ids = run_ids
            .iter()
            .copied()
            .filter(|run_id| !existing_ids.contains(run_id))
            .collect::<Vec<u32>>();
        if missing_ids.is_empty() {
            return Ok(false);
        }

        let run_configs = self.store.list_run_configs_by_ids(&missing_ids).await?;
        let input_pairs = self.store.list_input_run_ids(&missing_ids).await?;
        self.merge_run_configs(run_configs);
        self.loaded_input_pairs.extend(input_pairs);
        let loaded_input_pairs = self.loaded_input_pairs.clone();
        let known_output_run_ids = self.known_output_run_ids.clone();
        self.rebuild_run_config_tree(&loaded_input_pairs, &known_output_run_ids);
        self.rebuild_visible_run_configs(keep_run_id);
        Ok(true)
    }

    async fn ensure_run_children_loaded(&mut self, run_id: u32, keep_run_id: Option<u32>) -> Result<bool, ThoError> {
        let output_run_ids = self.store.list_output_run_ids(run_id).await?;
        self.known_output_run_ids.insert(run_id, output_run_ids.clone());
        let loaded_any = self.ensure_run_ids_loaded(&output_run_ids, keep_run_id).await?;
        if !loaded_any {
            let loaded_input_pairs = self.loaded_input_pairs.clone();
            let known_output_run_ids = self.known_output_run_ids.clone();
            self.rebuild_run_config_tree(&loaded_input_pairs, &known_output_run_ids);
            self.rebuild_visible_run_configs(keep_run_id);
        }
        Ok(!output_run_ids.is_empty())
    }

    async fn maybe_load_more_run_configs(&mut self) -> Result<bool, ThoError> {
        if self.search_filters.is_empty()
            && self.has_more_paged_run_configs
            && self.selected_idx + self.main_page_size() >= self.run_configs.len()
        {
            return self.load_next_run_config_page(self.selected_run_id()).await;
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
            if let Err(err) = self.maybe_load_more_run_configs().await {
                warn!("move_main_down_lazy failed to load more run configs: {err:?}");
                self.status = Some(format!("Failed to load more run_configs: {err:?}"));
            }
            return true;
        }

        match self.maybe_load_more_run_configs().await {
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
            if let Err(err) = self.ensure_all_run_configs_loaded(self.selected_run_id()).await {
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
        move_down(&mut self.search_popup.selected_field_idx, SearchField::ALL.len(), step)
    }

    fn move_search_popup_up(&mut self, step: usize) -> bool {
        move_up(&mut self.search_popup.selected_field_idx, SearchField::ALL.len(), step)
    }

    fn move_search_popup_to_top(&mut self) -> bool {
        move_to_top(&mut self.search_popup.selected_field_idx, SearchField::ALL.len())
    }

    fn move_search_popup_to_bottom(&mut self) -> bool {
        move_to_bottom(&mut self.search_popup.selected_field_idx, SearchField::ALL.len())
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
        self.run_config_types = self.store.list_run_config_types().await?;
        Ok(())
    }

    async fn open_search_popup(&mut self) {
        if let Err(err) = self.refresh_run_config_types().await {
            warn!("open_search_popup failed to refresh run_config types: {err:?}");
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
        self.status = Some("Search draft reset.".to_string());
    }

    async fn apply_search_from_popup(&mut self) -> bool {
        match self.search_popup.draft.normalized() {
            Ok(_) => {
                let keep_run_id = self.selected_run_id();
                self.search_filters = self.search_popup.draft.clone();
                if !self.search_filters.is_empty() {
                    if let Err(err) = self.ensure_all_run_configs_loaded(keep_run_id).await {
                        warn!("apply_search_from_popup failed to load all run configs: {err:?}");
                        self.status = Some(format!("Failed to list run_configs for search: {err:?}"));
                        return false;
                    }
                }
                self.rebuild_visible_run_configs(keep_run_id);
                self.selected_area = SelectedArea::LeftRunConfigs;
                self.prefer_right_motifs = false;
                self.status = Some(format!("Run config filters: {}", self.search_filters.summary()));
                true
            }
            Err(err) => {
                self.status = Some(format!("Invalid search: {err}"));
                false
            }
        }
    }

    fn append_to_search_popup(&mut self, ch: char) -> bool {
        match self.selected_search_field() {
            SearchField::Id => {
                if ch.is_ascii_digit() {
                    self.search_popup.draft.id.push(ch);
                    true
                } else {
                    false
                }
            }
            SearchField::Type => false,
            SearchField::Descr => {
                self.search_popup.draft.descr.push(ch);
                true
            }
            SearchField::RunOnFrom => append_run_on_char(&mut self.search_popup.draft.run_on_from, ch),
            SearchField::RunOnTo => append_run_on_char(&mut self.search_popup.draft.run_on_to, ch),
        }
    }

    fn pop_from_search_popup(&mut self) -> bool {
        match self.selected_search_field() {
            SearchField::Id => self.search_popup.draft.id.pop().is_some(),
            SearchField::Type => {
                let had_value = self.search_popup.draft.run_config_type.is_some();
                self.search_popup.draft.run_config_type = None;
                self.search_popup.type_selected_idx = 0;
                had_value
            }
            SearchField::Descr => self.search_popup.draft.descr.pop().is_some(),
            SearchField::RunOnFrom => self.search_popup.draft.run_on_from.pop().is_some(),
            SearchField::RunOnTo => self.search_popup.draft.run_on_to.pop().is_some(),
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
        lines.push(format!("failed OOM runs: {}", self.progress_state.oom_all_failed));
        match self.progress_last_update {
            Some(last_update) => {
                lines.push(format!("last update: {}ms ago", last_update.elapsed().as_millis()));
            }
            None => {
                lines.push("last update: <none>".to_string());
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

    fn rebuild_run_config_tree(&mut self, input_pairs: &[(u32, u32)], outputs_by_run_id: &HashMap<u32, Vec<u32>>) {
        let old_expanded = self
            .run_config_tree
            .iter()
            .map(|(run_id, node)| (*run_id, node.expanded))
            .collect::<HashMap<u32, bool>>();
        let entries_by_id = self
            .all_run_configs
            .iter()
            .cloned()
            .map(|entry| (entry.id, entry))
            .collect::<HashMap<u32, RunConfigEntry>>();
        let mut input_map = HashMap::<u32, HashSet<u32>>::new();
        let mut output_map = HashMap::<u32, HashSet<u32>>::new();

        for (run_id, input_run_id) in input_pairs {
            if entries_by_id.contains_key(run_id) && entries_by_id.contains_key(input_run_id) {
                input_map.entry(*run_id).or_default().insert(*input_run_id);
                output_map.entry(*input_run_id).or_default().insert(*run_id);
            }
        }

        for (run_id, output_run_ids) in outputs_by_run_id {
            if !entries_by_id.contains_key(run_id) {
                continue;
            }

            let outputs = output_map.entry(*run_id).or_default();
            for output_run_id in output_run_ids {
                if entries_by_id.contains_key(output_run_id) {
                    outputs.insert(*output_run_id);
                }
            }
        }

        self.run_config_tree = entries_by_id
            .into_iter()
            .map(|(run_id, entry)| {
                let mut input_run_ids = input_map.remove(&run_id).unwrap_or_default().into_iter().collect::<Vec<u32>>();
                let mut output_run_ids = output_map.remove(&run_id).unwrap_or_default().into_iter().collect::<Vec<u32>>();
                let children_loaded = outputs_by_run_id.contains_key(&run_id) || !self.has_more_paged_run_configs;
                let default_expanded = !output_run_ids.is_empty();
                sort_run_ids(&mut input_run_ids, &self.all_run_configs, self.run_on_sort_order);
                sort_run_ids(&mut output_run_ids, &self.all_run_configs, self.run_on_sort_order);
                (
                    run_id,
                    RunTreeNode {
                        entry,
                        input_run_ids,
                        output_run_ids,
                        children_loaded,
                        expanded: old_expanded.get(&run_id).copied().unwrap_or(default_expanded),
                    },
                )
            })
            .collect();
    }

    fn run_visible_with_descendants(
        &self,
        run_id: u32,
        filter_active: bool,
        matching_run_ids: &HashSet<u32>,
        memo: &mut HashMap<u32, bool>,
        path: &mut HashSet<u32>,
    ) -> bool {
        if let Some(visible) = memo.get(&run_id) {
            return *visible;
        }

        if !path.insert(run_id) {
            return false;
        }

        let visible = !filter_active
            || matching_run_ids.contains(&run_id)
            || self
                .run_config_tree
                .get(&run_id)
                .map(|node| {
                    node.output_run_ids.iter().any(|child_run_id| {
                        self.run_visible_with_descendants(*child_run_id, filter_active, matching_run_ids, memo, path)
                    })
                })
                .unwrap_or(false);
        path.remove(&run_id);
        memo.insert(run_id, visible);
        visible
    }

    fn append_visible_run_rows(
        &self,
        run_id: u32,
        depth: usize,
        filter_active: bool,
        matching_run_ids: &HashSet<u32>,
        memo: &mut HashMap<u32, bool>,
        path: &mut HashSet<u32>,
        rows: &mut Vec<VisibleRunConfigRow>,
    ) {
        if !path.insert(run_id) {
            return;
        }

        let Some(node) = self.run_config_tree.get(&run_id) else {
            path.remove(&run_id);
            return;
        };
        let visible_children = node
            .output_run_ids
            .iter()
            .filter(|child_run_id| self.run_visible_with_descendants(**child_run_id, filter_active, matching_run_ids, memo, path))
            .copied()
            .collect::<Vec<u32>>();

        rows.push(VisibleRunConfigRow {
            run_id,
            depth,
            has_children: !visible_children.is_empty() || !node.children_loaded,
            expanded: node.expanded,
        });

        if node.expanded {
            for child_run_id in visible_children {
                self.append_visible_run_rows(child_run_id, depth + 1, filter_active, matching_run_ids, memo, path, rows);
            }
        }

        path.remove(&run_id);
    }

    fn rebuild_visible_run_configs(&mut self, keep_run_id: Option<u32>) {
        let filter_active = !self.search_filters.is_empty();
        let matching_run_ids = if filter_active {
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

        let mut root_run_ids = self
            .run_config_tree
            .iter()
            .filter_map(|(run_id, node)| {
                if node
                    .input_run_ids
                    .iter()
                    .all(|input_run_id| !self.run_config_tree.contains_key(input_run_id))
                {
                    Some(*run_id)
                } else {
                    None
                }
            })
            .collect::<Vec<u32>>();
        sort_run_ids(&mut root_run_ids, &self.all_run_configs, self.run_on_sort_order);

        let mut memo = HashMap::new();
        let mut rows = vec![];
        for run_id in root_run_ids {
            if self.run_visible_with_descendants(run_id, filter_active, &matching_run_ids, &mut memo, &mut HashSet::new()) {
                self.append_visible_run_rows(run_id, 0, filter_active, &matching_run_ids, &mut memo, &mut HashSet::new(), &mut rows);
            }
        }

        let appended_run_ids = rows.iter().map(|row| row.run_id).collect::<HashSet<u32>>();
        let mut remaining_run_ids = self
            .run_config_tree
            .keys()
            .filter(|run_id| !appended_run_ids.contains(run_id))
            .copied()
            .collect::<Vec<u32>>();
        sort_run_ids(&mut remaining_run_ids, &self.all_run_configs, self.run_on_sort_order);
        for run_id in remaining_run_ids {
            if self.run_visible_with_descendants(run_id, filter_active, &matching_run_ids, &mut memo, &mut HashSet::new()) {
                self.append_visible_run_rows(run_id, 0, filter_active, &matching_run_ids, &mut memo, &mut HashSet::new(), &mut rows);
            }
        }

        self.run_configs = rows;

        if let Some(run_id) = keep_run_id {
            if let Some(idx) = self.run_configs.iter().position(|entry| entry.run_id == run_id) {
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
        let Some(run_id) = self.selected_run_id() else {
            return false;
        };
        let needs_children_load = self
            .run_config_tree
            .get(&run_id)
            .map(|node| !node.children_loaded)
            .unwrap_or(false);
        if needs_children_load {
            match self.ensure_run_children_loaded(run_id, Some(run_id)).await {
                Ok(false) => return false,
                Ok(true) => {}
                Err(e) => {
                    warn!("toggle_selected_run_expanded failed to lazy-load children: run_id={run_id} err={e:?}");
                    self.status = Some(format!("Failed to list child runs for run_id={run_id}: {e:?}"));
                    return false;
                }
            }
        }

        let Some(node) = self.run_config_tree.get_mut(&run_id) else {
            return false;
        };
        if node.output_run_ids.is_empty() {
            return false;
        }
        node.expanded = !node.expanded;
        self.rebuild_visible_run_configs(Some(run_id));
        true
    }

    async fn expand_selected_run(&mut self) -> bool {
        let Some(run_id) = self.selected_run_id() else {
            return false;
        };
        let needs_children_load = self
            .run_config_tree
            .get(&run_id)
            .map(|node| !node.children_loaded)
            .unwrap_or(false);
        if needs_children_load {
            match self.ensure_run_children_loaded(run_id, Some(run_id)).await {
                Ok(false) => return false,
                Ok(true) => {}
                Err(e) => {
                    warn!("expand_selected_run failed to lazy-load children: run_id={run_id} err={e:?}");
                    self.status = Some(format!("Failed to list child runs for run_id={run_id}: {e:?}"));
                    return false;
                }
            }
        }

        let Some(node) = self.run_config_tree.get_mut(&run_id) else {
            return false;
        };
        if node.output_run_ids.is_empty() || node.expanded {
            return false;
        }
        node.expanded = true;
        self.rebuild_visible_run_configs(Some(run_id));
        true
    }

    fn collapse_selected_run(&mut self) -> bool {
        let Some(run_id) = self.selected_run_id() else {
            return false;
        };
        let Some(node) = self.run_config_tree.get_mut(&run_id) else {
            return false;
        };
        if node.output_run_ids.is_empty() || !node.expanded {
            return false;
        }
        node.expanded = false;
        self.rebuild_visible_run_configs(Some(run_id));
        true
    }

    fn toggle_run_on_sort_order(&mut self) {
        self.run_on_sort_order = match self.run_on_sort_order {
            SortOrder::Asc => SortOrder::Desc,
            SortOrder::Desc => SortOrder::Asc,
        };

        let keep_run_id = self.selected_run_id();
        self.rebuild_visible_run_configs(keep_run_id);
        debug!("toggle_run_on_sort_order: order={}", self.sort_order_label());
    }

    fn reset_search_filters(&mut self) {
        if !self.search_filters.is_empty() {
            let keep_run_id = self.selected_run_id();
            self.search_filters = RunConfigSearch::default();
            self.search_popup = SearchPopupState::default();
            self.rebuild_visible_run_configs(keep_run_id);
            self.status = Some("Run config filters reset.".to_string());
            debug!("reset_search_filters");
        }
    }

    async fn apply_type_filter_from_popup(&mut self) {
        if let Some(selected_type) = self.selected_popup_type() {
            let keep_run_id = self.selected_run_id();
            self.search_filters.run_config_type = Some(selected_type.clone());
            if let Err(err) = self.ensure_all_run_configs_loaded(keep_run_id).await {
                warn!("apply_type_filter_from_popup failed to load all run configs: {err:?}");
                self.status = Some(format!("Failed to list run_configs for type filter: {err:?}"));
                return;
            }
            self.rebuild_visible_run_configs(keep_run_id);
            self.status = Some(format!("Type filter applied: {selected_type}"));
            debug!("apply_type_filter_from_popup: type={selected_type}");
        }
    }

    async fn open_type_filter_popup(&mut self) {
        match self.refresh_run_config_types().await {
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

    async fn refresh_run_configs(&mut self) {
        let keep_run_id = self.selected_run_id();
        self.all_run_configs.clear();
        self.loaded_input_pairs.clear();
        self.known_output_run_ids.clear();
        self.paged_run_config_offset = 0;
        self.has_more_paged_run_configs = true;
        self.run_config_tree.clear();
        self.run_configs.clear();

        let load_result = if self.search_filters.is_empty() {
            self.load_next_run_config_page(keep_run_id).await.map(|_| ())
        } else {
            self.ensure_all_run_configs_loaded(keep_run_id).await
        };

        match load_result {
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
                self.known_output_run_ids.clear();
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
        let keep_motif_id = self.selected_motif().map(|motif| motif.motif_id);
        self.selected_values.clear();
        self.selected_run_motifs.clear();

        let Some(run_id) = self.selected_run_id() else {
            if self.selected_area == SelectedArea::RightMotifs {
                self.selected_area = SelectedArea::LeftRunConfigs;
                self.prefer_right_motifs = false;
            }
            if self.status.is_none() {
                self.status = Some("No run_configs found.".to_string());
            }
            return;
        };

        match self.store.list_run_config_values(run_id).await {
            Ok(values) => {
                self.status = None;
                let mut rows = values
                    .into_iter()
                    .map(|(name, value)| (name, value_to_string(&value)))
                    .collect::<Vec<(String, String)>>();
                rows.sort_by(|a, b| a.0.cmp(&b.0));
                debug!("refresh_selected_values: run_id={} rows={}", run_id, rows.len());
                self.selected_values = rows;
            }
            Err(e) => {
                warn!("refresh_selected_details failed to list values: run_id={} err={e:?}", run_id);
                self.status = Some(format!("Failed to list run_config_values for run_id={run_id}: {e:?}"));
            }
        }

        match self.store.list_run_motifs(run_id).await {
            Ok(motifs) => {
                self.selected_run_motifs = motifs;
                if self.selected_run_motifs.is_empty() {
                    self.selected_motif_idx = 0;
                    self.motif_list_offset = 0;
                    if self.selected_area == SelectedArea::RightMotifs {
                        self.selected_area = SelectedArea::LeftRunConfigs;
                        self.prefer_right_motifs = false;
                    }
                } else {
                    self.selected_motif_idx = keep_motif_id
                        .and_then(|motif_id| self.selected_run_motifs.iter().position(|motif| motif.motif_id == motif_id))
                        .unwrap_or(0);
                    self.ensure_motif_selection_visible();
                }
            }
            Err(e) => {
                warn!("refresh_selected_details failed to list motifs: run_id={} err={e:?}", run_id);
                self.status = Some(format!("Failed to list motifs for run_id={run_id}: {e:?}"));
            }
        }
    }

    async fn open_selected_motif_plot(&mut self) {
        let Some(selected_motif) = self.selected_motif().cloned() else {
            self.status = Some("No motif selected.".to_string());
            return;
        };

        match resolve_motif_plot_target(&self.ws_path, selected_motif.motif_id).await {
            Ok(target_path) => {
                let target_path = if target_path.is_absolute() { target_path } else { self.ws_path.join(target_path) };
                if !target_path.exists() {
                    self.status = Some(format!("Generating plot for motif_id={}...", selected_motif.motif_id));
                    if let Err(err) = generate_motif_plot(&self.ws_path, selected_motif.motif_id).await {
                        self.status = Some(format!("Failed to generate plot for motif_id={}: {err}", selected_motif.motif_id));
                        return;
                    }
                }

                match open_path_in_system_viewer(&target_path) {
                    Ok(()) => {
                        self.status = Some(format!("Opened plot: {}", target_path.display()));
                    }
                    Err(err) => {
                        self.status = Some(format!("Failed to open plot {}: {err}", target_path.display()));
                    }
                }
            }
            Err(err) => {
                self.status = Some(format!("Failed to resolve plot for motif_id={}: {err}", selected_motif.motif_id));
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
    let store = SqlStore::from_path(&db_path).await.map_err(|e| io::Error::other(format!("{e:?}")))?;

    let mut app = App::new(store, args.ws_path.clone(), db_path);
    app.refresh_run_configs().await;
    app.refresh_selected_details().await;

    if !io::stdout().is_terminal() || !io::stdin().is_terminal() {
        warn!("no TTY detected, switching to non-interactive mode");
        return run_non_interactive(&app);
    }

    run_ui(&mut app).await
}

async fn run_ui(app: &mut App) -> Result<(), Box<dyn std::error::Error>> {
    let _terminal_guard = TerminalGuard::enter()?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;
    let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<ProgressClientEvent>();
    let progress_task = spawn_progress_client(COLLECT_ETA_PROGRESS_SOCKET_ID.to_string(), progress_tx);

    loop {
        drain_progress_events(app, &mut progress_rx);
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

                if key.code == KeyCode::Char('q') {
                    break;
                }

                if key.code == KeyCode::Char(' ') && key.modifiers.is_empty() && matches!(app.selected_area, SelectedArea::LeftRunConfigs | SelectedArea::RightMotifs | SelectedArea::ProgressView) {
                    app.open_view_popup();
                    continue;
                }

                if key.code == KeyCode::Char('/') && key.modifiers.is_empty() && app.selected_area == SelectedArea::LeftRunConfigs {
                    app.open_search_popup().await;
                    continue;
                }

                match app.selected_area {
                    SelectedArea::LeftRunConfigs => {
                        let mut selection_changed = false;

                        match key.code {
                            KeyCode::Char('r') if key.modifiers.is_empty() => {
                                app.refresh_run_configs().await;
                                app.refresh_selected_details().await;
                            }
                            KeyCode::Char('s') if key.modifiers.is_empty() => {
                                app.toggle_run_on_sort_order();
                                app.refresh_selected_details().await;
                            }
                            KeyCode::Char('t') if key.modifiers.is_empty() => {
                                app.open_type_filter_popup().await;
                            }
                            KeyCode::Char('T') if is_unmodified_or_shift(&key) => {
                                app.reset_search_filters();
                                app.refresh_selected_details().await;
                            }
                            KeyCode::Tab if key.modifiers.is_empty() => {
                                if app.has_selected_run_motifs() {
                                    app.selected_area = SelectedArea::RightMotifs;
                                    app.prefer_right_motifs = true;
                                }
                            }
                            KeyCode::Enter if key.modifiers.is_empty() => {
                                app.toggle_selected_run_expanded().await;
                            }
                            KeyCode::Right if key.modifiers.is_empty() => {
                                app.expand_selected_run().await;
                            }
                            KeyCode::Left if key.modifiers.is_empty() => {
                                app.collapse_selected_run();
                            }
                            KeyCode::Down | KeyCode::Char('j') if key.modifiers.is_empty() => {
                                selection_changed = app.move_main_down_lazy(1).await;
                            }
                            KeyCode::Up | KeyCode::Char('k') if key.modifiers.is_empty() => {
                                selection_changed = app.move_main_up(1);
                            }
                            KeyCode::Char('d') if is_ctrl_char(&key, 'd') => {
                                selection_changed = app.move_main_down_lazy(app.main_page_size()).await;
                            }
                            KeyCode::Char('u') if is_ctrl_char(&key, 'u') => {
                                selection_changed = app.move_main_up(app.main_page_size());
                            }
                            KeyCode::Char('G') if is_unmodified_or_shift(&key) => {
                                selection_changed = app.move_main_to_bottom_lazy().await;
                            }
                            KeyCode::Char('g') if key.modifiers.is_empty() => {
                                if app.handle_g_press(SelectedArea::LeftRunConfigs) {
                                    selection_changed = app.move_main_to_top();
                                }
                            }
                            _ => {}
                        }

                        if selection_changed {
                            app.refresh_selected_details().await;
                        }
                    }
                    SelectedArea::RightMotifs => {
                        match key.code {
                            KeyCode::Tab | KeyCode::Left if key.modifiers.is_empty() => {
                                app.selected_area = SelectedArea::LeftRunConfigs;
                                app.prefer_right_motifs = false;
                            }
                            KeyCode::Enter | KeyCode::Char('p') if key.modifiers.is_empty() => {
                                app.open_selected_motif_plot().await;
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
                                app.apply_type_filter_from_popup().await;
                                app.selected_area = app.default_run_configs_area();
                                app.refresh_selected_details().await;
                            }
                            KeyCode::Char('T') if is_unmodified_or_shift(&key) => {
                                app.reset_search_filters();
                                app.selected_area = app.default_run_configs_area();
                                app.refresh_selected_details().await;
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
                                if app.selected_search_field() == SearchField::Type {
                                    app.open_search_type_popup();
                                } else if app.apply_search_from_popup().await {
                                    app.refresh_selected_details().await;
                                }
                            }
                            KeyCode::Right | KeyCode::Left if app.selected_search_field() == SearchField::Type => {
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

    progress_task.abort();
    let _ = progress_task.await;
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

    let status_line = match app.selected_area {
        SelectedArea::LeftRunConfigs => format!(
            "area: run tree | sort(run_on): {} (s toggle) | j/k, Up/Down lazy-load | Ctrl-d/u page | gg top | G bottom | Enter toggle/load | Left/Right collapse/expand/load | Tab motifs | / search | t type | T reset filters | r reload | Space panels | q quit",
            app.sort_order_label()
        ),
        SelectedArea::RightMotifs => "area: motifs | j/k, Up/Down | Ctrl-d/u page | gg top | G bottom | Enter/p open plot | Tab/Left back | Space panels | q quit".to_string(),
        SelectedArea::ProgressView => "area: progress | Space panels | q quit".to_string(),
        SelectedArea::TypeFilterPopup => {
            "area: type-filter | j/k, Up/Down | Ctrl-d/u page | gg top | G bottom | Enter apply type | Esc close | T reset filters".to_string()
        }
        SelectedArea::SearchPopup => "area: search | j/k, Up/Down | type to edit | Backspace delete | Enter apply | Enter/Left/Right on type select | Ctrl-r reset draft | Esc close".to_string(),
        SelectedArea::SearchTypePopup => "area: search type | j/k, Up/Down | Ctrl-d/u page | gg top | G bottom | Enter choose | Esc close".to_string(),
        SelectedArea::ViewPopup => "area: panels | j/k, Up/Down | Ctrl-d/u page | gg top | G bottom | Enter apply | Esc close".to_string(),
    };
    frame.render_widget(Paragraph::new(status_line), outer[1]);
}

fn render_run_configs_view(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(48), Constraint::Percentage(52)])
        .split(area);

    let left_items = if app.run_configs.is_empty() {
        vec![ListItem::new("No entries in run_configs.")]
    } else {
        app.run_configs
            .iter()
            .map(|entry| ListItem::new(format_run_tree_row(app, entry)))
            .collect::<Vec<ListItem>>()
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

    let mut left_state = ListState::default().with_offset(app.main_list_offset);
    if !app.run_configs.is_empty() {
        left_state.select(Some(app.selected_idx));
    }
    frame.render_stateful_widget(left_list, layout[0], &mut left_state);

    if app.selected_run_motifs.is_empty() {
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
    let motif_items = app
        .selected_run_motifs
        .iter()
        .map(|motif| ListItem::new(format_run_motif_row(motif)))
        .collect::<Vec<ListItem>>();
    let motif_list = List::new(motif_items)
        .block(Block::default().title(format!("motifs [{}]", app.selected_run_motifs.len())).borders(Borders::ALL))
        .highlight_style(if app.selected_area == SelectedArea::RightMotifs {
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
        } else {
            Style::default().add_modifier(Modifier::BOLD)
        })
        .highlight_symbol(">> ");
    let mut motif_state = ListState::default().with_offset(app.motif_list_offset);
    motif_state.select(Some(app.selected_motif_idx.min(app.selected_run_motifs.len().saturating_sub(1))));
    frame.render_stateful_widget(motif_list, right_layout[0], &mut motif_state);

    let motif_details = Paragraph::new(render_motif_detail_lines(app).join("\n"))
        .block(Block::default().title("motif details").borders(Borders::ALL))
        .wrap(Wrap { trim: false });
    frame.render_widget(motif_details, right_layout[1]);
}

fn format_run_tree_row(app: &App, row: &VisibleRunConfigRow) -> String {
    let Some(node) = app.run_config_tree.get(&row.run_id) else {
        return format!("<missing run {}>", row.run_id);
    };
    let indent = "  ".repeat(row.depth);
    let tree_marker = if row.has_children {
        if row.expanded {
            "▼"
        } else if node.children_loaded || !node.output_run_ids.is_empty() {
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

fn render_run_value_lines(app: &App) -> Vec<String> {
    let mut right_lines = vec![
        format!("db: {}", app.db_path.display()),
        String::new(),
    ];

    if let Some(run_entry) = app.selected_run_entry() {
        right_lines.push(format!("run_id: {}", run_entry.id));
        right_lines.push(format!("type: {}", run_entry.run_config_type));
        right_lines.push(format!("run_on: {}", run_entry.run_on));
        right_lines.push(format!("descr: {}", run_entry.descr));
    } else {
        right_lines.push("run_id: <none>".to_string());
    }

    right_lines.push(format!("filters: {}", app.search_filters.summary()));
    right_lines.push(format!("id filter: {}", display_search_text(&app.search_filters.id, "<none>")));
    right_lines.push(format!("type filter: {}", app.search_filters.run_config_type.as_deref().unwrap_or(SEARCH_TYPE_NONE_LABEL)));
    right_lines.push(format!("descr filter: {}", display_search_text(&app.search_filters.descr, "<none>")));
    right_lines.push(format!("run_on from: {}", display_search_text(&app.search_filters.run_on_from, "<none>")));
    right_lines.push(format!("run_on to: {}", display_search_text(&app.search_filters.run_on_to, "<none>")));
    right_lines.push(String::new());

    for (name, value) in &app.selected_values {
        right_lines.push(format!("{name} = {value}"));
    }

    if let Some(status) = &app.status {
        right_lines.push(String::new());
        right_lines.push(format!("status: {status}"));
    }

    right_lines
}

fn format_run_motif_row(motif: &RunMotifEntry) -> String {
    match motif.score {
        Some(score) => format!(
            "{:>8}  ch {:<2}  score {:>8.5}  slices {:>5}  resources {:>4}",
            motif.motif_id, motif.channel_id, score, motif.slices_cnt, motif.resources_cnt
        ),
        None => format!(
            "{:>8}  ch {:<2}  slices {:>5}  resources {:>4}",
            motif.motif_id, motif.channel_id, motif.slices_cnt, motif.resources_cnt
        ),
    }
}

fn render_motif_detail_lines(app: &App) -> Vec<String> {
    let mut lines = vec![
        format!("db: {}", app.db_path.display()),
        String::new(),
    ];

    if let Some(run_entry) = app.selected_run_entry() {
        lines.push(format!("run_id: {}", run_entry.id));
        lines.push(format!("type: {}", run_entry.run_config_type));
        lines.push(format!("descr: {}", run_entry.descr));
    }

    lines.push(String::new());

    if let Some(motif) = app.selected_motif() {
        lines.push(format!("motif_id: {}", motif.motif_id));
        lines.push(format!("channel: {}", motif.channel_id));
        lines.push(format!("source find_motifs run_id: {}", motif.source_run_id));
        lines.push(format!("score: {}", motif.score.map(|value| format!("{value:.6}")).unwrap_or_else(|| "<n/a>".to_string())));
        lines.push(format!("slices: {}", motif.slices_cnt));
        lines.push(format!("resources: {}", motif.resources_cnt));
        lines.push(String::new());
        lines.push("[Enter] or [p] open plot".to_string());
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
    let popup_h = popup_height(area.height, SearchField::ALL.len() + 4);
    let popup_w = (area.width.saturating_mul(78) / 100).max(54).min(area.width.saturating_sub(2).max(1));
    let popup_x = area.x + area.width.saturating_sub(popup_w) / 2;
    let popup_y = area.y + area.height.saturating_sub(popup_h) / 2;
    let popup_rect = Rect::new(popup_x, popup_y, popup_w, popup_h);

    let popup_items = SearchField::ALL
        .iter()
        .map(|field| ListItem::new(format_search_field_row(*field, &app.search_popup)))
        .collect::<Vec<ListItem>>();

    let popup = List::new(popup_items)
        .block(Block::default().title("Search run_configs (Enter apply, Esc cancel, Ctrl-r reset draft)").borders(Borders::ALL))
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
        .block(Block::default().title("Select search type (Enter choose, Esc cancel)").borders(Borders::ALL))
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

fn sort_run_ids(run_ids: &mut Vec<u32>, all_run_configs: &[RunConfigEntry], sort_order: SortOrder) {
    let run_on_by_id = all_run_configs
        .iter()
        .map(|entry| (entry.id, entry.run_on.as_str()))
        .collect::<HashMap<u32, &str>>();
    match sort_order {
        SortOrder::Asc => run_ids.sort_by(|a, b| {
            run_on_by_id
                .get(a)
                .cmp(&run_on_by_id.get(b))
                .then_with(|| a.cmp(b))
        }),
        SortOrder::Desc => run_ids.sort_by(|a, b| {
            run_on_by_id
                .get(b)
                .cmp(&run_on_by_id.get(a))
                .then_with(|| a.cmp(b))
        }),
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

fn append_run_on_char(target: &mut String, ch: char) -> bool {
    if ch.is_ascii_digit() || matches!(ch, '-' | ':' | ' ') {
        target.push(ch);
        true
    } else {
        false
    }
}

fn parse_run_id_filter(value: &str) -> Result<Option<u32>, String> {
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
        run_progress_client(socket_id, tx).await;
    })
}

async fn run_progress_client(socket_id: String, tx: mpsc::UnboundedSender<ProgressClientEvent>) {
    let mut last_status = None;
    loop {
        if !send_progress_status(&tx, &mut last_status, "Connecting to progress socket.".to_string()) {
            break;
        }

        match connect_progress_stream(&socket_id).await {
            Ok(stream) => {
                if !send_progress_status(&tx, &mut last_status, "Connected to progress socket.".to_string()) {
                    break;
                }
                if let Err(err) = read_progress_stream(stream, &tx, &mut last_status).await {
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

        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

async fn read_progress_stream(
    stream: Stream,
    tx: &mpsc::UnboundedSender<ProgressClientEvent>,
    last_status: &mut Option<String>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut lines = BufReader::new(stream).lines();
    while let Some(line) = lines.next_line().await? {
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
        match Stream::connect(name.borrow()).await {
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
        .await
        .map_err(|err| format!("spawn failed: {err}"))?;
    if !output.status.success() {
        return Err(command_output_error(&output.stdout, &output.stderr));
    }
    parse_plot_target_path(&output.stdout)
}

async fn generate_motif_plot(ws_path: &Path, motif_id: u32) -> Result<(), String> {
    let output = plot_command(ws_path, motif_id, false)
        .output()
        .await
        .map_err(|err| format!("spawn failed: {err}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(command_output_error(&output.stdout, &output.stderr))
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
    if let Some(run_id) = app.selected_run_id() {
        println!("selected run_id: {run_id}");
    } else {
        println!("selected run_id: <none>");
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

    #[test]
    fn test_value_to_string_all_variants() {
        assert_eq!("1.5", value_to_string(&Val::Real(1.5)));
        assert_eq!("-2", value_to_string(&Val::Int(-2)));
        assert_eq!("7", value_to_string(&Val::UInt(7)));
        assert_eq!("abc", value_to_string(&Val::Str("abc".to_string())));
        assert_eq!("true", value_to_string(&Val::Bool(true)));
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
    async fn test_apply_search_from_popup_filters_visible_rows() {
        let store = Box::new(MockStore::new()) as Box<dyn Store>;
        let mut app = App::new(store, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));
        app.selected_area = SelectedArea::SearchPopup;
        app.has_more_paged_run_configs = false;
        app.all_run_configs = vec![
            RunConfigEntry {
                id: 7,
                run_config_type: "select_motifs".to_string(),
                descr: "first alpha".to_string(),
                run_on: "2026-03-13 08:00:00".to_string(),
            },
            RunConfigEntry {
                id: 8,
                run_config_type: "train_motif_nets".to_string(),
                descr: "second beta".to_string(),
                run_on: "2026-03-14 09:00:00".to_string(),
            },
        ];
        app.rebuild_run_config_tree(&[], &HashMap::new());
        app.search_popup.draft = RunConfigSearch {
            id: "7".to_string(),
            run_config_type: Some("select_motifs".to_string()),
            descr: "alpha".to_string(),
            run_on_from: "2026-03-13".to_string(),
            run_on_to: "2026-03-13".to_string(),
        };

        assert!(app.apply_search_from_popup().await);
        assert_eq!(app.selected_area, SelectedArea::LeftRunConfigs);
        assert_eq!(app.search_filters.id, "7");
        assert_eq!(app.run_configs.len(), 1);
        assert_eq!(app.run_configs[0].run_id, 7);
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
        store.expect_list_input_run_ids().times(1).returning(|ids| {
            assert_eq!(vec![2, 1], ids.to_vec());
            Ok(vec![(2, 1)])
        });
        let mut app = App::new(Box::new(store) as Box<dyn Store>, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));

        app.refresh_run_configs().await;

        assert_eq!(2, app.all_run_configs.len());
        assert_eq!(2, app.run_configs.len());
        assert!(!app.has_more_paged_run_configs);
        assert_eq!(Some(1), app.run_configs.first().map(|row| row.run_id));
        assert_eq!(Some(2), app.run_configs.get(1).map(|row| row.run_id));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_expand_selected_run_lazy_loads_children() {
        let mut store = MockStore::new();
        store.expect_list_output_run_ids().times(1).returning(|run_id| {
            assert_eq!(1, run_id);
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
        store.expect_list_input_run_ids().times(1).returning(|ids| {
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
        app.rebuild_run_config_tree(&[], &HashMap::new());
        app.rebuild_visible_run_configs(Some(1));

        assert!(app.expand_selected_run().await);
        assert_eq!(2, app.run_configs.len());
        assert_eq!(Some(2), app.run_configs.get(1).map(|row| row.run_id));
        assert!(app.run_config_tree.get(&1).map(|node| node.children_loaded).unwrap_or(false));
        assert!(app.run_config_tree.get(&1).map(|node| node.expanded).unwrap_or(false));
    }

    #[test]
    fn test_move_main_to_bottom_updates_offset() {
        let store = Box::new(MockStore::new()) as Box<dyn Store>;
        let mut app = App::new(store, PathBuf::from("/tmp/ws"), PathBuf::from("/tmp/test.db3"));
        app.run_configs = (0..30)
            .map(|idx| VisibleRunConfigRow {
                run_id: idx as u32,
                depth: 0,
                has_children: false,
                expanded: true,
            })
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
            .map(|idx| VisibleRunConfigRow {
                run_id: idx as u32,
                depth: 0,
                has_children: false,
                expanded: true,
            })
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
