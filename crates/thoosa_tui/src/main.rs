use interprocess::local_socket::{
    tokio::{prelude::*, Stream},
    GenericFilePath, GenericNamespaced, Name, ToFsName, ToNsName,
};
use chrono::{NaiveDate, NaiveDateTime};
use std::io;
use std::io::IsTerminal;
use std::path::PathBuf;
use std::time::{Duration, Instant};
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
use thoosa_store::{RunConfigEntry, SqlStore, Store};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

const SEARCH_TYPE_NONE_LABEL: &str = "no value";
const RUN_ON_DATE_FORMAT: &str = "%Y-%m-%d";
const RUN_ON_DATETIME_FORMAT: &str = "%Y-%m-%d %H:%M:%S";

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
    db_path: PathBuf,
    all_run_configs: Vec<RunConfigEntry>,
    run_configs: Vec<RunConfigEntry>,
    main_list_offset: usize,
    selected_idx: usize,
    selected_values: Vec<(String, String)>,
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
    fn new(store: Box<dyn Store>, db_path: PathBuf) -> Self {
        Self {
            store,
            db_path,
            all_run_configs: vec![],
            run_configs: vec![],
            main_list_offset: 0,
            selected_idx: 0,
            selected_values: vec![],
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
        self.run_configs.get(self.selected_idx).map(|entry| entry.id)
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

    fn main_page_size(&self) -> usize {
        let rows = terminal::size().map(|(_, r)| r).unwrap_or(24);
        rows.saturating_sub(3).max(1) as usize
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

    fn apply_search_from_popup(&mut self) -> bool {
        match self.search_popup.draft.normalized() {
            Ok(_) => {
                let keep_run_id = self.selected_run_id();
                self.search_filters = self.search_popup.draft.clone();
                self.rebuild_visible_run_configs(keep_run_id);
                self.selected_area = SelectedArea::LeftRunConfigs;
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
            ViewMode::RunConfigs => SelectedArea::LeftRunConfigs,
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

    fn rebuild_visible_run_configs(&mut self, keep_run_id: Option<u32>) {
        let mut rows = self.all_run_configs.clone();
        if let Ok(active_search) = self.search_filters.normalized() {
            rows.retain(|entry| active_search.matches(entry));
        }

        match self.run_on_sort_order {
            SortOrder::Asc => rows.sort_by(|a, b| a.run_on.cmp(&b.run_on)),
            SortOrder::Desc => rows.sort_by(|a, b| b.run_on.cmp(&a.run_on)),
        }

        self.run_configs = rows;

        if let Some(run_id) = keep_run_id {
            if let Some(idx) = self.run_configs.iter().position(|entry| entry.id == run_id) {
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

    fn apply_type_filter_from_popup(&mut self) {
        if let Some(selected_type) = self.selected_popup_type() {
            let keep_run_id = self.selected_run_id();
            self.search_filters.run_config_type = Some(selected_type.clone());
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
        match self.store.list_run_configs().await {
            Ok(run_configs) => {
                self.all_run_configs = run_configs;
                self.rebuild_visible_run_configs(keep_run_id);
                self.status = None;
                debug!(
                    "refresh_run_configs: loaded total={} visible={} filters={}",
                    self.all_run_configs.len(),
                    self.run_configs.len(),
                    self.search_filters.summary()
                );
            }
            Err(e) => {
                warn!("refresh_run_configs failed: {e:?}");
                self.all_run_configs.clear();
                self.run_configs.clear();
                self.selected_idx = 0;
                self.selected_values.clear();
                self.status = Some(format!("Failed to list run_configs: {e:?}"));
            }
        }
    }

    async fn refresh_selected_values(&mut self) {
        self.selected_values.clear();

        let Some(run_id) = self.selected_run_id() else {
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
                if rows.is_empty() {
                    self.status = Some(format!("No run_config_values for run_id={run_id}."));
                }
                self.selected_values = rows;
            }
            Err(e) => {
                warn!("refresh_selected_values failed: run_id={} err={e:?}", run_id);
                self.status = Some(format!("Failed to list run_config_values for run_id={run_id}: {e:?}"));
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

    let mut app = App::new(store, db_path);
    app.refresh_run_configs().await;
    app.refresh_selected_values().await;

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

                if key.code == KeyCode::Char(' ') && key.modifiers.is_empty() && matches!(app.selected_area, SelectedArea::LeftRunConfigs | SelectedArea::ProgressView) {
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
                                app.refresh_selected_values().await;
                            }
                            KeyCode::Char('s') if key.modifiers.is_empty() => {
                                app.toggle_run_on_sort_order();
                                app.refresh_selected_values().await;
                            }
                            KeyCode::Char('t') if key.modifiers.is_empty() => {
                                app.open_type_filter_popup().await;
                            }
                            KeyCode::Char('T') if is_unmodified_or_shift(&key) => {
                                app.reset_search_filters();
                                app.refresh_selected_values().await;
                            }
                            KeyCode::Down | KeyCode::Char('j') if key.modifiers.is_empty() => {
                                selection_changed = app.move_main_down(1);
                            }
                            KeyCode::Up | KeyCode::Char('k') if key.modifiers.is_empty() => {
                                selection_changed = app.move_main_up(1);
                            }
                            KeyCode::Char('d') if is_ctrl_char(&key, 'd') => {
                                selection_changed = app.move_main_down(app.main_page_size());
                            }
                            KeyCode::Char('u') if is_ctrl_char(&key, 'u') => {
                                selection_changed = app.move_main_up(app.main_page_size());
                            }
                            KeyCode::Char('G') if is_unmodified_or_shift(&key) => {
                                selection_changed = app.move_main_to_bottom();
                            }
                            KeyCode::Char('g') if key.modifiers.is_empty() => {
                                if app.handle_g_press(SelectedArea::LeftRunConfigs) {
                                    selection_changed = app.move_main_to_top();
                                }
                            }
                            _ => {}
                        }

                        if selection_changed {
                            app.refresh_selected_values().await;
                        }
                    }
                    SelectedArea::ProgressView => {}
                    SelectedArea::TypeFilterPopup => {
                        match key.code {
                            KeyCode::Esc => {
                                app.selected_area = SelectedArea::LeftRunConfigs;
                            }
                            KeyCode::Enter => {
                                app.apply_type_filter_from_popup();
                                app.selected_area = SelectedArea::LeftRunConfigs;
                                app.refresh_selected_values().await;
                            }
                            KeyCode::Char('T') if is_unmodified_or_shift(&key) => {
                                app.reset_search_filters();
                                app.selected_area = SelectedArea::LeftRunConfigs;
                                app.refresh_selected_values().await;
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
                                app.selected_area = SelectedArea::LeftRunConfigs;
                            }
                            KeyCode::Enter => {
                                if app.selected_search_field() == SearchField::Type {
                                    app.open_search_type_popup();
                                } else if app.apply_search_from_popup() {
                                    app.refresh_selected_values().await;
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
                                    ViewMode::RunConfigs => SelectedArea::LeftRunConfigs,
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
            "area: run_configs | sort(run_on): {} (s toggle) | j/k, Up/Down | Ctrl-d/u page | gg top | G bottom | / search | t type | T reset filters | r reload | Space panels | q quit",
            app.sort_order_label()
        ),
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
            .map(|entry| {
                ListItem::new(format!(
                    "{:>6}  {:<19}  {:<24}  {}",
                    entry.id, entry.run_on, entry.run_config_type, entry.descr
                ))
            })
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

    let mut right_lines = vec![
        format!("db: {}", app.db_path.display()),
        String::new(),
    ];

    if let Some(run_id) = app.selected_run_id() {
        right_lines.push(format!("run_id: {run_id}"));
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

    let right = Paragraph::new(right_lines.join("\n"))
        .block(Block::default().title("run_config_values").borders(Borders::ALL))
        .wrap(Wrap { trim: false });
    frame.render_widget(right, layout[1]);
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

    for entry in app.run_configs.iter().take(20) {
        println!("[{}] {} | {} | {}", entry.id, entry.run_config_type, entry.run_on, entry.descr);
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
        let mut app = App::new(store, PathBuf::from("/tmp/test.db3"));

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

    #[test]
    fn test_apply_search_from_popup_filters_visible_rows() {
        let store = Box::new(MockStore::new()) as Box<dyn Store>;
        let mut app = App::new(store, PathBuf::from("/tmp/test.db3"));
        app.selected_area = SelectedArea::SearchPopup;
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
        app.search_popup.draft = RunConfigSearch {
            id: "7".to_string(),
            run_config_type: Some("select_motifs".to_string()),
            descr: "alpha".to_string(),
            run_on_from: "2026-03-13".to_string(),
            run_on_to: "2026-03-13".to_string(),
        };

        assert!(app.apply_search_from_popup());
        assert_eq!(app.selected_area, SelectedArea::LeftRunConfigs);
        assert_eq!(app.search_filters.id, "7");
        assert_eq!(app.run_configs.len(), 1);
        assert_eq!(app.run_configs[0].id, 7);
    }

    #[test]
    fn test_move_main_to_bottom_updates_offset() {
        let store = Box::new(MockStore::new()) as Box<dyn Store>;
        let mut app = App::new(store, PathBuf::from("/tmp/test.db3"));
        app.run_configs = (0..30)
            .map(|idx| RunConfigEntry {
                id: idx as u32,
                run_config_type: "select_motifs".to_string(),
                descr: format!("descr-{idx}"),
                run_on: format!("2026-03-13 00:00:{idx:02}"),
            })
            .collect();

        assert!(app.move_main_to_bottom());
        assert_eq!(app.selected_idx, 29);
        assert_eq!(app.main_list_offset, 9);
    }

    #[test]
    fn test_move_main_up_keeps_offset_until_top_visible_row() {
        let store = Box::new(MockStore::new()) as Box<dyn Store>;
        let mut app = App::new(store, PathBuf::from("/tmp/test.db3"));
        app.run_configs = (0..30)
            .map(|idx| RunConfigEntry {
                id: idx as u32,
                run_config_type: "select_motifs".to_string(),
                descr: format!("descr-{idx}"),
                run_on: format!("2026-03-13 00:00:{idx:02}"),
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
