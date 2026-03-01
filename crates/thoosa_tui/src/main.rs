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
use thoosa_shared::val::Val;
use thoosa_store::{RunConfigEntry, SqlStore, Store};

#[derive(Parser, Debug)]
#[command(name = "thoosa_tui")]
#[command(about = "Inspect run configs and values from workspace database")]
struct Args {
    ws_path: PathBuf,
}

struct App {
    store: Box<dyn Store>,
    db_path: PathBuf,
    all_run_configs: Vec<RunConfigEntry>,
    run_configs: Vec<RunConfigEntry>,
    selected_idx: usize,
    selected_values: Vec<(String, String)>,
    status: Option<String>,
    selected_area: SelectedArea,
    run_on_sort_order: SortOrder,
    filter_type: Option<String>,
    run_config_types: Vec<String>,
    type_popup_selected_idx: usize,
    pending_gg: Option<(SelectedArea, Instant)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SelectedArea {
    LeftRunConfigs,
    TypeFilterPopup,
}

#[derive(Clone, Copy)]
enum SortOrder {
    Asc,
    Desc,
}

impl App {
    fn new(store: Box<dyn Store>, db_path: PathBuf) -> Self {
        Self {
            store,
            db_path,
            all_run_configs: vec![],
            run_configs: vec![],
            selected_idx: 0,
            selected_values: vec![],
            status: None,
            selected_area: SelectedArea::LeftRunConfigs,
            run_on_sort_order: SortOrder::Desc,
            filter_type: None,
            run_config_types: vec![],
            type_popup_selected_idx: 0,
            pending_gg: None,
        }
    }

    fn selected_run_id(&self) -> Option<u32> {
        self.run_configs.get(self.selected_idx).map(|entry| entry.id)
    }

    fn selected_popup_type(&self) -> Option<String> {
        self.run_config_types.get(self.type_popup_selected_idx).cloned()
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

    fn popup_page_size(&self) -> usize {
        let rows = terminal::size().map(|(_, r)| r).unwrap_or(24);
        popup_height(rows, self.run_config_types.len()).saturating_sub(2).max(1) as usize
    }

    fn move_main_down(&mut self, step: usize) -> bool {
        move_down(&mut self.selected_idx, self.run_configs.len(), step)
    }

    fn move_main_up(&mut self, step: usize) -> bool {
        move_up(&mut self.selected_idx, self.run_configs.len(), step)
    }

    fn move_main_to_top(&mut self) -> bool {
        move_to_top(&mut self.selected_idx, self.run_configs.len())
    }

    fn move_main_to_bottom(&mut self) -> bool {
        move_to_bottom(&mut self.selected_idx, self.run_configs.len())
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

    fn rebuild_visible_run_configs(&mut self, keep_run_id: Option<u32>) {
        let mut rows = self.all_run_configs.clone();
        if let Some(filter_type) = &self.filter_type {
            rows.retain(|entry| &entry.run_config_type == filter_type);
        }

        match self.run_on_sort_order {
            SortOrder::Asc => rows.sort_by(|a, b| a.run_on.cmp(&b.run_on)),
            SortOrder::Desc => rows.sort_by(|a, b| b.run_on.cmp(&a.run_on)),
        }

        self.run_configs = rows;

        if let Some(run_id) = keep_run_id {
            if let Some(idx) = self.run_configs.iter().position(|entry| entry.id == run_id) {
                self.selected_idx = idx;
                return;
            }
        }

        if self.run_configs.is_empty() {
            self.selected_idx = 0;
        } else if self.selected_idx >= self.run_configs.len() {
            self.selected_idx = self.run_configs.len() - 1;
        }
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

    fn reset_type_filter(&mut self) {
        if self.filter_type.is_some() {
            let keep_run_id = self.selected_run_id();
            self.filter_type = None;
            self.rebuild_visible_run_configs(keep_run_id);
            self.status = Some("Type filter reset.".to_string());
            debug!("reset_type_filter");
        }
    }

    fn apply_type_filter_from_popup(&mut self) {
        if let Some(selected_type) = self.selected_popup_type() {
            let keep_run_id = self.selected_run_id();
            self.filter_type = Some(selected_type.clone());
            self.rebuild_visible_run_configs(keep_run_id);
            self.status = Some(format!("Type filter applied: {selected_type}"));
            debug!("apply_type_filter_from_popup: type={selected_type}");
        }
    }

    async fn open_type_filter_popup(&mut self) {
        match self.store.list_run_config_types().await {
            Ok(types) => {
                self.run_config_types = types;
                if self.run_config_types.is_empty() {
                    self.status = Some("No run_config types found.".to_string());
                    return;
                }

                self.type_popup_selected_idx = self
                    .filter_type
                    .as_ref()
                    .and_then(|active| self.run_config_types.iter().position(|t| t == active))
                    .unwrap_or(0);

                self.selected_area = SelectedArea::TypeFilterPopup;
                self.status = None;
                debug!("open_type_filter_popup: count={}", self.run_config_types.len());
            }
            Err(e) => {
                warn!("open_type_filter_popup failed: {e:?}");
                self.status = Some(format!("Failed to list run_config types: {e:?}"));
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
                    "refresh_run_configs: loaded total={} visible={} filter={:?}",
                    self.all_run_configs.len(),
                    self.run_configs.len(),
                    self.filter_type
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

    loop {
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
                            KeyCode::Char('T') if key.modifiers.is_empty() => {
                                app.reset_type_filter();
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
                            KeyCode::Char('G') if key.modifiers.is_empty() => {
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
                            KeyCode::Char('T') if key.modifiers.is_empty() => {
                                app.reset_type_filter();
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
                            KeyCode::Char('G') if key.modifiers.is_empty() => {
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
                }
            }
        }
    }

    terminal.show_cursor()?;
    Ok(())
}

fn render(frame: &mut Frame<'_>, app: &App) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(frame.area());

    let layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(48), Constraint::Percentage(52)])
        .split(outer[0]);

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

    let left_title = if let Some(filter_type) = &app.filter_type {
        format!("run_configs [type={filter_type}]")
    } else {
        "run_configs [type=*]".to_string()
    };

    let left_block = Block::default().title(left_title).borders(Borders::ALL);
    let left_list = List::new(left_items)
        .block(left_block)
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        .highlight_symbol(">> ");

    let mut left_state = ListState::default();
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

    if let Some(filter_type) = &app.filter_type {
        right_lines.push(format!("type filter: {filter_type}"));
    } else {
        right_lines.push("type filter: <none>".to_string());
    }

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

    if app.selected_area == SelectedArea::TypeFilterPopup {
        render_type_filter_popup(frame, app);
    }

    let status_line = match app.selected_area {
        SelectedArea::LeftRunConfigs => format!(
            "area: run_configs | sort(run_on): {} (s toggle) | j/k, Up/Down | Ctrl-d/u page | gg top | G bottom | t filter | T reset | r reload | q quit",
            app.sort_order_label()
        ),
        SelectedArea::TypeFilterPopup => {
            "area: type-filter | j/k, Up/Down | Ctrl-d/u page | gg top | G bottom | Enter apply | Esc close | T reset filter".to_string()
        }
    };
    frame.render_widget(Paragraph::new(status_line), outer[1]);
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
                let marker = if app.filter_type.as_deref() == Some(t.as_str()) { "*" } else { " " };
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

fn popup_height(term_rows: u16, items_count: usize) -> u16 {
    let max_h = term_rows.saturating_sub(4).max(3);
    let desired = items_count.saturating_add(2) as u16;
    desired.clamp(3, max_h)
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

    #[test]
    fn test_value_to_string_all_variants() {
        assert_eq!("1.5", value_to_string(&Val::Real(1.5)));
        assert_eq!("-2", value_to_string(&Val::Int(-2)));
        assert_eq!("7", value_to_string(&Val::UInt(7)));
        assert_eq!("abc", value_to_string(&Val::Str("abc".to_string())));
        assert_eq!("true", value_to_string(&Val::Bool(true)));
    }
}
