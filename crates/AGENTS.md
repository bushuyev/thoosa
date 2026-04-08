
## General info
The app asynchronously executes a series of tensor-related tasks (Stages) on GPU devices and handles OutOfMemory errors (see OOMHandler in crates/thoosa_shared/src/runner/oom_handler.rs) by throttling GPU access, processing data in chunks, or otherwise reducing GPU memory usage (see the Adjuster trait in crates/thoosa_shared/src/runner/oom_handler.rs:33 and its implementations).
It's important to keep big tensors (or big number of small tensors) between async awaits either in Arc<DevicesWrap<..>> obtained from SomeStage::run_cached (crates/thoosa_common/src/runner/common/cachable.rs) or in TmpCached (crates/thoosa_common/src/runner/tmp_cache.rs) so that GlobalCtx::dump_all_to_cpu (crates/thoosa_common/src/runner/global_ctx.rs) could dump them to CPU when handling OOM.
The app’s entry point is run_cli (crates/thoosa_main/src/cli.rs), which is invoked from the sample project (crates/thoosa_projects/thoosa_hbac/src/main.rs).
Currently, the only implemented workflow is time series classification, located in crates/thoosa_main/src/tasks.
async engine is Tokio, rayon is used to parallel sync functions. 

## Important Agent Rules
- **Never run `cargo fmt` or any other bulk formatter.** Make only the minimal manual edits required for the task at hand.
- **Do not touch unrelated files or code paths.** If a change is not strictly required to implement the current request, leave it untouched.
- **Use fallible tensor APIs and propagate tensor errors.** In `thoosa_main` and code invoked from it, prefer `Tensor::f_*`/`tensor.f_*` forms and bubble errors up; do not `unwrap`/`expect` tensor operations.
- if you've got any questions, ask before starting tasks.
- before finishing task always check if everything compiles (run `cargo build --release && cargo test --no-run`) and relevant tests pass (`cargo nextest run --profile codex  --no-fail-fast --retries=0 -p [crate]`)

## Project Structure & Module Organization
- Workspace root with `Cargo.toml` (Rust 2024, nightly toolchain).
- Crates:
  - `thoosa_common/`: tch-dependent common stuff 
  - `thoosa_shared/`: tch-independent common stuff 
  - `thoosa_projects/`: sample projects - thoosa_hbac is active, thoosa_bird_clef is obsolete
  - `thoosa_store/`: sqlite for keeping run configuration and tracking results 
  - `thoosa_shims/`: couple of cuda shims for gpu-cpu tensors copy
  - `thoosa_main/`: time series classification stages
  - `thoosa_motifs`: motif search related stuff, to be moved into thoosa_common/thoosa_shared
- Build artifacts in `target/`. Optional GPU libs configured via `env.sh`.

## Build, Test, and Development Commands
- Build workspace: `cargo build`
- Build a crate: `cargo build -p torch_runner`
- Run tests (all): `cargo nextest run --profile codex --no-fail-fast --retries=0`
- Run tests for a crate: `cargo nextest run --profile codex -p [crate]`
- Lint (if installed): `cargo clippy --all-targets -- -D warnings`
Notes:
- Requires nightly toolchain (`rust-toolchain.toml` is pinned). Install via `rustup toolchain install nightly`.
- GPU/Torch runs may require `source ./env.sh` to set `LD_PRELOAD`.
- The effective workspace root for cargo commands in this repo is `/home/bu/dvl/rust/thoosa/crates`.
- For repeated CLI experiments, running the release binary directly is much faster than `cargo run`:
  `LD_LIBRARY_PATH=/home/bu/dvl/rust/thoosa/crates/target/release /home/bu/dvl/rust/thoosa/crates/target/x86_64-unknown-linux-gnu/release/thoosa_hbac ...`

## Coding Style & Naming Conventions
- Linting: keep `clippy` clean; deny warnings in CI/local before PRs.
- Naming: modules/files `snake_case`; types/traits `PascalCase`; functions/vars `snake_case`.
- Concurrency: prefer `tokio` primitives; shared maps via `dashmap`; avoid `unsafe`.
- asserts and logging should be on one line - no matter how long it gets.

## Testing Guidelines
- Frameworks: `cargo test` with `#[test]` and `#[tokio::test]` (multi-thread where needed).
- Location: unit tests inline in `src/*` modules;
- A struct being tested should receive its mocked dependencies which are configured through their Mock[StructName]Builder (placed next to StructName in a #[cfg(test)] mock_[struct_name] module. If a struct can't be hidden behind a trait (although it should not be a problem to hide a Stage struct behind Stage trait) it can be injected with mock_double.
- Naming: `mod tests { ... }`, test fns `snake_case` describing behavior.

## Misc info
masks for tensors of all sorts expected to contain false for padded (zeros)) values and true for actual values.

## Motif Search Notes
- For HBAC motif exploration, `plot-data --stdout` is the most useful inspection path because it emits machine-readable JSON. Redirect stderr separately because CUDA/libtorch setup can still print noise there even when stdout is clean JSON.
- The SQLite schema stores motif membership in `slices`; `motifs` itself does not contain slice counts. For motif screening, join/query `slices` for `COUNT(*)` and `COUNT(DISTINCT resource_id)`, and `selected_motifs` for selection scores.
- In March 2026 screening on `/home/bu/tmp/ws_test_hbac`, a `BandPassFilter(low_hz=4.0, band_hz=12.0, kernel_size=100, stride=10)` + notch family produced mostly oscillatory, low-correlation motifs, while `window=160`, `step=5`, `MedianFilter(window=100)`, `SavGolFilter(window_len=192, poly_order=1, stride=10)`, and `slices_filter=Accumulate(Quantile q≈0.0983)` produced more usable EEG motifs.
- `InterestingTurningPointsRatio` was not discriminative in that screening pass; if future selection still over-favors edge-heavy motifs, consider adding an explicit consensus-edge/peak-position criterion instead of relying on that metric.
