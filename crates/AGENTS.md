
## General info
The app asynchronously executes a series of tensor-related tasks (Stages) on GPU devices and handles OutOfMemory errors (see OOMHandler in crates/thoosa_shared/src/runner/oom_handler.rs) by throttling GPU access, processing data in chunks, or otherwise reducing GPU memory usage (see the Adjuster trait in crates/thoosa_shared/src/runner/oom_handler.rs:33 and its implementations).
The app’s entry point is run_cli (crates/thoosa_main/src/cli.rs), which is invoked from the sample project (crates/thoosa_projects/thoosa_hbac/src/main.rs).
Currently, the only implemented workflow is time series classification, located in crates/thoosa_main/src/tasks.
async engine is Tokio, rayon is used to parallel sync functions. 

## Important Agent Rules
- **Never run `cargo fmt` or any other bulk formatter.** Make only the minimal manual edits required for the task at hand.
- **Do not touch unrelated files or code paths.** If a change is not strictly required to implement the current request, leave it untouched.

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
- Run tests (all): `cargo nextest run --no-fail-fast`
- Run tests for a crate: `cargo nextest run -p caching_runner_test -- --nocapture`
- Lint (if installed): `cargo clippy --all-targets -- -D warnings`
Notes:
- Requires nightly toolchain (`rust-toolchain.toml` is pinned). Install via `rustup toolchain install nightly`.
- GPU/Torch runs may require `source ./env.sh` to set `LD_PRELOAD`.

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
