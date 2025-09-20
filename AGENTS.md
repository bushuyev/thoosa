# Repository Guidelines

## Project Structure & Module Organization
The Rust workspace lives under `crates/`, with `Cargo.toml` wiring all subcrates. `thoosa_main` hosts the primary binaries and scheduled tasks, while `thoosa_common`, `thoosa_shared`, and `thoosa_store` provide shared math, concurrency, and persistence utilities. CUDA and Torch integration is isolated in `thoosa_runner/torch_runner`, and project-specific entry points sit in `thoosa_projects/`. Most crates keep unit tests alongside source modules; broader stress tests reside in `thoosa_runner/caching_runner_test`. Exploratory Python notebooks and scripts are under `python/` and stay optional for Rust builds.

## Build, Test, and Development Commands
Compile everything with `cargo build` (nightly toolchain pinned in `crates/rust-toolchain.toml`). Use `cargo build -p thoosa_main` for targeted binaries or `cargo run -p thoosa_main -- --help` to inspect available tasks. Fast Rust tests run via `cargo test`; prefer `./crates/nextest.sh` for consistent filtering (`./crates/nextest.sh` for regular suites, `./crates/nextest.sh slow` for CUDA-heavy cases). Set `CARGO_TARGET_DIR=crates/target` if you need deterministic artifact locations.

## Coding Style & Naming Conventions
Format with `cargo fmt --all` before committing. Keep `cargo clippy --workspace --all-targets -- -D warnings` clean; add crate-specific allows only with a comment explaining the exception. Code follows standard Rust 4-space indentation, `snake_case` modules/functions, `PascalCase` types/traits, and `SCREAMING_SNAKE_CASE` constants. Prefer explicit error types and `?` propagation over panics in shared crates; log with `tracing` where relevant.

## Testing Guidelines
Async tests use `#[tokio::test(flavor = "multi_thread")]` to match production runtimes. Split slow GPU or probabilistic suites into `slow`-prefixed test names so `nextest.sh` can isolate them. Integration and caching regression tests live in `caching_runner_test`; keep fixtures small enough for CI and document any large reference files. Capture new failure modes with targeted unit tests rather than expanding broad stress suites.

## Commit & Pull Request Guidelines
Recent history favors short, component-scoped subjects (e.g., `ThoTensor2/CrossBatchClc`). Follow the `<Subsystem>/<Change>` pattern or a concise imperative sentence. Each PR should describe the behavior change, link tracking issues, note required follow-up, and list test commands executed. Never submit unformatted code or new clippy warnings; include coverage for new branches and mention any slow tests skipped locally.

## Runtime & GPU Setup
GPU runs require the libtorch bundle referenced in `crates/env.sh`. Source that script (or replicate the `LD_PRELOAD`/`LD_LIBRARY_PATH` exports) before invoking CUDA-dependent binaries or tests. When debugging memory pressure, lower batch sizes and monitor for OOM events; rerun with `RUST_LOG=info` to capture runner diagnostics.
