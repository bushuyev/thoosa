# Repository Guidelines

## Project Structure & Module Organization
- Workspace root with `Cargo.toml` (Rust 2024, nightly toolchain).
- Crates:
  - `caching_runner/`: core async caching/execution primitives (Tokio, DashMap).
  - `torch_runner/`: CUDA/Torch integration and tensor utilities.
  - `cached_derive/`: `proc-macro` crate providing `#[derive(Cached)]`.
  - `test_utils/`: shared testing helpers.
  - `caching_runner_test/`: stress and behavior tests for the cache/runner.
- Build artifacts in `target/`. Optional GPU libs configured via `env.sh`.

## Build, Test, and Development Commands
- Build workspace: `cargo build`
- Build a crate: `cargo build -p torch_runner`
- Run tests (all): `cargo test`
- Run tests for a crate: `cargo test -p caching_runner_test -- --nocapture`
- Lint (if installed): `cargo clippy --all-targets -- -D warnings`
- Format: `cargo fmt --all`
Notes:
- Requires nightly toolchain (`rust-toolchain.toml` is pinned). Install via `rustup toolchain install nightly`.
- GPU/Torch runs may require `source ./env.sh` to set `LD_PRELOAD`.

## Coding Style & Naming Conventions
- Formatting: `cargo fmt` (default rustfmt). 4‑space indent.
- Linting: keep `clippy` clean; deny warnings in CI/local before PRs.
- Naming: modules/files `snake_case`; types/traits `PascalCase`; functions/vars `snake_case`.
- Concurrency: prefer `tokio` primitives; shared maps via `dashmap`; avoid `unsafe`.

## Testing Guidelines
- Frameworks: `cargo test` with `#[test]` and `#[tokio::test]` (multi-thread where needed).
- Location: unit tests inline in `src/*` modules; crate-level scenarios in `caching_runner_test`.
- A struct being tested should receive its mocked dependencies which are configured through their Mock[StructName]Builder (placed next to StructName in a #[cfg(test)] mock_[struct_name] module. If a struct can't be hidden behind a trait (although should not be a problem to hide a Stage struct behind Stage trait) it can be injected with mock_double.
- Naming: `mod tests { ... }`, test fns `snake_case` describing behavior.
- Run with logs: `RUST_BACKTRACE=1 cargo test -- --nocapture`.

## Commit & Pull Request Guidelines
- Commits: short, imperative subjects (e.g., "cleanup", "tensor copy"). Group related changes.
- PRs: include a clear summary, rationale, and scope; reference issues (e.g., `Fixes #123`).
- Requirements: passing build/tests, formatted code, no new clippy warnings, add/adjust tests when behavior changes.
- Large changes: outline design in the PR description; include benchmarks if performance‑related.

## Security & Configuration Tips
- CUDA/Torch: ensure compatible CUDA (e.g., `cudarc` feature `cuda-12040`) and preload libs via `env.sh`.
- Avoid committing secrets or local paths; `.gitignore` already present—extend if adding new tooling.

## Resilience: Recovering From OOM Kills
- Symptom: Heavy tests may trigger the Linux OOM killer. The process dies abruptly (no Rust backtrace), leaving partial GPU allocations freed by the kernel.
- Safe Restart Steps:
  - Ensure the shell session has torch/CUDA env loaded if needed: `source ./env.sh`.
  - Reinitialize the global context on restart: the code already uses `GlobalCtx::init_devices(N)` in tests; simply rerun the test command.
  - Prefer smaller, focused test runs when iterating:
    - Single crate: `cargo test -p torch_runner -- --nocapture`
    - Single test: `cargo test -p torch_runner test_ab_loader_and_classifier_cpu -- --nocapture`
  - If GPU state feels stale (rare), reload libtorch CUDA once per session (happens implicitly in tests that `dlopen` `libtorch_cuda.so`).
- Preventive Tips During Heavy Runs:
  - Lower batch sizes or dimensions in stress tests when debugging.
  - Run CPU-only tests first; add CUDA only when needed.
  - Use environment variables to limit threads (Tokio/runtime) if memory pressure is high.
- What to Capture When It Happens:
  - Kernel OOM line from `dmesg` or syslog, test name, and the input sizes.
  - Rough free/total GPU memory at start (see `cuda_utils::cuda_mem()` used in tests).


## Recent Changes: Permits Arbiter uses out_id (Aug 2025)
- Summary: The FIFO Arbiter that manages permits now keys fairness and “supertask” escalation by the runnable’s output id (`out_id`) instead of the human-readable run context name. This preserves fairness when multiple parallel tasks share a context name and aligns permit decisions with the actual logical task instance.

- Rationale:
  - RunCtx `name()` is for humans; multiple tasks may share it.
  - `out_id` is stable per logical output and already defines caching keys.

- API Changes:
  - `PermitSource::acquire(n, out_id: Arc<dyn CachableId>) -> Result<RunnerPerm, MyError>`
  - `GlobalCtx::acquire_many(ordinal, n, out_id: Arc<dyn CachableId>) -> Result<RunnerPerm, MyError>`
  - `RunnerPerm` no longer stores a display name; constructors: `RunnerPerm::new(inner)`, `RunnerPerm::with_notify(inner, notify)`.

- Call‑site Pattern:
  - Build once per run attempt and reuse while acquiring permits:
    - `let out_id = task.out_id(input.id());`
    - `let out_key = Arc::new(out_id.clone()) as Arc<dyn CachableId>;`
    - `let perm = GlobalCtx::global().acquire_many(device_idx, n, out_key.clone()).await?;`

- OOM Handler:
  - Requires `R: OutId`. It derives `out_key` via `task.out_id(input.id())` and passes it to `acquire_many` for both initial and retry acquisitions.

- Arbiter Behavior (unchanged semantically, keyed by out_id):
  - Strict head‑of‑line grants when possible.
  - On repeated `OutOfPermits` for the same `out_id`, promote to a temporary “supertask” and attempt to grant full capacity; reject others until served.

- Follow‑ups / TODOs:
  - Consider exposing capacity and fail threshold via config.
  - Add tracing fields for hashed/type‑tagged `out_id` to aid debugging without leaking internals.
  - Keep clippy clean; prune unused imports in follow‑up.

## Embedding As Subcrates In A Parent Workspace

This repository can be vendored into a larger Rust workspace as a set of subcrates. Below is everything you need to wire it up and use it safely in a new location.

- Workspace membership
  - Add each crate under `[workspace.members]` in the parent `Cargo.toml` (or use local `path` dependencies if you don’t want them as members):
    - `caching_runner`
    - `torch_runner`
    - `cached_derive`
    - `test_utils` (optional; test‑only helpers)
    - `caching_runner_test` (optional; integration/stress tests)

- Rust toolchain
  - The project targets nightly (see `rust-toolchain.toml`). Install via `rustup toolchain install nightly` and set your workspace to use it, or keep a local override in this subtree.

- Typical dependency graph in the parent workspace
  - `cached_derive`: proc‑macro crate that defines `#[derive(Cached)]` for `Runnable` types.
  - `caching_runner`: core traits and runtime (Data, WithId, Runnable, OutId, Cached, GlobalCtx, permits Arbiter, DataCacheWrapper, etc.). Depends on Tokio and DashMap.
  - `torch_runner`: optional CUDA/Torch integration and tensor helpers built on `tch` + `cudarc`. Depends on `caching_runner`.
  - `test_utils` and `caching_runner_test`: optional for validation/examples.

- Add as dependencies (parent crate `Cargo.toml`)
  - Example using path deps:
    ```toml
    [dependencies]
    caching_runner = { path = "vendor/snipps/caching_runner" }
    cached_derive  = { path = "vendor/snipps/cached_derive" }
    # Only if using Torch/CUDA features
    torch_runner   = { path = "vendor/snipps/torch_runner", optional = true }

    [features]
    default = []
    gpu = ["torch_runner"]
    ```

- CUDA / Torch runtime
  - GPU runs require libtorch CUDA and a matching CUDA driver/runtime. Use `env.sh` as a reference for setting `LD_PRELOAD`/`LD_LIBRARY_PATH` to point at `libtorch_cuda.so`, etc. CPU‑only tests don’t require CUDA.
  - The repo includes test utilities that `dlopen` `libtorch_cuda.so` once per process; when embedding, ensure your CI or dev shell `source`s an equivalent setup if you plan to run GPU tests.

- Initialization and permits
  - Call `GlobalCtx::init_devices(n)` once during app startup/tests to register device count.
  - When acquiring permits, pass the logical output identifier (`out_id`) as an `Arc<dyn CachableId>` – permits fairness and escalation are keyed by `out_id`.
  - Typical pattern:
    ```rust
    let out_key: Arc<dyn CachableId> = Arc::new(task.out_id(input.id()));
    let perm = GlobalCtx::global().acquire_many(device_idx, to_acquire, out_key.clone()).await?;
    ctx.write().add_permit(perm);
    ```

- Caching and `#[derive(Cached)]`
  - Implement `Runnable` for your task. Add `impl OutId for YourTask` so the cache key and permit arbiter can identify logical outputs.
  - Derive `Cached` (from the `cached_derive` crate) on your `Runnable` to get a shared async cache keyed by `Out::ID`.
  - Call `run_cached` to get `Arc<DataCacheWrapper<Out>>` and use `with_data`/`with_data_array` to materialize data on a device lazily and safely.

- Breaking API note: ThoTensor id is generic (Aug 2025)
  - `torch_runner::data::ThoTensor` is now `ThoTensor<D: Dims, Id: Named + Clone>` rather than always using `SimpleId`.
  - In this repo’s usages we use `SimpleId` as the `Id` type: `ThoTensor<LoadedTensor, SimpleId>`, etc.
  - If you have custom id types, ensure they implement the bounds required by `WithId` in `caching_runner` (i.e., `CachableId + Send + Sync + Eq + Hash + Clone + Debug + Named`).

- Minimal example
  ```rust
  use std::sync::Arc;
  use parking_lot::RwLock;
  use caching_runner::common::{Runnable, OutId, WithId, SimpleId, Data, CopyToDevice, MyError};
  use cached_derive::Cached;

  #[derive(Debug, Clone, Cached)]
  #[cache_name = "MyOutId"]
  struct MyTask;
  impl OutId for MyTask { fn out_id(&self, input_id: &<Self::In as WithId>::ID) -> <Self::Out as WithId>::ID { SimpleId::from(format!("out_{input_id}")) } }

  impl Runnable for MyTask {
      type In = MyInput;  // implements Data + WithId
      type Out = MyOutput; // implements Data<Sub=Self> + WithId + CopyToDevice
      type Ctx = MyRunCtx; // implements RunCtx
      async fn run<I: WithId<ID=<Self::In as WithId>::ID> + Send + Sync + 'static>(
          self: Arc<Self>, input: Arc<I>, ctx: Arc<RwLock<Self::Ctx>>
      ) -> Result<Self::Out, MyError> { /* ... */ }
  }

  // Startup
  let _ = caching_runner::global_ctx::GlobalCtx::init_devices(2);
  // Later
  let out = Arc::new(MyTask).run_cached(Arc::new(my_input), Arc::new(RwLock::new(my_ctx))).await?;
  out.with_data(1, None, |o| { /* use o on device 1 */ Ok(()) }).await?;
  ```

- Testing in a parent workspace
  - CPU tests: no special env.
  - GPU tests: ensure a single `dlopen("libtorch_cuda.so")` per test process (already handled in this repo’s tests). In CI, guard GPU jobs behind a feature or tag as needed.
  - Heavy OOM stress tests live in `caching_runner_test` and `torch_runner` test modules; consider enabling them only in targeted pipelines.

- CI tips
  - Use `cargo fmt --all` and `cargo clippy --all-targets -- -D warnings` for PR checks.
  - For optional GPU checks, run a separate job (matrix) with the CUDA/Torch environment prepared. Keep CPU jobs as default.

- Code style / contribution
  - Keep changes small and focused; prefer trait‑based abstractions for extensibility.
  - Avoid `unsafe` unless strictly necessary and justified.
  - Keep caches, runners, and arbiter code async‑friendly and deadlock‑free; follow existing patterns for `Notify` and per‑device inflight copy coordination.
