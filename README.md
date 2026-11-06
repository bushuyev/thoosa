# Thoosa

## Design Goals and Constraints

This project is intended for **machine-learning experiments on a local workstation**, where GPU resources are limited and CUDA OOM conditions are expected.

It is built directly on **LibTorch**. Supporting other ML runtimes or GPU backends is not a goal.

A primary design objective is to make long-running experiments survive memory pressure whenever possible. On CUDA OOM, the system attempts to recover by progressively reducing resource usage—for example by decreasing data sizes, using lower-quality or less memory-intensive operations, or accepting slower execution.

When LibTorch encounters a CUDA error that requires the CUDA context to be restarted, the application can run GPU work in a separate **IPC worker process** ("IPC Mode"). That worker can be restarted independently, allowing training to continue without restarting the entire application.

The project also includes a simple **DashMap-based cache** that can keep data on the GPU for reuse and move cached GPU data back to CPU memory when additional VRAM needs to be freed.

The overall priority is **robustness under constrained GPU memory**, even when that requires trading quality and performance for continued execution.


## Workspace layout

The Cargo workspace is in [`crates/`](crates/), not at the repository root. Its main crates are:

| Crate | Purpose |
| --- | --- |
| [`thoosa_shared`](crates/thoosa_shared/) | Common types, stage and run-context traits, run configurations, resource descriptions, and utilities that do not depend on LibTorch. |
| [`thoosa_common`](crates/thoosa_common/) | Tensor-aware runner, GPU memory and cache handling, CUDA worker IPC, workspaces, and reusable stages and networks. |
| [`thoosa_store`](crates/thoosa_store/) | SQLite storage for resources, run configurations, and results. |
| [`thoosa_motifs`](crates/thoosa_motifs/) | Motif search, selection, statistics, model training, and the shared project CLI commands. |
| [`thoosa_shims`](crates/thoosa_shims/) | Rust bindings and C++/CUDA code for tensor copies, streams, and LibTorch integration. |
| [`thoosa_tui`](crates/thoosa_tui/) | Terminal viewer for run configurations and results in a workspace database. |
| [`thoosa_hbac`](crates/thoosa_projects/thoosa_hbac/) | Active HBAC EEG application and dataset-specific import and pipeline commands. |

The crates under [`crates/thoosa_macros/`](crates/thoosa_macros/) support configuration and data types:

| Crate | Purpose |
| --- | --- |
| `thoosa_run_config_args` | Traits for grid and optimization run-configuration sources. |
| `thoosa_run_config_args_derive` | `ThoosaRunConfig` derive for generating those configuration helpers. |
| `thoosa_properties_map` | Traits for converting properties to and from maps. |
| `thoosa_properties_map_derive` | `ThoosaPropertiesMap` derive for property maps. |
| `thoosa_digest_id` | Digest hashing trait and support code. |
| `thoosa_digest_id_derive` | `DigestHash` derive. |
| `thoosa_cpu_data_derive` | `CpuData` derive. |
| `thoosa_named_derive` | `Named` derive. |
| `cached_derive` | Cache-related derives. |
| `thoosa_macros_utils` | Helpers shared by the derive crates. |
| `macros_tests` | Fixtures and tests for macro expansion. |~~~~

## Build

This workspace targets Linux and uses Rust 2024 with a nightly toolchain. Install Rust through `rustup`, plus CMake, Ninja, a C++23 compiler, the CUDA toolkit, and a CUDA-enabled LibTorch installation. The `tch` and `torch-sys` dependencies use the patched `tch-rs` Git branch specified in [`crates/Cargo.toml`](crates/Cargo.toml).

Before building, set `LIBTORCH` to the root of your LibTorch installation (the directory containing `lib/` and `share/cmake/Torch/`). [`crates/.cargo/config.toml`](crates/.cargo/config.toml) also contains a machine-specific default path. The CUDA shim build currently expects the toolkit at `/usr/local/cuda`, the driver library at `/usr/lib/x86_64-linux-gnu/libcuda.so`, and CUDA architecture 89; adjust [`crates/thoosa_shims/build.rs`](crates/thoosa_shims/build.rs) and its [`CMakeLists.txt`](crates/thoosa_shims/src-cpp/CMakeLists.txt) for a different setup.

```sh
cd crates
rustup toolchain install nightly
export LIBTORCH=/path/to/libtorch
cargo build --release
```

For a faster compile check, use `cargo check --workspace --all-targets` from `crates/`. Build one package with `cargo build --release -p thoosa_hbac` (or another package name from the tables above).


## Overview

Asynchronous stages coordinate tensor work, while synchronous stages perform the computations on CPU or GPU. The runner controls GPU concurrency and retries work under memory pressure; managed tensor storage allows cached data to move between GPU and CPU. GPU execution can run locally or in a separate IPC worker process.

See the [architecture overview](docs/overview.md) for the main entities and how they fit together.

- [OOM Handling](docs/oom.md)
- [IPC subsystem](docs/ipc.md) 

## Run

Use the HBAC binary's help to see the available import, motif, training, and export commands and their arguments:

```sh
cd crates
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH="$PWD/target/x86_64-unknown-linux-gnu/release/build:$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"
cargo run --release -p thoosa_hbac -- --help
```

The release build places `libthoosa_shims.so` in the `build/` directory included above. Commands accept `--ws-path` to select a workspace directory; the default is the current directory. A workspace contains the SQLite database (`main_db.db3`) and experiment data such as saved tensors and models. To inspect an existing workspace in the terminal, run `cargo run --release -p thoosa_tui -- /path/to/workspace`.

## Tests

From `crates/`, compile all test targets with `cargo test --no-run`. With `cargo-nextest` installed, run the suite using `cargo nextest run --profile codex --no-fail-fast --retries=0`, or add `-p <crate>` for one package. Tests that exercise CUDA, LibTorch, or real datasets need the corresponding hardware, libraries, and data available at runtime.
