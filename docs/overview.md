# Overview

Thoosa executes tensor pipelines with asynchronous orchestration and synchronous computation. GPU work can run in the application process (Local mode) or in a separate process per GPU (IPC mode). Scheduling, memory residency, and OOM recovery support both modes.

## Stages and execution

| Entity | Role |
| --- | --- |
| [`Stage`](../crates/thoosa_shared/src/traits.rs) | Defines a computation's context (`Ctx`), input (`In`), and output (`Out`) types. It does not define an execution method; the separate execution traits describe how it runs. |
| [`AsyncStage`](../crates/thoosa_shared/src/traits.rs) | Adds an asynchronous `run` method. These stages coordinate dependencies, await other stages, and assemble results. Asynchronous orchestration stays in the application process in both modes. |
| [`SyncStage`](../crates/thoosa_common/src/runner/sync_stage.rs) | Defines a synchronous `do_run` method with borrowed input and context. Tensor calculations belong here; scheduling and IPC dispatch belong to the execution layer. |
| [`RunCtx`](../crates/thoosa_shared/src/traits.rs) | Carries execution identity and the target device index, supports branching for child work, and accepts GPU cleanup callbacks. [`RunCtxImpl`](../crates/thoosa_common/src/runner/run_ctx.rs) also carries concurrency limits and cache-area cleanup ownership. Device index 0 is CPU; subsequent indices identify GPUs. |
| [`StageExecution`](../crates/thoosa_common/src/runner/execution.rs) | Provides the asynchronous execution boundary used by the runner. `ExecutionContext` distinguishes CPU, Local CUDA, and IPC execution. Local execution retains input and stream resources through CUDA synchronization; an IPC [`ProxyStage`](../crates/thoosa_common/src/ipc/proxy.rs) dispatches work to its worker endpoint. |
| [`SyncStageRunner`](../crates/thoosa_common/src/runner/syn_stage_runner.rs) | Schedules admitted work per device and bounds active jobs using adaptive concurrency limits. Its retry scopes and CUDA admission gate coordinate OOM retries and device recovery. `OOMHandler` submits GPU attempts through this runner. |

For mutable execution, [`PrepareStage` and `StageWithState`](../crates/thoosa_common/src/runner/sync_stage.rs) separate asynchronous state preparation from synchronous computation with exclusively held state. `ConsumingStage` supports owned input and returns it on failure so an eligible operation can be retried.

## Caching and tensor residency

| Entity | Role |
| --- | --- |
| [`Cached`](../crates/thoosa_common/src/runner/common/cachable.rs) | Adds `run_cached` to an asynchronous stage. It derives an output ID through `OutId`, shares an in-flight computation between callers, and reuses its result. Failed computations are removed so later calls can try again. |
| [`DevicesWrap`](../crates/thoosa_common/src/runner/common/cachable.rs) | Owns managed data and its device replicas. Handles register with runtime residency so GPU copies can be offloaded to CPU and loaded again when needed. This registration also applies to handles outside a stage cache. |
| [`TmpCached`](../crates/thoosa_common/src/runner/tmp_cache.rs) | Holds a temporary collection of managed data handles, for example intermediate chunk results. It keeps those results visible to residency management while asynchronous work continues. |

Caching decides whether a computation can be reused; residency decides where its data lives. Keep large tensors, or many small tensors, in managed handles across asynchronous waits so OOM handling can reclaim GPU memory. Active input residency leases protect the device copies needed by a running attempt; device-reset invalidation can still discard copies from a failed device.

## CUDA backends

| Entity | Role |
| --- | --- |
| [`CudaActorHandle`](../crates/thoosa_common/src/cuda_copier.rs) | Common asynchronous interface for tensor copies and backend restart under recovery authority. `CudaLocalActorHandle` implements it in-process; [`CudaIPCActorHandle`](../crates/thoosa_common/src/ipc/cuda_process.rs) sends requests to a worker. Local stream access is a separate `LocalCudaExecution` capability. |
| [`CudaCopier`](../crates/thoosa_common/src/cuda_copier.rs) | Implements process-local CPU/GPU tensor transfers using pooled pinned host buffers and CUDA streams. It also supplies execution streams. It is used by the Local backend and inside IPC workers. |
| [`CudaWorker`](../crates/thoosa_common/src/ipc/cuda_wokrer.rs) | Runs the worker-side service for one GPU in IPC mode. It holds actual GPU tensors and installed stage endpoints, executes requests, and returns results or tensor references. The application refers to worker tensors by keys and worker identity. |

IPC tensor data travels through a shared CPU memory arena, while requests travel over Unix sockets. Worker generations prevent references to tensors from a retired worker from being used with its replacement. See [Local and IPC modes](ipc.md) for backend and endpoint recovery details.

## Runtime and recovery

[`Runtime`](../crates/thoosa_common/src/runner/global_ctx.rs) owns device services, runners, caches, residency management, and recovery coordination. `RunSession` binds a run identity to that runtime; `GlobalCtx` provides compatibility access to the installed runtime and session.

[`OOMHandler`](../crates/thoosa_common/src/runner/oom_handler.rs) retries recoverable out-of-memory failures using caller-supplied adjusters. Depending on the operation, these can reduce concurrency, free GPU memory, split input into chunks, or reduce a stage's memory requirements. The handler collects partial results when processing chunks. See [OOM handling](sync_oom.md) for the retry flow.

Fatal CUDA errors follow a separate recovery path: IPC mode can retire the worker, invalidate its GPU data, and start a replacement. Local mode requests application shutdown because the process's CUDA state may be unusable.

A typical GPU computation follows this path:

1. An `AsyncStage` requests dependencies, optionally through `Cached::run_cached`.
2. It invokes `OOMHandler` with a computation, managed input, context, and adjusters.
3. `SyncStageRunner` admits an attempt; the execution layer runs a locally or dispatches an IPC endpoint request.
4. The computation finishes and CUDA work is synchronized before execution resources are released. Successful output returns to the async stage; retryable OOM enters the adjustment loop.
5. Managed results remain available for reuse or CPU offload while later stages run.
