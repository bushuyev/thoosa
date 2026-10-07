---
name: anyhow-error-handling
description: >
  Apply Rust error-handling best practices using anyhow. Use when writing,
  reviewing, or refactoring application-level Rust error handling involving
  anyhow::Result, anyhow::Error, Context, bail!, ensure!, third-party errors,
  custom typed errors, recoverable error categories, downcasting, backtraces,
  async/task boundaries, or IPC error propagation. Also use when deciding
  whether an error should remain typed (for example with thiserror) or be
  converted to anyhow. Do not blindly replace public library error types with
  anyhow::Error.
---

# Anyhow error-handling best practices

Use `anyhow` primarily at application and orchestration boundaries, where callers
usually need a useful diagnostic rather than an exhaustive public error type.

Preserve typed errors wherever program logic needs to distinguish error kinds.

## Core rules

1. Prefer `anyhow::Result<T>` for application-level functions whose callers
   normally propagate or report failures rather than exhaustively match them.

2. Prefer a typed error (`enum`, usually derived with `thiserror`) when callers
   need to:
   - match variants;
   - inspect structured fields;
   - decide whether an error is retryable/recoverable;
   - determine whether a reset/restart/fallback is required;
   - expose a stable library API;
   - serialize the error across IPC or another process boundary.

3. Do not erase useful type information prematurely.

   Good:

   ```rust
   third_party_call()?;
   ```

   Good with context:

   ```rust
   third_party_call()
       .context("failed to load model metadata")?;
   ```

   Bad:

   ```rust
   third_party_call()
       .map_err(|e| anyhow::anyhow!("{e}"))?;
   ```

   The bad form stringifies the original error and loses its typed source.

4. Add context when crossing abstraction boundaries.

   Context should answer questions such as:
   - What operation was being attempted?
   - Which file/device/job/model/request was involved?
   - Which higher-level stage failed?

   Prefer:

   ```rust
   let config = std::fs::read_to_string(path)
       .with_context(|| format!("failed to read config {}", path.display()))?;
   ```

   over merely:

   ```rust
   let config = std::fs::read_to_string(path)?;
   ```

   Do not repeat the low-level error text in the context message.

5. Use `with_context` when constructing the message has nontrivial cost or needs
   captured values.

6. Never classify errors by matching their formatted text.

   Bad:

   ```rust
   if err.to_string().contains("out of memory") {
       ...
   }
   ```

   Prefer a typed error, typed marker, error code, or downcast.

## Third-party errors

When a dependency returns a useful error type such as `regex::Error`,
`std::io::Error`, `serde_json::Error`, or a database error:

### Generic failure is sufficient

Propagate it directly:

```rust
let regex = regex::Regex::new(pattern)?;
```

or add operation-level context:

```rust
let regex = regex::Regex::new(pattern)
    .with_context(|| format!("invalid internal regex pattern {pattern:?}"))?;
```

`anyhow` preserves the original error as part of the error chain.

### Higher layers need special handling

Do not convert the error to a string.

Either preserve/translate it into an application-specific typed error:

```rust
#[derive(Debug, thiserror::Error)]
enum AppError {
    #[error("configuration is invalid")]
    InvalidConfig {
        #[source]
        source: serde_json::Error,
    },

    #[error("CUDA context must be reset")]
    NeedsCudaReset {
        #[source]
        source: CudaError,
    },
}
```

or attach a typed context marker when that is a cleaner fit:

```rust
use anyhow::Context;

#[derive(Debug)]
struct NeedsCudaReset;

impl std::fmt::Display for NeedsCudaReset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CUDA context reset required")
    }
}

run_cuda_operation().context(NeedsCudaReset)?;
```

At a handling boundary:

```rust
if err.downcast_ref::<NeedsCudaReset>().is_some() {
    reset_cuda_context().await?;
}
```

`anyhow::Context` is designed so adding context does not prevent downcasting to
the original error type. A typed context value can itself also be downcast.

Use typed context markers sparingly. If the category has structured data,
multiple variants, or is part of a public API, prefer a real error enum.

## Combining custom errors with anyhow

It is normal to use both.

A useful architecture is:

```text
low-level/library/domain code
        |
        | typed Result<T, ModuleError>
        v
application/orchestration code
        |
        | anyhow::Result<T>
        | + Context
        v
top-level handler/logging/reporting
```

Example:

```rust
#[derive(Debug, thiserror::Error)]
enum StorageError {
    #[error("database is busy")]
    Busy,

    #[error("database corruption detected")]
    Corrupt,

    #[error(transparent)]
    Sqlite(#[from] rusqlite::Error),
}

fn read_record() -> Result<Record, StorageError> {
    // ...
}

fn run_job() -> anyhow::Result<()> {
    let record = read_record().context("failed to load input record")?;
    // ...
    Ok(())
}
```

At an application recovery boundary:

```rust
match run_job() {
    Ok(()) => {}
    Err(err) if matches!(
        err.downcast_ref::<StorageError>(),
        Some(StorageError::Busy)
    ) => {
        // retry/back off
    }
    Err(err) => return Err(err),
}
```

Do not create a giant application-wide error enum solely because every possible
dependency error needs a variant. Use typed variants for errors that carry
application semantics; let `anyhow` handle the rest.

## Nested/module error enums

Module-specific error enums are appropriate when the module has meaningful
failure semantics.

Prefer:

```rust
enum DatabaseError { ... }
enum WorkerError { ... }
enum ModelError { ... }
```

over one enormous enum containing every low-level error in the program.

A top-level typed enum is useful only when the application actually needs to
match those categories.

Do not create nesting merely for taxonomy.

## Recoverable flags and error properties

If code needs special properties such as:

```text
needs_cuda_ctx_reset
is_oom
retryable
worker_must_restart
```

prefer semantic variants or methods on a typed error rather than unrelated
boolean flags scattered through the code.

Example:

```rust
#[derive(Debug, thiserror::Error)]
enum ComputeError {
    #[error("CUDA out of memory")]
    CudaOom,

    #[error("CUDA context is unusable")]
    CudaContextLost,

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl ComputeError {
    fn needs_cuda_ctx_reset(&self) -> bool {
        matches!(self, Self::CudaContextLost)
    }

    fn is_oom(&self) -> bool {
        matches!(self, Self::CudaOom)
    }
}
```

However, do not put `anyhow::Error` inside a public library error enum merely as
a catch-all unless intentionally accepting an opaque application error.

When an application already has a useful custom error type, wrapping it in
`anyhow::Error` does not prevent later recovery through `downcast_ref`.

## Error inspection

At a boundary where special handling is required, prefer:

```rust
if let Some(err) = error.downcast_ref::<MyError>() {
    ...
}
```

Do not automatically call `root_cause()` before downcasting.

Reason: an application-specific typed context marker may be in the context
chain rather than be the root cause. `anyhow::Error::downcast_ref` can inspect
the anyhow error/context chain according to anyhow's downcasting semantics.

Use the error chain for diagnostics:

```rust
for cause in error.chain() {
    tracing::error!("{cause}");
}
```

Avoid relying on the textual order/content of that chain for control flow.

## `anyhow!`, `bail!`, and `ensure!`

Use `anyhow!` for genuinely ad-hoc application errors:

```rust
return Err(anyhow::anyhow!("worker {worker_id} disappeared"));
```

Use `bail!` for concise early returns:

```rust
if channels == 0 {
    anyhow::bail!("channel count must be nonzero");
}
```

Use `ensure!` for runtime preconditions:

```rust
anyhow::ensure!(
    buffer_size <= max_buffer_size,
    "buffer size {buffer_size} exceeds maximum {max_buffer_size}"
);
```

Do not use ad-hoc string errors for conditions that higher layers must
programmatically distinguish. Introduce a typed error instead.

## Assertions vs errors

Use an assertion/panic for violated programmer invariants that indicate a bug:

```rust
assert!(!ids.is_empty(), "internal invariant: ids must not be empty");
```

Return an error for runtime conditions that can legitimately occur because of
input, environment, I/O, resources, remote systems, or user configuration.

Do not turn every invariant into `anyhow::Error`, and do not panic for ordinary
runtime failures.

## Backtraces

Do not add a custom backtrace field merely because errors travel through many
in-process layers.

`anyhow::Error` can capture/preserve a backtrace when supported and when the
underlying error does not already provide one. Use the standard backtrace
environment variables when diagnostics require it.

Typical development/debugging setup:

```bash
RUST_BACKTRACE=1
```

or, when only error backtraces are desired:

```bash
RUST_LIB_BACKTRACE=1
```

If a project intentionally captures a backtrace at a specific semantic failure
site, preserve that behavior rather than removing it automatically.

Remember that an in-memory `anyhow::Error` and its backtrace are not an IPC wire
format.

## IPC / process boundaries

Do not attempt to serialize `anyhow::Error` as the application protocol.

Create an explicit serializable wire error containing the information needed by
the receiving process, for example:

```rust
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct WireError {
    kind: ErrorKind,
    message: String,
    causes: Vec<String>,
    backtrace: Option<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
enum ErrorKind {
    Generic,
    OutOfMemory,
    NeedsCudaContextReset,
    Retryable,
}
```

On the sending side:
- classify the typed error before serialization;
- retain a human-readable message and useful context chain;
- include a backtrace/location only if the protocol actually needs it.

On the receiving side:
- reconstruct application semantics from `kind`, not from `message`;
- use `anyhow` for additional local context if appropriate.

## Async and task boundaries

When awaiting spawned work, distinguish failure of the task itself from failure
returned by the task.

For example:

```rust
let result = tokio::spawn(async move {
    do_work().await
})
.await
.context("worker task failed to join")?;

result.context("worker operation failed")?;
```

If task cancellation or panic needs special behavior, inspect `JoinError`
explicitly instead of reducing it to a string.

Add context at concurrency boundaries where otherwise it would be unclear
which worker/task/stage failed.

## Logging and reporting

Normally return errors upward with `?` rather than logging them at every layer.
Log/report once at the boundary that owns the failure.

Avoid this pattern:

```rust
operation()
    .inspect_err(|e| tracing::error!("operation failed: {e}"))?;
```

when the caller will log the same error again.

Add context while propagating; log when handling.

Never include secrets, credentials, tokens, or unnecessarily sensitive data in
error context.

## Review procedure

When reviewing error-handling code, perform these checks in order:

1. Identify the error boundary:
   - library/domain;
   - module;
   - application orchestration;
   - top-level handler;
   - task/process/IPC boundary.

2. For every error conversion, ask:
   - Is type information being lost?
   - Is the source chain preserved?
   - Will a higher layer need to match this error?
   - Is useful operation context missing?

3. Search for suspicious patterns:
   - `map_err(|e| anyhow!("{e}"))`;
   - `format!("{e}")` used as an error conversion;
   - `.to_string().contains(...)`;
   - giant catch-all error enums;
   - repeated logging plus propagation;
   - uncontextualized file/I/O/database/network operations;
   - boolean error flags that should be semantic variants;
   - `unwrap`/`expect` on legitimate runtime failures.

4. Preserve third-party error types unless there is a deliberate semantic
   translation.

5. Add `context`/`with_context` at abstraction boundaries, not mechanically to
   every `?`.

6. If recovery logic exists, verify it depends on typed semantics rather than
   formatted messages.

7. If errors cross IPC, verify machine-readable classification survives the
   boundary.

8. If backtrace behavior matters, determine where it is captured and whether a
   custom capture is actually needed.

## Preferred refactoring style

When changing existing code:
- make the smallest change that improves semantics;
- preserve public APIs unless the task explicitly asks to change them;
- do not replace a useful custom error enum with anyhow just to reduce code;
- do not introduce a custom enum when ordinary propagation plus context is
  sufficient;
- show the resulting error chain and recovery behavior when that clarifies the
  change.

When proposing code, explain briefly why each error remains typed or becomes
opaque.
