---
name: rust-async-review
description: >
  Review, debug, and refactor Rust async code, especially Tokio-based code.
  Use for async/await, tokio::spawn, JoinSet, JoinHandle, select!, channels,
  Mutex/RwLock/Semaphore usage, cancellation safety, task lifetime, Send/Sync,
  shared state, backpressure, deadlocks, starvation, blocking work inside async
  tasks, async streams, and concurrency/performance bugs. Also use when asked
  why async tasks appear sequential, hang, leak, fail to make progress, or are
  unexpectedly serialized. Prefer concrete concurrency reasoning over generic
  style advice.
---

# Rust async review

Review Rust async code for correctness first, then cancellation behavior,
resource lifetime, liveness, and performance.

Assume Tokio unless the code clearly uses a different runtime.

Do not mechanically rewrite working async code. Identify concrete risks,
explain the execution scenario that triggers them, and propose the smallest
correct fix.

## Review priorities

Check in this order:

1. correctness and data races;
2. deadlocks and progress/liveness;
3. cancellation safety;
4. task ownership and lifetime;
5. `Send` / `Sync` and runtime compatibility;
6. backpressure and bounded concurrency;
7. blocking or CPU-heavy work on async executor threads;
8. unnecessary serialization and lost concurrency;
9. shutdown behavior and cleanup;
10. performance issues.

## Mental model

When analyzing async Rust, distinguish:

```text
future construction
        ↓
future polling
        ↓
suspension at .await
        ↓
executor schedules another task
        ↓
future is polled again when woken
```

Creating multiple futures does not imply they execute concurrently.

```rust
let a = do_a();
let b = do_b();
```

constructs two futures but runs neither until polled.

This runs them concurrently in the same task:

```rust
let (a, b) = tokio::join!(do_a(), do_b());
```

This creates independently scheduled tasks:

```rust
let a = tokio::spawn(do_a());
let b = tokio::spawn(do_b());
let (a, b) = tokio::try_join!(a, b)?;
```

Do not confuse concurrency with parallelism.

## Locks across `.await`

Always inspect lock guards whose lifetime can cross an `.await`.

Suspicious:

```rust
let guard = state.lock().await;
do_io().await;
use_state(&guard);
```

Risks:
- unrelated tasks are blocked for the duration of I/O;
- lock ordering can deadlock;
- throughput collapses;
- cancellation can leave higher-level state inconsistent.

Prefer extracting the required data first:

```rust
let value = {
    let guard = state.lock().await;
    guard.value.clone()
};

do_io(value).await;
```

Do not blindly say “never hold a lock across `.await`.” Sometimes it is
semantically required. In that case:
- verify lock ordering;
- verify no awaited operation can re-enter code needing the same lock;
- minimize the critical section;
- document why the lock must span the await.

For `std::sync::Mutex` in async code, use it only when the critical section is
short and never spans `.await`.

## Shared state

Ask whether shared mutable state is necessary.

Prefer:
- ownership by one task;
- message passing;
- immutable `Arc<T>`;
- atomics for simple independent state;
- partitioned/sharded state;

before defaulting to `Arc<Mutex<T>>`.

For `RwLock`, check whether write contention makes it worse than `Mutex`.
Do not assume `RwLock` is automatically faster.

Check for:
- nested locking;
- inconsistent lock order;
- lock acquisition inside callbacks;
- holding guards while sending to channels;
- awaiting another task while holding a guard.

## Deadlocks

For every lock, semaphore, or resource acquisition, identify:

```text
resource acquired
        ↓
code executed while held
        ↓
awaits performed
        ↓
other resources acquired
        ↓
release point
```

Look for cycles:

```text
Task A: lock X → await Y
Task B: lock Y → await X
```

Also consider logical deadlocks without mutexes:

```text
Task A waits for a message from Task B
Task B waits for Task A's JoinHandle
```

or:

```text
producer waits for a semaphore permit
consumer waits for producer completion
```

When reporting a deadlock, give the concrete wait cycle.

## Cancellation safety

Every `.await` is a possible cancellation point when the future can be dropped,
for example by:
- `tokio::select!`;
- timeout;
- task abort;
- dropped parent future;
- shutdown;
- racing futures.

For a multi-step mutation, ask:

```text
If this future is dropped at each await point, is state still valid?
```

Common risks:
- dequeue item, await, then record completion;
- partially consume a protocol message;
- reserve resource, await, then register ownership;
- mutate shared state before an await and restore afterward;
- perform half of a transaction.

Prefer:
- atomic state transitions;
- RAII guards;
- commit-after-success;
- idempotent retry;
- explicit cancellation handling;
- dedicated tasks for non-cancel-safe operations.

## `tokio::select!`

Review every branch for cancellation safety.

```rust
tokio::select! {
    value = receiver.recv() => { /* ... */ }
    _ = shutdown.cancelled() => { /* ... */ }
}
```

Ask what happens to every non-selected future when another branch wins.

Check for:
- non-cancel-safe branch futures;
- biased selection and starvation;
- loops that repeatedly recreate and discard partial work;
- shutdown branches that cannot win because another branch is always ready.

Do not assume `select!` means fair scheduling.

## Task lifetime and detached work

Every spawned task needs an ownership story.

Review carefully:

```rust
tokio::spawn(async move {
    /* ... */
});
```

when the `JoinHandle` is discarded.

Ask:
- Who observes task failure?
- Who cancels it on shutdown?
- Can it outlive the state it logically belongs to?
- Is detached execution intentional?

Prefer storing handles in a `JoinSet`, task manager, owning struct, or supervisor.

## `JoinHandle` and `JoinSet`

Distinguish task-join failure from the task's own returned error.

```rust
let task_result = handle.await
    .context("worker task panicked or was cancelled")?;

let value = task_result
    .context("worker operation failed")?;
```

For `JoinSet`, inspect:
- whether all tasks are drained;
- what happens after one task fails;
- whether remaining tasks should be aborted;
- whether shutdown waits for them;
- whether results are lost if the set is dropped.

## `Send` / `Sync`

`tokio::spawn` generally requires a `Send + 'static` future.

If a future is unexpectedly not `Send`, inspect values that live across
`.await`, especially:
- lock guards;
- `Rc`;
- `RefCell`;
- non-`Send` FFI handles;
- thread-affine resources.

Do not “fix” a `Send` error with unsafe trait implementations.

Instead:
- reduce the non-`Send` value's lifetime;
- use `Arc` when semantically correct;
- use `LocalSet` / `spawn_local` for thread-local futures;
- redesign ownership.

Do not add `'static` or clone large state merely to satisfy the compiler
without checking whether the ownership model is correct.

## Lost concurrency

Check for intentionally or accidentally sequential loops:

```rust
for item in items {
    process(item).await?;
}
```

If concurrency is intended, consider:
- `JoinSet`;
- `FuturesUnordered`;
- stream `.buffer_unordered(n)`;
- explicitly spawned tasks;
- `join!` for a small fixed set.

Also inspect:

```rust
let futures = items.iter().map(process).collect::<Vec<_>>();

for future in futures {
    future.await?;
}
```

This is still sequential unless those futures themselves spawn work.

When `buffer_unordered` appears sequential, inspect whether concurrency is
actually serialized by:
- a mutex;
- a semaphore with one permit;
- a single worker;
- blocking calls;
- dependency ordering;
- synchronous FFI;
- a single CUDA stream.

## Bounded concurrency

Unbounded spawning is suspicious:

```rust
for item in items {
    tokio::spawn(process(item));
}
```

for potentially large inputs.

Prefer an explicit bound using:
- `Semaphore`;
- `buffer_unordered(limit)`;
- a fixed worker pool;
- a bounded channel.

Check that the limit controls the resource that actually matters.

Task count is not necessarily equivalent to:
- GPU memory;
- open files;
- network connections;
- DB transactions;
- pinned host memory.

## Semaphores

Review:
- where permits are acquired;
- how long they are held;
- whether permits survive unrelated waits;
- whether `OwnedSemaphorePermit` is needed;
- whether cancellation releases them correctly;
- whether retry paths accidentally keep permits;
- whether tasks wait for input while consuming capacity.

Bad shape:

```rust
let permit = semaphore.acquire().await?;
let item = receiver.recv().await;
process(item).await;
drop(permit);
```

If waiting for the item does not need scarce capacity, acquire the permit later.

## Channels

### `mpsc`

Check:
- bounded vs unbounded;
- backpressure;
- sender clones keeping the channel alive unexpectedly;
- receiver ownership;
- shutdown after all senders drop.

Avoid `unbounded_channel` unless memory growth is bounded elsewhere.

### `broadcast`

Remember:
- slow receivers can lag;
- lagging receivers can lose messages;
- each receiver has its own cursor.

Handle `Lagged` intentionally.

### `watch`

Use it for latest-value state, not event history.

Remember:
- intermediate values may be skipped;
- `changed().await` and `borrow_and_update()` semantics matter;
- do not hold a borrow across unrelated awaits.

### `oneshot`

Use for one response or acknowledgement.
Check what happens when sender or receiver is dropped.

## Backpressure

For each producer/consumer pipeline identify:
- producer rate;
- queue capacity;
- consumer rate;
- overflow/full behavior;
- shutdown behavior.

If the queue is unbounded, identify what prevents unbounded memory growth.

If bounded, identify whether full means:
- producer waits;
- work is dropped;
- latest value replaces old;
- load is rejected.

## Blocking work

Do not perform long blocking operations directly on Tokio worker threads.

Suspicious work includes:
- synchronous I/O;
- process waits;
- compression;
- CPU-heavy loops;
- large serialization/deserialization;
- expensive FFI calls;
- host-blocking CUDA/library calls.

Use `spawn_blocking` when appropriate:

```rust
let value = tokio::task::spawn_blocking(move || expensive_sync_work())
    .await??;
```

Do not wrap tiny synchronous functions in `spawn_blocking`.

For sustained CPU-parallel work, consider a dedicated CPU pool such as Rayon.

For FFI, determine whether the call:
- quickly enqueues asynchronous work;
- synchronizes a device/stream;
- blocks on I/O;
- can run for a long time.

## Async + CUDA/libtorch

When reviewing async Rust around CUDA or libtorch:
- distinguish host task concurrency from CUDA stream concurrency;
- determine whether FFI calls synchronize;
- check whether Tokio tasks serialize through one CUDA stream;
- check for global locks around CUDA work;
- ensure buffers outlive asynchronous device operations;
- verify stream/event synchronization at ownership boundaries;
- do not assume return from an FFI call means GPU work is complete.

For CUDA OOM/retry logic:
- verify failed-attempt resources are released before retry;
- verify permits/capacity counters are restored;
- verify stale tasks cannot use a reset/invalid context;
- verify CUDA context reset coordinates all affected tasks.

## Timeouts

A timeout cancels by dropping the inner future:

```rust
tokio::time::timeout(duration, operation()).await
```

Therefore review the inner operation for cancellation safety.

Do not assume timeout means underlying work stopped:
- spawned child tasks can keep running;
- blocking calls can continue;
- GPU kernels can continue;
- remote requests can remain in flight.

If real cancellation is required, provide an explicit mechanism.

## Shutdown

For long-lived tasks, prefer explicit cooperative shutdown.

Typical shape:

```rust
loop {
    tokio::select! {
        _ = shutdown.cancelled() => break,
        item = receiver.recv() => {
            let Some(item) = item else { break };
            process(item).await;
        }
    }
}
```

Check:
- child task shutdown propagation;
- cleanup completion;
- channel close order;
- producer shutdown before consumer drain;
- cleanup operations that may hang forever.

## Atomics

Use atomics only when the state can be represented correctly as atomic values.

Review:
- whether several atomic fields must change consistently;
- whether `Relaxed` ordering is really sufficient;
- whether the atomic participates in synchronization;
- whether a mutex/channel would make invariants clearer.

Do not recommend stronger ordering merely “to be safe.”
Explain the required happens-before relationship.

For statistics/counters with no synchronization meaning, `Relaxed` is often
appropriate.

## Async traits and lifetimes

When reviewing async trait code:
- distinguish native async traits from boxed futures;
- inspect hidden lifetime captures;
- avoid unnecessary `'static` bounds;
- avoid boxing unless dynamic dispatch/type erasure requires it;
- verify the future has the required `Send` behavior.

For HRTB or async callbacks, inspect whether returned futures borrow inputs.

Do not solve lifetime errors by cloning large state unless that ownership is
actually desired.

## Streams

For `Stream` code:
- distinguish ordered vs unordered buffering;
- inspect concurrency limits;
- check how errors terminate or flow through the stream;
- inspect cancellation of in-flight futures.

`buffered(n)` preserves order and can cause head-of-line blocking.

`buffer_unordered(n)` permits out-of-order completion.

Choose based on semantics, not only performance.

## Error handling

Preserve distinctions between:
- task failure;
- task cancellation;
- panic;
- operation error;
- timeout;
- channel closure.

Add context at async boundaries.

Do not stringify typed errors when higher layers may need to inspect them.

If the project has an `anyhow-error-handling` skill, apply its conventions too
when the task involves error propagation or classification.

## Panic behavior

A panic inside a spawned Tokio task normally becomes a `JoinError`; it does not
automatically panic the spawning task.

Check whether the program:
- observes panics;
- logs them;
- restarts workers;
- intentionally propagates them;
- silently drops task handles.

Do not use panic for expected runtime failures.

## Resource cleanup

Prefer RAII for:
- semaphore permits;
- registrations;
- temporary state;
- resource ownership.

Remember that `Drop` cannot `.await`.

If cleanup is asynchronous, expose an explicit shutdown/close path.

## Fairness and starvation

Look for executor-monopolizing loops:

```rust
loop {
    do_sync_work();
}
```

Also inspect async loops whose operations are always immediately ready and never
yield.

`tokio::task::yield_now().await` can sometimes help, but do not use explicit
yielding to hide a flawed architecture.

## Performance

Only after correctness, inspect:
- excessive task spawning;
- allocation per iteration;
- unnecessary boxing;
- repeated channel hops;
- global locks;
- broad critical sections;
- tiny tasks dominated by scheduling overhead;
- duplicate synchronization;
- serial awaits that should be concurrent;
- excessive cloning;
- blocking work on executor threads.

More concurrency is not automatically better. It can increase:
- contention;
- memory pressure;
- latency;
- GPU OOM frequency;
- scheduling overhead.

## Instrumentation

For difficult async bugs, use targeted observability:
- `tracing`;
- `tokio-console`;
- task IDs/names;
- timestamps around waits;
- semaphore permit counts;
- queue depths;
- worker lifecycle logs;
- explicit state-transition logs.

For suspected deadlocks capture:
- task identity;
- resource awaited;
- resource currently held;
- time spent waiting.

## Testing

Use `#[tokio::test]` and make runtime assumptions explicit when relevant:

```rust
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_name() {
    // ...
}
```

For concurrency bugs test:
- shutdown while work is active;
- channel closure;
- cancellation at await points;
- task failure;
- permit exhaustion;
- retries;
- high contention.

Avoid tiny wall-clock timeouts that produce flaky tests.

`loom` can model low-level synchronization with Loom-compatible primitives, but
do not suggest it as a drop-in checker for arbitrary Tokio async code.

Miri is useful for undefined behavior and some correctness checks but is not a
general async deadlock detector and often cannot run FFI-heavy code.

## Suspicious patterns

Actively search for:

```rust
tokio::spawn(...); // handle discarded
```

```rust
let guard = mutex.lock().await;
something.await;
```

```rust
for x in xs {
    async_work(x).await;
}
```

when concurrency appears intended.

```rust
tokio::spawn(async move {
    blocking_ffi_call();
});
```

```rust
tokio::select! {
    _ = non_cancel_safe_operation() => { /* ... */ }
    // ...
}
```

```rust
let _ = sender.send(value).await;
```

when send failure matters.

```rust
Arc<Mutex<HashMap<...>>>
```

when one-task ownership or message passing may eliminate shared locking.

```rust
Semaphore::new(1)
```

when surrounding code expects actual concurrency.

```rust
join_all(...)
```

over a huge collection when bounded concurrency is required.

## Reporting findings

For each significant finding report:

1. **Problem**
2. **Why it can happen**
3. **Concrete execution sequence**
4. **Impact**
5. **Smallest safe fix**

Example:

```text
Problem:
The MutexGuard is held while awaiting `send_to_worker()`.

Execution:
Task A locks state → awaits worker response.
Task B receives the request → needs the same state lock before replying.
Task A waits for B; B waits for A.

Impact:
Deadlock.

Fix:
Copy the required state while locked, drop the guard, then await
`send_to_worker()`.
```

Avoid vague statements such as “this may cause concurrency issues” when the
actual execution sequence can be described.

## Refactoring policy

When modifying code:
- preserve semantics unless explicitly asked to redesign;
- make concurrency limits explicit;
- avoid introducing detached tasks;
- do not add `Arc<Mutex<_>>` merely to satisfy the borrow checker;
- do not clone large state merely to satisfy `'static`;
- never add unsafe `Send`/`Sync` simply to silence the compiler;
- do not replace async code with blocking code unless deliberately appropriate;
- prefer small, reviewable changes.

If several designs are valid, explain the tradeoff:
- ownership vs shared state;
- channel vs lock;
- spawned task vs in-task concurrency;
- ordered vs unordered processing;
- bounded throughput vs maximum parallelism.
