//! Asynchronous handle for rusqlite library.
//!
//! This is a revised version of tokio-rusqlite's core with better panic/error
//! handling for dispatched closures.

#![forbid(unsafe_code)]
#![warn(
    clippy::await_holding_lock,
    clippy::cargo_common_metadata,
    clippy::dbg_macro,
    clippy::empty_enum,
    clippy::enum_glob_use,
    clippy::inefficient_to_string,
    clippy::mem_forget,
    clippy::mutex_integer,
    clippy::needless_continue,
    clippy::todo,
    clippy::unimplemented,
    clippy::wildcard_imports,
    missing_docs,
    missing_debug_implementations,
    unreachable_pub
)]

use tokio::sync::mpsc::{self, Receiver, Sender};

// Bound queued closures; waiting callers keep ownership until asynchronous admission.
const REQUEST_QUEUE_CAPACITY: usize = 64;
use std::{
    any::Any,
    fmt::{self, Debug, Display},
    path::Path,
    thread,
};
use tokio::sync::oneshot;

pub(crate) use rusqlite::{self, *};

/// Information extracted from a panic payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PanicInfo {
    message: String,
}

impl PanicInfo {
    fn from_payload(payload: Box<dyn Any + Send + 'static>) -> Self {
        let message = if let Some(s) = payload.downcast_ref::<&'static str>() {
            (*s).to_string()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "panic with non-string payload".to_string()
        };

        Self { message }
    }

    /// Best-effort panic message.
    pub(crate) fn message(&self) -> &str {
        &self.message
    }
}

impl Display for PanicInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for PanicInfo {}

/// Represents the errors specific for this library.
#[derive(Debug)]
#[non_exhaustive]
pub(crate) enum Error<E = rusqlite::Error> {
    /// The connection is closed and can no longer accept new work.
    ConnectionClosed,

    /// The background worker thread disappeared after work was submitted.
    ///
    /// This should be rare after catching panics around dispatched closures.
    WorkerGone,

    /// An error occurred while closing the SQLite connection.
    ///
    /// This contains the [`Connection`] for retry purposes and the underlying
    /// [`rusqlite::Error`].
    #[cfg_attr(not(test), expect(dead_code, reason = "Adapter regression tests exercise this API; production uses the narrower SqlStore interface."))]
    Close((Connection, rusqlite::Error)),

    /// The dispatched closure panicked, but the worker thread survived and the
    /// wrapped SQLite connection remains usable.
    Panic(PanicInfo),

    /// An application-specific error occurred.
    Error(E),
}

impl<E: Display> Display for Error<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ConnectionClosed => write!(f, "ConnectionClosed"),
            Error::WorkerGone => write!(f, "WorkerGone"),
            Error::Close((_, e)) => write!(f, "Close((Connection, \"{e}\"))"),
            Error::Panic(p) => write!(f, "Panic(\"{}\")", p.message()),
            Error::Error(e) => write!(f, "Error(\"{e}\")"),
        }
    }
}

impl<E> std::error::Error for Error<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::ConnectionClosed => None,
            Error::WorkerGone => None,
            Error::Close((_, e)) => Some(e),
            Error::Panic(p) => Some(p),
            Error::Error(e) => Some(e),
        }
    }
}

impl From<rusqlite::Error> for Error<rusqlite::Error> {
    fn from(value: rusqlite::Error) -> Self {
        Error::Error(value)
    }
}

/// The result returned on method calls in this crate.
#[cfg_attr(not(test), expect(dead_code, reason = "Adapter regression tests exercise this API; production uses the narrower SqlStore interface."))]
pub(crate) type Result<T, E = rusqlite::Error> = std::result::Result<T, Error<E>>;

type CallFn = Box<dyn FnOnce(&mut rusqlite::Connection) + Send + 'static>;

enum Message {
    Execute(CallFn),
    #[cfg_attr(not(test), expect(dead_code, reason = "Adapter regression tests exercise this API; production uses the narrower SqlStore interface."))]
    Close(oneshot::Sender<rusqlite::Result<()>>),
}

#[derive(Debug, Clone, Copy)]
enum DispatchError {
    ConnectionClosed,
    WorkerGone,
}

/// A handle to call functions in a background thread.
#[derive(Clone)]
pub(crate) struct Connection {
    sender: Sender<Message>,
}

impl Connection {
    /// Open a new connection to a SQLite database.
    #[expect(dead_code, reason = "Preserve the adapter connection-opening API alongside the constructor used by SqlStore.")]
    pub(crate) async fn open<P: AsRef<Path>>(path: P) -> std::result::Result<Self, rusqlite::Error> {
        let path = path.as_ref().to_owned();
        start(move || rusqlite::Connection::open(path)).await
    }

    /// Open a new in-memory connection.
    #[cfg_attr(not(test), expect(dead_code, reason = "Adapter regression tests exercise this API; production uses the narrower SqlStore interface."))]
    pub(crate) async fn open_in_memory() -> std::result::Result<Self, rusqlite::Error> {
        start(rusqlite::Connection::open_in_memory).await
    }

    /// Open a new connection with flags.
    pub(crate) async fn open_with_flags<P: AsRef<Path>>(path: P, flags: OpenFlags) -> std::result::Result<Self, rusqlite::Error> {
        let path = path.as_ref().to_owned();
        start(move || rusqlite::Connection::open_with_flags(path, flags)).await
    }

    /// Open a new connection with flags and VFS.
    #[expect(dead_code, reason = "Preserve the adapter connection-opening API alongside the constructor used by SqlStore.")]
    pub(crate) async fn open_with_flags_and_vfs<P: AsRef<Path>>(path: P, flags: OpenFlags, vfs: &str) -> std::result::Result<Self, rusqlite::Error> {
        let path = path.as_ref().to_owned();
        let vfs = vfs.to_owned();
        start(move || rusqlite::Connection::open_with_flags_and_vfs(path, flags, &*vfs)).await
    }

    /// Open a new in-memory connection with flags.
    #[expect(dead_code, reason = "Preserve the adapter connection-opening API alongside the constructor used by SqlStore.")]
    pub(crate) async fn open_in_memory_with_flags(flags: OpenFlags) -> std::result::Result<Self, rusqlite::Error> {
        start(move || rusqlite::Connection::open_in_memory_with_flags(flags)).await
    }

    /// Open a new in-memory connection with flags and VFS.
    #[expect(dead_code, reason = "Preserve the adapter connection-opening API alongside the constructor used by SqlStore.")]
    pub(crate) async fn open_in_memory_with_flags_and_vfs(flags: OpenFlags, vfs: &str) -> std::result::Result<Self, rusqlite::Error> {
        let vfs = vfs.to_owned();
        start(move || rusqlite::Connection::open_in_memory_with_flags_and_vfs(flags, &*vfs)).await
    }

    fn map_dispatch_error<E>(e: DispatchError) -> Error<E> {
        match e {
            DispatchError::ConnectionClosed => Error::ConnectionClosed,
            DispatchError::WorkerGone => Error::WorkerGone,
        }
    }

    async fn dispatch<F, R>(&self, function: F) -> std::result::Result<std::thread::Result<R>, DispatchError>
    where
        F: FnOnce(&mut rusqlite::Connection) -> R + Send + 'static,
        R: Send + 'static,
    {
        // Cancellation before reserve completes submits nothing. Once sent, the
        // worker owns the request and executes it even if the reply is abandoned.
        let permit = self.sender.reserve().await.map_err(|_| DispatchError::ConnectionClosed)?;
        let (sender, receiver) = oneshot::channel::<std::thread::Result<R>>();
        permit.send(Message::Execute(Box::new(move |conn| {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| function(conn)));
            let _ = sender.send(outcome);
        })));

        receiver.await.map_err(|_| DispatchError::WorkerGone)
    }

    /// Call a function in the background thread and get the result asynchronously.
    ///
    /// If the closure panics via unwind, the panic is converted to `Error::Panic`
    /// and the wrapped SQLite connection stays alive for future calls.
    pub(crate) async fn call<F, R, E>(&self, function: F) -> std::result::Result<R, Error<E>>
    where
        F: FnOnce(&mut rusqlite::Connection) -> std::result::Result<R, E> + Send + 'static,
        R: Send + 'static,
        E: Send + 'static,
    {
        match self.dispatch(function).await.map_err(Self::map_dispatch_error)? {
            Ok(inner) => inner.map_err(Error::Error),
            Err(payload) => Err(Error::Panic(PanicInfo::from_payload(payload))),
        }
    }

    /// Call a function in the background thread and return the value directly.
    ///
    /// If the closure panics, the panic is rethrown on the caller side while the
    /// worker thread stays alive.
    ///
    /// Calling this on a closed connection still panics.
    #[cfg_attr(not(test), expect(dead_code, reason = "Adapter regression tests exercise this API; production uses the narrower SqlStore interface."))]
    pub(crate) async fn call_unwrap<F, R>(&self, function: F) -> R
    where
        F: FnOnce(&mut rusqlite::Connection) -> R + Send + 'static,
        R: Send + 'static,
    {
        match self.dispatch(function).await {
            Ok(Ok(value)) => value,
            Ok(Err(payload)) => std::panic::resume_unwind(payload),
            Err(DispatchError::ConnectionClosed) => {
                panic!("database connection should be open")
            }
            Err(DispatchError::WorkerGone) => {
                panic!("database worker thread exited unexpectedly")
            }
        }
    }

    /// Close the database connection.
    ///
    /// If a clone already closed it, this returns `Ok(())`.
    #[cfg_attr(not(test), expect(dead_code, reason = "Adapter regression tests exercise this API; production uses the narrower SqlStore interface."))]
    pub(crate) async fn close(self) -> Result<()> {
        let (sender, receiver) = oneshot::channel::<rusqlite::Result<()>>();

        if self.sender.send(Message::Close(sender)).await.is_err() {
            return Ok(());
        }

        match receiver.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(Error::Close((self, e))),
            Err(_) => Err(Error::WorkerGone),
        }
    }
}

impl Debug for Connection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Connection").finish()
    }
}

impl From<rusqlite::Connection> for Connection {
    fn from(conn: rusqlite::Connection) -> Self {
        let (sender, receiver) = mpsc::channel::<Message>(REQUEST_QUEUE_CAPACITY);
        thread::spawn(move || event_loop(conn, receiver));
        Self { sender }
    }
}

async fn start<F>(open: F) -> rusqlite::Result<Connection>
where
    F: FnOnce() -> rusqlite::Result<rusqlite::Connection> + Send + 'static,
{
    let (sender, receiver) = mpsc::channel::<Message>(REQUEST_QUEUE_CAPACITY);
    let (result_sender, result_receiver) = oneshot::channel::<rusqlite::Result<()>>();

    thread::spawn(move || {
        let conn = match open() {
            Ok(c) => c,
            Err(e) => {
                let _ = result_sender.send(Err(e));
                return;
            }
        };

        if result_sender.send(Ok(())).is_err() {
            return;
        }

        event_loop(conn, receiver);
    });

    result_receiver.await.expect("sqlite startup task unexpectedly dropped")?;
    Ok(Connection { sender })
}

fn event_loop(mut conn: rusqlite::Connection, mut receiver: Receiver<Message>) {
    while let Some(message) = receiver.blocking_recv() {
        match message {
            Message::Execute(f) => {
                // User panics are caught inside `dispatch` before they can escape.
                f(&mut conn);
            }
            Message::Close(reply) => match conn.close() {
                Ok(()) => {
                    let _ = reply.send(Ok(()));
                    break;
                }
                Err((c, e)) => {
                    conn = c;
                    let _ = reply.send(Err(e));
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::Duration;

    #[tokio::test]
    async fn test_queue_backpressure_and_accepted_request_ownership() -> anyhow::Result<()> {
        for from_existing in [false, true] {
            let connection = if from_existing { Connection::from(rusqlite::Connection::open_in_memory()?) } else { Connection::open_in_memory().await? };
            let (started_tx, started_rx) = oneshot::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let worker_connection = connection.clone();
            let first = tokio::spawn(async move {
                worker_connection
                    .call_unwrap(move |_| {
                        let _ = started_tx.send(());
                        let _ = release_rx.recv();
                    })
                    .await;
            });
            tokio::time::timeout(Duration::from_secs(5), started_rx).await??;
            let completed = Arc::new(AtomicUsize::new(0));
            let mut queued = Vec::new();
            for _ in 0..REQUEST_QUEUE_CAPACITY {
                let completed = Arc::clone(&completed);
                let mut call = Box::pin(connection.call_unwrap(move |_| {
                    completed.fetch_add(1, Ordering::SeqCst);
                }));
                assert!(futures::poll!(call.as_mut()).is_pending());
                queued.push(call);
            }
            assert_eq!(connection.sender.capacity(), 0);
            let retained = Arc::new(());
            let weak = Arc::downgrade(&retained);
            let mut waiting = Box::pin(connection.call_unwrap(move |_| {
                let _retained = retained;
                panic!("cancelled admission must never execute");
            }));
            assert!(futures::poll!(waiting.as_mut()).is_pending());
            drop(waiting);
            assert!(weak.upgrade().is_none(), "a caller waiting for capacity still owns its request");
            // Abandon replies after admission: accepted operations must still finish in FIFO order.
            drop(queued);
            release_tx.send(())?;
            tokio::time::timeout(Duration::from_secs(5), first).await??;
            tokio::time::timeout(Duration::from_secs(5), connection.call_unwrap(|_| ())).await?;
            assert_eq!(completed.load(Ordering::SeqCst), REQUEST_QUEUE_CAPACITY);
            connection.clone().close().await?;
            assert!(matches!(connection.call(|_| Ok::<_, rusqlite::Error>(())).await, Err(Error::ConnectionClosed)));
        }
        Ok(())
    }
}
