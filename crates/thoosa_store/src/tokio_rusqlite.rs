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
    future_incompatible,
    missing_docs,
    missing_debug_implementations,
    unreachable_pub
)]


use crossbeam_channel::{Receiver, Sender};
use std::{
    any::Any,
    fmt::{self, Debug, Display},
    path::Path,
    thread,
};
use tokio::sync::oneshot;

pub use rusqlite::{self, *};

/// Information extracted from a panic payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PanicInfo {
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
    pub fn message(&self) -> &str {
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
pub enum Error<E = rusqlite::Error> {
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
pub type Result<T, E = rusqlite::Error> = std::result::Result<T, Error<E>>;

type CallFn = Box<dyn FnOnce(&mut rusqlite::Connection) + Send + 'static>;

enum Message {
    Execute(CallFn),
    Close(oneshot::Sender<rusqlite::Result<()>>),
}

#[derive(Debug, Clone, Copy)]
enum DispatchError {
    ConnectionClosed,
    WorkerGone,
}

/// A handle to call functions in a background thread.
#[derive(Clone)]
pub struct Connection {
    sender: Sender<Message>,
}

impl Connection {
    /// Open a new connection to a SQLite database.
    pub async fn open<P: AsRef<Path>>(path: P) -> std::result::Result<Self, rusqlite::Error> {
        let path = path.as_ref().to_owned();
        start(move || rusqlite::Connection::open(path)).await
    }

    /// Open a new in-memory connection.
    pub async fn open_in_memory() -> std::result::Result<Self, rusqlite::Error> {
        start(rusqlite::Connection::open_in_memory).await
    }

    /// Open a new connection with flags.
    pub async fn open_with_flags<P: AsRef<Path>>(
        path: P,
        flags: OpenFlags,
    ) -> std::result::Result<Self, rusqlite::Error> {
        let path = path.as_ref().to_owned();
        start(move || rusqlite::Connection::open_with_flags(path, flags)).await
    }

    /// Open a new connection with flags and VFS.
    pub async fn open_with_flags_and_vfs<P: AsRef<Path>>(
        path: P,
        flags: OpenFlags,
        vfs: &str,
    ) -> std::result::Result<Self, rusqlite::Error> {
        let path = path.as_ref().to_owned();
        let vfs = vfs.to_owned();
        start(move || rusqlite::Connection::open_with_flags_and_vfs(path, flags, &*vfs)).await
    }

    /// Open a new in-memory connection with flags.
    pub async fn open_in_memory_with_flags(
        flags: OpenFlags,
    ) -> std::result::Result<Self, rusqlite::Error> {
        start(move || rusqlite::Connection::open_in_memory_with_flags(flags)).await
    }

    /// Open a new in-memory connection with flags and VFS.
    pub async fn open_in_memory_with_flags_and_vfs(
        flags: OpenFlags,
        vfs: &str,
    ) -> std::result::Result<Self, rusqlite::Error> {
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
        let (sender, receiver) = oneshot::channel::<std::thread::Result<R>>();

        self.sender
            .send(Message::Execute(Box::new(move |conn| {
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| function(conn)));
                let _ = sender.send(outcome);
            })))
            .map_err(|_| DispatchError::ConnectionClosed)?;

        receiver.await.map_err(|_| DispatchError::WorkerGone)
    }

    /// Call a function in the background thread and get the result asynchronously.
    ///
    /// If the closure panics via unwind, the panic is converted to `Error::Panic`
    /// and the wrapped SQLite connection stays alive for future calls.
    pub async fn call<F, R, E>(&self, function: F) -> std::result::Result<R, Error<E>>
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
    pub async fn call_unwrap<F, R>(&self, function: F) -> R
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
    pub async fn close(self) -> Result<()> {
        let (sender, receiver) = oneshot::channel::<rusqlite::Result<()>>();

        if self.sender.send(Message::Close(sender)).is_err() {
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
        let (sender, receiver) = crossbeam_channel::unbounded::<Message>();
        thread::spawn(move || event_loop(conn, receiver));
        Self { sender }
    }
}

async fn start<F>(open: F) -> rusqlite::Result<Connection>
where
    F: FnOnce() -> rusqlite::Result<rusqlite::Connection> + Send + 'static,
{
    let (sender, receiver) = crossbeam_channel::unbounded::<Message>();
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

fn event_loop(mut conn: rusqlite::Connection, receiver: Receiver<Message>) {
    while let Ok(message) = receiver.recv() {
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