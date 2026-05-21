//TODO rename  - remove _lock

#[macro_export]
macro_rules! sync_read_lock {
    ($lock:expr $(,)?) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        ($lock)
            .try_read_for($crate::LOCK_TIMEOUT)
            .expect(DESCR)
    }};

    ($lock:expr, $($msg:tt)+) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        ($lock)
            .try_read_for($crate::LOCK_TIMEOUT)
            .unwrap_or_else(|| {
                panic!("{DESCR}: {}", format_args!($($msg)+))
            })
    }};
}

#[macro_export]
macro_rules! sync_write_lock {
    ($lock:expr $(,)?) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        ($lock)
            .try_write_for($crate::LOCK_TIMEOUT)
            .expect(DESCR)
    }};

    ($lock:expr, $($msg:tt)+) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        ($lock)
            .try_write_for($crate::LOCK_TIMEOUT)
            .unwrap_or_else(|| {
                panic!("{DESCR}: {}", format_args!($($msg)+))
            })
    }};
}

#[macro_export]
macro_rules! sync_mutex_lock {
    ($lock:expr $(,)?) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        ($lock)
            .try_lock_for($crate::LOCK_TIMEOUT)
            .expect(DESCR)
    }};

    ($lock:expr, $($msg:tt)+) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        ($lock)
            .try_lock_for($crate::LOCK_TIMEOUT)
            .unwrap_or_else(|| {
                panic!("{DESCR}: {}", format_args!($($msg)+))
            })
    }};
}



#[macro_export]
macro_rules! async_read_lock {
    ($lock:expr $(,)?) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        tokio::time::timeout($crate::LOCK_TIMEOUT, ($lock).read())
            .await
            .expect(DESCR)
    }};

    ($lock:expr, $($msg:tt)+) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        tokio::time::timeout($crate::LOCK_TIMEOUT, ($lock).read())
            .await
            .unwrap_or_else(|_| {
                panic!("{DESCR}: {}", format_args!($($msg)+))
            })
    }};
}

#[macro_export]
macro_rules! async_write_lock {
    ($lock:expr $(,)?) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        tokio::time::timeout($crate::LOCK_TIMEOUT, ($lock).write())
            .await
            .expect(DESCR)
    }};

    ($lock:expr, $($msg:tt)+) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        tokio::time::timeout($crate::LOCK_TIMEOUT, ($lock).write())
            .await
            .unwrap_or_else(|_| {
                panic!("{DESCR}: {}", format_args!($($msg)+))
            })
    }};
}

#[macro_export]
macro_rules! async_mutex_lock {
    ($lock:expr $(,)?) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        tokio::time::timeout($crate::LOCK_TIMEOUT, ($lock).lock())
            .await
            .expect(DESCR)
    }};

    ($lock:expr, $($msg:tt)+) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        tokio::time::timeout($crate::LOCK_TIMEOUT, ($lock).lock())
            .await
            .unwrap_or_else(|_| {
                panic!("{DESCR}: {}", format_args!($($msg)+))
            })
    }};
}


//Tokio owned guards, for Arc<tokio::sync::RwLock<T>> / Arc<tokio::sync::Mutex<T>>:
#[macro_export]
macro_rules! async_read_owned_lock {
    ($lock:expr $(,)?) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        tokio::time::timeout($crate::LOCK_TIMEOUT, ($lock).read_owned())
            .await
            .expect(DESCR)
    }};

    ($lock:expr, $($msg:tt)+) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        tokio::time::timeout($crate::LOCK_TIMEOUT, ($lock).read_owned())
            .await
            .unwrap_or_else(|_| {
                panic!("{DESCR}: {}", format_args!($($msg)+))
            })
    }};
}

#[macro_export]
macro_rules! async_write_owned_lock {
    ($lock:expr $(,)?) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        tokio::time::timeout($crate::LOCK_TIMEOUT, ($lock).write_owned())
            .await
            .expect(DESCR)
    }};

    ($lock:expr, $($msg:tt)+) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        tokio::time::timeout($crate::LOCK_TIMEOUT, ($lock).write_owned())
            .await
            .unwrap_or_else(|_| {
                panic!("{DESCR}: {}", format_args!($($msg)+))
            })
    }};
}

#[macro_export]
macro_rules! async_mutex_owned_lock {
    ($lock:expr $(,)?) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        tokio::time::timeout($crate::LOCK_TIMEOUT, ($lock).lock_owned())
            .await
            .expect(DESCR)
    }};

    ($lock:expr, $($msg:tt)+) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        tokio::time::timeout($crate::LOCK_TIMEOUT, ($lock).lock_owned())
            .await
            .unwrap_or_else(|_| {
                panic!("{DESCR}: {}", format_args!($($msg)+))
            })
    }};
}


#[tracing::instrument(level="debug", skip_all, fields(agg_perf = %thoosa_shared::AggPerf("sync_lock", descr)))]
#[inline]
pub fn with_perf_log_<F, R>(f: F, descr: &str) -> R
where
    F: FnOnce() -> R,
{

    f()
}

#[tracing::instrument(level="debug", skip_all, fields(agg_perf = %thoosa_shared::AggPerf("async_lock", descr)))]
#[inline]
pub async fn with_async_perf_log_<F, R>(f: F, descr: &str) -> R
where
    F: AsyncFnOnce() -> R,
{
    f().await
}

#[macro_export]
macro_rules! perf_log {
    ($wrapped_fn:expr $(,)?) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        $crate::macros::with_perf_log_(
            || $wrapped_fn,
            DESCR,
        )
    }};

    ($wrapped_fn:expr, $($msg:tt)+) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        $crate::macros::with_perf_log_(
            || $wrapped_fn,
            format!("{DESCR}: {}", format_args!($($msg)+)).as_str(),
        )
    }};
}

#[macro_export]
macro_rules! async_perf_log {
    ($wrapped_fn:expr) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        $crate::macros::with_async_perf_log_(
            async || $wrapped_fn,
            DESCR,
        )
        .await
    }};

    ($wrapped_fn:expr, $($msg:tt)+) => {{
        const DESCR: &str = concat!(module_path!(), ":", line!(), ":", column!());

        $crate::macros::with_async_perf_log_(
            async || $wrapped_fn,
            format!("{DESCR}: {}", format_args!($($msg)+)).as_str(),
        )
        .await
    }};
}
