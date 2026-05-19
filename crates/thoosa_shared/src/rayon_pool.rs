
use std::sync::OnceLock;

static RAYON: OnceLock<rayon::ThreadPool> = OnceLock::new();

pub fn rayon_pool() -> &'static rayon::ThreadPool {
    RAYON.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(20) // tune!
            .thread_name(|i| format!("rayon-cpu-{i}"))
            .build()
            .expect("no rayon_pool")
    })
}