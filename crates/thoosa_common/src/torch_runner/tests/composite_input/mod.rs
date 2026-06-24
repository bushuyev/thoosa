pub mod loader_a;
pub mod loader_b;
pub mod ab_clc;
pub mod dims;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_ab_clc() {

}