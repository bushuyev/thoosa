#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/ipc.md"))]
pub mod cuda_process;
pub mod cuda_worker;
/// keeps stuff needed to run cuda operations in a separate process.
/// child process is started from CudaIPCActorHandle::new->CudaIPCActor::new->CudaIPCActor->spawn_worker_and_loops->spawn_worker_for_cuda_device
pub mod ipc_cuda_utils;
pub mod serialization;
pub mod tests;
pub mod wraps;
// Proxy fixtures and their typetag registrations belong to test binaries only.
pub mod endpoints;
mod pending;
#[cfg(test)]
mod perf_test;
pub mod proxy;
#[cfg(test)]
mod proxy_tests;
#[cfg(test)]
mod proxy_tests_with_state;
#[cfg(test)]
mod recovery_tests;
pub mod worker_msg;

pub mod tensor_info;

pub mod state_transfer;
