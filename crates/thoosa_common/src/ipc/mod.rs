
/// keeps stuff needed to run cuda operations in a separate process.
/// child process is started from CudaIPCActorHandle::new->CudaIPCActor::new->CudaIPCActor->spawn_worker_and_loops->spawn_worker_for_cuda_device
pub mod ipc_cuda_utils;
pub mod cuda_process;
pub mod cuda_wokrer;
pub mod codec;
pub mod tests;
pub mod json_ser;
