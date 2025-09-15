// use std::sync::Arc;
// use parking_lot::RwLock;
// use thoosa_shared::applicables_runner::{Runner, ToOrdinal};
// use thoosa_shared::ThoosaError;
//
// #[derive(Debug, cached_derive::Cached)]
// #[cache_name = "LoadedTensorId"]
// pub struct TensorLoader{
//
// }
//
// impl caching_runner::common::Runnable for TensorLoader {
//
// }
//
// impl TensorLoader {
//     pub fn new() -> Self {
//         Self {}
//     }
// }
// #[cfg(test)]
// mod tests {
//     use crate::tasks::tensor_loader::TensorLoader;
//
//     #[test]
//     fn test_load(){
//         let loader = TensorLoader::new();
//
//     }
//
// }