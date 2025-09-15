// use std::sync::Arc;
// use parking_lot::RwLock;
// use caching_runner::common::runnable::Stage;
// use thoosa_shared::applicables_runner::{Runner, ToOrdinal};
// use thoosa_shared::ThoosaError;
//
// #[derive(Debug, cached_derive::Cached)]
// #[cache_name = "LoadedTensorId"]
// pub struct TensorLoader{
//
// }
//
// impl Stage for TensorLoader {
//     type In = ();
//     type Out = ();
//     type Ctx = ();
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