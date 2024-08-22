use std::io::Write;
use std::str::FromStr;
use clap::{Args, Parser, Subcommand};
use log::debug;
use polars::export::rayon::iter::IntoParallelIterator;
use polars::export::rayon::iter::ParallelIterator;
use tch::{Device, Tensor};
use tch::nn::Module;

use thoosa_common::ResourceLoader;
use thoosa_common::utils::load_tensor;
use thoosa_shared::*;
use thoosa_shared::ThoosaError;
use thoosa_store::Store;

use cli::Commands::*;
use crate::data::parquet_to_tensor_by_path;
pub mod data;


pub mod version;
pub mod cli;


pub struct ParquetLoader {
    base_path:String,
    tensors_path:Option<String>,
    dsp_filters: Option<Box<dyn Fn(Tensor) -> Tensor + Send + Sync>>,
    resource_to_path:Box<dyn Fn(&str, &str)->String  + Send + Sync>
}

impl ParquetLoader {

    pub fn new(
        base_path: String,
        tensors_path: Option<String>,
        dsp_filters: Option<Box<dyn Fn(Tensor)->Tensor +Send + Sync>>,
        resource_to_path:Box<dyn Fn(&str, &str)->String  + Send + Sync>
    ) -> Self {
        Self {
            base_path, tensors_path, dsp_filters, resource_to_path
        }
    }

}

impl ResourceLoader for ParquetLoader {
    type K = String;
    type V = Tensor;

    fn load(&self, key: &String, device:Device) -> Box<Tensor> {

        let t = if let Some(tensors_path) = self.tensors_path.as_ref() {
            load_tensor(tensors_path.as_str(), key.as_str(), device)

        } else {
            parquet_to_tensor_by_path(
                (self.resource_to_path)(self.base_path.as_str(), key.as_str()),
                device
            )

        }.map(|t|{
            match self.dsp_filters.as_ref() {
                None => t,
                Some(f) => f(t)
            }

        }).expect(format!("can't load {}", key).as_str());

        Box::new(t)
    }
}


