use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::RwLock;
use tch::Device::Cpu;
use tch::kind::FLOAT_CPU;
use tch::Tensor;
use cached_derive::Cached;
use caching_runner::common::{Data, OutId, RunCtx};
use caching_runner::common::runnable::Stage;
use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{DataId, WithId};
use thoosa_store::Store;
use torch_runner::data::ThoTensor2;
use crate::tasks::cross_dims::MotifSlices;
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::{RawTensorLoader, TensorLoader};

#[derive(Debug, Cached)]
pub struct MotifLoader<SRC>
where
    SRC: TensorLoader + 'static
{
    store: Arc<RwLock<Box<dyn Store>>>,
    loader:Arc<SRC>
}

impl <SRC> MotifLoader<SRC>
    where SRC: TensorLoader + 'static
{
    pub fn new(
        store: Arc<RwLock<Box<dyn Store>>>,
        loader:Arc<SRC>
    ) -> Self {
        Self {
            store,
            loader
        }
    }
}

impl <SRC> OutId for MotifLoader<SRC>
where SRC: TensorLoader + 'static
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

#[async_trait]
impl <SRC> Stage for MotifLoader <SRC>
    where SRC: TensorLoader + 'static
{
    type In<'a> = DataId;
    type Out = ThoTensor2<MotifSlices, DataId>;

    type Ctx = RunCtx2;

    async fn run(self: Arc<Self>, input: Arc<Self::In<'static>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    where
            for<'a> Self::In<'a>: Send + Sync
    {
        let (channel, run_id, slices, resources_map) = self.store.read().list_motif_slices(**input, None, None)?;

        let find_motifs_run_config = FindMotifsRunConfig::from_properties_map__(
            run_id,
            &self.store.read().list_run_config_values(run_id)?,
        )?;

        let tensors_ids_slices = slices
            .iter()
            .fold(HashMap::new(), |mut a, slice| {
            a.entry(slice.0.clone()) //TODO clone
                .or_insert_with(|| vec![])
                .push(slice.1);
            a
        });

        for (resource_id, slices) in tensors_ids_slices {
            let t = Arc::clone(&self.loader).load(Arc::new(resources_map.get(&resource_id).expect(&format!("no resource with id={resource_id}")).clone()), Arc::clone(&ctx)).await?;
            //
            // t.with_data(ctx.read().device_idx(), None, None, |t|{
            //     Ok(())
            // }).await?;
            // t.da().i((channel, slice_idx as i64..(slice_idx + window )as i64))
        }

        Ok(ThoTensor2::new(DataId(0), Tensor::rand(&[10, 100], FLOAT_CPU)))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use parking_lot::RwLock;
    use caching_runner::common::cachable::Cached;
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::ThoosaError;
    use thoosa_shared::traits::DataId;
    use thoosa_store::{MockStore, MockStoreBuilder, Store};
    use crate::tasks::motif_loader::MotifLoader;
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::{mock_tensor_loader, RawTensorLoader};
    use crate::tasks::tensor_loader::mock_tensor_loader::make_wrapper;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_0()-> Result<(), ThoosaError> {
        GlobalCtx::init_devices(2);

        let run_id = 1;

        let mut store = MockStoreBuilder::new()
            .with_list_motif_slices(10u32, None, None, Ok((3i64, run_id, vec![], HashMap::new())))
            .with_list_run_config_values(
                run_id,
                Ok(FindMotifsRunConfig::new(Some(run_id), 100, 1000, None, None).to_properties_iter(None).collect())
            )
            .build();

        let motif_loader = Arc::new(MotifLoader::new(
            Arc::new(RwLock::new(store)),

            Arc::new(
                mock_tensor_loader::new()
                .with_data([
                    (1, make_wrapper("tensor_a", &[1.0, 2.0])),
                    (2, make_wrapper("tensor_b", &[3.0, 4.0])),
                ])
                .build()
            )
        ));

        let ctx = Arc::new(RwLock::new(RunCtx2::new(0, 0, "test")));

        motif_loader.run_cached(Arc::new(DataId::from(10)), ctx).await?;


        Ok(())
    }

}