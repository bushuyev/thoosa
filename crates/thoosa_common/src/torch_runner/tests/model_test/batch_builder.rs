use std::sync::Arc;

use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use tch::Tensor;
use thoosa_named_derive::Named;
use thoosa_shared::traits::WithId;
use thoosa_shared::{Cause, ThoError};

use crate as thoosa_common;
use crate::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use crate::runner::common::runnable::{AsyncStage, Stage};
use crate::runner::common::{OutId, RunCtx};
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_tensor::ThoTensor;

use super::common::{ABData, FileListInput, ModelTestId};
use super::loader::{new_file_input_cache, ABLoader};

#[derive(Debug, cached_derive::Cached, Named)]
#[cache_name = "ABBatchId"]
pub(crate) struct BatchBuilder {
    loader: Arc<ABLoader>,
    batch_size: usize,
}

impl BatchBuilder {
    pub(crate) fn new(loader: Arc<ABLoader>, batch_size: usize) -> Self {
        Self { loader, batch_size }
    }
}

pub(crate) fn new_file_list_input_cache(input: FileListInput) -> Arc<FileListInput> {
    Arc::new(input)
}

impl OutId for BatchBuilder {
    fn out_id(
        &self,
        input_id: &<Self::In<'static> as WithId>::ID,
    ) -> <Self::Out as WithId>::ID {
        ModelTestId(format!("batch_{}", input_id.0).leak())
    }
}

impl Stage for BatchBuilder {
    type In<'a> = FileListInput;
    type Out = ThoTensor<ABData, ModelTestId>;
    type Ctx = RunCtxImpl;
}

impl AsyncStage for BatchBuilder {
    type RunInput = Arc<Self::In<'static>>;

    fn run_input_id(input: &Self::RunInput) -> &<<Self as Stage>::In<'static> as thoosa_shared::traits::WithId>::ID {
        thoosa_shared::traits::WithId::id(input)
    }
    async fn run(
        self: Arc<Self>,
        input: Arc<Self::In<'static>>,
        mut ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let (id_key, files) = (input.id.0.to_string(), input.files.clone());

        if files.is_empty() {
            return Err(Cause::SomethingElse("empty file list".into()).into());
        }
        if files.len() < 2 {
            return Err(Cause::SomethingElse("need at least 2 files".into()).into());
        }
        if self.batch_size == 0 {
            return Err(Cause::SomethingElse("batch_size must be >= 1".into()).into());
        }

        let seed: u64 = id_key
            .as_bytes()
            .iter()
            .fold(0u64, |acc, b| acc.wrapping_mul(131).wrapping_add(*b as u64));
        let mut rng = StdRng::seed_from_u64(seed);
        let n_files = files.len();
        let chosen: Vec<usize> = if self.batch_size <= n_files {
            let mut idxs = (0..n_files).collect::<Vec<_>>();
            idxs.shuffle(&mut rng);
            idxs.truncate(self.batch_size);
            idxs
        } else {
            let mut idxs = (0..n_files).collect::<Vec<_>>();
            idxs.shuffle(&mut rng);
            let mut v = Vec::with_capacity(self.batch_size);
            while v.len() < self.batch_size {
                v.extend(idxs.iter().cloned());
            }
            v.truncate(self.batch_size);
            v
        };

        let mut label_vec: Vec<f64> = Vec::with_capacity(self.batch_size);
        let mut feats: Vec<Arc<DevicesWrap<ThoTensor<ABData, ModelTestId>>>> = Vec::with_capacity(self.batch_size);
        for &fi in &chosen {
            let (f, lbl) = files[fi];
            let feat = Arc::clone(&self.loader)
                .run_cached(new_file_input_cache(f), ctx.branch(false))
                .await?;
            feats.push(feat);
            label_vec.push(lbl);
        }

        let mut rows: Vec<Tensor> = Vec::with_capacity(label_vec.len());
        for (idx, feat) in feats.iter().enumerate() {
            let label = label_vec[idx];
            let row = feat.with_data_ref(0, ctx.branch(false), None, move |b_t: &ThoTensor<ABData, ModelTestId>, _ctx| {
                let t: &Tensor = b_t.as_ref();
                let y = Tensor::full([t.size()[0], 1], label, (t.kind(), t.device()));
                Ok(Tensor::cat(&[t.copy(), y], 1))
            }).await?;
            rows.push(row);
        }
        let output = Tensor::cat(&rows, 0);

        let output_id = self.out_id(input.id());
        Ok(Arc::new(DevicesWrap::wrap(ThoTensor::new(output_id, output)?)))
    }
}
