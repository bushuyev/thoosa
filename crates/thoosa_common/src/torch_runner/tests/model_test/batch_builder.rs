use std::sync::Arc;

use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::StdRng};
use tch::Tensor;
use thoosa_named_derive::Named;
use thoosa_shared::ThoError;
use thoosa_shared::traits::{AsyncStage, OutId, RunCtx, Stage, WithId};

use crate as thoosa_common;
use crate::runner::common::cachable::{Cached, DevicesWrap, SelectRef};
use crate::runner::run_ctx::RunCtxImpl;
use crate::tho_tensor::ThoTensor;

use super::common::{ABData, FileListInput, ModelTestId};
use super::loader::ABLoader;

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

impl OutId for BatchBuilder {
    fn out_id(&self, input_id: &<Self::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        ModelTestId(format!("batch_{}", input_id.0).leak())
    }
}

impl Stage for BatchBuilder {
    type In<'a> = FileListInput;
    type Out = Arc<DevicesWrap<ThoTensor<ABData, ModelTestId>>>;
    type Ctx = RunCtxImpl;
}

impl AsyncStage for BatchBuilder {
    async fn run(self: Arc<Self>, input: Self::In<'static>, mut ctx: Self::Ctx) -> Result<Self::Out, ThoError> {
        let (id_key, files) = (input.id.0.to_string(), input.files.clone());

        if files.is_empty() {
            return Err(ThoError::new("empty file list"));
        }
        if files.len() < 2 {
            return Err(ThoError::new("need at least 2 files"));
        }
        if self.batch_size == 0 {
            return Err(ThoError::new("batch_size must be >= 1"));
        }

        let seed: u64 = id_key.as_bytes().iter().fold(0u64, |acc, b| acc.wrapping_mul(131).wrapping_add(*b as u64));
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
            let feat = Arc::clone(&self.loader).run_cached(f.into(), ctx.branch(false)).await?;
            feats.push(feat);
            label_vec.push(lbl);
        }

        let mut rows: Vec<Tensor> = Vec::with_capacity(label_vec.len());
        for (idx, feat) in feats.iter().enumerate() {
            let label = label_vec[idx];
            let b_t = feat.select_ref(0, None).await?;

            let t: &Tensor = &*b_t.as_ref().local()?;
            let y = Tensor::full([t.size()[0], 1], label, (t.kind(), t.device()));

            rows.push(Tensor::cat(&[t.copy(), y], 1));
        }
        let output = Tensor::cat(&rows, 0);

        let output_id = self.out_id(input.id());
        Ok(Arc::new(DevicesWrap::wrap(ThoTensor::new(output_id, output)?)))
    }
}
