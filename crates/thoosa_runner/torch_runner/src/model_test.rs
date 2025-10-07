#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use std::ops::{Deref, Range};
    use std::sync::Arc;

    use parking_lot::RwLock;
    use tch::IndexOp;
    use tch::{Device, Kind, Tensor, nn, nn::OptimizerConfig};

    use crate::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
    use caching_runner::adjusters::permits::{PermitsAdjustContextImpl, PermitsAdjuster};
    use caching_runner::adjusters::splits::{SplitsAdjuster, SplitsAdjusterContextImpl};
    use ambassador::Delegate;
    use caching_runner::common::ambassador_impl_PermitsCtx;
    use caching_runner::common::cachable::{Cached, DataCacheWrapper};
    use caching_runner::common::runnable::Stage;
    use caching_runner::common::{DropIf, OutId, PermitsCtx, RunCtx, Select};
    use caching_runner::global_ctx::GlobalCtx;
    use caching_runner::oom_handler::{
        Adjustable, Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler,
    };
    use caching_runner::permits::{MAX_PERMITS, NumPermits, PermitStep, RunnerPerm};
    use futures_util::future::BoxFuture;
    use caching_runner::test_utils::parse_numeric_part;

    use crate::data::ThoTensor2;
    use thoosa_shared::traits::MonoMetrics;
    use crate::tests::TorchRunCtx;
    use dashmap::DashMap as DMap;
    use log::debug;
    use rand::seq::SliceRandom;
    use rand::{SeedableRng, rngs::StdRng};
    use std::sync::OnceLock;
    use thoosa_common::applicables::ThoDevice;
    use thoosa_common::dims::{AnyMetrics, AnySub, Dims};
    use thoosa_shared::CopyToDevice;
    use thoosa_shared::applicables_runner::{Input, Output};
    use thoosa_shared::traits::{CachableIdTr, Metrics, Named, SimpleId, WithId};
    use thoosa_shared::{MyError, ThoosaError};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    // Shared registry for mapping FileListInput IDs -> (file, label) entries
    static FILE_LISTS: OnceLock<DMap<String, Vec<(&'static str, f64)>>> = OnceLock::new();
    fn registry() -> &'static DMap<String, Vec<(&'static str, f64)>> {
        FILE_LISTS.get_or_init(|| DMap::new())
    }

    fn init_ctx_once() {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            let _ = GlobalCtx::init_devices(2);
        });
    }

    // ---------------- Run Context ----------------
    #[derive(Debug, Delegate)]
    #[delegate(PermitsCtx, target = "perms")]
    struct CpuRunCtx {
        device_idx: usize,
        cache_area: u64,
        perms: CpuPermsCtx,
    }

    #[derive(Debug)]
    struct CpuPermsCtx {
        acquired: NumPermits,
    }

    impl CpuPermsCtx {
        fn new() -> Self {
            Self { acquired: 0 }
        }
    }

    impl Named for CpuPermsCtx {}

    impl PermitsCtx for CpuPermsCtx {
        fn add_permit(&mut self, permit: RunnerPerm) {
            self.acquired = self.acquired.saturating_add(permit.num_permits());
        }

        fn drop_permits(&mut self, _drop_if: DropIf) {
            self.acquired = 0;
        }

        fn acquired_permits(&self) -> NumPermits {
            self.acquired
        }
    }

    impl CpuRunCtx {
        fn new(device_idx: usize, cache_area: u64) -> Self {
            Self {
                device_idx,
                cache_area,
                perms: CpuPermsCtx::new(),
            }
        }
    }

    impl Adjustable for CpuRunCtx {}

    impl RunCtx for CpuRunCtx {
        fn device_idx(&self) -> usize {
            self.device_idx
        }
        fn purge_gpu_caches(_ctx: Arc<RwLock<Self>>) -> BoxFuture<'static, Result<(), ThoosaError>>
        where
            Self: Sized,
        {
            Box::pin(async { Ok(()) })
        }
        fn cache_area(&self) -> u64 {
            self.cache_area
        }
    }

    impl Named for CpuRunCtx {}

    // ---------------- File Input / A/B Loader ----------------
    #[derive(Debug, Clone, Eq, PartialEq, Hash)]
    struct FileInputId(&'static str);
    impl Named for FileInputId {
        fn name(&self) -> &'static str {
            "FileInputId"
        }
    }

    #[derive(Debug, Clone)]
    struct FileInput {
        id: FileInputId,
    }
    impl Input for FileInput {}
    impl WithId for FileInput {
        type ID = FileInputId;
        fn id(&self) -> &Self::ID {
            &self.id
        }
    }
    impl From<&'static str> for FileInput {
        fn from(s: &'static str) -> Self {
            Self { id: FileInputId(s) }
        }
    }

    #[derive(Debug, Clone, Hash, Eq, PartialEq)]
    struct ABData;
    impl Dims for ABData {
        type Metrics = ABMetrics;

    }
    // impl Adjustable for ThoTensor2<ABData, SimpleId> {}

    impl Select<std::ops::Range<ABMetrics>> for ThoTensor2<ABData, SimpleId> {
        type Output = ThoTensor2<ABData, SimpleId>;

        fn subview(&self, selector: std::ops::Range<ABMetrics>) -> Result<Self::Output, ThoosaError> {
            use std::cmp::min;
            let size = ABMetrics::from(self.t_ref().size());
            let d0 = selector.start.dim_0..min(size.dim_0, selector.end.dim_0);

            Ok(ThoTensor2::new(self.id().clone(), self.t_ref().i((d0, ..))))
        }
    }

    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    struct ABMetrics {
        dim_0: i64,
        dim_1: i64,
    }
    impl Metrics for ABMetrics {}

    impl From<Vec<i64>> for ABMetrics {
        fn from(v: Vec<i64>) -> Self {
            ABMetrics {
                dim_0: *v.get(0).unwrap_or(&0),
                dim_1: *v.get(1).unwrap_or(&0),
            }
        }
    }
    impl std::ops::AddAssign<&ABMetrics> for ABMetrics {
        fn add_assign(&mut self, rhs: &ABMetrics) {
            self.dim_0 += rhs.dim_0;
        }
    }
    impl std::ops::DivAssign<i64> for ABMetrics {
        fn div_assign(&mut self, rhs: i64) {
            if rhs != 0 {
                self.dim_0 /= rhs;
            }
        }
    }
    impl PartialOrd for ABMetrics {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.dim_0.partial_cmp(&other.dim_0)
        }
    }
    impl<'a, 'b> std::ops::Add<&'b ABMetrics> for &'a ABMetrics {
        type Output = ABMetrics;
        fn add(self, rhs: &'b ABMetrics) -> Self::Output {
            ABMetrics {
                dim_0: self.dim_0 + rhs.dim_0,
                dim_1: self.dim_1,
            }
        }
    }

    /* #[derive(Debug)]
    struct ABLoaderOutput {
        id:SimpleId,
        t:ThoTensor2<ABData, SimpleId>
    }

    impl ABLoaderOutput {
        pub fn new(id: SimpleId, t: ThoTensor2<ABData, SimpleId>) -> Self {
            Self { id, t }
        }
    }

    impl WithId for ABLoaderOutput {
        type ID = SimpleId;

        fn id(&self) -> &Self::ID {
            &self.id
        }
    }
    impl Output for ABLoaderOutput {}

    #[async_trait]
    impl caching_runner::common::Data for ABLoaderOutput {
        type Metrics = ABMetrics;
        type Sub = Self;
        type SubSelect = std::ops::Range<Self::Metrics>;
        fn size(&self) -> Self::Metrics {
            ABMetrics::from(self.t.t().size())
        }
        fn subview(&self, select: &Self::SubSelect) -> Self::Sub {
            use std::cmp::min;
            let size = ABMetrics::from(self.t.t().size());
            let d0 = select.start.dim_0..min(size.dim_0, select.end.dim_0);

            ABLoaderOutput::new(self.id.clone(), ThoTensor2::new(self.id.clone(), self.t.t().i((d0, ..))))
        }
        fn track_used_in(&self, token: &'static str) { self.t._track_used_in(token); }

        fn device(&self) -> usize {
            self.device()
        }
    }

    #[async_trait]
    impl CopyToDevice for ABLoaderOutput {
        async fn copy_to_device(&self, dst_device: usize) ->Self {
            ABLoaderOutput::new(self.id.clone(), self.t.copy_to_device(dst_device).await)
        }
    }*/

    #[derive(Debug, cached_derive::Cached)]
    #[cache_name = "ABDataId"]
    struct ABLoader {
        input_len: i64,
        feat: i64,
        kind: Kind,
    }
    impl ABLoader {
        fn new(input_len: i64, feat: i64, kind: Kind) -> Self {
            Self {
                input_len,
                feat,
                kind,
            }
        }
    }

    impl OutId for ABLoader {
        fn out_id(
            &self,
            input_id: &<Self::In<'static> as WithId>::ID,
        ) -> <Self::Out as WithId>::ID {
            SimpleId::from(format!("features_{}", input_id.0))
        }
    }

    #[async_trait]
    impl Stage for ABLoader {
        type In<'a> = FileInput;
        type Out = ThoTensor2<ABData, SimpleId>;
        type Ctx = TorchRunCtx;

        async fn run(
            self: Arc<Self>,
            input: Arc<Self::In<'static>>,
            _ctx: Arc<RwLock<Self::Ctx>>,
        ) -> Result<Self::Out, ThoosaError> {
            // Features derived from file name only (no label knowledge): even index => +1, odd => -1.
            let idx = parse_numeric_part(input.id.0).unwrap_or(0.0) as i64;
            let base = if idx % 2 == 0 { 1.0 } else { -1.0 };
            let x = Tensor::ones([self.input_len, self.feat], (self.kind, Device::Cpu)) * base;
            let id = self.out_id(&input.id);
            Ok(ThoTensor2::new(id, x))
        }
    }

    // ---------------- Batch Builder: builds (features + label) batch ----------------

    /*
    #[derive(Debug)]
    struct ABBatchBuilderOutput {
        id:SimpleId,
        t:ThoTensor2<ABData, SimpleId>
    }

    impl ABBatchBuilderOutput {
        pub fn new(id: SimpleId, t: ThoTensor2<ABData, SimpleId>) -> Self {
            Self { id, t }
        }
    }

    impl Adjustable for ABBatchBuilderOutput{}

    impl WithId for ABBatchBuilderOutput {
        type ID = SimpleId;

        fn id(&self) -> &Self::ID {
            &self.id
        }
    }
    impl Output for ABBatchBuilderOutput {}
    impl Input for ABBatchBuilderOutput {}

    #[async_trait]
    impl caching_runner::common::Data for ABBatchBuilderOutput {
        type Metrics = ABMetrics;
        type Sub = Self;
        type SubSelect = std::ops::Range<Self::Metrics>;
        fn size(&self) -> Self::Metrics {
            ABMetrics::from(self.t.t().size())
        }
        fn subview(&self, select: &Self::SubSelect) -> Self::Sub {
            use std::cmp::min;
            let size = ABMetrics::from(self.t.t().size());
            let d0 = select.start.dim_0..min(size.dim_0, select.end.dim_0);

            ABBatchBuilderOutput::new(self.id.clone(), ThoTensor2::new(self.id.clone(), self.t.t().i((d0, ..))))
        }
        fn track_used_in(&self, token: &'static str) { self.t._track_used_in(token); }

        fn device(&self) -> usize {
            self.device()
        }
    }

    #[async_trait]
    impl CopyToDevice for ABBatchBuilderOutput {
        async fn copy_to_device(&self, dst_device: usize) ->Self {
            ABBatchBuilderOutput::new(self.id.clone(), self.t.copy_to_device(dst_device).await)
        }
    }*/

    #[derive(Debug, cached_derive::Cached)]
    #[cache_name = "ABBatchId"]
    struct BatchBuilder {
        loader: Arc<ABLoader>,
        batch_size: usize,
    }
    impl BatchBuilder {
        fn new(loader: Arc<ABLoader>, batch_size: usize) -> Self {
            Self { loader, batch_size }
        }
    }

    impl OutId for BatchBuilder {
        fn out_id(
            &self,
            input_id: &<Self::In<'static> as WithId>::ID,
        ) -> <Self::Out as WithId>::ID {
            SimpleId::from(format!("batch_{}", input_id))
        }
    }

    #[async_trait]
    impl Stage for BatchBuilder {
        type In<'a> = FileListInput;
        type Out = ThoTensor2<ABData, SimpleId>;
        type Ctx = TorchRunCtx;

        async fn run(
            self: Arc<Self>,
            input: Arc<Self::In<'static>>,
            ctx: Arc<RwLock<Self::Ctx>>,
        ) -> Result<Self::Out, ThoosaError> {
            // Look up the files and labels by input id
            let id_key = format!("{}", input.id);
            let files = registry()
                .get(&id_key)
                .ok_or_else(|| MyError::SomethingElse("file list not found".into()))?;

            // Ensure at least 2 files are present
            if files.is_empty() {
                return Err(MyError::SomethingElse("empty file list".into()).into());
            }
            if files.len() < 2 {
                return Err(MyError::SomethingElse("need at least 2 files".into()).into());
            }

            if self.batch_size == 0 {
                return Err(MyError::SomethingElse("batch_size must be >= 1".into()).into());
            }

            // Deterministic RNG seeded by input id for stable caching
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

            // Load features for all files in the list via cached loader
            let mut label_vec: Vec<f64> = Vec::with_capacity(self.batch_size);
            let mut feats: Vec<Arc<DataCacheWrapper<ThoTensor2<ABData, SimpleId>>>> =
                Vec::with_capacity(self.batch_size);
            for &fi in &chosen {
                let (f, lbl) = files[fi];
                let feat = Arc::clone(&self.loader)
                    .run_cached(
                        Arc::new(<&str as Into<FileInput>>::into(f)),
                        Arc::clone(&ctx),
                    )
                    .await?;
                feats.push(feat);
                label_vec.push(lbl);
            }

            let (device_idx, _cache_area, _num_devices) = {
                let g = ctx.read();
                (
                    g.device_idx(),
                    g.cache_area(),
                    GlobalCtx::global().device_count(),
                )
            };

            // Materialize all features to the target device and build a single stacked batch
            // let feat_refs: Vec<&DataCacheWrapper<ThoTensor2<ABData, SimpleId>>> = feats.iter().map(|a| &**a).collect();
            DataCacheWrapper::with_data_array(&feats, device_idx, None, |data| {
                // data: &[&ThoTensor<ABData>] aligned with label_vec
                let ax_id = SimpleId::from(format!("ax_{}", input.id));
                let first = data
                    .first()
                    .ok_or_else(|| MyError::SomethingElse("no tensors".into()))?;

                // Use first tensor as the callable host; pass the rest as params
                first.run_inner_with_params(
                    ax_id.clone(),
                    &|first_t: &Tensor, params: &[&Tensor]| {
                        let batch_size = first_t.size()[0];
                        let kind = first_t.kind();
                        let dev = first_t.device();

                        // Build [x | y] per file and then stack along batch (dim=0)
                        let mut rows: Vec<Tensor> = Vec::with_capacity(label_vec.len());

                        // First element
                        let y0 = Tensor::full([batch_size, 1], label_vec[0], (kind, dev));
                        rows.push(Tensor::cat(&[first_t.copy(), y0], 1));

                        // Remaining elements from params with corresponding labels
                        for (i, b_t) in params.iter().enumerate() {
                            let y = Tensor::full(
                                [batch_size, 1],
                                label_vec[i + 1],
                                (b_t.kind(), b_t.device()),
                            );
                            rows.push(Tensor::cat(&[(*b_t).copy(), y], 1));
                        }

                        Ok(Tensor::cat(&rows, 0))
                    },
                    &data[1..],
                )
            })
            .await
        }
    }

    // ---------------- Simple Classifier (1 linear layer) ---------------

    /*


    #[derive(Debug)]
    struct ClassifierOutput {
        id:SimpleId,
        t:ThoTensor2<ClsOut, SimpleId>
    }

    impl ClassifierOutput {
        pub fn new(id: SimpleId, t: ThoTensor2<ClsOut, SimpleId>) -> Self {
            Self { id, t }
        }
    }

    impl WithId for ClassifierOutput {
        type ID = SimpleId;

        fn id(&self) -> &Self::ID {
            &self.id
        }
    }
    impl Output for ClassifierOutput {}

    #[async_trait]
    impl caching_runner::common::Data for ClassifierOutput {
        type Metrics = [i64; 2];
        type Sub = Self;
        type SubSelect = std::ops::Range<Self::Metrics>;
        fn size(&self) -> Self::Metrics {
            let sz = self.size();
            [*sz.get(0).unwrap_or(&0), *sz.get(1).unwrap_or(&0)]
        }
        fn track_used_in(&self, _token: &'static str) {}

        fn device(&self) -> usize {
            self.device()
        }
    }

    #[async_trait]
    impl CopyToDevice for ClassifierOutput {
        async fn copy_to_device(&self, dst_device: usize) ->Self {
            ClassifierOutput::new(self.id.clone(), self.t.copy_to_device(dst_device).await)
        }
    }*/

    #[derive(Debug, Clone, Hash, Eq, PartialEq)]
    struct ClsOut;
    impl Dims for ClsOut {
        type Metrics = MonoMetrics<1, 2>;
    }

    impl ClsOut {
        fn accuracy(t: &ThoTensor2<ClsOut, SimpleId>) -> f64 {
            t.double_value(&[0])
        }
    }

    #[derive(Debug, cached_derive::Cached)]
    #[cache_name = "ClsOutId"]
    struct Classifier {
        id: SimpleId,
        in_dim: i64,
        epochs: i64,
        lr: f64,
        fail_until: Vec<caching_runner_test::FailUntil>,
    }
    impl Classifier {
        fn new(
            id: SimpleId,
            in_dim: i64,
            epochs: i64,
            lr: f64,
            fail_until: Vec<caching_runner_test::FailUntil>,
        ) -> Self {
            Self {
                id,
                in_dim,
                epochs,
                lr,
                fail_until,
            }
        }
    }

    impl Adjustable for Classifier {}

    impl OutId for Classifier {
        fn out_id(
            &self,
            input_id: &<Self::In<'static> as WithId>::ID,
        ) -> <Self::Out as WithId>::ID {
            SimpleId::from(format!("cls_{}_inp_{}", self.id, input_id))
        }
    }

    #[async_trait]
    impl Stage for Classifier {
        type In<'a> = ThoTensor2<ABData, SimpleId>;
        type Out = ThoTensor2<ClsOut, SimpleId>;
        type Ctx = TorchRunCtx;

        fn do_run<'a>(
            self: Arc<Self>,
            input: &Self::In<'a>,
            _ctx: Arc<RwLock<Self::Ctx>>,
        ) -> Result<Self::Out, ThoosaError>
        where
            Self: Sized,
        {
            // Split input into features and labels using scoped tensor access
            let feat = self.in_dim;
            let fails = &self.fail_until;
            let ctx_clone = Arc::clone(&_ctx);

            let out_id = self.out_id(input.id());
            Ok(
                      ThoTensor2::new(out_id, input.run_inner(//TODO id
                        |t: &Tensor| {
                            debug!("t.size()={:?}", t.size());
                        // Simulate OOM conditions based on batch size t.size()[0]
                        for fail_until in fails {
                            use caching_runner_test::FailUntil::{DumpedToCpu, MemLimit, MinPermits, Never};
                            match fail_until {
                                MemLimit(m) => {
                                    if t.size()[0] > *m as i64 {
                                        debug!("Err(MyError::OutOfMemory) MemLimit");
                                        return Err(MyError::OutOfMemory.into());
                                    }
                                }
                                MinPermits(min_permits) => {
                                    let acquired_permits = ctx_clone.read().acquired_permits();
                                    if acquired_permits < *min_permits {
                                        debug!("Err(MyError::OutOfMemory) MinPermits: acquired_permits={acquired_permits}");
                                        return Err(MyError::OutOfMemory.into());
                                    }
                                }
                                DumpedToCpu => {
                                    let is_all_on_cpu = GlobalCtx::global().is_all_on_cpu(ctx_clone.read().device_idx());
                                    if !is_all_on_cpu {
                                        debug!("Err(MyError::OutOfMemory) DumpedToCpu");
                                        return Err(MyError::OutOfMemory.into());
                                    }
                                }
                                Never => {}
                            }
                        }
                        let xs = t.narrow(1, 0, feat);           // [batch, feat]
                        let ys = t.narrow(1, feat, 1);           // [batch, 1]


                    // tch-rs linear layer training (CPU)
                        let vs = nn::VarStore::new(t.device());
                        let root = &vs.root();
                        let linear = nn::linear(root, feat, 1, Default::default());
                        let mut opt = nn::Sgd::default().build(&vs, self.lr)?;

                        for _epoch in 0..self.epochs {
                            let logits = xs.apply(&linear);                 // [batch, 1]
                            let loss = logits.binary_cross_entropy_with_logits::<&Tensor>(&ys, None, None, tch::Reduction::Mean);
                            opt.backward_step(&loss);
                        }

                        // Evaluate training accuracy on the same batch
                        let preds = xs.apply(&linear).sigmoid().ge(0.5).to_kind(Kind::Float); // [batch,1]
                        let correct = preds.eq_tensor(&ys).to_kind(Kind::Float).mean(Kind::Float);
                        debug!("Classifier::do_run: correct=\n{correct}");
                        Ok(correct.view([1, 1]))
                    }
                    )?
                ))
        }
    }

    // ---------------- Intermediary Join + Classify task ----------------
    #[derive(Debug)]
    struct FileListInput {
        files: Vec<(&'static str, f64)>,
        id: SimpleId,
    }
    impl FileListInput {
        fn new(files: Vec<(&'static str, f64)>, id: SimpleId) -> Self {
            Self { files, id }
        }
    }
    impl Input for FileListInput {}
    impl WithId for FileListInput {
        type ID = SimpleId;
        fn id(&self) -> &Self::ID {
            &self.id
        }
    }

    #[derive(Debug, cached_derive::Cached)]
    #[cache_name = "ClsOutId"]
    struct JoinAndClassifyTask {
        // Builds a cached batch (features + labels) from files
        batch_builder: Arc<BatchBuilder>,
        clf: Arc<Classifier>,
    }
    impl JoinAndClassifyTask {
        fn new(loader: Arc<ABLoader>, clf: Arc<Classifier>, batch_size: usize) -> Self {
            Self {
                batch_builder: Arc::new(BatchBuilder::new(loader, batch_size)),
                clf,
            }
        }
    }

    impl OutId for JoinAndClassifyTask {
        fn out_id(
            &self,
            input_id: &<Self::In<'static> as WithId>::ID,
        ) -> <Self::Out as WithId>::ID {
            SimpleId::from(format!("cls_{}_inp_{}", self.clf.id, input_id))
        }
    }

    #[async_trait]
    impl Stage for JoinAndClassifyTask {
        type In<'a> = FileListInput;
        type Out = ThoTensor2<ClsOut, SimpleId>;
        type Ctx = TorchRunCtx;

        async fn run(
            self: Arc<Self>,
            input: Arc<Self::In<'static>>,
            ctx: Arc<RwLock<Self::Ctx>>,
        ) -> Result<Self::Out, ThoosaError> {
            // Orchestrate: build cached batch and run classifier on it.
            let batched = Arc::clone(&self.batch_builder)
                .run_cached(Arc::clone(&input), Arc::clone(&ctx))
                .await?;

            let (device_idx, ctx_name) = {
                let guard = ctx.read();
                (guard.device_idx(), guard.name())
            };

            // Build out_key for fairness and acquisitions
            let out_id = self.clf.out_id(batched.id());
            let out_key = Arc::new(out_id.clone()) as Arc<dyn CachableIdTr>;

            // Acquire permits as needed using OOMHandler with Permits and Splits adjusters
            let size = *batched.size();
            let min_split_len = ABMetrics {
                dim_0: 32,
                dim_1: 0,
            };
            OOMHandler::new(
                vec![
                    Adjusters::RunContext(Box::new(PermitsAdjuster::new(
                        MAX_PERMITS,
                        *PermitStep::TEN,
                        move |n| GlobalCtx::global().acquire_many(device_idx, n, out_key.clone()),
                    ))),
                    Adjusters::Input(Box::new(
                        SplitsAdjuster::<ThoTensor2<ABData, SimpleId>, Range<ABMetrics>>::new(min_split_len),
                    )),
                ],
                None,
            )
            .run(
                &self.clf,
                batched,
                Arc::clone(&ctx),
                Contexts::new(
                    SplitsAdjusterContextImpl::new(
                        size,
                        Box::new(SingleDimTensorOutputBuilder::<ClsOut>::new(
                            out_id,
                            1,
                            &[1, 1],
                            (Kind::Float, ThoDevice::from(device_idx).deref().clone()),
                            BuildAs::Sum,
                        )),
                    ),
                    NoopRunnableContext {},
                    PermitsAdjustContextImpl::new(ctx_name),
                ),
            )
            .await
        }
    }

    // ---------------- Test ----------------

    #[tokio::test]
    async fn test_ab_loader_and_classifier_cpu() -> Result<(), ThoosaError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        // CPU-only global ctx
        init_ctx_once();
        let ctx = Arc::new(RwLock::new(TorchRunCtx::new(1, 0, "test")));

        let loader = Arc::new(ABLoader::new(64, 4, Kind::Float));
        let clf = Arc::new(Classifier::new(
            SimpleId::from("lin_cls"),
            4,
            200,
            0.1,
            vec![caching_runner_test::FailUntil::Never],
        ));

        let join_and_cls = Arc::new(JoinAndClassifyTask::new(
            Arc::clone(&loader),
            Arc::clone(&clf),
            2,
        ));
        // Define file list and labels in the test (top-level knows labels)
        // Register file list (top-level owns labels) in registry keyed by input id
        let files = vec![("file_0", 1.0), ("file_1", 0.0)];
        registry().insert("files_0_1".to_string(), files.clone());
        let input = FileListInput::new(files, SimpleId::from("files_0_1"));
        let acc = Arc::clone(&join_and_cls)
            .run_cached(Arc::new(input), Arc::clone(&ctx))
            .await?;

        debug!("acc={acc:?}");

        acc.with_data_ref(0, None, |acc_t| {
            debug!("acc_t={acc_t:?} t=\n{}", acc_t.t_ref());

            // Dim impls are to define domain-specific operations on tensors
            let v = ClsOut::accuracy(acc_t);
            assert!(v > 0.9, "training accuracy too low: {v}");
            Ok(())
        })
        .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_ab_loader_and_classifier_cpu_many_files() -> Result<(), ThoosaError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        // CPU-only global ctx
        init_ctx_once();
        let ctx = Arc::new(RwLock::new(TorchRunCtx::new(0, 0, "test")));

        let loader = Arc::new(ABLoader::new(64, 4, Kind::Float));
        let clf = Arc::new(Classifier::new(
            SimpleId::from("lin_cls_many"),
            4,
            100,
            0.1,
            vec![caching_runner_test::FailUntil::Never],
        ));

        let join_and_cls = Arc::new(JoinAndClassifyTask::new(
            Arc::clone(&loader),
            Arc::clone(&clf),
            4,
        ));

        // 4 files: 2 positives (even ids) and 2 negatives (odd ids)
        let files = vec![
            ("file_0", 1.0),
            ("file_1", 0.0),
            ("file_2", 1.0),
            ("file_3", 0.0),
        ];
        registry().insert("files_0_1_2_3".to_string(), files.clone());
        let input = FileListInput::new(files, SimpleId::from("files_0_1_2_3"));

        let acc = Arc::clone(&join_and_cls)
            .run_cached(Arc::new(input), Arc::clone(&ctx))
            .await?;

        acc.with_data_ref(1, None, |acc| {
            let v = ClsOut::accuracy(acc);
            assert!(v > 0.9, "training accuracy too low (4 files): {v}");
            Ok(())
        })
        .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_classifier_oom_permits_escalates() -> Result<(), ThoosaError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cpu.so").expect("no libtorch_cpu.so");
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        init_ctx_once();
        let ctx = Arc::new(RwLock::new(TorchRunCtx::new(1, 0, "test")));

        let loader = Arc::new(ABLoader::new(64, 4, Kind::Float));
        // Require MAX_PERMITS to proceed and also enforce MemLimit until splits reduce size
        let clf = Arc::new(Classifier::new(
            SimpleId::from("lin_cls_oom"),
            4,
            50,
            0.1,
            vec![
                caching_runner_test::FailUntil::MinPermits(MAX_PERMITS),
                caching_runner_test::FailUntil::MemLimit(128),
            ],
        ));

        let join_and_cls = Arc::new(JoinAndClassifyTask::new(
            Arc::clone(&loader),
            Arc::clone(&clf),
            16,
        ));

        // Build 16 files (even -> 1.0, odd -> 0.0)
        let files: Vec<(&'static str, f64)> = (0..16)
            .map(|i| {
                (
                    Box::leak(format!("file_{}", i).into_boxed_str()) as &'static str,
                    if i % 2 == 0 { 1.0 } else { 0.0 },
                )
            })
            .collect();
        registry().insert("files_oom_permits".to_string(), files.clone());
        let input = FileListInput::new(files, SimpleId::from("files_oom_permits"));

        let acc = Arc::clone(&join_and_cls)
            .run_cached(Arc::new(input), Arc::clone(&ctx))
            .await?;

        acc.with_data_ref(1, None, |acc| {
            debug!("acc={}", acc.t_ref());

            let v = ClsOut::accuracy(acc);
            assert!(v >= 0.5, "should train after permits escalation: {v}");
            Ok(())
        })
        .await?;

        Ok(())
    }
}
