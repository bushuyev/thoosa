#![allow(dead_code)]

#[cfg(test)]
#[allow(unused_variables)]
mod tests {
    use async_trait::async_trait;
    use std::ops::{Deref, Range};
    use std::sync::Arc;

    use parking_lot::RwLock;
    use tch::IndexOp;
    use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

    use crate::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
    use crate::runner::common::runnable::{AsyncStage, Stage, SyncStage};
    use crate::runner::common::{MaxParallelCtx, OutId, RunCtx};
    use crate as thoosa_common;
    #[double]
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use crate::runner::oom_handler::{
        Adjusters, Contexts, NoopRunnableContext, OOMHandler,
    };
    use crate::tho_tensor::ThoTensor;
    use thoosa_shared::traits::{Adjustable, Data, MonoMetrics, Select};
    use thoosa_named_derive::Named;
    use dashmap::DashMap as DMap;
    use log::debug;
    use rand::seq::SliceRandom;
    use rand::{rngs::StdRng, SeedableRng};
    use std::sync::OnceLock;
    use mockall_double::double;
    use crate::tho_device::ThoDevice;
    use crate::dims::Dims;
    use thoosa_shared::CopyToDevice;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::traits::{CachableIdTr, Metrics, WithId};
    use thoosa_shared::{Cause, ThoError};
    use thoosa_shared::copy_mode::CpuOnly;
    use crate::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelStep, MAX_PARALLEL};
    use crate::runner::adjusters::splits::{SplitsAdjuster, SplitsAdjusterContextImpl};
    use crate::runner::test_utils::parse_numeric_part;
    use thoosa_with_data::with_data;
    use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
    use crate::runner::run_ctx_2::RunCtxImpl;

    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::runner::test::FailUntil;

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

    // ---------------- File Input / A/B Loader ----------------
    #[derive(Debug, Clone, Eq, PartialEq, Hash, Named)]
    struct FileInputId(&'static str);

    #[derive(Debug, Clone, Eq, PartialEq, Hash, Named)]
    struct ModelTestId(&'static str);

    #[derive(Debug, Clone)]
    struct FileInput {
        id: FileInputId,
    }
    impl Input for FileInput {  }
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

    #[async_trait]
    impl CopyToDevice for FileInput {
        async fn copy_to_device(&self, _dst_device: usize) -> Result<Self, ThoError> {
            Ok(self.clone())
        }
    }

    #[async_trait]
    impl Data for FileInput {
        type Metrics = ();
        type M = CpuOnly<Self>;

        fn size(&self) -> Self::Metrics {
            ()
        }

        fn device(&self) -> usize {
            0
        }

        fn mem_size(&self) -> u64 {
            0
        }
    }

    fn new_file_input_cache(
        value: impl Into<FileInput>,
        device_idx: usize,
    ) -> Arc<DevicesWrap<FileInput>> {
        Arc::new(DevicesWrap::wrap(value.into()))
    }

    #[derive(Debug, Clone, Hash, Eq, PartialEq)]
    struct ABData;
    impl Dims for ABData {
        type Metrics = ABMetrics;

    }

    #[derive(Debug, Clone, Default)]
    struct ABRange {
        range:std::ops::Range<ABMetrics>
    }

    impl Adjustable for ABRange {}

    impl Select<ABRange> for ThoTensor<ABData, ModelTestId> {
        type Output = ThoTensor<ABData, ModelTestId>;

        fn subview(&self, selector: ABRange) -> Result<Self::Output, ThoError> {
            use std::cmp::min;
            let size = ABMetrics::from(self.t_ref().size());
            let d0 = selector.range.start.dim_0..min(size.dim_0, selector.range.end.dim_0);

            ThoTensor::new(self.id().clone(), self.t_ref().i((d0, ..)))
        }
    }

    impl From<Range<ABMetrics>> for ABRange {
        fn from(value: Range<ABMetrics>) -> Self {
            ABRange {
                range:value
            }
        }
    }

    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    struct ABMetrics {
        dim_0: i64,
        dim_1: i64,
    }
    impl Metrics for ABMetrics {
        fn cap(&self, other: &mut Self) {}
    }

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
    impl std::ops::AddAssign<ABMetrics> for ABMetrics {
        fn add_assign(&mut self, rhs: ABMetrics) {
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
    impl thoosa_shared::runner::common::Data for ABLoaderOutput {
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

    #[derive(Debug, cached_derive::Cached, Named)]
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
            ModelTestId(format!("features_{}", input_id.0).leak())
        }
    }

    impl Stage for ABLoader {
        type In<'a> = FileInput;
        type Out = ThoTensor<ABData, ModelTestId>;
        type Ctx = RunCtxImpl;
    }

    // #[async_trait]
    impl AsyncStage for ABLoader {
        async fn run(
            self: Arc<Self>,
            input: Arc<DevicesWrap<Self::In<'static>>>,
            ctx: Self::Ctx,
        ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
            // Features derived from file name only (no label knowledge): even index => +1, odd => -1.
            let idx = parse_numeric_part(WithId::id(input.as_ref()).0).unwrap_or(0.0) as i64;
            let base = if idx % 2 == 0 { 1.0 } else { -1.0 };
            let x = Tensor::ones([self.input_len, self.feat], (self.kind, Device::Cpu)) * base;
            let id = self.out_id(WithId::id(input.as_ref()));

            let device_idx = ctx.tho_device_idx();
            let num_devices = GlobalCtx::global().device_count();

            Ok(Arc::new(DevicesWrap::wrap(ThoTensor::new(id, x)?)))
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
    impl thoosa_shared::runner::common::Data for ABBatchBuilderOutput {
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

    #[derive(Debug, cached_derive::Cached, Named)]
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
            ModelTestId(format!("batch_{}", input_id.0).leak())
        }
    }

    impl Stage for BatchBuilder {
        type In<'a> = FileListInput;
        type Out = ThoTensor<ABData, ModelTestId>;
        type Ctx = RunCtxImpl;
    }

    // #[async_trait]
    impl AsyncStage for BatchBuilder {
        async fn run(
            self: Arc<Self>,
            input: Arc<DevicesWrap<Self::In<'static>>>,
            mut ctx: Self::Ctx,
        ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
            let (id_key, files) = {
                let input_ref = input.load_from_any_device_(0).await?;
                (input_ref.id.0.to_string(), input_ref.files.clone())
            };

            // Ensure at least 2 files are present
            if files.is_empty() {
                return Err(Cause::SomethingElse("empty file list".into()).into());
            }
            if files.len() < 2 {
                return Err(Cause::SomethingElse("need at least 2 files".into()).into());
            }

            if self.batch_size == 0 {
                return Err(Cause::SomethingElse("batch_size must be >= 1".into()).into());
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
            let mut feats: Vec<Arc<DevicesWrap<ThoTensor<ABData, ModelTestId>>>> =
                Vec::with_capacity(self.batch_size);

            let loader_device_idx = ctx.tho_device_idx();

            for &fi in &chosen {
                let (f, lbl) = files[fi];
                let feat = Arc::clone(&self.loader)
                    .run_cached(
                        new_file_input_cache(f, loader_device_idx),
                        ctx.branch(false),
                    )
                    .await?;
                feats.push(feat);
                label_vec.push(lbl);
            }

            let device_idx = ctx.tho_device_idx();
            // #[cfg(any(test, feature = "mock_global_ctx"))]
            // let _mock_ctx = MockGlobalCtxBuilder::install_global_ctx_mock();
            let num_devices = GlobalCtx::global().device_count();

            // Materialize all features to the target device and build a single stacked batch
            // let feat_refs: Vec<&DataCacheWrapper<ThoTensor2<ABData, SimpleId>>> = feats.iter().map(|a| &**a).collect();

            let ctx_for_data = ctx.branch(false);
            let output = with_data!(ctx_for_data, [feats];|_ctx, data:&[&ThoTensor<ABData, ModelTestId>]|{

                let first = data
                    .first()
                    .ok_or_else(|| Cause::SomethingElse("no tensors".into()))?;

                let first_t: &Tensor = first.as_ref();
                let params:&[&ThoTensor<ABData, ModelTestId>] = &data[1..];

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
                            (b_t.as_ref().kind(), b_t.as_ref().device()),
                        );
                        rows.push(Tensor::cat(&[(*b_t.as_ref()).copy(), y], 1));
                    }

                    Ok(Tensor::cat(&rows, 0))

            })?;

            let output_id = self.out_id(input.id());
            Ok(Arc::new(DevicesWrap::wrap(ThoTensor::new(output_id, output)?)))
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
    impl thoosa_shared::runner::common::Data for ClassifierOutput {
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
        fn accuracy(t: &ThoTensor<ClsOut, ModelTestId>) -> f64 {
            t.double_value(&[0])
        }
    }

    #[derive(Debug, cached_derive::Cached, Named)]
    #[cache_name = "ClsOutId"]
    struct Classifier {
        id: ModelTestId,
        in_dim: i64,
        epochs: i64,
        lr: f64,
        fail_until: Vec<crate::runner::test::FailUntil>,
    }
    impl Classifier {
        fn new(
            id: ModelTestId,
            in_dim: i64,
            epochs: i64,
            lr: f64,
            fail_until: Vec<crate::runner::test::FailUntil>,
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
            ModelTestId(format!("cls_{}_inp_{}", self.id.0, input_id.0).leak())
        }
    }

    impl Stage for Classifier {
        type In<'a> = ThoTensor<ABData, ModelTestId>;
        type Out = ThoTensor<ClsOut, ModelTestId>;
        type Ctx = RunCtxImpl;
    }

    impl SyncStage for Classifier {
        fn do_run<'a>(
            self: Arc<Self>,
            input: &Self::In<'a>,
            ctx: &Self::Ctx,
        ) -> Result<Self::Out, ThoError>
        where
            Self: Sized,
        {
            // Split input into features and labels using scoped tensor access
            let feat = self.in_dim;
            let fails = &self.fail_until;
            let out_id = self.out_id(input.id());
            Ok(ThoTensor::new(out_id, input.run_inner(|t: &Tensor| {
                        debug!("t.size()={:?}", t.size());
                    // Simulate OOM conditions based on batch size t.size()[0]
                    for fail_until in fails {
                        use crate::runner::test::FailUntil::{DumpedToCpu, MemLimit, NaxParallel, Never};
                        match fail_until {
                            MemLimit(m) => {
                                if t.size()[0] > *m as i64 {
                                    debug!("Err(MyError::OutOfMemory) MemLimit");
                                    return Err(ThoError::oom());
                                }
                            }
                            FailUntil::NaxParallel(max_other_parallel_threshold) => {
                                let max_other_parallel = ctx.max_other_parallel();
                                debug!("MulTask::do_run: ctx={ctx:?} max_other_parallel={max_other_parallel} max_other_parallel_threshold={max_other_parallel_threshold}");

                                if max_other_parallel > *max_other_parallel_threshold {
                                    return Err(ThoError::oom());
                                }
                            }
                            DumpedToCpu => {
                                let is_all_on_cpu = GlobalCtx::global().is_all_on_cpu(ctx.tho_device_idx());
                                if !is_all_on_cpu {
                                    debug!("Err(MyError::OutOfMemory) DumpedToCpu");
                                    return Err(ThoError::oom());
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
                })?
            )?)
        }
    }

    // #[async_trait]
    impl AsyncStage for Classifier {
        async fn run(
            self: Arc<Self>,
            input: Arc<DevicesWrap<Self::In<'static>>>,
            ctx: Self::Ctx,
        ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
            let device_idx = ctx.tho_device_idx();
            let num_devices = GlobalCtx::global().device_count();

            let stage = Arc::clone(&self);
            let ctx = ctx;
            let output = input
                .with_data_ref(device_idx, ctx, None, move |tensor: &Self::In<'_>, run_ctx| {
                    Arc::clone(&stage).do_run(tensor, run_ctx)
                })
                .await?;

            Ok(Arc::new(DevicesWrap::wrap(output)))
        }
    }

    // ---------------- Intermediary Join + Classify task ----------------
    #[derive(Debug)]
    struct FileListInput {
        files: Vec<(&'static str, f64)>,
        id: ModelTestId,
    }
    impl FileListInput {
        fn new(files: Vec<(&'static str, f64)>, id: ModelTestId) -> Self {
            Self { files, id }
        }
    }
    impl Input for FileListInput {  }
    impl WithId for FileListInput {
        type ID = ModelTestId;
        fn id(&self) -> &Self::ID {
            &self.id
        }
    }

    #[async_trait]
    impl CopyToDevice for FileListInput {
        async fn copy_to_device(&self, _dst_device: usize) -> Result<Self, ThoError> {
            Ok(Self {
                files: self.files.clone(),
                id: self.id.clone(),
            })
        }
    }

    #[async_trait]
    impl Data for FileListInput {
        type Metrics = ();
        type M = CpuOnly<Self>;

        fn size(&self) -> Self::Metrics {
            ()
        }

        fn device(&self) -> usize {
            0
        }

        fn mem_size(&self) -> u64 {
            0
        }
    }

    fn new_file_list_input_cache(
        input: FileListInput,
        device_idx: usize,
    ) -> Arc<DevicesWrap<FileListInput>> {
        Arc::new(DevicesWrap::wrap(input))
    }

    #[derive(Debug, cached_derive::Cached, Named)]
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
            ModelTestId(format!("cls_{}_inp_{}", self.clf.id.0, input_id.0).leak())
        }
    }

    impl Stage for JoinAndClassifyTask {
        type In<'a> = FileListInput;
        type Out = ThoTensor<ClsOut, ModelTestId>;
        type Ctx = RunCtxImpl;
    }

    // #[async_trait]
    impl AsyncStage for JoinAndClassifyTask {
        async fn run(
            self: Arc<Self>,
            input: Arc<DevicesWrap<Self::In<'static>>>,
            mut ctx: Self::Ctx,
        ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
            let device_idx = ctx.tho_device_idx();

            // Orchestrate: build cached batch and run classifier on it.
            let batched = Arc::clone(&self.batch_builder)
                .run_cached(Arc::clone(&input), ctx.branch(false))
                .await?;


            // Build out_key for fairness and acquisitions
            let out_id = self.clf.out_id(batched.id());
            let out_key = Arc::new(out_id.clone()) as Arc<dyn CachableIdTr>;

            // Acquire max_parallel as needed using OOMHandler with Permits and Splits adjusters
            let size = *batched.size();
            let min_split_len = ABMetrics {
                dim_0: 32,
                dim_1: 0,
            };

            //**********************
            //overflow evaluating the requirement `&Simd<_, _>: Add<&Simd<_, _>>`
            // Adjusters::Input(Box::new(
            //     SplitsAdjuster::<ThoTensor2<ABData, SimpleId>, ABRange>::new(min_split_len),
            // ));
            //**********************
            let result = OOMHandler::new(
                vec![
                    Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(
                        MAX_PARALLEL,
                        *MaxParallelStep::TEN,
                    ))),
                    Adjusters::Input(Box::new(
                        SplitsAdjuster::<ABRange, ABMetrics>::new(min_split_len),
                    )),
                ],
                None,
            )
            .run(
                &self.clf,
                batched,
                ctx,
                Contexts::new(
                    SplitsAdjusterContextImpl::new(
                        size,
                        SingleDimTensorOutputBuilder::<ModelTestId, ClsOut>::new(
                            out_id,
                            1,
                            &[1, 1],
                            (Kind::Float, ThoDevice::from(device_idx).deref().clone()),
                            BuildAs::Sum,
                        )?,
                    ),
                    NoopRunnableContext {},
                    MaxParallelAdjustContextImpl::new(),
                ),
            )
            .await?;

            let final_device_idx = device_idx;
            // #[cfg(any(test, feature = "mock_global_ctx"))]
            // let _mock_ctx = MockGlobalCtxBuilder::install_global_ctx_mock();
            let num_devices = GlobalCtx::global().device_count();

            Ok(Arc::new(DevicesWrap::wrap(result)))
            // todo!()
        }
    }

    // ---------------- Test ----------------

    #[tokio::test]
    async fn test_ab_loader_and_classifier_cpu() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        let global_mock_ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false);

        let mut ctx = RunCtxImpl::new(1, "test");

        let loader = Arc::new(ABLoader::new(64, 4, Kind::Float));
        let clf = Arc::new(Classifier::new(
            ModelTestId("lin_cls"),
            4,
            200,
            0.1,
            vec![crate::runner::test::FailUntil::Never],
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
        let input = FileListInput::new(files, ModelTestId("files_0_1"));

        let input_cache = new_file_list_input_cache(input, ctx.tho_device_idx());
        let acc = Arc::clone(&join_and_cls)
            .run_cached(input_cache, ctx.branch(false))
            .await?;

        debug!("acc={acc:?}");

        acc.with_data_ref(0,
                          ctx.branch(false), None, |acc_t: &ThoTensor<ClsOut, ModelTestId>, _ctx| {
                  debug!("acc_t={acc_t:?} t=\n{}", acc_t.t_ref());

                  // Dim impls are to define domain-specific operations on tensors
                  let v = ClsOut::accuracy(acc_t);
                  assert!(v > 0.8, "training accuracy too low: {v}");
                  Ok(())
              })
        .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_ab_loader_and_classifier_cpu_many_files() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        let global_mock_ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false);
        let mut ctx = RunCtxImpl::new(0, "test");

        let loader = Arc::new(ABLoader::new(64, 4, Kind::Float));
        let clf = Arc::new(Classifier::new(
            ModelTestId("lin_cls_many"),
            4,
            100,
            0.1,
            vec![crate::runner::test::FailUntil::Never],
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
        let input = FileListInput::new(files, ModelTestId("files_0_1_2_3"));

        let input_cache = new_file_list_input_cache(input, ctx.tho_device_idx());
        let acc = Arc::clone(&join_and_cls)
            .run_cached(input_cache, ctx.branch(false))
            .await?;

        let ctx_for_data = ctx.branch(false);
        acc.with_data_ref(1,
                          ctx_for_data, None, |acc: &ThoTensor<ClsOut, ModelTestId>, _ctx| {
                                      let v = ClsOut::accuracy(acc);
                                      assert!(v > 0.8, "training accuracy too low (4 files): {v}");
                                      Ok(())
                                  })
        .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_classifier_oom_max_parallel_escalates() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cpu.so").expect("no libtorch_cpu.so");
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        let _global_ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false);
        let mut ctx = RunCtxImpl::new(1, "test");


        let loader = Arc::new(ABLoader::new(64, 4, Kind::Float));
        // Require MAX_PERMITS to proceed and also enforce MemLimit until splits reduce size
        let clf = Arc::new(Classifier::new(
            ModelTestId("lin_cls_oom"),
            4,
            50,
            0.1,
            vec![
                crate::runner::test::FailUntil::NaxParallel(MAX_PARALLEL),
                crate::runner::test::FailUntil::MemLimit(128),
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
        let input = FileListInput::new(files, ModelTestId("files_oom_permits"));

        let input_cache = new_file_list_input_cache(input, ctx.tho_device_idx());

        let acc = Arc::clone(&join_and_cls)
            .run_cached(input_cache, ctx.branch(false))
            .await?;

        // ctx.add_permit(GlobalCtx::global().acquire_many(1, 1, Arc::new(simple_id!("zzz"))).await?);

        acc.with_data_ref(1,
                          ctx.branch(false), None, |acc: &ThoTensor<ClsOut, ModelTestId>, _ctx| {
              debug!("acc={}", acc.t_ref());

              let v = ClsOut::accuracy(acc);
              assert!(v >= 0.5, "should train after permits escalation: {v}");
              Ok(())
          })
        .await?;

        Ok(())
    }

}
