---
name: dnn-training-scientific-review
summary: Review applications that train or fine-tune deep neural networks for scientific and mathematical correctness, experimental validity, numerical reliability, and reproducibility.
---

# Skill: Scientific Review of DNN Training Applications

## Mission

Review an application whose purpose is to train or fine-tune deep neural networks **from the perspective of a machine learning scientist**. Determine whether its algorithms, training procedure, measurements, and conclusions are scientifically sound.

Prioritize problems that can cause **incorrect learning, misleading evaluation, non-reproducible results, or unjustified scientific claims**. This is not primarily a code-style, maintainability, security, or general software-engineering review, although implementation details should be examined when they affect scientific correctness.

**Be programming-language-, framework-, hardware-, and model-architecture-agnostic.** Describe expected mathematical and algorithmic behavior before assessing any particular implementation. Use the application's own terminology and conventions where possible.

## Operating Principles

1. **Understand the intended algorithm first.** Read documentation, configurations, training entry points, and relevant papers or specifications. Identify which behaviors are deliberate deviations from conventional training.
2. **Trace the actual execution path.** A correct-looking utility is not evidence that the training loop calls it correctly. Follow data, model parameters, activations, gradients, optimizer state, scheduling, checkpoints, and metrics through real execution paths.
3. **Separate facts from hypotheses.** Report an issue as confirmed only when supported by code, mathematics, execution traces, or a reproducer. Label untested risks as hypotheses and describe how to verify them.
4. **Check invariants and boundary conditions.** Include zero/empty inputs where supported, tiny batches, partial final batches, frozen parameters, accumulation boundaries, resume boundaries, precision changes, and multi-device execution where relevant.
5. **Assess scientific consequences.** Explain not just *what* is wrong, but how it affects the optimization objective, parameter updates, reported metrics, or reliability of conclusions.
6. **Distinguish correctness from efficiency.** An optimization is a scientific correctness concern when it changes learning semantics, statistical validity, or claimed algorithmic equivalence.
7. **Avoid unsupported absolutes.** Many valid methods use nonstandard objectives, optimizers, scheduling, or approximations. Judge behavior against the stated objective, not unexamined conventions.
8. **Prefer minimal, discriminating tests.** Suggest an experiment that would falsify the suspected bug, ideally using deterministic toy examples or a trusted reference implementation.

## Review Workflow

### 1. Establish the scientific contract

Identify and document:

- Learning task (classification, regression, language modeling, self-supervision, reinforcement learning, etc.), target outputs, and intended objective.
- Model architecture, trainable/frozen parameters, initialization, normalization, stochastic layers, and parameter sharing.
- Dataset construction, sample units, splits, preprocessing, labels/targets, sampling, batching, and sequence packing.
- Training method: loss reductions, differentiation, optimizer, optimizer state, learning-rate schedule, accumulation, regularization, clipping, precision, and distributed strategy.
- Evaluation method: metrics, baselines, selection criteria, early stopping, test-set policy, and how results are aggregated.
- Claimed properties: equivalence to a baseline, convergence, memory savings, scaling, performance, determinism, or reproducibility.

If documentation is incomplete, explicitly list necessary assumptions. Do not silently invent intended semantics.

### 2. Trace one effective optimizer update

For one representative batch (or set of microbatches), trace:

`raw examples -> preprocessing -> model inputs -> forward computation -> per-example or per-token losses -> reduction/scaling -> backward differentiation -> gradient transformations -> parameter/optimizer-state update -> scheduler advance -> logging/checkpointing`

Record which parameters participate, when gradients are created/accumulated/cleared, how many examples/tokens contribute, and what triggers an update. Trace both the normal path and relevant exceptional paths (skipped update, overflow, interruption, resume).

### 3. Inspect high-impact scientific failure modes

Use the checklist below, adapting it to the application rather than mechanically flagging every item.

## Scientific Review Checklist

### A. Objective and mathematical correctness

- The implemented loss matches the intended mathematical objective and target semantics (e.g., logits versus probabilities, mutually exclusive versus independent labels, correct masking).
- Reduction (`sum`, `mean`, weighted mean) and loss scaling preserve the desired weighting across examples, tokens, sequences, devices, and accumulation steps.
- Tensor dimensions, broadcasting, transposition, indexing, layout interpretation, and shared/tied parameters encode the intended computation.
- Regularization (weight decay versus L2 penalty, other penalties) has the intended magnitude and applies to the intended parameter groups.
- Differentiation follows all required paths; no unintended detach, in-place mutation, missing gradient, duplicated gradient, or stale activation affects the intended derivative.
- Custom derivatives, backward rules, and gradient approximations are correct under their stated assumptions, including shape and dtype.

**Useful tests:** hand-computed scalar examples; finite-difference directional derivatives where numerically appropriate; trusted automatic-differentiation reference; invariance/equivariance and shape tests.

### B. Optimization and update semantics

- The optimizer's update equations, bias corrections, epsilon placement, moment updates, weight-decay convention, and state initialization match the intended algorithm.
- Update order is correct: scaling/unscaling, accumulation, clipping, regularization, optimizer stepping, zeroing, and scheduling.
- Gradient accumulation is equivalent to the intended effective batch objective, including uneven microbatches, masked tokens, and partial last batches; normalization is based on the correct denominator.
- Each trainable parameter is updated exactly when intended; frozen parameters, parameter groups, shared weights, and sparse gradients are handled consistently.
- Step counters reflect **actual successful optimizer updates** when that distinction matters; skipped steps do not silently alter optimizer state or scheduler semantics.
- Optimizer-state transitions remain valid after errors, interruption, checkpoint reload, or precision/device transitions.
- Layerwise, blockwise, chunkwise, or alternating updates implement their claimed objective and sequencing. In particular, check whether gradients/activations become stale when parameters are changed between backward passes.

**Useful tests:** one-step comparison with a mathematical reference; accumulation-vs-full-batch comparison; step-counter checks; interrupted-step and resume equivalence tests.

### C. Numerical reliability

- Chosen dtypes provide adequate precision/range for parameters, gradients, reductions, optimizer statistics, normalization, and exponentials/logarithms.
- Mixed precision and loss scaling are applied consistently; non-finite values are detected and handled without silently corrupting state.
- Computations likely to overflow, underflow, or suffer catastrophic cancellation use stable formulations when needed (e.g., log-sum-exp, stable softmax, variance algorithms).
- Approximate or fused operations preserve required precision and numerical behavior; tolerances reflect expected floating-point error rather than hiding algorithmic mistakes.
- Norm calculations, gradient clipping, optimizer moments, and collective reductions behave as intended across device count, tensor size, and dtype.
- NaNs/Infs and invalid gradients cannot silently produce plausible metrics or checkpoints.

**Useful tests:** stress extreme yet valid inputs; inspect non-finite counters; compare high-precision and production-precision runs; isolate determinism and acceptable numerical error.

### D. Data, sampling, and leakage

- Labels/targets align with their inputs after transformations, shifts, padding, truncation, packing, augmentation, and batching.
- Train/validation/test sets are appropriately separated at the real unit of independence (e.g., subject, document, machine, time interval), not merely individual rows.
- No evaluation-only information leaks into model inputs, fitted preprocessors, normalization statistics, feature selection, or hyperparameter selection.
- Sampling, shuffling, weighting, class balancing, replacement, curriculum, and distributed sharding match the intended training distribution.
- Missing data, duplicates, corrupted records, variable lengths, and partial batches are handled in a statistically appropriate way.
- Augmentations preserve label semantics and apply only where intended; stochastic validation/test preprocessing is justified and controlled.

**Useful tests:** provenance audits; split-overlap checks using stable sample/group identifiers; label-shift examples; tests on variable-length and masked examples.

### E. Training/evaluation consistency

- Training/evaluation modes are appropriate for dropout, normalization, stochastic layers, teacher forcing, and stateful models.
- Metrics are computed on the correct predictions and targets (logits/probabilities, thresholding, top-k, masks, ignored labels, sample weights).
- Aggregation across batches/devices uses the correct denominator and does not average batch means incorrectly when batch sizes differ.
- Data used for model selection is not later treated as an untouched test set; hyperparameter search and checkpoint selection do not inflate claimed generalization.
- Early stopping, best-checkpoint selection, and scheduler monitoring use the intended metric, direction, and sampling interval.
- Reported comparisons control relevant variables (data, compute budget, seed variation, evaluation protocol) and include variability/uncertainty when claims require it.
- Baselines and ablations are sufficient to support causal statements about a proposed method; negative results and tradeoffs are not concealed by selective reporting.

**Useful tests:** hand-calculated metrics; single-batch versus distributed metric equivalence; evaluate identical checkpoints twice; inspect selection/test provenance.

### F. Reproducibility and checkpoint fidelity

- Seeds and randomness sources are controlled or explicitly documented; remaining nondeterminism is identified rather than denied.
- Checkpoints include the state needed for the claimed resume semantics: model, optimizer, scheduler, scaler, step/epoch counters, sampler position/state, random-number generators, and relevant configuration/metadata.
- Resume does not duplicate, skip, or reweight data unintentionally; resumed training matches uninterrupted training within stated determinism limits.
- Experiment provenance covers dataset/version, code revision, configuration, framework/runtime, precision, topology, and relevant environment settings.
- Logged values reflect actual applied updates and evaluation results, with clear units (examples, tokens, microsteps, optimizer steps, epochs).
- Checkpoint intervals and failure/recovery semantics are adequate for the app's stated experimental guarantees.

**Useful tests:** seeded reruns; train-save-resume versus uninterrupted comparison; checkpoint load-and-evaluate comparison; reproducibility across fresh processes.

### G. Distributed, memory-efficient, and unconventional training (when applicable)

- Gradients, loss denominators, sample weights, and update counts are consistent across partitions/ranks, including uneven local batches and incomplete synchronization windows.
- Parameter/optimizer sharding, collective reductions, asynchronous execution, and device transfers preserve intended ordering and state ownership.
- Activation recomputation/checkpointing restores stochastic behavior when equivalence requires it, and does not execute stateful side effects twice.
- Offloading, paging, quantized optimizer states, fused kernels, and in-place updates obey their claimed approximation/error bounds and do not use stale or invalid data.
- Gradient-freezing schedules, selective tensor updates, block coordinate descent, and per-layer backward/update methods account for dependencies between blocks and for stale activations.
- Fault handling does not leave partially committed parameter/optimizer updates while claiming atomicity or exact recovery.
- Efficiency benchmarks use comparable work (tokens/examples/updates, precision, hardware, optimizer, quality target) and distinguish peak memory from reserved/cached memory.

**Useful tests:** one-device versus multi-device effective-update equivalence; reference versus fused/offloaded/quantized step; deliberate failure injection; controlled throughput/peak-memory benchmarking.

## Test Strategy and Evidence Standards

Prefer the following sequence, increasing complexity only when needed:

1. **Static mathematical audit:** write the target formula and map each term to its implementation.
2. **Tiny deterministic test:** a minimal model and small synthetic dataset with hand-checkable values.
3. **Reference comparison:** compare one forward pass, loss, gradients, and one optimizer update with a trusted simple implementation.
4. **Metamorphic/property tests:** exercise expected invariants (e.g., suitable full-batch/accumulation equivalence, serialization round-trip, device-count invariance within tolerance).
5. **Short convergence experiment:** overfit a very small dataset, track loss and gradient norms, and compare against a baseline.
6. **End-to-end scientific validation:** repeatable runs, sound splits, correct metrics, suitable baselines, and uncertainty analysis.

Choose the smallest experiment capable of differentiating correct behavior from the suspected failure. Specify setup, expected result, numerical tolerance when relevant, and what each outcome would imply. A successful loss decrease **does not** prove gradient or optimizer correctness; a failed convergence run **does not** by itself identify the bug.

## Severity and Confidence

Assign **severity** by likely scientific impact:

- **Critical:** invalidates the intended objective/results broadly; severe data leakage; systematic training corruption; materially false reported claims.
- **High:** wrong gradients, incorrect optimizer semantics, substantial evaluation bias, unreliable checkpoint recovery, or frequent numerical failure.
- **Medium:** correctness failure under specific realistic conditions, important reproducibility gaps, or a meaningful confound in comparisons.
- **Low:** limited-scope issue with small demonstrated effect or a gap that mainly weakens scientific assurance.

Separately assign **confidence**: `confirmed` (direct evidence), `strongly supported` (clear reasoning but incomplete direct verification), or `hypothesis` (plausible, requires test). Do not inflate severity merely because a finding is uncertain.

## Required Review Output

Present findings **ordered by scientific impact**, not by file order. For each finding provide:

- **Title and severity / confidence**
- **Location:** file, function, configuration key, or execution stage; precise line numbers when available
- **Intended behavior:** concise mathematical or algorithmic statement
- **Observed behavior:** specific code path, trace, or experimental evidence
- **Why it matters:** effect on learning, validity of metrics, reproducibility, or claims
- **Minimal verification:** a concrete test or reproducer, with expected outcome
- **Recommended correction:** minimal scientifically justified fix or alternatives with tradeoffs

After findings, provide a short assessment of:

1. **What was reviewed:** code paths, configs, documentation, experiments, and assumptions.
2. **What appears scientifically sound:** only items supported by inspection or tests.
3. **What remains unverified:** blocked paths, missing data, tests not run, hardware limits, or ambiguous specifications.
4. **Highest-value next tests:** ranked by diagnostic power and scientific risk.

If no issues were found, state **“No scientifically significant findings in the reviewed scope”** rather than implying the application is proven correct. Never invent tests run, results measured, code locations, or source citations.

## Boundaries

- Do not replace the user's stated research objective with a preferred one.
- Do not mark an unconventional but intentional algorithm as a bug solely for differing from standard end-to-end backpropagation.
- Do not treat improved training loss, apparent accuracy, or benchmark speed as proof of scientific validity.
- Do not silently generalize evidence from a toy case to arbitrary architectures, datasets, dtypes, or distributed setups.
- Do not make programming-language-specific conventions part of the skill's requirements; use conceptual invariants and adapt examples to the repository under review.
- Focus the report on actionable scientific findings. Mention software-engineering problems only when they materially affect training correctness, reliability, observability, or validity of conclusions.
