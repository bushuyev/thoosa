# Parameter optimization

Thoosa supports parameter optimization through a partial implementation of [c-TPE: Tree-structured Parzen Estimator with Inequality Constraints for Expensive Hyperparameter Optimization](https://arxiv.org/pdf/2211.14411v2). The optimizer lives in [`thoosa_common::parzen`](../crates/thoosa_common/src/parzen/optimizer.rs).

This is still work in progress. Parameter configuration was intended to be used from a custom UI editor; currently it is only used in code.

## Defining parameters with RangeOrValue

[`RangeOrValue<V, T>`](../crates/thoosa_shared/src/lib.rs) describes a parameter's value or search domain. `V` is its value type; `T` identifies fields when resolving dependencies and defaults to `String`. Generated configuration types such as `CommonTrainingRunConfigRangeOrValue` use these declarations to build concrete run configurations.

| Declaration | Meaning |
| --- | --- |
| `Value(value)` | Keep this exact value in every configuration. |
| `Range { range, step: Some((increment, digits)) }` | Enumerate a grid with the given increment. `digits` controls decimal rounding for real-valued grid entries; it is not the number of trials. |
| `Range { range, step: None }` | Give the optimizer a hyperparameter search domain, without specifying grid spacing. Integer fields remain integer parameters; real fields define real-valued domains. |

These meanings apply to the run-configuration conversion path through `add_range_or_value`, rather than direct iteration over `RangeOrValue`.

### Exact values

Use `Value` for parameters that should remain fixed, including numeric values and booleans:

```rust
use thoosa_shared::RangeOrValue;

let epochs: RangeOrValue<u32> = RangeOrValue::Value(10);
let learning_rate: RangeOrValue<f64> = RangeOrValue::Value(0.0001);
let amsgrad: RangeOrValue<bool> = RangeOrValue::Value(true);
```

`run_all_config_0` fixes parameters such as batch size, patience, and AdamW settings this way while allowing other fields to vary.

### Grid search

Set `step` to `Some((increment, digits))` to enumerate candidates. Grid ranges exclude their end bound. Multiple independent grid fields form combinations of their candidate values.

```rust
use thoosa_shared::RangeOrValue;

// Candidate windows: 100, 200, ..., 900.
let window: RangeOrValue<u32> = RangeOrValue::Range {
    range: 100..1000,
    step: Some((100, 0)),
};

// Candidate quantiles: 0.00, 0.05, 0.10, 0.15.
let quantile: RangeOrValue<f64> = RangeOrValue::Range {
    range: 0.0..0.2,
    step: Some((0.05, 2)),
};
```

Choose a nonzero increment that moves toward the end, and enough decimal precision for real-valued steps to make progress. The grid conversion uses [`NamedRange`](../crates/thoosa_shared/src/val.rs), which validates these conditions. The HBAC motif-search helper uses a smaller grid of 40, 80, 120, and 160 filtered samples.

### Hyperparameter search spaces

Leave `step` as `None` to let the optimizer propose values within the domain:

```rust
use thoosa_shared::RangeOrValue;

let learning_rate: RangeOrValue<f64> = RangeOrValue::Range {
    range: 0.00001..0.001,
    step: None,
};
let num_layers: RangeOrValue<u32> = RangeOrValue::Range {
    range: 2..64,
    step: None,
};
```

These declare bounds, not a list of evenly spaced candidates or a trial count. `run_all_config_0` uses this form for learning rates and model parameters, including CosineNet coefficients, attention head sizes, and final-network layer counts.

Fixed values, grid fields, and optimization fields can coexist in an optimization configuration. The generated `opt_run_configs()` helper separates constants, grid combinations, and hyperparameter domains. The generated `grid_run_configs()` helper is for grid-only configurations and rejects optimization parameters. Their traits are defined in [`thoosa_run_config_args`](../crates/thoosa_macros/thoosa_run_config_args/src/lib.rs); the generated implementations are in the [derive crate](../crates/thoosa_macros/thoosa_run_config_args_derive/src/lib.rs).

### Parameters that depend on other fields

`RangeOrValue::dynamic` accepts a closure that reads selected `NamedVal` fields and returns another declaration. It can therefore produce a fixed value, a grid, or an optimization domain based on another parameter. For example, using string field names:

```rust
use thoosa_shared::RangeOrValue;

let filter_window: RangeOrValue<u32> = RangeOrValue::dynamic(|fields| {
    let window = fields.iter()
        .find(|field| field.path().as_str() == "window")
        .map(|field| field.val().to_i64(0) as u32)
        .unwrap_or(100);
    RangeOrValue::Value(window / 2)
});
```

The alternative HBAC test profile uses the same mechanism with the generated `FindMotifsRunConfigFields::Window` field identifier: its filter-window range depends on the selected motif window. Dynamic closures are code-only and cannot be serialized.
