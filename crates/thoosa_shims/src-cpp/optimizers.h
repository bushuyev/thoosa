#pragma once
#include <cstdint>
#include <torch/torch.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/adam.h>

#include "logging.h"

typedef struct ato_adam_options {
    double lr;
    double beta1;
    double beta2;
    double eps;
    double weight_decay;
    int32_t amsgrad;
} ato_adam_options;

typedef struct ato_adam_group {
    ato_adam_options options;
    size_t parameter_count;
} ato_adam_group;

typedef struct ato_adam_param_state {
    int64_t step;
    int32_t has_state;

    // Owned torch::Tensor pointers returned to Rust.
    //
    // Rust must wrap each non-null pointer in Tensor::from_ptr(), after which
    // Tensor::drop() eventually calls at_free().
    at::Tensor exp_avg;
    at::Tensor exp_avg_sq;
    at::Tensor max_exp_avg_sq;
} ato_adam_param_state;

extern "C"  ulong ato_adam_group_count(torch::optim::Optimizer* opt);

void ato_adam_get_defaults(
    torch::optim::Adam opt,
    ato_adam_options* out);

void ato_adam_get_group(
    torch::optim::Adam opt,
    size_t group_index,
    ato_adam_group* out);

void ato_adam_get_param_state(
    torch::optim::Adam opt,
    size_t group_index,
    size_t parameter_index,
    ato_adam_param_state* out);

void ato_adam_set_defaults(
    torch::optim::Adam opt,
    const ato_adam_options* options);

void ato_adam_set_group_options(
    torch::optim::Adam opt,
    size_t group_index,
    const ato_adam_options* options);

void ato_adam_set_param_state(
    torch::optim::Adam opt,
    size_t group_index,
    size_t parameter_index,
    int32_t has_state,
    int64_t step,
    at::Tensor exp_avg,
    at::Tensor exp_avg_sq,
    at::Tensor max_exp_avg_sq);
