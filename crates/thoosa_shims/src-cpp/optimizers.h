#pragma once
#include <cstdint>
#include <torch/torch.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/adam.h>

#include "logging.h"

typedef struct tho_adam_options {
    double lr;
    double beta1;
    double beta2;
    double eps;
    double weight_decay;
    int32_t amsgrad;
} tho_adam_options;

typedef struct tho_adam_group {
    tho_adam_options options;
    size_t parameter_count;
} tho_adam_group;

typedef struct tho_adam_param_state {
    int64_t step;
    int32_t has_state;

    // Owned torch::Tensor pointers returned to Rust.
    //
    // Rust must wrap each non-null pointer in Tensor::from_ptr(), after which
    // Tensor::drop() eventually calls at_free().
    at::Tensor* exp_avg;
    at::Tensor* exp_avg_sq;
    at::Tensor* max_exp_avg_sq;
} tho_adam_param_state;

extern "C"  ulong tho_adam_group_count(torch::optim::Optimizer* opt);

extern "C"   void tho_adam_get_defaults(
    torch::optim::Optimizer* opt,
    tho_adam_options* out);

extern "C"   void tho_adam_get_group(
    torch::optim::Optimizer* opt,
    size_t group_index,
    tho_adam_group* out);

extern "C"   void tho_adam_get_param_state(
    torch::optim::Optimizer* opt,
    size_t group_index,
    size_t parameter_index,
    tho_adam_param_state* out);

extern "C"   void tho_adam_set_defaults(
    torch::optim::Optimizer* opt,
    const tho_adam_options* options);

extern "C"   void tho_adam_set_group_options(
    torch::optim::Optimizer* opt,
    size_t group_index,
    const tho_adam_options* options);

extern "C"  void tho_adam_set_param_state(
    torch::optim::Optimizer* opt,
    size_t group_index,
    size_t parameter_index,
    int32_t has_state,
    int64_t step,
    at::Tensor exp_avg,
    at::Tensor exp_avg_sq,
    at::Tensor max_exp_avg_sq);
