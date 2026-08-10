#pragma once
#include <cstdint>
#include <torch/torch.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/adam.h>
#include <torch/optim/adamw.h>
#include <torch/optim/sgd.h>

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
    at::Tensor* exp_avg,
    at::Tensor* exp_avg_sq,
    at::Tensor* max_exp_avg_sq);

// AdamW has the same options and parameter-state layout as Adam, but LibTorch uses distinct runtime types that require separate checked entry points.
extern "C"  ulong tho_adam_w_group_count(torch::optim::Optimizer* opt);

extern "C"  void tho_adam_w_get_defaults(
    torch::optim::Optimizer* opt,
    tho_adam_options* out);

extern "C"  void tho_adam_w_get_group(
    torch::optim::Optimizer* opt,
    size_t group_index,
    tho_adam_group* out);

extern "C"  void tho_adam_w_get_param_state(
    torch::optim::Optimizer* opt,
    size_t group_index,
    size_t parameter_index,
    tho_adam_param_state* out);

extern "C"  void tho_adam_w_set_defaults(
    torch::optim::Optimizer* opt,
    const tho_adam_options* options);

extern "C"  void tho_adam_w_set_group_options(
    torch::optim::Optimizer* opt,
    size_t group_index,
    const tho_adam_options* options);

extern "C"  void tho_adam_w_set_param_state(
    torch::optim::Optimizer* opt,
    size_t group_index,
    size_t parameter_index,
    int32_t has_state,
    int64_t step,
    at::Tensor* exp_avg,
    at::Tensor* exp_avg_sq,
    at::Tensor* max_exp_avg_sq);

typedef struct tho_sgd_options {
    double lr;
    double momentum;
    double dampening;
    double weight_decay;
    int32_t nesterov;
} tho_sgd_options;

typedef struct tho_sgd_group {
    tho_sgd_options options;
    size_t parameter_count;
} tho_sgd_group;

typedef struct tho_sgd_param_state {
    int32_t has_state;

    // The returned pointer is owned by Rust after Tensor::from_ptr().
    at::Tensor* momentum_buffer;
} tho_sgd_param_state;

extern "C" ulong tho_sgd_group_count(torch::optim::Optimizer* opt);

extern "C" void tho_sgd_get_defaults(
    torch::optim::Optimizer* opt,
    tho_sgd_options* out);

extern "C" void tho_sgd_get_group(
    torch::optim::Optimizer* opt,
    size_t group_index,
    tho_sgd_group* out);

extern "C" void tho_sgd_get_param_state(
    torch::optim::Optimizer* opt,
    size_t group_index,
    size_t parameter_index,
    tho_sgd_param_state* out);

extern "C" void tho_sgd_set_defaults(
    torch::optim::Optimizer* opt,
    const tho_sgd_options* options);

extern "C" void tho_sgd_set_group_options(
    torch::optim::Optimizer* opt,
    size_t group_index,
    const tho_sgd_options* options);

extern "C" void tho_sgd_set_param_state(
    torch::optim::Optimizer* opt,
    size_t group_index,
    size_t parameter_index,
    int32_t has_state,
    at::Tensor* momentum_buffer);
