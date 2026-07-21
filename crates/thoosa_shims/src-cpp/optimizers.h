#pragma once
#include <cstdint>
#include <ATen/ATen.h>
#include <torch/optim/optimizer.h>

#include "logging.h"

struct AdamStateView {
    bool exists;
    int64_t step;

    at::Tensor exp_avg;
    at::Tensor exp_avg_sq;
    at::Tensor max_exp_avg_sq;
};

// std::vector<AdamStateView>
extern "C" void tho_snapshot_adam_states(
    // torch::optim::Optimizer& base
);