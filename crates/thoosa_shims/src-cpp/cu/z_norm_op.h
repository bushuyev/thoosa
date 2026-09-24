#pragma once

#include <ATen/ATen.h>

at::Tensor z_norm_cuda(
    const at::Tensor& x,
    at::IntArrayRef dims,
    double eps);