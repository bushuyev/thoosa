#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <vector>

namespace {

__global__ void z_norm_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t outer,
    int64_t inner,
    float eps)
{
    const int64_t row = blockIdx.x;

    if (row >= outer) {
        return;
    }

    const float* row_x = x + row * inner;
    float* row_out = out + row * inner;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // First pass: statistics.
    for (int64_t i = threadIdx.x; i < inner; i += blockDim.x) {
        const float v = row_x[i];

        local_sum += v;
        local_sum_sq += v * v;
    }

    extern __shared__ float shared[];

    float* shared_sum = shared;
    float* shared_sum_sq = shared + blockDim.x;

    shared_sum[threadIdx.x] = local_sum;
    shared_sum_sq[threadIdx.x] = local_sum_sq;

    __syncthreads();

    // blockDim.x is a power of two (256 below).
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] +=
                shared_sum[threadIdx.x + stride];

            shared_sum_sq[threadIdx.x] +=
                shared_sum_sq[threadIdx.x + stride];
        }

        __syncthreads();
    }

    const float mean =
        shared_sum[0] / static_cast<float>(inner);

    float var =
        shared_sum_sq[0] / static_cast<float>(inner)
        - mean * mean;

    // Protect against a tiny negative value caused by FP rounding.
    var = fmaxf(var, 0.0f);

    const float inv_std = rsqrtf(var + eps);

    // Second pass: normalize.
    for (int64_t i = threadIdx.x; i < inner; i += blockDim.x) {
        row_out[i] = (row_x[i] - mean) * inv_std;
    }
}

} // namespace


at::Tensor z_norm_cuda(
    const at::Tensor& x,
    at::IntArrayRef dims,
    double eps)
{
    TORCH_CHECK(x.is_cuda(), "z_norm: x must be CUDA");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat,
        "z_norm: first implementation only supports float32");
    TORCH_CHECK(
        x.is_contiguous(),
        "z_norm: first implementation requires contiguous tensor");
    TORCH_CHECK(!dims.empty(), "z_norm: dims must not be empty");

    const int64_t ndim = x.dim();

    std::vector<int64_t> normalized_dims;
    normalized_dims.reserve(dims.size());

    for (int64_t d : dims) {
        if (d < 0) {
            d += ndim;
        }

        TORCH_CHECK(
            d >= 0 && d < ndim,
            "z_norm: invalid dim ",
            d);

        normalized_dims.push_back(d);
    }

    std::sort(normalized_dims.begin(), normalized_dims.end());

    TORCH_CHECK(
        std::adjacent_find(
            normalized_dims.begin(),
            normalized_dims.end())
            == normalized_dims.end(),
        "z_norm: duplicate dimensions");

    // For version 1, dims must be the trailing dimensions.
    const int64_t first_normalized_dim =
        ndim - static_cast<int64_t>(normalized_dims.size());

    for (size_t i = 0; i < normalized_dims.size(); ++i) {
        TORCH_CHECK(
            normalized_dims[i] ==
                first_normalized_dim + static_cast<int64_t>(i),
            "z_norm: version 1 only supports trailing dimensions");
    }

    int64_t inner = 1;

    for (const int64_t d : normalized_dims) {
        inner *= x.size(d);
    }

    TORCH_CHECK(inner > 0, "z_norm: empty normalization dimension");

    const int64_t outer = x.numel() / inner;

    auto out = at::empty_like(x);

    constexpr int threads = 256;

    const size_t shared_mem =
        2 * threads * sizeof(float);

    at::cuda::CUDAGuard guard(x.device());

    auto stream =
        at::cuda::getCurrentCUDAStream(x.get_device());

    z_norm_kernel<<<
        outer,
        threads,
        shared_mem,
        stream.stream()>>>(
            x.const_data_ptr<float>(),
            out.mutable_data_ptr<float>(),
            outer,
            inner,
            static_cast<float>(eps));

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}