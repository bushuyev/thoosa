#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/cuda/block_reduce.cuh>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>

#include <thrust/pair.h>

#include <algorithm>
#include <vector>


namespace {

using WelfordType =
    at::native::WelfordData<float, int64_t>;

using WelfordOp =
    at::native::WelfordOps<
        float,
        float,
        int64_t,
        thrust::pair<float, float>
    >;


__global__ void z_norm_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t outer,
    int64_t inner,
    float eps)
{
    const int64_t row =
        static_cast<int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (row >= outer) {
        return;
    }

    const float* src = x + row * inner;
    float* dst = out + row * inner;

    WelfordOp welford{
        /* correction = */ 0.0f,
        /* take_sqrt = */ false
    };

    WelfordType acc{
        0.0f,
        0.0f,
        0,
        0.0f
    };

    for (int64_t i = 0; i < inner; ++i) {
        acc = welford.reduce(
            acc,
            src[i],
            i
        );
    }

    const auto stats = welford.project(acc);

    const float var = stats.first;

    const float mean = stats.second;

    const float rstd = c10::cuda::compat::rsqrt( var + eps );

    for (int64_t i = 0; i < inner; ++i) {
        dst[i] = (src[i] - mean) * rstd;
    }
}

template<typename T>
__global__ void z_norm_block_kernel(
    const T* __restrict__ x,
    T* __restrict__ out,
    int64_t outer,
    int64_t inner,
    float eps)
{
    using WelfordType =
        at::native::WelfordData<float, int64_t>;

    using WelfordOp =
        at::native::WelfordOps<
            T,
            float,
            int64_t,
            thrust::pair<float, float>
        >;

    // Same shared-storage approach used by
    // PyTorch LayerNorm.
    __shared__
        typename std::aligned_storage<
            sizeof(WelfordType),
            alignof(WelfordType)
        >::type
        reduction_storage[C10_WARP_SIZE];

    auto* reduction_buffer =
        reinterpret_cast<WelfordType*>(
            reduction_storage);

    __shared__ float shared_mean;
    __shared__ float shared_rstd;

    const int64_t row =
        blockIdx.x;

    if (row >= outer) {
        return;
    }

    const T* src =
        x + row * inner;

    T* dst =
        out + row * inner;

    WelfordOp op{
        0.0f,
        false
    };

    WelfordType acc{
        0.0f,
        0.0f,
        0,
        0.0f
    };

    //
    // Each thread handles part of the row.
    //
    for (int64_t i = threadIdx.x;
         i < inner;
         i += blockDim.x)
    {
        acc = op.reduce(
            acc,
            src[i],
            i
        );
    }

    //
    // Reuse PyTorch's warp/block reduction machinery.
    //
    acc =
        at::native::cuda_utils::BlockReduce(
            acc,
            op,
            WelfordType{
                0.0f,
                0.0f,
                0,
                0.0f
            },
            reduction_buffer
        );

    if (threadIdx.x == 0) {
        const auto stats =
            op.project(acc);

        shared_mean =
            stats.second;

        shared_rstd =
            c10::cuda::compat::rsqrt(
                stats.first + eps
            );
    }

    __syncthreads();

    const float mean =
        shared_mean;

    const float rstd =
        shared_rstd;

    //
    // Same kernel proceeds directly to normalization.
    //
    for (int64_t i = threadIdx.x;
         i < inner;
         i += blockDim.x)
    {
        dst[i] =
            static_cast<T>(
                (static_cast<float>(src[i]) - mean)
                * rstd
            );
    }
}

} // namespace


at::Tensor z_norm_cuda(
    const at::Tensor& x,
    at::IntArrayRef dims,
    double eps)
{
    TORCH_CHECK(
        x.is_cuda(),
        "z_norm: x must be CUDA");

    TORCH_CHECK(
        x.scalar_type() == at::kFloat,
        "z_norm: currently only float32 is supported");

    TORCH_CHECK(
        x.is_contiguous(),
        "z_norm: currently requires contiguous input");

    TORCH_CHECK(
        !dims.empty(),
        "z_norm: dims cannot be empty");

    const int64_t ndim = x.dim();

    std::vector<int64_t> normalized_dims;
    normalized_dims.reserve(dims.size());

    for (int64_t d : dims) {
        if (d < 0) {
            d += ndim;
        }

        TORCH_CHECK(
            d >= 0 && d < ndim,
            "z_norm: invalid dimension ",
            d);

        normalized_dims.push_back(d);
    }

    std::sort(
        normalized_dims.begin(),
        normalized_dims.end()
    );

    TORCH_CHECK(
        std::adjacent_find(
            normalized_dims.begin(),
            normalized_dims.end())
            == normalized_dims.end(),
        "z_norm: duplicate dimensions");

    // Version 1 only supports a trailing contiguous
    // set of normalization dimensions.
    //
    // Examples:
    //
    // [K, 2, 3], dims=[1,2]   OK
    // [K, 2, 3], dims=[2]     OK
    // [K, 2, 3], dims=[0,2]   NOT OK
    //
    const int64_t first_dim =
        ndim -
        static_cast<int64_t>(
            normalized_dims.size());

    for (size_t i = 0;
         i < normalized_dims.size();
         ++i)
    {
        TORCH_CHECK(
            normalized_dims[i] ==
                first_dim +
                static_cast<int64_t>(i),
            "z_norm: only trailing dimensions supported");
    }

    int64_t inner = 1;

    for (const int64_t d : normalized_dims) {
        inner *= x.size(d);
    }

    TORCH_CHECK(
        inner > 0,
        "z_norm: empty reduction dimension");

    const int64_t outer = x.numel() / inner;//number of independent normalization groups

    auto out =
        at::empty_like(x);

    c10::cuda::CUDAGuard device_guard(
        x.device());

    auto stream =
        at::cuda::getCurrentCUDAStream(
            x.get_device());



    if (inner <= 32) {
        constexpr int threads = 256;

        const int64_t blocks = (outer + threads - 1) / threads;

        z_norm_kernel<<<
           blocks,
           threads,
           0,
           stream.stream()
       >>>(
           x.const_data_ptr<float>(),
           out.mutable_data_ptr<float>(),
           outer,
           inner,
           static_cast<float>(eps)
       );

    } else {
        constexpr int threads = 512;

        const int64_t blocks = (outer + threads - 1) / threads;

        z_norm_block_kernel<<<
           blocks,
           threads,
           0,
           stream.stream()
       >>>(
           x.const_data_ptr<float>(),
           out.mutable_data_ptr<float>(),
           outer,
           inner,
           static_cast<float>(eps)
       );
    }


    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}