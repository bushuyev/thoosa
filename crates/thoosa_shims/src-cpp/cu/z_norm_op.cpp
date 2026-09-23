#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>

#include <torch/library.h>

#include <cstdint>
#include <string>

#include <exception>
#include <vector>

at::Tensor z_norm_cuda(
    const at::Tensor& x,
    at::IntArrayRef dims,
    double eps);


// --------------------------
// Operator registration
// --------------------------

TORCH_LIBRARY(thoosa, m) {
    m.def(
        "z_norm(Tensor x, int[] dims, float eps) -> Tensor");
}


TORCH_LIBRARY_IMPL(thoosa, CUDA, m) {
    m.impl(
        "z_norm",
        TORCH_FN(z_norm_cuda));
}



namespace {

    thread_local std::string last_error;

}


extern "C"
void* thoosa_z_norm(
    const void* x_ptr,
    const int64_t* dims,
    size_t dims_len,
    double eps) noexcept
{
    try {
        last_error.clear();

        const auto* x =
            static_cast<const at::Tensor*>(x_ptr);

        TORCH_CHECK(
            x != nullptr,
            "thoosa_z_norm: null input tensor");

        static auto op =
            c10::Dispatcher::singleton()
                .findSchemaOrThrow(
                    "thoosa::z_norm",
                    "")
                .typed<
                    at::Tensor(
                        const at::Tensor&,
                        at::IntArrayRef,
                        double)>();

        const at::IntArrayRef dims_ref(
            dims,
            dims_len);

        at::Tensor result =
            op.call(
                *x,
                dims_ref,
                eps);

        // Ownership will be transferred to tch::Tensor.
        return new at::Tensor(std::move(result));
    }
    catch (const std::exception& e) {
        last_error = e.what();
        return nullptr;
    }
    catch (...) {
        last_error = "unknown C++ exception";
        return nullptr;
    }
}


extern "C"
const char* thoosa_last_error() noexcept
{
    return last_error.c_str();
}