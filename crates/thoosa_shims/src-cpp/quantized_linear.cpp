#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/quantized/PackedParams.h>
#include <c10/core/InferenceMode.h>

#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>

using C_tensor = void;

namespace {


thread_local std::string g_last_error;

struct ThoQuantizedLinear {
    c10::intrusive_ptr<LinearPackedParamsBase> packed;
    int64_t in_features = 0;
    int64_t out_features = 0;
    double weight_scale = 1.0;
};

const at::Tensor& tensor_ref(const C_tensor* t) {
    return *reinterpret_cast<const at::Tensor*>(t);
}

C_tensor* tensor_to_c(at::Tensor t) {
    return reinterpret_cast<C_tensor*>(new at::Tensor(std::move(t)));
}

void clear_error() {
    g_last_error.clear();
}

void set_error(const char* op, const std::exception& e) {
    try {
        g_last_error = std::string(op) + " failed: " + e.what();
    } catch (...) {
        // Avoid throwing through FFI.
    }
}

void set_unknown_error(const char* op) {
    try {
        g_last_error = std::string(op) + " failed: unknown C++ exception";
    } catch (...) {
        // Avoid throwing through FFI.
    }
}

double symmetric_i8_scale(const at::Tensor& w) {
    const double max_abs = w.abs().max().item<double>();

    if (!std::isfinite(max_abs) || max_abs == 0.0) {
        return 1.0;
    }

    return max_abs / 127.0;
}

void check_cpu_float_2d_weight(const at::Tensor& w) {
    TORCH_CHECK(w.defined(), "weight is undefined");
    TORCH_CHECK(w.device().is_cpu(), "weight must be on CPU, got ", w.device());
    TORCH_CHECK(w.scalar_type() == at::kFloat, "weight must be Float, got ", w.scalar_type());
    TORCH_CHECK(w.dim() == 2, "weight must have shape [out_features, in_features], got dim=", w.dim());
}

std::optional<at::Tensor> checked_bias_or_none(const C_tensor* bias_ptr, int64_t out_features) {
    if (bias_ptr == nullptr) {
        return std::nullopt;
    }

    const at::Tensor& b_in = tensor_ref(bias_ptr);

    TORCH_CHECK(b_in.defined(), "bias is undefined");
    TORCH_CHECK(b_in.device().is_cpu(), "bias must be on CPU, got ", b_in.device());
    TORCH_CHECK(b_in.scalar_type() == at::kFloat, "bias must be Float, got ", b_in.scalar_type());
    TORCH_CHECK(b_in.dim() == 1, "bias must be 1D, got dim=", b_in.dim());
    TORCH_CHECK(
        b_in.size(0) == out_features,
        "bias size mismatch: expected ",
        out_features,
        ", got ",
        b_in.size(0)
    );

    return b_in.contiguous();
}

void check_input(const ThoQuantizedLinear* ql, const at::Tensor& x) {
    TORCH_CHECK(ql != nullptr, "QuantizedLinear pointer is null");
    TORCH_CHECK(x.defined(), "input is undefined");
    TORCH_CHECK(x.device().is_cpu(), "input must be on CPU, got ", x.device());
    TORCH_CHECK(x.scalar_type() == at::kFloat, "input must be Float, got ", x.scalar_type());
    TORCH_CHECK(x.dim() >= 2, "input must have at least 2 dims, got dim=", x.dim());
    TORCH_CHECK(
        x.size(x.dim() - 1) == ql->in_features,
        "input last dim mismatch: expected ",
        ql->in_features,
        ", got ",
        x.size(x.dim() - 1)
    );
}

} // namespace

extern "C" const char* tho_ql_last_error() noexcept {
    return g_last_error.c_str();
}

extern "C" void tho_ql_clear_error() noexcept {
    clear_error();
}

extern "C" ThoQuantizedLinear* tho_ql_prepack_dynamic_i8(
    const C_tensor* weight_f32,
    const C_tensor* bias_f32_or_null,
    double weight_scale_or_negative
) noexcept {
    constexpr const char* OP = "tho_ql_prepack_dynamic_i8";

    try {
        clear_error();

        const at::Tensor& w_in = tensor_ref(weight_f32);
        check_cpu_float_2d_weight(w_in);

        const int64_t out_features = w_in.size(0);
        const int64_t in_features = w_in.size(1);

        auto bias = checked_bias_or_none(bias_f32_or_null, out_features);

        at::Tensor w = w_in.contiguous();

        const double weight_scale =
            weight_scale_or_negative > 0.0
                ? weight_scale_or_negative
                : symmetric_i8_scale(w);

        TORCH_CHECK(
            std::isfinite(weight_scale) && weight_scale > 0.0,
            "invalid weight scale: ",
            weight_scale
        );

        at::Tensor qw = at::quantize_per_tensor(
            w,
            weight_scale,
            /*zero_point=*/0,
            c10::ScalarType::QInt8
        );

        static auto prepack_op = c10::Dispatcher::singleton()
            .findSchemaOrThrow("quantized::linear_prepack", "")
            .typed<c10::intrusive_ptr<LinearPackedParamsBase>(
                at::Tensor,
                std::optional<at::Tensor>
            )>();

        auto packed = prepack_op.call(qw, bias);

        return new ThoQuantizedLinear{
            std::move(packed),
            in_features,
            out_features,
            weight_scale,
        };
    } catch (const std::exception& e) {
        set_error(OP, e);
        return nullptr;
    } catch (...) {
        set_unknown_error(OP);
        return nullptr;
    }
}

extern "C" C_tensor* tho_ql_forward_dynamic_i8(
    const ThoQuantizedLinear* ql,
    const C_tensor* input_f32,
    int reduce_range
) noexcept {
    constexpr const char* OP = "tho_ql_forward_dynamic_i8";

    try {
        clear_error();

        c10::InferenceMode inference_mode_guard(true);

        const at::Tensor& x = tensor_ref(input_f32);
        check_input(ql, x);

        static auto linear_dynamic_op = c10::Dispatcher::singleton()
            .findSchemaOrThrow("quantized::linear_dynamic", "")
            .typed<at::Tensor(
                at::Tensor,
                const c10::intrusive_ptr<LinearPackedParamsBase>&,
                bool
            )>();

        at::Tensor y = linear_dynamic_op.call(
            x,
            ql->packed,
            reduce_range != 0
        );

        return tensor_to_c(std::move(y));
    } catch (const std::exception& e) {
        set_error(OP, e);
        return nullptr;
    } catch (...) {
        set_unknown_error(OP);
        return nullptr;
    }
}

extern "C" C_tensor* tho_ql_forward_relu_dynamic_i8(
    const ThoQuantizedLinear* ql,
    const C_tensor* input_f32,
    int reduce_range
) noexcept {
    constexpr const char* OP = "tho_ql_forward_relu_dynamic_i8";

    try {
        clear_error();

        c10::InferenceMode inference_mode_guard(true);

        const at::Tensor& x = tensor_ref(input_f32);
        check_input(ql, x);

        static auto linear_relu_dynamic_op = c10::Dispatcher::singleton()
            .findSchemaOrThrow("quantized::linear_relu_dynamic", "")
            .typed<at::Tensor(
                at::Tensor,
                const c10::intrusive_ptr<LinearPackedParamsBase>&,
                bool
            )>();

        at::Tensor y = linear_relu_dynamic_op.call(
            x,
            ql->packed,
            reduce_range != 0
        );

        return tensor_to_c(std::move(y));
    } catch (const std::exception& e) {
        set_error(OP, e);
        return nullptr;
    } catch (...) {
        set_unknown_error(OP);
        return nullptr;
    }
}

extern "C" int64_t tho_ql_in_features(const ThoQuantizedLinear* ql) noexcept {
    return ql == nullptr ? -1 : ql->in_features;
}

extern "C" int64_t tho_ql_out_features(const ThoQuantizedLinear* ql) noexcept {
    return ql == nullptr ? -1 : ql->out_features;
}

extern "C" double tho_ql_weight_scale(const ThoQuantizedLinear* ql) noexcept {
    return ql == nullptr ? -1.0 : ql->weight_scale;
}

extern "C" void tho_ql_free(ThoQuantizedLinear* ql) noexcept {
    delete ql;
}