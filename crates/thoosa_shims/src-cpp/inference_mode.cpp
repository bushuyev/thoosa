#include "errors.h"
#include <c10/core/InferenceMode.h>

extern "C" void* tho_inference_mode_new(int enabled) noexcept {
    try {
        auto* guard = new c10::InferenceMode(enabled != 0);
        tho_clear_last_error();
        return guard;
    } catch (const std::exception& e) {
        tho_set_last_error_ex("tho_inference_mode_new", e);
        return nullptr;
    } catch (...) {
        tho_set_last_error_code("tho_inference_mode_new", CUDA_ERROR_UNKNOWN, "unknown C++ exception");
        return nullptr;
    }
}

extern "C" void tho_inference_mode_delete(void* p) noexcept {
    delete static_cast<c10::InferenceMode*>(p);
}

extern "C" int tho_inference_mode_is_enabled() noexcept {
    return c10::InferenceMode::is_enabled() ? 1 : 0;
}