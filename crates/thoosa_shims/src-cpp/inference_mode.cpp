#include <c10/core/InferenceMode.h>

extern "C" void* tho_inference_mode_new(int enabled) noexcept {
    try {
        return new c10::InferenceMode(enabled != 0);
    } catch (...) {
        return nullptr;
    }
}

extern "C" void tho_inference_mode_delete(void* p) noexcept {
    delete static_cast<c10::InferenceMode*>(p);
}

extern "C" int tho_inference_mode_is_enabled() noexcept {
    return c10::InferenceMode::is_enabled() ? 1 : 0;
}