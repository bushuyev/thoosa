#include "logging.h"
#include <atomic>
#include <iostream>

static std::atomic<CppLogCallback> GLOBAL_LOG_CALLBACK = nullptr;

extern "C" void tho_set_log_callback(CppLogCallback cb) {
    GLOBAL_LOG_CALLBACK.store(cb, std::memory_order_release);
}

void tho_log(CppLogLevel level, const std::string& msg) {
    auto cb = GLOBAL_LOG_CALLBACK.load(std::memory_order_acquire);

    if (cb) {
        cb(level, msg.data(), msg.size());
    } else {
        // Fallback before Rust installs callback.
        std::cerr << "[cpp_shim] " << msg << std::endl;
    }
}