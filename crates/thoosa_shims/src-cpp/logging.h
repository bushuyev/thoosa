#pragma once
// logging.hpp
#pragma once

#include <cstddef>
#include <string>
#include <atomic>

extern "C" {

enum class CppLogLevel : int {
    Trace = 0,
    Debug = 1,
    Info  = 2,
    Warn  = 3,
    Error = 4,
};

using CppLogCallback = void (*)(CppLogLevel level, const char* msg, std::size_t len);

void tho_set_log_callback(CppLogCallback cb);

}

void tho_log(CppLogLevel level, const std::string& msg);