#pragma once

#include <cuda.h>

#include <exception>
#include <string>

extern thread_local std::string tho_last_error;
extern thread_local CUresult tho_last_error_code;

void tho_clear_last_error();
CUresult tho_set_last_error_code(const char* where, CUresult code, const char* msg);
CUresult tho_set_last_error_cuda_msg(const char* where, const char* msg, CUresult fallback);
CUresult tho_record_cuda_result(const char* where, CUresult code);

inline void tho_set_last_error_ex(const char* where, const std::exception& e) {
  tho_set_last_error_code(where, CUDA_ERROR_UNKNOWN, e.what());
}

extern "C" const char* tho_get_last_error_message();
extern "C" int tho_get_last_error_code();
