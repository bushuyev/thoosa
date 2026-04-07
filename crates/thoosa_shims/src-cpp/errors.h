#pragma once

#include <exception>
#include <string>

extern thread_local std::string tho_last_error;

void tho_clear_last_error();
void tho_set_last_error_msg(const char* where, const char* msg);

inline void tho_set_last_error_ex(const char* where, const std::exception& e) {
  tho_set_last_error_msg(where, e.what());
}

extern "C" const char* tho_get_last_error_message();
