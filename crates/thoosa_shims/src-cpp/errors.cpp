#include "errors.h"

thread_local std::string tho_last_error;

void tho_clear_last_error() {
  tho_last_error.clear();
}

void tho_set_last_error_msg(const char* where, const char* msg) {
  tho_last_error.clear();
  tho_last_error += where ? where : "<unknown>";
  tho_last_error += ": ";
  tho_last_error += msg ? msg : "<null>";
}

extern "C" const char* tho_get_last_error_message() {
  return tho_last_error.c_str();
}
