#include "errors.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <sstream>

thread_local std::string tho_last_error;
thread_local CUresult tho_last_error_code = CUDA_SUCCESS;

namespace {

std::string lower_copy(const char* msg) {
  std::string out = msg ? msg : "";
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

bool contains_any(const std::string& haystack, std::initializer_list<const char*> needles) {
  for (const char* needle : needles) {
    if (haystack.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

CUresult cuda_error_code_from_msg(const char* msg, CUresult fallback) {
  const std::string lower = lower_copy(msg);
  if (contains_any(lower, {"cuda out of memory", "outofmemoryerror", "out of memory", "cuda_error_out_of_memory", "cuda_error_memory_allocation"})) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  if (contains_any(lower, {"device-side assert", "cudaerrorassert", "cuda_error_assert", "error 710", "code=710", "code 710"})) {
    return static_cast<CUresult>(710);
  }
  if (contains_any(lower, {"illegal memory access", "cudaerrorillegaladdress", "cuda_error_illegal_address", "error 700", "code=700", "code 700"})) {
    return CUDA_ERROR_ILLEGAL_ADDRESS;
  }
  if (contains_any(lower, {"unspecified launch failure", "cudaerrorlaunchfailure", "cuda_error_launch_failed", "error 719", "code=719", "code 719"})) {
    return CUDA_ERROR_LAUNCH_FAILED;
  }
  if (contains_any(lower, {"launch timed out", "launch timeout", "cudaerrorlaunchtimeout", "cuda_error_launch_timeout", "error 702", "code=702", "code 702"})) {
    return CUDA_ERROR_LAUNCH_TIMEOUT;
  }
  if (contains_any(lower, {"context is destroyed", "context was destroyed", "cudaerrorcontextisdestroyed", "cuda_error_context_is_destroyed", "error 709", "code=709", "code 709"})) {
    return static_cast<CUresult>(709);
  }
  if (contains_any(lower, {"hardware stack error", "cudaerrorhardwarestackerror", "cuda_error_hardware_stack_error", "error 714", "code=714", "code 714"})) {
    return CUDA_ERROR_HARDWARE_STACK_ERROR;
  }
  if (contains_any(lower, {"illegal instruction", "cudaerrorillegalinstruction", "cuda_error_illegal_instruction", "error 715", "code=715", "code 715"})) {
    return CUDA_ERROR_ILLEGAL_INSTRUCTION;
  }
  if (contains_any(lower, {"misaligned address", "cudaerrormisalignedaddress", "cuda_error_misaligned_address", "error 716", "code=716", "code 716"})) {
    return CUDA_ERROR_MISALIGNED_ADDRESS;
  }
  if (contains_any(lower, {"invalid address space", "cudaerrorinvalidaddressspace", "cuda_error_invalid_address_space", "error 717", "code=717", "code 717"})) {
    return CUDA_ERROR_INVALID_ADDRESS_SPACE;
  }
  if (contains_any(lower, {"invalid program counter", "invalid pc", "cudaerrorinvalidpc", "cuda_error_invalid_pc", "error 718", "code=718", "code 718"})) {
    return CUDA_ERROR_INVALID_PC;
  }
  if (contains_any(lower, {"uncorrectable ecc", "ecc uncorrectable", "cudaerroreccuncorrectable", "cuda_error_ecc_uncorrectable", "error 214", "code=214", "code 214"})) {
    return CUDA_ERROR_ECC_UNCORRECTABLE;
  }
  if (contains_any(lower, {"nvlink uncorrectable", "cudaerrornvlinkuncorrectable", "cuda_error_nvlink_uncorrectable", "error 220", "code=220", "code 220"})) {
    return static_cast<CUresult>(220);
  }
  if (contains_any(lower, {"contained device exception", "cudaerrorcontained", "error 226", "code=226", "code 226"})) {
    return static_cast<CUresult>(226);
  }
  if (contains_any(lower, {"tensor memory leak", "cudaerrortensormemoryleak", "error 721", "code=721", "code 721"})) {
    return static_cast<CUresult>(721);
  }
  if (contains_any(lower, {"mps client terminated", "cudaerrormpsclientterminated", "cuda_error_mps_client_terminated", "error 810", "code=810", "code 810"})) {
    return static_cast<CUresult>(810);
  }
  if (contains_any(lower, {"external device", "cudaerrorexternaldevice", "error 911", "code=911", "code 911"})) {
    return static_cast<CUresult>(911);
  }
  if (contains_any(lower, {"invalid device ordinal", "invalid device", "cudaerrorinvaliddevice", "cuda_error_invalid_device"})) {
    return CUDA_ERROR_INVALID_DEVICE;
  }
  if (contains_any(lower, {"invalid argument", "invalid value", "cudaerrorinvalidvalue", "cuda_error_invalid_value"})) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (contains_any(lower, {"invalid resource handle", "cudaerrorinvalidresourcehandle", "cuda_error_invalid_handle"})) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  if (contains_any(lower, {"not ready", "cudaerrornotready", "cuda_error_not_ready"})) {
    return CUDA_ERROR_NOT_READY;
  }
  if (contains_any(lower, {"illegal state", "cudaerrorillegalstate"})) {
    return static_cast<CUresult>(401);
  }
  if (contains_any(lower, {"invalid configuration argument", "cudaerrorinvalidconfiguration"})) {
    return static_cast<CUresult>(9);
  }
  if (contains_any(lower, {"launch out of resources", "cudaerrorlaunchoutofresources", "cuda_error_launch_out_of_resources"})) {
    return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
  }
  if (contains_any(lower, {"cuda error: unknown error", "cudaerrorunknown", "cuda_error_unknown", "error 999", "code=999", "code 999"})) {
    const cudaError_t runtime_error = cudaPeekAtLastError();
    if (runtime_error != cudaSuccess) {
      return static_cast<CUresult>(runtime_error);
    }
    return CUDA_ERROR_UNKNOWN;
  }
  if (contains_any(lower, {"cuda", "cublas", "cudnn", "cusolver", "cusparse", "cufft"})) {
    const cudaError_t runtime_error = cudaPeekAtLastError();
    if (runtime_error != cudaSuccess) {
      return static_cast<CUresult>(runtime_error);
    }
  }
  return fallback;
}

std::string cuda_driver_error_text(CUresult code) {
  const char* name = nullptr;
  const char* text = nullptr;
  const CUresult name_result = cuGetErrorName(code, &name);
  const CUresult text_result = cuGetErrorString(code, &text);

  std::ostringstream msg;
  if (name_result == CUDA_SUCCESS && name) {
    msg << name;
  } else {
    msg << "unrecognized CUDA error";
  }
  if (text_result == CUDA_SUCCESS && text) {
    msg << ": " << text;
  }
  return msg.str();
}

} // namespace

void tho_clear_last_error() {
  tho_last_error.clear();
  tho_last_error_code = CUDA_SUCCESS;
}

CUresult tho_set_last_error_code(const char* where, CUresult code, const char* msg) {
  tho_last_error.clear();
  tho_last_error += where ? where : "<unknown>";
  tho_last_error += ": CUDA error code=";
  tho_last_error += std::to_string(static_cast<int>(code));
  tho_last_error += ": ";
  tho_last_error += msg ? msg : "<null>";
  tho_last_error_code = code;
  return code;
}

CUresult tho_set_last_error_cuda_msg(const char* where, const char* msg, CUresult fallback) {
  return tho_set_last_error_code(where, cuda_error_code_from_msg(msg, fallback), msg);
}

CUresult tho_record_cuda_result(const char* where, CUresult code) {
  if (code == CUDA_SUCCESS) {
    tho_clear_last_error();
    return CUDA_SUCCESS;
  }
  const std::string msg = cuda_driver_error_text(code);
  return tho_set_last_error_code(where, code, msg.c_str());
}

extern "C" const char* tho_get_last_error_message() {
  return tho_last_error.c_str();
}

extern "C" int tho_get_last_error_code() {
  return static_cast<int>(tho_last_error_code);
}
