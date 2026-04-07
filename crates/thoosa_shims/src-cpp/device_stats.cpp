#include "errors.h"

#include <ATen/Context.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <cstdint>

struct ThoDeviceStats
{
    int64_t allocated;
    int64_t active;
    int64_t reserved;
    int64_t requested;
    int64_t non_releasable;
};

extern "C" {
int tho_get_device_stats(int device, ThoDeviceStats* stats_out)
{
    try {
        if (stats_out == nullptr) {
            tho_set_last_error_msg("tho_get_device_stats", "null stats_out");
            return 0;
        }

        at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
        const auto device_count = c10::cuda::device_count();
        if (device < 0 || device >= device_count) {
            tho_set_last_error_msg("tho_get_device_stats", "invalid device index");
            return 0;
        }

        auto* allocator = c10::cuda::CUDACachingAllocator::get();
        if (!allocator->initialized()) {
            allocator->init(device_count);
        }

        using c10::CachingAllocator::StatType;

        const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
        constexpr size_t A = static_cast<size_t>(StatType::AGGREGATE);

        const auto allocated      = stats.allocated_bytes[A].current;
        const auto active         = stats.active_bytes[A].current;
        const auto reserved       = stats.reserved_bytes[A].current;
        const auto requested      = stats.requested_bytes[A].current;
        const auto non_releasable = stats.inactive_split_bytes[A].current;

        *stats_out = ThoDeviceStats{
            allocated,
            active,
            reserved,
            requested,
            non_releasable
        };
        return 1;
    } catch (const c10::Error& e) {
        tho_set_last_error_msg("tho_get_device_stats", e.what_without_backtrace());
        return 0;
    } catch (const std::exception& e) {
        tho_set_last_error_ex("tho_get_device_stats", e);
        return 0;
    } catch (...) {
        tho_set_last_error_msg("tho_get_device_stats", "unknown C++ exception");
        return 0;
    }
}
}
