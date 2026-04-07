#include <cstdint>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/core/CachingDeviceAllocator.h>

struct ThoDeviceStats
{
    int64_t allocated;
    int64_t active;
    int64_t reserved;
    int64_t requested;
    int64_t non_releasable;
};

extern "C" {
ThoDeviceStats tho_get_device_stats(int device)
{
    using c10::CachingAllocator::StatType;

    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    constexpr size_t A = static_cast<size_t>(StatType::AGGREGATE);

    const auto allocated      = stats.allocated_bytes[A].current;
    const auto active         = stats.active_bytes[A].current;
    const auto reserved       = stats.reserved_bytes[A].current;
    const auto requested      = stats.requested_bytes[A].current;
    const auto non_releasable = stats.inactive_split_bytes[A].current;

    return ThoDeviceStats{
        allocated,
        active,
        reserved,
        requested,
        non_releasable
    };
}
}
