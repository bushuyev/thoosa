#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>

#include <linux/memfd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "logging.h"

extern "C" {

/*
 * Implemented in Rust.
 */
int rust_talc_init(
    void* base,
    std::size_t size
);

void* rust_talc_alloc(
    std::size_t size,
    std::size_t alignment
);

void rust_talc_dealloc(
    void* ptr,
    std::size_t size,
    std::size_t alignment
);

}


namespace {

constexpr std::size_t CPU_ALIGNMENT = 64;

struct AllocationContext {
    void* ptr;
    std::size_t size;
};


class SharedArenaAllocator;


SharedArenaAllocator* global_allocator = nullptr;


void delete_allocation(void* ctx) {
    auto* allocation =
        static_cast<AllocationContext*>(ctx);

    if (allocation == nullptr) {
        return;
    }

    rust_talc_dealloc(
        allocation->ptr,
        allocation->size,
        CPU_ALIGNMENT
    );

    delete allocation;
}


class SharedArenaAllocator final
    : public c10::Allocator
{
public:
    bool init( std::size_t capacity ) {
        tho_log(CppLogLevel::Info, "SharedArenaAllocator::init");

        std::lock_guard lock(mutex_);

        if (base_ != nullptr) {
            errno = EALREADY;
            return false;
        }

        const int fd =
            static_cast<int>(
                ::syscall(
                    SYS_memfd_create,
                    "libtorch-shared-arena",
                    MFD_CLOEXEC
                )
            );

        if (fd < 0) {
            return false;
        }

        if (
            ::ftruncate(
                fd,
                static_cast<off_t>(capacity)
            ) != 0
        ) {
            const int saved_errno = errno;

            ::close(fd);

            errno = saved_errno;

            return false;
        }

        void* base =
            ::mmap(
                nullptr,
                capacity,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd,
                0
            );

        if (base == MAP_FAILED) {
            const int saved_errno = errno;

            ::close(fd);

            errno = saved_errno;

            return false;
        }

        /*
         * Hand the whole mmap to Talc.
         *
         * Talc will put its own allocator metadata inside
         * this region and manage all suballocations.
         */
        if (
            rust_talc_init(
                base,
                capacity
            ) != 0
        ) {
            ::munmap(base, capacity);
            ::close(fd);

            errno = ENOMEM;

            return false;
        }

        fd_ = fd;
        base_ = base;
        capacity_ = capacity;

        return true;
    }


    c10::DataPtr allocate(
        std::size_t nbytes
    ) override {
        if (nbytes == 0) {
            return c10::DataPtr(
                nullptr,
                c10::Device(
                    c10::DeviceType::CPU
                )
            );
        }

        /*
         * Actual allocation happens in Rust/Talc.
         */
        void* ptr =
            rust_talc_alloc(
                nbytes,
                CPU_ALIGNMENT
            );

        TORCH_CHECK(
            ptr != nullptr,
            "SharedArenaAllocator/Talc exhausted: ",
            nbytes,
            " bytes requested"
        );

        /*
         * DataPtr's deleter needs to know the Layout
         * originally used with Talc.
         */
        auto* ctx =
            new AllocationContext{
                .ptr = ptr,
                .size = nbytes,
            };

        return c10::DataPtr(
            ptr,
            ctx,
            &delete_allocation,
            c10::Device(
                c10::DeviceType::CPU
            )
        );
    }


    void copy_data(
        void* dst,
        const void* src,
        std::size_t count
    ) const override {
        std::memcpy(
            dst,
            src,
            count
        );
    }


    int dup_fd() const {
        std::lock_guard lock(mutex_);

        if (fd_ < 0) {
            errno = EBADF;
            return -1;
        }

        return ::dup(fd_);
    }


    std::int64_t offset_of(
        const void* ptr
    ) const {
        if (
            base_ == nullptr
            || ptr == nullptr
        ) {
            return -1;
        }

        const auto base =
            reinterpret_cast<std::uintptr_t>(
                base_
            );

        const auto address =
            reinterpret_cast<std::uintptr_t>(
                ptr
            );

        if (
            address < base
            || address >=
                base + capacity_
        ) {
            return -1;
        }

        return static_cast<std::int64_t>(
            address - base
        );
    }


    std::size_t capacity() const {
        return capacity_;
    }


private:
    mutable std::mutex mutex_;

    int fd_ = -1;

    void* base_ = nullptr;

    std::size_t capacity_ = 0;
};


SharedArenaAllocator allocator;

} // namespace


extern "C"
int shared_arena_init(
    std::size_t capacity
) {
    if (!allocator.init(capacity)) {
        return -1;
    }

    /*
     * From now on ordinary LibTorch CPU allocations use:
     *
     * c10
     *  -> SharedArenaAllocator
     *  -> rust_talc_alloc()
     *  -> Talc
     *  -> mmap
     */
    c10::SetCPUAllocator(
        &allocator,
        255
    );

    return 0;
}


extern "C" int shared_arena_dup_fd() {
    return allocator.dup_fd();
}


extern "C" std::int64_t shared_arena_offset_of(
    const void* ptr
) {
    return allocator.offset_of(ptr);
}


extern "C" std::size_t shared_arena_capacity() {
    return allocator.capacity();
}