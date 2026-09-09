use std::ffi::c_void;

unsafe extern "C" {
    pub fn shared_arena_init(
        capacity: usize
    ) -> i32;

    pub fn shared_arena_dup_fd() -> i32;

    pub fn shared_arena_offset_of(
        ptr: *const c_void
    ) -> i64;

    pub fn shared_arena_capacity() -> usize;
}
