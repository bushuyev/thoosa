
/*
 * Talc itself requires &mut self for allocation/deallocation.
 *
 * c10::Allocator can be invoked from arbitrary LibTorch threads,
 * therefore serialize Talc access with a Mutex.
 */
use std::alloc::Layout;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::OnceLock;
use parking_lot::Mutex;
use talc::base::Talc;
use talc::DefaultBinning;
use talc::source::Manual;

pub type ArenaTalc = Talc<Manual, DefaultBinning>;


static TALC: OnceLock<Mutex<ArenaTalc>> = OnceLock::new();

/*
 * Called exactly once by SharedArenaAllocator::init().
 *
 * base,size describe the mmap-backed memfd.
 */
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_talc_init(
    base: *mut c_void,
    size: usize,
) -> i32 {
    if base.is_null() || size == 0 {
        return -1;
    }

    let mut talc:
        Talc<Manual, DefaultBinning> =
        Talc::new(Manual);

    /*
     * SAFETY:
     *
     * SharedArenaAllocator owns this mmap and keeps it
     * mapped for the lifetime of the allocator.
     *
     * The parent must not directly write into unallocated
     * parts of the arena.
     */
    let heap_end = unsafe {
        talc.claim(
            base.cast::<u8>(),
            size,
        )
    };

    if heap_end.is_none() {
        return -2;
    }

    if TALC
        .set(Mutex::new(talc))
        .is_err()
    {
        return -3;
    }

    0
}


#[unsafe(no_mangle)]
pub extern "C" fn rust_talc_alloc(
    size: usize,
    alignment: usize,
) -> *mut c_void {
    if size == 0 {
        return null_mut();
    }

    let Ok(layout) =
        Layout::from_size_align(
            size,
            alignment,
        )
    else {
        return null_mut();
    };

    let Some(talc) =
        TALC.get()
    else {
        return null_mut();
    };

    let mut talc = talc.lock();

    /*
     * Talc::allocate returns None on arena exhaustion.
     */
    let allocation = unsafe {
        talc.allocate(layout)
    };

    allocation
        .map(|ptr| {
            ptr.as_ptr().cast::<c_void>()
        })
        .unwrap_or(null_mut())
}


#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_talc_dealloc(
    ptr: *mut c_void,
    size: usize,
    alignment: usize,
) {
    if ptr.is_null() || size == 0 {
        return;
    }

    let Ok(layout) =
        Layout::from_size_align(
            size,
            alignment,
        )
    else {
        return;
    };

    let Some(talc) =
        TALC.get()
    else {
        return;
    };


    let mut talc = talc.lock();

    unsafe {
        talc.deallocate(
            ptr.cast::<u8>(),
            layout,
        );
    }
}
