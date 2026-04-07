use std::os::raw::c_int;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ThoDeviceStats {
    pub allocated: i64,
    pub active: i64,
    pub reserved: i64,
    pub requested: i64,
    pub non_releasable: i64,
}

unsafe extern "C" {
    fn tho_get_device_stats(device: c_int) -> ThoDeviceStats;
}

pub fn get_device_stats(device: i32) -> ThoDeviceStats {
    unsafe { tho_get_device_stats(device as c_int) }
}

impl ThoDeviceStats {
    pub fn reusable_cached(self) -> i64 {
        self.reserved.saturating_sub(self.active)
    }

    pub fn pending_free(self) -> i64 {
        self.active.saturating_sub(self.allocated)
    }

    pub fn rounding_overhead(self) -> i64 {
        self.allocated.saturating_sub(self.requested)
    }
}

#[cfg(test)]
mod tests {
    use crate::device_stats::get_device_stats;

    #[test]
    fn test_get_device_stats(){
        let s = get_device_stats(0);

        println!("{s:#?}");
        println!("reusable_cached = {}", s.reusable_cached());
        println!("pending_free    = {}", s.pending_free());
        println!("rounding_overhead = {}", s.rounding_overhead());
    }
}