use std::os::raw::c_int;
use thoosa_shared::{device_stats::ThoDeviceStats, ThoError};

use crate::cuda_result::{check_cuda_status, CuResult};

unsafe extern "C" {
    fn tho_get_device_stats(device: c_int, stats_out: *mut ThoDeviceStats) -> CuResult;
}

pub fn get_device_stats(device: i32) -> Result<ThoDeviceStats, ThoError> {
    let mut stats = ThoDeviceStats::default();
    let status = unsafe { tho_get_device_stats(device as c_int, &mut stats) };
    check_cuda_status(status, "tho_get_device_stats")?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use crate::device_stats::get_device_stats;
    use log::debug;
    use thoosa_shared::test_utils::init_test_logging;
    use torch_sys::cuda::atc_cuda_device_count;

    #[ctor::ctor]
    fn init() {
        init_test_logging();
    }

    #[test]
    fn test_get_device_stats(){
        let device_count = unsafe { atc_cuda_device_count() };
        if device_count < 1 {
            debug!("Skipping test_get_device_stats: no CUDA devices visible to torch");
            return;
        }

        let s = get_device_stats(0).unwrap_or_else(|err| panic!("test_get_device_stats failed: {}", err.msg));

        debug!("{s:#?}");
        debug!("reusable_cached = {}", s.reusable_cached());
        debug!("pending_free    = {}", s.pending_free());
        debug!("rounding_overhead = {}", s.rounding_overhead());
    }
}
