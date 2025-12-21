use std::fmt::Debug;
use std::ops::Deref;
use tch::Device;
use thoosa_shared::applicables_runner::{
    CachableId, ToOrdinal,
};
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};

#[derive(Debug, Copy, Clone)]
pub struct ThoDevice(Device);

impl Deref for ThoDevice {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Device> for ThoDevice {
    fn from(value: Device) -> Self {
        ThoDevice(value)
    }
}

impl From<ThoDevice> for Device {
    fn from(value: ThoDevice) -> Self {
        value.0
    }
}

impl From<usize> for ThoDevice {
    fn from(value: usize) -> Self {
        match value {
            0 => ThoDevice(Device::Cpu),
            d => ThoDevice(Device::Cuda(d - 1)),
        }
    }
}

pub fn with_cuda_stream<F, R>(tho_device_idx:usize, mut f:F) -> (R, Option<ThoCudaStream>)
where
    F:FnMut() -> R
{

    if tho_device_idx > 0 {
        let motifs_stream = ThoCudaStream::new((tho_device_idx - 1) as i32); //TODO
        let _guard = ThoStreamGuard::set(&motifs_stream);

        (f(), Some(motifs_stream))
    } else {
        (f(), None)
    }
}


impl ThoDevice {
    pub fn ordinal(&self) -> usize {
        match &self.0 {
            Device::Cpu => 0,
            Device::Cuda(v) => v + 1,
            _ => {
                unimplemented!()
            }
        }
    }
}

#[derive(Debug)]
pub struct ThoRunCtx {
    cache_area: u64,

    loaded: Vec<CachableId>,
    find_motifs: Vec<CachableId>,
    motif_nets: Vec<CachableId>,
}

impl ThoRunCtx {
    pub fn new(cache_area: u64) -> Self {
        Self {
            cache_area,
            loaded: Vec::new(),
            find_motifs: Vec::new(),
            motif_nets: Vec::new(),
        }
    }
}


impl ToOrdinal for ThoDevice {
    fn ordinal(&self) -> usize {
        match self.0 {
            Device::Cpu => 0,
            Device::Cuda(d) => d + 1,
            _ => {
                unimplemented!()
            }
        }
    }
}



