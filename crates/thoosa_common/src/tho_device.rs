use std::fmt::Debug;
use std::ops::Deref;
use tch::Device;
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
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

    pub fn gpu_ordinal(&self) -> usize {
        match &self.0 {
            Device::Cpu => {
                panic!("not a gpu device")
            },
            Device::Cuda(v) => *v,
            _ => {
                unimplemented!()
            }
        }
    }

}


